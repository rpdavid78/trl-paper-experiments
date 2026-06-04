#!/usr/bin/env python
# ============================================================================
# A/B  GGN  vs  HESSIANA  no TRL   (DIAGNÓSTICO — não usar p/ tabelas do paper)
# ----------------------------------------------------------------------------
# Pergunta: o paper fala "Fisher/GGN" (espaço de funções), mas o código seleciona
# N pelos top-k autovetores da HESSIANA da CE (espaço de parâmetros, indefinida).
# Aqui implementamos o GGN-vector product EXATO (softmax: H_out = diag(p) - p pᵀ)
# e isolamos os DOIS efeitos num 2×2:
#
#                         SELEÇÃO de N        VARIÂNCIA (evals p/ inv_sqrt_prec)
#   base        (paper)   Hessiana            Hessiana
#   sel_ggn               GGN                 Hessiana(proj. em N_ggn)
#   var_ggn               Hessiana            GGN(proj. em N_hess)
#   full_ggn              GGN                 GGN
#
# Para isolar o operador, a TANGENTE fica fixa = random-complement (baseline),
# e a estrutura do loop é idêntica ao build original. evals seguem fixados no MAP.
# Não toca no arquivo de origem (monkey-patch). Não sobrescreve o spine do paper.
#
# Uso (servidor, env trl-iclr, do dir que contém o pacote):
#   python trl_ggn_ab.py --arch resnet18 --seed 0 \
#       --ckpt-dir /mnt/hd2/rpdavid/results_article_trl/checkpoints_resnet18_seed0 \
#       --experiments base sel_ggn
#
# Recomendo: base + sel_ggn primeiro (efeito da SELEÇÃO é o que pode mexer).
# Se diferirem, rodar var_ggn e full_ggn. Se base≈sel_ggn, decisão fechada.
# ============================================================================
import argparse
import importlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse.linalg as sla

MODULE_NAME = "cifar100_all_methods_iclr"
M = importlib.import_module(MODULE_NAME)
DEVICE = M.DEVICE


# ----------------------------------------------------------------------------
# GGN-vector product EXATO para softmax cross-entropy.
#   GGN = (1/m) Σ_batch (1/B) Σ_i J_iᵀ H_out,i J_i ,   H_out = diag(p) - p pᵀ
# Implementado com três passes de autograd (sem materializar J):
#   1) w(u) = Jᵀ u           (VJP, create_graph p/ diferenciar em u)
#   2) Jv  = ∂(w·v)/∂u        (extrai o JVP via double-backward)
#   3) GGNv = Jᵀ (H_out Jv)   (VJP)
# Escala (mean sobre batch e sobre buffer) casada com o HVP-ablation existente.
# BN em train(), igual ao get_hvp_function_ablation, p/ isolar só o operador.
# ----------------------------------------------------------------------------
def get_ggn_function(model, loader, device, num_batches):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    data_cache = []
    it = iter(loader)
    for _ in range(num_batches):
        try:
            data_cache.append(next(it))
        except StopIteration:
            break
    if len(data_cache) == 0:
        raise RuntimeError("GGN: data_cache vazio.")

    def ggnvp(v_numpy):
        v = torch.from_numpy(v_numpy).float().to(device)
        accum = None
        for x, _y in data_cache:
            x = x.to(device)
            model.zero_grad(set_to_none=True)

            logits = model(x)                       # (B, C), com grafo
            if not torch.isfinite(logits).all():
                raise RuntimeError("GGN: logits NaN/Inf")
            B = logits.shape[0]
            p = torch.softmax(logits, dim=1).detach()

            u = torch.zeros_like(logits, requires_grad=True)
            # w = Jᵀ u  (linear em u); create_graph p/ poder diferenciar em u
            w = torch.autograd.grad(logits, params, grad_outputs=u,
                                    create_graph=True, allow_unused=True)
            w_vec = M.flatten_grads(w, params)
            # Jv = ∂(w·v)/∂u
            Jv = torch.autograd.grad(w_vec @ v, u, retain_graph=True)[0]   # (B, C)

            # H_out @ Jv  (Hessiana do softmax-CE na saída), mean sobre batch
            pJv = (p * Jv).sum(dim=1, keepdim=True)
            R = (p * Jv - p * pJv) / B

            # GGNv = Jᵀ R
            JtR = torch.autograd.grad(logits, params, grad_outputs=R,
                                      retain_graph=False, allow_unused=True)
            gvec = M.flatten_grads(JtR, params).detach()

            accum = gvec if accum is None else accum + gvec
            del logits, p, u, w, w_vec, Jv, pJv, R, JtR, gvec

        accum = accum / len(data_cache)
        return accum.cpu().numpy()

    return ggnvp


def projected_diag(matvec_np, N_torch):
    """diag(Nᵀ Op N) via k matvecs (Op não precisa ser eigenbase de N)."""
    Ncpu = N_torch.detach().cpu().numpy()
    k = Ncpu.shape[1]
    out = torch.empty(k, device=DEVICE)
    for j in range(k):
        col = Ncpu[:, j].astype(np.float64)
        opcol = np.asarray(matvec_np(col)).ravel()
        out[j] = float(np.dot(col, opcol))
    return out


# ----------------------------------------------------------------------------
# build() com operador de SELEÇÃO e de VARIÂNCIA configuráveis.
# base (hessian/hessian) reproduz EXATAMENTE o build original.
# Tangente fixa = random-complement (isola o operador, não a tangente).
# ----------------------------------------------------------------------------
def build_ggn(self):
    sel_op = getattr(self, "sel_op", "hessian")
    var_op = getattr(self, "var_op", "hessian")

    self._reset()
    params = [p for p in self.model.parameters() if p.requires_grad]
    curr_theta = M.parameters_to_vector(params).detach()
    num_params = curr_theta.numel()

    print(f"    [TRL-GGN] sel={sel_op} var={var_op} | P={num_params}, T={self.T}, K={self.k}")

    hvp_fn = M.get_hvp_function_ablation(self.model, self.loader, DEVICE, num_batches=self.hvp_batches)
    ggn_fn = None
    if sel_op == "ggn" or var_op == "ggn":
        ggn_fn = get_ggn_function(self.model, self.loader, DEVICE, num_batches=self.hvp_batches)

    sel_matvec = ggn_fn if sel_op == "ggn" else hvp_fn
    op = sla.LinearOperator((num_params, num_params), matvec=sel_matvec)
    vals, vecs = sla.eigsh(op, k=self.k + 1, which="LA")
    idx = np.argsort(vals)[::-1][:self.k]
    N = torch.from_numpy(vecs[:, idx].copy()).float().to(DEVICE)
    sel_vals = torch.maximum(
        torch.from_numpy(vals[idx].copy()).float().to(DEVICE),
        torch.tensor(0.0, device=DEVICE),
    )

    # eigenvalues p/ a variância transversa
    if var_op == sel_op:
        evals = sel_vals                       # N é eigenbase do op de variância
    else:
        var_matvec = ggn_fn if var_op == "ggn" else hvp_fn
        evals = torch.clamp(projected_diag(var_matvec, N), min=0.0)
    print(f"    [TRL-GGN] evals[min/max]={float(evals.min()):.3e}/{float(evals.max()):.3e}")

    # tangente inicial: random-complement (fixa, p/ isolar o operador)
    vr = torch.randn(num_params, device=DEVICE)
    vr = vr - N @ (N.T @ vr)
    v = vr / (vr.norm() + 1e-9)

    data_iterator = iter(self.loader)
    for t in range(self.T):
        prior_proj = torch.sum((N ** 2) * self.prior.unsqueeze(1), dim=0)
        prec = torch.clamp(evals + prior_proj, min=1e-6)
        inv_sqrt_prec = torch.rsqrt(prec)

        if (t % self.store_every) == 0:
            self.spine.append({
                "theta": curr_theta.detach().cpu(),
                "N": N.detach().cpu(),
                "inv_sqrt_prec": inv_sqrt_prec.detach().cpu(),
            })

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        M.vector_to_parameters(curr_theta, self.model.parameters())

        try:
            xb, yb = next(data_iterator)
        except StopIteration:
            data_iterator = iter(self.loader)
            xb, yb = next(data_iterator)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        logits = self.model(xb)
        if not torch.isfinite(logits).all():
            raise RuntimeError("TRL build: logits NaN/Inf")
        loss = nn.CrossEntropyLoss()(logits, yb)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        g = M.flatten_grads(grads, params)

        g_perp = g - torch.dot(g, v) * v
        delta = self.ds * v - self.eta * g_perp

        dnorm = delta.norm()
        if dnorm > self.max_delta_norm:
            delta = delta * (self.max_delta_norm / (dnorm + 1e-12))

        theta_next = curr_theta + delta

        d = theta_next - curr_theta
        d_norm = d.norm()
        if d_norm > 1e-9:
            v_new = d / d_norm
            proj = v_new @ N
            N_ortho = N - torch.outer(v_new, proj)
            N, _ = torch.linalg.qr(N_ortho, mode="reduced")
            v = v_new

        curr_theta = theta_next
        del loss, grads, g, g_perp, delta
        if (t % 5) == 0:
            M.cleanup()


def make_cfg(arch, seed, ckpt_dir):
    cfg = M.CFG()
    cfg.seed = seed
    cfg.arch = arch
    cfg.ckpt_dir = ckpt_dir or f"./checkpoints_c100_{arch}_seed{seed}"
    cfg.map_ckpt = f"{arch}_cifar100_map.pth"
    cfg.mcdo_ckpt = f"{arch}_cifar100_mcdo.pth"
    cfg.ens_prefix = f"c100_{arch}_ens"
    cfg.swag_stats = f"c100_{arch}_swag_stats.pth"
    return cfg


EXPERIMENTS = {
    "base":     ("hessian", "hessian"),   # reproduz o paper
    "sel_ggn":  ("ggn",     "hessian"),   # isola: GGN seleciona N
    "var_ggn":  ("hessian", "ggn"),       # isola: GGN dá a variância
    "full_ggn": ("ggn",     "ggn"),       # GGN completa
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "wrn16_4", "vgg11_bn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--results", default="results/DIAG_ggn_ab.jsonl")
    ap.add_argument("--map-ckpt", default=None, help="nome do .pth do MAP se diferir do default")
    ap.add_argument("--experiments", nargs="+", default=["base", "sel_ggn"],
                    choices=list(EXPERIMENTS.keys()))
    args = ap.parse_args()

    M.PracticalTRLStage2.build = build_ggn

    # diagnostico: nao salvar spines gigantes (evita encher disco)
    _orig_torch_save = torch.save
    def _save_skip_diag(obj, f, *a, **k):
        fname = ""
        try:
            fname = str(f)
        except Exception:
            pass
        if "DIAG_" in fname:
            print("    [GGN-AB] skip torch.save (diagnostico): " + __import__("os").path.basename(fname))
            return
        return _orig_torch_save(obj, f, *a, **k)
    torch.save = _save_skip_diag

    cfg = make_cfg(args.arch, args.seed, args.ckpt_dir)
    if args.map_ckpt:
        cfg.map_ckpt = args.map_ckpt
    M.set_seed(cfg.seed)

    print(">>> [GGN-AB] dados + MAP + Laplace (uma vez)")
    tr_aug, bn_clean, val, ts, ood = M.get_data(cfg)
    targets_ts = M.get_targets(ts)
    model_map = M.load_or_train_map(tr_aug, cfg)
    la, base_val, *_ = M.laplace_fit_and_predict(
        model_map=model_map, bn_loader_clean=bn_clean, ts_loader=ts, ood_loader=ood, cfg=cfg
    )

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    print("\n" + "=" * 92)
    print("A/B GGN vs HESSIANA — DIAGNÓSTICO (não usar p/ tabelas do paper)")
    print("=" * 92)

    summary = []
    for exp in args.experiments:
        sel_op, var_op = EXPERIMENTS[exp]
        cfg.trl_spine = f"DIAG_ggn_{exp}_{cfg.arch}_seed{cfg.seed}.pth"
        M.PracticalTRLStage2.sel_op = sel_op
        M.PracticalTRLStage2.var_op = var_op

        print(f"\n>>> [GGN-AB] experimento: {exp} (sel={sel_op}, var={var_op})")
        M.set_seed(cfg.seed)
        t0 = time.perf_counter()
        p_trl, p_trl_ood, sweep, best_ts, best_nll, _tim = M.trl_stage2_run(
            model_map=model_map, base_val=base_val, bn_loader_clean=bn_clean,
            tr_loader_aug=tr_aug, val_loader=val, ts_loader=ts, ood_loader=ood, cfg=cfg
        )
        wall = time.perf_counter() - t0
        acc, nll, ece, brier = M.calc_metrics(p_trl, targets_ts, cfg.num_classes)
        row = {"experiment": exp, "sel_op": sel_op, "var_op": var_op,
               "acc": acc, "nll": nll, "ece": ece, "brier": brier,
               "best_tube_scale": best_ts, "val_nll": best_nll,
               "wall_sec": wall, "arch": cfg.arch, "seed": cfg.seed, "diagnostic": True}
        summary.append(row)
        with open(args.results, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"    -> acc={100*acc:.2f} nll={nll:.4f} ece={ece:.4f} brier={brier:.4f} "
              f"(beta*={best_ts}, {wall:.0f}s)")

    print("\n" + "=" * 92)
    print(f"{'experiment':12s} {'sel':>8s} {'var':>8s} {'acc':>7s} {'nll':>8s} {'ece':>8s} {'brier':>8s}")
    print("-" * 92)
    for r in summary:
        print(f"{r['experiment']:12s} {r['sel_op']:>8s} {r['var_op']:>8s} "
              f"{100*r['acc']:7.2f} {r['nll']:8.4f} {r['ece']:8.4f} {r['brier']:8.4f}")
    print("=" * 92)
    print("baseline do paper = 'base' (hessian/hessian). compare os deltas vs ele.")


if __name__ == "__main__":
    main()

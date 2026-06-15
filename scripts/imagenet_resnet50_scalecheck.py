#!/usr/bin/env python3
"""
TRL ImageNet scale-check — Etapa 2: pipeline com single-checkpoint ou full-spine pós-hoc.

Por seed:
  S0  gate MAP no val completo (esperado acc ~ 0.7613 +- 0.005; aborta se desviar)
  S1  carrega split de TRAIN persistido (imagenet_split_seed{seed}.pt; train_idx
      alimenta HVP/FixBN) + lambda_base do JSONL do marglik + cria/carrega split
      de VAL persistido (imagenet_valsplit_seed{seed}.pt: 25k val_tuning /
      25k val_test, offset seed+60000)
  S2  Lanczos rank-k via HVPs de cross-entropy em batches de train_idx
      (basis salva em .pt; reusada em resume/ablação)
  S3  sweep 2D (c x beta) em VAL_TUNING (25k): para cada célula, S amostras
      sequenciais (sample -> FixBN -> eval), acumulando posterior predictive
  S4  seleção conjunta (c*, beta*) por BMA-NLL em val_tuning
  S5  eval final da célula escolhida em VAL_TEST (25k) com amostras frescas

PROTOCOLO A (decisão Rodrigo, 12/jun/2026): tuning de hiperparâmetros em
metade held-out do val oficial (padrão da literatura de calibração), porque
a seleção em tuning_idx (subconjunto do TRAIN, visto pelo V1 no pré-treino)
é in-sample e estruturalmente cega ao benefício de calibração — confirmado
empiricamente: sweep in-sample 18 células monotônico com ótimo em beta->0
(nll 0.4666 vs 0.4665 do beta=0), enquanto (50,4) no val melhorava ECE.
O sweep in-sample fica no JSONL como evidência do viés (apêndice).

Protocolo (decisões Rodrigo, 11-12/jun/2026):
  - MAP fixo = torchvision ResNet50 IMAGENET1K_V1; seed controla só split/amostragem.
  - HVP: 5 batches de 64 de train_idx (convenção RandomRank30 do CIFAR).
  - Grid 2D revisado pós-controle beta=0 (12/jun): c in {50,150,450} x
    beta in {0.5,1,1.5,2,3,4}. Motivo: com lambda_bb=628 (c=50), beta=4 degrada
    o MAP em -2.6pts no val enquanto MAP+FixBN puro custa só -0.25pt =>
    a escala natural do ImageNet é menor que a do CIFAR; grid vai na direção
    de MENOS perturbacao (beta menor / c maior). z pareados entre c.
  - S=25 amostras sequenciais, FixBN por amostra (mesmos 25 batches fixos por seed),
    eval por amostra, BMA agregada. Sem batched-sample nesta versão.
  - Se --spine-steps > 0, constrói uma spine pós-hoc a partir do MAP por
    predictor--corrector em baixa loss; as amostras escolhem âncoras γ_t da
    spine e aplicam perturbações transversas em bases transportadas por
    projeção/reortonormalização ao longo da spine (--spine-transport transported).
  - Offsets de seed: split=seed; marglik=+10k; HVP=+20k; FixBN=+30k;
    samples sweep=+40k (+1000*idx_beta); samples final=+50k; val split=+60k; spine corrector/direction=+70k/+71k.
  - val oficial 50k = test intocado até S5.
  - val_tuning/val_test: metades seedadas do val oficial (offset seed+60000),
    persistidas com guard-rail. tuning_idx do split de train fica sem uso
    neste script (HVP/FixBN/marglik usam train_idx), mantido por compat.

Resume: registros completos em imagenet_trl_pipeline.jsonl são pulados
(mesma chave seed/rank/c/beta/samples). Basis em disco é reusada.

Uso:
  python imagenet_trl_pipeline.py --seeds 0
  python imagenet_trl_pipeline.py --seeds 0 1 2
"""

import argparse
import hashlib
import json
import fcntl
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

V1_CANONICAL_ACC = 0.7613


# ---------------------------------------------------------------- utils
def build_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def params_to_vector(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def set_params_from_vector(vec, params):
    with torch.no_grad():
        i = 0
        for p in params:
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p))
            i += n


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def append_jsonl(path, record):
    """Append com flock exclusivo: seguro para processos concorrentes
    (1 por GPU) escrevendo no mesmo JSONL. write único + lock garantem
    linha atômica; fsync garante durabilidade."""
    line = json.dumps(record) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [JSONL] AVISO: linha {i} malformada ignorada "
                      f"(leitura concorrente?)", flush=True)
    return rows


# ---------------------------------------------------------------- métricas
def ece_from(conf, hit, n_bins=15):
    ece = 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.any():
            ece += mask.float().mean().item() * abs(
                hit[mask].mean().item() - conf[mask].mean().item())
    return ece


def metrics_from_probs(probs, targets):
    """probs [N, C] (fp32, soma 1 por linha), targets [N]."""
    p_true = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(1e-12)
    nll = (-p_true.log()).mean().item()
    conf, pred = probs.max(dim=1)
    hit = (pred == targets).float()
    acc = hit.mean().item()
    brier = ((probs - F.one_hot(targets, probs.shape[1]).float()) ** 2
             ).sum(dim=1).mean().item()
    ece = ece_from(conf, hit)
    return {"acc": acc, "nll": nll, "ece": ece, "brier": brier}


@torch.no_grad()
def forward_probs(model, loader, accum=None, accum_sq=None, collect_targets=False):
    """Um passe pelo loader. Retorna métricas do passe.

    Se accum != None, soma as probs (CPU fp32) em accum para a BMA.
    Se accum_sq != None, soma probs**2 para estimar a variância funcional
    entre amostras: E[p^2] - E[p]^2. Ordem do loader deve ser fixa.
    """
    model.eval()
    confs, hits, nll_sum, total = [], [], 0.0, 0
    targets = [] if collect_targets else None
    pos = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        logp = F.log_softmax(model(x), dim=1)
        p = logp.exp()
        nll_sum += F.nll_loss(logp, y, reduction="sum").item()
        conf, pred = p.max(dim=1)
        confs.append(conf.cpu())
        hits.append((pred == y).float().cpu())
        p_cpu = p.float().cpu()
        if accum is not None:
            accum[pos:pos + y.numel()] += p_cpu
        if accum_sq is not None:
            accum_sq[pos:pos + y.numel()] += p_cpu.square()
        if collect_targets:
            targets.append(y.cpu())
        pos += y.numel()
        total += y.numel()
    conf = torch.cat(confs)
    hit = torch.cat(hits)
    m = {"acc": hit.mean().item(), "nll": nll_sum / total,
         "ece": ece_from(conf, hit), "n": total}
    if collect_targets:
        return m, torch.cat(targets)
    return m


# ---------------------------------------------------------------- HVP/Lanczos
def make_hvp(model, params, batches):
    model.eval()

    def hvp(v):
        i, v_params = 0, []
        for p in params:
            n = p.numel()
            v_params.append(v[i:i + n].view_as(p))
            i += n
        out = torch.zeros_like(v)
        for x, y in batches:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            loss = F.cross_entropy(model(x), y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((g * vp).sum() for g, vp in zip(grads, v_params))
            hv = torch.autograd.grad(dot, params, retain_graph=False)
            out += torch.cat([h.reshape(-1) for h in hv])
        return out / len(batches)

    return hvp


def lanczos_topk(hvp, dim, k, iters, basis_device):
    Q, alphas, betas = [], [], []
    q = torch.randn(dim, device=DEVICE)
    q /= q.norm()
    q_prev, beta_prev = None, 0.0
    for j in range(iters):
        t0 = time.time()
        w = hvp(q)
        alpha = torch.dot(w, q).item()
        w = w - alpha * q
        if q_prev is not None:
            w = w - beta_prev * q_prev
        for qi in Q:
            qi_dev = qi.to(DEVICE, non_blocking=True)
            w = w - torch.dot(w, qi_dev) * qi_dev
        beta = w.norm().item()
        alphas.append(alpha)
        Q.append(q.to(basis_device))
        print(f"  lanczos {j+1:3d}/{iters}  alpha={alpha:10.4f}  "
              f"beta={beta:10.4f}  ({time.time()-t0:.1f}s)", flush=True)
        if beta < 1e-8:
            print("  lanczos: early breakdown")
            break
        betas.append(beta)
        q_prev, beta_prev = q, beta
        q = w / beta

    m = len(alphas)
    T = torch.zeros(m, m)
    for i in range(m):
        T[i, i] = alphas[i]
        if i + 1 < m:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    evals, evecs = torch.linalg.eigh(T)
    idx = torch.argsort(evals, descending=True)[:k]
    top_evals = evals[idx]
    Qmat = torch.stack(Q)
    ritz = (evecs[:, idx].T.to(Qmat.dtype).to(Qmat.device) @ Qmat)
    ritz, _ = torch.linalg.qr(ritz.T.to(torch.float32))
    return top_evals, ritz.T.contiguous()


# ---------------------------------------------------------------- spine pós-hoc
def project_out_basis_rows(v, basis_rows):
    """Remove de v as componentes no span das linhas ortonormais de basis_rows."""
    coeff = basis_rows @ v
    return v - coeff @ basis_rows


def transport_basis_rows(basis_rows, v_parallel):
    """Transporta uma base transversa por projeção e reortonormalização.

    basis_rows tem forma [k, dim] e linhas aproximadamente ortonormais.
    O transporte discreto usado no TRL principal remove a componente ao longo
    da nova tangente longitudinal v_parallel e reortonormaliza as linhas.
    Não recomputa Lanczos/HVP em cada âncora.
    """
    v = normalize_or_fail(v_parallel, "v_parallel_transport")
    B = basis_rows - (basis_rows @ v).unsqueeze(1) * v.unsqueeze(0)
    # QR thin em B^T produz colunas ortonormais; voltamos para linhas.
    Q, _ = torch.linalg.qr(B.T.to(torch.float32), mode="reduced")
    B_new = Q.T.contiguous().to(basis_rows.dtype)
    # Alinha sinais com a base anterior para evitar flips arbitrários.
    dots = (B_new * basis_rows).sum(dim=1)
    signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
    return B_new * signs.unsqueeze(1)


def build_anchor_bases(basis, tangents, transport_mode, stored_steps=None):
    """Constrói Q_t para as âncoras da spine.

    Retorna None para fixed_N0; caso transported, retorna tensor
    [n_âncoras, k, dim] EM CPU (pré-alocado; só a base corrente vive na GPU
    durante o replay — fix do OOM do preflight de 12/jun: 9 bases na GPU +
    stack transiente pediam ~51GB). A BMA move Q_t pra GPU por amostra
    (~3GB H2D ≈ 1s, irrelevante vs 22s/amostra do eval). O replay percorre
    TODAS as tangentes (o transporte do build acontece a cada passo) e emite
    apenas nos índices de stored_steps, fidelidade com store_every>1.
    """
    if transport_mode == "fixed_N0":
        return None
    if transport_mode != "transported":
        raise ValueError(f"spine_transport inválido: {transport_mode}")
    if tangents is None:
        return None
    n_steps = int(tangents.shape[0])
    if stored_steps is None:
        stored_steps = list(range(n_steps))
    stored = sorted(set(int(s) for s in stored_steps))
    stored_set = set(stored)
    k, dim = basis.shape
    out = torch.empty((len(stored), k, dim), dtype=basis.dtype, device="cpu")
    B = basis
    tangents_dev = tangents.to(basis.device, non_blocking=True)
    j = 0
    for t in range(n_steps):
        B = transport_basis_rows(B, tangents_dev[t])
        if t in stored_set:
            out[j].copy_(B.detach().cpu())
            j += 1
    assert j == len(stored), f"replay emitiu {j} bases para {len(stored)} âncoras"
    return out


def normalize_or_fail(v, name, eps=1e-12):
    n = v.norm()
    if not torch.isfinite(n) or n.item() < eps:
        raise RuntimeError(f"{name} degenerou: norm={float(n)}")
    return v / n


def mean_ce_grad(model, params, batches):
    """Gradiente médio da CE em batches materializados; não constrói grafo de 2a ordem."""
    model.eval()
    acc = None
    for x, y in batches:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        loss = F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=False,
                                    retain_graph=False)
        g = torch.cat([gg.detach().reshape(-1) for gg in grads])
        acc = g if acc is None else acc + g
    return acc / len(batches)


@torch.no_grad()
def loss_on_cached_batches(model, batches):
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        total += F.cross_entropy(model(x), y, reduction="sum").item()
        n += y.numel()
    return total / n


def _float_tag(x):
    return str(x).replace("-", "m").replace(".", "p")


def get_or_create_spine(model, params, theta_map, map_state, basis, train_ds,
                        train_idx, split_sha, seed, args):
    """Constrói/carrega γ_0..γ_T e tangentes longitudinais v_t.

    Esta é a spine pós-hoc do TRL: parte de θ_MAP, escolhe uma direção
    longitudinal no complemento do subespaço rígido, avança por predictor e
    aplica correction por gradiente de CE projetado fora da direção longitudinal.
    No modo transported, a base transversa usada para definir o complemento é
    transportada por projeção/reortonormalização ao longo da spine.
    """
    if args.spine_steps <= 0:
        return theta_map.detach().cpu().unsqueeze(0), None, None, None, []

    tag = (f"imagenet_spine_seed{seed}_rank{args.rank}_T{args.spine_steps}"
           f"_ds{_float_tag(args.spine_step)}_eta{_float_tag(args.spine_corr_lr)}"
           f"_corr{args.spine_corr_steps}_b{args.spine_corr_batches}"
           f"_mdn{_float_tag(args.spine_max_delta_norm)}"
           f"_se{args.spine_store_every}_rot1"
           f"_upd{int(not args.spine_no_update_direction)}"
           f"_tr{args.spine_transport}.pt")
    spine_path = os.path.join(args.out_dir, tag)
    if os.path.exists(spine_path):
        print(f"  [SPINE] reusando {spine_path}")
        blob = torch.load(spine_path, map_location="cpu", weights_only=False)
        assert blob["seed"] == seed
        assert blob["rank"] == args.rank
        assert blob["split_sha256"] == split_sha
        assert blob["spine_steps"] == args.spine_steps
        assert blob["spine_step"] == args.spine_step
        assert blob["spine_corr_lr"] == args.spine_corr_lr
        assert blob["spine_corr_steps"] == args.spine_corr_steps
        assert blob["spine_corr_batches"] == args.spine_corr_batches
        assert blob.get("spine_max_delta_norm") == args.spine_max_delta_norm
        assert blob.get("spine_store_every", 1) == args.spine_store_every
        assert blob.get("spine_update_direction", True) == (not args.spine_no_update_direction)
        assert blob.get("spine_transport", blob.get("transport", "fixed_N0")) == args.spine_transport
        return (blob["anchors"], blob.get("tangents"),
                blob.get("stored_steps"), spine_path,
                blob.get("diagnostics", []))

    update_direction = not args.spine_no_update_direction
    print(f"  [SPINE] construindo spine pós-hoc (regra canônica CIFAR): "
          f"T={args.spine_steps}, ds={args.spine_step}, eta={args.spine_corr_lr}, "
          f"max_delta_norm={args.spine_max_delta_norm}, "
          f"store_every={args.spine_store_every}, "
          f"corr_steps={args.spine_corr_steps}, batches={args.spine_corr_batches}, "
          f"update_dir={update_direction}, transport={args.spine_transport}")
    # CANÔNICO (cifar100_all_methods_iclr.py): 1 batch ROTATIVO por passo do
    # corrector — batches fixos viram descida in-sample (preflight 12/jun:
    # loss 0.48→0.17 em 8 passos com 5 batches fixos = fine-tuning, não vale).
    # spine_corr_batches agora = tamanho do pool de rotação.
    pool_n = max(args.spine_corr_batches, min(args.spine_steps, 16))
    corr_pool = materialize_batches(
        train_ds, train_idx, pool_n, args.batch_size,
        seed + 70_000, args.workers, with_targets=True)
    # pool de DIAGNÓSTICO held-out (disjunto do corrector por seed): a loss
    # reportada nos [SPINE] é out-of-corrector-sample, drift honesto.
    diag_pool = materialize_batches(
        train_ds, train_idx, 5, args.batch_size,
        seed + 72_000, args.workers, with_targets=True)

    basis_dev = basis.to(DEVICE, non_blocking=True)
    basis_t = basis_dev
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 71_000)
    v = torch.randn(theta_map.numel(), device=DEVICE, generator=gen,
                    dtype=theta_map.dtype)
    v = project_out_basis_rows(v, basis_t)
    v = normalize_or_fail(v, "v_parallel")

    gamma = theta_map.detach().clone()
    anchors = [gamma.detach().cpu().clone()]
    tangents = [v.detach().cpu().clone()]   # TODAS as tangentes (replay do transporte)
    stored_steps = [0]                       # índices t com âncora armazenada
    diagnostics = []
    clip_hits = 0

    # loss no MAP no pool de diagnóstico (held-out do corrector).
    set_params_from_vector(gamma, params)
    loss0 = loss_on_cached_batches(model, diag_pool)
    print(f"  [SPINE] t=0 diag_loss={loss0:.4f} |gamma-map|=0.0000")
    diagnostics.append({"t": 0, "loss": loss0, "dist_from_map": 0.0,
                        "step_norm": 0.0, "stored": True})

    for t in range(args.spine_steps):
        t0 = time.time()
        # regra canônica (cifar100_all_methods_iclr.py, TRL.build):
        # delta = ds*v - eta*g_perp, norm control no DELTA combinado,
        # gradiente de UM batch rotativo por passo.
        # corr_steps>1 reaplica só o termo do corrector (canônico usa 1).
        gamma_next = gamma
        delta_total = torch.zeros_like(gamma)
        step_batch = [corr_pool[t % len(corr_pool)]]
        for ci in range(args.spine_corr_steps):
            set_params_from_vector(gamma_next, params)
            g = mean_ce_grad(model, params, step_batch)
            g_perp = g - torch.dot(g, v) * v
            if ci == 0:
                delta = args.spine_step * v - args.spine_corr_lr * g_perp
            else:
                delta = -args.spine_corr_lr * g_perp
            dnorm = delta.norm()
            if dnorm > args.spine_max_delta_norm:
                delta = delta * (args.spine_max_delta_norm / (dnorm + 1e-12))
                clip_hits += 1
            gamma_next = gamma_next + delta
            delta_total = delta_total + delta

        step_vec = delta_total
        # canônico: tangente do deslocamento realizado, sem projeção extra
        # fora de N (o transporte cuida da ortogonalidade da base, não do v).
        if update_direction:
            d_norm = step_vec.norm()
            if d_norm > 1e-9:
                v = (step_vec / d_norm).detach()
        if args.spine_transport == "transported":
            basis_t = transport_basis_rows(basis_t, v)
        elif args.spine_transport != "fixed_N0":
            raise ValueError(f"spine_transport inválido: {args.spine_transport}")
        gamma = gamma_next.detach()
        set_params_from_vector(gamma, params)
        loss_t = loss_on_cached_batches(model, diag_pool)
        dist = (gamma - theta_map).norm().item()
        step_norm = step_vec.norm().item()
        store_this = (((t + 1) % args.spine_store_every) == 0) or \
                     (t + 1 == args.spine_steps)
        tangents.append(v.detach().cpu().clone())   # sempre: replay do transporte
        if store_this:
            anchors.append(gamma.detach().cpu().clone())
            stored_steps.append(t + 1)
        diagnostics.append({"t": t + 1, "loss": loss_t,
                            "dist_from_map": dist, "step_norm": step_norm,
                            "stored": store_this,
                            "elapsed_s": round(time.time() - t0, 1)})
        print(f"  [SPINE] t={t+1}/{args.spine_steps} diag_loss={loss_t:.4f} "
              f"|gamma-map|={dist:.4f} step={step_norm:.4f} "
              f"{'[stored]' if store_this else ''} "
              f"({time.time()-t0:.1f}s)", flush=True)

    if clip_hits:
        print(f"  [SPINE] norm control ativou em {clip_hits} passo(s)")
    anchors = torch.stack(anchors, dim=0)
    tangents = torch.stack(tangents, dim=0)
    torch.save({"anchors": anchors, "tangents": tangents,
                "stored_steps": stored_steps, "diagnostics": diagnostics,
                "seed": seed, "rank": args.rank, "split_sha256": split_sha,
                "spine_steps": args.spine_steps, "spine_step": args.spine_step,
                "spine_corr_lr": args.spine_corr_lr,
                "spine_corr_steps": args.spine_corr_steps,
                "spine_corr_batches": args.spine_corr_batches,
                "spine_max_delta_norm": args.spine_max_delta_norm,
                "spine_store_every": args.spine_store_every,
                "spine_corr_seed": seed + 70_000,
                "spine_direction_seed": seed + 71_000,
                "spine_update_direction": update_direction,
                "spine_clip_hits": clip_hits,
                "spine_corr_rule": "one_rotating_batch_per_step_canonical",
                "spine_corr_pool_size": pool_n,
                "spine_diag_pool_seed": seed + 72_000,
                "spine_transport": args.spine_transport,
                "transport": args.spine_transport}, spine_path)
    print(f"  [SPINE] salva em {spine_path} ({anchors.shape[0]} âncoras)")

    # restaura estado MAP completo, incluindo BN stats.
    model.load_state_dict(map_state)
    set_params_from_vector(theta_map, params)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return anchors, tangents, stored_steps, spine_path, diagnostics


# ---------------------------------------------------------------- FixBN
def reset_bn(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None


def fixbn(model, cached_batches):
    reset_bn(model)
    model.train()
    with torch.no_grad():
        for x in cached_batches:
            model(x.to(DEVICE, non_blocking=True))
    model.eval()


# ---------------------------------------------------------------- dados
def materialize_batches(dataset, indices, n_batches, batch_size, seed, workers,
                        with_targets):
    """Sorteia n_batches*batch_size índices (sem reposição) e materializa em RAM."""
    g = torch.Generator().manual_seed(seed)
    sel = indices[torch.randperm(len(indices), generator=g)[:n_batches * batch_size]]
    loader = DataLoader(Subset(dataset, sel.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=False)
    out = []
    for x, y in loader:
        out.append((x, y) if with_targets else x)
    assert len(out) == n_batches, (len(out), n_batches)
    return out


# ---------------------------------------------------------------- pipeline
def draw_delta(basis, lam, gen):
    z = torch.randn(basis.shape[0], device=basis.device, generator=gen)
    return (z / lam.sqrt()) @ basis  # [dim]


def run_bma_eval(model, params, theta_map, basis, lam, beta, n_samples,
                 loader, n_eval, n_classes, fixbn_batches, sample_seed, tag,
                 anchors=None, spine_sampling="cycle", anchor_bases=None):
    """S amostras sequenciais: anchor -> sample -> FixBN -> eval.

    anchors=None reproduz o single-checkpoint. anchors=[γ_0..γ_T] ativa a
    versão full-spine: cada amostra escolhe uma âncora longitudinal. Se
    anchor_bases=[Q_0..Q_T] for fornecido, usa a base transportada Q_t da
    própria âncora; caso contrário, usa N_0 fixo.
    """
    gen = torch.Generator(device=basis.device).manual_seed(sample_seed)
    cpu_gen = torch.Generator().manual_seed(sample_seed + 123_456)
    accum = torch.zeros(n_eval, n_classes, dtype=torch.float32)
    accum_sq = torch.zeros(n_eval, n_classes, dtype=torch.float32)
    per_sample, targets = [], None
    n_anchors = 1 if anchors is None else int(anchors.shape[0])
    for s in range(n_samples):
        t0 = time.time()
        if anchors is None:
            anchor_idx = 0
            anchor = theta_map
        else:
            if spine_sampling == "cycle":
                anchor_idx = s % n_anchors
            elif spine_sampling == "random":
                anchor_idx = int(torch.randint(n_anchors, (1,), generator=cpu_gen))
            else:
                raise ValueError(f"spine_sampling inválido: {spine_sampling}")
            anchor = anchors[anchor_idx].to(theta_map.device, non_blocking=True)
        if anchor_bases is None:
            sample_basis = basis
            sample_lam = lam
        else:
            # anchor_bases vive em CPU (fix OOM); move só a Q_t da amostra.
            sample_basis = anchor_bases[anchor_idx].to(basis.device,
                                                       non_blocking=True)
            sample_lam = lam[anchor_idx]
        delta = draw_delta(sample_basis, sample_lam, gen)
        set_params_from_vector(anchor + beta * delta.to(theta_map.device), params)
        fixbn(model, fixbn_batches)
        if targets is None:
            m, targets = forward_probs(model, loader, accum=accum,
                                       accum_sq=accum_sq,
                                       collect_targets=True)
        else:
            m = forward_probs(model, loader, accum=accum, accum_sq=accum_sq)
        m["sample"] = s
        m["anchor_idx"] = anchor_idx
        m["elapsed_s"] = round(time.time() - t0, 1)
        per_sample.append(m)
        print(f"    [{tag} beta={beta:g} s={s+1:02d}/{n_samples} "
              f"anchor={anchor_idx}/{n_anchors-1}] "
              f"acc={m['acc']:.4f} nll={m['nll']:.4f} ece={m['ece']:.4f} "
              f"({m['elapsed_s']}s)", flush=True)
    mean_probs = accum / n_samples
    bma = metrics_from_probs(mean_probs, targets)
    if n_samples > 1:
        var_probs = (accum_sq / n_samples - mean_probs.square()).clamp_min(0.0)
        bma["func_var_mean"] = var_probs.mean().item()
        bma["func_var_trace"] = var_probs.sum(dim=1).mean().item()
        bma["func_var_max_mean"] = var_probs.max(dim=1).values.mean().item()
    else:
        bma["func_var_mean"] = 0.0
        bma["func_var_trace"] = 0.0
        bma["func_var_max_mean"] = 0.0
    print(f"    [{tag} beta={beta:g} BMA/{n_samples} anchors={n_anchors}] "
          f"acc={bma['acc']:.4f} nll={bma['nll']:.4f} "
          f"ece={bma['ece']:.4f} brier={bma['brier']:.4f} "
          f"fvar_trace={bma['func_var_trace']:.6g}", flush=True)
    return bma, per_sample


def get_or_create_val_split(val_ds, seed, out_dir, tuning_size=25000):
    """Split persistido do val oficial: 25k val_tuning / 25k val_test."""
    path = os.path.join(out_dir, f"imagenet_valsplit_seed{seed}.pt")
    n = len(val_ds)
    if os.path.exists(path):
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["classes"] == val_ds.classes, (
            "GUARD-RAIL: classes do val split divergem do ImageFolder. ABORTANDO.")
        assert loaded["n_dataset"] == n and loaded["tuning_size"] == tuning_size
        print(f"  [valsplit] reusando {path}")
        return loaded["val_tuning_idx"], loaded["val_test_idx"], path
    g = torch.Generator().manual_seed(seed + 60_000)
    perm = torch.randperm(n, generator=g)
    val_tuning_idx = perm[:tuning_size].clone()
    val_test_idx = perm[tuning_size:].clone()
    torch.save({"seed": seed, "val_split_seed": seed + 60_000, "n_dataset": n,
                "tuning_size": tuning_size, "val_tuning_idx": val_tuning_idx,
                "val_test_idx": val_test_idx, "classes": val_ds.classes,
                "created_unix": time.time()}, path)
    print(f"  [valsplit] criado {path} (tuning={len(val_tuning_idx)}, "
          f"test={len(val_test_idx)})")
    return val_tuning_idx, val_test_idx, path


def run_seed(seed, args, train_ds, val_ds, map_val_metrics):
    print(f"\n========== SEED {seed} ==========", flush=True)
    jsonl_path = os.path.join(args.out_dir, "imagenet_trl_pipeline.jsonl")
    done = load_jsonl(jsonl_path)

    def canonical_spine_cfg(steps, spine_step, corr_lr, corr_steps,
                             corr_batches, max_delta_norm, store_every,
                             update_direction, sampling, transport):
        """Canonical resume key for spine-only options.

        When spine_steps == 0, all spine-only arguments are irrelevant and
        must collapse to the same key used by historical single-checkpoint
        JSONL rows, independent of current CLI defaults. This prevents
        rerunning old single-checkpoint cells just because --spine-step or
        --spine-sampling has a nonzero default.
        """
        steps = int(steps or 0)
        if steps == 0:
            return (0, 0.0, 0.0, 0, 0, 0.0, 0, True, "cycle", "none")
        return (steps, float(spine_step), float(corr_lr), int(corr_steps),
                int(corr_batches), float(max_delta_norm), int(store_every),
                bool(update_direction), str(sampling), str(transport))

    def spine_cfg_from_record(r):
        steps = r.get("spine_steps", 0)
        return canonical_spine_cfg(
            steps,
            r.get("spine_step", 0.0),
            r.get("spine_corr_lr", 1e-3),
            r.get("spine_corr_steps", 1),
            r.get("spine_corr_batches", 5),
            r.get("spine_max_delta_norm", 0.02),
            r.get("spine_store_every", 1),
            r.get("spine_update_direction", True),
            r.get("spine_sampling", "cycle"),
            r.get("spine_transport", r.get("transport", "fixed_N0")),
        )

    args_spine_cfg = canonical_spine_cfg(
        args.spine_steps, args.spine_step, args.spine_corr_lr,
        args.spine_corr_steps, args.spine_corr_batches,
        args.spine_max_delta_norm, args.spine_store_every,
        not args.spine_no_update_direction, args.spine_sampling,
        args.spine_transport)

    def key_of(r):
        return (r.get("stage"), r.get("eval_set"), r.get("seed"),
                r.get("rank"), r.get("boost_c"), r.get("beta"),
                r.get("samples"), spine_cfg_from_record(r))

    done_keys = {key_of(r) for r in done}

    # ---- split persistido (fonte da verdade) ----
    split_path = os.path.join(args.out_dir, f"imagenet_split_seed{seed}.pt")
    assert os.path.exists(split_path), f"split ausente: {split_path}"
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    assert split["classes"] == train_ds.classes, (
        "GUARD-RAIL: classes do split divergem do ImageFolder atual. ABORTANDO.")
    train_idx, tuning_idx = split["train_idx"], split["tuning_idx"]
    assert split.get("n_dataset", len(train_ds)) == len(train_ds), (
        "GUARD-RAIL: n_dataset do split diverge do ImageFolder atual. ABORTANDO.")
    if "tuning_size" in split:
        assert split["tuning_size"] == len(tuning_idx), (
            "GUARD-RAIL: tuning_size do split diverge dos índices carregados. "
            "ABORTANDO.")
    assert len(train_idx) + len(tuning_idx) == len(train_ds), (
        "GUARD-RAIL: train_idx + tuning_idx != n_dataset. ABORTANDO.")
    split_sha = sha256_of(split_path)
    print(f"  split: train={len(train_idx)} tuning={len(tuning_idx)} "
          f"sha={split_sha[:12]}...")

    # ---- lambda_base do marglik ----
    lam_base = args.lambda_base
    if lam_base is None:
        rows = [r for r in load_jsonl(os.path.join(args.out_dir,
                                                   "imagenet_marglik.jsonl"))
                if r.get("seed") == seed]
        assert rows, f"lambda_base do seed {seed} ausente no JSONL do marglik"
        lam_base = rows[-1]["lambda_base"]
    print(f"  lambda_base={lam_base:.4f}  grid c={args.boost_c}  "
          f"lambda_bb={[round(max(c * lam_base, args.prior_floor), 1) for c in args.boost_c]}")

    # ---- modelo + MAP ----
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1").to(DEVICE)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)
    theta_map = params_to_vector(params).clone()
    map_state = {k: v.clone() for k, v in model.state_dict().items()}

    head_ids = set(id(p) for p in model.fc.parameters())
    head_mask = torch.zeros(dim, dtype=torch.bool)
    i = 0
    for p in params:
        n = p.numel()
        if id(p) in head_ids:
            head_mask[i:i + n] = True
        i += n

    # ---- S2: Lanczos (com cache em disco) ----
    basis_path = os.path.join(
        args.out_dir, f"imagenet_basis_seed{seed}_rank{args.rank}.pt")
    if os.path.exists(basis_path):
        print(f"  [S2] reusando basis {basis_path}")
        blob = torch.load(basis_path, map_location="cpu", weights_only=False)
        assert blob["rank"] == args.rank
        assert blob["seed"] == seed
        assert blob["split_sha256"] == split_sha, (
            "basis em disco foi gerada com split diferente — apague ou ajuste args")
        assert blob["hvp_seed"] == seed + 20_000
        assert blob["lanczos_iters"] == args.lanczos_iters and \
            blob["hvp_batches"] == args.hvp_batches, \
            "basis em disco tem config de Lanczos diferente — apague ou ajuste args"
        evals = blob["eigvals"]
        basis = blob["basis"].to(args.basis_device)
    else:
        t0 = time.time()
        hvp_data = materialize_batches(
            train_ds, train_idx, args.hvp_batches, args.batch_size,
            seed + 20_000, args.workers, with_targets=True)
        hvp = make_hvp(model, params, hvp_data)
        print(f"  [S2] Lanczos rank-{args.rank}, {args.lanczos_iters} iters, "
              f"{args.hvp_batches} batches/HVP de train_idx...")
        evals, basis = lanczos_topk(hvp, dim, args.rank, args.lanczos_iters,
                                    args.basis_device)
        del hvp_data
        torch.save({"eigvals": evals, "basis": basis.cpu(),
                    "rank": args.rank, "lanczos_iters": args.lanczos_iters,
                    "hvp_batches": args.hvp_batches,
                    "hvp_seed": seed + 20_000, "seed": seed,
                    "split_sha256": split_sha}, basis_path)
        print(f"  [S2] {time.time()-t0:.0f}s  top={evals[0]:.1f} "
              f"last={evals[-1]:.1f}  basis salva em {basis_path}")
        basis = basis.to(args.basis_device)

    # prior projetado por direção — depende da massa no head da base usada.
    hm = head_mask.to(basis.device)
    evals_dev = torch.clamp(evals.to(basis.device), min=0.0)

    def lam_for(c, basis_rows):
        lam_bb = max(c * lam_base, args.prior_floor)
        if basis_rows.dim() == 2:
            head_mass = (basis_rows[:, hm] ** 2).sum(dim=1)
            proj_prior = lam_bb + (lam_base - lam_bb) * head_mass
            return evals_dev + proj_prior, lam_bb
        if basis_rows.dim() == 3:
            # basis_rows pode estar em CPU (anchor_bases pós-fix de OOM):
            # head_mass por âncora em streaming, resultado no device de evals.
            n_anc = basis_rows.shape[0]
            hm_cpu = hm.cpu() if hm.device.type != "cpu" else hm
            masses = []
            for a in range(n_anc):
                rows = basis_rows[a]
                mask = hm_cpu if rows.device.type == "cpu" else hm
                masses.append((rows[:, mask] ** 2).sum(dim=1)
                              .to(evals_dev.device))
            head_mass = torch.stack(masses, dim=0)
            proj_prior = lam_bb + (lam_base - lam_bb) * head_mass
            return evals_dev.unsqueeze(0) + proj_prior, lam_bb
        raise ValueError(f"basis_rows com shape inesperado: {basis_rows.shape}")

    # ---- SPINE: γ_0..γ_T pós-hoc a partir do MAP (opcional) ----
    anchors, tangents, stored_steps, spine_path, spine_diag = get_or_create_spine(
        model, params, theta_map, map_state, basis, train_ds, train_idx,
        split_sha, seed, args)
    anchor_bases = None
    if args.spine_steps > 0:
        anchor_bases = build_anchor_bases(basis, tangents, args.spine_transport,
                                          stored_steps=stored_steps)
        print(f"  [SPINE] ativa: {anchors.shape[0]} âncoras "
              f"(de {args.spine_steps + 1} passos, store_every="
              f"{args.spine_store_every}), sampling={args.spine_sampling}, "
              f"transport={args.spine_transport}, path={spine_path}")
        if anchor_bases is not None:
            assert anchor_bases.shape[0] == anchors.shape[0], (
                f"bases ({anchor_bases.shape[0]}) != âncoras ({anchors.shape[0]})")
            print(f"  [SPINE] transported bases: shape={tuple(anchor_bases.shape)} "
                  f"device={anchor_bases.device} mem~"
                  f"{anchor_bases.numel()*anchor_bases.element_size()/1e9:.2f}GB")
    else:
        print("  [SPINE] desativada: single-checkpoint transverse")
    model.load_state_dict(map_state)
    set_params_from_vector(theta_map, params)

    # ---- FixBN: 25 batches fixos por seed, mesmos pra todas as amostras ----
    print(f"  [FixBN] materializando {args.fixbn_batches} batches fixos "
          f"(seed {seed + 30_000})...")
    fixbn_batches = materialize_batches(
        train_ds, train_idx, args.fixbn_batches, args.batch_size,
        seed + 30_000, args.workers, with_targets=False)

    # ---- splits do val (Protocolo A) + loaders (ordem fixa) ----
    val_tuning_idx, val_test_idx, valsplit_path = get_or_create_val_split(
        val_ds, seed, args.out_dir, args.val_tuning_size)
    valsplit_sha = sha256_of(valsplit_path)
    val_tuning_loader = DataLoader(
        Subset(val_ds, val_tuning_idx.tolist()), batch_size=args.eval_bs,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    val_test_loader = DataLoader(
        Subset(val_ds, val_test_idx.tolist()), batch_size=args.eval_bs,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    n_classes = len(train_ds.classes)

    # Canonical metadata: single-checkpoint rows keep spine-only fields neutral
    # so they remain compatible with historical JSONL records.
    (rec_spine_steps, rec_spine_step, rec_corr_lr, rec_corr_steps,
     rec_corr_batches, rec_max_delta_norm, rec_store_every,
     rec_update_direction, rec_sampling, rec_transport) = args_spine_cfg
    rec_spine_path = spine_path if args.spine_steps > 0 else None
    rec_spine_diag = spine_diag if args.spine_steps > 0 else []

    base_record = {
        "seed": seed, "rank": args.rank,
        "samples": args.samples, "lambda_base": lam_base,
        "prior_floor": args.prior_floor, "lanczos_iters": args.lanczos_iters,
        "hvp_batches": args.hvp_batches, "fixbn_batches": args.fixbn_batches,
        "batch_size": args.batch_size, "checkpoint_source": "torchvision",
        "weights": "IMAGENET1K_V1", "map_is_fixed": True,
        "seed_controls": "split_and_sampling_only",
        "protocol": "A_val_holdout_tuning",
        "split_sha256": split_sha, "valsplit_sha256": valsplit_sha,
        "eigval_top": float(evals[0]), "eigval_last": float(evals[-1]),
        "map_val": map_val_metrics,
        "seed_offsets": {"split": 0, "marglik": 10000, "hvp": 20000,
                         "fixbn": 30000, "sweep_samples": 40000,
                         "final_samples": 50000, "val_split": 60000,
                         "spine_corr": 70000, "spine_direction": 71000},
        "z_paired_across_c": True,  # mesmo sample_seed por beta => mesmos z entre c
        "trl_variant": ("posthoc_spine_transported_basis" if args.spine_steps > 0 and args.spine_transport == "transported"
                        else "posthoc_spine_fixed_N0" if args.spine_steps > 0
                        else "single_checkpoint_transverse"),
        "spine_anchor_count": int(anchors.shape[0]),
        "spine_steps": rec_spine_steps,
        "spine_step": rec_spine_step,
        "spine_corr_lr": rec_corr_lr,
        "spine_corr_steps": rec_corr_steps,
        "spine_corr_batches": rec_corr_batches,
        "spine_max_delta_norm": rec_max_delta_norm,
        "spine_store_every": rec_store_every,
        "spine_update_direction": rec_update_direction,
        "spine_sampling": rec_sampling,
        "spine_path": rec_spine_path,
        "spine_transport": rec_transport,
        "spine_rule": "canonical_cifar_delta_rotbatch" if args.spine_steps > 0 else None,
        "spine_divergences_from_canonical": [],
        "spine_diagnostics": rec_spine_diag,
    }

    # ---- S3: sweep 2D (c x beta) em val_tuning (25k held-out) ----
    sweep_results = {}  # (c, beta) -> bma
    for r in done:
        if (r.get("stage") == "sweep" and
                r.get("eval_set") == "val_tuning_25k" and
                r.get("seed") == seed and
                r.get("rank") == args.rank and r.get("boost_c") in args.boost_c
                and r.get("samples") == args.samples
                and spine_cfg_from_record(r) == args_spine_cfg):
            sweep_results[(r["boost_c"], r["beta"])] = r["bma"]

    # MAP+nada baseline nas duas metades (referência das tabelas)
    map_tuning = forward_probs(model, val_tuning_loader)
    map_test = forward_probs(model, val_test_loader)
    print(f"  MAP val_tuning: acc={map_tuning['acc']:.4f} "
          f"nll={map_tuning['nll']:.4f} ece={map_tuning['ece']:.4f}")
    print(f"  MAP val_test:   acc={map_test['acc']:.4f} "
          f"nll={map_test['nll']:.4f} ece={map_test['ece']:.4f}")
    base_record["map_val_tuning"] = map_tuning
    base_record["map_val_test"] = map_test

    for c in args.boost_c:
        lam, lam_bb = lam_for(c, anchor_bases if anchor_bases is not None else basis)
        for bi, beta in enumerate(args.betas):
            k = ("sweep", "val_tuning_25k", seed, args.rank, c, beta,
                 args.samples, args_spine_cfg)
            if k in done_keys:
                print(f"  [S3] c={c:g} beta={beta:g} já no JSONL — pulando (resume)")
                continue
            t0 = time.time()
            bma, per_sample = run_bma_eval(
                model, params, theta_map, basis, lam, beta, args.samples,
                val_tuning_loader, len(val_tuning_idx), n_classes,
                fixbn_batches,
                sample_seed=seed + 40_000 + 1000 * bi, tag=f"vtun c={c:g}",
                anchors=anchors if args.spine_steps > 0 else None,
                spine_sampling=args.spine_sampling,
                anchor_bases=anchor_bases)
            rec = dict(base_record)
            rec.update({"stage": "sweep", "boost_c": c, "lambda_bb": lam_bb,
                        "beta": beta, "eval_set": "val_tuning_25k",
                        "n_eval": len(val_tuning_idx), "bma": bma,
                        "per_sample": per_sample,
                        "sample_seed": seed + 40_000 + 1000 * bi,
                        "elapsed_s": round(time.time() - t0, 1),
                        "timestamp_unix": time.time()})
            append_jsonl(jsonl_path, rec)
            sweep_results[(c, beta)] = bma

    # ---- S4: seleção conjunta (c, beta) ----
    grid = [(c, b) for c in args.boost_c for b in args.betas]
    missing = [g for g in grid if g not in sweep_results]
    assert not missing, f"células sem resultado de sweep: {missing}"
    best_c, best_beta = min(grid,
                            key=lambda g: sweep_results[g][args.select_metric])
    print(f"\n  [S4] seleção por {args.select_metric} em val_tuning: "
          f"c*={best_c:g} beta*={best_beta:g}")
    print(f"    MAP ref: nll={map_tuning['nll']:.4f} ece={map_tuning['ece']:.4f}")
    print(f"    {'c \\ beta':>10s} | " +
          "  ".join(f"{b:>7g}" for b in args.betas) + f"   ({args.select_metric})")
    for c in args.boost_c:
        row = "  ".join(f"{sweep_results[(c, b)][args.select_metric]:7.4f}"
                        for b in args.betas)
        print(f"    {c:>10g} | {row}")
    on_border = (best_c in (args.boost_c[0], args.boost_c[-1]) or
                 best_beta in (args.betas[0], args.betas[-1]))
    if on_border:
        print("  [S4] AVISO: ótimo na borda do grid — considerar probe extra.")

    # ---- S5: eval final em val_test (amostras frescas) ----
    k = ("final", "val_test_25k", seed, args.rank, best_c, best_beta,
         args.samples, args_spine_cfg)
    if k in done_keys:
        print("  [S5] final já no JSONL — pulando (resume)")
    else:
        lam, lam_bb = lam_for(best_c, anchor_bases if anchor_bases is not None else basis)
        t0 = time.time()
        bma, per_sample = run_bma_eval(
            model, params, theta_map, basis, lam, best_beta, args.samples,
            val_test_loader, len(val_test_idx), n_classes, fixbn_batches,
            sample_seed=seed + 50_000, tag="TEST",
            anchors=anchors if args.spine_steps > 0 else None,
            spine_sampling=args.spine_sampling,
            anchor_bases=anchor_bases)
        rec = dict(base_record)
        rec.update({"stage": "final", "boost_c": best_c, "lambda_bb": lam_bb,
                    "beta": best_beta, "eval_set": "val_test_25k",
                    "n_eval": len(val_test_idx), "bma": bma,
                    "per_sample": per_sample,
                    "select_metric": args.select_metric,
                    "sweep_bma_by_cell": {f"{c}|{b}": sweep_results[(c, b)]
                                          for c, b in grid},
                    "sample_seed": seed + 50_000,
                    "elapsed_s": round(time.time() - t0, 1),
                    "timestamp_unix": time.time()})
        append_jsonl(jsonl_path, rec)
        print(f"  [S5] TEST vs MAP: nll {bma['nll']:.4f} vs "
              f"{map_test['nll']:.4f} | ece {bma['ece']:.4f} vs "
              f"{map_test['ece']:.4f} | acc {bma['acc']:.4f} vs "
              f"{map_test['acc']:.4f}")

    # restaura MAP (pesos + BN stats) pro próximo seed
    model.load_state_dict(map_state)
    del basis
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--rank", type=int, default=30)
    ap.add_argument("--spine-steps", type=int, default=0,
                    help="T da spine pós-hoc; 0 mantém single-checkpoint")
    ap.add_argument("--spine-step", type=float, default=0.01,
                    help="Δs do predictor (canônico CIFAR: trl_step_size=0.01)")
    ap.add_argument("--spine-corr-steps", type=int, default=1,
                    help="passos corrector por ponto (canônico: 1, fundido no delta)")
    ap.add_argument("--spine-corr-lr", type=float, default=1e-3,
                    help="eta do corrector (canônico CIFAR: trl_eta=1e-3)")
    ap.add_argument("--spine-corr-batches", type=int, default=5,
                    help="batches de train_idx do corrector (canônico usa 1 batch "
                         "rotativo/passo; 5 fixos = menos ruído, divergência "
                         "deliberada registrada no JSONL)")
    ap.add_argument("--spine-max-delta-norm", type=float, default=0.02,
                    help="norm control no DELTA combinado ds*v - eta*g_perp "
                         "(canônico CIFAR: trl_max_delta_norm=0.02)")
    ap.add_argument("--spine-store-every", type=int, default=1,
                    help="guarda 1 a cada N âncoras (canônico: trl_store_every; "
                         "controla custo de anchor_bases)")
    ap.add_argument("--spine-no-update-direction", action="store_true",
                    help="DESLIGA a atualização da tangente (canônico atualiza "
                         "sempre: v = d/||d||). Só para ablação.")
    ap.add_argument("--spine-transport", default="transported",
                    choices=["transported", "fixed_N0"],
                    help="transported = projeta/reortonormaliza Q_t ao longo da spine; "
                         "fixed_N0 = ablação com base fixa do MAP")
    ap.add_argument("--spine-sampling", default="cycle", choices=["cycle", "random"],
                    help="como escolher âncoras γ_t durante BMA")
    ap.add_argument("--lanczos-iters", type=int, default=90)
    ap.add_argument("--hvp-batches", type=int, default=5)
    ap.add_argument("--fixbn-batches", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", dest="eval_bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boost-c", type=float, nargs="+", default=[50.0, 150.0, 450.0])
    ap.add_argument("--prior-floor", type=float, default=5.0)
    ap.add_argument("--lambda-base", type=float, default=None,
                    help="override; default = ler do imagenet_marglik.jsonl")
    ap.add_argument("--select-metric", default="nll", choices=["nll", "ece"])
    ap.add_argument("--basis-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--train-root",
                    required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--out-dir", default="results/imagenet_resnet50_scalecheck")
    ap.add_argument("--val-tuning-size", type=int, default=25000)
    ap.add_argument("--skip-gate", action="store_true",
                    help="não abortar se MAP no val desviar do canônico")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={DEVICE}  torch={torch.__version__}  "
          f"torchvision={torchvision.__version__}", flush=True)

    tfm = build_eval_transform()
    print(">>> ImageFolder train (scan)...", flush=True)
    train_ds = datasets.ImageFolder(args.train_root, tfm)
    assert len(train_ds.classes) == 1000
    print(">>> ImageFolder val...", flush=True)
    val_ds = datasets.ImageFolder(args.val_root, tfm)
    assert len(val_ds) == 50000 and len(val_ds.classes) == 1000
    assert val_ds.classes == train_ds.classes, (
        "ordem de classes train != val — ABORTANDO")
    val_loader = DataLoader(val_ds, batch_size=args.eval_bs, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ---- S0: gate MAP no val completo ----
    print(">>> [S0] gate: MAP no val completo...", flush=True)
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1").to(DEVICE)
    map_val = forward_probs(model, val_loader)
    print(f"    MAP val: acc={map_val['acc']:.4f} nll={map_val['nll']:.4f} "
          f"ece={map_val['ece']:.4f}")
    if abs(map_val["acc"] - V1_CANONICAL_ACC) > 0.005 and not args.skip_gate:
        raise SystemExit(
            f"GATE FALHOU: MAP acc={map_val['acc']:.4f} vs canônico "
            f"{V1_CANONICAL_ACC} — dataset/transform suspeitos. "
            f"(--skip-gate pra ignorar)")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for seed in args.seeds:
        run_seed(seed, args, train_ds, val_ds, map_val)

    print("\n>>> Pipeline concluído. Resultados em "
          f"{os.path.join(args.out_dir, 'imagenet_trl_pipeline.jsonl')}")


if __name__ == "__main__":
    main()

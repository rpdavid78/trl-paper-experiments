# ============================================================================
# boost_betaperp_sweep_2d.py
#
# Sweep 2D conjunto (c, beta_perp) para o prior block-isotropic do TRL.
# Objetivo: testar se o "otimo agudo em c=50" do F.5 e um pico genuino no
# plano (c, beta_perp) ou apenas um ponto numa cordilheira c*beta_perp ~ const.
#
# Como rodar: cole esta funcao em cifar100_all_methods_iclr.py, ao lado de
# boost_ablation(), e dispare com a flag TEMP (mesmo padrao da boost_ablation):
#     enable run_boost_betaperp_sweep in a local experimental copy
# Depois REVERTER:
#     remove the temporary local hook after the sweep
#
# Custo: ~1 spine + 9*3 = 27 amostragens em 1 checkpoint (seed 0). Barato,
# porque spine e N0 NAO dependem de c nem de beta_perp (so a curvatura/HVPs
# e o complemento stiff os definem). c entra so em L_perp (prior projetado),
# beta_perp e o multiplicador da escala. Por isso: construir UMA vez, varrer
# so a amostragem.
#
# >>> 3 PONTOS A CONFIRMAR contra o seu codigo real (marcados [CONFIRM]) <<<
#   (1) assinatura de build_trl_prior_from_laplace / como o backbone boost entra
#   (2) como L_perp e montado a partir de (autovalores de curvatura + prior proj.)
#   (3) a funcao de avaliacao que devolve acc/nll/ece/brier
# Tudo isso ja existe na sua boost_ablation(); aqui so reaproveito os mesmos hooks.
# ============================================================================

import os
import csv
import numpy as np
import torch


def boost_betaperp_sweep_2d(cfg, model, laplace_ll, train_loader, val_loader,
                            device="cuda", checkpoint_seed=0):
    """Sweep conjunto (c, beta_perp) num unico checkpoint MAP.

    Reutiliza a maquinaria da boost_ablation(): construcao de spine, base
    transversa N0, montagem do prior block-isotropic e avaliacao. A unica
    diferenca conceitual e que aqui o spine/N0 sao construidos UMA vez e
    reutilizados em todas as celulas da grade.
    """

    # ---- grade geometrica (cai em cima da hipotese de cordilheira) ----------
    C_GRID = [25, 50, 100]
    BETA_GRID = [2.0, 4.0, 8.0]
    N_SAMPLINGS = 3          # "3 amostragens" por celula (igual boost_ablation)
    S = cfg.trl.n_samples    # mesmo budget de amostras posteriores (25)

    # base_val do checkpoint (marglik da Laplace last-layer). seed0 ~ 4.9016.
    base_val = float(laplace_ll.prior_precision.mean().item())  # [CONFIRM]
    print(f"[sweep2d] checkpoint seed={checkpoint_seed}  base_val={base_val:.4f}")

    # ========================================================================
    # 1) CONSTRUIR O TUBO UMA VEZ  (independe de c e de beta_perp)
    #    - spine predictor-corrector
    #    - base transversa N0 (HVPs/Lanczos) e autovalores de curvatura
    #    Cole aqui EXATAMENTE o mesmo bloco de construcao que a sua
    #    boost_ablation() usa ANTES de aplicar o boost; ele nao depende de c.
    # ========================================================================
    # spine, N0, curv_eigs = build_trl_spine_and_basis(cfg, model, ...)   # [CONFIRM]
    spine, N0, curv_eigs = _build_trl_once(cfg, model, train_loader, device)

    # ========================================================================
    # 2) VARRER (c, beta_perp) SO NA AMOSTRAGEM
    # ========================================================================
    results = {}  # (c, beta) -> dict de metricas (mean/std sobre N_SAMPLINGS)

    for c in C_GRID:
        # prior block-isotropic: head = base_val ; backbone = max(c*base_val, 5)
        backbone_prec = max(c * base_val, 5.0)
        # prior_vec = build_trl_prior_from_laplace(model, head_prec=base_val,
        #                                          backbone_prec=backbone_prec,
        #                                          floor=5.0)               # [CONFIRM]
        prior_vec = _build_block_prior(model, base_val, backbone_prec, device)

        # L_perp depende de c (via prior projetado no subespaco), nao de beta.
        # Monte o MESMO L_perp que o sampler usa, projetando prior_vec em N0:
        #   prec_perp = curv_eigs + <prior projetado em N0>
        #   L_perp    = 1/sqrt(prec_perp)        (inverse-sqrt, diagonal)
        L_perp = _build_Lperp(curv_eigs, prior_vec, N0)                    # [CONFIRM]

        for beta in BETA_GRID:
            accs, nlls, eces, briers = [], [], [], []
            for r in range(N_SAMPLINGS):
                torch.manual_seed(1000 * checkpoint_seed + 100 * int(c) + r)
                # amostrar S redes: theta = gamma_t + N0 @ (beta * L_perp * z)
                preds = _sample_predict(cfg, model, spine, N0, L_perp,
                                        beta_perp=beta, S=S,
                                        loader=val_loader, device=device)  # [CONFIRM]
                # FixBN ja deve estar embutido em _sample_predict (mesmo protocolo)
                m = _eval_metrics(preds, val_loader)                       # [CONFIRM]
                accs.append(m["acc"]); nlls.append(m["nll"])
                eces.append(m["ece"]); briers.append(m["brier"])

            results[(c, beta)] = dict(
                product=c * beta,
                acc=(np.mean(accs), np.std(accs)),
                nll=(np.mean(nlls), np.std(nlls)),
                ece=(np.mean(eces), np.std(eces)),
                brier=(np.mean(briers), np.std(briers)),
            )
            mu = results[(c, beta)]
            print(f"[sweep2d] c={c:>4}  beta={beta:>4}  prod={c*beta:>5.0f} | "
                  f"ECE {mu['ece'][0]:.4f}+/-{mu['ece'][1]:.4f}  "
                  f"NLL {mu['nll'][0]:.4f}  ACC {mu['acc'][0]:.4f}")

    # ========================================================================
    # 3) TABELA 2D + ANALISE DE CORDILHEIRA
    # ========================================================================
    _print_grid(results, C_GRID, BETA_GRID, metric="ece")
    _ridge_analysis(results)

    # CSV pra replotar / colar no paper
    out_csv = os.path.join(cfg.out_dir, f"boost_betaperp_sweep_seed{checkpoint_seed}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["c", "beta_perp", "product",
                    "acc_mean", "acc_std", "nll_mean", "nll_std",
                    "ece_mean", "ece_std", "brier_mean", "brier_std"])
        for (c, beta), m in results.items():
            w.writerow([c, beta, m["product"],
                        m["acc"][0], m["acc"][1], m["nll"][0], m["nll"][1],
                        m["ece"][0], m["ece"][1], m["brier"][0], m["brier"][1]])
    print(f"[sweep2d] CSV salvo em {out_csv}")
    return results


# ----------------------------------------------------------------------------
# ANALISE: cordilheira vs pico conjunto
# ----------------------------------------------------------------------------
def _ridge_analysis(results):
    """Agrupa celulas por produto c*beta. Se a dispersao DENTRO de cada grupo
    de produto-igual << dispersao ENTRE produtos, o ECE e governado pelo
    produto -> cordilheira (50 nao e especial). Se nao, ha pico conjunto."""
    from collections import defaultdict
    by_prod = defaultdict(list)
    for (c, beta), m in results.items():
        by_prod[m["product"]].append(((c, beta), m["ece"][0]))

    print("\n[sweep2d] === analise de cordilheira (agrupado por c*beta) ===")
    within_spreads = []
    prod_means = []
    for prod in sorted(by_prod):
        cells = by_prod[prod]
        eces = [e for _, e in cells]
        prod_means.append(np.mean(eces))
        if len(cells) > 1:
            spread = max(eces) - min(eces)
            within_spreads.append(spread)
            cell_str = "  ".join(f"{c,b}->ECE {e:.4f}" for (c, b), e in cells)
            print(f"  produto={prod:>5.0f} (n={len(cells)}): spread={spread:.4f} | {cell_str}")
        else:
            (c, b), e = cells[0]
            print(f"  produto={prod:>5.0f} (n=1): {c,b}->ECE {e:.4f}")

    across_spread = (max(prod_means) - min(prod_means)) if prod_means else 0.0
    mean_within = np.mean(within_spreads) if within_spreads else float("nan")
    print(f"\n  spread MEDIO dentro de produto-igual : {mean_within:.4f}")
    print(f"  spread ENTRE produtos                : {across_spread:.4f}")
    if within_spreads and mean_within < 0.3 * across_spread:
        print("  --> CORDILHEIRA: ECE segue o produto c*beta; 50 nao e um pico isolado.")
        print("      Acao: recuar o texto do F.5 de 'otimo agudo em c=50' para")
        print("      'estrutura de dois blocos necessaria + 50 e operating point robusto'.")
    else:
        print("  --> PICO CONJUNTO provavel: (50,4) nao e reproduzido por produto igual.")
        print("      Acao: o F.5 pode AFIRMAR otimo no plano (c,beta); paper fica mais forte.")
        print("      (Cheque em especial se (25,8) colapsou: se sim, refuta multiplicatividade")
        print("       pura e e a MELHOR evidencia a favor da estrutura de dois blocos.)")


def _print_grid(results, C_GRID, BETA_GRID, metric="ece"):
    print(f"\n[sweep2d] === grade 2D ({metric.upper()}, menor=melhor) ===")
    header = "   c \\ b |" + "".join(f"  beta={b:<5}" for b in BETA_GRID)
    print(header)
    print("   " + "-" * (len(header) - 3))
    for c in C_GRID:
        row = f"   {c:>4} |"
        for b in BETA_GRID:
            mu, sd = results[(c, b)][metric]
            row += f"  {mu:.4f}    "
        print(row)


# ----------------------------------------------------------------------------
# STUBS / hooks — substituir pelos do seu codigo (mesmos da boost_ablation)
# ----------------------------------------------------------------------------
def _build_trl_once(cfg, model, train_loader, device):
    """[CONFIRM] Cole o bloco de construcao de spine + base transversa N0 que a
    boost_ablation() ja usa (curvatura/HVPs/Lanczos + complemento stiff +
    predictor-corrector). Deve devolver:
        spine     : lista de checkpoints {gamma_t}
        N0        : base transversa rank-k_perp (K x k_perp)  [transportada ao longo do spine]
        curv_eigs : autovalores de curvatura nas k_perp direcoes (vetor k_perp)
    Nada aqui depende de c nem de beta_perp."""
    raise NotImplementedError("Cole o bloco de construcao do tubo da sua boost_ablation().")


def _build_block_prior(model, head_prec, backbone_prec, device):
    """[CONFIRM] Mesmo build_trl_prior_from_laplace: head (linear./fc.) recebe
    head_prec; demais parametros recebem backbone_prec; floor ja aplicado em
    backbone_prec = max(c*base_val, 5). Devolve vetor de precisao por parametro."""
    raise NotImplementedError("Use seu build_trl_prior_from_laplace(...).")


def _build_Lperp(curv_eigs, prior_vec, N0):
    """[CONFIRM] Mesmo fator de amostragem transversa do sampler:
        prec_perp = curv_eigs + (prior_vec projetado em N0)
        L_perp    = 1/sqrt(prec_perp)   (diagonal, inverse-sqrt)
    Importante: e AQUI que c entra (via prior_vec). beta_perp NAO entra aqui;
    ele multiplica na amostragem (_sample_predict)."""
    raise NotImplementedError("Monte L_perp como no sampler de Eq. (6).")


def _sample_predict(cfg, model, spine, N0, L_perp, beta_perp, S, loader, device):
    """[CONFIRM] Mesmo sampler de Eq.(6): por amostra, sorteia checkpoint t do
    spine, z~N(0,I_kperp), theta = gamma_t + N0 @ (beta_perp * L_perp * z),
    FixBN no mesmo protocolo, forward. Devolve probabilidades preditivas medias."""
    raise NotImplementedError("Use o mesmo caminho de amostragem+FixBN da boost_ablation().")


def _eval_metrics(preds, loader):
    """[CONFIRM] Mesma funcao de avaliacao do paper: acc / nll / ece / brier."""
    raise NotImplementedError("Use sua funcao de metricas (acc/nll/ece/brier).")

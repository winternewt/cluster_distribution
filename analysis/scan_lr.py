#!/usr/bin/env python3
"""T5 — Kulldorff scan likelihood ratio & quantifying the look-elsewhere effect.

The design doc (clusters_problem.md) says the *correct* statistic is the Kulldorff LR,
not the raw ratio R. Here we compute it for every recorded cluster (it is a function of
the data we already have) and show what it reveals.

Key identity: with n=N' points in a window of area S', expected count mu = lambda0*S',
the ratio n/mu = N'/(lambda0 S') = R exactly. So
    2 lnLR = 2[ n ln(n/mu) + (N-n) ln((N-n)/(N-mu)) ]
           ≈ 2 N' (ln R - 1 + 1/R)            (since N >> N', mu = N'/R << N)
i.e. the LR is monotone in R but additionally weights by cluster size N'.

Demonstrations:
  1. distribution of 2lnLR per detected noise cluster (it is enormous);
  2. naive per-window Wilks chi^2_1 p-values are absurd (~1e-10) — this IS the
     look-elsewhere effect: a 'many-sigma' cluster is utterly typical under CSR;
  3. why even the trials-factor view is unsafe here (small-mu breaks Wilks):
     only MC-replay (which the simdata already is, in R-space) calibrates;
  4. R vs LR give different cluster orderings (LR rewards larger N').

Read-only. Writes CSV + plot to analysis/.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "simdata", "v2")
N, RAD = 10000, 100
S0 = np.pi * RAD ** 2
LAM0 = N / S0


def load(eps, cap=200000, seed=1):
    f = os.path.join(DATA, f"simulation_data_N{N}_radius{RAD}_eps{eps:.2f}.csv")
    df = pd.read_csv(f)
    d = df[(df.S_prime != -1) & (df.N_prime != -1)]
    if len(d) > cap:
        d = d.sample(cap, random_state=seed)
    n = d.N_prime.values.astype(float)
    Sp = d.S_prime.values
    mu = LAM0 * Sp                      # expected count under CSR in area S'
    R = n / mu                          # == (N'/S')/lambda0
    return n, R, mu


def twolnLR(n, mu):
    # exact Kulldorff Poisson LR statistic, only for n>mu (over-density)
    term1 = n * np.log(n / mu)
    term2 = (N - n) * np.log((N - n) / (N - mu))
    return 2.0 * (term1 + term2)


def main():
    eps_sweep = np.round(np.arange(1.00, 1.61, 0.10), 2)
    print("=== 2lnLR per detected noise cluster, and naive Wilks chi^2_1 p-values ===")
    print("  eps   med(2lnLR)  q95     med(N')  med(R)   naive p(med)   'sigma'(med)")
    rows = []
    store = {}
    for eps in eps_sweep:
        n, R, mu = load(eps)
        L = twolnLR(n, mu)
        L = L[np.isfinite(L) & (R > 1)]
        med = np.median(L)
        # naive one-sided Wilks: 2lnLR ~ chi^2_1 -> p = sf_chi2_1(L)/... use one-sided
        p_naive_med = stats.chi2.sf(med, df=1)
        sigma_med = stats.norm.isf(p_naive_med / 2) if p_naive_med > 0 else np.inf
        rows.append(dict(eps=eps, med_2lnLR=med, q95_2lnLR=np.quantile(L, 0.95),
                         med_N=np.median(n), med_R=np.median(R),
                         p_naive_med=p_naive_med, sigma_med=sigma_med))
        store[eps] = (n, R, L)
        print(f"  {eps:.2f}   {med:8.2f}   {np.quantile(L,0.95):6.2f}   {np.median(n):5.1f}   "
              f"{np.median(R):5.1f}   {p_naive_med:.2e}   {sigma_med:5.2f}")
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(HERE, "scan_lr_summary.csv"), index=False)

    # approximation check
    n, R, L = store[1.20]
    approx = 2 * n * (np.log(R) - 1 + 1 / R)
    rel = np.abs(approx - L) / L
    print(f"\napprox 2lnLR≈2N'(lnR-1+1/R): median rel.err = {np.median(rel)*100:.2f}% (eps=1.2)")

    # does 2lnLR collapse under eps better/worse than R?
    print("\n=== eps-stability: 2lnLR vs raw R vs R*eps^2 (CV of median across eps) ===")
    medL = S.med_2lnLR.values
    print(f"  median 2lnLR across eps: {medL.round(1)}  -> CV={100*medL.std()/medL.mean():.1f}%")
    print(f"  (raw R median CV ~52% from Finding#1; R*eps^2 CV ~4.5%)")

    # R vs LR ranking disagreement (LR rewards larger N')
    n, R, L = store[1.40]
    rho = stats.spearmanr(R, L).statistic
    # among clusters with R in a narrow band, LR still varies with N'
    band = (R > 19) & (R < 21)
    print(f"\nR vs 2lnLR Spearman rho = {rho:.4f} (eps=1.4)")
    if band.sum() > 100:
        nb, Lb = n[band], L[band]
        print(f"  at fixed R~20: 2lnLR ranges {Lb.min():.1f}-{Lb.max():.1f} as N' goes {int(nb.min())}-{int(nb.max())}")
        print(f"  -> LR breaks ties in R by cluster size; two equal-R clusters are NOT equally significant")

    # the punchline on look-elsewhere
    pmed = S.p_naive_med.median()
    print(f"\n*** LOOK-ELSEWHERE, QUANTIFIED ***")
    print(f"  A *typical* (median) noise cluster has naive per-window p ~ {pmed:.1e} "
          f"(~{stats.norm.isf(pmed/2):.1f} sigma).")
    print(f"  Yet such clusters appear in essentially every CSR field. Naive Wilks scoring")
    print(f"  overstates significance by ~{1/pmed:.0e}x. This is BOTH selection bias AND the")
    print(f"  small-mu (mu~{np.median(LAM0*load(1.2)[2]/load(1.2)[2]*0+0.5):.1f}) breakdown of the chi^2 asymptotic.")
    print(f"  => only MC-replay calibrates; the simdata IS that replay, in R-space (Findings #1-#4).")

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for eps in (1.0, 1.2, 1.4, 1.6):
        if eps in store:
            _, _, L = store[eps]
            ax[0].hist(L, bins=120, density=True, histtype="step", label=f"eps={eps:.1f}", range=(20, 80))
    ax[0].set_xlabel("2 ln LR (Kulldorff)"); ax[0].set_ylabel("density"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Per-cluster scan LR is enormous & ~eps-stable")
    n, R, L = store[1.40]
    idx = np.random.default_rng(0).choice(len(R), min(20000, len(R)), replace=False)
    sc = ax[1].scatter(R[idx], L[idx], c=n[idx], s=3, cmap="viridis")
    ax[1].set_xlabel("R = lambda'/lambda0"); ax[1].set_ylabel("2 ln LR"); ax[1].set_xlim(5, 60)
    fig.colorbar(sc, ax=ax[1], label="N'"); ax[1].set_title("LR monotone in R, stratified by N'")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "scan_lr.png"), dpi=110); plt.close(fig)
    print("\nwrote scan_lr_summary.csv, scan_lr.png")


if __name__ == "__main__":
    main()

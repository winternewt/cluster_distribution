#!/usr/bin/env python3
"""T4 — eps-independent rare-event scorer for DBSCAN density clusters under CSR.

Finding #1 collapses the eps-dependence: R(eps) = scale(eps)*X with X eps-invariant.
So an eps-independent score is immediate — map an observed cluster to the master
variable and read its survival:

    R_tilde = R_obs * eps^2            (collapse variable; ~5% residual scale drift)
    p(R_obs, eps) = SF_master(R_tilde)
    z_equiv      = Phi^{-1}(1 - p)

We build SF_master four ways and compare:
  1. EMPIRICAL survival of pooled R_tilde   (gold standard, body+moderate tail)
  2. INVERSE-GAMMA fit to pooled R_tilde     (mechanism-based, Findings #2/#3)
  3. BETA-PRIME fit to pooled R_tilde        (phenomenological)
  4. legacy per-eps mixture & CONSERVATIVE ENVELOPE from results/regular_fit.csv
     (what the old handoffs were chasing — shown here to be unnecessary)

Read-only on simdata/v2 and results/. Writes scorer_table.csv + plots + a small JSON
of the master fit to analysis/.
"""
import os, json, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.simv2_data import load_raw_df

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
N, RAD = 10000, 100
LAM0 = N / (np.pi * RAD ** 2)
RCAP = 10 / (0.5 * LAM0)          # 62.83 raw-R cap

def load_R(eps, cap=50000, seed=0):
    d = load_raw_df(eps)
    if d is None:
        return None
    R = (d.N_prime.values / d.S_prime.values) / LAM0
    if len(R) > cap:
        R = np.random.default_rng(seed).choice(R, cap, replace=False)
    return R


def main():
    # pool R_tilde = R*eps^2 across a representative eps sweep (train),
    # hold out a couple of eps for validation
    train_eps = np.round(np.arange(1.00, 1.61, 0.05), 2)
    val_eps = [1.10, 1.40, 1.55]
    Rt_train = []
    for eps in train_eps:
        R = load_R(eps, cap=40000, seed=int(eps*100))
        if R is not None:
            Rt_train.append(R * eps ** 2)
    Rt = np.concatenate(Rt_train)
    print(f"pooled R_tilde: n={len(Rt)}, from {len(Rt_train)} eps in [{train_eps[0]},{train_eps[-1]}]")
    print(f"  R_tilde quantiles: " + ", ".join(f"q{q}={np.quantile(Rt,q/100):.2f}" for q in (50,90,95,99,99.9)))

    # --- master fits ---
    ig = stats.invgamma.fit(Rt, floc=0)             # mechanism-based
    bp = stats.betaprime.fit(Rt)                     # phenomenological
    ks_ig = stats.kstest(Rt, "invgamma", args=ig)[0]
    ks_bp = stats.kstest(Rt, "betaprime", args=bp)[0]
    print(f"\nmaster fits on R_tilde:")
    print(f"  inverse-gamma: shape={ig[0]:.3f}, scale={ig[2]:.3f}   KS={ks_ig:.4f}")
    print(f"  beta-prime:    a={bp[0]:.2f} b={bp[1]:.2f} loc={bp[2]:.2f} scale={bp[3]:.2f}   KS={ks_bp:.4f}")

    # empirical survival (sorted)
    xs_emp = np.sort(Rt)
    sf_emp = 1.0 - np.arange(1, len(xs_emp)+1)/len(xs_emp)

    def sf_empirical(r):
        return np.interp(r, xs_emp, sf_emp, left=1.0, right=sf_emp[-1])

    # --- VALIDATION: score held-out eps and check uniformity of p-values ---
    print("\n=== validation: does R_tilde collapse let one master SF score every eps? ===")
    print("  eps   median p (should ~0.5)   frac p<0.05 (should ~0.05)   KS(R~_eps vs master)")
    for eps in val_eps:
        R = load_R(eps, cap=40000, seed=999)
        if R is None: continue
        rt = R * eps**2
        p = sf_empirical(rt)
        ks = stats.kstest(rt, "invgamma", args=ig)[0]
        print(f"  {eps:.2f}   {np.median(p):.3f}                {np.mean(p<0.05):.3f}                       {ks:.4f}")

    # --- legacy mixture & conservative envelope from results/regular_fit.csv (raw-R space) ---
    print("\n=== legacy per-eps Beta-Prime mixture vs conservative envelope (raw R) ===")
    rf = pd.read_csv(os.path.join(RES, "regular_fit.csv"))
    def sf_mix(r):  # uniform-eps mixture survival in RAW R units
        s = np.zeros_like(np.atleast_1d(r), float)
        for _, q in rf.iterrows():
            s += stats.betaprime.sf(r, q['a'], q['b'], loc=q['loc'], scale=q['scale'])
        return s/len(rf)
    def sf_env(r):  # conservative envelope (max tail prob over eps)
        S = np.array([stats.betaprime.sf(r, q['a'], q['b'], loc=q['loc'], scale=q['scale']) for _, q in rf.iterrows()])
        return S.max(axis=0)
    for r in (20, 30, 40, 50):
        pm = np.ravel(sf_mix(r))[0]; pe = np.ravel(sf_env(r))[0]
        print(f"  R={r}: p_mix={pm:.2e}  p_env={pe:.2e}")

    # --- scorer table over a grid (report at a reference eps so users see raw-R thresholds) ---
    rows = []
    for r_tilde in np.arange(20, 76, 2.0):
        p_emp = float(sf_empirical(r_tilde))
        p_ig = float(stats.invgamma.sf(r_tilde, *ig))
        z_emp = stats.norm.ppf(1 - np.clip(p_emp, 1e-12, 1-1e-12))
        z_ig = stats.norm.ppf(1 - np.clip(p_ig, 1e-12, 1-1e-12))
        # what raw R this corresponds to at a few eps
        rows.append(dict(R_tilde=r_tilde,
                         R_at_eps1_1=r_tilde/1.10**2, R_at_eps1_3=r_tilde/1.30**2,
                         p_empirical=p_emp, p_invgamma=p_ig, z_empirical=z_emp, z_invgamma=z_ig))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(HERE, "scorer_table.csv"), index=False)

    # save master model
    with open(os.path.join(HERE, "scorer_master.json"), "w") as fh:
        json.dump(dict(collapse_variable="R*eps^2", lambda0=LAM0,
                       invgamma_shape=ig[0], invgamma_loc=ig[1], invgamma_scale=ig[2],
                       ks_invgamma=ks_ig, raw_R_cap=RCAP,
                       note="valid in body; censoring at R_tilde=62.83*eps^2"), fh, indent=2)

    # --- plots: master survival (log) empirical vs invgamma vs beta-prime ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].semilogy(xs_emp, sf_emp, "k-", lw=1.5, label="empirical")
    gx = np.linspace(xs_emp[0], xs_emp[-1], 400)
    ax[0].semilogy(gx, stats.invgamma.sf(gx, *ig), "b--", label=f"inv-gamma (KS={ks_ig:.3f})")
    ax[0].semilogy(gx, stats.betaprime.sf(gx, *bp), "r:", label=f"beta-prime (KS={ks_bp:.3f})")
    ax[0].axvline(RCAP, ls=":", c="grey", lw=0.8); ax[0].set_ylim(1e-5, 1)
    ax[0].set_xlabel("R_tilde = R*eps^2"); ax[0].set_ylabel("survival P(>=)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Master survival on the collapse variable")
    # validation overlay: SF of each held-out eps in R_tilde space
    for eps in val_eps:
        R = load_R(eps, cap=40000, seed=7)
        if R is None: continue
        rt = np.sort(R*eps**2); sf = 1-np.arange(1,len(rt)+1)/len(rt)
        ax[1].semilogy(rt, sf, lw=1.2, label=f"eps={eps:.2f}")
    ax[1].semilogy(gx, stats.invgamma.sf(gx, *ig), "k--", lw=1.5, label="master inv-gamma")
    ax[1].set_ylim(1e-5,1); ax[1].set_xlabel("R_tilde"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("Held-out eps survivals collapse onto one master")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "scorer_survival.png"), dpi=110); plt.close(fig)
    print("\nwrote scorer_table.csv, scorer_master.json, scorer_survival.png")


if __name__ == "__main__":
    main()

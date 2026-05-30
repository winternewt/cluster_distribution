#!/usr/bin/env python3
"""T1 — The R*eps^2 collapse law.

Central thesis: the eps-dependence of the density ratio R=(N'/S')/lambda0 is, to
leading order, the trivial geometric area scaling S' ~ eps^2. Rescaling
R_tilde = R*eps^2 should collapse the per-eps distributions onto one master curve.
Equivalently R_tilde = N' / (u*lambda0) with u = S'/eps^2 (rescaled hull area), so
the collapse holds iff the joint law of (N', u) is eps-invariant. Any residual drift
is the genuinely interesting second-order physics (fixed min_samples vs growing eps).

Read-only on simdata/v2. Writes summary CSV + plots to analysis/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "..", "simdata", "v2")
OUT = os.path.dirname(__file__)
N, RAD = 10000, 100
LAM0 = N / (np.pi * RAD ** 2)            # 0.318310
RCAP = 10 / (0.5 * LAM0)                 # 62.83 raw-R censoring cap
QS = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def load(eps):
    f = os.path.join(DATA, f"simulation_data_N{N}_radius{RAD}_eps{eps:.2f}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    d = df[(df.S_prime != -1) & (df.N_prime != -1)]
    if len(d) < 2000:
        return None
    return d


def main():
    eps_sweep = np.round(np.arange(0.95, 2.001, 0.05), 2)
    rows = []
    ecdf_store = {}   # eps -> R array (for plotting a subset)
    for eps in eps_sweep:
        d = load(eps)
        if d is None:
            continue
        Np = d.N_prime.values.astype(float)
        Sp = d.S_prime.values
        R = (Np / Sp) / LAM0
        u = Sp / eps ** 2                      # rescaled area
        Rt = R * eps ** 2                       # rescaled ratio
        frac_cens = np.mean(R > 0.999 * RCAP)   # how much mass sits at the cap
        row = dict(eps=eps, n=len(d), meanN=Np.mean(),
                   meanR=R.mean(), medR=np.median(R), stdR=R.std(),
                   meanRt=Rt.mean(), medRt=np.median(Rt), stdRt=Rt.std(),
                   meanS=Sp.mean(), meanU=u.mean(), medU=np.median(u),
                   frac_cens=frac_cens)
        for q in QS:
            row[f"Rt_q{int(q*100):02d}"] = np.quantile(Rt, q)
            row[f"R_q{int(q*100):02d}"] = np.quantile(R, q)
            row[f"u_q{int(q*100):02d}"] = np.quantile(u, q)
        rows.append(row)
        if eps in (1.00, 1.20, 1.40, 1.60, 1.80, 2.00):
            ecdf_store[eps] = (R, Rt)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "collapse_summary.csv"), index=False)

    # --- residual model: is meanRt drift explained by N' and u drift? ---
    # naive predictor: median(Rt) ~ medN / (medU * lambda0); but use mean of N'/u directly.
    print("=== R*eps^2 collapse: per-eps summary ===")
    cols = ["eps", "n", "meanN", "meanU", "medRt", "Rt_q05", "Rt_q50", "Rt_q95", "Rt_q99", "frac_cens"]
    print(S[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # collapse quality: coefficient of variation of each Rt quantile across eps,
    # restricted to the body (eps where censoring < 1%).
    body = S[S.frac_cens < 0.01]
    print("\n=== collapse quality (CV across eps, body only, frac_cens<1%) ===")
    print(f"   eps range in body: {body.eps.min():.2f}-{body.eps.max():.2f}  ({len(body)} eps)")
    for q in QS:
        rt = body[f"Rt_q{int(q*100):02d}"]
        r = body[f"R_q{int(q*100):02d}"]
        print(f"   q{int(q*100):02d}: R~ CV={rt.std()/rt.mean()*100:5.2f}%  (raw R CV={r.std()/r.mean()*100:5.2f}%)")

    # rescaled-area u drift (the residual source)
    print("\n=== rescaled area u=S'/eps^2 drift (residual source) ===")
    print(f"   meanU: {S.meanU.iloc[0]:.4f} (eps={S.eps.iloc[0]:.2f}) -> "
          f"{S.meanU.iloc[-1]:.4f} (eps={S.eps.iloc[-1]:.2f}); "
          f"rise {100*(S.meanU.iloc[-1]/S.meanU.iloc[0]-1):.1f}%")
    print(f"   meanN: {S.meanN.iloc[0]:.3f} -> {S.meanN.iloc[-1]:.3f}; "
          f"rise {100*(S.meanN.iloc[-1]/S.meanN.iloc[0]-1):.1f}%")

    # leading-order constant C = meanR * eps^2 (should be ~const)
    print(f"\n=== leading-order law R ~ C/eps^2 : C = mean(R)*eps^2 ===")
    for _, rr in S.iterrows():
        print(f"   eps={rr.eps:.2f}  C={rr.meanRt:.3f}")

    # --- plots ---
    # 1) collapse plot: ECDF of R_tilde overlaid (body) vs raw R
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for eps, (R, Rt) in sorted(ecdf_store.items()):
        xs = np.sort(R); ys = np.arange(1, len(xs)+1)/len(xs)
        ax[0].plot(xs, ys, lw=1.2, label=f"eps={eps:.2f}")
        xt = np.sort(Rt); yt = np.arange(1, len(xt)+1)/len(xt)
        ax[1].plot(xt, yt, lw=1.2, label=f"eps={eps:.2f}")
    ax[0].set_title("raw R — strong eps drift"); ax[0].set_xlabel("R=lambda'/lambda0"); ax[0].set_ylabel("ECDF")
    ax[1].set_title("R~=R*eps^2 — collapse"); ax[1].set_xlabel("R~ = R*eps^2")
    ax[1].axvline(RCAP*1.0**2, ls=":", c="grey", lw=0.8)  # cap at eps=1
    for a in ax: a.legend(fontsize=8); a.grid(alpha=0.3); a.set_xlim(0, 80)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "collapse_ecdf.png"), dpi=110); plt.close(fig)

    # 2) master curve: Rt quantiles vs eps (flat = collapse) and u drift
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for q in [0.05, 0.5, 0.95]:
        ax[0].plot(S.eps, S[f"Rt_q{int(q*100):02d}"], "o-", ms=3, label=f"R~ q{int(q*100)}")
        ax[0].plot(S.eps, S[f"R_q{int(q*100):02d}"], "s--", ms=3, alpha=0.5, label=f"raw R q{int(q*100)}")
    ax[0].set_xlabel("eps"); ax[0].set_ylabel("quantile"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title("R~ quantiles flat vs eps (collapse); raw R steeply drifts")
    ax[1].plot(S.eps, S.meanU, "o-", label="mean u=S'/eps^2")
    ax[1].plot(S.eps, S.meanN/10, "s-", label="mean N'/10")
    ax[1].set_xlabel("eps"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("residual sources: rescaled area & N' drift")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "collapse_master.png"), dpi=110); plt.close(fig)
    print("\nwrote collapse_summary.csv, collapse_ecdf.png, collapse_master.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T2 — Beta-Prime as an N'-mixture of inverse-gamma area laws.

Mechanism hypothesis:
  R = (N'/S')/lambda0.  Condition on N'=n:  R|n = n/(lambda0 * S').
  If the hull area S'|n ~ Gamma(k(n), theta(n)), then R|n ~ scaled-inverse-Gamma(k(n)).
  Since N' is dominated by n=10 (~68%), the marginal R ~ inverse-Gamma(shape ~ k(10)).
  And betaprime(a,b) with large a -> scaled inverse-Gamma(b). So the fitted b should
  equal k(10) (the Gamma shape of the 10-point hull area). This DERIVES the b~10 result
  and explains beta-prime as the envelope of an N'-mixture of inverse-gammas.

Read-only on simdata/v2. Writes CSV + plot to analysis/.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.simv2_data import load_raw_df

OUT = os.path.dirname(__file__)
N, RAD = 10000, 100
LAM0 = N / (np.pi * RAD ** 2)


def main():
    for eps in (1.00, 1.20, 1.40):
        d = load_raw_df(eps)
        if d is None:
            continue
        d = d.copy()
        d["R"] = (d.N_prime / d.S_prime) / LAM0
        print(f"\n================ eps = {eps:.2f}   (n_clusters={len(d)}) ================")
        # marginal Beta-Prime fit for reference (free loc)
        bp = stats.betaprime.fit(d.R.values)
        bp_ks = stats.kstest(d.R.values, "betaprime", args=bp)[0]
        print(f"marginal Beta-Prime fit: a={bp[0]:.2f} b={bp[1]:.2f} loc={bp[2]:.3f} scale={bp[3]:.3f}  KS={bp_ks:.4f}")

        # per-N' conditional fits
        print(f"\n  n   P(N'=n)   <S'|n>   Gamma_k(S'|n) Gamma_th  invG_k(R|n)  KS(R|n,invG)")
        recs = []
        tot = len(d)
        for n in range(10, 21):
            sub = d[d.N_prime == n]
            if len(sub) < 500:
                continue
            Sp = sub.S_prime.values
            R = sub.R.values
            # Gamma fit to hull area S'|n (floc=0)
            gk, _, gth = stats.gamma.fit(Sp, floc=0)
            # inverse-gamma fit to R|n (floc=0): R|n should be scaled-invgamma(k)
            ik, _, isc = stats.invgamma.fit(R, floc=0)
            ks_ig = stats.kstest(R, "invgamma", args=(ik, 0, isc))[0]
            w = len(sub) / tot
            recs.append(dict(eps=eps, n=n, w=w, meanS=Sp.mean(), gamma_k=gk,
                             gamma_th=gth, invg_k=ik, invg_scale=isc, ks_invg=ks_ig))
            print(f" {n:3d}   {w:6.4f}   {Sp.mean():6.3f}    {gk:7.3f}     {gth:6.4f}    {ik:7.3f}     {ks_ig:.4f}")
        rec = pd.DataFrame(recs)
        rec.to_csv(os.path.join(OUT, f"conditional_eps{eps:.2f}.csv"), index=False)

        # KEY TEST: does invgamma shape of R|10 match the fitted marginal beta-prime b?
        k10 = rec[rec.n == 10].invg_k.values
        gk10 = rec[rec.n == 10].gamma_k.values
        if len(k10):
            print(f"\n  KEY: invGamma shape of R|N'=10 = {k10[0]:.3f}   "
                  f"Gamma shape of S'|N'=10 = {gk10[0]:.3f}   marginal Beta-Prime b = {bp[1]:.3f}")

        # reconstruct marginal R as the N'-mixture of fitted scaled-invgamma(R|n)
        xs = np.linspace(d.R.quantile(0.001), d.R.quantile(0.999), 400)
        mix_pdf = np.zeros_like(xs)
        for _, r in rec.iterrows():
            mix_pdf += r.w * stats.invgamma.pdf(xs, r.invg_k, 0, r.invg_scale)
        # renormalize weights (we dropped rare n)
        wsum = rec.w.sum()
        mix_pdf /= wsum
        # mixture CDF for KS vs data
        def mix_cdf(x):
            c = np.zeros_like(np.atleast_1d(x), dtype=float)
            for _, r in rec.iterrows():
                c += r.w * stats.invgamma.cdf(x, r.invg_k, 0, r.invg_scale)
            return c / wsum
        ks_mix = stats.kstest(d.R.values, mix_cdf)[0]
        print(f"  N'-mixture-of-invgamma reconstruction: KS vs data = {ks_mix:.4f}  "
              f"(direct Beta-Prime KS = {bp_ks:.4f}; weight coverage = {wsum:.3f})")

        if eps == 1.20:  # one diagnostic plot
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(d.R.values, bins=200, density=True, alpha=0.35, color="grey",
                    label=f"data (eps={eps})", range=(xs[0], xs[-1]))
            ax.plot(xs, mix_pdf, "b-", lw=2, label="N'-mixture of inv-gamma(R|n)")
            ax.plot(xs, stats.betaprime.pdf(xs, *bp), "r--", lw=2, label="direct Beta-Prime")
            # show the n=10 component alone
            r10 = rec[rec.n == 10].iloc[0]
            ax.plot(xs, r10.w / wsum * stats.invgamma.pdf(xs, r10.invg_k, 0, r10.invg_scale),
                    "g:", lw=1.5, label=f"n=10 component (w={r10.w:.2f})")
            ax.set_xlabel("R = lambda'/lambda0"); ax.set_ylabel("density"); ax.legend()
            ax.set_title("Beta-Prime emerges as an N'-mixture of inverse-gamma area laws")
            fig.tight_layout(); fig.savefig(os.path.join(OUT, "conditional_mixture.png"), dpi=110)
            plt.close(fig)
    print("\nwrote conditional_eps*.csv, conditional_mixture.png")


if __name__ == "__main__":
    main()

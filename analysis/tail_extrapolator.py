#!/usr/bin/env python3
"""Deep-tail extrapolator with confidence bands + an independent mechanistic validator.

Two estimates of the null tail of the cluster statistic, agreeing => confidence:

  (A) EVT / Peaks-Over-Threshold:  fit a Generalized Pareto to exceedances of the
      statistic above a high threshold (captures the curvature of the LDT rate function
      via the GPD shape xi -- NOT a fixed linear slope). Bootstrap confidence bands on
      p(u) and z(u). xi<0 => finite right endpoint (light tail / censoring); xi~0 =>
      exponential (pure large-deviations); xi>0 => polynomial.

  (B) Mechanistic validator (conditional Monte Carlo = importance sampling via the
      derived structure): we KNOW R|N'=n = n/(lambda0 S') with S' = eps^2 * (hull area of
      n uniform points in a disk). So instead of brute-forcing whole CSR fields and waiting
      for a rare dense cluster, we sample the cluster geometry directly (cheap), weight by
      the empirical occupancy P(N'=n), and read the tail. This reaches depths the field-MC
      cannot, with no DBSCAN.

Brute-force data (where it exists) is the third, ground-truth check. Statistic = 2lnLR
(recommended: exponential/LDT tail). Also reports R. eps=1.20 reference.
"""
import os, sys, numpy as np, pandas as pd
from numpy.random import default_rng
from scipy import stats
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.simv2_data import load_raw_df

HERE = os.path.dirname(__file__)
N, RAD, EPS = 10000, 100, 1.20
LAM0 = N/(np.pi*RAD**2)
rng = default_rng(31415)

def twolnLR_from(nv, Sv):
    mu = LAM0*Sv
    return 2*(nv*np.log(nv/mu) + (N-nv)*np.log((N-nv)/(N-mu)))

# ---------- load brute-force ground truth ----------
def load():
    d = load_raw_df(EPS)
    if d is None:
        return None, None, None, None, None
    n = d.N_prime.values.astype(float); S = d.S_prime.values
    occ = d.N_prime.value_counts(normalize=True).sort_index()
    return n, S, (n/S)/LAM0, twolnLR_from(n,S), occ

# ---------- (A) GPD / POT with bootstrap bands ----------
def gpd_tail(stat, q_thresh=0.90, B=300):
    u0 = np.quantile(stat, q_thresh)
    exc = stat[stat>u0] - u0
    zeta = len(exc)/len(stat)               # P(stat>u0)
    xi, _, sig = stats.genpareto.fit(exc, floc=0)
    def sf(u, xi_, sig_, zeta_):
        return zeta_*stats.genpareto.sf(np.asarray(u)-u0, xi_, 0, sig_)
    # bootstrap
    boots=[]
    nstat=len(stat)
    for _ in range(B):
        s = rng.choice(stat, nstat, replace=True)
        e = s[s>u0]-u0
        if len(e)<50: continue
        try:
            xb,_,sb = stats.genpareto.fit(e, floc=0)
            boots.append((xb, sb, len(e)/nstat))
        except Exception: pass
    boots=np.array(boots)
    return dict(u0=u0, xi=xi, sigma=sig, zeta=zeta, sf=sf, boots=boots)

# ---------- (B) mechanistic validator: sample hull areas ----------
def hull_area_unitdisk(n, m):
    """m samples of convex-hull area of n uniform points in the unit disk."""
    out = np.empty(m)
    for i in range(m):
        r = np.sqrt(rng.uniform(size=n)); t = 2*np.pi*rng.uniform(size=n)
        p = np.column_stack((r*np.cos(t), r*np.sin(t)))
        try: out[i] = ConvexHull(p).volume
        except Exception: out[i] = 0.0
    return out

def mechanistic_samples(occ, per_n=400000, nmax=18):
    """Pooled R and 2lnLR samples drawn from the derived geometry, weighted by P(N'=n)."""
    Rs=[]; Ls=[]; ws=[]
    for n in range(10, nmax+1):
        w = float(occ.get(n,0.0))
        if w < 1e-4: continue
        m = max(20000, int(per_n))
        a = hull_area_unitdisk(n, m)                 # unit-disk hull area
        a = a[a>0]
        S = EPS**2 * a                               # actual hull area
        R = n/(LAM0*S)
        L = twolnLR_from(np.full_like(R, n), S)
        Rs.append(R); Ls.append(L); ws.append(np.full(len(R), w/len(R)))
    R=np.concatenate(Rs); L=np.concatenate(Ls); w=np.concatenate(ws); w/=w.sum()
    return R, L, w

def wsurv(x, w, grid):
    order=np.argsort(x); x=x[order]; w=w[order]; cw=np.cumsum(w)
    # survival at grid: 1 - CDF
    idx=np.searchsorted(x, grid, side="right")
    cdf=np.where(idx>0, cw[np.clip(idx-1,0,len(cw)-1)], 0.0)
    return 1-cdf

def z_of(p): return stats.norm.isf(np.clip(p,1e-300,1-1e-12))

def main():
    n,S,R,L,occ = load()
    if n is None:
        print("data unavailable"); return
    print(f"brute-force eps={EPS}: {len(L)} clusters; max 2lnLR={L.max():.1f}  (p_floor~{1/len(L):.1e})")

    # (A) GPD on 2lnLR
    g = gpd_tail(L, q_thresh=0.90, B=300)
    tag = "finite endpoint (light tail/censoring)" if g['xi']<-0.02 else ("exponential" if abs(g['xi'])<=0.02 else "polynomial")
    print(f"\n(A) GPD/POT on 2lnLR: threshold u0={g['u0']:.1f} (zeta={g['zeta']:.3f}); "
          f"xi={g['xi']:+.3f}, sigma={g['sigma']:.2f}  => {tag}")
    if g['xi']<0:
        endpoint=g['u0']-g['sigma']/g['xi']
        print(f"    implied finite max 2lnLR ~ {endpoint:.0f}  (NB: min_area censoring contributes; remove it to probe true tail)")

    # (B) mechanistic
    print("\n(B) mechanistic validator (geometry of n uniform disk points, weighted by P(N'=n)) ...")
    Rm, Lm, wm = mechanistic_samples(occ, per_n=300000)
    print(f"    drew {len(Lm)} geometry samples; mechanistic max 2lnLR={Lm.max():.1f}")

    # ---- table: p(2lnLR>u) and z, three ways + GPD 90% band ----
    grid = np.array([50,55,60,63,70,80,90,100], float)
    sf_gpd = g['sf'](grid, g['xi'], g['sigma'], g['zeta'])
    # bootstrap band
    bb = np.array([g['zeta']*0+ b[2]*stats.genpareto.sf(grid-g['u0'], b[0],0,b[1]) for b in g['boots']])
    lo,hi = np.nanpercentile(bb,5,axis=0), np.nanpercentile(bb,95,axis=0)
    sf_emp = np.array([np.mean(L>u) for u in grid])
    sf_mech= wsurv(Lm, wm, grid)
    print("\n  2lnLR |   p_bruteforce |  p_GPD [90% CI]            |  p_mechanistic |  z_GPD")
    for i,u in enumerate(grid):
        be = f"{sf_emp[i]:.2e}" if sf_emp[i]>0 else "  0 (none)"
        print(f"   {u:4.0f} |   {be:>11} |  {sf_gpd[i]:.2e} [{lo[i]:.1e},{hi[i]:.1e}] |  {sf_mech[i]:.2e}    |  {z_of(sf_gpd[i]):.2f}")

    # ---- validation: fit GPD on u<60 only, predict 60..78 vs brute-force truth ----
    Lcut = L[L<60]
    gv = gpd_tail(np.concatenate([Lcut, L[L>=60]*0+59.999]) if False else L[L<60], q_thresh=0.80, B=50) \
         if False else None
    # simpler holdout: refit threshold inside the kept region, predict beyond
    keep = L[L < 58]
    gh = gpd_tail(keep, q_thresh=0.85, B=80)
    print(f"\n  holdout check (fit GPD on 2lnLR<58 only; predict where brute force still has events):")
    print("   2lnLR | predicted p | brute-force p | ratio")
    for u in (60,66,72,78):
        pp = gh['sf'](u, gh['xi'], gh['sigma'], gh['zeta'])
        pe = np.mean(L>u)
        rr = f"{float(pp)/pe:.2f}" if pe>0 else "—"
        print(f"   {u:4d} |  {float(pp):.2e}  |  {pe:.2e}   | {rr}")

    # save table + plot
    out = pd.DataFrame(dict(twolnLR=grid, p_bruteforce=sf_emp, p_gpd=sf_gpd,
                            p_gpd_lo=lo, p_gpd_hi=hi, p_mechanistic=sf_mech,
                            z_gpd=z_of(sf_gpd)))
    out.to_csv(os.path.join(HERE,"tail_extrapolation_table.csv"), index=False)

    fig,ax=plt.subplots(figsize=(9,6))
    us=np.linspace(40,105,200)
    ax.semilogy(np.sort(L), 1-np.arange(len(L))/len(L), 'k-', lw=1.3, label='brute force (132k clusters)')
    ax.semilogy(us, g['sf'](us,g['xi'],g['sigma'],g['zeta']), 'b-', lw=2, label=f"GPD extrapolation (xi={g['xi']:+.2f})")
    ax.fill_between(us, g['sf'](us,*np.nanpercentile(g['boots'],5,axis=0)[[0,1]],np.nanpercentile(g['boots'],5,axis=0)[2]),
                        g['sf'](us,*np.nanpercentile(g['boots'],95,axis=0)[[0,1]],np.nanpercentile(g['boots'],95,axis=0)[2]),
                    color='b', alpha=0.2, label='GPD 90% band')
    sm=np.argsort(Lm); ax.semilogy(Lm[sm], wsurv(Lm,wm,Lm[sm]), 'r--', lw=1.5, label='mechanistic (geometry)')
    ax.axvline(63, ls=':', c='grey'); ax.text(63.5,1e-7,'min_area cap',rotation=90,fontsize=8)
    ax.set_xlabel('2 ln LR'); ax.set_ylabel('survival P(>=)'); ax.set_ylim(1e-9,1); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Deep-tail extrapolation: GPD (w/ band) vs mechanistic vs brute force')
    fig.tight_layout(); fig.savefig(os.path.join(HERE,'tail_extrapolation.png'),dpi=110); plt.close(fig)
    print("\nwrote tail_extrapolation_table.csv, tail_extrapolation.png")

if __name__ == "__main__":
    main()

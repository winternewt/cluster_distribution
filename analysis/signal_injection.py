#!/usr/bin/env python3
"""Q3 — the actual signal/noise classifier, tested by injection.

Setup mirrors the original 2020 problem: a CSR field plus an optional sub-region S'
carrying extra Poisson points. We run the IDENTICAL DBSCAN pipeline, score every
detected cluster against the CSR null, and ask whether the injected splat is flagged.

Two scores, both p-value -> z_equiv = Phi^{-1}(1-p) against the empirical CSR null:
  z_R   : from the density ratio R = (N'/S')/lambda0     (Findings #1-#4 master)
  z_LR  : from the Kulldorff 2lnLR = 2N'(lnR -1 +1/R)    (size-weighted; Finding #4)

Key expected lesson: DBSCAN noise clusters are TIGHT minimal clusters (N'~10, high R),
so an EXTENDED moderate-overdensity splat has only modest R but large N' -> z_LR >> z_R.
The right classifier for extended signal is LR, not raw R.

Cheap, self-contained. eps=1.2, N=10000, R_dom=100.
"""
import os, json
import numpy as np
from numpy.random import default_rng
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy import stats

N, RDOM, EPS = 10000, 100.0, 1.20
LAM0 = N/(np.pi*RDOM**2); S0 = np.pi*RDOM**2
MIN_S, MIN_N, MIN_A = 10, 10, 0.5

def hull_area(p):
    if len(p) < 3: return 0.0
    try: return ConvexHull(p).volume
    except Exception: return 0.0

def disk(n, R, rng, cx=0.0, cy=0.0):
    rr = R*np.sqrt(rng.uniform(size=n)); th = 2*np.pi*rng.uniform(size=n)
    return np.column_stack((cx+rr*np.cos(th), cy+rr*np.sin(th)))

def clusters_of(pts):
    lab = DBSCAN(eps=EPS, min_samples=MIN_S).fit(pts).labels_
    out = []
    for L in set(lab):
        if L == -1: continue
        cp = pts[lab == L]
        if len(cp) < MIN_N: continue
        S = hull_area(cp)
        if S < MIN_A: continue
        n = len(cp); R = (n/S)/LAM0; mu = LAM0*S
        twolnLR = 2*(n*np.log(n/mu) + (N-n)*np.log((N-n)/(N-mu)))
        out.append((n, S, R, twolnLR, cp[:,0].mean(), cp[:,1].mean()))
    return out

def main():
    rng = default_rng(2026)
    # ---- 1) build CSR null distributions of R and 2lnLR (pure noise) ----
    nullR, nullL = [], []
    for _ in range(6000):
        for (n,S,R,L,cx,cy) in clusters_of(disk(N, RDOM, rng)):
            nullR.append(R); nullL.append(L)
    nullR = np.sort(nullR); nullL = np.sort(nullL)
    Rmax, Lmax = nullR[-1], nullL[-1]
    def z_from(null, v):
        p = 1.0 - np.searchsorted(null, v, side="right")/len(null)
        p = min(max(p, 0.5/len(null)), 1-1e-9)
        return stats.norm.isf(p)
    zcap = stats.norm.isf(0.5/len(nullR))
    print(f"CSR null: {len(nullR)} clusters; R median={np.median(nullR):.1f} max={Rmax:.1f}; "
          f"2lnLR median={np.median(nullL):.1f} max={Lmax:.1f}; z resolvable up to {zcap:.2f} (MC-limited)")

    # ---- 2) inject splats of increasing strength; score the recovered cluster ----
    CX, CY, RHO = 45.0, 0.0, 5.0          # splat: radius 5 (~4*eps, extended), off-centre
    print(f"\nsplat: extended disk radius {RHO} at ({CX},{CY}); background N={N}")
    print(f"{'n_extra':>7} {'ratio':>6} | {'detect%':>7} {'medNp':>6} {'medR':>6} {'med_zR':>6} {'med_zLR':>7} | {'det_zLR>3':>10}")
    for n_extra in (0, 15, 30, 60, 120):
        dens_ratio = 1 + n_extra/(np.pi*RHO**2*LAM0)
        recovered = []   # (N', R, zR, zLR)
        TR = 250
        for _ in range(TR):
            pts = disk(N, RDOM, rng)
            if n_extra > 0:
                pts = np.vstack((pts, disk(n_extra, RHO, rng, CX, CY)))
            best = None
            for (n,S,R,L,cx,cy) in clusters_of(pts):
                if np.hypot(cx-CX, cy-CY) < RHO + EPS:    # cluster overlapping the splat region
                    if best is None or n > best[0]:
                        best = (n, R, z_from(nullR,R), z_from(nullL,L), R > Rmax, L > Lmax)
            recovered.append(best)
        det = [r for r in recovered if r is not None]
        if not det:
            print(f"{n_extra:>7} {dens_ratio:>6.2f} |  (no cluster recovered at splat site)")
            continue
        arr = np.array(det, dtype=float)
        frac_det = len(det)/TR
        # ceiling-free metric: fraction of recovered clusters exceeding the ENTIRE null
        exR = 100*np.mean(arr[:,4]); exL = 100*np.mean(arr[:,5])
        print(f"{n_extra:>7} {dens_ratio:>6.2f} | {100*frac_det:6.1f}% {np.median(arr[:,0]):>6.0f} "
              f"{np.median(arr[:,1]):>6.1f} {np.median(arr[:,2]):>6.2f} {np.median(arr[:,3]):>7.2f} | "
              f"R>null:{exR:4.0f}%  LR>null:{exL:4.0f}%")
    print("\nRead: z_R (density-ratio score) vs z_LR (size-weighted LR). For an EXTENDED splat,")
    print("R stays in/below the noise body (z_R<=0) but N' is huge -> LR exceeds the ENTIRE")
    print("null while R never does. z is MC-capped at the null size; 'LR>null%' is ceiling-free.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Q2 test: the collapse variable is mu_eps = lambda0*pi*eps^2 = (N/R_dom^2)*eps^2,
NOT eps^2. 'eps^2' only looks clean because the production config has N=R_dom^2=1e4.

Prediction: the dimensionless R = (N'/S')/lambda0 distribution depends on N, R_dom, eps
ONLY through mu_eps. So configs with DIFFERENT (N, R_dom, eps) but the SAME mu_eps must
give the SAME R distribution (in the bulk; the absolute min_area floor differs, so the
extreme high-R tail can differ slightly).

We MC three configs all at mu_eps = 1.44 and compare R quantiles. Cheap, self-contained.
"""
import numpy as np
from numpy.random import default_rng
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

MIN_SAMPLES, MIN_N, MIN_AREA = 10, 10, 0.5

def hull_area(p):
    if len(p) < 3: return 0.0
    try: return ConvexHull(p).volume
    except Exception: return 0.0

def run(N, Rdom, eps, iters, rng):
    lam0 = N/(np.pi*Rdom**2)
    Rs = []
    for _ in range(iters):
        rr = Rdom*np.sqrt(rng.uniform(size=N)); th = 2*np.pi*rng.uniform(size=N)
        pts = np.column_stack((rr*np.cos(th), rr*np.sin(th)))
        lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(pts).labels_
        for L in set(lab):
            if L == -1: continue
            cp = pts[lab == L]
            if len(cp) < MIN_N: continue
            S = hull_area(cp)
            if S < MIN_AREA: continue
            Rs.append((len(cp)/S)/lam0)
    return np.array(Rs)

def main():
    rng = default_rng(7)
    mu = 1.44
    configs = [  # (N, Rdom, eps) all with (N/Rdom^2)*eps^2 = mu
        (10000, 100.0, np.sqrt(mu*100.0**2/10000)),   # eps=1.20, lambda0=0.318
        (5000,  100.0, np.sqrt(mu*100.0**2/5000)),    # eps=1.697, lambda0=0.159
        (20000, 100.0, np.sqrt(mu*100.0**2/20000)),   # eps=0.849, lambda0=0.637
    ]
    qs = [10,25,50,75,90,95]
    print(f"mu_eps = (N/Rdom^2)*eps^2 = {mu} for all three; lambda0 differs 4x.")
    print("If R depends only on mu_eps, the R quantiles below should MATCH.\n")
    print(f"{'N':>6} {'eps':>6} {'lam0':>6} {'nClust':>7}  " + "  ".join(f"q{q}" for q in qs))
    res = {}
    for (N, Rdom, eps) in configs:
        R = run(N, Rdom, eps, iters=4000, rng=rng)
        res[N] = R
        lam0 = N/(np.pi*Rdom**2)
        print(f"{N:>6} {eps:>6.3f} {lam0:>6.3f} {len(R):>7}  " +
              "  ".join(f"{np.quantile(R,q/100):5.2f}" for q in qs))
    # pairwise max relative quantile difference in the bulk (q10-q90)
    base = res[10000]
    print("\nbulk (q10-q90) max |rel diff| vs N=10000 config:")
    for N in (5000, 20000):
        diffs = [abs(np.quantile(res[N],q/100)/np.quantile(base,q/100)-1) for q in (10,25,50,75,90)]
        print(f"  N={N}: {max(diffs)*100:.1f}%   (small => R depends on mu_eps, not eps^2 or N alone)")

if __name__ == "__main__":
    main()

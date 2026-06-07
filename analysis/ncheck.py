#!/usr/bin/env python3
"""Does the master null depend on N at fixed lambda0?

Prediction (RCA follow-up): the per-cluster law of Rt = R*eps^alpha(eps) is a LOCAL
object — at fixed lambda0 = 1/pi it should be independent of the field size N.
Specifically: hard floor Rt >= ~10.9 (= min_samples * 9-gon correction), same shifted
inv-gamma(10) master, same z-calibration. Only edge effects (fraction of clusters near
the disk boundary ~ eps/sqrt(N)) and the per-field yield should change.

Test: quick CSR sims at N in {10000, 20000} (radius = sqrt(N) so lambda0 = 1/pi),
same DBSCAN pipeline + filters as simulate.py, scored with the shipped master.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.cluster_detector import MASTER, alpha_eps, master_sf  # noqa: E402

EPS = 1.40
MIN_SAMPLES = 10
MIN_CLUSTER = 10
MIN_AREA = 0.5
LAM0 = 1 / np.pi


def run_fields(args):
    n_pts, radius, n_fields, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_fields):
        r = radius * np.sqrt(rng.uniform(size=n_pts))
        t = 2 * np.pi * rng.uniform(size=n_pts)
        pts = np.column_stack((r * np.cos(t), r * np.sin(t)))
        lab = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(pts).labels_
        for L in range(lab.max() + 1):
            cp = pts[lab == L]
            if len(cp) < MIN_CLUSTER:
                continue
            try:
                S = ConvexHull(cp).volume
            except Exception:
                continue
            if S < MIN_AREA:
                continue
            out.append((len(cp), S))
    return out


def summarize(name, clus, n_fields):
    n = np.array([c[0] for c in clus], float)
    S = np.array([c[1] for c in clus], float)
    rt = (n / S) / LAM0 * EPS ** alpha_eps(EPS)
    p = np.clip(master_sf(rt), 1e-300, 1 - 1e-16)
    z = stats.norm.isf(p)
    band = lambda a, b: np.mean((z > a) & (z <= b))
    ks = stats.kstest(z, "norm").statistic
    print(f"{name}: fields={n_fields} clusters={len(rt)} yield={len(rt)/n_fields:.2f}/field")
    print(f"  Rt: min={rt.min():.2f}  q0.1%={np.quantile(rt,1e-3):.2f}  q1%={np.quantile(rt,.01):.2f}"
          f"  median={np.median(rt):.2f}  mean={rt.mean():.2f}   meanN'={n.mean():.3f}")
    print(f"  z : median={np.median(z):+.3f}  mean={np.mean(z):+.3f}  std={np.std(z):.3f}"
          f"  d_1={band(0,1)-band(-1,0):+.4f}  d_2={band(0,2)-band(-2,0):+.4f}  KS(z)={ks:.4f}"
          f"  (SE_median~{1.25/np.sqrt(len(z)):.3f})")
    return rt


def main():
    nproc = 16
    for n_pts, n_fields in [(10000, 8000), (20000, 5000), (40000, 3000)]:
        radius = float(np.sqrt(n_pts))          # keeps lambda0 = 1/pi
        per = n_fields // nproc
        jobs = [(n_pts, radius, per, 1000 * n_pts + j) for j in range(nproc)]
        with Pool(nproc) as pool:
            res = pool.map(run_fields, jobs)
        clus = [c for chunk in res for c in chunk]
        summarize(f"N={n_pts:6d} R={radius:7.2f}", clus, per * nproc)
        print()


if __name__ == "__main__":
    main()

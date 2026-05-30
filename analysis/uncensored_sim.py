#!/usr/bin/env python3
"""Uncensored CSR resim at eps=1.20 with min_area=0 (no singularity censorship).
Directly observes the full R = (N'/S')/lambda0 tail so we can pin BOTH the power-law
index (expect alpha~7, matching hull geometry) AND its absolute normalization, with no
deep-tail brute force. Writes incrementally to analysis/uncensored_eps1.20.csv so partial
results are usable. Does NOT touch simdata/v2.
"""
import os, time, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

N, RAD, EPS = 10000, 100, 1.20
MIN_SAMPLES, MIN_N, MIN_AREA = 10, 10, 0.0      # <-- censorship removed
TARGET = 1_000_000
BATCH = 2000
OUT = os.path.join(os.path.dirname(__file__), "uncensored_eps1.20.csv")
NPROC = max(1, min(14, (os.cpu_count() or 4) - 2))

def one(args):
    seed, it = args
    rng = np.random.default_rng(seed)
    r = RAD*np.sqrt(rng.uniform(size=N)); th = 2*np.pi*rng.uniform(size=N)
    pts = np.column_stack((r*np.cos(th), r*np.sin(th)))
    lab = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(pts).labels_
    out = []
    for L in set(lab):
        if L == -1: continue
        cp = pts[lab == L]
        if len(cp) < MIN_N: continue
        try: S = ConvexHull(cp).volume
        except Exception: S = 0.0
        if S <= MIN_AREA: continue                # min_area=0 -> only drops exact-degenerate
        out.append({"S_prime": S, "N_prime": len(cp), "iteration": it})
    return out if out else [{"S_prime": -1.0, "N_prime": -1, "iteration": it}]

def main():
    start = time.time(); done = 0; found = 0
    rng = np.random.default_rng(20260531)
    if os.path.exists(OUT):
        done = int(pd.read_csv(OUT, usecols=["iteration"])["iteration"].max())
        print(f"resuming from iteration {done}")
    with Pool(NPROC) as pool:
        while done < TARGET:
            b = min(BATCH, TARGET-done)
            seeds = rng.integers(0, 2**31-1, size=b)
            args = [(int(seeds[i]), done+i+1) for i in range(b)]
            res = pool.map(one, args)
            rows = [x for sub in res for x in sub]
            valid = [x for x in rows if x["S_prime"] != -1]
            found += len(valid)
            df = pd.DataFrame(rows)
            df.to_csv(OUT, mode="a", index=False, header=not os.path.exists(OUT))
            done += b
            if done % 20000 == 0:
                rate = done/(time.time()-start)
                print(f"  {done}/{TARGET} fields, {found} clusters, {rate:.0f} fields/s, "
                      f"{(TARGET-done)/rate/60:.1f} min left", flush=True)
    print(f"DONE: {done} fields, {found} clusters -> {OUT}")

if __name__ == "__main__":
    main()

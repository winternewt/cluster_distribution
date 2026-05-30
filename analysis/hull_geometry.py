#!/usr/bin/env python3
"""T3 — First-principles hull-area geometry of a DBSCAN cluster under CSR.

Finding #2 reduced everything to S'|n ~ Gamma(k(n), scale ∝ eps^2). Can we DERIVE
that from stochastic geometry, with no DBSCAN at all?

Generative model (CSR + Poisson conditioning):
  Pick a core point p. Under complete spatial randomness, conditioned on there being
  m = n-1 other points within distance eps of p, those m points are i.i.d. UNIFORM in
  the disk of radius eps centred at p. So a minimal cluster ≈ {p} ∪ {m uniform in disk(eps)}.
  Its convex hull area, in units of eps^2, is a pure dimensionless random variable.

We MC three models and compare to the real data's <S'|n>/eps^2 and k(n)=mean^2/var:
  A: n points uniform in disk(radius 1)                  (no special core)
  B: 1 centre point + (n-1) uniform in disk(radius 1)    (single-core CSR-conditioned)
  C: small real DBSCAN sim (to measure chaining inflation vs B)

Everything is dimensionless (radius 1); multiply areas by eps^2 to compare to data.
Cheap: no large sim, ~5e4 hull computations per n. Writes CSV + plot to analysis/.
"""
import os
import numpy as np
from numpy.random import default_rng
from scipy.spatial import ConvexHull
from scipy import stats
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(__file__)
N_FULL, RAD = 10000, 100
LAM0 = N_FULL / (np.pi * RAD ** 2)
rng = default_rng(20260530)


def unit_disk(m, rng):
    r = np.sqrt(rng.uniform(size=m)); th = 2 * np.pi * rng.uniform(size=m)
    return np.column_stack((r * np.cos(th), r * np.sin(th)))


def hull_area(pts):
    if len(pts) < 3:
        return 0.0
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def mc_model(kind, n, trials=50000):
    """Return array of hull areas (radius-1 units)."""
    areas = np.empty(trials)
    for i in range(trials):
        if kind == "A":
            pts = unit_disk(n, rng)
        else:  # B: centre + (n-1) uniform
            pts = np.vstack(([[0.0, 0.0]], unit_disk(n - 1, rng)))
        areas[i] = hull_area(pts)
    return areas


def real_data_ref():
    """<S'|n>/eps^2 and k(n) from real data at eps=1.20 (representative)."""
    import pandas as pd
    eps = 1.20
    df = pd.read_csv(os.path.join(OUT, "..", "simdata", "v2",
                                  f"simulation_data_N{N_FULL}_radius{RAD}_eps{eps:.2f}.csv"))
    d = df[(df.S_prime != -1) & (df.N_prime != -1)]
    out = {}
    for n in range(10, 16):
        s = d[d.N_prime == n].S_prime.values / eps ** 2
        if len(s) > 500:
            out[n] = (s.mean(), s.mean() ** 2 / s.var())
    return out


def main():
    ref = real_data_ref()
    print("=== hull area in eps^2 units: models vs real data (eps=1.20) ===")
    print("  n |  modelA mean(k) | modelB mean(k) |  DATA mean(k)")
    rows = []
    for n in range(10, 16):
        aA = mc_model("A", n, 40000)
        aB = mc_model("B", n, 40000)
        kA = aA.mean() ** 2 / aA.var()
        kB = aB.mean() ** 2 / aB.var()
        dat = ref.get(n, (np.nan, np.nan))
        rows.append(dict(n=n, A_mean=aA.mean(), A_k=kA, B_mean=aB.mean(), B_k=kB,
                         data_mean=dat[0], data_k=dat[1]))
        print(f"  {n:2d} |  {aA.mean():.3f} ({kA:5.1f}) |  {aB.mean():.3f} ({kB:5.1f}) | "
              f" {dat[0]:.3f} ({dat[1]:5.1f})")
    import pandas as pd
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "hull_geometry.csv"), index=False)

    # k(n) slopes
    for col, name in [("A_k", "model A"), ("B_k", "model B"), ("data_k", "data")]:
        m = R.dropna(subset=[col])
        if len(m) >= 3:
            a, b = np.polyfit(m.n, m[col], 1)
            print(f"  k(n) {name:8s}: {a:.2f}*n + ({b:.2f})  -> k(10)={a*10+b:.1f}")

    # --- Model C: tiny real DBSCAN sim, measure chaining inflation of S'|10 vs model B ---
    print("\n=== Model C: real DBSCAN minimal-cluster hull area (eps=1.2, radius-1 units) ===")
    eps = 1.2
    s10 = []
    for _ in range(3000):
        pts = unit_disk_full(N_FULL, RAD, rng)
        lab = DBSCAN(eps=eps, min_samples=10).fit(pts).labels_
        for L in set(lab):
            if L == -1:
                continue
            cp = pts[lab == L]
            if len(cp) == 10:
                s10.append(hull_area(cp) / eps ** 2)
        if len(s10) > 4000:
            break
    s10 = np.array(s10)
    print(f"  DBSCAN n=10 clusters: {len(s10)} found, mean(S'/eps^2)={s10.mean():.3f}, "
          f"k={s10.mean()**2/s10.var():.1f}")
    bmean = R[R.n == 10].B_mean.values[0]
    print(f"  model B (centre + 9 uniform): mean={bmean:.3f}")
    print(f"  chaining inflation DBSCAN/modelB = {s10.mean()/bmean:.3f}")
    print(f"  (data <S'|10>/eps^2 = {ref[10][0]:.3f})")

    # plot k(n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(R.n, R.A_k, "o-", label="model A: n uniform in disk")
    ax.plot(R.n, R.B_k, "s-", label="model B: centre + (n-1) uniform")
    ax.plot(R.dropna(subset=["data_k"]).n, R.dropna(subset=["data_k"]).data_k, "k^-", label="real data (eps=1.2)")
    ax.set_xlabel("n = N'"); ax.set_ylabel("Gamma shape k(n) = mean^2/var of S'|n")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Hull-area Gamma shape: CSR geometry reproduces k(n)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "hull_geometry.png"), dpi=110); plt.close(fig)
    print("\nwrote hull_geometry.csv, hull_geometry.png")


def unit_disk_full(N, RAD, rng):
    r = RAD * np.sqrt(rng.uniform(size=N)); th = 2 * np.pi * rng.uniform(size=N)
    return np.column_stack((r * np.cos(th), r * np.sin(th)))


if __name__ == "__main__":
    main()

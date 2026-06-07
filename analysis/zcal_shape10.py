#!/usr/bin/env python3
"""Final master candidate: shifted inverse-gamma with INTEGER shape=10.

SF(Rt) = P(10, y) = 1 - exp(-y) * sum_{k=0}^{9} y^k/k!,  y = scale/(Rt - loc)
(for Rt <= loc: SF = 1).  For deep-tail precision (small y) use the direct
series  P(10,y) = exp(-y) * sum_{k=10}^{inf} y^k/k!  instead of 1-Q.
Trivial exact JS implementation, no gamma library.

Fits (loc, scale) by MLE with shape frozen at 10 on the equal-weight pooled
Rt sample; reports z-calibration per eps vs the free-shape pooled master and
the shipped LL master.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zcal_experiment import load_rt, zcal, LL_SHAPE, LL_SCALE  # noqa: E402

EPS_LIST = [1.10, 1.20, 1.30, 1.40]


def sf_poisson10(rt: np.ndarray, loc: float, scale: float) -> np.ndarray:
    """P(10, scale/(rt-loc)), exact Poisson sums (matches JS implementation).

    Body (y >= 15):   SF = 1 - exp(-y) * sum_{k=0}^{9} y^k/k!
    Tail (y < 15):    SF = exp(-y) * sum_{k=10}^{~60} y^k/k!   (no cancellation)
    """
    out = np.ones_like(rt, dtype=float)
    m = rt > loc
    y = scale / (rt[m] - loc)
    res = np.empty_like(y)

    hi = y >= 15.0
    yh = y[hi]
    s = np.ones_like(yh)
    term = np.ones_like(yh)
    for k in range(1, 10):
        term *= yh / k
        s += term
    res[hi] = 1.0 - np.exp(-yh) * s

    yl = y[~hi]
    # term k=10 computed in log to avoid overflow concerns, then recurse
    term = np.exp(10 * np.log(np.maximum(yl, 1e-300)) - yl
                  - 15.104412573075516)  # ln(10!) = 15.1044...
    s = term.copy()
    for k in range(11, 80):
        term *= yl / k
        s += term
    res[~hi] = s

    out[m] = res
    return out


def main() -> None:
    data = {e: load_rt(e) for e in EPS_LIST}
    rng = np.random.default_rng(1)
    m = min(len(df) for df in data.values())
    pooled = np.concatenate([
        rng.choice(df.Rt.values, size=m, replace=False) for df in data.values()])

    a_free, loc_free, sc_free = stats.invgamma.fit(pooled)
    a10, loc10, sc10 = stats.invgamma.fit(pooled, fa=10)
    print(f"free  : shape={a_free:.4f} loc={loc_free:.4f} scale={sc_free:.4f}")
    print(f"fixed : shape=10        loc={loc10:.4f} scale={sc10:.4f}")

    # sanity: poisson-sum SF == scipy invgamma SF
    test = np.linspace(8, 200, 50)
    err = np.max(np.abs(sf_poisson10(test, loc10, sc10)
                        - stats.invgamma.sf(test, 10, loc10, sc10)))
    print(f"poisson-sum vs scipy max|dSF| = {err:.2e}")

    rows = []
    for e in EPS_LIST:
        rt = data[e].Rt.values
        for label, sf in [
            ("LL-current", 1.0 / (1.0 + (rt / LL_SCALE) ** LL_SHAPE)),
            ("invgamma-free", stats.invgamma.sf(rt, a_free, loc_free, sc_free)),
            ("invgamma-10", sf_poisson10(rt, loc10, sc10)),
        ]:
            p = np.clip(sf, 1e-300, 1 - 1e-16)
            rows.append({"eps": e, "master": label, **zcal(stats.norm.isf(p))})

    out = pd.DataFrame(rows).sort_values(["master", "eps"])
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")
    print("\ntarget: median_z=0, d_1=0, d_2=0, std=1, ks_z->0")
    print(out.to_string(index=False))
    out.to_csv(Path(__file__).parent / "zcal_shape10.csv", index=False)


if __name__ == "__main__":
    main()

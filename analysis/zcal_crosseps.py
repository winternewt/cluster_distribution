#!/usr/bin/env python3
"""Cross-eps validation of the shifted-inverse-gamma master.

Fit invgamma(shape, loc, scale) on Rt at eps=1.20 (full data), then measure
z-calibration at eps in {1.10, 1.20, 1.30, 1.40} with the SAME master.
If loc collapses with eps^alpha like the rest, one 3-param formula suffices.
Also fits a pooled master across all four eps as the deliverable candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zcal_experiment import load_rt, zcal  # noqa: E402

EPS_LIST = [1.10, 1.20, 1.30, 1.40]


def main() -> None:
    data = {e: load_rt(e) for e in EPS_LIST}
    for e, df in data.items():
        print(f"eps={e:.2f}: {len(df)} clusters")

    # master fit at 1.20
    a, loc, sc = stats.invgamma.fit(data[1.20].Rt.values)
    print(f"\nmaster (fit @1.20): shape={a:.4f} loc={loc:.4f} scale={sc:.4f}")

    rows = []
    for e in EPS_LIST:
        p = np.clip(stats.invgamma.sf(data[e].Rt.values, a, loc, sc), 1e-300, 1 - 1e-16)
        rows.append({"eps": e, "master": "fit@1.20", **zcal(stats.norm.isf(p))})

    # pooled master (equal-weight subsample per eps so 1.40 doesn't dominate)
    rng = np.random.default_rng(1)
    m = min(len(df) for df in data.values())
    pooled = np.concatenate([
        rng.choice(df.Rt.values, size=m, replace=False) for df in data.values()])
    ap, locp, scp = stats.invgamma.fit(pooled)
    print(f"pooled master:      shape={ap:.4f} loc={locp:.4f} scale={scp:.4f}")
    for e in EPS_LIST:
        p = np.clip(stats.invgamma.sf(data[e].Rt.values, ap, locp, scp), 1e-300, 1 - 1e-16)
        rows.append({"eps": e, "master": "pooled", **zcal(stats.norm.isf(p))})

    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")
    print("\ntarget: median_z=0, d_1=0, d_2=0, std=1, ks_z->0")
    print(out.to_string(index=False))
    out.to_csv(Path(__file__).parent / "zcal_crosseps.csv", index=False)

    # per-eps free fit, to see whether (shape, loc, scale) are eps-stable
    print("\nper-eps free invgamma-loc fits on Rt:")
    for e in EPS_LIST:
        ae, le, se = stats.invgamma.fit(data[e].Rt.values)
        print(f"  eps={e:.2f}: shape={ae:8.4f} loc={le:8.4f} scale={se:9.4f}")


if __name__ == "__main__":
    main()

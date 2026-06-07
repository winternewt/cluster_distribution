#!/usr/bin/env python3
"""z-calibration shoot-out for the master SF (RCA follow-up).

RCA claim: "no 2-parameter family can eliminate the asymmetry".
Counter-hypothesis (Newton): the marginal pooled over N' is an N'-mixture of
scaled-inverse-gammas (analytic-night result); the log-skew +0.67 is mixture
skew + a left location shift, not an irreducible feature.  Test:

  marginal families:  log-logistic, invgamma(floc=0), invgamma(loc free),
                      betaprime(floc=0), betaprime(loc free), burr12, gengamma
  conditional on N':  per-n invgamma (floc=0 / loc free), per-n log-logistic

Honest protocol: 50/50 train/test split; fit on train, z-calibrate on test.
z = Phi^-1(1 - SF(Rt)) per cluster; report median_z, mean_z, band asymmetries
(the RCA fingerprint) and KS(z, N(0,1)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.simv2_data import load_raw_df  # noqa: E402

LAMBDA0 = 0.3183098861837907
C0, C1 = 2.031525, 0.258273          # alpha(eps) running exponent (keep)
LL_SHAPE, LL_SCALE = 7.87468, 24.05651  # current shipped master

RNG = np.random.default_rng(0)


def alpha(eps: float) -> float:
    return C0 + C1 * np.log(eps)


def load_rt(eps: float) -> pd.DataFrame:
    df = load_raw_df(eps)
    df = df[(df.S_prime != -1) & (df.N_prime != -1)].copy()
    df["R"] = (df.N_prime / df.S_prime) / LAMBDA0
    df["Rt"] = df.R * eps ** alpha(eps)
    return df[["Rt", "N_prime"]].reset_index(drop=True)


def zcal(z: np.ndarray) -> dict:
    """RCA fingerprint + KS against N(0,1)."""
    z = z[np.isfinite(z)]
    n = len(z)
    band = lambda a, b: np.mean((z > a) & (z <= b))
    d10 = band(0, 1) - band(-1, 0)
    d20 = band(0, 2) - band(-2, 0)
    ks = stats.kstest(z, "norm").statistic
    return dict(n=n, median_z=np.median(z), mean_z=np.mean(z),
                d_1=d10, d_2=d20, std=np.std(z), ks_z=ks)


# ---------------- marginal candidates ----------------

def sf_factory_marginal(name: str, train: np.ndarray):
    """Fit on train Rt, return (label, sf(rt)) or None on failure."""
    try:
        if name == "LL-current":
            return lambda x: 1.0 / (1.0 + (x / LL_SCALE) ** LL_SHAPE)
        if name == "LL-mle":
            c, loc, sc = stats.fisk.fit(train, floc=0)
            return lambda x: stats.fisk.sf(x, c, loc, sc)
        if name == "LL-mle-loc":
            c, loc, sc = stats.fisk.fit(train)
            return lambda x: stats.fisk.sf(x, c, loc, sc)
        if name == "invgamma":
            a, loc, sc = stats.invgamma.fit(train, floc=0)
            return lambda x: stats.invgamma.sf(x, a, loc, sc)
        if name == "invgamma-loc":
            a, loc, sc = stats.invgamma.fit(train)
            return lambda x: stats.invgamma.sf(x, a, loc, sc)
        if name == "betaprime":
            a, b, loc, sc = stats.betaprime.fit(train, floc=0)
            return lambda x: stats.betaprime.sf(x, a, b, loc, sc)
        if name == "betaprime-loc":
            a, b, loc, sc = stats.betaprime.fit(train)
            return lambda x: stats.betaprime.sf(x, a, b, loc, sc)
        if name == "burr12":
            c, d, loc, sc = stats.burr12.fit(train, floc=0)
            return lambda x: stats.burr12.sf(x, c, d, loc, sc)
        if name == "burr12-loc":
            c, d, loc, sc = stats.burr12.fit(train)
            return lambda x: stats.burr12.sf(x, c, d, loc, sc)
        if name == "gengamma":
            a, c, loc, sc = stats.gengamma.fit(train, floc=0)
            return lambda x: stats.gengamma.sf(x, a, c, loc, sc)
        if name == "dagum":  # burr (Dagum) = mirror of burr12
            c, d, loc, sc = stats.burr.fit(train, floc=0)
            return lambda x: stats.burr.sf(x, c, d, loc, sc)
    except Exception as e:  # noqa: BLE001
        print(f"  [fit fail] {name}: {e}")
        return None
    return None


# ---------------- conditional on N' ----------------

def fit_conditional(train: pd.DataFrame, dist, free_loc: bool, min_n: int = 500):
    """Per-n fits; linear extrapolation of params for sparse n."""
    params: dict[int, tuple] = {}
    counts = train.N_prime.value_counts()
    for n, cnt in counts.items():
        if cnt < min_n:
            continue
        x = train.Rt[train.N_prime == n].values
        try:
            if free_loc:
                params[n] = dist.fit(x)
            else:
                params[n] = dist.fit(x, floc=0)
        except Exception:  # noqa: BLE001
            continue
    if not params:
        return None
    ns = np.array(sorted(params))
    # linear extrapolation of each param in n for unfitted n
    pmat = np.array([params[n] for n in ns])
    coef = [np.polyfit(ns, pmat[:, j], 1) for j in range(pmat.shape[1])]

    def get_params(n: int) -> tuple:
        if n in params:
            return params[n]
        return tuple(np.polyval(c, n) for c in coef)

    def sf(rt: np.ndarray, nprime: np.ndarray) -> np.ndarray:
        out = np.empty(len(rt))
        for n in np.unique(nprime):
            m = nprime == n
            out[m] = dist.sf(rt[m], *get_params(int(n)))
        return out

    return sf, params


def main() -> None:
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 1.20
    df = load_rt(eps)
    print(f"eps={eps}  clusters={len(df)}  alpha={alpha(eps):.4f}")
    idx = RNG.permutation(len(df))
    tr, te = df.iloc[idx[: len(df) // 2]], df.iloc[idx[len(df) // 2:]]
    rt_te, np_te = te.Rt.values, te.N_prime.values

    rows = []
    names = ["LL-current", "LL-mle", "LL-mle-loc", "invgamma", "invgamma-loc",
             "betaprime", "betaprime-loc", "burr12", "burr12-loc",
             "gengamma", "dagum"]
    for name in names:
        sf = sf_factory_marginal(name, tr.Rt.values)
        if sf is None:
            continue
        p = np.clip(sf(rt_te), 1e-300, 1 - 1e-16)
        z = stats.norm.isf(p)
        rows.append({"model": name, **zcal(z)})

    for label, dist, free_loc in [
        ("cond-invgamma", stats.invgamma, False),
        ("cond-invgamma-loc", stats.invgamma, True),
        ("cond-LL", stats.fisk, False),
        ("cond-LL-loc", stats.fisk, True),
        ("cond-betaprime-loc", stats.betaprime, True),
    ]:
        res = fit_conditional(tr, dist, free_loc)
        if res is None:
            continue
        sf, params = res
        p = np.clip(sf(rt_te, np_te), 1e-300, 1 - 1e-16)
        z = stats.norm.isf(p)
        rows.append({"model": label, **zcal(z)})
        if label == "cond-invgamma-loc":
            print("\nper-n invgamma-loc params (n: shape, loc, scale):")
            for n in sorted(params):
                a, loc, sc = params[n]
                print(f"  {n}: {a:8.3f} {loc:8.3f} {sc:9.3f}")

    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")
    print("\ntarget: median_z=0, d_1=0, d_2=0, std=1, ks_z->0")
    print(out.to_string(index=False))
    out.to_csv(Path(__file__).parent / f"zcal_eps{eps:.2f}.csv", index=False)


if __name__ == "__main__":
    main()

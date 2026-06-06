"""simv2_data — data access layer for simdata/v2 simulation results.

Read priority per eps slice:
  1. simdata/v2_parquet/eps_{eps:.2f}.parquet  (local parquet, preferred)
  2. data_dir/simulation_data_N{N}_radius{R}_eps{eps:.2f}.csv  (legacy CSV, if present)
  3. HuggingFace download → cached to simdata/v2_parquet/  (auto, public repo)

Public API
----------
load_raw_df(eps, N, radius, data_dir)  -> filtered DataFrame (no placeholders)
load_data(eps, data_dir, N, radius)    -> density_ratio array  (legacy callers)
sample_data(density_ratio, ...)        -> subsampled array
load_density_ratio(eps, ...)           -> sampled density_ratio
load_fit_parameters(output_dir, eps)   -> (regular_fit, mixture_fit)
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HF_REPO = "Winternewt/cluster-distribution-simdata"
_REPO_ROOT = Path(__file__).parent.parent
PQ_DIR = _REPO_ROOT / "simdata" / "v2_parquet"


# ── internal helpers ──────────────────────────────────────────────────────────

def _pq_path(eps: float) -> Path:
    return PQ_DIR / f"eps_{eps:.2f}.parquet"


def _csv_path(eps: float, N: int, radius: int, data_dir: str) -> str:
    return os.path.join(data_dir, f"simulation_data_N{N}_radius{int(radius)}_eps{eps:.2f}.csv")


def _download_parquet(eps: float) -> Path | None:
    """Download one parquet from HuggingFace to PQ_DIR. Returns local path or None."""
    PQ_DIR.mkdir(parents=True, exist_ok=True)
    dest = _pq_path(eps)
    url = (f"https://huggingface.co/datasets/{HF_REPO}"
           f"/resolve/main/data/eps_{eps:.2f}.parquet")
    try:
        print(f"  [simv2_data] downloading eps={eps:.2f} from HuggingFace…")
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as e:
        print(f"  [simv2_data] download failed for eps={eps:.2f}: {e}")
        if dest.exists():
            dest.unlink()
        return None


def _read_raw(eps: float, N: int = 10000, radius: int = 100,
              data_dir: str | None = None) -> pd.DataFrame | None:
    """Return the full DataFrame for one eps (placeholders included), or None."""
    pq = _pq_path(eps)

    # 1. local parquet
    if pq.exists():
        return pd.read_parquet(pq)

    # 2. legacy CSV
    if data_dir:
        csv = _csv_path(eps, N, radius, data_dir)
        if os.path.exists(csv):
            return pd.read_csv(csv,
                               dtype={"S_prime": "float64",
                                      "N_prime": "int64",
                                      "iteration": "int64"})

    # 3. HuggingFace
    local = _download_parquet(eps)
    if local is not None:
        return pd.read_parquet(local)

    return None


# ── public API ────────────────────────────────────────────────────────────────

def load_raw_df(eps: float, N: int = 10000, radius: int = 100,
                data_dir: str | None = None) -> pd.DataFrame | None:
    """Load one eps slice with placeholder rows removed.

    Returns DataFrame with columns S_prime, N_prime, iteration, or None if
    unavailable / fewer than 2000 valid clusters.
    """
    df = _read_raw(eps, N, radius, data_dir)
    if df is None:
        return None
    d = df[(df["S_prime"] != -1) & (df["N_prime"] != -1)].copy()
    if len(d) < 2000:
        print(f"  [simv2_data] fewer than 2000 valid clusters for eps={eps:.2f}, skipping.")
        return None
    return d


def load_data(eps: float, data_dir: str = "./simdata/v2/",
              N: int = 10000, radius: float = 100.0) -> np.ndarray | None:
    """Return density_ratio array for eps (legacy API, unchanged signature).

    Reads parquet / CSV / HF in that order; data_dir is consulted only if no
    local parquet is found.
    """
    d = load_raw_df(eps, int(N), int(radius), data_dir)
    if d is None:
        return None
    lambda0 = N / (np.pi * radius ** 2)
    return (d["N_prime"].values / d["S_prime"].values) / lambda0


def sample_data(density_ratio: np.ndarray, sample_size: int = 100000,
                random_seed: int = 42) -> np.ndarray | None:
    """Subsample density_ratio for computational efficiency."""
    if density_ratio is None or len(density_ratio) == 0:
        return None
    n = min(sample_size, len(density_ratio))
    rng = np.random.default_rng(random_seed)
    return rng.choice(density_ratio, size=n, replace=False)


def load_density_ratio(eps: float, data_dir: str = "./simdata/v2/",
                       N: int = 10000, radius: float = 100.0,
                       sample_size: int = 100000,
                       random_seed: int = 42) -> np.ndarray | None:
    """Load and subsample density_ratio for one eps (legacy API)."""
    ratio = load_data(eps, data_dir, N, radius)
    if ratio is None:
        return None
    return sample_data(ratio, sample_size, random_seed)


def load_fit_parameters(output_dir: str, eps: float):
    """Load precomputed fit CSVs for eps. Returns (regular_fit, mixture_fit)."""
    def _load(fname):
        p = os.path.join(output_dir, fname)
        return pd.read_csv(p).iloc[0] if os.path.exists(p) else None

    return (_load(f"regular_fit_eps{eps:.2f}.csv"),
            _load(f"mixture_fit_eps{eps:.2f}.csv"))

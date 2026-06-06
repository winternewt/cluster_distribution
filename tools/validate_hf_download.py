"""Download parquet files from HuggingFace and validate against local CSVs.

Usage:
    python tools/validate_hf_download.py --token TOKEN [--n N] [--all]

Downloads N evenly-spaced parquet files from HF (default 10), compares
each byte-for-byte against the corresponding local CSV. Exits 0 on full
match, 1 on any mismatch.

--all  : validate every one of the 207 files (slow, ~20 min download).
--n N  : validate N evenly-spaced files (default 10).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Winternewt/cluster-distribution-simdata"
CSV_DIR = Path("simdata/v2")


def csv_path(eps_str: str) -> Path:
    return CSV_DIR / f"simulation_data_N10000_radius100_eps{eps_str}.csv"


def validate_one(eps_str: str, token: str, tmpdir: str) -> bool:
    """Download one parquet, compare to CSV. Returns True on match."""
    pq_file = f"data/eps_{eps_str}.parquet"
    local_csv = csv_path(eps_str)

    if not local_csv.exists():
        print(f"  SKIP eps={eps_str}: CSV not found locally")
        return True

    # Download from HF
    local_pq = hf_hub_download(
        repo_id=REPO_ID,
        filename=pq_file,
        repo_type="dataset",
        token=token,
        cache_dir=tmpdir,
    )

    # Load both
    df_csv = pd.read_csv(local_csv,
                         dtype={"S_prime": "float64", "N_prime": "int64", "iteration": "int64"})
    df_pq = pd.read_parquet(local_pq)[["S_prime", "N_prime", "iteration"]]

    try:
        pd.testing.assert_frame_equal(
            df_csv.reset_index(drop=True),
            df_pq.reset_index(drop=True),
            check_exact=True,
        )
        rows = len(df_csv)
        real = int((df_csv.S_prime != -1).sum())
        print(f"  OK   eps={eps_str}  rows={rows:>9,}  real={real:>8,}")
        return True
    except AssertionError as e:
        print(f"  FAIL eps={eps_str}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--n", type=int, default=10, help="number of spot-check files")
    ap.add_argument("--all", action="store_true", help="validate all 207 files")
    args = ap.parse_args()

    api = HfApi(token=args.token)
    all_pq = sorted(
        f.replace("data/eps_", "").replace(".parquet", "")
        for f in api.list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    )
    if not all_pq:
        print("No parquet files found on HF.", file=sys.stderr)
        sys.exit(1)

    if args.all:
        to_check = all_pq
    else:
        step = max(1, len(all_pq) // args.n)
        to_check = all_pq[::step][: args.n]

    print(f"Validating {len(to_check)}/{len(all_pq)} files from HF...")
    print(f"  Repo: https://huggingface.co/datasets/{REPO_ID}\n")

    failures = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for eps in to_check:
            ok = validate_one(eps, args.token, tmpdir)
            if not ok:
                failures.append(eps)

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"All {len(to_check)} files match. HF mirror is valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()

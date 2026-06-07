"""Upload simdata/v2_parquet/ to HuggingFace Datasets.

Usage:
    python tools/upload_to_hf.py --token HF_TOKEN [--repo REPO_ID] [--only EPS]

Creates (or updates) a public HuggingFace dataset repository and uploads
all parquet files from simdata/v2_parquet/.

Repo structure on HF:
    data/eps_X.XX.parquet   (one per eps, 207 files)
    README.md               (dataset card)

To load later:
    import pandas as pd
    df = pd.read_parquet("hf://datasets/Winternewt/cluster-distribution-simdata/data/eps_1.20.parquet")
"""

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo

PQ_DIR = Path("simdata/v2_parquet")
DEFAULT_REPO = "Winternewt/cluster-distribution-simdata"

DATASET_CARD = """\
---
license: cc-by-4.0
task_categories:
- other
language:
- en
tags:
- spatial-statistics
- DBSCAN
- point-process
- cluster-detection
- Monte-Carlo
pretty_name: DBSCAN Cluster Density Ratios on CSR Point Fields
size_categories:
- 1B<n<10B
---

# DBSCAN Cluster Density Ratios — CSR Simulation Data

Monte-Carlo simulation of DBSCAN clustering applied to complete-spatial-randomness (CSR)
point fields. Used to characterize the null distribution of detected cluster density ratios
and calibrate anomaly detection p-values (look-elsewhere correction).

## Simulation parameters

| Parameter | Value |
|---|---|
| N (points per field) | 10,000 |
| Domain | Disk, radius R=100 |
| Domain area S₀ | π·100² ≈ 31,415.93 |
| Background intensity λ₀ | N/S₀ = 1/π ≈ 0.31831 |
| DBSCAN min_samples | 10 |
| Post-filter min_cluster_size | 10 |
| Post-filter min_area | 0.5 |
| eps sweep | 0.80 → 2.86 (step 0.01, 207 values) |
| Iterations per eps | ~12,000 – 1,000,000 |

**Total compute**: ~60 h on a single workstation (16-process parallel).

## Schema

Each parquet file corresponds to one `eps` value (filename: `data/eps_X.XX.parquet`).

| Column | Type | Description |
|---|---|---|
| `S_prime` | float64 | Convex-hull area of detected cluster. **-1.0 = no cluster found** (placeholder row) |
| `N_prime` | int64 | Point count of detected cluster. **-1 = no cluster found** |
| `iteration` | int64 | Field index (1-based). Multiple clusters per iteration are all recorded. |

**Always filter** `S_prime != -1` before analysis. Placeholder rows mark iterations with no
valid DBSCAN cluster (common at low eps where clustering is rare).

## Key derived quantities

```python
lambda0 = 10000 / (np.pi * 100**2)       # ≈ 0.31831
R = (df.N_prime / df.S_prime) / lambda0  # density ratio (main statistic)
alpha = 2.031525 + 0.258273 * np.log(eps)  # running exponent (fit from 110 eps values)
R_tilde = R * eps**alpha                  # eps-collapsed statistic (<0.5% residual spread)
```

The density ratio `R` follows a heavy-tailed distribution (tail index α≈7). After
rescaling to `R̃ = R·ε^α(ε)` where `α(ε) = 2.031 + 0.258·ln ε`, the distribution
collapses to a single master (log-logistic with shape≈7.87, scale≈24.06; or inv-gamma
shape≈20.5) across all eps — the central empirical finding of this dataset.

## Usage

```python
import pandas as pd
import numpy as np

# Load one eps slice
eps = 1.20
df = pd.read_parquet(f"hf://datasets/Winternewt/cluster-distribution-simdata/data/eps_{eps:.2f}.parquet")
df = df[df.S_prime != -1]   # drop placeholder rows

lambda0 = 10000 / (np.pi * 100**2)
df['R'] = (df.N_prime / df.S_prime) / lambda0
alpha = 2.031525 + 0.258273 * np.log(eps)
df['R_tilde'] = df['R'] * eps**alpha
print(df.R_tilde.describe())
```

## Repository

Source code and analysis: https://github.com/winternewt/cluster_distribution
Analytic findings: see `docs/analytic_findings.md` in the source repo.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--only", default=None, help="single eps (e.g. 1.20) for testing")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not PQ_DIR.exists():
        print(f"ERROR: {PQ_DIR} does not exist. Run simdata_to_parquet.py first.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=args.token)

    # Create repo if needed
    try:
        create_repo(args.repo, repo_type="dataset", private=args.private,
                    token=args.token, exist_ok=True)
        print(f"Repo: https://huggingface.co/datasets/{args.repo}")
    except Exception as e:
        print(f"WARNING creating repo: {e}", file=sys.stderr)

    # Upload dataset card
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print("Uploaded README.md (dataset card)")

    parquets = sorted(PQ_DIR.glob("eps_*.parquet"))
    if args.only:
        parquets = [p for p in parquets if f"eps_{args.only}" in p.name]
    if not parquets:
        print("No parquet files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading {len(parquets)} parquet files...")
    errors = []
    for i, pq_path in enumerate(parquets, 1):
        dest = f"data/{pq_path.name}"
        try:
            api.upload_file(
                path_or_fileobj=str(pq_path),
                path_in_repo=dest,
                repo_id=args.repo,
                repo_type="dataset",
                commit_message=f"Add {pq_path.name}",
            )
            size_kb = pq_path.stat().st_size // 1024
            print(f"[{i:3d}/{len(parquets)}] {pq_path.name}  ({size_kb:,} KB)")
        except Exception as e:
            print(f"ERROR {pq_path.name}: {e}", file=sys.stderr)
            errors.append(pq_path.name)
            time.sleep(2)

    if errors:
        print(f"\nFAILED ({len(errors)}): {errors}", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone. Dataset at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

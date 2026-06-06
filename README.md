# cluster-distribution

Monte-Carlo study of DBSCAN clustering on random 2D point clouds (CSR / Poisson null),
plus analytic derivation of the null distribution and a calibrated anomaly detector.

```bash
uv sync
uv run cluster-distribution simulate   # or: fit, stats, visualize, plot, ...
```

Full catalog: [docs/overview.md](docs/overview.md) ·
Analytic results: [docs/analytic_findings.md](docs/analytic_findings.md) ·
Executive summary: [docs/EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md)

## Findings in brief

DBSCAN-on-noise produces structured, non-trivial cluster statistics. The density ratio
`R = (N'/S')/λ₀` obeys a factorization `R(eps) = scale(eps)·X` where `X` is
eps-invariant: rescaling to `R̃ = R·eps²` collapses the distribution onto a single
master inverse-gamma (shape ≈ 20.5). The correct detection statistic is the
**Kulldorff scan likelihood ratio**, not the bare density ratio. A typical CSR cluster
scores "~6σ" under naive per-window scoring — overstated ~10⁸×. See
[docs/analytic_findings.md](docs/analytic_findings.md) for the full derivation.

## Detector library

`modules/cluster_detector.py` — importable, calibrated two-arm anomaly detector.
Flags local over-densities and gives each a look-elsewhere-corrected p-value and
Gaussian-equivalent z via DBSCAN + Kulldorff scan-LR (tight clumps) and KDE peaks
(extended over-densities, hybrid analytic/MC null).

```python
import numpy as np
from modules.cluster_detector import Detector

det = Detector(radius=100.0, eps=1.2, n_background=10000)
det.calibrate(n_mc=2000, cache_path="null.npz")   # MC the null once; cached thereafter
# det = Detector.load("null.npz")                  # reuse a saved calibration

pts = np.load("my_points.npy")                     # (M, 2) coords inside the disk
for d in det.score(pts, zthr=3.0):
    print(d)   # Detection(method, z, p_value, x, y, n_points, scale)
```

For a non-CSR background: `calibrate(null_generator=lambda rng: my_points(rng))`.

## Live demo

`webapp/index.html` — static, dependency-free browser demo. Opens with no build step.
Three live panels: noise field + detected clusters, `R̃` histogram converging to the
master curve, cluster z-scores forming N(0,1). Move the eps slider to see the collapse.

```bash
cd webapp && python3 -m http.server 8000   # then open http://localhost:8000
```

GitHub Pages: copy `webapp/` to a `docs/` folder or `gh-pages` branch (see `webapp/README.md`).

## Data

207 simulation CSVs (~4.3 GB) are stored as Parquet on HuggingFace — **not in this repo**.
Scripts download slices on demand and cache them locally in `simdata/v2_parquet/`.

**HuggingFace dataset:** https://huggingface.co/datasets/Winternewt/cluster-distribution-simdata

```python
# Scripts handle this automatically; to fetch a slice manually:
import pandas as pd
df = pd.read_parquet(
    "hf://datasets/Winternewt/cluster-distribution-simdata/data/eps_1.20.parquet"
)
df = df[df.S_prime != -1]   # drop placeholder rows (no cluster found that iteration)
```

**Bulk download:**
```bash
huggingface-cli download Winternewt/cluster-distribution-simdata \
    --repo-type dataset --local-dir simdata/v2_parquet
```

Schema: `S_prime` (float64, convex-hull area), `N_prime` (int64, cluster size),
`iteration` (int64). Rows with `S_prime = -1` are placeholder (no cluster that field).

## Tools

| Script | Purpose |
|---|---|
| `tools/simdata_to_parquet.py` | Convert local CSVs → zstd parquet, validate 1-to-1 |
| `tools/upload_to_hf.py` | Upload parquet folder to HuggingFace (uses `upload_large_folder`) |
| `tools/validate_hf_download.py` | Download N spot-check files from HF and assert exact match vs local CSVs |

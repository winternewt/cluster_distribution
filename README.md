# cluster-distribution

DBSCAN cluster simulation on random 2D point clouds, plus Beta-Prime fitting on cluster density ratios.

```bash
uv sync
uv run cluster-distribution simulate   # or: fit, stats, visualize, plot, ...
```

Full catalog: [docs/overview.md](docs/overview.md). Analytic results & the detector theory: [docs/analytic_findings.md](docs/analytic_findings.md).

## Detector library

`modules/cluster_detector.py` is the importable, calibrated anomaly detector distilled from
the analysis. It flags local over-densities in a 2D point field and gives each a
look-elsewhere-corrected p-value and Gaussian-equivalent z, via two complementary arms:
DBSCAN + Kulldorff scan-LR (tight clumps) and kernel-density peaks (extended over-densities,
with a hybrid analytic/MC null).

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

For a non-CSR background, pass your own field sampler: `calibrate(null_generator=lambda rng: my_points(rng))`.
Caveats (see Finding #9): score raw points, not the bare density ratio; the null assumes the
configured `n_background`/`radius`/`eps` (rescale via the µ_eps law for other settings); RFT
z in the deep tail is a screening value.

## Data

Simulation CSVs live in `simdata/v2/` (~4.3 GB, Git LFS). Clone without downloading them:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <url>
```

## Push code, skip LFS upload

```bash
git add -A -- ':!simdata/v2/*.csv'   # optional: omit data from commit
git commit -m "your message"
GIT_LFS_SKIP_PUSH=1 git push -u origin HEAD
```

`GIT_LFS_SKIP_PUSH=1` pushes commits (and LFS pointer files if any) without uploading LFS blobs — stays within GitHub's free bandwidth quota.

# cluster-distribution

DBSCAN cluster simulation on random 2D point clouds, plus Beta-Prime fitting on cluster density ratios.

```bash
uv sync
uv run cluster-distribution simulate   # or: fit, stats, visualize, plot, ...
```

Full catalog: [docs/overview.md](docs/overview.md).

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

# Cluster Distribution — Project Overview

Statistical study of DBSCAN cluster properties on uniformly random 2D point clouds. The pipeline simulates clustering across a sweep of `eps` values, extracts per-cluster metrics (`S'`, `N'`), computes density ratios, and fits Beta-Prime (and mixture) distributions to the resulting data.

Last active: September 2024.

## Setup

This project uses [uv](https://docs.astral.sh/uv/). The old `requirements.txt` (conda-local paths) is superseded by `pyproject.toml` + `uv.lock`.

```bash
uv sync
uv run cluster-distribution simulate
```

Python: `>=3.11` (`.python-version` pins 3.14).

## Directory Layout

```
cluster_distribution/
├── docs/                  # Documentation (this file)
├── modules/               # Shared library code (incl. cluster_detector.py)
├── scripts/               # Analysis & simulation scripts, grouped semantically
│   ├── sim/               #   simulate.py, plotter.py
│   ├── fit/               #   beta_mix_vs_regular.py, mixure_of_betas.py, mixture_of_betas2.py, floc.py
│   ├── stats/             #   stat_tests.py, stat_test2.py, stat_test3.py, anova.py
│   └── viz/               #   visualize.py, beta_plot.py, plot_hist.py
├── analysis/              # Analytic-night & detector studies (scorer, zcal, ncheck, ...)
├── webapp/                # Live browser demo (GitHub Pages)
├── tools/                 # Parquet conversion / HuggingFace upload
├── simdata/               # Simulation inputs/outputs
│   ├── v1/                # Legacy data format + converter
│   └── v2_parquet/        # Parquet cache, auto-downloaded from HuggingFace
├── results/               # Fit outputs, plots, summaries (~1.9 MB)
├── main.py                # Typer CLI entry (cluster-distribution)
├── pyproject.toml         # Project metadata & dependencies
├── uv.lock                # Locked dependency versions
└── .gitattributes         # Git LFS rules for large data
```

## Pipeline (typical workflow)

1. **Simulate** — `scripts/sim/simulate.py` generates random points in a circle, runs DBSCAN, records cluster area/count per iteration.
2. **Explore** — `scripts/viz/visualize.py`, `scripts/stats/stat_tests.py`, `scripts/sim/plotter.py` inspect distributions across eps values.
3. **Fit** — `scripts/fit/beta_mix_vs_regular.py`, `scripts/fit/mixure_of_betas.py`, `scripts/fit/mixture_of_betas2.py` fit Beta-Prime models.
4. **Plot fits** — `scripts/viz/beta_plot.py`, `scripts/viz/plot_hist.py`, `scripts/fit/floc.py` visualize fitted parameters vs eps.
5. **Validate** — `scripts/stats/anova.py`, `scripts/stats/stat_test2.py`, `scripts/stats/stat_test3.py` run statistical tests.

## Scripts

### Simulation

| Script | Purpose |
|--------|---------|
| `scripts/sim/simulate.py` | Parallel DBSCAN simulation over an eps sweep. Writes `simulation_data_N{N}_radius{R}_eps{eps}.csv` to `simdata/v2/`. Resumable (appends to existing file). |
| `scripts/sim/plotter.py` | 2D histogram of cluster area vs count from a single simulation file. |

### Distribution fitting

| Script | Purpose |
|--------|---------|
| `scripts/fit/beta_mix_vs_regular.py` | Compare single Beta-Prime vs two-component mixture fits per eps. Writes `regular_fit_eps*.csv`, `mixture_fit_eps*.csv`, `regression_params.csv`. |
| `scripts/fit/mixure_of_betas.py` | Fit Beta-Prime mixture across merged eps data (typo in filename — kept for history). |
| `scripts/fit/mixture_of_betas2.py` | Fit eps-dependent Beta-Prime model with regression-linked parameters. Writes `merged_fit_params.csv`. |
| `scripts/fit/floc.py` | Explore fixed-location (`floc`) Beta-Prime fitting; compares gamma, lognormal, Weibull, betaprime. |
| `scripts/viz/plot_hist.py` | Histogram + fit overlay plots for density ratios. |

### Statistical analysis

| Script | Purpose |
|--------|---------|
| `scripts/stats/stat_tests.py` | Normality tests (Shapiro, Anderson-Darling, KS) on density ratios; Poisson tests on `N'`. Batch plots to `results/ratio_plots/` and `results/N_prime_plots/`. |
| `scripts/stats/stat_test2.py` | Distribution comparison with ECDF plots (gamma, lognormal, Weibull, betaprime). |
| `scripts/stats/stat_test3.py` | Gamma fit goodness-of-fit across eps values. |
| `scripts/stats/anova.py` | One-way ANOVA and linear regression on `lambda_prime` across eps. |

### Visualization

| Script | Purpose |
|--------|---------|
| `scripts/viz/visualize.py` | Q-Q plots, correlation heatmaps, regression diagnostics across eps. |
| `scripts/viz/beta_plot.py` | Overlay regular vs mixture Beta-Prime PDFs using saved fit params. |

### Modules (`modules/`)

| Module | Exports / role |
|--------|----------------|
| `simv2_data.py` | `load_data`, `sample_data`, `load_fit_parameters`, `load_density_ratio` — I/O for v2 CSVs and fit results. |
| `beta_stats.py` | Beta-Prime mixture PDF, negative log-likelihood, eps-dependent parameter helpers, linear regression on fit params. |
| `common_stats.py` | `compute_aic_bic`, `compute_ks_statistic` — shared fit quality metrics. |

### Entry point

| Item | Notes |
|------|-------|
| `main.py` | Typer CLI — `uv run cluster-distribution <command>` (`simulate`, `fit`, `stats`, `visualize`, `plot`, …) |
| `simdata/v1/convert.py` | Converts v1 simulation format to v2. v1 data dir is empty (8 KB). |

## Data Files

### Simulation data (`simdata/v2/`)

- **207 CSV files**, eps range **0.80 – 2.86** (step 0.01), naming pattern:
  `simulation_data_N10000_radius100_eps{eps:.2f}.csv`
- **Total size: ~4.3 GB** (~20–44 MB per file, ~1M rows each)
- **Columns:** `S_prime` (cluster convex-hull area), `N_prime` (cluster point count), `iteration`
- Placeholder rows (`S_prime=-1, N_prime=-1`) indicate iterations with no valid cluster
- **Git LFS:** all `simdata/v2/*.csv` are tracked via Git LFS (see `.gitattributes`)

### Results (`results/`)

| File / dir | Size | Description |
|------------|------|-------------|
| `analysis_summary.csv` | 6 KB | Per-eps normality/Poisson test p-values |
| `eps_experiment_results.csv` | 19 KB | Eps sweep summary from simulate.py experiment mode |
| `regular_fit.csv` | 5 KB | Aggregated regular Beta-Prime fits (eps 1.10–1.40) |
| `regular_fit_eps*.csv` | ~200 B each | Per-eps regular fit params (31 files, eps 1.10–1.40) |
| `mixture_fit_eps*.csv` | ~280 B each | Per-eps mixture fit params (7 files, eps 1.10–1.40 step 0.05) |
| `merged_fit_params.csv` | 260 B | Merged multi-eps fit output |
| `regression_params.csv` | 192 B | Linear regression coeffs linking fit params to eps |
| `beta_prime_fits.png` | 445 KB | Combined fit visualization |
| `a_vs_eps.png`, `b_vs_eps.png`, `loc_vs_eps.png`, `scale_vs_eps.png` | ~28–31 KB | Parameter trends vs eps |
| `ratio_plots/` | 5 PNGs, ~160 KB each | Batch ratio histograms |
| `N_prime_plots/` | 5 PNGs, ~50 KB each | Batch N' distribution plots |

Results total ~1.9 MB — small enough for regular git (no LFS needed).

## Git LFS

Large files marked for Git LFS in `.gitattributes`:

| Pattern | Count | Total size | Reason |
|---------|-------|------------|--------|
| `simdata/v2/*.csv` | 207 | ~4.3 GB | Simulation raw data, 20–44 MB each |

To track the data after cloning:

```bash
git lfs install
git add simdata/v2/
git commit -m "Add simulation data via LFS"
```

Verify LFS tracking:

```bash
git lfs ls-files
git lfs track
```

## Dependencies

Managed via uv (`pyproject.toml`):

| Package | Used by |
|---------|---------|
| numpy | All scripts |
| pandas | Data I/O (most scripts) |
| scipy | Distribution fitting, optimization, spatial (ConvexHull) |
| scikit-learn | DBSCAN clustering, linear regression |
| matplotlib | All plotting scripts |
| seaborn | `scripts/viz/visualize.py`, `scripts/stats/stat_tests.py` |
| statsmodels | `scripts/viz/visualize.py`, `scripts/stats/stat_tests.py`, `scripts/stats/stat_test2.py` |

Previous `requirements.txt` also listed PySide6 (unused by any script) and conda-local wheel paths — not carried forward.

## Key Concepts

- **λ₀** = N / (π·R²) — baseline point density in the simulation circle
- **λ'** = N' / S' — cluster-local density
- **Density ratio** = λ' / λ₀ — primary variable for distribution fitting
- **eps** — DBSCAN neighbourhood radius; swept from ~0.8 to ~2.86

# Repo State — Comprehensive Overview

*Synthesised May 2026 by Claude (Opus 4.8) from a full read of the code, data, results, and the four prior handoff docs. This is the "what exists and what it means" document. Companion docs: [roadmap.md](roadmap.md) (where to go next) and [../CLAUDE.md](../CLAUDE.md) (working memory).*

---

## 0. One-paragraph summary

The project asks: **given N points in a finite area under complete spatial randomness (CSR), how surprising is a dense-looking cluster that a density-seeking algorithm (DBSCAN) carved out?** A naive Poisson Z-score is invalid because the region was *chosen because it looks dense* (post-selection / look-elsewhere bias). The approach taken so far is **brute-force Monte Carlo**: generate uniform points in a disk, run DBSCAN, record each detected cluster's area `S'` and count `N'`, repeat ~10⁴–10⁶ times per DBSCAN `eps`, sweep `eps` from 0.80 to 2.86, and empirically characterise the null distribution of the **density ratio** `R = (N'/S') / λ₀`. The headline empirical result: `R` is well-described by a **Beta-Prime** distribution whose parameters drift quasi-linearly with `eps`. The headline *unsolved* problem: turning per-`eps` conditional fits `P(R|eps)` into a single, calibrated, `eps`-independent rare-event probability `P(R ≥ r)`.

**Compute cost of the data: ~60 hours on this machine** (perf metric / regeneration gauge — see §6). Do not casually regenerate `simdata/v2/`.

---

## 1. The statistical object (what we are actually measuring)

- Domain: disk of radius `R = 100`, area `S₀ = πR² = 31 415.93`.
- Points: `N = 10 000` drawn uniformly (`r = R√U`, `θ = 2πV`).
- Baseline intensity: `λ₀ = N/S₀ = 1/π = 0.31831` points per unit area.
- DBSCAN with `eps` (swept), `min_samples = 10`.
- Per detected cluster: `N'` = point count, `S'` = **convex-hull area** (`scipy.spatial.ConvexHull.volume` in 2D).
- Filters: `min_cluster_size = 10`, `min_area = 0.5`.
- Local intensity `λ' = N'/S'`; primary statistic **`R = λ'/λ₀`**.

**Crucial nuance (reconciles the two handoff docs).** `simulate_once` returns **every** cluster that passes the filters in an iteration, and these are *pooled* across iterations. So the dataset is the distribution of a **randomly-detected cluster**, *not* the max-over-windows Λ that `clusters_problem.md`/`cluster-detection-handoff.md` argue is the statistically correct object. The Kulldorff likelihood-ratio statistic those docs recommend is **never computed**; only the raw ratio `R` is. This is the single biggest theory-vs-implementation gap (see roadmap §1).

---

## 2. Pipeline and scripts — what each does, inputs, outputs

Entry point: `main.py` (Typer CLI, `uv run cluster-distribution <cmd>`). Note several analysis scripts are **not** wired into the CLI and are run directly (`uv run python <script>.py`).

### 2.1 Simulation

| Script | Role | Inputs | Outputs |
|---|---|---|---|
| `simulate.py` | Parallel DBSCAN MC over an `eps` sweep (`np.arange(0.80, 2.90, 0.01)`). 16-process `Pool`, per-worker seeded RNG. Resumable (reads existing CSV, continues from `max(iteration)`). | hard-coded params in `main()` | `simdata/v2/simulation_data_N10000_radius100_eps{eps:.2f}.csv`; experiment-mode summary `simdata/eps_experiment_results.csv` → lives at `results/eps_experiment_results.csv` |
| `plotter.py` | 2D histogram `S'` vs `N'` for one file (interactive). | one CSV | plot only |

Key `simulate.py` details:
- Output columns: `S_prime` (float), `N_prime` (int), `iteration` (int). Iterations with no valid cluster write a **placeholder** row `S_prime=-1.0, N_prime=-1`.
- RNG fix: parent draws `seeds = rng.integers(0,1e9,batch)`, each worker does `np.random.default_rng(seed)` — this fixed an early bug where forked workers shared RNG state and produced duplicate outputs.
- Early-termination guard: `if int(iteration//batch_size) > clusters_found: return` → stops an `eps` if it yields < ~1 cluster per 1000 iterations. This is why low-`eps` files (< ~0.88) are tiny.
- `main()` also has "experiment mode" book-keeping with two break conditions (`max_S' > 0.25·S₀`, `max_N' > √N = 100`) that in practice were never hit in the recorded range.

### 2.2 Distribution fitting

| Script | Role | Inputs | Outputs |
|---|---|---|---|
| `beta_mix_vs_regular.py` | **Main fitter** (CLI `fit`). Per-`eps` single Beta-Prime fit (all 4 params free; regression line used only as init guess) and optional 2-component mixture (`--mix`). Computes AIC/BIC/KS. | `simdata/v2/`, optional `results/regression_params.csv` | appends `results/regular_fit.csv`; `results/mixture_fit_eps*.csv`; rewrites `results/regression_params.csv` + `{a,b,loc,scale}_vs_eps.png` |
| `mixure_of_betas.py` | (CLI `fit-mixture`, filename typo kept) Beta-Prime mixture with **one component per `eps`**, SLSQP, weights sum to 1. | merged `eps` data | `results/merged_mixture_fit.csv` (note: has a latent bug, §5) |
| `mixture_of_betas2.py` | (CLI `fit-merged`) **`eps`-conditioned global model**: fits 8 regression coeffs so each datum's Beta-Prime params are linear in its own `eps`; one global NLL. | merged `eps`+ratio | `results/merged_fit_params.csv` |
| `floc.py` | `floc` (fixed-location) exploration for Beta-Prime; brute search of `floc` near a regression estimate; ratio≥5 filter. | `simdata/v2/` | `floc.png`, console |

### 2.3 Statistical analysis / tests

| Script | Role | Output |
|---|---|---|
| `stat_tests.py` | (CLI `stats`) Normality (Shapiro, Anderson, KS-vs-normal) on `R`; Poisson chi²/KS on `N'`. Batch histograms. | `results/analysis_summary.csv`, `results/ratio_plots/`, `results/N_prime_plots/` (note: `main()` writes under `data_dir` = `simdata/v2/` but the committed copies are under `results/`) |
| `stat_test2.py` | ECDF + AIC/BIC/KS comparison of Gamma/Lognormal/Weibull/Beta-Prime at `floc∈{0,5}`. | interactive plots |
| `stat_test3.py` | Gamma goodness-of-fit per `eps`; single-Gamma vs Gamma-mixture-by-`N'`. | console + plots |
| `anova.py` | One-way ANOVA + linregress of Gamma shape/scale vs `eps`; three methods (all data / mixture-by-N' / N'=10 only). | console + plots |

### 2.4 Visualization

| Script | Role |
|---|---|
| `visualize.py` | 3D Q-Q across `eps`, 2D density heatmap, 3D surface of `R` vs `eps`. |
| `beta_plot.py` | (CLI `plot`) Overlay saved regular/mixture Beta-Prime PDFs on histograms. |
| `plot_hist.py` | Re-fits + overlays histograms with regular & mixture fits; own `floc(eps)` line `SLOPE=-9.71, INTERCEPT=17.1674`. |

### 2.5 Modules (`modules/`)

| Module | Exports |
|---|---|
| `simv2_data.py` | `load_data` (returns density-ratio array, drops placeholders, requires ≥2000 valid clusters), `sample_data`, `load_fit_parameters`, `load_density_ratio`. |
| `beta_stats.py` | `betaprime_mixture_pdf` (auto-detects #components from param length `5N−1`), `negative_log_likelihood`, `betaprime_pdf/cdf_eps` (params linear in `eps`), `negative_log_likelihood_eps`, `perform_linear_regression`. |
| `common_stats.py` | `compute_aic_bic`, `compute_ks_statistic`. |

---

## 3. Data inventory

### `simdata/v2/` — the expensive artifact
- **207 CSVs**, `eps = 0.80 … 2.86` step 0.01, name `simulation_data_N10000_radius100_eps{eps:.2f}.csv`.
- ~4.3 GB total, ~20–44 MB each, ~1.0–1.1 M rows each (most rows are `-1` placeholders at low `eps`; valid-cluster fraction rises with `eps`).
- Columns: `S_prime, N_prime, iteration`. **No geometry** (no center, radius, hull perimeter, shape) — this is the format's main limitation (see §4, roadmap §4).
- Git LFS tracked (`.gitattributes`). ~60 h compute (§6).
- `simdata/v1/` is empty except `convert.py` (v1→v2: recomputed `N' = round(λ'·S')`, dropped timing columns).

### `results/`
- `regular_fit.csv` — per-`eps` single Beta-Prime fits, `eps = 1.10–1.40` step 0.01 (31 rows). The core result table.
- `regular_fit_eps*.csv` (31), `mixture_fit_eps*.csv` (7, step 0.05) — per-`eps` snapshots.
- `regression_params.csv` — linear `param = slope·eps + intercept` for a,b,loc,scale.
- `merged_fit_params.csv` — 8 coeffs from `mixture_of_betas2.py`.
- `analysis_summary.csv` — per-`eps` (0.90–1.39) means/maxes + normality/Poisson p-values.
- `eps_experiment_results.csv` — per-`eps` (0.81–2.16) avg/max ratio, N, cluster radius, cluster counts.
- `*_vs_eps.png`, `beta_prime_fits.png`, `ratio_plots/` (5), `N_prime_plots/` (5).

---

## 4. What the numbers say (interpretation)

### 4.1 Beta-Prime is the empirical winner
Across `eps = 1.10–1.40`, single Beta-Prime fits `R` with KS statistic **0.0025–0.006** (excellent for n=100 000). Parameters from `regular_fit.csv` and their `eps`-regression (`regression_params.csv`):

| param | eps=1.10 | eps=1.40 | slope | intercept | trend |
|---|---|---|---|---|---|
| a (shape) | 40.9 | 55.9 | **+50.06** | −14.18 | rises strongly with eps |
| b (shape) | 10.0 | 10.9 | +3.78 | 5.69 | nearly flat ~10 |
| loc | 7.42 | 4.03 | **−11.44** | 19.87 | left-shift with eps (R²≈0.99) |
| scale | 2.91 | 1.46 | −4.45 | 7.61 | shrinks with eps |

Mechanistic reading (from the design docs, holds up): `R` is a **ratio of gamma-like quantities** (count accumulation ÷ hull-area/support), and ratio-of-Gammas *is* Beta-Prime. `b ≈ 10` staying pinned near `min_samples`/`min_cluster_size = 10` is consistent with the denominator being governed by the detection threshold.

### 4.2 The right tail is **censored**, not intrinsic — KEY FINDING
`eps_experiment_results.csv` shows `max_ratio ≈ 61–68` at **every** `eps` from 0.81 to 2.16, while `avg_ratio` falls monotonically 40 → 4.4. The constant cap is an **artifact of the filters**:

```
R_max ≈ N_min / (min_area · λ₀) = 10 / (0.5 · 0.31831) = 62.83
```

This equals the observed cap to the decimal. The densest *detectable* cluster is the smallest allowed count (`N'=10`) squeezed into the smallest allowed hull (`S'=0.5`). **Consequence:** any extreme-value / deep-tail modelling on the *current* data is measuring the `min_area` knob, not Poisson geometry. The `loc` parameter (~4–7) is likewise a soft floor tied to the smallest ratios that survive filtering. Both bounds are *algorithmic*, and both move if you change `min_area`, `min_cluster_size`, or `min_samples`.

### 4.3 N' is heavily truncated
`N'` mean ≈ 10.2–10.6, mode at 10, geometric-like decay (P(N'=10)≈0.68, P(11)≈0.15, …). DBSCAN only ever returns the **upper tail** of an underlying count process — rarefactions are invisible. Poisson GoF on `N'` fails by ~50 orders of magnitude (expected: `N'` is detection-conditioned and threshold-truncated, not a free Poisson count).

### 4.4 Normality firmly rejected
`analysis_summary.csv`: Shapiro/KS p-values ~1e-40 to 1e-55 throughout. `R` is continuous, right-skewed, bounded below (~loc) and censored above (~62.8). A Gaussian Z-score is meaningless here.

### 4.5 eps-dependence is real
Histograms shift left and peak higher as `eps` grows (peak ~0.075@R≈20 at eps=1.10 → ~0.15@R≈12 at eps=1.40). So there is **no** single `eps`-free distribution; `eps` is a genuine nuisance dimension. The useful working window is `eps ≈ 0.9–1.4` (below ~0.84 ≈ no clusters; above ~2.86 clusters hit area limits). Cluster yield: ~35 k valid clusters at eps=1.10 → ~1.2 M at eps=1.40 (stable fits need ≳10 k).

### 4.6 Gamma vs Beta-Prime vs mixtures
- Exponential/Weibull ruled out. Gamma & lognormal decent but Gamma under-peaks.
- 2-component Beta-Prime mixture (`beta_mix_vs_regular.py --mix`) **collapses to two identical components** (`mixture_fit_eps1.10.csv`: both components equal, α=0.5, KS≈1.0 — degenerate). → no distinct modes; single Beta-Prime already absorbs the structure. The earlier "mixture over N' modes" hypothesis is superseded.

---

## 5. Known bugs / caveats in the code

1. **`mixure_of_betas.py` log-likelihood bug**: line ~182 calls `negative_log_likelihood(fitted_params)` with **one argument** (missing `data`) — `negative_log_likelihood(params, data)` requires two. AIC/BIC there are unreliable / would error. The KS uses a correct `mixture_cdf`, but AIC/BIC do not.
2. **Mixture PDF used as CDF**: historically `kstest` was fed a mixture *PDF* as if it were a CDF (still present in `beta_mix_vs_regular.py` mixture branch and `plot_hist.py`: `cdf_fitted = lambda x: betaprime_mixture_pdf(...)`). KS for a mixture needs `F(x)=Σ wᵢFᵢ(x)`. Single-component fits are fine (they use `common_stats.compute_ks_statistic` with the real CDF).
3. **Degenerate mixture**: see §4.6 — the two-component fit is not identifiable; don't trust its AIC/BIC.
4. **KS p-values ≈ 0 at large n**: with n=10⁵–10⁶, KS p-values are uninformative; judge by KS *statistic* magnitude and visual diagnostics (the docs already flag this).
5. **`stat_tests.py` output path**: writes plots/summary under `data_dir` (`simdata/v2/`) though the committed copies are in `results/`. Cosmetic.
6. **`eps`-conditioned global KS** (`mixture_of_betas2.py`) reports KS≈0.43 — large because it's computed against a single marginal ECDF while each datum has its own conditional CDF; not a like-for-like KS. The model fits `P(R|eps)`, which is *not* the `eps`-independent marginal that the project actually wants (see roadmap §2).
7. **`.python-version` pins 3.14** but `pyproject` requires ≥3.11; environment is uv-managed.

---

## 6. Reproduction & cost

```bash
uv sync
uv run cluster-distribution simulate      # regenerates simdata/v2 — ~60 h on this machine
uv run cluster-distribution fit           # per-eps Beta-Prime → results/regular_fit.csv
uv run cluster-distribution stats          # normality/Poisson summary
uv run python anova.py | stat_test3.py | ... # direct-run analysis scripts
```

**`simdata/v2/` ≈ 60 hours of compute on this exact machine** (user-reported, May 2026). Treat it as a precious artifact and a benchmark: it is the empirical null. Regenerate only with intent (e.g. to add geometry columns — roadmap §4). Clone without pulling 4.3 GB: `GIT_LFS_SKIP_SMUDGE=1 git clone …`.

---

## 7. Provenance of the four prior docs

- `clusters_problem.md` — fresh design-time analysis (Claude, no data seen). Recommends Kulldorff LR + MC-replay + GPD tails. **Theory north star.**
- `cluster-detection-handoff.md` — **the most detailed empirical handoff** (o3/GPT lineage; salvaged May 2026 after an accidental duplicate was overwritten). NB: the file is wrapped in a Python writer stub — the real content is the markdown inside the `content = r"""…"""` heredoc. Unique material vs v2: §8.1 ANOVA-needs-replicates, §12.1 four explicit null-score definitions (conditional / mixture / **conservative envelope** `S_env(r)=maxᵢ Sᵢ(r)` / extreme-over-eps), §17 rarefaction asymmetry (compressions ~+3 ↔ rarefactions ~−4), §18.2 the open "why max ratio ~60–70" question — **now answered** by the `min_area` censoring derivation in §4.2 above.
- `cluster_density_null_model_handoff_v2.md` — the other detailed empirical lab notebook (o3/GPT lineage). Heavy overlap with the above; most beta-prime parameter history lives here.
- `overview.md` — file/script catalog (accurate; this doc supersedes/expands it).
- Original problem: Newton, 2020, [math.stackexchange 3626685](https://math.stackexchange.com/questions/3626685). Repo born 2024 (pre-agentic-tooling, hand-edited from o3 output).

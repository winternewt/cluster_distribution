# CLAUDE.md — Working Memory for `cluster_distribution`

> Exhaustive project memory so a fresh instance can be productive in one read. Authored May 2026 (Opus 4.8) from a full code+data+docs sweep. If something here disagrees with the code, **the code wins** — verify before trusting. Companion long-form docs: `docs/repo_state.md` (current state), `docs/roadmap.md` (next steps). This file is the compressed union of all four `docs/*.md` plus things only visible in the code/data.

---

## ⚡ Analytic breakthrough (branch `analytic-night`, see `docs/analytic_findings.md`)

The "fundamental result beneath Beta-Prime" exists and is proven on the data:
- **The eps-dependence factorizes:** `R(eps) = scale(eps)·X`, `scale ∝ eps^(−2.205)`, `X` eps-invariant to ~1%. Rescaling `R̃=R·eps²` collapses cross-eps spread 52%→5%. (`−2`=geometry `S'∝eps²`; `−0.205` residual = fixed-`min_samples` effect, ~44% N'-occupancy broadening + ~56% per-n area loosening.)
- **Beta-Prime is derived & demoted:** `R|N'=n = n/(λ₀S')` is *exactly* scaled-inverse-gamma with shape = Gamma-shape of hull area `S'|n`. Marginal = N'-mixture of these (mode n=10), fits as well as Beta-Prime. **Beta-Prime params are non-identifiable** (two basins both fit KS~0.02) — never read mechanism off `b`. The robust quantity is hull-shape `k(10)≈20.5`.
- **First-principles geometry:** `S'|n` = convex-hull area of `n` uniform points in an eps-disk (CSR+Poisson conditioning); matches data ~1%, chaining +8%. Leading constant **derived: `C = N_min/f·k/(k−1) ≈ 25`**, `f≈0.42` fill fraction, no free params.
- **eps-independent scorer shipped — master corrected 2026-06-07 (`docs/RCA.md` §7):** the master on `R̃=R·eps^α(ε)` (α(ε)=2.031525+0.258273·ln ε) is a **SHIFTED inverse-gamma, integer shape 10 = min_samples**: `SF(R̃)=P(10, 157.7035/(R̃−7.5091))`. The location shift is essential — `R|n` is bounded away from 0 (hull area of an eps-connected cluster bounded above), so zero-loc families (the earlier inv-gamma(20.5), the log-logistic webapp master) mis-centre z by ~0.07σ. Shape 10 = the `a→∞` limit of legacy Beta-Prime(a≈46,b≈10). Pooled KS≈0.005 over eps 1.00–1.60; |median_z|≤0.02 per eps. Same constants in `analysis/scorer.py`/`scorer_master.json`, `modules/cluster_detector.py` (`master_sf`, `score_clusters`), `webapp/master.js` (exact Poisson-sum SF). **Never fit `floc=0` to `R̃`.** Evidence: `analysis/zcal_*.py/csv`.
- **Master parameters decoded (`docs/RCA.md` §8):** the loc is **DBSCAN's certification floor** — an n=min_samples cluster fits in one core's eps-ball, max hull = regular (m−1)-gon ⇒ `R̃_floor = m·(1+2π²/3(m−1)²) ≈ 10.87`; observed global min **10.92** over 1.2M clusters. Master mean `loc+scale/(shape−1) = 25.03` = the derived no-free-param `C≈25`. Shape 10 is the only *effective* param (N'-mixture roughly halves k(10)≈20.5; shape=min_samples link unproven — test with a min_samples sweep). **N-invariance confirmed** (`analysis/ncheck.py`): at fixed λ₀ (radius √N), N=10k/20k/40k → identical master & calibration, yield ∝ N. `R̃` is a purely local observable of (λ₀, eps, min_samples).
- **Look-elsewhere quantified:** typical CSR cluster = "~6σ" under naive Kulldorff/Wilks (overstated ~10⁸×); only MC-replay calibrates. LR beats raw `R` (size-weights: equal-`R`, larger-`N'` is rarer).
- **Bug fixes (committed):** mixture PDF-used-as-CDF KS; missing NLL data arg; **4-vs-5 stride in `betaprime_mixture_pdf`** — likely the cause of the repo's degenerate-mixture results.

The two genuinely fundamental objects: **the hull-area law of `n` uniform disk points** + **the `N'`-occupancy distribution `P(N'=n)`**. Everything else is downstream. This supersedes the framing below where it says "beta-prime parameters drift linearly with eps" (true, but it's just the `eps²` rescaling in awkward coordinates).

---

## TL;DR (read this first)

- **Problem.** Detect dense clusters in a random (Poisson/CSR) point field and assign them an *honest* null probability. Naive Poisson Z-scores are invalid once a density-seeking algorithm (DBSCAN) *chooses* the region — that's post-selection / look-elsewhere bias. Originally posed by Newton in 2020 ([math.SE 3626685](https://math.stackexchange.com/questions/3626685)).
- **Method to date.** Monte-Carlo the null: uniform points in a disk → DBSCAN → record each cluster's `(S'=hull area, N'=count)` → study the **density ratio `R = (N'/S')/λ₀`**. Sweep DBSCAN `eps` 0.80→2.86.
- **Headline empirical result.** `R | eps` ≈ **Beta-Prime**(a,b,loc,scale), KS ~0.003–0.006, with parameters drifting quasi-linearly in `eps` (`a` rises ~+50/eps, `loc` falls ~−11/eps, `b≈10` flat, `scale` shrinks).
- **Headline open problem.** Convert per-`eps` `P(R|eps)` into one calibrated `eps`-independent rare-event score `P(R≥r)` + `Z_equiv=Φ⁻¹(1−p)`.
- **Biggest gotcha I found.** The observed "tail bound" `max R ≈ 60–70` is **not physics** — it's `R_max = N_min/(min_area·λ₀) = 10/(0.5·0.31831) = 62.83`, a *censoring artifact* of the `min_area`/`min_cluster_size` filters. Don't fit extreme-value tails to current data.
- **Biggest theory gap.** The data records *pooled per-cluster* `R`, but the statistically correct object (per the design doc) is **`Λ = max over windows` of the Kulldorff LR**, obtained by **MC-replaying the whole pipeline**. LR is never computed; only raw `R`.
- **`simdata/v2/` = ~60 h compute on this machine.** Precious artifact. Don't casually regenerate; when you do, add geometry columns (see §Roadmap).

---

## Canonical numbers (memorize)

| quantity | value | note |
|---|---|---|
| N (points) | 10 000 | hard-coded |
| radius R | 100 | disk |
| S₀ = πR² | 31 415.93 | domain area |
| λ₀ = N/S₀ | **0.318310** (=1/π) | baseline intensity |
| min_samples (DBSCAN) | 10 | core-point threshold |
| min_cluster_size | 10 | post-filter |
| min_area | 0.5 | post-filter (hull area) |
| ratio cap | **62.83** = N_min/(min_area·λ₀) | censoring artifact, matches observed max |
| useful eps range | ~0.9–1.4 | modelled window; <0.84 no clusters, >2.86 hits area cap |
| stable-fit cluster count | ≳10 000 (≥2000 min) | `load_data` requires ≥2000 |
| sample size for fits | 100 000 (subsample) | `sample_data`, seed 42 |
| R ≈ λ'·π | since R=λ'/λ₀ and λ₀=1/π | handy conversion |

Beta-Prime fit (`results/regular_fit.csv`, single-component, free loc):

| eps | a | b | loc | scale | KS |
|---|---|---|---|---|---|
| 1.10 | 40.9 | 10.0 | 7.42 | 2.91 | 0.0025 |
| 1.20 | 45.9 | 10.0 | 6.12 | 2.16 | 0.0027 |
| 1.30 | 50.9 | 10.6 | 4.97 | 1.78 | 0.0054 |
| 1.40 | 55.9 | 10.9 | 4.03 | 1.46 | 0.0060 |

`regression_params.csv`: a: slope 50.06 / int −14.18 · b: 3.78 / 5.69 · loc: −11.44 / 19.87 (R²≈0.99) · scale: −4.45 / 7.61.
(Two regression vintages exist — the per-eps `regression_params.csv` above, and the merged-likelihood `merged_fit_params.csv` with very different coeffs, e.g. scale_slope≈0.005. They optimize different objectives; don't conflate.)

---

## Mental model of the math

1. **Why a ratio, not a rate.** Raw `λ'` is scale-uninformative (3 points in a tiny triangle → huge λ', meaningless). Dimensionless `R = λ'/λ₀` is the right observable (scale-invariance of Poisson). The `N'≥3`/triangulation guard removes `λ'∈{0,∞}` but not the selection bias.
2. **Why Beta-Prime.** Ratio of two independent Gammas `Γ(a,θ)/Γ(b,θ) ~ BetaPrime(a,b)`. `R` behaves like (count-accumulation Gamma)/(hull-area/support Gamma). `b≈10` pinned near `min_samples=10` is consistent with the denominator being threshold-governed. Same family as F-distribution and the large-count limit of the Poisson LR — so rate-ratio, variance-ratio, and LR framings are mutually consistent. **Diagnostic:** Beta-Prime fitting well ⇒ the operative statistic is a *ratio/contrast*, not a bare rate. A bare selected `λ'` would trend to Fréchet/GEV instead.
3. **Why selection breaks naive scoring.** DBSCAN returns (near) the *max* of `λ'` over the family of windows it could draw, and hugs the boundary to the points. So `N'/S'` is not a draw from `Poisson(λ₀S')/S'`, and `S'` is itself random/data-adaptive. Per-window Poisson-Z **and** per-window Bayesian Beta posteriors are *both* overconfident by the same mechanism.
4. **Correct null object.** `Λ = max_{Z∈W} T(Z)` under CSR, with `T` the **Kulldorff scan LR** and `W` = windows DBSCAN can actually return. Condition on total `N` ⇒ inside-count `N' ~ Binomial(N, |Z|/|S|)`.
   - LR: `LR(Z) = (n/μ)^n · ((N−n)/(N−μ))^(N−n) · 1[n/μ>1]`, `μ = N|Z|/|S|`.
5. **How to get the null** (three routes, design doc §2.3):
   - **(A) MC-replay the full pipeline** under CSR, record per-field `max LR`, rank observed against it. Field standard (SaTScan). Valid for any selector because selection is replayed. **← primary recommendation.**
   - **(B) Analytic trials factor** (Gross–Vitells / Euler-characteristic / Poisson-clumping): only for *structured* window families; **no clean closed form for DBSCAN blobs**.
   - **(C) Bayesian generative mixture**: homogeneous-Poisson + cluster component, with cluster location/shape/count as **latent vars integrated over** (RJMCMC / DP mixture). The prior over where a cluster *could* be **is** the multiplicity correction. Gamma conjugate for rate; Beta-Binomial for inside-fraction `Pr(n'|a,b)=C(N,n')B(a+n',b+N−n')/B(a,b)`. Most principled, heaviest.
   - **Moral:** selection must be *replayed* (A) or *integrated over* (C). Per-window calc is only valid for **pre-registered** windows (which is exactly why the coarse fixed-grid method was legitimate).
6. **Tails.** Beta-Prime is Fréchet-domain (polynomial right tail). For deep-tail p-values the design doc says fit **GPD / peaks-over-threshold**, not brute 10⁶ sims. **BUT** — see the censoring caveat: current data's tail is clipped at 62.83 by `min_area`, so GPD on it measures the filter, not the process. Must relax `min_area` / use LR / model the censoring first.
7. **Scoring.** Don't use Gaussian `(x−μ)/σ` (distribution is skewed, censored). Use `p = SF(r)` then `Z_equiv = Φ⁻¹(1−p)`. For an eps range: mixture `Σ wᵢ SFᵢ(r)` or conservative envelope `maxᵢ SFᵢ(r)`.

---

## The eps-weighting question (the actual unfinished modelling task)

`P(R) = Σᵢ wᵢ P(R|epsᵢ)`. The weights encode *what question you're asking*:
- **uniform-eps**: "eps drawn uniformly from search range, then a cluster."
- **cluster-count weighted**: "a random detected cluster pooled across all eps runs" (matches how the data was pooled).
- **conservative envelope** `S_env(r)=maxᵢSᵢ(r)`: weighting-free, never overstates significance — **recommended safe default**.
- **max-over-eps EVD**: "detector sweeps eps, reports the most extreme" — the true look-elsewhere null if eps is tuned after seeing data → but then just use route (A) replay over (eps, windows).

These are genuinely different answers; any deliverable must state which it computes.

---

## Repo map (files → role)

**Sim:** `simulate.py` (parallel DBSCAN MC + eps sweep; writes `simdata/v2/*.csv`; resumable; 16 procs; per-worker seeded RNG) · `plotter.py` (S'×N' 2D hist).
**Fit:** `beta_mix_vs_regular.py` (CLI `fit`; per-eps single + optional `--mix` mixture → `results/regular_fit.csv`, `regression_params.csv`) · `mixure_of_betas.py` (CLI `fit-mixture`; one component per eps; **has LL-arg bug, see Gotchas**) · `mixture_of_betas2.py` (CLI `fit-merged`; eps-linear global model → `merged_fit_params.csv`) · `floc.py` (floc exploration).
**Stats:** `stat_tests.py` (CLI `stats`; normality + Poisson → `analysis_summary.csv`, plots) · `stat_test2.py` (Gamma/LogN/Weibull/BetaPrime ECDF compare) · `stat_test3.py` (Gamma GoF per eps; mixture-by-N') · `anova.py` (ANOVA + regression of Gamma params vs eps; 3 methods).
**Viz:** `visualize.py` (3D QQ, heatmap, surface) · `beta_plot.py` (CLI `plot`; overlay saved fits) · `plot_hist.py` (refit+overlay).
**Modules:** `modules/simv2_data.py` (data I/O: `load_data`, `sample_data`, `load_fit_parameters`, `load_density_ratio`) · `modules/beta_stats.py` (mixture PDF/NLL, eps-linear betaprime, `perform_linear_regression`) · `modules/common_stats.py` (`compute_aic_bic`, `compute_ks_statistic`).
**Entry:** `main.py` (Typer; `uv run cluster-distribution {simulate,fit,fit-mixture,fit-merged,stats,visualize,plot}`). Several analysis scripts (`anova`, `stat_test2/3`, `floc`, `plot_hist`, `visualize`-internals) are run directly, not all via CLI.
**Data:** `simdata/v2/` (207 CSVs, cols `S_prime,N_prime,iteration`, ~4.3 GB, LFS, **~60 h compute**) · `simdata/v1/convert.py` (legacy v1→v2; v1 dir empty).
**Results:** `regular_fit.csv` (core table, eps 1.10–1.40), `regression_params.csv`, `merged_fit_params.csv`, `analysis_summary.csv`, `eps_experiment_results.csv`, `*_vs_eps.png`, `ratio_plots/`, `N_prime_plots/`.
**Docs:** `repo_state.md`, `roadmap.md` (these two are the curated pair), `clusters_problem.md` (=theory north star, dup of `cluster-detection-handoff`'s *intent* but different content now), `cluster-detection-handoff.md` (richest empirical handoff; **content is inside a Python heredoc stub**), `cluster_density_null_model_handoff_v2.md` (empirical notebook), `overview.md` (catalog).

---

## Data format & conventions (don't trip on these)

- Columns: `S_prime` (float, convex-hull area), `N_prime` (int), `iteration` (int).
- **Placeholder rows** `S_prime=-1.0, N_prime=-1` mark iterations with no valid cluster. **Always filter** `(S_prime!=-1)&(N_prime!=-1)` first. Most rows at low eps are placeholders.
- **Multiple clusters per iteration are possible** and all are recorded → data is "all detected clusters pooled" (≈ random-detected-cluster), *not* max-per-field.
- `R = (N'/S')/λ₀` computed downstream; `λ₀=N/(πR²)`.
- Some scripts apply a `ratio≥5` body filter (`floc.py`, `stat_test2.py`); the `loc` param (~4–7) reflects this soft floor.
- `load_data`/`load_density_ratio` skip eps with <2000 valid clusters and subsample to 100 000 (seed 42).

---

## Empirical facts to preserve (the lab-notebook distillate)

1. DBSCAN-on-noise produces *structured* cluster distributions, not chaos.
2. `eps` strongly changes distribution **shape** (peak shifts left + grows taller as eps↑), not just yield.
3. `S'` mean & max grow monotonically with eps (bigger neighborhoods absorb more points/area).
4. `N'` concentrated at the threshold: P(10)≈0.68, P(11)≈0.15, P(12)≈0.08, decaying; mean ≈10.2–10.6. DBSCAN sees only the **upper tail** of an underlying count process.
5. `R` is right-skewed: steep left rise, long right tail, censored above at 62.83.
6. Normality **rejected** (p~1e-40…1e-55). Poisson-on-N' **rejected** (truncated/selection-conditioned).
7. Gamma = decent but systematically biased (peak too low, skewed right). Exponential/Weibull ruled out. Lognormal close, usually loses to Beta-Prime with free loc.
8. Beta-Prime per-eps = excellent. Free/fitted `loc` beats `floc=0` (and `floc=5` beats `floc=0`).
9. **2-component Beta-Prime mixture collapses to identical components** (degenerate; see `mixture_fit_eps1.10.csv`, KS≈1.0 from the PDF-as-CDF bug). Single component suffices. The earlier "mixture over N' modes" idea is superseded.
10. Per-eps params strongly correlate with eps (linear, high R²).
11. Cluster yield: ~35k @eps1.10 → ~1.2M @eps1.40 (`eps_experiment_results.csv` / `analysis_summary.csv`).
12. `eps_experiment_results.csv` covers eps 0.81–2.16; `avg_ratio` falls 40→4.4 monotonically; `max_ratio` pinned ~61–68 (the censoring artifact).
13. Rarefactions ≠ mirror of compressions (old grid Z-score gave asymmetric +3 ↔ −4); needs separate treatment & localiser.

---

## Gotchas / bugs / pitfalls (verify before relying)

- **`min_area` censors the tail** at 62.83 — the #1 misread-as-physics result. (`docs/repo_state.md` §4.2; answers handoff §18.2.)
- **`mixure_of_betas.py` LL bug**: `negative_log_likelihood(fitted_params)` missing the `data` arg → AIC/BIC there unreliable.
- **Mixture PDF used as CDF**: `kstest(data, lambda x: betaprime_mixture_pdf(...))` is wrong; KS for mixtures needs `Σwᵢ Fᵢ`. Present in mixture branches of `beta_mix_vs_regular.py` & `plot_hist.py`. Single-component fits use the real CDF and are fine.
- **KS p-values ≈ 0 at large n** — meaningless; judge by KS *statistic* + eyeball, not p.
- **AIC/BIC not comparable across different datasets** (e.g. all-data vs N'=10-only).
- **ANOVA needs replicates**: one fit per eps ⇒ ANOVA across eps is ill-posed; do K-subsample fits per eps for dispersion (`anova.py` has this limitation; handoff §8.1).
- **Merging eps ≠ a single distribution** — it's a mixture; weights must be declared.
- **`mixture_of_betas2.py` global KS≈0.43** is not a like-for-like KS (each datum has its own conditional CDF); the model is `P(R|eps)`, *not* the wanted marginal.
- **`stat_tests.py` writes under `data_dir`** (simdata/v2) though committed plots live in `results/`. Cosmetic.
- **Two regression-param vintages** (`regression_params.csv` vs `merged_fit_params.csv`) — different objectives; keep straight.
- **`.python-version` says 3.14, pyproject says ≥3.11** — uv-managed env reconciles.

---

## Roadmap (priority order — full detail in `docs/roadmap.md`)

1. **(now, cheap)** Survival/`Z_equiv` scorer from `regular_fit.csv`: mixture SF **and conservative envelope** `maxᵢSFᵢ`, overlaid on empirical merged SF. Output `ratio→{p_mix,z_mix,p_env,z_env,p_emp}`. Body-only (don't extrapolate past ~55 due to censoring). ← closes the named "immediate next step".
2. **(now, cheap)** Bake the censoring caveat into all tail reporting; add Tobit-style or LR-based handling.
3. **★ (compute, high value)** Re-simulate with **geometry columns** (centroid, r_center, dist-to-boundary, edge-safe flag, r_eff=√(S'/π), hull perimeter, compactness, PCA axes/aspect, **LR & per-field max-LR**) + a run-manifest (params/seed/git-hash/wall-clock). Budget vs 60 h baseline.
4. **★ Compute the Kulldorff LR and per-field max-LR**; calibrate detection p-values by **MC-replay** over (eps, windows). This dissolves the eps-weighting puzzle for detection.
5. **(after data)** Edge-effect split by dist-to-boundary; refit. Compare hull vs miniball vs α-shape vs best-circular-aperture for `S'` (each = a different null, must be replayed).
6. **(optional, heavy)** Bayesian latent-configuration model (route C) as the principled capstone.
7. **Rarefactions**: separate object, separate localiser (grid/KDE deficit, OPTICS reachability), separate null — out of current scope.

**Tried & don't relitigate:** 2-comp Beta-Prime mixture (degenerate) · N'-mode mixture (superseded) · Exponential/Weibull (ruled out) · Gaussian Z / normality (rejected) · eps-conditioned global regression (models P(R|eps), not the marginal) · "max ratio 60–70 as physical bound" (it's the min_area artifact).

---

## How to run

```bash
uv sync
uv run cluster-distribution simulate    # regenerate simdata/v2 — ~60 h on this machine (avoid; add geometry first)
uv run cluster-distribution fit          # per-eps Beta-Prime → results/regular_fit.csv
uv run cluster-distribution stats        # normality/Poisson summary
uv run cluster-distribution plot         # overlay saved fits
uv run python anova.py                   # (and stat_test2.py / stat_test3.py / floc.py / visualize.py) run directly
# clone w/o 4.3GB: GIT_LFS_SKIP_SMUDGE=1 git clone <url>
# push code w/o LFS upload: GIT_LFS_SKIP_PUSH=1 git push -u origin HEAD
```

Stack: numpy, pandas, scipy, scikit-learn (DBSCAN), matplotlib, seaborn, statsmodels, typer. Python ≥3.11.

---

## Provenance & meta

Repo born 2024, hand-edited from o3 output (pre-agentic-tooling). Author: Newton Winter (winternewt / isoutthere@gmail.com / nikolay.usanov@uni-rostock.de). The four handoff docs are AI-assisted summaries of long o3/GPT + Claude dialogues; `clusters_problem.md`=`cluster-detection-handoff.md` were *intended* duplicates but the latter was re-salvaged with richer content May 2026. When in doubt about a "fact" from a handoff, check it against `simdata/`/`results/` or the code — the handoffs occasionally state superseded hypotheses as if current (e.g. the 60–70 tail, the N'-mode mixture).

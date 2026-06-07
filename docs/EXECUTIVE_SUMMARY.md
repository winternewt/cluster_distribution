# Executive Summary — Detecting Density Clusters Against Poisson Noise

*Status report, May 2026. Detail: [analytic_findings.md](analytic_findings.md). Detector: `modules/cluster_detector.py`.*

## Problem
Given N points in a bounded 2D region, decide whether an algorithm-detected local over-density is real signal or an ordinary fluctuation of complete spatial randomness (CSR / homogeneous Poisson). The hard part is **post-selection**: a density-seeking algorithm (DBSCAN) picks the region *because* it looks dense, so a naive per-window Poisson/Z test is invalid (look-elsewhere effect). Original framing: Newton, 2020.

## What was established

1. **The eps-dependence factorizes (and the "anomalous exponent" runs).** The detected density ratio `R=(N'/S')/λ₀` obeys `R(eps) = scale(eps)·X`, where `X` is an **eps-invariant master shape** (quantile ratios constant to ~1%) and `scale` is a power law whose exponent **runs**: exactly **−2 as eps→0** (pure geometry, eps is the only length scale), drifting to ~−2.5 by eps=2 (fixed `min_samples` vs growing eps). Rescaling collapses cross-eps spread from 52% to ~5%.

2. **The empirical "Beta-Prime" fit is not fundamental — it's derived.** `R|N'=n = n/(λ₀S')` is *exactly* a scaled inverse-gamma whose shape is the Gamma-shape of the cluster's convex-hull area; the marginal is the N'-occupancy mixture of these. This reproduces the data as well as Beta-Prime, whose parameters are non-identifiable.

3. **It reduces to elementary stochastic geometry.** The hull-area law is that of `n` uniform points in an eps-disk (CSR + Poisson conditioning), matching data to ~1%. The leading amplitude is derived with **no free parameters**: `C = N_min/f·k/(k−1) ≈ 25` (hull fill-fraction `f≈0.42`, shape `k≈20.5`).

4. **The look-elsewhere effect, quantified.** A *typical* CSR-detected cluster scores **"~6σ" under naive per-window Kulldorff/Wilks** scoring (overstated ~10⁸×). Only replaying the full pipeline (Monte Carlo) calibrates.

5. **The right statistic is the scan likelihood-ratio, not the density ratio.** Raw `R` *fails* on extended signal: a 5.8×-density splat scores `z=−6` (it looks *less* anomalous than tight noise blobs), while the size-weighted Kulldorff LR — and a KDE peak — flag it at high significance.

6. **The tail is an *integrable* power law; the censorship was a red herring.** `R` has tail index **α≈7** (three methods agree). Since α>2 the distribution is proper with finite moments, so the `min_area` cutoff is unnecessary — confirmed by an uncensored resim where it removed only 0.006% of clusters. The deep tail is reached by extrapolating the α≈7 power law (the Kulldorff LR *is* the large-deviations entropy rate), not brute force.

7. **A calibrated, two-arm detector.** `score(points)→{detections, p, z}` combines an LR arm (tight clumps) and a KDE arm (extended over-densities), each calibrated against the per-field null so the look-elsewhere correction is built in. KDE uses a **hybrid** null: analytic random-field theory where the smoothed field is Gaussian (coarse scale), Monte Carlo where it isn't (fine scale). Validated: silent on noise, correct localization and significance on injected tight and extended signal.

8. **A closed-form, z-calibrated per-cluster master (2026-06-07).** One analytic null covers every eps: `R̃ = R·ε^α(ε)` with `α(ε)=2.03+0.26·ln ε`, scored by a **shifted inverse-gamma with integer shape 10 = min_samples** — `SF(R̃)=P(10, 157.70/(R̃−7.51))`, exact via two Poisson sums (no special functions). Pooled KS≈0.005 over eps 1.00–1.60; |median_z|≤0.02 per eps. The location shift is what every zero-loc candidate (log-logistic, plain inv-gamma) missed — they mis-centre z by ~0.07σ (root-cause analysis: `docs/RCA.md`). Shape 10 is the `a→∞` limit of the legacy Beta-Prime fits, closing the loop on the original empirical result.

## Deliverable
An importable, calibrated, open detector (`modules/cluster_detector.py`) with a pluggable null model — a drop-in for scoring over-densities in real point-field data, replacing the invalid "threshold the density ratio" approach. Includes `score_clusters()`: instant per-cluster ratings against the analytic shifted inv-gamma(10) master (no MC; same constants as the live demo `webapp/`).

## Limitations
- Calibrated for homogeneous CSR on a disk; heterogeneous backgrounds require a custom null (supported via `null_generator`).
- Deep-tail (z≳5) significance is extrapolated, not directly sampled.
- No real-data application yet; results are on simulated CSR + injected signal.

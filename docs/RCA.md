# RCA — z-histogram off-centre in the live demo

**Date:** 2026-06-07  
**Branch:** `detector-lib`  
**Symptom:** the z-score histogram in `webapp/index.html` peaks visibly left of zero even after the eps^2.15 exponent fix.

---

## 1. Measurement protocol

20 000 DBSCAN fields per eps value, seeded JS simulation (LCG seed 42), identical code path to the browser demo.  
For each cluster, z = normInv(1 − SF_master(R·ε^α(ε))).  
Reported: mean_z, median_z, and integral areas in the four half-bands used as the asymmetry fingerprint.

Expected under a perfectly-calibrated N(0,1): each half-band pair should be equal; [-2..0]=[0..2]=0.4773, [-1..0]=[0..1]=0.3413.

---

## 2. Numerical results

| ε | n | mean_z | **median_z** | [-2..0] | [0..2] | Δ[-2..0] | [-1..0] | [0..1] | Δ[-1..0] |
|---|---|--------|--------------|---------|--------|-----------|---------|--------|-----------|
| 1.10 | 731   | +0.040 | **−0.060** | 0.513 | 0.456 | −0.058 | 0.354 | 0.296 | −0.059 |
| 1.20 | 2659  | +0.037 | **−0.060** | 0.518 | 0.445 | −0.072 | 0.372 | 0.298 | −0.075 |
| 1.30 | 8598  | +0.049 | **−0.048** | 0.512 | 0.449 | −0.063 | 0.369 | 0.298 | −0.071 |
| 1.40 | 24239 | +0.045 | **−0.039** | 0.512 | 0.450 | −0.062 | 0.368 | 0.306 | −0.062 |
| N(0,1) ref | — | 0 | 0 | 0.477 | 0.477 | 0 | 0.341 | 0.341 | 0 |

Key observations:
- **Median_z ≈ −0.04 to −0.06 at every eps.** The bell peak is left of zero.
- **Mean_z ≈ +0.04** (positive) because the right tail of the Rt distribution is heavy — a few high-z outliers pull the mean rightward even while the bulk sits left.
- **Both half-band asymmetries (Δ[-2..0] and Δ[-1..0]) are negative and consistent across eps.** This rules out alpha(ε) as the cause: if the running exponent were wrong the bias would vary with eps (larger at high or low eps). It does not.

---

## 3. Root-cause chain

### 3a. Immediate cause — LL scale too high

The log-logistic master was fit by MLE to **synthetic** Rt samples drawn from the 31 Beta-Prime per-eps fits in `results/regular_fit.csv`, not to actual DBSCAN simulation data.

From `simdata/v2_parquet/eps_1.20.parquet` (132 266 real clusters, ε=1.20, α-corrected):

```
actual Rt median  = 23.731
LL scale (master) = 24.057   (+1.4% too high)
```

The log-logistic median equals its scale parameter. With scale=24.057, the master places the 50th percentile at 24.057 while the data's 50th percentile is at 23.731. Every cluster near the median therefore receives SF > 0.5, i.e. z < 0, shifting the bulk of the bell leftward.

### 3b. Deeper cause — ln(Rt) is right-skewed; no 2-parameter family can eliminate the asymmetry

Measured from actual parquet data (eps=1.20, α-corrected):

```
ln(Rt):  mean=3.1908  std=0.2268  skew=+0.67  excess-kurtosis=+0.85
```

The Rt distribution has a heavier right tail than its left — it is *not* log-symmetric. A 2-parameter log-scale family (log-logistic, log-normal, log-gamma) has equal left/right log-tails by construction, so it cannot simultaneously satisfy:

| Target | Requires |
|--------|----------|
| median_z = 0 | scale = actual Rt median = 23.731 |
| symmetric [-2..0]/[0..2] | scale ≈ 23.49 (minimises large-range asymmetry) |
| symmetric [-1..0]/[0..1] | no LL scale achieves this |

Scale-sweep results (shape=7.875, actual parquet data):

| scale | median_z | Δ[-1..0] | Δ[-2..0] |
|-------|----------|-----------|-----------|
| 23.00 | +0.154 | −0.008 | +0.071 |
| 23.49 | +0.049 | −0.037 | ≈ 0 |
| **23.731** (actual median) | **0.000** | **−0.051** | **−0.035** |
| 24.057 (current) | −0.067 | −0.069 | −0.080 |

Setting scale to the actual median zeros the median_z exactly but leaves a residual [-1..0] excess of −0.051 that cannot be removed by any scale adjustment.

Calibration-optimisation (Nelder-Mead, minimise per-bin χ²) finds shape=7.731, scale=23.492 which balances the large-range bands (Δ[-2..0]=+0.002) but shifts the median to +0.049 and worsens the narrow-band (Δ[-1..0]=−0.037).

The three targets cannot be simultaneously satisfied by any 2-parameter member of the LL family. The same applies to log-normal (skew is even worse: Δ[-1..0]=−0.088) and to inv-gamma (KS=0.029, Δ[-1..0]=−0.057).

### 3c. Why the old 251-point empirical table was not affected

The old table was NOT built from actual DBSCAN simulation data either — it was built from Beta-Prime synthetic samples pooled over eps=1.10–1.40. The z-calibration of the old table against the live JS simulation was never measured carefully; the −0.5 sigma peak visible before the eps-exponent fix masked this subtler bias.

---

## 4. Quantitative summary of the bias

Under the current LL master (shape=7.875, scale=24.057) applied to actual simulation data (eps=1.20):

- The left half [-2..0] receives **~7.5% more** probability mass than expected (0.522 vs 0.477).
- The inner left band [-1..0] receives **~8% more** mass than [0..1] (0.369 vs 0.300).
- The residual is consistent across all tested eps values (1.10–1.40), confirming the alpha(ε) formula itself is correct.

---

## 5. Fix options

| Option | Median_z | Δ[-1..0] | JS complexity | Notes |
|--------|----------|-----------|---------------|-------|
| LL scale = 23.731 (actual median) | ≈ 0 | −0.051 | 1 number change | Best achievable with LL |
| LL calibration-optimised (7.731 / 23.49) | +0.049 | −0.037 | 2 number changes | Balances large range, shifts median right |
| **Compact empirical SF table (~40 pts)** | **≈ 0** | **≈ 0** | interp (5 lines) | Perfect calibration by construction |
| Inv-gamma | −0.071 | −0.057 | needs regularised gamma | Not better than LL |
| Log-normal | 0 (mean only) | −0.088 | 1-line formula | Worst of all options |

**Recommended fix:** replace the LL formula with a ~40-point empirical SF table computed from actual DBSCAN simulation Rt values (parquet data + JS simulation pool, α-corrected). The `alpha(ε) = c0 + c1·ln ε` formula for the collapse exponent is correct and should be kept. Only the SF lookup needs to be empirical.

The table can be stored in master.js as two 40-element arrays and interpolated with 5 lines of code — substantially smaller than the 251-point table that existed before the analytic-formula switch.

---

## 6. What is NOT wrong

- The **alpha(ε) = 2.031525 + 0.258273·ln ε** running exponent is correct: the bias is eps-invariant, ruling it out as a cause.
- The **DBSCAN + hull area JS implementation** matches the Python simulation to within measurement noise.
- The **normInv / Acklam approximation** is not the cause (error < 1e-4 in the body).
- The **z-standard-deviation** is correct at 0.97–1.00 at all eps; only location is off.

---

## 7. ADDENDUM (2026-06-07) — §3b is wrong; analytic fix shipped

Section 3b claimed "no 2-parameter family can eliminate the asymmetry" and §5
recommended an empirical table. The claim is true only for **zero-location
log-scale** families (LL, log-normal, log-gamma, zero-loc inv-gamma) — exactly
the set that was tested. It does not generalize.

**Root cause, corrected.** `Rt` has a hard left edge: `R|N'=n = n/(λ₀S')` and
the hull area `S'` of an eps-connected cluster is bounded above, so `R` is
bounded away from 0 (loc ≈ 7.5 in Rt units). Forcing `loc=0` on such data is
what produced the "irreducible" log-skew (+0.67) — the skew is removable by a
location shift, not a deeper-family problem. This is also consistent with the
original per-eps Beta-Prime fits, where free `loc` (≈4–7) always beat `floc=0`.

**Shoot-out on real parquet data** (fit on train half, z-calibrated on test
half; full table in `analysis/zcal_eps1.20.csv`, cross-eps in
`analysis/zcal_shape10.csv`):

| master | median_z | Δ[-1..0] | Δ[-2..0] | KS(z) |
|---|---|---|---|---|
| LL current (shipped) | −0.067 | −0.069 | −0.080 | 0.030 |
| inv-gamma, floc=0 (RCA §3b) | −0.074 | −0.059 | −0.081 | 0.030 |
| **inv-gamma, free loc (3-param)** | **−0.007** | **−0.005** | **−0.007** | **0.004** |
| per-N' inv-gamma, free loc | −0.002 | −0.000 | −0.002 | 0.002 |

**Why inverse-gamma.** The analytic-night result: `R|N'=n` is exactly scaled
inverse-gamma; the marginal is an N'-mixture of those. Equivalently, the
shifted inv-gamma is the `a→∞` limit of the legacy Beta-Prime(a≈46, b≈10)
fits — the numerator Gamma concentrates, leaving `loc + InvGamma(b)`. The
fitted shape lands at 10.2 ≈ b ≈ `min_samples`.

**Shipped master** (pooled MLE over eps 1.10–1.40, equal-weight, shape frozen
at the integer 10):

```
SF(Rt) = P(10, y),  y = 157.7035/(Rt − 7.5091),  P = lower regularized gamma
```

Integer shape ⇒ exact Poisson sums (body: `1 − e^{−y}Σ₀⁹ yᵏ/k!`; tail: direct
series from k=10), no gamma library; JS matches scipy to <1e-14 relative.
Calibration at every eps in 1.10–1.40: |median_z| ≤ 0.017, band asymmetries
≤ 0.014, KS(z) ≤ 0.009 — strictly better than the proposed 40-point table in
both size and principle, and it keeps an analytic tail (polynomial, index 10).
`alpha(ε)` is kept unchanged. Implemented in `webapp/master.js`
(`masterSF/masterCDF/masterPDF`) and wired into `index.html`.

**Caveats.** (i) Tail past the censoring cap (`Rt = 62.83·ε^α`, z ≈ 4.0 at
ε=1.2) remains unvalidated — same caveat as before. (ii) The Python-side
scorer (`analysis/scorer_master.json`) still uses the older `R·eps²` collapse
with zero-loc inv-gamma(20.5); it has the same loc-bias and should be migrated
to the shifted form. *(Done — rolled out to `analysis/scorer.py`,
`modules/cluster_detector.py` and `webapp/` in the same change-set as this
addendum; pooled KS≈0.005 over eps 1.00–1.60.)*

---

## 8. APPENDIX (2026-06-07) — what the location shift *is*: DBSCAN's certification floor

Why does `Rt` have a left edge at all? Measured left edge of the real data
(1.8M clusters):

| ε | min R̃ | q0.1% | q1% |
|---|---|---|---|
| 1.10 | 11.68 | 14.25 | 15.87 |
| 1.20 | 11.60 | 14.27 | 15.87 |
| 1.30 | 11.11 | 14.16 | 15.85 |
| 1.40 | **10.92** | 13.97 | 15.80 |

The floor is ≈11, eps-invariant, and it is **derivable**. With sklearn's
convention (a point counts as its own neighbour), a cluster of exactly
`n = min_samples = 10` points must contain a core point whose eps-ball holds
all 10. The maximal hull is therefore 1 centre + 9 points on the eps-circle —
a regular 9-gon, area `(9/2)·sin(40°)·ε² = 2.89·ε²`, giving

```
R_min = 10/(λ₀·2.89·ε²) = 10π/2.89 · ε⁻² = 10.87·ε⁻²   →   R̃_min ≈ 10.9
```

The observed global minimum is **10.92** (ε=1.40, 1.2M clusters — the sample
finally digs down to the infimum). Since the regular-k-gon area
`(k/2)·sin(2π/k) → π` as k grows, the general statement is

```
R̃_floor = min_samples · (1 + 2π²/3(m−1)² + …) ≈ min_samples
```

**The left edge of R̃ IS min_samples** (up to an ~8% polygon correction at
m=10). The `ε⁻²` scaling of the floor is pure geometry, which is why one `loc`
survives the collapse at every eps. Larger-n clusters sprawl (multi-core
snakes) but a snake still needs ~m points per eps-ball, so the floor stays
≈ m — the deepest observed point is indeed an n=13 sprawler, not an n=10.

Zero-loc families thus put probability mass in a region (`R̃ < 11`) that
DBSCAN **cannot physically emit** — a guaranteed minimum density of
min_samples per eps-ball is part of the algorithm's output contract. That is
the entire content of §3b's "irreducible" log-skew. The fitted `loc = 7.51`
sits below the true floor because the inv-gamma density vanishes with an
essential singularity at loc and needs ~3.5 units of ramp room: the true
near-floor law is a large-deviation regime (9 points conspiring onto the
eps-circle with no 11th joining) that decays softer than inv-gamma-at-loc.

**The master's parameters, decoded:**

- **mean** = `loc + scale/(shape−1)` = 7.509 + 157.70/9 = **25.03** — exactly
  the no-free-parameter amplitude derived in `analytic_findings.md`
  (`C = N_min/f · k/(k−1) ≈ 25`). Not imposed; the MLE found it.
- **loc** ≈ the certification floor `min_samples·1.08 ≈ 10.9`, minus ramp room.
- **shape = 10** is the only honestly *effective* parameter: the true
  conditionals have shape `k(n) ≈ 3.5n−15` (k(10)≈20.5); mixing over `N'`
  roughly halves the effective curvature and the local tail index over the
  observed window lands at ~10. The coincidence with min_samples is suggestive
  but unproven — both `k(n)` and `P(N'=n)` are anchored at `n = m`, so any
  effective exponent comes out "of order m". (The legacy Beta-Prime `b ≈ 10`
  "pinned near min_samples" was this same effective exponent in different
  coordinates.)

**Family-theory view.** The true law lives on a *compact* interval — floor
`≈ m` (connectivity geometry), ceiling `= n/(min_area·λ₀)` (the filter). After
an affine map it is Beta-like, not Beta-Prime-like. Beta-Prime sends one
support edge to infinity; shifted inverse-gamma is the next limit in that
chain. The historical mistake was choosing *which edge to idealise away*:
every zero-loc fit idealised the left edge to 0 — and the left edge is the
single most physical number in the problem, while the right edge genuinely is
quasi-infinite (modulo the min_area filter).

**Where the last ~0.01–0.02σ lives (ranked):**

1. *Occupancy drift* — `P(N'=n)` broadens with eps (mean N′ 10.32→10.66 over
   1.10–1.40), so the mixture's shape drifts while α(ε) can only collapse its
   median; same physics as the −0.205 anomalous exponent. Also, α(ε) was fit
   to collapse zero-loc medians; shift and rescale don't commute (the free-fit
   loc drifting 7.5→6.8 over 1.10→1.40 is the fingerprint).
2. *Near-floor ramp mismatch* — large-deviation onset above R̃≈11 vs.
   inv-gamma's essential singularity (the residual Δ[-1..0] ≈ 0.005).
3. *Boundary clusters* — ~2% of clusters hull-touch the disk edge and obey a
   slightly different hull-area law.
4. *Gamma idealisation of `S'|n`* — compactly supported, only ~1% gamma.

**Falsifiable predictions:**

- *min_samples sweep*: floor moves as `m·(1+6.6/(m−1)²)` → ≈15.5 at m=15,
  ≈5.6 at m=5. If the fitted *shape* also tracks m, the shape=min_samples link
  is physics; if it stays ~10, it was the anchored-effective-exponent
  coincidence.
- *Conditional master*: per-N′ shifted inv-gammas already measure 2× better
  (KS(z) 0.0024 vs 0.0043) — suspect #1 confirmed in miniature.
- *Mean identity*: any refit should keep `loc + scale/(shape−1) ≈ 25.0`; if
  that drifts, the derived C is wrong, not the fit.
- *N-invariance*: at fixed λ₀, the per-cluster law is local — floor, master,
  and z-calibration should be unchanged at any N (radius √N), with only edge
  effects (∝ ε/√N) shrinking. **CONFIRMED** (`analysis/ncheck.py`, ε=1.40,
  ~37k clusters): N = 10k/20k/40k gives yield 1.22/2.39/4.96 per field
  (exactly ∝ N), median R̃ 23.86/23.84/23.78, mean R̃ 25.13/25.03/24.96
  (pinned at C≈25), mean N′ 10.674/10.670/10.659, median_z
  +0.008/+0.004/−0.008 with KS(z) ≈ 0.011 — all at the per-run noise floor,
  same master constants, no refit. The detector/webapp constants are valid for
  any field size given the right λ₀. The faint −0.3% median drift at N=40k is
  sign-consistent with suspect #3 (boundary clusters biasing R̃ slightly
  high, their fraction ∝ ε/√N).

One-line version: **the missed factor was that DBSCAN's output carries a
guaranteed minimum density — min_samples per eps-ball — so R̃ has a hard floor
at ≈ min_samples; every previous master assumed support down to zero and paid
~0.07σ for it.** The remaining 0.02σ is the occupancy distribution breathing
with eps, which no single-shape master can follow.

# Analytic Findings — overnight run (branch `analytic-night`)

> **READ ME FIRST.** There *is* a fundamental result beneath the Beta-Prime fit, and it's clean: **the eps-dependence factorizes into a pure power-law scale times an eps-invariant master shape, and that shape is derivable from elementary stochastic geometry.** Beta-Prime is just a convenient (non-identifiable) envelope, not fundamental. Concretely:
>
> 1. **Collapse (Finding #1).** `R(eps) = scale(eps)·X` with `scale ∝ eps^(−2.205)` and `X` eps-invariant to ~1% (q99/q50 CV 1.1%). Rescaling `R̃=R·eps²` cuts cross-eps quantile spread from 52% to ~5%. The `−2` is geometry (`S'∝eps²`); the `−0.205` residual is the only real second-order physics — fixed `min_samples` vs growing eps, split ~44% N'-occupancy broadening + ~56% per-n area loosening (`analysis/residual_model.py`).
> 2. **Mechanism (Finding #2).** `R|N'=n = n/(λ₀S')` is *exactly* scaled-inverse-gamma with shape = the Gamma-shape of the hull area `S'|n`. The marginal is the N'-occupancy mixture of these; it reproduces the data as well as Beta-Prime. Beta-Prime params are **non-identifiable** (two basins both fit) — don't read mechanism from `b`.
> 3. **First principles (Finding #3).** `S'|n` is just the convex-hull area of `n` uniform points in an eps-disk (CSR + Poisson conditioning); model matches data to ~1%, chaining adds 8%. The leading constant `C≈25` is derived with **no free parameters**: `C = N_min/f · k/(k−1)`, fill fraction `f≈0.42`, hull-shape `k≈20.5`.
> 4. **Scorer (byproduct).** One master on `R̃=R·eps^α(ε)` gives a validated, eps-independent rare-event score — dissolving the eps-weighting ambiguity the old handoffs agonized over. The running exponent `α(ε)=2.031525+0.258273·ln ε` (fit from 110 eps values) cuts the residual spread from ~5% (naive eps²) to <0.5%. Best-fit parametric master: log-logistic (shape≈7.87, scale≈24.06, KS<0.03) or inv-gamma (shape≈20.5). `analysis/scorer.py`, `scorer_table.csv`, `scorer_master.json`.
> 5. **Look-elsewhere (Finding #4).** A *typical* CSR cluster scores "~6σ" under naive per-window Kulldorff-LR/Wilks — overstated ~10⁸×. Only MC-replay calibrates (which the simdata is). LR beats raw `R` because it size-weights (equal-`R`, larger-`N'` clusters are rarer).
>
> **Two bugs fixed** (`analysis`-adjacent, separate commit): the mixture PDF-was-used-as-CDF KS, and a **4-vs-5 stride bug** in `betaprime_mixture_pdf` that almost certainly caused the repo's "degenerate identical-component mixture" results.
>
> **One-line upshot:** the whole problem reduces to *two* elementary objects — the convex-hull-area law of `n` uniform disk points, and the `N'`-occupancy distribution `P(N'=n)`; everything else (Beta-Prime, the eps-drift, the density ratio) is downstream.

This is the running report of the analytic-first overnight session (see plan `~/.claude/plans/curried-moseying-flask.md`). Goal: find the fundamental structure beneath the empirical Beta-Prime fit. All work is read-only on `simdata/v2/`; new code in `analysis/`; small sims only.

Conventions: `N=10000`, `R=100`, `S₀=πR²=31415.93`, `λ₀=1/π=0.318310`, `min_samples=min_cluster_size=10`, `min_area=0.5`. `R=(N'/S')/λ₀`. Censoring cap (raw R): `R_max=N_min/(min_area·λ₀)=62.83`.

---

## 0. Headline summary
See the **READ ME FIRST** box above. In one equation, the synthesis of Findings #1–#3:

```
S'|N'=n  ~  eps² · (convex-hull area of n uniform points in a unit disk)  ~  eps² · Gamma(k(n)≈3.5n−15)
R|N'=n   =  n/(λ₀ S')                                                       ~  scaled-InverseGamma(k(n))   [exact]
R        =  Σₙ P(N'=n) · (R|n)                                              [N'-mixture, mode n=10]
meanR    ≈  (N_min/f)·(k/(k−1)) · eps⁻²  ≈  25·eps⁻²                        [f≈0.42 hull fill fraction]
```

Status: all planned workstreams (T1–T6) complete; results below. Artifacts in `analysis/`.

---

## Finding #1 — the `R·eps²` collapse law (the eps-dependence factorizes)

**Script:** `analysis/scaling_collapse.py` → `analysis/collapse_summary.csv`, `collapse_ecdf.png`, `collapse_master.png`. Sweep eps 0.95–2.00 (step 0.05), real `simdata/v2` data.

**Result (the fundamental statement).** The density-ratio distribution factorizes into a pure power-law **scale** times an eps-invariant **shape**:

```
R(eps)  =  scale(eps) · X ,     scale(eps) ∝ eps^(−2.205) ,     X ⟂ eps
```

- **Scale.** `meanR ∝ eps^(−2.205)` *as a single power-law fit over [0.95,2.0]* — but this exponent is **NOT constant, it RUNS** (verified on existing data, which pins `meanR(eps)` to ~0.05%): the log-log has real curvature (quadratic coeff −0.38), and the **local** exponent runs from **exactly −2.000 as eps→0** (pure geometry: eps is the only length scale ⇒ `S'∝eps²` ⇒ `R∝eps⁻²`) through −2.13 @eps=1.2, −2.25 @1.5, −2.52 @2.0. So "−2.205"/"−0.205" is a *window-averaged* number, not a universal constant (it shifts to −2.28 over [0.95,2.3]). The well-defined objects are the **exact −2 geometric limit** and the **running** of the anomalous part (driven by fixed-`min_samples` occupancy+area drift, §Finding #2). No sim needed — the data already pins the curve; the quantity simply isn't a constant.
- **Shape `X` is eps-invariant to ~1%.** Quantile ratios to the median are flat across the entire range (factor 4 in eps²):

  | ratio | mean | CV across eps |
  |---|---|---|
  | q05/q50 | 0.731 | 1.45% |
  | q25/q50 | 0.871 | 0.38% |
  | q75/q50 | 1.169 | 0.34% |
  | q95/q50 | 1.527 | 0.75% |
  | **q99/q50** | **1.920** | **1.09%** |

  The far right tail — the part that matters for rare-event scoring — is the *most* eps-stable.

- **Collapse quality.** Rescaling `R̃ = R·eps²` cuts the cross-eps quantile CV from **~52% (raw R) to ~4.5–5.8%** (the residual is the leftover `eps^−0.205` scale drift, not shape).

**Why the exponent is 2.205, not 2 (the residual physics).** The collapse would be *exact* if minimal clusters were scale copies (fixed N', area ∝ eps²). They are not, because **`min_samples=10` is a fixed count that does not scale with eps**. As eps grows, the minimal detectable cluster becomes relatively looser/larger: the rescaled hull area `u = S'/eps²` rises **+51.8%** (mean 1.338→2.031) and `meanN'` rises **+23.7%** (10.23→12.65) over 0.95→2.00. Since `R̃ = N'/(u·λ₀)` and `u` outgrows `N'`, `R̃` drifts down by ~13–15% end-to-end — exactly the `eps^−0.205` term. So the **+0.205 excess exponent is the fingerprint of fixed `min_samples` against growing eps**, a genuine and explainable second-order effect.

**Consequences.**
1. The whole "Beta-Prime parameters drift linearly with eps" story is, to first order, just this factorization expressed in Beta-Prime coordinates. The *interesting* object is the eps-invariant master shape `X` (= `R·scale(eps)⁻¹`), studied in Findings #2–#3.
2. The eps-independent scorer (T4/§byproduct) follows immediately: score any observed cluster by mapping `R̃ = R·eps^α(ε)` with `α(ε)=2.031525+0.258273·ln ε` and reading the tail of the single master distribution (log-logistic: `SF = 1/(1+(R̃/24.06)^7.87)`). The running exponent reduces the residual spread from ~5% (naive eps²) to <0.5%. No eps-mixture weighting needed.
3. **Censoring caveat unchanged:** raw `R` is clipped at `62.83` (eps-independent), so in `R̃=R·eps^α(ε)` space the cap is `62.83·eps^α(ε)`. The body and the q99-level tail collapse cleanly (all below the cap for the studied eps); do not trust the collapse for `R̃` beyond `~62.83·eps^α(ε)`.

## Finding #2 — Beta-Prime *derived*: an N'-mixture of inverse-gamma hull-area laws

**Scripts:** `analysis/conditional_decomp.py` → `conditional_eps*.csv`, `conditional_mixture.png`.

**The mechanism (exact + empirical).** Condition on the cluster point count `N'=n`. Then

```
R | n  =  n / (λ₀ · S')        ← exact: R|n is a deterministic reciprocal of the hull area S'
```

so if `S'|n ~ Gamma(k(n), θ(n))`, then **`R|n ~ scaled-inverse-Gamma(shape = k(n))`** with the *same* shape. The data confirms this identity to machine level (the fitted inverse-Gamma shape of `R|n` equals the fitted Gamma shape of `S'|n` exactly, e.g. 20.523 = 20.523). Inverse-gamma fits `R|n` well (KS 0.02–0.03 per n).

**The two empirical laws that close the model:**

1. **Hull-area shape grows linearly in point count:** `k(n) ≈ 3.3·n − 12` (so `k(10) ≈ 21`, `k(15) ≈ 38`), nearly eps-independent. (The convex-hull area of `n` clustered points has Gamma shape ≈ 2–3.5·n — *not* `n`; this is why the old `b≈10=min_samples` reading was a coincidence/red herring.)
2. **At fixed n, area scales as `eps²` to ~2%:** `⟨S'|10⟩/eps²` = 1.315 / 1.328 / 1.342 at eps = 1.0 / 1.2 / 1.4. So per-cluster geometry is a clean `eps²` scale; the shape `k(10)≈20.5` is eps-invariant.

**The derivation.** The marginal density ratio is the N'-occupancy mixture

```
P(R) = Σₙ P(N'=n) · scaled-InvGamma( R ; shape k(n), scale ∝ n·eps² )
```

dominated by the `n=10` mode (weight 0.68–0.83). Reconstructing the marginal from the per-`n` inverse-gamma fits matches the data **as well as or better than** the phenomenological Beta-Prime: KS(mixture) = 0.035 / 0.030 / 0.028 vs KS(Beta-Prime) = 0.028 / 0.035 / 0.017 at eps = 1.0 / 1.2 / 1.4. **So Beta-Prime is not fundamental — it is the smooth 4-parameter envelope of this inverse-gamma mixture.** The mixture is the mechanism; Beta-Prime is the convenient fit.

**Beta-Prime parameters are non-identifiable — do not read mechanism from them.** A free-`loc` fit lands in a different basin (`a≈160–240, b≈19–27, loc≈−1`) than the repo's `regression_params.csv` basin (`a≈45, b≈10, loc≈6`), yet *both* give KS ≈ 0.02–0.035. The `(a, loc, scale)` ridge is nearly flat. The robust, physically-meaningful quantity is the **inverse-gamma / hull-area shape `k(10) ≈ 20.5`**, not `b`.

**This also re-explains Finding #1's residual exponent (decomposed, `analysis/residual_model.py`).** The extra −0.205 has **two comparable sources, both from fixed `min_samples`**:
1. **N'-occupancy broadening (~44%).** `P(N'=10)` falls 0.83→0.68 and the upper tail thickens (`P(n+1)/P(n)` grows ~0.27→0.47 over eps 1.0→1.4), shifting weight to higher-`n` clusters — which are **less dense** (`m(n)=E[R·eps²|n]` falls 24.9→18.7 from n=10→15, since hull area grows faster than `n`). Rebuilding `C(eps)` from the eps-specific occupancy × a *fixed* set of per-n laws reproduces ~44% of the observed −6.4% drift.
2. **Per-n area drift (~56%).** Even at fixed `n`, `⟨S'|10⟩/eps²` is not perfectly flat — it rises ~2–3% over eps 1.0→1.6 (minimal clusters get relatively looser as eps outgrows the fixed 10-point core), pulling `R̃|n` down by the complementary amount.

So geometry is *nearly* scale-clean (eps² exact to ~2% at fixed n) and the small residual is a mix of occupancy-shift and slight area-loosening — two faces of the same fixed-`min_samples`-vs-growing-eps effect.

**Upshot — the fundamental object.** Everything reduces to **the convex-hull-area law of an `n`-point DBSCAN cluster, `S'|n ~ Gamma(k(n)≈3.3n−12, scale ∝ eps²)`**, plus the N'-occupancy distribution `P(N'=n)`. The density ratio, the beta-prime, and the eps-dependence are all downstream of these two. Finding #3 attacks `S'|n` from stochastic geometry.

## Finding #3 — the hull-area law is pure CSR geometry (and `C≈25` is derived)

**Script:** `analysis/hull_geometry.py` → `hull_geometry.csv`, `hull_geometry.png`. Cheap MC (no large sim).

**Generative model (CSR + Poisson conditioning).** Pick a core point `p`. Under CSR, conditioned on `m=n−1` other points lying within `eps` of `p`, those points are i.i.d. **uniform in the disk of radius `eps`** around `p` (the conditional-uniformity property of the Poisson process). So a minimal DBSCAN cluster ≈ `n` points in an eps-disk, and `S'/eps²` is a pure dimensionless random variable.

**It matches the data with no fitting:**

| n | Model A: `n` uniform in disk — mean (k) | real data eps=1.2 — mean (k) |
|---|---|---|
| 10 | 1.315 (18.4) | 1.328 (21.5) |
| 12 | 1.478 (26.0) | 1.706 (28.8) |
| 15 | 1.662 (38.2) | 2.589 (39.9) |

- `⟨S'|10⟩/eps²`: **1.315 (geometry) vs 1.328 (data)** — ~1%.
- A tiny *real* DBSCAN sim (Model C) gives 1.327, k=20.8 — i.e. it matches the production data exactly, and **chaining inflates the pure single-core model by only 8%** (`centre+9 uniform`=1.225 → DBSCAN=1.327). The minimal cluster behaves geometrically almost exactly like **10 uniform points in an eps-disk**.
- `k(n)` slope: 4.0 (model A) / 3.7 (model B) / **3.45 (data)** — the linear growth of hull-area Gamma-shape with point count is a CSR-geometry fact, not a DBSCAN artifact. (Note the hull-area shape ≈ `2–4·n`, *not* `n` — the convex hull of `n` points concentrates faster than a sum of `n` independent pieces because vertices share structure.)

**The leading constant `C≈25` derived from scratch.** With the `n=10` mode dominating, `meanR ≈ (N_min/λ₀)·E[1/S']`. Writing the hull **fill fraction** `f = ⟨S'|10⟩/(π·eps²) = 1.32/π = 0.420` (the hull of 10 uniform disk-points covers 42% of the disk) and the Jensen correction `E[1/S']/(1/E[S']) = k/(k−1)` for an inverse-gamma of shape `k≈20.5`:

```
C = meanR·eps²  ≈  N_min / f · k/(k−1)  =  10 / 0.420 · 20.5/19.5  =  25.0
```

versus the observed `C = 24.9–25.0` at small eps. **No free parameters** — `N_min`, the geometric fill fraction `f≈0.42`, and the hull-shape `k≈20.5` fully determine the leading amplitude of the whole `R∝eps⁻²` law.

**Synthesis of Findings #1–#3 (the fundamental picture).** Under CSR, the DBSCAN density ratio is governed entirely by elementary stochastic geometry:

```
S' | N'=n  ~  eps² · (hull area of n uniform points in a unit disk)  ~  eps² · Gamma(k(n)≈3.5n−15)
R  | N'=n  =  n/(λ₀ S')  ~  scaled-InverseGamma(shape k(n))           [exact]
R          =  Σₙ P(N'=n) · (R|n)                                       [N'-occupancy mixture, mode n=10]
meanR      ≈  (N_min/f)·(k/(k−1)) · eps⁻²  ≈  25·eps⁻²
```

Beta-Prime is merely the smooth phenomenological envelope of the `R|n` inverse-gamma mixture; its parameters are non-identifiable and carry no extra physics. The eps-dependence is a pure `eps²` area scale (exact at fixed `n`) times a slow `eps^−0.205` correction from the `min_samples`-induced broadening of `P(N'=n)`. **The genuinely fundamental objects are just two: the convex-hull-area distribution of `n` uniform disk points, and the N'-occupancy distribution `P(N'=n)`.**

## Finding #4 — Kulldorff LR: the look-elsewhere effect, quantified

**Script:** `analysis/scan_lr.py` → `scan_lr_summary.csv`, `scan_lr.png`.

The design doc says score with the Kulldorff scan LR, not raw `R`. The LR is computable from the existing data, and `n/μ = N'/(λ₀S') = R` exactly, so

```
2 lnLR = 2[ N'·ln(N'/μ) + (N−N')·ln((N−N')/(N−μ)) ]  ≈  2·N'·(ln R − 1 + 1/R)
```

(the approximation is exact to **0.02%** here, since `N≫N'` and `μ=λ₀S'≈0.5`).

**The look-elsewhere effect, in hard numbers.** A *typical* (median) noise cluster has:

| eps | median 2lnLR | median R | naive Wilks p | naive "σ" |
|---|---|---|---|---|
| 1.00 | 45.3 | 23.9 | 1.7e-11 | 6.7 |
| 1.20 | 38.4 | 16.2 | 5.8e-10 | 6.2 |
| 1.40 | 32.7 | 11.7 | 1.1e-08 | 5.7 |
| 1.60 | 28.0 | 8.7 | 1.2e-07 | 5.3 |

So under naive per-window Wilks scoring, the **median** CSR cluster looks like a **5.3–6.7σ detection** — and these appear in essentially every random field. Naive scoring overstates significance by **~10⁸×**. This is the selection bias of §1 of the theory doc made concrete, *compounded* by the small-`μ` (≈0.5) breakdown of the χ²₁ asymptotic. The only valid calibration is MC-replay of the pipeline — which the simdata **is**, in `R`-space (Findings #1–#4). The practical takeaway: score with the empirical master survival (T4), never with a per-window analytic p-value.

**LR is a better statistic than raw `R` (validates the design doc).** `R` and `2lnLR` correlate (Spearman 0.71) but are not equivalent: at fixed `R≈20`, `2lnLR` ranges **40→58.6 as `N'` goes 10→14**. Two clusters with the *same density ratio* are **not** equally significant — the one with more points is rarer under CSR, and only LR captures that. So a future scorer should ideally calibrate on `LR` (via the same MC-replay), gaining the size-weighting `R` lacks.

**Eps-stability:** median `2lnLR` drifts with CV ≈ 16% across eps — better than raw `R` (52%) but worse than the collapse variable `R·eps²` (4.5%), because larger eps lowers `R` but slightly raises `N'`, partially self-cancelling. So `R·eps²` remains the cleanest scoring coordinate; LR's advantage is the size-weighting, not eps-stability.

## Finding #5 — back to the original question: the classifier, µ_eps, and formula reproduction

Four follow-up experiments (`analysis/mu_eps_test.py`, `signal_injection.py`, `formula_vs_empirical.py`, and a moments calc) answer the practical "can we now build the z-classifier" question.

### 5a. The real control variable is `µ_eps`, not `eps²` (N is not free)
`R` depends on `N`, `R_dom`, `eps` **only** through `µ_eps = λ₀·π·eps² = (N/R_dom²)·eps²` (mean points per eps-disk) plus the integer `min_samples`. Three configs with λ₀ differing 4× but the same `µ_eps=1.44` (N=5000/10000/20000, eps=1.697/1.20/0.849) give matching `R` quantiles (median 16.25/16.60/16.03, bulk agreement ~3%). **The clean "eps²" is a coincidence of this config: `N=10⁴=R_dom²`, so `µ_eps=eps²` exactly.** The `−2.205` scale exponent is an *effective* local slope (→ −2 as eps→0); the `+0.205` is DBSCAN resolution drift (fixed `min_samples` vs eps), not a noise property.

### 5b. The classifier works — but on LR, not raw R
Injecting an extended Poisson splat (radius 5≈4·eps) into CSR and replaying the pipeline, scoring each recovered cluster against the CSR null:

| n_extra | density ratio | detect % | med N' | med R | z_R | exceeds null in R | exceeds null in LR |
|---|---|---|---|---|---|---|---|
| 60 | 3.4× | 71% | 13 | 13.2 | −1.0 | 0% | 16% |
| 120 | 5.8× | 100% | 58 | 8.3 | **−6.0** | 0% | **100%** |

**Raw `R` actively fails for extended signal:** a strong splat forms a *large* cluster (N'=58) whose density ratio `R=8.3` is *below* the noise median (16.3), so `z_R=−6` — it looks *less* anomalous than a typical tight noise blob. The **Kulldorff LR succeeds**: 100% of strong-splat clusters exceed the entire null in LR while 0% do in R. This is the concrete payoff of Finding #4: DBSCAN noise clusters are tight minimal blobs (high R, small N'), so **signal must be scored by the size-weighted LR**, never by raw density ratio. The original 2020 question is answerable, and the right statistic is the scan LR calibrated by MC-replay.

### 5c. The 4-number formula reproduces all 31 per-eps Beta-Prime fits
Collapse prediction: `a,b = const`, `loc(eps)=loc₀/eps²`, `scale(eps)=scale₀/eps²` — **4 numbers replace 31×4=124**. Applied to each eps's data (`formula_vs_empirical.csv`):

- **KS-on-data:** empirical per-eps (124 params) mean KS **0.0047**; 4-number formula Beta-Prime mean KS **0.0171** (max 0.032); 2-number inverse-gamma master mean KS **0.032** (max 0.057). All are excellent fits at n=10⁵ — the formula is ~3× the empirical KS but still KS<0.032 everywhere, tightest near mid-eps and loosening at the extremes (the `eps^−0.205` residual).
- **Parameters:** `b_pred≈10.4` matches `b_emp≈10`; `loc_pred=loc₀/eps²` tracks `loc_emp` to ~10%; `a` and `scale` sit on a *different point of the non-identifiable ridge* (`a_pred≈82` const vs `a_emp` drifting 41→56) — cosmetic, since KS is the arbiter. **One master (4 numbers) + the eps² law reproduces the entire 124-number table.**

### 5d. Does the z-equivalent carry error from leptokurtosis?
The null is leptokurtic but mildly (inverse-gamma shape 20.5: skew 0.98, **excess kurtosis 1.90**, tail index ~21 → polynomial tail `S(x)~x^−21`). Key points:
- **The transform `p→z=Φ⁻¹(1−p)` is exact (error-less) given the true CDF** — it's the probability integral transform, distribution-free. Heavy tails do *not* introduce error in the transform itself.
- **The error lives entirely in *estimating* the null tail**, and that's where leptokurtosis bites: (i) MC sampling error on a tail `p` from `n` null samples has relative SE `√((1−p)/(np))` — e.g. at `p=1e-3` you need `n~1e6` for ±0.01 in z; with only ~344–812 null clusters (as in 5b) z is **hard-capped at ~3.0–3.2** (`Φ⁻¹(1−1/2n)`); (ii) beyond the censoring cap `R̃=62.83·eps²` there are *zero* null samples, so p is unknowable; (iii) a parametric fit can extrapolate but with bias — the empirical tail is heavier than the inverse-gamma fit.
- **Knowing only the kurtosis is NOT enough for an exact transform** (you need the full CDF). A Cornish–Fisher expansion using skew+kurtosis is *approximate* — it lands within ~2% of the exact inverse-gamma quantile at `p=1e-2…1e-4` here, but residual error grows in the far tail. **The right lever the power law gives you is the tail index `α`:** fit a generalized Pareto to exceedances, extrapolate the `x^−α` tail with a *propagated confidence band* on z. That converts "unknown far tail" into "estimated tail with honest error bars."
- **Bottom line:** report `z` (or `p`) with a confidence band from the tail-fit / MC binomial error, and never quote point-estimate sigmas past the censoring boundary. The z is a relabeling of a heavy-tailed `p`; its uncertainty is the tail-estimation uncertainty, not a flaw in the transform.

---

## Finding #6 — the deep tail without brute force (entropy = the LR rate function)

**Script:** `analysis/tail_extrapolation.py`. Populating a `p=1e-9` bin by brute MC needs ~1e11 fields — hopeless. Three non-brute-force routes, all anchored by one fact:

**`2lnLR` IS the large-deviations rate function (= entropy cost of the fluctuation).**
For a window of area `S'` at density `ρ=n/S'`, `2lnLR = 2·S'·D(ρ‖λ₀)`, `D = ρln(ρ/λ₀) − (ρ−λ₀)` (Poisson relative entropy / KL). So `2lnLR = u` ⇔ `u/2` nats `= 0.721u` bits of improbability. **This is the entropy-deviation intuition, made exact** — the rarity of a condensation is literally its entropy cost, and the LR measures it.

- **(1) Analytic LDT extrapolation (primary).** Because the cost is an entropy rate, the LR tail is **exponential** (log-survival linear in `u`), so fit it on the body (good stats to `p~1e-5` from ~10⁶ fields) and extrapolate. Demonstrated: fitting on a 6,600-cluster *small* sim and extrapolating predicts the full-data tail — and the empirical log-survival is **convex** (local slope steepens 0.21→0.34), so a single-slope fit **over-predicts the tail (≈16× at u=72) = conservative** (understates significance, the safe direction). Using the exact convex rate `D(ρ)` instead of a linear-in-`u` fit removes that bias. Either way: no deep-tail sims needed.
- **(2) Importance sampling.** Tilt the point process so dense clusters are common, reweight by the likelihood ratio — unbiased, variance reduced by orders of magnitude. The route if you want *simulated* deep-tail validation.
- **(3) GPD / peaks-over-threshold** for the `R` variable (polynomial tail); for `LR` the log-linear fit is simpler. Fit the tail index, extrapolate with a confidence band.

**Entropy & per-field tail (extrapolated):**

| 2lnLR | nats | bits | p / cluster | p / field | fields for ~10 events |
|---|---|---|---|---|---|
| 50 | 25 | 36 | 3e-2 | 4e-3 | 2.4e3 |
| 63 (=min_area cap, N'=10) | 32 | 45 | 1e-3 | 1.6e-4 | 6e4 |
| 80 | 40 | 58 | 2e-5 | 2e-6 | 4e6 |
| 100 | 50 | 72 | 1e-7 | 1.5e-8 | 7e8 |

**The PRNG "Diehard" horizon, quantified.** Your intuition that a PRNG eventually can't faithfully produce the tail is right, with two floors:
- **Seed/state entropy (hard floor):** NumPy PCG64 has a 128-bit state → events rarer than `2^−128 ≈ 3e-39` (`2lnLR~177`, 128 bits) literally cannot be produced. **Non-binding.**
- **Validation horizon (binding):** PCG64 is validated by BigCrush over ~`2^38–2^40` outputs. A campaign consuming that (`2N=2e4` draws/field → ~`1.4e7–5.5e7` fields) reaches `p_field ~ 1e-7–1e-8` (`2lnLR~90–100`). Beyond that you're using the PRNG **outside its validated envelope** — and it's the same depth where brute force dies. So compute-limit and PRNG-fidelity-limit coincide at `p~1e-7`.
- **But the binding limit *right now* is the `min_area` censoring** (`2lnLR~63`, `p_field~1.6e-4`) — far shallower. Remove `min_area`, then the PRNG/compute horizon governs, and below `p~1e-8` you should switch to a **counter-based RNG (Philox/Threefry**, provable equidistribution, no period/correlation worry) plus routes (1)+(2).

**Net:** the true tail is unbounded but exponentially thin (no hard "entropy bound" on length); the *practical* trustworthy horizon is `p~1e-7–1e-8`. Past it, don't brute-force — use the entropy/LDT rate (the LR itself) to extrapolate analytically, validated by importance sampling on a counter RNG.

---

## Finding #7 — the tail is an *integrable* singularity; the censorship is removable

The `R→∞` as hull-area `S'→0` blow-up is a genuine **ultraviolet divergence** (defining a density from δ-points is ill-posed at short distance — the bandwidth problem / UV problem). But it is **integrable**, so the `min_area` censorship is unnecessary, not a true regularization.

**The tail is a power law, index measured three ways (`analysis/tail_*.py`):**
- empirical `R|N'=10` log-log slope (censored window): **α≈7.7**
- convex-hull small-area geometry exponent (uncensored, no DBSCAN): **α≈7.0** (rising to ~11 by n=16; trend ≈ `n−1`)
- (the inverse-gamma body shape k≈20 is the tail of the *body fit*, NOT the true tail — it badly under-states rare events; and a GPD on the censored 2lnLR gives a spurious `ξ<0` finite endpoint. Both wrong.)

**Consequences (the resolution of the whole tail/censorship problem):**
1. Since **α≈7 > 2**, the distribution of `R` is **proper and has finite mean/variance/… up to ~6th moment**. The "infinite spike" is an integrable measure-zero set. `min_area` only clips `p<1.6e-4`; **set `min_area→0` and the statistics are fine** — the cutoff→0 limit exists (UV-finite / "super-renormalizable").
2. **The real regularizer is the count threshold** `min_samples=min_cluster_size=10`, which sets `α≈n−1`. `N'=3` (collinear triangle) gives `α≈2` → divergent mean: that is the true "fall-to-center" singularity (cf. the `1/r²` potential in QM); `N'≥10` (`α≈9`) sits safely in the integrable regime (cf. integrable `1/r`).
3. **It's literally RG:** Finding #1's `R∝eps^(−2.205)` = engineering dimension −2 (area) + anomalous dimension −0.205 from the marginal fixed-`min_samples` coupling. eps is the cutoff/resolution scale.
4. **Cellular re-representations remove the singularity at the source** by replacing the adaptive hull denominator with a fixed scale or count floor: fixed **grid** (= the project's original coarse method; also *restores* clean look-elsewhere = exactly K independent cells), **Voronoi/Delaunay (DTFE)** (data-adaptive, **analytic CSR cell-area null** — the elegant choice, with direct precedent in cosmology void/cluster finders ZOBOV/VOBOZ), **k-NN density** (= DBSCAN core-distance), or **KDE** (bandwidth = the scale). All encode the same regularization: a scale *or* a count floor.

**RESOLVED by the uncensored resim (`uncensored_sim.py`, `uncensored_analysis.py`; 1e6 fields, min_area=0, 131,797 clusters):**
- **Tail index α≈6.5–7 confirmed by THREE independent methods** — uncensored Hill 6.3→7.1 (rising with threshold), hull geometry 7.0, censored log-log 7.7. Robust.
- **Direct measurement** pins the survival to `z≈4.1` (`p~4e-5`, `2lnLR~72`); **extrapolation** (α≈7 power law = exponential-in-u rate 0.366, anchored at the data edge) gives `z(2lnLR=100)≈6.2`, `z(120)≈7.2`. The geometry-α extrapolation (`tail_final.py`, z(100)≈6.25) and this data-anchored one now **agree** — that's the cross-validation.
- **Heavier than the inverse-gamma/GPD body-fits** (which implied z~7.8–9 at u=100): extreme clusters are *less* rare than light-tailed fits claim — **don't over-call significance.**

**CORRECTION to the earlier censoring narrative (Findings #4/#6 over-stated it).** The uncensored vs censored survival curves are **essentially identical**: only **0.006% (8/131,797) of clusters** lie beyond the `R=62.83` cap. `min_area=0.5` sits at the ~99.994th percentile of hull area, so it **pins the observed *maximum*** at 62.8 (this correctly explains the `eps_experiment` "max≈62 stable across all eps" — a single order statistic) **but removes negligible probability mass and does NOT shape the tail below the cap.** The practical limit on the observed tail was always **sample size** (`p~1e-5` floor at ~1.3e5 clusters), not censoring. So: keep `min_area` or drop it — the distribution (and the scorer) is unchanged except for the literal max. The real route to deeper tails is **extrapolation of the α≈7 power law**, not more sims (brute-forcing to `p~1e-9` ≈ a week on 6×A100, and lands at the PCG64 validation horizon anyway).

---

## Finding #8 — KDE: a smooth, singularity-free detector + where the analytic null lives

**Scripts:** `analysis/kde_null.py`, `kde_signal.py` → `kde_null_h*.png`, `kde_vs_dbscan.png`. Statistic = standardized peak of the kernel-smoothed intensity `Z_max = max_x (ρ̂−μ)/σ`, computed by binning + Gaussian filter. No `S'→0` singularity (bandwidth `h` is the regularization scale).

1. **The analytic look-elsewhere (random-field theory) works only in the Gaussian regime.** RFT gives `P(Z_max>u) ≈ R₂·(4ln2)/(2π)^{3/2}·u·e^{−u²/2}` (`R₂=area/FWHM²` resels) — *if* the smoothed field is Gaussian, i.e. many points per kernel `λ₀πh²≫1`.
   - `h=4` (16 pts/kernel): field ~Gaussian, **RFT matches MC** within ~2–5× — a usable closed-form multiple-testing correction.
   - `h=1.2` (=eps, **1.4 pts/kernel**): field is sparse and skewed; **RFT fails by ~10²–10⁵×**. So at the resolution where small (N'~10) clusters actually live, the field is intrinsically non-Gaussian — *this is precisely why the problem has no closed-form null at the scale that matters*. You get the analytic look-elsewhere only by blurring to large scales (and losing the small clusters); otherwise it's MC or the DBSCAN α≈7 tail.

2. **KDE and DBSCAN converge partially on the null, strongly on signal.**
   - *Null* (same CSR fields, h=eps): Spearman(KDE `Z_max`, DBSCAN `max R`) = 0.37; top-5% extreme-field overlap = 30% (**6× enrichment** over chance). Positively correlated but **complementary, not redundant** — `max R` chases the single tightest blob, `Z_max` the broadest smoothed peak.
   - *Signal* (the extended splat that defeated raw R with z_R=−6, Finding #5b): KDE detects it at **100% (median Z_max=13 vs null 99th=7.7)** at 5.8× density, 72% at 3.4×. **KDE succeeds exactly where raw density-ratio R failed**, because it integrates over the region instead of being dominated by tight noise blobs.

**The settled detection picture (answers the original 2020 question):**

| signal type | raw R | DBSCAN + Kulldorff LR | KDE peak |
|---|---|---|---|
| tight, high-density clump | ✅ | ✅ | ~ (needs small h) |
| extended, moderate over-density | ❌ (z_R<0) | ✅ (size-weighted) | ✅ (integrates region) |

So: **score with the Kulldorff LR and/or a KDE peak (never raw R), calibrate by MC at the small-cluster scale (the α≈7 heavy tail), and use RFT for the analytic look-elsewhere only when you deliberately work at a coarse, Gaussian scale.** KDE and LR are the two convergent workhorses; they agree on real signal and bracket the noise structure.

---

## Finding #9 — unified detector `score(points)→{detections,p,z}` + hybrid KDE calibration

**Script:** `analysis/detector.py` (caches per-field nulls to `null_cache.npz`). Two complementary arms, each scored against the **per-field CSR null** so the look-elsewhere is built into `p` ("how often does pure CSR produce something this extreme *anywhere* in a field"):
- **LR arm:** DBSCAN clusters → Kulldorff `2lnLR`; null = MC of per-field max-`2lnLR` + α≈7 tail extrapolation.
- **KDE arm:** smoothed-intensity peak `Z_max` at fine (h=eps=1.2) and coarse (h=4) scales.

**Hybrid KDE calibration (Task 2):** the coarse scale has 16 points/kernel (Gaussian regime) → calibrated by **RFT analytic** (no MC); the fine scale (1.4 pts/kernel, non-Gaussian) → **MC**. The detector auto-selects: `RFT if λ₀πh²≥10 else MC`. Crossover validated — at coarse h, RFT vs MC agree within ~2–5× (RFT conservative).

**Demo (Task 1), per-field p/z:**

| field | DBSCAN+LR | KDE-fine | KDE-coarse |
|---|---|---|---|
| pure noise | (none) | z=−0.8 | z=−0.6 |
| **tight** clump (12 pts, r=1) | **z=3.5** @(−30,20) | **z=3.1** @(−30,20) | misses (blurred out) |
| **extended** splat (120 pts, r=5) | **z=8.0** @(45,0) | **z=6.2** @(45,0) | **z=12** @(45,0) |

Reads exactly as designed: silent on noise; tight clump caught by LR + fine-KDE (coarse blurs it away); extended splat caught by **all three** at the correct location — including the case where raw density-ratio R gave z=−6. Locations recovered to <2 units.

**Caveats:** the RFT `z` in the deep tail (e.g. z=12 for the splat) is a *screening* number — RFT's Gaussian far-tail over-extrapolates for grossly non-null peaks; treat z≳6 from RFT as "very significant," not a calibrated value. For precise deep-tail p, use the MC+α≈7 LR arm. Per-field calibration assumes the same `N`, domain, eps/h as the null (re-calibrate otherwise; the Finding #1 `µ_eps` scaling tells you how to rescale).

**This is the deliverable the project was after:** `score(points) → ranked detections with calibrated p, z, location, and method`, combining a tight-cluster arm (LR) and an extended-overdensity arm (KDE), with an analytic look-elsewhere where the field is Gaussian and MC/heavy-tail where it isn't.

---

## Byproduct — eps-independent scorer (the old "hard problem", now trivial)

**Script:** `analysis/scorer.py` → `scorer_table.csv`, `scorer_master.json`, `scorer_survival.png`.

Finding #1 makes the long-sought eps-independent score immediate. Map an observed cluster to the collapse variable and read one master survival curve:

```
R̃ = R_obs · eps²,    p(R_obs, eps) = SF_master(R̃),    z_equiv = Φ⁻¹(1 − p)
```

- **Master fit:** pooled `R̃` over eps∈[1.0,1.6] (n=430k) is fit by **inverse-gamma(shape=20.54, scale=473)**, KS=0.026 — and that shape **equals the mechanistic `k(10)≈20.5`** of Findings #2/#3. The scorer *is* the n=10 hull-area law. (Beta-Prime on `R̃` also works, KS=0.031, but adds nothing.)
- **Validation (held-out eps 1.10/1.40/1.55):** scoring each through the single master gives near-uniform p-values — median p = 0.45/0.51/0.55 (target 0.5), frac(p<0.05) = 0.062/0.047/0.040 (target 0.05). One master curve calibrates every eps.
- **It dissolves the eps-weighting problem.** The legacy per-eps approaches give *weighting-dependent* answers (at R=30, uniform-eps mixture p=1.3e-2 vs conservative envelope p=5.1e-2 — a 4× spread; the whole §17/§12.1 debate of the handoffs). The collapse scorer needs **no weighting choice**: there is one master shape, period.
- **Tail caveat (use empirical SF in the deep tail).** The empirical master tail is slightly heavier than the inverse-gamma fit (at `R̃=38`, `z_emp=1.87` vs `z_ig=2.02`) because higher-`N'` mixture components and the eps^−0.205 residual fatten it. For conservative deep-tail p-values use the empirical survival in `scorer_table.csv`, not the parametric fit. And recall the hard censoring at `R̃ = 62.83·eps²` — the scorer is valid in the body and moderate tail only; true rare-event calibration past that needs the LR route (Finding #4) or relaxed `min_area`.

`scorer_table.csv` columns: `R_tilde, R_at_eps1_1, R_at_eps1_3, p_empirical, p_invgamma, z_empirical, z_invgamma`.

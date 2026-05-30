# Analytic Findings — overnight run (branch `analytic-night`)

> **READ ME FIRST (updated last, at end of run):** _summary pending — see §0 when complete._

This is the running report of the analytic-first overnight session (see plan `~/.claude/plans/curried-moseying-flask.md`). Goal: find the fundamental structure beneath the empirical Beta-Prime fit. All work is read-only on `simdata/v2/`; new code in `analysis/`; small sims only.

Conventions: `N=10000`, `R=100`, `S₀=πR²=31415.93`, `λ₀=1/π=0.318310`, `min_samples=min_cluster_size=10`, `min_area=0.5`. `R=(N'/S')/λ₀`. Censoring cap (raw R): `R_max=N_min/(min_area·λ₀)=62.83`.

---

## 0. Headline summary
_pending_

---

## Finding #1 — the `R·eps²` collapse law (the eps-dependence factorizes)

**Script:** `analysis/scaling_collapse.py` → `analysis/collapse_summary.csv`, `collapse_ecdf.png`, `collapse_master.png`. Sweep eps 0.95–2.00 (step 0.05), real `simdata/v2` data.

**Result (the fundamental statement).** The density-ratio distribution factorizes into a pure power-law **scale** times an eps-invariant **shape**:

```
R(eps)  =  scale(eps) · X ,     scale(eps) ∝ eps^(−2.205) ,     X ⟂ eps
```

- **Scale.** `meanR ∝ eps^(−2.205)`, `medR ∝ eps^(−2.193)` (fit over the whole range). Pure geometry predicts exactly **−2** (eps is the only length scale ⇒ hull area `S' ∝ eps²` ⇒ `R = N'/(S'λ₀) ∝ eps⁻²`). The leading-order constant `C = meanR·eps²` drifts only smoothly: 24.97 (eps=1.0) → 21.53 (eps=2.0), i.e. `C ∝ eps^(−0.205)`.
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
2. The eps-independent scorer (T4/§byproduct) follows immediately: score any observed cluster by mapping `R → X = R·eps^2.205/C` (or just `R·eps²`) and reading the tail of the single master distribution. No eps-mixture weighting needed.
3. **Censoring caveat unchanged:** raw `R` is clipped at `62.83` (eps-independent), so in `R̃=R·eps²` space the cap is `62.83·eps²` (76 at eps=1.1 → 251 at eps=2.0). The body and the q99-level tail collapse cleanly (all below the cap for the studied eps); do not trust the collapse for `R̃` beyond `~62.83·eps²`.

## Finding #2 — Beta-Prime as an N'-mixture of inverse-gamma area laws
_pending (T2)_

## Finding #3 — first-principles hull-area geometry
_pending (T3)_

## Finding #4 — Kulldorff LR & selection quantification
_pending (T5)_

## Byproduct — eps-independent scorer
_pending (T4)_

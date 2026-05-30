# Roadmap — Where To Go From Here

*Companion to [repo_state.md](repo_state.md) (what exists) and [../CLAUDE.md](../CLAUDE.md) (working memory). Written May 2026. Priorities are opinionated; the ★ items are the ones that actually move the project toward its stated goal — a calibrated, eps-independent rare-event score.*

---

## ⚡ Update (overnight `analytic-night`, see `analytic_findings.md`)

Several items below are now partly or wholly resolved — read `docs/analytic_findings.md` first:
- **§2 (eps-independent scorer): DONE.** The `R·eps²` collapse gives one master inverse-gamma(20.5); `analysis/scorer.py` ships a validated scorer. The eps-weighting debate is moot (no weights needed). Conservative envelope still available for comparison.
- **§1 (Kulldorff LR): computed.** `analysis/scan_lr.py` shows the LR per cluster, quantifies look-elsewhere (~10⁸×), and confirms LR > raw R (size-weighting). Still TODO: calibrate **max-LR per field** via re-sim (needs T-style geometry run).
- **§3 (censoring): deepened.** Confirmed `R_max=62.83` is the `min_area` artifact; the master shape's deep tail is also slightly heavier than the parametric fit. Tail work still blocked on relaxing `min_area`.
- **New TODOs from tonight:** (a) check `k(n)≈3.5n−15` hull-area-shape law against Rényi–Sulanke / Efron stochastic-geometry results; (b) model the `eps^−0.205` residual = `P(N'=n)` broadening analytically; (c) re-fit the now-bug-fixed mixture code (4-vs-5 stride fixed) to see if non-degenerate mixtures appear (likely still single-component-sufficient); (d) calibrate the scorer on `LR` (better statistic) rather than `R`.

## The goal, restated precisely

Produce a function `score(cluster, search_config) → {density_ratio, p_value, z_equiv, caveats}` that answers: *under CSR, how surprising is this detected cluster?* — honestly accounting for the fact that DBSCAN **selected** the region because it looked dense (post-selection / look-elsewhere). Everything below is in service of that.

---

## ★ 1. Fix the statistic: measure max-over-search Λ, not the pooled per-cluster ratio

**The core mismatch (see repo_state §1).** The theory docs say the correct null object is `Λ = max_Z T(Z)` over windows the selector could return, with `T` the **Kulldorff likelihood ratio** — not the raw ratio `R`. The current data pools *every* detected cluster across iterations, i.e. it's the distribution of a *randomly chosen* detected cluster, and `T = R` (a bare rate ratio). These answer different questions:

- Pooled `R` (have): "if I pick a random detected cluster, how dense is it?"
- `max Λ` per field (want, for detection): "scanning one field, how extreme is the *best* thing I'd report?"

**Action.** When re-simulating (§4), per iteration record **both**: (a) every cluster (as now, for the body), and (b) the **max LR cluster** of that field. The LR for a candidate zone with `n` points, expected `μ = N·|Z|/|S|`:

```
LR(Z) = (n/μ)^n · ((N−n)/(N−μ))^(N−n) · 1[n/μ > 1]
```

Then the detection p-value is the rank of observed `max LR` among the per-field `max LR` nulls. This is the SaTScan / spatial-scan recipe and is valid *regardless* of selector complexity because the selection is replayed. Use `R` (the existing data) for the *body* / mechanistic story; use `max Λ` for *calibration*.

Why this matters: it dissolves the "which eps weighting?" puzzle (§2) for the detection use-case — if the real detector scans `eps` and reports the best cluster, the null is `max over (eps, windows)` of the same procedure, computed by replay. No weighting choice needed.

---

## 2. If staying with the marginal-`R` framing: define the eps weighting explicitly

The unfinished modelling task (handoff_v2 §17–19, §25). Per-`eps` Beta-Prime fits exist; the missing piece is the marginal `P(R) = Σ wᵢ P(R|epsᵢ)` and `P(R≥r) = Σ wᵢ SF_i(r)`. The open question is **what `wᵢ` means**:

| scheme | answers the question | when to use |
|---|---|---|
| uniform over eps grid | "eps chosen uniformly from search range" | weak default |
| cluster-count weighted | "random detected cluster across all eps runs" | matches pooled-data semantics |
| max-over-eps (EVD) | "detector scans eps, reports most extreme" | **closest to a real detector** → but then see §1, just replay |
| conservative envelope `S_env(r)=maxᵢ Sᵢ(r)` | "worst-case tail prob over eps" | weighting-free, never overstates significance — good safe default (handoff §12.1) |

**Action (small, do-able now from existing fits):** write one script that loads `regular_fit.csv`, builds the mixture SF **and** the conservative envelope SF, exposes `p_value_for_ratio(r)`, `z_equiv_for_ratio(r) = Φ⁻¹(1−p)`, and overlays both on the **empirical merged survival** as a sanity check. Compare uniform vs cluster-count weights. Output a `ratio → {p_mix, z_mix, p_env, z_env, p_empirical}` table. The conservative envelope sidesteps the weighting argument entirely and is the recommended first deliverable. This is the "immediate next step" the handoff names and it needs no new simulation. **Caveat:** because the tail is censored at 62.8 (§3), the SF is only meaningful in the *body*; do not extrapolate past ~55 with the current data.

---

## ★ 3. The censored-tail problem must be confronted before any tail/EVD work

**Finding (repo_state §4.2):** `R_max = N_min/(min_area·λ₀) = 62.83` exactly — the right tail is clipped by the `min_area=0.5` + `min_cluster_size=10` filters, not by Poisson statistics. The `loc` floor (~4–7) is similarly filter-induced.

Implications and options:
- **Any Generalized-Pareto / Fréchet tail fit on current data is fitting the `min_area` knob.** The design doc's "fit GPD to exceedances, the tail index should match Beta-Prime's" plan is *not yet valid* on this dataset.
- To recover the *true* tail you must either (a) drop/relax `min_area` and instead control degeneracy via the LR statistic (which is size-aware and doesn't blow up for tiny hulls the way raw `R` does — this is exactly why the design doc prefers LR over raw `λ'`), or (b) treat the censoring explicitly (Tobit-style likelihood with a known upper bound).
- Sanity action: re-derive the cap symbolically into the docs (done) and **stop reporting "tail bounded ~60–70" as a physical result** — it's an artifact. The genuine, uncensored tail question is open.

---

## ★ 4. Re-simulate with rich geometry (the data-format debt)

The current 3-column format (`S', N', iteration`) blocks edge-effect and shape analysis (handoff_v2 §4, §20). This is the **highest-leverage simulation change** and worth spending compute on (budget against the ~60 h baseline; geometry adds little per-iteration cost since the hull is already computed).

Add per-cluster columns:
- cluster centroid `x, y`; radial distance from disk centre; distance to boundary; flag `touches_boundary`
- effective radius `√(S'/π)`; convex-hull **perimeter**; compactness `4πS'/P²`
- PCA eigenvalues / aspect ratio (elongation); #hull vertices
- **the LR value** and **per-field max-LR** (§1)
- optionally miniball radius (but miniball changes the measured object — see §6)

Also store a small **run-manifest** (params, seed policy, code git hash, wall-clock) alongside each CSV so the 60 h artifact is self-describing.

This finally lets you answer: are boundary clusters (truncated hulls, different geometry) inflating the body? Do elongated vs compact clusters have different `R`? Is `S'` (hull) the right area, or should it be miniball / α-shape?

---

## 5. Edge effects — suspected but unmeasured

A disk has an O(perimeter) boundary zone where convex hulls get clipped and density estimates distort. With no center/radius saved, current data cannot separate central from boundary clusters. After §4, split the dataset by `distance_to_boundary` and re-fit; if Beta-Prime params shift, edge clusters are a confound. Alternative framings: periodic/toroidal domain (kills edges, changes the problem), or a guard band (analyse only clusters fully inside `r < R − eps`).

---

## 6. Geometry of `S'` — convex hull vs alternatives

Convex hull over-counts area for non-convex/elongated clusters and is sensitive to single outlier points. Candidates to compare (after §4 gives shape data):
- **convex hull** (current) — simple, but inflates for elongated/ring shapes.
- **miniball** (min enclosing circle) — uniform circular geometry, but includes empty space and possibly non-member points → it becomes a *different scan statistic* (closer to "best circular aperture"). Changes the null.
- **α-shape / concave hull** — tighter, but α is another nuisance knob.
- **best-circular-aperture refinement**: after DBSCAN seeds a cluster, optimise a circle to maximise `λ'`. Cleaner geometry, but again a new null that must be replayed.

No free lunch: each choice defines a different `W` and thus a different null. Pick one, justify it, replay it.

---

## 7. Rarefactions are a separate object — don't force them into this model

DBSCAN sees only over-densities. Compressions and rarefactions are **not two tails of one distribution** (handoff_v2 §21); early symmetric Z-score attempts failed. If rarefaction detection is ever in scope: use a different localiser (grid/KDE deficit, or OPTICS reachability) and build it its own null. Out of scope for the current density-ratio line of work.

---

## 8. OPTICS / neighbor-graph reuse — efficiency, not correctness

OPTICS was floated to avoid fixing one `eps` and to reuse neighbor structure across thresholds (handoff_v2 §22). Verdict: it's a **performance** idea, not a statistics fix — and at this simulation scale DBSCAN is already the bottleneck. A precomputed neighbor graph / distance index *could* make multi-`eps` rescans cheaper (build once, threshold many times), which pairs well with §1's "max over (eps, windows)". Treat as an optimisation to enable §1/§4 at lower cost, not a research direction in itself.

---

## 9. The principled (heavy) alternative: Bayesian generative mixture

For completeness (clusters_problem.md §2.3): model the field as homogeneous-Poisson + latent cluster component, with cluster location/shape/count as **latent variables integrated over** (reversible-jump MCMC or DP mixture). The prior over *where a cluster could be* **is** the multiplicity correction — paid inside the model instead of bolted on. Gamma is conjugate for the rate; Beta-Binomial gives the inside-fraction marginal `Pr(n'|a,b) = C(N,n')·B(a+n', b+N−n')/B(a,b)`. This is the most defensible route and explains the Beta-Prime emergence from first principles, but it is a large build. Recommend only if the MC-replay route (§1) proves insufficient for the science.

---

## Things tried that did NOT pan out (don't relitigate)

- **2-component Beta-Prime mixture** → degenerate (identical components, repo_state §4.6). Single Beta-Prime suffices.
- **Mixture over discrete N' modes** → superseded; Beta-Prime already captures the ratio structure.
- **Exponential / Weibull** for `R` → clearly ruled out.
- **Gaussian Z-score / normality** → rejected at p~1e-50; use survival-function → `Φ⁻¹(1−p)` instead.
- **eps-conditioned global regression model** (`mixture_of_betas2.py`) → technically converges but models `P(R|eps)`, not the wanted marginal; its global KS (0.43) is not a valid like-for-like statistic.
- **Treating max_ratio ≈ 60–70 as a physical tail bound** → it's a `min_area` censoring artifact (§3).

---

## Suggested order of attack

1. **(now, cheap)** §2 marginal-`R` survival/`z_equiv` script from existing fits + empirical-survival overlay — closes the named "immediate next step", gives a usable (body-only) score.
2. **(now, cheap)** Bake the §3 censoring caveat into all tail reporting; add a Tobit or LR-based check.
3. **(compute, high value)** §4 re-simulate with geometry + per-field max-LR (§1). Budget vs the 60 h baseline.
4. **(after data)** §1 calibration by replay, §5 edge split, §6 geometry comparison.
5. **(optional, heavy)** §9 Bayesian latent-configuration model as the principled capstone.

from pathlib import Path
from textwrap import dedent

content = r"""# Cluster-Distribution Simulation Handoff

## 0. Purpose of this document

This handoff summarizes the conceptual and experimental thread around detecting weak spatial “signal” clusters inside a null field of random points. It is intended as a ground-layer continuation document for future work in the repository. The code itself lives in the repo; this document focuses on the problem framing, theoretical reasoning, empirical findings, script evolution, statistical diagnostics, modeling decisions, and unresolved questions.

The central task is to understand what kinds of dense substructures appear under a pure random null model, so that later, when a real dataset contains an apparent high-density region, we can assign an approximate null probability to that observation.

---

## 1. Original problem framing

We start with a finite spatial domain `S`, initially simplified to a circle of radius `r`. A finite number of points `N` is sampled uniformly in `S`, representing a complete spatial randomness null model. In the idealized infinite version this corresponds to a homogeneous Poisson point process with intensity:

\[
\lambda_0 = \frac{N}{|S|}
\]

The question is not merely whether a fixed pre-selected region contains unusually many points. The harder problem is:

> If an algorithm is allowed to look at the whole point cloud and select candidate dense subregions non-randomly, what distribution of apparent dense clusters does pure noise produce?

This is a multiple-testing / look-elsewhere / algorithmic-selection problem. The selected region is biased by the detection procedure, so the naive Poisson or binomial probability for a fixed cell does not apply directly.

The original coarse approach used a fixed grid:

- Partition `S` into `K` equal non-overlapping cells.
- Count points per cell.
- Use binomial / Poisson approximations for cell occupancy.
- Estimate the probability of observing at least a given count in one or more cells.
- Use Poisson Z-like thresholds.

That coarse grid method is valid as a proof of concept because candidate regions are pre-defined and independent enough. But it is too spatially crude and does not capture arbitrary cluster shapes.

The refined approach used density-based clustering, especially DBSCAN, to select candidate subregions. This immediately breaks the fixed-cell assumptions, because the candidate region is chosen after seeing the data.

---

## 2. Geometry and scale considerations

Several important invariance and boundary observations emerged.

### 2.1 Scale invariance

For a homogeneous Poisson point process, rescaling the spatial domain rescales `lambda` but should preserve dimensionless structure. For example, squashing an area of 4 into an area of 1 multiplies density by 4 but preserves the relative spatial configuration. Therefore the relevant quantities should often be dimensionless, such as:

\[
\frac{\lambda'}{\lambda_0}
\]

where:

\[
\lambda' = \frac{N'}{S'}
\]

for a detected cluster.

This pushed the work toward density ratios rather than raw densities.

### 2.2 Boundary effects

Shape of the global domain `S` matters near borders. A circular domain and star-shaped domain do not produce the same boundary behavior. However, far from the boundary, the process should approximate the infinite-plane limit.

This led to the idea that edge effects can be managed by storing cluster centers and effective radii, then filtering clusters whose enclosing region intersects or approaches the boundary. The initial datasets did not store this information, which limited later post-analysis.

### 2.3 Need for cluster center and radius

The current minimal stored fields were:

- `S_prime`: cluster area
- `N_prime`: number of points in cluster
- `iteration`: simulation iteration

Later it became clear this is insufficient. Future simulation output should also include:

- cluster center, probably point centroid / center of mass
- distance of center from origin
- angle of center
- an effective cluster radius
- possibly convex hull radius, miniball radius, or PCA axes
- shape descriptors

The most natural center for density analysis is the center of mass of cluster points:

\[
c = \frac{1}{N'} \sum_i x_i
\]

For edge filtering, a scalar radius is also needed. Candidate options:

- effective radius from convex hull area: \(\sqrt{S'/\pi}\)
- minimum enclosing circle / miniball radius
- maximum distance from centroid
- PCA major/minor axes

Miniball gives a clean boundary condition but changes the conceptual cluster shape: it forces circular support and may include additional points that DBSCAN did not assign to the cluster.

---

## 3. Simulation setup

The main simulation used:

- Circular domain.
- `N = 10,000` points.
- Points generated uniformly in polar coordinates:
  - radius proportional to `sqrt(U)`
  - angle uniform in `[0, 2π)`
- DBSCAN for cluster extraction.
- Convex hull area for `S_prime`.
- Minimum cluster size around 10.
- Very small / degenerate clusters filtered out.
- Multiprocessing used for speed.
- Data persisted as CSV so long simulations can resume and plotters can run separately.

Early runs explored `eps` behavior and established a practical range:

- At `eps ≈ 0.84`, no clusters are found.
- At `eps ≈ 2.86`, clusters become too large and hit the upper area constraint.
- The later useful range became approximately `eps = 1.0 .. 1.7`, with detailed modeling focused on `eps = 1.10 .. 1.40`.

A long run with 1M simulations showed that rare-event statistics need large samples. Smaller runs around 10k simulations can show coarse means but are insufficient for stable tail behavior.

A practical lower bound emerged:

- Fewer than ~2,000 clusters: fitted parameters vary too much.
- Around 10,000+ clusters: Gamma / Beta-prime parameters become more stable.

---

## 4. Script evolution and design choices

The script family evolved through several stages.

### 4.1 Initial simulator and real-time plotter

The first implementation did both simulation and plotting. This became impractical because:

- plotting slowed simulation
- matplotlib UI stalled after enough data
- interactive windows behaved inconsistently across backends
- the plot could disappear or block execution

The design shifted to separate scripts:

- simulator: generate and save data
- plotter: read saved data and plot once
- stat scripts: analyze saved CSVs

### 4.2 Multiprocessing issues

Early multiprocessing produced repeated or sparse valid clusters. The main culprit was random number generation state inheritance and seeding.

The fix was to give each worker or iteration an independent NumPy random generator, seeded from the parent or otherwise uniquely. After this, all processes contributed properly.

### 4.3 CSV writing consistency

When multiprocessing was introduced, there were concerns about data consistency and mutexes. The eventual pattern was:

- worker processes return results
- parent process appends batch results to CSV
- no worker writes directly

This avoided write races.

### 4.4 Placeholder rows

When no cluster is found for an iteration, the row uses sentinel values:

- `S_prime = -1.0`
- `N_prime = -1`

This avoids pandas nullable integer problems and keeps parsing simple. Plotting and analysis scripts filter these rows out.

### 4.5 Storage minimization

To reduce storage size, only the essential fields were kept:

- `S_prime`
- `N_prime`
- `iteration`

This helped with long runs, but later proved too restrictive because centers, boundary distances, and shape diagnostics could not be reconstructed.

---

## 5. Epsilon sweep findings

A script was created to sweep `eps` upward and evaluate cluster properties. The termination criteria were:

- if 1000 iterations yielded zero clusters, `eps` is too small / harsh, continue upward
- if maximum `S_prime` exceeds one quarter of the total circular area, stop

Initial attempted scoring function:

\[
\frac{\lambda'}{\lambda_0} \cdot \frac{S'}{S_0}
\]

turned out to grow monotonically and was dominated by the area term. It was not useful for identifying an optimal density-sensitive regime.

The density-only ratio:

\[
R = \frac{\lambda'}{\lambda_0}
\]

became the main observable.

### 5.1 `S_prime`

Both average and maximum `S_prime` grow monotonically with `eps`. This is expected:

- larger `eps` expands neighborhoods
- DBSCAN clusters absorb more points
- hulls grow
- cluster boundary length increases the chance of absorbing nearby points

### 5.2 `N_prime`

The number of points in clusters is discrete and heavily concentrated near the DBSCAN threshold.

For example around `eps = 1.4`, the approximate distribution was:

- `N_prime = 10`: ~68%
- `N_prime = 11`: ~15%
- `N_prime = 12`: ~8%
- `N_prime = 13`: ~6%
- `N_prime = 14`: ~3%
- `N_prime = 15`: ~1%
- `N_prime = 16`: ~0.4%

This means DBSCAN mostly detects threshold-minimal clusters, with a right tail that elongates as `eps` increases.

The discrete nature of `N_prime` seemed likely to induce step-like / mixture effects in `lambda_prime = N_prime / S_prime`.

### 5.3 `lambda_prime / lambda0`

The mean density ratio decreases with `eps`. Visually it looked like a function similar to:

\[
c_1 - \sqrt{\epsilon - c_2}
\]

or at least a smooth monotone decline.

The maximum density ratio in long simulations seemed to approach a quasi-constant tail boundary. Earlier estimates suggested a tail around 60–70, with values around 64 ± 5 in long runs. This was interpreted as possibly reflecting an intrinsic extreme-fluctuation envelope of the null distribution rather than a property of DBSCAN alone.

This “max density constant” was one of the main motivating observations.

---

## 6. Statistical testing: early phase

### 6.1 Normality

Normality was tested visually and with standard tests. It was rejected.

For density ratio distributions, the shape is:

- steep rise on the left
- longer right tail
- asymmetric
- not Gaussian

For `N_prime`, the observed distribution is a DBSCAN-thresholded tail. Since all values below the minimum cluster size are cut away, normality or Poisson analysis is distorted. It is not a full natural count distribution, but a truncated / selected distribution induced by the algorithm.

### 6.2 Poisson tests

`N_prime` is discrete and was initially considered a possible Poisson target. However, Poisson goodness-of-fit was rejected. Reasons:

- DBSCAN imposes a minimum threshold.
- Observed values are selected clusters, not independent raw counts.
- `N_prime` distribution is a truncated, algorithm-conditioned tail.
- Spatial dependence and cluster merging violate simple Poisson assumptions.

### 6.3 Gamma fit

The density ratio initially looked Poisson-like but is continuous. Gamma was tested as a natural continuous positive skewed distribution.

Gamma performed well visually compared to exponential, Weibull, and log-normal in early tests:

- Exponential ruled out.
- Weibull poor.
- Log-normal close but visibly shifted.
- Gamma best among early candidates.

But Gamma had systematic errors:

- fitted peak skewed right of actual peak
- fitted peak height too low
- left shoulder overestimated
- tails often reasonable

This hinted that Gamma was a good approximation but not the full structure.

---

## 7. Mixtures and the role of `N_prime`

A key hypothesis was that the distribution is a mixture over discrete `N_prime` modes:

\[
P(R) = \sum_n P(N' = n) P(R \mid N'=n)
\]

Because `N_prime` is concentrated around 10–20, with `N=10` dominant, it was plausible that each `N_prime` value contributed a slightly different Gamma-like component.

Scripts tested:

1. Single Gamma per `eps`.
2. Mixture of Gammas by `N_prime`.
3. Single Gamma for the main mode only, `N_prime = 10`.

Findings:

- Mixture by `N_prime` was systematically but only slightly better than single Gamma in AIC/BIC.
- With same sample size, KS, AIC, and BIC generally agreed that the mixture was better.
- `N_prime = 10` alone fit better for that subset, but AIC/BIC cannot be directly compared to full-data models because the sample differs.
- This did show that discrete `N_prime` influences fit quality, but it did not fully explain the global distribution.

Later, when Beta-prime fits were introduced, the `N_prime` mixture hypothesis became less central.

---

## 8. Regression and epsilon dependence

A major question was whether `eps` merely affects cluster detection frequency, or whether it changes the distribution of density ratios themselves.

Parameter fits showed clear `eps` dependence.

For Beta-prime fits, approximate linear regressions over sampled eps values yielded high R² values:

- parameter `a`: R² around 0.90
- parameter `b`: R² around 0.86
- `loc`: R² around 0.99
- `scale`: R² around 0.90 in one run, but later scale became nearly constant in merged optimization

Example regression from one phase:

- `a` slope ~50
- `b` slope ~3.6
- `loc` slope ~-11.6
- `scale` slope ~-4.6

Later optimized regression for merged likelihood gave:

- `a_slope ≈ 51.26`
- `a_intercept ≈ -8.27`
- `b_slope ≈ 11.12`
- `b_intercept ≈ -3.97`
- `loc_slope ≈ -21.03`
- `loc_intercept ≈ 32.01`
- `scale_slope ≈ 0.005`
- `scale_intercept ≈ 1.61`

This suggests the parameterization depends substantially on the fitting objective and whether fitting is per-eps or merged.

### 8.1 ANOVA discussion

ANOVA was considered to test whether `eps` influences parameters. A direct ANOVA failed / was inappropriate because each `eps` had only one parameter estimate per parameter. Having shape and scale does not solve this, because for each parameter individually there is still only one observation per group.

A viable alternative was proposed:

- take K random subsamples for each `eps`
- fit each subsample
- obtain K estimates per `eps`
- perform ANOVA or mixed modeling on those replicate estimates

This was not yet fully developed in the handoff, but it remains a good route for estimating parameter dispersion and testing stability.

### 8.2 Regression as practical alternative

Linear regression of parameters vs `eps` was used instead. Slopes were nonzero and statistically significant. This means:

- `eps` is not negligible.
- Merging data across `eps` without correction changes distribution shape.
- Any epsilon-independent null model must either:
  - integrate over `eps`, or
  - choose a conservative envelope, or
  - condition on `eps`.

---

## 9. Beta-prime discovery

A major turn was fitting additional positive skewed distributions separately by `eps`, including:

- Gamma
- Log-normal
- Weibull
- Beta-prime

Exponential was dropped after poor performance.

Location parameter `floc` was tested. `floc = 5` outperformed `floc = 0` in all cases initially. Later, regular fitting of `loc` directly worked better.

Beta-prime won most cases. Occasionally log-normal won under fixed `floc = 0`, but with free / better `loc`, Beta-prime became the main candidate.

### 9.1 Empirical per-eps Beta-prime fits

Sample fits:

- `eps = 1.10`
  - `a ≈ 39.48`
  - `b ≈ 10.06`
  - `loc ≈ 7.46`
  - `scale ≈ 3.01`
  - KS ≈ 0.0025, p ≈ 0.98

- `eps = 1.15`
  - `a ≈ 43.50`
  - `b ≈ 9.77`
  - `loc ≈ 6.86`
  - `scale ≈ 2.39`
  - KS ≈ 0.0033

- `eps = 1.20`
  - `a ≈ 45.35`
  - `b ≈ 10.02`
  - `loc ≈ 6.13`
  - `scale ≈ 2.18`
  - KS ≈ 0.0027

- `eps = 1.25`
  - `a ≈ 49.78`
  - `b ≈ 10.23`
  - `loc ≈ 5.49`
  - `scale ≈ 1.89`
  - KS ≈ 0.0045

- `eps = 1.30`
  - `a ≈ 53.64`
  - `b ≈ 10.55`
  - `loc ≈ 4.94`
  - `scale ≈ 1.69`
  - KS ≈ 0.0051

- `eps = 1.35`
  - `a ≈ 53.84`
  - `b ≈ 10.73`
  - `loc ≈ 4.45`
  - `scale ≈ 1.59`
  - KS ≈ 0.0052

- `eps = 1.40`
  - `a ≈ 53.17`
  - `b ≈ 10.91`
  - `loc ≈ 4.05`
  - `scale ≈ 1.54`
  - KS ≈ 0.0060

Observations:

- Fit is excellent at lower eps and gradually worsens but remains good.
- Peak shifts left and grows taller with increasing `eps`.
- Tail remains long and max values remain around the earlier observed boundary.
- Beta-prime captures peak and left shoulder better than Gamma.

### 9.2 Theoretical interpretation of Beta-prime

Beta-prime arises naturally as the ratio of two independent Gamma variables with a common scale:

\[
X \sim \Gamma(a, \theta), \quad Y \sim \Gamma(b, \theta)
\]

then:

\[
\frac{X}{Y} \sim \text{BetaPrime}(a, b)
\]

This is highly suggestive because the measured statistic is a density ratio:

\[
R = \frac{\lambda'}{\lambda_0}
\]

and the numerator and denominator can be viewed as outcomes of two stochastic aggregation processes:

- numerator: local selected cluster density
- denominator / normalizing baseline: global or ambient density scale

Even though `lambda0` is fixed in the finite simulation when `N` and `S` are fixed, the algorithmic selection process may introduce an effective random denominator through the cluster area, neighborhood geometry, or local spacing scale.

A plausible mechanism:

- DBSCAN-selected cluster density depends on local nearest-neighbor / spacing geometry.
- Spacings in a Poisson process have exponential-like components.
- Sums or products of spacing-derived variables can yield Gamma-like quantities.
- A density ratio involving count and random area behaves like a ratio of Gamma-like random variables.
- Therefore Beta-prime emerges as a natural distribution for selected density ratios.

This is not yet a proof, but it is a strong theoretical clue.

### 9.3 Why mixtures helped

The earlier Gamma mixture approach worked because Beta-prime can be related to Gamma ratios. The mixture over `N_prime` approximated some of the extra flexibility needed to capture the ratio distribution. Once Beta-prime was introduced directly, the mixture of Beta-primes tended to numerically collapse to identical components, suggesting no strong distinct modes at that level.

---

## 10. Mixture model issues

A two-component Beta-prime mixture was attempted. Results often converged to identical components:

- Component 1 and Component 2 parameters equal or nearly equal.
- Mixing proportion stayed around 0.5.
- AIC/BIC penalized the extra parameters.
- No evidence for true distinct Beta-prime modes.

This suggests that, at least for the density ratio marginal distribution, a single Beta-prime per `eps` is sufficient.

There was a KS bug in mixture code: the mixture PDF was mistakenly used where a CDF was required. This produced nonsensical KS statistics such as 1.0. Any future mixture KS must use a proper mixture CDF:

\[
F_{\text{mix}}(x) = \sum_i w_i F_i(x)
\]

not the PDF.

---

## 11. Merged-eps modeling

An attempt was made to model merged data by optimizing linear parameter functions:

\[
a(\epsilon) = m_a \epsilon + c_a
\]
\[
b(\epsilon) = m_b \epsilon + c_b
\]
\[
loc(\epsilon) = m_l \epsilon + c_l
\]
\[
scale(\epsilon) = m_s \epsilon + c_s
\]

This optimized successfully, but the result was not what was ultimately needed. It provided a conditional model \(P(R \mid \epsilon)\), not a single epsilon-independent probability distribution.

The optimized model had:

- AIC ≈ 14,485,165.71
- BIC ≈ 14,485,268.40
- KS ≈ 0.4345 when evaluated as a merged-data model

The high KS reflects that merged data is not well-described by simply sorting all `R` values and comparing each to its conditional CDF in that way. More importantly, the user clarified the goal:

> We need an approximate epsilon-independent probability function for rare-event scoring from an observed cluster lambda.

That is different from estimating how Beta-prime parameters depend on eps.

---

## 12. Current target: epsilon-independent null survival function

The desired final object is a function that takes an observed cluster density ratio:

\[
R_{\text{obs}} = \frac{\lambda'_{\text{obs}}}{\lambda_0}
\]

and returns something like:

\[
P(R \ge R_{\text{obs}} \mid \text{null})
\]

or a Z-equivalent score.

The purpose is to score rare dense clusters in future data.

### 12.1 Why this is subtle

Because `eps` changes the distribution, merging eps values naively produces a mixture distribution:

\[
P(R) = \int P(R \mid \epsilon) P(\epsilon) d\epsilon
\]

In discrete simulation:

\[
P(R) = \sum_i w_i P(R \mid \epsilon_i)
\]

where weights depend on how eps values are sampled or how much each eps is intended to matter.

This is not a single physical distribution unless the eps selection rule is defined.

There are several possible null-score definitions:

1. **Conditional score**
   - Given the algorithm uses a known fixed `eps`, use \(P(R \ge r \mid \epsilon)\).
   - Most statistically defensible.

2. **Mixture score over eps**
   - If eps is not fixed and is treated as part of the search procedure, integrate over eps values with chosen weights.
   - This accounts for eps-selection uncertainty.
   - But the weights must be justified.

3. **Conservative envelope**
   - For each `r`, take the maximum tail probability over eps:
     \[
     P_{\text{env}}(R \ge r) = \max_\epsilon P(R \ge r \mid \epsilon)
     \]
   - Conservative: does not overstate significance.
   - Useful if eps is tuned adaptively.

4. **Extreme-over-eps score**
   - If the detection algorithm actively sweeps eps and reports the most significant cluster, then the null distribution should be the distribution of the maximum statistic over eps:
     \[
     R_{\max} = \max_\epsilon R_\epsilon
     \]
   - This requires simulation or dependence-aware approximation.
   - This is the correct look-elsewhere version if eps is optimized after seeing data.

The current user request is closest to an approximate epsilon-independent probability function. The safest recommendation is to produce both:

- a mixture survival function over eps
- a conservative envelope survival function

and make clear what each means.

### 12.2 Z-equivalent scoring

For a tail probability \(p\), a one-sided Gaussian-equivalent Z score is:

\[
Z = \Phi^{-1}(1 - p)
\]

This allows comparing rare-event probabilities on a familiar scale, even if the underlying distribution is not Gaussian.

A “Z-score” in this context should not mean:

\[
\frac{x - \mu}{\sigma}
\]

because the distribution is skewed and heavy-tailed. Tail-probability Z-equivalent is more appropriate.

---

## 13. Known empirical facts to preserve

These are the key observations that should guide future continuation:

1. DBSCAN-selected clusters under pure noise produce a structured distribution, not arbitrary chaos.
2. `eps` has a strong effect on the distribution shape.
3. `S_prime` grows monotonically with `eps`.
4. `N_prime` is heavily concentrated at the DBSCAN minimum, usually 10.
5. `lambda_prime / lambda0` is skewed right, with steep left rise and long right tail.
6. Normality is rejected.
7. Poisson for `N_prime` is rejected due to selection/truncation.
8. Gamma is a decent but systematically biased approximation.
9. Beta-prime fits `lambda_prime / lambda0` extremely well per eps.
10. Free or properly fitted location is important.
11. Two-component Beta-prime mixtures tend to collapse to one component.
12. Per-eps Beta-prime parameters are strongly correlated with eps.
13. The maximum observed density ratio seems bounded around ~60–70 in long simulations, at least under current settings.
14. A practical analysis threshold is at least ~2,000 clusters, preferably 10,000+.
15. Existing data lacks center/radius/shape fields, limiting boundary and geometry corrections.

---

## 14. Important pitfalls encountered

### 14.1 Comparing AIC/BIC across different sample sizes

AIC/BIC should not be compared directly between models fitted to different datasets, such as full data vs only `N_prime = 10`. Use per-observation log-likelihood or compare only on the same data.

### 14.2 KS p-values with fitted parameters

KS p-values are not strictly valid when distribution parameters are estimated from the same data. The statistic is still useful for relative diagnostics, but p-values can be misleading, especially with huge samples.

### 14.3 Mixture KS must use CDF

Do not pass a PDF to `kstest` as if it were a CDF. For a mixture:

\[
F(x) = \sum_i w_i F_i(x)
\]

### 14.4 ANOVA needs replicates

One fitted parameter per eps is not enough for ANOVA. Use repeated subsampling fits if ANOVA or parameter dispersion estimates are needed.

### 14.5 Merging eps values changes the model

If data from multiple eps values is merged, the resulting distribution is a mixture over eps. It is not the same as a single per-eps distribution and should not be interpreted without defining eps weights.

---

## 15. Recommended next modeling script

The next useful script should build an epsilon-independent scorer from the per-eps Beta-prime fits already collected.

Inputs:

- `regular_fit.csv` or equivalent containing:
  - `eps`
  - `a`
  - `b`
  - `loc`
  - `scale`
  - maybe `AIC`, `BIC`, `KS`
  - maybe repeated subsample fit rows
- optional raw data for validation

Outputs:

1. A mixture survival function:
   \[
   S_{\text{mix}}(r) = \sum_i w_i S_i(r)
   \]
   where \(S_i(r) = 1 - F_i(r)\) from Beta-prime at eps `i`.

2. Conservative envelope survival:
   \[
   S_{\text{env}}(r) = \max_i S_i(r)
   \]

3. Optional empirical survival table from raw merged data:
   \[
   \hat{S}(r) = \frac{\#\{R_j \ge r\}}{n}
   \]

4. Z-equivalent:
   \[
   Z(r) = \Phi^{-1}(1 - S(r))
   \]

5. A CSV table over a grid of ratios:
   - `ratio`
   - `p_mix`
   - `z_mix`
   - `p_env`
   - `z_env`
   - optionally `p_empirical`
   - optionally per-eps tail probabilities

6. Optional interpolation function / JSON of fitted parameters.

### 15.1 Weighting choices

Possible weights for the eps-mixture:

- uniform over eps values
- proportional to number of clusters observed at that eps
- proportional to number of simulation iterations per eps
- user-defined prior over eps
- conservative envelope instead of weighted mixture

Uniform over eps is easiest but arbitrary.

Cluster-count weighting answers:

> If I pool all detected clusters from all eps files, what is the distribution of a random detected cluster?

Uniform-eps weighting answers:

> If eps is chosen uniformly first, then a random cluster is drawn from that eps, what is the distribution?

These are different. The script should make the choice explicit.

---

## 16. Future simulation changes

The next simulation version should store richer cluster geometry:

- `x_center`
- `y_center`
- `r_center`
- `theta_center`
- `r_eff_from_hull`
- `miniball_radius`
- maybe PCA major/minor axes
- convex hull perimeter
- cluster aspect ratio
- boundary distance:
  \[
  d_{\text{edge}} = R_{\text{domain}} - r_{\text{center}}
  \]
- edge-safe boolean:
  \[
  d_{\text{edge}} > r_{\text{cluster}}
  \]

This allows:

- boundary filtering
- center uniformity tests
- angular autocorrelation tests
- shape-conditioned distributions
- comparison of circular vs elongated cluster effects
- possible rarefaction analysis

---

## 17. Rarefactions / voids

The current DBSCAN workflow primarily detects compressions / dense clusters. It does not detect rarefactions / voids.

Earlier conceptual discussion noted that rarefactions and compressions may not be opposite tails of one distribution. They may be separate algorithm-conditioned distributions. The old grid Z-score approach produced systematic asymmetry, such as compressions around +3 corresponding to rarefactions around -4. This suggests a separate treatment is needed.

Possible future approaches:

- OPTICS, because it provides reachability / density structure across scales.
- Empty-ball / void-finding methods.
- Delaunay / Voronoi cell area analysis.
- Local nearest-neighbor distance statistics.
- KDE level-set methods.
- Compare compression and rarefaction distributions separately before attempting any unified score.

OPTICS was considered but postponed because:

- DBSCAN already took long runs.
- OPTICS post-processing is more complex.
- The current null is roughly uniform, not a multi-density biological/proteomics-like dataset where OPTICS shines.

However, OPTICS may still be valuable as a “canned food” precomputation because it captures neighborhood distance structure that can be reused across eps-like thresholds.

---

## 18. Open theoretical questions

### 18.1 Why Beta-prime exactly?

Hypothesis:

- Poisson spatial geometry creates exponential-like nearest-neighbor spacing components.
- Cluster area and cluster density are functions of aggregated spacings.
- Aggregated exponential components produce Gamma-like variables.
- Density ratio behaves like a ratio of Gamma-like quantities.
- Ratio of Gamma variables yields Beta-prime.

Needed future work:

- derive relationship between DBSCAN cluster area and Gamma variables
- test whether area or inverse area conditional on `N_prime` follows Gamma / inverse-Gamma
- test whether `lambda_prime` can be decomposed as a Gamma-ratio directly
- compare Beta-prime parameters to expected shape parameters from `N_prime` or local neighbor count

### 18.2 Why max ratio ~60–70?

Possible interpretations:

- finite simulation tail limit from 1M runs
- current DBSCAN/min_cluster_size/convex-hull estimator induces a practical envelope
- entropy / randomness constraints of the finite point process
- order-statistic extreme of a Beta-prime-like distribution
- artifact of `N=10000`, radius 100, min cluster size 10, and eps range

Needs testing across:

- different `N`
- different radius with same density
- different density with same radius
- different min cluster size
- edge-filtered vs unfiltered data
- PRNG comparisons
- possibly true QRNG if philosophically relevant to later work

### 18.3 How to handle eps in final significance?

Possible options:

- fixed-eps conditional p-values
- uniform eps mixture
- cluster-count weighted mixture
- conservative envelope
- max-over-eps null simulation

For adaptive detection, the max-over-eps null is statistically cleanest but requires more simulation or careful approximation.

---

## 19. Current conceptual conclusion

The project has moved from a fixed-cell Poisson intuition to an algorithm-conditioned null distribution.

The main empirical discovery is:

> Under homogeneous random points in a circular domain, DBSCAN-selected dense clusters have density ratios that are well modeled by Beta-prime distributions, with parameters strongly and smoothly dependent on `eps`.

This is a significant improvement over naive Poisson Z-scoring. It suggests that dense-cluster significance should be scored against the selected-cluster null distribution, not against raw cell counts.

The next practical milestone is to build a reusable epsilon-independent tail-probability scorer, ideally using per-eps Beta-prime fits and exposing both mixture and conservative-envelope survival functions.

---

## 20. Minimal continuation plan

1. Consolidate all per-eps Beta-prime fit CSVs.
2. Build survival-function script:
   - load fits
   - choose weighting mode
   - compute mixture survival
   - compute conservative envelope
   - output ratio-to-p-value/Z table
3. Validate against empirical merged survival from raw data.
4. Add repeated subsampling fits to estimate parameter uncertainty.
5. Plot:
   - per-eps CDF/SF
   - mixture SF
   - envelope SF
   - empirical SF
   - log-scale tail comparison
6. Update simulator to store geometry:
   - centers
   - radii
   - PCA / shape
   - boundary distance
7. Run new simulation in `eps = 1.0 .. 1.7`, prioritizing `1.1 .. 1.4`.
8. Refit Beta-prime models with edge-filtered data.
9. Revisit rarefactions separately.

---

## 21. Glossary

- `S`: full spatial domain.
- `S_prime`: detected cluster area.
- `N`: total number of points in the domain.
- `N_prime`: number of points in a detected cluster.
- `lambda0`: global point density, \(N / S\).
- `lambda_prime`: cluster density, \(N' / S'\).
- `density ratio`: \(lambda_prime / lambda0\).
- `eps`: DBSCAN neighborhood radius.
- `min_samples`: DBSCAN core-point threshold.
- `min_cluster_size`: post-filter minimum number of points in a detected cluster.
- `CSR`: complete spatial randomness.
- `KS`: Kolmogorov-Smirnov statistic.
- `AIC/BIC`: model-selection criteria.
- `Beta-prime`: positive continuous distribution, ratio of Gamma variables.
- `survival function`: \(S(x)=P(X \ge x)=1-F(x)\).
- `Z-equivalent`: one-sided Gaussian score corresponding to a tail probability.

---

## 22. Final note

The key shift is philosophical as much as technical:

The question is no longer “how unlikely is this count in this area?” but:

> “How unlikely is this cluster statistic after the same algorithm has searched random data for clusters?”

That is the correct null for this project.
"""

path = Path("/mnt/data/cluster_distribution_handoff.md")
path.write_text(content, encoding="utf-8")
print(f"Wrote {path} ({path.stat().st_size} bytes)")
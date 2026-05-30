# Cluster-Density Null Model Handoff

## Purpose of this handoff

This document summarizes the reasoning, experiments, findings, and open directions from the prior discussion about detecting weak spatial density anomalies in random point fields. It is intended as a continuation layer for future work on the repo code, without duplicating implementation details already present there.

The central project is to build an empirical/statistical null model for apparent point clusters in a finite spatial Poisson process, then use that model to estimate how surprising an observed local density is. The eventual goal is an epsilon-independent or at least algorithm-aware probability function for rare high-density clusters.

---

## 1. Original problem

We start with a finite area \(S\), initially simplified to a circular domain, populated by \(N\) random points. The null hypothesis is complete spatial randomness: points are drawn from a uniform spatial Poisson process with global intensity

\[
\lambda_0 = \frac{N}{S}.
\]

The possible alternative is that there may be an added local concentration of points inside some subarea \(S' \subset S\), producing a local intensity

\[
\lambda' = \frac{N'}{S'}.
\]

The hard part is not merely finding dense-looking regions, but assigning them a valid null probability after the region was selected by a density-seeking algorithm.

### Two-part problem

1. **Candidate selection**  
   Find subareas \(S'\) that look like clusters or density anomalies.

2. **Statistical discrimination**  
   Decide whether the candidate density is plausible under the null hypothesis or too rare to treat as noise.

The first crude method used a fixed non-overlapping grid and binomial/Poisson approximations. This made probability accounting easier because the searched cells were predefined and independent enough for a first approximation.

The refined method used DBSCAN and density maps to select arbitrary clusters. This immediately introduced selection bias: the cluster region is selected *because* it is dense, so one cannot directly apply a naive Poisson \(Z\)-score to the selected area as if it had been chosen independently.

---

## 2. Why analytical treatment became difficult

The analytical difficulty comes from the fact that arbitrary subarea selection creates a huge search space. Even if \(S\) is fixed, the algorithm implicitly scans many possible shapes and locations. The probability of “some dense subregion somewhere” depends on:

- total domain area and shape;
- total number of points;
- boundary effects;
- the clustering algorithm;
- algorithm parameters, especially DBSCAN `eps` and `min_samples`;
- minimum cluster size and minimum cluster area filters;
- how \(S'\) is defined from a detected cluster, such as convex hull, miniball, or other geometry.

A key insight was that the null distribution is not simply a Poisson count distribution for \(N'\), because \(S'\) is random and algorithmically selected. The resulting variable of interest,

\[
\lambda' = \frac{N'}{S'},
\]

is a compound, biased, algorithm-dependent statistic.

---

## 3. Early simulation strategy

The first robust direction was brute-force simulation under the null hypothesis:

1. Generate \(N=10{,}000\) points uniformly in a circle.
2. Cluster them using DBSCAN.
3. Extract cluster properties:
   - \(N'\), number of points in cluster;
   - \(S'\), cluster area, initially via convex hull;
   - \(\lambda' = N'/S'\);
   - ratio \(R = \lambda'/\lambda_0\).
4. Repeat many times.
5. Empirically estimate the null distribution of apparent clusters.

Uniform point generation in a circle was simplified using polar coordinates:

\[
r = R\sqrt{U}, \quad \theta = 2\pi V
\]

where \(U,V \sim U(0,1)\).

The simulation was moved from interactive plotting to a split simulator/plotter workflow because plotting during simulation caused stalls and GUI/backend issues.

---

## 4. Data format evolution

Early saved data contained more columns, including timing data. It was then reduced for storage efficiency to:

- `S_prime` as float;
- `N_prime` as int;
- `iteration` as int.

Placeholder rows use `S_prime = -1.0` and `N_prime = -1` for iterations where no valid cluster was found. This avoided nullable integer complications in pandas.

Later it became clear that this reduced format limits post-hoc analysis. Future simulation runs should probably also save:

- cluster center coordinates;
- cluster radial distance from domain center;
- cluster angular position;
- effective radius or boundary distance;
- possibly convex hull perimeter;
- PCA shape descriptors;
- miniball radius if that geometry is used;
- maybe DBSCAN label count per iteration.

The lack of cluster center and radius data prevents current data from separating edge effects from central clusters.

---

## 5. Multiprocessing and RNG lessons

Multiprocessing was introduced to speed up simulation. A critical issue appeared: child processes could inherit identical NumPy RNG state under fork-like multiprocessing behavior, causing repeated or synchronized outputs.

The fix was to generate independent seeds in the parent process and pass them into worker calls. Each worker creates a local `np.random.default_rng(seed)`.

This solved the “only one of sixteen workers is effectively doing unique work” issue.

---

## 6. DBSCAN parameter exploration

The simulation was expanded to scan DBSCAN `eps`.

### Empirical constraints

For \(N=10{,}000\), radius \(=100\), `min_samples=10`, and minimum cluster size around 10:

- Below approximately `eps ≈ 0.84`, no clusters are found.
- Around `eps ≈ 2.86`, cluster areas become too large and hit the upper area constraint.
- A useful exploration interval emerged around `eps ≈ 0.9–1.4`, later narrowed for some analyses to `1.10–1.40`.
- For future longer runs, `eps ≈ 1.0–1.7` was suggested as a practical range.

### Area behavior

Mean and maximum \(S'\) grow monotonically with `eps`.

This is expected because a larger neighborhood radius makes it easier for DBSCAN to connect points and absorb nearby points. The probability of adding more points grows with available boundary length / hull perimeter.

### Cluster count behavior

The number of detected valid clusters rises sharply with `eps`. Low `eps` values produce too few clusters for stable parameter estimates.

A rough empirical threshold emerged:

- fewer than ~2,000 valid clusters: parameter estimates are unstable;
- ~10,000+ valid clusters: gamma/beta-prime parameter trends become much more stable.

---

## 7. Important finding: density ratio behavior

The initial signal/noise ratio considered was:

\[
\frac{\lambda'}{\lambda_0} \cdot \frac{S'}{S_0}.
\]

This produced an exponential-like monotonic increase with `eps`, dominated by the area factor \(S'/S_0\). It was not useful for identifying intrinsic density rarity.

The more useful statistic became:

\[
R = \frac{\lambda'}{\lambda_0}.
\]

### Early empirical behavior

For `eps ≈ 0.9–1.4`, the maximum observed density ratio stabilized around a tail near 60–70 in million-scale simulations. This suggested a null-distribution tail that was more about the stochastic geometry of random fields than about the exact `eps` value, at least for extreme observed maxima.

However, later histogram overlays showed that the full distribution is definitely `eps`-dependent.

### Refined empirical observation

For `eps = 1.10–1.40`, overlapping histograms showed:

- peak shifts left as `eps` increases;
- peak height increases as `eps` increases;
- approximate examples:
  - around `eps=1.10`: peak density ~0.075 at ratio ~20;
  - around `eps=1.40`: peak density ~0.150 at ratio ~12;
- the distribution has a steep left shoulder and a longer right tail;
- the far tail still remains bounded in practice around ratio ~60–70 in observed samples.

---

## 8. Cluster size \(N'\) behavior

The distribution of \(N'\) is discrete and heavily dominated by the DBSCAN threshold. For example, around `eps=1.4`:

- \(P(N'=10) \approx 0.68\)
- \(P(N'=11) \approx 0.15\)
- \(P(N'=12) \approx 0.08\)
- \(P(N'=13) \approx 0.06\)
- \(P(N'=14) \approx 0.03\)
- \(P(N'=15) \approx 0.01\)
- \(P(N'=16) \approx 0.004\)

This means DBSCAN strongly truncates the count distribution at the minimum cluster size. The algorithm only sees the high-density tail of an underlying process; rarefactions are absent.

Initial speculation was that the density distribution might be a mixture over discrete \(N'\) modes. This turned out to be partially true but not the final explanation.

---

## 9. Normality and Poisson tests

Normality was rejected both visually and statistically.

For \(N'\), Poisson tests failed badly, which makes sense because:

- \(N'\) is not a raw Poisson count over a fixed region;
- it is conditioned on DBSCAN detection;
- it is truncated at the minimum cluster size;
- cluster membership is spatially dependent;
- the distribution is algorithm-induced.

For \(\lambda'/\lambda_0\), the distribution initially looked “Poisson-like” visually, but it is continuous and skewed. Gamma, lognormal, Weibull, beta-prime, and related distributions were tested.

---

## 10. Gamma and lognormal phase

A merged dataset across selected `eps` values was initially fit using several continuous distributions:

- gamma;
- exponential;
- lognormal;
- Weibull.

Exponential was clearly ruled out. Weibull performed poorly. Gamma and lognormal were the best among that set.

### Gamma findings

Gamma fit:

- aligned well with the left tail in some merged views;
- had a generally similar shape;
- but underestimated peak height and skewed peak location.

The fitted peak was around 0.115 while observed peak was around 0.13 in one comparison.

This suggested that gamma captured part of the mechanism but not the full distribution.

### Theoretical gamma hypothesis

Gamma appeared plausible because:

- sums of exponential variables often produce gamma distributions;
- interpoint distances / nearest-neighbor spacing processes in Poisson fields are related to exponential/gamma-like structures;
- clustering and area estimation transform many local spacing relationships into a continuous positive statistic.

However, gamma alone did not fully match the empirical distribution.

---

## 11. Mixtures by \(N'\)

A weighted mixture of gamma distributions by \(N'\) was tested:

- fit a gamma to \(\lambda'/\lambda_0\) for each \(N'\);
- weight each component by frequency of that \(N'\);
- compare with single gamma fit.

This improved AIC/BIC modestly and usually improved KS when comparisons were done correctly on matched samples. The earlier KS discrepancy came from accidentally accumulating multiple `eps` batches into the CDF comparison.

A single-mode fit using only \(N'=10\) showed that the discrete \(N'\) structure matters, but it did not become the final model. Later beta-prime fits suggested the apparent \(N'\)-mode mixture was only part of the story.

---

## 12. Discovery of beta-prime as best fit

When testing lognormal and beta-prime separately per `eps`, beta-prime won most cases. Lognormal occasionally won with `floc=0`, but beta-prime with a fitted or better-chosen `loc` generally dominated.

The beta-prime distribution became the best model for the density ratio \(R=\lambda'/\lambda_0\).

### Key beta-prime observation

Beta-prime is related to a ratio of gamma-distributed variables. If

\[
X \sim \Gamma(a, \theta), \quad Y \sim \Gamma(b, \theta)
\]

independently, then

\[
\frac{X}{Y} \sim \text{BetaPrime}(a,b).
\]

This is theoretically suggestive because the observed statistic is itself a ratio:

\[
R = \frac{N'/S'}{N/S}.
\]

The numerator and denominator are not literally independent gammas, but the local density ratio is shaped by competing stochastic quantities: count accumulation, local area/hull geometry, and global normalization.

### Working hypothesis for beta-prime emergence

The beta-prime form may arise because the cluster density statistic behaves like a ratio between two gamma-like quantities:

1. A “mass/count/intensity” component related to clustered point accumulation.
2. A “space/area/hull” component related to local spatial extent or available support.

The cluster extraction process imposes conditioning: we observe only regions that pass density and connectivity thresholds. That selection may transform the raw Poisson geometry into a ratio-distribution family.

Another plausible interpretation:

- gamma models the aggregate local spacing / area accumulation process;
- beta-prime models the ratio of local “compressed” density to an effective background or local support scale;
- `eps` shifts the effective support and selection threshold, producing systematic parameter drift.

This is not yet a proof, but it is much more theoretically coherent than treating \(R\) as normal or plain Poisson.

---

## 13. Beta-prime parameters and eps-dependence

For regular beta-prime fits across `eps = 1.10–1.40`, parameters showed strong quasi-linear dependence on `eps`.

Example fits:

```text
eps = 1.10:
a=39.4761, b=10.0552, loc=7.4627, scale=3.0064
AIC=203133.87, BIC=203167.73, KS=0.0025, p=0.9833

eps = 1.15:
a=43.4975, b=9.7678, loc=6.8561, scale=2.3920
AIC=390889.47, BIC=390926.08, KS=0.0033, p=0.4506

eps = 1.20:
a=45.3479, b=10.0154, loc=6.1304, scale=2.1798
AIC=542618.60, BIC=542656.65, KS=0.0027, p=0.4427

eps = 1.25:
a=49.7775, b=10.2319, loc=5.4933, scale=1.8896
AIC=524700.22, BIC=524738.27, KS=0.0045, p=0.0329

eps = 1.30:
a=53.6424, b=10.5453, loc=4.9350, scale=1.6873
AIC=507201.41, BIC=507239.46, KS=0.0051, p=0.0113

eps = 1.35:
a=53.8421, b=10.7307, loc=4.4540, scale=1.5939
AIC=491379.13, BIC=491417.18, KS=0.0052, p=0.0084

eps = 1.40:
a=53.1747, b=10.9131, loc=4.0521, scale=1.5369
AIC=476599.95, BIC=476638.00, KS=0.0060, p=0.0016
```

### Regression of parameters vs eps

A regression pass produced approximately:

```text
a:
  slope ~ 50.0568
  intercept ~ -14.1770
  R² ~ 0.9005

b:
  slope ~ 3.5925
  intercept ~ 5.8322
  R² ~ 0.8630

loc:
  slope ~ -11.5939
  intercept ~ 20.1187
  R² ~ 0.9918

scale:
  slope ~ -4.6409
  intercept ~ 7.8420
  R² ~ 0.8993
```

Later fits updated these values, including one eps-conditioned global optimization that produced:

```text
a_slope: 51.2613
a_intercept: -8.2723
b_slope: 11.1153
b_intercept: -3.9662
loc_slope: -21.0301
loc_intercept: 32.0072
scale_slope: 0.0052
scale_intercept: 1.6100
AIC: 14485165.71
BIC: 14485268.40
KS Statistic: 0.4345
```

That global eps-conditioned optimization succeeded but did not solve the desired problem, because it models conditional distributions \(P(R\mid \epsilon)\), not a single eps-independent marginal probability function.

---

## 14. Location parameter (`loc`) issue

The minimum ratio in practice is above 5. Fixing `floc=0` was often inferior. Testing `floc=5` improved fits systematically.

A brute-force search over `floc` from 0 to 10 in 0.1 increments showed:

- `floc=5` beats `floc=0` in all cases tested;
- beta-prime wins most cases;
- lognormal only occasionally wins, usually under less flexible location constraints.

The linear approximation for `loc` helped as an initial guess but had point-to-point errors up to ~1.0, too large for final fitting.

The best fitting should let `loc` be estimated directly, using regression only as an initial guess for faster convergence.

---

## 15. Mixture of beta-prime distributions

A two-component beta-prime mixture was tried. It converged to identical components:

```text
Component 1: a=..., b=..., loc=..., scale=...
Component 2: a=..., b=..., loc=..., scale=...
Mixing alpha: 0.5
```

This suggests there are no strong distinct beta-prime modes in the tested range. The mixture does not justify itself as two separated components.

This weakened the earlier “distinct \(N'\)-mode mixture” hypothesis. Instead, beta-prime itself may already absorb the ratio structure that gamma mixtures were trying to approximate.

---

## 16. Important correction: mixture CDF / KS

A bug appeared when computing KS for mixtures: the mixture PDF was accidentally used as if it were a CDF in one place.

Correct KS for a mixture requires a mixture CDF:

\[
F(x) = \sum_i w_i F_i(x).
\]

A mixture PDF cannot be passed to `kstest` as a CDF.

Similarly, trying to construct `rv_histogram((pdf_values, merged_data))` was invalid because `rv_histogram` expects `(hist_counts, bin_edges)` with lengths \(n\) and \(n+1\). Passing same-length PDF values and raw data points caused a shape mismatch.

For eps-conditioned beta-prime models, a naive global KS over merged data is not straightforward because every data point has its own conditional CDF depending on `eps`.

---

## 17. Why the eps-conditioned model was not enough

The eps-conditioned beta-prime regression model describes:

\[
P(R \mid \epsilon).
\]

But the desired practical output is an approximately epsilon-independent probability function:

\[
P(R \ge r)
\]

for a cluster's observed density ratio, to estimate rare-event probability or a Z-like score.

The merged-data model should answer:

> If the algorithm searches over a reasonable eps range, what is the null probability of observing a cluster with density ratio at least \(r\)?

This is a marginal distribution over eps:

\[
P(R) = \int P(R \mid \epsilon) P(\epsilon)\,d\epsilon.
\]

In discrete experimental form:

\[
P(R) \approx \sum_i w_i P(R \mid \epsilon_i).
\]

This is literally a mixture over eps-conditioned beta-prime distributions.

### Key modeling question

What should the weights \(w_i\) be?

Possible choices:

1. **Uniform over eps grid**  
   Treat each `eps` value equally. This answers: “If eps is selected uniformly from this search range…”

2. **Cluster-count weighted**  
   Weight by number of clusters produced at each eps. This answers: “If I sample a random detected cluster from all eps runs…”

3. **Algorithm-search weighted / multiple-comparison-aware**  
   Weight by the effective number of opportunities at each eps. This is probably the most relevant but hardest.

4. **Max-over-eps distribution**  
   If the real detector scans many eps values and reports the most extreme cluster, then the correct null is not the marginal mixture; it is an extreme-value distribution over the search procedure. This is more conservative and probably closer to the final anomaly detector.

---

## 18. Z-score and rare-event probability

For a non-normal skewed distribution, a standard Gaussian \(Z=(x-\mu)/\sigma\) is not the most meaningful rarity measure.

Better approach:

1. Estimate tail probability:

\[
p = P(R \ge r)
\]

using the survival function of the fitted null distribution.

2. Convert to Gaussian-equivalent Z if desired:

\[
Z_\text{equiv} = \Phi^{-1}(1-p).
\]

This gives an interpretable “sigma equivalent” without assuming \(R\) is normal.

For a beta-prime mixture over eps:

\[
P(R \ge r) = \sum_i w_i \operatorname{SF}_{\text{BetaPrime}_i}(r).
\]

Then:

\[
Z_\text{equiv} = \Phi^{-1}(1 - P(R \ge r)).
\]

This is likely the right path for an eps-independent scoring layer.

---

## 19. Suggested next modeling script

The next useful script should build a marginal eps-mixture probability model using the per-eps beta-prime fits already gathered.

### Inputs

- `regular_fit.csv` with rows:
  - `eps`
  - `iteration`
  - `a`
  - `b`
  - `loc`
  - `scale`
  - `AIC`
  - `BIC`
  - `KS_stat`
  - `KS_pval`

or equivalent per-eps fit files.

- Optional cluster count summary per eps, or infer it from raw data files.

### Outputs

- A serialized eps-mixture model:
  - component eps values;
  - beta-prime parameters per component;
  - weights;
  - chosen weighting scheme;
  - metadata on data range and sample size.

- Functions:
  - `pdf(r)`
  - `cdf(r)`
  - `sf(r)`
  - `p_value_for_ratio(r)`
  - `z_equiv_for_ratio(r)`

- Diagnostic plots:
  - empirical merged histogram vs eps-mixture PDF;
  - empirical CDF vs eps-mixture CDF;
  - survival plot / log survival plot;
  - tail comparison around \(R=40–80\).

### Candidate weighting schemes

Implement and compare:

1. `uniform_eps`
2. `cluster_count`
3. `inverse_cluster_count` maybe as a sensitivity check
4. `manual weights`
5. eventually `max_over_eps` simulation-derived weights or full search-procedure null

### Model comparison

Compare:

1. Single beta-prime fit to merged data.
2. Eps-mixture beta-prime with uniform weights.
3. Eps-mixture beta-prime with cluster-count weights.
4. Empirical survival function directly from merged data.
5. Extreme-value model for max cluster per simulation / per eps, if available later.

### Important

For tail scoring, empirical survival should be used as a sanity check. Parametric fits can extrapolate beyond observed data, but may under- or overestimate rare tails.

---

## 20. Shape and edge effects still unresolved

The current reduced dataset cannot answer some important questions:

### Edge effects

Clusters near the circular boundary may have different geometry from central clusters. Since centers and effective radii were not saved, the current data cannot separate boundary from central cases.

Future data should store:

- cluster center;
- distance from center;
- distance to boundary;
- effective radius;
- whether cluster intersects or approaches boundary.

### Shape effects

Convex hull area can vary strongly with shape. Elongated clusters, compact clusters, triangular clusters, and rare ring-like structures can have different \(S'\) behavior.

Future data should store shape metrics:

- PCA eigenvalues;
- aspect ratio;
- hull perimeter;
- hull compactness;
- area-to-miniball-area ratio;
- maybe number of hull vertices.

### Miniball alternative

A minimum enclosing ball could provide a uniform circular geometry, but it changes the object being measured:

- it includes empty space;
- it may include points not in the DBSCAN cluster;
- it turns the statistic into a different scan statistic;
- it may be closer to a “best circular aperture” method than a DBSCAN-hull method.

Possible future direction: after DBSCAN finds a seed cluster, optimize a circular aperture around its center to maximize \(\lambda'\). This would create a cleaner geometry but a different null model.

---

## 21. Rarefactions

DBSCAN detects high-density clusters, not rarefactions. Earlier attempts to score rarefactions using a Poisson \(Z\)-score produced systematic asymmetry: compressions and rarefactions did not behave like two tails of the same distribution.

Current working hypothesis:

- compressions and rarefactions are separate algorithmic/statistical objects;
- they should be modeled separately;
- OPTICS or grid/KDE methods may be better suited for rarefactions;
- rarefactions likely have different characteristic areas and tails.

---

## 22. OPTICS discussion

OPTICS was considered because it can produce richer “canned food” reachability information and can capture structures across density thresholds, possibly including rarefaction-like regions.

Pros:

- captures density hierarchy;
- avoids fixing one global `eps`;
- might allow post-hoc thresholding;
- could preserve more information for future analyses.

Cons:

- higher computational cost;
- post-processing complexity;
- may be overkill for a uniform Poisson disk;
- DBSCAN already takes long at large simulation scale.

The main open question:

> Can precomputing nearest-neighbor / reachability structures speed up later DBSCAN-like scans across many eps values?

Likely yes in principle: a neighbor graph or distance index can make repeated threshold clustering cheaper. But the practical benefit depends on implementation and memory.

---

## 23. Current best conceptual model

For a fixed eps:

\[
R = \lambda'/\lambda_0 \sim \text{BetaPrime}(a(\epsilon), b(\epsilon), loc(\epsilon), scale(\epsilon))
\]

with approximately linear or quasi-linear parameter dependence on eps.

For a range of eps:

\[
P(R) \approx \sum_i w_i \text{BetaPrime}_i(R)
\]

where \(\text{BetaPrime}_i\) uses parameters fit at \(\epsilon_i\), and weights encode what “merged over eps” means.

For anomaly scoring:

\[
p(r) = P(R \ge r) = \sum_i w_i \operatorname{SF}_i(r)
\]

and optionally:

\[
Z_\text{equiv}(r) = \Phi^{-1}(1-p(r)).
\]

This is the most promising direction for producing an eps-independent rare-event score.

---

## 24. Key cautions

1. **AIC/BIC cannot be compared across different datasets without care.**  
   Method C using only \(N'=10\) has fewer observations than full-data methods; absolute AIC/BIC are not directly comparable to full-data fits.

2. **KS p-values go to zero for huge samples.**  
   Focus on KS statistic magnitude and visual diagnostics, not p-values alone.

3. **KS for conditional models is subtle.**  
   If parameters depend on eps, a naive global KS needs a properly defined marginal CDF.

4. **Mixture PDF is not mixture CDF.**  
   Always use \(\sum w_i F_i(x)\) for CDF-based tests.

5. **The reduced data format blocks edge/shape analysis.**  
   Future runs should save richer geometry.

6. **An eps-independent score must define eps weighting.**  
   Uniform eps, cluster-count weighting, and max-over-search null answer different questions.

7. **The final detector likely needs search-procedure calibration.**  
   If real detection scans eps and reports the best cluster, the null must be calibrated on the same full procedure.

---

## 25. Practical continuation plan

### Immediate next step

Build an eps-mixture beta-prime model from existing per-eps fits.

Implement:

- load `regular_fit.csv`;
- group by eps;
- average parameters or use all subsample fit rows as components;
- choose weights:
  - uniform per eps;
  - cluster-count weighted;
  - equal per fit row;
- compute mixture PDF/CDF/SF;
- expose `p_value_for_ratio(r)` and `z_equiv_for_ratio(r)`;
- compare against empirical merged survival.

### Next simulation run

Update simulator output format to include:

- cluster center \(x,y\);
- distance to circular boundary;
- effective radius;
- convex hull perimeter;
- PCA eigenvalues/aspect ratio;
- miniball radius if feasible;
- maybe max-over-iteration cluster statistics.

### Next model refinement

Compare:

1. central clusters only vs all clusters;
2. compact clusters vs elongated clusters;
3. convex hull area vs miniball area;
4. fixed eps vs eps mixture;
5. marginal cluster distribution vs max-over-search distribution.

### Long-term target

Define a calibrated null scoring function:

```text
score(cluster, search_config) -> {
    density_ratio,
    p_value,
    z_equivalent,
    model_version,
    caveats
}
```

where `search_config` includes geometry, eps range, min samples, min cluster size, area definition, and whether the score is for a single cluster, random detected cluster, or best-of-search cluster.

---

## 26. Compact summary

The project began as a Poisson cluster-detection problem, but DBSCAN selection bias made direct Poisson \(Z\)-scores invalid. Large-scale null simulations showed that apparent cluster density ratios under CSR form stable, skewed distributions. Gamma was a good early approximation, but beta-prime became the best model, likely because the statistic is a ratio of gamma-like geometric quantities induced by Poisson spacing and cluster area estimation.

The distribution is not truly eps-independent: beta-prime parameters drift systematically with `eps`. However, the far right tail remains practically bounded around observed ratios of 60–70 in the simulation regime. The correct eps-independent probability function should likely be an eps-mixture of beta-prime survival functions, with explicit weighting depending on the intended interpretation. Tail probability should be converted to Gaussian-equivalent \(Z\) only after computing the survival probability; normality should not be assumed.

Future simulation must save richer geometry to resolve edge and shape effects. The current best path is to build a mixture-over-eps beta-prime null model from existing per-eps fits, validate it against empirical survival, and later calibrate a full max-over-search null if the detector scans eps and reports the most extreme cluster.

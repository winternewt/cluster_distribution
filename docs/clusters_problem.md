# Detecting Density Clusters Against Poisson Noise — Problem Statement & Approach

**Provenance:** synthesised from a discussion between Newton and Claude (Opus 4.8), May 2026. The reasoning in §2 was produced *without sight of the simulation data*; it is design-time analysis, not a fit to results. Empirical specifics (the fitted beta-prime parameters, tail-index mismatch, and parameter shifts under edge conditions) are held by Newton and will be supplied separately to the instance working the data.

---

## 0. Initial Formulation (original framing, lightly cleaned)

We search for arbitrary structure in seemingly random point data — conceptually, night-vision against a noisy sensor.

- A bounded area $S$ contains $N$ points.
- **Null hypothesis $H_0$:** $S$ is filled with pure noise, a homogeneous Poisson point process of intensity $\lambda$ (average point density).
- **Alternative $H_1$:** there *may* exist a sub-region $S' \subset S$ carrying additional points. For the hard case, the splat is itself Poisson — so it is **not** geometrically obvious against the surrounding noise — but has a local density differing from the ambient $\lambda$.

The task splits into two parts:

1. **Localisation** — select candidate sub-areas that contain clusters.
2. **Discrimination** — distinguish a genuine weak signal from an ordinary noise fluctuation.

A coarse solution exists for both; the unsolved piece is discrimination for the *refined* localiser.

### Coarse approach (works, proof-of-concept)

Partition $S$ into a fixed grid of $K$ disjoint adjacent cells, each of area $S/K$. Count $n_i$ per cell. For each cell compute $q_i = \Pr[\,\ge n_i\,]$ under $\mathrm{Poisson}(\lambda')$ with $\lambda' \to \lambda$ (binomial occupancy, Poisson-approximated). Combine across cells via Bernoulli trials to get the overall probability $P_i$ of finding $K_1$ cells holding $\ge n_i$ points. Apply a Poisson Z-score at 95% confidence (a Poisson analogue of the $3\sigma$ rule) to flag cells inconsistent with the random distribution. Gives positions **and** calibrated probabilities, but coarse.

### Refined approach (localisation fine, discrimination open)

Build a density map and run **DBSCAN** to carve out a sub-area $S'$. Because $S'$ is now selected **non-randomly from the density map**, the assumption $\lambda' \to \lambda$ no longer holds and the coarse Z-score is invalid. The probability of finding *some* denser sub-region depends on $S$ and $N$. With an effectively infinite family of dissections, the fixed-grid binomial combination does not apply.

The localiser does impose a boundary via triangulation:

$$
S' \cup S'' = S,\quad S' \cap S'' = \varnothing,\quad S',S'' > 0,\quad N',N'' \ge 3,\quad N = N' + N'',
$$
$$
\lambda' = N'/S',\qquad \lambda'' = N''/S''.
$$

This removes the degenerate $\lambda' \in \{0,\infty\}$ cases but nothing more.

**Open question:** can the probability of the deviation of the *apparent* $\lambda'$ of this biased sample from the initial $\lambda$ be characterised, and how?

---

## 1. Rectified Postulation

The refined problem is not a harder version of the coarse one — it is a **different statistical object**. This is **post-selection inference** / the **look-elsewhere effect**.

**Why the coarse method was legitimate.** Its grid cells are *fixed in advance and disjoint*. Each cell is therefore a pre-registered, independent test; the multiplicity is exactly $K$ trials and the Bernoulli combination corrects for it cleanly.

**Why it cannot be reused.** DBSCAN chooses $S'$ *because it looks dense*, and even draws $S'$'s boundary to hug the points. Consequences:

- The measured $\hat\lambda' = N'/S'$ is **not** a draw from $\mathrm{Poisson}(\lambda S')/S'$. It is (near) the **maximum** of $\hat\lambda'$ over the entire family of windows the algorithm could have returned.
- $S'$ itself is a random, data-adaptive quantity, so $p_0 = |S'|/|S|$ is not a fixed null fraction.
- Any per-window calibration — the Poisson Z-score **or** a Bayesian Beta posterior on the inside-fraction — is overconfident by the *same* selection mechanism. The error is identical in both paradigms.

**Correct statistical statement.** Fix a size-aware test statistic $T$ (see §2). The quantity to characterise is the null distribution of

$$
\Lambda \;=\; \max_{Z \in \mathcal{W}} \, T(Z)
$$

under complete spatial randomness (CSR), where $\mathcal{W}$ is the window family *actually reachable by the DBSCAN selector* (not all subsets of $S$). The "deviation of apparent $\lambda'$" is the upper tail of $\Lambda$. Its mean is biased high and its spread is governed by the **effective number of windows searched** (the trials factor), not by $S'$.

**Recommended conditioning.** Condition on the total count $N$. This removes the nuisance dependence on the overall rate and makes the inside/outside split exactly $N' \sim \mathrm{Binomial}(N,\,|Z|/|S|)$. (This is the conditional formulation Kulldorff uses.)

---

## 2. Conclusions & Approaches (design-time, no data seen)

### 2.1 Choice of statistic — not raw $\lambda'$

Raw $\hat\lambda'$ is a poor thing to threshold: it ignores window size (3 points in a tiny triangle → huge $\hat\lambda'$, no meaning), which is exactly why the $N' \ge 3$ guard feels necessary but insufficient. Use the **Kulldorff spatial-scan likelihood ratio**, the GLR for this two-density model. With $n$ points in candidate zone $Z$, expected $\mu = N\,|Z|/|S|$:

$$
\mathrm{LR}(Z) = \left(\frac{n}{\mu}\right)^{n}\left(\frac{N-n}{N-\mu}\right)^{N-n}\;\mathbf{1}\!\left[\tfrac{n}{\mu} > 1\right],
\qquad
\Lambda = \max_{Z} \mathrm{LR}(Z).
$$

The coarse grid method is the special case where $Z$ ranges over single fixed cells — this is the through-line connecting the two approaches.

### 2.2 No simple closed form — and why

Optimal scan over arbitrary **connected** regions is NP-hard. That is precisely why Kulldorff restricts the window family to circles/ellipses: a deliberate tractability sacrifice. DBSCAN is a *different* heuristic restriction of the same intractable search. Any null distribution obtained for it is conditional on that heuristic and is only valid if the simulation replays the identical heuristic.

### 2.3 Three routes to the null

1. **Monte Carlo of the full pipeline (primary recommendation).** Generate $N$ points uniformly on $S$ (conditioning on $N$), run the *identical* DBSCAN + LR pipeline, record $\Lambda^{(b)}$, repeat $10^3$–$10^4\times$. The p-value is the rank of observed $\Lambda$ among $\{\Lambda^{(b)}\}$. Valid regardless of selector complexity *because the selection is replayed*. This is what **SaTScan** implements; it is the field standard in epidemiology and astronomical source detection.

2. **Analytic trials factor (only for structured window families).** If $\mathcal{W}$ is constrained (fixed-radius circles on points; dyadic multiresolution), use the **Gross–Vitells** upcrossing / **Euler-characteristic** estimate, equivalently the **Poisson clumping heuristic**: $\Pr[\Lambda > u] \approx \mathbb{E}[\#\text{clumps above }u] = N_{\mathrm{eff}}\,p_{\mathrm{local}}(u)$ for small $p_{\mathrm{local}}$, with $N_{\mathrm{eff}}$ the effective resolution-element count ($\sim\!\sqrt{2\ln N_{\mathrm{eff}}}$ in Z-units). **No clean closed form exists for DBSCAN's arbitrary blobs** — this route requires taming $\mathcal{W}$ first.

3. **Bayesian generative mixture (most principled, heaviest).** Model the field as homogeneous Poisson + a cluster component. Conjugacy: **Gamma** is the conjugate density for the rate $\lambda$; **Beta** enters via the inside-fraction $p = N'/N \sim \mathrm{Binomial}(N, |S'|/|S|)$, giving the Beta-Binomial marginal likelihood
$$
\Pr(n' \mid a,b) = \binom{N}{n'}\frac{B(a+n',\,b+N-n')}{B(a,b)}
$$
(the Beta function appears exactly here). **Critical caveat:** applied to the DBSCAN window naively, the Bayes factor is computed conditional on a window chosen to maximise that very evidence — same selection bias. To be honest, the cluster's **location/shape/count must be latent variables integrated over** (reversible-jump MCMC or a Dirichlet-process mixture); the prior over where a cluster *could* be **is** the multiplicity correction, paid inside the model rather than bolted on.

**Unifying moral:** selection must be either *replayed* (Monte Carlo) or *integrated over* (latent configuration). The per-window calculation is correct only for pre-registered windows.

### 2.4 The beta-prime result — mechanistic explanation

> Stated honestly: beta-prime was **recognised post-hoc** as consistent with Newton's empirical fit, **not** predicted a priori from the formulation. The following is the rationalisation, which holds up.

With Gamma-conjugate rate posteriors $\lambda' \sim \mathrm{Gamma}(a', S')$ and $\lambda'' \sim \mathrm{Gamma}(a'', S'')$, the **ratio** $\lambda'/\lambda''$ is a **scaled beta-prime**: the ratio of two independent Gammas is $\mathrm{beta\text{-}prime}(a',a'')$, and the scale factor carries the $S''/S'$ geometry. This scale/geometry term is the plausible source of the empirically observed **"modified"** part.

$$
f_{\mathrm{beta\text{-}prime}}(x;\alpha,\beta) = \frac{x^{\alpha-1}(1+x)^{-(\alpha+\beta)}}{B(\alpha,\beta)},\quad x>0.
$$

Diagnostic implication worth keeping: **beta-prime emerging at all implies the operative statistic is a ratio/contrast, not a bare rate.** A bare *selected* $\hat\lambda'$ would trend toward Fréchet/GEV (max-over-windows), not beta-prime. Beta-prime is also the same family as the F-distribution and the large-count limit of the Poisson LR — so the rate-ratio, variance-ratio, and LR framings are mutually consistent.

### 2.5 Tail behaviour & practical leverage

- The polynomial right tail of beta-prime is the **selection signature**: a single fixed window gives the clean ratio-of-Gammas beta-prime; max-over-windows *fattens the tail*. The deviation of the fitted parameters from the vanilla analytic beta-prime is therefore the **look-elsewhere penalty made visible** — the trials factor in another guise.
- Beta-prime lies in the **Fréchet domain of attraction**. For deep-tail p-values, do **not** brute-force $10^6$ sims. Fit a **generalised Pareto (peaks-over-threshold)** to exceedances and extrapolate; its tail index should agree with the beta-prime's and is far more stable for the far tail.
- If a single parametric form must cover body and tail flexibility, **GB2** (generalised beta of the second kind) is the superfamily containing beta-prime, F, and log-logistic — usually overkill.

### 2.6 Notes for the data-working instance

Newton reports two empirical effects to be reconciled against the above (details to be supplied with the data):

- **Tail mismatch** — the empirical tail departs from the vanilla beta-prime tail. Interpret via §2.5: quantify the departure as an effective trials factor; cross-check the GPD tail index.
- **Parameter shift under edge conditions** (minimal area, the $N' \ge 3$ / triangulation boundary). Likely the geometric scale term $\propto S''/S'$ degenerating near the constraint boundary; check whether the shift is in the **scale** (geometry/edge) or the **shape/tail index** (selection strength). If the tail index itself moves, that is selection strength talking directly, not a geometry artefact.

---

### One-line summary

Localise with DBSCAN, score with the Kulldorff LR, and obtain the null by Monte-Carlo-replaying the *entire* pipeline under CSR (conditioned on $N$); the empirical beta-prime is the ratio-of-Gammas signature with a selection-fattened, Fréchet-domain tail best extrapolated via peaks-over-threshold.

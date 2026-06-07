# Live demo — Detecting clusters in Poisson noise

Static, dependency-free (vanilla JS + canvas). Two files: `index.html`, `master.js`.

Runs CSR simulations in the browser and shows, in real time:
- the noise field with DBSCAN-detected clusters outlined;
- the `R·ε^α(ε)` density-ratio histogram converging to the predicted master curve — a
  shifted inverse-gamma with integer shape 10 = `min_samples`,
  `SF(R̃) = P(10, 157.70/(R̃ − 7.51))`, computed exactly with two Poisson sums, no gamma
  library (the same curve at every `ε` — the collapse of Finding #1; `α(ε)=2.03+0.26·ln ε`
  corrects the geometric 2 for the min_samples occupancy effect);
- each detected cluster scored through the null → its z-equivalent piling into a standard-normal
  bell (the probability-integral transform: a heavy-tailed statistic, calibrated, becomes σ).

Move the `eps` slider: raw R shifts wildly, but `R·ε^α(ε)` keeps fitting the same master.

## Run locally
Open `index.html` directly, or:
```bash
cd webapp && python3 -m http.server 8000   # then visit http://localhost:8000
```

## Host on GitHub Pages
These are plain static files, so any of:
- **/docs route (simplest):** copy `index.html` + `master.js` into a `docs/` folder on your
  default branch, then repo *Settings → Pages → Deploy from branch → main → /docs*.
- **gh-pages branch:** `git subtree push --prefix webapp origin gh-pages` (or push the two files
  to the root of a `gh-pages` branch), then *Settings → Pages → gh-pages → /(root)*.
- **Pages Action:** add an upload-pages-artifact workflow pointing at `webapp/`.

Validated: JS `masterSF/masterPDF` match `scipy.stats.invgamma` to <1e-14 relative; on real
simdata (1.8M clusters, eps 1.10–1.40) the scored z-distribution has |median_z| ≤ 0.017,
half-band asymmetries ≤ 0.014, KS(z) ≤ 0.009 — the earlier log-logistic master sat ~0.06σ
left of centre (root cause + fix: `docs/RCA.md`).

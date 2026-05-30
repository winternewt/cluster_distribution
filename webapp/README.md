# Live demo — Detecting clusters in Poisson noise

Static, dependency-free (vanilla JS + canvas). Two files: `index.html`, `master.js`.

Runs CSR simulations in the browser and shows, in real time:
- the noise field with DBSCAN-detected clusters outlined;
- the `R·eps²` density-ratio histogram converging to the predicted inverse-gamma master curve
  (the same curve at every `eps` — the collapse of Finding #1);
- each detected cluster scored through the null → its z-equivalent piling into a standard-normal
  bell (the probability-integral transform: a heavy-tailed statistic, calibrated, becomes σ).

Move the `eps` slider: raw R shifts wildly, but `R·eps²` keeps fitting the same master.

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

Validated (Node): clusters/field ≈ 1.2 at eps=1.4, `R·eps²` mean ≈ 24 / median ≈ 23, and the
scored z-distribution has mean ≈ 0, sd ≈ 1 (N(0,1)).

"""cluster_detector — calibrated CSR anomaly detection for 2D point fields.

Detects local over-densities ("clusters of signal in noise") and assigns each a
look-elsewhere-corrected p-value and Gaussian-equivalent z, using two complementary arms:

  * LR arm  — DBSCAN clusters scored by the Kulldorff scan likelihood ratio (tight clumps).
  * KDE arm — kernel-smoothed intensity peaks at one or more bandwidths (extended over-densities).

Each statistic is calibrated against the distribution of its per-field MAXIMUM under a null
model (default: complete spatial randomness on a disk), so the multiple-comparison /
look-elsewhere correction is built into the p-value. The KDE arm uses a HYBRID calibration:
random-field-theory (analytic) where the smoothed field is ~Gaussian (lambda0*pi*h^2 >~ 10),
Monte Carlo otherwise.

A third, cheaper facility is the PER-CLUSTER density-ratio score (master_sf /
score_clusters): each DBSCAN cluster's R = (N'/S')/lambda0 is mapped to the collapse
variable Rt = R*eps^alpha(eps) and scored against the analytic master null — a shifted
inverse-gamma with integer shape 10 (= min_samples), SF(Rt) = P(10, scale/(Rt-loc)).
Calibrated on simdata eps 1.10-1.40: |median_z| <= 0.017, KS(z) <= 0.009 (see
docs/RCA.md §7). NOTE: per-cluster p, NOT look-elsewhere corrected — "is this cluster
unusual for a CSR cluster", not "does this field contain signal". For detection use
the LR/KDE arms.

Provenance: built from the analysis in docs/analytic_findings.md (Findings #1-#9). The CSR
density-ratio null is a heavy-tailed power law (tail index ~7); raw density ratio is a poor
statistic for extended signal (use LR or KDE). See that doc for the full derivation.

Quick start
-----------
    import numpy as np
    from modules.cluster_detector import Detector

    det = Detector(radius=100.0, eps=1.2, n_background=10000)
    det.calibrate(n_mc=2000)          # MC the null once (cached if cache_path given)
    pts = np.load("my_points.npy")    # (M, 2) coordinates inside the disk
    for d in det.score(pts):
        print(d)                      # Detection(method=..., z=..., p=..., x=..., y=...)

For a non-CSR background, pass your own field sampler to calibrate(null_generator=...).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import ConvexHull
from scipy import stats
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------------------------
# eps-independent per-cluster master null (matches webapp/master.js exactly).
# alpha(eps) = c0 + c1*ln(eps) collapses the eps-dependence of R = (N'/S')/lambda0;
# Rt = R*eps^alpha is scored by a shifted inverse-gamma, integer shape 10:
#     SF(Rt) = P(10, y),  y = ig_scale/(Rt - ig_loc),  P = lower regularized gamma.
# Fit: pooled MLE over simdata eps 1.10-1.40, shape frozen at 10 (= min_samples; the
# a->inf limit of the legacy Beta-Prime(a~46, b~10) fits). The location shift is
# essential — zero-loc families mis-centre z by ~0.07 sigma (docs/RCA.md §7).
MASTER = dict(c0=2.031525, c1=0.258273, ig_shape=10, ig_loc=7.5091, ig_scale=157.7035)


def alpha_eps(eps: float) -> float:
    """Running collapse exponent alpha(eps) = c0 + c1*ln(eps)."""
    return MASTER["c0"] + MASTER["c1"] * np.log(eps)


def master_sf(rt):
    """Survival P(Rt' >= rt) of the collapse variable Rt = R*eps^alpha(eps) under CSR.

    Valid in the body and moderate tail; data past the min_area censoring cap
    (Rt ~ 62.83*eps^alpha, z ~ 4) is unvalidated.
    """
    rt = np.atleast_1d(np.asarray(rt, float))
    return stats.invgamma.sf(rt, MASTER["ig_shape"], MASTER["ig_loc"], MASTER["ig_scale"])


def master_z(rt):
    """Gaussian-equivalent z = Phi^-1(1 - SF_master(rt)); ~N(0,1) over CSR clusters."""
    return stats.norm.isf(np.clip(master_sf(rt), 1e-300, 1 - 1e-12))


@dataclass
class Detection:
    method: str          # "DBSCAN+LR" | "KDE(h=..)"
    statistic: float     # 2lnLR or standardized Z peak
    x: float
    y: float
    p_value: float       # per-field (look-elsewhere corrected)
    z: float             # Gaussian-equivalent, Phi^-1(1-p)
    scale: float         # eps (LR) or bandwidth h (KDE)
    n_points: Optional[int] = None   # cluster size (LR arm only)

    def __repr__(self):
        npart = f", N'={self.n_points}" if self.n_points is not None else ""
        return (f"Detection({self.method}, z={self.z:.2f}, p={self.p_value:.2e}, "
                f"loc=({self.x:.1f},{self.y:.1f}){npart})")


class Detector:
    """Calibrated two-arm (LR + KDE) over-density detector for a disk-domain point field.

    Parameters
    ----------
    radius : float           domain radius (disk centred at origin).
    eps : float              DBSCAN neighbourhood radius; also the fine KDE bandwidth default.
    n_background : int       expected number of background points (sets the null density).
    min_samples, min_cluster_size : int   DBSCAN thresholds (default 10).
    bandwidths : tuple[float] KDE bandwidths to scan; default (eps, 4.0).
    grid_delta : float       KDE grid cell size (<= eps/2 recommended).
    """

    def __init__(self, radius: float, eps: float, n_background: int,
                 min_samples: int = 10, min_cluster_size: int = 10,
                 bandwidths: Optional[tuple] = None, grid_delta: Optional[float] = None):
        self.radius = float(radius)
        self.eps = float(eps)
        self.N = int(n_background)
        self.lam0 = self.N / (np.pi * self.radius ** 2)
        self.min_samples = int(min_samples)
        self.min_cluster = int(min_cluster_size)
        self.bandwidths = tuple(bandwidths) if bandwidths else (eps, 4.0)
        self.delta = float(grid_delta) if grid_delta else min(eps / 2.0, 0.5)
        self.gb = int(2 * self.radius / self.delta)
        self._null = {}          # statistic-key -> sorted null array of per-field maxima
        self._sf = {}            # statistic-key -> survival function callable
        self._rft = {}           # bandwidth -> True if calibrated analytically

    # ---------- null model ----------
    def _csr(self, rng) -> np.ndarray:
        r = self.radius * np.sqrt(rng.uniform(size=self.N))
        t = 2 * np.pi * rng.uniform(size=self.N)
        return np.column_stack((r * np.cos(t), r * np.sin(t)))

    # ---------- statistics ----------
    def _twolnLR(self, n, S):
        mu = self.lam0 * S
        if not (n > mu and S > 0):
            return 0.0
        return 2 * (n * np.log(n / mu) + (self.N - n) * np.log((self.N - n) / (self.N - mu)))

    def _lr_clusters(self, pts):
        lab = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pts).labels_
        out = []
        for L in set(lab):
            if L == -1:
                continue
            cp = pts[lab == L]
            if len(cp) < self.min_cluster:
                continue
            try:
                S = ConvexHull(cp).volume
            except Exception:
                continue
            if S <= 0:
                continue
            out.append((self._twolnLR(len(cp), S), len(cp), S, cp[:, 0].mean(), cp[:, 1].mean()))
        return out

    def _kde(self, pts, h):
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=self.gb,
                                 range=[[-self.radius, self.radius], [-self.radius, self.radius]])
        s = gaussian_filter(H, sigma=h / self.delta, mode="constant")
        ax = (np.arange(self.gb) + 0.5) * self.delta - self.radius
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        interior = (X ** 2 + Y ** 2) < (self.radius - 4 * h) ** 2
        si = s[interior]
        return s, X, Y, interior, si.mean(), si.std()

    def _kde_zmax(self, pts, h):
        s, X, Y, interior, mu, sd = self._kde(pts, h)
        return float(((s[interior] - mu) / sd).max())

    def _kde_peaks(self, pts, h, zthr):
        s, X, Y, interior, mu, sd = self._kde(pts, h)
        Z = (s - mu) / sd
        loc = (maximum_filter(s, size=max(1, int(2 * h / self.delta))) == s) & interior & (Z > zthr)
        return sorted(((float(Z[i, j]), float(X[i, j]), float(Y[i, j]))
                       for i, j in zip(*np.where(loc))), reverse=True)

    def _rft_sf(self, u, h):
        fwhm = 2.3548 * h
        r2 = (np.pi * (self.radius - 4 * h) ** 2) / fwhm ** 2
        ec = r2 * (4 * np.log(2)) / (2 * np.pi) ** 1.5 * u * np.exp(-u ** 2 / 2) + stats.norm.sf(u)
        return np.minimum(ec, 1.0)

    @staticmethod
    def _make_sf(sorted_null):
        s = np.asarray(sorted_null, float); n = len(s); floor = 1.0 / n
        u0 = np.quantile(s, 0.9); m = s > u0
        if m.sum() >= 5:
            sf_at = 1 - (np.searchsorted(s, s[m]) / n)
            c, lnA = np.polyfit(s[m], np.log(np.clip(sf_at, 1e-12, 1)), 1)
        else:
            c, lnA = -1.0, 0.0
        def sf(u):
            u = np.atleast_1d(u).astype(float); out = np.empty_like(u)
            for k, x in enumerate(u):
                pe = 1 - np.searchsorted(s, x, side="right") / n
                out[k] = pe if pe >= floor else np.exp(lnA + c * x)
            return out
        return sf

    # ---------- calibration ----------
    def calibrate(self, n_mc: int = 2000, null_generator: Optional[Callable] = None,
                  cache_path: Optional[str] = None, seed: int = 0, rft_threshold: float = 10.0):
        """Estimate per-field null distributions of each arm's maximum statistic.

        null_generator : callable(rng) -> (M,2) points. Default = CSR on the disk. Pass your
                         own to calibrate against a custom background model.
        cache_path     : if given and exists, load; else compute and save (.npz).
        rft_threshold  : KDE bandwidths with lambda0*pi*h^2 >= this are calibrated by RFT
                         (analytic) instead of MC.
        """
        if cache_path and os.path.exists(cache_path):
            self._load_arrays(cache_path); return self
        gen = null_generator or self._csr
        rng = default_rng(seed)
        maxlr = np.empty(n_mc)
        zmax = {h: np.empty(n_mc) for h in self.bandwidths}
        for i in range(n_mc):
            p = gen(rng)
            cl = self._lr_clusters(p)
            maxlr[i] = max((c[0] for c in cl), default=0.0)
            for h in self.bandwidths:
                zmax[h][i] = self._kde_zmax(p, h)
        self._null["lr"] = np.sort(maxlr); self._sf["lr"] = self._make_sf(self._null["lr"])
        for h in self.bandwidths:
            self._null[f"kde{h}"] = np.sort(zmax[h])
            if self.lam0 * np.pi * h ** 2 >= rft_threshold:
                self._rft[h] = True
                self._sf[f"kde{h}"] = (lambda u, hh=h: self._rft_sf(np.atleast_1d(u), hh))
            else:
                self._rft[h] = False
                self._sf[f"kde{h}"] = self._make_sf(self._null[f"kde{h}"])
        if cache_path:
            self._save_arrays(cache_path)
        return self

    # ---------- scoring ----------
    def _z(self, p):
        return float(stats.norm.isf(np.clip(p, 1e-300, 1 - 1e-12)))

    def score(self, points: np.ndarray, zthr: float = 3.0) -> List[Detection]:
        """Return detections (z >= zthr) for an observed field, ranked by significance."""
        if "lr" not in self._sf:
            raise RuntimeError("call calibrate() (or load()) before score().")
        pts = np.asarray(points, float)
        det: List[Detection] = []
        cl = self._lr_clusters(pts)
        if cl:
            stat, npts, _, cx, cy = max(cl, key=lambda c: c[0])
            p = float(self._sf["lr"](stat)[0])
            det.append(Detection("DBSCAN+LR", round(stat, 2), round(cx, 2), round(cy, 2),
                                 p, round(self._z(p), 2), self.eps, npts))
        for h in self.bandwidths:
            pk = self._kde_peaks(pts, h, zthr=zthr - 1)
            if pk:
                z0, cx, cy = pk[0]
                p = float(self._sf[f"kde{h}"](z0)[0])
                tag = f"KDE(h={h:g},{'RFT' if self._rft.get(h) else 'MC'})"
                det.append(Detection(tag, round(z0, 2), round(cx, 2), round(cy, 2),
                                     p, round(self._z(p), 2), h))
        return [d for d in sorted(det, key=lambda d: -d.z) if d.z >= zthr]

    def score_clusters(self, points: np.ndarray) -> List[Detection]:
        """Per-cluster density-ratio scores against the analytic CSR master (no MC needed).

        Each DBSCAN cluster's R = (N'/S')/lambda0 is collapsed to Rt = R*eps^alpha(eps)
        and scored by the shifted inverse-gamma master (module-level MASTER). The z's are
        ~N(0,1) over CSR clusters. NOT look-elsewhere corrected: this answers "how unusual
        is this cluster among CSR clusters", not "does this field contain signal" — for
        the latter use score() (per-field max statistics).
        """
        pts = np.asarray(points, float)
        a = alpha_eps(self.eps)
        det: List[Detection] = []
        for _, npts, S, cx, cy in self._lr_clusters(pts):
            rt = (npts / S) / self.lam0 * self.eps ** a
            p = float(master_sf(rt)[0])
            det.append(Detection("DBSCAN+Rmaster", round(rt, 2), round(cx, 2), round(cy, 2),
                                 p, round(float(master_z(rt)[0]), 2), self.eps, npts))
        return sorted(det, key=lambda d: -d.z)

    # ---------- persistence ----------
    def _save_arrays(self, path):
        meta = dict(radius=self.radius, eps=self.eps, N=self.N,
                    min_samples=self.min_samples, min_cluster=self.min_cluster,
                    bandwidths=np.array(self.bandwidths), delta=self.delta)
        np.savez(path, **{f"null_{k}": v for k, v in self._null.items()}, **meta)

    def _load_arrays(self, path):
        d = np.load(path, allow_pickle=False)
        for k in d.files:
            if k.startswith("null_"):
                key = k[5:]; self._null[key] = d[k]
        for key, arr in self._null.items():
            if key == "lr":
                self._sf["lr"] = self._make_sf(arr)
            else:
                h = float(key[3:])
                if self.lam0 * np.pi * h ** 2 >= 10.0:
                    self._rft[h] = True
                    self._sf[key] = (lambda u, hh=h: self._rft_sf(np.atleast_1d(u), hh))
                else:
                    self._rft[h] = False
                    self._sf[key] = self._make_sf(arr)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        det = cls(radius=float(d["radius"]), eps=float(d["eps"]), n_background=int(d["N"]),
                  min_samples=int(d["min_samples"]), min_cluster_size=int(d["min_cluster"]),
                  bandwidths=tuple(float(x) for x in d["bandwidths"]), grid_delta=float(d["delta"]))
        det._load_arrays(path)
        return det

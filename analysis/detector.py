#!/usr/bin/env python3
"""Unified CSR anomaly detector: score(points) -> {detections, p, z}.

Two complementary arms, each calibrated against the per-field CSR null (look-elsewhere
built in -> p = "how often does pure CSR produce something this extreme ANYWHERE in a field"):
  - LR arm  : DBSCAN clusters scored by the Kulldorff 2lnLR (catches tight clumps);
              null = MC of per-field max-2lnLR + alpha~7 heavy-tail extrapolation.
  - KDE arm : kernel-smoothed intensity peak Z_max at scale h (catches extended overdensity);
              HYBRID calibration -> RFT analytic when the field is Gaussian (lambda0*pi*h^2 >~10),
              else MC. (Finding #8: RFT valid coarse, fails fine.)

Caches null tables to analysis/null_cache.npz so re-runs are instant.
"""
import os, numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import ConvexHull
from scipy import stats
from sklearn.cluster import DBSCAN

HERE=os.path.dirname(__file__); N,RAD=10000,100; LAM0=N/(np.pi*RAD**2)
DELTA=0.5; GB=int(2*RAD/DELTA); CACHE=os.path.join(HERE,"null_cache.npz")

def csr(rng,m=N):
    r=RAD*np.sqrt(rng.uniform(size=m)); t=2*np.pi*rng.uniform(size=m)
    return np.column_stack((r*np.cos(t),r*np.sin(t)))

# ---------------- statistics ----------------
def twolnLR(n,S):
    mu=LAM0*S
    return 2*(n*np.log(n/mu)+(N-n)*np.log((N-n)/(N-mu))) if (n>mu and S>0) else 0.0

def lr_clusters(pts,eps):
    """return list of (2lnLR, N', S', cx, cy) for DBSCAN clusters."""
    lab=DBSCAN(eps=eps,min_samples=10).fit(pts).labels_
    out=[]
    for L in set(lab):
        if L==-1: continue
        cp=pts[lab==L]
        if len(cp)<10: continue
        try: S=ConvexHull(cp).volume
        except Exception: continue
        if S<=0: continue
        out.append((twolnLR(len(cp),S),len(cp),S,cp[:,0].mean(),cp[:,1].mean()))
    return out

def kde_field(pts,h):
    H,_,_=np.histogram2d(pts[:,0],pts[:,1],bins=GB,range=[[-RAD,RAD],[-RAD,RAD]])
    s=gaussian_filter(H,sigma=h/DELTA,mode="constant")
    ax=(np.arange(GB)+0.5)*DELTA-RAD; X,Y=np.meshgrid(ax,ax,indexing="ij")
    interior=(X**2+Y**2)<(RAD-4*h)**2
    si=s[interior]; mu,sd=si.mean(),si.std()
    return s,(X,Y,interior,mu,sd)

def kde_peaks(pts,h,zthr=3.0):
    s,(X,Y,interior,mu,sd)=kde_field(pts,h)
    Z=(s-mu)/sd
    loc=(maximum_filter(s,size=int(2*h/DELTA))==s)&interior&(Z>zthr)
    pk=[(Z[i,j],X[i,j],Y[i,j]) for i,j in zip(*np.where(loc))]
    return sorted(pk,reverse=True), float(Z[interior].max())

# ---------------- RFT analytic KDE null ----------------
def rft_sf(u,h):
    FWHM=2.3548*h; R2=(np.pi*(RAD-4*h)**2)/FWHM**2
    ec=R2*(4*np.log(2))/(2*np.pi)**1.5*u*np.exp(-u**2/2)+stats.norm.sf(u)
    return np.minimum(ec,1.0)   # EC is an upper bound on the exceedance prob; clip to [0,1]

# ---------------- calibration (cached) ----------------
def calibrate(eps=1.2, h_fine=1.2, h_coarse=4.0, n_mc=2000, rng=None):
    if os.path.exists(CACHE):
        d=np.load(CACHE)
        return {k:d[k] for k in d.files}
    rng=rng or default_rng(123)
    maxlr=np.empty(n_mc); zf=np.empty(n_mc); zc=np.empty(n_mc)
    for i in range(n_mc):
        p=csr(rng)
        cl=lr_clusters(p,eps); maxlr[i]=max([c[0] for c in cl],default=0.0)
        zf[i]=kde_peaks(p,h_fine,zthr=-9)[1]
        zc[i]=kde_peaks(p,h_coarse,zthr=-9)[1]
    out=dict(maxlr=np.sort(maxlr), zf=np.sort(zf), zc=np.sort(zc),
             eps=eps, h_fine=h_fine, h_coarse=h_coarse, n_mc=n_mc)
    np.savez(CACHE,**out); return out

def make_sf(sorted_null, tail_anchor_q=0.9):
    """empirical survival + exponential tail extrapolation beyond the data."""
    s=sorted_null; n=len(s); floor=1.0/n
    u0=np.quantile(s,tail_anchor_q); m=s>u0
    c,lnA=np.polyfit(s[m], np.log(1-(np.searchsorted(s,s[m])/n)+1e-12),1)
    def sf(u):
        u=np.atleast_1d(u).astype(float); out=np.empty_like(u)
        for k,x in enumerate(u):
            pe=1-np.searchsorted(s,x,side="right")/n
            out[k]= pe if pe>=floor else np.exp(lnA+c*x)      # extrapolate tail
        return out
    return sf

def z_of(p): return float(stats.norm.isf(np.clip(p,1e-300,1-1e-12)))

# ---------------- the unified scorer ----------------
class Detector:
    def __init__(self, cal):
        self.eps=float(cal["eps"]); self.hf=float(cal["h_fine"]); self.hc=float(cal["h_coarse"])
        self.sf_lr=make_sf(cal["maxlr"]); self.sf_zf=make_sf(cal["zf"])
        self.ppk_c=LAM0*np.pi*self.hc**2
        # hybrid: coarse KDE uses RFT if Gaussian regime, else MC
        self.coarse_rft = self.ppk_c>=10
        self.sf_zc=(lambda u: rft_sf(np.atleast_1d(u),self.hc)) if self.coarse_rft else make_sf(cal["zc"])

    def score(self, pts, zthr=3.0):
        det=[]
        cl=lr_clusters(pts,self.eps)
        if cl:
            best=max(cl,key=lambda c:c[0]); p=float(self.sf_lr(best[0])[0])
            det.append(dict(method="DBSCAN+LR", stat=round(best[0],1), Nprime=best[1],
                            loc=(round(best[3],1),round(best[4],1)), p=p, z=round(z_of(p),2)))
        for h,sf,tag in [(self.hf,self.sf_zf,"KDE-fine(MC)"),
                         (self.hc,self.sf_zc,"KDE-coarse(RFT)" if self.coarse_rft else "KDE-coarse(MC)")]:
            pk,_=kde_peaks(pts,h,zthr=zthr-1)
            if pk:
                z0,cx,cy=pk[0]; p=float(sf(z0)[0])
                det.append(dict(method=tag, stat=round(z0,2), loc=(round(cx,1),round(cy,1)),
                                p=p, z=round(z_of(p),2)))
        return sorted([d for d in det], key=lambda d:-d["z"])

def splat(rng,n,cx,cy,rho):
    r=rho*np.sqrt(rng.uniform(size=n)); t=2*np.pi*rng.uniform(size=n)
    return np.column_stack((cx+r*np.cos(t),cy+r*np.sin(t)))

def main():
    rng=default_rng(2026)
    print("calibrating per-field CSR nulls (cached)...")
    cal=calibrate(rng=rng)
    det=Detector(cal)
    print(f"  LR per-field max 2lnLR: median={np.median(cal['maxlr']):.1f} max={cal['maxlr'].max():.1f}")
    print(f"  KDE coarse (h={det.hc}, {det.ppk_c:.0f} pts/kernel): calibration = {'RFT analytic' if det.coarse_rft else 'MC'}")
    # validate hybrid crossover: RFT vs MC at coarse h
    zc=cal["zc"]
    print("  hybrid check at coarse h: u | MC P(>u) | RFT P(>u)")
    for q in (0.5,0.9,0.99):
        x=np.quantile(zc,q); print(f"     {x:.2f} | {np.mean(zc>x):.2e} | {float(rft_sf(np.array([x]),det.hc)[0]):.2e}")

    print("\n--- scoring a PURE-NOISE field (expect nothing significant) ---")
    for d in det.score(csr(rng)): print("   ",d)
    print("\n--- scoring NOISE + tight clump (12 pts in r=1.0 @ (-30,20)) ---")
    f=np.vstack((csr(rng),splat(rng,12,-30,20,1.0)))
    for d in det.score(f): print("   ",d)
    print("\n--- scoring NOISE + extended splat (120 pts in r=5 @ (45,0); raw R failed here) ---")
    f=np.vstack((csr(rng),splat(rng,120,45,0,5)))
    for d in det.score(f): print("   ",d)
    print("\nNote: p/z are per-field (look-elsewhere included). LR catches tight, KDE catches extended.")

if __name__=="__main__":
    main()

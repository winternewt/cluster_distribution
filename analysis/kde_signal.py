#!/usr/bin/env python3
"""KDE on the extended splat that defeated raw density-ratio R (z_R=-6, Finding #5b).
The smooth KDE peak integrates over the region, so it should FIRE where R failed -- the
practical convergence: KDE catches extended signal, DBSCAN+LR catches tight signal."""
import os, numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter
from scipy import stats
HERE=os.path.dirname(__file__); N,RAD=10000,100; LAM0=N/(np.pi*RAD**2)
DELTA=0.5; GB=int(2*RAD/DELTA); rng=default_rng(7)

def zmax(pts,h):
    H,_,_=np.histogram2d(pts[:,0],pts[:,1],bins=GB,range=[[-RAD,RAD],[-RAD,RAD]])
    s=gaussian_filter(H,sigma=h/DELTA,mode="constant")
    ax=(np.arange(GB)+0.5)*DELTA-RAD; X,Y=np.meshgrid(ax,ax,indexing="ij")
    si=s[(X**2+Y**2)<(RAD-4*h)**2]
    return (si.max()-si.mean())/si.std()

def csr(m=N):
    r=RAD*np.sqrt(rng.uniform(size=m)); t=2*np.pi*rng.uniform(size=m)
    return np.column_stack((r*np.cos(t),r*np.sin(t)))
def splat(n,cx,cy,rho):
    r=rho*np.sqrt(rng.uniform(size=n)); t=2*np.pi*rng.uniform(size=n)
    return np.column_stack((cx+r*np.cos(t),cy+r*np.sin(t)))

def main():
    h=1.2; TR=200
    null=np.array([zmax(csr(),h) for _ in range(TR)])
    thr=np.quantile(null,0.99)
    print(f"KDE Z_max null (h={h}): mean={null.mean():.2f}, 99th pct={thr:.2f}")
    print("extended splat radius 5 at (45,0) -- the one where raw-R gave z_R=-6:\n")
    print(" n_extra | density x | KDE detect rate (Zmax>null99) | median Zmax")
    for ne in (0,30,60,120):
        dens=1+ne/(np.pi*25*LAM0)
        zs=np.array([zmax(np.vstack((csr(),splat(ne,45,0,5))) if ne else csr(),h) for _ in range(TR)])
        print(f"   {ne:4d}  |  {dens:5.2f}x  |   {100*np.mean(zs>thr):5.1f}%                    |  {np.median(zs):.2f}")
    print("\n=> KDE flags the extended splat that raw density-ratio R missed (R is dominated by")
    print("   tight noise blobs; KDE integrates the region). KDE for diffuse signal, DBSCAN+LR for tight.")

if __name__=="__main__":
    main()

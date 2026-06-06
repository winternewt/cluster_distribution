#!/usr/bin/env python3
"""DEFINITIVE deep-tail extrapolator: per-N' power law with the tail index measured from
the convex-hull GEOMETRY (uncensored, no DBSCAN), anchored to the data body.

Resolution of the earlier iterations:
  - tail_extrapolator.py (GPD on 2lnLR): xi<0 finite endpoint = min_area CENSORING artifact (too light).
  - tail_analytic.py (inverse-gamma body): inverse-gamma fits the BODY (k~20) but its tail is far too light.
  - tail_powerlaw.py (Hill on censored R): right FORM (power law) but index biased by truncation at the cap.
  - HERE: the R|n tail index = the convex-hull small-area exponent of n uniform disk points,
    measured directly from geometry (alpha~7 for n=10), anchored to the data => the censoring is
    irrelevant because the index comes from uncensored geometry.

For each n: P(R>r|n) = Pu_n * (r/u_n)^(-alpha_n),  alpha_n from geometry,  Pu_n,u_n from data body.
Marginal P(2lnLR>u) = sum_n P(N'=n) P(R>R*(u,n)|n).
"""
import os, sys, numpy as np, pandas as pd
from numpy.random import default_rng
from scipy import stats, optimize
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.simv2_data import load_raw_df

HERE=os.path.dirname(__file__); N,RAD,EPS=10000,100,1.20; LAM0=N/(np.pi*RAD**2)
CAP=lambda n: n/(0.5*LAM0); rng=default_rng(99)

def twolnLR(n,R):
    mu=n/R; return 2*(n*np.log(R)+(N-n)*np.log((N-n)/(N-mu)))
def Rstar(u,n):
    f=lambda R: twolnLR(n,R)-u
    return 1.0001 if f(1.0001)>0 else optimize.brentq(f,1.0001,1e8)

def geom_alpha(n,M=300000):
    """tail index alpha_n = small-hull-area exponent of n uniform points in a disk."""
    a=np.empty(M)
    for i in range(M):
        r=np.sqrt(rng.uniform(size=n)); t=2*np.pi*rng.uniform(size=n)
        a[i]=ConvexHull(np.column_stack((r*np.cos(t),r*np.sin(t)))).volume
    a=np.sort(a); k=int(0.005*M); lo=a[:k]; cdf=np.arange(1,k+1)/M
    return float(np.polyfit(np.log(lo[lo>0]),np.log(cdf[lo>0]),1)[0])

def load():
    d=load_raw_df(EPS)
    if d is None:
        return None, None
    n=d.N_prime.values.astype(int)
    return n,(n/(d.S_prime.values))/LAM0

def main():
    nv,Rv=load()
    if nv is None:
        print("data unavailable"); return
    Lv=twolnLR(nv.astype(float),Rv)
    occ={}; pl={}
    print("measuring geometry tail index alpha(n) (uncensored, no DBSCAN):")
    for n in range(10,17):
        m=nv==n; w=m.mean()
        if w<2e-4: continue
        r=Rv[m]; u_n=np.quantile(r,0.55); Pu=np.mean(r>u_n)
        al=geom_alpha(n)
        occ[n]=w; pl[n]=dict(u=u_n,Pu=Pu,alpha=al)
        print(f"  n={n} (w={w:.3f}): alpha_geom={al:.2f}  (data-anchored at R>{u_n:.1f}, P={Pu:.3f})")
    s=sum(occ.values()); occ={k:v/s for k,v in occ.items()}

    def cond(r,p): return p['Pu']*(r/p['u'])**(-p['alpha']) if r>p['u'] else 1.0
    def marg(u,da=0.0):
        return sum(occ[n]*cond(Rstar(u,n),{**pl[n],'alpha':pl[n]['alpha']+da}) for n in occ)
    z=lambda p: stats.norm.isf(np.clip(p,1e-300,1-1e-12))

    grid=np.array([50,55,60,63,70,80,90,100,110,120],float)
    mid=np.array([marg(u) for u in grid])
    hi=np.array([marg(u,-0.5) for u in grid]); lo=np.array([marg(u,+0.5) for u in grid])  # +-0.5 in alpha
    bf=np.array([np.mean(Lv>u) for u in grid])
    print("\n  2lnLR | bruteforce(cens) | power-law(geom alpha) [alpha+-0.5] | z")
    for i,u in enumerate(grid):
        bfs=f"{bf[i]:.2e}" if bf[i]>0 else " 0 (cens)"
        print(f"   {u:5.0f} |   {bfs:>11}    | {mid[i]:.2e} [{lo[i]:.1e},{hi[i]:.1e}] | {z(mid[i]):5.2f}")
    print("\n  validation in uncensored window (u well below the N'=10 cap, R<~45):")
    for u in (48,52,56):
        print(f"   2lnLR={u}: model {marg(u):.2e}  vs brute force {np.mean(Lv>u):.2e}  (ratio {marg(u)/np.mean(Lv>u):.2f})")

    pd.DataFrame(dict(twolnLR=grid,p_bruteforce=bf,p_model=mid,lo=lo,hi=hi,z=z(mid))).to_csv(
        os.path.join(HERE,"tail_final_table.csv"),index=False)
    fig,ax=plt.subplots(figsize=(9,6)); us=np.linspace(45,122,150)
    ax.semilogy(np.sort(Lv),1-np.arange(len(Lv))/len(Lv),'k-',lw=1.3,label='brute force (censored)')
    ax.semilogy(us,[marg(u) for u in us],'b-',lw=2,label=f"power law, geometry alpha (n10~{pl[10]['alpha']:.1f})")
    ax.fill_between(grid,lo,hi,color='b',alpha=0.2,label='alpha +-0.5 band')
    ax.axvline(63,ls=':',c='grey'); ax.text(63.5,3e-12,'min_area cap',rotation=90,fontsize=8)
    ax.set_xlabel('2 ln LR'); ax.set_ylabel('survival'); ax.set_ylim(1e-13,1); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Definitive tail: per-N power law, geometry-measured index')
    fig.tight_layout(); fig.savefig(os.path.join(HERE,'tail_final.png'),dpi=110); plt.close(fig)
    print("\nwrote tail_final_table.csv, tail_final.png")

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""CORRECT deep-tail extrapolator: per-N' power-law (Pareto) tail from the small-hull-area
geometry. Supersedes the two earlier attempts, both of which under-stated the tail:

  - GPD on 2lnLR (tail_extrapolator.py): xi<0 finite endpoint = min_area CENSORING artifact.
  - inverse-gamma-body mixture (tail_analytic.py): inverse-gamma fits the BODY (shape ~20)
    but its tail is far too light. The MEASURED tail index of R|N'=10 is alpha~7.7,
    matching the convex-hull geometry P(area<a) ~ a^(n-1)  =>  P(R>r|n) ~ r^-(n-1).

Method (per count n, anchored to data, extrapolates PAST the min_area censoring):
  1. Hill estimate alpha(n) on R|n exceedances in the UNCENSORED window [u_n, 0.92*cap_n];
  2. conditional tail  P(R>r|n) = P(R>u_n|n) * (r/u_n)^(-alpha(n))   for r>u_n;
  3. marginal  P(2lnLR>u) = sum_n P(N'=n) * P(R > R*(u,n) | n);
  4. Hill SE alpha/sqrt(k) -> confidence band.
"""
import os, sys, numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.simv2_data import load_raw_df

HERE=os.path.dirname(__file__); N,RAD,EPS=10000,100,1.20; LAM0=N/(np.pi*RAD**2)
CAP=lambda n: n/(0.5*LAM0)

def twolnLR(n,R):
    mu=n/R
    return 2*(n*np.log(R)+(N-n)*np.log((N-n)/(N-mu)))
def Rstar(u,n):
    f=lambda R: twolnLR(n,R)-u
    return 1.0001 if f(1.0001)>0 else optimize.brentq(f,1.0001,1e7)

def load():
    d=load_raw_df(EPS)
    if d is None:
        return None, None
    n=d.N_prime.values.astype(int); R=(n/(d.S_prime.values))/LAM0
    return n,R

def hill_per_n(nv,Rv,nmax=20):
    occ={}; pl={}
    for n in range(10,nmax+1):
        m=nv==n; w=m.mean()
        if w<2e-4: continue
        r=Rv[m]; cap=CAP(n)
        u_n=np.quantile(r,0.55)                       # tail threshold (in body, below cap)
        seg=r[(r>u_n)&(r<0.92*cap)]                   # uncensored exceedances
        if len(seg)<80:
            occ[n]=w; pl[n]=None; continue
        k=len(seg)
        alpha=k/np.sum(np.log(seg/u_n))               # Hill estimator
        Pu=np.mean(r>u_n)                             # anchor P(R>u_n | n)
        occ[n]=w; pl[n]=dict(u=u_n,alpha=alpha,Pu=Pu,se=alpha/np.sqrt(k),k=k)
    s=sum(occ.values()); occ={k:v/s for k,v in occ.items()}
    return occ,pl

def cond_tail(r,p):
    if p is None: return 0.0
    return p['Pu']*(r/p['u'])**(-p['alpha']) if r>p['u'] else min(1.0, p['Pu']+(1-p['Pu'])*0)  # body~anchor

def marg(u,occ,pl,dalpha=0.0):
    s=0.0
    for n,w in occ.items():
        p=pl[n]
        if p is None: continue
        rs=Rstar(u,n)
        pp=dict(p); pp['alpha']=max(p['alpha']+dalpha*p['se'],1.5)
        s+=w*cond_tail(rs,pp)
    return s
def z_of(p): return stats.norm.isf(np.clip(p,1e-300,1-1e-12))

def main():
    nv,Rv=load()
    if nv is None:
        print("data unavailable"); return
    Lv=twolnLR(nv.astype(float),Rv)
    occ,pl=hill_per_n(nv,Rv)
    print(f"eps={EPS}: {len(nv)} clusters; max 2lnLR={Lv.max():.1f}")
    print("per-N' power-law tail (Hill):")
    for n in sorted(occ):
        if pl[n]: print(f"  n={n} (w={occ[n]:.3f}): alpha={pl[n]['alpha']:.2f}+-{pl[n]['se']:.2f}  (geometry n-1={n-1})")

    grid=np.array([50,55,60,63,70,80,90,100,110],float)
    mid=np.array([marg(u,occ,pl) for u in grid])
    # +-1.65 sigma band by jointly shifting alpha (conservative: lower alpha=heavier upper band)
    hi=np.array([marg(u,occ,pl,dalpha=-1.65) for u in grid])
    lo=np.array([marg(u,occ,pl,dalpha=+1.65) for u in grid])
    bf=np.array([np.mean(Lv>u) for u in grid])
    print("\n  2lnLR | bruteforce(cens) | power-law tail [90% band]      | z")
    for i,u in enumerate(grid):
        bfs=f"{bf[i]:.2e}" if bf[i]>0 else " 0 (cens)"
        print(f"   {u:5.0f} |   {bfs:>11}    | {mid[i]:.2e} [{lo[i]:.1e},{hi[i]:.1e}] | {z_of(mid[i]):5.2f}")

    print("\n  validation vs brute force in the UNCENSORED window (u where data is complete):")
    print("   2lnLR | power-law | brute force | ratio")
    for u in (50,55,60,63):
        pe=np.mean(Lv>u); pm=marg(u,occ,pl)
        print(f"   {u:4.0f} |  {pm:.2e} |  {pe:.2e}  | {pm/pe:.2f}")

    pd.DataFrame(dict(twolnLR=grid,p_bruteforce=bf,p_powerlaw=mid,lo=lo,hi=hi,z=z_of(mid))).to_csv(
        os.path.join(HERE,"tail_powerlaw_table.csv"),index=False)
    fig,ax=plt.subplots(figsize=(9,6)); us=np.linspace(45,112,140)
    ax.semilogy(np.sort(Lv),1-np.arange(len(Lv))/len(Lv),'k-',lw=1.3,label='brute force (censored)')
    mm=[marg(u,occ,pl) for u in us]
    ax.semilogy(us,mm,'b-',lw=2,label='power-law tail (alpha~n-1, uncensored)')
    ax.fill_between(grid,lo,hi,color='b',alpha=0.2,label='90% band (Hill SE)')
    ax.axvline(63,ls=':',c='grey'); ax.text(63.5,3e-11,'min_area cap (N=10)',rotation=90,fontsize=8)
    ax.set_xlabel('2 ln LR'); ax.set_ylabel('survival'); ax.set_ylim(1e-12,1); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Correct deep tail: per-N power law (heavy, geometry-set)')
    fig.tight_layout(); fig.savefig(os.path.join(HERE,'tail_powerlaw.png'),dpi=110); plt.close(fig)
    print("\nwrote tail_powerlaw_table.csv, tail_powerlaw.png")

if __name__=="__main__":
    main()

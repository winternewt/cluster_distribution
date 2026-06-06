#!/usr/bin/env python3
"""Principled analytic tail extrapolator: per-N' inverse-gamma mixture (uncensored)
+ GPD/POT cross-check, both with confidence bands. Fast (no hull MC).

WHY this beats GPD-on-the-raw-data: the brute-force statistic is censored by min_area
(it drops S'<0.5 => caps R at 6.28n per N'=n), so a GPD fit to it sees a spuriously
light, finite-endpoint tail and UNDER-states rare events. But Finding #2 proved
R|N'=n ~ scaled inverse-gamma(shape k(n)) EXACTLY, and inverse-gamma has an analytic
polynomial tail R^-k(n). So:

  1. fit inverse-gamma to R|n using only the UNCENSORED BODY of each n (R < 0.8*cap_n);
  2. the marginal survival is  P(2lnLR>u) = sum_n P(N'=n) * P(R > R*(u,n) | n),
     with R*(u,n) the (monotone) inverse of 2lnLR(n,R)=u and the conditional tail taken
     from the analytic inverse-gamma -> extrapolates PAST the censoring, for free;
  3. bootstrap over clusters for confidence bands.

Reports brute-force (ground truth, censored), GPD (censored lower bound), and the
inverse-gamma mixture (uncensored, recommended), with z and 90% CI. eps=1.20.
"""
import os, sys, numpy as np, pandas as pd
from numpy.random import default_rng
from scipy import stats, optimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.simv2_data import load_raw_df

HERE=os.path.dirname(__file__); N,RAD,EPS=10000,100,1.20; LAM0=N/(np.pi*RAD**2)
rng=default_rng(2718)
CAP=lambda n: n/(0.5*LAM0)                      # min_area=0.5 censoring cap on R for count n

def twolnLR(n,R):
    mu=n/R                                       # mu=lambda0*S', and n/mu=R
    return 2*(n*np.log(R)+(N-n)*np.log((N-n)/(N-mu)))

def Rstar(u,n):
    """invert 2lnLR(n,R)=u for R>1 (monotone increasing)."""
    f=lambda R: twolnLR(n,R)-u
    if f(1.0001)>0: return 1.0001
    hi=1e6
    return optimize.brentq(f,1.0001,hi)

def load():
    d=load_raw_df(EPS)
    if d is None:
        return None, None
    n=d.N_prime.values.astype(int); S=d.S_prime.values; R=(n/S)/LAM0
    return n,R

def fit_per_n(nv,Rv,nmax=20):
    """inverse-gamma shape/scale per n, fit on the uncensored body R<0.8*cap_n."""
    occ={}; fits={}
    for n in range(10,nmax+1):
        m=nv==n; w=m.mean()
        if w<2e-4: continue
        occ[n]=w
        r=Rv[m]; body=r[r<0.8*CAP(n)]           # avoid the censored region
        if len(body)<300:
            fits[n]=None; continue
        sh,_,sc=stats.invgamma.fit(body,floc=0)
        fits[n]=(sh,sc)
    s=sum(occ.values()); occ={k:v/s for k,v in occ.items()}   # renormalize over kept n
    return occ,fits

def marginal_sf(u,occ,fits):
    p=0.0
    for n,w in occ.items():
        if fits[n] is None: continue
        sh,sc=fits[n]
        Rs=Rstar(u,n)
        p+=w*stats.invgamma.sf(Rs,sh,0,sc)       # analytic UNCENSORED conditional tail
    return p

def gpd_sf_factory(stat,q=0.90):
    u0=np.quantile(stat,q); exc=stat[stat>u0]-u0; zeta=len(exc)/len(stat)
    xi,_,sig=stats.genpareto.fit(exc,floc=0)
    return (lambda u: zeta*stats.genpareto.sf(np.asarray(u)-u0,xi,0,sig)), xi,u0

def z_of(p): return stats.norm.isf(np.clip(p,1e-300,1-1e-12))

def main():
    nv,Rv=load()
    if nv is None:
        print("data unavailable"); return
    Lv=twolnLR(nv.astype(float),Rv)
    print(f"eps={EPS}: {len(nv)} clusters; max 2lnLR={Lv.max():.1f}")
    occ,fits=fit_per_n(nv,Rv)
    print("per-N' inverse-gamma (body fit): " + ", ".join(
        f"n{n}:k={fits[n][0]:.1f}(w{occ[n]:.2f})" for n in sorted(occ) if fits[n]))

    gpd,xi,u0=gpd_sf_factory(Lv)
    print(f"GPD on 2lnLR: xi={xi:+.3f}, u0={u0:.1f} (censored lower bound)\n")

    # bootstrap the inverse-gamma mixture (resample clusters)
    grid=np.array([50,55,60,63,70,80,90,100,110],float)
    B=120; boot=np.zeros((B,len(grid)))
    for b in range(B):
        idx=rng.integers(0,len(nv),len(nv))
        o,f=fit_per_n(nv[idx],Rv[idx])
        boot[b]=[marginal_sf(u,o,f) for u in grid]
    lo,hi=np.nanpercentile(boot,5,axis=0),np.nanpercentile(boot,95,axis=0)
    mix=np.array([marginal_sf(u,occ,fits) for u in grid])
    bf =np.array([np.mean(Lv>u) for u in grid])
    gp =gpd(grid)

    print("  2lnLR | bruteforce(cens) |  GPD(cens) | inv-gamma MIX [90% CI]        | z_mix")
    for i,u in enumerate(grid):
        bfs=f"{bf[i]:.2e}" if bf[i]>0 else " 0 (cens)"
        print(f"   {u:5.0f} |   {bfs:>11}    | {gp[i]:.2e} | {mix[i]:.2e} [{lo[i]:.1e},{hi[i]:.1e}] | {z_of(mix[i]):5.2f}")

    # holdout: fit inverse-gamma mix on data, predict where brute force has events -> compare
    print("\n  holdout vs brute-force (where brute force still has events):")
    print("   2lnLR | inv-gamma mix | brute force | ratio")
    for u in (60,66,72,78):
        pe=np.mean(Lv>u); pm=marginal_sf(u,occ,fits)
        print(f"   {u:4d} |   {pm:.2e}   |  {pe:.2e}  | {pm/pe:.2f}" if pe>0 else f"   {u:4d} |   {pm:.2e}   |   0(cens)  |  —")

    pd.DataFrame(dict(twolnLR=grid,p_bruteforce=bf,p_gpd=gp,p_invgamma_mix=mix,
                      lo=lo,hi=hi,z_mix=z_of(mix))).to_csv(os.path.join(HERE,"tail_analytic_table.csv"),index=False)

    fig,ax=plt.subplots(figsize=(9,6)); us=np.linspace(40,112,160)
    ax.semilogy(np.sort(Lv),1-np.arange(len(Lv))/len(Lv),'k-',lw=1.3,label='brute force (censored)')
    ax.semilogy(us,[gpd(u) for u in us],'g-.',lw=1.5,label=f'GPD (censored, xi={xi:+.2f})')
    mm=[marginal_sf(u,occ,fits) for u in us]
    ax.semilogy(us,mm,'b-',lw=2,label='inverse-gamma mixture (uncensored)')
    ax.fill_between(grid,lo,hi,color='b',alpha=0.2,label='inv-gamma 90% band')
    ax.axvline(63,ls=':',c='grey'); ax.text(63.5,3e-9,'min_area cap (N=10)',rotation=90,fontsize=8)
    ax.set_xlabel('2 ln LR'); ax.set_ylabel('survival P(>=)'); ax.set_ylim(1e-12,1); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Analytic tail: inverse-gamma mixture (uncensored) vs GPD vs brute force')
    fig.tight_layout(); fig.savefig(os.path.join(HERE,'tail_analytic.png'),dpi=110); plt.close(fig)
    print("\nwrote tail_analytic_table.csv, tail_analytic.png")

if __name__=="__main__":
    main()

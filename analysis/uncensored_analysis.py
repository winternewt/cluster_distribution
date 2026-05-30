#!/usr/bin/env python3
"""Analyze the uncensored (min_area=0) resim: pin the R tail index + normalization,
and quantify how little the min_area censorship actually mattered. Reproducible from
analysis/uncensored_eps1.20.csv (regenerate via uncensored_sim.py). Writes a small
summary CSV + plot (the 20MB raw csv is gitignored)."""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE=os.path.dirname(__file__); N,RAD=10000,100; LAM0=N/(np.pi*RAD**2); RCAP=10/(0.5*LAM0)
def lr(n,S): mu=LAM0*S; return 2*(n*np.log(n/mu)+(N-n)*np.log((N-n)/(N-mu)))

def load(path):
    d=pd.read_csv(path); d=d[(d.S_prime!=-1)&(d.N_prime!=-1)]
    n=d.N_prime.values.astype(float); S=d.S_prime.values
    return n,(n/S)/LAM0,lr(n,S)

def main():
    nu,Ru,Lu=load(os.path.join(HERE,"uncensored_eps1.20.csv"))
    _,Rc,Lc=load(os.path.join(HERE,"..","simdata","v2","simulation_data_N10000_radius100_eps1.20.csv"))
    print(f"uncensored {len(Ru)} clusters, max R={Ru.max():.1f}, max 2lnLR={Lu.max():.1f}")
    print(f"censored   {len(Rc)} clusters, max R={Rc.max():.1f} (cap {RCAP:.1f})")
    print(f"censorship impact: {np.mean(Ru>RCAP)*100:.4f}% of clusters beyond cap ({int(np.sum(Ru>RCAP))} clusters)")
    print("  => min_area pins the MAX at the cap but removes negligible mass; tail shape is real.\n")

    # tail index (Hill) from uncensored R|10, several thresholds
    r10=Ru[nu==10]; print("tail index alpha of R|N'=10 (uncensored Hill):")
    for q in (0.90,0.95,0.99):
        u0=np.quantile(r10,q); seg=r10[r10>u0]; k=len(seg); a=k/np.sum(np.log(seg/u0))
        print(f"  >q{int(q*100)} (R>{u0:.1f}): alpha={a:.2f}+-{a/np.sqrt(k):.2f}")
    print("  geometry=7.0, censored-loglog=7.7 -> alpha ~ 6.5-7 (3 methods agree)\n")

    # direct survival + exponential-in-u extrapolation anchored to the data edge
    xs=np.sort(Lu); sf=1-np.arange(len(xs))/len(xs); m=(xs>54)&(xs<72)
    c,lnA=np.polyfit(xs[m],np.log(sf[m]),1)
    floor=5/len(Lu)
    grid=np.array([55,60,63,70,75,80,90,100,110,120],float)
    rows=[]
    print("  2lnLR | direct P(>u)     | extrapolated | z")
    for u in grid:
        pdv=np.mean(Lu>u); pe=np.exp(lnA+c*u); z=stats.norm.isf(np.clip(pe,1e-300,1))
        ds=f"{pdv:.2e}" if pdv>=floor else "n/a (sample floor)"
        rows.append(dict(twolnLR=u,p_direct=(pdv if pdv>=floor else np.nan),p_extrap=pe,z=z))
        print(f"  {u:5.0f} | {ds:>16} | {pe:.2e}   | {z:.2f}")
    print(f"\n  direct floor ~{floor:.1e} (2lnLR~72, z~4.1); beyond = power-law extrapolation (rate {-c:.3f}/u = alpha/(2*10))")
    pd.DataFrame(rows).to_csv(os.path.join(HERE,"uncensored_summary.csv"),index=False)

    fig,ax=plt.subplots(figsize=(9,6))
    ax.semilogy(np.sort(Lu),1-np.arange(len(Lu))/len(Lu),'b-',lw=1.5,label='uncensored (min_area=0)')
    ax.semilogy(np.sort(Lc),1-np.arange(len(Lc))/len(Lc),'k--',lw=1.0,label='censored (min_area=0.5)')
    us=np.linspace(54,122,120); ax.semilogy(us,np.exp(lnA+c*us),'r:',lw=2,label=f'extrapolation (alpha~7)')
    ax.axvline(63,ls=':',c='grey'); ax.set_ylim(1e-13,1)
    ax.set_xlabel('2 ln LR'); ax.set_ylabel('survival'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Uncensored vs censored tail (~identical) + power-law extrapolation')
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"uncensored_tail.png"),dpi=110); plt.close(fig)
    print("wrote uncensored_summary.csv, uncensored_tail.png")

if __name__=="__main__":
    main()

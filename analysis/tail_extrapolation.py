#!/usr/bin/env python3
"""Non-brute-force deep tail: the Kulldorff LR tail is the large-deviations rate
function (= entropy cost of the fluctuation), so it is EXPONENTIAL and extrapolable
from a small sim. Also quantifies the PRNG fidelity horizon ('when would Diehard fail').

2lnLR = 2 * S' * D(rho || lambda0)  with  D = rho ln(rho/lambda0) - (rho - lambda0)
      = twice the relative-entropy (KL) cost of the density fluctuation.
Entropy cost in bits = (2lnLR)/(2 ln2) = 0.721 * (2lnLR).

Demonstrations:
  1. fit the exponential LR tail on a SMALL subsample, extrapolate, compare to the full
     132k-cluster empirical tail -> shows you don't need 1e9 sims;
  2. entropy (bits) at each tail depth;
  3. PRNG floors: seed-entropy (PCG64 128-bit) vs BigCrush-validation horizon.
"""
import numpy as np, pandas as pd

N, RAD = 10000, 100
LAM0 = N/(np.pi*RAD**2)

def lr(d):
    n = d.N_prime.values.astype(float); S = d.S_prime.values
    mu = LAM0*S
    return 2*(n*np.log(n/mu) + (N-n)*np.log((N-n)/(N-mu)))

def main():
    df = pd.read_csv('simdata/v2/simulation_data_N10000_radius100_eps1.20.csv')
    niter = int(df['iteration'].max())
    d = df[(df.S_prime!=-1)&(df.N_prime!=-1)]
    L_full = lr(d)
    cpf = len(d)/niter
    print(f"FULL data: {len(d)} clusters from {niter} fields ({cpf:.3f}/field); max 2lnLR={L_full.max():.1f}")

    # 1) extrapolate from a SMALL campaign (mimic ~50k fields => ~6600 clusters)
    rng = np.random.default_rng(0)
    small = rng.choice(L_full, 6600, replace=False)
    # fit exponential tail P(>u) ~ A exp(-c u) on the small sample's BODY (u in [40,48])
    Ls = np.sort(small); sf = 1-np.arange(len(Ls))/len(Ls)
    fitm = (Ls>40)&(Ls<48)
    c, lnA = np.polyfit(Ls[fitm], np.log(sf[fitm]), 1)
    print(f"\nfit on SMALL sim (6600 clusters), tail u in [40,48]: P(>u) ~ exp({lnA:.2f}{c:+.3f}*u)")
    print("  extrapolate vs FULL-data empirical (where full data still has events):")
    print("   u    predicted P(>u)   empirical P(>u)[full]   ratio")
    for u in (48, 54, 60, 66, 72):
        pred = np.exp(lnA + c*u)
        emp = np.mean(L_full > u)
        r = f"{pred/emp:.2f}" if emp>0 else "—(0 events)"
        print(f"  {u:3d}   {pred:.2e}        {emp:.2e}            {r}")

    # 2) entropy in bits and per-field probabilities
    print("\nentropy & per-field tail prob (extrapolated):")
    print("   2lnLR   nats   bits   p_per_cluster   p_per_field   #fields for ~10 events")
    for u in (50, 63, 80, 100):
        pc = np.exp(lnA + c*u); pf = pc*cpf
        print(f"   {u:4d}   {u/2:4.0f}  {u/(2*np.log(2)):5.0f}   {pc:.2e}     {pf:.2e}   {10/pf:.1e}")

    # 3) PRNG floors
    print("\n--- PRNG fidelity horizon (when can't we trust simulated tails?) ---")
    bits_per_field = 2*N*64/ (2*N*64)  # placeholder for clarity below
    draws_per_field = 2*N               # r, theta per point
    print(f"  draws/field = 2N = {draws_per_field}")
    # seed-entropy hard floor: PCG64 state 128 bits -> events rarer than 2^-128 cannot occur
    seed_bits = 128
    u_seed = seed_bits*2*np.log(2)
    print(f"  (a) PCG64 seed/state = {seed_bits} bits -> hard floor p~2^-128~3e-39 (2lnLR~{u_seed:.0f}); NON-binding")
    # BigCrush validation horizon: PCG64 validated to ~2^38-2^40 outputs
    for vbits in (38, 40):
        n_outputs = 2**vbits
        n_fields = n_outputs/draws_per_field
        p_floor = 1.0/n_fields            # smallest per-field prob with ~1 event in validated envelope
        # convert to 2lnLR via per-field = per-cluster*cpf -> per-cluster = p_floor/cpf
        u_floor = (np.log(p_floor/cpf)-lnA)/c
        print(f"  (b) BigCrush ~2^{vbits} outputs -> ~{n_fields:.1e} fields -> p_floor(field)~{p_floor:.1e} "
              f"(2lnLR~{u_floor:.0f}, ~{u_floor/(2*np.log(2)):.0f} bits)")
    print("  => PRNG-validation horizon ~2lnLR 90-100 (p_field~1e-7..1e-8) ~ same place brute")
    print("     force dies. NOTE: the min_area CENSORING artifact bites earlier (2lnLR~63,")
    print("     p_field~1.6e-4), so it is the binding limit now; remove it and the PRNG/compute")
    print("     horizon governs. Below ~1e-8: counter-based RNG (Philox) + analytic LDT")
    print("     extrapolation + importance sampling, not brute MC.")

if __name__ == "__main__":
    main()

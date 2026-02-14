# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import liquidcosmo as lc

# +
import numpy as np

def S8_tension(chain):
    # Get sigma8 and omega_m samples
    BASE_n = "/Users/bellamarco01/uni/1_master_thesis/montepython_chains/chains_mp/varying_zc_npipe/"
    print("chain", chain)
    chain = BASE_n + chain
    fo = lc.load(chain)
    sigma8_samples = fo['sigma8']  # Returns array of sigma8 values
    Omega_m_samples = fo['Omega_m']  # Returns array of omega_m values

    # Compute S8 for each sample
    S8_samples = sigma8_samples * np.sqrt(Omega_m_samples / 0.3)

    # Now compute statistics from S8_samples
    S8_mean = np.mean(S8_samples)
    # S8_std = np.std(S8_samples)

    # Get percentiles for asymmetric errors
    # S8_median = np.median(S8_samples)
    S8_16 = np.percentile(S8_samples, 16)  # -1σ
    S8_84 = np.percentile(S8_samples, 84)  # +1σ
    S8_up = S8_84 - S8_mean
    S8_down = S8_mean - S8_16

    # print(f"S8 = {S8_mean:.4f} ± {S8_std:.4f}")
    print(f"S8 = {S8_mean:.4f} +{S8_up:.4f} -{S8_down:.4f} (68% CL)")

    # Your measurement from liquidcosmo

    # Other measurement with asymmetric errors
    S8_des = 0.790
    S8_up_des = 0.018
    S8_down_des = 0.014  # positive value

    # Compute tension separately for upper and lower sides
    if S8_mean > S8_des:
        # Your measurement is above theirs - use their upper error
        delta_up = S8_mean - S8_des
        tension_up = delta_up / np.sqrt(S8_down**2 + S8_up_des**2)
        print(f"Tension (gaussian): {tension_up:.2f}σ")
    else:
        # Your measurement is below theirs - use their lower error
        delta_down = S8_des - S8_mean
        tension_down = delta_down / np.sqrt(S8_up**2 + S8_down_des**2)
        print(f"Tension (gaussian): {tension_down:.2f}σ")


# +
import liquidcosmo as lc
import getdist
import numpy as np
from tensiometer import mcmc_tension
from tensiometer.utilities import stats_utilities as utilities

def myTensioMeter(chain):
    BASE_n = "/Users/bellamarco01/uni/1_master_thesis/montepython_chains/chains_mp/varying_zc_npipe/"
    # print("chain", chain)
    chain = BASE_n + chain
    fo = lc.load(chain)
    samp = fo.to_getdist()

    # Calculate S8 values
    # s8_vals = gd.getParams().sigma8 * (gd.getParams().Omega_m / 0.3)**0.5

    # Add derived parameter
    # gd.addDerived(s8_vals, name='S8', label=r'S_8')
    # Other measurement with asymmetric errors
  

    # samp.addDerived(samp.getParams().sigma8*(samp.getParams().Omega_m/0.3)**0.5, name='S8', label=r'S_8')
    S8_chain = samp.getParams().sigma8 * (samp.getParams().Omega_m / 0.3)**0.5 

    def split_normal(mean, sigma_minus, sigma_plus, size):
        u = np.random.rand(size)
        samples = np.empty(size)

        left = u < 0.5
        right = ~left

        samples[left] = mean + np.random.normal(0, sigma_minus, left.sum())
        samples[right] = mean + np.random.normal(0, sigma_plus, right.sum())

        return samples


    S8_des_samples = split_normal(        #2305.17173
        mean=0.790,
        sigma_minus=0.014,
        sigma_plus=0.018,
        size=len(S8_chain)
    )


    # difference chain
    delta_S8 = S8_chain - S8_des_samples
    lo, hi = np.percentile(delta_S8, [16, 84])
    sigma_delta = 0.5 * (hi - lo)
    mu_delta = np.mean(delta_S8)

    # print(f"ΔS8 = {mu_delta:.4f} ± {sigma_delta:.4f}")
    T = mu_delta / sigma_delta
    print(f"Tension (my tensiometer): {T:.2f} sigma")



    # tension = fo.tension(lc.get_gaussian_chain(H0, std=sigma_H0, names=['H0']), metric="parameter_difference");


# +
# from pathlib import Path

chain_1pd = "1_axion/planck_TTTEEElensing_pantheon_plus_desi_2026-01-30"
chain_1psd = "1_axion/planck_TTTEEElensing_pantheon_plus_sh0es_desi_2026-01-30"
chain_2pd = "2_axions/planck_TTTEEElensing_pantheon_plus_desi_2026-02-02"
chain_2psd = "2_axions/planck_TTTEEElensing_pantheon_plus_sh0es_desi_2026-02-02"

chains = [chain_1pd, chain_1psd, chain_2pd, chain_2psd]

# fo_2psd = lc.load(str(chain_2psd))
for chain in chains:
    S8_tension(chain)
    myTensioMeter(chain)
    print("------------------------")

# +
BASE_n = "/Users/bellamarco01/uni/1_master_thesis/montepython_chains/chains_mp/varying_zc_npipe/"
for i in range(len(chains)):
    chain = BASE_n + chains[i]
    print(chains[i])

    fo = lc.load(chain)
    # tension = fo.tension(lc.get_gaussian_chain(H0, std=sigma_H0, names=['H0']), metric="parameter_difference");
    # print(tension)
    ###if your chain is big the evidence can be very long to compute; you can use a thinning factor.
    fo = fo.thin(10)
    evidence = fo.log_evidence(subdivide=True, cosmo_only=True);

    print(evidence)



# -





#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hierarchical Bayesian UV metallicity inference pipeline

Major features
--------------
1. Hierarchical metallicity population model
2. Joint EW likelihood with covariance matrix
3. Simultaneous inference of metallicity and stellar population age
4. ETC pipeline used to build EW model grid
5. Cached model grid to avoid recomputation
6. MCMC inference using emcee

Author: <your name>
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os
import glob
import h5py
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re

from scipy.interpolate import RegularGridInterpolator
from synthesizer.grid import Grid
from synthesizer.emissions import Sed
from unyt import angstrom

import etc_noise_pipeline as etc


# ==========================================================
# USER SETTINGS
# ==========================================================

OBS_EW_DIR = "observed_EWs/"
GRID_DIR = "/users/csant/data/grids"
GRID_NAME = "bpass-2.3-bin_bpl-0.1,1.0,300.0-1.3,2.35_alpha0.0.hdf5"

N_WALKERS = 80
N_STEPS = 8000

RNG_SEED = 42


# ==========================================================
# UV INDEX DEFINITIONS
# ==========================================================

def get_uv_indices():

    index = [1370,1400,1425,1460,1501,1533,1550,1719,1853]

    index_window = [
        [1360,1380],[1385,1410],[1413,1435],
        [1450,1470],[1496,1506],[1530,1537],
        [1530,1560],[1705,1729],[1838,1858]
    ]

    blue_window = [
        [1345,1354],[1345,1354],[1345,1354],
        [1436,1447],[1482,1491],[1482,1491],
        [1482,1491],[1675,1684],[1797,1807]
    ]

    red_window = [
        [1436,1447],[1436,1447],[1436,1447],
        [1482,1491],[1583,1593],[1583,1593],
        [1583,1593],[1751,1761],[1871,1883]
    ]

    return index,index_window,blue_window,red_window


# ==========================================================
# LOAD GALAXY DATA
# ==========================================================

def load_galaxies():

    galaxies = {}

    files = sorted(glob.glob(os.path.join(OBS_EW_DIR,"*.csv")))

    if len(files) == 0:
        raise RuntimeError("No EW files found")

    for f in files:

        name = os.path.basename(f).replace(".csv","")

        df = pd.read_csv(f)

        galaxies[name] = {
            "EW": df["EW_median_A"].values,
            "sigma": df["EW_sigma_A"].values
        }

        print("Loaded galaxy:",name)

    return galaxies


# ==========================================================
# BUILD MODEL GRID (Z, AGE)
# ==========================================================

def build_model_grid(grid,index,index_window,blue_window,red_window):

    Zvals = np.array(grid.metallicity)
    agevals = np.array(grid.log10age)

    model = np.zeros((len(index),len(Zvals),len(agevals)))

    for i,Z in enumerate(Zvals):

        for j,age in enumerate(agevals):

            lam,lnu = etc.set_spectra(grid, Z, age)

            lam_obs,flux_clean,flux_noisy,snr = etc.apply_g140m_noise(
                lam,lnu,6.14
            )

            mask = ~np.isnan(flux_clean)

            sed = Sed(
                lam=lam_obs[mask]*angstrom,
                lnu=flux_clean[mask]
            )

            for k in range(len(index)):

                feat = np.array(index_window[k])*angstrom
                blue = np.array(blue_window[k])*angstrom
                red  = np.array(red_window[k])*angstrom

                model[k,i,j] = sed.measure_index(feat,blue,red)

        print("Processed Z =",Z)
        
        print("EW dynamic range per index")

        for i in range(model.shape[0]):
            print(i, np.min(model[i]), np.max(model[i]))

    return model,Zvals,agevals


# ==========================================================
# LOAD OR BUILD MODEL GRID
# ==========================================================

def get_model_grid():

    index,index_window,blue_window,red_window = get_uv_indices()

    print("Building EW model grid")

    grid = Grid(GRID_NAME,grid_dir=GRID_DIR)

    model,Zvals,agevals = build_model_grid(
        grid,index,index_window,blue_window,red_window
    )

    return model, Zvals, agevals


# ==========================================================
# BUILD INTERPOLATORS
# ==========================================================

def build_interpolators(model,Zvals,agevals):

    interpolators = []

    for i in range(model.shape[0]):

        interp = RegularGridInterpolator(
            (Zvals,agevals),
            model[i],
            bounds_error=False,
            fill_value=np.nan
        )

        interpolators.append(interp)

    return interpolators


# ==========================================================
# COVARIANCE MATRIX
# ==========================================================

def build_covariance(obs_sigma):

    C = np.diag(obs_sigma**2)

    return C,np.linalg.inv(C)


# ==========================================================
# LIKELIHOOD
# ==========================================================

def log_likelihood(Z,age,obs,Cinv,interpolators):

    model = np.array([
        interp((Z,age))
        for interp in interpolators
    ])

    valid = np.isfinite(model)

    if not np.any(valid):
        return -np.inf

    delta = obs[valid] - model[valid]

    Cinv_sub = Cinv[np.ix_(valid,valid)]

    chi2 = delta.T @ Cinv_sub @ delta

    return -0.5 * chi2


# ==========================================================
# HIERARCHICAL POSTERIOR
# ==========================================================

def log_posterior(theta, galaxies, interpolators, Zvals, agevals):

    mu_Z = theta[0]
    sigma_Z = theta[1]

    n = len(galaxies)

    Z = theta[2:2+n]
    age = theta[2+n:]

    lp = 0.0

    # --------------------------------------------------
    # PRIORS ON POPULATION PARAMETERS
    # --------------------------------------------------

    # metallicity mean prior (log space)
    if not (-10 < mu_Z < 0):
        return -np.inf

    # metallicity scatter prior
    if not (0.01 < sigma_Z < 1):
        return -np.inf

    # weak prior contribution (optional)
    lp += -np.log(sigma_Z)

    # --------------------------------------------------
    # LOOP OVER GALAXIES
    # --------------------------------------------------

    for i, g in enumerate(galaxies):

        Zg = Z[i]
        ageg = age[i]

        # ----------------------------------------------
        # PHYSICAL PRIORS
        # ----------------------------------------------

        if not (Zvals.min() < Zg < Zvals.max()):
            return -np.inf

        if ageg < 6 or ageg > 8.5:
            return -np.inf

        # ----------------------------------------------
        # HIERARCHICAL PRIOR
        # ----------------------------------------------

        lp += -0.5 * ((np.log(Zg) - mu_Z) / sigma_Z)**2

        # ----------------------------------------------
        # LIKELIHOOD
        # ----------------------------------------------

        obs = galaxies[g]["EW"]
        sigma = galaxies[g]["sigma"]

        C, Cinv = build_covariance(sigma)

        ll = log_likelihood(
            Zg,
            ageg,
            obs,
            Cinv,
            interpolators
        )

        if not np.isfinite(ll):
            return -np.inf

        lp += ll

    return lp

# ==========================================================
# RUN MCMC
# ==========================================================

def run_sampler(galaxies, interpolators, Zvals, agevals):

    rng = np.random.default_rng(RNG_SEED)

    n = len(galaxies)
    ndim = 2 + 2*n

    # Determine reasonable log(Z) range for initialization
    logZ_min = np.log(max(Zvals.min(), 1e-4))  # avoid extreme tiny Z
    logZ_max = np.log(Zvals.max())

    # Age limits
    age_min, age_max = agevals.min(), agevals.max()

    p0 = []

    for i in range(N_WALKERS):

        # Population parameters (mu_Z, sigma_Z)
        mu_Z = rng.uniform(logZ_min + 0.5, logZ_max - 0.5)
        sigma_Z = rng.uniform(0.1, 1.0)  # slightly narrower range for stability

        # Galaxy metallicities in log-space
        Z_init = np.exp(rng.uniform(logZ_min + 0.1, logZ_max - 0.1, size=n))

        # Galaxy ages
        age_init = rng.uniform(age_min, age_max, size=n)

        p = np.concatenate([
            [mu_Z, sigma_Z],
            Z_init,
            age_init
        ])

        p0.append(p)

    p0 = np.array(p0)

    # Debug prints
    print("\nINITIAL WALKER CHECK")
    print("p0 shape:", p0.shape)
    print("mu_Z range:", np.min(p0[:,0]), np.max(p0[:,0]))
    print("sigma_Z range:", np.min(p0[:,1]), np.max(p0[:,1]))
    print("Z init range:", np.min(p0[:,2:2+n]), np.max(p0[:,2:2+n]))
    print("age init range:", np.min(p0[:,2+n:]), np.max(p0[:,2+n:]))

    sampler = emcee.EnsembleSampler(
        N_WALKERS,
        ndim,
        log_posterior,
        args=(galaxies, interpolators, Zvals, agevals)
    )

    sampler.run_mcmc(p0, N_STEPS, progress=True)

    return sampler


# ==========================================================
# MAIN
# ==========================================================

def main():

    index,_,_,_ = get_uv_indices()

    galaxies = load_galaxies()

    model,Zvals,agevals = get_model_grid()
    
    print("before model shape:", np.shape(model))

    interpolators = build_interpolators(model,Zvals,agevals)

    sampler = run_sampler(galaxies,interpolators,Zvals,agevals)

    samples = sampler.get_chain(discard=4000,flat=True)

    print("\n==============================")
    print("POSTERIOR SUMMARY")
    print("==============================")

    n = len(galaxies)

    # --------------------------------
    # Population parameters
    # --------------------------------

    mu_Z = samples[:,0]
    sigma_Z = samples[:,1]

    print("\nPopulation parameters")

    print("mu_Z (log metallicity mean):")
    print("  median =", np.median(mu_Z))
    print("  16-84  =", np.percentile(mu_Z,[16,84]))

    print("\nsigma_Z (metallicity scatter):")
    print("  median =", np.median(sigma_Z))
    print("  16-84  =", np.percentile(sigma_Z,[16,84]))

    print("\nTypical metallicity from mu_Z:")
    print("  Z ≈", np.exp(np.median(mu_Z)))

    # --------------------------------
    # Individual galaxy metallicities
    # --------------------------------

    print("\nGalaxy metallicities")

    galaxy_names = list(galaxies.keys())

    for i,g in enumerate(galaxy_names):

        Z_samples = samples[:,2+i]

        Z_med = np.median(Z_samples)
        Z16, Z84 = np.percentile(Z_samples,[16,84])

        print(f"\n{g}")
        print(f"  Z median = {Z_med}")
        print(f"  Z 16-84  = [{Z16}, {Z84}]")

    # --------------------------------
    # Individual galaxy ages
    # --------------------------------

    print("\nGalaxy stellar ages (log10 years)")

    for i,g in enumerate(galaxy_names):

        age_samples = samples[:,2+n+i]

        age_med = np.median(age_samples)
        age16, age84 = np.percentile(age_samples,[16,84])

        print(f"\n{g}")
        print(f"  log(age) median = {age_med}")
        print(f"  log(age) 16-84  = [{age16}, {age84}]")

    # --------------------------------
    # MCMC diagnostics
    # --------------------------------

    print("\nMCMC diagnostics")

    print("Mean acceptance fraction:",
          np.mean(sampler.acceptance_fraction))

    print("Total posterior samples:",
          samples.shape[0])

    print("Number of parameters:",
          samples.shape[1])
          
    
    print("\nINTERPOLATOR TEST")

    test_Z = np.median(Zvals)
    test_age = np.median(agevals)

    vals = [interp((test_Z,test_age)) for interp in interpolators]

    print("test Z:", test_Z)
    print("test age:", test_age)
    print("model EW predictions:", vals)
    
    print("\nGALAXY DATA CHECK")

    for g in galaxies:
        print(g)
        print("EW:", galaxies[g]["EW"])
        print("sigma:", galaxies[g]["sigma"])
        
    if np.any(~np.isfinite(model)):
        print("WARNING: model returned NaN")
        print("Z:", Z)
        print("age:", age)
        

    print("==============================\n")
    
    chains = sampler.get_chain()

    plt.plot(chains[:,:,0], alpha=0.3)
    plt.title("Trace plot: mu_Z")
    plt.savefig("trace_param_mcmc.png")
    
    chains = sampler.get_chain()

    print("\nCHAIN CHECK")

    print("chain shape:", chains.shape)

    print("mu_Z sampled range:",
          chains[:,:,0].min(),
          chains[:,:,0].max())

    print("sigma_Z sampled range:",
          chains[:,:,1].min(),
          chains[:,:,1].max())
          
    print("\nLIKELIHOOD TEST")

    theta_test = samples[np.random.randint(len(samples))]

    print("Random posterior sample:", theta_test)

    print("Posterior value:",
          log_posterior(theta_test,
                        galaxies,
                        interpolators,
                        Zvals,
                        agevals))
                        
    print("\nMODEL VS DATA CHECK")

    galaxy_names = list(galaxies.keys())

    for i,g in enumerate(galaxy_names):

        Z_med = np.median(samples[:,2+i])
        age_med = np.median(samples[:,2+len(galaxies)+i])

        model_pred = np.array([
            interp((Z_med,age_med))
            for interp in interpolators
        ])

        print("\nGalaxy:",g)
        print("Observed EW:", galaxies[g]["EW"])
        print("Model EW:", model_pred)
    
    print("after model shape:", np.shape(model))
    
    plt.scatter(samples[:,2], samples[:,2+len(galaxies)], s=1)
    plt.xlabel("Z")
    plt.ylabel("log age")
    plt.savefig('galaxy_sample_test.png')

    plt.imshow(model[0], aspect='auto')
    plt.xlabel("age index")
    plt.ylabel("metallicity index")
    plt.title("EW index 0 grid")
    plt.colorbar()
    plt.savefig('MCMC_Z_Test.png')
    
    output_file = "metallicity_results.csv"

    print("\nWriting CSV output:", output_file)

    with open(output_file, "w", newline="") as f:

        writer = csv.writer(f)

        n_index = len(index)

        # Header
        header = ["ID", "Z", "Age", "Z_error"]

        header += [f"EW{i}" for i in range(n_index)]
        header += [f"EWerr{i}" for i in range(n_index)]

        # Add population-level parameters
        header += ["mu_Z", "sigma_Z", "Age_error_low", "Age_error_high"]

        writer.writerow(header)

        galaxy_names = list(galaxies.keys())

        for i, g in enumerate(galaxy_names):

            # Extract numeric ID
            galaxy_id = re.findall(r"\d+", g)[-1]
    
            # Posterior samples
            Z_samples = samples[:, 2 + i]
            age_samples = samples[:, 2 + len(galaxies) + i]
    
            Z_med = np.median(Z_samples)
            age_med = np.median(age_samples)
    
            Z16, Z84 = np.percentile(Z_samples, [16, 84])
            Z_err = 0.5 * (Z84 - Z16)
    
            # Observed EW and errors
            EW = galaxies[g]["EW"]
            EW_err = galaxies[g]["sigma"]

            # Population-level parameters
            mu_Z_med = np.median(samples[:, 0])
            sigma_Z_med = np.median(samples[:, 1])
            age16, age84 = np.percentile(age_samples, [16, 84])

            row = [galaxy_id, Z_med, age_med, Z_err]

            row += list(EW)
            row += list(EW_err)

            row += [mu_Z_med, sigma_Z_med, age16, age84]

            writer.writerow(row)

    print("CSV writing complete.")
         
         

if __name__ == "__main__":

    main()

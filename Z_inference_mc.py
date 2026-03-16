#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.integrate import simpson
import csv

# ===============================
# USER SETTINGS
# ===============================
# COSMA_OBS_SPECTRUM_FITS = "/cosma8/data/dp004/dc-sant4/data/DJA_spectra/gdn-fujimoto-v4_g140m-f070lp_4762_14050.spec.fits"

OBS_SPECTRUM_FITS = "/users/csant/data/scripts/DJA_spectra/gdn-fujimoto-v4_g140m-f070lp_4762_16499.spec.fits"

REDSHIFT = 6.3677
# 9547 : z = 6.1689 : s/n = 2.2 : grade = 3 RUN
# 78773: z = 5.1885 : s/n = 2.0 : grade = 3 RUN
# aurora: z = 5.1885 : s/n = 3.2 : grade = 3 RUN
# bluejay: z = 6.9894 : s/n = 2.2 : grade = 3 RUN
# fujimoto 20756: z = 5.6143 : s/n = 9.3 : grade = 1 RUN
# fujimoto 17458: z = 5.5673 : s/n = 27.2 : grade = 1 RUN
# fujimoto 14050: z = 6.0154 : s/n = 35.7 : grade = 1 RUN
# fujimoto 13956: z = 6.1195 : s/n = 28.2 : grade = 1 RUN
# fujimoto 15523: z = 5.4373 : s/n = 6.6 : grade = 1
# fujimoto 20987: z = 5.4317 : s/n = 4.7 : grade = 1
# fujimoto 37393 gnz7q: z = 7.2059 : s/n = 4.6 : grade = 2
# JADES 58441: z = 5.9716 : s/n = 1.9 : grade = 3
# fujimoto 17264: z = 5.2725 : s/n = 1.9 : grade = 3
# udeep 201249: z = 5.0282 : s/n = 1.9 : grade = 3
# fujimoto 16499: z = 6.3677 : s/n = 2.0 : grade = 1


N_MC = 10000 # number of Monte Carlo realizations

# ===============================
# UV Indices & Windows
# ===============================
INDEX_LIST = [1370, 1400, 1425, 1460, 1501, 1533, 1550, 1719, 1853]

INDEX_WINDOW = [
    [1360, 1380], [1385, 1410], [1413, 1435],
    [1450, 1470], [1496, 1506], [1530, 1537],
    [1530, 1560], [1705, 1729], [1838, 1858]
]

BLUE_WINDOW = [
    [1345, 1354], [1345, 1354], [1345, 1354],
    [1436, 1447], [1482, 1491], [1482, 1491],
    [1482, 1491], [1675, 1684], [1797, 1807]
]

RED_WINDOW = [
    [1436, 1447], [1436, 1447], [1436, 1447],
    [1482, 1491], [1583, 1593], [1583, 1593],
    [1583, 1593], [1751, 1761], [1871, 1883]
]


# ===============================
# EW Measurement Function
# ===============================
def measure_EW(lam, flux, feature, blue, red):
    """
    Measure absorption EW using the same workflow as Synthesizer's measure_index.

    Parameters
    ----------
    lam : array
        Rest-frame wavelength array (Å)
    flux : array
        Flux density array (same units throughout)
    feature, blue, red : [λmin, λmax]
        Feature and continuum windows in Å

    Returns
    -------
    EW : float
        Equivalent width in Å, or NaN if not measurable
    """

    lam = np.asarray(lam)
    flux = np.asarray(flux)

    # --- masks ---
    mask_feat = (lam > feature[0]) & (lam < feature[1]) & np.isfinite(flux)
    mask_blue = (lam > blue[0]) & (lam < blue[1]) & np.isfinite(flux)
    mask_red  = (lam > red[0])  & (lam < red[1])  & np.isfinite(flux)

    # Require minimum sampling
    if mask_feat.sum() < 3 or mask_blue.sum() < 2 or mask_red.sum() < 2:
        return np.nan

    # --- continuum anchor points ---
    lnu_blue = np.mean(flux[mask_blue])
    lnu_red  = np.mean(flux[mask_red])
    mean_blue = np.mean(blue)
    mean_red  = np.mean(red)

    if not np.isfinite(lnu_blue) or not np.isfinite(lnu_red):
        return np.nan

    # --- linear continuum fit ---
    continuum_fit = np.polyfit(
        [mean_blue, mean_red],
        [lnu_blue, lnu_red],
        1
    )

    # --- continuum over feature window ---
    feature_lam = lam[mask_feat]
    continuum = continuum_fit[0] * feature_lam + continuum_fit[1]

    if np.any(continuum <= 0) or not np.all(np.isfinite(continuum)):
        return np.nan

    # --- continuum-normalised absorption ---
    feature_flux = flux[mask_feat]
    feature_norm = -(feature_flux - continuum) / continuum

    # --- integrate EW ---
    EW = np.trapz(feature_norm, x=feature_lam)

    return EW

# ===============================
# Load Observed Spectrum
# ===============================
hdul = fits.open(OBS_SPECTRUM_FITS)
spec = hdul[1].data

lam_obs  = spec["wave"] * 1e4  # µm → Å
flux_obs = spec["flux"]        # µJy
err_obs  = spec["err"]         # µJy

hdul.close()

print(f"Loaded spectrum: {lam_obs.min():.1f} – {lam_obs.max():.1f} Å, flux ~ {np.nanmin(flux_obs):.2e} – {np.nanmax(flux_obs):.2e} µJy")

# ===============================
# Rest-frame conversion
# ===============================
lam_rest  = lam_obs / (1 + REDSHIFT)
flux_rest = flux_obs * (1 + REDSHIFT)
err_rest  = err_obs * (1 + REDSHIFT)

for j, idx in enumerate(INDEX_LIST):
    mask_feat = (lam_rest > INDEX_WINDOW[j][0]) & (lam_rest < INDEX_WINDOW[j][1])
    mask_blue = (lam_rest > BLUE_WINDOW[j][0]) & (lam_rest < BLUE_WINDOW[j][1])
    mask_red  = (lam_rest > RED_WINDOW[j][0]) & (lam_rest < RED_WINDOW[j][1])

    print(f"{idx} Å:",
          f"N_feat={np.sum(mask_feat)}",
          f"N_feat_finite={np.sum(mask_feat & np.isfinite(flux_rest))}",
          f"N_blue_finite={np.sum(mask_blue & np.isfinite(flux_rest))}",
          f"N_red_finite={np.sum(mask_red & np.isfinite(flux_rest))}")

# ===============================
# Monte Carlo EW Estimation
# ===============================
rng = np.random.default_rng(42)
EW_MC = np.full((N_MC, len(INDEX_LIST)), np.nan)

for i in range(N_MC):
    flux_i = flux_rest + rng.normal(0, err_rest)
    for j, idx in enumerate(INDEX_LIST):
        EW_MC[i, j] = measure_EW(
            lam_rest, flux_i,
            INDEX_WINDOW[j], BLUE_WINDOW[j], RED_WINDOW[j]
        )

# ===============================
# EW Statistics
# ===============================
EW_med = np.nanmedian(EW_MC, axis=0)
EW_p16 = np.nanpercentile(EW_MC, 16, axis=0)
EW_p84 = np.nanpercentile(EW_MC, 84, axis=0)
EW_err = 0.5 * (EW_p84 - EW_p16)

print("\nObserved EWs (median ± sigma):")
for j, idx in enumerate(INDEX_LIST):
    print(f"{idx} Å : {EW_med[j]:.3f} ± {EW_err[j]:.3f} Å")

# ===============================
# Diagnostic Plots with ±1σ shading
# ===============================
fig, axes = plt.subplots(3, 3, figsize=(10, 8))
plt.rcParams["figure.dpi"] = 600
plt.style.use("matplotlibrc.txt")
axes = axes.flatten()

for j, idx in enumerate(INDEX_LIST):
    ax = axes[j]

    EW_vals = EW_MC[:, j]
    EW_vals = EW_vals[np.isfinite(EW_vals)]

    label = f"UV_{idx}" if idx in [1501, 1719] else f"F{idx}"

    if len(EW_vals) == 0 or not np.isfinite(EW_med[j]):
        # No valid EW measurements
        ax.text(
            0.5, 0.5,
            f"{label}\nInsufficient data: index could not be measured",
            ha="center", va="center",
            fontsize=8, transform=ax.transAxes
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Valid EW distribution
        ax.hist(EW_vals, bins=40, histtype="step", color="black")

        ax.axvline(EW_med[j], color="red", lw=1, label="Median EW")

        sigma_low  = EW_med[j] - EW_err[j]
        sigma_high = EW_med[j] + EW_err[j]
        ax.axvspan(sigma_low, sigma_high, color="red", alpha=0.2, label="±1σ")

        title_str = f"{label}\nEW: {EW_med[j]:.3f} ± {EW_err[j]:.2f} Å"
        ax.set_title(title_str, fontsize=8, y=0.6, x=0.25, zorder=10)

    # Axis labels
    if j == 7:
        ax.set_xlabel("EW (Å)", fontsize=8)
    if j == 3:
        ax.set_ylabel("Number of realizations", fontsize=8)

    ax.tick_params(labelsize=7)
    if j == 0 and len(EW_vals) > 0:
        ax.legend(fontsize=6)


plt.tight_layout()
plt.savefig("EW_MC_distributions.png")

# ===============================
# Sanity Check Function
# ===============================

def sanity_check(lam_obs, flux_obs, lam_rest, flux_rest,
                 EW_MC, EW_med,
                 INDEX_LIST, INDEX_WINDOW, BLUE_WINDOW, RED_WINDOW,
                 csv_out="observed_EWs_sanity.csv"):

    print("\n=== SANITY CHECK ===")
    print(f"Observed λ range: {lam_obs.min():.1f} – {lam_obs.max():.1f} Å")
    print(f"Rest-frame λ range: {lam_rest.min():.1f} – {lam_rest.max():.1f} Å")
    print(f"Flux range: {np.nanmin(flux_obs):.2e} – {np.nanmax(flux_obs):.2e} µJy")

    test_indices = [1501, 1719]
    for idx in test_indices:
        j = INDEX_LIST.index(idx)
        ew_test = measure_EW(
            lam_rest, flux_rest,
            INDEX_WINDOW[j], BLUE_WINDOW[j], RED_WINDOW[j]
        )
        print(f"EW check for {idx} Å: {ew_test:.3f} Å")

    print("\nEW median ± sigma for all indices:")

    rows = []
    for j, idx in enumerate(INDEX_LIST):
        p16 = np.nanpercentile(EW_MC[:, j], 16)
        p84 = np.nanpercentile(EW_MC[:, j], 84)
        ew_sigma = 0.5 * (p84 - p16)

        print(f"{idx} Å : {EW_med[j]:.3f} ± {ew_sigma:.3f} Å")

        rows.append({
            "index_A": idx,
            "EW_median_A": EW_med[j],
            "EW_sigma_A": ew_sigma,
            "EW_p16_A": p16,
            "EW_p84_A": p84,
            "EW_MC_N": EW_MC.shape[0]
        })

    # -------------------------------------------------
    # SAVE TO CSV
    # -------------------------------------------------
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved EW sanity-check table to: {csv_out}")
    print("=== END OF SANITY CHECK ===\n")


sanity_check(lam_obs, flux_obs, lam_rest, flux_rest, EW_MC, EW_med,
             INDEX_LIST, INDEX_WINDOW, BLUE_WINDOW, RED_WINDOW)


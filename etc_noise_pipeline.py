import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18
from scipy.ndimage import gaussian_filter1d
from spectres import spectres


from unyt import Msun, Myr, angstrom
from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy

import astropy.constants as const
import astropy.units as u

REDSHIFT = 5.0                # Redshift of synthetic galaxy
R_G140M = 1000.0              # Spectral resolving power (lambda/dlambda)
PIXEL_WIDTH_A = 1.0           # Detector sampling in Angstrom
REBIN_WIDTH = 6.0             # Rebin width in Angstrom (set to 1 to disable)
CORR_FWHM_PIX = 2.0           # Noise correlation length in pixels
RNG_SEED = 42                 # Random seed for reproducibility

ETC_PATH = "/users/csant/data/scripts/g140m_etc_noise_profile/lineplot"


def set_spectra(grid, Z, age):
    """
    Generate a synthetic stellar population spectrum using BPASS
    through the synthesizer framework.

    Returns
    -------
    lam : array-like
        Rest-frame wavelength array (Angstrom)
    lnu : array-like
        Spectral luminosity density (erg/s/Hz)
    """

    
    stellar_mass = 1e8 * Msun

    # Emission model
    model = IncidentEmission(grid)

    # Constant metallicity distribution
    metal_dist = ZDist.DeltaConstant(metallicity=Z)

    # Constant star formation history up to 100 Myr
    sfh = SFH.Constant(max_age=100 * Myr)

    # Build 2D star formation + metallicity history
    sfzh = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=stellar_mass,
    )

    # Create galaxy object and generate spectrum
    galaxy = Galaxy(sfzh)
    galaxy.stars.get_spectra(model)

    sed = galaxy.stars.spectra["incident"]

    return sed.lam, sed.lnu



def load_etc():
    """
    Load ETC output products used to reconstruct sensitivity,
    flux scaling, and noise properties.

    Returns
    -------
    etc_lam : array
        Wavelength grid (Angstrom)
    etc_flux_total : array
        Total extracted flux
    etc_flux_app : array
        Aperture extracted flux
    etc_noise : array
        Extracted noise spectrum
    target_flux : array
        Target model flux used in ETC
    """

    with fits.open(f"{ETC_PATH}/lineplot_total_flux.fits") as f:
        etc_lam = f[1].data['WAVELENGTH'] * 1e4
        etc_flux_total = f[1].data['total_flux']

    with fits.open(f"{ETC_PATH}/lineplot_extracted_flux.fits") as f:
        etc_flux_app = f[1].data['extracted_flux']

    with fits.open(f"{ETC_PATH}/lineplot_extracted_noise.fits") as f:
        etc_noise = f[1].data['extracted_noise']

    with fits.open(f"{ETC_PATH}/lineplot_target.fits") as f:
        target_flux = f[1].data["target"]

    return etc_lam, etc_flux_total, etc_flux_app, etc_noise, target_flux



def apply_redshift(etc_lam):
    """
    Convert ETC wavelength grid to observed-frame rest scaling.

    Parameters
    ----------
    etc_lam : array
        Wavelength array from ETC

    Returns
    -------
    lam_obs : array
        Observed-frame wavelength grid
    """
    return etc_lam / (1 + REDSHIFT)



def convert_flambda(lam, lnu):
    """
    Convert L_nu (erg/s/Hz) to observed-frame F_lambda
    (erg/s/cm^2/Angstrom).

    Uses Planck18 cosmology for luminosity distance.
    """

    c = 2.99792458e18  # Speed of light in Angstrom/s
    DL = Planck18.luminosity_distance(REDSHIFT).to(u.cm).value

    return (lnu * c) / (4 * np.pi * DL**2 * lam**2)



def compute_sensitivity(total_flux, flux_obs, etc_lam, lam_obs):
    """
    Derive effective sensitivity from ETC total flux and
    model flux by interpolation.
    """
    return np.interp(lam_obs, etc_lam, total_flux / flux_obs)



def flux_to_counts(flux_obs, lam_obs, A_tel_cm2, throughput):
    """
    Convert flux density to detector electron counts per second.
    """

    flux = flux_obs * u.erg / (u.s * u.cm**2 * u.angstrom)
    lam = lam_obs * u.angstrom

    photon_flux = (flux * lam / (const.h * const.c)).to(
        1/(u.s * u.cm**2 * u.angstrom)
    )

    photons_on_det = photon_flux * A_tel_cm2 * u.cm**2
    counts = photons_on_det * throughput

    return counts.value



def correlated_noise(sigma, fwhm_pix, rng):
    """
    Generate correlated Gaussian noise using convolution
    with a Gaussian kernel of specified FWHM in pixels.
    """

    white = rng.normal(0, 1, size=len(sigma))

    if fwhm_pix > 0:
        sigma_corr = fwhm_pix / 2.355
        white = gaussian_filter1d(white, sigma_corr)

    white /= np.std(white)

    return white * sigma



def apply_g140m_noise(lam_sed, lnu_sed, signal_ratio):
    """
    Apply JWST/NIRSpec G140M-like response including:

    - Redshift transformation
    - Sensitivity calibration via ETC
    - Line spread function convolution
    - Gaussian detector noise
    - Spectral rebinning

    Returns
    -------
    lam_rebin : array
        Observed wavelength grid after rebinning
    flux_clean : array
        Clean (noiseless) flux spectrum
    flux_noisy : array
        Noisy flux spectrum
    """

    rng = np.random.default_rng(RNG_SEED)

    etc_lam, etc_flux_total, etc_flux_app, etc_flux_noise, etc_flux_target = load_etc()

    lam_obs = apply_redshift(etc_lam)

    # Restrict synthetic SED to ETC wavelength range
    mask = (lam_sed >= min(lam_obs)) & (lam_sed <= max(lam_obs))
    lam_sed_trimmed = lam_sed[mask]
    flux_sed_trimmed = lnu_sed[mask]

    flux_sed_on_etc = np.interp(
        lam_obs * angstrom, lam_sed_trimmed, flux_sed_trimmed
    )

    # Convert to observed flux
    flux_obs = convert_flambda(lam_obs, flux_sed_on_etc)

    # Reconstruct sensitivity from ETC outputs
    sensitivity = compute_sensitivity(
        etc_flux_total, flux_obs, etc_lam, lam_obs
    )

    detector_counts = flux_obs * sensitivity

    # Compute spectral resolution element width
    delta_lambda = np.median(np.diff(lam_obs))
    fwhm_lambda = lam_obs / R_G140M
    sigma_lambda = fwhm_lambda / 2.355
    sigma_pixels = sigma_lambda / delta_lambda
    sigma_pix = np.nanmedian(sigma_pixels)

    # Apply line spread function convolution
    counts_lsf = gaussian_filter1d(detector_counts, sigma_pix)

    # Add Gaussian noise
    sigma_counts = np.interp(lam_obs, etc_lam, etc_flux_noise) / signal_ratio
    noise = rng.normal(0, 1, len(counts_lsf))
    counts_noisy = counts_lsf + noise * sigma_counts

    # Optional rebinning
    if REBIN_WIDTH > 1:
        lam_rebin = np.arange(lam_obs.min(), lam_obs.max(), REBIN_WIDTH)
        counts_clean_rebin = spectres(lam_rebin, lam_obs, counts_lsf)
        counts_noisy_rebin = spectres(lam_rebin, lam_obs, counts_noisy)
    else:
        lam_rebin = lam_obs
        counts_clean_rebin = counts_lsf
        counts_noisy_rebin = counts_noisy

    # Convert back to flux space
    sens_interp = np.interp(lam_rebin, etc_lam, sensitivity)
    flux_clean = counts_clean_rebin / sens_interp
    flux_noisy = counts_noisy_rebin / sens_interp
    
    median_snr = np.nanmedian(counts_lsf / sigma_counts)

    return lam_rebin, flux_clean, flux_noisy, median_snr



if __name__ == "__main__":

    # Generate synthetic SED
    
    # Load stellar population grid
    grid = Grid(
        "bpass-2.3-bin_bpl-0.1,1.0,300.0-1.3,2.35_alpha0.0.hdf5",
        grid_dir="/users/csant/data/grids",
    )

    Z = grid.metallicity
    lam, lnu = set_spectra(grid, Z, grid.log10age)
    signal_ratio = 1
    

    # Apply instrument + noise model
    lam_obs, flux_clean, flux_noisy = apply_g140m_noise(lam, lnu, signal_ratio)

import numpy as np
from unyt import angstrom, kpc, Mpc, Msun, Gyr, km, s, yr, arcsecond, Myr
import h5py
from astropy.cosmology import Planck15 as cosmo
import time
import argparse
import os

from synthesizer import check_openmp
print('OpenMP enabled:', check_openmp() )

from mpi4py import MPI as mpi
import multiprocessing as mp

from synthesizer.emission_models import IntrinsicEmission, IncidentEmission, ReprocessedEmission
from synthesizer.grid import Grid
from synthesizer.instruments import UVJ, FilterCollection, Instrument
from synthesizer.pipeline import Pipeline
from synthesizer.load_data.load_flares import load_FLARES
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.particle import Galaxy
from synthesizer.particle import Stars, Gas, BlackHoles
from synthesizer.kernel_functions import Kernel
from synthesizer.emissions import Sed

def _print(*args, **kwargs):
    """Overload print with rank info."""
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"[{str(rank).zfill(len(str(size)) + 1)}]: ", end="")
    print(*args, **kwargs)
    
    
def _get_galaxy(gal_ind, master_file_path, reg, snap, z):
    """
    Get a galaxy from the master file.

    Args:
        gal_ind (int): The index of the galaxy to get.
        master_file_path (str): The path to the master file.
        reg (str): The region to use.
        snap (str): The snapshot to use.
        z (float): The redshift of the snapshot.
    """
    # Get the galaxy data we need from the master file
    with h5py.File(master_file_path, "r") as hdf:
        reg_grp = hdf[reg]
        snap_grp = reg_grp[snap]
        gal_grp = snap_grp["Galaxy"]
        part_grp = snap_grp["Particle"]

        # Get the group and subgrp ids
        group_id = gal_grp["GroupNumber"][gal_ind]
        subgrp_id = gal_grp["SubGroupNumber"][gal_ind]

        # Get this galaxy's beginning and ending indices for stars
        s_len = gal_grp["S_Length"][...]
        start = np.sum(s_len[:gal_ind])
        end = np.sum(s_len[: gal_ind + 1])

        # Get this galaxy's beginning and ending indices for gas
        g_len = gal_grp["G_Length"][...]
        start_gas = np.sum(g_len[:gal_ind])
        end_gas = np.sum(g_len[: gal_ind + 1])

        # Get this galaxy's beginning and ending indices for black holes
        bh_len = gal_grp["BH_Length"][...]
        start_bh = np.sum(bh_len[:gal_ind])
        end_bh = np.sum(bh_len[: gal_ind + 1])

        # Get the star data
        star_pos = part_grp["S_Coordinates"][:, start:end].T / (1 + z) * Mpc
        star_mass = part_grp["S_Mass"][start:end] * Msun * 10**10
        star_init_mass = part_grp["S_MassInitial"][start:end] * Msun * 10**10
        star_age = part_grp["S_Age"][start:end] * Gyr
        star_met = part_grp["S_Z_smooth"][start:end]
        star_sml = part_grp["S_sml"][start:end] * Mpc
        star_vel = part_grp["S_Vel"][:, start:end].T * km / s

        # Get the gas data
        gas_pos = (
            part_grp["G_Coordinates"][:, start_gas:end_gas].T / (1 + z) * Mpc
        )
        gas_mass = part_grp["G_Mass"][start_gas:end_gas] * Msun * 10**10
        gas_met = part_grp["G_Z_smooth"][start_gas:end_gas]
        gas_sml = part_grp["G_sml"][start_gas:end_gas] * Mpc

        # Get the black hole data
        bh_pos = (
            part_grp["BH_Coordinates"][:, start_bh:end_bh].T / (1 + z) * Mpc
        )
        bh_mass = part_grp["BH_Mass"][start_bh:end_bh] * Msun * 10**10
        bh_mdot = (
            part_grp["BH_Mdot"][start_bh:end_bh]
            * (
                6.445909132449984 * 10**23
            )  # Unit conversion issue, need this
            * Msun
            / yr
        )

        # Get the centre of potential
        centre = gal_grp["COP"][:].T[gal_ind, :] / (1 + z) * Mpc

        # Compute the angular radii of each star in arcseconds
        radii = (np.linalg.norm(star_pos - centre, axis=1)).to("kpc")
        star_ang_rad = (
            radii.value * cosmo.arcsec_per_kpc_proper(z).value * arcsecond
        )

    # Define a mask to get a 30 kpc aperture
    mask = radii < 30 * kpc

    # Early exist if there are fewer than 100 stars
    if np.sum(mask) < 100:
        return None

    gal = Galaxy(
        name=f"{reg}_{snap}_{gal_ind}_{group_id}_{subgrp_id}",
        redshift=z,
        stars=Stars(
            initial_masses=star_init_mass[mask],
            current_masses=star_mass[mask],
            ages=star_age[mask],
            metallicities=star_met[mask],
            redshift=z,
            coordinates=star_pos[mask, :],
            velocities=star_vel[mask, :],
            smoothing_lengths=star_sml[mask],
            centre=centre,
            young_tau_v=star_met[mask] / 0.01,
            angular_radii=star_ang_rad[mask],
            radii=radii[mask].value,
        ),
        gas=Gas(
            masses=gas_mass,
            metallicities=gas_met,
            redshift=z,
            coordinates=gas_pos,
            smoothing_lengths=gas_sml,
            centre=centre,
        ),
        black_holes=BlackHoles(
            masses=bh_mass,
            accretion_rates=bh_mdot,
            coordinates=bh_pos,
            redshift=z,
            centre=centre,
        ),
    )

    # Calculate the DTM, we'll need it later
    gal.calculate_dust_to_metal_vijayan19()

    # Compute what we can compute out the gate and attach it to the galaxy
    # for later use
    gal.gas.half_mass_radius = gal.gas.get_half_mass_radius()
    gal.gas.mass_radii = {
        0.2: gal.gas.get_attr_radius("masses", frac=0.2),
        0.8: gal.gas.get_attr_radius("masses", frac=0.8),
    }
    gal.gas.half_dust_radius = gal.gas.get_attr_radius("dust_masses")
    gal.stars.half_mass_radius = gal.stars.get_half_mass_radius()
    gal.stars.mass_radii = {
        0.2: gal.stars.get_attr_radius("current_masses", frac=0.2),
        0.8: gal.stars.get_attr_radius("current_masses", frac=0.8),
    }

    # Calculate the 3D and 1D velocity dispersions
    gal.stars.vel_disp1d = np.array(
        [
            np.std(gal.stars.velocities[:, 0], ddof=0),
            np.std(gal.stars.velocities[:, 1], ddof=0),
            np.std(gal.stars.velocities[:, 2], ddof=0),
        ]
    )
    gal.stars.vel_disp3d = np.std(
        np.sqrt(np.sum(gal.stars.velocities**2, axis=1)), ddof=0
    )

    return gal


def get_flares_galaxies(
    master_file_path,
    region,
    snap,
    nthreads,
    comm,
    rank,
    size,
    kernel,
):
    """
    Get Galaxy objects for FLARES galaxies.

    Args:
        master_file_path (str): The path to the master file.
        region (int): The region to use.
        snap (str): The snapshot to use.
        filter_collection (FilterCollection): The filter collection to use.
        emission_model (StellarEmissionModel): The emission model to use.
    """
    
    # Get the region tag
    reg = str(region).zfill(2)

    # Get redshift from the snapshot tag
    z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))
    #if snap == "008_z007p000":
    #    z = 7.29
    #    _print("Shifted redshift to 7.29 for snapshot 008_z007p000")

    # How many galaxies are there?
    with h5py.File(master_file_path, "r") as hdf:
        reg_grp = hdf[reg]
        snap_grp = reg_grp[snap]
        gal_grp = snap_grp["Galaxy"]
        s_lens = gal_grp["S_Length"][:]
        n_gals = len(s_lens)

    # Early exist if there are no galaxies
    if n_gals == 0:
        return []

    # Randomize the order of galaxies
    np.random.seed(42)
    order = np.random.permutation(n_gals)

    # Distribute galaxies by number of particles
    parts_per_rank = np.zeros(size, dtype=int)
    gals_on_rank = {rank: [] for rank in range(size)}
    for i in order:
        if s_lens[i] < 100:
            continue
        select = np.argmin(parts_per_rank)
        gals_on_rank[select].append(i)
        parts_per_rank[select] += s_lens[i]

    # Prepare the arguments for each galaxy on this rank
    args = [
        (gal_ind, master_file_path, reg, snap, z)
        for gal_ind in gals_on_rank[rank]
    ]

    # Get all the galaxies using multiprocessing
    with mp.Pool(nthreads) as pool:
        galaxies = pool.starmap(_get_galaxy, args)

    # Remove any Nones
    galaxies = [gal for gal in galaxies if gal is not None]
    
    # Loop over galaxies and calculate the optical depths
    # This is needed if attenuation is included in the emission model
    for gal in galaxies:
        if gal.gas.nparticles > 0:
            # stars
            gal.stars.tau_v = gal.get_stellar_los_tau_v(
                kappa=0.0795,
                kernel=kernel,
            )
            # BH
            gal.black_holes.tau_v = gal.get_black_hole_los_tau_v(
                kappa=0.07,
                kernel=kernel,
            )
        else:
            gal.stars.tau_v = np.zeros(gal.stars.nparticles)
            gal.black_holes.tau_v = np.zeros(gal.stars.nparticles)

    return galaxies


def get_webb_filters(filepath):
    """Get the filter collection."""
    
    # Check if the filter collection file already exists
    if os.path.exists(filepath):
        filters = FilterCollection(path=filepath)
    else:

        lam = np.linspace(10**3, 10**5, 1000) * angstrom
        filters = FilterCollection(
            filter_codes=[
                f"JWST/NIRCam.{f}"
                for f in ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
            ],
            new_lam=lam,
        )

        # Write the filter collection
        filters.write_filters(path=filepath)

    return filters


def get_uvj_filters(filepath):
    """Get the filter collection."""
    
    # Check if the filter collection file already exists
    if os.path.exists(filepath):
        filters = FilterCollection(path=filepath)
    else:

        lam = np.linspace(10**3, 10**5, 1000) * angstrom
        filters = UVJ(new_lam=lam)

        # Write the filter collection
        filters.write_filters(path=filepath)

    return filters

def get_UV_slopes(obj):
    """
    Get the UV slope of the galaxy.

    Args:
        obj (Galaxy/Stars/BlackHoles): The object to get the UV slope for.

    Returns:
        float: The UV slope.
    """
    # Dictionary to hold the slopes
    slopes = {}

    # Loop over the spectra
    for spec_type, d in obj.spectra.items():
        slopes[spec_type] = obj.spectra[spec_type].measure_beta(
            window=(1300 * angstrom, 2000 * angstrom)
        )

    return slopes


def get_IR_slopes(obj):
    """
    Get the IR slopes of the galaxy.

    Args:
        obj (Galaxy/Stars/BlackHoles): The object to get the IR slopes for.

    Returns:
        float: The IR slopes.
    """
    # Dictionary to hold the slopes
    slopes = {}

    # Loop over the spectra
    for spec_type, d in obj.spectra.items():
        slopes[spec_type] = obj.spectra[spec_type].measure_beta(
            window=(4400 * angstrom, 7500 * angstrom)
        )

    return slopes

def estimate_uv_weighted_Z(stars):
    t_uv = 1e8  # 100 Myr
    weights = stars.initial_masses * np.exp(-stars.ages / t_uv)
    total_weight = weights.sum()

    if total_weight == 0:
        print("UV weights sum to zero")
        return 0.0  # or np.nan if you'd rather handle it explicitly

    weighted_Z = (stars.metallicities * weights).sum() / total_weight
    return weighted_Z
    
    
def get_equivalent_width(grids, uv_index, index_window, blue_window, red_window, galaxies, model):
    """
    Calculate and return equivalent widths for specified UV indices.

    Returns:
        eqws (list of lists): Outer list over UV indices, inner lists over galaxies.
    """
    # Load grid
    grid = Grid(grids, grid_dir=grid_dir)

    # Define accessors
    metallicity = lambda gal: gal.stars.metallicities
    stellar_mass = lambda gal: gal.stars.current_masses

    # Container for results
    eqw_lib = []

    # Loop over UV spectral indices
    for i, index in enumerate(uv_index):
        eqw = []  # holds EW values for this index

        # Define index windows
        feature = index_window[i] * angstrom
        blue = blue_window[i] * angstrom
        red = red_window[i] * angstrom

        # Loop over galaxies
        for gal in galaxies:
            try:
                ew = measure_equivalent_width(
                    index, feature, blue, red,
                    metallicity(gal), stellar_mass(gal),
                    grid, eqw, gal, model, 0
                )
                eqw.append(ew)
            except Exception as e:
                print(f"Error processing galaxy: {e}")
                eqw.append(np.nan)  # Optional: Fill failed values with NaN

        # Store this index's list of EWs
        eqw_lib.append(eqw)

        # Optional print summary
        if np.any(~np.isnan(eqw)):
            print(f"Mean Equivalent Width [{index}]: {np.nanmean(eqw):.4f}")
        else:
            print(f"No valid equivalent width results for index: {index}")

    return eqw_lib



def measure_equivalent_width(
    index, feature, blue, red, Z, smass, grid, eqw, gal, model, mode
):
    """
    Calculate equivalent width for a specified UV index.

    Args:
        index (int): The UV index for which the equivalent width is calculated.
        Z (float): Metallicity.
        smass (float): Stellar mass.
        grid (Grid): The grid object.
        eqw (float): Initial equivalent width.
        mode (str): Calculation mode.

    Returns:
        float: The calculated equivalent width.

    Raises:
        ValueError: If mode is invalid.
    """

    stellar_mass = smass

    Z_p = {"metallicity": Z}
    metal_dist = ZDist.DeltaConstant(**Z_p)  # constant metallicity
    sfh_p = {"max_age": 100 * Myr}
    sfh = SFH.Constant(**sfh_p)  # constant star formation

    # --- get 2D star formation and metal enrichment history for the
    # given SPS grid. This is (age, Z).
    
    gal.stars.get_spectra(model)

    # --- generate spectra
    if mode == 0:
        sed = gal.stars.spectra["incident"]
    else:
        sed = gal.stars.spectra["reprocessed"]

    return sed.measure_index(
        feature,
        blue,
        red,
    )
    

def set_index():
    """
    A function to define a dictionary of uv indices.

    Each index has a defined absorption window.
    A pseudo-continuum is defined, made up of a blue and red shifted window.

    Returns:
        tuple: A tuple containing the following list:
            - index (int): List of UV indices.
            - index_window (int): List of absorption window bounds.
            - blue_window (int): List of blue shifted window bounds.
            - red_window (int): List of red shifted window bounds.
    """

    index = [1370, 1400, 1425, 1460, 1501, 1533, 1550, 1719, 1853]
    index_window = [
        [1360, 1380],
        [1385, 1410],
        [1413, 1435],
        [1450, 1470],
        [1496, 1506],
        [1530, 1537],
        [1530, 1560],
        [1705, 1729],
        [1838, 1858],
    ]
    blue_window = [
        [1345, 1354],
        [1345, 1354],
        [1345, 1354],
        [1436, 1447],
        [1482, 1491],
        [1482, 1491],
        [1482, 1491],
        [1675, 1684],
        [1797, 1807],
    ]
    red_window = [
        [1436, 1447],
        [1436, 1447],
        [1436, 1447],
        [1482, 1491],
        [1583, 1593],
        [1583, 1593],
        [1583, 1593],
        [1751, 1761],
        [1871, 1883],
    ]

    return index, index_window, blue_window, red_window


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Derive synthetic observations for FLARES."
    )

    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use.",
    )

    # Parse the arguments
    args = parser.parse_args()
    nthreads = args.nthreads
    
    # Get the grid
    grid_dir = '/cosma7/data/dp004/dc-sant4/data/grids'
    grid_name = 'bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c23.01-sps'
    grid = Grid(grid_name, grid_dir=grid_dir)
    
    ages = grid.log10age
    Z = grid.metallicity

    # Emission model
    model = IntrinsicEmission(grid, fesc=0.1)
    model.set_per_particle(True)  # we want per particle emissions
    
    # We can use run the Pipeline for multiple grids by setting:
    #model.set_grid(new_grid, set_all=True)

    # Get the filters
    # Note that if running on a HPC you will need to make these filter files 
    # seperately to this code
    # webb_filters = get_webb_filters("/yourpath/nircam_filters.hdf5")
    # uvj_filters = get_uvj_filters("/yourpath/uvj_filters.hdf5")

    # Instatiate the instruments
    # webb_inst = Instrument("JWST", filters=webb_filters, resolution=0.1 * kpc)
    # uvj_inst = Instrument("UVJ", filters=uvj_filters)
    # instruments = webb_inst + uvj_inst
    # print(instruments)

    # Get the galaxies
    # You need access to COSMA for this
    master_file_path = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"
    region = "01" # region
    snap =  "010_z005p000" # the first snapshot

    # Print n CPUs for reference
    cpuCount = os.cpu_count()
    print("Number of CPUs in the system:", cpuCount)

    # Get MPI info
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get the SPH kernel
    sph_kernel = Kernel()
    kernel = sph_kernel.get_kernel()
        
    # Get the galaxies
    read_start = time.time()
    galaxies = get_flares_galaxies(
        master_file_path,
        region,
        snap,
        nthreads,
        comm,
        rank,
        size,
        kernel,
    )
    read_end = time.time()
    _print(
        f"Creating {len(galaxies)} galaxies took "
        f"{read_end - read_start:.2f} seconds."
    )

    # Start Pipeline object
    pipeline = Pipeline(
        emission_model=model,
        nthreads=nthreads,
        verbose=1,
        comm=mpi.COMM_WORLD,
    )

    pipeline.add_galaxies(galaxies)
    pipeline.get_spectra()
    
    pipeline.get_observed_spectra(cosmo=cosmo)
    pipeline.get_lines(line_ids=grid.available_lines)
    
    pipeline.get_observed_lines(cosmo)
       
    pipeline.get_sfzh(ages, Z)
    pipeline.get_sfh(ages)
    
    
    # Get Equivalent Widths for UV indices
    (
        index,
        index_window,
        blue_window,
        red_window,
    ) = set_index()  # Retrieve UV indices

    eqw = get_equivalent_width(grid_name, index, index_window, blue_window, red_window, galaxies, model)
    
    # Add galaxy info
    
    # Get slopes
    pipeline.add_analysis_func(
         lambda gal: get_UV_slopes(gal.stars),
         "Stars/UVSlope",
         )
    
    # Stellar Mass: sum of current masses (not initial) to reflect surviving mass
    pipeline.add_analysis_func(
        lambda gal: gal.stars.current_masses.sum() if gal.stars.current_masses.size > 0 else 0.0,
        "Stars/mstar_aperture"
    )

    # Mass-weighted stellar metallicity
    pipeline.add_analysis_func(
        lambda gal: (
            (gal.stars.metallicities * gal.stars.initial_masses).sum() / gal.stars.initial_masses.sum()
            if gal.stars.initial_masses.sum() > 0 else 0.0
        ),
        "Stars/MassWeightedStellarZ"
    )

    pipeline.add_analysis_func(
        lambda gal: float(estimate_uv_weighted_Z(gal.stars)),
        "Stars/UVWeightedStellarZ"
    )
    
    
    # Ensure each galaxy has a unique index
    for i, gal in enumerate(galaxies):
        gal.index = i

    # Add all EQW functions to the pipeline
    for i, uv_index in enumerate(index):
        eqws = eqw[i]

        # Create a function that returns the float EW for each galaxy
        def make_eqw_func(eqws):
            return lambda gal: float(eqws[gal.index].value)

        pipeline.add_analysis_func(
            make_eqw_func(eqws),
            f"UVIndices/EquivalentWidths{uv_index}"
        )



    pipeline.run()
    if rank == 0:
        pipeline.write("/cosma/home/dp004/dc-sant4/data/scripts/outputs/pipeline_results.hdf5", verbose=0)
    else:
        # other ranks skip writing
        pass

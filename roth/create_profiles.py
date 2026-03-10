#!/usr/bin/env python3
"""
Generates atmospheric temperature-pressure profiles using a hybrid physical model.

This script creates synthetic atmospheric profiles based on a set of sampled
physical parameters. All valid profiles, along with their generating parameters,
are saved into a single HDF5 dataset for efficient storage and access.
"""

import os
import logging
import json
from pathlib import Path
import numpy as np
import h5py
from scipy.special import expn
from tqdm import tqdm
from astropy import constants as const

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("numba").setLevel(logging.WARNING)

# --- Physical Constants and Reference Values (using astropy.constants) ---
# We extract the .value to use them as floats in numpy calculations
STEFAN_BOLTZMANN = const.sigma_sb.value
GRAVITATIONAL_CONSTANT_G = const.G.value
AU_TO_METERS = const.au.value
JUPITER_RADIUS_M = const.R_jup.value
JUPITER_MASS_KG = const.M_jup.value
SOLAR_RADIUS_M = const.R_sun.value
SOLAR_MASS_KG = const.M_sun.value

# --- Conversion and Reference Constants ---
BAR_TO_PASCAL = 1e5
BAR_TO_PA_CONVERSION = 1.0 * BAR_TO_PASCAL
ONE_BAR_REF_PRESSURE_BAR = 1.0


def compute_xi(gamma, tau):
    """
    Computes the ξ (xi) function for the temperature profile calculation.

    This function describes how radiation with a given mean opacity ratio (gamma)
    penetrates through the atmosphere as a function of optical depth (tau).
    It's derived from the two-stream approximation for radiative transfer.

    Parameters:
    -----------
    gamma : float or array
        Ratio of mean opacities in the visible to thermal wavelengths.
    tau : float or array
        Infrared optical depth.

    Returns:
    --------
    xi : float or array
        Dimensionless penetration function.
    """
    if np.any(gamma <= 0):
         raise ValueError("Negative Gamma")

    # Three components of the xi function from analytic radiative equilibrium
    term1 = 2/3
    term2 = (2/(3*gamma)) * (1 + (gamma*tau/2 - 1) * np.exp(-gamma*tau))
    term3 = (2*gamma/3) * (1 - tau**2/2) * expn(2, gamma*tau)

    return term1 + term2 + term3


def calculate_temperature_profile(pressure_bar, params):
    """
    Calculate the radiative temperature profile (K) using the three-stream model.

    This implements the analytic radiative equilibrium profile from Line et al. 2012,
    which assumes:
    - One upwelling stream of thermal emission from the planet
    - Two downwelling streams of stellar radiation with different mean opacities

    Parameters:
    -----------
    pressure_bar : array
        Atmospheric pressure levels in bars.
    params : dict
        Dictionary containing all physical parameters needed for the calculation.

    Returns:
    --------
    temperature_k : array
        Temperature profile in Kelvin at each pressure level.
    """
    # Extract planet properties
    mass_kg = params["planet_mass_kg"]
    radius_m = params["planet_radius_m"]

    # Calculate surface gravity (m/s²)
    gravity_m_s2 = (GRAVITATIONAL_CONSTANT_G * mass_kg) / (radius_m**2)

    # Extract opacity and power-law parameters
    kappa_ref = 10**params["log_kappa_ir_m2_kg"]
    n = params["power_law_n"]

    # The index 'n' must be positive for a physically meaningful profile
    if n <= 0:
        raise ValueError(f"power_law_n 'n' must be > 0, but got {n}")

    # Calculate a characteristic optical depth scale, based on κ_ref.
    # This term, (κ_ref * P_ref) / g, is a useful physical quantity.
    tau_scale_factor = (kappa_ref * BAR_TO_PA_CONVERSION) / gravity_m_s2

    # Calculate optical depth using the physically rigorous integrated formula:
    # τ(p) = [ (κ_ref * P_ref) / (g * n) ] * (P / P_ref)^n
    # This is equivalent to (tau_scale_factor / n) * (P/P_ref)^n
    optical_depth = (tau_scale_factor / n) * (pressure_bar / ONE_BAR_REF_PRESSURE_BAR)**n

    # Extract temperature parameters
    T_internal_k    = params["T_internal_k"]
    T_irradiation_k = params["T_irradiation_k"]

    # Pre-calculate fourth powers for efficiency
    T4_internal_k4 = T_internal_k**4
    T4_irr_eff_k4 = T_irradiation_k**4

    # 1. Deep atmosphere term (internal heat flux)
    # This represents heat from the planet's interior
    deep_term_k4 = (3 * T4_internal_k4 / 4) * (2/3 + optical_depth)

    # 2. First stellar radiation channel
    # Uses gamma1 as the visible-to-IR opacity ratio
    gamma1 = 10**params["log_gamma_channel1"]
    channel1_fraction = 1 - params["flux_partition_alpha"]  # Fraction of flux in channel 1
    channel1_term_k4 = (3 * T4_irr_eff_k4 / 4) * channel1_fraction * compute_xi(gamma1, optical_depth)

    # 3. Second stellar radiation channel
    # Uses gamma2 as the visible-to-IR opacity ratio
    gamma2 = 10**params["log_gamma_channel2"]
    channel2_fraction = params["flux_partition_alpha"]  # Fraction of flux in channel 2
    channel2_term_k4 = (3 * T4_irr_eff_k4 / 4) * channel2_fraction * compute_xi(gamma2, optical_depth)

    # Sum all contributions to get total temperature^4
    T4_total_k4 = deep_term_k4 + channel1_term_k4 + channel2_term_k4

    # Check for unphysical negative values
    if np.any(T4_total_k4 < 0):
        raise ValueError("Negative T^4 encountered in temperature calculation.")

    # Return temperature (take fourth root)
    return T4_total_k4**0.25


def apply_convection(pressure_bar, temperature_k, adiabatic_gradient):
    """
    Apply convective adjustment to a temperature profile.

    When the radiative temperature gradient exceeds the adiabatic gradient,
    the atmosphere becomes convectively unstable. This function adjusts the
    profile to follow an adiabat from the point of instability downward.

    Parameters:
    -----------
    pressure_bar : array
        Atmospheric pressure levels in bars.
    temperature_k : array
        Initial temperature profile in Kelvin.
    adiabatic_gradient : float
        The adiabatic temperature gradient d(ln T)/d(ln P).

    Returns:
    --------
    adjusted_k : array
        Temperature profile after convective adjustment.
    """
    adjusted_k = np.copy(temperature_k)
    log_p = np.log10(pressure_bar)
    log_t = np.log10(temperature_k)
    for i in range(1, len(adjusted_k)):
        gradient = (log_t[i] - log_t[i-1]) / (log_p[i] - log_p[i-1])
        if gradient > adiabatic_gradient:
            p_top_bar = pressure_bar[i-1]
            t_top_k = adjusted_k[i-1]
            # T/T_top = (P/P_top)^(adiabatic_gradient)
            adjusted_k[i:] = t_top_k * (pressure_bar[i:] / p_top_bar)**adiabatic_gradient
            break
    return adjusted_k


def validate_temperature_profile(temperature_k):
    """
    Check if a temperature profile meets physical validity criteria.

    Parameters:
    -----------
    temperature_k : array
        Temperature profile in Kelvin.

    Returns:
    --------
    is_valid : bool
        True if profile passes all checks, False otherwise.
    reason : str
        Description of why profile failed (empty string if valid).
    """
    if np.any(np.isnan(temperature_k)):
        return False, "Profile contains NaN values"
    if np.any(temperature_k < 50):
        return False, f"Temperature below 50 K found (min: {np.min(temperature_k):.1f} K)"
    if np.any(temperature_k > 4000):
        return False, f"Temperature above 4000 K found (max: {np.max(temperature_k):.1f} K)"
    bottom_temperature = temperature_k[-1]
    if bottom_temperature < 1000:
        return False, f"Bottom temperature too cold ({bottom_temperature:.1f} K < 1000 K)"
    top_temperature = temperature_k[0]
    if top_temperature > 2800:
        return False, f"Top temperature too hot"
    return True, ""


class ProfileGenerator:
    """A class to generate and manage the creation of atmospheric profiles."""

    def __init__(self, config):
        self.config = config
        p_config = config.get("pressure_bar_grid", {})
        self.pressure_bar = np.logspace(
            np.log10(p_config.get("min", 1e-5)),
            np.log10(p_config.get("max", 1e2)),
            p_config.get("points", 50)
        )

    def sample_parameters(self):
        """Sample a set of physical parameters from the configured distributions."""
        params = {}
        for key, dist_config in self.config.items():
            if isinstance(dist_config, dict) and "distribution" in dist_config:
                dist_type = dist_config["distribution"]
                if dist_type == "uniform":
                    value = np.random.uniform(dist_config["min"], dist_config["max"])
                elif dist_type == "normal":
                    value = np.random.normal(dist_config["mean"], dist_config["std"])
                elif dist_type == "fixed":
                    value = dist_config["value"]
                else:
                    raise ValueError(f"Unknown distribution type: {dist_type}")
                params[key] = value
        return params

    def generate_single_profile(self, index=0):
        """
        Generate a single atmospheric profile with enhanced validation.
        """
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                params = self.sample_parameters()

                temperature_k = calculate_temperature_profile(self.pressure_bar, params)
                if np.random.rand() < params.get("convection_fraction", 0.5):
                    temperature_k = apply_convection(
                        self.pressure_bar, temperature_k, params.get("adiabatic_gradient", 0.3)
                    )
                temperature_k += params.get("temperature_shift_k", 0.0)
                is_valid, failure_reason = validate_temperature_profile(temperature_k)
                if not is_valid:
                    logging.debug(f"Profile {index} attempt {attempt+1} failed: {failure_reason}")
                    continue

                # --- Assemble the minimal dictionary for the HDF5 dataset ---
                output_data = {}

                # 1. P-T profile
                output_data["pressure_bar"] = self.pressure_bar
                output_data["temperature_k"] = temperature_k

                # 2. Planet and Star properties (from sampled params)
                output_data["planet_radius_m"] = params["planet_radius_m"]
                output_data["planet_mass_kg"] = params["planet_mass_kg"]
                output_data["star_radius_m"] = params["star_radius_m"]
                output_data["star_temperature_k"] = params["star_temperature_k"]
                output_data["star_metallicity"] = params["star_metallicity"]
                output_data["star_logg"] = params["star_logg"]

                # 3. Orbital distance (using logic moved from script 3)
                orbital_distance_m = (params["star_radius_m"] / 2.0) * (params["beta"] * params["star_temperature_k"] / params["T_irradiation_k"])**2
                orbital_distance_m *= params["orbit_modifier"]
                if orbital_distance_m <= 0:
                    orbital_distance_m = 0.01 * AU_TO_METERS # Prevent non-physical distances
                output_data["orbital_distance_m"] = orbital_distance_m

                logging.debug(f"Profile {index} generated successfully after {attempt+1} attempts.")
                return output_data

            except Exception as e:
                logging.debug(f"Attempt {attempt+1} for profile {index} failed with exception: {e}")
                continue
        logging.warning(f"Profile {index} failed to generate after {max_attempts} attempts.")
        return None

    def generate_dataset(self, output_hdf5_path):
        """Generates profiles and saves them to a single HDF5 file."""
        n_profiles_to_generate = self.config.get("n_profiles", 10)
        
        # Dry run to get schema
        logging.info("Performing dry run to determine HDF5 schema...")
        first_profile_data = self.generate_single_profile(index=-1)
        if not first_profile_data:
            logging.critical("Dry run failed to generate a valid profile. Aborting.")
            return

        logging.info("Dry run successful. Creating HDF5 file.")
        
        with h5py.File(output_hdf5_path, 'w') as hf:
            hf.attrs['config'] = json.dumps(self.config, default=lambda o: '<not serializable>')
            
            # Create resizable datasets
            datasets = {}
            for key, value in first_profile_data.items():
                value_np = np.array(value)
                shape = (1,) + value_np.shape
                maxshape = (n_profiles_to_generate,) + value_np.shape
                dtype = np.float32 if value_np.dtype.kind == 'f' else value_np.dtype
                datasets[key] = hf.create_dataset(
                    key, shape, maxshape=maxshape, dtype=dtype, compression="gzip"
                )
                datasets[key][0] = value_np
            
            success_count = 1
            for i in tqdm(range(1, n_profiles_to_generate), desc="Generating Profiles"):
                result = self.generate_single_profile(index=i)
                if result:
                    # Resize and append
                    for key, dset in datasets.items():
                        dset.resize((success_count + 1,) + dset.shape[1:])
                        dset[success_count] = result[key]
                    success_count += 1
            
            logging.info(f"HDF5 generation complete. Generated {success_count}/{n_profiles_to_generate} valid profiles.")
            if success_count < n_profiles_to_generate:
                logging.info(f"Final dataset size is {success_count} profiles.")


# ==============================================================================
# --- Default Configuration with Explanatory Comments ---
# ==============================================================================
DEFAULT_CONFIG = {
    # --- General Script Settings ---
    "n_profiles": 100000,

    # --- Atmospheric Structure ---
    "pressure_bar_grid": {"min": 1e-5, "max": 1e2, "points": 75},

    # --- Planet and Star Physical Parameters (fixed values match script 1's defaults) ---
    "planet_radius_m": {"distribution": "fixed", "value": JUPITER_RADIUS_M},
    "planet_mass_kg": {"distribution": "fixed", "value": JUPITER_MASS_KG},
    "star_temperature_k": {"distribution": "fixed", "value": 6000.0},
    "star_radius_m": {"distribution": "fixed", "value": SOLAR_RADIUS_M},
    "star_metallicity": {"distribution": "fixed", "value": 0.0},
    "star_logg": {"distribution": "fixed", "value": 4.5},

    # --- Opacity and Radiative Transfer Parameters ---
    "log_kappa_ir_m2_kg": {"distribution": "normal", "mean": -2.5, "std": 2.5},
    "power_law_n": {"distribution": "uniform", "min": 0.5, "max": 2.0},
    "flux_partition_alpha": {"distribution": "uniform", "min": 0.0, "max": 1.0},
    "log_gamma_channel1": {"distribution": "uniform", "min": -2, "max": 2},
    "log_gamma_channel2": {"distribution": "uniform", "min": -2, "max": 2},
    "beta": {"distribution": "uniform", "min": 0.5, "max": 2**0.5},

    # --- Other Physical and Stochastic Parameters ---
    "T_internal_k": {"distribution": "uniform", "min": 400, "max": 700},
    "T_irradiation_k": {"distribution": "normal", "mean": 1800, "std": 500},
    "orbit_modifier": {"distribution": "normal", "mean": 1.0, "std": 0.5},
    "temperature_shift_k": {"distribution": "uniform", "min": -600, "max": 600},
    "convection_fraction": {"distribution": "fixed", "value": 0.333},
    "adiabatic_gradient": {"distribution": "uniform", "min": 0.25, "max": 0.35}
}


if __name__ == "__main__":
    # --- SCRIPT CONFIGURATION ---
    OUTPUT_HDF5_FILE = "synthetic_profiles.h5"

    # --- Pre-run Cleanup ---
    output_path = Path(OUTPUT_HDF5_FILE)
    if output_path.exists():
        logging.warning(f"Output file '{output_path}' already exists. It will be removed and recreated.")
        output_path.unlink()

    # --- Main Execution ---
    try:
        generator = ProfileGenerator(DEFAULT_CONFIG)
        generator.generate_dataset(OUTPUT_HDF5_FILE)
        file_size_mb = output_path.stat().st_size / (1024**2)
        logging.info(f"Created HDF5 file: {output_path} ({file_size_mb:.2f} MB)")
    except Exception as e:
        logging.critical(f"A critical error occurred during profile generation: {e}", exc_info=True)
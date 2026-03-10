#!/usr/bin/env python3
"""
Convert Roth GCM PT grid files into one standalone HDF5 profile dataset.

This script performs the following steps:
1.  Scans a directory for GCM output files with a specific naming convention.
2.  Parses physical parameters (Teq, metallicity, etc.) from filenames.
3.  Allows users to filter which source GCM files to process.
4.  Reads the multi-block GCM data, correctly handling its 3D structure.
5.  Iterates through every individual (longitude, latitude) column in each file.
    Each column is treated as a unique 1D pressure-temperature profile.
6.  Interpolates each 1D profile onto a new, user-defined pressure grid.
7.  Sets planet mass and radius to 1 M_jup and 1 R_jup, and derives gravity.
8.  Saves profiles to HDF5 (normal mode) or a set number of JSONs (test mode).

This utility is standalone and separate from the maintained `roth_sampler`
integration used by `src/main.py --gen`.
"""

import logging
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d
from tqdm import tqdm
from astropy import constants as const
import io

# ==============================================================================
# --- Logging Configuration ---
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# --- Global Physical Constants (from astropy) ---
# ==============================================================================
JUPITER_RADIUS_M = const.R_jup.value
JUPITER_MASS_KG = const.M_jup.value
GRAVITATIONAL_CONSTANT_G = const.G.value
SOLAR_RADIUS_M = const.R_sun.value
AU_TO_METERS = const.au.value


# ==============================================================================
# --- MAIN SCRIPT CONFIGURATION ---
# ==============================================================================
CONFIG = {
    # --- Mode Selection ---
    "test_mode": False,
    "test_mode_file_count": 5,
    "test_output_json_basename": "test_profile.json",

    # --- Path Settings ---
    "input_data_directory": "PTprofiles/",
    "output_hdf5_file": "gcm_profiles.h5",

    # --- File Filtering ---
    "file_filters": {
        'Mstar': [0.8, 1.3],
        'Rp': [0.8, 1.3],
        'Teq': [1199, 2201]
    },

    # --- Profile Interpolation Settings ---
    "new_pressure_grid": {
        "min_bar": 1e-5,
        "max_bar": 1e2,
        "points": 75,
    },

    # --- Default Physical Parameters ---
    "default_parameters": {
        "star_temperature_k": 6000.0,
        "star_radius_m": 1.0 * SOLAR_RADIUS_M,
        "star_logg": 4.5,
    }
}

# --- CORRECTED FUNCTION ---
def parse_gcm_filename(filepath):
    """Extracts all physical parameters from a GCM output filename using regex."""
    pattern = re.compile(
        r"Teq_([\d.]+)-"
        r"LogMet_([-]?[\d.]+)-"
        r"LogDrag_([-]?[\d.]+)-"
        r"Mstar_([\d.]+)-"
        r"Rp_([\d.]+)-"
        r"logG_([-]?[\d.]+)-"
        r"TiOVO_(true|false)"
    )
    match = pattern.search(filepath.name)
    if not match: return None
    return {
        "Teq": float(match.group(1)), "LogMet": float(match.group(2)),
        "LogDrag": float(match.group(3)), "Mstar": float(match.group(4)),
        "Rp": float(match.group(5)), "logG": float(match.group(6)),
        "TiOVO": match.group(7).lower() == 'true',
    }

# --- CORRECTED FUNCTION ---
def find_and_filter_files(directory, filters):
    """Scans a directory for .dat files, parses their parameters, and filters them."""
    input_dir = Path(directory)
    if not input_dir.is_dir():
        logging.error(f"Input directory '{input_dir}' not found."); return []
    all_files = []
    logging.info(f"Scanning for GCM files in '{input_dir}'...")
    for filepath in sorted(input_dir.glob("*.dat")):
        params = parse_gcm_filename(filepath)
        if params: all_files.append({"path": filepath, "params": params})
    if not filters:
        logging.info(f"No filters applied. Found {len(all_files)} files to process."); return all_files

    filtered_files = []
    for file_info in all_files:
        is_match = True
        for key, value in filters.items():
            # Check if the parameter required by the filter exists in the parsed filename
            if key not in file_info["params"]:
                is_match = False
                break
            
            param_val = file_info["params"][key]
            # Check range for list values, or equality for single values
            if isinstance(value, list) and len(value) == 2:
                if not (value[0] <= param_val <= value[1]):
                    is_match = False
                    break
            elif param_val != value:
                is_match = False
                break
        
        if is_match:
            filtered_files.append(file_info)
            
    logging.info(f"Found {len(all_files)} total files. After filtering, {len(filtered_files)} files will be processed.")
    return filtered_files


def extract_individual_profiles(filepath):
    """Reads a multi-block GCM file and returns a list of all individual 1D profiles."""
    column_names = [
        "Level", "Lon", "Lat", "Pressure_bar", "Temp_K", "Pot_Temp",
        "U_wind", "V_wind", "Vert_wind_Pa_s", "Vert_wind_m_s", "Density"
    ]
    data_lines = []
    try:
        with open(filepath, 'r') as f: lines = f.readlines()
        start_index = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line[0].isdigit() or (line.startswith('-') and line[1].isdigit())):
                start_index = i; break
        for line in lines[start_index:]:
            line = line.strip()
            if line and (line[0].isdigit() or line[0] == '-'): data_lines.append(line)
        if not data_lines:
            logging.warning(f"No data lines could be extracted from '{filepath.name}'."); return []
        data_buffer = io.StringIO("\n".join(data_lines))
        df = pd.read_csv(
            data_buffer, sep=r',|\s+', header=None, engine='python', names=column_names
        )
        if df.empty:
            logging.warning(f"DataFrame is empty for file '{filepath.name}' after parsing."); return []
        grouped = df.groupby(['Lon', 'Lat'])
        profiles = []
        for (lon, lat), group_df in grouped:
            profiles.append({'lon': lon, 'lat': lat, 'profile_df': group_df[['Pressure_bar', 'Temp_K']]})
        return profiles
    except Exception as e:
        logging.error(f"A critical error occurred while reading file '{filepath.name}': {e}", exc_info=True)
        return []

def numpy_to_list_converter(data_dict):
    """Converts numpy arrays in a dictionary to Python lists for JSON serialization."""
    serializable_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            serializable_dict[key] = value.tolist()
        else:
            serializable_dict[key] = value
    return serializable_dict

def main():
    """Main script execution. Dispatches to test or full mode."""
    files_to_process = find_and_filter_files(CONFIG["input_data_directory"], CONFIG["file_filters"])
    if not files_to_process:
        logging.info("No files to process. Exiting."); return
    p_grid_config = CONFIG["new_pressure_grid"]
    new_pressure_grid = np.logspace(np.log10(p_grid_config["min_bar"]), np.log10(p_grid_config["max_bar"]), p_grid_config["points"])
    if CONFIG.get("test_mode", False):
        run_test_mode(files_to_process, new_pressure_grid)
    else:
        run_full_mode(files_to_process, new_pressure_grid)

def process_single_profile(profile_data, parsed_params, new_pressure_grid):
    """Core logic to process one T-P column into a full set of parameters."""
    df_prof = profile_data['profile_df']
    p_raw, t_raw = df_prof['Pressure_bar'].values, df_prof['Temp_K'].values
    if len(p_raw) < 2: return None

    sort_indices = np.argsort(p_raw)
    p_raw, t_raw = p_raw[sort_indices], t_raw[sort_indices]

    interp_func = interp1d(np.log10(p_raw), t_raw, bounds_error=False, fill_value="extrapolate")
    t_interpolated = interp_func(np.log10(new_pressure_grid))

    output_data = {}
    output_data["pressure_bar"] = new_pressure_grid
    output_data["temperature_k"] = t_interpolated

    # --- HARDCODED & DERIVED PLANET PARAMETERS ---
    # Mass and radius are fixed to 1x Jupiter values.
    # Gravity is then derived from these for physical consistency.
    planet_radius_m = 1.0 * JUPITER_RADIUS_M
    planet_mass_kg = 1.0 * JUPITER_MASS_KG
    planet_gravity_m_s2 = (GRAVITATIONAL_CONSTANT_G * planet_mass_kg) / (planet_radius_m**2)

    output_data["planet_radius_m"] = planet_radius_m
    output_data["planet_mass_kg"] = planet_mass_kg
    output_data["planet_gravity_m_s2"] = planet_gravity_m_s2
    
    defaults = CONFIG["default_parameters"]
    output_data["star_temperature_k"] = defaults["star_temperature_k"]
    output_data["star_radius_m"] = defaults["star_radius_m"]
    output_data["star_logg"] = defaults["star_logg"]
    output_data["star_metallicity"] = 0.0

    Teq = parsed_params.get("Teq", 0)
    if Teq <= 0:
        output_data["orbital_distance_m"] = 0.01 * AU_TO_METERS
    else:
        output_data["orbital_distance_m"] = (output_data["star_radius_m"] / 2.0) * (output_data["star_temperature_k"] / Teq)**2
        
    return output_data

def run_test_mode(files_to_process, new_pressure_grid):
    """Finds a set number of valid profiles, saves them to JSON, and exits."""
    logging.info("--- RUNNING IN TEST MODE ---")
    target_count = CONFIG.get("test_mode_file_count", 1)
    logging.info(f"Script will find {target_count} valid profiles, save as JSON, and exit.")
    saved_files_count = 0
    base_path = Path(CONFIG["test_output_json_basename"])
    for file_info in files_to_process:
        filepath = file_info["path"]
        parsed_params = file_info["params"]
        individual_profiles = extract_individual_profiles(filepath)
        if not individual_profiles:
            continue
        for profile_data in individual_profiles:
            try:
                output_data = process_single_profile(profile_data, parsed_params, new_pressure_grid)
                if output_data is None: continue

                saved_files_count += 1
                output_filename = base_path.with_name(f"{base_path.stem}_{saved_files_count}{base_path.suffix}")
                serializable_data = numpy_to_list_converter(output_data)
                with open(output_filename, 'w') as f:
                    json.dump(serializable_data, f, indent=4)
                logging.info(f"Saved test file {saved_files_count}/{target_count}: {output_filename}")
                if saved_files_count >= target_count:
                    logging.info(f"Reached target of {target_count} test files. Exiting.")
                    return
            except Exception as e:
                lon, lat = profile_data['lon'], profile_data['lat']
                logging.warning(f"Skipping profile (Lon={lon}, Lat={lat}) from '{filepath.name}' due to processing error: {e}")
                continue
    logging.warning(f"Test mode finished. Only found {saved_files_count}/{target_count} valid profiles to save.")

def run_full_mode(files_to_process, new_pressure_grid):
    """Runs the original logic to generate a full HDF5 file."""
    logging.info("--- RUNNING IN FULL MODE ---")
    output_path = Path(CONFIG["output_hdf5_file"])
    if output_path.exists():
        logging.warning(f"Output file '{output_path}' already exists. It will be removed and recreated.")
        output_path.unlink()
    profile_count = 0
    with h5py.File(output_path, 'w') as hf:
        hf.attrs['config'] = json.dumps(CONFIG)
        datasets = {}
        for file_info in tqdm(files_to_process, desc="Processing GCM Files"):
            filepath = file_info["path"]
            parsed_params = file_info["params"]
            individual_profiles = extract_individual_profiles(filepath)
            if not individual_profiles:
                continue
            for profile_data in individual_profiles:
                try:
                    output_data = process_single_profile(profile_data, parsed_params, new_pressure_grid)
                    if output_data is None: continue
                    
                    if not datasets:
                        for key, value in output_data.items():
                            value_np = np.array(value)
                            shape = (1,) + value_np.shape
                            maxshape = (None,) + value_np.shape
                            dtype = np.float32 if value_np.dtype.kind == 'f' else value_np.dtype
                            datasets[key] = hf.create_dataset(key, shape=shape, maxshape=maxshape, dtype=dtype, compression="gzip")
                            datasets[key][0] = value_np
                    else:
                        for key, dset in datasets.items():
                            dset.resize((profile_count + 1,) + dset.shape[1:])
                            dset[profile_count] = output_data[key]
                    profile_count += 1
                except Exception as e:
                    lon, lat = profile_data['lon'], profile_data['lat']
                    logging.error(f"Failed profile (Lon={lon}, Lat={lat}) from '{filepath.name}': {e}")
                    continue
    logging.info(f"Processing complete. Generated HDF5 file with {profile_count} profiles.")
    if profile_count > 0:
        file_size_mb = output_path.stat().st_size / (1024**2)
        logging.info(f"Created HDF5 file: {output_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()

"""Centralized constants for the VULCAN emulator pipeline.

All shared constants live here so every module imports from a single
source of truth.  Module-private constants (used in exactly one file)
stay in their respective modules.

NOTE: ``standalone_inference.py`` is self-contained (embedded in NPZ
bundles) and intentionally duplicates some values defined here.  Do
not add imports from this file into ``standalone_inference.py``.
"""

from __future__ import annotations

# =========================================================================
# Physical constants — CGS units (CODATA 2018)
# =========================================================================

# Speed of light [cm s-1].
C_LIGHT = 2.99792458e10

# Planck constant [erg s].
H_PLANCK = 6.62607015e-27

# Boltzmann constant [erg K-1].
K_BOLTZMANN = 1.380649e-16

# Solar radius [cm] (IAU 2015 nominal).
R_SUN_CM = 6.957e10

# Astronomical unit [cm] (IAU 2012).
AU_CM = 1.495978707e13

# =========================================================================
# Solar reference abundances — Asplund et al. (2009)
# =========================================================================

# Hydrogen-normalized elemental number fractions (n_X / n_H):
# total X-element atoms (nuclei) per total H atoms (nuclei).
# NOT mass fractions, NOT molecular volume fractions. Matches VULCAN's
# convention (see VULCAN build_atm.py: the {O,C,N,S,He}_H values satisfy
# n_X = X_H * n_H in the equilibrium solver).
# Used as the baseline for metallicity-scaled sampling.
SOLAR_ABUNDANCES: dict[str, float] = {
    "He_H": 7.84e-2,
    "C_H": 2.69e-4,
    "O_H": 4.90e-4,
    "N_H": 6.76e-5,
    "S_H": 1.32e-5,
}

# =========================================================================
# Analytic PT sampler constants
# =========================================================================

# Reference surface gravity used by the Piette & Madhusudhan (2019) Guillot
# analytic PT sampler [cm s-2].
#
# This is fixed on purpose and deliberately NOT exposed as a sampling range:
# the Guillot form enters only through ``delta = kappa_IR / g``, and
# ``log10_delta_range`` already spans many decades.  Varying g alongside
# delta would re-sample the same PT manifold (gravity is fully degenerate
# with delta).  Kinetic / VULCAN runs sample gravity independently because
# there it drives scale height and Kzz — that is a different code path.
ANALYTIC_SAMPLER_GRAVITY_CM_S2 = 2500.0

# =========================================================================
# Species molar masses — g mol-1
# =========================================================================

# Molar masses for the 17 default output species tracked by the emulator.
SPECIES_MOLAR_MASS: dict[str, float] = {
    "H2": 2.016,
    "He": 4.003,
    "H": 1.008,
    "O": 15.999,
    "OH": 17.007,
    "H2O": 18.015,
    "CO": 28.010,
    "CO2": 44.009,
    "CH4": 16.043,
    "N2": 28.014,
    "NH3": 17.031,
    "H2S": 34.081,
    "SH": 33.073,
    "S": 32.065,
    "SO": 48.064,
    "SO2": 64.064,
    "S2": 64.130,
}

# =========================================================================
# Chemistry / model type enumerations
# =========================================================================

# Supported atmosphere base gases.
SUPPORTED_ATM_BASES = ("H2", "N2", "O2", "CO2", "H2O")

# Public VULCAN physics toggle names exposed in the config.
PUBLIC_PHYSICS_TOGGLES = (
    "use_photochemistry",
    "use_ion_chemistry",
    "use_eddy_diffusion",
    "use_molecular_diffusion",
    "use_upwind_molecular_diffusion",
    "use_boundary_conditions",
    "use_condensation",
    "use_settling",
    "use_initial_cold_trap",
    "use_sat_surface_h2o",
)

# Valid top-level chemistry_type values.
CHEMISTRY_TYPES = ("fastchem", "vulcan")

# Valid top-level model_type values.
MODEL_TYPES = ("transformer",)

# =========================================================================
# Elemental and global input orderings
# =========================================================================

# Fixed elemental order used internally for all chemistry types.
# These are hydrogen-normalized absolute abundances (n_X / n_H).
ELEMENT_INPUT_ORDER = ("He_H", "C_H", "O_H", "N_H", "S_H")

# FastChem conditioning uses the bare elemental order.
FASTCHEM_CONDITIONING_INPUT_ORDER = ELEMENT_INPUT_ORDER

# VULCAN conditioning prepends planetary parameters to the elemental order.
VULCAN_CONDITIONING_INPUT_ORDER = (
    "gravity_cm_s2",
    "planet_radius_cm",
    *FASTCHEM_CONDITIONING_INPUT_ORDER,
)

# Core global inputs for each chemistry type.
VULCAN_CORE_GLOBAL_INPUTS = VULCAN_CONDITIONING_INPUT_ORDER
FASTCHEM_CORE_GLOBAL_INPUTS = FASTCHEM_CONDITIONING_INPUT_ORDER

# Stellar irradiation geometry globals sampled per VULCAN run.
VULCAN_STELLAR_GLOBAL_INPUTS = (
    "r_star_rsun",
    "semi_major_axis_au",
    "zenith_angle_deg",
    "diurnal_factor",
)

# Optional VULCAN globals: physics toggles + atmosphere base one-hots.
VULCAN_OPTIONAL_GLOBAL_INPUTS = (
    *PUBLIC_PHYSICS_TOGGLES,
    *tuple(f"atm_base_{name}" for name in SUPPORTED_ATM_BASES),
)

# Full default global input vector for VULCAN models.
DEFAULT_REQUIRED_GLOBAL_INPUTS = (
    *VULCAN_CORE_GLOBAL_INPUTS,
    *VULCAN_STELLAR_GLOBAL_INPUTS,
    *VULCAN_OPTIONAL_GLOBAL_INPUTS,
)

# Default set of chemical species tracked in model output.
DEFAULT_STATE_SPECIES = (
    "H2",
    "He",
    "H",
    "O",
    "OH",
    "H2O",
    "CO",
    "CO2",
    "CH4",
    "N2",
    "NH3",
    "H2S",
    "SH",
    "S",
    "SO",
    "SO2",
    "S2",
)

# =========================================================================
# Config validation allowlists
# =========================================================================

# Valid activation function names.
_ALLOWED_ACTIVATIONS = {
    "elu",
    "gelu",
    "leaky_relu",
    "relu",
    "selu",
    "silu",
    "softplus",
    "tanh",
}

# Valid learning-rate scheduler names.
_ALLOWED_LR_SCHEDULERS = {"cosine", "reduce_on_plateau"}

# Valid temperature profile source modes.
_ALLOWED_TEMPERATURE_PROFILE_SOURCE_MODES = {"analytic", "pt_library", "mixed"}

# Valid normalization method names.
_ALLOWED_NORMALIZATION_METHODS = {"standard", "log-standard", "log-minmax", "none"}

# Numeric filter keys for the Roth PT-library profiles.
_ALLOWED_TEMPERATURE_PROFILE_NUMERIC_FILTER_KEYS = {
    "Teq",
    "LogMet",
    "LogDrag",
    "Mstar",
    "Rp",
    "logG",
}

# Boolean filter keys for the Roth PT-library profiles.
_ALLOWED_TEMPERATURE_PROFILE_BOOLEAN_FILTER_KEYS = {"TiOVO"}

# Combined filter key set (numeric + boolean).
_ALLOWED_TEMPERATURE_PROFILE_FILTER_KEYS = (
    _ALLOWED_TEMPERATURE_PROFILE_NUMERIC_FILTER_KEYS
    | _ALLOWED_TEMPERATURE_PROFILE_BOOLEAN_FILTER_KEYS
)

# =========================================================================
# Internal VULCAN runtime defaults
# =========================================================================

# Default values for optional vulcan.runtime config keys.
_INTERNAL_VULCAN_RUNTIME_DEFAULTS = {
    "python_executable": "python",
    "cfg_file": "vulcan_cfg.py",
    "worker_root": "data/vulcan_workers",
    "regenerate_chem_funs": False,
    "cfg_assignments": {},
    "use_lowT_limit_rates": True,
    "use_adaptive_rtol": True,
    "rocky": False,
    "top_bc_flux_file": None,
    "bot_bc_flux_file": None,
}

# Name of the default science preset when none is explicitly configured.
_DEFAULT_SCIENCE_PRESET_NAME = "default"

# =========================================================================
# Export / data versioning
# =========================================================================

# Format identifier embedded in exported NPZ bundles.
EXPORT_FORMAT = "jax_physical_bundle"

# Bump when the export layout changes in a backwards-incompatible way.
EXPORT_VERSION = 6

# Bump when the processed tensor layout changes incompatibly.
# v19: variable num_levels + variable pressure ranges. Adds per-run valid_mask
# and position_coord (normalized log10(P) in [0, 1]) tensors alongside
# sequence_inputs/target_outputs, and records (num_levels_range,
# pressure_top_bar_range, pressure_bottom_bar_range,
# log10_pressure_bar_union_range) in the data contract.
PROCESSED_DATA_VERSION = 19

# Subdirectory name for shared metadata within a processed dataset.
PROCESSED_INFO_DIRNAME = "info"

# =========================================================================
# Roth PT-library filter keys
# =========================================================================

# Numeric metadata columns available for profile filtering.
ROTH_NUMERIC_FILTER_KEYS = ("Teq", "LogMet", "LogDrag", "Mstar", "Rp", "logG")

# Boolean metadata columns available for profile filtering.
ROTH_BOOLEAN_FILTER_KEYS = ("TiOVO",)

# All supported filter keys (numeric + boolean).
ROTH_FILTER_KEYS = (*ROTH_NUMERIC_FILTER_KEYS, *ROTH_BOOLEAN_FILTER_KEYS)

# =========================================================================
# Neural network constants
# =========================================================================

# Base wavelength for the standard sinusoidal positional encoding
# (Vaswani et al., 2017).
_SINUSOIDAL_BASE_WAVELENGTH = 10_000.0

# Canonical pseudo-length used to scale the continuous positional
# encoding input. Positions live in ``[0, 1]`` (normalized log10-pressure)
# and are multiplied by this constant before entering the sinusoidal
# encoder, so the lowest frequency band spans roughly this many "steps"
# across the full training pressure range. Keeping it in the same order
# of magnitude as the historical fixed ``num_levels`` (~64) preserves
# the frequency bands the model was tuned against.
_POSITION_SCALE = 64.0

# =========================================================================
# ExoJAX-facing label lists
# =========================================================================

# Ordered global-input labels for FastChem ExoJAX wrappers.
FASTCHEM_GLOBAL_LABELS = list(FASTCHEM_CONDITIONING_INPUT_ORDER)

# Ordered global-input labels for VULCAN ExoJAX wrappers.
VULCAN_GLOBAL_LABELS = list(DEFAULT_REQUIRED_GLOBAL_INPUTS)

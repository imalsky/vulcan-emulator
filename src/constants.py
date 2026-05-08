"""Centralized constants for the VULCAN emulator pipeline.

All shared constants live here so every module imports from a single
source of truth.  Module-private constants (used in exactly one file)
stay in their respective modules.
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
# Used as the solar anchor for the per-element X/H sampling ranges and as
# the reference point for the ExoJAX ``global_inputs_from_metallicity``
# helper (retrieval-side metallicity reparam, not a sampling step).
SOLAR_ABUNDANCES: dict[str, float] = {
    "He_H": 7.84e-2,
    "C_H": 2.69e-4,
    "O_H": 4.90e-4,
    "N_H": 6.76e-5,
    "S_H": 1.32e-5,
}

# Lodders (2009) solar abundances shipped with the FastChem runtime at
# ``VULCAN-master/fastchem_vulcan/input/solar_element_abundances.dat``.
# FastChem reads these at every training run and overrides only He/C/O/N/S
# from the sampled globals, so any classical reference that claims to mirror
# the FastChem training contract must use *these* values as the background
# for refractories (P, Si, Ti, V, Cl, K, Na, Mg, F, Ca, Fe) — not the AAG21
# background that ExoGibbs defaults to.
# Values are n_X/n_H, converted from log_eps = log10(n_X/n_H) + 12 entries in
# the FastChem file.
FASTCHEM_LODDERS_SOLAR_ABUNDANCES: dict[str, float] = {
    "He": 10.0 ** (10.9864 - 12.0),
    "C": 10.0 ** (8.4434 - 12.0),
    "N": 10.0 ** (7.9130 - 12.0),
    "O": 10.0 ** (8.7826 - 12.0),
    "S": 10.0 ** (7.12 - 12.0),
    "P": 10.0 ** (5.5058 - 12.0),
    "Si": 10.0 ** (7.5867 - 12.0),
    "Ti": 10.0 ** (4.9794 - 12.0),
    "V": 10.0 ** (4.0437 - 12.0),
    "Cl": 10.0 ** (5.3002 - 12.0),
    "K": 10.0 ** (5.1619 - 12.0),
    "Na": 10.0 ** (6.3479 - 12.0),
    "Mg": 10.0 ** (7.5995 - 12.0),
    "F": 10.0 ** (4.49196 - 12.0),
    "Ca": 10.0 ** (6.3677 - 12.0),
    "Fe": 10.0 ** (7.5151 - 12.0),
}

# The 16 elements FastChem tracks via its solar file (H is implicit — the
# abundances are n_X/n_H). Elements in a broader classical-chemistry setup
# (e.g. Al/Ar/Co/Cr/Cu/Ge/Mn/Ne/Ni/Zn in ExoGibbs' 28-element setup) that
# are NOT in this set were never tracked during FastChem training and must
# be zeroed when mirroring the training contract.
FASTCHEM_TRACKED_ELEMENTS: frozenset[str] = frozenset(
    FASTCHEM_LODDERS_SOLAR_ABUNDANCES.keys()
)

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

# Gas-phase species for which VULCAN ships saturation-pressure data in
# build_atm.sp_sat. Anything outside this set cannot be a member of
# vulcan.runtime.condensation.condense_sp; the VULCAN runtime would raise
# IOError on startup.
VULCAN_SUPPORTED_CONDENSATE_SPECIES = frozenset(
    {"H2O", "NH3", "H2SO4", "S2", "S4", "S8", "C", "H2S"}
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

# Large-negative logit used to mask out padded keys before softmax in
# attention. Chosen so ``exp(-1e30)`` underflows to zero in float32 with
# headroom to spare while still being finite (avoiding ``-jnp.inf`` so
# downstream ``nan`` guards never trigger on fully-masked rows).
ATTN_MASK_NEG_INF = -1.0e30

# =========================================================================
# Numerical floors and tolerances
# =========================================================================

# Floor below which a standard-deviation or span of a normalized feature
# is treated as "effectively zero" and replaced by 1.0 to keep the
# normalization invertible. Also used as the rtol when checking that a
# per-layer column is vertically constant before reduction.
NORM_STD_FLOOR = 1.0e-8

# Floor used to stabilize divisions by a span-like quantity (gradient
# norm, pressure coordinate span, position-coord normalization).
NORM_SPAN_FLOOR = 1.0e-12

# Cutoff on the fitted log-standard/standard std of a global input below
# which the exporter treats the input as a fixed global and excises it
# from the public ExoJAX surface. Also used as the atol when rechecking
# that training data stayed within the fitted distribution.
NEAR_CONSTANT_STD_THRESHOLD = 1.0e-6

# Lower bound applied to eddy diffusivity before taking ``log10`` so
# ``Kzz = 0`` samples do not blow up the log-standard normalizer.
KZZ_LOG_FLOOR_CM2_S = 1.0e-30

# Tolerance used when validating that a set of mixing fractions sums to
# one (e.g., science-preset fractions in the generation config).
FRACTION_SUM_TOLERANCE = 1.0e-6

# =========================================================================
# Stellar spectrum defaults — wavelength binning
# =========================================================================

# Short-wavelength bin width in nm for the stellar flux grid.
DBIN1_NM_DEFAULT = 0.1

# Long-wavelength bin width in nm for the stellar flux grid.
DBIN2_NM_DEFAULT = 2.0

# Transition wavelength (nm) where the stellar flux binning switches
# from ``DBIN1_NM_DEFAULT`` to ``DBIN2_NM_DEFAULT``.
DBIN_12TRANS_NM_DEFAULT = 240.0

# =========================================================================
# Hyperparameter-tuning defaults
# =========================================================================

# Seed fed to the Optuna TPE sampler so successive tuning invocations
# explore the same trial sequence unless the user picks a different
# config-level seed. Kept separate from the config ``seed`` because the
# TPE proposer runs orthogonally to any per-trial training RNG.
TUNING_TPE_SEED = 123

# Early-stopping patience is computed from the trial epoch budget as
# ``clip(epochs // DIVISOR, MIN, MAX)``. These bounds keep very short
# trials from stopping too eagerly and very long trials from burning
# most of the budget in a dead run.
TUNING_EARLY_STOP_MIN_PATIENCE = 5
TUNING_EARLY_STOP_MAX_PATIENCE = 20
TUNING_EARLY_STOP_PATIENCE_DIVISOR = 5

# Default EMA decay used when a tuning trial enables EMA but the base
# config has no ``training.ema.decay`` value to inherit.
TUNING_EMA_DEFAULT_DECAY = 0.999

# Log-uniform sampling range for ``training.weight_decay`` inside a
# tuning trial.
TUNING_WEIGHT_DECAY_RANGE = (1.0e-6, 5.0e-3)

# =========================================================================
# ExoJAX-facing label lists
# =========================================================================

# Ordered global-input labels for FastChem ExoJAX wrappers.
FASTCHEM_GLOBAL_LABELS = list(FASTCHEM_CONDITIONING_INPUT_ORDER)

# Ordered global-input labels for VULCAN ExoJAX wrappers.
VULCAN_GLOBAL_LABELS = list(DEFAULT_REQUIRED_GLOBAL_INPUTS)

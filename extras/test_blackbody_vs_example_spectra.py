"""Compare official VULCAN example stellar spectra to blackbody reconstructions.

This script uses a single vendored CSV under ``assets/spectra`` that bundles
two official VULCAN example stellar spectra:

  - ``sun_gueymard``   : solar reference spectrum from Gueymard
  - ``hd189_moses11``  : default HD 189733 host-star spectrum used by VULCAN

For each case, the script:

1. loads the stellar-surface flux spectrum,
2. fits a single-temperature blackbody over a continuum-dominated window,
3. rescales both the example spectrum and the fitted blackbody to the planet
   using ``(R_star / a)^2``, and
4. saves a simple comparison plot.

Run from the repository root:

    python extras/test_blackbody_vs_example_spectra.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "assets"
    / "spectra"
    / "reference"
    / "official_vulcan_example_spectra.csv"
)
STYLE_PATH = PROJECT_ROOT / "extras" / "science.mplstyle"
DEFAULT_OUTPUT = PROJECT_ROOT / "extras" / "plots" / "blackbody_vs_example_spectra.png"

R_SUN_CM = 6.957e10
AU_CM = 1.495978707e13
PLANCK_H = 6.62607015e-27
LIGHT_C = 2.99792458e10
BOLTZMANN_K = 1.380649e-16

SPECTRUM_CONFIG = {
    "sun_gueymard": {
        "label": "Sun / Gueymard",
        "r_star_rsun": 1.0,
        "orbit_au": 1.0,
        "fit_window_nm": (300.0, 800.0),
        "plot_window_nm": (1.0, 800.0),
        "color": "tab:orange",
    },
    "hd189_moses11": {
        "label": "HD 189733 / Moses 2011",
        "r_star_rsun": 0.805,
        "orbit_au": 0.03142,
        "fit_window_nm": (300.0, 800.0),
        "plot_window_nm": (2.0, 800.0),
        "color": "tab:blue",
    },
}


def load_reference_spectra(data_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load bundled reference spectra grouped by spectrum identifier.

    Parameters
    ----------
    data_path : Path
        CSV containing ``spectrum_id``, wavelength, and stellar-surface flux.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping from spectrum identifier to sorted wavelength and flux arrays.
    """
    table = np.genfromtxt(data_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for spectrum_id in np.unique(table["spectrum_id"]):
        mask = table["spectrum_id"] == spectrum_id
        wavelength_nm = np.asarray(table["wavelength_nm"][mask], dtype=np.float64)
        surface_flux = np.asarray(table["surface_flux_erg_cm2_s_nm"][mask], dtype=np.float64)
        order = np.argsort(wavelength_nm)
        spectra[str(spectrum_id)] = (wavelength_nm[order], surface_flux[order])

    return spectra


def blackbody_surface_flux(wavelength_nm: np.ndarray, temperature_k: float) -> np.ndarray:
    """Return the hemispheric blackbody surface flux per nm.

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength grid in nanometers.
    temperature_k : float
        Blackbody temperature in kelvin.

    Returns
    -------
    np.ndarray
        Surface flux with units ``erg cm^-2 s^-1 nm^-1``.
    """
    wavelength_cm = wavelength_nm * 1.0e-7
    exponent = PLANCK_H * LIGHT_C / (wavelength_cm * BOLTZMANN_K * temperature_k)
    clipped_exponent = np.clip(exponent, a_min=None, a_max=700.0)

    denominator = np.expm1(clipped_exponent)
    specific_intensity = (2.0 * PLANCK_H * LIGHT_C**2) / (wavelength_cm**5 * denominator)
    surface_flux_per_cm = np.pi * specific_intensity
    surface_flux_per_nm = surface_flux_per_cm * 1.0e-7

    return np.where(exponent > 700.0, 0.0, surface_flux_per_nm)


def fit_blackbody_temperature(
    wavelength_nm: np.ndarray,
    surface_flux: np.ndarray,
    fit_window_nm: tuple[float, float],
) -> tuple[float, np.ndarray, float]:
    """Fit a blackbody temperature in log-flux space over a wavelength window.

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength grid in nanometers.
    surface_flux : np.ndarray
        Stellar-surface flux in ``erg cm^-2 s^-1 nm^-1``.
    fit_window_nm : tuple[float, float]
        Inclusive wavelength window used for the temperature fit.

    Returns
    -------
    tuple[float, np.ndarray, float]
        Best-fit temperature, full blackbody spectrum on the input grid, and
        root-mean-square error in dex over the fit window.
    """
    lower_nm, upper_nm = fit_window_nm
    fit_mask = (
        (wavelength_nm >= lower_nm)
        & (wavelength_nm <= upper_nm)
        & np.isfinite(surface_flux)
        & (surface_flux > 0.0)
    )
    if not np.any(fit_mask):
        raise ValueError(f"No positive flux samples found in fit window {fit_window_nm}.")

    temperature_grid = np.linspace(3000.0, 8000.0, 5001, dtype=np.float64)
    best_temperature = float(temperature_grid[0])
    best_error = np.inf
    best_model = blackbody_surface_flux(wavelength_nm, best_temperature)

    log_reference = np.log10(surface_flux[fit_mask])
    for temperature_k in temperature_grid:
        candidate_flux = blackbody_surface_flux(wavelength_nm, float(temperature_k))
        candidate_fit = np.clip(candidate_flux[fit_mask], a_min=1.0e-300, a_max=None)
        candidate_error = np.mean((np.log10(candidate_fit) - log_reference) ** 2)

        if candidate_error < best_error:
            best_error = candidate_error
            best_temperature = float(temperature_k)
            best_model = candidate_flux

    return best_temperature, best_model, float(np.sqrt(best_error))


def scale_to_planet(
    surface_flux: np.ndarray,
    *,
    r_star_rsun: float,
    orbit_au: float,
) -> np.ndarray:
    """Scale stellar-surface flux to the planetary orbit.

    Parameters
    ----------
    surface_flux : np.ndarray
        Stellar-surface flux in ``erg cm^-2 s^-1 nm^-1``.
    r_star_rsun : float
        Stellar radius in solar radii.
    orbit_au : float
        Orbital separation in astronomical units.

    Returns
    -------
    np.ndarray
        Flux incident at the planet in ``erg cm^-2 s^-1 nm^-1``.
    """
    dilution = (r_star_rsun * R_SUN_CM / (orbit_au * AU_CM)) ** 2
    return surface_flux * dilution


def build_comparison_plot(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    output_path: Path,
    show: bool,
) -> None:
    """Create and save the blackbody-vs-reference comparison figure.

    Parameters
    ----------
    spectra : dict[str, tuple[np.ndarray, np.ndarray]]
        Loaded reference spectra keyed by spectrum identifier.
    output_path : Path
        Destination path for the PNG plot.
    show : bool
        Whether to display the figure interactively after saving.

    Returns
    -------
    None
        The figure is written to disk and optionally shown.
    """
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        2,
        len(SPECTRUM_CONFIG),
        figsize=(6.6 * len(SPECTRUM_CONFIG), 7.6),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.2]},
        constrained_layout=True,
    )

    if len(SPECTRUM_CONFIG) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, (spectrum_id, metadata) in enumerate(SPECTRUM_CONFIG.items()):
        wavelength_nm, surface_flux = spectra[spectrum_id]

        best_temperature_k, blackbody_surface, rms_error_dex = fit_blackbody_temperature(
            wavelength_nm,
            surface_flux,
            metadata["fit_window_nm"],
        )

        example_planet_flux = scale_to_planet(
            surface_flux,
            r_star_rsun=float(metadata["r_star_rsun"]),
            orbit_au=float(metadata["orbit_au"]),
        )
        blackbody_planet_flux = scale_to_planet(
            blackbody_surface,
            r_star_rsun=float(metadata["r_star_rsun"]),
            orbit_au=float(metadata["orbit_au"]),
        )

        plot_min_nm, plot_max_nm = metadata["plot_window_nm"]
        plot_mask = (
            (wavelength_nm >= plot_min_nm)
            & (wavelength_nm <= plot_max_nm)
            & (example_planet_flux > 0.0)
            & (blackbody_planet_flux > 0.0)
        )
        ratio = example_planet_flux[plot_mask] / blackbody_planet_flux[plot_mask]

        flux_ax = axes[0, column]
        ratio_ax = axes[1, column]

        flux_ax.plot(
            wavelength_nm[plot_mask],
            example_planet_flux[plot_mask],
            color=metadata["color"],
            lw=2.2,
            label="Example spectrum",
        )
        flux_ax.plot(
            wavelength_nm[plot_mask],
            blackbody_planet_flux[plot_mask],
            color="black",
            lw=1.8,
            ls="--",
            label=f"Blackbody fit ({best_temperature_k:.0f} K)",
        )
        flux_ax.axvspan(
            metadata["fit_window_nm"][0],
            metadata["fit_window_nm"][1],
            color="0.85",
            alpha=0.25,
            zorder=0,
        )
        flux_ax.set_xscale("log")
        flux_ax.set_yscale("log")
        flux_floor = float(np.min(example_planet_flux[plot_mask]))
        flux_ceiling = float(np.max(example_planet_flux[plot_mask]))
        flux_ax.set_ylim(0.3 * flux_floor, 3.0 * flux_ceiling)
        flux_ax.set_title(metadata["label"])
        flux_ax.set_ylabel(r"Flux At Planet [erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$]")
        flux_ax.legend(loc="lower left")
        flux_ax.text(
            0.03,
            0.97,
            (
                f"R* = {metadata['r_star_rsun']:.3f} R$_\\odot$\n"
                f"a = {metadata['orbit_au']:.5f} AU\n"
                f"Fit RMS = {rms_error_dex:.3f} dex"
            ),
            transform=flux_ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )

        ratio_ax.plot(wavelength_nm[plot_mask], ratio, color="0.2", lw=1.6)
        ratio_ax.axhline(1.0, color="0.55", lw=1.0, ls=":")
        ratio_ax.axvspan(
            metadata["fit_window_nm"][0],
            metadata["fit_window_nm"][1],
            color="0.85",
            alpha=0.25,
            zorder=0,
        )
        ratio_ax.set_xscale("log")
        ratio_ax.set_yscale("log")
        ratio_ax.set_ylim(1.0e-2, 1.0e2)
        ratio_ax.set_xlabel("Wavelength [nm]")
        ratio_ax.set_ylabel("Example / Blackbody")

        print(
            f"{spectrum_id}: fitted blackbody T = {best_temperature_k:.1f} K, "
            f"fit RMS = {rms_error_dex:.4f} dex"
        )

    figure.suptitle("Blackbody Reconstructions vs Official VULCAN Example Spectra", fontsize=15)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path for the comparison figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the blackbody-vs-reference comparison."""
    warnings.filterwarnings(
        "ignore",
        message="overflow encountered in power",
        category=RuntimeWarning,
        module=r"matplotlib\.scale",
    )
    args = parse_args()
    spectra = load_reference_spectra(DATA_PATH)
    build_comparison_plot(spectra, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()

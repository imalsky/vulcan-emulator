from __future__ import annotations

import numpy as np
from src.data_generation.spectrum import (
    generate_wasp39b_template,
    load_spectrum_manifest,
    pack_spectrum_tokens,
    read_vulcan_spectrum_txt,
    sanitize_spectrum_arrays,
    save_spectrum_manifest,
    vulcan_wavelength_bins,
    write_vulcan_spectrum_txt,
    _flux_conserving_rebin,
    _vulcan_bin_edges,
)


def test_spectrum_roundtrip_and_manifest(tmp_path):
    record = generate_wasp39b_template(num_points=128)
    out_file = tmp_path / "sflux.txt"
    write_vulcan_spectrum_txt(record, out_file)
    loaded = read_vulcan_spectrum_txt(out_file)
    assert loaded.wavelength_nm.size == record.wavelength_nm.size
    assert np.allclose(loaded.flux_erg_cm2_s_nm, record.flux_erg_cm2_s_nm)

    manifest = save_spectrum_manifest([record], tmp_path / "library")
    restored = load_spectrum_manifest(manifest)
    assert record.name in restored
    assert restored[record.name].flux_erg_cm2_s_nm.shape == record.flux_erg_cm2_s_nm.shape


def test_sanitize_spectrum_arrays_sorts_deduplicates_and_clips():
    wavelength_nm = np.array([250.0, 150.0, np.nan, 200.0, 150.0, 900.0], dtype=np.float64)
    flux = np.array([3.0, -1.0, 9.0, 4.0, 5.0, 8.0], dtype=np.float64)

    clean_wavelength_nm, clean_flux = sanitize_spectrum_arrays(
        wavelength_nm,
        flux,
        wavelength_min_nm=100.0,
        wavelength_max_nm=400.0,
    )

    np.testing.assert_allclose(clean_wavelength_nm, np.array([150.0, 200.0, 250.0]))
    np.testing.assert_allclose(clean_flux, np.array([2.5, 4.0, 3.0]))


def test_vulcan_wavelength_bins_default():
    bins = vulcan_wavelength_bins()
    assert len(bins) == 2610
    assert bins[0] == 2.0
    assert bins[-1] == 698.0
    # Check spacing in UV region (< 240 nm)
    uv = bins[bins < 240.0]
    np.testing.assert_allclose(np.diff(uv), 0.1, atol=1e-10)
    # Check spacing in visible region (>= 240 nm)
    vis = bins[bins >= 240.0]
    np.testing.assert_allclose(np.diff(vis), 2.0, atol=1e-10)


def test_vulcan_wavelength_bins_custom_range():
    bins = vulcan_wavelength_bins(wavelength_min_nm=100.0, wavelength_max_nm=400.0)
    assert bins[0] == 100.0
    assert bins[-1] < 400.0
    uv = bins[bins < 240.0]
    vis = bins[bins >= 240.0]
    assert len(uv) == len(np.arange(100.0, 240.0, 0.1))
    assert len(vis) == len(np.arange(240.0, 400.0, 2.0))


def test_flux_conserving_rebin_preserves_integral():
    """Rebinning onto arbitrary edges must conserve the total flux integral."""
    wavelength_nm = np.linspace(100.0, 500.0, 2000, dtype=np.float64)
    flux = np.exp(-((wavelength_nm - 300.0) ** 2) / (2 * 50.0**2))

    edges = np.linspace(100.0, 500.0, 51, dtype=np.float64)
    rebinned = _flux_conserving_rebin(wavelength_nm, flux, edges)

    centers = 0.5 * (edges[:-1] + edges[1:])
    original_integral = np.trapz(flux, wavelength_nm)
    rebinned_integral = np.trapz(rebinned, centers)
    np.testing.assert_allclose(rebinned_integral, original_integral, rtol=0.02)


def test_pack_spectrum_tokens_vulcan_grid():
    """Pack a dense spectrum onto the VULCAN grid and verify output shape/mask."""
    wavelength_nm = np.linspace(1.0, 800.0, 5000, dtype=np.float64)
    flux = 1.0e3 / np.sqrt(wavelength_nm)

    packed_w, packed_f, packed_mask = pack_spectrum_tokens(
        wavelength_nm,
        flux,
        wavelength_min_nm=2.0,
        wavelength_max_nm=700.0,
        max_tokens=2610,
        dbin1_nm=0.1,
        dbin2_nm=2.0,
        dbin_12trans_nm=240.0,
    )

    assert packed_w.shape == (2610,)
    assert packed_f.shape == (2610,)
    assert packed_mask.shape == (2610,)
    assert packed_mask.dtype == np.bool_
    assert np.all(packed_mask)  # all 2610 bins are valid
    assert packed_w[0] == np.float32(2.0)
    # Verify wavelengths are monotonically increasing
    valid_w = packed_w[packed_mask].astype(np.float64)
    assert np.all(np.diff(valid_w) > 0.0)


def test_pack_spectrum_tokens_flux_conservation():
    """Total flux integral should be approximately conserved after rebinning."""
    wavelength_nm = np.geomspace(2.0, 700.0, 5000, dtype=np.float64)
    flux = 1.0e3 / np.sqrt(wavelength_nm)

    packed_w, packed_f, packed_mask = pack_spectrum_tokens(
        wavelength_nm,
        flux,
        wavelength_min_nm=2.0,
        wavelength_max_nm=700.0,
        max_tokens=2610,
    )

    valid_f = packed_f[packed_mask].astype(np.float64)
    # Rebinned values are average flux densities per bin. The correct
    # integral is sum(flux_density * bin_width), not trapz over centres.
    centers = vulcan_wavelength_bins()
    edges = _vulcan_bin_edges(centers, 0.1, 2.0, 240.0)
    widths = np.diff(edges)
    original_integral = np.trapz(flux, wavelength_nm)
    rebinned_integral = np.sum(valid_f * widths)
    assert rebinned_integral > 0.0
    assert abs(rebinned_integral - original_integral) / original_integral < 0.02


def test_pack_spectrum_tokens_max_tokens_too_small():
    """Should raise when max_tokens is smaller than the VULCAN bin count."""
    wavelength_nm = np.linspace(2.0, 700.0, 100, dtype=np.float64)
    flux = np.ones_like(wavelength_nm)

    try:
        pack_spectrum_tokens(
            wavelength_nm,
            flux,
            wavelength_min_nm=2.0,
            wavelength_max_nm=700.0,
            max_tokens=100,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "2610" in str(e)

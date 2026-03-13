from __future__ import annotations

import numpy as np

from src.spectrum import (
    fixed_wavelength_grid,
    generate_wasp39b_template,
    load_spectrum_manifest,
    read_vulcan_spectrum_txt,
    resample_spectrum,
    save_spectrum_manifest,
    write_vulcan_spectrum_txt,
)


def test_spectrum_roundtrip_and_manifest(tmp_path):
    record = generate_wasp39b_template(num_points=128)
    out_file = tmp_path / "sflux.txt"
    write_vulcan_spectrum_txt(record, out_file)
    loaded = read_vulcan_spectrum_txt(out_file)
    assert loaded.wavelength_nm.size == record.wavelength_nm.size
    assert np.allclose(loaded.flux_erg_cm2_s_nm, record.flux_erg_cm2_s_nm)

    grid = fixed_wavelength_grid(100.0, 700.0, 32)
    binned = resample_spectrum(record, grid)
    assert binned.shape == (32,)
    assert np.all(np.isfinite(binned))
    assert np.all(binned > 0.0)

    manifest = save_spectrum_manifest([record], tmp_path / "library")
    restored = load_spectrum_manifest(manifest)
    assert record.name in restored
    assert restored[record.name].flux_erg_cm2_s_nm.shape == record.flux_erg_cm2_s_nm.shape

# Roth Utilities

This directory now contains only standalone helper scripts.

The maintained Roth workflow for this repository is not these scripts. Use:

- `config/config.json` `roth_sampler`
- `python src/main.py --gen`
- `python extras/plot_roth_profile.py ...` to preview one sampled column

Standalone utilities kept here:

- `convert_roth_grid.py`: convert Roth `PTprofiles/*.dat` files into one HDF5 dataset
- `subsample_hdf5_shards.py`: randomly subsample one HDF5 dataset and split it into shards

These scripts are ad hoc research utilities with in-file configuration or
standalone CLI arguments. They are not part of the strict emulator contract.

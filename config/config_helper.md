# Config Helper

Use the task-specific helper that matches the config you are editing:

- `equilibrium_only_config.json`:
  see `config/equilibrium_only_config_helper.md`
- `full_vulcan_config.json`:
  see `config/full_vulcan_config_helper.md`

Shared rules:

- Supported tasks are only `equilibrium_only` and `full_vulcan`.
- The fixed elemental input order is internal and must not be set in the config:
  `["He_H", "C_H", "O_H", "N_H", "S_H"]`
- These elemental values use the FastChem-native hydrogen-normalized number abundance `n_X / n_H` and are still materialized internally for FastChem/VULCAN raw generation.
- The supported learned chemistry contract for both model families is now the ratio-global triplet `metallicity_log10`, `c_to_o`, and `s_to_o`.
- Supported activation names for both model families are:
  `relu`, `gelu`, `silu`, `tanh`, `elu`, `selu`, `softplus`, `leaky_relu`
- The default activation for both model families is `leaky_relu`.
- Both model families support a configurable `dropout_rate`, which defaults to `0.05`.
- Both model families support `training.early_stopping_patience`, which defaults to `30`.
- There is no supported trajectory, timestep-control, or live-sampling mode.

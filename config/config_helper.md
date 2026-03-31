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
- These elemental values use the FastChem-native hydrogen-normalized number abundance `n_X / n_H`.
- Supported activation names for both model families are:
  `relu`, `gelu`, `silu`, `tanh`, `elu`, `selu`, `softplus`, `leaky_relu`
- The default activation for both model families is `leaky_relu`.
- Both model families support a configurable `dropout_rate`, which defaults to `0.05`.
- There is no supported trajectory, timestep-control, or live-sampling mode.

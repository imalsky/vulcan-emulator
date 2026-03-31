# Equilibrium-Only Config Helper

Use `equilibrium_only_config.json` when you want the emulator to predict the
final FastChem equilibrium abundances for a full pressure-temperature column.

This mode is for local equilibrium chemistry only:

- output target: FastChem equilibrium abundances
- model family: FiLM-conditioned per-level MLP
- vertical coupling: none between neighboring layers
- required profile inputs at inference time: pressure, temperature, elemental abundances

## Top-Level Layout

The equilibrium config contains these sections:

- `task`
- `paths`
- `data_spec`
- `sampling`
- `temperature_profiles`
- `generation`
- `normalization`
- `training`
- `equilibrium_only`

## `task`

- `kind`

Set `task.kind = "equilibrium_only"`.

## `paths`

These control where the pipeline reads and writes data:

- `raw_root`: raw generated runs
- `processed_root`: processed train/val/test arrays
- `checkpoints_root`: saved checkpoints and metrics
- `vulcan_source_root`: location of the VULCAN/FastChem source tree used during generation

## `data_spec`

These define the species layout used throughout the pipeline:

- `state_species`: full species ordering stored in raw and processed data
- `output_species`: species ordering predicted by the emulator

Do not add `required_global_inputs` or `element_input_order` here. Those are
derived internally.

## `sampling`

These set the synthetic atmosphere parameter ranges:

- `num_levels`: number of vertical layers
- `pressure_top_bar`: pressure at the top layer
- `pressure_bottom_bar`: pressure at the bottom layer
- `temperature_range_k`: allowed temperature range
- `metallicity_log10_range`: metallicity sampler range
- `c_to_o_range`: carbon-to-oxygen ratio range
- `s_to_o_range`: sulfur-to-oxygen ratio range

The public chemistry input is still passed as explicit elemental abundances,
but the current generator samples chemistry from metallicity, C/O, and S/O and
then converts that to the fixed internal elemental channels.

## `temperature_profiles`

This controls how temperature profiles are chosen:

- `source_mode`: `analytic`, `pt_library`, or `mixed`
- `validation`: hard temperature sanity bounds
- `filters`: optional PT-library filters
- `analytic_sampler`: parameters for the analytic profile generator
- `data_glob`: PT-library file pattern
- `analytic_probability`: only used when `source_mode = "mixed"`

Use this section to decide whether equilibrium data should be drawn from an
analytic sampler, from the PT library, or from both.

## `generation`

This controls raw-dataset creation:

- `mode`: `synthetic` or `vulcan`
- `num_runs`: requested successful runs
- `seed`: generation seed
- `overwrite`: regenerate even if outputs already exist
- `reuse_raw_if_present`: skip generation when the raw dataset already exists
- `parallel_workers`: generation worker count

For equilibrium runs, generation stores only successful final equilibrium
outputs. There is no timestep or trajectory contract.

## `normalization`

This controls how inputs and targets are scaled before training:

- `split`: train/val/test fractions and split seed
- `state_floor`: lower bound applied before log scaling chemistry targets
- `target_method`: target scaling method
- `sequence_methods`: scaling per sequence feature
- `global_methods`: scaling per global feature

Supported normalization methods are:

- `standard`
- `log-standard`
- `none`

Typical chemistry scaling is `log-standard`, which is `log10` followed by
z-score normalization.

## `training`

This controls optimization:

- `seed`
- `batch_size`
- `epochs`
- `learning_rate`
- `min_lr`
- `warmup_epochs`
- `scheduler`
- `weight_decay`
- `gradient_clip`
- `loss`

`loss` contains:

- `lambda_z`: weight on normalized-space MSE
- `lambda_phys`: weight on denormalized log10-space MSE

Supported schedulers are:

- `reduce_on_plateau`
- `cosine`

## `equilibrium_only.model`

This defines the equilibrium MLP:

- `d_hidden`: hidden width
- `num_hidden_layers`: number of hidden blocks
- `conditioning_hidden_dim`: hidden size of the FiLM conditioning network
- `film_clamp`: clamp applied to FiLM modulation values
- `activation`: activation function, default `leaky_relu`
- `dropout_rate`: hidden-layer dropout probability, default `0.05`

Supported activations are:

- `relu`
- `gelu`
- `silu`
- `tanh`
- `elu`
- `selu`
- `softplus`
- `leaky_relu`

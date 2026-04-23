# Training diary — VULCAN emulator

Short log of what each training run has taught us about sizing, regularization,
and the relationship between sweep proxies and full-length deployment loss.
One block per run; append new blocks at the bottom.

Headline metric: `best_val_combined_loss` (EMA shadow weights on val split).
Reference test metrics: `test.combined_loss`, `test.mae_log10`.

---

## 2026-04-16 — baseline arch, 400 ep
**Config**: d_model=256, nhead=8, L=6, dim_ff=1024, cond_hidden=256, divisor=2,
norm=rmsnorm, ffn=dense, qk_norm=off, zero_init_film=off, silu, dropout=0,
wd=1e-4, ema=on (0.999).
**Result**: best val **0.0304** (ep 396). Test combined 0.0914, mae_log10 0.0292.
Final train 0.0064, final val 0.0305 → ~5× generalization gap.
**Takeaway**: our reference. 400 epochs + 40k train samples + this arch reliably
converges. Params: 5.57M. Gap width says data, not capacity, is the ceiling.

## 2026-04-17 — Optuna sweep v1 (100 trials × 100 epochs, MedianPruner)
**Config**: TPESampler over sizing + formulation + regularization; pruner
`MedianPruner(n_startup=5, n_warmup=10)`. Winner trial #47: d_model=192,
nhead=12, L=4, dim_ff=768, cond_hidden=1024, divisor=1, norm=layernorm,
ffn=swiglu, qk_norm=on, zero_init_film=on, wd=5.6e-4, dropout=0.026,
val **0.0473** @ ep 100.
**Result**: 12/100 complete, 88/100 pruned — mostly before epoch 30.
**Takeaway**: selection biased toward fast-early-learners. 100 epochs is too
short a proxy for our 400-epoch deployment. Incumbent was never enqueued so
the "winner" was never benchmarked against 0.0304.

## 2026-04-17 — trial #47 arch promoted to 400 ep
**Config**: trial #47 params, dropout forced to 0.0, wd forced to 5e-4.
ema=on (inherited from base).
**Result**: final val **0.0339**, final train 0.0067. Regression of +0.003 vs
baseline.
**Takeaway**: architecture ranking is NOT invariant to training length. Trial
#47 learned faster early (0.047 at 100 ep vs baseline's 0.054) but plateaued
higher. Train losses basically match baseline — regression is generalization-
gap widening, consistent with smaller attention stack (L: 6→4, d: 256→192) and
5× weight decay forcing more memorization through the FiLM path.

## 2026-04-17 — hybrid arch (baseline sizing + sweep formulation), 400 ep
**Config**: d_model=256, L=6, dim_ff=1024, cond=256, **divisor=1** (config
retained 1 at launch; retrain_log had said 2 — missed at review),
wd=1e-4, ema=on. Formulation knobs from sweep: norm=layernorm, ffn=swiglu,
qk_norm=on, zero_init_film=on, silu, dropout=0.
**Result**: best val **0.0395** at ep 137; early-stopped at ep 177
(40 ep without improvement). ep 100 val ~0.045 (*faster* than baseline's
0.054). Train kept falling to ~0.009 while val plateaued near 0.040.
**Takeaway**: same fast-early/plateau-higher signature as trial #47. The
sweep-era formulation wins were judged "neutral-or-good" on the same
100-ep proxy that mis-ranked trial #47 — we re-made the proxy mistake.
Primary suspect: `zero_init_film=true` starves composition at init, and
narrow `cond_hidden=256` can't catch up fast through L=6 while the larger
FFN (swiglu + divisor=1) memorizes T,P-only. Gap widened, not narrowed.

## 2026-04-17 — pure-baseline revert, 400 ep (in progress)
**Config**: all four formulation knobs reverted (norm=rmsnorm, qk_norm=off,
ffn=dense, zero_init_film=off); divisor=2; sizing d=256/L=6/dim_ff=1024/
cond=256; wd=1e-4; ema on. The config **model** block now matches the
April-16 run field-for-field.
**Result (live @ ep 390 of 400)**: best plain-weights val **~0.0334** at
ep 386 (train ~0.0048). LR has already stepped down twice to 1.875e-5
(plateau scheduler). Train-val gap ~6.7×, vs baseline's 5×.
**Takeaway — loss mechanism didn't change numerically**. Pulled baseline
config out of `best_exported.npz`: `lambda_z=1.0, lambda_log10_mae=0.25,
ema={enabled:true, decay:0.999}` — byte-identical to today's config.
What *did* change: the stored baseline's `model` block doesn't contain
`norm_type`/`use_qk_norm`/`ffn_type`/`zero_init_film` at all — those
fields were added to the schema *after* April 16. So baseline used
whatever the model code's defaults were on that day; if layer code,
normalization stats, or data-loader behavior drifted since then, we'd
see ~0.003 higher val without any config diff — exactly what we see.

**Decision**: accept **~0.033–0.034** (final number pending run completion
and EMA-weighted test eval) as the **new reference baseline** for
comparison. 0.0304 is no longer reproducible with current code without
bisecting back, and that's not worth it right now. All future ablations
compare to this run's `metrics.json`, not to the April-16 checkpoint.

## Sweep v2 (to follow baseline revert)
**Methodology changes in `src/tuning/__main__.py`**:
- HyperbandPruner (min=30, max=300 ep, reduction=3) replaces MedianPruner.
- 60 trials × up to 300 ep (was 100 × 100).
- Search space: size + regularization stay tunable; the four formulation
  knobs (`norm_type`, `use_qk_norm`, `ffn_type`, `zero_init_film`) **must
  be made tunable again** — they were fixed to sweep values under the
  (now-disproven) assumption they were neutral. Re-check
  `_SAMPLING_SEARCH_SPACE` / `_FIXED_FORMULATION` in the sweep module
  before launching. d_model ∈ {192, 256, 384}; L ∈ [4, 8]; divisor ∈
  {1, 2}; wd ∈ [1e-6, 5e-3]; new: `ema_enabled ∈ {true, false}`.
- Incumbent arch auto-enqueued as trial #0 (`INCUMBENT_PARAMS`) → every
  sweep has an apples-to-apples benchmark against the new ~0.033 baseline.

## Sweep v2 (to follow baseline revert)
**Methodology changes in `src/tuning/__main__.py`**:
- HyperbandPruner (min=30, max=300 ep, reduction=3) replaces MedianPruner.
- 60 trials × up to 300 ep (was 100 × 100).
- Search space: size + regularization stay tunable; the four formulation
  knobs (`norm_type`, `use_qk_norm`, `ffn_type`, `zero_init_film`) are now
  tunable again after the hybrid-run failure (previously fixed to sweep
  values on the false assumption they were neutral). d_model ∈ {192, 256,
  384}; L ∈ [4, 8]; output_head_divisor ∈ {1, 2}; wd ∈ [1e-6, 5e-3];
  new: `ema_enabled ∈ {true, false}`.
- Incumbent arch auto-enqueued as trial #0 (`INCUMBENT_PARAMS`) so every
  sweep has an apples-to-apples benchmark.

---

## 2026-04-18 — loss: MAE → Huber in log10 space (config: fastchem_no_condensation.json)
**Change**: replaced `MAE_log10` with `Huber_log10` as the physical-space loss
term. Renamed `training.loss.lambda_log10_mae` → `lambda_log10_huber`; added
`training.loss.huber_delta_log10` (dex, transition point; default 0.1). Form:
`0.5*r^2/delta` for `|r| <= delta`, `|r| - 0.5*delta` otherwise, where
`r = log10(pred_VMR / target_VMR)`.
**Motivation**: tail-species generalization gap (val 0.033 vs train 0.007) is
driven by rare-regime residuals; MAE's constant gradient wastes capacity
correcting already-accurate common species. Huber gives quadratic gradient
near zero (smoother on common species) and linear in the tail (outlier-
robust, like MAE). For `|r| > delta` Huber = `|r| - delta/2`, so tail-regime
values shift by ~`delta/2 = 0.05` vs old `mae_log10` but remain directly
comparable in trend.
**Compatibility**: metric renamed `mae_log10` → `huber_log10` in
`metrics.json` and Optuna CSV; older `history.json` files keep the old key.
`lambda` kept at 0.25 pending a retune.

## 2026-04-21 — NUTS retrieval fix: widen abundance box (config: fastchem_analytic_500k.json)
**Change**: widened all abundance sampling ranges to ±2 dex around AAS solar
(was ±1 dex for C/O/N/S, ±0.4 dex for He). Switched He to log-uniform
sampling with log-standard normalization (was linear + standard), matching
C/O/N/S.
**Motivation**: the `comparison.ipynb` diagnostics dump
(`diagnostics/comparison_20260421T183951Z`) shows NUTS on the emulator parks
its chain 0.2–2.4 dex outside the previous training box on O, N, He, C —
O_H posterior sits at +2.2 dex beyond the upper bound. Outside the box the
emulator extrapolates (`log10(ml/cl)` reaches 14 dex on atomic O); NUTS
adapts step size to that garbage gradient field and the chain stalls (14–15%
divergences, `accept_prob = 0.26`). Widening to ±2 dex gives ≥1 dex cushion
past the 3σ tail of a `Normal(solar, 0.4)` prior, which covers the observed
NUTS excursion. He asymmetry (linear sampling + standard normalization, while
every other element was log/log-standard) is the most likely cause of
`log_He_H` being pinned with `std = 0.003` dex; fixing the geometry should
let the He axis move.

## 2026-04-22 — loss: Huber → MAE reverted, loss type now configurable
**Change**: Huber regressed model quality meaningfully vs the earlier MAE
formulation. Restored MAE in log10 space as the default, and made the
choice explicit and first-class: `training.loss.type ∈ {"mae", "huber"}`.
Per-type required keys — MAE: `lambda_z`, `lambda_log10_mae`; Huber:
`lambda_z`, `lambda_log10_huber`, `huber_delta_log10`. All shipped configs
switched to `type: "mae"`.
**Tuning**: Optuna search space now samples `loss_type` as a categorical, and
samples `huber_delta_log10 ∈ [0.02, 0.3]` log-uniform only on Huber trials.
**Metrics**: unified physical-space metric key to `log10_loss` (replaces
`mae_log10` / `huber_log10`) across `metrics.json`, history, and Optuna CSV
so mixed-type studies share a column schema. Older history files retain their
type-specific keys and are not forward-compatible with this column.
**Validator**: `training.loss.type` is required (no default) — every config
must state loss intent explicitly after the known quality gap between the two
forms.
**Infra**: the `_notes.retrain_log` block was also removed from JSON configs;
this file is now the single source of truth for training-change history. New
run-level facts go here as dated sections; config JSON stays just config.

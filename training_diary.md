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

## Planned next — pure-baseline revert, 400 ep
**Config change**: revert all four formulation knobs to pre-sweep values:
norm_type layernorm→rmsnorm, use_qk_norm true→false, ffn_type swiglu→dense,
zero_init_film true→false. Also output_head_divisor 1→2 (missed last
revert). Sizing stays baseline (d=256, L=6, dim_ff=1024, cond=256),
wd=1e-4, ema on.
**Expectation**: val ≈ 0.0304 → the four formulation knobs *together* are
the regression; next sweep ablates them individually. val > 0.0304 → a
non-arch change drifted in the codebase since April 16; bisect commits.

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

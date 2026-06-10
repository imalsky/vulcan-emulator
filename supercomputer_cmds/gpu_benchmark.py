"""Standalone GPU throughput benchmark for VULCAN-JAX.

This script integrates a batch of *different* atmospheric profiles to steady
state in ONE vmapped device call (`OuterLoop.run_batch`, the vmap-across-
profiles path). That batched call is the workload a GPU accelerates: every
profile in the batch is integrated in parallel lanes on the device. The script
sweeps a few batch sizes and reports wall time, time per profile, and
throughput (profiles/sec), separating the first (compile + run) call from a
second cached call.

It is fully standalone: it imports only `vulcan_jax`, the stdlib, NumPy, and
JAX. It needs no sibling repos, no `../VULCAN-master/`, and no FastChem — it
uses `ini_mix='const_mix'` so initial abundances come from a config dict
(no external subprocess), and `atm_type='isothermal'` so each profile is
defined entirely by config (no atmosphere file). Photochemistry is off
(`run_batch` does not support it).

------------------------------------------------------------------------------
RUN IT
------------------------------------------------------------------------------
After `pip install vulcan-jax` (nothing else):

    # CPU (default backend on a laptop). It will say so up front.
    python examples/gpu_benchmark.py

    # Force the GPU backend (errors early if no GPU is visible to JAX):
    JAX_PLATFORM_NAME=gpu python examples/gpu_benchmark.py

    # Pick the batch sizes to sweep (default: 1 8 32 128):
    python examples/gpu_benchmark.py --batches 1 8 32 128

    # Smaller / faster smoke test:
    python examples/gpu_benchmark.py --batches 1 4 --count-max 60 --nz 60

On the GH200 / HPC (an H100 GPU), in a job that has `vulcan-jax` installed and
a visible NVIDIA GPU:

    export JAX_ENABLE_X64=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    JAX_PLATFORM_NAME=gpu python examples/gpu_benchmark.py --batches 8 32 128 256

The "profiles/sec" at the largest batch size is the number to watch: the GPU
amortizes a fixed per-call overhead across the whole batch, so throughput
should climb with batch size until the device saturates. On CPU it will be
roughly flat (vmap lanes time-share a few cores).
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config: a fast, fully-standalone, photo-off batched regime.
# ---------------------------------------------------------------------------
def build_cfg(nz: int, count_max: int):
    """A standalone config: const_mix init (no FastChem), isothermal T-P
    (no atm file), photochemistry off (run_batch requires this).

    `nz` and `count_max` are kept small so the whole sweep finishes quickly;
    raise them for a heavier device-saturation test.
    """
    import vulcan_jax

    return vulcan_jax.make_config(
        ini_mix="const_mix",  # init abundances from the const_mix dict -> no FastChem subprocess
        atm_type="isothermal",  # T-P fully set by Tiso -> no atmosphere file read
        Kzz_prof="Pfunc",  # pressure-analytic eddy diffusion -> no atm file
        use_photo=False,  # run_batch does not support photochemistry
        use_ion=False,
        use_condense=False,
        nz=nz,
        count_max=count_max,
        count_min=1,
        use_print_prog=False,
        use_live_plot=False,
        use_live_flux=False,
        use_save_movie=False,
        use_flux_movie=False,
        use_chunked_runner=False,
    )


def build_profiles(n: int, cfg, seed: int = 0):
    """Build `n` genuinely-different RunStates by perturbing the isothermal
    temperature per profile (a few percent around the base `Tiso`).

    Different `Tiso` gives each profile a different n_0 / Tco / initial
    state, so the batch is a real heterogeneous workload (not n copies of one
    atmosphere). const_mix + isothermal means each build is cheap host-side
    work (no FastChem, no file read).

    `skip_chem_warmup=True` drops the per-profile single-profile chem-RHS JIT
    warmup: `run_batch` compiles its own *batched* (vmapped) chem RHS, so the
    single-profile compile this would trigger is never used. Skipping it removes
    a one-time ~30-40 s compile from the host build (it instead shows up once,
    correctly, in the first 'cold' run_batch call).
    """
    import copy

    import vulcan_jax

    base_T = float(getattr(cfg, "Tiso", 1000.0))
    rng = np.random.default_rng(seed)
    # +/- ~5% temperature spread across the batch (deterministic for n==1).
    if n == 1:
        temps = [base_T]
    else:
        temps = base_T * (1.0 + 0.05 * np.linspace(-1.0, 1.0, n))
        # jitter so adjacent profiles aren't on a perfect line
        temps = temps * (1.0 + 0.01 * rng.standard_normal(n))

    states = []
    for T in temps:
        c = copy.copy(cfg)
        c.Tiso = float(T)
        states.append(vulcan_jax.RunState.with_pre_loop_setup(c, skip_chem_warmup=True))
    return states


def build_integ(cfg):
    """Construct the OuterLoop runner (Ros2 solver + output sink)."""
    import vulcan_jax.legacy_io as legacy_io
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop

    return outer_loop.OuterLoop(op_jax.Ros2JAX(), legacy_io.Output(cfg=cfg), cfg=cfg)


def stack_batch(integ, run_states):
    """prepare_runstate each profile, then stack into one batched
    (init_state, atm_static). Returns the stacked pair plus the count.
    """
    import vulcan_jax.outer_loop as outer_loop

    init_states, atm_statics = [], []
    for rs in run_states:
        init_state, atm_static = integ.prepare_runstate(rs)
        init_states.append(init_state)
        atm_statics.append(atm_static)
    return (
        outer_loop.stack_integ_states(init_states),
        outer_loop.stack_atm_statics(atm_statics),
    )


def print_backend_banner():
    """Print the JAX backend + devices and a clear CPU-fallback note."""
    import jax

    backend = jax.default_backend()
    devices = jax.devices()
    print("=" * 70)
    print("VULCAN-JAX GPU throughput benchmark")
    print("=" * 70)
    try:
        import vulcan_jax

        print(f"vulcan-jax version : {vulcan_jax.__version__}")
    except Exception:
        pass
    print(f"jax version        : {jax.__version__}")
    print(f"jax backend        : {backend}")
    print(f"jax devices        : {devices}")
    on_gpu = backend in ("gpu", "cuda", "rocm")
    if not on_gpu:
        print("")
        print(">> Running on CPU. This benchmark measures the vmap-across-profiles")
        print(">> path that the GPU is for; on CPU the lanes time-share a few cores,")
        print(">> so throughput will be roughly flat in batch size.")
        print(">> To use a GPU:  JAX_PLATFORM_NAME=gpu python examples/gpu_benchmark.py")
    else:
        print("")
        print(f">> Running on the {backend.upper()} backend. Throughput should climb")
        print(">> with batch size until the device saturates.")
    print("=" * 70)
    return on_gpu


def benchmark_one(integ, run_states):
    """Run `run_batch` on a stacked batch twice: once cold (compile + run),
    once warm (cached). Returns (n, cold_s, warm_s, n_converged).
    """
    import jax

    n = len(run_states)
    states_b, atm_b = stack_batch(integ, run_states)

    # Cold call: includes XLA compilation for this (nz, batch-size) shape.
    t0 = time.perf_counter()
    out = integ.run_batch(states_b, atm_b)
    jax.block_until_ready(out)
    cold_s = time.perf_counter() - t0

    # Warm call: same shapes, compilation cached.
    t0 = time.perf_counter()
    out = integ.run_batch(states_b, atm_b)
    jax.block_until_ready(out)
    warm_s = time.perf_counter() - t0

    # Per-lane termination_reason: 1 converged, 2 runtime, 3 step-count,
    # 4 stalled, 5 non-finite. With a small count_max most lanes hit 3
    # (step cap) rather than 1 — that's expected and fine for a throughput
    # benchmark; we just report how many actually converged.
    import vulcan_jax.outer_loop as outer_loop

    lanes = outer_loop.unstack_integ_states(out, n)
    n_converged = sum(int(s.termination_reason) == 1 for s in lanes)
    finite = all(bool(np.isfinite(np.asarray(s.ymix)).all()) for s in lanes)
    return n, cold_s, warm_s, n_converged, finite


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone VULCAN-JAX GPU throughput benchmark "
        "(vmap-across-profiles run_batch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128],
        help="Batch sizes (number of profiles) to sweep.",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=80,
        help="Vertical layers per profile (smaller = faster).",
    )
    parser.add_argument(
        "--count-max",
        type=int,
        default=120,
        help="Max accepted integration steps per profile (step cap; "
        "keeps the sweep quick).",
    )
    parser.add_argument(
        "--tiso",
        type=float,
        default=1200.0,
        help="Base isothermal temperature (K); profiles perturb around it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # JAX must see x64 on; the package sets this at import, but make it explicit
    # for users who run the file directly without the package's import guard.
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    on_gpu = print_backend_banner()
    del on_gpu  # banner already reported it

    cfg = build_cfg(nz=args.nz, count_max=args.count_max)
    cfg.Tiso = float(args.tiso)
    integ = build_integ(cfg)

    print(
        f"\nConfig: ini_mix=const_mix  atm_type=isothermal  use_photo=False  "
        f"nz={args.nz}  count_max={args.count_max}  Tiso~{args.tiso:g}K"
    )
    print(f"Batch sizes to sweep: {args.batches}\n")

    # Build the largest profile set once and slice it for each batch size, so
    # host-side setup happens once. (Each profile differs by temperature.)
    n_max = max(args.batches)
    print(f"Building {n_max} profiles (const_mix, no FastChem) ...")
    t0 = time.perf_counter()
    all_states = build_profiles(n_max, cfg, seed=args.seed)
    print(f"  host setup: {time.perf_counter() - t0:.1f}s for {n_max} profiles\n")

    header = (
        f"{'batch':>6} | {'cold s':>9} | {'warm s':>9} | "
        f"{'warm s/prof':>12} | {'profiles/s':>11} | {'converged':>9}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for bsz in sorted(set(args.batches)):
        run_states = all_states[:bsz]
        n, cold_s, warm_s, n_conv, finite = benchmark_one(integ, run_states)
        per_prof = warm_s / n
        throughput = n / warm_s
        flag = "" if finite else "  [NON-FINITE!]"
        print(
            f"{n:>6} | {cold_s:>9.3f} | {warm_s:>9.3f} | "
            f"{per_prof * 1e3:>9.2f} ms | {throughput:>11.1f} | "
            f"{n_conv:>4}/{n:<4}{flag}"
        )
        results.append((n, cold_s, warm_s, throughput))

    print()
    if len(results) >= 2:
        base = results[0]
        # Tie-break toward the larger batch so a flat (CPU) sweep still reports
        # the speedup at the biggest batch rather than at batch 1.
        best = max(results, key=lambda r: (r[3], r[0]))
        print(
            f"Throughput at batch {base[0]}: {base[3]:.1f} profiles/s  ->  "
            f"at batch {best[0]}: {best[3]:.1f} profiles/s  "
            f"({best[3] / base[3]:.1f}x)"
        )
    print(
        "\nNote: 'cold' includes one-time XLA compilation for that batch shape; "
        "'warm' is the cached device call. On CPU throughput is ~flat; on GPU it "
        "should rise with batch size. A small count_max means most lanes hit the "
        "step cap rather than full convergence — fine for timing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

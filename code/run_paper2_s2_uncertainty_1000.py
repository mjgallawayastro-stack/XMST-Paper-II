#!/usr/bin/env python3
"""
XMST Paper II: 1000-field S2 reddening-uncertainty test
=======================================================

For each of the 1000 empirical-template Monte Carlo fields this script:

  1. reconstructs the exact frozen Stage-1 spatial field;
  2. applies paired Gaussian A_V perturbations at
       sigma_Av = 0, 0.025, 0.05, 0.10, 0.20 mag;
  3. runs the frozen XMST-S2 refinement on each perturbed field;
  4. scores the recovered groups against the injected truth memberships;
  5. computes paired changes relative to the zero-error condition;
  6. calculates paired-bootstrap 95 per cent confidence intervals for Delta J.

Within each realisation the same standard-normal A_V draw is rescaled across
all uncertainty amplitudes, so every comparison is paired.

Required beside this script:
    run_paired_order_mc.py
    run_xmst_s3.py
    quintana_2025_mst_input.csv

The --zip-dir directory must contain the original mc_0000 ... mc_0999
directories (with DONE.json and placements.csv) or the corresponding ZIP
archives used by the Paper-II Monte Carlo validation.

Example:
    python3 run_paper2_s2_uncertainty_1000.py \
      --zip-dir Runs \
      --output-dir S2_uncertainty_1000 \
      --workers 4
"""

from __future__ import annotations

import argparse
import io
import os
import time
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import run_paired_order_mc as base

_ARGS = None


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("provide at least one value")
    if any(x < 0 for x in vals):
        raise argparse.ArgumentTypeError("uncertainty sigmas must be >= 0")
    return vals


def initialise_worker(q25_csv: str, args_dict: dict):
    global _ARGS
    _ARGS = argparse.Namespace(**args_dict)
    base.initialise_worker(q25_csv, args_dict)


def quiet_call(func, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink):
        return func(*args, **kwargs)


def make_df(source_ids, xyz, av, truth):
    zeros = np.zeros(len(truth), dtype=float)
    return pd.DataFrame(
        {
            "source_id": source_ids,
            "x_pc": xyz[:, 0],
            "y_pc": xyz[:, 1],
            "z_pc": xyz[:, 2],
            "reddening_mag": av,
            "vl_lsr_kms": zeros,
            "vb_lsr_kms": zeros,
            "true_label": truth,
        }
    )


def run_one(run_source_str: str):
    started = time.time()
    run_source = Path(run_source_str)

    run, xyz, av_true, source_ids, truth, placements, done = base.reconstruct_field(run_source)

    cfg = base.core_config()
    source_col = "source_id"

    s1 = base.core.stage1_xmst(xyz, cfg.min_group_size)
    s1_gid = np.asarray(s1["group_ids"], dtype=np.int64)
    fracture = float(s1["fracture_scale_pc"])

    archived_fracture = float(done["stage1_fracture_scale_pc"])
    archived_groups = int(done["stage1_groups"])
    fracture_error = abs(fracture - archived_fracture)
    exact = fracture_error < 1e-9 and int(s1_gid.max()) == archived_groups
    if not exact:
        raise RuntimeError(
            f"MC {run:04d}: spatial reconstruction checkpoint failed: "
            f"fracture error={fracture_error:.3e}, groups={int(s1_gid.max())} "
            f"vs archived={archived_groups}"
        )

    rng = np.random.default_rng(int(_ARGS.av_error_seed_base) + run)
    z_av = rng.normal(0.0, 1.0, size=len(truth))
    available_av = np.isfinite(av_true)

    rows = []
    for av_sigma in _ARGS.av_error_sigmas:
        av_sigma = float(av_sigma)
        av = av_true.copy()
        av[available_av] = av_true[available_av] + av_sigma * z_av[available_av]

        df = make_df(source_ids, xyz, av, truth)

        s2 = quiet_call(
            base.core.run_stage2,
            df,
            xyz,
            av,
            s1_gid,
            fracture,
            source_col,
            cfg,
        )
        s2_gid, _, _ = base.final_nodes_to_gid(len(df), s2)
        _, score = base.score_truth(truth, s2_gid)

        rows.append(
            {
                "realisation": run,
                "condition": "baseline" if av_sigma == 0 else f"av_sigma_{av_sigma:g}",
                "av_error_sigma_mag": av_sigma,
                "stage1_fracture_scale_pc": fracture,
                "stage1_fracture_abs_error_pc": fracture_error,
                "exact_spatial_reconstruction": exact,
                **score,
                **base._stage_counts(s2, "s2"),
            }
        )

    return run, time.time() - started, rows


def add_paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_completeness",
        "mean_purity",
        "mean_jaccard",
        "strict_recoveries_of_8",
        "formal_overlap_resolved",
        "overlap_best_groups_separate",
        "strict_overlap_resolved",
        "s2_accepted_splits",
        "s2_orphans",
    ]
    baseline = df.loc[df["av_error_sigma_mag"] == 0, ["realisation", *metrics]].copy()
    if len(baseline) != df["realisation"].nunique():
        raise RuntimeError("Expected exactly one S2 baseline row per realisation")

    baseline = baseline.rename(columns={m: f"{m}_baseline" for m in metrics})
    out = df.merge(baseline, on="realisation", how="left", validate="many_to_one")

    for m in metrics:
        lhs = out[m].astype(int) if out[m].dtype == bool else pd.to_numeric(out[m])
        rhs = out[f"{m}_baseline"]
        rhs = rhs.astype(int) if rhs.dtype == bool else pd.to_numeric(rhs)
        out[f"delta_{m}_vs_baseline"] = lhs - rhs
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, av_sigma), g in df.groupby(
        ["condition", "av_error_sigma_mag"], sort=False
    ):
        rows.append(
            {
                "condition": condition,
                "av_error_sigma_mag": float(av_sigma),
                "n_runs": int(len(g)),
                "all_spatial_reconstructions_exact": bool(
                    g.exact_spatial_reconstruction.all()
                ),
                "max_stage1_fracture_abs_error_pc": float(
                    g.stage1_fracture_abs_error_pc.max()
                ),
                "mean_completeness": float(g.mean_completeness.mean()),
                "mean_purity": float(g.mean_purity.mean()),
                "mean_jaccard": float(g.mean_jaccard.mean()),
                "mean_strict_recoveries_of_8": float(g.strict_recoveries_of_8.mean()),
                "formal_overlap_resolution_rate": float(
                    g.formal_overlap_resolved.mean()
                ),
                "best_groups_separate_rate": float(
                    g.overlap_best_groups_separate.mean()
                ),
                "strict_overlap_resolution_rate": float(
                    g.strict_overlap_resolved.mean()
                ),
                "mean_s2_accepted_splits": float(g.s2_accepted_splits.mean()),
                "mean_s2_orphans": float(g.s2_orphans.mean()),
                "mean_delta_completeness_vs_baseline": float(
                    g.delta_mean_completeness_vs_baseline.mean()
                ),
                "mean_delta_purity_vs_baseline": float(
                    g.delta_mean_purity_vs_baseline.mean()
                ),
                "mean_delta_jaccard_vs_baseline": float(
                    g.delta_mean_jaccard_vs_baseline.mean()
                ),
                "mean_delta_strict_recoveries_vs_baseline": float(
                    g.delta_strict_recoveries_of_8_vs_baseline.mean()
                ),
                "mean_delta_s2_accepted_splits_vs_baseline": float(
                    g.delta_s2_accepted_splits_vs_baseline.mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("av_error_sigma_mag").reset_index(drop=True)


def paired_bootstrap_ci(
    df: pd.DataFrame,
    n_boot: int = 20000,
    seed: int = 12345,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for sigma in sorted(x for x in df.av_error_sigma_mag.unique() if x > 0):
        g = df.loc[df.av_error_sigma_mag == sigma].sort_values("realisation")
        delta = pd.to_numeric(
            g["delta_mean_jaccard_vs_baseline"], errors="raise"
        ).to_numpy(float)

        n = len(delta)
        boot = np.empty(n_boot, dtype=float)
        # Chunked to keep memory use modest.
        chunk = 1000
        pos = 0
        while pos < n_boot:
            k = min(chunk, n_boot - pos)
            idx = rng.integers(0, n, size=(k, n))
            boot[pos:pos+k] = delta[idx].mean(axis=1)
            pos += k

        rows.append(
            {
                "av_error_sigma_mag": float(sigma),
                "n_runs": int(n),
                "mean_delta_jaccard": float(delta.mean()),
                "bootstrap_95_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_95_ci_high": float(np.quantile(boot, 0.975)),
                "n_bootstrap": int(n_boot),
                "bootstrap_seed": int(seed),
            }
        )

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent

    p.add_argument("--zip-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--q25-csv",
        type=Path,
        default=here / "quintana_2025_mst_input.csv",
    )
    p.add_argument(
        "--av-error-sigmas",
        type=parse_float_list,
        default=parse_float_list("0,0.025,0.05,0.10,0.20"),
    )
    p.add_argument("--av-error-seed-base", type=int, default=20260830)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--bootstrap-resamples", type=int, default=20000)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) - 1)),
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1000)

    # These are required by run_paired_order_mc.initialise_worker/core_config.
    p.add_argument("--delta-v", type=float, default=5.0)
    p.add_argument("--velocity-sigma", type=float, default=1.0)
    p.add_argument("--velocity-seed-base", type=int, default=20260825)

    args = p.parse_args()
    args.av_error_sigmas = list(
        dict.fromkeys([0.0] + [float(x) for x in args.av_error_sigmas])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(
        [
            q for q in args.zip_dir.glob("mc_*")
            if q.is_dir()
            and (q / "DONE.json").exists()
            and (q / "placements.csv").exists()
        ],
        key=base.parse_run_number,
    )
    if not runs:
        runs = sorted(args.zip_dir.glob("mc_*.zip"), key=base.parse_run_number)

    runs = [r for r in runs if args.start <= base.parse_run_number(r) < args.end]
    if not runs:
        raise SystemExit(
            f"No usable MC runs in {args.zip_dir} for [{args.start},{args.end})"
        )

    args_dict = vars(args).copy()
    args_dict["zip_dir"] = str(args.zip_dir)
    args_dict["output_dir"] = str(args.output_dir)
    args_dict["q25_csv"] = str(args.q25_csv)

    print("XMST Paper-II 1000-field S2 reddening-uncertainty test")
    print("  A_V sigmas:", args.av_error_sigmas)
    print(f"  runs found: {len(runs)}")
    print(f"  workers: {args.workers}")
    print(f"  output: {args.output_dir}")
    print()

    rows = []

    def report(done_n, total, result):
        run, elapsed, rr = result
        b = next(x for x in rr if x["av_error_sigma_mag"] == 0)
        w = max(rr, key=lambda x: x["av_error_sigma_mag"])
        print(
            f"[{done_n:04d}/{total:04d}] MC {run:04d} "
            f"J0={b['mean_jaccard']:.4f} "
            f"J0.20={w['mean_jaccard']:.4f} "
            f"splits {b['s2_accepted_splits']}->{w['s2_accepted_splits']} "
            f"{elapsed:.1f}s",
            flush=True,
        )

    if args.workers == 1:
        initialise_worker(str(args.q25_csv), args_dict)
        for i, r in enumerate(runs, 1):
            result = run_one(str(r))
            rows.extend(result[2])
            report(i, len(runs), result)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialise_worker,
            initargs=(str(args.q25_csv), args_dict),
        ) as ex:
            futures = {ex.submit(run_one, str(r)): r for r in runs}
            completed = 0
            for fut in as_completed(futures):
                r = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    print(f"FAILED {r.name}: {type(exc).__name__}: {exc}", flush=True)
                    raise
                rows.extend(result[2])
                completed += 1
                report(completed, len(runs), result)

    all_runs = pd.DataFrame(rows).sort_values(
        ["realisation", "av_error_sigma_mag"]
    ).reset_index(drop=True)
    all_runs = add_paired_deltas(all_runs)

    agg = aggregate(all_runs)
    ci = paired_bootstrap_ci(
        all_runs,
        n_boot=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    all_path = args.output_dir / "s2_uncertainty_all_runs.csv"
    agg_path = args.output_dir / "s2_uncertainty_aggregate.csv"
    ci_path = args.output_dir / "s2_uncertainty_paired_bootstrap_ci.csv"

    all_runs.to_csv(all_path, index=False)
    agg.to_csv(agg_path, index=False)
    ci.to_csv(ci_path, index=False)

    print("\nFINAL S2 AGGREGATE")
    print(
        agg[
            [
                "av_error_sigma_mag",
                "mean_completeness",
                "mean_purity",
                "mean_jaccard",
                "mean_strict_recoveries_of_8",
                "mean_s2_accepted_splits",
                "mean_delta_jaccard_vs_baseline",
            ]
        ].to_string(index=False)
    )

    print("\nPAIRED BOOTSTRAP 95% CI FOR DELTA J")
    print(ci.to_string(index=False))

    print("\nWrote:")
    print(" ", all_path)
    print(" ", agg_path)
    print(" ", ci_path)


if __name__ == "__main__":
    main()

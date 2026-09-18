#!/usr/bin/env python3
"""
XMST Paper II: exact-provenance S3 gate-sensitivity sweep
=========================================================

This reruns ONLY the targeted S3 gate-sensitivity experiment at DeltaV=3 km/s,
using exactly the shuffled field ordering and intrinsic velocity realisations
of the original Table 3 targeted S3 experiment.

Key provenance constraints reproduced here:
  * original MC field RNG seed = 20260725 + run, including the final row permutation
  * original velocity RNG seed = 20260725 + run + 30,000,000
  * intrinsic sigma_v = 1 km/s per component
  * DeltaV = 3 km/s
  * one random offset direction generated after the Gaussian velocity field
  * S3 is applied only to the Stage-1 parent containing the planted T01/T02
    merger
  * if Stage 1 already separates T01 and T02, Stage 1 is left unchanged
  * only the coupled S3 evidence gates are varied:
        factor 0.8 -> DeltaBIC=8,  D_2D=1.6
        factor 0.9 -> DeltaBIC=9,  D_2D=1.8
        factor 1.0 -> DeltaBIC=10, D_2D=2.0
        factor 1.1 -> DeltaBIC=11, D_2D=2.2
        factor 1.2 -> DeltaBIC=12, D_2D=2.4

The factor=1.0 aggregate MUST reproduce the Table 3 DeltaV=3 result:
  resolution = 0.310
  C = 0.9933
  P = 0.7767
  J = 0.7707
  strict / 8 = 6.575

Required beside this script:
    run_paired_order_mc.py
    run_xmst_s3.py
    quintana_2025_mst_input.csv

The --zip-dir directory must contain the original mc_0000 ... mc_0999
directories (DONE.json + placements.csv) or corresponding mc_XXXX.zip files.

Example:
    python3 run_paper2_s3_gate_sensitivity_exact_table3.py \
      --zip-dir Runs \
      --output-dir S3_gate_sensitivity_exact \
      --workers 8
"""

from __future__ import annotations

import argparse
import io
import os
import time
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import run_paired_order_mc as base

_ARGS = None

T1 = "empirical_template_01"
T2 = "empirical_template_02"


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("provide at least one factor")
    return vals


def initialise_worker(q25_csv: str, args_dict: dict):
    global _ARGS
    _ARGS = argparse.Namespace(**args_dict)
    base.initialise_worker(q25_csv, args_dict)


def quiet_call(func, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink):
        return func(*args, **kwargs)


def table3_velocity_field(run: int, truth: np.ndarray):
    """
    Exact original Table-3 velocity generator:
        seed = 20260725 + run + 30_000_000
        draw Gaussian 2-D field first
        then draw the random direction angle
    """
    seed = int(_ARGS.mc_seed_base) + int(run) + int(_ARGS.velocity_seed_offset)
    rng = np.random.default_rng(seed)

    velocity = rng.normal(
        0.0,
        float(_ARGS.velocity_sigma),
        size=(len(truth), 2),
    )

    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    t1 = truth == T1
    t2 = truth == T2
    velocity[t1] -= 0.5 * float(_ARGS.delta_v) * direction
    velocity[t2] += 0.5 * float(_ARGS.delta_v) * direction

    return velocity[:, 0], velocity[:, 1], theta, seed


def best_stage1_group(truth: np.ndarray, s1_gid: np.ndarray, label: str) -> int:
    mask = truth == label
    gids, counts = np.unique(s1_gid[mask & (s1_gid > 0)], return_counts=True)
    if len(gids) == 0:
        return 0
    order = np.lexsort((gids, -counts))
    return int(gids[order[0]])


def s3_cfg(factor: float):
    c = base.core_config()
    return replace(
        c,
        s3_delta_bic_threshold=10.0 * factor,
        s3_separation_d_threshold=2.0 * factor,
    )


def one_parent_nodes(s1_gid: np.ndarray, target_gid: int):
    idx = np.flatnonzero(s1_gid == target_gid).astype(np.int64)
    nid = f"G{target_gid:03d}"
    nodes = {
        nid: {
            "node_id": nid,
            "parent_id": "",
            "root_stage1_group_id": int(target_gid),
            "generation": 0,
            "created_pass": 0,
            "indices": idx,
            "status": "active",
        }
    }
    return nodes, [nid]


def merge_target_result_into_stage1(
    n: int,
    s1_gid: np.ndarray,
    target_gid: int,
    s3_result: dict,
) -> np.ndarray:
    """
    Keep every non-target Stage-1 group exactly as it was. Replace only the
    targeted parent with the retained final S3 descendants. S3 orphans remain 0.
    """
    final_gid = np.asarray(s1_gid, dtype=np.int64).copy()
    target_mask = final_gid == target_gid
    final_gid[target_mask] = 0

    next_gid = int(final_gid.max()) + 1
    for nid in s3_result["final_node_ids"]:
        idx = np.asarray(s3_result["nodes"][nid]["indices"], dtype=np.int64)
        final_gid[idx] = next_gid
        next_gid += 1

    return final_gid



def reconstruct_exact_table3_field(run_source: Path):
    """Recreate the original Table-3 MC field INCLUDING its final row shuffle.

    The earlier reconstruction from archived placements recovered the same spatial
    point set and Stage-1 partition, but not the original row ordering. That matters
    here because the Table-3 velocity RNG assigned Gaussian draws by row after the
    MC generator's final permutation. Replaying the original field generator fixes
    the per-star velocity provenance exactly.
    """
    archived_placements = base.read_run_csv(run_source, "placements.csv")
    done = base.read_run_json(run_source, "DONE.json")
    run = int(done["realisation"])

    coords = base._COORDS
    av_all = base._AV
    sid_all = base._SOURCE_IDS
    bg = base._BACKGROUND_INDICES

    field_xyz = coords[bg]
    field_av = av_all[bg]
    field_sid = sid_all[bg]
    field_dist = np.linalg.norm(field_xyz, axis=1)
    anchor_pool = np.flatnonzero((field_dist >= 250.0) & (field_dist <= 900.0))
    if len(anchor_pool) == 0:
        raise RuntimeError("No anchor candidates in exact Table-3 reconstruction")

    labels = list(base.TEMPLATE_S1_GROUPS.keys())
    templates = []
    for label in labels:
        idx = base._TEMPLATE_INDICES[label]
        original = coords[idx]
        origin = np.median(original, axis=0)
        templates.append({
            "label": label,
            "idx": idx,
            "rel_xyz": original - origin,
            "av": av_all[idx],
            "sid": sid_all[idx],
        })

    rng = np.random.default_rng(int(_ARGS.mc_seed_base) + run)
    rotations = [Rotation.random(random_state=rng).as_matrix() for _ in templates]
    centres = []
    placed = []
    generated_rows = []

    overlap_direction = field_xyz[int(rng.choice(anchor_pool))]
    overlap_direction = overlap_direction / np.linalg.norm(overlap_direction)
    d0 = float(rng.uniform(600.0, 820.0))
    delta = float(rng.uniform(57.0, 83.0))
    overlap_dists = [d0, d0 + delta]

    for i, template in enumerate(templates):
        rel = template["rel_xyz"]
        rot = rotations[i]
        if i < 2:
            centre = overlap_direction * overlap_dists[i]
            xyz = rel @ rot.T + centre
            tries = 0
            while (
                np.linalg.norm(xyz, axis=1).max() > 1000.0
                or np.linalg.norm(xyz, axis=1).min() < 30.0
            ) and tries < 100:
                rot = Rotation.random(random_state=rng).as_matrix()
                xyz = rel @ rot.T + centre
                tries += 1
        else:
            xyz = None
            for _tries in range(1000):
                centre = field_xyz[int(rng.choice(anchor_pool))].copy()
                if all(np.linalg.norm(centre - c) >= 120.0 for c in centres):
                    rot = Rotation.random(random_state=rng).as_matrix()
                    trial = rel @ rot.T + centre
                    radii = np.linalg.norm(trial, axis=1)
                    if radii.min() >= 30.0 and radii.max() <= 1000.0:
                        xyz = trial
                        break
            if xyz is None:
                raise RuntimeError(f"MC {run:04d}: could not reproduce template placement {i+1}")

        centres.append(centre)
        placed.append(xyz)
        generated_rows.append((template["label"], centre.copy(), rot.copy()))

    # Strong placement provenance check against the archived run.
    ap = archived_placements.sort_values("template_label").reset_index(drop=True)
    gp = sorted(generated_rows, key=lambda x: x[0])
    if len(ap) != len(gp):
        raise RuntimeError(f"MC {run:04d}: archived placement count mismatch")
    for j, (label, centre, rot) in enumerate(gp):
        row = ap.iloc[j]
        if str(row["template_label"]) != label:
            raise RuntimeError(f"MC {run:04d}: template-label order mismatch")
        archived_centre = np.array(
            [row["centre_x_pc"], row["centre_y_pc"], row["centre_z_pc"]], dtype=float
        )
        archived_rot = base.parse_rotation(row["rotation_matrix"])
        if not np.allclose(centre, archived_centre, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"MC {run:04d}: centre provenance mismatch for {label}")
        if not np.allclose(rot, archived_rot, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"MC {run:04d}: rotation provenance mismatch for {label}")

    cluster_xyz = np.vstack(placed)
    cluster_av = np.concatenate([t["av"] for t in templates])
    cluster_sid = np.concatenate([t["sid"] for t in templates])
    cluster_truth = np.concatenate(
        [np.repeat(t["label"], len(t["av"])) for t in templates]
    )

    xyz = np.vstack([field_xyz, cluster_xyz])
    av = np.concatenate([field_av, cluster_av])
    source_ids = np.concatenate([field_sid, cluster_sid])
    truth = np.concatenate([np.repeat("background", len(field_xyz)), cluster_truth])

    # CRITICAL: this is the step omitted by the first exact-provenance attempt.
    order = rng.permutation(len(xyz))
    xyz = xyz[order]
    av = av[order]
    source_ids = source_ids[order]
    truth = truth[order]

    archived_delta = done.get("overlap_distance_separation_pc")
    if archived_delta is not None and not np.isclose(
        delta, float(archived_delta), rtol=0.0, atol=1e-10
    ):
        raise RuntimeError(
            f"MC {run:04d}: overlap-distance provenance mismatch "
            f"{delta:.16g} vs {float(archived_delta):.16g}"
        )

    return run, xyz, av, source_ids, truth, archived_placements, done



def stage1_from_archived_fracture(xyz: np.ndarray, fracture: float, min_group_size: int = 10):
    """Recover the exact Stage-1 cut partition without rebuilding the MST.

    For a Euclidean MST cut at threshold L, the connected components are
    identical to those of the full geometric graph containing every pair with
    separation <= L. A cKDTree fixed-radius graph therefore recovers the same
    Stage-1 components much faster than recomputing Delaunay + MST.
    """
    n = len(xyz)
    pairs = cKDTree(xyz).query_pairs(r=float(fracture), output_type="ndarray")

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for u0, v0 in pairs:
        u, v = int(u0), int(v0)
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    vals, counts = np.unique(roots, return_counts=True)
    keep = vals[counts >= int(min_group_size)]

    components = [np.flatnonzero(roots == r).astype(np.int64) for r in keep]
    components.sort(key=lambda a: int(a.min()))

    gid = np.zeros(n, dtype=np.int64)
    for j, idx in enumerate(components, start=1):
        gid[idx] = j
    return gid, int(len(pairs))


def run_one(run_source_str: str):
    started = time.time()
    run_source = Path(run_source_str)

    run, xyz, av, source_ids, truth, placements, done = reconstruct_exact_table3_field(run_source)

    cfg0 = base.core_config()
    fracture = float(done["stage1_fracture_scale_pc"])
    archived_groups = int(done["stage1_groups"])
    s1_gid, stage1_radius_pairs = stage1_from_archived_fracture(
        xyz, fracture, cfg0.min_group_size
    )
    ferr = 0.0
    exact = int(s1_gid.max()) == archived_groups
    if not exact:
        raise RuntimeError(
            f"MC {run:04d}: archived-fracture Stage-1 reconstruction mismatch: "
            f"groups={int(s1_gid.max())} vs archived={archived_groups}"
        )

    vl, vb, theta, velocity_seed = table3_velocity_field(run, truth)

    # If the archived Table-3 direction was stored, require exact agreement
    # to numerical precision. This catches any provenance drift immediately.
    archived_theta = done.get("velocity_direction_angle_rad")
    angle_matches_archive = True
    if archived_theta is not None and np.isfinite(float(archived_theta)):
        angle_matches_archive = bool(
            np.isclose(theta, float(archived_theta), rtol=0.0, atol=1e-12)
        )
        if not angle_matches_archive:
            raise RuntimeError(
                f"MC {run:04d}: Table-3 velocity angle provenance mismatch: "
                f"generated={theta:.16g}, archived={float(archived_theta):.16g}"
            )

    df = pd.DataFrame(
        {
            "source_id": source_ids,
            "x_pc": xyz[:, 0],
            "y_pc": xyz[:, 1],
            "z_pc": xyz[:, 2],
            "reddening_mag": av,
            "vl_lsr_kms": vl,
            "vb_lsr_kms": vb,
            "true_label": truth,
        }
    )

    t1_gid = best_stage1_group(truth, s1_gid, T1)
    t2_gid = best_stage1_group(truth, s1_gid, T2)
    already_separate = (
        t1_gid > 0 and t2_gid > 0 and t1_gid != t2_gid
    )

    rows = []

    for factor in _ARGS.factors:
        factor = float(factor)
        cfg = s3_cfg(factor)

        if already_separate:
            # Exact Table-3 rule: if Stage 1 already separates T01/T02,
            # do not run S3 on unrelated parents.
            final_gid = s1_gid.copy()
            accepted_splits = 0
            orphans = 0
            target_parent_gid = 0
        else:
            if t1_gid <= 0 or t2_gid <= 0 or t1_gid != t2_gid:
                raise RuntimeError(
                    f"MC {run:04d}: could not identify common T01/T02 Stage-1 parent "
                    f"(T01={t1_gid}, T02={t2_gid})"
                )

            target_parent_gid = int(t1_gid)
            nodes, ids = one_parent_nodes(s1_gid, target_parent_gid)

            s3 = quiet_call(
                base.core.run_stage3,
                df,
                xyz,
                vl,
                vb,
                nodes,
                ids,
                fracture,
                "source_id",
                cfg,
            )
            final_gid = merge_target_result_into_stage1(
                len(df), s1_gid, target_parent_gid, s3
            )
            counts = base._stage_counts(s3, "s3")
            accepted_splits = int(counts["s3_accepted_splits"])
            orphans = int(counts["s3_orphans"])

        _, score = base.score_truth(truth, final_gid)

        rows.append(
            {
                "realisation": int(run),
                "gate_factor": factor,
                "delta_bic_threshold": float(cfg.s3_delta_bic_threshold),
                "separation_threshold": float(cfg.s3_separation_d_threshold),
                "delta_v_kms": float(_ARGS.delta_v),
                "intrinsic_velocity_sigma_kms": float(_ARGS.velocity_sigma),
                "velocity_seed": int(velocity_seed),
                "velocity_direction_angle_rad": float(theta),
                "angle_matches_archived_table3": bool(angle_matches_archive),
                "exact_spatial_reconstruction": bool(exact),
                "stage1_fracture_abs_error_pc": float(ferr),
                "stage1_radius_graph_pairs": int(stage1_radius_pairs),
                "t01_stage1_best_group": int(t1_gid),
                "t02_stage1_best_group": int(t2_gid),
                "stage1_already_separate": bool(already_separate),
                "target_parent_gid": int(target_parent_gid),
                "s3_accepted_splits": int(accepted_splits),
                "s3_orphans": int(orphans),
                **score,
            }
        )

    return run, time.time() - started, rows


def aggregate(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, g in all_runs.groupby("gate_factor", sort=True):
        rows.append(
            {
                "gate_factor": float(factor),
                "delta_bic_threshold": float(g.delta_bic_threshold.iloc[0]),
                "separation_threshold": float(g.separation_threshold.iloc[0]),
                "n_runs": int(len(g)),
                "all_spatial_reconstructions_exact": bool(
                    g.exact_spatial_reconstruction.all()
                ),
                "all_velocity_angles_match_table3_archive": bool(
                    g.angle_matches_archived_table3.all()
                ),
                "stage1_already_separate_count": int(
                    g.stage1_already_separate.sum()
                ),
                "mean_completeness": float(g.mean_completeness.mean()),
                "mean_purity": float(g.mean_purity.mean()),
                "mean_jaccard": float(g.mean_jaccard.mean()),
                "mean_strict_recoveries_of_8": float(
                    g.strict_recoveries_of_8.mean()
                ),
                "formal_overlap_resolution_rate": float(
                    g.formal_overlap_resolved.mean()
                ),
                "best_groups_separate_rate": float(
                    g.overlap_best_groups_separate.mean()
                ),
                "strict_overlap_resolution_rate": float(
                    g.strict_overlap_resolved.mean()
                ),
                "mean_s3_accepted_splits": float(g.s3_accepted_splits.mean()),
                "mean_s3_orphans": float(g.s3_orphans.mean()),
            }
        )
    return pd.DataFrame(rows)


def validate_nominal(agg: pd.DataFrame):
    row = agg.loc[np.isclose(agg.gate_factor, 1.0)]
    if len(row) != 1:
        raise RuntimeError("Expected exactly one nominal factor=1.0 aggregate row")
    r = row.iloc[0]

    # Exact integer/fraction checkpoint where possible.
    if int(r["n_runs"]) != 1000:
        raise RuntimeError(f"Nominal checkpoint needs 1000 runs, got {int(r['n_runs'])}")

    checks = {
        "formal_overlap_resolution_rate": (float(r.formal_overlap_resolution_rate), 0.310, 5e-4),
        "mean_completeness": (float(r.mean_completeness), 0.9933, 5e-5),
        "mean_purity": (float(r.mean_purity), 0.7767, 5e-5),
        "mean_jaccard": (float(r.mean_jaccard), 0.7707, 5e-5),
        "mean_strict_recoveries_of_8": (float(r.mean_strict_recoveries_of_8), 6.575, 5e-4),
    }

    failures = []
    for name, (value, expected, tol) in checks.items():
        if abs(value - expected) > tol:
            failures.append(
                f"{name}: got {value:.8f}, expected {expected:.8f}"
            )

    if failures:
        raise RuntimeError(
            "NOMINAL FACTOR=1.0 DOES NOT REPRODUCE TABLE 3:\n  "
            + "\n  ".join(failures)
        )


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
        "--factors",
        type=parse_float_list,
        default=parse_float_list("0.8,0.9,1.0,1.1,1.2"),
    )
    p.add_argument("--delta-v", type=float, default=3.0)
    p.add_argument("--velocity-sigma", type=float, default=1.0)

    # Exact original Table-3 provenance.
    p.add_argument("--mc-seed-base", type=int, default=20260725)
    p.add_argument("--velocity-seed-offset", type=int, default=30000000)

    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) - 1)),
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1000)

    # Required by base.initialise_worker/core_config.
    p.add_argument("--velocity-seed-base", type=int, default=20260825)

    args = p.parse_args()
    args.factors = sorted(set([1.0] + [float(x) for x in args.factors]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(
        [
            q
            for q in args.zip_dir.glob("mc_*")
            if q.is_dir()
            and (q / "DONE.json").exists()
            and (q / "placements.csv").exists()
        ],
        key=base.parse_run_number,
    )
    if not runs:
        runs = sorted(args.zip_dir.glob("mc_*.zip"), key=base.parse_run_number)

    runs = [
        r for r in runs
        if args.start <= base.parse_run_number(r) < args.end
    ]
    if not runs:
        raise SystemExit(
            f"No usable MC runs in {args.zip_dir} for [{args.start},{args.end})"
        )

    args_dict = vars(args).copy()
    args_dict["zip_dir"] = str(args.zip_dir)
    args_dict["output_dir"] = str(args.output_dir)
    args_dict["q25_csv"] = str(args.q25_csv)

    print("XMST Paper-II exact Table-3-provenance S3 gate sensitivity v3-fast")
    print("  exact original MC row permutation + archived-fracture Stage-1 + targeted parent only")
    print(f"  DeltaV = {args.delta_v:g} km/s")
    print(f"  intrinsic sigma = {args.velocity_sigma:g} km/s/component")
    print(f"  factors = {args.factors}")
    print(f"  Table-3 velocity seed = {args.mc_seed_base} + run + {args.velocity_seed_offset}")
    print(f"  runs = {len(runs)}; workers = {args.workers}")
    print()

    rows = []

    def report(done_n, total, result):
        run, elapsed, rr = result
        nominal = next(x for x in rr if np.isclose(x["gate_factor"], 1.0))
        print(
            f"[{done_n:04d}/{total:04d}] MC {run:04d} "
            f"nominal_res={int(nominal['formal_overlap_resolved'])} "
            f"J={nominal['mean_jaccard']:.4f} "
            f"{elapsed:.2f}s",
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
                    print(
                        f"FAILED {r.name}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise
                rows.extend(result[2])
                completed += 1
                report(completed, len(runs), result)

    all_runs = (
        pd.DataFrame(rows)
        .sort_values(["realisation", "gate_factor"])
        .reset_index(drop=True)
    )
    agg = aggregate(all_runs)

    all_path = args.output_dir / "s3_gate_sensitivity_exact_all_runs.csv"
    agg_path = args.output_dir / "s3_gate_sensitivity_exact_aggregate.csv"

    all_runs.to_csv(all_path, index=False)
    agg.to_csv(agg_path, index=False)

    # Only enforce the Table-3 numerical checkpoint for the complete run.
    if args.start == 0 and args.end >= 1000 and len(runs) == 1000:
        validate_nominal(agg)
        print("\nTABLE-3 NOMINAL CHECK: PASS")
    else:
        print("\nPartial run: nominal Table-3 aggregate checkpoint not enforced.")

    print("\nFINAL AGGREGATE")
    print(
        agg[
            [
                "gate_factor",
                "delta_bic_threshold",
                "separation_threshold",
                "formal_overlap_resolution_rate",
                "mean_completeness",
                "mean_purity",
                "mean_jaccard",
                "mean_strict_recoveries_of_8",
            ]
        ].to_string(index=False)
    )

    print("\nWrote:")
    print(" ", all_path)
    print(" ", agg_path)


if __name__ == "__main__":
    main()

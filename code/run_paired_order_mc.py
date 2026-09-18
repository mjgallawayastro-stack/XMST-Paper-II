#!/usr/bin/env python3
"""
XMST Paper II paired order-dependence control
======================================

Reconstruct the exact empirical-template Monte Carlo field stored by each
Paper-II result ZIP, inject the controlled transverse-velocity experiment,
then run BOTH pipelines on the same reconstructed field and the same velocity draw:

    forward: XMST-S1 -> reddening -> kinematics
    reverse: XMST-S1 -> kinematics -> reddening

The spatial MC is reconstructed from the stored placements.csv:
- Q25 parent catalogue: 24,706 stars
- empirical templates: S1 groups 81,10,12,90,52,74,25,63
- sizes: 192,183,109,57,51,43,37,25
- each template is centred on its coordinate-wise median, rotated by the
  stored rotation matrix, and translated to the stored placement centre.
- the 697 template stars are removed from the original field before injection.

For the controlled kinematic layer:
- all stars receive a common unimodal Gaussian transverse-velocity field;
- T01 and T02 receive opposite centroid shifts of +/- DeltaV/2;
- default DeltaV = 5 km/s and sigma = 1 km/s per component;
- the stored velocity direction angle from each result ZIP is reused.

Nothing in the S2 or S3 gates is retuned.

Required files beside this script:
    run_xmst_s3.py
    quintana_2025_mst_input.csv

Example:
    python3 run_paired_order_mc.py \
      --zip-dir "$HOME/Downloads/paper II - reverse/Runs" \
      --output-dir "$HOME/Downloads/paper II - reverse/Reverse_results" \
      --workers 4
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import run_xmst_s3 as core


TEMPLATE_S1_GROUPS = {
    "empirical_template_01": 81,   # 192 stars
    "empirical_template_02": 10,   # 183
    "empirical_template_03": 12,   # 109
    "empirical_template_04": 90,   # 57
    "empirical_template_05": 52,   # 51
    "empirical_template_06": 74,   # 43
    "empirical_template_07": 25,   # 37
    "empirical_template_08": 63,   # 25
}

EXPECTED_TEMPLATE_SIZES = {
    "empirical_template_01": 192,
    "empirical_template_02": 183,
    "empirical_template_03": 109,
    "empirical_template_04": 57,
    "empirical_template_05": 51,
    "empirical_template_06": 43,
    "empirical_template_07": 37,
    "empirical_template_08": 25,
}

EXPECTED_TEMPLATE_MEDIAN_AV = {
    "empirical_template_01": 1.334688965,
    "empirical_template_02": 0.11990647,
    "empirical_template_03": 0.12375247,
    "empirical_template_04": 1.38144134,
    "empirical_template_05": 1.18007979,
    "empirical_template_06": 0.63615707,
    "empirical_template_07": 0.32699393,
    "empirical_template_08": 0.61952747,
}

_BASE_DF = None
_COORDS = None
_AV = None
_SOURCE_IDS = None
_NOMINAL_GID = None
_TEMPLATE_INDICES = None
_BACKGROUND_INDICES = None
_ARGS = None


def read_run_json(run_source: Path, basename: str) -> dict:
    if run_source.is_dir():
        path = run_source / basename
        if not path.exists():
            raise RuntimeError(f"{run_source}: missing {basename}")
        return json.loads(path.read_text(encoding="utf-8"))
    if run_source.is_file() and run_source.suffix.lower() == ".zip":
        with zipfile.ZipFile(run_source) as zf:
            matches = [n for n in zf.namelist() if Path(n).name == basename]
            if len(matches) != 1:
                raise RuntimeError(f"{run_source}: expected exactly one {basename}, found {len(matches)}")
            with zf.open(matches[0]) as f:
                return json.load(f)
    raise RuntimeError(f"Unsupported run source: {run_source}")


def read_run_csv(run_source: Path, basename: str) -> pd.DataFrame:
    if run_source.is_dir():
        path = run_source / basename
        if not path.exists():
            raise RuntimeError(f"{run_source}: missing {basename}")
        return pd.read_csv(path)
    if run_source.is_file() and run_source.suffix.lower() == ".zip":
        with zipfile.ZipFile(run_source) as zf:
            matches = [n for n in zf.namelist() if Path(n).name == basename]
            if len(matches) != 1:
                raise RuntimeError(f"{run_source}: expected exactly one {basename}, found {len(matches)}")
            with zf.open(matches[0]) as f:
                return pd.read_csv(f)
    raise RuntimeError(f"Unsupported run source: {run_source}")


def parse_rotation(value) -> np.ndarray:
    if isinstance(value, str):
        try:
            arr = np.asarray(json.loads(value), dtype=float)
        except Exception:
            arr = np.asarray(ast.literal_eval(value), dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"Rotation matrix has shape {arr.shape}, expected (3,3)")
    return arr


def core_config(input_csv: str = "reconstructed_mc") -> core.Config:
    return core.Config(
        input_csv=input_csv,
        kinematics_csv=None,
        output_dir=".",
        min_group_size=10,

        # Frozen XMST-S2 gate.
        s2_delta_bic_threshold=10.0,
        s2_ashman_d_threshold=2.0,
        s2_min_gmm_component_n=5,
        s2_min_gmm_component_fraction=0.15,
        s2_min_reddening_coverage=1.0,
        s2_gmm_n_init=20,
        s2_gmm_random_state=20260825,
        s2_spatial_child_policy="all",
        max_s2_passes=20,

        # Frozen XMST-S3 gate.
        s3_delta_bic_threshold=10.0,
        s3_separation_d_threshold=2.0,
        s3_min_gmm_component_n=5,
        s3_min_gmm_component_fraction=0.15,
        s3_min_kinematic_coverage=1.0,
        s3_gmm_n_init=20,
        s3_gmm_random_state=20260825,
        s3_spatial_child_policy="all",
        max_s3_passes=20,

        vl_col="vl_lsr_kms",
        vb_col="vb_lsr_kms",
        velocity_frame="lsr",

        # These are MC fields, not the nominal real catalogue.
        strict_q25_checkpoint=False,
        strict_s2_checkpoint=False,
        skip_s3=False,
    )


def initialise_worker(q25_csv: str, args_dict: dict):
    global _BASE_DF, _COORDS, _AV, _SOURCE_IDS, _NOMINAL_GID
    global _TEMPLATE_INDICES, _BACKGROUND_INDICES, _ARGS

    _ARGS = argparse.Namespace(**args_dict)
    _BASE_DF = core.read_csv_preserve_ids(q25_csv)

    xcol = core.find_col(_BASE_DF, ["x_pc", "x"])
    ycol = core.find_col(_BASE_DF, ["y_pc", "y"])
    zcol = core.find_col(_BASE_DF, ["z_pc", "z"])
    avcol = core.find_col(_BASE_DF, ["reddening_mag", "a_v", "av", "reddening"])
    sidcol = core.find_col(_BASE_DF, ["source_id", "gaia_source_id", "id"])

    _COORDS = core.numeric_array(_BASE_DF, [xcol, ycol, zcol])
    _AV = pd.to_numeric(_BASE_DF[avcol], errors="coerce").to_numpy(float)
    _SOURCE_IDS = _BASE_DF[sidcol].map(core.canonical_source_id).to_numpy(object)

    if len(_BASE_DF) != 24706:
        raise RuntimeError(f"Q25 input must contain 24,706 stars; found {len(_BASE_DF):,}")

    nominal = core.stage1_xmst(_COORDS, 10)
    _NOMINAL_GID = np.asarray(nominal["group_ids"], dtype=np.int64)

    # Verify frozen real-catalogue checkpoint.
    if abs(float(nominal["fracture_scale_pc"]) - 17.048388529750305) > 1e-8:
        raise RuntimeError(
            "Q25 Stage-1 fracture checkpoint failed: "
            f"{nominal['fracture_scale_pc']}"
        )
    if int(_NOMINAL_GID.max()) != 103:
        raise RuntimeError(
            f"Q25 Stage-1 group checkpoint failed: {_NOMINAL_GID.max()} != 103"
        )
    if int(np.sum(_NOMINAL_GID > 0)) != 2339:
        raise RuntimeError(
            "Q25 Stage-1 grouped-star checkpoint failed: "
            f"{np.sum(_NOMINAL_GID > 0)} != 2339"
        )

    _TEMPLATE_INDICES = {}
    for label, gid in TEMPLATE_S1_GROUPS.items():
        idx = np.flatnonzero(_NOMINAL_GID == gid)
        n = len(idx)
        med = float(np.nanmedian(_AV[idx]))
        if n != EXPECTED_TEMPLATE_SIZES[label]:
            raise RuntimeError(f"{label}: expected {EXPECTED_TEMPLATE_SIZES[label]} stars, got {n}")
        if abs(med - EXPECTED_TEMPLATE_MEDIAN_AV[label]) > 1e-8:
            raise RuntimeError(
                f"{label}: median A_V mismatch {med} vs {EXPECTED_TEMPLATE_MEDIAN_AV[label]}"
            )
        _TEMPLATE_INDICES[label] = idx

    injected = np.concatenate(list(_TEMPLATE_INDICES.values()))
    if len(np.unique(injected)) != 697:
        raise RuntimeError("Template membership is not 697 unique stars")

    mask = np.ones(len(_BASE_DF), dtype=bool)
    mask[injected] = False
    _BACKGROUND_INDICES = np.flatnonzero(mask)

    if len(_BACKGROUND_INDICES) != 24009:
        raise RuntimeError(
            f"Expected 24,009 background stars, got {len(_BACKGROUND_INDICES):,}"
        )


def reconstruct_field(run_source: Path):
    placements = read_run_csv(run_source, "placements.csv")
    done = read_run_json(run_source, "DONE.json")

    run = int(done["realisation"])

    coords_parts = [_COORDS[_BACKGROUND_INDICES]]
    av_parts = [_AV[_BACKGROUND_INDICES]]
    sid_parts = [_SOURCE_IDS[_BACKGROUND_INDICES]]
    truth_parts = [np.full(len(_BACKGROUND_INDICES), "background", dtype=object)]

    placements = placements.sort_values("template_label").reset_index(drop=True)

    seen = set()
    for _, row in placements.iterrows():
        label = str(row["template_label"])
        if label not in _TEMPLATE_INDICES:
            raise RuntimeError(f"{run_source.name}: unknown template {label}")
        seen.add(label)

        idx = _TEMPLATE_INDICES[label]
        original = _COORDS[idx]

        # CRITICAL: the original generator centred each empirical template on
        # its coordinate-wise median before applying the stored rotation.
        origin = np.median(original, axis=0)
        centred = original - origin

        R = parse_rotation(row["rotation_matrix"])
        target = np.array(
            [row["centre_x_pc"], row["centre_y_pc"], row["centre_z_pc"]],
            dtype=float,
        )

        # scipy Rotation.apply convention: x' = R x, represented for row
        # vectors as centred @ R.T.
        transformed = centred @ R.T + target

        coords_parts.append(transformed)
        av_parts.append(_AV[idx])
        sid_parts.append(_SOURCE_IDS[idx])
        truth_parts.append(np.full(len(idx), label, dtype=object))

    if seen != set(TEMPLATE_S1_GROUPS):
        missing = sorted(set(TEMPLATE_S1_GROUPS) - seen)
        raise RuntimeError(f"{run_source.name}: missing placements for {missing}")

    xyz = np.vstack(coords_parts)
    av = np.concatenate(av_parts)
    source_ids = np.concatenate(sid_parts)
    truth = np.concatenate(truth_parts)

    if len(xyz) != 24706:
        raise RuntimeError(f"{run_source.name}: reconstructed N={len(xyz)}, expected 24706")

    return run, xyz, av, source_ids, truth, placements, done


def inject_kinematics(run: int, truth: np.ndarray, done: dict):
    sigma = float(_ARGS.velocity_sigma)
    delta = float(_ARGS.delta_v)

    rng = np.random.default_rng(int(_ARGS.velocity_seed_base) + run)
    velocity = rng.normal(0.0, sigma, size=(len(truth), 2))

    angle = done.get("velocity_direction_angle_rad")
    if angle is None or not np.isfinite(float(angle)):
        angle = rng.uniform(0.0, 2.0 * np.pi)
    angle = float(angle)

    direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)

    t1 = truth == "empirical_template_01"
    t2 = truth == "empirical_template_02"

    # Opposite shifts around the common null centroid.
    velocity[t1] -= 0.5 * delta * direction
    velocity[t2] += 0.5 * delta * direction

    return velocity[:, 0], velocity[:, 1], angle


def s1_seed_nodes(s1_gid: np.ndarray):
    nodes = {}
    ids = []
    for gid in sorted(int(x) for x in np.unique(s1_gid) if int(x) > 0):
        nid = f"G{gid:03d}"
        idx = np.flatnonzero(s1_gid == gid)
        nodes[nid] = {
            "node_id": nid,
            "parent_id": "",
            "root_stage1_group_id": gid,
            "generation": 0,
            "created_pass": 0,
            "indices": idx.astype(np.int64),
            "status": "active",
        }
        ids.append(nid)
    return nodes, ids


def final_nodes_to_gid(n: int, result: dict):
    gid = np.zeros(n, dtype=np.int64)
    node = np.full(n, "", dtype=object)
    gid_to_node = {}
    for j, nid in enumerate(result["final_node_ids"], start=1):
        idx = np.asarray(result["nodes"][nid]["indices"], dtype=np.int64)
        gid[idx] = j
        node[idx] = nid
        gid_to_node[j] = nid
    return gid, node, gid_to_node


def score_truth(truth: np.ndarray, final_gid: np.ndarray):
    rows = []
    best = {}

    for label in TEMPLATE_S1_GROUPS:
        tmask = truth == label
        n_true = int(tmask.sum())

        gids, counts = np.unique(final_gid[tmask & (final_gid > 0)], return_counts=True)
        if len(gids) == 0:
            best_gid, tp, group_n = 0, 0, 0
        else:
            # Maximum injected-member overlap; tie -> smaller group ID.
            order = np.lexsort((gids, -counts))
            best_gid = int(gids[order[0]])
            tp = int(counts[order[0]])
            group_n = int(np.sum(final_gid == best_gid))

        C = tp / n_true
        P = tp / group_n if group_n else 0.0
        union = n_true + group_n - tp
        J = tp / union if union else 0.0

        best[label] = best_gid
        rows.append(
            {
                "template_label": label,
                "true_size": n_true,
                "best_group_id": best_gid,
                "true_members_in_best_group": tp,
                "best_group_size": group_n,
                "completeness": C,
                "purity": P,
                "jaccard": J,
                "detected_ge50pct": C >= 0.50,
                "strict_recovery_c80_p50": C >= 0.80 and P >= 0.50,
            }
        )

    tab = pd.DataFrame(rows)

    r1 = tab.loc[tab.template_label == "empirical_template_01"].iloc[0]
    r2 = tab.loc[tab.template_label == "empirical_template_02"].iloc[0]
    separate = (
        int(r1.best_group_id) > 0
        and int(r2.best_group_id) > 0
        and int(r1.best_group_id) != int(r2.best_group_id)
    )

    summary = {
        "mean_completeness": float(tab.completeness.mean()),
        "mean_purity": float(tab.purity.mean()),
        "mean_jaccard": float(tab.jaccard.mean()),
        "templates_detected_ge50pct": int(tab.detected_ge50pct.sum()),
        "strict_recoveries_of_8": int(tab.strict_recovery_c80_p50.sum()),
        "overlap_best_groups_separate": bool(separate),
        "formal_overlap_resolved": bool(
            separate
            and float(r1.completeness) >= 0.50
            and float(r2.completeness) >= 0.50
        ),
        "strict_overlap_resolved": bool(
            separate
            and bool(r1.strict_recovery_c80_p50)
            and bool(r2.strict_recovery_c80_p50)
        ),
    }
    return tab, summary



def _stage_counts(result: dict, prefix: str) -> dict:
    passes = result["passes"]
    return {
        f"{prefix}_significant_flags": (
            int(passes["significant_bimodality"].sum()) if len(passes) else 0
        ),
        f"{prefix}_accepted_splits": (
            int(passes["accepted_splits"].sum()) if len(passes) else 0
        ),
        f"{prefix}_final_groups": int(len(result["final_node_ids"])),
        f"{prefix}_orphans": int(len(result["orphans"])),
    }


def _prefixed_score(score: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in score.items()}


def run_one(run_source_str: str):
    started = time.time()
    run_source = Path(run_source_str)

    run, xyz, av, source_ids, truth, placements, done = reconstruct_field(run_source)
    # IMPORTANT: generated ONCE, then reused by both orders.
    vl, vb, angle = inject_kinematics(run, truth, done)

    base_df = pd.DataFrame(
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

    cfg = core_config()
    source_col = "source_id"

    # ---------------- S1: shared by both pipelines ----------------
    s1 = core.stage1_xmst(xyz, cfg.min_group_size)
    s1_gid = np.asarray(s1["group_ids"], dtype=np.int64)
    fracture = float(s1["fracture_scale_pc"])

    archived_fracture = float(done["stage1_fracture_scale_pc"])
    archived_groups = int(done["stage1_groups"])
    fracture_error = abs(fracture - archived_fracture)
    exact_spatial_reconstruction = (
        fracture_error < 1e-9
        and int(s1_gid.max()) == archived_groups
    )

    if not exact_spatial_reconstruction:
        raise RuntimeError(
            f"MC {run:04d}: spatial reconstruction checkpoint failed. "
            f"fracture={fracture:.15f} archived={archived_fracture:.15f}; "
            f"groups={int(s1_gid.max())} archived={archived_groups}"
        )

    # ============================================================
    # FORWARD / STANDARD ORDER: S1 -> S2 (reddening) -> S3 (kinematics)
    # ============================================================
    f_s2 = core.run_stage2(
        base_df,
        xyz,
        av,
        s1_gid,
        fracture,
        source_col,
        cfg,
    )
    f_s2_gid, f_s2_node, _ = final_nodes_to_gid(len(base_df), f_s2)

    f_s3 = core.run_stage3(
        base_df,
        xyz,
        vl,
        vb,
        f_s2["nodes"],
        f_s2["final_node_ids"],
        fracture,
        source_col,
        cfg,
    )
    f_final_gid, f_final_node, _ = final_nodes_to_gid(len(base_df), f_s3)
    f_recovery, f_score = score_truth(truth, f_final_gid)

    # ============================================================
    # REVERSE ORDER: S1 -> S3 (kinematics) -> S2 (reddening)
    # ============================================================
    r_seed_nodes, r_seed_ids = s1_seed_nodes(s1_gid)
    r_s3 = core.run_stage3(
        base_df,
        xyz,
        vl,
        vb,
        r_seed_nodes,
        r_seed_ids,
        fracture,
        source_col,
        cfg,
    )
    r_s3_gid, r_s3_node, _ = final_nodes_to_gid(len(base_df), r_s3)

    r_s2 = core.run_stage2(
        base_df,
        xyz,
        av,
        r_s3_gid,
        fracture,
        source_col,
        cfg,
    )
    r_final_gid, r_final_node, _ = final_nodes_to_gid(len(base_df), r_s2)
    r_recovery, r_score = score_truth(truth, r_final_gid)

    # Pair the template-level scores for direct per-template comparison.
    paired_recovery = f_recovery.merge(
        r_recovery,
        on=["template_label", "true_size"],
        suffixes=("_forward", "_reverse"),
        how="inner",
        validate="one_to_one",
    )
    for metric in ["completeness", "purity", "jaccard"]:
        paired_recovery[f"delta_{metric}_reverse_minus_forward"] = (
            paired_recovery[f"{metric}_reverse"]
            - paired_recovery[f"{metric}_forward"]
        )

    # Per-field paired differences.
    delta_C = float(r_score["mean_completeness"] - f_score["mean_completeness"])
    delta_P = float(r_score["mean_purity"] - f_score["mean_purity"])
    delta_J = float(r_score["mean_jaccard"] - f_score["mean_jaccard"])
    delta_strict = int(r_score["strict_recoveries_of_8"] - f_score["strict_recoveries_of_8"])

    # How many individual stars receive the same retained/unretained status?
    # (Group numeric labels are arbitrary between pipelines, so don't compare IDs.)
    same_retention_fraction = float(
        np.mean((f_final_gid > 0) == (r_final_gid > 0))
    )

    summary = {
        "realisation": run,
        "run_source": run_source.name,
        "delta_v_kms": float(_ARGS.delta_v),
        "velocity_sigma_kms": float(_ARGS.velocity_sigma),
        "velocity_direction_angle_rad": angle,
        "velocity_seed": int(_ARGS.velocity_seed_base) + run,

        "stage1_fracture_scale_pc": fracture,
        "archived_stage1_fracture_scale_pc": archived_fracture,
        "stage1_fracture_abs_error_pc": fracture_error,
        "exact_spatial_reconstruction": exact_spatial_reconstruction,
        "stage1_groups": int(s1_gid.max()),
        "stage1_grouped_stars": int(np.sum(s1_gid > 0)),

        **_stage_counts(f_s2, "forward_s2_first"),
        "forward_s2_first_grouped_stars": int(np.sum(f_s2_gid > 0)),
        **_stage_counts(f_s3, "forward_s3_second"),
        "forward_s3_second_grouped_stars": int(np.sum(f_final_gid > 0)),
        **_prefixed_score(f_score, "forward"),

        **_stage_counts(r_s3, "reverse_s3_first"),
        "reverse_s3_first_grouped_stars": int(np.sum(r_s3_gid > 0)),
        **_stage_counts(r_s2, "reverse_s2_second"),
        "reverse_s2_second_grouped_stars": int(np.sum(r_final_gid > 0)),
        **_prefixed_score(r_score, "reverse"),

        "delta_completeness_reverse_minus_forward": delta_C,
        "delta_purity_reverse_minus_forward": delta_P,
        "delta_jaccard_reverse_minus_forward": delta_J,
        "delta_strict_recoveries_reverse_minus_forward": delta_strict,
        "delta_formal_overlap_reverse_minus_forward": (
            int(r_score["formal_overlap_resolved"])
            - int(f_score["formal_overlap_resolved"])
        ),
        "delta_strict_overlap_reverse_minus_forward": (
            int(r_score["strict_overlap_resolved"])
            - int(f_score["strict_overlap_resolved"])
        ),
        "same_retention_fraction": same_retention_fraction,
        "elapsed_seconds": float(time.time() - started),
    }

    out = Path(_ARGS.output_dir) / f"mc_{run:04d}"
    out.mkdir(parents=True, exist_ok=True)

    paired_recovery.to_csv(out / "paired_template_recovery.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "paired_summary.csv", index=False)
    (out / "paired_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    f_s2["passes"].to_csv(out / "forward_s2_first_passes.csv", index=False)
    f_s2["splits"].to_csv(out / "forward_s2_first_splits.csv", index=False)
    f_s3["passes"].to_csv(out / "forward_s3_second_passes.csv", index=False)
    f_s3["splits"].to_csv(out / "forward_s3_second_splits.csv", index=False)

    r_s3["passes"].to_csv(out / "reverse_s3_first_passes.csv", index=False)
    r_s3["splits"].to_csv(out / "reverse_s3_first_splits.csv", index=False)
    r_s2["passes"].to_csv(out / "reverse_s2_second_passes.csv", index=False)
    r_s2["splits"].to_csv(out / "reverse_s2_second_splits.csv", index=False)

    if _ARGS.save_membership:
        membership = base_df.copy()
        membership["stage1_group_id"] = s1_gid
        membership["forward_s2_group_id"] = f_s2_gid
        membership["forward_s2_node_id"] = f_s2_node
        membership["forward_final_group_id"] = f_final_gid
        membership["forward_final_node_id"] = f_final_node
        membership["reverse_s3_group_id"] = r_s3_gid
        membership["reverse_s3_node_id"] = r_s3_node
        membership["reverse_final_group_id"] = r_final_gid
        membership["reverse_final_node_id"] = r_final_node
        membership.to_csv(out / "paired_final_membership.csv", index=False)

    return summary


def parse_run_number(path: Path):
    m = re.search(r"mc_(\d+)", path.name, flags=re.I)
    return int(m.group(1)) if m else 10**9


def _cmp_counts(series: pd.Series, tol: float = 1e-12):
    arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return {
        "reverse_better": int(np.sum(arr > tol)),
        "equal": int(np.sum(np.abs(arr) <= tol)),
        "reverse_worse": int(np.sum(arr < -tol)),
    }


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
    p.add_argument("--delta-v", type=float, default=5.0)
    p.add_argument("--velocity-sigma", type=float, default=1.0)
    p.add_argument("--velocity-seed-base", type=int, default=20260825)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) - 1)),
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1000)
    p.add_argument("--save-membership", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(
        [
            q for q in args.zip_dir.glob("mc_*")
            if q.is_dir()
            and (q / "DONE.json").exists()
            and (q / "placements.csv").exists()
        ],
        key=parse_run_number,
    )

    if not runs:
        runs = sorted(args.zip_dir.glob("mc_*.zip"), key=parse_run_number)

    runs = [r for r in runs if args.start <= parse_run_number(r) < args.end]

    if not runs:
        raise SystemExit(
            f"No usable mc_XXXX directories or mc_XXXX.zip files found in {args.zip_dir} "
            f"for run range [{args.start},{args.end})"
        )

    args_dict = vars(args).copy()
    args_dict["zip_dir"] = str(args.zip_dir)
    args_dict["output_dir"] = str(args.output_dir)
    args_dict["q25_csv"] = str(args.q25_csv)

    print("XMST Paper-II paired order-dependence MC")
    print("  forward: S1 -> reddening -> kinematics")
    print("  reverse: S1 -> kinematics -> reddening")
    print("  SAME reconstructed field and SAME velocity draw used for both orders")
    print(f"  runs found: {len(runs)}")
    print(f"  DeltaV: {args.delta_v:g} km/s")
    print(f"  sigma: {args.velocity_sigma:g} km/s per component")
    print(f"  velocity seed base: {args.velocity_seed_base}")
    print(f"  workers: {args.workers}")
    print(f"  output: {args.output_dir}")
    print()

    summaries = []

    def report(i, total, s):
        print(
            f"[{i:04d}/{total:04d}] MC {s['realisation']:04d} "
            f"F:J={s['forward_mean_jaccard']:.4f} "
            f"strict={s['forward_strict_recoveries_of_8']}/8 "
            f"ov={int(s['forward_formal_overlap_resolved'])} | "
            f"R:J={s['reverse_mean_jaccard']:.4f} "
            f"strict={s['reverse_strict_recoveries_of_8']}/8 "
            f"ov={int(s['reverse_formal_overlap_resolved'])} | "
            f"dJ={s['delta_jaccard_reverse_minus_forward']:+.4f}",
            flush=True,
        )

    if args.workers == 1:
        initialise_worker(str(args.q25_csv), args_dict)
        for i, r in enumerate(runs, 1):
            s = run_one(str(r))
            summaries.append(s)
            report(i, len(runs), s)
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
                    s = fut.result()
                except Exception as exc:
                    print(
                        f"FAILED {r.name}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    raise
                summaries.append(s)
                completed += 1
                report(completed, len(runs), s)

    all_runs = (
        pd.DataFrame(summaries)
        .sort_values("realisation")
        .reset_index(drop=True)
    )
    all_runs.to_csv(args.output_dir / "paired_order_all_runs.csv", index=False)

    jc = _cmp_counts(all_runs["delta_jaccard_reverse_minus_forward"])
    cc = _cmp_counts(all_runs["delta_completeness_reverse_minus_forward"])
    pc = _cmp_counts(all_runs["delta_purity_reverse_minus_forward"])
    sc = _cmp_counts(all_runs["delta_strict_recoveries_reverse_minus_forward"], tol=0.0)

    aggregate = {
        "n_runs": int(len(all_runs)),
        "comparison": "forward S1->S2->S3 versus reverse S1->S3->S2",
        "delta_definition": "reverse_minus_forward",
        "delta_v_kms": float(args.delta_v),
        "velocity_sigma_kms": float(args.velocity_sigma),
        "velocity_seed_base": int(args.velocity_seed_base),

        "all_spatial_reconstructions_exact": bool(
            all_runs.exact_spatial_reconstruction.all()
        ),
        "max_stage1_fracture_abs_error_pc": float(
            all_runs.stage1_fracture_abs_error_pc.max()
        ),

        "forward_mean_completeness": float(all_runs.forward_mean_completeness.mean()),
        "forward_mean_purity": float(all_runs.forward_mean_purity.mean()),
        "forward_mean_jaccard": float(all_runs.forward_mean_jaccard.mean()),
        "forward_mean_strict_recoveries_of_8": float(
            all_runs.forward_strict_recoveries_of_8.mean()
        ),
        "forward_formal_overlap_resolution_rate": float(
            all_runs.forward_formal_overlap_resolved.mean()
        ),
        "forward_best_groups_separate_rate": float(
            all_runs.forward_overlap_best_groups_separate.mean()
        ),
        "forward_strict_overlap_resolution_rate": float(
            all_runs.forward_strict_overlap_resolved.mean()
        ),

        "reverse_mean_completeness": float(all_runs.reverse_mean_completeness.mean()),
        "reverse_mean_purity": float(all_runs.reverse_mean_purity.mean()),
        "reverse_mean_jaccard": float(all_runs.reverse_mean_jaccard.mean()),
        "reverse_mean_strict_recoveries_of_8": float(
            all_runs.reverse_strict_recoveries_of_8.mean()
        ),
        "reverse_formal_overlap_resolution_rate": float(
            all_runs.reverse_formal_overlap_resolved.mean()
        ),
        "reverse_best_groups_separate_rate": float(
            all_runs.reverse_overlap_best_groups_separate.mean()
        ),
        "reverse_strict_overlap_resolution_rate": float(
            all_runs.reverse_strict_overlap_resolved.mean()
        ),

        "mean_delta_completeness_reverse_minus_forward": float(
            all_runs.delta_completeness_reverse_minus_forward.mean()
        ),
        "mean_delta_purity_reverse_minus_forward": float(
            all_runs.delta_purity_reverse_minus_forward.mean()
        ),
        "mean_delta_jaccard_reverse_minus_forward": float(
            all_runs.delta_jaccard_reverse_minus_forward.mean()
        ),
        "mean_delta_strict_recoveries_reverse_minus_forward": float(
            all_runs.delta_strict_recoveries_reverse_minus_forward.mean()
        ),

        "jaccard_reverse_better_runs": jc["reverse_better"],
        "jaccard_equal_runs": jc["equal"],
        "jaccard_reverse_worse_runs": jc["reverse_worse"],
        "completeness_reverse_better_runs": cc["reverse_better"],
        "completeness_equal_runs": cc["equal"],
        "completeness_reverse_worse_runs": cc["reverse_worse"],
        "purity_reverse_better_runs": pc["reverse_better"],
        "purity_equal_runs": pc["equal"],
        "purity_reverse_worse_runs": pc["reverse_worse"],
        "strict_recovery_reverse_better_runs": sc["reverse_better"],
        "strict_recovery_equal_runs": sc["equal"],
        "strict_recovery_reverse_worse_runs": sc["reverse_worse"],

        "mean_same_retention_fraction": float(all_runs.same_retention_fraction.mean()),

        "forward_mean_s2_first_accepted_splits": float(
            all_runs.forward_s2_first_accepted_splits.mean()
        ),
        "forward_mean_s3_second_accepted_splits": float(
            all_runs.forward_s3_second_accepted_splits.mean()
        ),
        "reverse_mean_s3_first_accepted_splits": float(
            all_runs.reverse_s3_first_accepted_splits.mean()
        ),
        "reverse_mean_s2_second_accepted_splits": float(
            all_runs.reverse_s2_second_accepted_splits.mean()
        ),
    }

    pd.DataFrame([aggregate]).to_csv(
        args.output_dir / "paired_order_aggregate.csv", index=False
    )
    (args.output_dir / "paired_order_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    print("\nFINAL PAIRED AGGREGATE")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

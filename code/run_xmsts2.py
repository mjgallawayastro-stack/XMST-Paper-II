#!/usr/bin/env python3
"""
XMSTS2 - XMST Stage 2
=====================

Recursive reddening-supported decomposition of the frozen 3-D XMST catalogue.

Algorithm
---------
1. Rebuild the ordinary spatial XMST solution on the full input catalogue:
      XYZ -> Delaunay -> Kruskal MST -> Percolation -> 2-class Jenks fracture.
2. Retain Stage-1 groups with N >= min_group_size.
3. For every retained group, test A_V for conservative bimodality:
      delta BIC >= 10
      Ashman D >= 2
      each GMM component >= 5 stars
      each GMM component >= 15% of the finite-A_V sample
4. ONLY if the independent bimodality test passes, obtain the A_V division
   with an exact 2-class Jenks natural-break split.
5. Rebuild a spatial MST independently for each Jenks reddening side and cut
   it using the FIXED Stage-1 global spatial fracture scale.
6. By default, accept the binary subdivision only if each reddening side
   produces exactly one retained spatial XMST component with N >= 10.
7. Repeat the same test on every accepted child, breadth-first, until an
   entire pass produces no accepted split.

Important methodological point
------------------------------
Stage 2 does NOT re-estimate a new local Percolation-Jenks spatial fracture
scale inside each child. The Stage-1 spatial scale is held fixed so that
reddening is the only new information driving the decomposition.

Dependencies
------------
numpy pandas scipy scikit-learn
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import json
import math
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.spatial import Delaunay, QhullError
from sklearn import __version__ as sklearn_version
from sklearn.mixture import GaussianMixture


# Frozen Q25 / Paper-I checkpoint.
EXPECTED_N = 24706
EXPECTED_JENKS_PC = 17.048388529750305
EXPECTED_GROUPS = 103
EXPECTED_GROUPED_STARS = 2339


@dataclass
class Config:
    input_csv: str
    output_dir: str
    min_group_size: int = 10
    delta_bic_threshold: float = 10.0
    ashman_d_threshold: float = 2.0
    min_gmm_component_n: int = 5
    min_gmm_component_fraction: float = 0.15
    min_reddening_coverage: float = 1.0
    gmm_n_init: int = 20
    gmm_random_state: int = 20260825
    spatial_child_policy: str = "single"
    max_passes: int = 20
    strict_q25_checkpoint: bool = True


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lookup = {str(c).strip().casefold(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.casefold()
        if key in lookup:
            return lookup[key]
    raise KeyError(f"Could not find any of {list(candidates)}. Columns: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Exact Euclidean XMST utilities
# ---------------------------------------------------------------------------

def complete_graph_edges(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(coords)
    ii, jj = np.triu_indices(n, 1)
    pairs = np.column_stack((ii, jj)).astype(np.int64)
    weights = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    return pairs, weights


def delaunay_edges(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sparse exact-Euclidean-MST superset; complete graph fallback elsewhere."""
    tri = Delaunay(coords)
    s = tri.simplices
    pairs = np.concatenate(
        [
            s[:, [0, 1]], s[:, [0, 2]], s[:, [0, 3]],
            s[:, [1, 2]], s[:, [1, 3]], s[:, [2, 3]],
        ],
        axis=0,
    )
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    weights = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    return pairs.astype(np.int64), weights.astype(float)


def candidate_edges(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Delaunay normally; complete graph fallback for small/degenerate subsets."""
    if len(coords) < 4:
        p, w = complete_graph_edges(coords)
        return p, w, "complete"
    try:
        p, w = delaunay_edges(coords)
        return p, w, "delaunay"
    except QhullError:
        p, w = complete_graph_edges(coords)
        return p, w, "complete_fallback"


def kruskal_mst(n: int, pairs: np.ndarray, weights: np.ndarray):
    if n < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
        )

    order = np.argsort(weights, kind="mergesort")
    pairs = pairs[order]
    weights = weights[order]

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    us, vs, ws = [], [], []
    for (u0, v0), w in zip(pairs, weights):
        u, v = int(u0), int(v0)
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
        us.append(u)
        vs.append(v)
        ws.append(float(w))
        if len(ws) == n - 1:
            break

    if len(ws) != n - 1:
        raise RuntimeError(f"MST incomplete: {len(ws)} edges for {n} vertices")

    return np.asarray(us), np.asarray(vs), np.asarray(ws)


def percolation_limit(n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    """Edge causing the largest increase in the current largest component."""
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    largest = 1
    best_jump = -1
    best_w = np.nan

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for uu, vv, ww in zip(u, v, w):
        ru, rv = find(int(uu)), find(int(vv))
        if ru == rv:
            continue
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        size[ru] += size[rv]
        new_largest = max(largest, int(size[ru]))
        jump = new_largest - largest
        if jump > best_jump:
            best_jump = jump
            best_w = float(ww)
        largest = new_largest
    return best_w


def jenks_two_class_cut(values: np.ndarray) -> float:
    """Exact 1-D two-class Jenks boundary as midpoint of the optimum adjacent pair."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise ValueError("Need at least two finite values for two-class Jenks")

    cs = np.cumsum(x)
    cs2 = np.cumsum(x * x)
    total_s = cs[-1]
    total_s2 = cs2[-1]

    i = np.arange(len(x) - 1)
    n1 = i + 1
    n2 = len(x) - n1
    s1 = cs[:-1]
    ss1 = cs2[:-1]
    s2 = total_s - s1
    ss2 = total_s2 - ss1

    sse1 = ss1 - (s1 * s1) / n1
    sse2 = ss2 - (s2 * s2) / n2
    k = int(np.argmin(sse1 + sse2))
    return float((x[k] + x[k + 1]) / 2.0)


def roots_after_cut(n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray, cut: float):
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for uu, vv, ww in zip(u, v, w):
        if ww > cut:
            break
        ru, rv = find(int(uu)), find(int(vv))
        if ru == rv:
            continue
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        size[ru] += size[rv]

    return np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)


def stage1_xmst(coords: np.ndarray, min_group_size: int):
    pairs, weights, edge_mode = candidate_edges(coords)
    u, v, w = kruskal_mst(len(coords), pairs, weights)
    pcut = percolation_limit(len(coords), u, v, w)
    subcritical = w[w <= pcut]
    if len(subcritical) < 2:
        raise RuntimeError("Too few subcritical MST edges to determine Stage-1 Jenks fracture scale")
    fracture = jenks_two_class_cut(subcritical)

    roots = roots_after_cut(len(coords), u, v, w, fracture)
    counts = pd.Series(roots).value_counts()
    retained = set(counts[counts >= min_group_size].index.astype(int))

    # Preserve the historical Paper-I global-group numbering:
    # retained components are numbered by first occurrence in the frozen input table.
    records = []
    for root in retained:
        idx = np.flatnonzero(roots == root)
        records.append((int(idx.min()), int(root), int(len(idx))))
    records.sort()
    root_to_gid = {root: i + 1 for i, (_, root, _) in enumerate(records)}

    gids = np.array([root_to_gid.get(int(r), 0) for r in roots], dtype=np.int64)
    return {
        "u": u, "v": v, "w": w,
        "percolation_limit_pc": float(pcut),
        "fracture_scale_pc": float(fracture),
        "group_ids": gids,
        "edge_mode": edge_mode,
        "group_records": records,
    }


def spatial_components_at_fixed_cut(
    global_indices: np.ndarray,
    coords_all: np.ndarray,
    fixed_cut: float,
    min_group_size: int,
):
    """
    Rebuild the exact spatial MST for one reddening side, but cut it at the
    fixed Stage-1 fracture scale. Return every retained component.
    """
    global_indices = np.asarray(global_indices, dtype=np.int64)
    if len(global_indices) < min_group_size:
        return [], "too_small"

    local_coords = coords_all[global_indices]
    pairs, weights, mode = candidate_edges(local_coords)
    u, v, w = kruskal_mst(len(global_indices), pairs, weights)
    roots = roots_after_cut(len(global_indices), u, v, w, fixed_cut)
    counts = pd.Series(roots).value_counts()
    retained_roots = set(counts[counts >= min_group_size].index.astype(int))

    comps = []
    for root in retained_roots:
        local = np.flatnonzero(roots == root)
        comps.append(global_indices[local])

    # deterministic: largest first, then earliest parent-row index
    comps.sort(key=lambda arr: (-len(arr), int(np.min(arr))))
    return comps, mode


# ---------------------------------------------------------------------------
# Reddening bimodality gate
# ---------------------------------------------------------------------------

def bimodality_test(values: np.ndarray, cfg: Config) -> dict:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    xf = x[finite]

    result = {
        "n_total": int(len(x)),
        "n_finite": int(len(xf)),
        "coverage": float(len(xf) / len(x)) if len(x) else 0.0,
        "bic_1g": np.nan,
        "bic_2g": np.nan,
        "delta_bic": np.nan,
        "ashman_d": np.nan,
        "mean_low": np.nan,
        "mean_high": np.nan,
        "sigma_low": np.nan,
        "sigma_high": np.nan,
        "weight_low": np.nan,
        "weight_high": np.nan,
        "count_low": 0,
        "count_high": 0,
        "fraction_low": np.nan,
        "fraction_high": np.nan,
        "passes_bimodality": False,
        "bimodality_reason": "",
    }

    if len(x) == 0:
        result["bimodality_reason"] = "empty"
        return result
    if result["coverage"] < cfg.min_reddening_coverage:
        result["bimodality_reason"] = "insufficient_reddening_coverage"
        return result
    if len(xf) < 2 * cfg.min_gmm_component_n:
        result["bimodality_reason"] = "too_few_finite_reddening_values"
        return result

    X = xf.reshape(-1, 1)
    g1 = GaussianMixture(
        n_components=1,
        covariance_type="full",
        n_init=cfg.gmm_n_init,
        random_state=cfg.gmm_random_state,
        reg_covar=1e-6,
    ).fit(X)
    g2 = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=cfg.gmm_n_init,
        random_state=cfg.gmm_random_state,
        reg_covar=1e-6,
    ).fit(X)

    bic1 = float(g1.bic(X))
    bic2 = float(g2.bic(X))
    delta_bic = bic1 - bic2

    means = g2.means_.reshape(-1)
    sigmas = np.sqrt(g2.covariances_.reshape(-1))
    weights = g2.weights_.reshape(-1)
    order = np.argsort(means)

    means = means[order]
    sigmas = sigmas[order]
    weights = weights[order]

    raw_labels = g2.predict(X)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(z)] for z in raw_labels], dtype=int)
    counts = np.bincount(labels, minlength=2)

    denom = math.sqrt(sigmas[0] ** 2 + sigmas[1] ** 2)
    ashman_d = (
        math.sqrt(2.0) * abs(means[1] - means[0]) / denom
        if denom > 0 else np.inf
    )

    fractions = counts / len(xf)

    result.update({
        "bic_1g": bic1,
        "bic_2g": bic2,
        "delta_bic": float(delta_bic),
        "ashman_d": float(ashman_d),
        "mean_low": float(means[0]),
        "mean_high": float(means[1]),
        "sigma_low": float(sigmas[0]),
        "sigma_high": float(sigmas[1]),
        "weight_low": float(weights[0]),
        "weight_high": float(weights[1]),
        "count_low": int(counts[0]),
        "count_high": int(counts[1]),
        "fraction_low": float(fractions[0]),
        "fraction_high": float(fractions[1]),
    })

    tests = [
        (delta_bic >= cfg.delta_bic_threshold, "delta_bic"),
        (ashman_d >= cfg.ashman_d_threshold, "ashman_d"),
        (counts[0] >= cfg.min_gmm_component_n, "component_n_low"),
        (counts[1] >= cfg.min_gmm_component_n, "component_n_high"),
        (fractions[0] >= cfg.min_gmm_component_fraction, "component_fraction_low"),
        (fractions[1] >= cfg.min_gmm_component_fraction, "component_fraction_high"),
    ]
    failed = [name for ok, name in tests if not ok]
    result["passes_bimodality"] = len(failed) == 0
    result["bimodality_reason"] = "pass" if not failed else "failed:" + ",".join(failed)
    return result


# ---------------------------------------------------------------------------
# Recursive XMSTS2
# ---------------------------------------------------------------------------

def child_label(parent: str, branch: str) -> str:
    return f"{parent}.{branch}"


def run_xmsts2(df: pd.DataFrame, cfg: Config):
    xcol = find_col(df, ["x_pc", "x"])
    ycol = find_col(df, ["y_pc", "y"])
    zcol = find_col(df, ["z_pc", "z"])
    rcol = find_col(df, ["reddening_mag", "a_v", "av", "reddening"])
    scol = find_col(df, ["source_id", "gaia_source_id", "id"])

    coords = df[[xcol, ycol, zcol]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    reddening = pd.to_numeric(df[rcol], errors="coerce").to_numpy(float)
    if not np.isfinite(coords).all():
        bad = np.flatnonzero(~np.isfinite(coords).all(axis=1))
        raise ValueError(f"Non-finite XYZ values in {len(bad)} rows; first rows: {bad[:10].tolist()}")

    print("XMSTS2: rebuilding Stage-1 XMST...", flush=True)
    s1 = stage1_xmst(coords, cfg.min_group_size)
    s1_gid = s1["group_ids"]
    fracture = s1["fracture_scale_pc"]

    n_groups = int(s1_gid.max())
    n_grouped = int(np.sum(s1_gid > 0))
    print(f"  input stars: {len(df):,}")
    print(f"  percolation limit: {s1['percolation_limit_pc']:.12f} pc")
    print(f"  Stage-1 fracture scale: {fracture:.12f} pc")
    print(f"  Stage-1 retained groups: {n_groups}")
    print(f"  Stage-1 grouped stars: {n_grouped:,}")

    if cfg.strict_q25_checkpoint and len(df) == EXPECTED_N:
        problems = []
        if not np.isclose(fracture, EXPECTED_JENKS_PC, atol=1e-9, rtol=0):
            problems.append(f"fracture {fracture} != expected {EXPECTED_JENKS_PC}")
        if n_groups != EXPECTED_GROUPS:
            problems.append(f"groups {n_groups} != expected {EXPECTED_GROUPS}")
        if n_grouped != EXPECTED_GROUPED_STARS:
            problems.append(f"grouped stars {n_grouped} != expected {EXPECTED_GROUPED_STARS}")
        if problems:
            raise RuntimeError("Q25 Stage-1 checkpoint failed: " + "; ".join(problems))
        print("  Q25 Stage-1 checkpoint reproduced successfully.")

    # Node store. Indices are always global row indices.
    nodes: dict[str, dict] = {}
    active: list[str] = []

    for gid in range(1, n_groups + 1):
        idx = np.flatnonzero(s1_gid == gid)
        label = f"G{gid:03d}"
        nodes[label] = {
            "node_id": label,
            "parent_id": "",
            "root_stage1_group_id": gid,
            "generation": 0,
            "created_pass": 0,
            "indices": idx,
            "status": "active",
        }
        active.append(label)

    split_rows = []
    orphan_records = []
    pass_rows = []

    for pass_no in range(1, cfg.max_passes + 1):
        print(f"\nXMSTS2 pass {pass_no}: testing {len(active)} active cluster(s)...", flush=True)
        accepted_this_pass = 0
        next_active: list[str] = []
        tested_this_pass = 0
        bimodal_this_pass = 0

        for node_id in active:
            node = nodes[node_id]
            idx = node["indices"]
            tested_this_pass += 1

            b = bimodality_test(reddening[idx], cfg)
            node.update(b)

            if not b["passes_bimodality"]:
                node["status"] = "final_no_significant_bimodality"
                continue

            bimodal_this_pass += 1

            av = reddening[idx]
            finite = np.isfinite(av)
            if not finite.all():
                # Current Q25 input has complete A_V. For other catalogues we do
                # not silently invent assignments for missing reddening.
                node["status"] = "final_bimodal_missing_reddening_unassigned"
                continue

            jenks_cut = jenks_two_class_cut(av)
            low_idx = idx[av <= jenks_cut]
            high_idx = idx[av > jenks_cut]

            node["jenks_reddening_cut"] = float(jenks_cut)
            node["jenks_low_n"] = int(len(low_idx))
            node["jenks_high_n"] = int(len(high_idx))

            split_rec = {
                "pass": pass_no,
                "parent_node_id": node_id,
                "root_stage1_group_id": node["root_stage1_group_id"],
                "parent_n": len(idx),
                **{k: b[k] for k in [
                    "n_finite","coverage","bic_1g","bic_2g","delta_bic","ashman_d",
                    "mean_low","mean_high","sigma_low","sigma_high",
                    "weight_low","weight_high","count_low","count_high",
                    "fraction_low","fraction_high"
                ]},
                "jenks_reddening_cut": float(jenks_cut),
                "jenks_low_n": len(low_idx),
                "jenks_high_n": len(high_idx),
                "low_retained_components": 0,
                "high_retained_components": 0,
                "low_retained_sizes": "",
                "high_retained_sizes": "",
                "orphans_n": 0,
                "accepted": False,
                "rejection_reason": "",
                "child_low": "",
                "child_high": "",
            }

            if len(low_idx) < cfg.min_group_size or len(high_idx) < cfg.min_group_size:
                node["status"] = "final_bimodal_jenks_side_below_min_group"
                split_rec["rejection_reason"] = "jenks_side_below_min_group"
                split_rows.append(split_rec)
                continue

            low_comps, low_mode = spatial_components_at_fixed_cut(
                low_idx, coords, fracture, cfg.min_group_size
            )
            high_comps, high_mode = spatial_components_at_fixed_cut(
                high_idx, coords, fracture, cfg.min_group_size
            )

            split_rec["low_retained_components"] = len(low_comps)
            split_rec["high_retained_components"] = len(high_comps)
            split_rec["low_retained_sizes"] = ";".join(str(len(c)) for c in low_comps)
            split_rec["high_retained_sizes"] = ";".join(str(len(c)) for c in high_comps)
            split_rec["low_mst_mode"] = low_mode
            split_rec["high_mst_mode"] = high_mode

            if cfg.spatial_child_policy == "single":
                if len(low_comps) != 1 or len(high_comps) != 1:
                    node["status"] = "final_bimodal_spatial_split_not_binary"
                    split_rec["rejection_reason"] = "not_exactly_one_retained_spatial_component_per_reddening_side"
                    split_rows.append(split_rec)
                    continue
                child_components = [("a", low_comps[0]), ("b", high_comps[0])]

            elif cfg.spatial_child_policy == "all":
                if len(low_comps) == 0 or len(high_comps) == 0:
                    node["status"] = "final_bimodal_one_reddening_side_has_no_spatial_component"
                    split_rec["rejection_reason"] = "one_reddening_side_has_no_retained_spatial_component"
                    split_rows.append(split_rec)
                    continue
                # This mode intentionally allows >2 descendants.
                child_components = []
                for j, comp in enumerate(low_comps):
                    child_components.append((f"a{j+1}", comp))
                for j, comp in enumerate(high_comps):
                    child_components.append((f"b{j+1}", comp))
            else:
                raise ValueError(f"Unknown spatial_child_policy={cfg.spatial_child_policy}")

            kept = np.concatenate([c for _, c in child_components])
            kept_set = set(map(int, kept))
            orphans = np.array([i for i in idx if int(i) not in kept_set], dtype=np.int64)

            node["status"] = "split"
            node["split_pass"] = pass_no
            node["orphans_n"] = int(len(orphans))
            accepted_this_pass += 1

            child_ids = []
            for suffix, comp in child_components:
                cid = child_label(node_id, suffix)
                child_ids.append(cid)
                nodes[cid] = {
                    "node_id": cid,
                    "parent_id": node_id,
                    "root_stage1_group_id": node["root_stage1_group_id"],
                    "generation": int(node["generation"]) + 1,
                    "created_pass": pass_no,
                    "indices": np.asarray(comp, dtype=np.int64),
                    "status": "active",
                }
                next_active.append(cid)

            # Convenient binary columns in the normal default.
            if len(child_ids) >= 1:
                split_rec["child_low"] = child_ids[0]
            if len(child_ids) >= 2:
                split_rec["child_high"] = child_ids[-1]

            split_rec["orphans_n"] = int(len(orphans))
            split_rec["accepted"] = True
            split_rows.append(split_rec)

            for row_idx in orphans:
                orphan_records.append({
                    "row_index": int(row_idx),
                    "source_id": str(df.iloc[row_idx][scol]),
                    "stage1_group_id": int(node["root_stage1_group_id"]),
                    "parent_node_id": node_id,
                    "split_pass": pass_no,
                    "reason": "removed_by_fixed_scale_spatial_rerun_after_reddening_split",
                })

        pass_rows.append({
            "pass": pass_no,
            "clusters_tested": tested_this_pass,
            "significant_bimodality": bimodal_this_pass,
            "accepted_splits": accepted_this_pass,
            "new_children": len(next_active),
        })

        print(
            f"  significant bimodality: {bimodal_this_pass}; "
            f"accepted splits: {accepted_this_pass}; "
            f"new child clusters: {len(next_active)}"
        )

        if accepted_this_pass == 0:
            print("  stopping: no accepted split in this pass.")
            active = []
            break

        active = next_active
    else:
        raise RuntimeError(
            f"Reached max_passes={cfg.max_passes} while accepted splits were still being generated. "
            "Increase --max-passes only after inspecting the genealogy."
        )

    # Any still-active nodes only occur if loop structure changes; safeguard.
    for node_id in active:
        if nodes[node_id]["status"] == "active":
            nodes[node_id]["status"] = "final_max_passes"

    # Final leaves = nodes never accepted as split.
    final_node_ids = [nid for nid, n in nodes.items() if n["status"] != "split"]
    final_node_ids.sort(
        key=lambda nid: (
            int(nodes[nid]["root_stage1_group_id"]),
            int(nodes[nid]["generation"]),
            nid,
        )
    )

    final_group_map = {nid: i + 1 for i, nid in enumerate(final_node_ids)}

    membership = df.copy()
    membership.insert(0, "row_index", np.arange(len(df), dtype=np.int64))
    membership["stage1_group_id"] = s1_gid
    membership["xmsts2_final_group_id"] = 0
    membership["xmsts2_node_id"] = ""
    membership["xmsts2_generation"] = np.nan
    membership["xmsts2_member_status"] = np.where(s1_gid > 0, "stage1_grouped", "stage1_field")

    # Mark all final cluster members.
    for nid in final_node_ids:
        idx = nodes[nid]["indices"]
        gid = final_group_map[nid]
        membership.loc[idx, "xmsts2_final_group_id"] = gid
        membership.loc[idx, "xmsts2_node_id"] = nid
        membership.loc[idx, "xmsts2_generation"] = nodes[nid]["generation"]
        membership.loc[idx, "xmsts2_member_status"] = "xmsts2_final_cluster"

    # Explicitly mark Stage-2 orphans.
    orphan_idx = [r["row_index"] for r in orphan_records]
    if orphan_idx:
        membership.loc[orphan_idx, "xmsts2_final_group_id"] = 0
        membership.loc[orphan_idx, "xmsts2_node_id"] = ""
        membership.loc[orphan_idx, "xmsts2_generation"] = np.nan
        membership.loc[orphan_idx, "xmsts2_member_status"] = "xmsts2_orphan"

    # Node table.
    node_rows = []
    for nid, n in nodes.items():
        idx = n["indices"]
        row = {
            "node_id": nid,
            "parent_id": n.get("parent_id", ""),
            "root_stage1_group_id": n["root_stage1_group_id"],
            "generation": n["generation"],
            "created_pass": n["created_pass"],
            "status": n["status"],
            "n_members": len(idx),
            "median_reddening": float(np.nanmedian(reddening[idx])) if len(idx) else np.nan,
            "mean_reddening": float(np.nanmean(reddening[idx])) if len(idx) else np.nan,
            "final_group_id": final_group_map.get(nid, 0),
        }
        for key in [
            "n_total","n_finite","coverage","bic_1g","bic_2g","delta_bic","ashman_d",
            "mean_low","mean_high","sigma_low","sigma_high","weight_low","weight_high",
            "count_low","count_high","fraction_low","fraction_high",
            "passes_bimodality","bimodality_reason","jenks_reddening_cut",
            "jenks_low_n","jenks_high_n","split_pass","orphans_n"
        ]:
            row[key] = n.get(key, np.nan)
        node_rows.append(row)

    nodes_df = pd.DataFrame(node_rows).sort_values(
        ["root_stage1_group_id", "generation", "node_id"]
    )
    splits_df = pd.DataFrame(split_rows)
    passes_df = pd.DataFrame(pass_rows)
    orphans_df = pd.DataFrame(orphan_records)

    # Final group summary.
    fg_rows = []
    for nid in final_node_ids:
        n = nodes[nid]
        idx = n["indices"]
        fg_rows.append({
            "xmsts2_final_group_id": final_group_map[nid],
            "xmsts2_node_id": nid,
            "root_stage1_group_id": n["root_stage1_group_id"],
            "generation": n["generation"],
            "n_members": len(idx),
            "median_reddening": float(np.nanmedian(reddening[idx])),
            "mean_reddening": float(np.nanmean(reddening[idx])),
            "xmin_pc": float(np.min(coords[idx,0])),
            "xmax_pc": float(np.max(coords[idx,0])),
            "ymin_pc": float(np.min(coords[idx,1])),
            "ymax_pc": float(np.max(coords[idx,1])),
            "zmin_pc": float(np.min(coords[idx,2])),
            "zmax_pc": float(np.max(coords[idx,2])),
            "terminal_status": n["status"],
        })
    final_groups_df = pd.DataFrame(fg_rows)

    summary = {
        "input_stars": len(df),
        "stage1_percolation_limit_pc": s1["percolation_limit_pc"],
        "stage1_fracture_scale_pc": fracture,
        "stage1_groups": n_groups,
        "stage1_grouped_stars": n_grouped,
        "xmsts2_passes_run": len(passes_df),
        "xmsts2_significant_bimodality_tests_total": int(passes_df["significant_bimodality"].sum()) if len(passes_df) else 0,
        "xmsts2_accepted_splits_total": int(passes_df["accepted_splits"].sum()) if len(passes_df) else 0,
        "xmsts2_final_groups": len(final_node_ids),
        "xmsts2_grouped_stars": int((membership["xmsts2_final_group_id"] > 0).sum()),
        "xmsts2_orphans": int((membership["xmsts2_member_status"] == "xmsts2_orphan").sum()),
    }

    return {
        "membership": membership,
        "nodes": nodes_df,
        "splits": splits_df,
        "passes": passes_df,
        "orphans": orphans_df,
        "final_groups": final_groups_df,
        "summary": summary,
        "stage1": s1,
    }


def write_outputs(result: dict, cfg: Config):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result["membership"].to_csv(out / "xmsts2_final_membership.csv", index=False)
    result["nodes"].to_csv(out / "xmsts2_nodes.csv", index=False)
    result["splits"].to_csv(out / "xmsts2_splits.csv", index=False)
    result["passes"].to_csv(out / "xmsts2_passes.csv", index=False)
    result["orphans"].to_csv(out / "xmsts2_orphans.csv", index=False)
    result["final_groups"].to_csv(out / "xmsts2_final_groups.csv", index=False)

    config_payload = asdict(cfg)
    config_payload["python_version"] = sys.version
    config_payload["platform"] = platform.platform()
    config_payload["numpy_version"] = np.__version__
    config_payload["pandas_version"] = pd.__version__
    config_payload["scipy_version"] = scipy.__version__
    config_payload["sklearn_version"] = sklearn_version
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    s = result["summary"]
    lines = [
        "XMSTS2 SUMMARY",
        "==============",
        "",
        f"Input stars: {s['input_stars']:,}",
        f"Stage-1 percolation limit: {s['stage1_percolation_limit_pc']:.12f} pc",
        f"Stage-1 fracture scale: {s['stage1_fracture_scale_pc']:.12f} pc",
        f"Stage-1 groups: {s['stage1_groups']}",
        f"Stage-1 grouped stars: {s['stage1_grouped_stars']:,}",
        "",
        f"XMSTS2 passes run: {s['xmsts2_passes_run']}",
        f"Significant bimodality flags tested across passes: {s['xmsts2_significant_bimodality_tests_total']}",
        f"Accepted Stage-2 splits: {s['xmsts2_accepted_splits_total']}",
        f"Final XMSTS2 groups: {s['xmsts2_final_groups']}",
        f"Final XMSTS2 grouped stars: {s['xmsts2_grouped_stars']:,}",
        f"Stage-2 spatial orphans: {s['xmsts2_orphans']}",
        "",
        "Bimodality gate:",
        f"  delta BIC >= {cfg.delta_bic_threshold}",
        f"  Ashman D >= {cfg.ashman_d_threshold}",
        f"  each GMM component >= {cfg.min_gmm_component_n} stars",
        f"  each GMM component >= {100*cfg.min_gmm_component_fraction:.1f}% of finite A_V values",
        "",
        f"Spatial child policy: {cfg.spatial_child_policy}",
        "Spatial reruns use the fixed Stage-1 fracture scale.",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines), flush=True)
    print(f"\nOutputs written to: {out.resolve()}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="XMSTS2: recursive reddening-gated XMST Stage-2 decomposition"
    )
    p.add_argument(
        "input_csv", nargs="?", default="quintana_2025_mst_input.csv",
        help="Input CSV containing source_id, x_pc, y_pc, z_pc and reddening_mag"
    )
    p.add_argument("--output-dir", default="xmsts2_output")
    p.add_argument("--min-group-size", type=int, default=10)
    p.add_argument("--delta-bic", type=float, default=10.0)
    p.add_argument("--ashman-d", type=float, default=2.0)
    p.add_argument("--min-gmm-component-n", type=int, default=5)
    p.add_argument("--min-gmm-component-fraction", type=float, default=0.15)
    p.add_argument("--min-reddening-coverage", type=float, default=1.0)
    p.add_argument("--gmm-n-init", type=int, default=20)
    p.add_argument("--gmm-random-state", type=int, default=20260825)
    p.add_argument(
        "--spatial-child-policy",
        choices=["single", "all"],
        default="single",
        help=(
            "'single' (default): accept only a clean binary split with exactly one "
            "retained spatial component per Jenks reddening side. "
            "'all': retain all N>=10 spatial components from both sides."
        ),
    )
    p.add_argument("--max-passes", type=int, default=20)
    p.add_argument(
        "--no-strict-q25-checkpoint",
        action="store_true",
        help="Do not enforce the 24,706-star Paper-I checkpoint."
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        min_group_size=args.min_group_size,
        delta_bic_threshold=args.delta_bic,
        ashman_d_threshold=args.ashman_d,
        min_gmm_component_n=args.min_gmm_component_n,
        min_gmm_component_fraction=args.min_gmm_component_fraction,
        min_reddening_coverage=args.min_reddening_coverage,
        gmm_n_init=args.gmm_n_init,
        gmm_random_state=args.gmm_random_state,
        spatial_child_policy=args.spatial_child_policy,
        max_passes=args.max_passes,
        strict_q25_checkpoint=not args.no_strict_q25_checkpoint,
    )

    df = pd.read_csv(cfg.input_csv, low_memory=False)
    result = run_xmsts2(df, cfg)
    write_outputs(result, cfg)


if __name__ == "__main__":
    main()

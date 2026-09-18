#!/usr/bin/env python3
"""
XMST-S3
=======

Hierarchical extension of the frozen spatial XMST catalogue:

    Stage 1: XYZ spatial XMST
    Stage 2: recursive reddening-supported decomposition (A_V)
    Stage 3: recursive transverse-kinematic decomposition (V_l, V_b)

The central methodological rule is deliberately strict:

* the global Stage-1 spatial fracture scale is determined ONCE;
* Stage 2 does not re-estimate a spatial fracture scale;
* Stage 3 does not re-estimate a spatial fracture scale;
* no conversion between pc and km/s is ever introduced.

Stage 2
-------
For every retained spatial group, fit one- and two-component 1-D Gaussian
mixtures to A_V. A candidate split must satisfy the configured evidence gate
(delta BIC, Ashman D, minimum component size and fraction). Only after that
independent gate passes is exact two-class Jenks used to place the A_V split.
Each reddening side is then rebuilt as a spatial MST and cut at the frozen
Stage-1 fracture scale. The process recurses until no accepted split remains.

Stage 3
-------
Every final Stage-2 tree is then tested in the two-dimensional transverse
velocity plane (V_l, V_b). One- and two-component full-covariance Gaussian
mixtures are compared. The separation statistic is the pooled-covariance
Mahalanobis distance between the two fitted centroids:

    D_2D = sqrt( (mu1-mu2)^T [0.5(C1+C2)]^-1 (mu1-mu2) )

In one dimension this is exactly the usual Ashman D:

    sqrt(2) |mu1-mu2| / sqrt(sigma1^2 + sigma2^2).

A significant kinematic split is assigned by the two-component GMM itself;
there is no Jenks step because the discriminator is two-dimensional. Each
kinematic component is then rebuilt as a spatial MST and cut at the SAME
frozen Stage-1 fracture scale. All accepted descendants are retested
recursively until no further kinematic split is accepted.

Kinematics input
----------------
The frozen Q25 spatial/reddening table does not contain proper motions, so
XMST-S3 accepts a separate source-ID keyed kinematics CSV via
--kinematics-csv. Two input modes are supported:

1. Preferred: LSR-corrected transverse velocities already supplied. Use
   --vl-col and --vb-col if their names are not auto-detected.

2. Raw Gaia astrometry: RA, Dec, pmRA, pmDec supplied in the kinematics CSV.
   The code transforms the proper motions to Galactic (mu_l*, mu_b), converts
   them to km/s using the FROZEN Q25 distance_pc values, and corrects only for
   the solar peculiar motion relative to the LSR using the same Schönrich,
   Binney & Dehnen (2010) values stated by Quintana et al. (2026):

       (U, V, W)_sun = (11.1, 12.24, 7.25) km/s.

   This path requires astropy. Radial velocity is not required because only
   transverse components are used.

Important
---------
The default Stage-3 evidence thresholds deliberately mirror the Stage-2 gate
(delta BIC >= 10, separation D >= 2, component n >= 5, component fraction
>= 15%). This prevents tuning them to reproduce Q26. They are starting values,
not a claim of validation: the kinematic pass must be tested on controlled
Monte Carlo catalogues before publication use.

Dependencies
------------
numpy pandas scipy scikit-learn
astropy only if raw RA/Dec/pmRA/pmDec must be converted to V_l,V_b.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional
import json
import math
import platform
import re
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.spatial import Delaunay, QhullError
from sklearn import __version__ as sklearn_version
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Frozen Q25 / Paper-I checkpoints
# ---------------------------------------------------------------------------

EXPECTED_N = 24706
EXPECTED_JENKS_PC = 17.048388529750305
EXPECTED_GROUPS = 103
EXPECTED_GROUPED_STARS = 2339

# Latest all-descendants XMST-S2 checkpoint, using the defaults below.
EXPECTED_S2_GROUPS_ALL = 108
EXPECTED_S2_GROUPED_STARS_ALL = 2332
EXPECTED_S2_ORPHANS_ALL = 7
EXPECTED_S2_ACCEPTED_SPLITS_ALL = 3
EXPECTED_S2_PASSES_ALL = 3

# Q26 LSR correction values (Schönrich, Binney & Dehnen 2010), km/s.
DEFAULT_SOLAR_U = 11.1
DEFAULT_SOLAR_V = 12.24
DEFAULT_SOLAR_W = 7.25
KMS_PER_MASYR_KPC = 4.74047


@dataclass
class Config:
    input_csv: str
    kinematics_csv: Optional[str]
    output_dir: str

    min_group_size: int = 10

    # Stage 2: reddening
    s2_delta_bic_threshold: float = 10.0
    s2_ashman_d_threshold: float = 2.0
    s2_min_gmm_component_n: int = 5
    s2_min_gmm_component_fraction: float = 0.15
    s2_min_reddening_coverage: float = 1.0
    s2_gmm_n_init: int = 20
    s2_gmm_random_state: int = 20260825
    s2_spatial_child_policy: str = "all"
    max_s2_passes: int = 20

    # Stage 3: transverse kinematics
    s3_delta_bic_threshold: float = 10.0
    s3_separation_d_threshold: float = 2.0
    s3_min_gmm_component_n: int = 5
    s3_min_gmm_component_fraction: float = 0.15
    s3_min_kinematic_coverage: float = 1.0
    s3_gmm_n_init: int = 20
    s3_gmm_random_state: int = 20260825
    s3_spatial_child_policy: str = "all"
    max_s3_passes: int = 20

    # Kinematic input controls
    vl_col: Optional[str] = None
    vb_col: Optional[str] = None
    velocity_frame: str = "lsr"  # only relevant to direct velocity columns
    solar_u_kms: float = DEFAULT_SOLAR_U
    solar_v_kms: float = DEFAULT_SOLAR_V
    solar_w_kms: float = DEFAULT_SOLAR_W

    strict_q25_checkpoint: bool = True
    strict_s2_checkpoint: bool = True
    skip_s3: bool = False


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def find_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    lookup = {str(c).strip().casefold(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.casefold()
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(f"Could not find any of {list(candidates)}. Columns: {list(df.columns)}")
    return None


def canonical_source_id(value) -> str:
    """Preserve Gaia source IDs exactly when read as strings/integers."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    # Common nuisance if an integer ID has been exported as e.g. 12345.0.
    if re.fullmatch(r"[+-]?\d+\.0+", s):
        return s.split(".", 1)[0]
    return s


def read_csv_preserve_ids(path: str | Path) -> pd.DataFrame:
    # Reading as strings avoids catastrophic precision loss if a Gaia source_id
    # is ever inferred as floating point. Numeric columns are converted explicitly.
    return pd.read_csv(path, dtype=str, low_memory=False)


def numeric_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)


# ---------------------------------------------------------------------------
# Exact Euclidean XMST utilities (same spatial machinery as XMST-S2)
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

    records = []
    for root in retained:
        idx = np.flatnonzero(roots == root)
        records.append((int(idx.min()), int(root), int(len(idx))))
    records.sort()
    root_to_gid = {root: i + 1 for i, (_, root, _) in enumerate(records)}

    gids = np.array([root_to_gid.get(int(r), 0) for r in roots], dtype=np.int64)
    return {
        "u": u,
        "v": v,
        "w": w,
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
    """Rebuild a spatial subtree and cut it at the frozen Stage-1 fracture scale."""
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

    comps.sort(key=lambda arr: (-len(arr), int(np.min(arr))))
    return comps, mode


# ---------------------------------------------------------------------------
# Stage 2: reddening evidence gate
# ---------------------------------------------------------------------------


def reddening_bimodality_test(values: np.ndarray, cfg: Config) -> dict:
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
    if result["coverage"] < cfg.s2_min_reddening_coverage:
        result["bimodality_reason"] = "insufficient_reddening_coverage"
        return result
    if len(xf) < 2 * cfg.s2_min_gmm_component_n:
        result["bimodality_reason"] = "too_few_finite_reddening_values"
        return result

    X = xf.reshape(-1, 1)
    g1 = GaussianMixture(
        n_components=1,
        covariance_type="full",
        n_init=cfg.s2_gmm_n_init,
        random_state=cfg.s2_gmm_random_state,
        reg_covar=1e-6,
    ).fit(X)
    g2 = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=cfg.s2_gmm_n_init,
        random_state=cfg.s2_gmm_random_state,
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
        math.sqrt(2.0) * abs(means[1] - means[0]) / denom if denom > 0 else np.inf
    )
    fractions = counts / len(xf)

    result.update(
        {
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
        }
    )

    tests = [
        (delta_bic >= cfg.s2_delta_bic_threshold, "delta_bic"),
        (ashman_d >= cfg.s2_ashman_d_threshold, "ashman_d"),
        (counts[0] >= cfg.s2_min_gmm_component_n, "component_n_low"),
        (counts[1] >= cfg.s2_min_gmm_component_n, "component_n_high"),
        (fractions[0] >= cfg.s2_min_gmm_component_fraction, "component_fraction_low"),
        (fractions[1] >= cfg.s2_min_gmm_component_fraction, "component_fraction_high"),
    ]
    failed = [name for ok, name in tests if not ok]
    result["passes_bimodality"] = len(failed) == 0
    result["bimodality_reason"] = "pass" if not failed else "failed:" + ",".join(failed)
    return result


def run_stage2(
    df: pd.DataFrame,
    coords: np.ndarray,
    reddening: np.ndarray,
    s1_gid: np.ndarray,
    fracture: float,
    source_col: str,
    cfg: Config,
):
    n_groups = int(s1_gid.max())

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

    for pass_no in range(1, cfg.max_s2_passes + 1):
        print(f"\nXMST-S2 pass {pass_no}: testing {len(active)} active cluster(s)...", flush=True)
        accepted_this_pass = 0
        next_active: list[str] = []
        tested_this_pass = 0
        bimodal_this_pass = 0

        for node_id in active:
            node = nodes[node_id]
            idx = node["indices"]
            tested_this_pass += 1

            b = reddening_bimodality_test(reddening[idx], cfg)
            node.update(b)

            if not b["passes_bimodality"]:
                node["status"] = "final_no_significant_reddening_bimodality"
                continue

            bimodal_this_pass += 1
            av = reddening[idx]
            if not np.isfinite(av).all():
                node["status"] = "final_reddening_bimodal_missing_values_unassigned"
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
                **{
                    k: b[k]
                    for k in [
                        "n_finite",
                        "coverage",
                        "bic_1g",
                        "bic_2g",
                        "delta_bic",
                        "ashman_d",
                        "mean_low",
                        "mean_high",
                        "sigma_low",
                        "sigma_high",
                        "weight_low",
                        "weight_high",
                        "count_low",
                        "count_high",
                        "fraction_low",
                        "fraction_high",
                    ]
                },
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
                "child_ids": "",
            }

            if len(low_idx) < cfg.min_group_size or len(high_idx) < cfg.min_group_size:
                node["status"] = "final_reddening_bimodal_jenks_side_below_min_group"
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

            if cfg.s2_spatial_child_policy == "single":
                if len(low_comps) != 1 or len(high_comps) != 1:
                    node["status"] = "final_reddening_bimodal_spatial_split_not_binary"
                    split_rec["rejection_reason"] = (
                        "not_exactly_one_retained_spatial_component_per_reddening_side"
                    )
                    split_rows.append(split_rec)
                    continue
                child_components = [("a", low_comps[0]), ("b", high_comps[0])]

            elif cfg.s2_spatial_child_policy == "all":
                if len(low_comps) == 0 or len(high_comps) == 0:
                    node["status"] = "final_reddening_bimodal_one_side_no_spatial_component"
                    split_rec["rejection_reason"] = (
                        "one_reddening_side_has_no_retained_spatial_component"
                    )
                    split_rows.append(split_rec)
                    continue
                child_components = []
                for j, comp in enumerate(low_comps):
                    child_components.append((f"a{j+1}", comp))
                for j, comp in enumerate(high_comps):
                    child_components.append((f"b{j+1}", comp))
            else:
                raise ValueError(f"Unknown s2_spatial_child_policy={cfg.s2_spatial_child_policy}")

            kept = np.concatenate([c for _, c in child_components])
            kept_set = set(map(int, kept))
            orphans = np.array([i for i in idx if int(i) not in kept_set], dtype=np.int64)

            node["status"] = "split"
            node["split_pass"] = pass_no
            node["orphans_n"] = int(len(orphans))
            accepted_this_pass += 1

            child_ids = []
            for suffix, comp in child_components:
                cid = f"{node_id}.{suffix}"
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

            split_rec["child_ids"] = ";".join(child_ids)
            split_rec["orphans_n"] = int(len(orphans))
            split_rec["accepted"] = True
            split_rows.append(split_rec)

            for row_idx in orphans:
                orphan_records.append(
                    {
                        "row_index": int(row_idx),
                        "source_id": canonical_source_id(df.iloc[row_idx][source_col]),
                        "stage": "S2",
                        "stage1_group_id": int(node["root_stage1_group_id"]),
                        "parent_node_id": node_id,
                        "split_pass": pass_no,
                        "reason": "removed_by_fixed_scale_spatial_rerun_after_reddening_split",
                    }
                )

        pass_rows.append(
            {
                "pass": pass_no,
                "clusters_tested": tested_this_pass,
                "significant_bimodality": bimodal_this_pass,
                "accepted_splits": accepted_this_pass,
                "new_children": len(next_active),
            }
        )

        print(
            f"  significant reddening bimodality: {bimodal_this_pass}; "
            f"accepted splits: {accepted_this_pass}; "
            f"new child clusters: {len(next_active)}"
        )

        if accepted_this_pass == 0:
            print("  stopping S2: no accepted split in this pass.")
            active = []
            break
        active = next_active
    else:
        raise RuntimeError(
            f"Reached max_s2_passes={cfg.max_s2_passes} while accepted splits were still generated."
        )

    final_node_ids = [nid for nid, n in nodes.items() if n["status"] != "split"]
    final_node_ids.sort(
        key=lambda nid: (
            int(nodes[nid]["root_stage1_group_id"]),
            int(nodes[nid]["generation"]),
            nid,
        )
    )

    return {
        "nodes": nodes,
        "final_node_ids": final_node_ids,
        "splits": pd.DataFrame(split_rows),
        "passes": pd.DataFrame(pass_rows),
        "orphans": pd.DataFrame(orphan_records),
    }


# ---------------------------------------------------------------------------
# Kinematics preparation
# ---------------------------------------------------------------------------


DIRECT_VL_CANDIDATES = [
    "v_l_lsr_kms",
    "vl_lsr_kms",
    "v_l_lsr",
    "vl_lsr",
    "V_l_LSR",
]
DIRECT_VB_CANDIDATES = [
    "v_b_lsr_kms",
    "vb_lsr_kms",
    "v_b_lsr",
    "vb_lsr",
    "V_b_LSR",
]


def solar_transverse_projection(l_rad, b_rad, u, v, w):
    """Project the solar peculiar velocity onto Galactic l and b basis vectors."""
    corr_l = -u * np.sin(l_rad) + v * np.cos(l_rad)
    corr_b = (
        -u * np.cos(l_rad) * np.sin(b_rad)
        - v * np.sin(l_rad) * np.sin(b_rad)
        + w * np.cos(b_rad)
    )
    return corr_l, corr_b


def prepare_kinematics(base_df: pd.DataFrame, kin_df: pd.DataFrame, cfg: Config):
    base_source_col = find_col(base_df, ["source_id", "gaia_source_id", "id"])
    kin_source_col = find_col(kin_df, ["source_id", "gaia_source_id", "id"])

    base_ids = base_df[base_source_col].map(canonical_source_id)
    kin_ids = kin_df[kin_source_col].map(canonical_source_id)

    if (kin_ids == "").any():
        raise ValueError("Kinematics table contains blank source IDs")
    if kin_ids.duplicated().any():
        dup = kin_ids[kin_ids.duplicated()].head(10).tolist()
        raise ValueError(f"Kinematics table contains duplicate source IDs; first duplicates: {dup}")

    kin = kin_df.copy()
    kin["__sid__"] = kin_ids
    kin = kin.set_index("__sid__", drop=False)

    # Align rows exactly to the frozen spatial/reddening table.
    aligned = kin.reindex(base_ids.to_numpy())
    matched = aligned[kin_source_col].notna().to_numpy()

    # Direct LSR-corrected transverse velocities.
    vl_col = cfg.vl_col or find_col(kin_df, DIRECT_VL_CANDIDATES, required=False)
    vb_col = cfg.vb_col or find_col(kin_df, DIRECT_VB_CANDIDATES, required=False)

    if (vl_col is None) ^ (vb_col is None):
        raise ValueError("Found/provided only one of V_l and V_b; both are required")

    if vl_col is not None and vb_col is not None:
        vl = pd.to_numeric(aligned[vl_col], errors="coerce").to_numpy(float)
        vb = pd.to_numeric(aligned[vb_col], errors="coerce").to_numpy(float)

        if cfg.velocity_frame == "lsr":
            mode = f"direct_lsr_velocity_columns:{vl_col},{vb_col}"
        elif cfg.velocity_frame == "heliocentric":
            lcol = find_col(kin_df, ["l_deg", "gal_l_deg", "l", "glon"])
            bcol = find_col(kin_df, ["b_deg", "gal_b_deg", "b", "glat"])
            l = np.deg2rad(pd.to_numeric(aligned[lcol], errors="coerce").to_numpy(float))
            b = np.deg2rad(pd.to_numeric(aligned[bcol], errors="coerce").to_numpy(float))
            corr_l, corr_b = solar_transverse_projection(
                l, b, cfg.solar_u_kms, cfg.solar_v_kms, cfg.solar_w_kms
            )
            vl = vl + corr_l
            vb = vb + corr_b
            mode = f"direct_heliocentric_velocity_columns_LSR_corrected:{vl_col},{vb_col}"
        else:
            raise ValueError("velocity_frame must be 'lsr' or 'heliocentric'")

        return {
            "vl_lsr_kms": vl,
            "vb_lsr_kms": vb,
            "matched_source": matched,
            "mode": mode,
        }

    # Otherwise derive from raw Gaia astrometry.
    racol = find_col(kin_df, ["ra_deg", "ra", "ra_icrs"])
    deccol = find_col(kin_df, ["dec_deg", "dec", "de_icrs"])
    pmracol = find_col(kin_df, ["pmra_masyr", "pmra", "pm_ra"])
    pmdeccol = find_col(kin_df, ["pmdec_masyr", "pmdec", "pm_dec"])
    dcol = find_col(base_df, ["distance_pc", "distance", "d_pc"])

    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except ImportError as exc:
        raise RuntimeError(
            "Raw proper motions were supplied but astropy is not installed. "
            "Install astropy or supply precomputed LSR-corrected V_l,V_b columns."
        ) from exc

    ra = pd.to_numeric(aligned[racol], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(aligned[deccol], errors="coerce").to_numpy(float)
    pmra = pd.to_numeric(aligned[pmracol], errors="coerce").to_numpy(float)
    pmdec = pd.to_numeric(aligned[pmdeccol], errors="coerce").to_numpy(float)
    distance_pc = pd.to_numeric(base_df[dcol], errors="coerce").to_numpy(float)

    finite = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(pmra)
        & np.isfinite(pmdec)
        & np.isfinite(distance_pc)
        & (distance_pc > 0)
    )

    vl = np.full(len(base_df), np.nan, dtype=float)
    vb = np.full(len(base_df), np.nan, dtype=float)

    if np.any(finite):
        sc = SkyCoord(
            ra=ra[finite] * u.deg,
            dec=dec[finite] * u.deg,
            distance=distance_pc[finite] * u.pc,
            pm_ra_cosdec=pmra[finite] * u.mas / u.yr,
            pm_dec=pmdec[finite] * u.mas / u.yr,
            frame="icrs",
        )
        gal = sc.galactic
        mu_l_cosb = gal.pm_l_cosb.to_value(u.mas / u.yr)
        mu_b = gal.pm_b.to_value(u.mas / u.yr)
        l = gal.l.to_value(u.rad)
        b = gal.b.to_value(u.rad)
        d_kpc = distance_pc[finite] / 1000.0

        vl_helio = KMS_PER_MASYR_KPC * mu_l_cosb * d_kpc
        vb_helio = KMS_PER_MASYR_KPC * mu_b * d_kpc
        corr_l, corr_b = solar_transverse_projection(
            l, b, cfg.solar_u_kms, cfg.solar_v_kms, cfg.solar_w_kms
        )
        vl[finite] = vl_helio + corr_l
        vb[finite] = vb_helio + corr_b

    return {
        "vl_lsr_kms": vl,
        "vb_lsr_kms": vb,
        "matched_source": matched,
        "mode": (
            "derived_from_ra_dec_pmra_pmdec_using_frozen_Q25_distance_pc;"
            f"LSR_solar_UVW=({cfg.solar_u_kms},{cfg.solar_v_kms},{cfg.solar_w_kms})"
        ),
    }


# ---------------------------------------------------------------------------
# Stage 3: 2-D kinematic evidence gate
# ---------------------------------------------------------------------------


def kinematic_bimodality_test(vectors: np.ndarray, cfg: Config) -> dict:
    V = np.asarray(vectors, dtype=float)
    finite = np.isfinite(V).all(axis=1)
    Vf = V[finite]

    result = {
        "n_total": int(len(V)),
        "n_finite": int(len(Vf)),
        "coverage": float(len(Vf) / len(V)) if len(V) else 0.0,
        "bic_1g": np.nan,
        "bic_2g": np.nan,
        "delta_bic": np.nan,
        "separation_d": np.nan,
        "centroid_separation_kms": np.nan,
        "mean_vl_1": np.nan,
        "mean_vb_1": np.nan,
        "mean_vl_2": np.nan,
        "mean_vb_2": np.nan,
        "cov_vl_vl_1": np.nan,
        "cov_vl_vb_1": np.nan,
        "cov_vb_vb_1": np.nan,
        "cov_vl_vl_2": np.nan,
        "cov_vl_vb_2": np.nan,
        "cov_vb_vb_2": np.nan,
        "weight_1": np.nan,
        "weight_2": np.nan,
        "count_1": 0,
        "count_2": 0,
        "fraction_1": np.nan,
        "fraction_2": np.nan,
        "passes_bimodality": False,
        "bimodality_reason": "",
        "finite_mask": finite,
        "finite_labels": None,
    }

    if len(V) == 0:
        result["bimodality_reason"] = "empty"
        return result
    if result["coverage"] < cfg.s3_min_kinematic_coverage:
        result["bimodality_reason"] = "insufficient_kinematic_coverage"
        return result
    if len(Vf) < 2 * cfg.s3_min_gmm_component_n:
        result["bimodality_reason"] = "too_few_finite_kinematic_values"
        return result

    g1 = GaussianMixture(
        n_components=1,
        covariance_type="full",
        n_init=cfg.s3_gmm_n_init,
        random_state=cfg.s3_gmm_random_state,
        reg_covar=1e-6,
    ).fit(Vf)
    g2 = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=cfg.s3_gmm_n_init,
        random_state=cfg.s3_gmm_random_state,
        reg_covar=1e-6,
    ).fit(Vf)

    bic1 = float(g1.bic(Vf))
    bic2 = float(g2.bic(Vf))
    delta_bic = bic1 - bic2

    means = np.asarray(g2.means_, dtype=float)
    covs = np.asarray(g2.covariances_, dtype=float)
    weights = np.asarray(g2.weights_, dtype=float)

    # Deterministic component ordering: first by V_l centroid, then V_b.
    order = np.lexsort((means[:, 1], means[:, 0]))
    means = means[order]
    covs = covs[order]
    weights = weights[order]

    raw_labels = g2.predict(Vf)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(z)] for z in raw_labels], dtype=int)
    counts = np.bincount(labels, minlength=2)
    fractions = counts / len(Vf)

    delta = means[1] - means[0]
    pooled = 0.5 * (covs[0] + covs[1])
    inv_pooled = np.linalg.pinv(pooled, hermitian=True)
    separation_d2 = float(delta.T @ inv_pooled @ delta)
    separation_d = math.sqrt(max(0.0, separation_d2))
    centroid_sep = float(np.linalg.norm(delta))

    result.update(
        {
            "bic_1g": bic1,
            "bic_2g": bic2,
            "delta_bic": float(delta_bic),
            "separation_d": float(separation_d),
            "centroid_separation_kms": centroid_sep,
            "mean_vl_1": float(means[0, 0]),
            "mean_vb_1": float(means[0, 1]),
            "mean_vl_2": float(means[1, 0]),
            "mean_vb_2": float(means[1, 1]),
            "cov_vl_vl_1": float(covs[0, 0, 0]),
            "cov_vl_vb_1": float(covs[0, 0, 1]),
            "cov_vb_vb_1": float(covs[0, 1, 1]),
            "cov_vl_vl_2": float(covs[1, 0, 0]),
            "cov_vl_vb_2": float(covs[1, 0, 1]),
            "cov_vb_vb_2": float(covs[1, 1, 1]),
            "weight_1": float(weights[0]),
            "weight_2": float(weights[1]),
            "count_1": int(counts[0]),
            "count_2": int(counts[1]),
            "fraction_1": float(fractions[0]),
            "fraction_2": float(fractions[1]),
            "finite_labels": labels,
        }
    )

    tests = [
        (delta_bic >= cfg.s3_delta_bic_threshold, "delta_bic"),
        (separation_d >= cfg.s3_separation_d_threshold, "separation_d"),
        (counts[0] >= cfg.s3_min_gmm_component_n, "component_n_1"),
        (counts[1] >= cfg.s3_min_gmm_component_n, "component_n_2"),
        (fractions[0] >= cfg.s3_min_gmm_component_fraction, "component_fraction_1"),
        (fractions[1] >= cfg.s3_min_gmm_component_fraction, "component_fraction_2"),
    ]
    failed = [name for ok, name in tests if not ok]
    result["passes_bimodality"] = len(failed) == 0
    result["bimodality_reason"] = "pass" if not failed else "failed:" + ",".join(failed)
    return result


def run_stage3(
    df: pd.DataFrame,
    coords: np.ndarray,
    vl: np.ndarray,
    vb: np.ndarray,
    s2_nodes: dict[str, dict],
    s2_final_node_ids: list[str],
    fracture: float,
    source_col: str,
    cfg: Config,
):
    kin = np.column_stack((vl, vb))

    nodes: dict[str, dict] = {}
    active: list[str] = []
    for s2id in s2_final_node_ids:
        s2n = s2_nodes[s2id]
        nodes[s2id] = {
            "node_id": s2id,
            "parent_id": "",
            "source_s2_node_id": s2id,
            "root_stage1_group_id": s2n["root_stage1_group_id"],
            "kinematic_generation": 0,
            "created_pass": 0,
            "indices": np.asarray(s2n["indices"], dtype=np.int64),
            "status": "active",
        }
        active.append(s2id)

    split_rows = []
    orphan_records = []
    pass_rows = []

    for pass_no in range(1, cfg.max_s3_passes + 1):
        print(f"\nXMST-S3 pass {pass_no}: testing {len(active)} active cluster(s)...", flush=True)
        accepted_this_pass = 0
        next_active: list[str] = []
        tested_this_pass = 0
        bimodal_this_pass = 0

        for node_id in active:
            node = nodes[node_id]
            idx = node["indices"]
            tested_this_pass += 1

            b = kinematic_bimodality_test(kin[idx], cfg)
            # Do not store masks/labels in the node record.
            node.update({k: v for k, v in b.items() if k not in {"finite_mask", "finite_labels"}})

            if not b["passes_bimodality"]:
                node["status"] = "final_no_significant_kinematic_bimodality"
                continue

            bimodal_this_pass += 1

            # As in S2, do not silently invent assignments for missing values.
            if not b["finite_mask"].all():
                node["status"] = "final_kinematic_bimodal_missing_values_unassigned"
                continue

            labels = np.asarray(b["finite_labels"], dtype=int)
            comp1_idx = idx[labels == 0]
            comp2_idx = idx[labels == 1]

            node["gmm_component_1_n"] = int(len(comp1_idx))
            node["gmm_component_2_n"] = int(len(comp2_idx))

            split_rec = {
                "pass": pass_no,
                "parent_node_id": node_id,
                "source_s2_node_id": node["source_s2_node_id"],
                "root_stage1_group_id": node["root_stage1_group_id"],
                "parent_n": len(idx),
                **{
                    k: b[k]
                    for k in [
                        "n_finite",
                        "coverage",
                        "bic_1g",
                        "bic_2g",
                        "delta_bic",
                        "separation_d",
                        "centroid_separation_kms",
                        "mean_vl_1",
                        "mean_vb_1",
                        "mean_vl_2",
                        "mean_vb_2",
                        "cov_vl_vl_1",
                        "cov_vl_vb_1",
                        "cov_vb_vb_1",
                        "cov_vl_vl_2",
                        "cov_vl_vb_2",
                        "cov_vb_vb_2",
                        "weight_1",
                        "weight_2",
                        "count_1",
                        "count_2",
                        "fraction_1",
                        "fraction_2",
                    ]
                },
                "gmm_component_1_n": len(comp1_idx),
                "gmm_component_2_n": len(comp2_idx),
                "component_1_retained_components": 0,
                "component_2_retained_components": 0,
                "component_1_retained_sizes": "",
                "component_2_retained_sizes": "",
                "orphans_n": 0,
                "accepted": False,
                "rejection_reason": "",
                "child_ids": "",
            }

            if len(comp1_idx) < cfg.min_group_size or len(comp2_idx) < cfg.min_group_size:
                node["status"] = "final_kinematic_bimodal_component_below_min_group"
                split_rec["rejection_reason"] = "gmm_component_below_min_group"
                split_rows.append(split_rec)
                continue

            comps1, mode1 = spatial_components_at_fixed_cut(
                comp1_idx, coords, fracture, cfg.min_group_size
            )
            comps2, mode2 = spatial_components_at_fixed_cut(
                comp2_idx, coords, fracture, cfg.min_group_size
            )

            split_rec["component_1_retained_components"] = len(comps1)
            split_rec["component_2_retained_components"] = len(comps2)
            split_rec["component_1_retained_sizes"] = ";".join(str(len(c)) for c in comps1)
            split_rec["component_2_retained_sizes"] = ";".join(str(len(c)) for c in comps2)
            split_rec["component_1_mst_mode"] = mode1
            split_rec["component_2_mst_mode"] = mode2

            if cfg.s3_spatial_child_policy == "single":
                if len(comps1) != 1 or len(comps2) != 1:
                    node["status"] = "final_kinematic_bimodal_spatial_split_not_binary"
                    split_rec["rejection_reason"] = (
                        "not_exactly_one_retained_spatial_component_per_kinematic_component"
                    )
                    split_rows.append(split_rec)
                    continue
                child_components = [("k1", comps1[0]), ("k2", comps2[0])]

            elif cfg.s3_spatial_child_policy == "all":
                if len(comps1) == 0 or len(comps2) == 0:
                    node["status"] = "final_kinematic_bimodal_one_component_no_spatial_component"
                    split_rec["rejection_reason"] = (
                        "one_kinematic_component_has_no_retained_spatial_component"
                    )
                    split_rows.append(split_rec)
                    continue
                child_components = []
                for j, comp in enumerate(comps1):
                    child_components.append((f"k1s{j+1}", comp))
                for j, comp in enumerate(comps2):
                    child_components.append((f"k2s{j+1}", comp))
            else:
                raise ValueError(f"Unknown s3_spatial_child_policy={cfg.s3_spatial_child_policy}")

            kept = np.concatenate([c for _, c in child_components])
            kept_set = set(map(int, kept))
            orphans = np.array([i for i in idx if int(i) not in kept_set], dtype=np.int64)

            node["status"] = "split"
            node["split_pass"] = pass_no
            node["orphans_n"] = int(len(orphans))
            accepted_this_pass += 1

            child_ids = []
            for suffix, comp in child_components:
                cid = f"{node_id}.{suffix}"
                child_ids.append(cid)
                nodes[cid] = {
                    "node_id": cid,
                    "parent_id": node_id,
                    "source_s2_node_id": node["source_s2_node_id"],
                    "root_stage1_group_id": node["root_stage1_group_id"],
                    "kinematic_generation": int(node["kinematic_generation"]) + 1,
                    "created_pass": pass_no,
                    "indices": np.asarray(comp, dtype=np.int64),
                    "status": "active",
                }
                next_active.append(cid)

            split_rec["child_ids"] = ";".join(child_ids)
            split_rec["orphans_n"] = int(len(orphans))
            split_rec["accepted"] = True
            split_rows.append(split_rec)

            for row_idx in orphans:
                orphan_records.append(
                    {
                        "row_index": int(row_idx),
                        "source_id": canonical_source_id(df.iloc[row_idx][source_col]),
                        "stage": "S3",
                        "stage1_group_id": int(node["root_stage1_group_id"]),
                        "parent_node_id": node_id,
                        "split_pass": pass_no,
                        "reason": "removed_by_fixed_scale_spatial_rerun_after_kinematic_split",
                    }
                )

        pass_rows.append(
            {
                "pass": pass_no,
                "clusters_tested": tested_this_pass,
                "significant_bimodality": bimodal_this_pass,
                "accepted_splits": accepted_this_pass,
                "new_children": len(next_active),
            }
        )

        print(
            f"  significant kinematic bimodality: {bimodal_this_pass}; "
            f"accepted splits: {accepted_this_pass}; "
            f"new child clusters: {len(next_active)}"
        )

        if accepted_this_pass == 0:
            print("  stopping S3: no accepted split in this pass.")
            active = []
            break
        active = next_active
    else:
        raise RuntimeError(
            f"Reached max_s3_passes={cfg.max_s3_passes} while accepted splits were still generated."
        )

    final_node_ids = [nid for nid, n in nodes.items() if n["status"] != "split"]
    final_node_ids.sort(
        key=lambda nid: (
            int(nodes[nid]["root_stage1_group_id"]),
            nodes[nid]["source_s2_node_id"],
            int(nodes[nid]["kinematic_generation"]),
            nid,
        )
    )

    return {
        "nodes": nodes,
        "final_node_ids": final_node_ids,
        "splits": pd.DataFrame(split_rows),
        "passes": pd.DataFrame(pass_rows),
        "orphans": pd.DataFrame(orphan_records),
    }


# ---------------------------------------------------------------------------
# Tables / checkpoints / outputs
# ---------------------------------------------------------------------------


def node_dict_to_frame(nodes: dict[str, dict], stage: str, reddening=None, vl=None, vb=None):
    rows = []
    for nid, n in nodes.items():
        idx = np.asarray(n["indices"], dtype=np.int64)
        row = {k: v for k, v in n.items() if k != "indices"}
        row["n_members"] = len(idx)
        if reddening is not None and len(idx):
            row["median_reddening"] = float(np.nanmedian(reddening[idx]))
            row["mean_reddening"] = float(np.nanmean(reddening[idx]))
        if vl is not None and vb is not None and len(idx):
            row["median_vl_lsr_kms"] = float(np.nanmedian(vl[idx]))
            row["median_vb_lsr_kms"] = float(np.nanmedian(vb[idx]))
            row["mean_vl_lsr_kms"] = float(np.nanmean(vl[idx]))
            row["mean_vb_lsr_kms"] = float(np.nanmean(vb[idx]))
        row["stage"] = stage
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    sort_cols = [c for c in ["root_stage1_group_id", "generation", "kinematic_generation", "node_id"] if c in rows[0]]
    return pd.DataFrame(rows).sort_values(sort_cols)


def build_membership(
    df,
    s1_gid,
    s2,
    s3,
    reddening,
    vl,
    vb,
):
    membership = df.copy()
    membership.insert(0, "row_index", np.arange(len(df), dtype=np.int64))
    membership["stage1_group_id"] = s1_gid

    s2_map = {nid: i + 1 for i, nid in enumerate(s2["final_node_ids"])}
    membership["xmst_s2_final_group_id"] = 0
    membership["xmst_s2_node_id"] = ""
    membership["xmst_s2_member_status"] = np.where(s1_gid > 0, "stage1_grouped", "stage1_field")

    for nid in s2["final_node_ids"]:
        idx = s2["nodes"][nid]["indices"]
        membership.loc[idx, "xmst_s2_final_group_id"] = s2_map[nid]
        membership.loc[idx, "xmst_s2_node_id"] = nid
        membership.loc[idx, "xmst_s2_member_status"] = "xmst_s2_final_cluster"

    if len(s2["orphans"]):
        oi = s2["orphans"]["row_index"].astype(int).to_numpy()
        membership.loc[oi, "xmst_s2_final_group_id"] = 0
        membership.loc[oi, "xmst_s2_node_id"] = ""
        membership.loc[oi, "xmst_s2_member_status"] = "xmst_s2_orphan"

    membership["v_l_lsr_kms"] = vl
    membership["v_b_lsr_kms"] = vb

    if s3 is None:
        membership["xmst_s3_final_group_id"] = membership["xmst_s2_final_group_id"]
        membership["xmst_s3_node_id"] = membership["xmst_s2_node_id"]
        membership["xmst_s3_member_status"] = membership["xmst_s2_member_status"]
        return membership

    s3_map = {nid: i + 1 for i, nid in enumerate(s3["final_node_ids"])}
    membership["xmst_s3_final_group_id"] = 0
    membership["xmst_s3_node_id"] = ""
    membership["xmst_s3_member_status"] = membership["xmst_s2_member_status"]

    for nid in s3["final_node_ids"]:
        idx = s3["nodes"][nid]["indices"]
        membership.loc[idx, "xmst_s3_final_group_id"] = s3_map[nid]
        membership.loc[idx, "xmst_s3_node_id"] = nid
        membership.loc[idx, "xmst_s3_member_status"] = "xmst_s3_final_cluster"

    if len(s3["orphans"]):
        oi = s3["orphans"]["row_index"].astype(int).to_numpy()
        membership.loc[oi, "xmst_s3_final_group_id"] = 0
        membership.loc[oi, "xmst_s3_node_id"] = ""
        membership.loc[oi, "xmst_s3_member_status"] = "xmst_s3_orphan"

    return membership


def final_group_frame(membership: pd.DataFrame, coords, reddening, vl, vb):
    rows = []
    gids = sorted(int(g) for g in membership.loc[membership["xmst_s3_final_group_id"] > 0, "xmst_s3_final_group_id"].unique())
    for gid in gids:
        idx = membership.index[membership["xmst_s3_final_group_id"] == gid].to_numpy(dtype=int)
        node_id = str(membership.loc[idx[0], "xmst_s3_node_id"])
        rows.append(
            {
                "xmst_s3_final_group_id": gid,
                "xmst_s3_node_id": node_id,
                "stage1_group_id": int(membership.loc[idx[0], "stage1_group_id"]),
                "n_members": len(idx),
                "median_reddening": float(np.nanmedian(reddening[idx])),
                "mean_reddening": float(np.nanmean(reddening[idx])),
                "median_vl_lsr_kms": float(np.nanmedian(vl[idx])),
                "median_vb_lsr_kms": float(np.nanmedian(vb[idx])),
                "mean_vl_lsr_kms": float(np.nanmean(vl[idx])),
                "mean_vb_lsr_kms": float(np.nanmean(vb[idx])),
                "xmin_pc": float(np.min(coords[idx, 0])),
                "xmax_pc": float(np.max(coords[idx, 0])),
                "ymin_pc": float(np.min(coords[idx, 1])),
                "ymax_pc": float(np.max(coords[idx, 1])),
                "zmin_pc": float(np.min(coords[idx, 2])),
                "zmax_pc": float(np.max(coords[idx, 2])),
            }
        )
    return pd.DataFrame(rows)


def s2_defaults_match_checkpoint(cfg: Config) -> bool:
    return (
        cfg.min_group_size == 10
        and cfg.s2_delta_bic_threshold == 10.0
        and cfg.s2_ashman_d_threshold == 2.0
        and cfg.s2_min_gmm_component_n == 5
        and cfg.s2_min_gmm_component_fraction == 0.15
        and cfg.s2_min_reddening_coverage == 1.0
        and cfg.s2_gmm_n_init == 20
        and cfg.s2_gmm_random_state == 20260825
        and cfg.s2_spatial_child_policy == "all"
    )


def run_pipeline(base_df: pd.DataFrame, kin_df: Optional[pd.DataFrame], cfg: Config):
    xcol = find_col(base_df, ["x_pc", "x"])
    ycol = find_col(base_df, ["y_pc", "y"])
    zcol = find_col(base_df, ["z_pc", "z"])
    rcol = find_col(base_df, ["reddening_mag", "a_v", "av", "reddening"])
    source_col = find_col(base_df, ["source_id", "gaia_source_id", "id"])

    coords = numeric_array(base_df, [xcol, ycol, zcol])
    reddening = pd.to_numeric(base_df[rcol], errors="coerce").to_numpy(float)
    if not np.isfinite(coords).all():
        bad = np.flatnonzero(~np.isfinite(coords).all(axis=1))
        raise ValueError(f"Non-finite XYZ values in {len(bad)} rows; first rows: {bad[:10].tolist()}")

    print("XMST-S3: rebuilding frozen Stage-1 XMST...", flush=True)
    s1 = stage1_xmst(coords, cfg.min_group_size)
    s1_gid = s1["group_ids"]
    fracture = s1["fracture_scale_pc"]
    n_groups = int(s1_gid.max())
    n_grouped = int(np.sum(s1_gid > 0))

    print(f"  input stars: {len(base_df):,}")
    print(f"  percolation limit: {s1['percolation_limit_pc']:.12f} pc")
    print(f"  Stage-1 fracture scale: {fracture:.12f} pc")
    print(f"  Stage-1 retained groups: {n_groups}")
    print(f"  Stage-1 grouped stars: {n_grouped:,}")

    if cfg.strict_q25_checkpoint and len(base_df) == EXPECTED_N:
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

    s2 = run_stage2(base_df, coords, reddening, s1_gid, fracture, source_col, cfg)

    s2_groups = len(s2["final_node_ids"])
    s2_grouped = int(sum(len(s2["nodes"][nid]["indices"]) for nid in s2["final_node_ids"]))
    s2_orphans = len(s2["orphans"])
    s2_splits = int(s2["passes"]["accepted_splits"].sum()) if len(s2["passes"]) else 0
    s2_passes = len(s2["passes"])

    print("\nXMST-S2 checkpoint from this run:")
    print(f"  final groups: {s2_groups}")
    print(f"  grouped stars: {s2_grouped:,}")
    print(f"  S2 spatial orphans: {s2_orphans}")
    print(f"  accepted S2 splits: {s2_splits}")
    print(f"  S2 passes: {s2_passes}")

    if (
        cfg.strict_s2_checkpoint
        and len(base_df) == EXPECTED_N
        and s2_defaults_match_checkpoint(cfg)
    ):
        problems = []
        if s2_groups != EXPECTED_S2_GROUPS_ALL:
            problems.append(f"S2 groups {s2_groups} != expected {EXPECTED_S2_GROUPS_ALL}")
        if s2_grouped != EXPECTED_S2_GROUPED_STARS_ALL:
            problems.append(
                f"S2 grouped stars {s2_grouped} != expected {EXPECTED_S2_GROUPED_STARS_ALL}"
            )
        if s2_orphans != EXPECTED_S2_ORPHANS_ALL:
            problems.append(f"S2 orphans {s2_orphans} != expected {EXPECTED_S2_ORPHANS_ALL}")
        if s2_splits != EXPECTED_S2_ACCEPTED_SPLITS_ALL:
            problems.append(
                f"S2 accepted splits {s2_splits} != expected {EXPECTED_S2_ACCEPTED_SPLITS_ALL}"
            )
        if s2_passes != EXPECTED_S2_PASSES_ALL:
            problems.append(f"S2 passes {s2_passes} != expected {EXPECTED_S2_PASSES_ALL}")
        if problems:
            raise RuntimeError("XMST-S2 all-descendants checkpoint failed: " + "; ".join(problems))
        print("  latest all-descendants XMST-S2 checkpoint reproduced successfully.")

    if cfg.skip_s3:
        vl = np.full(len(base_df), np.nan)
        vb = np.full(len(base_df), np.nan)
        kin_info = {"mode": "S3 skipped", "matched_source": np.zeros(len(base_df), dtype=bool)}
        s3 = None
    else:
        if kin_df is None:
            # Permit kinematics embedded directly in the base table.
            kin_df = base_df
        kin_info = prepare_kinematics(base_df, kin_df, cfg)
        vl = kin_info["vl_lsr_kms"]
        vb = kin_info["vb_lsr_kms"]

        finite_kin = np.isfinite(vl) & np.isfinite(vb)
        print("\nKinematics:")
        print(f"  mode: {kin_info['mode']}")
        print(f"  source-ID matches: {int(np.sum(kin_info['matched_source'])):,} / {len(base_df):,}")
        print(f"  finite V_l,V_b pairs: {int(np.sum(finite_kin)):,} / {len(base_df):,}")

        s3 = run_stage3(
            base_df,
            coords,
            vl,
            vb,
            s2["nodes"],
            s2["final_node_ids"],
            fracture,
            source_col,
            cfg,
        )

    membership = build_membership(base_df, s1_gid, s2, s3, reddening, vl, vb)
    final_groups = final_group_frame(membership, coords, reddening, vl, vb)

    summary = {
        "input_stars": len(base_df),
        "stage1_percolation_limit_pc": s1["percolation_limit_pc"],
        "stage1_fracture_scale_pc": fracture,
        "stage1_groups": n_groups,
        "stage1_grouped_stars": n_grouped,
        "s2_passes": s2_passes,
        "s2_significant_bimodality_flags": int(s2["passes"]["significant_bimodality"].sum()) if len(s2["passes"]) else 0,
        "s2_accepted_splits": s2_splits,
        "s2_final_groups": s2_groups,
        "s2_grouped_stars": s2_grouped,
        "s2_orphans": s2_orphans,
        "kinematics_mode": kin_info["mode"],
        "finite_kinematic_pairs": int(np.sum(np.isfinite(vl) & np.isfinite(vb))),
    }

    if s3 is not None:
        summary.update(
            {
                "s3_passes": len(s3["passes"]),
                "s3_significant_bimodality_flags": int(s3["passes"]["significant_bimodality"].sum()) if len(s3["passes"]) else 0,
                "s3_accepted_splits": int(s3["passes"]["accepted_splits"].sum()) if len(s3["passes"]) else 0,
                "s3_final_groups": len(s3["final_node_ids"]),
                "s3_grouped_stars": int((membership["xmst_s3_final_group_id"] > 0).sum()),
                "s3_orphans": len(s3["orphans"]),
            }
        )
    else:
        summary.update(
            {
                "s3_passes": 0,
                "s3_significant_bimodality_flags": 0,
                "s3_accepted_splits": 0,
                "s3_final_groups": s2_groups,
                "s3_grouped_stars": s2_grouped,
                "s3_orphans": 0,
            }
        )

    return {
        "stage1": s1,
        "stage2": s2,
        "stage3": s3,
        "membership": membership,
        "final_groups": final_groups,
        "summary": summary,
        "reddening": reddening,
        "vl": vl,
        "vb": vb,
        "kinematics_mode": kin_info["mode"],
    }


def write_outputs(result: dict, cfg: Config):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s2 = result["stage2"]
    s3 = result["stage3"]

    result["membership"].to_csv(out / "xmst_s3_final_membership.csv", index=False)
    result["final_groups"].to_csv(out / "xmst_s3_final_groups.csv", index=False)

    node_dict_to_frame(
        s2["nodes"], "S2", reddening=result["reddening"]
    ).to_csv(out / "xmst_s2_nodes.csv", index=False)
    s2["splits"].to_csv(out / "xmst_s2_splits.csv", index=False)
    s2["passes"].to_csv(out / "xmst_s2_passes.csv", index=False)
    s2["orphans"].to_csv(out / "xmst_s2_orphans.csv", index=False)

    if s3 is not None:
        node_dict_to_frame(
            s3["nodes"],
            "S3",
            reddening=result["reddening"],
            vl=result["vl"],
            vb=result["vb"],
        ).to_csv(out / "xmst_s3_kinematic_nodes.csv", index=False)
        s3["splits"].to_csv(out / "xmst_s3_kinematic_splits.csv", index=False)
        s3["passes"].to_csv(out / "xmst_s3_kinematic_passes.csv", index=False)
        s3["orphans"].to_csv(out / "xmst_s3_kinematic_orphans.csv", index=False)

    kin_table = pd.DataFrame(
        {
            "row_index": np.arange(len(result["membership"]), dtype=np.int64),
            "source_id": result["membership"][find_col(result["membership"], ["source_id", "gaia_source_id", "id"])].map(canonical_source_id),
            "v_l_lsr_kms": result["vl"],
            "v_b_lsr_kms": result["vb"],
        }
    )
    kin_table.to_csv(out / "xmst_s3_kinematics_aligned.csv", index=False)

    config_payload = asdict(cfg)
    config_payload.update(
        {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "sklearn_version": sklearn_version,
            "kinematics_mode": result["kinematics_mode"],
        }
    )
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    s = result["summary"]
    lines = [
        "XMST-S3 SUMMARY",
        "===============",
        "",
        f"Input stars: {s['input_stars']:,}",
        f"Stage-1 percolation limit: {s['stage1_percolation_limit_pc']:.12f} pc",
        f"Stage-1 fracture scale: {s['stage1_fracture_scale_pc']:.12f} pc",
        f"Stage-1 groups: {s['stage1_groups']}",
        f"Stage-1 grouped stars: {s['stage1_grouped_stars']:,}",
        "",
        "XMST-S2 REDDENING PASS",
        f"  passes: {s['s2_passes']}",
        f"  significant flags: {s['s2_significant_bimodality_flags']}",
        f"  accepted splits: {s['s2_accepted_splits']}",
        f"  final groups: {s['s2_final_groups']}",
        f"  grouped stars: {s['s2_grouped_stars']:,}",
        f"  spatial orphans: {s['s2_orphans']}",
        f"  child policy: {cfg.s2_spatial_child_policy}",
        f"  gate: delta BIC >= {cfg.s2_delta_bic_threshold}; Ashman D >= {cfg.s2_ashman_d_threshold}; "
        f"component n >= {cfg.s2_min_gmm_component_n}; fraction >= {cfg.s2_min_gmm_component_fraction:.3f}",
        "",
        "XMST-S3 KINEMATIC PASS",
        f"  kinematics mode: {s['kinematics_mode']}",
        f"  finite V_l,V_b pairs: {s['finite_kinematic_pairs']:,}",
        f"  passes: {s['s3_passes']}",
        f"  significant flags: {s['s3_significant_bimodality_flags']}",
        f"  accepted splits: {s['s3_accepted_splits']}",
        f"  final groups: {s['s3_final_groups']}",
        f"  grouped stars: {s['s3_grouped_stars']:,}",
        f"  spatial orphans: {s['s3_orphans']}",
        f"  child policy: {cfg.s3_spatial_child_policy}",
        f"  gate: delta BIC >= {cfg.s3_delta_bic_threshold}; D_2D >= {cfg.s3_separation_d_threshold}; "
        f"component n >= {cfg.s3_min_gmm_component_n}; fraction >= {cfg.s3_min_gmm_component_fraction:.3f}",
        "",
        "All spatial subtree reconstructions in S2 and S3 use the fixed Stage-1 fracture scale.",
        "No position-velocity scaling factor is used.",
    ]

    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nOutputs written to: {out.resolve()}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "XMST-S3: frozen spatial XMST -> recursive reddening S2 -> "
            "recursive transverse-kinematic S3"
        )
    )
    p.add_argument(
        "input_csv",
        nargs="?",
        default="quintana_2025_mst_input.csv",
        help="Frozen spatial/reddening CSV containing source_id, x_pc, y_pc, z_pc, reddening_mag, distance_pc",
    )
    p.add_argument(
        "--kinematics-csv",
        default=None,
        help=(
            "Source-ID keyed kinematics CSV. May contain LSR-corrected V_l,V_b, "
            "or raw Gaia RA/Dec/pmRA/pmDec. If omitted, the base input is searched."
        ),
    )
    p.add_argument("--output-dir", default="xmst_s3_output")
    p.add_argument("--min-group-size", type=int, default=10)

    # S2
    p.add_argument("--s2-delta-bic", type=float, default=10.0)
    p.add_argument("--s2-ashman-d", type=float, default=2.0)
    p.add_argument("--s2-min-gmm-component-n", type=int, default=5)
    p.add_argument("--s2-min-gmm-component-fraction", type=float, default=0.15)
    p.add_argument("--s2-min-reddening-coverage", type=float, default=1.0)
    p.add_argument("--s2-gmm-n-init", type=int, default=20)
    p.add_argument("--s2-gmm-random-state", type=int, default=20260825)
    p.add_argument(
        "--s2-spatial-child-policy",
        choices=["single", "all"],
        default="all",
        help="Default 'all' reproduces the latest 108-group all-descendants XMST-S2 run.",
    )
    p.add_argument("--max-s2-passes", type=int, default=20)

    # S3
    p.add_argument("--s3-delta-bic", type=float, default=10.0)
    p.add_argument("--s3-separation-d", type=float, default=2.0)
    p.add_argument("--s3-min-gmm-component-n", type=int, default=5)
    p.add_argument("--s3-min-gmm-component-fraction", type=float, default=0.15)
    p.add_argument("--s3-min-kinematic-coverage", type=float, default=1.0)
    p.add_argument("--s3-gmm-n-init", type=int, default=20)
    p.add_argument("--s3-gmm-random-state", type=int, default=20260825)
    p.add_argument(
        "--s3-spatial-child-policy",
        choices=["single", "all"],
        default="all",
        help="Retain all N>=10 fixed-scale spatial descendants from each kinematic component.",
    )
    p.add_argument("--max-s3-passes", type=int, default=20)

    # Direct velocity-column override / frame
    p.add_argument("--vl-col", default=None, help="Explicit transverse Galactic-longitude velocity column")
    p.add_argument("--vb-col", default=None, help="Explicit transverse Galactic-latitude velocity column")
    p.add_argument(
        "--velocity-frame",
        choices=["lsr", "heliocentric"],
        default="lsr",
        help="Frame of direct V_l,V_b columns; raw proper motions are always converted and LSR-corrected.",
    )
    p.add_argument("--solar-u", type=float, default=DEFAULT_SOLAR_U)
    p.add_argument("--solar-v", type=float, default=DEFAULT_SOLAR_V)
    p.add_argument("--solar-w", type=float, default=DEFAULT_SOLAR_W)

    p.add_argument("--no-strict-q25-checkpoint", action="store_true")
    p.add_argument("--no-strict-s2-checkpoint", action="store_true")
    p.add_argument(
        "--skip-s3",
        action="store_true",
        help="Developer/reproducibility mode: run Stage 1 and S2 only.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        input_csv=args.input_csv,
        kinematics_csv=args.kinematics_csv,
        output_dir=args.output_dir,
        min_group_size=args.min_group_size,
        s2_delta_bic_threshold=args.s2_delta_bic,
        s2_ashman_d_threshold=args.s2_ashman_d,
        s2_min_gmm_component_n=args.s2_min_gmm_component_n,
        s2_min_gmm_component_fraction=args.s2_min_gmm_component_fraction,
        s2_min_reddening_coverage=args.s2_min_reddening_coverage,
        s2_gmm_n_init=args.s2_gmm_n_init,
        s2_gmm_random_state=args.s2_gmm_random_state,
        s2_spatial_child_policy=args.s2_spatial_child_policy,
        max_s2_passes=args.max_s2_passes,
        s3_delta_bic_threshold=args.s3_delta_bic,
        s3_separation_d_threshold=args.s3_separation_d,
        s3_min_gmm_component_n=args.s3_min_gmm_component_n,
        s3_min_gmm_component_fraction=args.s3_min_gmm_component_fraction,
        s3_min_kinematic_coverage=args.s3_min_kinematic_coverage,
        s3_gmm_n_init=args.s3_gmm_n_init,
        s3_gmm_random_state=args.s3_gmm_random_state,
        s3_spatial_child_policy=args.s3_spatial_child_policy,
        max_s3_passes=args.max_s3_passes,
        vl_col=args.vl_col,
        vb_col=args.vb_col,
        velocity_frame=args.velocity_frame,
        solar_u_kms=args.solar_u,
        solar_v_kms=args.solar_v,
        solar_w_kms=args.solar_w,
        strict_q25_checkpoint=not args.no_strict_q25_checkpoint,
        strict_s2_checkpoint=not args.no_strict_s2_checkpoint,
        skip_s3=args.skip_s3,
    )

    base_df = read_csv_preserve_ids(cfg.input_csv)
    kin_df = read_csv_preserve_ids(cfg.kinematics_csv) if cfg.kinematics_csv else None
    result = run_pipeline(base_df, kin_df, cfg)
    write_outputs(result, cfg)


if __name__ == "__main__":
    main()

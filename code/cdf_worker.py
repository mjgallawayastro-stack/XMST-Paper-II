#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from scipy.stats import mannwhitneyu
from sklearn.mixture import GaussianMixture

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'output'
Q25 = BASE / 'input' / 'quintana_2025_ob_stars_full.csv'
TEMPLATE_SUMMARY = BASE / 'input' / 'selected_template_groups.csv'
MEMBERSHIP = BASE / 'input' / 'q25_group_membership.csv'

CFG = {
    'n_realisations': 1000,
    'seed': 20260725,
    'min_group_size': 10,
    'min_branch_size': 20,
    'min_reddening_per_branch': 15,
    'min_reddening_fraction': 0.60,
    'min_effect_size': 1.5,
    'q_threshold': 0.01,
    'branch_bootstraps': 500,
    'min_bootstrap_stability': 0.95,
    'coherence_fraction': 0.80,
    'min_topology_persistence': 0.10,
    'mixture_bootstraps': 250,
    'cdf_bootstraps': 200,
    'min_cdf_segment': 200,
    'max_recursive_depth': 3,
    'max_bootstrap_candidates_per_group': 25,
    'overlap_distance_min_pc': 57.0,
    'overlap_distance_max_pc': 83.0,
    'centre_min_separation_pc': 120.0,
    'centre_distance_min_pc': 250.0,
    'centre_distance_max_pc': 900.0,
}


def robust_scatter(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan')
    med = np.median(x)
    sig = 1.4826 * np.median(np.abs(x - med))
    if not np.isfinite(sig) or sig <= 1e-12:
        q25, q75 = np.percentile(x, [25, 75])
        sig = (q75 - q25) / 1.349
    if not np.isfinite(sig) or sig <= 1e-12:
        sig = np.std(x, ddof=1)
    return float(sig)


def effect_size(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float, float]:
    ma, mb = float(np.median(a)), float(np.median(b))
    sa, sb = robust_scatter(a), robust_scatter(b)
    pooled = math.sqrt((sa * sa + sb * sb) / 2.0) if np.isfinite(sa) and np.isfinite(sb) else float('nan')
    d = abs(ma - mb) / pooled if np.isfinite(pooled) and pooled > 0 else float('inf')
    return float(d), ma, mb, sa, sb


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, float)
    out[order] = q
    return out


def exact_mst(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    tri = Delaunay(xyz, qhull_options='Qbb Qc Qz Q12')
    s = tri.simplices
    pairs = np.vstack([
        s[:, [0, 1]], s[:, [0, 2]], s[:, [0, 3]],
        s[:, [1, 2]], s[:, [1, 3]], s[:, [2, 3]],
    ])
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    n = len(xyz)
    pairs = pairs[(pairs[:, 0] < n) & (pairs[:, 1] < n)]
    w = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    g = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    g = g + g.T
    mst = minimum_spanning_tree(g).tocoo()
    u = mst.row.astype(np.int32)
    v = mst.col.astype(np.int32)
    ew = mst.data.astype(float)
    if len(ew) != n - 1:
        raise RuntimeError(f'MST incomplete: {len(ew)} != {n-1}')
    order = np.argsort(ew, kind='stable')
    return u[order], v[order], ew[order], len(pairs)


def regression_segment_stats(
    n: np.ndarray, sx: np.ndarray, sy: np.ndarray, sxx: np.ndarray,
    sxy: np.ndarray, syy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denominator = n * sxx - sx * sx
    slope = np.divide(
        n * sxy - sx * sy, denominator, out=np.zeros_like(n, dtype=float),
        where=np.abs(denominator) > 1e-20,
    )
    intercept = (sy - slope * sx) / n
    sse = (
        syy + slope * slope * sxx + n * intercept * intercept
        + 2.0 * slope * intercept * sx - 2.0 * slope * sxy
        - 2.0 * intercept * sy
    )
    return slope, intercept, np.maximum(sse, 0.0)


def fit_two_line_cdf(edge_lengths: np.ndarray, min_segment: int = 200) -> dict:
    """Fit the original two-line breakpoint to the empirical MST edge CDF.

    This is the same definition used in the earlier Paper-I CDF pipeline.
    The full global MST edge-length distribution is used.
    """
    x = np.sort(np.asarray(edge_lengths, dtype=float))
    n_edges = len(x)
    if n_edges < 2 * min_segment:
        min_segment = max(2, n_edges // 4)
    if n_edges < 4:
        y = np.arange(1, n_edges + 1, dtype=float) / max(n_edges, 1)
        return {
            'threshold': float(np.median(x)), 'threshold_method': 'median_fallback',
            'split_index': max(1, n_edges // 2), 'slope_1': float('nan'),
            'intercept_1': float('nan'), 'slope_2': float('nan'),
            'intercept_2': float('nan'), 'two_line_sse': float('nan'),
            'one_line_sse': float('nan'), 'fit_improvement_fraction': float('nan'),
            'minimum_segment': min_segment, 'split_edge_low': float('nan'),
            'split_edge_high': float('nan'),
        }
    y = np.arange(1, n_edges + 1, dtype=float) / n_edges
    cx = np.concatenate([[0.0], np.cumsum(x)])
    cy = np.concatenate([[0.0], np.cumsum(y)])
    cxx = np.concatenate([[0.0], np.cumsum(x * x)])
    cxy = np.concatenate([[0.0], np.cumsum(x * y)])
    cyy = np.concatenate([[0.0], np.cumsum(y * y)])
    split_positions = np.arange(min_segment, n_edges - min_segment + 1, dtype=int)
    n1 = split_positions.astype(float)
    sx1, sy1 = cx[split_positions], cy[split_positions]
    sxx1, sxy1, syy1 = cxx[split_positions], cxy[split_positions], cyy[split_positions]
    m1, b1, sse1 = regression_segment_stats(n1, sx1, sy1, sxx1, sxy1, syy1)
    n2 = (n_edges - split_positions).astype(float)
    sx2, sy2 = cx[-1] - sx1, cy[-1] - sy1
    sxx2, sxy2, syy2 = cxx[-1] - sxx1, cxy[-1] - sxy1, cyy[-1] - syy1
    m2, b2, sse2 = regression_segment_stats(n2, sx2, sy2, sxx2, sxy2, syy2)
    total_sse = sse1 + sse2
    bi = int(np.argmin(total_sse))
    split_index = int(split_positions[bi])
    slope_1, intercept_1 = float(m1[bi]), float(b1[bi])
    slope_2, intercept_2 = float(m2[bi]), float(b2[bi])
    denominator = slope_1 - slope_2
    intersection = ((intercept_2 - intercept_1) / denominator
                    if abs(denominator) > 1e-15 else float('nan'))
    local_low = float(x[max(0, split_index - 1)])
    local_high = float(x[min(n_edges - 1, split_index)])
    if not np.isfinite(intersection) or intersection < x.min() or intersection > x.max():
        threshold = (local_low + local_high) / 2.0
        threshold_method = 'midpoint_at_best_split'
    else:
        threshold = float(intersection)
        threshold_method = 'line_intersection'
    full_n = np.array([float(n_edges)])
    _, _, one_sse = regression_segment_stats(
        full_n, np.array([cx[-1]]), np.array([cy[-1]]),
        np.array([cxx[-1]]), np.array([cxy[-1]]), np.array([cyy[-1]])
    )
    two_sse = float(total_sse[bi])
    one_sse_value = float(one_sse[0])
    improvement = ((one_sse_value - two_sse) / one_sse_value
                   if one_sse_value > 0 else float('nan'))
    return {
        'threshold': threshold, 'threshold_method': threshold_method,
        'split_index': split_index, 'slope_1': slope_1, 'intercept_1': intercept_1,
        'slope_2': slope_2, 'intercept_2': intercept_2,
        'two_line_sse': two_sse, 'one_line_sse': one_sse_value,
        'fit_improvement_fraction': float(improvement), 'minimum_segment': min_segment,
        'split_edge_low': local_low, 'split_edge_high': local_high,
    }


def cdf_edge_bootstrap(values: np.ndarray, rng: np.random.Generator, nboot: int, min_segment: int) -> tuple[float, float, float, float, float]:
    vals = np.asarray(values, float)
    out = np.empty(nboot, float)
    for b in range(nboot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        out[b] = fit_two_line_cdf(sample, min_segment)['threshold']
    p16, med, p84 = np.percentile(out, [16, 50, 84])
    return float(np.mean(out)), float(np.std(out, ddof=1)), float(p16), float(med), float(p84)


def percolation_cut(u: np.ndarray, v: np.ndarray, w: np.ndarray, n: int) -> tuple[float, float, int, int, int]:
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    largest = 1
    best_jump = -1
    best_i = 0
    best_before = 1
    best_after = 1
    for k in range(len(w)):
        a, b = find(int(u[k])), find(int(v[k]))
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        old = largest
        parent[b] = a
        size[a] += size[b]
        largest = max(largest, int(size[a]))
        jump = largest - old
        if jump > best_jump:
            best_jump = jump
            best_i = k
            best_before = old
            best_after = largest
    crit = float(w[best_i])
    prev = float(w[max(0, best_i - 1)])
    return (crit + prev) / 2.0, crit, int(best_jump), int(best_before), int(best_after)


def jenks_break(values: np.ndarray) -> tuple[float, float, int]:
    x = np.sort(np.asarray(values, float))
    n = len(x)
    if n < 4:
        return float(np.median(x)), float('nan'), max(1, n // 2)
    cs = np.cumsum(x)
    cs2 = np.cumsum(x * x)
    idx = np.arange(1, n)
    ln = idx
    rn = n - idx
    lsse = cs2[idx - 1] - cs[idx - 1] ** 2 / ln
    rs = cs[-1] - cs[idx - 1]
    rs2 = cs2[-1] - cs2[idx - 1]
    rsse = rs2 - rs ** 2 / rn
    valid = (ln >= 2) & (rn >= 2)
    total = np.where(valid, lsse + rsse, np.inf)
    j = int(np.argmin(total)) + 1
    threshold = float((x[j - 1] + x[j]) / 2.0)
    grand_sse = float(np.sum((x - np.mean(x)) ** 2))
    gvf = 1.0 - float(total[j - 1]) / grand_sse if grand_sse > 0 else float('nan')
    return threshold, gvf, j


def jenks_edge_bootstrap(values: np.ndarray, rng: np.random.Generator, nboot: int) -> tuple[float, float, float, float, float]:
    vals = np.asarray(values, float)
    out = np.empty(nboot, float)
    for b in range(nboot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        out[b] = jenks_break(sample)[0]
    p16, med, p84 = np.percentile(out, [16, 50, 84])
    return float(np.mean(out)), float(np.std(out, ddof=1)), float(p16), float(med), float(p84)


def components_from_cut(n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray, threshold: float, min_size: int) -> tuple[list[np.ndarray], np.ndarray]:
    keep = w <= threshold
    g = coo_matrix((np.ones(int(keep.sum())), (u[keep], v[keep])), shape=(n, n))
    g = g + g.T
    _, labels = connected_components(g.tocsr(), directed=False)
    counts = np.bincount(labels)
    groups = [np.flatnonzero(labels == gid) for gid, c in enumerate(counts) if c >= min_size]
    return groups, labels


def branch_coherence_threshold(nodes: np.ndarray, edge_u: np.ndarray, edge_v: np.ndarray, edge_w: np.ndarray, fraction: float) -> float:
    n = len(nodes)
    target = int(math.ceil(fraction * n))
    if target <= 1:
        return 0.0
    node_set = set(map(int, nodes))
    mask = np.array([(int(a) in node_set and int(b) in node_set) for a, b in zip(edge_u, edge_v)], dtype=bool)
    eu, ev, ew = edge_u[mask], edge_v[mask], edge_w[mask]
    if len(ew) == 0:
        return float('inf')
    local = {int(node): i for i, node in enumerate(nodes)}
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    order = np.argsort(ew)
    largest = 1
    for idx in order:
        a, b = find(local[int(eu[idx])]), find(local[int(ev[idx])])
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            largest = max(largest, int(size[a]))
            if largest >= target:
                return float(ew[idx])
    return float(np.max(ew))


def bootstrap_branch(a: np.ndarray, b: np.ndarray, observed_direction: float, rng: np.random.Generator, nboot: int, min_d: float) -> tuple[float, float, float, float]:
    na, nb = len(a), len(b)
    ds = np.empty(nboot, float)
    stable = np.zeros(nboot, bool)
    for i in range(nboot):
        aa = rng.choice(a, size=na, replace=True)
        bb = rng.choice(b, size=nb, replace=True)
        d, ma, mb, _, _ = effect_size(aa, bb)
        ds[i] = d
        direction = np.sign(ma - mb)
        stable[i] = d >= min_d and direction == observed_direction
    return float(np.mean(stable)), float(np.median(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def _normal_loglik_1d(x: np.ndarray) -> tuple[float, float, float]:
    mu = float(np.mean(x))
    var = max(float(np.mean((x - mu) ** 2)), 1e-8)
    ll = float(-0.5 * len(x) * (math.log(2.0 * math.pi * var) + 1.0))
    return ll, mu, math.sqrt(var)


def _gmm2_loglik_1d(x: np.ndarray, init_shift: float = 0.7, max_iter: int = 100) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    mean = float(np.mean(x))
    sd = max(float(np.std(x)), 1e-4)
    mu = np.array([mean - init_shift * sd, mean + init_shift * sd], float)
    sig = np.array([sd, sd], float)
    wt = np.array([0.5, 0.5], float)
    prev = -np.inf
    log2pi = math.log(2.0 * math.pi)
    for _ in range(max_iter):
        logp0 = math.log(max(wt[0], 1e-8)) - np.log(sig[0]) - 0.5 * ((x - mu[0]) / sig[0]) ** 2 - 0.5 * log2pi
        logp1 = math.log(max(wt[1], 1e-8)) - np.log(sig[1]) - 0.5 * ((x - mu[1]) / sig[1]) ** 2 - 0.5 * log2pi
        mx = np.maximum(logp0, logp1)
        den = mx + np.log(np.exp(logp0 - mx) + np.exp(logp1 - mx))
        ll = float(np.sum(den))
        r0 = np.exp(logp0 - den)
        r1 = 1.0 - r0
        n0 = max(float(np.sum(r0)), 1e-6)
        n1 = max(float(np.sum(r1)), 1e-6)
        wt = np.array([n0 / len(x), n1 / len(x)])
        mu = np.array([float(np.sum(r0 * x) / n0), float(np.sum(r1 * x) / n1)])
        var0 = max(float(np.sum(r0 * (x - mu[0]) ** 2) / n0), 1e-8)
        var1 = max(float(np.sum(r1 * (x - mu[1]) ** 2) / n1), 1e-8)
        sig = np.sqrt([var0, var1])
        if abs(ll - prev) < 1e-7 * (1.0 + abs(ll)):
            break
        prev = ll
    order = np.argsort(mu)
    return ll, mu[order], sig[order], wt[order]


def _best_gmm2_1d(x: np.ndarray, quick: bool = False):
    shifts = (0.55, 0.9) if quick else (0.35, 0.55, 0.75, 1.0, 1.3)
    best = None
    for shift in shifts:
        fit = _gmm2_loglik_1d(x, init_shift=shift, max_iter=60 if quick else 120)
        if best is None or fit[0] > best[0]:
            best = fit
    return best


def mixture_diagnostic(values: np.ndarray, rng: np.random.Generator, nboot: int) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or np.std(x) <= 1e-9:
        return {'mixture_lr': np.nan, 'mixture_bootstrap_p': np.nan, 'ashman_d': np.nan, 'mixture_weight_min': np.nan}
    ll1, mu1, sigma1 = _normal_loglik_1d(x)
    ll2, means, sigmas, weights = _best_gmm2_1d(x, quick=False)
    lr = max(0.0, 2.0 * (ll2 - ll1))
    exceed = 0
    for _ in range(nboot):
        sim = rng.normal(mu1, sigma1, len(x))
        nll1, _, _ = _normal_loglik_1d(sim)
        nll2, _, _, _ = _best_gmm2_1d(sim, quick=True)
        if 2.0 * (nll2 - nll1) >= lr:
            exceed += 1
    p = (1.0 + exceed) / (nboot + 1.0)
    ashman = abs(means[1] - means[0]) / math.sqrt((sigmas[0] ** 2 + sigmas[1] ** 2) / 2.0)
    return {
        'mixture_lr': float(lr),
        'mixture_bootstrap_p': float(p),
        'ashman_d': float(ashman),
        'mixture_weight_min': float(np.min(weights)),
    }


def build_tree_euler(nodes: np.ndarray, eu: np.ndarray, ev: np.ndarray, ew: np.ndarray):
    node_set = set(map(int, nodes))
    mask = np.array([(int(a) in node_set and int(b) in node_set) for a, b in zip(eu, ev)], dtype=bool)
    a_all, b_all, w_all = eu[mask], ev[mask], ew[mask]
    local = {int(node): i for i, node in enumerate(nodes)}
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for a, b, w in zip(a_all, b_all, w_all):
        ia, ib = local[int(a)], local[int(b)]
        adj[ia].append((ib, float(w)))
        adj[ib].append((ia, float(w)))
    parent = np.full(n, -1, dtype=np.int32)
    parent_w = np.zeros(n, float)
    tin = np.full(n, -1, dtype=np.int32)
    tout = np.full(n, -1, dtype=np.int32)
    order_nodes: list[int] = []
    stack = [(0, -1, 0, 0.0)]
    while stack:
        node, par, state, pw = stack.pop()
        if state == 0:
            parent[node] = par
            parent_w[node] = pw
            tin[node] = len(order_nodes)
            order_nodes.append(node)
            stack.append((node, par, 1, pw))
            for nei, ww in reversed(adj[node]):
                if nei != par:
                    stack.append((nei, node, 0, ww))
        else:
            tout[node] = len(order_nodes)
    if len(order_nodes) != n:
        raise RuntimeError(f'Induced group is disconnected: {len(order_nodes)} of {n}')
    return np.array(order_nodes, dtype=np.int32), parent, parent_w, tin, tout, a_all, b_all, w_all


def analyse_and_split_group(
    nodes: np.ndarray,
    global_u: np.ndarray,
    global_v: np.ndarray,
    global_w: np.ndarray,
    av: np.ndarray,
    truth: np.ndarray,
    global_threshold: float,
    rng: np.random.Generator,
    run_id: int,
    group_id: str,
    depth: int,
    split_rows: list[dict],
) -> list[np.ndarray]:
    cfg = CFG
    if depth >= cfg['max_recursive_depth'] or len(nodes) < 2 * cfg['min_branch_size']:
        return [nodes]
    try:
        dfs_order, parent, parent_w, tin, tout, eu, ev, ew = build_tree_euler(nodes, global_u, global_v, global_w)
    except RuntimeError:
        return [nodes]
    n = len(nodes)
    all_local = np.arange(n, dtype=np.int32)
    candidates = []
    for child in range(1, n):
        start, stop = int(tin[child]), int(tout[child])
        a_local = dfs_order[start:stop]
        na = len(a_local)
        nb = n - na
        if na < cfg['min_branch_size'] or nb < cfg['min_branch_size']:
            continue
        flag = np.zeros(n, dtype=bool)
        flag[a_local] = True
        b_local = all_local[~flag]
        a_nodes = nodes[a_local]
        b_nodes = nodes[b_local]
        va = av[a_nodes]
        vb = av[b_nodes]
        va = va[np.isfinite(va)]
        vb = vb[np.isfinite(vb)]
        if len(va) < cfg['min_reddening_per_branch'] or len(vb) < cfg['min_reddening_per_branch']:
            continue
        if len(va) / na < cfg['min_reddening_fraction'] or len(vb) / nb < cfg['min_reddening_fraction']:
            continue
        d, ma, mb, sa, sb = effect_size(va, vb)
        try:
            p = float(mannwhitneyu(va, vb, alternative='two-sided', method='asymptotic').pvalue)
        except Exception:
            p = 1.0
        candidates.append({
            'child': child,
            'a_nodes': a_nodes,
            'b_nodes': b_nodes,
            'n_a': na,
            'n_b': nb,
            'n_red_a': len(va),
            'n_red_b': len(vb),
            'median_a': ma,
            'median_b': mb,
            'scatter_a': sa,
            'scatter_b': sb,
            'effect_size': d,
            'p': p,
            'edge_length_pc': float(parent_w[child]),
            'va': va,
            'vb': vb,
        })
    if not candidates:
        return [nodes]
    qvals = bh_qvalues(np.array([c['p'] for c in candidates]))
    for c, q in zip(candidates, qvals):
        c['q'] = float(q)
    screened = [c for c in candidates if c['effect_size'] >= cfg['min_effect_size'] and c['q'] <= cfg['q_threshold']]
    if not screened:
        return [nodes]
    screened.sort(key=lambda c: c['effect_size'] * min(c['n_a'], c['n_b']) / max(c['n_a'], c['n_b']), reverse=True)
    screened = screened[:cfg['max_bootstrap_candidates_per_group']]
    accepted = []
    for c in screened:
        direction = float(np.sign(c['median_a'] - c['median_b']))
        stability, bmed, blo, bhi = bootstrap_branch(c['va'], c['vb'], direction, rng, cfg['branch_bootstraps'], cfg['min_effect_size'])
        t_a = branch_coherence_threshold(c['a_nodes'], eu, ev, ew, cfg['coherence_fraction'])
        t_b = branch_coherence_threshold(c['b_nodes'], eu, ev, ew, cfg['coherence_fraction'])
        coherence_threshold = max(t_a, t_b)
        persistence = max(0.0, (global_threshold - coherence_threshold) / global_threshold) if np.isfinite(coherence_threshold) else 0.0
        c.update({
            'bootstrap_stability': stability,
            'bootstrap_effect_median': bmed,
            'bootstrap_effect_p2_5': blo,
            'bootstrap_effect_p97_5': bhi,
            'coherence_threshold_pc': coherence_threshold,
            'topology_persistence': persistence,
        })
        if stability >= cfg['min_bootstrap_stability'] and persistence >= cfg['min_topology_persistence']:
            balance = min(c['n_a'], c['n_b']) / max(c['n_a'], c['n_b'])
            c['selection_score'] = c['effect_size'] * balance * stability * max(persistence, 1e-9)
            accepted.append(c)
    if not accepted:
        return [nodes]
    best = max(accepted, key=lambda c: c['selection_score'])
    mix = mixture_diagnostic(av[nodes], rng, cfg['mixture_bootstraps'])
    counts_a = Counter(truth[best['a_nodes']])
    counts_b = Counter(truth[best['b_nodes']])
    dom_a, cnt_a = counts_a.most_common(1)[0]
    dom_b, cnt_b = counts_b.most_common(1)[0]
    correct = dom_a != 'background' and dom_b != 'background' and dom_a != dom_b
    noninjected = dom_a == 'background' or dom_b == 'background'
    row = {
        'realisation': run_id,
        'parent_group': group_id,
        'depth': depth,
        'parent_size': len(nodes),
        'n_a': best['n_a'],
        'n_b': best['n_b'],
        'edge_length_pc': best['edge_length_pc'],
        'median_av_a': best['median_a'],
        'median_av_b': best['median_b'],
        'robust_effect_size': best['effect_size'],
        'mann_whitney_p': best['p'],
        'mann_whitney_q': best['q'],
        'bootstrap_stability': best['bootstrap_stability'],
        'bootstrap_effect_median': best['bootstrap_effect_median'],
        'bootstrap_effect_p2_5': best['bootstrap_effect_p2_5'],
        'bootstrap_effect_p97_5': best['bootstrap_effect_p97_5'],
        'coherence_threshold_pc': best['coherence_threshold_pc'],
        'topology_persistence': best['topology_persistence'],
        'selection_score': best['selection_score'],
        'dominant_truth_a': dom_a,
        'dominant_truth_b': dom_b,
        'dominant_fraction_a': cnt_a / best['n_a'],
        'dominant_fraction_b': cnt_b / best['n_b'],
        'correct_injected_separation': correct,
        'noninjected_or_mixed_split': noninjected,
        **mix,
    }
    split_rows.append(row)
    left = analyse_and_split_group(best['a_nodes'], global_u, global_v, global_w, av, truth, global_threshold, rng, run_id, group_id + 'A', depth + 1, split_rows)
    right = analyse_and_split_group(best['b_nodes'], global_u, global_v, global_w, av, truth, global_threshold, rng, run_id, group_id + 'B', depth + 1, split_rows)
    return left + right


def sample_anchor_indices(field_xyz: np.ndarray, field_dist: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    good = np.flatnonzero((field_dist >= CFG['centre_distance_min_pc']) & (field_dist <= CFG['centre_distance_max_pc']))
    if len(good) == 0:
        raise RuntimeError('No anchor candidates')
    return good


def place_template(rel: np.ndarray, centre: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return rel @ rotation.T + centre


def make_realisation(field_df: pd.DataFrame, templates: list[dict], run: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict]:
    rng = np.random.default_rng(CFG['seed'] + run)
    field_xyz = field_df[['X', 'Y', 'Z']].to_numpy(float)
    field_av = pd.to_numeric(field_df['AV'], errors='coerce').to_numpy(float)
    field_dist = np.linalg.norm(field_xyz, axis=1)
    anchor_pool = sample_anchor_indices(field_xyz, field_dist, rng)

    rotations = [Rotation.random(random_state=rng).as_matrix() for _ in templates]
    centres: list[np.ndarray] = []
    placed: list[np.ndarray] = []
    placement_rows = []

    # The two largest empirical templates form the deliberately superposed pair.
    overlap_direction = field_xyz[int(rng.choice(anchor_pool))]
    overlap_direction = overlap_direction / np.linalg.norm(overlap_direction)
    d0 = float(rng.uniform(600.0, 820.0))
    delta = float(rng.uniform(CFG['overlap_distance_min_pc'], CFG['overlap_distance_max_pc']))
    overlap_dists = [d0, d0 + delta]

    for idx, template in enumerate(templates):
        rel = template['rel_xyz']
        rot = rotations[idx]
        if idx < 2:
            centre = overlap_direction * overlap_dists[idx]
            xyz = place_template(rel, centre, rot)
            # If an extreme orientation crosses the 1-kpc boundary, rotate again a few times.
            tries = 0
            while (np.linalg.norm(xyz, axis=1).max() > 1000.0 or np.linalg.norm(xyz, axis=1).min() < 30.0) and tries < 100:
                rot = Rotation.random(random_state=rng).as_matrix()
                xyz = place_template(rel, centre, rot)
                tries += 1
        else:
            xyz = None
            for tries in range(1000):
                centre = field_xyz[int(rng.choice(anchor_pool))].copy()
                if all(np.linalg.norm(centre - c) >= CFG['centre_min_separation_pc'] for c in centres):
                    rot = Rotation.random(random_state=rng).as_matrix()
                    trial = place_template(rel, centre, rot)
                    radii = np.linalg.norm(trial, axis=1)
                    if radii.min() >= 30.0 and radii.max() <= 1000.0:
                        xyz = trial
                        break
            if xyz is None:
                raise RuntimeError(f'Could not place template {idx}')
        centres.append(centre)
        placed.append(xyz)
        placement_rows.append({
            'realisation': run,
            'template_label': template['label'],
            'template_gid': template['gid'],
            'n_stars': len(xyz),
            'is_overlap_member': idx < 2,
            'centre_x_pc': float(centre[0]),
            'centre_y_pc': float(centre[1]),
            'centre_z_pc': float(centre[2]),
            'centre_distance_pc': float(np.linalg.norm(centre)),
            'rotation_matrix': json.dumps(rot.tolist()),
            'median_av': float(np.median(template['av'])),
        })

    cluster_xyz = np.vstack(placed)
    cluster_av = np.concatenate([t['av'] for t in templates])
    cluster_truth = np.concatenate([np.repeat(t['label'], len(t['av'])) for t in templates])
    xyz = np.vstack([field_xyz, cluster_xyz])
    av = np.concatenate([field_av, cluster_av])
    truth = np.concatenate([np.repeat('background', len(field_df)), cluster_truth])
    order = rng.permutation(len(xyz))
    xyz, av, truth = xyz[order], av[order], truth[order]
    meta = {
        'overlap_pair': [templates[0]['label'], templates[1]['label']],
        'overlap_distance_separation_pc': delta,
        'overlap_av_median_separation_mag': abs(float(np.median(templates[0]['av'])) - float(np.median(templates[1]['av']))),
        'n_field': len(field_df),
        'n_injected': len(cluster_xyz),
        'n_total': len(xyz),
    }
    return xyz, av, truth, pd.DataFrame(placement_rows), meta


def evaluate_groups(groups: list[np.ndarray], truth: np.ndarray, labels: list[str], run: int, stage: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    counters = [Counter(truth[g]) for g in groups]
    group_sizes = [len(g) for g in groups]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    best_group_map = {}
    for label in labels:
        true_size = int(np.sum(truth == label))
        overlaps = np.array([c.get(label, 0) for c in counters], dtype=int)
        if len(overlaps) and overlaps.max() > 0:
            best = int(np.argmax(overlaps))
            recovered = int(overlaps[best])
            gsize = int(group_sizes[best])
            best_id = best + 1
        else:
            best, recovered, gsize, best_id = -1, 0, 0, np.nan
        completeness = recovered / true_size if true_size else 0.0
        purity = recovered / gsize if gsize else 0.0
        detected = recovered >= max(CFG['min_group_size'], math.ceil(0.5 * true_size))
        strict = completeness >= 0.8 and purity >= 0.5
        fragments = int(sum(c.get(label, 0) >= max(5, math.ceil(0.10 * true_size)) for c in counters))
        merged_labels = []
        if best >= 0:
            for other in labels:
                if other == label:
                    continue
                other_true = int(np.sum(truth == other))
                if counters[best].get(other, 0) >= max(5, math.ceil(0.20 * other_true)):
                    merged_labels.append(other)
        best_group_map[label] = best_id
        rows.append({
            'realisation': run,
            'stage': stage,
            'template_label': label,
            'true_size': true_size,
            'best_group_id': best_id,
            'recovered_members': recovered,
            'best_group_size': gsize,
            'completeness': completeness,
            'purity': purity,
            'detected_ge50pct': bool(detected),
            'strict_recovery_c80_p50': bool(strict),
            'fragment_count': fragments,
            'merged_with_other_templates': ';'.join(merged_labels),
            'n_merged_templates': len(merged_labels),
        })
    tab = pd.DataFrame(rows)
    noninjected_groups = 0
    injected_dominant_groups = 0
    for c in counters:
        dom = c.most_common(1)[0][0]
        if dom == 'background':
            noninjected_groups += 1
        else:
            injected_dominant_groups += 1
    overlap_resolved = (
        bool(tab.loc[tab.template_label == labels[0], 'detected_ge50pct'].iloc[0])
        and bool(tab.loc[tab.template_label == labels[1], 'detected_ge50pct'].iloc[0])
        and best_group_map[labels[0]] != best_group_map[labels[1]]
    )
    summary = {
        'groups_ge10': len(groups),
        'injected_dominant_groups': injected_dominant_groups,
        'noninjected_dominant_groups': noninjected_groups,
        'templates_detected_ge50pct': int(tab.detected_ge50pct.sum()),
        'templates_strict_recovery': int(tab.strict_recovery_c80_p50.sum()),
        'all_eight_detected': bool(tab.detected_ge50pct.all()),
        'all_eight_strict': bool(tab.strict_recovery_c80_p50.all()),
        'mean_completeness': float(tab.completeness.mean()),
        'mean_purity': float(tab.purity.mean()),
        'overlap_pair_resolved': bool(overlap_resolved),
    }
    return tab, summary


def load_data():
    df = pd.read_csv(Q25, low_memory=False)
    mem = pd.read_csv(MEMBERSHIP)
    selected = pd.read_csv(TEMPLATE_SUMMARY)
    labels = []
    templates = []
    selected_rows = []
    for rank, r in selected.reset_index(drop=True).iterrows():
        gid = int(r.gid)
        idx = mem.loc[mem.group_label == gid, 'row'].to_numpy(int)
        tdf = df.iloc[idx].copy()
        xyz = tdf[['X', 'Y', 'Z']].to_numpy(float)
        centre = np.median(xyz, axis=0)
        rel = xyz - centre
        label = f'empirical_template_{rank+1:02d}'
        labels.append(label)
        templates.append({
            'label': label,
            'gid': gid,
            'indices': idx,
            'rel_xyz': rel,
            'av': pd.to_numeric(tdf['AV'], errors='coerce').to_numpy(float),
            'original_centre': centre,
        })
        selected_rows.extend(idx.tolist())
    if sum(len(t['indices']) for t in templates) != 697:
        raise RuntimeError('Template total is not 697')
    field_df = df.drop(index=np.unique(selected_rows)).reset_index(drop=True)
    if len(field_df) != 24706 - 697:
        raise RuntimeError('Field row count mismatch')
    return df, field_df, templates, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    outdir = ROOT / 'runs'
    outdir.mkdir(parents=True, exist_ok=True)
    _, field_df, templates, labels = load_data()
    (ROOT / 'cdf_mc1000_method_config.json').write_text(json.dumps(CFG, indent=2), encoding='utf-8')
    template_manifest = []
    for t in templates:
        template_manifest.append({
            'template_label': t['label'],
            'source_q25_jenks_group_id': t['gid'],
            'n_stars': len(t['indices']),
            'median_av': float(np.median(t['av'])),
            'original_centre_x_pc': float(t['original_centre'][0]),
            'original_centre_y_pc': float(t['original_centre'][1]),
            'original_centre_z_pc': float(t['original_centre'][2]),
            'max_radius_from_centre_pc': float(np.max(np.linalg.norm(t['rel_xyz'], axis=1))),
        })
    pd.DataFrame(template_manifest).to_csv(ROOT / 'cdf_mc1000_template_manifest.csv', index=False)

    for run in range(args.start, args.end):
        run_dir = outdir / f'mc_{run:04d}'
        done = run_dir / 'run_summary.json'
        if done.exists() and not args.overwrite:
            print(f'[{run+1:04d}/1000] skip completed', flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        rng = np.random.default_rng(CFG['seed'] + run + 10_000_000)
        xyz, av, truth, placements, meta = make_realisation(field_df, templates, run)
        placements.to_csv(run_dir / 'placements.csv', index=False)
        if run == 0:
            pd.DataFrame({
                'x_pc': xyz[:, 0], 'y_pc': xyz[:, 1], 'z_pc': xyz[:, 2],
                'reddening_mag': av, 'true_component': truth,
            }).to_csv(ROOT / 'mc_0000_example_catalogue.csv.gz', index=False, compression='gzip')
        u, v, w, candidate_edges = exact_mst(xyz)
        cdf = fit_two_line_cdf(w, CFG['min_cdf_segment'])
        ccut = float(cdf['threshold'])
        cbmean, cbsd, cbp16, cbmed, cbp84 = cdf_edge_bootstrap(
            w, rng, CFG['cdf_bootstraps'], CFG['min_cdf_segment']
        )
        cdf_groups, labels_all = components_from_cut(len(xyz), u, v, w, ccut, CFG['min_group_size'])
        cdf_recovery, cdf_summary = evaluate_groups(cdf_groups, truth, labels, run, 'global_cdf')
        cdf_recovery.to_csv(run_dir / 'template_recovery.csv', index=False)
        if run == 0:
            pd.DataFrame({'u': u, 'v': v, 'edge_length_pc': w}).to_csv(
                run_dir / 'mst_edges.csv.gz', index=False, compression='gzip'
            )
        summary = {
            'realisation': run,
            'seed': CFG['seed'] + run,
            'elapsed_seconds': time.time() - started,
            'n_stars': len(xyz),
            'n_field': meta['n_field'],
            'n_injected': meta['n_injected'],
            'candidate_graph_edges': candidate_edges,
            'cdf_threshold_pc': ccut,
            'cdf_threshold_method': cdf['threshold_method'],
            'cdf_split_index': cdf['split_index'],
            'cdf_slope_1': cdf['slope_1'],
            'cdf_intercept_1': cdf['intercept_1'],
            'cdf_slope_2': cdf['slope_2'],
            'cdf_intercept_2': cdf['intercept_2'],
            'cdf_two_line_sse': cdf['two_line_sse'],
            'cdf_one_line_sse': cdf['one_line_sse'],
            'cdf_fit_improvement_fraction': cdf['fit_improvement_fraction'],
            'cdf_bootstrap_mean_pc': cbmean,
            'cdf_bootstrap_sd_pc': cbsd,
            'cdf_bootstrap_p16_pc': cbp16,
            'cdf_bootstrap_median_pc': cbmed,
            'cdf_bootstrap_p84_pc': cbp84,
            'overlap_distance_separation_pc': meta['overlap_distance_separation_pc'],
            'overlap_av_median_separation_mag': meta['overlap_av_median_separation_mag'],
            **{f'global_{k}': v for k, v in cdf_summary.items()},
        }
        done.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(
            f"[{run+1:04d}/1000] {summary['elapsed_seconds']:.1f}s "
            f"CDF={ccut:.2f} pc recovered={cdf_summary['templates_detected_ge50pct']}/8 "
            f"purity={cdf_summary['mean_purity']:.3f} "
            f"overlap={cdf_summary['overlap_pair_resolved']}",
            flush=True,
        )


if __name__ == '__main__':
    main()

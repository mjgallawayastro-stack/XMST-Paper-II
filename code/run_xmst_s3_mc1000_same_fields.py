#!/usr/bin/env python3
"""Controlled XMST-S3 validation on the exact empirical-template MC fields.

No Stage-1, S2 or S3 algorithm function is modified or monkey-patched.

Two pre-declared tests are run for every MC field:

A) FULL-PIPELINE NULL CONTROL
   Exact Stage1 -> frozen S2 -> frozen S3, with every star drawn from one common
   bivariate Gaussian velocity distribution. There is no injected kinematic
   substructure. This measures false S3 splits and damage under the null.

B) S3-SPECIFIC OVERLAP RECOVERY CURVE
   The exact frozen S3 routine is applied only to the Stage-1 parent containing
   the planted T01/T02 merger (when Stage 1 actually merges them). S2 is bypassed
   only for this labelled diagnostic because frozen S2 already separates T01/T02
   in ~99.5% of the fields. T01/T02 receive opposite centroid shifts on a paired
   velocity-noise draw; the rest of the catalogue is unchanged. This measures
   the velocity separation at which the frozen S3 gate/reconstruction resolves
   the known spatial merger.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import cdf_worker
import run_xmsts2 as s2ref
import run_xmst_s3 as s3mod

BASE = Path(__file__).resolve().parent
ROOT = BASE / "output_s3_mc"
MC_CFG = cdf_worker.CFG
load_data = cdf_worker.load_data
make_realisation = cdf_worker.make_realisation
evaluate_groups = cdf_worker.evaluate_groups

FROZEN_S3_SHA256 = "cd95b3a87a0942732f007f7ca82c9986080706c4b46f575039efe00f4710eb32"
FROZEN_S2_SHA256 = "2cca819d10f9f7795cea25c09b05cff7c9b1c273a2c34d35dc83e815080bfcea"
FROZEN_CDF_WORKER_SHA256 = "ac3bd48026f890339103345d95278e8cbe7447c791b2ee10f7c47668d040eafe"

_FIELD_DF = None
_TEMPLATES = None
_LABELS = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> dict:
    expected = {
        "run_xmst_s3.py": FROZEN_S3_SHA256,
        "run_xmsts2.py": FROZEN_S2_SHA256,
        "cdf_worker.py": FROZEN_CDF_WORKER_SHA256,
    }
    got = {}
    for name, want in expected.items():
        p = BASE / name
        if not p.exists():
            raise RuntimeError(f"Required frozen source missing: {p}")
        digest = sha256(p)
        got[name] = digest
        if digest != want:
            raise RuntimeError(
                f"{name} SHA256 mismatch. Expected {want}, got {digest}. "
                "Aborting rather than validating changed code."
            )
    return got


def init_worker():
    global _FIELD_DF, _TEMPLATES, _LABELS
    _, _FIELD_DF, _TEMPLATES, _LABELS = load_data()


def make_input(xyz, av, truth, run):
    return pd.DataFrame({
        "source_id": [f"mc{run:04d}_{i:05d}" for i in range(len(xyz))],
        "x_pc": xyz[:, 0], "y_pc": xyz[:, 1], "z_pc": xyz[:, 2],
        "reddening_mag": av, "true_component": truth,
    })


def ids_to_groups(ids):
    ids = np.asarray(ids)
    return [np.flatnonzero(ids == gid) for gid in sorted(int(x) for x in np.unique(ids) if int(x) > 0)]


def nodes_to_groups(nodes, final_ids):
    return [np.asarray(nodes[nid]["indices"], dtype=np.int64) for nid in final_ids]


def partition_signature(groups):
    return sorted(tuple(sorted(map(int, g))) for g in groups)


def score_groups(groups, truth, labels, run, stage, delta=np.nan):
    tab, summary = evaluate_groups(groups, truth, labels, run, stage)
    denom = tab["true_size"] + tab["best_group_size"] - tab["recovered_members"]
    tab["jaccard"] = np.where(denom > 0, tab["recovered_members"] / denom, 0.0)
    tab.insert(2, "velocity_separation_kms", delta)
    a = tab.loc[tab.template_label == labels[0]].iloc[0]
    b = tab.loc[tab.template_label == labels[1]].iloc[0]
    both = pd.notna(a.best_group_id) and pd.notna(b.best_group_id)
    summary = dict(summary)
    summary["mean_jaccard"] = float(tab.jaccard.mean())
    summary["mean_fragment_count"] = float(tab.fragment_count.mean())
    summary["mean_n_merged_templates"] = float(tab.n_merged_templates.mean())
    summary["overlap_best_groups_separate"] = bool(both and a.best_group_id != b.best_group_id)
    summary["t01_best_group_id"] = None if pd.isna(a.best_group_id) else int(a.best_group_id)
    summary["t02_best_group_id"] = None if pd.isna(b.best_group_id) else int(b.best_group_id)
    return tab, summary


def frozen_config(run_dir):
    return s3mod.Config(
        input_csv="<generated>", kinematics_csv=None, output_dir=str(run_dir), min_group_size=10,
        s2_delta_bic_threshold=10.0, s2_ashman_d_threshold=2.0,
        s2_min_gmm_component_n=5, s2_min_gmm_component_fraction=0.15,
        s2_min_reddening_coverage=1.0, s2_gmm_n_init=20, s2_gmm_random_state=20260825,
        s2_spatial_child_policy="all", max_s2_passes=20,
        s3_delta_bic_threshold=10.0, s3_separation_d_threshold=2.0,
        s3_min_gmm_component_n=5, s3_min_gmm_component_fraction=0.15,
        s3_min_kinematic_coverage=1.0, s3_gmm_n_init=20, s3_gmm_random_state=20260825,
        s3_spatial_child_policy="all", max_s3_passes=20,
        strict_q25_checkpoint=False, strict_s2_checkpoint=False, skip_s3=False,
    )


def run_stage1_s2(df, xyz, av, cfg):
    """Call the exact embedded Stage-1 and S2 functions, avoiding unrelated I/O/membership work."""
    with contextlib.redirect_stdout(io.StringIO()) as log:
        s1 = s3mod.stage1_xmst(xyz, cfg.min_group_size)
        s2 = s3mod.run_stage2(
            df, xyz, av, s1["group_ids"], s1["fracture_scale_pc"], "source_id", cfg
        )
    return s1, s2, log.getvalue()


def run_s3(df, xyz, vl, vb, parent_nodes, parent_ids, fracture, cfg):
    with contextlib.redirect_stdout(io.StringIO()) as log:
        out = s3mod.run_stage3(
            df, xyz, vl, vb, parent_nodes, parent_ids, fracture, "source_id", cfg
        )
    return out, log.getvalue()


def s3_behavior(s3):
    p = s3["passes"]
    return {
        "s3_passes": int(len(p)),
        "s3_significant_flags": int(p.significant_bimodality.sum()) if len(p) else 0,
        "s3_accepted_splits": int(p.accepted_splits.sum()) if len(p) else 0,
        "s3_final_groups": int(len(s3["final_node_ids"])),
        "s3_orphans": int(len(s3["orphans"])),
    }


def velocity_base(truth, run, sigma):
    rng = np.random.default_rng(int(MC_CFG["seed"]) + int(run) + 30_000_000)
    noise = rng.normal(0.0, sigma, size=(len(truth), 2))
    theta = rng.uniform(0, 2*np.pi)
    unit = np.array([np.cos(theta), np.sin(theta)])
    return noise, unit, float(theta)


def shifted_velocities(noise, unit, truth, labels, delta):
    v = noise.copy()
    if float(delta) != 0.0:
        shift = 0.5 * float(delta) * unit
        v[truth == labels[0]] -= shift
        v[truth == labels[1]] += shift
    return v


def target_parent_s3(stage1_groups, s1_score, df, xyz, vl, vb, fracture, cfg):
    """Run frozen S3 only on the Stage-1 parent shared by T01/T02.

    If Stage 1 already separates them, return the unchanged Stage-1 groups.
    Otherwise replace only the shared parent by the exact S3 descendants.
    """
    g1 = s1_score["t01_best_group_id"]
    g2 = s1_score["t02_best_group_id"]
    if g1 is None or g2 is None or g1 != g2:
        return stage1_groups, None, "already_separate"

    target_pos = int(g1) - 1  # evaluate_groups best_group_id is 1-based list position
    target_idx = np.asarray(stage1_groups[target_pos], dtype=np.int64)
    node_id = f"S1_OVERLAP_PARENT_{int(g1):03d}"
    nodes = {node_id: {"root_stage1_group_id": int(g1), "indices": target_idx}}
    s3, log = run_s3(df, xyz, vl, vb, nodes, [node_id], fracture, cfg)
    descendants = nodes_to_groups(s3["nodes"], s3["final_node_ids"])
    combined = [g for i, g in enumerate(stage1_groups) if i != target_pos] + descendants
    return combined, s3, log


def run_one(run, deltas, sigma, overwrite=False):
    global _FIELD_DF, _TEMPLATES, _LABELS
    if _FIELD_DF is None:
        init_worker()
    run_dir = ROOT / "runs" / f"mc_{run:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "DONE.json"
    if done.exists() and not overwrite:
        return json.loads(done.read_text())

    t0 = time.time()
    xyz, av, truth, placements, meta = make_realisation(_FIELD_DF, _TEMPLATES, run)
    df = make_input(xyz, av, truth, run)
    cfg = frozen_config(run_dir)
    s1, s2, base_log = run_stage1_s2(df, xyz, av, cfg)
    (run_dir / "stage1_s2_console.txt").write_text(base_log)

    stage1_groups = ids_to_groups(s1["group_ids"])
    s2_groups = nodes_to_groups(s2["nodes"], s2["final_node_ids"])
    s1_rec, s1_score = score_groups(stage1_groups, truth, _LABELS, run, "stage1")
    s2_rec, s2_score = score_groups(s2_groups, truth, _LABELS, run, "xmst_s2")
    s1_rec.to_csv(run_dir / "stage1_template_recovery.csv", index=False)
    s2_rec.to_csv(run_dir / "s2_template_recovery.csv", index=False)
    placements.to_csv(run_dir / "placements.csv", index=False)

    noise, unit, theta = velocity_base(truth, run, sigma)
    rows = []
    recs = []
    splits = []

    for stage, sc in [("Stage 1", s1_score), ("XMST-S2", s2_score)]:
        rows.append({
            "realisation": run, "stage": stage, "velocity_separation_kms": np.nan,
            "mean_completeness": sc["mean_completeness"], "mean_purity": sc["mean_purity"],
            "mean_jaccard": sc["mean_jaccard"], "strict_recoveries_of_8": sc["templates_strict_recovery"],
            "formal_overlap_resolved": sc["overlap_pair_resolved"],
            "overlap_best_groups_separate": sc["overlap_best_groups_separate"],
            "mean_fragment_count": sc["mean_fragment_count"],
            "s3_significant_flags": 0, "s3_accepted_splits": 0, "s3_orphans": 0,
        })

    # A) Full pipeline null control: all stars share one Gaussian velocity population.
    vl0, vb0 = noise[:, 0], noise[:, 1]
    s3null, null_log = run_s3(df, xyz, vl0, vb0, s2["nodes"], s2["final_node_ids"], s1["fracture_scale_pc"], cfg)
    null_groups = nodes_to_groups(s3null["nodes"], s3null["final_node_ids"])
    null_rec, null_score = score_groups(null_groups, truth, _LABELS, run, "xmst_s2_to_s3_null", 0.0)
    recs.append(null_rec)
    beh = s3_behavior(s3null)
    rows.append({
        "realisation": run, "stage": "XMST-S2→S3 null", "velocity_separation_kms": 0.0,
        "mean_completeness": null_score["mean_completeness"], "mean_purity": null_score["mean_purity"],
        "mean_jaccard": null_score["mean_jaccard"], "strict_recoveries_of_8": null_score["templates_strict_recovery"],
        "formal_overlap_resolved": null_score["overlap_pair_resolved"],
        "overlap_best_groups_separate": null_score["overlap_best_groups_separate"],
        "mean_fragment_count": null_score["mean_fragment_count"], **beh,
    })
    (run_dir / "s3_pipeline_null_console.txt").write_text(null_log)
    if len(s3null["splits"]):
        x = s3null["splits"].copy(); x.insert(0, "mode", "pipeline_null"); x.insert(0, "velocity_separation_kms", 0.0); x.insert(0, "realisation", run); splits.append(x)

    # B) S3-specific recovery curve on the planted Stage-1 T01/T02 merged parent.
    for delta in deltas:
        V = shifted_velocities(noise, unit, truth, _LABELS, delta)
        groups, s3diag, diag_log = target_parent_s3(
            stage1_groups, s1_score, df, xyz, V[:, 0], V[:, 1], s1["fracture_scale_pc"], cfg
        )
        rec, sc = score_groups(groups, truth, _LABELS, run, "stage1_to_s3_overlap_diagnostic", float(delta))
        recs.append(rec)
        if s3diag is None:
            beh = {"s3_passes": 0, "s3_significant_flags": 0, "s3_accepted_splits": 0, "s3_final_groups": len(stage1_groups), "s3_orphans": 0}
        else:
            beh = s3_behavior(s3diag)
            if len(s3diag["splits"]):
                x = s3diag["splits"].copy(); x.insert(0, "mode", "stage1_overlap_diagnostic"); x.insert(0, "velocity_separation_kms", float(delta)); x.insert(0, "realisation", run); splits.append(x)
        rows.append({
            "realisation": run, "stage": "Stage1→S3 overlap diagnostic", "velocity_separation_kms": float(delta),
            "mean_completeness": sc["mean_completeness"], "mean_purity": sc["mean_purity"],
            "mean_jaccard": sc["mean_jaccard"], "strict_recoveries_of_8": sc["templates_strict_recovery"],
            "formal_overlap_resolved": sc["overlap_pair_resolved"],
            "overlap_best_groups_separate": sc["overlap_best_groups_separate"],
            "mean_fragment_count": sc["mean_fragment_count"], **beh,
        })
        if delta in (deltas[0], deltas[-1]):
            (run_dir / f"s3_overlap_delta_{delta:g}_console.txt").write_text(str(diag_log))

    pd.DataFrame(rows).to_csv(run_dir / "grid_summary.csv", index=False)
    pd.concat(recs, ignore_index=True).to_csv(run_dir / "s3_template_recovery.csv", index=False)
    if splits:
        pd.concat(splits, ignore_index=True).to_csv(run_dir / "s3_split_diagnostics.csv", index=False)

    payload = {
        "realisation": int(run), "elapsed_seconds": float(time.time()-t0),
        "stage1_fracture_scale_pc": float(s1["fracture_scale_pc"]),
        "stage1_groups": int(len(stage1_groups)), "s2_groups": int(len(s2_groups)),
        "s2_accepted_splits": int(s2["passes"].accepted_splits.sum()) if len(s2["passes"]) else 0,
        "velocity_sigma_kms": float(sigma), "velocity_direction_angle_rad": theta,
        "velocity_deltas_kms": list(map(float, deltas)),
        "overlap_distance_separation_pc": float(meta["overlap_distance_separation_pc"]),
    }
    done.write_text(json.dumps(payload, indent=2))
    return payload


def aggregate(start, end, deltas, sigma):
    S=[]; R=[]; X=[]; D=[]
    for run in range(start, end):
        d=ROOT/"runs"/f"mc_{run:04d}"
        if not (d/"DONE.json").exists(): continue
        D.append(json.loads((d/"DONE.json").read_text()))
        if (d/"grid_summary.csv").exists(): S.append(pd.read_csv(d/"grid_summary.csv"))
        if (d/"s3_template_recovery.csv").exists(): R.append(pd.read_csv(d/"s3_template_recovery.csv"))
        if (d/"s3_split_diagnostics.csv").exists(): X.append(pd.read_csv(d/"s3_split_diagnostics.csv"))
    if not S: return
    S=pd.concat(S,ignore_index=True); S.to_csv(ROOT/"xmst_s3_mc_realisation_grid.csv",index=False)
    pd.DataFrame(D).sort_values("realisation").to_csv(ROOT/"xmst_s3_mc_run_metadata.csv",index=False)
    if X: pd.concat(X,ignore_index=True).to_csv(ROOT/"xmst_s3_mc_split_diagnostics.csv",index=False)
    if R:
        R=pd.concat(R,ignore_index=True); R.to_csv(ROOT/"xmst_s3_mc_template_recovery.csv",index=False)
        rows=[]
        for (stage,delta,label),g in R.groupby(["stage","velocity_separation_kms","template_label"],dropna=False,sort=True):
            rows.append({"stage":stage,"velocity_separation_kms":delta,"template_label":label,"true_size":int(g.true_size.iloc[0]),"n_realisations":len(g),"mean_completeness":g.completeness.mean(),"mean_purity":g.purity.mean(),"mean_jaccard":g.jaccard.mean(),"detected_ge50pct_fraction":g.detected_ge50pct.mean(),"strict_recovery_c80_p50_fraction":g.strict_recovery_c80_p50.mean(),"mean_fragment_count":g.fragment_count.mean(),"mean_n_merged_templates":g.n_merged_templates.mean()})
        pd.DataFrame(rows).to_csv(ROOT/"xmst_s3_mc_template_summary.csv",index=False)

    H=[]
    for stage in ["Stage 1","XMST-S2","XMST-S2→S3 null"]:
        g=S[S.stage==stage]
        if len(g):
            H.append({"stage":stage,"velocity_separation_kms":np.nan if stage in ["Stage 1","XMST-S2"] else 0.0,"n_realisations":len(g),"mean_completeness":g.mean_completeness.mean(),"mean_purity":g.mean_purity.mean(),"mean_jaccard":g.mean_jaccard.mean(),"mean_strict_recoveries_of_8":g.strict_recoveries_of_8.mean(),"formal_overlap_resolution_rate":g.formal_overlap_resolved.mean(),"best_group_merger_break_rate":g.overlap_best_groups_separate.mean(),"mean_fragment_count":g.mean_fragment_count.mean(),"mean_s3_accepted_splits":g.s3_accepted_splits.mean(),"mean_s3_orphans":g.s3_orphans.mean()})
    stage="Stage1→S3 overlap diagnostic"
    for delta in deltas:
        g=S[(S.stage==stage)&np.isclose(S.velocity_separation_kms,delta)]
        if len(g):
            H.append({"stage":stage,"velocity_separation_kms":float(delta),"n_realisations":len(g),"mean_completeness":g.mean_completeness.mean(),"mean_purity":g.mean_purity.mean(),"mean_jaccard":g.mean_jaccard.mean(),"mean_strict_recoveries_of_8":g.strict_recoveries_of_8.mean(),"formal_overlap_resolution_rate":g.formal_overlap_resolved.mean(),"best_group_merger_break_rate":g.overlap_best_groups_separate.mean(),"mean_fragment_count":g.mean_fragment_count.mean(),"mean_s3_accepted_splits":g.s3_accepted_splits.mean(),"mean_s3_orphans":g.s3_orphans.mean()})
    H=pd.DataFrame(H); H.to_csv(ROOT/"xmst_s3_mc_headline.csv",index=False)
    H[H.stage==stage].to_csv(ROOT/"xmst_s3_mc_overlap_curve.csv",index=False)
    H[H.stage=="XMST-S2→S3 null"].to_csv(ROOT/"xmst_s3_mc_null_control.csv",index=False)

    report=["# XMST-S3 controlled validation","",f"Realisations aggregated: {S.realisation.nunique()}",f"Velocity sigma: {sigma:g} km/s per axis","Velocity-separation grid: "+", ".join(f"{d:g}" for d in deltas)+" km/s","","## Frozen gate","- ΔBIC >= 10","- D_2D >= 2","- each GMM component >= 5 stars","- each GMM component >= 15%","- 100% kinematic coverage","- all fixed-scale spatial descendants N >= 10","","## Pre-declared tests","- Full-pipeline null: Stage1 -> S2 -> S3 with no true kinematic substructure.","- Overlap diagnostic: exact S3 on the Stage-1 T01/T02 parent, bypassing S2 only because S2 already resolves that planted merger in almost every field.","- Formal overlap recovery and best-group merger breaking are reported separately.","","## Headline"]
    for _,r in H.iterrows():
        d="baseline" if pd.isna(r.velocity_separation_kms) else f"ΔV={r.velocity_separation_kms:g} km/s"
        report.append(f"- {r.stage} ({d}): C={r.mean_completeness:.4f}, P={r.mean_purity:.4f}, J={r.mean_jaccard:.4f}, strict={r.mean_strict_recoveries_of_8:.3f}/8, formal overlap={100*r.formal_overlap_resolution_rate:.1f}%, merger break={100*r.best_group_merger_break_rate:.1f}%, S3 splits={r.mean_s3_accepted_splits:.3f}")
    (ROOT/"XMST_S3_MC_REPORT.md").write_text("\n".join(report)+"\n")


def validate_guards():
    _,field_df,templates,_=load_data(); xyz,av,truth,_,_=make_realisation(field_df,templates,0)
    example=BASE/"output"/"mc_0000_example_catalogue.csv.gz"
    if example.exists():
        ref=pd.read_csv(example,low_memory=False)
        if not np.allclose(ref[["x_pc","y_pc","z_pc"]].to_numpy(float),xyz,rtol=0,atol=1e-12): raise RuntimeError("MC0000 guard failed: XYZ")
        if not np.allclose(pd.to_numeric(ref.reddening_mag,errors="coerce"),av,rtol=0,atol=1e-12,equal_nan=True): raise RuntimeError("MC0000 guard failed: AV")
        if not np.array_equal(ref.true_component.astype(str).to_numpy(),truth.astype(str)): raise RuntimeError("MC0000 guard failed: truth")
        print("Guard 1 OK: MC0000 exactly matches archived catalogue")
    df=make_input(xyz,av,truth,0); cfg=frozen_config(ROOT/"_guard")
    refcfg=s2ref.Config(input_csv="<guard>",output_dir="<guard>",min_group_size=10,delta_bic_threshold=10.0,ashman_d_threshold=2.0,min_gmm_component_n=5,min_gmm_component_fraction=0.15,min_reddening_coverage=1.0,gmm_n_init=20,gmm_random_state=20260825,spatial_child_policy="all",max_passes=20,strict_q25_checkpoint=False)
    with contextlib.redirect_stdout(io.StringIO()):
        r=s2ref.run_xmsts2(df,refcfg)
        s1,s2,_=run_stage1_s2(df,xyz,av,cfg)
    if not np.isclose(r["stage1"]["fracture_scale_pc"],s1["fracture_scale_pc"],atol=1e-12,rtol=0): raise RuntimeError("Guard 2 failed: fracture")
    if partition_signature(ids_to_groups(r["stage1"]["group_ids"])) != partition_signature(ids_to_groups(s1["group_ids"])): raise RuntimeError("Guard 2 failed: Stage1 partition")
    if partition_signature(ids_to_groups(r["membership"].xmsts2_final_group_id.to_numpy(int))) != partition_signature(nodes_to_groups(s2["nodes"],s2["final_node_ids"])): raise RuntimeError("Guard 2 failed: S2 partition")
    print("Guard 2 OK: run_xmst_s3 embedded Stage1/S2 exactly matches frozen XMST-S2 on MC0000")


def parse_deltas(text):
    vals=tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not vals or any(x<0 for x in vals): raise ValueError("deltas must be non-negative")
    return vals


def method_config(deltas,sigma):
    return {"mc_generator":"cdf_worker.make_realisation","mc_seed_base":int(MC_CFG["seed"]),"velocity_seed_offset":30000000,"velocity_sigma_kms_per_axis":float(sigma),"velocity_centroid_separations_kms":list(map(float,deltas)),"s3_source_sha256":FROZEN_S3_SHA256,"s2_reference_sha256":FROZEN_S2_SHA256,"cdf_worker_sha256":FROZEN_CDF_WORKER_SHA256,"algorithm_modifications":"none","full_pipeline_null":"Stage1 -> frozen S2 -> frozen S3; all stars one common bivariate Gaussian","s3_specific_overlap_diagnostic":"exact frozen S3 applied only to Stage1 T01/T02 parent; S2 bypassed only for diagnostic","s3_gate":{"delta_bic":10.0,"D_2D":2.0,"min_gmm_component_n":5,"min_gmm_component_fraction":0.15,"min_kinematic_coverage":1.0,"spatial_child_policy":"all","min_retained_group_size":10}}


def ensure_config(deltas,sigma,overwrite):
    ROOT.mkdir(exist_ok=True); p=ROOT/"xmst_s3_mc_method_config.json"; c=method_config(deltas,sigma)
    if p.exists() and not overwrite and json.loads(p.read_text()) != c: raise RuntimeError("output_s3_mc has a different method config; use a clean folder or --overwrite")
    p.write_text(json.dumps(c,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",type=int,default=0); ap.add_argument("--end",type=int,default=1000); ap.add_argument("--workers",type=int,default=2); ap.add_argument("--deltas",default="0,1,2,3,5,10"); ap.add_argument("--sigma",type=float,default=1.0); ap.add_argument("--overwrite",action="store_true"); ap.add_argument("--skip-guards",action="store_true"); args=ap.parse_args()
    if not (0<=args.start<args.end<=1000): raise SystemExit("Require 0 <= start < end <= 1000")
    if args.sigma<=0: raise SystemExit("--sigma must be > 0")
    deltas=parse_deltas(args.deltas)
    print("Frozen sources verified:"); [print(f"  {k}: {v}") for k,v in verify_sources().items()]
    ensure_config(deltas,args.sigma,args.overwrite)
    if not args.skip_guards: validate_guards()
    (ROOT/"runs").mkdir(parents=True,exist_ok=True)
    pending=[r for r in range(args.start,args.end) if args.overwrite or not (ROOT/"runs"/f"mc_{r:04d}"/"DONE.json").exists()]
    workers=max(1,min(args.workers,max(1,len(pending))))
    print("\nXMST-S3 controlled validation"); print(f"Runs {args.start}..{args.end-1}; workers={workers}"); print(f"sigma={args.sigma:g} km/s; deltas={','.join(f'{d:g}' for d in deltas)} km/s"); print("Tests: full-pipeline null + targeted Stage1-overlap S3 recovery curve"); print(f"Already complete: {args.end-args.start-len(pending)}/{args.end-args.start}")
    if workers==1:
        for i,r in enumerate(pending,1):
            d=run_one(r,deltas,args.sigma,args.overwrite); print(f"[{i:04d}/{len(pending):04d}] mc_{r:04d} S2splits={d['s2_accepted_splits']} elapsed={d['elapsed_seconds']:.1f}s",flush=True)
    else:
        for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(k,'1')
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as ex:
            futs={ex.submit(run_one,r,deltas,args.sigma,args.overwrite):r for r in pending}; n=0
            for f in as_completed(futs):
                r=futs[f]; d=f.result(); n+=1; print(f"[{n:04d}/{len(pending):04d}] mc_{r:04d} S2splits={d['s2_accepted_splits']} elapsed={d['elapsed_seconds']:.1f}s",flush=True)
    aggregate(args.start,args.end,deltas,args.sigma)
    print("\nDONE"); print(f"Headline: {ROOT/'xmst_s3_mc_headline.csv'}"); print(f"Overlap curve: {ROOT/'xmst_s3_mc_overlap_curve.csv'}"); print(f"Null control: {ROOT/'xmst_s3_mc_null_control.csv'}"); print(f"Report: {ROOT/'XMST_S3_MC_REPORT.md'}")

if __name__=="__main__": raise SystemExit(main())

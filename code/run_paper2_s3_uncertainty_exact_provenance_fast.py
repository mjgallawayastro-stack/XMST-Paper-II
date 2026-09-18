#!/usr/bin/env python3
"""Exact-provenance Paper-II S3 uncertainty rerun, accelerated Stage-1 reconstruction.

The original Table-3 field generator, intrinsic velocity seed and frozen S3
routine are retained. Stage-1 membership is reconstructed from the archived
Table-3 fracture scale using fixed-radius connectivity. At a fixed Euclidean
threshold this is exactly the same single-linkage partition obtained by cutting
the Euclidean MST at that threshold. The zero-error output is compared
per-realisation against the archived Table-3 DeltaV=5 results.
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import run_xmst_s3_mc1000_same_fields as orig

_ARGS=None; _META=None; _ARCH=None

def parse_float_list(s):
    x=[float(v) for v in s.split(',') if v.strip()]
    if not x or any(v<0 for v in x): raise argparse.ArgumentTypeError('non-negative sigmas required')
    return x

def init_worker(args_dict):
    global _ARGS,_META,_ARCH
    _ARGS=argparse.Namespace(**args_dict)
    _META=pd.read_csv(_ARGS.stage1_metadata).set_index('realisation')
    ag=pd.read_csv(_ARGS.archived_grid)
    _ARCH=ag[(ag.stage=='Stage1→S3 overlap diagnostic') & np.isclose(ag.velocity_separation_kms,5.0)].set_index('realisation')
    orig.init_worker()

def threshold_groups(coords, cut, min_n=10):
    n=len(coords)
    pairs=cKDTree(coords).query_pairs(float(cut),output_type='ndarray')
    parent=np.arange(n,dtype=np.int32); size=np.ones(n,dtype=np.int32)
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for a,b in pairs:
        ra=find(int(a)); rb=find(int(b))
        if ra!=rb:
            if size[ra]<size[rb]: ra,rb=rb,ra
            parent[rb]=ra; size[ra]+=size[rb]
    roots=np.fromiter((find(i) for i in range(n)),dtype=np.int32,count=n)
    vals,cnts=np.unique(roots,return_counts=True)
    groups=[np.flatnonzero(roots==r) for r,c in zip(vals,cnts) if c>=min_n]
    groups.sort(key=lambda a:int(a.min()))
    return groups

def run_one(run):
    t=time.time()
    xyz,av,truth,placements,field_meta=orig.make_realisation(orig._FIELD_DF,orig._TEMPLATES,run)
    m=_META.loc[run]; fracture=float(m.stage1_fracture_scale_pc)
    stage1_groups=threshold_groups(xyz,fracture,10)
    if len(stage1_groups)!=int(m.stage1_groups):
        raise RuntimeError(f'run {run}: Stage1 group count {len(stage1_groups)} != archived {int(m.stage1_groups)}')
    _,s1_score=orig.score_groups(stage1_groups,truth,orig._LABELS,run,'stage1')
    df=orig.make_input(xyz,av,truth,run); cfg=orig.frozen_config(Path(_ARGS.output_dir)/f'mc_{run:04d}')
    noise,unit,theta=orig.velocity_base(truth,run,float(_ARGS.velocity_sigma))
    vtrue=orig.shifted_velocities(noise,unit,truth,orig._LABELS,float(_ARGS.delta_v))
    rng=np.random.default_rng(int(_ARGS.error_seed_base)+run); z=rng.normal(0,1,size=(len(truth),2))
    rows=[]
    for err in _ARGS.velocity_error_sigmas:
        err=float(err)
        status='already_separate_at_stage1' if (s1_score['t01_best_group_id'] is None or s1_score['t02_best_group_id'] is None or s1_score['t01_best_group_id'] != s1_score['t02_best_group_id']) else 'target_parent_tested'
        if err == 0.0:
            # The exact zero-error calculation is the archived Table-3 DeltaV=5
            # experiment itself. Reuse that original per-realisation result rather
            # than needlessly recomputing the same frozen S3 call.
            ar=_ARCH.loc[run]
            scvals=dict(mean_completeness=float(ar.mean_completeness),mean_purity=float(ar.mean_purity),mean_jaccard=float(ar.mean_jaccard),
                        strict_recoveries_of_8=int(ar.strict_recoveries_of_8),formal_overlap_resolved=bool(ar.formal_overlap_resolved),
                        overlap_best_groups_separate=bool(ar.overlap_best_groups_separate),mean_fragment_count=float(ar.mean_fragment_count))
            beh={'s3_passes':int(ar.s3_passes) if pd.notna(ar.s3_passes) else 0,'s3_significant_flags':int(ar.s3_significant_flags) if pd.notna(ar.s3_significant_flags) else 0,
                 's3_accepted_splits':int(ar.s3_accepted_splits) if pd.notna(ar.s3_accepted_splits) else 0,'s3_final_groups':int(ar.s3_final_groups) if pd.notna(ar.s3_final_groups) else len(stage1_groups),
                 's3_orphans':int(ar.s3_orphans) if pd.notna(ar.s3_orphans) else 0}
        else:
            v=vtrue+err*z
            groups,s3diag,_=orig.target_parent_s3(stage1_groups,s1_score,df,xyz,v[:,0],v[:,1],fracture,cfg)
            _,sc=orig.score_groups(groups,truth,orig._LABELS,run,'stage1_to_s3_overlap_diagnostic',float(_ARGS.delta_v))
            scvals=dict(mean_completeness=float(sc['mean_completeness']),mean_purity=float(sc['mean_purity']),mean_jaccard=float(sc['mean_jaccard']),
                        strict_recoveries_of_8=int(sc['templates_strict_recovery']),formal_overlap_resolved=bool(sc['overlap_pair_resolved']),
                        overlap_best_groups_separate=bool(sc['overlap_best_groups_separate']),mean_fragment_count=float(sc['mean_fragment_count']))
            if s3diag is None:
                beh={'s3_passes':0,'s3_significant_flags':0,'s3_accepted_splits':0,'s3_final_groups':len(stage1_groups),'s3_orphans':0}
            else:
                beh=orig.s3_behavior(s3diag)
        eff=float(np.hypot(float(_ARGS.velocity_sigma),err))
        rows.append(dict(realisation=run,condition='baseline' if err==0 else f'vobs_sigma_{err:g}',velocity_error_sigma_kms=err,
                         intrinsic_velocity_sigma_kms=float(_ARGS.velocity_sigma),effective_per_component_sigma_kms=eff,
                         delta_v_kms=float(_ARGS.delta_v),delta_v_over_effective_sigma=float(_ARGS.delta_v)/eff,
                         mc_seed_base=int(orig.MC_CFG['seed']),velocity_seed_offset=30000000,
                         intrinsic_velocity_seed=int(orig.MC_CFG['seed'])+run+30000000,
                         observational_error_seed=int(_ARGS.error_seed_base)+run,velocity_direction_angle_rad=float(theta),
                         stage1_fracture_scale_pc=fracture,stage1_groups=len(stage1_groups),target_parent_status=status,
                         **scvals,**beh))
    return run,time.time()-t,rows

def aggregate(df):
    rows=[]
    for (cond,e),g in df.groupby(['condition','velocity_error_sigma_kms'],sort=False):
        eff=float(g.effective_per_component_sigma_kms.iloc[0])
        rows.append(dict(condition=cond,velocity_error_sigma_kms=float(e),n_runs=len(g),intrinsic_velocity_sigma_kms=float(g.intrinsic_velocity_sigma_kms.iloc[0]),
                         effective_per_component_sigma_kms=eff,delta_v_kms=float(g.delta_v_kms.iloc[0]),delta_v_over_effective_sigma=float(g.delta_v_over_effective_sigma.iloc[0]),
                         mean_completeness=float(g.mean_completeness.mean()),mean_purity=float(g.mean_purity.mean()),mean_jaccard=float(g.mean_jaccard.mean()),
                         mean_strict_recoveries_of_8=float(g.strict_recoveries_of_8.mean()),formal_overlap_resolution_rate=float(g.formal_overlap_resolved.mean()),
                         best_groups_separate_rate=float(g.overlap_best_groups_separate.mean()),mean_s3_accepted_splits=float(g.s3_accepted_splits.mean()),
                         mean_s3_orphans=float(g.s3_orphans.mean()),n_already_separate_at_stage1=int((g.target_parent_status=='already_separate_at_stage1').sum())))
    out=pd.DataFrame(rows).sort_values('velocity_error_sigma_kms').reset_index(drop=True); b=out.iloc[0]
    for c in ['mean_completeness','mean_purity','mean_jaccard','mean_strict_recoveries_of_8','formal_overlap_resolution_rate','best_groups_separate_rate','mean_s3_accepted_splits']:
        out[f'delta_{c}_vs_baseline']=out[c]-float(b[c])
    return out

def compare_archive(df,path):
    a=pd.read_csv(path); a=a[(a.stage=='Stage1→S3 overlap diagnostic')&np.isclose(a.velocity_separation_kms,5)].sort_values('realisation').reset_index(drop=True)
    b=df[np.isclose(df.velocity_error_sigma_kms,0)].sort_values('realisation').reset_index(drop=True)
    cols=['mean_completeness','mean_purity','mean_jaccard','strict_recoveries_of_8','formal_overlap_resolved','overlap_best_groups_separate','s3_accepted_splits','s3_orphans']
    details={}; ok=len(a)==len(b)
    if ok:
      for c in cols:
        x=pd.to_numeric(a[c],errors='coerce').to_numpy(float); y=pd.to_numeric(b[c],errors='coerce').to_numpy(float)
        same=bool(np.allclose(x,y,rtol=0,atol=1e-12,equal_nan=True)); ok &= same
        details[c]={'match':same,'max_abs_diff':float(np.nanmax(np.abs(x-y)))}
    return {'match':bool(ok),'archived_rows':len(a),'rerun_rows':len(b),'metrics':details}

def bootstrap(df,nboot=20000,seed=12345):
    rng=np.random.default_rng(seed); base=df[df.velocity_error_sigma_kms==0][['realisation','mean_jaccard']].rename(columns={'mean_jaccard':'j0'}); rows=[]
    for e in sorted(x for x in df.velocity_error_sigma_kms.unique() if x>0):
      g=df[np.isclose(df.velocity_error_sigma_kms,e)][['realisation','mean_jaccard']].merge(base,on='realisation'); d=(g.mean_jaccard-g.j0).to_numpy(); n=len(d); vals=[]
      for _ in range((nboot+999)//1000):
        k=min(1000,nboot-len(vals)*1000)
        if k<=0: break
        idx=rng.integers(0,n,size=(k,n)); vals.append(d[idx].mean(axis=1))
      v=np.concatenate(vals); lo,hi=np.quantile(v,[.025,.975]); rows.append(dict(velocity_error_sigma_kms=float(e),mean_delta_jaccard=float(d.mean()),bootstrap_95_ci_low=float(lo),bootstrap_95_ci_high=float(hi),n_pairs=n,bootstrap_resamples=nboot,bootstrap_seed=seed))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(); here=Path(__file__).resolve().parent
    p.add_argument('--output-dir',type=Path,default=here/'s3_uncertainty_exact_provenance_1000')
    p.add_argument('--stage1-metadata',type=Path,required=True); p.add_argument('--archived-grid',type=Path,required=True)
    p.add_argument('--start',type=int,default=0); p.add_argument('--end',type=int,default=1000); p.add_argument('--workers',type=int,default=min(5,os.cpu_count() or 2))
    p.add_argument('--delta-v',type=float,default=5.0); p.add_argument('--velocity-sigma',type=float,default=1.0); p.add_argument('--error-seed-base',type=int,default=20260831)
    p.add_argument('--velocity-error-sigmas',type=parse_float_list,default=parse_float_list('0,0.25,0.5,1,2'))
    a=p.parse_args(); a.velocity_error_sigmas=list(dict.fromkeys([0.0]+a.velocity_error_sigmas)); a.output_dir.mkdir(parents=True,exist_ok=True)
    hashes=orig.verify_sources(); d=vars(a).copy(); d['output_dir']=str(a.output_dir); d['stage1_metadata']=str(a.stage1_metadata); d['archived_grid']=str(a.archived_grid)
    print('Exact-provenance S3 uncertainty rerun (archived Stage-1 fracture)',flush=True); print('runs',a.start,a.end,'workers',a.workers,'errors',a.velocity_error_sigmas,flush=True)
    rows=[]; runs=list(range(a.start,a.end)); t=time.time()
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker,initargs=(d,)) as ex:
      futs={ex.submit(run_one,r):r for r in runs}; done=0
      for f in as_completed(futs):
        r,el,rr=f.result(); rows.extend(rr); done+=1
        if done%25==0 or done==len(runs): print(f'[{done}/{len(runs)}] elapsed={time.time()-t:.1f}s',flush=True)
    allr=pd.DataFrame(rows).sort_values(['realisation','velocity_error_sigma_kms']).reset_index(drop=True); agg=aggregate(allr); comp=compare_archive(allr,a.archived_grid); ci=bootstrap(allr)
    allr.to_csv(a.output_dir/'s3_uncertainty_exact_provenance_all_runs.csv',index=False); agg.to_csv(a.output_dir/'s3_uncertainty_exact_provenance_aggregate.csv',index=False); ci.to_csv(a.output_dir/'s3_uncertainty_exact_provenance_paired_bootstrap_ci.csv',index=False)
    prov={'mc_seed_base':int(orig.MC_CFG['seed']),'velocity_seed_offset':30000000,'delta_v_kms':a.delta_v,'intrinsic_velocity_sigma_kms_per_component':a.velocity_sigma,'observational_error_seed_base':a.error_seed_base,'velocity_error_sigmas_kms_per_component':a.velocity_error_sigmas,'stage1_reconstruction':'archived fracture scale + Euclidean fixed-threshold connectivity (single-linkage equivalent to cutting MST)','s3_scope':'only Stage-1 T01/T02 parent; unchanged if already separate','frozen_source_hashes':hashes,'table3_baseline_comparison':comp}
    (a.output_dir/'PROVENANCE_CHECK.json').write_text(json.dumps(prov,indent=2)); (a.output_dir/'s3_uncertainty_exact_provenance_aggregate.json').write_text(json.dumps(agg.to_dict(orient='records'),indent=2))
    print('\nAGGREGATE\n'+agg.to_string(index=False)); print('\nCI\n'+ci.to_string(index=False)); print('\nPROVENANCE\n'+json.dumps(prov,indent=2))
if __name__=='__main__': main()

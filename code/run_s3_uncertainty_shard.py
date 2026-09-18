#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import pandas as pd
import run_paper2_s3_uncertainty_exact_provenance_fast as h

p=argparse.ArgumentParser()
p.add_argument('--shard',type=int,required=True)
p.add_argument('--n-shards',type=int,default=5)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--stage1-metadata',type=Path,required=True)
p.add_argument('--archived-grid',type=Path,required=True)
p.add_argument('--error-seed-base',type=int,default=20260831)
a=p.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=True)
args_dict={
 'output_dir':str(a.output_dir),
 'stage1_metadata':str(a.stage1_metadata),
 'archived_grid':str(a.archived_grid),
 'delta_v':5.0,'velocity_sigma':1.0,'error_seed_base':a.error_seed_base,
 'velocity_error_sigmas':[0.0,0.25,0.5,1.0,2.0]
}
h.init_worker(args_dict)
rows=[]; runs=list(range(a.shard,1000,a.n_shards)); t=time.time()
for i,r in enumerate(runs,1):
    _,el,rr=h.run_one(r); rows.extend(rr)
    if i%10==0 or i==len(runs):
        pd.DataFrame(rows).to_csv(a.output_dir/f'shard_{a.shard}_partial.csv',index=False)
        print(f'shard {a.shard}: {i}/{len(runs)} last={r} elapsed={time.time()-t:.1f}s',flush=True)
pd.DataFrame(rows).sort_values(['realisation','velocity_error_sigma_kms']).to_csv(a.output_dir/f'shard_{a.shard}.csv',index=False)
print(f'shard {a.shard}: DONE {len(runs)} runs in {time.time()-t:.1f}s',flush=True)

#!/usr/bin/env python3
"""Reconstruct published TRAIN-development tables from aggregate evidence.

No model execution, new evaluation, threshold optimization, or case-level data.
P5 is a previously inspected threshold-selection population. Rolling-fold
results are descriptive; subsystem targets are retrospective, not predictors.
Runtime is an archived cached-input benchmark, not a new measurement.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

MODELS = [('primary_soft_full','Teacher--Student (soft)'),('logit_hard','XGBoost (19+logit)'),('full_hard','XGBoost full hard'),('pca48_hard','XGBoost PCA48'),('standalone_transformer','Standalone Transformer'),('sequence_lstm','Sequence LSTM'),('catboost_observed19_hard','Observed-only CatBoost'),('snapshot_lstm','Snapshot LSTM'),('observed_only_hard','Observed-only XGBoost')]
SEEDS = ['snapshot_lstm','sequence_lstm','standalone_transformer','primary_soft_full']
METRICS = [('ap','AP'),('auroc','AUROC'),('brier','Brier'),('fixed_primary_f1','Primary F1'),('fixed_D5_f1','D5 F1')]
HORIZONS = ['0','1','5','10','15']
GROUPS = ['SLS','FTS','ZLS']

def div(n,d): return n/d if d else 0.0

def prf(c):
    tp,fp,fn = (c[k] for k in ('tp','fp','fn'))
    if any(not isinstance(v,int) or v<0 for v in (tp,fp,fn)): raise ValueError('invalid confusion counts')
    p,r=div(tp,tp+fp),div(tp,tp+fn)
    return div(2*p*r,p+r),p,r

def reconstruct(d):
    if d['schema']!='train_development_aggregates_v1': raise ValueError('unsupported schema')
    p5=[]; horizon=[]
    for key,label in MODELS:
        m=d['p5'][key]; c=m['counts']; f,p,r=prf(c)
        if sum(c.values())!=6202 or c['tp']+c['fn']!=1661: raise ValueError('P5 cohort mismatch')
        p5.append([label,*[f'{x:.3f}' for x in (f,p,r,m['ap'],m['auroc'])],f"{m['threshold']:.2f}"])
        m=d['development_horizons'][key]
        if not 0<=m['clean_alerts']<=2217: raise ValueError('invalid clean alerts')
        vals=[]
        for h in HORIZONS:
            c=m['horizons'][h]
            if c['tp']+c['fn']!=1395: raise ValueError('development delayed denominator mismatch')
            vals.append(f'{prf(c)[0]:.3f}')
        horizon.append([label,*vals,str(m['clean_alerts'])])
    seeds=[]
    for k,label in METRICS:
        row=[label]
        for arm in SEEDS:
            runs=d['seed_runs'][arm]
            if sorted(int(s) for s in runs)!=[42,43,44]: raise ValueError('seed mismatch')
            x=[runs[str(s)][k] for s in (42,43,44)]
            if any(not math.isfinite(y) or not 0<=y<=1 for y in x): raise ValueError('invalid seed metric')
            row.append(f'{mean(x):.4f} ± {stdev(x):.4f}')
        seeds.append(row)
    stages=[('prefix_construction','Prefix packing'),('teacher_h2d_forward_d2h','Transformer and device transfers'),('pca_matrix_assembly','PCA48 and feature assembly'),('xgb_plus_platt','XGBoost and Platt calibration'),('median_ms_per_query','Full cached-input scoring loop')]
    runtime=[[name,*[f'{d["runtime"][b][k]:.4f}' for b in ('1','32')]] for k,name in stages]
    tol=[]
    for s in ('0','30','60','300'):
        m=d['tolerance'][s]; n=m['same']; u=m['unknown']
        if n+u!=1395 or min(n,u)<0: raise ValueError('tolerance counts do not conserve cases')
        tol.append([s,str(n),f'{100*n/1395:.2f}%',str(u)])
    target=[]
    for g in GROUPS:
        m=d['target_subsystem'][g]; n=m['cases']; counts=[m[k] for k in ('d5_correct','d5_opportunity','d10_correct','d10_opportunity','xgb_d5_correct')]
        if not (0<=counts[0]<=counts[1]<=n and 0<=counts[2]<=counts[3]<=n and 0<=counts[4]<=n): raise ValueError('subsystem counts inconsistent')
        target.append([g,str(n),*[f'{x}/{n}' for x in counts]])
    if sum(d['target_subsystem'][g]['cases'] for g in GROUPS)!=1395: raise ValueError('target denominator mismatch')
    return {
        'p5':(['Model','F1','Precision','Recall','AP','AUROC','threshold'],p5),
        'development_horizons':(['Model','D0','D1','D5','D10','D15','Clean alerts'],horizon),
        'seed_sensitivity':(['Metric','Snapshot LSTM','Sequence LSTM','Transformer','Soft reference'],seeds),
        'runtime':(['Stage','Batch 1','Batch 32'],runtime),
        'tolerance':(['Tolerance (s)','Same first delay','Share','Later status unknown'],tol),
        'target_subsystem':(['Target','Delayed','D5 correct','D5 opportunity','D10 correct','D10 opportunity','XGB D5'],target),
    }

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--aggregates',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    tables=reconstruct(json.loads(a.aggregates.read_text(encoding='utf-8')))
    a.out_dir.mkdir(parents=True,exist_ok=False)
    for name,(header,rows) in tables.items():
        with (a.out_dir/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(rows)
    (a.out_dir/'summary.json').write_text(json.dumps({'status':'ARCHIVED_DEVELOPMENT_TABLES_GENERATED','tables':{k:len(v[1]) for k,v in tables.items()},'training_performed':False,'new_model_inference':False,'independent_evaluation':False},indent=2)+'\n')

if __name__=='__main__':main()

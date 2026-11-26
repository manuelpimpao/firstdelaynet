#!/usr/bin/env python3
"""Verify frozen historical replay -> outcome join -> statistical evaluation.
Run outside the submission root with trusted fingerprint-pinned assets only.
No training, recalibration, threshold selection or original case order required.
The evaluation is retrospective, not a new independent holdout. Outputs are aggregates only.
"""
from __future__ import annotations
import argparse, hashlib, json, stat, sys
from pathlib import Path
sys.dont_write_bytecode=True
MANIFEST_PIN='3f20cad33f21d5bbe8d506f032da4fd7dd0f664db2994dfe1f7eec3700122712'
DOC_PINS={
 'README.md':'2ce99f33de3b6a72ad4b1fbe21cedfc2e138a32b5be8f179bdf88ec14d3f9e06',
 'docs/DATA_DICTIONARY.md':'6afd5d63854af5eac3230c83ebd703718fc53993235fd87ca8a729e0f2ac5c1c',
 'docs/FEATURE_DICTIONARY.md':'2bb1fd9dd2923b22992eca5c7ca9b73f53db7fcbea0c2c039e6629601a94a419',
 'docs/REPRODUCTION_SCOPE.md':'cf43bbce6c6cb1e4bb5e57890626ee3f116b0812d4855f84ae05aad9201dfc8a'}
DOC_PINS['docs/END_TO_END.md']='480c80cab2f7eb303d0baf247d7a80d004bc4a4254b9c1f9e4eb4c3d4da22778'
METRIC_TOL=1e-6 # existing V4.473 frozen metric tolerance, not probability tolerance

def need(x,msg):
    if not bool(x): raise ValueError(msg)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def unsafe(p):
    s=p.lstat()
    return p.is_symlink() or bool(getattr(s,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024))

def safe_dir(p):
    need(p.is_dir() and not unsafe(p),'UNSAFE_OR_MISSING_DIRECTORY')

def checkfile(p,pin):
    need(p.is_file() and not unsafe(p) and digest(p)==pin,'MISSING_UNSAFE_OR_MISMATCHED_PINNED_FILE')

def inventories(root):
    safe_dir(root)
    for k in ('code','models','data','docs'): safe_dir(root/k)
    mp=root/'code'/'candidate_manifest.json'; checkfile(mp,MANIFEST_PIN)
    m=json.loads(mp.read_text(encoding='utf8'))
    groups={'code':dict(m['code_files']),'models':m['bundle_files'],'data':m['processed_data_files']}
    groups['code']['candidate_manifest.json']=MANIFEST_PIN
    need([len(groups[k]) for k in ('code','models','data')]==[20,26,10],'UNEXPECTED_MANIFEST_COUNTS')
    # The external audit pins this installed script; this self-digest detects only changes during execution.
    groups['code']['evaluate.py']=digest(Path(__file__))
    for k,v in groups.items():
        for n,h in v.items():
            need(Path(n).name==n and '/' not in n and '\\' not in n and ':' not in n and n not in ('.','..'),'INVALID_ASSET_NAME')
            checkfile(root/k/n,h)
    for n,h in DOC_PINS.items(): checkfile(root/n,h)
    for n in ('historical_test_events.parquet','historical_test_queries.parquet'):
        need(groups['models'][n]==groups['data'][n],'DUPLICATE_DATA_PINS_DIFFER')
    allow={k:set(v) for k,v in groups.items()}
    allow['root']={'code','models','data','docs','README.md'}
    allow['docs']={Path(n).name for n in DOC_PINS if n.startswith('docs/')}
    counts={}
    for k,names in allow.items():
        d=root if k=='root' else root/k
        extra=[x for x in d.iterdir() if x.name not in names];counts[k]=len(extra)
        for x in extra: print('LOCAL ONLY — excluded from allowlist:',k,ascii(x.name))
    return groups,counts

def connect(keys,y,scores,queries,events,stats,replay):
    import pandas as pd
    import numpy as np
    need(list(keys.columns)==stats.KEYS and not keys.duplicated().any(),'DUPLICATE_OR_INVALID_REPLAY_KEY')
    need(set(scores)==set(stats.ORDER) and len(y)==len(keys),'SCORE_AXIS_MISMATCH')
    u=replay.distinct_queries(replay.normalize_tau(queries))
    # Merge the endpoint independently from the query export, then cross-check
    # both endpoint and target against the transport event log in align_scores.
    z=keys.merge(u[stats.KEYS+['t_next','y_next_delay']],on=stats.KEYS,how='left',sort=False,validate='one_to_one',indicator=True)
    need(len(z)==len(keys) and z[stats.KEYS].equals(keys) and z['_merge'].eq('both').all(),'ENDPOINT_KEY_ALIGNMENT_FAILED')
    need(np.array_equal(z.y_next_delay.to_numpy(),np.asarray(y)),'REPLAY_QUERY_LABEL_MISMATCH')
    z['next_static_delay']=y
    for arm,p in scores.items():
        need(np.asarray(p).shape==(len(z),),'SCORE_SHAPE_MISMATCH')
        z['prob_'+arm]=p
    return stats.align_scores(z,events)

def evaluate_comparison(report,ref,stats,tables):
    import numpy as np
    need(report['cohort']==ref['cohort'],'COHORT_MISMATCH')
    # Keep the original strict 1e-10 comparison; native-order bootstrap and
    # numerical differences are reported, never silently relabelled exact.
    comparison=stats.compare(report,ref)
    per={}
    for arm in tables.ORDER:
        a,b=report['models'][arm],ref['models'][arm]
        errors=[abs(a['primary'][k]-b['primary'][k]) for k in tables.METRICS]
        errors += [abs(a['case_equal'][k]-b['case_equal'][k]) for k in ('precision','recall','F1')]
        errors += [abs(a[k][j]-b[k][j]) for k in ('DL','AC') for j in ('precision','recall','f1')]
        errors += [abs(a['calibration']['ECE_equal_width_10']-b['calibration']['ECE_equal_width_10'])]
        discrete=(a['primary']['confusion']==b['primary']['confusion'] and all(a[k][j]==b[k][j] for k in ('DL','AC') for j in ('tp','fp','fn','tn')))
        for mode in ('primary','D5'):
            for h in tables.HORIZONS:
                x,z=a['horizons'][mode][h],b['horizons'][mode][h]
                discrete=discrete and all(x[k]==z[k] for k in ('tp','fp','fn'))
                errors.extend(abs(x[k]-z[k]) for k in ('precision','recall','f1'))
        per[arm]={'max_abs_metric_difference':float(max(errors)),'core_metrics_within_existing_1e_6':bool(max(errors)<=METRIC_TOL),'primary_and_first_alarm_and_DL_AC_counts_exact':bool(discrete),'fixed_thresholds_exact':a['thresholds']==b['thresholds']}
    return comparison,per

def load_modules(code):
    sys.path.insert(0,str(code.resolve()))
    import replay, retrospective_statistics as stats, retrospective_tables as tables
    for m in (replay,stats,tables): need(Path(m.__file__).resolve().parent==code.resolve(),'UNEXPECTED_MODULE_ORIGIN')
    return replay,stats,tables

def selftest(code):
    replay,stats,tables=load_modules(code)
    import pandas as pd
    import numpy as np
    # Two small first-delay histories, no private data.
    t=pd.Timestamp('2020-01-01',tz='UTC'); events=[];queries=[]
    for i in range(2):
        c=f'c{i}';t0=t+pd.Timedelta(hours=i);end=t0+pd.Timedelta(minutes=20)
        for j,ts in enumerate((t0,end)):
            events.append({'case_seg':c,'time:timestamp':ts,'category':'transport','is_delayed':i*j,'is_delayed_dyn':i*j,'planned:timestamp':ts-pd.Timedelta(minutes=2)})
        queries.append({'case_seg':c,'k_idx':0,'tau_ts':t0+pd.Timedelta(minutes=5),'t0':t0,'t_next':end,'y_next_delay':i,'has_post':1,'events_since_t0':1,'station':'s','time_since_t0_s':300})
    q=pd.DataFrame(queries);e=pd.DataFrame(events);q=replay.normalize_tau(q)
    keys=q[stats.KEYS].copy();y=np.array([0,1]);p={a:np.array([.1,.9]) for a in stats.ORDER}
    f=connect(keys,y,p,q,e,stats,replay);need(len(f)==2,'SYNTHETIC_JOIN')
    for kind in ('wrong_label','wrong_endpoint','duplicate_keys'):
        bad=q.copy(); kk=keys.copy()
        if kind=='wrong_label':bad.loc[0,'y_next_delay']=1
        elif kind=='wrong_endpoint':bad.loc[0,'t_next']+=pd.Timedelta(seconds=1)
        else:kk=pd.concat([kk,kk.iloc[:1]],ignore_index=True)
        try:connect(kk,y,p,bad,e,stats,replay)
        except (ValueError,replay.Blocked):pass
        else:raise AssertionError('FAILED_TO_REJECT_'+kind)
    # Exercise metric comparison on the reviewed aggregate file, no models.
    ref=json.loads((code/'retrospective_aggregates.json').read_text(encoding='utf8'));tables.tables(ref)
    comp,per=evaluate_comparison(ref,ref,stats,tables)
    need(all(v['mismatches']==0 for v in comp.values()) and all(v['core_metrics_within_existing_1e_6'] and v['primary_and_first_alarm_and_DL_AC_counts_exact'] for v in per.values()),'COMPARATOR_SELFTEST')
    stats.self_test()
    print('SYNTHETIC_ADAPTER_AND_COMPARATOR_PASS')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submission-root',type=Path);p.add_argument('--out-dir',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--acknowledge-retrospective',action='store_true');p.add_argument('--self-test-code-dir',type=Path)
    a=p.parse_args()
    if a.self_test_code_dir:selftest(a.self_test_code_dir);return
    need(a.submission_root and a.out_dir and a.acknowledge_retrospective,'EXPLICIT_PATHS_AND_SCOPE_REQUIRED')
    root=a.submission_root;out=a.out_dir
    need(not out.exists() and not out.is_symlink() and not out.resolve().is_relative_to(root.resolve()),'OUTPUT_MUST_BE_NEW_AND_EXTERNAL')
    groups,extra=inventories(root)
    need(extra['code']==0 and extra['models']==0,'UNEXPECTED_CODE_OR_MODEL_ENTRY_REVIEW_BEFORE_EXECUTION')
    print('62 listed files hash-verified (adapter self-digest is not authentication). Beginning frozen inference.',flush=True)
    replay,stats,tables=load_modules(root/'code')
    import torch,pandas as pd
    need(a.device!='cuda' or torch.cuda.is_available(),'CUDA_UNAVAILABLE')
    ref=json.loads((root/'code'/'retrospective_aggregates.json').read_text(encoding='utf8'));tables.tables(ref)
    keys,y,scores,metrics=replay.run(root/'models',torch.device(a.device))
    print('Frozen replay complete. Checking outcome join and recomputing matched case statistics.',flush=True)
    # Use the data directory's independently pinned identical copies for the outcome join.
    queries=pd.read_parquet(root/'data'/'historical_test_queries.parquet',columns=replay.TAU_COLS)
    events=pd.read_parquet(root/'data'/'historical_test_events.parquet',columns=stats.EVENT_COLS)
    f=connect(keys,y,scores,queries,events,stats,replay)
    report,valid=stats.compute(f);tables.tables(report)
    comparison,per=evaluate_comparison(report,ref,stats,tables)
    after,extra_after=inventories(root);need(after==groups and extra_after==extra,'INVENTORY_CHANGED')
    ok=valid==200 and all(all(v[k] for k in ('core_metrics_within_existing_1e_6','primary_and_first_alarm_and_DL_AC_counts_exact','fixed_thresholds_exact')) for v in per.values())
    summary={'version':'1.0','status':'FROZEN_REPLAY_TO_STATISTICS_CORE_PARITY_PASS' if ok else 'FROZEN_REPLAY_TO_STATISTICS_REVIEW_REQUIRED','scope':'retrospective previously exposed historical TEST, native pseudonym order','expected_files_matched':62,'unexpected_entry_counts':extra,'all_input_pins_unchanged':True,'cohort':report['cohort'],'core_metric_tolerance':METRIC_TOL,'strict_comparison_tolerance':1e-10,'model_comparison':per,'strict_aggregate_comparison':comparison,'bootstrap_valid_draws':valid,'bootstrap_original_order_used':False,'exact_archived_bootstrap_claimed':False,'row_level_output_written':False,'training_performed':False,'threshold_search':False,'independent_holdout':False,'confidentiality_certified':False,'release_authorization_claimed':False,'device':a.device,'torch':str(torch.__version__)}
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (out/'computed_aggregates.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(summary['status'])
    if not ok:raise SystemExit(2)
if __name__=='__main__': main()

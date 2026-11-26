#!/usr/bin/env python3
"""Regenerate historical evaluation tables from reviewed aggregate evidence.
No case-level data, training, model loading, or new bootstrap computation.
The historical TEST export was previously exposed; this is descriptive replay.
"""
import argparse
import csv
import json
import math
from pathlib import Path

ORDER = ('primary_soft_full', 'full_hard', 'pca48_hard', 'logit_hard',
         'standalone_transformer', 'sequence_lstm', 'observed_only_hard',
         'catboost_observed19_hard', 'snapshot_lstm')
HORIZONS = ('0', '1', '5', '10', '15')
METRICS = ('precision', 'recall', 'F1', 'average_precision', 'AUROC', 'Brier')
THRESHOLDS = ((.24,.35),(.28,.47),(.27,.44),(.31,.38),(.23,.41),(.26,.42),(.36,.77),(.30,.76),(.36,.67))

def need(test, msg):
    if not test:
        raise ValueError(msg)

def near(x, y):
    return math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10)

def ratio(a,b):
    return a/b if b else 0.0

def prf(c, m, f1_key):
    tp,fp,fn = (c[k] for k in ('tp','fp','fn'))
    need(all(math.isfinite(x) and x>=0 for x in (tp,fp,fn)), 'Invalid counts')
    for k,v in (('precision',ratio(tp,tp+fp)),('recall',ratio(tp,tp+fn)),(f1_key,ratio(2*tp,2*tp+fp+fn))):
        need(near(m[k],v), 'Confusion/metric inconsistency: '+k)

def tables(d):
    need(d['study_role']=='retrospective_previously_exposed_test_not_independent','Wrong study role')
    need(d['cohort']=={'queries':39775,'cases':7146,'positive_queries':5410,'delayed_cases':3423},'Wrong cohort')
    need(d['bootstrap']=={'draws':200,'seed':409,'unit':'case','matched':True},'Wrong bootstrap settings')
    need(set(d['models'])==set(ORDER),'Wrong model inventory')
    rows={k:[] for k in ('primary','secondary','horizons','paired','reliability','horizon_details','model_intervals','paired_intervals','opportunity')}
    for arm,threshold in zip(ORDER,THRESHOLDS):
        m=d['models'][arm]; p=m['primary']; c=p['confusion']
        need(tuple(m['thresholds'])==threshold,'Fixed threshold mismatch')
        need(sum(c.values())==39775 and c['tp']+c['fn']==5410,'Primary cohort mismatch')
        prf(c,p,'F1'); need(all(0<=p[k]<=1 for k in METRICS),'Metric outside unit interval')
        rows['primary'].append([arm,threshold[0],*[p[k] for k in METRICS]])
        w=m['case_equal']; prf(w['weighted_confusion'],w,'F1')
        need(near(sum(w['weighted_confusion'].values()),7146),'Case-equal mass mismatch')
        for k,n in [('DL',13836),('AC',37788)]:
            b=m[k]; prf(b,b,'f1'); need(sum(b[c] for c in ('tp','fp','fn','tn'))==n and b['tp']+b['fn']==3423,'Sensitivity cohort mismatch')
        rows['secondary'].append([arm,w['F1'],m['DL']['f1'],m['AC']['f1']])
        rows['horizons'].append([arm,threshold[1],*[m['horizons']['D5'][h]['f1'] for h in HORIZONS]])
        for mode in ('primary','D5'):
            for h in HORIZONS:
                a=m['horizons'][mode][h]; prf(a,a,'f1'); need(a['tp']+a['fn']==3423,'Delayed case denominator mismatch')
                rows['horizon_details'].append([arm,mode,int(h),*[a[k] for k in ('tp','fp','fn','precision','recall','f1')]])
        bins=m['calibration']['bins']; need(len(bins)==10 and sum(b['count'] for b in bins)==39775,'Calibration bins mismatch')
        ece=0.
        for i,b in enumerate(bins):
            need(b['bin']==i and b['count']>=0 and near(b['left_inclusive'],i/10) and near(b['right_inclusive_except_last'],(i+1)/10),'Bin definition mismatch')
            if b['count']:
                need(0<=b['mean_probability']<=1 and 0<=b['observed_positive_rate']<=1,'Invalid probability')
                ece+=b['count']/39775*abs(b['mean_probability']-b['observed_positive_rate'])
            rows['reliability'].append([arm,i,b['count'],b['mean_probability'],b['observed_positive_rate'],m['calibration']['ECE_equal_width_10']])
        need(near(ece,m['calibration']['ECE_equal_width_10']),'ECE mismatch')
        for k,v in m['intervals'].items():
            need(len(v)==2 and all(math.isfinite(x) for x in v) and 0<=v[0]<=v[1]<=1,'Invalid model interval')
            rows['model_intervals'].append([arm,k,*v])
        if arm!=ORDER[0]:
            q=d['models'][ORDER[0]]['primary']; a=m['paired_intervals']
            for k,v in a.items():
                lo,hi=v['percentile_95pct']; need(-1<=lo<=hi<=1 and math.isfinite(v['mean_bootstrap_difference']),'Invalid paired interval')
                rows['paired_intervals'].append([arm,k,lo,hi,v['mean_bootstrap_difference']])
            rows['paired'].append([arm,q['F1']-p['F1'],*a['primary_F1_reference_minus_comparator']['percentile_95pct'],q['average_precision']-p['average_precision'],*a['primary_average_precision_reference_minus_comparator']['percentile_95pct']])
    for h in HORIZONS:
        a=d['opportunity'][h]; need(a['eligible_delayed_cases']==3423 and near(a['fraction'],a['cases_with_observation_opportunity_at_horizon']/3423),'Opportunity mismatch')
        rows['opportunity'].append([int(h),a['eligible_delayed_cases'],a['cases_with_observation_opportunity_at_horizon'],a['fraction']])
    return rows

HEADERS={
 'primary':['model','threshold',*METRICS],
 'secondary':['model','case_equal_F1','DL_F1','AC_F1'],
 'horizons':['model','D5_threshold',*['D'+h for h in HORIZONS]],
 'paired':['comparator','delta_F1','F1_low','F1_high','delta_AP','AP_low','AP_high'],
 'reliability':['model','bin','count','mean_probability','observed_positive_rate','ECE10'],
 'horizon_details':['model','threshold_mode','horizon_minutes','tp','fp','fn','precision','recall','f1'],
 'model_intervals':['model','metric','lower','upper'],
 'paired_intervals':['comparator','metric','lower','upper','bootstrap_mean_difference'],
 'opportunity':['horizon_minutes','delayed_cases','cases_with_opportunity','fraction']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aggregates',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    rows=tables(json.loads(a.aggregates.read_text(encoding='utf-8')))
    a.out_dir.mkdir(parents=True,exist_ok=False)
    for k,rs in rows.items():
        with (a.out_dir/(k+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(HEADERS[k]);w.writerows(rs)
    print(json.dumps({'status':'ARCHIVED_AGGREGATE_TABLES_PASS','rows':{k:len(v) for k,v in rows.items()},'new_bootstrap':False,'new_inference':False,'independent_holdout':False},indent=2))
if __name__=='__main__':main()

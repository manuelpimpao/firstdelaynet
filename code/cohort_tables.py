#!/usr/bin/env python3
"""Reconstruct archived cohort counts, fixed thresholds and initial-subsystem table.
Aggregate-only reproduction: does not read logs, fit models or select thresholds.
The cohorts have distinct units and selection rules and must not be pooled.
"""
import argparse
import csv
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def f1(c):
    tp, fp, fn = (c[k] for k in ('tp', 'fp', 'fn'))
    require(all(isinstance(x, int) and x >= 0 for x in (tp, fp, fn)), 'Invalid confusion counts')
    return 2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.0


def reconstruct(d):
    require(d['schema']=='cohort_aggregates_v1','Invalid aggregate schema')
    initial=[]
    for g, a in d['initial_subsystems'].items():
        n, delayed, clean, alerts = (a[k] for k in ('cases','delayed','clean','clean_alerts'))
        require(n==delayed+clean and 0<=alerts<=clean and a['d5']['tp']+a['d5']['fn']==delayed,'Invalid subgroup')
        require(0<=a['d10_f1']<=1,'Invalid D10 F1')
        initial.append([g,f'{n:,}',f'{delayed:,} ({100*delayed/n:.1f}%)',f'{f1(a["d5"]):.3f}',f'{a["d10_f1"]:.3f}',f'{alerts:,}/{clean:,}'])
    require(sum(a['cases'] for a in d['initial_subsystems'].values())==3612 and sum(a['delayed'] for a in d['initial_subsystems'].values())==1395,'Development cases do not conserve')
    thresholds=[]
    for model, t in d['thresholds'].items():
        require(len(t)==2 and all(0<=x<=1 for x in t),'Invalid fixed threshold')
        thresholds.append([model,*[f'{x:.2f}' for x in t]])
    require(len(thresholds)==9,'Expected nine models')
    census=[]
    for scope, parts in d['train_populations'].items():
        for clock, c in parts.items():
            require(c['rows']==c['positive']+c['negative'] and c['rows']>=c['cases_with_queries']>0,'Invalid population count')
            census.append([scope,clock,c['rows'],c['positive'],c['negative'],c['cases_with_queries'],f'{100*c["positive"]/c["rows"]:.2f}%'])
        require(parts['full_clock']['negative']==parts['predeadline']['negative'] and parts['full_clock']['positive']>=parts['predeadline']['positive'],'Invalid archived predeadline partition')
    for clock in ('full_clock','predeadline'):
        for k in ('rows','positive','negative','cases_with_queries'):
            require(sum(d['train_populations'][s][clock][k] for s in ('P0','P1','P2','P3','P4','P5','PURGED'))==d['train_populations']['outer_student_fit'][clock][k],'Student period conservation failed')
            require(sum(d['train_populations'][s][clock][k] for s in ('P0','P1','P2','P3'))==d['train_populations']['final_fit'][clock][k],'Final fit conservation failed')
    r=d['recognition']; require(r['cases']==r['delayed_cases']+r['observed_clean_cases'] and r['full_clock_rows']==r['strictly_predeadline_rows']+r['at_or_after_archived_deadline_rows'],'Recognition population mismatch')
    require(r['delayed_cases']==r['ever_archived_deadline_passed_delayed_cases']+r['delayed_cases_with_no_late_marker'],'Recognition delay conservation failed')
    require(r['ever_archived_deadline_passed_clean_cases']==0 and r['flagged_delayed_case_first_flag_on_other_interval']==0 and r['all_strict_advance_warning_TP_D0_D1_D5_D10_D15']==0,'Recognition scope mismatch')
    a=d['anchors'];require(a['observed_next']+a['unknown_next']==a['total'],'Anchor conservation failed')
    require(a['observed_cases']+a['unknown_cases']-a['overlap_cases']==a['candidate_cases'],'Anchor case union failed')
    t=d['historical_test'];require(t['historical_raw_query_rows']-t['historical_unique_query_keys']==t['raw_duplicate_rows_not_independent'],'Duplicate counts disagree')
    require(t['historical_unique_query_keys']-t['eligible_queries']==t['excluded_unique_queries'],'Test cohort conservation failed')
    p=d['revised_processed_train'];require(sum(p['category_counts'].values())==p['events'],'Event count mismatch')
    l=d['legacy_source_flow'];require(l['input_event_rows']-l['consecutive_duplicate_events_removed']==l['retained_event_rows'],'Source flow mismatch')
    legacy=[]
    for split, m in d['older_private_train_val'].items():
        require(sum(m[k] for k in ('zero_transport_cases','one_transport_cases','at_least_two_transport_cases'))==m['cases'],'Legacy case conservation failed')
        legacy.append([split,*[m[k] for k in ('cases','events','zero_transport_cases','one_transport_cases','at_least_two_transport_cases','tau_cases','tau_rows')]])
    bd=[]
    for name,c in d['localization'].items():
        n=sum(c.values());tp,fp,fn,tn=(c[k] for k in ('tp','fp','fn','tn'))
        bd.append([name,n,tp+fn,tp,fp,fn,tn,f'{tp/(tp+fp):.3f}',f'{tp/(tp+fn):.3f}',f'{f1(c):.3f}',f'{(tp+tn)/n:.3f}'])
    return {'initial_subsystem':(['Subsystem','Cases','Delayed cases','D5 F1','D10 F1','Clean alarms'],initial), 'thresholds':(['Model','Primary theta','Secondary D5 theta'],thresholds), 'train_population':(['Population','Clock','Rows','Positive','Negative','Cases','Positive fraction'],census), 'old_private_train_val':(['Split','Cases','Events','Zero transport','One transport','At least two transports','Eligible query cases','Eligible queries'],legacy), 'localization':(['Subset','Rows','Positive rows','TP','FP','FN','TN','Precision','Recall','F1','Accuracy'],bd), 'processed_train_categories':(['Category','Count'],list(p['category_counts'].items()))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--aggregates',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.aggregates.read_text(encoding='utf-8'));tables=reconstruct(d)
    a.out_dir.mkdir(parents=True,exist_ok=False)
    for name,(header,rows) in tables.items():
        with (a.out_dir/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(rows)
    (a.out_dir/'summary.json').write_text(json.dumps({'status':'ARCHIVED_COHORT_TABLES_RECONSTRUCTED','tables':{k:len(v[1]) for k,v in tables.items()},'recognition_recall':d['recognition']['ever_archived_deadline_passed_delayed_cases']/d['recognition']['delayed_cases'],'unknown_anchor_fraction':d['anchors']['unknown_next']/d['anchors']['total'],'training_performed':False,'new_model_inference':False,'independent_evaluation':False},indent=2)+'\n')

if __name__=='__main__':main()

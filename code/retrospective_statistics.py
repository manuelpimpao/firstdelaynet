#!/usr/bin/env python3
"""Historical score-only evaluation, from the frozen source definitions.
No model loading, training, prediction, threshold selection or raw query extraction.
Optional event input reconstructs retrospective plan endpoints in the SAME ID/time
namespace as the saved scores. Not an independent or prospective evaluation.
No case-level output. Exact archived bootstrap replay requires original lexical
case order; a pseudonymized case order changes the seed-to-case mapping.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from retrospective_tables import tables, ORDER, THRESHOLDS

KEYS=['case_seg','k_idx','tau_ts']
HORIZONS=(0,1,5,10,15)
REFERENCE_ARM='primary_soft_full'
EVENT_COLS=['case_seg','time:timestamp','category','is_delayed','is_delayed_dyn','planned:timestamp']
SCORE_COLS=KEYS+['t_next','next_static_delay']+['prob_'+arm for arm in ORDER]

def require(ok,msg):
    if not ok: raise ValueError(msg)
ensure=need=check=require


# Source-preserved intervals functions.
def utc(s):
    return pd.to_datetime(s, utc=True, errors='coerce', format='mixed').astype('datetime64[ns, UTC]')

def normalize_events(ev):
    check(not set(EVENT_COLS).difference(ev), 'Missing event columns')
    e = ev[EVENT_COLS].copy()
    e['case_seg'] = e.case_seg.astype('string')
    e['time:timestamp'] = utc(e['time:timestamp'])
    e['planned:timestamp'] = utc(e['planned:timestamp'])
    e['category'] = e.category.astype('string').str.strip().str.lower()
    for c in ['is_delayed', 'is_delayed_dyn']:
        e[c] = pd.to_numeric(e[c], errors='coerce')
    check(e.case_seg.notna().all() and e['time:timestamp'].notna().all(),
          'Events have missing case/timestamp')
    return e

def build_intervals(events):
    """Raw event-order static next delay and first-delay risk set.

    Mirrors the V4.67 source's transport-only interval construction:
    chronological transport index k, next transport, static delay, and
    cumulative prior delayed state (including t0). Stable sort is required.
    No event after tau is ever a predictor; this is retrospective audit only.
    """
    e = normalize_events(events)
    tr = e.loc[e.category.eq('transport')].copy()
    check(len(tr) > 0, 'No transport events')
    tr = tr.sort_values(['case_seg', 'time:timestamp'], kind='stable')
    tr['k_idx'] = tr.groupby('case_seg', sort=False, observed=True).cumcount().astype('int64')
    g = tr.groupby('case_seg', sort=False, observed=True)
    tr['t0_src'] = tr['time:timestamp']
    tr['t_next_src'] = g['time:timestamp'].shift(-1)
    tr['next_archived_plan'] = g['planned:timestamp'].shift(-1)
    tr['next_static_delay'] = g['is_delayed'].shift(-1)
    tr['next_dynamic_delay'] = g['is_delayed_dyn'].shift(-1)
    # Missing current delays prohibit definitive first-delay eligibility.
    bad = ~tr['is_delayed'].isin([0, 1])
    tr['prior_unknown'] = bad.groupby(tr.case_seg, sort=False).cumsum().gt(0)
    tr['prior_delayed'] = tr['is_delayed'].eq(1).groupby(tr.case_seg, sort=False).cumsum().gt(0)
    tr['risk_clean'] = ~tr.prior_delayed & ~tr.prior_unknown
    tr['first_delay_next_static'] = tr.risk_clean & tr.next_static_delay.eq(1)
    intervals = tr.loc[tr.t_next_src.notna(), [
        'case_seg', 'k_idx', 't0_src', 't_next_src', 'next_archived_plan',
        'next_static_delay', 'next_dynamic_delay', 'prior_unknown',
        'prior_delayed', 'risk_clean', 'first_delay_next_static'
    ]].copy()
    check(not intervals.duplicated(['case_seg', 'k_idx']).any(), 'Repeated event interval key')
    return intervals, {'transport_events': int(len(tr)),
                       'transport_intervals': int(len(intervals)),
                       'transport_events_missing_static_delay': int(bad.sum()),
                       'cases_with_transport_events': int(tr.case_seg.nunique())}



# Source-preserved alarm functions.
def case_alarm_state(frame,pred):
    ensure(len(frame)==len(pred),'Score/key length mismatch')
    d=frame[KEYS+['y_next_delay','next_static_plan']].copy()
    d['p']=np.asarray(pred,dtype=np.float64)
    ensure(np.isfinite(d.p).all() and d.p.between(0,1).all(),'Invalid alarm probability')
    d=d.sort_values(['case_seg','tau_ts','k_idx'],kind='stable').reset_index(drop=True)
    ids=d.case_seg.astype(str).to_numpy()
    starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1]
    first_case_ids=ids[starts]
    labels=d.y_next_delay.to_numpy(dtype=np.int8)
    has=np.maximum.reduceat(labels,starts).astype(bool)
    lead=(d.next_static_plan-d.tau_ts).dt.total_seconds().to_numpy(dtype=np.float64)
    return d.p.to_numpy(dtype=np.float64),starts,labels,has,lead,first_case_ids

def rate(tp,fp,fn):
    return {'tp':int(tp),'fp':int(fp),'fn':int(fn),
            'precision':float(tp/(tp+fp)) if tp+fp else 0.,
            'recall':float(tp/(tp+fn)) if tp+fn else 0.,
            'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.}

def alarm_metrics(state,threshold,return_matrices=False):
    scores,starts,labels,has,lead,ids=state
    ensure(0<threshold<=1 and np.isfinite(threshold),'Invalid alarm threshold')
    n=len(scores)
    first=np.minimum.reduceat(np.where(scores>=threshold,np.arange(n),n),starts)
    alerted=first<n
    interval_correct=np.zeros(len(starts),dtype=bool)
    lead_first=np.full(len(starts),np.nan)
    if alerted.any():
        ii=first[alerted]
        interval_correct[alerted]=labels[ii]==1
        lead_first[alerted]=lead[ii]
    good=alerted & has & interval_correct & np.isfinite(lead_first) & (lead_first>0)
    d_metrics={};b_metrics={};mat={};bmat={}
    for horizon in HORIZONS:
        hit=good & (lead_first >= 60*horizon)
        tp=int(hit.sum());fp=int((alerted & ~hit).sum());fn=int((has & ~hit).sum())
        d_metrics[str(horizon)]=rate(tp,fp,fn)
        bfp=int((has & alerted & ~hit).sum())
        b_metrics[str(horizon)]=rate(tp,bfp,fn)
        mat[horizon]=np.stack([hit,alerted & ~hit,has & ~hit],axis=1).astype(np.int8)
        bmat[horizon]=np.stack([hit,has & alerted & ~hit,has & ~hit],axis=1).astype(np.int8)
    out={'strict_D':d_metrics['0'],
         'D_horizons':{str(h):d_metrics[str(h)] for h in HORIZONS if h},
         'strict_B_delayed_cases_only':b_metrics['0'],
         'B_horizons':{str(h):b_metrics[str(h)] for h in HORIZONS if h},
         'query_bearing_cases':int(len(starts)),
         'observed_delayed_cases':int(has.sum()),
         'observed_no_target_cases':int((~has).sum()),
         'first_alert_cases':int(alerted.sum()),
         'clean_case_false_alerts':int((alerted&~has).sum()),
         'wrong_interval_first_alerts':int((alerted&has&~interval_correct).sum()),
         'late_correct_interval_first_alerts':int((alerted&has&interval_correct&~good).sum()),
         'delayed_cases_without_first_alert':int((has&~alerted).sum())}
    ensure(out['strict_D']['tp']==out['strict_B_delayed_cases_only']['tp'] and
           out['strict_D']['fn']==out['strict_B_delayed_cases_only']['fn'] and
           out['strict_D']['fp']==out['strict_B_delayed_cases_only']['fp']+
               out['clean_case_false_alerts'],
           'B/D conservation failed')
    return (out,{'D':mat,'B':bmat},ids.copy()) if return_matrices else out



# Source-preserved windows functions.
def confusion(y,s,theta):
    y=np.asarray(y,dtype=np.int8);s=np.asarray(s,dtype=np.float64)
    need(y.ndim==1 and len(y)==len(s) and len(y)>0 and np.isfinite(s).all(),
         'Confusion score/label misalignment')
    need(np.isin(y,[0,1]).all() and ((0<=s)&(s<=1)).all(),'Invalid labels or probabilities')
    a=s>=float(theta)
    tp=int(np.count_nonzero(a & (y==1)))
    fp=int(np.count_nonzero(a & (y==0)))
    fn=int(np.count_nonzero(~a & (y==1)))
    tn=int(np.count_nonzero(~a & (y==0)))
    return {'n_rows':int(len(y)),'positive_rows':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':float(tp/(tp+fp)) if (tp+fp) else 0.,
            'recall':float(tp/(tp+fn)) if (tp+fn) else 0.,
            'f1':float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.}

def historical_windows(frame):
    """Only retrospective source analogues, never a new online-risk-set label.

    B selects rows on each subsequently observed-positive case UNTIL and
    INCLUDING its first y_next_delay positive observation; only that first
    positive observation is assigned label 1. D adds all rows from cases with
    no observed first delay in the current complete historical sequence.

    The historical source may have more special-case selectors; these outputs
    MUST always be called the explicitly reconstructed B/D analogues.
    """
    ordered=frame.sort_values(['case_seg','tau_ts','k_idx'],kind='mergesort').copy()
    need(not ordered.duplicated(list(KEYS)).any(),'Duplicate historical query key')
    need(ordered.y_next_delay.isin((0,1)).all(),'Historical target not binary')
    positive=ordered.groupby('case_seg',sort=False).y_next_delay.transform('max').eq(1)
    rank_pos=ordered.y_next_delay.eq(1) & positive
    first_pos=rank_pos & (rank_pos.groupby(ordered.case_seg,sort=False).cumsum()==1)
    # count chronological rows through earliest positive (inclusive)
    before_or_at=first_pos.groupby(ordered.case_seg,sort=False).cumsum().eq(0) | first_pos
    b_mask=positive & before_or_at
    d_mask=b_mask | (~positive)
    b=ordered.loc[b_mask].copy()
    d=ordered.loc[d_mask].copy()
    b['historical_one_positive_label']=first_pos.loc[b.index].astype(np.int8)
    d['historical_one_positive_label']=first_pos.loc[d.index].astype(np.int8)
    need(int(b.historical_one_positive_label.sum())==int(ordered.loc[positive,'case_seg'].nunique()),
         'B positive denominator drift')
    need(int(d.historical_one_positive_label.sum())==int(b.historical_one_positive_label.sum()),
         'D must not introduce more positive first-delay rows')
    need(len(d)>=len(b),'D must contain B rows')
    # C is a historically outcome-selected, one-row-per-case selector: use only
    # as a nonprospective illustrative diagnostic.
    c=ordered.loc[first_pos | ((~positive) & ~ordered.case_seg.duplicated(keep='last'))].copy()
    c['historical_one_positive_label']=first_pos.loc[c.index].astype(np.int8)
    need(c.case_seg.nunique()==len(c)==ordered.case_seg.nunique(),'Historical C not one row/case')
    return ordered,b,d,c



# Source-preserved scorer functions.
def row_metrics(y,p,t):
    from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p,dtype=np.float64)
    require(len(y)==len(p) and len(y)>0 and np.isfinite(p).all(),
            'Metrics require complete finite scores on frozen query set')
    require(set(np.unique(y))=={0,1},'Expected both target classes')
    flags=p>=t
    tp=int(np.sum(flags & (y==1)));fp=int(np.sum(flags & (y==0)))
    fn=int(np.sum(~flags & (y==1)));tn=int(np.sum(~flags & (y==0)))
    return {
      'average_precision':float(average_precision_score(y,p)),
      'AUROC':float(roc_auc_score(y,p)),
      'Brier':float(brier_score_loss(y,p)),
      'confusion':{'tp':tp,'fp':fp,'fn':fn,'tn':tn},
      'precision':float(tp/(tp+fp)) if tp+fp else 0.,
      'recall':float(tp/(tp+fn)) if tp+fn else 0.,
      'F1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
      'alert_query_fraction':float(flags.mean()),
    }

def calibration_ten_bins(y,p):
    """Prespecified equal-width 10-bin reliability and ECE, no TEST bin selection."""
    yy=np.asarray(y,dtype=np.int8);scores=np.asarray(p,dtype=np.float64)
    require(len(yy)==len(scores) and len(yy)>0 and np.isfinite(scores).all() and
            ((scores>=0)&(scores<=1)).all(),'Invalid calibration inputs')
    idx=np.minimum((scores*10).astype('int8'),9)
    bins=[];ece=0.
    for j in range(10):
        mask=idx==j;n=int(mask.sum())
        if n:
            confidence=float(scores[mask].mean());observed=float(yy[mask].mean())
            ece+=(n/len(yy))*abs(confidence-observed)
        else:confidence=observed=None
        bins.append({'bin':j,'left_inclusive':j/10.,
                     'right_inclusive_except_last':(j+1)/10.,
                     'count':n,'mean_probability':confidence,'observed_positive_rate':observed})
    require(sum(z['count'] for z in bins)==len(yy),'Reliability bin counts incomplete')
    return {'ECE_equal_width_10':float(ece),'bin_count':10,'bins':bins,
            'interpretation':'Descriptive only; source Platt models frozen before TEST; ECE is bin-sensitive'}

def case_equal_query_metrics(truth,p,theta):
    """Secondary: give every observed case total weight 1, not a retraining weight."""
    from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
    yy=truth.next_static_delay.to_numpy(dtype='int8')
    scores=np.asarray(p,dtype=float)
    counts=truth.groupby('case_seg',sort=False).case_seg.transform('size').to_numpy(dtype=float)
    require((counts>0).all(),'Invalid case weights')
    weights=1./counts
    require(np.isclose(weights.sum(),truth.case_seg.nunique(),atol=1e-8),
            'Case-equal weights must sum to number of cases')
    yes=scores>=theta
    tp=float(weights[(yy==1)&yes].sum());fp=float(weights[(yy==0)&yes].sum())
    fn=float(weights[(yy==1)&~yes].sum());tn=float(weights[(yy==0)&~yes].sum())
    return {'weight_unit':'each case contributes exactly total weight 1 over its retained queries',
      'case_count':int(truth.case_seg.nunique()),'effective_weight_sum':float(weights.sum()),
      'average_precision':float(average_precision_score(yy,scores,sample_weight=weights)),
      'AUROC':float(roc_auc_score(yy,scores,sample_weight=weights)),
      'Brier':float(brier_score_loss(yy,scores,sample_weight=weights)),
      'weighted_confusion':{'tp':tp,'fp':fp,'fn':fn,'tn':tn},
      'precision':float(tp/(tp+fp)) if tp+fp else 0.,
      'recall':float(tp/(tp+fn)) if tp+fn else 0.,
      'F1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
      'warning':'Secondary descriptive weighting, not the original primary P5 threshold selection or deployment prevalence'}

def historical_variant_metrics(truth,p,theta,v358):
    """Exact source-selected B/D/C variants; these are retrospective, NOT original full A-E."""
    f=truth[KEYS+['next_static_delay','next_archived_plan']].copy()
    f['y_next_delay']=f['next_static_delay'].astype('int8')
    f['next_static_plan']=f['next_archived_plan']
    f['_score']=np.asarray(p,dtype='float64')
    ordered,b,d,c=v358.historical_windows(f)
    require(int(b.historical_one_positive_label.sum()) ==
            int(ordered.groupby('case_seg').y_next_delay.max().sum()),
            'Historical B positive case denominator mismatch')
    require(d.historical_one_positive_label.sum()==b.historical_one_positive_label.sum(),
            'Historical D/B positive denominator mismatch')
    require(len(c)==truth.case_seg.nunique(), 'Oracle C one case per row contract violated')
    def outcome(z):
        x=v358.confusion(z.historical_one_positive_label.to_numpy(),z._score.to_numpy(),theta)
        return {'rows':int(len(z)),'cases':int(z.case_seg.nunique()),
                'positive_rows':int(z.historical_one_positive_label.sum()),
                'negative_rows':int(len(z)-z.historical_one_positive_label.sum()),
                'confusion_and_prf1':x}
    res={'B_source_analogue':outcome(b),'D_manuscript_inferred':outcome(d),
         'C_outcome_selected_oracle_DIAGNOSTIC_ONLY':{'row_count':int(len(c)),
            'positive_cases':int(c.historical_one_positive_label.sum()),
            'warning':'Outcome-selected row per case; invalid as prospective or primary classifier result'},
         'A_global_clock':{'status':'NOT_RECONSTRUCTIBLE_FROM_STRICT_FIRST_DELAY_COHORT'},
         'definitions':'Original frozen V4.358 historical_windows; B and D are reconstructed source analogues, not exact original manuscript executors'}
    require(res['B_source_analogue']['positive_rows']==res['D_manuscript_inferred']['positive_rows'],
            'B/D source analogue positive counts differ')
    return res

def opportunity_ceilings(truth):
    positive=truth.loc[truth.next_static_delay.eq(1)].copy()
    require(not positive.empty,'No positive first-delay rows; cannot compute opportunity ceilings')
    lag=(positive.next_archived_plan-positive.tau_ts).dt.total_seconds().div(60).to_numpy()
    require(np.isfinite(lag).all() and (lag>0).all(),'Selected positive queries must be strictly predeadline')
    result={}
    eligible_case_count=int(positive.case_seg.nunique())
    for h in HORIZONS:
        possible=positive.loc[lag >= h,'case_seg'].nunique()
        result[str(h)]={'eligible_delayed_cases':eligible_case_count,
            'cases_with_observation_opportunity_at_horizon':int(possible),
            'fraction':float(possible/eligible_case_count)}
    return result

def source_first_alarm_results(truth,p,theta,source163):
    """Frozen case-first-alarm policy: first alarm only, lead to *archived plan*."""
    f=truth[KEYS+['next_static_delay','next_archived_plan','t_next']].copy()
    f['y_next_delay']=f.next_static_delay.astype('int8')
    f['next_static_plan']=f.next_archived_plan
    state=source163.case_alarm_state(f,p)
    score,starts,labels,has,lead_s,ids=state
    r,mats,caseids=source163.alarm_metrics(state,theta,return_matrices=True)
    require(np.array_equal(ids,caseids),'Pinned first-alert source case ids differ')
    require(r['query_bearing_cases']==truth.case_seg.nunique(), 'First-alert case count mismatch')
    require(r['observed_delayed_cases']==int(truth.groupby('case_seg').next_static_delay.max().sum()),
            'First-alert observed first-delay case count mismatch')
    # Explicit, distinct realized completion lead on the FIRST alerted query,
    # restricted to correctly identified first-delay intervals. Not D5 criterion.
    g=f.copy(); g['_p']=np.asarray(p,dtype=float)
    g=g.sort_values(['case_seg','tau_ts','k_idx'],kind='stable')
    first=g.loc[g._p.ge(theta)].groupby('case_seg',sort=False).head(1)
    correct=first.loc[first.next_static_delay.eq(1)].copy()
    plan=(correct.next_archived_plan-correct.tau_ts).dt.total_seconds()/60.
    complete=(correct.t_next-correct.tau_ts).dt.total_seconds()/60.
    require(plan.gt(0).all() and complete.ge(0).all(),
            'Correct predeadline first alerts must precede plan and completion')
    require(int(len(correct))==int(r['strict_D']['tp']),
            'Source strict D0 true positives disagree with manually aligned first alarms')
    if len(correct):
        lead_stats={'correct_target_first_alert_cases':int(len(correct)),
             'lead_to_archived_plan_minutes':{'median':float(plan.median()),'mean':float(plan.mean()),
                    'p25':float(plan.quantile(.25)), 'p75':float(plan.quantile(.75))},
             'lead_to_observed_completion_minutes':{'median':float(complete.median()),
                    'mean':float(complete.mean()),'p25':float(complete.quantile(.25)),
                    'p75':float(complete.quantile(.75))},
             'count_at_least_h_minutes_before_completion':{str(h):int(complete.ge(h).sum()) for h in HORIZONS},
             'warning':'Only correct first alerts, conditional on recorded completion; plan lag defines D0..D15; completion lag is descriptive'}
    else:
        lead_stats={'correct_target_first_alert_cases':0,
            'lead_to_archived_plan_minutes':None,
            'lead_to_observed_completion_minutes':None,
            'count_at_least_h_minutes_before_completion':{str(h):0 for h in HORIZONS}}
    return {'source':'Original V4.163 pinned first-case-alarm implementation',
         'threshold':float(theta), 'first_alarm_locked_to_earliest_case_alert':True,
         'D_horizons_to_ARCHIVED_PLAN':{str(h):r['strict_D'] if h==0 else r['D_horizons'][str(h)] for h in HORIZONS},
         'clean_case_first_alert_count':r['clean_case_false_alerts'],
         'wrong_interval_first_alert_count':r['wrong_interval_first_alerts'],
         'late_correct_interval_first_alert_count':r['late_correct_interval_first_alerts'],
         'eligible_cases':r['query_bearing_cases'],
         'observed_first_delay_cases':r['observed_delayed_cases'],
         'observed_clean_cases':r['observed_no_target_cases'],
         'case_any_alert_E_PERMISSIVE_NOT_LOCALIZATION':{'alerted_cases':r['first_alert_cases'],
                  'fraction':float(r['first_alert_cases']/r['query_bearing_cases'])},
         'first_correct_alarm_lead_time_DESCRIPTIVE_ONLY':lead_stats}, mats,caseids

def paired_case_bootstrap(truth,scores,thresholds,alert_matrices,reps,seed):
    """One matched case draw for ALL models; CI for primary metrics and strict D5 F1."""
    names=list(scores)
    assert len(names)==9 and REFERENCE_ARM in scores
    ids, inv=np.unique(truth.case_seg.astype(str).to_numpy(),return_inverse=True)
    ncases=len(ids)
    # Ensure the original alert engine's unique case axes match the bootstrap axis.
    for name in names:
        require(np.array_equal(alert_matrices[name]['caseids'],ids),
                'First-alert matrix case order does not match shared bootstrap case axis: '+name)
    buckets=[np.flatnonzero(inv==k) for k in range(ncases)]
    y=truth.next_static_delay.to_numpy(dtype=np.int8)
    rng=np.random.default_rng(seed)
    names_metrics=['F1','average_precision','AUROC','Brier']
    vals={arm:{metric:[] for metric in names_metrics} for arm in names}
    strict={policy:{arm:[] for arm in names} for policy in ['primary','D5']}
    n_valid=0
    for rep in range(reps):
        draw=rng.integers(0,ncases,size=ncases)
        ix=np.concatenate([buckets[k] for k in draw])
        yy=y[ix]
        if yy.min()==yy.max(): continue
        n_valid+=1
        for arm in names:
            rr=row_metrics(yy,np.asarray(scores[arm])[ix],thresholds[arm][0])
            for k in names_metrics: vals[arm][k].append(rr[k])
            for policy in strict:
                mat=alert_matrices[arm][policy][5][draw,:]
                tp,fp,fn=mat.sum(axis=0).astype(int)
                strict[policy][arm].append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    require(n_valid>=max(20,int(reps*.8)), 'Too many one-class paired bootstrap samples')
    def ci(a): return [float(z) for z in np.percentile(a,[2.5,97.5])]
    by_arm={name:{'primary_'+k+'_95pct':ci(vals[name][k]) for k in names_metrics}|
                    {p+'_strict_D5_F1_95pct':ci(strict[p][name]) for p in strict} for name in names}
    differences={}
    for name in names:
        if name==REFERENCE_ARM:continue
        d={}
        for k in names_metrics:
            dif=np.asarray(vals[REFERENCE_ARM][k])-np.asarray(vals[name][k])
            d['primary_'+k+'_reference_minus_comparator']={'percentile_95pct':ci(dif),
                         'mean_bootstrap_difference':float(dif.mean())}
        for policy in strict:
            dif=np.asarray(strict[policy][REFERENCE_ARM])-np.asarray(strict[policy][name])
            d[policy+'_strict_D5_F1_reference_minus_comparator']={'percentile_95pct':ci(dif),
                         'mean_bootstrap_difference':float(dif.mean())}
        differences[name]=d
    return {'resampling_unit':'case_seg','matched_case_resamples':True,
       'reference_model':REFERENCE_ARM,'reps_requested':int(reps),'reps_valid':int(n_valid),
       'seed':int(seed),'descriptive_model_intervals':by_arm,
       'paired_reference_minus_all_other_eight':differences,
       'cautions':['Retrospective exposed holdout; interval estimates are descriptive only',
         'Multiple correlated comparisons; no unadjusted significance or superiority claims',
         'Case resampling does not address time-shift, original data-role exposure, or as-of archive limits']}


ALARM=SimpleNamespace(case_alarm_state=case_alarm_state,alarm_metrics=alarm_metrics)
WINDOWS=SimpleNamespace(historical_windows=historical_windows,confusion=confusion)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def align_scores(frame,events=None):
    cols=SCORE_COLS+(['next_archived_plan'] if events is None else [])
    require(set(cols)<=set(frame),'Missing required score fields')
    f=frame[cols].copy().reset_index(drop=True)
    f['case_seg']=f.case_seg.astype('string')
    for c in ['tau_ts','t_next']+(['next_archived_plan'] if events is None else []): f[c]=utc(f[c])
    f['k_idx']=pd.to_numeric(f.k_idx,errors='coerce')
    require(f[KEYS+['t_next']].notna().all().all() and f.k_idx.ge(0).all() and f.k_idx.mod(1).eq(0).all(),'Invalid key/endpoint')
    f['k_idx']=f.k_idx.astype('int64')
    require(not f.duplicated(KEYS).any() and f.next_static_delay.isin([0,1]).all(),'Duplicate keys or invalid target')
    require(f.tau_ts.lt(f.t_next).all(),'Observation must precede completion')
    if events is not None:
        interval,_=build_intervals(events)
        z=f.merge(interval[['case_seg','k_idx','t0_src','t_next_src','next_archived_plan','next_static_delay','risk_clean']],on=['case_seg','k_idx'],how='left',validate='many_to_one',sort=False,suffixes=('','_event'),indicator=True)
        require(len(z)==len(f) and z[KEYS].equals(f[KEYS]),'Endpoint join changed score order')
        require(z._merge.eq('both').all() and z.risk_clean.fillna(False).all(),'Missing or ineligible transport interval')
        require(z.t_next.eq(z.t_next_src).all() and z.next_static_delay.eq(z.next_static_delay_event).all(),'Event completion/label disagrees with saved score')
        require(z.t0_src.le(z.tau_ts).all() and z.next_archived_plan.gt(z.t0_src).all(),'Invalid historical interval')
        f['next_archived_plan']=z.next_archived_plan
    require(f.next_archived_plan.notna().all() and f.tau_ts.lt(f.next_archived_plan).all(),'Strict predeadline cohort required')
    require(set(f.next_static_delay.unique())=={0,1},'Both classes required')
    for arm in ORDER:
        x=f['prob_'+arm].to_numpy(dtype=np.float64)
        require(np.isfinite(x).all() and ((0<=x)&(x<=1)).all(),'Invalid probability')
    return f

def compute(f,reps=200,seed=409):
    # Retain row order. The original source groups lexically ordered string IDs.
    y=f.next_static_delay.to_numpy(dtype=np.int8)
    scores={arm:f['prob_'+arm].to_numpy(dtype=np.float64) for arm in ORDER}
    thresholds=dict(zip(ORDER,THRESHOLDS))
    report={'study_role':'retrospective_previously_exposed_test_not_independent',
            'cohort':{'queries':len(f),'cases':int(f.case_seg.nunique()),'positive_queries':int(y.sum()),'delayed_cases':int(f.groupby('case_seg').next_static_delay.max().sum())},
            'bootstrap':{'draws':reps,'seed':seed,'unit':'case','matched':True},'models':{},'opportunity':opportunity_ceilings(f)}
    matrices={}
    for arm,p in scores.items():
        a,b=thresholds[arm]
        pri=row_metrics(y,p,a)
        fp,mp,ids=source_first_alarm_results(f,p,a,ALARM)
        fd,md,ids2=source_first_alarm_results(f,p,b,ALARM)
        require(np.array_equal(ids,ids2),'Case axes differ')
        bv=historical_variant_metrics(f,p,a,WINDOWS)
        report['models'][arm]={'thresholds':[a,b],'primary':{k:pri[k] for k in ('precision','recall','F1','average_precision','AUROC','Brier','confusion')},
            'case_equal':{k:v for k,v in case_equal_query_metrics(f,p,a).items() if k in ('precision','recall','F1','weighted_confusion')},
            'DL':{k:v for k,v in bv['B_source_analogue']['confusion_and_prf1'].items() if k in ('tp','fp','fn','tn','precision','recall','f1')},
            'AC':{k:v for k,v in bv['D_manuscript_inferred']['confusion_and_prf1'].items() if k in ('tp','fp','fn','tn','precision','recall','f1')},
            'horizons':{'primary':fp['D_horizons_to_ARCHIVED_PLAN'],'D5':fd['D_horizons_to_ARCHIVED_PLAN']},
            'calibration':{k:v for k,v in calibration_ten_bins(y,p).items() if k in ('ECE_equal_width_10','bins')},
            'intervals':{},'paired_intervals':{}}
        matrices[arm]={'primary':mp['D'],'D5':md['D'],'caseids':ids}
    boot=paired_case_bootstrap(f,scores,thresholds,matrices,reps,seed)
    for arm,m in report['models'].items():
        m['intervals']=boot['descriptive_model_intervals'][arm]
        m['paired_intervals']=boot['paired_reference_minus_all_other_eight'].get(arm,{})
    return report,boot['reps_valid']

def compare(actual,expected):
    """Compare every exported numeric leaf and structure; no per-case values."""
    groups={k:{'numeric_values':0,'mismatches':0,'max_abs_difference':0.,'mismatch_fields':[]} for k in ('point_estimates','bootstrap_intervals')}
    def walk(a,b,path):
        g=groups['bootstrap_intervals' if any(k in ('intervals','paired_intervals') for k in path) else 'point_estimates']
        if isinstance(b,dict):
            require(isinstance(a,dict) and set(b)==set(a),'Aggregate field inventory differs: '+'.'.join(path))
            for k,v in b.items():walk(a[k],v,path+[str(k)])
        elif isinstance(b,list):
            require(isinstance(a,list) and len(a)==len(b),'Aggregate list inventory differs')
            for i,v in enumerate(b):walk(a[i],v,path+[str(i)])
        elif isinstance(b,(int,float)) and not isinstance(b,bool):
            require(isinstance(a,(int,float)) and np.isfinite(a),'Nonfinite/non-numeric aggregate')
            g['numeric_values']+=1;d=abs(a-b);g['max_abs_difference']=max(g['max_abs_difference'],d)
            if not np.isclose(a,b,rtol=0,atol=1e-10):
                g['mismatches']+=1;g['mismatch_fields'].append('.'.join(path))
        else: require(a==b,'Aggregate definition differs: '+'.'.join(path))
    walk(actual,expected,[])
    return groups

def self_test():
    base=pd.Timestamp('2020-01-01',tz='UTC'); ev=[]; rows=[]
    for i in range(24):
        cid=f'c{i:02d}';start=base+pd.Timedelta(hours=i)
        for j in range(3):
            ev.append({'case_seg':cid,'time:timestamp':start+pd.Timedelta(minutes=20*j),'category':'transport','is_delayed':int(j==2 and i%2==0),'is_delayed_dyn':0,'planned:timestamp':start+pd.Timedelta(minutes=20*j-2)})
        for k in range(2):
            for offset in (1,9,16):
                y=int(k==1 and i%2==0);row={'case_seg':cid,'k_idx':k,'tau_ts':start+pd.Timedelta(minutes=20*k+offset),'t_next':start+pd.Timedelta(minutes=20*(k+1)),'next_static_delay':y}
                for j,arm in enumerate(ORDER):row['prob_'+arm]=float((.08+.69*y+.03*((i+k+j+offset)%4)) if not (i%6==0 and k==0) else .8)
                rows.append(row)
    f=align_scores(pd.DataFrame(rows),pd.DataFrame(ev));require(len(f)==144,'synthetic alignment')
    # False early alert locks a delayed case out, including if a later alarm is right.
    m,mat,ids=source_first_alarm_results(f,f['prob_primary_soft_full'].to_numpy(),.24,ALARM)
    require(m['wrong_interval_first_alert_count']==4 and m['D_horizons_to_ARCHIVED_PLAN']['0']['tp']==8,'first-alarm locking')
    require(m['D_horizons_to_ARCHIVED_PLAN']['15']['tp']==8,'15 minute horizon')
    a,n=compute(f,20,409);b,n2=compute(f,20,409)
    require(n==n2==20 and all(x['mismatches']==0 for x in compare(a,b).values()),'matched bootstrap deterministic')
    broken=f.copy();broken.loc[0,'next_static_delay']=1-int(broken.loc[0,'next_static_delay'])
    try:align_scores(broken,pd.DataFrame(ev))
    except ValueError:pass
    else:raise AssertionError('label mismatch accepted')
    broken=f.copy();broken.loc[0,'next_archived_plan']=broken.loc[0,'tau_ts']
    try:align_scores(broken)
    except ValueError:pass
    else:raise AssertionError('postdeadline query accepted')
    print('SYNTHETIC_STATISTICS_PASS')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true');p.add_argument('--scores',type=Path);p.add_argument('--scores-sha256');p.add_argument('--events',type=Path);p.add_argument('--events-sha256');p.add_argument('--reference',type=Path);p.add_argument('--reference-sha256',default='4dbac85f03c67d60a7b5e6d804ab91236125ceb9b667a117a147bca021de6bf0');p.add_argument('--out-dir',type=Path);p.add_argument('--expect-original-bootstrap-order',action='store_true',help='Use only for original pinned archived case namespace/order.');p.add_argument('--acknowledge-retrospective',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    require(a.acknowledge_retrospective and all([a.scores,a.scores_sha256,a.reference,a.out_dir]),'Explicit historical scope and paths required')
    require(bool(a.events)==bool(a.events_sha256),'Event file and hash must be supplied together')
    require(not a.out_dir.exists(),'Output directory must be new')
    pins={a.scores:a.scores_sha256,a.reference:a.reference_sha256}
    if a.events:pins[a.events]=a.events_sha256
    require(all(digest(x)==h for x,h in pins.items()),'Input fingerprint mismatch')
    ref=json.loads(a.reference.read_text());tables(ref)
    frame=pd.read_parquet(a.scores,columns=SCORE_COLS+([] if a.events else ['next_archived_plan']))
    f=align_scores(frame,pd.read_parquet(a.events,columns=EVENT_COLS) if a.events else None)
    observed={'queries':len(f),'cases':int(f.case_seg.nunique()),'positive_queries':int(f.next_static_delay.sum()),'delayed_cases':int(f.groupby('case_seg').next_static_delay.max().sum())}
    require(observed==ref['cohort'],'Frozen score population differs')
    report,valid=compute(f);comp=compare(report,ref)
    unchanged=all(digest(x)==h for x,h in pins.items());require(unchanged,'Input changed')
    ok=comp['point_estimates']['mismatches']==0
    boot_ok=comp['bootstrap_intervals']['mismatches']==0 and valid==200
    status=('ARCHIVED_STATISTICS_EXACT_WITHIN_TOLERANCE' if ok and boot_ok else 'POINT_ESTIMATES_PASS_BOOTSTRAP_DIFFERS' if ok else 'POINT_ESTIMATES_DIFFER_REVIEW_REQUIRED')
    summary={'status':status,'cohort':observed,'point_estimate_and_interval_comparison':comp,'tolerance_absolute':1e-10,'bootstrap_valid_draws':valid,'bootstrap_requested_draws':200,'bootstrap_seed':409,'original_case_order_expected':a.expect_original_bootstrap_order,'all_input_pins_unchanged':unchanged,'historical_events_read_for_outcome_join':bool(a.events),'saved_historical_scores_read':True,'raw_query_export_read':False,'new_model_inference':False,'training_performed':False,'threshold_search':False,'independent_holdout':False,'release_authorization_claimed':False,'APs':{k:m['primary']['average_precision'] for k,m in report['models'].items()},'numpy':np.__version__,'pandas':pd.__version__}
    import sklearn;summary['sklearn']=sklearn.__version__
    a.out_dir.mkdir(parents=True,exist_ok=False)
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (a.out_dir/'computed_aggregates.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(status)
    if not ok or (a.expect_original_bootstrap_order and not boot_ok):raise SystemExit(2)

if __name__=='__main__':main()

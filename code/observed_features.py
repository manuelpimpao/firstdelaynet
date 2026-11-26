#!/usr/bin/env python3
"""Rebuild the 19 observed operational TRAIN features from processed event history.
The candidate is not a raw industrial extractor. Input projections exclude outcome,
plan, and future-event fields; historical query keys are fixed, not a prospective
scheduler. The optional reference t0 is used only for anchor verification.
No models are loaded, fitted or scored. Only aggregate summary is written.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

VERSION='1.0'
WIN=(5,15,60)
NS_MIN=60_000_000_000
NS_HOUR=60*NS_MIN
KEYS=['case_seg','k_idx','tau_ts']
TS='time:timestamp'
OBS19=[f'{v}_{w}m_event_end' for w in WIN for v in ('starts','burstiness_cv')]+['new_cases_1h','active_cases_1h','ewma_if_event_end','momentum_if_event_end']+[f'observed_transport_{w}m_event_end' for w in WIN]+[f'observed_transport_global_{w}m' for w in WIN]+['observed_transport_count_case_at_tau','seconds_since_last_observed_transport','seconds_since_observed_transport_t0']
EVENT_COLS=['case_seg',TS,'category','event_end']
TOL=1e-6
PINS={'revised_train_events': '4868bb39776e0a303a97397cb51f2ebb5a0e46ed52db45800120de2d3c17c1cc', 'train_student_candidates': 'c99d0eadc5be09f11a948824055989894e00b068e6d1a694ce91196c340e3bc5', 'train_retrospective_metadata': '0c09a13afcac4e3dab1276709fd0e05606476dc973fc87f8591c4541c50a5e62'}

def ns(x)->np.ndarray:
    d=pd.to_datetime(x,utc=True,errors='raise')
    a=d.to_numpy(dtype='datetime64[ns]')
    if np.isnat(a).any():raise ValueError('Null timestamps')
    return a.view('i8')

def win_counts(sorted_times:np.ndarray, query_times:np.ndarray, w:int)->np.ndarray:
    hi=np.searchsorted(sorted_times,query_times,side='right')
    lo=np.searchsorted(sorted_times,query_times-int(w)*NS_MIN,side='right')
    return (hi-lo).astype(np.int32)

def window_counts_cv(sorted_times:np.ndarray, queries:np.ndarray)->dict:
    """Faithful 60 one-minute-bin implementation; (tau-W,tau] endpoints."""
    t=np.sort(np.asarray(sorted_times,dtype=np.int64))
    q=np.asarray(queries,dtype=np.int64)
    out={}
    for w in WIN:
        out[f'starts_{w}m_event_end']=np.zeros(len(q),dtype=np.int32)
        out[f'burstiness_cv_{w}m_event_end']=np.zeros(len(q),dtype=np.float32)
    for beg in range(0,len(q),1024):
        block=q[beg:beg+1024]
        edges=block[:,None]-np.arange(61,dtype=np.int64)[None,:]*NS_MIN
        cum=np.searchsorted(t,edges,side='right')
        minute_bins=cum[:,:-1]-cum[:,1:]
        for w in WIN:
            v=minute_bins[:,:w].astype(np.float64)
            mu=v.mean(axis=1)
            std=v.std(axis=1,ddof=1)
            cv=np.divide(std,mu,out=np.zeros_like(std),where=mu>0)
            out[f'starts_{w}m_event_end'][beg:beg+len(block)]=(cum[:,0]-cum[:,w]).astype(np.int32)
            out[f'burstiness_cv_{w}m_event_end'][beg:beg+len(block)]=cv.astype(np.float32)
    return out

def reconstruct(events:pd.DataFrame,tau:pd.DataFrame)->dict[str,np.ndarray]:
    if tau.empty:raise ValueError('Empty query sample')
    if tau.duplicated(list(KEYS)).any():raise ValueError('Duplicate tau key')
    event_t=ns(events['time:timestamp'])
    query_t=ns(tau['tau_ts'])
    ev_st=events['event_end'].fillna('').to_numpy(dtype=str)
    q_st=tau.station.fillna('').to_numpy(dtype=str)
    ev_case=events.case_seg.to_numpy(dtype=str)
    out={name:np.zeros(len(tau),dtype=np.float32 if ('cv_' in name or name.startswith(('ewma','momentum','seconds_'))) else np.int32) for name in OBS19}
    # Source-grounded station event counts, CV and query-invariant event-clock EWMA.
    for station in np.unique(q_st):
        idx=np.flatnonzero(q_st==station)
        t=np.sort(event_t[(ev_st==station)&(ev_st!='')])
        values=window_counts_cv(t,query_t[idx])
        for name,arr in values.items():out[name][idx]=arr
        if not len(t):continue
        ticks=np.unique(t)
        past_hour=win_counts(t,ticks,60).astype(np.float64)
        smoothed=pd.Series(past_hour).ewm(alpha=.3,adjust=False).mean().to_numpy()
        momentum=np.diff(smoothed,prepend=smoothed[0])
        last=np.searchsorted(ticks,query_t[idx],side='right')-1
        valid=last>=0
        out['ewma_if_event_end'][idx[valid]]=smoothed[last[valid]].astype(np.float32)
        out['momentum_if_event_end'][idx[valid]]=momentum[last[valid]].astype(np.float32)
    # First-observed timestamp per case, and distinct active cases in prior hour.
    first=pd.DataFrame({'case':ev_case,'event_t':event_t}).groupby('case',sort=False).event_t.min().to_numpy(dtype=np.int64)
    first.sort()
    out['new_cases_1h']=win_counts(first,query_t,60)
    ev_order=np.argsort(event_t,kind='stable')
    et=event_t[ev_order];ec=ev_case[ev_order]
    q_order=np.argsort(query_t,kind='stable')
    live=defaultdict(int);left=right=0
    active=np.zeros(len(tau),dtype=np.int32)
    for ix in q_order:
        t=query_t[ix]
        while right<len(et) and et[right]<=t:
            live[ec[right]]+=1;right+=1
        while left<right and et[left]<=t-NS_HOUR:
            key=ec[left]
            live[key]-=1
            if not live[key]:del live[key]
            left+=1
        active[ix]=len(live)
    out['active_cases_1h']=active
    # Observed TRANSPORT-only counts, not scheduled arrivals.
    is_tr=events.category.eq('transport').fillna(False).to_numpy(dtype=bool)
    tr=event_t[is_tr]
    for station in np.unique(q_st):
        idx=np.flatnonzero(q_st==station)
        station_tr=np.sort(event_t[is_tr&(ev_st==station)])
        for w in WIN:
            out[f'observed_transport_{w}m_event_end'][idx]=win_counts(station_tr,query_t[idx],w)
    tr.sort()
    for w in WIN:
        out[f'observed_transport_global_{w}m']=win_counts(tr,query_t,w)
    tr_case=ev_case[is_tr]
    # Group transport timestamp histories only for cases in sampled queries.
    want=set(tau.case_seg.to_numpy(dtype=str))
    grouped_transport=pd.DataFrame({'case':tr_case,'time_ns':event_t[is_tr]})
    grouped_transport=grouped_transport.loc[grouped_transport.case.isin(want)]
    by_case={str(case):np.sort(g.time_ns.to_numpy(dtype=np.int64))
             for case,g in grouped_transport.groupby('case',sort=False)}
    counts=out['observed_transport_count_case_at_tau']
    age=np.full(len(tau),-1.,dtype=np.float32)
    for case,indices in tau.groupby('case_seg',sort=False).indices.items():
        t=by_case.get(str(case))
        if t is None or not len(t):continue
        q=query_t[indices]
        cut=np.searchsorted(t,q,side='right')
        counts[indices]=cut.astype(np.int32)
        ok=cut>0
        age[indices[ok]]=(q[ok]-t[cut[ok]-1])/1_000_000_000
    out['seconds_since_last_observed_transport']=age
    dt=(query_t-ns(tau.t0))/1_000_000_000
    if np.any(dt<0):raise ValueError('TRAIN tau precedes t0; cannot compute age safely')
    out['seconds_since_observed_transport_t0']=dt.astype(np.float32)
    for name,x in out.items():
        if len(x)!=len(tau) or not np.isfinite(x).all():raise ValueError(f'Invalid feature {name}')
    return out

class Blocked(Exception): pass

def need(ok,code):
    if not bool(ok): raise Blocked(code)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def utc(s):
    return pd.to_datetime(s,utc=True,errors='raise').astype('datetime64[ns, UTC]')

def keys(q):
    q=q.copy().reset_index(drop=True)
    q['case_seg']=q.case_seg.astype('string')
    q['tau_ts']=utc(q.tau_ts)
    k=pd.to_numeric(q.k_idx,errors='raise')
    need(k.notna().all() and k.ge(0).all() and k.mod(1).eq(0).all(),'INVALID_INTERVAL_INDEX')
    q['k_idx']=k.astype('int64')
    need(q[KEYS].notna().all().all() and not q.duplicated(KEYS).any(),'INVALID_QUERY_KEYS')
    return q

def anchor_inputs(events,query_keys):
    e=events[EVENT_COLS].copy()
    e['case_seg']=e.case_seg.astype('string'); e[TS]=utc(e[TS])
    e['category']=e.category.astype('string').str.strip().str.lower()
    e['event_end']=e.event_end.astype('string').fillna('')
    need(e[['case_seg',TS]].notna().all().all() and not e.duplicated(['case_seg',TS]).any(),'INVALID_EVENT_KEYS')
    q=keys(query_keys[KEYS]); need(len(q)>0,'EMPTY_QUERIES')
    tr=e.loc[e.category.eq('transport').fillna(False),['case_seg',TS,'event_end']].sort_values(['case_seg',TS],kind='stable').copy()
    tr['k_idx']=tr.groupby('case_seg',sort=False,observed=True).cumcount().astype('int64')
    tr=tr.rename(columns={TS:'t0','event_end':'station'})
    out=q.merge(tr,on=['case_seg','k_idx'],how='left',validate='many_to_one',sort=False,indicator=True)
    need(out._merge.eq('both').all() and out[KEYS].equals(q[KEYS]),'ANCHOR_JOIN')
    need(out.t0.le(out.tau_ts).all(),'ANCHOR_AFTER_QUERY')
    return e,out.drop(columns='_merge')

def compare(ref,rec):
    result={}
    for c in OBS19:
        a=pd.to_numeric(ref[c],errors='raise').to_numpy(np.float64); b=rec[c].astype(np.float64)
        need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'FEATURE_NONFINITE_OR_SHAPE')
        d=np.abs(a-b); is_count=np.issubdtype(rec[c].dtype,np.integer)
        threshold=0. if is_count else TOL
        result[c]={'values':len(a),'exact_values':int((d==0).sum()),'max_abs_difference':float(d.max()),'mean_abs_difference':float(d.mean()),'values_outside_tolerance':int((d>threshold).sum()),'absolute_tolerance':threshold}
    return result

def selftest():
    t=pd.Timestamp('2026-01-05T10:05:00Z'); m=pd.Timedelta(minutes=1)
    e=pd.DataFrame({'case_seg':['a','a','b','a','c'],TS:[t-5*m,t-2*m,t+m,t,t+4*m],'category':['transport']*5,'event_end':['s']*5})
    q=pd.DataFrame({'case_seg':['a'],'k_idx':[0],'tau_ts':[t]})
    ev,a=anchor_inputs(e,q); x=reconstruct(ev,a)
    need(x['starts_5m_event_end'][0]==2 and x['observed_transport_count_case_at_tau'][0]==3 and x['seconds_since_observed_transport_t0'][0]==300,'TEST_FORMULA')
    ev,b=anchor_inputs(e.loc[e[TS]<=t],q); y=reconstruct(ev,b)
    need(all(np.array_equal(x[c],y[c]) for c in OBS19),'TEST_FUTURE_INVARIANCE')
    need(all(v['values_outside_tolerance']==0 for v in compare(pd.DataFrame(x),y).values()),'TEST_COMPARE')
    for bad in [q.assign(k_idx=8),q.assign(tau_ts=t-6*m)]:
        try: anchor_inputs(e,bad)
        except Blocked: pass
        else: raise Blocked('TEST_ANCHOR_REJECTION')
    try: anchor_inputs(pd.concat([e,e.iloc[[0]]]),q)
    except Blocked: pass
    else: raise Blocked('TEST_DUPLICATE_REJECTION')
    print('Synthetic anchor, window, future-invariance, comparison and rejection checks passed.')

def run(data,r):
    paths={n:data/(n+'.parquet') for n in PINS}
    need(all(p.is_file() and not p.is_symlink() and sha(p)==PINS[n] for n,p in paths.items()),'INPUT_PIN')
    e=pd.read_parquet(paths['revised_train_events'],columns=EVENT_COLS)
    ref=keys(pd.read_parquet(paths['train_student_candidates'],columns=KEYS+OBS19))
    # t0 is a historical completed transport, not a next-event outcome.
    a=keys(pd.read_parquet(paths['train_retrospective_metadata'],columns=KEYS+['t0']))
    need(len(e)==874844 and len(ref)==331536 and len(a)==len(ref),'COHORT_ROWS')
    ev,q=anchor_inputs(e,ref[KEYS]); a['t0']=utc(a.t0)
    z=q[KEYS+['t0']].merge(a,on=KEYS,validate='one_to_one',how='left',sort=False,suffixes=('_derived','_reference'),indicator=True)
    need(z._merge.eq('both').all() and z[KEYS].equals(q[KEYS]),'REFERENCE_ANCHOR_JOIN')
    r.update(event_rows=len(ev),event_cases=int(ev.case_seg.nunique()),query_rows=len(q),query_cases=int(q.case_seg.nunique()),anchor_timestamp_mismatches=int(z.t0_derived.ne(z.t0_reference).sum()),query_order_preserved=True)
    need(r['event_cases']==82930 and r['query_cases']==44699,'COHORT_CASES')
    need(r['anchor_timestamp_mismatches']==0,'REFERENCE_ANCHOR_DIFFERENCE')
    print('Reconstructing 19 operational features on all fixed TRAIN query keys...',flush=True)
    rec=reconstruct(ev,q); need(list(rec)==OBS19,'FEATURE_ORDER')
    r['per_feature']=compare(ref,rec)
    r['features_within_tolerance']=sum(v['values_outside_tolerance']==0 for v in r['per_feature'].values())
    r['total_values']=sum(v['values'] for v in r['per_feature'].values());r['exact_values']=sum(v['exact_values'] for v in r['per_feature'].values());r['all_input_pins_unchanged']=all(sha(p)==PINS[n] for n,p in paths.items())
    need(r['all_input_pins_unchanged'],'INPUT_CHANGED')
    r['status']='TRAIN_OBS19_RECONSTRUCTION_PASS' if r['features_within_tolerance']==19 else 'TRAIN_OBS19_RECONSTRUCTION_DIFFERENCE'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path);p.add_argument('--out-dir',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    selftest()
    if a.self_test:return
    if a.data_dir is None or a.out_dir is None:p.error('--data-dir and --out-dir required')
    out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    r={'version':VERSION,'status':'BLOCKED','scope':'processed TRAIN event-history features on fixed query keys','official_TEST_or_VAL_read':False,'labels_or_future_outcomes_read':False,'model_training_or_inference':False,'raw_extraction_reproduced':False,'prospective_scheduler_validated':False,'release_authorization_claimed':False}
    try:run(a.data_dir.resolve(),r)
    except Exception as exc:r.update(exception_type=type(exc).__name__,check_code=str(exc) if isinstance(exc,Blocked) else 'LOCAL_REVIEW_REQUIRED')
    (out/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8');print(r['status']);raise SystemExit(0 if r['status']=='TRAIN_OBS19_RECONSTRUCTION_PASS' else 1)

if __name__=='__main__':main()

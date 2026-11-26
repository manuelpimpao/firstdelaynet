#!/usr/bin/env python3
"""Candidate-only frozen historical evaluation. No training, threshold selection,
private identifiers, original datasets, or original research project needed.
This is a draft for controlled local testing, not permission to redistribute assets.
The target is a retrospective first delay against the archived plan, not a new holdout.
"""
from __future__ import annotations
import argparse, hashlib, json, types
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import xgboost as xgb
import joblib
from catboost import CatBoostClassifier
PAD,UNK,QUERY=0,1,2
SPECIAL=['<PAD>','<UNK>','<QUERY>']
EPS=1.e-6
BATCH=512
SEG='case_seg'; TS='time:timestamp'; QTS='tau_ts'; K='k_idx'
KEYS=[SEG,K,QTS]
EVENT_COLS=[SEG,TS,'category','is_delayed','is_delayed_dyn','planned:timestamp']
TAU_COLS=KEYS+['t0','t_next','y_next_delay','has_post','events_since_t0','station','time_since_t0_s']
WIN=(5,15,60); NS_MIN=60_000_000_000; NS_HOUR=60*NS_MIN
EVENT_KEY=[SEG,TS]
LOCAL_COLS=['roll_ie_mean','roll_ie_min','roll_ie_max','case_length_prefix','case_mean_iet_prefix']
WINDOW_COLS=['service_rate_event_end_1h','in_flight_event_end_1h','starts_5m_event_end','starts_15m_event_end']
NINE=LOCAL_COLS+WINDOW_COLS
OBS19=[f'{v}_{w}m_event_end' for w in WIN for v in ('starts','burstiness_cv')]+['new_cases_1h','active_cases_1h','ewma_if_event_end','momentum_if_event_end']+[f'observed_transport_{w}m_event_end' for w in WIN]+[f'observed_transport_global_{w}m' for w in WIN]+['observed_transport_count_case_at_tau','seconds_since_last_observed_transport','seconds_since_observed_transport_t0']

EVENT_FEATURES=['inter_event_time', 'roll_ie_mean', 'roll_ie_min', 'roll_ie_max', 'hour_of_day', *[f'weekday_{i}' for i in range(7)], 'case_length_prefix', 'case_mean_iet_prefix', 'concept:name_repeat_flag', 'prefix_length', 'schichtwechsel', 'time_since_start', 'unique_stations_so_far', 'service_rate_event_end_1h', 'in_flight_event_end_1h', 'starts_5m_event_end', 'starts_15m_event_end']

QFEATURES=['query_seconds_since_last_event','query_seconds_since_observed_transport_t0','is_query_token']

class Blocked(Exception): pass
def need(x,code):
 if not bool(x): raise Blocked(code)

def tokens(s,pre):return np.array([pre['vocab'].get(x,1) for x in s.astype(str)],dtype=np.int32)

def numeric(e,q,pre):
 x=e[EVENT_FEATURES].to_numpy(np.float32); x=np.sign(x)*np.log1p(np.abs(x))
 mu=np.asarray(pre['event_mean'],np.float32);sd=np.asarray(pre['event_std'],np.float32)
 need(mu.shape==sd.shape==(23,) and np.all(sd>0),'BAD_EVENT_SCALER')
 ev=np.clip((x-mu)/sd,-12,12).astype(np.float32)
 x=q[QFEATURES[:2]].to_numpy(np.float32);need(np.isfinite(x).all() and (x>=0).all(),'BAD_QUERY_GAPS')
 qm=np.asarray(pre['query_log1p_mean'],np.float32);qs=np.asarray(pre['query_log1p_std'],np.float32)
 need(qm.shape==qs.shape==(2,) and np.all(qs>0),'BAD_QUERY_SCALER')
 qt=np.clip((np.log1p(x)-qm)/qs,-12,12).astype(np.float32)
 need(np.isfinite(ev).all() and np.isfinite(qt).all(),'NONFINITE_TENSORS');return ev,qt

def check(ok, message):
    need(ok, "ARCHIVED_HELPER_ASSERTION_FAILED")

def require(ok, message):
    need(ok, "ARCHIVED_HELPER_ASSERTION_FAILED")


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


def event_rolling_from_times(case_events: pd.DataFrame):
    """All observed events (including current), sorted per case; no later events."""
    part = case_events.sort_values(EVENT_KEY, kind='stable').copy()
    gr = part.groupby('case_seg', sort=False, observed=True)
    delta = gr['time:timestamp'].diff().dt.total_seconds().fillna(0).astype('float32')
    part['_inter_event_s'] = delta
    gd = part.groupby('case_seg', sort=False, observed=True)['_inter_event_s']
    for col, op in [('roll_ie_mean', 'mean'), ('roll_ie_min', 'min'),
                    ('roll_ie_max', 'max')]:
        part[col] = gd.transform(lambda s: getattr(s.rolling(3, min_periods=1), op)()).astype('float32')
    part['case_length_prefix'] = (gr.cumcount() + 1).astype('float32')
    # Numerator and denominator only include current and earlier observed events.
    gr_delta = part.groupby('case_seg', sort=False, observed=True)['_inter_event_s']
    part['case_mean_iet_prefix'] = (gr_delta.cumsum() /
                                   (gr_delta.cumcount() + 1)).astype('float32')
    return part

def station_window_counts(all_events: pd.DataFrame, target: pd.DataFrame):
    """V4.67 reconstruct_teacher_numeric strict (t-window,t] in NANOSECONDS.

    Explicit conversion is essential: pandas DatetimeArray.asi8 uses the
    array's storage unit, which is microseconds for the revised Parquet data.
    """
    res = target.copy()
    res['service_rate_event_end_1h'] = np.zeros(len(res), dtype=np.int32)
    res['in_flight_event_end_1h'] = np.zeros(len(res), dtype=np.int32)
    res['starts_5m_event_end'] = np.zeros(len(res), dtype=np.int32)
    res['starts_15m_event_end'] = np.zeros(len(res), dtype=np.int32)
    # Build indices from ALL TRAIN event rows, not only sampled query cases.
    groups = {}
    tns = all_events['time:timestamp'].to_numpy(dtype='datetime64[ns]').astype('int64')
    for station, positions in all_events.groupby('event_end', sort=False, observed=True).indices.items():
        if str(station).strip():
            groups[str(station)] = np.sort(tns[positions])
    qns = res['time:timestamp'].to_numpy(dtype='datetime64[ns]').astype('int64')
    for station, positions in res.groupby('event_end', sort=False, observed=True).indices.items():
        seq = groups.get(str(station))
        if seq is None or not len(seq):
            continue
        query = qns[positions]
        end = np.searchsorted(seq, query, side='right')
        counts = {}
        for minutes in (5, 15, 60):
            start = np.searchsorted(seq, query - minutes * 60 * 1_000_000_000,
                                    side='right')
            counts[minutes] = (end - start).astype('int32')
        for col, n in [('starts_5m_event_end', 5),
                       ('starts_15m_event_end', 15),
                       ('service_rate_event_end_1h', 60),
                       ('in_flight_event_end_1h', 60)]:
            res.iloc[positions, res.columns.get_loc(col)] = counts[n]
    return res

def rebuild(all_events, chosen_cases, max_tau_by_case):
    local = all_events.loc[all_events.case_seg.isin(chosen_cases)].copy()
    # Only events observed by each sampled case's latest query are local inputs.
    # The global station-window context remains the entire event-time-censored TRAIN source.
    cutoff = local.case_seg.map(max_tau_by_case)
    local = local.loc[cutoff.notna() & local['time:timestamp'].le(cutoff)].copy()
    require(len(local) > 0, 'No matching TRAIN source case-prefix events')
    require(not local.duplicated(EVENT_KEY, keep=False).any(),
            'Ambiguous per-case timestamp ties inside sampled TRAIN prefixes')
    derived = event_rolling_from_times(local)
    derived = station_window_counts(all_events, derived)
    derived = derived[EVENT_KEY + NINE].copy()
    require(not derived.duplicated(EVENT_KEY).any(),
            'Derived sampled prefix event keys are not unique')
    require(np.isfinite(derived[NINE].to_numpy(dtype=np.float64)).all(),
            'Nonfinite derived event channels')
    return derived


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

def normalize_tau(tau):
    check(not set(TAU_COLS).difference(tau), 'Missing query columns')
    t = tau[TAU_COLS].copy()
    t['case_seg'] = t.case_seg.astype('string')
    for c in ['tau_ts', 't0', 't_next']:
        t[c] = utc(t[c])
    for c in ['k_idx', 'y_next_delay', 'has_post', 'events_since_t0', 'time_since_t0_s']:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    check(t[KEYS + ['t0', 't_next']].notna().all().all(), 'Query keys or endpoints missing')
    check(t.k_idx.ge(0).all() and t.k_idx.mod(1).eq(0).all(), 'Invalid transport k_idx')
    t['k_idx'] = t.k_idx.astype('int64')
    return t

def distinct_queries(t):
    """Diagnostic key-universe only, after confirming audited duplicate equality."""
    g = t.groupby(KEYS, dropna=False, observed=True, sort=False)
    for c in ['t0', 't_next', 'y_next_delay', 'has_post', 'events_since_t0',
              'station', 'time_since_t0_s']:
        check(not g[c].nunique(dropna=False).gt(1).any(),
              'Conflicting repeated query input/target: ' + c)
    return t.drop_duplicates(KEYS, keep='first').copy()

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


def normalized_event_inputs(raw: pd.DataFrame, teacher, nine: list[str]) -> pd.DataFrame:
    required=set(['case_seg','time:timestamp','event_end','category','concept:name']) | (set(teacher.EVENT_FEATURES)-set(nine))
    check(required.issubset(raw.columns), 'Missing historical event input columns: '+str(sorted(required-set(raw.columns))))
    d=raw.copy()
    d['case_seg']=d.case_seg.astype('string')
    d['time:timestamp']=pd.to_datetime(d['time:timestamp'],utc=True,errors='raise').astype('datetime64[ns, UTC]')
    d['event_end']=d.event_end.astype('string').fillna('').str.strip()
    d['category']=d.category.astype('string').fillna('').str.strip().str.lower()
    d['concept:name']=d['concept:name'].astype('string').fillna('')
    check(d.case_seg.notna().all() and d['time:timestamp'].notna().all(), 'Invalid historical event IDs/time')
    return d

def prefix_arrays(events:pd.DataFrame, queries:pd.DataFrame, v403, teacher):
    """Local prefix events + case query pointer; global windows still use ALL events."""
    q=queries.copy().reset_index(drop=True)
    cut=q.groupby('case_seg',sort=False,observed=True).tau_ts.max()
    chosen=set(cut.index.astype(str).tolist())
    all_for_windows=events[['case_seg','time:timestamp','event_end']].copy()
    rebuilt=v403.rebuild(all_for_windows,chosen,cut)
    keep=events.case_seg.isin(chosen)
    scoped=events.loc[keep].copy()
    cutoff=scoped.case_seg.map(cut)
    scoped=scoped.loc[cutoff.notna() & scoped['time:timestamp'].le(cutoff)].copy()
    check(not scoped.duplicated(['case_seg','time:timestamp']).any(),
          'Ambiguous same-case timestamp ties in input prefixes')
    scoped=scoped.merge(rebuilt,on=['case_seg','time:timestamp'],how='left',
                        validate='one_to_one',indicator=True,suffixes=('','_reconstructed'))
    check(scoped['_merge'].eq('both').all() and len(scoped)==len(rebuilt),
          'Failed reconstructed teacher channels join')
    scoped=scoped.drop(columns='_merge').sort_values(['case_seg','time:timestamp'],kind='stable').reset_index(drop=True)
    all_23=list(teacher.EVENT_FEATURES)
    check(len(all_23)==23 and len(set(all_23))==23, 'Changed original teacher input contract')
    check(set(v403.NINE).issubset(set(all_23)), 'Reconstructed channel list is not teacher contract')
    numeric=scoped[all_23].apply(pd.to_numeric,errors='coerce')
    check(np.isfinite(numeric.to_numpy(dtype=np.float64)).all(),
          'Nonfinite teacher event numeric values')
    for col in all_23:
        scoped[col] = pd.to_numeric(numeric[col],errors='raise').astype('float32')
    positions=np.arange(len(scoped),dtype=np.int64)
    scoped['_global_event_index']=positions
    seen={}
    for case,g in scoped.groupby('case_seg',sort=False,observed=True):
        seen[str(case)]=(g['time:timestamp'].to_numpy(dtype='datetime64[ns]').astype('int64'),
                         g['_global_event_index'].to_numpy(dtype='int64'))
    q['observed_event_count']=np.zeros(len(q),dtype='int64')
    q['last_event_global_index']=np.zeros(len(q),dtype='int64')
    last_ts=[];gaps=[]
    for i,(case,timestamp) in enumerate(zip(q.case_seg.astype(str),q.tau_ts)):
        assertcase=seen.get(case)
        check(assertcase is not None, 'Missing observed source case prefix')
        timeline,global_indices=assertcase
        now=timestamp.value  # nanoseconds explicitly, never Parquet storage units
        pos=int(np.searchsorted(timeline,now,side='right'))
        check(pos>=1, 'No historical event visible at sampled tau')
        check(pos==len(timeline) or timeline[pos]>now,
              'Future case event was included in strict query prefix')
        q.at[i,'observed_event_count']=pos
        q.at[i,'last_event_global_index']=int(global_indices[pos-1])
        last_ts.append(pd.Timestamp(timeline[pos-1],tz='UTC'))
        gaps.append((now-int(timeline[pos-1]))/1e9)
    q['last_event_timestamp']=pd.to_datetime(last_ts,utc=True)
    q['query_seconds_since_last_event']=np.asarray(gaps,dtype='float32')
    q['query_seconds_since_observed_transport_t0']=((q.tau_ts-q.t0).dt.total_seconds()).astype('float32')
    q['is_query_token']=np.ones(len(q),dtype='float32')
    check(q['query_seconds_since_last_event'].ge(0).all() and
          q['query_seconds_since_observed_transport_t0'].ge(0).all(),
          'Negative query age/last-event gap')
    check(q['last_event_timestamp'].le(q.tau_ts).all(), 'Future last-event time')
    return scoped,q

def build_cohort(tau,events,v405):
    """Exactly V4.405 definition. TEST may yield different counts; never filter on scores."""
    t=v405.normalize_tau(tau)
    u=v405.distinct_queries(t)
    intervals,info=v405.build_intervals(events)
    joined=u.merge(intervals,on=['case_seg','k_idx'],how='left',indicator=True,
                   validate='many_to_one',sort=False)
    matched=joined['_merge'].eq('both')
    consistent=matched & joined.t0.eq(joined.t0_src) & joined.t_next.eq(joined.t_next_src)
    valid=consistent & joined.t0.le(joined.tau_ts) & joined.tau_ts.lt(joined.t_next)
    # An inconsistent crosswalk is an audit failure, NOT an incentive to drop rows.
    require(bool(valid.all()),'TEST query / transport interval mismatch; abort, do not silently exclude')
    clean=joined.risk_clean.fillna(False).astype(bool)
    plan=joined.next_archived_plan.notna() & joined.next_archived_plan.gt(joined.t0)
    before=plan & joined.tau_ts.lt(joined.next_archived_plan)
    known=joined.next_static_delay.isin([0,1])
    eligible=valid & clean & before & known
    selected=joined.loc[eligible].copy().reset_index(drop=True)
    require(len(selected)>0,'No eligible queries under frozen TRAIN-defined cohort rule')
    require(not selected.duplicated(KEYS).any(),'Duplicate unique TEST query keys')
    require(selected.y_next_delay.isin([0,1]).all(), 'Invalid binary historical legacy label')
    y=selected.next_static_delay.astype('int64').to_numpy()
    require(np.array_equal(y,selected.y_next_delay.to_numpy(dtype='int64')),
            'First-delay static target differs from legacy label on eligible TEST cohort; abort')
    require(np.unique(y).tolist()==[0,1], 'Both label classes are necessary for fixed ranking metrics')
    return selected,{
      'historical_raw_query_rows':int(len(t)),
      'historical_unique_query_keys':int(len(u)),
      'matched_valid_transport_interval_queries':int(valid.sum()),
      'prior_on_time_candidate_queries':int(clean.sum()),
      'pre_archived_deadline_queries':int(before.sum()),
      'eligible_queries':int(len(selected)),
      'eligible_cases':int(selected.case_seg.nunique()),
      'eligible_positive_queries':int(y.sum()),
      'eligible_positive_fraction':float(y.mean()),
      'excluded_unique_queries':int(len(u)-len(selected)),
      'raw_duplicate_rows_not_independent':int(len(t)-len(u)),
      'legacy_label_static_first_delay_agreement_in_eligible':1.0,
      'first_delay_is_retrospectively_defined':True,
      'transport_event_source_counts':info,
    }

EXPECTED_COHORT={'historical_raw_query_rows': 531363, 'historical_unique_query_keys': 128389, 'matched_valid_transport_interval_queries': 128389, 'prior_on_time_candidate_queries': 48376, 'pre_archived_deadline_queries': 54415, 'eligible_queries': 39775, 'eligible_cases': 7146, 'eligible_positive_queries': 5410, 'eligible_positive_fraction': 0.13601508485229416, 'excluded_unique_queries': 88614, 'raw_duplicate_rows_not_independent': 402974, 'legacy_label_static_first_delay_agreement_in_eligible': 1.0, 'first_delay_is_retrospectively_defined': True, 'transport_event_source_counts': {'transport_events': 86564, 'transport_intervals': 60851, 'transport_events_missing_static_delay': 0, 'cases_with_transport_events': 25713}}




TEACHER=types.SimpleNamespace(EVENT_FEATURES=EVENT_FEATURES,QUERY_FEATURES=QFEATURES)
V405=types.SimpleNamespace(normalize_tau=normalize_tau,distinct_queries=distinct_queries,build_intervals=build_intervals)
V403=types.SimpleNamespace(rebuild=rebuild,NINE=NINE)
E_COLS=list(dict.fromkeys(EVENT_COLS+['event_end','concept:name']+[x for x in EVENT_FEATURES if x not in NINE]))

def build_inputs(raw_e,raw_q):
 selected,cohort=build_cohort(raw_q,raw_e,V405)
 selected=selected.sort_values([SEG,QTS,K],kind='stable').reset_index(drop=True)
 q=selected[KEYS+['t0','station']].copy()
 e=normalized_event_inputs(raw_e[list(dict.fromkeys([SEG,TS,'event_end','category','concept:name']+[x for x in EVENT_FEATURES if x not in NINE]))],TEACHER,NINE)
 features=reconstruct(e,q)
 need(list(features)==OBS19,'OBS19_CONTRACT_CHANGED')
 for name in OBS19:q[name]=features[name]
 ep,q=prefix_arrays(e,q,V403,TEACHER)
 q=q.sort_values([SEG,QTS,K],kind='stable').reset_index(drop=True)
 need(not {'y_next_delay','t_next','next_archived_plan','next_static_delay'}.intersection(q),'OUTCOME_IN_INPUT')
 need(np.isfinite(q[OBS19].to_numpy(float)).all() and np.isfinite(ep[EVENT_FEATURES].to_numpy(float)).all(),'NONFINITE_INPUT')
 return ep,q,cohort


def signlog(x):
    return np.sign(x)*np.log1p(np.abs(x))

class Queries(Dataset):
    def __init__(self,index,ev_numeric,q_numeric,ev_cat,max_len):
        self.keys=index[KEYS].reset_index(drop=True)
        self.end=index.last_event_global_index.to_numpy(dtype=np.int64)
        self.count=index.observed_event_count.to_numpy(dtype=np.int64)
        self.y=index.y_next_delay.to_numpy(dtype=np.float32)
        self.w=index.selected_weight.to_numpy(dtype=np.float32)
        self.evnum=ev_numeric;self.qnum=q_numeric;self.evcat=ev_cat
        self.max_len=max_len
    def __len__(self):return len(self.end)
    def __getitem__(self,idx):
        end=int(self.end[idx]);k=min(int(self.count[idx]),self.max_len-1)
        first=end-k+1
        cats=np.full(self.max_len,PAD,dtype=np.int64)
        nums=np.zeros((self.max_len,26),dtype=np.float32)
        valid=np.zeros(self.max_len,dtype=bool)
        cats[-k-1:-1]=self.evcat[first:end+1]
        nums[-k-1:-1,:23]=self.evnum[first:end+1]
        cats[-1]=QUERY
        nums[-1,23:25]=self.qnum[idx]
        nums[-1,25]=1.
        valid[-k-1:]=True
        return (torch.from_numpy(cats),torch.from_numpy(nums),torch.from_numpy(valid),
                torch.tensor(self.y[idx]),torch.tensor(self.w[idx]))

class Teacher(nn.Module):
    def __init__(self,vocab_size,d_model=192,nhead=4,layers=2,dropout=.1,max_len=100):
        super().__init__()
        require(d_model%nhead==0,'d_model must be divisible by nhead')
        self.cat=nn.Embedding(vocab_size,64,padding_idx=PAD)
        self.fuse=nn.Linear(64+26,d_model)
        self.pos=nn.Embedding(max_len,d_model)
        encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,
                         dim_feedforward=2*d_model,dropout=dropout,activation='gelu',
                         batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=layers,
                                          enable_nested_tensor=False)
        self.head=nn.Linear(d_model,1)
    def forward(self,cat,num,valid):
        b,t=cat.shape
        require(bool(valid[:,-1].all()),'Query token must be valid in every sequence')
        pos=torch.arange(t,device=cat.device).unsqueeze(0)
        inp=torch.cat((self.cat(cat),num),dim=-1)
        x=self.fuse(inp)+self.pos(pos)
        h=self.encoder(x,src_key_padding_mask=~valid)
        emb=h[:,-1,:]
        return self.head(emb).squeeze(-1),emb


def apply_frozen_transform(e,q,pre,teacher,expected_fit_cases):
    need(pre['event_feature_names']==teacher.EVENT_FEATURES,'Event channel contract changed')
    need(pre['query_feature_names']==teacher.QUERY_FEATURES,'Query channel contract changed')
    need(isinstance(expected_fit_cases,int) and expected_fit_cases>0,
         'Fit-only scaler count must be verified from the pinned inner case manifest')
    need(pre['num_teacher_inner_fit_cases']==expected_fit_cases,
         f'Fitted preprocessing case count does not match inner manifest: '
         f"saved={pre['num_teacher_inner_fit_cases']} manifest={expected_fit_cases}")
    need(pre['numeric_fitted_only_on']=='teacher_weights_fit' and
         pre['vocab_fitted_only_on']=='teacher_weights_fit','Fitted transforms provenance changed')
    z=teacher.signlog(e[teacher.EVENT_FEATURES].to_numpy(np.float32))
    mu=np.asarray(pre['event_mean'],dtype=np.float32);sd=np.asarray(pre['event_std'],dtype=np.float32)
    need(z.shape[1]==len(mu)==len(sd)==23 and np.all(sd>0),'Bad event scaler contract')
    evnum=np.clip((z-mu)/sd,-12.,12.).astype(np.float32)
    qraw=np.stack([q[n].to_numpy(np.float32) for n in teacher.QUERY_FEATURES[:2]],axis=1)
    need(np.isfinite(qraw).all() and (qraw>=0).all(),'Invalid query time inputs')
    qm=np.asarray(pre['query_log1p_mean'],dtype=np.float32)
    qs=np.asarray(pre['query_log1p_std'],dtype=np.float32)
    need(qm.shape==(2,) and qs.shape==(2,) and np.all(qs>0),'Bad query scaler')
    qnum=np.clip((np.log1p(qraw)-qm)/qs,-12.,12.).astype(np.float32)
    vocabulary=pre['vocab']; need(len(vocabulary)==pre['vocab_size'],'Vocab size changed')
    need(all(vocabulary.get(s)==i for i,s in enumerate(teacher.SPECIAL)),
         'Special token vocabulary differs from original teacher')
    evcat=np.array([vocabulary.get(s,teacher.UNK) for s in e['concept:name'].astype(str)],dtype=np.int32)
    need(np.isfinite(evnum).all() and np.isfinite(qnum).all() and evcat.min()>=0 and
         evcat.max()<len(vocabulary),'Nonfinite/out-of-range transformed inputs')
    return evnum,qnum,evcat

def teacher_infer(teacher,model,arm,q,evnum,qnum,evcat,batch,device,max_len):
    # The Queries dataset needs outcome/weight fields for its return signature,
    # but teacher inputs depend ONLY on cat,num,valid. We pass constant dummy
    # labels and masses; real labels are loaded strictly AFTER scoring.
    clean=q.copy()
    clean['y_next_delay']=0.;clean['selected_weight']=1.
    ds=teacher.Queries(clean,evnum,qnum,evcat,max_len)
    dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,
                  pin_memory=device.type=='cuda')
    logits=[];embs=[]
    with torch.inference_mode():
        for cat,num,valid,_dummy_y,_dummy_w in dl:
            if arm=='binary_control':l,h=model(cat.to(device),num.to(device),valid.to(device))
            else:l,h=model.base(cat.to(device),num.to(device),valid.to(device))
            logits.append(l.detach().cpu().to(torch.float32).numpy())
            embs.append(h.detach().cpu().to(torch.float32).numpy())
    logit=np.concatenate(logits);embedding=np.concatenate(embs)
    need(len(logit)==len(q) and embedding.shape==(len(q),model.base.head.in_features if arm=='multitask' else model.head.in_features),
         'Teacher batch output dimension mismatch')
    need(np.isfinite(logit).all() and np.isfinite(embedding).all(),'Nonfinite OOS teacher prediction')
    return logit,embedding


class SnapshotLSTM(nn.Module):
    """Prespecified T=1 three-layer LSTM; NOT event-history sequence learning."""
    def __init__(self,dim=19,hidden=128,dropout=.2):
        super().__init__()
        self.rnn=nn.LSTM(dim,hidden,num_layers=3,batch_first=True,dropout=dropout)
        self.head=nn.Sequential(nn.Linear(hidden,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):
        check(x.ndim==2 and x.shape[1]==19,'Invalid T=1 feature input')
        _,(h,_)=self.rnn(x[:,None,:])
        return self.head(h[-1]).squeeze(-1)

class SequenceLSTM(nn.Module):
    """True variable event prefixes; query-specific token is last valid step."""
    def __init__(self,vocab_size,hidden=128,embed_dim=32,dropout=.2):
        super().__init__()
        self.cat=nn.Embedding(vocab_size,embed_dim,padding_idx=bridge.PAD)
        self.rnn=nn.LSTM(embed_dim+26,hidden,num_layers=3,dropout=dropout,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,cat,num,valid):
        check(cat.ndim==2 and num.shape[:2]==cat.shape and num.shape[2]==26
              and valid.shape==cat.shape,'Invalid query LSTM inputs')
        check(bool(valid[:,-1].all()),'Missing last-position τ query token')
        lengths=valid.sum(1).to(dtype=torch.long)
        t=cat.shape[1]
        # The audited bridge right-aligns valid prefix+query. Left-shift before
        # packing, to prevent any artificial padding recurrence before events.
        offsets=(t-lengths).unsqueeze(1)
        source=(torch.arange(t,device=cat.device).unsqueeze(0)+offsets).clamp(max=t-1)
        ids=cat.gather(1,source)
        nums=num.gather(1,source.unsqueeze(-1).expand(-1,-1,num.shape[-1]))
        positions=torch.arange(t,device=cat.device).unsqueeze(0)
        active=positions<lengths.unsqueeze(1)
        ids=ids.masked_fill(~active,bridge.PAD)
        nums=nums.masked_fill(~active.unsqueeze(-1),0.)
        # The final valid token is exactly the query; padding cannot enter h[-1].
        terminal=ids.gather(1,(lengths-1).view(-1,1)).squeeze(1)
        check(bool((terminal==bridge.QUERY).all()),'Terminal LSTM token is not τ query')
        x=torch.cat((self.cat(ids),nums),dim=-1)
        packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(hn,_)=self.rnn(packed)
        return self.head(hn[-1]).squeeze(-1)

def loader_sequence(df,evnum,qnum,evcat,max_len,batch,shuffle):
    # qnum rows are aligned to complete bridge q; _pos makes subset exact.
    pos=df['_pos'].to_numpy(dtype=np.int64)
    ds=bridge.Queries(df,evnum,qnum[pos],evcat,max_len)
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,
                      pin_memory=torch.cuda.is_available())


def transform(frame,scaler,features):
    require(list(features)==scaler['features'],'Wrong feature order for saved scaler')
    raw=frame[list(features)].to_numpy(dtype=np.float32)
    require(raw.ndim==2 and raw.shape[1]==19 and np.isfinite(raw).all(),
            'Nonfinite snapshot input feature values')
    raw=np.sign(raw)*np.log1p(np.abs(raw))
    mu=np.asarray(scaler['mu'],dtype=np.float32)
    sd=np.asarray(scaler['sd'],dtype=np.float32)
    out=np.clip((raw-mu)/sd,-12.,12.).astype(np.float32)
    require(out.shape==(len(frame),19) and np.isfinite(out).all(),
            'Invalid final snapshot normalized inputs')
    return out

def snapshot_infer(model,features,scaler,cols,device):
    x=transform(features,scaler,cols)
    logits=[]
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(x),BATCH):
            batch=torch.from_numpy(x[i:i+BATCH]).to(device)
            z=model(batch)
            require(z.ndim==1 and z.shape[0]==len(batch),'Snapshot inference shape drift')
            logits.append(z.detach().float().cpu().numpy())
    v=np.concatenate(logits)
    # Raw sigmoid probabilities, before P4 calibration.
    raw=1/(1+np.exp(-np.clip(v.astype(np.float64),-60.,60.)))
    require(len(raw)==len(features) and np.isfinite(raw).all() and
            ((raw>0)&(raw<1)).all(),'Invalid raw LSTM probabilities')
    return raw


BASE=[
 'starts_5m_event_end','burstiness_cv_5m_event_end',
 'starts_15m_event_end','burstiness_cv_15m_event_end',
 'starts_60m_event_end','burstiness_cv_60m_event_end',
 'new_cases_1h','active_cases_1h','ewma_if_event_end','momentum_if_event_end',
 'observed_transport_5m_event_end','observed_transport_15m_event_end',
 'observed_transport_60m_event_end','observed_transport_global_5m',
 'observed_transport_global_15m','observed_transport_global_60m',
 'observed_transport_count_case_at_tau',
 'seconds_since_last_observed_transport','seconds_since_observed_transport_t0',
]

EMBED=[f'teacher_embed_{i:03d}' for i in range(192)]

PC=[f'teacher_pca_{i:02d}' for i in range(48)]

ARMS={
 'primary_soft_full': BASE+['teacher_logit']+PC,
 'full_hard': BASE+['teacher_logit']+PC,
 'pca48_hard': BASE+PC,
 'logit_hard': BASE+['teacher_logit'],
 'observed_only_hard': BASE,
}

def probabilities(scores):
    p=np.asarray(scores,dtype=np.float64)
    ensure(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probability')
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

def platt_predict(model,raw):
    out=model.predict_proba(probabilities(raw))[:,1]
    ensure(np.isfinite(out).all() and ((out>=0)&(out<=1)).all(),'Nonfinite calibrated score')
    return out

def xgb_native_probability(model, x):
    xx=np.asarray(x,dtype=np.float32)
    probs=model.predict(xgb.DMatrix(xx))
    ensure(probs.shape==(len(xx),) and np.isfinite(probs).all() and
           np.all((probs>=0)&(probs<=1)), 'Invalid XGBoost native prediction')
    return probs


def make_frame_with_pcs(frame,pca,v):
    out=frame[v.BASE+['teacher_logit']].copy()
    pcs=pca.transform(frame[v.EMBED].to_numpy(np.float32))
    require(np.isfinite(pcs).all(),'Teacher PCA transform generated nonfinite values')
    for i in range(pcs.shape[1]):
        out[v.PC[i]]=pcs[:,i].astype(np.float32)
    return out


def calibrated_raw_lr(probs):
    raw=np.asarray(probs,dtype=np.float64)
    check(raw.ndim==1 and np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all(),'Invalid raw probabilities')
    v=np.clip(raw,EPS,1-EPS)
    return np.log(v)-np.log1p(-v)

def apply_calibrator(calibrator,raw):
    x=calibrated_raw_lr(raw)
    p=calibrator.predict_proba(x.reshape(-1,1))[:,1]
    check(len(p)==len(x) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),
          'Invalid calibrated probability')
    return p.astype(np.float64)


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


# All model inputs remain label-free; dummy fields are only dataset-signature placeholders.
bridge=types.SimpleNamespace(PAD=PAD,QUERY=QUERY,Queries=Queries)
teacher=types.SimpleNamespace(EVENT_FEATURES=EVENT_FEATURES,QUERY_FEATURES=QFEATURES,
    SPECIAL=SPECIAL,UNK=UNK,signlog=signlog,Queries=Queries)
features=types.SimpleNamespace(BASE=BASE,EMBED=EMBED,PC=PC)
ensure=require
THRESHOLDS={'primary_soft_full':(.24,.35),'observed_only_hard':(.36,.77),
 'catboost_observed19_hard':(.30,.76),'logit_hard':(.31,.38),
 'pca48_hard':(.27,.44),'full_hard':(.28,.47),'standalone_transformer':(.23,.41),
 'sequence_lstm':(.26,.42),'snapshot_lstm':(.36,.67)}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def sequence_forward(model,q,evnum,qnum,evcat,device):
 clean=q.copy();clean['_pos']=np.arange(len(clean),dtype=np.int64)
 clean['y_next_delay']=0.;clean['selected_weight']=1.
 loader=loader_sequence(clean,evnum,qnum,evcat,100,128,False)
 model.eval(); logits=[]
 with torch.no_grad():
  for cat,num,valid,_,__ in loader:
   z=model(cat.to(device),num.to(device),valid.to(device))
   logits.append(z.detach().float().cpu().numpy())
 return torch.sigmoid(torch.from_numpy(np.concatenate(logits))).numpy().astype(np.float64)

def run(bundle,device):
 manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf8'))
 need(manifest['version']==1,'MANIFEST_VERSION')
 def asset(name):
  need(name in manifest['files'] and Path(name).name==name,'ASSET_NAME')
  p=bundle/name;need(p.is_file() and sha(p)==manifest['files'][name],'ASSET_HASH');return p
 def j(name):return json.loads(asset(name).read_text(encoding='utf8'))
 def load(name):return joblib.load(asset(name)) # Load only reviewed, trusted, hash-verified assets.
 raw_e=pd.read_parquet(asset('historical_test_events.parquet'),columns=E_COLS)
 raw_q=pd.read_parquet(asset('historical_test_queries.parquet'),columns=TAU_COLS)
 e,q,co=build_inputs(raw_e,raw_q);need(co==EXPECTED_COHORT and len(e)==64897,'COHORT')
 selected,_=build_cohort(raw_q,raw_e,V405)
 truth=q[KEYS].merge(selected[KEYS+['next_static_delay']],on=KEYS,validate='one_to_one',sort=False)
 need(truth[KEYS].equals(q[KEYS]),'TRUTH_ORDER')
 y=truth.next_static_delay.to_numpy(np.int8)
 del raw_e,raw_q,selected,truth
 pre=j('teacher_preprocessing_contract.json');sp=j('sequence_lstm_preprocessing_contract.json')
 evnum,qnum,evcat=apply_frozen_transform(e,q,pre,teacher,14272)
 need((evcat!=UNK).all(),'UNKNOWN_TEACHER_TOKEN')
 ck=torch.load(asset('teacher.pt'),map_location='cpu',weights_only=True)
 need(ck['version']=='v4.194' and ck['arm']=='binary_control','TEACHER_CHECKPOINT')
 cfg=ck['teacher_cfg'];need(cfg['vocab_size']==pre['vocab_size'],'VOCAB_SIZE')
 net=Teacher(**cfg);net.load_state_dict(ck['model_state'],strict=True);net.to(device).eval()
 logits,emb=teacher_infer(teacher,net,'binary_control',q,evnum,qnum,evcat,128,device,cfg['max_len'])
 f=q.copy();f['teacher_logit']=logits.astype('float32')
 f[EMBED]=pd.DataFrame(emb.astype('float32'),index=f.index,columns=EMBED)
 expanded=make_frame_with_pcs(f,load('pca.joblib'),features)
 scores={}
 for arm in list(THRESHOLDS)[:6]:
  cols=BASE if arm=='catboost_observed19_hard' else ARMS[arm]
  need(manifest['feature_orders'][arm]==cols,'FEATURE_ORDER')
  if arm=='catboost_observed19_hard':
   model=CatBoostClassifier();model.load_model(str(asset(arm+'.model')))
   raw=model.predict_proba(q[cols])[:,1]
  else:
   model=xgb.Booster();model.load_model(str(asset(arm+'.json')))
   raw=xgb_native_probability(model,expanded[cols])
  scores[arm]=platt_predict(load(arm+'_calibrator.joblib'),raw)
 scores['standalone_transformer']=load('standalone_transformer_calibrator.joblib').predict_proba(logits.astype(np.float64).reshape(-1,1))[:,1].astype(np.float64)
 snum,sqnum,scat=apply_frozen_transform(e,q,sp,teacher,14272);need((scat!=UNK).all(),'UNKNOWN_SEQUENCE_TOKEN')
 ck=torch.load(asset('sequence_lstm.pt'),map_location='cpu',weights_only=True)
 need(ck['version']=='v4.124' and ck['kind']=='sequence' and ck['hidden']==128 and ck['max_len']==100 and ck['vocab_size']==sp['vocab_size'],'SEQUENCE_CHECKPOINT')
 net=SequenceLSTM(ck['vocab_size'],hidden=128,dropout=.2);net.load_state_dict(ck['state_dict'],strict=True);net.to(device).eval()
 raw=sequence_forward(net,q,snum,sqnum,scat,device)
 scores['sequence_lstm']=apply_calibrator(load('sequence_lstm_calibrator.joblib'),raw)
 ck=torch.load(asset('snapshot_lstm.pt'),map_location='cpu',weights_only=True)
 need(ck['kind']=='snapshot19' and ck['train_role']=='P0-P3 strictly predeadline','SNAPSHOT_CHECKPOINT')
 net=SnapshotLSTM(dim=19,hidden=128,dropout=.2);net.load_state_dict(ck['state_dict'],strict=True);net.to(device).eval()
 raw=snapshot_infer(net,q,j('snapshot_scaler.json'),BASE,device)
 scores['snapshot_lstm']=apply_calibrator(load('snapshot_lstm_calibrator.joblib'),raw)
 for n,z in scores.items():need(z.shape==(len(q),) and np.isfinite(z).all() and ((z>=0)&(z<=1)).all(),'PROBABILITIES')
 metrics={n:{k:row_metrics(y,z,t) for k,t in zip(('primary','D5'),THRESHOLDS[n])} for n,z in scores.items()}
 return q[KEYS].copy(),y,scores,{'cohort':co,'metrics':metrics,'device':str(device),'native_pseudonym_order':True,'batch_size':128,'torch':torch.__version__}

def selftest():
 # Full reconstruction on fabricated cases, then real neural forward methods on fabricated tensors.
 t=pd.Timestamp('2023-01-02T00:00:00Z');rows=[];queries=[]
 for i,c in enumerate(['c1','c2']):
  for j in range(3):
   row={SEG:c,TS:t+pd.Timedelta(minutes=j*10+i),'event_end':'s1','category':'transport','concept:name':'act','planned:timestamp':t+pd.Timedelta(minutes=j*10+i+2),'is_delayed':int(i==1 and j==1),'is_delayed_dyn':int(i==1 and j==1)}
   row.update({x:float(j) for x in EVENT_FEATURES if x not in NINE});rows.append(row)
  t0=t+pd.Timedelta(minutes=i);queries.append({SEG:c,K:0,QTS:t0+pd.Timedelta(minutes=5),'t0':t0,'t_next':t0+pd.Timedelta(minutes=10),'station':'s1','y_next_delay':i,'has_post':1,'events_since_t0':1,'time_since_t0_s':300})
 e,q,co=build_inputs(pd.DataFrame(rows),pd.DataFrame(queries));need(len(q)==2 and len(e)==2,'TEST_RECONSTRUCTION')
 torch.manual_seed(42);dev=torch.device('cpu');ev=np.zeros((len(e),23),np.float32);qn=np.zeros((len(q),2),np.float32);cat=np.full(len(e),3,np.int32)
 m=Teacher(4,d_model=192,nhead=4,layers=2,max_len=100).eval();l,h=teacher_infer(teacher,m,'binary_control',q,ev,qn,cat,128,dev,100);need(h.shape==(2,192),'TEST_TEACHER')
 m=SequenceLSTM(4).eval();z=sequence_forward(m,q,ev,qn,cat,dev);need(z.shape==(2,) and np.isfinite(z).all(),'TEST_SEQUENCE')
 m=SnapshotLSTM().eval();z=snapshot_infer(m,q,{'features':BASE,'mu':[0.]*19,'sd':[1.]*19},BASE,dev);need(z.shape==(2,),'TEST_SNAPSHOT')
 print('Synthetic reconstruction and neural forward tests passed.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=Path('.'));p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--self-test',action='store_true');p.add_argument('--metrics-out',type=Path);p.add_argument('--private-scores-out',type=Path,help='Optional local validation output; never publish prediction rows.');a=p.parse_args()
 if a.self_test:selftest();return
 need(a.device!='cuda' or torch.cuda.is_available(),'CUDA_UNAVAILABLE')
 q,y,scores,report=run(a.bundle.resolve(),torch.device(a.device))
 if a.metrics_out:
  with a.metrics_out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
 else:print(json.dumps(report,indent=2,allow_nan=False))
 if a.private_scores_out:
  q['next_static_delay']=y
  for n,z in scores.items():q['prob_'+n]=z
  need(not a.private_scores_out.exists(),'OUTPUT_EXISTS');q.to_parquet(a.private_scores_out,index=False)
if __name__=='__main__':main()

# -*- coding: utf-8 -*-
"""
Event-log preprocessing and tau-table construction for the research-release dataset.

This module derives leak-controlled temporal, congestion, routing, and tau-level
features from anonymized event logs. It is intended for use with the public
research-release data layout and does not contain internal paths or identifiers.
"""
import os
import pandas as pd
import numpy as np
import joblib
import gc
import time
from collections import deque, Counter


from data_loader import EventLogLoader

pd.options.mode.copy_on_write = True

LOCAL_TIMEZONE = "Europe/Berlin"  # change if the anonymized dataset uses a different local timezone


def _to_utc_floor_min(s: pd.Series) -> pd.Series:
    # robust: coerces to tz-aware UTC and floors to minute
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.floor("min")


def _safe_div(a, b, eps=1e-6):
    b = np.where(np.asarray(b) == 0, eps, b)
    return np.asarray(a) / b

def _ensure_event_end(df: pd.DataFrame) -> pd.DataFrame:
    if 'event_end' not in df.columns:
        if 'event:end' in df.columns:
            df['event_end'] = df['event:end']
        else:
            raise KeyError("Neither 'event_end' nor 'event:end' column is present.")
    return df


class EventLogPreprocessor:
    def __init__(
        self,
        filter_activities=None,
        drop_na=True,
        cluster_model_path="model_artifacts/kmeans_case_profile.joblib",
        scaler_path="model_artifacts/scaler_case_profile.joblib",
    ):
        self.filter_activities = filter_activities
        self.drop_na = drop_na
        self.cluster_model_path = cluster_model_path
        self.scaler_path = scaler_path
        self._kmeans = None
        self._scaler = None
        self.cid = 'case:concept:name'

    @property
    def kmeans(self):
        if self._kmeans is None:
            self._kmeans = joblib.load(self.cluster_model_path)
        return self._kmeans

    @property
    def scaler(self):
        if self._scaler is None:
            self._scaler = joblib.load(self.scaler_path)
        return self._scaler
    
    def attach_t0_static_context(self, events_df: pd.DataFrame, tau_df: pd.DataFrame, cols=None) -> pd.DataFrame:
        """
        For each (case_seg, k_idx) transport interval, copy static features from the
        transport row at t0 (i.e., the row whose category == 'transport' and k_idx matches).
        """
        cid = self.cid
        g = events_df.sort_values([cid, 'time:timestamp']).copy()
    
        # derive k_idx in the same way as add_tau_congestion_features
        g['k_idx'] = g.groupby(cid)['category'].transform(lambda s: (s.eq('transport')).cumsum() - 1)
    
        # default columns to carry
        default_cols = [
            'route','case:start','case:end','subsystemtype', 'event_route','event_route_code',
            'vehicle_id','sachnummer','start_code','end_code','cluster_id'
        ]
        use_cols = default_cols if (cols is None) else [c for c in cols if c in g.columns]
    
        # the t0 transport rows
        t0_ctx = (g[g['category'].eq('transport')]
                  [[cid, 'k_idx'] + use_cols]
                  .drop_duplicates([cid, 'k_idx'], keep='last'))
    
        out = tau_df.merge(t0_ctx, on=[cid, 'k_idx'], how='left')
        return out

    def attach_teacher_distillation(self, tau_df: pd.DataFrame, teacher_outputs: pd.DataFrame) -> pd.DataFrame:
        """
        Merge teacher logits/embeddings at τ into tau_df.
        Expected teacher_outputs columns: [self.cid, 'k_idx', 'tau_ts', 'teacher_logit', 'emb_pca_0'..]
        """
        cid = self.cid
        need = [cid, 'k_idx', 'tau_ts']
        if not all(c in teacher_outputs.columns for c in need):
            raise ValueError("teacher_outputs must include columns: {case_id, k_idx, tau_ts}")
    
        return tau_df.merge(teacher_outputs, on=[cid, 'k_idx', 'tau_ts'], how='left')


    def add_case_arrival_and_active_cases(self, df: pd.DataFrame, anchor_col: str = "anchor_ts") -> pd.DataFrame:
        ONE_H = np.timedelta64(1, 'h')
        cid   = self.cid

        # robust anchor -> fallback to event time if missing
        anchor = pd.to_datetime(df[anchor_col], utc=True, errors='coerce')
        if 'time:timestamp' in df.columns:
            fallback = pd.to_datetime(df['time:timestamp'], utc=True, errors='coerce')
            anchor = anchor.fillna(fallback)
        avals = anchor.values.astype('datetime64[ns]')


        # choose a base time column usable both for event rows and τ rows
        base_time_col = 'time:timestamp' if 'time:timestamp' in df.columns else anchor_col
        
        case_first_ts = (
            pd.to_datetime(
                df.groupby(cid)[base_time_col].transform('min'),
                utc=True, errors='coerce'
            ).values.astype('datetime64[ns]')
        )

        unique_first_sorted = np.sort(np.unique(case_first_ts))
        end_idx   = np.searchsorted(unique_first_sorted, avals,        side='right')
        start_idx = np.searchsorted(unique_first_sorted, avals - ONE_H, side='right')
        df['new_cases_1h'] = (end_idx - start_idx).astype(np.int32)

        # active cases in last hour
        ids          = df[cid].values
        order        = np.argsort(avals)
        t_sorted     = avals[order]
        ids_sorted   = ids[order]

        window, cnt  = deque(), Counter()
        active_sorted = np.empty(len(df), dtype=np.int32)
        for i, t in enumerate(t_sorted):
            window.append(i); cnt[ids_sorted[i]] += 1
            cutoff = t - ONE_H
            while window and t_sorted[window[0]] < cutoff:
                j = window.popleft()
                cnt[ids_sorted[j]] -= 1
                if cnt[ids_sorted[j]] == 0:
                    del cnt[ids_sorted[j]]
            active_sorted[i] = len(cnt)

        active1h = np.empty(len(df), dtype=np.int32)
        active1h[order] = active_sorted
        df['active_cases_1h'] = active1h
        return df


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        tic = time.time()
        def log(msg):
            nonlocal tic
            now = time.time()
            print(f"[transform] {msg} ({now - tic:.1f}s)", flush=True)
            tic = now
    
        print(">>> entered transform()", flush=True)   # fires as soon as function starts
        log("start")
        
        # 2) core cols + optional filter
        core = [
            'case:concept:name','event_id', 'concept:name',"subsystemtype",'sachnummer',
            'time:timestamp','planned:timestamp','case:start','case:end',
            'event:start','event:end','eventkey'
        ]
        
        optional = [
            'delay_s','is_delayed',
            'has_prev_start','has_prev_finish',
            'prev_start_dst_filled','prev_finish_dst_filled',
            'secs_since_prev_start_raw','secs_since_prev_finish_raw',
            'prev_finish_matches_src',
            'has_last_problem','secs_since_last_problem','has_last_topprio','secs_since_last_topprio',
            'hour_of_day_utc','weekday_utc',
            'vehicle_id',
            'category',  # <- keep category
        ]
        keep = [c for c in (core + optional) if c in df.columns]
        df = df[keep].copy()
        
        # normalize common string columns
        for c in ('event:start', 'event:end', 'case:start', 'case:end'):
            if c in df.columns:
                df[c] = (
                    df[c].astype('string')
                         .str.strip()
                         .str.upper()
                         .replace({'': pd.NA})
                )
        
        # normalize vehicle_id (don’t uppercase, it can be case-sensitive)
        if 'vehicle_id' in df.columns:
            df['vehicle_id'] = (
                df['vehicle_id'].astype('string')
                                .str.strip()
                                .replace({'': pd.NA})
            ).astype('category')
        
        # normalize category (lowercase; optional to keep as string)
        if 'category' in df.columns:
            df['category'] = (
                df['category'].astype('string')
                              .str.strip()
                              .str.lower()
                              .replace({'': pd.NA})
            )



        # --- Build anchor time (strict, leak-safe): use actual event time ---
        df['anchor_ts'] = pd.to_datetime(df['time:timestamp'], utc=True, errors='coerce')

    
        if self.filter_activities:
            df = df[df['concept:name'].isin(self.filter_activities)]
    
        # 3) drop NA timestamps
        if self.drop_na:
            bad = df.loc[
                df['time:timestamp'].isna() | df['planned:timestamp'].isna(),
                'case:concept:name'
            ].unique()
            if bad.size:
                df = df[~df['case:concept:name'].isin(bad)]
    
        # 4) sort and static deltas & label
        df = df.sort_values(['case:concept:name','time:timestamp']).reset_index(drop=True)
        
        # ── drop consecutive duplicates by event_id within each case ──
        df = df.loc[
            ~(df['event_id'] == df.groupby('case:concept:name')['event_id'].shift(1))
        ].reset_index(drop=True)

        
        # --- Segment-by-time-window inside each case ---
        cid = self.cid  # 'case:concept:name'
        TIME_WINDOW = pd.Timedelta(hours=168)
        
        # gap to previous event within the same case
        gap = df.groupby(cid)['time:timestamp'].diff()
        
        # start a new segment if:
        #  - it's the first event in the case (gap is NaT),
        #  - time goes backwards (negative gap),
        #  - gap exceeds the window
        new_seg = gap.isna() | (gap < pd.Timedelta(0)) | (gap > TIME_WINDOW)
        
        # monotone segment id per (possibly reused) case id
        df['segment_id'] = new_seg.groupby(df[cid]).cumsum().astype('int32')
        
        df['case_seg'] = df['case:concept:name'].astype(str) + '||SEG=' + df['segment_id'].astype(str)

        # make all downstream groupbys use segments as cases
        self.cid = 'case_seg'
        
        # from now on, group by both cid + segment
        gkey = [cid, 'segment_id']
        
        # continuity check: destination of current == source of next, within segment
        df['next_station'] = df.groupby(gkey)['event:start'].shift(-1)
        
        # Now the categorical alignment works
        vals_event_end = df['event:end'].astype('string')
        vals_next_stn  = df['next_station'].astype('string')
        
        cats = pd.Index(vals_event_end.dropna().unique()).union(
               pd.Index(vals_next_stn.dropna().unique())
        )
        
        df['event:end']    = pd.Categorical(vals_event_end, categories=cats)
        df['next_station'] = pd.Categorical(vals_next_stn,  categories=cats)
        df = _ensure_event_end(df)  # create alias early and reliably
        
        # stable alias used everywhere downstream
        df['event_end'] = df['event:end']


        # Base computation
        df['delay'] = (df['time:timestamp'] - df['planned:timestamp']).dt.total_seconds().astype(np.float32)
        df['is_delayed'] = (df['delay'] > 0).astype(np.int8)
        
        # Prefer upstream values if provided (to stay consistent with SQL extraction)
        if 'delay_s' in df.columns:
            df['delay'] = pd.to_numeric(df['delay_s'], errors='coerce').astype(np.float32)
        if 'is_delayed' in df.columns:
            df['is_delayed'] = pd.to_numeric(df['is_delayed'], errors='coerce').fillna(0).astype(np.int8)

        df['time_since_start'] = (
            df.groupby(self.cid)['time:timestamp']
              .transform(lambda x: (x - x.min()).dt.total_seconds())
              .astype(np.float32)
        )
        df['inter_event_time'] = (
            df.groupby(self.cid)['time:timestamp']
              .diff().dt.total_seconds().fillna(0)
              .astype(np.float32)
        )


        # Normalize sentinel values for “unknown yet” intervals and build flags/filled
        SENT = 9_999_999
        for col in ['secs_since_last_problem','secs_since_last_topprio',
                    'secs_since_prev_start_raw','secs_since_prev_finish_raw']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Known/unknown mask (1 = known signal)
                known_flag = (col.replace('_raw','') + '_known') if col.endswith('_raw') else (col + '_known')
                df[known_flag] = (~df[col].isna() & (df[col] != SENT)).astype(np.int8)
                # NaN for unknown so rolling stats don’t get distorted
                df[col] = df[col].where(df[col] != SENT, np.nan)
                # Filled version for tree models that prefer numeric
                df[col + '_filled'] = df[col].fillna(SENT).astype(np.int32)


        # Handle FIRST_LEG markers and make numeric encodings
        for nm in ['prev_start_dst_filled','prev_finish_dst_filled']:
            if nm in df.columns:
                df[nm] = df[nm].astype('string').str.strip()
                flag = nm.replace('_dst_filled','_is_first_leg')
                df[flag] = (df[nm] == 'FIRST_LEG').astype(np.int8)
                code_col = nm + '_code'
                df[code_col], _ = pd.factorize(df[nm].fillna('∅'))
                df[code_col] = df[code_col].astype(np.int32)



        # ─── DYNAMIC‑PLANNED SCHEDULE ────────────────────────────────
        dyn_list = []
        for case, grp in df.groupby(self.cid, sort=False):
            prev_time = None
            prev_planned = None
            for t, p in zip(grp['time:timestamp'], grp['planned:timestamp']):
                if prev_time is None or prev_time <= prev_planned:
                    dyn = p
                else:
                    delta = p - prev_planned
                    dyn = prev_time + delta
                dyn_list.append(dyn)
                prev_time, prev_planned = t, p
        df['dynamic_planned'] = pd.to_datetime(dyn_list)

        # ─── DYNAMIC‑DELAY & FLAG ───────────────────────────────────
        df['dynamic_delay'] = (
            df['time:timestamp'] - df['dynamic_planned']
        ).dt.total_seconds().astype(np.float32)
        df['is_delayed_dyn'] = (df['dynamic_delay'] > 0).astype(np.int8)
        
        # ─── COMBINED ROUTE FEATURE ───────────────────────────
        df['route'] = df['case:start'].astype(str) + '→' + df['case:end'].astype(str)
        df['route_code'], _ = pd.factorize(df['route'])
        
        df['event_route'] = df['event:start'].astype(str) + '→' + df['event_end'].astype(str)
        df['event_route_code'], _ = pd.factorize(df['event_route'])

        # ─── LOCAL-TIME FEATURES (prefer UTC parts from SQL; else fallback to Berlin) ───
        if 'hour_of_day_utc' in df.columns:
            df['hour_of_day'] = pd.to_numeric(df['hour_of_day_utc'], errors='coerce') \
                                    .fillna(df['time:timestamp'].dt.hour) \
                                    .astype(np.int8)
        else:
            df['ts_local']    = df['time:timestamp'].dt.tz_convert(LOCAL_TIMEZONE)
            df['hour_of_day'] = df['ts_local'].dt.hour.astype(np.int8)
        
        if 'weekday_utc' in df.columns:
            wd = pd.to_numeric(df['weekday_utc'], errors='coerce') \
                    .fillna(df['time:timestamp'].dt.weekday) \
                    .astype(int)
        else:
            # fallback uses local time if we made ts_local above, else UTC weekday
            wd = (df['ts_local'].dt.weekday if 'ts_local' in df.columns else df['time:timestamp'].dt.weekday)
        
        wdf = pd.get_dummies(wd, prefix='weekday', dtype=np.int8)
        for i in range(7):
            col = f'weekday_{i}'
            if col not in wdf: wdf[col] = 0
        # Drop any existing weekday_ columns to avoid duplicates then concat fresh
        df = df.drop(columns=[c for c in df.columns if c.startswith('weekday_')], errors='ignore')
        df = pd.concat([df, wdf[[f'weekday_{i}' for i in range(7)]]], axis=1)
        
        # Shift & Schichtwechsel only when we have local time (optional)
        if 'ts_local' not in df.columns:
            df['ts_local'] = df['time:timestamp'].dt.tz_convert(LOCAL_TIMEZONE)
        
        def _shift(ts):
            m = ts.hour * 60 + ts.minute
            if   390 <= m <  870: return 'early_shift'
            elif 870 <= m < 1350: return 'late_shift'
            else:                 return 'night_shift'
        df['shift'] = df['ts_local'].map(_shift)
        df = pd.concat([df, pd.get_dummies(df['shift'], dtype=np.int8)], axis=1)
        
        mins = df['ts_local'].dt.hour*60 + df['ts_local'].dt.minute
        df['shift_change'] = (
            mins.between(360,420) |
            mins.between(840,900) |
            mins.between(1320,1380)
        ).astype(np.int8)
        
        df.drop(columns=['ts_local','shift'], inplace=True)


        # ================== FAST STATION WINDOWS (drop-in replacement) ==================
        # Inputs used: df[['event_end','time:timestamp','planned:timestamp','is_delayed','anchor_ts']]
        # Outputs (same names): starts_*, delayed_*, delaysum_*, delay_rate_*, avg_delay_*,
        # planned_arrivals_*, backlog_*, burstiness_cv_*  for each W in WINDOWS_MIN
        
        
        if df['event_end'].isna().all():
            for W in [5, 15, 60]:
                sfx = f"{W}m_event_end"
                df[f"starts_{sfx}"] = 0
                df[f"delayed_{sfx}"] = 0
                df[f"delaysum_{sfx}"] = 0.0
                df[f"delay_rate_{sfx}"] = 0.0
                df[f"avg_delay_{sfx}"] = 0.0
                df[f"planned_arrivals_{sfx}"] = 0
                df[f"backlog_{sfx}"] = 0
                df[f"burstiness_cv_{sfx}"] = 0.0
        else:
        
        
            WINDOWS_MIN = [5, 15, 60]  # adjust if needed
            
            # 1) Build per-minute actuals & planned (sparse)
            df = _ensure_event_end(df)
            dest = (df.dropna(subset=['event_end'])
                      [['event_end','time:timestamp','planned:timestamp','is_delayed']]
                      .copy())
            dest = dest.rename(columns={'event_end':'station','time:timestamp':'event_ts'})
            dest['ts_min'] = pd.to_datetime(dest['event_ts'].dt.floor('min'), utc=True)
            dest['delay_s'] = (dest['event_ts'] - dest['planned:timestamp']) \
                .dt.total_seconds().clip(lower=0).astype(np.float32)
            
            # per-minute actuals
            agg = (
                dest.groupby(['station','ts_min'], observed=True)
                    .agg(starts=('event_ts','count'),
                         delayed=('is_delayed','sum'),
                         delay_sum_s=('delay_s','sum'))
                    .reset_index()
                    .sort_values(['station','ts_min'])
            )
            
            # per-minute planned arrivals
            planned_min = (
                df.dropna(subset=['event_end'])
                  [['event_end','planned:timestamp']]
                  .rename(columns={'event_end':'station','planned:timestamp':'ts'})
                  .assign(ts_min=lambda x: pd.to_datetime(x['ts'].dt.floor('min'), utc=True))
                  .groupby(['station','ts_min'], observed=True).size()
                  .rename('planned').reset_index()
                  .sort_values(['station','ts_min'])
            )
            
            # anchor minute keys per row (unique to shrink joins)
            df['anchor_min'] = pd.to_datetime(df['anchor_ts'].dt.floor('min'), utc=True)
            
            row_keys = (df.dropna(subset=['event_end'])
                          [['event_end','anchor_min']]
                          .rename(columns={'event_end':'station','anchor_min':'ts_min'}))
            # ensure tz-aware UTC for join key
            row_keys['ts_min'] = pd.to_datetime(row_keys['ts_min'], utc=True)
            
            uq_query = row_keys.drop_duplicates(['station','ts_min']).sort_values(['station','ts_min'])
    
            
            # 2) Process per station (keeps memory bounded, uses pure numpy)
            out_chunks = {W: [] for W in WINDOWS_MIN}
            cv_chunks  = {W: [] for W in WINDOWS_MIN}  # burstiness
            now_chunks = []  # cumulatives at "now" (optional diagnostics)
            
            # helper to compute sample std (ddof=1) from window sums (kept for clarity; used in older variant)
            def _window_mean_std(sum1, sum2, n, ddof=1):
                mean = sum1 / np.maximum(n, 1)
                ex2  = sum2 / np.maximum(n, 1)
                var  = (ex2 - mean**2) * (np.maximum(n, 1) / np.maximum(n - ddof, 1))
                var  = np.where(n <= ddof, 0.0, var)
                std  = np.sqrt(np.maximum(var, 0.0))
                return mean, std
            
            # convert tz-aware timestamps to epoch minutes (int64)
            NS_PER_MIN = np.int64(60_000_000_000)
            def _to_epoch_min_int64(ts_series: pd.Series) -> np.ndarray:
                # works for tz-aware Datetime64[ns, tz]; view('i8') -> epoch ns
                return (ts_series.values.view('i8') // NS_PER_MIN).astype(np.int64)
            
            for st, g_now in uq_query.groupby('station', sort=False, observed=True):
                # minutes we need outputs for (unique anchor minutes for this station)
                t_need_min = _to_epoch_min_int64(g_now['ts_min'])
            
                # pull this station's per-minute actual and planned series
                g_a = agg[agg['station'].eq(st)][['ts_min','starts','delayed','delay_sum_s']]
                g_p = planned_min[planned_min['station'].eq(st)][['ts_min','planned']]
            
                if g_a.empty and g_p.empty:
                    for W in WINDOWS_MIN:
                        out_chunks[W].append(pd.DataFrame({
                            'station': st, 'ts_min': g_now['ts_min'].values,
                            f'starts_{W}': 0, f'delayed_{W}': 0,
                            f'delaysum_{W}': 0.0, f'planned_{W}': 0
                        }))
                        cv_chunks[W].append(pd.DataFrame({
                            'station': st, 'ts_min': g_now['ts_min'].values,
                            f'cv_{W}': np.zeros(len(g_now), dtype=np.float32)
                        }))
                    now_chunks.append(pd.DataFrame({
                        'station': st, 'ts_min': g_now['ts_min'].values,
                        'c_starts': 0, 'c_delayed': 0, 'c_delay_sum': 0.0, 'c_planned': 0
                    }))
                    continue
            
                # build dense minute index covering actual & planned & needed minutes (as int minutes)
                # --- keep this part unchanged ---
                mins_all = np.unique(np.concatenate([
                    _to_epoch_min_int64(g_a['ts_min']) if not g_a.empty else np.array([], dtype=np.int64),
                    _to_epoch_min_int64(g_p['ts_min']) if not g_p.empty else np.array([], dtype=np.int64),
                    t_need_min
                ]))
                M = len(mins_all)
                
                # dense arrays (zero fill)
                starts  = np.zeros(M, dtype=np.int32)
                delayed = np.zeros(M, dtype=np.int32)
                delays  = np.zeros(M, dtype=np.float32)
                planned = np.zeros(M, dtype=np.int32)
                
                # --- change these three mappings ---
                
                if not g_a.empty:
                    ia = np.searchsorted(mins_all, _to_epoch_min_int64(g_a['ts_min'])).astype(np.intp)
                    starts[ia]  = g_a['starts'].to_numpy(np.int32)
                    delayed[ia] = g_a['delayed'].to_numpy(np.int32)
                    delays[ia]  = g_a['delay_sum_s'].to_numpy(np.float32)
                
                if not g_p.empty:
                    ip = np.searchsorted(mins_all, _to_epoch_min_int64(g_p['ts_min'])).astype(np.intp)
                    planned[ip] = g_p['planned'].to_numpy(np.int32)
                
                # positions for needed minutes (map query timestamps to indices)
                pos = np.searchsorted(mins_all, t_need_min).astype(np.intp)
                pos = np.clip(pos, 0, M - 1)
    
            
                # cumulatives once
                c_starts    = np.cumsum(starts, dtype=np.int64)
                c_delayed   = np.cumsum(delayed, dtype=np.int64)
                c_delay_sum = np.cumsum(delays, dtype=np.float64)
                c_planned   = np.cumsum(planned, dtype=np.int64)
            
                # for CV: cumulative sums of x and x^2
                starts_f = starts.astype(np.float32)
                c1 = np.cumsum(starts_f, dtype=np.float64)
                c2 = np.cumsum((starts_f ** 2).astype(np.float64), dtype=np.float64)
            
                # positions for needed minutes
    
            
                # cumulative "now" at t (keep original tz-aware ts_min for joining)
                now_chunks.append(pd.DataFrame({
                    'station': st,
                    'ts_min':  g_now['ts_min'].values,
                    'c_starts':    c_starts[pos],
                    'c_delayed':   c_delayed[pos],
                    'c_delay_sum': c_delay_sum[pos],
                    'c_planned':   c_planned[pos],
                }))
            
                # window deltas via shifted positions (t-W) + CV
                for W in WINDOWS_MIN:
                    prev_pos = np.maximum(pos - W, -1)
                    cs_prev  = np.where(prev_pos < 0, 0,   c_starts[prev_pos])
                    cd_prev  = np.where(prev_pos < 0, 0,   c_delayed[prev_pos])
                    ds_prev  = np.where(prev_pos < 0, 0.0, c_delay_sum[prev_pos])
                    cp_prev  = np.where(prev_pos < 0, 0,   c_planned[prev_pos])
            
                    starts_W   = (c_starts[pos]    - cs_prev).astype(np.int32)
                    delayed_W  = (c_delayed[pos]   - cd_prev).astype(np.int32)
                    delaysum_W = (c_delay_sum[pos] - ds_prev).astype(np.float32)
                    planned_W  = (c_planned[pos]   - cp_prev).astype(np.int32)
            
                    out_chunks[W].append(pd.DataFrame({
                        'station': st,
                        'ts_min': pd.to_datetime(g_now['ts_min'].values, utc=True),
                        f'starts_{W}': starts_W, f'delayed_{W}': delayed_W,
                        f'delaysum_{W}': delaysum_W, f'planned_{W}': planned_W
                    }))
            
                    # rolling CV over W minutes using cumulative sums (ddof=1)
                    l = np.maximum(pos - W + 1, 0)
                    r = pos
                    n = (r - l + 1).astype(np.int64)
                    sum1 = c1[r] - np.where(l > 0, c1[l-1], 0.0)
                    sum2 = c2[r] - np.where(l > 0, c2[l-1], 0.0)
                    mean = sum1 / np.maximum(n, 1)
                    ex2  = sum2 / np.maximum(n, 1)
                    var  = (ex2 - mean**2) * (np.maximum(n, 1) / np.maximum(n - 1, 1))
                    var  = np.where(n <= 1, 0.0, var)
                    std  = np.sqrt(np.maximum(var, 0.0))
                    cv   = (std / np.maximum(mean, 1e-6)).astype(np.float32)
            
                    cv_chunks[W].append(pd.DataFrame({
                        'station': st,
                        'ts_min': pd.to_datetime(g_now['ts_min'].values, utc=True),
                        f'cv_{W}': cv
                    }))
            
            # 3) Stitch per-station results, map back to all rows, and create final columns
            for W in WINDOWS_MIN:
                f = pd.concat(out_chunks[W], ignore_index=True)
                c = pd.concat(cv_chunks[W],  ignore_index=True)
                sfx = f"{W}m_event_end"
            
                tmp = (
                    row_keys.merge(
                        f.rename(columns={
                            f'starts_{W}':'starts',
                            f'delayed_{W}':'delayed',
                            f'delaysum_{W}':'delaysum',
                            f'planned_{W}':'planned'
                        }),
                        on=['station','ts_min'], how='left'
                    )
                    .merge(c, on=['station','ts_min'], how='left')
                    .fillna({'starts':0, 'delayed':0, 'delaysum':0.0, 'planned':0, f'cv_{W}':0.0})
                )
            
                starts   = tmp['starts'].to_numpy(np.int32)
                delayed  = tmp['delayed'].to_numpy(np.int32)
                delaysum = tmp['delaysum'].to_numpy(np.float32)
                planned  = tmp['planned'].to_numpy(np.int32)
                cv_vals  = tmp[f'cv_{W}'].to_numpy(np.float32)
            
                df[f"starts_{sfx}"]     = starts
                df[f"delayed_{sfx}"]    = delayed
                df[f"delaysum_{sfx}"]   = delaysum
                df[f"delay_rate_{sfx}"] = np.divide(delayed, np.maximum(starts, 1), dtype=np.float32)
                df[f"avg_delay_{sfx}"]  = np.divide(delaysum, np.maximum(starts, 1), dtype=np.float32)
                df[f"planned_arrivals_{sfx}"] = planned
                df[f"backlog_{sfx}"]          = np.maximum(planned - starts, 0).astype(np.int32)
                df[f"burstiness_cv_{sfx}"]    = cv_vals
            
            # cleanup
            df.drop(columns=['anchor_min'], inplace=True, errors='ignore')
            del dest, agg, planned_min, row_keys, uq_query, out_chunks, cv_chunks, now_chunks
            gc.collect()
        # ================= END FAST STATION WINDOWS ======================

        # ================= END FAST STATION WINDOWS ======================
        
        
        log("station windows done")
        
        
        
        # === Next-station congestion (leakage-safe, ETA-aligned) ===================
        # Assumptions:
        # - Location for “next step” is the current row’s destination station B = df['event_end'].
        # - Anchor time t is df['anchor_ts'] (already built earlier; <= current actual time).
        # - We only use information available at or before t (actuals) and PLANNED info (future plans are OK).
        # - We will produce:
        #     backlog_next_now
        #     arrivals_next_15m, services_next_15m, pressure_next_15m
        #     arrivals_next_60m, services_next_60m, pressure_next_60m
        #     delta_pressure_next_15m_60m
        #     next_planned_start, slack_to_planned_start_next (seconds)
        #     planned_arrivals_to_start_next, planned_services_to_start_next, planned_pressure_to_start_next
        #
        # Drop-in position:
        # Place this block AFTER you created/standardized:
        #   - df['event_end']  (destination of current transport)
        #   - df['anchor_ts']  (anchor timestamp for features; tz-aware UTC)
        #   - df['planned:timestamp'] and df['time:timestamp'] are present
        # and BEFORE heavy downstream features (so later features can use these).
        # ===============================================================================
        
        # 0) Guard & helpers
        df = _ensure_event_end(df)
        if df['anchor_ts'].isna().any():
            df['anchor_ts'] = df['time:timestamp']  # safety fallback
        
        # Next event’s PLANNED start (ETA-aligned but leakage-safe, since it’s a plan)
        df['next_planned_start'] = df.groupby(self.cid)['planned:timestamp'].shift(-1)
        
        # Slack until planned start (seconds). If NaT (no next event), set to 0
        df['slack_to_planned_start_next'] = (
            (df['next_planned_start'] - df['anchor_ts']).dt.total_seconds().fillna(0.0)
        ).astype(np.float32)
        
        # 1) Minute-level keys for joins
        df['anchor_min'] = df['anchor_ts'].dt.floor('min')
        df['time_min']   = df['time:timestamp'].dt.floor('min')
        df['plan_min']   = df['planned:timestamp'].dt.floor('min')
        
        # --- Determine if we have reliable finish timestamps for true arrivals at B ---
        finish_col_present = 'subsystem_finished_ts' in df.columns
        if finish_col_present:
            finish_ts = pd.to_datetime(df['subsystem_finished_ts'], utc=True, errors='coerce')
            has_any_finish = finish_ts.notna().any()
        else:
            finish_ts = None
            has_any_finish = False
        
        _HAS_FINISH_TS = has_any_finish  # only true if real, usable finish timestamps exist
        
        # 2) Build per-minute ACTUAL arrivals (to B) and ACTUAL services (from B)
        if _HAS_FINISH_TS:
            # Arrivals at B from finish timestamps
            arr_min = (
                df.assign(finish_min=finish_ts.dt.floor('min'))[['event_end','finish_min']]
                  .rename(columns={'event_end':'station','finish_min':'ts_min'})
                  .dropna(subset=['ts_min'])
                  .groupby(['station','ts_min'], observed=True).size()
                  .rename('arrivals_per_min').reset_index()
                  .sort_values(['station','ts_min'])
            )
        else:
            # No reliable finishes → empty frame (downstream gives zeros)
            arr_min = pd.DataFrame({
                'station': pd.Series(dtype='object'),
                'ts_min': pd.Series(dtype='datetime64[ns, UTC]'),
                'arrivals_per_min': pd.Series(dtype='int64')
            })
        
        # Services from B = starts at B (use actual start timestamps) — always safe
        srv_min = (
            df[['event:start','time:timestamp']]
              .assign(ts_min=lambda x: x['time:timestamp'].dt.floor('min'))
              .rename(columns={'event:start':'station'})
              .groupby(['station','ts_min'], observed=True).size()
              .rename('services_per_min').reset_index()
              .sort_values(['station','ts_min'])
        )

        
        # 3) Build per-minute PLANNED arrivals to B and PLANNED services from B (leakage-safe)
        parr_min = (
            df.dropna(subset=['event_end'])
              [['event_end','plan_min']]
              .rename(columns={'event_end':'station','plan_min':'ts_min'})
              .groupby(['station','ts_min'], observed=True).size()
              .rename('planned_arrivals_per_min').reset_index()
              .sort_values(['station','ts_min'])
        )
        
        psrv_min = (
            df.dropna(subset=['event:start'])
              [['event:start','plan_min']]
              .rename(columns={'event:start':'station','plan_min':'ts_min'})
              .groupby(['station','ts_min'], observed=True).size()
              .rename('planned_services_per_min').reset_index()
              .sort_values(['station','ts_min'])
        )
    
    
    
        # --- integer-epoch-minute joins for next-station congestion ---
        def _to_epoch_min_int64_from_any(s: pd.Series) -> np.ndarray:
            return (pd.to_datetime(s, utc=True).values.view('i8') // np.int64(60_000_000_000)).astype(np.int64)

        # Prepare unique (station, anchor_min) queries and integer keys
        df['anchor_min']   = pd.to_datetime(df['anchor_ts'].dt.floor('min'), utc=True)
        df['anchor_min_i'] = (df['anchor_min'].values.view('i8') // np.int64(60_000_000_000)).astype(np.int64)

        query_keys = (
            df.dropna(subset=['event_end'])
              [['event_end','anchor_min']]
              .rename(columns={'event_end':'station','anchor_min':'ts_min'})
              .drop_duplicates(['station','ts_min'])
              .sort_values(['station','ts_min'])
        )
        query_keys['ts_min']   = pd.to_datetime(query_keys['ts_min'], utc=True)
        query_keys['ts_min_i'] = _to_epoch_min_int64_from_any(query_keys['ts_min'])

        stations = query_keys['station'].unique()

        maps_now  = []                # backlog at now (cumulative arrivals − services)
        maps_wins = {5: [], 15: [], 60: []}  # arrivals/services/pressure over windows
        WINDOWS   = (5, 15, 60)
        MAXW      = max(WINDOWS)

        for st in stations:
            need   = query_keys.loc[query_keys['station'].eq(st), 'ts_min']
            need_i = _to_epoch_min_int64_from_any(need)
            if need_i.size == 0:
                continue

            # Bound the history we pull for this station to the smallest necessary range:
            lo_need = need_i.min() - (MAXW - 1)   # earliest minute that can still contribute
            hi_need = need_i.max()                # latest minute we need

            # Pull actual series for this station
            a = arr_min[arr_min['station'].eq(st)][['ts_min','arrivals_per_min']]
            s = srv_min[srv_min['station'].eq(st)][['ts_min','services_per_min']]

            # Convert to int-minute and keep only the bounded window
            if not a.empty:
                ai  = _to_epoch_min_int64_from_any(a['ts_min'])
                msk = (ai >= lo_need) & (ai <= hi_need)
                a   = a.loc[msk]
                ai  = ai[msk]
            else:
                ai = np.array([], dtype=np.int64)

            if not s.empty:
                si  = _to_epoch_min_int64_from_any(s['ts_min'])
                msk = (si >= lo_need) & (si <= hi_need)
                s   = s.loc[msk]
                si  = si[msk]
            else:
                si = np.array([], dtype=np.int64)

            # Dense minute grid covering bounded actual & needed minutes
            mins_all = np.unique(np.concatenate([ai, si, need_i]))
            if mins_all.size == 0:
                # Append zeros for this station across all needed times
                maps_now.append(pd.DataFrame({
                    'station': st,
                    'ts_min_i': need_i,
                    'backlog_next_now': np.zeros_like(need_i, dtype=np.int32)
                }))
                for W in WINDOWS:
                    maps_wins[W].append(pd.DataFrame({
                        'station': st,
                        'ts_min_i': need_i,
                        f'arrivals_next_{W}m': np.zeros_like(need_i, dtype=np.int32),
                        f'services_next_{W}m': np.zeros_like(need_i, dtype=np.int32),
                        f'pressure_next_{W}m': np.zeros_like(need_i, dtype=np.int32),
                    }))
                continue

            M = mins_all.size
            arrivals = np.zeros(M, dtype=np.int32)
            services = np.zeros(M, dtype=np.int32)

            if ai.size:
                ia = np.searchsorted(mins_all, ai).astype(np.intp)
                arrivals[ia] = a['arrivals_per_min'].to_numpy(np.int32)
            if si.size:
                is_ = np.searchsorted(mins_all, si).astype(np.intp)
                services[is_] = s['services_per_min'].to_numpy(np.int32)

            # Cumulatives (up to minute m)
            c_arr = np.cumsum(arrivals, dtype=np.int64)
            c_srv = np.cumsum(services, dtype=np.int64)

            # Map "now" (anchor minute) to cumulative backlog
            pos = np.clip(np.searchsorted(mins_all, need_i).astype(np.intp), 0, M-1)
            backlog_now = np.maximum(c_arr[pos] - c_srv[pos], 0).astype(np.int32)

            maps_now.append(pd.DataFrame({
                'station': st,
                'ts_min_i': need_i,
                'backlog_next_now': backlog_now
            }))

            # Rolling windows via cumulative differences
            for W in WINDOWS:
                prev_pos = np.maximum(pos - W, -1)
                arr_W = (c_arr[pos] - np.where(prev_pos < 0, 0, c_arr[prev_pos])).astype(np.int32)
                srv_W = (c_srv[pos] - np.where(prev_pos < 0, 0, c_srv[prev_pos])).astype(np.int32)
                presW = (arr_W - srv_W).astype(np.int32)

                maps_wins[W].append(pd.DataFrame({
                    'station': st,
                    'ts_min_i': need_i,
                    f'arrivals_next_{W}m': arr_W,
                    f'services_next_{W}m': srv_W,
                    f'pressure_next_{W}m': presW
                }))

        # Stitch and merge back to df using integer keys
        m_now = (pd.concat(maps_now, ignore_index=True)
                 if maps_now else query_keys[['station','ts_min_i']].assign(backlog_next_now=0))

        df = df.merge(
            m_now,
            left_on=['event_end','anchor_min_i'],
            right_on=['station','ts_min_i'],
            how='left'
        ).drop(columns=['station','ts_min_i'])

        for W in (5, 15, 60):
            mW = (pd.concat(maps_wins[W], ignore_index=True)
                  if maps_wins[W] else query_keys[['station','ts_min_i']].assign(**{
                      f'arrivals_next_{W}m': 0,
                      f'services_next_{W}m': 0,
                      f'pressure_next_{W}m': 0
                  }))
            df = df.merge(
                mW,
                left_on=['event_end','anchor_min_i'],
                right_on=['station','ts_min_i'],
                how='left'
            ).drop(columns=['station','ts_min_i'])

        # 6) Delta pressure (short vs long window) — captures trend/momentum
        df['delta_pressure_next_15m_60m'] = (
            df['pressure_next_15m'] - df['pressure_next_60m']
        ).astype(np.int32)

        # 7) Planned arrivals/services UNTIL the next planned start (leakage-safe)
        q2 = df[['event_end','anchor_ts','next_planned_start']].copy()
        q2.rename(columns={'event_end':'station'}, inplace=True)

        def _prep_planned_cums(frame: pd.DataFrame, value_col: str):
            if frame.empty:
                return {}
            out = {}
            for st, g in frame.groupby('station', observed=True):
                mins = np.sort(_to_epoch_min_int64_from_any(g['ts_min']))
                vals = g[value_col].to_numpy(np.int32)[np.argsort(_to_epoch_min_int64_from_any(g['ts_min']))]
                if mins.size > 1:
                    uniq, idx = np.unique(mins, return_index=True)
                    if uniq.size != mins.size:
                        acc = {}
                        for m, v in zip(mins, vals):
                            acc[m] = acc.get(m, 0) + v
                        uniq = np.fromiter(acc.keys(), dtype=np.int64)
                        vals = np.fromiter(acc.values(), dtype=np.int32)
                        order = np.argsort(uniq)
                        uniq, vals = uniq[order], vals[order]
                    mins = uniq
                csum = np.cumsum(vals, dtype=np.int64)
                out[st] = (mins, csum)
            return out

        parr_cums = _prep_planned_cums(parr_min, 'planned_arrivals_per_min')
        psrv_cums = _prep_planned_cums(psrv_min, 'planned_services_per_min')

        def _range_count(cums, station, lo_min_i, hi_min_i):
            # returns count in (lo, hi] on minute grid (open on left, closed on right)
            if station not in cums:
                return 0
            mins, csum = cums[station]
            r = np.searchsorted(mins, hi_min_i, side='right') - 1
            if r < 0:
                return 0
            l = np.searchsorted(mins, lo_min_i, side='right') - 1
            left_val = 0 if l < 0 else csum[l]
            return int(csum[r] - left_val) if r >= l else 0

        
        anchor_i  = (df['anchor_ts'].values.view('i8') // np.int64(60_000_000_000)).astype(np.int64)
        nps = pd.to_datetime(df['next_planned_start'], utc=True, errors='coerce')
        nps_i = np.where(nps.isna(), anchor_i, (nps.values.view('i8') // np.int64(60_000_000_000)).astype(np.int64))
        st_vec = df['event_end'].values
        
        planned_arr_to_start = np.empty(len(df), dtype=np.int32)
        planned_srv_to_start = np.empty(len(df), dtype=np.int32)
        
        for i in range(len(df)):
            lo = anchor_i[i]
            hi = nps_i[i]
            if hi <= lo:
                planned_arr_to_start[i] = 0
                planned_srv_to_start[i] = 0
                continue
            st = st_vec[i]
            planned_arr_to_start[i] = _range_count(parr_cums, st, lo, hi)
            planned_srv_to_start[i] = _range_count(psrv_cums, st, lo, hi)
        
        df['planned_arrivals_to_start_next'] = planned_arr_to_start
        df['planned_services_to_start_next'] = planned_srv_to_start
        
        df['planned_pressure_to_start_next'] = (
            df['backlog_next_now'].astype(np.int64)
            + df['planned_arrivals_to_start_next'].astype(np.int64)
            - df['planned_services_to_start_next'].astype(np.int64)
        ).astype(np.int32)
        
        # Gap awareness & normalized congestion
        df['planned_gap_sec'] = (
            (pd.to_datetime(df['next_planned_start'], utc=True, errors='coerce') - df['anchor_ts'])
            .dt.total_seconds()
        ).fillna(np.inf).astype(np.float32)
        
        df['gap_le_5m']  = (df['planned_gap_sec'] <=  5*60).astype(np.int8)
        df['gap_le_15m'] = (df['planned_gap_sec'] <= 15*60).astype(np.int8)
        df['gap_le_60m'] = (df['planned_gap_sec'] <= 60*60).astype(np.int8)
        
        for W, flag in [(5, 'gap_le_5m'), (15, 'gap_le_15m'), (60, 'gap_le_60m')]:
            df[f'arrivals_next_{W}m_gc']  = (df[f'arrivals_next_{W}m']  * df[flag]).astype(np.int32)
            df[f'services_next_{W}m_gc']  = (df[f'services_next_{W}m']  * df[flag]).astype(np.int32)
            df[f'pressure_next_{W}m_gc']  = (df[f'pressure_next_{W}m'] * df[flag]).astype(np.int32)
        
        eps = 1e-6
        df['planned_horizon_until_start'] = (
            df['planned_arrivals_to_start_next'].astype(np.float32) +
            df['planned_services_to_start_next'].astype(np.float32)
        ).clip(lower=0.0)
        
        df['backlog_rel_to_planned'] = (
            df['backlog_next_now'].astype(np.float32) / (df['planned_horizon_until_start'] + eps)
        ).astype(np.float32)
        
        for W in (5, 15, 60):
            df[f'pressure_next_{W}m_rel_planned'] = (
                df[f'pressure_next_{W}m'].astype(np.float32) / (df['planned_horizon_until_start'] + eps)
            ).astype(np.float32)
        
        gap_min = (df['planned_gap_sec'] / 60.0).clip(lower=1.0)
        for W in (5, 15, 60):
            df[f'pressure_next_{W}m_per_gapmin'] = (
                df[f'pressure_next_{W}m'].astype(np.float32) / gap_min
            ).astype(np.float32)
        
        df['route_x_gap15_pressure'] = (
            df['event_route_code'].astype('int32')
            * df['gap_le_15m'].astype('int32')
            * (df['pressure_next_15m'] > 0).astype('int32')
        ).astype('int32')
        
        # Clean up helper columns
        df.drop(columns=['anchor_min','time_min','plan_min'], inplace=True, errors='ignore')
        # ===============================================================================
        log("new congestion variables formed")
   
        # ─── FLEET / SUBSYSTEM CONGESTION (vehicle-agnostic, leakage-safe) ───
        subsys_col = 'subsystem' if 'subsystem' in df.columns else 'concept:name'
        
        # Anchor minute (UTC) + integer minute key
        df['anchor_min']   = pd.to_datetime(df['anchor_ts'].dt.floor('min'), utc=True)
        NS_PER_MIN         = np.int64(60_000_000_000)
        df['anchor_min_i'] = (df['anchor_min'].values.view('i8') // NS_PER_MIN).astype(np.int64)
        
        # Per-subsystem per-minute starts & delayed counts (at anchor times)
        subsys_min = (
            df[[subsys_col, 'anchor_min', 'is_delayed']]
              .groupby([subsys_col, 'anchor_min'], observed=True)
              .agg(starts=('is_delayed', 'count'),
                   delayed=('is_delayed', 'sum'))
              .reset_index()
              .sort_values([subsys_col, 'anchor_min'])
        )
        
        def _to_min_i(s: pd.Series) -> np.ndarray:
            return (pd.to_datetime(s, utc=True).values.view('i8') // NS_PER_MIN).astype(np.int64)
        
        # Query keys: one row per (subsystem, anchor_min_i)
        q = df[[subsys_col, 'anchor_min_i']].drop_duplicates([subsys_col, 'anchor_min_i'])
        
        outs_15, outs_60, outs_d60 = [], [], []
        for subsys, gq in q.groupby(subsys_col, sort=False, observed=True):
            need_i = gq['anchor_min_i'].to_numpy()
        
            g = subsys_min[subsys_min[subsys_col].eq(subsys)][['anchor_min', 'starts', 'delayed']]
            if g.empty:
                outs_15.append(gq.assign(subsys_starts_15m=0))
                outs_60.append(gq.assign(subsys_starts_60m=0))
                outs_d60.append(gq.assign(subsys_delayed_60m=0))
                continue
        
            mins    = _to_min_i(g['anchor_min'])
            order   = np.argsort(mins)
            mins    = mins[order]
            starts  = g['starts'].to_numpy(np.int32)[order]
            delayed = g['delayed'].to_numpy(np.int32)[order]
        
            c_st = np.cumsum(starts,  dtype=np.int64)
            c_de = np.cumsum(delayed, dtype=np.int64)
        
            pos = np.clip(np.searchsorted(mins, need_i), 0, len(mins) - 1)
        
            def _win(cum, W):
                prev = np.maximum(pos - W, -1)
                return (cum[pos] - np.where(prev < 0, 0, cum[prev])).astype(np.int32)
        
            outs_15.append(gq.assign(subsys_starts_15m=_win(c_st, 15)))
            outs_60.append(gq.assign(subsys_starts_60m=_win(c_st, 60)))
            outs_d60.append(gq.assign(subsys_delayed_60m=_win(c_de, 60)))
        
        m15  = pd.concat(outs_15,  ignore_index=True) if outs_15  else q.assign(subsys_starts_15m=0)
        m60  = pd.concat(outs_60,  ignore_index=True) if outs_60  else q.assign(subsys_starts_60m=0)
        md60 = pd.concat(outs_d60, ignore_index=True) if outs_d60 else q.assign(subsys_delayed_60m=0)
        
        # build a single keyed frame
        sub_map = (
            m15.merge(m60,  on=[subsys_col,'anchor_min_i'], how='outer')
               .merge(md60, on=[subsys_col,'anchor_min_i'], how='outer')
               .set_index([subsys_col,'anchor_min_i'])
               .astype({'subsys_starts_15m':'int32','subsys_starts_60m':'int32','subsys_delayed_60m':'int32'})
        )
        
        mi_sub = pd.MultiIndex.from_arrays([df[subsys_col].values, df['anchor_min_i'].values],
                                           names=[subsys_col,'anchor_min_i'])
        aligned = sub_map.reindex(mi_sub)
        
        df['subsys_starts_15m'] = aligned['subsys_starts_15m'].fillna(0).to_numpy(dtype='int32')
        df['subsys_starts_60m'] = aligned['subsys_starts_60m'].fillna(0).to_numpy(dtype='int32')
        df['subsys_delayed_60m'] = aligned['subsys_delayed_60m'].fillna(0).to_numpy(dtype='int32')

        
        df['subsys_delay_rate_60m'] = (
            df['subsys_delayed_60m'] / np.clip(df['subsys_starts_60m'], 1, None)
        ).astype(np.float32)
        # ─────────────────────────────────────────────────────────────────────────────

        # ─────────────────────────────────────────────────────────────────────────────

        # ─── NEXT-SOURCE (current event_end) STATION CONGESTION ───
        # station where the *next* event will start:
        df = _ensure_event_end(df)
        df['anchor_min'] = pd.to_datetime(df['anchor_ts'].dt.floor('min'), utc=True)
        
        # Per-minute PLANNED arrivals to station = events whose planned destination is this station
        parr = (
            df[['event_end','planned:timestamp']]
              .rename(columns={'event_end':'station','planned:timestamp':'ts'})
              .assign(ts_min=lambda x: pd.to_datetime(x['ts'].dt.floor('min'), utc=True))
              .groupby(['station','ts_min'], observed=True).size()
              .rename('planned_arrivals_per_min').reset_index()
              .sort_values(['station','ts_min'])
        )
        
        # Per-minute SERVICES (departures) from station = events whose actual start is at this station
        psrv = (
            df[['event:start','time:timestamp']]
              .rename(columns={'event:start':'station','time:timestamp':'ts'})
              .assign(ts_min=lambda x: pd.to_datetime(x['ts'].dt.floor('min'), utc=True))
              .groupby(['station','ts_min'], observed=True).size()
              .rename('services_per_min').reset_index()
              .sort_values(['station','ts_min'])
        )
        
        def _mk_window_maps(parr_df, psrv_df, query_df):
            NS_PER_MIN = np.int64(60_000_000_000)
            def _to_min_i(s): return (pd.to_datetime(s, utc=True).values.view('i8') // NS_PER_MIN).astype(np.int64)
        
            q = query_df[['event_end','anchor_min']].rename(columns={'event_end':'station'})
            q = q.drop_duplicates(['station','anchor_min']).copy()
            q['anchor_min'] = pd.to_datetime(q['anchor_min'], utc=True)
            q['min_i'] = _to_min_i(q['anchor_min'])
        
            outs = {5: [], 15: [], 60: []}   # ← add 15 here
            for st, gq in q.groupby('station', sort=False, observed=True):
                need_i = gq['min_i'].to_numpy()
        
                a = parr_df[parr_df['station'].eq(st)][['ts_min','planned_arrivals_per_min']]
                s = psrv_df[psrv_df['station'].eq(st)][['ts_min','services_per_min']]
        
                mins_all = np.unique(np.concatenate([
                    _to_min_i(a['ts_min']) if not a.empty else np.array([], dtype=np.int64),
                    _to_min_i(s['ts_min']) if not s.empty else np.array([], dtype=np.int64),
                    need_i
                ]))
                if mins_all.size == 0:
                    for W in (5, 15, 60):
                        outs[W].append(gq.assign(**{
                            f'srcnext_arrivals_{W}m': 0,
                            f'srcnext_services_{W}m': 0,
                            f'srcnext_pressure_{W}m': 0
                        }))
                    continue
        
                M = mins_all.size
                arr = np.zeros(M, dtype=np.int32)
                srv = np.zeros(M, dtype=np.int32)
        
                if not a.empty:
                    ia = np.searchsorted(mins_all, _to_min_i(a['ts_min'])).astype(np.intp)
                    arr[ia] = a['planned_arrivals_per_min'].to_numpy(np.int32)
                if not s.empty:
                    is_ = np.searchsorted(mins_all, _to_min_i(s['ts_min'])).astype(np.intp)
                    srv[is_] = s['services_per_min'].to_numpy(np.int32)
        
                c_a = np.cumsum(arr, dtype=np.int64)
                c_s = np.cumsum(srv, dtype=np.int64)
                pos = np.clip(np.searchsorted(mins_all, need_i).astype(np.intp), 0, M-1)
        
                for W in (5, 15, 60):                       # ← loop 5/15/60
                    prev = np.maximum(pos - W, -1)
                    aW = (c_a[pos] - np.where(prev < 0, 0, c_a[prev])).astype(np.int32)
                    sW = (c_s[pos] - np.where(prev < 0, 0, c_s[prev])).astype(np.int32)
                    pW = (aW - sW).astype(np.int32)
                    outs[W].append(gq.assign(**{
                        f'srcnext_arrivals_{W}m': aW,
                        f'srcnext_services_{W}m': sW,
                        f'srcnext_pressure_{W}m': pW,
                    }))
        
            m5   = pd.concat(outs[5],  ignore_index=True) if outs[5]  else q.assign(srcnext_arrivals_5m=0,  srcnext_services_5m=0,  srcnext_pressure_5m=0)
            m15  = pd.concat(outs[15], ignore_index=True) if outs[15] else q.assign(srcnext_arrivals_15m=0, srcnext_services_15m=0, srcnext_pressure_15m=0)
            m60  = pd.concat(outs[60], ignore_index=True) if outs[60] else q.assign(srcnext_arrivals_60m=0, srcnext_services_60m=0, srcnext_pressure_60m=0)
            return m5, m15, m60

        
        m5, m15, m60 = _mk_window_maps(parr, psrv, df)
        
        m5i = (m5.rename(columns={'station':'event_end'})
                 .set_index(['event_end','anchor_min'])
                 [['srcnext_arrivals_5m','srcnext_services_5m','srcnext_pressure_5m']])
        m15i = (m15.rename(columns={'station':'event_end'})
                  .set_index(['event_end','anchor_min'])
                  [['srcnext_arrivals_15m','srcnext_services_15m','srcnext_pressure_15m']])
        m60i = (m60.rename(columns={'station':'event_end'})
                  .set_index(['event_end','anchor_min'])
                  [['srcnext_arrivals_60m','srcnext_services_60m','srcnext_pressure_60m']])
        
        mi = pd.MultiIndex.from_arrays([df['event_end'].values, df['anchor_min'].values],
                                       names=['event_end','anchor_min'])
        
        a5   = m5i.reindex(mi)
        a15  = m15i.reindex(mi)
        a60  = m60i.reindex(mi)
        
        df['srcnext_arrivals_5m']   = a5['srcnext_arrivals_5m'  ].fillna(0).to_numpy(dtype='int32')
        df['srcnext_services_5m']   = a5['srcnext_services_5m'  ].fillna(0).to_numpy(dtype='int32')
        df['srcnext_pressure_5m']   = a5['srcnext_pressure_5m'  ].fillna(0).to_numpy(dtype='int32')
        df['srcnext_arrivals_15m']  = a15['srcnext_arrivals_15m' ].fillna(0).to_numpy(dtype='int32')
        df['srcnext_services_15m']  = a15['srcnext_services_15m' ].fillna(0).to_numpy(dtype='int32')
        df['srcnext_pressure_15m']  = a15['srcnext_pressure_15m' ].fillna(0).to_numpy(dtype='int32')
        df['srcnext_arrivals_60m']  = a60['srcnext_arrivals_60m' ].fillna(0).to_numpy(dtype='int32')
        df['srcnext_services_60m']  = a60['srcnext_services_60m' ].fillna(0).to_numpy(dtype='int32')
        df['srcnext_pressure_60m']  = a60['srcnext_pressure_60m' ].fillna(0).to_numpy(dtype='int32')
        
        for c in [
            'srcnext_arrivals_5m','srcnext_services_5m','srcnext_pressure_5m',
            'srcnext_arrivals_15m','srcnext_services_15m','srcnext_pressure_15m',
            'srcnext_arrivals_60m','srcnext_services_60m','srcnext_pressure_60m'
        ]:
            df[c] = df[c].fillna(0).astype(np.int32)

        # ─────────────────────────────────────────────────────────────────



        # ---- Planned counts at next-source (event_end) from anchor -> next_planned_start
        NS_PER_MIN = np.int64(60_000_000_000)
        to_min_i = lambda s: (pd.to_datetime(s, utc=True).values.view('i8') // NS_PER_MIN).astype(np.int64)
        
        def _prep_planned_cums_simple(frame: pd.DataFrame, value_col: str):
            if frame.empty: return {}
            out = {}
            for st, g in frame.groupby('station', observed=True):
                mins = to_min_i(g['ts_min'])
                vals = g[value_col].to_numpy(np.int32)
                if mins.size > 1:
                    uniq, idx = np.unique(mins, return_index=True)
                    if uniq.size != mins.size:
                        acc = {}
                        for m, v in zip(mins, vals):
                            acc[m] = acc.get(m, 0) + v
                        mins = np.fromiter(acc.keys(), dtype=np.int64)
                        vals = np.fromiter(acc.values(), dtype=np.int32)
                        order = np.argsort(mins)
                        mins, vals = mins[order], vals[order]
                csum = np.cumsum(vals, dtype=np.int64)
                out[st] = (mins, csum)
            return out
        
        # Build per-minute planned series for srcnext (you already created these as `parr` and `psrv`)
        parr_src = parr.rename(columns={'planned_arrivals_per_min':'val'})[['station','ts_min','val']]
        psrv_src = psrv.rename(columns={'services_per_min':'val'})[['station','ts_min','val']]
        
        parr_cums_src = _prep_planned_cums_simple(parr_src.rename(columns={'val':'planned_arrivals_per_min'}),
                                                  'planned_arrivals_per_min')
        psrv_cums_src = _prep_planned_cums_simple(psrv_src.rename(columns={'val':'planned_services_per_min'}),
                                                  'planned_services_per_min')
        
        anchor_i = to_min_i(df['anchor_ts'])
        nps = pd.to_datetime(df['next_planned_start'], utc=True, errors='coerce')
        nps_i = np.where(nps.isna(), anchor_i, (nps.values.view('i8') // NS_PER_MIN).astype(np.int64))
        st_vec = df['event_end'].values
        
        def _range_count_cums(cums, station, lo_i, hi_i):
            if station not in cums: return 0
            mins, csum = cums[station]
            r = np.searchsorted(mins, hi_i, side='right') - 1
            if r < 0: return 0
            l = np.searchsorted(mins, lo_i, side='right') - 1
            left = 0 if l < 0 else csum[l]
            return int(csum[r] - left) if r >= l else 0
        
        src_planned_arr = np.empty(len(df), dtype=np.int32)
        src_planned_srv = np.empty(len(df), dtype=np.int32)
        for i in range(len(df)):
            lo = anchor_i[i]; hi = nps_i[i]
            if hi <= lo:
                src_planned_arr[i] = 0; src_planned_srv[i] = 0
                continue
            st = st_vec[i]
            src_planned_arr[i] = _range_count_cums(parr_cums_src, st, lo, hi)
            src_planned_srv[i] = _range_count_cums(psrv_cums_src, st, lo, hi)
        
        df['srcnext_planned_arrivals_to_start_next'] = src_planned_arr
        df['srcnext_planned_services_to_start_next'] = src_planned_srv
        df['srcnext_planned_pressure_to_start_next'] = (
            df['backlog_next_now'].astype(np.int64)  # using the actual backlog you already computed
            + df['srcnext_planned_arrivals_to_start_next'].astype(np.int64)
            - df['srcnext_planned_services_to_start_next'].astype(np.int64)
        ).astype(np.int32)


        # --- Phase-aware operative congestion features ----------------------------
        # Identify phase
        is_start  = (df['eventkey'] == 'subsystem_started').astype(np.int8)
        is_finish = (df['eventkey'] == 'subsystem_finished').astype(np.int8)
        
        # Helper selector
        def pick(start_val, finish_val):
            # choose start_val if subsystem_started, else finish_val if subsystem_finished, else default to start_val
            return np.where(is_start == 1, start_val,
                   np.where(is_finish == 1, finish_val, start_val))
        
        # 0) backlog now (dest vs next-source)
        # If you want a true "srcnext_backlog_now", build it exactly like backlog_next_now but on the same station/time
        # using the srcnext minute series (actual arrivals vs services). If not available, you can proxy:
        # Use the same leakage-safe backlog at the next station for both phases
        df['op_backlog_now'] = df['backlog_next_now'].astype(np.int32)

        
        # 1) 5/15/60-minute arrivals/services/pressure
        for W in (5, 15, 60):
            dest_a = df.get(f'arrivals_next_{W}m', 0)
            dest_s = df.get(f'services_next_{W}m', 0)
            dest_p = df.get(f'pressure_next_{W}m', 0)
        
            src_a  = df.get(f'srcnext_arrivals_{W}m', 0)   # ensure you compute 15m, see note above
            src_s  = df.get(f'srcnext_services_{W}m', 0)
            src_p  = df.get(f'srcnext_pressure_{W}m', 0)
        
            df[f'op_arrivals_{W}m'] = pick(dest_a, src_a).astype(np.int32)
            df[f'op_services_{W}m'] = pick(dest_s, src_s).astype(np.int32)
            df[f'op_pressure_{W}m'] = pick(dest_p, src_p).astype(np.int32)
        
        # 2) Short–long pressure trend
        dest_trend = df.get('delta_pressure_next_15m_60m', 0)
        # For 'srcnext', if you add 15m you can mirror it; else compute on the fly:
        src_trend = df.get('srcnext_pressure_15m', 0) - df.get('srcnext_pressure_60m', 0)
        df['op_pressure_trend_15m_60m'] = pick(dest_trend, src_trend).astype(np.int32)
        
        # 3) Planned until next start (leak-safe; you already computed destination version)
        df['op_planned_arrivals_to_start_next'] = pick(
            df['planned_arrivals_to_start_next'],
            df.get('srcnext_planned_arrivals_to_start_next', df['planned_arrivals_to_start_next'])
        ).astype(np.int32)
        
        df['op_planned_services_to_start_next'] = pick(
            df['planned_services_to_start_next'],
            df.get('srcnext_planned_services_to_start_next', df['planned_services_to_start_next'])
        ).astype(np.int32)
        
        df['op_planned_pressure_to_start_next'] = pick(
            df['planned_pressure_to_start_next'],
            df.get('srcnext_planned_pressure_to_start_next', df['planned_pressure_to_start_next'])
        ).astype(np.int32)
        
        # 4) Gap to next planned start (seconds)
        df['op_gap_sec_to_next_planned_start'] = df['planned_gap_sec'].astype(np.float32)
        
        # 5) Normalized variants
        eps = 1e-6
        planned_h = (
            df['op_planned_arrivals_to_start_next'].astype(np.float32) +
            df['op_planned_services_to_start_next'].astype(np.float32)
        ).clip(lower=0.0)
        
        for W in (5, 15, 60):
            df[f'op_pressure_{W}m_rel_planned'] = (
                df[f'op_pressure_{W}m'].astype(np.float32) / (planned_h + eps)
            ).astype(np.float32)
        
        gap_min = (df['op_gap_sec_to_next_planned_start'] / 60.0).clip(lower=1.0)
        for W in (5, 15, 60):
            df[f'op_pressure_{W}m_per_gapmin'] = (
                df[f'op_pressure_{W}m'].astype(np.float32) / gap_min
            ).astype(np.float32)
        
        # Optional: keep "phase indicators" so the model can also use them
        df['phase_is_start']  = is_start
        df['phase_is_finish'] = is_finish
        # --------------------------------------------------------------------------



        # 4) Transition (edge) congestion at arrival: source → destination
        #    Build per-minute edge history and compute 1h window via cumulative diff.
        edge_hist = df[['event:start','event_end','time:timestamp','is_delayed']].copy()
        edge_hist = edge_hist.rename(columns={
            'event:start':'src',
            'event_end':'dst',
            'time:timestamp':'event_ts'
        })
        edge_hist['key_min'] = edge_hist['event_ts'].dt.floor('min')
        edge_hist = edge_hist.dropna(subset=['src','dst'])
        
        edge_agg = (edge_hist.groupby(['src','dst','key_min'], observed=True)
                            .agg(starts=('event_ts','count'),
                                 delayed=('is_delayed','sum'))
                            .reset_index()
                            .sort_values(['src','dst','key_min']))
        edge_agg['c_starts']  = edge_agg.groupby(['src','dst'], observed=True)['starts'].cumsum()
        edge_agg['c_delayed'] = edge_agg.groupby(['src','dst'], observed=True)['delayed'].cumsum()
        

        # Map cumulative edge starts at the row’s (src,dst,anchor_min)
        df['edge_src']   = df['event:start']
        df['edge_dst']   = df['event_end']
        df['anchor_min'] = df['anchor_ts'].dt.floor('min')
        
        edge_map = edge_agg.rename(columns={'key_min':'ts_min'}).set_index(['src','dst','ts_min'])['c_starts']
        mi_edge = pd.MultiIndex.from_arrays(
            [df['edge_src'].values, df['edge_dst'].values, df['anchor_min'].values],
            names=['src','dst','ts_min']
        )
        df['edge_c_starts_now'] = edge_map.reindex(mi_edge).fillna(0).to_numpy(dtype='int32')


        
        
        log("edge congestion done")
        
        # 7) Minimal sequence/progress context (no memory blow-up)
        df['prefix_length'] = df.groupby(self.cid).cumcount() + 1
                
        
        first_seen_flag = (
            df.groupby([self.cid, 'event_end']).cumcount() == 0
        ).astype(np.int32)
        df['unique_stations_so_far'] = (
            first_seen_flag.groupby(df[self.cid]).cumsum()
        ).astype(np.int32)

        df['concept:name'] = df['concept:name'].astype('category')

        
        # concept:name repeat flag within case  (vectorized; memory safe)
        # True (1) if this activity has appeared before in the same case, else 0
        df['concept:name_repeat_flag'] = (
            df.groupby([self.cid, 'concept:name'], observed=True).cumcount() > 0
        ).astype(np.int8)

        

        df = self.add_case_arrival_and_active_cases(df, anchor_col="anchor_ts")

        

        # ─── RESOURCE-LEVEL CONGESTION & DELAY FEATURES (memory-lean) ───
        df = _ensure_event_end(df)
        
        # work on a slim temp to avoid consolidating the whole df
        tmp = df[['event_end','anchor_ts','is_delayed','delay']].copy()
        tmp['is_delayed'] = tmp['is_delayed'].fillna(0).astype('int8', copy=False)
        tmp['delay_pos']  = tmp['delay'].where(tmp['is_delayed'].eq(1))
        
        N = len(tmp)
        sr = np.zeros(N, dtype='int32')
        dc = np.zeros(N, dtype='int32')
        ad = np.zeros(N, dtype='float32')
        sd = np.zeros(N, dtype='float32')
        
        # per-station rolling (sort tiny chunks only)
        for st, sub in tmp.groupby('event_end', sort=False, observed=True):
            if sub.empty: 
                continue
            sub = sub.sort_values('anchor_ts', kind='mergesort')
            r = sub.set_index('anchor_ts').rolling('60min', closed='both')
        
            sr_vals = r['is_delayed'].count().astype('int32')
            dc_vals = r['is_delayed'].sum().astype('int32')
            ad_vals = r['delay_pos'].mean().fillna(0.0).astype('float32')
            sd_vals = r['delay_pos'].std().fillna(0.0).astype('float32')
        
            idx = sub.index.to_numpy()
            sr[idx] = sr_vals.to_numpy()
            dc[idx] = dc_vals.to_numpy()
            ad[idx] = ad_vals.to_numpy()
            sd[idx] = sd_vals.to_numpy()
        
        # assign back (index-aligned)
        df['service_rate_event_end_1h'] = sr
        df['delays_count_event_end_1h'] = dc
        df['avg_delay_event_end_1h']    = ad
        df['std_delay_event_end_1h']    = sd
        
        # ── planned_60 (sparse, cumulative + searchsorted; no dense asfreq) ──
        pm = (
            df.dropna(subset=['event_end'])
              [['event_end','planned:timestamp']]
              .assign(ts_min=lambda x: pd.to_datetime(x['planned:timestamp'].dt.floor('min'), utc=True))
              .groupby(['event_end','ts_min'], observed=True).size()
              .rename('planned_per_min')
        )
        
        NS_PER_MIN = np.int64(60_000_000_000)
        
        def _to_min_i(s: pd.Index | pd.Series) -> np.ndarray:
            # works for tz-aware DatetimeIndex/Series to epoch minutes (int64)
            return (pd.to_datetime(s, utc=True).values.view('i8') // NS_PER_MIN).astype(np.int64)
        
        if len(pm) == 0:
            df['planned_60'] = np.int32(0)
        else:
            # Ensure station/minute ascending
            pm = pm.sort_index()
        
            # Build sparse cumulative sums per station
            per_station = {}  # st -> (mins_i, csum)
            for st, s in pm.groupby(level=0, observed=True):
                ts = s.droplevel(0)  # index: ts_min; values: planned_per_min
                mins_i = _to_min_i(ts.index)
                vals   = s.values.astype(np.int32)
                # (index is already sorted because pm.sort_index(); keep for safety)
                order  = np.argsort(mins_i)
                mins_i = mins_i[order]
                vals   = vals[order]
                csum   = np.cumsum(vals, dtype=np.int64)
                per_station[st] = (mins_i, csum)
        
            # Prepare query minutes per row (anchor minute) and output buffer
            anchor_min = pd.to_datetime(df['anchor_ts'].dt.floor('min'), utc=True)
            m_i        = (anchor_min.values.view('i8') // NS_PER_MIN).astype(np.int64)
            stations   = df['event_end'].values
        
            out = np.zeros(len(df), dtype=np.int32)
        
            # Vectorized per station to avoid Python loops per row
            # Window is the last 60 minutes inclusive: [m-59, m]
            for st in pd.unique(stations):
                if pd.isna(st) or st not in per_station:
                    continue
                mins_i, csum = per_station[st]
                if mins_i.size == 0:
                    continue
        
                idx = np.where(stations == st)[0]
                mi  = m_i[idx]
        
                # r = last index with minute <= m
                r = np.searchsorted(mins_i, mi, side='right') - 1
                # l = last index with minute < (m - 59)
                left_bound = mi - 59
                l = np.searchsorted(mins_i, left_bound, side='left') - 1
        
                # sum in [m-59, m] = csum[r] - csum[l]
                left_val = np.where(l < 0, 0, csum[np.clip(l, 0, csum.size-1)])
                val      = np.where(r < 0, 0, csum[np.clip(r, 0, csum.size-1)] - left_val)
        
                out[idx] = val.astype(np.int32, copy=False)
        
            df['planned_60'] = out


        
        # derived
        df['in_flight_event_end_1h']      = df['service_rate_event_end_1h'].astype('int32')
        df['arrival_rate_event_end_1h']   = df['planned_60'].astype('int32')
        df['delay_rate_event_end_1h']     = (df['delays_count_event_end_1h'] / np.clip(df['service_rate_event_end_1h'], 1, None)).astype('float32')
        df['log_delay_rate_event_end_1h'] = np.log1p(df['delay_rate_event_end_1h']).astype('float32')
        df['backlog_event_end_1h']        = np.maximum(df['arrival_rate_event_end_1h'] - df['service_rate_event_end_1h'], 0).astype('int32')
        
        df['delay_rate_event_end_1h'] = (
            df['delays_count_event_end_1h'] / np.clip(df['service_rate_event_end_1h'], 1, None)
        ).astype('float32')
        df['log_delay_rate_event_end_1h'] = np.log1p(df['delay_rate_event_end_1h']).astype('float32')
        
        # backlog proxy the same way you computed it later
        df['backlog_event_end_1h'] = (
            np.maximum(df['arrival_rate_event_end_1h'] - df['service_rate_event_end_1h'], 0)
        ).astype('int32')
        
        # (Optional) If you still want queue_len_event_end / peak_queue_event_end_1h / queue_growth_rate_event_end / time_since_queue_empty_event_end,
        # we can re-derive them later with a lighter approach; for now remove the old large loop and keep identical fields already computed above.
        
        # cleanup
        df.drop(columns=['delay_pos','anchor_min','ts_min'], errors='ignore', inplace=True)
        gc.collect()


        log("resource-level 1h congestion done")
        
        
        # ─── VEHICLE-LEVEL CONGESTION & UTILIZATION (memory-lean) ───
        if 'vehicle_id' in df.columns:
            # sort without deep copy
            df.sort_values(['vehicle_id','anchor_ts'], kind='mergesort', inplace=True)
        
            df['delay_pos'] = df['delay'].where(df['is_delayed'].eq(1))
        
            gveh  = df.set_index('anchor_ts').groupby('vehicle_id', group_keys=False, observed=True)
            roll5 = gveh.rolling('5min',  closed='both')
            roll1 = gveh.rolling('60min', closed='both')
        
            veh_events_5m     = roll5['is_delayed'].count().astype('int32').rename('veh_events_5m')
            veh_events_60m    = roll1['is_delayed'].count().astype('int32').rename('veh_events_60m')
            veh_delayed_60m   = roll1['is_delayed'].sum().astype('int32').rename('veh_delayed_60m')
            veh_avg_delay_60m = roll1['delay_pos'].mean().fillna(0.0).astype('float32').rename('veh_avg_delay_60m')
            veh_std_delay_60m = roll1['delay_pos'].std().fillna(0.0).astype('float32').rename('veh_std_delay_60m')
        
            fastv = pd.concat(
                [veh_events_5m, veh_events_60m, veh_delayed_60m, veh_avg_delay_60m, veh_std_delay_60m],
                axis=1
            )
            fastv.index.names = ['vehicle_id','anchor_ts']
            
            # ensure a unique (vehicle_id, anchor_ts) index
            # (if multiple rows share the same instant, keep the last — values should be identical)
            fastv = fastv[~fastv.index.duplicated(keep='last')]
            
            mi = pd.MultiIndex.from_arrays(
                [df['vehicle_id'].values, df['anchor_ts'].values],
                names=['vehicle_id','anchor_ts']
            )
            aligned = fastv.reindex(mi)

        
            df['veh_events_5m']     = aligned['veh_events_5m'].to_numpy(dtype='int32')
            df['veh_events_60m']    = aligned['veh_events_60m'].to_numpy(dtype='int32')
            df['veh_delayed_60m']   = aligned['veh_delayed_60m'].to_numpy(dtype='int32')
            df['veh_avg_delay_60m'] = aligned['veh_avg_delay_60m'].to_numpy(dtype='float32')
            df['veh_std_delay_60m'] = aligned['veh_std_delay_60m'].to_numpy(dtype='float32')
        
            df['veh_delay_rate_60m'] = (
                df['veh_delayed_60m'] / np.clip(df['veh_events_60m'], 1, None)
            ).astype('float32')
        
            df['veh_time_since_last_s'] = (
                df.groupby('vehicle_id', observed=True, sort=False)['time:timestamp']
                  .diff().dt.total_seconds().fillna(0.0).clip(lower=0.0).astype('float32')
            )
        
            df['event_route_code'] = df['event_route_code'].astype('int32', copy=False)
            df['veh_route_changed'] = (
                df.groupby('vehicle_id', observed=True, sort=False)['event_route_code']
                  .diff().fillna(0).ne(0).astype('int8')
            )
        
            # EWMA without merge
            # EWMA per vehicle directly on the column — keeps row cardinality 1:1, no MultiIndex, no reindex
            df.sort_values(['vehicle_id', 'anchor_ts'], kind='mergesort', inplace=True)
            df['veh_util_ewma_60m'] = (
                df.groupby('vehicle_id', observed=True)['veh_events_60m']
                  .transform(lambda s: s.ewm(alpha=0.3, adjust=False).mean())
                  .astype('float32')
            )

        
            df.drop(columns=['delay_pos'], inplace=True, errors='ignore')
        else:
            for col in ['veh_events_5m','veh_events_60m','veh_delayed_60m',
                        'veh_avg_delay_60m','veh_std_delay_60m','veh_delay_rate_60m',
                        'veh_time_since_last_s','veh_route_changed','veh_util_ewma_60m']:
                df[col] = 0 if col not in ('veh_avg_delay_60m','veh_std_delay_60m') else 0.0



        
        

        # ─── ROLLING INTER-EVENT ────────────────────────────────────
        grp_ie = df.groupby(self.cid)['inter_event_time']
        df['roll_ie_mean'] = grp_ie.transform(lambda x: x.rolling(3,1).mean()).astype(np.float32)
        df['roll_ie_min']  = grp_ie.transform(lambda x: x.rolling(3,1).min()).astype(np.float32)
        df['roll_ie_max']  = grp_ie.transform(lambda x: x.rolling(3,1).max()).astype(np.float32)

        # ─── PLANNED GAP & RATIO ─────────────────────────────────────
        df['planned_iet'] = (
            df.groupby(self.cid)['planned:timestamp']
              .diff().dt.total_seconds().fillna(0)
              .astype(np.float32)
        )
        df['iet_ratio'] = (
            df['inter_event_time'] / df['planned_iet'].replace(0, np.nan)
        ).fillna(1.0).astype(np.float32)
        
        
        # === On-time "closeness to plan" features (static & dynamic) ===================
        # Place this block AFTER you compute planned_iet / iet_ratio so we can
        # build relative deviations. Uses only info available at the current event.
        
        cid = self.cid
        
        # Signed deviations (seconds): actual - planned (static & dynamic)
        df['dev_from_plan_s'] = (
            df['time:timestamp'] - df['planned:timestamp']
        ).dt.total_seconds().astype(np.float32)
        
        df['dev_from_dynplan_s'] = (
            df['time:timestamp'] - df['dynamic_planned']
        ).dt.total_seconds().astype(np.float32)
        
        # Absolute deviations (seconds): how close, regardless of early/late
        df['abs_dev_from_plan_s']    = df['dev_from_plan_s'].abs().astype(np.float32)
        df['abs_dev_from_dynplan_s'] = df['dev_from_dynplan_s'].abs().astype(np.float32)
        
        # Dynamic planned IET (for a leakage-safe relative deviation)
        df['dynamic_planned_iet'] = (
            df.groupby(self.cid)['dynamic_planned']
              .diff().dt.total_seconds().fillna(0.0)
        ).astype(np.float32)
        
        # Relative deviations (unitless): abs deviation / expected gap
        # Use dynamic planned IET for dynamic version; fall back safely on zeros.
        rel_static = df['abs_dev_from_plan_s'] / df['planned_iet'].replace(0, np.nan)
        rel_dyn    = df['abs_dev_from_dynplan_s'] / df['dynamic_planned_iet'].replace(0, np.nan)
        
        df['rel_dev_from_plan']    = rel_static.fillna(0.0).astype(np.float32)
        df['rel_dev_from_dynplan'] = rel_dyn.fillna(0.0).astype(np.float32)
        
        # Smooth "closeness" scores in [0, 1], higher = closer to plan
        # Decay scale = 60s; adjust if you want tolerance bands (e.g., 120s).
        df['closeness_to_plan']    = np.exp(-df['abs_dev_from_plan_s']    / 60.0).astype(np.float32)
        df['closeness_to_dynplan'] = np.exp(-df['abs_dev_from_dynplan_s'] / 60.0).astype(np.float32)
        
        # (Optional) Clip extreme values to tame outliers
        df['abs_dev_from_plan_s']    = df['abs_dev_from_plan_s'].clip(0,  6*3600)  # up to 6h
        df['abs_dev_from_dynplan_s'] = df['abs_dev_from_dynplan_s'].clip(0,  6*3600)
        df['rel_dev_from_plan']      = df['rel_dev_from_plan'].clip(0.0, 100.0)
        df['rel_dev_from_dynplan']   = df['rel_dev_from_dynplan'].clip(0.0, 100.0)
        # ===============================================================================
        
       
        

        # ─── PROGRESS ONLY ──────────────────────────────────────────
        total = df.groupby(cid)['time:timestamp'].transform('count').astype(np.float32)
        pos   = df.groupby(cid).cumcount() + 1
        df['progress'] = (pos / total).astype(np.float32)

        df['hist_mean_delay'] = df.groupby('concept:name')['delay'].transform('mean').astype(np.float32)
        df['hist_std_delay']  = df.groupby('concept:name')['delay'].transform('std').fillna(0).astype(np.float32)


        # ─── NINE FIRST‑DELAY ENRICHMENTS ──────────────────────────
        first_idx = df[df['is_delayed']==1].groupby(cid,sort=False).head(1).index
        df_fd = df.loc[first_idx]

        # 1) origin & destination first‑delay
        o_stats = df_fd.groupby('case:start')['delay'].agg(o_fd_mean='mean', o_fd_std='std')
        d_stats = df_fd.groupby('case:end')['delay'].agg(d_fd_mean='mean', d_fd_std='std')
        
        df['o_fd_mean'] = df['case:start'].map(o_stats['o_fd_mean']).astype('float32')
        df['o_fd_std']  = df['case:start'].map(o_stats['o_fd_std']).astype('float32')
        df['d_fd_mean'] = df['case:end'].map(d_stats['d_fd_mean']).astype('float32')
        df['d_fd_std']  = df['case:end'].map(d_stats['d_fd_std']).astype('float32')
        
        od = df_fd.groupby(['case:start','case:end'])['delay'].agg(od_fd_mean='mean', od_fd_std='std')
        od = od.reset_index().set_index(['case:start','case:end'])
        
        mi_od = pd.MultiIndex.from_arrays([df['case:start'].values, df['case:end'].values],
                                          names=['case:start','case:end'])
        mapped = od.reindex(mi_od)
        
        df['od_fd_mean'] = mapped['od_fd_mean'].to_numpy(dtype='float32')
        df['od_fd_std']  = mapped['od_fd_std'].to_numpy(dtype='float32')


        def _window_count_per_key(df, time_col, key_col, window='1h'):
            # counts events for the SAME key within (t - window, t], per row
            g = df[[time_col, key_col]].copy()
            g = g.sort_values([key_col, time_col])
            out = np.empty(len(g), dtype=np.int32)
        
            for _, grp in g.groupby(key_col, sort=False):
                t = grp[time_col].values.astype('datetime64[ns]')
                # first index inside the window for each row
                j = np.searchsorted(t, t - np.timedelta64(1, 'h'), side='left')
                i = np.arange(len(grp), dtype=np.int32)
                out[grp.index.values] = i - j + 1
        
            # return in original df order
            result = np.empty(len(df), dtype=np.int32)
            result[g.index.values] = out
            return result
        
        # --- new (fast) ---
        df['cong_o_1h'] = _window_count_per_key(df, 'anchor_ts', 'case:start')
        df['cong_d_1h'] = _window_count_per_key(df, 'anchor_ts', 'case:end')


        # 4) factorized location embeddings
        df['start_code'], _ = pd.factorize(df['case:start'])
        df['end_code'],   _ = pd.factorize(df['case:end'])
        
        
        # ─── CASE‑LEVEL PREFIX STATICS ─────────────────────────────────────
        # prefix length: number of events observed so far in this case
        df['case_length_prefix'] = df.groupby(cid).cumcount() + 1

        # prefix mean inter‑event time up to (and including) this event
        g = df.groupby(self.cid)['inter_event_time']
        df['case_mean_iet_prefix'] = (g.cumsum() / g.cumcount().add(1)).astype(np.float32)


        # planned case duration (only if known at case start; otherwise drop)
        p_start = df.groupby(cid)['planned:timestamp'].transform('min')
        p_end   = df.groupby(cid)['planned:timestamp'].transform('max')
        df['case_planned_dur'] = (
            (p_end - p_start).dt.total_seconds().astype(np.float32)
        )

        cid   = self.cid  # usually 'case:concept:name'
        
        
        # ─── EWMA level & momentum of downstream in-flight (strict time order) ───
        df = _ensure_event_end(df)
        alpha = 0.3
        tmp = df[['event_end','anchor_ts','in_flight_event_end_1h']].copy()
        tmp.sort_values(['event_end','anchor_ts'], inplace=True)
        
        tmp['ewma'] = tmp.groupby('event_end')['in_flight_event_end_1h'] \
                         .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        tmp['mom']  = tmp.groupby('event_end')['ewma'].shift(1).fillna(0)
        
        df['ewma_if_event_end']     = tmp['ewma'].astype(np.float32)
        df['momentum_if_event_end'] = (tmp['ewma'] - tmp['mom']).astype(np.float32)

        
        
        # ─── APPLY FIRST‐EVENT CLUSTER MODEL ────────────────────────
        # pull out one “first‐event” row per case, in original order
        first_ev = (
            df.sort_values([self.cid, 'time:timestamp'], ignore_index=True)
              .groupby(self.cid, sort=False)
              .first()
              .reset_index()
        )

        # 1) time‐of‐day cyclic
        ts0 = first_ev['time:timestamp'].dt.tz_convert(None)
        hrs = ts0.dt.hour.astype(float)
        first_ev['hour_sin'] = np.sin(2 * np.pi * hrs / 24)
        first_ev['hour_cos'] = np.cos(2 * np.pi * hrs / 24)

        # 2) weekday dummies
        wd = ts0.dt.weekday
        wdf = pd.get_dummies(wd, prefix='weekday', dtype=np.int8)
        for i in range(7):
            col = f'weekday_{i}'
            if col not in wdf:
                wdf[col] = 0
        first_ev = pd.concat([first_ev, wdf[[f'weekday_{i}' for i in range(7)]]], axis=1)

        # 3) optional workload (already in df)
        #    we assume active_cases_1h was computed above
        #    if you didn’t compute it, drop next two lines
        # ⇒ first_ev['active_cases_1h'] is already there

        # 4) encode the static categoricals exactly as offline
        for col in ['case:start','case:end','sachnummer','route']:
            first_ev[col] = pd.factorize(first_ev[col])[0]

        # 5) assemble the exact same feature‐matrix you used to train the clusters
        cluster_feats = [
            'hour_sin','hour_cos'
        ] + [f'weekday_{i}' for i in range(7)] + [
            'case:start','case:end','sachnummer','route'
        ]
        if 'active_cases_1h' in first_ev.columns:
            cluster_feats.append('active_cases_1h')

        # 6) scale and predict
        X_new   = self.scaler.transform(first_ev[cluster_feats].values)
        labels  = self.kmeans.predict(X_new)

        # 7) merge back onto every event
        mapping = pd.DataFrame({
            self.cid:      first_ev[self.cid],
            'cluster_id':  labels
        })
        df = df.merge(mapping, on=self.cid, how='left')

        # ─── DROP LEAKAGE CANDIDATES ────────────────────────────────
        for col in [
            'tt_next_planned',
            'tt_next_planned_dyn',
            'remaining_planned_dyn',
            'slack_per_ev_dyn'
        ]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # ─── NEXT-EVENT LABELS ─────────────────────────────────────
        # make sure DataFrame is sorted by (case, time)
        df = df.sort_values([self.cid, 'time:timestamp']).reset_index(drop=True)
        
        # shift labels from event i+1 into row i
        df['next_activity']       = df.groupby(self.cid)['concept:name'].shift(-1)
        df['next_delay']          = df.groupby(self.cid)['delay'].shift(-1)
        df['next_is_delayed']     = df.groupby(self.cid)['is_delayed'].shift(-1)
        # dynamic-delay label too (we'll use this for the teacher)
        df['next_is_delayed_dyn'] = df.groupby(self.cid)['is_delayed_dyn'].shift(-1)
        
        # --- Teacher mask & join keys (multi-category aware) ---
        # Next event's category (needed to know if the NEXT step is a transport)
        if 'category' in df.columns:
            df['next_category'] = df.groupby(self.cid)['category'].shift(-1)
        else:
            df['next_category'] = pd.NA
        
        # Mask for teacher loss: only prefixes where the *next* event is a transport
        df['mask_transport'] = ((df['next_category'] == 'transport').fillna(False).astype('int8'))
        
        # Optional helper: whether the CURRENT row/event itself is a transport
        df['is_transport_start'] = ((df['category'] == 'transport') if 'category' in df.columns else pd.Series(False, index=df.index)).fillna(False).astype('int8')
        
        # Stable per-case position index to join teacher→XGB later
        # (compute before dropping tail rows so the index is consistent within case)
        df['prefix_index'] = df.groupby(self.cid).cumcount().astype('int32')
        
        # Convenience: the teacher's target at masked rows (NaN elsewhere)
        # You can use next_is_delayed_dyn or next_is_delayed; dyn is usually better aligned to the leakage-safe plan.
        df['y_teacher'] = df['next_is_delayed_dyn'].where(df['mask_transport'].eq(1))
        
        # Finally, drop rows that have no next-event
        df = df[df['next_activity'].notna()].reset_index(drop=True)

        
        cid = self.cid        
        
        # right before the final return
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ─── DOWN‑CAST FLOAT64 → FLOAT32 ───────────────────────────
        for c in df.select_dtypes('float64').columns:
            df[c] = df[c].astype(np.float32)
            
        log("remaining features done")
        log("finished all")
            
        return df
    
    
    def trim_to_first_delay_prefix(self, 
        df: pd.DataFrame,
        case_col: str = "case_seg",
        time_col: str = "time:timestamp",
        planned_col: str = "planned:timestamp",
        eventkey_col: str = "eventkey",
        drop_cases_with_k0: bool = True,
    ) -> pd.DataFrame:
        """
        Keep only prefixes up to (and including) the trigger row (k−1) for delayed cases.
        - k = index of first event with is_delayed_dyn==1 within the case
        - keep rows [0..k-1] if k>0
        - if k==0 (first event already delayed): drop the case entirely if drop_cases_with_k0
        - if the case has no delayed events: keep all rows
        """
        # stable sort (case, time, then optional tie-breakers)
        sort_keys = [case_col, time_col]
        if planned_col in df.columns:  sort_keys.append(planned_col)
        if eventkey_col in df.columns: sort_keys.append(eventkey_col)
        df = df.sort_values(sort_keys, kind="stable").copy()
    
        if "is_delayed_dyn" not in df.columns:
            raise ValueError("Column 'is_delayed_dyn' not found. Run the preprocessor first.")
    
        keep_idx = []
    
        for cid, g in df.groupby(case_col, sort=False):
            m = g["is_delayed_dyn"].to_numpy(dtype=bool)
            if m.any():
                k = int(np.argmax(m))  # first delayed event position
                if k > 0:
                    keep_idx.extend(g.index[:k].tolist())  # keep 0..k-1 (trigger is k-1)
                else:
                    # k == 0 → first event is already delayed; drop the case (keep none)
                    if not drop_cases_with_k0:
                        # optional: keep nothing anyway; branch here only for symmetry
                        pass
            else:
                # no-delay case → keep everything
                keep_idx.extend(g.index.tolist())
    
        out = df.loc[keep_idx].copy()
        out = out.sort_values(sort_keys, kind="stable")
        return out
    
    # ─────────────────────────────────────────────────────────────
    # τ-SAMPLING: build "live" observation rows between transports
    # ─────────────────────────────────────────────────────────────
    def _transport_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return one row per transport→transport interval within each case segment:
          cid, k_idx, t0, t_next, y_next_delay, n_events_in_interval
        Assumes df has columns: self.cid, 'category', 'time:timestamp', 'is_delayed_dyn'
        and is sorted by (cid, time).
        """
        cid = self.cid
        g = df.sort_values([cid, 'time:timestamp']).copy()
    
        # transport mask and event times
        is_tr = (g['category'] == 'transport')
        g['__is_tr'] = is_tr.astype(np.int8)
    
        # indices of transport events within each case
        g['__tr_idx'] = g.groupby(cid)['__is_tr'].cumsum() - 1
        # rows that are transports only
        tr = g[g['__is_tr'] == 1][[cid, 'time:timestamp', 'is_delayed_dyn', '__tr_idx']].rename(
            columns={'time:timestamp': 'tr_ts', 'is_delayed_dyn': 'tr_is_delayed'}
        )
    
        # next transport for each transport (within same case)
        tr['k_idx'] = tr['__tr_idx']
        tr['t0']    = tr['tr_ts']
        tr['y_k1']  = tr.groupby(cid)['tr_is_delayed'].shift(-1)   # label of next transport
        tr['t_next']= tr.groupby(cid)['tr_ts'].shift(-1)           # time of next transport
    
        intervals = tr.dropna(subset=['t_next']).copy()
        intervals['y_next_delay'] = intervals['y_k1'].astype(np.int8)
        intervals = intervals[[cid, 'k_idx', 't0', 't_next', 'y_next_delay']]
    
        # count events between t0 and t_next (across all categories)
        j = df[[cid, 'time:timestamp']].copy()
        intervals = intervals.merge(
            j.groupby(cid)['time:timestamp'].agg(list).rename('all_ts').reset_index(),
            on=cid, how='left'
        )
        def _count_between(row):
            if not isinstance(row['all_ts'], list): return 0
            t0, t1 = row['t0'], row['t_next']
            return sum((t > t0) and (t <= t1) for t in row['all_ts'])
        intervals['n_events_in_interval'] = intervals.apply(_count_between, axis=1).astype('int32')
        intervals.drop(columns=['all_ts'], inplace=True, errors='ignore')
        return intervals
    

    def _sample_taus(
        self,
        *,
        t0: pd.Timestamp,
        t_cap: pd.Timestamp,
        strategy: str = "milestones",
        n_grid: int = 3,
        case_ts_i: np.ndarray | None = None,  # int64 ns timestamps of all events in this case (sorted)
        left_idx: int | None = None           # index of first event strictly after t0 in case_ts_i
    ) -> list[pd.Timestamp]:
        """
        Choose observation times τ in (t0, t_cap], where t_cap is typically:
            t_cap = min(t_next, last_seen_event_time_in_case)
        This guarantees we never sample beyond the latest available event.
    
        strategy:
          - 'milestones': τ at {t_cap-60s, t_cap-180s, t_cap-300s} clipped to (t0, t_cap]
          - 'grid'      : n_grid evenly-spaced points in (t0, t_cap]
        """
        if t_cap <= t0:
            return []
    
        taus: list[pd.Timestamp] = []
    
        if strategy == "milestones":
            offsets = [pd.Timedelta(seconds=60),
                       pd.Timedelta(seconds=180),
                       pd.Timedelta(seconds=300)]
            for off in offsets:
                cand = t_cap - off
                if (cand > t0) and (cand <= t_cap):
                    taus.append(cand)
        else:  # grid
            total = (t_cap - t0).total_seconds()
            if total > 0:
                for i in range(1, n_grid + 1):
                    frac = i / (n_grid + 1)
                    taus.append(t0 + pd.to_timedelta(total * frac, unit="s"))
    
        # Ensure unique & sorted, make tz-aware UTC
        taus = sorted(pd.to_datetime(list(set(taus)), utc=True))
        return taus
    
    
    def build_tau_table(
        self,
        df: pd.DataFrame,
        tau_strategy: str = "milestones",
        grid_points: int = 3,
        include_t0: bool = True
    ) -> pd.DataFrame:
        """
        Build τ-rows for student training. Each row is an observation time τ within
        a transport→transport interval, plus optionally τ=t0.
    
        We cap sampling at the latest observed event in each case (t_cap) so we never
        create τ in the “future” where the case has no events yet.
        """
        cid = self.cid
        df = df.copy()
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
        if "category" not in df.columns:
            raise ValueError("build_tau_table() requires 'category' column.")
        df = df.sort_values([cid, "time:timestamp"], kind="stable")
    
        # Intervals (t0 -> t_next) per case; must provide at least columns: cid, k_idx, t0, t_next, y_next_delay
        intervals = self._transport_intervals(df)
        if intervals.empty:
            raise ValueError("No transport→transport intervals found.")
    
        # Per-case event timestamps as int64 ns for fast search
        per_case_ts = {}
        for case, g in df.groupby(cid, sort=False):
            ts_ns = pd.to_datetime(g["time:timestamp"], utc=True).values.astype("datetime64[ns]")
            per_case_ts[case] = ts_ns.view("i8")  # int64 ns, sorted
    
        rows = []
    
        for case, gI in intervals.groupby(cid, sort=False):
            ts_i = per_case_ts[case]                          # int64 ns timestamps
            if ts_i.size == 0:
                continue
            last_seen_i = ts_i[-1]
            last_seen_ts = pd.Timestamp(last_seen_i, tz="UTC", unit="ns")
    
            for _, r in gI.iterrows():
                # t0 / t_next as tz-aware
                t0 = pd.to_datetime(r["t0"], utc=True)
                t_next = pd.to_datetime(r["t_next"], utc=True)
    
                # cap τ sampling at the latest observed event time
                t_cap = min(t_next, last_seen_ts)
    
                # left index of events strictly after t0
                t0_i = np.int64(t0.to_datetime64().view("i8"))
                left_idx = np.searchsorted(ts_i, t0_i, side="right")
    
                # τ candidates
                taus: list[pd.Timestamp] = []
                if include_t0:
                    taus.append(t0)
    
                taus += self._sample_taus(
                    t0=t0,
                    t_cap=t_cap,
                    strategy=tau_strategy,
                    n_grid=grid_points,
                    case_ts_i=ts_i,
                    left_idx=left_idx,
                )
    
                if not taus:
                    continue
    
                # event counts in (t0, τ]
                taus_i = np.array([np.int64(pd.Timestamp(t).to_datetime64().view("i8")) for t in taus], dtype="int64")
                right_idx = np.searchsorted(ts_i, taus_i, side="right")
                events_since = (right_idx - left_idx).astype(np.int32)
                has_post = (events_since > 0).astype(np.int8)
    
                for tau_ts, hp, ev in zip(taus, has_post, events_since):
                    rows.append({
                        cid: case,
                        "k_idx": int(r["k_idx"]),
                        "tau_ts": tau_ts,
                        "t0": t0,
                        "t_next": t_next,
                        "y_next_delay": int(r["y_next_delay"]),
                        "has_post": int(hp),
                        "events_since_t0": int(ev),
                        "time_since_t0_s": float((tau_ts - t0).total_seconds()),
                    })
    
        tau_df = pd.DataFrame(rows)
        if not tau_df.empty:
            tau_df["has_post"] = tau_df["has_post"].astype("int8")
            tau_df["events_since_t0"] = tau_df["events_since_t0"].astype("int32")
            tau_df["time_since_t0_s"] = tau_df["time_since_t0_s"].astype("float32")
        return tau_df

    
    # --- Per-station minute series (reusable) ---------------------------------
    def _build_station_minute_series(self, df: pd.DataFrame):
        """
        Returns:
          agg_actual: per-minute actuals at destination station
                      columns: ['station','ts_min','starts','delayed','delay_sum_s']
          agg_planned: per-minute planned arrivals at destination station
                      columns: ['station','ts_min','planned']
        """
        df = _ensure_event_end(df)
        dest = (df.dropna(subset=['event_end'])
                  [['event_end','time:timestamp','planned:timestamp','is_delayed']]
                  .rename(columns={'event_end':'station','time:timestamp':'event_ts'}))
        dest['ts_min'] = pd.to_datetime(dest['event_ts'].dt.floor('min'), utc=True)
        dest['delay_s'] = (dest['event_ts'] - dest['planned:timestamp']) \
            .dt.total_seconds().clip(lower=0).astype(np.float32)
    
        agg_actual = (
            dest.groupby(['station','ts_min'], observed=True)
                .agg(starts=('event_ts','count'),
                     delayed=('is_delayed','sum'),
                     delay_sum_s=('delay_s','sum'))
                .reset_index()
                .sort_values(['station','ts_min'])
        )
    
        agg_planned = (
            df.dropna(subset=['event_end'])[['event_end','planned:timestamp']]
              .rename(columns={'event_end':'station','planned:timestamp':'ts'})
              .assign(ts_min=lambda x: pd.to_datetime(x['ts'].dt.floor('min'), utc=True))
              .groupby(['station','ts_min'], observed=True).size()
              .rename('planned').reset_index()
              .sort_values(['station','ts_min'])
        )
        return agg_actual, agg_planned
    
    @staticmethod
    def map_station_windows_at(keys_df: pd.DataFrame,
                               agg_actual: pd.DataFrame,
                               agg_planned: pd.DataFrame,
                               windows=(5,15,60)) -> pd.DataFrame:
        """
        keys_df: columns ['station','obs_min'] (tz-aware minute)
        returns: keys_df + starts/delayed/delaysum/planned/backlog/cv for each window
        """
        if keys_df.empty:
            return keys_df.copy()
    
        def _to_min_i(s: pd.Series) -> np.ndarray:
            return (pd.to_datetime(s, utc=True).values.view('i8') // np.int64(60_000_000_000)).astype(np.int64)
    
        # Normalize keys and build int-minute for searchsorted
        k = keys_df.copy()
        k['obs_min'] = pd.to_datetime(k['obs_min'], utc=True)
        k['min_i']   = _to_min_i(k['obs_min'])
    
        out_per_W   = {W: [] for W in windows}
        cv_per_W    = {W: [] for W in windows}
        rows_now    = []
    
        stations = k['station'].dropna().unique()
        for st in stations:
            gk   = k[k['station'].eq(st)]
            need = gk['obs_min']
            need_i = _to_min_i(need)
            if need_i.size == 0:
                continue
        
            # Bound the history we pull for this station to only what's needed for windows
            MAXW   = max(windows)              # e.g., 60
            lo_need = need_i.min() - (MAXW - 1)
            hi_need = need_i.max()
        
            # Pull actual & planned for this station
            a = agg_actual[agg_actual['station'].eq(st)][['ts_min','starts','delayed','delay_sum_s']]
            p = agg_planned[agg_planned['station'].eq(st)][['ts_min','planned']]
        
            # Convert to int-minute and keep only the bounded range
            if not a.empty:
                ai  = _to_min_i(a['ts_min'])
                msk = (ai >= lo_need) & (ai <= hi_need)
                a   = a.loc[msk]
                ai  = ai[msk]
            else:
                ai = np.array([], dtype=np.int64)
        
            if not p.empty:
                pi  = _to_min_i(p['ts_min'])
                msk = (pi >= lo_need) & (pi <= hi_need)
                p   = p.loc[msk]
                pi  = pi[msk]
            else:
                pi = np.array([], dtype=np.int64)
        
            # Dense minute grid only over the bounded window + needed minutes
            mins_all = np.unique(np.concatenate([ai, pi, need_i]))
            if mins_all.size == 0:
                # Fill zeros for this station
                for W in windows:
                    out_per_W[W].append(gk.assign(**{
                        f'starts_{W}':0, f'delayed_{W}':0,
                        f'delaysum_{W}':0.0, f'planned_{W}':0
                    }))
                    cv_per_W[W].append(gk.assign(**{f'cv_{W}':0.0}))
                rows_now.append(gk.assign(c_starts=0, c_delayed=0, c_delay_sum=0.0, c_planned=0))
                continue
        
            M = mins_all.size
            starts  = np.zeros(M, dtype=np.int32)
            delayed = np.zeros(M, dtype=np.int32)
            delays  = np.zeros(M, dtype=np.float32)
            planned = np.zeros(M, dtype=np.int32)
        
            if ai.size:
                ia = np.searchsorted(mins_all, ai).astype(np.intp)
                starts[ia]  = a['starts'].to_numpy(np.int32)
                delayed[ia] = a['delayed'].to_numpy(np.int32)
                delays[ia]  = a['delay_sum_s'].to_numpy(np.float32)
            if pi.size:
                ip = np.searchsorted(mins_all, pi).astype(np.intp)
                planned[ip] = p['planned'].to_numpy(np.int32)
        
            pos = np.clip(np.searchsorted(mins_all, need_i).astype(np.intp), 0, M-1)
        
            c_st = np.cumsum(starts,  dtype=np.int64)
            c_de = np.cumsum(delayed, dtype=np.int64)
            c_ds = np.cumsum(delays,  dtype=np.float64)
            c_pl = np.cumsum(planned, dtype=np.int64)
        
            rows_now.append(gk.assign(
                c_starts=c_st[pos], c_delayed=c_de[pos], c_delay_sum=c_ds[pos], c_planned=c_pl[pos]
            ))
        
            # CV precompute
            s_f = starts.astype(np.float32)
            c1 = np.cumsum(s_f, dtype=np.float64)
            c2 = np.cumsum((s_f**2).astype(np.float64), dtype=np.float64)
        
            for W in windows:
                prev = np.maximum(pos - W, -1)
                sW  = (c_st[pos] - np.where(prev < 0, 0,   c_st[prev])).astype(np.int32)
                dW  = (c_de[pos] - np.where(prev < 0, 0,   c_de[prev])).astype(np.int32)
                dsW = (c_ds[pos] - np.where(prev < 0, 0.0, c_ds[prev])).astype(np.float32)
                pW  = (c_pl[pos] - np.where(prev < 0, 0,   c_pl[prev])).astype(np.int32)
        
                out_per_W[W].append(gk.assign(**{
                    f'starts_{W}': sW, f'delayed_{W}': dW,
                    f'delaysum_{W}': dsW, f'planned_{W}': pW
                }))
        
                # rolling CV on starts
                l = np.maximum(pos - W + 1, 0)
                r = pos
                n = (r - l + 1).astype(np.int64)
                sum1 = c1[r] - np.where(l > 0, c1[l-1], 0.0)
                sum2 = c2[r] - np.where(l > 0, c2[l-1], 0.0)
                mean = sum1 / np.maximum(n, 1)
                ex2  = sum2 / np.maximum(n, 1)
                var  = (ex2 - mean**2) * (np.maximum(n, 1) / np.maximum(n - 1, 1))
                var  = np.where(n <= 1, 0.0, var)
                std  = np.sqrt(np.maximum(var, 0.0))
                cv   = (std / np.maximum(mean, 1e-6)).astype(np.float32)
                cv_per_W[W].append(gk.assign(**{f'cv_{W}': cv}))

        # === Memory-safe stitching: per-station assignment, no global merges ===
        # Prepare an output DataFrame aligned 1:1 to k
        res = k[['station', 'obs_min']].copy()
        
        # initialize output columns
        for W in windows:
            res[f'starts_{W}']   = 0
            res[f'delayed_{W}']  = 0
            res[f'delaysum_{W}'] = np.float32(0.0)
            res[f'planned_{W}']  = 0
            res[f'cv_{W}']       = np.float32(0.0)
        
        # fill from the per-station chunks we already computed
        for W in windows:
            chunks_vals = out_per_W[W]
            chunks_cv   = cv_per_W[W]
        
            if not chunks_vals and not chunks_cv:
                continue
        
            # process station-by-station to keep merges small
            # (both lists contain frames with columns ['station','obs_min', ...])
            # We concatenate first to simplify indexing, then split by station for tiny merges.
            M = pd.concat(chunks_vals, ignore_index=True) if chunks_vals else pd.DataFrame(
                columns=['station','obs_min', f'starts_{W}', f'delayed_{W}', f'delaysum_{W}', f'planned_{W}']
            )
            C = pd.concat(chunks_cv,   ignore_index=True) if chunks_cv else pd.DataFrame(
                columns=['station','obs_min', f'cv_{W}']
            )
        
            # de-dup just in case (keep last)
            if len(M):
                M = (M.drop_duplicates(['station','obs_min'], keep='last')
                       .set_index(['station','obs_min']))
            if len(C):
                C = (C.drop_duplicates(['station','obs_min'], keep='last')
                       .set_index(['station','obs_min']))
        
            # assign per station to avoid global join indexers
            for st, mask in res.groupby('station').groups.items():
                # mask is Index of rows in `res` for this station
                if st not in M.index.get_level_values(0) and st not in C.index.get_level_values(0):
                    continue
        
                res_st = res.loc[mask, ['obs_min']].copy()
                res_st['station'] = st
                res_st = res_st.set_index(['station', 'obs_min'])
        
                if len(M):
                    align_M = M.reindex(res_st.index)
                    if align_M is not None and len(align_M):
                        res.loc[mask, f'starts_{W}']    = align_M.get(f'starts_{W}',   0).fillna(0).to_numpy(dtype=np.int32, copy=False)
                        res.loc[mask, f'delayed_{W}']   = align_M.get(f'delayed_{W}',  0).fillna(0).to_numpy(dtype=np.int32, copy=False)
                        res.loc[mask, f'delaysum_{W}']  = align_M.get(f'delaysum_{W}', 0).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
                        res.loc[mask, f'planned_{W}']   = align_M.get(f'planned_{W}',  0).fillna(0).to_numpy(dtype=np.int32, copy=False)
        
                if len(C):
                    align_C = C.reindex(res_st.index)
                    if align_C is not None and len(align_C):
                        res.loc[mask, f'cv_{W}'] = align_C.get(f'cv_{W}', 0.0).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        
        # backlog per window
        for W in windows:
            res[f'backlog_{W}'] = np.maximum(res[f'planned_{W}'] - res[f'starts_{W}'], 0).astype(np.int32)
        
        return res

    
    def add_tau_congestion_features(self, df: pd.DataFrame, tau_df: pd.DataFrame,
                                    windows=(5,15,60)) -> pd.DataFrame:
        """
        Enriches tau_df with station-window features at τ. Uses the destination
        of transport k_idx as the 'next station' for that interval.
        """
        cid = self.cid
    
        # Build per-transport k_idx within each case segment to get its destination station
        g = df.sort_values([cid, 'time:timestamp']).copy()
        if 'category' not in g.columns:
            raise ValueError("add_tau_congestion_features requires 'category' in df.")
    
        is_tr = (g['category'] == 'transport').astype(np.int8)
        g['k_idx'] = g.groupby(cid)['category'].transform(lambda s: (s.eq('transport')).cumsum() - 1)
        trans = g[is_tr.eq(1)][[cid,'k_idx','event_end']].dropna()
    
        # Attach station and obs_min to tau rows
        tau = tau_df.copy()
        tau = tau.merge(trans, on=[cid,'k_idx'], how='left', validate='many_to_one')
        tau['obs_min'] = pd.to_datetime(tau['tau_ts'], utc=True).dt.floor('min')
        tau = tau.rename(columns={'event_end':'station'})
    
        # Build minute series once and map
        agg_actual, agg_planned = self._build_station_minute_series(df)
        mapped = self.map_station_windows_at(
            tau[['station','obs_min']], agg_actual, agg_planned, windows=windows
        )
    
        # Merge back and rename to the suffix scheme (_event_end)
        tau = tau.merge(mapped, on=['station','obs_min'], how='left')
    
        # Final column names to mirror the event-time block
        for W in windows:
            sfx = f"{W}m_event_end"
            tau[f"starts_{sfx}"]      = tau[f"starts_{W}"].fillna(0).astype('int32')
            tau[f"delayed_{sfx}"]     = tau[f"delayed_{W}"].fillna(0).astype('int32')
            tau[f"delaysum_{sfx}"]    = tau[f"delaysum_{W}"].fillna(0).astype('float32')
            tau[f"planned_arrivals_{sfx}"] = tau[f"planned_{W}"].fillna(0).astype('int32')
            tau[f"backlog_{sfx}"]     = tau[f"backlog_{W}"].fillna(0).astype('int32')
            tau[f"burstiness_cv_{sfx}"] = tau[f"cv_{W}"].fillna(0).astype('float32')
    
            # Optionally delay_rate/avg_delay at τ:
            tau[f"delay_rate_{sfx}"]  = (
                tau[f"delayed_{sfx}"] / np.clip(tau[f"starts_{sfx}"], 1, None)
            ).astype('float32')
            tau[f"avg_delay_{sfx}"]   = (
                tau[f"delaysum_{sfx}"] / np.clip(tau[f"starts_{sfx}"], 1, None)
            ).astype('float32')
    
        # Clean temp columns
        tau.drop(columns=[c for c in tau.columns if c.endswith(('_5','_15','_60','cv_5','cv_15','cv_60',
                                                                'starts_5','starts_15','starts_60',
                                                                'delayed_5','delayed_15','delayed_60',
                                                                'delaysum_5','delaysum_15','delaysum_60',
                                                                'planned_5','planned_15','planned_60',
                                                                'backlog_5','backlog_15','backlog_60',
                                                                'c_starts','c_delayed','c_delay_sum','c_planned'))],
                inplace=True, errors='ignore')
        return tau


    def cut_prefix_upto_tau(self, df_case: pd.DataFrame, tau: pd.Timestamp) -> pd.DataFrame:
        """
        Return a view of df_case with events time:timestamp ≤ τ.
        Assumes df_case already belongs to a single case segment and has 'time:timestamp' as UTC.
        """
        mask = df_case['time:timestamp'] <= tau
        return df_case.loc[mask]
    
    
    def build_anchor_features(
        self,
        df: pd.DataFrame,
        *,
        anchor_col: str = "anchor_ts",
        events_df_for_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Anchor-aware features for BOTH event rows (anchor_ts) and τ rows (tau_ts).
    
        Adds, on top of the existing station windows and 1h resource/subsystem blocks:
          • Next-station planned-to-start (dest side) at anchor:
            next_planned_start, slack_to_planned_start_next,
            planned_arrivals_to_start_next, planned_services_to_start_next,
            planned_pressure_to_start_next,
            backlog_rel_to_planned, pressure_next_*m_rel_planned, pressure_next_*m_per_gapmin,
            gap flags (gap_le_5m/15m/60m), trend delta_pressure_next_15m_60m,
            route_x_gap15_pressure.
          • Edge features at anchor: edge_c_starts_now (cumulative src→dst starts up to anchor).
          • Vehicle features at anchor: veh_events_5m/60m, veh_delayed_60m, veh_avg_delay_60m,
            veh_std_delay_60m (0 if not derivable), veh_delay_rate_60m, veh_time_since_last_s,
            veh_route_changed (carry last-event change), veh_util_ewma_60m.
        """
        df = df.copy()
        df = _ensure_event_end(df)
    
        # --- unify anchor time ---
        df[anchor_col] = pd.to_datetime(df[anchor_col], utc=True, errors="coerce")
        df["__anchor_min"] = df[anchor_col].dt.floor("min")
        NS_PER_MIN = np.int64(60_000_000_000)
        df["__anchor_min_i"] = (df["__anchor_min"].values.view("i8") // NS_PER_MIN).astype(np.int64)
    
        # Use full history (events) for any “past up to anchor” aggregations if provided
        hist_df = events_df_for_history if events_df_for_history is not None else df
        agg_actual, agg_planned = self._build_station_minute_series(hist_df)
    
        # ================== STATION WINDOWS @ anchor (as before) ==================
        keys = (
            df.dropna(subset=["event_end"])[["event_end", "__anchor_min"]]
            .rename(columns={"event_end": "station", "__anchor_min": "obs_min"})
        )
        mapped = self.map_station_windows_at(
            keys.drop_duplicates(["station", "obs_min"]),
            agg_actual,
            agg_planned,
            windows=(5, 15, 60),
        )
        df = df.merge(
            mapped,
            left_on=["event_end", "__anchor_min"],
            right_on=["station", "obs_min"],
            how="left",
        )
        for W in (5, 15, 60):
            sfx = f"{W}m_event_end"
            df[f"starts_{sfx}"] = df[f"starts_{W}"].fillna(0).astype("int32")
            df[f"delayed_{sfx}"] = df[f"delayed_{W}"].fillna(0).astype("int32")
            df[f"delaysum_{sfx}"] = df[f"delaysum_{W}"].fillna(0).astype("float32")
            df[f"planned_arrivals_{sfx}"] = df[f"planned_{W}"].fillna(0).astype("int32")
            df[f"backlog_{sfx}"] = df[f"backlog_{W}"].fillna(0).astype("int32")
            df[f"burstiness_cv_{sfx}"] = df[f"cv_{W}"].fillna(0).astype("float32")
            df[f"delay_rate_{sfx}"] = (
                df[f"delayed_{sfx}"] / np.clip(df[f"starts_{sfx}"], 1, None)
            ).astype("float32")
            df[f"avg_delay_{sfx}"] = (
                df[f"delaysum_{sfx}"] / np.clip(df[f"starts_{sfx}"], 1, None)
            ).astype("float32")
    
        # ===== 1h resource features @ anchor (derived) =====
        df["service_rate_event_end_1h"] = df["starts_60m_event_end"].astype("int32")
        df["delays_count_event_end_1h"] = df["delayed_60m_event_end"].astype("int32")
        df["avg_delay_event_end_1h"] = (
            df["delaysum_60m_event_end"] / np.clip(df["starts_60m_event_end"], 1, None)
        ).astype("float32")
        df["std_delay_event_end_1h"] = np.float32(0.0)  # not reconstructable here
    
        df["planned_60"] = df["planned_arrivals_60m_event_end"].astype("int32")
        df["in_flight_event_end_1h"] = df["service_rate_event_end_1h"]
        df["arrival_rate_event_end_1h"] = df["planned_60"]
        df["delay_rate_event_end_1h"] = (
            df["delays_count_event_end_1h"] / np.clip(df["service_rate_event_end_1h"], 1, None)
        ).astype("float32")
        df["log_delay_rate_event_end_1h"] = np.log1p(df["delay_rate_event_end_1h"]).astype("float32")
        df["backlog_event_end_1h"] = np.maximum(
            df["arrival_rate_event_end_1h"] - df["service_rate_event_end_1h"], 0
        ).astype("int32")
    
        # ================== NEXT-STATION planned-to-start (dest side) @ anchor ==================
        # (1) Next planned start for THIS CASE: first planned:timestamp >= anchor
        cid = getattr(self, "cid", "case_seg")
        # Build per-case sorted planned timestamps (use history DF)
        hist_case_plan = (
            hist_df[[cid, "planned:timestamp"]]
            .dropna()
            .sort_values([cid, "planned:timestamp"])
            .groupby(cid, observed=True)["planned:timestamp"]
            .agg(list)
        )
    
        def _next_plan_after_anchor(row):
            case = row[cid]
            anchor_ts = row[anchor_col]
            lst = hist_case_plan.get(case, [])
            if not lst:
                return pd.NaT
            i = np.searchsorted(np.array(pd.to_datetime(lst, utc=True)), anchor_ts, side="left")
            if i >= len(lst):
                return pd.NaT
            return pd.to_datetime(lst[i], utc=True)
    
        df["next_planned_start"] = df.apply(_next_plan_after_anchor, axis=1)
        df["slack_to_planned_start_next"] = (
            (pd.to_datetime(df["next_planned_start"], utc=True) - df[anchor_col])
            .dt.total_seconds()
            .fillna(0.0)
            .astype(np.float32)
        )
    
        # (2) Planned arrivals/services UNTIL that next planned start at destination station
        # Build per-minute planned series dictionaries → cumulative sums per station
        parr_min = (
            hist_df.dropna(subset=["event_end"])[["event_end", "planned:timestamp"]]
            .rename(columns={"event_end": "station", "planned:timestamp": "ts"})
            .assign(ts_min=lambda x: pd.to_datetime(x["ts"].dt.floor("min"), utc=True))
            .groupby(["station", "ts_min"], observed=True)
            .size()
            .rename("planned_arrivals_per_min")
            .reset_index()
            .sort_values(["station", "ts_min"])
        )
        psrv_min = (
            hist_df.dropna(subset=["event:start"])[["event:start", "planned:timestamp"]]
            .rename(columns={"event:start": "station", "planned:timestamp": "ts"})
            .assign(ts_min=lambda x: pd.to_datetime(x["ts"].dt.floor("min"), utc=True))
            .groupby(["station", "ts_min"], observed=True)
            .size()
            .rename("planned_services_per_min")
            .reset_index()
            .sort_values(["station", "ts_min"])
        )
    
        def _prep_cums(frame: pd.DataFrame, value_col: str):
            out = {}
            if frame is None or frame.empty:
                return out
            for st, g in frame.groupby("station", observed=True):
                mins = (pd.to_datetime(g["ts_min"], utc=True).values.view("i8") // NS_PER_MIN).astype(np.int64)
                vals = g[value_col].to_numpy(np.int32)
                # collapse duplicates
                if mins.size > 1:
                    uniq, idx = np.unique(mins, return_index=True)
                    if uniq.size != mins.size:
                        acc = {}
                        for m, v in zip(mins, vals):
                            acc[m] = acc.get(m, 0) + v
                        mins = np.fromiter(acc.keys(), dtype=np.int64)
                        vals = np.fromiter(acc.values(), dtype=np.int32)
                        order = np.argsort(mins)
                        mins, vals = mins[order], vals[order]
                csum = np.cumsum(vals, dtype=np.int64)
                out[st] = (mins, csum)
            return out
    
        parr_cums = _prep_cums(parr_min, "planned_arrivals_per_min")
        psrv_cums = _prep_cums(psrv_min, "planned_services_per_min")
    
        def _range_count(cums, station, lo_i, hi_i):
            # count in (lo, hi] on minute grid
            if station not in cums:
                return 0
            mins, csum = cums[station]
            r = np.searchsorted(mins, hi_i, side="right") - 1
            if r < 0:
                return 0
            l = np.searchsorted(mins, lo_i, side="right") - 1
            left = 0 if l < 0 else csum[l]
            return int(csum[r] - left) if r >= l else 0
    
        anchor_i = (df[anchor_col].values.view("i8") // NS_PER_MIN).astype(np.int64)
        next_i = np.where(
            pd.isna(df["next_planned_start"]),
            anchor_i,
            (pd.to_datetime(df["next_planned_start"], utc=True).values.view("i8") // NS_PER_MIN).astype(np.int64),
        )
        st_vec = df["event_end"].values
    
        arr2start = np.empty(len(df), dtype=np.int32)
        srv2start = np.empty(len(df), dtype=np.int32)
        for i in range(len(df)):
            lo = anchor_i[i]
            hi = next_i[i]
            if hi <= lo or pd.isna(st_vec[i]):
                arr2start[i] = 0
                srv2start[i] = 0
            else:
                st = st_vec[i]
                arr2start[i] = _range_count(parr_cums, st, lo, hi)
                srv2start[i] = _range_count(psrv_cums, st, lo, hi)
    
        df["planned_arrivals_to_start_next"] = arr2start
        df["planned_services_to_start_next"] = srv2start
        df["planned_pressure_to_start_next"] = (
            df["backlog_event_end_1h"].astype(np.int64)  # current backlog proxy
            + df["planned_arrivals_to_start_next"].astype(np.int64)
            - df["planned_services_to_start_next"].astype(np.int64)
        ).astype(np.int32)
    
        # (3) Normalized & flags & trends
        df["planned_gap_sec"] = (
            (pd.to_datetime(df["next_planned_start"], utc=True) - df[anchor_col]).dt.total_seconds()
        ).fillna(np.inf).astype(np.float32)
        df["gap_le_5m"]  = (df["planned_gap_sec"] <=  5*60).astype(np.int8)
        df["gap_le_15m"] = (df["planned_gap_sec"] <= 15*60).astype(np.int8)
        df["gap_le_60m"] = (df["planned_gap_sec"] <= 60*60).astype(np.int8)
    
        # Use already mapped windows for “pressure_next_*m” (past W up to anchor) as a leakage-safe proxy:
        # pressure_next_Wm := arrivals (planned) - services (actual starts) in last W minutes at dest.
        # We can approximate with backlog deltas: planned - actual starts over W.
        # Build these from mapped columns we already have.
        for W in (5, 15, 60):
            # arrivals proxy = planned_{W}, services proxy = starts_{W}
            df[f"arrivals_next_{W}m"] = df[f"planned_{W}"].fillna(0).astype("int32") if f"planned_{W}" in df.columns else 0
            df[f"services_next_{W}m"] = df[f"starts_{W}"].fillna(0).astype("int32") if f"starts_{W}" in df.columns else 0
            df[f"pressure_next_{W}m"] = (df[f"arrivals_next_{W}m"] - df[f"services_next_{W}m"]).astype("int32")
    
        df["delta_pressure_next_15m_60m"] = (df["pressure_next_15m"] - df["pressure_next_60m"]).astype("int32")
    
        eps = 1e-6
        planned_h = (
            df["planned_arrivals_to_start_next"].astype(np.float32)
            + df["planned_services_to_start_next"].astype(np.float32)
        ).clip(lower=0.0)
        df["backlog_rel_to_planned"] = (
            df["backlog_event_end_1h"].astype(np.float32) / (planned_h + eps)
        ).astype(np.float32)
        for W in (5, 15, 60):
            df[f"pressure_next_{W}m_rel_planned"] = (
                df[f"pressure_next_{W}m"].astype(np.float32) / (planned_h + eps)
            ).astype(np.float32)
        gap_min = (df["planned_gap_sec"] / 60.0).clip(lower=1.0)
        for W in (5, 15, 60):
            df[f"pressure_next_{W}m_per_gapmin"] = (
                df[f"pressure_next_{W}m"].astype(np.float32) / gap_min
            ).astype(np.float32)
    
        # if event_route_code is present, provide the interaction:
        if "event_route_code" in df.columns:
            df["route_x_gap15_pressure"] = (
                df["event_route_code"].astype("int32")
                * df["gap_le_15m"].astype("int32")
                * (df["pressure_next_15m"] > 0).astype("int32")
            ).astype("int32")
    
        # ================== EDGE FEATURES @ anchor ==================
        # cumulative starts on (src=event:start, dst=event_end) up to anchor
        # Build from history once
        edge_hist = hist_df[["event:start", "event:end", "time:timestamp"]].dropna().copy()
        edge_hist.rename(columns={"event:start": "src", "event:end": "dst", "time:timestamp": "event_ts"}, inplace=True)
        edge_hist["key_min"] = pd.to_datetime(edge_hist["event_ts"], utc=True).dt.floor("min")
        edge_agg = (
            edge_hist.groupby(["src", "dst", "key_min"], observed=True)
            .size()
            .rename("starts")
            .reset_index()
            .sort_values(["src", "dst", "key_min"])
        )
        edge_agg["c_starts"] = edge_agg.groupby(["src", "dst"], observed=True)["starts"].cumsum()
    
        df["__edge_src"] = df.get("event:start", pd.NA)
        df["__edge_dst"] = df.get("event_end", pd.NA)
        edge_map = (
            edge_agg.rename(columns={"key_min": "__anchor_min"})
            .set_index(["src", "dst", "__anchor_min"])["c_starts"]
        )
        mi_edge = pd.MultiIndex.from_arrays(
            [df["__edge_src"].values, df["__edge_dst"].values, df["__anchor_min"].values],
            names=["src", "dst", "__anchor_min"],
        )
        df["edge_c_starts_now"] = edge_map.reindex(mi_edge).fillna(0).to_numpy(dtype="int32")
    
        # ================== VEHICLE FEATURES @ anchor ==================
        if "vehicle_id" in df.columns:
            # Rolling windows over the *history* but evaluated at anchor times
            veh_tmp = hist_df[["vehicle_id", "time:timestamp", "is_delayed", "delay"]].copy()
            veh_tmp["time:timestamp"] = pd.to_datetime(veh_tmp["time:timestamp"], utc=True)
            veh_tmp["delay_pos"] = veh_tmp["delay"].where(veh_tmp["is_delayed"].fillna(0).astype(int).eq(1))
            veh_tmp.sort_values(["vehicle_id", "time:timestamp"], kind="mergesort", inplace=True)
    
            gveh = veh_tmp.set_index("time:timestamp").groupby("vehicle_id", group_keys=False, observed=True)
            roll5 = gveh.rolling("5min",  closed="both")
            roll1 = gveh.rolling("60min", closed="both")
    
            veh_5  = roll5["is_delayed"].count().astype("int32").rename("veh_events_5m")
            veh_60 = roll1["is_delayed"].count().astype("int32").rename("veh_events_60m")
            vdl_60 = roll1["is_delayed"].sum().astype("int32").rename("veh_delayed_60m")
            vad_60 = roll1["delay_pos"].mean().fillna(0.0).astype("float32").rename("veh_avg_delay_60m")
            vsd_60 = roll1["delay_pos"].std().fillna(0.0).astype("float32").rename("veh_std_delay_60m")
    
            vfast = pd.concat([veh_5, veh_60, vdl_60, vad_60, vsd_60], axis=1)
            vfast.index.names = ["vehicle_id", "anchor_ts"]  # reuse index name
    
            # dedupe
            vfast = vfast[~vfast.index.duplicated(keep="last")]
    
            # align to requested anchor times in df
            mi = pd.MultiIndex.from_arrays(
                [df["vehicle_id"].values, df[anchor_col].values],
                names=["vehicle_id", "anchor_ts"],
            )
            valigned = vfast.reindex(mi)
    
            df["veh_events_5m"]     = valigned["veh_events_5m"].fillna(0).to_numpy(dtype="int32")
            df["veh_events_60m"]    = valigned["veh_events_60m"].fillna(0).to_numpy(dtype="int32")
            df["veh_delayed_60m"]   = valigned["veh_delayed_60m"].fillna(0).to_numpy(dtype="int32")
            df["veh_avg_delay_60m"] = valigned["veh_avg_delay_60m"].fillna(0.0).to_numpy(dtype="float32")
            df["veh_std_delay_60m"] = valigned["veh_std_delay_60m"].fillna(0.0).to_numpy(dtype="float32")
            df["veh_delay_rate_60m"] = (
                df["veh_delayed_60m"] / np.clip(df["veh_events_60m"], 1, None)
            ).astype("float32")
    
            # time since last event of this vehicle up to anchor
            last_time = (
                hist_df.sort_values(["vehicle_id", "time:timestamp"])
                .groupby("vehicle_id", observed=True)["time:timestamp"]
                .tail(1)
            )
            # more accurate: per (vehicle, anchor) find the last event ≤ anchor
            veh_last = (
                hist_df[["vehicle_id", "time:timestamp"]]
                .rename(columns={"time:timestamp": "ev_ts"})
            )
            veh_last["ev_ts"] = pd.to_datetime(veh_last["ev_ts"], utc=True)
            # map per (vehicle, anchor) via groupby apply
            def _last_before(dfv):
                # dfv: rows for one vehicle at anchor times from main df
                vt = veh_last[veh_last["vehicle_id"].eq(dfv.name)]["ev_ts"].values
                if vt.size == 0:
                    return np.zeros(len(dfv), dtype=np.float32)
                vt = np.sort(vt)
                anchors = dfv[anchor_col].values
                out = np.zeros(len(dfv), dtype=np.float32)
                for i, a in enumerate(anchors):
                    idx = np.searchsorted(vt, a, side="right") - 1
                    if idx >= 0:
                        out[i] = float((a - vt[idx]).astype("timedelta64[s]").astype(np.int64))
                    else:
                        out[i] = 0.0
                return out
    
            df["veh_time_since_last_s"] = (
                df.groupby("vehicle_id", observed=True, sort=False)
                  .apply(_last_before)
                  .explode()
                  .astype(np.float32)
                  .to_numpy()
            )
    
            # route_changed: carry the last event-level change before anchor
            if "event_route_code" in hist_df.columns:
                tmp = hist_df[["vehicle_id", "time:timestamp", "event_route_code"]].dropna().copy()
                tmp.sort_values(["vehicle_id", "time:timestamp"], inplace=True)
                tmp["veh_route_changed_ev"] = (
                    tmp.groupby("vehicle_id", observed=True)["event_route_code"].diff().fillna(0).ne(0).astype("int8")
                )
                # carry-forward to anchor: for each (veh, anchor), take the last known indicator
                tmp["ts_min"] = pd.to_datetime(tmp["time:timestamp"], utc=True).dt.floor("min")
                last_change = (
                    tmp.set_index(["vehicle_id", "ts_min"])["veh_route_changed_ev"]
                    .groupby(level=0)
                    .cumsum()  # cumulative count of changes
                )
                tmp["change_count"] = last_change.values
                # align to anchor
                req = (
                    df[["vehicle_id", "__anchor_min"]]
                    .rename(columns={"__anchor_min": "ts_min"})
                    .set_index(["vehicle_id", "ts_min"])
                )
                aligned = tmp.set_index(["vehicle_id", "ts_min"]).reindex(req.index, method="ffill")["change_count"]
                # a boolean: changed at/just before anchor if cumulative count increased recently
                df["veh_route_changed"] = (aligned.diff().fillna(0) > 0).astype("int8").to_numpy()
            else:
                df["veh_route_changed"] = np.int8(0)
    
            # EWMA utilization (events_60m) along each vehicle's anchor timeline
            df.sort_values(["vehicle_id", anchor_col], kind="mergesort", inplace=True)
            df["veh_util_ewma_60m"] = (
                df.groupby("vehicle_id", observed=True)["veh_events_60m"]
                  .transform(lambda s: s.ewm(alpha=0.3, adjust=False).mean())
                  .astype("float32")
            )
        else:
            for col in [
                "veh_events_5m","veh_events_60m","veh_delayed_60m",
                "veh_avg_delay_60m","veh_std_delay_60m","veh_delay_rate_60m",
                "veh_time_since_last_s","veh_route_changed","veh_util_ewma_60m"
            ]:
                df[col] = 0 if col not in ("veh_avg_delay_60m","veh_std_delay_60m") else 0.0
                
        # --- EWMA & momentum of in-flight at the destination, evaluated at the anchor ---
        df.sort_values(['event_end', anchor_col], kind='mergesort', inplace=True)
        df['ewma_if_event_end'] = (
            df.groupby('event_end', observed=True)['in_flight_event_end_1h']
              .transform(lambda s: s.ewm(alpha=0.3, adjust=False).mean())
              .astype('float32')
        )
        df['momentum_if_event_end'] = (
            df['ewma_if_event_end'] - df.groupby('event_end')['ewma_if_event_end'].shift(1).fillna(0)
        ).astype('float32')

    
        # ================== CLEANUP TEMPS ==================
        df.drop(
            columns=[c for c in [
                "station","obs_min",
                "starts_5","starts_15","starts_60",
                "delayed_5","delayed_15","delayed_60",
                "delaysum_5","delaysum_15","delaysum_60",
                "planned_5","planned_15","planned_60",
                "backlog_5","backlog_15","backlog_60",
                "cv_5","cv_15","cv_60",
                "__edge_src","__edge_dst",
                "__anchor_min","__anchor_min_i",
            ] if c in df.columns],
            inplace=True, errors="ignore",
        )
    
        # Replace inf/NaN safeguards
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
        return df






    def split_and_save(
        self, df: pd.DataFrame, original_file_path: str,
        test_size: float = 0.2, val_size: float = 0.1,
        random_state: int = 42,  # kept for signature compatibility; unused here
        train_filename: str = 'train_with_events_last6h_final.csv',
        val_filename: str   = 'val_with_events_last6h_final.csv',
        test_filename: str  = 'test_with_events_last6h_final.csv',
        label_col: str = 'next_is_delayed_dyn',
        file_format: str = 'parquet',          # 'parquet' or 'csv'
        compression: str = 'snappy'            # parquet compression: 'snappy' (default), 'gzip', 'brotli', etc.
    ) -> tuple:
        """
        Time-linear, case-wise split with purge gaps to reduce temporal leakage.
        Saves splits as either Parquet or CSV based on `file_format`.
    
        Returns: (train_path, val_path, test_path)
        """
    
        # ---- helpers ----
        def _with_ext(name: str, new_ext: str) -> str:
            base, _ = os.path.splitext(name)
            return base + new_ext
    
        def _fmt_bytes(n: int) -> str:
            for unit in ['B','KB','MB','GB','TB']:
                if n < 1024:
                    return f"{n:.1f}{unit}"
                n /= 1024
            return f"{n:.1f}PB"
    
        def _save_frame(frame: pd.DataFrame, path: str):
            print(f"  -> writing {file_format.upper()} to {path} ... "
                  f"(rows={len(frame):,}, mem≈{frame.memory_usage(deep=True).sum()/1e6:.1f} MB)")
            if file_format == 'parquet':
                # Parquet needs pyarrow or fastparquet; pyarrow recommended.
                try:
                    frame.to_parquet(path, index=False, compression=compression)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to write Parquet. Ensure pyarrow is installed. Original error: {e}"
                    )
            elif file_format == 'csv':
                frame.to_csv(path, index=False)
            else:
                raise ValueError("file_format must be 'parquet' or 'csv'.")
    
            try:
                size = os.path.getsize(path)
                print(f"     done: {path} ({_fmt_bytes(size)})")
            except Exception:
                print(f"     done: {path}")
    
        # ---- resolve output paths/extension ----
        base_dir = os.path.dirname(os.path.abspath(original_file_path))
        ext = '.parquet' if file_format == 'parquet' else '.csv'
    
        train_path = os.path.join(base_dir, _with_ext(train_filename, ext))
        val_path   = os.path.join(base_dir, _with_ext(val_filename,   ext))
        test_path  = os.path.join(base_dir, _with_ext(test_filename,  ext))
    
        cid = self.cid
        ts_col = 'time:timestamp'
    
        if label_col not in df.columns:
            raise ValueError(f"Split requires '{label_col}' in df. "
                             f"Run preprocessing/labeling first or change label_col.")
    
        # ----- per-case anchor time (first event) and label
        case_first_ts = (
            df.groupby(cid, observed=True)[ts_col]
              .min()
              .rename('first_ts')
        )
        case_y = (
            df.groupby(cid, observed=True)[label_col]
              .max()
              .astype(int)  # 0/1 per case
        )
    
        meta = (
            pd.concat([case_first_ts, case_y], axis=1)
              .sort_values('first_ts')   # chronological order
              .reset_index()
        )
    
        n_cases = len(meta)
        if n_cases < 10:
            raise ValueError(f"Too few cases ({n_cases}) for a time-based split.")
    
        # ----- fold sizes (chronological)
        n_test = max(1, int(round(test_size * n_cases)))
        n_val  = max(1, int(round(val_size  * n_cases)))
        n_train = n_cases - n_val - n_test
        if n_train <= 0:
            raise ValueError("Split sizes leave no training cases. Reduce val/test_size.")
    
        # ----- chronological cut indices (no shuffling)
        test_start_idx = n_cases - n_test
        val_start_idx  = n_cases - n_test - n_val
    
        # ----- purge gaps to reduce boundary leakage (hardcoded to 3 days)
        purge_gap = pd.Timedelta(days=3)
    
        t_val_boundary  = meta.loc[val_start_idx,  'first_ts']
        t_test_boundary = meta.loc[test_start_idx, 'first_ts']
    
        # Train: strictly before (t_val_boundary - purge)
        train_cases = meta.loc[meta['first_ts'] < (t_val_boundary - purge_gap), cid]
    
        # Val: from t_val_boundary to before (t_test_boundary - purge)
        val_mask = (meta['first_ts'] >= t_val_boundary) & (meta['first_ts'] < (t_test_boundary - purge_gap))
        val_cases = meta.loc[val_mask, cid]
    
        # Test: from t_test_boundary onward
        test_cases = meta.loc[meta['first_ts'] >= t_test_boundary, cid]
    
        # Safety: if the purge wipes out val, relax to boundary without purge
        if len(val_cases) == 0:
            val_cases = meta.loc[(meta['first_ts'] >= t_val_boundary) & (meta['first_ts'] < t_test_boundary), cid]
    
        # ----- build dataframes (case-wise)
        print("[time-split] building splits ...")
        df_train = df[df[cid].isin(set(train_cases))].copy()
        df_val   = df[df[cid].isin(set(val_cases))].copy()
        df_test  = df[df[cid].isin(set(test_cases))].copy()
    
        # Sanity: ensure disjoint
        assert set(df_train[cid]).isdisjoint(df_val[cid])
        assert set(df_train[cid]).isdisjoint(df_test[cid])
        assert set(df_val[cid]).isdisjoint(df_test[cid])
    
        # ----- save
        print(f"[time-split] saving as {file_format.upper()} (compression={compression if file_format=='parquet' else 'n/a'})")
        _save_frame(df_train, train_path)
        _save_frame(df_val,   val_path)
        _save_frame(df_test,  test_path)
    
        # ----- report
        def _rate(df_sub):
            return float(df_sub[label_col].mean()) if len(df_sub) else float('nan')
    
        print(f"[time-split] train={df_train[cid].nunique()}  "
              f"val={df_val[cid].nunique()}  test={df_test[cid].nunique()} cases")
        print(f"  first_ts  train≤{(t_val_boundary - purge_gap)}   "
              f"val≥{t_val_boundary}   test≥{t_test_boundary}")
        print(f"  delay rate: train={_rate(df_train):.3f}  "
              f"val={_rate(df_val):.3f}  test={_rate(df_test):.3f}")
    
        return train_path, val_path, test_path


    def compute_case_splits(self, df: pd.DataFrame,
                            test_size: float = 0.2, val_size: float = 0.1,
                            purge_days: int = 3):
        cid = self.cid
        ts_col = 'time:timestamp'
        case_first_ts = df.groupby(cid, observed=True)[ts_col].min().rename('first_ts')
        meta = case_first_ts.sort_values().reset_index()
    
        n_cases = len(meta)
        n_test = max(1, int(round(test_size * n_cases)))
        n_val  = max(1, int(round(val_size  * n_cases)))
        test_start_idx = n_cases - n_test
        val_start_idx  = n_cases - n_test - n_val
    
        purge_gap = pd.Timedelta(days=purge_days)
        t_val_boundary  = meta.loc[val_start_idx,  'first_ts']
        t_test_boundary = meta.loc[test_start_idx, 'first_ts']
    
        train_cases = meta.loc[meta['first_ts'] < (t_val_boundary - purge_gap), cid]
        val_mask    = (meta['first_ts'] >= t_val_boundary) & (meta['first_ts'] < (t_test_boundary - purge_gap))
        val_cases   = meta.loc[val_mask, cid]
        if len(val_cases) == 0:  # relax if purge wipes val
            val_cases = meta.loc[(meta['first_ts'] >= t_val_boundary) & (meta['first_ts'] < t_test_boundary), cid]
        test_cases  = meta.loc[meta['first_ts'] >= t_test_boundary, cid]
    
        return set(train_cases), set(val_cases), set(test_cases)


if __name__ == "__main__":
    src = "data/raw/events.csv"
    loader = EventLogLoader(src, delimiter=",")
    df_raw = loader.load_df()     

    pre    = EventLogPreprocessor()
    df     = pre.transform(df_raw)
    print(f"Transformed {len(df)} events across {df[pre.cid].nunique()} cases.")
    
    #output_path = "data/raw/events.csv"
    #df.to_csv(output_path, index=False)
    #print(f"Saved to {output_path}")

    output_path = "data/processed/events_preprocessed.parquet"
    
    print("Saving as Parquet (fast)...")
    df.to_parquet(output_path, index=False)
    print(f"Done. Saved to {output_path}")

    train_csv, val_csv, test_csv = pre.split_and_save(df, original_file_path=src, file_format="parquet")
    print("Splits:", train_csv, val_csv, test_csv)
    
    # τ-dataset (already built)
    tau_df = pre.build_tau_table(df, tau_strategy='milestones', grid_points=3, include_t0=True)
    
    # τ congestion windows at next station (τ-safe)
    tau_df = pre.add_tau_congestion_features(df, tau_df)
    
    # τ generic anchor features (reuse the event-time builder at τ)
    # NOTE: rename to expose an 'event_end' column if not present (add_tau_congestion_features created 'station')
    tau_for_anchor = tau_df.rename(columns={'station': 'event_end'}) if 'station' in tau_df.columns else tau_df.copy()
    tau_for_anchor = pre.build_anchor_features(
        tau_for_anchor,
        anchor_col='tau_ts',
        events_df_for_history=df
    )
    
    # 1) choose columns to bring over from tau_for_anchor
    want_suffixes = ('_event_end', '_1h')
    all_anchor_cols = [c for c in tau_for_anchor.columns if c.endswith(want_suffixes)]
    
    # only bring columns that either (a) end with _1h (resource features) or (b) don't exist yet
    existing_left = set(tau_df.columns)
    keep_from_anchor = [c for c in all_anchor_cols if (c.endswith('_1h') or c not in existing_left)]
    
    # nothing to do? bail early
    if keep_from_anchor:
        # 2) align key dtypes
        for _df in (tau_df, tau_for_anchor):
            _df[pre.cid] = _df[pre.cid].astype('string')
            _df['k_idx'] = _df['k_idx'].astype('int32')
            _df['tau_ts'] = pd.to_datetime(_df['tau_ts'], utc=True)
    
        # 3) build the right side, de-duped to many-to-one
        right = tau_for_anchor[[pre.cid, 'k_idx', 'tau_ts'] + keep_from_anchor].copy()
        right = right.drop_duplicates([pre.cid, 'k_idx', 'tau_ts'], keep='last')
        right = right.set_index([pre.cid, 'k_idx', 'tau_ts'])
    
        # 4) left side indexed
        left_idx = tau_df.set_index([pre.cid, 'k_idx', 'tau_ts'])
    
        # 5) join in chunks to control peak memory; no overlaps because keep_from_anchor
        CHUNK = 40
        for i in range(0, len(keep_from_anchor), CHUNK):
            cols = keep_from_anchor[i:i+CHUNK]
            joined = left_idx.join(right[cols], how='left')
            # write back just these columns
            for c in cols:
                tau_df[c] = joined[c].values
            del joined  # free memory
    
    # carry static transport context from t0 (dispatch row for k_idx)
    tau_df = pre.attach_t0_static_context(df, tau_df, cols=[
        'route','case:start','case:end','event_route','event_route_code',
        'vehicle_id','sachnummer','start_code','end_code','cluster_id'
    ])
    
    # add simple τ counts
    tau_df = pre.add_case_arrival_and_active_cases(tau_df, anchor_col="tau_ts")
    
    # (after you have distilled outputs from the trained teacher)
    # teacher_outputs must have: [pre.cid, 'k_idx', 'tau_ts', 'teacher_logit', 'emb_pca_0', ...]
    # Example: teacher_outputs = distill_tau_split(...)  # the existing step
    # If you don’t have embeddings, just merge the logit.
    # tau_df = pre.attach_teacher_distillation(tau_df, teacher_outputs)
    
    # Split τ by the same case splits
    train_cases, val_cases, test_cases = pre.compute_case_splits(df, test_size=0.2, val_size=0.1, purge_days=3)
    
    tau_train = tau_df[tau_df[pre.cid].isin(train_cases)].copy()
    tau_val   = tau_df[tau_df[pre.cid].isin(val_cases)].copy()
    tau_test  = tau_df[tau_df[pre.cid].isin(test_cases)].copy()
    
    # (optional) final tidy: keep just the student columns you want
    student_cols_meta = [pre.cid, 'k_idx', 'tau_ts', 't0', 't_next', 'time_since_t0_s', 'events_since_t0', 'has_post']
    student_cols_static = ['route','case:start','case:end','event_route','event_route_code',
                           'vehicle_id','sachnummer','start_code','end_code','cluster_id']
    student_cols_dyn_tau = [c for c in tau_df.columns if any(
        c.startswith(p) for p in (
            'starts_', 'delayed_', 'delaysum_', 'delay_rate_', 'avg_delay_',
            'planned_arrivals_', 'backlog_', 'burstiness_cv_',
            'service_rate_', 'delays_count_', 'avg_delay_', 'std_delay_', 'planned_',
            'in_flight_', 'arrival_rate_', 'delay_rate_', 'log_delay_rate_',
        )
    ) and c.endswith('_event_end')] + [  # _event_end from build_anchor_features
        # from add_tau_congestion_features (these already end with _event_end)
    ]
    
    # teacher columns if present
    teacher_cols = [c for c in tau_df.columns if c.startswith('teacher_logit') or c.startswith('emb_pca_')]
    
    #student_cols = list(dict.fromkeys(student_cols_meta + student_cols_static + student_cols_dyn_tau + teacher_cols + ['y_next_delay']))
    #tau_train = tau_train[[c for c in student_cols if c in tau_train.columns]]
    #tau_val   = tau_val[[c for c in student_cols if c in tau_val.columns]]
    
    base_dir  = os.path.dirname(src)
    tau_train_path = os.path.join(base_dir, "tau_train.parquet")
    tau_val_path   = os.path.join(base_dir, "tau_val.parquet")
    tau_test_path  = os.path.join(base_dir, "tau_test.parquet")
    
    #tau_train.to_csv(tau_train_path, index=False)
    #tau_val.to_csv(tau_val_path, index=False)
    #tau_test.to_csv(tau_test_path, index=False)
    
    tau_train.to_parquet(tau_train_path, index=False)
    tau_val.to_parquet(tau_val_path, index=False)
    tau_test.to_parquet(tau_test_path, index=False)

    print("τ-splits:", tau_train_path, tau_val_path, tau_test_path)

    
    #xgb_bg, lstm_bg = pre.create_background_datasets(df)
    #print("Background samples:", xgb_bg, lstm_bg)
# -*- coding: utf-8 -*-
"""
features_xgb.py

Light, idempotent feature materialization for the XGBoost student model.

Provides
--------
- z-normalization for selected numeric process features
- cyclic hour encoding
- congestion-related feature interactions
- X/y assembly for prefix-based and tau-based prediction datasets
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Feature denylist
# ============================================================

DENY: set[str] = {
    # Direct targets / future information
    "progress", "progress_z",
    "next_activity", "next_delay", "next_is_delayed", "next_category",
    "is_delayed",
    "y_teacher",

    # Planned inter-event time and variants
    "planned_iet", "planned_iet_z",

    # Realized lateness of the current event
    "delay", "delay_z",
    "dynamic_delay", "dynamic_delay_z",
    "dev_from_plan_s", "dev_from_plan_s_z",
    "dev_from_dynplan_s", "dev_from_dynplan_s_z",
    "abs_dev_from_plan_s", "abs_dev_from_plan_s_z",
    "abs_dev_from_dynplan_s", "abs_dev_from_dynplan_s_z",
    "closeness_to_plan", "closeness_to_plan_z",
    "closeness_to_dynplan", "closeness_to_dynplan_z",

    # Next-step planning / lookahead variables
    "planned_gap_sec", "planned_gap_sec_z",
    "slack_to_planned_start_next", "slack_to_planned_start_next_z",
    "op_gap_sec_to_next_planned_start", "op_gap_sec_to_next_planned_start_z",
    "planned_arrivals_to_start_next", "planned_arrivals_to_start_next_z",
    "planned_services_to_start_next", "planned_services_to_start_next_z",
    "planned_pressure_to_start_next", "planned_pressure_to_start_next_z",
    "op_planned_services_to_start_next", "op_planned_services_to_start_next_z",
    "op_planned_pressure_to_start_next", "op_planned_pressure_to_start_next_z",
    "srcnext_planned_services_to_start_next", "srcnext_planned_services_to_start_next_z",
    "srcnext_planned_pressure_to_start_next", "srcnext_planned_pressure_to_start_next_z",
    "planned_horizon_until_start", "planned_horizon_until_start_z",

    # Direct structural / non-feature columns
    "mask_transport",
    "gap_le_5m",
    "gap_le_15m",
    "gap_le_60m",
    "phase_is_finish",
    "phase_is_start",
    "event_route_code",
    "start_code",
    "end_code",
    "has_post",
    "events_since_t0",

    # Historical target-like or teacher leakage-prone columns
    "hist_mean_delay",
    "hist_mean_delay_z",
    "teacher_logit_prev1",
}


# ============================================================
# Helpers
# ============================================================

def _is_binary_like(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False

    values = pd.unique(series.dropna())

    return len(values) <= 2 and set(map(float, values)) <= {0.0, 1.0}


def _safe_to_float32(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    array[~np.isfinite(array)] = 0.0
    return array


# ============================================================
# Z-normalization
# ============================================================

def add_global_z_inplace(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """
    Create z-scored versions of selected numeric columns on the given dataframe.

    Notes
    -----
    - Statistics are computed only on the provided dataframe.
    - Binary-like columns are skipped.
    - Existing *_z columns are not overwritten.
    """
    added: List[str] = []

    for col in cols:
        z_col = f"{col}_z"

        if col in DENY or z_col in DENY:
            continue

        if (
            col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and not _is_binary_like(df[col])
            and not col.endswith("_z")
            and not col.endswith("_z_arr")
            and z_col not in df.columns
        ):
            series = df[col].astype(float)

            mu = float(np.nanmean(series))
            sd = float(np.nanstd(series, ddof=0))

            if not np.isfinite(sd) or sd == 0.0:
                sd = 1.0

            df[z_col] = ((series - mu) / sd).astype(np.float32)
            added.append(z_col)

    return added


def _auto_detect_congestion_cols(df: pd.DataFrame) -> List[str]:
    """
    Detect numeric congestion, rate, queue, and schedule-alignment columns
    that should receive z-scored variants.
    """
    known_exact = {
        "service_rate_event_end_1h",
        "arrival_rate_event_end_1h",
        "backlog_event_end_1h",
        "delays_count_event_end_1h",
        "delay_rate_event_end_1h",
        "avg_delay_event_end_1h",
        "std_delay_event_end_1h",
        "log_delay_rate_event_end_1h",
        "in_flight_event_end_1h",

        "rel_dev_from_plan",
        "rel_dev_from_dynplan",
        "dynamic_planned_iet",
        "backlog_rel_to_planned",
        "delta_pressure_next_15m_60m",

        "subsys_starts_15m",
        "subsys_starts_60m",
        "subsys_delayed_60m",
        "subsys_delay_rate_60m",

        "veh_events_5m",
        "veh_events_60m",
        "veh_delayed_60m",
        "veh_avg_delay_60m",
        "veh_std_delay_60m",
        "veh_delay_rate_60m",
        "veh_time_since_last_s",
        "veh_route_changed",
        "veh_util_ewma_60m",

        "op_backlog_now",
        "op_arrivals_5m",
        "op_services_5m",
        "op_pressure_5m",
        "op_arrivals_15m",
        "op_services_15m",
        "op_pressure_15m",
        "op_arrivals_60m",
        "op_services_60m",
        "op_pressure_60m",
        "op_pressure_trend_15m_60m",
        "op_pressure_5m_rel_planned",
        "op_pressure_15m_rel_planned",
        "op_pressure_60m_rel_planned",
        "op_pressure_5m_per_gapmin",
        "op_pressure_15m_per_gapmin",
        "op_pressure_60m_per_gapmin",
    }

    station_window_prefixes = (
        "starts_",
        "delayed_",
        "delaysum_",
        "delay_rate_",
        "avg_delay_",
        "planned_arrivals_",
        "backlog_",
        "burstiness_cv_",
    )

    next_prefixes = (
        "arrivals_next_",
        "services_next_",
        "pressure_next_",
    )

    srcnext_prefixes = (
        "srcnext_arrivals_",
        "srcnext_services_",
        "srcnext_pressure_",
    )

    selected: List[str] = []

    for col in df.columns:
        if col in DENY or col.endswith("_z") or col.endswith("_z_arr"):
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if _is_binary_like(df[col]):
            continue

        if col in known_exact:
            selected.append(col)
            continue

        if any(col.startswith(prefix) and col.endswith("_event_end") for prefix in station_window_prefixes):
            selected.append(col)
            continue

        if col.startswith(next_prefixes):
            selected.append(col)
            continue

        if any(
            col.startswith(prefix)
            and (
                col.endswith("_rel_planned")
                or col.endswith("_per_gapmin")
                or col.endswith("_gc")
            )
            for prefix in next_prefixes
        ):
            selected.append(col)
            continue

        if col.startswith(srcnext_prefixes):
            selected.append(col)
            continue

        if "event_end" in col:
            selected.append(col)
            continue

    return sorted(set(selected))


# ============================================================
# Interaction features
# ============================================================

def add_xgb_interactions_inplace(df: pd.DataFrame) -> List[str]:
    """
    Add z-scored process features and selected interaction terms in-place.
    """
    added: List[str] = []

    curated_z_candidates = [
        "prefix_length",
        "unique_stations_so_far",
        "concept:name_repeat_flag",
        "new_cases_1h",
        "active_cases_1h",
        "roll_ie_mean",
        "roll_ie_min",
        "roll_ie_max",
        "iet_ratio",
        "hist_std_delay",
        "o_fd_mean",
        "o_fd_std",
        "d_fd_mean",
        "d_fd_std",
        "od_fd_mean",
        "od_fd_std",
        "cong_o_1h",
        "cong_d_1h",
        "case_length_prefix",
        "case_mean_iet_prefix",
        "case_planned_dur",

        "service_rate_event_end_1h",
        "arrival_rate_event_end_1h",
        "backlog_event_end_1h",
        "delays_count_event_end_1h",
        "delay_rate_event_end_1h",
        "avg_delay_event_end_1h",
        "std_delay_event_end_1h",
        "log_delay_rate_event_end_1h",
        "in_flight_event_end_1h",

        "rel_dev_from_plan",
        "rel_dev_from_dynplan",
        "dynamic_planned_iet",
        "backlog_rel_to_planned",
        "delta_pressure_next_15m_60m",

        "subsys_starts_15m",
        "subsys_starts_60m",
        "subsys_delayed_60m",
        "subsys_delay_rate_60m",

        "veh_events_5m",
        "veh_events_60m",
        "veh_delayed_60m",
        "veh_avg_delay_60m",
        "veh_std_delay_60m",
        "veh_delay_rate_60m",
        "veh_time_since_last_s",
        "veh_route_changed",
        "veh_util_ewma_60m",

        "op_backlog_now",
        "op_arrivals_5m",
        "op_services_5m",
        "op_pressure_5m",
        "op_arrivals_15m",
        "op_services_15m",
        "op_pressure_15m",
        "op_arrivals_60m",
        "op_services_60m",
        "op_pressure_60m",
        "op_pressure_trend_15m_60m",
        "op_pressure_5m_rel_planned",
        "op_pressure_15m_rel_planned",
        "op_pressure_60m_rel_planned",
        "op_pressure_5m_per_gapmin",
        "op_pressure_15m_per_gapmin",
        "op_pressure_60m_per_gapmin",
    ]

    added += add_global_z_inplace(df, curated_z_candidates)
    added += add_global_z_inplace(df, _auto_detect_congestion_cols(df))

    df_cols = set(df.columns)

    def has(*cols: str) -> bool:
        return all(col in df_cols for col in cols)

    def add(name: str, values: np.ndarray):
        if name not in df.columns and name not in DENY:
            df[name] = _safe_to_float32(values)
            added.append(name)

    queue_len_z = "queue_len_event_end_z"
    arrival_rate_z = "arrival_rate_event_end_1h_z"
    service_rate_z = "service_rate_event_end_1h_z"
    arrival_service_ratio_z = "arrival_vs_service_ratio_event_end_z"
    avg_delay_z = "avg_delay_event_end_1h_z"
    delay_rate_z = "delay_rate_event_end_1h_z"
    first_delay_count_z = "first_delays_count_event_end_1h_z"
    backlog_z = "backlog_event_end_1h_z"

    if has(arrival_rate_z, service_rate_z):
        add(
            "int_arrival_service_gap_z",
            df[arrival_rate_z].to_numpy() - df[service_rate_z].to_numpy(),
        )

    if has(arrival_service_ratio_z, queue_len_z):
        add(
            "int_arrival_service_ratio_x_queue_z",
            df[arrival_service_ratio_z].to_numpy() * df[queue_len_z].to_numpy(),
        )

    if has(first_delay_count_z, avg_delay_z):
        add(
            "int_first_delay_count_x_avg_delay_z",
            df[first_delay_count_z].to_numpy() * df[avg_delay_z].to_numpy(),
        )

    if has(backlog_z, avg_delay_z):
        add(
            "int_backlog_x_avg_delay_z",
            df[backlog_z].to_numpy() * df[avg_delay_z].to_numpy(),
        )

    if has(queue_len_z, avg_delay_z):
        add(
            "int_queue_x_avg_delay_z",
            df[queue_len_z].to_numpy() * df[avg_delay_z].to_numpy(),
        )

    if has(service_rate_z, delay_rate_z):
        add(
            "int_service_rate_x_delay_rate_z",
            df[service_rate_z].to_numpy() * df[delay_rate_z].to_numpy(),
        )

    if has(arrival_service_ratio_z, backlog_z):
        add(
            "int_arrival_service_ratio_x_backlog_z",
            df[arrival_service_ratio_z].to_numpy() * df[backlog_z].to_numpy(),
        )

    burstiness_cols = [
        col
        for col in df_cols
        if col.startswith("burstiness_cv_") and col.endswith("_event_end_z")
    ]

    for burst_col in burstiness_cols:
        window_tag = burst_col[len("burstiness_cv_"):-len("_event_end_z")]
        delay_col = f"delay_rate_{window_tag}_event_end_z"

        if has(burst_col, delay_col):
            add(
                f"int_burstiness_x_delay_rate_{window_tag}_z",
                df[burst_col].to_numpy(dtype=float) * df[delay_col].to_numpy(dtype=float),
            )

    if has("roll_ie_max", "roll_ie_min", "active_cases_1h_z"):
        eps = 1e-6
        ratio = df["roll_ie_max"].to_numpy(dtype=float) / (
            np.abs(df["roll_ie_min"].to_numpy(dtype=float)) + eps
        )
        add(
            "int_interevent_variability_x_active_cases_z",
            ratio * df["active_cases_1h_z"].to_numpy(dtype=float),
        )

    elif has("roll_ie_max_z", "roll_ie_min_z", "active_cases_1h_z"):
        ratio = df["roll_ie_max_z"].to_numpy(dtype=float) - df["roll_ie_min_z"].to_numpy(dtype=float)
        add(
            "int_interevent_variability_x_active_cases_z",
            ratio * df["active_cases_1h_z"].to_numpy(dtype=float),
        )

    next_pressure_15_z = "pressure_next_15m_z"
    next_backlog_z = "backlog_next_now_z"

    if has(next_pressure_15_z, delay_rate_z):
        add(
            "int_next_pressure15_x_delay_rate1h_z",
            df[next_pressure_15_z].to_numpy() * df[delay_rate_z].to_numpy(),
        )

    if has(next_backlog_z, avg_delay_z):
        add(
            "int_next_backlog_x_avg_delay1h_z",
            df[next_backlog_z].to_numpy() * df[avg_delay_z].to_numpy(),
        )

    op_pressure_15_z = "op_pressure_15m_z"

    if has(op_pressure_15_z, delay_rate_z):
        add(
            "int_op_pressure15_x_delay_rate1h_z",
            df[op_pressure_15_z].to_numpy() * df[delay_rate_z].to_numpy(),
        )

    drop_cols = [col for col in DENY if col in df.columns]

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    return added


# ============================================================
# Materialization
# ============================================================

def materialize_light_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight engineered features required by the XGBoost student.
    """
    if (
        ("hour_sin" not in df.columns or "hour_cos" not in df.columns)
        and "hour_of_day" in df.columns
    ):
        hours = df["hour_of_day"].astype(float)
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)

    add_xgb_interactions_inplace(df)

    return df


# Backwards-compatible alias
_materialize_light_engineering = materialize_light_engineering


# ============================================================
# X/y assembly
# ============================================================

def build_prefix_features(
    df: pd.DataFrame,
    target_col: str = "next_is_delayed_dyn",
    extra_drop: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build numeric X/y matrices for prefix-based student training or evaluation.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Missing target column '{target_col}'. "
            "Create the prediction labels before calling build_prefix_features()."
        )

    df_x = df.copy()

    bases_to_drop = set()

    for z_col in [col for col in df_x.columns if col.endswith("_z")]:
        base = z_col[:-2]

        if base in df_x.columns and base not in DENY:
            bases_to_drop.add(base)

    if bases_to_drop:
        df_x.drop(columns=list(bases_to_drop), inplace=True)

    if extra_drop:
        df_x.drop(
            columns=[col for col in extra_drop if col in df_x.columns],
            inplace=True,
            errors="ignore",
        )

    blocked_cols = {target_col}.union(DENY)

    feature_cols = [
        col
        for col in df_x.columns
        if col not in blocked_cols and pd.api.types.is_numeric_dtype(df_x[col])
    ]

    X = df_x[feature_cols].astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = df_x[target_col].fillna(0).astype(np.int8)

    return X, y, feature_cols


def build_tau_features(
    tau_df: pd.DataFrame,
    drop_extra: Optional[List[str]] = None,
    materialize: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build numeric X/y matrices for tau-anchored student training or evaluation.
    """
    df = tau_df.copy()

    if materialize:
        materialize_light_engineering(df)

    default_extra_drop = [
        "k_idx",
        "time_since_t0_s",
    ]

    if drop_extra:
        default_extra_drop += drop_extra

    return build_prefix_features(
        df,
        target_col="y_next_delay",
        extra_drop=default_extra_drop,
    )


# ============================================================
# Optional command-line interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Materialize numeric features for the XGBoost student model."
    )

    parser.add_argument("--in_csv", required=True)
    parser.add_argument("--out_x", required=True)
    parser.add_argument("--out_y", required=True)
    parser.add_argument("--target", default="next_is_delayed_dyn")
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Run lightweight feature engineering before assembling X.",
    )

    args = parser.parse_args()

    df_input = pd.read_csv(args.in_csv)

    if args.materialize:
        materialize_light_engineering(df_input)

    X, y, features = build_prefix_features(df_input, target_col=args.target)

    if args.out_x.lower().endswith(".parquet"):
        X.to_parquet(args.out_x, index=False)
    else:
        X.to_csv(args.out_x, index=False)

    y_df = y.to_frame("y")

    if args.out_y.lower().endswith(".parquet"):
        y_df.to_parquet(args.out_y, index=False)
    else:
        y_df.to_csv(args.out_y, index=False)

    print(
        f"[features_xgb] Wrote X shape={X.shape}, "
        f"y length={len(y)}, features={len(features)}"
    )
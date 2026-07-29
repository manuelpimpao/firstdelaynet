#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_train.py

Train the XGBoost student model with time-anchored teacher-feature distillation.

Workflow
--------
1. Load tau-level train/validation tables.
2. Load raw event logs for the corresponding cases.
3. Distill teacher logits and embeddings at each tau timestamp using only past events.
4. Merge teacher features into tau-level tables.
5. Build XGBoost features.
6. Train and save the student model.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features_xgb_anonymized import materialize_light_engineering, build_tau_features
from model_xgb_anonymized import train_xgb
from distill_features_anonymized import load_teacher_artifacts, distill_tau_split


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/anonymized")
ARTIFACT_DIR = Path("model_artifacts")
RESULT_DIR = Path("results")
SPLITS_DIR = Path("splits_tau")

TAU_TRAIN_PARQUET = DATA_DIR / "tau_train_anonymized.parquet"
TAU_VAL_PARQUET = DATA_DIR / "tau_val_anonymized.parquet"

RAW_TRAIN_PARQUET = DATA_DIR / "raw_train_anonymized.parquet"
RAW_VAL_PARQUET = DATA_DIR / "raw_val_anonymized.parquet"

FINAL_MODEL = RESULT_DIR / "xgb_student_tau.joblib"
PCA_PATH = ARTIFACT_DIR / "teacher_pca_tau.pkl"

TAU_COL = "tau_ts"
TARGET_COL = "y_next_delay"


# ============================================================
# Helpers
# ============================================================

def detect_tau_case_col(df: pd.DataFrame) -> str:
    if "case_seg" in df.columns:
        return "case_seg"
    if "case:concept:name" in df.columns:
        return "case:concept:name"
    raise KeyError("Tau table must contain 'case_seg' or 'case:concept:name'.")


def to_datetime_columns(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")


def compute_scale_pos_weight(n_pos: int, n_neg: int, cap=(1.0, 50.0)) -> float:
    if n_pos <= 0:
        return cap[1]
    return float(np.clip(n_neg / float(n_pos), cap[0], cap[1]))


def add_teacher_temporal_features_tau(
    df: pd.DataFrame,
    case_col: str,
    tau_col: str,
) -> pd.DataFrame:
    if "teacher_logit" not in df.columns:
        return df

    df = df.sort_values([case_col, tau_col], kind="stable").copy()
    grouped = df.groupby(case_col, sort=False)["teacher_logit"]

    df["teacher_logit_prev1"] = grouped.shift(1)
    df["teacher_logit_rm3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["teacher_logit_rstd3"] = grouped.transform(lambda s: s.rolling(3, min_periods=2).std())

    for col in ["teacher_logit_prev1", "teacher_logit_rm3", "teacher_logit_rstd3"]:
        df[col] = df[col].astype(np.float32).fillna(0.0)

    return df


def align_events_case_column(events_df: pd.DataFrame, desired_case_col: str) -> pd.DataFrame:
    events = events_df.copy()

    if desired_case_col in events.columns:
        return events

    for col in ["case_seg", "case:concept:name"]:
        if col in events.columns:
            events[col] = events[col].astype(str)

    if desired_case_col == "case:concept:name" and "case_seg" in events.columns:
        events["case:concept:name"] = events["case_seg"].str.split("||SEG=").str[0]
        return events

    if desired_case_col == "case_seg":
        if "case:concept:name" in events.columns and "segment_id" in events.columns:
            events["case_seg"] = (
                events["case:concept:name"].astype(str)
                + "||SEG="
                + events["segment_id"].astype(str)
            )
            return events

    raise KeyError(
        f"Cannot align events to '{desired_case_col}'. "
        f"Available columns: {list(events.columns)}"
    )


def strip_ids(*series: pd.Series) -> list[pd.Series]:
    return [s.astype(str).str.strip() for s in series]


def filter_tau_with_history(
    tau_df: pd.DataFrame,
    events_ctx: pd.DataFrame,
    case_col: str,
    tau_col: str,
    time_col: str = "time:timestamp",
) -> pd.DataFrame:
    tau = tau_df.copy()
    tau = tau[tau[tau_col].notna()].copy()

    events = events_ctx[[case_col, time_col]].dropna().copy()
    events[time_col] = pd.to_datetime(events[time_col], utc=True, errors="coerce")

    first_event = (
        events.groupby(case_col, sort=False)[time_col]
        .min()
        .rename("first_event_ts")
    )

    tau = tau.merge(first_event, on=case_col, how="left")
    tau = tau[tau["first_event_ts"].notna()].copy()
    tau = tau[tau[tau_col] >= tau["first_event_ts"]].copy()

    return tau.drop(columns=["first_event_ts"])


def read_events_stream(
    path: str | Path,
    keep_cases: set[str],
    case_col_name: str,
    usecols: list[str],
    num_cols: list[str],
    time_col: str,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    path = str(path)

    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path, columns=usecols if usecols else None)

        if case_col_name in df.columns:
            df[case_col_name] = df[case_col_name].astype(str).str.strip()
            if keep_cases:
                df = df[df[case_col_name].isin(keep_cases)]

        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

        return df

    chunks = []

    try:
        reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, engine="pyarrow")
    except Exception:
        reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=True)

    for chunk in reader:
        if case_col_name in chunk.columns:
            chunk[case_col_name] = chunk[case_col_name].astype(str).str.strip()
            if keep_cases:
                chunk = chunk[chunk[case_col_name].isin(keep_cases)]

        if time_col in chunk.columns:
            chunk[time_col] = pd.to_datetime(chunk[time_col], utc=True, errors="coerce")

        for col in num_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype(np.float32)

        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    return pd.concat(chunks, ignore_index=True)


def first_delay_km1_mask_aligned(
    df: pd.DataFrame,
    case_col: str,
    y_col: str,
    tau_col: str,
) -> pd.Series:
    df_sorted = df.sort_values([case_col, tau_col], kind="stable")

    y = df_sorted[y_col].fillna(0).astype(bool)

    prev_any = (
        y.groupby(df_sorted[case_col], sort=False)
        .apply(lambda s: s.shift(fill_value=False).cummax())
        .reset_index(level=0, drop=True)
    )

    mask_sorted = y & (~prev_any)

    return pd.Series(mask_sorted.values, index=df_sorted.index).reindex(df.index)


def keep_pre_first_mask_aligned(
    df: pd.DataFrame,
    case_col: str,
    y_col: str,
    tau_col: str,
) -> pd.Series:
    df_sorted = df.sort_values([case_col, tau_col], kind="stable")

    y = df_sorted[y_col].fillna(0).astype(bool)

    seen_first = (
        y.groupby(df_sorted[case_col], sort=False)
        .apply(lambda s: s.shift(fill_value=False).cummax())
        .reset_index(level=0, drop=True)
    )

    km1_sorted = first_delay_km1_mask_aligned(
        df_sorted,
        case_col,
        y_col,
        tau_col,
    ).reindex(df_sorted.index)

    keep_sorted = (~seen_first) | km1_sorted

    return pd.Series(keep_sorted.values, index=df_sorted.index).reindex(df.index)


def apply_mask_to_df(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return df.loc[mask.fillna(False).values].reset_index(drop=True)


def apply_mask_to_arr(array: np.ndarray, mask: pd.Series) -> np.ndarray:
    return array[mask.fillna(False).values]


# ============================================================
# Main training routine
# ============================================================

def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    tau_train = pd.read_parquet(TAU_TRAIN_PARQUET)
    tau_val = pd.read_parquet(TAU_VAL_PARQUET)

    case_col = detect_tau_case_col(tau_train)

    if detect_tau_case_col(tau_val) != case_col:
        raise ValueError("Train and validation tau tables use different case columns.")

    to_datetime_columns(tau_train, [TAU_COL])
    to_datetime_columns(tau_val, [TAU_COL])

    teacher_model, teacher_vocab, _, dataset_cfg, _ = load_teacher_artifacts(ARTIFACT_DIR)

    cat_col = dataset_cfg.get("cat_col", "concept:name")
    num_cols = list(dataset_cfg.get("num_cols", []))
    time_col = dataset_cfg.get("time_col", "time:timestamp")

    base_usecols = {case_col, "case_seg", "case:concept:name", time_col, cat_col}
    usecols = list(base_usecols.union(num_cols))

    tau_cases_train = set(tau_train[case_col].astype(str).str.strip().unique())
    tau_cases_val = set(tau_val[case_col].astype(str).str.strip().unique())

    print("[events] Reading train event context.")
    events_train = read_events_stream(
        RAW_TRAIN_PARQUET,
        tau_cases_train,
        case_col,
        usecols=usecols,
        num_cols=num_cols,
        time_col=time_col,
    )
    print(f"[events] Train context rows kept: {len(events_train):,}")

    print("[events] Reading validation event context.")
    events_val = read_events_stream(
        RAW_VAL_PARQUET,
        tau_cases_val,
        case_col,
        usecols=usecols,
        num_cols=num_cols,
        time_col=time_col,
    )
    print(f"[events] Validation context rows kept: {len(events_val):,}")

    to_datetime_columns(events_train, ["planned:timestamp"])
    to_datetime_columns(events_val, ["planned:timestamp"])

    materialize_light_engineering(events_train)
    materialize_light_engineering(events_val)

    events_all = pd.concat([events_train, events_val], ignore_index=True)
    events_all = align_events_case_column(events_all, case_col)

    tau_train[case_col], tau_val[case_col] = strip_ids(
        tau_train[case_col],
        tau_val[case_col],
    )

    events_all[case_col] = events_all[case_col].astype(str).str.strip()

    cases_train = set(tau_train[case_col].unique())
    cases_val = set(tau_val[case_col].unique())

    events_ctx_train = events_all[events_all[case_col].isin(cases_train)].copy()
    events_ctx_val = events_all[events_all[case_col].isin(cases_val)].copy()

    dataset_cfg = dict(dataset_cfg)
    dataset_cfg["case_col"] = case_col

    tau_train = filter_tau_with_history(
        tau_train,
        events_ctx_train,
        case_col,
        TAU_COL,
        time_col=time_col,
    )

    tau_val = filter_tau_with_history(
        tau_val,
        events_ctx_val,
        case_col,
        TAU_COL,
        time_col=time_col,
    )

    tau_train_feat = materialize_light_engineering(tau_train.copy())
    tau_val_feat = materialize_light_engineering(tau_val.copy())

    X_train_base, _, _ = build_tau_features(tau_train_feat)
    X_val_base, _, _ = build_tau_features(tau_val_feat)

    mask_train_first = first_delay_km1_mask_aligned(
        tau_train_feat,
        case_col,
        TARGET_COL,
        TAU_COL,
    )

    mask_val_first = first_delay_km1_mask_aligned(
        tau_val_feat,
        case_col,
        TARGET_COL,
        TAU_COL,
    )

    keep_train = keep_pre_first_mask_aligned(
        tau_train_feat,
        case_col,
        TARGET_COL,
        TAU_COL,
    )

    keep_val = keep_pre_first_mask_aligned(
        tau_val_feat,
        case_col,
        TARGET_COL,
        TAU_COL,
    )

    y_train = apply_mask_to_arr(mask_train_first.astype(np.uint8).to_numpy(), keep_train)
    y_val = apply_mask_to_arr(mask_val_first.astype(np.uint8).to_numpy(), keep_val)

    print("[teacher] Distilling teacher features at tau snapshots.")

    tau_train = tau_train.reset_index(drop=True).copy()
    tau_val = tau_val.reset_index(drop=True).copy()

    tau_train["__row_id__"] = np.arange(len(tau_train), dtype=np.int64)
    tau_val["__row_id__"] = np.arange(len(tau_val), dtype=np.int64)

    tau_train_key = tau_train[[case_col, TAU_COL, "__row_id__"]].copy()
    tau_val_key = tau_val[[case_col, TAU_COL, "__row_id__"]].copy()

    dist_train, pca = distill_tau_split(
        tau_df=tau_train_key,
        events_df=events_ctx_train,
        model=teacher_model,
        vocab=teacher_vocab,
        dataset_cfg=dataset_cfg,
        case_id_col=case_col,
        tau_col=TAU_COL,
        pca_components=48,
        pca=None,
    )

    dist_val, _ = distill_tau_split(
        tau_df=tau_val_key,
        events_df=events_ctx_val,
        model=teacher_model,
        vocab=teacher_vocab,
        dataset_cfg=dataset_cfg,
        case_id_col=case_col,
        tau_col=TAU_COL,
        pca_components=48,
        pca=pca,
    )

    joblib.dump(pca, PCA_PATH)
    print(f"Saved teacher PCA to: {PCA_PATH}")

    teacher_cols_basic = [
        "teacher_logit",
        *[col for col in dist_train.columns if str(col).startswith("teacher_embed_pca_")],
    ]

    for col in teacher_cols_basic:
        if col not in dist_train.columns:
            dist_train[col] = np.float32(0.0)
        if col not in dist_val.columns:
            dist_val[col] = np.float32(0.0)

    tau_train_attached = tau_train.merge(
        dist_train[["__row_id__", *teacher_cols_basic]],
        on="__row_id__",
        how="left",
    )

    tau_val_attached = tau_val.merge(
        dist_val[["__row_id__", *teacher_cols_basic]],
        on="__row_id__",
        how="left",
    )

    for col in teacher_cols_basic:
        tau_train_attached[col] = tau_train_attached[col].astype(np.float32).fillna(0.0)
        tau_val_attached[col] = tau_val_attached[col].astype(np.float32).fillna(0.0)

    tau_train_aug = add_teacher_temporal_features_tau(
        tau_train_attached,
        case_col,
        TAU_COL,
    )

    tau_val_aug = add_teacher_temporal_features_tau(
        tau_val_attached,
        case_col,
        TAU_COL,
    )

    X_train_base = apply_mask_to_df(X_train_base, keep_train)
    X_val_base = apply_mask_to_df(X_val_base, keep_val)

    tau_train_aug = apply_mask_to_df(tau_train_aug, keep_train)
    tau_val_aug = apply_mask_to_df(tau_val_aug, keep_val)

    tau_train_aug = tau_train_aug.drop(columns=["__row_id__"], errors="ignore")
    tau_val_aug = tau_val_aug.drop(columns=["__row_id__"], errors="ignore")

    teacher_cols = teacher_cols_basic + [
        "teacher_logit_prev1",
        "teacher_logit_rm3",
        "teacher_logit_rstd3",
    ]

    for col in teacher_cols:
        if col in tau_train_aug.columns:
            tau_train_aug[col] = tau_train_aug[col].astype(np.float32)
        if col in tau_val_aug.columns:
            tau_val_aug[col] = tau_val_aug[col].astype(np.float32)

    X_train = pd.concat(
        [
            X_train_base.reset_index(drop=True),
            tau_train_aug[teacher_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    X_val = pd.concat(
        [
            X_val_base.reset_index(drop=True),
            tau_val_aug[teacher_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    assert len(X_train) == len(y_train) == len(tau_train_aug), "Training rows misaligned."
    assert len(X_val) == len(y_val) == len(tau_val_aug), "Validation rows misaligned."

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = compute_scale_pos_weight(n_pos, n_neg, cap=(1.0, 50.0))

    print(
        f"[imbalance] train positives={n_pos:,}, "
        f"negatives={n_neg:,}, scale_pos_weight={scale_pos_weight:.2f}"
    )

    mask_train_first_final = first_delay_km1_mask_aligned(
        tau_train_aug,
        case_col,
        TARGET_COL,
        TAU_COL,
    )

    mask_val_first_final = first_delay_km1_mask_aligned(
        tau_val_aug,
        case_col,
        TARGET_COL,
        TAU_COL,
    )

    def build_sample_weights(y: np.ndarray, boost_mask: pd.Series, boost: float) -> np.ndarray:
        weights = np.ones_like(y, dtype=float)
        bm = boost_mask.fillna(False).values

        if bm.shape[0] != y.shape[0]:
            bm = bm[: y.shape[0]]

        weights[bm] *= float(boost)
        return weights

    sample_weight_train = build_sample_weights(
        y_train,
        mask_train_first_final,
        boost=8.0,
    )

    sample_weight_val = build_sample_weights(
        y_val,
        mask_val_first_final,
        boost=1.0,
    )

    print(f"Final tau-train rows: {X_train.shape[0]:,}")
    print(f"Final tau-validation rows: {X_val.shape[0]:,}")

    xgb_params = {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "colsample_bynode": 0.8,
        "gamma": 0.0,
        "reg_alpha": 8.0,
        "reg_lambda": 40.0,
        "scale_pos_weight": scale_pos_weight,
        "max_delta_step": 2,
        "tree_method": "hist",
    }

    print(f"\nTraining XGBoost student model -> {FINAL_MODEL}")

    train_xgb(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_path=FINAL_MODEL,
        sample_weight_train=sample_weight_train,
        sample_weight_val=sample_weight_val,
        first_delay_mask_val=mask_val_first_final,
        first_delay_boost=8.0,
        early_stopping_rounds=100,
        out_dir=RESULT_DIR,
        **xgb_params,
    )

    threshold_path = RESULT_DIR / "val_operating_threshold.txt"

    if threshold_path.exists():
        try:
            threshold = float(threshold_path.read_text(encoding="utf-8").strip())
            print(f"\nValidation operating threshold: {threshold:.3f}")
            print(f"Saved at: {threshold_path}")
        except Exception:
            print(f"\nValidation operating threshold saved at: {threshold_path}")

    print(f"\nDone. Final model saved to: {FINAL_MODEL}")
    print(f"Teacher features used: {teacher_cols}")


if __name__ == "__main__":
    main()
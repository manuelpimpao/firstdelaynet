#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_tau_and_first_delay.py — evaluate tau-trained XGBoost model on the test split

EXTENDED (Phase 3A/3B/3C production-realism metrics + Brier score)
------------------------------------------------------------------
Original metrics:
- Overall tau metrics on y_next_delay (row-level)
- k-1@tau (mask-derived; NOTE: trivial if you evaluate only on m_te_first rows)
- Pre-first window metrics (delayed cases only) at operating threshold

Additional metrics:

Brier score (calibration quality) on ALL tau rows:
- brier_all_tau
- baseline_brier (pos_rate*(1-pos_rate))
- brier_improvement_vs_baseline

3A) Non-delayed case false-alarm burden @ operating threshold
3B) Case-level alarm precision/recall (production-friendly)
3C) Early-warning timing / lead-time distribution (delayed cases only)

Config C evaluation: Config C evaluation (one row per case)
-------------------------------------------------------
Config C: Decision-Point per Case (C-Kminus1PerCase)
- Exactly one row per case is evaluated.
- If delayed case: evaluate the tau row at k-1 (the first-delay-at-tau row, is_first_delay_tau==1).
- If non-delayed case: evaluate the last available tau row.
- Unit of analysis: one row per case.
- Metrics: Precision, Recall (for delayed cases), F1 at operating threshold (thr_op).

All metrics are PRINTED clearly and SAVED as CSVs in out_dir.
Also saves:
- tau_rows_with_alerts.parquet
- case_level_alert_summary.csv
- test_tau_predictions.parquet
- test_tau_metrics.json
- existing plots + lead_time_histogram_minutes.pdf
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score, confusion_matrix, brier_score_loss
)

# ── Student feature API (tau) ──────────────────────────────────────────────────
from features_xgb_anonymized import materialize_light_engineering, build_tau_features

# ─── Defaults (you can override via CLI) ─────────────────────────────────────
DEF_TAU_TEST_PARQUET = "data/anonymized/tau_test_anonymized.parquet"
DEF_RAW_TEST_PARQUET = "data/anonymized/raw_test_anonymized.parquet"

DEF_RESULT_DIR = "results"
DEF_MODEL_PATH = os.path.join(DEF_RESULT_DIR, "xgb_student_tau.joblib")

DEF_ART_DIR = "model_artifacts"
DEF_PCA_PATH = os.path.join(DEF_ART_DIR, "teacher_pca_tau.pkl")

DEF_OUT_DIR = "results/test_tau"

DEFAULT_OPERATING_THR = 0.5
TAU_COL    = "tau_ts"
TARGET_COL = "y_next_delay"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def _savefig_pdf(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()

def detect_tau_case_col(df: pd.DataFrame) -> str:
    if "case_seg" in df.columns:
        return "case_seg"
    if "case:concept:name" in df.columns:
        return "case:concept:name"
    raise KeyError("tau table must contain 'case_seg' or 'case:concept:name'.")

def to_dt(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

def _read_events_stream(path: str,
                        keep_cases: set[str],
                        case_col_name: str,
                        usecols: list[str],
                        num_cols: list[str],
                        time_col: str,
                        chunksize: int = 1_000_000) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path, columns=usecols if usecols else None)
        if case_col_name in df.columns:
            df[case_col_name] = df[case_col_name].astype(str).str.strip()
            if keep_cases:
                df = df[df[case_col_name].isin(keep_cases)]
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        return df

    chunks = []
    try:
        reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, engine="pyarrow")
    except Exception:
        reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=True)

    for ch in reader:
        if case_col_name in ch.columns:
            ch[case_col_name] = ch[case_col_name].astype(str).str.strip()
            if keep_cases:
                ch = ch[ch[case_col_name].isin(keep_cases)]
        if time_col in ch.columns:
            ch[time_col] = pd.to_datetime(ch[time_col], utc=True, errors="coerce")
        for c in num_cols:
            if c in ch.columns:
                ch[c] = pd.to_numeric(ch[c], errors="coerce").astype(np.float32)
        chunks.append(ch)

    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)

def align_events_case_column(events_df: pd.DataFrame, desired_case_col: str) -> pd.DataFrame:
    ev = events_df.copy()
    if desired_case_col in ev.columns:
        return ev
    for c in ("case_seg", "case:concept:name"):
        if c in ev.columns:
            ev[c] = ev[c].astype(str)
    if desired_case_col == "case:concept:name" and "case_seg" in ev.columns:
        ev["case:concept:name"] = ev["case_seg"].str.split("||SEG=").str[0]
        return ev
    if desired_case_col == "case_seg":
        if "case:concept:name" in ev.columns and "segment_id" in ev.columns:
            ev["case_seg"] = ev["case:concept:name"].astype(str) + "||SEG=" + ev["segment_id"].astype(str)
            return ev
        if "case:concept:name" in ev.columns and "k_idx" in ev.columns:
            ev["case_seg"] = ev["case:concept:name"].astype(str) + "||SEG=" + ev["k_idx"].astype(str)
            return ev
    raise KeyError(f"Cannot align events to '{desired_case_col}'. Available columns: {list(ev.columns)}")

def add_teacher_temporal_features_tau(df: pd.DataFrame, case_col: str, tau_col: str) -> pd.DataFrame:
    if "teacher_logit" not in df.columns:
        return df
    df = df.sort_values([case_col, tau_col], kind="stable").copy()
    g  = df.groupby(case_col, sort=False)["teacher_logit"]
    df["teacher_logit_prev1"] = g.shift(1)
    df["teacher_logit_rm3"]   = g.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["teacher_logit_rstd3"] = g.transform(lambda s: s.rolling(3, min_periods=2).std())
    for c in ("teacher_logit_prev1","teacher_logit_rm3","teacher_logit_rstd3"):
        df[c] = df[c].astype(np.float32).fillna(0.0)
    return df

def eval_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    out = {}
    y = np.asarray(y_true, dtype=np.uint8)
    p = np.asarray(probs, dtype=float)
    out["n"] = int(y.size)
    out["n_pos"] = int(y.sum())
    out["n_neg"] = int(y.size - y.sum())
    out["pos_rate"] = float(out["n_pos"] / max(1, out["n"]))
    try:
        out["auprc"] = float(average_precision_score(y, p))
    except Exception:
        out["auprc"] = float("nan")
    try:
        out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    except Exception:
        out["auroc"] = float("nan")

    prec, rec, thr = precision_recall_curve(y, p)
    thr_full = np.append(thr, 1.0)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best_idx = int(np.nanargmax(f1s))
    out["best_thr"] = float(thr_full[best_idx])
    yhat = (p >= out["best_thr"]).astype(np.uint8)
    out["f1"] = float(f1_score(y, yhat, zero_division=0))
    out["precision"] = float(precision_score(y, yhat, zero_division=0))
    out["recall"] = float(recall_score(y, yhat, zero_division=0))
    return out

def plot_roc(labels, probs, out_pdf: Path):
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=1.5)
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    _savefig_pdf(out_pdf)

def plot_pr(labels, probs, out_pdf: Path):
    prec, rec, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(rec, prec, linewidth=1.5)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall")
    _savefig_pdf(out_pdf)

def plot_calibration(labels, probs, out_pdf: Path, n_bins=20):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(probs, bins) - 1
    accs, confs = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        accs.append(labels[m].mean()); confs.append(probs[m].mean())
    if not confs:
        return
    bs = brier_score_loss(labels, probs)
    plt.figure()
    plt.plot(confs, accs, marker="o", linewidth=1.5)
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Empirical positive rate")
    plt.title(f"Calibration (Brier={bs:.4f})")
    _savefig_pdf(out_pdf)

def plot_histograms(labels, probs, out_pdf: Path):
    plt.figure()
    plt.hist(probs[labels==0], bins=50, alpha=0.6, label="neg")
    plt.hist(probs[labels==1], bins=50, alpha=0.6, label="pos")
    plt.xlabel("Predicted probability"); plt.ylabel("Count"); plt.title("Score Distributions"); plt.legend()
    _savefig_pdf(out_pdf)

def plot_confusion(labels, probs, thr, out_pdf: Path):
    yhat = (probs >= thr).astype(np.uint8)
    cm = confusion_matrix(labels, yhat, labels=[0,1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.title(f"Confusion @ {thr:.3f}"); plt.xlabel("Predicted"); plt.ylabel("True")
    _savefig_pdf(out_pdf)

def load_operating_threshold(result_dir: str, default_thr: float) -> float:
    path = os.path.join(result_dir, "val_operating_threshold.txt")
    try:
        if os.path.exists(path):
            txt = open(path, "r", encoding="utf-8").read().strip()
            v = float(txt)
            if 0.0 <= v <= 1.0:
                print(f"[operating] Loaded validation operating threshold from: {path} (thr={v:.3f})")
                return v
            else:
                print(f"[operating] Ignoring out-of-range threshold in {path}: {txt}")
    except Exception as e:
        print(f"[operating] Could not read {path}: {e}")
    print(f"[operating] Using default threshold: {default_thr:.3f}")
    return float(default_thr)

# ── Aligned first-delay (k-1) mask ───────────────────────────────────────────
def first_delay_mask_tau_aligned(df: pd.DataFrame, case_col: str, y_col: str, tau_col: str) -> np.ndarray:
    df_sorted = df.sort_values([case_col, tau_col], kind="stable")
    y = df_sorted[y_col].fillna(0).astype(bool)
    prev_any = y.groupby(df_sorted[case_col], sort=False).apply(
        lambda s: s.shift(fill_value=False).cummax()
    ).reset_index(level=0, drop=True)
    km1_sorted = (y & (~prev_any))
    return pd.Series(km1_sorted.values, index=df_sorted.index).reindex(df.index).fillna(False).to_numpy(dtype=bool)

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Phase 3A/3B/3C helper computations
# ─────────────────────────────────────────────────────────────────────────────

def _quantiles(series: pd.Series, qs=(0.1, 0.5, 0.9)) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {f"q{int(q*100)}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(q*100)}"] = float(s.quantile(q))
    return out

def compute_case_level_tables(tau_df: pd.DataFrame,
                              case_col: str,
                              tau_col: str,
                              probs: np.ndarray,
                              thr_op: float,
                              m_te_first: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      row_df: per-tau-row table (case_col, tau_col, proba, alert, is_first_delay_tau)
      case_df: per-case aggregates (case_delayed, case_alerted, n_tau_rows, n_alerts, alert_rate)
    """
    row_df = tau_df[[case_col, tau_col]].copy()
    row_df["proba"] = probs.astype(np.float32)
    row_df["alert"] = (row_df["proba"].to_numpy() >= float(thr_op)).astype(np.uint8)
    row_df["is_first_delay_tau"] = m_te_first.astype(np.uint8)

    case_has_delay = row_df.groupby(case_col, sort=False)["is_first_delay_tau"].max().rename("case_delayed")
    case_n_rows    = row_df.groupby(case_col, sort=False).size().rename("n_tau_rows")
    case_n_alerts  = row_df.groupby(case_col, sort=False)["alert"].sum().rename("n_alerts")
    case_ever      = (case_n_alerts > 0).astype(np.uint8).rename("case_alerted")
    case_alert_rate = (case_n_alerts / case_n_rows.replace(0, np.nan)).rename("alert_rate")

    case_df = pd.concat([case_has_delay, case_ever, case_n_rows, case_n_alerts, case_alert_rate], axis=1).reset_index()
    return row_df, case_df

def compute_non_delayed_alarm_burden(case_df: pd.DataFrame) -> dict:
    nd = case_df[case_df["case_delayed"] == 0].copy()
    out = {
        "n_cases_non_delayed": int(len(nd)),
        "n_cases_non_delayed_alerted": int((nd["case_alerted"] == 1).sum()),
    }
    out["share_non_delayed_cases_alerted"] = float(out["n_cases_non_delayed_alerted"] / max(1, out["n_cases_non_delayed"]))
    out["mean_alerts_per_non_delayed_case"] = float(nd["n_alerts"].mean()) if len(nd) else float("nan")
    out["median_alerts_per_non_delayed_case"] = float(nd["n_alerts"].median()) if len(nd) else float("nan")
    out["alerts_per_100_non_delayed_cases"] = float(100.0 * nd["n_alerts"].sum() / max(1, len(nd))) if len(nd) else float("nan")
    out["mean_alert_rate_non_delayed"] = float(nd["alert_rate"].mean()) if len(nd) else float("nan")
    out.update({f"alert_rate_{k}": v for k, v in _quantiles(nd["alert_rate"]).items()})
    out.update({f"n_alerts_{k}": v for k, v in _quantiles(nd["n_alerts"]).items()})
    return out

def compute_case_level_precision_recall(case_df: pd.DataFrame) -> dict:
    y = case_df["case_delayed"].to_numpy(dtype=np.uint8)
    yhat = case_df["case_alerted"].to_numpy(dtype=np.uint8)
    TP = int(((yhat == 1) & (y == 1)).sum())
    FP = int(((yhat == 1) & (y == 0)).sum())
    FN = int(((yhat == 0) & (y == 1)).sum())
    TN = int(((yhat == 0) & (y == 0)).sum())
    prec = float(TP / max(1, TP + FP))
    rec  = float(TP / max(1, TP + FN))
    f1   = 0.0 if (prec + rec) == 0 else float(2*prec*rec/(prec+rec))
    out = {
        "n_cases_total": int(len(y)),
        "case_delay_rate": float(y.mean()) if len(y) else float("nan"),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": prec, "recall": rec, "f1": f1,
        "alert_rate_cases": float(yhat.mean()) if len(yhat) else float("nan"),
    }
    return out

def compute_lead_time_stats(row_df: pd.DataFrame, case_col: str, tau_col: str) -> tuple[dict, pd.DataFrame]:
    """
    Uses per-tau-row table with columns:
      - alert (0/1)
      - is_first_delay_tau (0/1)
    For each delayed case:
      - kminus1_tau: tau timestamp where is_first_delay_tau==1 (min)
      - first_alert_tau: min tau timestamp where alert==1 (if any)
      - lead_time = kminus1_tau - first_alert_tau (seconds), only if first_alert_tau <= kminus1_tau
    """
    df = row_df.copy()
    df[tau_col] = pd.to_datetime(df[tau_col], utc=True, errors="coerce")

    delayed_cases = df.groupby(case_col, sort=False)["is_first_delay_tau"].max()
    delayed_cases = delayed_cases[delayed_cases == 1].index
    sub = df[df[case_col].isin(delayed_cases)].copy()

    kminus1_tau = (
        sub[sub["is_first_delay_tau"] == 1]
        .groupby(case_col, sort=False)[tau_col]
        .min()
        .rename("kminus1_tau")
    )
    first_alert_tau = (
        sub[sub["alert"] == 1]
        .groupby(case_col, sort=False)[tau_col]
        .min()
        .rename("first_alert_tau")
    )

    per_case = pd.concat([kminus1_tau, first_alert_tau], axis=1).reset_index()
    per_case["has_alert"] = per_case["first_alert_tau"].notna().astype(np.uint8)

    per_case["on_time_or_early"] = (
        per_case["has_alert"].eq(1) &
        (per_case["first_alert_tau"] <= per_case["kminus1_tau"])
    ).astype(np.uint8)

    per_case["exact_at_kminus1"] = (
        per_case["has_alert"].eq(1) &
        (per_case["first_alert_tau"] == per_case["kminus1_tau"])
    ).astype(np.uint8)

    per_case["strictly_early"] = (
        per_case["has_alert"].eq(1) &
        (per_case["first_alert_tau"] < per_case["kminus1_tau"])
    ).astype(np.uint8)

    per_case["lead_time_sec"] = np.nan
    m_early = per_case["strictly_early"].eq(1)
    per_case.loc[m_early, "lead_time_sec"] = (
        (per_case.loc[m_early, "kminus1_tau"] - per_case.loc[m_early, "first_alert_tau"])
        .dt.total_seconds()
    )

    out = {
        "n_delayed_cases": int(len(per_case)),
        "share_delayed_cases_alerted_any": float(per_case["has_alert"].mean()) if len(per_case) else float("nan"),
        "share_alerted_on_time_or_early": float(per_case["on_time_or_early"].mean()) if len(per_case) else float("nan"),
        "share_alerted_exactly_at_kminus1": float(per_case["exact_at_kminus1"].mean()) if len(per_case) else float("nan"),
        "share_alerted_strictly_early": float(per_case["strictly_early"].mean()) if len(per_case) else float("nan"),
        "share_missed_no_alert": float((per_case["has_alert"] == 0).mean()) if len(per_case) else float("nan"),
        "n_strictly_early": int(per_case["strictly_early"].sum()),
    }

    lt = pd.to_numeric(per_case.loc[m_early, "lead_time_sec"], errors="coerce").dropna()
    out["lead_time_mean_min"] = float(lt.mean()/60.0) if len(lt) else float("nan")
    out["lead_time_median_min"] = float(lt.median()/60.0) if len(lt) else float("nan")
    if len(lt):
        out["lead_time_p10_min"] = float(lt.quantile(0.10)/60.0)
        out["lead_time_p90_min"] = float(lt.quantile(0.90)/60.0)
    else:
        out["lead_time_p10_min"] = float("nan")
        out["lead_time_p90_min"] = float("nan")

    return out, per_case

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Config C (one row per case) evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_config_c_kminus1_per_case(tau_df: pd.DataFrame,
                                      case_col: str,
                                      tau_col: str,
                                      probs: np.ndarray,
                                      thr_op: float,
                                      m_te_first: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """
    Config C: exactly one row per case.
      - delayed cases: choose the tau row where is_first_delay_tau==1 (k-1 at tau). If multiple, take earliest.
      - non-delayed cases: choose the last available tau row.
    Labels:
      - delayed case row => y=1
      - non-delayed case row => y=0
    Metrics computed at operating threshold.
    Returns:
      metrics_dict, per_case_selected_df
    """
    df = tau_df[[case_col, tau_col]].copy()
    df[tau_col] = pd.to_datetime(df[tau_col], utc=True, errors="coerce")
    df["proba"] = probs.astype(np.float32)
    df["is_first_delay_tau"] = m_te_first.astype(np.uint8)

    # Identify delayed cases (have at least one first-delay-at-tau row)
    case_delayed = df.groupby(case_col, sort=False)["is_first_delay_tau"].max()
    delayed_cases = case_delayed[case_delayed == 1].index
    nondelayed_cases = case_delayed[case_delayed == 0].index

    # For delayed: pick earliest tau where is_first_delay_tau==1
    df_del = df[df["is_first_delay_tau"] == 1]
    idx_del = df_del.groupby(case_col, sort=False)[tau_col].idxmin()

    # For non-delayed: pick last tau overall
    df_non = df[df[case_col].isin(nondelayed_cases)]
    idx_non = df_non.groupby(case_col, sort=False)[tau_col].idxmax()

    idx_all = pd.Index(idx_del.tolist() + idx_non.tolist())

    sel = df.loc[idx_all].copy()
    sel["case_delayed"] = sel[case_col].isin(delayed_cases).astype(np.uint8)
    sel["y_true"] = sel["case_delayed"]  # by construction
    sel["alert_op"] = (sel["proba"].to_numpy() >= float(thr_op)).astype(np.uint8)

    y = sel["y_true"].to_numpy(dtype=np.uint8)
    yhat = sel["alert_op"].to_numpy(dtype=np.uint8)

    TP = int(((yhat == 1) & (y == 1)).sum())
    FP = int(((yhat == 1) & (y == 0)).sum())
    FN = int(((yhat == 0) & (y == 1)).sum())
    TN = int(((yhat == 0) & (y == 0)).sum())

    prec = float(TP / max(1, TP + FP))
    rec  = float(TP / max(1, TP + FN))  # this is Recall_{k-1} per case
    f1   = 0.0 if (prec + rec) == 0 else float(2 * prec * rec / (prec + rec))

    metrics = {
        "config": "C-Kminus1PerCase",
        "operating_threshold": float(thr_op),
        "n_cases_total": int(len(sel)),
        "n_cases_delayed": int(y.sum()),
        "n_cases_non_delayed": int(len(sel) - y.sum()),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": prec,
        "recall_kminus1_per_case": rec,
        "f1": f1,
        "alert_rate_cases": float(yhat.mean()) if len(yhat) else float("nan"),
    }
    sel = sel.sort_values(case_col, kind="stable").reset_index(drop=True)
    return metrics, sel

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-test", default=DEF_TAU_TEST_PARQUET)
    parser.add_argument("--raw-test", default=DEF_RAW_TEST_PARQUET)
    parser.add_argument("--result-dir", default=DEF_RESULT_DIR)
    parser.add_argument("--model", default=DEF_MODEL_PATH)
    parser.add_argument("--out-dir", default=DEF_OUT_DIR)

    # Teacher artifacts (ignored if --student-only)
    parser.add_argument("--art-dir", default=DEF_ART_DIR)
    parser.add_argument("--pca-path", default=DEF_PCA_PATH)

    parser.add_argument("--op-thr", type=float, default=None,
                        help="Operating threshold override (0..1). If unset, tries <result-dir>/val_operating_threshold.txt, else default.")
    parser.add_argument("--student-only", action="store_true",
                        help="Evaluate student-only features (skip teacher artifacts entirely).")
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    plots_dir = Path(args.out_dir) / "plots_test_tau"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[info] Run timestamp: {ts}")
    print(f"[info] Mode: {'STUDENT-ONLY' if args.student_only else 'TEACHER+STUDENT'}")

    # 0) Load TEST tau table
    print("[test] Loading TEST tau Parquet…")
    tau_te = pd.read_parquet(args.tau_test)
    case_col = detect_tau_case_col(tau_te)
    to_dt(tau_te, [TAU_COL])
    
    # Canonical row id for stable snapshot identity
    tau_te = tau_te.reset_index(drop=True).copy()
    tau_te["__row_id__"] = np.arange(len(tau_te), dtype=np.int64)

    # 1) Build event column requirements
    if not args.student_only:
        from distill_features_anonymized import load_teacher_artifacts, distill_tau_split
        teacher_model, teacher_vocab, _model_cfg, dataset_cfg, _art_dir = load_teacher_artifacts(args.art_dir)
        cat_col  = dataset_cfg.get("cat_col", "concept:name")
        num_cols = list(dataset_cfg.get("num_cols", []))
        time_col = dataset_cfg.get("time_col", "time:timestamp")
    else:
        cat_col  = "concept:name"
        num_cols = []
        time_col = "time:timestamp"

    base_usecols = {case_col, "case_seg", "case:concept:name", time_col, cat_col}
    usecols = list(base_usecols.union(num_cols))

    tau_cases_te = set(tau_te[case_col].astype(str).str.strip().unique())

    # 2) Stream & filter FULL TEST events
    print("[events] streaming test events…")
    df_te_full = _read_events_stream(
        args.raw_test, tau_cases_te, case_col,
        usecols=usecols, num_cols=num_cols, time_col=time_col
    )
    print(f"[events] test context rows kept: {len(df_te_full):,}")

    to_dt(df_te_full, ["planned:timestamp"])
    materialize_light_engineering(df_te_full)  # harmless if student-only

    # 3) Align event case column to tau
    df_te_full = align_events_case_column(df_te_full, case_col)
    tau_te[case_col] = tau_te[case_col].astype(str).str.strip()
    df_te_full[case_col] = df_te_full[case_col].astype(str).str.strip()

    # 4) Build tau base features
    tau_te_feat = materialize_light_engineering(tau_te.copy())
    X_te_base, y_te, _ = build_tau_features(tau_te_feat)

    # 5) Optionally distill teacher at tau (Teacher+Student mode)
    teacher_cols = []
    if not args.student_only:
        print("[teacher] Distilling at tau …")
        if os.path.exists(args.pca_path):
            pca = joblib.load(args.pca_path)
            print(f"Loaded tau-PCA from {args.pca_path}")
        else:
            raise FileNotFoundError(
                f"PCA not found at {args.pca_path}. The model expects the fitted tau PCA."
            )
    
        # Use the public distillation module
        from distill_features_anonymized import distill_tau_split
    
        dataset_cfg = dict(dataset_cfg)
        dataset_cfg["case_col"] = case_col
    
        # --- Ensure tau has a canonical __row_id__ (created right after loading tau_te) ---
        if "__row_id__" not in tau_te.columns:
            raise RuntimeError("tau_te is missing __row_id__. Create it right after loading tau_te.")
    
        # 1) Distill using only identity columns + __row_id__
        tau_key = tau_te[[case_col, TAU_COL, "__row_id__"]].copy()
    
        dist_te, _ = distill_tau_split(
            tau_df=tau_key,           # includes __row_id__
            events_df=df_te_full,     # ok (distiller sorts internally)
            model=teacher_model,
            vocab=teacher_vocab,
            dataset_cfg=dataset_cfg,
            case_id_col=case_col,
            tau_col=TAU_COL,
            pca_components=48,
            pca=pca,
        )
    
        # (Optional) sanity prints once
        print("[diag] tau_key rows:", len(tau_key))
        print("[diag] dist_te rows:", len(dist_te))
        if "__row_id__" in dist_te.columns:
            print("[diag] dist_te __row_id__ unique:", dist_te["__row_id__"].nunique())
    
        # 2) Ensure tau_te_feat carries __row_id__ (feature engineering may drop IDs)
        if "__row_id__" not in tau_te_feat.columns:
            tau_te_feat = tau_te_feat.reset_index(drop=True).copy()
            tau_te_feat["__row_id__"] = tau_te["__row_id__"].values
    
        # 3) 1:1 merge teacher outputs on __row_id__
        tau_te_m = tau_te_feat.merge(dist_te, on="__row_id__", how="left")
        if len(tau_te_m) != len(tau_te_feat):
            raise RuntimeError(
                f"Row explosion detected after __row_id__ merge: {len(tau_te_m)} vs {len(tau_te_feat)}"
            )
    
        # 4) Re-attach identity columns needed for temporal features (always safe)
        #    This fixes your KeyError: 'case_seg' if feature pipeline dropped it.
        id_map = tau_key[["__row_id__", case_col, TAU_COL]]
        tau_te_m = tau_te_m.merge(id_map, on="__row_id__", how="left", suffixes=("", "_id"))
        # If there were clashes, prefer the _id version
        if f"{case_col}_id" in tau_te_m.columns:
            tau_te_m[case_col] = tau_te_m[f"{case_col}_id"]
            tau_te_m = tau_te_m.drop(columns=[f"{case_col}_id"])
        if f"{TAU_COL}_id" in tau_te_m.columns:
            tau_te_m[TAU_COL] = tau_te_m[f"{TAU_COL}_id"]
            tau_te_m = tau_te_m.drop(columns=[f"{TAU_COL}_id"])
    
        # 5) Add temporal teacher features ONCE
        tau_te_m = add_teacher_temporal_features_tau(tau_te_m, case_col, TAU_COL)
    
        # 6) Build teacher cols list
        teacher_cols_basic = ["teacher_logit"] + [
            c for c in tau_te_m.columns if str(c).startswith("teacher_embed_pca_")
        ]
        teacher_cols = teacher_cols_basic + [
            "teacher_logit_prev1", "teacher_logit_rm3", "teacher_logit_rstd3"
        ]
    
        # Ensure float32 + fill missing
        for c in teacher_cols:
            if c not in tau_te_m.columns:
                tau_te_m[c] = np.float32(0.0)
            tau_te_m[c] = tau_te_m[c].astype(np.float32, copy=False)
        tau_te_m[teacher_cols] = tau_te_m[teacher_cols].fillna(0.0)
    
        # 7) Create X_te
        X_te = pd.concat(
            [
                X_te_base.reset_index(drop=True),
                tau_te_m[teacher_cols].reset_index(drop=True),
            ],
            axis=1,
        )
    
    else:
        tau_te_m = tau_te_feat.copy()
        X_te = X_te_base

    # 6) Load tau-trained XGB + predict
    print(f"[test] Loading XGB model: {args.model}")
    model = joblib.load(args.model)

    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(X_te.columns)
        if missing:
            for m in missing:
                X_te[m] = 0.0
        X_te = X_te.reindex(columns=model.feature_names_in_, fill_value=0.0)

    probs = model.predict_proba(X_te)[:, 1].astype(np.float32)

    # 7) Metrics
    y_has_labels = (len(np.unique(y_te)) == 2)
    operating_block = None
    pre_first_block = None
    phase3_block = None
    brier_block = None
    config_c_block = None
    thr_op = args.op_thr if args.op_thr is not None else load_operating_threshold(args.result_dir, DEFAULT_OPERATING_THR)

    if y_has_labels:
        metrics_all = eval_metrics(y_te.to_numpy(), probs)

        # Brier (ALL tau rows)
        y_np_full = y_te.to_numpy().astype(np.uint8)
        brier = float(brier_score_loss(y_np_full, probs))
        pos_rate = float(y_np_full.mean())
        brier_baseline = float(pos_rate * (1.0 - pos_rate))
        brier_improvement = float(brier_baseline - brier)
        brier_block = {
            "brier_all_tau": brier,
            "baseline_brier": brier_baseline,
            "pos_rate": pos_rate,
            "brier_improvement_vs_baseline": brier_improvement
        }

        # First-delay-at-tau mask (k-1 at tau) — ALIGNED
        m_te_first = first_delay_mask_tau_aligned(tau_te_m, case_col, TARGET_COL, TAU_COL)
        metrics_first = eval_metrics(y_te.to_numpy()[m_te_first], probs[m_te_first])

        print("\n=== TEST(tau) Metrics ===")
        print(f"All tau         : AUPRC={metrics_all['auprc']:.4f}  AUROC={metrics_all['auroc']:.4f}  "
              f"F1={metrics_all['f1']:.3f}  P={metrics_all['precision']:.3f}  R={metrics_all['recall']:.3f}  "
              f"@thr={metrics_all['best_thr']:.3f}  |  n={metrics_all['n']:,} pos_rate={metrics_all['pos_rate']:.4f}")
        print(f"k-1 at tau      : AUPRC={metrics_first['auprc']:.4f}  AUROC={metrics_first['auroc']:.4f}  "
              f"F1={metrics_first['f1']:.3f}  P={metrics_first['precision']:.3f}  R={metrics_first['recall']:.3f}  "
              f"@thr={metrics_first['best_thr']:.3f}  |  n={metrics_first['n']:,} pos_rate={metrics_first['pos_rate']:.4f}")
        print(f"Brier (all tau)  : {brier:.6f} | baseline={brier_baseline:.6f} | improvement={brier_improvement:.6f}")

        # Operating threshold
        if args.op_thr is not None:
            print(f"[operating] Using threshold override: {thr_op:.3f}")

        yhat_all = (probs >= thr_op).astype(np.uint8)

        # Recall on k-1-at-tau (share of first-delay tau that are alerted)
        recall_km1_op = float((yhat_all[m_te_first] == 1).mean())

        # "Precision" of alerts for k-1-at-tau (share of all alerts that hit k-1 tau)
        pred_pos_all = int(yhat_all.sum())
        pred_pos_km1 = int(yhat_all[m_te_first].sum())
        precision_km1_op = float(pred_pos_km1 / max(1, pred_pos_all))

        # Ranking quality treating k-1-at-tau as positive across ALL tau
        ap_km1 = float(average_precision_score(m_te_first.astype(np.uint8), probs))
        try:
            auroc_km1 = float(roc_auc_score(m_te_first.astype(np.uint8), probs))
        except Exception:
            auroc_km1 = float("nan")

        # Pre-first window at tau: for each delayed case, consider tau ≤ k-1(tau)
        df_tmp = tau_te_m[[case_col, TAU_COL]].copy()
        df_tmp["is_kminus1"] = m_te_first
        kminus1_tau = (
            df_tmp[df_tmp["is_kminus1"]]
            .groupby(case_col)[TAU_COL]
            .min()
            .rename("kminus1_tau")
        )

        df_pf = tau_te_m.merge(kminus1_tau, on=case_col, how="left")
        pre_first_mask = df_pf["kminus1_tau"].notna() & (df_pf[TAU_COL] <= df_pf["kminus1_tau"])

        y_pf    = m_te_first[pre_first_mask.to_numpy()]          # 1 exactly at k-1 tau, 0 earlier tau
        p_pf    = probs[pre_first_mask.to_numpy()]
        yhat_pf = (p_pf >= thr_op).astype(np.uint8)

        TP = int(((yhat_pf == 1) & (y_pf == 1)).sum())   # alerts exactly at k-1 tau
        FP = int(((yhat_pf == 1) & (y_pf == 0)).sum())   # alerts too early (tau before first delay)
        FN = int(((yhat_pf == 0) & (y_pf == 1)).sum())   # missed first-delay tau
        TN = int(((yhat_pf == 0) & (y_pf == 0)).sum())   # correctly silent before k-1 tau

        n_cases_delayed = int(kminus1_tau.index.nunique())
        n_rows_pf       = int(pre_first_mask.sum())
        n_pos_kminus1   = n_cases_delayed
        n_neg_pre_first = n_rows_pf - n_pos_kminus1

        precision_fd = float(TP / max(1, (TP + FP)))
        recall_fd    = float(TP / max(1, (TP + FN)))
        f1_first     = 0.0 if (precision_fd + recall_fd) == 0 else (2*precision_fd*recall_fd / (precision_fd + recall_fd))

        precision_neg = float(TN / max(1, (TN + FN)))   # NPV
        recall_neg    = float(TN / max(1, (TN + FP)))   # specificity
        f1_neg        = 0.0 if (precision_neg + recall_neg) == 0 else (2*precision_neg*recall_neg / (precision_neg + recall_neg))

        total_pf  = TP + FP + FN + TN
        acc_pf    = float((TP + TN) / max(1, total_pf))
        f1_macro  = (f1_first + f1_neg) / 2.0

        # Save CSV blocks (existing)
        first_delay_op_csv = os.path.join(args.out_dir, "test_tau_metrics_first_delay_operating.csv")
        pd.DataFrame([{
            "n_total_tau": int(len(probs)),
            "n_kminus1_tau": int(m_te_first.sum()),
            "operating_threshold": float(thr_op),
            "recall_kminus1_at_thr": recall_km1_op,
            "alerts_total": pred_pos_all,
            "alerts_on_kminus1": pred_pos_km1,
            "share_of_alerts_on_kminus1": precision_km1_op,
            "auprc_kminus1_ranking": ap_km1,
            "auroc_kminus1_ranking": auroc_km1,
        }]).to_csv(first_delay_op_csv, index=False)
        print(f"Saved first-delay(tau) operating metrics -> {first_delay_op_csv}")

        pre_first_csv = os.path.join(args.out_dir, "test_tau_metrics_pre_first_window.csv")
        pd.DataFrame([{
            "threshold": float(thr_op),
            "n_cases_delayed": n_cases_delayed,
            "n_rows_pre_first_tau": n_rows_pf,
            "n_pos_kminus1_tau": n_pos_kminus1,
            "n_neg_pre_first_tau": n_neg_pre_first,
            "tp_at_kminus1_tau": TP,
            "fp_early_tau": FP,
            "fn_missed_kminus1_tau": FN,
            "tn_correct_silence_tau": TN,
            "precision_first_delay_tau": precision_fd,
            "recall_first_delay_tau":    recall_fd,
            "f1_first_delay_tau":        f1_first,
            "precision_non_delayed_NPV": precision_neg,
            "recall_non_delayed_specificity": recall_neg,
            "f1_non_delayed": f1_neg,
            "accuracy_pre_first_window_tau": acc_pf,
            "f1_macro_pre_first_tau":        f1_macro
        }]).to_csv(pre_first_csv, index=False)
        print(f"Saved pre-first window(tau) metrics -> {pre_first_csv}")

        # ── Console summary (existing)
        print("\n[Pre-first window (tau) @ thr={:.3f}]".format(thr_op))
        print("Delayed cases:", n_cases_delayed, "| rows pre-first:", n_rows_pf)
        print("TP@k-1 =", TP, "| FP early =", FP, "| FN missed =", FN, "| TN silent =", TN)
        print("First-delay      — P={:.3f}  R={:.3f}  F1={:.3f}".format(precision_fd, recall_fd, f1_first))
        print("Pre-first (neg)  — NPV={:.3f}  Spec={:.3f}  F1_neg={:.3f}".format(precision_neg, recall_neg, f1_neg))
        print("Window accuracy  — Acc={:.3f}  Macro-F1={:.3f}".format(acc_pf, f1_macro))

        operating_block = {
            "operating_threshold": float(thr_op),
            "recall_kminus1_at_thr": recall_km1_op,
            "alerts_total": pred_pos_all,
            "alerts_on_kminus1": pred_pos_km1,
            "share_of_alerts_on_kminus1": precision_km1_op,
            "auprc_kminus1_ranking": ap_km1,
            "auroc_kminus1_ranking": auroc_km1,
        }
        pre_first_block = {
            "threshold": float(thr_op),
            "tp_at_kminus1_tau": TP,
            "fp_early_tau": FP,
            "fn_missed_kminus1_tau": FN,
            "tn_correct_silence_tau": TN,
            "precision_first_delay_tau": precision_fd,
            "recall_first_delay_tau":    recall_fd,
            "f1_first_delay_tau":        f1_first,
            "accuracy_pre_first_window_tau": acc_pf,
            "f1_macro_pre_first_tau":        f1_macro
        }

        # ─────────────────────────────────────────────────────────────────────
        # Phase 3A / 3B / 3C metrics + prints + exports
        # ─────────────────────────────────────────────────────────────────────
        print("\n=== [Phase 3A/3B/3C] Production-realism metrics @ operating threshold ===")
        print(f"Operating threshold used: {thr_op:.3f}")

        # Build per-row and per-case tables (ALL cases)
        row_df, case_df = compute_case_level_tables(tau_te_m, case_col, TAU_COL, probs, thr_op, m_te_first)

        # Save these tables
        path_row = os.path.join(args.out_dir, "tau_rows_with_alerts.parquet")
        path_case = os.path.join(args.out_dir, "case_level_alert_summary.csv")
        row_df.to_parquet(path_row, index=False)
        case_df.to_csv(path_case, index=False)
        print(f"Saved tau-row table -> {path_row}")
        print(f"Saved case-level table -> {path_case}")

        # 3A) Non-delayed burden
        nd_stats = compute_non_delayed_alarm_burden(case_df)
        print("\n[3A] Non-delayed case false-alarm burden")
        for k, v in nd_stats.items():
            print(f"  {k}: {v}")
        path_nd = os.path.join(args.out_dir, "phase3a_non_delayed_alarm_burden.csv")
        pd.DataFrame([nd_stats]).to_csv(path_nd, index=False)
        print(f"Saved 3A CSV -> {path_nd}")

        # 3B) Case-level precision/recall (E-CaseAlarm style)
        case_pr = compute_case_level_precision_recall(case_df)
        print("\n[3B] Case-level alarm precision/recall (case_alerted vs case_delayed)")
        for k, v in case_pr.items():
            print(f"  {k}: {v}")
        path_3b = os.path.join(args.out_dir, "phase3b_case_level_precision_recall.csv")
        pd.DataFrame([case_pr]).to_csv(path_3b, index=False)
        print(f"Saved 3B CSV -> {path_3b}")

        # 3C) Lead time / timing distribution (delayed cases only)
        lt_stats, per_case_timing = compute_lead_time_stats(row_df, case_col, TAU_COL)
        print("\n[3C] Early-warning timing / lead-time (delayed cases)")
        for k, v in lt_stats.items():
            print(f"  {k}: {v}")

        path_3c = os.path.join(args.out_dir, "phase3c_lead_time_stats.csv")
        pd.DataFrame([lt_stats]).to_csv(path_3c, index=False)
        print(f"Saved 3C lead-time CSV -> {path_3c}")

        path_3c_cases = os.path.join(args.out_dir, "phase3c_per_case_timing.csv")
        per_case_timing.to_csv(path_3c_cases, index=False)
        print(f"Saved 3C per-case timing CSV -> {path_3c_cases}")

        # lead-time histogram
        try:
            lt_min = pd.to_numeric(per_case_timing["lead_time_sec"], errors="coerce") / 60.0
            lt_min = lt_min.replace([np.inf, -np.inf], np.nan).dropna()
            if len(lt_min):
                plt.figure()
                plt.hist(lt_min, bins=50)
                plt.xlabel("Lead time (minutes) = k-1 tau − first alert tau")
                plt.ylabel("Count (delayed cases with early alerts)")
                plt.title("Lead-time distribution (strictly early alerts)")
                _savefig_pdf(plots_dir / "lead_time_histogram_minutes.pdf")
                print(f"Saved lead-time histogram -> {plots_dir / 'lead_time_histogram_minutes.pdf'}")
        except Exception as e:
            print(f"[3C] Could not plot lead-time histogram: {e}")

        phase3_block = {"phase3a": nd_stats, "phase3b": case_pr, "phase3c": lt_stats}

        # ─────────────────────────────────────────────────────────────────────
        # NEW: Config C evaluation (one row per case)
        # ─────────────────────────────────────────────────────────────────────
        print("\n=== [Config C] C-Kminus1PerCase (one row per case) @ operating threshold ===")
        config_c_metrics, config_c_selected = compute_config_c_kminus1_per_case(
            tau_df=tau_te_m,
            case_col=case_col,
            tau_col=TAU_COL,
            probs=probs,
            thr_op=thr_op,
            m_te_first=m_te_first
        )
        for k, v in config_c_metrics.items():
            print(f"  {k}: {v}")

        path_c = os.path.join(args.out_dir, "test_tau_metrics_configC_kminus1_per_case.csv")
        pd.DataFrame([config_c_metrics]).to_csv(path_c, index=False)
        print(f"Saved Config C metrics -> {path_c}")

        path_c_sel = os.path.join(args.out_dir, "test_tau_configC_selected_rows.csv")
        config_c_selected.to_csv(path_c_sel, index=False)
        print(f"Saved Config C selected rows -> {path_c_sel}")

        config_c_block = config_c_metrics

        # Plots (existing)
        plot_roc(y_np_full, probs, plots_dir / "roc_curve.pdf")
        plot_pr(y_np_full, probs, plots_dir / "pr_curve.pdf")
        plot_calibration(y_np_full, probs, plots_dir / "calibration.pdf")
        plot_histograms(y_np_full, probs, plots_dir / "score_histograms.pdf")
        plot_confusion(y_np_full, probs, metrics_all["best_thr"], plots_dir / "confusion_matrix_at_bestF1.pdf")
        print(f"Plots saved to {plots_dir}")

    else:
        print("[info] Labels missing or constant on TEST tau; skipping metric computation.")
        metrics_all = None
        metrics_first = None
        m_te_first = first_delay_mask_tau_aligned(tau_te_m, case_col, TARGET_COL, TAU_COL)

    # 8) Save predictions as Parquet (+ teacher diagnostics if available)
    out_df = tau_te_m[[case_col, TAU_COL]].copy()
    if TARGET_COL in tau_te_m.columns:
        out_df["label"] = tau_te_m[TARGET_COL].astype(np.uint8)

    try:
        m_te_first_out = first_delay_mask_tau_aligned(tau_te_m, case_col, TARGET_COL, TAU_COL)
    except Exception:
        m_te_first_out = np.zeros(len(out_df), dtype=bool)

    out_df["is_first_delay_tau"] = m_te_first_out.astype(np.int8)
    out_df["proba"] = probs.astype(np.float32)
    out_df["alert_op"] = (out_df["proba"].to_numpy() >= float(thr_op)).astype(np.uint8)

    if not args.student_only:
        if "teacher_logit" in tau_te_m.columns:
            out_df["teacher_logit"] = tau_te_m["teacher_logit"].astype(np.float32)
        for c in [c for c in tau_te_m.columns if str(c).startswith("teacher_embed_pca_")]:
            out_df[c] = tau_te_m[c].astype(np.float32)

    pred_path_parquet = os.path.join(args.out_dir, "test_tau_predictions.parquet")
    out_df.to_parquet(pred_path_parquet, index=False)
    print(f"\nSaved predictions (Parquet) -> {pred_path_parquet}")

    # 9) Save metrics JSON
    metr = {
        "overall": metrics_all,
        "k_minus_1_tau": metrics_first,
        "brier": brier_block,
        "operating_point": operating_block if operating_block is not None else {"operating_threshold": float(thr_op)},
        "pre_first_window_tau": pre_first_block,
        "phase3": phase3_block,
        "configC_kminus1_per_case": config_c_block,
        "counts": {
            "rows": int(len(out_df)),
            "first_delay_tau_rows": int(np.asarray(out_df["is_first_delay_tau"]).sum()),
            "alerts_at_operating_threshold": int(np.asarray(out_df["alert_op"]).sum()),
        },
        "paths": {
            "predictions_parquet": pred_path_parquet,
            "out_dir": args.out_dir
        },
        "config": {
            "result_dir": args.result_dir,
            "art_dir": None if args.student_only else args.art_dir,
            "pca_path": None if args.student_only else args.pca_path,
            "case_col": case_col,
            "student_only": bool(args.student_only),
            "model_path": args.model,
            "tau_test_parquet": args.tau_test,
            "raw_test_parquet": args.raw_test,
        }
    }
    metr_path = os.path.join(args.out_dir, "test_tau_metrics.json")
    with open(metr_path, "w", encoding="utf-8") as f:
        json.dump(metr, f, indent=2)
    print(f"Saved metrics JSON -> {metr_path}")

    print("\n[DONE] tau-Test evaluation complete (EXTENDED + Config C).")

if __name__ == "__main__":
    main()
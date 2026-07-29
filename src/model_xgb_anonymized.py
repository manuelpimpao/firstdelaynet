# model_xgb.py
"""
Train an XGBoost student classifier on tau-anchored rows and export:
- ROC/PR/confusion metrics
- calibration plots
- threshold curves
- global feature importance by XGBoost gain
- first-event subset permutation importance
- validation operating threshold
"""

import os
import json
import inspect
from typing import Optional, Sequence

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    precision_recall_curve,
    roc_curve,
    auc as sk_auc,
)
from xgboost import XGBClassifier

from features_xgb_anonymized import build_tau_features


mpl.rcParams["savefig.format"] = "pdf"
mpl.rcParams["savefig.bbox"] = "tight"


TRAIN_TAU = "data/anonymized/tau_train_anonymized.csv"
VAL_TAU = "data/anonymized/tau_val_anonymized.csv"
RESULT_DIR = "results"
RAND_STATE = 42


def plot_and_save_basic_metrics(y_true, y_prob, y_pred, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "roc_curve.pdf"))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision, label=f"AP={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "pr_curve.pdf"))
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)

    plt.figure()
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.pdf"))
    plt.close()

    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(out_dir, "classification_report.csv")
    )

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": log_loss(y_true, y_prob),
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "metrics_summary.csv"),
        index=False,
    )


def plot_and_save_threshold_curves(y_true, y_prob, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-6, None)

    df_thresholds = pd.DataFrame(
        {
            "threshold": np.append(thresholds, 1.0),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    )

    df_thresholds.to_csv(os.path.join(out_dir, "threshold_metrics.csv"), index=False)

    plt.figure()
    plt.plot(df_thresholds["threshold"], df_thresholds["precision"], label="Precision")
    plt.plot(df_thresholds["threshold"], df_thresholds["recall"], label="Recall")
    plt.plot(df_thresholds["threshold"], df_thresholds["f1_score"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 by Threshold")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "threshold_curves.pdf"))
    plt.close()


def plot_and_save_calibration(y_true, y_prob, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    display = CalibrationDisplay.from_predictions(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=10,
        name="XGBoost student",
    )

    plt.figure()
    display.plot()
    plt.title("Calibration Curve")
    plt.savefig(os.path.join(out_dir, "calibration_curve.pdf"))
    plt.close()


def plot_and_save_extended_evals(clf: XGBClassifier, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    try:
        results = clf.evals_result()
    except Exception:
        return

    def _safe_series(results_dict, dataset_key, metric_key):
        try:
            return pd.Series(results_dict[dataset_key][metric_key])
        except Exception:
            return pd.Series(dtype=float)

    learning_curves = [
        ("auc", "AUC", "learning_curve_auc"),
        ("logloss", "Log-Loss", "learning_curve_logloss"),
        ("aucpr", "AUC-PR", "learning_curve_aucpr"),
    ]

    for metric_key, label, filename in learning_curves:
        train_series = _safe_series(results, "validation_0", metric_key)
        val_series = _safe_series(results, "validation_1", metric_key)

        if train_series.empty or val_series.empty:
            continue

        df_curve = pd.DataFrame(
            {
                "train": train_series,
                "validation": val_series,
            }
        )

        df_curve.to_csv(
            os.path.join(out_dir, f"{filename}.csv"),
            index_label="iteration",
        )

        plt.figure()
        plt.plot(df_curve["train"], label=f"Train {label}")
        plt.plot(df_curve["validation"], label=f"Validation {label}")
        plt.xlabel("Iteration")
        plt.ylabel(label)
        plt.title(f"Learning Curve ({label})")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{filename}.pdf"))
        plt.close()

    booster = clf.get_booster()
    importance = booster.get_score(importance_type="gain")

    df_importance = (
        pd.DataFrame.from_dict(importance, orient="index", columns=["gain"])
        .sort_values("gain", ascending=False)
    )

    df_importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), index_label="feature")

    plt.figure(figsize=(6, max(4, 0.2 * len(df_importance))))
    df_importance["gain"].plot.barh()
    plt.gca().invert_yaxis()
    plt.xlabel("Gain")
    plt.title("Feature Importance by Gain")
    plt.savefig(os.path.join(out_dir, "feature_importance.pdf"))
    plt.close()


def export_feature_gain_xgb(model, X, out_csv: str = "feature_gain.csv", ensure_dir: bool = True):
    if hasattr(X, "columns"):
        feature_names = list(X.columns)
    elif hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        n_features = getattr(X, "shape", (0, 0))[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]

    raw_gain = model.get_booster().get_score(importance_type="gain")
    feature_map = {f"f{i}": name for i, name in enumerate(feature_names)}
    mapped_gain = {feature_map.get(k, k): v for k, v in raw_gain.items()}

    output = pd.Series(mapped_gain, dtype=float, name="gain")

    for name in feature_names:
        if name not in output.index:
            output.loc[name] = 0.0

    output = output.sort_values(ascending=False)

    if ensure_dir:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    output.to_csv(out_csv, header=True)
    print(f"[feature_gain] Wrote {len(output)} entries to {os.path.abspath(out_csv)}")


def _subset_permutation_importance(
    model: XGBClassifier,
    X_val,
    y_val: np.ndarray,
    subset_mask_val: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "average_precision",
) -> list[tuple[str, float]]:
    rng = np.random.default_rng(random_state)
    X_array = X_val.values if hasattr(X_val, "values") else np.asarray(X_val)

    subset_idx = np.where(subset_mask_val.astype(bool))[0]

    if subset_idx.size == 0:
        return []

    base_pred = model.predict_proba(X_array)[:, 1]

    if scoring == "average_precision":
        base_score = average_precision_score(y_val[subset_idx], base_pred[subset_idx])
    elif scoring == "roc_auc":
        base_score = sk_auc(*roc_curve(y_val[subset_idx], base_pred[subset_idx])[:2])
    elif scoring == "neg_logloss":
        base_score = -log_loss(y_val[subset_idx], base_pred[subset_idx], labels=[0, 1])
    else:
        raise ValueError(f"Unsupported scoring: {scoring}")

    drops = np.zeros(X_array.shape[1], dtype=float)

    for j in range(X_array.shape[1]):
        repeated_drops = []

        for _ in range(n_repeats):
            X_perturbed = X_array.copy()

            shuffled = X_perturbed[subset_idx, j].copy()
            rng.shuffle(shuffled)
            X_perturbed[subset_idx, j] = shuffled

            pred = model.predict_proba(X_perturbed)[:, 1]

            if scoring == "average_precision":
                score = average_precision_score(y_val[subset_idx], pred[subset_idx])
            elif scoring == "roc_auc":
                score = sk_auc(*roc_curve(y_val[subset_idx], pred[subset_idx])[:2])
            elif scoring == "neg_logloss":
                score = -log_loss(y_val[subset_idx], pred[subset_idx], labels=[0, 1])

            repeated_drops.append(max(0.0, base_score - score))

        drops[j] = float(np.mean(repeated_drops))

    pairs = [(feature_names[j], float(drops[j])) for j in range(len(feature_names))]
    pairs.sort(key=lambda x: x[1], reverse=True)

    return pairs


def _plot_top_bars(
    pairs: list[tuple[str, float]],
    title: str,
    outpath: str,
    top_k: int = 30,
    xlabel: str = "Importance",
):
    top_pairs = pairs[:top_k]

    if not top_pairs:
        return

    names = [name for name, _ in top_pairs][::-1]
    values = [value for _, value in top_pairs][::-1]

    plt.figure(figsize=(8, max(4, 0.28 * len(top_pairs))))
    plt.barh(names, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()


def first_positive_tau_mask(
    df_tau: pd.DataFrame,
    case_col: str = "case_seg",
    k_idx_col: str = "k_idx",
    target_col: str = "y_next_delay",
) -> np.ndarray:
    if case_col not in df_tau.columns or k_idx_col not in df_tau.columns or target_col not in df_tau.columns:
        return np.zeros(len(df_tau), dtype=bool)

    interval_labels = (
        df_tau.groupby([case_col, k_idx_col], sort=False)[target_col]
        .max()
        .reset_index()
    )

    first_positive = (
        interval_labels[interval_labels[target_col] == 1]
        .sort_values([case_col, k_idx_col])
        .drop_duplicates(case_col, keep="first")
        .rename(columns={k_idx_col: "_first_positive_k"})
        [[case_col, "_first_positive_k"]]
    )

    df_mask = df_tau[[case_col, k_idx_col]].copy()
    df_mask = df_mask.merge(first_positive, on=case_col, how="left")

    mask = (
        df_mask["_first_positive_k"].notna()
        & (df_mask[k_idx_col] == df_mask["_first_positive_k"])
    )

    return mask.to_numpy(dtype=bool)


def train_xgb(
    *,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    model_path: str,
    sample_weight_train: Optional[np.ndarray] = None,
    sample_weight_val: Optional[np.ndarray] = None,
    first_delay_mask_val: Optional[np.ndarray] = None,
    first_delay_mask_train: Optional[np.ndarray] = None,
    first_delay_boost: float = 1.0,
    feature_names: Optional[Sequence[str]] = None,
    early_stopping_rounds: int = 50,
    out_dir: Optional[str] = None,
    **params,
):
    out_dir = out_dir or RESULT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("Hyperparameters:")
    for key, value in params.items():
        print(f"  - {key} = {value}")
    print()

    w_train = (
        np.ones(len(y_train), dtype=float)
        if sample_weight_train is None
        else np.asarray(sample_weight_train, dtype=float)
    )

    w_val = (
        None
        if sample_weight_val is None
        else np.asarray(sample_weight_val, dtype=float)
    )

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric=["aucpr", "auc", "logloss"],
        use_label_encoder=False,
        random_state=RAND_STATE,
        **params,
    )

    print(f"Training on {len(X_train)} tau samples, validating on {len(X_val)}.")

    fit_signature = inspect.signature(clf.fit)
    supported = set(fit_signature.parameters.keys())

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
    }

    if "sample_weight" in supported:
        fit_kwargs["sample_weight"] = w_train

    has_eval_set = "eval_set" in supported

    if has_eval_set:
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]

    if w_val is not None and "eval_sample_weight" in supported:
        fit_kwargs["eval_sample_weight"] = [w_train, w_val]

    if "verbose" in supported:
        fit_kwargs["verbose"] = 10

    if "early_stopping_rounds" in supported:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    elif (
        "callbacks" in supported
        and has_eval_set
        and early_stopping_rounds
        and early_stopping_rounds > 0
    ):
        fit_kwargs["callbacks"] = [
            xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                save_best=True,
                maximize=True,
                data_name="validation_1",
                metric_name="aucpr",
            )
        ]

    clf.fit(**fit_kwargs)
    print("Training finished.")

    print(f"Saving model to {model_path}")
    joblib.dump(clf, model_path)

    X_val_array = X_val.values if hasattr(X_val, "values") else X_val
    y_prob = clf.predict_proba(X_val_array)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-6, None)

    best_idx = int(np.nanargmax(f1)) if f1.size else 0
    best_threshold_overall = (
        float(thresholds[best_idx])
        if thresholds.size and best_idx < thresholds.size
        else 0.5
    )

    with open(os.path.join(out_dir, "chosen_threshold.txt"), "w", encoding="utf-8") as f:
        f.write(f"{best_threshold_overall:.6f}\n")

    if first_delay_mask_val is not None and np.any(first_delay_mask_val):
        subset_mask = np.asarray(first_delay_mask_val, dtype=bool)

        subset_precision, subset_recall, subset_thresholds = precision_recall_curve(
            subset_mask.astype(np.uint8),
            y_prob,
        )

        subset_f1 = (
            2
            * subset_precision
            * subset_recall
            / np.clip(subset_precision + subset_recall, 1e-6, None)
        )

        subset_best_idx = int(np.nanargmax(subset_f1)) if subset_f1.size else 0
        subset_thresholds_full = np.append(subset_thresholds, 1.0)

        best_threshold_subset = (
            float(subset_thresholds_full[subset_best_idx])
            if subset_thresholds_full.size
            else best_threshold_overall
        )

        yhat_all = (y_prob >= best_threshold_subset).astype(np.uint8)

        subset_recall_value = (
            float((yhat_all[subset_mask] == 1).mean())
            if subset_mask.any()
            else float("nan")
        )

        alerts_total = int(yhat_all.sum())
        alerts_on_subset = int(yhat_all[subset_mask].sum()) if subset_mask.any() else 0
        subset_precision_value = float(alerts_on_subset / max(1, alerts_total))

        print("\nValidation operating-threshold selection")
        print(f"Chosen threshold: {best_threshold_subset:.3f}")
        print(f"Recall on target subset: {subset_recall_value:.3f}")
        print(f"Total alerts: {alerts_total:,}")
        print(f"Alerts on target subset: {alerts_on_subset:,}")
        print(f"Target-subset precision: {subset_precision_value:.3f}")

        with open(os.path.join(out_dir, "val_operating_threshold.txt"), "w", encoding="utf-8") as f:
            f.write(f"{best_threshold_subset:.6f}\n")
    else:
        with open(os.path.join(out_dir, "val_operating_threshold.txt"), "w", encoding="utf-8") as f:
            f.write(f"{best_threshold_overall:.6f}\n")

        print(
            "\nTarget-subset mask not provided or empty. "
            "Saved overall best-F1 threshold as operating threshold."
        )

    y_pred_overall = (y_prob >= best_threshold_overall).astype(int)

    plot_and_save_basic_metrics(y_val, y_prob, y_pred_overall, out_dir)
    plot_and_save_threshold_curves(y_val, y_prob, out_dir)
    plot_and_save_calibration(y_val, y_prob, out_dir)
    plot_and_save_extended_evals(clf, out_dir)

    export_feature_gain_xgb(
        clf,
        X_train,
        out_csv=os.path.join(out_dir, "feature_gain.csv"),
    )

    booster = clf.get_booster()
    raw_gain = booster.get_score(importance_type="gain")

    if feature_names is None:
        if hasattr(X_train, "columns"):
            feature_names = list(X_train.columns)
        elif hasattr(clf, "feature_names_in_"):
            feature_names = list(clf.feature_names_in_)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    gain_pairs = [
        (name, float(raw_gain.get(f"f{i}", 0.0)))
        for i, name in enumerate(feature_names)
    ]

    gain_pairs.sort(key=lambda x: x[1], reverse=True)

    with open(
        os.path.join(out_dir, "feature_importance_global_gain.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(gain_pairs, f, ensure_ascii=False, indent=2)

    _plot_top_bars(
        gain_pairs,
        title="Global XGBoost Gain",
        outpath=os.path.join(out_dir, "feature_importance_global_gain_top30.pdf"),
        top_k=30,
        xlabel="Gain",
    )

    try:
        if first_delay_mask_val is not None and np.any(first_delay_mask_val):
            permutation_pairs = _subset_permutation_importance(
                model=clf,
                X_val=X_val,
                y_val=y_val,
                subset_mask_val=first_delay_mask_val,
                feature_names=list(feature_names),
                n_repeats=5,
                random_state=42,
                scoring="average_precision",
            )

            with open(
                os.path.join(out_dir, "feature_importance_subset_perm.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(permutation_pairs, f, ensure_ascii=False, indent=2)

            _plot_top_bars(
                permutation_pairs,
                title="Permutation Importance on Target Subset",
                outpath=os.path.join(out_dir, "feature_importance_subset_perm_top30.pdf"),
                top_k=30,
                xlabel="AP drop when permuted",
            )

            print("Wrote target-subset importance files.")
        else:
            print("Target-subset mask not provided or empty. Skipping subset importances.")

    except Exception as exc:
        print(f"Subset importance computation failed: {exc}")

    print(f"All metrics and plots saved under: {out_dir}/")

    return clf


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    train_tau = pd.read_csv(TRAIN_TAU)
    val_tau = pd.read_csv(VAL_TAU)

    X_train, y_train, _ = build_tau_features(
        train_tau,
        drop_extra=None,
        materialize=True,
    )

    X_val, y_val, _ = build_tau_features(
        val_tau,
        drop_extra=None,
        materialize=True,
    )

    case_col = "case_seg" if "case_seg" in val_tau.columns else "case:concept:name"

    target_subset_mask = first_positive_tau_mask(
        val_tau,
        case_col=case_col,
        k_idx_col="k_idx",
        target_col="y_next_delay",
    )

    defaults = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "tree_method": "hist",
    }

    train_xgb(
        X_train=X_train,
        y_train=y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train),
        X_val=X_val,
        y_val=y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val),
        model_path=os.path.join(RESULT_DIR, "xgb_student_tau.joblib"),
        early_stopping_rounds=50,
        first_delay_mask_val=target_subset_mask,
        feature_names=list(X_train.columns) if hasattr(X_train, "columns") else None,
        out_dir=RESULT_DIR,
        **defaults,
    )


if __name__ == "__main__":
    main()
# train_teacher.py
# One-file flow: load split CSV/Parquet -> build teacher datasets in memory -> train teacher sequence model
# Now with rich metrics & charts + numerically-stable eval + feature normalization.
# UPDATED:
#   1) Smart reader for CSV/Parquet
#   2) Auto-prune NUM_COLS to what's present
#   3) Initialize output bias from empirical positive rate (prior_pos_rate)

import os, json, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===== extra metrics & plotting =====
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score, confusion_matrix, brier_score_loss
)

# ========= 1) CONFIG (CONFIG) =============================================

# Preprocessed event-log splits
TRAIN_PATH = "data/anonymized/train_events_anonymized.parquet"
VAL_PATH   = "data/anonymized/val_events_anonymized.parquet"
# You can point these to .parquet files; the loader below will handle either extension.

ARTIFACTS_DIR = "model_artifacts/teacher"           # where to save model, vocab, logs, keys, metrics, plots

# Teacher sequence settings
MAX_LEN   = 100
CAT_COL   = "concept:name"
CASE_COL  = "case_seg"
PREFIX_COL= "prefix_index"
LABEL_COL = "next_is_delayed_dyn"
MASK_COL  = "mask_transport"          # 1 only if NEXT event is transport

# Desired numeric features (we will intersect with what's actually present after loading)
NUM_COLS_DESIRED = [
    # timing/cadence
    "inter_event_time", "roll_ie_mean", "roll_ie_min", "roll_ie_max",
    # calendar (weekday dummies are typical; hour_of_day often isn't emitted)
    # "hour_of_day",
    "weekday_0","weekday_1","weekday_2","weekday_3","weekday_4","weekday_5","weekday_6",
    # current event delay-ish
    #"delay", "dynamic_delay",
    # planned-vs-actual pressure at event_end (1h)
    "service_rate_event_end_1h","delays_count_event_end_1h","avg_delay_event_end_1h","std_delay_event_end_1h",
    "planned_60","arrival_rate_event_end_1h","in_flight_event_end_1h",
    "backlog_event_end_1h","delay_rate_event_end_1h","log_delay_rate_event_end_1h",
    # short windows at event_end (optional but useful)
    "starts_5m_event_end","delayed_5m_event_end","planned_arrivals_5m_event_end","backlog_5m_event_end",
    "starts_15m_event_end","delayed_15m_event_end","planned_arrivals_15m_event_end","backlog_15m_event_end",
    # leakage-prone items (usually removed upstream) – leave commented if not τ-safe
    # "slack_to_planned_start_next",
]

# Model/opt hyperparams (Transformer)
D_MODEL      = 192
N_LAYERS     = 2
N_HEADS      = 4
D_FF         = 384
DROPOUT      = 0.2
USE_CAUSAL   = False                  # True => strictly left-looking attention
CAT_EMB_DIM  = 64

BATCH_SIZE   = 128
EPOCHS       = 40

# ---- Stability-oriented optimizer settings
LR           = 1e-4       # was 3e-4
WEIGHT_DECAY = 1e-4       # was 1e-2
WARMUP_FRAC  = 0.10       # ~10% warmup before cosine
MAX_GRAD_NORM= 1.0
PATIENCE     = 8         # early stop on masked PR-AUC
AMP          = True       # use mixed precision on CUDA
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42

# ========= 2) IMPORT PROJECT MODULES =========================================

from teacher_dataset_anonymized import TeacherDatasetBuilder, DatasetPack
from teacher_model_anonymized import TeacherModel, TeacherConfig

# ========= 3) UTILS ===========================================================

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _smart_read(path: str) -> pd.DataFrame:
    path = os.path.expanduser(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

class MaskedBCELoss(nn.Module):
    """BCE-with-logits computed only where mask==1."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets, mask):
        # logits, targets, mask: (B, T)
        loss_all = self.bce(logits, targets)
        masked = loss_all * mask
        denom = torch.clamp(mask.sum(), min=1.0)
        return masked.sum() / denom

@torch.no_grad()
def masked_pr_auc(logits, targets, mask) -> float:
    m = mask > 0.5
    if m.sum().item() == 0:
        return 0.5
    y = targets[m].detach().cpu().numpy()
    if y.size == 0 or (y.max() == y.min()):
        return 0.5
    s = logits[m].detach().sigmoid().cpu().numpy()
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return 0.5

# ========= plotting helpers ===================================================

def _ensure_dir(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def _savefig_pdf(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()

def plot_train_loss(step_hist, loss_hist, out_pdf):
    plt.figure()
    plt.plot(step_hist, loss_hist, linewidth=1.5, label="Train BCE (masked)")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss")
    plt.legend(loc="best")
    _savefig_pdf(out_pdf)

def plot_epoch_loss(epochs, tr, vl, out_pdf):
    plt.figure()
    plt.plot(epochs, tr, linewidth=1.5, label="Train")
    plt.plot(epochs, vl, linewidth=1.5, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Masked BCE"); plt.title("Epoch Loss"); plt.legend(loc="best")
    _savefig_pdf(out_pdf)

def plot_pr(labels, probs, out_pdf):
    # AUPRC on-plot
    ap = average_precision_score(labels, probs) if len(np.unique(labels)) == 2 else float("nan")
    prec, rec, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(rec, prec, linewidth=1.5, label=f"PR curve (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall")
    plt.legend(loc="lower left")
    _savefig_pdf(out_pdf)

def plot_roc(labels, probs, out_pdf):
    if len(np.unique(labels)) < 2:
        return
    auc = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=1.5, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'--', linewidth=1, label="Chance")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(loc="lower right")
    _savefig_pdf(out_pdf)

def plot_calibration(labels, probs, out_pdf, n_bins=20):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(probs, bins) - 1
    accs, confs = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        accs.append(labels[m].mean()); confs.append(probs[m].mean())
    if not confs: return
    bs = brier_score_loss(labels, probs)
    plt.figure()
    plt.plot(confs, accs, marker="o", linewidth=1.5, label="Reliability")
    plt.plot([0,1],[0,1],'--', linewidth=1, label="Perfect")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Empirical positive rate")
    plt.title(f"Calibration (Brier={bs:.4f})"); plt.legend(loc="best")
    _savefig_pdf(out_pdf)

def plot_histograms(labels, probs, out_pdf):
    plt.figure()
    plt.hist(probs[labels==0], bins=50, alpha=0.6, label="neg")
    plt.hist(probs[labels==1], bins=50, alpha=0.6, label="pos")
    plt.xlabel("Predicted probability"); plt.ylabel("Count"); plt.title("Score Distributions"); plt.legend(loc="best")
    _savefig_pdf(out_pdf)

def plot_confusion(labels, probs, thr, out_pdf):
    yhat = (probs >= thr).astype(np.uint8)
    cm = confusion_matrix(labels, yhat, labels=[0,1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    for (i,j), v in np.ndenumerate(cm): plt.text(j, i, str(v), ha="center", va="center")
    plt.title(f"Confusion @ {thr:.3f}"); plt.xlabel("Predicted"); plt.ylabel("True")
    _savefig_pdf(out_pdf)

def plot_threshold_curves(labels, probs, out_pdf):
    prec, rec, thr = precision_recall_curve(labels, probs)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-6, None)
    thr_full = np.append(thr, 1.0)
    plt.figure()
    plt.plot(thr_full, prec, label="Precision")
    plt.plot(thr_full, rec,  label="Recall")
    plt.plot(thr_full, f1,   label="F1")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Threshold Sweep"); plt.legend(loc="best")
    _savefig_pdf(out_pdf)

# ========= 4) BUILD IN-MEMORY DATASETS (no .npz needed) ======================

class InMemoryTeacherDataset(Dataset):
    """
    Wrap a DatasetPack (from TeacherDatasetBuilder.build) for PyTorch.
    We tile y_t (N,) across time so loss can be computed per step but masked at last step.
    """
    def __init__(self, pack: DatasetPack):
        self.seq_cat = pack.seq_cat      # (N, T) int
        self.seq_num = pack.seq_num      # (N, T, F) float
        self.mask_t  = pack.mask_t       # (N, T) float
        self.y_vec   = pack.y_t          # (N,)  float
        self.T       = self.seq_cat.shape[1]
        self.keys    = pack.keys.copy()  # for distillation later
    def __len__(self): return self.seq_cat.shape[0]
    def __getitem__(self, idx):
        y_seq = np.repeat(self.y_vec[idx], self.T).astype(np.float32)  # (T,)
        return {"seq_cat": self.seq_cat[idx], "seq_num": self.seq_num[idx],
                "mask_t":  self.mask_t[idx],  "y_t":     y_seq}

def teacher_collate(batch):
    seq_cat = torch.tensor(np.stack([b["seq_cat"] for b in batch], axis=0), dtype=torch.long)     # (B,T)
    seq_num = torch.tensor(np.stack([b["seq_num"] for b in batch], axis=0), dtype=torch.float32)  # (B,T,F)
    mask_t  = torch.tensor(np.stack([b["mask_t"]  for b in batch], axis=0), dtype=torch.float32)  # (B,T)
    y_t     = torch.tensor(np.stack([b["y_t"]     for b in batch], axis=0), dtype=torch.float32)  # (B,T)
    return {"seq_cat":seq_cat, "seq_num":seq_num, "mask_t":mask_t, "y_t":y_t}

# ========= 5) EVALUATION (masked, transport-only) ============================

@torch.no_grad()
def evaluate_full_epoch(model, loader, device, criterion, mu_t: torch.Tensor, sd_t: torch.Tensor):
    model.eval()
    losses = []
    all_logits = []
    all_labels = []
    all_mask = []

    for batch in loader:
        seq_cat = batch["seq_cat"].to(device)
        seq_num = batch["seq_num"].to(device)
        mask_t  = batch["mask_t"].to(device)
        y_t     = batch["y_t"].to(device)

        # sanitize numerics + normalize on-device
        seq_num = torch.nan_to_num(seq_num, nan=0.0, posinf=1e6, neginf=-1e6).to(torch.float32)
        seq_num = (seq_num - mu_t) / sd_t

        logits, _ = model(seq_cat=seq_cat, seq_num=seq_num)  # (B,T)
        loss = criterion(logits, y_t, mask_t)
        losses.append(loss.item())

        # collect masked positions only
        m = mask_t.bool()
        all_logits.append(logits[m].detach().float().cpu())
        all_labels.append(y_t[m].detach().float().cpu())
        all_mask.append(int(m.sum().item()))

    if len(all_logits) == 0 or (all_mask and sum(all_mask) == 0):
        return {
            "val_bce": float(np.mean(losses)) if losses else float("nan"),
            "auprc": float("nan"), "auroc": float("nan"),
            "best_thr": 0.5, "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "n_pos": 0, "n_neg": 0, "labels": np.array([]), "probs": np.array([])
        }

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(np.uint8)
    logits_clipped = np.clip(logits, -80, 80)
    probs  = 1.0 / (1.0 + np.exp(-logits_clipped))

    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)

    try:
        auprc = average_precision_score(labels, probs)
    except Exception:
        auprc = float("nan")

    try:
        auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else float("nan")
    except Exception:
        auroc = float("nan")

    prec, rec, thr = precision_recall_curve(labels, probs)
    thr_full = np.append(thr, 1.0)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-6, None)
    best_idx = int(np.nanargmax(f1))
    best_thr = float(thr_full[best_idx])
    yhat = (probs >= best_thr).astype(np.uint8)
    f1_best = f1_score(labels, yhat, zero_division=0)
    p_best  = precision_score(labels, yhat, zero_division=0)
    r_best  = recall_score(labels, yhat, zero_division=0)

    return {
        "val_bce": float(np.mean(losses)),
        "auprc": float(auprc), "auroc": float(auroc),
        "best_thr": best_thr, "f1": float(f1_best), "precision": float(p_best), "recall": float(r_best),
        "n_pos": n_pos, "n_neg": n_neg, "labels": labels, "probs": probs
    }

# ========= 6) MAIN FLOW =======================================================

def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = (DEVICE == "cuda")
    _ensure_dir(ARTIFACTS_DIR, os.path.join(ARTIFACTS_DIR, "plots"))
    metrics_csv = os.path.join(ARTIFACTS_DIR, "metrics_epoch.csv")

    # init metrics CSV
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_bce,val_bce,auprc,auroc,f1,precision,recall,best_threshold,n_pos,n_neg\n")

    # 5.1 Load splits (CSV or Parquet)
    print("Loading splits…")
    train_df = _smart_read(TRAIN_PATH)
    val_df   = _smart_read(VAL_PATH)

    # 5.1.1 Auto-prune NUM_COLS to what's present in BOTH splits
    present = set(train_df.columns) & set(val_df.columns)
    NUM_COLS = [c for c in NUM_COLS_DESIRED if c in present]
    missing  = [c for c in NUM_COLS_DESIRED if c not in present]
    if missing:
        print(f"[warn] {len(missing)} desired numeric features not found and will be skipped (showing up to 12): {missing[:12]}")

    # 5.2 Build teacher datasets in memory
    print("Building teacher datasets in memory…")
    builder = TeacherDatasetBuilder(
        cat_col=CAT_COL, num_cols=NUM_COLS,
        case_col=CASE_COL, prefix_col=PREFIX_COL,
        label_col=LABEL_COL, mask_col=MASK_COL,
        max_len=MAX_LEN,
    )

    vocab = builder.build_vocab(train_df)

    train_pack = builder.build(train_df, drop_unmasked=True)
    val_pack   = builder.build(val_df,   drop_unmasked=True)

    # ---- Sanitization (dtype-aware) to avoid float16 overflow during builder steps
    def sanitize_pack_inplace(pack):
        X = pack.seq_num
        if X.size:
            # Use representable caps for float16; looser for float32/64
            cap = 6e4 if X.dtype == np.float16 else 1e6
            np.nan_to_num(X, copy=False, nan=0.0, posinf=cap, neginf=-cap)
            np.clip(X, -cap, cap, out=X)

    sanitize_pack_inplace(train_pack)
    sanitize_pack_inplace(val_pack)

    # ---- Per-feature normalization: compute stats ONCE (float32), then normalize ON DEVICE per-batch
    Xtr32 = train_pack.seq_num.astype(np.float32, copy=False)
    if Xtr32.size:
        mu = Xtr32.mean(axis=(0,1), keepdims=True)
        sd = Xtr32.std(axis=(0,1), keepdims=True)
        sd[sd < 1e-6] = 1.0
    else:
        mu = np.zeros((1,1,0), dtype=np.float32)
        sd = np.ones((1,1,0),  dtype=np.float32)

    # Save stats for reproducibility
    np.savez(os.path.join(ARTIFACTS_DIR, "num_feature_norm_stats.npz"),
             mean=mu, std=sd, cols=np.array(NUM_COLS, dtype=object))

    # Save vocab + keys (no need to mutate seq_num arrays)
    builder.save_vocab(ARTIFACTS_DIR)
    train_pack.keys.to_parquet(os.path.join(ARTIFACTS_DIR, "teacher_train_keys.parquet"), index=False)
    val_pack.keys.to_parquet(  os.path.join(ARTIFACTS_DIR, "teacher_val_keys.parquet"),   index=False)

    # 5.3 Datasets / loaders
    pin = (DEVICE == "cuda")
    train_ds = InMemoryTeacherDataset(train_pack)
    val_ds   = InMemoryTeacherDataset(val_pack)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=teacher_collate, num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=teacher_collate, num_workers=0, pin_memory=pin)

    NUM_NUMERIC_CHANNELS = train_ds.seq_num.shape[-1]

    # Prepare normalization tensors on device (shape (1,1,F))
    mu_t = torch.from_numpy(mu).to(DEVICE).to(torch.float32)
    sd_t = torch.from_numpy(sd).to(DEVICE).to(torch.float32)

    # ========= 7) MODEL =======================================================
    # Base rate from train (masked positions only) for output bias init
    pos_rate = float(train_pack.y_t.mean()) if train_pack.y_t.size else None
    if pos_rate is not None:
        print(f"[info] estimated positive rate ≈ {pos_rate:.4f}")

    cfg = TeacherConfig(
        num_cats={CAT_COL: len(vocab)},
        num_num_features=NUM_NUMERIC_CHANNELS,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, dropout=DROPOUT, max_len=MAX_LEN,
        use_causal=USE_CAUSAL, cat_emb_dim=CAT_EMB_DIM,
        prior_pos_rate=pos_rate
    )
    model = TeacherModel(cfg).to(DEVICE)
    print(f"Model params: {model.count_params()/1e6:.2f}M")

    criterion = MaskedBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / BATCH_SIZE))
    total_steps     = EPOCHS * steps_per_epoch

    # ---- Warmup -> Cosine LR schedule
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    cosine_steps = max(1, total_steps - warmup_steps)
    sched_warm  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    sched_cos   = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler   = SequentialLR(optimizer, [sched_warm, sched_cos], milestones=[warmup_steps])

    scaler = torch.amp.GradScaler(enabled=(AMP and DEVICE == "cuda"))
    use_amp = AMP and DEVICE == "cuda"

    # save model + trainer config for reproducibility
    with open(os.path.join(ARTIFACTS_DIR, "teacher_model_config.json"), "w", encoding="utf-8") as f:
        f.write(cfg.to_json())
    with open(os.path.join(ARTIFACTS_DIR, "teacher_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    best_auc = -1.0
    best_epoch = -1
    history = []
    step_hist, loss_hist = [], []
    epoch_train_bce, epoch_val_bce = [], []

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        running_loss, running_auc, n_seen = 0.0, 0.0, 0
        step = 0

        for batch in train_loader:
            seq_cat = batch["seq_cat"].to(DEVICE)  # (B,T)
            seq_num = batch["seq_num"].to(DEVICE)  # (B,T,F)
            mask_t  = batch["mask_t"].to(DEVICE)   # (B,T)
            y_t     = batch["y_t"].to(DEVICE)      # (B,T)

            optimizer.zero_grad(set_to_none=True)

            # sanitize numerics + normalize on-device
            seq_num = torch.nan_to_num(seq_num, nan=0.0, posinf=1e6, neginf=-1e6).to(torch.float32)
            seq_num = (seq_num - mu_t) / sd_t

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(seq_cat=seq_cat, seq_num=seq_num)
                    loss = criterion(logits, y_t, mask_t)
            else:
                logits, _ = model(seq_cat=seq_cat, seq_num=seq_num)
                loss = criterion(logits, y_t, mask_t)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer); scaler.update()
            scheduler.step()

            bs = seq_cat.size(0)
            running_loss += float(loss.item()) * bs
            running_auc  += masked_pr_auc(logits, y_t, mask_t) * bs
            n_seen += bs
            step += 1

            # record every ~few steps
            if step % max(1, steps_per_epoch // 10) == 0:
                step_hist.append(len(step_hist) + 1)
                loss_hist.append(running_loss / max(n_seen, 1))

        tr_loss = running_loss / max(n_seen, 1)
        tr_auc  = running_auc  / max(n_seen, 1)

        # ---- validate (rich eval) ----
        eval_out = evaluate_full_epoch(model, val_loader, DEVICE, criterion, mu_t=mu_t, sd_t=sd_t)
        vl_loss = eval_out["val_bce"]
        vl_auc  = eval_out["auprc"]
        epoch_train_bce.append(tr_loss)
        epoch_val_bce.append(vl_loss)

        row = {
            "epoch": epoch, "train_loss": tr_loss, "train_pr_auc": tr_auc,
            "val_loss": vl_loss, "val_pr_auc": vl_auc, "auroc": eval_out["auroc"],
            "f1": eval_out["f1"], "precision": eval_out["precision"], "recall": eval_out["recall"],
            "best_threshold": eval_out["best_thr"], "n_pos": eval_out["n_pos"], "n_neg": eval_out["n_neg"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        # console
        print(f"[epoch {epoch:02d}] "
              f"train loss={tr_loss:.4f} prAUC={tr_auc:.4f} | "
              f"val loss={vl_loss:.4f} prAUC={vl_auc:.4f} ROC-AUC={eval_out['auroc']:.4f} "
              f"F1={eval_out['f1']:.3f} P={eval_out['precision']:.3f} R={eval_out['recall']:.3f} "
              f"@thr={eval_out['best_thr']:.3f}")

        # append to metrics CSV
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{vl_loss:.6f},{vl_auc:.6f},{eval_out['auroc']:.6f},"
                    f"{eval_out['f1']:.6f},{eval_out['precision']:.6f},{eval_out['recall']:.6f},"
                    f"{eval_out['best_thr']:.6f},{eval_out['n_pos']},{eval_out['n_neg']}\n")

        # update plots each epoch
        plots_dir = os.path.join(ARTIFACTS_DIR, "plots")
        plot_train_loss(step_hist, loss_hist, os.path.join(plots_dir, "train_loss_curve.pdf"))
        plot_epoch_loss(list(range(1, len(epoch_train_bce)+1)), epoch_train_bce, epoch_val_bce,
                        os.path.join(plots_dir, "epoch_loss_curve.pdf"))

        labels_np, probs_np = eval_out["labels"], eval_out["probs"]
        if labels_np.size > 0 and len(np.unique(labels_np)) >= 2:
            plot_pr(labels_np, probs_np, os.path.join(plots_dir, "pr_curve.pdf"))
            plot_roc(labels_np, probs_np, os.path.join(plots_dir, "roc_curve.pdf"))
            plot_calibration(labels_np, probs_np, os.path.join(plots_dir, "calibration.pdf"))
            plot_histograms(labels_np, probs_np, os.path.join(plots_dir, "score_histograms.pdf"))
            plot_confusion(labels_np, probs_np, eval_out["best_thr"], os.path.join(plots_dir, "confusion_matrix.pdf"))
            plot_threshold_curves(labels_np, probs_np, os.path.join(plots_dir, "threshold_curves.pdf"))

        # save best checkpoint on PR-AUC
        if vl_auc > best_auc:
            best_auc, best_epoch = vl_auc, epoch
            torch.save({
                "state_dict": model.state_dict(),
                "teacher_cfg_json": cfg.to_json(),
                "trainer_cfg": {
                    "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "max_grad_norm": MAX_GRAD_NORM,
                    "patience": PATIENCE, "device": DEVICE, "amp": AMP, "max_len": MAX_LEN,
                    "cat_col": CAT_COL, "case_col": CASE_COL, "prefix_col": PREFIX_COL,
                    "label_col": LABEL_COL, "mask_col": MASK_COL, "num_cols": NUM_COLS,
                    "use_causal": USE_CAUSAL, "cat_emb_dim": CAT_EMB_DIM,
                },
                "val_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                for k, v in eval_out.items() if k not in ("labels","probs")}
            }, os.path.join(ARTIFACTS_DIR, "teacher_model_best.pt"))

        # early stopping
        if epoch - best_epoch >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val PR-AUC={best_auc:.4f} @ {best_epoch})")
            break

    # ========= 8) SAVE LOGS ===================================================

    with open(os.path.join(ARTIFACTS_DIR, "teacher_train_log.jsonl"), "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    # summary json
    summary = {
        "best_val_pr_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "final_epochs": int(len(history)),
        "artifacts_dir": ARTIFACTS_DIR,
        "plots": ["plots/train_loss_curve.pdf","plots/epoch_loss_curve.pdf","plots/pr_curve.pdf",
                  "plots/roc_curve.pdf","plots/calibration.pdf","plots/score_histograms.pdf",
                  "plots/confusion_matrix.pdf","plots/threshold_curves.pdf"],
        "metrics_csv": "metrics_epoch.csv",
        "best_checkpoint": "teacher_model_best.pt",
        "keys": ["teacher_train_keys.parquet","teacher_val_keys.parquet"],
        "normalization_stats": "num_feature_norm_stats.npz",
    }
    with open(os.path.join(ARTIFACTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Best masked PR-AUC={best_auc:.4f} @ epoch {best_epoch}")
    print(f"Saved artifacts under: {ARTIFACTS_DIR}")
    print(f"- metrics CSV: {metrics_csv}")
    print(f"- plots: {os.path.join(ARTIFACTS_DIR, 'plots')}")
    print(f"- best ckpt: {os.path.join(ARTIFACTS_DIR, 'teacher_model_best.pt')}")
    print(f"- normalization stats: {os.path.join(ARTIFACTS_DIR, 'num_feature_norm_stats.npz')}")

if __name__ == "__main__":
    main()

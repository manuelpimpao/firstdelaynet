# teacher_model.py
"""
Teacher sequence model.

Small Transformer with masked binary loss, training utilities, metric exports,
and PDF plots.

Features
--------
- Numeric sanitization inside forward() and extract_step_repr().
- Optional output-bias initialization from a prior positive rate.
- Optional logit clipping for downstream stability.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union
import json
import math
import time
import csv
import pathlib
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TeacherConfig:
    num_cats: Dict[str, int]
    num_num_features: int
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 384
    dropout: float = 0.1
    max_len: int = 2048
    use_causal: bool = False
    cat_emb_dim: int = 64
    pos_weight: Optional[float] = None
    verbose: bool = True
    log_every: int = 100
    prior_pos_rate: Optional[float] = None
    norm_num_in_forward: bool = True
    num_clip_value: float = 1e6
    logit_clip: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "TeacherConfig":
        return TeacherConfig(**json.loads(s))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TeacherModel(nn.Module):
    def __init__(self, cfg: TeacherConfig):
        super().__init__()

        self.cfg = cfg

        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=cfg.cat_emb_dim,
                    padding_idx=0,
                )
                for name, vocab_size in cfg.num_cats.items()
            }
        )

        cat_concat_dim = cfg.cat_emb_dim * max(1, len(cfg.num_cats))

        self.cat_proj = nn.Linear(cat_concat_dim, cfg.d_model)

        self.has_num = int(cfg.num_num_features) > 0
        self.num_proj = nn.Linear(cfg.num_num_features if self.has_num else 1, cfg.d_model)

        self.fuse = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.posenc = PositionalEncoding(cfg.d_model, cfg.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
        )

        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

        self._verbose = bool(cfg.verbose)
        self._log_every = max(1, int(cfg.log_every))
        self._fwd_calls = 0

        self._init_weights()

        if cfg.prior_pos_rate is not None:
            p = float(max(1e-6, min(1 - 1e-6, cfg.prior_pos_rate)))
            bias = math.log(p / (1.0 - p))

            with torch.no_grad():
                self.head[-1].bias.fill_(bias)

        if self._verbose:
            print(
                f"[TeacherModel] params={self.count_params():,}, "
                f"d_model={cfg.d_model}, layers={cfg.n_layers}, "
                f"heads={cfg.n_heads}, num_num_features={cfg.num_num_features}, "
                f"categorical_channels={list(cfg.num_cats.keys())}"
            )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)

                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _should_log(self) -> bool:
        if not self._verbose:
            return False

        self._fwd_calls += 1
        return self._fwd_calls == 1 or self._fwd_calls % self._log_every == 0

    def _timing_ctx(self, device):
        class _Timer:
            def __init__(self, device):
                self.device = device
                self.use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

                if self.use_cuda:
                    self.start_event = torch.cuda.Event(enable_timing=True)
                    self.end_event = torch.cuda.Event(enable_timing=True)
                else:
                    self.t0 = None
                    self.t1 = None

            def __enter__(self):
                if self.use_cuda:
                    torch.cuda.synchronize()
                    self.start_event.record()
                else:
                    self.t0 = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                if self.use_cuda:
                    self.end_event.record()
                    self.end_event.synchronize()
                else:
                    self.t1 = time.perf_counter()

            def ms(self):
                if self.use_cuda:
                    return float(self.start_event.elapsed_time(self.end_event))
                return float((self.t1 - self.t0) * 1000.0)

        return _Timer(device)

    def _normalize_seq_cat(
        self,
        seq_cat: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(seq_cat, dict):
            return seq_cat

        if len(self.cfg.num_cats) != 1:
            raise ValueError(
                "seq_cat was provided as one tensor, but the config defines "
                "multiple categorical channels."
            )

        name = next(iter(self.cfg.num_cats.keys()))
        return {name: seq_cat}

    def _build_padding_mask(self, cat_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        masks = [x.eq(0) for x in cat_dict.values()]

        padding_mask = masks[0]

        for mask in masks[1:]:
            padding_mask = padding_mask & mask

        return padding_mask

    def _causal_attn_mask(self, sequence_length: int, device) -> torch.Tensor:
        return torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _expand_labels_to_bt(self, labels: torch.Tensor, sequence_length: int) -> torch.Tensor:
        if labels.dim() == 1:
            labels = labels.view(-1, 1).expand(-1, sequence_length)
        elif labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.expand(-1, sequence_length)
        elif labels.dim() == 2 and labels.size(1) == sequence_length:
            pass
        else:
            raise ValueError(
                "labels must have shape (B,), (B,1), or (B,T); "
                f"got {tuple(labels.shape)}"
            )

        return labels

    def _sanitize_seq_num(self, seq_num: torch.Tensor) -> torch.Tensor:
        if not self.cfg.norm_num_in_forward:
            return seq_num

        if seq_num.dtype in (torch.float16, torch.bfloat16):
            seq_num = seq_num.float()

        seq_num = torch.nan_to_num(
            seq_num,
            nan=0.0,
            posinf=self.cfg.num_clip_value,
            neginf=-self.cfg.num_clip_value,
        )

        if self.cfg.num_clip_value is not None and self.cfg.num_clip_value > 0:
            seq_num = torch.clamp(
                seq_num,
                -self.cfg.num_clip_value,
                self.cfg.num_clip_value,
            )

        return seq_num

    def forward(
        self,
        seq_cat: Union[torch.Tensor, Dict[str, torch.LongTensor]],
        seq_num: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        extra_mask: Optional[torch.Tensor] = None,
        return_reprs: bool = False,
    ):
        seq_num = self._sanitize_seq_num(seq_num)

        batch_size, sequence_length, n_features = seq_num.shape

        if sequence_length > self.cfg.max_len:
            raise ValueError(
                f"Sequence length T={sequence_length} exceeds max_len={self.cfg.max_len}."
            )

        do_log = self._should_log()
        cat_dict = self._normalize_seq_cat(seq_cat)

        if do_log:
            print(
                f"[TeacherModel.forward] B={batch_size}, "
                f"T={sequence_length}, F={n_features}, "
                f"categorical_channels={list(cat_dict.keys())}"
            )

        with self._timing_ctx(seq_num.device) as t_cat:
            cat_embs = []

            for name, embedding in self.cat_embeddings.items():
                cat_embs.append(embedding(cat_dict[name]))

            if cat_embs:
                cat_cat = torch.cat(cat_embs, dim=-1)
            else:
                cat_cat = torch.zeros(
                    batch_size,
                    sequence_length,
                    self.cfg.cat_emb_dim,
                    device=seq_num.device,
                )

            cat_h = self.cat_proj(cat_cat)

        with self._timing_ctx(seq_num.device) as t_num:
            if self.has_num:
                num_h = self.num_proj(seq_num)
            else:
                num_h = self.num_proj(
                    torch.zeros(
                        batch_size,
                        sequence_length,
                        1,
                        device=seq_num.device,
                    )
                )

        with self._timing_ctx(seq_num.device) as t_fuse:
            h = torch.cat([cat_h, num_h], dim=-1)
            h = self.fuse(h)
            h = self.posenc(h)

            key_padding_mask = self._build_padding_mask(cat_dict).to(h.device)
            attn_mask = (
                self._causal_attn_mask(sequence_length, h.device)
                if self.cfg.use_causal
                else None
            )

        with self._timing_ctx(seq_num.device) as t_enc:
            hidden = self.encoder(
                h,
                mask=attn_mask,
                src_key_padding_mask=key_padding_mask,
            )

        with self._timing_ctx(seq_num.device) as t_head:
            logits = self.head(hidden).squeeze(-1)

            if self.cfg.logit_clip is not None and self.cfg.logit_clip > 0:
                logits = torch.clamp(
                    logits,
                    -self.cfg.logit_clip,
                    self.cfg.logit_clip,
                )

        loss = None

        if labels is not None and mask_t is not None:
            labels = labels.float()
            mask = mask_t.float()

            if extra_mask is not None:
                mask = mask * extra_mask.float().to(mask.device)

            labels = self._expand_labels_to_bt(labels, sequence_length).to(logits.device)
            mask = mask.to(logits.device)

            with self._timing_ctx(seq_num.device) as t_loss:
                pos_weight = None

                if self.cfg.pos_weight is not None:
                    pos_weight = torch.tensor([self.cfg.pos_weight], device=logits.device)

                bce = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="none",
                    pos_weight=pos_weight,
                )

                masked = bce * mask
                denom = mask.sum().clamp_min(1.0)
                loss = masked.sum() / denom

        if do_log:
            masked_count = int(mask_t.sum().item()) if mask_t is not None else 0
            mean_per_seq = (
                float(mask_t.sum(dim=1).float().mean().item())
                if mask_t is not None
                else 0.0
            )

            print(
                "[TeacherModel.forward] "
                f"mask_ones={masked_count}, "
                f"mean_per_sequence={mean_per_seq:.2f}, "
                f"t_cat={t_cat.ms():.1f}ms, "
                f"t_num={t_num.ms():.1f}ms, "
                f"t_fuse_pos={t_fuse.ms():.1f}ms, "
                f"t_enc={t_enc.ms():.1f}ms, "
                f"t_head={t_head.ms():.1f}ms"
                + (
                    f", t_loss={t_loss.ms():.1f}ms"
                    if labels is not None and mask_t is not None
                    else ""
                )
            )

        if return_reprs:
            return logits, hidden

        return logits, loss

    @torch.no_grad()
    def extract_step_repr(
        self,
        seq_cat: Union[torch.Tensor, Dict[str, torch.LongTensor]],
        seq_num: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_num = self._sanitize_seq_num(seq_num)

        batch_size, sequence_length, _ = seq_num.shape

        if sequence_length > self.cfg.max_len:
            raise ValueError(
                f"Sequence length T={sequence_length} exceeds max_len={self.cfg.max_len}."
            )

        do_log = self._should_log()
        cat_dict = self._normalize_seq_cat(seq_cat)

        with self._timing_ctx(seq_num.device) as t_all:
            cat_embs = []

            for name, embedding in self.cat_embeddings.items():
                cat_embs.append(embedding(cat_dict[name]))

            if cat_embs:
                cat_cat = torch.cat(cat_embs, dim=-1)
            else:
                cat_cat = torch.zeros(
                    batch_size,
                    sequence_length,
                    self.cfg.cat_emb_dim,
                    device=seq_num.device,
                )

            cat_h = self.cat_proj(cat_cat)
            num_h = (
                self.num_proj(seq_num)
                if self.has_num
                else self.num_proj(
                    torch.zeros(
                        batch_size,
                        sequence_length,
                        1,
                        device=seq_num.device,
                    )
                )
            )

            h = self.fuse(torch.cat([cat_h, num_h], dim=-1))
            h = self.posenc(h)

            key_padding_mask = self._build_padding_mask(cat_dict).to(h.device)
            attn_mask = (
                self._causal_attn_mask(sequence_length, h.device)
                if self.cfg.use_causal
                else None
            )

            hidden = self.encoder(
                h,
                mask=attn_mask,
                src_key_padding_mask=key_padding_mask,
            )

            if mask_t is not None:
                mask = mask_t.bool().to(hidden.device)
                idx = torch.arange(sequence_length, device=hidden.device).unsqueeze(0).expand(
                    batch_size,
                    sequence_length,
                )
                idx = torch.where(mask, idx, torch.full_like(idx, -1))
                pos = idx.max(dim=1).values
            else:
                pad = key_padding_mask
                idx = torch.arange(sequence_length, device=hidden.device).unsqueeze(0).expand(
                    batch_size,
                    sequence_length,
                )
                idx = torch.where(~pad, idx, torch.full_like(idx, -1))
                pos = idx.max(dim=1).values

            pos = torch.where(pos >= 0, pos, torch.full_like(pos, sequence_length - 1))
            representation = hidden[torch.arange(batch_size, device=hidden.device), pos]

        if do_log:
            print(
                f"[TeacherModel.extract_step_repr] "
                f"took {t_all.ms():.1f}ms for B={batch_size}, T={sequence_length}"
            )

        return representation

    def count_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def save_config(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.cfg.to_json())

    @staticmethod
    def load_config(path: str) -> "TeacherConfig":
        with open(path, "r", encoding="utf-8") as f:
            return TeacherConfig.from_json(f.read())


@dataclass
class TrainingConfig:
    out_dir: str
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 50
    early_stop_patience: int = 3
    early_stop_metric: str = "auprc"
    save_every_epoch: bool = True
    seed: int = 1337
    plot_max_points: int = 500_000


class TrainingRecorder:
    def __init__(self, out_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.base = pathlib.Path(out_dir) / timestamp
        self.base.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.base / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        self.plots_dir = self.base / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.metrics_csv = self.base / "metrics_epoch.csv"

        with open(self.metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_bce",
                    "val_bce",
                    "auroc",
                    "auprc",
                    "f1",
                    "precision",
                    "recall",
                    "best_threshold",
                    "positives",
                    "negatives",
                    "mask_ones",
                    "brier",
                    "ece",
                ]
            )

        self.last_metrics = None

    def write_epoch_row(self, epoch: int, metrics: dict):
        with open(self.metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{metrics['train_bce']:.6f}",
                    f"{metrics['val_bce']:.6f}",
                    f"{metrics['auroc']:.6f}",
                    f"{metrics['auprc']:.6f}",
                    f"{metrics['f1']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['best_threshold']:.6f}",
                    int(metrics["n_pos"]),
                    int(metrics["n_neg"]),
                    int(metrics["mask_ones"]),
                    f"{metrics.get('brier', float('nan')):.6f}",
                    f"{metrics.get('ece', float('nan')):.6f}",
                ]
            )

        self.last_metrics = metrics

    def write_json(self, name: str, obj: dict):
        path = self.base / f"{name}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)


from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


def _ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 20) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins) - 1

    ece = 0.0

    for bin_id in range(n_bins):
        mask = idx == bin_id

        if not np.any(mask):
            continue

        confidence = probs[mask].mean()
        accuracy = labels[mask].mean()
        weight = mask.mean()

        ece += weight * abs(accuracy - confidence)

    return float(ece)


@torch.no_grad()
def evaluate_teacher(model: TeacherModel, loader, device, plot_max_points=500_000):
    model.eval()

    losses = []
    all_logits = []
    all_labels = []
    all_mask_counts = []

    for batch in loader:
        seq_cat = batch["seq_cat"]
        seq_num = batch["seq_num"].to(device)
        labels = batch["labels"].to(device)
        mask_t = batch["mask_t"].to(device)

        if isinstance(seq_cat, dict):
            seq_cat = {key: value.to(device) for key, value in seq_cat.items()}
        else:
            seq_cat = seq_cat.to(device)

        logits, loss = model(seq_cat, seq_num, mask_t=mask_t, labels=labels)
        losses.append(loss.item() if loss is not None else 0.0)

        mask = mask_t.bool()
        logits_masked = logits[mask].detach().float().cpu()
        labels_expanded = model._expand_labels_to_bt(labels.float(), logits.size(1))
        labels_masked = labels_expanded[mask].detach().float().cpu()

        all_logits.append(logits_masked)
        all_labels.append(labels_masked)
        all_mask_counts.append(mask.sum().item())

    val_bce = float(np.mean(losses)) if losses else float("nan")

    logits_np = torch.cat(all_logits).numpy() if all_logits else np.array([])
    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    mask_ones = int(np.sum(all_mask_counts))

    if logits_np.size == 0:
        return {
            "val_bce": val_bce,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "best_threshold": 0.5,
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n_pos": 0,
            "n_neg": 0,
            "mask_ones": mask_ones,
            "logits": logits_np,
            "labels": labels_np,
            "pr_prec": np.array([]),
            "pr_rec": np.array([]),
            "pr_thr": np.array([]),
            "roc_fpr": np.array([]),
            "roc_tpr": np.array([]),
            "brier": float("nan"),
            "ece": float("nan"),
        }

    probs = 1.0 / (1.0 + np.exp(-logits_np))

    n_pos = int(labels_np.sum())
    n_neg = int(labels_np.size - n_pos)

    try:
        auroc = roc_auc_score(labels_np, probs)
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(labels_np, probs)
    except ValueError:
        auprc = float("nan")

    precision, recall, thresholds = precision_recall_curve(labels_np, probs)

    if len(np.unique(labels_np)) >= 2:
        fpr, tpr, _ = roc_curve(labels_np, probs)
    else:
        fpr, tpr = np.array([]), np.array([])

    threshold_full = np.append(thresholds, 1.0)
    f1_values = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )

    best_idx = int(np.nanargmax(f1_values))
    best_threshold = float(threshold_full[best_idx])

    predictions = (probs >= best_threshold).astype(np.uint8)

    f1 = f1_score(labels_np, predictions, zero_division=0)
    precision_value = precision_score(labels_np, predictions, zero_division=0)
    recall_value = recall_score(labels_np, predictions, zero_division=0)

    try:
        brier = brier_score_loss(labels_np, probs)
    except Exception:
        brier = float("nan")

    ece = _ece(labels_np, probs, n_bins=20)

    return {
        "val_bce": val_bce,
        "auroc": float(auroc),
        "auprc": float(auprc),
        "best_threshold": best_threshold,
        "f1": float(f1),
        "precision": float(precision_value),
        "recall": float(recall_value),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "mask_ones": mask_ones,
        "logits": probs,
        "labels": labels_np,
        "pr_prec": precision,
        "pr_rec": recall,
        "pr_thr": thresholds,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "brier": float(brier),
        "ece": float(ece),
    }


def _savefig_pdf(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_loss_curve(train_steps, train_losses, out_pdf: pathlib.Path):
    plt.figure()
    plt.plot(train_steps, train_losses, linewidth=1.5, label="Train BCE")
    plt.xlabel("Step")
    plt.ylabel("Train BCE")
    plt.title("Training Loss")
    plt.legend()
    _savefig_pdf(out_pdf)


def plot_epoch_curve(epochs, train_bce, val_bce, out_pdf: pathlib.Path):
    plt.figure()
    plt.plot(epochs, train_bce, linewidth=1.5, label="Train")
    plt.plot(epochs, val_bce, linewidth=1.5, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Masked BCE")
    plt.title("Epoch Loss")
    plt.legend()
    _savefig_pdf(out_pdf)


def plot_roc(fpr, tpr, auroc, out_pdf: pathlib.Path):
    if fpr.size == 0 or tpr.size == 0:
        return

    plt.figure()
    plt.plot(fpr, tpr, linewidth=1.8, label=f"ROC (AUROC={auroc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    _savefig_pdf(out_pdf)


def plot_pr(precision, recall, auprc, out_pdf: pathlib.Path):
    plt.figure()
    plt.plot(recall, precision, linewidth=1.8, label=f"PR (AUPRC={auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    _savefig_pdf(out_pdf)


def plot_calibration(labels, probs, out_pdf: pathlib.Path, n_bins=20, brier=None, ece=None):
    if probs.size == 0:
        return

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins) - 1

    accuracies = []
    confidences = []

    for bin_id in range(n_bins):
        mask = idx == bin_id

        if mask.sum() == 0:
            continue

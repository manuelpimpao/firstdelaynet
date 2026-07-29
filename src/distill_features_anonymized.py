# -*- coding: utf-8 -*-
"""
distill_features.py — OOM-safe teacher-feature distillation

Provides
--------
- load_teacher_artifacts(...)
- distill_split(...)
- distill_tau_split(...)

Purpose
-------
Computes teacher-model logits and optional PCA-compressed embeddings for
event-log prefixes or time-anchored snapshots.

Notes
-----
- Numeric channels are sanitized and normalized using saved teacher statistics.
- Batches are length-sorted and token-budgeted to reduce GPU memory usage.
- PCA is fit on the training split and reused for validation/test splits.
- If the input snapshot DataFrame contains "__row_id__", it is preserved to allow
  safe one-to-one merging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import IncrementalPCA, PCA

from teacher_model_anonymized import TeacherConfig, TeacherModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_TOKENS_TARGET = 250_000
MIN_BATCH_SIZE = 16
MAX_BATCH_SIZE = 512

LOGIT_CLIP: Optional[float] = None

torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


_MU_T: Optional[torch.Tensor] = None
_SD_T: Optional[torch.Tensor] = None


def _row_get(row, key: str, pos: Optional[int] = None):
    if isinstance(row, dict):
        return row[key]
    if isinstance(row, pd.Series):
        return row[key]
    if hasattr(row, key):
        return getattr(row, key)
    if pos is not None:
        return row[pos]
    return row[key]


def load_teacher_artifacts(artifact_dir: str | Path = "model_artifacts"):
    artifact_dir = Path(artifact_dir)

    vocab_path = artifact_dir / "teacher_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocabulary file: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    dataset_cfg_path = artifact_dir / "teacher_config.json"

    if dataset_cfg_path.exists():
        dataset_cfg = json.loads(dataset_cfg_path.read_text(encoding="utf-8"))
    else:
        dataset_cfg = {
            "cat_col": "concept:name",
            "num_cols": [],
            "case_col": "case:concept:name",
            "prefix_col": "prefix_index",
            "mask_col": "mask_transport",
            "max_len": 100,
            "time_col": "time:timestamp",
        }

    cat_col = dataset_cfg.get("cat_col", "concept:name")
    num_cols = list(dataset_cfg.get("num_cols", []))
    max_len = int(dataset_cfg.get("max_len", 100))

    checkpoint_candidates = [
        artifact_dir / "teacher_model_best.pt",
        artifact_dir / "best_auprc.pt",
        artifact_dir / "best_bce.pt",
        artifact_dir / "teacher_model.pt",
    ]

    checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), None)

    if checkpoint_path is None:
        tried = ", ".join(str(p) for p in checkpoint_candidates)
        raise FileNotFoundError(f"No teacher checkpoint found. Tried: {tried}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_cfg_json = None

    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("teacher_cfg_json"), str):
            model_cfg_json = checkpoint["teacher_cfg_json"]
        elif isinstance(checkpoint.get("cfg"), str):
            model_cfg_json = checkpoint["cfg"]

    if model_cfg_json is None:
        cfg_file = artifact_dir / "teacher_model_config.json"
        if cfg_file.exists():
            model_cfg_json = cfg_file.read_text(encoding="utf-8")

    if model_cfg_json is None:
        model_cfg_json = json.dumps(
            {
                "num_cats": {cat_col: len(vocab)},
                "num_num_features": len(num_cols),
                "d_model": 192,
                "n_heads": 4,
                "n_layers": 2,
                "d_ff": 384,
                "dropout": 0.1,
                "max_len": max_len,
                "use_causal": False,
                "cat_emb_dim": 64,
                "pos_weight": None,
                "verbose": False,
                "log_every": 100,
            }
        )

    model_cfg = TeacherConfig.from_json(model_cfg_json)

    if list(model_cfg.num_cats.keys()) != [cat_col]:
        model_cfg.num_cats = {cat_col: len(vocab)}

    model = TeacherModel(model_cfg)

    if isinstance(checkpoint, dict) and ("state_dict" in checkpoint or "model_state" in checkpoint):
        state = checkpoint.get("state_dict", checkpoint.get("model_state"))
    else:
        state = checkpoint

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        print(
            "[load_teacher_artifacts] Warning:",
            f"missing keys: {missing} | unexpected keys: {unexpected}",
        )

    model.to(DEVICE)
    model.eval()

    dataset_cfg.setdefault("max_len", getattr(model_cfg, "max_len", max_len))
    dataset_cfg.setdefault("time_col", "time:timestamp")

    _load_norm_stats(artifact_dir, expect_num_cols=len(num_cols))

    return model, vocab, model_cfg, dataset_cfg, artifact_dir


def _load_norm_stats(artifact_dir: Path, expect_num_cols: Optional[int] = None):
    global _MU_T, _SD_T

    stats_path = artifact_dir / "num_feature_norm_stats.npz"

    if not stats_path.exists():
        _MU_T, _SD_T = None, None
        print(f"[distill] Normalization stats not found at {stats_path}. Using raw numeric features.")
        return

    stats = np.load(stats_path, allow_pickle=True)

    mu = stats["mean"].astype(np.float32)
    sd = stats["std"].astype(np.float32)

    if expect_num_cols is not None and int(mu.shape[-1]) != int(expect_num_cols):
        print(
            f"[distill] Warning: stats F={mu.shape[-1]} "
            f"but dataset_cfg.num_cols={expect_num_cols}. Proceeding."
        )

    sd = np.clip(sd, 1e-6, None)

    _MU_T = torch.from_numpy(mu).to(DEVICE)
    _SD_T = torch.from_numpy(sd).to(DEVICE)

    print(f"[distill] Loaded normalization stats from {stats_path}.")


def _prepare_events_fast(
    events_df: pd.DataFrame,
    *,
    cat_col: str,
    num_cols: List[str],
    case_col: str,
    time_col: Optional[str] = None,
    prefix_col: Optional[str] = None,
    vocab: Dict[str, int],
) -> Tuple[pd.DataFrame, Dict[Any, Tuple[int, int]]]:
    df = events_df.copy()

    unk = vocab.get("<unk>", 1)

    df[cat_col] = df[cat_col].astype("string")
    df["_cat_id"] = df[cat_col].map(vocab).fillna(unk).astype(np.int64)

    for col in num_cols:
        if col not in df.columns:
            df[col] = 0.0

    if num_cols:
        df[num_cols] = df[num_cols].astype(np.float32, copy=False)

    sort_keys = [case_col]

    if time_col is not None and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        sort_keys.append(time_col)
    elif prefix_col is not None and prefix_col in df.columns:
        df[prefix_col] = pd.to_numeric(df[prefix_col], errors="coerce").astype("Int64")
        sort_keys.append(prefix_col)

    df = df.sort_values(sort_keys, kind="stable").reset_index(drop=True)

    grouped = df.groupby(case_col, sort=False, observed=True)
    spans = {case_id: (idx.index[0], idx.index[-1] + 1) for case_id, idx in grouped}

    return df, spans


def _make_seq_builder_by_time_fast(
    df: pd.DataFrame,
    spans: Dict[Any, Tuple[int, int]],
    *,
    cat_id_col: str,
    num_cols: List[str],
    case_col: str,
    time_col: str,
    max_len: int,
):
    times = df[time_col].to_numpy()
    cats = df[cat_id_col].to_numpy()
    nums = df[num_cols].to_numpy(np.float32, copy=False) if num_cols else None

    def build(row):
        case_id = _row_get(row, case_col, pos=0)
        tau = pd.to_datetime(_row_get(row, "tau_ts", pos=1), utc=True)

        a, b = spans.get(case_id, (None, None))

        if a is None:
            raise KeyError(f"Case id {case_id!r} not found.")

        lo, hi = a, b

        while lo < hi:
            mid = (lo + hi) // 2
            if times[mid] <= tau:
                lo = mid + 1
            else:
                hi = mid

        end = lo

        if end <= a:
            raise ValueError(f"No events before or at tau_ts={tau} for case={case_id!r}.")

        start = max(a, end - max_len)

        cat_ids = cats[start:end]

        if num_cols:
            num_mat = nums[start:end, :]
        else:
            num_mat = np.empty((end - start, 0), dtype=np.float32)

        seq_len = end - start
        mask = np.zeros((seq_len,), dtype=np.float32)
        mask[-1] = 1.0
        masked_idx = seq_len - 1

        seq_cat = {cat_id_col: torch.from_numpy(cat_ids.copy())}
        seq_num = torch.from_numpy(num_mat.copy())
        mask_t = torch.from_numpy(mask.copy())

        return seq_cat, seq_num, mask_t, masked_idx

    return build


def _make_seq_builder_by_prefix_fast(
    df: pd.DataFrame,
    spans: Dict[Any, Tuple[int, int]],
    *,
    cat_id_col: str,
    num_cols: List[str],
    case_col: str,
    prefix_col: str,
    max_len: int,
):
    prefixes = df[prefix_col].to_numpy()
    cats = df[cat_id_col].to_numpy()
    nums = df[num_cols].to_numpy(np.float32, copy=False) if num_cols else None

    def build(row):
        case_id = _row_get(row, case_col, pos=0)
        prefix_index = int(_row_get(row, prefix_col, pos=1))

        a, b = spans.get(case_id, (None, None))

        if a is None:
            raise KeyError(f"Case id {case_id!r} not found.")

        lo, hi = a, b

        while lo < hi:
            mid = (lo + hi) // 2
            if prefixes[mid] <= prefix_index:
                lo = mid + 1
            else:
                hi = mid

        end = lo

        if end <= a:
            raise ValueError(
                f"No events before or at prefix_index={prefix_index} "
                f"for case={case_id!r}."
            )

        start = max(a, end - max_len)

        cat_ids = cats[start:end]

        if num_cols:
            num_mat = nums[start:end, :]
        else:
            num_mat = np.empty((end - start, 0), dtype=np.float32)

        seq_len = end - start
        mask = np.zeros((seq_len,), dtype=np.float32)
        mask[-1] = 1.0
        masked_idx = seq_len - 1

        seq_cat = {cat_id_col: torch.from_numpy(cat_ids.copy())}
        seq_num = torch.from_numpy(num_mat.copy())
        mask_t = torch.from_numpy(mask.copy())

        return seq_cat, seq_num, mask_t, masked_idx

    return build


def _adaptive_batch_indices(lengths: List[int]) -> List[Tuple[int, int]]:
    batches = []
    n = len(lengths)
    i = 0

    while i < n:
        j = i
        count = 0
        max_len = 0

        while j < n:
            length = lengths[j]
            max_len = max(max_len, length)
            prospective_tokens = (count + 1) * max_len

            if (
                (count + 1) >= MIN_BATCH_SIZE
                and (
                    prospective_tokens > BATCH_TOKENS_TARGET
                    or (count + 1) > MAX_BATCH_SIZE
                )
            ):
                break

            j += 1
            count += 1

        if j == i:
            j = min(i + 1, n)

        batches.append((i, j))
        i = j

    return batches


def _collate_batch(
    items: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, int]],
    *,
    src_key: str,
    dst_key: str,
    vocab: Dict[str, int],
):
    pad_id = vocab.get("<pad>", 0)

    cat_streams = []
    num_streams = []
    lengths = []
    masked_idx_list = []

    for seq_cat, seq_num, _, masked_idx in items:
        cat_streams.append(seq_cat[src_key])
        num_streams.append(seq_num)
        lengths.append(int(seq_num.shape[0]))
        masked_idx_list.append(int(masked_idx))

    batch_cat_cpu = pad_sequence(cat_streams, batch_first=True, padding_value=pad_id)
    batch_num_cpu = pad_sequence(num_streams, batch_first=True, padding_value=0.0)

    if DEVICE == "cuda":
        batch_cat = batch_cat_cpu.pin_memory().to(DEVICE, non_blocking=True)
        batch_num = batch_num_cpu.pin_memory().to(DEVICE, non_blocking=True)
    else:
        batch_cat = batch_cat_cpu.to(DEVICE)
        batch_num = batch_num_cpu.to(DEVICE)

    batch_size, max_time = batch_num.shape[0], batch_num.shape[1]

    key_padding_mask = torch.ones((batch_size, max_time), dtype=torch.bool, device=DEVICE)

    for i, length in enumerate(lengths):
        key_padding_mask[i, :length] = False

    masked_indices = torch.tensor(masked_idx_list, dtype=torch.long, device=DEVICE)

    return {dst_key: batch_cat}, batch_num, key_padding_mask, masked_indices


@torch.inference_mode()
def _iter_forward_batched(
    model: TeacherModel,
    rows: List[Any],
    build_sequence_fn,
    *,
    cat_src_key: str,
    cat_dst_key: str,
    vocab: Dict[str, int],
    use_autocast: bool = True,
):
    built = []
    lengths = []
    original_indices = []

    for i, row in enumerate(rows):
        seq_cat, seq_num, mask_t, masked_idx = build_sequence_fn(row)
        built.append((seq_cat, seq_num, mask_t, int(masked_idx)))
        lengths.append(int(seq_num.shape[0]))
        original_indices.append(i)

    if not built:
        return

    order = np.argsort(lengths, kind="stable")

    built = [built[i] for i in order]
    original_indices = [original_indices[i] for i in order]
    lengths = [lengths[i] for i in order]

    batches = _adaptive_batch_indices(lengths)

    def _forward_with_repr(cat, num):
        try:
            return model(seq_cat=cat, seq_num=num, return_reprs=True)
        except TypeError:
            logits_seq, _ = model(cat, num)
            return logits_seq, None

    autocast_cm = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if DEVICE == "cuda" and use_autocast
        else nullcontext()
    )

    with autocast_cm:
        for start, end in batches:
            items = built[start:end]
            idxs = np.array(original_indices[start:end], dtype=np.int64)

            batch_cat, batch_num, _, masked_indices = _collate_batch(
                items,
                src_key=cat_src_key,
                dst_key=cat_dst_key,
                vocab=vocab,
            )

            batch_num = torch.nan_to_num(
                batch_num,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            ).to(torch.float32)

            if _MU_T is not None and _SD_T is not None:
                batch_num = (batch_num - _MU_T) / _SD_T

            logits_seq, reprs_seq = _forward_with_repr(batch_cat, batch_num)

            batch_size = batch_num.shape[0]
            logits_b = np.empty((batch_size,), dtype=np.float32)

            if reprs_seq is None:
                time_steps = batch_num.shape[1]
                mask_onehot = torch.zeros(
                    (batch_size, time_steps),
                    dtype=torch.float32,
                    device=DEVICE,
                )

                for bi in range(batch_size):
                    mask_onehot[bi, int(masked_indices[bi])] = 1.0

                reprs = model.extract_step_repr(
                    seq_cat=batch_cat,
                    seq_num=batch_num,
                    mask_t=mask_onehot,
                )

                emb_cpu = reprs.detach().float().cpu().numpy()

                for bi in range(batch_size):
                    mi = int(masked_indices[bi])
                    value = float(logits_seq[bi, mi].detach().float().cpu())

                    if LOGIT_CLIP is not None and LOGIT_CLIP > 0:
                        value = float(np.clip(value, -LOGIT_CLIP, LOGIT_CLIP))

                    logits_b[bi] = value

                yield idxs, logits_b, emb_cpu

            else:
                d_model = int(reprs_seq.shape[-1])
                emb_cpu = np.empty((batch_size, d_model), dtype=np.float32)

                for bi in range(batch_size):
                    mi = int(masked_indices[bi])
                    value = float(logits_seq[bi, mi].detach().float().cpu())

                    if LOGIT_CLIP is not None and LOGIT_CLIP > 0:
                        value = float(np.clip(value, -LOGIT_CLIP, LOGIT_CLIP))

                    logits_b[bi] = value
                    emb_cpu[bi, :] = reprs_seq[bi, mi].detach().float().cpu().numpy()

                yield idxs, logits_b, emb_cpu


def _maybe_memmap(path: Optional[Path], shape: Tuple[int, ...], dtype=np.float32):
    if path is None:
        return np.empty(shape, dtype=dtype), False

    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape), True


def _slice_events_minimal(
    events_df: pd.DataFrame,
    case_col: str,
    time_col: str,
    cat_col: str,
    num_cols: List[str],
) -> pd.DataFrame:
    required_cols = []

    for col in [case_col, time_col, cat_col]:
        if col in events_df.columns:
            required_cols.append(col)

    for col in num_cols:
        if col in events_df.columns:
            required_cols.append(col)

    df = events_df.loc[:, required_cols].copy()

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32, copy=False)

    return df


def distill_split(
    df_pref: pd.DataFrame,
    events_df: pd.DataFrame,
    model: TeacherModel,
    vocab: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    *,
    case_id_col: str = "case:concept:name",
    prefix_index_col: str = "prefix_index",
    pca_components: int = 0,
    pca: Optional[PCA] = None,
    memmap_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Optional[PCA]]:
    if df_pref.empty:
        out = (
            df_pref[[case_id_col, prefix_index_col]].copy()
            if all(c in df_pref.columns for c in [case_id_col, prefix_index_col])
            else pd.DataFrame({case_id_col: [], prefix_index_col: []})
        )
        out["teacher_logit"] = np.array([], dtype=np.float32)
        return out, pca if pca_components > 0 else None

    work = df_pref.copy()

    if prefix_index_col not in work.columns:
        work[prefix_index_col] = (
            work.groupby(case_id_col, sort=False)
            .cumcount()
            .astype(np.int32)
        )

    cat_col = dataset_cfg.get("cat_col", "concept:name")
    num_cols = list(dataset_cfg.get("num_cols", []))
    case_col = dataset_cfg.get("case_col", case_id_col)
    prefix_col = dataset_cfg.get("prefix_col", prefix_index_col)
    max_len = int(dataset_cfg.get("max_len", 100))

    events_fast, spans = _prepare_events_fast(
        events_df,
        cat_col=cat_col,
        num_cols=num_cols,
        case_col=case_col,
        prefix_col=prefix_col,
        vocab=vocab,
    )

    build_seq = _make_seq_builder_by_prefix_fast(
        events_fast,
        spans,
        cat_id_col="_cat_id",
        num_cols=num_cols,
        case_col=case_col,
        prefix_col=prefix_col,
        max_len=max_len,
    )

    work2 = work

    if case_col not in work2.columns and case_id_col in work2.columns:
        work2 = work2.rename(columns={case_id_col: case_col})

    if prefix_col not in work2.columns and prefix_index_col in work2.columns:
        work2 = work2.rename(columns={prefix_index_col: prefix_col})

    rows = work2[[case_col, prefix_col]].to_dict("records")

    cat_keys = list(model.cfg.num_cats.keys())

    if not cat_keys:
        raise RuntimeError("TeacherConfig.num_cats is empty.")

    model_cat_key = cat_keys[0]

    n_rows = len(rows)
    mmap_dir = Path(memmap_dir) if memmap_dir else None

    logits_buf, _ = _maybe_memmap(
        mmap_dir / "logits_prefix.memmap" if mmap_dir else None,
        (n_rows,),
        np.float32,
    )

    if pca_components > 0 and pca is None:
        ipca = IncrementalPCA(
            n_components=pca_components,
            batch_size=max(1024, MIN_BATCH_SIZE),
        )

        for _, _, emb_b in _iter_forward_batched(
            model,
            rows,
            build_seq,
            cat_src_key="_cat_id",
            cat_dst_key=model_cat_key,
            vocab=vocab,
        ):
            if emb_b.size:
                ipca.partial_fit(np.nan_to_num(emb_b, copy=False))

        z_buf, _ = _maybe_memmap(
            mmap_dir / "embeddings_prefix.memmap" if mmap_dir else None,
            (n_rows, pca_components),
            np.float32,
        )

        for idxs, log_b, emb_b in _iter_forward_batched(
            model,
            rows,
            build_seq,
            cat_src_key="_cat_id",
            cat_dst_key=model_cat_key,
            vocab=vocab,
        ):
            logits_buf[idxs] = log_b

            if emb_b.size:
                z_buf[idxs, :] = ipca.transform(
                    np.nan_to_num(emb_b, copy=False)
                ).astype(np.float32, copy=False)
            else:
                z_buf[idxs, :] = 0.0

        out = pd.DataFrame(
            {
                case_id_col: work2[case_col].to_numpy(),
                prefix_index_col: work2[prefix_col].to_numpy(),
                "teacher_logit": np.asarray(logits_buf, dtype=np.float32),
            }
        )

        for i in range(pca_components):
            out[f"teacher_embed_pca_{i + 1}"] = np.asarray(z_buf[:, i], dtype=np.float32)

        return out, ipca

    if pca_components > 0 and pca is not None:
        z_buf, _ = _maybe_memmap(
            mmap_dir / "embeddings_prefix.memmap" if mmap_dir else None,
            (n_rows, pca.n_components_),
            np.float32,
        )

        for idxs, log_b, emb_b in _iter_forward_batched(
            model,
            rows,
            build_seq,
            cat_src_key="_cat_id",
            cat_dst_key=model_cat_key,
            vocab=vocab,
        ):
            logits_buf[idxs] = log_b

            if emb_b.size:
                z_buf[idxs, :] = pca.transform(
                    np.nan_to_num(emb_b, copy=False)
                ).astype(np.float32, copy=False)
            else:
                z_buf[idxs, :] = 0.0

        out = pd.DataFrame(
            {
                case_id_col: work2[case_col].to_numpy(),
                prefix_index_col: work2[prefix_col].to_numpy(),
                "teacher_logit": np.asarray(logits_buf, dtype=np.float32),
            }
        )

        for i in range(z_buf.shape[1]):
            out[f"teacher_embed_pca_{i + 1}"] = np.asarray(z_buf[:, i], dtype=np.float32)

        return out, pca

    for idxs, log_b, _ in _iter_forward_batched(
        model,
        rows,
        build_seq,
        cat_src_key="_cat_id",
        cat_dst_key=model_cat_key,
        vocab=vocab,
    ):
        logits_buf[idxs] = log_b

    out = pd.DataFrame(
        {
            case_id_col: work2[case_col].to_numpy(),
            prefix_index_col: work2[prefix_col].to_numpy(),
            "teacher_logit": np.asarray(logits_buf, dtype=np.float32),
        }
    )

    return out, None


def distill_tau_split(
    tau_df: pd.DataFrame,
    events_df: pd.DataFrame,
    model: TeacherModel,
    vocab: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    *,
    case_id_col: str = "case:concept:name",
    tau_col: str = "tau_ts",
    pca_components: int = 0,
    pca: Optional[PCA] = None,
    include_k_idx: bool = True,
    memmap_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Optional[PCA]]:
    if tau_df.empty:
        out = (
            tau_df[[case_id_col, tau_col]].copy()
            if all(c in tau_df.columns for c in [case_id_col, tau_col])
            else pd.DataFrame({case_id_col: [], tau_col: []})
        )
        out["teacher_logit"] = np.array([], dtype=np.float32)
        return out, pca if pca_components > 0 else None

    work = tau_df.copy()
    work[tau_col] = pd.to_datetime(work[tau_col], utc=True, errors="coerce")
    work = work[work[tau_col].notna()].copy()

    if work.empty:
        out = pd.DataFrame({case_id_col: [], tau_col: [], "teacher_logit": []})
        return out, pca if pca_components > 0 else None

    cat_col = dataset_cfg.get("cat_col", "concept:name")
    num_cols = list(dataset_cfg.get("num_cols", []))
    case_col = dataset_cfg.get("case_col", case_id_col)
    time_col = dataset_cfg.get("time_col", "time:timestamp")
    max_len = int(dataset_cfg.get("max_len", 100))

    events_df_min = _slice_events_minimal(
        events_df,
        case_col,
        time_col,
        cat_col,
        num_cols,
    )

    first_event = (
        events_df_min[[case_col, time_col]]
        .dropna()
        .groupby(case_col, sort=False)[time_col]
        .min()
        .rename("first_event_ts")
    )

    work = work.merge(first_event, how="left", left_on=case_col, right_index=True)
    work = work[work["first_event_ts"].notna()].copy()
    work = work[work[tau_col] >= work["first_event_ts"]].copy()
    work = work.drop(columns=["first_event_ts"])

    if work.empty:
        out = pd.DataFrame({case_id_col: [], tau_col: [], "teacher_logit": []})
        return out, pca if pca_components > 0 else None

    events_fast, spans = _prepare_events_fast(
        events_df_min,
        cat_col=cat_col,
        num_cols=num_cols,
        case_col=case_col,
        time_col=time_col,
        vocab=vocab,
    )

    build_seq = _make_seq_builder_by_time_fast(
        events_fast,
        spans,
        cat_id_col="_cat_id",
        num_cols=num_cols,
        case_col=case_col,
        time_col=time_col,
        max_len=max_len,
    )

    work2 = work

    if case_col not in work2.columns and case_id_col in work2.columns:
        work2 = work2.rename(columns={case_id_col: case_col})

    if tau_col != "tau_ts":
        work2 = work2.rename(columns={tau_col: "tau_ts"})

    base_cols = [case_col, "tau_ts"]

    if "__row_id__" in work2.columns:
        base_cols.append("__row_id__")

    rows = work2[base_cols].to_dict("records")

    cat_keys = list(model.cfg.num_cats.keys())

    if not cat_keys:
        raise RuntimeError("TeacherConfig.num_cats is empty.")

    model_cat_key = cat_keys[0]

    n_rows = len(rows)
    mmap_dir = Path(memmap_dir) if memmap_dir else None

    logits_buf, _ = _maybe_memmap(
        mmap_dir / "logits_tau.memmap" if mmap_dir else None,
        (n_rows,),
        np.float32,
    )

    def _make_output(z_buf=None, fitted_pca=None):
        out = pd.DataFrame(
            {
                case_id_col: work2[case_col].to_numpy(),
                tau_col: work2["tau_ts"].to_numpy(),
                "teacher_logit": np.asarray(logits_buf, dtype=np.float32),
            }
        )

        if "__row_id__" in work2.columns:
            out["__row_id__"] = work2["__row_id__"].to_numpy()

        if include_k_idx and "k_idx" in work2.columns:
            out["k_idx"] = work2["k_idx"].to_numpy()

        if z_buf is not None:
            for i in range(z_buf.shape[1]):
                out[f"teacher_embed_pca_{i + 1}"] = np.asarray(z_buf[:, i], dtype=np.float32)

        return out, fitted_pca

    if pca_components > 0 and pca is None:
        ipca = IncrementalPCA(
            n_components=pca_components,
            batch_size=max(1024, MIN_BATCH_SIZE),
        )

        for _, _, emb_b in _iter_forward_batched(
            model,
            rows,
            build_seq,
            cat_src_key="_cat_id",
            cat_dst_key=model_cat_key,
            vocab=vocab,
        ):
            if emb_b.size:
                ipca.partial_fit(np.nan_to_num(emb_b, copy=False))

        z_buf, _ = _maybe_memmap(
            mmap_dir / "embeddings_tau.memmap" if mmap_dir else None,
            (n_rows, pca_components),
            np.float32,
        )

        for idxs, log_b, emb_b in _iter_forward_batched(
            model,
            rows,
            build_seq,
            cat_src_key="_cat_id",
            cat_dst_key=model_cat_key,
            vocab=vocab,
        ):
            logits_buf[idxs] = log_b

            if emb_b.size:
                z_buf[idxs, :] = ipca.transform(
                    np.nan_to_num(emb_b, copy=False)
                ).astype(np.float32, copy=False)
            else:
                z_buf[idxs, :] = 0.0

        return _make_output(z_buf, ipca)

    if pca_components > 0 and pca is not None:
        z_buf, _ = _maybe_memmap(
            mmap_dir / "embeddings_tau.memmap" if mmap_dir else None,
            (n_rows, pca.n_components_),
            np.float32,
        )

        for idxs, log_b, emb_b in _iter_forward_batched(
            model,
            rows,
            build_seq,
            cat_src_key="_cat_id",
            cat_dst_key=model_cat_key,
            vocab=vocab,
        ):
            logits_buf[idxs] = log_b

            if emb_b.size:
                z_buf[idxs, :] = pca.transform(
                    np.nan_to_num(emb_b, copy=False)
                ).astype(np.float32, copy=False)
            else:
                z_buf[idxs, :] = 0.0

        return _make_output(z_buf, pca)

    for idxs, log_b, _ in _iter_forward_batched(
        model,
        rows,
        build_seq,
        cat_src_key="_cat_id",
        cat_dst_key=model_cat_key,
        vocab=vocab,
    ):
        logits_buf[idxs] = log_b

    return _make_output(None, None)
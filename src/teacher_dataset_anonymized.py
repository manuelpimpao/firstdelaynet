# teacher_dataset.py
"""
Build masked, leakage-safe sequence datasets for the teacher model.

Features
--------
- Compact dtypes: tokens -> uint16 when possible, numerics/masks/labels -> float16.
- Optional sharding to disk for large datasets.
- Stable sorting and one-pass per-case sequence construction.
- Optional max_items for smoke tests.

Outputs
-------
- teacher_vocab.json
- teacher_config.json
- teacher_<split>_ds.npz
- teacher_<split>_keys.parquet

Or, when sharding is enabled:
- teacher_shard_###.npz
- teacher_shard_###_keys.parquet
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class DatasetPack:
    """Container for one teacher-model dataset split."""

    seq_cat: np.ndarray
    seq_num: np.ndarray
    mask_t: np.ndarray
    y_t: np.ndarray
    keys: pd.DataFrame


class TeacherDatasetBuilder:
    """
    Build masked, leakage-safe sequence datasets from a preprocessed event-log split.

    Required input columns
    ----------------------
    - case_col
    - prefix_col
    - cat_col
    - num_cols
    - label_col
    - mask_col

    Leakage safety
    --------------
    Each sequence contains only events up to and including the current prefix.
    """

    def __init__(
        self,
        cat_col: str,
        num_cols: List[str],
        case_col: str = "case_seg",
        prefix_col: str = "prefix_index",
        label_col: str = "next_is_delayed_dyn",
        mask_col: str = "mask_transport",
        max_len: int = 100,
    ):
        self.cat_col = cat_col
        self.num_cols = list(num_cols)
        self.case_col = case_col
        self.prefix_col = prefix_col
        self.label_col = label_col
        self.mask_col = mask_col
        self.max_len = int(max_len)

        self._vocab: Optional[Dict[str, int]] = None

    def build_vocab(self, df: pd.DataFrame) -> Dict[str, int]:
        """Fit token vocabulary on the training split only."""
        values = (
            df[self.cat_col]
            .astype("string")
            .fillna("")
            .astype(str)
            .unique()
            .tolist()
        )

        vocab = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }

        for i, value in enumerate(sorted(values), start=2):
            if value:
                vocab[value] = i

        self._vocab = vocab
        return vocab

    def set_vocab(self, vocab: Dict[str, int]) -> None:
        self._vocab = dict(vocab)

    def get_vocab(self) -> Dict[str, int]:
        if self._vocab is None:
            raise RuntimeError("Vocabulary not set. Call build_vocab() or set_vocab() first.")
        return self._vocab

    def save_vocab(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "teacher_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_vocab(), f, ensure_ascii=False, indent=2)

        config = {
            "num_cols": self.num_cols,
            "cat_col": self.cat_col,
            "case_col": self.case_col,
            "prefix_col": self.prefix_col,
            "label_col": self.label_col,
            "mask_col": self.mask_col,
            "max_len": self.max_len,
        }

        with open(os.path.join(out_dir, "teacher_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def load_vocab(path: str) -> Dict[str, int]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build(
        self,
        df: pd.DataFrame,
        drop_unmasked: bool = True,
        max_items: Optional[int] = None,
        write_shards_dir: Optional[str] = None,
        shard_size: int = 100_000,
    ) -> DatasetPack:
        """
        Construct a teacher-model sequence dataset.

        Parameters
        ----------
        df:
            Preprocessed event-log split.
        drop_unmasked:
            If True, keep only rows where mask_col == 1.
        max_items:
            Optional cap for smoke tests.
        write_shards_dir:
            If provided, write shards to disk while building.
        shard_size:
            Number of items per shard.

        Returns
        -------
        DatasetPack
            Full in-memory arrays if write_shards_dir is None.
            Empty arrays if sharding is enabled.
        """
        vocab = self.get_vocab()

        required_cols = {
            self.case_col,
            self.prefix_col,
            self.cat_col,
            self.label_col,
            *self.num_cols,
        }

        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.mask_col not in df.columns:
            df = df.copy()
            df[self.mask_col] = 1

        df = df.sort_values([self.case_col, self.prefix_col], kind="mergesort")

        if self.num_cols:
            df[self.num_cols] = df[self.num_cols].fillna(0.0)

            for col in self.num_cols:
                if df[col].dtype != np.float32:
                    df[col] = df[col].astype(np.float32, copy=False)

        unk_id = vocab.get(UNK_TOKEN, 1)

        token_ids_int32 = (
            df[self.cat_col]
            .astype("string")
            .fillna("")
            .map(vocab)
            .fillna(unk_id)
            .astype(np.int32)
            .to_numpy(copy=False)
        )

        case_ids = df[self.case_col].to_numpy(copy=False)
        prefix_values = df[self.prefix_col].to_numpy(dtype=np.int32, copy=False)
        mask_values = df[self.mask_col].to_numpy(dtype=np.float32, copy=False)
        labels = df[self.label_col].to_numpy(dtype=np.float32, copy=False)

        if self.num_cols:
            num_matrix = df[self.num_cols].to_numpy(dtype=np.float32, copy=False)
        else:
            num_matrix = np.empty((len(df), 0), np.float32)

        if num_matrix.size:
            np.nan_to_num(num_matrix, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
            np.clip(num_matrix, -1e6, 1e6, out=num_matrix)

        case_change = np.r_[True, case_ids[1:] != case_ids[:-1]]
        case_starts = np.flatnonzero(case_change)
        case_ends = np.r_[case_starts[1:], len(df)]

        max_len = int(self.max_len)
        n_num_features = num_matrix.shape[1]

        token_dtype = np.uint16 if len(vocab) < 65_535 else np.int32
        token_ids = token_ids_int32.astype(token_dtype, copy=False)

        seq_cat_list: List[np.ndarray] = []
        seq_num_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        label_list: List[np.ndarray] = []
        keys: List[Tuple[object, int]] = []

        produced = 0
        shard_idx = 0

        if write_shards_dir is not None:
            os.makedirs(write_shards_dir, exist_ok=True)

        def flush_shard() -> None:
            nonlocal shard_idx

            if not seq_cat_list:
                return

            seq_cat = np.stack(seq_cat_list, axis=0)
            seq_num = np.ascontiguousarray(np.stack(seq_num_list, axis=0))
            mask_t = np.stack(mask_list, axis=0)
            y_t = np.asarray(label_list, dtype=np.float16)

            np.savez_compressed(
                os.path.join(write_shards_dir, f"teacher_shard_{shard_idx:03d}.npz"),
                seq_cat=seq_cat,
                seq_num=seq_num,
                mask_t=mask_t,
                y_t=y_t,
            )

            pd.DataFrame(
                keys,
                columns=[self.case_col, self.prefix_col],
            ).to_parquet(
                os.path.join(write_shards_dir, f"teacher_shard_{shard_idx:03d}_keys.parquet"),
                index=False,
            )

            shard_idx += 1

            seq_cat_list.clear()
            seq_num_list.clear()
            mask_list.clear()
            label_list.clear()
            keys.clear()

        for start, end in zip(case_starts, case_ends):
            case_tokens = token_ids[start:end]
            case_nums = num_matrix[start:end, :]
            case_mask = mask_values[start:end]
            case_labels = labels[start:end]
            case_prefix = prefix_values[start:end]
            case_id = case_ids[start]

            if drop_unmasked:
                output_positions = np.flatnonzero(case_mask == 1.0)
            else:
                output_positions = np.arange(end - start, dtype=np.int64)

            if output_positions.size == 0:
                continue

            for pos in output_positions:
                lo = max(0, pos + 1 - max_len)
                hi = pos + 1

                cat_tail = case_tokens[lo:hi]

                if n_num_features:
                    num_tail = case_nums[lo:hi, :].astype(np.float16, copy=False)
                else:
                    num_tail = np.empty((hi - lo, 0), dtype=np.float16)

                seq_len = cat_tail.shape[0]

                mask_tail = np.zeros((seq_len,), dtype=np.float16)
                mask_tail[-1] = 1.0

                if seq_len < max_len:
                    pad_len = max_len - seq_len

                    cat_padded = np.empty((max_len,), dtype=token_dtype)
                    cat_padded[:pad_len] = 0
                    cat_padded[pad_len:] = cat_tail

                    if n_num_features:
                        num_padded = np.empty((max_len, n_num_features), dtype=np.float16)
                        num_padded[:pad_len, :] = 0
                        num_padded[pad_len:, :] = num_tail
                    else:
                        num_padded = np.empty((max_len, 0), dtype=np.float16)

                    mask_padded = np.empty((max_len,), dtype=np.float16)
                    mask_padded[:pad_len] = 0
                    mask_padded[pad_len:] = mask_tail
                else:
                    cat_padded = cat_tail
                    num_padded = num_tail
                    mask_padded = mask_tail

                seq_cat_list.append(cat_padded)
                seq_num_list.append(num_padded)
                mask_list.append(mask_padded)
                label_list.append(np.float16(case_labels[pos]))
                keys.append((case_id, int(case_prefix[pos])))

                produced += 1

                if write_shards_dir is not None and len(seq_cat_list) >= shard_size:
                    flush_shard()

                if max_items is not None and produced >= max_items:
                    break

            if max_items is not None and produced >= max_items:
                break

        if write_shards_dir is not None:
            flush_shard()

            return DatasetPack(
                seq_cat=np.empty((0, self.max_len), dtype=token_dtype),
                seq_num=np.empty((0, self.max_len, n_num_features), dtype=np.float16),
                mask_t=np.empty((0, self.max_len), dtype=np.float16),
                y_t=np.empty((0,), dtype=np.float16),
                keys=pd.DataFrame(columns=[self.case_col, self.prefix_col]),
            )

        if not seq_cat_list:
            return DatasetPack(
                seq_cat=np.empty((0, self.max_len), dtype=token_dtype),
                seq_num=np.empty((0, self.max_len, n_num_features), dtype=np.float16),
                mask_t=np.empty((0, self.max_len), dtype=np.float16),
                y_t=np.empty((0,), dtype=np.float16),
                keys=pd.DataFrame(columns=[self.case_col, self.prefix_col]),
            )

        return DatasetPack(
            seq_cat=np.stack(seq_cat_list, axis=0).astype(token_dtype, copy=False),
            seq_num=np.ascontiguousarray(np.stack(seq_num_list, axis=0)),
            mask_t=np.stack(mask_list, axis=0).astype(np.float16, copy=False),
            y_t=np.asarray(label_list, dtype=np.float16),
            keys=pd.DataFrame(keys, columns=[self.case_col, self.prefix_col]),
        )

    def save_npz(self, pack: DatasetPack, out_dir: str, split_name: str) -> None:
        """Save dataset arrays and keys for one split."""
        os.makedirs(out_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(out_dir, f"teacher_{split_name}_ds.npz"),
            seq_cat=pack.seq_cat,
            seq_num=pack.seq_num,
            mask_t=pack.mask_t,
            y_t=pack.y_t,
        )

        pack.keys.to_parquet(
            os.path.join(out_dir, f"teacher_{split_name}_keys.parquet"),
            index=False,
        )

    @staticmethod
    def load_npz(path_npz: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load seq_cat, seq_num, mask_t, and y_t from a saved .npz file."""
        npz = np.load(path_npz)
        return npz["seq_cat"], npz["seq_num"], npz["mask_t"], npz["y_t"]
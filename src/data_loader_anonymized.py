# -*- coding: utf-8 -*-
"""
data_loader.py — Event-log loader for the research dataset

Key points
- Loads event-log data from CSV files or directories of CSV files.
- Optional parquet caching.
- Supports two output schemas:
    - "pm4py": normalized PM4Py-style event-log columns
    - "passthrough": preserves the input schema where possible
- Optional sequence-model enrichment:
    - is_transport_start
    - is_transport_finish
    - mask_transport_start
    - target_station_at_start
    - current_station

Example
-------
from data_loader import EventLogLoader

loader = EventLogLoader(
    file_path="data/raw/events.csv",
    delimiter=",",
    use_parquet_cache=True,
    schema="pm4py",
)

df = loader.load_df()
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd


try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False


try:
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    _HAS_PM4PY = True
except Exception:
    _HAS_PM4PY = False


def _default_dtypes() -> Dict[str, str]:
    str_type = "string[pyarrow]" if _HAS_PYARROW else "string"

    return {
        "case_id": str_type,
        "event_id": str_type,
        "activity": str_type,
        "category": str_type,
        "subsystem": str_type,
        "subsystemtype": str_type,
        "source_location": str_type,
        "destination_location": str_type,
        "case_start": str_type,
        "case_end": str_type,
        "material_id": str_type,
        "vehicle_id": str_type,
        "event_type": str_type,
        "timestamp": str_type,
        "planned_timestamp": str_type,
        "event_end_timestamp": str_type,
    }


class EventLogLoader:
    def __init__(
        self,
        file_path: str,
        delimiter: str = ",",
        use_parquet_cache: bool = False,
        parquet_path: Optional[str] = None,
        schema: str = "pm4py",
        enrich_for_sequence_model: bool = False,
    ):
        self.file_path = file_path
        self.delimiter = delimiter
        self.use_parquet_cache = use_parquet_cache
        self.parquet_path = parquet_path
        self.schema = schema
        self.enrich_for_sequence_model = enrich_for_sequence_model

    def load_df(self) -> pd.DataFrame:
        if os.path.isdir(self.file_path):
            df = self._read_csv_directory(self.file_path, self.delimiter)
            return self._postprocess_df(df)

        parquet_path = self._resolve_parquet_path(self.file_path, self.parquet_path)

        if self.use_parquet_cache and os.path.isfile(parquet_path):
            df = pd.read_parquet(parquet_path)
            if self.enrich_for_sequence_model:
                df = self._ensure_sequence_model_columns(df)
            return df

        df = self._read_csv_file(self.file_path, self.delimiter)
        df = self._postprocess_df(df)

        if self.use_parquet_cache:
            try:
                df.to_parquet(parquet_path, index=False)
            except Exception:
                pass

        return df

    def load_preprocessed_df(self) -> pd.DataFrame:
        parse_cols = ["time:timestamp", "planned:timestamp"]

        if os.path.isdir(self.file_path):
            frames = []
            for path in self._list_csvs(self.file_path):
                frames.append(
                    pd.read_csv(
                        path,
                        sep=self.delimiter,
                        parse_dates=parse_cols,
                    )
                )
            return pd.concat(frames, ignore_index=True)

        return pd.read_csv(
            self.file_path,
            sep=self.delimiter,
            parse_dates=parse_cols,
        )

    def load_eventlog(self):
        if not _HAS_PM4PY:
            raise ImportError("pm4py is not installed. Use load_df() instead.")

        df = self.load_df().copy()
        df = dataframe_utils.convert_timestamp_columns_in_df(df)
        return log_converter.apply(df)

    def load_preprocessed_eventlog(self):
        if not _HAS_PM4PY:
            raise ImportError("pm4py is not installed. Use load_preprocessed_df() instead.")

        df = self.load_preprocessed_df()
        df = dataframe_utils.convert_timestamp_columns_in_df(df)
        return log_converter.apply(df)

    def _resolve_parquet_path(self, file_path: str, parquet_path: Optional[str]) -> str:
        if parquet_path:
            return parquet_path

        root, _ = os.path.splitext(file_path)
        return root + ".parquet"

    def _list_csvs(self, directory: str) -> List[str]:
        return [
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.lower().endswith(".csv")
        ]

    def _read_csv_directory(self, directory: str, sep: str) -> pd.DataFrame:
        frames = []

        for path in self._list_csvs(directory):
            frames.append(self._read_csv_file(path, sep))

        return pd.concat(frames, ignore_index=True)

    def _read_csv_file(
        self,
        path: str,
        sep: str,
        dtypes: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        if dtypes is None:
            dtypes = _default_dtypes()

        engine = "pyarrow" if _HAS_PYARROW else "c"

        read_kwargs = {
            "sep": sep,
            "engine": engine,
            "dtype": dtypes,
        }

        if engine == "c":
            read_kwargs["low_memory"] = False

        return pd.read_csv(path, **read_kwargs)

    def _postprocess_df(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()

        rename_map = {
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp",
            "planned_timestamp": "planned:timestamp",
            "case_start": "case:start",
            "case_end": "case:end",
            "source_location": "event:start",
            "destination_location": "event:end",
            "material_id": "sachnummer",
            "event_type": "eventkey",
            "event_end_timestamp": "last_event_start_timestamp",
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "time:timestamp" in df.columns:
            df["time:timestamp"] = pd.to_datetime(
                df["time:timestamp"],
                errors="coerce",
                utc=True,
            )
        else:
            df["time:timestamp"] = pd.NaT

        if "planned:timestamp" in df.columns:
            df["planned:timestamp"] = pd.to_datetime(
                df["planned:timestamp"],
                errors="coerce",
                utc=True,
            )
        else:
            df["planned:timestamp"] = pd.NaT

        if "last_event_start_timestamp" in df.columns:
            df["last_event_start_timestamp"] = pd.to_datetime(
                df["last_event_start_timestamp"],
                errors="coerce",
                utc=True,
            )
        else:
            df["last_event_start_timestamp"] = pd.NaT

        for col in ["category", "subsystemtype", "vehicle_id"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        if self.enrich_for_sequence_model:
            df = self._ensure_sequence_model_columns(df)

        if self.schema == "passthrough":
            return df

        required_pm_cols = [
            "case:concept:name",
            "event_id",
            "concept:name",
            "time:timestamp",
            "planned:timestamp",
            "case:start",
            "case:end",
            "event:start",
            "event:end",
            "sachnummer",
            "eventkey",
            "last_event_start_timestamp",
        ]

        for col in required_pm_cols:
            if col not in df.columns:
                df[col] = pd.Series(pd.NA, index=df.index)

        for col in [
            "case:concept:name",
            "event_id",
            "concept:name",
            "case:start",
            "case:end",
            "event:start",
            "event:end",
            "sachnummer",
            "eventkey",
        ]:
            if col in df.columns:
                df[col] = df[col].astype("string").astype("category")

        other_cols = [col for col in df.columns if col not in required_pm_cols]
        return df[required_pm_cols + other_cols]

    def _ensure_sequence_model_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["category", "eventkey", "event:end", "event:start"]:
            if col not in df.columns:
                df[col] = pd.Series(pd.NA, index=df.index)

        is_start = (
            df["category"].astype("string").eq("transport")
            & df["eventkey"].astype("string").eq("subsystem_started")
        )

        is_finish = (
            df["category"].astype("string").eq("transport")
            & df["eventkey"].astype("string").eq("subsystem_finished")
        )

        df["is_transport_start"] = is_start.astype("int8")
        df["is_transport_finish"] = is_finish.astype("int8")
        df["mask_transport_start"] = df["is_transport_start"].astype("int8")

        if "target_station_at_start" not in df.columns:
            df["target_station_at_start"] = pd.Series(pd.NA, index=df.index, dtype="object")

        need_fill = df["is_transport_start"].eq(1) & df["target_station_at_start"].isna()
        df.loc[need_fill, "target_station_at_start"] = df.loc[need_fill, "event:end"]

        sort_cols = ["case:concept:name", "time:timestamp"]
        available_sort_cols = [col for col in sort_cols if col in df.columns]

        if available_sort_cols:
            df = df.sort_values(
                by=available_sort_cols,
                kind="mergesort",
            )

        if "current_station" not in df.columns:
            df["current_station"] = pd.Series(pd.NA, index=df.index, dtype="object")

        def _carry_forward_current_station(group: pd.DataFrame) -> pd.DataFrame:
            finish_destinations = group.loc[
                group["is_transport_finish"] == 1,
                "event:end",
            ].astype("object")

            carry = finish_destinations.reindex(group.index).ffill()

            out = group.copy()
            out["current_station"] = carry
            return out

        if "case:concept:name" in df.columns:
            df = (
                df.groupby("case:concept:name", group_keys=False, observed=True)
                .apply(_carry_forward_current_station)
            )

        if "case:start" in df.columns:
            missing_current = df["current_station"].isna()
            df.loc[missing_current, "current_station"] = df.loc[missing_current, "case:start"]

        if "ev_seq" not in df.columns and "case:concept:name" in df.columns:
            df["ev_seq"] = (
                df.groupby("case:concept:name", observed=True)
                .cumcount()
                .add(1)
                .astype("int32")
            )

        for col in ["is_transport_start", "is_transport_finish", "mask_transport_start"]:
            df[col] = df[col].astype("int8")

        return df


if __name__ == "__main__":
    example_path = "data/raw/events.csv"

    loader = EventLogLoader(
        file_path=example_path,
        delimiter=",",
        use_parquet_cache=False,
        schema="pm4py",
        enrich_for_sequence_model=False,
    )

    df = loader.load_df()
    print("Loaded DataFrame:", df.shape)
    print("Columns:", list(df.columns))
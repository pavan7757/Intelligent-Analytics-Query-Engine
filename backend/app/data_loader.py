import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings


def _safe_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if str(column).lower() in {
            "amount", "revenue", "sales", "quantity", "units", "price",
            "cost", "profit", "target", "actual", "percentage",
        }:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _safe_date_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if any(token in str(column).lower() for token in ("date", "time", "period")):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    if path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return _safe_date_frame(_safe_numeric_frame(frame.copy()))


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class DatasetLoader:
    def __init__(self) -> None:
        self.dataset_dir = settings.dataset_dir
        self.sales = load_csv(self.dataset_dir / settings.sales_file)
        self.targets = load_csv(self.dataset_dir / settings.targets_file)
        self.dictionary = load_json(self.dataset_dir / settings.dictionary_file)
        self.queries = load_json(self.dataset_dir / settings.queries_file)
        self.feedback_path = self.dataset_dir / settings.feedback_file

    def get_schema(self) -> dict[str, Any]:
        return {
            "sales_columns": [str(column) for column in self.sales.columns],
            "sales_empty": self.sales.empty,
            "targets_columns": [str(column) for column in self.targets.columns],
            "sales_dtypes": {
                str(column): str(dtype)
                for column, dtype in self.sales.dtypes.items()
            },
            "targets_dtypes": {
                str(column): str(dtype)
                for column, dtype in self.targets.dtypes.items()
            },
            "dictionary": self.dictionary,
            "queries": self.queries,
        }

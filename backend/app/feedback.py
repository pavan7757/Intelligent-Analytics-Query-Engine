import csv
from datetime import datetime
from typing import Any

from .config import settings


class FeedbackStore:
    def __init__(self) -> None:
        self.path = settings.dataset_dir / settings.feedback_file

    def get_relevant(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        normalized = query.casefold()
        return [
            row for row in rows
            if normalized and normalized in row.get("query", "").casefold()
        ][:limit]

    def append(self, request: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["query", "incorrect_interpretation", "correct_interpretation", "notes", "created_at"]
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        row = {
            **request,
            "created_at": datetime.now().isoformat(),
        }
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        return row

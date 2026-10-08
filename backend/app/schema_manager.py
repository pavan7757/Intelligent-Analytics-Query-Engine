import re
from typing import Any, Callable

import pandas as pd

from .data_loader import DatasetLoader


def _normalize_identifier(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _singular_entity(value: str) -> str:
    normalized = _normalize_identifier(value)
    prefix, _, word = normalized.rpartition("_")
    word = word or normalized
    if word.endswith("ies") and len(word) > 3:
        word = word[:-3] + "y"
    elif word.endswith(("ches", "shes", "xes", "zes")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us")):
        word = word[:-1]
    return f"{prefix}_{word}" if prefix else word


class SchemaManager:
    def __init__(self, loader: DatasetLoader) -> None:
        self.loader = loader

    def available_columns(self) -> dict[str, list[str]]:
        return {
            "sales": [str(column) for column in self.loader.sales.columns],
            "targets": [str(column) for column in self.loader.targets.columns],
        }

    def canonical_field(self, requested: str) -> str | None:
        normalized = requested.strip().lower().replace("-", "_").replace(" ", "_")
        for source in ("sales", "targets"):
            for column in self.available_columns()[source]:
                if column.strip().lower() == requested.strip().lower() or (
                    column.strip().lower().replace("-", "_").replace(" ", "_")
                    == normalized
                ):
                    return column
        return None

    def resolve_derived_dimension(self, requested: str) -> str | None:
        normalized = requested.strip().lower().replace("-", "_").replace(" ", "_")
        if not self.date_columns():
            return None
        if normalized in {"month", "year", "quarter", "day", "week"}:
            return normalized
        return None

    def is_numeric_field(self, field: str) -> bool:
        canonical = self.canonical_field(field)
        if canonical is not None:
            return pd.api.types.is_numeric_dtype(self.loader.sales[canonical])
        if field.strip().lower() in {"revenue", "sales", "amount", "profit"}:
            return True
        return False

    def resolve_business_term(self, term: str) -> str | None:
        normalized = term.strip().lower().replace("-", "_").replace(" ", "_")
        dictionary = self.loader.dictionary
        aliases = dictionary.get("terms", {}) if isinstance(dictionary, dict) else {}
        if isinstance(aliases, dict):
            for key, value in aliases.items():
                if key.lower().replace("-", "_").replace(" ", "_") == normalized:
                    return value if isinstance(value, str) else value.get("column")
                if isinstance(value, dict) and value.get("aliases"):
                    if normalized in {
                        alias.lower().replace("-", "_").replace(" ", "_")
                        for alias in value["aliases"]
                    }:
                        return value.get("column")
        return self.canonical_field(term)

    def resolve_entity_dimension(self, requested: str) -> str | None:
        """Resolve an entity's identity, rather than one of its attributes.

        A dictionary may declare ``entities.<entity>.identity_column`` (or
        ``id_column``/``name_column``), with optional aliases. Column definitions
        may instead declare ``entity`` and ``role`` as identity/id/name. Without
        metadata, an unambiguous matching ``<entity>_id`` takes precedence over
        ``<entity>_name``. A segment or category never identifies an entity.
        """
        entity = _singular_entity(requested)
        columns = [str(column) for column in self.loader.sales.columns]

        def identity_column(field: Any) -> str | None:
            if not isinstance(field, str):
                return None
            candidates = [column for column in columns
                          if _normalize_identifier(column) == _normalize_identifier(field)]
            if len(candidates) != 1:
                return None
            column = candidates[0]
            if _normalize_identifier(column).endswith(("_segment", "_category")):
                return None
            return column

        dictionary = self.loader.dictionary if isinstance(self.loader.dictionary, dict) else {}
        declared_ids: list[str] = []
        declared_names: list[str] = []
        invalid_declaration = False
        entities = dictionary.get("entities", {})
        if isinstance(entities, dict):
            for name, definition in entities.items():
                if not isinstance(definition, dict):
                    continue
                aliases = definition.get("aliases", [])
                aliases = aliases if isinstance(aliases, list) else []
                names = [name, *[alias for alias in aliases if isinstance(alias, str)]]
                if entity not in {_singular_entity(name) for name in names}:
                    continue
                for key in ("identity_column", "identity_field", "id_column", "name_column"):
                    if key not in definition:
                        continue
                    column = identity_column(definition[key])
                    if column is None:
                        invalid_declaration = True
                    elif key == "name_column":
                        declared_names.append(column)
                    else:
                        declared_ids.append(column)
        for section in ("columns", "definitions"):
            definitions = dictionary.get(section, {})
            if not isinstance(definitions, dict):
                continue
            for name, definition in definitions.items():
                if not isinstance(definition, dict) or not isinstance(definition.get("entity"), str):
                    continue
                if _singular_entity(definition["entity"]) != entity:
                    continue
                role = str(definition.get("role", "")).strip().lower()
                if role not in {"identity", "id", "name"}:
                    continue
                column = identity_column(definition.get("column", name))
                if column is None:
                    invalid_declaration = True
                elif role == "name":
                    declared_names.append(column)
                else:
                    declared_ids.append(column)
        if invalid_declaration:
            return None
        inferred_ids = [column for column in columns
                        if _normalize_identifier(column).endswith("_id")
                        and _singular_entity(_normalize_identifier(column)[:-3]) == entity]
        inferred_names = [column for column in columns
                          if _normalize_identifier(column).endswith("_name")
                          and _singular_entity(_normalize_identifier(column)[:-5]) == entity]
        for collection in (declared_ids, inferred_ids, declared_names, inferred_names):
            candidates = list(dict.fromkeys(collection))
            if candidates:
                return candidates[0] if len(candidates) == 1 else None
        return None

    def validate_field(self, field: str, source: str = "sales") -> bool:
        return self.canonical_field(field) in self.available_columns()[source]

    def validate_fields(self, fields: list[str], source: str = "sales") -> list[str]:
        return [field for field in fields if self.validate_field(field, source)]

    def categorical_index(self) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for column in self.loader.sales.columns:
            series = self.loader.sales[column]
            if pd.api.types.is_numeric_dtype(series):
                continue
            values = series.dropna().astype(str).unique()
            unique_values = list(dict.fromkeys(str(value).strip() for value in values))
            if unique_values:
                index[str(column)] = unique_values
        return index

    def resolve_categorical_value(self, value: str) -> tuple[str, str] | None:
        normalized = value.strip().casefold()
        matches = []
        for column, candidates in self.categorical_index().items():
            for candidate in candidates:
                if candidate.casefold() == normalized:
                    matches.append((column, candidate))
        if len(matches) == 1:
            return matches[0]
        return None

    def resolve_categorical_phrase(
        self, phrase: str, resolve_dimension: Callable[[str], str | None],
    ) -> tuple[str, str] | None:
        """Resolve a complete value description without discarding its qualifier."""
        literal = phrase.strip().strip("\"'")
        exact = self.resolve_categorical_value(literal)
        if exact:
            return exact
        phrase = re.sub(r"^(?:(?:the|a|an|these|those)\s+)+", "", phrase.strip(), flags=re.IGNORECASE)
        exact = self.resolve_categorical_value(phrase.strip("\"'"))
        if exact:
            return exact
        matches = []
        for column, values in self.categorical_index().items():
            if column in self.date_columns():
                continue
            for value in values:
                match = re.fullmatch(rf"[\"']?{re.escape(value)}[\"']?\s+(.+)", phrase, re.IGNORECASE)
                if not match:
                    continue
                qualifier = match.group(1).strip()
                field = resolve_dimension(qualifier)
                if field == column:
                    matches.append((column, value))
                    continue
                # A known value of an entity attribute may describe that entity:
                # e.g. a category value followed by "products". It does not
                # change the entity's identity used for grouping or ranking.
                entity = _singular_entity(qualifier)
                if field and _normalize_identifier(field).endswith(("_id", "_name")):
                    entity = _normalize_identifier(field).rsplit("_", 1)[0]
                if _normalize_identifier(column).startswith(entity + "_"):
                    matches.append((column, value))
        matches = list(dict.fromkeys(matches))
        return matches[0] if len(matches) == 1 else None

    def metric_candidates(self) -> list[str]:
        return [
            str(column)
            for column in self.loader.sales.columns
            if str(column).lower() in {
                "amount", "revenue", "sales", "quantity", "units", "price",
                "cost", "profit", "target", "actual",
            }
        ]

    def find_target_column(self, metric: str) -> str | None:
        metric_name = _normalize_identifier(metric)
        target_columns = list(self.loader.targets.columns)
        for column in target_columns:
            normalized = _normalize_identifier(column)
            if normalized in {metric_name, f"{metric_name}_target", f"target_{metric_name}"}:
                return column
            if "target" in normalized and metric_name in normalized:
                return column
            if normalized.endswith("_target") and metric_name in normalized:
                return column
        if len(target_columns) == 1:
            return target_columns[0]
        for column in target_columns:
            normalized = _normalize_identifier(column)
            if normalized in {"target", "actual", "target_amount", "actual_target"}:
                return column
        return None

    def resolve_target_metric(self, metric: str) -> str | None:
        if not metric:
            return None
        return self.find_target_column(metric)

    def join_keys(self) -> list[str]:
        sales = self.available_columns()["sales"]
        targets = self.available_columns()["targets"]
        shared = [column for column in sales if column in targets]
        return shared or []

    def compatible_join_keys(self, group_by: list[str] | None = None) -> list[str]:
        target_columns = [str(column) for column in self.loader.targets.columns]
        join_candidates = self.join_keys()
        if group_by:
            normalized_group = {_normalize_identifier(group) for group in group_by}
            group_matches = [
                column for column in target_columns
                if _normalize_identifier(column) in normalized_group
            ]
            if group_matches:
                return group_matches
        if join_candidates:
            return join_candidates
        if target_columns:
            return [target_columns[0]]
        return []

    def date_columns(self) -> list[str]:
        dictionary_fields = self._dictionary_date_fields()
        return [
            str(column) for column in self.loader.sales.columns
            if pd.api.types.is_datetime64_any_dtype(self.loader.sales[column])
            or any(token in str(column).lower() for token in ("date", "time", "period"))
            or str(column) in dictionary_fields
        ]

    def _dictionary_date_fields(self) -> list[str]:
        dictionary = self.loader.dictionary
        fields = []
        for key in ("default_time_field", "default_date_column", "time_column", "date_column"):
            value = dictionary.get(key)
            if isinstance(value, str):
                fields.append(value)
        terms = dictionary.get("terms", {})
        if isinstance(terms, dict):
            for name, definition in terms.items():
                if _normalize_identifier(name) in {"date", "time", "timestamp"}:
                    field = definition if isinstance(definition, str) else None
                    if isinstance(definition, dict):
                        field = definition.get("column")
                    if field:
                        fields.append(field)
        for section in ("columns", "definitions", "terms"):
            definitions = dictionary.get(section, {})
            if isinstance(definitions, dict):
                for name, definition in definitions.items():
                    if isinstance(definition, dict):
                        kind = str(definition.get("type") or definition.get("dtype") or definition.get("data_type") or "").lower()
                        if any(token in kind for token in ("date", "time")):
                            fields.append(definition.get("column", name))
        return list(dict.fromkeys(
            canonical for field in fields if isinstance(field, str)
            if (canonical := self.canonical_field(field)) in self.loader.sales.columns
        ))

    def resolve_date_column(
        self, requested: str | None = None, preferred_fields: list[str] | None = None,
    ) -> str | None:
        candidates = self.date_columns()

        def resolve(field: str) -> str | None:
            canonical = self.resolve_business_term(field)
            return canonical if canonical in candidates else None

        if requested:
            return resolve(requested)
        preferred = list(dict.fromkeys(
            field for name in preferred_fields or [] if (field := resolve(name))
        ))
        if preferred:
            return preferred[0] if len(preferred) == 1 else None
        dictionary = self.loader.dictionary
        for key in ("default_time_field", "default_date_column", "time_column", "date_column"):
            configured = dictionary.get(key)
            if isinstance(configured, str):
                return resolve(configured)
        aliases = list(dict.fromkeys(
            field for name in ("date", "time", "timestamp")
            if (field := resolve(name)) and name in dictionary.get("terms", {})
        ))
        if aliases:
            return aliases[0] if len(aliases) == 1 else None
        return candidates[0] if len(candidates) == 1 else None

    def dictionary_summary(self) -> dict[str, Any]:
        return {
            "terms": self.loader.dictionary.get("terms", {}),
            "definitions": self.loader.dictionary.get("definitions", {}),
        }

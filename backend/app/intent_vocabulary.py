"""Schema-aware business concepts shared by interpretation and validation.

This module classifies intent; it does not rewrite predicates, resolve metrics,
or authorize execution. Quoted values, explicit field references and known
category values are masked only in the classification copy. Callers must keep
the original query for filter parsing and semantic completeness checks.

Optional dictionary metadata can extend the vocabulary without adding query
branches::

    "semantic_vocabulary": {
        "target": ["sales objective"],
        "target_comparison": ["performance relative to"],
        "percentage": ["percent"]
    }

Presentation helpers remove only recognized terminal representation clauses.
Any unrecognized suffix, predicate or extra calculation remains in the query.
"""

import re

from .schema_manager import SchemaManager


class BusinessIntentVocabulary:
    _DEFAULTS = {
        "target": ("target", "targets", "goal", "goals", "objective", "objectives", "quota", "quotas"),
        "target_comparison": (
            "compare", "compared to", "compared with", "comparison", "against",
            "relative to", "versus", "vs", "achievement", "attainment",
            "achieve", "achieved", "achieves", "attain", "attained", "attains",
            "perform", "performed", "performs", "performance", "met", "meeting",
            "meet", "meets", "exceeded", "exceed", "exceeds", "reached", "reach", "reaches",
        ),
        "percentage": ("percentage", "percentages", "percent", "pct", "%"),
    }
    _CONCEPT_NAMES = {
        "target": ("target", "targets", "target_nouns"),
        "target_comparison": ("target_comparison", "comparison"),
        "percentage": ("percentage", "percent"),
    }

    def __init__(self, schema: SchemaManager) -> None:
        self.schema = schema
        dictionary = schema.loader.dictionary
        dictionary = dictionary if isinstance(dictionary, dict) else {}
        configured = dictionary.get("semantic_vocabulary", {})
        configured = configured if isinstance(configured, dict) else {}
        self.aliases: dict[str, tuple[str, ...]] = {}
        for concept, defaults in self._DEFAULTS.items():
            extras = []
            for name in self._CONCEPT_NAMES[concept]:
                values = configured.get(name, [])
                if isinstance(values, dict):
                    values = values.get("aliases", [])
                if isinstance(values, list):
                    extras.extend(value.strip().casefold() for value in values if isinstance(value, str) and value.strip())
            self.aliases[concept] = tuple(dict.fromkeys((*defaults, *extras)))

    @staticmethod
    def _phrase_pattern(phrases: tuple[str, ...] | list[str]) -> str:
        escaped = [re.escape(phrase).replace(r"\ ", r"\s+") for phrase in sorted(phrases, key=len, reverse=True)]
        return r"(?<![\w])(?:" + "|".join(escaped) + r")(?![\w])"

    def target_reference_pattern(self) -> str:
        """Return an escaped concept pattern, not a pattern for literal fields."""
        return self._phrase_pattern(self.aliases["target"])

    def _concept(self, concept: str, text: str) -> bool:
        return bool(re.search(self._phrase_pattern(self.aliases[concept]), text, re.IGNORECASE))

    def business_text(self, query: str) -> str:
        """Return a same-length classification copy with literal values hidden.

        Known fields are protected in explicit reference positions (predicates,
        grouping and quoted names), rather than removing a business word from
        every position. Thus ``goal = 5`` is a field while ``against its goal``
        remains a comparison request. Identifier fields such as sales_goal do
        not become concept words through underscore splitting.
        """
        text = re.sub(r"'(?:[^']*)'|\"(?:[^\"]*)\"", lambda match: " " * len(match.group()), query)
        columns = self.schema.available_columns()
        fields = list(dict.fromkeys([*columns["sales"], *columns["targets"]]))
        for field in sorted(fields, key=len, reverse=True):
            pattern = self._phrase_pattern([str(field)])
            for match in list(re.finditer(pattern, text, re.IGNORECASE)):
                before = text[:match.start()]
                after = text[match.end():]
                explicit = (
                    "_" in str(field)
                    or re.match(r"\s*(?:>=|<=|!=|==|>|<|=|\bis\b|\bequals\b|\bcontains\b)", after, re.IGNORECASE)
                    or re.search(r"\b(?:by|per|grouped\s+by|sorted\s+by|sum\s+of|mean\s+of|average\s+of|count\s+of)\s*$", before, re.IGNORECASE)
                )
                if explicit:
                    text = text[:match.start()] + " " * (match.end() - match.start()) + text[match.end():]
        # Category names are literal data, even when they coincide with a
        # business concept. Masking is conservative and never edits filters.
        values = [value for candidates in self.schema.categorical_index().values() for value in candidates]
        for value in sorted(set(values), key=len, reverse=True):
            if value:
                text = re.sub(self._phrase_pattern([value]), lambda match: " " * len(match.group()), text, flags=re.IGNORECASE)
        return text

    def target_intent(self, query: str) -> bool:
        text = self.business_text(query)
        if re.search(r"\b(?:achievement|attainment)\b", text, re.IGNORECASE):
            return True
        if not self._concept("target", text):
            return False
        # Goal/objective/quota are business targets when a measure, comparison,
        # assigned obligation or percentage accompanies them. A bare target
        # request is unambiguous inside the analytics vocabulary.
        if re.search(r"\btargets?\b", text, re.IGNORECASE):
            return True
        if self._concept("target_comparison", text) or self._concept("percentage", text):
            return True
        dictionary = self.schema.loader.dictionary
        dictionary = dictionary if isinstance(dictionary, dict) else {}
        metrics = dictionary.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        synonyms = dictionary.get("synonyms", {})
        synonyms = synonyms if isinstance(synonyms, dict) else {}
        measures = [*metrics, *synonyms, "sales", "revenue", "profit", "income", "assigned", "business"]
        return bool(re.search(self._phrase_pattern([str(value) for value in measures]), text, re.IGNORECASE))

    def target_comparison_intent(self, query: str) -> bool:
        text = self.business_text(query)
        return self.target_intent(query) and (
            self._concept("target_comparison", text) or self._concept("percentage", text)
        )

    def percentage_target(self, query: str) -> bool:
        return self.target_intent(query) and self._concept("percentage", self.business_text(query))

    def presentation_spans(self, query: str) -> list[tuple[int, int]]:
        """Locate a fully recognized formatting suffix, retaining constraints.

        The scalar percentage request must still be interpreted from the
        original text. These spans only prevent its representation clause from
        being mistaken for part of a grouping field or additional metric.
        """
        visible = re.sub(r"'(?:[^']*)'|\"(?:[^\"]*)\"", lambda match: " " * len(match.group()), query)
        start_pattern = r"\b(?:expressed|shown|displayed|reported|presented|formatted)\s+(?:as|in)\s+"
        representation = (
            r"(?:(?:a|the)\s+)?" + self._phrase_pattern(self.aliases["percentage"])
            + r"(?:\s+(?:of|relative\s+to|against)\s+"
            + r"(?:(?:that|this|its|their|the|an?|assigned|sales|revenue|business|performance)\s+)*"
            + r"(?:" + self.target_reference_pattern() + r"|(?:grand\s+)?total))?"
        )
        for match in re.finditer(start_pattern, visible, re.IGNORECASE):
            suffix = visible[match.end():].strip().rstrip(".?!").strip()
            if not re.fullmatch(representation, suffix, re.IGNORECASE):
                continue
            start = match.start()
            prefix = visible[:start]
            separator = re.search(r"(?:\s*[,;:]\s*|\s+and\s+)$", prefix, re.IGNORECASE)
            if separator:
                start = separator.start()
            return [(start, len(query))]
        return []

    def strip_presentation_clauses(self, query: str) -> str:
        """Remove safe formatting suffixes only; never use for filter parsing."""
        for start, end in reversed(self.presentation_spans(query)):
            query = query[:start] + query[end:]
        return query.rstrip(" ,;:")

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FilterPlan(BaseModel):
    column: str
    operator: Literal["=", "!=", ">", ">=", "<", "<=", "in", "contains"]
    value: Any


class FilterExpression(BaseModel):
    op: Literal["and", "or"]
    predicates: list[FilterPlan] = Field(default_factory=list)
    groups: list[FilterExpression] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_operands(self) -> FilterExpression:
        if len(self.predicates) + len(self.groups) < 2:
            raise ValueError("A Boolean filter expression requires at least two operands.")
        return self


class TimeFilterPlan(BaseModel):
    column: str
    start: str | None = None
    end: str | None = None
    year: int | None = None
    month: int | None = None
    quarter: int | None = None


class ComparisonPlan(BaseModel):
    kind: Literal["actual_vs_target", "difference", "achievement"]
    metric: str
    target_metric: str | None = None
    target_group: str | None = None
    target_column: str | None = None


class QueryPlan(BaseModel):
    operation: Literal[
        "aggregation", "filter", "ranking", "top_n_per_group",
        "contribution", "comparison", "nested", "time_series",
    ]
    metric: str | None = None
    aggregation: Literal["sum", "mean", "count", "min", "max"] = "sum"
    group_by: list[str] = Field(default_factory=list)
    filters: list[FilterPlan] = Field(default_factory=list)
    filter_expression: FilterExpression | None = None
    time_filter: TimeFilterPlan | None = None
    sort: Literal["asc", "desc"] = "desc"
    limit: int | None = Field(default=None, ge=1)
    top_n_per_group: int | None = Field(default=None, ge=1)
    rank_dimension: str | None = None
    partition_by: list[str] = Field(default_factory=list)
    limit_scope: Literal["global", "per_group"] = "global"
    comparison: ComparisonPlan | None = None
    contribution_percentage: bool = False
    requires_targets: bool = False
    output_fields: list[str] = Field(default_factory=list)
    interpretation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity: list[str] = Field(default_factory=list)
    is_supported: bool = True
    unresolved_terms: list[str] = Field(default_factory=list)
    resolved_fields: list[str] = Field(default_factory=list)
    semantic_errors: list[str] = Field(default_factory=list)

    @field_validator("group_by", "output_fields")
    @classmethod
    def deduplicate_fields(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class FeedbackRequest(BaseModel):
    query: str = Field(min_length=1)
    incorrect_interpretation: str = Field(min_length=1)
    correct_interpretation: str = Field(min_length=1)
    notes: str = ""

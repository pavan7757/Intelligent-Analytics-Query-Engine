from typing import Any

from .models import QueryPlan
from .boolean_filters import expression_predicates
from .schema_manager import SchemaManager


class QueryValidationError(ValueError):
    pass


class QueryValidator:
    def __init__(self, schema_manager: SchemaManager) -> None:
        self.schema_manager = schema_manager

    def validate(self, plan: QueryPlan) -> dict[str, Any]:
        checks = {
            "schema_valid": True,
            "metric_valid": False,
            "group_valid": True,
            "filter_valid": True,
            "time_valid": True,
            "target_available": True,
        }
        errors = []
        available = self.schema_manager.available_columns()
        sales_fields = set(available["sales"])
        targets_fields = set(available["targets"])

        if not plan.is_supported or plan.unresolved_terms or plan.semantic_errors:
            raise QueryValidationError(
                "; ".join([*plan.unresolved_terms, *plan.semantic_errors])
                or "The query contains unresolved analytical semantics."
            )

        if not plan.metric and plan.operation == "filter":
            checks["metric_valid"] = True
        elif not plan.metric:
            errors.append("A metric is required.")
        elif plan.metric == "revenue" and "revenue" in self.schema_manager.loader.dictionary.get("metrics", {}):
            checks["metric_valid"] = True
        elif plan.metric not in sales_fields and plan.metric not in targets_fields:
            canonical = self.schema_manager.resolve_business_term(plan.metric)
            if canonical is None:
                errors.append(f"Unknown metric or field: {plan.metric}")
            else:
                plan.metric = canonical
                checks["metric_valid"] = True
            checks["metric_valid"] = plan.metric in sales_fields

        for index, group in enumerate(plan.group_by):
            canonical = self.schema_manager.canonical_field(group)
            derived = self.schema_manager.resolve_derived_dimension(group)
            if canonical is None and derived is None:
                errors.append(f"Unknown grouping field: {group}")
                checks["group_valid"] = False
            else:
                plan.group_by[index] = canonical or derived or group

        predicates = [*plan.filters]
        if plan.filter_expression:
            predicates.extend(expression_predicates(plan.filter_expression))
        for filter_item in predicates:
            canonical = self.schema_manager.canonical_field(filter_item.column)
            if canonical is None and not self.schema_manager.is_numeric_field(filter_item.column):
                errors.append(f"Unknown filter field: {filter_item.column}")
                checks["filter_valid"] = False
            else:
                filter_item.column = canonical or filter_item.column

        if plan.time_filter:
            canonical = self.schema_manager.resolve_date_column(plan.time_filter.column)
            if canonical is None:
                errors.append(f"Unknown or invalid time field: {plan.time_filter.column}")
                checks["time_valid"] = False
            else:
                plan.time_filter.column = canonical
            if not any((plan.time_filter.start, plan.time_filter.end,
                        plan.time_filter.year, plan.time_filter.month,
                        plan.time_filter.quarter)) and not (
                plan.operation == "time_series"
                or any(self.schema_manager.resolve_derived_dimension(group) for group in plan.group_by)
            ):
                errors.append("A time filter requires a date, year, month, or quarter.")
                checks["time_valid"] = False

        if plan.requires_targets:
            if self.schema_manager.loader.targets.empty:
                errors.append("The requested comparison requires a non-empty targets dataset.")
                checks["target_available"] = False
            else:
                target_column = self.schema_manager.find_target_column(plan.metric or "")
                if target_column is None:
                    errors.append("No usable target field was found for the requested metric.")
                    checks["target_available"] = False
                else:
                    checks["target_available"] = True

        if errors:
            checks["schema_valid"] = False
            raise QueryValidationError("; ".join(errors))
        checks["metric_valid"] = True
        checks["group_valid"] = True
        checks["filter_valid"] = True
        checks["time_valid"] = True
        return {"checks": checks, "plan": plan}

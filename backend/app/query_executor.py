import math
from typing import Any

import pandas as pd

from .data_loader import DatasetLoader
from .boolean_filters import evaluate_filter_expression, expression_predicates, format_filter_expression
from .explanation import format_filter_predicate, format_time_filter
from .models import QueryPlan
from .schema_manager import SchemaManager


class QueryExecutionError(ValueError):
    pass


class QueryExecutor:
    _TIME_DIMENSIONS = {"month", "year", "quarter", "day", "week"}

    def __init__(self, loader: DatasetLoader) -> None:
        self.loader = loader
        self.schema = SchemaManager(loader)

    def execute(self, plan: QueryPlan) -> dict[str, Any]:
        try:
            execution = self._execute(plan)
            execution["result"] = self._safe_result(execution["result"])
            return execution
        except QueryExecutionError:
            raise
        except (KeyError, IndexError, ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise QueryExecutionError(
                "The analytical plan cannot be executed with the available fields and values. "
                "Check the requested dimensions, filters, and metric types."
            ) from exc

    def _execute(self, plan: QueryPlan) -> dict[str, Any]:
        if not plan.is_supported:
            raise QueryExecutionError(
                "This interpretation is not supported by the available dataset. "
                "Execution was blocked to avoid an unsupported substitution."
            )
        frame = self.loader.sales.copy()
        group_by = list(plan.group_by)
        partitions = list(plan.partition_by)
        if plan.operation == "top_n_per_group" and not partitions:
            rank_dimension = plan.rank_dimension or (group_by[-1] if group_by else None)
            partitions = [column for column in group_by if column != rank_dimension]
        if plan.operation in {"ranking", "top_n_per_group"}:
            group_by = list(dict.fromkeys([*partitions, *group_by]))
            if plan.rank_dimension and plan.rank_dimension not in group_by:
                group_by.append(plan.rank_dimension)
            if plan.limit_scope == "per_group" and not partitions:
                raise QueryExecutionError("Ranking within groups requires partition fields.")
        self._prepare_revenue(frame, plan)
        time_column = self._prepare_time_dimensions(frame, plan, group_by)
        missing_groups = set(group_by) - set(frame.columns)
        if missing_groups:
            raise QueryExecutionError(
                "Unknown grouping fields: " + ", ".join(sorted(missing_groups))
            )
        if plan.metric and plan.metric not in frame.columns:
            raise QueryExecutionError(f"The requested metric is unavailable: {plan.metric}")
        if plan.time_filter:
            if plan.time_filter.year:
                frame = frame[frame[time_column].dt.year == plan.time_filter.year]
            if plan.time_filter.month:
                frame = frame[frame[time_column].dt.month == plan.time_filter.month]
            if plan.time_filter.quarter:
                frame = frame[frame[time_column].dt.quarter == plan.time_filter.quarter]
            if not any((plan.time_filter.year, plan.time_filter.month,
                        plan.time_filter.quarter)):
                if plan.time_filter.start:
                    frame = frame[frame[time_column] >= pd.Timestamp(plan.time_filter.start)]
                if plan.time_filter.end:
                    frame = frame[frame[time_column] <= pd.Timestamp(plan.time_filter.end)]

        if plan.filter_expression:
            frame = frame[evaluate_filter_expression(frame, plan.filter_expression)]
        for filter_item in plan.filters:
            column = filter_item.column
            if column not in frame.columns:
                raise QueryExecutionError(f"The requested filter field is unavailable: {column}")
            if filter_item.operator == "in":
                frame = frame[frame[column].isin(filter_item.value)]
            elif filter_item.operator == "contains":
                frame = frame[frame[column].astype(str).str.contains(str(filter_item.value), case=False, na=False, regex=False)]
            elif filter_item.operator == "=":
                frame = frame[frame[column] == filter_item.value]
            elif filter_item.operator == "!=":
                frame = frame[frame[column] != filter_item.value]
            elif filter_item.operator == ">=":
                frame = frame[frame[column] >= filter_item.value]
            elif filter_item.operator == "<=":
                frame = frame[frame[column] <= filter_item.value]
            elif filter_item.operator == ">":
                frame = frame[frame[column] > filter_item.value]
            elif filter_item.operator == "<":
                frame = frame[frame[column] < filter_item.value]

        if plan.operation == "filter":
            result = frame.to_dict(orient="records")
        else:
            if not plan.metric:
                raise QueryExecutionError("A metric is required for this analytical operation.")
            if plan.operation == "time_series" and not any(
                column in self._TIME_DIMENSIONS for column in group_by
            ):
                frequency = "M" if plan.time_filter and plan.time_filter.month else "D"
                frame[time_column] = frame[time_column].dt.to_period(frequency).dt.start_time
                if time_column not in group_by:
                    group_by.insert(0, time_column)
            if group_by:
                for column in group_by:
                    if not (
                        pd.api.types.is_datetime64_any_dtype(frame[column])
                        or pd.api.types.is_numeric_dtype(frame[column])
                    ):
                        frame[column] = frame[column].astype(object).where(frame[column].notna(), "Unknown")
                grouped = frame.groupby(group_by, dropna=False, observed=False)
            else:
                grouped = frame
            if plan.aggregation == "count":
                if group_by:
                    result_frame = grouped.size().reset_index(name=plan.metric)
                else:
                    result_frame = pd.DataFrame({plan.metric: [len(frame)]})
            elif not group_by:
                result_frame = pd.DataFrame({plan.metric: [grouped[plan.metric].agg(plan.aggregation)]})
            else:
                result_frame = grouped[plan.metric].agg(plan.aggregation).reset_index()
            if plan.contribution_percentage:
                result_frame = self._contribution(result_frame, plan.metric)
            if plan.operation == "top_n_per_group":
                group_limit = plan.top_n_per_group or plan.limit
                if group_limit is None:
                    raise QueryExecutionError("Top N within groups requires a positive limit.")
                result_frame = self._rank_rows(result_frame, plan, partitions, group_limit)
            elif plan.operation == "ranking":
                result_frame = self._rank_rows(
                    result_frame, plan,
                    partitions if plan.limit_scope == "per_group" else [], plan.limit,
                )
            elif plan.limit:
                sort_column = plan.metric if plan.metric in result_frame.columns else result_frame.columns[-1]
                result_frame = result_frame.sort_values(by=sort_column, ascending=plan.sort == "asc", na_position="last").head(plan.limit)
            elif group_by:
                result_frame = result_frame.sort_values(by=group_by, na_position="last")
            result = result_frame.to_dict(orient="records")

        if plan.requires_targets:
            result = self._compare_with_targets(result, plan)
        executed_plan = QueryPlan.model_validate({
            **plan.model_dump(),
            "group_by": [] if plan.operation == "filter" else group_by,
            "partition_by": partitions,
        })
        return {
            "result": result,
            "generated_logic": self._generated_logic(executed_plan, time_column),
            "executed_plan": executed_plan,
        }

    def _generated_logic(self, plan: QueryPlan, time_column: str | None) -> str:
        steps = []
        predicates = [format_filter_predicate(item.model_dump()) for item in plan.filters]
        if plan.filter_expression:
            predicates.append(format_filter_expression(plan.filter_expression))
        time_predicate = format_time_filter(plan.time_filter.model_dump() if plan.time_filter else None)
        if time_predicate:
            predicates.append(time_predicate)
        if predicates:
            steps.append("filter: " + " AND ".join(predicates))
        if plan.operation == "filter":
            steps.append("return matching records without aggregation or grouping")
        else:
            if time_column:
                steps.append(f"time field: {time_column}")
            steps.append("group_by: " + (", ".join(plan.group_by) or "all matching records"))
            steps.append(f"{plan.aggregation}({plan.metric})")
            if plan.operation in {"ranking", "top_n_per_group"} or plan.limit is not None:
                steps.append(f"rank by {plan.metric} {plan.sort}")
                per_group = plan.operation == "top_n_per_group" or plan.limit_scope == "per_group"
                if per_group:
                    steps.append("partition_by: " + ", ".join(plan.partition_by))
                limit = plan.top_n_per_group or plan.limit if plan.operation == "top_n_per_group" else plan.limit
                steps.append(f"limit: {limit if limit is not None else 'all'} ({'per_group' if per_group else 'global'})")
            if plan.contribution_percentage:
                steps.append(f"percentage contribution to total {plan.metric} after filtering")
            if plan.requires_targets:
                comparison = plan.comparison
                target_metric = (comparison.target_metric or comparison.target_column) if comparison else None
                target_metric = target_metric or self.schema.find_target_column(plan.metric or "")
                steps.append(f"compare with target {target_metric}: actual, target, variance, achievement_percentage")
        if plan.ambiguity:
            steps.append("assumptions/ambiguities: " + "; ".join(plan.ambiguity))
        return " -> ".join(steps)

    def _prepare_revenue(self, frame: pd.DataFrame, plan: QueryPlan) -> None:
        predicates = [*plan.filters]
        if plan.filter_expression:
            predicates.extend(expression_predicates(plan.filter_expression))
        needs_revenue = plan.metric == "revenue" or any(item.column == "revenue" for item in predicates)
        if not needs_revenue or "revenue" in frame.columns:
            return
        required_columns = {"quantity", "unit_price", "discount"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise QueryExecutionError(
                "Revenue cannot be calculated because the required columns are missing: "
                + ", ".join(sorted(missing))
            )
        frame["revenue"] = (
            pd.to_numeric(frame["quantity"])
            * pd.to_numeric(frame["unit_price"])
            * (1 - pd.to_numeric(frame["discount"]))
        )

    def _prepare_time_dimensions(
        self, frame: pd.DataFrame, plan: QueryPlan, group_by: list[str],
    ) -> str | None:
        derived = [
            column for column in group_by
            if column in self._TIME_DIMENSIONS and column not in frame.columns
        ]
        needs_time = bool(derived or plan.time_filter or (
            plan.operation == "time_series"
            and not any(column in self._TIME_DIMENSIONS for column in group_by)
        ))
        if not needs_time:
            return None
        requested = plan.time_filter.column if plan.time_filter else None
        time_column = self.schema.resolve_date_column(
            requested=requested, preferred_fields=[*group_by, *plan.resolved_fields],
        )
        if time_column is None:
            candidates = self.schema.date_columns()
            if candidates and requested is None:
                raise QueryExecutionError(
                    "Time grouping is ambiguous. Specify a date field from: "
                    + ", ".join(candidates)
                )
            raise QueryExecutionError("No usable date/time field was found for this analytical plan.")
        dates = pd.to_datetime(frame[time_column], errors="coerce", format="mixed")
        if (frame[time_column].notna() & dates.isna()).any():
            raise QueryExecutionError(f"The time field contains invalid datetime values: {time_column}")
        if not frame.empty and not dates.notna().any():
            raise QueryExecutionError(f"The time field has no usable datetime values: {time_column}")
        frame[time_column] = dates
        for dimension in derived:
            if dimension == "year":
                frame[dimension] = dates.dt.year.astype("Int64")
            else:
                frequency = {"month": "M", "quarter": "Q", "week": "W", "day": "D"}[dimension]
                buckets = dates.dt.to_period(frequency)
                if dimension in {"day", "week"}:
                    frame[dimension] = buckets.dt.start_time.dt.strftime("%Y-%m-%d")
                else:
                    frame[dimension] = buckets.astype(str).where(dates.notna())
        return time_column

    @classmethod
    def _safe_result(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._safe_result(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._safe_result(item) for item in value]
        if value is None or pd.isna(value):
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, pd.Period):
            return str(value)
        return value

    def _contribution(self, frame: pd.DataFrame, metric: str) -> pd.DataFrame:
        total = frame[metric].sum()
        if total == 0:
            frame[f"{metric}_contribution_percentage"] = 0.0
            return frame
        frame[f"{metric}_contribution_percentage"] = frame[metric] / total * 100
        return frame

    def _rank_rows(
        self, frame: pd.DataFrame, plan: QueryPlan, partitions: list[str], limit: int | None,
    ) -> pd.DataFrame:
        if not plan.metric:
            raise QueryExecutionError("Ranking requires a metric field.")
        frame = frame.sort_values(
            by=[*partitions, plan.metric],
            ascending=[True] * len(partitions) + [plan.sort == "asc"],
            na_position="last",
            kind="stable",
        )
        if limit is None:
            return frame
        if partitions:
            return frame.groupby(partitions, dropna=False, observed=False).head(limit).reset_index(drop=True)
        return frame.head(limit).reset_index(drop=True)

    def _compare_with_targets(self, result: list[dict[str, Any]], plan: QueryPlan) -> list[dict[str, Any]]:
        if not plan.metric:
            raise QueryExecutionError("A target comparison requires an actual metric.")
        target_column = (
            plan.comparison.target_metric or plan.comparison.target_column
            if plan.comparison and (plan.comparison.target_metric or plan.comparison.target_column)
            else self.schema.find_target_column(plan.metric)
        )
        if self.loader.targets.empty:
            raise QueryExecutionError("The requested comparison requires a non-empty targets dataset.")
        if target_column not in self.loader.targets.columns:
            raise QueryExecutionError("No usable target field was found for the requested metric.")
        if plan.aggregation != "sum":
            raise QueryExecutionError(
                "The available targets represent additive totals; target comparison requires sum aggregation."
            )
        missing_groups = [column for column in plan.group_by if column not in self.loader.targets.columns]
        if missing_groups:
            raise QueryExecutionError(
                "The targets dataset cannot support the requested comparison grouping: "
                + ", ".join(missing_groups)
            )
        if plan.time_filter and format_time_filter(plan.time_filter.model_dump()):
            raise QueryExecutionError(
                "The targets dataset does not define a matching date field for this time-filtered comparison."
            )

        target = self.loader.targets.copy()
        if plan.filter_expression:
            predicates = expression_predicates(plan.filter_expression)
            if any(item.column not in target.columns or item.column == target_column for item in predicates):
                raise QueryExecutionError("The targets dataset cannot support the requested Boolean comparison constraints.")
            target = target[evaluate_filter_expression(target, plan.filter_expression)]
        for item in plan.filters:
            if item.column not in target.columns or item.column == target_column:
                raise QueryExecutionError(
                    "The targets dataset cannot support the requested comparison filter: " + item.column
                )
            values = target[item.column]
            if item.operator == "in":
                keep = values.isin(item.value)
            elif item.operator == "contains":
                keep = values.astype(str).str.contains(str(item.value), case=False, na=False, regex=False)
            else:
                comparisons = {
                    "=": values.eq, "!=": values.ne, ">": values.gt,
                    ">=": values.ge, "<": values.lt, "<=": values.le,
                }
                keep = comparisons[item.operator](item.value)
            target = target[keep]
        try:
            target[target_column] = pd.to_numeric(target[target_column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise QueryExecutionError("The target metric must contain numeric values.") from exc
        has_comparison_groups = bool(plan.group_by or (plan.comparison and plan.comparison.target_group))
        if not has_comparison_groups and not target.empty and target[target_column].isna().all():
            raise QueryExecutionError("The target metric contains missing values; a complete comparison cannot be calculated.")
        if not result:
            return result
        actual = pd.DataFrame(result)

        target_keys = [
            column for column in plan.group_by
            if column in target.columns
        ]
        target_group = plan.comparison.target_group if plan.comparison else None
        if target_group and target_group not in target_keys:
            if target_group not in target.columns:
                raise QueryExecutionError("The requested target grouping field is unavailable: " + target_group)
            target_keys.append(target_group)
        target_keys = list(dict.fromkeys(target_keys))
        if not target_keys:
            if len(actual) != 1:
                raise QueryExecutionError("An overall target comparison requires one aggregated actual total.")
            comparison = actual.copy()
            comparison["target"] = (
                target[target_column].sum(min_count=1)
                if not target[target_column].isna().any() else math.nan
            )
        else:
            grouped_targets = target.groupby(target_keys, dropna=False, observed=False)[target_column]
            target_totals = grouped_targets.sum(min_count=1)
            # A partially missing target total is unknown rather than an incomplete sum.
            target_totals = target_totals.where(grouped_targets.count() == grouped_targets.size())
            target = target_totals.reset_index()
            target = target.rename(columns={target_column: "target"})
            for key in target_keys:
                if key in actual.columns:
                    continue
                matching_filter = next(
                    (item for item in plan.filters if item.column == key and item.operator == "="),
                    None,
                )
                if matching_filter is None:
                    raise QueryExecutionError(
                        "Target comparison grain requires grouping or a single equality filter for: " + key
                    )
                actual[key] = matching_filter.value
            comparison = actual.merge(target, on=target_keys, how="left", validate="one_to_one")
        comparison["actual"] = comparison[plan.metric]
        comparison["variance"] = comparison["actual"] - comparison["target"]
        comparison["achievement_percentage"] = comparison["actual"] / comparison["target"].replace(0, math.nan) * 100
        comparison = comparison.drop(columns=[plan.metric])
        return comparison.to_dict(orient="records")

from typing import Any

from .boolean_filters import format_filter_expression
from .models import FilterExpression


def format_filter_predicate(filter_plan: dict[str, Any]) -> str:
    return (
        f"{filter_plan['column']} {filter_plan['operator']} "
        f"{filter_plan['value']!r}"
    )


def format_time_filter(time_filter: dict[str, Any] | None) -> str:
    if not time_filter:
        return ""
    column = time_filter["column"]
    calendar_parts = [
        f"{dimension}({column}) = {time_filter[dimension]}"
        for dimension in ("year", "month", "quarter")
        if time_filter.get(dimension) is not None
    ]
    if calendar_parts:
        return " AND ".join(calendar_parts)
    boundaries = []
    if time_filter.get("start"):
        boundaries.append(f"{column} >= {time_filter['start']!r}")
    if time_filter.get("end"):
        boundaries.append(f"{column} <= {time_filter['end']!r}")
    return " AND ".join(boundaries)


def _provenance(interpreter_source: str, fallback_used: bool) -> str:
    if interpreter_source == "gemini":
        return "The structured plan was supplied by Gemini GenAI."
    if interpreter_source == "openrouter":
        return (
            "The structured plan was supplied by OpenRouter GenAI after Gemini "
            "was unavailable or returned an invalid response."
        )
    if interpreter_source == "grok":
        return (
            "The structured plan was supplied by Grok GenAI after earlier GenAI "
            "providers were unavailable or returned invalid responses."
        )
    if interpreter_source == "genai" and not fallback_used:
        return "The structured plan was supplied by GenAI."
    return (
        "The structured plan was produced by the deterministic fallback parser "
        "because the GenAI provider chain was unavailable or invalid."
    )


def build_explanation(
    plan: dict[str, Any],
    query: str,
    interpreter_source: str = "genai",
    fallback_used: bool = False,
) -> str:
    if (
        not plan.get("is_supported", True)
        or plan.get("unresolved_terms")
        or plan.get("semantic_errors")
    ):
        reasons = list(dict.fromkeys([
            *plan.get("semantic_errors", []), *plan.get("unresolved_terms", []),
        ]))
        operation = plan.get("operation", "analytics")
        article = "an" if operation == "aggregation" else "a"
        parts = [
            f"I could not resolve '{query}' as {article} {operation} operation supported by the available dataset.",
            "No calculations were performed.",
        ]
        if reasons:
            parts.append("Unresolved requirements: " + "; ".join(reasons) + ".")
        elif not plan.get("ambiguity"):
            parts.append("The requested operation could not be represented safely by the available QueryPlan.")
        if plan.get("ambiguity"):
            parts.append("Ambiguities: " + "; ".join(plan["ambiguity"]) + ".")
        parts.append(_provenance(interpreter_source, fallback_used))
        return " ".join(parts)

    operation = plan.get("operation", "analytics")
    metric = plan.get("metric") or "requested metric"
    article = "an" if operation == "aggregation" else "a"
    parts = [f"I interpreted '{query}' as {article} {operation} operation."]
    predicates = [format_filter_predicate(item) for item in plan.get("filters", [])]
    if plan.get("filter_expression"):
        predicates.append(format_filter_expression(FilterExpression.model_validate(plan["filter_expression"])))
    time_predicate = format_time_filter(plan.get("time_filter"))
    if time_predicate:
        predicates.append(time_predicate)
    if predicates:
        parts.append("Applied filters: " + " AND ".join(predicates) + ".")

    group_by = list(plan.get("group_by", []))
    partitions = list(plan.get("partition_by", []))
    if operation in {"ranking", "top_n_per_group"}:
        if operation == "top_n_per_group" and not partitions:
            rank_dimension = plan.get("rank_dimension") or (group_by[-1] if group_by else None)
            partitions = [field for field in group_by if field != rank_dimension]
        group_by = list(dict.fromkeys([*partitions, *group_by]))
        if plan.get("rank_dimension") and plan["rank_dimension"] not in group_by:
            group_by.append(plan["rank_dimension"])

    if operation == "filter":
        parts.append("Matching records were returned.")
        parts.append("Pandas applied the validated predicates and returned the matching records.")
    else:
        calculation = f"{plan.get('aggregation', 'sum')}({metric})"
        if group_by:
            parts.append(f"Results were grouped by {', '.join(group_by)} and calculated with {calculation}.")
        else:
            parts.append(f"Calculated {calculation} across all matching records.")
        if plan.get("time_filter") and any(
            field in {"year", "month", "quarter", "week", "day"} for field in group_by
        ):
            parts.append(f"Calendar grouping used the date field {plan['time_filter']['column']}.")
        if operation in {"ranking", "top_n_per_group"} or plan.get("limit"):
            direction = "ascending" if plan.get("sort", "desc") == "asc" else "descending"
            limit = (
                plan.get("top_n_per_group") or plan.get("limit")
                if operation == "top_n_per_group" else plan.get("limit")
            )
            parts.append(f"Results were ranked by {metric} in {direction} order.")
            if partitions and (operation == "top_n_per_group" or plan.get("limit_scope") == "per_group"):
                parts.append(
                    f"Selected up to {limit or 'all'} results per group, "
                    f"partition_by: {', '.join(partitions)}."
                )
            elif limit:
                parts.append(f"Selected up to {limit} results with a global limit.")
            else:
                parts.append("All ranked results were returned without a limit.")
        if plan.get("contribution_percentage"):
            parts.append(f"Contribution percentages were calculated against total {metric} after filtering and aggregation.")
        parts.append("Pandas calculated the numerical values from the supplied data.")

    if plan.get("requires_targets"):
        comparison = plan.get("comparison") or {}
        target = comparison.get("target_metric") or comparison.get("target_column") or "the resolved target field"
        target_group = comparison.get("target_group") or ", ".join(group_by) or "all matching records"
        parts.append(
            f"Target comparison: {comparison.get('kind', 'actual_vs_target')} "
            f"for {comparison.get('metric') or metric} against {target}, using {target_group}."
        )
        parts.append(
            "Actual totals were matched to compatible target totals; variance is actual minus target, "
            "and achievement_percentage is actual divided by target times 100. "
            "Missing target matches have null target-derived values; zero targets have no achievement percentage."
        )
    if plan.get("ambiguity"):
        parts.append("Assumptions and ambiguities: " + "; ".join(plan["ambiguity"]) + ". Confidence was reduced.")
    parts.append(_provenance(interpreter_source, fallback_used))
    return " ".join(parts)

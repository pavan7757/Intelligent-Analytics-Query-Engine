import pytest

from app.confidence import calculate_confidence
from app.explanation import build_explanation, format_time_filter
from app.models import QueryPlan


VALID_CHECKS = {
    "metric_valid": True,
    "group_valid": True,
    "filter_valid": True,
    "time_valid": True,
    "target_available": True,
    "schema_valid": True,
}


@pytest.mark.parametrize("unsupported", [
    {"is_supported": False, "unresolved_terms": ["median"]},
    {"unresolved_terms": ["employee_count"]},
    {"semantic_errors": ["Unknown filter field: missing_field"]},
])
def test_unsupported_explanation_does_not_claim_execution(unsupported):
    plan = QueryPlan(operation="aggregation", metric="revenue", **unsupported)

    explanation = build_explanation(plan.model_dump(), "unsupported analytical request", "deterministic_fallback", True)

    assert "No calculations were performed" in explanation
    assert "Unresolved requirements:" in explanation
    assert "Pandas calculated" not in explanation
    assert "Results were grouped" not in explanation
    assert "deterministic fallback parser" in explanation
    assert calculate_confidence(plan.model_dump(), VALID_CHECKS) == 0.0


def test_filter_only_explanation_describes_predicate_and_matching_records():
    plan = QueryPlan(
        operation="filter", metric="quantity",
        filters=[{"column": "quantity", "operator": "=", "value": 2}],
    )

    explanation = build_explanation(plan.model_dump(), "Find matching records", "genai", False)

    assert "quantity = 2" in explanation
    assert "Matching records were returned" in explanation
    assert "Results were grouped" not in explanation
    assert "sum(" not in explanation
    assert "Pandas calculated the numerical values" not in explanation
    assert "GenAI" in explanation


def test_explanation_preserves_filters_grouping_partitioned_ranking_and_assumption():
    assumption = "The default metric is revenue because it is the dataset's primary metric"
    plan = QueryPlan(
        operation="ranking", metric="revenue", aggregation="sum",
        group_by=["product_name"], rank_dimension="product_name",
        partition_by=["region"], limit_scope="per_group", limit=2,
        filters=[
            {"column": "region", "operator": "in", "value": ["EMEA", "APAC"]},
            {"column": "quantity", "operator": ">=", "value": 2},
        ],
        ambiguity=[assumption],
    )

    explanation = build_explanation(plan.model_dump(), "Rank selected products", "deterministic_fallback", True)

    assert "region in ['EMEA', 'APAC']" in explanation
    assert "quantity >= 2" in explanation
    assert "grouped by region, product_name" in explanation
    assert "sum(revenue)" in explanation
    assert "ranked by revenue in descending order" in explanation
    assert "up to 2 results per group" in explanation
    assert "partition_by: region" in explanation
    assert assumption in explanation
    assert "deterministic fallback parser" in explanation


def test_top_n_operation_reports_effective_partition_and_limit():
    plan = QueryPlan(
        operation="top_n_per_group", metric="revenue",
        group_by=["region", "product_name"], rank_dimension="product_name",
        limit=8, top_n_per_group=3,
    )

    explanation = build_explanation(plan.model_dump(), "Rank within partitions")

    assert "up to 3 results per group" in explanation
    assert "partition_by: region" in explanation
    assert "up to 8" not in explanation


def test_ungrouped_target_explanation_describes_overall_comparison():
    plan = QueryPlan(
        operation="comparison", metric="revenue", requires_targets=True,
        comparison={"kind": "actual_vs_target", "metric": "revenue", "target_metric": "revenue_target"},
    )

    explanation = build_explanation(plan.model_dump(), "Compare metric to target")

    assert "Calculated sum(revenue) across all matching records" in explanation
    assert "Target comparison: actual_vs_target for revenue against revenue_target" in explanation
    assert "using all matching records" in explanation
    assert "joined" not in explanation


@pytest.mark.parametrize("aggregation", ["min", "max"])
def test_explanation_reports_actual_aggregation(aggregation):
    plan = QueryPlan(operation="aggregation", metric="revenue", aggregation=aggregation)

    explanation = build_explanation(plan.model_dump(), "Summarize metric")

    assert f"{aggregation}(revenue)" in explanation
    assert "sum(revenue)" not in explanation


def test_column_only_time_binding_is_not_reported_as_a_filter():
    plan = QueryPlan(
        operation="time_series", metric="revenue", group_by=["month"],
        time_filter={"column": "recorded_at"},
    )

    explanation = build_explanation(plan.model_dump(), "Group metric by month")

    assert "grouped by month" in explanation
    assert "Calendar grouping used the date field recorded_at" in explanation
    assert "Applied filters:" not in explanation
    assert format_time_filter(plan.time_filter.model_dump()) == ""


@pytest.mark.parametrize("time_filter,predicate", [
    ({"column": "recorded_at", "year": 2025, "month": 2}, "year(recorded_at) = 2025 AND month(recorded_at) = 2"),
    ({"column": "recorded_at", "quarter": 3}, "quarter(recorded_at) = 3"),
    ({"column": "recorded_at", "start": "2025-01-01", "end": "2025-01-31"}, "recorded_at >= '2025-01-01' AND recorded_at <= '2025-01-31'"),
])
def test_explanation_describes_actual_time_predicates(time_filter, predicate):
    plan = QueryPlan(operation="aggregation", metric="revenue", time_filter=time_filter)

    explanation = build_explanation(plan.model_dump(), "Summarize selected period")

    assert predicate in explanation


def test_low_interpretation_confidence_cannot_saturate_validated_score():
    plan = QueryPlan(operation="aggregation", metric="revenue", interpretation_confidence=0.0)

    assert calculate_confidence(plan.model_dump(), VALID_CHECKS) <= 0.5
    plan.interpretation_confidence = 0.2
    low_score = calculate_confidence(plan.model_dump(), VALID_CHECKS)
    plan.interpretation_confidence = 0.9
    high_score = calculate_confidence(plan.model_dump(), VALID_CHECKS)
    assert low_score < high_score
    assert high_score >= 0.9


def test_clear_fallback_is_high_confidence_and_ambiguity_reduces_it():
    plan = QueryPlan(operation="aggregation", metric="revenue", interpretation_confidence=0.65)
    clear_score = calculate_confidence(plan.model_dump(), VALID_CHECKS)
    plan.ambiguity = ["A default metric was assumed"]
    ambiguous_score = calculate_confidence(plan.model_dump(), VALID_CHECKS)

    assert clear_score >= 0.8
    assert ambiguous_score < clear_score


@pytest.mark.parametrize("failed_check", list(VALID_CHECKS))
def test_failed_validation_never_produces_high_confidence(failed_check):
    plan = QueryPlan(operation="aggregation", metric="revenue", interpretation_confidence=1.0)
    checks = {**VALID_CHECKS, failed_check: False}

    assert calculate_confidence(plan.model_dump(), checks) == 0.0


def test_empty_result_confidence_remains_limited():
    plan = QueryPlan(operation="aggregation", metric="revenue", interpretation_confidence=1.0)

    assert calculate_confidence(plan.model_dump(), VALID_CHECKS, has_results=False) <= 0.5

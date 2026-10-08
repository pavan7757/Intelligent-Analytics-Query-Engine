import pandas as pd

from app.data_loader import DatasetLoader
from app.models import FilterPlan
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter


def make_loader() -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "region": ["APAC", "EMEA", "APAC", "EMEA"],
        "country": ["India", "Germany", "India", "Germany"],
        "city": ["Mumbai", "Berlin", "Delhi", "Paris"],
        "customer_segment": ["Consumer", "Corporate", "Home Office", "Consumer"],
        "product_category": ["Technology", "Technology", "Furniture", "Furniture"],
        "product_name": ["Dell XPS", "MacBook Air", "Ergo Chair", "iPhone 14"],
        "quantity": [1, 2, 3, 4],
        "revenue": [100.0, 200.0, 300.0, 400.0],
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {
        "metrics": {"revenue": "quantity * unit_price * (1 - discount)"},
        "dimensions": [
            "region",
            "country",
            "city",
            "customer_segment",
            "product_category",
            "product_name",
        ],
        "synonyms": {"sales": "revenue", "income": "revenue"},
    }
    loader.queries = []
    loader.feedback_path = None
    return loader


def interpret(query: str):
    interpreter = QueryInterpreter(make_loader())
    interpreter.client = None
    return interpreter.interpret(query)


def test_implicit_and_explicit_filters_are_structured():
    implicit = interpret("Show total revenue for EMEA.")
    explicit = interpret("Show revenue where region is EMEA.")

    assert implicit.filters == [FilterPlan(column="region", operator="=", value="EMEA")]
    assert explicit.filters == [FilterPlan(column="region", operator="=", value="EMEA")]
    assert implicit.is_supported is True
    assert explicit.is_supported is True


def test_implicit_filter_uses_data_evidence_for_a_different_dimension():
    plan = interpret("Show total revenue for Technology.")

    assert plan.filters == [FilterPlan(column="product_category", operator="=", value="Technology")]
    assert plan.is_supported is True


def test_filter_and_grouping_remain_separate():
    plan = interpret("Show revenue for EMEA by product category.")

    assert plan.filters == [FilterPlan(column="region", operator="=", value="EMEA")]
    assert plan.group_by == ["product_category"]
    assert plan.metric == "revenue"
    assert plan.aggregation == "sum"


def test_multiple_filters_compose():
    plan = interpret("Show revenue for EMEA by product category where product category is Technology.")

    assert plan.filters == [
        FilterPlan(column="region", operator="=", value="EMEA"),
        FilterPlan(column="product_category", operator="=", value="Technology"),
    ]
    assert plan.group_by == ["product_category"]


def test_numeric_comparison_filters_are_generic():
    greater = interpret("Show products with revenue greater than 200.")
    less_equal = interpret("Show products with revenue less than or equal to 300.")

    assert greater.filters == [FilterPlan(column="revenue", operator=">", value=200)]
    assert less_equal.filters == [FilterPlan(column="revenue", operator="<=", value=300)]


def test_unknown_filter_value_is_rejected_without_substitution():
    plan = interpret("Show total revenue for NotARealRegion.")

    assert plan.is_supported is False
    assert plan.filters == []
    assert plan.interpretation_confidence == 0.0
    assert "NotARealRegion" in plan.unresolved_terms


def test_ambiguous_value_does_not_guess():
    loader = make_loader()
    loader.sales["name"] = ["Shared", "Shared", "Shared", "Shared"]
    loader.sales["other_name"] = ["Shared", "Shared", "Shared", "Shared"]
    loader.dictionary["dimensions"] = [
        "region", "product_category", "product_name", "name", "other_name"
    ]
    interpreter = QueryInterpreter(loader)
    interpreter.client = None
    plan = interpreter.interpret("Show revenue for Shared.")

    assert plan.is_supported is False
    assert plan.filters == []
    assert plan.interpretation_confidence == 0.0


def test_filtering_applies_before_grouping_and_contribution():
    loader = make_loader()
    loader.sales["revenue"] = [100.0, 200.0, 300.0, 400.0]
    interpreter = QueryInterpreter(loader)
    interpreter.client = None
    plan = interpreter.interpret("What percentage of EMEA revenue came from each product category?")

    assert plan.operation == "contribution"
    assert plan.filters == [FilterPlan(column="region", operator="=", value="EMEA")]
    assert plan.group_by == ["product_category"]
    assert plan.metric == "revenue"
    assert plan.aggregation == "sum"

    result = QueryExecutor(loader).execute(plan)["result"]
    assert {row["product_category"] for row in result} == {"Technology", "Furniture"}
    assert abs(sum(row["revenue_contribution_percentage"] for row in result) - 100.0) < 1e-9


def test_top_n_filter_and_partition_semantics_are_preserved():
    filtered = interpret("Show the top 3 products by revenue in EMEA.")
    partitioned = interpret("Show the top 2 products by revenue in each region.")

    assert filtered.filters == [FilterPlan(column="region", operator="=", value="EMEA")]
    assert filtered.group_by == ["product_name"]
    assert filtered.limit == 3
    assert filtered.limit_scope == "global"
    assert partitioned.partition_by == ["region"]
    assert partitioned.rank_dimension == "product_name"
    assert partitioned.limit == 2
    assert partitioned.limit_scope == "per_group"

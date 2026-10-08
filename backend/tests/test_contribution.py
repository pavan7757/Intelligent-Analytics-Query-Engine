import math

import pandas as pd

from app.data_loader import DatasetLoader
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter


def make_loader() -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "region": ["APAC", "EMEA", "Unknown", "APAC", "EMEA", "Unknown"],
        "product_name": ["A", "A", "A", "B", "B", "B"],
        "product_category": ["Technology", "Furniture", "Office Supplies", "Technology", "Furniture", "Office Supplies"],
        "revenue": [100.0, 50.0, 25.0, 200.0, 100.0, 50.0],
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {
        "metrics": {"revenue": "quantity * unit_price * (1 - discount)"},
        "dimensions": ["region", "product_name", "product_category"],
        "synonyms": {"sales": "revenue", "income": "revenue"},
    }
    loader.queries = []
    loader.feedback_path = None
    return loader


def interpret(query: str):
    interpreter = QueryInterpreter(make_loader())
    interpreter.client = None
    return interpreter.interpret(query)


def test_contribution_by_product_category_groups_and_percentages():
    query = "What percentage of total revenue was contributed by each product category?"
    plan = interpret(query)

    assert plan.operation == "contribution"
    assert plan.metric == "revenue"
    assert plan.aggregation == "sum"
    assert plan.group_by == ["product_category"]
    assert plan.filters == []
    assert plan.is_supported is True
    assert plan.interpretation_confidence > 0.0

    result = QueryExecutor(make_loader()).execute(plan)["result"]
    assert {row["product_category"] for row in result} == {
        "Technology", "Furniture", "Office Supplies"
    }
    assert abs(sum(row["revenue_contribution_percentage"] for row in result) - 100.0) < 1e-9
    assert all(
        row["revenue_contribution_percentage"] == row["revenue"] / 525.0 * 100
        for row in result
    )


def test_contribution_dimensions_are_resolved_without_partial_matching():
    for query, expected_group in [
        ("What percentage of revenue was contributed by each region?", "region"),
        ("What percentage of revenue came from each product?", "product_name"),
        ("What share of total revenue came from each product category?", "product_category"),
    ]:
        plan = interpret(query)
        assert plan.operation == "contribution"
        assert plan.group_by == [expected_group]
        assert plan.metric == "revenue"
        assert plan.aggregation == "sum"
        assert plan.filters == []


def test_contribution_zero_total_is_safe():
    loader = make_loader()
    loader.sales["revenue"] = 0.0
    interpreter = QueryInterpreter(loader)
    interpreter.client = None
    plan = interpreter.interpret("What percentage of revenue was contributed by each product category?")
    result = QueryExecutor(loader).execute(plan)["result"]

    assert result
    assert all(row["revenue_contribution_percentage"] == 0.0 for row in result)


def test_contribution_preserves_missing_group_values():
    loader = make_loader()
    loader.sales.loc[0, "product_category"] = None
    interpreter = QueryInterpreter(loader)
    interpreter.client = None
    plan = interpreter.interpret("What percentage of revenue was contributed by each product category?")
    result = QueryExecutor(loader).execute(plan)["result"]

    assert any(row["product_category"] == "Unknown" for row in result)


def test_product_and_product_category_remain_distinct():
    plan = interpret("What percentage of revenue was contributed by each product?")
    assert plan.group_by == ["product_name"]

    category_plan = interpret("What percentage of revenue was contributed by each product category?")
    assert category_plan.group_by == ["product_category"]
    assert category_plan.group_by != plan.group_by


def test_contribution_helper_handles_zero_total_without_division_error():
    loader = make_loader()
    loader.sales["revenue"] = 0.0
    interpreter = QueryInterpreter(loader)
    interpreter.client = None
    plan = interpreter.interpret("What percentage of revenue was contributed by each product category?")
    result = QueryExecutor(loader).execute(plan)["result"]

    assert result
    assert all(math.isfinite(row["revenue_contribution_percentage"]) for row in result)


def test_required_existing_analytics_cases_remain_unchanged():
    loader = DatasetLoader()
    interpreter = QueryInterpreter(loader)
    interpreter.client = None
    executor = QueryExecutor(loader)

    cases = [
        ("What is the total revenue?", lambda result: result == [{"revenue": 6134.4}]),
        (
            "Show total revenue by region.",
            lambda result: {
                row["region"]: row["revenue"]
                for row in result
            } == {"APAC": 474.0, "EMEA": 2398.0, "Unknown": 3262.4},
        ),
        (
            "Show the top 5 products by revenue.",
            lambda result: [row["product_name"] for row in result]
            == ["Dell XPS", "Samsung Galaxy", "MacBook Air", "iPhone 14", "Ergo Chair"],
        ),
        (
            "Which five products generated the most revenue?",
            lambda result: [row["product_name"] for row in result]
            == ["Dell XPS", "Samsung Galaxy", "MacBook Air", "iPhone 14", "Ergo Chair"],
        ),
        (
            "Which product category generated the highest revenue?",
            lambda result: result == [{"product_category": "Technology", "revenue": 5402.0}],
        ),
    ]

    for query, assertion in cases:
        plan = interpreter.interpret(query)
        assert plan.is_supported is True, (query, plan.model_dump())
        result = executor.execute(plan)["result"]
        assert assertion(result), (query, result)

    unsupported = interpreter.interpret("What is the average temperature on Mars by city?")
    assert unsupported.is_supported is False
    assert unsupported.confidence if False else unsupported.interpretation_confidence == 0.0

    top_two = interpreter.interpret("Show the top 2 products by revenue in each region.")
    assert top_two.operation == "ranking"
    assert top_two.metric == "revenue"
    assert top_two.aggregation == "sum"
    assert top_two.rank_dimension == "product_name"
    assert top_two.partition_by == ["region"]
    assert top_two.limit == 2
    assert top_two.limit_scope == "per_group"
    top_two_result = executor.execute(top_two)["result"]
    assert all(
        sum(row["region"] == region for row in top_two_result) <= 2
        for region in {row["region"] for row in top_two_result}
    )

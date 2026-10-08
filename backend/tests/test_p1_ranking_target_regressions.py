import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data_loader import DatasetLoader
from app.models import ComparisonPlan, FilterPlan, QueryPlan
from app.query_executor import QueryExecutionError, QueryExecutor
from app.query_interpreter import QueryInterpreter
from app.query_validator import QueryValidator
from app.schema_manager import SchemaManager


main_module = importlib.import_module("app.main")


def make_loader():
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "region": ["EMEA", "APAC", "EMEA", "APAC", "EMEA"],
        "product_name": ["A", "B", "C", "D", "B"],
        "revenue": [100.0, 90.0, 80.0, 70.0, 10.0],
        "quantity": [1, 2, 3, 1, 2],
    })
    loader.targets = pd.DataFrame({
        "region": ["EMEA", "APAC"],
        "target_revenue": [300.0, 200.0],
    })
    loader.dictionary = {
        "terms": {"revenue": "revenue"},
        "dimensions": ["region", "product_name"],
    }
    loader.queries = []
    loader.feedback_path = None
    return loader


def fallback(query, loader=None):
    loader = loader or make_loader()
    interpreter = QueryInterpreter(loader)
    interpreter.client = None
    return interpreter.interpret(query)


@pytest.mark.parametrize("phrase,limit", [
    ("Three highest revenue products", 3),
    ("3 highest revenue products", 3),
    ("Show top two products by revenue", 2),
    ("Show twenty-one highest revenue products", 21),
    ("Show top twenty one products by revenue", 21),
    ("Show one hundred and five highest revenue products", 105),
    ("Show top one hundred products by revenue", 100),
])
def test_explicit_number_precedes_ranking_superlative(phrase, limit):
    plan = fallback(phrase)
    assert plan.is_supported is True, plan.model_dump()
    assert plan.group_by == ["product_name"]
    assert plan.limit == limit


@pytest.mark.parametrize("word,number", [
    ("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5),
    ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9), ("ten", 10),
])
def test_number_words_before_highest_are_generic(word, number):
    assert fallback(f"Show {word} highest revenue products").limit == number


@pytest.mark.parametrize("query,provider_limit,expected", [
    ("Three highest revenue products", 3, 3),
    ("Three highest revenue products", 1, 3),
    ("Show highest revenue products", 3, 3),
])
def test_valid_provider_limit_is_not_replaced_by_implicit_one(monkeypatch, query, provider_limit, expected):
    interpreter = QueryInterpreter(make_loader())
    interpreter.client = object()
    monkeypatch.setattr(interpreter, "_call_llm", lambda context: {
        "operation": "ranking", "metric": "revenue", "aggregation": "sum",
        "group_by": ["product_name"], "limit": provider_limit,
        "rank_dimension": "product_name", "interpretation_confidence": 0.95,
    })
    plan = interpreter.interpret(query)
    assert interpreter.interpreter_source == "gemini"
    assert plan.limit == expected


def test_top_n_revenue_threshold_does_not_become_limit():
    interpreter = QueryInterpreter(make_loader())
    assert interpreter._extract_number("Show highest revenue products where revenue >= 500".lower()) is None


def test_three_highest_products_executes_three_aggregated_rows():
    loader = make_loader()
    plan = fallback("Three highest revenue products", loader)
    QueryValidator(SchemaManager(loader)).validate(plan)
    rows = QueryExecutor(loader).execute(plan)["result"]
    assert len(rows) == 3
    assert [row["revenue"] for row in rows] == sorted(
        loader.sales.groupby("product_name")["revenue"].sum(), reverse=True,
    )[:3]


def test_overall_target_comparison_rolls_up_without_join_keys():
    loader = make_loader()
    plan = fallback("Compare total revenue against target", loader)
    assert plan.operation == "comparison"
    assert plan.group_by == []
    assert plan.comparison.target_group is None
    QueryValidator(SchemaManager(loader)).validate(plan)
    rows = QueryExecutor(loader).execute(plan)["result"]
    assert rows == [{
        "actual": loader.sales["revenue"].sum(),
        "target": loader.targets["target_revenue"].sum(),
        "variance": loader.sales["revenue"].sum() - loader.targets["target_revenue"].sum(),
        "achievement_percentage": loader.sales["revenue"].sum() / loader.targets["target_revenue"].sum() * 100,
    }]


def test_overall_target_comparison_with_single_target_value():
    loader = make_loader()
    loader.targets = pd.DataFrame({"target_revenue": [500.0]})
    plan = fallback("Compare total revenue against target", loader)
    assert plan.is_supported is True
    assert QueryExecutor(loader).execute(plan)["result"][0]["target"] == 500.0


def target_plan(**updates):
    return QueryPlan(
        operation="comparison", metric="revenue", requires_targets=True,
        comparison=ComparisonPlan(kind="actual_vs_target", metric="revenue", target_metric="target_revenue"),
        **updates,
    )


@pytest.mark.parametrize("targets,message", [
    (pd.DataFrame(), "non-empty targets"),
    (pd.DataFrame({"region": ["EMEA"]}), "usable target field"),
    (pd.DataFrame({"target_revenue": ["not a number"]}), "numeric values"),
    (pd.DataFrame({"target_revenue": [None]}), "missing values"),
])
def test_unusable_targets_raise_controlled_analytical_errors(targets, message):
    loader = make_loader()
    loader.targets = targets
    with pytest.raises(QueryExecutionError, match=message):
        QueryExecutor(loader).execute(target_plan())


def test_target_comparison_rejects_unsupported_grouping_grain():
    with pytest.raises(QueryExecutionError, match="comparison grouping: product_name"):
        QueryExecutor(make_loader()).execute(target_plan(group_by=["product_name"]))


def test_target_comparison_preserves_equal_filter_for_both_actual_and_target():
    loader = make_loader()
    plan = fallback("Compare revenue against target for EMEA", loader)
    rows = QueryExecutor(loader).execute(plan)["result"]
    assert len(rows) == 1
    assert rows[0]["region"] == "EMEA"
    assert rows[0]["actual"] == loader.sales.loc[loader.sales["region"] == "EMEA", "revenue"].sum()
    assert rows[0]["target"] == loader.targets.loc[loader.targets["region"] == "EMEA", "target_revenue"].sum()


def test_target_comparison_rejects_filter_without_matching_target_grain():
    plan = target_plan(filters=[FilterPlan(column="quantity", operator=">", value=1)])
    with pytest.raises(QueryExecutionError, match="comparison filter: quantity"):
        QueryExecutor(make_loader()).execute(plan)


def test_target_comparison_rejects_non_additive_aggregation():
    with pytest.raises(QueryExecutionError, match="requires sum aggregation"):
        QueryExecutor(make_loader()).execute(target_plan(aggregation="mean"))


@pytest.mark.parametrize("unsupported_grain", [False, True])
def test_target_comparison_endpoint_returns_result_or_controlled_422(monkeypatch, unsupported_grain):
    loader = make_loader()
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    if unsupported_grain:
        monkeypatch.setattr(QueryInterpreter, "interpret", lambda self, query, feedback: target_plan(group_by=["product_name"]))
    with TestClient(main_module.app) as client:
        response = client.post("/query", json={"query": "Compare total revenue against target"})
    if unsupported_grain:
        assert response.status_code == 422, response.text
        assert "comparison grouping" in response.json()["detail"]
    else:
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["query_plan"]["group_by"] == []
        assert payload["result"][0]["actual"] == loader.sales["revenue"].sum()
        assert payload["result"][0]["target"] == loader.targets["target_revenue"].sum()

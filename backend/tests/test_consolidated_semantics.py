"""Generic aggregation/ranking regression coverage without provider network calls."""

import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data_loader import DatasetLoader
from app.models import QueryPlan
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter
from app.query_validator import QueryValidator


main_module = importlib.import_module("app.main")


@pytest.fixture
def loader():
    value = object.__new__(DatasetLoader)
    value.sales = pd.DataFrame({
        "order_id": list(range(1, 10)),
        "ordered_at": pd.to_datetime([
            "2023-02-01", "2023-08-01", "2023-12-31", "2024-01-01",
            "2024-05-11", "2024-06-14", "2025-01-01", "2025-04-02", "2025-06-04",
        ]),
        "region": ["EMEA", "EMEA", "EMEA", "APAC", "APAC", "APAC", "AMER", "AMER", "AMER"],
        "product_name": ["Widget", "Chair", "Cable", "Widget", "Chair", "Cable", "Widget", "Chair", "Cable"],
        "product_category": ["Tools", "Home", "Tools", "Tools", "Home", "Tools", "Tools", "Home", "Tools"],
        "customer_id": ["C1", "C2", "C3", "C1", "C4", "C5", "C2", "C5", "C6"],
        "customer_segment": ["Business", "Consumer", "Business", "Business", "Consumer", "Business", "Consumer", "Business", "Business"],
        "quantity": [1, 2, 4, 2, 5, 3, 4, 1, 2],
        "unit_price": [19.0, 27.0, 13.0, 23.0, 17.0, 29.0, 31.0, 37.0, 41.0],
        "revenue": [19.0, 54.0, 52.0, 46.0, 85.0, 87.0, 124.0, 37.0, 82.0],
    })
    value.targets = pd.DataFrame()
    value.dictionary = {
        "metrics": {"revenue": "revenue"},
        "dimensions": ["region", "product_name", "product_category", "customer_segment"],
        "default_time_field": "ordered_at",
    }
    return value


@pytest.fixture(autouse=True)
def prohibit_provider_requests(monkeypatch):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)

    def prohibited(*args, **kwargs):
        raise AssertionError("Regression tests must not make provider requests")

    monkeypatch.setattr(QueryInterpreter, "_call_llm", prohibited)


def interpret_and_execute(loader, query):
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret(query)
    assert plan.is_supported, plan.model_dump()
    assert interpreter.interpreter_source == "deterministic_fallback"
    QueryValidator(interpreter.schema).validate(plan)
    return plan, QueryExecutor(loader).execute(plan)


@pytest.mark.parametrize("metric", ["revenue", "quantity", "unit_price"])
@pytest.mark.parametrize("word,aggregation", [
    ("average", "mean"), ("avg", "mean"), ("mean", "mean"),
    ("minimum", "min"), ("min", "min"), ("maximum", "max"),
    ("max", "max"), ("total", "sum"), ("sum", "sum"),
])
def test_aggregation_vocabulary_is_separate_from_numeric_metric(loader, metric, word, aggregation):
    plan, execution = interpret_and_execute(loader, f"What is the {word} {metric}?")
    assert plan.metric == metric
    assert plan.aggregation == aggregation
    assert plan.group_by == []
    assert plan.limit is None
    expected = loader.sales[metric].agg(aggregation)
    assert execution["result"] == [{metric: pytest.approx(expected)}]
    assert f"{aggregation}({metric})" in execution["generated_logic"]


@pytest.mark.parametrize("query", [
    "What is the average revenue for all orders?",
    "Calculate mean revenue across all records.",
    "Show average revenue for all rows.",
])
def test_all_records_scope_is_not_an_unknown_field(loader, query):
    plan, execution = interpret_and_execute(loader, query)
    assert plan.metric == "revenue"
    assert plan.aggregation == "mean"
    assert plan.group_by == []
    assert not plan.filters and plan.filter_expression is None
    assert execution["result"] == [{"revenue": pytest.approx(loader.sales.revenue.mean())}]


@pytest.mark.parametrize("query,ascending", [
    ("Rank product categories by revenue from highest to lowest.", False),
    ("Show product categories ordered by revenue descending.", False),
    ("Show revenue by product category from largest to smallest.", False),
    ("Sort product categories by revenue in descending order.", False),
    ("Rank product categories by revenue from lowest to highest.", True),
    ("Show product categories ordered by revenue ascending.", True),
    ("Show revenue by product category from smallest to largest.", True),
])
def test_ranking_direction_does_not_invent_a_limit(loader, query, ascending):
    plan, execution = interpret_and_execute(loader, query)
    assert plan.operation == "ranking"
    assert plan.group_by == ["product_category"]
    assert plan.rank_dimension == "product_category"
    assert plan.sort == ("asc" if ascending else "desc")
    assert plan.limit is None and plan.top_n_per_group is None
    expected = loader.sales.groupby("product_category").revenue.sum().sort_values(ascending=ascending)
    assert [row["product_category"] for row in execution["result"]] == list(expected.index)
    assert [row["revenue"] for row in execution["result"]] == pytest.approx(list(expected))


@pytest.mark.parametrize("query", [
    "Show revenue by product category for EMEA from highest to lowest.",
    "Show revenue by product category only for EMEA and rank the categories from highest to lowest.",
    "Rank product categories by revenue for EMEA in descending order.",
])
def test_categorical_filter_and_sort_suffix_remain_separate(loader, query):
    plan, execution = interpret_and_execute(loader, query)
    assert plan.group_by == ["product_category"]
    assert plan.sort == "desc" and plan.limit is None
    expected = loader.sales.loc[loader.sales.region == "EMEA"].groupby("product_category").revenue.sum().sort_values(ascending=False)
    assert [row["product_category"] for row in execution["result"]] == list(expected.index)
    assert [row["revenue"] for row in execution["result"]] == pytest.approx(list(expected))
    assert "EMEA" in execution["generated_logic"]


@pytest.mark.parametrize("query,ascending", [
    ("Show top three product categories by revenue from highest to lowest.", False),
    ("Show bottom two product categories by revenue from lowest to highest.", True),
])
def test_explicit_top_n_survives_direction_phrases(loader, query, ascending):
    plan, execution = interpret_and_execute(loader, query)
    assert plan.limit == (2 if ascending else 3)
    assert plan.sort == ("asc" if ascending else "desc")
    expected = loader.sales.groupby("product_category").revenue.sum().sort_values(ascending=ascending).head(plan.limit)
    assert [row["product_category"] for row in execution["result"]] == list(expected.index)


@pytest.mark.parametrize("query,dimension", [
    ("Show the top 2 products by revenue in each region.", "product_name"),
    ("List the two highest revenue customers per region.", "customer_id"),
    ("Show top two customer segments by revenue in each region.", "customer_segment"),
])
def test_per_group_ranking_preserves_entity_granularity(loader, query, dimension):
    plan, execution = interpret_and_execute(loader, query)
    assert plan.rank_dimension == dimension
    assert plan.group_by == ["region", dimension]
    assert plan.partition_by == ["region"]
    assert plan.limit == 2 and plan.limit_scope == "per_group"
    totals = loader.sales.groupby(["region", dimension]).revenue.sum().reset_index()
    expected = totals.sort_values("revenue", ascending=False).groupby("region", sort=False).head(2)
    assert {(row["region"], row[dimension]): row["revenue"] for row in execution["result"]} == {
        (row["region"], row[dimension]): row["revenue"] for row in expected.to_dict(orient="records")
    }


@pytest.mark.parametrize("query", [
    "Show total revenue for EMEA or APAC.",
    "Sum revenue where region is EMEA or region is APAC.",
])
def test_boolean_or_filter_preserved_after_ranking_changes(loader, query):
    plan, execution = interpret_and_execute(loader, query)
    assert plan.filter_expression is not None and plan.filter_expression.op == "or"
    expected = loader.sales.loc[loader.sales.region.isin(["EMEA", "APAC"]), "revenue"].sum()
    assert execution["result"] == [{"revenue": pytest.approx(expected)}]


@pytest.mark.parametrize("query", [
    "Show total revenue where quantity is not 2.",
    "Sum revenue where quantity != 2.",
])
def test_numeric_negation_preserved_after_ranking_changes(loader, query):
    _, execution = interpret_and_execute(loader, query)
    expected = loader.sales.loc[loader.sales.quantity != 2, "revenue"].sum()
    assert execution["result"] == [{"revenue": pytest.approx(expected)}]


@pytest.mark.parametrize("query", [
    "Show total revenue before 2024.",
    "Calculate revenue prior to 2024.",
])
def test_date_boundary_preserved_after_ranking_changes(loader, query):
    plan, execution = interpret_and_execute(loader, query)
    assert plan.time_filter.column == "ordered_at"
    expected = loader.sales.loc[loader.sales.ordered_at < pd.Timestamp("2024-01-01"), "revenue"].sum()
    assert execution["result"] == [{"revenue": pytest.approx(expected)}]


@pytest.mark.parametrize("query,payload", [
    ("What is the average revenue?", {"operation": "aggregation", "metric": "avg_order_value", "aggregation": "mean"}),
    ("Show mean quantity.", {"operation": "aggregation", "metric": "quantity", "aggregation": "sum"}),
    ("Rank product categories by revenue from highest to lowest.", {
        "operation": "ranking", "metric": "revenue", "group_by": ["product_category"],
        "rank_dimension": "product_category", "sort": "desc", "limit": 1,
    }),
    ("Show product categories ordered by revenue descending.", {
        "operation": "ranking", "metric": "revenue", "group_by": ["product_category"],
        "rank_dimension": "product_category", "sort": "desc", "limit": 3,
    }),
])
def test_provider_cannot_invent_a_metric_aggregation_or_sort_limit(monkeypatch, loader, query, payload):
    payload = {**payload, "interpretation_confidence": 0.99}
    QueryPlan.model_validate(payload)  # Structural correctness alone is insufficient.
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: payload)

    def no_execution(self, plan):
        raise AssertionError("Semantically incorrect provider plan reached Pandas")

    monkeypatch.setattr(QueryExecutor, "execute", no_execution)
    with TestClient(main_module.app) as client:
        response = client.post("/query", json={"query": query})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_supported"] is False
    assert body["confidence_score"] == 0
    assert body["query_plan"]["interpretation_confidence"] == 0
    assert body["result"] == [] and body["generated_logic"] is None
    assert "No calculations were performed" in body["explanation"]
    assert body["interpreter_source"] == "gemini"
    assert body["fallback_used"] is False


def test_valid_provider_explicit_limit_is_not_changed_to_one(monkeypatch, loader):
    payload = {
        "operation": "ranking", "metric": "revenue", "aggregation": "sum",
        "group_by": ["product_name"], "rank_dimension": "product_name",
        "sort": "desc", "limit": 3, "interpretation_confidence": 0.9,
    }
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: payload)
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret("Three highest revenue products")
    assert plan.is_supported and plan.limit == 3
    assert interpreter.interpreter_source == "gemini"
    assert len(QueryExecutor(loader).execute(plan)["result"]) == 3


def test_provider_failure_preserves_generic_mean_fallback(monkeypatch, loader):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())

    def unavailable(self, context):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(QueryInterpreter, "_call_llm", unavailable)
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret("Show mean revenue.")
    assert plan.is_supported and plan.aggregation == "mean" and plan.metric == "revenue"
    assert interpreter.interpreter_source == "deterministic_fallback"
    execution = QueryExecutor(loader).execute(plan)
    assert execution["result"] == [{"revenue": pytest.approx(loader.sales.revenue.mean())}]


def test_declared_derived_metric_does_not_replace_requested_base_metric(loader):
    loader.dictionary["metrics"]["avg_order_value"] = "revenue / orders"
    loader.dictionary["synonyms"] = {"aov": "avg_order_value"}
    plan, execution = interpret_and_execute(loader, "Average revenue for all orders")
    assert plan.metric == "revenue" and plan.aggregation == "mean"
    assert execution["result"] == [{"revenue": pytest.approx(loader.sales.revenue.mean())}]


@pytest.mark.parametrize("query,payload", [
    ("What is the average revenue for all orders?", {
        "operation": "aggregation", "metric": "revenue", "aggregation": "mean",
    }),
    ("Rank product categories by revenue from highest to lowest.", {
        "operation": "ranking", "metric": "revenue", "group_by": ["product_category"],
        "rank_dimension": "product_category", "sort": "desc", "limit": None,
    }),
    ("Compare actual revenue against target for each region and show the achievement percentage.", {
        "operation": "comparison", "metric": "revenue", "group_by": ["region"],
        "requires_targets": True,
        "comparison": {"kind": "actual_vs_target", "metric": "revenue", "target_metric": "target_revenue", "target_group": "region"},
    }),
])
def test_valid_provider_plans_pass_semantic_validation_and_retain_provenance(monkeypatch, loader, query, payload):
    loader.targets = pd.DataFrame({
        "region": ["EMEA", "APAC"], "target_revenue": [1000.0, 1500.0],
    })
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: payload)
    with TestClient(main_module.app) as client:
        response = client.post("/query", json={"query": query})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_supported"] and body["result"]
    assert body["interpreter_source"] == "gemini" and body["fallback_used"] is False
    assert not body["query_plan"]["semantic_errors"]
    assert body["generated_logic"]

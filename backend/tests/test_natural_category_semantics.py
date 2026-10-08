"""Conversational grammar must preserve real analytical requirements."""

import importlib
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data_loader import DatasetLoader
from app.models import QueryPlan
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter
from app.query_validator import QueryValidator


main = importlib.import_module("app.main")
interpreter_module = importlib.import_module("app.query_interpreter")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    for variable in ("OPENROUTER_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(interpreter_module.settings, "openrouter_api_key", None)
    monkeypatch.setattr(interpreter_module.settings, "xai_api_key", None)


@pytest.fixture
def loader():
    value = object.__new__(DatasetLoader)
    value.sales = pd.DataFrame({
        "product_category": ["Technology", "Furniture", "Office Supplies", "Technology", "Furniture", "Office Supplies"],
        "product_name": ["Router", "Desk", "Pen", "Switch", "Chair", "Paper"],
        "region": ["EMEA", "APAC", "EMEA", "APAC", "EMEA", "APAC"],
        "quantity": [1, 2, 4, 1, 3, 2],
        "revenue": [17.0, 41.0, 73.0, 97.0, 19.0, 23.0],
    })
    value.targets = pd.DataFrame()
    # A declaration of an absent profit column does not make it calculable.
    value.dictionary = {
        "metrics": {"revenue": "revenue", "profit": "profit"},
        "synonyms": {"sales": "revenue", "income": "revenue", "earnings": "profit"},
        "dimensions": ["product_category", "product_name", "region"],
    }
    value.queries = []
    return value


def execute(loader, query):
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret(query)
    assert plan.is_supported, plan.model_dump()
    assert not plan.unresolved_terms and not plan.semantic_errors
    QueryPlan.model_validate(plan.model_dump())
    QueryValidator(interpreter.schema).validate(plan)
    return plan, QueryExecutor(loader).execute(plan)["result"]


@pytest.mark.parametrize("query", [
    "Show total revenue by category.",
    "Please show me overall sales by the category.",
    "Tell me the revenue generated per category.",
])
def test_category_grouping_preserves_sum(loader, query):
    plan, result = execute(loader, query)
    assert plan.operation == "aggregation"
    assert plan.metric == "revenue" and plan.aggregation == "sum"
    assert plan.group_by == ["product_category"] and plan.limit is None
    expected = loader.sales.groupby("product_category").revenue.sum().to_dict()
    assert {row["product_category"]: row["revenue"] for row in result} == expected


@pytest.mark.parametrize("query,limit,ascending", [
    ("Which category has the highest total revenue?", 1, False),
    ("Show the top 2 categories by revenue.", 2, False),
    ("Show the 2 categories with the highest total revenue.", 2, False),
    ("Give me the top 2 categories based on their total sales revenue.", 2, False),
    ("Show the bottom 2 categories by revenue.", 2, True),
    ("Please tell me the two categories with the largest overall revenue.", 2, False),
    ("Give me two categories with the lowest total sales.", 2, True),
    ("List the three products with the smallest total revenue.", 3, True),
    ("Which product makes the most revenue?", 1, False),
    ("Which category generates the least revenue?", 1, True),
    ("Which category has the best total revenue?", 1, False),
    ("Which category has the worst total revenue?", 1, True),
    ("Show the two categories based on their revenue, ranked highest first.", 2, False),
    ("Please rank those categories by their revenue from highest to lowest.", None, False),
    ("Rank these categories based on their overall sales.", None, False),
    ("Give me the top two categories based on its revenue.", 2, False),
    ("Show the top two categories based on these total sales.", 2, False),
    ("Rank the categories based on them generating revenue, highest first.", None, False),
])
def test_ranking_grammar_preserves_metric_grain_direction_and_limit(loader, query, limit, ascending):
    plan, result = execute(loader, query)
    dimension = "product_name" if "product" in query else "product_category"
    assert plan.operation == "ranking" and plan.metric == "revenue"
    assert plan.aggregation == "sum" and plan.group_by == [dimension]
    assert plan.rank_dimension == dimension and plan.partition_by == []
    assert plan.limit_scope == "global" and plan.limit == limit
    assert plan.sort == ("asc" if ascending else "desc")
    assert not plan.filters and plan.filter_expression is None
    expected = loader.sales.groupby(dimension).revenue.sum().sort_values(ascending=ascending)
    if limit is not None:
        expected = expected.head(limit)
    assert [row[dimension] for row in result] == list(expected.index)
    assert [row["revenue"] for row in result] == pytest.approx(list(expected))


@pytest.mark.parametrize("query,category", [
    ("Show total revenue for the Technology category.", "Technology"),
    ("Show total revenue where product_category equals Technology.", "Technology"),
    ("How much revenue did Technology generate?", "Technology"),
    ("Please tell me total sales for Technology products.", "Technology"),
    ("Show overall revenue for the Office Supplies category.", "Office Supplies"),
    ("Give me the revenue from those Furniture products.", "Furniture"),
    ("Show total revenue in the EMEA region.", "EMEA"),
    ("Show revenue for the APAC region.", "APAC"),
])
def test_natural_value_and_dimension_descriptions_are_filters(loader, query, category):
    plan, result = execute(loader, query)
    column = "region" if category in {"EMEA", "APAC"} else "product_category"
    assert plan.operation == "aggregation" and plan.aggregation == "sum"
    assert plan.group_by == []
    assert [(item.column, item.operator, item.value) for item in plan.filters] == [(column, "=", category)]
    expected = loader.sales.loc[loader.sales[column] == category, "revenue"].sum()
    assert result == [{"revenue": expected}]


@pytest.mark.parametrize("word", ["money", "earnings", "sales money"])
def test_conversational_monetary_terms_use_only_available_metric(loader, word):
    plan, result = execute(loader, f"Which category makes us the most {word}?")
    assert plan.metric == "revenue" and plan.aggregation == "sum" and plan.limit == 1
    expected = loader.sales.groupby("product_category").revenue.sum().idxmax()
    assert result[0]["product_category"] == expected


def test_money_with_competing_available_metrics_requires_clarification(loader, monkeypatch):
    loader.sales["profit"] = [20.0, 40.0, 60.0, 0.0, 90.0, 10.0]
    monkeypatch.setattr(main, "get_loader", lambda: loader)
    monkeypatch.setattr(main.FeedbackStore, "get_relevant", lambda self, query: [])
    execution = Mock(side_effect=AssertionError("Ambiguous money must not execute"))
    monkeypatch.setattr(QueryExecutor, "execute", execution)
    with TestClient(main.app) as client:
        body = client.post("/query", json={"query": "Which category makes us the most money?"}).json()
    assert body["is_supported"] is False and body["confidence_score"] == 0
    assert "clarif" in body["explanation"].lower()
    assert "revenue" in body["explanation"] and "profit" in body["explanation"]
    execution.assert_not_called()


def test_dictionary_monetary_alias_disambiguates_available_profit(loader):
    loader.sales["profit"] = [20.0, 40.0, 60.0, 0.0, 90.0, 10.0]
    loader.dictionary["synonyms"]["money"] = "profit"
    plan, result = execute(loader, "Which category makes the most money?")
    assert plan.metric == "profit" and plan.limit == 1
    assert result[0]["product_category"] == loader.sales.groupby("product_category").profit.sum().idxmax()


def test_custom_schema_aliases_do_not_require_fixed_metric_or_dimension_names(loader):
    loader.sales = loader.sales.rename(columns={"product_category": "department_class", "revenue": "receipts_total"})
    loader.dictionary = {
        "metrics": {"receipts_total": "receipts_total"},
        "synonyms": {"sales": "receipts_total", "income": "receipts_total", "money": "receipts_total"},
        "terms": {"category": {"column": "department_class"}},
        "dimensions": ["department_class", "product_name", "region"],
    }
    plan, result = execute(loader, "Give me the two categories with the highest total sales.")
    assert plan.metric == "receipts_total" and plan.group_by == ["department_class"] and plan.limit == 2
    expected = loader.sales.groupby("department_class").receipts_total.sum().nlargest(2)
    assert [row["department_class"] for row in result] == list(expected.index)
    filtered, _ = execute(loader, "Show total sales for the Technology category.")
    assert filtered.filters[0].column == "department_class"


def test_natural_category_filters_preserve_boolean_or_and_numeric_predicates(loader):
    query = "Show total revenue for the Technology category or the Furniture category where quantity >= 2."
    plan, result = execute(loader, query)
    assert plan.filter_expression is not None
    expected = loader.sales.loc[
        loader.sales.product_category.isin(["Technology", "Furniture"]) & (loader.sales.quantity >= 2), "revenue",
    ].sum()
    assert result == [{"revenue": expected}]


def test_ranking_qualifier_does_not_remove_real_filter(loader):
    plan, result = execute(loader, "Show the 2 categories with the highest total revenue where quantity is at least 2.")
    assert plan.limit == 2 and plan.sort == "desc" and plan.aggregation == "sum"
    assert [(item.column, item.operator, item.value) for item in plan.filters] == [("quantity", ">=", 2)]
    expected = loader.sales.loc[loader.sales.quantity >= 2].groupby("product_category").revenue.sum().nlargest(2)
    assert [row["product_category"] for row in result] == list(expected.index)


@pytest.mark.parametrize("query", [
    "Show total revenue for the Technology region.",
    "Show total revenue for the Technology nonexistent_field.",
    "Show total revenue for the Unknown category.",
    "Show the 2 categories with the highest total revenue and employee_count.",
    "Show total revenue with quantity between 1 and 3.",
    "Show the top two categories based on quantity and total revenue.",
    "Rank categories based on their revenue and employee_count.",
])
def test_grammatical_tolerance_does_not_drop_unknown_requirements(loader, query):
    plan = QueryInterpreter(loader).interpret(query)
    assert not plan.is_supported and plan.interpretation_confidence == 0
    assert plan.unresolved_terms or plan.semantic_errors


@pytest.mark.parametrize("aggregation,supported", [("sum", True), ("count", False)])
def test_provider_plan_total_semantics_are_still_mandatory(loader, monkeypatch, aggregation, supported):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: {
        "operation": "ranking", "metric": "revenue", "aggregation": aggregation,
        "group_by": ["product_category"], "rank_dimension": "product_category", "limit": 2,
        "interpretation_confidence": 0.97,
    })
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret("Show the 2 categories with the highest total revenue.")
    assert interpreter.interpreter_source == "gemini"
    assert plan.is_supported is supported
    if not supported:
        assert plan.interpretation_confidence == 0 and plan.semantic_errors


def test_provider_cannot_drop_natural_category_filter(loader, monkeypatch):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: {
        "operation": "aggregation", "metric": "revenue", "group_by": [],
        "interpretation_confidence": 0.97,
    })
    plan = QueryInterpreter(loader).interpret("Show total revenue for the Technology category.")
    assert not plan.is_supported and plan.interpretation_confidence == 0
    assert any("filter" in reason.lower() for reason in plan.semantic_errors)


@pytest.mark.parametrize("query,value,ascending", [
    ("Show the top 2 products by total revenue in the Technology category.", "Technology", False),
    ("Rank the top two products by total revenue for the Office Supplies category.", "Office Supplies", False),
    ("Show bottom 2 products by revenue from Furniture products.", "Furniture", True),
    ("List 2 products with the highest total sales in Technology category.", "Technology", False),
    ("Show top 2 products by revenue within the Technology category.", "Technology", False),
    ("In the Technology category, show top two products by total revenue.", "Technology", False),
    ("Show top 2 products by revenue in product_category equals Technology.", "Technology", False),
])
def test_ranked_entity_is_separate_from_categorical_filter(loader, query, value, ascending):
    plan, result = execute(loader, query)
    assert plan.group_by == ["product_name"] and plan.rank_dimension == "product_name"
    assert plan.partition_by == [] and plan.limit_scope == "global" and plan.limit == 2
    assert [(item.column, item.operator, item.value) for item in plan.filters] == [("product_category", "=", value)]
    expected = (loader.sales.loc[loader.sales.product_category == value]
                .groupby("product_name").revenue.sum().sort_values(ascending=ascending).head(2))
    assert [row["product_name"] for row in result] == list(expected.index)
    assert [row["revenue"] for row in result] == pytest.approx(list(expected))


@pytest.mark.parametrize("partition", ["per region", "in each region"])
def test_ranked_entity_filter_and_partition_are_independent(loader, partition):
    plan, result = execute(loader, f"Show top 2 products by revenue in the Technology category {partition}.")
    assert plan.rank_dimension == "product_name" and set(plan.group_by) == {"product_name", "region"}
    assert plan.partition_by == ["region"] and plan.limit_scope == "per_group" and plan.limit == 2
    assert [(item.column, item.operator, item.value) for item in plan.filters] == [("product_category", "=", "Technology")]
    assert all(row["product_name"] in {"Router", "Switch"} for row in result)


@pytest.mark.parametrize("dimension", ["product_name", "product_category"])
def test_gemini_ranking_plan_validation_distinguishes_filter_and_grain(loader, monkeypatch, dimension):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: {
        "operation": "ranking", "metric": "revenue", "aggregation": "sum",
        "group_by": [dimension], "rank_dimension": dimension, "limit": 2,
        "filters": [{"column": "product_category", "operator": "=", "value": "Technology"}],
        "interpretation_confidence": 0.98,
    })
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret("Show the top 2 products by total revenue in the Technology category.")
    assert interpreter.interpreter_source == "gemini" and not interpreter.fallback_used
    assert plan.is_supported is (dimension == "product_name")
    if dimension == "product_category":
        assert plan.interpretation_confidence == 0 and plan.semantic_errors


def test_filter_qualifier_does_not_override_other_ranked_entities(loader):
    loader.sales["customer_id"] = ["C1", "C2", "C3", "C4", "C5", "C6"]
    plan, _ = execute(loader, "Show top two customers by revenue in the Technology category.")
    assert plan.rank_dimension == "customer_id" and plan.group_by == ["customer_id"]
    assert plan.filters[0].column == "product_category"


def test_unknown_filter_qualifier_still_rejects_ranked_query(loader):
    plan = QueryInterpreter(loader).interpret("Show top 2 products by revenue in the Technology nonexistent_field.")
    assert not plan.is_supported and plan.interpretation_confidence == 0
    assert plan.semantic_errors

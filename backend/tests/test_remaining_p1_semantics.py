"""Independent semantic completeness regressions, without live provider calls."""

import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data_loader import DatasetLoader
from app.models import QueryPlan
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter


main_module = importlib.import_module("app.main")


def make_loader():
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "sale_id": list(range(1, 10)),
        "booked_at": pd.to_datetime([
            "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01",
            "2024-03-01", "2024-08-01", "2024-12-31", "2025-01-01", "2025-04-01",
        ]),
        "region": ["EMEA", "APAC", "EMEA", "EMEA", "APAC", "AMER", "AMER", "APAC", "EMEA"],
        "customer_id": ["C1", "C2", "C3", "C1", "C4", "C5", "C6", "C7", "C8"],
        "customer_segment": ["Business", "Consumer", "Business", "Business", "Consumer", "Business", "Consumer", "Consumer", "Business"],
        "quantity": [1, 2, 2, 3, 4, 1, 5, 2, 6],
        "revenue": [13.0, 31.0, 47.0, 19.0, 59.0, 23.0, 41.0, 17.0, 71.0],
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {
        "metrics": {"revenue": "revenue"},
        "dimensions": ["region", "customer_segment", "booked_at"],
        "default_time_field": "booked_at",
    }
    return loader


@pytest.fixture
def loader():
    return make_loader()


@pytest.fixture
def offline_client(monkeypatch, loader):
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    with TestClient(main_module.app) as client:
        yield client


def submit(client, query):
    response = client.post("/query", json={"query": query})
    assert response.status_code == 200, response.text
    return response.json()


def assert_rejected(body):
    assert body["is_supported"] is False
    assert body["confidence_score"] == 0.0
    assert body["query_plan"]["interpretation_confidence"] == 0.0
    assert body["result"] == []
    assert body["generated_logic"] is None
    assert body["unresolved_terms"] or body["query_plan"]["semantic_errors"]
    assert "No calculations were performed" in body["explanation"]


@pytest.mark.parametrize("query,year", [
    ("Show revenue for 2023", 2023),
    ("Show total revenue during 2024", 2024),
    ("Calculate revenue in 2025", 2025),
])
def test_calendar_year_constraints_execute_on_detected_date_field(offline_client, loader, query, year):
    body = submit(offline_client, query)
    assert body["is_supported"] is True, body
    expected = loader.sales.loc[loader.sales.booked_at.dt.year == year, "revenue"].sum()
    assert body["result"] == [{"revenue": pytest.approx(expected)}]
    assert body["query_plan"]["time_filter"]["column"] == "booked_at"
    assert str(year) in body["generated_logic"]
    assert str(year) in body["explanation"]


@pytest.mark.parametrize("query,year", [
    ("Show revenue before 2024", 2024),
    ("Calculate total revenue before 2023", 2023),
    ("Revenue prior to 2025", 2025),
])
def test_calendar_comparison_never_loses_boundary(offline_client, loader, query, year):
    body = submit(offline_client, query)
    assert body["is_supported"] is True, body
    expected = loader.sales.loc[loader.sales.booked_at < pd.Timestamp(year=year, month=1, day=1), "revenue"].sum()
    assert body["result"] == [{"revenue": pytest.approx(expected)}]
    assert body["query_plan"]["time_filter"]["column"] == "booked_at"
    assert str(year) in body["generated_logic"] or str(year - 1) in body["generated_logic"]


@pytest.mark.parametrize("query", [
    "Show revenue where quantity is not 2",
    "Calculate total revenue where quantity != 2",
    "Sum revenue where quantity is not equal to 2",
])
def test_numeric_negation_is_executed_as_negation(offline_client, loader, query):
    body = submit(offline_client, query)
    assert body["is_supported"] is True, body
    expected = loader.sales.loc[loader.sales.quantity != 2, "revenue"].sum()
    assert body["result"] == [{"revenue": pytest.approx(expected)}]
    assert "quantity != 2" in body["generated_logic"]
    assert "quantity != 2" in body["explanation"]


@pytest.mark.parametrize("query", [
    "Show revenue in EMEA or APAC",
    "Total revenue for EMEA or APAC",
    "Show revenue where region is EMEA or region is APAC",
])
def test_alternative_categories_preserve_or_and_explicit_boolean_plan(offline_client, loader, query):
    body = submit(offline_client, query)
    assert body["is_supported"] is True, body
    expected = loader.sales.loc[loader.sales.region.isin(["EMEA", "APAC"]), "revenue"].sum()
    assert body["result"] == [{"revenue": pytest.approx(expected)}]
    expression = body["query_plan"]["filter_expression"]
    assert expression is not None
    assert expression["op"] == "or"
    assert "OR" in body["generated_logic"].upper()
    assert "OR" in body["explanation"].upper()


@pytest.mark.parametrize("query,operator", [
    ("Show records where region is EMEA and quantity >= 2", "and"),
    ("Show records where region is EMEA or quantity >= 4", "or"),
])
def test_boolean_composition_across_different_fields(offline_client, loader, query, operator):
    body = submit(offline_client, query)
    assert body["is_supported"] is True, body
    predicate_a = loader.sales.region == "EMEA"
    predicate_b = loader.sales.quantity >= (2 if operator == "and" else 4)
    mask = predicate_a & predicate_b if operator == "and" else predicate_a | predicate_b
    assert {row["sale_id"] for row in body["result"]} == set(loader.sales.loc[mask, "sale_id"])
    assert body["query_plan"]["operation"] == "filter"
    assert body["query_plan"]["filter_expression"]["op"] == operator


@pytest.mark.parametrize("query", [
    "Show records where (region is EMEA or region is APAC) and quantity >= 2",
    "Show records where quantity >= 2 and (region is EMEA or region is APAC)",
])
def test_parentheses_keep_boolean_grouping_or_are_explicitly_rejected(offline_client, loader, query):
    body = submit(offline_client, query)
    if not body["is_supported"]:
        assert_rejected(body)
        return
    mask = loader.sales.region.isin(["EMEA", "APAC"]) & (loader.sales.quantity >= 2)
    assert {row["sale_id"] for row in body["result"]} == set(loader.sales.loc[mask, "sale_id"])
    assert body["query_plan"]["filter_expression"] is not None


@pytest.mark.parametrize("query", [
    "Show revenue sorted by quantity",
    "Show total revenue ordered by quantity ascending",
    "Calculate revenue by region sorted by quantity descending",
])
def test_unrepresentable_sorting_rejects_instead_of_disappearing(offline_client, monkeypatch, query):
    def forbidden_execution(self, plan):
        raise AssertionError("A semantically incomplete sorting plan reached execution")

    monkeypatch.setattr(QueryExecutor, "execute", forbidden_execution)
    body = submit(offline_client, query)
    assert_rejected(body)
    assert "sort" in body["explanation"].lower() or "order" in body["explanation"].lower()


@pytest.mark.parametrize("query", [
    "What percentage of revenue does each region contribute?",
    "Show each region's share of total revenue",
    "What portion of total revenue comes from each region?",
])
def test_contribution_business_phrasings_are_not_plain_totals(offline_client, loader, query):
    body = submit(offline_client, query)
    assert body["is_supported"] is True, body
    assert body["query_plan"]["operation"] == "contribution"
    assert body["query_plan"]["contribution_percentage"] is True
    assert body["query_plan"]["group_by"] == ["region"]
    totals = loader.sales.groupby("region").revenue.sum()
    percentages = totals / totals.sum() * 100
    assert set(row["region"] for row in body["result"]) == set(totals.index)
    for row in body["result"]:
        assert row["revenue"] == pytest.approx(totals.loc[row["region"]])
        percentage_keys = [key for key in row if "contribution" in key or "percentage" in key]
        assert len(percentage_keys) == 1, row
        assert row[percentage_keys[0]] == pytest.approx(percentages.loc[row["region"]], abs=0.02)
    assert "contribution" in body["generated_logic"].lower() or "percentage" in body["generated_logic"].lower()
    assert "percentage" in body["explanation"].lower() or "contribution" in body["explanation"].lower()


@pytest.mark.parametrize("query", [
    "Show top 3 customers by revenue per region",
    "List the three highest revenue customers in each region",
])
def test_customer_ranking_uses_identity_even_when_dictionary_lists_segments(offline_client, loader, query):
    body = submit(offline_client, query)
    assert body["is_supported"] is True, body
    plan = body["query_plan"]
    assert plan["rank_dimension"] == "customer_id"
    assert "customer_segment" not in plan["group_by"]
    assert plan["partition_by"] == ["region"]
    assert plan["limit"] == 3
    totals = loader.sales.groupby(["region", "customer_id"]).revenue.sum().reset_index()
    expected = totals.sort_values("revenue", ascending=False).groupby("region", sort=False).head(3)
    assert {(row["region"], row["customer_id"]): row["revenue"] for row in body["result"]} == {
        (row.region, row.customer_id): row.revenue for row in expected.itertuples()
    }


def test_entity_granularity_uses_renamed_dictionary_identity(monkeypatch):
    loader = make_loader()
    loader.sales = loader.sales.rename(columns={
        "customer_id": "buyer_id", "customer_segment": "buyer_segment", "region": "territory",
    })
    loader.dictionary["terms"] = {"buyer": {"column": "buyer_id", "aliases": ["buyers"]}}
    loader.dictionary["dimensions"] = ["territory", "buyer_segment"]
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    plan = QueryInterpreter(loader).interpret("Show top two buyers by revenue in each territory")
    assert plan.is_supported, plan.model_dump()
    assert plan.rank_dimension == "buyer_id"
    assert plan.partition_by == ["territory"]
    assert "buyer_segment" not in plan.group_by
    totals = loader.sales.groupby(["territory", "buyer_id"]).revenue.sum().reset_index()
    expected = totals.sort_values("revenue", ascending=False).groupby("territory", sort=False).head(2)
    rows = QueryExecutor(loader).execute(plan)["result"]
    assert {(row["territory"], row["buyer_id"]): row["revenue"] for row in rows} == {
        (row.territory, row.buyer_id): row.revenue for row in expected.itertuples()
    }


@pytest.mark.parametrize("query", [
    "Revenue of top 3 customers per region",
    "Sales of the top three customers within each region",
    "Show top 3 customers by revenue per region, then sum their revenue by region",
    "Find the top two customers per region and aggregate the selected customers' revenue by region",
])
def test_unrepresentable_multistage_request_never_runs_a_simpler_plan(offline_client, monkeypatch, query):
    def forbidden_execution(self, plan):
        raise AssertionError("An unresolved multi-stage request reached execution")

    monkeypatch.setattr(QueryExecutor, "execute", forbidden_execution)
    assert_rejected(submit(offline_client, query))


@pytest.mark.parametrize("query,payload", [
    ("Show revenue for 2023", {"operation": "aggregation", "metric": "revenue", "interpretation_confidence": 0.99}),
    ("Show revenue before 2024", {"operation": "aggregation", "metric": "revenue", "interpretation_confidence": 0.99}),
    ("Show revenue where quantity is not 2", {"operation": "aggregation", "metric": "revenue", "filters": [{"column": "quantity", "operator": "=", "value": 2}], "interpretation_confidence": 0.99}),
    ("Show revenue in EMEA or APAC", {"operation": "aggregation", "metric": "revenue", "filters": [{"column": "region", "operator": "=", "value": "EMEA"}, {"column": "region", "operator": "=", "value": "APAC"}], "interpretation_confidence": 0.99}),
    ("Show revenue sorted by quantity", {"operation": "aggregation", "metric": "revenue", "interpretation_confidence": 0.99}),
    ("What percentage of revenue does each region contribute?", {"operation": "aggregation", "metric": "revenue", "group_by": ["region"], "interpretation_confidence": 0.99}),
    ("Show top 3 customers by revenue per region", {"operation": "ranking", "metric": "revenue", "group_by": ["region", "customer_segment"], "rank_dimension": "customer_segment", "partition_by": ["region"], "limit_scope": "per_group", "limit": 3, "interpretation_confidence": 0.99}),
    ("Show top 3 customers by revenue per region, then sum their revenue by region", {"operation": "ranking", "metric": "revenue", "group_by": ["region", "customer_id"], "rank_dimension": "customer_id", "partition_by": ["region"], "limit_scope": "per_group", "limit": 3, "interpretation_confidence": 0.99}),
])
def test_structurally_valid_provider_plan_cannot_erase_essential_intent(monkeypatch, loader, query, payload):
    QueryPlan.model_validate(payload)
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: payload)

    def forbidden_execution(self, plan):
        raise AssertionError("An incomplete provider interpretation reached execution")

    monkeypatch.setattr(QueryExecutor, "execute", forbidden_execution)
    with TestClient(main_module.app) as client:
        body = submit(client, query)
    assert_rejected(body)
    assert body["interpreter_source"] == "gemini"
    assert body["fallback_used"] is False


def test_provider_cannot_change_explicit_or_expression_to_and(monkeypatch, loader):
    payload = {
        "operation": "aggregation", "metric": "revenue", "interpretation_confidence": 0.99,
        "filter_expression": {"op": "and", "predicates": [
            {"column": "region", "operator": "=", "value": "EMEA"},
            {"column": "region", "operator": "=", "value": "APAC"},
        ]},
    }
    QueryPlan.model_validate(payload)
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: payload)
    with TestClient(main_module.app) as client:
        assert_rejected(submit(client, "Show revenue where region is EMEA or region is APAC"))

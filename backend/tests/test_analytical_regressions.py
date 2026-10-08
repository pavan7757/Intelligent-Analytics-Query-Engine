import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data_loader import DatasetLoader
from app.models import QueryPlan
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter


main_module = importlib.import_module("app.main")


def make_loader(time_field="purchased_at", dictionary_time_field=False):
    loader = object.__new__(DatasetLoader)
    dates = [
        "2024-01-03", "2024-01-10", "2024-02-01", "2024-02-05",
        "2024-02-09", "2024-03-05", "2024-01-04", "2024-02-02",
        "2024-02-06", "2024-03-07", "2025-01-05", "2025-01-11",
    ]
    loader.sales = pd.DataFrame({
        time_field: dates if dictionary_time_field else pd.to_datetime(dates),
        "region": ["EMEA"] * 6 + ["APAC"] * 6,
        "product_name": ["A", "B", "C", "D", "A", "B", "A", "B", "C", "D", "A", "C"],
        "product_category": [
            "Tools", "Furniture", "Tools", "Furniture", "Tools", "Furniture",
            "Tools", "Furniture", "Tools", "Furniture", "Tools", "Tools",
        ],
        "quantity": [1] * 12,
        "unit_price": [100., 90., 80., 70., 60., 40., 20., 120., 110., 100., 30., 40.],
        "discount": [0.] * 12,
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {
        "metrics": {"revenue": "quantity * unit_price * (1 - discount)"},
        "dimensions": ["region", "product_name", "product_category"],
    }
    if dictionary_time_field:
        loader.dictionary["columns"] = {time_field: {"type": "datetime"}}
    loader.queries = []
    loader.feedback_path = None
    return loader


@pytest.fixture
def api(monkeypatch):
    loader = make_loader()
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    with TestClient(main_module.app) as client:
        yield client, loader


def query_fallback(client, query):
    response = client.post("/query", json={"query": query})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["is_supported"] is True, payload
    assert payload["interpreter_source"] == "deterministic_fallback"
    assert payload["fallback_used"] is True
    return payload


@pytest.mark.parametrize("time_field,dictionary_time_field", [
    ("purchased_at", False),
    ("booked_at", True),
])
def test_revenue_by_month_resolves_generic_datetime_field(api, time_field, dictionary_time_field):
    client, loader = api
    replacement = make_loader(time_field, dictionary_time_field)
    loader.sales = replacement.sales
    loader.dictionary = replacement.dictionary

    payload = query_fallback(client, "Show revenue by month")

    assert payload["query_plan"]["group_by"] == ["month"]
    assert payload["result"] == [
        {"month": "2024-01", "revenue": 210.0},
        {"month": "2024-02", "revenue": 440.0},
        {"month": "2024-03", "revenue": 140.0},
        {"month": "2025-01", "revenue": 70.0},
    ]


def test_highest_revenue_month_ranks_monthly_totals(api):
    client, _ = api

    payload = query_fallback(client, "Show highest revenue month")

    assert payload["query_plan"]["group_by"] == ["month"]
    assert payload["query_plan"]["limit"] == 1
    assert payload["query_plan"]["sort"] == "desc"
    assert payload["result"] == [{"month": "2024-02", "revenue": 440.0}]


def test_top_two_products_in_each_region_defaults_metric_and_limits_each_partition(api):
    client, _ = api

    payload = query_fallback(client, "Show top 2 products in each region")

    plan = payload["query_plan"]
    assert plan["metric"] == "revenue"
    assert set(plan["group_by"]) == {"region", "product_name"}
    assert plan["rank_dimension"] == "product_name"
    assert plan["partition_by"] == ["region"]
    assert plan["limit"] == 2
    assert plan["limit_scope"] == "per_group"
    assert plan["sort"] == "desc"
    assert payload["result"] == [
        {"region": "APAC", "product_name": "C", "revenue": 150.0},
        {"region": "APAC", "product_name": "B", "revenue": 120.0},
        {"region": "EMEA", "product_name": "A", "revenue": 160.0},
        {"region": "EMEA", "product_name": "B", "revenue": 130.0},
    ]


def test_top_three_products_in_emea_filters_before_global_ranking(api):
    client, _ = api

    payload = query_fallback(client, "Show top 3 products in EMEA")

    plan = payload["query_plan"]
    assert plan["metric"] == "revenue"
    assert plan["group_by"] == ["product_name"]
    assert plan["partition_by"] == []
    assert plan["limit_scope"] == "global"
    assert plan["limit"] == 3
    assert plan["filters"] == [{"column": "region", "operator": "=", "value": "EMEA"}]
    assert payload["result"] == [
        {"product_name": "A", "revenue": 160.0},
        {"product_name": "B", "revenue": 130.0},
        {"product_name": "C", "revenue": 80.0},
    ]


def test_revenue_by_region_and_product_category_aggregates_both_dimensions(api):
    client, _ = api

    payload = query_fallback(client, "Show revenue by region and product category")

    assert set(payload["query_plan"]["group_by"]) == {"region", "product_category"}
    assert payload["query_plan"]["filters"] == []
    assert payload["result"] == [
        {"region": "APAC", "product_category": "Furniture", "revenue": 220.0},
        {"region": "APAC", "product_category": "Tools", "revenue": 200.0},
        {"region": "EMEA", "product_category": "Furniture", "revenue": 200.0},
        {"region": "EMEA", "product_category": "Tools", "revenue": 240.0},
    ]


def use_validated_provider_plan(monkeypatch, fields):
    plan = QueryPlan.model_validate(fields)

    def interpret(self, query, feedback=None):
        self.interpreter_source = "genai"
        return plan.model_copy(deep=True)

    monkeypatch.setattr(QueryInterpreter, "interpret", interpret)


def test_provider_shaped_time_series_plan_resolves_month_without_date_column(api, monkeypatch):
    client, _ = api
    use_validated_provider_plan(monkeypatch, {
        "operation": "time_series", "metric": "revenue",
        "aggregation": "sum", "group_by": ["month"],
    })

    response = client.post("/query", json={"query": "Show revenue by month"})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["interpreter_source"] == "genai"
    assert payload["fallback_used"] is False
    assert payload["result"] == [
        {"month": "2024-01", "revenue": 210.0},
        {"month": "2024-02", "revenue": 440.0},
        {"month": "2024-03", "revenue": 140.0},
        {"month": "2025-01", "revenue": 70.0},
    ]


def test_provider_explicit_time_field_resolves_ambiguous_month_grouping(api, monkeypatch):
    client, loader = api
    loader.sales["settled_at"] = pd.to_datetime(["2026-06-01"] * len(loader.sales))
    use_validated_provider_plan(monkeypatch, {
        "operation": "time_series", "metric": "revenue",
        "aggregation": "sum", "group_by": ["month"],
        "time_filter": {"column": "purchased_at"},
    })

    response = client.post("/query", json={"query": "Show revenue by purchase month"})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["interpreter_source"] == "genai"
    assert payload["fallback_used"] is False
    assert payload["query_plan"]["time_filter"]["column"] == "purchased_at"
    assert payload["result"] == [
        {"month": "2024-01", "revenue": 210.0},
        {"month": "2024-02", "revenue": 440.0},
        {"month": "2024-03", "revenue": 140.0},
        {"month": "2025-01", "revenue": 70.0},
    ]


@pytest.mark.parametrize("date_failure", ["missing", "ambiguous"])
def test_unresolvable_time_series_plan_returns_controlled_analytical_error(api, monkeypatch, date_failure):
    client, loader = api
    if date_failure == "missing":
        loader.sales = loader.sales.drop(columns=["purchased_at"])
    else:
        loader.sales["settled_at"] = pd.to_datetime([
            "2024-01-10", "2024-01-17", "2024-02-08", "2024-02-12",
            "2024-02-16", "2024-03-12", "2024-01-11", "2024-02-09",
            "2024-02-13", "2024-03-14", "2025-01-12", "2025-01-18",
        ])
    use_validated_provider_plan(monkeypatch, {
        "operation": "time_series", "metric": "revenue", "aggregation": "sum",
    })

    response = client.post("/query", json={"query": "Show revenue over time"})

    assert response.status_code == 422, response.text
    detail = response.json()["detail"].lower()
    assert "time" in detail or "date" in detail
    if date_failure == "ambiguous":
        assert "ambiguous" in detail


def test_valid_plan_execution_exception_returns_sanitized_analytical_error(api, monkeypatch):
    client, _ = api
    use_validated_provider_plan(monkeypatch, {
        "operation": "aggregation", "metric": "revenue", "aggregation": "sum",
    })

    def fail_execution(self, plan):
        raise RuntimeError("Private execution diagnostic must not appear in the HTTP response")

    monkeypatch.setattr(QueryExecutor, "execute", fail_execution)

    response = client.post("/query", json={"query": "Show total revenue"})

    assert response.status_code == 422, response.text
    assert isinstance(response.json()["detail"], str)
    assert "Private execution diagnostic" not in response.text


def test_nonfinite_execution_result_returns_controlled_analytical_error(api, monkeypatch):
    client, _ = api
    use_validated_provider_plan(monkeypatch, {
        "operation": "aggregation", "metric": "revenue", "aggregation": "sum",
    })
    monkeypatch.setattr(QueryExecutor, "execute", lambda self, plan: {
        "result": [{"revenue": float("nan")}],
        "generated_logic": "aggregate synthetic revenue",
    })

    response = client.post("/query", json={"query": "Show total revenue"})

    assert response.status_code == 422, response.text
    assert isinstance(response.json()["detail"], str)
    assert "NaN" not in response.text

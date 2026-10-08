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
        "region": ["EMEA", "EMEA", "APAC"],
        "product_name": ["A", "B", "A"],
        "product_category": ["Tools", "Furniture", "Tools"],
        "quantity": [2, 1, 3],
        "revenue": [100.0, 200.0, 300.0],
    })
    loader.targets = pd.DataFrame({"region": ["EMEA", "APAC"], "target_revenue": [400., 500.]})
    loader.dictionary = {"metrics": {"revenue": "revenue"}}
    return loader


@pytest.mark.parametrize("query,payload", [
    ("Show median revenue", {"operation": "aggregation", "metric": "revenue"}),
    ("Show revenue and employee_count", {"operation": "aggregation", "metric": "revenue"}),
    ("Show revenue where nonexistent>=2", {"operation": "aggregation", "metric": "revenue"}),
    ("Show records where quantity == 2", {"operation": "filter", "metric": "quantity"}),
    ("Show revenue by region", {"operation": "aggregation", "metric": "revenue"}),
    ("Show total revenue by region in EMEA", {"operation": "aggregation", "metric": "revenue", "group_by": ["region"]}),
    ("Show revenue by region, product category", {"operation": "aggregation", "metric": "revenue", "group_by": ["region"]}),
    ("Show minimum revenue", {"operation": "aggregation", "metric": "revenue", "aggregation": "sum"}),
    ("Show revenue last month", {"operation": "aggregation", "metric": "revenue"}),
    ("Show revenue from 2024-01-01 to 2024-02-01", {"operation": "aggregation", "metric": "revenue"}),
    ("Compare total revenue against target", {"operation": "aggregation", "metric": "revenue"}),
    ("Show records where quantity == 2", {"operation": "aggregation", "metric": "quantity", "filters": [{"column": "quantity", "operator": "=", "value": 2}]}),
])
def test_provider_cannot_drop_explicit_intent(monkeypatch, query, payload):
    loader = make_loader()
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: payload)
    QueryPlan.model_validate(payload)  # These fail semantics, not structural validation.
    with TestClient(main_module.app) as client:
        response = client.post("/query", json={"query": query})
    assert response.status_code == 200
    body = response.json()
    assert body["is_supported"] is False
    assert body["result"] == []
    assert body["confidence_score"] == 0.0
    assert body["generated_logic"] is None
    assert "No calculations were performed" in body["explanation"]


def test_logic_contains_executed_predicates_partition_limit_and_assumption():
    plan = QueryPlan(
        operation="ranking", metric="revenue", group_by=["product_name"],
        rank_dimension="product_name", partition_by=["region"],
        limit_scope="per_group", limit=1,
        filters=[{"column": "quantity", "operator": ">=", "value": 2}],
        ambiguity=["Using the dataset default metric 'revenue'."],
    )
    execution = QueryExecutor(make_loader()).execute(plan)
    logic = execution["generated_logic"]
    assert "quantity >= 2" in logic
    assert "group_by: region, product_name" in logic
    assert "partition_by: region" in logic
    assert "limit: 1 (per_group)" in logic
    assert "Using the dataset default metric 'revenue'." in logic
    assert execution["result"] == [
        {"region": "APAC", "product_name": "A", "revenue": 300.0},
        {"region": "EMEA", "product_name": "A", "revenue": 100.0},
    ]


def test_logic_for_records_never_claims_aggregation():
    execution = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="filter", metric="quantity",
        filters=[{"column": "quantity", "operator": "=", "value": 2}],
    ))
    assert "quantity = 2" in execution["generated_logic"]
    assert "sum(" not in execution["generated_logic"]
    assert "group_by:" not in execution["generated_logic"]
    assert [row["quantity"] for row in execution["result"]] == [2]


def test_grouped_numeric_filter_executes_aggregation_not_raw_rows(monkeypatch):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    loader = make_loader()
    plan = QueryInterpreter(loader).interpret("Show total revenue by region where quantity >= 2")
    assert plan.is_supported
    assert plan.operation == "aggregation"
    execution = QueryExecutor(loader).execute(plan)
    assert execution["result"] == [
        {"region": "APAC", "revenue": 300.0}, {"region": "EMEA", "revenue": 100.0},
    ]
    assert "quantity >= 2" in execution["generated_logic"]
    assert "sum(revenue)" in execution["generated_logic"]


@pytest.mark.parametrize("targets", [
    pd.DataFrame(),
    pd.DataFrame({"region": ["EMEA"], "period": ["2024-01"], "unrelated": [100.]}),
])
def test_fallback_target_request_without_usable_targets_returns_422(monkeypatch, targets):
    loader = make_loader()
    loader.targets = targets
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    with TestClient(main_module.app) as client:
        response = client.post("/query", json={"query": "Compare total revenue against target"})
    assert response.status_code == 422
    assert "target" in response.json()["detail"].lower()

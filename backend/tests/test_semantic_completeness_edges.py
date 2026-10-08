"""Independent paraphrase checks for intent completeness, without provider calls."""

import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data_loader import DatasetLoader
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter


main_module = importlib.import_module("app.main")


@pytest.fixture
def loader():
    instance = object.__new__(DatasetLoader)
    instance.sales = pd.DataFrame({
        "posted_at": pd.to_datetime(["2022-12-31", "2023-06-01", "2024-01-01", "2024-07-01"]),
        "region": ["EMEA", "EMEA", "APAC", "EMEA"],
        "customer_id": ["A", "B", "C", "A"],
        "customer_segment": ["Enterprise", "Consumer", "Consumer", "Enterprise"],
        "quantity": [1, 2, 3, 4],
        "revenue": [11.0, 23.0, 37.0, 17.0],
    })
    instance.targets = pd.DataFrame()
    instance.dictionary = {"metrics": {"revenue": "revenue"}, "default_time_field": "posted_at"}
    return instance


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


def forbid_execution(monkeypatch):
    def forbidden(self, plan):
        raise AssertionError("An incomplete interpretation reached the executor")
    monkeypatch.setattr(QueryExecutor, "execute", forbidden)


def assert_rejected(body):
    assert body["is_supported"] is False, body
    assert body["confidence_score"] == 0.0
    assert body["query_plan"]["interpretation_confidence"] == 0.0
    assert body["result"] == []
    assert body["generated_logic"] is None
    assert "No calculations were performed" in body["explanation"]


@pytest.mark.parametrize("query", [
    "Show revenue for 2023 and 2024",
    "Show revenue for 2023, 2024",
])
def test_ambiguous_year_lists_do_not_discard_a_year(offline_client, monkeypatch, query):
    forbid_execution(monkeypatch)
    assert_rejected(submit(offline_client, query))


@pytest.mark.parametrize("query", [
    "Show revenue as of 2023",
    "Show revenue up to 2023",
])
def test_unrecognized_temporal_boundary_is_executed_or_rejected(offline_client, loader, query):
    body = submit(offline_client, query)
    if not body["is_supported"]:
        assert_rejected(body)
        return
    expected = loader.sales.loc[loader.sales.posted_at < pd.Timestamp("2024-01-01"), "revenue"].sum()
    assert body["result"] == [{"revenue": pytest.approx(expected)}]
    assert body["query_plan"]["time_filter"] is not None


@pytest.mark.parametrize("query", [
    "Sort revenue by quantity",
    "Show revenue ordered ascending by quantity",
    "Show revenue arranged descending by quantity",
])
def test_sorting_paraphrases_never_become_grouping(offline_client, monkeypatch, query):
    forbid_execution(monkeypatch)
    assert_rejected(submit(offline_client, query))


@pytest.mark.parametrize("query", [
    "Sum the revenue of the three highest customers per region",
    "Total revenue of the two lowest customers per region",
])
def test_second_aggregation_stage_rejects_number_word_ranking(offline_client, monkeypatch, query):
    forbid_execution(monkeypatch)
    body = submit(offline_client, query)
    assert_rejected(body)
    assert "stage" in body["explanation"].lower()


@pytest.mark.parametrize("query", [
    "Show top 3 customer segments by revenue per region",
    "List the two highest revenue customer segments in each region",
])
def test_explicit_segment_ranking_does_not_change_to_identity(offline_client, loader, query):
    body = submit(offline_client, query)
    assert body["is_supported"] is True, body
    plan = body["query_plan"]
    assert plan["rank_dimension"] == "customer_segment"
    assert "customer_id" not in plan["group_by"]
    assert plan["partition_by"] == ["region"]
    totals = loader.sales.groupby(["region", "customer_segment"]).revenue.sum().reset_index()
    expected = totals.sort_values("revenue", ascending=False).groupby("region", sort=False).head(plan["limit"])
    assert {(row["region"], row["customer_segment"]): row["revenue"] for row in body["result"]} == {
        (row.region, row.customer_segment): row.revenue for row in expected.itertuples()
    }


@pytest.mark.parametrize("query", [
    "Show top two customers by revenue",
    "Show revenue by customers",
])
def test_ambiguous_entity_ids_are_not_resolved_by_fixed_alias_fallback(offline_client, loader, monkeypatch, query):
    loader.sales["customers_id"] = ["X", "Y", "Z", "W"]
    forbid_execution(monkeypatch)
    assert_rejected(submit(offline_client, query))


def test_explicit_identity_field_survives_ambiguous_entity_metadata(offline_client, loader):
    loader.sales["customers_id"] = ["X", "Y", "Z", "W"]
    body = submit(offline_client, "Show revenue by customer_id")
    assert body["is_supported"] is True, body
    totals = loader.sales.groupby("customer_id").revenue.sum()
    assert {row["customer_id"]: row["revenue"] for row in body["result"]} == totals.to_dict()

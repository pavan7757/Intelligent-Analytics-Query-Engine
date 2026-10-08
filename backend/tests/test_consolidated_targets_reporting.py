import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data_loader import DatasetLoader
from app.explanation import build_explanation
from app.models import ComparisonPlan, FilterPlan, QueryPlan
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter
from app.query_validator import QueryValidator
from app.schema_manager import SchemaManager


main_module = importlib.import_module("app.main")


def comparison_loader():
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "region": ["North", "North", "South", "Unassigned"],
        "customer_id": ["buyer-a", "buyer-b", "buyer-a", "buyer-c"],
        "customer_segment": ["retail", "retail", "retail", "wholesale"],
        "revenue": [17.0, 23.0, 8.0, 11.0],
        "quantity": [1, 3, 2, 1],
    })
    loader.targets = pd.DataFrame({
        "region": ["North", "North", "South"],
        "target_revenue": [50.0, 70.0, 25.0],
    })
    loader.dictionary = {
        "metrics": {"revenue": "revenue"},
        "dimensions": ["region", "customer_id", "customer_segment"],
    }
    loader.queries = []
    loader.feedback_path = None
    return loader


def target_plan(**updates):
    return QueryPlan(
        operation="comparison", metric="revenue", aggregation="sum",
        group_by=["region"], requires_targets=True,
        comparison=ComparisonPlan(
            kind="actual_vs_target", metric="revenue", target_metric="target_revenue",
            target_group="region",
        ),
        **updates,
    )


@pytest.mark.parametrize("query", [
    "Compare actual revenue against target for each region and show the achievement percentage.",
    "Compare actual revenue with target for each region.",
    "Show target achievement percentage by region.",
    "Compare revenue versus target by region and report percentage achievement.",
])
def test_target_comparison_paraphrases_preserve_comparison_semantics(monkeypatch, query):
    loader = DatasetLoader()
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret(query)
    assert plan.is_supported, plan.model_dump()
    assert plan.operation == "comparison"
    assert plan.group_by == ["region"]
    assert plan.metric == "revenue"
    assert plan.requires_targets
    assert not plan.contribution_percentage
    validated = QueryValidator(SchemaManager(loader)).validate(plan)["plan"]
    execution = QueryExecutor(loader).execute(validated)
    actual_frame = loader.sales.copy()
    actual_frame["revenue"] = (
        actual_frame["quantity"] * actual_frame["unit_price"] * (1 - actual_frame["discount"])
    )
    actual_frame["region"] = actual_frame["region"].fillna("Unknown")
    expected_actual = actual_frame.groupby("region")["revenue"].sum()
    expected_targets = loader.targets.groupby("region")["target_revenue"].sum()
    for row in execution["result"]:
        assert row["actual"] == pytest.approx(expected_actual.loc[row["region"]])
        if row["region"] not in expected_targets.index:
            assert row["target"] is None
            assert row["variance"] is None
            assert row["achievement_percentage"] is None
        else:
            expected_target = expected_targets.loc[row["region"]]
            assert row["target"] == pytest.approx(expected_target)
            assert row["variance"] == pytest.approx(row["actual"] - expected_target)
            assert row["achievement_percentage"] == pytest.approx(row["actual"] / expected_target * 100)
    assert "achievement_percentage" in execution["generated_logic"]


def test_target_comparison_with_custom_numeric_metric_stays_schema_driven():
    loader = comparison_loader()
    loader.sales = loader.sales.rename(columns={"revenue": "profit"})
    loader.targets = loader.targets.rename(columns={"target_revenue": "target_profit"})
    plan = QueryPlan(
        operation="comparison", metric="profit", group_by=["region"], requires_targets=True,
        comparison=ComparisonPlan(kind="achievement", metric="profit", target_metric="target_profit"),
    )
    validated = QueryValidator(SchemaManager(loader)).validate(plan)["plan"]
    result = QueryExecutor(loader).execute(validated)["result"]
    totals = loader.sales.groupby("region")["profit"].sum()
    assert {row["region"]: row["actual"] for row in result} == totals.to_dict()


def test_missing_target_match_preserves_actual_with_null_target_derived_fields():
    loader = comparison_loader()
    result = QueryExecutor(loader).execute(target_plan())["result"]
    missing = next(row for row in result if row["region"] not in set(loader.targets["region"]))
    assert missing["actual"] == loader.sales.loc[loader.sales["region"] == missing["region"], "revenue"].sum()
    assert missing["target"] is None
    assert missing["variance"] is None
    assert missing["achievement_percentage"] is None


def test_filter_with_no_compatible_target_rows_retains_actual_and_nulls():
    loader = comparison_loader()
    missing_region = next(region for region in loader.sales["region"].unique() if region not in set(loader.targets["region"]))
    plan = target_plan(filters=[FilterPlan(column="region", operator="=", value=missing_region)])
    result = QueryExecutor(loader).execute(plan)["result"]
    assert len(result) == 1
    assert result[0]["actual"] == loader.sales.loc[loader.sales["region"] == missing_region, "revenue"].sum()
    assert all(result[0][field] is None for field in ("target", "variance", "achievement_percentage"))


def test_partial_missing_target_total_is_unknown_instead_of_incomplete_sum():
    loader = comparison_loader()
    affected_region = loader.targets.loc[0, "region"]
    loader.targets.loc[0, "target_revenue"] = None
    rows = QueryExecutor(loader).execute(target_plan())["result"]
    affected = next(row for row in rows if row["region"] == affected_region)
    assert affected["actual"] == loader.sales.loc[loader.sales["region"] == affected_region, "revenue"].sum()
    assert all(affected[field] is None for field in ("target", "variance", "achievement_percentage"))
    unaffected = next(row for row in rows if row["region"] in set(loader.targets["region"]) and row["region"] != affected_region)
    assert unaffected["target"] == loader.targets.loc[loader.targets["region"] == unaffected["region"], "target_revenue"].sum()


def test_grouped_comparison_with_missing_target_values_preserves_actual_rows():
    loader = comparison_loader()
    loader.targets["target_revenue"] = None
    rows = QueryExecutor(loader).execute(target_plan())["result"]
    assert len(rows) == loader.sales["region"].nunique()
    assert all(row["actual"] is not None for row in rows)
    assert all(row[field] is None for row in rows for field in ("target", "variance", "achievement_percentage"))


def test_zero_target_has_null_achievement_but_valid_variance():
    loader = comparison_loader()
    zero_region = loader.targets.loc[0, "region"]
    loader.targets.loc[loader.targets["region"] == zero_region, "target_revenue"] = 0.0
    rows = QueryExecutor(loader).execute(target_plan())["result"]
    row = next(row for row in rows if row["region"] == zero_region)
    assert row["target"] == 0.0
    assert row["variance"] == row["actual"]
    assert row["achievement_percentage"] is None


def test_target_column_alias_is_respected_without_changing_actual_metric():
    loader = comparison_loader()
    loader.targets = loader.targets.rename(columns={"target_revenue": "budget_amount"})
    plan = QueryPlan(
        operation="comparison", metric="revenue", group_by=["region"], requires_targets=True,
        comparison=ComparisonPlan(kind="actual_vs_target", metric="revenue", target_column="budget_amount"),
    )
    rows = QueryExecutor(loader).execute(plan)["result"]
    available = next(row for row in rows if row["region"] in set(loader.targets["region"]))
    assert available["target"] == loader.targets.loc[loader.targets["region"] == available["region"], "budget_amount"].sum()


@pytest.mark.parametrize("query", [
    "Show the top 3 customers in each region and then rank the regions by their combined revenue.",
    "Select the top two customers by revenue per region, then sum their revenue by region.",
])
def test_unsupported_multistage_endpoint_never_claims_execution(monkeypatch, query):
    loader = comparison_loader()
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    monkeypatch.setattr(QueryExecutor, "execute", lambda *args: pytest.fail("Rejected query was executed"))
    with TestClient(main_module.app) as client:
        response = client.post("/query", json={"query": query})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_supported"] is False
    assert body["confidence_score"] == 0.0
    assert body["query_plan"]["interpretation_confidence"] == 0.0
    assert body["generated_logic"] is None
    assert body["result"] == []
    assert "No calculations were performed" in body["explanation"]
    assert "stage" in body["explanation"].lower()
    assert "Results were grouped" not in body["explanation"]
    assert "Pandas calculated" not in body["explanation"]
    assert "customer_segment" not in body["query_plan"]["group_by"]


@pytest.mark.parametrize("targets", [
    pd.DataFrame(),
    pd.DataFrame({"region": ["North"], "unrelated_amount": [45.0]}),
])
@pytest.mark.parametrize("query", [
    "Show the top 3 customers in each region and then compare their combined revenue against target.",
    "Select the top two customers by revenue per region, then compare their revenue with target.",
])
def test_missing_targets_do_not_mask_unsupported_multistage_intent(monkeypatch, targets, query):
    loader = comparison_loader()
    loader.targets = targets
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    monkeypatch.setattr(QueryExecutor, "execute", lambda *args: pytest.fail("Rejected query was executed"))
    with TestClient(main_module.app) as client:
        response = client.post("/query", json={"query": query})
    assert response.status_code == 200, response.text
    body = response.json()
    assert not body["is_supported"]
    assert body["confidence_score"] == 0.0
    assert body["result"] == []
    assert body["generated_logic"] is None
    assert "No calculations were performed" in body["explanation"]
    assert "stage" in body["explanation"].lower()
    assert "Pandas calculated" not in body["explanation"]
    assert "Results were grouped" not in body["explanation"]


@pytest.mark.parametrize("invalid_updates", [
    {"is_supported": False, "ambiguity": ["A second calculation stage cannot be represented."]},
    {"unresolved_terms": ["missing_dimension"]},
    {"semantic_errors": ["The required predicate was not preserved."]},
])
def test_rejected_plan_reporting_is_consistent_even_with_high_provider_confidence(monkeypatch, invalid_updates):
    loader = comparison_loader()
    loader.targets = pd.DataFrame()
    plan = QueryPlan.model_validate({
        **target_plan().model_dump(), "interpretation_confidence": 0.99, **invalid_updates,
    })
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    monkeypatch.setattr(QueryInterpreter, "interpret", lambda *args: plan)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    monkeypatch.setattr(QueryExecutor, "execute", lambda *args: pytest.fail("Rejected query was executed"))
    with TestClient(main_module.app) as client:
        response = client.post("/query", json={"query": "Unsupported target request"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert not body["is_supported"]
    assert body["confidence_score"] == 0.0
    assert body["query_plan"]["interpretation_confidence"] == 0.0
    assert body["generated_logic"] is None
    assert body["result"] == []
    assert "No calculations were performed" in body["explanation"]
    assert "Pandas calculated" not in body["explanation"]
    assert "Results were grouped" not in body["explanation"]


def test_rejected_plan_with_no_provider_reason_has_safe_reporting():
    plan = QueryPlan(operation="nested", is_supported=False)
    explanation = build_explanation(plan.model_dump(), "unsupported request", "genai", False)
    assert "No calculations were performed" in explanation
    assert "could not be represented safely" in explanation
    assert "GenAI" in explanation
    assert "Pandas" not in explanation


def test_target_explanation_describes_actual_comparison_and_missing_match_handling():
    explanation = build_explanation(target_plan().model_dump(), "Compare actuals with targets")
    assert "variance is actual minus target" in explanation
    assert "actual divided by target times 100" in explanation
    assert "null target-derived values" in explanation

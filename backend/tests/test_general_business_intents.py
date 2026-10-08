"""Business phrasing preserves the calculation denominator and schema grain."""

import pandas as pd
import pytest

from app.data_loader import DatasetLoader
from app.models import QueryPlan
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter
from app.query_validator import QueryValidator


@pytest.fixture(autouse=True)
def offline_provider(monkeypatch):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)

    def prohibited(*args, **kwargs):
        raise AssertionError("Business-intent regressions must never contact a provider")

    monkeypatch.setattr(QueryInterpreter, "_call_llm", prohibited)


@pytest.fixture
def loader():
    value = object.__new__(DatasetLoader)
    value.sales = pd.DataFrame({
        "region": ["EMEA", "APAC", "EMEA", "APAC", "AMER"],
        "product_name": ["Desk", "Lamp", "Lamp", "Desk", "Chair"],
        "quantity": [1, 3, 2, 4, 2],
        "revenue": [19.0, 31.0, 47.0, 59.0, 83.0],
    })
    value.targets = pd.DataFrame({
        "region": ["EMEA", "APAC", "AMER"],
        "target_revenue": [101.0, 137.0, 173.0],
    })
    value.dictionary = {
        "metrics": {"revenue": "revenue"},
        "synonyms": {"sales": "revenue", "income": "revenue"},
        "dimensions": ["region", "product_name"],
    }
    return value


def execute_supported(loader, query):
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret(query)
    assert plan.is_supported, plan.model_dump()
    assert not plan.semantic_errors
    validated = QueryValidator(interpreter.schema).validate(plan)["plan"]
    return plan, QueryExecutor(loader).execute(validated)


def assert_target_ratios(loader, plan, execution, metric="revenue", dimension="region"):
    assert plan.operation == "comparison"
    assert plan.metric == metric
    assert plan.group_by == [dimension]
    assert plan.requires_targets and plan.comparison is not None
    assert not plan.contribution_percentage
    assert plan.comparison.metric == metric
    assert plan.comparison.target_metric == f"target_{metric}"
    assert plan.limit is None
    actual = loader.sales.groupby(dimension)[metric].sum()
    targets = loader.targets.groupby(dimension)[f"target_{metric}"].sum()
    assert len(execution["result"]) == len(actual)
    for row in execution["result"]:
        key = row[dimension]
        assert row["actual"] == pytest.approx(actual.loc[key])
        assert row["target"] == pytest.approx(targets.loc[key])
        assert row["achievement_percentage"] == pytest.approx(actual.loc[key] / targets.loc[key] * 100)
    assert "achievement_percentage" in execution["generated_logic"]


@pytest.mark.parametrize("query", [
    "How did each region perform relative to its assigned sales goal, expressed as a percentage of that goal?",
    "Show revenue by region relative to the sales goal, expressed as a percentage.",
    "Compare revenue with the assigned goal for each region.",
    "How much of its sales quota did each region achieve?",
    "Show sales as a percentage of quota by region.",
    "Compare revenue against the objective for each region.",
    "Show revenue by region as a percentage of its target.",
    "What percentage of the sales target did each region achieve?",
    "Show revenue relative to target by region, expressed as a percentage of that target.",
    "Show revenue against goal by region, and report the achievement percentage.",
    "For each region, show revenue versus its sales goal.",
    "Show revenue by region compared with quota.",
])
def test_goal_quota_objective_paraphrases_use_target_denominator(loader, query):
    plan, execution = execute_supported(loader, query)
    assert_target_ratios(loader, plan, execution)


@pytest.mark.parametrize("metric,dimension", [("margin", "district"), ("profit", "territory")])
@pytest.mark.parametrize("goal_word", ["goal", "quota", "objective", "target"])
def test_target_intent_is_generic_across_renamed_numeric_metrics_and_dimensions(
    loader, metric, dimension, goal_word,
):
    loader.sales = loader.sales.rename(columns={"revenue": metric, "region": dimension})
    loader.targets = loader.targets.rename(columns={"target_revenue": f"target_{metric}", "region": dimension})
    loader.dictionary = {"metrics": {metric: metric}, "dimensions": [dimension, "product_name"]}
    plan, execution = execute_supported(
        loader, f"Show {metric} by {dimension} relative to its {goal_word}, expressed as a percentage.",
    )
    assert_target_ratios(loader, plan, execution, metric, dimension)


@pytest.mark.parametrize("query", [
    "What percentage of revenue does each region contribute?",
    "Show each region's share of total revenue.",
    "Show revenue contribution percentage by region.",
    "Show revenue by region as a percentage of total revenue.",
])
def test_contribution_percentage_keeps_total_denominator_even_when_targets_exist(loader, query):
    plan, execution = execute_supported(loader, query)
    assert plan.operation == "contribution"
    assert plan.group_by == ["region"]
    assert plan.contribution_percentage
    assert not plan.requires_targets and plan.comparison is None
    expected = loader.sales.groupby("region").revenue.sum()
    for row in execution["result"]:
        assert row["revenue"] == pytest.approx(expected.loc[row["region"]])
        assert row["revenue_contribution_percentage"] == pytest.approx(expected.loc[row["region"]] / expected.sum() * 100)


@pytest.mark.parametrize("query", [
    "Show revenue and employee_count relative to goal by region.",
    "Show revenue, employee_count by region.",
    "Show revenue against quota by employee.",
    "Show revenue relative to goal by region where nonexistent >= 2.",
    "Show revenue relative to target by region where quantity ~~ 2.",
    "Show revenue relative to goal by region before next Tuesday.",
])
def test_business_vocabulary_never_hides_unknown_fields_or_constraints(loader, query):
    plan = QueryInterpreter(loader).interpret(query)
    assert not plan.is_supported, plan.model_dump()
    assert plan.interpretation_confidence == 0
    assert plan.semantic_errors or plan.unresolved_terms


@pytest.mark.parametrize("payload", [
    {
        "operation": "contribution", "metric": "revenue", "group_by": ["region"],
        "contribution_percentage": True, "interpretation_confidence": 0.98,
    },
    {
        "operation": "aggregation", "metric": "revenue", "group_by": ["region"],
        "interpretation_confidence": 0.98,
    },
    {
        "operation": "comparison", "metric": "revenue", "group_by": ["region"],
        "requires_targets": True, "comparison": None, "interpretation_confidence": 0.98,
    },
])
def test_structurally_valid_provider_plan_with_wrong_denominator_is_rejected(monkeypatch, loader, payload):
    QueryPlan.model_validate(payload)
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: payload)
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret("Show revenue by region as a percentage of its assigned sales goal.")
    assert interpreter.interpreter_source == "gemini"
    assert not plan.is_supported, plan.model_dump()
    assert plan.interpretation_confidence == 0
    assert plan.semantic_errors


def test_provider_target_comparison_remains_genai_and_executes_validated_plan(monkeypatch, loader):
    payload = {
        "operation": "comparison", "metric": "revenue", "group_by": ["region"],
        "requires_targets": True,
        "comparison": {"kind": "achievement", "metric": "revenue", "target_metric": "target_revenue", "target_group": "region"},
        "interpretation_confidence": 0.97,
    }
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: payload)
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret("How did each region perform relative to its assigned sales goal, expressed as a percentage of that goal?")
    assert interpreter.interpreter_source == "gemini"
    assert plan.is_supported, plan.model_dump()
    validated = QueryValidator(interpreter.schema).validate(plan)["plan"]
    execution = QueryExecutor(loader).execute(validated)
    assert_target_ratios(loader, plan, execution)

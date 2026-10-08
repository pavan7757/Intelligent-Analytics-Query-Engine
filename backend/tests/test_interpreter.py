import json

import pandas as pd

from app.confidence import calculate_confidence
from app.data_loader import DatasetLoader
from app.explanation import build_explanation
from app.query_interpreter import QueryInterpreter
from app.models import QueryPlan


def test_provider_prompt_contains_exact_query_plan_schema():
    interpreter = QueryInterpreter(make_loader())
    prompt = interpreter._system_prompt()
    schema = json.loads(prompt.split("The exact required QueryPlan JSON Schema follows:\n", 1)[1])
    assert schema == QueryPlan.model_json_schema()
    assert "ONLY one JSON object" in prompt
    assert "at the top level" in prompt


def test_wrapped_provider_plan_still_falls_back(monkeypatch):
    interpreter = QueryInterpreter(make_loader())
    interpreter.client = object()
    monkeypatch.setattr(interpreter, "_call_llm", lambda context: {
        "plan": [{"operation": "aggregation", "metric": "revenue"}],
    })
    plan = interpreter.interpret("Show total revenue")
    assert interpreter.interpreter_source == "deterministic_fallback"
    assert plan.metric == "revenue"


def test_provider_context_includes_dictionary_metric_definitions():
    loader = make_loader()
    context = json.loads(QueryInterpreter(loader)._build_context("Show total revenue", None))
    assert context["dictionary"] == loader.dictionary
    assert context["dictionary"]["metrics"]["revenue"] == "quantity * unit_price * (1 - discount)"


def test_matching_provider_plan_uses_genai(monkeypatch):
    interpreter = QueryInterpreter(make_loader())
    interpreter.client = object()
    monkeypatch.setattr(interpreter, "_call_llm", lambda context: {
        "operation": "aggregation", "metric": "revenue", "aggregation": "sum",
    })
    plan = interpreter.interpret("Show total revenue")
    assert interpreter.interpreter_source == "gemini"
    assert plan.metric == "revenue"


def make_loader() -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "region": ["North", "South", "North"],
        "revenue": [100.0, 200.0, 300.0],
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {"terms": {"revenue": "revenue"}, "metrics": {"revenue": "quantity * unit_price * (1 - discount)"}}
    return loader


def test_group_dimension_does_not_create_filter():
    loader = make_loader()
    interpreter = QueryInterpreter(loader)
    interpreter.client = None

    plan = interpreter.interpret("Show total revenue by region.")

    assert plan.metric == "revenue"
    assert plan.aggregation == "sum"
    assert plan.group_by == ["region"]
    assert plan.filters == []
    assert "Applied the requested filters" not in build_explanation(
        plan.model_dump(), "Show total revenue by region."
    )
    assert calculate_confidence(plan.model_dump(), {
        "metric_valid": True,
        "group_valid": True,
        "filter_valid": True,
        "time_valid": True,
        "target_available": True,
        "schema_valid": True,
    }, has_results=False) < 1.0


def test_explanation_uses_correct_articles():
    loader = make_loader()
    interpreter = QueryInterpreter(loader)
    interpreter.client = None

    aggregation = interpreter.interpret("Show total revenue.")
    ranking = interpreter.interpret("Show the top 5 products by revenue.")

    assert "as an aggregation operation" in build_explanation(
        aggregation.model_dump(), "Show total revenue."
    )
    assert "as a ranking operation" in build_explanation(
        ranking.model_dump(), "Show the top 5 products by revenue."
    )

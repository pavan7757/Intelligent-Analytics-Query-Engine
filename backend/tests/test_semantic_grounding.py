import pandas as pd
from fastapi.testclient import TestClient

from app.data_loader import DatasetLoader
from app.main import app
from app.query_executor import QueryExecutionError, QueryExecutor
from app.query_interpreter import QueryInterpreter


def make_loader() -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "city": ["Paris", "London", "Paris"],
        "region": ["EMEA", "EMEA", "EMEA"],
        "product_name": ["A", "B", "A"],
        "product_category": ["Tools", "Furniture", "Tools"],
        "quantity": [2, 1, 3],
        "unit_price": [10.0, 20.0, 10.0],
        "discount": [0.0, 0.0, 0.0],
        "revenue": [20.0, 20.0, 30.0],
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {
        "metrics": {
            "revenue": "quantity * unit_price * (1 - discount)",
            "profit": "profit",
        },
        "dimensions": ["city", "region", "product_category", "product_name"],
        "synonyms": {"sales": "revenue", "income": "revenue"},
    }
    loader.queries = []
    loader.feedback_path = None
    return loader


def interpret(query: str):
    interpreter = QueryInterpreter(make_loader())
    interpreter.client = None
    return interpreter.interpret(query)


def test_unresolved_metric_and_context_are_explicitly_rejected():
    plan = interpret("What is the average temperature on Mars by city?")

    assert plan.operation == "aggregation"
    assert plan.metric is None
    assert plan.aggregation == "mean"
    assert plan.group_by == ["city"]
    assert plan.is_supported is False
    assert set(plan.unresolved_terms) >= {"temperature", "Mars"}
    assert plan.resolved_fields == ["city"]


def test_no_arbitrary_numeric_column_fallback():
    plan = interpret("What is the average temperature on Mars by city?")

    assert plan.metric is None
    assert plan.aggregation == "mean"
    assert "quantity" not in plan.resolved_fields


def test_existing_valid_queries_remain_supported():
    valid_queries = [
        "What is the total revenue?",
        "Show total revenue by region.",
        "Show the top 5 products by revenue.",
        "Which five products generated the most revenue?",
        "Which product category generated the highest revenue?",
    ]

    for query in valid_queries:
        plan = interpret(query)
        assert plan.is_supported is True, (query, plan.model_dump())


def test_unsupported_examples_are_rejected_generically():
    for query in [
        "Show employee salary by department.",
        "Average rainfall by region.",
        "Show website visitors by browser.",
        "Calculate patient age by hospital.",
    ]:
        plan = interpret(query)
        assert plan.is_supported is False, (query, plan.model_dump())
        assert plan.unresolved_terms, (query, plan.model_dump())


def test_execution_blocks_unsupported_plans():
    plan = interpret("What is the average temperature on Mars by city?")

    try:
        QueryExecutor(make_loader()).execute(plan)
    except QueryExecutionError as exc:
        assert "not supported" in str(exc).lower()
    else:
        raise AssertionError("Unsupported plans must not execute")


def test_api_returns_safe_unsupported_response():
    response = TestClient(app).post(
        "/query",
        json={"query": "What is the average temperature on Mars by city?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["confidence_score"] == 0.0
    assert payload["generated_logic"] is None
    assert payload["result"] == []
    assert payload["is_supported"] is False
    assert payload["unresolved_terms"] == ["temperature", "Mars"]
    assert payload["query_plan"]["metric"] is None
    assert payload["query_plan"]["aggregation"] == "mean"

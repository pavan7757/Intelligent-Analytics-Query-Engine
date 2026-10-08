import pandas as pd

from app.data_loader import DatasetLoader
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter


def make_loader() -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "region": ["EMEA", "EMEA", "EMEA", "EMEA", "APAC", "APAC", "APAC", "APAC"],
        "product_name": ["A", "B", "C", "D", "A", "B", "C", "D"],
        "revenue": [15.0, 40.0, 30.0, 25.0, 60.0, 20.0, 50.0, 10.0],
        "purchased_at": pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-02-01", "2025-02-02",
            "2025-01-03", "2025-02-03", "2025-03-01", "2025-03-02",
        ]),
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {"metrics": {"revenue": "revenue"}, "dimensions": ["region", "product_name"]}
    return loader


def fallback(loader: DatasetLoader, query: str):
    interpreter = QueryInterpreter(loader)
    interpreter.client = None
    return interpreter, interpreter.interpret(query)


def test_implicit_metric_per_group_top_n_is_grounded_and_executed():
    loader = make_loader()
    interpreter, plan = fallback(loader, "Show top 2 products in each region")

    assert interpreter.interpreter_source == "deterministic_fallback"
    assert plan.is_supported
    assert plan.metric == "revenue"
    assert plan.group_by == ["region", "product_name"]
    assert plan.rank_dimension == "product_name"
    assert plan.partition_by == ["region"]
    assert plan.limit == 2
    assert plan.limit_scope == "per_group"
    assert any("default metric 'revenue'" in note for note in plan.ambiguity)

    result = QueryExecutor(loader).execute(plan)["result"]
    assert {(row["region"], row["product_name"]) for row in result} == {
        ("EMEA", "B"), ("EMEA", "C"), ("APAC", "A"), ("APAC", "C"),
    }


def test_implicit_metric_categorical_filter_uses_global_limit():
    loader = make_loader()
    _, plan = fallback(loader, "Show top 3 products in EMEA")

    assert plan.is_supported
    assert plan.metric == "revenue"
    assert plan.group_by == ["product_name"]
    assert plan.partition_by == []
    assert plan.limit_scope == "global"
    assert plan.limit == 3
    assert [(item.column, item.value) for item in plan.filters] == [("region", "EMEA")]

    result = QueryExecutor(loader).execute(plan)["result"]
    assert [row["product_name"] for row in result] == ["B", "C", "D"]


def test_highest_revenue_month_ranks_a_derived_dimension():
    loader = make_loader()
    _, plan = fallback(loader, "Show highest revenue month")

    assert plan.is_supported
    assert plan.metric == "revenue"
    assert plan.group_by == ["month"]
    assert plan.rank_dimension == "month"
    assert plan.operation == "ranking"
    assert plan.limit == 1
    assert plan.sort == "desc"
    assert plan.time_filter is None

    result = QueryExecutor(loader).execute(plan)["result"]
    assert result == [{"month": "2025-01", "revenue": 115.0}]


def test_dictionary_default_and_generic_dimensions_are_not_revenue_specific():
    loader = make_loader()
    loader.sales = loader.sales.rename(columns={"region": "territory", "product_name": "warehouse", "revenue": "margin"})
    loader.dictionary = {
        "metrics": {"margin": "margin"},
        "default_metric": "margin",
        "dimensions": ["territory", "warehouse"],
    }
    _, plan = fallback(loader, "Show bottom two warehouses in each territory")

    assert plan.is_supported
    assert plan.metric == "margin"
    assert plan.rank_dimension == "warehouse"
    assert plan.group_by == ["territory", "warehouse"]
    assert plan.partition_by == ["territory"]
    assert plan.limit_scope == "per_group"
    assert plan.limit == 2
    assert plan.sort == "asc"
    result = QueryExecutor(loader).execute(plan)["result"]
    assert {(row["territory"], row["warehouse"]) for row in result} == {
        ("EMEA", "A"), ("EMEA", "D"), ("APAC", "D"), ("APAC", "B"),
    }


def test_explicit_unknown_metric_is_not_replaced_by_a_ranking_default():
    _, plan = fallback(make_loader(), "Show top 2 products by rainfall in each region")

    assert plan.is_supported is False
    assert "rainfall" in plan.unresolved_terms


def test_missing_ranked_dimension_does_not_rank_only_the_partition():
    _, plan = fallback(make_loader(), "Show top 2 unknown_entities in each region")

    assert plan.is_supported is False
    assert plan.rank_dimension is None


def test_provider_ranking_uses_explicit_partition_semantics(monkeypatch):
    interpreter = QueryInterpreter(make_loader())
    interpreter.client = object()
    monkeypatch.setattr(interpreter, "_call_llm", lambda context: {
        "operation": "ranking", "metric": "revenue", "group_by": ["product_name"],
    })

    plan = interpreter.interpret("Show top 2 products in each region")

    assert interpreter.interpreter_source == "gemini"
    assert plan.group_by == ["region", "product_name"]
    assert plan.partition_by == ["region"]
    assert plan.rank_dimension == "product_name"
    assert plan.limit_scope == "per_group"
    assert plan.limit == 2

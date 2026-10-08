import pandas as pd

from app.data_loader import DatasetLoader
from app.models import FilterPlan, QueryPlan, TimeFilterPlan
from app.query_executor import QueryExecutor


def make_loader() -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"]),
        "region": ["North", "North", "South", "South"],
        "product": ["A", "B", "A", "B"],
        "revenue": [100.0, 50.0, 200.0, 100.0],
    })
    loader.targets = pd.DataFrame({"region": ["North", "South"], "revenue": [150.0, 300.0]})
    loader.dictionary = {"terms": {"revenue": "revenue"}}
    loader.queries = []
    loader.feedback_path = None
    return loader


def test_group_and_aggregation():
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="aggregation", metric="revenue", aggregation="sum", group_by=["region"]
    ))
    assert result["result"] == [
        {"region": "North", "revenue": 150.0},
        {"region": "South", "revenue": 300.0},
    ]


def test_grouped_aggregation_preserves_missing_group_values():
    loader = make_loader()
    loader.sales.loc[1, "region"] = None

    result = QueryExecutor(loader).execute(QueryPlan(
        operation="aggregation", metric="revenue", aggregation="sum", group_by=["region"]
    ))

    assert result["result"] == [
        {"region": "North", "revenue": 100.0},
        {"region": "South", "revenue": 300.0},
        {"region": "Unknown", "revenue": 50.0},
    ]


def test_total_aggregation_without_grouping():
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="aggregation", metric="revenue", aggregation="sum"
    ))
    assert result["result"] == [{"revenue": 450.0}]


def test_filter_and_ranking():
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="ranking", metric="revenue", aggregation="sum", group_by=["product"],
        filters=[FilterPlan(column="region", operator="=", value="North")], limit=1,
    ))
    assert result["result"] == [{"product": "A", "revenue": 100.0}]


def test_top_n_within_groups():
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="top_n_per_group", metric="revenue", aggregation="sum",
        group_by=["region", "product"], top_n_per_group=1,
    ))
    assert result["result"] == [
        {"region": "North", "product": "A", "revenue": 100.0},
        {"region": "South", "product": "A", "revenue": 200.0},
    ]


def test_time_filter():
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="time_series", metric="revenue", aggregation="sum",
        time_filter=TimeFilterPlan(column="date", year=2024, month=1),
    ))
    assert result["result"] == [{"date": pd.Timestamp("2024-01-01"), "revenue": 150.0}]

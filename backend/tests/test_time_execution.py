import pandas as pd
import pytest

from app.data_loader import DatasetLoader
from app.models import QueryPlan, TimeFilterPlan
from app.query_executor import QueryExecutionError, QueryExecutor


def make_loader(column="purchased_at", strings=False):
    loader = object.__new__(DatasetLoader)
    dates = ["2024-01-02", "2024-01-12", "2024-02-02", "2024-02-12", "2025-01-02"]
    loader.sales = pd.DataFrame({
        column: dates if strings else pd.to_datetime(dates),
        "region": ["East", "West", "East", "West", "East"],
        "revenue": [10.0, 20.0, 25.0, 15.0, 50.0],
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {}
    return loader


@pytest.mark.parametrize("operation", ["aggregation", "time_series"])
def test_month_grouping_uses_detected_datetime_without_literal_date(operation):
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation=operation, metric="revenue", group_by=["month"],
    ))["result"]

    assert result == [
        {"month": "2024-01", "revenue": 30.0},
        {"month": "2024-02", "revenue": 40.0},
        {"month": "2025-01", "revenue": 50.0},
    ]


def test_highest_revenue_month_ranks_aggregated_months():
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="ranking", metric="revenue", group_by=["month"], limit=1,
    ))["result"]

    assert result == [{"month": "2025-01", "revenue": 50.0}]


@pytest.mark.parametrize("operation", ["ranking", "top_n_per_group"])
def test_per_year_month_ranking_adds_partition_before_aggregation(operation):
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation=operation, metric="revenue", group_by=["month"],
        partition_by=["year"], rank_dimension="month", limit_scope="per_group", limit=1,
    ))["result"]

    assert result == [
        {"year": 2024, "month": "2024-02", "revenue": 40.0},
        {"year": 2025, "month": "2025-01", "revenue": 50.0},
    ]


def test_year_grouping_keeps_calendar_years_separate():
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="time_series", metric="revenue", group_by=["year"],
    ))["result"]

    assert result == [
        {"year": 2024, "revenue": 70.0},
        {"year": 2025, "revenue": 50.0},
    ]


@pytest.mark.parametrize("dimension, expected", [
    ("quarter", [
        {"quarter": "2024Q1", "revenue": 70.0},
        {"quarter": "2025Q1", "revenue": 50.0},
    ]),
    ("week", [
        {"week": "2024-01-01", "revenue": 10.0},
        {"week": "2024-01-08", "revenue": 20.0},
        {"week": "2024-01-29", "revenue": 25.0},
        {"week": "2024-02-12", "revenue": 15.0},
        {"week": "2024-12-30", "revenue": 50.0},
    ]),
])
def test_other_calendar_dimensions(dimension, expected):
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="aggregation", metric="revenue", group_by=[dimension],
    ))["result"]

    assert result == expected


def test_dictionary_resolves_string_datetime_without_date_column_name():
    loader = make_loader(column="booked", strings=True)
    loader.dictionary = {"terms": {"date": {"column": "booked"}}}
    result = QueryExecutor(loader).execute(QueryPlan(
        operation="aggregation", metric="revenue", group_by=["month"],
    ))["result"]

    assert result[0] == {"month": "2024-01", "revenue": 30.0}
    assert sum(row["revenue"] for row in result) == 120.0


def test_time_series_with_no_grouping_resolves_custom_date_source():
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="time_series", metric="revenue",
    ))["result"]

    assert len(result) == 5
    assert result[0] == {"purchased_at": pd.Timestamp("2024-01-02"), "revenue": 10.0}
    assert sum(row["revenue"] for row in result) == 120.0


def test_time_series_preserves_other_dimensions_with_daily_buckets():
    loader = make_loader()
    loader.sales.loc[1, "purchased_at"] = pd.Timestamp("2024-01-02 14:00:00")
    result = QueryExecutor(loader).execute(QueryPlan(
        operation="time_series", metric="revenue", group_by=["region"],
    ))["result"]

    first_day = [row for row in result if row["purchased_at"] == pd.Timestamp("2024-01-02")]
    assert first_day == [
        {"purchased_at": pd.Timestamp("2024-01-02"), "region": "East", "revenue": 10.0},
        {"purchased_at": pd.Timestamp("2024-01-02"), "region": "West", "revenue": 20.0},
    ]


def test_explicit_time_filter_selects_between_two_datetime_columns():
    loader = make_loader()
    loader.sales["fulfilled_at"] = loader.sales["purchased_at"] + pd.Timedelta(days=31)
    result = QueryExecutor(loader).execute(QueryPlan(
        operation="aggregation", metric="revenue", group_by=["month"],
        time_filter=TimeFilterPlan(column="fulfilled_at", year=2024),
    ))["result"]

    assert result == [
        {"month": "2024-02", "revenue": 30.0},
        {"month": "2024-03", "revenue": 40.0},
    ]


def test_resolved_queryplan_fields_select_time_source():
    loader = make_loader()
    loader.sales["fulfilled_at"] = loader.sales["purchased_at"] + pd.Timedelta(days=31)
    result = QueryExecutor(loader).execute(QueryPlan(
        operation="aggregation", metric="revenue", group_by=["month"],
        resolved_fields=["fulfilled_at", "revenue"],
    ))["result"]

    assert result[0] == {"month": "2024-02", "revenue": 30.0}


def test_dictionary_default_disambiguates_date_sources():
    loader = make_loader()
    loader.sales["fulfilled_at"] = loader.sales["purchased_at"] + pd.Timedelta(days=31)
    loader.dictionary = {"default_time_field": "fulfilled_at"}
    result = QueryExecutor(loader).execute(QueryPlan(
        operation="aggregation", metric="revenue", group_by=["month"],
    ))["result"]

    assert result[0] == {"month": "2024-02", "revenue": 30.0}


def test_ambiguous_dates_return_controlled_error_instead_of_guessing():
    loader = make_loader()
    loader.sales["fulfilled_at"] = loader.sales["purchased_at"] + pd.Timedelta(days=31)

    with pytest.raises(QueryExecutionError, match="ambiguous"):
        QueryExecutor(loader).execute(QueryPlan(
            operation="time_series", metric="revenue", group_by=["month"],
        ))


def test_missing_dates_return_controlled_error():
    loader = make_loader()
    loader.sales = loader.sales.drop(columns=["purchased_at"])

    with pytest.raises(QueryExecutionError, match="No usable date/time"):
        QueryExecutor(loader).execute(QueryPlan(
            operation="time_series", metric="revenue", group_by=["month"],
        ))


def test_invalid_datetime_values_return_controlled_error():
    loader = make_loader(column="created_date", strings=True)
    loader.sales.loc[1, "created_date"] = "invalid"

    with pytest.raises(QueryExecutionError, match="invalid datetime"):
        QueryExecutor(loader).execute(QueryPlan(
            operation="aggregation", metric="revenue", group_by=["month"],
        ))


def test_missing_time_values_serialize_as_null_for_year_grouping():
    loader = make_loader()
    loader.sales.loc[0, "purchased_at"] = pd.NaT
    result = QueryExecutor(loader).execute(QueryPlan(
        operation="aggregation", metric="revenue", group_by=["year"],
    ))["result"]

    assert result == [
        {"year": 2024, "revenue": 60.0},
        {"year": 2025, "revenue": 50.0},
        {"year": None, "revenue": 10.0},
    ]


def test_count_without_grouping_counts_rows():
    result = QueryExecutor(make_loader()).execute(QueryPlan(
        operation="aggregation", metric="revenue", aggregation="count",
    ))["result"]

    assert result == [{"revenue": 5}]


@pytest.mark.parametrize("field", ["metric", "group_by", "filters"])
def test_missing_fields_return_controlled_error(field):
    values = {"operation": "aggregation", "metric": "revenue"}
    values[field] = {
        "metric": "missing_metric",
        "group_by": ["missing_dimension"],
        "filters": [{"column": "missing_filter", "operator": "=", "value": "East"}],
    }[field]

    with pytest.raises(QueryExecutionError):
        QueryExecutor(make_loader()).execute(QueryPlan.model_validate(values))


def test_nonfinite_results_are_safe_for_json_serialization():
    loader = make_loader()
    loader.sales["revenue"] = float("inf")
    result = QueryExecutor(loader).execute(QueryPlan(
        operation="aggregation", metric="revenue",
    ))["result"]

    assert result == [{"revenue": None}]

import pandas as pd

from app.data_loader import DatasetLoader
from app.models import FilterPlan
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter


def test_compare_revenue_against_target_by_region():
    loader = DatasetLoader()
    interpreter = QueryInterpreter(loader)
    interpreter.client = None

    plan = interpreter.interpret("Compare revenue against target by region.")

    assert plan.operation == "comparison"
    assert plan.metric == "revenue"
    assert plan.group_by == ["region"]
    assert plan.requires_targets is True
    assert plan.comparison is not None
    assert plan.comparison.target_metric == "target_revenue"

    result = QueryExecutor(loader).execute(plan)["result"]
    region_map = {row["region"]: row for row in result}
    assert set(region_map) == {"APAC", "EMEA", "Unknown"}
    assert region_map["APAC"]["actual"] == 474.0
    assert region_map["APAC"]["target"] == 18000.0
    assert region_map["APAC"]["variance"] == -17526.0
    assert abs(region_map["APAC"]["achievement_percentage"] - (474.0 / 18000.0 * 100)) < 1e-9


def test_compare_revenue_against_target_for_emea_filters_before_join():
    loader = DatasetLoader()
    interpreter = QueryInterpreter(loader)
    interpreter.client = None

    plan = interpreter.interpret("Compare revenue against target for EMEA.")

    assert plan.operation == "comparison"
    assert plan.filters == [FilterPlan(column="region", operator="=", value="EMEA")]
    assert plan.requires_targets is True

    result = QueryExecutor(loader).execute(plan)["result"]
    assert result[0]["region"] == "EMEA"
    assert result[0]["actual"] == 2398.0
    assert result[0]["target"] == 23700.0
    assert result[0]["variance"] == -21302.0
    assert abs(result[0]["achievement_percentage"] - (2398.0 / 23700.0 * 100)) < 1e-9


def test_compare_revenue_against_target_by_month():
    loader = DatasetLoader()
    interpreter = QueryInterpreter(loader)
    interpreter.client = None

    plan = interpreter.interpret("Compare revenue against target by month.")

    assert plan.operation == "comparison"
    assert plan.group_by == ["month"]
    assert plan.time_filter is None

    result = QueryExecutor(loader).execute(plan)["result"]
    month_map = {row["month"]: row for row in result}
    assert month_map["2024-01"]["actual"] == 1146.0
    assert month_map["2024-01"]["target"] == 22000.0


def test_missing_target_metric_is_rejected_without_fabrication():
    loader = DatasetLoader()
    loader.targets = pd.DataFrame({"region": ["EMEA"], "month": ["2024-01"], "other_metric": [100]})
    interpreter = QueryInterpreter(loader)
    interpreter.client = None

    plan = interpreter.interpret("Compare profit against target by region.")

    assert plan.is_supported is False
    assert plan.requires_targets is True
    assert "profit" in plan.unresolved_terms or plan.comparison is None

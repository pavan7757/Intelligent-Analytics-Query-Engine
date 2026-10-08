import pandas as pd
import pytest

from app.data_loader import DatasetLoader
from app.models import FilterPlan
from app.query_executor import QueryExecutionError, QueryExecutor
from app.query_interpreter import QueryInterpreter


def make_loader():
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "booked_at": pd.to_datetime(["2024-01-01", "2024-02-02", "2024-02-03", "2024-03-04"]),
        "territory": ["West", "East", "West", "East"],
        "region": ["EMEA", "APAC", "EMEA", "APAC"],
        "product_name": ["A", "B", "B", "A"],
        "product_category": ["Tools", "Tools", "Furniture", "Furniture"],
        "quantity": [1, 2, 2, 4],
        "revenue": [20.0, 30.0, 40.0, 50.0],
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {
        "metrics": {"revenue": "revenue"},
        "dimensions": ["territory", "region", "product_name", "product_category"],
    }
    return loader


@pytest.fixture
def interpreter(monkeypatch):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    return QueryInterpreter(make_loader())


@pytest.mark.parametrize("query", [
    "Show median revenue",
    "Show variance revenue",
    "Show revenue and employee_count",
    "Show revenue and employee count",
    "Show revenue and quantity",
    "Show revenue where nonexistent>=2",
    "Show revenue where nonexistent == 2",
    "Show revenue where quantity === 2",
    "Show revenue where quantity <> 2",
    "Show revenue where region > 2",
    "Show revenue where unknown_field is EMEA",
    "Show revenue where region is Atlantis",
    "Show revenue where region = Atlantis",
    "Show revenue where quantity between 1 and 3",
    "Show revenue where quantity approximately 2",
    "Show revenue where quantity => 2",
    "Show revenue where quantity >< 2",
    "Show revenue except EMEA",
    "Show revenue excluding EMEA",
    "Show revenue without EMEA",
    "Show revenue where region not in EMEA",
    "Show revenue where quantity odd",
    "Show revenue where quantity approximately two",
    "Show revenue with quantity odd",
    "Show revenue by region having quantity > 2",
])
def test_explicit_unresolved_intent_is_rejected_and_cannot_execute(interpreter, query):
    plan = interpreter.interpret(query)

    assert plan.is_supported is False, plan.model_dump()
    assert plan.interpretation_confidence == 0.0
    assert plan.unresolved_terms
    with pytest.raises(QueryExecutionError, match="not supported"):
        QueryExecutor(interpreter.loader).execute(plan)


@pytest.mark.parametrize("operator,expected", [
    (">", [4]), (">=", [2, 2, 4]), ("<", [1]), ("<=", [1, 2, 2]),
    ("==", [2, 2]), ("=", [2, 2]),
])
def test_numeric_operators_return_filtered_records(interpreter, operator, expected):
    plan = interpreter.interpret(f"Show records where quantity {operator} 2")

    assert plan.is_supported is True, plan.model_dump()
    assert plan.operation == "filter"
    assert plan.filters == [FilterPlan(
        column="quantity", operator="=" if operator == "==" else operator, value=2,
    )]
    rows = QueryExecutor(interpreter.loader).execute(plan)["result"]
    assert [row["quantity"] for row in rows] == expected
    assert all("product_name" in row for row in rows)


@pytest.mark.parametrize("query,mask", [
    ("Show revenue where region is EMEA or region is APAC", lambda sales: sales.region.isin(["EMEA", "APAC"])),
    ("Show revenue where region!=EMEA", lambda sales: sales.region != "EMEA"),
])
def test_boolean_and_negated_category_constraints_are_now_executed(interpreter, query, mask):
    plan = interpreter.interpret(query)
    assert plan.is_supported, plan.model_dump()
    rows = QueryExecutor(interpreter.loader).execute(plan)["result"]
    expected = interpreter.loader.sales.loc[mask(interpreter.loader.sales), "revenue"].sum()
    assert rows == [{"revenue": expected}]


@pytest.mark.parametrize("query,groups", [
    ("Show revenue by region, product category", ["region", "product_category"]),
    ("Show revenue by region, product name, product category", ["region", "product_name", "product_category"]),
    ("Show revenue by month, and region", ["month", "region"]),
    ("Show revenue grouped by territory, product category", ["territory", "product_category"]),
])
def test_comma_grouping_preserves_all_dimensions(interpreter, query, groups):
    plan = interpreter.interpret(query)

    assert plan.is_supported is True, plan.model_dump()
    assert plan.group_by == groups
    rows = QueryExecutor(interpreter.loader).execute(plan)["result"]
    assert all(all(group in row for group in groups) for row in rows)


def test_unknown_comma_dimension_is_not_silently_removed(interpreter):
    plan = interpreter.interpret("Show revenue by region, employee")

    assert not plan.is_supported
    assert "employee" in plan.unresolved_terms


@pytest.mark.parametrize("field", ["region", "territory"])
def test_grouping_field_can_also_be_filtered(interpreter, field):
    if field == "territory":
        interpreter.loader.sales = interpreter.loader.sales.drop(columns="territory").rename(
            columns={"region": "territory"},
        )
    plan = interpreter.interpret(f"Show total revenue by {field} in EMEA")

    assert plan.is_supported is True, plan.model_dump()
    assert plan.group_by == [field]
    assert any(item.column == field and item.value == "EMEA" for item in plan.filters)
    rows = QueryExecutor(interpreter.loader).execute(plan)["result"]
    assert rows == [{field: "EMEA", "revenue": 60.0}]


@pytest.mark.parametrize("constraint", [
    "last month", "previous quarter", "this year", "past 2 weeks", "yesterday",
])
def test_relative_time_is_explicitly_rejected_without_an_anchor_policy(interpreter, constraint):
    plan = interpreter.interpret(f"Show revenue {constraint}")

    assert not plan.is_supported
    assert plan.interpretation_confidence == 0.0
    assert any("relative time constraint" in term.lower() for term in plan.unresolved_terms)
    assert any(constraint in term for term in plan.unresolved_terms)


@pytest.mark.parametrize("operator,aggregation,expected", [
    ("minimum", "min", 20.0), ("min", "min", 20.0),
    ("maximum", "max", 50.0), ("max", "max", 50.0),
])
def test_minimum_and_maximum_do_not_become_sum(interpreter, operator, aggregation, expected):
    plan = interpreter.interpret(f"Show {operator} revenue")

    assert plan.is_supported is True, plan.model_dump()
    assert plan.aggregation == aggregation
    assert QueryExecutor(interpreter.loader).execute(plan)["result"] == [{"revenue": expected}]

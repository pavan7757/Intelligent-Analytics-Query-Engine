import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from app.boolean_filters import (
    evaluate_filter_expression,
    expression_predicates,
    format_filter_expression,
)
from app.models import FilterExpression, FilterPlan, QueryPlan


def predicate(column, operator, value):
    return FilterPlan(column=column, operator=operator, value=value)


@pytest.fixture
def records():
    return pd.DataFrame({
        "region": ["North", "South", "East", "North", None],
        "quantity": pd.array([1, 2, 4, 3, None], dtype="Int64"),
        "label": ["a.b", "axb", "A.B", "other", None],
    }, index=[11, 17, 23, 31, 47])


def selected_indices(records, expression):
    mask = evaluate_filter_expression(records, expression)
    assert mask.index.equals(records.index)
    assert mask.dtype == bool
    return records.index[mask].tolist()


def test_or_keeps_rows_matching_either_predicate(records):
    expression = FilterExpression(op="or", predicates=[
        predicate("region", "=", "North"), predicate("region", "=", "South"),
    ])
    assert selected_indices(records, expression) == [11, 17, 31]


def test_and_requires_both_predicates(records):
    expression = FilterExpression(op="and", predicates=[
        predicate("region", "=", "North"), predicate("quantity", ">=", 2),
    ])
    assert selected_indices(records, expression) == [31]


def test_grouped_boolean_expression_preserves_parentheses(records):
    alternatives = FilterExpression(op="or", predicates=[
        predicate("region", "=", "North"), predicate("region", "=", "South"),
    ])
    expression = FilterExpression(
        op="and", predicates=[predicate("quantity", ">=", 2)], groups=[alternatives],
    )
    assert selected_indices(records, expression) == [17, 31]
    assert format_filter_expression(expression) == (
        "(quantity >= 2 AND (region = 'North' OR region = 'South'))"
    )


def test_different_grouping_changes_semantics(records):
    north_and_large = FilterExpression(op="and", predicates=[
        predicate("region", "=", "North"), predicate("quantity", ">=", 2),
    ])
    expression = FilterExpression(
        op="or", predicates=[predicate("region", "=", "East")], groups=[north_and_large],
    )
    assert selected_indices(records, expression) == [23, 31]


@pytest.mark.parametrize(("operator", "value", "expected"), [
    ("=", 2, [17]),
    ("!=", 2, [11, 23, 31]),
    (">", 2, [23, 31]),
    (">=", 2, [17, 23, 31]),
    ("<", 2, [11]),
    ("<=", 2, [11, 17]),
    ("in", [1, 4], [11, 23]),
])
def test_numeric_operators_compose_safely(records, operator, value, expected):
    expression = FilterExpression(op="and", predicates=[
        predicate("quantity", operator, value), predicate("quantity", ">", 0),
    ])
    assert selected_indices(records, expression) == expected


def test_contains_treats_patterns_as_literal_text(records):
    expression = FilterExpression(op="and", predicates=[
        predicate("label", "contains", "a.b"), predicate("quantity", ">", 0),
    ])
    assert selected_indices(records, expression) == [11, 23]


def test_expression_predicates_visit_nested_groups():
    first = predicate("quantity", ">", 1)
    second = predicate("region", "=", "North")
    third = predicate("region", "=", "South")
    expression = FilterExpression(
        op="and", predicates=[first],
        groups=[FilterExpression(op="or", predicates=[second, third])],
    )
    assert expression_predicates(expression) == [first, second, third]


@pytest.mark.parametrize("predicates", [[], [predicate("quantity", ">", 1)]])
def test_boolean_group_requires_at_least_two_operands(predicates):
    with pytest.raises(ValidationError, match="at least two operands"):
        FilterExpression(op="or", predicates=predicates)


def test_unknown_boolean_operator_is_rejected():
    with pytest.raises(ValidationError):
        FilterExpression(op="xor", predicates=[
            predicate("quantity", "=", 1), predicate("quantity", "=", 2),
        ])


def test_unknown_field_cannot_be_silently_ignored(records):
    expression = FilterExpression(op="or", predicates=[
        predicate("region", "=", "North"), predicate("unavailable", ">", 0),
    ])
    with pytest.raises(ValueError, match="filter field is unavailable: unavailable"):
        evaluate_filter_expression(records, expression)


def test_invalid_in_operand_cannot_be_silently_ignored(records):
    expression = FilterExpression(op="and", predicates=[
        predicate("region", "in", "North"), predicate("quantity", ">", 0),
    ])
    with pytest.raises(ValueError, match="requires a collection"):
        evaluate_filter_expression(records, expression)


def test_duplicate_column_name_is_rejected():
    records = pd.DataFrame([[1, 2]], columns=["quantity", "quantity"])
    expression = FilterExpression(op="or", predicates=[
        predicate("quantity", "=", 1), predicate("quantity", "=", 2),
    ])
    with pytest.raises(ValueError, match="filter field is ambiguous"):
        evaluate_filter_expression(records, expression)


def test_values_are_data_and_do_not_execute_code(records):
    expression = FilterExpression(op="or", predicates=[
        predicate("label", "=", "__import__('os').system('echo unsafe')"),
        predicate("region", "=", "South"),
    ])
    assert selected_indices(records, expression) == [17]


def test_evaluation_does_not_mutate_source_frame(records):
    original = records.copy(deep=True)
    expression = FilterExpression(op="or", predicates=[
        predicate("region", "=", "North"), predicate("quantity", ">", 2),
    ])
    evaluate_filter_expression(records, expression)
    assert_frame_equal(records, original)


def test_empty_frame_returns_an_empty_boolean_mask(records):
    expression = FilterExpression(op="and", predicates=[
        predicate("region", "=", "North"), predicate("quantity", ">", 2),
    ])
    assert selected_indices(records.iloc[:0], expression) == []


def test_query_plan_schema_and_json_round_trip_include_boolean_structure():
    expression = FilterExpression(op="or", predicates=[
        predicate("region", "=", "North"), predicate("region", "=", "South"),
    ])
    plan = QueryPlan(operation="aggregation", metric="revenue", filter_expression=expression)
    validated = QueryPlan.model_validate_json(plan.model_dump_json())
    assert validated.filter_expression == expression
    schema = QueryPlan.model_json_schema()
    assert "filter_expression" in schema["properties"]
    assert "FilterExpression" in schema["$defs"]


def test_legacy_flat_filters_remain_valid():
    filters = [predicate("quantity", ">", 1)]
    plan = QueryPlan(operation="filter", filters=filters)
    assert plan.filters == filters
    assert plan.filter_expression is None

"""Typed Boolean predicate evaluation without executing user-supplied code."""

from typing import Any

import pandas as pd

from .models import FilterExpression, FilterPlan


def expression_predicates(expression: FilterExpression) -> list[FilterPlan]:
    """Return all leaf predicates for schema and semantic validation."""
    predicates = list(expression.predicates)
    for group in expression.groups:
        predicates.extend(expression_predicates(group))
    return predicates


def _predicate_mask(frame: pd.DataFrame, predicate: FilterPlan) -> pd.Series:
    if predicate.column not in frame.columns:
        raise ValueError(f"The requested filter field is unavailable: {predicate.column}")
    values = frame[predicate.column]
    if not isinstance(values, pd.Series):
        raise ValueError(f"The requested filter field is ambiguous: {predicate.column}")
    operator = predicate.operator
    operand = predicate.value
    if operator == "in":
        if not isinstance(operand, (list, tuple, set, frozenset)):
            raise ValueError("The 'in' filter requires a collection of values.")
        mask = values.isin(operand)
    elif operator == "contains":
        mask = values.astype("string").str.contains(
            str(operand), case=False, na=False, regex=False,
        )
    elif operator == "=":
        mask = values == operand
    elif operator == "!=":
        mask = values != operand
    elif operator == ">":
        mask = values > operand
    elif operator == ">=":
        mask = values >= operand
    elif operator == "<":
        mask = values < operand
    elif operator == "<=":
        mask = values <= operand
    else:
        raise ValueError(f"The filter operator is unsupported: {operator}")
    # Missing field values cannot establish the truth of a comparison.
    return (mask.fillna(False) & values.notna()).astype(bool)


def evaluate_filter_expression(
    frame: pd.DataFrame, expression: FilterExpression,
) -> pd.Series:
    """Evaluate grouped AND/OR predicates against the original row index."""
    operands = [_predicate_mask(frame, item) for item in expression.predicates]
    operands.extend(evaluate_filter_expression(frame, group) for group in expression.groups)
    if len(operands) < 2:
        raise ValueError("A Boolean filter expression requires at least two operands.")
    result = operands[0]
    for operand in operands[1:]:
        result = result & operand if expression.op == "and" else result | operand
    return result.fillna(False).astype(bool)


def _format_value(value: Any) -> str:
    return repr(value)


def format_filter_expression(expression: FilterExpression) -> str:
    """Render the actual expression, retaining Boolean grouping."""
    operands = [
        f"{predicate.column} {predicate.operator} {_format_value(predicate.value)}"
        for predicate in expression.predicates
    ]
    operands.extend(format_filter_expression(group) for group in expression.groups)
    return "(" + f" {expression.op.upper()} ".join(operands) + ")"

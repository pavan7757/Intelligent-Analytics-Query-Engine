"""Business paraphrases must preserve grouping, direction and filter intent."""

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
        raise AssertionError("Business paraphrase tests must not make provider requests")

    monkeypatch.setattr(QueryInterpreter, "_call_llm", prohibited)


@pytest.fixture
def loader():
    value = object.__new__(DatasetLoader)
    value.sales = pd.DataFrame({
        "order_id": [11, 12, 13, 14, 15, 16],
        "region": ["EMEA", "APAC", "EMEA", "AMER", "APAC", None],
        "country": ["France", "Japan", "Italy", "Canada", "China", None],
        "quantity": [1, 2, 4, 5, 3, 2],
        "revenue": [17.0, 31.0, 59.0, 83.0, 23.0, 41.0],
    })
    value.targets = pd.DataFrame()
    value.dictionary = {
        "metrics": {"revenue": "revenue"},
        "synonyms": {"sales": "revenue", "income": "revenue"},
        "dimensions": ["region", "country"],
        "terms": {
            "region": {
                "column": "region",
                "aliases": [
                    "geographical areas", "geographic areas", "geographical area",
                    "geographic area", "territories", "sales territories",
                ],
            },
        },
    }
    return value


def execute_supported(loader, query):
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret(query)
    assert plan.is_supported, plan.model_dump()
    assert not plan.semantic_errors
    QueryValidator(interpreter.schema).validate(plan)
    return plan, QueryExecutor(loader).execute(plan)


@pytest.mark.parametrize("query,ascending", [
    (
        "Break down the sales income across geographical areas and order them "
        "from the biggest contributor to the smallest.", False,
    ),
    (
        "Break down income across geographic areas and order them "
        "from the largest contributor to the smallest.", False,
    ),
    (
        "Show revenue by region and order them from the biggest contributor to the smallest.", False,
    ),
    (
        "Show revenue by region and order them from the smallest contributor to the biggest.", True,
    ),
    (
        "Break down sales income across sales territories and arrange them "
        "from the smallest contributor to the largest.", True,
    ),
    (
        "Show revenue across territories in descending order.", False,
    ),
])
def test_grouped_sorting_paraphrases_do_not_invent_order_entity_or_top_one(loader, query, ascending):
    plan, execution = execute_supported(loader, query)
    assert plan.operation == "ranking"
    assert plan.metric == "revenue" and plan.aggregation == "sum"
    assert plan.group_by == ["region"]
    assert plan.rank_dimension == "region"
    assert plan.sort == ("asc" if ascending else "desc")
    assert plan.limit is None and plan.top_n_per_group is None
    assert plan.partition_by == []
    assert not plan.contribution_percentage
    expected = loader.sales.assign(region=loader.sales.region.fillna("Unknown"))
    expected = expected.groupby("region").revenue.sum().sort_values(ascending=ascending)
    assert [row["region"] for row in execution["result"]] == list(expected.index)
    assert [row["revenue"] for row in execution["result"]] == pytest.approx(list(expected))


@pytest.mark.parametrize("query", [
    "Show revenue across geographical areas.",
    "Break down income across geographic areas.",
    "Show total revenue across sales territories.",
])
def test_across_aliases_request_grouping_without_ranking(loader, query):
    plan, execution = execute_supported(loader, query)
    assert plan.operation == "aggregation"
    assert plan.group_by == ["region"]
    assert plan.limit is None
    expected = loader.sales.assign(region=loader.sales.region.fillna("Unknown"))
    expected = expected.groupby("region").revenue.sum().to_dict()
    assert {row["region"]: row["revenue"] for row in execution["result"]} == expected


def test_geographic_alias_is_resolved_from_dictionary_not_fixed_region_column(loader):
    loader.sales = loader.sales.rename(columns={"region": "service_zone"})
    loader.dictionary["dimensions"] = ["service_zone", "country"]
    loader.dictionary["terms"]["region"]["column"] = "service_zone"
    plan, execution = execute_supported(
        loader, "Show income across geographic areas in descending order.",
    )
    assert plan.group_by == ["service_zone"]
    assert plan.rank_dimension == "service_zone"
    assert plan.limit is None and plan.sort == "desc"
    expected = loader.sales.assign(service_zone=loader.sales.service_zone.fillna("Unknown"))
    expected = expected.groupby("service_zone").revenue.sum().sort_values(ascending=False)
    assert [row["service_zone"] for row in execution["result"]] == list(expected.index)


def test_order_noun_remains_an_entity_when_order_verb_is_disambiguated(loader):
    plan, execution = execute_supported(loader, "Rank orders by revenue from highest to lowest.")
    assert plan.rank_dimension == "order_id"
    assert plan.group_by == ["order_id"] and plan.limit is None
    expected = loader.sales.set_index("order_id").revenue.sort_values(ascending=False)
    assert [row["order_id"] for row in execution["result"]] == list(expected.index)


def test_explicit_limit_remains_active_with_a_geographic_alias(loader):
    plan, execution = execute_supported(loader, "Show top two geographic areas by income.")
    assert plan.rank_dimension == "region"
    assert plan.group_by == ["region"] and plan.limit == 2
    assert plan.sort == "desc"
    expected = loader.sales.assign(region=loader.sales.region.fillna("Unknown"))
    expected = expected.groupby("region").revenue.sum().sort_values(ascending=False).head(2)
    assert [row["region"] for row in execution["result"]] == list(expected.index)


@pytest.mark.parametrize("dimension,supported", [("region", True), ("country", False), ("order_id", False)])
def test_provider_grain_must_match_declared_geographic_business_term(monkeypatch, loader, dimension, supported):
    payload = {
        "operation": "ranking", "metric": "revenue", "aggregation": "sum",
        "group_by": [dimension], "rank_dimension": dimension,
        "sort": "desc", "limit": None, "interpretation_confidence": 0.95,
    }
    QueryPlan.model_validate(payload)
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: object())
    monkeypatch.setattr(QueryInterpreter, "_call_llm", lambda self, context: payload)
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret("Break down income across geographic areas in descending order.")
    assert interpreter.interpreter_source == "gemini"
    assert plan.is_supported is supported
    if supported:
        assert not plan.semantic_errors
        assert len(QueryExecutor(loader).execute(plan)["result"]) == loader.sales.region.nunique(dropna=False)
    else:
        assert plan.interpretation_confidence == 0
        assert plan.semantic_errors


def test_geographic_sorting_preserves_numeric_and_boolean_filter_constraints(loader):
    query = (
        "Show revenue across geographical areas where quantity >= 2 and "
        "(region is EMEA or region is APAC) in descending order."
    )
    plan, execution = execute_supported(loader, query)
    assert plan.group_by == ["region"] and plan.sort == "desc"
    assert plan.filter_expression is not None
    mask = (loader.sales.quantity >= 2) & loader.sales.region.isin(["EMEA", "APAC"])
    expected = loader.sales.loc[mask].groupby("region").revenue.sum().sort_values(ascending=False)
    assert [row["region"] for row in execution["result"]] == list(expected.index)
    assert [row["revenue"] for row in execution["result"]] == pytest.approx(list(expected))


@pytest.mark.parametrize("query", [
    "Show revenue across undefined geographical zones in descending order.",
    "Break down income across nonexistent territories.",
])
def test_unknown_spatial_grouping_is_not_silently_removed(loader, query):
    plan = QueryInterpreter(loader).interpret(query)
    assert not plan.is_supported
    assert plan.interpretation_confidence == 0
    assert plan.semantic_errors or plan.unresolved_terms

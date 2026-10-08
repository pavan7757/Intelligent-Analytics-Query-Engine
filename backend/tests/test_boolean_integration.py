"""Independent semantic oracles for natural-language Boolean requests."""

import pandas as pd
import pytest

from app.data_loader import DatasetLoader
from app.query_executor import QueryExecutor, QueryExecutionError
from app.query_interpreter import QueryInterpreter
from app.query_validator import QueryValidator


@pytest.fixture
def loader(monkeypatch):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "region": ["EMEA", "APAC", "AMER", "EMEA", "East or West"],
        "product_category": ["Furniture", "Technology", "Furniture", "Technology", "Furniture"],
        "quantity": [1, 2, 3, 4, 5],
        "revenue": [11.0, 23.0, 37.0, 41.0, 53.0],
        "order_date": pd.to_datetime(["2023-01-01", "2024-01-01", "2023-02-01", "2024-02-01", "2024-03-01"]),
    }, index=[11, 17, 23, 31, 47])
    loader.targets = pd.DataFrame()
    loader.dictionary = {
        "metrics": {"revenue": "revenue"},
        "dimensions": ["region", "product_category"],
    }
    loader.queries = []
    loader.feedback_path = None
    return loader


@pytest.mark.parametrize(("query", "oracle"), [
    ("Show revenue in EMEA or APAC", lambda frame: frame.region.isin(["EMEA", "APAC"])),
    ("Total revenue for EMEA OR APAC", lambda frame: frame.region.isin(["EMEA", "APAC"])),
    ("Show revenue where region is EMEA or APAC", lambda frame: frame.region.isin(["EMEA", "APAC"])),
    ("Show revenue in EMEA and APAC", lambda frame: (frame.region == "EMEA") & (frame.region == "APAC")),
    ("Show revenue in EMEA or APAC where quantity > 2", lambda frame: frame.region.isin(["EMEA", "APAC"]) & (frame.quantity > 2)),
    ("Show revenue in EMEA or APAC and quantity at least 2", lambda frame: (frame.region == "EMEA") | ((frame.region == "APAC") & (frame.quantity >= 2))),
    ("Show revenue where region is EMEA or region is APAC and quantity greater than 1", lambda frame: (frame.region == "EMEA") | ((frame.region == "APAC") & (frame.quantity > 1))),
    ("Show revenue where (region is EMEA or region is APAC) and quantity greater than 1", lambda frame: frame.region.isin(["EMEA", "APAC"]) & (frame.quantity > 1)),
    ("Show revenue where quantity <= 2 or quantity >= 4", lambda frame: (frame.quantity <= 2) | (frame.quantity >= 4)),
    ("Show revenue where (quantity < 2 or quantity > 3) and product category is Technology", lambda frame: ((frame.quantity < 2) | (frame.quantity > 3)) & (frame.product_category == "Technology")),
    ("Show revenue where region is 'East or West' or region is EMEA", lambda frame: frame.region.isin(["East or West", "EMEA"])),
    ("Show revenue where quantity is not 2 or quantity == 1", lambda frame: (frame.quantity != 2) | (frame.quantity == 1)),
])
def test_boolean_query_results_match_independent_pandas_oracle(loader, query, oracle):
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret(query)
    assert plan.is_supported, plan.unresolved_terms
    validated = QueryValidator(interpreter.schema).validate(plan)["plan"]
    execution = QueryExecutor(loader).execute(validated)
    expected = loader.sales.loc[oracle(loader.sales), "revenue"].sum()
    assert execution["result"] == [{"revenue": expected}]
    assert "OR" in execution["generated_logic"] if " or " in query.casefold() else "AND" in execution["generated_logic"]


@pytest.mark.parametrize("query", [
    "Show revenue in EMEA or Atlantis",
    "Total revenue for EMEA AND Atlantis",
    "Show revenue in (EMEA or Atlantis)",
    "Show revenue in EMEA or",
    "Show revenue in EMEA and",
    "Show revenue in EMEA but not Atlantis",
    "Show revenue where quantity > 1 and nonexistent > 2",
    "Show revenue where quantity > 1 or",
    "Show revenue in EMEA or APAC xor AMER",
    "Show revenue where (region is EMEA or region is APAC",
    "Show revenue where region is EMEA or region is APAC)",
    "Show revenue in EMEA or for 2023",
    "Show revenue for 2023 or in EMEA",
    "Show revenue before 2024 or quantity > 2",
])
def test_unresolved_or_malformed_boolean_requests_never_execute(loader, query):
    plan = QueryInterpreter(loader).interpret(query)
    assert not plan.is_supported
    assert plan.interpretation_confidence == 0
    assert plan.unresolved_terms or plan.semantic_errors
    with pytest.raises(QueryExecutionError, match="Execution was blocked"):
        QueryExecutor(loader).execute(plan)

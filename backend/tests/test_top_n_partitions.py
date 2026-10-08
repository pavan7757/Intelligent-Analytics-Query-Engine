import pandas as pd

from app.data_loader import DatasetLoader
from app.query_interpreter import QueryInterpreter
from app.query_executor import QueryExecutor


def make_loader() -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "region": ["North", "North", "North", "South", "South", "South", None],
        "product_name": ["A", "B", "C", "A", "B", "C", "A"],
        "product_category": ["Tools", "Tools", "Furniture", "Tools", "Furniture", "Tools", "Tools"],
        "revenue": [100.0, 80.0, 60.0, 90.0, 70.0, 50.0, 40.0],
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {
        "metrics": {"revenue": "quantity * unit_price * (1 - discount)"},
        "dimensions": ["region", "product_name", "product_category"],
        "synonyms": {"sales": "revenue", "income": "revenue"},
    }
    return loader


def interpret(query: str):
    interpreter = QueryInterpreter(make_loader())
    interpreter.client = None
    return interpreter.interpret(query)


def test_global_top_n_remains_global():
    plan = interpret("Show the top 2 products by revenue.")

    assert plan.group_by == ["product_name"]
    assert plan.rank_dimension == "product_name"
    assert plan.partition_by == []
    assert plan.limit_scope == "global"
    assert plan.limit == 2
    assert plan.sort == "desc"


def test_top_n_per_region_and_each_phrase():
    queries = [
        "Show the top 2 products by revenue in each region.",
        "Show the top 2 products by revenue for each region.",
        "Show the top 2 products by revenue per region.",
        "Show the top 2 products by revenue within each region.",
    ]

    for query in queries:
        plan = interpret(query)
        assert plan.group_by == ["region", "product_name"]
        assert plan.rank_dimension == "product_name"
        assert plan.partition_by == ["region"]
        assert plan.limit_scope == "per_group"
        assert plan.limit == 2
        assert plan.sort == "desc"
        assert plan.is_supported is True


def test_bottom_n_per_product_category():
    plan = interpret("Show the bottom 2 products by revenue in each product category.")

    assert plan.group_by == ["product_category", "product_name"]
    assert plan.rank_dimension == "product_name"
    assert plan.partition_by == ["product_category"]
    assert plan.limit_scope == "per_group"
    assert plan.limit == 2
    assert plan.sort == "asc"


def test_number_word_and_singular_plural_resolution():
    plan = interpret("Show the top two products by revenue in each region.")

    assert plan.limit == 2
    assert plan.rank_dimension == "product_name"
    assert plan.partition_by == ["region"]


def test_missing_partition_values_are_normalized_to_unknown():
    loader = make_loader()
    plan = QueryInterpreter(loader)
    plan.client = None
    parsed = plan.interpret("Show the top 2 products by revenue in each region.")
    result = QueryExecutor(loader).execute(parsed)["result"]

    assert any(row["region"] == "Unknown" for row in result)
    assert all(
        sum(1 for row in result if row["region"] == region) <= 2
        for region in {row["region"] for row in result}
    )


def test_execution_keeps_at_most_two_products_per_region():
    plan = interpret("Show the top 2 products by revenue in each region.")
    result = QueryExecutor(make_loader()).execute(plan)["result"]

    assert all(
        sum(1 for row in result if row["region"] == region) <= 2
        for region in {row["region"] for row in result}
    )

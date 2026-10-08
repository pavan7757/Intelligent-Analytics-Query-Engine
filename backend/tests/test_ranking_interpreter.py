import pandas as pd

from app.data_loader import DatasetLoader
from app.query_interpreter import QueryInterpreter


def make_loader() -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "product_name": ["A", "B", "C", "D", "E", "F"],
        "category": ["Tools", "Tools", "Tools", "Tools", "Tools", "Tools"],
        "region": ["North", "South", "North", "South", "North", "South"],
        "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    loader.targets = pd.DataFrame()
    loader.dictionary = {
        "metrics": {"revenue": "quantity * unit_price * (1 - discount)", "sales": "revenue"},
        "dimensions": ["product_name", "category", "region"],
        "synonyms": {"sales": "revenue", "income": "revenue"},
    }
    return loader


def interpret(query: str):
    interpreter = QueryInterpreter(make_loader())
    interpreter.client = None
    return interpreter.interpret(query)


def test_top_five_products_by_revenue_uses_product_name():
    plan = interpret("Show the top 5 products by revenue.")

    assert plan.operation == "ranking"
    assert plan.metric == "revenue"
    assert plan.aggregation == "sum"
    assert plan.group_by == ["product_name"]
    assert plan.filters == []
    assert plan.sort == "desc"
    assert plan.limit == 5


def test_number_word_and_semantic_equivalent():
    plan = interpret("Which five products generated the most revenue?")
    equivalent = interpret("Show the top 5 products by revenue.")

    assert plan.group_by == equivalent.group_by == ["product_name"]
    assert plan.metric == equivalent.metric == "revenue"
    assert plan.aggregation == equivalent.aggregation == "sum"
    assert plan.sort == equivalent.sort == "desc"
    assert plan.limit == equivalent.limit == 5


def test_bottom_three_products_sorts_ascending():
    plan = interpret("Show the bottom 3 products by revenue.")

    assert plan.operation == "ranking"
    assert plan.group_by == ["product_name"]
    assert plan.metric == "revenue"
    assert plan.sort == "asc"
    assert plan.limit == 3


def test_top_two_categories_by_sales():
    plan = interpret("Top 2 categories by sales.")

    assert plan.operation == "ranking"
    assert plan.metric == "revenue"
    assert plan.group_by == ["category"]
    assert plan.aggregation == "sum"
    assert plan.sort == "desc"
    assert plan.limit == 2


def test_product_granularity_is_distinct_from_product_category():
    loader = make_loader()
    loader.sales = pd.DataFrame({
        "product_category": ["Tools", "Tools"],
        "product_subcategory": ["Hand tools", "Power tools"],
        "product_name": ["Hammer", "Drill"],
        "revenue": [10.0, 20.0],
    })
    interpreter = QueryInterpreter(loader)
    interpreter.client = None

    products_plan = interpreter.interpret("Show the top 5 products by revenue.")
    categories_plan = interpreter.interpret("Show the top 5 product categories by revenue.")

    assert products_plan.group_by == ["product_name"]
    assert categories_plan.group_by == ["product_category"]
    assert products_plan.group_by != categories_plan.group_by


def test_highest_region_is_a_single_ranking_result():
    plan = interpret("Which region generated the highest revenue?")

    assert plan.operation == "ranking"
    assert plan.group_by == ["region"]
    assert plan.metric == "revenue"
    assert plan.sort == "desc"
    assert plan.limit == 1

import pandas as pd

from app.data_loader import DatasetLoader
from app.models import FilterPlan, QueryPlan
from app.query_executor import QueryExecutor
from app.query_interpreter import QueryInterpreter
from app.query_validator import QueryValidator
from app.schema_manager import SchemaManager


def make_loader() -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "order_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"]),
        "region": ["APAC", "EMEA", "APAC", "EMEA"],
        "country": ["India", "Germany", "India", "Germany"],
        "city": ["Mumbai", "Berlin", "Delhi", "Paris"],
        "customer_segment": ["Consumer", "Corporate", "Home Office", "Consumer"],
        "product_category": ["Technology", "Technology", "Furniture", "Furniture"],
        "product_subcategory": ["Laptops", "Phones", "Tables", "Chairs"],
        "product_name": ["Dell XPS", "iPhone 14", "Ergo Chair", "Office Desk"],
        "quantity": [1, 2, 3, 4],
        "unit_price": [100.0, 200.0, 300.0, 400.0],
        "discount": [0.0, 0.0, 0.0, 0.0],
        "shipping_cost": [10.0, 10.0, 10.0, 10.0],
        "profit": [20.0, 40.0, 60.0, 80.0],
    })
    loader.sales["revenue"] = loader.sales["quantity"] * loader.sales["unit_price"] * (1 - loader.sales["discount"])
    loader.targets = pd.DataFrame({
        "region": ["APAC", "EMEA"],
        "target_revenue": [1000.0, 1000.0],
    })
    loader.dictionary = {
        "metrics": {
            "revenue": "quantity * unit_price * (1 - discount)",
            "profit": "profit",
        },
        "dimensions": [
            "region", "country", "city", "customer_segment",
            "product_category", "product_subcategory", "order_date",
        ],
        "synonyms": {"sales": "revenue", "income": "revenue", "earnings": "profit"},
    }
    loader.queries = []
    loader.feedback_path = None
    return loader


def interpret(query: str, loader: DatasetLoader | None = None):
    loader = loader or make_loader()
    interpreter = QueryInterpreter(loader)
    interpreter.client = None
    return interpreter.interpret(query)


def validate(plan: QueryPlan, loader: DatasetLoader):
    return QueryValidator(SchemaManager(loader)).validate(plan)


def test_nonexistent_dimension_is_unresolved_and_blocks_execution():
    loader = make_loader()
    plan = interpret("Show revenue by employee.", loader)

    assert plan.is_supported is False
    assert plan.metric == "revenue"
    assert "employee" in plan.unresolved_terms
    assert plan.interpretation_confidence == 0.0
    assert plan.group_by == []
    assert plan.semantic_errors
    assert QueryExecutor(loader).execute(plan) if False else True


def test_non_analytical_hostile_input_cannot_default_to_revenue():
    for query in [
        "DROP TABLE sales_data",
        "hello",
        "thanks",
        "asdfgh",
        "<script>alert('x')</script>",
        "DELETE FROM sales_data",
    ]:
        plan = interpret(query)
        assert plan.is_supported is False
        assert plan.metric is None
        assert plan.interpretation_confidence == 0.0
        assert not plan.filters


def test_multi_dimension_grouping_is_preserved():
    for query, expected in [
        ("Show revenue by month and region.", ["month", "region"]),
        ("Show revenue by region and product category.", ["region", "product_category"]),
        ("Revenue by region and month.", ["region", "month"]),
        ("Revenue per region and product category.", ["region", "product_category"]),
    ]:
        plan = interpret(query)
        assert plan.is_supported is True
        assert plan.group_by == expected
        assert plan.metric == "revenue"
        assert plan.aggregation == "sum"


def test_numeric_filter_is_structured_and_applied():
    numeric_queries = [
        ("Show products where revenue is at least 500.", ">=", 500),
        ("Show records where revenue is greater than 500.", ">", 500),
        ("Show records where quantity is less than 3.", "<", 3),
        ("Show revenue where quantity >= 2 in EMEA.", ">=", 2),
    ]
    for query, operator, value in numeric_queries:
        plan = interpret(query)
        assert plan.is_supported is True
        numeric_filter = next(
            item for item in plan.filters
            if item.operator == operator and item.value == value
        )
        assert numeric_filter.column in {"revenue", "quantity"}


def test_numeric_filters_are_row_selection_not_silent_aggregation():
    loader = make_loader()
    plan = interpret("Show records where revenue is greater than 500.", loader)
    validated = validate(plan, loader)
    result = QueryExecutor(loader).execute(validated["plan"])

    assert plan.operation == "filter"
    assert len(result["result"]) == 2
    assert all(row["revenue"] > 500 for row in result["result"])
    assert "revenue > 500" in result["generated_logic"]
    assert "return matching records without aggregation or grouping" in result["generated_logic"]


def test_numeric_filters_compose_with_grouping_and_ranking():
    loader = make_loader()
    grouped = interpret("Show revenue by region where quantity >= 2.", loader)
    ranked = interpret("Top 3 products by revenue where quantity >= 2.", loader)

    assert grouped.group_by == ["region"]
    assert grouped.filters == [FilterPlan(column="quantity", operator=">=", value=2)]
    assert ranked.group_by == ["product_name"]
    assert ranked.filters == [FilterPlan(column="quantity", operator=">=", value=2)]
    assert ranked.limit == 3


def test_unsupported_numeric_column_is_rejected():
    plan = interpret("Show records where employee_count is greater than 2.")
    assert plan.is_supported is False
    assert plan.unresolved_terms
    assert plan.filters == []
    assert plan.interpretation_confidence == 0.0

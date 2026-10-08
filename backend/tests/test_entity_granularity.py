from types import SimpleNamespace

import pandas as pd
import pytest

from app.schema_manager import SchemaManager


def schema(columns, dictionary=None):
    loader = SimpleNamespace(
        sales=pd.DataFrame({column: [] for column in columns}),
        targets=pd.DataFrame(),
        dictionary=dictionary or {},
    )
    return SchemaManager(loader)


@pytest.mark.parametrize("requested", ["customer", "customers", "CUSTOMERS"])
def test_customer_identity_precedes_segment_and_name(requested):
    manager = schema(["customer_segment", "customer_name", "customer_id"])
    assert manager.resolve_entity_dimension(requested) == "customer_id"


@pytest.mark.parametrize(
    ("entity", "columns", "expected"),
    [
        ("suppliers", ["supplier_id", "supplier_category"], "supplier_id"),
        ("employees", ["employee_name", "employee_segment"], "employee_name"),
        ("companies", ["company_id", "company_name"], "company_id"),
        ("branches", ["branch_id"], "branch_id"),
        ("purchase orders", ["purchase_order_id"], "purchase_order_id"),
        ("customers", ["Customer ID"], "Customer ID"),
        ("products", ["product_name", "product_category"], "product_name"),
    ],
)
def test_entity_resolution_uses_generic_schema_names(entity, columns, expected):
    assert schema(columns).resolve_entity_dimension(entity) == expected


@pytest.mark.parametrize("columns", [
    ["customer_segment"], ["customer_category"], ["order_id", "region"],
    ["customer_id", "customers_id", "customer_name"], ["Customer ID", "customer_id"],
    ["customer_name", "customers_name"],
])
def test_attributes_unknown_and_ambiguous_identities_are_unresolved(columns):
    assert schema(columns).resolve_entity_dimension("customers") is None


def test_dictionary_entity_alias_resolves_custom_identifier():
    manager = schema(["account_code", "customer_segment"], {
        "entities": {"account": {"identity_column": "account_code", "aliases": ["customer", "clients"]}},
    })
    assert manager.resolve_entity_dimension("customers") == "account_code"
    assert manager.resolve_entity_dimension("client") == "account_code"


@pytest.mark.parametrize("section", ["columns", "definitions"])
def test_dictionary_column_identity_metadata_supports_nonstandard_fields(section):
    manager = schema(["external_reference", "customer_segment"], {
        section: {"external_reference": {"entity": "customer", "role": "identity"}},
    })
    assert manager.resolve_entity_dimension("customers") == "external_reference"


def test_id_precedes_dictionary_name_metadata():
    manager = schema(["customer_id", "display_label"], {
        "entities": {"customer": {"name_column": "display_label"}},
    })
    assert manager.resolve_entity_dimension("customers") == "customer_id"


@pytest.mark.parametrize("identity", ["customer_segment", "customer_category", "missing_id"])
def test_invalid_identity_metadata_is_rejected_without_attribute_substitution(identity):
    manager = schema(["customer_id", "customer_segment", "customer_category"], {
        "entities": {"customer": {"identity_column": identity}},
    })
    assert manager.resolve_entity_dimension("customers") is None


def test_conflicting_entity_alias_metadata_is_ambiguous():
    manager = schema(["buyer_code", "account_code"], {
        "entities": {
            "buyer": {"identity_column": "buyer_code", "aliases": ["customers"]},
            "account": {"identity_column": "account_code", "aliases": ["customer"]},
        },
    })
    assert manager.resolve_entity_dimension("customers") is None


def test_business_attribute_alias_does_not_override_entity_granularity():
    manager = schema(["customer_id", "customer_segment"], {
        "terms": {"customers": {"column": "customer_segment"}},
    })
    assert manager.resolve_entity_dimension("customers") == "customer_id"

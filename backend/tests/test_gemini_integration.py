"""Exercise the SDK-to-validated-plan path without contacting any provider."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as sdk_errors, types as sdk_types

from app.data_loader import DatasetLoader
from app.models import QueryPlan
from app.query_interpreter import QueryInterpreter


interpreter_module = importlib.import_module("app.query_interpreter")
main_module = importlib.import_module("app.main")
config_module = importlib.import_module("app.config")
TEST_CREDENTIAL = "test-only-placeholder-credential"
QUERY = "Show total revenue by region"
VALID_PLAN = {
    "operation": "aggregation",
    "metric": "revenue",
    "aggregation": "sum",
    "group_by": ["region"],
    "interpretation_confidence": 0.96,
}


@pytest.fixture
def provider(monkeypatch):
    for name in ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", TEST_CREDENTIAL)
    monkeypatch.setattr(interpreter_module.settings, "llm_provider", "gemini")
    monkeypatch.setattr(interpreter_module.settings, "llm_api_key", None)
    monkeypatch.setattr(interpreter_module.settings, "llm_model", "gemini-2.5-flash")
    # A configured backup must not turn these isolated primary-provider tests live.
    for name in ("openrouter_api_key", "xai_api_key"):
        monkeypatch.setattr(interpreter_module.settings, name, None, raising=False)
    generate = Mock(return_value=SimpleNamespace(text=json.dumps(VALID_PLAN), parsed=None))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    constructor = Mock(return_value=client)
    monkeypatch.setattr(interpreter_module.genai, "Client", constructor)
    return SimpleNamespace(generate=generate, constructor=constructor, client=client)


@pytest.fixture
def loader():
    value = object.__new__(DatasetLoader)
    value.sales = pd.DataFrame({
        "region": ["EMEA", "APAC", "EMEA"],
        "revenue": [17.0, 31.0, 43.0],
    })
    value.targets = pd.DataFrame()
    value.dictionary = {"metrics": {"revenue": "revenue"}, "dimensions": ["region"]}
    value.queries = []
    return value


@pytest.fixture
def api(monkeypatch, provider, loader):
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    with TestClient(main_module.app) as client:
        yield client


def send_query(api):
    response = api.post("/query", json={"query": QUERY})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_supported"] is True
    QueryPlan.model_validate(body["query_plan"])
    return body


def assert_safe_fallback(body, expected_reason, caplog):
    assert body["interpreter_source"] == "deterministic_fallback"
    assert body["fallback_used"] is True
    assert expected_reason in body["fallback_reason"].lower()
    assert body["provider_model"] is None
    assert body["provider_attempts"][0]["model"] == "gemini-2.5-flash"
    assert TEST_CREDENTIAL not in json.dumps(body)
    assert TEST_CREDENTIAL not in caplog.text
    assert "Traceback" not in json.dumps(body)
    assert body["result"] == [
        {"region": "APAC", "revenue": 31.0},
        {"region": "EMEA", "revenue": 60.0},
    ]


def test_sdk_success_preserves_provenance_and_pandas_calculates(api, provider, loader):
    first = send_query(api)
    assert first["interpreter_source"] == "gemini"
    assert first["fallback_used"] is False
    assert first.get("fallback_reason") is None
    assert first["provider_http_status"] == 200
    assert first["query_plan_validation"] == "PASS"
    assert first["result"] == [
        {"region": "APAC", "revenue": 31.0},
        {"region": "EMEA", "revenue": 60.0},
    ]
    # The same provider plan must give new values when only dataset values change.
    loader.sales["revenue"] *= 7
    second = send_query(api)
    assert second["interpreter_source"] == "gemini"
    assert second["result"] == [
        {"region": "APAC", "revenue": 217.0},
        {"region": "EMEA", "revenue": 420.0},
    ]
    assert provider.generate.call_count == 2
    assert provider.constructor.call_args.kwargs["api_key"] == TEST_CREDENTIAL


def test_sdk_request_supplies_schema_and_separate_system_instruction(api, provider):
    send_query(api)
    arguments = provider.generate.call_args.kwargs
    assert arguments["model"] == "gemini-2.5-flash"
    configuration = arguments["config"]
    if hasattr(configuration, "model_dump"):
        configuration = configuration.model_dump(exclude_none=True)
    assert configuration["response_mime_type"] == "application/json"
    assert configuration["response_json_schema"] == QueryPlan.model_json_schema()
    prompt = configuration["system_instruction"]
    assert "ONLY one JSON object" in prompt
    assert "final numerical results" in prompt
    assert json.loads(prompt.split("The exact required QueryPlan JSON Schema follows:\n", 1)[1]) == QueryPlan.model_json_schema()
    assert json.loads(arguments["contents"])["query"] == QUERY
    # Verify the dictionary is accepted by the installed SDK, not just our mock.
    sdk_types.GenerateContentConfig.model_validate(configuration)


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-custom-preview"])
def test_configured_model_is_not_silently_replaced(provider, loader, monkeypatch, model):
    monkeypatch.setattr(interpreter_module.settings, "llm_model", model)
    QueryInterpreter(loader).interpret(QUERY)
    assert provider.generate.call_args.kwargs["model"] == model
    assert config_module._normalize_model_name(model, "gemini") == model


@pytest.mark.parametrize("text", ["not JSON", "", "```json\n" + json.dumps(VALID_PLAN) + "\n```"])
def test_invalid_or_markdown_json_uses_safe_fallback(api, provider, caplog, text):
    provider.generate.return_value = SimpleNamespace(text=text, parsed=None)
    body = send_query(api)
    assert_safe_fallback(body, "json", caplog)
    assert body["provider_http_status"] == 200
    assert body["query_plan_validation"] == "FAIL"


@pytest.mark.parametrize("payload", [
    {"plan": VALID_PLAN},
    {"plan": [VALID_PLAN]},
    [VALID_PLAN],
    None,
    {"metric": "revenue"},
    {**VALID_PLAN, "operation": "execute_python"},
    {**VALID_PLAN, "limit": 0},
    {**VALID_PLAN, "filters": [{"column": "revenue", "operator": "~~", "value": 2}]},
    {**VALID_PLAN, "answer": 987654321},
])
def test_json_that_does_not_match_query_plan_is_rejected(api, provider, caplog, payload):
    provider.generate.return_value = SimpleNamespace(text=json.dumps(payload), parsed=None)
    body = send_query(api)
    assert_safe_fallback(body, "validation", caplog)
    assert body["provider_http_status"] == 200
    assert body["query_plan_validation"] == "FAIL"


@pytest.mark.parametrize("status,reason", [
    (401, "authentication"), (403, "authentication"),
    (429, "quota"), (500, "temporarily"), (502, "temporarily"), (503, "temporarily"),
])
def test_sdk_http_failures_are_safely_categorized(api, provider, caplog, status, reason):
    exception_type = sdk_errors.ClientError if status < 500 else sdk_errors.ServerError
    provider.generate.side_effect = exception_type(status, {
        "error": {"message": f"Unsafe diagnostic credential={TEST_CREDENTIAL}", "status": "PROVIDER_FAILURE"},
    })
    body = send_query(api)
    assert_safe_fallback(body, reason, caplog)
    assert body["provider_http_status"] == status
    assert body["query_plan_validation"] == "NOT_ATTEMPTED"
    provider.generate.assert_called_once()


@pytest.mark.parametrize("exception,reason", [
    (TimeoutError(f"Unsafe timeout {TEST_CREDENTIAL}"), "temporarily"),
    (httpx.ReadTimeout(f"Unsafe timeout {TEST_CREDENTIAL}"), "temporarily"),
    (ConnectionError(f"Unsafe network diagnostic {TEST_CREDENTIAL}"), "connection"),
    (httpx.ConnectError(f"Unsafe network diagnostic {TEST_CREDENTIAL}"), "connection"),
])
def test_timeout_and_network_failures_are_safe(api, provider, caplog, exception, reason):
    provider.generate.side_effect = exception
    assert_safe_fallback(send_query(api), reason, caplog)
    provider.generate.assert_called_once()


def test_missing_key_does_not_create_client_or_call_provider(api, provider, monkeypatch, caplog):
    monkeypatch.delenv("GEMINI_API_KEY")
    body = send_query(api)
    assert_safe_fallback(body, "key", caplog)
    assert body["provider_http_status"] is None
    assert body["query_plan_validation"] == "NOT_ATTEMPTED"
    provider.constructor.assert_not_called()
    provider.generate.assert_not_called()


def test_missing_sdk_has_safe_explanation(api, provider, monkeypatch, caplog):
    monkeypatch.setattr(interpreter_module, "genai", None)
    assert_safe_fallback(send_query(api), "sdk", caplog)
    provider.generate.assert_not_called()


def test_client_initialization_failure_does_not_expose_exception(api, provider, caplog):
    provider.constructor.side_effect = RuntimeError(f"Unsafe initialization {TEST_CREDENTIAL}")
    assert_safe_fallback(send_query(api), "initialization", caplog)
    provider.generate.assert_not_called()


def test_unexpected_sdk_failure_is_sanitized(api, provider, caplog):
    provider.generate.side_effect = RuntimeError(f"Unsafe provider diagnostic {TEST_CREDENTIAL}")
    body = send_query(api)
    assert body["interpreter_source"] == "deterministic_fallback"
    assert body["fallback_used"] is True
    assert body["fallback_reason"]
    assert TEST_CREDENTIAL not in json.dumps(body)
    assert TEST_CREDENTIAL not in caplog.text
    assert "Traceback" not in json.dumps(body)


@pytest.mark.parametrize("credentials,selected", [
    ({"LLM_API_KEY": "test-generic-placeholder", "GEMINI_API_KEY": "test-gemini-placeholder", "GOOGLE_API_KEY": "test-google-placeholder"}, "test-generic-placeholder"),
    ({"GEMINI_API_KEY": "test-gemini-placeholder", "GOOGLE_API_KEY": "test-google-placeholder"}, "test-gemini-placeholder"),
    ({"GOOGLE_API_KEY": "test-google-placeholder"}, "test-google-placeholder"),
])
def test_runtime_credential_precedence_is_consistent(provider, loader, monkeypatch, credentials, selected):
    for variable in ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in credentials.items():
        monkeypatch.setenv(variable, value)
    QueryInterpreter(loader)
    assert provider.constructor.call_args.kwargs["api_key"] == selected

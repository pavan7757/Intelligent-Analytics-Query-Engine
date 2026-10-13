"""Mock every provider boundary while exercising the real API and Pandas executor."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as sdk_errors

from app.data_loader import DatasetLoader
from app.models import QueryPlan
from app.query_interpreter import QueryInterpreter


interpreter_module = importlib.import_module("app.query_interpreter")
main_module = importlib.import_module("app.main")
QUERY = "Show total revenue by region"
KEYS = {
    "gemini": "test-only-gemini-placeholder",
    "openrouter": "test-only-openrouter-placeholder",
    "grok": "test-only-xai-placeholder",
}
MODELS = {
    "gemini": "gemini-configured-test-model",
    "openrouter": "configured/router-test-model",
    "grok": "grok-configured-test-model",
}
VALID_PLAN = {
    "operation": "aggregation", "metric": "revenue", "aggregation": "sum",
    "group_by": ["region"], "interpretation_confidence": 0.96,
}
PROVIDERS = ("gemini", "openrouter", "grok")
BACKUPS = ("openrouter", "grok")


def http_error_type(status):
    if status in {401, 403}:
        return "authentication_error"
    if status == 429:
        return "quota_rate_limit"
    return "provider_unavailable"


def sdk_failure(status):
    exception_class = sdk_errors.ClientError if status < 500 else sdk_errors.ServerError
    return exception_class(status, {"error": {
        "message": "Unsafe provider diagnostic " + " ".join(KEYS.values()),
        "status": "PROVIDER_FAILURE",
    }})


def chat_response(provider, content=None, status=200, envelope=None):
    endpoint = (
        "https://openrouter.ai/api/v1/chat/completions" if provider == "openrouter"
        else "https://api.x.ai/v1/chat/completions"
    )
    if envelope is None:
        envelope = {"choices": [{"message": {
            "content": json.dumps(VALID_PLAN) if content is None else content,
        }}]}
    return httpx.Response(status, json=envelope, request=httpx.Request("POST", endpoint))


@pytest.fixture
def loader():
    value = object.__new__(DatasetLoader)
    value.sales = pd.DataFrame({
        "region": ["EMEA", "APAC", "EMEA"], "revenue": [17.0, 31.0, 43.0],
    })
    value.targets = pd.DataFrame()
    value.dictionary = {
        "metrics": {"revenue": "revenue"}, "dimensions": ["region"],
        "terms": {"income": "revenue", "geographical areas": "region"},
    }
    value.queries = []
    return value


@pytest.fixture
def chain(monkeypatch):
    for name in ("GROQ_API_KEY", "GROQ_MODEL", "OPENROUTER_MODEL", "XAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(interpreter_module.settings, "groq_api_key", None)
    for name in ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", KEYS["gemini"])
    monkeypatch.setenv("OPENROUTER_API_KEY", KEYS["openrouter"])
    monkeypatch.setenv("XAI_API_KEY", KEYS["grok"])
    monkeypatch.setattr(interpreter_module.settings, "llm_provider", "gemini")
    monkeypatch.setattr(interpreter_module.settings, "llm_api_key", None)
    monkeypatch.setattr(interpreter_module.settings, "llm_model", MODELS["gemini"])
    for provider, prefix in (("openrouter", "openrouter"), ("grok", "xai")):
        monkeypatch.setattr(interpreter_module.settings, prefix + "_api_key", KEYS[provider], raising=False)
        monkeypatch.setattr(interpreter_module.settings, prefix + "_model", MODELS[provider], raising=False)
    gemini = Mock(return_value=SimpleNamespace(text=json.dumps(VALID_PLAN), parsed=None))
    constructor = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=gemini)))
    monkeypatch.setattr(interpreter_module.genai, "Client", constructor)
    openrouter = Mock(return_value=chat_response("openrouter"))
    grok = Mock(return_value=chat_response("grok"))

    def route(url, **kwargs):
        if url == "https://openrouter.ai/api/v1/chat/completions":
            return openrouter(url, **kwargs)
        if url == "https://api.x.ai/v1/chat/completions":
            return grok(url, **kwargs)
        raise AssertionError("Unexpected external provider URL")

    post = Mock(side_effect=route)
    monkeypatch.setattr(interpreter_module.httpx, "post", post)
    return SimpleNamespace(
        gemini=gemini, openrouter=openrouter, grok=grok,
        constructor=constructor, post=post,
    )


@pytest.fixture
def api(monkeypatch, chain, loader):
    monkeypatch.setattr(main_module, "get_loader", lambda: loader)
    monkeypatch.setattr(main_module.FeedbackStore, "get_relevant", lambda self, query: [])
    with TestClient(main_module.app) as client:
        yield client


def submit(api):
    response = api.post("/query", json={"query": QUERY})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_supported"] is True
    QueryPlan.model_validate(body["query_plan"])
    assert body["result"] == [
        {"region": "APAC", "revenue": 31.0}, {"region": "EMEA", "revenue": 60.0},
    ]
    return body


def prepare_provider(chain, provider):
    if provider in BACKUPS:
        chain.gemini.side_effect = sdk_failure(503)
    if provider == "grok":
        chain.openrouter.return_value = chat_response("openrouter", status=503)


def set_text(chain, provider, text):
    if provider == "gemini":
        chain.gemini.return_value = SimpleNamespace(text=text, parsed=None)
    else:
        getattr(chain, provider).return_value = chat_response(provider, content=text)


def assert_safe(body, caplog):
    serialized = json.dumps(body)
    for key in KEYS.values():
        assert key not in serialized and key not in caplog.text
    assert "Authorization" not in serialized
    assert "Traceback" not in serialized
    assert "Unsafe provider diagnostic" not in serialized + caplog.text


def assert_success(body, source):
    assert body["interpreter_source"] == source
    assert body["fallback_used"] is (source != "gemini")
    assert body["provider_http_status"] == 200
    assert body["provider_model"] == MODELS[source]
    assert body["query_plan_validation"] == "PASS"
    attempts = body["provider_attempts"]
    assert [item["provider"] for item in attempts] == list(PROVIDERS[:PROVIDERS.index(source) + 1])
    assert attempts[-1]["status"] == "success"
    assert attempts[-1]["http_status"] == 200
    assert attempts[-1]["query_plan_validation"] == "PASS"
    if source == "gemini":
        assert body["fallback_reason"] is None
    else:
        assert body["fallback_reason"]
        assert "gemini" in body["fallback_reason"].lower()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_provider_success_short_circuits_chain_and_pandas_calculates(api, chain, loader, provider):
    prepare_provider(chain, provider)
    body = submit(api)
    assert_success(body, provider)
    chain.gemini.assert_called_once()
    if provider == "gemini":
        chain.openrouter.assert_not_called()
    else:
        chain.openrouter.assert_called_once()
    if provider == "grok":
        chain.grok.assert_called_once()
    else:
        chain.grok.assert_not_called()
    assert "Pandas" in body["explanation"]
    assert ("Gemini" if provider == "gemini" else "OpenRouter" if provider == "openrouter" else "Grok") in body["explanation"]
    # Reusing the identical provider plan on changed data must change numerical results.
    loader.sales["revenue"] *= 7
    second = api.post("/query", json={"query": QUERY}).json()
    assert second["interpreter_source"] == provider
    assert second["result"] == [
        {"region": "APAC", "revenue": 217.0}, {"region": "EMEA", "revenue": 420.0},
    ]


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502, 503, 504])
def test_gemini_http_failure_tries_openrouter_once(api, chain, caplog, status):
    chain.gemini.side_effect = sdk_failure(status)
    body = submit(api)
    assert_success(body, "openrouter")
    attempt = body["provider_attempts"][0]
    assert attempt["status"] == "failed" and attempt["http_status"] == status
    assert attempt["query_plan_validation"] == "NOT_ATTEMPTED"
    assert attempt["error_type"] == http_error_type(status)
    assert attempt["reason"]
    chain.gemini.assert_called_once()
    chain.openrouter.assert_called_once()
    chain.grok.assert_not_called()
    assert_safe(body, caplog)


@pytest.mark.parametrize("provider", BACKUPS)
@pytest.mark.parametrize("status", [401, 403, 429, 500, 502, 503, 504])
def test_secondary_and_tertiary_http_errors_advance_safely(api, chain, caplog, provider, status):
    prepare_provider(chain, provider)
    getattr(chain, provider).return_value = chat_response(provider, status=status, envelope={
        "error": {"message": "Unsafe provider diagnostic " + " ".join(KEYS.values())},
    })
    body = submit(api)
    expected_source = "grok" if provider == "openrouter" else "deterministic_fallback"
    assert body["interpreter_source"] == expected_source
    assert body["fallback_used"] is True
    attempt = next(item for item in body["provider_attempts"] if item["provider"] == provider)
    assert attempt["status"] == "failed" and attempt["http_status"] == status
    assert attempt["error_type"] == http_error_type(status)
    assert attempt["reason"]
    getattr(chain, provider).assert_called_once()
    assert_safe(body, caplog)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("exception", [
    TimeoutError("Unsafe provider diagnostic " + KEYS["gemini"]),
    httpx.ReadTimeout("Unsafe provider diagnostic " + KEYS["openrouter"]),
    ConnectionError("Unsafe provider diagnostic " + KEYS["grok"]),
    httpx.ConnectError("Unsafe provider diagnostic " + KEYS["grok"]),
])
def test_timeout_and_network_failures_advance_without_secret_leaks(api, chain, caplog, provider, exception):
    prepare_provider(chain, provider)
    getattr(chain, provider).side_effect = exception
    body = submit(api)
    expected_source = {"gemini": "openrouter", "openrouter": "grok", "grok": "deterministic_fallback"}[provider]
    assert body["interpreter_source"] == expected_source
    attempt = next(item for item in body["provider_attempts"] if item["provider"] == provider)
    assert attempt["status"] == "failed" and attempt["http_status"] is None
    assert attempt["query_plan_validation"] == "NOT_ATTEMPTED"
    expected_error = (
        "provider_timeout" if isinstance(exception, (TimeoutError, httpx.TimeoutException))
        else "connection_error"
    )
    assert attempt["error_type"] == expected_error
    assert attempt["reason"]
    getattr(chain, provider).assert_called_once()
    assert_safe(body, caplog)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("text", ["not JSON", "", "```json\n" + json.dumps(VALID_PLAN) + "\n```"])
def test_each_provider_invalid_json_advances_chain(api, chain, caplog, provider, text):
    prepare_provider(chain, provider)
    set_text(chain, provider, text)
    body = submit(api)
    assert body["interpreter_source"] == {"gemini": "openrouter", "openrouter": "grok", "grok": "deterministic_fallback"}[provider]
    attempt = next(item for item in body["provider_attempts"] if item["provider"] == provider)
    assert attempt["status"] == "failed" and attempt["http_status"] == 200
    assert attempt["query_plan_validation"] == "FAIL"
    expected_error = "response_validation_error" if provider == "gemini" and not text.strip() else "invalid_json"
    assert attempt["error_type"] == expected_error
    assert_safe(body, caplog)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("payload", [
    {"plan": VALID_PLAN}, {"plan": [VALID_PLAN]}, [VALID_PLAN], None,
    {"metric": "revenue"}, {**VALID_PLAN, "operation": "execute_python"},
    {**VALID_PLAN, "limit": 0}, {**VALID_PLAN, "answer": 987654321},
    {**VALID_PLAN, "filters": [{"column": "revenue", "operator": "~~", "value": 2}]},
])
def test_each_provider_invalid_plan_or_final_answer_is_never_trusted(api, chain, caplog, provider, payload):
    prepare_provider(chain, provider)
    set_text(chain, provider, json.dumps(payload))
    body = submit(api)
    assert body["interpreter_source"] == {"gemini": "openrouter", "openrouter": "grok", "grok": "deterministic_fallback"}[provider]
    attempt = next(item for item in body["provider_attempts"] if item["provider"] == provider)
    assert attempt["status"] == "failed" and attempt["http_status"] == 200
    assert attempt["query_plan_validation"] == "FAIL"
    assert attempt["error_type"] == "response_validation_error"
    assert_safe(body, caplog)


@pytest.mark.parametrize("provider", BACKUPS)
@pytest.mark.parametrize("envelope", [
    {}, {"choices": []}, {"choices": [{"message": {}}]},
    {"choices": [{"message": {"content": None}}]},
    {"choices": [{"message": {"content": VALID_PLAN}}]},
    {"error": {"message": "Unsafe provider diagnostic " + KEYS["grok"]}},
])
def test_chat_provider_malformed_envelope_is_rejected(api, chain, caplog, provider, envelope):
    prepare_provider(chain, provider)
    getattr(chain, provider).return_value = chat_response(provider, envelope=envelope)
    body = submit(api)
    assert body["interpreter_source"] == ("grok" if provider == "openrouter" else "deterministic_fallback")
    attempt = next(item for item in body["provider_attempts"] if item["provider"] == provider)
    assert attempt["status"] == "failed" and attempt["query_plan_validation"] == "FAIL"
    assert_safe(body, caplog)


@pytest.mark.parametrize("provider", BACKUPS)
@pytest.mark.parametrize("missing", ["key", "model"])
def test_missing_backup_configuration_is_skipped_without_network_call(api, chain, monkeypatch, provider, missing):
    prepare_provider(chain, provider)
    prefix = "openrouter" if provider == "openrouter" else "xai"
    setting = prefix + ("_api_key" if missing == "key" else "_model")
    monkeypatch.setattr(interpreter_module.settings, setting, None if missing == "key" else "")
    monkeypatch.delenv(prefix.upper() + ("_API_KEY" if missing == "key" else "_MODEL"), raising=False)
    body = submit(api)
    assert body["interpreter_source"] == ("grok" if provider == "openrouter" else "deterministic_fallback")
    attempt = next(item for item in body["provider_attempts"] if item["provider"] == provider)
    assert attempt["status"] == "skipped" and attempt["http_status"] is None
    assert missing in attempt["reason"].lower()
    getattr(chain, provider).assert_not_called()


def test_missing_gemini_key_safely_uses_secondary(api, chain, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    body = submit(api)
    assert_success(body, "openrouter")
    assert body["provider_attempts"][0]["status"] == "skipped"
    chain.constructor.assert_not_called()
    chain.gemini.assert_not_called()
    chain.grok.assert_not_called()


def test_all_provider_failures_preserve_existing_deterministic_parser(api, chain, caplog):
    chain.gemini.side_effect = sdk_failure(503)
    chain.openrouter.return_value = chat_response("openrouter", status=429)
    chain.grok.return_value = chat_response("grok", status=403)
    body = submit(api)
    assert body["interpreter_source"] == "deterministic_fallback"
    assert body["fallback_used"] is True
    assert body["provider_model"] is None
    assert [item["provider"] for item in body["provider_attempts"]] == list(PROVIDERS)
    assert [item["http_status"] for item in body["provider_attempts"]] == [503, 429, 403]
    assert all(item["status"] == "failed" for item in body["provider_attempts"])
    assert "deterministic" in body["explanation"].lower()
    assert body["fallback_reason"]
    chain.gemini.assert_called_once()
    chain.openrouter.assert_called_once()
    chain.grok.assert_called_once()
    assert_safe(body, caplog)


@pytest.mark.parametrize("provider", BACKUPS)
def test_backup_receives_same_schema_dictionary_feedback_and_no_calculation_request(chain, loader, provider):
    prepare_provider(chain, provider)
    feedback = [{
        "query": "Revenue by region", "incorrect_interpretation": "Ungrouped revenue",
        "correct_interpretation": "Group revenue by region", "notes": "business context",
    }]
    interpreter = QueryInterpreter(loader)
    plan = interpreter.interpret(QUERY, feedback)
    assert plan.is_supported and interpreter.interpreter_source == provider
    gemini_context = json.loads(chain.gemini.call_args.kwargs["contents"])
    request = getattr(chain, provider).call_args.kwargs
    messages = request["json"]["messages"]
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
    assert messages[0]["content"] == interpreter._system_prompt()
    assert json.loads(messages[1]["content"]) == gemini_context
    assert gemini_context["query"] == QUERY
    assert gemini_context["dictionary"] == loader.dictionary
    assert gemini_context["feedback_examples"] == [
        feedback[0]["query"] + " -> " + feedback[0]["correct_interpretation"],
    ]
    schema_text = messages[0]["content"].split("The exact required QueryPlan JSON Schema follows:\n", 1)[1]
    assert json.loads(schema_text) == QueryPlan.model_json_schema()
    assert "ONLY one JSON object" in messages[0]["content"]
    assert "final numerical results" in messages[0]["content"]
    assert request["json"]["model"] == MODELS[provider]
    assert request["json"]["response_format"] == {"type": "json_object"}
    assert request["headers"]["Authorization"] == "Bearer " + KEYS[provider]
    assert 0 < request["timeout"] <= 60


@pytest.mark.parametrize("provider", PROVIDERS)
def test_explicit_provider_rejection_never_executes_or_calls_later_provider(api, chain, monkeypatch, provider):
    prepare_provider(chain, provider)
    set_text(chain, provider, json.dumps({
        **VALID_PLAN, "is_supported": False, "interpretation_confidence": 0,
        "semantic_errors": ["The requested calculation cannot be represented safely."],
    }))

    def forbidden_execute(self, plan):
        raise AssertionError("An explicitly unsupported QueryPlan reached Pandas")

    monkeypatch.setattr(main_module.QueryExecutor, "execute", forbidden_execute)
    response = api.post("/query", json={"query": QUERY})
    assert response.status_code == 200
    body = response.json()
    assert body["interpreter_source"] == provider
    assert body["is_supported"] is False and body["confidence_score"] == 0
    assert body["result"] == [] and body["generated_logic"] is None
    assert "No calculations were performed" in body["explanation"]
    if provider == "gemini":
        chain.openrouter.assert_not_called()
    if provider != "grok":
        chain.grok.assert_not_called()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_provider_plan_passes_pydantic_before_executor(api, chain, monkeypatch, provider):
    prepare_provider(chain, provider)
    events = []
    original_validation = QueryPlan.model_validate
    original_execution = main_module.QueryExecutor.execute

    def validate(payload, *args, **kwargs):
        events.append(("pydantic", payload))
        return original_validation(payload, *args, **kwargs)

    def execute(self, plan):
        events.append(("pandas", plan))
        assert isinstance(plan, QueryPlan)
        return original_execution(self, plan)

    monkeypatch.setattr(QueryPlan, "model_validate", validate)
    monkeypatch.setattr(main_module.QueryExecutor, "execute", execute)
    response = api.post("/query", json={"query": QUERY})
    assert response.status_code == 200
    assert response.json()["interpreter_source"] == provider
    provider_validations = [i for i, event in enumerate(events) if event == ("pydantic", VALID_PLAN)]
    pandas_calls = [i for i, (kind, _) in enumerate(events) if kind == "pandas"]
    assert provider_validations and len(pandas_calls) == 1
    assert provider_validations[0] < pandas_calls[0]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_unexpected_provider_exception_is_safe_and_advances_chain(api, chain, caplog, provider):
    prepare_provider(chain, provider)
    getattr(chain, provider).side_effect = RuntimeError(
        "Unsafe provider diagnostic " + " ".join(KEYS.values())
    )
    body = submit(api)
    assert body["interpreter_source"] == {
        "gemini": "openrouter", "openrouter": "grok", "grok": "deterministic_fallback",
    }[provider]
    assert_safe(body, caplog)


@pytest.mark.parametrize("provider", BACKUPS)
def test_chat_provider_non_json_http_body_is_safe(api, chain, caplog, provider):
    prepare_provider(chain, provider)
    request = chat_response(provider).request
    getattr(chain, provider).return_value = httpx.Response(
        200, text="Unsafe provider diagnostic " + " ".join(KEYS.values()), request=request,
    )
    body = submit(api)
    attempt = next(item for item in body["provider_attempts"] if item["provider"] == provider)
    assert attempt["status"] == "failed" and attempt["http_status"] == 200
    assert attempt["query_plan_validation"] == "FAIL"
    assert_safe(body, caplog)


def test_missing_gemini_sdk_still_attempts_openrouter(api, chain, monkeypatch):
    monkeypatch.setattr(interpreter_module, "genai", None)
    body = submit(api)
    assert_success(body, "openrouter")
    assert body["provider_attempts"][0]["status"] == "skipped"
    chain.gemini.assert_not_called()
    chain.grok.assert_not_called()


def test_gemini_client_initialization_failure_still_attempts_openrouter(api, chain, caplog):
    chain.constructor.side_effect = RuntimeError("Unsafe provider diagnostic " + KEYS["gemini"])
    body = submit(api)
    assert_success(body, "openrouter")
    chain.gemini.assert_not_called()
    chain.grok.assert_not_called()
    assert_safe(body, caplog)


def test_groq_backup_uses_its_own_endpoint_and_reports_provenance(api, chain, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only-groq-placeholder")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    chain.gemini.side_effect = sdk_failure(503)
    original_route = chain.post.side_effect

    def route(url, **kwargs):
        if url == "https://api.groq.com/openai/v1/chat/completions":
            assert kwargs["headers"]["Authorization"] == "Bearer test-only-groq-placeholder"
            assert kwargs["json"]["model"] == "openai/gpt-oss-120b"
            return chat_response("grok")
        return original_route(url, **kwargs)

    chain.openrouter.return_value = chat_response("openrouter", status=503)
    chain.post.side_effect = route
    body = submit(api)
    assert body["interpreter_source"] == "groq"
    assert body["fallback_used"] is True
    assert body["provider_model"] == "openai/gpt-oss-120b"
    assert "Groq GenAI" in body["explanation"]
    assert body["provider_attempts"][-1]["status"] == "success"
    chain.grok.assert_not_called()


def test_gemini_retries_only_transient_server_errors(chain, loader):
    QueryInterpreter(loader)
    retry = chain.constructor.call_args.kwargs["http_options"]["retry_options"]
    assert retry["attempts"] == 3
    assert retry["http_status_codes"] == [502, 503, 504]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_structurally_valid_plan_echoing_credentials_is_rejected(api, chain, caplog, provider):
    prepare_provider(chain, provider)
    set_text(chain, provider, json.dumps({
        **VALID_PLAN, "ambiguity": ["Unsafe provider diagnostic " + " ".join(KEYS.values())],
    }))
    body = submit(api)
    assert body["interpreter_source"] == {
        "gemini": "openrouter", "openrouter": "grok", "grok": "deterministic_fallback",
    }[provider]
    attempt = next(item for item in body["provider_attempts"] if item["provider"] == provider)
    assert attempt["status"] == "failed" and attempt["query_plan_validation"] == "FAIL"
    assert_safe(body, caplog)

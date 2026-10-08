import json
import logging
import os
import re
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from .config import settings

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency depending on installation
    genai = None

from .data_loader import DatasetLoader
from .models import ComparisonPlan, FilterExpression, FilterPlan, QueryPlan, TimeFilterPlan
from .boolean_filters import expression_predicates
from .filter_parser import RequestedFilterParser, FilterSyntaxError, combine_filters, filter_signature
from .schema_manager import SchemaManager


logger = logging.getLogger(__name__)


class QueryInterpretationError(ValueError):
    pass


class ProviderResponseError(ValueError):
    """An unusable provider response; its body is never logged or exposed."""


class QueryInterpreter:
    _ORDERING_DESCRIPTION = r"(?:\s+(?:contributors?|contributions?))?"
    _DESCENDING_RANGE = (
        r"\b(?:the\s+)?(?:highest|largest|biggest|greatest|most)"
        + _ORDERING_DESCRIPTION
        + r"\s+(?:down\s+)?to\s+(?:the\s+)?(?:lowest|smallest|least)"
        + _ORDERING_DESCRIPTION + r"\b"
    )
    _ASCENDING_RANGE = (
        r"\b(?:the\s+)?(?:lowest|smallest|least)"
        + _ORDERING_DESCRIPTION
        + r"\s+(?:up\s+)?to\s+(?:the\s+)?(?:highest|largest|biggest|greatest|most)"
        + _ORDERING_DESCRIPTION + r"\b"
    )

    def __init__(self, loader: DatasetLoader) -> None:
        self.loader = loader
        self.schema = SchemaManager(loader)
        self.interpreter_source = "deterministic_fallback"
        self.fallback_reason: str | None = None
        self.provider_http_status: int | None = None
        self.query_plan_validation = "NOT_ATTEMPTED"
        self.provider_error_type: str | None = None
        self.provider_model: str | None = None
        self.provider_attempts: list[dict[str, Any]] = []
        self.fallback_used = True
        self.client = self._build_client()
        self._gemini_unavailable_reason = self.fallback_reason
        self._gemini_unavailable_type = self.provider_error_type

    def _provider_failure(self, error_type: str, reason: str) -> None:
        self.provider_error_type = error_type
        self.fallback_reason = reason
        logger.warning("Provider interpretation unavailable: %s", reason)

    def _record_provider_error(self, exc: Exception, provider: str = "gemini") -> None:
        """Classify errors without formatting provider bodies, URLs or secrets."""
        label = {"gemini": "Gemini", "openrouter": "OpenRouter", "grok": "Grok"}[provider]
        code = getattr(exc, "code", None)
        if not isinstance(code, int):
            code = getattr(exc, "status_code", None)
        if not isinstance(code, int):
            response = getattr(exc, "response", None)
            code = getattr(response, "status_code", None)
        if isinstance(code, int) and 100 <= code <= 599:
            self.provider_http_status = code
        else:
            code = None
        if code in {401, 403}:
            self._provider_failure("authentication_error", f"{label} authentication/API-key error (HTTP {code}).")
        elif code == 429:
            self._provider_failure("quota_rate_limit", f"{label} quota or rate limit exceeded (HTTP 429).")
        elif code is not None and code >= 500:
            self._provider_failure("provider_unavailable", f"{label} is temporarily unavailable (HTTP {code}).")
        elif isinstance(exc, (TimeoutError, httpx.TimeoutException)) or code == 408:
            self._provider_failure("provider_timeout", f"{label} timed out and is temporarily unavailable.")
        elif isinstance(exc, (ConnectionError, httpx.RequestError)):
            self._provider_failure("connection_error", f"{label} connection failed. Check network connectivity.")
        elif isinstance(exc, json.JSONDecodeError):
            self.query_plan_validation = "FAIL"
            self._provider_failure("invalid_json", f"{label} returned invalid JSON; QueryPlan validation could not proceed.")
        elif isinstance(exc, (ValidationError, ProviderResponseError)):
            self.query_plan_validation = "FAIL"
            self._provider_failure("response_validation_error", f"{label} response validation failed: a single valid QueryPlan object is required.")
        elif code == 404:
            self._provider_failure("configuration_error", f"The configured {label} model or endpoint was not found (HTTP 404).")
        elif code is not None:
            self._provider_failure("provider_error", f"{label} rejected the request (HTTP {code}). Check provider configuration.")
        else:
            self._provider_failure("provider_error", f"{label} request failed; the next configured provider will be tried.")

    def _build_client(self):
        provider = (os.getenv("LLM_PROVIDER") or settings.llm_provider or "gemini").strip().lower()
        if provider not in {"gemini", "google", "google_gemini"}:
            self._provider_failure("configuration_error", "The provider chain requires LLM_PROVIDER=gemini for its primary provider.")
            return None
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or settings.llm_api_key
        )

        if not api_key:
            self._provider_failure("missing_api_key", "Gemini API key is missing from the application configuration.")
            return None
        if genai is None:
            self._provider_failure("sdk_unavailable", "Gemini SDK is unavailable. Install the backend requirements.")
            return None
        try:
            return genai.Client(api_key=api_key, http_options={
                "timeout": 30000,
                "retry_options": {"attempts": 1},
            })
        except Exception:
            self._provider_failure("configuration_error", "Gemini client initialization failed. Check provider configuration.")
            return None

    def _call_llm(self, context: str) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("No provider client available.")

        if genai is None:
            raise RuntimeError("Gemini SDK unavailable")
        response = self.client.models.generate_content(
            model=settings.llm_model,
            contents=context,
            config={
                "system_instruction": self._system_prompt(),
                "response_mime_type": "application/json",
                "response_json_schema": QueryPlan.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            },
        )
        self.provider_http_status = 200
        text = getattr(response, "text", None)
        if text is not None:
            return json.loads(text)
        payload = getattr(response, "parsed", None)
        if isinstance(payload, BaseModel):
            return payload.model_dump()
        if isinstance(payload, dict):
            return payload
        raise ProviderResponseError("Empty provider response.")

    @staticmethod
    def _backup_configuration(provider: str) -> tuple[str | None, str]:
        if provider == "openrouter":
            return (
                os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key,
                (os.getenv("OPENROUTER_MODEL") or settings.openrouter_model or "").strip(),
            )
        return (
            os.getenv("XAI_API_KEY") or settings.xai_api_key,
            (os.getenv("XAI_MODEL") or settings.xai_model or "").strip(),
        )

    def _call_chat_provider(self, provider: str, context: str) -> dict[str, Any]:
        api_key, model = self._backup_configuration(provider)
        endpoint = {
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "grok": "https://api.x.ai/v1/chat/completions",
        }[provider]
        response = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": context},
                ],
                # QueryPlan has recursive Boolean filters. JSON mode keeps the exact
                # schema in the prompt without unsupported circular schema refs.
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
            timeout=20,
        )
        self.provider_http_status = response.status_code
        response.raise_for_status()
        envelope = response.json()
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderResponseError("Provider response has no QueryPlan content.") from None
        if not isinstance(content, str):
            raise ProviderResponseError("Provider content must be a JSON string.")
        return json.loads(content)

    def _call_openrouter(self, context: str) -> dict[str, Any]:
        return self._call_chat_provider("openrouter", context)

    def _call_grok(self, context: str) -> dict[str, Any]:
        return self._call_chat_provider("grok", context)

    def _validate_provider_payload(self, payload: Any) -> QueryPlan:
        if not isinstance(payload, dict) or set(payload) - set(QueryPlan.model_fields):
            raise ProviderResponseError("Provider response contains an unsupported structure or fields.")
        credentials = [
            os.getenv("LLM_API_KEY"), os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY"),
            os.getenv("OPENROUTER_API_KEY"), os.getenv("XAI_API_KEY"),
            settings.llm_api_key, settings.openrouter_api_key, settings.xai_api_key,
        ]
        serialized = json.dumps(payload, ensure_ascii=False)
        if any(secret and secret in serialized for secret in credentials):
            raise ProviderResponseError("Provider response contains private configuration values.")
        return QueryPlan.model_validate(payload)

    def interpret(self, query: str, feedback: list[dict[str, Any]] | None = None) -> QueryPlan:
        if not query.strip():
            raise QueryInterpretationError("The query cannot be empty.")
        context = self._build_context(query, feedback)
        self.provider_attempts = []
        self.provider_model = None
        self.provider_http_status = None
        self.query_plan_validation = "NOT_ATTEMPTED"
        self.fallback_reason = None
        self.provider_error_type = None
        reasons = []
        providers = (
            ("gemini", settings.llm_model, self._call_llm),
            ("openrouter", self._backup_configuration("openrouter")[1], self._call_openrouter),
            ("grok", self._backup_configuration("grok")[1], self._call_grok),
        )
        for provider, model, call in providers:
            skip_reason = None
            skip_type = "configuration_error"
            if provider == "gemini" and not self.client:
                skip_reason = self._gemini_unavailable_reason or "No Gemini provider client is available."
                skip_type = self._gemini_unavailable_type or "provider_unavailable"
            elif provider != "gemini":
                api_key, _ = self._backup_configuration(provider)
                label = "OpenRouter" if provider == "openrouter" else "Grok"
                if not api_key:
                    skip_reason = f"{label} API key is missing from the application configuration."
                    skip_type = "missing_api_key"
                elif not model:
                    variable = "OPENROUTER_MODEL" if provider == "openrouter" else "XAI_MODEL"
                    skip_reason = f"{label} model is missing; configure {variable}."
            if skip_reason:
                self.provider_attempts.append({
                    "provider": provider, "model": model or None, "status": "skipped",
                    "http_status": None, "error_type": skip_type,
                    "query_plan_validation": "NOT_ATTEMPTED", "reason": skip_reason,
                })
                reasons.append(skip_reason)
                continue
            self.provider_http_status = None
            self.query_plan_validation = "NOT_ATTEMPTED"
            self.provider_error_type = None
            try:
                plan = self._validate_provider_payload(call(context))
                self.query_plan_validation = "PASS"
                plan = self._validate_requested_constraints(self._normalize_plan(plan, query), query)
                self.provider_attempts.append({
                    "provider": provider, "model": model, "status": "success",
                    "http_status": self.provider_http_status, "error_type": None,
                    "query_plan_validation": "PASS", "reason": None,
                })
                self.interpreter_source = provider
                self.provider_model = model
                self.fallback_used = provider != "gemini"
                self.fallback_reason = "; ".join(reasons) if self.fallback_used else None
                return plan
            except Exception as exc:
                self._record_provider_error(exc, provider)
                self.provider_attempts.append({
                    "provider": provider, "model": model, "status": "failed",
                    "http_status": self.provider_http_status, "error_type": self.provider_error_type,
                    "query_plan_validation": self.query_plan_validation, "reason": self.fallback_reason,
                })
                reasons.append(self.fallback_reason)
        self.interpreter_source = "deterministic_fallback"
        self.fallback_used = True
        self.fallback_reason = "; ".join(reasons)
        return self._validate_requested_constraints(self._deterministic_fallback(query), query)

    def _validate_requested_constraints(self, plan: QueryPlan, query: str) -> QueryPlan:
        """Reject explicit intent that a structurally valid plan failed to preserve."""
        if not plan.is_supported:
            plan.interpretation_confidence = 0.0
            return plan
        requested_filters, filter_errors = self._requested_filter_intent(query)
        actual_filters = self._plan_filter_intent(plan)
        filter_items = self._filter_items(actual_filters)
        errors = self._find_unresolved_terms(query, plan.metric, plan.group_by, filter_items)
        errors.extend(filter_errors)
        errors.extend(self._extract_unresolved_grouping_terms(query, plan.group_by))
        numeric_filters = self._extract_numeric_filters(query)
        errors.extend(self._extract_unresolved_numeric_terms(query, numeric_filters))
        if filter_signature(requested_filters) != filter_signature(actual_filters):
            errors.append("Required filters or their Boolean AND/OR meaning were not preserved.")
        requested_time, time_errors = self._requested_calendar_filter(query)
        errors.extend(time_errors)
        if requested_time and requested_time != plan.time_filter:
            errors.append("The requested calendar/date constraint was not preserved.")
        ordering = self._requested_ordering(query, plan.metric)
        errors.extend(ordering["errors"])
        errors.extend(self._multi_stage_requirements(query))
        requested_metric = self._infer_metric(query)
        ranking = self._infer_ranking_request(query)
        if ranking["is_ranking"]:
            expected_dimension = ranking.get("rank_dimension")
            if expected_dimension and (expected_dimension not in plan.group_by or plan.rank_dimension != expected_dimension):
                errors.append(f"Requested entity/ranking granularity was not preserved: {expected_dimension}")
            if plan.operation not in {"ranking", "top_n_per_group"}:
                errors.append("The requested ranking operation was not preserved.")
            if set(plan.group_by) != set(ranking["group_by"]):
                errors.append("The requested ranking grain was changed by additional or missing dimensions.")
            if plan.sort != ranking["sort"]:
                errors.append("The requested ranking direction was not preserved.")
            if ranking["limit"] is None and (plan.limit is not None or plan.top_n_per_group is not None):
                errors.append("An unrequested ranking limit would remove required results.")
            elif ranking["limit"] is not None and plan.limit != ranking["limit"]:
                errors.append("The requested ranking limit was not preserved.")
        if self._is_contribution_query(query):
            if not plan.contribution_percentage or plan.operation == "filter":
                errors.append("The requested contribution/percentage-of-total calculation was not preserved.")
            contribution_groups = self._infer_group_by(query) or self._infer_contribution_group(query)
            if not contribution_groups:
                errors.append("A contribution dimension could not be resolved.")
            for dimension in contribution_groups:
                if dimension not in plan.group_by:
                    errors.append(f"Required contribution grouping was not preserved: {dimension}")
        for dimension in self._infer_group_by(query):
            if ranking["is_ranking"] and dimension == requested_metric:
                continue
            if dimension not in plan.group_by:
                errors.append(f"Required grouping was not preserved: {dimension}")
        if self._is_filter_only_query(query) and not self._infer_group_by(query) and plan.operation != "filter":
            errors.append("The request requires matching records without aggregation.")
        if re.search(r"\b(?:targets?|achievement)\b", query, re.IGNORECASE) and not plan.requires_targets:
            errors.append("The requested target comparison was not preserved.")
        requested_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
        if requested_dates:
            preserved_dates = [
                str(bound) for bound in (
                    plan.time_filter.start if plan.time_filter else None,
                    plan.time_filter.end if plan.time_filter else None,
                ) if bound
            ]
            preserved_dates.extend(str(item.value) for item in filter_items if item.column in self.schema.date_columns())
            if any(not any(bound.startswith(date) for bound in preserved_dates) for date in requested_dates):
                errors.append("The requested date constraint could not be preserved.")
        if requested_metric and plan.operation != "filter" and requested_metric != plan.metric:
            errors.append(f"Requested metric was not preserved: {requested_metric}")
        if re.search(r"\b(?:average|avg|mean|minimum|min|maximum|max|sum|total|count)\b", query, re.IGNORECASE):
            requested_aggregation = self._infer_aggregation(query, requested_metric)
            if plan.operation != "filter" and plan.aggregation != requested_aggregation:
                errors.append(f"Requested aggregation was not preserved: {requested_aggregation}")
        if errors:
            plan.is_supported = False
            plan.interpretation_confidence = 0.0
            plan.unresolved_terms = list(dict.fromkeys([*plan.unresolved_terms, *errors]))
            plan.semantic_errors = list(dict.fromkeys([*plan.semantic_errors, *errors]))
            plan.ambiguity = list(dict.fromkeys([*plan.ambiguity, *errors]))
        return QueryPlan.model_validate(plan.model_dump())

    def _system_prompt(self) -> str:
        return """You are an analytics query planner. Return ONLY one JSON object matching QueryPlan exactly.
Put all QueryPlan fields, including operation, at the top level. Do not return an array or wrap the object
in plan, configuration, confidence, or any other wrapper. Use interpretation_confidence and ambiguity
as defined in QueryPlan. Do not include Markdown fences, commentary, or fields outside QueryPlan.
Never invent columns or metrics. Use actual schema fields and data dictionary mappings.
Use supplied dictionary term aliases for business dimensions. Ordering verbs such as "order them"
do not request an order entity. Magnitude ranges such as biggest contributor to smallest mean
sorting the aggregate metric descending; contributor alone does not request a percentage.
Aggregation vocabulary is separate from metric identity: average/avg/mean means aggregation="mean",
minimum/min means "min", maximum/max means "max", and total/sum means "sum" on the requested
numeric metric. A request for average revenue does not request a different derived metric.
Articles and conversational filler are grammar, not fields. Resolve pronouns from an entity named in
the same query. "N entities with the highest/lowest total metric" is ranking of summed values, not
a filter or a count. A singular entity with most/least requests one result unless an explicit N overrides it.
Resolve natural categorical descriptions using supplied values and their dimension/entity qualifiers.
A categorical filter qualifier does not replace the entity being ranked or add a grouping dimension.
Use monetary_metric_resolution for conversational money/earnings: choose only a safely resolved
available metric; if multiple monetary metrics remain plausible, request clarification without execution.
Dictionary-defined derived metrics are valid even when absent from the physical columns; put their
dictionary name in metric, not a formula. Use supplied derived grouping dimensions when requested.
For rankings within each group, include the ranked and partition dimensions in group_by, put only
the partition dimensions in partition_by, set rank_dimension, limit, and limit_scope="per_group".
Preserve every requested predicate, date constraint, entity identity, and operation. For Boolean
filters use filter_expression with explicit and/or groups; flat filters are additional AND predicates.
Never turn OR into AND. Bare entity names refer to identity fields, not segment/category attributes.
Use contribution_percentage=true for share/contribution/percentage-of-total requests and retain their
grouping. If a separate sorting field or multiple aggregation stages cannot be represented, return
is_supported=false, interpretation_confidence=0, and explain the unsupported requirements.
Ordering all grouped results by their aggregate metric uses operation="ranking" with limit=null.
Highest to lowest/largest to smallest/descending means sort="desc"; lowest to highest/smallest
to largest/ascending means sort="asc". Ordering language alone does not request Top 1. Set a
limit only for an explicit number of results or a singular superlative entity request.
Target achievement percentages use requires_targets and comparison, not contribution_percentage.
For rejected requests describe the unsupported requirement without claiming any execution.
Do not return executable Python, SQL, or final numerical results. Supported operations are aggregation, filter,
ranking, top_n_per_group, contribution, comparison, nested, and time_series. Use requested fields only when
present in the schema. If the request is ambiguous, lower interpretation confidence and list the ambiguity.
The exact required QueryPlan JSON Schema follows:
""" + json.dumps(QueryPlan.model_json_schema(), ensure_ascii=False)

    def _build_context(self, query: str, feedback: list[dict[str, Any]] | None) -> str:
        monetary, resolved_metric, metric_errors = self._monetary_metric_intent(query)
        return json.dumps({
            "query": query,
            "schema": self.schema.available_columns(),
            "dictionary": self.loader.dictionary,
            "available_monetary_metrics": self._monetary_metric_candidates(),
            "monetary_metric_resolution": {
                "requested": monetary, "metric": resolved_metric, "clarification": metric_errors,
            },
            "derived_grouping_dimensions": ["year", "quarter", "month", "week", "day"] if self.schema.date_columns() else [],
            "feedback_examples": [
                f"{item.get('query', '')} -> {item.get('correct_interpretation', '')}"
                for item in feedback or []
            ],
            "supported_operations": ["aggregation", "filter", "ranking", "top_n_per_group", "contribution", "comparison", "nested", "time_series"],
        }, ensure_ascii=False)

    def _deterministic_fallback(self, query: str) -> QueryPlan:
        lowered = query.lower()
        ranking = self._infer_ranking_request(query)
        metric = ranking["metric"] or self._infer_metric(query)
        group_by = ranking["group_by"] or self._infer_group_by(query)
        if self._is_contribution_query(query) and not group_by:
            group_by = self._infer_contribution_group(query)
        time_filter = self._infer_time_filter(query)
        requested_filters, filter_errors = self._requested_filter_intent(query)
        filter_expression = requested_filters if isinstance(requested_filters, FilterExpression) else None
        filters = [requested_filters] if isinstance(requested_filters, FilterPlan) else []
        filter_items = self._filter_items(requested_filters)
        if filter_expression and filter_expression.op == "and" and not filter_expression.groups and not re.search(r"\b(?:and|or)\b", query, re.IGNORECASE):
            # Existing flat filters express an implicit conjunction exactly.
            legacy_order = self._infer_filters(query)
            filters = [item for item in legacy_order if item in filter_items]
            filters.extend(item for item in filter_items if item not in filters)
            filter_expression = None
        requested_numeric_filters = self._extract_numeric_filters(query)
        comparison = self._infer_comparison_request(query, metric, group_by, filters)
        required_targets = bool(comparison) or any(term in lowered for term in ("target", "achievement", "gap"))
        aggregation = self._infer_aggregation(query, metric)
        unresolved_terms = self._find_unresolved_terms(query, metric, group_by, filter_items)
        unresolved_terms.extend(filter_errors)
        _, time_errors = self._requested_calendar_filter(query)
        unresolved_terms.extend(time_errors)
        unresolved_terms.extend(
            term for term in self._extract_unresolved_grouping_terms(query, group_by)
            if term not in unresolved_terms
        )
        unresolved_terms.extend(
            term for term in self._extract_unresolved_numeric_terms(query, requested_numeric_filters)
            if term not in unresolved_terms
        )
        ambiguity = []
        if ranking.get("metric_inferred") and metric:
            ambiguity.append(f"No ranking metric was specified; using the dataset default metric '{metric}'.")
        has_analytic_intent = bool(
            metric or group_by or ranking["is_ranking"] or filter_items
            or self._is_contribution_query(query) or comparison or time_filter
        )
        if not has_analytic_intent:
            unresolved_terms = list(dict.fromkeys([*unresolved_terms, query.strip()]))
            ambiguity.append("No legitimate analytical intent could be grounded.")
        if not metric and filters and not self._is_filter_only_query(query):
            ambiguity.append("Metric could not be mapped safely.")
        if unresolved_terms:
            ambiguity.extend(f"Unresolved term: {term}" for term in unresolved_terms)
        if ranking["is_ranking"] and not ranking["group_by"]:
            ambiguity.append("A product-level or other ranked dimension was unavailable.")
        if comparison and comparison.target_metric and comparison.target_group and not self._compatible_target_grain(query, comparison.target_group):
            ambiguity.append("Target comparison grain is not compatible with the requested grouping.")
        operation = "ranking" if ranking["is_ranking"] else "aggregation"
        if self._is_contribution_query(query):
            operation = "contribution"
        if comparison:
            operation = "comparison"
        if self._is_filter_only_query(query) and not ranking["is_ranking"] and operation != "comparison" and not group_by and not re.search(
            r"\b(?:total|sum|average|avg|mean|count|minimum|min|maximum|max)\b", lowered,
        ):
            operation = "filter"
        target_request = bool(
            any(term in lowered for term in ("target", "achievement", "gap"))
        )
        target_metric = comparison.target_metric if comparison else None
        if target_request and not target_metric:
            unresolved_terms = list(dict.fromkeys([*unresolved_terms, metric or "target metric"]))
            ambiguity.append("No target metric was found for the requested comparison.")
        is_supported = bool(
            (metric or operation == "filter")
            and not unresolved_terms
            and (not target_request or target_metric)
            and has_analytic_intent
            and (not ranking["is_ranking"] or ranking.get("rank_dimension"))
        )
        plan = QueryPlan(
            operation=operation,
            metric=metric,
            aggregation=aggregation,
            group_by=group_by,
            filters=filters,
            filter_expression=filter_expression,
            time_filter=time_filter,
            sort=ranking["sort"],
            limit=ranking["limit"],
            rank_dimension=ranking.get("rank_dimension"),
            partition_by=ranking.get("partition_by", []),
            limit_scope=ranking.get("limit_scope", "global"),
            top_n_per_group=ranking["limit"] if ranking.get("limit_scope") == "per_group" else None,
            contribution_percentage=self._is_contribution_query(query),
            requires_targets=required_targets,
            comparison=comparison,
            interpretation_confidence=0.65 if is_supported else 0.0,
            ambiguity=ambiguity,
            is_supported=is_supported,
            unresolved_terms=unresolved_terms,
            resolved_fields=list(dict.fromkeys([*group_by, metric] if metric else group_by)),
            semantic_errors=[
                f"Could not ground {term} against the available schema or data dictionary."
                for term in unresolved_terms
            ],
        )
        return self._normalize_plan(plan, query)

    def _infer_aggregation(self, query: str, metric: str | None) -> str:
        lowered = self._analytical_head(query).lower()
        if re.search(r"\b(?:minimum|min)\b", lowered):
            return "min"
        if re.search(r"\b(?:maximum|max)\b", lowered):
            return "max"
        if re.search(r"\b(?:average|avg|mean)\b", lowered):
            return "mean"
        if re.search(r"\b(?:count|number\s+of)\b", lowered):
            return "count"
        if metric and metric in {"revenue", "sales", "amount"}:
            return "sum"
        if any(term in lowered for term in ("total", "sum", "revenue", "sales", "amount")):
            return "sum"
        return "count" if metric is None else "sum"

    def _has_entity_reference(self, query: str) -> bool:
        words = re.findall(r"[a-z][a-z0-9_]*", query.lower())
        columns = self.schema.available_columns()["sales"]
        for width in (3, 2, 1):
            for index in range(len(words) - width + 1):
                field = self._resolve_dimension(" ".join(words[index:index + width]))
                if field and (
                    field not in columns or field.lower().endswith("_id")
                    or not self.schema.is_numeric_field(field)
                ):
                    return True
        return False

    def _resolved_metric_phrase(self, phrase: str, query: str) -> bool:
        """Recognize a whole metric description, not just one familiar word."""
        metric = self._infer_metric(phrase)
        if not metric:
            return False
        dictionary = self.loader.dictionary
        aliases = {
            "revenue", "sales", "income", "amount", "quantity", "units", "price", "cost", "profit",
            "money", "earnings", "turnover", "receipts",
            *dictionary.get("metrics", {}), *dictionary.get("synonyms", {}),
            *self.schema.available_columns()["sales"],
        }
        remainder = phrase.lower()
        for alias in sorted(aliases, key=len, reverse=True):
            if self._infer_metric(alias) == metric:
                pattern = re.escape(alias.lower()).replace("_", r"[ _]")
                remainder = re.sub(rf"\b{pattern}\b", " ", remainder)
        grammar = r"the|a|an|total|overall|sum|average|avg|mean|minimum|min|maximum|max|count|of|generated|generating"
        if self._has_entity_reference(query):
            grammar += r"|their|its|them|these|those"
        remainder = re.sub(rf"\b(?:{grammar})\b", " ", remainder)
        return not remainder.strip(" \t\r\n.,;?!")

    def _is_ranking_qualifier(self, query: str, marker: re.Match) -> bool:
        if marker.group(1).lower() != "with" or not self._has_entity_reference(query[:marker.start()]):
            return False
        clause = re.split(
            r"\b(?:where|with|having|in|for|from|by|grouped|sorted|ordered|against|versus|then)\b|[,.;?!]",
            query[marker.end():], maxsplit=1, flags=re.IGNORECASE,
        )[0].strip()
        match = re.fullmatch(
            r"(?:(?:the|their|its)\s+)?(?:highest|lowest|most|least|largest|smallest|biggest|greatest|best|worst)\s+(.+)",
            clause, re.IGNORECASE,
        )
        return bool(match and self._resolved_metric_phrase(match.group(1), query))

    def _filter_marker(self, query: str):
        for marker in re.finditer(r"\b(where|having|with)\s+", query, re.IGNORECASE):
            if not self._is_ranking_qualifier(query, marker):
                return marker
        return None

    def _analytical_head(self, query: str) -> str:
        marker = self._filter_marker(query)
        return query[:marker.start()] if marker else query

    @staticmethod
    def _filter_items(node: FilterPlan | FilterExpression | None) -> list[FilterPlan]:
        if isinstance(node, FilterExpression):
            return expression_predicates(node)
        return [node] if node is not None else []

    @staticmethod
    def _plan_filter_intent(plan: QueryPlan):
        nodes = [*plan.filters]
        if plan.filter_expression:
            nodes.append(plan.filter_expression)
        return combine_filters("and", nodes) if nodes else None

    def _implicit_categorical_clauses(self, query: str) -> list[tuple[int, int, str]]:
        """Locate value constraints so their qualifiers cannot become ranked entities."""
        clauses = []
        remainder = query
        categorical_values = [
            value for column, candidates in self.schema.categorical_index().items()
            if column not in self.schema.date_columns() for value in candidates
        ]
        for context in re.finditer(r"\b(?:in|for|from|within)\s+(?!each\b)", query, re.IGNORECASE):
            if not remainder[context.start():context.end()].strip():
                continue
            tail = remainder[context.end():]
            boundary = re.search(
                r"\b(?:where|with|having|by|per|grouped|sorted|ordered|against|versus|then|descending|ascending)\b"
                r"|\b(?:in|for|within)\s+each\b"
                r"|\bin\s+(?:descending|ascending)\b"
                r"|,\s*(?=(?:show|display|rank|sort|order|arrange)\b)"
                r"|\b(?:and\s+)?(?:show|display|rank|sort|order|arrange)\b"
                r"|\bfrom\s+(?:" + self._DESCENDING_RANGE + "|" + self._ASCENDING_RANGE + r")"
                r"|\b(?:for|in|during|before|after|since)\s+(?:year\s+)?\d{4}\b",
                tail, re.IGNORECASE,
            )
            end = context.end() + (boundary.start() if boundary else len(tail))
            clause = remainder[context.end():end].strip()
            if not any(
                re.search(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", clause, re.IGNORECASE)
                for value in categorical_values
            ):
                continue
            # Preserve the full clause, even when it contains an unknown qualifier
            # or Boolean operand. The filter parser must still reject that intent.
            clauses.append((context.start(), end, clause))
            remainder = remainder[:context.start()] + " " * (end - context.start()) + remainder[end:]
        return clauses

    def _requested_filter_intent(self, query: str):
        """Retain Boolean structure and reject any unconsumed explicit predicate."""
        parser = RequestedFilterParser(self.schema, self._resolve_dimension)
        nodes = []
        errors = []
        remainder = query
        marker = self._filter_marker(query)
        if marker and marker.group(1).lower() == "with" and re.match(r"targets?\b", query[marker.end():], re.IGNORECASE):
            marker = None
        if marker:
            if marker.group(1).lower() == "having":
                errors.append("Unsupported post-aggregation HAVING constraint")
            else:
                clause = query[marker.end():]
                # These introduce separate grouping, time, or categorical clauses.
                boundary = re.search(
                    r"\b(?:grouped\s+by|by|sorted|ordered|ranked|against|versus|during|for|from|in)\b"
                    r"|\b(?:and\s+)?(?:show|display|rank|sort|order|arrange)\b",
                    clause, re.IGNORECASE,
                )
                end = marker.end() + (boundary.start() if boundary else len(clause))
                try:
                    nodes.append(parser.parse(query[marker.end():end]))
                except (FilterSyntaxError, ValueError) as exc:
                    errors.append(str(exc))
                remainder = query[:marker.start()] + " " * (end - marker.start()) + query[end:]
        # Consume the complete implicit categorical clause, including unknown
        # operands and dangling connectors after the final grounded value.
        for start, end, clause in self._implicit_categorical_clauses(remainder):
            try:
                nodes.append(parser.parse(clause))
            except (FilterSyntaxError, ValueError) as exc:
                errors.append(str(exc))
            remainder = remainder[:start] + " " * (end - start) + remainder[end:]
        # Infer categorical predicates outside an explicit WHERE. When connectors
        # join them, parse that entire span rather than conjoining the values.
        occurrences = []
        for column, candidates in self.schema.categorical_index().items():
            if column in self.schema.date_columns():
                continue
            for value in candidates:
                if self.schema.resolve_categorical_value(value) != (column, value):
                    continue
                for match in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", remainder, re.IGNORECASE):
                    start = match.start()
                    explicit = re.search(
                        r"\b[a-z][a-z0-9_]*\s*(?:is\s+not(?:\s+equal\s+to)?|is|equals|contains|!=|==|>=|<=|=|>|<)\s*$",
                        remainder[:start], re.IGNORECASE,
                    )
                    if explicit:
                        start = explicit.start()
                    occurrences.append((start, match.end(), FilterPlan(column=column, operator="=", value=value)))
        for field, operator, value in self._numeric_predicates(remainder):
            for match in re.finditer(rf"\b{re.escape(field)}\s*(?:is\s+)?{re.escape(operator)}\s*{re.escape(value)}(?![\d.])", remainder, re.IGNORECASE):
                column = self._numeric_filter_field(field)
                normalized_operator = self._numeric_operator(operator)
                if column and normalized_operator:
                    number = float(value)
                    predicate = FilterPlan(column=column, operator=normalized_operator, value=int(number) if number.is_integer() else number)
                    occurrences.append((match.start(), match.end(), predicate))
        occurrences = list({(start, end): (start, end, item) for start, end, item in occurrences}.values())
        occurrences.sort(key=lambda item: item[0])
        if len(occurrences) > 1:
            fragment = remainder[occurrences[0][0]:occurrences[-1][1]]
            if re.search(r"\b(?:and|or|xor)\b|[()]", fragment, re.IGNORECASE):
                start = occurrences[0][0]
                # Preserve parentheses surrounding the categorical expression.
                while start > 0 and remainder[start - 1] in " (":
                    start -= 1
                end = occurrences[-1][1]
                while end < len(remainder) and remainder[end] in " )":
                    end += 1
                try:
                    nodes.append(parser.parse(remainder[start:end]))
                except (FilterSyntaxError, ValueError) as exc:
                    errors.append(str(exc))
                remainder = remainder[:start] + " " * (end - start) + remainder[end:]
            else:
                # Multiple categorical values require explicit logical intent.
                errors.append("Multiple categorical constraints require explicit AND/OR connectors.")
        else:
            for start, end, _ in occurrences:
                try:
                    nodes.append(parser.parse(remainder[start:end]))
                except (FilterSyntaxError, ValueError) as exc:
                    errors.append(str(exc))
        for predicate in self._extract_numeric_filters(remainder):
            if predicate not in self._filter_items(combine_filters("and", nodes) if nodes else None):
                nodes.append(predicate)
        # Categorical values alone cannot consume unsupported Boolean operators.
        if re.search(r"\b(?:xor|nand|nor)\b", query, re.IGNORECASE):
            errors.append("Unsupported Boolean operator; only AND/OR composition is available.")
        return (combine_filters("and", nodes) if nodes else None), list(dict.fromkeys(errors))

    def _requested_calendar_filter(self, query: str):
        matches = list(re.finditer(
            r"\b(for|in|during|year|before|prior\s+to|earlier\s+than|after|later\s+than|since|from|to|until|through|as\s+of|up\s+to)"
            r"\s+(?:the\s+)?(?:year\s+)?(\d{4}(?:-\d{2}-\d{2})?)\b", query, re.IGNORECASE,
        ))
        if not matches:
            return None, []
        requested_column = next((column for column in self.schema.date_columns() if column.lower() in query.lower()), None)
        column = self.schema.resolve_date_column(requested_column)
        if not column:
            return None, ["The calendar constraint requires an unambiguous date field."]
        values = {}
        errors = []
        for match in matches:
            operator, date = match.groups()
            operator = " ".join(operator.lower().split())
            year = int(date[:4])
            if not 1 < year < 9999:
                errors.append("The requested calendar year is outside the supported date range.")
                continue
            if len(date) == 4 and operator in {"for", "in", "during", "year"}:
                if "year" in values and values["year"] != year:
                    errors.append("Multiple calendar years require an explicit Boolean time expression.")
                values["year"] = year
            elif len(date) == 4 and operator in {"before", "prior to", "earlier than"}:
                if "end" in values:
                    errors.append("Repeated date boundaries require a combined temporal expression.")
                values["end"] = f"{year - 1:04d}-12-31T23:59:59.999999999"
            elif len(date) == 4 and operator in {"after", "later than"}:
                if "start" in values:
                    errors.append("Repeated date boundaries require a combined temporal expression.")
                values["start"] = f"{year + 1:04d}-01-01"
            elif operator in {"since", "from"}:
                if "start" in values:
                    errors.append("Repeated date boundaries require a combined temporal expression.")
                values["start"] = f"{year:04d}-01-01" if len(date) == 4 else date
            elif operator in {"to", "until", "through", "as of", "up to"}:
                if "end" in values:
                    errors.append("Repeated date boundaries require a combined temporal expression.")
                values["end"] = f"{year:04d}-12-31T23:59:59.999999999" if len(date) == 4 else date
            else:
                errors.append("This date comparison cannot be represented safely by the current calendar filter.")
        if "year" in values and ("start" in values or "end" in values):
            errors.append("Combined year and date boundaries cannot be represented without dropping a constraint.")
        if re.search(r"\d{4}(?:-\d{2}-\d{2})?\s+or\b|\bor\s+(?:(?:in|during|for|year|before|after|since|prior\s+to)\s+)?\d{4}\b", query, re.IGNORECASE):
            errors.append("Boolean calendar alternatives are unsupported; no date constraint was dropped.")
        if re.search(r"\d{4}\s*(?:and|or|,)\s*\d{4}\b", query, re.IGNORECASE):
            errors.append("A list of calendar years requires a Boolean time expression; no year was dropped.")
        return (TimeFilterPlan(column=column, **values) if values else None), errors

    def _multi_stage_requirements(self, query: str) -> list[str]:
        metric = self._infer_metric(query)
        # An ordering range is one stage; it is not selection from a ranked
        # subquery merely because the range begins with "from highest".
        calculation = re.sub(
            r"\bfrom\s+(?:" + self._DESCENDING_RANGE + "|" + self._ASCENDING_RANGE + r")",
            "", query, flags=re.IGNORECASE,
        )
        ranked_subquery = re.search(r"\b(?:of|from)\s+(?:the\s+)?(?:(?:\d+|[a-z-]+)\s+)?(?:top|bottom|highest|lowest|best|worst)\b", calculation, re.IGNORECASE)
        staged = re.search(r"\b(?:then|followed\s+by|after\s+(?:selecting|ranking|aggregating)|reaggregate|re-aggregate|aggregate\s+(?:the\s+)?selected)\b", query, re.IGNORECASE)
        if staged or (metric and ranked_subquery):
            return ["The request requires selecting/ranking entities and another calculation stage. The current QueryPlan supports one aggregation stage; no simpler calculation was substituted."]
        return []

    def _find_unresolved_terms(
        self, query: str, metric: str | None, group_by: list[str], filters: list[FilterPlan]
    ) -> list[str]:
        lowered = query.lower()
        unresolved = []

        # A syntactically valid plan must still account for every explicit request.
        unsupported_operations = (
            r"\b(?:median|mode|variance|percentile|quantile)\b",
            r"\b(?:standard\s+deviation|std(?:dev)?)\b",
        )
        for pattern in unsupported_operations:
            unresolved.extend(match.group(0) for match in re.finditer(pattern, lowered))
        unresolved.extend(self._relative_time_constraints(query))
        unresolved.extend(self._monetary_metric_intent(query)[2])
        unresolved.extend(self._extract_unresolved_requested_fields(query, metric))
        for basis in re.finditer(r"\bbased\s+on\s+([^.;?!]+)", query, re.IGNORECASE):
            clause = re.split(
                r"\b(?:where|with|having|by|per|in|for|from|across|within|ordered|sorted|ranked|then)\b|,",
                basis.group(1), maxsplit=1, flags=re.IGNORECASE,
            )[0].strip()
            if not self._resolved_metric_phrase(clause, query):
                unresolved.append(f"Unresolved ranking basis: {clause}")
            elif self._infer_metric(clause) != metric:
                unresolved.append(f"The requested ranking basis was not preserved: {clause}")
        _, filter_errors = self._requested_filter_intent(query)
        unresolved.extend(filter_errors)
        for match in re.finditer(r"\b(?:exclude|excluding|except|without|not\s+in)\b", lowered):
            unresolved.append(f"Unsupported exclusion constraint: {match.group(0)}")

        for term in ("temperature", "salary", "rainfall", "visitors", "age"):
            standalone = re.search(rf"\b{term}\b", lowered)
            if standalone and term not in {metric or "", *group_by}:
                unresolved.append(term)

        for context_match in re.finditer(
            r"\b(?:on|in|for|from|across|among|within)\s+([A-Za-z][A-Za-z0-9_]*)",
            query,
            flags=re.IGNORECASE,
        ):
            context = context_match.group(1).strip()
            tail = query[context_match.start():].split(None, 1)[-1]
            dimension_clause = re.split(
                r"\b(?:and|or|where|with|having|in|for|from|by|ordered|sorted|ascending|descending)\b|[.;?!,]",
                tail, maxsplit=1, flags=re.IGNORECASE,
            )[0].strip()
            dimension_clause = re.sub(r"^(?:the|each|every)\s+", "", dimension_clause, flags=re.IGNORECASE)
            if self.schema.resolve_categorical_phrase(dimension_clause, self._resolve_dimension):
                continue
            if self._resolve_dimension(dimension_clause):
                continue
            if context.casefold() in {"their", "its", "them", "these", "those"} and self._has_entity_reference(query):
                if self._resolved_metric_phrase(dimension_clause, query):
                    continue
                dimension_clause = re.sub(r"^(?:their|its|them|these|those)\s+", "", dimension_clause, flags=re.IGNORECASE)
            if re.search(r"\bbased\s+$", query[:context_match.start()], re.IGNORECASE):
                if self._resolved_metric_phrase(dimension_clause, query) and self._infer_metric(dimension_clause) == metric:
                    continue
            if dimension_clause:
                context = dimension_clause.split()[0]
            if context.casefold() in {"descending", "ascending"} and re.match(r"\s+order\b", query[context_match.end():], re.IGNORECASE):
                continue
            if context.casefold() in {"all", "every"}:
                tail = query[context_match.end():].strip().rstrip(".?!")
                entity = re.split(r"\b(?:by|where|with|having|in|for|from|and)\b", tail, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                if entity.casefold() in {"rows", "records", "data"} or self._resolve_dimension(entity):
                    continue
            if re.match(self._DESCENDING_RANGE + "|" + self._ASCENDING_RANGE, tail, re.IGNORECASE):
                continue
            if context.casefold() == "each" or context.casefold().startswith("each "):
                continue
            if context.casefold() == "year" and re.search(r"\byear\s+\d{4}\b", query, re.IGNORECASE):
                continue
            if re.fullmatch(r"(?:city|region|country|product|category|channel|customer|customer_segment|product_category|product_subcategory|product_name)", context, re.IGNORECASE):
                continue
            if self.schema.resolve_categorical_value(context):
                continue
            if self.schema.canonical_field(context):
                continue
            unresolved.append(context)
        return list(dict.fromkeys(unresolved))

    def _relative_time_constraints(self, query: str) -> list[str]:
        """Relative dates need an explicit clock/anchor policy; do not use all rows."""
        pattern = (
            r"\b(?:last|previous|this|current|next|past|latest)\s+"
            r"(?:(?:\d+|[a-z]+)\s+)?(?:days?|weeks?|months?|quarters?|years?)\b"
            r"|\b(?:today|yesterday|tomorrow|year\s+to\s+date|month\s+to\s+date)\b"
        )
        return [
            f"Unsupported relative time constraint: {match.group(0)}"
            for match in re.finditer(pattern, query, re.IGNORECASE)
        ]

    def _extract_unresolved_requested_fields(self, query: str, metric: str | None) -> list[str]:
        unresolved = []
        # Identifier-shaped tokens are explicit field references, wherever they occur.
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b", query):
            if not self._resolve_dimension(token) and not self._infer_metric(token):
                unresolved.append(token)

        # QueryPlan represents one metric. A second requested metric cannot be dropped.
        head = re.split(
            r"\b(?:grouped|by|per|where|with|having|in|for|from|across|within|over|during|against|versus|vs)\b",
            query, maxsplit=1, flags=re.IGNORECASE,
        )[0]
        number_token = (
            r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
            r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
            r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion"
        )
        head = re.sub(
            rf"\b({number_token})\s+and\s+(?=(?:{number_token})\b)",
            r"\1 ", head, flags=re.IGNORECASE,
        )
        requested = re.split(r"\s+(?:and|or)\s+|\s*,\s*", head)
        for candidate in requested[1:]:
            ordering_description = (
                r"(?:(?:ranked|sorted|ordered|arranged)\s+)?(?:"
                + self._DESCENDING_RANGE + "|" + self._ASCENDING_RANGE
                + r"|(?:highest|lowest|largest|smallest|most|least)\s+first"
                + r"|(?:ascending|descending)(?:\s+order)?)"
            )
            if re.fullmatch(ordering_description, candidate.strip(" .?!"), re.IGNORECASE):
                continue
            candidate = re.sub(
                r"^(?:(?:the|total|sum|average|avg|mean|minimum|min|maximum|max|count|of)\s+)+",
                "", candidate.strip(), flags=re.IGNORECASE,
            ).strip(" .?!")
            if not candidate or candidate.lower() in {"target", "targets"}:
                continue
            exact = self.schema.resolve_business_term(candidate)
            dictionary = self.loader.dictionary if isinstance(self.loader.dictionary, dict) else {}
            declared = dictionary.get("metrics", {})
            alias = dictionary.get("synonyms", {}).get(candidate.lower())
            resolved = exact or (candidate.lower() if candidate.lower() in declared else alias)
            if not resolved:
                unresolved.append(candidate)
            elif resolved != metric:
                unresolved.append(f"Additional requested field cannot be represented: {candidate}")
        return unresolved

    def _extract_unparsed_filter_constraints(
        self, query: str, filters: list[FilterPlan],
    ) -> list[str]:
        unresolved = []
        for match in re.finditer(r"\b(where|having|with)\s+([^.;?!]+)", query, re.IGNORECASE):
            marker, clause = match.groups()
            if marker.lower() == "having":
                unresolved.append("Unsupported post-aggregation HAVING constraint")
                continue
            if marker.lower() == "with":
                if re.search(r"\btargets?\b", clause, re.IGNORECASE):
                    continue
                first_term = clause.split()[0]
                if not (
                    self._resolve_dimension(first_term) or self._infer_metric(first_term)
                    or "_" in first_term
                    or re.search(r"\b(?:records?|rows?|products?)\b", query, re.IGNORECASE)
                ):
                    continue
            clause = re.split(
                r"\b(?:by|grouped|ranked|in|for|from|during|against|versus|vs)\b",
                clause, maxsplit=1, flags=re.IGNORECASE,
            )[0].strip()
            for predicate in re.split(r"\s+(?:and|or(?!\s+equal\s+to))\s+", clause, flags=re.IGNORECASE):
                predicate = predicate.strip()
                recognized = False
                for field, operator, value in self._numeric_predicates(predicate):
                    direct = rf"{re.escape(field)}\s*(?:is\s+)?{re.escape(operator)}\s*{re.escape(value)}"
                    inverse = rf"{re.escape(operator)}\s+{re.escape(value)}\s+(?:for|of)\s+{re.escape(field)}"
                    if re.fullmatch(direct + "|" + inverse, predicate, re.IGNORECASE):
                        recognized = True
                        break
                if not recognized:
                    recognized = bool(re.fullmatch(
                        r"[a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)*?"
                        r"(?:\s+(?:is\s+not|is|equals|contains)\s+|\s*[><=!]+\s*)\S.*",
                        predicate, re.IGNORECASE,
                    ))
                if not recognized:
                    resolved_value = self.schema.resolve_categorical_value(predicate.strip("\"'"))
                    recognized = bool(resolved_value and any(
                        item.column == resolved_value[0] and item.operator == "="
                        and item.value == resolved_value[1] for item in filters
                    ))
                if not recognized:
                    unresolved.append(f"Unparsed filter constraint: {predicate}")
        return unresolved

    def _extract_unresolved_categorical_terms(
        self, query: str, filters: list[FilterPlan],
    ) -> list[str]:
        unresolved = []
        predicate_start = r"\b(?:where|with|and|or)\s+([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)*?)"
        predicate_value = r"([^.,;?!]+?)(?=\s+(?:and|or|by|grouped|for)\b|[.,;?!]|$)"
        patterns = (
            predicate_start + r"\s+(is\s+not|is|equals|contains)\s+" + predicate_value,
            predicate_start + r"\s*([><=!]+)\s*" + predicate_value,
        )
        predicates = [
            match for pattern in patterns
            for match in re.finditer(pattern, query, re.IGNORECASE)
        ]
        numeric_words = r"^(?:at\s+(?:least|most)|(?:greater|more|less)\s+than|above|over|below|under|equal\s+to)\b"
        for match in predicates:
            column_phrase, operator, value_phrase = match.groups()
            value_phrase = re.split(
                r"\b(?:and|or|by|grouped|in|for)\b", value_phrase, maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip().strip("\"'")
            if re.match(numeric_words, value_phrase, re.IGNORECASE) or re.fullmatch(
                r"[-+]?\d+(?:\.\d+)?", value_phrase,
            ):
                continue
            column = self._resolve_dimension(column_phrase)
            if not column:
                unresolved.append(column_phrase)
                continue
            expected_operator = {
                "is": "=", "equals": "=", "is not": "!=", "contains": "contains",
                "=": "=", "==": "=", "!=": "!=",
            }.get(" ".join(operator.lower().split()))
            if not any(
                item.column == column and item.operator == expected_operator
                and str(item.value).casefold() == value_phrase.casefold()
                for item in filters
            ):
                unresolved.append(f"Unresolved filter: {column_phrase} {operator} {value_phrase}")
        # These constraints cannot be represented by this plan's flat AND predicates.
        filter_clause = re.search(r"\b(?:where|with|having)\b(.+)", query, re.IGNORECASE)
        if filter_clause:
            clause = filter_clause.group(1)
            if re.search(r"\bor\b", re.sub(r"\bor\s+equal\s+to\b", "", clause, flags=re.IGNORECASE), re.IGNORECASE):
                unresolved.append("Unsupported disjunctive filter constraint")
            for match in re.finditer(
                r"\b([a-z][a-z0-9_]*)\s+(?:is\s+)?"
                r"(between|approximately|roughly|around|outside|not\s+equal\s+to)\s+[-+]?\d",
                clause, re.IGNORECASE,
            ):
                unresolved.append(f"Unsupported filter operator: {match.group(2)}")
        return unresolved

    def _normalize_plan(self, plan: QueryPlan, query: str) -> QueryPlan:
        if plan.metric:
            canonical_metric = self._infer_metric(plan.metric)
            if canonical_metric:
                plan.metric = canonical_metric
        ranking = self._infer_ranking_request(query)
        if not ranking["is_ranking"] or plan.operation not in {"ranking", "top_n_per_group"}:
            return plan
        if not plan.group_by:
            plan.group_by = ranking["group_by"]
        if not plan.metric:
            plan.metric = ranking["metric"]
            if ranking.get("metric_inferred") and plan.metric:
                assumption = f"No ranking metric was specified; using the dataset default metric '{plan.metric}'."
                if assumption not in plan.ambiguity:
                    plan.ambiguity.append(assumption)
        if not plan.rank_dimension:
            plan.rank_dimension = ranking["rank_dimension"]
        if ranking["partition_by"]:
            plan.partition_by = list(dict.fromkeys([*plan.partition_by, *ranking["partition_by"]]))
            plan.group_by = list(dict.fromkeys([*plan.partition_by, *plan.group_by]))
            if plan.rank_dimension and plan.rank_dimension not in plan.group_by:
                plan.group_by.append(plan.rank_dimension)
            plan.limit_scope = "per_group"
        if plan.aggregation == "sum" and ranking["metric"]:
            plan.aggregation = "sum"
        plan.sort = ranking["sort"]
        explicit_limit = self._extract_number(query.lower())
        if explicit_limit is not None:
            plan.limit = explicit_limit
        elif plan.limit is None and ranking["limit"] is not None:
            plan.limit = ranking["limit"]
        if plan.operation == "top_n_per_group" and plan.limit_scope == "per_group":
            plan.top_n_per_group = plan.limit
        return plan

    def _is_contribution_query(self, query: str) -> bool:
        lowered = query.lower()
        # An achievement percentage uses the target as its denominator, not
        # the actual metric's total. Preserve that distinction across wording.
        if re.search(r"\b(?:targets?|achievement|attainment)\b", lowered):
            return bool(re.search(r"\b(?:contribut(?:e|es|ed|ing|ion)|share|portion|proportion)\b", lowered))
        return bool(re.search(r"\b(?:contribut(?:e|es|ed|ing|ion)|share|portion|proportion)\b", lowered)) or any(term in lowered for term in (
            "percentage of total", "percentage contribution", "contributed by",
            "share of total", "share came from", "came from each", "contribution",
        )) or bool(
            re.search(r"\b(?:percentage|percent)\b|%", lowered)
            and re.search(r"\b(?:total|each|every|by|per|from)\b", lowered)
        )

    def _infer_contribution_group(self, query: str) -> list[str]:
        lowered = query.lower()
        patterns = (
            r"\bcontributed\s+by\s+each\s+([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)?)",
            r"\b(?:came from|came from each|from each)\s+([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)?)",
            r"\beach\s+([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)?)",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            candidate = match.group(1).strip()
            resolved = self._resolve_dimension(candidate)
            if not resolved:
                resolved = self._resolve_dimension(candidate.split()[0])
            if resolved:
                return [resolved]
        return []

    def _infer_metric(self, query: str) -> str | None:
        lowered = query.lower()
        if not self._is_filter_only_query(query):
            lowered = self._analytical_head(query).lower()
        conversational, monetary_metric, _ = self._monetary_metric_intent(lowered)
        if conversational:
            return monetary_metric
        dictionary = self.schema.loader.dictionary
        metrics = dictionary.get("metrics", {}) if isinstance(dictionary, dict) else {}
        synonyms = dictionary.get("synonyms", {}) if isinstance(dictionary, dict) else {}
        metric_aliases = {
            "revenue": "revenue",
            "sales": "revenue",
            "income": "revenue",
            "amount": "revenue",
            "quantity": "quantity",
            "units": "quantity",
            "price": "unit_price",
            "cost": "cost",
            "profit": "profit",
        }
        for term, fallback in metric_aliases.items():
            if not re.search(rf"\b{re.escape(term)}\b", lowered):
                continue
            if term in synonyms and isinstance(synonyms[term], str):
                resolved = synonyms[term]
                if resolved in metrics:
                    return resolved
                if self.schema.validate_field(resolved, "sales"):
                    return resolved
            if term in metrics and term not in synonyms:
                return term
            if self.schema.validate_field(fallback, "sales"):
                return fallback
        grounded_aliases = {**{name: name for name in metrics}, **synonyms}
        for alias, target in sorted(grounded_aliases.items(), key=lambda item: len(str(item[0])), reverse=True):
            if not isinstance(target, str):
                continue
            phrase = re.escape(str(alias)).replace("_", r"[ _]")
            if not re.search(rf"\b{phrase}\b", lowered):
                continue
            if target in metrics:
                return target
            canonical = self.schema.canonical_field(target)
            if canonical in self.schema.available_columns()["sales"] and self.schema.is_numeric_field(canonical):
                return canonical
        for column in self.schema.available_columns()["sales"]:
            phrase = re.escape(column.lower()).replace("_", r"[ _]")
            if self.schema.is_numeric_field(column) and re.search(rf"\b{phrase}\b", lowered):
                return column
        if re.search(r"\b(?:targets?|achievement|attainment)\b", lowered):
            # Infer an omitted actual metric only when target metadata identifies
            # one unambiguous compatible metric. Never pick an arbitrary measure.
            candidates = [*metrics, *self.schema.available_columns()["sales"]]
            compatible = list(dict.fromkeys(
                candidate for candidate in candidates
                if self.schema.is_numeric_field(candidate)
                and self.schema.resolve_target_metric(candidate) is not None
            ))
            if len(compatible) == 1:
                return compatible[0]
        return None

    def _executable_metric(self, metric: str) -> bool:
        columns = self.schema.available_columns()["sales"]
        if metric in columns:
            return self.schema.is_numeric_field(metric)
        return bool(
            metric == "revenue" and "revenue" in self.loader.dictionary.get("metrics", {})
            and {"quantity", "unit_price", "discount"}.issubset(columns)
        )

    def _monetary_metric_candidates(self) -> list[str]:
        dictionary = self.loader.dictionary
        names = {"revenue", "sales", "income", "amount", "profit", "earnings", "net_income", "turnover", "receipts", "margin"}
        synonyms = dictionary.get("synonyms", {})
        for alias, target in synonyms.items():
            if alias.lower().replace("_", " ") in names or alias.lower() in {"money", "sales money"}:
                if isinstance(target, str):
                    names.add(target)
        return sorted(name for name in names if self._executable_metric(name))

    def _monetary_metric_intent(self, query: str) -> tuple[bool, str | None, list[str]]:
        lowered = self._analytical_head(query).lower()
        words = re.findall(r"\b(?:money|earnings|turnover|receipts)\b", lowered)
        if not words:
            return False, None, []
        dictionary = self.loader.dictionary
        synonyms = dictionary.get("synonyms", {})
        explicit = []
        for name in {"revenue", "sales", "income", "amount", "profit", "net_income", "margin"}:
            if re.search(rf"\b{re.escape(name).replace('_', '[ _]')}\b", lowered):
                target = synonyms.get(name, "revenue" if name in {"sales", "income", "amount"} else name)
                if isinstance(target, str) and self._executable_metric(target):
                    explicit.append(target)
                elif self._executable_metric(name):
                    explicit.append(name)
                else:
                    return True, None, [f"Requested monetary metric is unavailable: {name}."]
        mapped = [synonyms[word] for word in words if isinstance(synonyms.get(word), str) and self._executable_metric(synonyms[word])]
        candidates = sorted(set(explicit or mapped)) or self._monetary_metric_candidates()
        default = dictionary.get("default_metric")
        if not explicit and not mapped and default in candidates:
            return True, default, []
        if len(candidates) == 1:
            return True, candidates[0], []
        if len(candidates) > 1:
            return True, None, ["Ambiguous monetary metric; please clarify " + " or ".join(candidates) + "."]
        return True, None, ["No available monetary metric can safely resolve the requested business wording."]

    def _default_ranking_metric(self) -> str | None:
        """Use declared business defaults, and avoid choosing an arbitrary numeric column."""
        dictionary = self.loader.dictionary if isinstance(self.loader.dictionary, dict) else {}
        metrics = dictionary.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        default = dictionary.get("default_metric")
        if isinstance(default, str):
            resolved = self._infer_metric(default)
            if resolved:
                return resolved
        if "revenue" in metrics or self.schema.validate_field("revenue", "sales"):
            return "revenue"
        if len(metrics) == 1:
            return next(iter(metrics))
        dimensions = dictionary.get("dimensions", [])
        candidates = [
            column for column in self.schema.available_columns()["sales"]
            if self.schema.is_numeric_field(column)
            and column not in dimensions
            and not column.lower().endswith("_id")
            and column.lower() != "id"
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _requested_ordering(self, query: str, metric: str | None) -> dict[str, Any]:
        lowered = query.lower()
        descending = self._DESCENDING_RANGE + r"|\b(?:descending|desc)\b|\b(?:highest|largest|biggest|most)\s+first\b"
        ascending = self._ASCENDING_RANGE + r"|\b(?:ascending|asc)\b|\b(?:lowest|smallest|least)\s+first\b"
        directions = list(re.finditer(descending + "|" + ascending, lowered))
        direction = None
        errors = []
        if directions:
            values = ["desc" if re.fullmatch(descending, item.group()) else "asc" for item in directions]
            if len(set(values)) > 1:
                errors.append("Conflicting sorting directions cannot be represented by one ordering.")
            direction = values[0]
        ordering = re.search(r"\b(?:sort(?:ed)?|order(?:ed)?|arrange(?:d)?)\b", lowered)
        field_match = re.search(
            r"\b(?:sort(?:ed)?|order(?:ed)?|arrange(?:d)?)\b[^.;?!]*?\b(?:by|on)\s+([^.;?!]+)",
            lowered,
        )
        if field_match:
            requested = re.split(
                r"\b(?:ascending|descending|asc|desc|from|in|for|where|and|then)\b",
                field_match.group(1), maxsplit=1,
            )[0].strip()
            field = self._infer_metric(requested) or self._resolve_dimension(requested)
            if not field or field != metric:
                errors.append("Explicit sorting by a separate or unresolved field is unsupported by this QueryPlan; no calculation was executed.")
        return {"requested": bool(ordering or directions), "sort": direction, "errors": errors}

    def _infer_ranking_request(self, query: str) -> dict[str, Any]:
        lowered = self._analytical_head(query).lower()
        if self._is_filter_only_query(query):
            return {"is_ranking": False, "metric": None, "group_by": [], "sort": "desc", "limit": None}
        ranking_markers = (
            "top", "bottom", "highest", "lowest", "least", "smallest",
            "largest", "biggest", "greatest", "most", "best", "worst", "ranked", "rank", "ranking", "first", "last",
        )
        ordering = self._requested_ordering(query, self._infer_metric(query))
        comparison_safe = re.sub(r"\bat\s+(?:least|most)\b", "", lowered)
        is_ranking = ordering["requested"] or any(
            re.search(rf"\b{re.escape(marker)}\b", comparison_safe)
            for marker in ranking_markers
        )
        if not is_ranking:
            return {"is_ranking": False, "metric": None, "group_by": [], "sort": "desc", "limit": None}

        ascending = any(
            re.search(rf"\b{re.escape(marker)}\b", comparison_safe)
            for marker in ("bottom", "lowest", "least", "smallest", "worst")
        )
        sort = ordering["sort"] or ("asc" if ascending else "desc")
        numeric = self._extract_number(lowered)
        dimension = self._infer_ranked_dimension(query)
        singular_superlative = bool(
            dimension and not ordering["requested"]
            and re.search(r"\b(?:highest|lowest|most|least|best|worst|largest|biggest|greatest|smallest)\b", lowered)
            and not re.search(r"\b(?:rank|ranked|ranking|sort|sorted|order|ordered)\b", lowered)
            and not any(
                self._resolve_dimension(word) == dimension
                for word in re.findall(r"\b[a-z_]+s\b", lowered)
            )
        )
        limit = numeric if numeric is not None else (1 if singular_superlative else None)
        metric = self._infer_metric(query)
        metric_inferred = not metric
        if not metric:
            metric = self._default_ranking_metric()
        partition_by = self._infer_partition_dimensions(query)
        partition_by = [field for field in partition_by if field != dimension]
        limit_scope = "per_group" if partition_by else "global"
        group_by = [*partition_by, dimension] if dimension else list(partition_by)
        return {
            "is_ranking": True,
            "metric": metric,
            "metric_inferred": metric_inferred and bool(metric),
            "group_by": group_by,
            "rank_dimension": dimension,
            "partition_by": partition_by,
            "limit_scope": limit_scope,
            "sort": sort,
            "limit": limit,
        }

    def _extract_number(self, lowered: str) -> int | None:
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
            "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
            "ninety": 90,
        }
        scales = {"hundred": 100, "thousand": 1000, "million": 1000000}
        word = "|".join([*number_words, *scales])
        number = rf"(?:\d+|(?:{word})(?:[ -]+(?:{word}|and))*)"
        ranking_word = r"(?:top|bottom|highest|lowest|best|worst|largest|biggest|greatest|smallest|ranked|first|last)"

        def parse(value: str) -> int:
            if value.isdigit():
                return int(value)
            total = current = 0
            for token in re.split(r"[ -]+", value):
                if token in number_words:
                    current += number_words[token]
                elif token == "hundred":
                    current = max(current, 1) * 100
                elif token in {"thousand", "million"}:
                    total += max(current, 1) * scales[token]
                    current = 0
            return total + current

        # Limit phrases surround ranking words, so numeric filter thresholds and
        # calendar years cannot accidentally become the requested Top N.
        for pattern in (
            rf"\b{ranking_word}\s+({number})\b",
            rf"\b({number})\s+{ranking_word}\b",
        ):
            match = re.search(pattern, lowered)
            if match:
                return parse(match.group(1))
        for match in re.finditer(rf"\b({number})\s+([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)?)", lowered):
            candidate = match.group(2)
            if self._resolve_dimension(candidate) or self._resolve_dimension(candidate.split()[0]):
                return parse(match.group(1))
        return None

    def _infer_partition_dimensions(self, query: str) -> list[str]:
        lowered = query.lower()
        patterns = (
            r"\b(?:in|for|within)\s+(?:each\s+)?([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)?)",
            r"\bper\s+([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)?)",
        )
        partitions = []
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                candidate = match.group(1).strip()
                resolved = self._resolve_dimension(candidate)
                if resolved and resolved not in partitions:
                    partitions.append(resolved)
        return partitions

    def _infer_ranked_dimension(self, query: str) -> str | None:
        entity_text = query
        for start, end, _ in self._implicit_categorical_clauses(query):
            entity_text = entity_text[:start] + " " * (end - start) + entity_text[end:]
        lowered = entity_text.lower()
        # A partition dimension must not replace the dimension being ranked.
        lowered = re.split(r"\b(?:in|for|within)\s+each\b|\bper\b", lowered, maxsplit=1)[0]
        lowered = re.split(r"\b(?:where|with|having)\b", lowered, maxsplit=1)[0]
        # Explicit compound fields must precede their bare entity aliases.
        for field in sorted(self.schema.available_columns()["sales"], key=len, reverse=True):
            if "_" not in field or field == self._infer_metric(query):
                continue
            phrase = re.escape(field.lower()).replace("_", r"[ _]")
            if re.search(rf"\b{phrase}s?\b", lowered):
                return field
        requested_groups = [field for field in self._infer_group_by(lowered) if field != self._infer_metric(query)]
        if len(requested_groups) == 1:
            return requested_groups[0]
        if re.search(r"\bproduct\s+categor(?:y|ies)\b", lowered):
            return self._resolve_dimension("product_category")
        if re.search(r"\bcategor(?:y|ies)\b", lowered):
            return self._resolve_dimension("category")

        dimension_patterns = (
            r"(?:top|bottom)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z][a-z0-9_]*(?:s)?)\s+by\s+([a-z][a-z0-9_]*)",
            r"([a-z][a-z0-9_]*(?:s)?)\s+by\s+([a-z][a-z0-9_]*)",
            r"([a-z][a-z0-9_]*(?:s)?)\s+(?:generated|produced|represented|ranked)\s+(?:the\s+)?(?:most|highest|lowest|least|largest|smallest|best|worst)\s+([a-z][a-z0-9_]*)",
        )
        for pattern in dimension_patterns:
            match = re.search(pattern, lowered)
            if match:
                candidates = [match.group(1), match.group(2)] if len(match.groups()) > 1 else [match.group(1)]
                for candidate in candidates:
                    resolved = self._resolve_dimension(candidate)
                    if resolved and resolved != self._infer_metric(query):
                        return resolved
        for term in ("product", "category", "region", "customer", "channel"):
            if re.search(rf"\b{term}(?:s)?\b", lowered):
                resolved = self._resolve_dimension(term)
                if resolved:
                    return resolved
        metric = self._infer_metric(query)
        words = re.findall(r"[a-z][a-z0-9_]*", lowered)
        for width in (3, 2, 1):
            for index in range(len(words) - width + 1):
                if width == 1 and words[index] == "order" and re.search(r"\border\s+(?:them|it|the|by|from)\b", lowered):
                    continue
                resolved = self._resolve_dimension(" ".join(words[index:index + width]))
                if resolved and resolved != metric:
                    return resolved
        return None

    def _resolve_dimension(self, term: str) -> str | None:
        term = re.sub(r"^(?:(?:the|a|an|their|its|these|those)\s+)+", "", term.strip(), flags=re.IGNORECASE)
        normalized = term.strip().lower().replace("-", "_").replace(" ", "_")
        singular_aliases = {
            "products": "product",
            "categories": "category",
            "regions": "region",
            "customers": "customer",
            "channels": "channel",
            "months": "month",
            "years": "year",
            "quarters": "quarter",
            "weeks": "week",
            "days": "day",
        }
        normalized = singular_aliases.get(normalized, normalized)
        derived = self.schema.resolve_derived_dimension(normalized)
        if derived:
            return derived

        exact = self.schema.canonical_field(normalized)
        if exact:
            return exact
        if normalized.endswith("s"):
            singular = normalized[:-1]
            if normalized.endswith("ies"):
                singular = normalized[:-3] + "y"
            exact = self.schema.canonical_field(singular)
            if exact:
                return exact

        columns = self.schema.available_columns()["sales"]
        normalized_columns = {
            str(column).strip().lower().replace("-", "_"): str(column)
            for column in columns
        }
        semantic_aliases = {
            "product": ("product_name", "product_id", "item_name", "item"),
            "category": ("product_category", "category"),
            "region": ("region",),
            "customer": ("customer_id", "customer_name"),
            "channel": ("channel",),
        }
        identity = self.schema.resolve_entity_dimension(normalized)
        if identity:
            return identity
        entity_candidates = [
            field for field in normalized_columns
            if field.endswith(("_id", "_name"))
            and field.rsplit("_", 1)[0].rstrip("s") == normalized.rstrip("s")
        ]
        if entity_candidates and not identity:
            return None
        for candidate in semantic_aliases.get(normalized, (normalized,)):
            if candidate in normalized_columns:
                return normalized_columns[candidate]

        dictionary = self.schema.loader.dictionary
        aliases = dictionary.get("terms", {}) if isinstance(dictionary, dict) else {}
        if isinstance(aliases, dict):
            term_definition = aliases.get(normalized) or aliases.get(term)
            if isinstance(term_definition, dict):
                declared_column = term_definition.get("column")
                if declared_column and self.schema.canonical_field(declared_column):
                    if declared_column.lower().endswith(("_segment", "_category")) and not normalized.endswith(("segment", "category")):
                        return None
                    return self.schema.canonical_field(declared_column)
            if isinstance(term_definition, str) and self.schema.canonical_field(term_definition):
                if term_definition.lower().endswith(("_segment", "_category")) and not normalized.endswith(("segment", "category")):
                    return None
                return self.schema.canonical_field(term_definition)

        declared = self.schema.resolve_business_term(normalized)
        if declared and not declared.lower().endswith(("_segment", "_category")):
            return declared if declared in columns else None

        return None

    def _infer_group_by(self, query: str) -> list[str]:
        candidates = self._extract_grouping_candidates(query)
        resolved = []
        for candidate in candidates:
            field = self._resolve_dimension(candidate)
            if field and field not in resolved:
                resolved.append(field)
        return resolved

    def _extract_grouping_candidates(self, query: str) -> list[str]:
        lowered = query.lower()
        candidates: list[str] = []
        pattern = r"\b(?:grouped\s+(?:by|on)|by|per|within|across|among|for\s+each)\s+([^.;?!]+)"
        for match in re.finditer(pattern, lowered):
            clause = re.split(
                r"\b(?:where|with|having|in|for|from|across|within|over|during|against|versus|vs|sorted|ordered|ascending|descending)\b"
                r"|\bbased\s+on\b"
                r"|\b(?:and\s+)?(?:show|display|rank|sort|order|arrange|report|return|calculate)\b",
                match.group(1), maxsplit=1,
            )[0]
            for candidate in self._split_grouping_candidates(clause):
                if re.match(r"^(?:all|every)\s+", candidate):
                    scope = re.sub(r"^(?:all|every)\s+", "", candidate)
                    dimension = self._resolve_dimension(scope)
                    if scope in {"rows", "records", "data"} or (dimension and dimension.endswith("_id")):
                        continue
                    candidate = scope
                if self.schema.resolve_categorical_phrase(candidate, self._resolve_dimension):
                    continue
                candidates.append(candidate)
        return list(dict.fromkeys(candidates))

    def _split_grouping_candidates(self, clause: str) -> list[str]:
        candidates = []
        for candidate in re.split(r"\s*,\s*(?:and\s+)?|\s+(?:and|or)\s+", clause):
            candidate = re.sub(r"^(?:each|every|the)\s+", "", candidate.strip())
            candidate = re.sub(r"\s+only$", "", candidate).strip()
            if candidate and candidate not in {"each", "every"}:
                candidates.append(candidate)
        return candidates

    def _extract_unresolved_grouping_terms(self, query: str, group_by: list[str]) -> list[str]:
        unresolved = []
        metric = self._infer_metric(query)
        for candidate in self._extract_grouping_candidates(query):
            if (metric and self._infer_metric(candidate) == metric) or candidate in {"each", "every"}:
                continue
            if self._resolve_dimension(candidate) is None:
                unresolved.append(candidate)
        return list(dict.fromkeys(unresolved))

    def _is_filter_only_query(self, query: str) -> bool:
        lowered = query.lower()
        head = self._analytical_head(query).lower()
        if re.search(r"\b(?:top|bottom|highest|lowest|least|best|worst|ranked)\b", head):
            return False
        if re.search(r"\b(?:sum|total|count|average|mean|minimum|min|maximum|max)\b", head):
            return False
        return bool(
            re.search(r"\b(?:records?|products?|rows?)\b", head)
            and re.search(r"\bwhere\b", lowered)
        )

    def _infer_comparison_request(self, query: str, metric: str | None, group_by: list[str], filters: list[FilterPlan]) -> Any | None:
        lowered = query.lower()
        comparison_markers = (
            "compare", "vs", "versus", "against target", "actual vs target",
            "achievement", "performance against target", "target achievement",
        )
        if not any(marker in lowered for marker in comparison_markers):
            return None
        if not metric:
            return None
        target_metric = self.schema.resolve_target_metric(metric)
        if not target_metric:
            return None
        comparison_groups = [
            field for field in group_by
            if field in self.loader.targets.columns
        ]
        if not comparison_groups:
            filter_group = next(
                (
                    filter_item.column
                    for filter_item in filters
                    if filter_item.column in self.loader.targets.columns
                ),
                None,
            )
            if filter_group:
                comparison_groups = [filter_group]
        if not comparison_groups:
            detected_group = self._detect_target_group_dimension(query)
            comparison_groups = [detected_group] if detected_group else []
        if comparison_groups:
            group_by = list(dict.fromkeys([*group_by, *comparison_groups]))
        target_group = comparison_groups[0] if comparison_groups else None
        if not self._compatible_target_grain(query, target_group):
            return None
        return ComparisonPlan(
            kind="actual_vs_target",
            metric=metric,
            target_metric=target_metric,
            target_group=target_group,
            target_column=target_metric,
        )

    def _detect_target_group_dimension(self, query: str) -> str | None:
        lowered = query.lower()
        for term in ("region", "category", "product", "customer", "channel", "country", "city"):
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                resolved = self._resolve_dimension(term)
                if resolved in self.loader.targets.columns:
                    return resolved
        return None

    def _compatible_target_grain(self, query: str, target_group: str | None) -> bool:
        if self.loader.targets.empty:
            return False
        target_metric = self.schema.resolve_target_metric(self._infer_metric(query))
        if target_metric not in self.loader.targets.columns:
            return False
        # An overall sum does not require a join key: targets can be rolled up
        # from their available dimensional grain into a single total.
        return target_group is None or target_group in self.loader.targets.columns

    def _infer_time_filter(self, query: str) -> TimeFilterPlan | None:
        lowered = query.lower()
        if self._relative_time_constraints(query):
            return None
        requested, errors = self._requested_calendar_filter(query)
        if requested or errors:
            return requested
        date_columns = self.schema.date_columns()
        if not date_columns or not any(token in lowered for token in ("year", "quarter", "date", "latest", "previous")):
            return None
        year_match = re.search(r"\byear\s+(\d{4})\b", lowered)
        temporal_groups = {"year", "quarter", "month", "week", "day"}
        if not year_match and (
            temporal_groups.intersection(self._infer_group_by(query))
            or self._infer_ranked_dimension(query) in temporal_groups
        ):
            return None
        requested_column = next((column for column in date_columns if column.lower() in lowered), None)
        column = self.schema.resolve_date_column(requested_column)
        if column is None:
            raise QueryInterpretationError("A unique date/time field could not be resolved for the requested time filter.")
        if "year" in lowered:
            return TimeFilterPlan(column=column, year=int(year_match.group(1)) if year_match else None)
        return TimeFilterPlan(column=column)

    def _infer_filters(self, query: str) -> list[FilterPlan]:
        filters = []
        seen = set()
        for filter_item in self._extract_numeric_filters(query):
            key = (filter_item.column, filter_item.operator, filter_item.value)
            if key not in seen:
                filters.append(filter_item)
                seen.add(key)

        for column, candidates in self.schema.categorical_index().items():
            for value in candidates:
                normalized = value.casefold()
                if normalized not in query.casefold():
                    continue
                match = re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])",
                    query,
                    flags=re.IGNORECASE,
                )
                if not match:
                    continue
                if self.schema.resolve_categorical_value(value) != (column, value):
                    continue
                key = (column, "=", value)
                if key not in seen:
                    filters.append(FilterPlan(column=column, operator="=", value=value))
                    seen.add(key)
        return filters

    def _extract_numeric_filters(self, query: str) -> list[FilterPlan]:
        results = []
        for raw_column, raw_operator, raw_value in self._numeric_predicates(query):
            column = self._numeric_filter_field(raw_column)
            operator = self._numeric_operator(raw_operator)
            if column is None or operator is None:
                continue
            value = float(raw_value)
            results.append(FilterPlan(
                column=column, operator=operator,
                value=int(value) if value.is_integer() else value,
            ))
        return list({(item.column, item.operator, item.value): item for item in results}.values())

    def _numeric_operator(self, raw_operator: str) -> str | None:
        return {
            "at least": ">=", "at most": "<=", "less than or equal to": "<=",
            "greater than or equal to": ">=", "greater than": ">", "more than": ">",
            "above": ">", "over": ">", "less than": "<", "below": "<", "under": "<",
            "equal to": "=", "equals": "=", "is": "=", ">=": ">=", "<=": "<=",
            ">": ">", "<": "<", "=": "=", "==": "=", "!=": "!=",
            "is not": "!=", "not equal to": "!=", "does not equal": "!=",
            "is not equal to": "!=",
        }.get(" ".join(raw_operator.lower().split()))

    def _numeric_predicates(self, query: str) -> list[tuple[str, str, str]]:
        verbal = (
            r"(?:less|greater)\s+than\s+or\s+equal\s+to|at\s+least|at\s+most|"
            r"is\s+not\s+equal\s+to|is\s+not|not\s+equal\s+to|does\s+not\s+equal|"
            r"greater\s+than|more\s+than|above|over|less\s+than|below|under|equal\s+to|equals"
        )
        results = []
        patterns = (
            (rf"\b([a-z][a-z0-9_]*)\s+(?:is\s+)?({verbal}|is)\s+([-+]?\d+(?:\.\d+)?)", (1, 2, 3)),
            (r"\b([a-z][a-z0-9_]*)\s*(?:is\s+)?([><=!]+)\s*([-+]?\d+(?:\.\d+)?)", (1, 2, 3)),
            (rf"\b({verbal})\s+([-+]?\d+(?:\.\d+)?)\s+(?:for|of)\s+([a-z][a-z0-9_]*)", (3, 1, 2)),
        )
        for pattern, groups in patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                results.append(tuple(match.group(index) for index in groups))
        return list(dict.fromkeys(results))

    def _numeric_filter_field(self, raw_column: str) -> str | None:
        canonical = self.schema.canonical_field(raw_column)
        if canonical:
            if canonical not in self.loader.sales.columns:
                return None
            return canonical if self.schema.is_numeric_field(canonical) else None
        metric = self._infer_metric(raw_column)
        return metric if metric and self.schema.is_numeric_field(metric) else None

    def _extract_unresolved_numeric_terms(self, query: str, filters: list[FilterPlan]) -> list[str]:
        unresolved = []
        for raw_column, raw_operator, raw_value in self._numeric_predicates(query):
            column = self._numeric_filter_field(raw_column)
            operator = self._numeric_operator(raw_operator)
            if column is None:
                unresolved.append(raw_column)
            elif operator is None:
                unresolved.append(f"Unsupported numeric operator: {raw_operator}")
            elif not any(
                item.column == column and item.operator == operator
                and item.value == float(raw_value) for item in filters
            ):
                unresolved.append(f"Unresolved numeric filter: {raw_column} {raw_operator} {raw_value}")
        return list(dict.fromkeys(unresolved))

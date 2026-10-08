from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings
from .data_loader import DatasetLoader
from .explanation import build_explanation
from .feedback import FeedbackStore
from .models import QueryPlan, QueryRequest, FeedbackRequest
from .query_executor import QueryExecutor, QueryExecutionError
from .query_interpreter import QueryInterpreter, QueryInterpretationError
from .query_validator import QueryValidator, QueryValidationError
from .confidence import calculate_confidence
from .schema_manager import SchemaManager


app = FastAPI(title="Intelligent Analytics Query Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_loader() -> DatasetLoader:
    return DatasetLoader()


def get_context():
    loader = get_loader()
    schema = SchemaManager(loader)
    return loader, schema


def _target_availability_intent_probe(
    interpreter: QueryInterpreter, plan: QueryPlan, query: str,
) -> QueryPlan | None:
    """Separate missing target data from unsupported requested semantics."""
    if plan.unresolved_terms != [plan.metric]:
        return None
    availability_probe = QueryPlan.model_validate({
        **plan.model_dump(),
        "is_supported": True,
        "unresolved_terms": [],
        "semantic_errors": [],
    })
    return interpreter._validate_requested_constraints(availability_probe, query)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict:
    try:
        loader, manager = get_context()
        return loader.get_schema()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/query")
def query(request: QueryRequest) -> JSONResponse:
    plan = None
    try:
        loader, manager = get_context()
        if loader.sales.empty:
            raise QueryExecutionError(
                "Cannot calculate this query because the sales dataset is empty. "
                "Add sales rows to sales_data.csv and retry."
            )
        interpreter = QueryInterpreter(loader)
        plan = interpreter.interpret(request.query, FeedbackStore().get_relevant(request.query))
        interpreter_source = getattr(interpreter, "interpreter_source", "deterministic_fallback")
        fallback_used = interpreter_source in {"openrouter", "grok", "deterministic_fallback"}
        provider_metadata = {
            "fallback_reason": getattr(interpreter, "fallback_reason", None) if fallback_used else None,
            "provider_http_status": getattr(interpreter, "provider_http_status", None),
            "query_plan_validation": getattr(interpreter, "query_plan_validation", "NOT_ATTEMPTED"),
            "provider_model": getattr(interpreter, "provider_model", None),
            "provider_attempts": getattr(interpreter, "provider_attempts", []),
        }
        target_data_error = None
        if plan.requires_targets and plan.metric:
            if loader.targets.empty:
                target_data_error = "The requested comparison requires a non-empty targets dataset."
            elif manager.find_target_column(plan.metric) is None:
                target_data_error = f"No usable target field was found for the requested metric: {plan.metric}"
        if target_data_error and interpreter_source == "deterministic_fallback":
            intent_probe = _target_availability_intent_probe(interpreter, plan, request.query)
            if intent_probe is not None:
                if (
                    intent_probe.is_supported
                    and not intent_probe.unresolved_terms
                    and not intent_probe.semantic_errors
                ):
                    raise QueryExecutionError(target_data_error)
                # Keep the unsupported capability explicit when missing target data
                # prevented fallback interpretation from reaching semantic checks.
                plan = QueryPlan.model_validate({
                    **plan.model_dump(),
                    "is_supported": False,
                    "interpretation_confidence": 0.0,
                    "unresolved_terms": list(dict.fromkeys([*plan.unresolved_terms, *intent_probe.unresolved_terms])),
                    "semantic_errors": list(dict.fromkeys([*plan.semantic_errors, *intent_probe.semantic_errors])),
                    "ambiguity": list(dict.fromkeys([*plan.ambiguity, *intent_probe.ambiguity])),
                })
        if not plan.is_supported or plan.unresolved_terms or plan.semantic_errors:
            plan = QueryPlan.model_validate({
                **plan.model_dump(),
                "is_supported": False,
                "interpretation_confidence": 0.0,
            })
            return JSONResponse(content=jsonable_encoder({
                "query": request.query,
                "query_plan": plan.model_dump(),
                "generated_logic": None,
                "result": [],
                "confidence_score": 0.0,
                "explanation": build_explanation(plan.model_dump(), request.query, interpreter_source, fallback_used),
                "warnings": plan.ambiguity,
                "is_supported": False,
                "unresolved_terms": plan.unresolved_terms,
                "interpreter_source": interpreter_source,
                "fallback_used": fallback_used,
                **provider_metadata,
            }))

        if target_data_error:
            raise QueryExecutionError(target_data_error)

        validator = QueryValidator(manager)
        validation = validator.validate(plan)
        execution = QueryExecutor(loader).execute(validation["plan"])
        plan = execution.get("executed_plan", validation["plan"])
        checks = validation["checks"]
        confidence = calculate_confidence(
            plan.model_dump(),
            checks,
            has_results=bool(execution["result"]),
        )
        explanation = build_explanation(plan.model_dump(), request.query, interpreter_source, fallback_used)
        return JSONResponse(content=jsonable_encoder({
            "query": request.query,
            "query_plan": plan.model_dump(),
            "generated_logic": execution["generated_logic"],
            "result": execution["result"],
            "confidence_score": confidence,
            "explanation": explanation,
            "warnings": plan.ambiguity,
            "is_supported": True,
            "unresolved_terms": plan.unresolved_terms,
            "interpreter_source": interpreter_source,
            "fallback_used": fallback_used,
            **provider_metadata,
        }))
    except (QueryInterpretationError, QueryValidationError, QueryExecutionError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        if isinstance(plan, QueryPlan):
            raise HTTPException(
                status_code=422,
                detail="Cannot execute this analytical query with the available data. "
                "Check the requested fields and data types.",
            ) from exc
        raise HTTPException(
            status_code=500,
            detail="Query processing failed. Please retry or check the backend configuration.",
        ) from exc


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    try:
        result = FeedbackStore().append(request.model_dump())
        return {"status": "saved", "feedback": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to save feedback: {exc}") from exc

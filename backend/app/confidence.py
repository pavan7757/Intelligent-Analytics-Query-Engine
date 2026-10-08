from typing import Any


def calculate_confidence(
    plan: dict[str, Any],
    checks: dict[str, Any],
    has_results: bool = True,
) -> float:
    if (
        not plan.get("is_supported", True)
        or plan.get("unresolved_terms")
        or plan.get("semantic_errors")
        or any(value is False for value in checks.values())
    ):
        return 0.0

    score = 0.4
    if checks.get("metric_valid"):
        score += 0.2
    if checks.get("group_valid"):
        score += 0.1
    if checks.get("filter_valid"):
        score += 0.1
    if checks.get("time_valid"):
        score += 0.05
    if checks.get("target_available"):
        score += 0.05
    if checks.get("schema_valid"):
        score += 0.1
    interpretation_confidence = plan.get("interpretation_confidence", 0.0)
    score *= 0.5 + 0.5 * interpretation_confidence
    if plan.get("ambiguity"):
        score -= 0.15 * len(plan["ambiguity"])
    if not has_results:
        score = min(score, 0.5)
    return max(0.0, min(1.0, round(score, 3)))

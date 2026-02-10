"""Tool definitions exposed to the model in the new stateful harness."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from runtime import RuntimeActionError, SalesEpisodeRuntime


def _as_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _safe_tool_call(
    runtime: SalesEpisodeRuntime,
    tool_name: str,
    fn: Callable[[], dict[str, Any]],
) -> str:
    runtime.register_tool_call(tool_name)
    try:
        data = fn()
        return _as_json({"ok": True, **data})
    except RuntimeActionError as exc:
        runtime.record_invalid_action(str(exc))
        return _as_json({"ok": False, "error": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive boundary
        runtime.record_invalid_action(f"unexpected error: {exc}")
        return _as_json({"ok": False, "error": f"unexpected error: {exc}"})


async def crm_search_leads(
    runtime: SalesEpisodeRuntime,
    min_income: int | None = None,
    max_age: int | None = None,
    min_need: float | None = None,
    limit: int = 10,
    include_called: bool = False,
) -> str:
    """Search active leads.

    Args:
        min_income: Optional minimum annual income.
        max_age: Optional maximum age.
        min_need: Optional minimum latent need score in [0, 1].
        limit: Maximum number of leads to return.
        include_called: Include leads already contacted in the episode.
    """

    return _safe_tool_call(
        runtime,
        "crm_search_leads",
        lambda: runtime.search_leads(
            min_income=min_income,
            max_age=max_age,
            min_need=min_need,
            limit=limit,
            include_called=include_called,
        ),
    )


async def crm_get_lead(runtime: SalesEpisodeRuntime, lead_id: str) -> str:
    """Get detailed lead profile by lead ID."""

    return _safe_tool_call(runtime, "crm_get_lead", lambda: runtime.get_lead(lead_id=lead_id))


async def crm_add_note(runtime: SalesEpisodeRuntime, lead_id: str, note: str) -> str:
    """Add an internal CRM note to a lead record."""

    return _safe_tool_call(
        runtime,
        "crm_add_note",
        lambda: runtime.add_note(lead_id=lead_id, note=note),
    )


async def crm_pipeline_summary(runtime: SalesEpisodeRuntime) -> str:
    """Return pipeline-level status, metrics, and remaining time."""

    return _safe_tool_call(runtime, "crm_pipeline_summary", runtime.pipeline_summary)


async def calendar_schedule_callback(
    runtime: SalesEpisodeRuntime,
    lead_id: str,
    hours_from_now: int,
    reason: str,
) -> str:
    """Schedule a callback in N hours from the current simulated time."""

    return _safe_tool_call(
        runtime,
        "calendar_schedule_callback",
        lambda: runtime.schedule_callback(
            lead_id=lead_id,
            hours_from_now=hours_from_now,
            reason=reason,
        ),
    )


async def calendar_list_callbacks(
    runtime: SalesEpisodeRuntime,
    within_hours: int = 48,
    include_completed: bool = False,
) -> str:
    """List callback tasks due within the next N hours."""

    return _safe_tool_call(
        runtime,
        "calendar_list_callbacks",
        lambda: runtime.list_callbacks(
            within_hours=within_hours,
            include_completed=include_completed,
        ),
    )


async def calling_start_call(runtime: SalesEpisodeRuntime, lead_id: str) -> str:
    """Start a call with an active lead."""

    return _safe_tool_call(
        runtime,
        "calling_start_call",
        lambda: runtime.start_call(lead_id=lead_id),
    )


async def calling_propose_offer(
    runtime: SalesEpisodeRuntime,
    plan_type: str,
    coverage_amount: int,
    monthly_premium: float,
    next_step: str,
    term_years: int | None = None,
) -> str:
    """Propose an offer to the active buyer and receive a decision."""

    return _safe_tool_call(
        runtime,
        "calling_propose_offer",
        lambda: runtime.propose_offer(
            plan_type=plan_type,
            coverage_amount=coverage_amount,
            monthly_premium=monthly_premium,
            next_step=next_step,
            term_years=term_years,
        ),
    )


async def calling_end_call(runtime: SalesEpisodeRuntime, disposition: str = "follow_up") -> str:
    """End the active call and log a disposition."""

    return _safe_tool_call(
        runtime,
        "calling_end_call",
        lambda: runtime.end_call(disposition=disposition),
    )


async def products_list_plans(runtime: SalesEpisodeRuntime) -> str:
    """List all product plans available in the catalog."""

    return _safe_tool_call(
        runtime,
        "products_list_plans",
        lambda: {"plans": runtime.catalog.list_plans()},
    )


async def products_quote_plan(
    runtime: SalesEpisodeRuntime,
    lead_id: str,
    plan_type: str,
    coverage_amount: int,
    term_years: int | None = None,
) -> str:
    """Get a deterministic premium quote for a lead and plan."""

    return _safe_tool_call(
        runtime,
        "products_quote_plan",
        lambda: runtime.quote_plan(
            lead_id=lead_id,
            plan_type=plan_type,
            coverage_amount=coverage_amount,
            term_years=term_years,
        ),
    )


ALL_TOOLS = [
    crm_search_leads,
    crm_get_lead,
    crm_add_note,
    crm_pipeline_summary,
    calendar_schedule_callback,
    calendar_list_callbacks,
    calling_start_call,
    calling_propose_offer,
    calling_end_call,
    products_list_plans,
    products_quote_plan,
]

"""Tool definitions exposed to the model in the new stateful harness."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from models import RuntimeActionError
from runtime import SalesEpisodeRuntime

logger = logging.getLogger("salesbench")


def _as_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


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
        logger.warning("Invalid action in %s: %s", tool_name, exc)
        runtime.record_invalid_action(str(exc))
        return _as_json({"ok": False, "error": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.warning("Unexpected error in %s: %s", tool_name, exc)
        runtime.record_invalid_action(f"unexpected error: {exc}")
        return _as_json({"ok": False, "error": f"unexpected error: {exc}"})


async def _safe_tool_call_async(
    runtime: SalesEpisodeRuntime,
    tool_name: str,
    coro: Coroutine[Any, Any, dict[str, Any]],
) -> str:
    runtime.register_tool_call(tool_name)
    try:
        data = await coro
        return _as_json({"ok": True, **data})
    except RuntimeActionError as exc:
        logger.warning("Invalid action in %s: %s", tool_name, exc)
        runtime.record_invalid_action(str(exc))
        return _as_json({"ok": False, "error": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.warning("Unexpected error in %s: %s", tool_name, exc)
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
    """Search active leads in the CRM pipeline, sorted by need and budget.

    Use this to find promising leads to call. Results are ranked by latent need
    (descending), then budget (descending), then call count (ascending).

    Prerequisites: None (available any time).
    Time cost: 1 minute.
    Returns: {ok, count, leads: [{lead_id, name, status, age, annual_income, ...}], active_leads_remaining}

    Args:
        min_income: Optional minimum annual income filter.
        max_age: Optional maximum age filter.
        min_need: Optional minimum latent need score in [0, 1].
        limit: Maximum number of leads to return (default 10).
        include_called: Include leads already contacted this episode (default false).
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
    """Get the full profile for a single lead by ID.

    Use before calling or proposing to understand the lead's financials,
    risk class, trust level, price sensitivity, and notes history.

    Prerequisites: Valid lead_id from a prior search.
    Time cost: 0 minutes (instant lookup).
    Returns: {ok, lead: {lead_id, name, age, annual_income, household_size, dependents,
             risk_class, trust_level, price_sensitivity, budget_monthly, max_calls, notes, ...}}
    """

    return _safe_tool_call(runtime, "crm_get_lead", lambda: runtime.get_lead(lead_id=lead_id))


async def crm_add_note(runtime: SalesEpisodeRuntime, lead_id: str, note: str) -> str:
    """Add an internal CRM note to a lead record.

    Use to record observations, objections heard, or follow-up items for a lead.
    Notes persist across calls within the episode.

    Prerequisites: Valid lead_id.
    Time cost: 0 minutes.
    Returns: {ok, lead_id, note_count, latest_note}
    """

    return _safe_tool_call(
        runtime,
        "crm_add_note",
        lambda: runtime.add_note(lead_id=lead_id, note=note),
    )


async def crm_pipeline_summary(runtime: SalesEpisodeRuntime) -> str:
    """Return pipeline-level status counts, episode metrics, and remaining time.

    Use to check how many leads are active/converted/DNC/exhausted, review
    cumulative stats, and plan remaining time allocation.

    Prerequisites: None.
    Time cost: 0 minutes.
    Returns: {ok, status_counts, stats, time: {current_minute, max_minutes, remaining_minutes},
             active_call, done, termination_reason}
    """

    return _safe_tool_call(runtime, "crm_pipeline_summary", runtime.pipeline_summary)


async def calendar_schedule_callback(
    runtime: SalesEpisodeRuntime,
    lead_id: str,
    hours_from_now: int,
    reason: str,
) -> str:
    """Schedule a callback for a lead N hours from the current simulated time.

    Use when an immediate close is unlikely and a follow-up would improve
    conversion odds. The callback is auto-completed when you start a call
    with the same lead after the due time.

    Prerequisites: Lead must be ACTIVE and not DNC. Max 2 callbacks per lead.
    Constraints: Callback must fall within the episode time budget.
    Time cost: 1 minute.
    Returns: {ok, callback: {callback_id, lead_id, due_minute, reason, status}, scheduled_count_for_lead}
    """

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
    """List callback tasks due within the next N hours from current time.

    Use to check which leads need follow-up and prioritize your next calls.

    Prerequisites: None.
    Time cost: 0 minutes.
    Returns: {ok, count, callbacks: [{callback_id, lead_id, due_minute, reason, status}]}
    """

    return _safe_tool_call(
        runtime,
        "calendar_list_callbacks",
        lambda: runtime.list_callbacks(
            within_hours=within_hours,
            include_completed=include_completed,
        ),
    )


async def calling_start_call(runtime: SalesEpisodeRuntime, lead_id: str) -> str:
    """Start a phone call with an active lead.

    Opens a call session. After starting, speak directly to the buyer to build
    rapport and discover needs. Use quote tools to get pricing, then propose_offer
    to close. End the call explicitly when done.

    Prerequisites: No active call. Lead must be ACTIVE and not DNC.
    Constraints: Only one active call at a time. Calling a DNC lead records a violation.
    Time cost: 1 minute.
    Returns: {ok, call_id, lead: {brief profile}, message}
    """

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
    messages: list | None = None,
) -> str:
    """Propose an insurance offer to the buyer on the active call and receive their decision.

    The buyer evaluates the offer based on their profile, budget, and trust level.
    Possible decisions: ACCEPT (conversion!), REJECT (try again), HANG_UP (call ends).

    Prerequisites: Active call in progress. Use products_quote_plan first to get accurate premiums.
    Constraints: premium and coverage must be > 0, next_step must be non-empty.
    Time cost: 4 minutes.
    Returns: {ok, offer: {...}, decision: {decision, reason, score, request_dnc}, message}
    """

    return await _safe_tool_call_async(
        runtime,
        "calling_propose_offer",
        runtime.propose_offer(
            plan_type=plan_type,
            coverage_amount=coverage_amount,
            monthly_premium=monthly_premium,
            next_step=next_step,
            term_years=term_years,
            messages=messages,
        ),
    )


async def calling_end_call(runtime: SalesEpisodeRuntime, disposition: str = "follow_up") -> str:
    """End the active call and log a disposition reason.

    Always end calls explicitly to free the call slot for the next lead.
    The disposition is recorded in the call history for analytics.

    Prerequisites: Active call in progress.
    Time cost: 1 minute.
    Returns: {ok, call: {call_id, lead_id, duration_minutes, outcome, ...}, disposition}
    """

    return _safe_tool_call(
        runtime,
        "calling_end_call",
        lambda: runtime.end_call(disposition=disposition),
    )


async def products_list_plans(runtime: SalesEpisodeRuntime) -> str:
    """List all insurance product plans available in the catalog.

    Returns specifications for TERM, WHOLE, UL, and DI plans including
    age ranges, coverage limits, and term options. Use to understand what
    products you can offer before quoting.

    Prerequisites: None.
    Time cost: 0 minutes.
    Returns: {ok, plans: [{plan_type, name, min_age, max_age, min_coverage, max_coverage, description, ...}]}
    """

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
    """Get a deterministic premium quote for a specific lead and plan configuration.

    Returns the exact monthly premium the lead would pay. Always quote before
    proposing to ensure the premium matches the catalog price.

    Prerequisites: Valid lead_id and plan_type (TERM, WHOLE, UL, or DI).
    Constraints: Coverage must be within plan limits, age within plan range.
    Time cost: 1 minute.
    Returns: {ok, quote: {plan_type, age, coverage_amount, risk_class, term_years, monthly_premium,
             lead_budget_monthly, premium_to_budget_ratio}}
    """

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

"""Tool definitions exposed to the model in the new stateful harness.

Tools are thin wrappers around :class:`SalesEpisodeRuntime` methods.  All
tools are deterministic except ``calling_propose_offer`` which delegates to
the buyer policy (injected by the orchestrator via ``update_tool_args``).
Buyer LLM failures are isolated — they produce a deterministic REJECT
fallback instead of penalizing the seller with an ``invalid_action``.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

from models import BuyerDecision, DecisionResult, RuntimeActionError
from runtime import SalesEpisodeRuntime

logger = logging.getLogger("verifiers.salesbench")


def _as_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _safe_tool_call(
    runtime: SalesEpisodeRuntime,
    tool_name: str,
    fn: Callable[[], dict[str, Any]],
) -> str:
    """Execute a deterministic tool call with error handling."""
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


async def crm_search_leads(
    runtime: SalesEpisodeRuntime,
    min_income: int | None = None,
    max_age: int | None = None,
    min_need: float | None = None,
    limit: int = 10,
    include_called: bool = False,
) -> str:
    """Search active leads sorted by need and budget. Filters are optional."""

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
    """Get full profile for a lead including financials, trust, and notes."""

    return _safe_tool_call(runtime, "crm_get_lead", lambda: runtime.get_lead(lead_id=lead_id))


async def crm_add_note(runtime: SalesEpisodeRuntime, lead_id: str, note: str) -> str:
    """Add a CRM note to a lead. Notes persist within the episode."""

    return _safe_tool_call(
        runtime,
        "crm_add_note",
        lambda: runtime.add_note(lead_id=lead_id, note=note),
    )


async def crm_pipeline_summary(runtime: SalesEpisodeRuntime) -> str:
    """Get pipeline status counts, episode stats, and remaining time."""

    return _safe_tool_call(runtime, "crm_pipeline_summary", runtime.pipeline_summary)


async def calendar_schedule_callback(
    runtime: SalesEpisodeRuntime,
    lead_id: str,
    hours_from_now: int,
    reason: str,
) -> str:
    """Schedule a follow-up callback for a lead. Max 2 per lead, must fit in time budget."""

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
    """List upcoming callback tasks due within N hours."""

    return _safe_tool_call(
        runtime,
        "calendar_list_callbacks",
        lambda: runtime.list_callbacks(
            within_hours=within_hours,
            include_completed=include_completed,
        ),
    )


async def calling_start_call(runtime: SalesEpisodeRuntime, lead_id: str) -> str:
    """Start a phone call with an active lead. Only one call at a time."""

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
    buyer_policy: Any = None,
) -> str:
    """Propose an offer to the buyer. Decisions: ACCEPT, REJECT, or HANG_UP. Quote first.

    This is the only tool that involves a buyer LLM call.  The buyer policy is
    injected by the orchestrator (``update_tool_args``).  Buyer failures produce
    a deterministic REJECT — the seller is never penalized for buyer errors.
    """
    runtime.register_tool_call("calling_propose_offer")

    # --- Step 1: Record offer (deterministic — seller errors ARE invalid_actions) ---
    try:
        result = runtime.record_offer(
            plan_type=plan_type,
            coverage_amount=coverage_amount,
            monthly_premium=monthly_premium,
            next_step=next_step,
            term_years=term_years,
        )
    except RuntimeActionError as exc:
        logger.warning("Invalid action in calling_propose_offer: %s", exc)
        runtime.record_invalid_action(str(exc))
        return _as_json({"ok": False, "error": str(exc)})
    except Exception as exc:  # pragma: no cover
        logger.warning("Unexpected error in calling_propose_offer: %s", exc)
        runtime.record_invalid_action(f"unexpected error: {exc}")
        return _as_json({"ok": False, "error": f"unexpected error: {exc}"})

    # Time expired during recording
    if result.get("interrupted"):
        return _as_json({"ok": True, "decision": {"decision": "interrupted", "reason": "time expired"}})

    offer = result["offer"]
    lead = result["lead"]

    # --- Step 2: Buyer evaluation (buyer errors → fallback, NOT seller penalty) ---
    try:
        decision_or_coro = buyer_policy.evaluate_offer(
            lead=lead, offer=offer, messages=messages,
        )
        if inspect.isawaitable(decision_or_coro):
            decision: DecisionResult = await decision_or_coro
        else:
            decision = decision_or_coro
    except Exception as exc:
        logger.warning("Buyer policy failed, using deterministic REJECT: %s", exc)
        decision = DecisionResult(
            decision=BuyerDecision.REJECT,
            reason="I need some time to think about this.",
            score=0.40,
            request_dnc=False,
        )

    # --- Step 3: Apply decision to state (deterministic) ---
    response = runtime.apply_buyer_decision(
        decision=decision.decision,
        reason=decision.reason,
        request_dnc=decision.request_dnc,
    )

    return _as_json({"ok": True, **response})


async def calling_end_call(runtime: SalesEpisodeRuntime, disposition: str = "follow_up") -> str:
    """End the active call with a disposition reason."""

    return _safe_tool_call(
        runtime,
        "calling_end_call",
        lambda: runtime.end_call(disposition=disposition),
    )


async def products_list_plans(runtime: SalesEpisodeRuntime) -> str:
    """List available insurance plans (TERM, WHOLE, UL, DI) with coverage limits."""

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
    """Get a premium quote for a lead and plan. Always quote before proposing."""

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

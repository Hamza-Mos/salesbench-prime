"""Rubric reward and metric functions for the new harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from models import BuyerDecision
from runtime import SalesEpisodeRuntime


def _get_runtime(state: dict[str, Any]) -> SalesEpisodeRuntime | None:
    runtime = state.get("runtime")
    if runtime is None:
        return None
    if not isinstance(runtime, SalesEpisodeRuntime):
        return None
    return runtime


# ---------------------------------------------------------------------------
# Reward / penalty functions (explicit, kept as named functions)
# ---------------------------------------------------------------------------


async def reward_revenue_mrr(state: dict[str, Any]) -> float:
    """Primary objective: normalized converted monthly recurring revenue."""

    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return runtime.stats.revenue_mrr / max(1.0, runtime.max_achievable_mrr)


async def reward_conversion_rate(state: dict[str, Any]) -> float:
    """Secondary reward: conversion rate over total leads."""

    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return runtime.stats.conversions / max(1, runtime.config.num_leads)


async def reward_efficiency(state: dict[str, Any]) -> float:
    """Reward efficiency as conversions per call started."""

    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return runtime.stats.conversions / max(1, runtime.stats.calls_started)


async def penalty_dnc_violations(state: dict[str, Any]) -> float:
    """Compliance penalty for do-not-call violations."""

    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.dnc_violations)


async def penalty_invalid_actions(state: dict[str, Any]) -> float:
    """Penalty for invalid tool calls or malformed actions."""

    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return 0.1 * float(runtime.stats.invalid_actions)


async def reward_episode_completion(state: dict[str, Any]) -> float:
    """Bonus for completing the episode (not truncated by infrastructure)."""

    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return 1.0 if runtime.done else 0.0


async def reward_budget_utilization(state: dict[str, Any]) -> float:
    """Reward for capturing a high fraction of converted leads' budgets.

    For each accepted offer, computes accepted_premium / lead_budget.
    Normalized by total leads so the signal scales with episode scope.
    """

    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    total_ratio = 0.0
    for session in runtime.call_history:
        if session.outcome == BuyerDecision.ACCEPT and session.offers:
            lead = runtime.leads.get(session.lead_id)
            if lead and lead.budget_monthly > 0:
                accepted_premium = session.offers[-1].monthly_premium
                total_ratio += min(1.0, accepted_premium / lead.budget_monthly)
    return total_ratio / max(1, runtime.config.num_leads)


# ---------------------------------------------------------------------------
# Metric functions (data-driven generation)
# ---------------------------------------------------------------------------

_METRIC_SPECS: list[tuple[str, Callable[[SalesEpisodeRuntime], float]]] = [
    ("metric_raw_revenue_mrr", lambda rt: rt.stats.revenue_mrr),
    ("metric_conversions", lambda rt: float(rt.stats.conversions)),
    ("metric_dnc_violations", lambda rt: float(rt.stats.dnc_violations)),
    ("metric_invalid_actions", lambda rt: float(rt.stats.invalid_actions)),
    ("metric_time_utilization", lambda rt: rt.current_minute / max(1, rt.config.max_minutes)),
    ("metric_calls_started", lambda rt: float(rt.stats.calls_started)),
    ("metric_episode_done", lambda rt: 1.0 if rt.done else 0.0),
    ("metric_calls_completed", lambda rt: float(rt.stats.calls_completed)),
    ("metric_offers_proposed", lambda rt: float(rt.stats.offers_proposed)),
    ("metric_offers_accepted", lambda rt: float(rt.stats.offers_accepted)),
    ("metric_offers_rejected", lambda rt: float(rt.stats.rejected_offers)),
    ("metric_hang_ups", lambda rt: float(rt.stats.hang_ups)),
    ("metric_avg_revenue_per_call", lambda rt: rt.stats.revenue_mrr / max(1, rt.stats.calls_started)),
    ("metric_leads_contacted", lambda rt: float(rt.stats.leads_contacted)),
    ("metric_leads_remaining", lambda rt: float(rt.num_active_leads)),
    ("metric_callbacks_scheduled", lambda rt: float(rt.stats.callbacks_scheduled)),
    ("metric_callbacks_completed", lambda rt: float(rt.stats.callbacks_completed)),
    ("metric_time_used_minutes", lambda rt: float(rt.current_minute)),
    ("metric_minutes_per_conversion", lambda rt: (
        float(rt.current_minute) / rt.stats.conversions if rt.stats.conversions > 0 else 0.0
    )),
    ("metric_max_possible_mrr", lambda rt: rt.max_achievable_mrr),
    ("metric_revenue_capture_ratio", lambda rt: (
        rt.stats.revenue_mrr / rt.max_achievable_mrr if rt.max_achievable_mrr > 0 else 0.0
    )),
    ("metric_budget_utilization_raw", lambda rt: _budget_util_raw(rt)),
]


def _budget_util_raw(rt: SalesEpisodeRuntime) -> float:
    """Average premium/budget ratio across accepted offers (for observability)."""
    total = 0.0
    count = 0
    for session in rt.call_history:
        if session.outcome == BuyerDecision.ACCEPT and session.offers:
            lead = rt.leads.get(session.lead_id)
            if lead and lead.budget_monthly > 0:
                total += min(1.0, session.offers[-1].monthly_premium / lead.budget_monthly)
                count += 1
    return total / max(1, count) if count > 0 else 0.0


def _make_metric(
    name: str, extractor: Callable[[SalesEpisodeRuntime], float]
) -> Callable[[dict[str, Any]], Any]:
    async def fn(state: dict[str, Any]) -> float:
        rt = _get_runtime(state)
        return extractor(rt) if rt else 0.0

    fn.__name__ = name
    fn.__qualname__ = name
    return fn


_GENERATED_METRICS = [_make_metric(n, f) for n, f in _METRIC_SPECS]


async def metric_context_summary_count(state: dict[str, Any]) -> float:
    """Number of times context summarization was triggered during this episode."""
    return float(state.get("_context_summary_count", 0))


# ---------------------------------------------------------------------------
# Buyer LLM observability metrics
# ---------------------------------------------------------------------------

def _get_llm_policy(state: dict[str, Any]):
    """Return the LLMBuyerPolicy from runtime, or None."""
    from policy import LLMBuyerPolicy
    rt = _get_runtime(state)
    if rt is None:
        return None
    return rt.policy if isinstance(rt.policy, LLMBuyerPolicy) else None


async def metric_buyer_llm_call_count(state: dict[str, Any]) -> float:
    """Total buyer LLM API calls this episode."""
    p = _get_llm_policy(state)
    return float(p.call_count) if p else 0.0


async def metric_buyer_llm_timeout_count(state: dict[str, Any]) -> float:
    """Buyer LLM calls that hit the hard timeout."""
    p = _get_llm_policy(state)
    return float(p.timeout_count) if p else 0.0


async def metric_buyer_llm_slow_count(state: dict[str, Any]) -> float:
    """Buyer LLM calls that took >30s."""
    p = _get_llm_policy(state)
    return float(p.slow_call_count) if p else 0.0


async def metric_buyer_llm_avg_latency(state: dict[str, Any]) -> float:
    """Average latency (seconds) of buyer LLM calls."""
    p = _get_llm_policy(state)
    if p and p.call_count > 0:
        return p.total_latency / p.call_count
    return 0.0


async def metric_buyer_llm_max_latency(state: dict[str, Any]) -> float:
    """Max latency (seconds) of any single buyer LLM call."""
    p = _get_llm_policy(state)
    return p.max_latency if p else 0.0


_BUYER_LLM_METRICS = [
    metric_buyer_llm_call_count,
    metric_buyer_llm_timeout_count,
    metric_buyer_llm_slow_count,
    metric_buyer_llm_avg_latency,
    metric_buyer_llm_max_latency,
]

# ---------------------------------------------------------------------------
# Assembled rubric
# ---------------------------------------------------------------------------

_REWARD_FUNCS = [
    reward_revenue_mrr,
    reward_conversion_rate,
    reward_efficiency,
    penalty_dnc_violations,
    penalty_invalid_actions,
    reward_episode_completion,
    reward_budget_utilization,
]

_REWARD_WEIGHTS = [
    1.00,   # reward_revenue_mrr        — primary objective: maximize normalized revenue
    0.15,   # reward_conversion_rate     — raised: 3 leads harder, need conversion signal
    0.00,   # reward_efficiency          — disabled
    -0.30,  # penalty_dnc_violations     — hard compliance
    -0.05,  # penalty_invalid_actions    — keep low; unavoidable schema errors add noise
    0.10,   # reward_episode_completion  — raised: 3 leads = longer episodes, completion may drop
    0.35,   # reward_budget_utilization  — reduced: conversion matters more at higher difficulty
]

_ERROR_TYPE_MAP = {
    "BadRequestError": 1.0,
    "APITimeoutError": 2.0,
    "APIConnectionError": 3.0,
    "RateLimitError": 4.0,
    "InternalServerError": 5.0,
    "APIStatusError": 6.0,
    "APIError": 7.0,
}

_error_logger = __import__("logging").getLogger("verifiers.salesbench.errors")


async def metric_error_type(state: dict[str, Any]) -> float:
    """Encode the original error type as a float for metric tracking.

    0=no error, 1=BadRequest, 2=Timeout, 3=Connection, 4=RateLimit,
    5=InternalServer, 6=APIStatus, 7=APIError, 9=unknown.
    Also logs the full cause so it appears in ``prime rl logs``.
    """
    err = state.get("error")
    if err is None:
        return 0.0
    cause = getattr(err, "__cause__", None)
    if cause is not None:
        cause_name = type(cause).__name__
        _error_logger.error("ModelError cause: %s: %s", cause_name, cause)
        return _ERROR_TYPE_MAP.get(cause_name, 9.0)
    _error_logger.error("ModelError (no __cause__): %s", err)
    return 8.0


_STATE_METRICS = [metric_context_summary_count, metric_error_type]

_ALL_METRICS = _GENERATED_METRICS + _STATE_METRICS + _BUYER_LLM_METRICS
RUBRIC_FUNCS = _REWARD_FUNCS + _ALL_METRICS
RUBRIC_WEIGHTS = _REWARD_WEIGHTS + [0.00] * len(_ALL_METRICS)

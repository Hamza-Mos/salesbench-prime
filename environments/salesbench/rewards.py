"""Rubric reward and metric functions for the new harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
    return -float(runtime.stats.dnc_violations)


async def penalty_invalid_actions(state: dict[str, Any]) -> float:
    """Penalty for invalid tool calls or malformed actions."""

    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return -0.1 * float(runtime.stats.invalid_actions)


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
]


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

# ---------------------------------------------------------------------------
# Assembled rubric
# ---------------------------------------------------------------------------

_REWARD_FUNCS = [
    reward_revenue_mrr,
    reward_conversion_rate,
    reward_efficiency,
    penalty_dnc_violations,
    penalty_invalid_actions,
]

_REWARD_WEIGHTS = [
    1.00,  # reward_revenue_mrr
    0.00,  # reward_conversion_rate
    0.00,  # reward_efficiency
    0.00,  # penalty_dnc_violations
    0.00,  # penalty_invalid_actions
]

RUBRIC_FUNCS = _REWARD_FUNCS + _GENERATED_METRICS
RUBRIC_WEIGHTS = _REWARD_WEIGHTS + [0.00] * len(_GENERATED_METRICS)

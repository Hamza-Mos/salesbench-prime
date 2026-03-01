"""Rubric reward and metric functions for the new harness."""

from __future__ import annotations

from typing import Any

from runtime import SalesEpisodeRuntime


def _get_runtime(state: dict[str, Any]) -> SalesEpisodeRuntime | None:
    runtime = state.get("runtime")
    if runtime is None:
        return None
    if not isinstance(runtime, SalesEpisodeRuntime):
        return None
    return runtime


async def reward_revenue_mrr(state: dict[str, Any]) -> float:
    """Primary objective: total converted monthly recurring revenue (raw)."""

    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return runtime.stats.revenue_mrr


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


async def metric_revenue_mrr(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return runtime.stats.revenue_mrr


async def metric_conversions(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.conversions)


async def metric_dnc_violations(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.dnc_violations)


async def metric_invalid_actions(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.invalid_actions)


async def metric_time_utilization(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return runtime.current_minute / max(1, runtime.config.max_minutes)


async def metric_calls_started(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.calls_started)


async def metric_episode_done(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return 1.0 if runtime.done else 0.0


async def metric_calls_completed(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.calls_completed)


async def metric_offers_proposed(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.offers_proposed)


async def metric_offers_accepted(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.offers_accepted)


async def metric_offers_rejected(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.rejected_offers)


async def metric_hang_ups(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.hang_ups)


async def metric_avg_revenue_per_call(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return runtime.stats.revenue_mrr / max(1, runtime.stats.calls_started)


async def metric_leads_contacted(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.leads_contacted)


async def metric_leads_remaining(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.num_active_leads)


async def metric_callbacks_scheduled(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.callbacks_scheduled)


async def metric_callbacks_completed(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.stats.callbacks_completed)


async def metric_time_used_minutes(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    return float(runtime.current_minute)


async def metric_minutes_per_conversion(state: dict[str, Any]) -> float:
    runtime = _get_runtime(state)
    if runtime is None:
        return 0.0
    if runtime.stats.conversions == 0:
        return 0.0
    return float(runtime.current_minute) / runtime.stats.conversions


RUBRIC_FUNCS = [
    reward_revenue_mrr,
    reward_conversion_rate,
    reward_efficiency,
    penalty_dnc_violations,
    penalty_invalid_actions,
    metric_revenue_mrr,
    metric_conversions,
    metric_dnc_violations,
    metric_invalid_actions,
    metric_time_utilization,
    metric_calls_started,
    metric_episode_done,
    metric_calls_completed,
    metric_offers_proposed,
    metric_offers_accepted,
    metric_offers_rejected,
    metric_hang_ups,
    metric_avg_revenue_per_call,
    metric_leads_contacted,
    metric_leads_remaining,
    metric_callbacks_scheduled,
    metric_callbacks_completed,
    metric_time_used_minutes,
    metric_minutes_per_conversion,
]

RUBRIC_WEIGHTS = [
    1.00,  # reward_revenue_mrr
    0.00,  # reward_conversion_rate
    0.00,  # reward_efficiency
    0.00,  # penalty_dnc_violations
    0.00,  # penalty_invalid_actions
    0.00,  # metric_revenue_mrr
    0.00,  # metric_conversions
    0.00,  # metric_dnc_violations
    0.00,  # metric_invalid_actions
    0.00,  # metric_time_utilization
    0.00,  # metric_calls_started
    0.00,  # metric_episode_done
    0.00,  # metric_calls_completed
    0.00,  # metric_offers_proposed
    0.00,  # metric_offers_accepted
    0.00,  # metric_offers_rejected
    0.00,  # metric_hang_ups
    0.00,  # metric_avg_revenue_per_call
    0.00,  # metric_leads_contacted
    0.00,  # metric_leads_remaining
    0.00,  # metric_callbacks_scheduled
    0.00,  # metric_callbacks_completed
    0.00,  # metric_time_used_minutes
    0.00,  # metric_minutes_per_conversion
]

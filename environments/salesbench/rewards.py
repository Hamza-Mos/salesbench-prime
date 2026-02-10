"""Rubric reward and metric functions for the new harness."""

from __future__ import annotations

from typing import Any


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
]

RUBRIC_WEIGHTS = [
    1.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
]

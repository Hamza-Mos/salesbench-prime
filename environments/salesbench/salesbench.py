from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

import verifiers as vf

from config import EpisodeConfig
from dataset import build_salesbench_dataset
from rewards import RUBRIC_FUNCS, RUBRIC_WEIGHTS
from runtime import SalesEpisodeRuntime
from tools import ALL_TOOLS

logger = logging.getLogger("salesbench")

SYSTEM_PROMPT = """
You are a sales agent operating a structured insurance pipeline.

Core objective:
- Maximize converted monthly premium while staying compliant and efficient.

Execution rules:
- Use tools every turn. Do not hallucinate lead data or prices.
- Verify lead profile before making a proposal.
- Start exactly one active call at a time.
- End calls explicitly when done.
- Avoid do-not-call violations and invalid actions.

Strategy hints:
- Prioritize high-need leads with adequate budgets.
- Use quote tools before proposing premiums.
- Use callbacks selectively when immediate close is unlikely.
""".strip()


class SalesBenchPrimeRLEnv(vf.StatefulToolEnv):
    """Stateful, tool-use sales environment designed for RL post-training."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        eval_dataset: Dataset | None,
        default_seed: int,
        default_num_leads: int,
        default_work_days: int,
        default_hours_per_day: int,
        max_turns: int,
        **kwargs: Any,
    ) -> None:
        self.default_seed = default_seed
        self.default_num_leads = default_num_leads
        self.default_work_days = default_work_days
        self.default_hours_per_day = default_hours_per_day

        rubric = vf.Rubric(funcs=RUBRIC_FUNCS, weights=RUBRIC_WEIGHTS)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            tools=[],
            max_turns=max_turns,
            rubric=rubric,
            system_prompt=SYSTEM_PROMPT,
            **kwargs,
        )

        for tool in ALL_TOOLS:
            self.add_tool(tool, args_to_skip=["runtime"])

    async def setup_state(self, state: vf.State) -> vf.State:
        input_data = state.get("input", {})
        if not isinstance(input_data, dict):
            input_data = {}

        config = EpisodeConfig.from_input(
            input_data,
            default_seed=self.default_seed,
            default_num_leads=self.default_num_leads,
            default_work_days=self.default_work_days,
            default_hours_per_day=self.default_hours_per_day,
        )
        runtime = SalesEpisodeRuntime(config=config)

        state["runtime"] = runtime
        state["episode_config"] = config.to_dict()
        state["episode_seed"] = config.seed
        logger.info(
            "setup_state: seed=%d leads=%d budget=%dm",
            config.seed,
            config.num_leads,
            config.max_minutes,
        )

        briefing = runtime.render_briefing()
        prompt = state.get("prompt")
        if isinstance(prompt, list) and len(prompt) > 0:
            last_message = prompt[-1]
            if isinstance(last_message, dict) and last_message.get("role") == "user":
                existing = str(last_message.get("content", "")).strip()
                last_message["content"] = f"{existing}\n\n{briefing}" if existing else briefing
            else:
                prompt.append({"role": "user", "content": briefing})
        else:
            state["prompt"] = [{"role": "user", "content": briefing}]

        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        runtime = state.get("runtime")
        if runtime is None:
            raise RuntimeError("runtime missing from state")
        if not isinstance(tool_args, dict):
            raise RuntimeError(f"tool_args must be dict, got {type(tool_args)!r}")

        updated = dict(tool_args)
        updated["runtime"] = runtime
        return updated

    @vf.stop
    async def episode_done(self, state: vf.State) -> bool:
        runtime = state.get("runtime")
        if runtime is None:
            return True
        if not isinstance(runtime, SalesEpisodeRuntime):
            return True
        if runtime.done and "episode_summary" not in state:
            summary = runtime.export_summary()
            state["episode_summary"] = summary
            logger.info(
                "Episode complete: reason=%s conversions=%d mrr=%.2f calls=%d leads_contacted=%d",
                summary.get("termination_reason"),
                summary["stats"]["conversions"],
                summary["stats"]["revenue_mrr"],
                summary["stats"]["calls_started"],
                summary["stats"]["leads_contacted"],
            )
        return runtime.done


def load_environment(
    split: str = "train",
    num_examples: int = 256,
    eval_num_examples: int = 64,
    base_seed: int = 42,
    seed: int | None = None,
    num_leads: int = 100,
    work_days: int = 10,
    hours_per_day: int = 8,
    # Large guardrail to emulate legacy safety_max_turns=None default.
    max_turns: int = 10_000,
    max_examples: int = -1,
) -> vf.Environment:
    """Entry-point for Prime verifiers/Prime Lab.

    Args:
        split: Dataset split to load (train/eval/test).
        num_examples: Number of examples for the main dataset.
        eval_num_examples: Number of examples for eval dataset.
        base_seed: Base seed for deterministic scenario generation.
        seed: Optional alias for `base_seed` used by Prime eval args.
        num_leads: Baseline leads per episode (curriculum scales around this).
        work_days: Simulated work days in an episode.
        hours_per_day: Simulated working hours per day.
        max_turns: Safety cap on rollout turns.
        max_examples: Optional cap after dataset construction.
    """

    resolved_seed = base_seed if seed is None else seed

    dataset = build_salesbench_dataset(
        split=split,
        num_examples=num_examples,
        base_seed=resolved_seed,
        base_num_leads=num_leads,
        work_days=work_days,
        hours_per_day=hours_per_day,
    )

    eval_split = "eval" if split == "train" else split
    eval_dataset = build_salesbench_dataset(
        split=eval_split,
        num_examples=eval_num_examples,
        base_seed=resolved_seed,
        base_num_leads=num_leads,
        work_days=work_days,
        hours_per_day=hours_per_day,
    )

    if max_examples > 0:
        capped = min(max_examples, len(dataset))
        dataset = dataset.select(range(capped))

    return SalesBenchPrimeRLEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        default_seed=resolved_seed,
        default_num_leads=num_leads,
        default_work_days=work_days,
        default_hours_per_day=hours_per_day,
        max_turns=max_turns,
    )

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

logger = logging.getLogger("verifiers.salesbench")

SYSTEM_PROMPT = """
You are a sales agent operating a structured insurance pipeline.

Core objective:
- Maximize converted monthly premium by CLOSING DEALS with the calling_propose_offer tool.
- Revenue is ONLY earned when a buyer ACCEPTs an offer via calling_propose_offer. Follow-ups, emails, and callbacks generate zero revenue.

Critical workflow — every call must follow this pattern:
1. calling_start_call — open the call.
2. One brief conversational message to greet and discover needs (1-2 sentences).
3. products_quote_plan — get ONE quote that fits the lead's budget.
4. calling_propose_offer — propose the offer. This is the ONLY way to convert.
5. If REJECT: try ONE revised offer, then end the call.
6. calling_end_call — close the call and move to the next lead.

Execution rules:
- You MUST call calling_propose_offer on every call before ending it. Never end a call without proposing.
- Keep conversations SHORT. Do not write long pitches, tables, or detailed breakdowns — the buyer decides based on the offer tool, not your prose.
- Get ONE quote per lead, propose it, and move on. Do not pull multiple quotes for the same lead.
- Do not promise emails, PDF packets, or materials — they do not exist in this system.
- Do not hallucinate prices — use products_quote_plan first, then pass the exact premium to calling_propose_offer.
- Start exactly one active call at a time. End calls explicitly.
- Avoid do-not-call violations and invalid actions.

Strategy:
- Prioritize warm/hot leads with high need scores and adequate budgets.
- Match coverage to the lead's budget (premium_to_budget_ratio between 0.5 and 1.0 is ideal).
- Move quickly through leads — more calls with proposals beats fewer calls with long conversations.
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
        default_total_hours: int,
        max_turns: int,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.default_seed = default_seed
        self.default_num_leads = default_num_leads
        self.default_total_hours = default_total_hours

        rubric = vf.Rubric(funcs=RUBRIC_FUNCS, weights=RUBRIC_WEIGHTS)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            tools=[],
            max_turns=max_turns,
            rubric=rubric,
            system_prompt=system_prompt or SYSTEM_PROMPT,
            **kwargs,
        )

        for tool in ALL_TOOLS:
            self.add_tool(tool, args_to_skip=["runtime", "messages"])

    async def setup_state(self, state: vf.State) -> vf.State:
        input_data = state.get("input", {})
        if not isinstance(input_data, dict):
            input_data = {}

        config = EpisodeConfig.from_input(
            input_data,
            default_seed=self.default_seed,
            default_num_leads=self.default_num_leads,
            default_total_hours=self.default_total_hours,
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
        if tool_name == "calling_propose_offer":
            updated["messages"] = messages
        return updated

    async def env_response(self, messages, state, **kwargs):
        last_msg = messages[-1] if messages else {}
        has_tools = "tool_calls" in last_msg and last_msg.get("tool_calls") is not None

        # Sanitize null/empty tool call arguments (Qwen models send None
        # instead of "{}" for zero-arg tools, which crashes json.loads).
        if has_tools:
            for tc in last_msg.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("arguments") is None or fn.get("arguments") == "":
                    fn["arguments"] = "{}"

        if has_tools:
            tool_results = await super().env_response(messages, state, **kwargs)

            runtime = state.get("runtime")
            if runtime:
                # Inject buyer spoken response after propose_offer decisions
                if runtime._pending_buyer_speech:
                    speech = runtime._pending_buyer_speech
                    runtime._pending_buyer_speech = None
                    tool_results = list(tool_results) + [
                        {"role": "user", "content": speech}
                    ]
                # Also generate buyer conversation response when agent sent
                # text alongside tool calls during an active call (e.g. the
                # seller greets the buyer while simultaneously quoting).
                elif runtime.active_call and not runtime.done:
                    agent_text = str(last_msg.get("content", "")).strip()
                    if agent_text:
                        buyer_reply = await runtime.conversation_turn(
                            agent_text=agent_text, messages=messages
                        )
                        if buyer_reply:
                            lead = runtime.leads.get(runtime.active_call.lead_id)
                            name = lead.full_name if lead else "Buyer"
                            tool_results = list(tool_results) + [
                                {"role": "user", "content": f"[{name} (buyer)]: {buyer_reply}"}
                            ]

            return tool_results

        # Plain text from agent — inject buyer response if in a call
        runtime = state.get("runtime")
        if runtime and not runtime.done and runtime.active_call:
            agent_text = str(last_msg.get("content", "")).strip()
            if agent_text:
                buyer_reply = await runtime.conversation_turn(
                    agent_text=agent_text, messages=messages
                )
                if buyer_reply:
                    lead = runtime.leads.get(runtime.active_call.lead_id)
                    name = lead.full_name if lead else "Buyer"
                    return [{"role": "user", "content": f"[{name} (buyer)]: {buyer_reply}"}]
        return []

    @vf.stop
    async def no_tools_called(self, state):
        if len(state["trajectory"]) == 0:
            return False
        last_message = state["trajectory"][-1]["completion"][-1]
        is_assistant = last_message["role"] == "assistant"
        no_tool_calls = (
            "tool_calls" not in last_message or last_message["tool_calls"] is None
        )
        if is_assistant and no_tool_calls:
            runtime = state.get("runtime")
            if runtime and runtime.active_call and not runtime.done:
                return False  # keep conversation going
            return True
        return False

    async def render_completion(self, state):
        """Override to inject episode summary as the final conversation message."""
        if state.get("final_env_response") is None:
            runtime = state.get("runtime")
            if runtime:
                summary = runtime.export_summary()
                if not runtime.termination_reason:
                    summary["termination_reason"] = state.get("stop_condition", "completed")
                state["episode_summary"] = summary
                state["final_env_response"] = [
                    {"role": "user", "content": self._format_episode_summary(summary)}
                ]
        await super().render_completion(state)

    @staticmethod
    def _format_episode_summary(summary: dict) -> str:
        stats = summary.get("stats", {})
        funnel = summary.get("funnel", {})
        time_used = summary.get("time_used_minutes", 0)
        time_budget = summary.get("time_budget_minutes", 0)
        reason = summary.get("termination_reason", "unknown")

        mrr = stats.get("revenue_mrr", 0)
        conversions = stats.get("conversions", 0)
        total_leads = funnel.get("total_leads", 0)
        calls = stats.get("calls_started", 0)
        efficiency = (conversions / calls * 100) if calls > 0 else 0
        mins_per_conv = (time_used / conversions) if conversions > 0 else 0

        lines = [
            "── Episode Summary ──",
            f"Result: {reason} ({time_used}/{time_budget} min used)",
            "",
            f"Revenue MRR: ${mrr:.2f}",
            f"Conversions: {conversions}/{total_leads} leads ({efficiency:.0f}% call efficiency)",
            f"Minutes/conversion: {mins_per_conv:.1f}" if conversions > 0 else "Minutes/conversion: N/A",
            "",
            f"Calls: {stats.get('calls_started', 0)} started, {stats.get('calls_completed', 0)} completed",
            f"Offers: {stats.get('offers_proposed', 0)} proposed, "
            f"{stats.get('offers_accepted', 0)} accepted, "
            f"{stats.get('rejected_offers', 0)} rejected",
            f"Hang-ups: {stats.get('hang_ups', 0)}",
            "",
            f"Leads: {funnel.get('leads_contacted', 0)} contacted, "
            f"{funnel.get('leads_converted', 0)} converted, "
            f"{funnel.get('leads_remaining', 0)} remaining",
            f"DNC violations: {stats.get('dnc_violations', 0)}",
            f"Invalid actions: {stats.get('invalid_actions', 0)}",
        ]
        return "\n".join(lines)

    @vf.stop
    async def episode_done(self, state: vf.State) -> bool:
        runtime = state.get("runtime")
        if runtime is None:
            return True
        if not isinstance(runtime, SalesEpisodeRuntime):
            return True
        # Always update the summary so it captures the latest state,
        # even when the episode ends via no_tools_called rather than runtime.done.
        summary = runtime.export_summary()
        state["episode_summary"] = summary
        if runtime.done:
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
    total_hours: int = 80,
    buyer_policy: str = "llm",
    buyer_model: str = "gpt-5-mini",
    buyer_base_url: str = "https://api.openai.com/v1",
    buyer_api_key_var: str = "OPENAI_API_KEY",
    # Large guardrail to emulate legacy safety_max_turns=None default.
    max_turns: int = 10_000,
    max_examples: int = -1,
    system_prompt: str | None = None,
) -> vf.Environment:
    """Entry-point for Prime verifiers/Prime Lab.

    Args:
        split: Dataset split to load (train/eval/test).
        num_examples: Number of examples for the main dataset.
        eval_num_examples: Number of examples for eval dataset.
        base_seed: Base seed for deterministic scenario generation.
        seed: Optional alias for `base_seed` used by Prime eval args.
        num_leads: Baseline leads per episode (curriculum scales around this).
        total_hours: Total simulated hours for the episode.
        buyer_policy: Buyer model type — "llm" (default) or "rule_based".
        buyer_model: LLM model identifier for the buyer (used when buyer_policy="llm").
        buyer_base_url: Base URL for the buyer LLM API.
        buyer_api_key_var: Environment variable name for the buyer LLM API key.
        max_turns: Safety cap on rollout turns.
        max_examples: Optional cap after dataset construction.
    """

    resolved_seed = base_seed if seed is None else seed

    # Load .env / secrets.env so API keys are available without manual export.
    # When installed from a wheel, __file__ is in site-packages, so also
    # search from CWD (where `prime eval run` is executed).
    from pathlib import Path

    from dotenv import load_dotenv

    for root in (Path.cwd(), Path(__file__).resolve().parent.parent.parent):
        load_dotenv(root / "secrets.env", override=False)
        load_dotenv(root / ".env", override=False)

    if buyer_policy == "llm":
        vf.ensure_keys([buyer_api_key_var])

    dataset = build_salesbench_dataset(
        split=split,
        num_examples=num_examples,
        base_seed=resolved_seed,
        base_num_leads=num_leads,
        total_hours=total_hours,
        buyer_policy=buyer_policy,
        buyer_model=buyer_model,
        buyer_base_url=buyer_base_url,
        buyer_api_key_var=buyer_api_key_var,
    )

    eval_split = "eval" if split == "train" else split
    eval_dataset = build_salesbench_dataset(
        split=eval_split,
        num_examples=eval_num_examples,
        base_seed=resolved_seed,
        base_num_leads=num_leads,
        total_hours=total_hours,
        buyer_policy=buyer_policy,
        buyer_model=buyer_model,
        buyer_base_url=buyer_base_url,
        buyer_api_key_var=buyer_api_key_var,
    )

    if max_examples > 0:
        capped = min(max_examples, len(dataset))
        dataset = dataset.select(range(capped))

    return SalesBenchPrimeRLEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        default_seed=resolved_seed,
        default_num_leads=num_leads,
        default_total_hours=total_hours,
        max_turns=max_turns,
        system_prompt=system_prompt,
    )

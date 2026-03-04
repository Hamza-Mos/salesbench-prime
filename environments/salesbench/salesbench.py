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
- Be concise. No narration between tool calls.

Tool time costs (min): search=1, start_call=1, quote=1, propose=4, end_call=1, schedule_cb=1. Others=0.
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
        context_rewrite_threshold: float = 0.80,
        context_keep_recent: int = 10,
        **kwargs: Any,
    ) -> None:
        self.default_seed = default_seed
        self.default_num_leads = default_num_leads
        self.default_total_hours = default_total_hours
        self.context_rewrite_threshold = context_rewrite_threshold
        self.context_keep_recent = context_keep_recent

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

    # ------------------------------------------------------------------
    # Context summarization — compress older messages when nearing
    # max_seq_len to prevent truncation during training.
    # ------------------------------------------------------------------

    def _should_summarize(self, state: Any) -> bool:
        """Return True when the last turn's prompt exceeded the threshold."""
        if not self.max_seq_len:
            return False

        trajectory = state.get("trajectory", [])
        if not trajectory:
            return False

        last_step = trajectory[-1]
        response = last_step.get("response")
        if response is None:
            return False

        usage = getattr(response, "usage", None)
        if usage is None:
            return False

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        threshold = int(self.max_seq_len * self.context_rewrite_threshold)
        return prompt_tokens >= threshold

    def _apply_context_summary(
        self, messages: list, state: Any
    ) -> list:
        """Replace older messages with a deterministic context summary.

        Keeps messages[0:2] (system + briefing), inserts a summary from
        runtime state, then keeps the last ``context_keep_recent`` messages.
        """
        runtime = state.get("runtime")
        if runtime is None:
            return messages

        prefix_count = 2  # system prompt + briefing
        keep_recent = min(self.context_keep_recent, len(messages) - prefix_count)

        if keep_recent <= 0 or len(messages) <= prefix_count + keep_recent:
            return messages  # nothing to compress

        prefix = messages[:prefix_count]
        recent = messages[-keep_recent:]

        summary_text = runtime.render_context_summary()
        summary_msg = {"role": "user", "content": summary_text}

        count = state.get("_context_summary_count", 0) + 1
        state["_context_summary_count"] = count

        logger.info(
            "Context summarized (#%d): %d msgs → %d "
            "(%d prefix + 1 summary + %d recent)",
            count,
            len(messages),
            prefix_count + 1 + keep_recent,
            prefix_count,
            keep_recent,
        )

        return prefix + [summary_msg] + recent

    async def get_prompt_messages(self, state):
        """Override to compress context when nearing max_seq_len.

        The base class builds messages and calls env_response (which runs
        tool execution and buyer LLM) with full uncompressed context.
        After that, we optionally summarize older messages so the training
        model's next prompt stays within budget.
        """
        all_messages = await super().get_prompt_messages(state)

        if self._should_summarize(state):
            all_messages = self._apply_context_summary(all_messages, state)

        return all_messages

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
        """Store episode summary dict for reward computation without injecting text into the conversation."""
        if state.get("episode_summary") is None:
            runtime = state.get("runtime")
            if runtime:
                summary = runtime.export_summary()
                if not runtime.termination_reason:
                    summary["termination_reason"] = state.get("stop_condition", "completed")
                state["episode_summary"] = summary
        await super().render_completion(state)

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


def _patch_eval_display_bug() -> None:
    """Fix verifiers <=0.1.10 bug: eval_display.py line 718 crashes when error is a dict.

    ``EvalDisplay._make_env_detail`` does ``rich.Text.append(error_0)`` where
    ``error_0`` is an ``ErrorInfo`` dict.  Rich only accepts ``str | Text``.
    We wrap ``_make_env_detail`` to stringify dict errors in the outputs list
    before the original method tries to render them.
    """
    try:
        from verifiers.utils.eval_display import EvalDisplay

        _orig = EvalDisplay._make_env_detail
        if getattr(_orig, "_salesbench_patched", False):
            return

        def _patched(self, config, env_state, results):
            for output in results.get("outputs", []):
                err = output.get("error")
                if isinstance(err, dict):
                    output["error"] = err.get("error_chain_str") or str(err)
            return _orig(self, config, env_state, results)

        _patched._salesbench_patched = True  # type: ignore[attr-defined]
        EvalDisplay._make_env_detail = _patched
    except Exception:
        pass  # Silently skip if framework changes in future versions


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
    context_rewrite_threshold: float = 0.80,
    context_keep_recent: int = 10,
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
        context_rewrite_threshold: Fraction of max_seq_len at which to compress
            older messages into a summary (default 0.80). Disabled when
            max_seq_len is not set by the training infrastructure.
        context_keep_recent: Number of recent messages to keep verbatim
            after context summarization (default 10).
    """

    resolved_seed = base_seed if seed is None else seed

    _patch_eval_display_bug()

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
        context_rewrite_threshold=context_rewrite_threshold,
        context_keep_recent=context_keep_recent,
    )

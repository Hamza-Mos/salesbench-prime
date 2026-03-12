"""SalesBench Prime RL environment — orchestrator layer.

This module implements the 3-party orchestration pattern (inspired by
tau2-bench):

* **Runtime** (``runtime.py``) — pure deterministic state machine
* **Buyer policy** (``policy.py``) — LLM or rule-based buyer participant
* **Orchestrator** (this file) — routes between runtime and buyer, owns
  the buyer policy, handles error isolation and context compression

The buyer policy is created per-episode in :meth:`setup_state` and stored
in ``state["buyer_policy"]``.  It is injected into tool functions that
need it via :meth:`update_tool_args`.  Buyer LLM failures are isolated
from seller scoring — they never produce ``invalid_action`` penalties.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

from datasets import Dataset

import verifiers as vf

from config import EpisodeConfig
from dataset import build_salesbench_dataset
from policy import LLMBuyerPolicy, RuleBasedBuyerPolicy
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
        context_max_seq_len: int | None = None,
        buyer_policy_type: str = "llm",
        buyer_model: str = "gpt-5-mini",
        buyer_base_url: str = "https://api.openai.com/v1",
        buyer_api_key_var: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ) -> None:
        self.default_seed = default_seed
        self.default_num_leads = default_num_leads
        self.default_total_hours = default_total_hours
        self.context_rewrite_threshold = context_rewrite_threshold
        self.context_keep_recent = context_keep_recent
        self.context_max_seq_len = context_max_seq_len

        # Buyer policy config — used per-episode in setup_state
        self.buyer_policy_type = buyer_policy_type
        self.buyer_model = buyer_model
        self.buyer_base_url = buyer_base_url
        self.buyer_api_key_var = buyer_api_key_var

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
            self.add_tool(tool, args_to_skip=["runtime", "messages", "buyer_policy"])

    # ------------------------------------------------------------------
    # State setup
    # ------------------------------------------------------------------

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

        # Buyer policy — separate participant, NOT inside runtime
        if self.buyer_policy_type == "llm":
            buyer_policy: LLMBuyerPolicy | RuleBasedBuyerPolicy = LLMBuyerPolicy(
                model=self.buyer_model,
                base_url=self.buyer_base_url,
                api_key=os.getenv(self.buyer_api_key_var, ""),
            )
        else:
            buyer_policy = RuleBasedBuyerPolicy(seed=config.seed + 17)

        state["runtime"] = runtime
        state["buyer_policy"] = buyer_policy
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

    # ------------------------------------------------------------------
    # Tool argument injection
    # ------------------------------------------------------------------

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
            updated["buyer_policy"] = state.get("buyer_policy")
        return updated

    # ------------------------------------------------------------------
    # Context summarization — compress older messages when nearing
    # max_seq_len to prevent truncation during training.
    # ------------------------------------------------------------------

    def _effective_max_seq_len(self) -> int | None:
        """Return max_seq_len from infra, or the explicit env arg override."""
        return self.max_seq_len or self.context_max_seq_len

    def _should_summarize(self, state: Any) -> bool:
        """Return True when the last turn's prompt exceeded the threshold."""
        max_seq = self._effective_max_seq_len()
        if not max_seq:
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
        threshold = int(max_seq * self.context_rewrite_threshold)
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

        # Avoid consecutive user messages at the summary/recent boundary.
        # If the first recent message is also role=user (e.g. buyer speech),
        # merge the summary into it instead of inserting a separate message.
        if recent and recent[0].get("role") == "user":
            merged = {
                "role": "user",
                "content": summary_text + "\n\n" + str(recent[0].get("content", "")),
            }
            compressed = prefix + [merged] + recent[1:]
        else:
            compressed = prefix + [{"role": "user", "content": summary_text}] + recent

        count = state.get("_context_summary_count", 0) + 1
        state["_context_summary_count"] = count

        logger.info(
            "Context summarized (#%d): %d msgs → %d",
            count,
            len(messages),
            len(compressed),
        )

        return compressed

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

    # ------------------------------------------------------------------
    # Buyer conversation orchestration
    # ------------------------------------------------------------------

    async def _get_buyer_conversation_reply(
        self,
        buyer_policy: LLMBuyerPolicy | RuleBasedBuyerPolicy,
        lead: Any,
        agent_text: str,
        messages: list | None,
    ) -> str | None:
        """Get a buyer conversation reply with error isolation."""
        try:
            result = buyer_policy.generate_response(
                lead=lead, agent_message=agent_text, messages=messages,
            )
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as exc:
            logger.warning("Buyer conversation LLM failed: %s", exc)
            return "I'm listening."

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

            runtime: SalesEpisodeRuntime | None = state.get("runtime")
            buyer_policy = state.get("buyer_policy")
            if runtime:
                # Inject buyer spoken response after propose_offer decisions
                if runtime._pending_buyer_speech:
                    speech = runtime._pending_buyer_speech
                    runtime._pending_buyer_speech = None
                    tool_results = list(tool_results) + [
                        {"role": "user", "content": speech}
                    ]
                # Generate buyer conversation response when agent sent
                # text alongside tool calls during an active call
                elif runtime.active_call and not runtime.done and buyer_policy:
                    agent_text = str(last_msg.get("content", "")).strip()
                    if agent_text:
                        lead = runtime.advance_conversation(agent_text)
                        if lead:
                            buyer_reply = await self._get_buyer_conversation_reply(
                                buyer_policy, lead, agent_text, messages,
                            )
                            if buyer_reply:
                                tool_results = list(tool_results) + [
                                    {"role": "user", "content": f"[{lead.full_name} (buyer)]: {buyer_reply}"}
                                ]

            return tool_results

        # Plain text from agent — inject buyer response if in a call
        runtime = state.get("runtime")
        buyer_policy = state.get("buyer_policy")
        if runtime and not runtime.done and runtime.active_call and buyer_policy:
            agent_text = str(last_msg.get("content", "")).strip()
            if agent_text:
                lead = runtime.advance_conversation(agent_text)
                if lead:
                    buyer_reply = await self._get_buyer_conversation_reply(
                        buyer_policy, lead, agent_text, messages,
                    )
                    if buyer_reply:
                        return [{"role": "user", "content": f"[{lead.full_name} (buyer)]: {buyer_reply}"}]
        return []

    # ------------------------------------------------------------------
    # Stop conditions and episode lifecycle
    # ------------------------------------------------------------------

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
                return False  # keep conversation going during a call
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


def _coerce_int(val: Any, default: int) -> int:
    """Coerce a value to int (platform may pass strings for numeric args)."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _coerce_float(val: Any, default: float) -> float:
    """Coerce a value to float (platform may pass strings for numeric args)."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def load_environment(
    split: str = "train",
    num_examples: int | str = 256,
    eval_num_examples: int | str = 64,
    base_seed: int | str = 42,
    seed: int | str | None = None,
    num_leads: int | str = 100,
    total_hours: int | str = 80,
    buyer_policy: str = "llm",
    buyer_model: str = "gpt-5-mini",
    buyer_base_url: str = "https://api.openai.com/v1",
    buyer_api_key_var: str = "OPENAI_API_KEY",
    max_turns: int | str = 10_000,
    max_examples: int | str = -1,
    system_prompt: str | None = None,
    context_rewrite_threshold: float | str = 0.80,
    context_keep_recent: int | str = 10,
    context_max_seq_len: int | str | None = None,
) -> vf.Environment:
    """Entry-point for Prime verifiers/Prime Lab."""

    # Platform may pass numeric args as strings — coerce them.
    num_examples = _coerce_int(num_examples, 256)
    eval_num_examples = _coerce_int(eval_num_examples, 64)
    base_seed = _coerce_int(base_seed, 42)
    seed = _coerce_int(seed, None) if seed is not None else None  # type: ignore[arg-type]
    num_leads = _coerce_int(num_leads, 100)
    total_hours = _coerce_int(total_hours, 80)
    max_turns = _coerce_int(max_turns, 10_000)
    max_examples = _coerce_int(max_examples, -1)
    context_rewrite_threshold = _coerce_float(context_rewrite_threshold, 0.80)
    context_keep_recent = _coerce_int(context_keep_recent, 10)
    context_max_seq_len = _coerce_int(context_max_seq_len, None) if context_max_seq_len is not None else None  # type: ignore[arg-type]

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
        context_max_seq_len=context_max_seq_len,
        buyer_policy_type=buyer_policy,
        buyer_model=buyer_model,
        buyer_base_url=buyer_base_url,
        buyer_api_key_var=buyer_api_key_var,
    )

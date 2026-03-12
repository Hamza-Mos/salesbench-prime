"""Buyer decision policies for training and evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import openai

from archetypes import (
    ARCHETYPE_PROFILES,
    CRITERIA_EMPHASIS,
    CRITERION_PROMPTS,
)
from models import (
    BuyerArchetype,
    BuyerDecision,
    DecisionResult,
    Lead,
    LeadTemperature,
    Offer,
    PlanType,
    RuntimeActionError,
)

logger = logging.getLogger("verifiers.salesbench")


_TEMPERATURE_THRESHOLD_OFFSET: dict[LeadTemperature, float] = {
    LeadTemperature.COLD: 0.12,
    LeadTemperature.LUKEWARM: 0.04,
    LeadTemperature.WARM: -0.02,
    LeadTemperature.HOT: -0.08,
}

_ARCHETYPE_RESPONSES: dict[BuyerArchetype, list[str]] = {
    BuyerArchetype.ANALYTICAL: [
        "Can you walk me through the numbers on that?",
        "What's the exact breakdown of coverage versus premium?",
        "I'd like to see the ROI on this compared to alternatives.",
    ],
    BuyerArchetype.RELATIONSHIP: [
        "I appreciate you taking the time to explain that.",
        "How long have you been helping families with this?",
        "That's good to know — tell me more about your experience.",
    ],
    BuyerArchetype.SKEPTIC: [
        "What's the catch here?",
        "Are there any hidden fees I should know about?",
        "I've heard that before — what makes this different?",
    ],
    BuyerArchetype.BUDGET_HAWK: [
        "How much would that cost me monthly?",
        "What's the cheapest option you have?",
        "I'm pretty careful with my budget right now.",
    ],
    BuyerArchetype.DELEGATOR: [
        "What would you recommend for someone in my situation?",
        "Just tell me what the best option is.",
        "I trust your expertise — what should I do?",
    ],
    BuyerArchetype.PROCRASTINATOR: [
        "I'll need to think about it.",
        "Can we schedule a follow-up to discuss this further?",
        "I'm not ready to make a decision right now.",
    ],
    BuyerArchetype.STATUS_SEEKER: [
        "What's your premium option?",
        "I want the best coverage available.",
        "What do your top clients typically choose?",
    ],
    BuyerArchetype.PROTECTOR: [
        "What would happen to my family if something happened to me?",
        "How would this protect my kids?",
        "My family's security is my top priority.",
    ],
    BuyerArchetype.COMPARISON_SHOPPER: [
        "How does this compare to what others are offering?",
        "What are people in my situation typically paying?",
        "Can you show me how this stacks up against competitors?",
    ],
    BuyerArchetype.IMPULSE_DECIDER: [
        "That sounds great — what's the next step?",
        "Let's do it, I'm ready.",
        "Sounds good, how do we get started?",
    ],
}


class RuleBasedBuyerPolicy:
    """Rule-based and stochastic buyer model with deterministic seeding."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def evaluate_offer(self, *, lead: Lead, offer: Offer, **kwargs: Any) -> DecisionResult:
        profile = ARCHETYPE_PROFILES[lead.archetype]

        monthly_income = max(lead.annual_income / 12.0, 1.0)
        affordability_ratio = offer.monthly_premium / monthly_income

        target_coverage = max(100_000.0, lead.annual_income * 8.0)
        coverage_error = abs(offer.coverage_amount - target_coverage) / target_coverage
        coverage_fit = max(0.0, 1.0 - coverage_error)

        plan_fit = self._plan_fit(plan_type=offer.plan_type, lead=lead)
        pressure_penalty = max(0, lead.call_count - 1) * 0.08
        price_penalty = affordability_ratio * (1.10 + lead.price_sensitivity) * profile.price_sensitivity_mod

        need_w = 0.42 + profile.need_weight_mod
        trust_w = 0.24 + profile.trust_weight_mod
        coverage_w = 0.22 + profile.coverage_weight_mod
        plan_w = 0.12 + profile.plan_weight_mod

        score = (
            need_w * lead.latent_need
            + trust_w * lead.trust_level
            + coverage_w * coverage_fit
            + plan_w * plan_fit
            - price_penalty
            - pressure_penalty
        )
        score += self._rng.uniform(-0.05, 0.05)

        threshold = 0.62 + _TEMPERATURE_THRESHOLD_OFFSET[lead.temperature]
        request_dnc = lead.call_count >= lead.max_calls and score < 0.45

        if affordability_ratio > 0.060:
            return DecisionResult(
                decision=BuyerDecision.REJECT,
                reason="Premium exceeds practical monthly budget.",
                score=score,
                request_dnc=request_dnc,
            )

        if score >= threshold:
            return DecisionResult(
                decision=BuyerDecision.ACCEPT,
                reason="Coverage and premium align with household priorities.",
                score=score,
                request_dnc=False,
            )

        if request_dnc:
            return DecisionResult(
                decision=BuyerDecision.HANG_UP,
                reason="Repeated outreach reduced trust; do-not-call requested.",
                score=score,
                request_dnc=True,
            )

        if score <= 0.28 and lead.call_count >= 2:
            return DecisionResult(
                decision=BuyerDecision.HANG_UP,
                reason="Offer quality is low and buyer disengaged.",
                score=score,
                request_dnc=False,
            )

        return DecisionResult(
            decision=BuyerDecision.REJECT,
            reason="Buyer declined this offer but remains in pipeline.",
            score=score,
            request_dnc=False,
        )

    def _plan_fit(self, *, plan_type: PlanType, lead: Lead) -> float:
        if plan_type == PlanType.TERM:
            return min(1.0, 0.55 + lead.latent_need * 0.45)
        if plan_type == PlanType.WHOLE:
            return min(1.0, 0.30 + (1.0 - lead.price_sensitivity) * 0.70)
        if plan_type == PlanType.UL:
            return min(1.0, 0.45 + lead.trust_level * 0.55)
        if plan_type == PlanType.DI:
            return min(1.0, 0.40 + lead.latent_need * 0.60)
        return 0.50

    def generate_response(
        self, *, lead: Lead, agent_message: str, **kwargs: Any
    ) -> str:
        """Return a canned buyer response based on archetype and attributes."""
        archetype_pool = _ARCHETYPE_RESPONSES.get(lead.archetype, [])
        # 60% archetype-specific, 40% attribute-based fallback
        if archetype_pool and self._rng.random() < 0.60:
            return self._rng.choice(archetype_pool)

        if lead.price_sensitivity > 0.7:
            options = [
                "How much would that cost me monthly?",
                "I'm pretty careful with my budget right now.",
                "What's the cheapest option you have?",
            ]
        elif lead.latent_need > 0.7:
            options = [
                "Tell me more about the coverage options.",
                "I've been thinking about getting something like this.",
                "What would you recommend for someone in my situation?",
            ]
        elif lead.trust_level < 0.4:
            options = [
                "I'm not sure I'm interested.",
                "Can you send me something in writing first?",
                "I'll need to think about it.",
            ]
        else:
            options = [
                "Okay, what do you have in mind?",
                "I'm listening.",
                "Go ahead, I've got a few minutes.",
            ]
        return self._rng.choice(options)


_TEMPERATURE_READINESS: dict[LeadTemperature, str] = {
    LeadTemperature.COLD: (
        "Your default is to reject. You have no prior interest and are skeptical. "
        "Only an exceptional, well-tailored pitch that clearly addresses your specific "
        "needs could change your mind."
    ),
    LeadTemperature.LUKEWARM: (
        "You are open to persuasion but not eager. A solid pitch with good reasoning "
        "could win you over, but you won't accept anything that feels generic or pushy."
    ),
    LeadTemperature.WARM: (
        "You have expressed interest or been referred. A reasonable pitch with good fit "
        "to your budget and needs is likely enough for you to accept."
    ),
    LeadTemperature.HOT: (
        "You are actively shopping and ready to buy soon. You are likely to accept if "
        "the offer fits your budget and coverage needs. Don't overthink it."
    ),
}

_TEMPERATURE_CONVERSATION_HINT: dict[LeadTemperature, str] = {
    LeadTemperature.COLD: "guarded and skeptical",
    LeadTemperature.LUKEWARM: "politely curious",
    LeadTemperature.WARM: "engaged and interested",
    LeadTemperature.HOT: "proactive and ready",
}


def _build_buyer_decision_prompt(lead: Lead) -> str:
    """Build an instance-specific decision prompt with archetype and rubric criteria."""
    profile = ARCHETYPE_PROFILES[lead.archetype]
    emphasis = CRITERIA_EMPHASIS[lead.archetype]

    primary_lines = "\n".join(
        f"- {c.value}: {CRITERION_PROMPTS[c]}" for c in emphasis.primary
    )
    secondary_lines = "\n".join(
        f"- {c.value}: {CRITERION_PROMPTS[c]}" for c in emphasis.secondary
    )
    de_emphasized_lines = "\n".join(
        f"- {c.value}: {CRITERION_PROMPTS[c]}" for c in emphasis.de_emphasized
    )

    return f"""\
You are a prospective insurance buyer in a simulated sales call. Your job is to \
realistically decide whether to ACCEPT, REJECT, or HANG UP on the offer being \
proposed to you by a sales agent.

## Your Profile
- Name: {lead.full_name}
- Age: {lead.age}
- Annual income: ${lead.annual_income:,}
- Household size: {lead.household_size} ({lead.dependents} dependents)
- Risk class: {lead.risk_class.value}
- Monthly budget for insurance: ${lead.budget_monthly:.2f}
- Need for insurance (0-1): {lead.latent_need:.2f}
- Trust level toward sales agents (0-1): {lead.trust_level:.2f}
- Price sensitivity (0-1): {lead.price_sensitivity:.2f}
- Times contacted so far: {lead.call_count} (max tolerance: {lead.max_calls})
- Temperature: {lead.temperature.value}
- Personality: {profile.label}

## Your Personality
{profile.prompt_modifier}

## How to Evaluate the Agent's Approach
### Most Important to You:
{primary_lines}
### Also Important:
{secondary_lines}
### Less Important:
{de_emphasized_lines}

## Your Readiness Level
{_TEMPERATURE_READINESS[lead.temperature]}

## Decision Guidelines
- You should behave realistically given your profile and personality.
- If the premium is far above your budget, lean toward REJECT.
- If the agent has been pushy or called too many times, consider HANG_UP.
- If the coverage, plan type, and premium genuinely fit your needs and budget, ACCEPT.
- Weigh the agent's selling approach against the criteria above — a well-tailored, \
respectful pitch that addresses your personality should be rewarded.
- Provide a short, natural reason for your decision as the buyer would say it.

## Response format
Respond with a JSON object (no markdown, no extra text):
{{"decision": "accept" | "reject" | "hang_up", "reason": "<short buyer reason>", "request_dnc": true | false}}

Set request_dnc to true only if you are hanging up AND want to be placed on a do-not-call list \
(e.g. too many calls, feeling harassed)."""


def _is_buyer_speech(content: str) -> bool:
    """Check if a user-role message is actual buyer speech (injected by env_response).

    Buyer speech is always formatted as ``[Full Name (buyer)]: reply text``
    by salesbench.py.  Other user-role messages (briefings, context summaries,
    tool results) should NOT be shown to the buyer LLM.
    """
    return content.startswith("[") and "(buyer)]:" in content


def _build_buyer_conversation_context(messages: list | None) -> list[dict[str, str]]:
    """Extract seller-buyer conversation turns for the buyer LLM.

    Keeps only:
    - Seller speech (role=assistant with text content) → flipped to "user"
    - Buyer's own prior replies (role=user matching buyer speech pattern) → flipped to "assistant"

    Filters out: system prompts, tool results (role=tool), briefings,
    context summaries, and other infrastructure user messages.
    """
    if not messages:
        return []
    result: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        content = str(content).strip()
        if not content:
            continue
        if role == "assistant":
            # Seller's speech → buyer sees as "user" (what agent said to me)
            result.append({"role": "user", "content": content})
        elif role == "user" and _is_buyer_speech(content):
            # Buyer's own prior speech → buyer sees as "assistant" (what I said)
            result.append({"role": "assistant", "content": content})
    return result


def _build_buyer_conversation_prompt(lead: Lead) -> str:
    """Build an instance-specific conversation prompt with archetype personality."""
    profile = ARCHETYPE_PROFILES[lead.archetype]
    hint = _TEMPERATURE_CONVERSATION_HINT[lead.temperature]

    return f"""\
You are a prospective insurance buyer in a simulated sales call. You are having \
a conversation with a sales agent. Respond naturally as a real buyer would.

## Your Profile
- Name: {lead.full_name}
- Age: {lead.age}
- Annual income: ${lead.annual_income:,}
- Household size: {lead.household_size} ({lead.dependents} dependents)
- Risk class: {lead.risk_class.value}
- Monthly budget for insurance: ${lead.budget_monthly:.2f}
- Need for insurance (0-1): {lead.latent_need:.2f}
- Trust level toward sales agents (0-1): {lead.trust_level:.2f}
- Price sensitivity (0-1): {lead.price_sensitivity:.2f}
- Times contacted so far: {lead.call_count} (max tolerance: {lead.max_calls})
- Temperature: {lead.temperature.value}
- Personality: {profile.label}

## Your Personality
{profile.prompt_modifier}

## Conversation Guidelines
- Stay in character as a {profile.label.lower()} buyer who is {hint}.
- Respond naturally: ask questions, share concerns, express interest or skepticism.
- Keep responses to 1-3 sentences.
- Do NOT make a purchase decision in this response. Purchase decisions happen \
only when the agent formally proposes an offer.

Respond with plain text only (no JSON, no markdown)."""


# Module-level shared resources so every LLMBuyerPolicy instance (one per
# rollout) reuses the same thread pool and HTTP connection pool instead of
# spawning thousands of threads.
_BUYER_THREAD_POOL = ThreadPoolExecutor(max_workers=128, thread_name_prefix="buyer-llm")
_BUYER_CLIENTS: dict[str, openai.OpenAI] = {}


_BUYER_TIMEOUT = openai.Timeout(120.0, connect=20.0)


def _get_buyer_client(base_url: str, api_key: str) -> openai.OpenAI:
    """Return a shared OpenAI client per (base_url, api_key) pair."""
    key = f"{base_url}:{api_key[:8]}"
    if key not in _BUYER_CLIENTS:
        _BUYER_CLIENTS[key] = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=_BUYER_TIMEOUT
        )
    return _BUYER_CLIENTS[key]


class LLMBuyerPolicy:
    """LLM-based buyer model that uses a cheap model to simulate realistic buyer behavior.

    Uses a synchronous OpenAI client + shared ThreadPoolExecutor to avoid
    blocking the orchestrator's asyncio event loop (same pattern as tau2-synth).
    """

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self._sync_client = _get_buyer_client(base_url, api_key)
        self._thread_pool = _BUYER_THREAD_POOL
        self.model = model

        # Observability counters (per-episode, reset on new policy instance)
        self.call_count: int = 0
        self.timeout_count: int = 0
        self.slow_call_count: int = 0  # >30s
        self.total_latency: float = 0.0
        self.max_latency: float = 0.0

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run a blocking function in the thread pool without blocking the event loop."""
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._thread_pool, partial(func, *args, **kwargs)),
                timeout=180.0,  # 3 min hard cap on any buyer LLM call
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            self.timeout_count += 1
            self.call_count += 1
            self.total_latency += elapsed
            self.max_latency = max(self.max_latency, elapsed)
            logger.error("buyer LLM call timed out after %.1fs (func=%s)", elapsed, func.__name__)
            raise RuntimeActionError(f"buyer LLM call timed out after {elapsed:.1f}s") from None

    def _sync_evaluate_offer(
        self, *, lead: Lead, offer: Offer, messages: list | None = None
    ) -> DecisionResult:
        system_prompt = _build_buyer_decision_prompt(lead)

        offer_description = (
            f"The agent is proposing: {offer.plan_type.value} plan, "
            f"${offer.coverage_amount:,} coverage, "
            f"${offer.monthly_premium:.2f}/month premium"
        )
        if offer.term_years:
            offer_description += f", {offer.term_years}-year term"
        offer_description += f". Next step: {offer.next_step}"

        llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        llm_messages.extend(_build_buyer_conversation_context(messages))
        llm_messages.append({"role": "user", "content": offer_description})

        t0 = time.monotonic()
        try:
            response = self._sync_client.chat.completions.create(
                model=self.model,
                messages=llm_messages,
                temperature=1.0,
                max_completion_tokens=4096,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
        except openai.APITimeoutError as exc:
            elapsed = time.monotonic() - t0
            logger.warning("buyer evaluate_offer timed out after %.1fs: %s", elapsed, exc)
            raise RuntimeActionError(f"buyer LLM timeout after {elapsed:.1f}s: {exc}") from exc
        except Exception as exc:
            raise RuntimeActionError(f"buyer LLM API error: {exc}") from exc

        elapsed = time.monotonic() - t0
        self.call_count += 1
        self.total_latency += elapsed
        self.max_latency = max(self.max_latency, elapsed)
        if elapsed > 30.0:
            self.slow_call_count += 1
            logger.warning("buyer evaluate_offer slow: %.1fs", elapsed)
        logger.debug("buyer LLM raw response (%.1fs): %s", elapsed, raw[:500])

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeActionError(f"buyer LLM returned invalid JSON: {raw[:200]}") from exc

        decision_str = parsed.get("decision", "reject").lower()
        reason = parsed.get("reason", "No reason provided.")
        request_dnc = bool(parsed.get("request_dnc", False))

        decision_map = {
            "accept": BuyerDecision.ACCEPT,
            "reject": BuyerDecision.REJECT,
            "hang_up": BuyerDecision.HANG_UP,
        }
        decision = decision_map.get(decision_str, BuyerDecision.REJECT)

        score_map = {
            BuyerDecision.ACCEPT: 0.80,
            BuyerDecision.REJECT: 0.40,
            BuyerDecision.HANG_UP: 0.15,
        }

        return DecisionResult(
            decision=decision,
            reason=reason,
            score=score_map[decision],
            request_dnc=request_dnc,
        )

    async def evaluate_offer(
        self, *, lead: Lead, offer: Offer, messages: list | None = None
    ) -> DecisionResult:
        return await self._run_in_thread(
            self._sync_evaluate_offer, lead=lead, offer=offer, messages=messages
        )

    def _sync_generate_response(
        self, *, lead: Lead, agent_message: str, messages: list | None = None
    ) -> str:
        system_prompt = _build_buyer_conversation_prompt(lead)

        llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        llm_messages.extend(_build_buyer_conversation_context(messages))
        llm_messages.append({"role": "user", "content": agent_message})

        t0 = time.monotonic()
        try:
            response = self._sync_client.chat.completions.create(
                model=self.model,
                messages=llm_messages,
                temperature=1.0,
                max_completion_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
        except openai.APITimeoutError as exc:
            elapsed = time.monotonic() - t0
            logger.warning("buyer generate_response timed out after %.1fs: %s", elapsed, exc)
            raise RuntimeActionError(f"buyer LLM timeout after {elapsed:.1f}s: {exc}") from exc
        except Exception as exc:
            raise RuntimeActionError(f"buyer LLM API error: {exc}") from exc

        elapsed = time.monotonic() - t0
        self.call_count += 1
        self.total_latency += elapsed
        self.max_latency = max(self.max_latency, elapsed)
        if elapsed > 30.0:
            self.slow_call_count += 1
            logger.warning("buyer generate_response slow: %.1fs", elapsed)
        logger.debug("buyer LLM conversation response (%.1fs): %s", elapsed, raw[:500])
        return raw.strip() or "I'm listening."

    async def generate_response(
        self, *, lead: Lead, agent_message: str, messages: list | None = None
    ) -> str:
        return await self._run_in_thread(
            self._sync_generate_response, lead=lead, agent_message=agent_message, messages=messages
        )

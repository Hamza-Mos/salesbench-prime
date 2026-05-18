"""Buyer LLM policy for training and evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import openai

from archetypes import (
    ARCHETYPE_PROFILES,
    CRITERIA_EMPHASIS,
    CRITERION_PROMPTS,
)
from models import (
    BuyerDecision,
    DecisionResult,
    Lead,
    LeadTemperature,
    Offer,
    RuntimeActionError,
)

logger = logging.getLogger("verifiers.salesbench")


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


# Buyer prompt variants for ablation studies (publication-ready).
# Each variant overrides the "Decision Guidelines" section with a
# deliberately different decision-making style. All variants keep the
# same profile/personality/criteria sections so the model still has the
# same information — only the *interpretation* differs.

_DECISION_GUIDELINES_DEFAULT = """\
- You should behave realistically given your profile and personality.
- If the premium is far above your budget, lean toward REJECT.
- If the agent has been pushy or called too many times, consider HANG_UP.
- If the coverage, plan type, and premium genuinely fit your needs and budget, ACCEPT.
- Weigh the agent's selling approach against the criteria above — a well-tailored, \
respectful pitch that addresses your personality should be rewarded.
- Provide a short, natural reason for your decision as the buyer would say it."""

_DECISION_GUIDELINES_SKEPTICAL = """\
- You are NATURALLY SKEPTICAL of insurance sales pitches. You have been burned by \
hidden fees and pushy agents before, and you default to suspicion.
- Your DEFAULT response to any offer is REJECT or "let me think about it" (REJECT). \
Acceptance must be EARNED through clear, undeniable fit.
- For ACCEPTANCE you require ALL of these to be true:
  * Premium is comfortably within budget (≤ 75% of monthly budget — not "fits" but \
"clearly fits with room to spare")
  * Coverage amount is well-aligned to your dependents and income (5-10× annual income)
  * Plan type clearly matches your stated need or risk profile
  * The agent has demonstrated specific understanding of YOUR situation, not just \
their product
- Even when ALL the above are met, you should STILL reject ~30% of the time with a \
"this is interesting but I want to think about it more" reason. Insurance is a \
serious commitment and you don't decide on the spot.
- For BORDERLINE offers (premium near budget OR coverage roughly fits OR plan type \
arguable): always REJECT.
- If the agent has called more than once without earning real trust, lean toward HANG_UP.
- Target acceptance rate: ~25-30% across all offers. Most pitches get a "no" or \
"I need to think about it" from you.
- Provide a short, naturally skeptical reason — even on accepts, voice one concern."""

_DECISION_GUIDELINES_IMPULSIVE = """\
- You make insurance decisions QUICKLY based on gut feel and emotional resonance. \
You don't second-guess yourself once a pitch lands.
- HARD FLOORS (impulsive does NOT mean reckless — even you have limits):
  * If premium > 1.5× your monthly budget, REJECT (you can't actually pay it).
  * If plan type is obviously wrong for your stated need (e.g. WHOLE life when you \
explicitly need DI for income protection, or DI when you explicitly want family \
death benefit), REJECT.
  * If coverage is obviously inadequate for your dependents (e.g. < 3× income with \
multiple dependents), REJECT.
- For EVERYTHING ELSE: lean strongly toward ACCEPT. If the agent has built any \
rapport AND the premium is reachable AND the plan generally fits, you say yes.
- You don't dwell on coverage details — if the plan SOUNDS like it covers what \
worries you, that's enough.
- A pitch that names your concerns, family, or stated need explicitly is a near-instant ACCEPT.
- Only HANG_UP if the agent is openly disrespectful — not because of a bad offer.
- Roughly: ACCEPT ~70-80% of offers that pass the hard floors. You are an "easy yes" \
buyer who still has basic self-preservation.
- Provide a short, warm, decisive reason."""

_DECISION_GUIDELINES_ANALYTICAL = """\
- You evaluate insurance offers PURELY on numerical fit. Personality, rapport, and \
pitch style are IRRELEVANT to your decision — only the numbers matter.
- Step 1: compute affordability_ratio = monthly_premium / (annual_income/12). If > 6%, \
REJECT regardless of other factors.
- Step 2: compute coverage_fit = coverage_amount / (annual_income × 8). If < 0.5 or \
> 1.5, REJECT (under- or over-insured).
- Step 3: plan_type fit. TERM matches high latent_need; WHOLE/UL match high \
trust_level + low price_sensitivity; DI matches high need + income protection focus. \
Mismatch → REJECT.
- ACCEPT only when ALL three checks pass AND premium is within your stated budget \
(strict, not lenient).
- HANG_UP only if call_count > max_calls. Personality, pushiness, or rapport do NOT \
trigger HANG_UP for you.
- Your reason MUST cite specific numbers (premium, coverage, ratios) — not feelings."""


_DECISION_GUIDELINES_VARIANTS: dict[str, str] = {
    "default": _DECISION_GUIDELINES_DEFAULT,
    "skeptical": _DECISION_GUIDELINES_SKEPTICAL,
    "impulsive": _DECISION_GUIDELINES_IMPULSIVE,
    "analytical": _DECISION_GUIDELINES_ANALYTICAL,
}


def _build_buyer_decision_prompt(lead: Lead, variant: str = "default") -> str:
    """Build an instance-specific decision prompt with archetype and rubric criteria.

    The ``variant`` arg selects the Decision Guidelines block — used for
    publication-ready buyer-prompt ablation studies. Profile, personality,
    and criteria sections are identical across variants; only the
    decision-making interpretation differs.
    """
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

    decision_guidelines = _DECISION_GUIDELINES_VARIANTS.get(
        variant, _DECISION_GUIDELINES_DEFAULT
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
{decision_guidelines}

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

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        prompt_variant: str = "default",
    ) -> None:
        self._sync_client = _get_buyer_client(base_url, api_key)
        self._thread_pool = _BUYER_THREAD_POOL
        self.model = model
        self.prompt_variant = prompt_variant

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
        system_prompt = _build_buyer_decision_prompt(lead, variant=self.prompt_variant)

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

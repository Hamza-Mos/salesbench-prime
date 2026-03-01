"""Buyer decision policies for training and evaluation."""

from __future__ import annotations

import json
import logging
import random
from typing import Any

import openai

from models import BuyerDecision, DecisionResult, Lead, Offer, PlanType

logger = logging.getLogger("salesbench")


class RuleBasedBuyerPolicy:
    """Rule-based and stochastic buyer model with deterministic seeding."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def evaluate_offer(self, *, lead: Lead, offer: Offer, **kwargs: Any) -> DecisionResult:
        monthly_income = max(lead.annual_income / 12.0, 1.0)
        affordability_ratio = offer.monthly_premium / monthly_income

        target_coverage = max(100_000.0, lead.annual_income * 8.0)
        coverage_error = abs(offer.coverage_amount - target_coverage) / target_coverage
        coverage_fit = max(0.0, 1.0 - coverage_error)

        plan_fit = self._plan_fit(plan_type=offer.plan_type, lead=lead)
        pressure_penalty = max(0, lead.call_count - 1) * 0.08
        price_penalty = affordability_ratio * (1.10 + lead.price_sensitivity)

        score = (
            0.42 * lead.latent_need
            + 0.24 * lead.trust_level
            + 0.22 * coverage_fit
            + 0.12 * plan_fit
            - price_penalty
            - pressure_penalty
        )
        score += self._rng.uniform(-0.05, 0.05)

        request_dnc = lead.call_count >= lead.max_calls and score < 0.45

        if affordability_ratio > 0.060:
            return DecisionResult(
                decision=BuyerDecision.REJECT,
                reason="Premium exceeds practical monthly budget.",
                score=score,
                request_dnc=request_dnc,
            )

        if score >= 0.62:
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


_LLM_BUYER_SYSTEM = """\
You are a prospective insurance buyer in a simulated sales call. Your job is to \
realistically decide whether to ACCEPT, REJECT, or HANG UP on the offer being \
proposed to you by a sales agent.

## Your Profile
- Name: {name}
- Age: {age}
- Annual income: ${income:,}
- Household size: {household} ({dependents} dependents)
- Risk class: {risk_class}
- Monthly budget for insurance: ${budget:.2f}
- Need for insurance (0-1): {need:.2f}
- Trust level toward sales agents (0-1): {trust:.2f}
- Price sensitivity (0-1): {price_sensitivity:.2f}
- Times contacted so far: {call_count} (max tolerance: {max_calls})

## Decision guidelines
- You should behave realistically given your profile.
- If the premium is far above your budget, lean toward REJECT.
- If the agent has been pushy or called too many times, consider HANG_UP.
- If the coverage, plan type, and premium genuinely fit your needs and budget, ACCEPT.
- Provide a short, natural reason for your decision as the buyer would say it.

## Response format
Respond with a JSON object (no markdown, no extra text):
{{"decision": "accept" | "reject" | "hang_up", "reason": "<short buyer reason>", "request_dnc": true | false}}

Set request_dnc to true only if you are hanging up AND want to be placed on a do-not-call list \
(e.g. too many calls, feeling harassed).
"""


class LLMBuyerPolicy:
    """LLM-based buyer model that uses a cheap model to simulate realistic buyer behavior."""

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def evaluate_offer(
        self, *, lead: Lead, offer: Offer, messages: list | None = None
    ) -> DecisionResult:
        system_prompt = _LLM_BUYER_SYSTEM.format(
            name=lead.full_name,
            age=lead.age,
            income=lead.annual_income,
            household=lead.household_size,
            dependents=lead.dependents,
            risk_class=lead.risk_class.value,
            budget=lead.budget_monthly,
            need=lead.latent_need,
            trust=lead.trust_level,
            price_sensitivity=lead.price_sensitivity,
            call_count=lead.call_count,
            max_calls=lead.max_calls,
        )

        offer_description = (
            f"The agent is proposing: {offer.plan_type.value} plan, "
            f"${offer.coverage_amount:,} coverage, "
            f"${offer.monthly_premium:.2f}/month premium"
        )
        if offer.term_years:
            offer_description += f", {offer.term_years}-year term"
        offer_description += f". Next step: {offer.next_step}"

        llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if messages:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    llm_messages.append({"role": role, "content": str(content)})

        llm_messages.append({"role": "user", "content": offer_description})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=llm_messages,
            temperature=0.7,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

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

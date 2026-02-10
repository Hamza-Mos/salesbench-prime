"""Deterministic buyer decision policy for training and evaluation."""

from __future__ import annotations

import random


class BuyerPolicy:
    """Rule-based and stochastic buyer model with deterministic seeding."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def evaluate_offer(self, *, lead: Lead, offer: Offer) -> DecisionResult:
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

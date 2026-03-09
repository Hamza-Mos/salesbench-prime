"""Domain models for the SalesBench Prime RL harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimeActionError(ValueError):
    """Raised when the agent attempts an invalid runtime action."""


class LeadTemperature(str, Enum):
    COLD = "cold"          # No prior interest, skeptical, hard to engage
    LUKEWARM = "lukewarm"  # Some awareness, open to hearing more
    WARM = "warm"          # Has expressed interest or been referred
    HOT = "hot"            # Actively shopping, ready to buy soon


class BuyerArchetype(str, Enum):
    ANALYTICAL = "analytical"
    RELATIONSHIP = "relationship"
    SKEPTIC = "skeptic"
    BUDGET_HAWK = "budget_hawk"
    DELEGATOR = "delegator"
    PROCRASTINATOR = "procrastinator"
    STATUS_SEEKER = "status_seeker"
    PROTECTOR = "protector"
    COMPARISON_SHOPPER = "comparison_shopper"
    IMPULSE_DECIDER = "impulse_decider"


class LeadStatus(str, Enum):
    ACTIVE = "active"
    CONVERTED = "converted"
    DNC = "dnc"
    EXHAUSTED = "exhausted"


class BuyerDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    HANG_UP = "hang_up"


class PlanType(str, Enum):
    TERM = "TERM"
    WHOLE = "WHOLE"
    UL = "UL"
    DI = "DI"


class RiskClass(str, Enum):
    PREFERRED = "preferred"
    STANDARD = "standard"
    SUBSTANDARD = "substandard"


class CallbackStatus(str, Enum):
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELED = "canceled"


@dataclass(slots=True)
class Offer:
    plan_type: PlanType
    coverage_amount: int
    monthly_premium: float
    next_step: str
    term_years: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "plan_type": self.plan_type.value,
            "coverage_amount": self.coverage_amount,
            "monthly_premium": self.monthly_premium,
            "next_step": self.next_step,
        }
        if self.term_years is not None:
            payload["term_years"] = self.term_years
        return payload


@dataclass(slots=True)
class DecisionResult:
    decision: BuyerDecision
    reason: str
    score: float
    request_dnc: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "score": round(self.score, 4),
            "request_dnc": self.request_dnc,
        }


@dataclass(slots=True)
class Lead:
    lead_id: str
    full_name: str
    age: int
    annual_income: int
    state_code: str
    household_size: int
    dependents: int
    risk_class: RiskClass
    latent_need: float
    trust_level: float
    price_sensitivity: float
    budget_monthly: float
    max_calls: int
    temperature: LeadTemperature = LeadTemperature.LUKEWARM
    archetype: BuyerArchetype = BuyerArchetype.ANALYTICAL
    status: LeadStatus = LeadStatus.ACTIVE
    smoker: bool = False
    stated_need: str | None = None
    do_not_call: bool = False
    call_count: int = 0
    notes: list[str] = field(default_factory=list)
    accepted_offer: Offer | None = None

    def to_search_dict(self) -> dict[str, Any]:
        """Compact dict for search results — shorter keys, omits always-active status and always-false dnc."""
        d: dict[str, Any] = {
            "id": self.lead_id,
            "name": self.full_name,
            "age": self.age,
            "income": self.annual_income,
            "state": self.state_code,
            "need": round(self.latent_need, 3),
            "budget": round(self.budget_monthly, 2),
            "temp": self.temperature.value,
            "archetype": self.archetype.value,
            "smoker": self.smoker,
            "calls": self.call_count,
        }
        if self.stated_need is not None:
            d["stated_need"] = self.stated_need
        return d

    def to_brief_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "lead_id": self.lead_id,
            "name": self.full_name,
            "status": self.status.value,
            "age": self.age,
            "annual_income": self.annual_income,
            "state": self.state_code,
            "need_score": round(self.latent_need, 3),
            "budget_monthly": round(self.budget_monthly, 2),
            "temperature": self.temperature.value,
            "archetype": self.archetype.value,
            "smoker": self.smoker,
            "calls_made": self.call_count,
            "do_not_call": self.do_not_call,
        }
        if self.stated_need is not None:
            d["stated_need"] = self.stated_need
        return d

    def to_detail_dict(self) -> dict[str, Any]:
        payload = self.to_brief_dict()
        payload.update(
            {
                "household_size": self.household_size,
                "dependents": self.dependents,
                "risk_class": self.risk_class.value,
                "trust_level": round(self.trust_level, 3),
                "price_sensitivity": round(self.price_sensitivity, 3),
                "max_calls": self.max_calls,
                "notes": self.notes,
                "accepted_offer": (self.accepted_offer.to_dict() if self.accepted_offer else None),
            }
        )
        return payload


@dataclass(slots=True)
class CallbackTask:
    callback_id: str
    lead_id: str
    due_minute: int
    reason: str
    status: CallbackStatus = CallbackStatus.SCHEDULED

    def to_dict(self) -> dict[str, Any]:
        return {
            "callback_id": self.callback_id,
            "lead_id": self.lead_id,
            "due_minute": self.due_minute,
            "reason": self.reason,
            "status": self.status.value,
        }

    def to_compact_dict(self) -> dict[str, Any]:
        """Compact dict for list responses — shorter keys, omits status."""
        return {
            "id": self.callback_id,
            "lead": self.lead_id,
            "due_min": self.due_minute,
            "reason": self.reason,
        }


@dataclass(slots=True)
class CallSession:
    call_id: str
    lead_id: str
    started_minute: int
    offers: list[Offer] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    messages_sent: int = 0
    ended_minute: int | None = None
    outcome: BuyerDecision | None = None

    @property
    def duration_minutes(self) -> int:
        if self.ended_minute is None:
            return 0
        return max(1, self.ended_minute - self.started_minute)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "lead_id": self.lead_id,
            "started_minute": self.started_minute,
            "ended_minute": self.ended_minute,
            "duration_minutes": self.duration_minutes,
            "messages_sent": self.messages_sent,
            "offers": [offer.to_dict() for offer in self.offers],
            "notes": self.notes,
            "outcome": self.outcome.value if self.outcome else None,
        }


@dataclass(slots=True)
class EpisodeStats:
    tool_calls: int = 0
    invalid_actions: int = 0
    calls_started: int = 0
    calls_completed: int = 0
    conversions: int = 0
    rejected_offers: int = 0
    hang_ups: int = 0
    dnc_violations: int = 0
    revenue_mrr: float = 0.0
    offers_proposed: int = 0
    offers_accepted: int = 0
    leads_contacted: int = 0
    callbacks_scheduled: int = 0
    callbacks_completed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "invalid_actions": self.invalid_actions,
            "calls_started": self.calls_started,
            "calls_completed": self.calls_completed,
            "conversions": self.conversions,
            "rejected_offers": self.rejected_offers,
            "hang_ups": self.hang_ups,
            "dnc_violations": self.dnc_violations,
            "revenue_mrr": round(self.revenue_mrr, 2),
            "offers_proposed": self.offers_proposed,
            "offers_accepted": self.offers_accepted,
            "leads_contacted": self.leads_contacted,
            "callbacks_scheduled": self.callbacks_scheduled,
            "callbacks_completed": self.callbacks_completed,
        }

"""Stateful sales simulation runtime for Prime verifiers tooling."""

from __future__ import annotations

import logging
from typing import Any

from catalog import ProductCatalog
from config import EpisodeConfig
from generator import LeadGenerator
from models import (
    BuyerDecision,
    CallbackStatus,
    CallSession,
    CallbackTask,
    EpisodeStats,
    Lead,
    LeadStatus,
    Offer,
    PlanType,
)
from policy import BuyerPolicy

logger = logging.getLogger("salesbench")


class RuntimeActionError(ValueError):
    """Raised when the agent attempts an invalid runtime action."""


class SalesEpisodeRuntime:
    """Canonical owner of per-episode state."""

    def __init__(self, config: EpisodeConfig) -> None:
        config.validate()
        self.config = config
        self.catalog = ProductCatalog()
        self.policy = BuyerPolicy(seed=config.seed + 17)

        leads = LeadGenerator(seed=config.seed).generate(config.num_leads)
        self.leads: dict[str, Lead] = {lead.lead_id: lead for lead in leads}

        self.stats = EpisodeStats()
        self.current_minute = 0
        self.done = False
        self.termination_reason: str | None = None

        self.active_call: CallSession | None = None
        self.call_history: list[CallSession] = []
        self.callbacks: dict[str, CallbackTask] = {}

        self._call_counter = 0
        self._callback_counter = 0
        self._contacted_leads: set[str] = set()
        self.event_log: list[str] = []

        self._record_event(
            f"episode_started seed={config.seed} leads={config.num_leads} budget={config.max_minutes}m"
        )
        logger.info(
            "Episode started: seed=%d leads=%d budget=%dm",
            config.seed,
            config.num_leads,
            config.max_minutes,
        )

    @property
    def num_active_leads(self) -> int:
        return sum(
            1
            for lead in self.leads.values()
            if lead.status == LeadStatus.ACTIVE and not lead.do_not_call
        )

    def register_tool_call(self, tool_name: str) -> None:
        self.stats.tool_calls += 1
        self._record_event(f"tool={tool_name}")
        logger.debug("tool_call=%s total=%d", tool_name, self.stats.tool_calls)

    def record_invalid_action(self, message: str) -> None:
        self.stats.invalid_actions += 1
        self._record_event(f"invalid_action={message}")
        logger.warning("Invalid action (%d total): %s", self.stats.invalid_actions, message)
        self._check_termination()

    def search_leads(
        self,
        *,
        min_income: int | None,
        max_age: int | None,
        min_need: float | None,
        limit: int,
        include_called: bool,
    ) -> dict[str, Any]:
        self._raise_if_done()
        if limit <= 0:
            raise RuntimeActionError("limit must be > 0")

        leads = []
        for lead in self.leads.values():
            if lead.status != LeadStatus.ACTIVE or lead.do_not_call:
                continue
            if min_income is not None and lead.annual_income < min_income:
                continue
            if max_age is not None and lead.age > max_age:
                continue
            if min_need is not None and lead.latent_need < min_need:
                continue
            if not include_called and lead.call_count > 0:
                continue
            leads.append(lead)

        leads.sort(
            key=lambda item: (
                -item.latent_need,
                -(item.budget_monthly),
                item.call_count,
            )
        )

        filtered = leads[:limit]
        self._advance(self.config.tool_costs.crm_search_minutes, "crm_search")
        return {
            "count": len(filtered),
            "leads": [lead.to_brief_dict() for lead in filtered],
            "active_leads_remaining": self.num_active_leads,
        }

    def get_lead(self, *, lead_id: str) -> dict[str, Any]:
        lead = self._get_lead_or_error(lead_id)
        return {"lead": lead.to_detail_dict()}

    def add_note(self, *, lead_id: str, note: str) -> dict[str, Any]:
        if not note.strip():
            raise RuntimeActionError("note cannot be empty")
        lead = self._get_lead_or_error(lead_id)
        lead.notes.append(note.strip())
        return {
            "lead_id": lead_id,
            "note_count": len(lead.notes),
            "latest_note": lead.notes[-1],
        }

    def pipeline_summary(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for lead in self.leads.values():
            status_counts.setdefault(lead.status.value, 0)
            status_counts[lead.status.value] += 1

        return {
            "status_counts": status_counts,
            "stats": self.stats.to_dict(),
            "time": {
                "current_minute": self.current_minute,
                "max_minutes": self.config.max_minutes,
                "remaining_minutes": max(0, self.config.max_minutes - self.current_minute),
            },
            "active_call": self.active_call.to_dict() if self.active_call else None,
            "done": self.done,
            "termination_reason": self.termination_reason,
        }

    def schedule_callback(
        self,
        *,
        lead_id: str,
        hours_from_now: int,
        reason: str,
    ) -> dict[str, Any]:
        self._raise_if_done()
        lead = self._get_lead_or_error(lead_id)
        if lead.status != LeadStatus.ACTIVE or lead.do_not_call:
            raise RuntimeActionError("cannot schedule callback for inactive or DNC lead")
        if hours_from_now <= 0:
            raise RuntimeActionError("hours_from_now must be > 0")
        if not reason.strip():
            raise RuntimeActionError("reason cannot be empty")

        existing = [
            task
            for task in self.callbacks.values()
            if task.lead_id == lead_id and task.status == CallbackStatus.SCHEDULED
        ]
        if len(existing) >= self.config.max_callbacks_per_lead:
            raise RuntimeActionError("callback limit reached for this lead")

        due_minute = self.current_minute + (hours_from_now * 60)
        if due_minute > self.config.max_minutes:
            raise RuntimeActionError("callback exceeds remaining episode budget")

        self._callback_counter += 1
        callback_id = f"cb_{self._callback_counter:04d}"
        callback = CallbackTask(
            callback_id=callback_id,
            lead_id=lead_id,
            due_minute=due_minute,
            reason=reason.strip(),
        )
        self.callbacks[callback_id] = callback
        self.stats.callbacks_scheduled += 1
        logger.debug(
            "Callback scheduled: %s for lead=%s due_minute=%d",
            callback_id,
            lead_id,
            due_minute,
        )
        self._advance(self.config.tool_costs.schedule_callback_minutes, "schedule_callback")
        return {
            "callback": callback.to_dict(),
            "scheduled_count_for_lead": len(existing) + 1,
        }

    def list_callbacks(
        self,
        *,
        within_hours: int,
        include_completed: bool,
    ) -> dict[str, Any]:
        if within_hours <= 0:
            raise RuntimeActionError("within_hours must be > 0")

        horizon = self.current_minute + (within_hours * 60)
        callbacks = []
        for task in self.callbacks.values():
            if task.due_minute > horizon:
                continue
            if not include_completed and task.status != CallbackStatus.SCHEDULED:
                continue
            callbacks.append(task)

        callbacks.sort(key=lambda item: item.due_minute)
        return {
            "count": len(callbacks),
            "callbacks": [task.to_dict() for task in callbacks],
        }

    def start_call(self, *, lead_id: str) -> dict[str, Any]:
        self._raise_if_done()
        if self.active_call is not None:
            raise RuntimeActionError(
                "an active call already exists; end it before starting another"
            )

        lead = self._get_lead_or_error(lead_id)
        if lead.do_not_call or lead.status == LeadStatus.DNC:
            self.stats.dnc_violations += 1
            raise RuntimeActionError("lead is on do-not-call list")
        if lead.status != LeadStatus.ACTIVE:
            raise RuntimeActionError("lead is not active")

        self._call_counter += 1
        call_id = f"call_{self._call_counter:04d}"
        session = CallSession(
            call_id=call_id,
            lead_id=lead_id,
            started_minute=self.current_minute,
        )
        self.active_call = session
        lead.call_count += 1
        self.stats.calls_started += 1
        if lead_id not in self._contacted_leads:
            self._contacted_leads.add(lead_id)
            self.stats.leads_contacted += 1
        self._mark_due_callbacks_completed(lead_id=lead_id)
        self._advance(self.config.tool_costs.start_call_minutes, "start_call")
        logger.debug(
            "Call started: %s lead=%s (%s) minute=%d",
            call_id,
            lead_id,
            lead.full_name,
            session.started_minute,
        )

        return {
            "call_id": session.call_id,
            "lead": lead.to_brief_dict(),
            "message": "Call connected; gather requirements then propose an offer.",
        }

    def quote_plan(
        self,
        *,
        lead_id: str,
        plan_type: str,
        coverage_amount: int,
        term_years: int | None,
    ) -> dict[str, Any]:
        self._raise_if_done()
        lead = self._get_lead_or_error(lead_id)
        plan = self._parse_plan_type(plan_type)

        quote = self.catalog.quote(
            plan_type=plan,
            age=lead.age,
            coverage_amount=coverage_amount,
            risk_class=lead.risk_class,
            term_years=term_years,
        )
        self._advance(self.config.tool_costs.quote_minutes, "quote_plan")

        affordability_ratio = quote["monthly_premium"] / max(lead.budget_monthly, 1.0)
        quote["lead_budget_monthly"] = round(lead.budget_monthly, 2)
        quote["premium_to_budget_ratio"] = round(affordability_ratio, 3)
        return {"quote": quote}

    def propose_offer(
        self,
        *,
        plan_type: str,
        coverage_amount: int,
        monthly_premium: float,
        next_step: str,
        term_years: int | None,
    ) -> dict[str, Any]:
        self._raise_if_done()
        if self.active_call is None:
            raise RuntimeActionError("no active call")
        if monthly_premium <= 0:
            raise RuntimeActionError("monthly_premium must be > 0")
        if coverage_amount <= 0:
            raise RuntimeActionError("coverage_amount must be > 0")
        if not next_step.strip():
            raise RuntimeActionError("next_step cannot be empty")

        lead = self._get_lead_or_error(self.active_call.lead_id)
        plan = self._parse_plan_type(plan_type)
        offer = Offer(
            plan_type=plan,
            coverage_amount=coverage_amount,
            monthly_premium=round(monthly_premium, 2),
            next_step=next_step.strip(),
            term_years=term_years,
        )

        self.active_call.offers.append(offer)
        self.stats.offers_proposed += 1
        self._advance(self.config.tool_costs.propose_offer_minutes, "propose_offer")
        decision = self.policy.evaluate_offer(lead=lead, offer=offer)
        logger.debug(
            "Offer proposed: lead=%s plan=%s premium=%.2f decision=%s",
            lead.lead_id,
            plan.value,
            monthly_premium,
            decision.decision.value,
        )

        response: dict[str, Any] = {
            "offer": offer.to_dict(),
            "decision": decision.to_dict(),
        }

        if decision.decision == BuyerDecision.ACCEPT:
            lead.status = LeadStatus.CONVERTED
            lead.accepted_offer = offer
            self.active_call.outcome = BuyerDecision.ACCEPT
            self.stats.conversions += 1
            self.stats.offers_accepted += 1
            self.stats.revenue_mrr += offer.monthly_premium
            response["message"] = "Buyer accepted. End the call to finalize the conversion."
            logger.info(
                "Conversion: lead=%s premium=%.2f total_mrr=%.2f",
                lead.lead_id,
                offer.monthly_premium,
                self.stats.revenue_mrr,
            )

        elif decision.decision == BuyerDecision.REJECT:
            self.stats.rejected_offers += 1
            response["message"] = "Buyer rejected this offer. You can try a revised proposal."

        elif decision.decision == BuyerDecision.HANG_UP:
            self.stats.hang_ups += 1
            self.active_call.outcome = BuyerDecision.HANG_UP
            response["message"] = "Buyer ended the call."
            if decision.request_dnc:
                lead.status = LeadStatus.DNC
                lead.do_not_call = True
                response["message"] += " Do-not-call requested."
                logger.debug("Lead %s marked DNC after hang-up", lead.lead_id)
            self._finalize_active_call(reason="buyer_hang_up")

        self._check_termination()
        return response

    def end_call(self, *, disposition: str) -> dict[str, Any]:
        if self.active_call is None:
            raise RuntimeActionError("no active call")
        if not disposition.strip():
            raise RuntimeActionError("disposition cannot be empty")

        self._advance(self.config.tool_costs.end_call_minutes, "end_call")
        call = self._finalize_active_call(reason=disposition.strip())
        self._check_termination()
        return {
            "call": call.to_dict(),
            "disposition": disposition.strip(),
        }

    def render_briefing(self) -> str:
        return (
            "Episode briefing:\n"
            f"- Leads loaded: {len(self.leads)}\n"
            f"- Time budget: {self.config.max_minutes} minutes\n"
            "- Objective: maximize monthly recurring premium while avoiding compliance failures\n"
            "- Termination: time exhausted, pipeline exhausted, or too many invalid actions"
        )

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "time": {
                "current_minute": self.current_minute,
                "max_minutes": self.config.max_minutes,
                "remaining_minutes": max(0, self.config.max_minutes - self.current_minute),
            },
            "done": self.done,
            "termination_reason": self.termination_reason,
            "stats": self.stats.to_dict(),
            "active_call": self.active_call.to_dict() if self.active_call else None,
            "callbacks": [task.to_dict() for task in self.callbacks.values()],
            "recent_events": self.event_log[-20:],
        }

    def export_summary(self) -> dict[str, Any]:
        converted = [lead for lead in self.leads.values() if lead.status == LeadStatus.CONVERTED]

        status_counts: dict[str, int] = {}
        for lead in self.leads.values():
            status_counts.setdefault(lead.status.value, 0)
            status_counts[lead.status.value] += 1

        calls_detail = []
        for session in self.call_history:
            lead = self.leads.get(session.lead_id)
            revenue = 0.0
            if lead and lead.accepted_offer and session.outcome == BuyerDecision.ACCEPT:
                revenue = lead.accepted_offer.monthly_premium
            calls_detail.append(
                {
                    "call_id": session.call_id,
                    "lead_id": session.lead_id,
                    "lead_name": lead.full_name if lead else "unknown",
                    "started_minute": session.started_minute,
                    "ended_minute": session.ended_minute,
                    "duration_minutes": session.duration_minutes,
                    "offers_made": len(session.offers),
                    "outcome": session.outcome.value if session.outcome else None,
                    "revenue": round(revenue, 2),
                }
            )

        return {
            "termination_reason": self.termination_reason,
            "time_used_minutes": self.current_minute,
            "time_budget_minutes": self.config.max_minutes,
            "funnel": {
                "total_leads": len(self.leads),
                "leads_contacted": self.stats.leads_contacted,
                "leads_converted": status_counts.get("converted", 0),
                "leads_dnc": status_counts.get("dnc", 0),
                "leads_exhausted": status_counts.get("exhausted", 0),
                "leads_remaining": status_counts.get("active", 0),
            },
            "calls": calls_detail,
            "stats": self.stats.to_dict(),
            "converted_leads": [lead.to_detail_dict() for lead in converted],
        }

    def _raise_if_done(self) -> None:
        if self.done:
            reason = self.termination_reason or "episode has terminated"
            raise RuntimeActionError(reason)

    def _get_lead_or_error(self, lead_id: str) -> Lead:
        lead = self.leads.get(lead_id)
        if lead is None:
            raise RuntimeActionError(f"unknown lead_id: {lead_id}")
        return lead

    def _parse_plan_type(self, plan_type: str) -> PlanType:
        try:
            return PlanType(plan_type)
        except ValueError as exc:
            valid = ", ".join(plan.value for plan in PlanType)
            raise RuntimeActionError(f"invalid plan_type '{plan_type}'. valid: {valid}") from exc

    def _advance(self, minutes: int, action: str) -> None:
        if minutes < 0:
            raise RuntimeActionError("minutes cannot be negative")
        self.current_minute += minutes
        self._record_event(f"time+{minutes} action={action} now={self.current_minute}")
        self._check_termination()

    def _mark_due_callbacks_completed(self, *, lead_id: str) -> None:
        for task in self.callbacks.values():
            if task.lead_id != lead_id:
                continue
            if task.status != CallbackStatus.SCHEDULED:
                continue
            if task.due_minute <= self.current_minute:
                task.status = CallbackStatus.COMPLETED
                self.stats.callbacks_completed += 1
                logger.debug("Callback completed: %s lead=%s", task.callback_id, lead_id)

    def _finalize_active_call(self, *, reason: str) -> CallSession:
        assert self.active_call is not None
        session = self.active_call
        session.ended_minute = self.current_minute
        if session.outcome is None:
            session.outcome = BuyerDecision.REJECT
        session.notes.append(f"finalized:{reason}")
        self.call_history.append(session)
        self.active_call = None
        self.stats.calls_completed += 1
        return session

    def _check_termination(self) -> None:
        if self.done:
            return
        if self.current_minute >= self.config.max_minutes:
            self.done = True
            self.termination_reason = "time_budget_exhausted"
            self._record_event("episode_done=time_budget_exhausted")
            logger.info("Episode terminated: time_budget_exhausted at minute %d", self.current_minute)
            return
        if self.stats.invalid_actions >= self.config.max_invalid_actions:
            self.done = True
            self.termination_reason = "invalid_action_limit_reached"
            self._record_event("episode_done=invalid_action_limit_reached")
            logger.info("Episode terminated: invalid_action_limit_reached (%d)", self.stats.invalid_actions)
            return
        if self.num_active_leads == 0 and self.active_call is None:
            self.done = True
            self.termination_reason = "pipeline_exhausted"
            self._record_event("episode_done=pipeline_exhausted")
            logger.info("Episode terminated: pipeline_exhausted")

    def _record_event(self, event: str) -> None:
        self.event_log.append(event)
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]

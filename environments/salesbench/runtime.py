"""Stateful sales simulation runtime for Prime verifiers tooling."""

from __future__ import annotations

import logging
import os
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
    RuntimeActionError,
)
from policy import LLMBuyerPolicy, RuleBasedBuyerPolicy

logger = logging.getLogger("verifiers.salesbench")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _require_str(value: Any, name: str) -> str:
    if value is None:
        raise RuntimeActionError(f"{name} is required")
    return str(value).strip()


class SalesEpisodeRuntime:
    """Canonical owner of per-episode state."""

    def __init__(self, config: EpisodeConfig) -> None:
        config.validate()
        self.config = config
        self.catalog = ProductCatalog()
        if config.buyer_policy == "llm":
            self.policy: RuleBasedBuyerPolicy | LLMBuyerPolicy = LLMBuyerPolicy(
                model=config.buyer_model,
                base_url=config.buyer_base_url,
                api_key=os.getenv(config.buyer_api_key_var, ""),
            )
        else:
            self.policy = RuleBasedBuyerPolicy(seed=config.seed + 17)

        leads = LeadGenerator(seed=config.seed).generate(
            config.num_leads
        )
        self.leads: dict[str, Lead] = {lead.lead_id: lead for lead in leads}

        self.max_achievable_mrr = sum(lead.budget_monthly for lead in leads)
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
        self._pending_buyer_speech: str | None = None

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
        include_called = _coerce_bool(include_called)
        # Coerce filter args — LLMs often pass strings instead of ints/floats.
        if min_income is not None:
            min_income = int(min_income)
        if max_age is not None:
            max_age = int(max_age)
        if min_need is not None:
            min_need = float(min_need)
        limit = int(limit)
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
            "leads": [lead.to_search_dict() for lead in filtered],
            "remaining": self.num_active_leads,
        }

    def get_lead(self, *, lead_id: str) -> dict[str, Any]:
        lead = self._get_lead_or_error(lead_id)
        return {"lead": lead.to_detail_dict()}

    def add_note(self, *, lead_id: str, note: str) -> dict[str, Any]:
        note = _require_str(note, "note")
        if not note:
            raise RuntimeActionError("note cannot be empty")
        lead = self._get_lead_or_error(lead_id)
        lead.notes.append(note)
        return {
            "notes": len(lead.notes),
        }

    def pipeline_summary(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for lead in self.leads.values():
            status_counts.setdefault(lead.status.value, 0)
            status_counts[lead.status.value] += 1

        return {
            "status": status_counts,
            "time_left": max(0, self.config.max_minutes - self.current_minute),
            "minute": self.current_minute,
            "conversions": self.stats.conversions,
            "mrr": round(self.stats.revenue_mrr, 2),
            "calls": self.stats.calls_started,
            "offers": self.stats.offers_proposed,
            "done": self.done,
        }

    def schedule_callback(
        self,
        *,
        lead_id: str,
        hours_from_now: int,
        reason: str,
    ) -> dict[str, Any]:
        # Coerce args — LLMs often pass strings instead of ints.
        hours_from_now = int(hours_from_now)
        self._raise_if_done()
        lead = self._get_lead_or_error(lead_id)
        if lead.status != LeadStatus.ACTIVE or lead.do_not_call:
            raise RuntimeActionError("cannot schedule callback for inactive or DNC lead")
        if hours_from_now <= 0:
            raise RuntimeActionError("hours_from_now must be > 0")
        reason = _require_str(reason, "reason")
        if not reason:
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
            reason=reason,
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
            "cb_id": callback_id,
            "due_min": due_minute,
            "count": len(existing) + 1,
        }

    def list_callbacks(
        self,
        *,
        within_hours: int,
        include_completed: bool,
    ) -> dict[str, Any]:
        within_hours = int(within_hours)
        include_completed = _coerce_bool(include_completed)
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
            "callbacks": [task.to_compact_dict() for task in callbacks],
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
        if lead.call_count >= lead.max_calls and lead.status == LeadStatus.ACTIVE:
            lead.status = LeadStatus.EXHAUSTED
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
            "lead_id": lead_id,
            "name": lead.full_name,
            "temp": lead.temperature.value,
            "budget": round(lead.budget_monthly, 2),
        }

    def quote_plan(
        self,
        *,
        lead_id: str,
        plan_type: str,
        coverage_amount: int,
        term_years: int | None,
    ) -> dict[str, Any]:
        # Coerce args — LLMs often pass strings instead of ints.
        coverage_amount = int(coverage_amount)
        if term_years is not None:
            term_years = int(term_years)
        self._raise_if_done()
        lead = self._get_lead_or_error(lead_id)
        plan = self._parse_plan_type(plan_type)

        try:
            quote = self.catalog.quote(
                plan_type=plan,
                age=lead.age,
                coverage_amount=coverage_amount,
                risk_class=lead.risk_class,
                term_years=term_years,
            )
        except ValueError as exc:
            raise RuntimeActionError(str(exc)) from exc
        self._advance(self.config.tool_costs.quote_minutes, "quote_plan")

        affordability_ratio = quote["monthly_premium"] / max(lead.budget_monthly, 1.0)
        return {
            "premium": quote["monthly_premium"],
            "budget": round(lead.budget_monthly, 2),
            "ratio": round(affordability_ratio, 3),
        }

    async def conversation_turn(
        self, *, agent_text: str, messages: list | None = None
    ) -> str | None:
        """Process one agent->buyer conversation exchange. Returns buyer reply or None."""
        if self.active_call is None or self.done:
            return None
        if not agent_text.strip():
            return None

        lead = self._get_lead_or_error(self.active_call.lead_id)
        self._advance(self.config.tool_costs.send_message_minutes, "conversation")

        if self.done:
            return None  # time ran out during advance

        if isinstance(self.policy, LLMBuyerPolicy):
            buyer_reply = await self.policy.generate_response(
                lead=lead, agent_message=agent_text, messages=messages
            )
        else:
            buyer_reply = self.policy.generate_response(
                lead=lead, agent_message=agent_text
            )

        self.active_call.messages_sent += 1
        return buyer_reply

    async def propose_offer(
        self,
        *,
        plan_type: str,
        coverage_amount: int,
        monthly_premium: float,
        next_step: str,
        term_years: int | None,
        messages: list | None = None,
    ) -> dict[str, Any]:
        # Coerce args — LLMs often pass strings instead of ints/floats.
        coverage_amount = int(coverage_amount)
        monthly_premium = float(monthly_premium)
        if term_years is not None:
            term_years = int(term_years)
        self._raise_if_done()
        if self.active_call is None:
            raise RuntimeActionError("no active call")
        lead = self._get_lead_or_error(self.active_call.lead_id)
        if lead.status == LeadStatus.CONVERTED:
            raise RuntimeActionError("lead is already converted")
        next_step = _require_str(next_step, "next_step")
        if monthly_premium <= 0:
            raise RuntimeActionError("monthly_premium must be > 0")
        if coverage_amount <= 0:
            raise RuntimeActionError("coverage_amount must be > 0")
        if not next_step:
            raise RuntimeActionError("next_step cannot be empty")
        plan = self._parse_plan_type(plan_type)
        offer = Offer(
            plan_type=plan,
            coverage_amount=coverage_amount,
            monthly_premium=round(monthly_premium, 2),
            next_step=next_step,
            term_years=term_years,
        )

        self.active_call.offers.append(offer)
        self.stats.offers_proposed += 1
        self._advance(self.config.tool_costs.propose_offer_minutes, "propose_offer")

        # Time may have expired during _advance, which finalizes the active
        # call (setting it to None).  Return early to avoid AttributeError.
        if self.done:
            return {
                "decision": {"decision": "interrupted", "reason": "time expired"},
            }

        if isinstance(self.policy, LLMBuyerPolicy):
            decision = await self.policy.evaluate_offer(
                lead=lead, offer=offer, messages=messages
            )
        else:
            decision = self.policy.evaluate_offer(lead=lead, offer=offer)
        logger.debug(
            "Offer proposed: lead=%s plan=%s premium=%.2f decision=%s",
            lead.lead_id,
            plan.value,
            monthly_premium,
            decision.decision.value,
        )

        response: dict[str, Any] = {
            "decision": decision.to_dict(),
        }

        if decision.decision == BuyerDecision.ACCEPT:
            lead.status = LeadStatus.CONVERTED
            lead.accepted_offer = offer
            self.active_call.outcome = BuyerDecision.ACCEPT
            self.stats.conversions += 1
            self.stats.offers_accepted += 1
            self.stats.revenue_mrr += offer.monthly_premium
            response["msg"] = "Accepted. End call to finalize."
            logger.info(
                "Conversion: lead=%s premium=%.2f total_mrr=%.2f",
                lead.lead_id,
                offer.monthly_premium,
                self.stats.revenue_mrr,
            )

        elif decision.decision == BuyerDecision.REJECT:
            self.stats.rejected_offers += 1
            response["msg"] = "Rejected. Try a revised offer."

        elif decision.decision == BuyerDecision.HANG_UP:
            self.stats.hang_ups += 1
            self.active_call.outcome = BuyerDecision.HANG_UP
            response["msg"] = "Hung up."
            if decision.request_dnc:
                lead.status = LeadStatus.DNC
                lead.do_not_call = True
                response["msg"] += " DNC requested."
                logger.debug("Lead %s marked DNC after hang-up", lead.lead_id)
            self._finalize_active_call(reason="buyer_hang_up")

        # Store buyer's spoken response for conversation injection
        self._pending_buyer_speech = f"[{lead.full_name} (buyer)]: {decision.reason}"

        self._check_termination()
        return response

    def end_call(self, *, disposition: str) -> dict[str, Any]:
        if self.active_call is None:
            raise RuntimeActionError("no active call")
        disposition = _require_str(disposition, "disposition")
        if not disposition:
            raise RuntimeActionError("disposition cannot be empty")

        self._advance(self.config.tool_costs.end_call_minutes, "end_call")
        call = self._finalize_active_call(reason=disposition)
        self._check_termination()
        return {
            "call_id": call.call_id,
            "duration": call.duration_minutes,
            "outcome": call.outcome.value if call.outcome else None,
        }

    def render_briefing(self) -> str:
        n = len(self.leads)
        mins = self.config.max_minutes
        return f"Briefing: {n} leads, {mins}min budget. Maximize MRR. Adapt to lead temperature and archetype."

    def render_context_summary(self) -> str:
        """Deterministic context summary from runtime state (no LLM call).

        Used by get_prompt_messages to compress older messages when nearing
        the training max_seq_len. Includes KPIs, call history, active call
        status, and pipeline snapshot.
        """
        remaining = max(0, self.config.max_minutes - self.current_minute)
        lines = [
            "[CONTEXT SUMMARY — previous turns compressed]",
            (
                f"Time: {self.current_minute}/{self.config.max_minutes} min "
                f"({remaining} remaining) | "
                f"Revenue: ${self.stats.revenue_mrr:.2f}/mo | "
                f"Conversions: {self.stats.conversions} | "
                f"Offers: {self.stats.offers_proposed}"
            ),
        ]

        if self.call_history:
            lines.append(f"Calls completed ({len(self.call_history)}):")
            for session in self.call_history:
                lead = self.leads.get(session.lead_id)
                name = lead.full_name if lead else session.lead_id
                temp = lead.temperature.value if lead else "?"
                outcome = session.outcome.value if session.outcome else "no-outcome"
                lines.append(
                    f"  - {name} ({temp}): {outcome} — "
                    f"{len(session.offers)} offer(s), {session.duration_minutes}min"
                )

        if self.active_call:
            lead = self.leads.get(self.active_call.lead_id)
            name = lead.full_name if lead else self.active_call.lead_id
            lines.append(
                f"Active call: {name} — "
                f"{len(self.active_call.offers)} offer(s) so far"
            )

        scheduled = [
            t for t in self.callbacks.values()
            if t.status == CallbackStatus.SCHEDULED
        ]
        if scheduled:
            lines.append(f"Pending callbacks: {len(scheduled)}")

        lines.append(
            f"Pipeline: {self.num_active_leads} active leads, "
            f"{self.stats.leads_contacted} contacted"
        )
        return "\n".join(lines)

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
            "max_possible_mrr": round(self.max_achievable_mrr, 2),
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
        remaining = max(0, self.config.max_minutes - self.current_minute)
        pct = min(100.0, self.current_minute / max(1, self.config.max_minutes) * 100)
        logger.info(
            "[%d/%d min] (%.0f%%) %d remaining | action=%s | mrr=$%.2f convs=%d calls=%d",
            self.current_minute,
            self.config.max_minutes,
            pct,
            remaining,
            action,
            self.stats.revenue_mrr,
            self.stats.conversions,
            self.stats.calls_started,
        )
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
        if self.active_call is None:
            raise RuntimeActionError("no active call to finalize")
        session = self.active_call
        session.ended_minute = self.current_minute
        if session.outcome is None:
            session.outcome = BuyerDecision.REJECT
        session.notes.append(f"finalized:{reason}")
        self.call_history.append(session)
        self.active_call = None
        self.stats.calls_completed += 1
        return session

    def _finalize_if_active(self, reason: str) -> None:
        if self.active_call is not None:
            self._finalize_active_call(reason=reason)

    def _check_termination(self) -> None:
        if self.done:
            return
        if self.current_minute >= self.config.max_minutes:
            self.done = True
            self.termination_reason = "time_budget_exhausted"
            self._finalize_if_active("time_budget_exhausted")
            self._record_event("episode_done=time_budget_exhausted")
            logger.info("Episode terminated: time_budget_exhausted at minute %d", self.current_minute)
            return
        if self.stats.invalid_actions >= self.config.max_invalid_actions:
            self.done = True
            self.termination_reason = "invalid_action_limit_reached"
            self._finalize_if_active("invalid_action_limit_reached")
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

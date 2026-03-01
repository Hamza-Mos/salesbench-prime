"""Comprehensive tests for the SalesBench Prime RL environment."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure the salesbench package is importable when running from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import ProductCatalog
from config import DIFFICULTY_PRESETS, EpisodeConfig
from generator import LeadGenerator
from models import (
    BuyerDecision,
    CallSession,
    LeadStatus,
    Offer,
    PlanType,
    RiskClass,
    RuntimeActionError,
)
from policy import RuleBasedBuyerPolicy
from runtime import SalesEpisodeRuntime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> EpisodeConfig:
    defaults = {
        "seed": 42,
        "num_leads": 10,
        "work_days": 2,
        "hours_per_day": 8,
        "buyer_policy": "rule_based",
        "difficulty": "custom",
    }
    defaults.update(overrides)
    return EpisodeConfig(**defaults)


def _make_runtime(**overrides) -> SalesEpisodeRuntime:
    return SalesEpisodeRuntime(config=_make_config(**overrides))


# ---------------------------------------------------------------------------
# TestGeneratorDeterminism
# ---------------------------------------------------------------------------


class TestGeneratorDeterminism:
    def test_same_seed_same_leads(self):
        leads_a = LeadGenerator(seed=99).generate(20)
        leads_b = LeadGenerator(seed=99).generate(20)
        assert len(leads_a) == len(leads_b)
        for a, b in zip(leads_a, leads_b):
            assert a.lead_id == b.lead_id
            assert a.full_name == b.full_name
            assert a.age == b.age
            assert a.annual_income == b.annual_income
            assert a.latent_need == b.latent_need

    def test_different_seed_different_leads(self):
        leads_a = LeadGenerator(seed=1).generate(20)
        leads_b = LeadGenerator(seed=2).generate(20)
        names_a = {lead.full_name for lead in leads_a}
        names_b = {lead.full_name for lead in leads_b}
        # Very unlikely to be identical
        assert names_a != names_b or any(
            a.annual_income != b.annual_income for a, b in zip(leads_a, leads_b)
        )

    def test_difficulty_affects_generation(self):
        leads_easy = LeadGenerator(seed=42, difficulty="easy").generate(50)
        leads_hard = LeadGenerator(seed=42, difficulty="hard").generate(50)
        # Easy should have higher average latent need
        avg_need_easy = sum(l.latent_need for l in leads_easy) / len(leads_easy)
        avg_need_hard = sum(l.latent_need for l in leads_hard) / len(leads_hard)
        assert avg_need_easy > avg_need_hard

        # Easy should have higher average budget multiplier
        avg_budget_easy = sum(l.budget_monthly for l in leads_easy) / len(leads_easy)
        avg_budget_hard = sum(l.budget_monthly for l in leads_hard) / len(leads_hard)
        assert avg_budget_easy > avg_budget_hard


# ---------------------------------------------------------------------------
# TestCatalogPricing
# ---------------------------------------------------------------------------


class TestCatalogPricing:
    def setup_method(self):
        self.catalog = ProductCatalog()

    def test_quote_returns_premium(self):
        quote = self.catalog.quote(
            plan_type=PlanType.TERM,
            age=30,
            coverage_amount=500_000,
            risk_class=RiskClass.STANDARD,
            term_years=20,
        )
        assert quote["monthly_premium"] > 0
        assert quote["plan_type"] == "TERM"

    def test_preferred_cheaper_than_substandard(self):
        preferred = self.catalog.quote(
            plan_type=PlanType.TERM,
            age=40,
            coverage_amount=500_000,
            risk_class=RiskClass.PREFERRED,
            term_years=20,
        )
        substandard = self.catalog.quote(
            plan_type=PlanType.TERM,
            age=40,
            coverage_amount=500_000,
            risk_class=RiskClass.SUBSTANDARD,
            term_years=20,
        )
        assert preferred["monthly_premium"] < substandard["monthly_premium"]

    def test_age_out_of_range_raises(self):
        with pytest.raises(ValueError, match="age"):
            self.catalog.quote(
                plan_type=PlanType.DI,
                age=65,  # DI max is 60
                coverage_amount=5_000,
                risk_class=RiskClass.STANDARD,
                term_years=None,
            )

    def test_term_years_required_for_term(self):
        # When term_years is None for TERM, it defaults to 20
        quote = self.catalog.quote(
            plan_type=PlanType.TERM,
            age=30,
            coverage_amount=500_000,
            risk_class=RiskClass.STANDARD,
            term_years=None,
        )
        assert quote["term_years"] == 20

    def test_term_years_invalid_for_non_term(self):
        with pytest.raises(ValueError, match="term_years"):
            self.catalog.quote(
                plan_type=PlanType.WHOLE,
                age=30,
                coverage_amount=500_000,
                risk_class=RiskClass.STANDARD,
                term_years=20,
            )


# ---------------------------------------------------------------------------
# TestRuleBasedBuyerPolicy
# ---------------------------------------------------------------------------


class TestRuleBasedBuyerPolicy:
    def test_high_need_affordable_accepts(self):
        from models import Lead

        policy = RuleBasedBuyerPolicy(seed=42)
        lead = Lead(
            lead_id="test_001",
            full_name="Test Buyer",
            age=35,
            annual_income=120_000,
            state_code="CA",
            household_size=3,
            dependents=2,
            risk_class=RiskClass.PREFERRED,
            latent_need=0.90,
            trust_level=0.70,
            price_sensitivity=0.30,
            budget_monthly=200.0,
            max_calls=3,
            call_count=1,
        )
        offer = Offer(
            plan_type=PlanType.TERM,
            coverage_amount=960_000,  # ~8x income
            monthly_premium=60.0,  # Well under budget
            next_step="Sign application",
            term_years=20,
        )
        result = policy.evaluate_offer(lead=lead, offer=offer)
        assert result.decision == BuyerDecision.ACCEPT

    def test_unaffordable_rejects(self):
        from models import Lead

        policy = RuleBasedBuyerPolicy(seed=42)
        lead = Lead(
            lead_id="test_002",
            full_name="Test Buyer 2",
            age=30,
            annual_income=60_000,
            state_code="TX",
            household_size=2,
            dependents=1,
            risk_class=RiskClass.STANDARD,
            latent_need=0.80,
            trust_level=0.60,
            price_sensitivity=0.80,
            budget_monthly=80.0,
            max_calls=2,
            call_count=1,
        )
        offer = Offer(
            plan_type=PlanType.WHOLE,
            coverage_amount=500_000,
            monthly_premium=500.0,  # Way over 6% of monthly income
            next_step="Sign application",
        )
        result = policy.evaluate_offer(lead=lead, offer=offer)
        assert result.decision == BuyerDecision.REJECT

    def test_generate_response_returns_string(self):
        from models import Lead

        policy = RuleBasedBuyerPolicy(seed=42)
        lead = Lead(
            lead_id="test_003",
            full_name="Test Buyer 3",
            age=30,
            annual_income=80_000,
            state_code="CA",
            household_size=2,
            dependents=1,
            risk_class=RiskClass.STANDARD,
            latent_need=0.70,
            trust_level=0.50,
            price_sensitivity=0.50,
            budget_monthly=100.0,
            max_calls=3,
        )
        response = policy.generate_response(lead=lead, agent_message="Hello!")
        assert isinstance(response, str)
        assert len(response) > 0


# ---------------------------------------------------------------------------
# TestRuntimeStateTransitions
# ---------------------------------------------------------------------------


class TestRuntimeStateTransitions:
    def test_start_propose_end_flow(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]

        # Start call
        result = rt.start_call(lead_id=lead_id)
        assert result["call_id"] == "call_0001"
        assert rt.active_call is not None

        # Propose offer (async)
        propose_result = asyncio.new_event_loop().run_until_complete(
            rt.propose_offer(
                plan_type="TERM",
                coverage_amount=500_000,
                monthly_premium=50.0,
                next_step="sign",
                term_years=20,
            )
        )
        assert "decision" in propose_result

        # End call
        end_result = rt.end_call(disposition="completed")
        assert end_result["call"]["call_id"] == "call_0001"
        assert rt.active_call is None

    def test_double_start_raises(self):
        rt = _make_runtime()
        lead_ids = list(rt.leads.keys())

        rt.start_call(lead_id=lead_ids[0])
        with pytest.raises(RuntimeActionError, match="active call already exists"):
            rt.start_call(lead_id=lead_ids[1])

    def test_end_without_start_raises(self):
        rt = _make_runtime()
        with pytest.raises(RuntimeActionError, match="no active call"):
            rt.end_call(disposition="test")

    def test_propose_without_call_raises(self):
        rt = _make_runtime()
        with pytest.raises(RuntimeActionError):
            asyncio.new_event_loop().run_until_complete(
                rt.propose_offer(
                    plan_type="TERM",
                    coverage_amount=500_000,
                    monthly_premium=50.0,
                    next_step="sign",
                    term_years=20,
                )
            )


# ---------------------------------------------------------------------------
# TestRewardNormalization
# ---------------------------------------------------------------------------


class TestRewardNormalization:
    def test_max_achievable_mrr_positive(self):
        rt = _make_runtime()
        assert rt.max_achievable_mrr > 0

    def test_normalized_reward_in_range(self):
        rt = _make_runtime()
        # No conversions yet, reward should be 0
        normalized = rt.stats.revenue_mrr / max(1.0, rt.max_achievable_mrr)
        assert 0.0 <= normalized <= 1.0

    def test_normalized_reward_after_conversion(self):
        rt = _make_runtime()
        # Simulate a conversion
        lead = list(rt.leads.values())[0]
        rt.stats.revenue_mrr = lead.budget_monthly
        normalized = rt.stats.revenue_mrr / max(1.0, rt.max_achievable_mrr)
        assert 0.0 < normalized <= 1.0


# ---------------------------------------------------------------------------
# TestCallFinalizationOnTermination
# ---------------------------------------------------------------------------


class TestCallFinalizationOnTermination:
    def test_active_call_finalized_on_time_exhaustion(self):
        rt = _make_runtime(work_days=1, hours_per_day=1)  # 60 minute budget
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)
        assert rt.active_call is not None

        # Burn through time
        rt.current_minute = rt.config.max_minutes
        rt._check_termination()

        assert rt.done
        assert rt.termination_reason == "time_budget_exhausted"
        assert rt.active_call is None
        assert len(rt.call_history) == 1

    def test_active_call_finalized_on_invalid_action_limit(self):
        rt = _make_runtime(max_invalid_actions=2)
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        # Exceed invalid actions
        rt.stats.invalid_actions = 2
        rt._check_termination()

        assert rt.done
        assert rt.termination_reason == "invalid_action_limit_reached"
        assert rt.active_call is None


# ---------------------------------------------------------------------------
# TestExhaustedStatus
# ---------------------------------------------------------------------------


class TestExhaustedStatus:
    def test_lead_exhausted_when_max_calls_reached(self):
        rt = _make_runtime()
        # Find a lead with max_calls we can control
        lead_id = list(rt.leads.keys())[0]
        lead = rt.leads[lead_id]
        lead.max_calls = 1  # Will exhaust after one call

        rt.start_call(lead_id=lead_id)
        assert lead.status == LeadStatus.EXHAUSTED
        assert lead.call_count == 1

        # End the call
        rt.end_call(disposition="done")
        assert rt.active_call is None

    def test_lead_not_exhausted_below_max(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        lead = rt.leads[lead_id]
        lead.max_calls = 3

        rt.start_call(lead_id=lead_id)
        assert lead.status != LeadStatus.EXHAUSTED
        assert lead.call_count == 1


# ---------------------------------------------------------------------------
# TestConversationTurn
# ---------------------------------------------------------------------------


class TestConversationTurn:
    def test_conversation_turn_returns_buyer_response(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        result = asyncio.new_event_loop().run_until_complete(
            rt.conversation_turn(agent_text="Hello, how are you today?")
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_conversation_turn_increments_count(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        asyncio.new_event_loop().run_until_complete(
            rt.conversation_turn(agent_text="First message")
        )
        asyncio.new_event_loop().run_until_complete(
            rt.conversation_turn(agent_text="Second message")
        )

        assert rt.active_call is not None
        assert rt.active_call.messages_sent == 2

    def test_conversation_turn_without_call_returns_none(self):
        rt = _make_runtime()
        result = asyncio.new_event_loop().run_until_complete(
            rt.conversation_turn(agent_text="Hello")
        )
        assert result is None

    def test_conversation_turn_empty_returns_none(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        result = asyncio.new_event_loop().run_until_complete(
            rt.conversation_turn(agent_text="   ")
        )
        assert result is None

    def test_conversation_turn_advances_time(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)
        time_before = rt.current_minute

        asyncio.new_event_loop().run_until_complete(
            rt.conversation_turn(agent_text="Hello!")
        )

        assert rt.current_minute > time_before
        assert rt.current_minute == time_before + rt.config.tool_costs.send_message_minutes


# ---------------------------------------------------------------------------
# TestDifficultyPresets
# ---------------------------------------------------------------------------


class TestDifficultyPresets:
    def test_easy_preset_applies(self):
        cfg = EpisodeConfig.from_input(
            {"difficulty": "easy"},
            default_seed=42,
            default_num_leads=100,
            default_work_days=10,
            default_hours_per_day=8,
        )
        assert cfg.num_leads == DIFFICULTY_PRESETS["easy"]["num_leads"]
        assert cfg.work_days == DIFFICULTY_PRESETS["easy"]["work_days"]
        assert cfg.difficulty == "easy"

    def test_hard_preset_applies(self):
        cfg = EpisodeConfig.from_input(
            {"difficulty": "hard"},
            default_seed=42,
            default_num_leads=10,
            default_work_days=2,
            default_hours_per_day=8,
        )
        assert cfg.num_leads == DIFFICULTY_PRESETS["hard"]["num_leads"]
        assert cfg.work_days == DIFFICULTY_PRESETS["hard"]["work_days"]

    def test_custom_uses_defaults(self):
        cfg = EpisodeConfig.from_input(
            {"difficulty": "custom"},
            default_seed=42,
            default_num_leads=50,
            default_work_days=3,
            default_hours_per_day=6,
        )
        assert cfg.num_leads == 50
        assert cfg.work_days == 3
        assert cfg.hours_per_day == 6

    def test_difficulty_in_to_dict(self):
        cfg = _make_config(difficulty="easy")
        d = cfg.to_dict()
        assert d["difficulty"] == "easy"


# ---------------------------------------------------------------------------
# TestCallSessionModel
# ---------------------------------------------------------------------------


class TestCallSessionModel:
    def test_messages_sent_in_to_dict(self):
        session = CallSession(
            call_id="call_0001",
            lead_id="lead_0001",
            started_minute=0,
            messages_sent=3,
        )
        d = session.to_dict()
        assert d["messages_sent"] == 3

    def test_messages_sent_default_zero(self):
        session = CallSession(
            call_id="call_0001",
            lead_id="lead_0001",
            started_minute=0,
        )
        assert session.messages_sent == 0

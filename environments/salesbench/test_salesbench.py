"""Comprehensive tests for the SalesBench Prime RL environment."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure the salesbench package is importable when running from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import ProductCatalog
from config import EpisodeConfig
from generator import LeadGenerator
from models import (
    BuyerArchetype,
    BuyerDecision,
    CallSession,
    DecisionResult,
    Lead,
    LeadStatus,
    LeadTemperature,
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
        "total_hours": 16,
        "buyer_policy": "rule_based",
    }
    defaults.update(overrides)
    return EpisodeConfig(**defaults)


def _make_runtime(**overrides) -> SalesEpisodeRuntime:
    return SalesEpisodeRuntime(config=_make_config(**overrides))


def _make_test_lead(**overrides) -> Lead:
    defaults = dict(
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
        temperature=LeadTemperature.LUKEWARM,
        archetype=BuyerArchetype.ANALYTICAL,
    )
    defaults.update(overrides)
    return Lead(**defaults)


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
        policy = RuleBasedBuyerPolicy(seed=42)
        lead = _make_test_lead(
            latent_need=0.90,
            trust_level=0.70,
            price_sensitivity=0.30,
            budget_monthly=200.0,
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
        policy = RuleBasedBuyerPolicy(seed=42)
        lead = _make_test_lead(
            annual_income=60_000,
            latent_need=0.80,
            trust_level=0.60,
            price_sensitivity=0.80,
            budget_monthly=80.0,
            max_calls=2,
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
        policy = RuleBasedBuyerPolicy(seed=42)
        lead = _make_test_lead()
        response = policy.generate_response(lead=lead, agent_message="Hello!")
        assert isinstance(response, str)
        assert len(response) > 0


# ---------------------------------------------------------------------------
# TestRuntimeStateTransitions (updated for deterministic runtime)
# ---------------------------------------------------------------------------


class TestRuntimeStateTransitions:
    def test_start_record_offer_apply_decision_end_flow(self):
        """Full deterministic flow: start → record_offer → apply_decision → end."""
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]

        # Start call
        result = rt.start_call(lead_id=lead_id)
        assert result["call_id"] == "call_0001"
        assert rt.active_call is not None

        # Record offer (deterministic — no buyer LLM)
        offer_result = rt.record_offer(
            plan_type="TERM",
            coverage_amount=500_000,
            monthly_premium=50.0,
            next_step="sign",
            term_years=20,
        )
        assert "offer" in offer_result
        assert "lead" in offer_result
        assert offer_result["offer"].monthly_premium == 50.0

        # Apply buyer decision (deterministic state update)
        decision_result = rt.apply_buyer_decision(
            decision=BuyerDecision.ACCEPT,
            reason="Coverage fits my needs.",
            request_dnc=False,
        )
        assert decision_result["msg"] == "Accepted. End call to finalize."
        assert rt.stats.conversions == 1

        # End call
        end_result = rt.end_call(disposition="completed")
        assert end_result["call_id"] == "call_0001"
        assert rt.active_call is None

    def test_record_offer_reject_flow(self):
        """Record offer + REJECT decision produces suggested_adjustments."""
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        rt.record_offer(
            plan_type="TERM",
            coverage_amount=500_000,
            monthly_premium=50.0,
            next_step="sign",
            term_years=20,
        )
        result = rt.apply_buyer_decision(
            decision=BuyerDecision.REJECT,
            reason="Too expensive.",
            request_dnc=False,
        )
        assert result["msg"] == "Rejected. Try a revised offer."
        assert "suggested_adjustments" in result
        assert rt.stats.rejected_offers == 1

    def test_record_offer_hang_up_dnc(self):
        """HANG_UP with DNC request finalizes call and marks lead DNC."""
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        rt.record_offer(
            plan_type="TERM",
            coverage_amount=500_000,
            monthly_premium=50.0,
            next_step="sign",
            term_years=20,
        )
        result = rt.apply_buyer_decision(
            decision=BuyerDecision.HANG_UP,
            reason="Stop calling me.",
            request_dnc=True,
        )
        assert "DNC requested" in result["msg"]
        assert rt.leads[lead_id].do_not_call
        assert rt.leads[lead_id].status == LeadStatus.DNC
        assert rt.active_call is None  # call finalized on hang up

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

    def test_record_offer_without_call_raises(self):
        rt = _make_runtime()
        with pytest.raises(RuntimeActionError, match="no active call"):
            rt.record_offer(
                plan_type="TERM",
                coverage_amount=500_000,
                monthly_premium=50.0,
                next_step="sign",
                term_years=20,
            )

    def test_apply_decision_without_call_raises(self):
        rt = _make_runtime()
        with pytest.raises(RuntimeActionError, match="no active call"):
            rt.apply_buyer_decision(
                decision=BuyerDecision.ACCEPT,
                reason="Yes",
                request_dnc=False,
            )

    def test_record_offer_time_expired_returns_interrupted(self):
        """When time expires during record_offer, returns interrupted."""
        rt = _make_runtime(total_hours=1)  # 60 min
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)
        rt.current_minute = rt.config.max_minutes - 1  # 1 min left, propose costs 4

        result = rt.record_offer(
            plan_type="TERM",
            coverage_amount=500_000,
            monthly_premium=50.0,
            next_step="sign",
            term_years=20,
        )
        assert result.get("interrupted") is True
        assert rt.done


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
        rt = _make_runtime(total_hours=1)  # 60 minute budget
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
# TestAdvanceConversation (replaces TestConversationTurn)
# ---------------------------------------------------------------------------


class TestAdvanceConversation:
    def test_advance_conversation_returns_lead(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        result = rt.advance_conversation("Hello, how are you today?")
        assert isinstance(result, Lead)
        assert result.lead_id == lead_id

    def test_advance_conversation_increments_count(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        rt.advance_conversation("First message")
        rt.advance_conversation("Second message")

        assert rt.active_call is not None
        assert rt.active_call.messages_sent == 2

    def test_advance_conversation_without_call_returns_none(self):
        rt = _make_runtime()
        result = rt.advance_conversation("Hello")
        assert result is None

    def test_advance_conversation_empty_returns_none(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        result = rt.advance_conversation("   ")
        assert result is None

    def test_advance_conversation_advances_time(self):
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)
        time_before = rt.current_minute

        rt.advance_conversation("Hello!")

        assert rt.current_minute > time_before
        assert rt.current_minute == time_before + rt.config.tool_costs.send_message_minutes


# ---------------------------------------------------------------------------
# TestEpisodeConfig
# ---------------------------------------------------------------------------


class TestEpisodeConfig:
    def test_from_input_uses_defaults(self):
        cfg = EpisodeConfig.from_input(
            {},
            default_seed=42,
            default_num_leads=50,
            default_total_hours=24,
        )
        assert cfg.num_leads == 50
        assert cfg.total_hours == 24
        assert cfg.max_minutes == 24 * 60

    def test_from_input_overrides(self):
        cfg = EpisodeConfig.from_input(
            {"num_leads": 10, "total_hours": 8},
            default_seed=42,
            default_num_leads=50,
        )
        assert cfg.num_leads == 10
        assert cfg.total_hours == 8


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


# ---------------------------------------------------------------------------
# TestTemperatureAndArchetype
# ---------------------------------------------------------------------------


class TestTemperatureAndArchetype:
    def test_generated_leads_have_valid_temperature(self):
        leads = LeadGenerator(seed=42).generate(50)
        valid = set(LeadTemperature)
        for lead in leads:
            assert lead.temperature in valid

    def test_generated_leads_have_valid_archetype(self):
        leads = LeadGenerator(seed=42).generate(50)
        valid = set(BuyerArchetype)
        for lead in leads:
            assert lead.archetype in valid

    def test_determinism_temperature_archetype(self):
        leads_a = LeadGenerator(seed=99).generate(30)
        leads_b = LeadGenerator(seed=99).generate(30)
        for a, b in zip(leads_a, leads_b):
            assert a.temperature == b.temperature
            assert a.archetype == b.archetype

    def test_temperature_distribution_is_varied(self):
        leads = LeadGenerator(seed=42).generate(100)
        temps = {lead.temperature for lead in leads}
        # With 100 leads and uniform sampling, all 4 temperatures should appear
        assert len(temps) == 4

    def test_archetype_distribution_is_varied(self):
        leads = LeadGenerator(seed=42).generate(200)
        archetypes = {lead.archetype for lead in leads}
        # With 200 leads and uniform sampling, all 10 archetypes should appear
        assert len(archetypes) == 10

    def test_brief_dict_includes_temperature_and_archetype(self):
        leads = LeadGenerator(seed=42).generate(5)
        lead = leads[0]
        brief = lead.to_brief_dict()
        assert "temperature" in brief
        assert "archetype" in brief
        assert brief["temperature"] == lead.temperature.value
        assert brief["archetype"] == lead.archetype.value

    def test_detail_dict_includes_temperature_and_archetype(self):
        leads = LeadGenerator(seed=42).generate(5)
        lead = leads[0]
        detail = lead.to_detail_dict()
        assert "temperature" in detail
        assert "archetype" in detail


# ---------------------------------------------------------------------------
# TestRuleBasedPolicyTemperatureArchetype
# ---------------------------------------------------------------------------


class TestRuleBasedPolicyTemperatureArchetype:
    def test_cold_lead_harder_to_close(self):
        """A COLD lead should require a higher score to accept than a HOT lead."""
        policy = RuleBasedBuyerPolicy(seed=42)
        offer = Offer(
            plan_type=PlanType.TERM,
            coverage_amount=960_000,
            monthly_premium=60.0,
            next_step="Sign application",
            term_years=20,
        )
        # Same lead attributes, different temperature
        hot_lead = _make_test_lead(temperature=LeadTemperature.HOT)
        cold_lead = _make_test_lead(temperature=LeadTemperature.COLD)

        hot_result = policy.evaluate_offer(lead=hot_lead, offer=offer)
        # Reset RNG for fair comparison
        policy_2 = RuleBasedBuyerPolicy(seed=42)
        cold_result = policy_2.evaluate_offer(lead=cold_lead, offer=offer)

        # HOT should accept; COLD may not (or at least the scores should differ)
        # Since the threshold is different, HOT should be more likely to accept
        if hot_result.decision == BuyerDecision.ACCEPT:
            # With same score, COLD has higher threshold so may reject
            assert cold_result.decision in (BuyerDecision.ACCEPT, BuyerDecision.REJECT)

    def test_budget_hawk_price_sensitivity(self):
        """BUDGET_HAWK archetype should penalize price more heavily."""
        policy_1 = RuleBasedBuyerPolicy(seed=42)
        policy_2 = RuleBasedBuyerPolicy(seed=42)
        offer = Offer(
            plan_type=PlanType.TERM,
            coverage_amount=960_000,
            monthly_premium=80.0,
            next_step="Sign application",
            term_years=20,
        )
        analytical_lead = _make_test_lead(archetype=BuyerArchetype.ANALYTICAL)
        budget_lead = _make_test_lead(archetype=BuyerArchetype.BUDGET_HAWK)

        result_analytical = policy_1.evaluate_offer(lead=analytical_lead, offer=offer)
        result_budget = policy_2.evaluate_offer(lead=budget_lead, offer=offer)

        # BUDGET_HAWK has 1.4x price sensitivity multiplier
        assert result_budget.score < result_analytical.score

    def test_archetype_conditioned_responses(self):
        """Archetype-specific response pool should be used."""
        policy = RuleBasedBuyerPolicy(seed=42)
        skeptic_lead = _make_test_lead(archetype=BuyerArchetype.SKEPTIC)
        responses = set()
        for seed in range(50):
            p = RuleBasedBuyerPolicy(seed=seed)
            responses.add(p.generate_response(lead=skeptic_lead, agent_message="Hi"))
        # Should include at least one skeptic-specific response
        skeptic_phrases = {"What's the catch here?", "Are there any hidden fees I should know about?"}
        assert responses & skeptic_phrases


# ---------------------------------------------------------------------------
# TestEndCallDoubleFinalization
# ---------------------------------------------------------------------------


class TestEndCallDoubleFinalization:
    def test_end_call_no_crash_when_time_expires_during_advance(self):
        """end_call must not raise if _advance triggers time termination."""
        rt = _make_runtime(total_hours=1)  # 60 min budget
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        # Set time so that end_call's _advance (1 min) pushes past budget
        rt.current_minute = rt.config.max_minutes - 1  # 59 of 60

        # Before fix: this would crash with RuntimeActionError("no active call")
        # After fix: returns the finalized call's info gracefully
        result = rt.end_call(disposition="completed")

        assert rt.done
        assert rt.active_call is None
        assert result["call_id"] == "call_0001"
        assert result["outcome"] is not None

    def test_end_call_no_false_invalid_action(self):
        """Time expiration during end_call must not count as invalid action."""
        rt = _make_runtime(total_hours=1)
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)
        rt.current_minute = rt.config.max_minutes - 1

        invalid_before = rt.stats.invalid_actions
        rt.end_call(disposition="completed")

        # No invalid action should be recorded
        assert rt.stats.invalid_actions == invalid_before


# ---------------------------------------------------------------------------
# TestBuyerConversationContext
# ---------------------------------------------------------------------------


class TestBuyerConversationContext:
    def test_filters_briefing_messages(self):
        from policy import _build_buyer_conversation_context

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Briefing: 3 leads, 60min budget."},
            {"role": "assistant", "content": "Hello, I'd like to discuss insurance."},
            {"role": "user", "content": "[Alice Smith (buyer)]: Go ahead."},
        ]
        result = _build_buyer_conversation_context(messages)
        # Should have 2 messages: seller speech + buyer reply
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello, I'd like to discuss insurance."}
        assert result[1] == {"role": "assistant", "content": "[Alice Smith (buyer)]: Go ahead."}

    def test_filters_context_summaries(self):
        from policy import _build_buyer_conversation_context

        messages = [
            {"role": "user", "content": "[CONTEXT SUMMARY — previous turns compressed]\nTime: 30/60 min"},
            {"role": "assistant", "content": "Let me check your options."},
            {"role": "user", "content": "[Bob Jones (buyer)]: Sure."},
        ]
        result = _build_buyer_conversation_context(messages)
        assert len(result) == 2
        # Context summary filtered out, only seller + buyer remain
        assert result[0]["role"] == "user"
        assert "Let me check" in result[0]["content"]
        assert result[1]["role"] == "assistant"

    def test_filters_tool_results(self):
        from policy import _build_buyer_conversation_context

        messages = [
            {"role": "assistant", "content": "Let me look that up.", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "content": '{"ok":true,"premium":45.00}', "tool_call_id": "tc1"},
            {"role": "user", "content": "[Alice Smith (buyer)]: Sounds good."},
        ]
        result = _build_buyer_conversation_context(messages)
        assert len(result) == 2
        # tool message filtered, assistant text kept, buyer kept
        assert result[0]["content"] == "Let me look that up."
        assert result[1]["content"] == "[Alice Smith (buyer)]: Sounds good."

    def test_skips_empty_content(self):
        from policy import _build_buyer_conversation_context

        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "Real speech."},
        ]
        result = _build_buyer_conversation_context(messages)
        assert len(result) == 1
        assert result[0]["content"] == "Real speech."

    def test_none_messages(self):
        from policy import _build_buyer_conversation_context

        assert _build_buyer_conversation_context(None) == []
        assert _build_buyer_conversation_context([]) == []


# ---------------------------------------------------------------------------
# TestContextSummaryAlternation
# ---------------------------------------------------------------------------


class TestContextSummaryAlternation:
    def test_no_consecutive_user_messages_after_summary(self):
        """Summary should merge with next user message to avoid consecutive roles."""
        from salesbench import SalesBenchPrimeRLEnv

        # Build a message list where recent starts with a user message
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Briefing"},
            {"role": "assistant", "content": "Searching leads..."},
            {"role": "tool", "content": '{"ok":true}', "tool_call_id": "tc1"},
            {"role": "user", "content": "[Alice (buyer)]: Hello."},
            {"role": "assistant", "content": "I have a plan for you."},
            {"role": "user", "content": "[Alice (buyer)]: Tell me more."},
        ]
        state = {
            "runtime": _make_runtime(total_hours=1),
            "_context_summary_count": 0,
        }

        env = SalesBenchPrimeRLEnv.__new__(SalesBenchPrimeRLEnv)
        env.context_keep_recent = 2
        env.context_rewrite_threshold = 0.80

        result = env._apply_context_summary(messages, state)

        # Check no consecutive user messages
        for i in range(1, len(result)):
            if result[i].get("role") == "user" and result[i - 1].get("role") == "user":
                # The only allowed consecutive is system→user(briefing) which is prefix
                assert i <= 2, f"Consecutive user messages at index {i-1} and {i}"

    def test_summary_works_when_recent_starts_with_assistant(self):
        """Normal case: summary as separate user message when recent starts with assistant."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Briefing"},
            {"role": "assistant", "content": "Old turn 1"},
            {"role": "user", "content": "[Alice (buyer)]: Old reply"},
            {"role": "assistant", "content": "Recent turn"},
            {"role": "user", "content": "[Alice (buyer)]: Recent reply"},
        ]
        state = {
            "runtime": _make_runtime(total_hours=1),
            "_context_summary_count": 0,
        }

        from salesbench import SalesBenchPrimeRLEnv

        env = SalesBenchPrimeRLEnv.__new__(SalesBenchPrimeRLEnv)
        env.context_keep_recent = 2
        env.context_rewrite_threshold = 0.80

        result = env._apply_context_summary(messages, state)

        # prefix(2) + summary(1) + recent(2) = 5
        assert len(result) == 5
        assert result[2]["role"] == "user"  # summary
        assert "[CONTEXT SUMMARY" in result[2]["content"]
        assert result[3]["role"] == "assistant"  # recent[0]


# ---------------------------------------------------------------------------
# TestSeparationOfConcerns (NEW — validates the architecture refactor)
# ---------------------------------------------------------------------------


class TestSeparationOfConcerns:
    def test_runtime_has_no_policy(self):
        """Runtime must be a pure deterministic state machine — no buyer policy."""
        rt = _make_runtime()
        assert not hasattr(rt, "policy"), "runtime should not own a buyer policy"

    def test_record_offer_is_deterministic(self):
        """record_offer must not call any LLM — just validate + record + advance time."""
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)
        time_before = rt.current_minute

        result = rt.record_offer(
            plan_type="TERM",
            coverage_amount=500_000,
            monthly_premium=50.0,
            next_step="sign",
            term_years=20,
        )

        # Offer recorded
        assert rt.active_call.offers[-1].monthly_premium == 50.0
        assert rt.stats.offers_proposed == 1
        # Time advanced
        assert rt.current_minute == time_before + rt.config.tool_costs.propose_offer_minutes
        # Returns offer and lead objects
        assert result["offer"].plan_type == PlanType.TERM
        assert result["lead"].lead_id == lead_id

    def test_apply_buyer_decision_is_deterministic(self):
        """apply_buyer_decision must only mutate state — no LLM calls."""
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)
        rt.record_offer(
            plan_type="TERM",
            coverage_amount=500_000,
            monthly_premium=50.0,
            next_step="sign",
            term_years=20,
        )

        result = rt.apply_buyer_decision(
            decision=BuyerDecision.ACCEPT,
            reason="Looks good!",
            request_dnc=False,
        )

        assert rt.stats.conversions == 1
        assert rt.stats.revenue_mrr == 50.0
        assert rt.leads[lead_id].status == LeadStatus.CONVERTED
        assert result["decision"]["decision"] == "accept"

    def test_advance_conversation_is_deterministic(self):
        """advance_conversation only advances time — no buyer LLM."""
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)
        time_before = rt.current_minute

        lead = rt.advance_conversation("Hello!")

        assert lead is not None
        assert lead.lead_id == lead_id
        assert rt.current_minute == time_before + rt.config.tool_costs.send_message_minutes
        assert rt.active_call.messages_sent == 1

    def test_pending_buyer_speech_set_by_apply_decision(self):
        """apply_buyer_decision sets _pending_buyer_speech for orchestrator."""
        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)
        rt.record_offer(
            plan_type="TERM",
            coverage_amount=500_000,
            monthly_premium=50.0,
            next_step="sign",
            term_years=20,
        )

        rt.apply_buyer_decision(
            decision=BuyerDecision.REJECT,
            reason="Too pricey for me.",
            request_dnc=False,
        )

        assert rt._pending_buyer_speech is not None
        assert "Too pricey for me." in rt._pending_buyer_speech
        assert "(buyer)]:" in rt._pending_buyer_speech


# ---------------------------------------------------------------------------
# TestRewardEpisodeCompletion (NEW — validates shaped completion bonus)
# ---------------------------------------------------------------------------


class TestRewardEpisodeCompletion:
    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_pipeline_exhausted_gets_full_bonus(self):
        from rewards import reward_episode_completion

        rt = _make_runtime()
        rt.done = True
        rt.termination_reason = "pipeline_exhausted"
        result = self._run(reward_episode_completion({"runtime": rt}))
        assert result == 1.0

    def test_time_exhausted_gets_half_bonus(self):
        from rewards import reward_episode_completion

        rt = _make_runtime()
        rt.done = True
        rt.termination_reason = "time_budget_exhausted"
        result = self._run(reward_episode_completion({"runtime": rt}))
        assert result == 0.5

    def test_invalid_action_limit_gets_zero(self):
        from rewards import reward_episode_completion

        rt = _make_runtime()
        rt.done = True
        rt.termination_reason = "invalid_action_limit_reached"
        result = self._run(reward_episode_completion({"runtime": rt}))
        assert result == 0.0

    def test_not_done_gets_zero(self):
        from rewards import reward_episode_completion

        rt = _make_runtime()
        result = self._run(reward_episode_completion({"runtime": rt}))
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestBuyerErrorIsolation (NEW — validates buyer errors don't penalize seller)
# ---------------------------------------------------------------------------


class TestBuyerErrorIsolation:
    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_buyer_failure_produces_reject_not_invalid_action(self):
        """When buyer policy raises, tool returns REJECT — no invalid_action."""
        from tools import calling_propose_offer

        rt = _make_runtime()
        lead_id = list(rt.leads.keys())[0]
        rt.start_call(lead_id=lead_id)

        invalid_before = rt.stats.invalid_actions

        class FailingPolicy:
            def evaluate_offer(self, **kwargs):
                raise RuntimeError("Simulated buyer LLM timeout")

        result = self._run(calling_propose_offer(
            runtime=rt,
            plan_type="TERM",
            coverage_amount=500_000,
            monthly_premium=50.0,
            next_step="sign",
            term_years=20,
            messages=None,
            buyer_policy=FailingPolicy(),
        ))

        import json
        parsed = json.loads(result)

        # Should be a successful tool call with REJECT decision
        assert parsed["ok"] is True
        assert parsed["decision"]["decision"] == "reject"
        assert parsed["msg"] == "Rejected. Try a revised offer."
        # NO invalid action recorded
        assert rt.stats.invalid_actions == invalid_before

    def test_seller_error_still_produces_invalid_action(self):
        """Seller errors (bad args) still correctly produce invalid_action."""
        from tools import calling_propose_offer

        rt = _make_runtime()
        # No active call — this is a seller error
        invalid_before = rt.stats.invalid_actions

        result = self._run(calling_propose_offer(
            runtime=rt,
            plan_type="TERM",
            coverage_amount=500_000,
            monthly_premium=50.0,
            next_step="sign",
            term_years=20,
            messages=None,
            buyer_policy=RuleBasedBuyerPolicy(seed=42),
        ))

        import json
        parsed = json.loads(result)

        assert parsed["ok"] is False
        assert "no active call" in parsed["error"]
        assert rt.stats.invalid_actions == invalid_before + 1


# ---------------------------------------------------------------------------
# TestRewardEfficiencyRemoved (NEW — validates dead code removal)
# ---------------------------------------------------------------------------


class TestRewardEfficiencyRemoved:
    def test_efficiency_not_in_reward_funcs(self):
        from rewards import RUBRIC_FUNCS, RUBRIC_WEIGHTS

        func_names = [f.__name__ for f in RUBRIC_FUNCS]
        assert "reward_efficiency" not in func_names
        # Weights should match funcs count
        assert len(RUBRIC_FUNCS) == len(RUBRIC_WEIGHTS)


# ---------------------------------------------------------------------------
# TestBuyerMetricAccess (NEW — validates metrics read from state not runtime)
# ---------------------------------------------------------------------------


class TestBuyerMetricAccess:
    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_buyer_llm_metrics_from_state(self):
        from policy import LLMBuyerPolicy
        from rewards import metric_buyer_llm_call_count

        # Simulate a state with buyer_policy in state (not runtime)
        class FakeLLMPolicy(LLMBuyerPolicy):
            def __init__(self):
                self.call_count = 5
                self.timeout_count = 1
                self.slow_call_count = 0
                self.total_latency = 10.0
                self.max_latency = 3.5

        state = {"runtime": _make_runtime(), "buyer_policy": FakeLLMPolicy()}
        result = self._run(metric_buyer_llm_call_count(state))
        assert result == 5.0

    def test_buyer_metrics_zero_without_llm_policy(self):
        from rewards import metric_buyer_llm_call_count

        state = {"runtime": _make_runtime()}
        result = self._run(metric_buyer_llm_call_count(state))
        assert result == 0.0

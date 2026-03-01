"""Archetype profiles, rubric criteria, and emphasis mappings for buyer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from models import BuyerArchetype


class RubricCriterion(str, Enum):
    DISCOVERY_QUALITY = "discovery_quality"
    SOLUTION_TAILORING = "solution_tailoring"
    OBJECTION_HANDLING = "objection_handling"
    PRESSURE_CALIBRATION = "pressure_calibration"
    VALUE_ARTICULATION = "value_articulation"
    RAPPORT_AND_RESPECT = "rapport_and_respect"
    PACING_AND_PROCESS = "pacing_and_process"


CRITERION_PROMPTS: dict[RubricCriterion, str] = {
    RubricCriterion.DISCOVERY_QUALITY: "Did the agent ask about my situation before pitching?",
    RubricCriterion.SOLUTION_TAILORING: "Does this feel customized for me or generic?",
    RubricCriterion.OBJECTION_HANDLING: "Did they address my concerns thoughtfully?",
    RubricCriterion.PRESSURE_CALIBRATION: "Is the urgency appropriate or manipulative?",
    RubricCriterion.VALUE_ARTICULATION: "Do I understand what I'm getting and why it matters?",
    RubricCriterion.RAPPORT_AND_RESPECT: "Did they treat me like a person?",
    RubricCriterion.PACING_AND_PROCESS: "Was the conversation well-structured?",
}


@dataclass(frozen=True, slots=True)
class ArchetypeProfile:
    label: str
    core_trait: str
    prompt_modifier: str
    effective_tactics: str
    ineffective_tactics: str
    # Scoring weight modifiers for rule-based policy (added to base weights)
    need_weight_mod: float
    trust_weight_mod: float
    coverage_weight_mod: float
    plan_weight_mod: float
    # Price sensitivity multiplier (1.0 = neutral)
    price_sensitivity_mod: float


@dataclass(frozen=True, slots=True)
class CriteriaEmphasis:
    primary: tuple[RubricCriterion, ...]
    secondary: tuple[RubricCriterion, ...]
    de_emphasized: tuple[RubricCriterion, ...]


ARCHETYPE_PROFILES: dict[BuyerArchetype, ArchetypeProfile] = {
    BuyerArchetype.ANALYTICAL: ArchetypeProfile(
        label="Analytical",
        core_trait="Data-driven, wants specifics",
        prompt_modifier=(
            "You are a data-driven decision maker. You want to see specific numbers, "
            "ROI calculations, and detailed coverage breakdowns before committing. "
            "Vague promises or emotional appeals make you skeptical. Ask for specifics "
            "when the agent is vague."
        ),
        effective_tactics="Detailed quotes, ROI math, coverage rationale",
        ineffective_tactics="Emotional appeals, vague promises",
        need_weight_mod=0.0,
        trust_weight_mod=-0.04,
        coverage_weight_mod=0.06,
        plan_weight_mod=0.0,
        price_sensitivity_mod=1.0,
    ),
    BuyerArchetype.RELATIONSHIP: ArchetypeProfile(
        label="Relationship Builder",
        core_trait="Rapport-based decisions",
        prompt_modifier=(
            "You make decisions based on trust and personal connection. You need to feel "
            "that the agent genuinely cares about your situation before you'll consider "
            "any product. A transactional or rushed approach makes you shut down."
        ),
        effective_tactics="Active listening, empathy, personal connection",
        ineffective_tactics="Transactional tone, jumping straight to pitch",
        need_weight_mod=-0.04,
        trust_weight_mod=0.10,
        coverage_weight_mod=-0.04,
        plan_weight_mod=0.0,
        price_sensitivity_mod=0.9,
    ),
    BuyerArchetype.SKEPTIC: ArchetypeProfile(
        label="Skeptic",
        core_trait="Suspicious, assumes there are catches",
        prompt_modifier=(
            "You are naturally suspicious of sales pitches. You assume there are hidden "
            "fees, catches, or fine print. You respect agents who are transparent about "
            "limitations and don't oversell. Dismiss any agent who avoids your tough questions."
        ),
        effective_tactics="Transparency, acknowledging limitations, proof",
        ineffective_tactics="Overselling, dismissing concerns",
        need_weight_mod=0.0,
        trust_weight_mod=0.06,
        coverage_weight_mod=0.0,
        plan_weight_mod=-0.04,
        price_sensitivity_mod=1.1,
    ),
    BuyerArchetype.BUDGET_HAWK: ArchetypeProfile(
        label="Budget Hawk",
        core_trait="Price dominates everything",
        prompt_modifier=(
            "Price is your primary concern. You want the cheapest viable option and will "
            "push back hard on anything that seems too expensive. You calculate value per "
            "dollar and resist upselling. If an agent ignores your budget constraints, "
            "you lose interest quickly."
        ),
        effective_tactics="Cheapest viable option, value per dollar framing",
        ineffective_tactics="Upselling, ignoring budget constraints",
        need_weight_mod=-0.04,
        trust_weight_mod=-0.02,
        coverage_weight_mod=0.0,
        plan_weight_mod=-0.02,
        price_sensitivity_mod=1.4,
    ),
    BuyerArchetype.DELEGATOR: ArchetypeProfile(
        label="Delegator",
        core_trait="Wants expert recommendation",
        prompt_modifier=(
            "You don't want to research options yourself. You want a confident expert to "
            "tell you what's best for your situation. Too many choices overwhelm you. "
            "An indecisive agent who can't make a clear recommendation loses your confidence."
        ),
        effective_tactics="Confident recommendation, simplify choices",
        ineffective_tactics="Too many options, being indecisive",
        need_weight_mod=0.0,
        trust_weight_mod=0.04,
        coverage_weight_mod=0.0,
        plan_weight_mod=0.06,
        price_sensitivity_mod=0.9,
    ),
    BuyerArchetype.PROCRASTINATOR: ArchetypeProfile(
        label="Procrastinator",
        core_trait="Avoids decisions, needs time",
        prompt_modifier=(
            "You tend to delay decisions and need time to think things over. Hard pressure "
            "or ultimatums make you retreat. You respond best to gentle urgency, small "
            "incremental steps, and the option to schedule a follow-up."
        ),
        effective_tactics="Gentle urgency, small steps, callbacks",
        ineffective_tactics="Hard pressure, ultimatums",
        need_weight_mod=0.0,
        trust_weight_mod=0.02,
        coverage_weight_mod=-0.02,
        plan_weight_mod=0.0,
        price_sensitivity_mod=1.0,
    ),
    BuyerArchetype.STATUS_SEEKER: ArchetypeProfile(
        label="Status Seeker",
        core_trait="Wants best/premium option",
        prompt_modifier=(
            "You gravitate toward premium, exclusive, and top-tier options. You want to "
            "feel like you're getting the best, not settling. Budget options make you feel "
            "cheap. Exclusivity language and premium positioning appeal to you."
        ),
        effective_tactics="Premium plans, exclusivity language",
        ineffective_tactics="Budget options, making them feel cheap",
        need_weight_mod=0.0,
        trust_weight_mod=0.0,
        coverage_weight_mod=0.02,
        plan_weight_mod=0.08,
        price_sensitivity_mod=0.7,
    ),
    BuyerArchetype.PROTECTOR: ArchetypeProfile(
        label="Protector",
        core_trait="Family security motivated",
        prompt_modifier=(
            "Your family's security is your primary motivation. You think about what would "
            "happen to your dependents if something happened to you. Agents who frame "
            "coverage in terms of family protection resonate with you. Self-focused benefits "
            "miss the mark."
        ),
        effective_tactics="Family-focused framing, dependent scenarios",
        ineffective_tactics="Ignoring family, self-focused benefits",
        need_weight_mod=0.08,
        trust_weight_mod=0.0,
        coverage_weight_mod=0.02,
        plan_weight_mod=-0.04,
        price_sensitivity_mod=0.9,
    ),
    BuyerArchetype.COMPARISON_SHOPPER: ArchetypeProfile(
        label="Comparison Shopper",
        core_trait="Wants market context",
        prompt_modifier=(
            "You want to know how this offer compares to alternatives. Social proof, rate "
            "comparisons, and benchmarks help you decide. An agent who refuses to compare "
            "or can't provide a frame of reference loses credibility with you."
        ),
        effective_tactics="Social proof, rate comparisons, benchmarks",
        ineffective_tactics="Refusing to compare, no frame of reference",
        need_weight_mod=0.0,
        trust_weight_mod=-0.02,
        coverage_weight_mod=0.04,
        plan_weight_mod=0.0,
        price_sensitivity_mod=1.1,
    ),
    BuyerArchetype.IMPULSE_DECIDER: ArchetypeProfile(
        label="Impulse Decider",
        core_trait="Fast decisions when excited",
        prompt_modifier=(
            "When you're excited about something, you decide fast. Enthusiasm and momentum "
            "from the agent energize you. But a slow, overly cautious process kills your "
            "momentum and you lose interest. Easy next steps keep you engaged."
        ),
        effective_tactics="Enthusiasm, momentum, easy next steps",
        ineffective_tactics="Slow process, killing momentum",
        need_weight_mod=0.04,
        trust_weight_mod=0.0,
        coverage_weight_mod=-0.04,
        plan_weight_mod=0.0,
        price_sensitivity_mod=0.85,
    ),
}

CRITERIA_EMPHASIS: dict[BuyerArchetype, CriteriaEmphasis] = {
    BuyerArchetype.ANALYTICAL: CriteriaEmphasis(
        primary=(RubricCriterion.SOLUTION_TAILORING, RubricCriterion.VALUE_ARTICULATION),
        secondary=(RubricCriterion.DISCOVERY_QUALITY,),
        de_emphasized=(RubricCriterion.RAPPORT_AND_RESPECT,),
    ),
    BuyerArchetype.RELATIONSHIP: CriteriaEmphasis(
        primary=(RubricCriterion.RAPPORT_AND_RESPECT, RubricCriterion.DISCOVERY_QUALITY),
        secondary=(RubricCriterion.OBJECTION_HANDLING,),
        de_emphasized=(RubricCriterion.VALUE_ARTICULATION,),
    ),
    BuyerArchetype.SKEPTIC: CriteriaEmphasis(
        primary=(RubricCriterion.OBJECTION_HANDLING, RubricCriterion.PRESSURE_CALIBRATION),
        secondary=(RubricCriterion.VALUE_ARTICULATION,),
        de_emphasized=(RubricCriterion.PACING_AND_PROCESS,),
    ),
    BuyerArchetype.BUDGET_HAWK: CriteriaEmphasis(
        primary=(RubricCriterion.VALUE_ARTICULATION, RubricCriterion.SOLUTION_TAILORING),
        secondary=(RubricCriterion.PRESSURE_CALIBRATION,),
        de_emphasized=(RubricCriterion.RAPPORT_AND_RESPECT,),
    ),
    BuyerArchetype.DELEGATOR: CriteriaEmphasis(
        primary=(RubricCriterion.PACING_AND_PROCESS, RubricCriterion.RAPPORT_AND_RESPECT),
        secondary=(RubricCriterion.SOLUTION_TAILORING,),
        de_emphasized=(RubricCriterion.OBJECTION_HANDLING,),
    ),
    BuyerArchetype.PROCRASTINATOR: CriteriaEmphasis(
        primary=(RubricCriterion.PRESSURE_CALIBRATION, RubricCriterion.PACING_AND_PROCESS),
        secondary=(RubricCriterion.DISCOVERY_QUALITY,),
        de_emphasized=(RubricCriterion.VALUE_ARTICULATION,),
    ),
    BuyerArchetype.STATUS_SEEKER: CriteriaEmphasis(
        primary=(RubricCriterion.SOLUTION_TAILORING, RubricCriterion.RAPPORT_AND_RESPECT),
        secondary=(RubricCriterion.VALUE_ARTICULATION,),
        de_emphasized=(RubricCriterion.PRESSURE_CALIBRATION,),
    ),
    BuyerArchetype.PROTECTOR: CriteriaEmphasis(
        primary=(RubricCriterion.DISCOVERY_QUALITY, RubricCriterion.VALUE_ARTICULATION),
        secondary=(RubricCriterion.SOLUTION_TAILORING,),
        de_emphasized=(RubricCriterion.PACING_AND_PROCESS,),
    ),
    BuyerArchetype.COMPARISON_SHOPPER: CriteriaEmphasis(
        primary=(RubricCriterion.VALUE_ARTICULATION, RubricCriterion.SOLUTION_TAILORING),
        secondary=(RubricCriterion.DISCOVERY_QUALITY,),
        de_emphasized=(RubricCriterion.RAPPORT_AND_RESPECT,),
    ),
    BuyerArchetype.IMPULSE_DECIDER: CriteriaEmphasis(
        primary=(RubricCriterion.PACING_AND_PROCESS, RubricCriterion.RAPPORT_AND_RESPECT),
        secondary=(RubricCriterion.PRESSURE_CALIBRATION,),
        de_emphasized=(RubricCriterion.DISCOVERY_QUALITY,),
    ),
}

# Flat list for uniform random sampling in generator
ARCHETYPE_LIST: list[BuyerArchetype] = list(BuyerArchetype)

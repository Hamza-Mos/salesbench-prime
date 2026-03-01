"""Deterministic lead generation for the Prime RL harness."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from archetypes import ARCHETYPE_LIST
from models import BuyerArchetype, Lead, LeadTemperature, RiskClass

_FIRST_NAMES = [
    "Alex",
    "Jordan",
    "Taylor",
    "Morgan",
    "Cameron",
    "Riley",
    "Drew",
    "Parker",
    "Casey",
    "Avery",
    "Sam",
    "Dakota",
    "Jamie",
    "Quinn",
    "Emerson",
    "Skyler",
    "Reese",
    "Logan",
    "Blake",
    "Harper",
]

_LAST_NAMES = [
    "Smith",
    "Johnson",
    "Lee",
    "Brown",
    "Davis",
    "Martinez",
    "Miller",
    "Garcia",
    "Wilson",
    "Anderson",
    "Thomas",
    "Moore",
    "Clark",
    "Hernandez",
    "Walker",
    "King",
    "Scott",
    "Young",
    "Allen",
    "Rivera",
]

_STATES = [
    "CA",
    "TX",
    "FL",
    "NY",
    "WA",
    "IL",
    "CO",
    "NC",
    "GA",
    "AZ",
    "OH",
    "PA",
]

_SEGMENTS = (
    {
        "age": (26, 38),
        "income": (50_000, 130_000),
        "household": (2, 5),
        "need": (0.55, 0.95),
        "trust": (0.35, 0.75),
        "price": (0.45, 0.90),
        "max_calls": (2, 4),
        "risk_weights": (0.5, 0.4, 0.1),
    },
    {
        "age": (35, 52),
        "income": (80_000, 220_000),
        "household": (2, 4),
        "need": (0.45, 0.85),
        "trust": (0.40, 0.70),
        "price": (0.30, 0.65),
        "max_calls": (2, 5),
        "risk_weights": (0.4, 0.45, 0.15),
    },
    {
        "age": (44, 66),
        "income": (120_000, 350_000),
        "household": (1, 3),
        "need": (0.35, 0.75),
        "trust": (0.45, 0.80),
        "price": (0.20, 0.55),
        "max_calls": (1, 3),
        "risk_weights": (0.25, 0.50, 0.25),
    },
)

_RISK_CLASSES = (RiskClass.PREFERRED, RiskClass.STANDARD, RiskClass.SUBSTANDARD)

_TEMPERATURE_LIST = list(LeadTemperature)

_TEMPERATURE_MODIFIERS: dict[LeadTemperature, dict[str, float]] = {
    LeadTemperature.COLD: {"trust_delta": -0.10, "need_delta": -0.05, "max_calls_delta": -1},
    LeadTemperature.LUKEWARM: {"trust_delta": 0.00, "need_delta": 0.00, "max_calls_delta": 0},
    LeadTemperature.WARM: {"trust_delta": 0.05, "need_delta": 0.05, "max_calls_delta": 0},
    LeadTemperature.HOT: {"trust_delta": 0.10, "need_delta": 0.10, "max_calls_delta": 1},
}


_DIFFICULTY_SEGMENT_WEIGHTS: dict[str, tuple[float, ...]] = {
    "easy": (0.60, 0.30, 0.10),
    "hard": (0.15, 0.35, 0.50),
}

_BUDGET_RANGE: tuple[float, float] = (0.008, 0.040)


@dataclass(slots=True)
class LeadGenerator:
    """Generates a deterministic lead set from a seed."""

    seed: int
    difficulty: str = "custom"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def generate(self, num_leads: int) -> list[Lead]:
        segment_weights = _DIFFICULTY_SEGMENT_WEIGHTS.get(self.difficulty)

        leads: list[Lead] = []
        for idx in range(num_leads):
            if segment_weights is not None:
                segment = self._rng.choices(_SEGMENTS, weights=segment_weights, k=1)[0]
            else:
                segment = self._rng.choice(_SEGMENTS)
            age = self._rng.randint(*segment["age"])
            income = self._rng.randint(*segment["income"])
            household_size = self._rng.randint(*segment["household"])
            dependents = max(0, household_size - 1)
            latent_need = self._rng.uniform(*segment["need"])
            trust = self._rng.uniform(*segment["trust"])
            price = self._rng.uniform(*segment["price"])
            max_calls = self._rng.randint(*segment["max_calls"])

            risk_class = self._rng.choices(
                _RISK_CLASSES,
                weights=segment["risk_weights"],
                k=1,
            )[0]

            # Temperature — uniform 25/25/25/25
            temperature: LeadTemperature = self._rng.choice(_TEMPERATURE_LIST)
            mods = _TEMPERATURE_MODIFIERS[temperature]
            trust = max(0.0, min(1.0, trust + mods["trust_delta"]))
            latent_need = max(0.0, min(1.0, latent_need + mods["need_delta"]))
            max_calls = max(1, max_calls + int(mods["max_calls_delta"]))

            # Archetype — uniform across all 10
            archetype: BuyerArchetype = self._rng.choice(ARCHETYPE_LIST)

            # Budget — single wide range for even deal-size distribution
            budget_multiplier = self._rng.uniform(*_BUDGET_RANGE)
            budget_monthly = (income / 12.0) * budget_multiplier

            full_name = f"{self._rng.choice(_FIRST_NAMES)} {self._rng.choice(_LAST_NAMES)}"
            lead = Lead(
                lead_id=f"lead_{idx + 1:04d}",
                full_name=full_name,
                age=age,
                annual_income=income,
                state_code=self._rng.choice(_STATES),
                household_size=household_size,
                dependents=dependents,
                risk_class=risk_class,
                latent_need=latent_need,
                trust_level=trust,
                price_sensitivity=price,
                budget_monthly=budget_monthly,
                max_calls=max_calls,
                temperature=temperature,
                archetype=archetype,
            )
            leads.append(lead)
        return leads

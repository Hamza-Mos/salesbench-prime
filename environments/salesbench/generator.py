"""Deterministic lead generation for the Prime RL harness."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

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


@dataclass(slots=True)
class LeadGenerator:
    """Generates a deterministic lead set from a seed."""

    seed: int
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def generate(self, num_leads: int) -> list[Lead]:
        leads: list[Lead] = []
        for idx in range(num_leads):
            segment = self._rng.choice(_SEGMENTS)
            age = self._rng.randint(*segment["age"])
            income = self._rng.randint(*segment["income"])
            household_size = self._rng.randint(*segment["household"])
            dependents = max(0, household_size - 1)
            latent_need = self._rng.uniform(*segment["need"])
            trust = self._rng.uniform(*segment["trust"])
            price = self._rng.uniform(*segment["price"])
            budget_multiplier = self._rng.uniform(0.010, 0.030)
            budget_monthly = (income / 12.0) * budget_multiplier
            max_calls = self._rng.randint(*segment["max_calls"])

            risk_class = self._rng.choices(
                _RISK_CLASSES,
                weights=segment["risk_weights"],
                k=1,
            )[0]

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
            )
            leads.append(lead)
        return leads

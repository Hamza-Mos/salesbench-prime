"""Product catalog and deterministic pricing for the new harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models import PlanType, RiskClass


@dataclass(slots=True)
class PlanSpec:
    plan_type: PlanType
    name: str
    min_age: int
    max_age: int
    min_coverage: int
    max_coverage: int
    description: str
    term_options: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "plan_type": self.plan_type.value,
            "name": self.name,
            "min_age": self.min_age,
            "max_age": self.max_age,
            "min_coverage": self.min_coverage,
            "max_coverage": self.max_coverage,
            "description": self.description,
        }
        if self.term_options:
            payload["term_options"] = list(self.term_options)
        return payload


_AGE_BUCKETS: tuple[tuple[int, str], ...] = (
    (34, "25_34"),
    (44, "35_44"),
    (54, "45_54"),
    (64, "55_64"),
)


_RISK_MULTIPLIER = {
    RiskClass.PREFERRED: 0.85,
    RiskClass.STANDARD: 1.00,
    RiskClass.SUBSTANDARD: 1.35,
}


_TERM_MULTIPLIER = {
    10: 0.75,
    15: 0.88,
    20: 1.00,
    30: 1.28,
}


_BASE_RATES = {
    PlanType.TERM: {
        "25_34": 0.07,
        "35_44": 0.12,
        "45_54": 0.24,
        "55_64": 0.50,
        "65_plus": 1.05,
    },
    PlanType.WHOLE: {
        "25_34": 0.78,
        "35_44": 1.05,
        "45_54": 1.45,
        "55_64": 2.18,
        "65_plus": 3.25,
    },
    PlanType.UL: {
        "25_34": 0.40,
        "35_44": 0.56,
        "45_54": 0.80,
        "55_64": 1.15,
        "65_plus": 1.85,
    },
    PlanType.DI: {
        "25_34": 28.0,
        "35_44": 34.0,
        "45_54": 42.0,
        "55_64": 54.0,
        "65_plus": 68.0,
    },
}


class ProductCatalog:
    """Deterministic product metadata and premium quoting."""

    def __init__(self) -> None:
        self._specs = {
            PlanType.TERM: PlanSpec(
                plan_type=PlanType.TERM,
                name="Term Life",
                min_age=18,
                max_age=75,
                min_coverage=50_000,
                max_coverage=5_000_000,
                description="Low-cost temporary life coverage.",
                term_options=(10, 15, 20, 30),
            ),
            PlanType.WHOLE: PlanSpec(
                plan_type=PlanType.WHOLE,
                name="Whole Life",
                min_age=18,
                max_age=80,
                min_coverage=25_000,
                max_coverage=10_000_000,
                description="Permanent life coverage with cash value.",
            ),
            PlanType.UL: PlanSpec(
                plan_type=PlanType.UL,
                name="Universal Life",
                min_age=18,
                max_age=80,
                min_coverage=50_000,
                max_coverage=10_000_000,
                description="Flexible permanent life coverage.",
            ),
            PlanType.DI: PlanSpec(
                plan_type=PlanType.DI,
                name="Disability Income",
                min_age=18,
                max_age=60,
                min_coverage=1_000,
                max_coverage=15_000,
                description="Monthly income replacement if disabled.",
            ),
        }

    def list_plans(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self._specs.values()]

    def get_plan(self, plan_type: PlanType) -> PlanSpec:
        return self._specs[plan_type]

    def quote(
        self,
        *,
        plan_type: PlanType,
        age: int,
        coverage_amount: int,
        risk_class: RiskClass,
        term_years: int | None,
    ) -> dict[str, Any]:
        spec = self.get_plan(plan_type)
        if age < spec.min_age or age > spec.max_age:
            raise ValueError(
                f"age {age} outside allowed range {spec.min_age}-{spec.max_age} for {plan_type.value}"
            )
        if coverage_amount < spec.min_coverage or coverage_amount > spec.max_coverage:
            raise ValueError(
                "coverage_amount outside allowed range "
                f"{spec.min_coverage}-{spec.max_coverage} for {plan_type.value}"
            )
        if plan_type == PlanType.TERM:
            if term_years is None:
                term_years = 20
            if term_years not in _TERM_MULTIPLIER:
                raise ValueError("term_years must be one of 10, 15, 20, or 30 for TERM")
        elif term_years is not None:
            raise ValueError("term_years is only valid for TERM plans")

        bucket = self._get_age_bucket(age)
        base_rate = _BASE_RATES[plan_type][bucket]
        risk_multiplier = _RISK_MULTIPLIER[risk_class]

        premium = (coverage_amount / 1000.0) * base_rate * risk_multiplier
        if plan_type == PlanType.TERM and term_years is not None:
            premium *= _TERM_MULTIPLIER[term_years]

        premium = round(premium, 2)
        return {
            "plan_type": plan_type.value,
            "age": age,
            "coverage_amount": coverage_amount,
            "risk_class": risk_class.value,
            "term_years": term_years,
            "monthly_premium": premium,
        }

    def _get_age_bucket(self, age: int) -> str:
        for upper, bucket in _AGE_BUCKETS:
            if age <= upper:
                return bucket
        return "65_plus"

"""Configuration models for the SalesBench Prime RL harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCostConfig:
    """Simulated time costs (in minutes) for each tool action."""

    crm_search_minutes: int = 1
    quote_minutes: int = 1
    start_call_minutes: int = 1
    propose_offer_minutes: int = 4
    end_call_minutes: int = 1
    schedule_callback_minutes: int = 1


@dataclass(slots=True)
class EpisodeConfig:
    """Episode-level configuration for one rollout."""

    # Defaults aligned to original harness production preset.
    seed: int = 42
    num_leads: int = 100
    work_days: int = 10
    hours_per_day: int = 8
    max_invalid_actions: int = 12
    max_callbacks_per_lead: int = 2
    tool_costs: ToolCostConfig = field(default_factory=ToolCostConfig)

    @property
    def max_minutes(self) -> int:
        return self.work_days * self.hours_per_day * 60

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.num_leads <= 0:
            raise ValueError("num_leads must be > 0")
        if self.work_days <= 0:
            raise ValueError("work_days must be > 0")
        if self.hours_per_day <= 0:
            raise ValueError("hours_per_day must be > 0")
        if self.max_invalid_actions <= 0:
            raise ValueError("max_invalid_actions must be > 0")
        if self.max_callbacks_per_lead < 0:
            raise ValueError("max_callbacks_per_lead must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "num_leads": self.num_leads,
            "work_days": self.work_days,
            "hours_per_day": self.hours_per_day,
            "max_minutes": self.max_minutes,
            "max_invalid_actions": self.max_invalid_actions,
            "max_callbacks_per_lead": self.max_callbacks_per_lead,
            "tool_costs": {
                "crm_search_minutes": self.tool_costs.crm_search_minutes,
                "quote_minutes": self.tool_costs.quote_minutes,
                "start_call_minutes": self.tool_costs.start_call_minutes,
                "propose_offer_minutes": self.tool_costs.propose_offer_minutes,
                "end_call_minutes": self.tool_costs.end_call_minutes,
                "schedule_callback_minutes": self.tool_costs.schedule_callback_minutes,
            },
        }

    @classmethod
    def from_input(
        cls,
        data: dict[str, Any],
        *,
        default_seed: int,
        default_num_leads: int,
        default_work_days: int,
        default_hours_per_day: int,
    ) -> "EpisodeConfig":
        costs = ToolCostConfig(
            crm_search_minutes=int(data.get("crm_search_minutes", 1)),
            quote_minutes=int(data.get("quote_minutes", 1)),
            start_call_minutes=int(data.get("start_call_minutes", 1)),
            propose_offer_minutes=int(data.get("propose_offer_minutes", 4)),
            end_call_minutes=int(data.get("end_call_minutes", 1)),
            schedule_callback_minutes=int(data.get("schedule_callback_minutes", 1)),
        )
        cfg = cls(
            seed=int(data.get("seed", default_seed)),
            num_leads=int(data.get("num_leads", default_num_leads)),
            work_days=int(data.get("work_days", default_work_days)),
            hours_per_day=int(data.get("hours_per_day", default_hours_per_day)),
            max_invalid_actions=int(data.get("max_invalid_actions", 12)),
            max_callbacks_per_lead=int(data.get("max_callbacks_per_lead", 2)),
            tool_costs=costs,
        )
        cfg.validate()
        return cfg

"""Dataset construction for the SalesBench Prime environment."""

from __future__ import annotations

from datasets import Dataset

_SPLIT_OFFSETS: dict[str, int] = {
    "train": 0,
    "eval": 10_000,
    "test": 20_000,
}


def build_salesbench_dataset(
    *,
    split: str,
    num_examples: int,
    base_seed: int,
    base_num_leads: int,
    total_hours: int,
    buyer_policy: str = "llm",
    buyer_model: str = "openai/gpt-5-mini",
    buyer_base_url: str = "https://api.openai.com/v1",
    buyer_api_key_var: str = "OPENAI_API_KEY",
    difficulty: str = "custom",
) -> Dataset:
    """Build a deterministic HuggingFace `Dataset` for verifiers rollouts.

    Each example encodes an episode configuration (seed/leads/time budget).
    The verifiers base `Environment` will render the `question` field into the
    initial user message, and `SalesBenchPrimeRLEnv.setup_state` will read the
    config fields from the rollout input.
    """

    if num_examples <= 0:
        raise ValueError("num_examples must be > 0")

    normalized_split = split.strip().lower()
    offset = _SPLIT_OFFSETS.get(normalized_split, 0)

    rows: list[dict[str, object]] = []
    for idx in range(num_examples):
        rows.append(
            {
                "question": "Start the episode. Use tools every turn.",
                "split": normalized_split,
                "seed": int(base_seed + offset + idx),
                "num_leads": int(base_num_leads),
                "total_hours": int(total_hours),
                "buyer_policy": buyer_policy,
                "buyer_model": buyer_model,
                "buyer_base_url": buyer_base_url,
                "buyer_api_key_var": buyer_api_key_var,
                "difficulty": difficulty,
            }
        )

    return Dataset.from_list(rows)


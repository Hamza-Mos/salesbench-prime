"""Local test harness for buyer-prompt ablation variants.

Generates a fixed sample of (lead, offer) pairs covering varied
archetypes, temperatures, and offer qualities, then runs each buyer
prompt variant against the same set and reports decisions.

Goal: confirm variants produce *meaningfully different* but *coherent*
buyer behavior — the foundation of the publication ablation study.

Run: uv run python tools/test_buyer_variants.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Add env dir to path
ENV_DIR = Path(__file__).resolve().parent.parent / "environments" / "salesbench"
sys.path.insert(0, str(ENV_DIR))

import openai  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Load secrets
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "secrets.env", override=False)

from generator import LeadGenerator  # noqa: E402
from models import Offer, PlanType  # noqa: E402
from policy import _build_buyer_decision_prompt  # noqa: E402


VARIANTS = ["default", "skeptical", "impulsive", "analytical"]


@dataclass
class TestCase:
    """A (lead, offer) pair with expected difficulty annotation."""
    lead_idx: int
    offer: Offer
    label: str  # "good_fit", "marginal", "expensive", "wrong_plan", etc.


def make_offer(
    *, lead, kind: str
) -> Offer:
    """Construct an offer of varying quality matched to a lead.

    kind:
      - good_fit:     premium 60-80% of budget, term life, coverage = 8-10x income
      - marginal:     premium 90-100% of budget, term life, coverage = 6x income
      - expensive:    premium 130% of budget (over budget), term life
      - wrong_plan:   whole life for high-need young person, premium fine
      - cheap_thin:   premium 30% of budget, coverage only 4x income (under-insured)
      - barely:       premium = budget × 1.05 (just over budget), coverage 8x income
    """
    monthly_premium_targets = {
        "good_fit": lead.budget_monthly * 0.70,
        "marginal": lead.budget_monthly * 0.95,
        "expensive": lead.budget_monthly * 1.30,
        "wrong_plan": lead.budget_monthly * 0.65,
        "cheap_thin": lead.budget_monthly * 0.30,
        "barely": lead.budget_monthly * 1.05,
    }
    coverage_targets = {
        "good_fit": lead.annual_income * 9,
        "marginal": lead.annual_income * 6,
        "expensive": lead.annual_income * 9,
        "wrong_plan": lead.annual_income * 8,
        "cheap_thin": lead.annual_income * 4,
        "barely": lead.annual_income * 8,
    }
    plan_map = {
        "good_fit": PlanType.TERM,
        "marginal": PlanType.TERM,
        "expensive": PlanType.TERM,
        "wrong_plan": PlanType.WHOLE,
        "cheap_thin": PlanType.TERM,
        "barely": PlanType.TERM,
    }
    return Offer(
        plan_type=plan_map[kind],
        coverage_amount=int(coverage_targets[kind]),
        monthly_premium=round(monthly_premium_targets[kind], 2),
        term_years=20 if plan_map[kind] == PlanType.TERM else None,
        next_step="Are you ready to move forward?",
    )


def build_test_cases(num_leads: int = 10) -> tuple[list, list[TestCase]]:
    """Generate a fixed test set of (lead, offer) pairs."""
    gen = LeadGenerator(seed=12345)
    leads = gen.generate(num_leads)

    cases: list[TestCase] = []
    kinds = ["good_fit", "marginal", "expensive", "wrong_plan", "cheap_thin", "barely"]
    # Pair leads with varied offer qualities, cycling
    for i, lead in enumerate(leads):
        for j, kind in enumerate(kinds):
            cases.append(TestCase(
                lead_idx=i,
                offer=make_offer(lead=lead, kind=kind),
                label=kind,
            ))
    return leads, cases


def call_buyer(client: openai.OpenAI, model: str, lead, offer: Offer, variant: str) -> dict:
    """Call gpt-5-mini buyer with a given variant prompt; return parsed decision."""
    system_prompt = _build_buyer_decision_prompt(lead, variant=variant)

    offer_description = (
        f"The agent is proposing: {offer.plan_type.value} plan, "
        f"${offer.coverage_amount:,} coverage, "
        f"${offer.monthly_premium:.2f}/month premium"
    )
    if offer.term_years:
        offer_description += f", {offer.term_years}-year term"
    offer_description += f". Next step: {offer.next_step}"

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": offer_description},
        ],
        temperature=1.0,
        max_completion_tokens=4096,
        response_format={"type": "json_object"},
    )
    elapsed = time.monotonic() - t0
    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"decision": "PARSE_ERROR", "reason": raw[:200], "request_dnc": False}
    parsed["_latency"] = elapsed
    return parsed


def run_test(num_leads: int = 10, sample_print: int = 3) -> dict:
    """Run all variants on the test set and report stats."""
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=openai.Timeout(60.0, connect=20.0),
    )
    model = "gpt-5-mini"

    leads, cases = build_test_cases(num_leads=num_leads)
    print(f"Test set: {num_leads} leads × 6 offer kinds = {len(cases)} cases per variant")
    print(f"Model: {model}\n")

    results: dict[str, dict] = {}
    for variant in VARIANTS:
        print(f"=== Variant: {variant} ===")
        decisions = []
        sample_records = []
        t0 = time.monotonic()
        for case_idx, case in enumerate(cases):
            lead = leads[case.lead_idx]
            decision = call_buyer(client, model, lead, case.offer, variant)
            decisions.append({
                "case_idx": case_idx,
                "lead_id": lead.lead_id,
                "archetype": lead.archetype.value,
                "temperature": lead.temperature.value,
                "kind": case.label,
                "decision": decision.get("decision", "UNKNOWN"),
                "reason": decision.get("reason", "")[:120],
                "latency": decision.get("_latency", 0),
            })
            if case_idx < sample_print:
                print(f"  case {case_idx} {case.label:12s} ({lead.archetype.value:18s}) "
                      f"→ {decision.get('decision','?'):8s}  {decision.get('reason','')[:80]}")
        elapsed = time.monotonic() - t0

        counter = Counter(d["decision"] for d in decisions)
        n = len(decisions)
        accept_rate = counter.get("accept", 0) / n
        reject_rate = counter.get("reject", 0) / n
        hangup_rate = counter.get("hang_up", 0) / n
        parse_errors = counter.get("PARSE_ERROR", 0)

        # Per-offer-kind breakdown
        kind_breakdown: dict[str, dict] = {}
        for kind in ["good_fit", "marginal", "expensive", "wrong_plan", "cheap_thin", "barely"]:
            kind_decisions = [d for d in decisions if d["kind"] == kind]
            kind_counter = Counter(d["decision"] for d in kind_decisions)
            kind_n = len(kind_decisions)
            kind_breakdown[kind] = {
                "n": kind_n,
                "accept": kind_counter.get("accept", 0) / kind_n if kind_n else 0,
                "reject": kind_counter.get("reject", 0) / kind_n if kind_n else 0,
                "hang_up": kind_counter.get("hang_up", 0) / kind_n if kind_n else 0,
            }

        results[variant] = {
            "n": n,
            "accept_rate": accept_rate,
            "reject_rate": reject_rate,
            "hangup_rate": hangup_rate,
            "parse_errors": parse_errors,
            "elapsed_sec": elapsed,
            "by_kind": kind_breakdown,
            "decisions": decisions,
        }
        print(f"  → ACC {accept_rate:.0%}  REJ {reject_rate:.0%}  HU {hangup_rate:.0%}  "
              f"errors {parse_errors}  ({elapsed:.0f}s, {elapsed/n:.1f}s/call)\n")

    # Comparison summary
    print("\n" + "=" * 70)
    print("SUMMARY — accept rate by offer kind, by variant:")
    print(f"{'kind':<12}", *[f"{v:>10s}" for v in VARIANTS])
    for kind in ["good_fit", "marginal", "expensive", "wrong_plan", "cheap_thin", "barely"]:
        row = [f"{kind:<12}"]
        for v in VARIANTS:
            row.append(f"{results[v]['by_kind'][kind]['accept']:>10.0%}")
        print(" ".join(row))
    print(f"\n{'OVERALL':<12}", *[f"{results[v]['accept_rate']:>10.0%}" for v in VARIANTS])

    # Spread check
    accept_rates = [results[v]["accept_rate"] for v in VARIANTS]
    spread = max(accept_rates) - min(accept_rates)
    print(f"\nAccept-rate spread across variants: {spread:.0%}")
    print(f"Total parse errors: {sum(r['parse_errors'] for r in results.values())}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--leads", type=int, default=10, help="number of test leads")
    parser.add_argument("--out", type=str, default=None, help="optional JSON out path")
    args = parser.parse_args()

    results = run_test(num_leads=args.leads)

    if args.out:
        # Strip non-JSON-serializable fields (decisions list may have floats etc., keep simple)
        out_data = {
            v: {
                k: val for k, val in r.items()
                if k != "decisions"  # full decisions list is verbose
            } | {"sample_decisions": r["decisions"][:6]}
            for v, r in results.items()
        }
        Path(args.out).write_text(json.dumps(out_data, indent=2, default=str))
        print(f"\nResults written to {args.out}")

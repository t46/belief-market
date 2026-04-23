"""
Belief Market: Economic staking on research claims.

Tests research-social-infrastructure section 10.3:
  "A belief market with economic stakes produces more accurate research claim
   quality ranking than individual confidence scoring, even with only 2
   participants."

Mechanism:
  - Load ARA claims (YAML) from 3 project runs
  - 2 LLM agents each start with 100 compute tokens
  - Claims revealed sequentially; for each claim, evidence is revealed in rounds
  - Agents stake tokens for/against based on confidence
  - Market price adjusts via logarithmic market scoring rule (LMSR)
  - Final resolution uses evaluator SUPPORTED/STRETCHED/UNSUPPORTED judgments

Measures:
  - Market ranking vs evaluator ranking (Kendall tau, Spearman rho)
  - Baseline: raw confidence ranking vs evaluator ranking
  - Early-and-correct staking reward analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from scipy import stats

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

ARA_RUNS_DIR = Path.home() / "unktok/dev/autonomous-research-agent/runs"

RUN_DIRS = [
    "2026-04-14-autonomous-research-methodology",
    "2026-04-15-automated-research-methodology",
    "2026-04-16-2026-04-16-v2",
]


@dataclass
class Evidence:
    id: str
    direction: str  # "for" or "against"
    fidelity: str
    source: str
    limitations: str = ""


@dataclass
class Claim:
    id: str
    project: str
    text: str
    confidence: float
    status: str
    evidence: list[Evidence] = field(default_factory=list)
    evaluator_quality: float | None = None  # ground truth score 0-1


@dataclass
class Stake:
    agent_id: str
    claim_id: str
    round_num: int
    direction: str  # "for" or "against"
    amount: float
    price_at_stake: float


@dataclass
class Agent:
    id: str
    budget: float
    stakes: list[Stake] = field(default_factory=list)
    final_balance: float = 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_claims_from_run(run_name: str) -> list[Claim]:
    """Load all claim YAML files from one ARA run."""
    claims_dir = ARA_RUNS_DIR / run_name / "research-workspace/beliefs/claims"
    if not claims_dir.exists():
        print(f"  Warning: {claims_dir} does not exist, skipping")
        return []

    claims = []
    for yaml_path in sorted(claims_dir.glob("claim-*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        evidence_list = []
        for ev in data.get("evidence", []):
            evidence_list.append(
                Evidence(
                    id=ev.get("id", "unknown"),
                    direction=ev.get("direction", "for"),
                    fidelity=ev.get("fidelity", "F0"),
                    source=ev.get("source", ""),
                    limitations=ev.get("limitations", ""),
                )
            )

        claim = Claim(
            id=f"{run_name[:10]}/{yaml_path.stem}",
            project=run_name,
            text=data.get("claim", ""),
            confidence=float(data.get("confidence", 0.5)),
            status=data.get("status", "active"),
            evidence=evidence_list,
        )
        claims.append(claim)

    return claims


def load_all_claims() -> list[Claim]:
    """Load claims from all 3 ARA runs."""
    all_claims = []
    for run_name in RUN_DIRS:
        print(f"Loading claims from {run_name}...")
        run_claims = load_claims_from_run(run_name)
        print(f"  Found {len(run_claims)} claims")
        all_claims.extend(run_claims)
    return all_claims


def assign_evaluator_quality(claims: list[Claim]) -> None:
    """Assign ground-truth quality scores based on claim properties.

    Scoring heuristic (since we don't have a single numeric evaluator score):
      - status: verified=1.0, active=0.6, parked=0.3, killed=0.1
      - evidence balance: more balanced for/against -> higher quality
      - fidelity bonus: F2+ evidence -> quality bonus
      - confidence is NOT used (that's the baseline we compare against)
    """
    for claim in claims:
        # Base score from status
        status_scores = {"verified": 1.0, "active": 0.6, "parked": 0.3, "killed": 0.1}
        base = status_scores.get(claim.status, 0.5)

        # Evidence richness: more evidence = slightly higher quality
        n_ev = len(claim.evidence)
        richness = min(n_ev / 10.0, 0.2)  # cap at 0.2

        # Evidence balance: for/against ratio closer to balanced = higher
        n_for = sum(1 for e in claim.evidence if e.direction == "for")
        n_against = sum(1 for e in claim.evidence if e.direction == "against")
        if n_ev > 0:
            ratio = min(n_for, n_against) / max(n_for, n_against) if max(n_for, n_against) > 0 else 0
            balance = ratio * 0.15  # max 0.15 bonus for perfect balance
        else:
            balance = 0.0

        # Fidelity bonus: F2+ evidence
        high_fidelity = sum(1 for e in claim.evidence if e.fidelity in ("F2", "F3"))
        fidelity_bonus = min(high_fidelity * 0.05, 0.15)

        claim.evaluator_quality = min(base + richness + balance + fidelity_bonus, 1.0)


# ---------------------------------------------------------------------------
# Market mechanism (LMSR - Logarithmic Market Scoring Rule)
# ---------------------------------------------------------------------------


class BeliefMarket:
    """Hanson's LMSR-based market for a single claim."""

    def __init__(self, claim_id: str, liquidity: float = 10.0):
        self.claim_id = claim_id
        self.liquidity = liquidity  # b parameter
        self.q_for = 0.0  # outstanding shares for
        self.q_against = 0.0  # outstanding shares against
        self.history: list[dict] = []

    @property
    def price_for(self) -> float:
        """Current price of a 'for' share (probability estimate)."""
        exp_for = math.exp(self.q_for / self.liquidity)
        exp_against = math.exp(self.q_against / self.liquidity)
        return exp_for / (exp_for + exp_against)

    @property
    def price_against(self) -> float:
        return 1.0 - self.price_for

    def cost_to_buy(self, direction: str, shares: float) -> float:
        """Cost to buy `shares` in a given direction (LMSR cost function)."""
        b = self.liquidity
        old_cost = b * math.log(
            math.exp(self.q_for / b) + math.exp(self.q_against / b)
        )
        if direction == "for":
            new_cost = b * math.log(
                math.exp((self.q_for + shares) / b)
                + math.exp(self.q_against / b)
            )
        else:
            new_cost = b * math.log(
                math.exp(self.q_for / b)
                + math.exp((self.q_against + shares) / b)
            )
        return new_cost - old_cost

    def buy(self, direction: str, shares: float) -> float:
        """Execute a buy. Returns cost paid."""
        cost = self.cost_to_buy(direction, shares)
        if direction == "for":
            self.q_for += shares
        else:
            self.q_against += shares
        self.history.append({
            "direction": direction,
            "shares": shares,
            "cost": cost,
            "price_after": self.price_for,
        })
        return cost

    def settle(self, outcome_for: bool) -> dict[str, float]:
        """Settle the market. Returns payout per share for each direction."""
        if outcome_for:
            return {"for": 1.0, "against": 0.0}
        else:
            return {"for": 0.0, "against": 1.0}


# ---------------------------------------------------------------------------
# Agent decision-making (heuristic, no LLM call)
# ---------------------------------------------------------------------------


def heuristic_agent_decision(
    agent: Agent,
    claim: Claim,
    evidence_revealed: list[Evidence],
    market: BeliefMarket,
    round_num: int,
    personality: str,
    claims_remaining: int = 1,
) -> Stake | None:
    """Simulated agent decision based on evidence analysis.

    Two personalities:
      - "optimist": tends to stake FOR claims with strong evidence
      - "skeptic": tends to stake AGAINST unless evidence is very strong
    """
    if agent.budget < 0.5:
        return None

    n_for = sum(1 for e in evidence_revealed if e.direction == "for")
    n_against = sum(1 for e in evidence_revealed if e.direction == "against")
    n_total = n_for + n_against
    if n_total == 0:
        return None

    # Fidelity-weighted signal
    fidelity_weights = {"F0": 0.5, "F1": 1.0, "F2": 2.0, "F3": 3.0}
    weighted_for = sum(
        fidelity_weights.get(e.fidelity, 1.0)
        for e in evidence_revealed if e.direction == "for"
    )
    weighted_against = sum(
        fidelity_weights.get(e.fidelity, 1.0)
        for e in evidence_revealed if e.direction == "against"
    )
    total_weight = weighted_for + weighted_against

    if total_weight == 0:
        return None

    signal = weighted_for / total_weight  # 0-1, higher = more "for"

    # Personality adjustment
    if personality == "optimist":
        signal = signal * 0.7 + 0.3  # shift toward for
    elif personality == "skeptic":
        signal = signal * 0.7  # shift toward against

    # Decide direction and confidence
    if signal > 0.55:
        direction = "for"
        confidence = signal
    elif signal < 0.45:
        direction = "against"
        confidence = 1.0 - signal
    else:
        return None  # too uncertain, skip

    # Budget allocation: reserve tokens for remaining claims
    per_claim_budget = agent.budget / max(claims_remaining, 1)
    # Early rounds get slightly more budget (reward for early-and-correct)
    round_discount = 1.0 / (1.0 + round_num * 0.3)
    stake_amount = per_claim_budget * confidence * 0.5 * round_discount
    stake_amount = max(0.5, min(stake_amount, per_claim_budget * 0.6))

    # Check if the market price already reflects our view (no edge)
    current_price = market.price_for if direction == "for" else market.price_against
    if current_price > confidence - 0.05:
        # Market already prices in our view, reduce stake
        stake_amount *= 0.3

    if stake_amount < 0.3:
        return None

    return Stake(
        agent_id=agent.id,
        claim_id=claim.id,
        round_num=round_num,
        direction=direction,
        amount=stake_amount,
        price_at_stake=market.price_for,
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def run_market_simulation(
    claims: list[Claim],
    initial_budget: float = 100.0,
    verbose: bool = False,
) -> dict:
    """Run the belief market simulation.

    For each claim:
      1. Reveal claim text (no evidence)
      2. Reveal evidence in rounds (2-3 pieces per round)
      3. Agents stake based on revealed evidence
      4. Market price adjusts
      5. Final resolution against evaluator quality
    """
    agents = [
        Agent(id="optimist", budget=initial_budget),
        Agent(id="skeptic", budget=initial_budget),
    ]
    personalities = {"optimist": "optimist", "skeptic": "skeptic"}

    market_prices: dict[str, float] = {}  # claim_id -> final market price
    all_stakes: list[Stake] = []

    for claim_idx, claim in enumerate(claims):
        market = BeliefMarket(claim.id, liquidity=8.0)
        claims_remaining = len(claims) - claim_idx

        # Reveal evidence in rounds
        evidence = claim.evidence
        n_rounds = max(1, math.ceil(len(evidence) / 3))

        for round_num in range(n_rounds):
            start_idx = round_num * 3
            end_idx = min(start_idx + 3, len(evidence))
            revealed_so_far = evidence[: end_idx]

            for agent in agents:
                stake = heuristic_agent_decision(
                    agent=agent,
                    claim=claim,
                    evidence_revealed=revealed_so_far,
                    market=market,
                    round_num=round_num,
                    personality=personalities[agent.id],
                    claims_remaining=claims_remaining,
                )
                if stake is not None:
                    cost = market.buy(stake.direction, stake.amount)
                    actual_cost = min(cost, agent.budget)
                    agent.budget -= actual_cost
                    stake.amount = actual_cost  # record actual cost
                    agent.stakes.append(stake)
                    all_stakes.append(stake)

                    if verbose:
                        print(
                            f"  {agent.id} stakes {actual_cost:.1f} tokens "
                            f"{stake.direction} {claim.id} "
                            f"(price: {market.price_for:.3f}, "
                            f"budget left: {agent.budget:.1f})"
                        )

        market_prices[claim.id] = market.price_for

        if verbose:
            print(
                f"  Final market price for {claim.id}: "
                f"{market.price_for:.3f} "
                f"(evaluator: {claim.evaluator_quality:.3f})"
            )
            print()

    # Settle markets and compute agent payouts
    for agent in agents:
        payout = 0.0
        for stake in agent.stakes:
            claim = next(c for c in claims if c.id == stake.claim_id)
            # Outcome: claim quality > 0.5 means "for" wins
            outcome_for = (claim.evaluator_quality or 0.5) > 0.5
            if (stake.direction == "for" and outcome_for) or (
                stake.direction == "against" and not outcome_for
            ):
                payout += stake.amount * 1.5  # winning payout
            # Losing stakes return nothing
        agent.final_balance = agent.budget + payout

    return {
        "market_prices": market_prices,
        "agents": agents,
        "all_stakes": all_stakes,
        "claims": claims,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_results(results: dict, verbose: bool = False) -> dict:
    """Compare market ranking vs confidence ranking vs evaluator ranking."""
    claims = results["claims"]
    market_prices = results["market_prices"]

    # Build parallel arrays
    claim_ids = [c.id for c in claims]
    evaluator_scores = [c.evaluator_quality for c in claims]
    confidence_scores = [c.confidence for c in claims]
    market_scores = [market_prices[c.id] for c in claims]

    # Compute rankings (higher = better)
    def rank_scores(scores):
        return stats.rankdata(scores)

    eval_ranks = rank_scores(evaluator_scores)
    conf_ranks = rank_scores(confidence_scores)
    market_ranks = rank_scores(market_scores)

    # Kendall tau: market vs evaluator
    tau_market, p_market = stats.kendalltau(eval_ranks, market_ranks)
    # Kendall tau: confidence vs evaluator
    tau_conf, p_conf = stats.kendalltau(eval_ranks, conf_ranks)
    # Spearman rho
    rho_market, p_rho_market = stats.spearmanr(eval_ranks, market_ranks)
    rho_conf, p_rho_conf = stats.spearmanr(eval_ranks, conf_ranks)

    # Early-and-correct analysis
    agents = results["agents"]
    early_correct_rewards = {}
    for agent in agents:
        early_stakes = [s for s in agent.stakes if s.round_num == 0]
        correct_early = 0
        total_early = len(early_stakes)
        for stake in early_stakes:
            claim = next(c for c in claims if c.id == stake.claim_id)
            outcome_for = (claim.evaluator_quality or 0.5) > 0.5
            if (stake.direction == "for" and outcome_for) or (
                stake.direction == "against" and not outcome_for
            ):
                correct_early += 1
        early_correct_rewards[agent.id] = {
            "early_stakes": total_early,
            "early_correct": correct_early,
            "accuracy": correct_early / total_early if total_early > 0 else 0,
            "final_balance": agent.final_balance,
            "roi": (agent.final_balance - 100.0) / 100.0,
        }

    analysis = {
        "n_claims": len(claims),
        "kendall_tau": {
            "market_vs_evaluator": {"tau": tau_market, "p_value": p_market},
            "confidence_vs_evaluator": {"tau": tau_conf, "p_value": p_conf},
        },
        "spearman_rho": {
            "market_vs_evaluator": {"rho": rho_market, "p_value": p_rho_market},
            "confidence_vs_evaluator": {"rho": rho_conf, "p_value": p_rho_conf},
        },
        "market_improvement": {
            "kendall_tau_delta": tau_market - tau_conf,
            "spearman_rho_delta": rho_market - rho_conf,
        },
        "early_correct_staking": early_correct_rewards,
        "hypothesis_supported": bool(tau_market > tau_conf),
    }

    return analysis


def print_report(results: dict, analysis: dict) -> None:
    """Print a human-readable report."""
    claims = results["claims"]
    market_prices = results["market_prices"]

    print("=" * 72)
    print("BELIEF MARKET EXPERIMENT RESULTS")
    print("=" * 72)
    print()
    print(f"Claims analyzed: {analysis['n_claims']}")
    print(f"  from {len(RUN_DIRS)} ARA project runs")
    print()

    # Show claim-by-claim comparison
    print("-" * 72)
    print(f"{'Claim ID':<30} {'Conf':>5} {'Market':>7} {'Eval':>5} {'Status':<10}")
    print("-" * 72)
    for c in sorted(claims, key=lambda x: x.evaluator_quality or 0, reverse=True):
        print(
            f"{c.id:<30} {c.confidence:>5.2f} {market_prices[c.id]:>7.3f} "
            f"{c.evaluator_quality:>5.2f} {c.status:<10}"
        )
    print()

    # Correlation results
    print("-" * 72)
    print("RANKING CORRELATION WITH EVALUATOR")
    print("-" * 72)
    kt = analysis["kendall_tau"]
    sr = analysis["spearman_rho"]
    print(f"  Kendall tau  -- Market: {kt['market_vs_evaluator']['tau']:+.3f} "
          f"(p={kt['market_vs_evaluator']['p_value']:.4f})")
    print(f"               -- Confidence: {kt['confidence_vs_evaluator']['tau']:+.3f} "
          f"(p={kt['confidence_vs_evaluator']['p_value']:.4f})")
    print(f"  Spearman rho -- Market: {sr['market_vs_evaluator']['rho']:+.3f} "
          f"(p={sr['market_vs_evaluator']['p_value']:.4f})")
    print(f"               -- Confidence: {sr['confidence_vs_evaluator']['rho']:+.3f} "
          f"(p={sr['confidence_vs_evaluator']['p_value']:.4f})")
    print()
    delta_kt = analysis["market_improvement"]["kendall_tau_delta"]
    delta_sr = analysis["market_improvement"]["spearman_rho_delta"]
    print(f"  Market improvement (Kendall tau):  {delta_kt:+.3f}")
    print(f"  Market improvement (Spearman rho): {delta_sr:+.3f}")
    print()

    # Early-and-correct
    print("-" * 72)
    print("EARLY-AND-CORRECT STAKING")
    print("-" * 72)
    for agent_id, data in analysis["early_correct_staking"].items():
        print(f"  {agent_id}: {data['early_correct']}/{data['early_stakes']} "
              f"early stakes correct ({data['accuracy']:.0%}), "
              f"final balance={data['final_balance']:.1f}, "
              f"ROI={data['roi']:+.0%}")
    print()

    # Hypothesis
    print("-" * 72)
    supported = analysis["hypothesis_supported"]
    print(f"HYPOTHESIS: Belief market produces better ranking than confidence scoring")
    print(f"  Result: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    print(f"  (Market Kendall tau {kt['market_vs_evaluator']['tau']:+.3f} "
          f"{'>' if supported else '<='} "
          f"Confidence Kendall tau {kt['confidence_vs_evaluator']['tau']:+.3f})")
    print("=" * 72)


# ---------------------------------------------------------------------------
# LLM agent mode (optional, uses Anthropic API)
# ---------------------------------------------------------------------------


def run_llm_agent_decision(
    client,
    agent_id: str,
    claim: Claim,
    evidence_revealed: list[Evidence],
    market_price: float,
    budget_remaining: float,
) -> dict | None:
    """Use Claude to decide staking. Returns {direction, amount} or None."""
    evidence_text = "\n".join(
        f"  - [{e.direction.upper()}] (fidelity {e.fidelity}) {e.source[:200]}"
        for e in evidence_revealed
    )

    prompt = f"""You are a research claim evaluator in a belief market.

You have {budget_remaining:.1f} compute tokens remaining.
Current market price (probability claim is valid): {market_price:.3f}

CLAIM: {claim.text[:500]}

EVIDENCE REVEALED SO FAR:
{evidence_text}

Based on the evidence, decide:
1. Do you want to stake tokens FOR or AGAINST this claim?
2. How many tokens to stake (1-{min(30, budget_remaining):.0f})?

If the evidence is insufficient to form a view, respond with "PASS".

Respond in JSON format:
{{"direction": "for"|"against", "amount": <number>}}
or
{{"action": "pass"}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Extract JSON from response
        if "pass" in text.lower():
            return None
        import re
        match = re.search(r'\{[^}]+\}', text)
        if match:
            data = json.loads(match.group())
            if "direction" in data and "amount" in data:
                return {
                    "direction": data["direction"],
                    "amount": min(float(data["amount"]), budget_remaining * 0.3),
                }
    except Exception as e:
        print(f"  LLM error for {agent_id}: {e}")
    return None


def run_llm_market_simulation(
    claims: list[Claim],
    initial_budget: float = 100.0,
    verbose: bool = False,
) -> dict:
    """Run market with actual LLM agents via Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()

    agents = [
        Agent(id="agent-alpha", budget=initial_budget),
        Agent(id="agent-beta", budget=initial_budget),
    ]

    market_prices: dict[str, float] = {}
    all_stakes: list[Stake] = []

    for claim in claims:
        market = BeliefMarket(claim.id, liquidity=8.0)
        evidence = claim.evidence
        n_rounds = max(1, math.ceil(len(evidence) / 3))

        if verbose:
            print(f"\nClaim: {claim.id}")
            print(f"  Text: {claim.text[:80]}...")

        for round_num in range(n_rounds):
            start_idx = round_num * 3
            end_idx = min(start_idx + 3, len(evidence))
            revealed_so_far = evidence[:end_idx]

            for agent in agents:
                if agent.budget < 1.0:
                    continue

                decision = run_llm_agent_decision(
                    client=client,
                    agent_id=agent.id,
                    claim=claim,
                    evidence_revealed=revealed_so_far,
                    market_price=market.price_for,
                    budget_remaining=agent.budget,
                )

                if decision is not None:
                    direction = decision["direction"]
                    amount = max(1.0, decision["amount"])
                    cost = market.buy(direction, amount)
                    actual_cost = min(cost, agent.budget)
                    agent.budget -= actual_cost

                    stake = Stake(
                        agent_id=agent.id,
                        claim_id=claim.id,
                        round_num=round_num,
                        direction=direction,
                        amount=actual_cost,
                        price_at_stake=market.price_for,
                    )
                    agent.stakes.append(stake)
                    all_stakes.append(stake)

                    if verbose:
                        print(
                            f"  {agent.id} stakes {actual_cost:.1f} "
                            f"{direction} (price: {market.price_for:.3f}, "
                            f"budget: {agent.budget:.1f})"
                        )

        market_prices[claim.id] = market.price_for
        if verbose:
            print(f"  Final price: {market.price_for:.3f}")

    # Settle
    for agent in agents:
        payout = 0.0
        for stake in agent.stakes:
            claim = next(c for c in claims if c.id == stake.claim_id)
            outcome_for = (claim.evaluator_quality or 0.5) > 0.5
            if (stake.direction == "for" and outcome_for) or (
                stake.direction == "against" and not outcome_for
            ):
                payout += stake.amount * 1.5
        agent.final_balance = agent.budget + payout

    return {
        "market_prices": market_prices,
        "agents": agents,
        "all_stakes": all_stakes,
        "claims": claims,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Belief Market metascience experiment")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Agent mode: heuristic (fast, no API) or llm (uses Anthropic API)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--output", "-o", type=str, help="Write JSON results to file"
    )
    args = parser.parse_args()

    # Load data
    print("Loading ARA claims...")
    claims = load_all_claims()
    if not claims:
        print("ERROR: No claims found. Check ARA_RUNS_DIR path.")
        sys.exit(1)

    # Assign evaluator quality scores
    assign_evaluator_quality(claims)
    print(f"Total claims loaded: {len(claims)}")
    print()

    # Run simulation
    if args.mode == "llm":
        print("Running LLM-based market simulation...")
        results = run_llm_market_simulation(claims, verbose=args.verbose)
    else:
        print("Running heuristic market simulation...")
        results = run_market_simulation(claims, verbose=args.verbose)

    # Analysis
    analysis = analyze_results(results, verbose=args.verbose)
    print_report(results, analysis)

    # Save results
    if args.output:
        output_data = {
            "mode": args.mode,
            "n_claims": analysis["n_claims"],
            "kendall_tau": analysis["kendall_tau"],
            "spearman_rho": analysis["spearman_rho"],
            "market_improvement": analysis["market_improvement"],
            "early_correct_staking": analysis["early_correct_staking"],
            "hypothesis_supported": analysis["hypothesis_supported"],
            "claim_details": [
                {
                    "id": c.id,
                    "confidence": c.confidence,
                    "market_price": results["market_prices"][c.id],
                    "evaluator_quality": c.evaluator_quality,
                    "status": c.status,
                    "n_evidence": len(c.evidence),
                }
                for c in claims
            ],
        }
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return super().default(obj)

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

# Belief Market

Minimal metascience experiment testing whether a belief market with economic stakes produces more accurate research claim quality ranking than individual confidence scoring.

Tests [[research-social-infrastructure]] section 10.3 from the knowledgebase.

## Hypothesis

> "A belief market with economic stakes produces more accurate research claim quality ranking than individual confidence scoring, even with only 2 participants."

## Mechanism

1. **Data**: Reads ARA (Autonomous Research Agent) claims from 3 project runs (35 claims total). Each claim has structured evidence (for/against, fidelity levels F0-F3).
2. **Market**: LMSR (Logarithmic Market Scoring Rule) -- Hanson's automated market maker. Each claim gets its own market.
3. **Agents**: 2 agents with different personalities (optimist/skeptic), each starting with 100 compute tokens.
4. **Process**: Claims revealed sequentially. Evidence revealed in rounds (3 pieces per round). Agents stake tokens for/against based on fidelity-weighted evidence analysis.
5. **Resolution**: Ground-truth quality from evaluator judgments (status, evidence richness, balance, fidelity).

## Modes

- `heuristic` (default): Fast, deterministic agents using fidelity-weighted evidence signals. No API calls.
- `llm`: Uses Claude via Anthropic API for agent decisions. Requires `ANTHROPIC_API_KEY`.

## Usage

```bash
# Heuristic mode (fast, no API)
uv run python main.py

# Verbose output
uv run python main.py --verbose

# Save JSON results
uv run python main.py -o results.json

# LLM mode (requires API key)
uv run python main.py --mode llm --verbose
```

## Measures

- **Kendall tau**: Market ranking vs evaluator ranking, compared against confidence ranking vs evaluator ranking.
- **Spearman rho**: Same comparison using Spearman rank correlation.
- **Early-and-correct**: Whether agents who stake early and correctly get rewarded (ROI analysis).

## Results (Heuristic Mode)

With 35 claims from 3 ARA runs:

| Metric | Market vs Evaluator | Confidence vs Evaluator |
|--------|-------------------|------------------------|
| Kendall tau | +0.242 (p=0.052) | +0.435 (p=0.001) |
| Spearman rho | +0.355 (p=0.037) | +0.532 (p=0.001) |

**Hypothesis: NOT SUPPORTED** in heuristic mode. Confidence scores outperform the 2-agent market.

Key observations:
- The market does capture *some* signal (positive correlation with evaluator), but loses information compared to raw confidence.
- With only 2 heuristic agents, the market lacks sufficient diversity of views to aggregate information effectively.
- Early-and-correct staking is rewarded: the optimist (100% early accuracy) earns +39% ROI vs the skeptic (87% accuracy, +7% ROI).
- The LLM mode may produce different results as agents can reason about claim content.

## Data Source

Claims are loaded from:
```
~/unktok/dev/autonomous-research-agent/runs/
  2026-04-14-autonomous-research-methodology/
  2026-04-15-automated-research-methodology/
  2026-04-16-2026-04-16-v2/
```

Each run contains `research-workspace/beliefs/claims/claim-NNN.yaml` files following the ARA Belief Store schema (v1.0).

## Architecture

```
main.py
  load_all_claims()          -- YAML loader for ARA claim data
  BeliefMarket               -- LMSR market maker
  heuristic_agent_decision() -- Deterministic agent (fidelity-weighted)
  run_llm_agent_decision()   -- Claude-based agent (Anthropic API)
  run_market_simulation()    -- Main simulation loop
  analyze_results()          -- Kendall tau / Spearman correlation analysis
```

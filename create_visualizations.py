"""Create publication-quality visualizations for the Belief Market experiment."""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Load results
with open(Path(__file__).parent / "results.json") as f:
    data = json.load(f)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# -- Style setup --
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "market": "#2563EB",     # blue
    "confidence": "#DC2626", # red
    "evaluator": "#059669",  # green
    "optimist": "#F59E0B",   # amber
    "skeptic": "#8B5CF6",    # purple
    "neutral": "#6B7280",    # gray
    "sig": "#16A34A",        # green for significance
    "nonsig": "#DC2626",     # red for non-significance
}

claims = data["claim_details"]
n_claims = len(claims)


# =========================================================================
# Chart 1: Market vs Confidence vs Evaluator Ranking Comparison
# =========================================================================
def chart_ranking_comparison():
    from scipy.stats import rankdata

    # Sort claims by evaluator quality for display
    sorted_claims = sorted(claims, key=lambda c: c["evaluator_quality"], reverse=True)

    ids = [c["id"].split("/")[1] for c in sorted_claims]
    eval_scores = [c["evaluator_quality"] for c in sorted_claims]
    conf_scores = [c["confidence"] for c in sorted_claims]
    mkt_scores = [c["market_price"] for c in sorted_claims]

    # Use ranks instead of raw scores for clearer comparison
    eval_ranks = rankdata([-s for s in eval_scores], method="min")
    conf_ranks = rankdata([-s for s in conf_scores], method="min")
    mkt_ranks = rankdata([-s for s in mkt_scores], method="min")

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

    y_pos = np.arange(n_claims)
    bar_h = 0.65

    # Panel 1: Evaluator (ground truth)
    bars = axes[0].barh(y_pos, eval_scores, height=bar_h, color=COLORS["evaluator"], alpha=0.85)
    axes[0].set_title("Evaluator Score\n(Ground Truth)", fontweight="bold")
    axes[0].set_xlim(0, 1.1)
    axes[0].set_xlabel("Score")

    # Panel 2: Confidence
    bars = axes[1].barh(y_pos, conf_scores, height=bar_h, color=COLORS["confidence"], alpha=0.85)
    axes[1].set_title("Raw Confidence\n(tau=0.435, p=0.001)", fontweight="bold")
    axes[1].set_xlim(0, 1.1)
    axes[1].set_xlabel("Score")

    # Panel 3: Market Price
    bars = axes[2].barh(y_pos, mkt_scores, height=bar_h, color=COLORS["market"], alpha=0.85)
    axes[2].set_title("Market Price\n(tau=0.242, p=0.052)", fontweight="bold")
    axes[2].set_xlim(0, 1.1)
    axes[2].set_xlabel("Score")

    # Y labels on left panel only
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ids, fontsize=7)
    axes[0].invert_yaxis()

    fig.suptitle("Claim Scores: Evaluator vs Confidence vs Market Price",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02,
             "Claims sorted by evaluator score (descending). Confidence tracks evaluator better than market price.",
             ha="center", fontsize=10, color="#666")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ranking_comparison.png")
    plt.close()
    print("Saved: ranking_comparison.png")


# =========================================================================
# Chart 2: Agent Portfolio Over Time (simulated from stakes)
# =========================================================================
def chart_agent_portfolio():
    """Reconstruct agent balance over claim rounds from stake data."""
    # We need to re-run parts of the simulation to get per-claim balance snapshots
    # Since we have claim_details + early_correct_staking, we'll simulate the trajectory

    # Re-run simulation to extract per-claim budgets
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from main import (load_all_claims, assign_evaluator_quality,
                      run_market_simulation)

    claims_data = load_all_claims()
    assign_evaluator_quality(claims_data)

    # Run simulation with tracking
    from main import Agent, BeliefMarket, heuristic_agent_decision

    initial_budget = 100.0
    agents = [
        Agent(id="optimist", budget=initial_budget),
        Agent(id="skeptic", budget=initial_budget),
    ]
    personalities = {"optimist": "optimist", "skeptic": "skeptic"}

    # Track balance after each claim
    optimist_balance = [initial_budget]
    skeptic_balance = [initial_budget]
    claim_labels = ["Start"]

    for claim_idx, claim in enumerate(claims_data):
        market = BeliefMarket(claim.id, liquidity=8.0)
        claims_remaining = len(claims_data) - claim_idx
        evidence = claim.evidence
        n_rounds = max(1, math.ceil(len(evidence) / 3))

        for round_num in range(n_rounds):
            end_idx = min((round_num + 1) * 3, len(evidence))
            revealed = evidence[:end_idx]

            for agent in agents:
                stake = heuristic_agent_decision(
                    agent=agent, claim=claim, evidence_revealed=revealed,
                    market=market, round_num=round_num,
                    personality=personalities[agent.id],
                    claims_remaining=claims_remaining,
                )
                if stake is not None:
                    cost = market.buy(stake.direction, stake.amount)
                    actual_cost = min(cost, agent.budget)
                    agent.budget -= actual_cost
                    stake.amount = actual_cost
                    agent.stakes.append(stake)

        # Record balance after this claim (staking costs deducted)
        optimist_balance.append(agents[0].budget)
        skeptic_balance.append(agents[1].budget)
        short_id = claim.id.split("/")[1][-3:]
        claim_labels.append(short_id)

    # After settlement: compute final with payouts
    for agent in agents:
        payout = 0.0
        for stake in agent.stakes:
            c = next(cl for cl in claims_data if cl.id == stake.claim_id)
            outcome_for = (c.evaluator_quality or 0.5) > 0.5
            if (stake.direction == "for" and outcome_for) or \
               (stake.direction == "against" and not outcome_for):
                payout += stake.amount * 1.5
        agent.final_balance = agent.budget + payout

    optimist_balance.append(agents[0].final_balance)
    skeptic_balance.append(agents[1].final_balance)
    claim_labels.append("Settlement")

    x = np.arange(len(optimist_balance))

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x, optimist_balance, color=COLORS["optimist"], linewidth=2.2,
            marker="", label="Optimist Agent", zorder=3)
    ax.plot(x, skeptic_balance, color=COLORS["skeptic"], linewidth=2.2,
            marker="", label="Skeptic Agent", zorder=3)

    # Shade the settlement region
    ax.axvspan(len(x)-2, len(x)-1, alpha=0.12, color=COLORS["sig"], zorder=1)
    ax.annotate("Settlement\npayout", xy=(len(x)-1.5, max(optimist_balance[-1], skeptic_balance[-1])+3),
                fontsize=9, ha="center", color=COLORS["sig"], fontweight="bold")

    # Starting line
    ax.axhline(y=100, color="#999", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0.5, 101.5, "Initial budget: 100 tokens", fontsize=8, color="#999")

    # Final annotations
    ax.annotate(f"Final: {agents[0].final_balance:.1f}\n(+{agents[0].final_balance-100:.0f}% ROI)",
                xy=(len(x)-1, optimist_balance[-1]),
                xytext=(len(x)-5, optimist_balance[-1]+12),
                arrowprops=dict(arrowstyle="->", color=COLORS["optimist"]),
                fontsize=9, color=COLORS["optimist"], fontweight="bold")
    ax.annotate(f"Final: {agents[1].final_balance:.1f}\n(+{agents[1].final_balance-100:.0f}% ROI)",
                xy=(len(x)-1, skeptic_balance[-1]),
                xytext=(len(x)-5, skeptic_balance[-1]-15),
                arrowprops=dict(arrowstyle="->", color=COLORS["skeptic"]),
                fontsize=9, color=COLORS["skeptic"], fontweight="bold")

    # Thin x-tick labels (show every 5th + start/end)
    tick_positions = [0] + list(range(5, len(x)-1, 5)) + [len(x)-1]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([claim_labels[i] for i in tick_positions], rotation=45, ha="right")

    ax.set_xlabel("Claim Round")
    ax.set_ylabel("Token Balance")
    ax.set_title("Agent Portfolio Over Time: Optimist vs Skeptic", fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim(bottom=min(min(optimist_balance), min(skeptic_balance)) - 10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "agent_portfolio.png")
    plt.close()
    print("Saved: agent_portfolio.png")


# =========================================================================
# Chart 3: Market Price Evolution for Selected Claims
# =========================================================================
def chart_price_evolution():
    """Show market price evolution as evidence is revealed for interesting claims."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from main import (load_all_claims, assign_evaluator_quality,
                      Agent, BeliefMarket, heuristic_agent_decision)

    claims_data = load_all_claims()
    assign_evaluator_quality(claims_data)

    # Pick interesting claims:
    # - Highest evidence count (claim-009 from run 14, 22 evidence pieces)
    # - A verified claim with high confidence (claim-001 from run 16)
    # - A parked claim (claim-006 from run 14)
    # - A claim where market and confidence diverge most

    interesting_ids = [
        "2026-04-14/claim-009",  # 22 evidence pieces, rich trajectory
        "2026-04-16/claim-001",  # verified, conf=0.98
        "2026-04-14/claim-006",  # parked, low evaluator score
        "2026-04-15/claim-012",  # high evaluator, active
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    colors_rounds = plt.cm.viridis(np.linspace(0.2, 0.9, 10))

    for ax_idx, target_id in enumerate(interesting_ids):
        ax = axes[ax_idx]
        claim = next(c for c in claims_data if c.id == target_id)

        # Run simulation just for this claim, recording price after each round
        agents = [
            Agent(id="optimist", budget=100.0),
            Agent(id="skeptic", budget=100.0),
        ]
        personalities = {"optimist": "optimist", "skeptic": "skeptic"}
        market = BeliefMarket(claim.id, liquidity=8.0)

        evidence = claim.evidence
        n_rounds = max(1, math.ceil(len(evidence) / 3))

        prices = [market.price_for]  # initial price (0.5)
        round_labels = ["Init"]

        for round_num in range(n_rounds):
            end_idx = min((round_num + 1) * 3, len(evidence))
            revealed = evidence[:end_idx]

            for agent in agents:
                stake = heuristic_agent_decision(
                    agent=agent, claim=claim, evidence_revealed=revealed,
                    market=market, round_num=round_num,
                    personality=personalities[agent.id],
                    claims_remaining=35,
                )
                if stake is not None:
                    cost = market.buy(stake.direction, stake.amount)
                    actual_cost = min(cost, agent.budget)
                    agent.budget -= actual_cost

            prices.append(market.price_for)
            n_for = sum(1 for e in revealed if e.direction == "for")
            n_against = sum(1 for e in revealed if e.direction == "against")
            round_labels.append(f"R{round_num+1}\n({n_for}F/{n_against}A)")

        x = np.arange(len(prices))

        # Plot price line
        ax.plot(x, prices, color=COLORS["market"], linewidth=2.5, marker="o",
                markersize=5, zorder=3)
        ax.fill_between(x, 0.5, prices, alpha=0.1, color=COLORS["market"])

        # Ground truth line
        ax.axhline(y=claim.evaluator_quality, color=COLORS["evaluator"],
                   linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(len(x)-0.5, claim.evaluator_quality + 0.02,
                f"Eval: {claim.evaluator_quality:.2f}", fontsize=8,
                color=COLORS["evaluator"], ha="right")

        # Confidence line
        ax.axhline(y=claim.confidence, color=COLORS["confidence"],
                   linestyle=":", linewidth=1.5, alpha=0.7)
        ax.text(0.5, claim.confidence - 0.04,
                f"Conf: {claim.confidence:.2f}", fontsize=8,
                color=COLORS["confidence"])

        # Neutral line
        ax.axhline(y=0.5, color="#ccc", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(round_labels, fontsize=7)
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel("Market Price")

        short_id = target_id.split("/")[1]
        status = claim.status
        ax.set_title(f"{short_id} ({status}, {len(evidence)} evidence)", fontweight="bold")

    fig.suptitle("Market Price Evolution as Evidence is Revealed",
                 fontsize=15, fontweight="bold", y=1.01)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["market"], linewidth=2.5, marker="o", label="Market Price"),
        Line2D([0], [0], color=COLORS["evaluator"], linestyle="--", linewidth=1.5, label="Evaluator (ground truth)"),
        Line2D([0], [0], color=COLORS["confidence"], linestyle=":", linewidth=1.5, label="Raw Confidence"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "price_evolution.png")
    plt.close()
    print("Saved: price_evolution.png")


# =========================================================================
# Chart 4: Kendall Tau Comparison Bar Chart
# =========================================================================
def chart_kendall_tau():
    """Bar chart comparing market tau vs confidence tau with significance."""
    kt = data["kendall_tau"]
    sr = data["spearman_rho"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel 1: Kendall Tau
    ax = axes[0]
    methods = ["Market\nPrice", "Raw\nConfidence"]
    taus = [kt["market_vs_evaluator"]["tau"], kt["confidence_vs_evaluator"]["tau"]]
    p_vals = [kt["market_vs_evaluator"]["p_value"], kt["confidence_vs_evaluator"]["p_value"]]

    bar_colors = [COLORS["market"], COLORS["confidence"]]
    bars = ax.bar(methods, taus, color=bar_colors, width=0.55, alpha=0.85, edgecolor="white", linewidth=1.5)

    # Add p-value annotations
    for i, (bar, p) in enumerate(zip(bars, p_vals)):
        sig_marker = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        sig_color = COLORS["sig"] if p < 0.05 else COLORS["nonsig"]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"tau={taus[i]:.3f}\np={p:.4f} {sig_marker}",
                ha="center", fontsize=9, color=sig_color, fontweight="bold")

    ax.set_ylabel("Kendall Tau Correlation")
    ax.set_title("Kendall Tau vs Evaluator", fontweight="bold")
    ax.set_ylim(0, 0.6)
    ax.axhline(y=0, color="#ccc", linewidth=0.8)

    # Delta annotation
    delta = kt["market_vs_evaluator"]["tau"] - kt["confidence_vs_evaluator"]["tau"]
    ax.annotate("", xy=(0, taus[0]), xytext=(1, taus[1]),
                arrowprops=dict(arrowstyle="<->", color="#666", linewidth=1.5))
    mid_y = (taus[0] + taus[1]) / 2
    ax.text(0.5, mid_y - 0.04, f"Delta: {delta:+.3f}", ha="center",
            fontsize=9, color="#666", fontstyle="italic")

    # Panel 2: Spearman Rho
    ax = axes[1]
    rhos = [sr["market_vs_evaluator"]["rho"], sr["confidence_vs_evaluator"]["rho"]]
    p_vals_r = [sr["market_vs_evaluator"]["p_value"], sr["confidence_vs_evaluator"]["p_value"]]

    bars = ax.bar(methods, rhos, color=bar_colors, width=0.55, alpha=0.85, edgecolor="white", linewidth=1.5)

    for i, (bar, p) in enumerate(zip(bars, p_vals_r)):
        sig_marker = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        sig_color = COLORS["sig"] if p < 0.05 else COLORS["nonsig"]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"rho={rhos[i]:.3f}\np={p:.4f} {sig_marker}",
                ha="center", fontsize=9, color=sig_color, fontweight="bold")

    ax.set_ylabel("Spearman Rho Correlation")
    ax.set_title("Spearman Rho vs Evaluator", fontweight="bold")
    ax.set_ylim(0, 0.7)
    ax.axhline(y=0, color="#ccc", linewidth=0.8)

    delta_r = sr["market_vs_evaluator"]["rho"] - sr["confidence_vs_evaluator"]["rho"]
    ax.annotate("", xy=(0, rhos[0]), xytext=(1, rhos[1]),
                arrowprops=dict(arrowstyle="<->", color="#666", linewidth=1.5))
    mid_y_r = (rhos[0] + rhos[1]) / 2
    ax.text(0.5, mid_y_r - 0.04, f"Delta: {delta_r:+.3f}", ha="center",
            fontsize=9, color="#666", fontstyle="italic")

    fig.suptitle("Ranking Correlation with Evaluator: Market vs Confidence",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
             "Confidence scoring significantly outperforms 2-agent market on both metrics.\n"
             "Market tau is marginally significant (p=0.052); confidence tau is highly significant (p<0.001).",
             ha="center", fontsize=9, color="#666")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "kendall_tau_comparison.png")
    plt.close()
    print("Saved: kendall_tau_comparison.png")


# =========================================================================
# Run all charts
# =========================================================================
if __name__ == "__main__":
    print("Creating visualizations...")
    chart_ranking_comparison()
    chart_agent_portfolio()
    chart_price_evolution()
    chart_kendall_tau()
    print(f"\nAll charts saved to {OUTPUT_DIR}/")

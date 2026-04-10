"""
Visualize 3 blind simulation runs against real Kalshi outcomes.

Generates professor-ready charts:
  1. Portfolio balance trajectories per persona (each run overlaid)
  2. Per-persona accuracy, ROI, and Brier score comparison across runs
  3. Action distribution heatmap (BUY_YES / BUY_NO / SKIP per persona)
  4. Consistency analysis: how often personas agree across runs
  5. Per-event profit/loss breakdown

Usage:
    uv run python scripts/visualize_runs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Config ──────────────────────────────────────────────────────────────────

RUN_FILES = [Path("output/run1.jsonl"), Path("output/run2.jsonl"), Path("output/run3.jsonl")]
EVENTS_FILE = Path("data/benchmark/events.json")
OUTPUT_DIR = Path("output/figures")
STARTING_BALANCE = 100.0

PERSONA_LABELS = {
    "overconfident": "Overconfident\nOracle",
    "risk_averse": "Cautious\nAnalyst",
    "recency_biased": "Recency\nBiased",
    "base_rate": "Base Rate\nAnalyst",
    "contrarian": "Contrarian\nTrader",
    "anchoring": "Anchoring\nBiased",
}

PERSONA_COLORS = {
    "overconfident": "#e74c3c",
    "risk_averse": "#3498db",
    "recency_biased": "#e67e22",
    "base_rate": "#2ecc71",
    "contrarian": "#9b59b6",
    "anchoring": "#1abc9c",
}

RUN_STYLES = {0: "-", 1: "--", 2: ":"}


# ── Data Loading ────────────────────────────────────────────────────────────

def load_decisions(path: Path) -> list[dict]:
    """Load only decision records (skip reveal records) from JSONL."""
    decisions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("record_type") == "reveal":
                continue
            decisions.append(record)
    return decisions


def load_outcomes(path: Path) -> dict[str, str]:
    """Load event_ticker -> outcome mapping from events.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {e["event_ticker"]: e["outcome"] for e in data["events"]}


def load_event_titles(path: Path) -> dict[str, str]:
    """Load event_ticker -> short title."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {e["event_ticker"]: e["title"][:40] for e in data["events"]}


def settle_decision(d: dict, outcome: str) -> dict:
    """Settle a single decision. Returns dict with profit, won fields added."""
    action = d["action"]
    stake = d["stake_dollars"]
    yes_price = d["yes_price_cents"]
    no_price = d["no_price_cents"]

    if action == "SKIP":
        return {**d, "profit": 0.0, "won": None, "payout": 0.0}

    if action == "BUY_YES":
        if outcome == "yes":
            payout = stake * 100 / yes_price
            return {**d, "profit": round(payout - stake, 2), "won": True, "payout": round(payout, 2)}
        else:
            return {**d, "profit": round(-stake, 2), "won": False, "payout": 0.0}

    # BUY_NO
    if outcome == "no":
        payout = stake * 100 / no_price
        return {**d, "profit": round(payout - stake, 2), "won": True, "payout": round(payout, 2)}
    else:
        return {**d, "profit": round(-stake, 2), "won": False, "payout": 0.0}


def build_run_dataframe(run_idx: int, decisions: list[dict], outcomes: dict[str, str]) -> pd.DataFrame:
    """Build a settled DataFrame for one run."""
    settled = []
    for d in decisions:
        ticker = d["event_ticker"]
        if ticker in outcomes:
            settled.append(settle_decision(d, outcomes[ticker]))
    df = pd.DataFrame(settled)
    df["run"] = run_idx + 1
    return df


def compute_balance_trajectory(df: pd.DataFrame, persona_id: str) -> tuple[list[str], list[float]]:
    """Compute running balance for a persona across events in order."""
    persona_df = df[df["persona_id"] == persona_id].copy()
    # Get event order from the data
    event_order = []
    seen = set()
    for _, row in persona_df.iterrows():
        if row["event_ticker"] not in seen:
            event_order.append(row["event_ticker"])
            seen.add(row["event_ticker"])

    balance = STARTING_BALANCE
    events = []
    balances = []
    for ticker in event_order:
        event_decisions = persona_df[persona_df["event_ticker"] == ticker]
        profit = event_decisions["profit"].sum()
        balance += profit
        events.append(ticker)
        balances.append(balance)

    return events, balances


# ── Chart Builders ──────────────────────────────────────────────────────────

def plot_balance_trajectories(all_dfs: list[pd.DataFrame], output_dir: Path) -> None:
    """Chart 1: Portfolio balance over time per persona, all runs overlaid."""
    personas = sorted(PERSONA_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for i, pid in enumerate(personas):
        ax = axes[i]
        for run_idx, df in enumerate(all_dfs):
            events, balances = compute_balance_trajectory(df, pid)
            short_events = [e.replace("KX", "").replace("KXKLARSTRIP", "KLARSTRIP")[:12] for e in events]
            ax.plot(
                range(len(balances)), balances,
                linestyle=RUN_STYLES[run_idx],
                color=PERSONA_COLORS[pid],
                linewidth=2,
                alpha=0.85,
                label=f"Run {run_idx + 1}",
            )

        ax.axhline(y=STARTING_BALANCE, color="gray", linestyle="-", alpha=0.3, linewidth=1)
        ax.set_title(PERSONA_LABELS[pid].replace("\n", " "), fontsize=12, fontweight="bold")
        ax.set_xlabel("Event #", fontsize=9)
        if i % 3 == 0:
            ax.set_ylabel("Balance ($)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Portfolio Balance Trajectories — 3 Independent Blind Runs", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "01_balance_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [1/6] Balance trajectories")


def plot_metrics_comparison(all_dfs: list[pd.DataFrame], output_dir: Path) -> None:
    """Chart 2: Accuracy, ROI, and Brier score bar charts across runs."""
    personas = sorted(PERSONA_LABELS.keys())
    metrics_data = []

    for run_idx, df in enumerate(all_dfs):
        for pid in personas:
            pdf = df[df["persona_id"] == pid]
            bets = pdf[pdf["action"] != "SKIP"]
            total_bets = len(bets)
            wins = len(bets[bets["won"] == True])
            accuracy = (wins / total_bets * 100) if total_bets > 0 else 0

            total_profit = pdf["profit"].sum()
            roi = total_profit / STARTING_BALANCE * 100

            # Brier score
            brier_scores = []
            for _, row in bets.iterrows():
                forecast = 1.0 if row["action"] == "BUY_YES" else 0.0
                actual = 1.0 if row.get("won") else 0.0
                # More precise: check if BUY_YES and outcome=yes, etc.
                if row["action"] == "BUY_YES":
                    actual_outcome = 1.0 if row["won"] else 0.0
                else:
                    actual_outcome = 0.0 if row["won"] else 1.0
                brier_scores.append((forecast - actual_outcome) ** 2)
            brier = np.mean(brier_scores) if brier_scores else 0

            metrics_data.append({
                "persona": PERSONA_LABELS[pid].replace("\n", " "),
                "persona_id": pid,
                "run": f"Run {run_idx + 1}",
                "accuracy": accuracy,
                "roi": roi,
                "brier": brier,
                "wins": wins,
                "losses": total_bets - wins,
                "skips": len(pdf[pdf["action"] == "SKIP"]),
            })

    mdf = pd.DataFrame(metrics_data)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Accuracy
    sns.barplot(data=mdf, x="persona", y="accuracy", hue="run", ax=axes[0], palette="Set2")
    axes[0].set_title("Prediction Accuracy (%)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Accuracy %")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].axhline(y=50, color="gray", linestyle="--", alpha=0.4, label="Random baseline")
    axes[0].legend(fontsize=8)

    # ROI
    sns.barplot(data=mdf, x="persona", y="roi", hue="run", ax=axes[1], palette="Set2")
    axes[1].set_title("Return on Investment (%)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("ROI %")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    # Brier Score
    sns.barplot(data=mdf, x="persona", y="brier", hue="run", ax=axes[2], palette="Set2")
    axes[2].set_title("Brier Score (lower = better)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Brier Score")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend(fontsize=8)

    fig.suptitle("Per-Persona Metrics Across 3 Runs", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "02_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [2/6] Metrics comparison")


def plot_action_distribution(all_dfs: list[pd.DataFrame], output_dir: Path) -> None:
    """Chart 3: Action distribution heatmap (BUY_YES / BUY_NO / SKIP) per persona."""
    personas = sorted(PERSONA_LABELS.keys())
    actions = ["BUY_YES", "BUY_NO", "SKIP"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for run_idx, df in enumerate(all_dfs):
        matrix = []
        for pid in personas:
            pdf = df[df["persona_id"] == pid]
            row = []
            for action in actions:
                count = len(pdf[pdf["action"] == action])
                row.append(count)
            matrix.append(row)

        matrix_arr = np.array(matrix)
        labels = [PERSONA_LABELS[p].replace("\n", " ") for p in personas]

        sns.heatmap(
            matrix_arr, ax=axes[run_idx],
            xticklabels=actions, yticklabels=labels,
            annot=True, fmt="d", cmap="YlOrRd",
            cbar_kws={"label": "Count"},
        )
        axes[run_idx].set_title(f"Run {run_idx + 1}", fontsize=12, fontweight="bold")

    fig.suptitle("Action Distribution per Persona", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "03_action_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [3/6] Action distribution heatmap")


def plot_consistency_analysis(all_dfs: list[pd.DataFrame], output_dir: Path) -> None:
    """Chart 4: Cross-run consistency — how often each persona makes the same decision."""
    personas = sorted(PERSONA_LABELS.keys())

    # For each persona + event + window, check if all 3 runs agree on action
    # Build lookup: (persona, event, window) -> [action_run1, action_run2, action_run3]
    action_map: dict[tuple[str, str, str], list[str]] = {}
    for run_idx, df in enumerate(all_dfs):
        for _, row in df.iterrows():
            key = (row["persona_id"], row["event_ticker"], row["window"])
            if key not in action_map:
                action_map[key] = [None, None, None]
            action_map[key][run_idx] = row["action"]

    consistency = {pid: {"all_agree": 0, "majority": 0, "all_differ": 0, "total": 0} for pid in personas}

    for (pid, _, _), actions in action_map.items():
        if None in actions:
            continue
        consistency[pid]["total"] += 1
        unique = len(set(actions))
        if unique == 1:
            consistency[pid]["all_agree"] += 1
        elif unique == 2:
            consistency[pid]["majority"] += 1
        else:
            consistency[pid]["all_differ"] += 1

    labels = [PERSONA_LABELS[p].replace("\n", " ") for p in personas]
    agree_pcts = []
    majority_pcts = []
    differ_pcts = []

    for pid in personas:
        total = consistency[pid]["total"]
        if total == 0:
            agree_pcts.append(0)
            majority_pcts.append(0)
            differ_pcts.append(0)
        else:
            agree_pcts.append(consistency[pid]["all_agree"] / total * 100)
            majority_pcts.append(consistency[pid]["majority"] / total * 100)
            differ_pcts.append(consistency[pid]["all_differ"] / total * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(personas))
    width = 0.25

    bars1 = ax.bar(x - width, agree_pcts, width, label="All 3 agree", color="#2ecc71", alpha=0.85)
    bars2 = ax.bar(x, majority_pcts, width, label="2 of 3 agree", color="#f39c12", alpha=0.85)
    bars3 = ax.bar(x + width, differ_pcts, width, label="All differ", color="#e74c3c", alpha=0.85)

    ax.set_ylabel("% of decisions", fontsize=11)
    ax.set_title("Cross-Run Decision Consistency per Persona", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)

    # Add percentage labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "04_consistency_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [4/6] Consistency analysis")


def plot_event_profitability(all_dfs: list[pd.DataFrame], outcomes: dict[str, str], output_dir: Path) -> None:
    """Chart 5: Per-event total profit/loss averaged across runs."""
    combined = pd.concat(all_dfs, ignore_index=True)

    # Average profit per event across all personas and runs
    event_profit = combined.groupby("event_ticker")["profit"].mean().reset_index()
    event_profit = event_profit.sort_values("profit", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ["#e74c3c" if p < 0 else "#2ecc71" for p in event_profit["profit"]]

    # Add outcome labels
    labels = []
    for ticker in event_profit["event_ticker"]:
        outcome = outcomes.get(ticker, "?").upper()
        labels.append(f"{ticker}\n(actual: {outcome})")

    bars = ax.barh(range(len(event_profit)), event_profit["profit"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(event_profit)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Avg Profit per Decision ($)", fontsize=11)
    ax.set_title("Per-Event Average Profitability (across all personas & runs)", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "05_event_profitability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [5/6] Event profitability")


def plot_final_balance_summary(all_dfs: list[pd.DataFrame], output_dir: Path) -> None:
    """Chart 6: Final balance boxplot across runs per persona."""
    personas = sorted(PERSONA_LABELS.keys())
    balance_data = []

    for run_idx, df in enumerate(all_dfs):
        for pid in personas:
            _, balances = compute_balance_trajectory(df, pid)
            final = balances[-1] if balances else STARTING_BALANCE
            balance_data.append({
                "persona": PERSONA_LABELS[pid].replace("\n", " "),
                "persona_id": pid,
                "final_balance": final,
                "run": f"Run {run_idx + 1}",
            })

    bdf = pd.DataFrame(balance_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Grouped bar chart
    persona_order = [PERSONA_LABELS[p].replace("\n", " ") for p in personas]
    sns.barplot(data=bdf, x="persona", y="final_balance", hue="run", ax=ax,
                palette="Set2", order=persona_order)

    ax.axhline(y=STARTING_BALANCE, color="gray", linestyle="--", alpha=0.5, linewidth=1.5, label="Starting ($100)")
    ax.set_title("Final Portfolio Balance per Persona", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Final Balance ($)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)

    # Annotate values
    for container in ax.containers:
        ax.bar_label(container, fmt="$%.0f", fontsize=7, padding=2)

    plt.tight_layout()
    fig.savefig(output_dir / "06_final_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [6/6] Final balance summary")


def plot_aggregate_summary(all_dfs: list[pd.DataFrame], outcomes: dict[str, str], output_dir: Path) -> None:
    """Chart 7: Single aggregate dashboard combining all 3 runs.

    Shows mean +/- std across runs for each persona: accuracy, ROI, Brier,
    final balance, win rate, and total profit.
    """
    personas = sorted(PERSONA_LABELS.keys())

    # Compute per-run metrics for each persona
    run_metrics: dict[str, list[dict]] = {pid: [] for pid in personas}

    for run_idx, df in enumerate(all_dfs):
        for pid in personas:
            pdf = df[df["persona_id"] == pid]
            bets = pdf[pdf["action"] != "SKIP"]
            total_bets = len(bets)
            wins = len(bets[bets["won"] == True])
            accuracy = (wins / total_bets * 100) if total_bets > 0 else 0
            total_profit = pdf["profit"].sum()
            roi = total_profit / STARTING_BALANCE * 100
            _, balances = compute_balance_trajectory(df, pid)
            final_bal = balances[-1] if balances else STARTING_BALANCE

            # Brier
            brier_scores = []
            for _, row in bets.iterrows():
                forecast = 1.0 if row["action"] == "BUY_YES" else 0.0
                if row["action"] == "BUY_YES":
                    actual_outcome = 1.0 if row["won"] else 0.0
                else:
                    actual_outcome = 0.0 if row["won"] else 1.0
                brier_scores.append((forecast - actual_outcome) ** 2)
            brier = float(np.mean(brier_scores)) if brier_scores else 0.0

            run_metrics[pid].append({
                "accuracy": accuracy,
                "roi": roi,
                "brier": brier,
                "final_balance": final_bal,
                "total_profit": total_profit,
                "wins": wins,
                "losses": total_bets - wins,
                "skips": int(len(pdf[pdf["action"] == "SKIP"])),
                "total_bets": total_bets,
            })

    # Build aggregate table
    agg_rows = []
    for pid in personas:
        metrics_list = run_metrics[pid]
        accs = [m["accuracy"] for m in metrics_list]
        rois = [m["roi"] for m in metrics_list]
        briers = [m["brier"] for m in metrics_list]
        finals = [m["final_balance"] for m in metrics_list]
        profits = [m["total_profit"] for m in metrics_list]
        wins = [m["wins"] for m in metrics_list]
        losses = [m["losses"] for m in metrics_list]
        skips = [m["skips"] for m in metrics_list]
        total_bets = [m["total_bets"] for m in metrics_list]

        agg_rows.append({
            "persona": PERSONA_LABELS[pid].replace("\n", " "),
            "persona_id": pid,
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "roi_mean": np.mean(rois),
            "roi_std": np.std(rois),
            "brier_mean": np.mean(briers),
            "brier_std": np.std(briers),
            "bal_mean": np.mean(finals),
            "bal_std": np.std(finals),
            "profit_mean": np.mean(profits),
            "wins_mean": np.mean(wins),
            "losses_mean": np.mean(losses),
            "skips_mean": np.mean(skips),
            "bets_mean": np.mean(total_bets),
        })

    adf = pd.DataFrame(agg_rows)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Accuracy (mean +/- std)
    ax = axes[0, 0]
    colors = [PERSONA_COLORS[r["persona_id"]] for _, r in adf.iterrows()]
    bars = ax.bar(adf["persona"], adf["acc_mean"], yerr=adf["acc_std"],
                  color=colors, alpha=0.85, capsize=5, edgecolor="white", linewidth=0.5)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.4)
    ax.set_title("Accuracy % (mean ± std)", fontsize=12, fontweight="bold")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, adf["acc_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # 2. ROI (mean +/- std)
    ax = axes[0, 1]
    bar_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in adf["roi_mean"]]
    bars = ax.bar(adf["persona"], adf["roi_mean"], yerr=adf["roi_std"],
                  color=bar_colors, alpha=0.85, capsize=5, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title("ROI % (mean ± std)", fontsize=12, fontweight="bold")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, adf["roi_mean"]):
        offset = 3 if val >= 0 else -8
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{val:+.0f}%", ha="center", fontsize=9, fontweight="bold")

    # 3. Brier Score (mean +/- std)
    ax = axes[0, 2]
    bars = ax.bar(adf["persona"], adf["brier_mean"], yerr=adf["brier_std"],
                  color=[PERSONA_COLORS[r["persona_id"]] for _, r in adf.iterrows()],
                  alpha=0.85, capsize=5, edgecolor="white", linewidth=0.5)
    ax.set_title("Brier Score (mean ± std, lower = better)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, adf["brier_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    # 4. Final Balance (mean +/- std)
    ax = axes[1, 0]
    bar_colors = ["#2ecc71" if r >= STARTING_BALANCE else "#e74c3c" for r in adf["bal_mean"]]
    bars = ax.bar(adf["persona"], adf["bal_mean"], yerr=adf["bal_std"],
                  color=bar_colors, alpha=0.85, capsize=5, edgecolor="white", linewidth=0.5)
    ax.axhline(y=STARTING_BALANCE, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_title("Final Balance (mean ± std)", fontsize=12, fontweight="bold")
    ax.set_ylabel("$")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, adf["bal_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"${val:.0f}", ha="center", fontsize=9, fontweight="bold")

    # 5. Win / Loss / Skip breakdown (stacked, averaged)
    ax = axes[1, 1]
    x = np.arange(len(adf))
    w = 0.6
    ax.bar(x, adf["wins_mean"], w, label="Wins", color="#2ecc71", alpha=0.85)
    ax.bar(x, adf["losses_mean"], w, bottom=adf["wins_mean"], label="Losses", color="#e74c3c", alpha=0.85)
    ax.bar(x, adf["skips_mean"], w,
           bottom=adf["wins_mean"].values + adf["losses_mean"].values,
           label="Skips", color="#95a5a6", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(adf["persona"], rotation=25)
    ax.set_title("Avg Decisions Breakdown", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # 6. Profit waterfall (sorted)
    ax = axes[1, 2]
    sorted_adf = adf.sort_values("profit_mean", ascending=True)
    bar_colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in sorted_adf["profit_mean"]]
    bars = ax.barh(sorted_adf["persona"], sorted_adf["profit_mean"], color=bar_colors, alpha=0.85)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title("Total Profit (mean across runs)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Profit ($)")
    for bar, val in zip(bars, sorted_adf["profit_mean"]):
        offset = 3 if val >= 0 else -3
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f"${val:+.0f}", va="center", fontsize=9, fontweight="bold",
                ha="left" if val >= 0 else "right")

    fig.suptitle(
        "Aggregate Results: 3 Independent Blind Runs (6 Personas × 15 Events × 3 Windows)",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "07_aggregate_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [7/7] Aggregate summary dashboard")


def print_aggregate_table(all_dfs: list[pd.DataFrame]) -> None:
    """Print a text summary table to console."""
    personas = sorted(PERSONA_LABELS.keys())
    print("\n" + "=" * 95)
    print("AGGREGATE RESULTS (mean ± std across 3 runs)")
    print("=" * 95)
    print(f"{'Persona':<22} {'Accuracy':>12} {'ROI':>12} {'Brier':>12} {'Final Bal':>12} {'Profit':>12}")
    print("-" * 95)

    for pid in personas:
        accs, rois, briers, finals, profits = [], [], [], [], []
        for df in all_dfs:
            pdf = df[df["persona_id"] == pid]
            bets = pdf[pdf["action"] != "SKIP"]
            total_bets = len(bets)
            wins = len(bets[bets["won"] == True])
            accs.append((wins / total_bets * 100) if total_bets > 0 else 0)
            total_profit = pdf["profit"].sum()
            rois.append(total_profit / STARTING_BALANCE * 100)
            profits.append(total_profit)
            _, balances = compute_balance_trajectory(df, pid)
            finals.append(balances[-1] if balances else STARTING_BALANCE)

            brier_scores = []
            for _, row in bets.iterrows():
                forecast = 1.0 if row["action"] == "BUY_YES" else 0.0
                actual = 1.0 if row["won"] else 0.0 if row["action"] == "BUY_YES" else (0.0 if row["won"] else 1.0)
                if row["action"] == "BUY_YES":
                    actual = 1.0 if row["won"] else 0.0
                else:
                    actual = 0.0 if row["won"] else 1.0
                brier_scores.append((forecast - actual) ** 2)
            briers.append(float(np.mean(brier_scores)) if brier_scores else 0.0)

        name = PERSONA_LABELS[pid].replace("\n", " ")
        print(f"{name:<22} {np.mean(accs):>5.1f}±{np.std(accs):>4.1f}% "
              f"{np.mean(rois):>+6.0f}±{np.std(rois):>4.0f}% "
              f"{np.mean(briers):>6.3f}±{np.std(briers):>.3f} "
              f"${np.mean(finals):>7.0f}±{np.std(finals):>4.0f} "
              f"${np.mean(profits):>+7.0f}±{np.std(profits):>4.0f}")

    print("=" * 95)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    print("Loading data...")
    outcomes = load_outcomes(EVENTS_FILE)
    print(f"  {len(outcomes)} events with outcomes loaded")

    all_dfs = []
    for i, path in enumerate(RUN_FILES):
        decisions = load_decisions(path)
        df = build_run_dataframe(i, decisions, outcomes)
        all_dfs.append(df)
        print(f"  Run {i+1}: {len(decisions)} decisions -> {len(df)} settled")

    print(f"\nGenerating charts in {OUTPUT_DIR}/...")
    plot_balance_trajectories(all_dfs, OUTPUT_DIR)
    plot_metrics_comparison(all_dfs, OUTPUT_DIR)
    plot_action_distribution(all_dfs, OUTPUT_DIR)
    plot_consistency_analysis(all_dfs, OUTPUT_DIR)
    plot_event_profitability(all_dfs, outcomes, OUTPUT_DIR)
    plot_final_balance_summary(all_dfs, OUTPUT_DIR)
    plot_aggregate_summary(all_dfs, outcomes, OUTPUT_DIR)

    print_aggregate_table(all_dfs)

    print(f"\nDone! 7 charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

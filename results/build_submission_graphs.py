from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
METRICS_CSV = ROOT / "final_1p5b_run_metrics.csv"
BASELINE_OUTPUT = ROOT / "baseline_output.txt"
REWARD_CHART = ROOT / "final_1p5b_reward_chart.svg"
QUALITY_CHART = ROOT / "final_1p5b_quality_chart.svg"
HEALTH_CHART = ROOT / "final_1p5b_training_health_chart.svg"
BASELINE_COMPARISON_CHART = ROOT / "baseline_vs_trained_benchmark_chart.svg"


def load_rows() -> list[dict[str, float]]:
    with METRICS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
        return rows


def load_baseline_average() -> float:
    score_pattern = re.compile(r"score=([0-9.]+)")
    scores: list[float] = []
    with BASELINE_OUTPUT.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = score_pattern.search(line)
            if match:
                scores.append(float(match.group(1)))

    if not scores:
        raise SystemExit("No baseline scores found in baseline_output.txt")
    return sum(scores) / len(scores)


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_reward_chart(rows: list[dict[str, float]]) -> None:
    steps = [int(row["step"]) for row in rows]
    mean_reward = [row["mean_reward"] for row in rows]
    reward_std = [row["reward_std"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(steps, mean_reward, marker="o", linewidth=2.4, color="#0f766e", label="Mean reward")
    ax.plot(steps, reward_std, marker="o", linewidth=2.2, color="#dc2626", label="Reward std")
    style_axes(ax, "Final 1.5B GRPO Reward Dynamics", "Reward / std")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REWARD_CHART, bbox_inches="tight")
    plt.close(fig)


def build_quality_chart(rows: list[dict[str, float]]) -> None:
    steps = [int(row["step"]) for row in rows]
    benchmark = [row["mean_benchmark_score"] for row in rows]
    exact_json = [row["exact_json_rate"] for row in rows]
    safety = [row["mean_safety_reward"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(steps, benchmark, marker="o", linewidth=2.4, color="#2563eb", label="Benchmark score")
    ax1.plot(steps, exact_json, marker="o", linewidth=2.2, color="#7c3aed", label="Exact JSON rate")
    style_axes(ax1, "Final 1.5B Output Quality Signals", "Score / rate")

    ax2 = ax1.twinx()
    ax2.plot(steps, safety, marker="o", linewidth=2.0, color="#b45309", label="Safety reward")
    ax2.set_ylabel("Safety reward")
    ax2.spines["top"].set_visible(False)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(QUALITY_CHART, bbox_inches="tight")
    plt.close(fig)


def build_health_chart(rows: list[dict[str, float]]) -> None:
    steps = [int(row["step"]) for row in rows]
    grad_norm = [row["grad_norm"] for row in rows]
    collapse_flag = [1.0 if row["reward_std"] == 0.0 else 0.0 for row in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(steps, grad_norm, marker="o", linewidth=2.4, color="#111827", label="Grad norm")
    style_axes(ax1, "Final 1.5B Training Health", "Grad norm")

    ax2 = ax1.twinx()
    ax2.bar(steps, collapse_flag, width=0.45, color="#ef4444", alpha=0.28, label="Zero reward-std batch")
    ax2.set_ylabel("Collapse flag")
    ax2.set_ylim(0, 1.2)
    ax2.spines["top"].set_visible(False)

    lines = ax1.get_lines() + [ax2.patches[0]]
    labels = ["Grad norm", "Zero reward-std batch"]
    ax1.legend(lines, labels, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(HEALTH_CHART, bbox_inches="tight")
    plt.close(fig)


def build_baseline_comparison_chart(rows: list[dict[str, float]], baseline_average: float) -> None:
    avg_benchmark = sum(row["mean_benchmark_score"] for row in rows) / len(rows)
    best_benchmark = max(row["mean_benchmark_score"] for row in rows)
    final_benchmark = rows[-1]["mean_benchmark_score"]

    labels = [
        "Legacy baseline\navg score",
        "GRPO run\navg benchmark",
        "GRPO run\nbest benchmark",
        "GRPO run\nfinal benchmark",
    ]
    values = [baseline_average, avg_benchmark, best_benchmark, final_benchmark]
    colors = ["#475569", "#2563eb", "#0f766e", "#dc2626"]

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    style_axes(ax, "Baseline vs Trained Benchmark Signals", "Score")
    ax.set_ylim(0, max(values) + 0.08)

    # This is intentionally explicit: the baseline comes from a smaller legacy run.
    ax.text(
        0.02,
        0.98,
        "Baseline is the checked-in 3-task fallback run.\nGRPO values come from the 18-step 1.5B online RL run.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.35"},
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    fig.tight_layout()
    fig.savefig(BASELINE_COMPARISON_CHART, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit("No metric rows found in final_1p5b_run_metrics.csv")
    baseline_average = load_baseline_average()
    build_reward_chart(rows)
    build_quality_chart(rows)
    build_health_chart(rows)
    build_baseline_comparison_chart(rows, baseline_average)
    print("Wrote:")
    print(f"- {REWARD_CHART}")
    print(f"- {QUALITY_CHART}")
    print(f"- {HEALTH_CHART}")
    print(f"- {BASELINE_COMPARISON_CHART}")


if __name__ == "__main__":
    main()

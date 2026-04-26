import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from environment import MULTITURN_TRAJECTORIES, MultiTurnPrenatalEnvironment


OUT_DIR = Path(__file__).resolve().parent


def _danger_flag_count(trajectory_id: str, day_index: int) -> int:
    env = MultiTurnPrenatalEnvironment()
    env.reset(trajectory_id=trajectory_id)
    env.current_day = day_index
    return sum(1 for flag in env._risk_flags() if flag.startswith("DANGER"))


def plot_case_mix() -> None:
    condition_counts = Counter()
    urgency_counts = Counter()

    for traj in MULTITURN_TRAJECTORIES.values():
        condition_counts[traj.target_condition] += 1
        urgency_counts[traj.target_urgency] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    cond_labels = list(condition_counts.keys())
    cond_values = [condition_counts[label] for label in cond_labels]
    cond_colors = ["#7c3aed", "#0f766e", "#dc2626", "#2563eb", "#ca8a04", "#475569"]
    axes[0].bar(cond_labels, cond_values, color=cond_colors[: len(cond_labels)])
    axes[0].set_title("Benchmark Case Mix by Target Condition")
    axes[0].set_ylabel("Trajectory Count")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)

    urg_labels = list(urgency_counts.keys())
    urg_values = [urgency_counts[label] for label in urg_labels]
    urg_colors = ["#16a34a", "#f59e0b", "#dc2626"]
    axes[1].bar(urg_labels, urg_values, color=urg_colors[: len(urg_labels)])
    axes[1].set_title("Benchmark Case Mix by Target Urgency")
    axes[1].set_ylabel("Trajectory Count")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("MAAS Benchmark Coverage", fontsize=17)
    fig.tight_layout()
    out = OUT_DIR / "benchmark_case_mix.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


def plot_risk_progression() -> None:
    grouped = defaultdict(list)
    for traj in MULTITURN_TRAJECTORIES.values():
        grouped[traj.target_urgency].append(traj)

    urgency_order = ["monitor_at_home", "visit_phc_this_week", "go_to_hospital_today"]
    colors = {
        "monitor_at_home": "#16a34a",
        "visit_phc_this_week": "#f59e0b",
        "go_to_hospital_today": "#dc2626",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    day_axis = [1, 2, 3]

    for urgency in urgency_order:
        trajectories = grouped.get(urgency, [])
        if not trajectories:
            continue

        mean_bp = []
        mean_kicks = []
        mean_danger = []
        for day_idx in day_axis:
            day_slice = [traj.days[day_idx - 1] for traj in trajectories]
            mean_bp.append(sum(day.bp_systolic for day in day_slice) / len(day_slice))
            mean_kicks.append(sum(day.kick_count for day in day_slice) / len(day_slice))
            mean_danger.append(
                sum(_danger_flag_count(traj.trajectory_id, day_idx) for traj in trajectories) / len(trajectories)
            )

        axes[0].plot(day_axis, mean_bp, marker="o", linewidth=2.5, color=colors[urgency], label=urgency)
        axes[1].plot(day_axis, mean_kicks, marker="o", linewidth=2.5, color=colors[urgency], label=urgency)
        axes[2].plot(day_axis, mean_danger, marker="o", linewidth=2.5, color=colors[urgency], label=urgency)

    axes[0].set_title("Average Systolic BP by Day")
    axes[0].set_xlabel("Episode Day")
    axes[0].set_ylabel("Systolic BP")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Average Kick Count by Day")
    axes[1].set_xlabel("Episode Day")
    axes[1].set_ylabel("Kick Count")
    axes[1].grid(True, alpha=0.25)

    axes[2].set_title("Average Danger Flags by Day")
    axes[2].set_xlabel("Episode Day")
    axes[2].set_ylabel("Danger Flag Count")
    axes[2].grid(True, alpha=0.25)

    axes[2].legend(title="Target urgency", loc="upper left")
    fig.suptitle("Why Escalation Matters in MAAS", fontsize=17)
    fig.tight_layout()
    out = OUT_DIR / "benchmark_risk_progression.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


def plot_partial_observability() -> None:
    signals = [
        "latest_blood_pressure",
        "latest_kick_count",
        "latest_symptoms",
        "history_flags",
        "avg_meals",
        "avg_sleep",
        "latest_energy",
    ]
    visible_by_day = {
        1: {"latest_blood_pressure", "latest_kick_count"},
        2: {"latest_blood_pressure", "latest_kick_count", "latest_symptoms"},
        3: set(signals),
    }
    matrix = [
        [1 if signal in visible_by_day[day] else 0 for day in (1, 2, 3)]
        for signal in signals
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Partial Observability Schedule")
    ax.set_xlabel("Episode Day")
    ax.set_ylabel("Signal")
    ax.set_xticks([0, 1, 2], labels=["Day 1", "Day 2", "Day 3"])
    ax.set_yticks(range(len(signals)), labels=signals)

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(
                col_idx,
                row_idx,
                "Visible" if value else "Hidden",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_ticks([0, 1], labels=["Hidden", "Visible"])
    fig.tight_layout()
    out = OUT_DIR / "benchmark_partial_observability.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


def main() -> None:
    plot_case_mix()
    plot_risk_progression()
    plot_partial_observability()


if __name__ == "__main__":
    main()

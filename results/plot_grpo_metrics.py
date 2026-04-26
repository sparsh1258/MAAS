import json
from pathlib import Path

import matplotlib.pyplot as plt


SUMMARY_PATH = Path(__file__).with_name("grpo_training_summary.json")
REWARD_PLOT_PATH = SUMMARY_PATH.with_name("grpo_reward_curve.png")
LOSS_PLOT_PATH = SUMMARY_PATH.with_name("grpo_loss_curve.png")
COMBINED_PLOT_PATH = SUMMARY_PATH.with_name("grpo_training_plots.png")
REWARD_PLOT_FRESH_PATH = SUMMARY_PATH.with_name("grpo_reward_curve_fixed.png")
LOSS_PLOT_FRESH_PATH = SUMMARY_PATH.with_name("grpo_loss_curve_fixed.png")
COMBINED_PLOT_FRESH_PATH = SUMMARY_PATH.with_name("grpo_training_plots_fixed.png")


def _series_from_history(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    points = [
        (row["step"], row[key])
        for row in history
        if "step" in row and key in row and row[key] is not None
    ]
    if not points:
        return [], []
    xs, ys = zip(*points)
    return list(xs), list(ys)


def _save_single_plot(
    steps: list[float],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    color: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, values, marker="o", linewidth=2.5, color=color)
    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"Missing file: {SUMMARY_PATH}")

    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    history = [row for row in data.get("log_history", []) if "step" in row]
    if not history:
        raise SystemExit("No step-level training history found in grpo_training_summary.json")

    reward_steps, rewards = _series_from_history(history, "reward")
    loss_steps, losses = _series_from_history(history, "loss")
    benchmark_steps, benchmark_scores = _series_from_history(history, "maas/mean_benchmark_score")
    grad_steps, grad_norms = _series_from_history(history, "grad_norm")

    if not reward_steps:
        raise SystemExit("No reward values found in grpo_training_summary.json")
    if not loss_steps:
        raise SystemExit("No loss values found in grpo_training_summary.json")

    _save_single_plot(
        reward_steps,
        rewards,
        title="GRPO Reward by Step",
        ylabel="Reward",
        color="#0f766e",
        output_path=REWARD_PLOT_PATH,
    )
    _save_single_plot(
        reward_steps,
        rewards,
        title="GRPO Reward by Step",
        ylabel="Reward",
        color="#0f766e",
        output_path=REWARD_PLOT_FRESH_PATH,
    )
    _save_single_plot(
        loss_steps,
        losses,
        title="GRPO Loss by Step",
        ylabel="Loss",
        color="#be123c",
        output_path=LOSS_PLOT_PATH,
    )
    _save_single_plot(
        loss_steps,
        losses,
        title="GRPO Loss by Step",
        ylabel="Loss",
        color="#be123c",
        output_path=LOSS_PLOT_FRESH_PATH,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    axes[0].plot(reward_steps, rewards, marker="o", color="#0f766e")
    axes[0].set_title("GRPO Reward by Step")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)

    if benchmark_steps:
        axes[1].plot(benchmark_steps, benchmark_scores, marker="o", color="#2563eb")
    axes[1].set_title("Benchmark Score by Step")
    axes[1].set_ylabel("Benchmark Score")
    axes[1].grid(True, alpha=0.3)

    if grad_steps:
        axes[2].plot(grad_steps, grad_norms, marker="o", color="#dc2626")
    axes[2].set_title("Gradient Norm by Step")
    axes[2].set_ylabel("Grad Norm")
    axes[2].set_xlabel("Training Step")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(COMBINED_PLOT_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(COMBINED_PLOT_FRESH_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {REWARD_PLOT_PATH.name}")
    print(f"Saved {REWARD_PLOT_FRESH_PATH.name}")
    print(f"Saved {LOSS_PLOT_PATH.name}")
    print(f"Saved {LOSS_PLOT_FRESH_PATH.name}")
    print(f"Saved {COMBINED_PLOT_PATH.name}")
    print(f"Saved {COMBINED_PLOT_FRESH_PATH.name}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import matplotlib.pyplot as plt


SUMMARY_PATH = Path(__file__).with_name("grpo_training_summary.json")


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"Missing file: {SUMMARY_PATH}")

    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    history = [
        row for row in data.get("log_history", [])
        if "step" in row and any(k in row for k in ("reward", "maas/mean_benchmark_score", "grad_norm"))
    ]
    if not history:
        raise SystemExit("No step-level training history found in grpo_training_summary.json")

    steps = [row["step"] for row in history]
    rewards = [row.get("reward") for row in history]
    benchmark_scores = [row.get("maas/mean_benchmark_score") for row in history]
    grad_norms = [row.get("grad_norm") for row in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(steps, rewards, marker="o", color="#0f766e")
    axes[0].set_title("GRPO Reward by Step")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, benchmark_scores, marker="o", color="#2563eb")
    axes[1].set_title("Benchmark Score by Step")
    axes[1].set_ylabel("Benchmark Score")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, grad_norms, marker="o", color="#dc2626")
    axes[2].set_title("Gradient Norm by Step")
    axes[2].set_ylabel("Grad Norm")
    axes[2].set_xlabel("Training Step")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    combined_path = SUMMARY_PATH.with_name("grpo_training_plots.png")
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {combined_path.name}")


if __name__ == "__main__":
    main()

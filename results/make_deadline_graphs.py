import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "final_1p5b_run_metrics.csv"
BENCHMARK_OUT = BASE_DIR / "maas_benchmark_score_trend.png"
JSON_OUT = BASE_DIR / "maas_json_rate_trend.png"


def rolling_mean(values: list[float], window: int = 3) -> list[float]:
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def load_metrics() -> tuple[list[int], list[float], list[float]]:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing file: {CSV_PATH}")

    steps: list[int] = []
    benchmark_scores: list[float] = []
    json_rates: list[float] = []

    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            steps.append(int(row["step"]))
            benchmark_scores.append(float(row["mean_benchmark_score"]))
            json_rates.append(float(row["exact_json_rate"]))

    if not steps:
        raise SystemExit("No training rows found in final_1p5b_run_metrics.csv")

    return steps, benchmark_scores, json_rates


def plot_benchmark_trend(steps: list[int], benchmark_scores: list[float]) -> None:
    benchmark_ma = rolling_mean(benchmark_scores, window=3)

    plt.figure(figsize=(10, 5.5))
    plt.plot(
        steps,
        benchmark_scores,
        color="#93c5fd",
        linewidth=1.8,
        alpha=0.6,
        marker="o",
        label="Raw benchmark score",
    )
    plt.plot(
        steps,
        benchmark_ma,
        color="#2563eb",
        linewidth=3,
        marker="o",
        label="3-step moving average",
    )
    plt.title("MAAS Benchmark Score Trend", fontsize=16)
    plt.xlabel("Training Step")
    plt.ylabel("Benchmark Score")
    plt.ylim(0, max(0.3, max(benchmark_scores) + 0.03))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BENCHMARK_OUT, dpi=220, bbox_inches="tight")
    plt.close()


def plot_json_trend(steps: list[int], json_rates: list[float]) -> None:
    json_ma = rolling_mean(json_rates, window=3)

    plt.figure(figsize=(10, 5.5))
    plt.plot(
        steps,
        json_rates,
        color="#fde68a",
        linewidth=1.8,
        alpha=0.65,
        marker="o",
        label="Raw exact JSON rate",
    )
    plt.plot(
        steps,
        json_ma,
        color="#ca8a04",
        linewidth=3,
        marker="o",
        label="3-step moving average",
    )
    plt.title("MAAS Exact JSON Rate Trend", fontsize=16)
    plt.xlabel("Training Step")
    plt.ylabel("Exact JSON Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(JSON_OUT, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    steps, benchmark_scores, json_rates = load_metrics()
    plot_benchmark_trend(steps, benchmark_scores)
    plot_json_trend(steps, json_rates)
    print(f"Saved {BENCHMARK_OUT.name}")
    print(f"Saved {JSON_OUT.name}")


if __name__ == "__main__":
    main()

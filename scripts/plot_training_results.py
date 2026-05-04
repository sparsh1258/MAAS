from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SERIES = [
    ("mean_reward", "#0f766e", "Mean reward"),
    ("mean_benchmark_score", "#2563eb", "Benchmark score"),
    ("exact_json_rate", "#7c3aed", "Exact JSON rate"),
    ("mean_safety_reward", "#dc2626", "Safety reward"),
]


def _read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items() if value not in ("", None)})
    return rows


def _scale(values: Iterable[float], lo: float, hi: float) -> tuple[float, float]:
    vals = list(values)
    if not vals:
        return lo, hi
    observed_lo = min(vals)
    observed_hi = max(vals)
    if observed_lo == observed_hi:
        return observed_lo - 1.0, observed_hi + 1.0
    pad = (observed_hi - observed_lo) * 0.1
    return observed_lo - pad, observed_hi + pad


def _polyline(rows: list[dict[str, float]], key: str, x0: int, y0: int, width: int, height: int, y_min: float, y_max: float) -> str:
    if not rows:
        return ""
    points = []
    denom_x = max(len(rows) - 1, 1)
    denom_y = max(y_max - y_min, 1e-9)
    for index, row in enumerate(rows):
        x = x0 + (index / denom_x) * width
        y = y0 + height - ((row[key] - y_min) / denom_y) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def write_svg(rows: list[dict[str, float]], output: Path, title: str) -> None:
    width = 980
    height = 560
    x0 = 80
    y0 = 70
    chart_w = 820
    chart_h = 360
    all_values = [row[key] for row in rows for key, _, _ in SERIES if key in row]
    y_min, y_max = _scale(all_values, -1.0, 1.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{x0}" y="34" font-family="Arial" font-size="24" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{x0}" y="56" font-family="Arial" font-size="13" fill="#4b5563">Generated {datetime.now(timezone.utc).isoformat()} from real CSV metrics.</text>',
        f'<line x1="{x0}" y1="{y0 + chart_h}" x2="{x0 + chart_w}" y2="{y0 + chart_h}" stroke="#9ca3af"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + chart_h}" stroke="#9ca3af"/>',
        f'<text x="{x0 + chart_w - 10}" y="{y0 + chart_h + 36}" font-family="Arial" font-size="12" fill="#4b5563">Training step</text>',
        f'<text x="18" y="{y0 + 16}" font-family="Arial" font-size="12" fill="#4b5563">Metric value</text>',
        f'<text x="{x0 - 54}" y="{y0 + 5}" font-family="Arial" font-size="11" fill="#6b7280">{y_max:.2f}</text>',
        f'<text x="{x0 - 54}" y="{y0 + chart_h}" font-family="Arial" font-size="11" fill="#6b7280">{y_min:.2f}</text>',
    ]
    for idx, (key, color, label) in enumerate(SERIES):
        if not rows or key not in rows[0]:
            continue
        points = _polyline(rows, key, x0, y0, chart_w, chart_h, y_min, y_max)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        legend_y = y0 + chart_h + 70 + idx * 24
        lines.append(f'<rect x="{x0 + idx * 220}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{x0 + 20 + idx * 220}" y="{legend_y}" font-family="Arial" font-size="13" fill="#111827">{label}</text>')
    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate MAAS training summary plots from checked-in CSV metrics.")
    parser.add_argument("--metrics", default="results/final_1p5b_run_metrics.csv")
    parser.add_argument("--output-svg", default="results/training_results_overview.svg")
    parser.add_argument("--output-json", default="results/training_results_overview.json")
    args = parser.parse_args()

    rows = _read_rows(Path(args.metrics))
    summary = {
        "source_metrics": args.metrics,
        "steps": len(rows),
        "best_mean_reward": max((row.get("mean_reward", 0.0) for row in rows), default=None),
        "best_benchmark_score": max((row.get("mean_benchmark_score", 0.0) for row in rows), default=None),
        "best_exact_json_rate": max((row.get("exact_json_rate", 0.0) for row in rows), default=None),
        "final_step": rows[-1] if rows else None,
        "interpretation": "Training telemetry is real run evidence, not proof of robust multi-turn model mastery.",
    }
    write_svg(rows, Path(args.output_svg), "MAAS Qwen2.5-1.5B GRPO Training Metrics")
    Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"svg": args.output_svg, "summary": args.output_json, **summary}, indent=2))


if __name__ == "__main__":
    main()

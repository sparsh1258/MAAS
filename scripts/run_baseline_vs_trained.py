from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the exact multi-turn evaluator and summarize baseline/trained evidence honestly."
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--predictions-jsonl", help="Optional trained/base model action traces keyed by trajectory_id.")
    parser.add_argument("--base-model", help="Optional Hugging Face base model id for live inference.")
    parser.add_argument("--trained-model", help="Optional Hugging Face trained model id for live inference.")
    args = parser.parse_args()

    command = [sys.executable, "scripts/evaluate_multiturn.py", "--output-dir", args.output_dir]
    if args.predictions_jsonl:
        command.extend(["--predictions-jsonl", args.predictions_jsonl])
    if args.base_model:
        command.extend(["--base-model", args.base_model])
    if args.trained_model:
        command.extend(["--trained-model", args.trained_model])
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode

    output_dir = Path(args.output_dir)
    summary = _load_summary(output_dir / "multiturn_eval_summary.json")
    model_rows = [
        row
        for row in summary
        if row["policy"] == "model_predictions"
        or row["policy"].startswith("base_model:")
        or row["policy"].startswith("trained_model:")
    ]
    trained_rows = [row for row in summary if row["policy"].startswith("trained_model:") or row["policy"] == "model_predictions"]
    report = {
        "summary_source": str(output_dir / "multiturn_eval_summary.json"),
        "model_rows_present": bool(model_rows),
        "trained_rows_present": bool(trained_rows),
        "claim_trained_improvement": bool(trained_rows),
        "policies": summary,
        "interpretation": (
            "A trained before/after claim is supported only if trained-model or model_predictions rows are present "
            "and outperform the matched baseline under the same eight trajectories."
        ),
    }
    out_json = output_dir / "baseline_vs_trained_multiturn.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Baseline vs Trained Multi-Turn Evidence",
        "",
        f"Model rows present: `{'yes' if model_rows else 'no'}`",
        f"Trained-improvement claim allowed: `{'yes' if trained_rows else 'no'}`",
        "",
        "| Policy | Mean Reward | Condition Acc. | Urgency Acc. | Under-Esc. | JSON Valid | Mean Steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['policy']} | {row['mean_reward']:.4f} | {row['condition_accuracy']:.4f} | "
            f"{row['urgency_accuracy']:.4f} | {row['under_escalation_rate']:.4f} | "
            f"{row['json_valid_rate']:.4f} | {row['mean_steps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Do not claim trained model improvement from this file unless a trained row is present and beats the matched baseline.",
        ]
    )
    out_md = output_dir / "baseline_vs_trained_multiturn.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "markdown": str(out_md), "trained_rows_present": bool(trained_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

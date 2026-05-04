from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import MULTITURN_TRAJECTORIES, MultiTurnPrenatalEnvironment  # noqa: E402


def _step(env: MultiTurnPrenatalEnvironment, action: dict[str, Any]) -> dict[str, Any]:
    result = env.step(action)
    return {
        "action": action,
        "reward": result.reward,
        "done": result.done,
        "day": result.observation.episode_day_index,
        "predicted_condition": result.predicted_condition,
        "urgency": result.urgency,
        "reference_condition": result.reference_condition,
        "reference_urgency": result.reference_urgency,
        "under_escalated": result.under_escalated,
        "reward_components": result.reward_components,
    }


def run_smoke_eval() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for trajectory_id, trajectory in sorted(MULTITURN_TRAJECTORIES.items()):
        env = MultiTurnPrenatalEnvironment()
        prompt = env.reset(trajectory_id)
        trace = [
            {
                "event": "reset",
                "trajectory_id": trajectory_id,
                "day": prompt.observation.episode_day_index,
                "target_condition": trajectory.target_condition,
                "target_urgency": trajectory.target_urgency,
            }
        ]
        trace.append(_step(env, {"action_type": "advance_day", "rationale": "Smoke test day 2 transition."}))
        trace.append(_step(env, {"action_type": "advance_day", "rationale": "Smoke test day 3 transition."}))
        trace.append(
            _step(
                env,
                {
                    "action_type": "diagnose",
                    "target": trajectory.target_condition,
                    "urgency": trajectory.target_urgency,
                    "rationale": "Smoke test oracle final diagnosis.",
                },
            )
        )
        final = trace[-1]
        cases.append(
            {
                "trajectory_id": trajectory_id,
                "passed": bool(final["done"])
                and final["predicted_condition"] == trajectory.target_condition
                and final["urgency"] == trajectory.target_urgency
                and 0.0 <= float(final["reward"]) <= 1.0,
                "final_reward": final["reward"],
                "trace": trace,
            }
        )
    return {
        "ok": all(item["passed"] for item in cases),
        "cases": cases,
        "case_count": len(cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fast deterministic MAAS multi-turn smoke evaluation.")
    parser.add_argument("--output", default="results/smoke_eval.json")
    parser.add_argument("--markdown", default="results/smoke_eval.md")
    args = parser.parse_args()

    report = run_smoke_eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MAAS Smoke Evaluation",
        "",
        f"Overall: `{'pass' if report['ok'] else 'fail'}`",
        "",
        "| Trajectory | Pass | Final Reward |",
        "|---|---:|---:|",
    ]
    for case in report["cases"]:
        lines.append(f"| {case['trajectory_id']} | {str(case['passed']).lower()} | {float(case['final_reward']):.4f} |")
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "output": str(output), "markdown": args.markdown}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

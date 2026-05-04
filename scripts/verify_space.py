from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 20.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"raw": body[:500]}
            return {"ok": 200 <= response.status < 400, "status": response.status, "elapsed_ms": elapsed_ms, "body": parsed}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1), "body": body[:500]}
    except Exception as exc:
        return {"ok": False, "status": None, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1), "body": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the live MAAS OpenEnv Space/API.")
    parser.add_argument("--base-url", default="https://sparsh122-maas-openenv.hf.space", help="Base Space URL.")
    parser.add_argument("--output", default="results/demo_verification.md", help="Markdown report path.")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    checks = [
        ("health", "GET", f"{base}/health", None),
        ("reset", "POST", f"{base}/reset", {"trajectory_id": "traj_preeclampsia_slow"}),
        ("step_advance_day2", "POST", f"{base}/step", {"action_type": "advance_day", "rationale": "Verify day 2 transition."}),
        ("step_advance_day3", "POST", f"{base}/step", {"action_type": "advance_day", "rationale": "Verify day 3 transition."}),
        (
            "step_diagnose",
            "POST",
            f"{base}/step",
            {
                "action_type": "diagnose",
                "target": "preeclampsia",
                "urgency": "go_to_hospital_today",
                "rationale": "Verify final diagnosis response.",
            },
        ),
        ("state", "GET", f"{base}/state", None),
        ("openenv_demo", "GET", f"{base}/openenv-demo", None),
    ]

    results = []
    for name, method, url, payload in checks:
        result = _request(method, url, payload)
        results.append({"name": name, "method": method, "url": url, **result})

    passed = all(item["ok"] for item in results)
    diagnose_body = next((item["body"] for item in results if item["name"] == "step_diagnose"), {})
    reward_components = diagnose_body.get("reward_components", {}) if isinstance(diagnose_body, dict) else {}
    expected_reward_fields = {
        "raw_reward",
        "adjusted_raw_reward",
        "trajectory_condition_score",
        "trajectory_urgency_score",
        "trajectory_under_escalation_penalty",
        "trajectory_over_escalation_penalty",
        "premature_diagnosis_penalty",
    }
    missing_reward_fields = sorted(expected_reward_fields - set(reward_components))
    schema_current = not missing_reward_fields
    lines = [
        "# MAAS Demo Verification",
        "",
        f"Base URL: `{base}`",
        f"Endpoint status: `{'pass' if passed else 'fail'}`",
        f"Reward schema current: `{'yes' if schema_current else 'no'}`",
        f"Missing new reward fields: `{', '.join(missing_reward_fields) if missing_reward_fields else 'none'}`",
        "",
        "| Check | Method | Status | Latency ms | Result |",
        "|---|---|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['name']} | {item['method']} | {item['status']} | "
            f"{item['elapsed_ms']} | {'pass' if item['ok'] else 'fail'} |"
        )
    lines.extend(["", "## Raw Responses", ""])
    for item in results:
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(item["body"], indent=2)[:4000])
        lines.append("```")
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": passed, "output": str(output), "checks": results}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

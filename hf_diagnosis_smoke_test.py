#!/usr/bin/env python3
"""
Smoke-test one or more Hugging Face chat models on a maternal triage case.

Examples:
  $env:HF_TOKEN="hf_xxx"
  python hf_diagnosis_smoke_test.py

  python hf_diagnosis_smoke_test.py --model sparsh122/maas-grpo-qwen05b-fix2

  python hf_diagnosis_smoke_test.py --model Qwen/Qwen2-0.5B-Instruct --strict
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

from huggingface_hub import InferenceClient

DEFAULT_BASELINE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_TRAINED_MODEL = "sparsh122/maas-grpo-qwen05b-fix2"

SYSTEM_PROMPT = (
    "You are Niva, an AI maternal health assistant. "
    'Respond ONLY with JSON: {"action_type": "diagnose", "target": "<condition>", '
    '"urgency": "<urgency>"}'
)

DANGER_CASE_OBSERVATION = """Patient observation:
- Weeks pregnant: 34 (Trimester 3)
- Region: Bihar
- Risk flags: DANGER_BP_CRITICAL, DANGER_VISION_HEADACHE
- BP trend: rising
- Avg kick count: 8.0
- Avg meals/day: 2.0
- Avg sleep hrs: 5.5
- History: first_pregnancy
- Days of data: 4

Diagnose this patient. Respond ONLY with the JSON object."""

EXPECTED_ACTION = {
    "action_type": "diagnose",
    "target": "preeclampsia",
    "urgency": "go_to_hospital_today",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a maternal-health JSON diagnosis smoke test against one or more HF chat models."
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help=(
            "Model ID to test. Repeat to compare multiple models. "
            "If omitted, the script compares the default baseline and trained MAAS model."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--base-url",
        help="Optional OpenAI-compatible base URL or HF endpoint URL.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum completion tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Optional seed for more repeatable outputs.",
    )
    parser.add_argument(
        "--obs-file",
        help="Optional path to a text file containing the observation prompt.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any model errors or misses the expected action.",
    )
    return parser.parse_args()


def load_observation(obs_file: str | None) -> str:
    if not obs_file:
        return DANGER_CASE_OBSERVATION
    with open(obs_file, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def create_client(token: str | None, base_url: str | None) -> InferenceClient:
    if base_url:
        client_kwargs: dict[str, Any] = {"base_url": base_url}
        if token:
            client_kwargs["api_key"] = token
        return InferenceClient(**client_kwargs)

    client_kwargs = {}
    if token:
        client_kwargs["token"] = token
    return InferenceClient(**client_kwargs)


def create_chat_completion(client: InferenceClient, **kwargs: Any) -> Any:
    try:
        return client.chat.completions.create(**kwargs)
    except AttributeError:
        return client.chat_completion(**kwargs)


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = "\n".join(
            line for line in clean.splitlines() if not line.strip().startswith("```")
        ).strip()

    try:
        payload = json.loads(clean)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.S)
        if not match:
            return None
        try:
            payload = json.loads(match.group())
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None


def normalize_action(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None

    target = payload.get("target") or payload.get("condition")
    urgency = payload.get("urgency")
    action_type = payload.get("action_type")

    if not target or not urgency:
        return payload

    if action_type not in {"diagnose", "assess", "request_signal"}:
        action_type = "diagnose"

    return {
        "action_type": action_type,
        "target": target,
        "urgency": urgency,
    }


def run_model(
    client: InferenceClient,
    model_name: str,
    observation: str,
    max_tokens: int,
    temperature: float,
    seed: int | None,
) -> tuple[str, dict[str, Any] | None]:
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        request_kwargs["seed"] = seed

    response = create_chat_completion(client, **request_kwargs)
    raw_text = (response.choices[0].message.content or "").strip()
    return raw_text, normalize_action(parse_json_object(raw_text))


def selected_models(models: list[str] | None) -> list[str]:
    if models:
        return models
    return [DEFAULT_BASELINE_MODEL, DEFAULT_TRAINED_MODEL]


def main() -> int:
    args = parse_args()
    if not args.token:
        print(
            "No HF token provided. The request may still work for public endpoints, "
            "but gated or provider-routed models often require HF_TOKEN.",
            file=sys.stderr,
        )

    observation = load_observation(args.obs_file)
    models = selected_models(args.models)
    client = create_client(token=args.token, base_url=args.base_url)

    had_error = False
    had_mismatch = False

    print("OBSERVATION:")
    print(observation)
    print()
    print("EXPECTED:", json.dumps(EXPECTED_ACTION, ensure_ascii=True))
    print()

    for model_name in models:
        print("=" * 80)
        print(f"MODEL: {model_name}")
        try:
            raw_text, parsed_action = run_model(
                client=client,
                model_name=model_name,
                observation=observation,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
            matches_expected = parsed_action == EXPECTED_ACTION
            had_mismatch = had_mismatch or not matches_expected

            print("RAW:", raw_text or "<empty>")
            print(
                "PARSED:",
                json.dumps(parsed_action, ensure_ascii=True)
                if parsed_action is not None
                else "null",
            )
            print(f"MATCHES_EXPECTED: {str(matches_expected).lower()}")
        except Exception as exc:
            had_error = True
            print(f"ERROR: {exc}")

    if args.strict and (had_error or had_mismatch):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

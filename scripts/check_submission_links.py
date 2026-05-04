from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_LINKS = [
    "https://github.com/sparsh1258/MAAS",
    "https://huggingface.co/spaces/sparsh122/maas-openenv",
    "https://sparsh122-maas-openenv.hf.space/health",
    "https://sparsh122-maas-openenv.hf.space/openenv-demo",
    "https://huggingface.co/spaces/nancyyyyyyy/niva-prenatal-health",
    "https://sparsh122-maternaai.hf.space/coordinator",
    "https://huggingface.co/sparsh122/maas-grpo-hackathon-final",
]


def check_url(url: str, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "MAAS-submission-link-check/1.0"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1024)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            return {"url": url, "ok": 200 <= response.status < 400, "status": response.status, "elapsed_ms": elapsed_ms}
    except urllib.error.HTTPError as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {"url": url, "ok": False, "status": exc.code, "elapsed_ms": elapsed_ms, "error": str(exc)}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {"url": url, "ok": False, "status": None, "elapsed_ms": elapsed_ms, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MAAS submission links and write judge-facing evidence.")
    parser.add_argument("--links", nargs="*", default=DEFAULT_LINKS)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-json", default="results/submission_link_check.json")
    parser.add_argument("--output-md", default="results/submission_link_check.md")
    args = parser.parse_args()

    results = [check_url(url, args.timeout) for url in args.links]
    ok = all(item["ok"] for item in results)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({"ok": ok, "results": results}, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MAAS Submission Link Check",
        "",
        f"Overall: `{'pass' if ok else 'fail'}`",
        "",
        "| URL | Status | Latency ms | Result |",
        "|---|---:|---:|---|",
    ]
    for item in results:
        lines.append(f"| {item['url']} | {item['status']} | {item['elapsed_ms']} | {'pass' if item['ok'] else 'fail'} |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": ok, "json": args.output_json, "markdown": args.output_md}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

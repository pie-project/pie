#!/usr/bin/env python3

import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "benchmark-vllm-plex-policy.py"
PACKAGES = ROOT / "tests" / "policies" / "target" / "packages"
MODES = {
    "agentix": "schedule",
    "continuum": "cache-pressure",
    "kvflow": "cache-pressure",
    "helium": "schedule",
    "vtc": "schedule",
    "fairserve": "schedule",
    "marconi": "cache-pressure",
    "ragcache": "cache-pressure",
    "dlpm": "schedule",
    "infercept": "cache-pressure",
    "peek": "cache-pressure",
    "qlm": "schedule",
    "slos-serve": "schedule",
    "dynasor": "schedule",
    "justitia": "schedule",
    "chameleon": "cache-pressure",
    "hotprefix": "cache-pressure",
    "pard": "schedule",
    "branch-regulation": "schedule",
    "thunderagent": "schedule",
    "pythia": "schedule",
    "parrot": "schedule",
    "saga": "schedule",
}
WORKER_RE = re.compile(
    r"submitted=(\d+) dropped=(\d+) completed=(\d+) "
    r"success=(\d+) fallback=(\d+) unavailable=(\d+) rejected=(\d+) "
    r"schedule_success=(\d+) cache_success=(\d+) feedback_success=(\d+) "
    r"schedule_enacted=(\d+) schedule_partial=(\d+) cache_enacted=(\d+)"
)
WORKER_FIELDS = [
    "submitted",
    "dropped",
    "completed",
    "success",
    "fallback",
    "unavailable",
    "rejected",
    "schedule_success",
    "cache_success",
    "feedback_success",
    "schedule_enacted",
    "schedule_partial",
    "cache_enacted",
]


def package_path(policy_id: str) -> Path:
    return PACKAGES / f"plex_paper_{policy_id.replace('-', '_')}.plexpkg"


def run_one(
    args: argparse.Namespace,
    policy_id: str,
    mode: str,
    package: Path | None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    label = f"{policy_id}-{mode}"
    output = args.output_dir / f"{label}.json"
    log = args.output_dir / f"{label}.log"
    if not (args.resume and output.exists() and log.exists()):
        command = [
            str(args.python),
            str(RUNNER),
            "--policy-id",
            policy_id,
            "--mode",
            mode,
            "--repeats",
            str(args.repeats),
            "--output",
            str(output),
        ]
        if package is not None:
            command.extend(["--package", str(package)])
        env = os.environ.copy()
        env["PATH"] = f"{args.python.parent}:{env.get('PATH', '')}"
        env["VLLM_LOGGING_LEVEL"] = "INFO"
        with log.open("w") as handle:
            subprocess.run(
                command,
                cwd=args.vllm_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
    result = json.loads(output.read_text())
    matches = WORKER_RE.findall(log.read_text())
    worker = (
        dict(zip(WORKER_FIELDS, map(int, matches[-1])))
        if matches
        else None
    )
    return result, worker


def median_run_metric(result: dict[str, Any], key: str) -> float:
    return statistics.median(run["aggregate"][key] for run in result["runs"])


def compare(
    policy: dict[str, Any],
    baseline: dict[str, Any],
    worker: dict[str, int] | None,
) -> dict[str, Any]:
    policy_hashes = [run["token_sha256"] for run in policy["runs"]]
    baseline_hashes = [run["token_sha256"] for run in baseline["runs"]]
    policy_throughput = policy["median_throughput_tokens_per_second"]
    baseline_throughput = baseline["median_throughput_tokens_per_second"]
    return {
        "policy_id": policy["policy_id"],
        "mode": policy["mode"],
        "repeats": policy["repeats"],
        "throughput_tokens_per_second": policy_throughput,
        "native_throughput_tokens_per_second": baseline_throughput,
        "throughput_delta_percent": (
            policy_throughput / baseline_throughput - 1
        )
        * 100,
        "favored_ttft_s": median_run_metric(
            policy, "favored_mean_ttft_s"
        ),
        "native_favored_ttft_s": median_run_metric(
            baseline, "favored_mean_ttft_s"
        ),
        "favored_ttft_improvement_ratio": (
            median_run_metric(baseline, "favored_mean_ttft_s")
            / median_run_metric(policy, "favored_mean_ttft_s")
        ),
        "favored_queue_s": median_run_metric(
            policy, "favored_mean_queue_s"
        ),
        "native_favored_queue_s": median_run_metric(
            baseline, "favored_mean_queue_s"
        ),
        "token_outputs_equal": policy_hashes == baseline_hashes,
        "worker": worker,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policy", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.policy or list(MODES)
    unknown = set(selected) - MODES.keys()
    if unknown:
        raise SystemExit(f"unknown live policies: {sorted(unknown)}")

    baselines = {}
    for mode in sorted({MODES[policy_id] for policy_id in selected}):
        baselines[mode], _ = run_one(args, "native", mode, None)

    comparisons = []
    for policy_id in selected:
        mode = MODES[policy_id]
        package = package_path(policy_id)
        if not package.exists():
            raise SystemExit(f"missing package {package}")
        measured, worker = run_one(args, policy_id, mode, package)
        comparison = compare(measured, baselines[mode], worker)
        comparisons.append(comparison)
        print(
            f"{policy_id}: throughput={comparison['throughput_delta_percent']:+.3f}% "
            f"favored-ttft={comparison['favored_ttft_improvement_ratio']:.3f}x "
            f"equal={comparison['token_outputs_equal']}"
        )

    report = {
        "schema_version": 1,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "repeats": args.repeats,
        "policy_count": len(comparisons),
        "comparisons": comparisons,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

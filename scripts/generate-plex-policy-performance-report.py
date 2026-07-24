#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path | None) -> dict[str, Any]:
    return json.loads(path.read_text()) if path is not None else {}


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    if value > 999:
        return ">999x"
    return f"{value:.3f}x"


def format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:+.3f}%"


def format_live_mechanism(live: dict[str, Any] | None) -> str:
    if live is None or live.get("worker") is None:
        return "-"
    worker = live["worker"]
    return (
        f"S{worker['schedule_success']}/C{worker['cache_success']}/"
        f"F{worker['feedback_success']}; "
        f"enact {worker['schedule_enacted'] + worker['cache_enacted']}; "
        f"drop {worker['dropped']}; fallback {worker['fallback']}"
    )


def build_report(
    offline: dict[str, Any],
    live: dict[str, Any],
    fidelity: dict[str, Any],
    replication: dict[str, Any],
    robustness: dict[str, Any],
    pie_commit: str | None,
    vllm_commit: str | None,
) -> dict[str, Any]:
    live_by_id = {
        entry["policy_id"]: entry
        for entry in live.get("comparisons", [])
    }
    fidelity_by_id = {
        entry["id"]: entry
        for entry in fidelity.get("entries", [])
    }
    replication_by_id = {
        entry["id"]: entry
        for entry in replication.get("entries", [])
    }
    entries = []
    for offline_entry in offline["results"]:
        policy_id = offline_entry["id"]
        measurement = offline_entry["measurement"]
        live_entry = live_by_id.get(policy_id)
        fidelity_entry = fidelity_by_id.get(policy_id)
        evidence_level = replication_by_id.get(policy_id, {}).get(
            "evidence_level", "inspired-adaptation"
        )
        entries.append(
            {
                "id": policy_id,
                "paper_claim": offline_entry["paper_claim"],
                "benchmark_family": offline_entry["benchmark_family"],
                "primary_metric": offline_entry["primary_metric"],
                "baseline": offline_entry["baseline"],
                "offline": {
                    "unit": measurement["unit"],
                    "policy_value": measurement["policy_value"],
                    "baseline_value": measurement["baseline_value"],
                    "proxy_improvement_ratio": measurement[
                        "improvement_ratio"
                    ],
                    "win_rate": measurement["win_rate"],
                    "wins": measurement["wins"],
                    "ties": measurement["ties"],
                    "losses": measurement["losses"],
                    "trend_reproduced": measurement["trend_reproduced"],
                    "decision_latency": offline_entry["decision_latency"],
                    "operation_counts": offline_entry["operation_counts"],
                    "package_sha256": offline_entry["package_sha256"],
                    "robustness": robustness.get(policy_id),
                },
                "live": live_entry,
                "fidelity": fidelity_entry,
                "evidence": (
                    f"{evidence_level}-proxy+live-mechanism"
                    if live_entry is not None
                    else f"{evidence_level}-proxy"
                ),
                "paper_end_to_end_ratio_reproduced": False,
            }
        )
    fidelity_counts = Counter(
        entry["fidelity"]["classification"]
        if entry["fidelity"] is not None
        else "pending-independent-review"
        for entry in entries
    )
    live_entries = [entry["live"] for entry in entries if entry["live"] is not None]
    return {
        "schema_version": 1,
        "pie_commit": pie_commit,
        "vllm_commit": vllm_commit,
        "offline_environment": offline.get("environment"),
        "live_model": live.get("model"),
        "policy_count": len(entries),
        "offline_trend_reproduced_count": sum(
            entry["offline"]["trend_reproduced"] for entry in entries
        ),
        "live_policy_count": sum(entry["live"] is not None for entry in entries),
        "live_output_equal_count": sum(
            entry["token_outputs_equal"] for entry in live_entries
        ),
        "live_zero_drop_count": sum(
            entry.get("worker") is not None
            and entry["worker"]["dropped"] == 0
            for entry in live_entries
        ),
        "live_zero_fallback_count": sum(
            entry.get("worker") is not None
            and entry["worker"]["fallback"] == 0
            for entry in live_entries
        ),
        "live_median_throughput_delta_percent": (
            sorted(
                entry["throughput_delta_percent"]
                for entry in live_entries
            )[len(live_entries) // 2]
            if live_entries
            else None
        ),
        "robust_seed_count": max(
            (
                entry["offline"]["robustness"]["seed_count"]
                for entry in entries
                if entry["offline"]["robustness"] is not None
            ),
            default=0,
        ),
        "robust_trend_count": sum(
            entry["offline"]["robustness"] is not None
            and entry["offline"]["robustness"]["trend_stable"]
            for entry in entries
        ),
        "paper_end_to_end_ratio_reproduced_count": 0,
        "fidelity_counts": dict(sorted(fidelity_counts.items())),
        "entries": entries,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PLEX 31-policy performance reproduction evidence",
        "",
        "This report separates three claims:",
        "",
        "1. **Policy-kernel proxy trend**: the compiled `.plexpkg` beats the declared",
        "   baseline on a deterministic paper-anchored synthetic workload.",
        "2. **Live-engine mechanism**: the policy executes on A100 vLLM without",
        "   queue drops or unexpected fallback and preserves output tokens.",
        "3. **Paper end-to-end ratio**: the original system, dataset, scale,",
        "   hardware, and baseline are reproduced closely enough to compare the",
        "   reported numeric ratio.",
        "",
        "Proxy ratios are not presented as the original paper ratios.",
        "",
        f"- Pie commit: `{report['pie_commit'] or 'unspecified'}`",
        f"- vLLM commit: `{report['vllm_commit'] or 'unspecified'}`",
        f"- Policies: {report['policy_count']}",
        f"- Offline proxy trends reproduced: {report['offline_trend_reproduced_count']}",
        f"- Live vLLM policies: {report['live_policy_count']}",
        f"- Live output-equivalent policies: {report['live_output_equal_count']}",
        f"- Live zero-drop / zero-fallback policies: "
        f"{report['live_zero_drop_count']} / {report['live_zero_fallback_count']}",
        f"- Median live throughput delta: "
        f"{format_percent(report['live_median_throughput_delta_percent'])}",
        f"- Multi-seed stable trends: {report['robust_trend_count']} "
        f"across {report['robust_seed_count']} additional seeds",
        "- Exact paper end-to-end ratios reproduced: 0 (current milestone)",
        "",
        "## Results",
        "",
        "| Policy | Paper north star | Proxy | Win rate | Decision p50 | Live throughput | Live mechanism | Fidelity |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for entry in report["entries"]:
        live = entry["live"]
        fidelity = entry["fidelity"]
        claim = entry["paper_claim"].replace("|", "\\|")
        lines.append(
            f"| `{entry['id']}` | {claim} | "
            f"{format_ratio(entry['offline']['proxy_improvement_ratio'])} | "
            f"{entry['offline']['win_rate']:.1%} | "
            f"{entry['offline']['decision_latency']['median_us']:.1f} us | "
            f"{format_percent(live['throughput_delta_percent'] if live else None)} | "
            f"{format_live_mechanism(live)} | "
            f"{fidelity['classification'] if fidelity else 'pending'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive proxy result proves that the committed policy kernel",
            "implements a useful ordering, admission, routing, reclaim, or",
            "feedback mechanism on the declared scenario. It does not prove the",
            "paper's full-system speedup when predictor training, migration,",
            "multi-GPU execution, cache movement, provisioning, or private traces",
            "remain deferred.",
            "",
            "The machine-readable companion records per-trial wins/losses,",
            "decision latency, live worker counters, output equivalence, and",
            "independent fidelity findings.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--live", type=Path)
    parser.add_argument("--fidelity", type=Path)
    parser.add_argument("--replication", type=Path)
    parser.add_argument("--robustness-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--pie-commit")
    parser.add_argument("--vllm-commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robustness: dict[str, Any] = {}
    if args.robustness_dir is not None:
        per_policy: dict[str, list[bool]] = {}
        for path in sorted(args.robustness_dir.glob("*.json")):
            for entry in read_json(path)["results"]:
                per_policy.setdefault(entry["id"], []).append(
                    entry["measurement"]["trend_reproduced"]
                )
        robustness = {
            policy_id: {
                "seed_count": len(outcomes),
                "trend_stable": all(outcomes),
            }
            for policy_id, outcomes in per_policy.items()
        }
    report = build_report(
        read_json(args.offline),
        read_json(args.live),
        read_json(args.fidelity),
        read_json(args.replication),
        robustness,
        args.pie_commit,
        args.vllm_commit,
    )
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    args.output_md.write_text(markdown(report))


if __name__ == "__main__":
    main()

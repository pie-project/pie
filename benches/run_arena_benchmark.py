#!/usr/bin/env python3
"""Arena-style benchmark runner for the custom test-time-scaling inferlets.

The professor's reference is Agent Arena by Arena/LMArena. The real Agent Arena
is a live, web-based platform: an "agent" is a (model + framework + tools)
configuration (e.g. GPT-4o + LangChain + Brave Search), and agents are ranked by
pairwise human votes on community tasks from its Prompt Hub, scored with an
Extended Bradley-Terry model that attributes credit to the model, framework, and
tool components. See https://arena.ai/blog/agent-arena/.

Because that leaderboard is produced by live human preference votes, it cannot be
reproduced offline: there is no public Agent Arena task export or scoring API to
run locally against these inferlets. Instead, this runner
creates a local, deterministic Arena-style harness around the three inferlet
families that mirrors Agent Arena's tool-using workflow categories. The five
proxy signals below (task success, steerability, error recovery, user-feedback,
tool hallucination) are OUR local design choices, not numbers reported by Arena:

  * modular-cache: repeated related launches on one persistent engine
  * hierarchical-attention: long-context task prompts with selected visibility
  * mcts: retry/search-style reasoning over verifiable agent tasks

It writes JSONL and CSV rows with common proxy metrics plus family-specific
control-flow metrics parsed from the inferlet logs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PIE_REPO = ROOT
# Canonical Agent Arena reference (Arena / LMArena). This is the live, human-voted
# platform we mirror locally; it is not an offline benchmark we can score against.
ARENA_REFERENCE = "https://arena.ai/blog/agent-arena/"

LANGUAGES = ("rust", "python", "js")
FAMILIES = ("modular-cache", "hierarchical-attention", "mcts")


TASKS: list[dict[str, Any]] = [
    {
        "id": "code-debug-test-failure",
        "category": "coding_debugging",
        "workflow": "Diagnose a failing unit test from a terminal log and patch notes.",
        "stable_context": (
            "Repository note: a scheduler changed from inclusive to exclusive end "
            "indices. Terminal log: test_window_end fails with expected [2, 3, 4] "
            "but got [2, 3]. The likely bug is an off-by-one range end."
        ),
        "initial_request": (
            "Explain the smallest code fix and the regression test that should be "
            "added. Mention the off-by-one range end."
        ),
        "followup_request": (
            "Now phrase the same fix as a short pull-request review comment."
        ),
        "required_terms": ["off-by-one", "range", "test"],
        "forbidden_terms": ["i ran pytest", "i opened the file"],
        "error_terms": ["fails", "expected", "got"],
    },
    {
        "id": "web-research-source-triage",
        "category": "web_research",
        "workflow": "Synthesize a web-research answer from supplied search snippets.",
        "stable_context": (
            "Search snippets supplied by the environment: Source A says Agent Arena "
            "uses real sessions with web search, filesystem, and terminal tools. "
            "Source B says its signals include task success, steerability, error "
            "recovery, user praise versus complaint, and tool hallucination."
        ),
        "initial_request": (
            "Summarize the benchmark idea in three sentences and name the five "
            "evaluation signals."
        ),
        "followup_request": (
            "Turn the same research into a concise note for an engineering team."
        ),
        "required_terms": ["task success", "steerability", "tool hallucination"],
        "forbidden_terms": ["i searched the web", "live browser"],
        "error_terms": [],
    },
    {
        "id": "document-analysis-action-items",
        "category": "document_analysis",
        "workflow": "Extract action items from a provided project memo.",
        "stable_context": (
            "Project memo: We verified smoke tests with the dummy driver. Next, "
            "we need an Arena-style benchmark runner, JSONL/CSV output, a real-model "
            "path, and documentation that dummy results are not quality claims."
        ),
        "initial_request": (
            "Extract the action items and separate completed smoke-test work from "
            "remaining benchmark work."
        ),
        "followup_request": (
            "Rewrite the action items as next-week priorities with owners omitted."
        ),
        "required_terms": ["jsonl", "csv", "dummy"],
        "forbidden_terms": ["attached pdf", "i read the file system"],
        "error_terms": [],
    },
    {
        "id": "app-build-acceptance-criteria",
        "category": "app_building",
        "workflow": "Turn app requirements into implementation acceptance criteria.",
        "stable_context": (
            "App request: build a small benchmark dashboard with task filters, "
            "per-inferlet rows, pass/fail status, raw-output links, and a chart for "
            "task-success proxy by family. Constraint: no marketing landing page."
        ),
        "initial_request": (
            "Write implementation acceptance criteria for the dashboard."
        ),
        "followup_request": (
            "Condense the acceptance criteria into a QA checklist."
        ),
        "required_terms": ["filters", "pass", "raw-output"],
        "forbidden_terms": ["hero section", "landing page"],
        "error_terms": [],
    },
    {
        "id": "slide-deck-outline",
        "category": "slide_deck",
        "workflow": "Create a short slide outline from project notes.",
        "stable_context": (
            "Talk notes: introduce Pie inferlets, explain the three examples, show "
            "how Arena-style evaluation differs from dummy smoke tests, then close "
            "with limitations and next steps."
        ),
        "initial_request": (
            "Create a five-slide outline with one key takeaway per slide."
        ),
        "followup_request": (
            "Rewrite the outline for a professor-facing project update."
        ),
        "required_terms": ["pie", "arena", "limitations"],
        "forbidden_terms": ["image generation", "speaker notes created"],
        "error_terms": [],
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _excerpt(text: str, limit: int = 500) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _last_int(pattern: str, text: str) -> int | None:
    values = re.findall(pattern, text)
    return int(values[-1]) if values else None


def _last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None


def inferlet_name(family: str, language: str) -> str:
    return f"{family}-{language}"


def make_modules(task: dict[str, Any], request: str) -> list[dict[str, Any]]:
    """Build stable modules plus one task module for modular-cache runs."""
    return [
        {
            "id": "system/arena-evaluator",
            "role": "system",
            "deps": [],
            "text": (
                "You are evaluating an agentic workflow. Use only the supplied "
                "task notes and tool transcript. Do not claim to have used tools."
            ),
        },
        {
            "id": "policy/output",
            "role": "user",
            "deps": ["system/arena-evaluator"],
            "text": (
                "Prefer concise, checkable answers. If an environment error is "
                "shown, explain the recovery step."
            ),
        },
        {
            "id": f"context/{task['category']}",
            "role": "user",
            "deps": ["policy/output"],
            "text": f"Workflow category: {task['category']}. {task['workflow']}",
        },
        {
            "id": f"evidence/{task['id']}",
            "role": "user",
            "deps": [f"context/{task['category']}"],
            "text": task["stable_context"],
        },
        {
            "id": "task/current",
            "role": "user",
            "deps": [f"evidence/{task['id']}"],
            "text": request,
        },
    ]


def make_ha_prompt(task: dict[str, Any]) -> str:
    distractors = [
        (
            "Chunk: deployment note. The build cache stores generated artifacts "
            "that should not be committed. This is unrelated to the answer."
        ),
        (
            "Chunk: meeting note. The team prefers short updates, plain language, "
            "and direct limitations when reporting prototype results."
        ),
        (
            "Chunk: benchmark evidence. "
            + task["stable_context"]
            + " Request: "
            + task["initial_request"]
        ),
        (
            "Chunk: unrelated research. A generic leaderboard may rank chat "
            "quality, but this task is about tool-using agent workflows."
        ),
    ]
    return (
        "You are answering an Arena-style agent evaluation task. Use the relevant "
        "chunk and ignore unrelated chunks.\n\n"
        + "\n\n".join(distractors)
        + "\n\nFinal question: "
        + task["initial_request"]
    )


def make_mcts_prompt(task: dict[str, Any]) -> str:
    return (
        f"Agent task category: {task['category']}\n"
        f"Workflow: {task['workflow']}\n"
        f"Evidence:\n{task['stable_context']}\n\n"
        f"Goal: {task['initial_request']}\n"
        "Return a concise answer that would pass an objective project review."
    )


def proxy_grade(output: str, task: dict[str, Any]) -> dict[str, Any]:
    """Local proxies for Arena-style quality signals.

    These are intentionally simple and deterministic. They are not equivalent
    to Arena's live user-feedback or causal leaderboard methodology.
    """
    lower = output.lower()
    required = [term.lower() for term in task.get("required_terms", [])]
    forbidden = [term.lower() for term in task.get("forbidden_terms", [])]
    error_terms = [term.lower() for term in task.get("error_terms", [])]

    required_hits = [term for term in required if term in lower]
    forbidden_hits = [term for term in forbidden if term in lower]
    error_hits = [term for term in error_terms if term in lower]

    hallucination_phrases = [
        "i searched",
        "i ran",
        "i opened",
        "i clicked",
        "browser showed",
        "terminal output shows",
    ]
    hallucinations = [phrase for phrase in hallucination_phrases if phrase in lower]

    task_success = 1.0 if required and len(required_hits) == len(required) else 0.0
    if not required:
        task_success = 1.0 if output.strip() else 0.0

    steerability = 1.0 if not forbidden_hits else 0.0
    error_recovery = 1.0
    if error_terms:
        recovery_words = ("fix", "recover", "correct", "because", "root cause")
        error_recovery = 1.0 if error_hits and any(w in lower for w in recovery_words) else 0.0

    tool_hallucination = 1.0 if hallucinations else 0.0
    user_feedback = (task_success + steerability + error_recovery + (1.0 - tool_hallucination)) / 4.0

    return {
        "task_success_proxy": round(task_success, 3),
        "steerability_proxy": round(steerability, 3),
        "error_recovery_proxy": round(error_recovery, 3),
        "user_feedback_proxy": round(user_feedback, 3),
        "tool_hallucination_proxy": round(tool_hallucination, 3),
        "required_hits": required_hits,
        "forbidden_hits": forbidden_hits,
        "tool_hallucination_hits": hallucinations,
    }


def parse_modular_metrics(output: str) -> dict[str, Any]:
    module_values = re.findall(r"(?m)^modules=(\d+)", output)
    return {
        "modules": int(module_values[-1]) if module_values else None,
        "cache_miss_seen": "cache_miss" in output,
        "cache_hit_modules": _last_int(r"cache_hit_modules=(\d+)", output),
        "saved_count": len(re.findall(r"\bsaved=", output)),
        "save_skipped_count": len(re.findall(r"\bsave_skipped=", output)),
        "use_cache_false_seen": "use_cache=false" in output,
    }


def modular_first_launch_ok(output: str, expected_full_hit: int) -> bool:
    """First cache-reuse launch can miss or reuse a shorter warm prefix.

    In a full benchmark suite, earlier tasks can warm shared system/policy
    modules in the same persistent engine. That is valid as long as the first
    launch is not already a full hit for this exact task.
    """
    metrics = parse_modular_metrics(output)
    first_hit = metrics.get("cache_hit_modules")
    return "cache_miss" in output or (first_hit is not None and first_hit < expected_full_hit)


def parse_ha_metrics(output: str) -> dict[str, Any]:
    selected_match = re.search(r"selected_chunk=\[([^\]]*)\]", output)
    selected = []
    if selected_match:
        selected = [int(x) for x in re.findall(r"\d+", selected_match.group(1))]

    mask_match = re.search(r"mask_true_tokens=(\d+)\s*/\s*total=(\d+)", output)
    true_tokens = int(mask_match.group(1)) if mask_match else None
    total_tokens = int(mask_match.group(2)) if mask_match else None
    visible_ratio = None
    if true_tokens is not None and total_tokens:
        visible_ratio = round(true_tokens / total_tokens, 4)

    return {
        "chunks": _last_int(r"chunks=(\d+)", output),
        "selected_chunks": selected,
        "mask_true_tokens": true_tokens,
        "mask_total_tokens": total_tokens,
        "visible_token_ratio": visible_ratio,
        "generated_tokens": _last_int(r"generated_tokens=(\d+)", output),
    }


def parse_mcts_metrics(output: str) -> dict[str, Any]:
    rollout_scores = [float(x) for x in re.findall(r"rollout_score=([0-9.]+)", output)]
    return {
        "iterations": _last_int(r"iterations=(\d+)", output),
        "nodes": _last_int(r"nodes=(\d+)", output),
        "best_score": _last_float(r"best_score=([0-9.]+)", output),
        "rollout_scores": rollout_scores,
        "score_variance_seen": len(set(rollout_scores)) > 1,
    }


def base_row(family: str, language: str, inferlet: str, task: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "arena_reference": ARENA_REFERENCE,
        "benchmark_kind": "local_arena_style_proxy",
        "family": family,
        "language": language,
        "inferlet": inferlet,
        "mode": mode,
        "task_id": task["id"],
        "task_category": task["category"],
        "workflow": task["workflow"],
    }


async def launch_case(run_inferlet_fn, client, inferlet: str, params: dict[str, Any], timeout: int) -> tuple[str, str, float]:
    start = time.time()
    output = await run_inferlet_fn(client, inferlet, params, timeout=timeout)
    return output, "pass", time.time() - start


def failure_row(
    family: str,
    language: str,
    inferlet: str,
    task: dict[str, Any],
    mode: str,
    exc: Exception,
    elapsed: float,
) -> dict[str, Any]:
    row = base_row(family, language, inferlet, task, mode)
    row.update(
        {
            "status": "fail",
            "control_flow_ok": False,
            "runtime_seconds": round(elapsed, 3),
            "error": str(exc)[:500],
        }
    )
    return row


async def run_modular_cache(run_inferlet_fn, client, language: str, task: dict[str, Any], args) -> list[dict[str, Any]]:
    family = "modular-cache"
    inferlet = inferlet_name(family, language)
    rows = []

    baseline_params = {
        "modules": make_modules(task, task["initial_request"]),
        "max_tokens": args.max_tokens,
        "use_cache": False,
        "save_cache": False,
    }
    start = time.time()
    try:
        output, status, elapsed = await launch_case(
            run_inferlet_fn, client, inferlet, baseline_params, args.timeout
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(failure_row(family, language, inferlet, task, "baseline_no_cache", exc, time.time() - start))
    else:
        row = base_row(family, language, inferlet, task, "baseline_no_cache")
        row.update(proxy_grade(output, task))
        row.update(parse_modular_metrics(output))
        row.update(
            {
                "status": status,
                "control_flow_ok": "use_cache=false" in output and "cache_hit_modules" not in output,
                "runtime_seconds": round(elapsed, 3),
                "launches": 1,
                "output_excerpt": _excerpt(output),
            }
        )
        rows.append(row)

    modules_first = make_modules(task, task["initial_request"])
    modules_followup = make_modules(task, task["followup_request"])
    expected_full = len(modules_first)
    expected_partial = len(modules_first) - 1
    params_first = {"modules": modules_first, "max_tokens": args.max_tokens}
    params_followup = {"modules": modules_followup, "max_tokens": args.max_tokens}

    start = time.time()
    try:
        out1, _, _ = await launch_case(run_inferlet_fn, client, inferlet, params_first, args.timeout)
        out2, _, _ = await launch_case(run_inferlet_fn, client, inferlet, params_first, args.timeout)
        out3, _, _ = await launch_case(run_inferlet_fn, client, inferlet, params_followup, args.timeout)
        elapsed = time.time() - start
    except Exception as exc:  # noqa: BLE001
        rows.append(failure_row(family, language, inferlet, task, "method_cache_reuse", exc, time.time() - start))
    else:
        hit_full = parse_modular_metrics(out2).get("cache_hit_modules")
        hit_partial = parse_modular_metrics(out3).get("cache_hit_modules")
        first_metrics = parse_modular_metrics(out1)
        first_hit = first_metrics.get("cache_hit_modules")
        first_launch_ok = modular_first_launch_ok(out1, expected_full)
        combined = "\n".join([out1, out2, out3])
        row = base_row(family, language, inferlet, task, "method_cache_reuse")
        row.update(proxy_grade(out3, task))
        row.update(parse_modular_metrics(combined))
        row.update(
            {
                "status": "pass",
                "control_flow_ok": (
                    first_launch_ok
                    and hit_full == expected_full
                    and hit_partial == expected_partial
                ),
                "runtime_seconds": round(elapsed, 3),
                "launches": 3,
                "first_launch_hit_modules": first_hit,
                "expected_full_hit_modules": expected_full,
                "full_hit_modules": hit_full,
                "expected_partial_hit_modules": expected_partial,
                "partial_hit_modules": hit_partial,
                "output_excerpt": _excerpt(combined),
            }
        )
        rows.append(row)

    return rows


async def run_hierarchical_attention(run_inferlet_fn, client, language: str, task: dict[str, Any], args) -> list[dict[str, Any]]:
    family = "hierarchical-attention"
    inferlet = inferlet_name(family, language)
    rows = []
    prompt = make_ha_prompt(task)
    common = {
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "chunk_size_words": args.ha_chunk_words,
        "sink_tokens": args.ha_sink_tokens,
        "summary_tokens_per_chunk": args.ha_summary_tokens,
        "local_window_tokens": args.ha_local_window_tokens,
    }

    modes = [
        ("baseline_all_visible", {**common, "selected_chunks": 999, "selection_mode": "all-visible-baseline"}),
        ("method_lexical_selection", {**common, "selected_chunks": args.ha_selected_chunks, "selection_mode": "lexical"}),
    ]
    for mode, params in modes:
        start = time.time()
        try:
            output, status, elapsed = await launch_case(run_inferlet_fn, client, inferlet, params, args.timeout)
        except Exception as exc:  # noqa: BLE001
            rows.append(failure_row(family, language, inferlet, task, mode, exc, time.time() - start))
            continue

        metrics = parse_ha_metrics(output)
        row = base_row(family, language, inferlet, task, mode)
        row.update(proxy_grade(output, task))
        row.update(metrics)
        row.update(
            {
                "status": status,
                "control_flow_ok": bool(metrics.get("chunks")) and metrics.get("mask_true_tokens") is not None,
                "runtime_seconds": round(elapsed, 3),
                "launches": 1,
                "output_excerpt": _excerpt(output),
            }
        )
        rows.append(row)

    return rows


async def run_mcts(run_inferlet_fn, client, language: str, task: dict[str, Any], args) -> list[dict[str, Any]]:
    family = "mcts"
    inferlet = inferlet_name(family, language)
    rows = []
    prompt = make_mcts_prompt(task)
    modes = [
        (
            "baseline_single_search",
            {
                "prompt": prompt,
                "max_iterations": 1,
                "max_depth": 1,
                "branch_factor": 1,
                "rollout_tokens": args.mcts_rollout_tokens,
                "final_tokens": args.max_tokens,
                "show_trace": True,
            },
        ),
        (
            "method_mcts_search",
            {
                "prompt": prompt,
                "max_iterations": args.mcts_iterations,
                "max_depth": args.mcts_depth,
                "branch_factor": args.mcts_branch_factor,
                "rollout_tokens": args.mcts_rollout_tokens,
                "final_tokens": args.max_tokens,
                "show_trace": True,
            },
        ),
    ]
    for mode, params in modes:
        start = time.time()
        try:
            output, status, elapsed = await launch_case(run_inferlet_fn, client, inferlet, params, args.timeout)
        except Exception as exc:  # noqa: BLE001
            rows.append(failure_row(family, language, inferlet, task, mode, exc, time.time() - start))
            continue

        metrics = parse_mcts_metrics(output)
        row = base_row(family, language, inferlet, task, mode)
        row.update(proxy_grade(output, task))
        row.update(metrics)
        row.update(
            {
                "status": status,
                "control_flow_ok": bool(metrics.get("iterations")) and metrics.get("nodes") is not None,
                "runtime_seconds": round(elapsed, 3),
                "launches": 1,
                "output_excerpt": _excerpt(output),
            }
        )
        rows.append(row)

    return rows


def parse_csv_list(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    raw = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in raw if v not in allowed]
    if unknown:
        raise SystemExit(f"Unknown {label}: {', '.join(unknown)}. Allowed: {', '.join(allowed)}")
    return raw


def selected_tasks(args) -> list[dict[str, Any]]:
    if args.tasks == "all":
        tasks = list(TASKS)
    else:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [task for task in TASKS if task["id"] in wanted or task["category"] in wanted]
        missing = wanted - {task["id"] for task in tasks} - {task["category"] for task in tasks}
        if missing:
            raise SystemExit(f"Unknown task ids/categories: {', '.join(sorted(missing))}")
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    return tasks


def write_results(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "arena_results.jsonl"
    csv_path = out_dir / "arena_results.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def print_summary(rows: list[dict[str, Any]], out_dir: Path | None = None) -> None:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("status") == "pass")
    control_ok = sum(1 for row in rows if row.get("control_flow_ok") is True)
    print(f"\nArena-style rows: {total}")
    print(f"Launch status pass: {passed}/{total}")
    print(f"Control-flow pass: {control_ok}/{total}")
    if out_dir is not None:
        print(f"JSONL: {out_dir / 'arena_results.jsonl'}")
        print(f"CSV:  {out_dir / 'arena_results.csv'}")


def dry_run_plan(args) -> list[dict[str, Any]]:
    families = parse_csv_list(args.families, FAMILIES, "families")
    languages = parse_csv_list(args.languages, LANGUAGES, "languages")
    tasks = selected_tasks(args)
    plan = []
    for task in tasks:
        for family in families:
            for language in languages:
                plan.append(
                    {
                        "family": family,
                        "language": language,
                        "inferlet": inferlet_name(family, language),
                        "task_id": task["id"],
                        "task_category": task["category"],
                    }
                )
    return plan


def load_pie_helpers():
    sys.path.insert(0, str(PIE_REPO / "client" / "python" / "src"))
    sys.path.insert(0, str(PIE_REPO / "tests" / "inferlets"))
    from pie_client import PieClient  # noqa: PLC0415
    from conftest import run_inferlet  # noqa: PLC0415

    return PieClient, run_inferlet


async def run(args) -> int:
    families = parse_csv_list(args.families, FAMILIES, "families")
    languages = parse_csv_list(args.languages, LANGUAGES, "languages")
    tasks = selected_tasks(args)

    PieClient, run_inferlet_fn = load_pie_helpers()
    client = PieClient(args.url)
    await client.connect()
    if args.token and args.token.lower() != "none":
        await client.auth_by_token(args.token)

    rows: list[dict[str, Any]] = []
    try:
        for task in tasks:
            print(f"\n== {task['id']} ({task['category']}) ==")
            for language in languages:
                for family in families:
                    print(f"  {family}-{language}", flush=True)
                    if family == "modular-cache":
                        rows.extend(await run_modular_cache(run_inferlet_fn, client, language, task, args))
                    elif family == "hierarchical-attention":
                        rows.extend(await run_hierarchical_attention(run_inferlet_fn, client, language, task, args))
                    elif family == "mcts":
                        rows.extend(await run_mcts(run_inferlet_fn, client, language, task, args))
    finally:
        await client.close()

    out_dir = Path(args.out_dir)
    write_results(rows, out_dir)
    print_summary(rows, out_dir)
    return 0 if all(row.get("status") == "pass" for row in rows) else 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local Arena-style proxy benchmarks for the custom Pie inferlets."
    )
    parser.add_argument("url", nargs="?", help="Pie websocket URL, e.g. ws://127.0.0.1:8080")
    parser.add_argument("token", nargs="?", default="none", help="Pie internal token, or none")
    parser.add_argument("--dry-run", action="store_true", help="Print/write the planned benchmark matrix only")
    parser.add_argument("--out-dir", default="benchmark_results/arena", help="Output directory for JSONL/CSV results")
    parser.add_argument("--families", default=",".join(FAMILIES), help="Comma-separated inferlet families")
    parser.add_argument("--languages", default=",".join(LANGUAGES), help="Comma-separated languages")
    parser.add_argument("--tasks", default="all", help="all, or comma-separated task ids/categories")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks for a quick smoke run")
    parser.add_argument("--timeout", type=int, default=240, help="Timeout per inferlet launch in seconds")
    parser.add_argument("--max-tokens", type=int, default=16, help="Small generation cap per launch")
    parser.add_argument("--ha-chunk-words", type=int, default=45)
    parser.add_argument("--ha-selected-chunks", type=int, default=1)
    parser.add_argument("--ha-sink-tokens", type=int, default=48)
    parser.add_argument("--ha-summary-tokens", type=int, default=20)
    parser.add_argument("--ha-local-window-tokens", type=int, default=96)
    parser.add_argument("--mcts-iterations", type=int, default=3)
    parser.add_argument("--mcts-depth", type=int, default=2)
    parser.add_argument("--mcts-branch-factor", type=int, default=2)
    parser.add_argument("--mcts-rollout-tokens", type=int, default=32)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    if args.dry_run:
        plan = dry_run_plan(args)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / "arena_plan.json"
        plan_path.write_text(json.dumps({"arena_reference": ARENA_REFERENCE, "plan": plan}, indent=2), encoding="utf-8")
        print(f"Planned benchmark cases: {len(plan)}")
        print(f"Plan: {plan_path}")
        for item in plan[:10]:
            print(f"  {item['inferlet']} :: {item['task_id']}")
        if len(plan) > 10:
            print(f"  ... {len(plan) - 10} more")
        return 0

    if not args.url:
        parser.error("url is required unless --dry-run is set")

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

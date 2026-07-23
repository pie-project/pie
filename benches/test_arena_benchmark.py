"""Pure-helper tests for the Arena-style benchmark runner.

Run directly:

    python3 tests/test_arena_benchmark.py
"""

import importlib.util
import sys
from pathlib import Path


def _load_runner():
    path = Path(__file__).resolve().parent / "run_arena_benchmark.py"
    spec = importlib.util.spec_from_file_location("arena_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = _load_runner()


def test_make_modules_keeps_only_task_suffix_variable():
    task = B.TASKS[0]
    a = B.make_modules(task, task["initial_request"])
    b = B.make_modules(task, task["followup_request"])
    assert len(a) == 5
    assert [m["id"] for m in a] == [m["id"] for m in b]
    assert [m["text"] for m in a[:-1]] == [m["text"] for m in b[:-1]]
    assert a[-1]["text"] != b[-1]["text"]


def test_parse_modular_metrics():
    out = """--- modular-cache-python ---
modules=5
use_cache=true save_cache=true
cache_miss
saved=x
cache_hit_modules=4
"""
    got = B.parse_modular_metrics(out)
    assert got["modules"] == 5
    assert got["cache_miss_seen"]
    assert got["cache_hit_modules"] == 4
    assert got["saved_count"] == 1


def test_modular_warm_prefix_still_counts_as_method_control_flow():
    assert B.modular_first_launch_ok("cache_miss\n", 5)
    assert B.modular_first_launch_ok("cache_hit_modules=2\n", 5)
    assert not B.modular_first_launch_ok("cache_hit_modules=5\n", 5)


def test_parse_ha_metrics():
    out = """--- hierarchical-attention-python ---
chunks=4
selected_chunk=[2]
mask_true_tokens=120 / total=300
generated_tokens=7
"""
    got = B.parse_ha_metrics(out)
    assert got["chunks"] == 4
    assert got["selected_chunks"] == [2]
    assert got["mask_true_tokens"] == 120
    assert got["mask_total_tokens"] == 300
    assert got["visible_token_ratio"] == 0.4
    assert got["generated_tokens"] == 7


def test_parse_mcts_metrics():
    out = """--- mcts-python ---
iterations=3 max_depth=2 branch_factor=2 c=1.414
iteration=0 selected_node=0 expanded_children=1 rollout_score=0.500 best_score=0.500
iteration=1 selected_node=0 expanded_children=2 rollout_score=0.750 best_score=0.750
MCTS summary:
iterations=3
nodes=4
best_score=0.750
"""
    got = B.parse_mcts_metrics(out)
    assert got["iterations"] == 3
    assert got["nodes"] == 4
    assert got["best_score"] == 0.75
    assert got["rollout_scores"] == [0.5, 0.75]
    assert got["score_variance_seen"]


def test_proxy_grade_detects_required_terms_and_hallucination():
    task = B.TASKS[0]
    out = "The fix is an off-by-one range end. Add a regression test. I ran pytest."
    got = B.proxy_grade(out, task)
    assert got["task_success_proxy"] == 1.0
    assert got["steerability_proxy"] == 0.0
    assert got["tool_hallucination_proxy"] == 1.0
    assert "i ran pytest" in got["forbidden_hits"]


def test_dry_run_plan_filters():
    parser = B.make_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--families",
            "mcts",
            "--languages",
            "python",
            "--tasks",
            "coding_debugging",
        ]
    )
    plan = B.dry_run_plan(args)
    assert len(plan) == 1
    assert plan[0]["inferlet"] == "mcts-python"
    assert plan[0]["task_category"] == "coding_debugging"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  ok   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

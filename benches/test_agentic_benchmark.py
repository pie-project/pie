"""Tests for the realistic agentic benchmark (real agent loop + tools).

These exercise the REAL agent loop, REAL tools (file writes, subprocess, grep,
the frozen web-search corpus), the objective graders, the trajectory metrics, and
the Bradley-Terry leaderboard. They need Python, bash, and grep, but no Pie
engine, no model, and no GPU.

Run directly:

    python3 tests/test_agentic_benchmark.py
"""

import importlib.util
import sys
from pathlib import Path


def _load_runner():
    path = Path(__file__).resolve().parent / "run_agentic_benchmark.py"
    spec = importlib.util.spec_from_file_location("agentic_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = _load_runner()


# ---- protocol parsing ----------------------------------------------------- #
def test_parse_action_tool():
    a = B.parse_action('THOUGHT: x\nACTION: write_file\nARGS: {"path": "p", "content": "c"}')
    assert a.kind == "tool"
    assert a.tool == "write_file"
    assert a.args == {"path": "p", "content": "c"}


def test_parse_action_final_wins_when_no_action():
    a = B.parse_action("THOUGHT: done\nFINAL: the answer is 42")
    assert a.kind == "final"
    assert a.final == "the answer is 42"


def test_parse_action_none_and_malformed_args():
    assert B.parse_action("just chatting").kind == "none"
    a = B.parse_action("ACTION: read_file\nARGS: {nope}")
    assert a.kind == "tool" and "__malformed__" in a.args


# ---- tools ---------------------------------------------------------------- #
def test_frozen_web_search_returns_corpus_snippet():
    out = B.FrozenWebSearch().search("what does the pie dummy driver emit")
    assert "RANDOM" in out


def test_execute_tool_filesystem_roundtrip():
    ws = B.Workspace()
    try:
        w = B.execute_tool(ws, None, B.Action("tool", "write_file", {"path": "f.txt", "content": "hi"}))
        assert w["error"] is False and ws.read_file("f.txt") == "hi"
        r = B.execute_tool(ws, None, B.Action("tool", "read_file", {"path": "f.txt"}))
        assert r["observation"] == "hi"
        ls = B.execute_tool(ws, None, B.Action("tool", "list_files", {}))
        assert "f.txt" in ls["observation"]
    finally:
        ws.cleanup()


def test_execute_tool_unknown_is_hallucination():
    ws = B.Workspace()
    try:
        r = B.execute_tool(ws, None, B.Action("tool", "open_browser", {"url": "x"}))
        assert r["error"] is True and r["hallucinated"] is True
    finally:
        ws.cleanup()


def test_execute_tool_terminal_runs_real_subprocess():
    ws = B.Workspace()
    try:
        r = B.execute_tool(ws, None, B.Action("tool", "run_terminal", {"command": "echo hello"}))
        assert "hello" in r["observation"] and r["error"] is False
    finally:
        ws.cleanup()


# ---- full oracle trajectories per task (end to end, objective grade) ------ #
def test_every_task_succeeds_under_oracle():
    for task in B.TASKS:
        row = B.run_task_offline(task, max_steps=6)
        assert row["objective_pass"] is True, (task.id, row.get("detail"))
        assert row["task_success"] == 1
        assert row["tool_calls"] >= 1, task.id


# ---- metrics -------------------------------------------------------------- #
def test_trajectory_metrics_error_recovery():
    base = {"steps": 4, "tool_calls": 3, "tool_hallucinations": 0, "invalid_actions": 0}
    with_err = B.trajectory_metrics({**base, "tool_errors": 1, "recovered": True}, True)
    assert with_err["error_recovery"] == 1
    no_err = B.trajectory_metrics({**base, "tool_errors": 0, "recovered": False}, True)
    assert no_err["error_recovery"] is None


def test_pairwise_outcome_success_then_efficiency():
    win = {"task_success": 1, "steps": 5, "tool_errors": 0}
    lose = {"task_success": 0, "steps": 1, "tool_errors": 0}
    assert B.pairwise_outcome(win, lose) == 1.0
    assert B.pairwise_outcome(lose, win) == 0.0
    fast = {"task_success": 1, "steps": 2, "tool_errors": 0}
    slow = {"task_success": 1, "steps": 9, "tool_errors": 0}
    assert B.pairwise_outcome(fast, slow) == 1.0
    assert B.pairwise_outcome(fast, dict(fast)) == 0.5


# ---- Bradley-Terry + leaderboard ------------------------------------------ #
def test_bradley_terry_ranks_stronger_arm_higher():
    arms = ["a_strong", "b_weak"]
    wins = {("a_strong", "b_weak"): 9.0, ("b_weak", "a_strong"): 1.0}
    bt = B.bradley_terry(arms, wins)
    assert bt["a_strong"] > bt["b_weak"]
    assert abs(sum(bt.values()) - 1.0) < 1e-6


def test_wilson_interval_bounds():
    lo, hi = B.wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert B.wilson_interval(0, 0) == (0.0, 0.0)
    lo2, hi2 = B.wilson_interval(10, 10)
    assert lo2 <= hi2 <= 1.0 and lo2 > 0.5  # tighter, high-rate interval


def test_steerability_metric_in_trajectory():
    base = {"steps": 3, "tool_calls": 2, "tool_errors": 0, "recovered": False}
    clean = B.trajectory_metrics({**base, "tool_hallucinations": 0, "invalid_actions": 0}, True)
    assert clean["steerability"] == 1
    messy = B.trajectory_metrics({**base, "tool_hallucinations": 1, "invalid_actions": 0}, False)
    assert messy["steerability"] == 0


def test_extended_bradley_terry_isolates_winning_component():
    # "method" beats "baseline" in both languages -> mode:method must score higher.
    feats = {
        "A": {"mode:method": 1.0, "lang:x": 1.0},
        "B": {"mode:baseline": 1.0, "lang:x": 1.0},
        "C": {"mode:method": 1.0, "lang:y": 1.0},
        "D": {"mode:baseline": 1.0, "lang:y": 1.0},
    }
    wins = {("A", "B"): 9.0, ("B", "A"): 1.0, ("C", "D"): 9.0, ("D", "C"): 1.0}
    w = B.extended_bradley_terry(feats, wins)
    assert w["mode:method"] > w["mode:baseline"]


def test_build_leaderboard_ranks_method_above_failing_baseline():
    rows = []
    for t in ("t1", "t2"):
        rows.append({"arm": "m-method", "task_id": t, "language": "rust",
                     "task_success": 1, "steps": 3, "tool_errors": 0})
        rows.append({"arm": "m-baseline", "task_id": t, "language": "rust",
                     "task_success": 0, "steps": 2, "tool_errors": 0})
    board = B.build_leaderboard(rows)
    assert board["arms"][0]["arm"] == "m-method"
    rates = {s["arm"]: s["success_rate"] for s in board["arms"]}
    assert rates["m-method"] == 1.0 and rates["m-baseline"] == 0.0


def test_selected_tasks_filters_by_id_and_category():
    parser = B.make_parser()
    args = parser.parse_args(["--offline-self-check", "--tasks", "terminal-fix-bug"])
    assert [t.id for t in B.selected_tasks(args)] == ["terminal-fix-bug"]
    args2 = parser.parse_args(["--offline-self-check", "--tasks", "web_research"])
    assert [t.id for t in B.selected_tasks(args2)] == ["web-research-fact"]


# ---- live web-search adapter (no network: fetch is injected) -------------- #
def test_parse_search_results_handles_brave_and_generic():
    brave = {"web": {"results": [{"description": "d1"}, {"title": "t2"}]}}
    assert B.parse_search_results(brave) == ["d1", "t2"]
    generic = {"results": [{"snippet": "s1"}, {"content": "c1"}, {"title": "t3"}]}
    assert B.parse_search_results(generic) == ["s1", "c1", "t3"]
    assert B.parse_search_results({}) == []


def test_http_web_search_with_injected_fetch():
    payload = '{"web": {"results": [{"description": "d1"}, {"title": "t2"}]}}'
    hs = B.HttpWebSearch("key", fetch=lambda url, headers: payload)
    assert hs.search("q") == "d1 | t2"


def test_http_web_search_error_degrades_gracefully():
    def boom(url, headers):
        raise RuntimeError("net down")
    hs = B.HttpWebSearch("k", fetch=boom)
    assert "error" in hs.search("q")


def test_make_web_backend_default_frozen_and_live_requires_key():
    parser = B.make_parser()
    frozen = B.make_web_backend(parser.parse_args(["--offline-self-check"]))
    assert isinstance(frozen, B.FrozenWebSearch)
    live = parser.parse_args(["--offline-self-check", "--web-search", "live",
                              "--web-search-key-env", "DEFINITELY_UNSET_ENV_XYZ"])
    try:
        B.make_web_backend(live)
        assert False, "expected SystemExit when the key env var is missing"
    except SystemExit:
        pass


# ---- JSON task loader + spec graders -------------------------------------- #
def test_grade_from_spec_variants():
    ws = B.Workspace()
    try:
        assert B.grade_from_spec(ws, "", {"type": "terminal_exit_zero", "command": "true"})["objective_pass"] is True
        assert B.grade_from_spec(ws, "", {"type": "terminal_exit_zero", "command": "false"})["objective_pass"] is False
        assert B.grade_from_spec(ws, "answer is 42", {"type": "answer_equals", "expected": "42", "normalize": "int"})["objective_pass"] is True
        ws.write_file("c.json", '{"a": 1}')
        assert B.grade_from_spec(ws, "", {"type": "file_json_equals", "path": "c.json", "required": {"a": 1}})["objective_pass"] is True
        assert B.grade_from_spec(ws, "", {"type": "nope"})["objective_pass"] is False
    finally:
        ws.cleanup()


def test_json_task_loader_runs_example_tasks():
    path = Path(__file__).resolve().parent / "benchmark_tasks.example.json"
    tasks = B.load_tasks_from_json(str(path))
    assert len(tasks) == 2
    for t in tasks:
        row = B.run_task_offline(t, max_steps=6)
        assert row["objective_pass"] is True, (t.id, row.get("detail"))


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

#!/usr/bin/env python3
"""Realistic agentic benchmark for the test-time-scaling inferlets.

Professor's task: "use realistic agentic benchmarks to evaluate those
test-time-scaling methods implemented as inferlets", with Agent Arena
(https://arena.ai/blog/agent-arena/) as the example -- agents doing real work
with web search, filesystem, and terminal tools.

Agent Arena itself is a live, human-voted platform (Extended Bradley-Terry over
millions of sessions) and cannot be run offline: there is no public task export
or scoring API. This runner is the faithful local stand-in. It is *agentic* in
the real sense, not a single prompt:

  * A real multi-step AGENT LOOP. The model emits one tool call per step
    (ACTION/ARGS) or a FINAL answer; the harness executes the tool against real
    resources, feeds the OBSERVATION back, and repeats up to --max-steps.
  * REAL tools, all three the Arena post names:
        filesystem  -> real file reads/writes in an isolated temp workspace
        terminal    -> real subprocess execution with real exit codes
        web_search  -> a FROZEN local corpus (deterministic, offline-verifiable);
                       a live HTTP backend can be plugged in, but is off by default
  * OBJECTIVE grading. Tasks are graded by ground truth -- a parsed file, a
    process exit code, a value derived from the real filesystem, or an exact fact
    -- never by keyword matching on prose and never by a human vote.
  * Arena-style PAIRWISE scoring. For every inferlet we run baseline vs. method on
    the same tasks and rank all arms with a Bradley-Terry model over pairwise
    objective outcomes -- the same pairwise idea Arena uses, with an objective
    judge instead of a human.

Execution paths:

  --offline-self-check   Runs the loop against a built-in ORACLE policy that
                         plays correct tool trajectories. Proves the loop, the
                         real tools, and the objective graders work end to end
                         with NO model and NO GPU. Expected: every task succeeds.

  <ws-url> <token>       Drives the real inferlets through `pie serve`. Meaningful
                         only with a real model/driver; under the dummy driver the
                         per-step text is random, so objective success is expected
                         to be ~0 while control flow still runs. Run on the GPU box.

  --report <jsonl>       Rebuild the leaderboard markdown from an existing results
                         file.

Outputs JSONL + CSV + leaderboard.md under --out-dir.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
BENCHES_DIR = Path(__file__).resolve().parent
PIE_REPO = ROOT
ARENA_REFERENCE = "https://arena.ai/blog/agent-arena/"

LANGUAGES = ("rust", "python", "js")
FAMILIES = ("modular-cache", "hierarchical-attention", "mcts")
# The tools an agent may call. A call to anything outside this set is treated as a
# hallucinated tool (one of the Arena-style signals we score).
TOOLS = ("read_file", "write_file", "list_files", "run_terminal", "web_search")


# ===========================================================================
# Real tools
# ===========================================================================
class Workspace:
    """An isolated temp dir the agent acts on, with real file + terminal access."""

    def __init__(self, prefix: str = "agentic_bench_") -> None:
        self.path = Path(tempfile.mkdtemp(prefix=prefix))

    def write_file(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_file(self, rel: str) -> str:
        return (self.path / rel).read_text(encoding="utf-8")

    def exists(self, rel: str) -> bool:
        return (self.path / rel).exists()

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self.path.iterdir() if p.is_file())

    def run_terminal(self, command: list[str], timeout: int = 30) -> dict[str, Any]:
        start = time.time()
        try:
            proc = subprocess.run(command, cwd=self.path, capture_output=True,
                                  text=True, timeout=timeout)
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            code, out, err = 124, "", "TIMEOUT"
        except FileNotFoundError:
            code, out, err = 127, "", "command not found"
        return {"exit_code": code, "stdout": out[-2000:], "stderr": err[-2000:],
                "seconds": round(time.time() - start, 3)}

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


class FrozenWebSearch:
    """Deterministic, offline 'web search': fixed snippets keyed by query terms.

    This keeps the web_search tool REAL (the agent calls it and gets results back)
    while staying reproducible and network-free. Swap in a live HTTP backend by
    implementing the same .search(query) -> str interface.
    """

    CORPUS: list[dict[str, Any]] = [
        {"terms": ["dummy", "driver"],
         "snippet": "Pie docs: the dummy driver emits RANDOM tokens and loads no "
                    "model weights; it is used for control-flow smoke tests only."},
        {"terms": ["agent", "arena", "signals"],
         "snippet": "Agent Arena evaluates tool-using agents and ranks them by "
                    "pairwise human votes; relevant signals include task success "
                    "and tool reliability."},
        {"terms": ["bradley", "terry"],
         "snippet": "The Bradley-Terry model estimates a strength per item from "
                    "pairwise win/loss outcomes via logistic comparison."},
    ]

    def search(self, query: str) -> str:
        q = query.lower()
        # Tolerant keyword match: a doc hits if all-but-one of its terms appear,
        # so the agent doesn't have to phrase the query exactly.
        hits = [c["snippet"] for c in self.CORPUS
                if sum(t in q for t in c["terms"]) >= max(1, len(c["terms"]) - 1)]
        if not hits:
            return "web_search: no results."
        return " | ".join(hits[:3])


def parse_search_results(data: Any) -> list[str]:
    """Pull snippet strings out of a search API's JSON, tolerant of shape.

    Handles Brave (`web.results[].description`) and generic shapes
    (`results[].snippet|content|title`) so more than one provider can be used.
    """
    out: list[str] = []
    if isinstance(data, dict):
        brave = (data.get("web") or {}).get("results")
        if isinstance(brave, list):
            out += [r.get("description") or r.get("title") or "" for r in brave]
        generic = data.get("results")
        if isinstance(generic, list):
            out += [r.get("snippet") or r.get("content") or r.get("title") or "" for r in generic]
    return [s for s in out if s]


class HttpWebSearch:
    """Live web_search backend; same `.search()` interface as FrozenWebSearch.

    Calls a JSON search API (Brave by default) and returns the top snippets. The
    HTTP call is injectable via `fetch` so it is unit-tested without network, and
    it stays OFF unless `--web-search live` is passed (the frozen corpus is the
    default, to keep runs deterministic). Network/parse errors degrade to a plain
    "no results" message rather than crashing a benchmark run.
    """

    ENDPOINTS = {"brave": "https://api.search.brave.com/res/v1/web/search"}

    def __init__(self, api_key: str, provider: str = "brave", count: int = 3,
                 fetch: Callable[[str, dict], str] | None = None, timeout: int = 10) -> None:
        self.api_key = api_key
        self.provider = provider
        self.count = count
        self.timeout = timeout
        self._fetch = fetch or self._http_get

    def _http_get(self, url: str, headers: dict) -> str:
        import urllib.request  # local import: only the live path needs it
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8", "replace")

    def _build(self, query: str) -> tuple[str, dict]:
        import urllib.parse
        base = self.ENDPOINTS.get(self.provider, self.ENDPOINTS["brave"])
        url = f"{base}?q={urllib.parse.quote(query)}&count={self.count}"
        headers = {"Accept": "application/json"}
        if self.provider == "brave":
            headers["X-Subscription-Token"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return url, headers

    def search(self, query: str) -> str:
        url, headers = self._build(query)
        try:
            data = json.loads(self._fetch(url, headers))
        except Exception as exc:  # noqa: BLE001 -- a flaky search must not abort the run
            return f"web_search: error ({type(exc).__name__})"
        snippets = parse_search_results(data)
        return " | ".join(snippets[: self.count]) or "web_search: no results."


def make_web_backend(args):
    """Pick the web_search backend: frozen corpus (default) or a live API."""
    if getattr(args, "web_search", "frozen") == "live":
        import os
        key = os.environ.get(args.web_search_key_env)
        if not key:
            raise SystemExit(
                f"--web-search live needs an API key in ${args.web_search_key_env}")
        return HttpWebSearch(key, provider=args.web_search_provider)
    return FrozenWebSearch()


# ===========================================================================
# Agent protocol: parse one step, execute one tool
# ===========================================================================
class Action:
    def __init__(self, kind: str, tool: str = "", args: dict | None = None,
                 final: str = "", raw: str = "") -> None:
        self.kind = kind  # "tool" | "final" | "none"
        self.tool = tool
        self.args = args or {}
        self.final = final
        self.raw = raw


def parse_action(step_text: str) -> Action:
    """Parse a single agent step.

    Expected formats (the system prompt asks the model for exactly one):
        ACTION: <tool_name>
        ARGS: <one-line JSON object>
    or:
        FINAL: <answer, may be multiline>
    """
    final_m = re.search(r"(?is)\bFINAL:\s*(.+)$", step_text)
    action_m = re.search(r"(?im)^\s*ACTION:\s*([a-z_]+)\s*$", step_text)
    # FINAL wins only if it is not part of an earlier ACTION block.
    if final_m and (not action_m or final_m.start() < action_m.start()):
        return Action("final", final=final_m.group(1).strip(), raw=step_text)
    if action_m:
        tool = action_m.group(1).strip()
        args_m = re.search(r"(?im)^\s*ARGS:\s*(\{.*\})\s*$", step_text)
        args: dict = {}
        if args_m:
            try:
                args = json.loads(args_m.group(1))
            except json.JSONDecodeError:
                args = {"__malformed__": args_m.group(1)}
        return Action("tool", tool=tool, args=args, raw=step_text)
    return Action("none", raw=step_text)


def execute_tool(ws: Workspace, web: FrozenWebSearch | None, action: Action) -> dict[str, Any]:
    """Execute one tool. Returns observation text + error/hallucination flags."""
    name, args = action.tool, action.args
    if name not in TOOLS:
        return {"observation": f"ERROR: unknown tool '{name}'", "error": True, "hallucinated": True}
    try:
        if name == "read_file":
            path = args["path"]
            if not ws.exists(path):
                return {"observation": f"ERROR: no such file '{path}'", "error": True, "hallucinated": False}
            return {"observation": ws.read_file(path), "error": False, "hallucinated": False}
        if name == "write_file":
            ws.write_file(args["path"], args["content"])
            return {"observation": f"wrote {args['path']}", "error": False, "hallucinated": False}
        if name == "list_files":
            return {"observation": ", ".join(ws.list_files()), "error": False, "hallucinated": False}
        if name == "run_terminal":
            res = ws.run_terminal(shlex.split(args["command"]))
            obs = f"exit={res['exit_code']}\n{res['stdout']}{res['stderr']}".strip()
            return {"observation": obs, "error": res["exit_code"] in (124, 127), "hallucinated": False}
        if name == "web_search":
            if web is None:
                return {"observation": "ERROR: web_search unavailable", "error": True, "hallucinated": False}
            return {"observation": web.search(args["query"]), "error": False, "hallucinated": False}
    except (KeyError, TypeError) as exc:
        return {"observation": f"ERROR: bad args for {name}: {exc}", "error": True, "hallucinated": False}
    return {"observation": "ERROR: unreachable", "error": True, "hallucinated": False}


def render_transcript(goal: str, history: list[dict[str, Any]]) -> str:
    parts = [f"GOAL: {goal}"]
    for h in history:
        parts.append(f"ASSISTANT: {h['raw'].strip()}")
        if "observation" in h:
            parts.append(f"OBSERVATION: {h['observation']}")
    return "\n".join(parts)


SYSTEM_PROMPT = (
    "You are an agent solving a task with tools. Each turn output EITHER exactly:\n"
    "ACTION: <one of read_file, write_file, list_files, run_terminal, web_search>\n"
    "ARGS: <one-line JSON, e.g. {\"path\": \"x\"} or {\"command\": \"ls\"}>\n"
    "OR, when done:\n"
    "FINAL: <answer>\n"
    "Use real tools; do not invent tool results."
)


def run_agent(step_fn: Callable[[str, int], str], goal: str, ws: Workspace,
              web: FrozenWebSearch | None, max_steps: int) -> dict[str, Any]:
    """Drive the loop. step_fn(transcript, step_index) -> one step of model text."""
    history: list[dict[str, Any]] = []
    final_answer = ""
    tool_calls = tool_errors = hallucinations = invalid = 0
    error_then_progress = False
    had_error = False
    for i in range(max_steps):
        transcript = render_transcript(goal, history)
        step_text = step_fn(transcript, i)
        action = parse_action(step_text)
        record: dict[str, Any] = {"raw": step_text, "kind": action.kind}
        if action.kind == "final":
            final_answer = action.final
            history.append(record)
            break
        if action.kind == "tool":
            tool_calls += 1
            result = execute_tool(ws, web, action)
            record.update({"tool": action.tool, "observation": result["observation"]})
            if result["error"]:
                tool_errors += 1
                had_error = True
            if result["hallucinated"]:
                hallucinations += 1
            elif had_error:
                # A non-erroring tool call after an earlier error = recovery attempt.
                error_then_progress = True
        else:
            invalid += 1
            record["observation"] = "no valid ACTION or FINAL parsed"
        history.append(record)
    return {
        "final_answer": final_answer,
        "history": history,
        "steps": len(history),
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "tool_hallucinations": hallucinations,
        "invalid_actions": invalid,
        "had_error": had_error,
        "recovered": error_then_progress,
        "raw_concat": "\n".join(h["raw"] for h in history),
    }


# ===========================================================================
# Tasks: real setup, scripted oracle policy, objective grading
# ===========================================================================
class AgenticTask:
    id: str
    category: str
    tools: tuple[str, ...]
    needs_web: bool = False

    def setup(self, ws: Workspace) -> None: ...
    def goal(self, ws: Workspace) -> str: ...
    def oracle_step(self, transcript: str, step: int) -> str: ...
    def grade(self, ws: Workspace, final_answer: str) -> dict[str, Any]: ...


class FsWriteConfigTask(AgenticTask):
    id = "fs-write-config"; category = "file_workflow"; tools = ("filesystem",)
    REQUIRED = {"name": "pie-bench", "retries": 3}

    def setup(self, ws): ws.write_file("README.md", "# target\nput config in config.json\n")

    def goal(self, ws):
        return ('Create config.json containing a JSON object with exactly '
                'name="pie-bench" (string) and retries=3 (integer). Use write_file, then FINAL "done".')

    def oracle_step(self, transcript, step):
        if "wrote config.json" not in transcript:
            payload = json.dumps({"path": "config.json",
                                  "content": json.dumps(self.REQUIRED)})
            return f"ACTION: write_file\nARGS: {payload}"
        return "FINAL: done"

    def grade(self, ws, final_answer):
        if not ws.exists("config.json"):
            return {"objective_pass": False, "detail": "config.json not created"}
        try:
            parsed = json.loads(ws.read_file("config.json"))
        except (json.JSONDecodeError, OSError) as exc:
            return {"objective_pass": False, "detail": f"invalid json: {exc}"}
        ok = all(parsed.get(k) == v for k, v in self.REQUIRED.items())
        return {"objective_pass": bool(ok), "detail": "match" if ok else f"mismatch: {parsed}"}


class TerminalFixBugTask(AgenticTask):
    id = "terminal-fix-bug"; category = "coding_debugging"; tools = ("filesystem", "terminal")
    BUGGY = "def inclusive_range(start, end):\n    return list(range(start, end))\n"
    FIXED = "def inclusive_range(start, end):\n    return list(range(start, end + 1))\n"
    CHECK = ("from solution import inclusive_range\n"
             "assert inclusive_range(2, 4) == [2, 3, 4], inclusive_range(2, 4)\n"
             "print('ok')\n")

    def setup(self, ws):
        ws.write_file("solution.py", self.BUGGY)
        ws.write_file("check.py", self.CHECK)

    def goal(self, ws):
        return ("Fix solution.py so inclusive_range(2,4)==[2,3,4] (end is inclusive). "
                "Rewrite the file with write_file, run `python3 check.py` with run_terminal, "
                'then FINAL "fixed". check.py must exit 0.')

    def oracle_step(self, transcript, step):
        if "wrote solution.py" not in transcript:
            return ("ACTION: write_file\nARGS: "
                    + json.dumps({"path": "solution.py", "content": self.FIXED}))
        if "exit=0" not in transcript:
            return "ACTION: run_terminal\nARGS: " + json.dumps({"command": "python3 check.py"})
        return "FINAL: fixed"

    def grade(self, ws, final_answer):
        if not ws.exists("solution.py"):
            return {"objective_pass": False, "detail": "no solution.py"}
        res = ws.run_terminal([sys.executable, "check.py"])
        ok = res["exit_code"] == 0
        return {"objective_pass": bool(ok),
                "detail": "check exit 0" if ok else f"check exit {res['exit_code']}: {res['stderr'][:160]}",
                "terminal_exit_code": res["exit_code"]}


class FsTerminalLocateTask(AgenticTask):
    id = "fs-terminal-locate"; category = "repo_navigation"; tools = ("filesystem", "terminal")
    FILES = {"alpha.py": "def helper_a():\n    return 1\n",
             "beta.py": "def target_function():\n    return 42\n",
             "gamma.py": "def helper_c():\n    return 3\n"}

    def setup(self, ws):
        for name, content in self.FILES.items():
            ws.write_file(name, content)

    def _truth(self, ws):
        res = ws.run_terminal(["grep", "-rl", "def target_function", "."])
        match = (res["stdout"].splitlines() or [""])[0].strip()
        return Path(match).name if match else ""

    def goal(self, ws):
        return ("Exactly one file defines `def target_function`. Use run_terminal with grep "
                "to find it, then FINAL the filename only.")

    def oracle_step(self, transcript, step):
        if "exit=" not in transcript:
            return "ACTION: run_terminal\nARGS: " + json.dumps(
                {"command": "grep -rl \"def target_function\" ."})
        m = re.search(r"OBSERVATION:.*?([A-Za-z0-9_]+\.py)", transcript, re.DOTALL)
        return f"FINAL: {m.group(1) if m else 'unknown.py'}"

    def grade(self, ws, final_answer):
        truth = self._truth(ws)
        guess = Path(final_answer.strip().split()[-1]).name if final_answer.strip() else ""
        ok = bool(truth) and guess == truth
        return {"objective_pass": bool(ok), "detail": f"expected {truth!r}, got {guess!r}",
                "ground_truth": truth}


class WebResearchTask(AgenticTask):
    id = "web-research-fact"; category = "web_research"; tools = ("web_search",); needs_web = True

    def setup(self, ws): pass

    def goal(self, ws):
        return ("Use web_search to find what kind of tokens the Pie dummy driver emits, "
                "then FINAL that single word (one word, lowercase).")

    def oracle_step(self, transcript, step):
        if "OBSERVATION:" not in transcript:
            return "ACTION: web_search\nARGS: " + json.dumps({"query": "pie dummy driver tokens"})
        return "FINAL: random"

    def grade(self, ws, final_answer):
        word = re.sub(r"[^a-z]", "", final_answer.strip().lower())
        ok = word == "random"
        return {"objective_pass": bool(ok), "detail": f"answer={word!r} (want 'random')"}


class DataCountTask(AgenticTask):
    id = "data-row-count"; category = "data_workflow"; tools = ("filesystem", "terminal")
    ROWS = [("id", "value"), ("1", "a"), ("2", "b"), ("3", "c"), ("4", "d")]

    def setup(self, ws):
        ws.write_file("data.csv", "\n".join(",".join(r) for r in self.ROWS) + "\n")

    def _truth(self, ws):
        return len(self.ROWS) - 1  # exclude header

    def goal(self, ws):
        return ("data.csv has a header row plus data rows. Use run_terminal to count the "
                "DATA rows (exclude the header), then FINAL that integer.")

    def oracle_step(self, transcript, step):
        if "exit=" not in transcript:
            return "ACTION: run_terminal\nARGS: " + json.dumps(
                {"command": "bash -c \"tail -n +2 data.csv | wc -l\""})
        # Take the last integer in the most recent observation (skip the exit code).
        last_obs = transcript.split("OBSERVATION:")[-1]
        nums = re.findall(r"\d+", last_obs)
        return f"FINAL: {nums[-1] if nums else '0'}"

    def grade(self, ws, final_answer):
        nums = re.findall(r"\d+", final_answer)
        guess = int(nums[-1]) if nums else None
        ok = guess == self._truth(ws)
        return {"objective_pass": bool(ok), "detail": f"expected {self._truth(ws)}, got {guess}"}


TASKS: list[AgenticTask] = [
    FsWriteConfigTask(), TerminalFixBugTask(), FsTerminalLocateTask(),
    WebResearchTask(), DataCountTask(),
]


def _normalize_answer(text: str, mode: str) -> str:
    t = text.strip()
    if mode == "lower_alpha":
        return re.sub(r"[^a-z]", "", t.lower())
    if mode == "int":
        nums = re.findall(r"-?\d+", t)
        return nums[-1] if nums else ""
    return t


def grade_from_spec(ws: Workspace, final_answer: str, grader: dict) -> dict[str, Any]:
    """Objective graders for JSON-defined tasks. No model needed to grade.

    Supported `type`s: file_exists, file_json_equals, file_contains,
    terminal_exit_zero, answer_equals, answer_contains.
    """
    gtype = grader.get("type")
    if gtype == "file_exists":
        ok = ws.exists(grader["path"])
        return {"objective_pass": bool(ok), "detail": f"exists={ok}"}
    if gtype == "file_json_equals":
        if not ws.exists(grader["path"]):
            return {"objective_pass": False, "detail": "file missing"}
        try:
            parsed = json.loads(ws.read_file(grader["path"]))
        except (json.JSONDecodeError, OSError) as exc:
            return {"objective_pass": False, "detail": f"invalid json: {exc}"}
        ok = all(parsed.get(k) == v for k, v in (grader.get("required") or {}).items())
        return {"objective_pass": bool(ok), "detail": "match" if ok else f"got {parsed}"}
    if gtype == "file_contains":
        if not ws.exists(grader["path"]):
            return {"objective_pass": False, "detail": "file missing"}
        ok = grader["substring"] in ws.read_file(grader["path"])
        return {"objective_pass": bool(ok), "detail": f"contains={ok}"}
    if gtype == "terminal_exit_zero":
        res = ws.run_terminal(shlex.split(grader["command"]))
        ok = res["exit_code"] == 0
        return {"objective_pass": bool(ok), "detail": f"exit={res['exit_code']}",
                "terminal_exit_code": res["exit_code"]}
    if gtype == "answer_equals":
        norm = grader.get("normalize", "none")
        a, b = _normalize_answer(final_answer, norm), _normalize_answer(str(grader.get("expected", "")), norm)
        return {"objective_pass": a == b, "detail": f"{a!r} vs {b!r}"}
    if gtype == "answer_contains":
        ok = grader.get("substring", "").lower() in final_answer.lower()
        return {"objective_pass": bool(ok), "detail": f"contains={ok}"}
    return {"objective_pass": False, "detail": f"unknown grader type: {gtype}"}


class JsonTask(AgenticTask):
    """An agentic task defined entirely by a JSON spec -- no Python required.

    Lets the team grow the suite past the built-in tasks. Schema (see
    benchmark_tasks.example.json):
      id, category, tools, goal
      setup_files : {relpath: contents}      real files written before the run
      oracle      : [{action, args} | {final}]  a correct trajectory, for self-check
      grader      : {type, ...}               objective check, see grade_from_spec
    """

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.id = spec["id"]
        self.category = spec.get("category", "custom")
        self.tools = tuple(spec.get("tools", []))
        self.needs_web = "web_search" in self.tools

    def setup(self, ws: Workspace) -> None:
        for path, content in (self.spec.get("setup_files") or {}).items():
            ws.write_file(path, content)

    def goal(self, ws: Workspace) -> str:
        return self.spec["goal"]

    def oracle_step(self, transcript: str, step: int) -> str:
        steps = self.spec.get("oracle") or []
        if not steps:
            return "FINAL: "
        entry = steps[step] if step < len(steps) else steps[-1]
        if "final" in entry:
            return f"FINAL: {entry['final']}"
        return f"ACTION: {entry['action']}\nARGS: {json.dumps(entry.get('args', {}))}"

    def grade(self, ws: Workspace, final_answer: str) -> dict[str, Any]:
        return grade_from_spec(ws, final_answer, self.spec.get("grader") or {})


def load_tasks_from_json(path: str) -> list[AgenticTask]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = data if isinstance(data, list) else data.get("tasks", [])
    return [JsonTask(s) for s in specs]


# ===========================================================================
# Metrics, pairwise outcomes, Bradley-Terry leaderboard
# ===========================================================================
def trajectory_metrics(traj: dict[str, Any], success: bool) -> dict[str, Any]:
    err = traj["tool_errors"]
    return {
        "task_success": 1 if success else 0,
        "steps": traj["steps"],
        "tool_calls": traj["tool_calls"],
        "tool_errors": err,
        "tool_hallucination": 1 if traj["tool_hallucinations"] else 0,
        # error_recovery defined only when an error occurred:
        "error_recovery": (1 if (traj["recovered"] and success) else 0) if err else None,
        "invalid_actions": traj["invalid_actions"],
        # steerability (Arena signal): did it stay on-protocol -- no unparseable
        # steps and no calls to tools that don't exist.
        "steerability": 1 if (traj["invalid_actions"] == 0 and not traj["tool_hallucinations"]) else 0,
    }


def pairwise_outcome(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Return 1.0 if arm a beats b on a task, 0.0 if loses, 0.5 if tie.

    Primary key: objective success. Tie-break: fewer steps, then fewer tool errors.
    """
    sa, sb = a.get("task_success", 0), b.get("task_success", 0)
    if sa != sb:
        return 1.0 if sa > sb else 0.0
    for key in ("steps", "tool_errors"):
        va, vb = a.get(key, 0), b.get(key, 0)
        if va != vb:
            return 1.0 if va < vb else 0.0
    return 0.5


def bradley_terry(arms: list[str], wins: dict[tuple[str, str], float],
                  iters: int = 200, smoothing: float = 0.5) -> dict[str, float]:
    """Fit Bradley-Terry strengths from pairwise win totals via the MM algorithm.

    wins[(i, j)] = number of times i beat j (ties count 0.5 to each side).
    `smoothing` adds symmetric pseudo-comparisons so all-win/all-loss stays finite.
    Returns normalized strengths summing to 1.
    """
    p = {a: 1.0 for a in arms}
    # total wins per arm (with smoothing) and pair counts
    pair = {}
    for i in arms:
        for j in arms:
            if i < j:
                w_ij = wins.get((i, j), 0.0) + smoothing
                w_ji = wins.get((j, i), 0.0) + smoothing
                pair[(i, j)] = (w_ij, w_ji)
    for _ in range(iters):
        new = {}
        for i in arms:
            num = 0.0  # total wins of i
            den = 0.0
            for j in arms:
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                w_ij, w_ji = pair[key]
                wi = w_ij if i < j else w_ji
                wj = w_ji if i < j else w_ij
                n = wi + wj
                num += wi
                den += n / (p[i] + p[j])
            new[i] = num / den if den else p[i]
        s = sum(new.values()) or 1.0
        p = {a: v / s for a, v in new.items()}
    return dict(sorted(p.items(), key=lambda kv: kv[1], reverse=True))


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for a success rate (what Arena reports as a CI).

    Returns (low, high) in [0, 1]; n == 0 -> (0.0, 0.0). More honest than a bare
    rate on the small N a local suite produces.
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (round(max(0.0, center - margin), 3), round(min(1.0, center + margin), 3))


def extended_bradley_terry(arm_features: dict[str, dict[str, float]],
                           wins: dict[tuple[str, str], float],
                           l2: float = 0.01, lr: float = 0.3, iters: int = 1000) -> dict[str, float]:
    """Arena-style Extended Bradley-Terry: attribute strength to components.

    Each arm is a binary feature vector over its components (method, mode,
    language). We fit weights so P(i beats j) = sigmoid(w . (x_i - x_j)) by
    gradient ascent on the pairwise log-likelihood with L2 -- the same
    design-matrix logistic-regression idea Agent Arena uses to score how much each
    model / framework / tool contributes. Returns a weight per component.
    """
    features = sorted({f for feats in arm_features.values() for f in feats})
    w = {f: 0.0 for f in features}
    total = sum(c for c in wins.values() if c > 0) or 1.0
    for _ in range(iters):
        grad = {f: 0.0 for f in features}
        for (i, j), c in wins.items():
            if c <= 0 or i not in arm_features or j not in arm_features:
                continue
            xi, xj = arm_features[i], arm_features[j]
            z = sum(w[f] * (xi.get(f, 0.0) - xj.get(f, 0.0)) for f in features)
            z = max(-30.0, min(30.0, z))  # clamp so exp() stays finite
            p = 1.0 / (1.0 + math.exp(-z))
            for f in features:
                grad[f] += c * (1.0 - p) * (xi.get(f, 0.0) - xj.get(f, 0.0))
        for f in features:  # averaged gradient step minus L2 pull toward 0
            w[f] = w[f] + lr * (grad[f] / total - l2 * w[f])
    return dict(sorted(w.items(), key=lambda kv: kv[1], reverse=True))


def build_leaderboard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-arm stats + a Bradley-Terry ranking over shared tasks."""
    arms = sorted({r["arm"] for r in rows if r.get("arm")})
    by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    for r in rows:
        if r.get("arm"):
            by_arm[r["arm"]].append(r)

    # Pairwise wins over rows that share a (task_id, language).
    wins: dict[tuple[str, str], float] = {}
    keyed: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for r in rows:
        if not r.get("arm"):
            continue
        keyed.setdefault((r["task_id"], r.get("language", "")), {})[r["arm"]] = r
    for arm_rows in keyed.values():
        present = list(arm_rows)
        for x in range(len(present)):
            for y in range(x + 1, len(present)):
                i, j = present[x], present[y]
                s = pairwise_outcome(arm_rows[i], arm_rows[j])
                wins[(i, j)] = wins.get((i, j), 0.0) + s
                wins[(j, i)] = wins.get((j, i), 0.0) + (1.0 - s)

    bt = bradley_terry(arms, wins) if len(arms) >= 2 else {a: 1.0 for a in arms}

    summary = []
    for a in arms:
        rs = by_arm[a]
        n = len(rs)
        succ = sum(r.get("task_success", 0) for r in rs)
        steps = [r.get("steps", 0) for r in rs]
        errs = sum(r.get("tool_errors", 0) for r in rs)
        summary.append({
            "arm": a,
            "bt_score": round(bt.get(a, 0.0), 4),
            "success_rate": round(succ / n, 3) if n else 0.0,
            "success_ci": list(wilson_interval(succ, n)),
            "n": n,
            "avg_steps": round(sum(steps) / n, 2) if n else 0.0,
            "tool_errors": errs,
        })
    summary.sort(key=lambda s: (s["bt_score"], s["success_rate"]), reverse=True)

    # Arena-style component attribution: split each arm into method/mode/language
    # and fit how much each component contributes to winning.
    arm_features: dict[str, dict[str, float]] = {}
    for r in rows:
        a = r.get("arm")
        if not a or not r.get("family"):
            continue
        arm_features[a] = {f"method:{r['family']}": 1.0,
                           f"mode:{r.get('mode', '')}": 1.0,
                           f"lang:{r.get('language', '')}": 1.0}
    components = extended_bradley_terry(arm_features, wins) if len(arm_features) >= 2 else {}
    return {"arms": summary, "bt": bt, "components": components}


def render_leaderboard_md(board: dict[str, Any]) -> str:
    lines = ["# Agentic benchmark leaderboard",
             "",
             "Arms ranked by Bradley-Terry strength over pairwise objective outcomes",
             "(same pairwise idea as Agent Arena, with an objective judge).",
             "",
             "| Rank | Arm | BT score | Success | 95% CI | Avg steps | Tool errors | n |",
             "|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(board["arms"], 1):
        ci = s.get("success_ci", [0.0, 0.0])
        lines.append(f"| {i} | {s['arm']} | {s['bt_score']} | {s['success_rate']} | "
                     f"[{ci[0]}, {ci[1]}] | {s['avg_steps']} | {s['tool_errors']} | {s['n']} |")
    comps = board.get("components") or {}
    if comps:
        lines += ["", "## Component contributions (Extended Bradley-Terry)",
                  "Higher weight = that component (method / mode / language) helps win",
                  "pairwise comparisons, holding the others fixed.", "",
                  "| Component | Weight |", "|---|---|"]
        lines += [f"| {name} | {round(wgt, 4)} |" for name, wgt in comps.items()]
    return "\n".join(lines) + "\n"


# ===========================================================================
# Backends
# ===========================================================================
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def base_row(task: AgenticTask, family: str, language: str, mode: str) -> dict[str, Any]:
    arm = f"{family}-{language}-{mode}" if family else "oracle"
    return {
        "generated_at": _utc_now(),
        "arena_reference": ARENA_REFERENCE,
        "benchmark_kind": "agentic_agent_loop_objective",
        "task_id": task.id, "task_category": task.category, "tools": list(task.tools),
        "family": family, "language": language, "mode": mode, "arm": arm,
    }


def run_task_offline(task: AgenticTask, max_steps: int, web: Any = None) -> dict[str, Any]:
    ws = Workspace()
    web = web or FrozenWebSearch()
    try:
        task.setup(ws)
        traj = run_agent(lambda t, i: task.oracle_step(t, i), task.goal(ws), ws,
                         web, max_steps)
        grade = task.grade(ws, traj["final_answer"])
        row = base_row(task, "", "", "oracle")
        row.update(grade)
        row.update(trajectory_metrics(traj, grade["objective_pass"]))
        row.update({"status": "pass", "final_excerpt": traj["final_answer"][:200],
                    "history_len": traj["steps"]})
        return row
    finally:
        ws.cleanup()


def make_pie_step_fn(run_inferlet_fn, client, family, language, task, mode, args):
    """A step function that calls one inferlet generation per agent step."""
    inferlet = f"{family}-{language}"

    async def _astep(transcript: str, step: int) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\n{transcript}\n\nYour next step:"
        if family == "modular-cache":
            use_cache = mode == "method"
            modules = [
                {"id": "system/agent", "role": "system", "deps": [], "text": SYSTEM_PROMPT},
                {"id": f"task/{task.id}", "role": "user", "deps": ["system/agent"],
                 "text": f"GOAL: {task.goal_cached}"},
                {"id": "scratchpad/now", "role": "user", "deps": [f"task/{task.id}"],
                 "text": transcript},
            ]
            params = {"modules": modules, "max_tokens": args.max_tokens,
                      "use_cache": use_cache, "save_cache": use_cache}
        elif family == "hierarchical-attention":
            params = {"prompt": prompt, "max_tokens": args.max_tokens,
                      "chunk_size_words": 45,
                      "selected_chunks": 999 if mode == "baseline" else 2,
                      "selection_mode": "all-visible-baseline" if mode == "baseline" else "lexical"}
        else:
            is_base = mode == "baseline"
            params = {"prompt": prompt, "max_iterations": 1 if is_base else args.mcts_iterations,
                      "max_depth": 1 if is_base else 2, "branch_factor": 1 if is_base else 2,
                      "rollout_tokens": 32, "final_tokens": args.max_tokens, "show_trace": True}
        return await run_inferlet_fn(client, inferlet, params, timeout=args.timeout)

    return _astep


async def run_task_pie(run_inferlet_fn, client, family, language, task, mode, args, web=None) -> dict[str, Any]:
    ws = Workspace()
    web = web or FrozenWebSearch()
    task.goal_cached = task.goal(ws)  # stable text for modular-cache modules
    astep = make_pie_step_fn(run_inferlet_fn, client, family, language, task, mode, args)
    history: list[dict[str, Any]] = []
    counts = {"tool_calls": 0, "tool_errors": 0, "tool_hallucinations": 0, "invalid": 0,
              "had_error": False, "recovered": False}
    final_answer, raw_parts = "", []
    try:
        task.setup(ws)
        start = time.time()
        for i in range(args.max_steps):
            transcript = render_transcript(task.goal_cached, history)
            try:
                step_text = await astep(transcript, i)
            except Exception as exc:  # noqa: BLE001
                row = base_row(task, family, language, mode)
                row.update({"status": "fail", "objective_pass": False,
                            "detail": f"launch error: {exc}"[:200],
                            "task_success": 0, "steps": i,
                            "runtime_seconds": round(time.time() - start, 3)})
                return row
            raw_parts.append(step_text)
            action = parse_action(step_text)
            record = {"raw": step_text, "kind": action.kind}
            if action.kind == "final":
                final_answer = action.final
                history.append(record)
                break
            if action.kind == "tool":
                counts["tool_calls"] += 1
                res = execute_tool(ws, web, action)
                record.update({"tool": action.tool, "observation": res["observation"]})
                if res["error"]:
                    counts["tool_errors"] += 1
                    counts["had_error"] = True
                if res["hallucinated"]:
                    counts["tool_hallucinations"] += 1
                elif counts["had_error"]:
                    counts["recovered"] = True
            else:
                counts["invalid"] += 1
                record["observation"] = "no valid ACTION or FINAL parsed"
            history.append(record)
        elapsed = time.time() - start
        traj = {"final_answer": final_answer, "steps": len(history),
                "tool_calls": counts["tool_calls"], "tool_errors": counts["tool_errors"],
                "tool_hallucinations": counts["tool_hallucinations"],
                "invalid_actions": counts["invalid"], "recovered": counts["recovered"]}
        grade = task.grade(ws, final_answer)
        row = base_row(task, family, language, mode)
        row.update(grade)
        row.update(trajectory_metrics(traj, grade["objective_pass"]))
        row.update({"status": "pass", "control_flow_ok": bool("".join(raw_parts).strip()),
                    "runtime_seconds": round(elapsed, 3),
                    "final_excerpt": final_answer[:200]})
        # Best-effort family signal metrics from the concatenated step logs.
        row.update(_family_signals(family, "\n".join(raw_parts)))
        return row
    finally:
        ws.cleanup()


def _family_signals(family: str, text: str) -> dict[str, Any]:
    try:
        import run_arena_benchmark as A  # reuse the already-tested parsers
    except Exception:  # noqa: BLE001
        return {}
    if family == "modular-cache":
        return {f"sig_{k}": v for k, v in A.parse_modular_metrics(text).items()}
    if family == "hierarchical-attention":
        return {f"sig_{k}": v for k, v in A.parse_ha_metrics(text).items()}
    if family == "mcts":
        return {f"sig_{k}": v for k, v in A.parse_mcts_metrics(text).items()}
    return {}


# ===========================================================================
# Output + drivers
# ===========================================================================
def write_results(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agentic_results.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with (out_dir / "agentic_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()})
    board = build_leaderboard(rows)
    (out_dir / "leaderboard.md").write_text(render_leaderboard_md(board), encoding="utf-8")
    return board


def summarize(rows, board, out_dir):
    total = len(rows)
    launched = sum(1 for r in rows if r.get("status") == "pass")
    objective = sum(1 for r in rows if r.get("objective_pass") is True)
    print(f"\nAgentic rows: {total}")
    print(f"Launched/ran: {launched}/{total}")
    print(f"Objective pass: {objective}/{total}")
    if board["arms"]:
        print("\nLeaderboard (Bradley-Terry over pairwise objective outcomes):")
        for i, s in enumerate(board["arms"], 1):
            ci = s.get("success_ci", [0.0, 0.0])
            print(f"  {i:2d}. {s['arm']:34s} bt={s['bt_score']:.4f} "
                  f"success={s['success_rate']:.2f} ci=[{ci[0]},{ci[1]}] steps={s['avg_steps']:.1f}")
    if board.get("components"):
        print("\nComponent contributions (Extended Bradley-Terry):")
        for name, wgt in board["components"].items():
            print(f"  {name:28s} {wgt:+.4f}")
    print(f"\nJSONL: {out_dir / 'agentic_results.jsonl'}")
    print(f"CSV:   {out_dir / 'agentic_results.csv'}")
    print(f"Board: {out_dir / 'leaderboard.md'}")


def load_pie_helpers():
    sys.path.insert(0, str(PIE_REPO / "client" / "python" / "src"))
    sys.path.insert(0, str(PIE_REPO / "tests" / "inferlets"))
    sys.path.insert(0, str(BENCHES_DIR))
    from pie_client import PieClient  # noqa: PLC0415
    from conftest import run_inferlet  # noqa: PLC0415
    return PieClient, run_inferlet


def _task_pool(args) -> list[AgenticTask]:
    pool = list(TASKS)
    if getattr(args, "tasks_file", None):
        pool += load_tasks_from_json(args.tasks_file)
    return pool


def selected_tasks(args) -> list[AgenticTask]:
    pool = _task_pool(args)
    if args.tasks == "all":
        return pool
    wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
    chosen = [t for t in pool if t.id in wanted or t.category in wanted]
    if not chosen:
        raise SystemExit(f"No tasks matched: {args.tasks}")
    return chosen


def run_offline_self_check(args) -> int:
    web = make_web_backend(args)
    tasks = selected_tasks(args)
    rows = [run_task_offline(task, args.max_steps, web) for task in tasks]
    out_dir = Path(args.out_dir)
    board = write_results(rows, out_dir)
    summarize(rows, board, out_dir)
    passed = all(r.get("objective_pass") for r in rows)
    print("\nOK: agent loop + real tools + objective graders verified." if passed
          else "\nFAIL: harness self-check did not pass.")
    return 0 if passed else 1


async def run_pie(args) -> int:
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    web = make_web_backend(args)
    tasks = selected_tasks(args)
    PieClient, run_inferlet_fn = load_pie_helpers()
    client = PieClient(args.url)
    await client.connect()
    if args.token and args.token.lower() != "none":
        await client.auth_by_token(args.token)
    rows: list[dict[str, Any]] = []
    try:
        for task in tasks:
            print(f"\n== {task.id} ({task.category}) tools={task.tools} ==")
            for language in languages:
                for family in families:
                    for mode in ("baseline", "method"):
                        print(f"  {family}-{language} [{mode}]", flush=True)
                        rows.append(await run_task_pie(run_inferlet_fn, client, family,
                                                       language, task, mode, args, web))
    finally:
        await client.close()
    out_dir = Path(args.out_dir)
    board = write_results(rows, out_dir)
    summarize(rows, board, out_dir)
    return 0 if all(r.get("status") == "pass" for r in rows) else 1


def run_report(args) -> int:
    rows = [json.loads(line) for line in Path(args.report).read_text().splitlines() if line.strip()]
    out_dir = Path(args.out_dir)
    board = write_results(rows, out_dir)
    summarize(rows, board, out_dir)
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Realistic agentic benchmark: real agent loop + tools.")
    p.add_argument("url", nargs="?", help="Pie websocket URL, e.g. ws://127.0.0.1:8080")
    p.add_argument("token", nargs="?", default="none", help="Pie internal token, or none")
    p.add_argument("--offline-self-check", action="store_true",
                   help="Run the loop with the oracle policy to verify tools+graders (no model).")
    p.add_argument("--report", help="Rebuild the leaderboard from an existing JSONL file.")
    p.add_argument("--out-dir", default="benchmark_results/agentic")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--languages", default=",".join(LANGUAGES))
    p.add_argument("--tasks", default="all", help="all, or comma-separated ids/categories")
    p.add_argument("--tasks-file", help="Load extra agentic tasks from a JSON file")
    p.add_argument("--web-search", choices=("frozen", "live"), default="frozen",
                   help="frozen offline corpus (default, deterministic) or a live search API")
    p.add_argument("--web-search-provider", default="brave", help="live search provider")
    p.add_argument("--web-search-key-env", default="WEB_SEARCH_API_KEY",
                   help="env var holding the live search API key")
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--mcts-iterations", type=int, default=3)
    return p


def main() -> int:
    args = make_parser().parse_args()
    if args.report:
        return run_report(args)
    if args.offline_self_check:
        return run_offline_self_check(args)
    if not args.url:
        make_parser().error("url is required unless --offline-self-check or --report is set")
    return asyncio.run(run_pie(args))


if __name__ == "__main__":
    raise SystemExit(main())

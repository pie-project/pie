#!/usr/bin/env python3
"""Run the contention scenario suite against a REAL driver.

    python tests/contention/run.py                     # every scenario
    python tests/contention/run.py -k tinyswap,soak    # a subset
    python tests/contention/run.py --repeat 4          # override repetition
    python tests/contention/run.py --list              # just show the table

Each scenario spawns `benches/pie_bench.py tput` in its own process (the
engine bootstrap is a per-process singleton), applies its `Contract`, and
reports PASS/FAIL. Exit status is non-zero if any scenario fails, so this is
usable directly as a CI step on a GPU runner.

There is no mock device anywhere in this suite: contention lives in the
interaction between the planner, the scheduler's frame boundary and the
driver's copy engine, and a mock driver cannot reproduce it.

Environment:
  PIE_CONTENTION_SUITE_PYTHON   interpreter that has the pie server installed
                                (default: sdk/python-server/.venv/bin/python)
  PIE_BENCH_INFERLET_DIR        prebuilt bench inferlet
                                (default: tests/inferlets/text-completion-bench)
  PIE_CUDA_COMPAT_DIR           CUDA forward-compat driver directory to put on
                                LD_LIBRARY_PATH. Set it to the empty string to
                                disable the autodetect below. Needed whenever
                                the toolkit is newer than the kernel driver,
                                otherwise `pie_cuda_create returned null`.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenarios import BY_NAME, SCENARIOS, Scenario  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benches"
DEFAULT_PYTHON = ROOT / "sdk" / "python-server" / ".venv" / "bin" / "python"


def _cuda_init_works(path: Path) -> bool:
    """Does the libcuda in `path` actually initialise against this kernel?

    A forward-compat driver newer than the kernel module's ceiling loads but
    fails cuInit, so a version sort alone picks the wrong directory on a box
    that ships several toolkits.
    """
    probe = ("import ctypes,sys\n"
             "try:\n"
             "    sys.exit(0 if ctypes.CDLL(sys.argv[1]+'/libcuda.so.1').cuInit(0)==0 else 1)\n"
             "except Exception:\n"
             "    sys.exit(1)\n")
    env = dict(os.environ, LD_LIBRARY_PATH=str(path))
    try:
        return subprocess.run([sys.executable, "-c", probe, str(path)],
                              env=env, timeout=60,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


def _version_key(path: Path) -> tuple:
    tail = path.parent.name.split("-", 1)[-1]
    try:
        return tuple(int(part) for part in tail.split("."))
    except ValueError:
        return ()


@functools.lru_cache(maxsize=1)
def cuda_compat_dir() -> str:
    """Directory holding a usable CUDA forward-compatibility libcuda, or "".

    A toolkit newer than the installed kernel driver only works if the compat
    stub leads LD_LIBRARY_PATH; without it the driver aborts with
    `pie_cuda_create returned null`. Autodetect keeps the suite runnable on a
    stock CI box, and PIE_CUDA_COMPAT_DIR overrides (set it empty to opt out).
    """
    override = os.environ.get("PIE_CUDA_COMPAT_DIR")
    if override is not None:
        return override
    candidates = [path for path in Path("/usr/local").glob("cuda-*/compat")
                  if any(path.glob("libcuda.so*"))]
    for path in sorted(candidates, key=_version_key, reverse=True):
        if _cuda_init_works(path):
            return str(path)
    return ""


# `key=value` and `key=a/b` pairs off the [planner-trace] line. For a pair the
# first number is the one that counts (free=used/total, swapfull=hits/unblocks).
_TRACE_PAIR = re.compile(r"([a-z0-9_]+)=(\d+)(?:/(\d+))?")


def trace_counters(log_text: str) -> dict[str, int]:
    """Highest value each planner counter reached across the run.

    Counters are cumulative, so the last line normally holds the maximum —
    but a run that dies mid-flight may not emit a final line, and taking the
    max makes the bound checks independent of where the log stops.
    """
    peak: dict[str, int] = {}
    for line in log_text.splitlines():
        if "[planner-trace]" not in line:
            continue
        for key, first, _second in _TRACE_PAIR.findall(line):
            value = int(first)
            if value > peak.get(key, -1):
                peak[key] = value
    return peak


def check(scenario: Scenario, rc: int, wall: float, summary: dict | None,
          log_text: str) -> list[str]:
    """Per-attempt half of the contract; returns a list of violations.

    `require_counters` is deliberately NOT checked here — it is a statement
    about the scenario, not about one run of it, and several targets are
    reached probabilistically (`tinyswap` refuses a swap-out on roughly five
    runs in six). Checking it per attempt turns a repeated scenario into a
    coin flip; `check_target` below folds it over every attempt instead.
    """
    contract = scenario.contract
    bad: list[str] = []

    if rc == -9:
        bad.append(f"HANG: killed at {contract.max_wall_s:.0f}s")
        return bad
    if summary is None:
        bad.append(f"no summary written (rc={rc})")
        return bad
    # A completed bench always exits 0 — across 458 recorded attempts every
    # nonzero exit was a process that died before it served anything (all
    # under 3s, none with a planner trace). So a nonzero rc alongside a
    # summary means the summary is not this attempt's.
    if rc != 0:
        bad.append(f"bench exited rc={rc} — summary is not this attempt's")
    if wall > contract.max_wall_s:
        bad.append(f"wall {wall:.1f}s > {contract.max_wall_s:.0f}s")

    completed = summary.get("completed") or 0
    failed = summary.get("failed") or 0
    requests = summary.get("requests") or 0

    if completed < contract.min_completed:
        bad.append(f"completed {completed} < {contract.min_completed}")
    if contract.max_failed is not None and failed > contract.max_failed:
        bad.append(f"failed {failed} > {contract.max_failed}")
    if contract.accounted and requests and completed + failed != requests:
        bad.append(f"unaccounted: {completed}+{failed} != {requests}")

    counters = trace_counters(log_text)
    for name, cap in contract.max_counters.items():
        value = counters.get(name, 0)
        if value > cap:
            bad.append(f"counter {name}={value} > {cap} (runaway)")
    for needle in contract.forbid_log:
        if needle in log_text:
            bad.append(f"log contains {needle!r}")
    return bad


def check_target(scenario: Scenario, records: list[dict],
                 traced: bool) -> list[str]:
    """Scenario-level half: did any attempt actually reach the target path?"""
    contract = scenario.contract
    if not contract.require_counters:
        return []
    if not traced:
        return ["no [planner-trace] line was emitted in any attempt — "
                "counters cannot be checked; lower --trace-ms below the "
                "scenario's runtime"]
    bad = []
    for name in contract.require_counters:
        best = max((record["counters"].get(name, 0) for record in records),
                   default=0)
        if best <= 0:
            bad.append(f"counter {name!r} never fired in "
                       f"{len(records)} attempt(s) — scenario no longer "
                       f"reaches {scenario.target}")
    return bad


def await_free_gpu(budget_s: float = 180.0, floor_mib: int = 1500) -> None:
    """Block until the previous scenario's device memory is actually gone.

    Best effort: if `nvidia-smi` is missing or never drains we return anyway
    and let the scenario report the real failure, rather than turning a
    diagnostic wait into a second failure mode.
    """
    deadline = time.time() + budget_s
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
        except Exception:
            return
        if out.returncode != 0:
            return
        try:
            used = max(int(line) for line in out.stdout.split() if line.strip())
        except ValueError:
            return
        if used < floor_mib:
            return
        time.sleep(2.0)


def run_once(scenario: Scenario, args: argparse.Namespace,
             attempt: int) -> tuple[bool, dict]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = scenario.name if scenario.repeat == 1 else f"{scenario.name}.{attempt}"
    jout = out / f"{stem}.json"
    log = out / f"{stem}.log"

    env = dict(os.environ)
    env.setdefault("PIE_BENCH_INFERLET_DIR",
                   str(ROOT / "tests" / "inferlets" / "text-completion-bench"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    compat = cuda_compat_dir()
    if compat:
        current = env.get("LD_LIBRARY_PATH", "")
        if compat not in current.split(":"):
            env["LD_LIBRARY_PATH"] = f"{compat}:{current}" if current else compat
    # Counter assertions need the planner trace; a coarse interval keeps the
    # log small without losing the cumulative peaks.
    env["PIE_CONTENTION_TRACE_MS"] = str(args.trace_ms)
    env.update(scenario.env)

    cmd = [args.python, "pie_bench.py", "tput",
           "--driver", args.driver, "--model", args.model,
           "--warmup", str(scenario.warmup),
           "--request-timeout", str(args.request_timeout),
           "--json-out", str(jout)]
    if "--max-model-len" not in scenario.args:
        cmd += ["--max-model-len", str(args.max_model_len)]
    cmd += scenario.args

    # A crashed attempt writes no `--json-out`, and a leftover file from an
    # EARLIER run of the same scenario would then be read as this attempt's
    # summary — a run that died at model load scored `completed=2048` that
    # way and passed. Unlink first so a crash reads as "no summary written",
    # which is what it is.
    jout.unlink(missing_ok=True)
    # A finished bench can hold tens of GiB for up to a minute; the next
    # scenario then dies with `pie_cuda_load_model failed with status -5`.
    # That is an artifact of running scenarios back to back, not a finding
    # about the engine, so wait the previous one out.
    await_free_gpu()

    started = time.time()
    with log.open("wb") as handle:
        try:
            rc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                env=env, cwd=BENCH,
                                timeout=scenario.contract.max_wall_s).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    wall = time.time() - started

    try:
        summary = json.load(jout.open())["summary"]
    except Exception:
        summary = None
    log_text = log.read_text(errors="replace")

    violations = check(scenario, rc, wall, summary, log_text)
    record = dict(scenario=scenario.name, target=scenario.target,
                  attempt=attempt, rc=rc, wall_s=round(wall, 2),
                  completed=(summary or {}).get("completed"),
                  failed=(summary or {}).get("failed"),
                  out_tok_s=(summary or {}).get("output_tok_per_s"),
                  counters=trace_counters(log_text),
                  traced="[planner-trace]" in log_text,
                  violations=violations)
    with (out / "results.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return not violations, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", "--only", default="",
                        help="comma-separated scenario names")
    parser.add_argument("--driver", default="cuda_native",
                        help="real driver to exercise (no mock devices)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--python",
                        default=os.environ.get("PIE_CONTENTION_SUITE_PYTHON",
                                               str(DEFAULT_PYTHON)))
    parser.add_argument("--out", default=str(ROOT / "target" / "contention-suite"))
    parser.add_argument("--repeat", type=int, default=0,
                        help="override every scenario's repeat count")
    parser.add_argument("--trace-ms", type=int, default=200,
                        help="planner-trace sampling period. Must stay well "
                             "below the shortest scenario's runtime or its "
                             "counters are never observed: the fast-fail "
                             "scenarios finish in under a second.")
    parser.add_argument("--request-timeout", type=float, default=150.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    selected = SCENARIOS
    if args.only:
        names = [name.strip() for name in args.only.split(",") if name.strip()]
        unknown = [name for name in names if name not in BY_NAME]
        if unknown:
            parser.error(f"unknown scenario(s): {', '.join(unknown)}")
        selected = [BY_NAME[name] for name in names]

    if args.list:
        for scenario in selected:
            print(f"{scenario.name:18} {scenario.target}")
        return 0

    if not Path(args.python).exists():
        print(f"interpreter not found: {args.python}\n"
              "build it with: uv --project sdk/python-server sync", file=sys.stderr)
        return 2

    failures: list[str] = []
    for scenario in selected:
        repeat = args.repeat or scenario.repeat
        records: list[dict] = []
        for attempt in range(1, repeat + 1):
            ok, record = run_once(scenario, args, attempt)
            records.append(record)
            label = scenario.name if repeat == 1 else f"{scenario.name} [{attempt}/{repeat}]"
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {label:26} {record['wall_s']:7.1f}s "
                  f"completed={record['completed']} failed={record['failed']} "
                  f"<- {scenario.target}", flush=True)
            for violation in record["violations"]:
                print(f"         {violation}", flush=True)
            if not ok:
                failures.append(label)
        traced = any(record["traced"] for record in records)
        for violation in check_target(scenario, records, traced):
            print(f"[FAIL] {scenario.name:26} {violation}", flush=True)
            failures.append(f"{scenario.name} (target)")

    print()
    if failures:
        print(f"{len(failures)} scenario(s) FAILED: {', '.join(failures)}")
        return 1
    print(f"all {len(selected)} scenario(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

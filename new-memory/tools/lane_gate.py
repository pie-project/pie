#!/usr/bin/env python3
"""What the engine lane's wait costs: the same serving columns for several
`pie` binaries of the same engine (a base, the base again, a candidate).

usage:
  python3 new-memory/tools/lane_gate.py \\
      --tree base=PIE --tree base-again=PIE --tree park=PIE \\
      --model MODEL --home PIE_HOME --wasm TEXT_COMPLETION_BENCH_WASM \\
      --out DIR [--engine cuda --device cuda:0] [--lock /tmp/pie-gpu.lock]
  python3 new-memory/tools/lane_gate.py --out DIR --report

`--tree NAME=PIE`: a `pie` binary under a name; the first is the base every
other is paired with. Name the base binary twice (`base-again`) and its
column against the base is the spread of the base against itself, taken the
same way as a candidate's. `--home`: a PIE_HOME that holds the model's
import (`pie model import MODEL`); each serve runs under its own home beside
`--out`, sharing only that import, and the home is deleted after it.
`--wasm`: `text_completion_bench.wasm`, built from `examples`
(`cargo build --release --target wasm32-wasip3 -p text-completion-bench`).
The python needs the client's packages (`websockets`, `msgpack`, `blake3`);
the client itself is read from this repository.

A round is every measure for every tree, the trees' order turned each round.
Each (tree, measure) is one hold of `--lock`: one serve and one measure,
under `--hold-timeout`. No timed piece starts while the lane still arms its
rungs: the serve's log is waited on for the runtime's "every rung's bodies
are armed", and the wait is recorded (a tree that arms all at load prints
neither arming line and waits nothing). Every timed piece is bracketed by
the cgroup's `nr_throttled`: a piece with a period throttled is dropped and
listed, and a (tree, measure) with no piece left is no round. Rounds are
added until every tree has `--rounds` whole rounds of every measure (or
`--max-rounds`); each round says which headline columns it took past the
base's range so far, against itself, so more rounds can be asked for by
running the same line again with the same `--out`.

Measures (`--only` picks some):
  c1, c64        N pipelines decoding with the guest's own run-ahead:
                 tok/s, per-token p50/p99 (guest drain stamps on the host's
                 monotonic clock), the scheduler's seal and device-idle
                 counters, the lane thread's CPU share, and the host's
                 one-minute load when the piece ended (a parked lane wakes
                 as fast as the host finds it a CPU).
  c1-loop,       the same with one frame queued a pipeline: the guest is in
  c64-loop       the loop of every token, as a guest that masks or samples
                 on the host is.
  trace-c1,      the same as c1/c64 under PIE_WAVE_TRACE=buffer: the gaps
  trace-c64      between a frame retiring with nothing in flight and the
                 next dispatch (sum, p50, p99), and dispatch-to-retire per
                 frame. The scheduler's view: the lane's own wake is inside
                 dispatch-to-retire, not in the gap.
  pause          one-token requests, each straight after the last or after
                 `--pause` seconds idle: request to result, the guest's time
                 to first token, and the lane thread's CPU over the pause.
"""
import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEASURES = ["c1", "c64", "c1-loop", "c64-loop", "pause", "trace-c1", "trace-c64"]


def arguments():
    a = argparse.ArgumentParser(usage=__doc__)
    a.add_argument("--tree", action="append", default=[], metavar="NAME=PIE")
    a.add_argument("--model")
    a.add_argument("--home")
    a.add_argument("--wasm")
    a.add_argument("--out", required=True)
    a.add_argument("--engine", default="cuda")
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--engine-line", action="append", default=[], help="a further line of [engine]")
    a.add_argument("--lock", default="/tmp/pie-gpu.lock")
    a.add_argument("--only", default=",".join(MEASURES))
    a.add_argument("--rounds", type=int, default=3, help="whole rounds a tree and measure")
    a.add_argument("--max-rounds", type=int, default=12, help="rounds this call runs, at most")
    a.add_argument("--reps", type=int, default=6, help="timed pieces a decode hold")
    a.add_argument("--tokens", type=int, default=512, help="tokens a pipeline a timed piece")
    a.add_argument("--pause", type=float, default=3.0, help="seconds idle before a paused request")
    a.add_argument("--pauses", type=int, default=5, help="paused requests a hold")
    a.add_argument("--boot-timeout", type=float, default=600.0)
    a.add_argument("--request-timeout", type=float, default=300.0)
    a.add_argument("--hold-timeout", type=float, default=1500.0)
    a.add_argument("--report", action="store_true", help="print the table of --out and stop")
    a.add_argument("--worker", nargs=3, metavar=("ROUND", "NAME", "MEASURE"), help=argparse.SUPPRESS)
    return a.parse_args()


# ── what the host says ──────────────────────────────────────────────────

def throttled():
    """Periods this process's cgroup, or one above it, was throttled; None
    where there is no such count (not Linux, cgroup v1, no cpu controller)."""
    try:
        path = [line.split("::", 1)[1].strip() for line in open("/proc/self/cgroup") if line.startswith("0::")][0]
    except (OSError, IndexError):
        return None
    total, found = 0, False
    at = Path("/sys/fs/cgroup") / path.lstrip("/")
    while True:
        try:
            for line in open(at / "cpu.stat"):
                if line.startswith("nr_throttled "):
                    total += int(line.split()[1])
                    found = True
        except OSError:
            pass
        if at == Path("/sys/fs/cgroup"):
            break
        at = at.parent
    return total if found else None


def lane_cpu(pid, engine_idx=0):
    """(seconds the engine lane's thread has run, where that was read).
    Linux reads the thread; macOS asks libproc for it; failing both, the
    whole process's CPU time stands in, and says so."""
    name = f"pie-engine-{engine_idx}"
    task = Path(f"/proc/{pid}/task")
    if task.is_dir():
        for thread in task.iterdir():
            try:
                if (thread / "comm").read_text().strip() != name:
                    continue
                try:
                    return int((thread / "schedstat").read_text().split()[0]) / 1e9, "thread"
                except (OSError, ValueError, IndexError):
                    fields = (thread / "stat").read_text().rsplit(")", 1)[1].split()
                    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"), "thread"
            except OSError:
                continue
    if sys.platform == "darwin":
        try:
            return _darwin_thread_cpu(pid, name), "thread"
        except Exception:
            pass
    out = subprocess.run(["ps", "-o", "cputime=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    days, _, clock = out.rpartition("-")
    seconds = 0.0
    for part in clock.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds + 86400 * int(days or 0), "process"


def _darwin_thread_cpu(pid, name):
    import ctypes
    import ctypes.util

    class ThreadInfo(ctypes.Structure):
        _fields_ = [("user", ctypes.c_uint64), ("system", ctypes.c_uint64)] + [
            (field, ctypes.c_int32)
            for field in ("usage", "policy", "run_state", "flags", "sleep", "curpri", "priority", "maxpriority")
        ] + [("name", ctypes.c_char * 64)]

    class Timebase(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    base = Timebase()
    libc.mach_timebase_info(ctypes.byref(base))
    threads = (ctypes.c_uint64 * 8192)()
    size = libc.proc_pidinfo(pid, 6, ctypes.c_uint64(0), threads, ctypes.sizeof(threads))  # PROC_PIDLISTTHREADS
    for handle in threads[: max(size, 0) // 8]:
        info = ThreadInfo()
        got = libc.proc_pidinfo(pid, 5, ctypes.c_uint64(handle), ctypes.byref(info), ctypes.sizeof(info))  # PROC_PIDTHREADINFO
        if got > 0 and info.name == name.encode():
            return (info.user + info.system) * base.numer / base.denom / 1e9
    raise LookupError(name)


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


# ── one serve, one measure ──────────────────────────────────────────────

class Serve:
    def __init__(self, args, name, pie, measure):
        self.args, self.name, self.pie, self.measure = args, name, pie, measure
        self.out = Path(args.out)
        self.home = self.out / f"home-{name}-{measure}-{os.getpid()}"
        self.log_path = self.out / "logs" / f"{name}-{measure}-{int(time.time())}.log"
        self.srv = None
        self.client = None

    async def start(self, trace):
        from pie_client import PieClient

        self.home.mkdir(parents=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        for shared in ("artifacts", "models"):
            if (Path(self.args.home) / shared).exists():
                (self.home / shared).symlink_to((Path(self.args.home) / shared).resolve())
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        engine = [f'type = "{self.args.engine}"', f'device = ["{self.args.device}"]'] + self.args.engine_line
        config = self.home / "config.toml"
        config.write_text(
            f'[server]\nlisten = "127.0.0.1:{port}"\nverbose = false\n\n'
            f'[model]\nname = "default"\nsource = "{self.args.model}"\n\n'
            "[engine]\n" + "\n".join(engine) + "\n\n[runtime]\n"
        )
        # The arming lines are the runtime's, at info.
        env = dict(os.environ, PIE_HOME=str(self.home), RUST_LOG="info")
        env.pop("PIE_WAVE_TRACE", None)
        if trace:
            env["PIE_WAVE_TRACE"] = "buffer"
        began = time.perf_counter()
        self.srv = subprocess.Popen(
            [self.pie, "--config", str(config), "serve"],
            stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        while "Server ready" not in self.log_path.read_text(errors="ignore"):
            if self.srv.poll() is not None:
                raise SystemExit(f"the server exited; see {self.log_path}")
            if time.perf_counter() - began > self.args.boot_timeout:
                raise SystemExit(f"no server after {self.args.boot_timeout:.0f} s; see {self.log_path}")
            await asyncio.sleep(0.05)
        boot = time.perf_counter() - began
        self.client = PieClient(f"ws://127.0.0.1:{port}")
        await asyncio.wait_for(self.client.connect(), 120)
        self.program = await asyncio.wait_for(self.client.install_program(self.args.wasm), 300)
        self.ready_at, self.armed_by = time.perf_counter(), None
        await asyncio.sleep(1.0)
        wait = await self.armed()
        return {"boot_s": round(boot, 2), "arming_wait_s": wait, "armed_by": self.armed_by}

    async def armed(self):
        """Waits out a lane that arms its rungs while it serves (the
        runtime's "the lane arms the rest of the rungs" line, until its
        "every rung's bodies are armed"); called before every timed piece.
        The seconds from the server's ready line to the armed line, 0 for a
        tree that arms all at load (neither line)."""
        if self.armed_by == "lane":
            return 0.0
        while True:
            text = self.log_path.read_text(errors="ignore")
            if "every rung's bodies are armed" in text:
                self.armed_by = "lane"
                return round(time.perf_counter() - self.ready_at, 2)
            if "the lane arms the rest" not in text:
                self.armed_by = "load"
                return 0.0
            if time.perf_counter() - self.ready_at > self.args.boot_timeout:
                raise SystemExit(f"arming did not end in {self.args.boot_timeout:.0f} s; see {self.log_path}")
            await asyncio.sleep(0.05)

    async def run(self, inp):
        """One process to its end: (seconds from the request, its result)."""
        began = time.perf_counter()
        process = await asyncio.wait_for(self.client.launch_process(self.program, input=inp), self.args.request_timeout)
        result = await asyncio.wait_for(process.result(), self.args.request_timeout)
        return time.perf_counter() - began, result

    async def counters(self):
        ok, status = await self.client.query("model_status", "")
        status = json.loads(status) if ok else {}
        return {key.split(".fire.quorum.", 1)[1]: value for key, value in status.items() if ".fire.quorum." in key}

    async def stop(self):
        try:
            if self.client is not None:
                await asyncio.wait_for(self.client.close(), 30)
        except Exception:
            pass
        if self.srv is not None and self.srv.poll() is None:
            os.killpg(self.srv.pid, signal.SIGTERM)
            try:
                self.srv.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.srv.pid, signal.SIGKILL)
                self.srv.wait()
        shutil.rmtree(self.home, ignore_errors=True)


def prompt(pipeline, piece):
    # As many digits whatever the pipeline and the piece: every prompt is as
    # long, and no two of a hold are the same.
    return f"Count upward from {1000 + (97 * pipeline + 6211 * piece) % 9000}, one number a line, and never stop."


async def decode_piece(serve, pipelines, in_loop, piece, traced):
    """`pipelines` single-pipeline processes at once, `--tokens` each."""
    args = serve.args
    inp = {"max_tokens": args.tokens, "ignore_eos": True, "return_text": False, "report_arrivals": True}
    if in_loop:
        inp["run_ahead_frames"] = 1
    await serve.armed()
    if traced:
        await asyncio.sleep(2.5)  # the trace's buffer is written out each second
        log_from = serve.log_path.stat().st_size
    before = (throttled(), lane_cpu(serve.srv.pid), await serve.counters(), time.perf_counter())
    runs = await asyncio.gather(*[serve.run(dict(inp, prompt=prompt(p, piece))) for p in range(pipelines)])
    after = (throttled(), lane_cpu(serve.srv.pid), await serve.counters(), time.perf_counter())
    stamps = [result["token_monotonic_ns"] for _, result in runs]
    if any(len(s) < 3 for s in stamps):
        raise SystemExit(f"a pipeline returned {min(len(s) for s in stamps)} token stamps")
    # The window every pipeline decodes in: from the last first token to the
    # first last token.
    start, end = max(s[0] for s in stamps), min(s[-1] for s in stamps)
    gaps = sorted((b - a) / 1e3 for s in stamps for a, b in zip(s, s[1:]) if a >= start and b <= end)
    tokens = sum(1 for s in stamps for at in s if start < at <= end)
    wall = after[3] - before[3]
    delta = {key: after[2].get(key, 0) - before[2].get(key, 0) for key in after[2] if isinstance(after[2][key], (int, float))}
    out = {
        "throttled": None if before[0] is None else after[0] - before[0],
        "load1": os.getloadavg()[0],
        "wall_s": round(wall, 4),
        "window_s": round((end - start) / 1e9, 4),
        "tokens": tokens,
        "tok_s": tokens / ((end - start) / 1e9) if end > start else None,
        "token_p50_us": quantile(gaps, 0.50),
        "token_p99_us": quantile(gaps, 0.99),
        "token_max_us": gaps[-1] if gaps else None,
        "lane_cpu_share": (after[1][0] - before[1][0]) / wall,
        "lane_cpu_from": after[1][1],
        "seal_events": delta.get("seal_events"),
        "seal_while_executing": delta.get("seal_while_executing"),
        "seal_executing_share": (delta["seal_while_executing"] / delta["seal_events"]) if delta.get("seal_events") else None,
        "device_idle_us": delta.get("device_idle_us"),
        "device_idle_gaps": delta.get("device_idle_gaps"),
        "device_idle_share": (delta["device_idle_us"] / 1e6 / wall) if "device_idle_us" in delta else None,
    }
    if traced:
        await asyncio.sleep(2.5)
        with open(serve.log_path, errors="ignore") as log:
            log.seek(log_from)
            out.update(trace_columns(log.read(), wall))
    return out


def trace_columns(text, wall):
    """Frames by the scheduler's trace: a dispatch hands the lane a frame, a
    retire is its completion seen. A gap is a retire that leaves nothing in
    flight, to the next dispatch."""
    dispatched, in_flight, idle_from = [], 0, None
    gaps, frames = [], []
    for line in text.splitlines():
        if not line.startswith("[wave-trace] t="):
            continue
        fields = line.split()
        at = int(fields[1][2:-2])
        if fields[2] == "dispatch":
            if idle_from is not None:
                gaps.append(at - idle_from)
                idle_from = None
            dispatched.append(at)
            in_flight += 1
        elif fields[2] == "retire" and dispatched:
            frames.append(at - dispatched.pop(0))
            in_flight -= 1
            if in_flight == 0:
                idle_from = at
    gaps.sort()
    frames.sort()
    return {
        "trace_frames": len(frames),
        "trace_unpaired": len(dispatched),
        "trace_gaps": len(gaps),
        "trace_gap_sum_us": sum(gaps),
        "trace_gap_share": sum(gaps) / 1e6 / wall,
        "trace_gap_p50_us": quantile(gaps, 0.50),
        "trace_gap_p99_us": quantile(gaps, 0.99),
        "trace_frame_p50_us": quantile(frames, 0.50),
        "trace_frame_p99_us": quantile(frames, 0.99),
    }


async def pause_pieces(serve):
    """One-token requests: one after `--pause` idle, one straight after it."""
    args = serve.args
    inp = {"max_tokens": 1, "ignore_eos": True, "return_text": False, "report_timing": True}
    pieces = []
    for piece in range(args.pauses):
        await serve.armed()
        cpu_before, began = lane_cpu(serve.srv.pid), time.perf_counter()
        await asyncio.sleep(args.pause)
        idle_cpu = (lane_cpu(serve.srv.pid)[0] - cpu_before[0]) / (time.perf_counter() - began)
        row = {"lane_idle_cpu_share": idle_cpu, "lane_cpu_from": cpu_before[1], "load1": os.getloadavg()[0]}
        for column, seed in (("paused", 2 * piece), ("straight", 2 * piece + 1)):
            before = throttled()
            took, result = await serve.run(dict(inp, prompt=prompt(0, seed)))
            spoiled = None if before is None else throttled() - before
            row[f"{column}_ms"] = took * 1e3
            row[f"{column}_first_token_ms"] = result["ttft_us"] / 1e3
            row["throttled"] = (row.get("throttled") or 0) + spoiled if spoiled is not None else None
        pieces.append(row)
    return pieces


async def worker(args):
    round_, name, measure = args.worker
    sys.path.insert(0, str(REPO / "packages/python/client/src"))
    pie = dict(tree.split("=", 1) for tree in args.tree)[name]
    record = {"round": int(round_), "tree": name, "measure": measure, "pie": pie, "at": time.time()}
    serve = Serve(args, name, pie, measure)
    try:
        record.update(await serve.start(trace=measure.startswith("trace-")))
        record["log"] = str(serve.log_path)
        if measure == "pause":
            await serve.armed()
            for warm in range(3):
                await serve.run({"max_tokens": 1, "ignore_eos": True, "return_text": False, "prompt": prompt(1, warm)})
            record["pieces"] = await pause_pieces(serve)
        else:
            shape = measure.removeprefix("trace-")
            pipelines, in_loop, traced = int(shape.split("-")[0][1:]), shape.endswith("-loop"), measure.startswith("trace-")
            pieces = []
            # Piece 0 warms the shape and is not kept.
            for piece in range((1 if traced else args.reps) + 1):
                got = await decode_piece(serve, pipelines, in_loop, piece, traced)
                if piece:
                    pieces.append(got)
            record["pieces"] = pieces
    except (SystemExit, Exception) as error:
        record["error"] = f"{type(error).__name__}: {error}"[:600]
    finally:
        await serve.stop()
        with open(Path(args.out) / "samples.jsonl", "a") as samples:
            samples.write(json.dumps(record) + "\n")
    return 1 if "error" in record else 0


# ── the rounds ──────────────────────────────────────────────────────────

# Columns of a measure: (name, larger is better or None, headline).
DECODE = [
    ("tok_s", True), ("token_p50_us", True), ("token_p99_us", False), ("lane_cpu_share", False),
    ("seal_executing_share", False), ("seal_events", False), ("seal_while_executing", False),
    ("device_idle_share", False), ("device_idle_gaps", False), ("load1", False),
]
TRACE = [
    ("trace_gap_share", True), ("trace_gap_sum_us", False), ("trace_gaps", False), ("trace_gap_p50_us", False),
    ("trace_gap_p99_us", False), ("trace_frame_p50_us", True), ("trace_frame_p99_us", False), ("trace_frames", False),
    ("trace_unpaired", False), ("tok_s", False), ("load1", False),
]
PAUSE = [
    ("paused_ms", True), ("straight_ms", True), ("paused_first_token_ms", True), ("straight_first_token_ms", True),
    ("lane_idle_cpu_share", True), ("load1", False),
]


def columns(measure):
    return PAUSE if measure == "pause" else TRACE if measure.startswith("trace-") else DECODE


def rounds_of(out):
    """{(tree, measure): {round: {column: the median of its clean pieces}}}, and what was dropped."""
    table, dropped = {}, []
    path = Path(out) / "samples.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        record = json.loads(line)
        key = (record["tree"], record["measure"])
        if "error" in record:
            dropped.append((record["round"], *key, record["error"]))
            continue
        clean = [piece for piece in record["pieces"] if not piece.get("throttled")]
        spoiled = len(record["pieces"]) - len(clean)
        if spoiled:
            dropped.append((record["round"], *key, f"{spoiled} of {len(record['pieces'])} pieces throttled"))
        if not clean:
            continue
        values = {}
        for column, _ in columns(record["measure"]):
            got = [piece[column] for piece in clean if piece.get(column) is not None]
            if got:
                values[column] = statistics.median(got)
        table.setdefault(key, {})[record["round"]] = values
    return table, dropped


def number(value):
    if value is None:
        return "-"
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.1f}" if abs(value) >= 100 else f"{value:.2f}" if abs(value) >= 1 else f"{value:.4f}"


def spread(values):
    if not values:
        return "-"
    return f"{number(statistics.median(values))} [{number(min(values))} .. {number(max(values))}]"


def report(out, trees, measures):
    table, dropped = rounds_of(out)
    base = trees[0]
    print(f"\nEach cell: the median over a tree's rounds [the least .. the most]; n = whole rounds.")
    print(f"'vs {base}': the tree's value less {base}'s in the same round, over the rounds both have.\n")
    for measure in measures:
        print(f"== {measure}")
        for column, _ in columns(measure):
            print(f"  {column}")
            for tree in trees:
                rounds = table.get((tree, measure), {})
                values = [rounds[r][column] for r in sorted(rounds) if column in rounds[r]]
                line = f"    {tree:<12} n={len(values):<2} {spread(values)}"
                if tree != base:
                    ours, theirs = rounds, table.get((base, measure), {})
                    paired = [ours[r][column] - theirs[r][column] for r in sorted(ours) if r in theirs and column in ours[r] and column in theirs[r]]
                    line += f"   vs {base}: {spread(paired)} (n={len(paired)})"
                print(line)
    if dropped:
        print("\nDropped:")
        for round_, tree, measure, why in dropped:
            print(f"  round {round_} {tree} {measure}: {why}")


def widened(out, trees, again, measures, last):
    """Headline columns where round `last` took the base's range, or the
    range of the base named again less the base, past every round before."""
    table, _ = rounds_of(out)
    base = trees[0]
    wider = []
    for measure in measures:
        for column, headline in columns(measure):
            if not headline:
                continue
            series = {base: {r: v[column] for r, v in table.get((base, measure), {}).items() if column in v}}
            for tree in again:
                theirs = table.get((tree, measure), {})
                series[f"{tree}-{base}"] = {r: theirs[r][column] - series[base][r] for r in theirs if r in series[base] and column in theirs[r]}
            for name, values in series.items():
                before = [v for r, v in values.items() if r < last]
                if last in values and before and not min(before) <= values[last] <= max(before):
                    wider.append(f"{measure}/{column} ({name})")
    return wider


def driver(args):
    import fcntl

    trees = [tree.split("=", 1)[0] for tree in args.tree]
    binaries = [os.path.realpath(tree.split("=", 1)[1]) for tree in args.tree]
    again = [tree for tree, pie in zip(trees[1:], binaries[1:]) if pie == binaries[0]]
    measures = [m for m in args.only.split(",") if m]
    for measure in measures:
        if measure not in MEASURES:
            raise SystemExit(f"no measure {measure!r}: {MEASURES}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table, _ = rounds_of(out)
    # Rounds go on from the last one these measures have: a later call may
    # add measures to an --out, or rounds to the ones it has.
    done = max([r for (_, measure), rounds in table.items() if measure in measures for r in rounds], default=0)
    passed = [arg for arg in sys.argv[1:]]
    holding = []

    def stopped(signum, _frame):
        # Stopped while a hold runs: its worker and its serve go first, the
        # lock after them.
        for child, name, measure in holding:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except OSError:
                pass
            subprocess.run(["pkill", "-KILL", "-f", f"{out}/home-{name}-{measure}-"], capture_output=True)
            for home in out.glob(f"home-{name}-{measure}-*"):
                shutil.rmtree(home, ignore_errors=True)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    for round_ in range(done + 1, done + args.max_rounds + 1):
        turn = (round_ - 1) % len(trees)
        for measure in measures:
            for name in trees[turn:] + trees[:turn]:
                with open(args.lock, "a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    began = time.time()
                    child = subprocess.Popen(
                        [sys.executable, __file__, *passed, "--worker", str(round_), name, measure],
                        start_new_session=True,
                    )
                    holding.append((child, name, measure))
                    try:
                        code = child.wait(args.hold_timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                        code = "timeout"
                        with open(out / "samples.jsonl", "a") as samples:
                            samples.write(json.dumps({"round": round_, "tree": name, "measure": measure, "error": f"no answer in {args.hold_timeout:.0f} s"}) + "\n")
                    # A serve its worker did not live to stop.
                    subprocess.run(["pkill", "-KILL", "-f", f"{out}/home-{name}-{measure}-"], capture_output=True)
                    for home in out.glob(f"home-{name}-{measure}-*"):
                        shutil.rmtree(home, ignore_errors=True)
                    holding.clear()
                    print(f"round {round_} {measure} {name}: {code} in {time.time() - began:.0f} s", flush=True)
        table, _ = rounds_of(out)
        whole = min(len(table.get((tree, measure), {})) for tree in trees for measure in measures)
        wider = widened(out, trees, again, measures, round_)
        print(f"round {round_} done: {whole} whole rounds a tree at least; widened this round: {wider or 'nothing'}", flush=True)
        if whole >= args.rounds:
            break
    report(out, trees, measures)


def main():
    args = arguments()
    if args.worker:
        sys.exit(asyncio.run(worker(args)))
    trees = [tree.split("=", 1)[0] for tree in args.tree]
    if args.report:
        table, _ = rounds_of(args.out)
        trees = trees or sorted({tree for tree, _ in table})
        report(args.out, trees, [m for m in MEASURES if any((tree, m) in table for tree in trees)])
        return
    if not (args.tree and args.model and args.home and args.wasm):
        raise SystemExit(__doc__)
    driver(args)


if __name__ == "__main__":
    main()

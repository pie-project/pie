"""Host-side driver: run agent-os strategies over MATH-500 and print a table.

    python3 bench.py --n 24 --levels 1,2,3 --configs fixed1,fixed4,adaptive

Each config is one `pie run` of main.py (one engine boot, all problems in one
supervisor so the batch stays full). Results land in results/<label>.json.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


import signal
import threading


def free_pct():
    """System-wide memory free percentage (macOS `memory_pressure`); None if unreadable."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5).stdout
        return int(re.search(r"free percentage:\s*(\d+)%", out).group(1))
    except Exception:  # noqa: BLE001
        return None


def rss_gb(pid):
    try:
        kb = int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip() or 0)
        return kb / 1048576
    except Exception:  # noqa: BLE001
        return 0.0


class MemGuard(threading.Thread):
    """Keep the machine under MAX_USED_PCT memory use. Above the pause line the engine is
    frozen (SIGSTOP: it cannot allocate while stopped) until memory is back under the
    resume line; if that never happens the run is aborted so the PC never swaps itself dead."""

    MAX_USED_PCT = 90
    PAUSE_AT_USED = 86      # freeze with a margin: one in-flight wave can allocate a little more
    RESUME_AT_USED = 78
    ABORT_AFTER_S = 240

    def __init__(self, proc):
        super().__init__(daemon=True)
        self.proc, self.stop_flag = proc, threading.Event()
        self.max_used, self.max_rss, self.pauses, self.paused_s, self.aborted = 0, 0.0, 0, 0.0, False

    def run(self):
        paused_at = None
        while not self.stop_flag.is_set():
            f = free_pct()
            if f is not None:
                used = 100 - f
                self.max_used = max(self.max_used, used)
                if paused_at is None and used >= self.PAUSE_AT_USED:
                    os.kill(self.proc.pid, signal.SIGSTOP)
                    paused_at = time.time()
                    self.pauses += 1
                    sys.stderr.write(f"[memguard] {used}% used: engine paused\n")
                elif paused_at is not None:
                    if used <= self.RESUME_AT_USED:
                        os.kill(self.proc.pid, signal.SIGCONT)
                        self.paused_s += time.time() - paused_at
                        paused_at = None
                        sys.stderr.write(f"[memguard] {used}% used: engine resumed\n")
                    elif time.time() - paused_at > self.ABORT_AFTER_S:
                        self.aborted = True
                        os.kill(self.proc.pid, signal.SIGCONT)
                        self.proc.kill()
                        sys.stderr.write("[memguard] pressure did not ease: run aborted\n")
                        return
            if self.proc.poll() is None:
                self.max_rss = max(self.max_rss, rss_gb(self.proc.pid))
            self.stop_flag.wait(0.3)
        if paused_at is not None:
            os.kill(self.proc.pid, signal.SIGCONT)

HERE = os.path.dirname(os.path.abspath(__file__))
PIE = os.environ.get("PIE_BIN", os.path.expanduser("~/workspace/pie/target/release/pie"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load_math(levels, n, offset=0):
    path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl"))[0]
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["level"] in levels]
    rows = rows[offset:offset + n]
    return [{"kind": "math", "question": r["problem"], "gold": r["answer"], "level": r["level"], "subject": r["subject"]} for r in rows]


def run(label, problems, policy, budget_per_problem=None, max_new=320, max_rows=32, temperature=0.7,
        seed=17, trace=True, stream=False, timeout=7200, solver_sys=None, state_slots=192, engine=None, model_name=None, arena_pages=None):
    cfg = {"mode": "solve", "problems": problems, "policy": policy, "max_new": max_new, "max_rows": max_rows,
           "temperature": temperature, "seed": seed, "trace": trace, "stream": stream, "state_slots": state_slots}
    if budget_per_problem:
        cfg["budget_per_problem"] = budget_per_problem
    if solver_sys:
        cfg["solver_sys"] = solver_sys
    if arena_pages:
        cfg["arena_pages"] = arena_pages
    # never start on a machine that is already under pressure
    start_free = free_pct()
    if start_free is not None and 100 - start_free > MemGuard.RESUME_AT_USED:
        raise RuntimeError(f"{label}: {100 - start_free}% of memory in use; refusing to start (limit {MemGuard.RESUME_AT_USED}%)")
    conf = open(os.path.join(HERE, "agentos.toml")).read()
    if model_name:
        conf = re.sub(r'(?m)^model = ".*"$', f'model = "{model_name}"', conf)
    over = {"max_state_slots": state_slots, **(engine or {})}
    for k, v in over.items():
        conf = re.sub(rf"(?m)^{k}\s*=.*$", "", conf)
        conf += f"\n{k} = {v}\n"
    cpath = os.path.join(HERE, f".run-{os.getpid()}.toml")
    open(cpath, "w").write(conf)
    cmd = [PIE, "-c", cpath, "run", os.path.join(HERE, "main.py"), "--", "--cfg", json.dumps(cfg)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=HERE)
    guard = MemGuard(proc)
    guard.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    finally:
        guard.stop_flag.set()
        guard.join(timeout=2)
        if os.path.exists(cpath):
            os.remove(cpath)

    class _P:  # shape the rest of the function expects
        pass
    proc_res = _P()
    proc_res.stdout, proc_res.stderr, proc_res.returncode = stdout, stderr, proc.returncode
    proc = proc_res
    out = ANSI.sub("", proc.stdout)
    result = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"summary"' in line:
            result = json.loads(line)
            break
    if result is None:
        sys.stderr.write(ANSI.sub("", proc.stderr)[-3000:] + "\n" + out[-2000:] + "\n")
        raise RuntimeError(f"{label}: no result (exit {proc.returncode}{', memory guard aborted' if guard.aborted else ''})")
    result["wall_total_s"] = round(time.time() - t0, 1)
    result["memory"] = {"start_used_pct": None if start_free is None else 100 - start_free, "max_used_pct": guard.max_used,
                        "max_engine_rss_gb": round(guard.max_rss, 2), "pauses": guard.pauses, "paused_s": round(guard.paused_s, 1)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(result, open(os.path.join(HERE, "results", f"{label}.json"), "w"))
    return result


def row(label, r):
    s = r["summary"]
    return (f"{label:34s} acc {s['accuracy']*100:5.1f}%  ({s['correct']}/{s['problems']})  "
            f"tok/prob {s['decode_tokens_per_problem']:7.1f}  waves {s['waves']:4d}  agg {s['agg_tok_s']:6.1f} tok/s  "
            f"peak rows {s['peak_rows']:2d}  wall {s['wall_s']:6.1f}s  killed {s['killed']:3d}  "
            f"mem max {r.get('memory', {}).get('max_used_pct', '?')}% rss {r.get('memory', {}).get('max_engine_rss_gb', '?')}GB")


CONFIGS = {
    "fixed1": {"name": "fixed", "n": 1},
    "fixed2": {"name": "fixed", "n": 2},
    "fixed4": {"name": "fixed", "n": 4},
    "fixed8": {"name": "fixed", "n": 8},
    "fixed16": {"name": "fixed", "n": 16},
    "fixed6": {"name": "fixed", "n": 6},
    "fixed7": {"name": "fixed", "n": 7},
    "council4x2": {"name": "council", "n": 4, "j": 2},
    "council4x3": {"name": "council", "n": 4, "j": 3},
    "council3x3": {"name": "council", "n": 3, "j": 3},
    "adaptive-lean": {"name": "adaptive", "w0": 2, "step": 1, "wmax": 6, "delta": 0.15, "judge": False, "giveup": 3},
    "adaptive-w8": {"name": "adaptive", "w0": 2, "step": 1, "wmax": 8, "delta": 0.15, "judge": False},
    "adaptive-w6": {"name": "adaptive", "w0": 2, "step": 1, "wmax": 6, "delta": 0.15, "judge": False},
    "adaptive": {"name": "adaptive", "w0": 2, "step": 2, "wmax": 12, "delta": 0.08, "judge": True, "judge_at": 8},
    "adaptive-nojudge": {"name": "adaptive", "w0": 2, "step": 2, "wmax": 12, "delta": 0.08, "judge": False},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--configs", default="fixed1,fixed4,adaptive")
    ap.add_argument("--budget", type=float, default=None, help="decode tokens per problem for adaptive")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--max-rows", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--arena-pages", type=int, default=None)
    ap.add_argument("--state-slots", type=int, default=192)
    ap.add_argument("--engine", action="append", default=[], help="engine override key=value (repeatable)")
    ap.add_argument("--model", default=None, help="HF id of an imported model, e.g. Qwen/Qwen3.5-4B")
    a = ap.parse_args()
    probs = load_math([int(x) for x in a.levels.split(",")], a.n, a.offset)
    print(f"{len(probs)} problems, levels {a.levels}")
    for name in a.configs.split(","):
        pol = CONFIGS[name]
        r = run(f"{a.tag}-{name}", probs, pol, budget_per_problem=a.budget if pol["name"] == "adaptive" else None,
                max_new=a.max_new, max_rows=a.max_rows, temperature=a.temperature, model_name=a.model,
                arena_pages=a.arena_pages, state_slots=a.state_slots, engine=dict(kv.split("=", 1) for kv in a.engine) or None)
        print(row(name, r), flush=True)

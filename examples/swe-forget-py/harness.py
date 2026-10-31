"""Run coding tasks with the swarm on pie, or with the same agent loop as one
trajectory against any OpenAI-compatible server (vLLM, or pie's own), and
score both the same way.

    pie serve &
    python harness.py swarm  --uri ws://127.0.0.1:8080          # the tree, on pie
    python harness.py linear --uri ws://127.0.0.1:8080          # one branch, on pie
    python harness.py openai --base-url http://127.0.0.1:8000/v1 --model NAME

Each run writes one JSON line per task to --out: solved, tokens, seconds.
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tasks import TOY, Task

HERE = Path(__file__).parent
PROGRAM = HERE / "main.py"
SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
COMMAND = re.compile(r"```([a-z0-9]*)[ \t]*\n(.*?)```", re.S)


def command(text: str) -> str | None:
    """The reply's command, read as the inferlet reads it: a python block runs as a script."""
    match = COMMAND.search(text)
    if not match:
        return None
    lang, body = match.group(1), match.group(2).strip()
    if lang in ("python", "py", "python3"):
        return f"python3 - <<'PIE_EOF'\n{body}\nPIE_EOF"
    return body
SYSTEM = (
    "You are a careful software engineer fixing an issue in the repository in the current "
    "directory. You work through a shell. Every reply is a short THOUGHT, then exactly one "
    "command in a ```bash block; it runs in a fresh subshell in the repository root and its "
    "output comes back to you. Find the relevant code, reproduce the bug with a small script, "
    "fix the source (not the tests), and check the fix. When you are done, reply with the "
    f"command: echo {SUBMIT}"
)
FORMAT_ERROR = (
    "Your reply must contain exactly one command in a ```bash block. "
    f"To finish, run: echo {SUBMIT}"
)


# the agent's changes, new files included, without bytecode
DIFF = "git add -A -N . && git diff HEAD -- . ':(exclude)*.pyc' ':(exclude)*/__pycache__/*'"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def clone(src: Path, dst: Path) -> None:
    """Copy-on-write where the filesystem has it: forks are cheap on APFS and XFS/btrfs."""
    flag = ["-c"] if platform.system() == "Darwin" else ["-a", "--reflink=auto"]
    subprocess.run(["cp", "-R", *flag, f"{src}/.", str(dst)], check=True)
    # the copy's file times differ from the index's: refresh it here, natively, or the
    # first diff inside a sandbox re-reads every file
    subprocess.run(["git", "-C", str(dst), "update-index", "-q", "--refresh"], capture_output=True)


class Sandboxes:
    """One directory per branch, each a copy of the task's repository."""

    def __init__(self, task: Task, timeout: int = 60) -> None:
        self.root = Path(tempfile.mkdtemp(prefix=f"swarm-{task.id}-", dir=os.environ.get("SWARM_TMP")))
        self.base = self.root / "base"
        self.base.mkdir()
        task.setup(self.base)
        self.dirs: dict[int, Path] = {}
        self.timeout = timeout

    def run(self, path: Path, cmd: str) -> tuple[int, str]:
        p = subprocess.run(["bash", "-c", cmd], cwd=path, capture_output=True, text=True,
                           timeout=self.timeout, env=ENV)
        return p.returncode, p.stdout + p.stderr

    def dir(self, bid: int) -> Path:
        if bid not in self.dirs:
            path = self.root / f"b{bid}"
            path.mkdir()
            clone(self.base, path)
            self.dirs[bid] = path
        return self.dirs[bid]

    def fork(self, src: int, dsts: list[int]) -> None:
        for d in dsts:
            path = self.root / f"b{d}"
            path.mkdir()
            clone(self.dir(src), path)
            self.dirs[d] = path

    def kill(self, bid: int) -> None:
        path = self.dirs.pop(bid, None)
        if path:
            shutil.rmtree(path, ignore_errors=True)

    def state(self, path: Path) -> str:
        return hashlib.sha1(self.patch(path).encode()).hexdigest()[:12]

    def patch(self, path: Path) -> str:
        return subprocess.run(DIFF, shell=True, cwd=path, capture_output=True, text=True, errors="replace").stdout

    def exec(self, bid: int, cmd: str) -> dict:
        path = self.dir(bid)
        if SUBMIT in cmd:
            return {"bid": bid, "rc": 0, "out": "", "state": self.state(path), "patch": self.patch(path)}
        try:
            rc, out = self.run(path, cmd)
        except subprocess.TimeoutExpired:
            rc, out = 124, f"timed out after {self.timeout} s"
        return {"bid": bid, "rc": rc, "out": out, "state": self.state(path)}

    def evaluate(self, task: Task, patch: str) -> bool:
        if not patch.strip():
            return False
        path = self.root / "eval"
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir()
        clone(self.base, path)
        applied = subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=path, capture_output=True)
        if applied.returncode:
            return False
        for name, text in task.hidden.items():
            (path / name).write_text(text)
        return subprocess.run(task.test, shell=True, cwd=path, capture_output=True,
                              timeout=self.timeout).returncode == 0

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class SweSandboxes(Sandboxes):
    """SWE-bench without Docker: each branch is a copy of the instance's installed
    base, bound at /testbed in the instance's environment by /swe/bin/exec."""

    def __init__(self, task: "SweTask", timeout: int = 120) -> None:
        self.task = task
        self.root = Path(tempfile.mkdtemp(prefix=f"swarm-{task.id}-", dir=os.environ.get("SWARM_TMP")))
        self.base = Path("/swe/bases") / task.id
        self.dirs: dict[int, Path] = {}
        self.timeout = timeout

    def run(self, path: Path, cmd: str) -> tuple[int, str]:
        p = subprocess.run(["/swe/bin/exec", self.task.id, str(path), str(self.timeout)], input=cmd,
                           capture_output=True, text=True, errors="replace", timeout=self.timeout + 30)
        return p.returncode, p.stdout + p.stderr

    def evaluate(self, task: "SweTask", patch: str) -> bool:
        if not patch.strip():
            return False
        path = self.root / "final.patch"
        path.write_text(patch)
        p = subprocess.run(["/swe/bin/eval", task.id, str(path)], capture_output=True, text=True, timeout=3600)
        lines = [line for line in p.stdout.splitlines() if line.startswith("{")]
        return bool(lines) and json.loads(lines[-1]).get("resolved", False)


class SweTask:
    def __init__(self, row: dict) -> None:
        self.id = row["instance_id"]
        self.issue = row["problem_statement"]
        self.row = row


def swe_tasks(limit: int) -> list[SweTask]:
    rows = {json.loads(line)["instance_id"]: json.loads(line) for line in open("/swe/instances.jsonl")}
    good = [line.strip() for line in open("/swe/good_ids.txt") if line.strip()]
    return [SweTask(rows[i]) for i in sorted(good)[:limit]]


def sandboxes(task) -> Sandboxes:
    return SweSandboxes(task) if isinstance(task, SweTask) else Sandboxes(task)


INSTALL = asyncio.Lock()
INSTALLED: dict = {}


async def run_pie(task: Task, uri: str, config: dict, quiet: bool) -> dict:
    from pie_client import Event, PieClient

    box = sandboxes(task)
    pool = ThreadPoolExecutor(16)
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    try:
        async with PieClient(uri) as client:
            await client.authenticate("swarm")
            async with INSTALL:  # once: tasks starting together overwrite it under each other
                if INSTALLED.get("program") is None:
                    INSTALLED["program"] = await client.install_program_bytes(
                        PROGRAM.read_bytes(), PROGRAM.stem + ".py", "0.1.0", force_overwrite=True)
            process = await client.launch_process(
                INSTALLED["program"], {"system": SYSTEM, "task": task.issue, "config": config})
            tool_seconds = 0.0
            while True:
                event, value = await process.recv()
                if event is Event.Message:
                    msg = json.loads(value)
                    op = msg.get("op")
                    if op == "fork":
                        box.fork(msg["from"], msg["to"])
                    elif op == "kill":
                        box.kill(msg["bid"])
                    elif op == "exec":
                        t = time.monotonic()
                        results = await asyncio.gather(*[
                            loop.run_in_executor(pool, box.exec, c["bid"], c["cmd"]) for c in msg["calls"]])
                        tool_seconds += time.monotonic() - t
                        await process.signal(json.dumps({"results": results}))
                    elif op == "log" and not quiet:
                        print(f"  {task.id}: {msg}", file=sys.stderr)
                elif event is Event.Return:
                    result = json.loads(value)
                    break
                elif event is Event.Error:
                    raise RuntimeError(value)
        result["solved"] = await loop.run_in_executor(None, box.evaluate, task, result["patch"])
        result["tool_seconds"] = round(tool_seconds, 2)
        result["wall_seconds"] = round(time.monotonic() - t0, 2)
        return result
    finally:
        box.close()


def chat(base_url: str, model: str, messages: list, max_tokens: int, temperature: float, reply: str = "") -> dict:
    # thinking off and plain temperature sampling, as the swarm does: no top-k/top-p or
    # penalties the model's generation_config would otherwise switch on
    messages = [{k: v for k, v in m.items() if k != "turn"} for m in messages]
    extra = {}
    if reply:  # go on from what the reply has so far
        messages.append({"role": "assistant", "content": reply})
        extra = {"add_generation_prompt": False, "continue_final_message": True}
    # a turn ends at the command block's closing fence, as the inferlet's does
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                       "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
                       "stop": ["\n```\n"], "include_stop_str_in_output": True, **extra,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def clear_stale(messages: list, turn: int, config: dict) -> None:
    """Client-side context editing: what `forget` does, on an API. Cutting a message
    changes the prefix, so the server re-reads everything after the earliest cut."""
    after, keep = config.get("clear_after", 0), 4 * config.get("forget_keep", 64)
    if not after:
        return
    for m in messages:
        if m["role"] == "assistant" and config.get("clear_replies") and m.get("turn") is None:
            m["turn"] = turn
        if m.get("turn") is not None and turn - m["turn"] >= after:
            del m["turn"]
            text = m["content"]
            if len(text) > 2 * keep:
                m["content"] = f"{text[:keep]}\n... (old output cut) ...\n{text[-keep:]}"


async def run_openai(task: Task, base_url: str, model: str, config: dict) -> dict:
    """The same loop as one trajectory over a chat API: every turn re-sends the conversation."""
    box = sandboxes(task)
    t0 = time.monotonic()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.issue}]
    prompt = completion = cached = turns = format_errors = forced_actions = rewinds = 0
    state, seen, runs = "base", set(), {}
    tool_seconds = 0.0
    patch = ""
    loop = asyncio.get_running_loop()
    try:
        last_prompt = 0
        # the swarm's limits: the sampled-token budget, and a context past which a branch dies
        while completion < config["token_budget"] and last_prompt + config["max_turn_tokens"] < config["max_context"]:
            # the inferlet's turn over an API: past `max_thought_tokens` with no command, or
            # ending with none, the command block is opened for the reply; a command already
            # run in this state is thrown away and sampled again
            for attempt in range(config["max_rewinds"] + 1):
                text, forced, budget = "", False, config["max_turn_tokens"]
                limit = config["max_thought_tokens"]
                while budget > 0:
                    r = await loop.run_in_executor(None, chat, base_url, model, messages, min(limit, budget),
                                                   config["temperature"], text)
                    usage = r.get("usage", {})
                    last_prompt = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                    prompt += usage.get("prompt_tokens", 0)
                    completion += usage.get("completion_tokens", 0)
                    budget -= usage.get("completion_tokens", 0)
                    cached += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                    text += r["choices"][0]["message"]["content"] or ""
                    if COMMAND.search(text):
                        break
                    if "```" in text:
                        if r["choices"][0].get("finish_reason") == "stop":
                            break
                    elif forced:
                        break
                    else:
                        forced = True
                        forced_actions += 1
                        text += "\n```bash\n"
                    limit = budget
                if (state, command(text)) not in seen or attempt == config["max_rewinds"]:
                    break
                rewinds += 1
            messages.append({"role": "assistant", "content": text})
            turns += 1
            cmd = command(text)
            if cmd is None:
                format_errors += 1
                messages.append({"role": "user", "content": FORMAT_ERROR})
                continue
            t = time.monotonic()
            seen.add((state, cmd))
            runs[(state, cmd)] = runs.get((state, cmd), 0) + 1
            if config.get("kill_loops") and runs[(state, cmd)] > 2:  # the third run of a command that changed nothing
                break
            res = await loop.run_in_executor(None, box.exec, 0, cmd)
            state = res["state"]
            tool_seconds += time.monotonic() - t
            if "patch" in res:
                patch = res["patch"]
                break
            clear_stale(messages, turns, config)
            out = res["out"]
            limit = config["max_output_chars"]
            if len(out) > limit:
                head, tail = out[: limit * 2 // 3], out[-limit // 3:]
                out = f"{head}\n... ({len(res['out']) - len(head) - len(tail)} characters cut) ...\n{tail}"
            messages.append({"role": "user", "content": f"<returncode>{res['rc']}</returncode>\n<output>\n{out}</output>",
                             "turn": turns})
        if config.get("dump"):
            Path(config["dump"]).mkdir(parents=True, exist_ok=True)
            Path(config["dump"], f"{task.id}.json").write_text(json.dumps(messages))
        return {
            "patch": patch, "solved": await loop.run_in_executor(None, box.evaluate, task, patch),
            "decoded_tokens": completion,
            "prompt_tokens": prompt, "cached_prompt_tokens": cached,
            "prefilled_tokens": prompt - cached, "turns": turns, "format_errors": format_errors,
            "forced_actions": forced_actions, "command_rewinds": rewinds,
            "tool_seconds": round(tool_seconds, 2), "wall_seconds": round(time.monotonic() - t0, 2),
        }
    finally:
        box.close()


DEFAULTS = {"temperature": 0.7, "max_turn_tokens": 1536, "max_output_chars": 6000, "token_budget": 60000,
            "max_context": 48000, "max_thought_tokens": 384, "max_rewinds": 2}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["swarm", "linear", "lean", "forget", "openai"])
    ap.add_argument("--program", default="main.py", help="the inferlet source to run")
    ap.add_argument("--uri", default="ws://127.0.0.1:8080")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (all toy tasks if empty)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--config", default="{}", help="JSON overrides for the swarm's settings")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--swe", type=int, default=0, help="run this many SWE-bench instances from /swe instead of the toy tasks")
    ap.add_argument("--parallel", type=int, default=1, help="tasks in flight at once")
    args = ap.parse_args()

    tasks = swe_tasks(args.swe) if args.swe else [t for t in TOY if not args.tasks or t.id in args.tasks.split(",")]
    if args.tasks and args.swe:
        tasks = [t for t in tasks if t.id in args.tasks.split(",")]
    config = {**DEFAULTS, **json.loads(args.config)}
    if args.mode in ("linear", "lean"):
        config.update(width=1, judge=False, dedup=False, memo=False)
    if args.mode == "linear":
        config.update(kill_loops=False, note_repeats=False)
    if args.mode == "forget":
        # the API arm's loop has neither rule
        config.update(width=1, judge=False, dedup=False, memo=False, kill_loops=False, note_repeats=False)
        config.setdefault("forget_after", 4)
    if args.mode == "lean":  # one agent, plus the three token savers
        config.update(verify_stop=True, no_rewrite=True, max_thought_tokens=192)
    global PROGRAM
    PROGRAM = HERE / args.program
    done = set()
    if Path(args.out).exists():  # resume: skip what this mode already finished
        for line in open(args.out):
            r = json.loads(line)
            if r["mode"] == args.mode and r.get("config") == json.loads(args.config):
                done.add((r["task"], r["rep"]))
    gate = asyncio.Semaphore(args.parallel)
    out = open(args.out, "a")

    async def one(task, rep: int) -> None:
        if (task.id, rep) in done:
            return
        async with gate:
            try:
                if args.mode == "openai":
                    r = await run_openai(task, args.base_url, args.model, config)
                else:
                    r = await run_pie(task, args.uri, config, args.quiet)
            except Exception as e:  # one task's failure is recorded, not fatal
                r = {"error": repr(e)[:500], "solved": False}
        r.update(task=task.id, mode=args.mode, rep=rep, config=json.loads(args.config))
        out.write(json.dumps(r) + "\n")
        out.flush()
        shown = {k: r.get(k) for k in ("solved", "decoded_tokens", "prefilled_tokens", "wall_seconds",
                                        "branches", "rewinds", "error")}
        print(f"{args.mode} {task.id} rep{rep}: {shown}", flush=True)

    await asyncio.gather(*[one(t, rep) for rep in range(args.repeat) for t in tasks])


if __name__ == "__main__":
    asyncio.run(main())

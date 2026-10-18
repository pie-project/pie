#!/usr/bin/env python3
"""Agent-trace replay against any OpenAI-compatible server.

Request-level benches send independent prompts. Agents do not: a program
prefills again each time a tool answers, its context grows turn by turn,
programs of one kind share a long system prompt, and a search forks one
context into many. This replays such programs, closed loop (each next
prompt is the conversation so far, with what the server actually said),
against `/v1/completions`, so pie and a baseline see the same workload.

    replay.py gen --out trace.jsonl --rate 0.5 --duration 120 --seed 1
    replay.py run --trace trace.jsonl --url http://127.0.0.1:8080 --model default --out pie.json

Program kinds:
  react  a tool-using loop: shared system prompt + task, then turns of
         (generate, wait on a tool, append the tool's output)
  tree   a search: a shared context forks into `branch` children per
         level; siblings run at once and share everything above them
  chat   multi-turn chat: user turns separated by think time

Every output is `max_tokens` long (`ignore_eos`), sampled greedily, so
servers do equal decode work. Text is pseudo-words from a seeded
vocabulary; token counts are what the server reports.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------- traces

_SYLLABLES = ["ka", "lo", "mi", "ne", "ru", "sa", "to", "vi", "ze", "po",
              "da", "fe", "gu", "hi", "jo", "qu", "we", "xi", "yo", "bu"]


def words(rng: random.Random, n: int) -> str:
    out = []
    for i in range(n):
        w = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(1, 3)))
        out.append(w + ("." if i % 17 == 16 else ""))
    return " ".join(out)


def gen(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    # Shared system prompts: a few tool suites, one per agent family.
    suites = [f"[suite {s}] You are an agent with tools. " + words(random.Random(f"suite{s}"), args.system_words)
              for s in range(args.suites)]
    t, n = 0.0, 0
    kinds = [k for k, w in (("react", args.mix_react), ("tree", args.mix_tree), ("chat", args.mix_chat)) for _ in range(w)]
    with open(args.out, "w") as f:
        while True:
            t += rng.expovariate(args.rate)
            if t > args.duration:
                break
            kind = rng.choice(kinds)
            p: dict = {"id": n, "kind": kind, "arrival": round(t, 3)}
            if kind == "react":
                p["root"] = suites[rng.randrange(len(suites))] + "\nTask: " + words(rng, rng.randint(150, 500))
                p["steps"] = [{
                    "gen": rng.choice([64, 128, 192, 256]),
                    "wait": round(min(rng.lognormvariate(0.3, 0.8), 15.0), 3),
                    "tool": "\nObservation: " + words(rng, rng.randint(60, 500)) + "\nThought:",
                } for _ in range(rng.randint(3, 8))]
            elif kind == "tree":
                p["root"] = suites[rng.randrange(len(suites))] + "\nProblem: " + words(rng, rng.randint(300, 900))
                p["branch"] = [rng.randint(2, 4), rng.randint(2, 3)]
                p["gen"] = rng.choice([64, 128])
                p["hint_seed"] = rng.randrange(1 << 30)
            else:
                p["root"] = "A conversation with a helpful assistant.\n"
                p["steps"] = [{
                    "user": "\nUser: " + words(rng, rng.randint(20, 200)) + "\nAssistant:",
                    "gen": rng.choice([96, 160, 256]),
                    "wait": round(rng.uniform(2.0, 10.0), 3),
                } for _ in range(rng.randint(2, 6))]
            f.write(json.dumps(p) + "\n")
            n += 1
    print(f"{n} programs over {args.duration}s -> {args.out}", file=sys.stderr)


# ------------------------------------------------------------------- run

@dataclass
class Req:
    program: int
    kind: str
    sent: float
    ttft: float | None = None
    e2e: float | None = None
    gaps: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    retries: int = 0


@dataclass
class Prog:
    id: int
    kind: str
    arrival: float
    started: float = 0.0
    done: float | None = None
    waited: float = 0.0
    failed: bool = False


class Runner:
    def __init__(self, args: argparse.Namespace):
        import httpx
        self.args = args
        self.client = httpx.AsyncClient(
            base_url=args.url,
            timeout=httpx.Timeout(args.timeout),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=256),
        )
        self.reqs: list[Req] = []
        self.progs: list[Prog] = []
        self.t0 = 0.0

    async def complete(self, prog: Prog, prompt: str, n: int) -> str:
        r = Req(program=prog.id, kind=prog.kind, sent=time.perf_counter())
        self.reqs.append(r)
        # A client that retries sends the request again whole, as an agent
        # framework does: what it was streamed is thrown away, and the
        # request's times run from the first send.
        while True:
            try:
                return await self.once(prog, r, prompt, n)
            except Exception as e:  # noqa: BLE001
                if r.retries >= self.args.retries:
                    r.error = f"{type(e).__name__}: {e}"
                    prog.failed = True
                    raise
                r.retries += 1
                r.ttft, r.gaps = None, []
                r.prompt_tokens = r.cached_tokens = r.completion_tokens = 0
                await asyncio.sleep(0.5)

    async def once(self, prog: Prog, r: Req, prompt: str, n: int) -> str:
        body = {
            "model": self.args.model, "prompt": prompt, "max_tokens": n,
            "temperature": 0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True},
        }
        pieces, last = [], None
        try:
            async with self.client.stream("POST", "/v1/completions", json=body) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    d = json.loads(data)
                    if d.get("usage"):
                        u = d["usage"]
                        r.prompt_tokens = u.get("prompt_tokens") or 0
                        r.completion_tokens = u.get("completion_tokens") or 0
                        r.cached_tokens = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
                    for c in d.get("choices") or []:
                        text = c.get("text") or ""
                        if not text:
                            continue
                        now = time.perf_counter()
                        if r.ttft is None:
                            r.ttft = now - r.sent
                        elif last is not None:
                            r.gaps.append(now - last)
                        last = now
                        pieces.append(text)
            r.e2e = time.perf_counter() - r.sent
        except Exception:
            raise
        # A stream cut without an error ends with no usage, or short.
        if r.completion_tokens < n:
            raise RuntimeError(f"short answer: {r.completion_tokens} of {n} tokens")
        return "".join(pieces)

    async def wait(self, prog: Prog, s: float) -> None:
        s *= self.args.wait_scale
        prog.waited += s
        await asyncio.sleep(s)

    async def program(self, p: dict) -> None:
        delay = self.t0 + p["arrival"] / self.args.speed - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        prog = Prog(p["id"], p["kind"], p["arrival"], started=time.perf_counter())
        self.progs.append(prog)
        try:
            if p["kind"] == "react":
                ctx = p["root"]
                for step in p["steps"]:
                    ctx += await self.complete(prog, ctx, step["gen"])
                    await self.wait(prog, step["wait"])
                    ctx += step["tool"]
            elif p["kind"] == "tree":
                rng = random.Random(p["hint_seed"])

                async def node(ctx: str, depth: int) -> None:
                    if depth == len(p["branch"]):
                        return
                    hints = [f"\nCandidate {depth}.{i}: " + words(rng, 30) + "\n" for i in range(p["branch"][depth])]

                    async def child(h: str) -> None:
                        c = ctx + h
                        c += await self.complete(prog, c, p["gen"])
                        await node(c, depth + 1)
                    await asyncio.gather(*(child(h) for h in hints))
                await node(p["root"], 0)
            else:
                ctx = p["root"]
                for step in p["steps"]:
                    ctx += step["user"]
                    ctx += await self.complete(prog, ctx, step["gen"])
                    await self.wait(prog, step["wait"])
            prog.done = time.perf_counter()
        except Exception:  # noqa: BLE001
            prog.failed = True

    async def run(self, trace: list[dict]) -> None:
        self.t0 = time.perf_counter()
        await asyncio.gather(*(self.program(p) for p in trace))
        await self.client.aclose()


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def summarize(runner: Runner, args: argparse.Namespace, wall: float) -> dict:
    ok = [r for r in runner.reqs if r.error is None and r.e2e is not None]
    ttft = [r.ttft for r in ok if r.ttft is not None]
    gaps = [g for r in ok for g in r.gaps]
    tpot = [statistics.mean(r.gaps) for r in ok if r.gaps]
    done = [p for p in runner.progs if p.done is not None and not p.failed]
    # A program's time on the server: its span less the tool/user waits.
    service = [p.done - p.started - p.waited for p in done]
    span = [p.done - p.started for p in done]
    prompt = sum(r.prompt_tokens for r in ok)
    cached = sum(r.cached_tokens for r in ok)
    out_tok = sum(r.completion_tokens for r in ok)
    by_prog: dict[int, list[Req]] = {}
    for r in runner.reqs:
        by_prog.setdefault(r.program, []).append(r)
    good = sum(
        1 for p in done
        if all(r.error is None and r.ttft is not None and r.ttft <= args.slo_ttft
               and (not r.gaps or statistics.mean(r.gaps) <= args.slo_tpot)
               for r in by_prog.get(p.id, []))
    )
    kinds = sorted({p.kind for p in runner.progs})
    s = {
        "label": args.label, "url": args.url, "wall_s": round(wall, 2),
        "programs": len(runner.progs), "programs_done": len(done),
        "programs_failed": sum(1 for p in runner.progs if p.failed),
        "requests": len(runner.reqs), "requests_failed": len(runner.reqs) - len(ok),
        "requests_retried": sum(1 for r in runner.reqs if r.retries),
        "prompt_tokens": prompt, "cached_tokens": cached,
        "cache_hit": round(cached / prompt, 4) if prompt else None,
        "output_tokens": out_tok, "output_tok_per_s": round(out_tok / wall, 1),
        "ttft_ms": {q: round(1000 * v, 1) for q in (50, 90, 99) if (v := pct(ttft, q)) is not None},
        "tpot_ms": {q: round(1000 * v, 2) for q in (50, 90, 99) if (v := pct(tpot, q)) is not None},
        "itl_p99_ms": round(1000 * pct(gaps, 99), 1) if gaps else None,
        "program_service_s": {q: round(v, 2) for q in (50, 90, 99) if (v := pct(service, q)) is not None},
        "program_span_s": {q: round(v, 2) for q in (50, 90, 99) if (v := pct(span, q)) is not None},
        "goodput": {"slo_ttft_s": args.slo_ttft, "slo_tpot_s": args.slo_tpot,
                    "programs": good, "per_s": round(good / wall, 3)},
        "by_kind": {},
    }
    for k in kinds:
        kr = [r for r in ok if r.kind == k]
        kp = [p.done - p.started - p.waited for p in done if p.kind == k]
        kprompt = sum(r.prompt_tokens for r in kr)
        s["by_kind"][k] = {
            "programs": sum(1 for p in runner.progs if p.kind == k),
            "ttft_p50_ms": round(1000 * pct([r.ttft for r in kr if r.ttft], 50), 1) if kr else None,
            "ttft_p99_ms": round(1000 * pct([r.ttft for r in kr if r.ttft], 99), 1) if kr else None,
            "service_p50_s": round(pct(kp, 50), 2) if kp else None,
            "cache_hit": round(sum(r.cached_tokens for r in kr) / kprompt, 4) if kprompt else None,
        }
    errors = sorted({r.error for r in runner.reqs if r.error})
    if errors:
        s["errors"] = errors[:5]
    return s


def run(args: argparse.Namespace) -> None:
    trace = [json.loads(l) for l in open(args.trace)]
    if args.limit:
        trace = trace[: args.limit]
    runner = Runner(args)
    t = time.perf_counter()
    asyncio.run(runner.run(trace))
    s = summarize(runner, args, time.perf_counter() - t)
    print(json.dumps(s, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": s, "requests": [r.__dict__ for r in runner.reqs]}, f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--rate", type=float, default=0.5, help="programs per second (Poisson)")
    g.add_argument("--duration", type=float, default=120)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--suites", type=int, default=4, help="distinct shared system prompts")
    g.add_argument("--system-words", type=int, default=1200)
    g.add_argument("--mix-react", type=int, default=3)
    g.add_argument("--mix-tree", type=int, default=1)
    g.add_argument("--mix-chat", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--trace", required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--model", default="default")
    r.add_argument("--label", default="")
    r.add_argument("--out")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--speed", type=float, default=1.0, help="arrival speed-up")
    r.add_argument("--wait-scale", type=float, default=1.0, help="scales tool/user waits")
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--retries", type=int, default=0, help="times a failed request is sent again whole")
    r.add_argument("--slo-ttft", type=float, default=2.0)
    r.add_argument("--slo-tpot", type=float, default=0.1)
    args = ap.parse_args()
    gen(args) if args.cmd == "gen" else run(args)


if __name__ == "__main__":
    main()

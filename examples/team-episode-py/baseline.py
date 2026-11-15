#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from episode import STOP, Episode  # noqa: E402

DEFAULT_BUDGET = {"aggregate": 2048, "maze": 8192}
EPISODE_KWARGS = ("budget_tokens", "chunk_tokens", "depth", "lam")
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
VLLM_EXTRAS = {"include_stop_str_in_output": True, "skip_special_tokens": False, "return_token_ids": True,
               "return_tokens_as_token_ids": True}
TOOL_CLOSE_ID = 151658  # Qwen3 `</tool_call>`: vLLM reports a stop token by id in stop_reason
TOOL_OPEN, TOOL_CLOSE, END_TURN = "<tool_call>", "</tool_call>", "<|im_end|>"
EOS_TEXTS = ("<|endoftext|>",)


class Server:

    def __init__(self, base_url: str, model: str, max_connections: int = 512):
        self.url = base_url.rstrip("/") + "/completions"
        self.model = model
        self.extras_ok: bool | None = None
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=30.0),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        )

    async def close(self):
        await self.client.aclose()

    async def complete(self, prompt: str, max_tokens: int, *, seed: int, temperature: float, top_p: float) -> dict:
        body = {
            "model": self.model, "prompt": prompt, "max_tokens": max_tokens, "stop": list(STOP),
            "temperature": temperature, "top_p": top_p, "seed": seed, "logprobs": 1,
        }
        extras = self.extras_ok is not False
        delay = 0.5
        for attempt in range(6):
            req = dict(body, **VLLM_EXTRAS) if extras else body
            try:
                r = await self.client.post(self.url, json=req)
            except httpx.TransportError as e:
                if attempt == 5:
                    raise RuntimeError(f"transport error after retries: {e!r}") from e
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 200:
                if extras and self.extras_ok is None:
                    self.extras_ok = True
                return r.json()
            if r.status_code in (400, 422) and extras and self.extras_ok is None:
                extras = False
                self.extras_ok = False
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        raise RuntimeError("unreachable")


@dataclass
class Chunk:
    text: str                      # new text, ending in the stop string when one ended it
    reason: str                    # "tool_call" | "end_turn" | "length"
    n_tokens: int                  # tokens generated and kept (the stop token among them)
    token_ids: list[int] | None
    logprobs: list[float] | None
    pieces: list[str] | None       # the server's per-token text for the kept tokens, when it gave it
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int         # what the server generated, kept or not
    wasted: int = 0                # generated past the first stop string (the server did not stop there)
    lossy: int = 0                 # tokens whose text the server left empty (part of a split character)
    rebuilt: bool = False          # text rebuilt from logprobs tokens, not the server's text


def _first_stop(pieces: list[str]) -> tuple[int, int, str] | None:
    acc = ""
    longest = max(len(s) for s in STOP)
    for i, piece in enumerate(pieces):
        start = max(0, len(acc) - longest)
        acc += piece
        best = None
        for s in STOP:
            p = acc.find(s, start)
            if p >= 0 and (best is None or p < best[0]):
                best = (p, s)
        if best is not None:
            return i, best[0] + len(best[1]), best[1]
    return None


def _ends_in_open_call(text: str) -> bool:
    return text.rfind(TOOL_OPEN) > text.rfind(TOOL_CLOSE)


def parse_chunk(resp: dict) -> Chunk:
    choice = resp["choices"][0]
    text = choice.get("text") or ""
    finish = choice.get("finish_reason")
    stop_reason = choice.get("stop_reason")
    usage = resp.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    completion = int(usage.get("completion_tokens") or 0)
    lp = choice.get("logprobs") or {}
    toks = lp.get("tokens")
    vals = lp.get("token_logprobs")

    ids: list[int] | None = None
    pieces: list[str] | None = None
    if isinstance(choice.get("token_ids"), list) and choice["token_ids"]:
        ids = [int(i) for i in choice["token_ids"]]
    if toks:
        if all(isinstance(t, str) and t.startswith("token_id:") for t in toks):
            ids = [int(t.split(":", 1)[1]) for t in toks]
        else:
            pieces = [t if isinstance(t, str) else "" for t in toks]
    n_gen = len(toks) if toks else completion
    logprobs = [float(v) if v is not None else None for v in vals] if vals else None

    ch = Chunk(text=text, reason="length", n_tokens=n_gen, token_ids=ids, logprobs=logprobs, pieces=pieces,
               prompt_tokens=prompt_tokens, cached_tokens=cached, completion_tokens=max(completion, n_gen))

    if pieces is not None:
        joined = "".join(pieces)
        hit = _first_stop(pieces)
        swallowed = TOOL_OPEN in joined and TOOL_OPEN not in text
        ran_past = hit is not None and hit[0] < len(pieces) - 1
        if swallowed or ran_past:
            ch.rebuilt = True
            if hit is not None:
                i, end, stop = hit
                ch.text = "".join(pieces[: i + 1])[:end]
                ch.n_tokens = i + 1
                ch.wasted = len(pieces) - (i + 1)
                ch.reason = "tool_call" if stop == TOOL_CLOSE else "end_turn"
            else:
                ch.text = joined
                ch.n_tokens = len(pieces)
                ch.reason = "end_turn" if finish == "stop" else "length"
                if ch.reason == "end_turn":
                    ch.text = _close_turn(ch.text)
            ch.lossy = sum(1 for p in pieces[: ch.n_tokens] if p == "")
            ch.pieces = pieces[: ch.n_tokens]
            if ch.logprobs is not None:
                ch.logprobs = ch.logprobs[: ch.n_tokens]
            return ch

    if finish == "length":
        ch.reason = "length"
        return ch
    stop = None
    if isinstance(stop_reason, str) and stop_reason in STOP:
        stop = stop_reason
    elif isinstance(stop_reason, int) and not isinstance(stop_reason, bool):  # vLLM: a stop token's id
        stop = TOOL_CLOSE if stop_reason == TOOL_CLOSE_ID else END_TURN
    elif any(text.endswith(s) for s in STOP):
        stop = next(s for s in STOP if text.endswith(s))
    elif pieces:
        last = pieces[-1]
        if last in STOP:
            stop = last
        elif last in EOS_TEXTS:
            stop = END_TURN
    if stop is None:
        stop = TOOL_CLOSE if _ends_in_open_call(text) else END_TURN
    if stop == TOOL_CLOSE:
        ch.reason = "tool_call"
        if not text.endswith(TOOL_CLOSE):
            ch.text = text + TOOL_CLOSE
    else:
        ch.reason = "end_turn"
        ch.text = _close_turn(text)
    return ch


def _close_turn(text: str) -> str:
    for eos in EOS_TEXTS:
        if text.endswith(eos):
            text = text[: -len(eos)]
    return text if text.endswith(END_TURN) else text + END_TURN


class Tokens:

    def __init__(self, path: str):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(path)
        self._by_text: dict[str, list[int]] | None = None

    @classmethod
    def load(cls, path: str | None = None) -> "Tokens | None":
        path = path or os.environ.get("TEAM_TOKENIZER")
        if not path:
            hub = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")
            found = sorted(glob.glob(os.path.join(hub, "models--Qwen--Qwen3-0.6B", "snapshots", "*", "tokenizer.json")))
            path = found[-1] if found else None
        if not path or not os.path.exists(path):
            return None
        try:
            return cls(path)
        except Exception:
            return None

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def _piece(self, i: int) -> str:
        t = self.tok.decode([i], skip_special_tokens=False)
        return "" if "\ufffd" in t else t  # as pie reports a token that splits a character

    def generated(self, text: str, pieces: list[str] | None) -> list[int] | None:
        if pieces is None:
            return None
        ids = self.encode(text)
        if len(ids) == len(pieces) and all(self._piece(i) == p for i, p in zip(ids, pieces)):
            return ids
        if self._by_text is None:
            self._by_text = {}
            for i in range(self.tok.get_vocab_size()):
                self._by_text.setdefault(self._piece(i), []).append(i)
        out = []
        for p in pieces:
            cand = self._by_text.get(p) if p else None
            if not cand or len(cand) != 1:
                return None
            out.append(cand[0])
        return out


def flat_view(traj: list[dict]) -> dict:
    if any(seg.get("tokens") is None for seg in traj):
        return {"tokens": None, "mask": None, "logprobs": None}
    tokens, mask, lps = [], [], []
    for seg in traj:
        k = len(seg["tokens"])
        tokens += seg["tokens"]
        mask += [seg["mask"]] * k
        lps += list(seg["logprobs"]) if seg["mask"] and seg.get("logprobs") is not None else [None] * k
    return {"tokens": tokens, "mask": mask, "logprobs": lps}


def make_episode(spec: dict) -> Episode:
    task = spec["task"]
    kw = {k: spec[k] for k in EPISODE_KWARGS if spec.get(k) is not None}
    kw.setdefault("budget_tokens", DEFAULT_BUDGET.get(task, 2048))
    return Episode(task, int(spec.get("n_agents", 4)), int(spec.get("seed", 0)), **kw)


def request_seed(ep_seed: int, agent: int, chunk: int) -> int:
    return (ep_seed * 1_000_003 + agent * 10_007 + chunk * 101 + 17) % (2**31 - 1)


@dataclass
class AgentState:
    ctx: str
    traj: list = field(default_factory=list)
    used: int = 0
    chunks: int = 0
    last_prompt: int | None = None
    last_completion: int = 0
    stuck: bool = False             # budget spent (driver side) or the server failed it


async def run_episode(server: Server, spec: dict, tokens: Tokens | None = None) -> dict:
    ep = make_episode(spec)
    n = ep.n_agents
    budget = int(spec.get("budget_tokens") or DEFAULT_BUDGET.get(spec["task"], 2048))
    chunk_tokens = int(spec.get("chunk_tokens") or 1024)
    temperature = float(spec.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(spec.get("top_p", DEFAULT_TOP_P))
    seed = int(spec.get("seed", 0))
    max_wall_s = float(spec.get("max_wall_s") or 600.0)
    shared = ep.shared_prompt()
    agents = []
    for a in range(n):
        ctx = shared + ep.private_prompt(a)
        agents.append(AgentState(ctx=ctx, traj=[_seg("prompt", ctx, tokens)]))
    t = {"wall_s": 0.0, "gen_tokens": 0, "injected_tokens": 0, "forward_passes": 0, "requests": 0,
         "prompt_tokens_sent": 0, "resent_prompt_tokens": 0, "cached_prompt_tokens": 0,
         "server_completion_tokens": 0, "wasted_tokens": 0, "lossy_tokens": 0, "rebuilt_chunks": 0, "client_token_ids": 0, "errors": 0}
    errors: list[str] = []
    inflight: dict[asyncio.Task, int] = {}
    busy: set[int] = set()

    def launch(a: int):
        st = agents[a]
        left = ep.max_new_tokens(a) if hasattr(ep, "max_new_tokens") else min(chunk_tokens, budget - st.used)
        if left <= 0:
            st.stuck = True
            return
        coro = server.complete(st.ctx, left, seed=request_seed(seed, a, st.chunks),
                               temperature=temperature, top_p=top_p)
        st.chunks += 1
        inflight[asyncio.create_task(coro)] = a
        busy.add(a)

    def schedule():
        for a in ep.runnable():
            if a not in busy and not agents[a].stuck:
                launch(a)

    def inject(a: int, text: str):
        agents[a].ctx += text
        seg = _seg("inject", text, tokens)
        agents[a].traj.append(seg)
        if seg["tokens"] is not None:
            t["injected_tokens"] += len(seg["tokens"])

    t0 = time.perf_counter()
    timed_out = False
    schedule()
    while inflight and not ep.done:
        left_s = max_wall_s - (time.perf_counter() - t0)
        if left_s <= 0:
            timed_out = True
            break
        finished, _ = await asyncio.wait(list(inflight), timeout=left_s, return_when=asyncio.FIRST_COMPLETED)
        for task in sorted(finished, key=lambda tk: inflight[tk]):
            a = inflight.pop(task)
            busy.discard(a)
            st = agents[a]
            try:
                ch = parse_chunk(task.result())
            except Exception as e:  # the server failed this agent: stop driving it
                st.stuck = True
                t["errors"] += 1
                errors.append(f"agent {a}: {e}"[:300])
                continue
            t["requests"] += 1
            t["prompt_tokens_sent"] += ch.prompt_tokens
            t["cached_prompt_tokens"] += ch.cached_tokens
            t["server_completion_tokens"] += ch.completion_tokens
            if st.last_prompt is not None:
                carried = st.last_prompt + st.last_completion
                t["resent_prompt_tokens"] += min(ch.prompt_tokens, carried)
                if tokens is None:
                    t["injected_tokens"] += max(0, ch.prompt_tokens - carried)
            st.last_prompt, st.last_completion = ch.prompt_tokens, ch.n_tokens
            st.used += ch.n_tokens
            t["gen_tokens"] += ch.n_tokens
            t["wasted_tokens"] += ch.wasted
            t["lossy_tokens"] += ch.lossy
            t["rebuilt_chunks"] += int(ch.rebuilt)
            st.ctx += ch.text
            ids = ch.token_ids[: ch.n_tokens] if ch.token_ids is not None else None
            if ids is None and tokens is not None:
                ids = tokens.generated(ch.text, ch.pieces)
                t["client_token_ids"] += int(ids is not None)
            st.traj.append({"kind": "gen", "text": ch.text, "tokens": ids, "logprobs": ch.logprobs, "mask": 1})
            out = ep.step(a, ch.text, ch.reason, ch.n_tokens)
            if out:  # None: idle/done; "": keep going, nothing to append
                inject(a, out)
        if ep.done:
            break
        for b in range(n):
            if ep.status(b) == "idle":
                w = ep.wake(b)
                if w:
                    inject(b, w)
        schedule()
    for task in inflight:  # the episode ended under them: their tokens are not the team's
        task.cancel()
    if inflight:
        await asyncio.gather(*inflight, return_exceptions=True)
    t["wall_s"] = round(time.perf_counter() - t0, 4)

    notes = [
        "http driver: forward_passes unknown (0); requests counted instead",
        ("injected_tokens counted with the client tokenizer" if tokens is not None else
         "injected_tokens estimated from the growth of usage.prompt_tokens between an agent's requests"),
        "resent_prompt_tokens = prompt tokens of each request already in that agent's previous request "
        "(its prompt plus what it generated)",
        "logprobs as the server reports them: pie's /v1 and vLLM's default are raw (before temperature/top-p); "
        "vLLM matches SPEC only with --logprobs-mode processed_logprobs",
    ]
    missing = sum(1 for st in agents for seg in st.traj if seg["tokens"] is None)
    if t["client_token_ids"]:
        notes.append(f"{t['client_token_ids']} gen chunk(s) took token ids from the client tokenizer (the server "
                     f"returned text only), each checked against the server's per-token text")
    if missing:
        notes.append(f"{missing} segment(s) without token ids (tokens null): the server returned none and the "
                     f"client tokenizer was {'unable to match them' if tokens is not None else 'not available'}")
    if t["rebuilt_chunks"]:
        notes.append(f"{t['rebuilt_chunks']} chunk(s) rebuilt from logprobs tokens: the server swallowed tool-call "
                     f"markup or ran past a stop string ({t['wasted_tokens']} tokens past the stop dropped, "
                     f"{t['lossy_tokens']} split-character tokens lost)")
    t["timeout"] = timed_out
    if timed_out:
        notes.append(f"stopped at max_wall_s={max_wall_s}")
    elif not ep.done:
        notes.append("episode stopped without Episode.done: no agent left to drive")
    if errors:
        notes.append("errors: " + " | ".join(errors[:5]))
    t["notes"] = "; ".join(notes)
    t["temperature"], t["top_p"] = temperature, top_p
    trajs = [st.traj for st in agents]
    out = {"driver": "http", "result": ep.result(), "timing": t, "trajectories": trajs, "input": dict(spec)}
    flats = [flat_view(tr) for tr in trajs]
    if all(f["tokens"] is not None for f in flats):  # else left out: there are no ids to flatten
        out["flat"] = flats
    return out


def _seg(kind: str, text: str, tokens: Tokens | None = None) -> dict:
    ids = tokens.encode(text) if tokens is not None else None
    return {"kind": kind, "text": text, "tokens": ids, "logprobs": None, "mask": 0}


async def run_http(base_url: str, model: str, episodes: list[dict], concurrency: int,
                   tokenizer: str | None = None) -> list[dict]:
    tokens = Tokens.load(tokenizer)
    n_max = max((int(e.get("n_agents", 4)) for e in episodes), default=4)
    server = Server(base_url, model, max_connections=max(16, concurrency * n_max * 2))
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(spec):
        async with sem:
            try:
                return await run_episode(server, spec, tokens)
            except Exception as e:  # one failed episode must not sink the batch (as run.py)
                return {"driver": "http", "error": f"{type(e).__name__}: {e}", "input": dict(spec), "timing": {}}

    try:
        return list(await asyncio.gather(*(one(e) for e in episodes)))
    finally:
        await server.close()


def episode_specs(task: str, n_agents: int, episodes: int, seed: int, *, budget: int | None = None,
                  chunk: int = 1024, depth: int = 2, lam: float | None = None,
                  temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P,
                  max_wall_s: float = 600.0) -> list[dict]:
    budget = budget or DEFAULT_BUDGET.get(task, 2048)
    specs = []
    for i in range(episodes):
        e = {"task": task, "n_agents": n_agents, "seed": seed + i, "budget_tokens": budget, "chunk_tokens": chunk,
             "depth": depth, "temperature": temperature, "top_p": top_p, "max_wall_s": max_wall_s}
        if lam is not None:
            e["lam"] = lam
        specs.append(e)
    return specs


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results: list[dict], wall_total: float | None = None) -> dict:
    ok = [r for r in results if "error" not in r]
    tm = [r["timing"] for r in ok]
    s = {
        "episodes": len(results),
        "errors": len(results) - len(ok),
        "timeouts": sum(1 for x in tm if x.get("timeout")),
        "success_rate": _mean(1.0 if r["result"].get("success") else 0.0 for r in ok),
        "mean_reward": _mean(float(r["result"].get("reward") or 0.0) for r in ok),
        "gen_tokens_per_episode": _mean(x.get("gen_tokens", 0) for x in tm),
        "mean_wall_s": _mean(x.get("wall_s", 0.0) for x in tm),
        "requests_per_episode": _mean(x.get("requests", 0) for x in tm),
        "forward_passes_per_episode": _mean(x.get("forward_passes", 0) or 0 for x in tm),
        "prompt_tokens_sent_per_episode": _mean(x.get("prompt_tokens_sent", 0) or 0 for x in tm),
        "resent_prompt_tokens_per_episode": _mean(x.get("resent_prompt_tokens", 0) or 0 for x in tm),
        "cached_prompt_tokens_per_episode": _mean(x.get("cached_prompt_tokens", 0) or 0 for x in tm),
    }
    if wall_total:
        finished = sum(1 for x in tm if not x.get("timeout"))
        s["wall_total_s"] = wall_total
        s["finished"] = finished
        s["episodes_per_hour"] = finished / wall_total * 3600
        s["gen_tokens_per_s"] = sum(x.get("gen_tokens", 0) for x in tm) / wall_total
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--task", default="aggregate", choices=["aggregate", "maze"])
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="episode i runs seed + i")
    ap.add_argument("--concurrency", type=int, default=8, help="episodes in flight at once")
    ap.add_argument("--budget-tokens", "--budget", type=int, default=None,
                    help="per-agent token budget (default 2048 aggregate, 8192 maze)")
    ap.add_argument("--chunk-tokens", "--chunk", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--max-wall-s", type=float, default=600.0)
    ap.add_argument("--out", default=None, help="JSONL, one SPEC output object per episode")
    args = ap.parse_args(argv)

    specs = episode_specs(args.task, args.n_agents, args.episodes, args.seed, budget=args.budget_tokens,
                          chunk=args.chunk_tokens, depth=args.depth, lam=args.lam, temperature=args.temperature,
                          top_p=args.top_p, max_wall_s=args.max_wall_s)
    t0 = time.perf_counter()
    results = asyncio.run(run_http(args.base_url, args.model, specs, args.concurrency))
    wall = time.perf_counter() - t0
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    for r in results:
        if "error" in r:
            print(f"seed {r['input'].get('seed')}: ERROR {r['error']}", file=sys.stderr)
            continue
        res, tm = r["result"], r["timing"]
        print(f"seed {res.get('seed')}: success={res.get('success')} reward={float(res.get('reward') or 0):.3f} "
              f"end={res.get('end_reason')} tokens={res.get('tokens')} messages={res.get('messages')} "
              f"wall={tm['wall_s']:.1f}s requests={tm['requests']} prompt_sent={tm['prompt_tokens_sent']} "
              f"resent={tm['resent_prompt_tokens']}" + (f" errors={tm['errors']}" if tm.get("errors") else ""))
    s = summarize(results, wall)
    print(f"{s['episodes']} episodes ({s['errors']} errors, {s['timeouts']} timeouts) in {wall:.1f}s "
          f"at concurrency {args.concurrency}")
    print(f"success rate {s['success_rate']:.3f}  mean reward {s['mean_reward']:.3f}")
    print(f"team episodes/hour {s.get('episodes_per_hour', 0):.1f}  generated tokens/s {s.get('gen_tokens_per_s', 0):.1f}"
          f"  ({s['gen_tokens_per_episode']:.0f} per episode)")
    print(f"mean wall/episode {s['mean_wall_s']:.2f}s  requests/episode {s['requests_per_episode']:.1f}  "
          f"prompt tokens sent/episode {s['prompt_tokens_sent_per_episode']:.0f} "
          f"(re-sent {s['resent_prompt_tokens_per_episode']:.0f}, server-cached {s['cached_prompt_tokens_per_episode']:.0f})")


if __name__ == "__main__":
    main()

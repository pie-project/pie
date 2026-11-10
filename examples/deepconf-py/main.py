"""DeepConf on pie: parallel reasoning that stops weak attempts while they think.

Deep Think with Confidence (Fu, Wang, Tian and Zhao, 2025) samples many
reasoning traces for one question and votes on their answers, but reads each
trace's confidence as it is written: a trace that loses confidence stops, and
generation stops once the votes agree.

Here every trace is a fork of the question's working set, so the prompt is read
once. Each step the GPU samples the next token and, beside it, the token's
confidence: minus the mean log-probability of its `k_conf` most likely
candidates. The program keeps each trace's mean confidence over its last
`window` tokens (its group confidence).

- warmup: `n_init` traces run to the end. Each one's lowest group confidence is
  its score, and the threshold is the `100 - eta` percentile of those scores.
- online: more traces start, `parallel` at a time, until `budget` have started
  or a `tau` share of the confidence-weighted vote agrees. A trace whose group
  confidence falls below the threshold stops on the spot and frees its KV.
- vote: of the finished traces, the top `eta` percent by score vote, each
  weighted by its score.

Without `eta` this is plain majority voting over `budget` traces (cons@budget).
"""

import asyncio
import json
import time

from inferlet import chat, model
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    cast,
    channel_capacity,
    dtype,
    gather,
    intrinsics,
    iota,
    log_softmax,
    nucleus_sample,
    reduce_argmax,
    reduce_sum,
    reshape,
    top_k,
)


def boxed(text: str) -> str | None:
    """The contents of the last \\boxed{...} in `text`, braces balanced."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    out = []
    while i < len(text):
        c = text[i]
        depth += c == "{"
        depth -= c == "}"
        if depth == 0:
            return "".join(out).strip()
        out.append(c)
        i += 1
    return None


def percentile(values: list[float], q: float) -> float:
    """numpy's default (linear) percentile."""
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class Trace:
    def __init__(self, tid: int) -> None:
        self.id = tid
        self.tokens: list[int] = []
        self.window_sum = 0.0
        self.window: list[float] = []  # the last `window` confidences, a ring
        self.head = 0
        self.total = 0.0
        self.lowest: float | None = None  # lowest full-window mean so far
        self.state = "running"  # finished, stopped or cancelled
        self.answer: str | None = None

    def score(self) -> float:
        """Lowest group confidence: the mean of the whole trace if it never filled a window."""
        if self.lowest is not None:
            return self.lowest
        return self.total / len(self.tokens) if self.tokens else 0.0


class DeepConf:
    def __init__(self, cfg: dict, prompt: list[int]) -> None:
        self.cfg = cfg
        self.prompt = prompt
        self.stops = set(cfg["stop_tokens"])
        self.threshold: float | None = None
        self.traces: list[Trace] = []
        self.done = False  # consensus reached: running traces are cancelled
        self.decoded = 0

    async def prefill(self) -> None:
        """Read the question once; every trace forks it."""
        self.root, self.root_pipe = WorkingSet(), Pipeline()
        fwd = ForwardPass()
        fwd.embed(self.prompt[:-1])
        fwd.state([self.root], StateGeometry.folding())
        ready = Channel([1], dtype.i32)

        @fwd.epilogue
        def _():
            ready.put(reshape(reduce_argmax(intrinsics.output(0, "logits")), [1]))

        self.root_pipe.submit(fwd)
        await ready.take_scalar()

    def forks(self, n: int) -> list:
        """n forks of the question; the root moves to a new pipeline so each fork can take its own."""
        children = [self.root.fork(self.root_pipe) for _ in range(n)]
        self.root_pipe.close()
        self.root_pipe = Pipeline()
        return children

    async def run_trace(self, t: Trace, ws, stop_early: bool) -> None:
        cfg = self.cfg
        k, window = cfg["k_conf"], cfg["window"]
        pipe = Pipeline()
        tok_in = Channel.from_([self.prompt[-1]], dtype.i32)
        tok_out = Channel([1], dtype.i32).capacity(channel_capacity())
        conf_out = Channel([1], dtype.f32).capacity(channel_capacity())
        rng = Channel.from_([cfg["seed"] + 7919 * t.id, 0], dtype.u32)
        temperature, top_p, top_n = cfg["temperature"], cfg["top_p"], cfg["top_k"]
        decode = ForwardPass()
        decode.embed(tok_in, Channel.from_([0, 1], dtype.u32))
        decode.state([ws], StateGeometry.folding())

        @decode.epilogue
        def _():
            logits = intrinsics.output(0, "logits")
            # the token's confidence, on the model's own distribution
            best, _ = top_k(log_softmax(logits), k)
            conf_out.put(reshape(reduce_sum(best) * (-1.0 / k), [1]))
            # the next token: top-k, then top-p, at the temperature
            values, ids = top_k(logits, top_n)
            r = rng.take()
            pick = nucleus_sample(values * (1.0 / temperature), top_p, r)
            token = reshape(cast(gather(ids, pick), dtype.i32), [1])
            tok_in.put(token)
            tok_out.put(token)
            rng.put(r + iota(2))

        async def on_token() -> bool:
            token = await tok_out.take_scalar()
            conf = await conf_out.take_scalar()
            self.decoded += 1
            if self.done:
                t.state = "cancelled"
                return False
            if token in self.stops:
                t.state = "finished"
                return False
            t.tokens.append(token)
            t.total += conf
            if len(t.window) < window:
                t.window.append(conf)
                t.window_sum += conf
            else:
                t.window_sum += conf - t.window[t.head]
                t.window[t.head] = conf
                t.head = (t.head + 1) % window
            if len(t.window) == window:
                mean = t.window_sum / window
                t.lowest = mean if t.lowest is None else min(t.lowest, mean)
                if stop_early and mean < self.threshold:
                    t.state = "stopped"
                    return False
            return True

        await pipe.run_ahead(decode, cfg["max_tokens"], on_token)
        if t.state == "running":
            t.state = "finished"  # out of tokens: it votes if it boxed an answer
        if t.state == "finished":
            t.answer = boxed(model.decode(t.tokens))

    def votes(self, filtered: bool) -> dict[str, float]:
        """Answer -> vote weight over finished traces with an answer."""
        done = [t for t in self.traces if t.state == "finished" and t.answer is not None]
        eta = self.cfg["eta"]
        if eta is None:
            out: dict[str, float] = {}
            for t in done:
                out[t.answer] = out.get(t.answer, 0) + 1
            return out
        if filtered and done:
            keep = max(1, round(len(done) * eta / 100))
            done = sorted(done, key=lambda t: t.score(), reverse=True)[:keep]
        out = {}
        for t in done:
            out[t.answer] = out.get(t.answer, 0.0) + t.score()
        return out

    def consensus(self) -> float:
        v = self.votes(filtered=False)
        total = sum(v.values())
        return max(v.values()) / total if total else 0.0

    async def run(self) -> dict:
        cfg = self.cfg
        t0 = time.monotonic()
        await self.prefill()
        budget, parallel = cfg["budget"], cfg["parallel"]
        online = cfg["eta"] is not None
        started = 0

        def new() -> Trace:
            nonlocal started
            t = Trace(started)
            started += 1
            self.traces.append(t)
            return t

        if online:
            # warmup: n_init whole traces set the stopping threshold
            warm = [new() for _ in range(min(cfg["n_init"], budget))]
            await asyncio.gather(*[self.run_trace(t, ws, False) for t, ws in zip(warm, self.forks(len(warm)))])
            scores = [t.score() for t in warm if t.tokens]
            self.threshold = percentile(scores, 100 - cfg["eta"]) if scores else 0.0
            if self.consensus() >= cfg["tau"]:
                self.done = True
        # the rest, `parallel` at a time; with eta, each stops on low confidence
        async def worker() -> None:
            while not self.done and started < budget:
                await self.run_trace(new(), self.forks(1)[0], online)
                if online and self.consensus() >= cfg["tau"]:
                    self.done = True

        await asyncio.gather(*[worker() for _ in range(parallel)])
        self.root_pipe.close()

        v = self.votes(filtered=True)
        answer = max(v, key=v.get) if v else None
        count = lambda s: sum(t.state == s for t in self.traces)
        return {
            "answer": answer,
            "votes": v,
            "threshold": self.threshold,
            "traces": len(self.traces),
            "finished": count("finished"),
            "stopped": count("stopped"),
            "cancelled": count("cancelled"),
            "decoded_tokens": self.decoded,
            "trace_tokens": [len(t.tokens) for t in self.traces],
            "seconds": round(time.monotonic() - t0, 2),
        }


DEFAULTS = {
    "budget": 64,  # most traces per question
    "n_init": 16,  # warmup traces
    "eta": None,  # keep the top eta percent: 10 is DeepConf-low, 90 DeepConf-high; None is plain voting
    "tau": 0.95,  # the share of the vote that ends generation
    "window": 2048,  # tokens per group
    "k_conf": 20,  # candidates a token's confidence averages over
    "parallel": 16,  # traces in flight after the warmup
    "max_tokens": 32768,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "seed": 1,
}


SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


async def main(input: dict) -> dict:
    config = input.get("config", {})
    cfg = {**DEFAULTS, **(json.loads(config) if isinstance(config, str) else config)}
    if "prompt_tokens" in input:  # the client's own template, so every engine reads the same tokens
        prompt = [int(t) for t in input["prompt_tokens"]]
        cfg["stop_tokens"] = [int(t) for t in input["stop_tokens"]]
    else:
        prompt = chat.user(input["question"] + SUFFIX) + chat.cue()
        cfg["stop_tokens"] = chat.stop_tokens()
    return await DeepConf(cfg, prompt).run()

"""RL rollouts as a tree: one group of samples for GRPO, with shared prefixes.

GRPO samples a group of answers per prompt and learns from how their rewards
differ. Flat sampling writes every answer from the prompt on. Here the group
grows as a tree (TreeRL, TreePO): `roots` answers start from the prompt, and
at every `segment` tokens the least confident unfinished answer forks, so the
child goes on from that point with its own randomness. The child reads the
parent's KV as it stands: the shared part is written once and never read again.

Confidence is DeepConf's: minus the mean log-probability of the `k_conf`
likeliest next tokens, computed on the GPU beside each sampled token. A segment
with low mean confidence is where the answer was unsure, so that is where a
second path is most likely to end differently.

With `mode: flat` the program writes `group` answers from the prompt instead,
the way plain GRPO sampling does.
"""

import asyncio
import json
import time

from inferlet import model
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    channel_capacity,
    dtype,
    gumbel_max,
    intrinsics,
    iota,
    log_softmax,
    reduce_argmax,
    reduce_sum,
    reshape,
    top_k,
)


class Chain:
    """One answer: its working set, the tokens it holds and what it cost."""

    ids = 0

    def __init__(self, ws: WorkingSet, tokens: list[int], last: int, parent=None) -> None:
        self.id = Chain.ids
        Chain.ids += 1
        self.ws, self.tokens, self.last = ws, list(tokens), last
        self.written = 0  # tokens this chain wrote itself, not inherited
        self.conf_sum, self.conf_n = 0.0, 0  # the last segment's confidence
        self.done = False
        self.parent = parent.id if parent else None
        self.shared = len(tokens)  # tokens inherited at the fork


class Rollouts:
    def __init__(self, cfg: dict, prompt: list[int]) -> None:
        self.cfg, self.prompt = cfg, prompt
        self.stops = set(cfg["stop_tokens"])
        self.chains: list[Chain] = []
        self.decoded = 0

    async def prefill(self) -> None:
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

    def fork(self, ws: WorkingSet, pipe: Pipeline, n: int) -> tuple[list, Pipeline]:
        """n forks of `ws`; the parent moves to a new pipeline so each fork can take its own."""
        kids = [ws.fork(pipe) for _ in range(n)]
        pipe.close()
        return kids, Pipeline()

    async def segment(self, c: Chain, pipe: Pipeline, budget: int) -> Pipeline:
        """Write up to `budget` tokens into `c`; the run stops at a stop token or the budget."""
        cfg = self.cfg
        k, temperature = cfg["k_conf"], cfg["temperature"]
        tok_in = Channel.from_([c.last], dtype.i32)
        tok_out = Channel([1], dtype.i32).capacity(channel_capacity())
        conf_out = Channel([1], dtype.f32).capacity(channel_capacity())
        rng = Channel.from_([cfg["seed"] + 7919 * c.id, len(c.tokens)], dtype.u32)
        step = ForwardPass()
        step.embed(tok_in, Channel.from_([0, 1], dtype.u32))
        step.state([c.ws], StateGeometry.folding())

        @step.epilogue
        def _():
            logits = intrinsics.output(0, "logits")
            best, _ = top_k(log_softmax(logits), k)
            conf_out.put(reshape(reduce_sum(best) * (-1.0 / k), [1]))
            r = rng.take()
            token = reshape(gumbel_max(logits * (1.0 / temperature), r), [1])
            tok_in.put(token)
            tok_out.put(token)
            rng.put(r + iota(2))

        c.conf_sum, c.conf_n = 0.0, 0

        async def on_token() -> bool:
            token = await tok_out.take_scalar()
            conf = await conf_out.take_scalar()
            self.decoded += 1
            c.written += 1
            if token in self.stops:
                c.done = True
                return False
            c.tokens.append(token)
            c.last = token
            c.conf_sum += conf
            c.conf_n += 1
            if len(c.tokens) >= cfg["max_tokens"]:
                c.done = True
                return False
            return True

        await pipe.run_ahead(step, budget, on_token)
        # the run closed this pipeline; the chain goes on, if it does, on a new one
        return Pipeline()

    async def run(self) -> dict:
        cfg = self.cfg
        t0 = time.monotonic()
        await self.prefill()
        group, last = cfg["group"], self.prompt[-1]
        if cfg["mode"] == "flat":
            kids, self.root_pipe = self.fork(self.root, self.root_pipe, group)
            self.chains = [Chain(ws, [], last) for ws in kids]
            await asyncio.gather(*[self.segment(c, Pipeline(), cfg["max_tokens"]) for c in self.chains])
        else:
            kids, self.root_pipe = self.fork(self.root, self.root_pipe, cfg["roots"])
            self.chains = [Chain(ws, [], last) for ws in kids]
            pipes = {c.id: Pipeline() for c in self.chains}
            while True:
                live = [c for c in self.chains if not c.done]
                if not live:
                    break
                fresh = await asyncio.gather(*[self.segment(c, pipes[c.id], cfg["segment"]) for c in live])
                for c, p in zip(live, fresh):
                    pipes[c.id] = p
                # grow the group where it was least sure, `per_round` forks a round, so the
                # branch points spread over the answer's length
                room = min(group - len(self.chains), cfg["per_round"])
                unsure = sorted((c for c in live if not c.done and c.conf_n),
                                key=lambda c: c.conf_sum / c.conf_n)[:room]
                for c in unsure:
                    (kid_ws,), pipes[c.id] = self.fork(c.ws, pipes[c.id], 1)
                    kid = Chain(kid_ws, c.tokens, c.last, parent=c)
                    pipes[kid.id] = Pipeline()
                    self.chains.append(kid)
            for p in pipes.values():
                p.close()
            # answers that ended before the tree was full: the rest of the group starts from the prompt
            if len(self.chains) < group:
                kids, self.root_pipe = self.fork(self.root, self.root_pipe, group - len(self.chains))
                extra = [Chain(ws, [], last) for ws in kids]
                self.chains += extra
                await asyncio.gather(*[self.segment(c, Pipeline(), cfg["max_tokens"]) for c in extra])
        self.root_pipe.close()
        return {
            "mode": cfg["mode"],
            "answers": [model.decode(c.tokens) for c in self.chains],
            "lengths": [len(c.tokens) for c in self.chains],
            "shared": [c.shared for c in self.chains],
            "parents": [c.parent for c in self.chains],
            "decoded_tokens": self.decoded,
            "seconds": round(time.monotonic() - t0, 2),
        }


DEFAULTS = {
    "mode": "tree",  # or "flat"
    "group": 8,  # answers per prompt
    "roots": 4,  # answers the tree starts from the prompt
    "segment": 256,  # tokens between branch points
    "per_round": 2,  # forks at most at each branch point
    "k_conf": 20,
    "temperature": 1.0,
    "max_tokens": 4096,
    "seed": 1,
}


async def main(input: dict) -> dict:
    config = input.get("config", {})
    cfg = {**DEFAULTS, **(json.loads(config) if isinstance(config, str) else config)}
    cfg["stop_tokens"] = [int(t) for t in input["stop_tokens"]]
    return await Rollouts(cfg, [int(t) for t in input["prompt_tokens"]]).run()

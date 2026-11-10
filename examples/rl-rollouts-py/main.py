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

A prompt's answers are the lanes of one pass on one pipeline, and every fork
goes on that pipeline too, so a working set never moves between pipelines.
"""

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
    frame_size,
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
        self.passes = 0

    async def prefill(self) -> None:
        self.root, self.pipe = WorkingSet(), Pipeline()
        fwd = ForwardPass()
        fwd.embed(self.prompt[:-1])
        fwd.state([self.root], StateGeometry.folding())
        ready = Channel([1], dtype.i32)

        @fwd.epilogue
        def _():
            ready.put(reshape(reduce_argmax(intrinsics.output(0, "logits")), [1]))

        self.pipe.submit(fwd)
        await ready.take_scalar()

    def fork(self, ws: WorkingSet, n: int) -> list:
        """n forks of `ws`, after everything already on the prompt's pipeline."""
        return [ws.fork(self.pipe) for _ in range(n)]

    async def write(self, chains: list[Chain], budget: int) -> None:
        """Write up to `budget` tokens into each chain, the chains as lanes of one pass.

        The pass runs ahead on the prompt's pipeline, which stays open. A chain
        that ends stops new fires; the fires in flight still write every lane,
        so their tokens are kept, and the rest go on in a pass without it.
        """
        cfg = self.cfg
        k, temperature, v = cfg["k_conf"], cfg["temperature"], model.output_vocab_size()
        r = frame_size()
        window = max(model.run_ahead_window() // r, 1) * r
        for c in chains:
            c.conf_sum, c.conf_n = 0.0, 0
        live, left = [c for c in chains if not c.done], budget
        while live and left > 0:
            n = len(live)
            tok_in = Channel.from_([c.last for c in live], dtype.i32)
            tok_out = Channel([n], dtype.i32).capacity(channel_capacity())
            conf_out = Channel([n], dtype.f32).capacity(channel_capacity())
            rng = Channel.from_([cfg["seed"] + 7919 * self.passes, 0], dtype.u32)
            self.passes += 1
            step = ForwardPass()
            step.embed(tok_in, Channel.from_(list(range(n + 1)), dtype.u32))
            step.state([c.ws for c in live], StateGeometry.folding())

            @step.epilogue
            def _():
                logits = reshape(intrinsics.output(0, "logits"), [n, v])
                best, _ = top_k(log_softmax(logits), k)
                conf_out.put(reduce_sum(best) * (-1.0 / k))
                state = rng.take()
                token = gumbel_max(logits * (1.0 / temperature), state)
                tok_in.put(token)
                tok_out.put(token)
                rng.put(state + iota(2))

            submitted = taken = 0
            stop = False
            while True:
                while not stop and submitted < left and submitted - taken < window:
                    m = min(r, left - submitted)
                    self.pipe.submit_frame([step] * m)
                    submitted += m
                if taken == submitted:
                    break
                tokens = await tok_out.take_host()
                confs = await conf_out.take_host()
                taken += 1
                for c, token, conf in zip(live, tokens, confs):
                    if c.done:
                        continue
                    self.decoded += 1
                    c.written += 1
                    if token in self.stops:
                        c.done = stop = True
                        continue
                    c.tokens.append(token)
                    c.last = token
                    c.conf_sum += conf
                    c.conf_n += 1
                    if len(c.tokens) >= cfg["max_tokens"]:
                        c.done = stop = True
            left -= taken
            live = [c for c in live if not c.done]

    async def run(self) -> dict:
        cfg = self.cfg
        t0 = time.monotonic()
        await self.prefill()
        group, last = cfg["group"], self.prompt[-1]
        if cfg["mode"] == "flat":
            self.chains = [Chain(ws, [], last) for ws in self.fork(self.root, group)]
            await self.write(self.chains, cfg["max_tokens"])
        else:
            self.chains = [Chain(ws, [], last) for ws in self.fork(self.root, cfg["roots"])]
            while True:
                live = [c for c in self.chains if not c.done]
                if not live:
                    break
                await self.write(live, cfg["segment"])
                # grow the group where it was least sure, `per_round` forks a round, so the
                # branch points spread over the answer's length
                room = min(group - len(self.chains), cfg["per_round"])
                unsure = sorted((c for c in live if not c.done and c.conf_n),
                                key=lambda c: c.conf_sum / c.conf_n)[:room]
                for c in unsure:
                    (kid_ws,) = self.fork(c.ws, 1)
                    self.chains.append(Chain(kid_ws, c.tokens, c.last, parent=c))
            # answers that ended before the tree was full: the rest of the group starts from the prompt
            if len(self.chains) < group:
                extra = [Chain(ws, [], last) for ws in self.fork(self.root, group - len(self.chains))]
                self.chains += extra
                await self.write(extra, cfg["max_tokens"])
        self.pipe.close()
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

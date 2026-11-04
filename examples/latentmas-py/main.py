"""LatentMAS: agents that think in the model's hidden states and hand those thoughts to the next agent
through one shared KV context (Zou et al., "Latent Collaboration in Multi-Agent Systems", ICML 2026).
One process runs one question's team.

A latent step feeds the last layer's hidden state back in as the next input instead of a sampled
token: `e = h * scale / |h|`, `scale` the mean norm of the input embeddings, which is the paper's
realignment as its code runs it by default. The steps run on the device: each step's epilogue puts
the next input into the channel the step reads, so `m` steps cost `m` fires and no host trip. Every
pass of a team goes on the team's one pipeline, so its working set never moves between pipelines.

Messages in (JSON), each answered with one JSON message:

- `{"op": "think", "messages" | "tokens", "steps", "scale", "exits"}`: read the prompt after the context, then
  take up to `steps` latent steps there, stopping early once a step's likeliest token is one of `exits`
  (one host trip a step then). Replies `{"len", "read", "taken"}`.
- `{"op": "say", "messages" | "tokens", "max_tokens", "temperature", "top_p", "top_k", "seed", "fresh"}`:
  read the prompt (on an empty context when `fresh`) and sample a reply. Replies
  `{"tokens", "text", "read", "len"}`.
- `{"op": "probe", "tokens", "steps", "scale"}`: `think` on an empty context, one host trip a step,
  sending each latent input as `{"input"}` as it lands (a parity check against Hugging Face). Replies
  `{"steps"}`.
- `{"op": "render", "messages"}`: replies the templated `{"tokens"}`.
- `"quit"`.

`after` (token ids) is read right after the prompt, e.g. an empty thinking block.
`messages` (`[{"role", "content"}]`) are templated here with the model's own chat template. With
`"stream": true`, `think` sends `{"step": k}` as latent step `k` lands and `say` sends
`{"delta": text}` as it writes, before the reply.
"""

import json

from inferlet import chat, model, session
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    broadcast,
    cast,
    channel_capacity,
    dtype,
    frame_size,
    intrinsics,
    iota,
    mask_apply,
    nucleus_sample,
    pivot_threshold,
    rank_le,
    reduce_argmax,
    reduce_sum,
    reshape,
    rsqrt,
)


def render(messages: list[dict]) -> list[int]:
    """The chat-templated prompt for `messages`, ending with the assistant cue."""
    if len(messages) == 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
        tokens = chat.system_user(messages[0]["content"], messages[1]["content"])
    else:
        tokens = []
        for i, m in enumerate(messages):
            role, text = m["role"], m["content"]
            tokens += (chat.system(text) if role == "system" else chat.assistant(text) if role == "assistant"
                       else chat.first_user(text) if i == 0 else chat.user(text))
    return tokens + chat.cue()


def prompt(request: dict) -> list[int]:
    tokens = render(request["messages"]) if request.get("messages") else [int(t) for t in request["tokens"]]
    return tokens + [int(t) for t in request.get("after") or ()]


def realign(h, width: int, scale: float):
    """The next latent input from a hidden row: `h * scale / |h|`, as one `[1, width]` f32 row."""
    h = cast(reshape(h, [1, width]), dtype.f32)
    return h * broadcast(reshape(rsqrt(reduce_sum(h * h)) * scale, [1, 1]), [1, width])


def sample(logits, temperature: float, top_p: float, top_k: int, state):
    """The sampled next token, as a `[1]` cell: greedy at temperature 0, else top-p, among the `top_k`
    likeliest first when set (Hugging Face's and vLLM's order)."""
    if temperature == 0:
        return reshape(reduce_argmax(logits), [1])
    if top_k:
        logits = mask_apply(logits, pivot_threshold(logits, rank_le(top_k)))
    return reshape(nucleus_sample(logits * (1.0 / temperature), top_p, state), [1])


async def drive(pipe: Pipeline, fwd: ForwardPass, budget: int, consume, wide: int, window: int = 0) -> int:
    """Submit `fwd` up to `budget` fires, `wide` a frame, keeping the run-ahead window full while
    `consume` drains one fire's result a call; `consume` returning False stops new fires, and the
    ones in flight are drained. The pipeline stays open for the team's next pass."""
    window = window or max(model.run_ahead_window() // wide, 1) * wide
    submitted = taken = 0
    stop = False
    while True:
        while not stop and submitted < budget and submitted - taken < window:
            n = min(wide, budget - submitted)
            pipe.submit_frame([fwd] * n)
            submitted += n
        if taken == submitted:
            return taken
        stop = (await consume()) is False or stop
        taken += 1


class Team:
    """One question's agents: one working set the agents share, one pipeline every pass goes on."""

    def __init__(self) -> None:
        self.pipe = Pipeline()
        self.fresh()
        self.width = None

    def latent_lane(self) -> None:
        """The latent reading's input width, and whether its rows embed a token too: asked at the first
        latent step, so a team that only reads and writes text runs on any model."""
        if self.width is None:
            lane = model.lane([("reading", "latent")])
            self.width = next(i.width for i in lane.inputs if i.name == "latents")
            self.latent_embeds = lane.embeds

    def fresh(self) -> None:
        self.ws, self.len = WorkingSet(), 0

    def read(self, tokens: list[int], reading: str, epilogue) -> None:
        """One pass reading `tokens` after the context under `reading`, its epilogue on the last row."""
        fwd = ForwardPass()
        if reading != "text":
            fwd.switch(0, "reading", reading)
        fwd.embed(tokens)
        fwd.state([self.ws], StateGeometry.folding())
        fwd.epilogue(epilogue)
        self.pipe.submit(fwd)
        self.len += len(tokens)

    async def first_latent(self, tokens: list[int], scale: float) -> list[float]:
        """Reads `tokens`; the first latent input (the last row's hidden state, realigned), on the host."""
        self.latent_lane()
        first = Channel([1, self.width], dtype.f32)
        self.read(tokens, "text", lambda: first.put(realign(intrinsics.output(0, "hidden"), self.width, scale)))
        return await first.take_host()

    def latent_pass(self, lat: Channel, scale: float, ack: Channel | None, echo: bool = False,
                    probe: bool = False) -> ForwardPass:
        """One latent step: the row's input is `lat`, and its epilogue puts the next one there."""
        fwd = ForwardPass()
        fwd.switch(0, "reading", "latent")
        token = Channel.from_([0], dtype.i32)
        if self.latent_embeds:
            fwd.embed(token, Channel.from_([0, 1], dtype.u32))
        fwd.input(0, "latents", lat)
        fwd.state([self.ws], StateGeometry.folding())

        @fwd.epilogue
        def _():
            if self.latent_embeds:
                token.put(token.take())  # the row's token is unread, but each fire consumes one
            lat.take()
            nxt = realign(intrinsics.output(0, "hidden"), self.width, scale)
            lat.put(nxt)
            if ack is not None:
                ack.put(nxt if echo else reshape(reduce_argmax(intrinsics.output(0, "logits")), [1]) if probe else iota(1))

        return fwd

    async def think(self, tokens: list[int], steps: int, scale: float, stream: bool, echo: bool = False,
                    exits: set[int] = frozenset()) -> int:
        first = await self.first_latent(tokens, scale)
        if steps == 0:
            return 0
        lat = Channel.from_shaped([1, self.width], first, dtype.f32)
        # each step acknowledges with a count, or, when the trajectory is asked for, with its latent input
        probe = bool(exits) and not echo
        ack = (Channel([1, self.width], dtype.f32) if echo else Channel([1], dtype.i32 if probe else dtype.u32)
               ).capacity(channel_capacity())
        done = 0
        self.trace = [first]

        async def consume():
            nonlocal done
            value = await (ack.take_scalar() if probe else ack.take_host())
            if echo:
                self.trace.append(value)
            done += 1
            if stream:
                session.send(json.dumps({"step": done}))
            if probe and value in exits:
                return False  # the latent thought reads as finished: stop here (LaTER's exit)

        # a pass off the model's default reading posts one fire a frame; an early exit runs one step at a time,
        # so no step past the exit lands in the context
        taken = await drive(self.pipe, self.latent_pass(lat, scale, ack, echo, probe), steps, consume, 1,
                            1 if probe else 0)
        self.len += taken
        return taken

    async def probe(self, tokens: list[int], steps: int, scale: float) -> int:
        """Each latent input goes out as its own message: a whole run of them exceeds a frame."""
        inputs = [await self.first_latent(tokens, scale)]
        session.send(json.dumps({"input": inputs[-1]}))
        for _ in range(steps):
            lat = Channel.from_shaped([1, self.width], inputs[-1], dtype.f32)
            out = Channel([1, self.width], dtype.f32)
            fwd = ForwardPass()
            fwd.switch(0, "reading", "latent")
            if self.latent_embeds:
                fwd.embed([0])
            fwd.input(0, "latents", lat)
            fwd.state([self.ws], StateGeometry.folding())
            fwd.epilogue(lambda: out.put(realign(intrinsics.output(0, "hidden"), self.width, scale)))
            self.pipe.submit(fwd)
            inputs.append(await out.take_host())
            session.send(json.dumps({"input": inputs[-1]}))
            self.len += 1
        return steps

    async def say(self, tokens: list[int], count: int, temperature: float, top_p: float, top_k: int, seed: int,
                  stops: set[int], stream: bool) -> list[int]:
        rng_first = Channel.from_([seed, 0], dtype.u32)
        first_out = Channel([1], dtype.i32)

        def first_epilogue():
            state = rng_first.take()
            first_out.put(sample(intrinsics.output(0, "logits"), temperature, top_p, top_k, state))

        self.read(tokens, "text", first_epilogue)
        first = await first_out.take_scalar()
        made = [first]
        if count <= 1 or first in stops:
            return made

        tok_in = Channel.from_([first], dtype.i32)
        tok_out = Channel([1], dtype.i32).capacity(channel_capacity())
        rng = Channel.from_([seed, 1 << 20], dtype.u32)
        fwd = ForwardPass()
        fwd.embed(tok_in, Channel.from_([0, 1], dtype.u32))
        fwd.state([self.ws], StateGeometry.folding())

        @fwd.epilogue
        def _():
            state = rng.take()
            token = sample(intrinsics.output(0, "logits"), temperature, top_p, top_k, state)
            tok_in.put(token)
            tok_out.put(token)
            rng.put(state + iota(2))

        shown = 0
        ended = False

        async def consume():
            nonlocal shown, ended
            token = await tok_out.take_scalar()
            if ended:
                return False  # fired past the stop: drained, not kept
            made.append(token)
            if stream and len(made) % 8 == 0:
                # decode it all and send the new tail, so a character split across tokens arrives whole
                text = model.decode([t for t in made if t not in stops])
                if len(text) > shown and not text.endswith("�"):
                    session.send(json.dumps({"delta": text[shown:]}))
                    shown = len(text)
            if token in stops:
                ended = True
                return False
            return True

        await drive(self.pipe, fwd, count - 1, consume, frame_size())
        self.len += len(made)
        return made


async def main(input: dict) -> dict:
    team = Team()
    request = input
    while True:
        op = request.get("op")
        if op == "think":
            tokens = prompt(request)
            taken = await team.think(tokens, int(request["steps"]), float(request["scale"]), bool(request.get("stream")),
                                     bool(request.get("echo")), set(int(t) for t in request.get("exits") or ()))
            reply = {"len": team.len, "read": len(tokens), "taken": taken}
            if request.get("echo"):
                reply["trace"] = team.trace
        elif op == "render":
            reply = {"tokens": render(request["messages"])}
        elif op == "probe":
            team.fresh()
            reply = {"steps": await team.probe(prompt(request), int(request["steps"]), float(request["scale"]))}
        elif op == "say":
            if request.get("fresh"):
                team.fresh()
            tokens = prompt(request)
            stops = set(int(t) for t in request.get("stops") or chat.stop_tokens())
            made = await team.say(tokens, int(request["max_tokens"]), float(request.get("temperature", 0.0)),
                                  float(request.get("top_p", 1.0)), int(request.get("top_k", 0)),
                                  int(request.get("seed", 0)), stops, bool(request.get("stream")))
            reply = {"tokens": made, "text": model.decode([t for t in made if t not in stops]),
                     "read": len(tokens), "len": team.len}
        else:
            reply = {"error": f"unknown op {op!r}"}
        session.send(json.dumps(reply))
        message = await session.receive()
        if message == "quit":
            team.pipe.close()
            return {}
        request = json.loads(message)

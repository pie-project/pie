"""SWE swarm with stale tool output forgotten: a tree of coding agents on one shared KV cache.

- forget: `forget_after` turns after a command, the middle of its output leaves
  the working set: later tokens stop reading those pages, their KV stops costing
  attention, and nothing is prefilled again. A chat API has to re-send a context
  with the output cut, and re-read everything after the cut.

One agent loop (THOUGHT, then one ```bash``` command, then its output) runs as
a tree of branches inside the engine. The client runs each branch's commands
in that branch's own copy of the repository and sends back the output.

What the engine lets it do that a chat API cannot:

- share: the system prompt and the issue are prefilled once; every branch is
  a fork of that working set, so no branch prefills them again.
- stop: a turn ends the token its command block closes; nothing is decoded
  past it, and no turn is re-sent: each turn's KV stays where it is.
- branch: when the agent is unsure what to do next (the entropy of its next
  reply's first token, measured on the GPU) or its last command failed, the branch forks and samples alternatives
  for the next turn; siblings that pick the same command are dropped.
- memo: a command that changed nothing in the sandbox is reused by any branch
  in the same sandbox state instead of being run again.
- judge: to rank branches, a fork of each asks the model itself "is this on
  track?" and reads p(Yes) off one prefill pass; the fork is then dropped.
- prune: beyond `width` live branches, the lowest-ranked are killed and their
  KV freed; a branch that repeats a command three times is killed.
- rewind: a fork at the start of every turn is a bookmark; a reply that
  repeats a sentence, or a command this branch already ran in the same state,
  is thrown away and resampled from the bookmark with a new seed.
- force: a reply that thinks past `max_thought_tokens` without acting gets
  the opening of a command block written for it.
- verify: once a check that failed passes after an edit, the agent is told
  to submit, and `submit_after` turns later the runtime submits for it.
- edit: a reply that starts overwriting a file the agent has read is rewound
  and restarted with words that commit it to a small, targeted edit.
- vote: submitted patches that agree count as votes; the swarm stops as soon
  as a patch has `quorum` votes or the token budget runs out.

Messages to the client, one JSON object each, keyed by `op`:
  fork {from, to}   copy branch `from`'s sandbox into each id in `to`
  exec {calls}      run each {bid, cmd}; reply {"results": [{bid, rc, out, state, patch?}]}
  kill {bid}        drop a branch's sandbox
  log {...}         progress, for the client to print
The return value is the chosen patch and the run's counts.
"""

import asyncio
import json
import re
import time

from inferlet import chat, model, session
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    constant,
    dtype,
    entropy,
    channel_capacity,
    gumbel_max,
    intrinsics,
    iota,
    reduce_argmax,
    reshape,
    scalar_gather,
    softmax,
)

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FORMAT_ERROR = (
    "Your reply must contain exactly one command in a ```bash block. "
    f"To finish, run: echo {SUBMIT}"
)
VERIFIED = (
    "\n\nNote: this check failed before your edit and passes now. If the fix is complete, "
    f"submit now with: echo {SUBMIT}"
)
REWRITE_HINT = (
    "THOUGHT: I will change only the lines that need it with a small targeted edit "
    "(a python str.replace or sed), not by rewriting the whole file.\n\n```bash\n"
)
CHECK = re.compile(r"\b(python3?|pytest|tox|\./runtests\.py|bin/test)\b")
REWRITE = re.compile(r"cat\s+>\s*([^\s<>|;&]+)\s*<<")
PATH = re.compile(r"(?<![\w/])((?:\./|/testbed/)?[\w-]+(?:/[\w.-]+)*\.(?:py|pyx|pyi|cfg|toml|ini|rst|txt|c|h))\b")


def norm(path: str) -> str:
    for prefix in ("/testbed/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path


JUDGE = "\n\n(Self-check before acting: is my approach so far likely to end in a correct fix? Answer Yes or No.) Answer:"
COMMAND = re.compile(r"```([a-z0-9]*)[ \t]*\n(.*?)```", re.S)


def command(text: str) -> str | None:
    """The reply's command: a bash block as is, a python block run as a script."""
    match = COMMAND.search(text)
    if not match:
        return None
    lang, body = match.group(1), match.group(2).strip()
    if lang in ("python", "py", "python3"):
        return f"python3 - <<'PIE_EOF'\n{body}\nPIE_EOF"
    return body


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def repeats_sentence(text: str) -> bool:
    """Whether the last finished sentence or line already appeared earlier in `text`."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text)]
    if len(parts) < 3:
        return False
    last = parts[-2]  # parts[-1] is still being written
    return len(last) >= 24 and last in parts[:-2]


def envelope(render) -> tuple[list[int], list[int]]:
    """A role's tokens before and after its text, read off two renderings."""
    a, b = render("a"), render("b")
    head, tail = common_prefix(a, b), common_prefix(a[::-1], b[::-1])
    return a[:head], a[len(a) - tail:]


class Stats:
    def __init__(self) -> None:
        self.decoded = 0  # tokens sampled, across every branch
        self.prefilled = 0  # tokens the GPU prefilled
        self.shared = 0  # prefix tokens a fork reused instead of prefilling
        self.judge_tokens = 0
        self.execs = 0
        self.memo_hits = 0
        self.spawned = 1
        self.killed = {"duplicate": 0, "pruned": 0, "loop": 0, "context": 0, "budget": 0, "stopped": 0}
        self.turns = 0
        self.format_errors = 0
        self.rewinds = {"sentence": 0, "command": 0, "rewrite": 0}
        self.verified = 0  # failing checks that came to pass after an edit
        self.auto_submits = 0  # patches the runtime submitted after the agent kept going
        self.rewound_tokens = 0  # sampled, then thrown away by a rewind
        self.forced = 0  # replies whose command block the runtime opened
        self.repeated_outputs = 0  # outputs replaced by a note that they repeat an earlier one
        self.forgotten = 0  # KV slots dropped from working sets
        self.attn_reads = 0  # keys attention read, rows times the context they saw
        self.peak_live = 0  # most KV slots a branch read at once


class Branch:
    """One agent: its working set, its pipeline, and what it has done."""

    ids = 0

    def __init__(self, ws: WorkingSet, pipe: Pipeline, length: int, pending: int, parent=None) -> None:
        self.id = Branch.ids
        Branch.ids += 1
        self.ws, self.pipe = ws, pipe
        self.length = length  # tokens in the working set
        self.forgotten = parent.forgotten if parent else 0  # of them, no longer read
        self.outputs_at: list[list[int]] = [list(o) for o in parent.outputs_at] if parent else []  # [start, end, turn]
        self.pending = pending  # sampled, not yet in the working set
        self.unfed: list[int] = []  # the last reply, decoded on a scratch fork: it goes in with the next observation
        self.commands: list[str] = list(parent.commands) if parent else []
        self.seen: set[tuple[str, str]] = set(parent.seen) if parent else set()  # (state, command) run
        self.loops: dict[tuple[str, str], int] = dict(parent.loops) if parent else {}
        self.outputs: set[int] = set(parent.outputs) if parent else set()  # hashes of outputs seen
        self.checks: dict[str, tuple[int, str]] = dict(parent.checks) if parent else {}  # check -> (rc, state)
        self.verified_at: int | None = parent.verified_at if parent else None  # turn a failing check passed
        self.files: set[str] = set(parent.files) if parent else set()  # repository files it has seen
        self.hint: list[int] | None = None  # tokens to start the next reply with
        self.state = parent.state if parent else "base"
        self.failed = parent.failed if parent else False
        self.entropy = parent.entropy if parent else 0.0
        self.turns = parent.turns if parent else 0
        self.score = parent.score if parent else 0.5
        self.patch: str | None = None
        self.alive = True
        self.text = ""  # the current turn's reply
        self.seed = (self.id + 1) * 7919

    def close(self) -> None:
        self.alive = False
        self.pipe.close()


class Swarm:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.stats = Stats()
        self.branches: list[Branch] = []
        self.memo: dict[tuple[str, str], dict] = {}
        self.submitted: list[Branch] = []
        self.asst_head, self.asst_tail = envelope(chat.assistant)
        self.user_head, self.user_tail = envelope(chat.user)
        self.cue = chat.cue_without_thinking()
        self.stops = set(chat.stop_tokens())
        self.vocab = model.output_vocab_size()
        self.yes = model.encode(" Yes")[0]
        self.no = model.encode(" No")[0]

    # ── the GPU side ────────────────────────────────────────────────────────

    def submit_pass(self, b: Branch, tokens: list[int], temperature: float, measure: bool):
        """Feed `tokens` and sample the next token on the device; with `measure`,
        also the entropy of that token's distribution (one softmax over the vocabulary)."""
        fwd = ForwardPass()
        fwd.embed(tokens)
        fwd.state([b.ws], StateGeometry.folding())
        tok = Channel([1], dtype.i32)
        ent = Channel([1], dtype.f32)
        rng = Channel.from_([b.seed, b.length], dtype.u32)

        @fwd.epilogue
        def _():
            logits = intrinsics.output(0, "logits")
            # always measured: on the CUDA engine a Gumbel draw over logits nothing else reads
            # comes out as token 0; reading them for the entropy first sidesteps it
            ent.put(reshape(entropy(softmax(logits)), [1]))
            if temperature > 0:
                token = gumbel_max(logits / temperature, rng.take())
            else:
                token = reduce_argmax(logits)
            tok.put(reshape(token, [1]))

        b.pipe.submit(fwd)
        b.length += len(tokens)
        live = b.length - b.forgotten
        self.stats.attn_reads += len(tokens) * live
        self.stats.peak_live = max(self.stats.peak_live, live)
        return tok, ent

    async def feed(self, b: Branch, tokens: list[int], temperature: float, measure: bool = False) -> tuple[int, float]:
        tok, ent = self.submit_pass(b, tokens, temperature, measure)
        token = await tok.take_scalar()
        return token, await ent.take_scalar()

    async def stream(self, b: Branch, prefix: list[int], budget: int, keep) -> None:
        """Sample on a scratch fork of `b` on the run-ahead path until `keep(token)`
        says stop. Tokens past the stop were decoded ahead and die with the fork; the
        reply goes into `b` with the next observation, so `b` is the turn's bookmark."""
        cfg, stats = self.cfg, self.stats
        temperature = cfg["temperature"]
        scratch = b.ws.fork(b.pipe)
        live = b.length - b.forgotten + len(prefix)
        stats.attn_reads += len(prefix) * live
        stats.peak_live = max(stats.peak_live, live)
        first_rng = Channel.from_([b.seed, b.length], dtype.u32)
        first = Channel([1], dtype.i32)
        fwd = ForwardPass()
        fwd.embed(prefix)
        fwd.state([scratch], StateGeometry.folding())

        @fwd.epilogue
        def _():
            first.put(reshape(gumbel_max(intrinsics.output(0, "logits") / temperature, first_rng.take()), [1]))

        b.pipe.submit(fwd)
        token = await first.take_scalar()
        if keep(token):
            tok_in = Channel.from_([token], dtype.i32)
            tok_out = Channel([1], dtype.i32).capacity(channel_capacity())
            rng = Channel.from_([b.seed ^ 0x5BD1, b.length], dtype.u32)
            decode = ForwardPass()
            decode.embed(tok_in, Channel.from_([0, 1], dtype.u32))
            decode.state([scratch], StateGeometry.folding())

            @decode.epilogue
            def _():
                r = rng.take()
                sampled = reshape(gumbel_max(intrinsics.output(0, "logits") / temperature, r), [1])
                tok_in.put(sampled)
                tok_out.put(sampled)
                rng.put(r + iota(2))

            async def on_token() -> bool:
                stats.attn_reads += live
                return keep(await tok_out.take_scalar())

            await b.pipe.run_ahead(decode, budget, on_token, end=False)

    async def sample_ahead(self, b: Branch, check: bool) -> tuple[list[int], bool, str | None]:
        """`sample_reply` on the run-ahead path: the same stops, rewinds and forced openers."""
        cfg = self.cfg
        out = list(b.hint) if b.hint else []
        forced = bool(b.hint)
        prefix = ([b.pending] if b.pending is not None else []) + out
        if b.hint:
            b.hint = None
        elif b.pending is None or b.pending in self.stops:
            return [], False, None
        else:
            out = [b.pending]
        verdict: dict = {}

        def keep(token: int) -> bool:
            if token in self.stops:
                verdict["end"] = "stop"
                return False
            out.append(token)
            tail = model.decode(out[-2:])
            if "`" in tail and COMMAND.search(model.decode(out)):
                verdict["end"] = "closed"
                return False
            if check and cfg["no_rewrite"] and "\n" in tail:
                match = REWRITE.search(model.decode(out))
                if match and norm(match.group(1)) in b.files:
                    verdict.update(end="rewind", why="rewrite")
                    return False
            if check and any(c in tail for c in ".!?\n"):
                text = model.decode(out)
                if "```" not in text and repeats_sentence(text):
                    verdict.update(end="rewind", why="sentence")
                    return False
            if not forced and len(out) >= cfg["max_thought_tokens"] and "```" not in model.decode(out):
                verdict["end"] = "long"
                return False
            if len(out) >= cfg["max_turn_tokens"]:
                verdict["end"] = "cap"
                return False
            return True

        while True:
            verdict.clear()
            await self.stream(b, prefix, cfg["max_turn_tokens"] - len(out), keep)
            end = verdict.get("end")
            if end == "closed":
                return out, True, None
            if end == "rewind":
                return out, False, verdict["why"]
            needs_block = end == "long" or (end == "stop" and "```" not in model.decode(out))
            if forced or not needs_block:
                return out, False, None
            # thinking ran long, or the reply tried to end with no command: open the block for it
            forced = True
            self.stats.forced += 1
            out += model.encode("\n```bash\n")
            prefix = list(out)

    async def decode_ahead(self, b: Branch) -> None:
        """One reply on the run-ahead path; a rewind just samples again from `b`."""
        cfg = self.cfg
        pending = b.pending
        for attempt in range(cfg["max_rewinds"] + 1):
            b.pending = pending
            out, closed, why = await self.sample_ahead(b, check=attempt < cfg["max_rewinds"])
            self.stats.decoded += len(out)
            text = model.decode(out)
            if why is None and closed and (b.state, command(text)) in b.seen:
                why = "command"
            if why is None or attempt == cfg["max_rewinds"]:
                break
            self.stats.rewinds[why] += 1
            self.stats.rewound_tokens += len(out)
            b.seed += 104729
            if why == "rewrite":
                b.hint = model.encode(REWRITE_HINT)
        b.text = text
        b.unfed = out[:-1] if closed else out
        b.pending = out[-1] if closed else None

    async def decode_turn(self, b: Branch) -> None:
        """One reply. A fork taken first is the turn's bookmark: a reply that repeats
        itself, or repeats a command already run here, is thrown away and resampled
        from it with a new seed, so the loop never enters the agent's context."""
        cfg = self.cfg
        if cfg["ahead"]:
            return await self.decode_ahead(b)
        mark, mark_length, mark_pending = b.ws.fork(b.pipe), b.length, b.pending
        for attempt in range(cfg["max_rewinds"] + 1):
            out, closed, why = await self.sample_reply(b, check=attempt < cfg["max_rewinds"])
            self.stats.decoded += len(out)
            text = model.decode(out)
            if why is None and closed and (b.state, command(text)) in b.seen:
                why = "command"
            if why is None or attempt == cfg["max_rewinds"]:
                break
            self.stats.rewinds[why] += 1
            self.stats.rewound_tokens += len(out)
            b.ws = mark.fork(b.pipe)  # the bookmark stays untouched for another rewind
            b.length, b.pending = mark_length, mark_pending
            b.seed += 104729
            if why == "rewrite":
                b.hint = model.encode(REWRITE_HINT)
        b.text = text
        b.pending = out[-1] if closed else None
        if cfg.get("debug"):
            session.send({"op": "log", "bid": b.id, "closed": closed, "text": text[:300], "stops": sorted(self.stops)})

    async def sample_reply(self, b: Branch, check: bool) -> tuple[list[int], bool, str | None]:
        """Sample until the command block closes; returns (tokens, closed, why to rewind)."""
        cfg = self.cfg
        temperature = cfg["temperature"]
        out: list[int] = []
        token = b.pending
        forced = False
        if b.hint:
            # the reply starts with words written for it, in place of its sampled first token
            out = list(b.hint)
            token, _ = await self.feed(b, b.hint, temperature)
            b.hint = None
            forced = True
        while len(out) < cfg["max_turn_tokens"] and token not in self.stops:
            out.append(token)
            tail = model.decode(out[-2:])
            if "`" in tail and COMMAND.search(model.decode(out)):
                return out, True, None  # the token that closed the block goes in with the observation
            if check and cfg["no_rewrite"] and "\n" in tail:
                match = REWRITE.search(model.decode(out))
                if match and norm(match.group(1)) in b.files:
                    return out, False, "rewrite"  # overwriting a file it has read: edit it instead
            # repetition is looked for in the prose only: code repeats lines legitimately
            if check and any(c in tail for c in ".!?\n"):
                text = model.decode(out)
                if "```" not in text and repeats_sentence(text):
                    return out, False, "sentence"
            if not forced and len(out) >= cfg["max_thought_tokens"] and "```" not in model.decode(out):
                # thinking ran long: write the block's opening for the model, so it has to act
                forced = True
                self.stats.forced += 1
                opener = model.encode("\n```bash\n")
                token, _ = await self.feed(b, [token] + opener, temperature)
                out += opener
                continue
            token, _ = await self.feed(b, [token], temperature)
            if token in self.stops and not forced and "```" not in model.decode(out):
                # about to end the turn without a command: open the block instead of ending
                forced = True
                self.stats.forced += 1
                opener = model.encode("\n```bash\n")
                token, _ = await self.feed(b, opener, temperature)
                out += opener
        return out, False, None

    async def observe(self, b: Branch, observation: str) -> None:
        """Close the reply, add the command's output as the user's turn, open the next reply."""
        tokens = b.unfed + ([b.pending] if b.pending is not None else []) + self.asst_tail
        b.unfed = []
        tokens += self.user_head
        body = model.encode(observation)
        start = b.length + len(tokens)
        tokens += body + self.user_tail + self.cue
        # the entropy of the reply's first token: how unsure the agent is about what to do next
        b.pending, b.entropy = await self.feed(b, tokens, self.cfg["temperature"], measure=True)
        self.stats.prefilled += len(tokens)
        b.outputs_at.append([start, start + len(body), b.turns])
        self.forget_stale(b)

    def forget_stale(self, b: Branch) -> None:
        """The middle of each output old enough leaves the KV; its ends stay, so the
        turn still reads as a command that printed something."""
        after, keep = self.cfg["forget_after"], self.cfg["forget_keep"]
        if not after:
            return
        for output in b.outputs_at:
            start, end, turn = output
            if end > start and b.turns - turn >= after:
                dropped = b.ws.forget(start + keep, end - keep) if end - start > 2 * keep else 0
                b.forgotten += dropped
                self.stats.forgotten += dropped
                output[1] = start

    async def judge(self, b: Branch) -> float:
        """p(Yes) of a fork asked whether this branch is on track; the fork is dropped."""
        probe = b.ws.fork(b.pipe)
        question = ([b.pending] if b.pending is not None else []) + model.encode(JUDGE)
        fwd = ForwardPass()
        fwd.embed(question)
        fwd.state([probe], StateGeometry.folding())
        yes_out = Channel([1], dtype.f32)
        no_out = Channel([1], dtype.f32)
        yes_id, no_id = self.yes, self.no

        @fwd.epilogue
        def _():
            p = softmax(intrinsics.output(0, "logits"))
            yes_out.put(reshape(scalar_gather(p, constant(yes_id, dtype.u32)), [1]))
            no_out.put(reshape(scalar_gather(p, constant(no_id, dtype.u32)), [1]))

        b.pipe.submit(fwd)
        yes, no = await yes_out.take_scalar(), await no_out.take_scalar()
        self.stats.judge_tokens += len(question)
        return yes / (yes + no) if yes + no > 0 else 0.5

    # ── the tree ────────────────────────────────────────────────────────────

    def fork(self, b: Branch, k: int) -> list[Branch]:
        """k children of `b`, sharing its KV; `b` keeps going beside them."""
        children = [b.ws.fork(b.pipe) for _ in range(k)]
        b.pipe.close()
        b.pipe = Pipeline()
        kids = [Branch(ws, Pipeline(), b.length, b.pending, parent=b) for ws in children]
        self.stats.spawned += k
        self.stats.shared += k * b.length
        session.send({"op": "fork", "from": b.id, "to": [kid.id for kid in kids]})
        self.branches += kids
        return kids

    def kill(self, b: Branch, why: str) -> None:
        if not b.alive:
            return
        b.close()
        self.stats.killed[why] += 1
        session.send({"op": "kill", "bid": b.id})

    def live(self) -> list[Branch]:
        return [b for b in self.branches if b.alive]

    def budget_left(self) -> float:
        return 1 - self.stats.decoded / self.cfg["token_budget"]

    def should_branch(self, b: Branch) -> bool:
        cfg = self.cfg
        if cfg["width"] <= 1 or len(self.live()) >= cfg["width"] or self.budget_left() < 0.5:
            return False
        # only a struggling agent spreads out: a failed command, or unsure what to do next
        return b.turns >= cfg["branch_after_turns"] and (b.failed or b.entropy > cfg["entropy_branch"])

    # ── the loop ────────────────────────────────────────────────────────────

    async def run(self, system: str, task: str) -> dict:
        cfg = self.cfg
        t0 = time.monotonic()
        prompt = chat.prefix() + chat.system(system) + chat.user(task) + self.cue
        root = Branch(WorkingSet(), Pipeline(), 0, 0)
        root.pending, _ = await self.feed(root, prompt, cfg["temperature"])
        self.stats.prefilled += len(prompt)
        self.branches.append(root)

        while self.live():
            live = self.live()
            # branch: uncertain or failing agents spread out before their next turn
            for b in live:
                if self.should_branch(b):
                    self.fork(b, min(cfg["fanout"] - 1, cfg["width"] - len(self.live())))
            live = self.live()
            # a branch that kept going well after its check came to pass is submitted for it
            due = [b for b in live if b.verified_at is not None
                   and b.turns - b.verified_at >= cfg["submit_after"]]
            for b in due:
                b.text = f"```bash\necho {SUBMIT}\n```"
                self.stats.auto_submits += 1
            await asyncio.gather(*[self.decode_turn(b) for b in live if b not in due])
            self.stats.turns += len(live)

            # dedup: siblings in one sandbox state that chose one command are one branch
            seen: dict[tuple[str, str], Branch] = {}
            calls = []
            for b in live:
                cmd = command(b.text)
                b.turns += 1
                if cmd is None:
                    self.stats.format_errors += 1
                    b.next_obs = FORMAT_ERROR
                    continue
                key = (b.state, cmd)
                if key in seen and cfg["dedup"]:
                    self.kill(b, "duplicate")
                    continue
                seen[key] = b
                if b.loops.get(key, 0) >= 2:  # the third run of a command that changed nothing
                    self.kill(b, "loop")
                    continue
                b.loops[key] = b.loops.get(key, 0) + 1
                b.commands.append(cmd)
                b.seen.add(key)
                if cfg["memo"] and key in self.memo:
                    self.stats.memo_hits += 1
                    b.result = self.memo[key]
                else:
                    calls.append({"bid": b.id, "cmd": cmd})
                    b.result = None
                b.next_obs = None

            if calls:
                # tools can run for minutes: tell the engine these pipelines are waiting, not stuck
                for b in self.live():
                    b.pipe.park()
                session.send({"op": "exec", "calls": calls})
                self.stats.execs += len(calls)
                reply = json.loads(await session.receive())
                by_id = {r["bid"]: r for r in reply["results"]}
                for b in self.live():
                    if b.id in by_id:
                        r = by_id[b.id]
                        if r["state"] == b.state and r.get("patch") is None:
                            self.memo[(b.state, b.commands[-1])] = r
                        b.result = r

            for b in self.live():
                r = getattr(b, "result", None)
                if b.next_obs is None and r is not None:
                    b.state = r["state"]
                    b.failed = r["rc"] != 0
                    if r.get("patch") is not None:
                        b.patch = r["patch"]
                        self.submitted.append(b)
                        b.close()
                        continue
                    b.next_obs = self.render(b, r)
                    self.after_command(b, b.commands[-1], r)

            # stop: a patch enough branches agree on, or the budget is gone
            winner = self.consensus()
            if winner or self.budget_left() <= 0:
                for b in self.live():
                    self.kill(b, "stopped" if winner else "budget")
                break

            live = [b for b in self.live() if b.next_obs is not None]
            for b in live:
                if b.length - b.forgotten + len(model.encode(b.next_obs)) + 64 > cfg["max_context"]:
                    self.kill(b, "context")
            live = self.live()
            await asyncio.gather(*[self.observe(b, b.next_obs) for b in live])

            # prune: rank by the model's own judgment, keep the best `width`
            if cfg["judge"] and len(live) > 1:
                scores = await asyncio.gather(*[self.judge(b) for b in live])
                for b, s in zip(live, scores):
                    b.score = s - 0.15 * b.failed
            if len(live) > cfg["width"]:
                for b in sorted(live, key=lambda b: b.score)[: len(live) - cfg["width"]]:
                    self.kill(b, "pruned")
            session.send({"op": "log", "live": len(self.live()), "decoded": self.stats.decoded,
                          "submitted": len(self.submitted)})

        best = self.consensus() or self.best_submitted()
        s = self.stats
        return {
            "patch": best.patch if best else "",
            "winner": best.id if best else None,
            "votes": self.votes(best) if best else 0,
            "seconds": round(time.monotonic() - t0, 2),
            "decoded_tokens": s.decoded,
            "prefilled_tokens": s.prefilled,
            "shared_prefix_tokens": s.shared,
            "judge_tokens": s.judge_tokens,
            "execs": s.execs,
            "memo_hits": s.memo_hits,
            "branches": s.spawned,
            "killed": s.killed,
            "turns": s.turns,
            "format_errors": s.format_errors,
            "rewinds": s.rewinds,
            "rewound_tokens": s.rewound_tokens,
            "forced_actions": s.forced,
            "repeated_outputs": s.repeated_outputs,
            "forgotten": s.forgotten,
            "attn_reads": s.attn_reads,
            "peak_live": s.peak_live,
            "verified": s.verified,
            "auto_submits": s.auto_submits,
            "submitted": len(self.submitted),
        }

    def after_command(self, b: Branch, cmd: str, r: dict) -> None:
        """Learn the files it has seen; notice a failing check that now passes."""
        if not self.cfg["no_rewrite"] and not self.cfg["verify_stop"]:
            return
        for text in (cmd, r["out"][:20000]):
            b.files.update(norm(p) for p in PATH.findall(text) if not p.startswith("/tmp"))
        if not self.cfg["verify_stop"] or not CHECK.search(cmd) or REWRITE.search(cmd) or "sed -i" in cmd:
            return
        before = b.checks.get(cmd)
        if r["rc"] == 0 and before and before[0] != 0 and before[1] != r["state"]:
            b.verified_at = b.turns
            self.stats.verified += 1
            b.next_obs += VERIFIED
        elif r["rc"] != 0 and b.verified_at is not None:
            b.verified_at = None  # a check fails again: not done after all
        b.checks[cmd] = (r["rc"], r["state"])

    def render(self, b: Branch, r: dict) -> str:
        out = r["out"]
        if len(out) > 200:
            h = hash(out)
            if h in b.outputs:
                self.stats.repeated_outputs += 1
                out = "(the same output as an earlier command printed)\n"
            b.outputs.add(h)
        limit = self.cfg["max_output_chars"]
        if len(out) > limit:
            head, tail = out[: limit * 2 // 3], out[-limit // 3:]
            out = f"{head}\n... ({len(r['out']) - len(head) - len(tail)} characters cut) ...\n{tail}"
        return f"<returncode>{r['rc']}</returncode>\n<output>\n{out}</output>"

    @staticmethod
    def normalize(patch: str) -> str:
        return "\n".join(line.rstrip() for line in patch.splitlines() if not line.startswith("index "))

    def votes(self, b: Branch) -> int:
        key = self.normalize(b.patch or "")
        return sum(1 for s in self.submitted if s.patch and self.normalize(s.patch) == key)

    def consensus(self) -> Branch | None:
        for b in self.submitted:
            if not b.patch:
                continue
            # enough branches agree, the model itself is confident, or nobody is left to vote
            if self.votes(b) >= self.cfg["quorum"] or b.score >= self.cfg["accept"] or not self.live():
                return b
        return None

    def best_submitted(self) -> Branch | None:
        good = [b for b in self.submitted if b.patch]
        return max(good, key=lambda b: (self.votes(b), b.score), default=None)


DEFAULTS = {
    "width": 4,  # most live branches
    "fanout": 3,  # children a branching agent spreads into, itself included
    "entropy_branch": 1.0,  # entropy (nats) of a reply's first token above which that reply branches
    "temperature": 0.7,
    "max_turn_tokens": 1536,
    "max_thought_tokens": 384,  # past this with no command block, the runtime opens one
    "max_rewinds": 2,  # resamples of one turn after a repeated sentence or command
    "max_output_chars": 6000,
    "max_context": 48000,
    "token_budget": 60000,  # sampled tokens across all branches
    "quorum": 2,  # agreeing patches that end the run
    "accept": 0.8,  # a patch whose branch the model judged at least this likely to be right ends it too
    "branch_after_turns": 2,  # no branching in the first turns: exploring the repo rarely needs it
    "dedup": True,
    "memo": True,
    "judge": True,
    "verify_stop": False,  # nudge to submit once a failing check passes, submit `submit_after` turns later
    "submit_after": 3,
    "no_rewrite": False,  # rewind a reply that overwrites a file it has read, and start it as an edit
    "ahead": True,  # decode on the run-ahead path, a fork per reply, instead of one host round trip a token
    "forget_after": 0,  # turns after a command that its output leaves the KV; 0 keeps it
    "forget_keep": 64,  # tokens kept at each end of a forgotten output
}


async def main(input: dict) -> dict:
    cfg = {**DEFAULTS, **input.get("config", {})}
    if cfg["width"] <= 1:
        cfg["quorum"] = 1
    return await Swarm(cfg).run(input["system"], input["task"])

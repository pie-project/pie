"""Prefill while the user types: by the time they press Enter, the prompt is
already in the KV cache and only the last word and the turn's end remain.

The client sends the whole draft after every keystroke, `{"draft": "..."}`,
and the final text with `{"send": "..."}`. Only words already followed by a
space are prefilled, since the word being typed can still change how the
tokenizer splits it. Cached slots are frozen, so before each word the working
set is forked (the child shares the frozen KV pages and copies any recurrent
state): a backspace past what is cached goes back to the last checkpoint at or
before the edit and prefills only from there.

Replies, one JSON message each: `{"prefilled": n}` after a draft, then
`{"ttft_ms": .., "tail": ..}`, `{"token": ".."}` per token, and the return.
"""

import json
import time

from inferlet import chat, model, session
from inferlet.eta import ForwardPass, Pipeline, StateGeometry, WorkingSet, intrinsics, reduce_argmax


MAX_CHECKPOINTS = 64  # words a backspace can go back without recomputing


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Prompt:
    """The tokens in the working set, and the passes that put them there."""

    def __init__(self) -> None:
        self.ws = WorkingSet()
        self.pipe = Pipeline()
        self.tokens: list[int] = []
        self.checkpoints: list[tuple[int, WorkingSet]] = []  # (length, a fork of the set at that length)
        self.recomputed = 0  # tokens prefilled again after a backspace

    def extend_to(self, target: list[int]) -> None:
        """Make the working set hold `target`, prefilling only what is new."""
        keep = common_prefix(self.tokens, target)
        if keep < len(self.tokens):
            while self.checkpoints and self.checkpoints[-1][0] > keep:
                self.checkpoints.pop()
            length, snapshot = self.checkpoints[-1] if self.checkpoints else (0, None)
            # fork again so the checkpoint itself stays untouched for the next backspace
            self.ws = snapshot.fork(self.pipe) if snapshot else WorkingSet()
            self.recomputed += keep - length
            self.tokens = self.tokens[:length]
            keep = length
        new = target[keep:]
        if new:
            if self.tokens:
                self.checkpoints.append((len(self.tokens), self.ws.fork(self.pipe)))
                del self.checkpoints[:-MAX_CHECKPOINTS]
            self.pass_(new)

    def pass_(self, new: list[int]):
        fwd = ForwardPass()
        fwd.embed(new)
        fwd.state([self.ws], StateGeometry.folding())
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.logits()))
        self.pipe.submit(fwd)
        self.tokens += new
        return out


async def main(input: dict) -> dict:
    max_tokens = int(input.get("max_tokens", 128))
    system = input.get("system")

    # the user turn's tokens around its text, read off two renderings
    a, b = chat.user("a"), chat.user("b")
    head_len = common_prefix(a, b)
    tail_len = common_prefix(a[::-1], b[::-1])
    start = chat.prefix() + (chat.system(system) if system else [])
    turn_head, turn_tail = a[:head_len], a[len(a) - tail_len:]

    def user_tokens(text: str) -> list[int]:
        return start + turn_head + (model.encode(text) if text else [])

    prompt = Prompt()
    while True:
        message = json.loads(await session.receive())
        if "draft" in message:
            draft = message["draft"]
            stable = draft[: max(draft.rfind(" "), 0)]
            prompt.extend_to(user_tokens(stable))
            session.send({"prefilled": len(prompt.tokens)})
        elif "send" in message:
            break

    # Enter: the last word, the end of the turn and the cue are all that is left
    t0 = time.monotonic()
    full = user_tokens(message["send"]) + turn_tail + chat.cue_without_thinking()
    before = len(prompt.tokens)
    keep = common_prefix(prompt.tokens, full)
    if keep < len(prompt.tokens):
        prompt.extend_to(full[:-1])
        tail = full[-1:]
    else:
        tail = full[keep:]
    out = prompt.pass_(tail)

    stops = set(chat.stop_tokens())
    generated: list[int] = []
    shown = ""
    token = await out
    session.send({"ttft_ms": round((time.monotonic() - t0) * 1000, 1), "tail": len(full) - before})
    while token not in stops and len(generated) < max_tokens:
        generated.append(token)
        text = model.decode(generated)
        if not text.endswith("�"):
            session.send({"token": text[len(shown):]})
            shown = text
        token = await prompt.pass_([token])
    prompt.pipe.close()

    return {
        "text": model.decode(generated),
        "prompt_tokens": len(full),
        "prefilled_while_typing": min(before, keep),
        "recomputed_after_backspace": prompt.recomputed,
        "cached_tokens": prompt.ws.cached_tokens(),
    }

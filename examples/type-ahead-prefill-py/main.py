"""Prefill while the user types: by the time they press Enter, the prompt is
already in the KV cache and only the last word and the turn's end remain.

The client sends the draft each time a word is finished, `{"draft": "..."}`,
and the final text with `{"send": "..."}`. Only words already followed by a
space are prefilled, since the word being typed can still change how the
tokenizer splits it. Cached slots are frozen, so before each word the working
set is forked (the child shares the frozen KV pages and copies any recurrent
state): a backspace past what is cached goes back to the last checkpoint at or
before the edit and prefills only from there.

Replies, one JSON message each: `{"prefilled": n}` after a draft, then
`{"ttft_ms": .., "tail": ..}`, `{"token": ".."}` per token, and the return.
"""

# The big idea, like a story:
# A model reads your question before it can answer. Reading takes time.
# So we let it read each word the moment you finish typing it.
# When you press Enter, it has almost nothing left to read, and answers right away.

# `json` turns messages into text and back, so the client and this program can talk.
import json
# `time` is a stopwatch. We use it to measure how fast the first word of the answer comes.
import time

# `chat` knows the special "who is talking" marks the model expects (user, assistant, stop).
# `model` turns words into numbers (tokens) and numbers back into words.
# `session` is the walkie-talkie between this program and the client typing on the keyboard.
from inferlet import chat, model, session
# These are the parts that make the GPU (the model's fast brain) work:
# ForwardPass   = "read these tokens" (one reading job)
# Pipeline      = a line where reading jobs wait their turn, first come, first served
# StateGeometry = where in memory the tokens just read are written down
# WorkingSet    = the model's memory notebook for this conversation (the "KV cache")
# intrinsics, reduce_argmax = look at the model's guesses and pick the most likely next token
from inferlet.eta import ForwardPass, Pipeline, StateGeometry, WorkingSet, intrinsics, reduce_argmax


# We save a "bookmark" after every word. We keep at most 64 of them,
# so you can backspace up to 64 words without the model re-reading anything.
MAX_CHECKPOINTS = 64  # words a backspace can go back without recomputing


def common_prefix(a: list[int], b: list[int]) -> int:
    # How many tokens at the start of `a` and `b` are exactly the same?
    # Like comparing two sentences letter by letter until they differ.
    n = 0  # nothing matched yet
    for x, y in zip(a, b):  # walk both lists side by side
        if x != y:  # the first place they are different...
            break  # ...stop counting
        n += 1  # same token: count one more
    return n  # the number of matching tokens at the start


class Prompt:
    """The tokens in the working set, and the passes that put them there."""

    def __init__(self) -> None:
        self.ws = WorkingSet()  # a fresh, empty memory notebook on the GPU
        self.pipe = Pipeline()  # the waiting line for reading jobs
        self.tokens: list[int] = []  # our own list of what is already written in the notebook
        # bookmarks: (how many tokens long, a copy of the notebook at that moment)
        self.checkpoints: list[tuple[int, WorkingSet]] = []  # (length, a fork of the set at that length)
        self.recomputed = 0  # tokens prefilled again after a backspace (we hope this stays 0)

    def extend_to(self, target: list[int]) -> None:
        """Make the working set hold `target`, prefilling only what is new."""
        # How much of what the model already read still matches the new text?
        keep = common_prefix(self.tokens, target)
        # If less matches than it read, you erased or changed something it already read.
        # Ink in the notebook can't be erased, so we go back to a bookmark instead.
        if keep < len(self.tokens):
            # Throw away bookmarks made after the place you changed.
            while self.checkpoints and self.checkpoints[-1][0] > keep:
                self.checkpoints.pop()
            # Take the newest bookmark left (or start from an empty page if there is none).
            length, snapshot = self.checkpoints[-1] if self.checkpoints else (0, None)
            # fork again so the checkpoint itself stays untouched for the next backspace
            # (a "fork" is a cheap photocopy: it shares the pages that are already written)
            self.ws = snapshot.fork(self.pipe) if snapshot else WorkingSet()
            # Count the tokens between the bookmark and your change: those get read again.
            self.recomputed += keep - length
            # Our list now ends where the bookmark ends.
            self.tokens = self.tokens[:length]
            keep = length
        # Whatever is after the matching part is new and still has to be read.
        new = target[keep:]
        if new:  # only if there is something new
            if self.tokens:  # and the notebook is not empty
                # Put a bookmark here first, so a later backspace can come back to this spot.
                self.checkpoints.append((len(self.tokens), self.ws.fork(self.pipe)))
                # Keep only the newest 64 bookmarks; throw away older ones.
                del self.checkpoints[:-MAX_CHECKPOINTS]
            self.pass_(new)  # now let the model read the new tokens

    def pass_(self, new: list[int]):
        fwd = ForwardPass()  # make a new reading job
        fwd.embed(new)  # put the new tokens into it
        # write what it read at the end of the notebook, in pen (frozen, it won't change)
        fwd.state([self.ws], StateGeometry.folding())
        # right after reading, on the GPU, pick the most likely next token: the model's guess
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.logits()))
        self.pipe.submit(fwd)  # put the job in line; it runs while we keep going
        self.tokens += new  # remember that these tokens are now in the notebook
        return out  # a "promise": the guessed token, ready a moment later


async def main(input: dict) -> dict:
    # The longest answer we allow, in tokens (a token is about 3/4 of a word). 128 if not given.
    max_tokens = int(input.get("max_tokens", 128))
    # An optional instruction for the model, like "answer like a pirate".
    system = input.get("system")

    # the user turn's tokens around its text, read off two renderings
    # Every model wraps your message in special marks, like an envelope:
    # [start-of-user] your text [end-of-user]. We find the envelope by wrapping
    # "a" and "b": whatever is the same in both is the envelope, not the text.
    a, b = chat.user("a"), chat.user("b")
    head_len = common_prefix(a, b)  # how long the front of the envelope is
    tail_len = common_prefix(a[::-1], b[::-1])  # how long the back is (compare backwards)
    # The very first marks the model wants, plus the optional instruction.
    start = chat.prefix() + (chat.system(system) if system else [])
    # Cut out the front and the back of the envelope.
    turn_head, turn_tail = a[:head_len], a[len(a) - tail_len:]

    def user_tokens(text: str) -> list[int]:
        # Everything the model should read so far: the start, the front of the
        # envelope, and your words. No back of the envelope yet: you're still typing.
        return start + turn_head + (model.encode(text) if text else [])

    prompt = Prompt()  # our notebook helper from above
    while True:  # keep listening until you press Enter
        # wait for the next message from the keyboard side
        message = json.loads(await session.receive())
        if "draft" in message:  # you finished a word (or erased one)
            draft = message["draft"]
            # Keep only the words before the last space. The word you are typing
            # might still grow ("Kyo" -> "Kyoto"), so the model waits for it.
            stable = draft[: max(draft.rfind(" "), 0)]
            prompt.extend_to(user_tokens(stable))  # let the model read those words now
            # tell the keyboard side how many tokens are already read (the grey counter)
            session.send({"prefilled": len(prompt.tokens)})
        elif "send" in message:  # you pressed Enter
            break  # stop listening and go answer

    # Enter: the last word, the end of the turn and the cue are all that is left
    t0 = time.monotonic()  # start the stopwatch
    # The whole thing to read: your words + the back of the envelope +
    # "now it's the assistant's turn" (with the model's thinking switched off).
    full = user_tokens(message["send"]) + turn_tail + chat.cue_without_thinking()
    before = len(prompt.tokens)  # how much was already read when you pressed Enter
    keep = common_prefix(prompt.tokens, full)  # how much of that still matches
    if keep < len(prompt.tokens):
        # You changed something already read at the last second: go back to a bookmark,
        # read everything except the very last token, and read that one by itself below.
        prompt.extend_to(full[:-1])
        tail = full[-1:]
    else:
        # The normal case: only the last word and the closing marks are left (~11 tokens).
        tail = full[keep:]
    out = prompt.pass_(tail)  # read them; the guess after them is the answer's first token

    stops = set(chat.stop_tokens())  # tokens that mean "I'm done talking"
    generated: list[int] = []  # the answer's tokens, one by one
    shown = ""  # the part of the answer already sent to the screen
    token = await out  # wait for the first token of the answer
    # Report the stopwatch time and how many tokens were left to read at Enter.
    session.send({"ttft_ms": round((time.monotonic() - t0) * 1000, 1), "tail": len(full) - before})
    # Keep answering until the model says "done" or the answer gets too long.
    while token not in stops and len(generated) < max_tokens:
        generated.append(token)  # add the new token to the answer
        text = model.decode(generated)  # turn all answer tokens back into words
        # "�" (a broken-letter sign) means a letter is only half done: wait for the next token
        if not text.endswith("�"):
            session.send({"token": text[len(shown):]})  # send only the new bit to the screen
            shown = text  # remember what's on the screen now
        # Feed the new token back so the model can guess the one after it.
        token = await prompt.pass_([token])
    prompt.pipe.close()  # no more reading jobs: close the line

    # The final report sent back to the client when everything is done.
    return {
        "text": model.decode(generated),  # the full answer
        "prompt_tokens": len(full),  # how long your question was, in tokens
        "prefilled_while_typing": min(before, keep),  # how much was read before Enter
        "recomputed_after_backspace": prompt.recomputed,  # tokens read twice (should be 0)
        "cached_tokens": prompt.ws.cached_tokens(),  # tokens Pie reused from its own memory
    }

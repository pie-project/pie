"""engrain - constrained decoding whose parser state lives on the GPU.

A constrained decoder answers one question at every step: which of the model's
tokens may come next? Every deployed system answers it on the host, which means
the answer cannot be inside the CUDA graph a serving engine records for its
decode step - a graph holds device work, and host work put inside it does not
go in at all. This library moves the parser onto the device so that it can.

The whole of a decode loop:

    import engrain

    engine  = engrain.Engine(vocabulary)          # bytes per token id
    grammar = engine.compile(json_schema=schema)

    slots = engine.slots(64)
    slots.admit(0, grammar)                       # a request arrives

    while slots:
        logits  = model(...)
        tokens  = slots.sample(logits)            # the constraint is applied here
        verdict = slots.commit(tokens)            # advance; next mask is ready
        slots.release(finished)                   # a request leaves

There is no `capture()` to remember: the first step records the graph and every
step after replays it. `mask()` is there for an integration that must hand a
bitmask to a sampler it does not own.

**Slots, not a batch.** The rectangle is not an implementation detail, it is
what makes the step capturable, so it is the thing you are given. A slot is
where one request lives; `admit` puts a request in one and `release` takes it
out, which is what continuous batching does and what an API that resets the
whole batch cannot express.

Lower layers, when you need them: `grammar.matcher()` is a host-side reference
parser, and `engrain.internals` exposes the compiler, the pool and the arena.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import threading as _threading
from importlib import metadata as _metadata

from engrain._engrain import CompiledGrammar, CompileError, Matcher

__all__ = [
    "CompileError",
    "CompiledGrammar",
    "Engine",
    "Matcher",
    "Slots",
    "Verdict",
    "__version__",
]

try:
    __version__ = _metadata.version("engrain")
except _metadata.PackageNotFoundError:  # running from a checkout
    __version__ = "0.0.0+source"


@dataclass(frozen=True)
class Verdict:
    """What happened to each slot when it consumed a token.

    Held as device tensors, because reading one is a synchronisation and this
    library exists not to make one. `ok` is the moment you choose to look.
    """

    terminated: object
    """The parser refused the token this slot was given, per slot."""

    narrowed: object
    """A ceiling was reached, so the *next* mask for this slot may be narrower
    than its grammar allows. Blocking a legal token is the one failure this
    engine must not commit quietly, so it is reported rather than absorbed."""

    @property
    def ok(self) -> bool:
        """True when nothing went wrong. **Synchronises with the device.**"""
        return not bool(self.terminated.any()) and not bool(self.narrowed.any())

    def failures(self) -> list[int]:
        """Which slots have a problem. **Synchronises with the device.**"""
        flagged = (self.terminated != 0) | (self.narrowed != 0)
        return flagged.nonzero().flatten().tolist()


class Engine:
    """A vocabulary, the grammars compiled against it, and their tables.

    One engine per model. Grammars live in a single device arena so that a step
    holding many of them is one launch rather than one per grammar - which is
    what a serving step looks like, since requests bring their own.

    `table_budget_bytes` bounds the arena. Past it a grammar no slot is running
    under is evicted and re-admitted from its compiled form if it comes back;
    nothing moves and no identifier changes, so a recorded graph survives.

    `max_stack` is how deep a parse may go. It is a ceiling rather than a
    prediction, because the depth a parse reaches is a property of the document
    and not of the grammar, and the buffers scale with it - so raising it costs
    memory whether or not anything uses the room. A document that needs more is
    reported through `Verdict.narrowed` rather than silently truncated; one
    schema in 425 of JSONSchemaBench needs 257.
    """

    def __init__(
        self,
        vocabulary: Sequence[bytes],
        *,
        max_configurations: int = 128,
        max_stack: int = 256,
        table_budget_bytes: int | None = None,
    ) -> None:
        from engrain._engrain import Compiler
        from engrain.device import DeviceGrammar

        self._compiler = Compiler(list(vocabulary))
        self._pool = DeviceGrammar(
            max_configs=max_configurations,
            max_stack=max_stack,
            budget_bytes=table_budget_bytes,
        )
        self._ids: dict[int, tuple[int, int]] = {}
        self._held: dict[int, CompiledGrammar] = {}
        self._lock = _threading.Lock()

    def compile(
        self,
        *,
        json_schema: str | None = None,
        regex: str | None = None,
        ebnf: str | None = None,
        root: str | None = None,
        **options,
    ) -> CompiledGrammar:
        """Compile one grammar and put its tables on the device.

        Exactly one of `json_schema`, `regex` or `ebnf`. One verb rather than
        three methods, because which front end lowered a grammar is not
        something the rest of the API cares about.

        **Read `grammar.relaxations` before you trust the mask.** Each entry
        names the keyword this grammar does *not* enforce, points at it with a
        JSON pointer, says what the mask now admits, and gives the edit that
        would enforce it - and the list is empty when there is nothing to
        check. The mask may admit more than the source allows and never less,
        so a caller that needs the source itself validates the finished
        document against exactly these.

        Useful options: `exact=True` on a schema enforces a declared property's
        type even where `additionalProperties` is open, at several times the
        compile cost; `lexer_states=` raises the DFA budget for a large
        pattern.

        Raises `CompileError` - a `ValueError` - carrying `.stage`, one of
        `lowering`, `lexer`, `productions`, `conflict` or `emit`. The stage is
        the answer to what to do next: a budget can be raised and retried, a
        lowering failure cannot. `ENGRAIN_WHY=1` prints the diagnostic.

        A ceiling reached at *decode* time is not an exception - nothing can
        raise from inside a graph replay - and arrives as `Verdict.narrowed`.
        """
        given = [
            name
            for name, value in (
                ("json_schema", json_schema),
                ("regex", regex),
                ("ebnf", ebnf),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise TypeError(
                "compile() takes exactly one of json_schema, regex or ebnf, "
                f"got {given or 'none'}"
            )
        if json_schema is not None:
            grammar = self._compiler.compile_json_schema(json_schema, **options)
        elif regex is not None:
            grammar = self._compiler.compile_regex(regex, **options)
        else:
            if root is None:
                raise TypeError("compile(ebnf=...) needs root=")
            grammar = self._compiler.compile_ebnf(ebnf, root, **options)
        self._admit(grammar)
        return grammar

    def _admit(self, grammar: CompiledGrammar) -> int:
        """Put a grammar's tables on the device and return its pool id.

        Idempotent, and safe from several threads: a serving engine compiles on
        a thread pool while a decode loop runs, so the check and the admission
        have to be one step or two requests take the same slot.
        """
        with self._lock:
            key = id(grammar)
            held = self._ids.get(key)
            # An identifier does not identify a grammar across an eviction: the
            # slot is reused, deliberately, so that nothing moves and a
            # recorded graph survives. Ask whether it is still ours.
            if held is not None and self._pool.holds(*held):
                return held[0]
            identifier = self._pool.admit(grammar)
            self._ids[key] = (identifier, self._pool.generation(identifier))
            # Holding the grammar is what makes `id` a key: CPython reuses the
            # address of a dropped object, so without this a caller who let a
            # schema go out of scope could compile another, land on the same
            # address, and be handed the first one's tables. It is also what an
            # eviction re-admits from.
            self._held[key] = grammar
            return identifier

    def slots(self, count: int, *, lookahead: int = 0) -> Slots:
        """`count` places for requests to live in.

        `lookahead` is how many steps of history to keep so `rollback` can undo
        them, which speculative decoding needs and an ordinary loop does not;
        it costs one parse state per step kept.

        The count is fixed for the life of the object because it is the shape
        the CUDA graph is recorded against. Size it once, at the concurrency
        the server is provisioned for, and use `admit` and `release`.
        """
        if not self._ids:
            raise RuntimeError(
                "compile a grammar before asking for slots: their buffers are "
                "sized from the grammars the engine holds"
            )
        return Slots(self, self._pool.new_batch(count, rollback=lookahead))

    @property
    def mask_words(self) -> int:
        """Words in one mask row: the vocabulary rounded up to 32."""
        return self._pool.mask_words

    @property
    def resident_bytes(self) -> int:
        """What the tables occupy on the device, capacity included."""
        return self._pool.resident_bytes()


class Slots:
    """Places a request lives in while it is being decoded.

    Every operation is device-resident and none of them synchronises. The graph
    is recorded on first use and replayed after, and the recording is valid for
    any assignment of grammars to slots, which is why `admit` and `release` are
    free to change it every step.
    """

    def __init__(self, engine: Engine, batch) -> None:
        self._engine = engine
        self._raw = batch
        self._live: set[int] = set()
        self._mask = None
        self._fresh = False
        self._captured = False
        self._capturable = True

    # -- the request lifecycle ---------------------------------------------

    def admit(self, slot: int, grammar: CompiledGrammar) -> None:
        """Put a request in `slot`, under `grammar`, at the start of it.

        The slot may be free or hold a request being replaced; either way it is
        reset. This is a host-to-device write of a few words, paid when a
        request arrives rather than when a token is sampled.
        """
        self._check(slot)
        identifier = self._engine._admit(grammar)
        self._raw.grammar_of[slot] = identifier
        # Writing one slot's grammar is still an assignment, and the batch
        # refuses to fill an unassigned one once the pool holds more than a
        # single grammar - correctly, since zero is a real identifier and an
        # unassigned batch would mask every slot against whatever holds it.
        self._raw.assigned = True
        self._raw.set_configurations(slot, grammar.matcher(0).configurations())
        self._live.add(slot)
        self._fresh = False

    def admit_all(self, grammars: Sequence[CompiledGrammar]) -> None:
        """Fill every slot at once. For a benchmark or a fixed workload."""
        if len(grammars) != len(self):
            raise ValueError(f"{len(grammars)} grammars for {len(self)} slots")
        self._raw.set_grammars([self._engine._admit(g) for g in grammars])
        self._live = set(range(len(self)))
        self._fresh = False

    def release(self, slot: int) -> None:
        """Take the request out of `slot`. Its row stops meaning anything."""
        self._check(slot)
        self._live.discard(slot)

    def resume(self, slot: int, matcher: Matcher) -> None:
        """Put a slot into the state a host matcher is already in.

        For an integration that keeps host matchers too - a request that was
        preempted and is coming back, or a prompt consumed on the host.
        """
        self._check(slot)
        self._raw.set_configurations(slot, matcher.configurations())
        self._live.add(slot)
        self._fresh = False

    # -- the decode step ----------------------------------------------------

    def mask(self):
        """The allowed-token bitmask, `(slots, words)`, on the device.

        For an integration that must hand a bitmask to a sampler it does not
        own. Prefer `sample` or `apply`, which do not make you unpack it.
        """
        self._ready()
        if not self._fresh:
            self._mask = self._raw.fill_mask()
            self._fresh = True
        return self._mask

    def apply(self, logits) -> None:
        """Set every forbidden token's logit to `-inf`, in place.

        `logits` is `(slots, vocabulary)`, and its row may be wider than the
        grammar's - a model's output is padded and a tokenizer's is not - in
        which case the tail is left alone rather than assumed to match.
        """
        import torch

        from engrain import _engrain

        mask = self.mask()
        if logits.dim() != 2 or logits.shape[0] != len(self):
            raise ValueError(
                f"logits must be ({len(self)}, vocabulary), got {tuple(logits.shape)}"
            )
        name = {
            torch.float32: "en_apply_f32",
            torch.float16: "en_apply_f16",
        }.get(logits.dtype)
        if name is None or not logits.is_cuda or not logits.is_contiguous():
            self._apply_with_torch(logits, mask)
            return
        _engrain.cuda_launch(
            name,
            len(self),
            256,
            torch.cuda.current_stream().cuda_stream,
            [mask.data_ptr(), logits.data_ptr()],
            [
                self._engine.mask_words,
                min(logits.shape[1], self._raw.grammar.vocab_size),
                logits.shape[1],
            ],
            grid_y=64,
        )

    def _apply_with_torch(self, logits, mask) -> None:
        """The fallback for a dtype or a layout the kernel does not take."""
        import torch

        bits = torch.arange(32, device=mask.device, dtype=torch.int32)
        flags = ((mask.unsqueeze(-1) >> bits) & 1).to(torch.bool).reshape(len(self), -1)
        width = min(logits.shape[1], flags.shape[1])
        logits[:, :width].masked_fill_(~flags[:, :width], float("-inf"))

    def sample(
        self,
        logits,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        generator=None,
    ):
        """Draw one token per slot, from the tokens the grammar allows.

        Returns `(slots,)` int32 on the device, ready for `commit`.

        **What this is today.** The constraint is applied with a kernel and the
        draw is `torch`'s. The gathered path - sampling from the allowed list
        itself, which is what makes a constrained step cheaper than an
        unconstrained one - is implemented underneath and measured: 2.68x on a
        batch whose rows are all sparse, and 1.05x on the half-dense mixture a
        real workload sits at, because a sampler's output shape is its row
        count and choosing per row needs a host synchronisation this engine
        will not make. When a sampler accepts a device-side row count this
        method changes and its callers do not, which is what it is for.
        """
        import torch

        self.apply(logits)
        work = logits.float()
        if temperature != 1.0:
            if temperature <= 0.0:
                return work.argmax(dim=-1).to(torch.int32)
            work = work / temperature
        if top_k:
            kept = work.topk(min(top_k, work.shape[-1]), dim=-1)
            work = work.masked_fill(work < kept.values[:, -1:], float("-inf"))
        probabilities = work.softmax(dim=-1)
        if top_p < 1.0:
            ordered, index = probabilities.sort(dim=-1, descending=True)
            # Keep the first token past the threshold, so a single token whose
            # mass already exceeds `top_p` is not sampled from an empty set.
            drop = ordered.cumsum(dim=-1) - ordered > top_p
            probabilities = torch.zeros_like(probabilities).scatter_(
                -1, index, ordered.masked_fill(drop, 0.0)
            )
        drawn = torch.multinomial(probabilities, 1, generator=generator)
        return drawn.flatten().to(torch.int32)

    def commit(self, tokens) -> Verdict:
        """Consume one token per slot and get the next mask ready.

        The whole of a decode step after sampling, in one graph replay: the
        advance and the fill that follows it are recorded together, because
        nothing sits between them. `tokens` is `(slots,)` on the device and is
        never read on the host.
        """
        self._ready()
        self._mask = self._raw.advance_and_fill(tokens)
        self._fresh = True
        terminated, narrowed = self._raw.problems()
        return Verdict(terminated=terminated, narrowed=narrowed)

    def rollback(self, steps: int) -> None:
        """Undo `steps` commits. Needs `lookahead` room at construction."""
        self._raw.rollback(steps)
        self._fresh = False

    # -- inspection ---------------------------------------------------------

    def parses(self, slot: int) -> list[tuple[int, list[int]]]:
        """One slot's parse states, as `(lexer state, parser stack)`.

        A device-to-host copy, for checking against a reference matcher; not
        for a decode loop. A slot holds a *set* because scanning a generated
        lexicon is ambiguous - `{` may be a token or the start of a longer one
        - and because a grammar may be ambiguous too.
        """
        return self._raw.configurations(slot)

    @property
    def live(self) -> frozenset[int]:
        """Which slots hold a request."""
        return frozenset(self._live)

    @property
    def free(self) -> frozenset[int]:
        """Which slots do not."""
        return frozenset(range(len(self))) - self._live

    def __len__(self) -> int:
        return self._raw.batch

    def __bool__(self) -> bool:
        return bool(self._live)

    # -- internals ----------------------------------------------------------

    def _check(self, slot: int) -> None:
        if not 0 <= slot < len(self):
            raise IndexError(f"slot {slot} outside 0..{len(self) - 1}")
        if self._raw.outgrown:
            # A grammar admitted since these slots were made raised a ceiling
            # they were sized from, and a kernel indexes what it is given. The
            # engine refuses rather than reading past a buffer.
            raise RuntimeError(
                "a grammar admitted since these slots were made needs more "
                "room than they have; ask the engine for new slots"
            )

    def _ready(self) -> None:
        """Record the graph, once, on first use.

        Capture is not something a caller should have to remember: forgetting
        it costs a replay's worth of dispatch on every token and says nothing
        about it. The differential backend cannot be captured - it compares on
        the host, and a graph would hold whichever side ran last - so a refusal
        is remembered rather than retried every step.
        """
        if self._captured or not self._capturable:
            return
        if not self._live:
            raise RuntimeError("admit a request before stepping")
        try:
            self._raw.fill_mask()
            self._raw.capture()
            self._raw.capture_step()
        except RuntimeError:
            self._capturable = False
        else:
            self._captured = True

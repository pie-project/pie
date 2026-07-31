"""engrain - constrained decoding whose parser state lives on the GPU.

A constrained decoder has to answer one question at every step: which of the
model's tokens may come next? Every deployed system answers it on the host,
which means the answer cannot be inside the CUDA graph a serving engine records
for its decode step - a graph holds device work, and host work put inside it
does not go in at all. This library moves the parser onto the device so that it
can.

The shortest useful program:

    import engrain, torch

    engine = engrain.Engine(vocabulary)          # bytes per token id
    grammar = engine.compile_json_schema(schema)    # or compile_regex(...)

    batch = engine.batch(size=64)
    batch.set_grammars([grammar] * 64)

    mask = batch.fill_mask()                        # (64, words) on the device
    batch.advance(sampled_tokens)                   # (64,) on the device

    # Or skip the vocabulary-wide mask entirely and sample from the set:
    ids, counts = batch.allowed()                  # (64, cap), (64,) on device

`fill_mask` and `advance` are both capturable: call `batch.capture()` once and
every later call replays a recorded graph, with no host work and no
synchronisation on the path. That is the property the design exists for.

Three layers, and you can reach any of them:

- `Engine` and `Batch` here, which is what a serving integration wants.
- `CompiledGrammar.matcher()`, a host-side reference parser used to check the
  device against, and to seed a sequence's state.
- `DeviceGrammar` and `DeviceBatch` in `engrain.device`, which `Engine`
  wraps and which expose the pool, the arena and the graph directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import threading as _threading
from importlib import metadata as _metadata

from engrain._engrain import (
    CompiledGrammar,
    CompileError,
    Compiler,
    Matcher,
    pack_configurations,
)

__all__ = [
    "Batch",
    "CompileError",
    "CompiledGrammar",
    "Compiler",
    "Engine",
    "Matcher",
    "pack_configurations",
    "__version__",
]

# One source, which is the wheel's own metadata. It had been written out in
# three places - here, `pyproject.toml` and the Cargo workspace - and the
# first of those is the one a caller reads.
try:
    __version__ = _metadata.version("engrain")
except _metadata.PackageNotFoundError:  # running from a checkout
    __version__ = "0.0.0+source"


class Engine:
    """A vocabulary, the grammars compiled against it, and their tables.

    One engine per model. Grammars are admitted into a single device arena so
    that a batch under many of them is one launch rather than one per grammar -
    which is what a serving batch looks like, since requests bring their own.

    The arena is bounded by `table_budget_bytes` if you give one. Past it a
    grammar no sequence is running under is evicted, and re-admitted from its
    compiled form if it comes back; nothing moves and no identifier changes, so
    a recorded graph survives the churn.
    """

    def __init__(
        self,
        vocabulary: Sequence[bytes],
        *,
        max_configurations: int = 128,
        table_budget_bytes: int | None = None,
    ) -> None:
        from engrain.device import DeviceGrammar

        self.compiler = Compiler(list(vocabulary))
        self._pool = DeviceGrammar(
            max_configs=max_configurations, budget_bytes=table_budget_bytes
        )
        self._ids: dict[int, tuple[int, int]] = {}
        # Keyed on `id`, so the grammar has to stay alive for the key to mean
        # anything; and a grammar the pool evicts is re-admitted from here.
        self._held: dict[int, CompiledGrammar] = {}
        self._lock = _threading.Lock()

    def compile_json_schema(self, schema: str, **kwargs) -> CompiledGrammar:
        """Compile a JSON Schema and put its tables on the device.

        The mask may admit more than the schema allows, and never less, so a
        caller that needs the schema itself checks the finished document. Read
        `grammar.approximations` for what to check: it lists exactly what this
        grammar does not enforce, and is empty when there is nothing to do.

        Pass `exact=True` to enforce a declared property's type even where
        `additionalProperties` is open. It is off by default because it costs
        every object its own key terminal - compile p50 27 ms to 159 ms, and a
        captured step at batch 512 from 72 us to 155 us.

        Raises `CompileError` - a `ValueError` - carrying `stage`, one of
        `lowering`, `lexer`, `productions`, `conflict` or `emit`. The stage is
        the answer to what a caller should do: a budget may be raised and
        retried, a lowering failure will not be. Set `ENGRAIN_WHY=1` in the
        environment for the underlying diagnostic.

        A ceiling reached at *decode* time is not an exception - nothing can
        raise from inside a graph replay - and is reported by `Batch.problems`.
        """
        return self._admitted(self.compiler.compile_json_schema(schema, **kwargs))

    def compile_regex(self, pattern: str, **kwargs) -> CompiledGrammar:
        """Compile a regular expression. Not everything here is JSON.

        Bounded by the same `lexer_states` budget as a schema, because a
        pattern is the one grammar a request supplies directly and a DFA is
        exponential in the worst case. Raise it with `lexer_states=` when a
        legitimate pattern needs it.
        """
        return self._admitted(self.compiler.compile_regex(pattern, **kwargs))

    def compile_ebnf(self, source: str, root: str, **kwargs) -> CompiledGrammar:
        return self._admitted(self.compiler.compile_ebnf(source, root, **kwargs))

    def _admitted(self, grammar: CompiledGrammar) -> CompiledGrammar:
        # Admitted as soon as it is compiled, rather than when a batch first
        # uses it. A batch's buffers are sized from the pool - the mask width
        # among them - so a pool holding nothing cannot size one, and the
        # alternative was an API where `batch()` had to come second.
        self.admit(grammar)
        return grammar

    def admit(self, grammar: CompiledGrammar) -> int:
        """Put a grammar's tables on the device and return its pool id.

        Idempotent: a grammar already in the pool keeps the id it has, and one
        the pool has since evicted is re-admitted rather than reported at an
        identifier that now names something else.

        Safe to call from several threads. A serving engine compiles on a
        thread pool while a decode loop runs, and the check and the admission
        have to be one step or two requests take the same slot.
        """
        with self._lock:
            key = id(grammar)
            held = self._ids.get(key)
            # The identifier alone does not identify a grammar across an
            # eviction - the slot is reused, deliberately, so that nothing
            # moves and a recorded graph survives. Ask whether it is still ours.
            if held is not None and self._pool.holds(*held):
                return held[0]
            identifier = self._pool.admit(grammar)
            self._ids[key] = (identifier, self._pool.generation(identifier))
            # Holding the grammar is what makes `id` a key at all: CPython
            # reuses the address of a dropped object, so without this a caller
            # who let a schema go out of scope could compile a different one,
            # land on the same address, and be handed the first one's tables.
            # It is also what an eviction needs to re-admit from.
            self._held[key] = grammar
            return identifier

    def batch(self, size: int, *, rollback: int = 0) -> Batch:
        """A batch of `size` sequences.

        `rollback` is how many steps of history to keep, which speculative
        decoding needs and an ordinary decode loop does not; it costs one
        parse state per step kept.
        """
        if not self._ids:
            raise RuntimeError(
                "compile a grammar before making a batch: a batch's buffers "
                "are sized from the grammars the engine holds"
            )
        return Batch(self, self._pool.new_batch(size, rollback=rollback))

    @property
    def mask_words(self) -> int:
        """Words in one mask row, which is the vocabulary rounded up to 32."""
        return self._pool.mask_words

    @property
    def resident_bytes(self) -> int:
        """What the tables occupy on the device, capacity included."""
        return self._pool.resident_bytes()

    @property
    def pool(self):
        """The underlying `DeviceGrammar`, for callers that need the arena."""
        return self._pool


class Batch:
    """Sequences being decoded, and the parse state of each.

    A sequence's state is a *set* of configurations rather than one, because
    scanning a generated lexicon is ambiguous - `{` may be a token or the start
    of a longer one - and because a grammar may be ambiguous too. Everything
    below carries the set.
    """

    def __init__(self, engine: Engine, batch) -> None:
        self._engine = engine
        self._batch = batch

    def set_grammars(self, grammars: Iterable[CompiledGrammar | int]) -> None:
        """Say which grammar each sequence is under, and reset them to its start.

        Takes compiled grammars or pool ids. A grammar not yet on the device is
        admitted here.
        """
        ids = [
            item if isinstance(item, int) else self._engine.admit(item)
            for item in grammars
        ]
        self._batch.set_grammars(ids)

    def set_matchers(self, matchers: Sequence[Matcher]) -> None:
        """Take each sequence's state from a host matcher.

        The fast path for an integration that keeps host matchers as well, and
        the one a serving backend uses: the states go straight to the packer
        rather than through Python lists.
        """
        self._batch.set_matchers(list(matchers))

    def fill_mask(self):
        """The allowed-token bitmask for every sequence, `(batch, words)`.

        Stays on the device. Replays a recorded graph once `capture` has been
        called, which is the deployed path.
        """
        return self._batch.fill_mask()

    def shortlist(self, capacity: int = 8192):
        """The shorter of the two lists per sequence: `(ids, counts, kind)`.

        `kind[i]` is 0 when `ids[i]` names the tokens the sequence *admits* and
        1 when it names the ones it *forbids*. Which one is smaller is a
        property of where the parse is, and it is bimodal rather than spread:
        a structural position admits a few hundred of a hundred and fifty
        thousand, a position inside a string body forbids a few thousand, and
        almost nothing sits between them.

        This is what makes a constrained step cheap at *both* ends. Gathering
        the allowed set is 8.9x the mask path when the set is small and 0.30x
        when it is not, so a caller that always gathers is worse off than one
        that never does. Here the short list is always short, so applying the
        constraint is always a small operation.

        Decided on the device. A caller choosing for itself would have to read
        a count on the host, which is the synchronisation this engine exists
        not to make.

        **What this does not solve.** Acting on `kind` still costs a host
        synchronisation, because a sampler's output shape is its row count and
        `nonzero` has to be read to get one. Measured at batch 512 against
        applying the mask to every row: 2.68x when every row is sparse, 1.05x
        when half are dense - which is the steady state a real workload sits
        at - and 0.86x for the sync-free alternative of sampling both ways and
        selecting. The set being resident is necessary and not sufficient; the
        sampler has to be able to ask for it raggedly, and today it cannot.
        """
        return self._batch.compact(capacity, both=True)

    def allowed(self, capacity: int = 4096):
        """The allowed tokens as sorted ids, `(ids, counts)`, on the device.

        The set the mask stands for, without the vocabulary-wide detour. A
        sampler handed this draws from a few hundred candidates instead of
        sorting a hundred and fifty thousand, which is how a constrained step
        becomes cheaper than an unconstrained one rather than more expensive.

        `ids` is `(batch, capacity)` and only the first `counts[i]` entries of
        row `i` mean anything. `counts` is the *true* size of each set even
        where it exceeds `capacity`, so a caller can see that a row was
        truncated and fall back to `fill_mask` for it - which is what the dense
        regime wants anyway, since a JSON string body admits most of the
        vocabulary and gathering it buys nothing.

        Call after `fill_mask` or `step`; it reads the mask those produced.
        Prefer `shortlist` unless you know every row is sparse: this always
        emits the allowed list, and for a row inside a string body that list
        is nearly the whole vocabulary.
        """
        ids, counts, _ = self._batch.compact(capacity)
        return ids, counts

    def advance(self, tokens) -> None:
        """Consume one sampled token per sequence, `(batch,)` on the device.

        The tokens are never read on the host, so this does not synchronise.
        """
        self._batch.advance(tokens)

    def step(self, tokens):
        """Consume one sampled token per sequence and return the next mask.

        The whole of a decode step after sampling, in one graph replay. Only
        the sample sits between a fill and the advance that follows it, and
        nothing at all sits between that advance and the next fill, so the two
        are one recording - which is a replay's fixed cost saved per token.

        Equivalent to `advance(tokens)` then `fill_mask()`, and returns what
        `fill_mask` returns. This is the path to use in a decode loop.
        """
        return self._batch.advance_and_fill(tokens)

    def capture(self) -> None:
        """Record the fill, the advance, and the two together, as CUDA graphs.

        Every later `fill_mask`, `advance` and `step` is a replay. The
        recording is valid for any assignment of grammars to sequences and any
        batch composition, which is why it can live inside a serving engine's
        own graph; it is invalidated only when the pool's arrays move, and that
        is detected rather than assumed.
        """
        self._batch.capture()
        self._batch.capture_advance()
        self._batch.capture_step()

    def rollback(self, steps: int) -> None:
        """Undo `steps` advances. Needs `rollback` room at construction."""
        self._batch.rollback(steps)

    def configurations(self, sequence: int) -> list[tuple[int, list[int]]]:
        """One sequence's parse states, as `(lexer state, parser stack)`.

        A device-to-host copy. For checking against a reference matcher, not
        for a decode loop.
        """
        return self._batch.configurations(sequence)

    def problems(self):
        """`(terminated, overflow)`, one flag per sequence.

        `terminated` means the parser refused the token it was given.
        `overflow` means a ceiling was reached - the replay window, the
        candidate slots, the configuration set - and the mask that follows may
        be narrower than the grammar allows. Narrowing is the one failure this
        engine must not do quietly, so it is reported here rather than absorbed.
        """
        return self._batch.problems()

    @property
    def outgrown(self) -> bool:
        """Has a grammar admitted since this batch was made outgrown it?

        A batch's buffers are sized from the pool's ceilings, and admitting a
        grammar can raise one. A batch cannot resize itself - a recorded graph
        holds the addresses it recorded, and you hold the mask tensor - so
        make a new batch when this is true. Every operation checks it and
        raises rather than letting a kernel index past a buffer, so ignoring
        it is safe; asking is how you avoid the exception.
        """
        return self._batch.outgrown

    @property
    def size(self) -> int:
        return self._batch.batch

    @property
    def raw(self):
        """The underlying `DeviceBatch`."""
        return self._batch

"""
Generator — multi-step token-generation state machine.

Configure with kwargs on :meth:`Context.generate` (or chain methods on
the returned :class:`Generator`) and iterate with ``async for``::

    g = ctx.generate(
        Sampler.top_p(0.6, 0.95),
        max_tokens=256,
        constrain=JsonSchema(schema_str),
    )

    async for step in g:
        out = await step.execute()
        # out.tokens, out.distribution(probe_handle), ...

For the common case, terminal sugars cover everything in one line:

* :meth:`Generator.collect_text` — drains, decodes through a chat
  decoder, returns the full string.
* :meth:`Generator.collect_tokens` — drains, returns all tokens.
* :meth:`Generator.collect_json` — adds an internal JSON-schema
  constraint, drains, parses (with an optional ``parse=`` validator).

For per-step control (custom sampling, watermarking), iterate manually
with ``async for step in gen``: each step is a :class:`GenStep` you can
tweak (clear sampler, add probes) before ``step.execute()``. Use
:meth:`Generator.accept` to register a manually-sampled token.
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterable, TypeVar

from wit_world.imports import inference as _inf
from wit_world.imports import zo as _zo
from wit_world.imports.inference import SlotOutput_Token

from . import chat as _chat
from ._async import await_future
from .forward import Output, ProbeHandle, SampleHandle, _probe_kind
from .grammar import (
    AnyJson,
    Constraint,
    JsonSchema,
    Schema,
    _brle_and_many,
)

if TYPE_CHECKING:
    from .adapter import Adapter
    from .context import Context
    from .sample import Sampler
    from .spec import Speculator


# =============================================================================
# Internal: static-mask constraint (used by `logit_mask=` kwarg)
# =============================================================================


class _StaticMaskConstraint:
    """Wraps a static BRLE mask as a :class:`Constraint` (returned every step)."""

    __slots__ = ("_mask",)

    def __init__(self, mask: list[int]) -> None:
        self._mask = mask

    def step(self, accepted: list[int]) -> list[int]:
        return self._mask


# =============================================================================
# Generator
# =============================================================================


T = TypeVar("T")


class Generator:
    """Builder + async iterator for token generation. See module docs.

    Construct via :meth:`Context.generate` — never directly.
    """

    __slots__ = (
        "_ctx",
        "_sampler",
        "_stop",
        "_max_tokens",
        "_horizon",
        "_constraints",
        "_constraint_pending",
        "_speculator",
        "_use_system_spec",
        "_spec_drafts",
        "_adapter",
        "_zo_seed",
        "_step_probes",
        "_tokens_generated",
        "_done",
    )

    def __init__(
        self,
        ctx: Context,
        sampler: Sampler,
        *,
        max_tokens: int | None = None,
        stop: Iterable[int] | None = None,
        constrain: Schema | Constraint | list[Schema | Constraint] | None = None,
        logit_mask: list[int] | None = None,
        speculator: Speculator | None = None,
        system_speculation: bool = False,
        adapter: Adapter | None = None,
        zo_seed: int | None = None,
        horizon: int | None = None,
    ) -> None:
        self._ctx = ctx
        self._sampler = sampler
        self._stop: list[int] = list(stop) if stop is not None else []
        self._max_tokens = max_tokens
        self._horizon = horizon
        self._constraints: list[Constraint] = []
        self._constraint_pending: list[int] = []
        self._adapter = adapter
        self._zo_seed = zo_seed
        self._step_probes: list[tuple[int, Any]] = []  # (index, probe)
        self._tokens_generated = 0
        self._done = False

        # Constraints — accept Schema, Constraint, or list of either.
        if constrain is not None:
            items = constrain if isinstance(constrain, list) else [constrain]
            for c in items:
                self._add_constraint(c)
        if logit_mask is not None:
            self._constraints.append(_StaticMaskConstraint(list(logit_mask)))

        # Speculation — system-driven OR custom drafter (or neither).
        if system_speculation and speculator is not None:
            raise ValueError(
                "speculator and system_speculation are mutually exclusive"
            )
        self._speculator: Speculator | None = speculator
        self._use_system_spec: bool = system_speculation
        # Cache for next-iter system drafts (populated each step from
        # the WIT output's spec channel when system speculation is on).
        self._spec_drafts: tuple[list[int], list[int]] = ([], [])

    def _add_constraint(self, c: Schema | Constraint) -> None:
        if hasattr(c, "build_constraint"):
            self._constraints.append(c.build_constraint(self._ctx._model))
        elif hasattr(c, "step"):
            self._constraints.append(c)
        else:
            raise TypeError(
                "constrain must be a Schema (with build_constraint) "
                "or a Constraint (with step)"
            )

    # ── Chain methods (alternative to kwargs) ────────────────────────

    def max_tokens(self, n: int) -> Generator:
        """Hard cap on tokens generated across all steps."""
        self._max_tokens = n
        return self

    def stop(self, tokens: Iterable[int]) -> Generator:
        """Stop tokens. Generation halts when any of these is sampled."""
        self._stop = list(tokens)
        return self

    def add_stop(self, tokens: Iterable[int]) -> Generator:
        """Append to the stop set."""
        self._stop.extend(tokens)
        return self

    def constrain(self, c: Schema | Constraint) -> Generator:
        """Attach a constraint. Multiple calls compose by AND-ing
        per-step BRLE masks."""
        self._add_constraint(c)
        return self

    def horizon(self, n: int) -> Generator:
        """Hint expected output length for budget planning."""
        self._horizon = n
        return self

    def adapter(self, a: Adapter) -> Generator:
        """Apply an adapter (LoRA etc.) on every forward pass."""
        self._adapter = a
        return self

    def zo_seed(self, seed: int) -> Generator:
        """Set zo (Evolution Strategies) seed on every forward pass."""
        self._zo_seed = seed
        return self

    def probe_each_step(self, index: int, probe: Any) -> ProbeHandle:
        """Attach a probe to every step at ``index``. Returns a handle
        reusable across each :class:`Output`."""
        # Slot 0 reserved for auto-sampler; per-step probes follow.
        slot = 1 + len(self._step_probes)
        self._step_probes.append((index, probe))
        return ProbeHandle(slot=slot, kind=_probe_kind(probe))

    # ── Iteration ────────────────────────────────────────────────────

    @property
    def tokens_generated(self) -> int:
        return self._tokens_generated

    @property
    def is_done(self) -> bool:
        """Whether generation has terminated. ``True`` after a stop
        token, max-tokens cap, or empty step (no input + no drafts)."""
        return self._done or (
            self._max_tokens is not None
            and self._tokens_generated >= self._max_tokens
        )

    def __aiter__(self) -> AsyncIterator[GenStep]:
        return self

    async def __anext__(self) -> GenStep:
        if self.is_done:
            raise StopAsyncIteration

        # Drain context buffer (filled by `system / user / cue / ...`).
        pending = self._ctx._pending_tokens
        self._ctx._pending_tokens = []

        # Pull drafts from the speculator.
        if self._use_system_spec:
            drafts, draft_positions = self._spec_drafts
            self._spec_drafts = ([], [])
        elif self._speculator is not None:
            drafts, draft_positions = self._speculator.draft()
        else:
            drafts, draft_positions = ([], [])

        # Compose constraint masks.
        mask: list[int] | None = None
        if self._constraints:
            advance = self._constraint_pending
            self._constraint_pending = []
            masks = [c.step(advance) for c in self._constraints]
            masks = [m for m in masks if m]
            if masks:
                mask = _brle_and_many(masks)

        return GenStep(
            self,
            list(pending),
            list(drafts),
            list(draft_positions),
            mask,
        )

    # ── User-sampled mode ────────────────────────────────────────────

    def accept(self, tokens: list[int]) -> list[int]:
        """Register manually-sampled tokens with the generator. Use
        after :meth:`GenStep.clear_sampler` when the inferlet sampled by
        hand off a probe — the generator updates max-tokens / stop /
        constraint counters and seeds the next iteration's input."""
        if not tokens:
            return []

        accepted = list(tokens)
        # Stop-token truncation.
        for i, t in enumerate(accepted):
            if t in self._stop:
                accepted = accepted[:i]
                self._done = True
                break
        # Max-tokens enforcement.
        if self._max_tokens is not None:
            remaining = self._max_tokens - self._tokens_generated
            if len(accepted) > remaining:
                accepted = accepted[:remaining]
                self._done = True

        if not accepted:
            return []

        # Stage for next forward pass via the buffer; advance counters.
        self._ctx._pending_tokens.extend(accepted)
        self._constraint_pending.extend(accepted)
        self._tokens_generated += len(accepted)
        if self._speculator is not None:
            self._speculator.accept(accepted)
        return accepted

    # ── Terminal sugar ───────────────────────────────────────────────

    async def collect_tokens(self) -> list[int]:
        """Drain to completion; return the full token stream."""
        out: list[int] = []
        async for step in self:
            res = await step.execute()
            out.extend(res.tokens)
        return out

    async def collect_text(self) -> str:
        """Drain, decode through a chat decoder, return the response text.

        Returns the chat decoder's ``Done.text`` if the model emits a
        clean end-of-turn (the expected case); otherwise concatenates
        every ``Delta`` chunk. The two are equal when the host honors
        the chat-template contract (Done's text == sum of deltas)."""
        decoder = _chat.Decoder(self._ctx._model)
        text_parts: list[str] = []
        async for step in self:
            res = await step.execute()
            ev = decoder.feed(res.tokens)
            if isinstance(ev, _chat.Event.Delta):
                text_parts.append(ev.text)
            elif isinstance(ev, _chat.Event.Done):
                return ev.text
        return "".join(text_parts)

    async def collect_json(
        self,
        *,
        schema: str | None = None,
        parse: Callable[[str], T] | None = None,
    ) -> Any:
        """Generate JSON-constrained output and parse it.

        Three calling conventions:

        * **Schema string**: ``await g.collect_json(schema=schema_str)``
          — returns a parsed ``dict`` / ``list`` / primitive.
        * **Custom parser**: ``await g.collect_json(schema=schema_str,
          parse=my_validator)`` — runs ``parse(text)`` on the generated
          text after generation completes; return value is whatever
          ``parse`` returns.
        * **No args**: ``await g.collect_json()`` — falls back to
          :class:`AnyJson` (any valid JSON), returns parsed value.

        Note: validation libraries with native-code extensions (pydantic
        v2, msgspec, orjson, …) do not load inside the WASM runtime
        today. Use ``schema=`` + ``parse=`` with a pure-Python validator
        if you need typed output.
        """
        if schema is not None:
            self._add_constraint(JsonSchema(schema))
        else:
            self._add_constraint(AnyJson())

        text = await self.collect_text()
        if parse is not None:
            return parse(text)
        return _json.loads(text)


# =============================================================================
# GenStep — short-lived per-iteration handle
# =============================================================================


class GenStep:
    """Configuration handle for the upcoming forward pass. Yielded by
    iterating a :class:`Generator`. Pre-populated with the generator's
    pending fills, configured sampler, constraint mask, and any
    speculator drafts.

    Tweak (call :meth:`probe`, :meth:`clear_sampler`) before
    :meth:`execute`.
    """

    __slots__ = (
        "_gen",
        "_pending",
        "_drafts",
        "_draft_positions",
        "_mask",
        "_extra_probes",
        "_user_cleared_sampler",
    )

    def __init__(
        self,
        gen: Generator,
        pending: list[int],
        drafts: list[int],
        draft_positions: list[int],
        mask: list[int] | None,
    ) -> None:
        self._gen = gen
        self._pending = pending
        self._drafts = drafts
        self._draft_positions = draft_positions
        self._mask = mask
        self._extra_probes: list[tuple[int, Any]] = []
        self._user_cleared_sampler = False

    def clear_sampler(self) -> GenStep:
        """Drop the generator's auto-attached sampler. The caller must
        read the distribution off a probe and register their own pick
        via :meth:`Generator.accept` after :meth:`execute`."""
        self._user_cleared_sampler = True
        return self

    def probe(self, index: int, probe: Any) -> ProbeHandle:
        """Attach an extra probe at ``index`` for this iteration only."""
        base = 0 if self._user_cleared_sampler else 1
        slot = base + len(self._gen._step_probes) + len(self._extra_probes)
        self._extra_probes.append((index, probe))
        return ProbeHandle(slot=slot, kind=_probe_kind(probe))

    async def execute(self) -> Output:
        """Run the forward pass and fold the result into the
        generator's state."""
        gen = self._gen
        ctx = gen._ctx
        n_pending = len(self._pending)
        n_drafted = len(self._drafts)

        # Truly nothing to do — no input, no auto-sampler, no extra probes.
        if (
            n_pending == 0
            and n_drafted == 0
            and self._user_cleared_sampler
            and not self._extra_probes
        ):
            gen._done = True
            return Output(
                _inf.Output(slots=[], spec_tokens=[], spec_positions=[])
            )

        # Reserve pages for pending + drafts.
        n_total = n_pending + n_drafted
        if n_total > 0:
            total_after = ctx._working_tokens + n_total
            pages_needed = (total_after + ctx._page_size - 1) // ctx._page_size
            additional = max(0, pages_needed - ctx._working_pages)
            if additional > 0:
                ctx._handle.reserve_working_pages(additional)
                ctx._working_pages = pages_needed

        # Build forward pass.
        fwd = _inf.ForwardPass(ctx._model._handle)
        fwd.context(ctx._handle)
        if gen._adapter is not None:
            fwd.adapter(gen._adapter._handle)
        if gen._zo_seed is not None:
            _zo.adapter_seed(fwd, gen._zo_seed)

        if n_pending > 0:
            positions = list(range(ctx._seq_len, ctx._seq_len + n_pending))
            fwd.input_tokens(self._pending, positions)
        if self._drafts:
            fwd.input_speculative_tokens(self._drafts, self._draft_positions)
        if gen._use_system_spec:
            fwd.output_speculative_tokens(True)

        # Sampler at last input position (or 0 if drafts only / no input).
        sample_idx = max(0, n_pending - 1)
        if not self._user_cleared_sampler:
            fwd.sampler([sample_idx], gen._sampler._variant)

        # Per-generator step probes.
        for idx, probe in gen._step_probes:
            fwd.sampler([idx], probe._to_wit())
        # Per-step extra probes.
        for idx, probe in self._extra_probes:
            fwd.sampler([idx], probe._to_wit())

        if self._mask is not None:
            fwd.logit_mask(self._mask)

        raw = await await_future(fwd.execute(), "GenStep.execute failed")

        # Collect accepted tokens off slot 0 (and following Token slots
        # in spec mode — verifier produces a sequence).
        if self._user_cleared_sampler:
            accepted: list[int] = []
        else:
            accepted = []
            for slot in raw.slots:
                if isinstance(slot, SlotOutput_Token):
                    accepted.append(slot.value)
                else:
                    break

        # Stash next-iter system drafts; let custom speculators see accepted.
        if gen._use_system_spec:
            gen._spec_drafts = (list(raw.spec_tokens), list(raw.spec_positions))
        elif gen._speculator is not None:
            gen._speculator.accept(accepted)

        # Truncate rejected drafts.
        if n_drafted > 0:
            n_verified = max(0, len(accepted) - 1)
            n_rejected = n_drafted - n_verified
            if n_rejected > 0:
                ctx._handle.truncate_working_page_tokens(n_rejected)
                if gen._speculator is not None:
                    gen._speculator.rollback(n_rejected)

        # Commit pages: pending always commit (real KV); verified drafts too.
        n_verified_drafts = max(0, len(accepted) - 1) if n_drafted > 0 else 0
        n_kv = n_pending + n_verified_drafts
        if n_kv > 0:
            new_working = ctx._working_tokens + n_kv
            to_commit = new_working // ctx._page_size
            if to_commit > 0:
                ctx._handle.commit_working_pages(to_commit)
            ctx._committed_pages += to_commit
            ctx._working_pages -= to_commit
            ctx._working_tokens = new_working % ctx._page_size
            ctx._seq_len += n_kv
        elif n_drafted > 0 and not accepted:
            # All drafts rejected with no anchor — re-sync from host.
            ctx._committed_pages = ctx._handle.committed_page_count()
            ctx._working_pages = ctx._handle.working_page_count()
            ctx._working_tokens = ctx._handle.working_page_token_count()
            ctx._seq_len = (
                ctx._committed_pages * ctx._page_size + ctx._working_tokens
            )

        # Advance constraint state with accepted tokens.
        if gen._constraints:
            gen._constraint_pending.extend(accepted)

        # Apply stop / max truncation, accumulate counters, seed buffer.
        tokens = list(accepted)
        for i, t in enumerate(tokens):
            if t in gen._stop:
                tokens = tokens[:i]
                gen._done = True
                break
        if gen._max_tokens is not None:
            remaining = gen._max_tokens - gen._tokens_generated
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
                gen._done = True
        gen._tokens_generated += len(tokens)
        if tokens:
            ctx._pending_tokens.append(tokens[-1])

        auto_sampler = (
            None if self._user_cleared_sampler else SampleHandle(slot=0, arity=1)
        )
        return Output(raw, tokens=tokens, auto_sampler=auto_sampler)

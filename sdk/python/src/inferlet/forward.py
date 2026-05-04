"""
Forward — single forward-pass primitive with auto page management.

``ctx.forward()`` returns a :class:`Forward` builder. Attach inputs,
samplers, probes, masks, then ``await forward.execute()``::

    fwd = ctx.forward()
    fwd.input(prompt_tokens)
    h = fwd.sample([len(prompt_tokens) - 1], Sampler.argmax())
    out = await fwd.execute()
    token = out.token(h)

For prefill / scoring / custom decode loops. The :class:`Generator` layer
is built on top of this for the common token-generation case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wit_world.imports import inference as _inf
from wit_world.imports.inference import (
    SlotOutput_Distribution,
    SlotOutput_Entropy,
    SlotOutput_Logits,
    SlotOutput_Logprobs,
    SlotOutput_Token,
)

from ._async import await_future

if TYPE_CHECKING:
    from .adapter import Adapter
    from .context import Context
    from .sample import Sampler


# =============================================================================
# Slot handles
# =============================================================================


@dataclass(frozen=True, slots=True)
class SampleHandle:
    """Reference to a sampler slot. Pass to :meth:`Output.token` /
    :meth:`Output.tokens_at` to read the result."""

    slot: int
    arity: int = 1


@dataclass(frozen=True, slots=True)
class ProbeHandle:
    """Reference to a probe slot. Carries the probe ``kind`` so the
    Output's accessor can sanity-check at runtime."""

    slot: int
    kind: str  # "logits" | "distribution" | "logprobs" | "entropy"


def _probe_kind(probe) -> str:
    """Map a probe spec instance to its accessor-kind string."""
    # Local import — `sample.py` is a leaf module (no upward deps), so
    # bringing it in at module load time is fine, but doing it lazily
    # here keeps `forward.py` independently importable for testing.
    from .sample import Distribution, Entropy, Logits, Logprob, Logprobs

    if isinstance(probe, Logits):
        return "logits"
    if isinstance(probe, Distribution):
        return "distribution"
    if isinstance(probe, (Logprob, Logprobs)):
        return "logprobs"
    if isinstance(probe, Entropy):
        return "entropy"
    raise TypeError(f"Unknown probe type: {type(probe).__name__}")


# =============================================================================
# Slot specs (internal)
# =============================================================================


@dataclass(slots=True)
class _SampleSlot:
    indices: list[int]
    sampler: Any  # Sampler


@dataclass(slots=True)
class _ProbeSlot:
    index: int
    probe: Any  # Logits | Distribution | Logprob | Logprobs | Entropy


# =============================================================================
# Forward
# =============================================================================


class Forward:
    """Single forward pass. Construct via :meth:`Context.forward`.

    Builder methods return ``self`` so chains compose. ``await execute()``
    runs the host call, commits any newly-filled pages, and returns an
    :class:`Output`.
    """

    __slots__ = (
        "_ctx",
        "_auto_inputs",
        "_explicit_inputs",
        "_slots",
        "_next_slot",
        "_mask",
        "_attn_mask",
        "_adapter",
        "_zo_seed",
    )

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._auto_inputs: list[int] = []
        self._explicit_inputs: list[tuple[list[int], list[int]]] = []
        self._slots: list[_SampleSlot | _ProbeSlot] = []
        self._next_slot = 0
        self._mask: list[int] | None = None
        self._attn_mask: list[list[int]] | None = None
        self._adapter: Adapter | None = None
        self._zo_seed: int | None = None

    # ── Position accessors ────────────────────────────────────────────

    def start_position(self) -> int:
        """Position the *first* auto-input token will occupy. Equal to the
        owning context's ``seq_len`` at the time ``forward()`` was called.

        The sampler at index ``i`` (when ``forward.sample([i], ...)``)
        lands at ``start_position() + i``.
        """
        return self._ctx._seq_len

    # ── Inputs ────────────────────────────────────────────────────────

    def input(self, tokens: list[int]) -> Forward:
        """Append ``tokens`` at positions starting at the context's current
        sequence length. Multiple calls accumulate. After ``execute()``
        these tokens occupy KV slots and ``seq_len`` advances."""
        self._auto_inputs.extend(tokens)
        return self

    def input_at(self, tokens: list[int], positions: list[int]) -> Forward:
        """Feed ``tokens`` at caller-supplied ``positions``. Use for
        scoring at arbitrary positions (e.g. multi-candidate evaluation).
        These tokens are NOT auto-committed — caller manages page
        bookkeeping if positions overlap or extend beyond ``seq_len``."""
        if len(tokens) != len(positions):
            raise ValueError("tokens and positions must be the same length")
        self._explicit_inputs.append((list(tokens), list(positions)))
        return self

    # ── Slot attach ───────────────────────────────────────────────────

    def sample(self, indices: list[int], sampler: Sampler) -> SampleHandle:
        """Attach a sampler at one or more ``indices`` (0-based into the
        auto-input window). Returns a handle for reading the sampled
        token(s) on the resulting :class:`Output`.

        A multi-arity sampler produces ``len(indices)`` Token slots in
        the output, so the next slot index advances by that count — any
        subsequent ``sample`` / ``probe`` call sees the right offset.
        """
        arity = len(indices)
        h = SampleHandle(slot=self._next_slot, arity=arity)
        self._slots.append(_SampleSlot(list(indices), sampler))
        self._next_slot += arity
        return h

    def probe(self, index: int, probe: Any) -> ProbeHandle:
        """Attach a probe at a single ``index``. Returns a handle whose
        ``kind`` selects which ``output.*`` accessor decodes the result."""
        h = ProbeHandle(slot=self._next_slot, kind=_probe_kind(probe))
        self._slots.append(_ProbeSlot(index, probe))
        self._next_slot += 1
        return h

    # ── Decoration ────────────────────────────────────────────────────

    def mask(self, brle: list[int]) -> Forward:
        """Set a static logit mask (BRLE) applied at every sampled position."""
        self._mask = list(brle)
        return self

    def attention_mask(self, masks: list[list[int]]) -> Forward:
        """Set per-query-position attention masks. Length must match the
        total number of query positions across all ``input`` /
        ``input_at`` calls. If unset, the runtime synthesizes a causal mask."""
        self._attn_mask = [list(m) for m in masks]
        return self

    def adapter(self, adapter: Adapter) -> Forward:
        """Apply an adapter (LoRA, etc.) for this forward pass."""
        self._adapter = adapter
        return self

    def zo_seed(self, seed: int) -> Forward:
        """Set a zo (Evolution Strategies) seed for this forward pass."""
        self._zo_seed = seed
        return self

    # ── Execute ───────────────────────────────────────────────────────

    async def execute(self) -> Output:
        """Run the forward pass. Reserves working pages for any
        auto-inputs, submits all attached inputs and slots, awaits the
        host, commits any newly-filled pages, and updates the context's
        cached state.

        Raises ``ValueError`` if no inputs and no slots are attached —
        a vacuous Forward almost always indicates a missed
        ``input(...)`` or ``sample(...)`` call.
        """
        ctx = self._ctx
        n_auto = len(self._auto_inputs)
        n_total = n_auto + sum(len(t) for t, _ in self._explicit_inputs)

        if n_total == 0 and not self._slots:
            raise ValueError(
                "Forward.execute() called with no inputs and no slots. "
                "Attach at least one input (`forward.input(...)`) or "
                "slot (`forward.sample(...)` / `forward.probe(...)`) "
                "before executing."
            )

        # Reserve pages for auto-inputs (those occupy KV and commit on
        # the way out). Explicit inputs are scoring-only — the caller
        # manages their pages.
        if n_auto > 0:
            total_after = ctx._working_tokens + n_auto
            pages_needed = (total_after + ctx._page_size - 1) // ctx._page_size
            additional = max(0, pages_needed - ctx._working_pages)
            if additional > 0:
                ctx._handle.reserve_working_pages(additional)
                ctx._working_pages = pages_needed

        # Build forward pass.
        fwd = _inf.ForwardPass(ctx._model._handle)
        fwd.context(ctx._handle)
        if self._adapter is not None:
            fwd.adapter(self._adapter._handle)
        if self._zo_seed is not None:
            from wit_world.imports import zo as _zo

            _zo.adapter_seed(fwd, self._zo_seed)

        if n_auto > 0:
            positions = list(range(ctx._seq_len, ctx._seq_len + n_auto))
            fwd.input_tokens(self._auto_inputs, positions)
        for tokens, positions in self._explicit_inputs:
            fwd.input_tokens(tokens, positions)

        # Slot attaches go in declaration order — slot indices match
        # what we handed back via SampleHandle / ProbeHandle.
        for spec in self._slots:
            if isinstance(spec, _SampleSlot):
                fwd.sampler(spec.indices, spec.sampler._variant)
            else:  # _ProbeSlot
                fwd.sampler([spec.index], spec.probe._to_wit())

        if self._mask is not None:
            fwd.logit_mask(self._mask)
        if self._attn_mask is not None:
            fwd.attention_mask(self._attn_mask)

        raw = await await_future(fwd.execute(), "Forward.execute failed")

        # Commit pages that auto-input tokens fully filled.
        if n_auto > 0:
            new_working = ctx._working_tokens + n_auto
            to_commit = new_working // ctx._page_size
            if to_commit > 0:
                ctx._handle.commit_working_pages(to_commit)
            ctx._committed_pages += to_commit
            ctx._working_pages -= to_commit
            ctx._working_tokens = new_working % ctx._page_size
            ctx._seq_len += n_auto

        return Output(raw)


# =============================================================================
# Output
# =============================================================================


class Output:
    """Result of one forward-pass execution — produced by both
    :meth:`Forward.execute` and :meth:`GenStep.execute`.

    **Common path** (Generator): read :attr:`tokens` for the accepted
    tokens this step (post stop / max-tokens truncation).

    **Raw Forward**: read sampler slots via :meth:`token` / :meth:`tokens_at`
    using handles returned at attach time. The :attr:`tokens` field is empty.

    **Probes** (both paths): :meth:`distribution` / :meth:`logits` /
    :meth:`logprobs` / :meth:`entropy` take a :class:`ProbeHandle`.

    Mismatched access (reading a sampler slot through a probe handle, or
    vice versa) returns ``None``.
    """

    __slots__ = ("_raw", "tokens", "auto_sampler")

    def __init__(
        self,
        raw: _inf.Output,
        tokens: list[int] | None = None,
        auto_sampler: SampleHandle | None = None,
    ) -> None:
        self._raw = raw
        #: Generator-accepted tokens this step, post stop / max-tokens
        #: truncation. Empty for raw ``Forward.execute()`` (no Generator state).
        self.tokens: list[int] = tokens if tokens is not None else []
        #: Handle for the Generator's auto-attached sampler. ``None`` for
        #: raw Forward results and for steps where ``clear_sampler()``
        #: was called on the GenStep.
        self.auto_sampler: SampleHandle | None = auto_sampler

    @property
    def raw(self) -> _inf.Output:
        """Underlying WIT output (slot list + speculative side channel)."""
        return self._raw

    # ── Sampler accessors ────────────────────────────────────────────

    def token(self, h: SampleHandle) -> int | None:
        """First token from a single-index sampler slot."""
        slot = self._slot(h.slot)
        if isinstance(slot, SlotOutput_Token):
            return slot.value
        return None

    def tokens_at(self, h: SampleHandle) -> list[int]:
        """Tokens at the slot range a multi-index sampler covers. In
        speculative mode the list may be shorter than ``arity`` if the
        verifier rejected drafts."""
        out: list[int] = []
        for i in range(h.arity):
            slot = self._slot(h.slot + i)
            if isinstance(slot, SlotOutput_Token):
                out.append(slot.value)
            else:
                break
        return out

    # ── Probe accessors ──────────────────────────────────────────────

    def distribution(self, h: ProbeHandle) -> tuple[list[int], list[float]] | None:
        """Distribution as ``(ids, probs)`` for a :class:`Distribution` probe."""
        slot = self._slot(h.slot)
        if isinstance(slot, SlotOutput_Distribution):
            ids, probs = slot.value
            return list(ids), list(probs)
        return None

    def logits(self, h: ProbeHandle) -> bytes | None:
        """Raw logits bytes for a :class:`Logits` probe (length
        ``vocab_size * 4``, native-endian f32)."""
        slot = self._slot(h.slot)
        if isinstance(slot, SlotOutput_Logits):
            return bytes(slot.value)
        return None

    def logprobs(self, h: ProbeHandle) -> list[float] | None:
        """Logprob list for a :class:`Logprob` / :class:`Logprobs` probe.
        Length is 1 for a single-token query, K for a list query."""
        slot = self._slot(h.slot)
        if isinstance(slot, SlotOutput_Logprobs):
            return list(slot.value)
        return None

    def entropy(self, h: ProbeHandle) -> float | None:
        """Entropy for an :class:`Entropy` probe."""
        slot = self._slot(h.slot)
        if isinstance(slot, SlotOutput_Entropy):
            return slot.value
        return None

    def _slot(self, idx: int):
        if 0 <= idx < len(self._raw.slots):
            return self._raw.slots[idx]
        return None

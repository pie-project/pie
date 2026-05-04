"""
Context — host-managed conversation state.

Wraps ``pie:core/context``. Buffers tokens via ``system / user /
assistant / cue / seal / append``, drains via ``flush()`` or by handing
the buffer to a :class:`Forward` / :class:`Generator`.

Usage::

    ctx = Context(model)
    ctx.system("You are helpful.")
    ctx.user("Tell me a joke.")

    # Auto-flushed by `generate(...)` — Python convention.
    text = await ctx.generate(Sampler.top_p(0.6, 0.95), max_tokens=256).collect_text()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable

from wit_world.imports import context as _ctx
from wit_world.imports import inference as _inf

from . import chat as _chat
from . import scheduling as _sched
from ._async import await_future
from .forward import Forward
from .generation import Generator
from .grammar import Constraint, Schema
from .model import Model

if TYPE_CHECKING:
    from .adapter import Adapter
    from .sample import Sampler
    from .spec import Speculator


# =============================================================================
# Bid math (internal)
# =============================================================================


def _compute_bid(
    balance: float,
    pages: float,
    mu: float,
    cv2: float,
    page_size: float,
    dividend: float,
) -> float:
    """Budget-exhausting bid formula (matches Rust SDK)."""
    mu = max(mu, 1.0)
    numerator = balance / mu + dividend
    denominator = pages + mu * (1.0 + cv2) / (2.0 * page_size)
    return numerator / denominator if denominator > 0 else numerator


# =============================================================================
# Context
# =============================================================================


class Context:
    """Host-managed conversation context.

    Construct with a model, fill via chat methods (or :meth:`append`),
    then either drain explicitly with :meth:`flush` or let
    :meth:`generate` / :meth:`forward` drain on demand.
    """

    __slots__ = (
        "_handle",
        "_model",
        "_pending_tokens",
        "_page_size",
        "_seq_len",
        "_committed_pages",
        "_working_pages",
        "_working_tokens",
    )

    # ── Construction / lifecycle ──────────────────────────────────────

    def __init__(self, model: Model) -> None:
        self._handle = _ctx.Context.create(model._handle)
        self._model = model
        self._pending_tokens: list[int] = []
        self._sync_from_host()

    def _sync_from_host(self) -> None:
        self._page_size = self._handle.tokens_per_page()
        self._committed_pages = self._handle.committed_page_count()
        self._working_pages = self._handle.working_page_count()
        self._working_tokens = self._handle.working_page_token_count()
        self._seq_len = (
            self._committed_pages * self._page_size + self._working_tokens
        )

    @classmethod
    def open(cls, model: Model, name: str) -> Context | None:
        """Open a saved snapshot (implicit fork — snapshot stays immutable)."""
        raw = _ctx.Context.open(model._handle, name)
        if raw is None:
            return None
        obj = object.__new__(cls)
        obj._handle = raw
        obj._model = model
        obj._pending_tokens = []
        obj._sync_from_host()
        return obj

    @classmethod
    def take(cls, model: Model, name: str) -> Context | None:
        """Take ownership of a snapshot (snapshot is deleted)."""
        raw = _ctx.Context.take(model._handle, name)
        if raw is None:
            return None
        obj = object.__new__(cls)
        obj._handle = raw
        obj._model = model
        obj._pending_tokens = []
        obj._sync_from_host()
        return obj

    @staticmethod
    def delete(model: Model, name: str) -> None:
        """Delete a saved snapshot by name."""
        _ctx.Context.delete(model._handle, name)

    def fork(self) -> Context:
        """Fork into a new anonymous context (working pages copied)."""
        raw = self._handle.fork()
        obj = object.__new__(Context)
        obj._handle = raw
        obj._model = self._model
        obj._pending_tokens = list(self._pending_tokens)
        obj._sync_from_host()
        return obj

    def save(self, name: str) -> None:
        """Save this context with a name."""
        self._handle.save(name)

    def snapshot(self) -> str:
        """Anonymous save — returns a runtime-generated name."""
        return self._handle.snapshot()

    def release(self) -> None:
        """Force-destroy this context immediately."""
        self._handle.destroy()

    def __enter__(self) -> Context:
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"Context({id(self._handle):#x})"

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def model(self) -> Model:
        return self._model

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def seq_len(self) -> int:
        """Total committed + working tokens (excludes the buffer)."""
        return self._seq_len

    def buffer(self) -> list[int]:
        """Pending (buffered but not yet flushed) tokens."""
        return list(self._pending_tokens)

    # ── Chat fillers ─────────────────────────────────────────────────
    #
    # All return self for chaining: ctx.system("...").user("...")

    def system(self, message: str) -> Context:
        """Fill a system-role message."""
        self._pending_tokens.extend(_chat.system(self._model, message))
        return self

    def user(self, message: str) -> Context:
        """Fill a user-role message."""
        self._pending_tokens.extend(_chat.user(self._model, message))
        return self

    def assistant(self, message: str) -> Context:
        """Fill an assistant-role message (history replay)."""
        self._pending_tokens.extend(_chat.assistant(self._model, message))
        return self

    def cue(self) -> Context:
        """Cue the model to generate (fills the generation header)."""
        self._pending_tokens.extend(_chat.cue(self._model))
        return self

    def seal(self) -> Context:
        """Seal the current turn (insert stop token)."""
        self._pending_tokens.extend(_chat.seal(self._model))
        return self

    def append(self, tokens: Iterable[int]) -> Context:
        """Append raw tokens to the buffer directly."""
        self._pending_tokens.extend(tokens)
        return self

    # ── Flush / truncate ─────────────────────────────────────────────

    async def flush(self) -> None:
        """Drain buffered tokens through a forward pass and commit pages.

        After flush, the buffer is empty and ``seq_len`` reflects all
        consumed tokens.
        """
        if not self._pending_tokens:
            return
        tokens = self._pending_tokens
        self._pending_tokens = []
        n = len(tokens)

        # Reserve pages.
        total_after = self._working_tokens + n
        pages_needed = (total_after + self._page_size - 1) // self._page_size
        additional = max(0, pages_needed - self._working_pages)
        if additional > 0:
            self._handle.reserve_working_pages(additional)
            self._working_pages = pages_needed

        # Forward pass without sampler — just write to KV.
        fwd = _inf.ForwardPass(self._model._handle)
        fwd.context(self._handle)
        positions = list(range(self._seq_len, self._seq_len + n))
        fwd.input_tokens(tokens, positions)
        await await_future(fwd.execute(), "Context.flush failed")

        # Commit full pages.
        new_working = self._working_tokens + n
        to_commit = new_working // self._page_size
        if to_commit > 0:
            self._handle.commit_working_pages(to_commit)
        self._committed_pages += to_commit
        self._working_pages -= to_commit
        self._working_tokens = new_working % self._page_size
        self._seq_len += n

    def truncate(self, n: int) -> None:
        """Drop the trailing ``n`` working-page tokens. Use after a
        speculative-decoding pass to roll back the rejected suffix.

        ``n`` counts only working-page tokens — pages already committed
        cannot be truncated through this API.
        """
        if n == 0:
            return
        self._handle.truncate_working_page_tokens(n)
        self._sync_from_host()

    # ── Forward (single forward-pass primitive) ──────────────────────

    def forward(self) -> Forward:
        """Build a single :class:`Forward` — a forward pass with auto
        page reservation, position derivation, and post-execute commit.
        Use for prefill, scoring, custom decode loops, and anywhere the
        :meth:`generate` loop is too high-level."""
        return Forward(self)

    # ── Generate (multi-step loop) ───────────────────────────────────

    def generate(
        self,
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
        auto_flush: bool = True,
    ) -> Generator:
        """Build a :class:`Generator` for token generation.

        **Auto-flush**: when ``auto_flush=True`` (default), this method
        appends ``cue()`` tokens to the buffer before returning the
        Generator. The first ``execute()`` call drains the buffer
        through a forward pass — no separate flush call needed. Pass
        ``auto_flush=False`` to inspect the buffer before generation
        starts (or call ``cue()`` yourself).

        Args:
            sampler: Token-producing sampler.
            max_tokens: Hard cap on tokens generated.
            stop: Stop tokens. Defaults to chat template's stop tokens
                if ``auto_flush=True``; otherwise empty.
            constrain: A :class:`Schema` (compiled to a stateful
                grammar matcher), a :class:`Constraint`, or a list of
                either. Multiple constraints compose by AND-ing their
                per-step BRLE masks.
            logit_mask: Static BRLE mask applied every step. Composes
                with ``constrain`` like any other constraint.
            speculator: Custom speculative-decoding drafter implementing
                the :class:`Speculator` protocol.
            system_speculation: If True, the runtime drives drafts via
                its built-in NGRAM/etc. drafter. Mutually exclusive with
                ``speculator``.
            adapter: Adapter to apply on every forward pass.
            zo_seed: Evolution Strategies seed for every forward pass.
            horizon: Expected output length for budget planning.
            auto_flush: Append ``cue()`` and use chat stop tokens by default.
        """
        if auto_flush:
            self.cue()
            if stop is None:
                stop = _chat.stop_tokens(self._model)

        return Generator(
            self,
            sampler,
            max_tokens=max_tokens,
            stop=stop,
            constrain=constrain,
            logit_mask=logit_mask,
            speculator=speculator,
            system_speculation=system_speculation,
            adapter=adapter,
            zo_seed=zo_seed,
            horizon=horizon,
        )

    # ── Bidding / scheduling ─────────────────────────────────────────

    def set_bid(self, value: float) -> None:
        """Override the auto-computed bid (willingness to pay per page
        per step). Most callers should NOT use this — the
        :class:`Generator` auto-bids each step."""
        self._handle.bid(value)

    @contextmanager
    def idle(self):
        """Mark this context as idle: drop the bid to zero so other
        contexts can take its pages under contention. Yields a context
        manager that restores the truthful generation bid on exit::

            with ctx.idle():
                result = await http_get(url)
            # bid restored

        On an uncontended device the runtime charges zero rent anyway —
        ``idle`` is a no-op cost-wise but still safe to call. Under
        load, it yields priority to other workloads for the duration.
        """
        # Snapshot the truthful generation bid to restore on exit.
        pages = float(self._committed_pages + self._working_pages)
        if pages > 0.0:
            balance = _sched.balance(self._model)
            dividend = _sched.dividend(self._model)
            saved = _compute_bid(
                balance, pages, 4096.0, 1.0, float(self._page_size), dividend
            )
        else:
            saved = 0.0

        self._handle.bid(0.0)
        try:
            yield
        finally:
            self._handle.bid(saved)

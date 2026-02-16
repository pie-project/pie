"""Context, TokenStream, EventStream — the core of the inferlet SDK.

Usage::

    from inferlet import Model, Context, Sampler, Event, runtime

    model = Model.load(runtime.models()[0])
    ctx = Context(model)

    ctx.system("You are helpful.")
    ctx.user("Hello!")

    # One-shot generation
    text = await ctx.generate_text(Sampler.top_p())

    # Streaming with events
    async for event in await ctx.generate(Sampler.top_p(), decode=True):
        match event:
            case Event.Text(text=t):
                print(t, end="")
            case Event.Done():
                break
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from wit_world.imports import context as _ctx
from wit_world.imports import inference as _inf
from wit_world.imports import chat as _chat
from wit_world.imports import tool_use as _tool
from wit_world.imports import reasoning as _reasoning

from ._async import await_future
from .model import Model, Tokenizer
from .sampler import Sampler

# Global counter for auto-generated context names
_NAME_SEQ = 0


def _next_name() -> str:
    global _NAME_SEQ
    _NAME_SEQ += 1
    return f"ctx-{_NAME_SEQ:08x}"


# =============================================================================
# Event types
# =============================================================================


class Event:
    """Discriminated union of generation events.

    Match with ``match``/``case``::

        match event:
            case Event.Text(text=t):
                print(t, end="")
            case Event.Done(text=full):
                break
    """

    __slots__ = ()

    @dataclass(slots=True)
    class Text:
        """A chunk of generated text."""
        text: str

    @dataclass(slots=True)
    class Thinking:
        """A chunk of reasoning / thinking text."""
        text: str

    @dataclass(slots=True)
    class ThinkingDone:
        """Reasoning complete — ``text`` is the full accumulated reasoning."""
        text: str

    @dataclass(slots=True)
    class ToolCallStart:
        """A tool call has been detected (more tokens expected)."""
        pass

    @dataclass(slots=True)
    class ToolCall:
        """A complete tool call."""
        name: str
        arguments: str

    @dataclass(slots=True)
    class Done:
        """Generation complete — ``text`` is the full accumulated text."""
        text: str


# Type alias for event instances
AnyEvent = (
    Event.Text
    | Event.Thinking
    | Event.ThinkingDone
    | Event.ToolCallStart
    | Event.ToolCall
    | Event.Done
)


# =============================================================================
# Decoder (unified mux of chat + reasoning + tool-use decoders)
# =============================================================================


class Decoder:
    """Unified decoder muxing chat, reasoning, and tool-use WIT decoders.

    Event priority:
    1. Reasoning transitions (start/complete) override chat deltas.
    2. Tool calls override chat deltas.
    3. Chat done is always forwarded.
    4. Otherwise, chat deltas are forwarded as ``Event.Text``.
    """

    __slots__ = ("_chat_dec", "_reasoning_dec", "_tool_dec", "_in_reasoning")

    def __init__(
        self,
        model: Model,
        *,
        reasoning: bool = False,
        tool_use: bool = False,
    ) -> None:
        self._chat_dec = _chat.create_decoder(model._handle)
        self._reasoning_dec = _reasoning.create_decoder(model._handle) if reasoning else None
        self._tool_dec = _tool.create_decoder(model._handle) if tool_use else None
        self._in_reasoning = False

    def feed(self, tokens: list[int]) -> AnyEvent:
        """Feed a batch of token IDs and return a unified event."""
        # Always feed the chat decoder
        chat_event = self._chat_dec.feed(tokens)

        # Check reasoning first (highest priority)
        if self._reasoning_dec is not None:
            reasoning_event = self._reasoning_dec.feed(tokens)
            if isinstance(reasoning_event, _reasoning.Event_Start):
                self._in_reasoning = True
                return Event.Thinking("")
            elif isinstance(reasoning_event, _reasoning.Event_Delta):
                if self._in_reasoning and reasoning_event.value:
                    return Event.Thinking(reasoning_event.value)
            elif isinstance(reasoning_event, _reasoning.Event_Complete):
                self._in_reasoning = False
                return Event.ThinkingDone(reasoning_event.value)

        # Check tool use (second priority)
        if self._tool_dec is not None:
            tool_event = self._tool_dec.feed(tokens)
            if isinstance(tool_event, _tool.Event_Start):
                return Event.ToolCallStart()
            elif isinstance(tool_event, _tool.Event_Call):
                name, args = tool_event.value
                return Event.ToolCall(name=name, arguments=args)

        # Fall through to chat events
        if isinstance(chat_event, _chat.Event_Done):
            return Event.Done(chat_event.value)
        elif isinstance(chat_event, _chat.Event_Delta):
            return Event.Text(chat_event.value)
        elif isinstance(chat_event, _chat.Event_Interrupt):
            return Event.Done("")

        # Should not reach here
        return Event.Text("")

    def reset(self) -> None:
        """Reset all sub-decoders."""
        self._chat_dec.reset()
        self._in_reasoning = False
        if self._reasoning_dec is not None:
            self._reasoning_dec.reset()
        if self._tool_dec is not None:
            self._tool_dec.reset()


# =============================================================================
# Page management helpers
# =============================================================================


async def _reserve_and_run(
    ctx_handle,
    model_handle,
    tokens: list[int],
    seq_start: int,
    cursor: int,
    page_size: int,
    sampler_variant=None,
    logit_mask: list[int] | None = None,
):
    """Reserve pages, run a forward pass, commit pages, update cursor.

    If ``sampler_variant`` is provided, sampling is applied at the last
    position.  Returns the Output (or None if no sampler was given).
    """
    num_tokens = len(tokens)

    # Reserve pages
    total_tokens = cursor + num_tokens
    pages_needed = (total_tokens + page_size - 1) // page_size
    if pages_needed > 0:
        ctx_handle.reserve_pages(pages_needed)

    # Build forward pass
    fwd = _inf.ForwardPass(model_handle)
    fwd.context(ctx_handle)
    fwd.input_tokens(tokens, list(range(seq_start, seq_start + num_tokens)))

    if sampler_variant is not None:
        fwd.sampler([num_tokens - 1], sampler_variant)

    if logit_mask is not None:
        fwd.logit_mask(logit_mask)

    # Execute
    future = fwd.execute()
    output = await await_future(future, "Forward pass failed")

    # Commit pages and update cursor
    new_cursor_abs = cursor + num_tokens
    pages_to_commit = new_cursor_abs // page_size
    if pages_to_commit > 0:
        ctx_handle.commit_pages(list(range(pages_to_commit)))

    ctx_handle.set_cursor(new_cursor_abs % page_size)

    return output


# =============================================================================
# TokenStream
# =============================================================================


class TokenStream:
    """Async iterator of generated token batches.

    Created by ``Context.generate(sampler)``. Handles the full generation
    loop internally: page reservation, forward pass, page commit, cursor
    update.

    Usage::

        async for tokens in await ctx.generate(Sampler.top_p(), max_tokens=256):
            print(tokens)
    """

    __slots__ = (
        "_ctx",
        "_model_handle",
        "_page_size",
        "_stop_tokens",
        "_sampler",
        "_logit_mask",
        "_done",
        "_max_tokens",
        "_tokens_generated",
    )

    def __init__(
        self,
        ctx: Context,
        sampler: Sampler,
        *,
        max_tokens: int | None = None,
        logit_mask: list[int] | None = None,
    ) -> None:
        self._ctx = ctx
        self._model_handle = ctx._handle.model()
        self._page_size: int = ctx._handle.tokens_per_page()
        self._stop_tokens: frozenset[int] = frozenset(_chat.stop_tokens(self._model_handle))
        self._sampler = sampler
        self._logit_mask = logit_mask
        self._done = False
        self._max_tokens = max_tokens
        self._tokens_generated = 0

    # --- Generation ---

    async def _step(self) -> list[int]:
        """One generation step: forward pass → sample → commit."""
        raw = self._ctx._handle
        buffered = raw.buffered_tokens()
        if not buffered:
            raise RuntimeError("generate requires at least one buffered token")

        last_pos = raw.last_position()
        seq_start = (last_pos + 1) if last_pos is not None else 0

        output = await _reserve_and_run(
            raw,
            self._model_handle,
            buffered,
            seq_start,
            raw.cursor(),
            self._page_size,
            sampler_variant=self._sampler._variant,
            logit_mask=self._logit_mask,
        )

        # Extract tokens from output
        new_tokens = self._extract_tokens(output)
        if not new_tokens:
            return []

        # Set the last generated token as the next input
        raw.set_buffered_tokens([new_tokens[-1]])

        return new_tokens

    @staticmethod
    def _extract_tokens(output) -> list[int]:
        """Pull token IDs from the ``Output`` variant."""
        if isinstance(output, _inf.Output_Tokens):
            return list(output.value)
        elif isinstance(output, _inf.Output_TokensWithSpeculation):
            accepted, _, _ = output.value
            return list(accepted)
        return []

    # --- Convenience ---

    async def text(self) -> str:
        """Consume the stream, decode, and return full text."""
        tokenizer = Tokenizer(self._model_handle.tokenizer())
        return tokenizer.decode(await self.tokens())

    async def tokens(self) -> list[int]:
        """Consume the stream and return all tokens as a flat list."""
        all_tokens: list[int] = []
        async for batch in self:
            all_tokens.extend(batch)
        return all_tokens

    # Deprecated aliases
    collect_text = text
    collect_tokens = tokens

    # --- Iterator protocol ---

    def __aiter__(self) -> AsyncIterator[list[int]]:
        return self

    async def __anext__(self) -> list[int]:
        if self._done:
            raise StopAsyncIteration

        if self._max_tokens is not None and self._tokens_generated >= self._max_tokens:
            raise StopAsyncIteration

        tokens = await self._step()
        if not tokens:
            raise StopAsyncIteration

        # Truncate at stop token
        for i, t in enumerate(tokens):
            if t in self._stop_tokens:
                tokens = tokens[:i]
                self._done = True
                if not tokens:
                    raise StopAsyncIteration
                break

        # Enforce max tokens
        if self._max_tokens is not None:
            remaining = self._max_tokens - self._tokens_generated
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
                self._done = True

        self._tokens_generated += len(tokens)
        return tokens


# =============================================================================
# EventStream
# =============================================================================


class EventStream:
    """Async decoded event stream wrapping a ``TokenStream`` + ``Decoder``.

    Created by ``Context.generate(sampler, decode=True)``.

    Usage::

        async for event in await ctx.generate(Sampler.top_p(), decode=True, reasoning=True):
            match event:
                case Event.Text(text=t):
                    print(t, end="")
    """

    __slots__ = ("_stream", "_decoder")

    def __init__(
        self,
        stream: TokenStream,
        *,
        reasoning: bool = False,
        tool_use: bool = False,
    ) -> None:
        self._stream = stream
        model = Model(stream._model_handle)
        self._decoder = Decoder(model, reasoning=reasoning, tool_use=tool_use)

    def __aiter__(self) -> AsyncIterator[AnyEvent]:
        return self

    async def __anext__(self) -> AnyEvent:
        tokens = await self._stream.__anext__()
        return self._decoder.feed(tokens)


# =============================================================================
# Context
# =============================================================================


class Context:
    """Host-managed conversation context.

    Wraps ``pie:core/context`` and ``pie:instruct/*``.

    Usage::

        ctx = Context(model)
        ctx.system("You are helpful.")
        ctx.user("Tell me a joke.")

        async for event in await ctx.generate(Sampler.top_p(), decode=True):
            ...
    """

    __slots__ = ("_handle", "_model")

    def __init__(
        self,
        model: Model,
        name: str | None = None,
        fill: list[int] | None = None,
    ) -> None:
        """Create a new context.

        Args:
            model: The model to create a context for.
            name: Optional name (auto-generated if omitted).
            fill: Optional initial token IDs.
        """
        ctx_name = name or _next_name()
        self._handle = _ctx.Context.create(model._handle, ctx_name, fill)
        self._model = model

    @classmethod
    def lookup(cls, model: Model, name: str) -> Context | None:
        """Look up an existing context by name."""
        raw = _ctx.Context.lookup(model._handle, name)
        if raw is None:
            return None
        obj = object.__new__(cls)
        obj._handle = raw
        obj._model = model
        return obj

    # --- Fill (ContextExt) ---

    def fill(self, text: str) -> None:
        """Fill the context buffer with text (encodes to tokens)."""
        tokenizer = self._model.tokenizer()
        tokens = tokenizer.encode(text)
        self._handle.append_buffered_tokens(tokens)

    def fill_tokens(self, tokens: list[int]) -> None:
        """Fill the context buffer with raw token IDs."""
        self._handle.append_buffered_tokens(tokens)

    async def flush(self) -> None:
        """Flush buffered tokens: run forward pass and commit pages.

        Processes all buffered tokens except the last one (which stays
        as the seed for the next generation step).
        """
        buffered = self._handle.buffered_tokens()
        if len(buffered) <= 1:
            return

        tokens_to_flush = buffered[:-1]
        last_token = buffered[-1]

        last_pos = self._handle.last_position()
        seq_start = (last_pos + 1) if last_pos is not None else 0

        await _reserve_and_run(
            self._handle,
            self._model._handle,
            tokens_to_flush,
            seq_start,
            self._handle.cursor(),
            self._handle.tokens_per_page(),
        )

        self._handle.set_buffered_tokens([last_token])

    # --- Instruct (pie:instruct/chat) ---

    def system(self, message: str) -> None:
        """Fill a system message."""
        _chat.system(self._handle, message)

    def user(self, message: str) -> None:
        """Fill a user message."""
        _chat.user(self._handle, message)

    def assistant(self, message: str) -> None:
        """Fill a (previous) assistant message."""
        _chat.assistant(self._handle, message)

    def cue(self) -> None:
        """Cue the model to generate (fills the generation header)."""
        _chat.cue(self._handle)

    def seal(self) -> None:
        """Seal the current turn (inserts stop token)."""
        _chat.seal(self._handle)

    def stop_tokens(self) -> list[int]:
        """Get the stop token IDs for this model."""
        return list(_chat.stop_tokens(self._model._handle))

    # --- Tool Use (pie:instruct/tool-use) ---

    def equip_tools(self, tools: list[str]) -> None:
        """Register available tools (list of JSON schema strings)."""
        _tool.equip(self._handle, tools)

    def answer_tool(self, name: str, value: str) -> None:
        """Provide a tool call result."""
        _tool.answer(self._handle, name, value)

    # --- Generate (ContextExt) ---

    async def generate(
        self,
        sampler: Sampler,
        *,
        max_tokens: int | None = None,
        logit_mask: list[int] | None = None,
        decode: bool = False,
        reasoning: bool = False,
        tool_use: bool = False,
        auto_flush: bool = True,
    ) -> TokenStream | EventStream:
        """Start generating tokens.

        By default, calls ``cue()`` + ``flush()`` before starting
        the generation stream. Set ``auto_flush=False`` to skip this.

        Args:
            sampler: The sampling strategy to use.
            max_tokens: Maximum tokens to generate.
            logit_mask: Optional BRLE logit mask.
            decode: If ``True``, return an ``EventStream`` instead of ``TokenStream``.
            reasoning: Enable reasoning/thinking decoding (requires ``decode=True``).
            tool_use: Enable tool-use decoding (requires ``decode=True``).
            auto_flush: Auto-call ``cue()`` + ``flush()`` before generation (default ``True``).

        Returns:
            ``TokenStream`` (default) or ``EventStream`` (when ``decode=True``).
        """
        if auto_flush:
            self.cue()
            await self.flush()

        stream = TokenStream(self, sampler, max_tokens=max_tokens, logit_mask=logit_mask)

        if decode or reasoning or tool_use:
            return EventStream(stream, reasoning=reasoning, tool_use=tool_use)

        return stream

    async def generate_text(
        self,
        sampler: Sampler,
        *,
        max_tokens: int | None = None,
        logit_mask: list[int] | None = None,
        auto_flush: bool = True,
    ) -> str:
        """Generate and return the full response text (one-shot).

        Convenience wrapper that calls ``generate(decode=True)`` and
        collects all ``Event.Text`` chunks into a single string.
        """
        parts: list[str] = []
        async for event in await self.generate(
            sampler,
            max_tokens=max_tokens,
            logit_mask=logit_mask,
            decode=True,
            auto_flush=auto_flush,
        ):
            if isinstance(event, Event.Text):
                parts.append(event.text)
            elif isinstance(event, Event.Done):
                break
        return "".join(parts)

    # --- Lifecycle ---

    def fork(self, name: str | None = None) -> Context:
        """Fork this context into a new one.

        The forked context shares committed KV pages with the parent.
        """
        fork_name = name or _next_name()
        raw = self._handle.fork(fork_name)
        obj = object.__new__(Context)
        obj._handle = raw
        obj._model = self._model
        return obj

    def release(self) -> None:
        """Explicitly release this context's resources."""
        self._handle.destroy()

    def __enter__(self) -> Context:
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"Context({id(self._handle):#x})"

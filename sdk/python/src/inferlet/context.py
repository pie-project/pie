"""
Context — SDK-owned facade over the runtime KV working set.

The runtime's opaque context resource has been replaced by an inferlet-owned
``kv-working-set`` plus explicit forward-pass read/write descriptors.  This
class owns the semantic metadata (buffer, token positions, sequence length, and
replay history) and mirrors the Rust SDK facade.

Single-model runtime: the working set binds to the one bound model implicitly
(``KvWorkingSet()`` takes no handle) and model metadata is reached through the
global ``wit_world.imports.model`` / ``chat`` functions.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Iterable

from wit_world.imports import chat as _wit_chat
from wit_world.imports import inference as _inf
from wit_world.imports import media as _media
from wit_world.imports import model as _model
from wit_world.imports import runtime as _runtime
from wit_world.imports import working_set as _ws

from . import chat as _chat
from .forward import Forward
from .generation import Generator
from .grammar import Constraint, Schema

if TYPE_CHECKING:
    from .adapter import Adapter
    from .sample import Sampler
    from .spec import Speculator


_SNAPSHOT_VERSION = 1
_SNAPSHOT_COUNTER = 0


def _snapshot_path(name: str) -> str:
    # The runtime preopens the per-instance scratch dir as `/scratch` in the
    # guest; a relative path has no matching preopen, so blobs go there.
    return f"/scratch/{name}.pie-snapshot"


def _read_manifest(name: str) -> dict:
    with open(_snapshot_path(name), encoding="utf-8") as f:
        manifest = json.load(f)
    version = manifest.get("version")
    if version != _SNAPSHOT_VERSION:
        raise RuntimeError(
            f"snapshot '{name}': version {version} unsupported "
            f"(expected {_SNAPSHOT_VERSION})"
        )
    return manifest


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


class Context:
    """High-level inference context backed by a KV working set."""

    __slots__ = (
        "_kv",
        "_page_size",
        "_buffer",
        "_pending_system",
        "_seq_len",
        "_history",
        "_snapshottable",
    )

    # ── Construction / lifecycle ──────────────────────────────────────

    def __init__(self) -> None:
        kv = _ws.KvWorkingSet()
        self._kv = kv
        self._page_size = kv.page_size()
        self._buffer: list[int] = []
        self._pending_system: str | None = None
        self._seq_len = 0
        self._history: list[int] = []
        self._snapshottable = True

    def fork(self) -> Context:
        """Fork into a new anonymous context sharing KV pages by CoW."""
        kv = self._kv.fork()
        obj = object.__new__(Context)
        obj._kv = kv
        obj._page_size = self._page_size
        obj._buffer = list(self._buffer)
        obj._pending_system = self._pending_system
        obj._seq_len = self._seq_len
        obj._history = list(self._history)
        obj._snapshottable = self._snapshottable
        return obj

    def release(self) -> None:
        """Force-release this context's working set handle."""
        self._kv.__exit__(None, None, None)

    def __enter__(self) -> Context:
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"Context({id(self._kv):#x})"

    # ── Snapshots ─────────────────────────────────────────────────────

    def save(self, name: str) -> None:
        """Save this context as a replayable CPU-resident snapshot."""
        if not self._snapshottable:
            raise RuntimeError(
                "Context.save: multimodal contexts are not snapshottable in v1 "
                "(soft-token KV cannot be replayed from a token log)"
            )
        manifest = {
            "version": _SNAPSHOT_VERSION,
            "page_size": self._page_size,
            "seq_len": self._seq_len,
            "tokens": list(self._history),
            "buffer": list(self._buffer),
            "pending_system": self._pending_system,
            "cas_hashes": [],
        }
        with open(_snapshot_path(name), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

    def snapshot(self) -> str:
        """Anonymous save — returns a freshly-generated snapshot name."""
        global _SNAPSHOT_COUNTER
        name = f"anon-{_runtime.instance_id()}-{_SNAPSHOT_COUNTER}"
        _SNAPSHOT_COUNTER += 1
        self.save(name)
        return name

    @classmethod
    async def open(cls, name: str) -> Context:
        """Open a saved snapshot, replaying its token log into a fresh set."""
        manifest = _read_manifest(name)
        return await cls._from_manifest(manifest)

    @classmethod
    async def take(cls, name: str) -> Context:
        """Open a saved snapshot and then best-effort delete it."""
        ctx = await cls.open(name)
        try:
            os.remove(_snapshot_path(name))
        except OSError:
            pass
        return ctx

    @staticmethod
    def delete(name: str) -> None:
        """Delete a saved snapshot by name. Missing snapshots are ignored."""
        try:
            os.remove(_snapshot_path(name))
        except OSError:
            pass

    @classmethod
    async def _from_manifest(cls, manifest: dict) -> Context:
        ctx = cls()
        tokens = list(manifest.get("tokens", []))
        if tokens:
            ctx._buffer = tokens
            await ctx.flush()
        ctx._buffer = list(manifest.get("buffer", []))
        ctx._pending_system = manifest.get("pending_system")
        return ctx

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def seq_len(self) -> int:
        """Total materialized tokens (excludes the buffer)."""
        return self._seq_len

    def buffer(self) -> list[int]:
        """Pending (buffered but not yet flushed) tokens."""
        return list(self._buffer)

    def working_set(self) -> "_ws.KvWorkingSet":
        """Escape hatch: the underlying KV working set (power users)."""
        return self._kv

    # ── Chat fillers ─────────────────────────────────────────────────

    def _flush_pending_system(self) -> None:
        if self._pending_system is not None:
            self._buffer.extend(_chat.system(self._pending_system))
            self._pending_system = None

    def _is_first_chat_fill(self) -> bool:
        return self._seq_len == 0 and not self._buffer

    def system(self, message: str) -> Context:
        """Fill a system-role message."""
        self._flush_pending_system()
        self._pending_system = message
        return self

    def user(self, message: str) -> Context:
        """Fill a user-role message."""
        if self._pending_system is not None:
            tokens = list(_wit_chat.system_user(self._pending_system, message))
            self._pending_system = None
        elif self._is_first_chat_fill():
            tokens = list(_wit_chat.first_user(message))
        else:
            tokens = _chat.user(message)
        self._buffer.extend(tokens)
        return self

    def assistant(self, message: str) -> Context:
        """Fill an assistant-role message (history replay)."""
        self._flush_pending_system()
        self._buffer.extend(_chat.assistant(message))
        return self

    def cue(self) -> Context:
        """Cue the model to generate (fills the generation header)."""
        self._flush_pending_system()
        self._buffer.extend(_chat.cue())
        return self

    def seal(self) -> Context:
        """Seal the current turn (insert stop token)."""
        self._flush_pending_system()
        self._buffer.extend(_chat.seal())
        return self

    def append(self, tokens: Iterable[int]) -> Context:
        """Append raw tokens to the buffer directly."""
        self._flush_pending_system()
        self._buffer.extend(tokens)
        return self

    # ── Sequence / page bookkeeping ──────────────────────────────────

    def truncate(self, n: int) -> None:
        """Drop the trailing ``n`` materialized tokens and free empty slots."""
        n = min(max(n, 0), self._seq_len)
        if n == 0:
            return
        self._seq_len -= n
        keep = max(0, len(self._history) - n)
        del self._history[keep:]

        live_pages = _ceil_div(self._seq_len, self._page_size)
        have = self._kv.size()
        if have > live_pages:
            try:
                self._kv.free(list(range(live_pages, have)))
            except Exception:
                pass

    def _prepare_write(self, n: int) -> tuple[int, list[int], list[int], int]:
        p = self._page_size
        first_write_page = self._seq_len // p
        total_after = self._seq_len + n
        total_pages = _ceil_div(total_after, p)
        have = self._kv.size()
        if total_pages > have:
            self._kv.alloc(total_pages - have)
        generation = self._kv.generation()
        indices = list(range(first_write_page, total_pages))
        valid_lens = [min(total_after - pg * p, p) for pg in indices]
        return generation, indices, valid_lens, first_write_page

    def _attach_kv(
        self,
        fwd: _inf.ForwardPass,
        generation: int,
        indices: list[int],
        valid_lens: list[int],
        ctx_pages: int,
    ) -> None:
        if ctx_pages > 0:
            fwd.kv_context(
                _inf.KvContext(
                    set=self._kv,
                    start=0,
                    len=ctx_pages,
                    valid_tokens=ctx_pages * self._page_size,
                )
            )
        fwd.kv_output(
            _inf.KvOutput(
                set=self._kv,
                generation=generation,
                indices=indices,
                per_page_valid_lens=valid_lens,
            )
        )

    def _attach_full_context(self, fwd: _inf.ForwardPass) -> None:
        ctx_pages = _ceil_div(self._seq_len, self._page_size)
        if ctx_pages > 0:
            fwd.kv_context(
                _inf.KvContext(
                    set=self._kv,
                    start=0,
                    len=ctx_pages,
                    valid_tokens=self._seq_len,
                )
            )

    # ── Flush / multimodal append ────────────────────────────────────

    async def flush(self) -> None:
        """Drain buffered tokens through a forward pass into KV slots."""
        self._flush_pending_system()
        if not self._buffer:
            return
        tokens = self._buffer
        self._buffer = []
        n = len(tokens)
        positions = list(range(self._seq_len, self._seq_len + n))
        generation, indices, valid_lens, ctx_pages = self._prepare_write(n)

        fwd = _inf.ForwardPass()
        self._attach_kv(fwd, generation, indices, valid_lens, ctx_pages)
        fwd.input_tokens(tokens, positions)
        await fwd.execute()

        self._history.extend(tokens)
        self._seq_len += n

    async def append_image(self, image: _media.Image) -> None:
        """Splice an encoded image/video frame into the context."""
        prefix = list(image.prefix_tokens())
        suffix = list(image.suffix_tokens())
        if prefix:
            self.append(prefix)
        await self.flush()

        num_tokens = image.token_count()
        if num_tokens == 0:
            if suffix:
                self.append(suffix)
            return

        generation, indices, valid_lens, ctx_pages = self._prepare_write(num_tokens)
        fwd = _inf.ForwardPass()
        self._attach_kv(fwd, generation, indices, valid_lens, ctx_pages)
        fwd.input_image(image, self._seq_len)
        await fwd.execute()

        self._seq_len += num_tokens
        self._snapshottable = False
        if suffix:
            self.append(suffix)

    async def append_audio(self, audio: _media.Audio) -> None:
        """Splice an encoded audio clip into the context."""
        prefix = list(audio.prefix_tokens())
        suffix = list(audio.suffix_tokens())
        if prefix:
            self.append(prefix)
        await self.flush()

        num_tokens = audio.token_count()
        if num_tokens == 0:
            if suffix:
                self.append(suffix)
            return

        generation, indices, valid_lens, ctx_pages = self._prepare_write(num_tokens)
        fwd = _inf.ForwardPass()
        self._attach_kv(fwd, generation, indices, valid_lens, ctx_pages)
        fwd.input_audio(audio, self._seq_len)
        await fwd.execute()

        self._seq_len += num_tokens
        self._snapshottable = False
        if suffix:
            self.append(suffix)

    async def append_video(self, video: _media.Video) -> None:
        """Splice a decoded video clip into the context frame-by-frame."""
        for i in range(video.frame_count()):
            secs = max(video.timestamp(i), 0.0)
            marker = f" {int(secs) // 60:02}:{int(secs) % 60:02} "
            self.append(_model.encode(marker))
            await self.append_image(video.frame(i))

    # ── Forward (single forward-pass primitive) ──────────────────────

    def forward(self) -> Forward:
        """Build a single :class:`Forward` pass."""
        self._flush_pending_system()
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
        system_speculation: bool | None = None,
        adapter: Adapter | None = None,
        zo_seed: int | None = None,
        horizon: int | None = None,
        auto_flush: bool = True,
    ) -> Generator:
        """Build a :class:`Generator` for token generation."""
        if auto_flush:
            self.cue()
            if stop is None:
                stop = _chat.stop_tokens()

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

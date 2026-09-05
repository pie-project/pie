"""
The author-facing ETA bridge over the WIT forward surface — port of
`crates/inferlet/src/eta.rs`.

`ForwardPass` wraps the `pie:inferlet/forward*` resources and drives the
neutral `Builder`, lowering author stage closures to the ETA container. A
`Channel` owns both the trace declaration and the WIT resource.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence, TypeAlias

from componentize_py_types import Err as _WitErr
from wit_world.imports import channel as _wit_channel
from wit_world.imports import forward as _wit_attention
from wit_world.imports import forward_hybrid as _wit_hybrid
from wit_world.imports import forward_recurrent as _wit_recurrent
from wit_world.imports import model as _wit_model
from wit_world.imports import pie_inferlet_types as _wit_types
from wit_world.imports import pipeline as _wit_pipeline
from wit_world.imports import working_set as _wit_ws

from ..media import Audio, Image
from .builder import Builder, DslChannel
from .ir import Dtype, Port, Shape, Stage, numel, shape_of
from .trace import TraceError
from .value import ConstData, Tensor, const_data, unpack_elems

ForwardKind = _wit_model.ForwardKind


class InferletError(Exception):
    """A host-side refusal (`pie:inferlet` declares `type error = string`)."""


def _wit(fn: Callable, *args, what: str = ""):
    """Call a WIT binding, turning its `Err` into an `InferletError` that
    says where it came from."""
    try:
        return fn(*args)
    except _WitErr as e:
        msg = e.value if isinstance(e.value, str) else repr(e.value)
        raise InferletError(f"{what}: {msg}" if what else msg) from None


async def _wit_async(coro: Awaitable, what: str = ""):
    try:
        return await coro
    except _WitErr as e:
        msg = e.value if isinstance(e.value, str) else repr(e.value)
        raise InferletError(f"{what}: {msg}" if what else msg) from None


# ---------------------------------------------------------------------------
# gid -> WIT channel registry
# ---------------------------------------------------------------------------

_WIT_CHANNELS: dict[int, Any] = {}
_DECLARED: dict[int, tuple[list[int], Any, int]] = {}

_WIT_DTYPE = {
    Dtype.F32: _wit_types.Dtype.F32,
    Dtype.I32: _wit_types.Dtype.I32,
    Dtype.U32: _wit_types.Dtype.U32,
    Dtype.BOOL: _wit_types.Dtype.BOOL,
}


def _declare_channel(gid: int, dims: Shape, dt: Dtype, capacity: int) -> None:
    _DECLARED[gid] = (list(dims), _WIT_DTYPE[dt], capacity)


def _channel_exists(gid: int) -> bool:
    return gid in _WIT_CHANNELS


def _set_declared_capacity(gid: int, capacity: int) -> None:
    spec = _DECLARED.get(gid)
    if spec is not None:
        _DECLARED[gid] = (spec[0], spec[1], capacity)


def _lookup_channel(gid: int):
    """The WIT handle for `gid`, creating it from the declaration on first ask."""
    wit = _WIT_CHANNELS.get(gid)
    if wit is not None:
        return wit
    spec = _DECLARED.get(gid)
    if spec is None:
        raise TraceError(f"channel gid {gid} was never declared")
    wit = _wit_channel.Channel(spec[0], spec[1], spec[2])
    _WIT_CHANNELS[gid] = wit
    return wit


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

TOKEN_PAD = -1


def pad_tokens(tokens: Sequence[int], envelope: int) -> list[int]:
    """Pad a token window to `envelope` slots with `TOKEN_PAD`."""
    if len(tokens) > envelope:
        raise ValueError(f"window of {len(tokens)} tokens exceeds its envelope of {envelope}")
    return [int(t) for t in tokens] + [TOKEN_PAD] * (envelope - len(tokens))


def unpad_tokens(window: Sequence[int]) -> list[int]:
    return [int(t) for t in window if t != TOKEN_PAD]


class Channel:
    """A GPU-resident bounded queue, backing both the trace and the WIT
    `channel` resource.

    ``Channel([1], dtype.i32)`` declares an empty capacity-1 channel;
    ``Channel.from_([0, 1], dtype.u32)`` a channel seeded full with a value.
    Inside a stage body, ``take()``/``read()``/``put(tensor)`` record device
    ops; on the host, ``put(data)`` stages a cell and ``await take_host()``
    reads one back.
    """

    __slots__ = ("gid", "_shape", "_dtype")

    def __init__(self, shape=None, dtype: Dtype | None = None, *, _gid: int | None = None):
        if _gid is not None:
            self.gid = _gid
            self._shape, self._dtype = self._dsl().shape, self._dsl().dtype
            return
        if shape is None or dtype is None:
            raise TypeError("Channel(shape, dtype)")
        ch = Channel._build(shape_of(shape), dtype, False)
        self.gid, self._shape, self._dtype = ch.gid, ch._shape, ch._dtype

    @staticmethod
    def _build(shape: Shape, dt: Dtype, seeded: bool, seed: ConstData | None = None) -> "Channel":
        if seed is not None:
            dsl = DslChannel.from_const(seed)
        elif seeded:
            dsl = DslChannel.seeded(shape, dt)
        else:
            dsl = DslChannel.new(shape, dt)
        _declare_channel(dsl.gid, shape, dt, 1)
        ch = object.__new__(Channel)
        ch.gid, ch._shape, ch._dtype = dsl.gid, shape, dt
        return ch

    @staticmethod
    def writer(shape, dtype: Dtype) -> "Channel":
        """An initially empty channel whose producer is the host."""
        ch = Channel._build(shape_of(shape), dtype, False)
        ch._dsl().note_host_put()
        return ch

    @staticmethod
    def from_(v, dtype: Dtype | None = None) -> "Channel":
        """A channel seeded full with the per-instance value `v` (rides as a
        pre-submit `put`, never the container)."""
        data = const_data(v, dtype)
        ch = Channel._build(data.shape, data.dtype, True, data)
        _wit(ch._wit().put, data.data, what="stage seed on a fresh channel")
        return ch

    @staticmethod
    def from_shaped(shape, v, dtype: Dtype | None = None) -> "Channel":
        """Like `from_`, but reinterprets the flat seed under `shape`."""
        data = const_data(v, dtype)
        shape = shape_of(shape)
        if numel(shape) != numel(data.shape):
            raise ValueError("from_shaped: element count mismatch")
        data = ConstData(shape, data.dtype, data.data)
        ch = Channel._build(shape, data.dtype, True, data)
        _wit(ch._wit().put, data.data, what="stage seed on a fresh channel")
        return ch

    @staticmethod
    def seeded(shape, dtype: Dtype) -> "Channel":
        """A seeded channel whose seed value is supplied at instantiation."""
        return Channel._build(shape_of(shape), dtype, True)

    def _dsl(self) -> DslChannel:
        dsl = DslChannel.by_gid(self.gid)
        if dsl is None:
            raise TraceError(f"channel gid {self.gid} is not registered")
        return dsl

    def _wit(self):
        return _lookup_channel(self.gid)

    def capacity(self, n: int) -> "Channel":
        """Widen the ring to `n` cells (deeper run-ahead). Must precede first use."""
        if _channel_exists(self.gid):
            raise TraceError("capacity must be set before the channel is used")
        self._dsl().capacity(n)
        _set_declared_capacity(self.gid, n)
        return self

    def named(self, name: str) -> "Channel":
        self._dsl().named(name)
        return self

    @property
    def name(self) -> str:
        return self._dsl().name

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @property
    def shape(self) -> Shape:
        return self._shape

    # -- in-program ------------------------------------------------------------

    def take(self) -> Tensor:
        """Consume a cell inside a stage body — records a `ChanTake`."""
        return self._dsl().take()

    def read(self) -> Tensor:
        """Peek a cell inside a stage body — records a `ChanRead`."""
        return self._dsl().read()

    def put(self, v, dtype: Dtype | None = None) -> None:
        """Inside a stage body with a `Tensor`: record a device `ChanPut`. On
        the host with data: stage the next cell for the following submit."""
        if isinstance(v, Tensor):
            self._dsl().put_tensor(v)
            return
        data = const_data(v, dtype if dtype is not None else self._dtype)
        if data.dtype is not self._dtype:
            raise TypeError(f"channel {self.name} holds {self._dtype.wire_name}, put {data.dtype.wire_name}")
        self._dsl().note_host_put()
        try:
            self._wit().put(data.data)
        except _WitErr:
            # Fire-and-forget, like the Rust SDK: failures surface at take.
            pass

    def set(self, v, dtype: Dtype | None = None) -> None:
        """Atomically replace the committed front cell (a host operation)."""
        data = const_data(v, dtype if dtype is not None else self._dtype)
        _wit(self._wit().set, data.data, what=f"{self.name} set")

    # -- host readback ---------------------------------------------------------

    async def take_host(self) -> list:
        """Consume a cell on the host, decoded to a list of Python numbers
        (bools for a bool channel). Awaits in-flight fires; a poisoned
        channel raises `InferletError`."""
        self._dsl().note_host_take()
        raw = await _wit_async(self._wit().take(), what=f"{self.name} take")
        return unpack_elems(bytes(raw), self._dtype)

    async def read_host(self) -> list:
        """Peek a cell on the host (leaves it full)."""
        self._dsl().note_host_read()
        raw = await _wit_async(self._wit().read(), what=f"{self.name} read")
        return unpack_elems(bytes(raw), self._dtype)

    async def take_scalar(self):
        """`take_host()` for a one-element cell."""
        v = await self.take_host()
        if not v:
            raise InferletError(f"{self.name} take: channel cell is empty")
        return v[0]

    async def read_scalar(self):
        v = await self.read_host()
        if not v:
            raise InferletError(f"{self.name} read: channel cell is empty")
        return v[0]


# ---------------------------------------------------------------------------
# Working sets
# ---------------------------------------------------------------------------


@dataclass
class PageRange:
    start: int
    len: int

    def _wit(self):
        return _wit_ws.PageRange(self.start, self.len)


class PageGrant:
    """A grant of fresh logical page indexes from `WorkingSet.reserve`."""

    def __init__(self, start: int, length: int) -> None:
        self.start = start
        self.ids = list(range(start, start + length))

    def range(self) -> PageRange:
        return PageRange(self.start, len(self.ids))


class WorkingSet:
    """The attention working set — a logical page address space over the KV
    mapping trie. Every page reference is working-set-relative."""

    def __init__(self, _kv=None) -> None:
        self.kv = _kv if _kv is not None else _wit_ws.KvWorkingSet()

    def page_len(self) -> int:
        return self.kv.page_len()

    def reserve(self, pages: int) -> PageGrant:
        r = _wit(self.kv.reserve, pages, what="reserve KV")
        return PageGrant(r.start, r.len)

    def update_index(self, key: bytes) -> None:
        _wit(self.kv.update_index, bytes(key), what="update_index")

    @staticmethod
    def from_index(key: bytes) -> "WorkingSet | None":
        kv = _wit(_wit_ws.KvWorkingSet.from_index, bytes(key), what="from_index")
        return WorkingSet(kv) if kv is not None else None

    @staticmethod
    def remove_index(key: bytes) -> bool:
        return _wit(_wit_ws.KvWorkingSet.remove_index, bytes(key), what="remove_index")

    def discard(self, on: "Pipeline", ranges: Sequence[PageRange]) -> None:
        _wit(self.kv.discard, on.wit, [r._wit() for r in ranges], what="discard")

    def fork(self, on: "Pipeline") -> "WorkingSet":
        return WorkingSet(_wit(self.kv.fork, on.wit, what="fork"))

    def slice(self, on: "Pipeline", start: int, length: int) -> "WorkingSet":
        return WorkingSet(_wit(self.kv.slice, on.wit, _wit_ws.PageRange(start, length), what="slice"))

    def copy_into(self, on: "Pipeline", dst_page_ids, dst_tok_idx, src_page_ids, src_tok_idx) -> None:
        _wit(
            self.kv.copy_into,
            on.wit,
            list(dst_page_ids),
            list(dst_tok_idx),
            list(src_page_ids),
            list(src_tok_idx),
            what="copy_into",
        )


class RsWorkingSet:
    """Runtime recurrent-state slots for hybrid / linear-attention models."""

    def __init__(self, _rs=None) -> None:
        self.rs = _rs if _rs is not None else _wit_ws.RsWorkingSet()

    def state_size(self) -> int:
        return _wit_model.rs_state_size()

    def buffer_size(self) -> int:
        return self.rs.buffer_size()

    def buffer_page_size(self) -> int:
        return _wit_model.rs_buffer_page_size()

    def alloc_buffer(self, n: int) -> PageRange:
        r = _wit(self.rs.alloc_buffer, n, what="alloc_buffer")
        return PageRange(r.start, r.len)

    def free_buffer(self, indices: Sequence[int]) -> None:
        _wit(self.rs.free_buffer, list(indices), what="free_buffer")

    def discard_buffered(self, count: int) -> None:
        _wit(self.rs.discard_buffered, count, what="discard_buffered")

    def reorder_buffer(self, perm: Sequence[int]) -> None:
        _wit(self.rs.reorder_buffer, list(perm), what="reorder_buffer")

    def fork(self, on: "Pipeline") -> "RsWorkingSet":
        return RsWorkingSet(_wit(self.rs.fork, on.wit, what="fork"))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """A run-ahead ordering domain — every command on it linearizes in
    submission order. Concurrent streams need separate pipelines."""

    def __init__(self) -> None:
        self.wit = _wit_pipeline.Pipeline()

    def close(self) -> None:
        """End the stream; already-submitted fires still drain."""
        self.wit.close()

    def park(self) -> None:
        """Leave the frame wait-set until this pipeline submits again."""
        _wit_attention.park(self.wit)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


PageDecl: TypeAlias = "int | tuple[int, int | None] | None"
"""A page span over the working set: `None` = everything, an `int` =
`start..` (open-ended), `(start, end)` = `start..end` (`end=None` for
open-ended) — the Rust SDK's `..`, `n..`, `a..b`."""


def _page_decl(r: PageDecl) -> tuple[int, int | None]:
    if r is None:
        return 0, None
    if isinstance(r, int):
        return r, None
    start, end = r
    if end is not None and start > end:
        raise ValueError(f"attention page-span start {start} exceeds end {end}")
    return start, end


def _page_span(decl: tuple[int, int | None]):
    return _wit_ws.PageSpan(decl[0], decl[1])


@dataclass
class KvGeometry:
    """The attention geometry of one fire — mirrors WIT `kv-geometry`."""

    kv_len: Channel
    pages: Channel
    page_indptr: Channel
    w_slot: Channel
    w_off: Channel
    positions: Channel
    mask: Channel | None = None
    readable_pages: PageDecl = None
    writable_pages: PageDecl = None


@dataclass
class RsGeometry:
    """Where the bound recurrent state's folded boundary lands — mirrors WIT
    `rs-geometry`. `fold_len=None` folds everything."""

    fold_len: Channel | None = None
    buffer: PageDecl = (0, 0)


@dataclass
class KvBinding:
    working_set: WorkingSet
    geometry: KvGeometry


_FOLD_ALL: Channel | None = None


def _fold_all() -> Channel:
    global _FOLD_ALL
    if _FOLD_ALL is None:
        _FOLD_ALL = Channel.from_([0xFFFF_FFFF], Dtype.U32)
    return _FOLD_ALL


# ---------------------------------------------------------------------------
# Model constants (cached where the Rust SDK caches)
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def frame_size() -> int:
    """Waves per frame (k) for this deployment (cached)."""
    if "frame_size" not in _cache:
        _cache["frame_size"] = max(_wit_model.frame_size(), 1)
    return _cache["frame_size"]


def submit_deadline_us() -> int:
    if "submit_deadline_us" not in _cache:
        _cache["submit_deadline_us"] = _wit_model.submit_deadline_us()
    return _cache["submit_deadline_us"]


def channel_capacity() -> int:
    """Host-reader channel capacity that sustains run-ahead (not cached)."""
    return max(_wit_model.channel_capacity(), 2)


def live_slots() -> int:
    """Live slots per frame: k for dense, 1 for recurrent (linear/hybrid)."""
    if "live_slots" not in _cache:
        if _wit_model.pass_kind() != ForwardKind.ATTENTION:
            _cache["live_slots"] = 1
        else:
            _cache["live_slots"] = frame_size()
    return _cache["live_slots"]


def kv_page_size() -> int:
    if "kv_page_size" not in _cache:
        _cache["kv_page_size"] = _wit_model.kv_page_size()
    return _cache["kv_page_size"]


def max_embed_length() -> int:
    if "max_embed_length" not in _cache:
        _cache["max_embed_length"] = max(_wit_model.max_embed_length(), 1)
    return _cache["max_embed_length"]


def even_spans(n: int, cap: int) -> list[tuple[int, int]]:
    if n == 0:
        return []
    cap = max(min(cap, n), 1)
    k = max(-(-n // cap), 1)
    q, r = divmod(n, k)
    out = []
    base = 0
    for i in range(k):
        end = base + q + (1 if i < r else 0)
        out.append((base, end))
        base = end
    return out


def prefill_chunks(n: int, cap: int | None = None) -> list[tuple[int, int]]:
    """The `[start, end)` spans a prompt of `n` tokens must be prefilled in,
    respecting `max_embed_length()`."""
    c = min(cap if cap is not None else 0xFFFF_FFFF, max(max_embed_length(), 1))
    return even_spans(n, c)


# ---------------------------------------------------------------------------
# ForwardPass
# ---------------------------------------------------------------------------

_SITE_BITS = {"q": 1, "k": 2, "v": 4, "o": 8, "gate_up": 16, "down": 32}


class ForwardPass:
    """The forward-pass builder over one `pie:inferlet/forward*` interface,
    selected by `kind` (default: `model.pass_kind()`).

    Attach stage bodies with `epilogue(fn)` (also usable as a decorator),
    bind the working set + geometry with `attention(...)` (or the
    kind-independent `bind_state(...)`), and `submit(pipe)`. The first submit
    traces the stage bodies once into the ETA container.
    """

    def __init__(self, kind: ForwardKind | None = None) -> None:
        if kind is None:
            kind = _wit_model.pass_kind()
        self.kind = kind
        if kind == ForwardKind.ATTENTION:
            self._mod = _wit_attention
        elif kind == ForwardKind.HYBRID:
            self._mod = _wit_hybrid
        elif kind == ForwardKind.RECURRENT:
            self._mod = _wit_recurrent
        else:
            raise ValueError(f"unknown forward kind {kind!r}")
        self.wit = self._mod.ForwardPass()
        self._ports: list[tuple[Port, DslChannel]] = []
        self._stages: list[tuple[Stage, Callable[[], None]]] = []
        self._vocab = _wit_model.output_vocab_size()
        self._page_size = kv_page_size()
        self._program_attached = False
        self._adapter_lowrank_sites = 0
        self._adapter_scale_sites = 0

    # -- ports ----------------------------------------------------------------

    def _ensure_ports_available(self, ports: Sequence[Port]) -> None:
        if self._program_attached:
            raise InferletError("forward pass program is already attached")
        bound = {p for p, _ in self._ports}
        for p in ports:
            if p in bound:
                raise InferletError(f"forward pass port {p.wire_name} is already bound")

    def _claim(self, port: Port, ch: Channel) -> tuple[Port, DslChannel]:
        dsl = ch._dsl()
        dsl.note_desc_claim(port.consumes)
        return port, dsl

    def binds_device_mask(self) -> bool:
        return any(p == Port.ATTN_MASK for p, _ in self._ports)

    def embed(self, tokens: Channel, indptr: Channel) -> None:
        """Bind token ids and CSR row indptr (both channels)."""
        self._ensure_ports_available([Port.EMBED_TOKENS, Port.EMBED_INDPTR])
        _wit(self.wit.embed, tokens._wit(), indptr._wit(), what="embed")
        self._ports.append(self._claim(Port.EMBED_TOKENS, tokens))
        self._ports.append(self._claim(Port.EMBED_INDPTR, indptr))

    def readout(self, indices: Channel) -> None:
        self._ensure_ports_available([Port.READOUT])
        _wit(self.wit.readout, indices._wit(), what="readout")
        self._ports.append(self._claim(Port.READOUT, indices))

    def set_max_layers(self, max_layers: int) -> None:
        _wit(self.wit.set_max_layers, max_layers, what="set_max_layers")

    def set_drafting_block(self, on: bool) -> None:
        _wit(self.wit.set_drafting_block, on, what="set_drafting_block")

    def media(self, spans: Sequence[Image | Audio]) -> None:
        """Carry the payloads of `media.Image` / `media.Audio` spans,
        order-matched to their placeholder token runs in the embed."""
        if self.kind == ForwardKind.RECURRENT:
            raise InferletError("a recurrent-only pass carries no media")
        wrapped = []
        for s in spans:
            if isinstance(s, Audio):
                wrapped.append(self._mod.MediaSpan_Audio(s.handle))
            elif isinstance(s, Image):
                wrapped.append(self._mod.MediaSpan_Image(s.handle))
            else:
                raise TypeError(f"media span must be media.Image or media.Audio, got {type(s).__name__}")
        _wit(self.wit.media, wrapped, what="media")

    def _stage_kv(self, ws: WorkingSet, geom: KvGeometry) -> dict:
        rebind = self._program_attached
        if not rebind:
            ports = [Port.KV_LEN, Port.PAGES, Port.PAGE_INDPTR, Port.W_SLOT, Port.W_OFF, Port.POSITIONS]
            if geom.mask is not None:
                ports.append(Port.ATTN_MASK)
            self._ensure_ports_available(ports)
        staged = {
            "ws": ws.kv,
            "readable": _page_decl(geom.readable_pages),
            "writable": _page_decl(geom.writable_pages),
            "kv_len": geom.kv_len._wit(),
            "pages": geom.pages._wit(),
            "page_indptr": geom.page_indptr._wit(),
            "w_slot": geom.w_slot._wit(),
            "w_off": geom.w_off._wit(),
            "positions": geom.positions._wit(),
            "mask": geom.mask._wit() if geom.mask is not None else None,
        }
        if not rebind:
            self._ports.extend(
                [
                    self._claim(Port.KV_LEN, geom.kv_len),
                    self._claim(Port.PAGES, geom.pages),
                    self._claim(Port.PAGE_INDPTR, geom.page_indptr),
                    self._claim(Port.W_SLOT, geom.w_slot),
                    self._claim(Port.W_OFF, geom.w_off),
                    self._claim(Port.POSITIONS, geom.positions),
                ]
            )
            if geom.mask is not None:
                self._ports.append(self._claim(Port.ATTN_MASK, geom.mask))
        return staged

    def _kv_geometry_wit(self, staged: dict):
        return self._mod.KvGeometry(
            readable_pages=_page_span(staged["readable"]),
            writable_pages=_page_span(staged["writable"]),
            kv_len=staged["kv_len"],
            pages=staged["pages"],
            page_indptr=staged["page_indptr"],
            w_slot=staged["w_slot"],
            w_off=staged["w_off"],
            positions=staged["positions"],
            mask=staged["mask"],
        )

    def _stage_rs(self, working_sets: Sequence[RsWorkingSet], geom: RsGeometry) -> dict:
        if not working_sets:
            raise InferletError("forward pass needs one recurrent-state working set per request")
        buffer = _page_decl(geom.buffer)
        if geom.fold_len is not None:
            if not self._program_attached:
                self._ensure_ports_available([Port.RS_FOLD_LEN])
                self._ports.append(self._claim(Port.RS_FOLD_LEN, geom.fold_len))
            fold_len = geom.fold_len._wit()
        else:
            fold_len = _fold_all()._wit()
        return {"working_sets": [rs.rs for rs in working_sets], "fold_len": fold_len, "buffer": buffer}

    def attention(self, *args, **kwargs) -> None:
        """Bind the state. Attention kind: `attention(ws, geom)`. Hybrid:
        `attention(kv: KvBinding | None, rs: [RsWorkingSet], rs_geom)`.
        Recurrent: `attention(rs: [RsWorkingSet], geom: RsGeometry)`."""
        if self.kind == ForwardKind.ATTENTION:
            ws, geom = args if args else (kwargs["ws"], kwargs["geom"])
            kv = self._stage_kv(ws, geom)
            _wit(self.wit.attention, kv["ws"], self._kv_geometry_wit(kv), what="attention")
        elif self.kind == ForwardKind.HYBRID:
            if args:
                kvb, rs, rs_geom = args
            else:
                kvb, rs, rs_geom = kwargs.get("kv"), kwargs["rs"], kwargs["rs_geom"]
            kv = self._stage_kv(kvb.working_set, kvb.geometry) if kvb is not None else None
            staged_rs = self._stage_rs(rs, rs_geom)
            binding = (
                _wit_hybrid.KvBinding(working_set=kv["ws"], geometry=self._kv_geometry_wit(kv))
                if kv is not None
                else None
            )
            _wit(
                self.wit.attention,
                binding,
                staged_rs["working_sets"],
                _wit_hybrid.RsGeometry(fold_len=staged_rs["fold_len"], buffer=_page_span(staged_rs["buffer"])),
                what="attention",
            )
        else:
            rs, geom = args if args else (kwargs["rs"], kwargs["geom"])
            staged_rs = self._stage_rs(rs, geom)
            _wit(
                self.wit.attention,
                staged_rs["working_sets"],
                _wit_recurrent.RsGeometry(fold_len=staged_rs["fold_len"], buffer=_page_span(staged_rs["buffer"])),
                what="attention",
            )

    def bind_state(self, ws: WorkingSet, geom: KvGeometry, rs: Sequence[RsWorkingSet] = ()) -> None:
        """Kind-independent binding for the common text program: the KV
        geometry, plus — on a hybrid model — the recurrent working set(s),
        folding every token straight into the recurrence."""
        if self.kind == ForwardKind.ATTENTION:
            self.attention(ws, geom)
        elif self.kind == ForwardKind.HYBRID:
            self.attention(KvBinding(ws, geom), list(rs), RsGeometry(fold_len=None, buffer=(0, 0)))
        else:
            raise InferletError("bind_state: a recurrent-only model has no KV geometry to bind")

    # -- adapters ------------------------------------------------------------

    def adapter(self, site: str, f: Callable[["AdapterExpr", "AdapterExpr"], "AdapterExpr"]) -> None:
        """Attach a PEFT adapter at `site` (`"q"|"k"|"v"|"o"|"gate_up"|"down"`):
        `f(x, y)` returns the corrected expression. Lowers the LoRA
        (`y + mm(b, mm(a, x))`), IA3 (`scale(y, l)`) and DoRA forms."""
        from . import intrinsics as _intr

        bit = _SITE_BITS[site]
        expr = f(AdapterExpr("x"), AdapterExpr("y"))

        def is_lowrank(e: AdapterExpr):
            # y + mm(b, mm(a, x)) → (a, b)
            if e.kind != "add":
                return None
            lhs, rhs = e.args
            delta = rhs if lhs.kind == "y" else (lhs if rhs.kind == "y" else None)
            if delta is None or delta.kind != "mm":
                return None
            b, mid = delta.args
            if mid.kind != "mm":
                return None
            a, x = mid.args
            if x.kind != "x":
                return None
            return a, b

        if expr.kind == "scale":
            l, inner = expr.args  # noqa: E741
            lr = is_lowrank(inner)
            if lr is not None:
                a, b = lr
                if (self._adapter_lowrank_sites | self._adapter_scale_sites) & bit:
                    raise InferletError(f"adapter: site {site} already carries an adapter on this pass")
                self._adapter_lowrank_sites |= bit
                self._adapter_scale_sites |= bit

                def body():
                    _intr.kernel.lora(a.read(), b.read(), Tensor.constant(bit, Dtype.U32))
                    _intr.kernel.adapter_scale(l.read(), Tensor.constant(bit, Dtype.U32))

                self.prologue(body)
                return
            if inner.kind == "y":
                if self._adapter_scale_sites & bit:
                    raise InferletError(f"adapter: site {site} already carries a scale on this pass")
                self._adapter_scale_sites |= bit
                self.prologue(lambda: _intr.kernel.adapter_scale(l.read(), Tensor.constant(bit, Dtype.U32)))
                return
        lr = is_lowrank(expr)
        if lr is None:
            raise InferletError("adapter: form not lowerable (v0 lowers `y + mm(b, mm(a, x))`, `scale(y, l)`)")
        a, b = lr
        if self._adapter_lowrank_sites & bit:
            raise InferletError(f"adapter: site {site} already carries an adapter on this pass")
        self._adapter_lowrank_sites |= bit
        self.prologue(lambda: _intr.kernel.lora(a.read(), b.read(), Tensor.constant(bit, Dtype.U32)))

    # -- stages ---------------------------------------------------------------

    def _set_stage(self, stage: Stage, body: Callable[[], None]) -> Callable[[], None]:
        if self._program_attached:
            raise InferletError("stage attachment is construction-only")
        for i, (s, _) in enumerate(self._stages):
            if s == stage:
                self._stages[i] = (stage, body)
                return body
        self._stages.append((stage, body))
        return body

    def prologue(self, body: Callable[[], None]) -> Callable[[], None]:
        return self._set_stage(Stage.PROLOGUE, body)

    def epilogue(self, body: Callable[[], None]) -> Callable[[], None]:
        """Attach the `epilogue` stage (sampling programs; after the forward)."""
        return self._set_stage(Stage.EPILOGUE, body)

    def on_attn_proj(self, body: Callable[[], None]) -> Callable[[], None]:
        if self.kind == ForwardKind.RECURRENT:
            raise InferletError("a recurrent-only pass has no attention layer to tap")
        return self._set_stage(Stage.ON_ATTN_PROJ, body)

    def on_attn(self, body: Callable[[], None]) -> Callable[[], None]:
        if self.kind == ForwardKind.RECURRENT:
            raise InferletError("a recurrent-only pass has no attention layer to tap")
        return self._set_stage(Stage.ON_ATTN, body)

    # -- program + submit -----------------------------------------------------

    def attach_program(self) -> None:
        if self._program_attached:
            return
        builder = Builder(self._vocab, self._page_size)
        for port, ch in self._ports:
            builder.bind_port_recorded(port, ch)
        for stage, body in self._stages:
            builder.stage(stage, body)
        traced = builder.build()
        handles = [_lookup_channel(gid) for gid in traced.channel_order]
        _wit(self.wit.program, traced.encode(), handles, what="program")
        self._program_attached = True

    def submit(self, on: Pipeline) -> None:
        """Enqueue this pass as a single-slot frame on `on`."""
        submit_frame(on, [self])


class AdapterExpr:
    __slots__ = ("kind", "args")

    def __init__(self, kind: str, *args) -> None:
        self.kind = kind
        self.args = args

    def __add__(self, other: "AdapterExpr") -> "AdapterExpr":
        return AdapterExpr("add", self, other)


def mm(w: Channel, e: AdapterExpr) -> AdapterExpr:
    """`mm(w, e)` — multiply by the channel-borne weight."""
    return AdapterExpr("mm", w, e)


def scale(e: AdapterExpr, l: Channel) -> AdapterExpr:  # noqa: E741
    """`scale(e, l)` — elementwise multiply by the channel-borne vector."""
    return AdapterExpr("scale", l, e)


def submit_frame(on: Pipeline, slots: Sequence[ForwardPass | None]) -> None:
    """Submit ONE FRAME on `on`: up to `frame_size()` slots, slot i executing
    in wave i; trailing slots pad with no-ops."""
    k = frame_size()
    if len(slots) > k:
        raise InferletError(f"frame holds {len(slots)} slot(s); model.frame-size() is {k}")
    live = [p for p in slots if p is not None]
    for p in live:
        p.attach_program()
    if not live:
        return
    mod = live[0]._mod
    borrows: list = [p.wit if p is not None else None for p in slots]
    borrows.extend([None] * (k - len(borrows)))
    _wit(mod.submit, on.wit, borrows, what="submit")


async def run_ahead(
    on: Pipeline,
    pass_: ForwardPass,
    budget: int,
    on_token: Callable[[], Awaitable[bool]],
) -> int:
    """Keep the runtime's run-ahead window full while `on_token` consumes
    results, until `budget` fires submit or `on_token` returns False.
    Returns the run count."""
    if budget == 0:
        return 0
    r = 1 if pass_.binds_device_mask() else live_slots()
    window_frames = max((channel_capacity() - 1) // max(r, 1), 1)
    submitted = 0
    consumed = 0

    def submit_one_frame() -> None:
        nonlocal submitted
        live = min(r, budget - submitted)
        if live == 0:
            return
        submit_frame(on, [pass_] * live)
        submitted += live

    for _ in range(window_frames):
        if submitted >= budget:
            break
        submit_one_frame()

    ended = False
    if submitted >= budget and not ended:
        on.close()
        ended = True
    while consumed < submitted:
        cont = on_token()
        if inspect.isawaitable(cont):
            cont = await cont
        if cont is False:
            if not ended:
                on.close()
            return consumed + 1
        consumed += 1
        if submitted < budget and submitted - consumed <= (window_frames - 1) * r:
            submit_one_frame()
        if submitted >= budget and not ended:
            on.close()
            ended = True
    if not ended:
        on.close()
    return consumed

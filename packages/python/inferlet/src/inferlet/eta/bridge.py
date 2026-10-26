"""
The author-facing ETA bridge over the WIT forward surface — port of
`crates/inferlet/src/eta.rs`.

`ForwardPass` wraps the `pie:inferlet/forward` resource and drives the
neutral `Builder`, lowering author stage closures to the ETA container. A
`Channel` owns both the trace declaration and the WIT resource.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Sequence, TypeAlias

from componentize_py_types import Err as _WitErr
from wit_world.imports import channel as _wit_channel
from wit_world.imports import forward as _wit_forward
from wit_world.imports import model as _wit_model
from wit_world.imports import pie_inferlet_types as _wit_types
from wit_world.imports import pipeline as _wit_pipeline
from wit_world.imports import working_set as _wit_ws

from ..media import Audio, Image
from .builder import Builder, DslChannel
from .ir import Axis, Dtype, OutputDecl, Port, Shape, Stage, numel, shape_of
from .trace import ChannelState, TraceError
from .value import ConstData, Tensor, const_data, reshape, unpack_elems

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
# Host handles
# ---------------------------------------------------------------------------

_WIT_DTYPE = {
    Dtype.F32: _wit_types.Dtype.F32,
    Dtype.I32: _wit_types.Dtype.I32,
    Dtype.U32: _wit_types.Dtype.U32,
    Dtype.BOOL: _wit_types.Dtype.BOOL,
}


def _host_channel(state: ChannelState):
    """The WIT `channel` resource behind a trace channel, created from its
    declaration on first host use (which is what lets `capacity()` still
    widen a channel nobody has touched)."""
    if state.host is None:
        state.host = _wit_channel.Channel(list(state.shape), _WIT_DTYPE[state.dtype], state.capacity)
    return state.host


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

    __slots__ = ("_dsl", "_pending")

    def __init__(self, shape, dtype: Dtype) -> None:
        self._dsl = DslChannel.new(shape_of(shape), dtype)
        self._pending = None

    @classmethod
    def _wrap(cls, dsl: DslChannel) -> "Channel":
        ch = cls.__new__(cls)
        ch._dsl = dsl
        ch._pending = None
        return ch

    @classmethod
    def _lazy(cls) -> "Channel":
        """An output channel declared by a stage body's return value: its
        shape and dtype are the returned tensor's, known once the pass
        traces (at first submit). `capacity`/`named` before that are kept."""
        ch = cls.__new__(cls)
        ch._dsl = None
        ch._pending = {}
        return ch

    def _materialize(self, value: Tensor) -> None:
        """Inside a stage body: declare this lazy channel as the returned
        tensor's cell and record the put."""
        shape = value.shape if numel(value.shape) != 1 or len(value.shape) == 1 else (1,)
        if shape != value.shape:
            value = reshape(value, shape)
        self._dsl = DslChannel.new(shape, value.dtype)
        pending, self._pending = self._pending, None
        if "capacity" in pending:
            self._dsl.capacity(pending["capacity"])
        if "name" in pending:
            self._dsl.named(pending["name"])
        self._dsl.put_tensor(value)

    @property
    def dsl(self) -> DslChannel:
        if self._dsl is None:
            raise InferletError("this output channel is declared by its stage body; submit the pass first")
        return self._dsl

    @staticmethod
    def writer(shape, dtype: Dtype) -> "Channel":
        """An initially empty channel whose producer is the host."""
        ch = Channel(shape, dtype)
        ch._dsl.note_host_put()
        return ch

    @staticmethod
    def from_(v, dtype: Dtype | None = None) -> "Channel":
        """A channel seeded full with the per-instance value `v` (rides as a
        pre-submit `put`, never the container)."""
        return Channel._seeded_with(const_data(v, dtype))

    @staticmethod
    def from_shaped(shape, v, dtype: Dtype | None = None) -> "Channel":
        """Like `from_`, but reinterprets the flat seed under `shape`."""
        data = const_data(v, dtype)
        shape = shape_of(shape)
        if numel(shape) != numel(data.shape):
            raise ValueError("from_shaped: element count mismatch")
        return Channel._seeded_with(ConstData(shape, data.dtype, data.data))

    @staticmethod
    def _seeded_with(data: ConstData) -> "Channel":
        ch = Channel._wrap(DslChannel.from_const(data))
        _wit(ch._wit().put, data.data, what="stage seed on a fresh channel")
        return ch

    @staticmethod
    def seeded(shape, dtype: Dtype) -> "Channel":
        """A seeded channel whose seed value is supplied at instantiation."""
        return Channel._wrap(DslChannel.seeded(shape_of(shape), dtype))

    def _wit(self):
        return _host_channel(self.dsl.state)

    def capacity(self, n: int) -> "Channel":
        """Widen the ring to `n` cells (deeper run-ahead). Must precede first use."""
        if self._dsl is None:
            self._pending["capacity"] = n
            return self
        if self._dsl.state.host is not None:
            raise TraceError("capacity must be set before the channel is used")
        self._dsl.capacity(n)
        return self

    def named(self, name: str) -> "Channel":
        if self._dsl is None:
            self._pending["name"] = name
            return self
        self._dsl.named(name)
        return self

    @property
    def gid(self) -> int:
        """Declaration order — the container keys channels by it."""
        return self.dsl.gid

    @property
    def name(self) -> str:
        return self.dsl.name

    @property
    def dtype(self) -> Dtype:
        return self.dsl.dtype

    @property
    def shape(self) -> Shape:
        return self.dsl.shape

    # -- in-program ------------------------------------------------------------

    def take(self) -> Tensor:
        """Consume a cell inside a stage body — records a `ChanTake`."""
        return self.dsl.take()

    def read(self) -> Tensor:
        """Peek a cell inside a stage body — records a `ChanRead`."""
        return self.dsl.read()

    def put(self, v, dtype: Dtype | None = None) -> None:
        """Inside a stage body with a `Tensor`: record a device `ChanPut`. On
        the host with data: stage the next cell for the following submit."""
        if isinstance(v, Tensor):
            self.dsl.put_tensor(v)
            return
        data = const_data(v, dtype if dtype is not None else self.dtype)
        if data.dtype is not self.dtype:
            raise TypeError(f"channel {self.name} holds {self.dtype.wire_name}, put {data.dtype.wire_name}")
        self.dsl.note_host_put()
        try:
            self._wit().put(data.data)
        except _WitErr:
            # Fire-and-forget: failures surface at take.
            pass

    def set(self, v, dtype: Dtype | None = None) -> None:
        """Atomically replace the committed front cell (a host operation)."""
        data = const_data(v, dtype if dtype is not None else self.dtype)
        _wit(self._wit().set, data.data, what=f"{self.name} set")

    # -- host readback ---------------------------------------------------------

    async def take_host(self) -> list:
        """Consume a cell on the host, decoded to a list of Python numbers
        (bools for a bool channel). Awaits in-flight fires; a poisoned
        channel raises `InferletError`."""
        self.dsl.note_host_take()
        raw = await _wit_async(self._wit().take(), what=f"{self.name} take")
        return unpack_elems(bytes(raw), self.dtype)

    async def read_host(self) -> list:
        """Peek a cell on the host (leaves it full)."""
        self.dsl.note_host_read()
        raw = await _wit_async(self._wit().read(), what=f"{self.name} read")
        return unpack_elems(bytes(raw), self.dtype)

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

    def __await__(self):
        """`await ch` consumes a cell on the host: the number for a
        one-element cell, else the list (`take_scalar` / `take_host`)."""
        return (self.take_scalar() if numel(self.shape) == 1 else self.take_host()).__await__()


# ---------------------------------------------------------------------------
# Working sets
# ---------------------------------------------------------------------------


class WorkingSet:
    """One sequence's model state; slots below `folded()` are frozen, the rest is a buffer."""

    __slots__ = ("ws",)

    def __init__(self) -> None:
        self.ws = _wit_ws.WorkingSet()

    @classmethod
    def _wrap(cls, ws) -> "WorkingSet":
        out = cls.__new__(cls)
        out.ws = ws
        return out

    def reserve(self, end: int) -> None:
        """Lets passes write slots below `end`; needed only when the device computes the slots."""
        _wit(self.ws.reserve, end, what="reserve")

    def folded(self) -> int:
        """The frozen length; while a device-counted fold is in flight, the length before it."""
        return self.ws.folded()

    def cached_tokens(self) -> int:
        """Slots the runtime mapped in from state it already held rather than computed."""
        return self.ws.cached_tokens()

    def fork(self, on: "Pipeline") -> "WorkingSet":
        """A child sharing the frozen part, with a copy of the buffer."""
        return WorkingSet._wrap(_wit(self.ws.fork, on.wit, what="fork"))


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
        _wit_forward.park(self.wit)

    def submit(self, fwd: "ForwardPass") -> None:
        """Enqueue one pass as a single-slot frame."""
        self.submit_frame([fwd])

    def submit_frame(self, slots: Sequence["ForwardPass | None"]) -> None:
        """Submit ONE FRAME: up to `frame_size()` slots, slot i executing in
        wave i; trailing slots pad with no-ops."""
        k = frame_size()
        if len(slots) > k:
            raise InferletError(f"frame holds {len(slots)} slot(s); model.frame-size() is {k}")
        live = [p for p in slots if p is not None]
        for p in live:
            p.attach_program()
        if not live:
            return
        borrows: list = [p.wit if p is not None else None for p in slots]
        borrows.extend([None] * (k - len(borrows)))
        _wit(_wit_forward.submit, self.wit, borrows, what="submit")

    async def run_ahead(
        self,
        fwd: "ForwardPass",
        budget: int,
        on_token: Callable[[], Awaitable[bool] | bool | None],
    ) -> int:
        """Keep the runtime's run-ahead window full while `on_token` consumes
        results, until `budget` fires submit or `on_token` returns False.
        Returns the run count."""
        if budget == 0:
            return 0
        # Live slots per frame. A pass that binds a dense device mask takes one;
        # so does a pass on a model with recurrent state, whose frame's waves
        # carry the same sequence's state one into the next (a frame of one
        # live slot measured faster than a full one on Qwen3.5-0.8B), and a
        # pass off its model's default path (a canvas redrawn whole), which
        # redraws what the last wrote. Any other pass fills the frame.
        single = fwd.binds_device_mask() or _rs_state_size() > 0 or fwd.switched()
        r = 1 if single else frame_size()
        window_frames = max(_wit_model.run_ahead_window() // max(r, 1), 1)
        submitted = 0
        consumed = 0

        def submit_one_frame() -> None:
            nonlocal submitted
            live = min(r, budget - submitted)
            if live == 0:
                return
            self.submit_frame([fwd] * live)
            submitted += live

        for _ in range(window_frames):
            if submitted >= budget:
                break
            submit_one_frame()

        ended = False
        if submitted >= budget and not ended:
            self.close()
            ended = True
        while consumed < submitted:
            cont = on_token()
            if inspect.isawaitable(cont):
                cont = await cont
            if cont is False:
                if not ended:
                    self.close()
                return consumed + 1
            consumed += 1
            if submitted < budget and submitted - consumed <= (window_frames - 1) * r:
                submit_one_frame()
            if submitted >= budget and not ended:
                self.close()
                ended = True
        if not ended:
            self.close()
        return consumed


# ---------------------------------------------------------------------------
# State geometry
# ---------------------------------------------------------------------------


Fold: TypeAlias = "Channel | Literal['all'] | None"
"""How many more slots each lane freezes after a fire: `None`, `"all"`, or a u32 count channel."""


@dataclass
class StateGeometry:
    """Where a pass's rows land and what they read; omitted fields mean a causal append."""

    positions: Channel | None = None
    """Per row, the RoPE position."""
    slots: Channel | None = None
    """Per row, the slot it writes; a lane's rows write consecutive slots."""
    kv_len: Channel | None = None
    """Per lane, how many slots the lane reads."""
    fold: Fold = None

    @staticmethod
    def folding(**fields) -> "StateGeometry":
        """Append and freeze everything written: for a prompt or a decode of final tokens."""
        return StateGeometry(fold="all", **fields)


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

@functools.cache
def frame_size() -> int:
    """Waves per frame (k) for this deployment (cached)."""
    return max(_wit_model.frame_size(), 1)


@functools.cache
def submit_deadline_us() -> int:
    return _wit_model.submit_deadline_us()


def channel_capacity() -> int:
    """Host-reader channel capacity that sustains run-ahead (not cached: the
    host does not promise it static)."""
    return max(_wit_model.channel_capacity(), 2)


@functools.cache
def kv_page_size() -> int:
    return _wit_model.kv_page_size()


@functools.cache
def _rs_state_size() -> int:
    return _wit_model.rs_state_size()


def _even_spans(n: int, cap: int) -> list[tuple[int, int]]:
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


def readout_spans(n: int, max_rows: int) -> list[tuple[int, int]]:
    """Even `[start, end)` spans of `n` rows, each at most `max_rows` (an output's `max_rows`)."""
    return _even_spans(n, max(max_rows, 1))


def _output_decl(decl: "_wit_model.OutputDecl") -> OutputDecl:
    """An output as the program reads it: an i32 output as i32, any float one f32."""
    return OutputDecl(
        name=decl.name,
        axis=Axis(decl.axis.name.lower()),
        width=decl.width,
        dtype=Dtype.I32 if decl.dtype == _wit_types.Dtype.I32 else Dtype.F32,
        layers=decl.layers,
        readout=decl.readout,
    )


@functools.cache
def _lane_outputs(switches: tuple[tuple[str, str], ...]) -> list[OutputDecl]:
    """The outputs a lane of these switches reads, by the model's lane decl."""
    decl = _wit(_wit_model.lane, list(switches), what="model.lane")
    return [_output_decl(output) for output in decl.outputs]


# ---------------------------------------------------------------------------
# ForwardPass
# ---------------------------------------------------------------------------

_SITE_BITS = {"q": 1, "k": 2, "v": 4, "o": 8, "gate_up": 16, "down": 32}


class ForwardPass:
    """The forward-pass builder over `pie:inferlet/forward`.

    Attach stage bodies with `epilogue(fn)` (also usable as a decorator),
    bind the lanes' working sets with `state([ws], geometry)`, and
    `submit(pipe)`. The first submit traces the stage bodies once into the
    ETA container.
    """

    def __init__(self) -> None:
        self.wit = _wit_forward.ForwardPass()
        self._ports: list[tuple[Port, Channel]] = []
        self._inputs: list[tuple[int, str, Channel]] = []
        self._stages: list[tuple[Stage, Callable[[], None]]] = []
        self._vocab = _wit_model.output_vocab_size()
        self._page_size = kv_page_size()
        self._program_attached = False
        self._adapter_lowrank_sites = 0
        self._adapter_scale_sites = 0
        # Keeps the bound working sets alive while the pass is bound.
        self._lanes: list[WorkingSet] = []
        self._fold_all: Channel | None = None
        self._switches: list[tuple[int, str, str]] = []

    # -- ports ----------------------------------------------------------------

    def _ensure_ports_available(self, ports: Sequence[Port]) -> None:
        if self._program_attached:
            raise InferletError("forward pass program is already attached")
        bound = {p for p, _ in self._ports}
        for p in ports:
            if p in bound:
                raise InferletError(f"forward pass port {p.wire_name} is already bound")

    def binds_device_mask(self) -> bool:
        return any(p == Port.ATTN_MASK for p, _ in self._ports)

    def embed(self, tokens: Channel | Sequence[int], indptr: Channel | Sequence[int] | None = None) -> None:
        """Bind token ids and the CSR row indptr. A token list becomes an i32
        channel; `indptr` omitted is the one row `[0, n]`, a list is a u32
        channel."""
        if not isinstance(tokens, Channel):
            tokens = Channel.from_(list(tokens), Dtype.I32)
        if indptr is None:
            indptr = Channel.from_([0, numel(tokens.shape)], Dtype.U32)
        elif not isinstance(indptr, Channel):
            indptr = Channel.from_(list(indptr), Dtype.U32)
        self._ensure_ports_available([Port.EMBED_TOKENS, Port.EMBED_INDPTR])
        _wit(self.wit.embed, tokens._wit(), indptr._wit(), what="embed")
        self._ports += [(Port.EMBED_TOKENS, tokens), (Port.EMBED_INDPTR, indptr)]

    def readout(self, indices: Channel) -> None:
        self._ensure_ports_available([Port.READOUT])
        _wit(self.wit.readout, indices._wit(), what="readout")
        self._ports.append((Port.READOUT, indices))

    def media(self, lane: int, spans: Sequence[Image | Audio]) -> None:
        """Lane `lane`'s media span payloads, in placeholder order."""
        wrapped = []
        for s in spans:
            if isinstance(s, Audio):
                wrapped.append(_wit_forward.MediaSpan_Audio(s.handle))
            elif isinstance(s, Image):
                wrapped.append(_wit_forward.MediaSpan_Image(s.handle))
            else:
                raise TypeError(f"media span must be media.Image or media.Audio, got {type(s).__name__}")
        _wit(self.wit.media, int(lane), wrapped, what="media")

    def state(self, lanes: Sequence[WorkingSet], geom: StateGeometry | None = None) -> None:
        """Bind one working set per lane, in lane order; lanes sharing a set must state `slots`."""
        geom = geom if geom is not None else StateGeometry()
        rebind = self._program_attached
        fold = geom.fold
        if isinstance(fold, str):
            if fold != "all":
                raise ValueError(f"fold is None, 'all' or a channel, not {fold!r}")
            if self._fold_all is None:
                self._fold_all = Channel.from_([0xFFFF_FFFF], Dtype.U32)
            fold = self._fold_all
        stated = [
            (Port.POSITIONS, geom.positions),
            (Port.SLOTS, geom.slots),
            (Port.KV_LEN, geom.kv_len),
            (Port.FOLD, fold),
        ]
        if not rebind:
            self._ensure_ports_available([port for port, ch in stated if ch is not None])
        wits = [ch._wit() if ch is not None else None for _, ch in stated]
        _wit(
            self.wit.state,
            [lane.ws for lane in lanes],
            _wit_forward.StateGeometry(
                positions=wits[0], slots=wits[1], kv_len=wits[2], fold=wits[3]
            ),
            what="state",
        )
        self._lanes = list(lanes)
        if not rebind:
            self._ports += [(port, ch) for port, ch in stated if ch is not None]

    def checkpoints(self, slots: Sequence[int]) -> None:
        """Slots to keep a recurrent state at, so a later pass over these tokens resumes there."""
        _wit(self.wit.checkpoints, [int(s) for s in slots], what="checkpoints")

    # -- switches and inputs ---------------------------------------------------

    def switch(self, lane: int, name: str, value: str) -> None:
        """Set lane `lane`'s model switch `name` to `value`, before the program."""
        if self._program_attached:
            raise InferletError("forward pass program is already attached")
        _wit(self.wit.switch, int(lane), name, value, what="switch")
        self._switches.append((int(lane), name, value))

    def switched(self) -> bool:
        """Whether a switch is off its default; such a pass posts one fire at a time."""
        if not self._switches:
            return False
        declared = {d.name: d.values for d in _wit_model.switches()}
        return any(
            name in declared and declared[name][:1] != [value] for _, name, value in self._switches
        )

    def input(self, lane: int, name: str, ch: Channel) -> None:
        """Bind `ch` to lane `lane`'s model input `name`, read from its committed cell."""
        if self._program_attached:
            raise InferletError("forward pass program is already attached")
        if any(at == lane and bound == name for at, bound, _ in self._inputs):
            raise InferletError(f"input `{name}` is already bound on lane {lane}")
        _wit(self.wit.input, int(lane), name, ch._wit(), what="input")
        self._inputs.append((int(lane), name, ch))

    def group(self, lane: int, id: int) -> None:
        """Put lane `lane` in this pass's attention group `id`."""
        if self._program_attached:
            raise InferletError("forward pass program is already attached")
        _wit(self.wit.group, int(lane), int(id), what="group")

    def mask(self, rows: Channel) -> None:
        """Per row, the kv slots it attends, or its attention group's rows without kv."""
        self._ensure_ports_available([Port.ATTN_MASK])
        _wit(self.wit.mask, rows._wit(), what="mask")
        self._ports.append((Port.ATTN_MASK, rows))

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

    def epilogue(self, body: Callable[[], Tensor | None]) -> Channel:
        """Attach the `epilogue` stage (sampling programs; after the forward).
        A body that returns a tensor declares an output channel of that
        cell, put once per fire; the channel is returned and read on the
        host (`await out`). `out.capacity(n)` widens it for run-ahead."""
        out = Channel._lazy()

        def traced() -> None:
            value = body()
            if value is None:
                return
            if not isinstance(value, Tensor):
                raise TypeError(f"an epilogue returns a Tensor to publish, or None; got {type(value).__name__}")
            out._materialize(value)

        self._set_stage(Stage.EPILOGUE, traced)
        return out

    # -- program + submit -----------------------------------------------------

    def attach_program(self) -> None:
        if self._program_attached:
            return
        builder = Builder(self._vocab, self._page_size)
        lanes = max(
            [1, len(self._lanes)]
            + [lane + 1 for lane, _, _ in self._switches]
            + [lane + 1 for lane, _, _ in self._inputs]
        )
        builder.set_outputs(
            [
                _lane_outputs(
                    tuple((name, value) for at, name, value in self._switches if at == lane)
                )
                for lane in range(lanes)
            ]
        )
        for port, ch in self._ports:
            builder.bind_port(port, ch._dsl)
        # A channel several lanes read is one program input.
        listed: list[Channel] = []
        for _, _, ch in self._inputs:
            if not any(seen is ch for seen in listed):
                listed.append(ch)
                builder.bind_input(ch._dsl)
        for stage, body in self._stages:
            builder.stage(stage, body)
        traced = builder.build()
        handles = [_host_channel(st) for st in traced.channels]
        _wit(self.wit.program, traced.encode(), handles, what="program")
        self._program_attached = True


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



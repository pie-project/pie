"""
The trace-recording context — port of `eta-dsl/src/context.rs`.

A module-level session holds the stage currently being traced. Channels are
plain objects the author holds; a trace interns the ones it touches.
Single-threaded by construction (wasm inferlets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

from .ir import Dtype, Op, Shape, SinkScope, Stage


class TraceError(Exception):
    """An authoring mistake found while tracing (the Rust SDK's
    `TraceError::Authoring` and the span lints). Raised at the call site
    rather than collected, so the Python traceback points at the author's
    line."""


class ValueType(NamedTuple):
    shape: Shape
    dtype: Dtype


@dataclass
class ChannelState:
    """A channel's shared state: the trace declaration, the seed flag, and
    the endpoint claims the host-role derivation + lints read."""

    gid: int
    name: str
    shape: Shape
    dtype: Dtype
    capacity: int
    seeded: bool
    has_seed: bool
    prog_puts: list[Stage] = field(default_factory=list)
    prog_takes: list[Stage] = field(default_factory=list)
    prog_reads: list[Stage] = field(default_factory=list)
    host_puts: int = 0
    host_takes: int = 0
    host_reads: int = 0
    desc_takes: int = 0
    desc_reads: int = 0
    # The bridge's host-side handle for this channel (the WIT resource),
    # created on first host use. Opaque to the trace.
    host: Any = None

    def elem_ty(self) -> ValueType:
        return ValueType(self.shape, self.dtype)


@dataclass
class SinkCall:
    name: str
    scope: SinkScope


class Recorder:
    """The stage currently being traced."""

    def __init__(self, stage: Stage, rows: int) -> None:
        self.stage = stage
        self.rows = rows
        self.ops: list[Op] = []
        self.types: list[ValueType] = []
        self.sinks: list[SinkCall] = []

    def push(self, op: Op, result_tys: tuple[ValueType, ...] | list[ValueType]) -> int:
        base = len(self.types)
        if op.result_count != len(result_tys):
            raise AssertionError(
                f"result arity mismatch for {op.name}: recording {len(result_tys)} types "
                f"against {op.result_count} results would shift every later value id"
            )
        self.types.extend(result_tys)
        self.ops.append(op)
        return base


@dataclass
class StageResult:
    stage: Stage
    ops: list[Op]
    sinks: list[SinkCall]


class Session:
    def __init__(self) -> None:
        self.chan_by_gid: dict[int, int] = {}
        self.channels: list[ChannelState] = []
        self.current: Recorder | None = None
        self.names: list[str] = []

    def intern(self, ch: ChannelState) -> int:
        idx = self.chan_by_gid.get(ch.gid)
        if idx is not None:
            return idx
        idx = len(self.channels)
        self.chan_by_gid[ch.gid] = idx
        self.channels.append(ch)
        return idx


_session: Session | None = None
_next_gid = 1

# Trace-known model constants (`eta-dsl/src/model.rs`), installed by the
# builder for the duration of one build.
_vocab = 32_000
_page_size = 16


def next_gid() -> int:
    global _next_gid
    gid = _next_gid
    _next_gid += 1
    return gid


def is_tracing() -> bool:
    return _session is not None and _session.current is not None


def _sess() -> Session:
    if _session is None:
        raise TraceError("no trace session is active")
    return _session


def _rec() -> Recorder:
    s = _sess()
    if s.current is None:
        raise TraceError("op emitted outside a traced stage")
    return s.current


def intern_channel(ch: ChannelState) -> int:
    return _sess().intern(ch)


def with_session(f: Callable[[], object]) -> tuple[object, list[ChannelState], list[str]]:
    """Run `f` with a fresh session active; return its result plus the
    interned channels and the name table."""
    global _session
    if _session is not None:
        raise TraceError("nested trace session")
    _session = Session()
    try:
        r = f()
        return r, _session.channels, _session.names
    finally:
        _session = None


def trace_stage(stage: Stage, rows: int, body: Callable[[], None]) -> StageResult:
    s = _sess()
    if s.current is not None:
        raise TraceError("nested stage")
    s.current = Recorder(stage, rows)
    try:
        body()
    finally:
        rec = s.current
        s.current = None
    return StageResult(rec.stage, rec.ops, rec.sinks)


def current_rows() -> int:
    if _session is None or _session.current is None:
        return 1
    return _session.current.rows


def current_stage() -> Stage:
    return _rec().stage


def emit(op: Op, result_tys: tuple[ValueType, ...] | list[ValueType]) -> int:
    """Emit an op into the current stage; returns its first result id."""
    return _rec().push(op, result_tys)


def record_channel_read(ch: ChannelState, consume: bool) -> tuple[int, ValueType]:
    """Record a channel `take`/`read` inside a stage: intern, push the op,
    register the endpoint claim; return the produced value id + type."""
    s = _sess()
    dense = s.intern(ch)
    elem = ch.elem_ty()
    rec = _rec()
    if consume:
        ch.prog_takes.append(rec.stage)
        op = Op.chan_take(dense)
    else:
        ch.prog_reads.append(rec.stage)
        op = Op.chan_read(dense)
    vid = rec.push(op, (elem,))
    return vid, elem


def record_channel_put(ch: ChannelState, value: int) -> None:
    """Record a channel `put` inside a stage. A channel bound to a peeked
    descriptor port (geometry/masks) is drained first, so a bare re-put does
    not grow the ring forever; an explicit take in the same trace is honoured
    rather than repeated."""
    s = _sess()
    dense = s.intern(ch)
    rec = _rec()
    peeked_port = ch.desc_reads > 0 and ch.desc_takes == 0
    drain = peeked_port and not ch.prog_takes
    if drain:
        ch.prog_takes.append(rec.stage)
        rec.push(Op.chan_take(dense), (ch.elem_ty(),))
    ch.prog_puts.append(rec.stage)
    rec.push(Op.chan_put(dense, value), ())


def intern_name(name: str) -> int:
    """Intern a second-party name into the session's shared name table."""
    s = _sess()
    try:
        return s.names.index(name)
    except ValueError:
        s.names.append(name)
        return len(s.names) - 1


def record_sink(name: str, scope: SinkScope) -> None:
    _rec().sinks.append(SinkCall(name, scope))


def with_constants(vocab: int, page_size: int, f: Callable[[], object]) -> object:
    global _vocab, _page_size
    prev = (_vocab, _page_size)
    _vocab, _page_size = vocab, page_size
    try:
        return f()
    finally:
        _vocab, _page_size = prev


def vocab() -> int:
    return _vocab


def page_size() -> int:
    return _page_size

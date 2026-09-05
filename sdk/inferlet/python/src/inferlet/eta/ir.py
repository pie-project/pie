"""
ETA IR — the representation layer, ported from `crates/eta-ir`.

Stage-tagged op programs, channel declarations, descriptor-port bindings,
and the versioned trace container whose canonical bytes are the pass's
identity (`container_hash`). This module produces bytes that agree with the
Rust encoder byte for byte: the same program traced from Python and from
Rust hashes to the same FNV-1a value and shares the host's program cache.

Only the encode direction is here; the host decodes and validates.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

# ---------------------------------------------------------------------------
# Scalar types, shapes
# ---------------------------------------------------------------------------


class Dtype(IntEnum):
    """The four dtypes ETA computes in; the value is the wire byte."""

    F32 = 0
    I32 = 1
    U32 = 2
    BOOL = 3

    @property
    def is_float(self) -> bool:
        return self is Dtype.F32

    @property
    def is_int(self) -> bool:
        return self in (Dtype.I32, Dtype.U32)

    @property
    def is_numeric(self) -> bool:
        return self is not Dtype.BOOL

    @property
    def elem_size(self) -> int:
        """Bytes per element in a payload: 4, or 1 for bool."""
        return 1 if self is Dtype.BOOL else 4

    @property
    def wire_name(self) -> str:
        return {Dtype.F32: "f32", Dtype.I32: "i32", Dtype.U32: "u32", Dtype.BOOL: "bool"}[self]


class dtype:  # noqa: N801 — spelled like the Rust `dtype::f32` module.
    """`dtype.f32` / `dtype.i32` / `dtype.u32` / `dtype.bool`."""

    f32 = Dtype.F32
    i32 = Dtype.I32
    u32 = Dtype.U32
    bool = Dtype.BOOL


MAX_RANK = 4

# A shape is a tuple of extents, outermost first; `()` is the scalar.
Shape = tuple

SCALAR: Shape = ()


def shape_of(dims: Sequence[int] | int) -> Shape:
    """Validate and normalize `dims` into a shape tuple.

    Raises ``ValueError`` for a rank above ``MAX_RANK``, a zero extent, or a
    negative extent — every extent must be at least 1 (a dim of `0` is not
    expressible on the wire).
    """
    if isinstance(dims, int):
        dims = (dims,)
    shape = tuple(int(d) for d in dims)
    if len(shape) > MAX_RANK:
        raise ValueError(f"shape {shape} has rank {len(shape)}, above MAX_RANK={MAX_RANK}")
    for d in shape:
        if d <= 0:
            raise ValueError(f"shape {shape} has a non-positive extent; every extent must be >= 1")
        if d > 0xFFFF_FFFF:
            raise ValueError(f"shape {shape} has an extent that does not fit u32")
    return shape


def numel(shape: Shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def rows(shape: Shape) -> int:
    """The number of rows: the product of every axis but the last."""
    if len(shape) <= 1:
        return 1
    return numel(shape[:-1])


def drop_last(shape: Shape) -> Shape:
    """The shape with its last axis dropped — a reduction's result; the
    scalar reduces to the scalar."""
    if not shape:
        return SCALAR
    return shape[:-1]


class RngKind(IntEnum):
    UNIFORM = 0
    GUMBEL = 1


# ---------------------------------------------------------------------------
# Registry: stages, ports, intrinsics, sinks
# ---------------------------------------------------------------------------


class Stage(IntEnum):
    PROLOGUE = 0
    ON_ATTN_PROJ = 1
    ON_ATTN = 2
    EPILOGUE = 3

    @property
    def wire_name(self) -> str:
        return ("prologue", "on_attn_proj", "on_attn", "epilogue")[self]


class Port(IntEnum):
    EMBED_TOKENS = 0
    EMBED_INDPTR = 1
    POSITIONS = 2
    PAGES = 3
    PAGE_INDPTR = 4
    KV_LEN = 5
    W_SLOT = 6
    W_OFF = 7
    READOUT = 8
    ATTN_MASK = 9
    RS_BUFFER_PAGES = 10
    RS_BUFFER_INDPTR = 11
    RS_BUFFER_LEN = 12
    RS_W_SLOT = 13
    RS_W_OFF = 14
    RS_FOLD_LEN = 15

    @property
    def consumes(self) -> bool:
        """True iff a channel bound to this port is consumed (take) by the
        pass; false = peeked (read). Token-indexed ports consume; geometry and
        masks are state, peeked."""
        return self in (
            Port.EMBED_TOKENS,
            Port.POSITIONS,
            Port.W_SLOT,
            Port.W_OFF,
            Port.RS_W_SLOT,
            Port.RS_W_OFF,
            Port.RS_FOLD_LEN,
        )

    @property
    def wire_name(self) -> str:
        return self.name.lower()


class Intrinsic(IntEnum):
    LOGITS = 0
    MTP_LOGITS = 1
    HIDDEN = 2
    QUERY = 3
    VALUE_HEAD = 4
    LAYER = 5
    MTP_DRAFTS = 6
    ATTN_SCORE = 7


class SinkScope(IntEnum):
    PASS_WIDE = 0
    ATTENTION = 1


KNOWN_SINKS = {
    "attn_page_mask": SinkScope.ATTENTION,
    "lora": SinkScope.PASS_WIDE,
    "minference_sparse": SinkScope.PASS_WIDE,
}

ATTN_SCORE_KV_MAX = 2048


class HostRole(IntEnum):
    NONE = 0
    WRITER = 1
    READER = 2


# Predicate wire tags (pivot_threshold).
PRED_RANK_LE = 0
PRED_CUMMASS_LE = 1
PRED_PROB_GE = 2


# ---------------------------------------------------------------------------
# The op table
# ---------------------------------------------------------------------------

# Wire fields, in encode order. Mirrors `eta_ir::op::WireField`.
VALUE = "value"
CHAN = "chan"
IMM = "imm"
DTYPE = "dtype"
SHAPE = "shape"
RNG_KIND = "rng_kind"
PREDICATE = "predicate"
LITERAL = "literal"
NAME = "name"
INTRINSIC = "intrinsic"
ARGS = "args"


class tags:  # noqa: N801
    EXP = 0x01
    LOG = 0x02
    NEG = 0x03
    RECIP = 0x04
    ABS = 0x05
    SIGN = 0x06
    CAST = 0x07
    ADD = 0x10
    SUB = 0x11
    MUL = 0x12
    DIV = 0x13
    MAX_ELEM = 0x14
    MIN_ELEM = 0x15
    GT = 0x16
    GE = 0x17
    EQ = 0x18
    NE = 0x19
    LT = 0x1A
    LE = 0x1B
    AND = 0x1C
    OR = 0x1D
    NOT = 0x1E
    REM = 0x1F
    SELECT = 0x20
    REDUCE_SUM = 0x30
    REDUCE_MAX = 0x31
    REDUCE_MIN = 0x32
    REDUCE_ARGMAX = 0x33
    BROADCAST = 0x38
    RESHAPE = 0x39
    TRANSPOSE = 0x3A
    CUMSUM = 0x40
    CUMPROD = 0x41
    SORT_DESC = 0x50
    TOP_K = 0x51
    MATMUL = 0x55
    PIVOT_THRESHOLD = 0x58
    GATHER = 0x60
    GATHER_ROW = 0x61
    SCATTER_ADD = 0x62
    SCATTER_SET = 0x63
    IOTA = 0x64
    MASK_APPLY_PACKED = 0x65
    CAUSAL_MASK = 0x66
    SLIDING_WINDOW_MASK = 0x67
    SINK_WINDOW_MASK = 0x68
    RNG = 0x70
    RNG_KEYED = 0x71
    CONST = 0x81
    CHAN_TAKE = 0x90
    CHAN_READ = 0x91
    CHAN_PUT = 0x92
    INTRINSIC_VAL = 0xA0
    KERNEL_CALL = 0xA1
    SINK_CALL = 0xA2


# tag -> (name, result count, wire layout). One row per `declare_ops!` row.
OP_TABLE: dict[int, tuple[str, int, tuple[str, ...]]] = {
    tags.EXP: ("exp", 1, (VALUE,)),
    tags.LOG: ("log", 1, (VALUE,)),
    tags.NEG: ("neg", 1, (VALUE,)),
    tags.RECIP: ("recip", 1, (VALUE,)),
    tags.ABS: ("abs", 1, (VALUE,)),
    tags.SIGN: ("sign", 1, (VALUE,)),
    tags.CAST: ("cast", 1, (VALUE, DTYPE)),
    tags.ADD: ("add", 1, (VALUE, VALUE)),
    tags.SUB: ("sub", 1, (VALUE, VALUE)),
    tags.MUL: ("mul", 1, (VALUE, VALUE)),
    tags.DIV: ("div", 1, (VALUE, VALUE)),
    tags.MAX_ELEM: ("max_elem", 1, (VALUE, VALUE)),
    tags.MIN_ELEM: ("min_elem", 1, (VALUE, VALUE)),
    tags.GT: ("gt", 1, (VALUE, VALUE)),
    tags.GE: ("ge", 1, (VALUE, VALUE)),
    tags.EQ: ("eq", 1, (VALUE, VALUE)),
    tags.NE: ("ne", 1, (VALUE, VALUE)),
    tags.LT: ("lt", 1, (VALUE, VALUE)),
    tags.LE: ("le", 1, (VALUE, VALUE)),
    tags.AND: ("and", 1, (VALUE, VALUE)),
    tags.OR: ("or", 1, (VALUE, VALUE)),
    tags.NOT: ("not", 1, (VALUE,)),
    tags.REM: ("rem", 1, (VALUE, VALUE)),
    tags.SELECT: ("select", 1, (VALUE, VALUE, VALUE)),
    tags.REDUCE_SUM: ("reduce_sum", 1, (VALUE,)),
    tags.REDUCE_MAX: ("reduce_max", 1, (VALUE,)),
    tags.REDUCE_MIN: ("reduce_min", 1, (VALUE,)),
    tags.REDUCE_ARGMAX: ("reduce_argmax", 1, (VALUE,)),
    tags.BROADCAST: ("broadcast", 1, (VALUE, SHAPE)),
    tags.RESHAPE: ("reshape", 1, (VALUE, SHAPE)),
    tags.TRANSPOSE: ("transpose", 1, (VALUE,)),
    tags.CUMSUM: ("cumsum", 1, (VALUE,)),
    tags.CUMPROD: ("cumprod", 1, (VALUE,)),
    tags.SORT_DESC: ("sort_desc", 2, (VALUE,)),
    tags.TOP_K: ("top_k", 2, (VALUE, IMM)),
    tags.MATMUL: ("matmul", 1, (VALUE, VALUE)),
    tags.PIVOT_THRESHOLD: ("pivot_threshold", 1, (VALUE, PREDICATE)),
    tags.GATHER: ("gather", 1, (VALUE, VALUE)),
    tags.GATHER_ROW: ("gather_row", 1, (VALUE, VALUE)),
    tags.SCATTER_ADD: ("scatter_add", 1, (VALUE, VALUE, VALUE)),
    tags.SCATTER_SET: ("scatter_set", 1, (VALUE, VALUE, VALUE)),
    tags.IOTA: ("iota", 1, (IMM,)),
    tags.MASK_APPLY_PACKED: ("mask_apply_packed", 1, (VALUE, VALUE)),
    tags.CAUSAL_MASK: ("causal_mask", 1, (VALUE, IMM)),
    tags.SLIDING_WINDOW_MASK: ("sliding_window_mask", 1, (VALUE, IMM, IMM)),
    tags.SINK_WINDOW_MASK: ("sink_window_mask", 1, (VALUE, IMM, IMM, IMM)),
    tags.RNG: ("rng", 1, (IMM, SHAPE, RNG_KIND)),
    tags.RNG_KEYED: ("rng_keyed", 1, (VALUE, SHAPE, RNG_KIND)),
    tags.CONST: ("const", 1, (LITERAL,)),
    tags.CHAN_TAKE: ("chan_take", 1, (CHAN,)),
    tags.CHAN_READ: ("chan_read", 1, (CHAN,)),
    tags.CHAN_PUT: ("chan_put", 0, (CHAN, VALUE)),
    tags.INTRINSIC_VAL: ("intrinsic_val", 1, (INTRINSIC, DTYPE, SHAPE)),
    tags.KERNEL_CALL: ("kernel_call", 1, (NAME, DTYPE, SHAPE, ARGS)),
    tags.SINK_CALL: ("sink_call", 0, (NAME, ARGS)),
}


@dataclass
class Op:
    """One op as its flat wire record — `eta_ir::wire::OpWire`. The encoder
    walks the tag's layout and reads the fields it names."""

    tag: int
    args: list[int] = field(default_factory=list)
    chan: int = -1
    imms: tuple[int, ...] = ()
    dtype: int = 0
    shape: Shape = SCALAR
    kind: int = 0
    pred_tag: int = 0
    pred_payload: int = 0
    lit_dtype: int = 0
    lit_bits: int = 0
    name_idx: int = 0
    intr: int = 0

    @property
    def name(self) -> str:
        return OP_TABLE[self.tag][0]

    @property
    def result_count(self) -> int:
        return OP_TABLE[self.tag][1]

    # -- constructors -------------------------------------------------------

    @staticmethod
    def unary(tag: int, a: int) -> "Op":
        return Op(tag, [a])

    @staticmethod
    def binary(tag: int, a: int, b: int) -> "Op":
        return Op(tag, [a, b])

    @staticmethod
    def ternary(tag: int, a: int, b: int, c: int) -> "Op":
        return Op(tag, [a, b, c])

    @staticmethod
    def const(dt: Dtype, value) -> "Op":
        return Op(tags.CONST, lit_dtype=int(dt), lit_bits=literal_bits(dt, value))

    @staticmethod
    def cast(value: int, dt: Dtype) -> "Op":
        return Op(tags.CAST, [value], dtype=int(dt))

    @staticmethod
    def reshape(value: int, shape: Shape) -> "Op":
        return Op(tags.RESHAPE, [value], shape=shape)

    @staticmethod
    def broadcast(value: int, shape: Shape) -> "Op":
        return Op(tags.BROADCAST, [value], shape=shape)

    @staticmethod
    def iota(length: int) -> "Op":
        return Op(tags.IOTA, imms=(length,))

    @staticmethod
    def top_k(value: int, k: int) -> "Op":
        return Op(tags.TOP_K, [value], imms=(k,))

    @staticmethod
    def pivot_threshold(value: int, pred_tag: int, pred_value: int) -> "Op":
        return Op(tags.PIVOT_THRESHOLD, [value], pred_tag=pred_tag, pred_payload=pred_value)

    @staticmethod
    def causal_mask(positions: int, length: int) -> "Op":
        return Op(tags.CAUSAL_MASK, [positions], imms=(length,))

    @staticmethod
    def sliding_window_mask(positions: int, length: int, window: int) -> "Op":
        return Op(tags.SLIDING_WINDOW_MASK, [positions], imms=(length, window))

    @staticmethod
    def sink_window_mask(positions: int, length: int, sink: int, window: int) -> "Op":
        return Op(tags.SINK_WINDOW_MASK, [positions], imms=(length, sink, window))

    @staticmethod
    def rng(stream: int, shape: Shape, kind: RngKind) -> "Op":
        return Op(tags.RNG, imms=(stream,), shape=shape, kind=int(kind))

    @staticmethod
    def rng_keyed(state: int, shape: Shape, kind: RngKind) -> "Op":
        return Op(tags.RNG_KEYED, [state], shape=shape, kind=int(kind))

    @staticmethod
    def chan_take(chan: int) -> "Op":
        return Op(tags.CHAN_TAKE, chan=chan)

    @staticmethod
    def chan_read(chan: int) -> "Op":
        return Op(tags.CHAN_READ, chan=chan)

    @staticmethod
    def chan_put(chan: int, value: int) -> "Op":
        return Op(tags.CHAN_PUT, [value], chan=chan)

    @staticmethod
    def intrinsic_val(intr: Intrinsic, shape: Shape, dt: Dtype) -> "Op":
        return Op(tags.INTRINSIC_VAL, intr=int(intr), shape=shape, dtype=int(dt))

    @staticmethod
    def kernel_call(name: int, args: list[int], shape: Shape, dt: Dtype) -> "Op":
        return Op(tags.KERNEL_CALL, list(args), name_idx=name, shape=shape, dtype=int(dt))

    @staticmethod
    def sink_call(name: int, args: list[int]) -> "Op":
        return Op(tags.SINK_CALL, list(args), name_idx=name)

    @property
    def channel(self) -> int | None:
        return self.chan if self.chan >= 0 else None


def literal_bits(dt: Dtype, value) -> int:
    """The 4 raw payload bytes of a `const` literal, as a u32."""
    if dt is Dtype.F32:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if dt is Dtype.I32:
        return int(value) & 0xFFFF_FFFF
    if dt is Dtype.U32:
        v = int(value)
        if not 0 <= v <= 0xFFFF_FFFF:
            raise ValueError(f"{v} does not fit u32")
        return v
    if dt is Dtype.BOOL:
        return 1 if value else 0
    raise ValueError(f"{dt!r} has no literal form")


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

ETA_MAGIC = b"ETA\0"
ETA_VERSION = 1
ETA_VERSION_EXTERN = 2


@dataclass
class ChannelDecl:
    shape: Shape
    dtype: Dtype
    capacity: int
    host_role: HostRole
    seeded: bool


@dataclass
class PortBinding:
    port: Port
    channel: int  # the DSL binds ports to channels only (no const sources)


@dataclass
class StageProgram:
    stage: Stage
    ops: list[Op]


@dataclass
class ExternDecl:
    name: int
    direction: int
    chan: int


@dataclass
class TraceContainer:
    names: list[str] = field(default_factory=list)
    channels: list[ChannelDecl] = field(default_factory=list)
    ports: list[PortBinding] = field(default_factory=list)
    stages: list[StageProgram] = field(default_factory=list)
    externs: list[ExternDecl] = field(default_factory=list)

    def encode(self) -> bytes:
        return encode(self)

    def hash(self) -> int:
        return container_hash(self.encode())


def _u16(w: bytearray, v: int) -> None:
    if not 0 <= v <= 0xFFFF:
        raise ValueError(f"{v} exceeds its u16 wire width")
    w += struct.pack("<H", v)


def _u32(w: bytearray, v: int) -> None:
    if not 0 <= v <= 0xFFFF_FFFF:
        raise ValueError(f"{v} exceeds its u32 wire width")
    w += struct.pack("<I", v)


def _u8(w: bytearray, v: int) -> None:
    if not 0 <= v <= 0xFF:
        raise ValueError(f"{v} exceeds its u8 wire width")
    w.append(v)


def encode_shape(w: bytearray, shape: Shape) -> None:
    _u8(w, len(shape))
    for d in shape:
        _u32(w, d)


def encode_op(w: bytearray, op: Op) -> None:
    _u8(w, op.tag)
    layout = OP_TABLE[op.tag][2]
    value = 0
    imm = 0
    for f in layout:
        if f == VALUE:
            _u32(w, op.args[value])
            value += 1
        elif f == CHAN:
            if op.chan < 0:
                raise ValueError(f"{op.name} carries no channel index")
            _u32(w, op.chan)
        elif f == IMM:
            _u32(w, op.imms[imm])
            imm += 1
        elif f == DTYPE:
            _u8(w, op.dtype)
        elif f == SHAPE:
            encode_shape(w, op.shape)
        elif f == RNG_KIND:
            _u8(w, op.kind)
        elif f == PREDICATE:
            _u8(w, op.pred_tag)
            _u32(w, op.pred_payload)
        elif f == LITERAL:
            _u8(w, op.lit_dtype)
            _u32(w, op.lit_bits)
        elif f == NAME:
            _u16(w, op.name_idx)
        elif f == INTRINSIC:
            _u16(w, op.intr)
        elif f == ARGS:
            rest = op.args[value:]
            _u8(w, len(rest))
            for a in rest:
                _u32(w, a)
        else:  # pragma: no cover
            raise AssertionError(f"unknown wire field {f}")


def encode(c: TraceContainer) -> bytes:
    w = bytearray()
    w += ETA_MAGIC
    v2 = bool(c.externs)
    _u16(w, ETA_VERSION_EXTERN if v2 else ETA_VERSION)
    _u16(w, 0)  # flags
    _u32(w, len(c.names))
    _u32(w, len(c.channels))
    _u32(w, len(c.ports))
    _u32(w, len(c.stages))
    if v2:
        _u32(w, len(c.externs))
    for n in c.names:
        b = n.encode("utf-8")
        _u16(w, len(b))
        w += b
    for ch in c.channels:
        _u8(w, int(ch.dtype))
        encode_shape(w, ch.shape)
        _u32(w, ch.capacity)
        _u8(w, int(ch.host_role))
        _u8(w, 1 if ch.seeded else 0)
    for p in c.ports:
        _u8(w, int(p.port))
        _u8(w, 0)  # PortSource::Channel
        _u32(w, p.channel)
    for s in c.stages:
        _u8(w, int(s.stage))
        _u32(w, len(s.ops))
        for op in s.ops:
            encode_op(w, op)
    for e in c.externs:
        _u16(w, e.name)
        _u8(w, e.direction)
        _u32(w, e.chan)
    return bytes(w)


FNV_OFFSET = 0xCBF2_9CE4_8422_2325
FNV_PRIME = 0x0000_0100_0000_01B3


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return h


def container_hash(container_bytes: bytes) -> int:
    """FNV-1a 64 over the canonical container bytes — the pass's identity."""
    return fnv1a64(container_bytes)


# ---------------------------------------------------------------------------
# Expansions (`eta_ir::expand`): composed ops as fixed op sequences over the
# core, so a backend that fuses the core fuses these too. `push(op, step)`
# appends and returns the result id; `step` names the result's shape class.
# ---------------------------------------------------------------------------

STEP_ROW = 0
STEP_REDUCED = 1
STEP_SCALAR = 2
STEP_ROW_MASK = 3
STEP_REDUCED_INDEX = 4


def expand_gumbel(push, state: int, shape: Shape) -> int:
    return push(Op.rng_keyed(state, shape, RngKind.GUMBEL), STEP_ROW)


def expand_mask_apply(push, logits: int, mask: int) -> int:
    ninf = push(Op.const(Dtype.F32, float("-inf")), STEP_SCALAR)
    return push(Op.ternary(tags.SELECT, mask, logits, ninf), STEP_ROW)


def expand_softmax(push, x: int, shape: Shape) -> int:
    m = push(Op.unary(tags.REDUCE_MAX, x), STEP_REDUCED)
    mb = push(Op.broadcast(m, shape), STEP_ROW)
    c = push(Op.binary(tags.SUB, x, mb), STEP_ROW)
    e = push(Op.unary(tags.EXP, c), STEP_ROW)
    s = push(Op.unary(tags.REDUCE_SUM, e), STEP_REDUCED)
    sb = push(Op.broadcast(s, shape), STEP_ROW)
    return push(Op.binary(tags.DIV, e, sb), STEP_ROW)


def expand_log_softmax(push, x: int, shape: Shape) -> int:
    m = push(Op.unary(tags.REDUCE_MAX, x), STEP_REDUCED)
    mb = push(Op.broadcast(m, shape), STEP_ROW)
    c = push(Op.binary(tags.SUB, x, mb), STEP_ROW)
    e = push(Op.unary(tags.EXP, c), STEP_ROW)
    s = push(Op.unary(tags.REDUCE_SUM, e), STEP_REDUCED)
    lg = push(Op.unary(tags.LOG, s), STEP_REDUCED)
    lb = push(Op.broadcast(lg, shape), STEP_ROW)
    return push(Op.binary(tags.SUB, c, lb), STEP_ROW)


def expand_l2norm(push, x: int, shape: Shape) -> int:
    sq = push(Op.binary(tags.MUL, x, x), STEP_ROW)
    s = push(Op.unary(tags.REDUCE_SUM, sq), STEP_REDUCED)
    lg = push(Op.unary(tags.LOG, s), STEP_REDUCED)
    half = push(Op.const(Dtype.F32, 0.5), STEP_SCALAR)
    h = push(Op.binary(tags.MUL, lg, half), STEP_REDUCED)
    rt = push(Op.unary(tags.EXP, h), STEP_REDUCED)
    rb = push(Op.broadcast(rt, shape), STEP_ROW)
    return push(Op.binary(tags.DIV, x, rb), STEP_ROW)


def expand_nucleus_sample(push, logits: int, top_p: int, state: int, shape: Shape) -> int:
    probabilities = expand_softmax(push, logits, shape)
    keep = push(Op.pivot_threshold(probabilities, PRED_CUMMASS_LE, top_p), STEP_ROW_MASK)
    masked = expand_mask_apply(push, logits, keep)
    noise = expand_gumbel(push, state, shape)
    perturbed = push(Op.binary(tags.ADD, masked, noise), STEP_ROW)
    return push(Op.unary(tags.REDUCE_ARGMAX, perturbed), STEP_REDUCED_INDEX)

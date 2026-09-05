"""
`Tensor` — an SSA value — plus the free-function op surface. Port of
`eta-dsl/src/value.rs`: every op emits the IR's canonical `Op`, and the
composed ops (`softmax`, `gumbel_max`, `nucleus_sample`, …) inline the IR's
expansions so the emitted op stream is identical to the Rust SDK's.

Operators: `+ - * // %` and unary `-` on a `Tensor` are the arithmetic ops.
ETA has one `div` per dtype (truncating for integers), so `/` and `//` emit
the same op; write `//` on integer tensors so the source reads the way the
op behaves. `< <= > >=` are the ordering comparisons (a `bool` tensor);
`==`/`!=` stay ordinary Python identity so tensors can sit in sets and
dicts — spell elementwise equality `eq(a, b)` / `ne(a, b)`. A Python scalar
operand (`int` / `float` / `bool`) takes the dtype of the tensor it is
combined with.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Callable, Sequence

from . import ir
from .ir import (
    PRED_CUMMASS_LE,
    PRED_PROB_GE,
    PRED_RANK_LE,
    SCALAR,
    Dtype,
    Intrinsic,
    Op,
    RngKind,
    Shape,
    drop_last,
    numel,
    shape_of,
    tags,
)
from .trace import TraceError, ValueType, emit

__all__ = [
    "Tensor",
    "ConstData",
    "const_data",
    "constant",
    "neg",
    "abs_",
    "sign",
    "recip",
    "exp",
    "log",
    "cast",
    "add",
    "sub",
    "mul",
    "div",
    "rem",
    "max_elem",
    "min_elem",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and_",
    "or_",
    "not_",
    "select",
    "reshape",
    "broadcast",
    "transpose",
    "iota",
    "indptr",
    "gather",
    "gather_row",
    "scatter_set",
    "scatter_add",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_argmax",
    "cumsum",
    "cumprod",
    "softmax",
    "log_softmax",
    "l2norm",
    "top_k",
    "sort_desc",
    "rank_le",
    "cummass_le",
    "prob_ge",
    "pivot_threshold",
    "matmul",
    "gumbel",
    "rng",
    "mask_apply",
    "causal_mask",
    "sliding_window_mask",
    "sink_window_mask",
    "row_membership",
    "masked_argmax",
    "gumbel_max",
    "entropy",
    "entropy_from_logprobs",
    "scalar_gather",
    "nucleus_sample",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


@dataclass
class ConstData:
    """A trace-known constant value: a typed scalar/vector immediate. `data`
    holds raw little-endian element bytes (4 per element; bool is 1)."""

    shape: Shape
    dtype: Dtype
    data: bytes

    def elem(self, i: int) -> float:
        return _elem_at(self.dtype, self.data, i)


def _pack_scalar(v, dt: Dtype) -> bytes:
    if dt is Dtype.F32:
        return struct.pack("<f", float(v))
    if dt is Dtype.I32:
        return struct.pack("<i", int(v))
    if dt is Dtype.U32:
        return struct.pack("<I", int(v))
    if dt is Dtype.BOOL:
        return b"\x01" if v else b"\x00"
    raise ValueError(f"{dt!r} is not a dtype ETA computes in")


def _elem_at(dt: Dtype, data: bytes, i: int) -> float:
    if dt is Dtype.BOOL:
        return 1.0 if data[i] else 0.0
    if dt is Dtype.F32:
        return struct.unpack_from("<f", data, i * 4)[0]
    if dt is Dtype.I32:
        return float(struct.unpack_from("<i", data, i * 4)[0])
    if dt is Dtype.U32:
        return float(struct.unpack_from("<I", data, i * 4)[0])
    raise ValueError(f"{dt!r} is not a dtype ETA computes in")


def pack_elems(values: Sequence, dt: Dtype) -> bytes:
    """`values` as a little-endian payload of `dt` elements."""
    if dt is Dtype.F32:
        return struct.pack(f"<{len(values)}f", *[float(v) for v in values])
    if dt is Dtype.I32:
        return struct.pack(f"<{len(values)}i", *[int(v) for v in values])
    if dt is Dtype.U32:
        return struct.pack(f"<{len(values)}I", *[int(v) for v in values])
    if dt is Dtype.BOOL:
        return bytes(1 if v else 0 for v in values)
    raise ValueError(f"{dt!r} is not a dtype ETA computes in")


def unpack_elems(data: bytes, dt: Dtype) -> list:
    """The inverse of `pack_elems`: a payload as a list of Python numbers."""
    if dt is Dtype.BOOL:
        return [b != 0 for b in data]
    n = len(data) // 4
    fmt = {Dtype.F32: "f", Dtype.I32: "i", Dtype.U32: "I"}[dt]
    return list(struct.unpack(f"<{n}{fmt}", data[: n * 4]))


def const_data(v, dt: Dtype | None = None) -> ConstData:
    """Coerce a Python value into a `ConstData`.

    - `bool` → scalar `bool`; `float` → scalar `f32`; `int` → scalar `i32`
      (or `dt` when given — combine with a tensor and the scalar takes that
      tensor's dtype anyway).
    - a sequence of `bool` → `[n] bool`; of `float` → `[n] f32`; of `int` →
      **needs an explicit `dt`** (`dtype.i32` for tokens, `dtype.u32` for
      geometry), the way the Rust SDK needs a literal suffix.
    """
    if isinstance(v, ConstData):
        if dt is not None and dt is not v.dtype:
            raise TypeError(f"constant is {v.dtype.wire_name}, asked for {dt.wire_name}")
        return v
    if isinstance(v, Tensor):
        raise TypeError("a Tensor is not a constant")
    if isinstance(v, bool):
        d = dt if dt is not None else Dtype.BOOL
        return ConstData(SCALAR, d, _pack_scalar(v, d))
    if isinstance(v, int):
        d = dt if dt is not None else Dtype.I32
        return ConstData(SCALAR, d, _pack_scalar(v, d))
    if isinstance(v, float):
        d = dt if dt is not None else Dtype.F32
        return ConstData(SCALAR, d, _pack_scalar(v, d))
    if isinstance(v, (bytes, bytearray, memoryview)):
        if dt is None:
            raise TypeError("raw bytes need an explicit dtype")
        raw = bytes(v)
        n = len(raw) // dt.elem_size
        if n * dt.elem_size != len(raw):
            raise ValueError(f"{len(raw)} bytes is not a whole number of {dt.wire_name} elements")
        return ConstData(shape_of([n]) if n else SCALAR, dt, raw)
    try:
        items = list(v)
    except TypeError:
        raise TypeError(f"cannot make a constant from {type(v).__name__}") from None
    if dt is None:
        if items and all(isinstance(x, bool) for x in items):
            dt = Dtype.BOOL
        elif items and all(isinstance(x, float) for x in items):
            dt = Dtype.F32
        elif items and all(isinstance(x, int) for x in items):
            raise TypeError(
                "an integer sequence needs an explicit dtype (dtype.i32 for tokens, "
                "dtype.u32 for geometry): pass dtype=..."
            )
        else:
            raise TypeError("cannot infer a dtype for this sequence; pass dtype=...")
    if not items:
        raise ValueError("a constant needs at least one element (every extent is >= 1)")
    return ConstData(shape_of([len(items)]), dt, pack_elems(items, dt))


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


class Tensor:
    """An SSA value: a node in the current stage, or a deferred trace-known
    constant that materializes into ops on first use."""

    __slots__ = ("_id", "_ty", "_const")

    def __init__(self, *, _id: int | None = None, _ty: ValueType | None = None, _const: ConstData | None = None):
        self._id = _id
        self._ty = _ty
        self._const = _const

    @staticmethod
    def node(vid: int, ty: ValueType) -> "Tensor":
        return Tensor(_id=vid, _ty=ty)

    @staticmethod
    def constant(v, dt: Dtype | None = None) -> "Tensor":
        """A trace-known constant. Only a uniform tensor (broadcast) or a
        `u32` affine ramp `a + b*i` (iota) can lower; bulk data belongs in a
        seeded channel instead."""
        return Tensor(_const=const_data(v, dt))

    @property
    def is_const(self) -> bool:
        return self._const is not None

    @property
    def ty(self) -> ValueType:
        if self._const is not None:
            return ValueType(self._const.shape, self._const.dtype)
        return self._ty  # type: ignore[return-value]

    @property
    def dtype(self) -> Dtype:
        return self.ty.dtype

    @property
    def shape(self) -> Shape:
        return self.ty.shape

    def __repr__(self) -> str:
        ty = self.ty
        if self._const is not None:
            return f"Tensor(const {ty.dtype.wire_name}{list(ty.shape)})"
        return f"Tensor(%{self._id}: {ty.dtype.wire_name}{list(ty.shape)})"

    # -- arithmetic operators ---------------------------------------------

    def __add__(self, o):
        return add(self, o)

    def __radd__(self, o):
        return add(o, self)

    def __sub__(self, o):
        return sub(self, o)

    def __rsub__(self, o):
        return sub(o, self)

    def __mul__(self, o):
        return mul(self, o)

    def __rmul__(self, o):
        return mul(o, self)

    def __truediv__(self, o):
        return div(self, o)

    def __rtruediv__(self, o):
        return div(o, self)

    def __floordiv__(self, o):
        return div(self, o)

    def __rfloordiv__(self, o):
        return div(o, self)

    def __mod__(self, o):
        return rem(self, o)

    def __rmod__(self, o):
        return rem(o, self)

    def __neg__(self):
        return neg(self)

    # Ordering comparisons yield a bool tensor. Python reflects `3 < t` into
    # `t.__gt__(3)`, so the four forward methods cover both operand orders.
    def __lt__(self, o):
        return lt(self, o)

    def __le__(self, o):
        return le(self, o)

    def __gt__(self, o):
        return gt(self, o)

    def __ge__(self, o):
        return ge(self, o)

    def __and__(self, o):
        return and_(self, o)

    def __or__(self, o):
        return or_(self, o)

    def __invert__(self):
        return not_(self)

    def __bool__(self):
        raise TypeError("a traced Tensor has no Python truth value; use select()/and_()/or_()")

    __hash__ = object.__hash__

    def div_ceil(self, rhs) -> "Tensor":
        """Ceiling division, spelled like `u32::div_ceil`. A trace-known
        scalar divisor has its `- 1` folded here, so the emitted ops match the
        hand-written form."""
        d = _to_arg(rhs)
        v = _const_scalar(d)
        if v is not None:
            one_less = Tensor(_const=ConstData(SCALAR, d.dtype, _pack_scalar(v - 1.0, d.dtype)))
            return (self + one_less) / d
        return (self + d - constant(1, Dtype.U32)) / d


def constant(v, dt: Dtype | None = None) -> Tensor:
    return Tensor.constant(v, dt)


# ---------------------------------------------------------------------------
# Operand plumbing
# ---------------------------------------------------------------------------


def _to_arg(x) -> Tensor:
    """Anything usable as an operand, as a `Tensor` (node or const)."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (bool, int, float, ConstData)):
        return Tensor(_const=const_data(x))
    raise TypeError(f"{type(x).__name__} is not a tensor operand")


def _const_scalar(a: Tensor) -> float | None:
    c = a._const
    if c is None or c.shape != SCALAR:
        return None
    return c.elem(0)


def _scalar_literal_op(dt: Dtype, data: bytes) -> Op:
    if dt is Dtype.F32:
        return Op.const(dt, struct.unpack_from("<f", data, 0)[0])
    if dt is Dtype.I32:
        return Op.const(dt, struct.unpack_from("<i", data, 0)[0])
    if dt is Dtype.U32:
        return Op.const(dt, struct.unpack_from("<I", data, 0)[0])
    return Op.const(dt, data[0] != 0)


def _materialize_const(c: ConstData) -> tuple[int, ValueType]:
    """Lower a trace-known constant to IR ops (see `Tensor.constant`)."""
    ty = ValueType(c.shape, c.dtype)
    if c.shape == SCALAR:
        vid = emit(_scalar_literal_op(c.dtype, c.data), (ValueType(SCALAR, c.dtype),))
        return vid, ty
    n = numel(c.shape)
    vals = [c.elem(i) for i in range(n)]
    # uniform ⇒ broadcast(scalar).
    if vals and all(v == vals[0] for v in vals):
        s = emit(
            _scalar_literal_op(c.dtype, c.data[: c.dtype.elem_size]),
            (ValueType(SCALAR, c.dtype),),
        )
        vid = emit(Op.broadcast(s, c.shape), (ty,))
        return vid, ty
    # affine `a + b*i` over U32 ⇒ iota (+ optional mul/add).
    if c.dtype is Dtype.U32 and n >= 2:
        a = vals[0]
        b = vals[1] - vals[0]
        if b >= 0.0 and all(v == a + b * i for i, v in enumerate(vals)):
            cur = emit(Op.iota(n), (ty,))
            if b != 1.0:
                bc = emit(Op.const(Dtype.U32, int(b)), (ValueType(SCALAR, Dtype.U32),))
                cur = emit(Op.binary(tags.MUL, cur, bc), (ty,))
            if a != 0.0:
                ac = emit(Op.const(Dtype.U32, int(a)), (ValueType(SCALAR, Dtype.U32),))
                cur = emit(Op.binary(tags.ADD, cur, ac), (ty,))
            return cur, ty
    raise TraceError(
        f"a {c.dtype.wire_name} constant of shape {list(c.shape)} is bulk data, and the op "
        "set carries constants as scalars: `const` holds one literal, so only a uniform "
        "tensor (broadcast) and a u32 affine ramp a+b*i (iota) are reachable from it. Seed "
        "a channel with the values and read it in the body — `Channel.from_(values)` — or "
        "build the tensor from an arithmetic expression"
    )


def _materialize(a: Tensor) -> tuple[int, ValueType]:
    if a._const is not None:
        return _materialize_const(a._const)
    return a._id, a._ty  # type: ignore[return-value]


def _coerce(c: ConstData, to: Dtype) -> ConstData | None:
    if c.dtype is to or c.shape != SCALAR:
        return None
    return ConstData(SCALAR, to, _pack_scalar(c.elem(0), to))


def _reconcile(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    if a._const is not None and b._const is None:
        c = _coerce(a._const, b.dtype)
        if c is not None:
            return Tensor(_const=c), b
    elif a._const is None and b._const is not None:
        c = _coerce(b._const, a.dtype)
        if c is not None:
            return a, Tensor(_const=c)
    return a, b


def _non_scalar_shape(a: Shape, b: Shape) -> Shape:
    return b if a == SCALAR else a


def _emit_unary(x, tag: int, out: Callable[[ValueType], ValueType]) -> Tensor:
    vid, ty = _materialize(_to_arg(x))
    rty = out(ty)
    return Tensor.node(emit(Op.unary(tag, vid), (rty,)), rty)


def _emit_binary(a, b, tag: int, result_dtype: Callable[[Dtype], Dtype]) -> Tensor:
    aa, bb = _reconcile(_to_arg(a), _to_arg(b))
    shape = _non_scalar_shape(aa.shape, bb.shape)
    ia, tya = _materialize(aa)
    ib, _ = _materialize(bb)
    rty = ValueType(shape, result_dtype(tya.dtype))
    return Tensor.node(emit(Op.binary(tag, ia, ib), (rty,)), rty)


def _same(t: ValueType) -> ValueType:
    return t


def _identity_dtype(d: Dtype) -> Dtype:
    return d


def _bool_dtype(_: Dtype) -> Dtype:
    return Dtype.BOOL


# ---------------------------------------------------------------------------
# The free-function op surface
# ---------------------------------------------------------------------------


def neg(x) -> Tensor:
    return _emit_unary(x, tags.NEG, _same)


def abs_(x) -> Tensor:
    return _emit_unary(x, tags.ABS, _same)


def sign(x) -> Tensor:
    return _emit_unary(x, tags.SIGN, _same)


def recip(x) -> Tensor:
    return _emit_unary(x, tags.RECIP, _same)


def exp(x) -> Tensor:
    return _emit_unary(x, tags.EXP, _same)


def log(x) -> Tensor:
    return _emit_unary(x, tags.LOG, _same)


def cast(x, to: Dtype) -> Tensor:
    """`x` converted elementwise to `to`. A cast to the dtype `x` already has
    is the identity and emits nothing."""
    t = _to_arg(x)
    if t.dtype is to:
        return t
    vid, ty = _materialize(t)
    rty = ValueType(ty.shape, to)
    return Tensor.node(emit(Op.cast(vid, to), (rty,)), rty)


def add(a, b) -> Tensor:
    return _emit_binary(a, b, tags.ADD, _identity_dtype)


def sub(a, b) -> Tensor:
    return _emit_binary(a, b, tags.SUB, _identity_dtype)


def mul(a, b) -> Tensor:
    return _emit_binary(a, b, tags.MUL, _identity_dtype)


def div(a, b) -> Tensor:
    return _emit_binary(a, b, tags.DIV, _identity_dtype)


def rem(a, b) -> Tensor:
    return _emit_binary(a, b, tags.REM, _identity_dtype)


def max_elem(a, b) -> Tensor:
    return _emit_binary(a, b, tags.MAX_ELEM, _identity_dtype)


def min_elem(a, b) -> Tensor:
    return _emit_binary(a, b, tags.MIN_ELEM, _identity_dtype)


def eq(a, b) -> Tensor:
    return _emit_binary(a, b, tags.EQ, _bool_dtype)


def ne(a, b) -> Tensor:
    return _emit_binary(a, b, tags.NE, _bool_dtype)


def lt(a, b) -> Tensor:
    return _emit_binary(a, b, tags.LT, _bool_dtype)


def le(a, b) -> Tensor:
    return _emit_binary(a, b, tags.LE, _bool_dtype)


def gt(a, b) -> Tensor:
    return _emit_binary(a, b, tags.GT, _bool_dtype)


def ge(a, b) -> Tensor:
    return _emit_binary(a, b, tags.GE, _bool_dtype)


def and_(a, b) -> Tensor:
    return _emit_binary(a, b, tags.AND, _bool_dtype)


def or_(a, b) -> Tensor:
    return _emit_binary(a, b, tags.OR, _bool_dtype)


def not_(x) -> Tensor:
    return _emit_unary(x, tags.NOT, lambda t: ValueType(t.shape, Dtype.BOOL))


def select(cond, a, b) -> Tensor:
    ca, _ = _materialize(_to_arg(cond))
    aa, bb = _reconcile(_to_arg(a), _to_arg(b))
    shape = _non_scalar_shape(aa.shape, bb.shape)
    ia, tya = _materialize(aa)
    ib, _ = _materialize(bb)
    rty = ValueType(shape, tya.dtype)
    return Tensor.node(emit(Op.ternary(tags.SELECT, ca, ia, ib), (rty,)), rty)


def reshape(x, shape) -> Tensor:
    s = shape_of(shape)
    vid, ty = _materialize(_to_arg(x))
    rty = ValueType(s, ty.dtype)
    return Tensor.node(emit(Op.reshape(vid, s), (rty,)), rty)


def broadcast(x, shape) -> Tensor:
    s = shape_of(shape)
    vid, ty = _materialize(_to_arg(x))
    rty = ValueType(s, ty.dtype)
    return Tensor.node(emit(Op.broadcast(vid, s), (rty,)), rty)


def transpose(x) -> Tensor:
    def out(t: ValueType) -> ValueType:
        d = t.shape
        s = (d[1], d[0]) if len(d) == 2 else d
        return ValueType(s, t.dtype)

    return _emit_unary(x, tags.TRANSPOSE, out)


def iota(length: int) -> Tensor:
    ty = ValueType(shape_of([length]), Dtype.U32)
    return Tensor.node(emit(Op.iota(length), (ty,)), ty)


def indptr(rows: int, run_len) -> Tensor:
    """The CSR row-offset vector for `rows` runs of equal length `run_len`."""
    n = rows + 1
    return iota(n) * broadcast(run_len, [n])


def gather(src, idx) -> Tensor:
    is_, tys = _materialize(_to_arg(src))
    ii, tyi = _materialize(_to_arg(idx))
    dims = list(tyi.shape) + list(tys.shape[min(len(tys.shape), 1) :])
    try:
        rshape = shape_of(dims)
    except ValueError:
        raise TraceError(
            f"gather of {list(tys.shape)} by {list(tyi.shape)} has result shape {dims}, "
            f"whose rank exceeds {ir.MAX_RANK}"
        ) from None
    rty = ValueType(rshape, tys.dtype)
    return Tensor.node(emit(Op.binary(tags.GATHER, is_, ii), (rty,)), rty)


def gather_row(src, idx) -> Tensor:
    is_, tys = _materialize(_to_arg(src))
    ii, _ = _materialize(_to_arg(idx))
    m = tys.shape[0] if tys.shape else 0
    rty = ValueType(shape_of([m]), tys.dtype)
    return Tensor.node(emit(Op.binary(tags.GATHER_ROW, is_, ii), (rty,)), rty)


def _scatter(tag: int, base, idx, vals) -> Tensor:
    # A scalar-literal `vals` takes the base's dtype, the way the Rust
    # author's literal suffix would say.
    bb, vv = _reconcile(_to_arg(base), _to_arg(vals))
    ib, tyb = _materialize(bb)
    ii, _ = _materialize(_to_arg(idx))
    iv, _ = _materialize(vv)
    return Tensor.node(emit(Op.ternary(tag, ib, ii, iv), (tyb,)), tyb)


def scatter_set(base, idx, vals) -> Tensor:
    return _scatter(tags.SCATTER_SET, base, idx, vals)


def scatter_add(base, idx, vals) -> Tensor:
    return _scatter(tags.SCATTER_ADD, base, idx, vals)


def _reduced(t: ValueType) -> ValueType:
    return ValueType(drop_last(t.shape), t.dtype)


def reduce_sum(x) -> Tensor:
    return _emit_unary(x, tags.REDUCE_SUM, _reduced)


def reduce_max(x) -> Tensor:
    return _emit_unary(x, tags.REDUCE_MAX, _reduced)


def reduce_min(x) -> Tensor:
    return _emit_unary(x, tags.REDUCE_MIN, _reduced)


def reduce_argmax(x) -> Tensor:
    return _emit_unary(x, tags.REDUCE_ARGMAX, lambda t: ValueType(drop_last(t.shape), Dtype.I32))


def cumsum(x) -> Tensor:
    return _emit_unary(x, tags.CUMSUM, _same)


def cumprod(x) -> Tensor:
    return _emit_unary(x, tags.CUMPROD, _same)


# -- expansions ---------------------------------------------------------------


class _Traced:
    """Records `ir.expand_*` steps into the trace, attaching the result type
    each step lands in."""

    def __init__(self, row: ValueType) -> None:
        self.row = row
        self.reduced = ValueType(drop_last(row.shape), row.dtype)

    def push(self, op: Op, step: int) -> int:
        if step == ir.STEP_ROW:
            ty = self.row
        elif step == ir.STEP_REDUCED:
            ty = self.reduced
        elif step == ir.STEP_SCALAR:
            ty = ValueType(SCALAR, Dtype.F32)
        elif step == ir.STEP_ROW_MASK:
            ty = ValueType(self.row.shape, Dtype.BOOL)
        else:
            ty = ValueType(self.reduced.shape, Dtype.I32)
        return emit(op, (ty,))


def _expanded(x, seq) -> Tensor:
    xid, ty = _materialize(_to_arg(x))
    row = ValueType(ty.shape, Dtype.F32)
    sink = _Traced(row)
    return Tensor.node(seq(sink.push, xid, ty.shape), row)


def softmax(x) -> Tensor:
    return _expanded(x, ir.expand_softmax)


def log_softmax(x) -> Tensor:
    return _expanded(x, ir.expand_log_softmax)


def l2norm(x) -> Tensor:
    return _expanded(x, ir.expand_l2norm)


# -- order --------------------------------------------------------------------


def top_k(x, k: int) -> tuple[Tensor, Tensor]:
    ix, tyx = _materialize(_to_arg(x))
    dims = list(tyx.shape)
    if dims:
        dims[-1] = k
    try:
        out_shape = shape_of(dims)
    except ValueError:
        out_shape = shape_of([k])
    val_ty = ValueType(out_shape, tyx.dtype)
    idx_ty = ValueType(out_shape, Dtype.U32)
    base = emit(Op.top_k(ix, k), (val_ty, idx_ty))
    return Tensor.node(base, val_ty), Tensor.node(base + 1, idx_ty)


def sort_desc(x) -> tuple[Tensor, Tensor]:
    ix, tyx = _materialize(_to_arg(x))
    n = tyx.shape[-1] if tyx.shape else 0
    val_ty = ValueType(shape_of([n]), Dtype.F32)
    idx_ty = ValueType(shape_of([n]), Dtype.U32)
    base = emit(Op.unary(tags.SORT_DESC, ix), (val_ty, idx_ty))
    return Tensor.node(base, val_ty), Tensor.node(base + 1, idx_ty)


class Predicate:
    __slots__ = ("tag", "arg")

    def __init__(self, tag: int, arg: Tensor) -> None:
        self.tag = tag
        self.arg = arg


def rank_le(k) -> Predicate:
    return Predicate(PRED_RANK_LE, _to_arg(k))


def cummass_le(p) -> Predicate:
    return Predicate(PRED_CUMMASS_LE, _to_arg(p))


def prob_ge(thr) -> Predicate:
    return Predicate(PRED_PROB_GE, _to_arg(thr))


def pivot_threshold(x, predicate: Predicate) -> Tensor:
    ii, tyi = _materialize(_to_arg(x))
    pv, _ = _materialize(predicate.arg)
    rty = ValueType(tyi.shape, Dtype.BOOL)
    return Tensor.node(emit(Op.pivot_threshold(ii, predicate.tag, pv), (rty,)), rty)


def matmul(a, b) -> Tensor:
    ia, tya = _materialize(_to_arg(a))
    ib, tyb = _materialize(_to_arg(b))
    m = tya.shape[0] if tya.shape else 0
    n = tyb.shape[-1] if tyb.shape else 0
    rty = ValueType(shape_of([m, n]), Dtype.F32)
    return Tensor.node(emit(Op.binary(tags.MATMUL, ia, ib), (rty,)), rty)


# -- sampling -----------------------------------------------------------------


def _rng_noise(state, shape, kind: RngKind) -> Tensor:
    s = shape_of(shape)
    istate, _ = _materialize(_to_arg(state))
    rty = ValueType(s, Dtype.F32)
    return Tensor.node(emit(Op.rng_keyed(istate, s, kind), (rty,)), rty)


def gumbel(state, shape) -> Tensor:
    return _rng_noise(state, shape, RngKind.GUMBEL)


def rng(state, shape) -> Tensor:
    return _rng_noise(state, shape, RngKind.UNIFORM)


def mask_apply(logits, mask) -> Tensor:
    il, tyl = _materialize(_to_arg(logits))
    im, _ = _materialize(_to_arg(mask))
    sink = _Traced(tyl)
    return Tensor.node(ir.expand_mask_apply(sink.push, il, im), tyl)


def _append_mask_axis(shape: Shape, length: int) -> Shape:
    dims = list(shape) + [length]
    try:
        return shape_of(dims)
    except ValueError:
        raise TraceError(
            f"a structured mask over {list(shape)} with length {length} has shape {dims}, "
            f"whose rank exceeds {ir.MAX_RANK}"
        ) from None


def causal_mask(positions, length: int) -> Tensor:
    p, ty = _materialize(_to_arg(positions))
    rty = ValueType(_append_mask_axis(ty.shape, length), Dtype.BOOL)
    return Tensor.node(emit(Op.causal_mask(p, length), (rty,)), rty)


def sliding_window_mask(positions, length: int, window: int) -> Tensor:
    p, ty = _materialize(_to_arg(positions))
    rty = ValueType(_append_mask_axis(ty.shape, length), Dtype.BOOL)
    return Tensor.node(emit(Op.sliding_window_mask(p, length, window), (rty,)), rty)


def sink_window_mask(positions, length: int, sink: int, window: int) -> Tensor:
    p, ty = _materialize(_to_arg(positions))
    rty = ValueType(_append_mask_axis(ty.shape, length), Dtype.BOOL)
    return Tensor.node(emit(Op.sink_window_mask(p, length, sink, window), (rty,)), rty)


def row_membership(rows, keys) -> Tensor:
    """For every row and key, whether the key occurs anywhere in the row.
    Ordinary SSA composition; no wire opcode."""
    rows_t = _to_arg(rows)
    keys_t = _to_arg(keys)
    row_type = rows_t.ty
    key_type = keys_t.ty
    if len(row_type.shape) != 2:
        raise TraceError(f"row_membership rows must have shape [R, D], got {list(row_type.shape)}")
    row_count, depth = row_type.shape
    if len(key_type.shape) != 1:
        raise TraceError(f"row_membership keys must have shape [K], got {list(key_type.shape)}")
    (key_count,) = key_type.shape
    if row_type.dtype is not key_type.dtype:
        raise TraceError(
            f"row_membership rows and keys must have the same dtype, got "
            f"{row_type.dtype.wire_name} and {key_type.dtype.wire_name}"
        )
    row_stride = key_count * depth
    row_flat_len = row_count * depth
    flat_len = row_count * key_count * depth
    if max(row_stride, row_flat_len, flat_len) > 0xFFFF_FFFF:
        raise TraceError(
            f"row_membership over {row_count} rows x {key_count} keys x depth {depth} needs a "
            f"{flat_len}-element intermediate, which overflows the wire's u32 extents"
        )
    rid, _ = _materialize(rows_t)
    kid, _ = _materialize(keys_t)
    rows_n = Tensor.node(rid, row_type)
    keys_n = Tensor.node(kid, key_type)
    linear = iota(flat_len)
    row_index = div(linear, constant(row_stride, Dtype.U32))
    depth_index = rem(linear, constant(depth, Dtype.U32))
    row_value_index = add(mul(row_index, constant(depth, Dtype.U32)), depth_index)
    row_values = gather(reshape(rows_n, [row_flat_len]), row_value_index)
    key_index = rem(div(linear, constant(depth, Dtype.U32)), constant(key_count, Dtype.U32))
    key_values = gather(keys_n, key_index)
    matches = eq(
        reshape(row_values, [row_count, key_count, depth]),
        reshape(key_values, [row_count, key_count, depth]),
    )
    return cast(reduce_max(cast(matches, Dtype.U32)), Dtype.BOOL)


def masked_argmax(logits, mask) -> Tensor:
    lid, lty = _materialize(_to_arg(logits))
    mid, _ = _materialize(_to_arg(mask))
    result_type = ValueType(drop_last(lty.shape), Dtype.I32)
    ninf = emit(Op.const(Dtype.F32, -math.inf), (ValueType(SCALAR, Dtype.F32),))
    masked = emit(Op.ternary(tags.SELECT, mid, lid, ninf), (lty,))
    result = emit(Op.unary(tags.REDUCE_ARGMAX, masked), (result_type,))
    return Tensor.node(result, result_type)


def gumbel_max(logits, state) -> Tensor:
    """Semantic Gumbel-max sampler over the input's complete shape."""
    lid, lty = _materialize(_to_arg(logits))
    sid, _ = _materialize(_to_arg(state))
    result_type = ValueType(drop_last(lty.shape), Dtype.I32)
    noise = emit(Op.rng_keyed(sid, lty.shape, RngKind.GUMBEL), (ValueType(lty.shape, Dtype.F32),))
    perturbed = emit(Op.binary(tags.ADD, lid, noise), (lty,))
    result = emit(Op.unary(tags.REDUCE_ARGMAX, perturbed), (result_type,))
    return Tensor.node(result, result_type)


F32_MIN_POSITIVE = 2.0**-126  # `f32::MIN_POSITIVE`, the smallest normal


def entropy(probabilities) -> Tensor:
    """Shannon entropy `-sum(p * log(p))`. A softmax tail underflows to
    exactly 0 in f32 and `0 * log 0` is NaN, so the log sees the
    probabilities floored at the smallest normal: a zero term contributes 0."""
    pid, pty = _materialize(_to_arg(probabilities))
    result_type = ValueType(drop_last(pty.shape), Dtype.F32)
    fid, _ = _materialize(max_elem(Tensor.node(pid, pty), F32_MIN_POSITIVE))
    lp = emit(Op.unary(tags.LOG, fid), (pty,))
    terms = emit(Op.binary(tags.MUL, pid, lp), (pty,))
    s = emit(Op.unary(tags.REDUCE_SUM, terms), (result_type,))
    result = emit(Op.unary(tags.NEG, s), (result_type,))
    return Tensor.node(result, result_type)


def entropy_from_logprobs(probabilities, log_probabilities) -> Tensor:
    pid, pty = _materialize(_to_arg(probabilities))
    lid, _ = _materialize(_to_arg(log_probabilities))
    result_type = ValueType(drop_last(pty.shape), Dtype.F32)
    terms = emit(Op.binary(tags.MUL, pid, lid), (pty,))
    s = emit(Op.unary(tags.REDUCE_SUM, terms), (result_type,))
    result = emit(Op.unary(tags.NEG, s), (result_type,))
    return Tensor.node(result, result_type)


def scalar_gather(src, index) -> Tensor:
    sid, sty = _materialize(_to_arg(src))
    iid, ity = _materialize(_to_arg(index))
    if len(sty.shape) == 2:
        rows_ = sty.shape[0]
        if tuple(ity.shape) != (rows_,):
            raise TraceError(
                f"scalar_gather over a {list(sty.shape)} matrix requires one index per row "
                f"([{rows_}]), got {list(ity.shape)}"
            )
        op = Op.binary(tags.GATHER_ROW, sid, iid)
        result_shape = shape_of([rows_])
    else:
        dims = list(ity.shape) + list(sty.shape[min(len(sty.shape), 1) :])
        try:
            result_shape = shape_of(dims)
        except ValueError:
            raise TraceError(
                f"scalar_gather of {list(sty.shape)} by {list(ity.shape)} has result shape "
                f"{dims}, whose rank exceeds {ir.MAX_RANK}"
            ) from None
        op = Op.binary(tags.GATHER, sid, iid)
    result_type = ValueType(result_shape, sty.dtype)
    return Tensor.node(emit(op, (result_type,)), result_type)


def nucleus_sample(logits, top_p, state) -> Tensor:
    """Exact nucleus sampler as ordinary composable SSA (the IR's
    `nucleus_sample` expansion, which the region matcher fuses)."""
    lid, lty = _materialize(_to_arg(logits))
    pid, _ = _materialize(_to_arg(top_p))
    sid, _ = _materialize(_to_arg(state))
    sink = _Traced(lty)
    token_type = ValueType(drop_last(lty.shape), Dtype.I32)
    result = ir.expand_nucleus_sample(sink.push, lid, pid, sid, lty.shape)
    return Tensor.node(result, token_type)


# -- intrinsic leaf / internal helpers -------------------------------------------


def intrinsic_val(intr: Intrinsic, shape: Shape, dt: Dtype) -> Tensor:
    ty = ValueType(shape, dt)
    return Tensor.node(emit(Op.intrinsic_val(intr, shape, dt), (ty,)), ty)


def reshape_id_to(vid: int, frm: ValueType, target: Shape) -> int:
    """Reshape a value id to `target` if it differs (used by `Channel.put` to
    fit a scalar into a `[1]` channel)."""
    if frm.shape == target:
        return vid
    return emit(Op.reshape(vid, target), (ValueType(target, frm.dtype),))


def materialize(x) -> tuple[int, ValueType]:
    """Resolve an operand to an SSA id + type inside a traced stage."""
    return _materialize(_to_arg(x))

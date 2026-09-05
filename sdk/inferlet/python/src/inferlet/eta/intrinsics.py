"""
`intrinsics.*`: first-party stage-scoped values and model constants, and the
`intrinsics.kernel.*` second-party surface. Port of `eta-dsl/src/intrinsics.rs`.
"""

from __future__ import annotations

from .ir import ATTN_SCORE_KV_MAX, SCALAR, Dtype, Intrinsic, Op, SinkScope, shape_of
from .trace import ValueType, current_rows, emit, intern_name, record_sink
from .trace import page_size as _page_size
from .trace import vocab as _vocab
from .value import Tensor, intrinsic_val, materialize, reshape

# The interpreter-visible activation dtype. Always F32.
activation_type = Dtype.F32


def vocab() -> int:
    """Model vocabulary size (trace-known)."""
    return _vocab()


def page_size() -> int:
    """Tokens per KV page (trace-known)."""
    return _page_size()


def _single_row_reshape(t: Tensor) -> Tensor:
    s = t.shape
    if len(s) == 2 and s[0] == 1:
        return reshape(t, [s[1]])
    return t


def logits() -> Tensor:
    """The LM-head logits, `[n_out, vocab]` f32 — reshaped to `[vocab]` for a
    single read-out row so single-position samplers read a vector."""
    rows = max(current_rows(), 1)
    t = intrinsic_val(Intrinsic.LOGITS, shape_of([rows, vocab()]), Dtype.F32)
    return _single_row_reshape(t)


def mtp_logits(k: int) -> Tensor:
    """The model's `k` draft/future-token heads, `[k, vocab]`; model-gated."""
    return intrinsic_val(Intrinsic.MTP_LOGITS, shape_of([k, vocab()]), Dtype.F32)


def mtp_drafts(n: int) -> Tensor:
    """The draft head's token ids for the fire's readout rows, `[n]` i32."""
    return intrinsic_val(Intrinsic.MTP_DRAFTS, shape_of([max(n, 1)]), Dtype.I32)


def hidden(width: int) -> Tensor:
    """The residual stream at read-out (epilogue), `[n_out, width]`."""
    rows = max(current_rows(), 1)
    return intrinsic_val(Intrinsic.HIDDEN, shape_of([rows, max(width, 1)]), activation_type)


def query(width: int) -> Tensor:
    """This layer's projected query (attn taps), `[width]`."""
    return intrinsic_val(Intrinsic.QUERY, shape_of([max(width, 1)]), activation_type)


def value_head() -> Tensor:
    """Model-gated scalar value head (epilogue), `[rows]`."""
    return intrinsic_val(Intrinsic.VALUE_HEAD, shape_of([max(current_rows(), 1)]), Dtype.F32)


def layer() -> Tensor:
    """The invocation's layer index (attn taps; u32 scalar)."""
    return intrinsic_val(Intrinsic.LAYER, SCALAR, Dtype.U32)


def attn_score(planes: int) -> Tensor:
    """Per-(layer, head) attention mass over live KV, `[planes, ATTN_SCORE_KV_MAX]`."""
    return intrinsic_val(
        Intrinsic.ATTN_SCORE, shape_of([max(planes, 1), ATTN_SCORE_KV_MAX]), Dtype.F32
    )


def attn_score_kv_max() -> int:
    return ATTN_SCORE_KV_MAX


class kernel:  # noqa: N801 — spelled like `intrinsics::kernel::*`.
    """Second-party kernels and configuration sinks."""

    @staticmethod
    def envelope_dot(p_max: int) -> Tensor:
        """Quest page criticality for this layer, `[p_max]` f32; model-gated."""
        query_ty = ValueType(shape_of([1]), activation_type)
        q = emit(Op.intrinsic_val(Intrinsic.QUERY, query_ty.shape, query_ty.dtype), (query_ty,))
        score_ty = ValueType(shape_of([p_max]), Dtype.F32)
        name = intern_name("envelope_dot")
        return Tensor.node(
            emit(Op.kernel_call(name, [q], score_ty.shape, score_ty.dtype), (score_ty,)),
            score_ty,
        )

    @staticmethod
    def attn_page_mask(mask) -> None:
        """A configuration sink: this layer's attention consumes the page mask."""
        mid, _ = materialize(mask)
        name = intern_name("attn_page_mask")
        emit(Op.sink_call(name, [mid]), ())
        record_sink("attn_page_mask", SinkScope.ATTENTION)

    @staticmethod
    def lora(a, b, sites) -> None:
        """Pass-wide sink: apply the low-rank delta `B(Ax)` at `sites`.
        Legal only in the prologue."""
        aid, _ = materialize(a)
        bid, _ = materialize(b)
        sid, _ = materialize(sites)
        name = intern_name("lora")
        emit(Op.sink_call(name, [aid, bid, sid]), ())
        record_sink("lora", SinkScope.PASS_WIDE)

    @staticmethod
    def adapter_scale(l, sites) -> None:  # noqa: E741
        """The scale form of the adapter sink (IA3): `y = l ⊙ y` at `sites`."""
        lid, _ = materialize(l)
        sid, _ = materialize(sites)
        name = intern_name("lora")
        emit(Op.sink_call(name, [lid, sid]), ())
        record_sink("lora", SinkScope.PASS_WIDE)

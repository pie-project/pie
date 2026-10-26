"""
`intrinsics.*`: first-party stage-scoped values and model constants, and the
`intrinsics.kernel.*` second-party surface. Port of `eta-dsl/src/intrinsics.rs`.
"""

from __future__ import annotations

from .ir import Axis, Dtype, Op, OutputDecl, shape_of
from .trace import current_rows, declared_output, emit, intern_name, record_sink
from .trace import page_size as _page_size
from .trace import vocab as _vocab
from .value import Tensor, materialize, output_val, reshape

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


def _declared(lane: int, name: str) -> OutputDecl:
    decl = declared_output(lane, name)
    if decl is None:
        raise ValueError(f"lane {lane} of this pass reads out no output `{name}`")
    return decl


def output(lane: int, name: str) -> Tensor:
    """Lane `lane`'s output `name` as the model declares it: its readout rows
    (a lone row read as its width alone), or its layers stacked."""
    decl = _declared(lane, name)
    if decl.axis == Axis.LANE:
        rows = max(decl.layers, 1)
    elif decl.readout:
        rows = max(current_rows(), 1)
    else:
        raise ValueError(
            f"output `{name}` is one row a token or voxel of lane {lane}: "
            "`output_rows` reads the rows it names"
        )
    t = output_val(intern_name(name), lane, shape_of([rows, decl.width]), decl.dtype)
    return _single_row_reshape(t) if decl.readout else t


def output_rows(lane: int, name: str, rows: int) -> Tensor:
    """`rows` rows of lane `lane`'s output `name`, one a token or voxel."""
    decl = _declared(lane, name)
    return output_val(intern_name(name), lane, shape_of([max(rows, 1), decl.width]), decl.dtype)


class kernel:  # noqa: N801 — spelled like `intrinsics::kernel::*`.
    """Second-party kernels and configuration sinks."""

    @staticmethod
    def attn_page_mask(mask) -> None:
        """A configuration sink: this layer's attention consumes the page mask."""
        mid, _ = materialize(mask)
        name = intern_name("attn_page_mask")
        emit(Op.sink_call(name, [mid]), ())
        record_sink("attn_page_mask")

    @staticmethod
    def lora(a, b, sites) -> None:
        """Pass-wide sink: apply the low-rank delta `B(Ax)` at `sites`.
        Legal only in the prologue."""
        aid, _ = materialize(a)
        bid, _ = materialize(b)
        sid, _ = materialize(sites)
        name = intern_name("lora")
        emit(Op.sink_call(name, [aid, bid, sid]), ())
        record_sink("lora")

    @staticmethod
    def adapter_scale(l, sites) -> None:  # noqa: E741
        """The scale form of the adapter sink (IA3): `y = l ⊙ y` at `sites`."""
        lid, _ = materialize(l)
        sid, _ = materialize(sites)
        name = intern_name("lora")
        emit(Op.sink_call(name, [lid, sid]), ())
        record_sink("lora")

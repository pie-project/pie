"""
A program that reads attention scores states what it observes on its builder:
the container carries the optional observe section after every table, byte
for byte as the Rust `eta-ir` encoder writes it (tag 0x4f, a kind byte, and
for `rows` the u32 count); a program that states nothing carries no section,
its bytes as they always were.
"""

from __future__ import annotations

from inferlet.eta import intrinsics
from inferlet.eta.builder import Builder, DslChannel
from inferlet.eta.ir import OBSERVE_SECTION, Axis, Dtype, Observe, OutputDecl, Stage, dtype
from inferlet.eta.value import reduce_argmax

VOCAB = 8


def traced(observe: Observe | None) -> bytes:
    out = DslChannel.new([1], dtype.i32)
    b = Builder(VOCAB, 2)
    b.set_outputs([[OutputDecl("logits", Axis.ROW, VOCAB, Dtype.F32, 0, True)]])
    if observe is not None:
        b.set_observe(observe)

    def epilogue() -> None:
        out.put_tensor(reduce_argmax(intrinsics.output(0, "logits")))

    b.stage(Stage.EPILOGUE, epilogue)
    return b.build().encode()


def test_the_observe_section_round_trips() -> None:
    plain = traced(None)
    rows = traced(Observe.rows_(32))
    assert rows[: len(plain)] == plain
    assert list(rows[len(plain) :]) == [OBSERVE_SECTION, 0, 32, 0, 0, 0]
    fire = traced(Observe.fire_rows())
    assert fire[: len(plain)] == plain
    assert list(fire[len(plain) :]) == [OBSERVE_SECTION, 1]

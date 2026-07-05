//! Regression: dropping a reader auto-drain mid-stage must renumber the
//! stage's positional SSA ids. Before the fix, `assemble` retained-out the
//! synthesized drain `ChanTake` without remapping, so every op recorded after
//! an early terminal-output `put` referenced shifted (corrupt) value ids —
//! guests had to order terminal puts last as a workaround.

use ptir_dsl::builder::Builder;
use ptir_dsl::prelude::*;
use ptir_dsl::ptir::op::Op;
use ptir_dsl::{Channel, Traced};

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// A loop-carried accumulator with TWO terminal outputs, the first put EARLY
/// in the stage (drain dropped mid-body) and more value-producing ops after.
fn build_early_terminal_put() -> Traced {
    let acc: &'static Channel = leak(Channel::from([1u32]).named("acc"));
    let out: &'static Channel = leak(Channel::new([1], dtype::u32).named("out"));
    let out2: &'static Channel = leak(Channel::new([1], dtype::u32).named("out2"));

    let mut b = Builder::new();
    b.stage(Stage::Epilogue, move || {
        let v = acc.take().tensor();
        out.put(&v); // early terminal put: drain injected here, then dropped
        let w = add(&v, 1u32); // records AFTER the dropped drain
        out2.put(&w);
        acc.put(w);
    });
    b.build().expect("early terminal put must assemble cleanly")
}

/// Every operand must reference an already-defined id (the positional SSA
/// contract of `pie_ptir::op`). The pre-fix corruption shows up here as
/// forward references.
fn assert_ssa_well_formed(ops: &[Op]) {
    let mut defined = 0u32;
    for op in ops {
        for id in op.operands() {
            assert!(
                id < defined,
                "operand {id} of {op:?} references an undefined value (defined so far: {defined})"
            );
        }
        defined += op.result_count();
    }
}

#[test]
fn early_terminal_put_renumbers_ssa_ids() {
    let traced = build_early_terminal_put();
    let ops = &traced.container().stages[0].ops;

    let dense = |name: &str| -> u32 {
        traced.channel_names().iter().position(|n| n == name).unwrap() as u32
    };
    let (out, out2, acc) = (dense("out"), dense("out2"), dense("acc"));

    // The synthesized drains on the terminal outputs are gone...
    for op in ops.iter() {
        if let Op::ChanTake(c) | Op::ChanRead(c) = op {
            assert_ne!(*c, out, "drain on `out` must be dropped");
            assert_ne!(*c, out2, "drain on `out2` must be dropped");
        }
    }
    // ...and what survives is well-formed SSA.
    assert_ssa_well_formed(ops);

    // Pin the renumbering exactly: take=id0; the early put references it; the
    // post-drop ops' operands shift down past the dropped drain ids.
    assert!(matches!(ops[0], Op::ChanTake(c) if c == acc));
    let put_value = |chan: u32| -> u32 {
        ops.iter()
            .find_map(|op| match op {
                Op::ChanPut { chan: c, value } if *c == chan => Some(*value),
                _ => None,
            })
            .unwrap()
    };
    assert_eq!(put_value(out), 0, "early put references the take's id");
    let add_operands = ops
        .iter()
        .find_map(|op| match op {
            Op::Add(a, b) => Some((*a, *b)),
            _ => None,
        })
        .expect("the add survives");
    assert_eq!(add_operands.0, 0, "add's lhs is the take");
    assert_eq!(
        put_value(out2),
        put_value(acc),
        "both late puts reference the same (renumbered) add result"
    );
    assert!(
        put_value(out2) < ops.iter().map(Op::result_count).sum::<u32>(),
        "late puts reference a defined id"
    );
}

#[test]
fn early_terminal_put_builds_repeatably() {
    let a = build_early_terminal_put().identity_hash();
    let b = build_early_terminal_put().identity_hash();
    assert_eq!(a, b, "renumbering is deterministic");
}

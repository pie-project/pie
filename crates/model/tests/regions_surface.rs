//! The Guard/Peel surface unification (`.wiki/tart/dsl.md` migration
//! step 2), pinned.
//!
//! Here rather than in `model-compiler` for the reason `lowering.rs`
//! gives: `trace_finish` checks every launch against the kernel!
//! signature registry, and the registry is the BACKEND crates'. A trace
//! built where they are not linked refuses itself.
//!
//! The claim these pin is narrow and total: `regions` traces what the two
//! constructs it replaces traced, byte for byte in the op stream. The
//! goldens can only say that for families that already migrated; this
//! says it for the construct.

use model_compiler::dsl::*;
use model_compiler::trace::{DType, Dim, GuardPred, OpKind, Shape};

/// A statement the registry knows, so `trace_finish` accepts it, and
/// one with no operands so the two chains differ in nothing but the
/// construct around them.
fn stmt(t: &Trace) {
    let x = input(t, 8);
    let _ = cuda::residual_add(&x, &x, 8);
}

/// The unified surface must trace what the two it replaces traced.
/// Not a formality: the whole claim of this step is that the SURFACE
/// changed and nothing else did, and the goldens can only pin the
/// families that already migrated. This pins the construct itself.
#[test]
fn a_fire_armed_chain_traces_what_guarded_value_did() {
    let a = trace_named("regions.cuda.decode", |t| {
        let (g, _) = guarded_value(t, None, (Shape(vec![Dim::Tokens]), DType::BF16));
        g.arm(GuardPred::HasLora, || stmt(t)).otherwise(|| stmt(t));
    });
    let b = trace_named("regions.cuda.decode", |t| {
        regions(
            t,
            None,
            Some((Shape(vec![Dim::Tokens]), DType::BF16)),
            |c| c.arm(Region::Fire(GuardPred::HasLora), || stmt(t)),
            || stmt(t),
        );
    });
    assert_eq!(a.ops.len(), b.ops.len());
    assert!(matches!(a.ops[0].kind, OpKind::Guard { .. }));
    assert_eq!(
        format!("{:?}", a.ops[0].kind),
        format!("{:?}", b.ops[0].kind)
    );
}

#[test]
fn a_rows_armed_chain_traces_what_by_rows_did() {
    let a = trace_named("regions.cuda.decode", |t| {
        by_rows(t, None, None, |c| {
            c.arm(RowPred::Unmasked, || stmt(t));
            c.rest(|| stmt(t));
        });
    });
    let b = trace_named("regions.cuda.decode", |t| {
        regions(
            t,
            None,
            None,
            |c| c.arm(Region::Rows(RowPred::Unmasked), || stmt(t)),
            || stmt(t),
        );
    });
    assert_eq!(a.ops.len(), b.ops.len());
    assert!(matches!(a.ops[0].kind, OpKind::Peel { .. }));
    assert_eq!(
        format!("{:?}", a.ops[0].kind),
        format!("{:?}", b.ops[0].kind)
    );
}

/// A mix is REFUSED, not flattened into whichever op opened first.
#[test]
#[should_panic(expected = "cannot be both disciplines")]
fn a_mixed_chain_is_refused() {
    let _ = trace_named("regions.cuda.decode", |t| {
        regions(
            t,
            None,
            None,
            |c| {
                c.arm(Region::Rows(RowPred::Unmasked), || stmt(t));
                c.arm(Region::Fire(GuardPred::HasLora), || stmt(t));
            },
            || stmt(t),
        );
    });
}

/// `select` states a WINDOW: no rectangle, and a buffer offset INTO its
/// operand's. Both halves matter — a `Select` that lowered to a launch
/// would be a kernel the driver has to have, and one whose value got its
/// own allocation would be a copy the model does not make.
#[test]
fn a_select_launches_nothing_and_windows_its_operand() {
    use model_compiler::lower::{Buffers, Fire, Row, lower};

    let plan = trace_named("sel.cuda.decode", |t| {
        let x = input(t, 8);
        // A rank-3 value to window: [2, Tokens, 8].
        let streams = cuda::hc_expand(&x, 2, 8);
        let one = select(&streams, 1);
        let _ = cuda::residual_add(&one, &one, 8);
    });

    let rows: Vec<Row> = (0..4).map(|_| Row::default()).collect();
    let out = lower(&plan, &rows, Fire::default()).expect("must lower");
    assert!(
        out.residue.is_empty(),
        "a Select is not residue — it is a stated window: {:#?}",
        out.residue
    );
    // hc_expand and residual_add are the only two rectangles.
    assert_eq!(out.launches.len(), 2, "a Select must launch nothing");

    let sel = plan
        .ops
        .iter()
        .find(|o| matches!(o.kind, OpKind::Select { .. }))
        .expect("the plan states a Select");
    let b = Buffers::assign(&plan, &rows);
    let src = b.offset[sel.inputs[0] as usize];
    let win = b.offset[sel.outputs[0] as usize];
    assert_ne!(src, Buffers::NAMED);
    assert!(
        win > src,
        "index 1's window must sit past the source's start ({win} vs {src})"
    );
}

/// Accumulating into a `select` window keeps the result IN the window.
///
/// This is the pair `select` needed to be useful: a readable window is
/// only half of gemma3n's AltUp, whose per-layer embedding is added back
/// into K-1 corrected streams IN PLACE. Without `in_place` the add would
/// get its own allocation, the window would keep its pre-update value,
/// and the streams would silently never see it — which is the failure
/// mode this whole architecture exists to make impossible.
#[test]
fn an_in_place_add_lands_in_the_window_it_reads() {
    use model_compiler::lower::{Buffers, Fire, Row};

    let plan = trace_named("inplace.cuda.decode", |t| {
        let x = input(t, 8);
        let streams = cuda::hc_expand(&x, 2, 8);
        let one = select(&streams, 1);
        let ple = input(t, 8);
        let _ = cuda::residual_add(&one, &ple, 8);
    });

    let rows: Vec<Row> = (0..4).map(|_| Row::default()).collect();
    let b = Buffers::assign(&plan, &rows);

    let sel = plan
        .ops
        .iter()
        .find(|o| matches!(o.kind, OpKind::Select { .. }))
        .expect("the plan states a Select");
    let add = plan
        .ops
        .iter()
        .find(|o| {
            matches!(&o.kind, OpKind::Launch { kernel, .. }
                           if kernel == "norm::residual_add_bf16")
        })
        .expect("the plan states the in-place add");

    let window = b.offset[sel.outputs[0] as usize];
    let result = b.offset[add.outputs[0] as usize];
    assert_eq!(
        result, window,
        "the add must land IN the window it read ({result} vs {window})"
    );
    // And the window is still a window: past the streams' own start.
    let src = b.offset[sel.inputs[0] as usize];
    assert!(window > src, "index 1 must sit past the source start");
}

//! Shape and dtype mistakes made while writing a trace are reported, not
//! trapped.
//!
//! `eta-dsl` runs inside a `wasm32-wasip2` guest, where a panic is a trap: the
//! host sees an aborted instance with no stack, no file and no line. Every
//! mistake an author can make therefore has to arrive through
//! [`Builder::build`]'s `Err`, carrying the span of the call that made it.

use eta_dsl::builder::Builder;
use eta_dsl::error::TraceError;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, TraceErrors};

fn authoring(errors: &TraceErrors) -> Vec<&TraceError> {
    errors
        .0
        .iter()
        .filter(|e| matches!(e, TraceError::Authoring { .. }))
        .collect()
}

fn detail(error: &TraceError) -> &str {
    match error {
        TraceError::Authoring { detail, .. } => detail,
        other => panic!("expected an authoring error, got {other:?}"),
    }
}

#[test]
fn a_dtype_mismatch_is_reported_with_a_span() {
    let rows = Channel::seeded([2, 3], dtype::u32);
    let keys = Channel::seeded([4], dtype::f32);
    let output = Channel::new([2, 4], dtype::bool);
    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        output.put(row_membership(rows.take(), keys.take()));
    });

    let errors = builder.build().expect_err("mismatched dtypes must fail");
    let authoring = authoring(&errors);
    assert_eq!(authoring.len(), 1, "{errors}");
    assert!(
        detail(authoring[0]).contains("same dtype"),
        "{}",
        detail(authoring[0])
    );
    // The span is what makes the report actionable in a guest: it names this
    // file, not a frame inside the tracer.
    let TraceError::Authoring { span, .. } = authoring[0] else {
        unreachable!()
    };
    assert!(span.to_string().contains("authoring_errors.rs"), "{span}");
}

#[test]
fn every_mistake_in_one_trace_is_reported_together() {
    let rows = Channel::seeded([2, 3], dtype::u32);
    let keys = Channel::seeded([2, 2], dtype::u32);
    let matrix = Channel::seeded([2, 3], dtype::f32);
    let index = Channel::seeded([3], dtype::u32);
    let output = Channel::new([2, 4], dtype::bool);
    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        // Two independent mistakes: rank-2 keys, then one index too many.
        let _ = row_membership(rows.take(), keys.take());
        let _ = scalar_gather(matrix.take(), index.take());
        output.put(Channel::seeded([2, 4], dtype::bool).take());
    });

    let errors = builder.build().expect_err("two mistakes must fail");
    let authoring = authoring(&errors);
    assert_eq!(
        authoring.len(),
        2,
        "recording must not stop at the first mistake: {errors}"
    );
    assert!(detail(authoring[0]).contains("shape [K]"), "{errors}");
    assert!(
        detail(authoring[1]).contains("one index per row"),
        "{errors}"
    );
}


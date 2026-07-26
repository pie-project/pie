//! Shape and dtype mistakes made while writing a trace are reported, not
//! trapped.
//!
//! `pie-dsl` runs inside a `wasm32-wasip2` guest, where a panic is a trap: the
//! host sees an aborted instance with no stack, no file and no line. Every
//! mistake an author can make therefore has to arrive through
//! [`Builder::build`]'s `Err`, carrying the span of the call that made it.

use core::cell::RefCell;

use pie_dsl::builder::Builder;
use pie_dsl::error::TraceError;
use pie_dsl::prelude::*;
use pie_dsl::{Channel, TraceErrors};

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
        output.put(Channel::seeded([2, 4], dtype::bool).take().tensor());
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

#[test]
fn a_bad_channel_seed_survives_until_the_next_build() {
    // Channels are declared before any session exists, so this error has to be
    // held until there is a `build` to report it through.
    let seeded = Channel::from_shaped([2, 3], vec![1u32, 2, 3, 4]);
    let output = Channel::new([2, 3], dtype::u32);
    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        output.put(seeded.take().tensor());
    });

    let errors = builder.build().expect_err("a mis-shaped seed must fail");
    let authoring = authoring(&errors);
    assert_eq!(authoring.len(), 1, "{errors}");
    assert!(detail(authoring[0]).contains("from_shaped"), "{errors}");
}

#[test]
fn a_host_take_used_as_a_tensor_is_reported() {
    let host = Channel::new([4], dtype::u32);
    let output = Channel::new([4], dtype::u32);
    // Taken outside the trace, so this is a host take: its bytes cross the
    // driver boundary. Carrying it into a stage does not make it in-program.
    let taken = RefCell::new(Some(host.take()));
    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        let taken = taken.borrow_mut().take().expect("the stage runs once");
        output.put(taken.tensor());
    });

    let errors = builder
        .build()
        .expect_err("a host take has no in-program value");
    assert!(
        authoring(&errors)
            .iter()
            .any(|e| detail(e).contains("host channel")),
        "{errors}"
    );
}

//! The op table's two partitions, and the differential that used to be here.
//!
//! This file was "the Rust op table against the engine's C++ one": it read
//! `crates/driver/include/pie/driver/launch/op_table.hpp` as text and checked
//! that every `OP_TABLE` row had a `case` in the engine's `op_info`, that the
//! arities and result counts agreed, and that the engine declared no rows the
//! Rust table did not know about.
//!
//! That header was deleted with the C++ Metal engine, and the three tests that
//! read it went with it here. They had not been failing quietly -- they panic
//! on a missing file, which means `cargo test --workspace` has been red on
//! this branch since the deletion. Nothing noticed because CI runs on pushes
//! to `main` and on pull requests, and this branch has had neither.
//!
//! The reason those tests existed does not survive the deletion, so they are
//! not being ported to something else. `op_info`'s failure mode -- an
//! `OpCode` with no `case` falling through to `{c, 0xFF, 1, "?"}`, so a new
//! op made the engine quietly claim an unknown operand count rather than
//! complain -- was a property of a C++ function that no longer exists. The
//! engines that replaced it are Rust and match on `tensor_ir::op::OP_TABLE`
//! itself, so there is no second table to drift from.
//!
//! A generated `ptir_abi.h` briefly stood between the two, projecting the
//! table into C for whatever still `#include`d it. Nothing did: it went with
//! the C++, and `ptir_header.rs` -- which pinned it, and whose companion gate
//! against re-typed tags had already degraded to a no-op once both files it
//! named were deleted -- went with it.
//!
//! What is left is the half that was always Rust's own: the planner's
//! library/generated partition, which nothing outside this crate can check.

use tensor_ir::op::{OP_TABLE, tags};

// ===========================================================================
// The planner's library/generated partition
// ===========================================================================

/// Every op the fused *generated* kernel emits inline.
///
/// This list exists to be complete, not to be read. `tensor_compiler::plan` answers "is
/// this a library op?" with `library_op_for_tag`, whose `_ => None` arm means
/// "emit it inline" -- a fine default for an elementwise op and a wrong one
/// for a library op, because inline is not a kernel that exists for `matmul`.
/// A default cannot tell the two apart, so the partition is pinned instead:
/// add an op to `declare_ops!` and `every_op_is_classified` fails until it
/// appears here or in `library_op_for_tag`.
const GENERATED_TAGS: &[u8] = &[
    tags::EXP,
    tags::LOG,
    tags::NEG,
    tags::RECIP,
    tags::ABS,
    tags::SIGN,
    tags::CAST,
    tags::ADD,
    tags::SUB,
    tags::MUL,
    tags::DIV,
    tags::MAX_ELEM,
    tags::MIN_ELEM,
    tags::GT,
    tags::GE,
    tags::EQ,
    tags::NE,
    tags::LT,
    tags::LE,
    tags::AND,
    tags::OR,
    tags::NOT,
    tags::REM,
    tags::SELECT,
    tags::REDUCE_SUM,
    tags::REDUCE_MAX,
    tags::REDUCE_MIN,
    tags::REDUCE_ARGMAX,
    tags::BROADCAST,
    tags::RESHAPE,
    tags::TRANSPOSE,
    tags::PIVOT_THRESHOLD,
    tags::GATHER,
    tags::GATHER_ROW,
    tags::SCATTER_ADD,
    tags::SCATTER_SET,
    tags::IOTA,
    tags::MASK_APPLY_PACKED,
    tags::CAUSAL_MASK,
    tags::SLIDING_WINDOW_MASK,
    tags::SINK_WINDOW_MASK,
    tags::RNG,
    tags::RNG_KEYED,
    tags::CONST,
    tags::CHAN_TAKE,
    tags::CHAN_READ,
    tags::CHAN_PUT,
    tags::INTRINSIC_VAL,
];

#[test]
fn every_op_is_classified() {
    let mut unclassified: Vec<&str> = Vec::new();
    let mut both: Vec<&str> = Vec::new();
    for row in OP_TABLE {
        let library = tensor_compiler::plan::library_op_for_tag(row.tag).is_some();
        let generated = GENERATED_TAGS.contains(&row.tag);
        if library && generated {
            both.push(row.name);
        } else if !library && !generated {
            unclassified.push(row.name);
        }
    }
    assert!(
        unclassified.is_empty(),
        "these ops are in neither partition, so the planner will emit them \
         inline by default: {unclassified:?} -- add each to \
         `library_op_for_tag` or to GENERATED_TAGS above"
    );
    assert!(both.is_empty(), "ops claimed by both partitions: {both:?}");
}

#[test]
fn generated_tags_are_real_ops() {
    // A stale entry here would mask a missing classification: an op deleted
    // from `declare_ops!` but left in the list keeps the count looking right.
    for tag in GENERATED_TAGS {
        assert!(
            tensor_ir::op::spec(*tag).is_some(),
            "GENERATED_TAGS lists {tag:#04x}, which is not an OP_TABLE row"
        );
    }
    let mut sorted = GENERATED_TAGS.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        GENERATED_TAGS.len(),
        "duplicate tag in GENERATED_TAGS"
    );
}

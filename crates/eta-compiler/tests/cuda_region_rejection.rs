//! What the CUDA region gate refuses, and what it deliberately does not.
//!
//! Well-formedness of a plan — indices point at things that exist, an operand
//! is defined before it is read, a value's rank fits the wire — is not a
//! backend question, but it lived inside Metal's singleton validator. So Metal
//! checked it on every stage it compiled and CUDA checked node ordering and
//! node range and nothing else, which meant the two disagreed about which
//! plans were even legal. No golden could show it: it is not a difference in
//! emitted text.
//!
//! The gap was not theoretical. `pivot_threshold` carries its threshold as a
//! predicate payload rather than an operand, and the fused emitter turns it
//! into a `scratch + offsets[...]` read like any other value — so an
//! out-of-range payload reached the generated kernel as an out-of-bounds
//! device access, in a kernel that compiles and runs.
//!
//! Every mutation is classified here, because a subset chosen by hand is a
//! subset someone has to re-derive to trust. [`MUST_REJECT`] and
//! [`NO_OPINION`] have to partition the whole list, and a mutation only
//! belongs in [`NO_OPINION`] if it is *never* rejected — a partially caught
//! one fits neither and fails, rather than being quietly filed under the
//! forgiving heading.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;
#[path = "common/msl_mutations.rs"]
mod msl_mutations;

use eta_compiler::codegen::cuda::validate_generated_region;
use msl_corpus::{corpus_stages, extended_stages};
use msl_mutations::mutate;

/// Damage to a plan-level property, and what it breaks.
const MUST_REJECT: &[(&str, &str)] = &[
    ("unordered_region_nodes", "a use before def"),
    ("reverse_ops", "operands defined after they are read"),
    ("region_input_out_of_range", "an input past the value table"),
    (
        "region_output_out_of_range",
        "an output past the value table",
    ),
    ("region_sink_out_of_range", "a sink past the channel table"),
    (
        "bad_channel_slot",
        "a channel op on a slot the stage never bound",
    ),
    (
        "pivot_payload_out_of_range",
        "a predicate payload read as a scratch offset",
    ),
    ("library_scan_claim", "a boundary op inside generated code"),
    ("rank5_value_type", "a rank the wire cannot carry"),
    ("zero_static_dim", "an empty extent"),
    ("overflow_static_dims", "an element count past u32"),
    ("extra_value_type", "a value slot no op ever writes"),
    ("drop_last_value_type", "a result past the value table"),
    (
        "drop_last_value_type_and_refs",
        "a result past the value table",
    ),
];

/// How many corpus stages the mutation applied to, and how many of those the
/// gate then rejected.
///
/// Only regions the gate accepts *before* the mutation take part, so a region
/// declined for an unrelated reason — a library region, say — cannot make a
/// rejection look earned.
fn verdicts(mutation: &str) -> (usize, usize) {
    let (mut applied, mut rejected) = (0, 0);
    for stage in corpus_stages().into_iter().chain(extended_stages()) {
        let plan = stage.plan;
        let Some(region) = plan.fused.regions.first() else {
            continue;
        };
        if validate_generated_region(&plan, region).is_err() {
            continue;
        }
        let mut damaged = plan.clone();
        if !mutate(&mut damaged, mutation) {
            continue;
        }
        applied += 1;
        if validate_generated_region(&damaged, &damaged.fused.regions[0]).is_err() {
            rejected += 1;
        }
    }
    (applied, rejected)
}

#[test]
fn cuda_rejects_every_plan_level_damage() {
    let mut holes = Vec::new();
    for (mutation, damage) in MUST_REJECT {
        let (applied, rejected) = verdicts(mutation);
        if rejected != applied {
            holes.push(format!(
                "`{mutation}` ({damage}) slipped past on {} of {applied} regions",
                applied - rejected
            ));
        }
    }
    assert!(
        holes.is_empty(),
        "the CUDA gate accepted plans that are not well formed:\n  {}",
        holes.join("\n  ")
    );
}


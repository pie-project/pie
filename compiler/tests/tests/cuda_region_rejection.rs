//! The CUDA region gate refuses the malformed plans the Metal one refuses.
//!
//! Well-formedness — a region's indices point at things that exist, a value's
//! rank fits the wire, its static extents multiply within `u32` — is a property
//! of the plan, so the two backends have to agree about it. They did not:
//! `validate_singleton_plan` checked all of it on every stage Metal compiled,
//! while `validate_generated_region` checked node ordering and nothing else, so
//! a plan CUDA would have indexed straight off the end of was one Metal
//! rejected outright. Neither goldens nor emitter tests could see the gap,
//! because it is not a difference in emitted text.
//!
//! Each mutation below damages exactly one of those neutral properties. The
//! test requires the region to be accepted *before* the damage and rejected
//! after, so a region CUDA declines for some unrelated reason — a library
//! region, say — proves nothing and is not counted.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;
#[path = "common/msl_mutations.rs"]
mod msl_mutations;

use msl_corpus::{corpus_stages, extended_stages};
use msl_mutations::mutate;
use pie_codegen::cuda::validate_generated_region;

/// The mutations that damage a plan-level property, paired with what each one
/// breaks. Every other entry in `MUTATIONS` describes damage to the singleton
/// tier, which CUDA does not emit and so has no opinion about.
const NEUTRAL_MUTATIONS: &[(&str, &str)] = &[
    ("unordered_region_nodes", "use before def"),
    ("region_input_out_of_range", "input indexes past the values"),
    (
        "region_output_out_of_range",
        "output indexes past the values",
    ),
    ("region_sink_out_of_range", "sink indexes past the channels"),
    ("rank5_value_type", "rank the wire cannot carry"),
    ("zero_static_dim", "an empty extent"),
    ("overflow_static_dims", "element count past u32"),
];

/// How many corpus stages each mutation applied to, and how many of those the
/// CUDA gate then rejected.
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
fn every_neutral_mutation_reaches_a_cuda_region() {
    let dead: Vec<&str> = NEUTRAL_MUTATIONS
        .iter()
        .filter(|(mutation, _)| verdicts(mutation).0 == 0)
        .map(|(mutation, _)| *mutation)
        .collect();
    assert!(
        dead.is_empty(),
        "these mutations applied to no accepted CUDA region, so the rejection \
         test below proves nothing about them: {dead:?}"
    );
}

#[test]
fn cuda_rejects_every_plan_level_damage() {
    let mut holes = Vec::new();
    for (mutation, damage) in NEUTRAL_MUTATIONS {
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

use crate::msl_corpus::{corpus_stages, extended_stages};
use crate::msl_mutations::mutate;
use eta_compiler::codegen::cuda::validate_generated_region;

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

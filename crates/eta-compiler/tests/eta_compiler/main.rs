//! eta-compiler's integration tests, one binary: each module was a test file.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;
#[path = "common/msl_mutations.rs"]
mod msl_mutations;
#[path = "common/traces.rs"]
mod traces;

mod cuda_region_rejection;
mod emitted_source_shape;
mod emitter_version;
mod eta_examples;
mod eta_golden;
mod extended_golden;
mod gumbel_argmax_scans_the_logits_once;
mod launch_package;
mod op_table_drift;
mod singleton_plan_rejection;
mod the_epilogue_streams_through_registers;
mod the_score_rectangle_beside_many_channels;
mod wgsl_compiles;
mod wire_totality;

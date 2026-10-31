//! engine-cerebras's integration tests, one binary: each module was a test file.

mod common;

mod a_guest_pass_samples_on_the_fabric;
mod a_hand_written_phase_runs_through_the_bench;
mod a_program_compiles_and_runs_on_the_simulator;
mod an_epilogue_agrees_with_the_host_interpreter;
mod guest4_translation_is_pinned;
mod kernels_act;
mod kernels_attn;
mod kernels_attn_dense;
mod kernels_dynconv;
mod kernels_hc;
mod kernels_index;
mod kernels_layout;
mod kernels_linear;
mod kernels_moe;
mod kernels_norm;
mod kernels_ple;
mod kernels_rope;
mod kernels_ssm;

//! The attention suite again with at most three rows an attention phase:
//! every decode, prefill and mask test runs over row windows (views of
//! the row-shaped buffers, one phase a window), as a request whose rows
//! and pages exceed a PE does.

fn before() {
    kernels_cerebras::attn::cap_rows_per_phase(3);
}

#[path = "kernels_attn.rs"]
#[allow(dead_code)]
mod suite;

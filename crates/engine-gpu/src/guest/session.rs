//! What a guest session shares across APIs: the bindings a stage's
//! tables lie at, and how wide it dispatches. The tables themselves are
//! `eta_exec::tables`, one definition for every backend.

use eta_compiler::codegen::wgsl;
use eta_exec::ValueDesc;
pub use eta_exec::tables::{
    Cfg, DESC_WORDS, StageTables, Stepping, alias_reshapes, desc_words, program_table_sizes,
    stage_tables, table_sizes, words_le,
};

pub const BINDINGS: u32 = 6;

pub const CFG_BINDING: u32 = 5;

/// The binding of a step's uniform, read at a dynamic offset a dispatch:
/// its node, rung and selector (`eta_compiler::codegen::steps::StepUniform`).
pub const STEP_BINDING: u32 = wgsl::STEP_BINDING;

/// How far apart a stage's step uniforms lie, for a device whose uniform
/// offsets align to `align`.
#[must_use]
pub fn step_stride(align: u64) -> u64 {
    u64::from(wgsl::STEP_BYTES).max(align).next_power_of_two()
}

/// A stage's step uniforms, `stride` apart, in dispatch order.
#[must_use]
pub fn step_uniforms(steps: &[wgsl::Step], stride: u64) -> Vec<u8> {
    let stride = usize::try_from(stride).expect("a step stride fits the host");
    let mut bytes = vec![0u8; steps.len().max(1) * stride];
    for (at, step) in steps.iter().enumerate() {
        let uniform = step.uniform().bytes();
        bytes[at * stride..at * stride + uniform.len()].copy_from_slice(&uniform);
    }
    bytes
}

pub const WORKGROUP: u32 = eta_compiler::codegen::wgsl::WORKGROUP;

pub const GROUPS_PER_CORE: u32 = 4;

pub fn groups_for(descriptors: &[ValueDesc], cores: u32) -> u32 {
    let widest = descriptors.iter().map(|d| d.len).max().unwrap_or(1).max(1);
    let wanted = widest.div_ceil(WORKGROUP);
    let cap = cores.max(1).saturating_mul(GROUPS_PER_CORE);
    wanted.clamp(1, cap)
}

/// A stage's bind-time launch tables, in the order they lie in the
/// instance's `Instance` slots: descriptors, op params, value offsets and
/// the config, bound at these bindings.
pub const TABLE_BINDINGS: [u32; 4] = [1, 2, 3, CFG_BINDING];

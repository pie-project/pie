//! What a guest session shares across APIs: the bindings a stage's
//! tables lie at, and how wide it dispatches. The tables themselves are
//! `eta_exec::tables`, one definition for every backend.

use eta_exec::ValueDesc;
pub use eta_exec::tables::{
    Cfg, DESC_WORDS, StageTables, Stepping, alias_reshapes, desc_words, program_table_sizes,
    stage_tables, table_sizes, words_le,
};

pub const BINDINGS: u32 = 6;

pub const CFG_BINDING: u32 = 5;

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

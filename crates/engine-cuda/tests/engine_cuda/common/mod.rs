//! What the engine's integration tests share: a load's boot with the
//! defaults a test states nothing of, and the hashes and roundings their
//! fixtures are seeded by.

use std::path::Path;

use checkpoint::contract::ModelContract;
use engine_cuda::Boot;
use poem_compiler::Budget;
use poem_ir::Trace;

/// A test load of `trace` from `checkpoint`: `budget`'s fires, `slots`
/// slots of `context` tokens on 16-token pages, the tests' pool
/// ([`crate::POOL`]), the whole card allowed, GPU 0 alone, the default
/// knobs and no cache directory. A test states what it changes with
/// struct update: `Boot { knobs, ..boot(..) }`.
pub fn boot<'a>(
    trace: Trace,
    contract: &'a ModelContract,
    checkpoint: &'a Path,
    budget: Budget,
    context: u32,
    slots: u32,
) -> Boot<'a> {
    Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract,
        checkpoint,
        budget,
        patches: None,
        page_size: 16,
        context,
        slots,
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
        steps: engine::StepRules::runtime(),
    }
}

/// FNV-1a of a plane's name: the seed its fixture's values are drawn by.
pub fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// `value` rounded to the nearest bf16, ties to even.
pub fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

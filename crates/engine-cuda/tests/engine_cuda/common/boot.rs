//! A test load's boot: every field no test changes, at the tests' defaults.
//! Shared by the engine's test binaries (`engine_cuda`, `golden`).

use std::path::Path;

use checkpoint::contract::ModelContract;
use engine_cuda::Boot;
use poem_compiler::Budget;
use poem_ir::Trace;

/// The pool a test's load states: a served load takes what the card has
/// left, and the tests of one process load beside each other.
pub const POOL: u64 = 256 << 20;

/// A test load of `trace` from `checkpoint`: `budget`'s fires, `slots`
/// slots of `context` tokens on 16-token pages, the tests' pool
/// ([`POOL`]), the whole card allowed, GPU 0 alone, the default
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
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(POOL)),
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

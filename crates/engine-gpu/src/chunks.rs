//! The compiler's GPU recipe a load's one chunk pool is laid by: the chunk
//! profile of a Vulkan or WebGPU device, and the plan under it.

use poem_compiler::chunk::ChunkProfile;
use poem_compiler::{Budgets, DeviceProfile};

use crate::error::{Fault, Result};

/// The chunk profile of a Vulkan or WebGPU device whose one binding spans
/// `binding` bytes: chunks of a few MiB, segments no larger than a binding
/// (a pool is at most four of them, `kernels/prim/pool.slang`), kv pages
/// of the load's `page_size`, a window list a distinct window narrower than
/// the `context` (each its own `T_w` and page table, CONTRACT-BC 21), state
/// rows in f32.
#[must_use]
pub fn profile(page_size: u32, context: u32, binding: u64, target: Option<u64>) -> ChunkProfile {
    ChunkProfile {
        target_chunk_bytes: target.unwrap_or(crate::store::layout::TARGET_CHUNK),
        segment_bytes: binding,
        segment_grain: 256,
        max_binding_bytes: binding,
        row_align: 256,
        tile_bytes: 0,
        concurrency: 4,
        context,
        // An expert is one slot (the expert table names one place an
        // expert): the chunk is never smaller than an expert.
        cut_experts: false,
        tma_rows: false,
        state_f32: true,
        rs_mamba2: false,
        token_slot: Some(page_size),
        one_window: false,
        // A window as wide as the context keeps every token: a list row.
        full_from: Some(context),
    }
}

/// The compiler's GPU recipe for `trace` under `budgets`: the pool's one
/// layout source.
pub fn recipe(
    trace: &poem_ir::Trace,
    budgets: &Budgets,
    page_size: u32,
    context: u32,
    binding: u64,
    target: Option<u64>,
) -> Result<poem_compiler::chunk::ChunkRecipe> {
    let device = DeviceProfile {
        chunk: profile(page_size, context, binding, target),
        ..DeviceProfile::default()
    };
    poem_compiler::chunk::plan(trace, budgets, &device).map_err(|why| Fault::Unbound {
        what: format!("the GPU chunk recipe of this trace: {why}"),
    })
}

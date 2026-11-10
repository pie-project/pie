//! What a load tells the runtime about its pool: the chunk facts of the one
//! chunk pool ([`crate::store::Pools`]), the legacy pool facts the runtime
//! reads beside them, and the compiler's wgpu recipe the pool is laid by.

use engine::caps::PoolFacts;
use engine::chunk::ChunkFacts;
use poem_compiler::chunk::ChunkProfile;
use poem_compiler::{Budgets, DeviceProfile};

use crate::error::{Fault, Result};
use crate::serve::Shell;

impl Shell {
    /// The pool as the legacy facts state it: its kv pages, its records;
    /// `disk_kv_pages` is what [`Shell::seat_disk_kv`] seated.
    #[must_use]
    pub fn pool_facts(&self, disk_kv_pages: u32) -> PoolFacts {
        let paging = self.paging();
        let state_rows = self
            .trace()
            .caches
            .iter()
            .any(|row| matches!(row, poem_ir::CacheRow::State { .. }));
        PoolFacts {
            kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
            kv_page_size: paging.page_size,
            state_slots: if state_rows { paging.slots } else { 0 },
            state_slot_bytes: self.state_slot_bytes(),
            adapter_banks: u32::try_from(self.banks().len()).unwrap_or(u32::MAX),
            slab_kv_page_bytes: self.kv_page_bytes(),
            disk_kv_pages,
            // The sliding rows are paged in full here: the model's window
            // is a fact of the model, its pool is not.
            window_tokens: poem_exec::store::kv::declared_window(self.trace()).unwrap_or(0),
            ..PoolFacts::default()
        }
    }

    /// The pool as chunks: shared slot shapes over one chunk space.
    pub fn chunk_facts(&self) -> Result<ChunkFacts> {
        let facts = self.pool_chunk_facts();
        facts.validate().map_err(|why| Fault::Unbound {
            what: format!("the pool's chunk facts: {why}"),
        })?;
        Ok(facts)
    }
}

/// The chunk profile of a Vulkan or WebGPU device whose one binding spans
/// `binding` bytes: chunks of a few MiB, segments no larger than a binding
/// (a pool is at most four of them, `kernels/prim/pool.slang`), kv pages
/// of the load's `page_size` holding windowed rows in full (these kernels
/// page every kv row by one table), state rows in f32.
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
        cut_experts: true,
        state_f32: true,
        token_slot: Some(page_size),
        one_window: true,
        // Every window holds its rows whole: no window list.
        full_from: Some(1),
    }
}

/// The compiler's wgpu recipe for `trace` under `budgets`: the pool's one
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
        what: format!("the wgpu chunk recipe of this trace: {why}"),
    })
}

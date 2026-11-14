//! The compiler's GPU recipe a load's one chunk pool is laid by (the chunk
//! profile of a Vulkan or WebGPU device, and the plan under it), and what a
//! load tells the runtime about its pool: the chunk facts of the one chunk
//! pool, the legacy pool facts the runtime reads beside them.

use poem_compiler::chunk::ChunkProfile;
use poem_compiler::{Budgets, DeviceProfile};

use engine::caps::PoolFacts;
use engine::chunk::ChunkFacts;

use crate::error::{Fault, Result};
use crate::serve::{Serve, Shell};

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

/// The pool's kv pages as its facts state them, a u32: a count past it is
/// refused at the load, where a clamp would tell the runtime of fewer pages
/// than the pool holds.
fn stated_pages(pages: u64) -> Result<u32> {
    u32::try_from(pages).map_err(|_| Fault::Ceiling {
        what: "kv pages (the pool's facts state the count as a u32)",
        need: pages,
        have: u64::from(u32::MAX),
    })
}

impl<B: Serve> Shell<B> {
    /// The pool as the legacy facts state it: its kv pages, its records;
    /// `disk_kv_pages` is what [`Shell::seat_disk_kv`] seated.
    pub fn pool_facts(&self, disk_kv_pages: u32) -> Result<PoolFacts> {
        let paging = self.paging();
        let state_rows = self
            .trace()
            .caches
            .iter()
            .any(|row| matches!(row, poem_ir::CacheRow::State { .. }));
        Ok(PoolFacts {
            kv_pages: stated_pages(paging.pages())?,
            kv_page_size: paging.page_size,
            state_slots: if state_rows { paging.slots } else { 0 },
            state_slot_bytes: self.state_slot_bytes(),
            adapter_banks: u32::try_from(self.banks().len()).unwrap_or(u32::MAX),
            disk_kv_pages,
            // The model's window; its rows lie in window lists.
            window_tokens: poem_exec::store::kv::declared_window(self.trace()).unwrap_or(0),
            ..PoolFacts::default()
        })
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

#[cfg(test)]
mod tests {
    use super::stated_pages;
    use crate::error::Fault;

    /// The facts state every page the pool holds or the load is refused.
    #[test]
    fn a_page_count_past_u32_is_refused() {
        assert_eq!(stated_pages(0).ok(), Some(0));
        assert_eq!(stated_pages(u64::from(u32::MAX)).ok(), Some(u32::MAX));
        let refused = stated_pages(1 << 32).expect_err("past u32");
        assert!(
            matches!(
                refused,
                Fault::Ceiling { need, have, .. } if need == 1 << 32 && have == u64::from(u32::MAX)
            ),
            "{refused}"
        );
    }
}

//! The pools as chunks, as they stand today: each pool a pinned run of
//! one-slot chunks (`ChunkFacts::degenerate`). Where the planes are sparse
//! buffers mapped up to a watermark, the pool grows a page or slot at a
//! time, so a segment is one chunk and the committed ones are those the
//! watermark maps; each pinned run grows from its own bottom.
//!
//! Beside the facts the load logs, at debug, the chunk recipe the compiler
//! would pack this trace into on an on-device Metal GPU: informational, for
//! the pool switch.

use engine::caps::PoolFacts;
use engine::chunk::{ChunkFacts, Growth, SegmentFacts};
use poem_compiler::chunk::{ChunkProfile, human};
use poem_compiler::{Budgets, DeviceProfile};

use crate::error::Result;
use crate::serve::Shell;

impl Shell {
    /// Today's pools, as the load states them; `disk_kv_pages` is what
    /// [`Shell::seat_disk_kv`] seated.
    #[must_use]
    pub fn pool_facts(&self, disk_kv_pages: u32) -> PoolFacts {
        let paging = self.paging();
        let state_rows = self
            .trace()
            .caches
            .iter()
            .any(|row| matches!(row, poem_ir::CacheRow::State { .. }));
        let (elastic_page_bytes, elastic_budget_pages) = self.pool_elastic();
        PoolFacts {
            kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
            kv_page_size: paging.page_size,
            state_slots: if state_rows { paging.slots } else { 0 },
            state_slot_bytes: self.state_slot_bytes(),
            elastic_state_slot_bytes: self.elastic_state_slot_bytes(),
            adapter_banks: self
                .banks()
                .iter()
                .map(|&(_, adapters, _)| adapters)
                .min()
                .unwrap_or(0),
            elastic_page_bytes,
            elastic_budget_pages,
            disk_kv_pages,
            one_frame_when_short: false,
            seated_first: false,
            seat_to_end: false,
            seat_spread: false,
            buffer_pages: self.buffer_pages(),
            host_state_slots: self.parked_state_slots(),
            host_buffer_pages: self.parked_buffer_pages(),
            // The sliding rows are paged in full here: the model's window
            // is a fact of the model, its pool is not.
            window_tokens: poem_exec::store::kv::declared_window(self.trace()).unwrap_or(0),
            ..PoolFacts::default()
        }
    }

    /// The pool as chunks: the chunk pool's own facts, or (the per-plane
    /// pools of before) `pools` as degenerate chunks.
    pub fn chunk_facts(&self, pools: &PoolFacts) -> Result<ChunkFacts> {
        if let Some(facts) = self.pool_chunk_facts() {
            return Ok(facts);
        }
        let facts = ChunkFacts::degenerate(pools, &self.degenerate_bytes()?);
        if pools.elastic_page_bytes == 0 {
            return Ok(facts);
        }
        Ok(grown(facts, pools, self.committed_slots()))
    }

    /// Logs, at debug, beside the degenerate `facts`, the chunk recipe
    /// `poem_compiler::chunk::plan` packs this trace into on an on-device Metal GPU.
    pub fn log_chunk_recipe(&self, facts: &ChunkFacts) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let profile = DeviceProfile {
            chunk: ChunkProfile::ON_DEVICE_METAL,
            ..DeviceProfile::default()
        };
        let budgets = Budgets::of(self.budget().clone());
        match poem_compiler::chunk::plan(self.trace(), &budgets, &profile) {
            Ok(recipe) => tracing::debug!(
                backend = "metal",
                profile = "on-device-metal",
                chunk = %human(recipe.chunk_bytes),
                degenerate_chunks = facts.total_chunks,
                degenerate_chunk = %human(facts.chunk_bytes),
                shapes = %recipe.shapes(),
                chunks_per_working_set = recipe.chunks_per_ws(),
                waste_percent = recipe.waste.percent(),
                "the chunk recipe this trace would pack into"
            ),
            Err(error) => tracing::debug!(
                backend = "metal",
                %error,
                degenerate_chunks = facts.total_chunks,
                degenerate_chunk = %human(facts.chunk_bytes),
                "this trace packs into no chunk recipe"
            ),
        }
    }
}

/// The chunk the compiler's on-device Metal recipe packs `trace` into
/// under `budgets`; [`crate::store::layout::TARGET_CHUNK`] where it packs
/// into none.
#[must_use]
pub fn recipe_chunk(trace: &poem_ir::Trace, budgets: &Budgets) -> u64 {
    let profile = DeviceProfile {
        chunk: ChunkProfile::ON_DEVICE_METAL,
        ..DeviceProfile::default()
    };
    poem_compiler::chunk::plan(trace, budgets, &profile).map_or(
        crate::store::layout::TARGET_CHUNK,
        |recipe| recipe.chunk_bytes,
    )
}

/// `facts` of a pool that maps its pages and slots as they are committed:
/// one-chunk segments, the null chunk and `(kv pages, state slots, buffer
/// pages)` of them committed.
#[must_use]
pub fn grown(mut facts: ChunkFacts, pools: &PoolFacts, committed: (u32, u32, u32)) -> ChunkFacts {
    let (pages, slots, buffers) = committed;
    let committed = 1u32
        .saturating_add(pages.min(pools.kv_pages))
        .saturating_add(slots.min(pools.state_slots))
        .saturating_add(buffers.min(pools.buffer_pages));
    facts.segment_chunks = 1;
    facts.segments = SegmentFacts {
        committed: committed.min(facts.total_chunks),
        max: facts.total_chunks,
        growth: Growth::Bottom,
    };
    facts
}

#[cfg(test)]
mod tests {
    use engine::chunk::{DegenerateBytes, DegenerateCounts};

    use super::*;

    #[test]
    fn an_elastic_pool_grows_a_chunk_at_a_time_from_what_it_maps() {
        // A GDN hybrid on a sparse heap: nothing mapped at load, the rs
        // buffers whole beside the state slots.
        let pools = PoolFacts {
            kv_pages: 2_048,
            kv_page_size: 16,
            state_slots: 8,
            state_slot_bytes: 3 << 20,
            buffer_pages: 8,
            elastic_page_bytes: 16 << 10,
            elastic_budget_pages: 1 << 16,
            host_state_slots: 16,
            host_buffer_pages: 16,
            disk_kv_pages: 4_096,
            window_tokens: 4_096,
            ..PoolFacts::default()
        };
        let bytes = DegenerateBytes {
            kv_page: 64 << 10,
            window_page: 0,
            buffer_page: 1 << 20,
        };
        let facts = grown(ChunkFacts::degenerate(&pools, &bytes), &pools, (0, 0, 8));
        facts.validate().unwrap();
        assert_eq!(facts.degenerate_counts(), DegenerateCounts::of(&pools));
        assert_eq!(facts.total_chunks, 1 + 2_048 + 8 + 8);
        assert_eq!(facts.segment_chunks, 1);
        assert_eq!(facts.segments.max, facts.total_chunks);
        assert_eq!(facts.segments.committed, 1 + 8);
        assert_eq!(facts.segments.growth, Growth::Bottom);
        assert_eq!(facts.chunk_bytes, 3 << 20);

        let whole = grown(
            ChunkFacts::degenerate(&pools, &bytes),
            &pools,
            (u32::MAX, u32::MAX, u32::MAX),
        );
        whole.validate().unwrap();
        assert_eq!(whole.segments.committed, whole.segments.max);
    }
}

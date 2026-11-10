//! The pools as chunks, as they stand today: one buffer per plane, whole
//! from the load, each pool a pinned run of one-slot chunks
//! (`ChunkFacts::degenerate`). `commit_pool` may move the kv into smaller
//! slabs and back between frames, but never past what the load seated, so
//! the pool reads as fixed.
//!
//! Beside the facts the load logs, at debug, the chunk recipe the compiler
//! would pack this trace into under the server preset (no wgpu preset
//! yet): informational, for the pool switch.

use engine::caps::PoolFacts;
use engine::chunk::ChunkFacts;
use poem_compiler::chunk::{ChunkProfile, human};
use poem_compiler::{Budgets, DeviceProfile};

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
        PoolFacts {
            kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
            kv_page_size: paging.page_size,
            state_slots: if state_rows { paging.slots } else { 0 },
            buffer_pages: self.buffer_pages(),
            state_slot_bytes: self.state_slot_bytes(),
            adapter_banks: u32::try_from(self.banks().len()).unwrap_or(u32::MAX),

            elastic_page_bytes: 0,
            elastic_budget_pages: 0,
            slab_kv_page_bytes: self.kv_page_bytes(),
            elastic_state_slot_bytes: 0,
            disk_kv_pages,
            host_state_slots: self.parked_state_slots(),
            host_buffer_pages: self.parked_buffer_pages(),
            ..PoolFacts::default()
        }
    }

    /// `pools` as degenerate chunks.
    #[must_use]
    pub fn chunk_facts(&self, pools: &PoolFacts) -> ChunkFacts {
        ChunkFacts::degenerate(pools, &self.degenerate_bytes())
    }

    /// Logs, at debug, beside the degenerate `facts`, the chunk recipe
    /// `poem_compiler::chunk::plan` packs this trace into under the server preset.
    pub fn log_chunk_recipe(&self, facts: &ChunkFacts) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let profile = DeviceProfile {
            chunk: ChunkProfile::SERVER_CUDA,
            ..DeviceProfile::default()
        };
        let budgets = Budgets::of(self.budget().clone());
        match poem_compiler::chunk::plan(self.trace(), &budgets, &profile) {
            Ok(recipe) => tracing::debug!(
                backend = "wgpu",
                profile = "server-cuda",
                chunk = %human(recipe.chunk_bytes),
                degenerate_chunks = facts.total_chunks,
                degenerate_chunk = %human(facts.chunk_bytes),
                shapes = %recipe.shapes(),
                chunks_per_working_set = recipe.chunks_per_ws(),
                waste_percent = recipe.waste.percent(),
                "the chunk recipe this trace would pack into"
            ),
            Err(error) => tracing::debug!(
                backend = "wgpu",
                %error,
                degenerate_chunks = facts.total_chunks,
                degenerate_chunk = %human(facts.chunk_bytes),
                "this trace packs into no chunk recipe"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use engine::chunk::{
        DEGENERATE_BUFFER, DEGENERATE_KV, DEGENERATE_STATE, DegenerateBytes, DegenerateCounts,
        Growth,
    };

    use super::*;

    #[test]
    fn the_pools_read_back_as_fixed_degenerate_chunks() {
        // A GDN hybrid as this shell states it: kv pages, state slots and
        // buffer pages, parked slots and disk pages below, the window paged
        // in full.
        let pools = PoolFacts {
            kv_pages: 4_096,
            kv_page_size: 16,
            state_slots: 16,
            state_slot_bytes: 3 << 20,
            buffer_pages: 16,
            slab_kv_page_bytes: 96 << 10,
            host_state_slots: 32,
            host_buffer_pages: 32,
            disk_kv_pages: 8_192,
            window_tokens: 4_096,
            ..PoolFacts::default()
        };
        let bytes = DegenerateBytes {
            kv_page: 96 << 10,
            window_page: 0,
            buffer_page: 1 << 20,
        };
        let facts = ChunkFacts::degenerate(&pools, &bytes);
        facts.validate().unwrap();
        assert_eq!(facts.degenerate_counts(), DegenerateCounts::of(&pools));
        assert_eq!(facts.total_chunks, 1 + 4_096 + 16 + 16);
        assert_eq!(facts.host_chunks, 32 + 32);
        assert_eq!(facts.disk_chunks, 8_192);
        assert_eq!(facts.segments.growth, Growth::Fixed);
        assert_eq!(facts.chunk_bytes, 3 << 20);
        assert_eq!(facts.shape(DEGENERATE_KV).unwrap().slot_bytes, 96 << 10);
        assert_eq!(
            facts.shape(DEGENERATE_STATE).unwrap().sets_per_working_set,
            1
        );
        assert_eq!(facts.shape(DEGENERATE_BUFFER).unwrap().slot_bytes, 1 << 20);
    }
}

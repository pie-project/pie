//! The pools as chunks, as they stand today: one buffer per plane fixed at
//! load, each pool a pinned run of one-slot chunks (`ChunkFacts::degenerate`),
//! and past them the sink page and slot a fire's padding lanes write.
//!
//! Beside the facts the load logs, at debug, the chunk recipe the compiler
//! would pack this trace into on a TPU: informational, for the pool switch.

use engine::caps::PoolFacts;
use engine::chunk::{ChunkFacts, Reserved, ReservedRun};
use poem_compiler::chunk::{ChunkProfile, human};
use poem_compiler::{Budgets, DeviceProfile};

use crate::error::Result;
use crate::serve::Shell;

impl Shell {
    /// Today's pools, as the load states them.
    #[must_use]
    pub fn pool_facts(&self) -> PoolFacts {
        let paging = self.paging();
        PoolFacts {
            kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
            kv_page_size: paging.page_size,
            state_slots: if self.has_state() { paging.slots } else { 0 },
            buffer_pages: self.buffer_pages(),
            state_slot_bytes: self.state_slot_bytes(),
            adapter_banks: u32::try_from(self.banks().len()).unwrap_or(u32::MAX),
            elastic_page_bytes: 0,
            elastic_budget_pages: 0,
            window_pages: paging.window.map_or(0, |_| {
                u32::try_from(paging.window_pages()).unwrap_or(u32::MAX)
            }),
            window_tokens: paging.window.map_or(0, |window| window.tokens),
            ..PoolFacts::default()
        }
    }

    /// `pools` as degenerate chunks, the sink page of each kv pool and the
    /// sink slot of the state slabs reserved past them.
    pub fn chunk_facts(&self, pools: &PoolFacts) -> Result<ChunkFacts> {
        let bytes = crate::store::page_bytes(self.trace(), self.paging())?;
        let sinks = u32::from(pools.kv_pages > 0 && bytes.kv_page > 0)
            + u32::from(pools.window_pages > 0 && bytes.window_page > 0)
            + u32::from(pools.state_slots > 0);
        Ok(with_sinks(ChunkFacts::degenerate(pools, &bytes), sinks))
    }

    /// Logs, at debug, beside the degenerate `facts`, the chunk recipe
    /// `poem_compiler::chunk::plan` packs this trace into on a TPU.
    pub fn log_chunk_recipe(&self, facts: &ChunkFacts) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let profile = DeviceProfile {
            chunk: ChunkProfile::TPU,
            ..DeviceProfile::default()
        };
        let budgets = Budgets::of(self.budget().clone());
        match poem_compiler::chunk::plan(self.trace(), &budgets, &profile) {
            Ok(recipe) => tracing::debug!(
                backend = "xla",
                profile = "tpu",
                chunk = %human(recipe.chunk_bytes),
                degenerate_chunks = facts.total_chunks,
                degenerate_chunk = %human(facts.chunk_bytes),
                shapes = %recipe.shapes(),
                chunks_per_working_set = recipe.chunks_per_ws(),
                waste_percent = recipe.waste.percent(),
                "the chunk recipe this trace would pack into"
            ),
            Err(error) => tracing::debug!(
                backend = "xla",
                %error,
                degenerate_chunks = facts.total_chunks,
                degenerate_chunk = %human(facts.chunk_bytes),
                "this trace packs into no chunk recipe"
            ),
        }
    }
}

/// `facts` with `sinks` chunks reserved past its pool for padding lanes,
/// the one segment grown over them.
#[must_use]
pub fn with_sinks(mut facts: ChunkFacts, sinks: u32) -> ChunkFacts {
    if sinks == 0 {
        return facts;
    }
    facts.reserved.push(ReservedRun {
        what: Reserved::Sink,
        first: facts.total_chunks,
        chunks: sinks,
    });
    facts.total_chunks += sinks;
    facts.segment_chunks = facts.total_chunks;
    facts
}

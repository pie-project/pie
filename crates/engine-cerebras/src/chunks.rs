//! What a load tells the runtime about its pool: the chunk facts of the
//! host chunk pool ([`crate::store::pool`]), the legacy pool facts the
//! runtime still reads beside them, and, at debug, the chunk recipe the
//! compiler would pack this trace into on a server card (no Cerebras preset:
//! the host pool lays its slots out from the engine's own planes).

use engine::caps::PoolFacts;
use engine::chunk::ChunkFacts;
use poem_compiler::chunk::{ChunkProfile, human};
use poem_compiler::{Budgets, DeviceProfile};

use crate::error::{Fault, Result};
use crate::serve::Shell;

impl Shell {
    /// The pools as the budget sized them, as the load states them.
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

    /// The host chunk pool as the runtime counts it: one shared pool, its
    /// shapes unpinned.
    pub fn chunk_facts(&self) -> Result<ChunkFacts> {
        let facts = self.pool().facts();
        facts.validate().map_err(|why| Fault::Unbound {
            what: format!("the host pool's chunk facts: {why}"),
        })?;
        Ok(facts)
    }

    /// Logs, at debug, beside the pool's `facts`, the chunk recipe
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
                backend = "cerebras",
                profile = "server-cuda",
                chunk = %human(recipe.chunk_bytes),
                pool_chunks = facts.total_chunks,
                pool_chunk = %human(facts.chunk_bytes),
                shapes = %recipe.shapes(),
                chunks_per_working_set = recipe.chunks_per_ws(),
                waste_percent = recipe.waste.percent(),
                "the chunk recipe this trace would pack into"
            ),
            Err(error) => tracing::debug!(
                backend = "cerebras",
                %error,
                pool_chunks = facts.total_chunks,
                pool_chunk = %human(facts.chunk_bytes),
                "this trace packs into no chunk recipe"
            ),
        }
    }
}

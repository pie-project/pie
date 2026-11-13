//! What a load tells the runtime about its pool: the chunk facts of the one
//! chunk pool ([`crate::store::Pools`]), the legacy pool facts the runtime
//! reads until it counts chunks, and the compiler's on-device Metal recipe
//! the chunk size starts from.

use engine::caps::PoolFacts;
use engine::chunk::ChunkFacts;
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
        PoolFacts {
            kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
            kv_page_size: paging.page_size,
            state_slots: if state_rows { paging.slots } else { 0 },
            state_slot_bytes: self.state_slot_bytes(),
            adapter_banks: self
                .banks()
                .iter()
                .map(|&(_, adapters, _)| adapters)
                .min()
                .unwrap_or(0),
            disk_kv_pages,
            seated_first: false,
            seat_to_end: false,
            seat_spread: false,
            buffer_pages: self.buffer_pages(),
            // The sliding rows are paged in full here: the model's window
            // is a fact of the model, its pool is not.
            window_tokens: poem_exec::store::kv::declared_window(self.trace()).unwrap_or(0),
            ..PoolFacts::default()
        }
    }

    /// The pool as chunks: shared slot shapes over one chunk space.
    pub fn chunk_facts(&self) -> Result<ChunkFacts> {
        let facts = self.pool_chunk_facts();
        facts
            .validate()
            .map_err(|why| crate::error::Fault::Unbound {
                what: format!("the pool's chunk facts: {why}"),
            })?;
        Ok(facts)
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
                pool_chunks = facts.total_chunks,
                pool_chunk = %human(facts.chunk_bytes),
                shapes = %recipe.shapes(),
                chunks_per_working_set = recipe.chunks_per_ws(),
                waste_percent = recipe.waste.percent(),
                "the chunk recipe this trace would pack into"
            ),
            Err(error) => tracing::debug!(
                backend = "metal",
                %error,
                pool_chunks = facts.total_chunks,
                pool_chunk = %human(facts.chunk_bytes),
                "this trace packs into no chunk recipe"
            ),
        }
    }
}

/// The compiler's on-device Metal recipe for `trace` under `budgets`: the
/// pool's one layout source.
pub fn recipe(
    trace: &poem_ir::Trace,
    budgets: &Budgets,
) -> Result<poem_compiler::chunk::ChunkRecipe> {
    recipe_near(trace, budgets, None)
}

/// [`recipe`] aiming its chunk near `target` bytes where stated (a
/// diagnostic: a miniature's records cut into pieces as a full model's are).
pub fn recipe_near(
    trace: &poem_ir::Trace,
    budgets: &Budgets,
    target: Option<u64>,
) -> Result<poem_compiler::chunk::ChunkRecipe> {
    let mut chunk = ChunkProfile::ON_DEVICE_METAL;
    if let Some(target) = target {
        chunk.target_chunk_bytes = target;
    }
    let profile = DeviceProfile {
        chunk,
        ..DeviceProfile::default()
    };
    poem_compiler::chunk::plan(trace, budgets, &profile).map_err(|why| {
        crate::error::Fault::Unbound {
            what: format!("the on-device Metal chunk recipe of this trace: {why}"),
        }
    })
}

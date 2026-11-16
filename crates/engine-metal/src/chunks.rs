//! What a load tells the runtime about its pool: the chunk facts of the one
//! chunk pool ([`crate::store::Pools`]), the pool facts beside them, and the compiler's on-device Metal recipe
//! the chunk size starts from.

use engine::caps::PoolFacts;
use engine::chunk::ChunkFacts;
use poem_compiler::chunk::{ChunkProfile, human};
use poem_compiler::{Budgets, DeviceProfile};

use crate::error::Result;
use crate::serve::Shell;

impl Shell {
    /// What the load states beside its chunk facts.
    #[must_use]
    pub fn pool_facts(&self) -> PoolFacts {
        PoolFacts {
            adapter_banks: self
                .banks()
                .iter()
                .map(|&(_, adapters, _)| adapters)
                .min()
                .unwrap_or(0),
            seated_first: false,
            seat_to_end: false,
            seat_spread: false,
            // The sliding rows are paged in full here: the model's window
            // is a fact of the model, its pool is not.
            window_tokens: poem_exec::store::kv::declared_window(self.trace()).unwrap_or(0),
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

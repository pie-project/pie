//! The pool as the load states it: one buffer of chunks fixed at load
//! (`store::Pools::facts`), every shape's slots drawn from it, and today's
//! pool counts beside them for what still reads them.
//!
//! Beside the facts the load logs, at debug, the chunk recipe the compiler
//! packs this trace into on a TPU, for comparison with the layout this
//! engine lays (`store::layout`), whose runs every plane's view addresses
//! as whole blocks.

use engine::caps::PoolFacts;
use engine::chunk::ChunkFacts;
use poem_compiler::chunk::{ChunkProfile, human};
use poem_compiler::{Budgets, DeviceProfile};

use crate::serve::Shell;

impl Shell {
    /// What the load states beside its chunk facts.
    #[must_use]
    pub fn pool_facts(&self) -> PoolFacts {
        // (no window list: windowed layers are served as ring + buffer,
        // `Shell::serves_windowed_state`)
        PoolFacts {
            adapter_banks: u32::try_from(self.bank_seats().len()).unwrap_or(u32::MAX),
            ..PoolFacts::default()
        }
    }

    /// The pool as chunks: one segment fixed at load (`Growth::Fixed`; XLA
    /// buffers are program parameters, sized before anything traces), the
    /// null and sink chunks reserved, every shape shared.
    #[must_use]
    pub fn chunk_facts(&self) -> ChunkFacts {
        self.tiered_facts()
    }

    /// Logs, at debug, beside `facts`, the chunk recipe
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
                pool_chunks = facts.total_chunks,
                pool_chunk = %human(facts.chunk_bytes),
                shapes = %recipe.shapes(),
                chunks_per_working_set = recipe.chunks_per_ws(),
                waste_percent = recipe.waste.percent(),
                "the chunk recipe this trace would pack into"
            ),
            Err(error) => tracing::debug!(
                backend = "xla",
                %error,
                pool_chunks = facts.total_chunks,
                pool_chunk = %human(facts.chunk_bytes),
                "this trace packs into no chunk recipe"
            ),
        }
    }
}

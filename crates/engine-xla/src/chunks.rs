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
    /// The pool's counts as the pre-chunk facts state them: the kv list's
    /// slots as kv pages, a record's bytes as a state slot's.
    #[must_use]
    pub fn pool_facts(&self) -> PoolFacts {
        let pools = self.pools();
        let paging = self.paging();
        let facts = pools.facts();
        let slots = |id: Option<engine::chunk::ShapeId>| {
            id.and_then(|id| facts.shape(id))
                .map_or(0, |shape| shape.device_slots(&facts))
        };
        PoolFacts {
            kv_pages: u32::try_from(slots(pools.kv_shape())).unwrap_or(u32::MAX),
            kv_page_size: paging.page_size,
            state_slots: 0,
            buffer_pages: 0,
            state_slot_bytes: pools.state_slot_bytes(),
            adapter_banks: u32::try_from(self.banks().len()).unwrap_or(u32::MAX),
            window_tokens: paging.window.map_or(0, |window| window.tokens),
            seated_first: self.seats()[0],
            seat_to_end: self.seats()[1],
            seat_spread: self.seats()[2],
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

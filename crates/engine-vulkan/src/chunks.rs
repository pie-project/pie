//! What a load tells the runtime about its pool: the chunk facts of the one
//! chunk pool ([`crate::store::Pools`]), the legacy pool facts the runtime
//! reads beside them; the recipe the pool is laid by is [`engine_gpu::chunks`].

use engine::caps::PoolFacts;
use engine::chunk::ChunkFacts;

use crate::error::{Fault, Result};
use crate::serve::Shell;

pub use engine_gpu::chunks::{profile, recipe};

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
            disk_kv_pages,
            // The model's window; its rows lie in window lists.
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

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
        }
    }

    /// The pool as chunks: shared slot shapes over one chunk space.
    pub fn chunk_facts(&self) -> Result<ChunkFacts> {
        let mut facts = self.pool_chunk_facts();
        // The record's state bytes beside the rings the pool states.
        facts.record_bytes += self.state_slot_bytes();
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
    poem_compiler::chunk::plan_windowed(trace, budgets, &profile, &MetalRing).map_err(|why| {
        crate::error::Fault::Unbound {
            what: format!("the on-device Metal chunk recipe of this trace: {why}"),
        }
    })
}

/// Metal's ring hook: a windowed lane's folded rows lie in
/// its record as pages laid exactly as a window-buffer slot (`P` rows of
/// every windowed layer's K and V, layer-major), so the attention kernels
/// read ring pages and buffer pages through one composed page table, at the
/// window slot's stride. The ring holds `R = ceil(W / P) * P + P` rows (the
/// widest layer's window, a page of slack so the pages a window touches are
/// distinct), row `r` at `r mod R`: `R / P` pages, each a whole window slot,
/// tiled from a chunk's start (the page id grid), so its pieces start at
/// chunk boundaries and cut only between pages.
pub struct MetalRing;

impl MetalRing {
    /// The ring's rows for windows up to `window` over pages of `page` rows.
    #[must_use]
    pub fn rows(window: u32, page: u32) -> u32 {
        let page = page.max(1);
        window.div_ceil(page) * page + page
    }
}

impl poem_compiler::chunk::RingLayout for MetalRing {
    fn ring_region(
        &self,
        layers: &[poem_compiler::chunk::WindowedRow],
        buffer: &poem_compiler::chunk::WindowBufferSlot<'_>,
    ) -> poem_compiler::chunk::RingRegion {
        let page = buffer.tokens.max(1);
        let widest = layers.iter().map(|layer| layer.window).max().unwrap_or(0);
        let pages = u64::from(Self::rows(widest, page) / page);
        let bytes = pages * buffer.bytes;
        // Each layer's share: its K and V rows over every page; the slot's
        // padding past the layers' rows with the last.
        let mut layer_bytes: Vec<u64> = layers
            .iter()
            .map(|layer| pages * u64::from(page) * layer.per_token)
            .collect();
        let stated: u64 = layer_bytes.iter().sum();
        if let Some(last) = layer_bytes.last_mut() {
            *last += bytes.saturating_sub(stated);
        }
        poem_compiler::chunk::RingRegion {
            bytes,
            atom: buffer.bytes,
            placement: poem_compiler::chunk::RingPlacement::ChunkStart,
            layer_bytes,
        }
    }
}

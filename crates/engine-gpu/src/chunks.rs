//! The compiler's GPU recipe a load's one chunk pool is laid by (the chunk
//! profile of a Vulkan or WebGPU device, and the plan under it), and what a
//! load tells the runtime about its pool: the chunk facts of the one chunk
//! pool, and the pool facts beside them.

use poem_compiler::chunk::ChunkProfile;
use poem_compiler::{Budgets, DeviceProfile};

use engine::caps::PoolFacts;
use engine::chunk::ChunkFacts;

use crate::error::{Fault, Result};
use crate::serve::{Serve, Shell};

/// The chunk profile of a Vulkan or WebGPU device whose one binding spans
/// `binding` bytes: chunks of a few MiB, segments no larger than a binding
/// (a pool is at most four of them, `kernels/prim/pool.slang`), kv pages
/// of the load's `page_size`, windowed rows as ring + buffer (CONTRACT-BC
/// 19, 21: [`Rings`]), state rows in f32.
#[must_use]
pub fn profile(page_size: u32, context: u32, binding: u64, target: Option<u64>) -> ChunkProfile {
    ChunkProfile {
        target_chunk_bytes: target.unwrap_or(crate::store::layout::TARGET_CHUNK),
        segment_bytes: binding,
        segment_grain: 256,
        max_binding_bytes: binding,
        row_align: 256,
        tile_bytes: 0,
        concurrency: 4,
        context,
        // An expert is one slot (the expert table names one place an
        // expert): the chunk is never smaller than an expert.
        cut_experts: false,
        tma_rows: false,
        state_f32: true,
        rs_mamba2: false,
        // A gated-delta decode step reads its carried rows as a linear
        // attention over their keys and corrected values
        // (`kernels/ssm/gdn_carried.slang`).
        rs_corrected: true,
        rs_scalars: true,
        token_slot: Some(page_size),
        // A window as wide as the context keeps every token: a list row.
        full_from: Some(context),
    }
}

/// The compiler's GPU recipe for `trace` under `budgets`: the pool's one
/// layout source.
pub fn recipe(
    trace: &poem_ir::Trace,
    budgets: &Budgets,
    page_size: u32,
    context: u32,
    binding: u64,
    target: Option<u64>,
) -> Result<poem_compiler::chunk::ChunkRecipe> {
    let device = DeviceProfile {
        chunk: profile(page_size, context, binding, target),
        ..DeviceProfile::default()
    };
    poem_compiler::chunk::plan_windowed(trace, budgets, &device, &Rings).map_err(|why| {
        Fault::Unbound {
            what: format!("the GPU chunk recipe of this trace: {why}"),
        }
    })
}

/// What a ring's rows align to inside the record: a row of any kv head
/// width starts on a word, and a ring at the record's row alignment.
pub const RING_ALIGN: u64 = 256;

/// This backend's ring hook: the plain per-layer rings, each layer's W rows
/// of its keys then as many of its values at [`RING_ALIGN`]
/// (`kernels_slang::Ring`), row `r` at place `r % W`.
pub struct Rings;

impl poem_compiler::chunk::RingLayout for Rings {
    fn ring_region(
        &self,
        layers: &[poem_compiler::chunk::WindowedRow],
        _buffer: &poem_compiler::chunk::WindowBufferSlot<'_>,
    ) -> poem_compiler::chunk::RingRegion {
        poem_compiler::chunk::plain_rings(layers, RING_ALIGN).region
    }
}

impl<B: Serve> Shell<B> {
    /// What the load states beside its chunk facts: its adapter banks (its
    /// windowed layers are ring + buffer, stated in the chunk facts).
    pub fn pool_facts(&self) -> Result<PoolFacts> {
        Ok(PoolFacts {
            adapter_banks: u32::try_from(self.banks().len()).unwrap_or(u32::MAX),
            ..PoolFacts::default()
        })
    }

    /// The pool as chunks: shared slot shapes over one chunk space.
    pub fn chunk_facts(&self) -> Result<ChunkFacts> {
        let mut facts = self.pool_chunk_facts();
        facts.windowed = self.pools().windowed().to_vec();
        facts.record_bytes = self.state_slot_bytes() + facts.ring_bytes();
        facts.validate().map_err(|why| Fault::Unbound {
            what: format!("the pool's chunk facts: {why}"),
        })?;
        Ok(facts)
    }
}

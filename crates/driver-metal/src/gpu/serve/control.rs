//! KV/state copies and pool resizes — the control ops, over the storage the
//! forward path shares.
//!
//! The planning is next door in [`transfer`](super::transfer); this is what
//! applies a plan to the pool this shell holds.

use crate::error::{Error, Result};
use crate::gpu::serve::state::Shell;

impl Shell {
    /// Move KV pages, and the rows inside them, within this pool.
    ///
    /// **Page order is load-bearing**, and the plan says so: a chain like
    /// `{1→0, 2→1}` reads page 1 for the second pair *after* the first has
    /// overwritten it. Each pair is independent and the caller sequences; a
    /// true swap needs a scratch page or separate calls.
    ///
    /// Settled on return: the move runs on the host, so nothing is in flight
    /// and a completion the caller waits on would wait for nothing.
    ///
    /// # Errors
    ///
    /// A call before `load_model`, a refusal from the planner (a foreign
    /// memory domain, a page the pool does not have), a copy that leaves a
    /// layer's region, or a pool with two page strides.
    pub fn copy_kv(&mut self, desc: &driver_api::KvCopyPlan) -> Result<()> {
        let pool = self.need_pool("copy_kv")?;
        let caps = super::transfer::Capabilities {
            has_linear_attn: self.has_linear_attn,
            kv_total_pages: pool.pages(),
            rs_slots: 0,
        };
        // ONE stride for the whole pool, or no copy at all.
        //
        // A move plan states byte offsets and applies them to every layer, so
        // it needs the pool to be page-major at one stride. gemma-4's is not:
        // its full-attention layers pack their pages at 4 heads x 512 where
        // its sliding ones use 16 x 256. Planning at either and applying to
        // both lands a page apart rather than obviously wrong.
        //
        // Refused by name rather than approximated. A KV copy is prefix
        // sharing and forking, which is a feature a deployment can be without
        // -- a corrupted cache is not.
        let (Some(grid), Some(page_bytes)) = (pool.shape().grid(), pool.shape().page_bytes())
        else {
            return Err(Error::Unserved {
                what: "copy_kv",
                message: format!(
                    "one page stride is needed for the pool and this model has two \
                     -- its full-attention layers are {} kv heads x {} against {} x \
                     {} on the sliding ones. Prefix sharing is unavailable on this \
                     checkpoint",
                    pool.shape().heads_at(0).0,
                    pool.shape().heads_at(0).1,
                    pool.shape().kv_heads,
                    pool.shape().head_dim,
                ),
            });
        };
        let work = super::transfer::plan_kv_copy(desc, caps, grid).map_err(|why| {
            Error::Unserved {
                what: "copy_kv",
                message: format!("{why:?}"),
            }
        })?;

        // Whole-page moves first, as page pairs; then the row cells. Both run
        // over every layer's K and V, which the stride check above is what
        // makes true.
        let mut cells = Vec::new();
        for &(src, dst) in &work.pages {
            cells.push(crate::layout::CellCopy {
                src_off: u64::from(src) * page_bytes,
                dst_off: u64::from(dst) * page_bytes,
                bytes: page_bytes,
            });
        }
        if !cells.is_empty() {
            pool.apply(&crate::layout::CellMovePlan {
                copies: cells,
                pages_touched: work.pages_touched,
            })?;
        }
        if let Some(plan) = work.cells.as_ref() {
            pool.apply(plan)?;
        }
        Ok(())
    }

    /// Move recurrent state between slots.
    ///
    /// # Errors
    ///
    /// Always, today, and the refusal says why: no model this backend serves
    /// has any recurrent state to move.
    pub fn copy_state(&mut self, _desc: &driver_api::StateCopyPlan) -> Result<()> {
        Err(Error::Unserved {
            what: "copy_state",
            message: "recurrent state is unreachable on this backend. It belongs to the \
                      qwen3_5 family and its neighbours, which `model::text` refuses at \
                      load, so no model this backend serves has any state to copy. \
                      `serve::transfer::plan_state_copy` and `layout::LinearStateSlots` \
                      are planned and stored ahead of that family being served, not \
                      behind it."
                .to_string(),
        })
    }

    /// Commit or release KV pages so the pool holds `target_pages`.
    ///
    /// # What is read, and what is not
    ///
    /// `target_pages` and nothing else. The plan also carries `map_ranges`
    /// and `unmap_ranges`, which describe WHICH pages to attach memory to --
    /// a CUDA VMM pool is addressed in ranges the caller chooses. These pages
    /// are not: a Metal sparse buffer is mapped from its start, so the target
    /// count fully determines the mapping and a range list could only agree
    /// with it or contradict it. Honouring the count and ignoring the ranges
    /// is the honest reading; picking ranges out of it would invent a
    /// placement this backend cannot express.
    ///
    /// Settled on return: `Stepper::trim` waits for the GPU to pass the unmap
    /// before it returns, and a growth is complete once the memory is
    /// attached — there is nothing left in flight for a caller to wait on.
    ///
    /// # Errors
    ///
    /// No pool loaded; a target past what the pool reserved address space
    /// for; or an arena without the memory to grow back into.
    pub fn resize_pool(&mut self, desc: &driver_api::PoolResizePlan) -> Result<()> {
        // WHICH pool. The trim task asks about three of them on every tick --
        // KV, recurrent state, and workspace -- and only the first exists
        // here. Ignoring the id would resize the KV pool to the state pool's
        // target, which is a page count derived from a high-water mark of
        // zero: the pool would be trimmed to nothing on the first tick, and
        // every fire after it would read pages that are no longer mapped.
        //
        // The other two are answered rather than refused. They have no
        // storage on this backend, so "resize the thing that holds nothing"
        // is satisfied by doing nothing -- and refusing would make the trim
        // task log a failure every tick for a pool it is right to ask about.
        if desc.pool_id != driver_api::PIE_ELASTIC_POOL_KV {
            return Ok(());
        }
        if self.pool.is_none() {
            return Err(Error::Unserved {
                what: "resize_pool",
                message: "there is no KV pool to resize. `load_model` allocates it, so a \
                          resize before a load is asking about a pool that does not exist \
                          yet rather than one this backend cannot change."
                    .to_string(),
            });
        }
        let target = u32::try_from(desc.target_pages).map_err(|_| Error::Unserved {
            what: "resize_pool",
            message: format!(
                "{} pages is not a pool this device could hold",
                desc.target_pages
            ),
        })?;
        let pool = self.pool.as_mut().expect("just inspected");
        pool.resize(&mut self.stepper, target)?;
        Ok(())
    }
}

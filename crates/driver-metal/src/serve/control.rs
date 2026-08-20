//! KV/state copies and pool resizes — the control ops, over the storage the
//! forward path shares.
//!
//! The planning is next door in [`transfer`](super::transfer); this is what
//! applies a plan to the pool this shell holds.

use crate::error::{Error, Result};
use crate::serve::state::Shell;

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
            // THE POOL'S OWN COUNT, not the literal `0` that was here. It is
            // `plan_kv_copy` being called, which does not read this field --
            // but `Capabilities` is one statement about what this shell can
            // do, and a zero in it was a claim that there are no recurrent
            // seats. There are, for every hybrid this backend now serves.
            rs_slots: self.recurrent.as_ref().map_or(0, |r| r.shape().slots),
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
        let work =
            super::transfer::plan_kv_copy(desc, caps, grid).map_err(|why| Error::Unserved {
                what: "copy_kv",
                message: format!("{why:?}"),
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
    /// Forking a conversation whose layers are linear: `copy_kv` moves the
    /// attention prefix and this moves the compressed one. A branch that took
    /// the first without the second would attend over the right pages with a
    /// history that never saw the prompt.
    ///
    /// Settled on return, like [`Self::copy_kv`] and for the same reason: the
    /// move runs on the host.
    ///
    /// # Errors
    ///
    /// A call before `load_model`; a checkpoint with no linear-attention
    /// layers, which has no state to move; or a slot outside the seats the
    /// pool was allocated with.
    pub fn copy_state(&mut self, desc: &driver_api::StateCopyPlan) -> Result<()> {
        // THE PREMISE THIS USED TO REFUSE ON WAS TRUE ONCE AND IS NOT NOW.
        //
        // The refusal said "no model this backend serves has any recurrent
        // state to move ... whose rows this build has no Metal text for --
        // `load_model` asks each row before it stages, and refuses there".
        // That was accurate when it was written. It stopped being accurate
        // the day the qwen3.5 forward path landed: both Qwen3.6 checkpoints
        // now load, allocate a `pools::recurrent::Pool`, and generate --
        // 27B agrees with `mlx_lm.generate` for sixty greedy tokens. The
        // prose stayed, and a comment that was true when written is the most
        // dangerous artifact here: nothing goes red when the world moves out
        // from under one.
        //
        // So the capability is READ rather than asserted. `has_linear_attn`
        // is `deployment.recurrent.is_some()`, set at load, and `rs_slots` is
        // the pool's own count -- which is also what fixes the second half of
        // the old bug, `copy_kv` passing a hardcoded `rs_slots: 0` into a
        // planner that bounds-checks slots against it.
        let Some(recurrent) = self.recurrent.as_ref() else {
            return Err(Error::Unserved {
                what: "copy_state",
                message: "there is no recurrent-state pool. `load_model` allocates one only \
                          for a deployment that states a recurrent stack, so this is a \
                          checkpoint whose layers are all attention -- there is no \
                          compressed history for a fork to take, and `copy_kv` moves all \
                          of what this row remembers."
                    .to_string(),
            });
        };
        let caps = super::transfer::Capabilities {
            has_linear_attn: self.has_linear_attn,
            kv_total_pages: self.pool.as_ref().map_or(0, crate::pools::kv::Pool::pages),
            rs_slots: recurrent.shape().slots,
        };
        let pairs =
            super::transfer::plan_state_copy(desc, caps).map_err(|why| Error::Unserved {
                what: "copy_state",
                message: format!("{why:?}"),
            })?;
        // IN PLAN ORDER, and the same warning `copy_kv` carries applies: a
        // chain reads a seat after an earlier pair has written it. The plan
        // sequences; this applies.
        for (src, dst) in pairs {
            // SAFETY: the driver verbs are serialized against the fire path,
            // and every fire this shell launches is waited for inside
            // `Shell::launch` before it returns -- so nothing is reading
            // either seat here.
            unsafe { recurrent.copy_slot(src, dst)? };
        }
        Ok(())
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
        // The other two are answered rather than refused. Workspace has no
        // storage on this backend, so "resize the thing that holds nothing"
        // is satisfied by doing nothing.
        //
        // THE RECURRENT POOL IS A DIFFERENT ANSWER THAT LOOKS THE SAME, and
        // it used to be given for the wrong reason. The comment here said
        // both of the other pools "have no storage on this backend", which
        // stopped being true when the hybrids landed: `self.recurrent` is a
        // real allocation for every Qwen3.6 row. Doing nothing is still
        // correct, and now for the reason `pools::recurrent`'s own module doc
        // states -- it is not a pager. A seat is held from a request's first
        // token to its last, the count is advertised once at load through
        // `rs_cache_slots`, and the trim task's target is derived from a
        // high-water mark that knows nothing about who is still sitting.
        // Honouring it would pull seats out from under live requests, which
        // is worse than the silence it replaced.
        //
        // Refusing is not the alternative: it would make the trim task log a
        // failure every tick for a pool it is right to ask about.
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

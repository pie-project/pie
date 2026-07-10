//! RS (recurrent-state) working set — runtime core (Lane D, Phase 3).
//!
//! Backend domain object behind the WIT `rs-working-set` resource and
//! `inference.fold`. It holds a **folded recurrent-state object** (one arena
//! [`ArenaKind::RsSlab`]) plus a dense, ordered array of mutable **buffered
//! page slots**. A forward pass may write buffered RS state WITHOUT folding it
//! (W10); folding into the recurrent state is the separate, explicit `fold(n)`
//! operation (W9). Forks share the folded + buffered objects by arena refcount
//! and copy-on-write the first object that is mutated (W11). `fold(n)` is
//! validated against the model fold granularity BEFORE any driver dispatch.
//!
//! This module depends only on [`crate::arena`] — no WIT/bindgen — so it is
//! unit-testable against a real [`Arena`]. The WIT host binding
//! (`HostRsWorkingSet`) lives in `crate::api::rs_working_set`; `echo`'s forward
//! `execute()` and `delta`'s `inference.fold` call the methods here. The
//! `&mut Arena` each method takes is obtained by the caller from the per-(model,
//! driver) `arena::get(model_idx, driver_idx).lock()` registry handle (sync
//! lock, never held across an await). See the Source note `workingset-rs-design`.

use crate::arena::{Arena, ArenaError, ArenaKind, ArenaTxn, CowPlan, MovePlan, ObjectId};

/// Arena kind backing the folded recurrent-state object.
const FOLDED_KIND: ArenaKind = ArenaKind::RsSlab;
/// Arena kind backing one buffered RS page. v1 keeps buffered pages in the RS
/// pool (each a single-block slab) so all recurrent-state memory shares one
/// accounting pool; a dedicated buffer kind can split it later.
const BUFFER_KIND: ArenaKind = ArenaKind::RsSlab;
/// Blocks per buffered RS page (v1: single block).
const BUFFER_BLOCKS: u32 = 1;

/// Per-model RS geometry, sourced from the driver capabilities / `model.wit`
/// caps. Carried on the working set so the host methods and `fold` validation
/// have the model's memory shape without re-querying the model each call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsGeometry {
    /// Bytes of one folded recurrent-state object (`model.rs-state-size`).
    pub state_size: u64,
    /// Arena blocks the folded slab occupies (≥ 1). v1 = 1 (single slot).
    pub state_blocks: u32,
    /// Tokens per buffered RS page (`model.rs-buffer-page-size`).
    pub buffer_page_tokens: u32,
    /// Fold granularity in tokens (`model.rs-fold-granularity`; ≥ 1). 1 means
    /// any positive length is foldable (token-causal: Qwen3.5 GDN, Nemotron-H
    /// Mamba2). Stored normalized so 0 from the cap is treated as 1.
    pub fold_granularity: u32,
}

impl RsGeometry {
    fn normalized_granularity(&self) -> u32 {
        self.fold_granularity.max(1)
    }
}

/// A contiguous, half-open span `[start, start + len)` of buffered page slots,
/// returned by `alloc_buffer` (mirrors the WIT `page-range`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageRange {
    pub start: u32,
    pub len: u32,
}

/// Errors from RS working-set operations. All are recoverable — the core never
/// panics on bad caller input (the WIT host maps these to `result<_, error>`).
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum RsError {
    #[error("fold: tokens must be > 0")]
    FoldZero,
    #[error("fold: {tokens} tokens exceed buffered capacity {capacity}")]
    FoldExceedsBuffer { tokens: u32, capacity: u32 },
    #[error("fold: {tokens} tokens is not a positive multiple of fold granularity {granularity}")]
    FoldGranularity { tokens: u32, granularity: u32 },
    #[error("rs working set: index {index} out of range (size {size})")]
    IndexOutOfRange { index: u32, size: u32 },
    #[error("rs working set: duplicate index {index}")]
    DuplicateIndex { index: u32 },
    #[error("rs working set: permutation is not a bijection over 0..{size}")]
    BadPermutation { size: u32 },
    #[error(
        "rs working set: buffer token range [{start}, {start}+{len}) exceeds capacity {capacity}"
    )]
    BufferRangeOutOfRange { start: u32, len: u32, capacity: u32 },
    #[error("rs working set: buffered slot {index} read before it was written (materialized)")]
    UnmaterializedRead { index: u32 },
    #[error(
        "rs working set: already bound to driver {bound}; cannot rebind to driver {requested} (no cross-driver migration in v1)"
    )]
    DriverRebind {
        bound: crate::driver::DriverId,
        requested: crate::driver::DriverId,
    },
    #[error(transparent)]
    Arena(#[from] ArenaError),
}

/// Physical plan for a single `fold(n)` produced by [`RsWorkingSet::prepare_fold`].
/// `delta`'s fold orchestration lowers this onto the driver: write the folded
/// recurrent state of `folded_slot`, setting `RS_FLAG_RESET` when `reset` and
/// `RS_FLAG_FOLD` with `rs_fold_lens = fold_len`; issue the `cow_move` d2d
/// (`copy_slot_d2d`) first when present (shared-fork CoW).
#[derive(Debug)]
pub struct RsFoldPlan {
    /// Arena object that receives the advanced folded state (= driver slot via
    /// `arena.blocks(folded_slot)[0]`).
    pub folded_slot: ObjectId,
    /// Freshly-allocated folded slab ⇒ the driver must zero it first
    /// (`RS_FLAG_RESET`).
    pub reset: bool,
    /// Device copy to issue before folding when the folded slab was shared
    /// (first fold after a fork copies it; W11).
    pub cow_move: Option<MovePlan>,
    /// Tokens to fold (`rs_fold_lens` / cuda `commit_len`).
    pub fold_len: u32,
}

/// Physical plan for an in-forward recurrent-state WRITE, produced by
/// [`RsWorkingSet::prepare_write`]. Distinct from [`RsFoldPlan`]: the GDN /
/// linear-attention forward writes the updated recurrent state INTO the folded
/// slot directly (the driver's in-forward `commit_len` path), rather than
/// folding separately-buffered RS slabs. `delta` / the PTIR pipeline lowers it:
/// `rs_slot_ids[r] = arena.blocks(folded_slot)[0]`, `RS_FLAG_RESET` when `reset`
/// (fresh slab → the driver zeroes it before writing), and the `cow_move` d2d
/// (`copy_slot_d2d`) issued first when present (shared-fork CoW).
#[derive(Debug)]
pub struct RsWritePlan {
    /// Arena object that receives the written folded state (= driver slot via
    /// `arena.blocks(folded_slot)[0]`).
    pub folded_slot: ObjectId,
    /// Freshly-allocated folded slab ⇒ the driver must zero it first
    /// (`RS_FLAG_RESET`); `false` on a continuing fire (prior state CoW'd in).
    pub reset: bool,
    /// Device copy to issue before the write when the folded slab was shared
    /// (first write after a fork copies it; W11).
    pub cow_move: Option<MovePlan>,
}

/// RS working set: folded recurrent state + mutable buffered suffix (W8).
#[derive(Debug)]
pub struct RsWorkingSet {
    geom: RsGeometry,
    /// Folded recurrent-state object. `None` for a fresh set with no folded
    /// state yet; allocated (with reset) on the first `fold`.
    folded: Option<ObjectId>,
    /// Dense, ordered array of buffered page slots. `None` = reserved (logical,
    /// no physical backing yet); `Some(id)` = materialized arena slab. Slots are
    /// reserved by `alloc_buffer` (arena-free — no driver needed) and
    /// materialized lazily on the first write (`cow_write_buffer`), at which
    /// point the driver is bound. Token-agnostic: the inferlet owns the
    /// token↔slot bookkeeping (W4).
    buffer: Vec<Option<ObjectId>>,
    /// The model this working set is bound to — known eagerly from the WIT
    /// `constructor(model)` (used for caps + the model half of the
    /// `arena::get(model, driver)` key + forward-model validation).
    model_id: usize,
    /// The driver this working set's arena objects live in. `None` until first
    /// materialization (the scheduler assigns the driver at forward time);
    /// `echo`'s `execute()` sets it via [`bind_driver`](Self::bind_driver) on the
    /// first write. Arena-touching methods use `bound_driver.unwrap_or(0)`
    /// (v1 single-driver). A working set is pinned to ONE driver — cross-driver
    /// migration is out of v1 scope.
    bound_driver: Option<crate::driver::DriverId>,
}

impl RsWorkingSet {
    /// A fresh, empty RS working set bound to `model_id` (from the WIT
    /// `constructor(model)`). No folded state, no buffered slots, driver unbound.
    pub fn new(model_id: usize, geom: RsGeometry) -> Self {
        RsWorkingSet {
            geom,
            folded: None,
            buffer: Vec::new(),
            model_id,
            bound_driver: None,
        }
    }

    /// The model this working set is bound to (eager, from construction).
    /// `echo`'s `execute()` asserts `pass.model_id == ws.model()`.
    pub fn model(&self) -> usize {
        self.model_id
    }

    /// The driver this working set is bound to, if materialized yet. Arena-
    /// touching methods use `driver().unwrap_or(0)` for the `arena::get` key.
    pub fn driver(&self) -> Option<crate::driver::DriverId> {
        self.bound_driver
    }

    /// Pin this working set to `driver` on first materialization (the scheduler
    /// picks the driver at forward time; `echo`'s `execute()` calls this on the
    /// first write). Idempotent for the same driver; rebinding to a different
    /// driver is rejected (a working set's arena objects cannot migrate across
    /// drivers in v1).
    pub fn bind_driver(&mut self, driver: crate::driver::DriverId) -> Result<(), RsError> {
        match self.bound_driver {
            None => {
                self.bound_driver = Some(driver);
                Ok(())
            }
            Some(d) if d == driver => Ok(()),
            Some(d) => Err(RsError::DriverRebind {
                bound: d,
                requested: driver,
            }),
        }
    }

    // ── Accessors (WIT + echo's resolve contract) ───────────────────────────

    /// Bytes of one folded recurrent-state object (`rs-working-set.state-size`).
    pub fn state_size(&self) -> u64 {
        self.geom.state_size
    }

    /// Current number of buffered page slots (`rs-working-set.buffer-size`).
    pub fn buffer_size(&self) -> u32 {
        self.buffer.len() as u32
    }

    /// Tokens per buffered RS page (`rs-working-set.buffer-page-size`).
    pub fn buffer_page_size(&self) -> u32 {
        self.geom.buffer_page_tokens
    }

    /// Model fold granularity in tokens (≥ 1).
    pub fn fold_granularity(&self) -> u32 {
        self.geom.normalized_granularity()
    }

    /// The folded recurrent-state arena object, if one has been allocated.
    pub fn folded_object(&self) -> Option<ObjectId> {
        self.folded
    }

    // ── Buffer structural mutators (dense ordered array) ─────────────────────

    /// Append `n` fresh **reserved** buffered page slots; returns the contiguous
    /// range added. Arena-free (no physical allocation, no driver) — slots are
    /// materialized lazily on the first write (`cow_write_buffer`), per the
    /// "alloc is lazy, driver-on-first-write" rule.
    pub fn alloc_buffer(&mut self, n: u32) -> Result<PageRange, RsError> {
        let start = self.buffer.len() as u32;
        self.buffer.resize(self.buffer.len() + n as usize, None);
        Ok(PageRange { start, len: n })
    }

    /// Remove the buffered slots at `indices` and densely compact the array.
    /// `indices` are interpreted against the array at call time; out-of-range or
    /// duplicate indices return `error` (no slot is freed in that case).
    /// Reserved (`None`) slots are free of arena cost; materialized slots are
    /// decref'd.
    pub fn free_buffer(&mut self, arena: &mut Arena, indices: &[u32]) -> Result<(), RsError> {
        let size = self.buffer.len() as u32;
        let mut remove = vec![false; self.buffer.len()];
        for &i in indices {
            if i >= size {
                return Err(RsError::IndexOutOfRange { index: i, size });
            }
            if remove[i as usize] {
                return Err(RsError::DuplicateIndex { index: i });
            }
            remove[i as usize] = true;
        }
        let mut kept = Vec::with_capacity(self.buffer.len() - indices.len());
        for (idx, slot) in self.buffer.iter().enumerate() {
            if remove[idx] {
                if let Some(id) = slot {
                    arena.decref(*id)?;
                }
            } else {
                kept.push(*slot);
            }
        }
        self.buffer = kept;
        Ok(())
    }

    /// Reorder the buffered slots by the full bijection `perm` over `0..size`:
    /// new slot `i` takes old slot `perm[i]`.
    pub fn reorder_buffer(&mut self, perm: &[u32]) -> Result<(), RsError> {
        let size = self.buffer.len();
        if perm.len() != size {
            return Err(RsError::BadPermutation { size: size as u32 });
        }
        let mut seen = vec![false; size];
        for &p in perm {
            let p = p as usize;
            if p >= size || seen[p] {
                return Err(RsError::BadPermutation { size: size as u32 });
            }
            seen[p] = true;
        }
        let old = self.buffer.clone();
        for (i, &p) in perm.iter().enumerate() {
            self.buffer[i] = old[p as usize];
        }
        Ok(())
    }

    // ── echo's resolve contract (forward path) ───────────────────────────────

    /// Materialized buffered page object ids covering the token range
    /// `[start_token, start_token + len_tokens)`, for a `rs-context` read. No
    /// mutation; `echo` pins them and maps each to a driver slot via
    /// `arena.blocks(obj)[0]`. Reading a reserved (never-written) slot is an
    /// error — a forward only reads buffered RS it previously wrote. A zero-
    /// length read returns empty (mirrors KV `resolve_read(_, 0)`), so a
    /// pure-prefill hybrid forward (no buffered RS read) needs no special-case
    /// at the call site.
    pub fn resolve_buffer(
        &self,
        start_token: u32,
        len_tokens: u32,
    ) -> Result<Vec<ObjectId>, RsError> {
        if len_tokens == 0 {
            return Ok(Vec::new());
        }
        let (first, last) = self.page_span(start_token, len_tokens)?;
        let mut ids = Vec::with_capacity(last - first + 1);
        for idx in first..=last {
            match self.buffer[idx] {
                Some(obj) => ids.push(obj),
                None => return Err(RsError::UnmaterializedRead { index: idx as u32 }),
            }
        }
        Ok(ids)
    }

    /// Prepare the buffered pages covering `[start_token, start_token +
    /// len_tokens)` for a `rs-output` write that does NOT fold (W10). Reserved
    /// slots are **materialized** (a fresh arena slab is allocated and marked as
    /// a write target — no copy). Shared materialized slots are copied-on-write
    /// (refcount > 1) and repointed to the private copy. Uniquely-owned
    /// materialized slots are written in place. The combined `MovePlan`
    /// (concatenated source/dest blocks of the CoW copies) is returned for the
    /// caller's single d2d; `None` when nothing was copied. Returns the
    /// post-materialization object ids for every page in the range.
    pub fn cow_write_buffer(
        &mut self,
        start_token: u32,
        len_tokens: u32,
        txn: &mut ArenaTxn,
        arena: &mut Arena,
    ) -> Result<(Vec<ObjectId>, Option<MovePlan>), RsError> {
        let (first, last) = self.page_span(start_token, len_tokens)?;
        let mut ids = Vec::with_capacity(last - first + 1);
        let mut from_all = Vec::new();
        let mut to_all = Vec::new();
        for idx in first..=last {
            match self.buffer[idx] {
                None => {
                    // First write materializes the reserved slot — fresh slab.
                    let h = arena.txn_alloc(txn, BUFFER_KIND, BUFFER_BLOCKS)?;
                    arena.txn_mark_write(txn, h.object_id)?;
                    self.buffer[idx] = Some(h.object_id);
                    ids.push(h.object_id);
                }
                Some(obj) => match arena.txn_cow(txn, obj)? {
                    CowPlan::InPlace { handle } => ids.push(handle.object_id),
                    CowPlan::Copy { handle, from, to } => {
                        self.buffer[idx] = Some(handle.object_id);
                        ids.push(handle.object_id);
                        from_all.extend(from);
                        to_all.extend(to);
                    }
                },
            }
        }
        let move_plan = if from_all.is_empty() {
            None
        } else {
            Some(MovePlan {
                from: from_all,
                to: to_all,
            })
        };
        Ok((ids, move_plan))
    }

    // ── fork (lazy CoW) ──────────────────────────────────────────────────────

    /// Fork into a new RS working set sharing the folded state and every
    /// buffered slab by reference (arena refcount bump; no copy). The first
    /// `fold`/buffered-write on a shared object copies it (W11).
    pub fn fork(&self, arena: &mut Arena) -> Result<RsWorkingSet, RsError> {
        if let Some(id) = self.folded {
            arena.incref(id)?;
        }
        for slot in &self.buffer {
            if let Some(id) = slot {
                arena.incref(*id)?;
            }
        }
        Ok(RsWorkingSet {
            geom: self.geom,
            folded: self.folded,
            buffer: self.buffer.clone(),
            model_id: self.model_id,
            bound_driver: self.bound_driver,
        })
    }

    // ── fold (W9): explicit, granularity-checked, no rollback ────────────────

    /// Validate a `fold(n)` request BEFORE any driver dispatch: `n > 0`, a
    /// positive multiple of the fold granularity, and within the buffered token
    /// capacity (`buffer_size * buffer_page_size`). The exact valid buffered
    /// token count is inferlet-owned (W4); the runtime bounds `n` by capacity.
    pub fn validate_fold(&self, n: u32) -> Result<(), RsError> {
        if n == 0 {
            return Err(RsError::FoldZero);
        }
        let g = self.geom.normalized_granularity();
        if g > 1 && n % g != 0 {
            return Err(RsError::FoldGranularity {
                tokens: n,
                granularity: g,
            });
        }
        let capacity = (self.buffer.len() as u32).saturating_mul(self.geom.buffer_page_tokens);
        if n > capacity {
            return Err(RsError::FoldExceedsBuffer {
                tokens: n,
                capacity,
            });
        }
        Ok(())
    }

    /// Prepare a `fold(n)`: validate, begin a transaction, and stage the folded
    /// write target — a freshly-allocated (reset) slab on the first fold, or a
    /// CoW of the existing folded slab (copied when shared after a fork). The
    /// returned [`ArenaTxn`] lives across the async driver round-trip; finish it
    /// with [`commit_fold`](Self::commit_fold) or [`abort_fold`](Self::abort_fold).
    pub fn prepare_fold(
        &mut self,
        arena: &mut Arena,
        n: u32,
    ) -> Result<(RsFoldPlan, ArenaTxn), RsError> {
        self.validate_fold(n)?;
        let mut txn = arena.txn_begin();
        let (folded_slot, reset, cow_move) = match self.folded {
            None => {
                let h = arena.txn_alloc(&mut txn, FOLDED_KIND, self.geom.state_blocks)?;
                arena.txn_mark_write(&mut txn, h.object_id)?;
                (h.object_id, true, None)
            }
            Some(id) => match arena.txn_cow(&mut txn, id)? {
                CowPlan::InPlace { handle } => (handle.object_id, false, None),
                CowPlan::Copy { handle, from, to } => {
                    (handle.object_id, false, Some(MovePlan { from, to }))
                }
            },
        };
        if let Err(e) = arena.txn_pin(&mut txn, folded_slot) {
            arena.txn_abort(txn);
            return Err(RsError::Arena(e));
        }
        Ok((
            RsFoldPlan {
                folded_slot,
                reset,
                cow_move,
                fold_len: n,
            },
            txn,
        ))
    }

    /// Commit a prepared fold after the driver round-trip succeeded: publish the
    /// transaction (release pins, fold in CoW-original decrefs) and adopt the
    /// advanced folded slab. There is NO rollback across a fold (W9) — the
    /// pre-fold folded state is only retained by a `fork` taken before the fold.
    pub fn commit_fold(
        &mut self,
        arena: &mut Arena,
        txn: ArenaTxn,
        plan: &RsFoldPlan,
    ) -> Result<(), RsError> {
        arena.txn_commit(txn)?;
        self.folded = Some(plan.folded_slot);
        Ok(())
    }

    /// Abort a prepared fold (driver failure): discard the staged slab / CoW
    /// copy and leave the prior folded state visible and unchanged.
    pub fn abort_fold(&self, arena: &mut Arena, txn: ArenaTxn) {
        arena.txn_abort(txn);
    }

    /// Prepare an in-forward recurrent-state WRITE: begin a transaction and stage
    /// the folded write target — a freshly-allocated (reset) slab on the first
    /// fire, or a CoW of the existing folded slab (copied when shared after a
    /// fork) on a continuing fire. The returned [`ArenaTxn`] lives across the
    /// async driver round-trip; finish it with [`commit_write`](Self::commit_write)
    /// or [`abort_write`](Self::abort_write). Unlike [`prepare_fold`](Self::prepare_fold)
    /// this is the direct in-forward write path used by the GDN / linear-attention
    /// forward (the `commit_len` primitive) — no buffered slabs, no fold-length
    /// validation. Shares the alloc/CoW staging with `prepare_fold`; kept separate
    /// so the fold-from-buffer contract (`validate_fold` + `fold_len`) is not
    /// entangled with the plain write.
    pub fn prepare_write(&mut self, arena: &mut Arena) -> Result<(RsWritePlan, ArenaTxn), RsError> {
        let mut txn = arena.txn_begin();
        let (folded_slot, reset, cow_move) = match self.folded {
            None => {
                let h = arena.txn_alloc(&mut txn, FOLDED_KIND, self.geom.state_blocks)?;
                arena.txn_mark_write(&mut txn, h.object_id)?;
                (h.object_id, true, None)
            }
            Some(id) => match arena.txn_cow(&mut txn, id)? {
                CowPlan::InPlace { handle } => (handle.object_id, false, None),
                CowPlan::Copy { handle, from, to } => {
                    (handle.object_id, false, Some(MovePlan { from, to }))
                }
            },
        };
        if let Err(e) = arena.txn_pin(&mut txn, folded_slot) {
            arena.txn_abort(txn);
            return Err(RsError::Arena(e));
        }
        Ok((
            RsWritePlan {
                folded_slot,
                reset,
                cow_move,
            },
            txn,
        ))
    }

    /// Commit a prepared in-forward write after the driver round-trip succeeded:
    /// publish the transaction (release pins, decref CoW originals) and adopt the
    /// written folded slab so it persists as the recurrent state for the next
    /// fire. No rollback across a committed write (W9), like `commit_fold`.
    pub fn commit_write(
        &mut self,
        arena: &mut Arena,
        txn: ArenaTxn,
        plan: &RsWritePlan,
    ) -> Result<(), RsError> {
        arena.txn_commit(txn)?;
        self.folded = Some(plan.folded_slot);
        Ok(())
    }

    /// Abort a prepared in-forward write (driver failure): discard the staged
    /// slab / CoW copy and leave the prior folded state visible and unchanged.
    pub fn abort_write(&self, arena: &mut Arena, txn: ArenaTxn) {
        arena.txn_abort(txn);
    }

    /// Prepare an in-forward recurrent-state write on the CALLER's transaction —
    /// the shared-txn variant of [`prepare_write`](Self::prepare_write) for callers
    /// that thread ONE arena txn across the whole forward prepare (e.g.
    /// `execute_impl`'s RS block, so the KV + RS staging commit/abort atomically on
    /// the same txn). Stages the folded write target — a freshly-allocated (reset)
    /// slab on the first fire, or a CoW of the existing folded slab on a continuing
    /// fire — on `txn`, mirroring `cow_write_buffer`'s `&mut txn` signature. Takes
    /// `&self`: it is READ-ONLY on the working set (it stages only on the arena
    /// txn), so [`adopt_write`](Self::adopt_write) is provably the ONLY mutation of
    /// `self.folded` (echo's atomic-by-construction rule). Adopt the slab with
    /// `adopt_write` AFTER the caller commits the shared txn (success); an abort of
    /// the shared txn reverts the staged alloc/CoW automatically (the caller owns
    /// the txn, so this does NOT abort on pin failure — the error propagates and the
    /// caller aborts).
    pub fn prepare_write_in_txn(
        &self,
        txn: &mut ArenaTxn,
        arena: &mut Arena,
    ) -> Result<RsWritePlan, RsError> {
        let (folded_slot, reset, cow_move) = match self.folded {
            None => {
                let h = arena.txn_alloc(txn, FOLDED_KIND, self.geom.state_blocks)?;
                arena.txn_mark_write(txn, h.object_id)?;
                (h.object_id, true, None)
            }
            Some(id) => match arena.txn_cow(txn, id)? {
                CowPlan::InPlace { handle } => (handle.object_id, false, None),
                CowPlan::Copy { handle, from, to } => {
                    (handle.object_id, false, Some(MovePlan { from, to }))
                }
            },
        };
        arena.txn_pin(txn, folded_slot)?;
        Ok(RsWritePlan {
            folded_slot,
            reset,
            cow_move,
        })
    }

    /// Adopt the folded slab staged by [`prepare_write_in_txn`](Self::prepare_write_in_txn)
    /// AFTER the caller's shared txn has committed — the written recurrent state
    /// persists as the folded state for the next fire. The ONLY mutation site of
    /// `self.folded` for the shared-txn path (echo's atomic-by-construction rule):
    /// call it in the finalize SUCCESS branch, never on abort (the caller's
    /// `txn_abort` reverts the staged slab and the prior folded state stays visible).
    pub fn adopt_write(&mut self, plan: &RsWritePlan) {
        self.folded = Some(plan.folded_slot);
    }

    /// Advance the folded boundary by `n` buffered tokens after a *committed*
    /// fold forward (W9 piggyback). Called from echo's `finalize_forward_txn`
    /// success branch (arena lock held, after the forward txn commit), guarded
    /// on the pass carrying `fold-buffered`. Drops the head buffer slabs fully
    /// covered by the folded prefix `[0, n)` (the first `n / buffer_page_size`
    /// slabs) and releases their refcounts; a partial tail page (n not
    /// page-aligned) stays buffered — the inferlet owns the token↔slot
    /// bookkeeping (W4). No rollback across a fold (W9): the pre-fold folded
    /// state is not retained. `n` is the same value lowered to `rs_fold_lens`.
    pub fn advance_fold(&mut self, n: u32, arena: &mut Arena) -> Result<(), RsError> {
        let page = self.geom.buffer_page_tokens.max(1);
        let drop = ((n / page) as usize).min(self.buffer.len());
        for slot in self.buffer.drain(..drop) {
            if let Some(id) = slot {
                arena.decref(id)?;
            }
        }
        Ok(())
    }

    // ── teardown ─────────────────────────────────────────────────────────────

    /// Release this working set's references to the folded + buffered objects
    /// (drop one refcount each). Call when the resource is destroyed.
    pub fn release(self, arena: &mut Arena) -> Result<(), RsError> {
        if let Some(id) = self.folded {
            arena.decref(id)?;
        }
        for slot in self.buffer {
            if let Some(id) = slot {
                arena.decref(id)?;
            }
        }
        Ok(())
    }

    // ── internals ────────────────────────────────────────────────────────────

    /// Map a buffered token range to the inclusive page-index span that covers
    /// it, validating the range against the buffered capacity.
    fn page_span(&self, start_token: u32, len_tokens: u32) -> Result<(usize, usize), RsError> {
        let page = self.geom.buffer_page_tokens;
        let capacity = (self.buffer.len() as u32).saturating_mul(page);
        let end = start_token.checked_add(len_tokens);
        if page == 0 || len_tokens == 0 || end.map_or(true, |e| e > capacity) {
            return Err(RsError::BufferRangeOutOfRange {
                start: start_token,
                len: len_tokens,
                capacity,
            });
        }
        let first = (start_token / page) as usize;
        let last = ((end.unwrap() - 1) / page) as usize;
        Ok((first, last))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{Arena, ArenaConfig, Residency};

    fn geom() -> RsGeometry {
        RsGeometry {
            state_size: 4096,
            state_blocks: 1,
            buffer_page_tokens: 4,
            fold_granularity: 1,
        }
    }

    fn arena() -> Arena {
        Arena::new(ArenaConfig {
            device: 0,
            block_size: 16,
            kv_pages: 64,
            rs_blocks: 64,
            scratch_blocks: 8,
            cpu_blocks: 64,
        })
    }

    /// Materialize the buffered slots covering `[0, n_tokens)` (write path) and
    /// return their fresh object ids — for tests that need concrete buffer
    /// objects (reserved slots are arena-free `None` until first write).
    fn materialize(ws: &mut RsWorkingSet, a: &mut Arena, n_tokens: u32) -> Vec<ObjectId> {
        let mut txn = a.txn_begin();
        let (ids, mv) = ws.cow_write_buffer(0, n_tokens, &mut txn, a).unwrap();
        assert!(mv.is_none(), "fresh materialization copies nothing");
        a.txn_commit(txn).unwrap();
        ids
    }

    // ── Gate: wire-level RESET regression (echo, finalize-review check 3) ───
    // Red-first guard for the marshal stub (folded_object()=None shipped as
    // slot 0 + RS_FLAG_RESET never set): fresh fire => reset + REAL alloc;
    // continuing fire => no reset, same slab; aborted fire adopts nothing
    // and the retry is fresh again.

    #[test]
    fn wire_fresh_fire_demands_reset_and_real_alloc_then_continues() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        let mut txn = a.txn_begin();
        let plan = ws.prepare_write_in_txn(&mut txn, &mut a).unwrap();
        assert!(plan.reset, "fresh set must demand RS_FLAG_RESET");
        assert!(plan.cow_move.is_none());
        assert!(
            a.blocks(plan.folded_slot).is_ok(),
            "plan must carry a REAL allocation (never the None->0 stub fallback)"
        );
        a.txn_commit(txn).unwrap();
        ws.adopt_write(&plan);
        assert_eq!(ws.folded_object(), Some(plan.folded_slot));
        let mut txn2 = a.txn_begin();
        let plan2 = ws.prepare_write_in_txn(&mut txn2, &mut a).unwrap();
        assert!(!plan2.reset, "continuing fire must NOT reset");
        assert_eq!(plan2.folded_slot, plan.folded_slot);
        a.txn_commit(txn2).unwrap();
        ws.adopt_write(&plan2);
    }

    #[test]
    fn wire_abort_adopts_nothing_and_retry_is_fresh() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        let mut txn = a.txn_begin();
        let plan = ws.prepare_write_in_txn(&mut txn, &mut a).unwrap();
        assert!(plan.reset);
        a.txn_abort(txn); // forward failed: adopt_write never called
        assert_eq!(ws.folded_object(), None, "aborted write must not adopt");
        let mut txn2 = a.txn_begin();
        let plan2 = ws.prepare_write_in_txn(&mut txn2, &mut a).unwrap();
        assert!(plan2.reset, "retry after abort is a fresh fire again");
        a.txn_commit(txn2).unwrap();
        ws.adopt_write(&plan2);
        assert!(ws.folded_object().is_some());
    }

    // ── Gate: fresh RS state ─────────────────────────────────────────────────

    #[test]
    fn fresh_state_is_empty() {
        let ws = RsWorkingSet::new(0, geom());
        assert_eq!(ws.buffer_size(), 0);
        assert_eq!(ws.state_size(), 4096);
        assert_eq!(ws.buffer_page_size(), 4);
        assert_eq!(ws.fold_granularity(), 1);
        assert_eq!(ws.folded_object(), None);
    }

    // ── prepare_write_in_txn / adopt_write (shared-txn in-forward write) ──────
    // The shared-txn variant execute_impl's RS block uses to allocate the fresh
    // folded slot for the in-forward GDN write (write_state=true path), fixing
    // the `folded_object().unwrap_or(0)` stub that shipped rs_slot_ids=[0].
    // prepare_write_in_txn is &self (stages only on the txn); adopt_write is the
    // ONLY self.folded mutation (echo's atomic-by-construction rule).

    #[test]
    fn prepare_write_in_txn_fresh_allocates_resets_and_adopts() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        // Stage on the CALLER's txn (as execute_impl does, alongside the KV writes).
        let mut txn = a.txn_begin();
        let plan = ws.prepare_write_in_txn(&mut txn, &mut a).unwrap();
        assert!(
            plan.reset,
            "fresh folded slab ⇒ reset (driver zeroes before write)"
        );
        assert!(plan.cow_move.is_none(), "fresh alloc ⇒ no CoW d2d");
        // Resolves to a REAL driver block — not the `0` the old stub marshaled.
        let block = a.blocks(plan.folded_slot).unwrap()[0];
        let _ = block;
        assert_eq!(
            ws.folded_object(),
            None,
            "not adopted until the shared txn commits"
        );
        // Commit the shared txn, THEN adopt (finalize success branch).
        a.txn_commit(txn).unwrap();
        ws.adopt_write(&plan);
        assert_eq!(ws.folded_object(), Some(plan.folded_slot));
    }

    #[test]
    fn prepare_write_in_txn_continuing_cows_without_reset() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        // Fire 1 (fresh): commit + adopt so the folded state exists.
        let mut t0 = a.txn_begin();
        let p0 = ws.prepare_write_in_txn(&mut t0, &mut a).unwrap();
        a.txn_commit(t0).unwrap();
        ws.adopt_write(&p0);
        // Fire 2 (continuing): CoW the existing folded slab, NO reset.
        let mut t1 = a.txn_begin();
        let p1 = ws.prepare_write_in_txn(&mut t1, &mut a).unwrap();
        assert!(
            !p1.reset,
            "existing folded state is CoW-continued, not reset"
        );
        a.txn_commit(t1).unwrap();
        ws.adopt_write(&p1);
        assert!(
            ws.folded_object().is_some(),
            "recurrent state persists across fires"
        );
    }

    #[test]
    fn prepare_write_in_txn_abort_leaves_prior_state() {
        let mut a = arena();
        let ws = RsWorkingSet::new(0, geom());
        // Stage a fresh write, then ABORT the caller's txn WITHOUT adopting.
        let mut txn = a.txn_begin();
        let _plan = ws.prepare_write_in_txn(&mut txn, &mut a).unwrap();
        a.txn_abort(txn);
        assert_eq!(
            ws.folded_object(),
            None,
            "aborted fresh stage ⇒ nothing adopted (no drift; re-fresh next fire)"
        );
    }

    #[test]
    fn fresh_fold_allocates_folded_slab_with_reset() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        ws.alloc_buffer(2).unwrap(); // capacity 8 tokens

        let (plan, txn) = ws.prepare_fold(&mut a, 5).unwrap();
        assert!(plan.reset, "first fold must reset the fresh folded slab");
        assert!(plan.cow_move.is_none());
        assert_eq!(plan.fold_len, 5);
        ws.commit_fold(&mut a, txn, &plan).unwrap();
        assert_eq!(ws.folded_object(), Some(plan.folded_slot));
        assert_eq!(a.residency(plan.folded_slot).unwrap(), Residency::Gpu);
    }

    // ── Gate: buffer mutators ────────────────────────────────────────────────

    #[test]
    fn alloc_free_reorder_buffer() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());

        let r = ws.alloc_buffer(4).unwrap();
        assert_eq!(r, PageRange { start: 0, len: 4 });
        assert_eq!(ws.buffer_size(), 4);
        // materialize so the 4 slots carry distinct object ids (page_tokens 4)
        let mat = materialize(&mut ws, &mut a, 16);
        let ids: Vec<Option<ObjectId>> = mat.iter().map(|&i| Some(i)).collect();
        assert_eq!(ws.buffer, ids);

        // reorder: reverse
        ws.reorder_buffer(&[3, 2, 1, 0]).unwrap();
        assert_eq!(ws.buffer, vec![ids[3], ids[2], ids[1], ids[0]]);

        // free middle two (indices against current array), compacts
        ws.free_buffer(&mut a, &[1, 2]).unwrap();
        assert_eq!(ws.buffer, vec![ids[3], ids[0]]);
        assert_eq!(ws.buffer_size(), 2);
    }

    #[test]
    fn buffer_mutators_reject_bad_input() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        ws.alloc_buffer(3).unwrap();

        assert_eq!(
            ws.free_buffer(&mut a, &[3]),
            Err(RsError::IndexOutOfRange { index: 3, size: 3 })
        );
        assert_eq!(
            ws.free_buffer(&mut a, &[1, 1]),
            Err(RsError::DuplicateIndex { index: 1 })
        );
        assert_eq!(
            ws.reorder_buffer(&[0, 1]),
            Err(RsError::BadPermutation { size: 3 })
        );
        assert_eq!(
            ws.reorder_buffer(&[0, 1, 1]),
            Err(RsError::BadPermutation { size: 3 })
        );
        // a rejected free leaves the array intact
        assert_eq!(ws.buffer_size(), 3);
    }

    // ── Gate: fold success / error ───────────────────────────────────────────

    #[test]
    fn validate_fold_rules() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom()); // page_tokens 4, granularity 1
        ws.alloc_buffer(2).unwrap(); // capacity 8

        assert_eq!(ws.validate_fold(0), Err(RsError::FoldZero));
        assert_eq!(
            ws.validate_fold(9),
            Err(RsError::FoldExceedsBuffer {
                tokens: 9,
                capacity: 8
            })
        );
        assert_eq!(ws.validate_fold(8), Ok(())); // exact capacity ok
        assert_eq!(ws.validate_fold(1), Ok(()));
    }

    #[test]
    fn validate_fold_granularity_gt_one() {
        let mut a = arena();
        let mut g = geom();
        g.fold_granularity = 4;
        let mut ws = RsWorkingSet::new(0, g);
        ws.alloc_buffer(4).unwrap(); // capacity 16

        assert_eq!(ws.validate_fold(4), Ok(()));
        assert_eq!(ws.validate_fold(8), Ok(()));
        assert_eq!(
            ws.validate_fold(5),
            Err(RsError::FoldGranularity {
                tokens: 5,
                granularity: 4
            })
        );
        // granularity 0 from the cap normalizes to 1 (unconstrained length)
        let mut g0 = geom();
        g0.fold_granularity = 0;
        let mut ws0 = RsWorkingSet::new(0, g0);
        ws0.alloc_buffer(2).unwrap();
        assert_eq!(ws0.validate_fold(3), Ok(()));
    }

    #[test]
    fn fold_error_stages_nothing() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        ws.alloc_buffer(1).unwrap(); // capacity 4
        let live_before = a.live_objects();
        assert!(ws.prepare_fold(&mut a, 0).is_err());
        assert!(ws.prepare_fold(&mut a, 5).is_err());
        // no folded slab allocated, no leaked staging
        assert_eq!(ws.folded_object(), None);
        assert_eq!(a.live_objects(), live_before);
    }

    #[test]
    fn fold_abort_leaves_prior_state() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        ws.alloc_buffer(2).unwrap();
        let live_before = a.live_objects();

        let (plan, txn) = ws.prepare_fold(&mut a, 4).unwrap();
        ws.abort_fold(&mut a, txn);
        // staged folded slab reclaimed; working set unchanged
        assert_eq!(ws.folded_object(), None);
        assert_eq!(a.live_objects(), live_before);
        let _ = plan;
    }

    // ── Gate: lazy fork + first-mutation CoW + no-rollback ────────────────────

    #[test]
    fn lazy_fork_shares_without_copy() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        ws.alloc_buffer(2).unwrap();
        // materialize the buffer + give it a folded slab first
        let buf_ids = materialize(&mut ws, &mut a, 8);
        let (plan, txn) = ws.prepare_fold(&mut a, 4).unwrap();
        ws.commit_fold(&mut a, txn, &plan).unwrap();
        let folded = ws.folded_object().unwrap();
        let live_before = a.live_objects();

        let child = ws.fork(&mut a).unwrap();
        // shares the same objects, no new allocation
        assert_eq!(child.folded_object(), Some(folded));
        assert_eq!(child.buffer, ws.buffer);
        assert_eq!(a.live_objects(), live_before, "fork must not allocate");
        assert_eq!(a.refcount(folded).unwrap(), 2, "folded slab now shared");
        for id in buf_ids {
            assert_eq!(a.refcount(id).unwrap(), 2, "materialized buffer shared");
        }
    }

    #[test]
    fn first_fold_after_fork_copies_on_write() {
        let mut a = arena();
        let mut parent = RsWorkingSet::new(0, geom());
        parent.alloc_buffer(2).unwrap();
        let (p0, t0) = parent.prepare_fold(&mut a, 4).unwrap();
        parent.commit_fold(&mut a, t0, &p0).unwrap();
        let parent_folded = parent.folded_object().unwrap();

        let mut child = parent.fork(&mut a).unwrap();
        assert_eq!(a.refcount(parent_folded).unwrap(), 2);

        // child's first fold must CoW the shared folded slab
        let (cp, ct) = child.prepare_fold(&mut a, 4).unwrap();
        assert!(!cp.reset, "existing folded state is advanced, not reset");
        let mv = cp.cow_move.as_ref().expect("shared folded slab must CoW");
        assert_eq!(mv.from.len(), mv.to.len());
        assert_ne!(mv.from, mv.to, "copy targets fresh blocks");
        assert_ne!(cp.folded_slot, parent_folded, "child got a private copy");
        child.commit_fold(&mut a, ct, &cp).unwrap();

        // parent keeps its original folded slab (no-rollback snapshot via fork)
        assert_eq!(parent.folded_object(), Some(parent_folded));
        assert_eq!(child.folded_object(), Some(cp.folded_slot));
        assert_ne!(parent.folded_object(), child.folded_object());
        assert_eq!(
            a.refcount(parent_folded).unwrap(),
            1,
            "writer left the sharing group"
        );
    }

    #[test]
    fn read_only_fork_never_copies() {
        let mut a = arena();
        let mut parent = RsWorkingSet::new(0, geom());
        parent.alloc_buffer(2).unwrap();
        let (p0, t0) = parent.prepare_fold(&mut a, 4).unwrap();
        parent.commit_fold(&mut a, t0, &p0).unwrap();

        let child = parent.fork(&mut a).unwrap();
        let live = a.live_objects();
        // neither mutates: no copy, both still share
        assert_eq!(parent.folded_object(), child.folded_object());
        assert_eq!(a.live_objects(), live);
    }

    #[test]
    fn fork_before_fold_keeps_independent_snapshots() {
        // Forking BEFORE any fold preserves a truly fresh snapshot: the parent
        // can later fold independently of the child.
        let mut a = arena();
        let mut parent = RsWorkingSet::new(0, geom());
        parent.alloc_buffer(2).unwrap();

        let mut child = parent.fork(&mut a).unwrap();
        // both fold independently from empty → each allocates its own slab+reset
        let (cp, ct) = child.prepare_fold(&mut a, 4).unwrap();
        assert!(cp.reset);
        child.commit_fold(&mut a, ct, &cp).unwrap();
        let (pp, pt) = parent.prepare_fold(&mut a, 4).unwrap();
        assert!(pp.reset);
        parent.commit_fold(&mut a, pt, &pp).unwrap();
        assert_ne!(parent.folded_object(), child.folded_object());
    }

    // ── Gate: write buffered RS without folding (W10) ────────────────────────

    #[test]
    fn resolve_buffer_token_range_to_pages() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom()); // page_tokens 4
        ws.alloc_buffer(3).unwrap(); // 3 pages, capacity 12

        // reading a reserved (never-written) slot is an error
        assert!(matches!(
            ws.resolve_buffer(0, 4),
            Err(RsError::UnmaterializedRead { .. })
        ));

        // materialize all 3 pages, then read
        let mat = materialize(&mut ws, &mut a, 12);
        // tokens [0,4) -> page 0
        assert_eq!(ws.resolve_buffer(0, 4).unwrap(), vec![mat[0]]);
        // tokens [2,8) -> pages 0,1
        assert_eq!(ws.resolve_buffer(2, 6).unwrap(), vec![mat[0], mat[1]]);
        // tokens [0,12) -> all pages
        assert_eq!(ws.resolve_buffer(0, 12).unwrap(), mat);
        // out of range
        assert!(matches!(
            ws.resolve_buffer(0, 13),
            Err(RsError::BufferRangeOutOfRange { .. })
        ));
        // zero-length read returns empty (mirrors KV resolve_read(_, 0));
        // pure-prefill hybrid forwards read no buffered RS.
        assert_eq!(ws.resolve_buffer(0, 0).unwrap(), Vec::<ObjectId>::new());
        assert_eq!(ws.resolve_buffer(7, 0).unwrap(), Vec::<ObjectId>::new());
    }

    #[test]
    fn cow_write_buffer_materializes_then_copies_shared() {
        let mut a = arena();
        let mut parent = RsWorkingSet::new(0, geom());
        parent.alloc_buffer(3).unwrap();

        // first write materializes the reserved slot — fresh slab, no CoW move
        let mut txn = a.txn_begin();
        let (ids, mv) = parent.cow_write_buffer(0, 4, &mut txn, &mut a).unwrap();
        assert_eq!(ids.len(), 1);
        assert!(mv.is_none(), "materialization copies nothing");
        a.txn_commit(txn).unwrap();
        let page0 = parent.buffer[0];
        assert_eq!(page0, Some(ids[0]));
        // materialize the rest so the fork shares concrete slabs
        let _ = materialize(&mut parent, &mut a, 12);
        let orig = parent.buffer.clone();

        // after fork the materialized pages are shared: writing CoWs the touched
        let mut child = parent.fork(&mut a).unwrap();
        let mut txn = a.txn_begin();
        let (ids, mv) = child.cow_write_buffer(0, 8, &mut txn, &mut a).unwrap();
        // pages 0 and 1 touched → both copied, repointed
        assert_eq!(ids.len(), 2);
        assert_ne!(Some(ids[0]), orig[0]);
        assert_ne!(Some(ids[1]), orig[1]);
        let mv = mv.expect("shared pages must CoW");
        assert_eq!(mv.from.len(), mv.to.len());
        a.txn_commit(txn).unwrap();
        // child's untouched page 2 still shares parent's slab
        assert_eq!(child.buffer[2], orig[2]);
        assert_eq!(parent.buffer[0], orig[0], "parent slot pointer unchanged");
    }

    #[test]
    fn release_drops_all_refs() {
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        ws.alloc_buffer(3).unwrap();
        materialize(&mut ws, &mut a, 12); // 3 buffer slabs
        let (plan, txn) = ws.prepare_fold(&mut a, 4).unwrap();
        ws.commit_fold(&mut a, txn, &plan).unwrap();
        assert!(a.live_objects() >= 4, "3 buffer slabs + 1 folded");
        ws.release(&mut a).unwrap();
        assert_eq!(a.live_objects(), 0, "release reclaims folded + buffer");
    }

    // ── Device binding: model eager, driver lazy (manager ruling) ────────────

    #[test]
    fn driver_binding_pins_one_driver() {
        let mut ws = RsWorkingSet::new(3, geom());
        assert_eq!(ws.model(), 3);
        assert_eq!(ws.driver(), None);
        ws.bind_driver(0).unwrap();
        assert_eq!(ws.driver(), Some(0));
        // idempotent for the same driver
        ws.bind_driver(0).unwrap();
        // rebinding to a different driver is rejected (no cross-driver migration)
        assert!(matches!(
            ws.bind_driver(1),
            Err(RsError::DriverRebind { .. })
        ));
        assert_eq!(ws.driver(), Some(0));
    }

    #[test]
    fn fork_inherits_model_and_driver() {
        let mut a = arena();
        let mut parent = RsWorkingSet::new(2, geom());
        parent.bind_driver(0).unwrap();
        parent.alloc_buffer(1).unwrap();
        let child = parent.fork(&mut a).unwrap();
        assert_eq!(child.model(), 2);
        assert_eq!(child.driver(), Some(0));
    }

    /// Documents the fold-consumption + single-token-decode drain semantics for
    /// the GDN decode path (bravo's `generate-gdn`): `advance_fold(n)` consumes
    /// at PAGE granularity — a partial fold (`n < page`) folds into the state but
    /// drops NO buffer page (the token offset does not reset), while a
    /// page-multiple fold drops exactly `floor(n/page)` front pages (the offset
    /// re-bases by `page`). A token-causal (`fold_granularity = 1`) decode folds
    /// each token inline for state currency and drains a full page via the
    /// guest-owned `free_buffer` once its `page` tokens are folded — keeping the
    /// buffer bounded. (`geom()`: page = 4 tokens.)
    #[test]
    fn fold_consumes_at_page_granularity_decode_drain() {
        let page = geom().buffer_page_tokens; // 4
        let mut a = arena();
        let mut ws = RsWorkingSet::new(0, geom());
        ws.alloc_buffer(2).unwrap(); // 2 pages = 8 tokens
        materialize(&mut ws, &mut a, 2 * page);
        assert_eq!(ws.buffer_size(), 2);

        // Partial fold (3 < page): folds into the state, drops NO page.
        ws.advance_fold(page - 1, &mut a).unwrap();
        assert_eq!(ws.buffer_size(), 2, "partial fold consumes no buffer page");

        // Page-multiple fold: drops exactly floor(n/page) front pages.
        ws.advance_fold(page, &mut a).unwrap();
        assert_eq!(ws.buffer_size(), 1, "one full page consumed");

        // Guest-owned explicit drain (the decode path: fold(1) inline for
        // currency, then free the fully-folded front page).
        ws.free_buffer(&mut a, &[0]).unwrap();
        assert_eq!(ws.buffer_size(), 0);

        // A single-token decode over a page boundary stays buffer-bounded: write
        // at the front-relative offset, fold(1) inline, free the front page each
        // `page` tokens → the buffer never exceeds one page.
        let mut b = 0u32; // write offset relative to the current buffer front
        for step in 0..(2 * page + 1) {
            if (b + 1).div_ceil(page) > ws.buffer_size() {
                ws.alloc_buffer(1).unwrap();
            }
            let mut txn = a.txn_begin();
            ws.cow_write_buffer(b, 1, &mut txn, &mut a).unwrap();
            a.txn_commit(txn).unwrap();
            b += 1;
            if b == page {
                ws.free_buffer(&mut a, &[0]).unwrap(); // drop the fully-folded page
                b = 0; // offset re-bases to the new front
            }
            assert!(
                ws.buffer_size() <= 1,
                "decode buffer stays bounded (step {step})"
            );
        }
    }
}

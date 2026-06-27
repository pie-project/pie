//! KV working set — dense ordered page array with relative-index semantics.
//!
//! Lane C / Phase 2. Spec: [[workingset-brief]] §KV working set (W2/W3/W6/W7) and
//! the design note `workingset-kv-design`. This is the structural core; the WIT
//! `HostKvWorkingSet` shell lives in `runtime/src/api/kv_working_set.rs` (echo
//! owns the bindgen `with:` / `add_to_linker` mapping).
//!
//! ## Model
//! A [`KvWorkingSet`] is a dense ordered array of page **slots** plus a
//! `generation` counter. The only references that cross the inferlet API are
//! relative `u32` indices into this array (W2). Slots are **lazily materialised**:
//! `alloc(n)` reserves `n` empty slots, and the physical KV page is allocated from
//! the arena on first write (`cow_write_slot`). A slot is therefore
//! `Option<ObjectId>` — `None` = reserved/empty, `Some(id)` = materialised page
//! object in the unified arena.
//!
//! ## Ownership split (Seam 2, bravo)
//! The unified arena is **one instance per driver**, shared by KV and RS, and owns
//! all refcounting + copy-on-write + physical blocks. The KV layer owns only the
//! dense slot array + generation (per working set) and the [`KvCas`] content-hash
//! index (per model/driver). Every method that touches physical state takes a
//! `&mut Arena` (and, for sealing/release, a `&mut KvCas`) supplied by the caller,
//! so the core is agnostic to where the arena/cas live (echo's `Arc<Mutex<Arena>>`
//! registry).
//!
//! ## Forward contract (echo)
//! `resolve_read` / `resolve_write` (validate-only) / `page_size` / `size` /
//! `generation` / `cow_write_slot` / `seal`, matching the frozen signatures echo
//! drives from the forward-pass txn. CAS sealing reuses
//! `crate::page_hash::compute_page_hashes` (unchanged).

use std::collections::HashMap;
use std::fmt;

use crate::arena::{Arena, ArenaError, ArenaKind, ArenaTxn, CowPlan, MovePlan, ObjectId};

/// Content hash of a sealed full KV page. Identical to `pagestore::PageHash`;
/// produced by `compute_page_hashes` at seal time.
pub type PageHash = u64;

/// Contiguous run of freshly reserved slots, returned by [`KvWorkingSet::alloc`].
/// Mirrors the WIT `record page-range { start: u32, len: u32 }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageRange {
    pub start: u32,
    pub len: u32,
}

/// Errors are **returned**, never trapped (W2/W3). The WIT shell maps these to
/// the inferlet-facing `error` via `Display`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkingSetError {
    /// A relative index was `>= size`.
    IndexOutOfRange { index: u32, size: u32 },
    /// `free`/`reorder` received the same index twice.
    DuplicateIndex { index: u32 },
    /// `reorder` permutation was not a full bijection over `0..size`.
    BadPermutation { size: u32 },
    /// `slice`/`resolve_read` range exceeded the array.
    RangeOutOfBounds { start: u32, len: u32, size: u32 },
    /// A read targeted a slot that has not been written yet (still reserved).
    UnwrittenPage { index: u32 },
    /// A write/seal ran against a captured generation that no longer matches —
    /// a concurrent structural mutation (alloc/free/reorder/append) invalidated it.
    StaleGeneration { captured: u32, current: u32 },
    /// An underlying arena operation failed (e.g. out of KV blocks).
    Arena(String),
}

impl fmt::Display for WorkingSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkingSetError::IndexOutOfRange { index, size } => {
                write!(f, "index {index} out of range (size {size})")
            }
            WorkingSetError::DuplicateIndex { index } => write!(f, "duplicate index {index}"),
            WorkingSetError::BadPermutation { size } => {
                write!(f, "permutation is not a bijection over 0..{size}")
            }
            WorkingSetError::RangeOutOfBounds { start, len, size } => {
                write!(f, "range start={start} len={len} exceeds size {size}")
            }
            WorkingSetError::UnwrittenPage { index } => {
                write!(f, "slot {index} has no written page")
            }
            WorkingSetError::StaleGeneration { captured, current } => {
                write!(f, "stale generation: captured {captured}, current {current}")
            }
            WorkingSetError::Arena(e) => write!(f, "arena: {e}"),
        }
    }
}

impl std::error::Error for WorkingSetError {}

impl From<ArenaError> for WorkingSetError {
    fn from(e: ArenaError) -> Self {
        WorkingSetError::Arena(e.to_string())
    }
}

type Result<T> = std::result::Result<T, WorkingSetError>;

/// Per-(model, driver) KV content-addressed-store index (W6/W7). Maps a sealed
/// full-page hash to its canonical sharer object, with a reverse map so a
/// canonical entry is cleared when its last reference is released. Shared by all
/// KV working sets of a model — lives beside the unified [`Arena`] in echo's
/// registry; constructed directly in tests.
#[derive(Debug, Default)]
pub struct KvCas {
    /// Sealed full-page hash -> canonical sharer object.
    index: HashMap<PageHash, ObjectId>,
    /// Canonical object -> its hash (reverse, for cleanup on release).
    canonical: HashMap<ObjectId, PageHash>,
}

impl KvCas {
    pub fn new() -> Self {
        KvCas::default()
    }

    /// Number of distinct sealed pages currently registered (diagnostics/tests).
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// The sealed full-page hash of `id`, if `id` is a sealed canonical page
    /// (else `None`).
    ///
    /// Lets the forward txn chain a newly-written page's hash from the **context
    /// tip's** sealed hash (`compute_page_hashes(..., prev_hash = this, ...)`), so
    /// CAS sealing works for pages appended onto a non-empty committed context —
    /// not only prefill-from-empty. A partial / never-sealed object returns
    /// `None` (and partial pages never seal anyway, W7).
    pub fn hash_of(&self, id: ObjectId) -> Option<PageHash> {
        self.canonical.get(&id).copied()
    }

    /// Release one reference to `id`. If this drops its last reference, clear any
    /// canonical CAS entry first, then free the arena object. KV-layer wrapper
    /// around `Arena::decref` so CAS bookkeeping stays consistent.
    fn release(&mut self, arena: &mut Arena, id: ObjectId) {
        if let Ok(1) = arena.refcount(id) {
            if let Some(h) = self.canonical.remove(&id) {
                if self.index.get(&h) == Some(&id) {
                    self.index.remove(&h);
                }
            }
        }
        let _ = arena.decref(id);
    }

    /// Seal a freshly-written, uniquely-owned (`rc == 1`) full page. On a CAS hit
    /// against a different canonical object, share it (incref canonical, release
    /// the duplicate) and return the canonical id; otherwise register `writable`
    /// as canonical and return it. Only ever touches the writing set's mapping (W6).
    fn seal(&mut self, arena: &mut Arena, writable: ObjectId, hash: PageHash) -> Result<ObjectId> {
        if let Some(&canon) = self.index.get(&hash) {
            if canon == writable {
                return Ok(writable);
            }
            arena.incref(canon)?;
            self.release(arena, writable);
            Ok(canon)
        } else {
            self.index.insert(hash, writable);
            self.canonical.insert(writable, hash);
            Ok(writable)
        }
    }
}

/// A dense ordered array of KV page slots. Relative `u32` indices into `slots`
/// are the only references that cross the API (W2).
#[derive(Debug)]
pub struct KvWorkingSet {
    /// Dense ordered page slots. `None` = reserved/empty (not yet materialised),
    /// `Some(id)` = a page object in the unified arena.
    slots: Vec<Option<ObjectId>>,
    /// Monotonic counter bumped by every **structural** mutation (alloc/free/
    /// reorder/append). Captured by a forward write and rejected on mismatch.
    /// `u32` to match the WIT `generation()` accessor + `kv-output.generation`.
    generation: u32,
    /// Tokens per KV page, cached from the model/driver at construction.
    page_size: u32,
    /// The model this working set belongs to (eager, from the constructor's
    /// `model`; v1 single-model ⇒ 0). The model half of the `(model, driver)`
    /// arena key.
    model_idx: usize,
    /// The driver this working set's materialised pages live on. Bound **lazily**
    /// on the first forward write (echo's `execute()` calls
    /// [`bind_driver`](Self::bind_driver)); the scheduler assigns the driver at
    /// forward time. `None` while every slot is still reserved — a working set
    /// with no materialised pages needs no driver/arena (W2 lazy slots).
    bound_driver: Option<usize>,
    /// Abort-revert log for an in-flight forward write: `(slot_index, prev_value)`.
    /// Drained by `commit_writes` (publish) or `abort_writes` (revert).
    pending: Vec<(usize, Option<ObjectId>)>,
}

impl KvWorkingSet {
    /// Fresh, empty working set for `model_idx` whose KV page holds `page_size`
    /// tokens. The driver is unbound until the first forward write materialises a
    /// page (see [`bind_driver`](Self::bind_driver)).
    pub fn new(page_size: u32, model_idx: usize) -> Self {
        KvWorkingSet {
            slots: Vec::new(),
            generation: 0,
            page_size,
            model_idx,
            bound_driver: None,
            pending: Vec::new(),
        }
    }

    // =====================================================================
    // Accessors (forward contract)
    // =====================================================================

    /// Number of page slots in the array.
    pub fn size(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Current structural generation (for stale-mutation rejection).
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Tokens per KV page for this working set's model/driver.
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// The `(model_idx, driver_idx)` whose arena/`KvCas` back this working set's
    /// pages — the keys for `arena::registry::get` / `kv_cas::get`. The driver is
    /// the bound driver, or `0` (v1 single-driver default) while unbound.
    pub fn device(&self) -> (usize, usize) {
        (self.model_idx, self.bound_driver.unwrap_or(0))
    }

    /// The model this working set belongs to (eager). echo asserts
    /// `pass.model_id == ws.model_idx()` in `execute()`.
    pub fn model_idx(&self) -> usize {
        self.model_idx
    }

    /// The bound driver, or `None` while no page has been materialised yet.
    pub fn bound_driver(&self) -> Option<usize> {
        self.bound_driver
    }

    /// Bind the driver on the first forward write (echo's `execute()`). Idempotent
    /// for the same driver; v1 is single-driver so this is always `0`.
    pub fn bind_driver(&mut self, driver_idx: usize) {
        match self.bound_driver {
            None => self.bound_driver = Some(driver_idx),
            Some(d) => debug_assert_eq!(
                d, driver_idx,
                "kv working set rebound to a different driver (cross-driver migration is out of v1 scope)"
            ),
        }
    }

    // =====================================================================
    // Structural mutators (inferlet-facing; all return error, never trap)
    // =====================================================================

    /// Append `n` fresh reserved slots; returns the contiguous range. Slots are
    /// materialised lazily on first write, so this never touches the arena and
    /// cannot fail on capacity. `alloc(0)` is a structural no-op.
    pub fn alloc(&mut self, n: u32) -> Result<PageRange> {
        let start = self.slots.len() as u32;
        if n > 0 {
            self.slots
                .extend(std::iter::repeat(None).take(n as usize));
            self.generation += 1;
        }
        Ok(PageRange { start, len: n })
    }

    /// Remove the slots at `indices` and densely compact survivors left,
    /// preserving order (W3, all-at-once, call-time). Out-of-range or duplicate
    /// indices return `error` and leave the array untouched.
    pub fn free(&mut self, indices: &[u32], arena: &mut Arena, cas: &mut KvCas) -> Result<()> {
        let size = self.slots.len();
        let mut seen = vec![false; size];
        for &i in indices {
            let iu = i as usize;
            if iu >= size {
                return Err(WorkingSetError::IndexOutOfRange {
                    index: i,
                    size: size as u32,
                });
            }
            if seen[iu] {
                return Err(WorkingSetError::DuplicateIndex { index: i });
            }
            seen[iu] = true;
        }
        if indices.is_empty() {
            return Ok(());
        }

        let mut new_slots = Vec::with_capacity(size - indices.len());
        let mut removed = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            if seen[i] {
                if let Some(id) = *slot {
                    removed.push(id);
                }
            } else {
                new_slots.push(*slot);
            }
        }
        self.slots = new_slots;
        for id in removed {
            cas.release(arena, id);
        }
        self.generation += 1;
        Ok(())
    }

    /// Apply the full bijection `perm` over `0..size`: `new[i] = old[perm[i]]`. A
    /// wrong length, out-of-range, or duplicated entry returns `error` (no trap).
    pub fn reorder(&mut self, perm: &[u32]) -> Result<()> {
        let size = self.slots.len();
        if perm.len() != size {
            return Err(WorkingSetError::BadPermutation { size: size as u32 });
        }
        let mut seen = vec![false; size];
        for &p in perm {
            let pu = p as usize;
            if pu >= size || seen[pu] {
                return Err(WorkingSetError::BadPermutation { size: size as u32 });
            }
            seen[pu] = true;
        }
        let new_slots: Vec<Option<ObjectId>> =
            perm.iter().map(|&p| self.slots[p as usize]).collect();
        self.slots = new_slots;
        self.generation += 1;
        Ok(())
    }

    /// Return a new working set sharing slots `[start, start+len)` **by reference**
    /// (materialised pages incref'd; reserved slots copy as reserved). First
    /// divergent write CoWs. Read-only on self.
    pub fn slice(&self, start: u32, len: u32, arena: &mut Arena) -> Result<KvWorkingSet> {
        let size = self.slots.len();
        let end = start as usize + len as usize;
        if end > size {
            return Err(WorkingSetError::RangeOutOfBounds {
                start,
                len,
                size: size as u32,
            });
        }
        let mut slots = Vec::with_capacity(len as usize);
        for slot in &self.slots[start as usize..end] {
            if let Some(id) = *slot {
                arena.incref(id)?;
            }
            slots.push(*slot);
        }
        Ok(KvWorkingSet {
            slots,
            generation: 0,
            page_size: self.page_size,
            model_idx: self.model_idx,
            bound_driver: self.bound_driver,
            pending: Vec::new(),
        })
    }

    /// Append `other`'s slots onto self **by reference** (materialised pages
    /// incref'd). Shares page objects; first divergent write CoWs. `other` is
    /// unchanged.
    pub fn append(&mut self, other: &KvWorkingSet, arena: &mut Arena) -> Result<()> {
        self.append_shared(&other.slots, arena)
    }

    /// Snapshot of the slot→object mapping (for the WIT shell, which must read
    /// `other`'s slots before taking a `&mut` borrow of `self` from the same
    /// resource table — avoids a double-borrow).
    pub fn slot_objects(&self) -> Vec<Option<ObjectId>> {
        self.slots.clone()
    }

    /// Append the given shared slots onto self by reference (incref each
    /// materialised page). Used by `append` and the WIT shell.
    pub fn append_shared(&mut self, slots: &[Option<ObjectId>], arena: &mut Arena) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }
        for slot in slots {
            if let Some(id) = *slot {
                arena.incref(id)?;
            }
            self.slots.push(*slot);
        }
        self.generation += 1;
        Ok(())
    }

    /// Lazy-CoW fork: a new working set sharing **all** current slots by reference
    /// (materialised pages incref'd). First divergent write on either side CoWs.
    pub fn fork(&self, arena: &mut Arena) -> Result<KvWorkingSet> {
        for slot in &self.slots {
            if let Some(id) = *slot {
                arena.incref(id)?;
            }
        }
        Ok(KvWorkingSet {
            slots: self.slots.clone(),
            generation: 0,
            page_size: self.page_size,
            model_idx: self.model_idx,
            bound_driver: self.bound_driver,
            pending: Vec::new(),
        })
    }

    /// Release every page this working set references. Called by the WIT resource
    /// `drop` handler. Reverts any in-flight write first.
    pub fn destroy(&mut self, arena: &mut Arena, cas: &mut KvCas) {
        // Undo any half-prepared forward so we don't leak/misattribute copies.
        self.abort_writes();
        for slot in self.slots.drain(..) {
            if let Some(id) = slot {
                cas.release(arena, id);
            }
        }
    }

    // =====================================================================
    // Forward-pass contract (driven by echo's txn lifecycle)
    // =====================================================================

    /// Context read: the arena `ObjectId`s for slots `[start, start+len)`. echo
    /// `txn_pin`s these and reads `arena.blocks()` for physical page ids. Errors
    /// if the range is out of bounds or any slot in it is unwritten.
    pub fn resolve_read(&self, start: u32, len: u32) -> Result<Vec<ObjectId>> {
        let size = self.slots.len();
        let end = start as usize + len as usize;
        if end > size {
            return Err(WorkingSetError::RangeOutOfBounds {
                start,
                len,
                size: size as u32,
            });
        }
        let mut out = Vec::with_capacity(len as usize);
        for (offset, slot) in self.slots[start as usize..end].iter().enumerate() {
            match *slot {
                Some(id) => out.push(id),
                None => {
                    return Err(WorkingSetError::UnwrittenPage {
                        index: start + offset as u32,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Output validation **only** (no mutation): the captured generation must
    /// still match, and `indices` must be in range and unique. echo calls this
    /// before driving per-slot CoW.
    pub fn resolve_write(&self, indices: &[u32], captured_gen: u32) -> Result<()> {
        if captured_gen != self.generation {
            return Err(WorkingSetError::StaleGeneration {
                captured: captured_gen,
                current: self.generation,
            });
        }
        let size = self.slots.len();
        let mut seen = vec![false; size];
        for &i in indices {
            let iu = i as usize;
            if iu >= size {
                return Err(WorkingSetError::IndexOutOfRange {
                    index: i,
                    size: size as u32,
                });
            }
            if seen[iu] {
                return Err(WorkingSetError::DuplicateIndex { index: i });
            }
            seen[iu] = true;
        }
        Ok(())
    }

    /// Prepare a single output slot for writing, inside echo's forward txn:
    /// - reserved slot (`None`) → allocate a fresh KV page (`txn_alloc`); no copy.
    /// - shared materialised slot (`rc > 1`) → `txn_cow` a private copy and repoint;
    ///   the returned [`MovePlan`] is the d2d the caller must issue.
    /// - uniquely-owned materialised slot (`rc == 1`) → write in place; no copy.
    ///
    /// The slot is repointed to the post-write object and an abort-revert is
    /// recorded (use [`commit_writes`]/[`abort_writes`]). Returns the post-write
    /// `ObjectId` and an optional copy plan.
    pub fn cow_write_slot(
        &mut self,
        idx: u32,
        txn: &mut ArenaTxn,
        arena: &mut Arena,
    ) -> Result<(ObjectId, Option<MovePlan>)> {
        let size = self.slots.len() as u32;
        if idx >= size {
            return Err(WorkingSetError::IndexOutOfRange { index: idx, size });
        }
        let i = idx as usize;
        match self.slots[i] {
            None => {
                // Lazily materialise a fresh page for this reserved slot.
                let handle = arena.txn_alloc(txn, ArenaKind::KvPage, 1)?;
                arena.txn_mark_write(txn, handle.object_id)?;
                self.pending.push((i, None));
                self.slots[i] = Some(handle.object_id);
                Ok((handle.object_id, None))
            }
            Some(orig) => match arena.txn_cow(txn, orig)? {
                CowPlan::InPlace { handle } => {
                    self.pending.push((i, Some(orig)));
                    Ok((handle.object_id, None))
                }
                CowPlan::Copy { handle, from, to } => {
                    self.pending.push((i, Some(orig)));
                    self.slots[i] = Some(handle.object_id);
                    Ok((handle.object_id, Some(MovePlan { from, to })))
                }
            },
        }
    }

    /// Publish the in-flight write: the repointed slots are now the live mapping
    /// (called after `arena.txn_commit`). Clears the abort-revert log.
    pub fn commit_writes(&mut self) {
        self.pending.clear();
    }

    /// Revert the in-flight write: restore every repointed slot to its prior value
    /// (called on `arena.txn_abort`). The arena frees the staged copies/allocs.
    pub fn abort_writes(&mut self) {
        while let Some((i, prev)) = self.pending.pop() {
            self.slots[i] = prev;
        }
    }

    /// Seal a committed full-page write target (`valid_len == page_size`). echo
    /// host-hashes via `compute_page_hashes` and calls this per eligible page.
    /// On a CAS hit the slot is repointed to the canonical object and the
    /// duplicate freed. Partial pages are never sealed (echo simply doesn't call).
    pub fn seal(
        &mut self,
        idx: u32,
        hash: PageHash,
        arena: &mut Arena,
        cas: &mut KvCas,
    ) -> Result<()> {
        let size = self.slots.len() as u32;
        if idx >= size {
            return Err(WorkingSetError::IndexOutOfRange { index: idx, size });
        }
        let id = self.slots[idx as usize].ok_or(WorkingSetError::UnwrittenPage { index: idx })?;
        let final_id = cas.seal(arena, id, hash)?;
        self.slots[idx as usize] = Some(final_id);
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{Arena, ArenaConfig};
    use crate::page_hash::compute_page_hashes;
    use pie_driver_abi::Brle;

    const PAGE: u32 = 4;

    fn arena(kv_pages: u32) -> Arena {
        Arena::new(ArenaConfig {
            device: 0,
            block_size: PAGE,
            kv_pages,
            rs_blocks: 0,
            scratch_blocks: 0,
            cpu_blocks: 0,
        })
    }

    /// Drive a single-slot forward write end to end (prepare → commit → optional
    /// seal), mirroring echo's txn lifecycle. Uses the working set's current
    /// generation (the valid case); stale-generation rejection is covered
    /// separately via `resolve_write`.
    fn write_slot(
        ws: &mut KvWorkingSet,
        a: &mut Arena,
        cas: &mut KvCas,
        idx: u32,
        seal_hash: Option<PageHash>,
    ) {
        let captured_gen = ws.generation();
        ws.resolve_write(&[idx], captured_gen).unwrap();
        let mut txn = a.txn_begin();
        ws.cow_write_slot(idx, &mut txn, a).unwrap();
        a.txn_commit(txn).unwrap();
        ws.commit_writes();
        if let Some(h) = seal_hash {
            ws.seal(idx, h, a, cas).unwrap();
        }
    }

    fn slot(ws: &KvWorkingSet, i: usize) -> Option<ObjectId> {
        ws.slots[i]
    }

    fn full_hash(toks: &[u32], prev: PageHash) -> PageHash {
        let positions: Vec<u32> = (0..toks.len() as u32).collect();
        let masks: Vec<Brle> = (0..toks.len()).map(|i| Brle::all_true(i + 1)).collect();
        *compute_page_hashes(PAGE as usize, toks, &positions, &masks, prev, None)
            .last()
            .unwrap()
    }

    // 1. alloc — reserves empty slots lazily (no arena pages), bumps generation.
    #[test]
    fn alloc_reserves_lazily_and_bumps_generation() {
        let mut a = arena(16);
        let mut ws = KvWorkingSet::new(PAGE, 0);
        assert_eq!(ws.size(), 0);
        assert_eq!(ws.page_size(), PAGE);

        let r = ws.alloc(3).unwrap();
        assert_eq!(r, PageRange { start: 0, len: 3 });
        assert_eq!(ws.size(), 3);
        assert_eq!(ws.generation(), 1);
        assert_eq!(slot(&ws, 0), None); // reserved, not materialised
        assert_eq!(a.live_objects(), 0); // lazy: no physical pages yet

        assert_eq!(ws.alloc(0).unwrap(), PageRange { start: 3, len: 0 });
        assert_eq!(ws.generation(), 1); // no-op, no bump
    }

    // 2. free — dense compaction + validation; releases materialised pages.
    #[test]
    fn free_dense_compacts_and_validates() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(5).unwrap();
        // Materialise slots 1 and 4 so we can observe their pages being freed.
        write_slot(&mut ws, &mut a, &mut cas, 1, None);
        write_slot(&mut ws, &mut a, &mut cas, 4, None);
        assert_eq!(a.live_objects(), 2);
        let gen0 = ws.generation();

        ws.free(&[3, 1], &mut a, &mut cas).unwrap();
        assert_eq!(ws.size(), 3);
        assert_eq!(slot(&ws, 0), None);
        assert_eq!(slot(&ws, 1), None); // old slot 2
        assert!(slot(&ws, 2).is_some()); // old slot 4 (materialised) survives
        assert_eq!(a.live_objects(), 1); // slot-1's page freed; slot-4's survives
        assert_eq!(ws.generation(), gen0 + 1);

        assert_eq!(
            ws.free(&[9], &mut a, &mut cas),
            Err(WorkingSetError::IndexOutOfRange { index: 9, size: 3 })
        );
        assert_eq!(
            ws.free(&[0, 0], &mut a, &mut cas),
            Err(WorkingSetError::DuplicateIndex { index: 0 })
        );
        assert_eq!(ws.generation(), gen0 + 1); // unchanged after errors
    }

    // 3. reorder — bijection applied; non-permutations rejected.
    #[test]
    fn reorder_applies_bijection_and_rejects_others() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(3).unwrap();
        for i in 0..3 {
            write_slot(&mut ws, &mut a, &mut cas, i, None);
        }
        let ids: Vec<_> = (0..3).map(|i| slot(&ws, i)).collect();
        let gen0 = ws.generation();

        ws.reorder(&[2, 0, 1]).unwrap();
        assert_eq!(slot(&ws, 0), ids[2]);
        assert_eq!(slot(&ws, 1), ids[0]);
        assert_eq!(slot(&ws, 2), ids[1]);
        assert_eq!(ws.generation(), gen0 + 1);

        assert!(matches!(
            ws.reorder(&[0, 1]),
            Err(WorkingSetError::BadPermutation { .. })
        ));
        assert!(matches!(
            ws.reorder(&[0, 1, 5]),
            Err(WorkingSetError::BadPermutation { .. })
        ));
        assert!(matches!(
            ws.reorder(&[0, 0, 1]),
            Err(WorkingSetError::BadPermutation { .. })
        ));
        assert_eq!(ws.generation(), gen0 + 1); // unchanged after errors
    }

    // 4. slice — shares materialised pages by reference; reserved stay reserved.
    #[test]
    fn slice_shares_by_reference() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(4).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 1, None);
        write_slot(&mut ws, &mut a, &mut cas, 2, None);

        let sub = ws.slice(1, 2, &mut a).unwrap();
        assert_eq!(sub.size(), 2);
        assert_eq!(sub.generation(), 0);
        assert_eq!(slot(&sub, 0), slot(&ws, 1));
        assert_eq!(slot(&sub, 1), slot(&ws, 2));
        assert_eq!(a.refcount(slot(&ws, 1).unwrap()).unwrap(), 2); // shared
        assert_eq!(a.live_objects(), 2); // sharing did not allocate

        assert!(matches!(
            ws.slice(3, 2, &mut a),
            Err(WorkingSetError::RangeOutOfBounds { .. })
        ));
    }

    // 5. append — shares borrowed slots, bumps self generation, other unchanged.
    #[test]
    fn append_shares_borrowed_slots() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut dst = KvWorkingSet::new(PAGE, 0);
        dst.alloc(2).unwrap();
        let mut src = KvWorkingSet::new(PAGE, 0);
        src.alloc(1).unwrap();
        write_slot(&mut src, &mut a, &mut cas, 0, None);
        let src_id = slot(&src, 0);
        let gen0 = dst.generation();

        dst.append(&src, &mut a).unwrap();
        assert_eq!(dst.size(), 3);
        assert_eq!(slot(&dst, 2), src_id);
        assert_eq!(a.refcount(src_id.unwrap()).unwrap(), 2); // shared with src
        assert_eq!(dst.generation(), gen0 + 1);
        assert_eq!(src.size(), 1); // unchanged
    }

    // 6. fork — lazy CoW: shares all; first write copies only the touched slot.
    #[test]
    fn fork_is_lazy_first_write_cows_touched_slot() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut parent = KvWorkingSet::new(PAGE, 0);
        parent.alloc(2).unwrap();
        write_slot(&mut parent, &mut a, &mut cas, 0, None);
        write_slot(&mut parent, &mut a, &mut cas, 1, None);
        let p0 = slot(&parent, 0);
        let p1 = slot(&parent, 1);

        let mut child = parent.fork(&mut a).unwrap();
        assert_eq!(child.size(), 2);
        assert_eq!(child.generation(), 0);
        assert_eq!(slot(&child, 0), p0); // shared
        assert_eq!(a.refcount(p0.unwrap()).unwrap(), 2);
        assert_eq!(a.live_objects(), 2); // fork did not allocate

        // First write on the child CoWs only slot 0.
        write_slot(&mut child, &mut a, &mut cas, 0, None);
        assert_ne!(slot(&child, 0), p0); // child slot 0 copied
        assert_eq!(slot(&parent, 0), p0); // parent untouched
        assert_eq!(a.refcount(p0.unwrap()).unwrap(), 1); // parent now sole owner
        assert_eq!(slot(&child, 1), p1); // slot 1 still shared
        assert_eq!(a.live_objects(), 3); // one CoW copy
    }

    // 7. CAS reuse of sealed full pages (W6).
    #[test]
    fn cas_reuses_identical_sealed_full_pages() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let h = full_hash(&[10, 11, 12, 13], 0);

        let mut wa = KvWorkingSet::new(PAGE, 0);
        wa.alloc(1).unwrap();
        write_slot(&mut wa, &mut a, &mut cas, 0, Some(h));
        let canon = slot(&wa, 0);
        assert_eq!(cas.len(), 1);
        assert_eq!(a.live_objects(), 1);

        // A second working set seals identical content → repoints to the canonical
        // object and frees its own duplicate; wa's mapping is untouched (W6).
        let mut wb = KvWorkingSet::new(PAGE, 0);
        wb.alloc(1).unwrap();
        write_slot(&mut wb, &mut a, &mut cas, 0, Some(h));
        assert_eq!(slot(&wb, 0), canon); // CAS reuse
        assert_eq!(slot(&wa, 0), canon); // wa unchanged
        assert_eq!(a.refcount(canon.unwrap()).unwrap(), 2); // shared by wa + wb
        assert_eq!(cas.len(), 1);
        assert_eq!(a.live_objects(), 1); // duplicate freed
    }

    // 8. partial-page non-sharing (W7).
    #[test]
    fn partial_pages_are_never_sealed_or_shared() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut wa = KvWorkingSet::new(PAGE, 0);
        wa.alloc(1).unwrap();
        write_slot(&mut wa, &mut a, &mut cas, 0, None); // partial

        let mut wb = KvWorkingSet::new(PAGE, 0);
        wb.alloc(1).unwrap();
        write_slot(&mut wb, &mut a, &mut cas, 0, None);

        assert_ne!(slot(&wa, 0), slot(&wb, 0)); // distinct objects
        assert_eq!(cas.len(), 0); // nothing entered the CAS index
        assert_eq!(a.live_objects(), 2);
    }

    // 9. stale-generation rejection.
    #[test]
    fn write_against_stale_generation_is_rejected() {
        let mut a = arena(16);
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(1).unwrap();
        let captured = ws.generation();
        ws.alloc(1).unwrap(); // concurrent structural mutation bumps gen
        assert_ne!(captured, ws.generation());

        assert_eq!(
            ws.resolve_write(&[0], captured),
            Err(WorkingSetError::StaleGeneration {
                captured,
                current: ws.generation(),
            })
        );
        ws.resolve_write(&[0], ws.generation()).unwrap(); // fresh gen ok
        let _ = a; // arena unused here
    }

    // 10. CoW isolation through a shared (sliced) slot.
    #[test]
    fn cow_isolates_shared_source_set() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut base = KvWorkingSet::new(PAGE, 0);
        base.alloc(2).unwrap();
        write_slot(&mut base, &mut a, &mut cas, 0, None);
        let base0 = slot(&base, 0);

        let mut view = base.slice(0, 2, &mut a).unwrap();
        assert_eq!(a.refcount(base0.unwrap()).unwrap(), 2);

        write_slot(&mut view, &mut a, &mut cas, 0, None);
        assert_ne!(slot(&view, 0), base0); // view copied
        assert_eq!(slot(&base, 0), base0); // base untouched
        assert_eq!(a.refcount(base0.unwrap()).unwrap(), 1);

        base.destroy(&mut a, &mut cas);
        view.destroy(&mut a, &mut cas);
    }

    // 11. no-leak invariant — destroying all sets returns every block.
    #[test]
    fn destroying_all_sets_releases_every_block() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(4).unwrap();
        for i in 0..4 {
            write_slot(&mut ws, &mut a, &mut cas, i, None);
        }
        let mut forked = ws.fork(&mut a).unwrap();
        write_slot(&mut forked, &mut a, &mut cas, 0, None); // CoW
        let mut sub = ws.slice(1, 2, &mut a).unwrap();
        ws.free(&[3], &mut a, &mut cas).unwrap();
        assert!(a.live_objects() > 0);

        ws.destroy(&mut a, &mut cas);
        forked.destroy(&mut a, &mut cas);
        sub.destroy(&mut a, &mut cas);
        assert_eq!(a.live_objects(), 0);
    }

    // 12. real compute_page_hashes drives prefix-sensitive CAS hit/miss.
    #[test]
    fn real_hashes_drive_prefix_sensitive_cas() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let h_a = full_hash(&[1, 2, 3, 4], 0);
        assert_eq!(h_a, full_hash(&[1, 2, 3, 4], 0)); // deterministic
        assert_ne!(h_a, full_hash(&[1, 2, 3, 9], 0)); // content-sensitive
        assert_ne!(h_a, full_hash(&[1, 2, 3, 4], 999)); // prefix-sensitive

        let mut w1 = KvWorkingSet::new(PAGE, 0);
        w1.alloc(1).unwrap();
        write_slot(&mut w1, &mut a, &mut cas, 0, Some(h_a));
        let canon = slot(&w1, 0);

        let mut w2 = KvWorkingSet::new(PAGE, 0);
        w2.alloc(1).unwrap();
        write_slot(&mut w2, &mut a, &mut cas, 0, Some(h_a));
        assert_eq!(slot(&w2, 0), canon); // hit

        let mut w3 = KvWorkingSet::new(PAGE, 0);
        w3.alloc(1).unwrap();
        let h_c = full_hash(&[1, 2, 3, 9], 0);
        write_slot(&mut w3, &mut a, &mut cas, 0, Some(h_c));
        assert_ne!(slot(&w3, 0), canon); // miss
        assert_eq!(cas.len(), 2);
    }

    // 13. abort reverts slot pointers and frees staged copies.
    #[test]
    fn abort_reverts_slots_and_frees_staged() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut base = KvWorkingSet::new(PAGE, 0);
        base.alloc(2).unwrap();
        write_slot(&mut base, &mut a, &mut cas, 0, None);
        let base0 = slot(&base, 0);

        // Share slot 0 so the write target is CoW'd (a copy is staged).
        let mut view = base.slice(0, 2, &mut a).unwrap();
        let live_before = a.live_objects();

        // Prepare a write on the shared slot, then abort.
        let mut txn = a.txn_begin();
        let (_new, mv) = view.cow_write_slot(0, &mut txn, &mut a).unwrap();
        assert!(mv.is_some()); // shared ⇒ a copy was staged
        assert_ne!(slot(&view, 0), base0); // repointed to the staged copy
        a.txn_abort(txn);
        view.abort_writes();

        assert_eq!(slot(&view, 0), base0); // reverted
        assert_eq!(a.live_objects(), live_before); // staged copy freed
        assert_eq!(a.refcount(base0.unwrap()).unwrap(), 2); // original intact

        base.destroy(&mut a, &mut cas);
        view.destroy(&mut a, &mut cas);
    }

    // 14. resolve_read returns ObjectIds; errors on unwritten / out-of-range.
    #[test]
    fn resolve_read_returns_objects_and_validates() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(3).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        write_slot(&mut ws, &mut a, &mut cas, 1, None);

        let objs = ws.resolve_read(0, 2).unwrap();
        assert_eq!(objs, vec![slot(&ws, 0).unwrap(), slot(&ws, 1).unwrap()]);

        // slot 2 is reserved/unwritten.
        assert_eq!(
            ws.resolve_read(0, 3),
            Err(WorkingSetError::UnwrittenPage { index: 2 })
        );
        assert!(matches!(
            ws.resolve_read(2, 5),
            Err(WorkingSetError::RangeOutOfBounds { .. })
        ));

        ws.destroy(&mut a, &mut cas);
    }

    // 15. binding: model eager, driver lazy; slice/fork inherit; device() default.
    #[test]
    fn binding_model_eager_driver_lazy() {
        let mut a = arena(16);
        let mut ws = KvWorkingSet::new(PAGE, 3); // model_idx = 3
        assert_eq!(ws.model_idx(), 3);
        assert_eq!(ws.bound_driver(), None); // unbound until first forward
        assert_eq!(ws.device(), (3, 0)); // driver defaults to 0 while unbound

        ws.bind_driver(0); // echo binds on first materialisation (v1 = 0)
        assert_eq!(ws.bound_driver(), Some(0));
        assert_eq!(ws.device(), (3, 0));
        ws.bind_driver(0); // idempotent for the same driver

        // slice/fork inherit the binding.
        ws.alloc(2).unwrap();
        let mut sub = ws.slice(0, 2, &mut a).unwrap();
        assert_eq!(sub.model_idx(), 3);
        assert_eq!(sub.bound_driver(), Some(0));
        let mut forked = ws.fork(&mut a).unwrap();
        assert_eq!(forked.device(), (3, 0));

        let mut cas = KvCas::new();
        ws.destroy(&mut a, &mut cas);
        sub.destroy(&mut a, &mut cas);
        forked.destroy(&mut a, &mut cas);
    }

    // 16. hash_of + chained sealing onto a non-empty context (echo's accessor).
    #[test]
    fn hash_of_enables_chained_sealing() {
        let mut a = arena(16);
        let mut cas = KvCas::new();

        // Seal page A (chained from empty context, prev = 0).
        let h_a = full_hash(&[1, 2, 3, 4], 0);
        let mut ctx = KvWorkingSet::new(PAGE, 0);
        ctx.alloc(2).unwrap();
        write_slot(&mut ctx, &mut a, &mut cas, 0, Some(h_a));
        let obj_a = slot(&ctx, 0).unwrap();

        // hash_of returns A's sealed hash; an unsealed slot returns None.
        assert_eq!(cas.hash_of(obj_a), Some(h_a));
        write_slot(&mut ctx, &mut a, &mut cas, 1, None); // partial — never sealed
        assert_eq!(cas.hash_of(slot(&ctx, 1).unwrap()), None);

        // Seal page B chained FROM A's hash (the non-empty-context tip).
        let prev = cas.hash_of(obj_a).unwrap();
        let h_b = full_hash(&[5, 6, 7, 8], prev);
        // free the partial slot 1 first, then write a fresh full page-1 = B.
        ctx.free(&[1], &mut a, &mut cas).unwrap();
        ctx.alloc(1).unwrap();
        write_slot(&mut ctx, &mut a, &mut cas, 1, Some(h_b));
        let obj_b = slot(&ctx, 1).unwrap();
        assert_eq!(cas.hash_of(obj_b), Some(h_b));

        // A second working set replaying the SAME A→B prefix dedups both pages.
        let mut other = KvWorkingSet::new(PAGE, 0);
        other.alloc(2).unwrap();
        write_slot(&mut other, &mut a, &mut cas, 0, Some(h_a));
        let prev2 = cas.hash_of(slot(&other, 0).unwrap()).unwrap();
        let h_b2 = full_hash(&[5, 6, 7, 8], prev2);
        write_slot(&mut other, &mut a, &mut cas, 1, Some(h_b2));
        assert_eq!(slot(&other, 0).unwrap(), obj_a); // CAS reuse of A
        assert_eq!(slot(&other, 1).unwrap(), obj_b); // CAS reuse of B (chained)

        ctx.destroy(&mut a, &mut cas);
        other.destroy(&mut a, &mut cas);
    }
}

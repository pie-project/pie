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
//! `crate::working_set::page_hash::compute_page_hashes` (unchanged).

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use crate::arena::{Arena, ArenaError, ArenaKind, ArenaTxn, CowPlan, MovePlan, ObjectId};

/// PTIR Thrust 1 / Phase M1 toggle (feature `ws-slot-ids`). Compiled as a
/// `const` so **both** semantic branches type-check and the dead branch is
/// eliminated by the optimizer (no `#[cfg]` sprawl). OFF ⇒ legacy dense array
/// (compaction-on-free, contiguous `alloc`, generation bumps on every structural
/// mutation). ON ⇒ stable slot table (tombstone-on-free with trailing-truncate,
/// recycling `alloc_slots`, `size()` = live count, generation bumps only on
/// `reorder`/`compact`, W8). Flipped default-on at the M1 exit gate.
pub const SLOT_IDS: bool = true;

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

/// A **live** page slot in the slot table (W1). `Reserved` = allocated but not
/// yet materialised (lazy; the physical page is created on first write);
/// `Page(id)` = a materialised page object in the unified arena. A freed
/// (tombstoned) slot is the OUTER `None` in `slots: Vec<Option<PageSlot>>` and
/// its id sits on the free list until recycled — the id itself never renumbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageSlot {
    Reserved,
    Page(ObjectId),
}

impl PageSlot {
    /// The materialised page object, or `None` while still reserved.
    fn object(self) -> Option<ObjectId> {
        match self {
            PageSlot::Reserved => None,
            PageSlot::Page(id) => Some(id),
        }
    }
}

/// Sentinel `slot_to_block_table` entry for a slot with no physical page — a
/// tombstoned (freed), reserved, or unmaterialised slot. Chosen as `u32::MAX` so
/// it matches the driver `launch_resolve_slot_to_block` out-of-range sentinel: a
/// device-produced `pages` slot id that lands on an unmapped entry resolves to a
/// LOUD `0xFFFFFFFF` physical page, never a silent wrong-page gather.
pub const SLOT_UNMAPPED: u32 = u32::MAX;

/// A run of live tokens in the current page layout, the input to
/// [`KvWorkingSet::compact`] (M4 / W5). `len` tokens starting at in-page offset
/// `start` of slot `src_slot` survive; `compact` packs every run densely into
/// fresh slots via `gather_tokens` and returns the [`CompactRemap`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenRun {
    pub src_slot: u32,
    pub start: u32,
    pub len: u32,
}

/// One entry of the `gather_tokens` plan (M4/M5): copy `len` tokens from in-page
/// offset `src_off` of `src_slot` to offset `dst_off` of `dst_slot`. Engine
/// machinery, never a program op (W5); the driver executes it (a streaming copy),
/// the mock/host path records it. Emitted densely by [`KvWorkingSet::compact`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherOp {
    pub src_slot: u32,
    pub src_off: u32,
    pub dst_slot: u32,
    pub dst_off: u32,
    pub len: u32,
}

/// A [`GatherOp`] resolved to PHYSICAL page ids — the plan the driver's
/// `gather_tokens` kernel (M3) consumes (mirrors the C++ `GatherTokenOp`).
/// `compact` resolves each op's `src_slot`/`dst_slot` to its arena block while the
/// source pages are still live (before the grace-free), so the driver copy reads
/// valid source KV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageGatherOp {
    pub src_page: u32,
    pub src_off: u32,
    pub dst_page: u32,
    pub dst_off: u32,
    pub len: u32,
}

/// Result of [`KvWorkingSet::compact`] (W5): the freshly allocated packed slots,
/// the `gather_tokens` plan the driver runs, and the old slots freed (grace-
/// period recycled). The host re-feeds `gather` / `new_slots` through the
/// geometry channels (§5.1's `take`→`put`); the new token position of the `k`-th
/// live token is `(new_slots[k / page_size], k % page_size)`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct CompactRemap {
    pub new_slots: Vec<u32>,
    pub gather: Vec<GatherOp>,
    /// The gather plan resolved to physical page ids — what the driver's
    /// `gather_tokens` kernel consumes (empty until the dst pages are
    /// materialised, i.e. populated by [`KvWorkingSet::compact`]).
    pub page_gather: Vec<PageGatherOp>,
    pub freed_slots: Vec<u32>,
}

/// Pure classification of a working set's suspendability (no mutation) — the
/// orchestrator uses it to gate victim eligibility + estimate reclaimable pages
/// BEFORE committing to a suspend. See [`KvWorkingSet::classify_for_suspend`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SuspendClass {
    /// Uniquely-owned (`rc == 1`) materialised pages — STASHED (D2H offload) or
    /// dropped-to-replay on suspend; each frees one block.
    pub owned: Vec<(u32, ObjectId)>,
    /// Shared (`rc > 1`) materialised pages — a prefix another set/beam
    /// references. NEVER stashed: suspend only releases this set's ref; the page
    /// stays resident for the other holders (W6). Frees no block here.
    pub shared: Vec<(u32, ObjectId)>,
    /// Owned pages freeable NOW — `rc==1` AND not arena-pinned by an in-flight
    /// forward (safe to offload/release immediately). The orchestrator's
    /// "keep-evicting" math: how many blocks this suspend yields right away.
    pub freed_now: u32,
    /// Owned pages an in-flight forward arena-PINS (run-ahead co-batch: a sibling
    /// forward of this same set is still in flight over the page — reads or write
    /// targets). NOT stashable now; their blocks free at that forward's finalize
    /// (`txn_unpin`), so the orchestrator's "wait-for-pin-release" math counts
    /// them toward the eventual reclaim without over-crediting `freed_now`.
    pub freed_on_grace: u32,
}

/// Record of a warm-suspend (Task-B): the stashed pages' D2H copies + the shared
/// refs released. The orchestrator holds this and hands it to
/// [`KvWorkingSet::restore_pages_warm`]. `stash` = `(slot, ObjectId, D2H plan)`
/// for the uniquely-owned pages moved to the CPU arena; `released_shared` =
/// `(slot, ObjectId)` shared pages whose ref was dropped (recomputed/re-shared
/// on restore).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SuspendPlan {
    pub stash: Vec<(u32, ObjectId, MovePlan)>,
    pub released_shared: Vec<(u32, ObjectId)>,
    /// Owned pages whose CPU-stash alloc failed at STAGE time: cold-dropped at
    /// commit (ref released → page frees; restore replays from lineage). Kept
    /// separate so `commit_suspend` releases (not repoints) them.
    pub cold: Vec<(u32, ObjectId)>,
    /// Blocks freed by THIS call (== `stash.len()` + owned drops + shared last-ref
    /// frees). The orchestrator adds this to its running reclaim total.
    pub freed_now: u32,
    /// Blocks that free later at an in-flight forward's finalize: owned pages the
    /// forward arena-PINS (run-ahead co-batch) are skipped by `classify_for_suspend`
    /// (left mapped, never stashed), so their blocks free when that forward's
    /// `txn_unpin` drives `rc→0`. The orchestrator counts these toward the eventual
    /// reclaim (waits for the pin release) rather than evicting another victim to
    /// cover them.
    pub freed_on_grace: u32,
}

/// Errors are **returned**, never trapped (W2/W3). The WIT shell maps these to
/// the inferlet-facing `error` via `Display`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkingSetError {
    /// A relative index was `>= size`.
    IndexOutOfRange { index: u32, size: u32 },
    /// `free`/`reorder` received the same index twice.
    DuplicateIndex { index: u32 },
    /// `free-slots` targeted a slot that is already tombstoned (double-free).
    DoubleFree { index: u32 },
    /// A read/write/seal targeted a freed (tombstoned) slot — the id is valid
    /// range but no longer live (recycle it via `alloc-slots` before reuse).
    FreedSlot { index: u32 },
    /// `reorder` permutation was not a full bijection over `0..size`.
    BadPermutation { size: u32 },
    /// `slice`/`resolve_read` range exceeded the array.
    RangeOutOfBounds { start: u32, len: u32, size: u32 },
    /// A read targeted a slot that has not been written yet (still reserved).
    UnwrittenPage { index: u32 },
    /// A write/seal ran against a captured generation that no longer matches —
    /// a concurrent structural mutation (alloc/free/reorder/append) invalidated it.
    StaleGeneration { captured: u32, current: u32 },
    /// The arena pool is exhausted. Kept TYPED (Task-B): the prep seam routes
    /// this — and only this — arena failure to the contention orchestrator
    /// (preempt/restore) instead of surfacing an inferlet error.
    OutOfBlocks {
        kind: ArenaKind,
        requested: u32,
        available: u32,
    },
    /// Any other underlying arena failure (e.g. an invalid residency).
    Arena(String),
}

/// Identity of one forward's KV write-transaction (thrust-2 phase S4). Opened by
/// [`KvWorkingSet::begin_write_txn`] and passed to [`KvWorkingSet::cow_write_slot`]
/// / [`commit_writes`](KvWorkingSet::commit_writes) /
/// [`abort_writes`](KvWorkingSet::abort_writes) so commit/abort touch ONLY that
/// forward's repointed slots — letting more than one prepared forward be
/// outstanding against disjoint slots of one working set safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WriteTxnId(pub u64);

impl fmt::Display for WorkingSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkingSetError::IndexOutOfRange { index, size } => {
                write!(f, "index {index} out of range (size {size})")
            }
            WorkingSetError::DuplicateIndex { index } => write!(f, "duplicate index {index}"),
            WorkingSetError::DoubleFree { index } => write!(f, "double free of slot {index}"),
            WorkingSetError::FreedSlot { index } => write!(f, "slot {index} was freed"),
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
                write!(
                    f,
                    "stale generation: captured {captured}, current {current}"
                )
            }
            WorkingSetError::OutOfBlocks {
                kind,
                requested,
                available,
            } => {
                write!(
                    f,
                    "arena: out of {kind:?} blocks (requested {requested}, available {available})"
                )
            }
            WorkingSetError::Arena(e) => write!(f, "arena: {e}"),
        }
    }
}

impl std::error::Error for WorkingSetError {}

impl From<ArenaError> for WorkingSetError {
    fn from(e: ArenaError) -> Self {
        match e {
            // The contention trigger stays typed end-to-end (Task-B).
            ArenaError::OutOfBlocks {
                kind,
                requested,
                available,
                ..
            } => WorkingSetError::OutOfBlocks {
                kind,
                requested,
                available,
            },
            other => WorkingSetError::Arena(other.to_string()),
        }
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
            // Only dedup onto a canonical that is (a) still LIVE, (b) still OURS
            // (`canon ↦ hash`), and (c) GPU-RESIDENT.
            //  (c) is the BAR-1 replay fix: a page STASHED to CPU under preemption
            //      stays in this index (offload does NOT unindex it), so without
            //      this check a lane sealing the same prefix would dedup onto a
            //      CPU-resident page — its forward then reads the freed GPU blocks
            //      (empty context) and decodes a degenerate `[198,…]` replay.
            //  (a)/(b) guard a freed/recycled dangling entry: `txn_commit`'s
            //      `on_commit_decref` frees a canonical via raw `arena.decref`
            //      (never cleaning this index), so a stale `index[hash]` could
            //      otherwise `incref` a recycled/wrong object.
            // A stale hit is dropped and `writable` registered as the fresh
            // canonical (correctness over dedup; the stashed/dead page keeps its
            // own mapping, just stops being a dedup target).
            let usable = arena.refcount(canon).is_ok()
                && self.canonical.get(&canon) == Some(&hash)
                && arena
                    .residency(canon)
                    .map_or(false, |r| r == crate::arena::Residency::Gpu);
            if usable {
                arena.incref(canon)?;
                self.release(arena, writable);
                return Ok(canon);
            }
            self.index.remove(&hash);
        }
        self.index.insert(hash, writable);
        self.canonical.insert(writable, hash);
        Ok(writable)
    }
}

/// A slot table of KV page slots (W1/W2). A slot **id** is the index into
/// `slots` and is a **stable handle**: it never renumbers across `free`
/// (survivors keep their ids). `None` = tombstoned/free (id parked on
/// `free_list`); `Some(PageSlot)` = a live slot. Relative `u32` slot ids are the
/// only references that cross the API.
#[derive(Debug)]
pub struct KvWorkingSet {
    /// Slot table indexed by stable slot id. `None` = tombstoned/free,
    /// `Some(PageSlot::Reserved)` = live but not materialised,
    /// `Some(PageSlot::Page(id))` = a page object in the unified arena.
    slots: Vec<Option<PageSlot>>,
    /// Tombstoned (freed) slot ids available for recycling by `alloc_slots`,
    /// ordered so recycling pops the LOWEST id first (keeps recycled runs as
    /// contiguous as possible). Invariant: `free_list == { i : slots[i].is_none() }`
    /// and the array never ends in a tombstone (trailing tombstones are
    /// truncated on `free`). `size()` = `slots.len() - free_list.len()`.
    free_list: BTreeSet<u32>,
    /// Monotonic counter. Under `ws-slot-ids` it bumps ONLY on `reorder`/
    /// `compact` (W8); legacy builds bump on every structural mutation. Captured
    /// by a forward write and rejected on mismatch. `u32` to match the WIT
    /// `generation()` accessor + `kv-output.generation`.
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
    /// Per-forward abort-revert logs, keyed by [`WriteTxnId`] (thrust-2 S4): each
    /// prepared forward's repointed slots `(slot_id, prev_value)` live under its
    /// own key, so [`commit_writes`](Self::commit_writes) /
    /// [`abort_writes`](Self::abort_writes) touch only that forward's slots and
    /// two outstanding forwards against disjoint slots stay isolated.
    pending: HashMap<WriteTxnId, Vec<(usize, Option<PageSlot>)>>,
    /// Slot id → the write-txn that currently owns its live (uncommitted) write.
    /// Guards against two outstanding forwards repointing the same slot
    /// (`OverlappingWrite`). Cleared per slot on that txn's commit/abort.
    live_write_slots: HashMap<usize, WriteTxnId>,
    /// Monotonic source of [`WriteTxnId`]s (per working set).
    next_write_txn: u64,
}

impl KvWorkingSet {
    /// Fresh, empty working set for `model_idx` whose KV page holds `page_size`
    /// tokens. The driver is unbound until the first forward write materialises a
    /// page (see [`bind_driver`](Self::bind_driver)).
    pub fn new(page_size: u32, model_idx: usize) -> Self {
        KvWorkingSet {
            slots: Vec::new(),
            free_list: BTreeSet::new(),
            generation: 0,
            page_size,
            model_idx,
            bound_driver: None,
            pending: HashMap::new(),
            live_write_slots: HashMap::new(),
            next_write_txn: 0,
        }
    }

    // =====================================================================
    // Accessors (forward contract)
    // =====================================================================

    /// Number of **live** page slots (W1: `size()` counts live slots, not the
    /// physical array length — tombstoned ids do not count).
    pub fn size(&self) -> u32 {
        (self.slots.len() - self.free_list.len()) as u32
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

    /// Append `n` fresh reserved slots at the end; returns the contiguous range.
    /// **Grow-only** migration shim (never recycles) so the `page-range` contract
    /// stays contiguous even under slot-id semantics; use [`alloc_slots`] to
    /// recycle freed ids. Slots are materialised lazily on first write, so this
    /// never touches the arena and cannot fail on capacity. `alloc(0)` is a
    /// structural no-op.
    pub fn alloc(&mut self, n: u32) -> Result<PageRange> {
        let start = self.slots.len() as u32;
        for _ in 0..n {
            self.slots.push(Some(PageSlot::Reserved));
        }
        if n > 0 && !SLOT_IDS {
            self.generation += 1;
        }
        Ok(PageRange { start, len: n })
    }

    /// Stable-id allocation (W1/W2): return `n` slot ids, recycling tombstoned
    /// ids (lowest first) before growing. The returned ids need not be
    /// contiguous. Never touches the arena (lazy materialisation). Does not bump
    /// `generation` under slot-id semantics (W8). `alloc_slots(0)` returns `[]`.
    pub fn alloc_slots(&mut self, n: u32) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let id = if SLOT_IDS {
                // Recycle the lowest freed id, else grow.
                match self.free_list.iter().next().copied() {
                    Some(id) => {
                        self.free_list.remove(&id);
                        id
                    }
                    None => self.slots.len() as u32,
                }
            } else {
                self.slots.len() as u32
            };
            let idu = id as usize;
            if idu < self.slots.len() {
                self.slots[idu] = Some(PageSlot::Reserved);
            } else {
                self.slots.push(Some(PageSlot::Reserved));
            }
            ids.push(id);
        }
        if n > 0 && !SLOT_IDS {
            self.generation += 1;
        }
        Ok(ids)
    }

    /// Remove the slots at `indices`. MIGRATION SHIM: under slot-id semantics
    /// (`ws-slot-ids`) this delegates to [`free_slots`] (non-compacting tombstone
    /// + trailing-truncate; survivors keep their ids). Legacy builds densely
    /// compact survivors left, preserving order (all-at-once, call-time).
    /// Out-of-range or duplicate indices return `error` and leave the array
    /// untouched.
    pub fn free(&mut self, indices: &[u32], arena: &mut Arena, cas: &mut KvCas) -> Result<()> {
        if SLOT_IDS {
            return self.free_slots(indices, arena, cas);
        }
        // ---- Legacy dense compaction (feature off) ----
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
                if let Some(id) = slot.and_then(PageSlot::object) {
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

    /// Tombstone the slots at `ids` (W1/W4, non-compacting): survivors keep their
    /// ids, freed ids are parked on the free list for [`alloc_slots`] to recycle,
    /// and a freed trailing run is truncated so a freed suffix leaves no hole. All
    /// removals apply together (validated first). Out-of-range → `IndexOutOfRange`;
    /// an already-tombstoned id → `DoubleFree`; a repeated id in the call →
    /// `DuplicateIndex` (the call never traps). Materialised pages are released to
    /// the arena/CAS. Debug builds assert each freed id is unreferenced by any
    /// in-flight pass (M1.4 grace-period precondition, W4).
    pub fn free_slots(&mut self, ids: &[u32], arena: &mut Arena, cas: &mut KvCas) -> Result<()> {
        let size = self.slots.len();
        let mut seen = vec![false; size];
        for &i in ids {
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
            if self.slots[iu].is_none() {
                return Err(WorkingSetError::DoubleFree { index: i });
            }
            seen[iu] = true;
        }
        if ids.is_empty() {
            return Ok(());
        }
        for &i in ids {
            let iu = i as usize;
            if let Some(id) = self.slots[iu].take().and_then(PageSlot::object) {
                cas.release(arena, id);
            }
            self.free_list.insert(i);
        }
        self.truncate_trailing_tombstones();
        Ok(())
    }

    /// Pop trailing tombstones so the array never ends in a hole (bounds growth
    /// and keeps a freed suffix hole-free — the SDK's dense tail-trim pattern).
    /// Survivors are never touched, so no id renumbers.
    fn truncate_trailing_tombstones(&mut self) {
        while matches!(self.slots.last(), Some(None)) {
            let last = (self.slots.len() - 1) as u32;
            self.slots.pop();
            self.free_list.remove(&last);
        }
    }

    // =====================================================================
    // Mark-sweep GC helper (M4) — "mark is host code, sweep is `free`" (§5.2)
    // =====================================================================

    /// Host-side mark: the live slot ids NOT in `reachable` — the **dead** set the
    /// host then sweeps with [`free_slots`]. Pure over the slot table (no arena),
    /// so the GC needs no engine API (overview §5.2). The caller supplies
    /// `reachable` as the peeked reachability snapshot UNIONED with every id
    /// granted since the snapshot (so a freshly `alloc`-d id is never swept before
    /// the program that requested it can reference it). Returned ids are sorted
    /// ascending and exclude tombstoned ids (already free).
    pub fn mark_dead(&self, reachable: &BTreeSet<u32>) -> Vec<u32> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(i, s)| s.is_some() && !reachable.contains(&(*i as u32)))
            .map(|(i, _)| i as u32)
            .collect()
    }

    // =====================================================================
    // Token-space compaction (M4 / W5) — `gather_tokens` orchestration
    // =====================================================================

    /// Explicit token-space compaction (W5, never a side effect of `free`): pack
    /// the live token `runs` densely into freshly allocated + materialised slots,
    /// run `gather` (the `gather_tokens` driver op — mock/host no-op or the CUDA
    /// kernel in M3) to copy the live tokens into them, free the source slots, and
    /// return the [`CompactRemap`] — the resolved page-level gather plan plus the
    /// old→new slot geometry the host re-feeds (§5.1's `take`→`put`). Renumbers
    /// token positions, so it bumps `generation` (W8).
    ///
    /// Ordering (load-bearing): dst pages are materialised and the plan resolved
    /// to physical page ids **while the source pages are still live**, then
    /// `gather` runs (reads src pages → writes dst pages) **before** the source
    /// slots are freed — so the copy always reads valid source KV. `gather` is
    /// called synchronously with the resolved [`PageGatherOp`] plan; the CALLER's
    /// driver binding launches the kernel + stream-syncs inside it (mock: no-op).
    ///
    /// Quiescent-point contract (overview §6.2/§6.4): the CALLER drains its
    /// in-flight step first when geometry lives on device, or re-feeds ordinarily
    /// when host-owned. Errors (out-of-range/freed source, a run past the page)
    /// leave the set untouched.
    pub fn compact<F: FnOnce(&[PageGatherOp])>(
        &mut self,
        runs: &[TokenRun],
        arena: &mut Arena,
        cas: &mut KvCas,
        gather: F,
    ) -> Result<CompactRemap> {
        // Validate every run first (atomic — no mutation on error).
        for run in runs {
            let iu = run.src_slot as usize;
            if iu >= self.slots.len() {
                return Err(WorkingSetError::IndexOutOfRange {
                    index: run.src_slot,
                    size: self.slots.len() as u32,
                });
            }
            if self.slots[iu].is_none() {
                return Err(WorkingSetError::FreedSlot {
                    index: run.src_slot,
                });
            }
            if run.start + run.len > self.page_size {
                return Err(WorkingSetError::RangeOutOfBounds {
                    start: run.start,
                    len: run.len,
                    size: self.page_size,
                });
            }
        }

        let total: u32 = runs.iter().map(|r| r.len).sum();
        let fresh_pages = total.div_ceil(self.page_size);
        let new_slots = self.alloc_slots(fresh_pages)?;

        // Materialise the dst slots (allocate a fresh KV page each) so the gather
        // has physical pages to write into. Fresh, uniquely-owned pages.
        for &ds in &new_slots {
            let handle = arena.alloc(ArenaKind::KvPage, 1)?;
            self.slots[ds as usize] = Some(PageSlot::Page(handle.object_id));
        }

        // Pack runs densely; a run straddling a destination page boundary splits
        // into per-page gather ops (the driver copies contiguous within a page).
        // Resolve each op to physical page ids WHILE THE SOURCE IS STILL LIVE.
        let mut gather_ops = Vec::new();
        let mut page_gather = Vec::new();
        let mut cursor = 0u32; // token index in the packed output
        let page_of = |ws: &KvWorkingSet, slot: u32, arena: &Arena| -> Result<u32> {
            match ws.slots[slot as usize].and_then(PageSlot::object) {
                Some(obj) => Ok(arena.blocks(obj)?[0]),
                None => Err(WorkingSetError::FreedSlot { index: slot }),
            }
        };
        for run in runs {
            let mut remaining = run.len;
            let mut src_off = run.start;
            while remaining > 0 {
                let dst_slot = new_slots[(cursor / self.page_size) as usize];
                let dst_off = cursor % self.page_size;
                let room = self.page_size - dst_off;
                let chunk = remaining.min(room);
                gather_ops.push(GatherOp {
                    src_slot: run.src_slot,
                    src_off,
                    dst_slot,
                    dst_off,
                    len: chunk,
                });
                page_gather.push(PageGatherOp {
                    src_page: page_of(self, run.src_slot, arena)?,
                    src_off,
                    dst_page: page_of(self, dst_slot, arena)?,
                    dst_off,
                    len: chunk,
                });
                cursor += chunk;
                src_off += chunk;
                remaining -= chunk;
            }
        }

        // Run the gather (reads src pages → writes dst pages) BEFORE freeing the
        // source — so the driver copy always reads valid source KV.
        gather(&page_gather);

        // Free the distinct source slots (excluding any reused as a fresh dst —
        // there are none, since `alloc_slots` returns fresh/recycled-dead ids).
        let mut src_slots: Vec<u32> = runs.iter().map(|r| r.src_slot).collect();
        src_slots.sort_unstable();
        src_slots.dedup();
        let freed_slots: Vec<u32> = src_slots
            .into_iter()
            .filter(|s| !new_slots.contains(s))
            .collect();
        self.free_slots(&freed_slots, arena, cas)?;

        self.generation += 1; // W8: compact renumbers → bump

        Ok(CompactRemap {
            new_slots,
            gather: gather_ops,
            page_gather,
            freed_slots,
        })
    }

    /// Apply the full bijection `perm` over the physical array `0..slots.len()`:
    /// `new[i] = old[perm[i]]`. A wrong length, out-of-range, or duplicated entry
    /// returns `error` (no trap). Renumbers slot ids, so it bumps `generation`
    /// (W8). The free list is rebuilt from the permuted array.
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
        let new_slots: Vec<Option<PageSlot>> =
            perm.iter().map(|&p| self.slots[p as usize]).collect();
        self.slots = new_slots;
        self.rebuild_free_list();
        self.generation += 1;
        Ok(())
    }

    /// Recompute `free_list` from `slots` (tombstone positions) after a bulk
    /// rewrite, then truncate any trailing tombstones.
    fn rebuild_free_list(&mut self) {
        self.free_list = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_none())
            .map(|(i, _)| i as u32)
            .collect();
        self.truncate_trailing_tombstones();
    }

    /// Return a new working set whose array is the slots in `[start, start+len)`
    /// (physical range), sharing materialised pages **by reference** (incref'd);
    /// reserved slots copy as reserved and tombstones copy as tombstones. First
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
            if let Some(id) = slot.and_then(PageSlot::object) {
                arena.incref(id)?;
            }
            slots.push(*slot);
        }
        let mut out = KvWorkingSet {
            slots,
            free_list: BTreeSet::new(),
            generation: 0,
            page_size: self.page_size,
            model_idx: self.model_idx,
            bound_driver: self.bound_driver,
            pending: HashMap::new(),
            live_write_slots: HashMap::new(),
            next_write_txn: 0,
        };
        out.rebuild_free_list();
        Ok(out)
    }

    /// Append `other`'s live slots onto self **by reference** (materialised pages
    /// incref'd). Shares page objects; first divergent write CoWs. `other` is
    /// unchanged.
    pub fn append(&mut self, other: &KvWorkingSet, arena: &mut Arena) -> Result<()> {
        let snap = other.slot_objects();
        self.append_shared(&snap, arena)
    }

    /// Snapshot of the **live** slot→object mapping (for the WIT shell, which must
    /// read `other`'s slots before taking a `&mut` borrow of `self` from the same
    /// resource table — avoids a double-borrow). Tombstones are dropped; a
    /// reserved slot maps to `None`, a materialised slot to `Some(id)`.
    pub fn slot_objects(&self) -> Vec<Option<ObjectId>> {
        self.slots
            .iter()
            .filter_map(|s| s.map(PageSlot::object))
            .collect()
    }

    /// Append the given shared slots onto self as fresh live slots at the end
    /// (incref each materialised page). Used by `append` and the WIT shell.
    /// `None` = reserved, `Some(id)` = a shared materialised page.
    pub fn append_shared(&mut self, slots: &[Option<ObjectId>], arena: &mut Arena) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }
        for slot in slots {
            match *slot {
                Some(id) => {
                    arena.incref(id)?;
                    self.slots.push(Some(PageSlot::Page(id)));
                }
                None => self.slots.push(Some(PageSlot::Reserved)),
            }
        }
        if !SLOT_IDS {
            self.generation += 1;
        }
        Ok(())
    }

    /// Lazy-CoW fork: a new working set sharing **all** current slots by reference
    /// (materialised pages incref'd; tombstones preserved). First divergent write
    /// on either side CoWs.
    pub fn fork(&self, arena: &mut Arena) -> Result<KvWorkingSet> {
        for slot in &self.slots {
            if let Some(id) = slot.and_then(PageSlot::object) {
                arena.incref(id)?;
            }
        }
        Ok(KvWorkingSet {
            slots: self.slots.clone(),
            free_list: self.free_list.clone(),
            generation: 0,
            page_size: self.page_size,
            model_idx: self.model_idx,
            bound_driver: self.bound_driver,
            pending: HashMap::new(),
            live_write_slots: HashMap::new(),
            next_write_txn: 0,
        })
    }

    /// Release every page this working set references. Called by the WIT resource
    /// `drop` handler. Reverts any in-flight write first.
    pub fn destroy(&mut self, arena: &mut Arena, cas: &mut KvCas) {
        // Undo any half-prepared forward so we don't leak/misattribute copies.
        self.abort_all_writes();
        for slot in self.slots.drain(..) {
            if let Some(id) = slot.and_then(PageSlot::object) {
                cas.release(arena, id);
            }
        }
        self.free_list.clear();
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
            let idx = start + offset as u32;
            match slot {
                None => return Err(WorkingSetError::FreedSlot { index: idx }),
                Some(PageSlot::Reserved) => {
                    if std::env::var("PTIR_COW_TRACE").is_ok() {
                        eprintln!(
                            "[COW] slot {idx}: Reserved -> txn_alloc FRESH page (no persist!)"
                        );
                    }
                    return Err(WorkingSetError::UnwrittenPage { index: idx });
                }
                Some(PageSlot::Page(id)) => out.push(*id),
            }
        }
        Ok(out)
    }

    /// Dense slot → physical page-pool `BlockId` table for the C1-FINAL device-
    /// geometry resolve (beam/§6.1 Design B). Entry `[i]` is slot `i`'s physical
    /// KV page block, or [`SLOT_UNMAPPED`] for a tombstoned / reserved /
    /// unmaterialised slot. DENSE over the whole slot domain `0..slots.len()`
    /// (NOT compacted like [`slot_objects`]) so a device-produced `pages` channel's
    /// slot ids index it DIRECTLY (`launch_resolve_slot_to_block`
    /// `page_indices[i] = slot_to_block[pages[i]]`); slot id 0 is a real entry.
    /// The host uploads this compact `[num_slots]` dict once per fire — its OWN
    /// authoritative slot→page map, NOT the per-beam geometry (which stays device-
    /// produced and host-unread) — and the driver resolves `pages` through it on
    /// device, so the forward geometry never leaves the device (Design B). Each KV
    /// page-slot is exactly one page = one arena block, so the block is `blocks()[0]`.
    pub fn slot_to_block_table(&self, arena: &Arena) -> Vec<u32> {
        self.slots
            .iter()
            .map(|slot| match slot.and_then(PageSlot::object) {
                Some(obj) => arena
                    .blocks(obj)
                    .ok()
                    .and_then(|b| b.first().copied())
                    .unwrap_or(SLOT_UNMAPPED),
                None => SLOT_UNMAPPED,
            })
            .collect()
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
            if self.slots[iu].is_none() {
                return Err(WorkingSetError::FreedSlot { index: i });
            }
            seen[iu] = true;
        }
        Ok(())
    }

    /// Open a fresh per-forward write-transaction (thrust-2 S4). Each prepared
    /// forward opens one; its repointed slots are logged under this id so
    /// commit/abort are isolated from any other outstanding forward.
    pub fn begin_write_txn(&mut self) -> WriteTxnId {
        self.next_write_txn += 1;
        WriteTxnId(self.next_write_txn)
    }

    /// Prepare a single output slot for writing, inside echo's forward txn:
    /// - reserved slot (`None`) → allocate a fresh KV page (`txn_alloc`); no copy.
    /// - shared materialised slot (`rc > 1`) → `txn_cow` a private copy and repoint;
    ///   the returned [`MovePlan`] is the d2d the caller must issue.
    /// - uniquely-owned materialised slot (`rc == 1`) → write in place; no copy.
    ///
    /// The slot is repointed to the post-write object and an abort-revert is
    /// recorded under `write_txn` (use [`commit_writes`]/[`abort_writes`] with the
    /// same id). Returns the post-write `ObjectId` and an optional copy plan.
    /// Rejects a slot already carrying a live uncommitted write from a *different*
    /// forward (`OverlappingWrite`) — the S4 disjoint-slot guarantee.
    pub fn cow_write_slot(
        &mut self,
        write_txn: WriteTxnId,
        idx: u32,
        txn: &mut ArenaTxn,
        arena: &mut Arena,
    ) -> Result<(ObjectId, Option<MovePlan>)> {
        let size = self.slots.len() as u32;
        if idx >= size {
            return Err(WorkingSetError::IndexOutOfRange { index: idx, size });
        }
        let i = idx as usize;
        // A slot already carrying a live write from an EARLIER forward is an
        // overwrite-after dependency, not a race: the run-ahead decode chains
        // `t` → `t+1` appends into the same KV page, and the scheduler fires `t`
        // before `t+1` (`would_depend_on_batch`). We allow it — ownership of the
        // live-write marker transfers to this (later) txn so its commit/abort
        // governs the slot's final value; the disjoint-slot isolation (the S4
        // guarantee) is unaffected. (Full dependency-ordered revert for the rare
        // both-abort-same-slot chain is a follow-up; the shipped run-ahead path
        // commits both.)
        match self.slots[i] {
            None => Err(WorkingSetError::FreedSlot { index: idx }),
            Some(PageSlot::Reserved) => {
                // Lazily materialise a fresh page for this reserved slot.
                let handle = arena.txn_alloc(txn, ArenaKind::KvPage, 1)?;
                arena.txn_mark_write(txn, handle.object_id)?;
                self.record_pending(write_txn, i, Some(PageSlot::Reserved));
                self.slots[i] = Some(PageSlot::Page(handle.object_id));
                Ok((handle.object_id, None))
            }
            Some(PageSlot::Page(orig)) => match arena.txn_cow(txn, orig)? {
                CowPlan::InPlace { handle } => {
                    if std::env::var("PTIR_COW_TRACE").is_ok() {
                        eprintln!(
                            "[COW] slot {idx}: Page({orig:?}) -> InPlace (persisted, retained)"
                        );
                    }
                    self.record_pending(write_txn, i, Some(PageSlot::Page(orig)));
                    Ok((handle.object_id, None))
                }
                CowPlan::Copy { handle, from, to } => {
                    if std::env::var("PTIR_COW_TRACE").is_ok() {
                        eprintln!(
                            "[COW] slot {idx}: Page({orig:?}) -> Copy (rc>1, freed original)"
                        );
                    }
                    self.record_pending(write_txn, i, Some(PageSlot::Page(orig)));
                    self.slots[i] = Some(PageSlot::Page(handle.object_id));
                    Ok((handle.object_id, Some(MovePlan { from, to })))
                }
            },
        }
    }

    /// Prepare slot `idx` for an IN-PLACE SHARED write — the beam-freeze heir's
    /// continuation onto a committed tail page the frozen sibling also references
    /// (overview §6.2 / G2). Unlike [`cow_write_slot`], this does **NOT** CoW even
    /// when the page is shared (`rc > 1`), does NOT repoint the slot, and does NOT
    /// `decref`: the shared physical page is preserved (rc unchanged) so the M4
    /// grace period + arena recycle keep it live while ANY beam references it, and
    /// the driver writes it in place at `(w_slot, w_off)`. Isolation is the per-
    /// beam `kvm` MASK (the frozen fork masks the heir's freshly-written cell),
    /// not a private copy — the freeze model's mask-based isolation, vs
    /// `cow_write_slot`'s copy-based isolation. The write is logged under
    /// `write_txn` for the S4 lifecycle (commit drops the marker; abort is a
    /// slot-mapping no-op — the mapping never changed — the stale in-place cell is
    /// masked/overwritten, never read as valid). Returns the shared page's
    /// [`ObjectId`] (for the host-replay path / validation; the device path
    /// resolves `w_slot`→BlockId itself). No arena/txn needed: no alloc, no CoW,
    /// and a partial tail page never seals (W7), so it is never a write target.
    ///
    /// SAFETY PRECONDITION (caller): the page's sharers must be EXACTLY the beam
    /// group (all applying the mask). Guaranteed for the active tail page —
    /// partial ⇒ never CAS-sealed (`KvCas::seal` requires `rc==1`+full), and it
    /// was uniquely owned by the parent's decode before `fork`, so its post-fork
    /// sharers are only the beam group. Only the active tail is written in place;
    /// read-only prefix history (possibly externally shared) is never touched.
    pub fn write_slot_shared_inplace(
        &mut self,
        write_txn: WriteTxnId,
        idx: u32,
    ) -> Result<ObjectId> {
        let size = self.slots.len() as u32;
        if idx >= size {
            return Err(WorkingSetError::IndexOutOfRange { index: idx, size });
        }
        let i = idx as usize;
        match self.slots[i] {
            None => Err(WorkingSetError::FreedSlot { index: idx }),
            // A shared in-place append requires a materialised page — a reserved
            // (never-written) slot has no shared history to continue.
            Some(PageSlot::Reserved) => Err(WorkingSetError::UnwrittenPage { index: idx }),
            Some(PageSlot::Page(orig)) => {
                // Mapping unchanged (no CoW / no repoint / no decref ⇒ rc>1
                // preserved). Log for the S4 txn so commit/abort/overlap track it
                // exactly like a CoW write; abort restores `orig` (a no-op).
                self.record_pending(write_txn, i, Some(PageSlot::Page(orig)));
                Ok(orig)
            }
        }
    }

    /// Log one slot's pre-write value under `write_txn` and mark it that txn's
    /// live write target.
    fn record_pending(&mut self, write_txn: WriteTxnId, i: usize, prev: Option<PageSlot>) {
        self.pending.entry(write_txn).or_default().push((i, prev));
        self.live_write_slots.insert(i, write_txn);
    }

    /// Publish `write_txn`'s in-flight write: the repointed slots are now the live
    /// mapping (called after `arena.txn_commit`). Drops only this txn's revert log
    /// + live-write markers, leaving any other outstanding forward untouched.
    pub fn commit_writes(&mut self, write_txn: WriteTxnId) {
        if let Some(entries) = self.pending.remove(&write_txn) {
            for (i, _) in &entries {
                if self.live_write_slots.get(i) == Some(&write_txn) {
                    self.live_write_slots.remove(i);
                }
            }
        }
    }

    /// Revert `write_txn`'s in-flight write: restore every slot it repointed to
    /// its prior value (called on `arena.txn_abort`), LIFO. Only this txn's slots
    /// are reverted — a concurrently-prepared forward's writes are untouched.
    pub fn abort_writes(&mut self, write_txn: WriteTxnId) {
        if let Some(entries) = self.pending.remove(&write_txn) {
            for (i, prev) in entries.into_iter().rev() {
                self.slots[i] = prev;
                if self.live_write_slots.get(&i) == Some(&write_txn) {
                    self.live_write_slots.remove(&i);
                }
            }
        }
    }

    /// Revert EVERY outstanding forward's in-flight write (all txns). Used at
    /// teardown ([`destroy`](Self::destroy)) where no forward will finalize.
    pub fn abort_all_writes(&mut self) {
        let txns: Vec<WriteTxnId> = self.pending.keys().copied().collect();
        for t in txns {
            self.abort_writes(t);
        }
    }

    // =====================================================================
    // Preempt/restore state-save (Task-B, alpha) — arena-side wrappers over
    // offload/release with the working-set safety guards. In Gim directive:
    // KV contention is preempt/restore, not admission. On an alloc-fail the
    // orchestrator (guru) FCFS-picks a victim working set and calls
    // `suspend_pages_warm`; on free it restores. These wrappers own the STATE-
    // SAVE CORRECTNESS (grace/refcount/txn), not the scheduling policy (queues /
    // victim pick / thrash guard live in guru's orchestrator). The trigger/seam
    // shape adapts to guru's `when_allocated` interface.
    // =====================================================================

    /// Classify this set for suspension (pure). Splits materialised pages into
    /// uniquely-owned (`rc==1`, stashable) vs shared (`rc>1`, ref-release-only),
    /// and the freed-now vs pin-deferred page accounting the orchestrator's victim
    /// loop needs (keep-evicting vs wait-for-pin-release). A page an in-flight
    /// forward arena-PINS is counted as pin-deferred (`freed_on_grace`), not
    /// freed_now, and left mapped. Reserved (unmaterialised) slots hold no page
    /// and are skipped. Does not consider in-flight write-txns — the caller aborts
    /// those via [`suspend_pages_warm`] before stashing.
    pub fn classify_for_suspend(&self, arena: &Arena) -> SuspendClass {
        let mut class = SuspendClass::default();
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(id) = slot.and_then(PageSlot::object) {
                match arena.refcount(id) {
                    Ok(1) => {
                        // A page an in-flight forward arena-PINS (run-ahead
                        // co-batch: a sibling forward of this same set is still in
                        // flight) must NOT be stashed — the pinned-check defers its
                        // free to that forward's finalize. Skip it here (leave the
                        // slot mapped) and count it as pin-deferred, not freed_now,
                        // so the orchestrator doesn't over-credit blocks this
                        // suspend cannot actually free right now.
                        if arena.is_pinned(id).unwrap_or(false) {
                            class.freed_on_grace += 1;
                            continue;
                        }
                        class.owned.push((i as u32, id));
                        class.freed_now += 1;
                    }
                    Ok(_) => class.shared.push((i as u32, id)),
                    // A dangling object id is a bug elsewhere; skip it here
                    // rather than trap during contention handling (W2/W3).
                    Err(_) => {}
                }
            }
        }
        class
    }

    /// Whether this set holds pages it could free NOW via [`suspend_pages_warm`]
    /// (uniquely-owned, unpinned, not grace-deferred). Drives the v2
    /// `SelfSuspendFirst` gate — a blocked requester only self-suspends when it
    /// actually has something to yield (else it parks as before).
    pub fn has_reclaimable_pages(&self, arena: &Arena) -> bool {
        self.classify_for_suspend(arena).freed_now > 0
    }

    /// **Phase 1** of warm-suspend: classify the set + STAGE the D2H offload of
    /// its uniquely-owned pages — allocate the CPU stash destinations and build
    /// the [`SuspendPlan`], but DO NOT free any GPU block or repoint any slot yet.
    /// The pages stay owned + GPU-resident, so the caller can safely issue every
    /// `copy_d2h` BEFORE [`commit_suspend`](Self::commit_suspend) frees the GPU
    /// blocks. This closes the **stash-free-before-copy race**: freeing a GPU
    /// block before its D2H copy lets a starving requester grab + overwrite the
    /// page mid-copy, stashing corrupted KV. SAFETY (state-save correctness):
    /// - **Txn guard:** aborts any in-flight write-txn first, so the committed
    ///   mapping (not a half-written page) is what gets staged.
    /// - **Pinned/shared:** an in-flight forward's pages are arena-PINNED and are
    ///   pin-deferred by `classify_for_suspend` (skipped, left mapped, counted in
    ///   `freed_on_grace`); shared (`rc>1`) pages are ref-released at commit
    ///   (never stashed).
    /// A page whose CPU-stash alloc fails is recorded in `plan.cold` (cold-dropped
    /// at commit; restore replays). Finish with [`commit_suspend`].
    pub fn stash_pages_warm(&mut self, arena: &mut Arena) -> SuspendPlan {
        // NOTE: we deliberately do NOT `abort_all_writes()` here. Under run-ahead a
        // PRIOR forward's write can be legitimately in flight (fired, not yet
        // finalized) when a co-resident lane self-suspends; aborting it would
        // revert that forward's output slot to `Reserved`, so its harvest (or a
        // later `resolve_read`) trips `UnwrittenPage` — "output: slot N has no
        // written page" (the carrier+preempt bug). An in-flight forward's pages
        // (reads AND write targets) are arena-PINNED (execute_impl txn_pin), and
        // `classify_for_suspend` skips pinned pages, so they are never staged — the
        // pinned-skip now supersedes the abort as the in-flight-page guard, WITHOUT
        // disturbing the slot mapping. (Teardown still uses `abort_all_writes` via
        // `destroy`, where no forward will finalize.)

        let class = self.classify_for_suspend(arena);
        let mut plan = SuspendPlan {
            stash: Vec::with_capacity(class.owned.len()),
            released_shared: Vec::with_capacity(class.shared.len()),
            cold: Vec::new(),
            freed_now: 0, // set at commit
            // Pages an in-flight forward pins (run-ahead co-batch) are grace-
            // deferred (classify), NOT stashed — the pinned-check frees them at
            // that forward's finalize. Carry the count so the orchestrator waits
            // for the grace free rather than over-crediting.
            freed_on_grace: class.freed_on_grace,
        };

        // Stage each owned page's offload: allocate its CPU dest, keep the GPU
        // block HELD + resident (no free, no repoint, no slot change yet). CPU
        // exhaustion → the page stays RESIDENT (not yielded): drop-to-replay is
        // NOT wired (v2 restore is warm-only), so a cold drop leaves the slot
        // Reserved with no content and the next written-slot read fails with
        // "slot N has no written page" (the C6+carrier BAR-2 error). An
        // un-stashable page is simply not freed — correctness over yield.
        for (slot, id) in class.owned {
            match arena.offload_stage(id) {
                Ok(mv) => plan.stash.push((slot, id, mv)),
                Err(_) => {}
            }
        }
        // Shared (rc>1, CAS-deduped) pages KEEP their ref — slots stay mapped.
        // Releasing them assumed "another holder keeps the page alive +
        // restore re-shares" — but restore has NO re-share path, and under
        // FLEET-WIDE preemption every holder releases → rc→0 → the shared
        // page (the deduped common prefix, or a sealed full page) is FREED →
        // every restored lane replays degenerately from token 1 / trips
        // "no written page" on later cycles. Shared releases never counted
        // toward freed_now anyway (only a last-ref release frees a block —
        // exactly the loss case), so keeping them costs no reclaim credit.
        let _ = class.shared;
        plan
    }

    /// **Phase 3** of warm-suspend: after the caller has issued every `copy_d2h`,
    /// free the GPU blocks + repoint the stashed pages to their CPU stash,
    /// ref-release the shared pages, cold-drop the CPU-exhausted ones, and set
    /// every affected slot `Reserved` (materialise-on-restore; slot ids DO NOT
    /// renumber, W1). Returns `freed_now` (GPU blocks freed).
    pub fn commit_suspend(
        &mut self,
        plan: &mut SuspendPlan,
        arena: &mut Arena,
        cas: &mut KvCas,
    ) -> u32 {
        let mut freed_now = 0u32;
        for (slot, id, mv) in &plan.stash {
            // GPU→CPU now safe (the copy is done): free the GPU block, repoint the
            // object to the staged CPU block.
            let _ = arena.offload_commit(*id, &mv.to);
            self.slots[*slot as usize] = Some(PageSlot::Reserved);
            freed_now += 1;
        }
        // `plan.cold` is populated only once drop-to-replay is wired into the
        // restore path; until then stage keeps CPU-exhausted pages resident and
        // this loop must be dead (a cold drop without replay loses written KV).
        debug_assert!(plan.cold.is_empty(), "cold drop without a replay path");
        for (slot, id) in &plan.cold {
            cas.release(arena, *id);
            self.slots[*slot as usize] = Some(PageSlot::Reserved);
            freed_now += 1;
        }
        // `released_shared` is populated only once restore has a real re-share
        // path; until then suspend keeps shared refs (slots stay mapped) —
        // a release here under fleet-wide preemption frees the deduped page
        // when the LAST holder suspends, and restore cannot recover it.
        debug_assert!(
            plan.released_shared.is_empty(),
            "shared release without a restore re-share path"
        );
        for (slot, id) in &plan.released_shared {
            cas.release(arena, *id);
            self.slots[*slot as usize] = Some(PageSlot::Reserved);
        }
        plan.freed_now = freed_now;
        freed_now
    }

    /// Convenience shim (host tests / non-concurrent callers): [`stash_pages_warm`]
    /// + [`commit_suspend`] with NO intervening D2H copy. The REAL concurrent path
    /// (`self_suspend_park_restore`) MUST use the split so the copy runs while the
    /// GPU blocks are still held — the shim skips the copy and is only safe where
    /// no concurrent allocator can reuse the freed blocks (e.g. unit tests).
    pub fn suspend_pages_warm(&mut self, arena: &mut Arena, cas: &mut KvCas) -> SuspendPlan {
        let mut plan = self.stash_pages_warm(arena);
        self.commit_suspend(&mut plan, arena, cas);
        plan
    }

    /// Warm-restore: re-materialise this set's stashed pages (H2D from the CPU
    /// arena) into fresh GPU blocks, reversing [`suspend_pages_warm`]. Returns
    /// the `(slot, MovePlan)` H2D copies the caller issues to the driver; the
    /// arena has repointed the objects back to `Residency::Gpu`. Restore alloc
    /// can itself contend — the caller must have admitted this set first
    /// (the orchestrator's `can_restore` / thrash guard), so a stash H2D that
    /// hits `OutOfBlocks` surfaces the error (never silently drops).
    ///
    /// NOTE: this handles the CPU-warm case (pages the set stashed to CPU). The
    /// cold/replay case (pages dropped on suspend, or a fresh empty set restored
    /// by replaying committed lineage) is the orchestrator's replay-pass domain,
    /// coordinated with the rs.rs fold for rs_cache — a follow-up once guru's
    /// replay-source ownership is settled.
    pub fn restore_pages_warm(
        &mut self,
        arena: &mut Arena,
        plan: &SuspendPlan,
    ) -> Result<Vec<(u32, MovePlan)>> {
        // All-or-nothing for ANY per-page failure, not just the pool pre-check.
        //
        // The pool pre-check below refuses (leaving the set fully stashed) unless
        // the GPU pool can take EVERY stashed page — a restore-race `OutOfBlocks`
        // (the orchestrator's `can_restore` admitted us but another lane grabbed
        // the blocks first) leaves the set exactly as suspended so the caller
        // re-reports the SAME `freed_now` and re-parks.
        //
        // But a per-page `arena.restore` also frees the CPU source + flips
        // residency→Gpu IN PLACE, so a naive `restore` loop that hit a mid-loop
        // NON-OOB error (e.g. `InvalidResidency`) would leave the earlier pages
        // committed (Gpu, CPU freed, slot mapped) yet UNPOPULATED — the caller
        // got `Err` before issuing their H2D copies. Those mapped-but-unpopulated
        // slots read silent garbage (a mapped slot never trips `UnwrittenPage`)
        // and can't be recovered on a re-park retry (their CPU source is gone).
        //
        // So we STAGE every page first (`restore_stage` allocs the GPU dest but
        // leaves the object Cpu, still owning its CPU blocks); on ANY staging
        // error we `restore_abort` the pages staged so far and return — nothing
        // was committed, the set stays cleanly stashed. Only once EVERY page is
        // staged do we `restore_commit` (free CPU + flip residency + map slot).
        // The whole call runs under the arena lock, so stage→commit is atomic.
        let need = plan.stash.len() as u32;
        let available = arena.available(ArenaKind::KvPage);
        if available < need {
            return Err(WorkingSetError::OutOfBlocks {
                kind: ArenaKind::KvPage,
                requested: need,
                available,
            });
        }
        // STAGE: allocate every GPU dest up front; roll back all on any failure.
        let mut staged: Vec<(u32, ObjectId, MovePlan)> = Vec::with_capacity(plan.stash.len());
        for &(slot, id, _) in &plan.stash {
            match arena.restore_stage(id) {
                Ok(mv) => staged.push((slot, id, mv)),
                Err(e) => {
                    for (_, sid, smv) in &staged {
                        // Best-effort rollback; the object was never mutated.
                        let _ = arena.restore_abort(*sid, &smv.to);
                    }
                    return Err(e.into());
                }
            }
        }
        // COMMIT: every stage landed — free CPU, flip residency, map the slot.
        let mut moves = Vec::with_capacity(staged.len());
        for (slot, id, mv) in staged {
            arena.restore_commit(id, &mv.to)?;
            self.slots[slot as usize] = Some(PageSlot::Page(id));
            moves.push((slot, mv));
        }
        Ok(moves)
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
        let id = match self.slots[idx as usize] {
            None => return Err(WorkingSetError::FreedSlot { index: idx }),
            Some(PageSlot::Reserved) => return Err(WorkingSetError::UnwrittenPage { index: idx }),
            Some(PageSlot::Page(id)) => id,
        };
        let final_id = cas.seal(arena, id, hash)?;
        self.slots[idx as usize] = Some(PageSlot::Page(final_id));
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
    use crate::working_set::page_hash::compute_page_hashes;
    use pie_grammar::brle::RunMask;

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
        let wtx = ws.begin_write_txn();
        ws.cow_write_slot(wtx, idx, &mut txn, a).unwrap();
        a.txn_commit(txn).unwrap();
        ws.commit_writes(wtx);
        if let Some(h) = seal_hash {
            ws.seal(idx, h, a, cas).unwrap();
        }
    }

    /// The generation delta a single `alloc`/`free`/`append` contributes: 0 under
    /// slot-id semantics (W8 — only reorder/compact bump), 1 in legacy builds.
    const STRUCT_BUMP: u32 = if SLOT_IDS { 0 } else { 1 };

    fn slot(ws: &KvWorkingSet, i: usize) -> Option<ObjectId> {
        ws.slots[i].and_then(PageSlot::object)
    }

    fn full_hash(toks: &[u32], prev: PageHash) -> PageHash {
        let positions: Vec<u32> = (0..toks.len() as u32).collect();
        let masks: Vec<RunMask> = (0..toks.len()).map(|i| RunMask::all_true(i + 1)).collect();
        *compute_page_hashes(PAGE as usize, toks, &positions, &masks, prev, None)
            .last()
            .unwrap()
    }

    // CAS-clean deferred free: a sealed (CAS-indexed) page pinned by an in-flight
    // forward, then freed via free_slots while pinned. `cas.release` cleans the
    // hash->canonical reverse-map at refcount==1 BEFORE the decref that the
    // pinned-check defers — so the deferred free at unpin leaves NO dangling CAS
    // entry (which `KvCas::seal` would otherwise `incref` blindly on a recycled id).
    #[test]
    fn deferred_free_of_pinned_sealed_page_is_cas_clean() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(1).unwrap();
        let h = full_hash(&[1, 2, 3, 4], 0);
        write_slot(&mut ws, &mut a, &mut cas, 0, Some(h)); // seal → CAS entry
        let id = slot(&ws, 0).unwrap();
        assert_eq!(cas.len(), 1, "the sealed page is CAS-indexed");

        // A forward pins the sealed page (reads it); it is then freed while pinned.
        a.pin(id).unwrap();
        ws.free_slots(&[0], &mut a, &mut cas).unwrap();
        // cas.release cleaned the index at rc==1, before the pin-deferred decref.
        assert_eq!(
            cas.len(),
            0,
            "CAS reverse-map cleaned before the deferred free"
        );
        assert_eq!(a.refcount(id).unwrap(), 0, "rc hit 0");
        assert!(
            a.is_pinned(id).unwrap(),
            "physical free deferred by the pin"
        );
        assert!(
            a.residency(id).is_ok(),
            "object still resolvable (a live zombie)"
        );

        // The forward finalizes: the last unpin frees the object. CAS stays clean.
        a.unpin(id).unwrap();
        assert!(
            matches!(
                a.residency(id),
                Err(crate::arena::ArenaError::UnknownObject(_))
            ),
            "freed at the last unpin"
        );
        assert_eq!(
            cas.len(),
            0,
            "no dangling CAS entry after the deferred free"
        );
    }

    // 1. alloc — reserves empty slots lazily (no arena pages); generation policy.
    #[test]
    fn alloc_reserves_lazily_and_bumps_generation() {
        let mut a = arena(16);
        let mut ws = KvWorkingSet::new(PAGE, 0);
        assert_eq!(ws.size(), 0);
        assert_eq!(ws.page_size(), PAGE);

        let r = ws.alloc(3).unwrap();
        assert_eq!(r, PageRange { start: 0, len: 3 });
        assert_eq!(ws.size(), 3);
        assert_eq!(ws.generation(), STRUCT_BUMP); // W8: no bump under slot-ids
        assert_eq!(slot(&ws, 0), None); // reserved, not materialised
        assert_eq!(a.live_objects(), 0); // lazy: no physical pages yet

        assert_eq!(ws.alloc(0).unwrap(), PageRange { start: 3, len: 0 });
        assert_eq!(ws.generation(), STRUCT_BUMP); // no-op, no bump
    }
    // 2b. free (slot-id tombstone) — stable ids; interior hole recycled ascending;
    // trailing free truncates; double-free / out-of-range rejected; no gen bump.
    #[test]
    fn free_tombstones_keeps_stable_ids() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(5).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 1, None);
        write_slot(&mut ws, &mut a, &mut cas, 4, None);
        let id1 = slot(&ws, 1);
        assert_eq!(a.live_objects(), 2);
        let gen0 = ws.generation();

        // Interior free of slot 1 → tombstone (survivors keep ids); slot 4 stays
        // slot 4 (NOT renumbered to 3, unlike compaction).
        ws.free(&[1], &mut a, &mut cas).unwrap();
        assert_eq!(ws.size(), 4); // live count dropped by 1
        assert_eq!(slot(&ws, 0), None); // slot 0 unchanged (reserved)
        assert_eq!(slot(&ws, 4), slot(&ws, 4)); // slot 4 id unchanged
        assert!(ws.slots[4].is_some()); // slot 4 still live, same position
        assert!(ws.slots[1].is_none()); // slot 1 tombstoned
        assert_eq!(a.live_objects(), 1); // slot-1's page released
        assert_eq!(ws.generation(), gen0); // W8: free does not bump

        // Recycle: alloc_slots pops the lowest freed id (1) before growing.
        assert_eq!(ws.alloc_slots(1).unwrap(), vec![1]);
        assert_eq!(ws.size(), 5);
        assert!(matches!(ws.slots[1], Some(PageSlot::Reserved)));
        let _ = id1;

        // Double-free and out-of-range are errors, never traps.
        ws.free(&[1], &mut a, &mut cas).unwrap();
        assert_eq!(
            ws.free(&[1], &mut a, &mut cas),
            Err(WorkingSetError::DoubleFree { index: 1 })
        );
        assert!(matches!(
            ws.free(&[99], &mut a, &mut cas),
            Err(WorkingSetError::IndexOutOfRange { .. })
        ));

        // Trailing free truncates: freeing the last live slots shrinks the array
        // with no interior hole (SDK dense tail-trim), free list stays empty.
        let mut tail = KvWorkingSet::new(PAGE, 0);
        tail.alloc(3).unwrap();
        tail.free(&[1, 2], &mut a, &mut cas).unwrap();
        assert_eq!(tail.slots.len(), 1); // truncated to the live prefix
        assert!(tail.free_list.is_empty());
        assert_eq!(tail.size(), 1);
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
        assert_eq!(dst.generation(), gen0 + STRUCT_BUMP); // W8: no bump under slot-ids
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

    // 7b. seal refuses a STALE dangling canonical (freed via a non-`release` path,
    //     e.g. on_commit_decref) — validates liveness + ownership before incref.
    #[test]
    fn seal_skips_a_stale_dangling_canonical() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let p1 = a.alloc(ArenaKind::KvPage, 1).unwrap().object_id;
        let p2 = a.alloc(ArenaKind::KvPage, 1).unwrap().object_id;
        assert_ne!(p1, p2);
        let h: PageHash = 0xABCD_1234;

        assert_eq!(cas.seal(&mut a, p1, h).unwrap(), p1); // index[h] = p1
        a.decref(p1).unwrap(); // raw decref (as on_commit_decref) → index dangles
        assert!(a.refcount(p1).is_err(), "p1 freed, index entry stale");

        let got = cas.seal(&mut a, p2, h).unwrap();
        assert_eq!(
            got, p2,
            "sealed onto the fresh page, not the dead canonical"
        );
        assert_eq!(a.refcount(p2).unwrap(), 1, "p2 not spuriously increffed");
    }

    // 7c. BAR-1 replay fix: seal must NOT dedup onto a canonical that was STASHED
    //     to CPU under preemption (it stays in the index — offload doesn't
    //     unindex it). Deduping onto a CPU-resident page → the sharer's forward
    //     reads freed GPU blocks (empty context) → degenerate replay.
    #[test]
    fn seal_does_not_dedup_onto_a_stashed_cpu_canonical() {
        let mut a = arena_with_cpu(16, 16);
        let mut cas = KvCas::new();
        let h = full_hash(&[7, 8, 9, 10], 0);

        // w1 seals its page, then STASHES it (offload → CPU); index[h] still → p1.
        let mut w1 = KvWorkingSet::new(PAGE, 0);
        w1.alloc(1).unwrap();
        write_slot(&mut w1, &mut a, &mut cas, 0, Some(h));
        let p1 = slot(&w1, 0).unwrap();
        assert_eq!(cas.len(), 1);
        let plan = w1.suspend_pages_warm(&mut a, &mut cas);
        assert_eq!(
            a.residency(p1).unwrap(),
            crate::arena::Residency::Cpu,
            "p1 stashed to CPU but still CAS-indexed"
        );

        // w2 seals the SAME content: must keep its OWN GPU page, NOT dedup onto p1.
        let mut w2 = KvWorkingSet::new(PAGE, 0);
        w2.alloc(1).unwrap();
        write_slot(&mut w2, &mut a, &mut cas, 0, Some(h));
        let p2 = slot(&w2, 0).unwrap();
        assert_ne!(
            p2, p1,
            "w2 did NOT dedup onto the stashed (CPU-resident) p1"
        );
        assert_eq!(
            a.residency(p2).unwrap(),
            crate::arena::Residency::Gpu,
            "w2's page is GPU-resident (usable by a forward)"
        );

        let _ = w1.restore_pages_warm(&mut a, &plan);
        w1.destroy(&mut a, &mut cas);
        w2.destroy(&mut a, &mut cas);
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

    // 9. stale-generation rejection. Uses `reorder` (bumps generation in both
    // modes, W8) as the concurrent structural mutation that invalidates a
    // captured generation — under slot-ids, alloc/free no longer bump.
    #[test]
    fn write_against_stale_generation_is_rejected() {
        let mut a = arena(16);
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(1).unwrap();
        let captured = ws.generation();
        ws.reorder(&[0]).unwrap(); // reorder bumps generation (W8)
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
        let wtx = view.begin_write_txn();
        let (_new, mv) = view.cow_write_slot(wtx, 0, &mut txn, &mut a).unwrap();
        assert!(mv.is_some()); // shared ⇒ a copy was staged
        assert_ne!(slot(&view, 0), base0); // repointed to the staged copy
        a.txn_abort(txn);
        view.abort_writes(wtx);

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

    // 14b. Design B: slot_to_block_table is DENSE over the slot domain, resolves
    //      materialised slots to their physical block, and sentinels reserved /
    //      tombstoned / unmaterialised slots (SLOT_UNMAPPED) — slot 0 is a real
    //      entry, interior tombstones keep their dense index.
    #[test]
    fn slot_to_block_table_dense_with_unmapped_sentinel() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(4).unwrap();
        // Materialise 0, 2, 3; leave 1 reserved.
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        write_slot(&mut ws, &mut a, &mut cas, 2, None);
        write_slot(&mut ws, &mut a, &mut cas, 3, None);

        let table = ws.slot_to_block_table(&a);
        assert_eq!(table.len(), 4, "dense over the whole slot domain");
        // Slot 0 (a VALID slot id) → its physical page-pool block.
        let obj0 = slot(&ws, 0).unwrap();
        assert_eq!(
            table[0],
            a.blocks(obj0).unwrap()[0],
            "slot 0 resolves, not special-cased"
        );
        assert_eq!(table[1], SLOT_UNMAPPED, "reserved → sentinel");
        assert_eq!(table[2], a.blocks(slot(&ws, 2).unwrap()).unwrap()[0]);
        assert_eq!(table[3], a.blocks(slot(&ws, 3).unwrap()).unwrap()[0]);

        // Tombstone an INTERIOR slot (2) → its dense index stays, entry sentinels.
        ws.free_slots(&[2], &mut a, &mut cas).unwrap();
        let table2 = ws.slot_to_block_table(&a);
        assert_eq!(table2.len(), 4, "interior tombstone keeps the dense index");
        assert_eq!(table2[2], SLOT_UNMAPPED, "tombstoned → sentinel");
        assert_eq!(
            table2[0],
            a.blocks(obj0).unwrap()[0],
            "unaffected slots stable"
        );

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

    // 18. M1 exit soak: randomized alloc/free/fork/append/slice with simulated
    // in-flight passes continuously pinning live ids. Every step re-checks the
    // stable-id invariants: no live id ever renumbers (a materialised slot keeps
    // its ObjectId until freed), `size()` is the live count, `free_list` is
    // exactly the tombstone set with no trailing hole, and commit round-trips.
    #[test]
    fn soak_stable_ids_under_churn() {
        let mut a = arena(16384);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);

        // Expected ObjectId of each live materialised slot id (renumber detector).
        let mut expected: HashMap<u32, ObjectId> = HashMap::new();

        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        for step in 0..1500u32 {
            let live_ids: Vec<u32> = (0..ws.slots.len() as u32)
                .filter(|&i| ws.slots[i as usize].is_some())
                .collect();

            // Bound growth: trim live ids when the set gets large.
            if live_ids.len() > 128 {
                let victims: Vec<u32> = live_ids.iter().copied().take(64).collect();
                ws.free_slots(&victims, &mut a, &mut cas).unwrap();
                for id in &victims {
                    expected.remove(id);
                }
                continue;
            }

            match next() % 6 {
                0 => {
                    // Recycling allocation + materialise some of the ids.
                    let n = (next() % 4) as u32;
                    let ids = ws.alloc_slots(n).unwrap();
                    for id in ids {
                        // A recycled id must not point at stale data.
                        assert!(
                            expected.get(&id).is_none(),
                            "recycled id {id} still tracked"
                        );
                        if next() % 2 == 0 {
                            write_slot(&mut ws, &mut a, &mut cas, id, None);
                            expected.insert(id, slot(&ws, id as usize).unwrap());
                        }
                    }
                }
                1 => {
                    // Grow-only allocation (contiguous shim).
                    let n = (next() % 4) as u32;
                    ws.alloc(n).unwrap();
                }
                2 => {
                    // Free a random subset of live ids.
                    let free_ids: Vec<u32> = live_ids
                        .iter()
                        .copied()
                        .filter(|_| next() % 3 == 0)
                        .collect();
                    if !free_ids.is_empty() {
                        ws.free_slots(&free_ids, &mut a, &mut cas).unwrap();
                        for id in &free_ids {
                            expected.remove(id);
                        }
                    }
                }
                3 => {
                    // Fork + drop the child (exercises incref / destroy paths).
                    let mut child = ws.fork(&mut a).unwrap();
                    child.destroy(&mut a, &mut cas);
                }
                4 => {
                    // Append a small fresh set by reference.
                    let mut other = KvWorkingSet::new(PAGE, 0);
                    other.alloc(1 + (next() % 2) as u32).unwrap();
                    ws.append(&other, &mut a).unwrap();
                    other.destroy(&mut a, &mut cas);
                }
                _ => {
                    // Slice a random range and drop it.
                    if !ws.slots.is_empty() {
                        let len = ws.slots.len() as u32;
                        let s = (next() as u32) % len;
                        let l = (next() as u32) % (len - s + 1);
                        if let Ok(mut sub) = ws.slice(s, l, &mut a) {
                            sub.destroy(&mut a, &mut cas);
                        }
                    }
                }
            }

            // ---- Invariants ----
            let tomb: BTreeSet<u32> = ws
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| s.is_none())
                .map(|(i, _)| i as u32)
                .collect();
            assert_eq!(ws.free_list, tomb, "free_list drift at step {step}");
            assert!(
                !matches!(ws.slots.last(), Some(None)),
                "trailing tombstone at step {step}"
            );
            assert_eq!(
                ws.size() as usize,
                ws.slots.len() - tomb.len(),
                "size != live count at step {step}"
            );
            for (&id, &obj) in &expected {
                assert_eq!(
                    slot(&ws, id as usize),
                    Some(obj),
                    "slot {id} renumbered/aliased at step {step}"
                );
            }
        }

        ws.destroy(&mut a, &mut cas);
        assert_eq!(a.live_objects(), 0, "leaked pages after soak");
    }

    // 20. S4 (thrust-2): per-forward write-transaction ownership. Two prepared
    //     forwards against DISJOINT slots of one working set may be outstanding
    //     at once; aborting one reverts only its slot and cannot touch the other.
    #[test]
    fn per_forward_txn_isolates_disjoint_writes() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(2).unwrap();
        // Materialise + commit both slots so each holds a page object.
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        write_slot(&mut ws, &mut a, &mut cas, 1, None);
        let base0 = slot(&ws, 0);
        let base1 = slot(&ws, 1);
        // Fork a sharer so both pages are shared (rc=2) → writes CoW-copy and
        // repoint the slot (observable), rather than write in place.
        let mut sharer = ws.fork(&mut a).unwrap();

        // Forward A prepares a write on slot 0; forward B on slot 1 — both
        // outstanding at once (two separate arena txns + write-txn ids).
        let mut txn_a = a.txn_begin();
        let wtx_a = ws.begin_write_txn();
        let (_a_obj, a_mp) = ws.cow_write_slot(wtx_a, 0, &mut txn_a, &mut a).unwrap();
        assert!(a_mp.is_some(), "shared slot ⇒ CoW copy");
        assert_ne!(slot(&ws, 0), base0, "A repointed slot 0");

        let mut txn_b = a.txn_begin();
        let wtx_b = ws.begin_write_txn();
        let (_b_obj, b_mp) = ws.cow_write_slot(wtx_b, 1, &mut txn_b, &mut a).unwrap();
        assert!(b_mp.is_some());
        let b_repoint = slot(&ws, 1);
        assert_ne!(b_repoint, base1, "B repointed slot 1");

        // Abort A: slot 0 reverts to its base; slot 1 (B's, still in flight) is
        // untouched — the isolation the set-level log could not provide.
        a.txn_abort(txn_a);
        ws.abort_writes(wtx_a);
        assert_eq!(slot(&ws, 0), base0, "A's slot reverted");
        assert_eq!(
            slot(&ws, 1),
            b_repoint,
            "B's in-flight write must NOT be reverted by A's abort"
        );

        // Commit B: slot 1 publishes its repointed object; slot 0 stays reverted.
        a.txn_commit(txn_b).unwrap();
        ws.commit_writes(wtx_b);
        assert_eq!(slot(&ws, 0), base0);
        assert_eq!(slot(&ws, 1), b_repoint);

        ws.destroy(&mut a, &mut cas);
        sharer.destroy(&mut a, &mut cas);
    }

    // 21. S4: two outstanding forwards writing the SAME slot is an overwrite-after
    //     dependency (the run-ahead t → t+1 same-page append), ALLOWED — the later
    //     txn's commit governs the slot's final value; the earlier commit does not
    //     clobber it. The disjoint-slot isolation (test 20) is the S4 guarantee;
    //     overlapping is the run-ahead decode's legitimate append chain.
    #[test]
    fn overlapping_write_allowed_as_overwrite_after() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(1).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        // Share the page so each write CoW-copies (observable repoint).
        let mut sharer = ws.fork(&mut a).unwrap();

        // Forward t writes slot 0 (in flight).
        let mut txn_t = a.txn_begin();
        let wtx_t = ws.begin_write_txn();
        ws.cow_write_slot(wtx_t, 0, &mut txn_t, &mut a).unwrap();

        // Forward t+1 appends to the SAME slot 0 while t is in flight — allowed
        // (overwrite-after). t already CoW'd the page to unique ownership, so
        // t+1's append writes in place (no second copy).
        let mut txn_t1 = a.txn_begin();
        let wtx_t1 = ws.begin_write_txn();
        ws.cow_write_slot(wtx_t1, 0, &mut txn_t1, &mut a).unwrap();
        let t1_obj = slot(&ws, 0);

        // t commits first (run-ahead finalize order) — it must NOT clobber t+1's
        // in-flight repoint (ownership transferred to t+1).
        a.txn_commit(txn_t).unwrap();
        ws.commit_writes(wtx_t);
        assert_eq!(
            slot(&ws, 0),
            t1_obj,
            "t's commit leaves t+1's write in place"
        );

        // t+1 commits — publishes the latest (append-chain) value.
        a.txn_commit(txn_t1).unwrap();
        ws.commit_writes(wtx_t1);
        assert_eq!(slot(&ws, 0), t1_obj);

        ws.destroy(&mut a, &mut cas);
        sharer.destroy(&mut a, &mut cas);
    }

    // 21b. G2 beam-freeze: write_slot_shared_inplace on a SHARED (rc>1) page does
    //      NOT CoW — same ObjectId returned, mapping unchanged, rc PRESERVED (no
    //      decref) so the shared page survives for every beam. Contrast cow_write_slot
    //      (test #20) which forks + repoints. abort is a mapping no-op; commit drops
    //      the marker. This is the heir's mask-isolated in-place append primitive.
    #[test]
    fn shared_inplace_write_does_not_cow_or_decref() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(1).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        let base = slot(&ws, 0).unwrap();
        // Fork a sibling so the tail page is SHARED (rc=2) — the beam-group case
        // where cow_write_slot WOULD fork but the freeze needs in-place.
        let mut sibling = ws.fork(&mut a).unwrap();
        assert_eq!(a.refcount(base).unwrap(), 2, "shared by the beam group");

        // Heir's in-place shared append: no arena/txn, no CoW.
        let wtx = ws.begin_write_txn();
        let obj = ws.write_slot_shared_inplace(wtx, 0).unwrap();
        assert_eq!(obj, base, "returns the SHARED page (no fork)");
        assert_eq!(
            slot(&ws, 0),
            Some(base),
            "slot mapping UNCHANGED (no repoint)"
        );
        assert_eq!(
            a.refcount(base).unwrap(),
            2,
            "rc PRESERVED (no decref) — grace/recycle protect the shared page"
        );

        // Abort is a slot-mapping no-op (the mapping never changed); rc still 2.
        ws.abort_writes(wtx);
        assert_eq!(
            slot(&ws, 0),
            Some(base),
            "abort: mapping stays the shared page"
        );
        assert_eq!(a.refcount(base).unwrap(), 2, "abort does not decref");

        // Commit path: drops the live-write marker, mapping still the shared page.
        let wtx2 = ws.begin_write_txn();
        ws.write_slot_shared_inplace(wtx2, 0).unwrap();
        ws.commit_writes(wtx2);
        assert_eq!(slot(&ws, 0), Some(base));
        assert_eq!(a.refcount(base).unwrap(), 2);

        // A reserved (never-written) slot has no shared history to continue.
        ws.alloc(1).unwrap();
        let wtx3 = ws.begin_write_txn();
        assert!(matches!(
            ws.write_slot_shared_inplace(wtx3, 1),
            Err(WorkingSetError::UnwrittenPage { index: 1 })
        ));

        ws.destroy(&mut a, &mut cas);
        sibling.destroy(&mut a, &mut cas);
    }

    // ── Task-B preempt/restore state-save (alpha) ──────────────────────────

    fn arena_with_cpu(kv_pages: u32, cpu_blocks: u32) -> Arena {
        Arena::new(ArenaConfig {
            device: 0,
            block_size: PAGE,
            kv_pages,
            rs_blocks: 0,
            scratch_blocks: 0,
            cpu_blocks,
        })
    }

    // 21c. classify_for_suspend splits owned (rc==1) vs shared (rc>1) pages and
    //      reports the reclaimable count — pure, no mutation.
    #[test]
    fn classify_for_suspend_splits_owned_and_shared() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(3).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None); // owned
        write_slot(&mut ws, &mut a, &mut cas, 1, None); // shared below
        // slot 2 stays Reserved (unmaterialised) → skipped.
        let sibling = ws.fork(&mut a).unwrap(); // shares slots 0,1 → rc=2

        let class = ws.classify_for_suspend(&a);
        // Both materialised slots are now shared (fork shared BOTH).
        assert_eq!(class.owned.len(), 0);
        assert_eq!(class.shared.len(), 2);
        assert_eq!(class.freed_now, 0, "shared pages free nothing here");
        assert_eq!(class.freed_on_grace, 0);

        drop(sibling); // sibling leaks its refs (test-only); destroy ws below.
        ws.destroy(&mut a, &mut cas);
    }

    // Multi-preempt KV-loss fix (ab981d32): a shared (rc>1, CAS-deduped/forked)
    // page must NOT be released on suspend. Under fleet-wide preemption every
    // holder releasing would drive rc→0 and FREE the deduped page, and restore
    // has NO re-share path → degenerate replay / no-written-page across cycles.
    // Suspend keeps the ref (slot stays mapped + resident); restore is a no-op
    // for it; KV is preserved across the suspend/restore cycle.
    #[test]
    fn suspend_keeps_shared_pages_mapped_across_restore() {
        let mut a = arena_with_cpu(16, 16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(1).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        let shared_id = slot(&ws, 0).unwrap();
        let sibling = ws.fork(&mut a).unwrap(); // shares slot 0 → rc=2
        assert_eq!(a.refcount(shared_id).unwrap(), 2, "page is shared");

        // Suspend: the shared page stays mapped + resident, NEVER released.
        let plan = ws.suspend_pages_warm(&mut a, &mut cas);
        assert!(
            plan.released_shared.is_empty(),
            "shared pages are never released"
        );
        assert_eq!(plan.freed_now, 0, "a shared page frees no block");
        assert_eq!(
            slot(&ws, 0),
            Some(shared_id),
            "shared page STILL mapped after suspend"
        );
        assert_eq!(
            a.residency(shared_id).unwrap(),
            crate::arena::Residency::Gpu,
            "shared page still GPU-resident (not stashed)"
        );
        assert_eq!(
            a.refcount(shared_id).unwrap(),
            2,
            "our ref kept (rc unchanged)"
        );

        // Restore is a no-op for the shared page; KV intact across the cycle.
        let _ = ws.restore_pages_warm(&mut a, &plan).unwrap();
        assert_eq!(
            slot(&ws, 0),
            Some(shared_id),
            "KV preserved across suspend/restore"
        );
        assert_eq!(
            a.refcount(shared_id).unwrap(),
            2,
            "still shared, still ours"
        );

        drop(sibling);
        ws.destroy(&mut a, &mut cas);
    }

    #[test]
    fn classify_skips_arena_pinned_page() {
        // Run-ahead co-batch: a sibling forward of THIS set is still in flight,
        // arena-pinning one of its pages. `classify_for_suspend` must NOT offer
        // that page as stashable (the pinned-check defers its free to the sibling
        // forward's finalize) — it is grace-deferred, never counted as freed_now.
        // (CPU pool present: this test exercises the WARM stash path.)
        let mut a = arena_with_cpu(16, 16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(2).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None); // owned, will be pinned
        write_slot(&mut ws, &mut a, &mut cas, 1, None); // owned, free-now
        let pinned_id = slot(&ws, 0).unwrap();
        a.pin(pinned_id).unwrap(); // an in-flight forward holds slot 0's page

        let class = ws.classify_for_suspend(&a);
        assert_eq!(class.owned.len(), 1, "only the unpinned page is stashable");
        assert_eq!(
            class.owned[0].0, 1,
            "slot 1 (unpinned) is the stashable one"
        );
        assert_eq!(class.freed_now, 1, "only the unpinned page frees now");
        assert_eq!(class.freed_on_grace, 1, "the pinned page is grace-deferred");

        // suspend_pages_warm must respect it: the pinned page stays mapped, and
        // the plan reports the deferred count (not over-credited as freed).
        let plan = ws.suspend_pages_warm(&mut a, &mut cas);
        assert_eq!(plan.freed_now, 1, "suspend frees only the unpinned page");
        assert_eq!(
            plan.freed_on_grace, 1,
            "pinned page carried as grace-deferred"
        );
        assert_eq!(slot(&ws, 0), Some(pinned_id), "pinned page still mapped");
        assert!(slot(&ws, 1).is_none(), "unpinned page stashed (Reserved)");

        a.unpin(pinned_id).unwrap();
        ws.destroy(&mut a, &mut cas);
    }

    /// CPU-stash exhaustion keeps pages RESIDENT — never a silent drop.
    /// Drop-to-replay is not wired (v2 restore is warm-only): a cold drop
    /// loses written KV, surfacing as "slot N has no written page" (BAR-2
    /// C6+carrier) or as silent context truncation (the restore-divergence
    /// replay). With NO CPU pool the suspend must yield nothing and leave
    /// every page mapped.
    #[test]
    fn cpu_exhausted_stash_keeps_pages_resident() {
        let mut a = arena(16); // cpu_blocks: 0 — no stash headroom at all
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(2).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        write_slot(&mut ws, &mut a, &mut cas, 1, None);
        let (id0, id1) = (slot(&ws, 0).unwrap(), slot(&ws, 1).unwrap());

        let mut plan = ws.stash_pages_warm(&mut a);
        assert!(
            plan.stash.is_empty(),
            "nothing stashable without CPU headroom"
        );
        assert!(
            plan.cold.is_empty(),
            "cold is never populated (no replay path)"
        );
        let freed = ws.commit_suspend(&mut plan, &mut a, &mut cas);
        assert_eq!(freed, 0, "no blocks freed — pages kept resident");
        assert_eq!(slot(&ws, 0), Some(id0), "slot 0 still mapped");
        assert_eq!(slot(&ws, 1), Some(id1), "slot 1 still mapped");

        ws.destroy(&mut a, &mut cas);
    }

    // 21e. Warm suspend→restore round-trip: an owned page offloads to CPU (freeing
    //      a GPU block) and restores back, with the slot id preserved (W1).
    #[test]
    fn suspend_restore_warm_roundtrip_frees_and_restores() {
        let mut a = arena_with_cpu(2, 4);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(2).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        write_slot(&mut ws, &mut a, &mut cas, 1, None);
        let gpu_used_before = a.used(ArenaKind::KvPage);
        assert_eq!(gpu_used_before, 2, "both pages on GPU");
        let id0 = slot(&ws, 0).unwrap();

        // Suspend: owned pages D2H → CPU, GPU blocks freed.
        let plan = ws.suspend_pages_warm(&mut a, &mut cas);
        assert_eq!(plan.stash.len(), 2, "both owned pages stashed");
        assert_eq!(plan.freed_now, 2, "both blocks freed now (unpinned set)");
        assert_eq!(
            plan.freed_on_grace, 0,
            "nothing grace-held on an unpinned set"
        );
        assert_eq!(a.used(ArenaKind::KvPage), 0, "GPU blocks reclaimed");
        assert!(
            slot(&ws, 0).is_none(),
            "slot repointed to Reserved (off-GPU)"
        );

        // Restore: H2D back to fresh GPU blocks; slot ids preserved.
        let moves = ws.restore_pages_warm(&mut a, &plan).unwrap();
        assert_eq!(moves.len(), 2, "both pages restored");
        assert_eq!(a.used(ArenaKind::KvPage), 2, "GPU blocks re-acquired");
        assert_eq!(slot(&ws, 0), Some(id0), "same ObjectId restored to slot 0");

        ws.destroy(&mut a, &mut cas);
    }

    // 21d. stash-then-free: stash_pages_warm must NOT free the GPU blocks (so the
    //      D2H copy reads valid data); commit_suspend frees them AFTER the copy —
    //      the fix for guru's stash-free-before-copy race (a starving requester
    //      grabbing a freed block and overwriting the page mid-copy).
    #[test]
    fn stash_holds_gpu_resident_until_commit() {
        let mut a = arena_with_cpu(2, 4);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(2).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        write_slot(&mut ws, &mut a, &mut cas, 1, None);
        let id0 = slot(&ws, 0).unwrap();
        assert_eq!(a.used(ArenaKind::KvPage), 2, "both pages on GPU");

        // STAGE: CPU dests allocated, GPU blocks STILL held + resident + mapped.
        let mut plan = ws.stash_pages_warm(&mut a);
        assert_eq!(plan.stash.len(), 2);
        assert_eq!(plan.freed_now, 0, "nothing freed at stage yet");
        assert_eq!(
            a.used(ArenaKind::KvPage),
            2,
            "GPU NOT freed at stage (race fix)"
        );
        assert_eq!(
            a.residency(id0).unwrap(),
            crate::arena::Residency::Gpu,
            "still GPU-resident — the copy can read it"
        );
        assert_eq!(slot(&ws, 0), Some(id0), "slot still maps the resident page");
        assert_eq!(a.used(ArenaKind::CpuStash), 2, "CPU dests staged");

        // (caller issues copy_d2h here in the real path)

        // COMMIT: NOW the GPU blocks free + pages repoint to CPU + slots Reserved.
        let freed = ws.commit_suspend(&mut plan, &mut a, &mut cas);
        assert_eq!(freed, 2);
        assert_eq!(plan.freed_now, 2);
        assert_eq!(a.used(ArenaKind::KvPage), 0, "GPU freed at commit");
        assert_eq!(a.residency(id0).unwrap(), crate::arena::Residency::Cpu);
        assert!(slot(&ws, 0).is_none(), "slot Reserved");

        // Restore still round-trips (same ObjectId).
        let moves = ws.restore_pages_warm(&mut a, &plan).unwrap();
        assert_eq!(moves.len(), 2);
        assert_eq!(slot(&ws, 0), Some(id0));
        ws.destroy(&mut a, &mut cas);
    }

    // 21e. restore is all-or-nothing under a lost restore-race: if the pool cannot
    //      take every stashed page, it restores NONE (the set stays fully stashed)
    //      so the v2 prologue re-reports the same freed_now and re-parks.
    #[test]
    fn restore_pages_warm_is_all_or_nothing_on_contention() {
        let mut a = arena_with_cpu(2, 4); // GPU = 2 blocks
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(2).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        write_slot(&mut ws, &mut a, &mut cas, 1, None);
        let id0 = slot(&ws, 0).unwrap();
        let id1 = slot(&ws, 1).unwrap();

        let plan = ws.suspend_pages_warm(&mut a, &mut cas);
        assert_eq!(plan.stash.len(), 2);
        assert_eq!(a.used(ArenaKind::KvPage), 0, "both pages off-GPU");

        // A lost restore race: another lane grabs one of the two freed GPU blocks.
        let hog = a.alloc(ArenaKind::KvPage, 1).unwrap();
        assert_eq!(
            a.available(ArenaKind::KvPage),
            1,
            "only 1 free, restore needs 2"
        );

        let r = ws.restore_pages_warm(&mut a, &plan);
        assert!(
            matches!(r, Err(WorkingSetError::OutOfBlocks { requested: 2, .. })),
            "all-or-nothing: refuses without room for every page"
        );
        assert!(slot(&ws, 0).is_none(), "slot 0 still stashed");
        assert!(slot(&ws, 1).is_none(), "slot 1 still stashed");
        assert_eq!(a.residency(id0).unwrap(), crate::arena::Residency::Cpu);
        assert_eq!(a.residency(id1).unwrap(), crate::arena::Residency::Cpu);

        // Room frees → the retry restores fully (what the prologue's re-park does).
        a.decref(hog.object_id).unwrap();
        let moves = ws.restore_pages_warm(&mut a, &plan).unwrap();
        assert_eq!(moves.len(), 2, "both restored once room is available");
        assert_eq!(slot(&ws, 0), Some(id0));
        assert_eq!(slot(&ws, 1), Some(id1));

        ws.destroy(&mut a, &mut cas);
    }

    // 21e-bis. restore is all-or-nothing for a mid-batch NON-OOB failure too, not
    //      just the pool pre-check. `arena.restore` freed the CPU source + flipped
    //      residency IN PLACE, so a naive `restore` loop that failed on page N left
    //      pages 0..N committed (Gpu, CPU freed, slot mapped) yet UNPOPULATED — the
    //      caller issues the H2D copies only AFTER the whole call returns. A mapped
    //      slot reads silent garbage (never trips `UnwrittenPage`), defeating the
    //      v2 fail-loud guarantee. The stage→commit split rolls the staged GPU
    //      allocs back on ANY error: nothing mapped, nothing flipped, no leak — and
    //      the re-park retry then recovers cleanly once the interference clears.
    #[test]
    fn restore_pages_warm_rolls_back_on_midbatch_non_oob_failure() {
        let mut a = arena_with_cpu(4, 8); // GPU = 4 blocks
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(2).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        write_slot(&mut ws, &mut a, &mut cas, 1, None);
        let id0 = slot(&ws, 0).unwrap();
        let id1 = slot(&ws, 1).unwrap();

        let plan = ws.suspend_pages_warm(&mut a, &mut cas);
        assert_eq!(plan.stash.len(), 2);
        assert_eq!(a.residency(id0).unwrap(), crate::arena::Residency::Cpu);
        assert_eq!(a.residency(id1).unwrap(), crate::arena::Residency::Cpu);

        // Inject a mid-batch non-OOB failure: another lane already restored the
        // SECOND stashed page (id1) — e.g. it CAS-deduped onto that shared page —
        // so id1 is Gpu, not Cpu, when THIS set tries to restore. Staging id1 then
        // fails `InvalidResidency` AFTER id0 has already been staged.
        a.restore(id1).unwrap();
        assert_eq!(a.residency(id1).unwrap(), crate::arena::Residency::Gpu);
        let avail_before = a.available(ArenaKind::KvPage);

        let r = ws.restore_pages_warm(&mut a, &plan);
        match r {
            Err(WorkingSetError::Arena(ref msg)) => {
                assert!(
                    msg.contains("expected Cpu") && msg.contains("restore"),
                    "mid-batch InvalidResidency surfaced, got: {msg}"
                );
            }
            other => panic!("expected Arena(InvalidResidency), got {other:?}"),
        }
        // All-or-nothing: NOTHING committed. id0 stays stashed (Cpu, slot None);
        // its staged GPU alloc was rolled back (no leak — avail unchanged). No
        // slot is left mapped-but-unpopulated.
        assert!(slot(&ws, 0).is_none(), "slot 0 still stashed, not mapped");
        assert!(slot(&ws, 1).is_none(), "slot 1 still stashed, not mapped");
        assert_eq!(a.residency(id0).unwrap(), crate::arena::Residency::Cpu);
        assert_eq!(
            a.available(ArenaKind::KvPage),
            avail_before,
            "id0's staged GPU alloc was aborted — no leak"
        );

        // The interference clears (id1 goes back to CPU) → the re-park retry now
        // restores the FULL set cleanly.
        a.offload(id1).unwrap();
        let moves = ws.restore_pages_warm(&mut a, &plan).unwrap();
        assert_eq!(moves.len(), 2, "both restored once id1 is Cpu again");
        assert_eq!(slot(&ws, 0), Some(id0));
        assert_eq!(slot(&ws, 1), Some(id1));

        ws.destroy(&mut a, &mut cas);
    }

    // 21f. suspend PRESERVES a pinned in-flight write-txn (carrier+preempt bug):
    //      under run-ahead a prior forward's write is in flight (write target
    //      arena-pinned, pending write-txn) when a co-resident lane self-suspends.
    //      Suspend must NOT abort it or revert its slot — else that forward's
    //      output page becomes Reserved and its harvest reads UnwrittenPage
    //      ("output: slot 0 has no written page"). The pinned-skip preserves it.
    #[test]
    fn suspend_preserves_pinned_inflight_write() {
        let mut a = arena_with_cpu(4, 8);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(1).unwrap();
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        let committed = slot(&ws, 0).unwrap();
        // Share so the write CoW-repoints (observable), then PIN the new target
        // exactly like execute_impl does for an in-flight forward's write.
        let sibling = ws.fork(&mut a).unwrap();
        let mut txn = a.txn_begin();
        let wtx = ws.begin_write_txn();
        let (target, _mv) = ws.cow_write_slot(wtx, 0, &mut txn, &mut a).unwrap();
        a.txn_pin(&mut txn, target).unwrap();
        let written = slot(&ws, 0);
        assert_ne!(
            written,
            Some(committed),
            "slot repointed to the in-flight target"
        );

        // Suspend must LEAVE the in-flight write intact.
        let plan = ws.suspend_pages_warm(&mut a, &mut cas);
        assert!(
            !ws.pending.is_empty(),
            "in-flight write-txn PRESERVED (not aborted)"
        );
        assert_eq!(
            slot(&ws, 0),
            written,
            "slot still points at the pinned target"
        );
        assert_eq!(
            a.residency(target).unwrap(),
            crate::arena::Residency::Gpu,
            "pinned in-flight target NOT stashed"
        );
        assert!(
            !plan.stash.iter().any(|(_, id, _)| *id == target),
            "the pinned target is never staged"
        );

        // Cleanup: unpin + abort the arena txn, then teardown.
        a.txn_abort(txn);
        ws.abort_all_writes();
        drop(sibling);
        ws.destroy(&mut a, &mut cas);
    }

    // ── M4: mark-sweep, compact, grace period, generation-on-compact/reorder ──

    // 22. mark_dead — the live slots not in the reachable snapshot (host GC).
    #[test]
    fn mark_dead_returns_live_unreachable_slots() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(5).unwrap();
        ws.free(&[1], &mut a, &mut cas).unwrap(); // tombstone interior slot 1
        // live = {0,2,3,4}; reachable = {0,2} ⇒ dead = {3,4}.
        let reachable: BTreeSet<u32> = [0u32, 2].into_iter().collect();
        assert_eq!(ws.mark_dead(&reachable), vec![3, 4]);
        // A tombstoned id is never "dead" (already free); an unreachable-but-
        // granted-since id must be unioned into `reachable` by the caller.
        let none_reachable: BTreeSet<u32> = BTreeSet::new();
        assert_eq!(ws.mark_dead(&none_reachable), vec![0, 2, 3, 4]);
    }

    // 23. compact — pack live token runs densely; remap + gather plan + free src;
    //     generation bumps (W8). A run straddling the dst page boundary splits.
    #[test]
    fn compact_packs_runs_remaps_and_bumps_generation() {
        let mut a = arena(32);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0); // PAGE = 4
        ws.alloc(3).unwrap();
        for i in 0..3 {
            write_slot(&mut ws, &mut a, &mut cas, i, None);
        }
        let gen0 = ws.generation();
        let live0 = a.live_objects();

        // Keep slot0[0..2] (2 tok) + slot2[1..4] (3 tok) = 5 tok ⇒ 2 fresh pages.
        let runs = [
            TokenRun {
                src_slot: 0,
                start: 0,
                len: 2,
            },
            TokenRun {
                src_slot: 2,
                start: 1,
                len: 3,
            },
        ];
        // src pages captured before compact (they are freed after).
        let src0_pg = a.blocks(slot(&ws, 0).unwrap()).unwrap()[0];
        let src2_pg = a.blocks(slot(&ws, 2).unwrap()).unwrap()[0];
        let mut gather_called = false;
        let mut plan_seen: Vec<PageGatherOp> = Vec::new();
        let remap = ws
            .compact(&runs, &mut a, &mut cas, |plan| {
                gather_called = true;
                plan_seen = plan.to_vec();
            })
            .unwrap();

        assert!(gather_called, "the gather op ran before the source free");
        assert_eq!(remap.new_slots.len(), 2); // ceil(5/4)
        // run0 → ns0[0..2]; run1 splits ns0[2..4] (2) + ns1[0..1] (1).
        let ns0 = remap.new_slots[0];
        let ns1 = remap.new_slots[1];
        assert_eq!(
            remap.gather,
            vec![
                GatherOp {
                    src_slot: 0,
                    src_off: 0,
                    dst_slot: ns0,
                    dst_off: 0,
                    len: 2
                },
                GatherOp {
                    src_slot: 2,
                    src_off: 1,
                    dst_slot: ns0,
                    dst_off: 2,
                    len: 2
                },
                GatherOp {
                    src_slot: 2,
                    src_off: 3,
                    dst_slot: ns1,
                    dst_off: 0,
                    len: 1
                },
            ]
        );
        // The page-level plan the gather received: same ops resolved to physical
        // pages (src pages captured pre-free; dst = the materialised fresh pages).
        assert_eq!(plan_seen, remap.page_gather);
        let ns0_pg = a.blocks(slot(&ws, ns0 as usize).unwrap()).unwrap()[0];
        let ns1_pg = a.blocks(slot(&ws, ns1 as usize).unwrap()).unwrap()[0];
        assert_eq!(
            remap.page_gather,
            vec![
                PageGatherOp {
                    src_page: src0_pg,
                    src_off: 0,
                    dst_page: ns0_pg,
                    dst_off: 0,
                    len: 2
                },
                PageGatherOp {
                    src_page: src2_pg,
                    src_off: 1,
                    dst_page: ns0_pg,
                    dst_off: 2,
                    len: 2
                },
                PageGatherOp {
                    src_page: src2_pg,
                    src_off: 3,
                    dst_page: ns1_pg,
                    dst_off: 0,
                    len: 1
                },
            ]
        );
        assert_eq!(remap.freed_slots, vec![0, 2]); // src slots freed
        assert_eq!(ws.generation(), gen0 + 1); // W8 compact bump
        assert!(ws.slots[1].is_some()); // slot 1 (untouched) still live
        // Net arena objects: +2 materialised dst pages, −2 freed src pages.
        assert_eq!(a.live_objects(), live0);

        ws.destroy(&mut a, &mut cas);
    }

    // 26. generation bumps EXACTLY on compact/reorder (W8) — not alloc/free/append.
    #[test]
    fn generation_bumps_exactly_on_compact_and_reorder() {
        let mut a = arena(32);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(4).unwrap();
        assert_eq!(ws.generation(), 0); // alloc: no bump
        for i in 0..4 {
            write_slot(&mut ws, &mut a, &mut cas, i, None);
        }
        let mut other = KvWorkingSet::new(PAGE, 0);
        other.alloc(1).unwrap();
        write_slot(&mut other, &mut a, &mut cas, 0, None);
        ws.append(&other, &mut a).unwrap();
        assert_eq!(ws.generation(), 0); // append: no bump
        ws.free(&[4], &mut a, &mut cas).unwrap();
        assert_eq!(ws.generation(), 0); // free: no bump (tombstone)

        ws.reorder(&[0, 1, 2, 3]).unwrap();
        assert_eq!(ws.generation(), 1); // reorder: bump

        let runs = [TokenRun {
            src_slot: 0,
            start: 0,
            len: 1,
        }];
        ws.compact(&runs, &mut a, &mut cas, |_| {}).unwrap();
        assert_eq!(ws.generation(), 2); // compact: bump

        ws.destroy(&mut a, &mut cas);
        other.destroy(&mut a, &mut cas);
    }

    // 27. §6.2-shaped beam step: a lane dies and strands its private tail; the
    //     host waste model (stranded token count) matches what compact reclaims.
    #[test]
    fn beam_strand_and_compact_reclaims_exact_waste() {
        let mut a = arena(64);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0); // PAGE = 4
        // Beam prefix: 2 shared pages (8 tokens), then 3 lanes each write a
        // private tail page (slots 2,3,4).
        ws.alloc(5).unwrap();
        for i in 0..5 {
            write_slot(&mut ws, &mut a, &mut cas, i, None);
        }
        // Step outcome: lanes at slots 3 and 4 die → their tails (2 tokens each,
        // in-page [0,2)) are stranded waste; lane at slot 2 survives (full page).
        // Host waste model: 2 dead lanes × 2 stranded tokens = 4 stranded tokens.
        let host_waste_tokens = 2 * 2;

        // Compact the survivors: shared prefix (slots 0,1 full = 8 tok) + the live
        // lane's tail (slot 2 [0,2) = 2 tok) = 10 live tokens ⇒ 3 packed pages.
        let live_runs = [
            TokenRun {
                src_slot: 0,
                start: 0,
                len: 4,
            },
            TokenRun {
                src_slot: 1,
                start: 0,
                len: 4,
            },
            TokenRun {
                src_slot: 2,
                start: 0,
                len: 2,
            },
        ];
        let live_tokens: u32 = live_runs.iter().map(|r| r.len).sum();
        let total_tokens = 5 * PAGE; // 5 pages fully materialised
        // The waste compact reclaims = everything not in the live runs.
        let reclaimed = total_tokens - live_tokens;
        // Two full dead lanes (slots 3,4 = 8 tok) + slot-2's masked residual
        // (2 tok) = 10 reclaimed; of which the strand model's 4 are the dead
        // lanes' *written* tails — the rest is unwritten page tail.
        assert_eq!(reclaimed, 10);
        assert!(host_waste_tokens <= reclaimed);

        let remap = ws.compact(&live_runs, &mut a, &mut cas, |_| {}).unwrap();
        assert_eq!(remap.new_slots.len(), 3); // ceil(10/4)
        // Slots 0,1,2 freed (compacted into fresh); slots 3,4 stay as the dead
        // lanes' tombstonable pages (host frees them next).
        assert_eq!(remap.freed_slots, vec![0, 1, 2]);
        ws.free(&[3, 4], &mut a, &mut cas).unwrap(); // sweep the dead lanes
        assert_eq!(ws.generation(), 1); // exactly one compact bump

        ws.destroy(&mut a, &mut cas);
    }

    /// Bug#2 retention invariant (the concurrent-decode contamination root):
    /// a decode appends the new token into the RETAINED tail slot. Each fire
    /// must keep the SAME physical page (same `ObjectId`, same block, rc≥1) —
    /// an rc==1 continuing page is written IN PLACE, never CoW-copied+freed.
    /// If a fire frees the tail block, a concurrent request re-allocs the same
    /// physical page and clobbers it (charlie's page-9783 churn on the 4090).
    #[test]
    fn decode_write_page_retained_across_fires() {
        let mut a = arena(16);
        let mut cas = KvCas::new();
        let mut ws = KvWorkingSet::new(PAGE, 0);
        ws.alloc(1).unwrap();
        // Prefill: materialise the tail page (a partial page — not sealed).
        write_slot(&mut ws, &mut a, &mut cas, 0, None);
        let page0 = slot(&ws, 0).unwrap();
        let block0 = a.blocks(page0).unwrap()[0];
        assert_eq!(a.refcount(page0).unwrap(), 1);

        for fire in 0..5 {
            // One decode fire: append into the retained tail slot 0.
            let wtx = ws.begin_write_txn();
            let mut txn = a.txn_begin();
            let (obj, mp) = ws.cow_write_slot(wtx, 0, &mut txn, &mut a).unwrap();
            assert!(
                mp.is_none(),
                "fire {fire}: rc==1 continuing decode page must NOT CoW-copy"
            );
            a.txn_pin(&mut txn, obj).unwrap();
            a.txn_commit(txn).unwrap();
            ws.commit_writes(wtx);

            // RETENTION: the tail page is unchanged and still live at rc≥1.
            assert_eq!(
                slot(&ws, 0),
                Some(page0),
                "fire {fire}: tail ObjectId churned"
            );
            assert_eq!(
                a.blocks(page0).unwrap()[0],
                block0,
                "fire {fire}: block moved"
            );
            assert!(
                a.refcount(page0).unwrap() >= 1,
                "fire {fire}: tail page freed!"
            );

            // A concurrent request's fresh alloc must NOT be handed the live block.
            let other = a.alloc(ArenaKind::KvPage, 1).unwrap();
            assert_ne!(
                a.blocks(other.object_id).unwrap()[0],
                block0,
                "fire {fire}: concurrent alloc stole the still-live tail block {block0}"
            );
            a.decref(other.object_id).unwrap();
        }
    }

    /// Bug#2, the concurrent-fleet shape: two contexts (same prompt) each
    /// prefill + decode against the shared arena, interleaved as concurrent
    /// admission does. Their live tail pages must occupy DISJOINT physical
    /// blocks at every step — no request is handed a block another still holds.
    #[test]
    fn two_concurrent_contexts_get_disjoint_tail_blocks() {
        let mut a = arena(32);
        let mut cas = KvCas::new();
        let mut wa = KvWorkingSet::new(PAGE, 0);
        let mut wb = KvWorkingSet::new(PAGE, 0);
        wa.alloc(1).unwrap();
        wb.alloc(1).unwrap();
        write_slot(&mut wa, &mut a, &mut cas, 0, None);
        write_slot(&mut wb, &mut a, &mut cas, 0, None);

        let decode_fire = |ws: &mut KvWorkingSet, a: &mut Arena| {
            let wtx = ws.begin_write_txn();
            let mut txn = a.txn_begin();
            let (obj, _mp) = ws.cow_write_slot(wtx, 0, &mut txn, a).unwrap();
            a.txn_pin(&mut txn, obj).unwrap();
            a.txn_commit(txn).unwrap();
            ws.commit_writes(wtx);
        };

        for step in 0..5 {
            decode_fire(&mut wa, &mut a);
            decode_fire(&mut wb, &mut a);
            let ba = a.blocks(slot(&wa, 0).unwrap()).unwrap()[0];
            let bb = a.blocks(slot(&wb, 0).unwrap()).unwrap()[0];
            assert_ne!(
                ba, bb,
                "step {step}: contexts A and B share physical block {ba} (contamination!)"
            );
        }
        wa.destroy(&mut a, &mut cas);
        wb.destroy(&mut a, &mut cas);
        assert_eq!(a.live_objects(), 0, "leaked pages");
    }
}

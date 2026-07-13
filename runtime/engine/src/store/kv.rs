//! KV store: WorkingSets, the mapping trie, typed pool access, implicit CoW
//! allocation, hash lifecycle, and the prepare/commit/abort protocol
//! (kv_refact.md, Runtime Module Architecture).
//!
//! Layering:
//! - [`hash`]: pure token-slot / page / cached-path hash calculations.
//! - [`page_table`]: `KvPageTable` — the radix-compressed mapping trie,
//!   `Pages::ParentSelection` structural sharing, reachability lifetime, and
//!   flattening. It never allocates physical ids or calls driver APIs.
//! - [`write`]: `KvPreparedWrite`, the per-fire prepared operation.
//! - [`KvStore`] (this module): the single authority over which
//!   `PhysicalKvPageId`s are live. Owns the table and the typed pool,
//!   classifies write intents (fresh / in-place / CoW), and commits or aborts
//!   prepared writes on driver completion epochs.
//!
//! WIT resource wiring (`store/kv/working_set.rs`) and CAS/CacheFabric
//! integration (`store/kv/cas.rs`) land in later increments.
//!
//! Complete typed-store API (kv_refact.md): some methods here are not yet
//! called by the live single-model fire path (only a subset of the typed
//! store surface is currently wired) but are exercised by this module's
//! own unit test suite and reserved for upcoming increments (contention/
//! reclaim expansion, RS buffer-write paths, etc.) — kept rather than
//! deleted, allowed rather than silently masked.
#![allow(dead_code)]

pub mod hash;
pub mod page_table;
pub mod project;
pub mod working_set;
pub mod write;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;

use hash::Hash256;
use page_table::{
    HostKvSlotId, KvPageBacking, KvPageTable, KvTableError, NodeId, PhysicalKvPageId,
    PublishedPage, TriePageLocation, WorkingSetId,
};
use write::{KvPreparedWrite, PageCommit, PreparedTarget};

use crate::store::pool::Pool;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvStoreError {
    #[error(transparent)]
    Table(#[from] KvTableError),
    /// Pool exhaustion. Raised only at forward preparation (`reserve` is
    /// logical); the scheduler routes this through the contention ladder.
    #[error("kv pool exhausted: requested {requested}, available {available}")]
    OutOfPages { requested: usize, available: usize },
    #[error("invalid write set: {reason}")]
    BadWriteSet { reason: &'static str },
    #[error("commit metadata does not match the prepared targets")]
    CommitMismatch,
    #[error("allocation grant size mismatch: required {required}, granted {granted}")]
    GrantMismatch { required: usize, granted: usize },
    #[error("host KV swap pool exhausted: requested {requested}, available {available}")]
    HostSwapFull { requested: usize, available: usize },
    #[error("working set has a prepared write in flight")]
    Busy,
}

#[derive(Default)]
struct FlatEntry {
    version: u64,
    cache: Option<Vec<PhysicalKvPageId>>,
}

#[derive(Debug)]
struct PendingOverlay {
    seq: u64,
    mappings: Vec<(u64, PhysicalKvPageId)>,
    pages: Vec<(u64, PageCommit)>,
    chain_state: Hash256,
    token_len_after: Option<u64>,
}

#[derive(Debug)]
struct DeferredRelease {
    cache_roots_max: Option<usize>,
}

#[derive(Debug)]
struct WriteClassification {
    mapped: u64,
    in_place: Vec<(u64, PhysicalKvPageId)>,
    cow_start: Option<u64>,
    cow_srcs: Vec<PhysicalKvPageId>,
    fresh: Vec<u64>,
}

impl WriteClassification {
    fn required_pages(&self) -> usize {
        self.cow_srcs.len() + self.fresh.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuspendDisposition {
    NothingReclaimable,
    GraceDeferred,
}

#[derive(Debug)]
pub enum KvSuspendPrepare {
    Prepared(KvSuspendTxn),
    Deferred(SuspendDisposition),
}

#[derive(Debug)]
pub struct SuspendPage {
    pub location: TriePageLocation,
    pub gpu_id: PhysicalKvPageId,
    pub host_slot: HostKvSlotId,
}

#[derive(Debug)]
pub struct KvSuspendTxn {
    working_sets: HashSet<WorkingSetId>,
    pages: Vec<SuspendPage>,
    pinned: Vec<NodeId>,
}

impl KvSuspendTxn {
    pub fn gpu_ids(&self) -> Vec<u32> {
        self.pages.iter().map(|page| page.gpu_id.0).collect()
    }

    pub fn host_slots(&self) -> Vec<u32> {
        self.pages.iter().map(|page| page.host_slot.0).collect()
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}

#[derive(Debug)]
pub struct RestorePage {
    pub location: TriePageLocation,
    pub host_slot: HostKvSlotId,
    pub gpu_id: PhysicalKvPageId,
}

#[derive(Debug)]
pub struct KvRestoreTxn {
    working_sets: HashSet<WorkingSetId>,
    pages: Vec<RestorePage>,
    pinned: Vec<NodeId>,
}

impl KvRestoreTxn {
    pub fn gpu_ids(&self) -> Vec<u32> {
        self.pages.iter().map(|page| page.gpu_id.0).collect()
    }

    pub fn host_slots(&self) -> Vec<u32> {
        self.pages.iter().map(|page| page.host_slot.0).collect()
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}

/// The KV store: mapping trie + physical pool + prepared-write protocol.
pub struct KvStore {
    table: KvPageTable,
    pool: Pool<PhysicalKvPageId>,
    host_pool: Pool<HostKvSlotId>,
    /// Per-WorkingSet flattened-table cache. Versioned; a version bump means
    /// the device-shared buffer must be republished. Mutations that do not
    /// change any logical->physical value (owner compaction, collection) do
    /// not bump versions.
    flat: HashMap<WorkingSetId, FlatEntry>,
    /// Prepared writes visible only to later prepares of the same WorkingSet.
    /// The committed trie, CAS, reclaim accounting, and public flat table never
    /// consult this FIFO.
    pending: HashMap<WorkingSetId, VecDeque<PendingOverlay>>,
    deferred_releases: HashMap<WorkingSetId, DeferredRelease>,
    opaque_nonce: Hash256,
    opaque_counter: u64,
    /// The pass-wide cache domain folded into every canonical token-slot
    /// hash (model/weights identity; today boot-scoped via the nonce).
    domain: Hash256,
    /// Content-addressable index over canonical FULL pages: the page's last
    /// token-slot hash (the chain value at its boundary) -> its trie
    /// location. Entries are validated on lookup (owner compaction may move
    /// locals; collection may free the node) and pruned lazily.
    cas: HashMap<Hash256, CasEntry>,
    /// Bounded FIFO of auto-retained cache roots (prefix caching on release):
    /// the oldest lease is evicted past the cap, and the contention ladder's
    /// rung 1 drops any of them the moment memory is needed.
    retained: VecDeque<NodeId>,
    /// Monotonic submission sequence: bumped per prepared write. Freed slots
    /// are recycled tagged with the current value and retired once the fire
    /// carrying that sequence completes (FIFO stream order), or immediately
    /// via [`Self::retire_idle`] when nothing is in flight.
    seq: u64,
    in_flight: u64,
}

#[derive(Debug, Clone, Copy)]
struct CasEntry {
    node: NodeId,
    local: u64,
}

impl KvStore {
    pub fn new(capacity: u32, opaque_nonce: Hash256) -> Self {
        Self::new_with_swap(capacity, 0, opaque_nonce)
    }

    pub fn new_with_swap(capacity: u32, host_capacity: u32, opaque_nonce: Hash256) -> Self {
        Self::new_with_swap_range(0, capacity, host_capacity, opaque_nonce)
    }

    pub fn new_with_swap_range(
        base_page: u32,
        capacity: u32,
        host_capacity: u32,
        opaque_nonce: Hash256,
    ) -> Self {
        Self {
            table: KvPageTable::new(),
            pool: Pool::new_range(base_page, capacity),
            host_pool: Pool::new(host_capacity),
            flat: HashMap::new(),
            pending: HashMap::new(),
            deferred_releases: HashMap::new(),
            opaque_nonce,
            opaque_counter: 0,
            domain: hash::cache_domain(&opaque_nonce),
            cas: HashMap::new(),
            retained: VecDeque::new(),
            seq: 0,
            in_flight: 0,
        }
    }

    /// The cache domain for canonical token-slot hashing (see
    /// [`hash::chain_token_slot_hash`]).
    pub fn domain(&self) -> Hash256 {
        self.domain
    }

    /// The epoch to tag frees with right now (see `seq`).
    pub fn current_epoch(&self) -> u64 {
        self.seq
    }

    /// Retire everything immediately when no prepared write is in flight.
    pub fn retire_idle(&mut self) {
        if self.in_flight == 0 {
            self.pool.retire_through(self.seq);
        }
    }

    // ------------------------------------------------------------------
    // WorkingSet lifecycle (delegates to the table; pool-aware where freeing)
    // ------------------------------------------------------------------

    pub fn create_working_set(&mut self) -> WorkingSetId {
        self.table.create_working_set()
    }

    pub fn fork(&mut self, ws: WorkingSetId) -> Result<WorkingSetId, KvStoreError> {
        Ok(self.table.fork(ws)?)
    }

    pub fn slice(
        &mut self,
        ws: WorkingSetId,
        range: Range<u64>,
    ) -> Result<WorkingSetId, KvStoreError> {
        let parent_mapped = self.table.mapped_len(ws)?;
        let parent_chain = self.table.chain_state(ws)?;
        let child = self.table.slice(ws, range.clone())?;
        if range.start == 0 && range.end == parent_mapped {
            // Identical visible content: continuations must hash identically.
            self.table.set_chain_state(child, parent_chain)?;
        } else {
            self.refresh_chain_after_surgery(child, range.start == 0)?;
        }
        Ok(child)
    }

    pub fn reserve(&mut self, ws: WorkingSetId, pages: u64) -> Result<Range<u64>, KvStoreError> {
        Ok(self.table.reserve(ws, pages)?)
    }

    /// Freed slots recycle after `epoch` retires (all in-flight users done).
    pub fn discard(
        &mut self,
        ws: WorkingSetId,
        ranges: &[Range<u64>],
        epoch: u64,
    ) -> Result<(), KvStoreError> {
        // Chain-state bookkeeping, decided pre-mutation: a discard confined
        // to the mapped TAIL leaves the visible prefix intact (the last
        // surviving slot hash stays the exact continuation identity); any
        // front/interior removal changes the visible context and forces a
        // recompute from the surviving pages' identities.
        let old_mapped = self.table.mapped_len(ws)?;
        let removed: u64 = ranges
            .iter()
            .map(|r| {
                r.end
                    .min(old_mapped)
                    .saturating_sub(r.start.min(old_mapped))
            })
            .sum();
        let new_mapped = old_mapped - removed;
        let prefix_intact = ranges
            .iter()
            .filter(|r| r.start < old_mapped)
            .all(|r| r.start >= new_mapped);

        let freed = self.table.discard(ws, ranges)?;
        self.pool.recycle_after_epoch(freed, epoch);
        self.invalidate_flat(ws);
        if removed > 0 {
            self.refresh_chain_after_surgery(ws, prefix_intact)?;
        }
        Ok(())
    }

    /// Recompute a WorkingSet's chain state after mapping surgery. With the
    /// prefix intact (tail-only edits) the last surviving slot hash IS the
    /// exact continuation identity; otherwise fold the visible pages'
    /// identities (path-hash domain), substituting opaque draws for pages
    /// with nothing recorded so unknown content never matches anything.
    fn refresh_chain_after_surgery(
        &mut self,
        ws: WorkingSetId,
        prefix_intact: bool,
    ) -> Result<(), KvStoreError> {
        let mapped = self.table.mapped_len(ws)?;
        if mapped == 0 {
            self.table.set_chain_state(ws, None)?;
            return Ok(());
        }
        let state = if prefix_intact {
            let last = self
                .table
                .page_token_hashes(ws, mapped - 1)?
                .iter()
                .rev()
                .find_map(|h| *h);
            Some(last.unwrap_or_else(|| self.next_opaque_hash()))
        } else {
            let idents = self.table.visible_page_identities(ws)?;
            let mut acc: Option<Hash256> = None;
            for ident in idents {
                let v = ident.unwrap_or_else(|| self.next_opaque_hash());
                acc = hash::fold_path_hash(acc, &[v]);
            }
            acc
        };
        self.table.set_chain_state(ws, state)?;
        Ok(())
    }

    pub fn release_working_set(&mut self, ws: WorkingSetId, epoch: u64) {
        if self
            .pending
            .get(&ws)
            .is_some_and(|entries| !entries.is_empty())
        {
            self.deferred_releases.insert(
                ws,
                DeferredRelease {
                    cache_roots_max: None,
                },
            );
            return;
        }
        self.release_working_set_now(ws, epoch);
    }

    fn release_working_set_now(&mut self, ws: WorkingSetId, epoch: u64) {
        let freed = self.table.release_working_set_backings(ws);
        self.recycle_backings(freed, epoch);
        self.flat.remove(&ws);
        self.pending.remove(&ws);
    }

    fn recycle_backings(&mut self, backings: Vec<KvPageBacking>, epoch: u64) {
        let mut resident = Vec::new();
        let mut swapped = Vec::new();
        for backing in backings {
            match backing {
                KvPageBacking::Resident(id) => resident.push(id),
                KvPageBacking::Swapped(slot) => swapped.push(slot),
            }
        }
        self.pool.recycle_after_epoch(resident, epoch);
        self.host_pool.release_reserved(swapped);
    }

    /// Release a WorkingSet, RETAINING its path as a prefix-cache root when
    /// it carries canonical content (page 0 committed with a page hash). The
    /// lease keeps the pages reachable for CAS matches; pressure reclaims
    /// them via the contention ladder's rung 1, and the FIFO cap
    /// (`max_roots`) bounds steady-state retention. Non-canonical paths
    /// release exactly like [`Self::release_working_set`].
    pub fn release_working_set_cached(&mut self, ws: WorkingSetId, epoch: u64, max_roots: usize) {
        if self
            .pending
            .get(&ws)
            .is_some_and(|entries| !entries.is_empty())
        {
            self.deferred_releases.insert(
                ws,
                DeferredRelease {
                    cache_roots_max: Some(max_roots),
                },
            );
            return;
        }
        let retain = match (self.table.terminal(ws), self.table.page_hash_at(ws, 0)) {
            (Ok(Some(root)), Ok(Some(_))) if max_roots > 0 => Some(root),
            _ => None,
        };
        if let Some(root) = retain {
            self.table.lease_cache_root(root);
            self.retained.push_back(root);
            while self.retained.len() > max_roots {
                if let Some(old) = self.retained.pop_front() {
                    self.table.release_cache_root(old);
                }
            }
        }
        // The release's reachability sweep also collects any just-evicted
        // roots' pages.
        self.release_working_set_now(ws, epoch);
    }

    // ------------------------------------------------------------------
    // Prepare / commit / abort
    // ------------------------------------------------------------------

    /// Classify the fire's KV write intents and allocate physical slots.
    ///
    /// `write_indexes` are the WorkingSet-relative page indexes the pass
    /// writes. Fresh indexes (at or past the mapped end) must be contiguous
    /// from it. A committed page is written in place when nothing but `ws`
    /// observes its owning node; otherwise the tail from the lowest shared
    /// written index is CoW'd: every page in `[cow_start, mapped)` gets a
    /// fresh slot and a copy plan entry, written or not, because the mapping
    /// rebase is a growth-boundary edit and cannot skip interior pages.
    pub fn prepare_write(
        &mut self,
        ws: WorkingSetId,
        write_indexes: &[u64],
    ) -> Result<KvPreparedWrite, KvStoreError> {
        let classification = self.classify_write(ws, write_indexes)?;
        let need = classification.required_pages();
        let allocated = self
            .pool
            .try_alloc_n(need)
            .ok_or(KvStoreError::OutOfPages {
                requested: need,
                available: self.pool.available(),
            })?;
        self.finish_prepare_write(ws, classification, allocated)
    }

    /// Prepare using concrete ids reserved by the contention orchestrator.
    pub fn prepare_write_granted(
        &mut self,
        ws: WorkingSetId,
        write_indexes: &[u64],
        mut granted: Vec<PhysicalKvPageId>,
    ) -> Result<KvPreparedWrite, KvStoreError> {
        let classification = match self.classify_write(ws, write_indexes) {
            Ok(classification) => classification,
            Err(error) => {
                self.pool.release_reserved(granted);
                return Err(error);
            }
        };
        let required = classification.required_pages();
        if granted.len() < required {
            let granted_len = granted.len();
            self.pool.release_reserved(granted);
            return Err(KvStoreError::GrantMismatch {
                required,
                granted: granted_len,
            });
        }
        if granted.len() > required {
            let extra = granted.split_off(required);
            self.pool.release_reserved(extra);
        }
        self.finish_prepare_write(ws, classification, granted)
    }

    pub fn required_pages(
        &mut self,
        ws: WorkingSetId,
        write_indexes: &[u64],
    ) -> Result<usize, KvStoreError> {
        Ok(self.classify_write(ws, write_indexes)?.required_pages())
    }

    fn classify_write(
        &mut self,
        ws: WorkingSetId,
        write_indexes: &[u64],
    ) -> Result<WriteClassification, KvStoreError> {
        let visible = self.visible_flat_table(ws)?;
        let mapped = visible.len() as u64;
        let page_len = self.table.page_len(ws)?;

        let mut indexes: Vec<u64> = write_indexes.to_vec();
        indexes.sort_unstable();
        indexes.dedup();
        if indexes.last().is_some_and(|&max| max >= page_len) {
            return Err(KvStoreError::BadWriteSet {
                reason: "write beyond the logical reservation",
            });
        }

        let fresh: Vec<u64> = indexes.iter().copied().filter(|&i| i >= mapped).collect();
        for (offset, &index) in fresh.iter().enumerate() {
            if index != mapped + offset as u64 {
                return Err(KvStoreError::BadWriteSet {
                    reason: "fresh writes must be contiguous from the mapped end",
                });
            }
        }

        let mut in_place: Vec<(u64, PhysicalKvPageId)> = Vec::new();
        let mut cow_start: Option<u64> = None;
        for &index in indexes.iter().filter(|&&i| i < mapped) {
            if self.pending_target(ws, index).is_some() {
                in_place.push((index, visible[index as usize]));
            } else if self.table.privately_writable(ws, index)? {
                in_place.push((index, visible[index as usize]));
            } else {
                cow_start = Some(cow_start.map_or(index, |c| c.min(index)));
            }
        }
        // A private write inside the rebased region rides the CoW instead:
        // its in-place result would be shadowed by the copied page.
        if let Some(cs) = cow_start {
            in_place.retain(|&(i, _)| i < cs);
        }

        // Resolve current ids before allocating so no failure path leaks ids.
        let cow_srcs: Vec<PhysicalKvPageId> = match cow_start {
            Some(cs) => visible[cs as usize..mapped as usize].to_vec(),
            None => Vec::new(),
        };
        Ok(WriteClassification {
            mapped,
            in_place,
            cow_start,
            cow_srcs,
            fresh,
        })
    }

    fn finish_prepare_write(
        &mut self,
        ws: WorkingSetId,
        classification: WriteClassification,
        allocated: Vec<PhysicalKvPageId>,
    ) -> Result<KvPreparedWrite, KvStoreError> {
        let WriteClassification {
            mapped,
            in_place,
            cow_start,
            cow_srcs,
            fresh,
        } = classification;
        let mut targets = Vec::with_capacity(in_place.len() + allocated.len());
        for &(index, dst) in &in_place {
            targets.push(PreparedTarget::InPlace { index, dst });
        }
        let mut fresh_ids = allocated.iter().copied();
        if let Some(cs) = cow_start {
            for (offset, &src) in cow_srcs.iter().enumerate() {
                targets.push(PreparedTarget::Cow {
                    index: cs + offset as u64,
                    src,
                    dst: fresh_ids.next().expect("allocated covers cow region"),
                });
            }
        }
        for &index in &fresh {
            targets.push(PreparedTarget::Fresh {
                index,
                dst: fresh_ids.next().expect("allocated covers fresh pages"),
            });
        }

        // Pin the terminal snapshot after classification: the pin guards the
        // captured path against concurrent mutation and collection until
        // commit/abort, but must not make this fire's own pages look shared.
        let pinned = self.table.terminal(ws)?;
        if let Some(terminal) = pinned {
            self.table.pin(terminal);
        }

        self.seq += 1;
        self.in_flight += 1;
        Ok(KvPreparedWrite {
            ws,
            pinned,
            targets,
            allocated,
            old_mapped: mapped,
            cow_start,
            seq: self.seq,
        })
    }

    /// Commit a prepared write after the driver reports success at `epoch`.
    /// `commits` aligns with `prepared.targets()`. All-or-nothing for now;
    /// partial driver commits abort instead.
    pub fn commit(
        &mut self,
        prepared: KvPreparedWrite,
        commits: &[PageCommit],
        epoch: u64,
    ) -> Result<(), KvStoreError> {
        if commits.len() != prepared.targets.len() {
            self.abort(prepared, epoch);
            return Err(KvStoreError::CommitMismatch);
        }
        let ws = prepared.ws;
        let pending_chain = self
            .pending
            .get(&ws)
            .and_then(|entries| entries.iter().find(|entry| entry.seq == prepared.seq))
            .map(|entry| entry.chain_state);

        // The fire is done; release the snapshot pin before publishing so a
        // sole-user terminal can extend in place instead of growing a node
        // chain per fire.
        if let Some(terminal) = prepared.pinned {
            self.table.unpin(terminal);
        }

        for (target, commit) in prepared.targets.iter().zip(commits) {
            if let PreparedTarget::InPlace { index, .. } = *target {
                self.table.commit_in_place(
                    ws,
                    index,
                    commit.token_hashes.clone(),
                    commit.page_hash,
                )?;
            }
        }

        let mut pages = Vec::new();
        for (target, commit) in prepared.targets.iter().zip(commits) {
            match *target {
                PreparedTarget::Cow { dst, .. } | PreparedTarget::Fresh { dst, .. } => {
                    pages.push(PublishedPage {
                        id: dst,
                        token_hashes: commit.token_hashes.clone(),
                        page_hash: commit.page_hash,
                    });
                }
                PreparedTarget::InPlace { .. } => {}
            }
        }
        match prepared.cow_start {
            Some(cs) => self.table.replace_tail(ws, cs, pages)?,
            None if !pages.is_empty() => self.table.publish_appended(ws, pages)?,
            None => {}
        }

        // Chain state: the next appended slot chains from this fire's
        // highest committed slot hash (an opaque draw when the caller
        // recorded nothing — unknown content must never match anything).
        if let Some((_, commit)) = prepared
            .targets
            .iter()
            .zip(commits)
            .max_by_key(|(t, _)| t.index())
        {
            let last = commit.token_hashes.iter().rev().find_map(|h| *h);
            let state =
                pending_chain.unwrap_or_else(|| last.unwrap_or_else(|| self.next_opaque_hash()));
            self.table.set_chain_state(ws, Some(state))?;
        }

        self.invalidate_flat(ws);
        let freed = self.table.collect_backings();
        self.recycle_backings(freed, epoch);

        // CAS index: canonical FULL pages (the fire path records their
        // page_hash) become content-addressable at their boundary chain
        // value. Located after collection so owner compaction has settled.
        for (target, commit) in prepared.targets.iter().zip(commits) {
            if commit.page_hash.is_none() {
                continue;
            }
            let Some(key) = commit.token_hashes.iter().rev().find_map(|h| *h) else {
                continue;
            };
            if let Ok((node, local)) = self.table.locate_page(ws, target.index()) {
                self.cas.insert(key, CasEntry { node, local });
            }
        }
        self.prune_cas_if_bloated();
        self.retire_pending_overlay(ws, prepared.seq);

        self.in_flight = self.in_flight.saturating_sub(1);
        Ok(())
    }

    /// Failure, poison, or readiness dummy-run: release pending slots and
    /// metadata; the committed mapping and hashes remain authoritative.
    pub fn abort(&mut self, prepared: KvPreparedWrite, epoch: u64) {
        let ws = prepared.ws;
        let seq = prepared.seq;
        if let Some(terminal) = prepared.pinned {
            self.table.unpin(terminal);
        }
        self.pool.recycle_after_epoch(prepared.allocated, epoch);
        let freed = self.table.collect_backings();
        self.recycle_backings(freed, epoch);
        self.retire_pending_overlay(ws, seq);
        self.in_flight = self.in_flight.saturating_sub(1);
    }

    /// Retire completion epochs `<= epoch`, making recycled slots allocatable.
    pub fn retire_through(&mut self, epoch: u64) {
        self.pool.retire_through(epoch);
    }

    // ------------------------------------------------------------------
    // Flattened tables
    // ------------------------------------------------------------------

    /// The WorkingSet's flattened logical->physical table and its version.
    /// The version bumps exactly when a mapping value could have changed;
    /// the driver republishes the device-shared buffer on version change.
    pub fn flat_table(
        &mut self,
        ws: WorkingSetId,
    ) -> Result<(u64, &[PhysicalKvPageId]), KvStoreError> {
        let cached = self.flat.get(&ws).is_some_and(|f| f.cache.is_some());
        if !cached {
            let flat = self.table.flatten(ws)?;
            self.flat.entry(ws).or_default().cache = Some(flat);
        }
        let entry = self.flat.get(&ws).expect("just populated");
        Ok((
            entry.version,
            entry.cache.as_deref().expect("just populated"),
        ))
    }

    /// The pipeline-scoped speculative mapping: committed pages overlaid by
    /// prepared writes in submit order. Callers must not publish this view to
    /// the CAS or expose it through cross-pipeline store APIs.
    pub fn visible_flat_table(
        &mut self,
        ws: WorkingSetId,
    ) -> Result<Vec<PhysicalKvPageId>, KvStoreError> {
        let mut table = {
            let (_, committed) = self.flat_table(ws)?;
            committed.to_vec()
        };
        if let Some(entries) = self.pending.get(&ws) {
            for entry in entries {
                for &(index, page) in &entry.mappings {
                    if table.len() <= index as usize {
                        table.resize(index as usize + 1, PhysicalKvPageId(0));
                    }
                    table[index as usize] = page;
                }
            }
        }
        Ok(table)
    }

    pub fn visible_mapped_len(&mut self, ws: WorkingSetId) -> Result<u64, KvStoreError> {
        let (_, committed) = self.flat_table(ws)?;
        let committed_len = committed.len() as u64;
        let pending_len = self
            .pending
            .get(&ws)
            .into_iter()
            .flat_map(|entries| entries.iter())
            .flat_map(|entry| entry.mappings.iter().map(|(index, _)| index + 1))
            .max()
            .unwrap_or(0);
        Ok(committed_len.max(pending_len))
    }

    pub fn visible_chain_state(&self, ws: WorkingSetId) -> Result<Option<Hash256>, KvStoreError> {
        if let Some(state) = self
            .pending
            .get(&ws)
            .and_then(|entries| entries.back())
            .map(|entry| entry.chain_state)
        {
            return Ok(Some(state));
        }
        self.chain_state(ws)
    }

    pub fn visible_page_hash_at(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Option<Hash256>, KvStoreError> {
        if let Some(commit) = self.pending_page(ws, index) {
            return Ok(commit.page_hash);
        }
        self.page_hash_at(ws, index)
    }

    pub fn visible_page_token_hashes(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Vec<Option<Hash256>>, KvStoreError> {
        if let Some(commit) = self.pending_page(ws, index) {
            return Ok(commit.token_hashes.clone());
        }
        self.page_token_hashes(ws, index)
    }

    pub fn visible_token_len(&self, ws: WorkingSetId, page_size: u32) -> Result<u64, KvStoreError> {
        if let Some(token_len) = self
            .pending
            .get(&ws)
            .and_then(|entries| entries.iter().rev().find_map(|entry| entry.token_len_after))
        {
            return Ok(token_len);
        }
        self.committed_token_len(ws, page_size)
    }

    pub fn track_pending_write(
        &mut self,
        prepared: &KvPreparedWrite,
        commits: &[PageCommit],
        token_len_after: Option<u64>,
    ) {
        debug_assert_eq!(prepared.targets.len(), commits.len());
        let chain_state = commits
            .iter()
            .rev()
            .find_map(|commit| commit.token_hashes.iter().rev().find_map(|hash| *hash))
            .unwrap_or_else(|| self.next_opaque_hash());
        let entry = PendingOverlay {
            seq: prepared.seq,
            mappings: prepared
                .targets
                .iter()
                .map(|target| (target.index(), target.dst()))
                .collect(),
            pages: prepared
                .targets
                .iter()
                .zip(commits)
                .map(|(target, commit)| (target.index(), commit.clone()))
                .collect(),
            chain_state,
            token_len_after,
        };
        let entries = self.pending.entry(prepared.ws).or_default();
        debug_assert!(
            entries.back().is_none_or(|prior| prior.seq < prepared.seq),
            "pending KV overlays must be registered in submit order"
        );
        entries.push_back(entry);
    }

    fn pending_target(&self, ws: WorkingSetId, index: u64) -> Option<PhysicalKvPageId> {
        self.latest_pending(ws, |entry| {
            entry
                .mappings
                .iter()
                .rev()
                .find_map(|&(candidate, page)| (candidate == index).then_some(page))
        })
    }

    fn pending_page(&self, ws: WorkingSetId, index: u64) -> Option<&PageCommit> {
        self.latest_pending(ws, |entry| {
            entry
                .pages
                .iter()
                .rev()
                .find_map(|(candidate, commit)| (*candidate == index).then_some(commit))
        })
    }

    fn latest_pending<'a, T>(
        &'a self,
        ws: WorkingSetId,
        mut find: impl FnMut(&'a PendingOverlay) -> Option<T>,
    ) -> Option<T> {
        self.pending
            .get(&ws)
            .and_then(|entries| entries.iter().rev().find_map(&mut find))
    }

    fn retire_pending_overlay(&mut self, ws: WorkingSetId, seq: u64) {
        let Some(entries) = self.pending.get_mut(&ws) else {
            return;
        };
        if entries.front().is_some_and(|entry| entry.seq == seq) {
            entries.pop_front();
        } else if let Some(position) = entries.iter().position(|entry| entry.seq == seq) {
            debug_assert_eq!(
                position, 0,
                "KV overlays must finalize in pipeline FIFO order"
            );
            entries.remove(position);
        }
        if entries.is_empty() {
            self.pending.remove(&ws);
            if let Some(release) = self.deferred_releases.remove(&ws) {
                let epoch = self.current_epoch();
                match release.cache_roots_max {
                    Some(max_roots) => self.release_working_set_cached(ws, epoch, max_roots),
                    None => self.release_working_set_now(ws, epoch),
                }
            }
        }
    }

    fn invalidate_flat(&mut self, ws: WorkingSetId) {
        let entry = self.flat.entry(ws).or_default();
        entry.version += 1;
        entry.cache = None;
    }

    // ------------------------------------------------------------------
    // Hashes, cache roots, and introspection passthroughs
    // ------------------------------------------------------------------

    /// A fresh opaque token-slot hash (for slots no recipe covers).
    pub fn next_opaque_hash(&mut self) -> Hash256 {
        let counter = self.opaque_counter;
        self.opaque_counter += 1;
        hash::opaque_token_slot_hash(&self.opaque_nonce, counter)
    }

    /// The token-slot hash the next appended slot chains from (`None` =
    /// empty mapping / chain start).
    pub fn chain_state(&self, ws: WorkingSetId) -> Result<Option<Hash256>, KvStoreError> {
        Ok(self.table.chain_state(ws)?)
    }

    /// Committed page hash of the page at `index`, if valid.
    pub fn page_hash_at(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Option<Hash256>, KvStoreError> {
        Ok(self.table.page_hash_at(ws, index)?)
    }

    /// Committed token-slot hashes of the page at `index` (per slot; `None`
    /// = unwritten). The fire path reads this to carry preserved slots
    /// through in-place and CoW commits.
    pub fn page_token_hashes(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Vec<Option<Hash256>>, KvStoreError> {
        Ok(self.table.page_token_hashes(ws, index)?)
    }

    /// The PUBLISHED committed token extent of `ws`: full pages below the
    /// last mapped page plus the written prefix of the last page (every
    /// committed slot carries a token-slot hash — chained or opaque — so the
    /// last `Some` bounds the written prefix). Excludes pending
    /// (prepared-but-unfinalized) fires: a pass's own run-ahead cursor covers
    /// those.
    pub fn committed_token_len(
        &self,
        ws: WorkingSetId,
        page_size: u32,
    ) -> Result<u64, KvStoreError> {
        let mapped = self.table.mapped_len(ws)?;
        if mapped == 0 {
            return Ok(0);
        }
        let hashes = self.table.page_token_hashes(ws, mapped - 1)?;
        let last = hashes
            .iter()
            .rposition(|h| h.is_some())
            .map_or(0, |i| i + 1);
        Ok((mapped - 1) * page_size as u64 + last as u64)
    }

    /// Validated CAS lookup: a canonical full page's boundary chain value ->
    /// its live trie location. Entries whose location no longer carries that
    /// content (owner compaction moved locals, collection freed the node)
    /// are pruned and miss.
    pub fn lookup_cached_page(&mut self, key: &Hash256) -> Option<(NodeId, u64)> {
        let entry = *self.cas.get(key)?;
        let location = TriePageLocation {
            node: entry.node,
            local: entry.local,
        };
        let resident = matches!(
            self.table.backing_at(&location),
            Ok(KvPageBacking::Resident(_))
        );
        if self.table.node_page_last_slot_hash(entry.node, entry.local) != Some(*key) {
            self.cas.remove(key);
            return None;
        }
        (resident && !self.table.page_location_pinned(location))
            .then_some((entry.node, entry.local))
    }

    /// Lazy CAS hygiene: when dead entries outnumber any plausible live set,
    /// sweep by revalidation (lookups already prune what they touch).
    fn prune_cas_if_bloated(&mut self) {
        let cap = (self.pool.capacity() as usize).saturating_mul(4).max(1024);
        if self.cas.len() > cap {
            let table = &self.table;
            self.cas
                .retain(|key, e| table.node_page_last_slot_hash(e.node, e.local) == Some(*key));
        }
    }

    pub fn terminal(&self, ws: WorkingSetId) -> Result<Option<NodeId>, KvStoreError> {
        Ok(self.table.terminal(ws)?)
    }

    pub fn terminal_path_hash(
        &mut self,
        ws: WorkingSetId,
    ) -> Result<Option<Hash256>, KvStoreError> {
        Ok(self.table.terminal_path_hash(ws)?)
    }

    pub fn lease_cache_root(&mut self, node: NodeId) {
        self.table.lease_cache_root(node);
    }

    pub fn release_cache_root(&mut self, node: NodeId, epoch: u64) {
        self.table.release_cache_root(node);
        let freed = self.table.collect_backings();
        self.recycle_backings(freed, epoch);
    }

    /// Contention-ladder rung 1: drop every cache-root lease no live
    /// WorkingSet or in-flight fire reaches (pure cache — no work lost) and
    /// collect. Returns the number of pages recycled (allocatable once
    /// `epoch` retires).
    pub fn drop_unused_cache_leases(&mut self, epoch: u64) -> usize {
        if self.table.drop_unused_cache_leases() == 0 {
            return 0;
        }
        let table = &self.table;
        self.retained.retain(|n| table.is_cache_root(*n));
        let freed = self.table.collect_backings();
        let count = freed.len();
        self.recycle_backings(freed, epoch);
        count
    }

    /// Prefix-cache graft: adopt the cached canonical prefix whose boundary
    /// chain value is `key` into the EMPTY WorkingSet `ws`. On a hit the
    /// matched pages become the WS's visible mapping (structurally shared —
    /// writes CoW like any shared path) and the chain state continues from
    /// `key`, so appends hash exactly like the original continuation.
    /// `expected_pages` cross-checks the structural path length against the
    /// probe's chain position; a mismatch misses rather than grafting wrong
    /// content. Returns the adopted page count.
    pub fn adopt_cached_prefix(
        &mut self,
        ws: WorkingSetId,
        key: &Hash256,
        expected_pages: u64,
    ) -> Result<Option<u64>, KvStoreError> {
        let Some((node, local)) = self.lookup_cached_page(key) else {
            return Ok(None);
        };
        if self.table.path_prefix_len(node, local) != expected_pages {
            return Ok(None);
        }
        let pages = self.table.adopt_path_prefix(ws, node, local)?;
        self.table.set_chain_state(ws, Some(*key))?;
        self.invalidate_flat(ws);
        Ok(Some(pages))
    }

    pub fn adopt_offloaded_prefix(
        &mut self,
        ws: WorkingSetId,
        tokens: &[u32],
        pages: Vec<PhysicalKvPageId>,
        page_size: u32,
    ) -> Result<u64, KvStoreError> {
        if tokens.is_empty()
            || page_size == 0
            || !tokens.len().is_multiple_of(page_size as usize)
            || pages.len() * page_size as usize != tokens.len()
        {
            self.pool.release_reserved(pages);
            return Err(KvStoreError::BadWriteSet {
                reason: "offloaded adoption requires a non-empty full-page token prefix",
            });
        }
        let empty = match (self.mapped_len(ws), self.chain_state(ws)) {
            (Ok(mapped), Ok(chain)) => {
                mapped == 0
                    && chain.is_none()
                    && !self
                        .pending
                        .get(&ws)
                        .is_some_and(|pending| !pending.is_empty())
            }
            (Err(error), _) | (_, Err(error)) => {
                self.pool.release_reserved(pages);
                return Err(error);
            }
        };
        if !empty {
            self.pool.release_reserved(pages);
            return Err(KvStoreError::BadWriteSet {
                reason: "offloaded adoption requires an empty working set",
            });
        }

        let page_count = pages.len() as u64;
        if let Err(error) = self.reserve(ws, page_count) {
            self.pool.release_reserved(pages);
            return Err(error);
        }
        let indexes = (0..page_count).collect::<Vec<_>>();
        let prepared = self.prepare_write_granted(ws, &indexes, pages)?;
        let sequence = prepared.seq();
        let mut previous = None;
        let mut commits = Vec::with_capacity(page_count as usize);
        for (page_index, page_tokens) in tokens.chunks_exact(page_size as usize).enumerate() {
            let mut token_hashes = Vec::with_capacity(page_size as usize);
            for (offset, &token) in page_tokens.iter().enumerate() {
                let position = page_index * page_size as usize + offset;
                let hash = hash::chain_token_slot_hash(
                    &self.domain,
                    previous.as_ref(),
                    token,
                    position as u32,
                );
                previous = Some(hash);
                token_hashes.push(Some(hash));
            }
            commits.push(PageCommit {
                page_hash: Some(hash::page_hash(&token_hashes)),
                token_hashes,
            });
        }
        let epoch = self.current_epoch();
        self.commit(prepared, &commits, epoch)?;
        self.retire_through(sequence);
        Ok(page_count)
    }

    /// Contention-ladder rung 2 victim sizing: pages reachable only from
    /// `ws`'s terminal (its private trie suffix) — what releasing this
    /// WorkingSet would actually free.
    pub fn exclusive_footprint(&self, ws: WorkingSetId) -> Result<u64, KvStoreError> {
        Ok(self.table.exclusive_footprint(ws)?)
    }

    pub fn reserve_device_pages(&mut self, count: usize) -> Option<Vec<PhysicalKvPageId>> {
        self.pool.try_alloc_n(count)
    }

    pub fn release_device_reservation(&mut self, pages: Vec<PhysicalKvPageId>) {
        self.pool.release_reserved(pages);
    }

    pub fn prepare_suspend(
        &mut self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<KvSuspendPrepare, KvStoreError> {
        if working_sets.iter().any(|ws| {
            self.pending
                .get(ws)
                .is_some_and(|entries| !entries.is_empty())
        }) {
            return Ok(KvSuspendPrepare::Deferred(
                SuspendDisposition::GraceDeferred,
            ));
        }
        let (pages, pinned) = self.table.private_resident_pages(working_sets)?;
        if pinned {
            return Ok(KvSuspendPrepare::Deferred(
                SuspendDisposition::GraceDeferred,
            ));
        }
        if pages.is_empty() {
            return Ok(KvSuspendPrepare::Deferred(
                SuspendDisposition::NothingReclaimable,
            ));
        }
        let host_slots =
            self.host_pool
                .try_alloc_n(pages.len())
                .ok_or(KvStoreError::HostSwapFull {
                    requested: pages.len(),
                    available: self.host_pool.available(),
                })?;
        let pinned = match self.table.pin_working_sets(working_sets) {
            Ok(pinned) => pinned,
            Err(error) => {
                self.host_pool.release_reserved(host_slots);
                return Err(error.into());
            }
        };
        let pages = pages
            .into_iter()
            .zip(host_slots)
            .map(|((location, gpu_id), host_slot)| SuspendPage {
                location,
                gpu_id,
                host_slot,
            })
            .collect();
        Ok(KvSuspendPrepare::Prepared(KvSuspendTxn {
            working_sets: working_sets.clone(),
            pages,
            pinned,
        }))
    }

    pub fn suspendable_page_count(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<usize, KvStoreError> {
        let (pages, pinned) = self.table.private_resident_pages(working_sets)?;
        Ok(if pinned { 0 } else { pages.len() })
    }

    pub fn post_drain_reclaimable_page_count(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<usize, KvStoreError> {
        let mut pages: HashSet<_> = self
            .table
            .post_drain_private_resident_pages(working_sets)?
            .into_iter()
            .map(|(_, page)| page)
            .collect();
        for ws in working_sets {
            if let Some(entries) = self.pending.get(ws) {
                pages.extend(
                    entries
                        .iter()
                        .flat_map(|entry| entry.mappings.iter().map(|(_, page)| *page)),
                );
            }
        }
        Ok(pages.len())
    }

    pub fn commit_suspend(&mut self, txn: KvSuspendTxn) -> Result<usize, KvStoreError> {
        let replacements: Vec<_> = txn
            .pages
            .iter()
            .map(|page| {
                (
                    page.location,
                    KvPageBacking::Resident(page.gpu_id),
                    KvPageBacking::Swapped(page.host_slot),
                )
            })
            .collect();
        if let Err(error) = self.table.replace_backings(&replacements) {
            self.table.unpin_terminals(&txn.pinned);
            self.host_pool
                .release_reserved(txn.pages.iter().map(|page| page.host_slot).collect());
            let freed = self.table.collect_backings();
            let epoch = self.current_epoch();
            self.recycle_backings(freed, epoch);
            self.retire_idle();
            return Err(error.into());
        }
        self.table.unpin_terminals(&txn.pinned);
        for ws in txn.working_sets {
            self.invalidate_flat(ws);
        }
        let count = txn.pages.len();
        self.pool.recycle_after_epoch(
            txn.pages.into_iter().map(|page| page.gpu_id).collect(),
            self.current_epoch(),
        );
        self.retire_idle();
        Ok(count)
    }

    pub fn abort_suspend(&mut self, txn: KvSuspendTxn) {
        self.table.unpin_terminals(&txn.pinned);
        self.host_pool
            .release_reserved(txn.pages.into_iter().map(|page| page.host_slot).collect());
        let freed = self.table.collect_backings();
        let epoch = self.current_epoch();
        self.recycle_backings(freed, epoch);
        self.retire_idle();
    }

    pub fn swapped_page_count(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<usize, KvStoreError> {
        Ok(self.table.swapped_pages(working_sets)?.len())
    }

    pub fn prepare_restore(
        &mut self,
        working_sets: &HashSet<WorkingSetId>,
        gpu_pages: Vec<PhysicalKvPageId>,
    ) -> Result<KvRestoreTxn, KvStoreError> {
        if working_sets.iter().any(|ws| {
            self.pending
                .get(ws)
                .is_some_and(|entries| !entries.is_empty())
        }) {
            self.pool.release_reserved(gpu_pages);
            return Err(KvStoreError::Busy);
        }
        let swapped = match self.table.swapped_pages(working_sets) {
            Ok(swapped) => swapped,
            Err(error) => {
                self.pool.release_reserved(gpu_pages);
                return Err(error.into());
            }
        };
        if swapped.len() != gpu_pages.len() {
            let required = swapped.len();
            let granted = gpu_pages.len();
            self.pool.release_reserved(gpu_pages);
            return Err(KvStoreError::GrantMismatch { required, granted });
        }
        let pinned = match self.table.pin_working_sets(working_sets) {
            Ok(pinned) => pinned,
            Err(error) => {
                self.pool.release_reserved(gpu_pages);
                return Err(error.into());
            }
        };
        let pages = swapped
            .into_iter()
            .zip(gpu_pages)
            .map(|((location, host_slot), gpu_id)| RestorePage {
                location,
                host_slot,
                gpu_id,
            })
            .collect();
        Ok(KvRestoreTxn {
            working_sets: working_sets.clone(),
            pages,
            pinned,
        })
    }

    pub fn commit_restore(&mut self, txn: KvRestoreTxn) -> Result<usize, KvStoreError> {
        let replacements: Vec<_> = txn
            .pages
            .iter()
            .map(|page| {
                (
                    page.location,
                    KvPageBacking::Swapped(page.host_slot),
                    KvPageBacking::Resident(page.gpu_id),
                )
            })
            .collect();
        if let Err(error) = self.table.replace_backings(&replacements) {
            self.table.unpin_terminals(&txn.pinned);
            self.pool
                .release_reserved(txn.pages.iter().map(|page| page.gpu_id).collect());
            let freed = self.table.collect_backings();
            let epoch = self.current_epoch();
            self.recycle_backings(freed, epoch);
            self.retire_idle();
            return Err(error.into());
        }
        self.table.unpin_terminals(&txn.pinned);
        for ws in txn.working_sets {
            self.invalidate_flat(ws);
        }
        let count = txn.pages.len();
        self.host_pool
            .release_reserved(txn.pages.into_iter().map(|page| page.host_slot).collect());
        Ok(count)
    }

    pub fn abort_restore(&mut self, txn: KvRestoreTxn) {
        self.table.unpin_terminals(&txn.pinned);
        self.pool
            .release_reserved(txn.pages.into_iter().map(|page| page.gpu_id).collect());
        let freed = self.table.collect_backings();
        let epoch = self.current_epoch();
        self.recycle_backings(freed, epoch);
        self.retire_idle();
    }

    pub fn lookup(&self, ws: WorkingSetId, index: u64) -> Result<PhysicalKvPageId, KvStoreError> {
        Ok(self.table.lookup(ws, index)?)
    }

    pub fn mapped_len(&self, ws: WorkingSetId) -> Result<u64, KvStoreError> {
        Ok(self.table.mapped_len(ws)?)
    }

    pub fn page_len(&self, ws: WorkingSetId) -> Result<u64, KvStoreError> {
        Ok(self.table.page_len(ws)?)
    }

    pub fn available_pages(&self) -> usize {
        self.pool.available()
    }

    pub fn capacity_pages(&self) -> u32 {
        self.pool.capacity()
    }

    pub fn pending_recycle_pages(&self) -> usize {
        self.pool.pending_recycle()
    }

    pub fn host_swap_available(&self) -> usize {
        self.host_pool.available()
    }

    pub fn host_swap_capacity(&self) -> u32 {
        self.host_pool.capacity()
    }

    pub fn backing_counts(&self) -> (usize, usize) {
        self.table.backing_counts()
    }

    #[cfg(test)]
    pub(crate) fn table(&mut self) -> &mut KvPageTable {
        &mut self.table
    }
}

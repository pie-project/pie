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

pub mod hash;
pub mod page_table;
pub mod working_set;
pub mod write;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque};
use std::ops::Range;

use hash::Hash256;
use page_table::{
    KvPageTable, KvTableError, NodeId, PhysicalKvPageId, PublishedPage, WorkingSetId,
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
}

#[derive(Default)]
struct FlatEntry {
    version: u64,
    cache: Option<Vec<PhysicalKvPageId>>,
}

/// The KV store: mapping trie + physical pool + prepared-write protocol.
pub struct KvStore {
    table: KvPageTable,
    pool: Pool<PhysicalKvPageId>,
    /// Per-WorkingSet flattened-table cache. Versioned; a version bump means
    /// the device-shared buffer must be republished. Mutations that do not
    /// change any logical->physical value (owner compaction, collection) do
    /// not bump versions.
    flat: HashMap<WorkingSetId, FlatEntry>,
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
        Self {
            table: KvPageTable::new(),
            pool: Pool::new(capacity),
            flat: HashMap::new(),
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
            .map(|r| r.end.min(old_mapped).saturating_sub(r.start.min(old_mapped)))
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
        let freed = self.table.release_working_set(ws);
        self.pool.recycle_after_epoch(freed, epoch);
        self.flat.remove(&ws);
    }

    /// Release a WorkingSet, RETAINING its path as a prefix-cache root when
    /// it carries canonical content (page 0 committed with a page hash). The
    /// lease keeps the pages reachable for CAS matches; pressure reclaims
    /// them via the contention ladder's rung 1, and the FIFO cap
    /// (`max_roots`) bounds steady-state retention. Non-canonical paths
    /// release exactly like [`Self::release_working_set`].
    pub fn release_working_set_cached(&mut self, ws: WorkingSetId, epoch: u64, max_roots: usize) {
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
        self.release_working_set(ws, epoch);
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
        let mapped = self.table.mapped_len(ws)?;
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

        let mut in_place: Vec<u64> = Vec::new();
        let mut cow_start: Option<u64> = None;
        for &index in indexes.iter().filter(|&&i| i < mapped) {
            if self.table.privately_writable(ws, index)? {
                in_place.push(index);
            } else {
                cow_start = Some(cow_start.map_or(index, |c| c.min(index)));
            }
        }
        // A private write inside the rebased region rides the CoW instead:
        // its in-place result would be shadowed by the copied page.
        if let Some(cs) = cow_start {
            in_place.retain(|&i| i < cs);
        }

        let cow_count = cow_start.map_or(0, |cs| (mapped - cs) as usize);
        // Resolve current ids before allocating so no failure path leaks ids.
        let in_place_ids: Vec<PhysicalKvPageId> = in_place
            .iter()
            .map(|&i| self.table.lookup(ws, i))
            .collect::<Result<_, _>>()?;
        let cow_srcs: Vec<PhysicalKvPageId> = match cow_start {
            Some(cs) => (cs..mapped)
                .map(|i| self.table.lookup(ws, i))
                .collect::<Result<_, _>>()?,
            None => Vec::new(),
        };

        let need = cow_count + fresh.len();
        let allocated = self
            .pool
            .try_alloc_n(need)
            .ok_or(KvStoreError::OutOfPages {
                requested: need,
                available: self.pool.available(),
            })?;

        let mut targets = Vec::with_capacity(in_place.len() + need);
        for (&index, &dst) in in_place.iter().zip(&in_place_ids) {
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
            let state = last.unwrap_or_else(|| self.next_opaque_hash());
            self.table.set_chain_state(ws, Some(state))?;
        }

        self.invalidate_flat(ws);
        let freed = self.table.collect();
        self.pool.recycle_after_epoch(freed, epoch);

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

        self.in_flight = self.in_flight.saturating_sub(1);
        Ok(())
    }

    /// Failure, poison, or readiness dummy-run: release pending slots and
    /// metadata; the committed mapping and hashes remain authoritative.
    pub fn abort(&mut self, prepared: KvPreparedWrite, epoch: u64) {
        if let Some(terminal) = prepared.pinned {
            self.table.unpin(terminal);
        }
        self.pool.recycle_after_epoch(prepared.allocated, epoch);
        let freed = self.table.collect();
        self.pool.recycle_after_epoch(freed, epoch);
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
        Ok((entry.version, entry.cache.as_deref().expect("just populated")))
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
        let last = hashes.iter().rposition(|h| h.is_some()).map_or(0, |i| i + 1);
        Ok((mapped - 1) * page_size as u64 + last as u64)
    }

    /// Validated CAS lookup: a canonical full page's boundary chain value ->
    /// its live trie location. Entries whose location no longer carries that
    /// content (owner compaction moved locals, collection freed the node)
    /// are pruned and miss.
    pub fn lookup_cached_page(&mut self, key: &Hash256) -> Option<(NodeId, u64)> {
        let entry = *self.cas.get(key)?;
        if self.table.node_page_last_slot_hash(entry.node, entry.local) == Some(*key) {
            Some((entry.node, entry.local))
        } else {
            self.cas.remove(key);
            None
        }
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
        let freed = self.table.collect();
        self.pool.recycle_after_epoch(freed, epoch);
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
        let freed = self.table.collect();
        let count = freed.len();
        self.pool.recycle_after_epoch(freed, epoch);
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

    /// Contention-ladder rung 2 victim sizing: pages reachable only from
    /// `ws`'s terminal (its private trie suffix) — what releasing this
    /// WorkingSet would actually free.
    pub fn exclusive_footprint(&self, ws: WorkingSetId) -> Result<u64, KvStoreError> {
        Ok(self.table.exclusive_footprint(ws)?)
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

    #[cfg(test)]
    pub(crate) fn table(&mut self) -> &mut KvPageTable {
        &mut self.table
    }
}

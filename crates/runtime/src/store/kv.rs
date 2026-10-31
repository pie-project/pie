pub mod hash;
pub mod journal;
pub mod page_table;
pub mod working_set;
pub mod write;

#[cfg(test)]
mod tests;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use ::engine::transfer::MemoryDomain;
use hash::Hash256;
use page_table::{
    DiskKvSlotId, HostKvSlotId, KeptPath, KvPageBacking, KvPageTable, KvTableError, NodeId,
    PhysicalKvPageId, PublishedPage, ReclaimQuote, TriePageLocation, WorkingSetId,
};
use write::{KvPreparedWrite, PageCommit, PreparedTarget};

use crate::inferlet::process::ProcessId;
use crate::policy::retention::Keys;
use crate::policy::{By, Event};
use crate::rt::Instant;
use crate::store::pool::Pool;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvStoreError {
    #[error(transparent)]
    Table(#[from] KvTableError),
    #[error("kv pool exhausted: requested {requested}, available {available}")]
    OutOfPages { requested: usize, available: usize },
    #[error("invalid write set: {reason}")]
    BadWriteSet { reason: &'static str },
    #[error("commit metadata does not match the prepared targets")]
    CommitMismatch,
    #[error("allocation grant size mismatch: required {required}, granted {granted}")]
    GrantMismatch { required: usize, granted: usize },
    #[error("host and disk KV swap exhausted: requested {requested}, available {available}")]
    HostSwapFull { requested: usize, available: usize },
    #[error("another residency move holds pages of these working sets")]
    MoveInFlight,
}

#[derive(Debug)]
struct WriteClassification {
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

/// A suspended or restored page: its trie location, the device page it
/// leaves or lands in, and its `Swapped` or `OnDisk` backing.
#[derive(Debug)]
pub struct SwapPage {
    pub location: TriePageLocation,
    pub gpu_id: PhysicalKvPageId,
    pub swapped: KvPageBacking,
}

#[derive(Debug)]
pub struct KvResidencyTxn {
    working_sets: HashSet<WorkingSetId>,
    pages: Vec<SwapPage>,
    pinned: Vec<NodeId>,
}

impl KvResidencyTxn {
    pub fn copy_plans(&self) -> Vec<(MemoryDomain, Vec<u32>, Vec<u32>)> {
        let mut plans: Vec<(MemoryDomain, Vec<u32>, Vec<u32>)> = Vec::new();
        for page in &self.pages {
            let (tier, slot) = match page.swapped {
                KvPageBacking::Swapped(slot) => (MemoryDomain::HostPinned, slot.0),
                KvPageBacking::OnDisk(slot) => (MemoryDomain::LocalDisk, slot.0),
                KvPageBacking::Resident(_) => {
                    unreachable!("a swap page is backed below the device")
                }
            };
            match plans.iter_mut().find(|(at, ..)| *at == tier) {
                Some((_, gpu, slots)) => {
                    gpu.push(page.gpu_id.0);
                    slots.push(slot);
                }
                None => plans.push((tier, vec![page.gpu_id.0], vec![slot])),
            }
        }
        plans
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}

pub struct KvStore {
    table: KvPageTable,
    pool: Pool<PhysicalKvPageId>,
    host_pool: Pool<HostKvSlotId>,
    disk_pool: Pool<DiskKvSlotId>,
    /// `None` while some are off the device.
    flat: HashMap<WorkingSetId, Option<Vec<PhysicalKvPageId>>>,
    domain: Hash256,
    cache: HashMap<u64, KeptEntry>,
    /// Entries already split at a step, by the entry and the pages kept.
    splits: HashSet<(u64, u64)>,
    owners: HashMap<ProcessId, usize>,
    /// Not on a model with recurrent state, whose KV alone resumes nothing.
    keeps: bool,
    kept: u64,
    page_size: u32,
    log_seq: u64,
    seq: u64,
    outstanding: BTreeSet<u64>,
    moved: Arc<tokio::sync::Notify>,
    /// A persistent disk tier's journal, and the slots it names.
    persisted: Option<Persisted>,
    /// How many leading tokens each owner's props ask to keep on disk
    /// (`persist-prefix-tokens`, `persist-system`): told to the cluster with
    /// the runs it leaves, until it owns none.
    hints: HashMap<ProcessId, u64>,
}

/// A persistent disk tier: a copy on disk of sealed pages, by key. A page
/// is immutable once sealed, so a copy serves every place its key is found.
/// A copy beside a page held on the device or the host costs nothing there;
/// a page settled on disk is backed by its copy.
struct Persisted {
    journal: journal::Journal,
    /// Each copy's page, by its slot.
    slots: HashMap<DiskKvSlotId, Hash256>,
    by_key: HashMap<Hash256, DiskKvSlotId>,
    /// The copies pages are backed by now, with how many places each backs.
    backing: HashMap<DiskKvSlotId, u32>,
    /// The cluster retention key each copy is kept under: swap takes the
    /// lowest first.
    retention: HashMap<DiskKvSlotId, i64>,
}

impl Persisted {
    fn back(&mut self, slot: DiskKvSlotId) {
        *self.backing.entry(slot).or_insert(0) += 1;
    }

    /// A place backed by `slot` lets it go: true when `slot` is a copy, which
    /// stays.
    fn let_go(&mut self, slot: DiskKvSlotId) -> bool {
        if !self.slots.contains_key(&slot) {
            return false;
        }
        if let Some(count) = self.backing.get_mut(&slot) {
            *count -= 1;
            if *count == 0 {
                self.backing.remove(&slot);
            }
        }
        true
    }
}

/// Persisting a run: its resident pages, each copied to a fresh disk slot
/// while it stays where it is.
#[derive(Debug)]
pub struct KvPersistTxn {
    pages: Vec<(
        TriePageLocation,
        PhysicalKvPageId,
        DiskKvSlotId,
        journal::PageRecord,
    )>,
    pinned: NodeId,
}

impl KvPersistTxn {
    /// The device pages to copy, and the disk slots they go to.
    pub fn copy_plan(&self) -> (Vec<u32>, Vec<u32>) {
        self.pages
            .iter()
            .map(|(_, gpu, slot, _)| (gpu.0, slot.0))
            .unzip()
    }
}

type HeldPage = (DiskKvSlotId, journal::PageRecord);

/// The pages a journal names, as runs to import in order, each after the
/// keys of the pages before it (from token 0); and the slots of pages that
/// follow on from no page held, or do not span a page, which nothing can
/// take back.
#[allow(clippy::type_complexity)]
fn runs(
    mut held: HashMap<Hash256, HeldPage>,
    page_size: usize,
) -> (Vec<(Vec<Hash256>, Vec<HeldPage>)>, Vec<DiskKvSlotId>) {
    let mut orphans: Vec<DiskKvSlotId> = Vec::new();
    held.retain(|_, (slot, record)| {
        let whole = record.tokens.len() == page_size;
        if !whole {
            orphans.push(*slot);
        }
        whole
    });
    let mut children: HashMap<Option<Hash256>, Vec<Hash256>> = HashMap::new();
    for (page, (_, record)) in &held {
        children.entry(record.parent).or_default().push(*page);
    }
    for pages in children.values_mut() {
        pages.sort_unstable();
    }
    let mut out = Vec::new();
    let mut stack: Vec<(Vec<Hash256>, Hash256)> = children
        .get(&None)
        .into_iter()
        .flatten()
        .rev()
        .map(|&page| (Vec::new(), page))
        .collect();
    while let Some((prefix, first)) = stack.pop() {
        let mut run = Vec::new();
        let mut at = first;
        while let Some(page) = held.remove(&at) {
            run.push(page);
            match children.get(&Some(at)).map(Vec::as_slice) {
                Some([only]) => at = *only,
                Some(next) => {
                    let mut keys = prefix.clone();
                    keys.extend(run.iter().map(|(_, record)| record.page));
                    stack.extend(next.iter().rev().map(|&page| (keys.clone(), page)));
                    break;
                }
                None => break,
            }
        }
        if !run.is_empty() {
            out.push((prefix, run));
        }
    }
    orphans.extend(held.into_values().map(|(slot, _)| slot));
    (out, orphans)
}

#[derive(Debug)]
#[must_use]
pub struct ReaderPin {
    nodes: Vec<NodeId>,
}

#[derive(Debug, Clone, Copy)]
struct KeptEntry {
    path: KeptPath,
    owner: Option<ProcessId>,
    last_use: Instant,
    /// Reclaimed before prompts among equals: an answer is shared less than a
    /// prompt.
    decoded: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct CachedWorkingSet {
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub owner: ProcessId,
    pub entries: usize,
    pub pages: u64,
    pub last_use: Instant,
}

impl KvStore {
    #[cfg(test)]
    pub fn new(capacity: u32, seed: Hash256) -> Self {
        Self::new_with_swap(capacity, 0, 0, seed)
    }

    pub fn new_with_swap(
        capacity: u32,
        host_capacity: u32,
        disk_capacity: u32,
        seed: Hash256,
    ) -> Self {
        Self {
            table: KvPageTable::new(),
            pool: Pool::new(capacity),
            host_pool: Pool::new(host_capacity),
            disk_pool: Pool::new(disk_capacity),
            flat: HashMap::new(),
            domain: hash::cache_domain(&seed),
            cache: HashMap::new(),
            splits: HashSet::new(),
            owners: HashMap::new(),
            keeps: false,
            kept: 0,
            page_size: 0,
            log_seq: 0,
            seq: 0,
            outstanding: BTreeSet::new(),
            moved: Arc::new(tokio::sync::Notify::new()),
            persisted: None,
            hints: HashMap::new(),
        }
    }

    pub fn set_domain(&mut self, domain: Hash256) {
        self.domain = domain;
    }

    /// `Some(None)` at slot 0; `None` when that slot is opaque and no hash
    /// names what follows.
    pub fn chain_root(
        &self,
        ws: WorkingSetId,
        start: u64,
        page_size: u32,
    ) -> Result<Option<Option<Hash256>>, KvStoreError> {
        let page = u64::from(page_size.max(1));
        let Some(last) = start.checked_sub(1) else {
            return Ok(Some(None));
        };
        let hashes = self.table.page_token_hashes(ws, last / page)?;
        Ok(hashes
            .get((last % page) as usize)
            .copied()
            .flatten()
            .map(Some))
    }

    #[cfg(test)]
    pub fn slot_hashes(
        &self,
        ws: WorkingSetId,
        start: u64,
        tokens: &[u32],
        page_size: u32,
    ) -> Result<Option<Vec<Hash256>>, KvStoreError> {
        Ok(self
            .chain_root(ws, start, page_size)?
            .map(|prev| hash::chain_slots(&self.domain, prev, start, tokens)))
    }

    /// A page whose every slot is hashed seals, its hash naming it and all its
    /// prefix.
    pub fn seal_rows(
        &mut self,
        ws: WorkingSetId,
        start: u64,
        slots: &[Hash256],
        page_size: u32,
    ) -> Result<u32, KvStoreError> {
        let page = u64::from(page_size.max(1));
        if slots.is_empty() {
            return Ok(0);
        }
        let end = start + slots.len() as u64;
        let first = start / page;
        let mut pages = Vec::new();
        let mut sealed = 0;
        for index in first..end.div_ceil(page) {
            let mut hashes = self.table.page_token_hashes(ws, index)?;
            hashes.resize(page as usize, None);
            for at in (index * page).max(start)..((index + 1) * page).min(end) {
                hashes[(at % page) as usize] = Some(slots[(at - start) as usize]);
            }
            let page_hash = hashes
                .iter()
                .all(Option::is_some)
                .then(|| hash::page_hash(&hashes));
            sealed += u32::from(page_hash.is_some());
            pages.push((hashes, page_hash));
        }
        self.table.commit_hashes(ws, first, pages)?;
        Ok(sealed)
    }

    /// `None` when an earlier page is not sealed: no key names what follows.
    pub fn page_keys(
        &self,
        ws: WorkingSetId,
        start: u64,
        slots: &[Hash256],
        page_size: u32,
    ) -> Result<Option<Vec<Hash256>>, KvStoreError> {
        let page = u64::from(page_size.max(1));
        let mut keys = Vec::new();
        for index in 0..start / page {
            match self.table.page_hash_at(ws, index)? {
                Some(key) => keys.push(key),
                None => return Ok(None),
            }
        }
        let mut partial = if start.is_multiple_of(page) {
            Vec::new()
        } else {
            let mut held = self.table.page_token_hashes(ws, start / page)?;
            held.resize((start % page) as usize, None);
            if held.iter().any(Option::is_none) {
                return Ok(None);
            }
            held
        };
        for &slot in slots {
            partial.push(Some(slot));
            if partial.len() as u64 == page {
                keys.push(hash::page_hash(&partial));
                partial.clear();
            }
        }
        Ok(Some(keys))
    }

    pub fn cached_run(&self, keys: &[Hash256]) -> Vec<TriePageLocation> {
        self.table.sealed_run(keys, true)
    }

    pub fn cached_run_any_tier(&self, keys: &[Hash256]) -> Vec<TriePageLocation> {
        self.table.sealed_run(keys, false)
    }

    #[cfg(test)]
    pub fn sealed_page_at(
        &self,
        location: TriePageLocation,
        key: Hash256,
    ) -> Option<PhysicalKvPageId> {
        self.table.sealed_page_at(location, key)
    }

    #[cfg(test)]
    pub fn sealed_in(
        &self,
        ws: WorkingSetId,
        written: std::ops::Range<u64>,
    ) -> Result<Vec<u64>, KvStoreError> {
        let mapped = self.table.mapped_len(ws)?;
        let mut sealed = Vec::new();
        for index in written.start..written.end.min(mapped) {
            if self.table.sealed(ws, index)? {
                sealed.push(index);
            }
        }
        Ok(sealed)
    }

    pub fn domain(&self) -> Hash256 {
        self.domain
    }

    pub fn current_epoch(&self) -> u64 {
        self.seq
    }

    pub fn retire_idle(&mut self) {
        let through = self
            .outstanding
            .first()
            .map_or(self.seq, |&oldest| oldest.saturating_sub(1));
        self.pool.retire_through(through);
    }

    pub fn create_working_set(&mut self) -> WorkingSetId {
        let ws = self.table.create_working_set();
        self.flat.insert(ws, Some(Vec::new()));
        ws
    }

    fn release_entry(&mut self, entry: KeptEntry, epoch: u64) -> usize {
        let freed = self.table.release_cache_root(entry.path.terminal);
        // Device pages: a host or disk slot given back seats no one.
        let count = freed
            .iter()
            .filter(|backing| matches!(backing, KvPageBacking::Resident(_)))
            .count();
        self.recycle_backings(freed, epoch);
        self.retire_idle();
        count
    }

    fn disown(&mut self, entry: &KeptEntry, by: By) {
        let Some(owner) = entry.owner else {
            return;
        };
        if let Some(count) = self.owners.get_mut(&owner) {
            *count -= 1;
            if *count == 0 {
                self.owners.remove(&owner);
                self.hints.remove(&owner);
                crate::policy::report(Event::CacheEmptied(owner, by));
            }
        }
    }

    pub fn cached_working_sets(&self) -> Vec<CachedWorkingSet> {
        let mut sets: HashMap<ProcessId, CachedWorkingSet> = HashMap::new();
        for entry in self.cache.values() {
            let Some(owner) = entry.owner else {
                continue;
            };
            let pages = entry.path.pages;
            sets.entry(owner)
                .and_modify(|set| {
                    set.entries += 1;
                    set.pages = set.pages.max(pages);
                    set.last_use = set.last_use.max(entry.last_use);
                })
                .or_insert(CachedWorkingSet {
                    owner,
                    entries: 1,
                    pages,
                    last_use: entry.last_use,
                });
        }
        sets.into_values().collect()
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn owns_cache(&self, owner: ProcessId) -> bool {
        self.owners.contains_key(&owner)
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn discard_cache(&mut self, owner: ProcessId, by: By) -> usize {
        let keys: Vec<u64> = self
            .cache
            .iter()
            .filter(|(_, entry)| entry.owner == Some(owner))
            .map(|(&key, _)| key)
            .collect();
        self.drop_entries(keys, by, self.current_epoch())
    }

    fn drop_entries(&mut self, keys: Vec<u64>, by: By, epoch: u64) -> usize {
        let mut freed = 0;
        for key in keys {
            if let Some(entry) = self.cache.remove(&key) {
                self.splits.retain(|&(split, _)| split != key);
                self.disown(&entry, by);
                freed += self.release_entry(entry, epoch);
            }
        }
        self.forget_lost_copies();
        freed
    }

    /// Only what no live working set reads, lowest key first, then least
    /// recently used.
    /// Pages the disk tier keeps a copy of settle there instead, at no cost,
    /// and their entries stay.
    pub fn reclaim_cache(&mut self, epoch: u64, keys: impl Keys) -> usize {
        self.split_at_steps(&keys);
        self.split_persisted();
        let mut freed = 0;
        for (key, ids) in self.reclaim_order(&keys) {
            let before = freed;
            freed += self.settle_on_disk(&ids, key, &keys, epoch);
            if freed == before {
                let ids = ids
                    .into_iter()
                    .filter(|id| !self.cache.get(id).is_some_and(|e| self.persisted_whole(e)))
                    .collect();
                freed += self.drop_entries(ids, By::Host, epoch);
            }
            if freed >= (self.pool.capacity() as usize / 64).max(1) {
                break;
            }
        }
        tracing::debug!(target: "pie::kv", freed, "reclaimed");
        freed
    }

    /// The terminals of entries whose key is above `key`: what they cover
    /// stays with them.
    fn above(&self, key: i64, keys: &impl Keys) -> Vec<NodeId> {
        self.cache
            .values()
            .filter(|entry| keys.key_at(entry.owner, self.key_token(entry.path.pages)) > key)
            .map(|entry| entry.path.terminal)
            .collect()
    }

    /// Moves the device pages only `ids` read, that the disk tier keeps a
    /// copy of, onto their copies: no copy is made. Answers the device
    /// pages freed.
    fn settle_on_disk(&mut self, ids: &[u64], key: i64, keys: &impl Keys, epoch: u64) -> usize {
        if self.persisted.as_ref().is_none_or(|p| p.by_key.is_empty()) {
            return 0;
        }
        let above = self.above(key, keys);
        let moves: Vec<(TriePageLocation, PhysicalKvPageId, DiskKvSlotId)> = self
            .table
            .cache_only_pages(&self.roots(ids), &above)
            .into_iter()
            .filter_map(|(location, gpu)| Some((location, gpu, self.copy_of(&location)?)))
            .collect();
        if moves.is_empty() {
            return 0;
        }
        let replacements: Vec<_> = moves
            .iter()
            .map(|&(location, gpu, slot)| {
                (
                    location,
                    KvPageBacking::Resident(gpu),
                    KvPageBacking::OnDisk(slot),
                )
            })
            .collect();
        if self.table.replace_backings(&replacements).is_err() {
            return 0;
        }
        if let Some(persisted) = &mut self.persisted {
            moves.iter().for_each(|&(_, _, slot)| persisted.back(slot));
        }
        let freed = moves.len();
        self.pool
            .recycle_after_epoch(moves.into_iter().map(|(_, gpu, _)| gpu).collect(), epoch);
        tracing::debug!(target: "pie::kv", pages = freed, "settled on disk");
        freed
    }

    /// The token an entry's key is read at: its last page's first, where a
    /// page of KV alone resumes a sequence, else (a checkpoint) its last.
    fn key_token(&self, pages: u64) -> u64 {
        let page = u64::from(self.page_size);
        if self.keeps {
            pages.saturating_sub(1) * page
        } else {
            (pages * page).saturating_sub(1)
        }
    }

    /// Keeps the head of each entry its owner's keys step down within, as an
    /// entry of its own, so the tail past it can go alone.
    fn split_at_steps(&mut self, keys: &impl Keys) {
        let page = u64::from(self.page_size.max(1));
        if !self.keeps {
            return;
        }
        let mut heads = Vec::new();
        for (&id, entry) in &self.cache {
            let Some(owner) = entry.owner else {
                continue;
            };
            for upto in keys.steps(owner) {
                let pages = upto.div_ceil(page);
                if pages > 0 && pages < entry.path.pages && !self.splits.contains(&(id, pages)) {
                    heads.push((id, *entry, pages));
                }
            }
        }
        self.keep_heads(heads);
    }

    /// Keeps the head of each entry whose first pages the disk tier keeps a
    /// copy of, as an entry of its own, so that the tail can go alone while
    /// the head settles on disk.
    fn split_persisted(&mut self) {
        let Some(persisted) = &self.persisted else {
            return;
        };
        if persisted.by_key.is_empty() || !self.keeps {
            return;
        }
        let mut heads = Vec::new();
        for (&id, entry) in &self.cache {
            let Some(keys) = self.table.path_keys(entry.path.terminal) else {
                continue;
            };
            let pages = keys
                .iter()
                .take(entry.path.pages as usize)
                .take_while(|key| persisted.by_key.contains_key(*key))
                .count() as u64;
            if pages > 0 && pages < entry.path.pages && !self.splits.contains(&(id, pages)) {
                heads.push((id, *entry, pages));
            }
        }
        self.keep_heads(heads);
    }

    /// Keeps each entry's first `pages` as an entry of its own.
    fn keep_heads(&mut self, heads: Vec<(u64, KeptEntry, u64)>) {
        for (id, entry, pages) in heads {
            self.splits.insert((id, pages));
            let path = self.table.keep_prefix_of(entry.path, pages);
            self.table.lease_cache_root(path.terminal);
            if let Some(owner) = entry.owner {
                *self.owners.entry(owner).or_insert(0) += 1;
            }
            self.kept += 1;
            self.cache.insert(self.kept, KeptEntry { path, ..entry });
        }
    }

    /// An owner's answers before its prompts; groups that free no device page
    /// go last, and one wholly below the device never.
    fn reclaim_order(&self, keys: &impl Keys) -> Vec<(i64, Vec<u64>)> {
        let unused: HashSet<NodeId> = self.table.unused_cache_roots().into_iter().collect();
        if unused.is_empty() {
            return Vec::new();
        }
        type Group = (Vec<u64>, Instant);
        let mut groups: HashMap<(Option<ProcessId>, i64, bool), Group> = HashMap::new();
        for (&id, entry) in &self.cache {
            let root = entry.path.terminal;
            if !unused.contains(&root)
                || self.table.in_move_at(root)
                || !self.table.holds_resident(root)
            {
                continue;
            }
            let key = keys.key_at(entry.owner, self.key_token(entry.path.pages));
            let group = groups
                .entry((entry.owner, key, entry.decoded))
                .or_insert_with(|| (Vec::new(), entry.last_use));
            group.0.push(id);
            group.1 = group.1.max(entry.last_use);
        }
        let mut order: Vec<_> = groups
            .into_iter()
            .map(|((_, key, decoded), (ids, last_use))| {
                let frees = self.table.frees_device_pages(&self.roots(&ids));
                ((!frees, key, !decoded, last_use), (key, ids))
            })
            .collect();
        order.sort_unstable_by_key(|(order, _)| *order);
        order.into_iter().map(|(_, group)| group).collect()
    }

    fn roots(&self, keys: &[u64]) -> Vec<NodeId> {
        keys.iter()
            .filter_map(|key| Some(self.cache.get(key)?.path.terminal))
            .collect()
    }

    pub fn prepare_demote(
        &mut self,
        room: usize,
        keys: impl Keys,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        let room = room.min(self.host_pool.available());
        if room == 0 {
            return Ok(None);
        }
        self.split_at_steps(&keys);
        self.split_persisted();
        for (key, ids) in self.reclaim_order(&keys) {
            // What a higher key also covers stays with it.
            let above = self.above(key, &keys);
            let mut pages = self.table.cache_only_pages(&self.roots(&ids), &above);
            // A page with a copy on disk settles there when its room is
            // wanted, at no cost: host room is for the rest.
            pages.retain(|(location, _)| self.copy_of(location).is_none());
            if pages.is_empty() {
                continue;
            }
            pages.truncate(room);
            tracing::debug!(target: "pie::kv", pages = pages.len(), "demote");
            return Ok(Some(self.prepare_move_out(&HashSet::new(), pages)?));
        }
        Ok(None)
    }

    pub fn below_device(&self, run: &[TriePageLocation]) -> usize {
        run.iter()
            .filter(|location| {
                !matches!(
                    self.table.backing_at(location),
                    Ok(KvPageBacking::Resident(_))
                )
            })
            .count()
    }

    /// The run stays held, after the commit too, until
    /// [`Self::release_reader`]: what brought it up maps it in first.
    pub fn prepare_promote(
        &mut self,
        run: &[TriePageLocation],
        granted: &mut Vec<PhysicalKvPageId>,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        let Some(last) = run.last() else {
            return Ok(None);
        };
        let mut below = Vec::new();
        for &location in run {
            if self.table.page_location_pinned(location) {
                return Ok(None);
            }
            match self.table.backing_at(&location)? {
                KvPageBacking::Resident(_) => {}
                swapped => below.push((location, swapped)),
            }
        }
        if below.is_empty() {
            return Ok(None);
        }
        if granted.len() < below.len() {
            return Err(KvStoreError::GrantMismatch {
                required: below.len(),
                granted: granted.len(),
            });
        }
        self.table.pin(last.node);
        let gpu: Vec<PhysicalKvPageId> = granted.drain(..below.len()).collect();
        let pages: Vec<SwapPage> = below
            .into_iter()
            .zip(gpu)
            .map(|((location, swapped), gpu_id)| SwapPage {
                location,
                gpu_id,
                swapped,
            })
            .collect();
        self.table
            .pin_swap_locations(pages.iter().map(|page| page.location));
        Ok(Some(KvResidencyTxn {
            working_sets: HashSet::new(),
            pages,
            pinned: vec![last.node],
        }))
    }

    pub fn commit_promote(&mut self, mut txn: KvResidencyTxn) -> Result<ReaderPin, KvStoreError> {
        let pinned = std::mem::take(&mut txn.pinned);
        match self.commit_restore(txn) {
            Ok(_) => Ok(ReaderPin { nodes: pinned }),
            Err(error) => {
                self.release_reader(ReaderPin { nodes: pinned });
                Err(error)
            }
        }
    }

    /// Each cache entry's pages by key, from token 0: the runs this worker
    /// can give whole.
    pub fn cached_runs(&self) -> Vec<Vec<Hash256>> {
        self.cache
            .values()
            .filter_map(|entry| {
                let mut keys = self.table.path_keys(entry.path.terminal)?;
                keys.truncate(entry.path.pages as usize);
                (!keys.is_empty()).then_some(keys)
            })
            .collect()
    }

    /// Every sealed page's key here, on any tier.
    pub fn sealed_keys(&self) -> std::collections::HashSet<Hash256> {
        self.table.sealed_keys()
    }

    /// Host slots to land pages in that come from another worker; given
    /// back with [`Self::free_host_slots`] unless an import takes them.
    /// The tokens written to `ws`, as far as every write since is known.
    pub fn logged(&self, ws: WorkingSetId) -> Result<&[u32], KvStoreError> {
        Ok(self.table.logged(ws)?)
    }

    pub fn alloc_host_slots(&mut self, n: usize) -> Option<Vec<HostKvSlotId>> {
        self.host_pool.try_alloc_n(n)
    }

    /// Gives reserved backings back to the pools they came from.
    fn release_backings(&mut self, backings: Vec<KvPageBacking>) {
        let (mut device, mut host, mut disk) = (Vec::new(), Vec::new(), Vec::new());
        for backing in backings {
            match backing {
                KvPageBacking::Resident(page) => device.push(page),
                KvPageBacking::Swapped(slot) => host.push(slot),
                KvPageBacking::OnDisk(slot) => disk.push(slot),
            }
        }
        self.pool.release_reserved(device);
        self.host_pool.release_reserved(host);
        self.release_disk(disk);
    }

    pub fn free_host_slots(&mut self, slots: Vec<HostKvSlotId>) {
        self.host_pool.release_reserved(slots);
    }

    /// Pages another worker sealed, kept here as one cache entry in the host
    /// tier: `slots` already hold their bytes. A pass whose rows they cover
    /// promotes them as it would pages this worker demoted. They extend the
    /// sealed pages `prefix` names, which must be found here whole; an empty
    /// `prefix` starts at token 0.
    pub fn import_cached(
        &mut self,
        prefix: &[Hash256],
        token_hashes: Vec<Vec<Option<Hash256>>>,
        page_hashes: Vec<Hash256>,
        slots: Vec<HostKvSlotId>,
    ) -> bool {
        self.import_cached_at(
            prefix,
            token_hashes,
            page_hashes,
            slots.into_iter().map(KvPageBacking::Swapped).collect(),
        )
    }

    /// [`Self::import_cached`] for pages that sit anywhere: host slots, or
    /// device pages another worker's were read into. The backings are the
    /// caller's reservations, given back if the prefix went.
    pub fn import_cached_at(
        &mut self,
        prefix: &[Hash256],
        token_hashes: Vec<Vec<Option<Hash256>>>,
        page_hashes: Vec<Hash256>,
        backings: Vec<KvPageBacking>,
    ) -> bool {
        debug_assert_eq!(token_hashes.len(), backings.len());
        debug_assert_eq!(page_hashes.len(), backings.len());
        if backings.is_empty() {
            return true;
        }
        let after = if prefix.is_empty() {
            None
        } else {
            let run = self.table.sealed_run(prefix, false);
            if run.len() != prefix.len() {
                self.release_backings(backings);
                return false;
            }
            run.last().copied()
        };
        let node = self
            .table
            .insert_cached(after, backings, token_hashes, page_hashes);
        self.table.lease_cache_root(node);
        // An entry's pages are its path's, from token 0: the prefix it was
        // grafted below as well as its own.
        let pages = self.table.path_len(node);
        self.kept += 1;
        tracing::debug!(target: "pie::kv", pages, "imported");
        self.cache.insert(
            self.kept,
            KeptEntry {
                path: KeptPath {
                    terminal: node,
                    pages,
                },
                owner: None,
                last_use: Instant::now(),
                decoded: false,
            },
        );
        true
    }

    /// The tokens' hashes of every page of `run`, page after page: what a
    /// worker that keeps the run without computing it needs to go on from
    /// it. `None` when a page of it is not hashed whole.
    pub fn run_slots(&self, run: &[TriePageLocation]) -> Option<Vec<Hash256>> {
        let mut slots = Vec::new();
        for location in run {
            for hash in self.table.token_hashes_at(location)? {
                slots.push((*hash)?);
            }
        }
        Some(slots)
    }

    /// A run of sealed pages, held so it is neither moved nor reclaimed while
    /// its bytes are read out, with each page's device id; `None` unless all
    /// of it is on the device.
    pub fn pin_resident(
        &mut self,
        run: &[TriePageLocation],
    ) -> Option<(ReaderPin, Vec<PhysicalKvPageId>)> {
        let last = run.last()?;
        let mut ids = Vec::with_capacity(run.len());
        for location in run {
            match self.table.backing_at(location).ok()? {
                KvPageBacking::Resident(id) => ids.push(id),
                _ => return None,
            }
        }
        self.table.pin(last.node);
        Some((
            ReaderPin {
                nodes: vec![last.node],
            },
            ids,
        ))
    }

    pub fn release_reader(&mut self, pin: ReaderPin) {
        let freed = self.table.unpin_terminals(&pin.nodes);
        let epoch = self.current_epoch();
        self.recycle_backings(freed, epoch);
        self.retire_idle();
    }

    /// A move that found its pages in another waits for it here.
    pub fn moves_ended(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.moved)
    }

    pub fn fork(&mut self, ws: WorkingSetId) -> Result<WorkingSetId, KvStoreError> {
        let child = self.table.fork(ws)?;
        self.refresh_flat(child);
        Ok(child)
    }

    pub fn forget(
        &mut self,
        ws: WorkingSetId,
        pages: Range<u64>,
        start: u64,
    ) -> Result<(), KvStoreError> {
        Ok(self.table.forget(ws, pages, start)?)
    }

    pub fn forgotten(
        &self,
        ws: WorkingSetId,
    ) -> Result<std::collections::BTreeSet<u64>, KvStoreError> {
        Ok(self.table.forgotten(ws)?.clone())
    }

    pub fn reserve(&mut self, ws: WorkingSetId, pages: u64) -> Result<Range<u64>, KvStoreError> {
        Ok(self.table.reserve(ws, pages)?)
    }

    pub fn backing_demand(&self, ws: WorkingSetId, end: u64) -> Result<usize, KvStoreError> {
        let page_len = self.table.page_len(ws)?;
        if end > page_len {
            return Err(KvStoreError::BadWriteSet {
                reason: "backing frontier exceeds the logical reservation",
            });
        }
        let mapped = self.table.mapped_len(ws)?;
        usize::try_from(end.saturating_sub(mapped)).map_err(|_| KvStoreError::OutOfPages {
            requested: usize::MAX,
            available: self.pool.available(),
        })
    }

    pub fn ensure_backed_reserved(
        &mut self,
        ws: WorkingSetId,
        end: u64,
        granted: &mut Vec<PhysicalKvPageId>,
    ) -> Result<usize, KvStoreError> {
        let count = self.backing_demand(ws, end)?;
        if granted.len() < count {
            return Err(KvStoreError::GrantMismatch {
                required: count,
                granted: granted.len(),
            });
        }
        if count == 0 {
            return Ok(0);
        }
        let allocated: Vec<_> = granted.drain(..count).collect();
        let published = allocated
            .iter()
            .copied()
            .map(|id| PublishedPage {
                id,
                token_hashes: Vec::new(),
                page_hash: None,
            })
            .collect();
        if let Err(error) = self.table.publish_appended(ws, published) {
            self.pool.release_reserved(allocated);
            return Err(error.into());
        }
        self.refresh_flat(ws);
        Ok(count)
    }

    pub fn privately_writable(&self, ws: WorkingSetId, index: u64) -> Result<bool, KvStoreError> {
        Ok(self.table.privately_writable(ws, index)?)
    }

    pub fn release_working_set(&mut self, ws: WorkingSetId, epoch: u64) {
        if self.keeps {
            self.keep_sealed_prefix(ws);
        }
        // A hint is for what its owner leaves in the cache: one that left
        // nothing needs none.
        if let Some(owner) = self.owner_of(ws)
            && !self.owners.contains_key(&owner)
        {
            self.hints.remove(&owner);
        }
        let freed = self.table.release_working_set_backings(ws);
        self.recycle_backings(freed, epoch);
        self.flat.remove(&ws);
    }

    fn keep_sealed_prefix(&mut self, ws: WorkingSetId) {
        let owner = self.owner_of(ws);
        let all = self.table.sealed_resident_prefix(ws).unwrap_or(0);
        let prompt = self.table.prompt_pages(ws).unwrap_or(0).min(all);
        if prompt > 0 {
            self.keep(ws, prompt, owner, false);
        }
        if all > prompt {
            self.keep(ws, all, owner, true);
        }
    }

    fn keep(&mut self, ws: WorkingSetId, pages: u64, owner: Option<ProcessId>, decoded: bool) {
        let Ok(path) = self.table.keep_prefix(ws, pages) else {
            return;
        };
        self.table.lease_cache_root(path.terminal);
        if let Some(owner) = owner {
            *self.owners.entry(owner).or_insert(0) += 1;
        }
        self.kept += 1;
        tracing::debug!(target: "pie::kv", pages, decoded, "kept");
        if let Some(owner) = owner {
            crate::policy::report(Event::Kept(owner, pages * u64::from(self.page_size)));
        }
        self.cache.insert(
            self.kept,
            KeptEntry {
                path,
                owner,
                last_use: Instant::now(),
                decoded,
            },
        );
    }

    pub fn owner_of(&self, ws: WorkingSetId) -> Option<ProcessId> {
        self.table
            .owner(ws)
            .ok()
            .flatten()
            .map(crate::policy::schedule::plex_id)
    }

    pub fn keep_checkpoint(&mut self, ws: WorkingSetId, pages: u64, owner: Option<ProcessId>) {
        if self
            .table
            .sealed_resident_prefix(ws)
            .is_ok_and(|sealed| sealed >= pages)
        {
            self.keep(ws, pages, owner, false);
        }
    }

    pub fn set_keeps(&mut self, keeps: bool) {
        self.keeps = keeps;
    }

    pub fn set_page_size(&mut self, page_size: u32) {
        self.page_size = page_size;
    }

    pub fn log_rows(
        &mut self,
        ws: WorkingSetId,
        start: u64,
        tokens: &[u32],
    ) -> Result<(), KvStoreError> {
        Ok(self.table.log_rows(ws, start, tokens)?)
    }

    pub fn log_begin(&mut self, ws: WorkingSetId, cut: u64) -> Result<u64, KvStoreError> {
        self.log_seq += 1;
        self.table.log_begin(ws, self.log_seq, cut)?;
        Ok(self.log_seq)
    }

    pub fn log_end(
        &mut self,
        ws: WorkingSetId,
        seq: u64,
        rows: Option<(u64, &[u32])>,
    ) -> Result<(), KvStoreError> {
        Ok(self.table.log_end(ws, seq, rows)?)
    }

    /// `decoded` when a decode wrote them, which the cache lets go before a
    /// prompt.
    pub fn fold(&mut self, ws: WorkingSetId, to: u64, decoded: bool) -> Result<u64, KvStoreError> {
        let folded = self.table.freeze(ws, to)?;
        let page = u64::from(self.page_size);
        if page == 0 {
            return Ok(0);
        }
        let sealed = self.table.sealed_prefix(ws)?;
        let logged = self.table.logged(ws)?;
        let full = (logged.len() as u64)
            .min(folded)
            .div_euclid(page)
            .min(self.table.mapped_len(ws)?);
        if full <= sealed {
            return Ok(0);
        }
        let start = sealed * page;
        let Some(prev) = self.chain_root(ws, start, self.page_size)? else {
            return Ok(0);
        };
        let tokens = logged[start as usize..(full * page) as usize].to_vec();
        let slots = hash::chain_slots(&self.domain, prev, start, &tokens);
        self.seal_rows(ws, start, &slots, self.page_size)?;
        if !decoded {
            self.table.reach_prompt(ws, full)?;
        }
        tracing::debug!(target: "pie::kv", pages = full - sealed, decoded, "sealed at a fold");
        Ok(full - sealed)
    }

    pub fn folded(&self, ws: WorkingSetId) -> Result<u64, KvStoreError> {
        Ok(self.table.folded(ws)?)
    }

    pub fn keeps(&self) -> bool {
        self.keeps
    }

    pub fn set_owner(&mut self, ws: WorkingSetId, owner: ProcessId) -> Result<(), KvStoreError> {
        Ok(self.table.set_owner(ws, owner)?)
    }

    pub fn supplied_tokens(&self, ws: WorkingSetId) -> Result<u64, KvStoreError> {
        Ok(self.table.supplied(ws)?)
    }

    /// Maps in only when the pages `ws` holds are the run's first.
    pub fn attach_run(
        &mut self,
        ws: WorkingSetId,
        run: &[TriePageLocation],
        limit: u64,
        page_size: u32,
    ) -> Result<u64, KvStoreError> {
        let held = self.table.mapped_len(ws)?;
        let found = run.len();
        let run = &run[..run.len().min(limit as usize)];
        let run = &run[..self.table.resident_run(run)];
        // What attaching maps is the trie path to the run's last page; with
        // several copies of one content (a node start's copies, one moved
        // below the device), that path may leave the run: only as far as
        // they agree.
        let run = match run.last() {
            Some(last) => {
                let path = self.table.path_locations(last.node, last.local);
                let agree = path.iter().zip(run).take_while(|(a, b)| a == b).count();
                &run[..agree]
            }
            None => run,
        };
        tracing::debug!(target: "pie::kv", found, limit, resident = run.len(), held, "attach");
        let Some(&last) = run.last().filter(|_| run.len() as u64 > held) else {
            return Ok(0);
        };
        if self.table.path_prefix_len(last.node, last.local) != run.len() as u64 {
            return Ok(0);
        }
        match self.table.attach(ws, last.node, last.local) {
            Ok(_) => {}
            Err(KvTableError::BadRange { .. }) => return Ok(0),
            Err(error) => return Err(error.into()),
        }
        let pages = run.len() as u64 - held;
        self.table
            .add_supplied(ws, pages * u64::from(page_size.max(1)))?;
        // What was mapped in is sealed, so frozen: a prompt's.
        self.table
            .freeze(ws, run.len() as u64 * u64::from(page_size.max(1)))?;
        self.table.reach_prompt(ws, run.len() as u64)?;
        self.refresh_flat(ws);
        // The cache's entries that were used whole go last; one that holds
        // more than was used ages by what it holds alone.
        let now = Instant::now();
        for entry in self.cache.values_mut() {
            if self
                .table
                .within(entry.path.terminal, last, run.len() as u64)
            {
                entry.last_use = now;
            }
        }
        Ok(pages)
    }

    fn recycle_backings(&mut self, backings: Vec<KvPageBacking>, epoch: u64) {
        let mut resident = Vec::new();
        let mut swapped = Vec::new();
        for backing in backings {
            match backing {
                KvPageBacking::Resident(id) => resident.push(id),
                swapped_backing => swapped.push(swapped_backing),
            }
        }
        self.pool.recycle_after_epoch(resident, epoch);
        self.release_swapped(swapped);
    }

    fn release_swapped(&mut self, backings: impl IntoIterator<Item = KvPageBacking>) {
        let (mut host, mut disk) = (Vec::new(), Vec::new());
        for backing in backings {
            match backing {
                KvPageBacking::Swapped(slot) => host.push(slot),
                KvPageBacking::OnDisk(slot) => disk.push(slot),
                KvPageBacking::Resident(_) => unreachable!("released below the device"),
            }
        }
        self.host_pool.release_reserved(host);
        self.release_disk(disk);
    }

    /// Gives disk slots back; a persistent tier's copies stay, for the pages
    /// they hold (given up by [`Self::forget_lost_copies`] once no page is).
    fn release_disk(&mut self, slots: Vec<DiskKvSlotId>) {
        let slots = match &mut self.persisted {
            Some(persisted) => slots
                .into_iter()
                .filter(|&slot| !persisted.let_go(slot))
                .collect(),
            None => slots,
        };
        self.disk_pool.release_reserved(slots);
    }

    /// Gives up copies no page is backed by, in the journal, durably, then
    /// gives their slots back.
    fn give_up_copies(&mut self, slots: Vec<DiskKvSlotId>) {
        let Some(persisted) = &mut self.persisted else {
            return;
        };
        let slots: Vec<DiskKvSlotId> = slots
            .into_iter()
            .filter(|slot| !persisted.backing.contains_key(slot))
            .filter(|slot| match persisted.slots.remove(slot) {
                Some(key) => {
                    persisted.by_key.remove(&key);
                    persisted.retention.remove(slot);
                    true
                }
                None => false,
            })
            .collect();
        if slots.is_empty() {
            return;
        }
        let named: Vec<u32> = slots.iter().map(|slot| slot.0).collect();
        if let Err(error) = persisted.journal.give_up(&named) {
            tracing::error!(target: "pie::kv", %error, "disk kv journal not written; the tier is kept no longer");
            self.abandon_persisted();
        }
        self.disk_pool.release_reserved(slots);
    }

    /// Gives up the copies of pages no longer held anywhere here.
    fn forget_lost_copies(&mut self) {
        let Some(persisted) = &self.persisted else {
            return;
        };
        if persisted.slots.is_empty() {
            return;
        }
        let live = self.table.sealed_keys();
        let lost: Vec<DiskKvSlotId> = persisted
            .slots
            .iter()
            .filter(|(_, key)| !live.contains(*key))
            .map(|(&slot, _)| slot)
            .collect();
        self.give_up_copies(lost);
    }

    /// Stops keeping the disk tier, and removes its journal so that the
    /// next run trusts none of its slots. Copies no page is backed by are
    /// free again; the rest go back as their pages let them go.
    fn abandon_persisted(&mut self) {
        let Some(persisted) = self.persisted.take() else {
            return;
        };
        let spare: Vec<DiskKvSlotId> = persisted
            .slots
            .keys()
            .filter(|slot| !persisted.backing.contains_key(slot))
            .copied()
            .collect();
        if let Err(error) = persisted.journal.abandon() {
            tracing::error!(target: "pie::kv", %error, "disk kv journal not removed");
        }
        self.disk_pool.release_reserved(spare);
    }

    /// The copy of the page at `location`, if the disk tier keeps one.
    fn copy_of(&self, location: &TriePageLocation) -> Option<DiskKvSlotId> {
        let persisted = self.persisted.as_ref()?;
        persisted
            .by_key
            .get(&self.table.page_key_at(location)?)
            .copied()
    }

    /// Whether every page of `entry` has a copy on disk.
    fn persisted_whole(&self, entry: &KeptEntry) -> bool {
        let Some(persisted) = &self.persisted else {
            return false;
        };
        let pages = entry.path.pages as usize;
        pages > 0
            && !persisted.by_key.is_empty()
            && self
                .table
                .path_keys(entry.path.terminal)
                .is_some_and(|keys| {
                    keys.len() >= pages
                        && keys[..pages]
                            .iter()
                            .all(|key| persisted.by_key.contains_key(key))
                })
    }

    /// Keeps the disk tier for the next run, its journal at `path`, and
    /// takes back what the last run left there, as ownerless cache entries
    /// whose pages are on disk: a hit promotes them as it would pages this
    /// run demoted. `kept`: the engine found its slot files as it left
    /// them. Answers the pages taken back.
    pub fn keep_disk(&mut self, path: &std::path::Path, kept: bool) -> std::io::Result<usize> {
        if !self.keeps {
            tracing::warn!(target: "pie::kv", "a model with recurrent state or a window resumes from kv alone nowhere; the disk tier is not kept");
            return Ok(0);
        }
        let header = journal::Header {
            domain: self.domain,
            page_size: self.page_size,
            slots: self.disk_pool.capacity(),
        };
        let (mut journal, held) = journal::Journal::open(path, header, kept)?;
        let mut pages: HashMap<Hash256, HeldPage> = HashMap::new();
        let mut doubled = Vec::new();
        for (slot, record) in held {
            let slot = DiskKvSlotId(slot);
            if pages.contains_key(&record.page) || !self.disk_pool.take(slot) {
                doubled.push(slot.0);
            } else {
                pages.insert(record.page, (slot, record));
            }
        }
        if !doubled.is_empty() {
            journal.give_up(&doubled)?;
        }
        let mut persisted = Persisted {
            journal,
            slots: HashMap::new(),
            by_key: HashMap::new(),
            backing: HashMap::new(),
            retention: HashMap::new(),
        };
        for (key, (slot, record)) in &pages {
            persisted.slots.insert(*slot, *key);
            persisted.by_key.insert(*key, *slot);
            persisted.retention.insert(*slot, record.retention);
        }
        self.persisted = Some(persisted);
        let (runs, _) = runs(pages, self.page_size as usize);
        let mut taken = 0;
        for (prefix, run) in runs {
            let (mut tokens, mut keys, mut backings) = (Vec::new(), Vec::new(), Vec::new());
            for (slot, record) in run {
                keys.push(record.page);
                tokens.push(record.tokens);
                backings.push(KvPageBacking::OnDisk(slot));
            }
            let slots: Vec<DiskKvSlotId> = backings
                .iter()
                .map(|backing| match backing {
                    KvPageBacking::OnDisk(slot) => *slot,
                    _ => unreachable!("taken back on disk"),
                })
                .collect();
            if let Some(persisted) = &mut self.persisted {
                slots.iter().for_each(|&slot| persisted.back(slot));
            }
            let len = keys.len();
            if self.import_cached_at(&prefix, tokens, keys, backings) {
                taken += len;
            }
        }
        // Pages that followed on from nothing, or did not take: no page
        // holds their copies.
        self.forget_lost_copies();
        tracing::info!(target: "pie::kv", pages = taken, "disk kv tier kept; pages taken back");
        Ok(taken)
    }

    pub fn write_demand(
        &mut self,
        ws: WorkingSetId,
        write_indexes: &[u64],
    ) -> Result<usize, KvStoreError> {
        Ok(self.classify_write(ws, write_indexes)?.required_pages())
    }

    pub fn prepare_write_reserved(
        &mut self,
        ws: WorkingSetId,
        write_indexes: &[u64],
        granted: &mut Vec<PhysicalKvPageId>,
    ) -> Result<KvPreparedWrite, KvStoreError> {
        let classification = self.classify_write(ws, write_indexes)?;
        let required = classification.required_pages();
        if granted.len() < required {
            return Err(KvStoreError::GrantMismatch {
                required,
                granted: granted.len(),
            });
        }
        let allocated = granted.drain(..required).collect();
        self.finish_prepare_write(ws, classification, allocated)
    }

    fn classify_write(
        &mut self,
        ws: WorkingSetId,
        write_indexes: &[u64],
    ) -> Result<WriteClassification, KvStoreError> {
        let visible = self.flat_table(ws)?.to_vec();
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
            // A sealed page never changes in place: a write copies it.
            if self.table.privately_writable(ws, index)? && !self.table.sealed(ws, index)? {
                in_place.push((index, visible[index as usize]));
            } else {
                cow_start = Some(cow_start.map_or(index, |c| c.min(index)));
            }
        }
        if let Some(cs) = cow_start {
            in_place.retain(|&(i, _)| i < cs);
        }

        let cow_srcs: Vec<PhysicalKvPageId> = match cow_start {
            Some(cs) => visible[cs as usize..mapped as usize].to_vec(),
            None => Vec::new(),
        };
        Ok(WriteClassification {
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

        self.seq += 1;
        self.outstanding.insert(self.seq);
        Ok(KvPreparedWrite {
            ws,
            targets,
            allocated,
            cow_start,
            seq: self.seq,
        })
    }

    pub fn publish_prepared(
        &mut self,
        prepared: KvPreparedWrite,
        commits: &[PageCommit],
    ) -> Result<u64, KvStoreError> {
        if commits.len() != prepared.targets.len() {
            self.cancel_prepared(prepared);
            return Err(KvStoreError::CommitMismatch);
        }
        let ws = prepared.ws;
        let seq = prepared.seq;
        let mut freed = Vec::new();

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
        const fn changes_mapping(target: &PreparedTarget) -> bool {
            matches!(
                target,
                PreparedTarget::Cow { .. } | PreparedTarget::Fresh { .. }
            )
        }
        let mapping_changed = prepared.targets.iter().any(changes_mapping);
        match prepared.cow_start {
            Some(cs) => freed.extend(self.table.replace_tail(ws, cs, pages)?),
            None if !pages.is_empty() => self.table.publish_appended(ws, pages)?,
            None => {}
        }

        if mapping_changed {
            self.refresh_flat(ws);
        }
        self.recycle_backings(freed, seq);

        Ok(seq)
    }

    pub fn cancel_prepared(&mut self, prepared: KvPreparedWrite) {
        self.pool
            .recycle_after_epoch(prepared.allocated, prepared.seq);
        self.outstanding.remove(&prepared.seq);
        self.retire_idle();
    }

    pub fn settle(&mut self, seq: u64) {
        self.outstanding.remove(&seq);
        self.retire_idle();
    }

    pub fn flat_table(&mut self, ws: WorkingSetId) -> Result<&[PhysicalKvPageId], KvStoreError> {
        if !self.flat.get(&ws).is_some_and(Option::is_some) {
            let flat = self.table.flatten(ws)?;
            self.flat.insert(ws, Some(flat));
        }
        Ok(self.flat[&ws].as_deref().expect("just populated"))
    }

    fn refresh_flat(&mut self, ws: WorkingSetId) {
        self.flat.insert(ws, self.table.flatten(ws).ok());
    }

    pub fn page_hash_at(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Option<Hash256>, KvStoreError> {
        Ok(self.table.page_hash_at(ws, index)?)
    }

    pub fn page_token_hashes(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Vec<Option<Hash256>>, KvStoreError> {
        Ok(self.table.page_token_hashes(ws, index)?)
    }

    pub fn reclaimable_cache_pages(&self) -> u32 {
        self.table.unused_cache_lease_pages()
    }

    pub fn reclaim_quotes(
        &self,
        groups: &[HashSet<WorkingSetId>],
        budget: u32,
    ) -> Vec<ReclaimQuote> {
        self.table.reclaim_quotes(groups, budget)
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
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        let mut pages = self.table.movable_resident_pages(working_sets)?;
        // Host room is for what is paused before what is cached.
        if self.host_pool.available() < pages.len() {
            self.drop_entries_off_device();
            // Dropping can compact the pages just listed: list them again.
            pages = self.table.movable_resident_pages(working_sets)?;
        }
        if pages.is_empty() {
            return Ok(None);
        }
        let available = self.make_swap_room(pages.len());
        if available < pages.len() {
            return Err(KvStoreError::HostSwapFull {
                requested: pages.len(),
                available,
            });
        }
        Ok(Some(self.prepare_move_out(working_sets, pages)?))
    }

    /// Its readers are suspended already and stay so.
    pub fn prepare_sweep(
        &mut self,
        keep: &HashSet<WorkingSetId>,
        room: usize,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        let mut pages = self.table.stranded_pages(keep);
        if self.swap_available() < pages.len().min(room) {
            self.drop_entries_off_device();
            // Dropping can compact the pages just listed: list them again.
            pages = self.table.stranded_pages(keep);
        }
        let available = self.make_swap_room(pages.len().min(room));
        pages.truncate(room.min(available));
        if pages.is_empty() {
            return Ok(None);
        }
        // Only the readers of what moves: one whose pages all stayed may be
        // restored while this is out.
        let readers = self.table.suspended_readers(&pages);
        Ok(Some(self.prepare_move_out(&readers, pages)?))
    }

    fn swap_available(&self) -> usize {
        self.host_pool.available() + self.disk_swap_available()
    }

    /// Frees swap slots for `wanted` pages, giving up the disk tier's copies
    /// if it must: suspended kv cannot be computed again, so it wins. Answers
    /// the slots free.
    fn make_swap_room(&mut self, wanted: usize) -> usize {
        let free = self.host_pool.available() + self.disk_pool.available();
        if free < wanted {
            self.yield_disk(wanted - free);
        }
        self.host_pool.available() + self.disk_pool.available()
    }

    /// Copies whose pages are held on the device or the host, lowest
    /// retention key first; and the unused entries kept only on disk, at
    /// the lowest key among their pages, then least recently used.
    fn disk_yield_plan(&self) -> (Vec<DiskKvSlotId>, Vec<(Instant, u64)>) {
        let Some(persisted) = &self.persisted else {
            return (Vec::new(), Vec::new());
        };
        let key = |slot: &DiskKvSlotId| persisted.retention.get(slot).copied().unwrap_or(0);
        let mut spare: Vec<DiskKvSlotId> = persisted
            .slots
            .keys()
            .filter(|slot| !persisted.backing.contains_key(slot))
            .copied()
            .collect();
        spare.sort_unstable_by_key(|slot| (key(slot), slot.0));
        if persisted.backing.is_empty() {
            return (spare, Vec::new());
        }
        let unused: HashSet<NodeId> = self.table.unused_cache_roots().into_iter().collect();
        // An entry goes at the lowest key among its pages' copies.
        let lowest = |entry: &KeptEntry| {
            self.table
                .path_keys(entry.path.terminal)
                .into_iter()
                .flatten()
                .take(entry.path.pages as usize)
                .filter_map(|page| persisted.by_key.get(&page))
                .map(key)
                .min()
                .unwrap_or(0)
        };
        let mut whole: Vec<(i64, Instant, u64)> = self
            .cache
            .iter()
            .filter(|(_, entry)| {
                let root = entry.path.terminal;
                unused.contains(&root)
                    && !self.table.holds_resident(root)
                    && !self.table.in_move_at(root)
                    && self.persisted_whole(entry)
            })
            .map(|(&id, entry)| (lowest(entry), entry.last_use, id))
            .collect();
        whole.sort_unstable();
        (
            spare,
            whole.into_iter().map(|(_, at, id)| (at, id)).collect(),
        )
    }

    /// Gives up the disk tier's copies until `wanted` slots are free:
    /// [`Self::disk_yield_plan`] in order.
    fn yield_disk(&mut self, wanted: usize) {
        let (spare, whole) = self.disk_yield_plan();
        let start = self.disk_pool.available();
        let freed = |store: &Self| store.disk_pool.available().saturating_sub(start);
        let spare: Vec<DiskKvSlotId> = spare.into_iter().take(wanted).collect();
        self.give_up_copies(spare);
        for (_, id) in whole {
            if freed(self) >= wanted {
                break;
            }
            self.drop_entries(vec![id], By::Host, self.current_epoch());
        }
        tracing::debug!(target: "pie::kv", wanted, freed = freed(self), "disk copies given up for swap");
    }

    /// Persists the sealed run `keys` (from token 0) as far as its pages
    /// are found here, in a row, held on the device or copied already: a
    /// copy of each resident page goes to a fresh disk slot while the page
    /// stays where it is. `None` when nothing is left to copy, or the disk
    /// tier is not kept or full; never at swap's cost.
    pub fn prepare_persist(&mut self, keys: &[Hash256], retention: i64) -> Option<KvPersistTxn> {
        let run = self.table.sealed_run(keys, false);
        self.raise_retention(keys, &run, retention);
        let persisted = self.persisted.as_ref()?;
        let mut wanted = Vec::new();
        for (i, location) in run.iter().enumerate() {
            if persisted.by_key.contains_key(&keys[i]) {
                continue;
            }
            match self.table.backing_at(location) {
                Ok(KvPageBacking::Resident(gpu)) if !self.table.page_location_pinned(*location) => {
                    let tokens = self.table.token_hashes_at(location)?.to_vec();
                    wanted.push((
                        *location,
                        gpu,
                        journal::PageRecord {
                            page: keys[i],
                            parent: i.checked_sub(1).map(|p| keys[p]),
                            tokens,
                            retention,
                        },
                    ));
                }
                // A copy is made from the device: the run ends at a page
                // that is not there.
                _ => break,
            }
        }
        wanted.truncate(self.disk_pool.available());
        if wanted.is_empty() {
            return None;
        }
        let slots = self.disk_pool.try_alloc_n(wanted.len())?;
        // The last page's node holds the whole run while it is copied.
        let pinned = wanted.last()?.0.node;
        self.table.pin(pinned);
        self.table
            .pin_swap_locations(wanted.iter().map(|(location, ..)| *location));
        Some(KvPersistTxn {
            pages: wanted
                .into_iter()
                .zip(slots)
                .map(|((location, gpu, record), slot)| (location, gpu, slot, record))
                .collect(),
            pinned,
        })
    }

    /// Keeps the copies `run` already has under `retention` where that is
    /// higher than the key they were kept under, in the journal too.
    fn raise_retention(&mut self, keys: &[Hash256], run: &[TriePageLocation], retention: i64) {
        let Some(persisted) = &mut self.persisted else {
            return;
        };
        let mut raised = Vec::new();
        for (i, location) in run.iter().enumerate() {
            let Some(&slot) = persisted.by_key.get(&keys[i]) else {
                continue;
            };
            if persisted
                .retention
                .get(&slot)
                .is_some_and(|&kept| kept >= retention)
            {
                continue;
            }
            let Some(tokens) = self.table.token_hashes_at(location) else {
                continue;
            };
            persisted.retention.insert(slot, retention);
            raised.push((
                slot,
                journal::PageRecord {
                    page: keys[i],
                    parent: i.checked_sub(1).map(|p| keys[p]),
                    tokens: tokens.to_vec(),
                    retention,
                },
            ));
        }
        if raised.is_empty() {
            return;
        }
        let written = raised
            .iter()
            .try_for_each(|(slot, record)| persisted.journal.put(slot.0, record))
            .and_then(|()| persisted.journal.sync());
        if let Err(error) = written {
            tracing::error!(target: "pie::kv", %error, "disk kv journal not written; the tier is kept no longer");
            self.abandon_persisted();
        }
    }

    /// The copies landed: the journal names them, durably. Answers the pages
    /// persisted.
    pub fn commit_persist(&mut self, txn: KvPersistTxn) -> usize {
        self.unpin_persist(&txn);
        let Some(persisted) = &mut self.persisted else {
            self.disk_pool
                .release_reserved(txn.pages.into_iter().map(|(.., slot, _)| slot).collect());
            return 0;
        };
        let written = txn
            .pages
            .iter()
            .try_for_each(|(_, _, slot, record)| persisted.journal.put(slot.0, record))
            .and_then(|()| persisted.journal.sync());
        if let Err(error) = written {
            tracing::error!(target: "pie::kv", %error, "disk kv journal not written; the tier is kept no longer");
            self.disk_pool
                .release_reserved(txn.pages.into_iter().map(|(.., slot, _)| slot).collect());
            self.abandon_persisted();
            return 0;
        }
        let count = txn.pages.len();
        for (_, _, slot, record) in txn.pages {
            persisted.slots.insert(slot, record.page);
            persisted.by_key.insert(record.page, slot);
            persisted.retention.insert(slot, record.retention);
        }
        // A page dropped while it was copied leaves its copy to no one.
        self.forget_lost_copies();
        // The prefix persisted is a run of its own: the cluster sees it as a
        // group on disk.
        self.split_persisted();
        tracing::debug!(target: "pie::kv", pages = count, "persisted");
        count
    }

    /// The copies did not land: their slots go back.
    pub fn abort_persist(&mut self, txn: KvPersistTxn) {
        self.unpin_persist(&txn);
        self.disk_pool
            .release_reserved(txn.pages.into_iter().map(|(.., slot, _)| slot).collect());
    }

    fn unpin_persist(&mut self, txn: &KvPersistTxn) {
        let mut freed = self.table.unpin(txn.pinned);
        freed.extend(
            self.table
                .unpin_swap_locations(txn.pages.iter().map(|(location, ..)| *location)),
        );
        self.end_move(freed);
    }

    /// `owner`'s props ask `tokens` leading tokens kept on disk, or none.
    pub fn set_persist_hint(&mut self, owner: ProcessId, tokens: Option<u64>) {
        match tokens {
            Some(tokens) if tokens > 0 => {
                self.hints.insert(owner, tokens);
            }
            _ => {
                self.hints.remove(&owner);
            }
        }
    }

    /// The runs whose owner asked a prefix of them kept on disk, by the
    /// run's last key, with the tokens asked.
    pub fn persist_hints(&self) -> Vec<(Hash256, u64)> {
        if self.hints.is_empty() {
            return Vec::new();
        }
        self.cache
            .values()
            .filter_map(|entry| {
                let tokens = *self.hints.get(&entry.owner?)?;
                let keys = self.table.path_keys(entry.path.terminal)?;
                let last = *keys.get(usize::try_from(entry.path.pages).ok()?.checked_sub(1)?)?;
                Some((last, tokens))
            })
            .collect()
    }

    /// The pages the disk tier keeps a copy of, by key.
    pub fn persisted_keys(&self) -> Vec<Hash256> {
        self.persisted
            .as_ref()
            .map(|p| p.by_key.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Whether the disk tier is kept across runs.
    pub fn disk_persistent(&self) -> bool {
        self.persisted.is_some()
    }

    /// Pages the disk tier keeps a copy of.
    #[cfg(test)]
    pub fn persisted_pages(&self) -> usize {
        self.persisted.as_ref().map_or(0, |p| p.slots.len())
    }

    /// No lookup hits them, and no one will bring them back.
    fn drop_entries_off_device(&mut self) {
        let unused: HashSet<NodeId> = self.table.unused_cache_roots().into_iter().collect();
        let keys: Vec<u64> = self
            .cache
            .iter()
            .filter(|(_, entry)| {
                unused.contains(&entry.path.terminal)
                    && !self.table.settled_on_device(entry.path.terminal)
                    && !self.persisted_whole(entry)
            })
            .map(|(&key, _)| key)
            .collect();
        tracing::debug!(target: "pie::kv", entries = keys.len(), "dropped off device");
        self.drop_entries(keys, By::Host, self.current_epoch());
    }

    pub fn mark_suspended(&mut self, working_sets: &HashSet<WorkingSetId>) {
        self.table.set_suspended(working_sets, true);
    }

    fn prepare_move_out(
        &mut self,
        working_sets: &HashSet<WorkingSetId>,
        pages: Vec<(TriePageLocation, PhysicalKvPageId)>,
    ) -> Result<KvResidencyTxn, KvStoreError> {
        // Host first; what host cannot take goes to disk.
        let on_host = self.host_pool.available().min(pages.len());
        let host = self.host_pool.try_alloc_n(on_host).expect("counted above");
        let disk = self
            .disk_pool
            .try_alloc_n(pages.len() - on_host)
            .expect("counted above");
        let swapped: Vec<KvPageBacking> = host
            .into_iter()
            .map(KvPageBacking::Swapped)
            .chain(disk.into_iter().map(KvPageBacking::OnDisk))
            .collect();
        let pinned = match self.table.pin_working_sets(working_sets) {
            Ok(pinned) => pinned,
            Err(error) => {
                self.release_swapped(swapped);
                return Err(error.into());
            }
        };
        let pages: Vec<SwapPage> = pages
            .into_iter()
            .zip(swapped)
            .map(|((location, gpu_id), swapped)| SwapPage {
                location,
                gpu_id,
                swapped,
            })
            .collect();
        self.table
            .pin_swap_locations(pages.iter().map(|page| page.location));
        Ok(KvResidencyTxn {
            working_sets: working_sets.clone(),
            pages,
            pinned,
        })
    }

    fn unpin_move(&mut self, txn: &KvResidencyTxn) -> Vec<KvPageBacking> {
        let mut freed = self.table.unpin_terminals(&txn.pinned);
        freed.extend(
            self.table
                .unpin_swap_locations(txn.pages.iter().map(|page| page.location)),
        );
        freed
    }

    fn end_move(&mut self, freed: Vec<KvPageBacking>) {
        self.recycle_backings(freed, self.current_epoch());
        self.retire_idle();
        self.moved.notify_waiters();
    }

    pub fn commit_suspend(&mut self, txn: KvResidencyTxn) -> Result<usize, KvStoreError> {
        let replacements: Vec<_> = txn
            .pages
            .iter()
            .map(|page| {
                (
                    page.location,
                    KvPageBacking::Resident(page.gpu_id),
                    page.swapped,
                )
            })
            .collect();
        if let Err(error) = self.table.replace_backings(&replacements) {
            self.abort_suspend(txn);
            return Err(error.into());
        }
        self.table.set_suspended(&txn.working_sets, true);
        let freed = self.unpin_move(&txn);
        for &ws in &txn.working_sets {
            self.refresh_flat(ws);
        }
        let count = txn.pages.len();
        self.pool.recycle_after_epoch(
            txn.pages.into_iter().map(|page| page.gpu_id).collect(),
            self.current_epoch(),
        );
        self.end_move(freed);
        Ok(count)
    }

    pub fn abort_suspend(&mut self, txn: KvResidencyTxn) {
        let freed = self.unpin_move(&txn);
        self.release_swapped(txn.pages.into_iter().map(|page| page.swapped));
        self.end_move(freed);
    }

    pub fn swapped_page_count(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<usize, KvStoreError> {
        Ok(self.table.swapped_pages(working_sets)?.len())
    }

    pub fn held_page_count(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<usize, KvStoreError> {
        Ok(self.table.held_pages(working_sets)?)
    }

    pub fn prepare_restore(
        &mut self,
        working_sets: &HashSet<WorkingSetId>,
        granted: &mut Vec<PhysicalKvPageId>,
    ) -> Result<KvResidencyTxn, KvStoreError> {
        if self.table.in_move(working_sets)? {
            return Err(KvStoreError::MoveInFlight);
        }
        let swapped = self.table.swapped_pages(working_sets)?;
        if granted.len() < swapped.len() {
            return Err(KvStoreError::GrantMismatch {
                required: swapped.len(),
                granted: granted.len(),
            });
        }
        let pinned = self.table.pin_working_sets(working_sets)?;
        let gpu_pages: Vec<PhysicalKvPageId> = granted.drain(..swapped.len()).collect();
        let pages: Vec<SwapPage> = swapped
            .into_iter()
            .zip(gpu_pages)
            .map(|((location, swapped), gpu_id)| SwapPage {
                location,
                gpu_id,
                swapped,
            })
            .collect();
        self.table
            .pin_swap_locations(pages.iter().map(|page| page.location));
        Ok(KvResidencyTxn {
            working_sets: working_sets.clone(),
            pages,
            pinned,
        })
    }

    pub fn commit_restore(&mut self, txn: KvResidencyTxn) -> Result<usize, KvStoreError> {
        let replacements: Vec<_> = txn
            .pages
            .iter()
            .map(|page| {
                (
                    page.location,
                    page.swapped,
                    KvPageBacking::Resident(page.gpu_id),
                )
            })
            .collect();
        if let Err(error) = self.table.replace_backings(&replacements) {
            self.abort_restore(txn);
            return Err(error.into());
        }
        self.table.set_suspended(&txn.working_sets, false);
        let freed = self.unpin_move(&txn);
        for &ws in &txn.working_sets {
            self.refresh_flat(ws);
        }
        let count = txn.pages.len();
        self.release_swapped(txn.pages.into_iter().map(|page| page.swapped));
        self.end_move(freed);
        Ok(count)
    }

    pub fn abort_restore(&mut self, txn: KvResidencyTxn) {
        let freed = self.unpin_move(&txn);
        self.pool
            .release_reserved(txn.pages.into_iter().map(|page| page.gpu_id).collect());
        self.end_move(freed);
    }

    #[cfg(test)]
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

    pub fn host_swap_available(&self) -> usize {
        self.host_pool.available()
    }

    pub fn host_swap_capacity(&self) -> u32 {
        self.host_pool.capacity()
    }

    /// Free disk slots, and those the disk tier's copies would give up for
    /// swap.
    pub fn disk_swap_available(&self) -> usize {
        let (spare, whole) = self.disk_yield_plan();
        let mut backing: HashSet<DiskKvSlotId> = HashSet::new();
        if let Some(persisted) = &self.persisted {
            for (_, id) in &whole {
                let Some(entry) = self.cache.get(id) else {
                    continue;
                };
                let keys = self
                    .table
                    .path_keys(entry.path.terminal)
                    .unwrap_or_default();
                backing.extend(
                    keys.iter()
                        .take(entry.path.pages as usize)
                        .filter_map(|key| persisted.by_key.get(key)),
                );
            }
        }
        self.disk_pool.available() + spare.len() + backing.len()
    }

    pub fn disk_swap_capacity(&self) -> u32 {
        self.disk_pool.capacity()
    }

    #[cfg(test)]
    pub(crate) fn table(&mut self) -> &mut KvPageTable {
        &mut self.table
    }
}

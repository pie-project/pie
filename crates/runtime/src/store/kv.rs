pub mod hash;
pub mod journal;
pub mod page_table;
pub mod working_set;
pub mod write;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

/// Chosen pages a beam's finish copied because something else read them
/// too ([`KvStore::relist_beam_with`]).
pub static RELIST_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a beam search's finish needs first ([`KvStore::relist_beam_needs`]).
#[derive(Debug)]
pub enum RelistNeeds {
    /// The set is in a move: its end, told here.
    MoveEnd(Arc<tokio::sync::Notify>),
    /// The chosen pages something else reads too: each copied first onto
    /// a page the planner grants, as any shared page a write changes
    /// (none: go).
    Copies(Vec<PhysicalKvPageId>),
}

use ::engine::transfer::MemoryDomain;
use hash::Hash256;
use page_table::{
    DiskKvSlotId, HostKvSlotId, KeptPath, KvPageBacking, KvPageTable, KvTableError, NodeId,
    PhysicalKvPageId, PublishedPage, ReclaimQuote, TriePageLocation, WorkingSetId,
};
use write::{KvPreparedWrite, PageCommit, PreparedTarget};

use crate::inferlet::process::ProcessId;
use crate::policy::retention::{self, Keys};
use crate::policy::{By, Event};
use crate::rt::Instant;

#[cfg(test)]
type CacheMap = std::collections::BTreeMap<u64, KeptEntry>;
#[cfg(not(test))]
type CacheMap = HashMap<u64, KeptEntry>;
use crate::store::chunks::{ShapePool, ShapeRun};
use crate::store::pool::Pool;
use crate::store::slots::ShapeKind;

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
    /// The list shape the pages are slots of, where they are a shared
    /// pool's (CONTRACT-BC 22): the copy is the engine's `copy_slots`.
    shape: Option<::engine::chunk::ShapeId>,
    /// A demote's sealed pages, by place, that a persistent disk tier keeps
    /// as copies once they land on disk: the journal names them, so the next
    /// run takes them back.
    records: HashMap<TriePageLocation, journal::PageRecord>,
}

impl KvResidencyTxn {
    /// The list shape the pages are slots of, on a shared pool.
    pub fn shape(&self) -> Option<::engine::chunk::ShapeId> {
        self.shape
    }

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

/// Idle cache parked on the host going on down to disk (CONTRACT-BC 22:
/// a shared pool's kv list slots, HostPinned -> LocalDisk by `copy_slots`):
/// each page's host slot and the disk slot it lands in.
#[derive(Debug)]
pub struct KvSpillTxn {
    pages: Vec<(TriePageLocation, HostKvSlotId, DiskKvSlotId)>,
    shape: ::engine::chunk::ShapeId,
    /// The sealed pages a persistent disk tier keeps as copies once they
    /// land: journaled, so the next run takes them back.
    records: HashMap<TriePageLocation, journal::PageRecord>,
}

impl KvSpillTxn {
    /// The list shape the slots are of.
    pub fn shape(&self) -> ::engine::chunk::ShapeId {
        self.shape
    }

    /// The host slots to copy, and the disk slots they go to.
    pub fn copy_plan(&self) -> (Vec<u32>, Vec<u32>) {
        self.pages
            .iter()
            .map(|(_, host, disk)| (host.0, disk.0))
            .unzip()
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}

/// What a list compaction empties ([`KvStore::plan_list_compaction`]):
/// list chunks of the device whose every slot holds a page that may move,
/// the fewest pages first, and the processes whose sets read those pages:
/// those running, each fenced while its pages move (no fire of theirs
/// reads them then), and those parked (their sets suspended), whose
/// restore waits for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListCompaction {
    /// The list chunks it empties, logical.
    pub chunks: Vec<u32>,
    /// The pages it moves.
    pub pages: usize,
    /// The running processes whose sets read them.
    pub readers: Vec<ProcessId>,
    /// The parked processes whose suspended sets read them.
    pub parked: Vec<ProcessId>,
}

/// Who a list compaction's page may move for: a running set's process
/// (`suspended` false) fenced for it, a suspended set's process held from
/// its restore ([`KvStore::plan_list_compaction`]).
pub type MayMove<'a> = &'a dyn Fn(ProcessId, bool) -> bool;

/// A list chunk's pages and the processes reading them, running and
/// parked; `None` for a chunk one of whose pages stays.
type ChunkPages = Option<(
    Vec<(TriePageLocation, PhysicalKvPageId)>,
    BTreeSet<ProcessId>,
    BTreeSet<ProcessId>,
)>;

/// A list compaction under way ([`KvStore::prepare_relocate`]): each page,
/// its device slot and the slot it moves to, both of the device's list
/// shape. The pages are pinned as a move's until it commits or aborts.
#[derive(Debug)]
pub struct KvRelocateTxn {
    pages: Vec<(TriePageLocation, PhysicalKvPageId, PhysicalKvPageId)>,
    shape: Option<::engine::chunk::ShapeId>,
    /// The chunks it empties, logical.
    chunks: Vec<u32>,
}

impl KvRelocateTxn {
    /// The list shape the pages are slots of.
    pub fn shape(&self) -> Option<::engine::chunk::ShapeId> {
        self.shape
    }

    /// The device slots copied, and the device slots they go to.
    pub fn copy_plan(&self) -> (Vec<u32>, Vec<u32>) {
        self.pages
            .iter()
            .map(|(_, from, to)| (from.0, to.0))
            .unzip()
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// The chunks it empties, logical.
    pub fn chunks(&self) -> &[u32] {
        &self.chunks
    }
}

/// The kv pages held on every tier ([`KvStore::census`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvCensus {
    /// Pages held anywhere: the device's list slots out, the host tier's,
    /// and the disk's slots less the whole disk chunks other state holds
    /// (`guest_disk`). A page a move holds on two tiers counts twice while
    /// the move is under way: both slots are taken.
    pub held: u64,
    /// The disk slots whole disk chunks hold for parked guest parts
    /// ([`KvStore::take_disk_chunks`]).
    pub guest_disk: u64,
    /// Of `held`, the second slots of the pages moves hold on two tiers.
    pub moving: u64,
    /// Of those, the pages only idle cache holds, on every tier.
    pub idle: u64,
    /// The disk tier's slots.
    pub disk_slots: u32,
}

pub struct KvStore {
    table: KvPageTable,
    /// The list shape's slots, one a page.
    pool: ShapePool<PhysicalKvPageId>,
    /// Pages a waiter found short, past what is committed.
    short_pages: u32,
    /// Host slots: a pinned run's, or the slot refs of the list shape of
    /// an engine's host tier (CONTRACT-BC 22).
    host_pool: ShapePool<HostKvSlotId>,
    disk_pool: Pool<DiskKvSlotId>,
    /// The list shape the pages are slots of, where they are a shared
    /// pool's: tier moves name it.
    shape: Option<::engine::chunk::ShapeId>,
    /// `None` while some are off the device.
    flat: HashMap<WorkingSetId, Option<Vec<PhysicalKvPageId>>>,
    /// The model's domain ([`state::identity::ModelIdentity::domain`]):
    /// what every set's own domain is derived under, and what the disk
    /// journal and snapshots are checked against.
    domain: Hash256,
    /// Each live set's domain, and the scope it asked if it did
    /// ([`crate::policy::share`]).
    scopes: HashMap<WorkingSetId, SetScope>,
    /// Random each run: what a set that asked nothing keys under.
    private_seed: Hash256,
    private_nonce: u64,
    /// By entry id. Under test in id order (the order entries were kept),
    /// so a seeded simulation reclaims the same entries on each run.
    cache: CacheMap,
    /// The cache's entries by the node their last pages are in: an attach
    /// ages the entries on its own path, not every entry.
    by_owner: HashMap<NodeId, Vec<u64>>,
    /// Entries already split at a step, by the entry and the pages kept.
    splits: HashSet<(u64, u64)>,
    owners: HashMap<ProcessId, Vec<u64>>,
    /// Not on a model with recurrent state, whose KV alone resumes nothing.
    keeps: bool,
    /// The readers outside the fleet, and who is told as one ends.
    readers: usize,
    reader_signal: Option<super::slots::RoomSignal>,
    kept: u64,
    page_size: u32,
    log_seq: u64,
    seq: u64,
    outstanding: BTreeSet<u64>,
    moved: Arc<tokio::sync::Notify>,
    /// Told when pages may have joined the cache: a fire's pages settle, or
    /// a working set is released and keeps its sealed prefix.
    cached: Arc<tokio::sync::Notify>,
    /// Told when an entry joins or leaves the cache: what a lookup of a
    /// waiting prompt's prefix finds may have changed.
    changed: Arc<tokio::sync::Notify>,
    /// The page keys on the paths of the entries that joined the cache, and
    /// of those that left it, since [`Self::take_changes`]; kept only while
    /// a prompt waits ([`crate::store::waiting::watched`]).
    joined: HashSet<Hash256>,
    left: HashSet<Hash256>,
    /// A persistent disk tier's journal, and the slots it names.
    persisted: Option<Persisted>,
    /// The last count of reclaimable cache pages, when it was taken and what
    /// the walk took: see [`Self::reclaimable_cache_pages`].
    reclaimable: std::cell::Cell<Option<(Instant, std::time::Duration, u32)>>,
    /// The same for idle cache below the device: see
    /// [`Self::below_device_cache_pages`].
    below_cached: std::cell::Cell<Option<(Instant, std::time::Duration, (u32, u32))>>,
    /// The same for what of it the retention keys let go, and the keys'
    /// revision it was read under: see [`Self::droppable_below_device`].
    droppable_below: std::cell::Cell<Option<(Instant, std::time::Duration, u64, (u32, u32))>>,
    /// The reclaim order the last reclaim left: see [`Self::kept_order`].
    order: Option<KeptOrder>,
    /// The owners (an entry of none, by its id) whose entries joined or left
    /// the cache, or were used, since the kept order read them.
    reorder: rustc_hash::FxHashSet<Result<ProcessId, u64>>,
    /// Device pages a reclaim freed that wait on a fire's retirement: they
    /// are returning, and count toward what a waiter is short of, so a
    /// reclaim for the same shortfall frees none past them
    /// ([`Self::reclaim_returning`]).
    returning: HashSet<PhysicalKvPageId>,
    /// While a reclaim runs, the device pages it lets go.
    reclaiming: Option<Vec<PhysicalKvPageId>>,
    /// The disk slots whole disk chunks hold for state other than kv
    /// ([`Self::take_disk_chunks`]): a parked guest part's.
    guest_disk_slots: u32,
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
    /// The retention key each copy's page was demoted under: swap takes the
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
    /// Read outside the fleet ([`KvStore::readers_hold`]); else by a fire
    /// on its way.
    outside: bool,
}

#[derive(Debug, Clone, Copy)]
struct KeptEntry {
    path: KeptPath,
    owner: Option<ProcessId>,
    last_use: Instant,
}

/// A device chunk idle cache alone holds: its pages, and the entries they
/// are of ([`KvStore::idle_chunks`]).
#[derive(Debug, Default)]
struct IdleChunk {
    pages: Vec<(TriePageLocation, PhysicalKvPageId)>,
    by: HashSet<u64>,
}

/// The reclaim order ([`KvStore::kept_order`]) a reclaim left, and the
/// revisions it was read under: the entries' and the keys', and the
/// table's where the disk tier keeps copies.
/// Any change to what the order reads moves one of them, and the order is
/// walked again; the reclaim's own drops and settles leave it in order.
#[derive(Debug)]
struct KeptOrder {
    table: u64,
    keys: u64,
    order: Order,
}

/// A group of the reclaim order: its retention key, its entries, and the
/// owner they are of (an entry of none, by its id).
type OrderGroup = (i64, Vec<u64>, Result<ProcessId, u64>);

/// Where a group stands in the order: the lowest retention key first, the
/// least recently used, the oldest entry.
type Place = (i64, Instant, u64);

/// The reclaim order's groups by where they stand, and where each owner's
/// stand: an owner's groups go and come back without the rest being read.
#[derive(Debug, Default, PartialEq)]
struct Order {
    groups: BTreeMap<Place, OrderGroup>,
    of: rustc_hash::FxHashMap<Result<ProcessId, u64>, Vec<Place>>,
}

impl Order {
    fn insert(&mut self, place: Place, group: OrderGroup) {
        self.of.entry(group.2).or_default().push(place);
        self.groups.insert(place, group);
    }

    /// Takes out the groups of `of`.
    fn remove_of(&mut self, of: &Result<ProcessId, u64>) {
        for place in self.of.remove(of).into_iter().flatten() {
            self.groups.remove(&place);
        }
    }

    /// The groups, lowest place first.
    fn groups(&self) -> impl Iterator<Item = &OrderGroup> {
        self.groups.values()
    }
}

/// A set's domain: where its keys chain from at row 0. A set that asked
/// nothing has one of its own; one that shares, its scope's, resolved again
/// at each keyed pass ([`KvStore::pass_domain`]).
#[derive(Debug, Clone)]
struct SetScope {
    domain: Hash256,
    share: Option<crate::policy::share::ShareAs>,
}

#[derive(Debug, Clone, Copy)]
pub struct CachedWorkingSet {
    #[cfg_attr(
        target_arch = "wasm32",
        allow(dead_code, reason = "an idle cache is demoted on native targets alone")
    )]
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

    #[cfg(test)]
    pub fn new_with_swap(
        capacity: u32,
        host_capacity: u32,
        disk_capacity: u32,
        seed: Hash256,
    ) -> Self {
        Self::with_pages(
            ShapeRun {
                device: capacity,
                ..ShapeRun::default()
            },
            host_capacity,
            disk_capacity,
            seed,
        )
    }

    /// A store whose device pages are the list shape `pages` states.
    pub fn with_pages(
        pages: ShapeRun,
        host_capacity: u32,
        disk_capacity: u32,
        seed: Hash256,
    ) -> Self {
        Self::with_pool(
            ShapePool::new(
                ShapeKind::List,
                pages.slot_bytes,
                pages.ids_from,
                pages.device,
            ),
            host_capacity,
            disk_capacity,
            seed,
        )
    }

    /// A store whose device pages are `pool`'s ids: a pinned run's, or the
    /// slot refs of a shared pool's list shape.
    pub fn with_pool(
        pool: ShapePool<PhysicalKvPageId>,
        host_capacity: u32,
        disk_capacity: u32,
        seed: Hash256,
    ) -> Self {
        let slot_bytes = pool.slot_bytes();
        Self {
            table: KvPageTable::new(),
            pool,
            short_pages: 0,
            host_pool: ShapePool::new(ShapeKind::List, slot_bytes, 0, host_capacity),
            disk_pool: Pool::new(disk_capacity),
            shape: None,
            flat: HashMap::new(),
            domain: hash::cache_domain(&seed),
            scopes: HashMap::new(),
            private_seed: seed,
            private_nonce: 0,
            cache: CacheMap::new(),
            by_owner: HashMap::new(),
            splits: HashSet::new(),
            owners: HashMap::new(),
            keeps: false,
            readers: 0,
            reader_signal: None,
            kept: 0,
            page_size: 0,
            log_seq: 0,
            seq: 0,
            outstanding: BTreeSet::new(),
            moved: Arc::new(tokio::sync::Notify::new()),
            cached: Arc::new(tokio::sync::Notify::new()),
            changed: Arc::new(tokio::sync::Notify::new()),
            joined: HashSet::new(),
            left: HashSet::new(),
            persisted: None,
            reclaimable: std::cell::Cell::new(None),
            below_cached: std::cell::Cell::new(None),
            droppable_below: std::cell::Cell::new(None),
            order: None,
            reorder: rustc_hash::FxHashSet::default(),
            returning: HashSet::new(),
            reclaiming: None,
            guest_disk_slots: 0,
        }
    }

    /// The pages are slot refs of the shared pool's list shape `shape`;
    /// `host` (slot refs of the host tier's list shape) is where they are
    /// suspended and demoted to, and `disk_slots` slots of the engine's
    /// kv slot file below it (CONTRACT-BC 22).
    pub fn set_tiers(
        &mut self,
        shape: ::engine::chunk::ShapeId,
        host: Option<ShapePool<HostKvSlotId>>,
        disk_slots: u32,
    ) {
        self.shape = Some(shape);
        if let Some(host) = host {
            self.host_pool = host;
        }
        self.disk_pool = Pool::new(disk_slots);
    }

    /// Calls `signal` each time a reader outside the fleet ends, or host or
    /// disk slots come free: room a waiter blocked on swap room (a victim's
    /// move with no slot below) waits on.
    pub fn set_room_signal(&mut self, signal: super::slots::RoomSignal) {
        self.host_pool
            .set_room_signal(signal.clone(), super::slots::RoomTier::Swap);
        self.disk_pool.set_room_signal(signal.clone());
        self.reader_signal = Some(signal);
    }

    pub fn set_domain(&mut self, domain: Hash256) {
        self.domain = domain;
    }

    /// Whether `ws` has keyed nothing yet: no row logged, no page sealed.
    fn unkeyed(&self, ws: WorkingSetId) -> Result<bool, KvStoreError> {
        Ok(self.table.logged(ws)?.is_empty() && self.table.sealed_prefix(ws)? == 0)
    }

    /// `ws` shares as `ask` says (`working-set.share`): asked before it
    /// keys anything, since its keys chain from its domain. Asking what it
    /// shares as already is no change.
    pub fn share(
        &mut self,
        ws: WorkingSetId,
        ask: crate::policy::share::ShareAs,
    ) -> Result<(), String> {
        let domain = crate::policy::share::domain_now(&self.domain, &ask);
        let unkeyed = self.unkeyed(ws).map_err(|error| error.to_string())?;
        let Some(entry) = self.scopes.get_mut(&ws) else {
            return Err("share: the working set is gone".to_string());
        };
        if entry.share.as_ref() == Some(&ask) && entry.domain == domain {
            return Ok(());
        }
        if !unkeyed {
            return Err(
                "share: the working set holds keyed rows already; ask before its first pass"
                    .to_string(),
            );
        }
        *entry = SetScope {
            domain,
            share: Some(ask),
        };
        Ok(())
    }

    /// A set shared within the test tenant ([`crate::policy::share::test_tenant`]).
    #[cfg(test)]
    pub fn create_shared_working_set(&mut self) -> WorkingSetId {
        let ws = self.create_working_set();
        self.share(ws, crate::policy::share::test_tenant())
            .expect("a new set shares");
        ws
    }

    /// Whether `ws` shares (asked, and its scope still stands).
    pub fn shares(&self, ws: WorkingSetId) -> bool {
        self.scopes
            .get(&ws)
            .is_some_and(|scope| scope.share.is_some())
    }

    /// The domain a keyed pass on `ws` hashes its rows under, and whether
    /// it shares: its scope resolved again (a group's grant may have been
    /// revoked or reissued). A set that keyed nothing yet takes the scope's
    /// domain now; one whose rows chain from a domain its scope no longer
    /// resolves to shares nothing from here on, its rows still keyed where
    /// they began.
    pub fn pass_domain(&mut self, ws: WorkingSetId) -> Result<(Hash256, bool), KvStoreError> {
        let Some(entry) = self.scopes.get(&ws) else {
            return Err(page_table::KvTableError::UnknownWorkingSet.into());
        };
        let Some(ask) = &entry.share else {
            return Ok((entry.domain, false));
        };
        let now = crate::policy::share::domain_now(&self.domain, ask);
        if now == entry.domain {
            return Ok((now, true));
        }
        let unkeyed = self.unkeyed(ws)?;
        let entry = self.scopes.get_mut(&ws).expect("looked up above");
        if unkeyed {
            entry.domain = now;
            Ok((now, true))
        } else {
            tracing::debug!(target: "pie::kv", "a set's share scope moved past its rows; it shares no more");
            entry.share = None;
            Ok((entry.domain, false))
        }
    }

    /// `ws` asked nothing: a domain of its own, which no other set keys under.
    fn private_scope(&mut self, ws: WorkingSetId) {
        self.private_nonce += 1;
        let domain = crate::policy::share::private_domain(
            &self.domain,
            &self.private_seed,
            self.private_nonce,
        );
        self.scopes.insert(
            ws,
            SetScope {
                domain,
                share: None,
            },
        );
    }

    /// Where `ws`'s keys chain from at row 0.
    /// Every set the store makes has one ([`Self::private_scope`]); a set
    /// it does not know keys under no scope's domain either.
    fn domain_of(&self, ws: WorkingSetId) -> Hash256 {
        self.scopes.get(&ws).map_or_else(
            || crate::policy::share::private_domain(&self.domain, &self.private_seed, 0),
            |scope| scope.domain,
        )
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
            .map(|prev| hash::chain_slots(&self.domain_of(ws), prev, start, tokens)))
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
        let published = self.shares(ws);
        self.table.commit_hashes(ws, first, pages, published)?;
        if sealed > 0 {
            // Passes waiting on these pages look again.
            self.note_cached();
        }
        Ok(sealed)
    }

    /// Seals the pages a pass wrote from `start`, keyed by `slots`, once
    /// its fires have landed, as far as its rows are frozen: whole, computed
    /// and keyed, they never change in place, and the passes waiting on them
    /// map them in now rather than once the pass is finalized. What was
    /// sealed is a prompt's.
    pub fn seal_landed(
        &mut self,
        ws: WorkingSetId,
        start: u64,
        slots: &[Hash256],
        page_size: u32,
    ) -> Result<u32, KvStoreError> {
        let frozen = self.table.folded(ws)?.saturating_sub(start);
        let slots = &slots[..slots.len().min(frozen as usize)];
        let sealed = self.seal_rows(ws, start, slots, page_size)?;
        let end = start + slots.len() as u64;
        self.table
            .reach_prompt(ws, end / u64::from(page_size.max(1)))?;
        Ok(sealed)
    }

    /// [`Self::page_keys`] from row 0, which reads nothing of the store: a
    /// caller hashes them before it takes the store's lock.
    pub fn fresh_page_keys(slots: &[Hash256], page_size: u32) -> Vec<Hash256> {
        slots
            .chunks_exact(page_size.max(1) as usize)
            .map(|page| hash::page_hash(&page.iter().copied().map(Some).collect::<Vec<_>>()))
            .collect()
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

    /// The model's domain, not any set's: see [`Self::pass_domain`].
    pub fn domain(&self) -> Hash256 {
        self.domain
    }

    /// The epoch of the last fire (or write) handed out: an id let go now
    /// waits on it, and with it on every fire that may read it.
    pub fn current_epoch(&self) -> u64 {
        self.seq
    }

    /// Opens the epoch of a fire about to be submitted (CONTRACT-BC 17):
    /// every fire takes one, a decode and a prefill that copy nothing
    /// included, so what is let go while it runs (a freed page or slot, a
    /// segment an evacuation drained) waits until it retires. Closed by
    /// [`Self::settle`].
    pub fn begin_fire(&mut self) -> u64 {
        self.hand_out();
        self.seq
    }

    fn hand_out(&mut self) {
        self.seq += 1;
        self.outstanding.insert(self.seq);
        self.table.epochs(self.seq, self.retired_through());
    }

    pub fn retire_idle(&mut self) {
        let retired = self.retired_through();
        self.table.epochs(self.seq, retired);
        self.pool.retire_through(retired);
    }

    /// The last epoch every fire through which has retired.
    pub fn retired_through(&self) -> u64 {
        self.outstanding
            .first()
            .map_or(self.seq, |&oldest| oldest.saturating_sub(1))
    }

    pub fn create_working_set(&mut self) -> WorkingSetId {
        let ws = self.table.create_working_set();
        self.flat.insert(ws, Some(Vec::new()));
        self.private_scope(ws);
        ws
    }

    /// Its device pages go back now, or where a fire in flight may still
    /// read them (a set that left its path under one, a pin let go, a copy
    /// into it) once that fire retires, never behind fires that read none
    /// of them.
    fn release_entry(&mut self, entry: KeptEntry) -> usize {
        let epoch = self.table.read_through(entry.path.terminal);
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

    fn disown(&mut self, id: u64, entry: &KeptEntry, by: By) {
        let Some(owner) = entry.owner else {
            return;
        };
        if let Some(ids) = self.owners.get_mut(&owner) {
            ids.retain(|&owned| owned != id);
            if ids.is_empty() {
                self.owners.remove(&owner);
                crate::policy::report(Event::CacheEmptied(owner, by));
            }
        }
    }

    /// Every page the trie keeps resident on the device, by its device slot
    /// (for tests: each must lie in a slot the pool counts held).
    #[cfg(test)]
    pub(crate) fn device_page_slots(&self) -> Vec<crate::store::slots::Slot> {
        let Some((shared, shape)) = self.pool.shared_shape() else {
            return Vec::new();
        };
        let base = self.pool.base();
        let shared = shared.lock().unwrap();
        self.table
            .device_pages_by_reader()
            .into_iter()
            .map(|(_, id, _)| shared.slot(shape, id.0.saturating_sub(base)))
            .collect()
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

    pub fn live_working_sets(&self) -> usize {
        self.table.live_working_sets()
    }

    pub fn rekey(&mut self, domain: Hash256) -> usize {
        let keys: Vec<u64> = self.cache.keys().copied().collect();
        let freed = self.drop_entries(keys, By::Client);
        self.domain = domain;
        freed
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn owns_cache(&self, owner: ProcessId) -> bool {
        self.owners.contains_key(&owner)
    }

    /// Whether `owner` (a PLEX id) has KV here: cached, or in a working set
    /// not yet released. A process's last fire settles after it exits, and
    /// only then is what it leaves cached: until then its working set is
    /// still here, not gone.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn holds_kv_of(&self, owner: ProcessId) -> bool {
        self.owns_cache(owner)
            || self
                .table
                .working_set_owners()
                .any(|of| crate::policy::schedule::plex_id(of) == owner)
    }

    /// Drops every idle entry `owner` left, at once (tests set a store up
    /// with it; no runtime path drops cache unasked by its rows).
    #[cfg(test)]
    pub fn discard_cache(&mut self, owner: ProcessId, by: By) -> usize {
        let keys: Vec<u64> = self
            .cache
            .iter()
            .filter(|(_, entry)| entry.owner == Some(owner))
            .map(|(&key, _)| key)
            .collect();
        self.drop_entries(keys, by)
    }

    fn drop_entries(&mut self, keys: Vec<u64>, by: By) -> usize {
        let mut freed = 0;
        for key in keys {
            if let Some(entry) = self.cache.remove(&key) {
                self.note_order(entry.owner.ok_or(key));
                self.note_change(entry.path.terminal, false);
                self.unlist(key, entry.path.terminal);
                self.splits.retain(|&(split, _)| split != key);
                self.disown(key, &entry, by);
                freed += self.release_entry(entry);
            }
        }
        self.forget_lost_copies();
        freed
    }

    /// Frees `short` device pages of idle cache and no more (policy-out
    /// A18: exactly what is short): only what no live working set reads,
    /// lowest key first, then least recently used, a set at a time (the
    /// entries that hold one set's pages go together, oldest set first),
    /// stopping once what is short is freed. The runtime runs it once a
    /// demote had nothing to move (`prepare_demote` goes first where the
    /// planner demotes): what may sit on a lower tier with room moved there,
    /// so what is left has none, and cached KV yields to running work,
    /// whatever its key. Pages the disk tier keeps a copy of settle there
    /// instead, at no cost, and their entries stay.
    pub fn reclaim_cache(&mut self, short: usize, keys: impl Keys) -> usize {
        // Pages an earlier reclaim freed are on their way back: what they
        // cover is not short.
        let short = short.saturating_sub(self.reclaim_returning());
        if short == 0 {
            return 0;
        }
        self.reclaiming = Some(Vec::new());
        let mut freed = 0;
        let mut order = self.kept_order(&keys);
        // Groups that free no device page go last; the walk stops as soon as
        // what is short is freed, so it reads only the groups it reaches.
        let mut later: Vec<Place> = Vec::new();
        let mut reached = Vec::new();
        let reads = self.table.live_reads();
        'passes: for last in [false, true] {
            let walk: Box<dyn Iterator<Item = Place>> = if last {
                Box::new(std::mem::take(&mut later).into_iter())
            } else {
                Box::new(order.groups.keys().copied())
            };
            for at in walk {
                let (key, ids, _) = &order.groups[&at];
                let ids = self.at_rest(ids, &reads);
                if ids.is_empty() {
                    continue;
                }
                let ids = &ids;
                let roots = self.roots(ids);
                if !last && !self.table.frees_device_pages(&roots) {
                    later.push(at);
                    continue;
                }
                let key = *key;
                let before = freed;
                freed += self.settle_on_disk(ids, key, &keys, short - freed);
                if freed == before {
                    let ids: Vec<u64> = ids
                        .iter()
                        .copied()
                        .filter(|id| !self.cache.get(id).is_some_and(|e| self.persisted_whole(e)))
                        .collect();
                    tracing::debug!(target: "pie::kv", key, entries = ids.len(), "reclaim drops");
                    // A set at a time, oldest first: every entry that holds
                    // the set's pages (a kept checkpoint and the sealed
                    // prefix it stands on name the same pages) goes with
                    // it, since dropping one alone frees nothing.
                    for id in ids {
                        let Some(terminal) = self.cache.get(&id).map(|e| e.path.terminal) else {
                            continue;
                        };
                        let set: Vec<u64> = self
                            .table
                            .path_owner(terminal)
                            .and_then(|owner| self.by_owner.get(&owner))
                            .into_iter()
                            .flatten()
                            .copied()
                            .filter(|id| {
                                self.cache
                                    .get(id)
                                    .is_some_and(|e| e.path.terminal == terminal)
                            })
                            .collect();
                        freed += self.drop_entries(set, By::Host);
                        if freed >= short {
                            break;
                        }
                    }
                    reached.push(at);
                }
                if freed >= short {
                    break 'passes;
                }
            }
        }
        // A group's dropped entries leave it; the owner they were of reads
        // again at the next reclaim (its drops are noted).
        for at in reached {
            let cache = &self.cache;
            if let Some((_, ids, _)) = order.groups.get_mut(&at) {
                ids.retain(|id| cache.contains_key(id));
                if ids.is_empty() {
                    order.groups.remove(&at);
                }
            }
        }
        self.keep_order(order, &keys);
        let mut let_go: HashSet<PhysicalKvPageId> = self
            .reclaiming
            .take()
            .unwrap_or_default()
            .into_iter()
            .collect();
        self.pool.retain_pending(&mut let_go);
        self.returning.extend(let_go);
        tracing::debug!(target: "pie::kv", short, freed, "reclaimed");
        freed
    }

    /// Opens `want` whole device chunks from idle cache, as
    /// [`Self::reclaim_cache`] frees pages ([`Self::idle_chunks`]): what
    /// holds a chunk's pages goes together; dropped an entry at a time, each
    /// a page of many chunks, nearly a key's whole cache goes before one
    /// opens. Answers the device pages freed: 0 where no chunk is idle
    /// cache's whole.
    pub fn reclaim_chunks(&mut self, want: u32, keys: impl Keys) -> usize {
        let mut freed = 0;
        for chunk in self.idle_chunks(want, &keys) {
            freed += self.drop_entries(
                chunk
                    .by
                    .into_iter()
                    .filter(|id| self.cache.contains_key(id))
                    .collect(),
                By::Host,
            );
        }
        tracing::debug!(target: "pie::kv", want, freed, "reclaimed whole chunks");
        freed
    }

    /// Moves the pages of `want` whole device chunks of idle cache below the
    /// device ([`Self::idle_chunks`]), as [`Self::prepare_demote`] moves a
    /// group's: those chunks' pages alone, where every row of what holds
    /// them lets it on the host and the tiers below have the room.
    pub fn prepare_demote_chunks(
        &mut self,
        want: u32,
        keys: impl Keys,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        if !keys.cache_may_go_below() {
            return Ok(None);
        }
        let mut pages = Vec::new();
        let mut disk = self.disk_pool.available();
        for chunk in self.idle_chunks(want, &keys) {
            let tiers = chunk.by.iter().fold(retention::ALL_TIERS, |tiers, &id| {
                tiers & self.tiers_of(id, &keys)
            });
            if tiers & retention::HOST == 0 {
                return Ok(None);
            }
            if tiers & retention::DISK == 0 {
                disk = 0;
            }
            pages.extend(chunk.pages);
        }
        let host = self.host_pool.available();
        if pages.is_empty() || host + disk < pages.len() {
            return Ok(None);
        }
        let on_host = host.min(pages.len());
        self.prepare_move_out_split(&HashSet::new(), pages, on_host)
            .map(Some)
    }

    /// The device chunks to open for `want` from idle cache, each held by
    /// nothing else: the lowest retention key first, as far as the keys let
    /// it, and of one key's chunks the one whose newest entry was used
    /// longest ago. None while the disk tier keeps copies (a page settles
    /// there at no cost, whatever its chunk).
    fn idle_chunks(&mut self, want: u32, keys: &impl Keys) -> Vec<IdleChunk> {
        if want == 0
            || self
                .persisted
                .as_ref()
                .is_some_and(|p| !p.by_key.is_empty())
        {
            return Vec::new();
        }
        let Some((shared, shape)) = self.pool.shared_shape() else {
            return Vec::new();
        };
        let (spc, chunks) = shared.lock().unwrap().single_chunks(shape);
        if spc == 0 {
            return Vec::new();
        }
        let order = self.kept_order(keys);
        let reads = self.table.live_reads();
        let base = self.pool.base();
        let mut ids: Vec<u64> = Vec::new();
        let mut opens: Vec<(Instant, IdleChunk)> = Vec::new();
        let mut groups = order.groups().peekable();
        while let Some((key, group, _)) = groups.next() {
            let key = *key;
            ids.extend(self.at_rest(group, &reads));
            while let Some((_, group, _)) = groups.next_if(|(other, ..)| *other == key) {
                ids.extend(self.at_rest(group, &reads));
            }
            if ids.is_empty() {
                continue;
            }
            let mut holders: HashMap<TriePageLocation, Vec<u64>> = HashMap::new();
            for &id in &ids {
                for (at, _) in self.table.path_pages(self.cache[&id].path.terminal) {
                    holders.entry(at).or_default().push(id);
                }
            }
            let mut held: HashMap<u32, IdleChunk> = HashMap::new();
            let above = self.above(key, keys);
            for (at, page) in self.table.cache_only_pages(&self.roots(&ids), &above) {
                let chunk = held.entry(page.0.saturating_sub(base) / spc).or_default();
                chunk.pages.push((at, page));
                chunk.by.extend(holders.get(&at).into_iter().flatten());
            }
            opens = chunks
                .iter()
                .filter(|chunk| chunk.settled && chunk.used > 0)
                .filter_map(|chunk| {
                    let idle = held
                        .remove(&chunk.logical)
                        .filter(|idle| idle.pages.len() == chunk.used as usize)?;
                    let newest = idle.by.iter().map(|id| self.cache[id].last_use).max()?;
                    Some((newest, idle))
                })
                .collect();
            if opens.len() >= want as usize {
                break;
            }
        }
        drop(groups);
        self.keep_order(order, keys);
        opens.sort_by_key(|(newest, _)| *newest);
        opens.truncate(want as usize);
        opens.into_iter().map(|(_, chunk)| chunk).collect()
    }

    /// Whether anything is cached here.
    pub fn holds_cache(&self) -> bool {
        !self.cache.is_empty()
    }

    /// The device pages reclaims freed that still wait on a fire's
    /// retirement: returning, with nothing more to do but wait for it (a
    /// fire's settle retires them, and tells the planner).
    pub fn reclaim_returning(&mut self) -> usize {
        self.pool.retain_pending(&mut self.returning);
        self.returning.len()
    }

    /// Every entry in [`Self::order_of`]'s groups. Kept between reclaims, and
    /// of what changed since (an owner's entries, its rows) only that owner's
    /// groups are read again and put where they stand now, its entries split
    /// at their steps first: a reclaim short of a page at a time walked every
    /// cached entry under the lock every guest's fire waits on, each time.
    /// What is in use or off the device is read as the reclaim reaches it, so
    /// a decode's appends and a fork move the order not at all. All of it is
    /// walked where a column's default changed, or where pages of persisted
    /// heads, which split entries too, moved.
    fn kept_order(&mut self, keys: &impl Keys) -> Order {
        let copies = self
            .persisted
            .as_ref()
            .is_some_and(|p| !p.by_key.is_empty());
        if let Some(kept) = self.order.take()
            && (!copies || (kept.table == self.table.revision() && self.reorder.is_empty()))
            && let Some(rows) = keys.changed_since(kept.keys)
        {
            let mut changed = std::mem::take(&mut self.reorder);
            changed.extend(rows.into_iter().map(Ok));
            return self.reordered(kept.order, changed, keys);
        }
        self.split_at_steps(keys, self.cache.keys().copied().collect());
        self.split_persisted();
        self.reorder.clear();
        let mut order = Order::default();
        for (place, group) in self.placed(keys, self.cache.keys().copied()) {
            order.insert(place, group);
        }
        order
    }

    /// The entries of an owner (or one of none, by its id) changed: the kept
    /// order reads them again.
    fn note_order(&mut self, of: Result<ProcessId, u64>) {
        if self.order.is_some() || self.reclaiming.is_some() {
            self.reorder.insert(of);
        }
    }

    /// `groups` with those of the owners `changed` read again.
    fn reordered(
        &mut self,
        mut order: Order,
        changed: rustc_hash::FxHashSet<Result<ProcessId, u64>>,
        keys: &impl Keys,
    ) -> Order {
        if changed.is_empty() {
            return order;
        }
        for of in &changed {
            order.remove_of(of);
        }
        let owned = |store: &Self| -> Vec<u64> {
            changed
                .iter()
                .flat_map(|of| match of {
                    Ok(owner) => store.owners.get(owner).cloned().unwrap_or_default(),
                    Err(id) => vec![*id],
                })
                .filter(|id| store.cache.contains_key(id))
                .collect()
        };
        self.split_at_steps(keys, owned(self));
        self.reorder.clear();
        for (place, group) in self.placed(keys, owned(self).into_iter()) {
            order.insert(place, group);
        }
        order
    }

    /// Of a kept order's group, what no live set reads, at rest on the
    /// device now: the order says none of that.
    fn at_rest(&self, ids: &[u64], reads: &page_table::LiveReads) -> Vec<u64> {
        ids.iter()
            .copied()
            .filter(|id| {
                self.cache.get(id).is_some_and(|entry| {
                    let root = entry.path.terminal;
                    self.table.unused_under(reads, root)
                        && !self.table.in_move_at(root)
                        && self.table.holds_resident(root)
                })
            })
            .collect()
    }

    /// Keeps what of `order` a reclaim left (its dropped entries taken out),
    /// under the revisions it leaves: its settles move pages down, which the
    /// next reclaim reads as it goes; nothing else it does reorders the rest.
    fn keep_order(&mut self, order: Order, keys: &impl Keys) {
        let Some(revision) = keys.revision() else {
            return;
        };
        self.order = Some(KeptOrder {
            table: self.table.revision(),
            keys: revision,
            order,
        });
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
    /// copy of, onto their copies, up to `most`: no copy is made. Answers
    /// the device pages freed.
    fn settle_on_disk(&mut self, ids: &[u64], key: i64, keys: &impl Keys, most: usize) -> usize {
        if self.persisted.as_ref().is_none_or(|p| p.by_key.is_empty()) {
            return 0;
        }
        let above = self.above(key, keys);
        let moves: Vec<(TriePageLocation, PhysicalKvPageId, DiskKvSlotId)> = self
            .table
            .cache_only_pages(&self.roots(ids), &above)
            .into_iter()
            .filter_map(|(location, gpu)| Some((location, gpu, self.copy_of(&location)?)))
            .take(most)
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
        let epoch = moves
            .iter()
            .map(|(location, ..)| self.table.node_read_through(location.node))
            .max()
            .unwrap_or(0);
        tracing::debug!(
            target: "runtime::store::kv",
            pages = ?moves.iter().map(|&(_, gpu, slot)| (gpu.0, slot.0)).collect::<Vec<_>>(),
            "cache-only device pages settled on their disk copies"
        );
        let gpu: Vec<PhysicalKvPageId> = moves.into_iter().map(|(_, gpu, _)| gpu).collect();
        if let Some(reclaiming) = &mut self.reclaiming {
            reclaiming.extend(&gpu);
        }
        self.pool.recycle_after_epoch(gpu, epoch);
        // With no fire in flight the epoch has retired already: the pages
        // are free now, not at a next fire that may wait on them.
        self.retire_idle();
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
    fn split_at_steps(&mut self, keys: &impl Keys, ids: Vec<u64>) {
        let page = u64::from(self.page_size.max(1));
        if !self.keeps {
            return;
        }
        let mut heads = Vec::new();
        for (id, entry) in ids
            .into_iter()
            .filter_map(|id| Some((id, self.cache.get(&id)?)))
        {
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
            self.kept += 1;
            if let Some(owner) = entry.owner {
                self.owners.entry(owner).or_default().push(self.kept);
            }
            self.list(self.kept, path.terminal);
            retention::idle_cache_appeared();
            self.note_order(entry.owner.ok_or(self.kept));
            self.cache.insert(self.kept, KeptEntry { path, ..entry });
            self.note_change(path.terminal, true);
        }
    }

    /// [`Self::order_of`] the entries no live set reads, at rest, that hold a
    /// page on the tier `held` reads.
    fn order_where(
        &self,
        keys: &impl Keys,
        held: fn(&KvPageTable, NodeId) -> bool,
    ) -> Vec<OrderGroup> {
        let unused: HashSet<NodeId> = self.table.unused_cache_roots().into_iter().collect();
        if unused.is_empty() {
            return Vec::new();
        }
        let ids = self.cache.iter().filter(|(_, entry)| {
            let root = entry.path.terminal;
            unused.contains(&root) && !self.table.in_move_at(root) && held(&self.table, root)
        });
        self.order_of(keys, ids.map(|(&id, _)| id))
    }

    /// The entries `ids` (each in the cache), in groups: the lowest retention
    /// key first, least recently used among equals.
    fn order_of(&self, keys: &impl Keys, ids: impl Iterator<Item = u64>) -> Vec<OrderGroup> {
        self.placed(keys, ids)
            .into_iter()
            .map(|(_, group)| group)
            .collect()
    }

    /// [`Self::order_of`], each group with its [`Place`].
    fn placed(&self, keys: &impl Keys, ids: impl Iterator<Item = u64>) -> Vec<(Place, OrderGroup)> {
        type Group = (Vec<u64>, Instant);
        // An owner's entries group by their row's key; an entry no process
        // owns (a sealed prefix an exited set left) is a group of its own,
        // never pooled with every other exited set's.
        let mut groups: HashMap<(Result<ProcessId, u64>, i64), Group> = HashMap::new();
        for id in ids {
            let entry = &self.cache[&id];
            let key = keys.key_at(entry.owner, self.key_token(entry.path.pages));
            let group = groups
                .entry((entry.owner.ok_or(id), key))
                .or_insert_with(|| (Vec::new(), entry.last_use));
            group.0.push(id);
            group.1 = group.1.max(entry.last_use);
        }
        // Lowest retention priority first, then least recently used (the
        // engine's own tie, PLEX 1.1 §4.2): what a decode wrote has no order
        // of its own over a prompt unless a row says so (`default` keys the
        // decoded tail below the prompt). Ties (one key, one last use) go
        // oldest entry first, never by the map's order.
        let mut order: Vec<_> = groups
            .into_iter()
            .map(|((of, key), (mut ids, last_use))| {
                ids.sort_unstable();
                ((key, last_use, ids[0]), (key, ids, of))
            })
            .collect();
        order.sort_unstable_by_key(|(order, _)| *order);
        tracing::debug!(
            target: "pie::retention",
            order = ?order.iter().map(|((key, ..), (_, ids, _))| (*key, ids.len())).collect::<Vec<_>>(),
            "idle cache, lowest key first"
        );
        order
    }

    fn roots(&self, keys: &[u64]) -> Vec<NodeId> {
        keys.iter()
            .filter_map(|key| Some(self.cache.get(key)?.path.terminal))
            .collect()
    }

    /// Moves idle cache off the device, host first and what the host
    /// cannot take to disk (as a suspend's pages go), lowest retention key
    /// first: one group, up to `room` pages. A persistent disk tier keeps
    /// what lands on disk as its copies, journaled. Only cache whose tiers
    /// name the host moves: the rest is dropped when its room is wanted.
    pub fn prepare_demote(
        &mut self,
        room: usize,
        keys: impl Keys,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        // No row lets idle cache below the device: nothing moves down, and
        // nothing need be walked to find that.
        if !keys.cache_may_go_below() {
            return Ok(None);
        }
        self.prepare_demote_where(room, keys, false)
    }

    /// The tiers idle cache entry `id`'s KV may sit on.
    fn tiers_of(&self, id: u64, keys: &impl Keys) -> u8 {
        self.cache.get(&id).map_or(retention::ALL_TIERS, |entry| {
            keys.cache_tiers_at(entry.owner, self.key_token(entry.path.pages))
        })
    }

    /// Idle cache a `retention/tiers` row keeps off the device, moved down
    /// now, whatever its key and whether or not its room is wanted.
    pub fn prepare_demote_now(
        &mut self,
        keys: impl Keys,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        let room = self.host_pool.available() + self.disk_pool.available();
        self.prepare_demote_where(room, keys, true)
    }

    /// Idle cache whose row lets it sit nowhere, or not where it is (on
    /// the host, a row without the host; on disk, a row without the disk),
    /// dropped now. How many entries.
    pub fn drop_disallowed(&mut self, keys: impl Keys) -> usize {
        self.split_at_steps(&keys, self.cache.keys().copied().collect());
        let unused: HashSet<NodeId> = self.table.unused_cache_roots().into_iter().collect();
        let gone: Vec<u64> = self
            .cache
            .iter()
            .filter(|(_, entry)| unused.contains(&entry.path.terminal))
            .filter(|(id, entry)| {
                let tiers = self.tiers_of(**id, &keys);
                let root = entry.path.terminal;
                tiers == 0
                    || (tiers & retention::HOST == 0 && self.table.holds_on_host(root))
                    || (tiers & retention::DISK == 0 && self.table.holds_on_disk(root))
            })
            .map(|(&id, _)| id)
            .collect();
        if gone.is_empty() {
            return 0;
        }
        tracing::debug!(target: "pie::retention", entries = gone.len(), "dropped: their row allows no tier they are on");
        self.drop_entries(gone, By::Policy)
    }

    /// [`Self::prepare_demote`], or (`now`) only cache whose row keeps it
    /// off the device and lets it on the host. Cache whose row keeps it off
    /// the host is never moved down: it is dropped when its room is wanted.
    fn prepare_demote_where(
        &mut self,
        room: usize,
        keys: impl Keys,
        now: bool,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        if room == 0 {
            return Ok(None);
        }
        let order = self.kept_order(&keys);
        let reads = self.table.live_reads();
        // The device holds the highest keys first: idle cache keyed lower
        // that may not move down is dropped (the caller's reclaim) before
        // anything keyed above it moves down. A sweep (`now`) moves what its
        // row keeps off the device whatever lies below.
        let mut floor = None;
        let mut below = Vec::new();
        for (key, ids, _) in order.groups() {
            let ids = self.at_rest(ids, &reads);
            if ids.is_empty() {
                continue;
            }
            if !now
                && ids
                    .iter()
                    .all(|&id| self.tiers_of(id, &keys) & retention::HOST == 0)
                && self.table.frees_device_pages(&self.roots(&ids))
            {
                floor = Some(*key);
                break;
            }
            below.push((*key, ids));
        }
        self.keep_order(order, &keys);
        let mut groups: Vec<(i64, Vec<u64>)> = below
            .into_iter()
            .filter(|(key, _)| floor.is_none_or(|floor| *key < floor))
            .map(|(key, ids)| {
                let ids: Vec<u64> = ids
                    .into_iter()
                    .filter(|&id| {
                        let tiers = self.tiers_of(id, &keys);
                        tiers & retention::HOST != 0 && (!now || tiers & retention::DEVICE == 0)
                    })
                    .collect();
                (key, ids)
            })
            .filter(|(_, ids)| !ids.is_empty())
            .collect();
        for last in [false, true] {
            let mut later = Vec::new();
            for (key, ids) in std::mem::take(&mut groups) {
                if !last && !self.table.frees_device_pages(&self.roots(&ids)) {
                    later.push((key, ids));
                    continue;
                }
                // What a higher key also covers stays with it.
                let above = self.above(key, &keys);
                let mut pages = self.table.cache_only_pages(&self.roots(&ids), &above);
                // A page with a copy on disk settles there when its room is
                // wanted, at no cost: host room is for the rest.
                pages.retain(|(location, _)| self.copy_of(location).is_none());
                if pages.is_empty() {
                    continue;
                }
                // Each tier holds the highest keys first: short of free slots
                // below, the group's room comes from idle cache the host
                // holds keyed lower, lowest first. Keyed alike (no policy),
                // nothing is displaced and a demote takes free slots only.
                // Disk is room only for a group every row of which lets it
                // on disk; the rest moves to the host or nowhere.
                let disk = if ids
                    .iter()
                    .all(|&id| self.tiers_of(id, &keys) & retention::DISK != 0)
                {
                    self.disk_pool.available()
                } else {
                    0
                };
                let want = pages.len().min(room);
                if self.host_pool.available() + disk < want {
                    let host = want.saturating_sub(disk);
                    if self.drop_entries_off_device(host, &keys, Some(key)) > 0 {
                        // Dropping can compact the pages just listed.
                        pages = self.table.cache_only_pages(&self.roots(&ids), &above);
                        pages.retain(|(location, _)| self.copy_of(location).is_none());
                    }
                }
                let free = self.host_pool.available() + disk;
                if pages.is_empty() {
                    continue;
                }
                // The lowest group finds no room below: it is dropped (the
                // caller's reclaim) before anything keyed above it moves down.
                if free == 0 {
                    return Ok(None);
                }
                pages.truncate(room.min(free));
                tracing::debug!(target: "pie::kv", pages = pages.len(), "demote");
                let records = self.demote_records(&ids, pages.iter().map(|&(at, _)| at), key);
                // A run split over the tiers would leave its disk tail
                // following on from pages a restart loses: a persistent
                // tier takes a group the host cannot hold whole.
                let host = self.host_pool.available();
                let on_host = if disk == 0
                    || pages.len() <= host
                    || records.is_empty()
                    || self.disk_pool.available() < pages.len()
                {
                    host.min(pages.len())
                } else {
                    0
                };
                let mut txn = self.prepare_move_out_split(&HashSet::new(), pages, on_host)?;
                txn.records = records;
                return Ok(Some(txn));
            }
            groups = later;
        }
        Ok(None)
    }

    /// The journal records of the sealed whole pages of `pages`, read off
    /// the paths of entries `ids`, kept under `retention`; none where the
    /// disk tier is not kept.
    fn demote_records(
        &self,
        ids: &[u64],
        pages: impl IntoIterator<Item = TriePageLocation>,
        retention: i64,
    ) -> HashMap<TriePageLocation, journal::PageRecord> {
        if self.persisted.is_none() || !self.keeps {
            return HashMap::new();
        }
        let wanted: HashSet<TriePageLocation> = pages.into_iter().collect();
        let mut records = HashMap::new();
        for terminal in self.roots(ids) {
            let mut parent = None;
            for (at, key) in self.table.path_pages(terminal) {
                let Some(key) = key else {
                    break;
                };
                if wanted.contains(&at) && !records.contains_key(&at) {
                    let tokens = self.table.token_hashes_at(&at).unwrap_or_default();
                    if tokens.len() == self.page_size as usize {
                        records.insert(
                            at,
                            journal::PageRecord {
                                page: key,
                                parent,
                                tokens: tokens.to_vec(),
                                retention,
                            },
                        );
                    }
                }
                parent = Some(key);
            }
        }
        records
    }

    /// The demoted pages that landed on disk become the persistent tier's
    /// copies, each backing its page: journaled, durably.
    fn keep_demoted(
        &mut self,
        pages: impl IntoIterator<Item = (TriePageLocation, KvPageBacking)>,
        mut records: HashMap<TriePageLocation, journal::PageRecord>,
    ) {
        let Some(persisted) = &mut self.persisted else {
            return;
        };
        let mut kept = 0;
        let mut written = Ok(());
        for (location, backing) in pages {
            let KvPageBacking::OnDisk(slot) = backing else {
                continue;
            };
            let Some(record) = records.remove(&location) else {
                continue;
            };
            if persisted.by_key.contains_key(&record.page) || persisted.slots.contains_key(&slot) {
                continue;
            }
            written = persisted.journal.put(slot.0, &record);
            if written.is_err() {
                break;
            }
            persisted.slots.insert(slot, record.page);
            persisted.by_key.insert(record.page, slot);
            persisted.retention.insert(slot, record.retention);
            persisted.back(slot);
            kept += 1;
        }
        if let Err(error) = written.and_then(|()| persisted.journal.sync()) {
            tracing::error!(target: "pie::kv", %error, "disk kv journal not written; the tier is kept no longer");
            self.abandon_persisted();
            return;
        }
        if kept > 0 {
            tracing::debug!(target: "pie::kv", pages = kept, "demoted to disk, journaled");
        }
    }

    /// Moves idle cache parked on the host on down to disk, lowest
    /// retention key first, until `wanted` pages go or the disk is full:
    /// what a host short of room would otherwise drop. A page the disk tier
    /// keeps a copy of settles on it here, nothing copied. On a persistent
    /// tier a group goes whole or not at all (a run split over host and
    /// disk would leave its disk tail an orphan at restart), so more than
    /// `wanted` may go; what lands is journaled. Only a shared pool's kv
    /// list moves host -> disk (CONTRACT-BC 22), and only cache whose tiers
    /// name the disk is copied: the rest on the host is dropped when the
    /// host's room is wanted (`prepare_suspend`). `None`
    /// when nothing is left to copy.
    pub fn prepare_spill(&mut self, wanted: usize, keys: impl Keys) -> Option<KvSpillTxn> {
        let shape = self.shape?;
        let mut room = self.disk_pool.available();
        self.split_at_steps(&keys, self.cache.keys().copied().collect());
        self.split_persisted();
        let mut pages: Vec<(TriePageLocation, HostKvSlotId)> = Vec::new();
        let mut taken: HashSet<TriePageLocation> = HashSet::new();
        let mut records = HashMap::new();
        for (key, ids, _) in self.order_where(&keys, KvPageTable::holds_on_host) {
            if pages.len() >= wanted || room == 0 {
                break;
            }
            // Only what its tiers let on disk goes there: the rest is dropped
            // when the host's room is wanted.
            let ids: Vec<u64> = ids
                .into_iter()
                .filter(|&id| self.tiers_of(id, &keys) & retention::DISK != 0)
                .collect();
            if ids.is_empty() {
                continue;
            }
            // What a higher key also covers stays with it.
            let above = self.above(key, &keys);
            let mut group = self.table.cache_only_host_slots(&self.roots(&ids), &above);
            group.retain(|(location, _)| !taken.contains(location));
            self.settle_host_on_disk(&mut group);
            if group.is_empty() {
                continue;
            }
            let kept = self.demote_records(&ids, group.iter().map(|&(at, _)| at), key);
            if kept.is_empty() {
                group.truncate(room.min(wanted - pages.len()));
            } else if group.len() > room {
                continue;
            }
            room -= group.len();
            taken.extend(group.iter().map(|&(at, _)| at));
            pages.extend(group);
            records.extend(kept);
        }
        self.below_cached.set(None);
        self.droppable_below.set(None);
        if pages.is_empty() {
            return None;
        }
        let disk = self
            .disk_pool
            .try_alloc_n(pages.len())
            .expect("counted above");
        self.table
            .pin_swap_locations(pages.iter().map(|&(location, _)| location));
        tracing::debug!(target: "pie::kv", pages = pages.len(), "spill host to disk");
        Some(KvSpillTxn {
            pages: pages
                .into_iter()
                .zip(disk)
                .map(|((location, host), disk)| (location, host, disk))
                .collect(),
            shape,
            records,
        })
    }

    /// `n` whole disk chunks of the slot file for something other than kv
    /// (a parked role's slots, laid out in a chunk of their own shape as on
    /// the device, CONTRACT-BC 22, [`crate::store::roles::DiskSlots`]), a
    /// disk chunk `per_chunk` kv slots, as chunk ids, all or none: counted
    /// out of the disk's room as kv's are.
    pub fn take_disk_chunks(&mut self, n: usize, per_chunk: u32) -> Option<Vec<u32>> {
        let per_chunk = per_chunk.max(1);
        let firsts = self.disk_pool.try_alloc_aligned_runs(n, per_chunk)?;
        self.below_cached.set(None);
        self.droppable_below.set(None);
        self.guest_disk_slots += firsts.len() as u32 * per_chunk;
        Some(firsts.into_iter().map(|slot| slot.0 / per_chunk).collect())
    }

    /// The kv pages held on every tier, each once, and the idle cache's
    /// among them, read now ([`KvCensus`]): what the capacity gate counts
    /// kv by (`planner::PoolPort::capacity_census`).
    #[must_use]
    pub fn census(&self) -> KvCensus {
        let disk_out = self
            .disk_pool
            .capacity()
            .saturating_sub(self.disk_pool.available() as u32);
        let roots = self.table.unused_cache_roots();
        let (idle_host, idle_disk) =
            if self.host_pool.capacity() == 0 && self.disk_pool.capacity() == 0 {
                (0, 0)
            } else {
                self.table.cache_only_below_device(&roots, &[])
            };
        KvCensus {
            held: u64::from(self.pool.out())
                + u64::from(self.host_pool.out())
                + u64::from(disk_out.saturating_sub(self.guest_disk_slots)),
            guest_disk: u64::from(self.guest_disk_slots),
            moving: self.table.moving_pages() as u64,
            idle: u64::from(self.table.device_pages_freed(&roots))
                + u64::from(idle_host)
                + u64::from(idle_disk),
            disk_slots: self.disk_pool.capacity(),
        }
    }

    /// The device's list chunks and the pages in each that may move, by
    /// logical chunk: each chunk's used slots, and its pages with the
    /// processes reading them (`None` for a chunk one of whose slots stays:
    /// a page a move or pin holds, one read by a set of a process
    /// `may_move` refuses, or a slot let go and still waiting on a fire).
    /// Chunks a chunk move or copy holds are left out as sources and as
    /// destinations alike.
    fn list_chunk_pages(
        &self,
        may_move: MayMove<'_>,
    ) -> Option<(
        u32,
        Vec<crate::store::slots::SingleChunk>,
        BTreeMap<u32, ChunkPages>,
    )> {
        let (shared, shape) = self.pool.shared_shape()?;
        let (spc, chunks) = shared.lock().unwrap().single_chunks(shape);
        if spc == 0 {
            return None;
        }
        let base = self.pool.base();
        let mut by_chunk: BTreeMap<u32, ChunkPages> = BTreeMap::new();
        for (location, id, readers) in self.table.device_pages_by_reader() {
            let chunk = id.0.saturating_sub(base) / spc;
            let entry = by_chunk
                .entry(chunk)
                .or_insert_with(|| Some((Vec::new(), BTreeSet::new(), BTreeSet::new())));
            let Some((pages, running, parked)) = entry else {
                continue;
            };
            let movable = readers.is_some_and(|readers| {
                readers.into_iter().all(|(ws, suspended)| {
                    match self.table.owner(ws).ok().flatten() {
                        Some(pid) if may_move(pid, suspended) => {
                            if suspended {
                                parked.insert(pid);
                            } else {
                                running.insert(pid);
                            }
                            true
                        }
                        _ => false,
                    }
                })
            });
            if movable {
                pages.push((location, id));
            } else {
                *entry = None;
            }
        }
        Some((spc, chunks, by_chunk))
    }

    /// The list chunks a compaction empties for `wanted` whole chunks
    /// ([`ListCompaction`]): free list slots scattered across partly used
    /// chunks are room a bind or a restore of whole chunks cannot take, and
    /// moving the pages of the emptiest chunks into the free slots of the
    /// rest frees those chunks whole. A chunk is a source only where every
    /// slot it holds is a page that may move (`may_move` names the
    /// processes whose sets may be fenced for it, and the parked ones whose
    /// restore may wait for it); its pages go to the free slots of the
    /// chunks left, fullest first. Fewest pages first, then the lowest
    /// chunk: as many chunks as `wanted`, or as many as the free slots
    /// elsewhere take. `None` where none empties.
    #[must_use]
    pub fn plan_list_compaction(
        &self,
        wanted: u32,
        may_move: MayMove<'_>,
    ) -> Option<ListCompaction> {
        let (spc, chunks, by_chunk) = self.list_chunk_pages(may_move)?;
        let mut room: u64 = chunks
            .iter()
            .filter(|chunk| chunk.settled)
            .map(|chunk| u64::from(spc - chunk.used))
            .sum();
        #[allow(clippy::type_complexity)]
        let mut sources: Vec<(u32, u32, &BTreeSet<ProcessId>, &BTreeSet<ProcessId>)> = chunks
            .iter()
            .filter(|chunk| chunk.settled && chunk.used > 0)
            .filter_map(|chunk| {
                let (pages, running, parked) = by_chunk.get(&chunk.logical)?.as_ref()?;
                (pages.len() as u32 == chunk.used).then_some((
                    chunk.used,
                    chunk.logical,
                    running,
                    parked,
                ))
            })
            .collect();
        sources.sort_unstable_by_key(|&(used, logical, ..)| (used, logical));
        let (mut picked, mut moving) = (Vec::new(), 0u64);
        let (mut readers, mut parked) = (BTreeSet::new(), BTreeSet::new());
        for (used, logical, running, held) in sources {
            if picked.len() as u32 >= wanted {
                break;
            }
            // Emptied, it is no destination: its free slots leave the room.
            let left = room - u64::from(spc - used);
            if moving + u64::from(used) > left {
                continue;
            }
            room = left;
            moving += u64::from(used);
            picked.push(logical);
            readers.extend(running.iter().copied());
            parked.extend(held.iter().copied());
        }
        (!picked.is_empty()).then(|| ListCompaction {
            chunks: picked,
            pages: moving as usize,
            readers: readers.into_iter().collect(),
            parked: parked.into_iter().collect(),
        })
    }

    /// Takes `plan`'s pages out of their chunks: each chunk whose every
    /// slot still holds a page that may move (`fenced` names the running
    /// processes whose sets are fenced now, none of their fires in flight;
    /// the plan's parked ones are held from their restore) has its
    /// pages given fresh slots in the partly filled chunks outside the plan
    /// and pinned as a move's until [`Self::commit_relocate`] or
    /// [`Self::abort_relocate`]. A chunk that changed since the plan (a
    /// page taken into it, a reader not fenced) is left where it lies.
    /// `None` where no chunk of the plan still empties.
    pub fn prepare_relocate(
        &mut self,
        plan: &ListCompaction,
        fenced: &HashSet<ProcessId>,
    ) -> Option<KvRelocateTxn> {
        let may_move = |pid: ProcessId, suspended: bool| {
            if suspended {
                plan.parked.contains(&pid)
            } else {
                fenced.contains(&pid)
            }
        };
        let (_, chunks, by_chunk) = self.list_chunk_pages(&may_move)?;
        let used: HashMap<u32, (u32, bool)> = chunks
            .iter()
            .map(|chunk| (chunk.logical, (chunk.used, chunk.settled)))
            .collect();
        let mut emptied = Vec::new();
        let mut pages = Vec::new();
        for &logical in &plan.chunks {
            let Some(&(count, true)) = used.get(&logical) else {
                continue;
            };
            let Some(Some((movable, ..))) = by_chunk.get(&logical) else {
                continue;
            };
            if movable.len() as u32 != count || count == 0 {
                continue;
            }
            emptied.push(logical);
            pages.extend(movable.iter().copied());
        }
        if pages.is_empty() {
            return None;
        }
        // Chunks a copy or a chunk move holds take no page either.
        let outside: BTreeSet<u32> = emptied
            .iter()
            .copied()
            .chain(
                chunks
                    .iter()
                    .filter(|chunk| !chunk.settled)
                    .map(|chunk| chunk.logical),
            )
            .collect();
        let to = self.pool.try_alloc_outside(pages.len(), &outside)?;
        self.table
            .pin_swap_locations(pages.iter().map(|&(location, _)| location));
        Some(KvRelocateTxn {
            pages: pages
                .into_iter()
                .zip(to)
                .map(|((location, from), to)| (location, from, to))
                .collect(),
            shape: self.shape,
            chunks: emptied,
        })
    }

    /// The copy of `txn` landed: each page reads its new slot from now on
    /// (every working set's table is read again), and its old slot goes
    /// back, its chunk free once all of them have. No fire reads the old
    /// slots: their readers were fenced with no fire in flight, and the
    /// copy out of them, ordered behind every fire before it, has landed.
    pub fn commit_relocate(&mut self, txn: KvRelocateTxn) -> Result<usize, KvStoreError> {
        let replacements: Vec<_> = txn
            .pages
            .iter()
            .map(|&(location, from, to)| {
                (
                    location,
                    KvPageBacking::Resident(from),
                    KvPageBacking::Resident(to),
                )
            })
            .collect();
        if let Err(error) = self.table.replace_backings(&replacements) {
            self.abort_relocate(txn);
            return Err(error.into());
        }
        let freed = self
            .table
            .unpin_swap_locations(txn.pages.iter().map(|&(location, ..)| location));
        for flat in self.flat.values_mut() {
            *flat = None;
        }
        let count = txn.pages.len();
        self.pool
            .release_reserved(txn.pages.into_iter().map(|(_, from, _)| from).collect());
        self.end_move(freed);
        Ok(count)
    }

    /// `txn` did not land: its pages stay where they lay, and the slots
    /// taken for them go back (the copy into them settled first).
    pub fn abort_relocate(&mut self, txn: KvRelocateTxn) {
        let freed = self
            .table
            .unpin_swap_locations(txn.pages.iter().map(|&(location, ..)| location));
        self.pool
            .release_reserved(txn.pages.into_iter().map(|(_, _, to)| to).collect());
        self.end_move(freed);
    }

    /// The disk tier's free slots, idle cache's not counted.
    #[must_use]
    pub fn disk_free_slots(&self) -> usize {
        self.disk_pool.available()
    }

    /// Chunks [`Self::take_disk_chunks`] took, free again.
    pub fn give_disk_chunks(&mut self, chunks: &[u32], per_chunk: u32) {
        let per_chunk = per_chunk.max(1);
        let firsts: Vec<DiskKvSlotId> = chunks
            .iter()
            .map(|&chunk| DiskKvSlotId(chunk * per_chunk))
            .collect();
        self.disk_pool.release_runs(&firsts, per_chunk);
        self.below_cached.set(None);
        self.droppable_below.set(None);
        self.guest_disk_slots = self
            .guest_disk_slots
            .saturating_sub(firsts.len() as u32 * per_chunk);
    }

    /// Parked sets' pages on the host going on down to disk, the sets of
    /// `order` first (the victim order: what would be evicted first), until
    /// `wanted` pages go or the disk is full: host room for a parked
    /// process's role slots, older parked kv going below the newer role
    /// state. A page the disk tier keeps a copy of
    /// settles on it, nothing copied; a page another move holds stays.
    /// `None` when nothing goes.
    pub fn prepare_parked_spill(
        &mut self,
        wanted: usize,
        order: &[WorkingSetId],
    ) -> Option<KvSpillTxn> {
        let shape = self.shape?;
        let mut room = self.disk_pool.available();
        let mut pages: Vec<(TriePageLocation, HostKvSlotId)> = Vec::new();
        let mut taken: HashSet<TriePageLocation> = HashSet::new();
        for &ws in order {
            if pages.len() >= wanted || room == 0 {
                break;
            }
            let Ok(swapped) = self.table.swapped_pages(&HashSet::from([ws])) else {
                continue;
            };
            let mut group: Vec<(TriePageLocation, HostKvSlotId)> = swapped
                .into_iter()
                .filter_map(|(location, backing)| match backing {
                    KvPageBacking::Swapped(host) => Some((location, host)),
                    _ => None,
                })
                .filter(|&(location, _)| {
                    !taken.contains(&location) && !self.table.page_location_pinned(location)
                })
                .collect();
            let before = group.len();
            self.settle_host_on_disk(&mut group);
            let settled = before - group.len();
            group.truncate(room.min(wanted.saturating_sub(pages.len() + settled)));
            room -= group.len();
            taken.extend(group.iter().map(|&(at, _)| at));
            pages.extend(group);
        }
        self.below_cached.set(None);
        self.droppable_below.set(None);
        if pages.is_empty() {
            return None;
        }
        let disk = self
            .disk_pool
            .try_alloc_n(pages.len())
            .expect("counted above");
        self.table
            .pin_swap_locations(pages.iter().map(|&(location, _)| location));
        tracing::debug!(target: "pie::kv", pages = pages.len(), "parked kv host to disk");
        Some(KvSpillTxn {
            pages: pages
                .into_iter()
                .zip(disk)
                .map(|((location, host), disk)| (location, host, disk))
                .collect(),
            shape,
            records: HashMap::new(),
        })
    }

    /// The spill that makes host room for suspending `working_sets`, where
    /// the host is short of it: host room is for what is paused before what
    /// is cached, and cache that goes on down to disk is not dropped.
    pub fn prepare_suspend_spill(
        &mut self,
        working_sets: &HashSet<WorkingSetId>,
        keys: impl Keys,
    ) -> Option<KvSpillTxn> {
        self.shape?;
        let pages = self.table.movable_resident_pages(working_sets).ok()?.len();
        let short = pages.saturating_sub(self.host_pool.available());
        if short == 0 {
            return None;
        }
        self.prepare_spill(short, keys)
    }

    /// Moves the host pages of `group` the disk tier keeps a copy of onto
    /// their copies, giving their host slots back; leaves the rest.
    fn settle_host_on_disk(&mut self, group: &mut Vec<(TriePageLocation, HostKvSlotId)>) {
        let settled: Vec<(TriePageLocation, HostKvSlotId, DiskKvSlotId)> = group
            .iter()
            .filter_map(|&(location, host)| Some((location, host, self.copy_of(&location)?)))
            .collect();
        if settled.is_empty() {
            return;
        }
        let replacements: Vec<_> = settled
            .iter()
            .map(|&(location, host, slot)| {
                (
                    location,
                    KvPageBacking::Swapped(host),
                    KvPageBacking::OnDisk(slot),
                )
            })
            .collect();
        if self.table.replace_backings(&replacements).is_err() {
            return;
        }
        if let Some(persisted) = &mut self.persisted {
            settled
                .iter()
                .for_each(|&(_, _, slot)| persisted.back(slot));
        }
        let gone: HashSet<TriePageLocation> = settled.iter().map(|&(at, ..)| at).collect();
        group.retain(|(location, _)| !gone.contains(location));
        tracing::debug!(target: "pie::kv", pages = settled.len(), "host pages settled on disk");
        self.host_pool
            .release_reserved(settled.into_iter().map(|(_, host, _)| host).collect());
    }

    /// The spill's pages are on disk now: their host slots go back, and a
    /// persistent tier keeps what it journals. Answers the pages moved.
    pub fn commit_spill(&mut self, mut txn: KvSpillTxn) -> usize {
        let replacements: Vec<_> = txn
            .pages
            .iter()
            .map(|&(location, host, disk)| {
                (
                    location,
                    KvPageBacking::Swapped(host),
                    KvPageBacking::OnDisk(disk),
                )
            })
            .collect();
        if self.table.replace_backings(&replacements).is_err() {
            self.abort_spill(txn);
            return 0;
        }
        let freed = self
            .table
            .unpin_swap_locations(txn.pages.iter().map(|&(location, ..)| location));
        let records = std::mem::take(&mut txn.records);
        if !records.is_empty() {
            let landed: Vec<_> = txn
                .pages
                .iter()
                .map(|&(location, _, disk)| (location, KvPageBacking::OnDisk(disk)))
                .collect();
            self.keep_demoted(landed, records);
        }
        let count = txn.pages.len();
        self.host_pool
            .release_reserved(txn.pages.into_iter().map(|(_, host, _)| host).collect());
        self.below_cached.set(None);
        self.droppable_below.set(None);
        self.end_move(freed);
        tracing::debug!(target: "pie::kv", pages = count, "spilled host to disk");
        count
    }

    /// The pages stay on the host; the disk slots go back.
    pub fn abort_spill(&mut self, txn: KvSpillTxn) {
        let freed = self
            .table
            .unpin_swap_locations(txn.pages.iter().map(|&(location, ..)| location));
        self.disk_pool
            .release_reserved(txn.pages.into_iter().map(|(.., disk)| disk).collect());
        self.end_move(freed);
    }

    /// Idle cache pages parked on the host that only the cache reads: what
    /// a host short of room can move on down to disk.
    pub fn host_cache_pages(&self) -> u32 {
        self.below_device_cache_pages().0
    }

    /// Idle cache pages below the device that only the cache reads,
    /// `(on the host, on disk)`: dropping them frees their slots for any
    /// demand. A count stands as [`Self::reclaimable_cache_pages`]' does.
    pub fn below_device_cache_pages(&self) -> (u32, u32) {
        if self.host_pool.capacity() == 0 && self.disk_pool.capacity() == 0 {
            return (0, 0);
        }
        if let Some((at, took, pages)) = self.below_cached.get()
            && at.elapsed() < std::time::Duration::from_secs(1).max(took * 50)
        {
            return pages;
        }
        let began = Instant::now();
        let roots = self.table.unused_cache_roots();
        let pages = self.table.cache_only_below_device(&roots, &[]);
        self.below_cached.set(Some((began, began.elapsed(), pages)));
        pages
    }

    /// [`Self::below_device_cache_pages`] as a demand may take it by
    /// dropping idle cache ([`Self::drop_entries_off_device`]): below the
    /// device no key keeps it, it only orders what goes. A count stands as
    /// [`Self::below_device_cache_pages`]' does, under the keys' revision;
    /// under keys of none it is taken each time.
    pub fn droppable_below_device(&self, keys: impl Keys) -> (u32, u32) {
        if self.host_pool.capacity() == 0 && self.disk_pool.capacity() == 0 {
            return (0, 0);
        }
        let revision = keys.revision();
        if let (Some(revision), Some((at, took, read, pages))) =
            (revision, self.droppable_below.get())
            && read == revision
            && at.elapsed() < std::time::Duration::from_secs(1).max(took * 50)
        {
            return pages;
        }
        let began = Instant::now();
        let unused: HashSet<NodeId> = self.table.unused_cache_roots().into_iter().collect();
        let (mut droppable, kept): (Vec<NodeId>, Vec<NodeId>) = (Vec::new(), Vec::new());
        for entry in self.cache.values() {
            let root = entry.path.terminal;
            if !unused.contains(&root) {
                continue;
            }
            // Keys order what goes, never keep it (PLEX 1.1): all of it is room.
            droppable.push(root);
        }
        let pages = self.table.cache_only_below_device(&droppable, &kept);
        if let Some(revision) = revision {
            self.droppable_below
                .set(Some((began, began.elapsed(), revision, pages)));
        }
        pages
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
            shape: self.shape,
            records: HashMap::new(),
        }))
    }

    pub fn commit_promote(&mut self, mut txn: KvResidencyTxn) -> Result<ReaderPin, KvStoreError> {
        let pinned = std::mem::take(&mut txn.pinned);
        match self.commit_restore(txn) {
            Ok(_) => Ok(ReaderPin {
                nodes: pinned,
                outside: false,
            }),
            Err(error) => {
                self.release_reader(ReaderPin {
                    nodes: pinned,
                    outside: false,
                });
                Err(error)
            }
        }
    }

    /// Each cache entry's pages by key, from token 0: the runs this worker
    /// can give whole.
    #[cfg(test)]
    pub fn cached_runs(&self) -> Vec<Vec<Hash256>> {
        self.cached_runs_of(&self.cache_ids())
    }

    pub fn cache_ids(&self) -> Vec<u64> {
        self.cache.keys().copied().collect()
    }

    /// [`Self::cached_runs`] of the entries of `ids` still here.
    pub fn cached_runs_of(&self, ids: &[u64]) -> Vec<Vec<Hash256>> {
        ids.iter()
            .filter_map(|id| {
                let entry = self.cache.get(id)?;
                let mut keys = self.table.path_keys(entry.path.terminal)?;
                keys.truncate(entry.path.pages as usize);
                (!keys.is_empty()).then_some(keys)
            })
            .collect()
    }

    pub fn node_ids(&self) -> Vec<NodeId> {
        self.table.node_ids()
    }

    /// Every sealed page's key in the nodes of `ids` still here, on any tier.
    pub fn sealed_key_list(&self, ids: &[NodeId]) -> Vec<Hash256> {
        self.table.sealed_key_list(ids)
    }

    /// Those of them off the device: `false` on the host, `true` on disk.
    pub fn off_device_key_list(&self, ids: &[NodeId]) -> Vec<(Hash256, bool)> {
        self.table.off_device_key_list(ids)
    }

    /// Host slots to land pages in that come from another worker; given
    /// back with [`Self::free_host_slots`] unless an import takes them.
    /// The tokens written to `ws`, as far as every write since is known.
    pub fn logged(&self, ws: WorkingSetId) -> Result<&[u32], KvStoreError> {
        Ok(self.table.logged(ws)?)
    }

    ///
    /// None on a shared pool: its host slots are slot refs of the host
    /// tier, which the engine's byte reads and writes (`read_host_kv`) do
    /// not name.
    /// The shared pool's list shape the pages are slots of; `None` for a
    /// pinned pool.
    pub fn shared_list(&self) -> Option<::engine::chunk::ShapeId> {
        self.shape
    }

    pub fn alloc_host_slots(&mut self, n: usize) -> Option<Vec<HostKvSlotId>> {
        if self.shape.is_some() {
            return None;
        }
        self.host_pool.try_alloc_n(n)
    }

    /// A shared pool's host slots for a read or a write none of the
    /// planner's: kv list slots of its host tier, whose bytes the engine
    /// reads and writes by those refs (`Tiers::read`), not by
    /// `read_host_kv`. `None` off a shared pool, or if the host lacks them.
    #[cfg_attr(
        target_arch = "wasm32",
        allow(dead_code, reason = "the browser has no host or disk tier")
    )]
    pub fn alloc_tier_slots(&mut self, n: usize) -> Option<Vec<HostKvSlotId>> {
        self.shape?;
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
        self.check_given_back(&device, "reserved backings given back");
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
        self.list(self.kept, node);
        tracing::debug!(target: "pie::kv", pages, "imported");
        retention::idle_cache_appeared();
        self.note_order(Err(self.kept));
        self.cache.insert(
            self.kept,
            KeptEntry {
                path: KeptPath {
                    terminal: node,
                    pages,
                },
                owner: None,
                last_use: Instant::now(),
            },
        );
        self.note_change(node, true);
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
        self.readers += 1;
        Some((
            ReaderPin {
                nodes: vec![last.node],
                outside: true,
            },
            ids,
        ))
    }

    /// `ws` as a snapshot reads it, pinned where it stands until the pin is
    /// released.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export(
        &mut self,
        ws: WorkingSetId,
    ) -> Result<(ReaderPin, page_table::ExportedSet), KvStoreError> {
        let set = self.table.exported(ws)?;
        let nodes = self.table.pin_working_sets(&HashSet::from([ws]))?;
        self.readers += 1;
        Ok((
            ReaderPin {
                nodes,
                outside: true,
            },
            set,
        ))
    }

    /// A working set of its own from what `slots` hold.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn import(
        &mut self,
        set: page_table::ExportedSet,
        slots: Vec<HostKvSlotId>,
    ) -> Result<WorkingSetId, KvStoreError> {
        let backings = slots.into_iter().map(KvPageBacking::Swapped).collect();
        let ws = self.table.imported(set, backings)?;
        self.refresh_flat(ws);
        // Its rows came keyed: it shares nothing until a scope it asks.
        self.private_scope(ws);
        Ok(ws)
    }

    pub fn release_reader(&mut self, pin: ReaderPin) {
        let freed = self.table.unpin_terminals(&pin.nodes);
        let epoch = self.current_epoch();
        self.recycle_backings(freed, epoch);
        self.retire_idle();
        if pin.outside {
            self.readers -= 1;
            if let Some(signal) = &self.reader_signal {
                (signal.0)(super::slots::RoomCount {
                    tier: super::slots::RoomTier::Reader,
                    chunks: 0,
                    slots: 0,
                });
            }
        }
    }

    /// Whether a slot below the device is in use.
    pub fn holds_below_device(&self) -> bool {
        self.host_pool.available() < self.host_pool.capacity() as usize
            || self.disk_pool.available() < self.disk_pool.capacity() as usize
    }

    /// Whether a reader outside the fleet holds pages here
    /// ([`Self::pin_resident`], [`Self::export`]).
    pub fn readers_hold(&self) -> bool {
        self.readers > 0
    }

    /// Told when pages may have joined the cache (a fire's pages settle, a
    /// working set is released): arm it before looking.
    pub fn cached(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.cached)
    }

    /// Told when an entry joins or leaves the cache: arm it before looking.
    pub fn changed(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.changed)
    }

    /// The entry ending at `terminal` joined the cache, or is leaving it
    /// (before its pages are released): while a prompt waits, its path's
    /// page keys are kept for [`Self::take_changes`].
    fn note_change(&mut self, terminal: NodeId, joined: bool) {
        if crate::store::waiting::watched() {
            let keys = self
                .table
                .path_pages(terminal)
                .into_iter()
                .filter_map(|(_, key)| key);
            if joined {
                self.joined.extend(keys);
            } else {
                self.left.extend(keys);
            }
        }
        self.changed.notify_waiters();
    }

    /// The page keys of the entries that joined the cache and of those that
    /// left it since the last call, `(joined, left)`.
    pub fn take_changes(&mut self) -> (HashSet<Hash256>, HashSet<Hash256>) {
        (
            std::mem::take(&mut self.joined),
            std::mem::take(&mut self.left),
        )
    }

    /// Pages may have joined the cache.
    pub fn note_cached(&self) {
        self.cached.notify_waiters();
    }

    /// A move that found its pages in another waits for it here.
    pub fn moves_ended(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.moved)
    }

    /// A parked set reads nothing: whatever it shares is free to move, and
    /// it takes back what moved before it runs.
    pub fn set_parked(&mut self, ws: WorkingSetId, parked: bool) {
        self.table.set_suspended(&HashSet::from([ws]), parked);
        // A peer's moves refresh only its own map.
        self.refresh_flat(ws);
    }

    /// Whether a move holds any page `working_sets` map: what they name is
    /// not where the flat map says until it ends.
    pub fn in_move(&self, working_sets: &HashSet<WorkingSetId>) -> Result<bool, KvStoreError> {
        Ok(self.table.in_move(working_sets)?)
    }

    pub fn fork(&mut self, ws: WorkingSetId) -> Result<WorkingSetId, KvStoreError> {
        let child = self.table.fork(ws)?;
        self.refresh_flat(child);
        // The child's rows chain from its parent's, and share as they do.
        if let Some(scope) = self.scopes.get(&ws).cloned() {
            self.scopes.insert(child, scope);
        }
        Ok(child)
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
        // Its fires through `epoch` may still read what its path holds,
        // whatever stays cached of it.
        if let Ok(Some(terminal)) = self.table.terminal(ws) {
            self.table.stamp_path(terminal, None, epoch);
        }
        // Only a set that shares leaves its prefix for others to find.
        if self.keeps && self.shares(ws) {
            // What its decodes folded seals as it goes, unless a prompt
            // sealed it already.
            if let Err(error) = self.seal_folded(ws, false) {
                tracing::debug!(%error, "kv: sealing what a released set's decodes folded");
            }
            self.keep_sealed_prefix(ws);
        }
        let freed = self.table.release_working_set_backings(ws);
        tracing::debug!(target: "runtime::store::kv", ?ws, ?freed, "a set released");
        self.recycle_backings(freed, epoch);
        // What no fire in flight reads is free now, not at a next fire.
        self.retire_idle();
        self.flat.remove(&ws);
        self.scopes.remove(&ws);
        self.note_cached();
    }

    fn keep_sealed_prefix(&mut self, ws: WorkingSetId) {
        let owner = self.owner_of(ws);
        let all = self.table.sealed_resident_prefix(ws).unwrap_or(0);
        let prompt = self.table.prompt_pages(ws).unwrap_or(0).min(all);
        if prompt > 0 {
            self.keep(ws, prompt, owner);
        }
        if all > prompt {
            self.keep(ws, all, owner);
        }
    }

    fn keep(&mut self, ws: WorkingSetId, pages: u64, owner: Option<ProcessId>) {
        let Ok(path) = self.table.keep_prefix(ws, pages) else {
            return;
        };
        self.table.lease_cache_root(path.terminal);
        self.kept += 1;
        if let Some(owner) = owner {
            self.owners.entry(owner).or_default().push(self.kept);
        }
        self.list(self.kept, path.terminal);
        tracing::debug!(target: "pie::kv", pages, "kept");
        retention::idle_cache_appeared();
        let terminal = path.terminal;
        self.note_order(owner.ok_or(self.kept));
        self.cache.insert(
            self.kept,
            KeptEntry {
                path,
                owner,
                last_use: Instant::now(),
            },
        );
        self.note_change(terminal, true);
    }

    /// The device pages of each of `groups` that would leave the device
    /// with it: resident, read by none of the running working sets outside
    /// it, pinned by nothing, walked once for all the groups
    /// ([`KvPageTable::movable_resident_pages_each`]).
    pub fn movable_resident_pages_each(
        &self,
        groups: &[HashSet<WorkingSetId>],
    ) -> Vec<Vec<PhysicalKvPageId>> {
        self.table
            .movable_resident_pages_each(groups)
            .into_iter()
            .map(|pages| pages.into_iter().map(|(_, page)| page).collect())
            .collect()
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
            self.keep(ws, pages, owner);
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

    /// What a fire folds is frozen from its prepare, before its rows are
    /// written: the one folded length a working set has. [`KvStore::fold`]
    /// seals the pages once the fire has settled.
    pub fn freeze(&mut self, ws: WorkingSetId, to: u64) -> Result<u64, KvStoreError> {
        Ok(self.table.freeze(ws, to)?)
    }

    /// A fire on its way folds `ws` by a count placed when it settles: it
    /// is not forked or exported until [`Self::fold_settled`].
    pub fn unsettle(&mut self, ws: WorkingSetId) -> Result<(), KvStoreError> {
        Ok(self.table.unsettle(ws)?)
    }

    /// That fire settled, `placed` when its count was: a fold never told
    /// leaves the folded length a lower bound.
    pub fn fold_settled(&mut self, ws: WorkingSetId, placed: bool) -> Result<(), KvStoreError> {
        Ok(self.table.fold_settled(ws, placed)?)
    }

    pub fn unsettled(&self, ws: WorkingSetId) -> Result<u32, KvStoreError> {
        Ok(self.table.unsettled(ws)?)
    }

    /// Whether [`Self::folded`] is all `ws` has folded.
    pub fn folded_exact(&self, ws: WorkingSetId) -> Result<bool, KvStoreError> {
        Ok(self.table.folded_exact(ws)?)
    }

    /// `decoded` when a decode wrote them. What a decode folds is frozen and not sealed: a decode runs
    /// ahead of the program that reads its tokens, which may take back the
    /// rows past where it stops ([`Self::truncate`]). Its rows seal with the
    /// next prompt the set folds, or as the set is released.
    pub fn fold(&mut self, ws: WorkingSetId, to: u64, decoded: bool) -> Result<u64, KvStoreError> {
        self.table.freeze(ws, to)?;
        if decoded {
            return Ok(0);
        }
        self.seal_folded(ws, true)
    }

    /// Seals the whole pages `ws` has folded and logged; `prompt` when they
    /// reach its prompt.
    fn seal_folded(&mut self, ws: WorkingSetId, prompt: bool) -> Result<u64, KvStoreError> {
        let folded = self.table.folded(ws)?;
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
        let slots = hash::chain_slots(&self.domain_of(ws), prev, start, &tokens);
        self.seal_rows(ws, start, &slots, self.page_size)?;
        if prompt {
            self.table.reach_prompt(ws, full)?;
        }
        tracing::debug!(target: "pie::kv", pages = full - sealed, prompt, "sealed");
        Ok(full - sealed)
    }

    /// `ws` ends at row `rows` ([`KvPageTable::truncate`]): what a program
    /// that ran ahead of its stop takes back once its fires have landed.
    /// Every fire of the set has settled, so what this frees waits on no
    /// fire in flight.
    /// Refused with [`KvStoreError::MoveInFlight`] while the set is in a
    /// move: its pages pinned for the move, what this freed went back again
    /// as the move committed.
    pub fn truncate(&mut self, ws: WorkingSetId, rows: u64) -> Result<(), KvStoreError> {
        if self.table.in_move(&HashSet::from([ws]))? {
            return Err(KvStoreError::MoveInFlight);
        }
        let freed = self.table.truncate(ws, rows, self.page_size)?;
        tracing::debug!(target: "runtime::store::kv", ?ws, rows, ?freed, "a set cut");
        let epoch = self.retired_through();
        self.recycle_backings(freed, epoch);
        self.refresh_flat(ws);
        Ok(())
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

    /// Maps in only when the pages `ws` holds are the run's first. What is
    /// mapped in is sealed and not frozen: the folded length is what a pass
    /// states, whatever the cache held, and a row written into a mapped
    /// page copies it first, as any write to a sealed page does.
    pub fn attach_run(
        &mut self,
        ws: WorkingSetId,
        run: &[TriePageLocation],
        limit: u64,
    ) -> Result<u64, KvStoreError> {
        let held = self.table.mapped_len(ws)?;
        let found = run.len();
        let run = &run[..run.len().min(limit as usize)];
        // What attaching maps is the trie path to the run's last page. A run
        // found across several copies of one content (a node start's copies,
        // computed by passes that raced) may leave that path, and the path's
        // page is then another copy: keys name a page's rows and all its
        // prefix, so the path maps as far as its keys are the run's and its
        // pages are on the device.
        let path = run.last().map_or_else(Vec::new, |last| {
            self.table.path_locations(last.node, last.local)
        });
        let keyed = path
            .iter()
            .zip(run)
            .take_while(|(on, at)| {
                self.table
                    .page_key_at(on)
                    .is_some_and(|key| Some(key) == self.table.page_key_at(at))
            })
            .count();
        let path = &path[..self.table.resident_run(&path[..keyed])];
        // Why the found pages within the limit were not all mapped, for the
        // pass's log line.
        let short = |why: &'static str| {
            tracing::debug!(
                target: "pie::kv",
                found,
                limit,
                held,
                mapped = path.len(),
                why,
                "attach: found pages not mapped"
            );
        };
        if keyed < run.len() {
            short("another copy's path leaves the run's keys");
        } else if path.len() < keyed {
            short("a page is moving or below the device");
        }
        let Some(&last) = path.last().filter(|_| path.len() as u64 > held) else {
            return Ok(0);
        };
        if self.table.path_prefix_len(last.node, last.local) != path.len() as u64 {
            short("the path is a selection that does not start at row 0");
            return Ok(0);
        }
        match self.table.attach(ws, last.node, last.local) {
            Ok(_) => {}
            Err(KvTableError::BadRange { .. }) => {
                short("the pages the set holds are not the path's");
                return Ok(0);
            }
            Err(error) => return Err(error.into()),
        }
        let run = path;
        let pages = run.len() as u64 - held;
        // What was mapped in is a prompt's.
        self.table.reach_prompt(ws, run.len() as u64)?;
        self.refresh_flat(ws);
        // The cache's entries that were used whole go last; one that holds
        // more than was used ages by what it holds alone.
        let now = Instant::now();
        let mut changed = Vec::new();
        for node in self.table.lineage(last.node) {
            for id in self.by_owner.get(&node).into_iter().flatten() {
                if let Some(entry) = self.cache.get_mut(id)
                    && self
                        .table
                        .within(entry.path.terminal, last, run.len() as u64)
                {
                    entry.last_use = now;
                    changed.push(entry.owner.ok_or(*id));
                }
            }
        }
        changed.into_iter().for_each(|of| self.note_order(of));
        Ok(pages)
    }

    fn list(&mut self, id: u64, terminal: NodeId) {
        if let Some(owner) = self.table.path_owner(terminal) {
            self.by_owner.entry(owner).or_default().push(id);
        }
    }

    fn unlist(&mut self, id: u64, terminal: NodeId) {
        let Some(owner) = self.table.path_owner(terminal) else {
            return;
        };
        if let Some(ids) = self.by_owner.get_mut(&owner) {
            ids.retain(|listed| *listed != id);
            if ids.is_empty() {
                self.by_owner.remove(&owner);
            }
        }
    }

    fn recycle_backings(&mut self, backings: Vec<KvPageBacking>, epoch: u64) {
        if !backings.is_empty() {
            tracing::debug!(target: "runtime::store::kv", epoch, ?backings, "backings let go");
        }
        let resident: Vec<PhysicalKvPageId> = backings
            .iter()
            .filter_map(|backing| match backing {
                KvPageBacking::Resident(id) => Some(*id),
                _ => None,
            })
            .collect();
        self.check_given_back(&resident, "backings let go");
        let mut resident = Vec::new();
        let mut swapped = Vec::new();
        for backing in backings {
            match backing {
                KvPageBacking::Resident(id) => resident.push(id),
                swapped_backing => swapped.push(swapped_backing),
            }
        }
        if let Some(reclaiming) = &mut self.reclaiming {
            reclaiming.extend(&resident);
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

        self.hand_out();
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

    /// The chosen pool pages of a beam search's finish
    /// ([`Self::relist_beam_with`]), each checked: past the set's backed pages,
    /// or named twice, is a bad list.
    fn chosen_pages(
        &self,
        ws: WorkingSetId,
        f0: u64,
        order: &[u32],
    ) -> Result<Vec<(u64, PhysicalKvPageId)>, KvStoreError> {
        let flat = self.table.flatten(ws)?;
        let mut chosen: Vec<(u64, PhysicalKvPageId)> = Vec::with_capacity(order.len());
        for &q in order {
            let index = f0 + u64::from(q);
            let page = *flat.get(index as usize).ok_or(KvStoreError::BadWriteSet {
                reason: "a chosen beam names a pool page past its set's backed pages",
            })?;
            if chosen.iter().any(|&(_, seen)| seen == page) {
                return Err(KvStoreError::BadWriteSet {
                    reason: "a chosen beam names one pool page twice",
                });
            }
            chosen.push((index, page));
        }
        Ok(chosen)
    }

    /// What a beam search's finish of `ws` from `f0` by `order`
    /// ([`Self::relist_beam_with`]) needs before it can run: nothing, the end of
    /// a move the set is in (a suspend or a sweep pins its old tail, and a
    /// relist then would leave its pages named twice), or copies of the
    /// chosen pages something else still reads.
    pub fn relist_beam_needs(
        &mut self,
        ws: WorkingSetId,
        f0: u64,
        order: &[u32],
    ) -> Result<RelistNeeds, KvStoreError> {
        if self.table.in_move(&HashSet::from([ws]))? {
            return Ok(RelistNeeds::MoveEnd(Arc::clone(&self.moved)));
        }
        let chosen = self.chosen_pages(ws, f0, order)?;
        let mut shared = Vec::new();
        for &(index, page) in &chosen {
            if !self.table.privately_writable(ws, index)? {
                shared.push(page);
            }
        }
        Ok(RelistNeeds::Copies(shared))
    }

    /// Device pages given back must be in no live set's table: one still
    /// named reads or writes a page another may take (a set's own table
    /// naming a page whose segment was released). Said at ERROR naming the
    /// set and the page, in a debug build (a walk of every set's table a
    /// give-back).
    fn check_given_back(&self, ids: &[PhysicalKvPageId], what: &str) {
        if ids.is_empty() || !cfg!(debug_assertions) {
            return;
        }
        let given: std::collections::HashSet<u32> = ids.iter().map(|id| id.0).collect();
        for (ws, pages) in self.table.every_resident_table() {
            if let Some(page) = pages.iter().find(|page| given.contains(page)) {
                tracing::error!(
                    target: "runtime::store::kv",
                    ?ws,
                    page,
                    what,
                    "kv: a page given back is still named by a live set's table"
                );
            }
        }
    }

    /// The slot refs of `ws`'s pages on the device as its table stands now
    /// (each page id less the pool's base), `None` where its table does not
    /// resolve (a page below the device): what a lane of its names must lie
    /// among, for a check at the engine's door.
    pub fn device_refs_of(&mut self, ws: WorkingSetId) -> Option<std::collections::HashSet<u32>> {
        let base = self.pool.base();
        let table = self.flat_table(ws).ok()?;
        Some(
            table
                .iter()
                .map(|page| page.0.saturating_sub(base))
                .collect(),
        )
    }

    /// The shared pool's kv list shape its pages are slots of, if any.
    pub fn list_shape(&self) -> Option<::engine::chunk::ShapeId> {
        self.shape
    }

    /// Whether `ws` is in a move now.
    pub fn set_in_move(&self, ws: WorkingSetId) -> bool {
        self.table.in_move(&HashSet::from([ws])).unwrap_or(false)
    }

    /// A beam search's finish (`BEAM-PAGE-LISTS.md` §6.6): the set's pages
    /// from `f0` become its chosen beam's, page `f0 + j` the pool page
    /// `order[j]` (pool page `q` is the set's page `f0 + q` now), for each
    /// column `j` the search wrote; every other page past `f0` is let go
    /// once the fires that may read it retire. A chosen page only this set
    /// reads moves, nothing copied; one something else reads too (a cache
    /// entry or another set's path through the old tail) is named by its
    /// copy in `copies`, `(page, copy)`, made on the device before this:
    /// moved, both would name it, and it would go back twice (a suspend
    /// committed under a relist giving back the chosen pages the new tail
    /// still names). `false`, nothing done, where the set is
    /// in a move or `copies` no longer covers what is shared: the caller
    /// asks [`Self::relist_beam_needs`] again.
    pub fn relist_beam_with(
        &mut self,
        ws: WorkingSetId,
        f0: u64,
        order: &[u32],
        copies: &[(PhysicalKvPageId, PhysicalKvPageId)],
    ) -> Result<bool, KvStoreError> {
        if self.table.in_move(&HashSet::from([ws]))? {
            return Ok(false);
        }
        let chosen = self.chosen_pages(ws, f0, order)?;
        let mut listed = Vec::with_capacity(chosen.len());
        for &(index, page) in &chosen {
            if self.table.privately_writable(ws, index)? {
                listed.push(page);
                continue;
            }
            match copies.iter().find(|&&(from, _)| from == page) {
                Some(&(_, copy)) => listed.push(copy),
                None => return Ok(false),
            }
        }
        let page_size = self.page_size as usize;
        let pages = listed
            .iter()
            .map(|&id| crate::store::kv::page_table::PublishedPage {
                id,
                token_hashes: vec![None; page_size],
                page_hash: None,
            })
            .collect();
        let mut freed = self.table.replace_tail(ws, f0, pages)?;
        freed.retain(
            |backing| !matches!(backing, KvPageBacking::Resident(id) if listed.contains(id)),
        );
        tracing::debug!(
            target: "runtime::store::kv",
            ?ws,
            f0,
            listed = ?listed.iter().map(|page| page.0).collect::<Vec<_>>(),
            ?freed,
            copies = ?copies.iter().map(|&(from, to)| (from.0, to.0)).collect::<Vec<_>>(),
            "a beam's finish relisted its set's tail"
        );
        // Copies made and not named (their page turned private meanwhile)
        // go back at once: no fire read them.
        let unused: Vec<PhysicalKvPageId> = copies
            .iter()
            .map(|&(_, copy)| copy)
            .filter(|copy| !listed.contains(copy))
            .collect();
        let named = copies.len() - unused.len();
        if named > 0 {
            RELIST_COPIES.fetch_add(named as u64, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(target: "pie::kv", copies = named, "a beam's finish copied the chosen pages another reads");
        }
        self.pool.release_reserved(unused);
        let epoch = self.current_epoch();
        self.recycle_backings(freed, epoch);
        self.refresh_flat(ws);
        Ok(true)
    }

    pub fn flat_table(&mut self, ws: WorkingSetId) -> Result<&[PhysicalKvPageId], KvStoreError> {
        if !self.flat.get(&ws).is_some_and(Option::is_some) {
            let flat = self.table.flatten(ws)?;
            tracing::debug!(
                target: "runtime::store::kv",
                ?ws,
                pages = ?flat.iter().map(|page| page.0).collect::<Vec<_>>(),
                "a set's flat table rebuilt"
            );
            self.flat.insert(ws, Some(flat));
        }
        Ok(self.flat[&ws].as_deref().expect("just populated"))
    }

    fn refresh_flat(&mut self, ws: WorkingSetId) {
        let flat = self.table.flatten(ws).ok();
        tracing::debug!(
            target: "runtime::store::kv",
            ?ws,
            pages = ?flat.as_ref().map(|flat| flat.iter().map(|page| page.0).collect::<Vec<_>>()),
            "a set's flat table refreshed"
        );
        self.flat.insert(ws, flat);
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

    /// Counting walks every cache root and every live reader under the lock
    /// every guest's fire waits on, and the planner, the governor's tick and
    /// the census each ask: a count stands a second, or fifty times what its
    /// walk took when that is longer.
    pub fn reclaimable_cache_pages(&self) -> u32 {
        if let Some((at, took, pages)) = self.reclaimable.get()
            && at.elapsed() < std::time::Duration::from_secs(1).max(took * 50)
        {
            return pages;
        }
        self.reclaimable_cache_pages_now()
    }

    /// [`Self::reclaimable_cache_pages`] counted now, however recent the
    /// last count.
    pub fn reclaimable_cache_pages_now(&self) -> u32 {
        let began = Instant::now();
        let pages = self.table.unused_cache_lease_pages();
        self.reclaimable.set(Some((began, began.elapsed(), pages)));
        pages
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

    /// Reserved device pages no fire has touched, given back and taken
    /// again at once: the fullest partly filled list chunks first, so the
    /// chunks only they held stand free whole, with no copy. As many come
    /// back as went. Only where that frees more chunks than the take opens
    /// again: else they stay as they are (taken again, they would free and
    /// fill the same room on every ask).
    pub fn repack_reserved(&mut self, pages: Vec<PhysicalKvPageId>) -> Vec<PhysicalKvPageId> {
        let n = pages.len();
        if n == 0 || self.repack_frees(&pages) == 0 {
            return pages;
        }
        self.pool.release_reserved(pages);
        self.pool
            .try_alloc_n(n)
            .expect("the slots just given back stand free")
    }

    /// The chunks [`Self::repack_reserved`] would free past those its take
    /// opens again: the chunks only `pages` hold, less the fresh chunks the
    /// pages need past the free slots of the chunks left.
    fn repack_frees(&self, pages: &[PhysicalKvPageId]) -> u32 {
        let Some((shared, shape)) = self.pool.shared_shape() else {
            return 0;
        };
        let (spc, chunks) = shared.lock().unwrap().single_chunks(shape);
        if spc == 0 {
            return 0;
        }
        let base = self.pool.base();
        let mut mine: HashMap<u32, u32> = HashMap::new();
        for page in pages {
            *mine.entry(page.0.saturating_sub(base) / spc).or_default() += 1;
        }
        let (mut emptied, mut room) = (0u32, 0u64);
        for chunk in &chunks {
            let held = mine.get(&chunk.logical).copied().unwrap_or(0);
            if held == chunk.used {
                emptied += 1;
            } else {
                room += u64::from(spc - chunk.used + held);
            }
        }
        let opened = (pages.len() as u64)
            .saturating_sub(room)
            .div_ceil(u64::from(spc));
        emptied.saturating_sub(u32::try_from(opened).unwrap_or(u32::MAX))
    }

    /// Host room is for what is paused before what is cached, as far as the
    /// retention keys let the cache go ([`Self::drop_entries_off_device`],
    /// [`Self::make_swap_room`]): past that the suspend is refused as
    /// [`KvStoreError::HostSwapFull`], and waits for room.
    pub fn prepare_suspend(
        &mut self,
        working_sets: &HashSet<WorkingSetId>,
        keys: impl Keys,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        let mut pages = self.table.movable_resident_pages(working_sets)?;
        if self.host_pool.available() < pages.len() {
            self.drop_entries_off_device(pages.len(), &keys, None);
            // Dropping can compact the pages just listed: list them again.
            pages = self.table.movable_resident_pages(working_sets)?;
        }
        if pages.is_empty() {
            return Ok(None);
        }
        let available = self.make_swap_room(pages.len(), &keys);
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
        order: &[(WorkingSetId, u64)],
        wanted: usize,
        keys: impl Keys,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        self.prepare_sweep_with(keep, room, order, wanted, true, &keys)
    }

    /// [`Self::prepare_sweep`] of the tails alone: the heads `order` names
    /// stay even when fewer pages than wanted move.
    pub fn prepare_tail_sweep(
        &mut self,
        keep: &HashSet<WorkingSetId>,
        room: usize,
        order: &[(WorkingSetId, u64)],
        wanted: usize,
        keys: impl Keys,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        self.prepare_sweep_with(keep, room, order, wanted, false, &keys)
    }

    fn prepare_sweep_with(
        &mut self,
        keep: &HashSet<WorkingSetId>,
        room: usize,
        order: &[(WorkingSetId, u64)],
        wanted: usize,
        spare_heads: bool,
        keys: &impl Keys,
    ) -> Result<Option<KvResidencyTxn>, KvStoreError> {
        // In tokens as a policy states them, in pages here.
        let page = u64::from(self.page_size.max(1));
        let order: Vec<_> = order
            .iter()
            .map(|&(ws, tokens)| (ws, tokens.div_ceil(page)))
            .collect();
        let order = &order[..];
        let mut pages = self.table.stranded_pages(keep, order, wanted, spare_heads);
        if self.swap_available(keys) < pages.len().min(room) {
            let wanted = pages
                .len()
                .min(room)
                .saturating_sub(self.disk_swap_free(keys));
            self.drop_entries_off_device(wanted, keys, None);
            // Dropping can compact the pages just listed: list them again.
            pages = self.table.stranded_pages(keep, order, wanted, spare_heads);
        }
        let available = self.make_swap_room(pages.len().min(room), keys);
        pages.truncate(room.min(available));
        if pages.is_empty() {
            return Ok(None);
        }
        // Only the readers of what moves: one whose pages all stayed may be
        // restored while this is out.
        let readers = self.table.suspended_readers(&pages);
        Ok(Some(self.prepare_move_out(&readers, pages)?))
    }

    fn swap_available(&self, keys: &impl Keys) -> usize {
        self.host_pool.available() + self.disk_swap_free(keys)
    }

    /// Frees swap slots for `wanted` pages, giving up the disk tier's copies
    /// if it must ([`Self::disk_yield_plan`]): the disk is the last tier, so
    /// what it holds has nowhere lower to go, and yields to running work
    /// whatever its key. Answers the slots free.
    fn make_swap_room(&mut self, wanted: usize, keys: &impl Keys) -> usize {
        let free = self.host_pool.available() + self.disk_pool.available();
        if free < wanted {
            self.yield_disk(wanted - free, keys);
        }
        let free = self.host_pool.available() + self.disk_pool.available();
        if free < wanted {
            // What idle cache below the device was counted to give is gone.
            self.below_cached.set(None);
            self.droppable_below.set(None);
        }
        free
    }

    /// Copies whose pages are held on the device or the host, lowest
    /// retention key first; and the unused entries kept only on disk: first
    /// those whose tiers row does not name the disk, then the rest, each at
    /// the lowest key among their pages, then least recently used. Keys and
    /// rows only order: everything here may go (the disk has no lower
    /// tier), and dropping an entry gives up all its copies.
    fn disk_yield_plan(&self, keys: &impl Keys) -> (Vec<DiskKvSlotId>, Vec<(Instant, u64)>) {
        let Some(persisted) = &self.persisted else {
            return (Vec::new(), self.unkept_on_disk(keys));
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
        // An entry goes at the lowest key among its pages' copies, and only
        // where the highest says it may.
        let copies = |entry: &KeptEntry| -> (i64, i64) {
            self.table
                .path_keys(entry.path.terminal)
                .into_iter()
                .flatten()
                .take(entry.path.pages as usize)
                .filter_map(|page| persisted.by_key.get(&page))
                .map(key)
                .fold(None, |range: Option<(i64, i64)>, key| {
                    Some(range.map_or((key, key), |(lo, hi)| (lo.min(key), hi.max(key))))
                })
                .unwrap_or((0, 0))
        };
        let mut whole: Vec<(bool, i64, Instant, u64)> = self
            .cache
            .iter()
            .filter(|(_, entry)| {
                let root = entry.path.terminal;
                unused.contains(&root)
                    && !self.table.holds_resident(root)
                    && !self.table.in_move_at(root)
                    && self.persisted_whole(entry)
            })
            .filter_map(|(&id, entry)| {
                let (lowest, _) = copies(entry);
                let kept = self.tiers_of(id, keys) & retention::DISK != 0;
                Some((kept, lowest, entry.last_use, id))
            })
            .collect();
        whole.sort_unstable();
        (
            spare,
            whole.into_iter().map(|(_, _, at, id)| (at, id)).collect(),
        )
    }

    /// The unused entries holding pages on a disk tier that keeps no copies,
    /// as [`Self::disk_yield_plan`] orders entries; one with pages on the
    /// device too goes after those with none.
    fn unkept_on_disk(&self, keys: &impl Keys) -> Vec<(Instant, u64)> {
        if self.disk_pool.capacity() == 0 || self.cache.is_empty() {
            return Vec::new();
        }
        let unused: HashSet<NodeId> = self.table.unused_cache_roots().into_iter().collect();
        let mut entries: Vec<(bool, bool, i64, Instant, u64)> = self
            .cache
            .iter()
            .filter(|(_, entry)| {
                let root = entry.path.terminal;
                unused.contains(&root)
                    && !self.table.in_move_at(root)
                    && self.table.holds_on_disk(root)
            })
            .map(|(&id, entry)| {
                let resident = self.table.holds_resident(entry.path.terminal);
                let kept = self.tiers_of(id, keys) & retention::DISK != 0;
                let key = keys.key_at(entry.owner, self.key_token(entry.path.pages));
                (resident, kept, key, entry.last_use, id)
            })
            .collect();
        entries.sort_unstable();
        entries.into_iter().map(|(.., at, id)| (at, id)).collect()
    }

    /// Gives up the disk tier's copies until `wanted` slots are free:
    /// [`Self::disk_yield_plan`] in order.
    fn yield_disk(&mut self, wanted: usize, keys: &impl Keys) {
        let (spare, whole) = self.disk_yield_plan(keys);
        let start = self.disk_pool.available();
        let freed = |store: &Self| store.disk_pool.available().saturating_sub(start);
        let spare: Vec<DiskKvSlotId> = spare.into_iter().take(wanted).collect();
        self.give_up_copies(spare);
        for (_, id) in whole {
            if freed(self) >= wanted {
                break;
            }
            self.drop_entries(vec![id], By::Host);
        }
        tracing::debug!(target: "pie::kv", wanted, freed = freed(self), "disk copies given up for swap");
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

    /// Drops idle cache held on the host until it has `wanted` slots free:
    /// the lowest retention key first, then the least recently used, so the
    /// host keeps the higher keys as the device does. Only what holds host
    /// slots: an entry held on disk alone frees none, and dropping it lost
    /// cache for no room. What its row lets on disk went there first, where
    /// the disk had room (`prepare_suspend_spill`, a demote's spill); the
    /// rest has no lower tier and is dropped. No lookup hits what goes, and
    /// no one will bring it back.
    fn drop_entries_off_device(
        &mut self,
        wanted: usize,
        keys: &impl Keys,
        below: Option<i64>,
    ) -> usize {
        let unused: HashSet<NodeId> = self.table.unused_cache_roots().into_iter().collect();
        let mut entries: Vec<(i64, Instant, u64)> = self
            .cache
            .iter()
            .filter(|(_, entry)| {
                unused.contains(&entry.path.terminal)
                    && !self.table.settled_on_device(entry.path.terminal)
                    && !self.persisted_whole(entry)
                    && self.table.holds_on_host(entry.path.terminal)
            })
            .map(|(&id, entry)| {
                let key = keys.key_at(entry.owner, self.key_token(entry.path.pages));
                (key, entry.last_use, id)
            })
            .filter(|&(key, _, _)| below.is_none_or(|below| key < below))
            .collect();
        entries.sort_unstable();
        let mut dropped = Vec::new();
        for (key, _, id) in entries {
            if self.host_pool.available() >= wanted {
                break;
            }
            self.drop_entries(vec![id], By::Host);
            dropped.push(key);
        }
        tracing::debug!(target: "pie::kv", entries = dropped.len(), keys = ?dropped, wanted, ?below, "dropped off device");
        dropped.len()
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
        self.prepare_move_out_split(working_sets, pages, on_host)
    }

    /// [`Self::prepare_move_out`] with the first `on_host` pages to the host
    /// and the rest to disk; both counted free by the caller.
    fn prepare_move_out_split(
        &mut self,
        working_sets: &HashSet<WorkingSetId>,
        pages: Vec<(TriePageLocation, PhysicalKvPageId)>,
        on_host: usize,
    ) -> Result<KvResidencyTxn, KvStoreError> {
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
            shape: self.shape,
            records: HashMap::new(),
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

    pub fn commit_suspend(&mut self, mut txn: KvResidencyTxn) -> Result<usize, KvStoreError> {
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
        let records = std::mem::take(&mut txn.records);
        if !records.is_empty() {
            let landed: Vec<_> = txn
                .pages
                .iter()
                .map(|page| (page.location, page.swapped))
                .collect();
            self.keep_demoted(landed, records);
        }
        let count = txn.pages.len();
        tracing::debug!(
            target: "runtime::store::kv",
            sets = ?txn.working_sets,
            moved = ?txn.pages.iter().map(|page| page.gpu_id.0).collect::<Vec<_>>(),
            left_resident = ?self.table.resident_ids(&txn.working_sets),
            "a suspend committed: its device pages go back, those left resident another reads"
        );
        // No fire still in flight can read these pages: the suspend waited
        // for every fire of the process to settle (its fence is up and its
        // leases quiesced), the pages are its own (a shared page does not
        // move), and the copy out of them, ordered behind every fire
        // submitted before it, has landed. They go back now, not behind the
        // newest fire's epoch: held there, they stood as deficit while
        // unrelated fires ran, and each suspend that landed so was followed
        // by as many more (Qwen3-8B c256: 841 suspends where main moves 338).
        let given: Vec<PhysicalKvPageId> = txn.pages.iter().map(|page| page.gpu_id).collect();
        self.check_given_back(&given, "a suspend's commit");
        self.pool.release_reserved(given);
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
            shape: self.shape,
            records: HashMap::new(),
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
        tracing::debug!(
            target: "runtime::store::kv",
            sets = ?txn.working_sets,
            placed = ?txn.pages.iter().map(|page| page.gpu_id.0).collect::<Vec<_>>(),
            resident = ?self.table.resident_ids(&txn.working_sets),
            "a restore committed"
        );
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

    /// What is free now.
    pub fn available_pages(&self) -> usize {
        self.free_committed()
    }

    pub fn free_committed(&self) -> usize {
        self.pool.available()
    }

    pub fn note_short(&mut self, pages: u32) {
        self.short_pages = self.short_pages.max(pages);
        self.pool.wait(u64::from(self.short_pages));
    }

    /// No waiter is short any more (a shared pool: a reservation went through).
    pub fn clear_short(&mut self) {
        if self.short_pages > 0 {
            self.short_pages = 0;
            self.pool.wait(0);
        }
    }

    /// Pages handed out: what is said of the pool to anyone deciding on its
    /// fullness comes from here.
    pub fn pages_out(&self) -> u32 {
        self.pool.out()
    }

    /// Pages out below the limit, and pages a waiter is short of on top. The
    /// tail is on its way out and asks for nothing.
    pub fn pages_wanted(&self) -> u32 {
        self.pool.out().saturating_add(self.short_pages)
    }

    pub fn capacity_pages(&self) -> u32 {
        self.pool.capacity()
    }

    /// The device pages' slot pool: what is held and wanted of it.
    pub fn page_pool(&self) -> &ShapePool<PhysicalKvPageId> {
        &self.pool
    }

    pub fn host_swap_available(&self) -> usize {
        self.host_pool.available()
    }

    pub fn host_swap_capacity(&self) -> u32 {
        self.host_pool.capacity()
    }

    /// Free disk slots, and those the disk tier's copies would give up for
    /// swap.
    /// The disk tier's free slots, and those copies the retention keys let
    /// go hold ([`Self::disk_yield_plan`]).
    pub fn disk_swap_available(&self, keys: impl Keys) -> usize {
        self.disk_swap_free(&keys)
    }

    fn disk_swap_free(&self, keys: &impl Keys) -> usize {
        // With no copies kept, what idle cache would give is counted apart
        // ([`Self::droppable_below_device`]).
        if self.persisted.is_none() {
            return self.disk_pool.available();
        }
        let (spare, whole) = self.disk_yield_plan(keys);
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

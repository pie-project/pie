pub mod working_set;
pub mod write;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::inferlet::process::ProcessId;
use crate::rt::Instant;

use crate::store::chunks::{ShapePool, StorePools};
use crate::store::genmap::{GenKey, GenMap};
use crate::store::kv::hash::Hash256;
use crate::store::pool::PoolId;
use crate::store::shared::Shared;
use crate::store::slots::{Demand, Holder, ShapeKind};
use write::{
    RsBufferTarget, RsPendingFold, RsPendingFolds, RsPreparedWrite, RsPublished, RsSnapshot,
    RsStateTarget,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsWsMarker {}
pub type RsWorkingSetId = GenKey<RsWsMarker>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RsSlotId(pub u32);

impl PoolId for RsSlotId {
    fn from_index(index: u32) -> Self {
        Self(index)
    }
    fn index(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsGeometry {
    pub state_size: u64,
    pub buffer_page_tokens: u32,
}

/// The most window lists a store holds, one a distinct window.
pub const MAX_WINDOW_LISTS: usize = 4;

/// One window list of a model whose kv is read through sliding windows
/// (CONTRACT-BC 19, 21): its rows retain `window` tokens, a slot holds
/// `tokens` (`T_w`) of them, slot `i` of a set's table rows
/// `[i * T_w, (i + 1) * T_w)`.
///
/// Window-list path: removed by the windowed-state change (the integration
/// commit deletes it once every engine states `Capabilities::windowed_state`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowGeometry {
    pub window: u32,
    pub tokens: u32,
}

/// A set's windows: `committed` is the position they are anchored at and
/// `head` the position written up to, in tokens; `retired[l]` counts the
/// leading slots of list `l`'s table already given back.
///
/// Window-list path: removed by the windowed-state change (the integration
/// commit deletes it once every engine states `Capabilities::windowed_state`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WindowState {
    retired: [u32; MAX_WINDOW_LISTS],
    committed: u32,
    head: u32,
}

/// What a fire moves a set's windows by once it is published: `fold` of the
/// tokens up to `commit_end` are committed, and the head lands at `head`.
/// With a device-decided length (no `fold`) the host commits nothing, but
/// every token before `settled`, the first the fire writes, is part of the
/// sequence it continues, so the windows catch up to it. `positioned` says
/// the host knows the fire's positions: each list's slots whose last row
/// lies below the committed position less its window retire. A
/// device-positioned fire retires behind `reach`, the first token it may
/// write, as no row's window reaches back past it.
///
/// Window-list path: removed by the windowed-state change (the integration
/// commit deletes it once every engine states `Capabilities::windowed_state`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowAdvance {
    pub(crate) positioned: bool,
    pub(crate) settled: u32,
    pub(crate) reach: u32,
    pub(crate) fold: Option<u32>,
    pub(crate) commit_end: u32,
    pub(crate) head: Option<u32>,
    /// A checkpoint the fire lands inside it, at this position: no slot its
    /// windows reach from there retires with this fire.
    pub(crate) keep: Option<u32>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RsError {
    #[error("unknown rs working set")]
    UnknownWorkingSet,
    #[error("a beam search's finish: {0}")]
    BeamFinish(&'static str),
    #[error("a fire writes slot {slot}, which its state already holds (it holds {frozen})")]
    BelowFrozen { slot: u32, frozen: u32 },
    #[error("rs batch contains the same working set more than once")]
    DuplicateWorkingSet,
    #[error("rs pools exhausted: requested {requested}, available {available}")]
    OutOfSlots {
        requested: RsUnits,
        available: RsUnits,
    },
    #[error("rs grant mismatch: required {required}, granted {granted}")]
    GrantMismatch { required: RsUnits, granted: RsUnits },
    #[error(
        "a fire on its way folds this working set's recurrent state by a count the device \
         computes; it is not forked, checkpointed, exported or moved until that fire settles"
    )]
    DeviceFoldUnsettled,
    #[error(
        "a fire that folded this working set's recurrent state by a count the device computes \
         did not settle ({0}): how far its record is folded is not known"
    )]
    DeviceFoldFailed(String),
    #[error(
        "its state holds the rows below {committed}, past the cut at {cut}: a fold is not taken \
         back"
    )]
    FoldedPastCut { committed: u32, cut: u32 },
    #[error(
        "its state buffers no row (a short conv state, which no buffered plane carries): every \
         row folds as it is written, and a fold is not taken back"
    )]
    Unbuffered,
}

pub const RS_TRANSLATION_UNMAPPED: u32 = u32::MAX;

/// A count in each recurrent pool: state slots and buffer pages, or, of a
/// windowed store, its window lists' slots: `pages` their sum and
/// `windows[l]` list `l`'s (each its own shape's slots). In the degenerate
/// mode a slot is a chunk, so this is the chunks the shapes ask for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RsUnits {
    pub slots: u32,
    pub pages: u32,
    pub windows: [u32; MAX_WINDOW_LISTS],
    /// Of `pages`, the window buffer's slots, where a hybrid's window
    /// buffer is a pool beside its rs list (CONTRACT-BC 19, 21): the rest
    /// of `pages` is the rs list's. 0 elsewhere (a store of windowed layers
    /// alone keeps its window buffer as its pages).
    pub wbuf: u32,
}

impl std::fmt::Display for RsUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} state slot(s) and {} buffer page(s)",
            self.slots, self.pages
        )?;
        if self.windows.iter().any(|&n| n > 0) {
            write!(f, " (window lists {:?})", self.windows)?;
        }
        if self.wbuf > 0 {
            write!(f, " ({} of them window-buffer slots)", self.wbuf)?;
        }
        Ok(())
    }
}

impl RsUnits {
    /// `slots` state slots and `pages` buffer pages, no window slot.
    #[must_use]
    pub const fn of(slots: u32, pages: u32) -> RsUnits {
        RsUnits {
            slots,
            pages,
            windows: [0; MAX_WINDOW_LISTS],
            wbuf: 0,
        }
    }

    /// Window list slots, `windows[l]` of list `l`; `pages` their sum.
    #[must_use]
    pub fn of_windows(windows: [u32; MAX_WINDOW_LISTS]) -> RsUnits {
        RsUnits {
            slots: 0,
            pages: windows.iter().fold(0u32, |sum, &n| sum.saturating_add(n)),
            windows,
            wbuf: 0,
        }
    }

    #[must_use]
    pub fn is_zero(self) -> bool {
        self.slots == 0 && self.pages == 0 && self.windows.iter().all(|&n| n == 0)
    }

    #[must_use]
    pub fn covers(self, need: RsUnits) -> bool {
        self.slots >= need.slots
            && self.pages >= need.pages
            && self.wbuf >= need.wbuf
            && self.pages.saturating_sub(self.wbuf) >= need.pages.saturating_sub(need.wbuf)
            && self
                .windows
                .iter()
                .zip(need.windows)
                .all(|(&have, need)| have >= need)
    }

    #[must_use]
    pub fn add(self, other: RsUnits) -> RsUnits {
        let mut windows = self.windows;
        for (have, more) in windows.iter_mut().zip(other.windows) {
            *have = have.saturating_add(more);
        }
        RsUnits {
            slots: self.slots.saturating_add(other.slots),
            pages: self.pages.saturating_add(other.pages),
            windows,
            wbuf: self.wbuf.saturating_add(other.wbuf),
        }
    }

    #[must_use]
    pub fn sub(self, other: RsUnits) -> RsUnits {
        let mut windows = self.windows;
        for (have, less) in windows.iter_mut().zip(other.windows) {
            *have = have.saturating_sub(less);
        }
        RsUnits {
            slots: self.slots.saturating_sub(other.slots),
            pages: self.pages.saturating_sub(other.pages),
            windows,
            wbuf: self.wbuf.saturating_sub(other.wbuf),
        }
    }
}

/// Reserved ids: state slots, buffer pages, and each window list's slots.
#[derive(Debug, Default)]
pub struct RsGrant {
    pub slots: Vec<RsSlotId>,
    pub pages: Vec<RsSlotId>,
    pub windows: Vec<Vec<RsSlotId>>,
    /// A hybrid's window-buffer slots ([`RsUnits::wbuf`]).
    pub wpages: Vec<RsSlotId>,
}

impl RsGrant {
    #[must_use]
    pub fn units(&self) -> RsUnits {
        let mut windows = [0; MAX_WINDOW_LISTS];
        for (count, list) in windows.iter_mut().zip(&self.windows) {
            *count = list.len() as u32;
        }
        let window_slots: usize = self.windows.iter().map(Vec::len).sum();
        RsUnits {
            slots: self.slots.len() as u32,
            pages: (self.pages.len() + self.wpages.len() + window_slots) as u32,
            windows,
            wbuf: self.wpages.len() as u32,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
            && self.pages.is_empty()
            && self.wpages.is_empty()
            && self.windows.iter().all(Vec::is_empty)
    }
}

/// A move between a shared pool and its host tier, stated by slot shape
/// (CONTRACT-BC 22): for each shape, the `(device, host)` slot refs it
/// moves, a record's every piece of every fixed shape.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RsCopyPlan {
    pub shared: Vec<(::engine::chunk::ShapeId, Vec<(u32, u32)>)>,
    /// The same, `(device, disk)`: slot refs of the disk tier's slot file,
    /// a disk chunk laid out as a device chunk of the shape, where no host
    /// tier holds the state ([`RsStore::set_disk`]).
    pub disk: Vec<(::engine::chunk::ShapeId, Vec<(u32, u32)>)>,
}

/// The first id of a disk row ([`RsStore::set_disk`]): past every host
/// pool's ids.
const DISK_ROWS: u32 = 1 << 30;

/// The disk tier a shared pool's state parks on where no host tier holds
/// it (a load on unified memory states none, so an evicted process's
/// records and rs list slots kept their device chunks): whole chunks of the
/// slot file, each laid out as a device chunk of one shape (CONTRACT-BC 22),
/// counted against the disk's budget with kv's.
struct RsDisk {
    slots: std::sync::Arc<dyn crate::store::roles::DiskSlots>,
    /// The slots of each engine shape a disk chunk holds, as a device chunk.
    per_chunk: HashMap<::engine::chunk::ShapeId, u32>,
    /// Each shape's disk chunks and the slots free in each.
    chunks: HashMap<::engine::chunk::ShapeId, BTreeMap<u32, BTreeSet<u32>>>,
    rows: HashMap<RsSlotId, DiskRow>,
    free_ids: Vec<u32>,
    next: u32,
    /// A record's shapes and its pieces of each; the pages' shape (the rs
    /// list's, or window list 0's); window lists 1..'s.
    records: Vec<(::engine::chunk::ShapeId, u32)>,
    lists: Vec<::engine::chunk::ShapeId>,
    /// A hybrid's window buffer's shape.
    wshape: Option<::engine::chunk::ShapeId>,
}

/// A device slot parked on disk: what it was (a window list's slot, a
/// page, a record) and its disk refs by shape.
struct DiskRow {
    list: Option<usize>,
    page: bool,
    /// A hybrid's window-buffer slot.
    wpage: bool,
    refs: Vec<(::engine::chunk::ShapeId, Vec<u32>)>,
}

impl RsDisk {
    /// `n` free slots of `shape` in its disk chunks, opening whole chunks
    /// for what they lack; all or none.
    fn take(&mut self, shape: ::engine::chunk::ShapeId, n: usize) -> Option<Vec<u32>> {
        let per_chunk = *self.per_chunk.get(&shape)?;
        let chunks = self.chunks.entry(shape).or_default();
        let free: usize = chunks.values().map(BTreeSet::len).sum();
        if free < n {
            let opened = (n - free).div_ceil(per_chunk as usize);
            for chunk in self.slots.take_chunks(opened)? {
                chunks.insert(chunk, (0..per_chunk).collect());
            }
        }
        let mut refs = Vec::with_capacity(n);
        for (&chunk, slots) in chunks.iter_mut() {
            while refs.len() < n {
                let Some(index) = slots.pop_first() else {
                    break;
                };
                refs.push(chunk * per_chunk + index);
            }
            if refs.len() == n {
                break;
            }
        }
        Some(refs)
    }

    /// `refs` of `shape` free again; a chunk wholly free goes back to the
    /// slot file.
    fn give(&mut self, shape: ::engine::chunk::ShapeId, refs: &[u32]) {
        let Some(&per_chunk) = self.per_chunk.get(&shape) else {
            return;
        };
        let chunks = self.chunks.entry(shape).or_default();
        let mut emptied = Vec::new();
        for &slot in refs {
            let (chunk, index) = (slot / per_chunk, slot % per_chunk);
            if let Some(free) = chunks.get_mut(&chunk) {
                free.insert(index);
                if free.len() == per_chunk as usize {
                    emptied.push(chunk);
                }
            }
        }
        emptied.sort_unstable();
        emptied.dedup();
        for chunk in &emptied {
            chunks.remove(chunk);
        }
        if !emptied.is_empty() {
            self.slots.give_chunks(&emptied);
        }
    }
}

/// What a set holds, in its own numbering: see [`RsEntry`].
#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct RsExport {
    pub folded: Option<RsSlotId>,
    pub buffer: Vec<Option<RsSlotId>>,
    pub base: u32,
    pub frozen: u32,
    /// Each window list's table, when the state is windows.
    pub windows: Vec<Vec<Option<RsSlotId>>>,
    /// The windows' position, when the state is windows.
    pub window: Option<WindowState>,
    /// A hybrid's window-buffer table and its base.
    pub wbuffer: Vec<Option<RsSlotId>>,
    pub wbase: u32,
}

/// Token `t` of page `p` holds slot `base + p * page + t`. A windowed set
/// holds no buffer: `windows[l]` is window list `l`'s table, slot `i` its
/// rows `[i * T_w, (i + 1) * T_w)`, `None` where trimmed or not written.
#[derive(Clone)]
struct RsEntry {
    geom: RsGeometry,
    folded: Option<RsSlotId>,
    buffer: Vec<Option<RsSlotId>>,
    base: u32,
    frozen: u32,
    windows: Vec<Vec<Option<RsSlotId>>>,
    window: Option<WindowState>,
    /// A hybrid's window-buffer table and the row its entry 0 holds
    /// (CONTRACT-BC 19, 21), beside the rs list's `buffer`; empty elsewhere.
    wbuffer: Vec<Option<RsSlotId>>,
    wbase: u32,
    /// Fires on their way that fold the state by a count the device
    /// computes ([`RsStore::begin_device_fold`]). Until the last settles the
    /// engine's record stands ahead of `frozen` by slots only the engine
    /// counts, and only where the record stands now: the set is not forked,
    /// checkpointed, exported or moved.
    device_folds: u32,
    /// Why such a fire did not settle its fold: `frozen` no longer says
    /// where the record stands, until the state is forgotten. The set is
    /// refused a fork, a checkpoint and an export by it, and its record
    /// stays where the engine holds it.
    device_fold_failed: Option<String>,
}

#[derive(Debug)]
pub struct RsResidencyTxn {
    scope: Vec<RsWorkingSetId>,
    moves: Vec<(RsSlotId, RsSlotId)>,
    plan: RsCopyPlan,
}

impl RsResidencyTxn {
    pub fn copy_plan(&self) -> &RsCopyPlan {
        &self.plan
    }
}

#[derive(Debug, Default)]
struct Touch {
    writes: Vec<RsSlotId>,
    reads: Vec<RsSlotId>,
}

/// One numbering covers four pools: device state slots from 0, then device
/// buffer pages, then host slots and pages. An engine numbers each kind from 0
/// ([`RsStore::engine_id`]).
pub struct RsStore {
    slots: ShapePool<RsSlotId>,
    pages: ShapePool<RsSlotId>,
    host_slots: ShapePool<RsSlotId>,
    host_pages: ShapePool<RsSlotId>,
    /// Host slots of window lists 1.. (list 0 parks in `host_pages`), each
    /// numbered past the one before.
    host_windows: Vec<ShapePool<RsSlotId>>,
    pages_at: u32,
    /// The window lists, where the state is windows (no state slots, id 0 of
    /// each the engine's null page): list 0's slots are `pages`, list
    /// `l > 0`'s `window_pools[l - 1]`, each numbered past the one before.
    window: Vec<WindowGeometry>,
    window_pools: Vec<ShapePool<RsSlotId>>,
    refs: HashMap<RsSlotId, u32>,
    working_sets: GenMap<RsWsMarker, RsEntry>,
    seq: u64,
    /// Unsettled fires by seq, with the slots each writes and reads.
    inflight: BTreeMap<u64, Touch>,
    /// By slot number, the newest seq whose fire touched the slot.
    last_touch: Vec<u64>,
    /// By slot number, how many times unsettled fires' touches name it.
    pending_touch: Vec<u32>,
    /// Checkpoints: hidden forks no process holds.
    checkpoints: HashMap<Hash256, RsCheckpoint>,
    /// Slots a suspend or restore has between its prepare and commit.
    moving: HashSet<RsSlotId>,
    /// Told whenever a move lets slots go: what waits for a set to stop
    /// moving waits on it.
    moved: std::sync::Arc<tokio::sync::Notify>,
    /// Told whenever a fold the device counted settles, however its fire
    /// ended: what waits for a set's frozen length to be the record's waits
    /// on it.
    device_folded: std::sync::Arc<tokio::sync::Notify>,
    /// The tokens an rs list slot holds, where the buffer pages are a
    /// shared pool's rs list (CONTRACT-BC 18): a page holds that many rows,
    /// whatever a set's geometry says, and the buffer starts at a multiple
    /// of it.
    rs_tokens: Option<u32>,
    /// The disk tier state parks on where no host row is free
    /// ([`Self::set_disk`]).
    disk: Option<RsDisk>,
    /// The buffer pages are the window buffer's slots (CONTRACT-BC 19,
    /// 21): an empty buffer starts where a fire's rows first stay unfolded
    /// ([`Self::own_folds_unbuffered`]), not where the set is folded.
    window_buffer: bool,
    /// A hybrid's window buffer (an rs list beside it, CONTRACT-BC 19, 21):
    /// its device slots numbered past the pages, its host rows past the host
    /// pages', and its slot's rows. Empty pools and `None` elsewhere.
    wpages: ShapePool<RsSlotId>,
    host_wpages: ShapePool<RsSlotId>,
    w_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct RsCheckpoint {
    snapshot: RsWorkingSetId,
    owner: Option<ProcessId>,
    last_use: Instant,
}

impl RsStore {
    #[cfg(test)]
    pub fn new(slots: u32, pages: u32) -> Self {
        Self::fixed(0, slots, 0, pages)
    }

    /// The fixed shapes' pools: state slots of `slot_bytes` and buffer
    /// pages of `page_bytes`, on the device alone.
    fn fixed(slot_bytes: u64, slots: u32, page_bytes: u64, pages: u32) -> Self {
        let host = slots + pages;
        Self::build(
            ShapePool::new(ShapeKind::Fixed, slot_bytes, 0, slots),
            ShapePool::new(ShapeKind::Fixed, page_bytes, slots, pages),
            ShapePool::new(ShapeKind::Fixed, slot_bytes, host, 0),
            ShapePool::new(ShapeKind::Fixed, page_bytes, host, 0),
            slots,
            Vec::new(),
            Vec::new(),
        )
    }

    /// The store pinned pools state: the state slots and buffer pages, on
    /// the device alone (the one engine on pinned pools parks no state).
    pub fn of(pools: &StorePools) -> Self {
        Self::fixed(
            pools.state.slot_bytes,
            pools.state.device,
            pools.buffer.slot_bytes,
            pools.buffer.device,
        )
    }

    /// The store an engine's shared pool backs: where there are window lists,
    /// each list's slot refs (ref 0, in the null chunk, is never handed
    /// out), list 0's as the pages and each other's numbered past the one
    /// before, every list trimmed by its own window (CONTRACT-BC 19, 21);
    /// else the pool's fixed records as state slots
    /// and the rs list's slot refs as buffer pages (ids from past the
    /// records', an engine id the slot ref; CONTRACT-BC 18). A pool with no
    /// rs list buffers nothing.
    ///
    /// A load whose engine serves windowed layers as ring + buffer
    /// (CONTRACT-BC 19, 21) is a store of the second kind: each windowed
    /// layer's ring is record bytes, so a record is the set's folded window,
    /// and the window buffer's slot refs are the buffer pages, as the rs
    /// list's are. Its sets fold, fork, share, park and snapshot as an rs
    /// set does; the lanes name the pages in `Lane::tables` and the fold in
    /// `Lane::window` ([`crate::pipeline::fire::rs::apply_shared`]).
    ///
    /// The window-list store (the first kind, `windows` non-empty) is
    /// removed by the windowed-state change: the integration commit deletes
    /// it, once every engine states `Capabilities::windowed_state`.
    ///
    /// `host`, the engine's host tier (CONTRACT-BC 22), backs the host rows
    /// a suspend parks state in: records, rs list slots and every window
    /// list's slots, each of its own shape, past the device's ids.
    pub fn shared(shared: &Shared, host: Option<&Shared>) -> Self {
        let (windows, fixed, rs_list, hybrid) = {
            let pool = shared.lock().unwrap();
            let hybrid = pool.rs_list().and(pool.window_buffer());
            (
                pool.windows().to_vec(),
                !pool.fixed().is_empty(),
                pool.buffer_list(),
                hybrid,
            )
        };
        if let Some((first, rest)) = windows.split_first() {
            let pages = ShapePool::shared(shared, *first);
            let mut end = pages.ids_end();
            let more: Vec<ShapePool<RsSlotId>> = rest
                .iter()
                .map(|list| {
                    let pool = ShapePool::shared_from(shared, *list, end);
                    end = pool.ids_end();
                    pool
                })
                .collect();
            let base = end.max(1);
            let mut host_lists: Vec<ShapePool<RsSlotId>> = windows
                .iter()
                .scan(base, |at, list| {
                    let pool = host.map_or_else(
                        || ShapePool::new(ShapeKind::List, list.slot_bytes, *at, 0),
                        |host| ShapePool::shared_from(host, *list, *at),
                    );
                    *at = pool.ids_end();
                    Some(pool)
                })
                .collect();
            let host_pages = host_lists.remove(0);
            let mut store = Self::build(
                ShapePool::new(ShapeKind::Fixed, 0, 0, 0),
                pages,
                ShapePool::new(ShapeKind::Fixed, 0, base, 0),
                host_pages,
                0,
                windows
                    .iter()
                    .map(|list| WindowGeometry {
                        window: list.window,
                        tokens: list.tokens,
                    })
                    .collect(),
                more,
            );
            store.host_windows = host_lists;
            return store;
        }
        let slots = if fixed {
            ShapePool::records(shared, 0)
        } else {
            ShapePool::new(ShapeKind::Fixed, 0, 0, 0)
        };
        let records = slots.capacity();
        let pages = rs_list.map_or_else(
            || ShapePool::new(ShapeKind::List, 0, records, 0),
            |list| ShapePool::shared_from(shared, list, records),
        );
        // A hybrid's window buffer: device slots past the pages, host rows
        // past the host pages'.
        let wpages = hybrid.map_or_else(
            || ShapePool::new(ShapeKind::List, 0, pages.ids_end(), 0),
            |list| ShapePool::shared_from(shared, list, pages.ids_end()),
        );
        let base = wpages.ids_end();
        let host_slots = match host {
            Some(host) if fixed => ShapePool::records(host, base),
            _ => ShapePool::new(ShapeKind::Fixed, 0, base, 0),
        };
        let at = host_slots.ids_end();
        let host_pages = match (host, rs_list) {
            (Some(host), Some(list)) => ShapePool::shared_from(host, list, at),
            _ => ShapePool::new(ShapeKind::List, 0, at, 0),
        };
        let host_wpages = match (host, hybrid) {
            (Some(host), Some(list)) => ShapePool::shared_from(host, list, host_pages.ids_end()),
            _ => ShapePool::new(ShapeKind::List, 0, host_pages.ids_end(), 0),
        };
        let mut store = Self::build(
            slots,
            pages,
            host_slots,
            host_pages,
            records,
            Vec::new(),
            Vec::new(),
        );
        store.rs_tokens = rs_list.map(|list| list.tokens.max(1));
        store.window_buffer = shared.lock().unwrap().window_buffer().is_some();
        store.wpages = wpages;
        store.host_wpages = host_wpages;
        store.w_tokens = hybrid.map(|list| list.tokens.max(1));
        store
    }

    /// The rows of `writes` a fire on `ws` buffers, `(rs, window)`: the
    /// rs list's and the window buffer's (CONTRACT-BC 19, 21). A store of
    /// recurrent layers alone buffers every row in its rs list; one of
    /// windowed layers alone keeps its window buffer as its pages (the
    /// first of the pair); a hybrid buffers every row in its rs list and
    /// the window rows in its window buffer (the second).
    ///
    /// The window buffer holds no row a lone lane folds itself: a lane
    /// carrying no row earlier fires buffered there (that buffer empty, its
    /// rows from where the set is folded) takes no slot for the rows it
    /// folds by a count the host knows, their K/V living in the fire; only
    /// the rows past them are buffered, the window buffer then starting at
    /// the folded index after the fire, rounded down to a slot. Otherwise
    /// (a carrying lane, a group, a device-counted fold) every row is
    /// buffered, from the folded index before the fire.
    pub fn split_writes(
        &self,
        ws: RsWorkingSetId,
        alone: bool,
        fold: Option<u32>,
        writes: Option<std::ops::Range<u32>>,
    ) -> Result<(Option<std::ops::Range<u32>>, Option<std::ops::Range<u32>>), RsError> {
        let entry = self.entry(ws)?;
        if !self.window_buffer {
            return Ok((writes, None));
        }
        let table = if self.hybrid() {
            &entry.wbuffer
        } else {
            &entry.buffer
        };
        let window = match (writes.clone(), fold) {
            (Some(writes), Some(fold))
                if alone
                    && table.iter().all(Option::is_none)
                    && writes.start == entry.frozen =>
            {
                let start = writes.start.saturating_add(fold).min(writes.end);
                (start < writes.end).then_some(start..writes.end)
            }
            _ => writes.clone(),
        };
        Ok(if self.hybrid() {
            (writes, window)
        } else {
            (window, None)
        })
    }

    /// A hybrid's window buffer is a pool beside its rs list.
    fn hybrid(&self) -> bool {
        self.w_tokens.is_some()
    }

    /// [`Self::hybrid`], for the fire's lowering: its window buffer's slots
    /// are not the pages.
    #[must_use]
    pub fn window_buffer_apart(&self) -> bool {
        self.hybrid()
    }

    /// Whether `id` is a hybrid's window-buffer slot or row.
    fn is_wpage(&self, id: RsSlotId) -> bool {
        if let Some(row) = self.disk_row(id) {
            return row.wpage;
        }
        self.hybrid() && (self.wpages.owns(id) || self.host_wpages.contains(id))
    }

    /// Whether `id` holds windowed layers' rows: a hybrid's window-buffer
    /// slot, or a page of a store of windowed layers alone.
    #[must_use]
    pub fn is_window_page(&self, id: RsSlotId) -> bool {
        self.window_buffer && if self.hybrid() { self.is_wpage(id) } else { self.is_page(id) }
    }

    /// The id of the hybrid's window-buffer slot the engine numbers `slot`.
    #[must_use]
    pub fn wpage_id(&self, slot: u32) -> RsSlotId {
        RsSlotId(slot + self.wpages.base())
    }

    /// The window buffer's table of `ws` (engine numbering) and the row its
    /// entry 0 holds, where the store buffers windowed layers' rows: a
    /// hybrid's window buffer, or the pages of a store of windowed layers
    /// alone. `None` elsewhere.
    pub fn window_table(&self, ws: RsWorkingSetId) -> Result<Option<(u32, Vec<u32>)>, RsError> {
        if !self.window_buffer {
            return Ok(None);
        }
        if !self.hybrid() {
            return Ok(Some((self.base(ws)?, self.buffer_translation(ws)?)));
        }
        let entry = self.entry(ws)?;
        let table = entry
            .wbuffer
            .iter()
            .map(|slot| slot.map_or(RS_TRANSLATION_UNMAPPED, |id| self.engine_id(id)))
            .collect();
        Ok(Some((self.wbuffer_base(entry), table)))
    }

    /// Whether the state slots are a shared pool's records.
    pub fn is_shared(&self) -> bool {
        self.slots.is_shared() || self.pages.is_shared()
    }

    fn build(
        slots: ShapePool<RsSlotId>,
        pages: ShapePool<RsSlotId>,
        host_slots: ShapePool<RsSlotId>,
        host_pages: ShapePool<RsSlotId>,
        pages_at: u32,
        window: Vec<WindowGeometry>,
        window_pools: Vec<ShapePool<RsSlotId>>,
    ) -> Self {
        Self {
            slots,
            pages,
            host_slots,
            host_pages,
            host_windows: Vec::new(),
            pages_at,
            window,
            window_pools,
            refs: HashMap::new(),
            working_sets: GenMap::new(),
            seq: 0,
            inflight: BTreeMap::new(),
            last_touch: Vec::new(),
            pending_touch: Vec::new(),
            checkpoints: HashMap::new(),
            moving: HashSet::new(),
            moved: std::sync::Arc::new(tokio::sync::Notify::new()),
            device_folded: std::sync::Arc::new(tokio::sync::Notify::new()),
            rs_tokens: None,
            disk: None,
            window_buffer: false,
            wpages: ShapePool::new(ShapeKind::List, 0, 0, 0),
            host_wpages: ShapePool::new(ShapeKind::List, 0, 0, 0),
            w_tokens: None,
        }
    }

    /// Parks on `disk` too, the disk tier's slot file, where no host row is
    /// free (no host tier at all on unified memory): each record's pieces
    /// and each rs list or window slot in disk chunks of its own shape, as
    /// the engine lays them out (CONTRACT-BC 22). Without it an evicted
    /// process's state stayed on the device, its chunks held.
    pub fn set_disk(&mut self, shared: &Shared, disk: std::sync::Arc<dyn crate::store::roles::DiskSlots>) {
        if !self.is_shared() && self.window.is_empty() {
            return;
        }
        let (per_chunk, records, lists, wshape) = {
            let pool = shared.lock().unwrap();
            let wshape = pool
                .rs_list()
                .and(pool.window_buffer())
                .map(|list| list.id);
            let records: Vec<(::engine::chunk::ShapeId, u32)> =
                pool.fixed().iter().map(|shape| (shape.id, shape.pieces)).collect();
            let lists: Vec<::engine::chunk::ShapeId> = pool
                .buffer_list()
                .map(|list| list.id)
                .into_iter()
                .chain(pool.windows().iter().map(|list| list.id))
                .collect();
            let engines: Vec<::engine::chunk::ShapeId> = pool
                .fixed()
                .iter()
                .map(|shape| shape.id)
                .chain(pool.buffer_list().map(|list| list.id))
                .chain(wshape)
                .chain(pool.windows().iter().map(|list| list.id))
                .collect();
            let per_chunk = engines
                .into_iter()
                .filter_map(|engine| {
                    let shape = pool.pool_shape(engine)?;
                    let per_chunk = pool.pool().shape(shape)?.slots_per_chunk;
                    (per_chunk > 0).then_some((engine, per_chunk))
                })
                .collect();
            (per_chunk, records, lists, wshape)
        };
        debug_assert!(
            std::iter::once(&self.host_slots)
                .chain([&self.host_pages, &self.host_wpages])
                .chain(&self.host_windows)
                .all(|pool| pool.ids_end() < DISK_ROWS),
            "host ids reach the disk rows'"
        );
        self.disk = Some(RsDisk {
            slots: disk,
            per_chunk,
            chunks: HashMap::new(),
            rows: HashMap::new(),
            free_ids: Vec::new(),
            next: 0,
            records,
            lists,
            wshape,
        });
    }

    /// The disk tier's slot file, where state parks on it: what its free
    /// room is read from, outside this store's lock.
    #[must_use]
    pub fn disk_slots(&self) -> Option<std::sync::Arc<dyn crate::store::roles::DiskSlots>> {
        self.disk.as_ref().map(|disk| disk.slots.clone())
    }

    /// The rows the disk tier takes, `free_chunks` of its chunks free: for
    /// each shape, the free chunks at that shape's slots a chunk (a disk
    /// chunk holds any shape, CONTRACT-BC 22) and the free slots of its
    /// partly used disk chunks; records by their pieces of each shape. What
    /// a victim's state finds room in below the device where no host row is
    /// free.
    #[must_use]
    pub fn disk_units(&self, free_chunks: u32) -> RsUnits {
        let Some(disk) = self.disk.as_ref() else {
            return RsUnits::default();
        };
        let share = u64::from(free_chunks);
        let room = |shape: ::engine::chunk::ShapeId| -> u64 {
            let open: u64 = disk
                .chunks
                .get(&shape)
                .map_or(0, |chunks| chunks.values().map(|free| free.len() as u64).sum());
            open + share * u64::from(disk.per_chunk.get(&shape).copied().unwrap_or(0))
        };
        let clamp = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        let slots = disk
            .records
            .iter()
            .map(|&(shape, pieces)| room(shape) / u64::from(pieces.max(1)))
            .min()
            .map_or(0, clamp);
        if self.is_window() {
            let mut windows = [0; MAX_WINDOW_LISTS];
            for (out, &shape) in windows.iter_mut().zip(&disk.lists) {
                *out = clamp(room(shape));
            }
            return RsUnits::of_windows(windows);
        }
        let pages = disk.lists.first().map_or(0, |&shape| clamp(room(shape)));
        let wbuf = disk.wshape.map_or(0, |shape| clamp(room(shape)));
        RsUnits {
            wbuf,
            ..RsUnits::of(slots, pages.saturating_add(wbuf))
        }
    }

    /// Whether `id` is a disk row.
    fn is_disk(&self, id: RsSlotId) -> bool {
        id.0 >= DISK_ROWS
    }

    /// Whether `id` is a disk row: its state lies in the disk tier.
    #[must_use]
    pub fn on_disk(&self, id: RsSlotId) -> bool {
        self.is_disk(id)
    }

    /// Disk rows for each of `ids` (device slots), in order; none unless
    /// all fit.
    fn take_disk_rows(&mut self, ids: &[RsSlotId]) -> Option<Vec<RsSlotId>> {
        self.disk.as_ref()?;
        type Row = (Option<usize>, bool, bool, Vec<(::engine::chunk::ShapeId, Vec<u32>)>);
        let rows: Vec<Row> = ids
            .iter()
            .map(|&id| {
                (
                    self.list_of(id),
                    self.is_page(id),
                    self.is_wpage(id),
                    self.pool_of(id).slot_refs(id),
                )
            })
            .collect();
        let disk = self.disk.as_mut()?;
        let mut taken: Vec<RsSlotId> = Vec::with_capacity(ids.len());
        let mut failed = false;
        'rows: for (list, page, wpage, device) in rows {
            let mut refs = Vec::with_capacity(device.len());
            for (shape, slots) in &device {
                match disk.take(*shape, slots.len()) {
                    Some(at) => refs.push((*shape, at)),
                    None => {
                        for (shape, at) in &refs {
                            disk.give(*shape, at);
                        }
                        failed = true;
                        break 'rows;
                    }
                }
            }
            let index = disk.free_ids.pop().unwrap_or_else(|| {
                disk.next += 1;
                disk.next - 1
            });
            let id = RsSlotId(DISK_ROWS + index);
            disk.rows.insert(
                id,
                DiskRow {
                    list,
                    page,
                    wpage,
                    refs,
                },
            );
            taken.push(id);
        }
        if failed {
            self.give_disk_rows(taken);
            return None;
        }
        Some(taken)
    }

    /// Disk rows free again, their slots back in their disk chunks.
    fn give_disk_rows(&mut self, ids: impl IntoIterator<Item = RsSlotId>) {
        let Some(disk) = self.disk.as_mut() else {
            return;
        };
        for id in ids {
            if let Some(row) = disk.rows.remove(&id) {
                for (shape, refs) in &row.refs {
                    disk.give(*shape, refs);
                }
                disk.free_ids.push(id.0 - DISK_ROWS);
            }
        }
    }

    /// The disk rows parked state holds.
    #[must_use]
    #[cfg(test)]
    pub fn disk_rows(&self) -> usize {
        self.disk.as_ref().map_or(0, |disk| disk.rows.len())
    }

    /// The refs of an off-device row (a host row, or a disk row), by shape.
    fn row_refs(&self, id: RsSlotId) -> Vec<(::engine::chunk::ShapeId, Vec<u32>)> {
        if self.is_disk(id) {
            return self
                .disk
                .as_ref()
                .and_then(|disk| disk.rows.get(&id))
                .map(|row| row.refs.clone())
                .unwrap_or_default();
        }
        self.host_pool_of(id).slot_refs(id)
    }

    /// The disk row `id`, where it is one.
    fn disk_row(&self, id: RsSlotId) -> Option<&DiskRow> {
        if !self.is_disk(id) {
            return None;
        }
        self.disk.as_ref()?.rows.get(&id)
    }

    /// In a shared pool, records that waiters are short `need` of each of
    /// this store's shapes (window lists each their own), so the pool's
    /// segment walker commits for them; `RsUnits::default()` clears it.
    pub fn note_short_shared(&mut self, need: RsUnits) {
        self.slots.wait(u64::from(need.slots));
        if self.is_window() {
            // List 0 is the pages: an ask states it in `windows[0]`, and a
            // seating reserve (`ResidencyPlanner`'s, growth in pages) in
            // `pages`; the walker grows for the larger.
            self.pages.wait(u64::from(need.windows[0].max(need.pages)));
            for (at, pool) in self.window_pools.iter_mut().enumerate() {
                pool.wait(u64::from(need.windows.get(at + 1).copied().unwrap_or(0)));
            }
        } else {
            self.pages.wait(u64::from(need.pages.saturating_sub(need.wbuf)));
            self.wpages.wait(u64::from(need.wbuf));
        }
    }

    /// What stands free.
    pub fn free(&self) -> RsUnits {
        self.available()
    }

    /// What is asked for now and, of that, what checkpoints do not merely
    /// keep: as far as the highest number out, so that holding it cuts
    /// nothing off, and a waiter's on top.
    pub fn demand(&self) -> (u32, u32) {
        let reach = self
            .slots
            .high_end()
            .max(self.pages.high_end())
            .max(self.wpages.high_end());
        let out =
            |pool: &ShapePool<RsSlotId>| pool.capacity().saturating_sub(pool.available() as u32);
        let wanted = out(&self.slots)
            .max(out(&self.pages))
            .max(out(&self.wpages));
        let kept: HashSet<RsSlotId> = self
            .checkpoints
            .values()
            .filter_map(|entry| self.working_sets.get(entry.snapshot))
            .flat_map(Self::slots)
            .filter(|&id| !self.is_page(id))
            .collect();
        let live = kept.len() as u32;
        (reach.max(wanted), wanted.saturating_sub(live))
    }

    /// The state is window lists of kv rows, drawn from the pages (list 0)
    /// and the window pools.
    pub fn is_window(&self) -> bool {
        !self.window.is_empty()
    }

    /// Rows can stand unfolded in a set: in its buffer pages, or past its
    /// windows' committed position. A state with neither (a pool with no rs
    /// list: Inkling's short conv state, which no buffered plane carries)
    /// folds every row a fire writes.
    pub fn holds_unfolded(&self) -> bool {
        self.is_window() || self.pages.capacity() > 0
    }

    /// The window lists, in their order.
    pub fn window_lists(&self) -> &[WindowGeometry] {
        &self.window
    }

    /// What is held and wanted of the device's state slots and buffer (or
    /// window) pages, in bytes.
    pub fn device_demand(&self) -> Demand {
        let mut demand = Demand::default();
        for pool in [&self.slots, &self.pages, &self.wpages]
            .into_iter()
            .chain(&self.window_pools)
        {
            let of = pool.demand();
            demand.live += of.live;
            demand.wanted += of.wanted;
        }
        demand
    }

    /// The bytes of a state slot and of a buffer (or ring) page.
    pub fn unit_bytes(&self) -> (u64, u64) {
        (self.slots.slot_bytes(), self.pages.slot_bytes())
    }

    pub fn retire_idle(&mut self) {
        let completed = match self.inflight.keys().next() {
            Some(&oldest) => oldest.saturating_sub(1),
            None => u64::MAX,
        };
        self.slots.retire_through(completed);
        self.pages.retire_through(completed);
        self.wpages.retire_through(completed);
        for pool in &mut self.window_pools {
            pool.retire_through(completed);
        }
    }

    /// The window list `id` is a slot of, where the state is windows.
    fn list_of(&self, id: RsSlotId) -> Option<usize> {
        if self.window.is_empty() {
            return None;
        }
        if let Some(row) = self.disk_row(id) {
            return row.list;
        }
        if self.pages.owns(id) || self.host_pages.contains(id) {
            return Some(0);
        }
        self.window_pools
            .iter()
            .position(|pool| pool.owns(id))
            .or_else(|| self.host_windows.iter().position(|pool| pool.contains(id)))
            .map(|at| at + 1)
    }

    /// The host pool of window list `list > 0`'s slot `id`, where it is one.
    fn host_window(&self, id: RsSlotId) -> Option<&ShapePool<RsSlotId>> {
        self.host_windows.iter().find(|pool| pool.contains(id))
    }

    fn list_pool(&mut self, list: usize) -> &mut ShapePool<RsSlotId> {
        if list == 0 {
            &mut self.pages
        } else {
            &mut self.window_pools[list - 1]
        }
    }

    #[must_use]
    pub fn engine_id(&self, id: RsSlotId) -> u32 {
        if self.host_wpages.contains(id) {
            id.0 - self.host_wpages.base()
        } else if self.wpages.owns(id) {
            id.0 - self.wpages.base()
        } else if self.host_pages.contains(id) {
            id.0 - self.host_pages.base()
        } else if let Some(pool) = self.host_window(id) {
            id.0 - pool.base()
        } else if self.host_slots.contains(id) {
            id.0 - self.host_slots.base()
        } else if let Some(list) = self.list_of(id).filter(|&list| list > 0) {
            id.0 - self.window_pools[list - 1].base()
        } else if self.is_page(id) {
            id.0 - self.pages_at
        } else {
            id.0
        }
    }

    #[must_use]
    pub fn page_id(&self, page: u32) -> RsSlotId {
        RsSlotId(page + self.pages_at)
    }

    /// The id of window list `list`'s slot the engine numbers `slot`.
    #[must_use]
    pub fn window_slot(&self, list: usize, slot: u32) -> RsSlotId {
        match list {
            0 => self.page_id(slot),
            _ => RsSlotId(slot + self.window_pools[list - 1].base()),
        }
    }

    fn is_page(&self, id: RsSlotId) -> bool {
        if let Some(row) = self.disk_row(id) {
            return row.page;
        }
        (id.0 >= self.pages_at && id.0 < self.host_slots.base())
            || self.host_pages.contains(id)
            || self.host_wpages.contains(id)
            || self.host_window(id).is_some()
    }

    fn units(&self, ids: impl IntoIterator<Item = RsSlotId>) -> RsUnits {
        let mut units = RsUnits::default();
        for id in ids {
            if self.is_page(id) {
                units.pages += 1;
                if self.is_wpage(id) {
                    units.wbuf += 1;
                }
                if let Some(list) = self.list_of(id) {
                    units.windows[list] += 1;
                }
            } else {
                units.slots += 1;
            }
        }
        units
    }

    fn device_pool(&mut self, id: RsSlotId) -> &mut ShapePool<RsSlotId> {
        if self.is_wpage(id) {
            &mut self.wpages
        } else if let Some(list) = self.list_of(id).filter(|&list| list > 0) {
            &mut self.window_pools[list - 1]
        } else if self.is_page(id) {
            &mut self.pages
        } else {
            &mut self.slots
        }
    }

    fn host_pool(&mut self, id: RsSlotId) -> &mut ShapePool<RsSlotId> {
        if let Some(at) = self.host_windows.iter().position(|pool| pool.contains(id)) {
            &mut self.host_windows[at]
        } else if self.host_wpages.contains(id) {
            &mut self.host_wpages
        } else if self.is_page(id) {
            &mut self.host_pages
        } else {
            &mut self.host_slots
        }
    }

    /// The host pool a device slot `id` parks in: its window list's, the
    /// pages' (the rs list, or window list 0), or the records'.
    fn host_pool_for(&mut self, id: RsSlotId) -> &mut ShapePool<RsSlotId> {
        match self.list_of(id) {
            Some(list) if list > 0 => &mut self.host_windows[list - 1],
            _ if self.is_wpage(id) => &mut self.host_wpages,
            _ if self.is_page(id) => &mut self.host_pages,
            _ => &mut self.host_slots,
        }
    }

    fn recycle(&mut self, ids: Vec<RsSlotId>, epoch: u64) {
        for id in ids {
            self.device_pool(id).recycle_after_epoch(vec![id], epoch);
        }
    }

    fn release_device(&mut self, ids: impl IntoIterator<Item = RsSlotId>) {
        for id in ids {
            self.device_pool(id).release_reserved(vec![id]);
        }
    }

    fn release_host(&mut self, ids: impl IntoIterator<Item = RsSlotId>) {
        for id in ids {
            if self.is_disk(id) {
                self.give_disk_rows([id]);
            } else {
                self.host_pool(id).release_reserved(vec![id]);
            }
        }
    }

    /// Records slots the fire published under `seq` reads beyond those it
    /// writes or copies, so none is recycled while it still reads them.
    pub fn note_reads(&mut self, seq: u64, ids: impl IntoIterator<Item = RsSlotId>) {
        if let Some(touch) = self.inflight.get_mut(&seq) {
            let ids: Vec<RsSlotId> = ids.into_iter().collect();
            touch.reads.extend(&ids);
            self.touch_ids(seq, ids);
        }
    }

    fn busy_slots(&self) -> HashSet<RsSlotId> {
        self.inflight
            .values()
            .flat_map(|touch| touch.writes.iter().copied())
            .collect()
    }

    /// A set's windows, where the engine keeps window pools.
    fn live_window(&self, entry: &RsEntry) -> Option<WindowState> {
        entry.window.filter(|_| self.pages.capacity() > 0)
    }

    pub fn create_working_set(&mut self, geom: RsGeometry) -> RsWorkingSetId {
        let windowed = !self.window.is_empty();
        self.working_sets.insert(RsEntry {
            geom,
            folded: None,
            buffer: Vec::new(),
            base: 0,
            frozen: 0,
            windows: vec![Vec::new(); self.window.len()],
            window: windowed.then(WindowState::default),
            wbuffer: Vec::new(),
            wbase: 0,
            device_folds: 0,
            device_fold_failed: None,
        })
    }

    /// What a snapshot keeps of `ws`, on the rows it stands on now: not
    /// while its frozen length is behind its record
    /// ([`Self::device_fold_gate`]), which an export leaves the engine with.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn export(&self, ws: RsWorkingSetId) -> Result<RsExport, RsError> {
        self.device_fold_gate(ws)?;
        let entry = self.entry(ws)?;
        Ok(RsExport {
            folded: entry.folded,
            buffer: entry.buffer.clone(),
            base: entry.base,
            frozen: entry.frozen,
            windows: entry.windows.clone(),
            window: entry.window,
            wbuffer: entry.wbuffer.clone(),
            wbase: entry.wbase,
        })
    }

    /// Whether `id` is a host row already.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn on_host(&self, id: RsSlotId) -> bool {
        self.is_host(id) && !self.is_disk(id)
    }

    /// Host rows for a read or a write that is none of the planner's: state
    /// slots first, then buffer pages. `None` if the host lacks either.
    ///
    /// A shared pool's host rows are slot refs of its host tier: records
    /// (a slot of each fixed shape a piece) and rs list or window slots,
    /// whose bytes the engine reads and writes by those refs
    /// ([`Self::tier_refs`], `Engine::read_tier_slots`).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn take_host(&mut self, slots: usize, pages: usize) -> Option<Vec<RsSlotId>> {
        self.take_host_of(slots, pages, 0)
    }

    /// [`Self::take_host`], and `wpages` of a hybrid's window-buffer rows
    /// after the pages.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn take_host_of(
        &mut self,
        slots: usize,
        pages: usize,
        wpages: usize,
    ) -> Option<Vec<RsSlotId>> {
        let mut rows = self.host_slots.try_alloc_n(slots)?;
        let Some(buffer) = self.host_pages.try_alloc_n(pages) else {
            self.host_slots.release_reserved(rows);
            return None;
        };
        let Some(window) = self.host_wpages.try_alloc_n(wpages) else {
            self.host_slots.release_reserved(rows);
            self.host_pages.release_reserved(buffer);
            return None;
        };
        rows.extend(buffer);
        rows.extend(window);
        Some(rows)
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn give_host(&mut self, rows: Vec<RsSlotId>) {
        self.release_host(rows);
    }

    /// A shared pool's host rows `rows` as the engine names them: for each
    /// row in order, the refs of each shape it spans (a record's pieces, a
    /// list slot), runs of one shape merged. The order the bytes of
    /// `Engine::read_tier_slots` lie in.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn tier_refs(&self, rows: &[RsSlotId]) -> Vec<(::engine::chunk::ShapeId, Vec<u32>)> {
        let mut out: Vec<(::engine::chunk::ShapeId, Vec<u32>)> = Vec::new();
        for &row in rows {
            for (shape, refs) in self.row_refs(row) {
                match out.last_mut() {
                    Some((last, all)) if *last == shape => all.extend(refs),
                    _ => out.push((shape, refs)),
                }
            }
        }
        out
    }

    /// A shared pool's copy of device rows to host rows, `(device, host)`
    /// pairs, shape by shape: what `copy_slots` moves.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn tier_copies(
        &self,
        moves: &[(RsSlotId, RsSlotId)],
    ) -> Vec<(::engine::chunk::ShapeId, Vec<(u32, u32)>)> {
        let mut out: Vec<(::engine::chunk::ShapeId, Vec<(u32, u32)>)> = Vec::new();
        for &(device, host) in moves {
            let host_refs = self.row_refs(host);
            for ((shape, device), (_, host)) in self
                .pool_of(device)
                .slot_refs(device)
                .into_iter()
                .zip(host_refs)
            {
                let pairs = device.into_iter().zip(host);
                match out.iter_mut().find(|(at, _)| *at == shape) {
                    Some((_, all)) => all.extend(pairs),
                    None => out.push((shape, pairs.collect())),
                }
            }
        }
        out
    }

    /// A set of its own on host rows, as `export` said one stood.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn import(&mut self, geom: RsGeometry, set: RsExport) -> RsWorkingSetId {
        self.working_sets.insert(RsEntry {
            geom,
            folded: set.folded,
            buffer: set.buffer,
            base: set.base,
            frozen: set.frozen,
            windows: set.windows,
            window: set.window,
            wbuffer: set.wbuffer,
            wbase: set.wbase,
            device_folds: 0,
            device_fold_failed: None,
        })
    }

    /// Not while a fire on its way folds `ws` by a count the device
    /// computes, nor once such a fire failed ([`Self::device_fold_gate`]):
    /// the child would state a frozen length behind its copy of the record.
    pub fn fork(&mut self, ws: RsWorkingSetId) -> Result<RsWorkingSetId, RsError> {
        self.device_fold_gate(ws)?;
        let entry = self.entry(ws)?.clone();
        for id in Self::slots(&entry) {
            *self.refs.entry(id).or_insert(1) += 1;
        }
        Ok(self.working_sets.insert(entry))
    }

    /// A fire that folds `ws` by a count the device computes was submitted:
    /// from here until [`Self::end_device_fold`] its record is ahead of its
    /// frozen length by slots the runtime cannot count.
    pub fn begin_device_fold(&mut self, ws: RsWorkingSetId) {
        if let Ok(entry) = self.entry_mut(ws) {
            entry.device_folds += 1;
        }
    }

    /// The fire [`Self::begin_device_fold`] counted is done, however it
    /// ended: `slots` it froze, where it ran and reported them, and
    /// `failure` why it did not run to its end. The count lands and the
    /// mark lifts under one lock, so whoever waited sees the frozen length
    /// the fold left.
    pub fn end_device_fold(
        &mut self,
        ws: RsWorkingSetId,
        slots: Option<u32>,
        failure: Option<&str>,
    ) {
        if let Some(slots) = slots {
            self.fold(ws, slots);
        }
        if let Ok(entry) = self.entry_mut(ws) {
            entry.device_folds = entry.device_folds.saturating_sub(1);
            if let Some(failure) = failure {
                entry
                    .device_fold_failed
                    .get_or_insert_with(|| failure.to_string());
            }
        }
        self.device_folded.notify_waiters();
    }

    /// Whether `ws`'s frozen length is where its record stands: what a copy
    /// of the record, or the record somewhere the engine does not follow it
    /// (another tier, an export), is told to replay from.
    pub fn device_fold_gate(&self, ws: RsWorkingSetId) -> Result<(), RsError> {
        let entry = self.entry(ws)?;
        if entry.device_folds > 0 {
            return Err(RsError::DeviceFoldUnsettled);
        }
        match &entry.device_fold_failed {
            Some(failure) => Err(RsError::DeviceFoldFailed(failure.clone())),
            None => Ok(()),
        }
    }

    /// Told whenever a fold the device counted settles
    /// ([`Self::device_fold_gate`] may have turned): arm it before looking.
    pub fn device_folds_ended(&self) -> std::sync::Arc<tokio::sync::Notify> {
        std::sync::Arc::clone(&self.device_folded)
    }

    /// The slots of `working_sets` a suspend leaves on the device: those a
    /// fire on its way writes, and every slot of a set a fire on its way
    /// folds by a count the device computes, or one that failed did, whose
    /// record the engine alone knows the length of, and only where it
    /// stands.
    fn unsettled_slots(&self, working_sets: &HashSet<RsWorkingSetId>) -> HashSet<RsSlotId> {
        let mut slots = self.busy_slots();
        slots.extend(
            working_sets
                .iter()
                .filter_map(|&ws| self.working_sets.get(ws))
                .filter(|entry| entry.device_folds > 0 || entry.device_fold_failed.is_some())
                .flat_map(Self::slots),
        );
        slots
    }

    /// Told whenever a move lets slots go ([`Self::in_move`] may have
    /// turned false): arm it before looking.
    pub fn moves_ended(&self) -> std::sync::Arc<tokio::sync::Notify> {
        std::sync::Arc::clone(&self.moved)
    }

    /// Whether a move holds a slot `ws` names.
    pub fn in_move(&self, ws: RsWorkingSetId) -> bool {
        self.entry(ws)
            .is_ok_and(|entry| Self::slots(entry).any(|id| self.moving.contains(&id)))
    }

    /// Releases `ws`, returning how many slots it was the last holder of.
    pub fn release_working_set(&mut self, ws: RsWorkingSetId) -> usize {
        let Some(entry) = self.working_sets.remove(ws) else {
            return 0;
        };
        if entry.device_folds > 0 {
            // Nothing is left to wait for.
            self.device_folded.notify_waiters();
        }
        let ids: Vec<RsSlotId> = Self::slots(&entry).collect();
        self.decref_all(&ids)
    }

    /// The tokens a set's state stands at: a window's committed position,
    /// else the count its state holds.
    fn position(entry: &RsEntry) -> u32 {
        entry.window.map_or(entry.frozen, |window| window.committed)
    }

    fn slots(entry: &RsEntry) -> impl Iterator<Item = RsSlotId> + '_ {
        entry
            .folded
            .into_iter()
            .chain(entry.buffer.iter().flatten().copied())
            .chain(entry.wbuffer.iter().flatten().copied())
            .chain(entry.windows.iter().flatten().flatten().copied())
    }

    fn drop_checkpoints(&mut self, drop: impl Fn(&Self, RsWorkingSetId) -> bool) -> usize {
        let keys: Vec<Hash256> = self
            .checkpoints
            .iter()
            .filter(|&(_, entry)| drop(self, entry.snapshot))
            .map(|(&key, _)| key)
            .collect();
        self.drop_keys(&keys);
        keys.len()
    }

    fn drop_keys(&mut self, keys: &[Hash256]) -> usize {
        let mut freed = 0;
        for key in keys {
            if let Some(entry) = self.checkpoints.remove(key) {
                self.cache_holds(*key, entry.snapshot, false);
                freed += self.release_working_set(entry.snapshot);
            }
        }
        freed
    }

    /// Drops the snapshots sharing a slot with `working_sets` whose retention
    /// key lets the runtime drop them unasked (`pressure::may_drop`), so the
    /// sets' next write lands in place instead of copying to a slot the pool
    /// lacks. One a policy keyed higher stays: room for it is asked for with
    /// tier pressure. Returns how many were dropped.
    pub fn reclaim_sharing(
        &mut self,
        working_sets: &[RsWorkingSetId],
        keys: impl crate::policy::retention::Keys,
    ) -> usize {
        let sharing = self.sharing_checkpoints(&working_sets.iter().copied().collect());
        let droppable: HashSet<RsWorkingSetId> = self
            .checkpoints
            .values()
            .filter(|entry| sharing.contains(&entry.snapshot))
            .filter(|entry| {
                self.working_sets.get(entry.snapshot).is_some_and(|ws| {
                    let at = u64::from(Self::position(ws)).saturating_sub(1);
                    crate::policy::pressure::may_drop(keys.key_at(entry.owner, at))
                })
            })
            .map(|entry| entry.snapshot)
            .collect();
        self.drop_checkpoints(|_, snapshot| droppable.contains(&snapshot))
    }

    /// The rs half of idle reclaim, in the kv half's order. The runtime runs
    /// it only as [`Self::reclaim_droppable`].
    #[cfg(test)]
    pub fn reclaim_cache(&mut self, keys: impl crate::policy::retention::Keys) -> usize {
        self.reclaim_cache_where(keys, |_| true)
    }

    /// [`Self::reclaim_cache`] as the runtime may run it unasked: only
    /// checkpoints whose key says they may go (`pressure::may_drop`).
    pub fn reclaim_droppable(&mut self, keys: impl crate::policy::retention::Keys) -> usize {
        self.reclaim_cache_where(keys, crate::policy::pressure::may_drop)
    }

    fn reclaim_cache_where(
        &mut self,
        keys: impl crate::policy::retention::Keys,
        drops: fn(i64) -> bool,
    ) -> usize {
        if self.checkpoints.is_empty() {
            return 0;
        }
        let snapshots: HashSet<RsWorkingSetId> = self
            .checkpoints
            .values()
            .map(|entry| entry.snapshot)
            .collect();
        let live: HashSet<RsSlotId> = self
            .working_sets
            .iter()
            .filter(|(ws, _)| !snapshots.contains(ws))
            .flat_map(|(_, entry)| Self::slots(entry))
            .collect();
        let mut groups: HashMap<(Option<ProcessId>, i64), (Vec<Hash256>, Instant)> = HashMap::new();
        for (&key, entry) in &self.checkpoints {
            let Some(ws) = self.working_sets.get(entry.snapshot) else {
                continue;
            };
            if Self::slots(ws).any(|id| !live.contains(&id)) {
                let at = u64::from(Self::position(ws)).saturating_sub(1);
                let group = groups
                    .entry((entry.owner, keys.key_at(entry.owner, at)))
                    .or_insert_with(|| (Vec::new(), entry.last_use));
                group.0.push(key);
                group.1 = group.1.max(entry.last_use);
            }
        }
        let mut order: Vec<_> = groups
            .into_iter()
            .filter(|((_, key), _)| drops(*key))
            .map(|((_, key), (keys, last_use))| ((key, last_use), keys))
            .collect();
        order.sort_unstable_by_key(|(order, _)| *order);
        let mut freed = 0;
        for (_, keys) in order {
            freed += self.drop_keys(&keys);
            if freed > 0 {
                break;
            }
        }
        self.retire_idle();
        freed
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn discard_cache(&mut self, owner: ProcessId) -> usize {
        let keys: Vec<Hash256> = self
            .checkpoints
            .iter()
            .filter(|(_, entry)| entry.owner == Some(owner))
            .map(|(&key, _)| key)
            .collect();
        let freed = self.drop_keys(&keys);
        self.retire_idle();
        freed
    }

    fn unsettled(&self, snapshot: RsWorkingSetId) -> bool {
        let busy = self.busy_slots();
        self.working_sets
            .get(snapshot)
            .is_none_or(|entry| Self::slots(entry).any(|id| busy.contains(&id)))
    }

    /// Not when a fire on its way still writes it, or a suspend took it to the
    /// host. A window's checkpoint is its pages within the window, shared by
    /// refcount like the kv list's (CONTRACT-BC 19); another state's, its
    /// folded state. Refused by name while the frozen length is behind the
    /// record ([`Self::device_fold_gate`]): a checkpoint is a fork.
    pub fn checkpoint(
        &mut self,
        ws: RsWorkingSetId,
        key: Hash256,
        owner: Option<ProcessId>,
    ) -> Result<bool, RsError> {
        let entry = self.entry(ws)?;
        let empty = match entry.window {
            Some(window) => {
                window.committed == 0 || entry.windows.iter().flatten().all(Option::is_none)
            }
            None => entry.folded.is_none(),
        };
        if empty || self.holds_host(entry) || self.unsettled(ws) {
            return Ok(false);
        }
        let snapshot = self.fork(ws)?;
        self.keep(snapshot, key, owner);
        Ok(true)
    }

    /// A window's checkpoint as of position `at`, at or behind where its
    /// windows stand: the slots each list's window reaches from `at` (those
    /// `[at - window, at)` touches), shared by refcount. A slot holding rows
    /// past `at` too is shared as it is; a resume writing into it copies it
    /// first (the tail rule of [`Self::prepare_window`]). None when a slot it
    /// needs has retired, or as [`Self::checkpoint`]; at the windows' own
    /// position, [`Self::checkpoint`]. Another state checkpoints only there.
    ///
    /// Window-list path: removed by the windowed-state change (the integration
    /// commit deletes it once every engine states `Capabilities::windowed_state`).
    pub fn checkpoint_at(
        &mut self,
        ws: RsWorkingSetId,
        key: Hash256,
        owner: Option<ProcessId>,
        at: u32,
    ) -> Result<bool, RsError> {
        let entry = self.entry(ws)?;
        let Some(window) = entry.window else {
            return self.checkpoint(ws, key, owner);
        };
        if at == window.committed {
            return self.checkpoint(ws, key, owner);
        }
        if at == 0 || at > window.committed || self.holds_host(entry) || self.unsettled(ws) {
            return Ok(false);
        }
        self.device_fold_gate(ws)?;
        let mut snapshot = entry.clone();
        let mut state = WindowState {
            committed: at,
            head: at,
            ..window
        };
        for (list, (geom, table)) in self.window.iter().zip(&mut snapshot.windows).enumerate() {
            let tokens = geom.tokens.max(1);
            let first = (at.saturating_sub(geom.window) / tokens) as usize;
            let end = at.div_ceil(tokens) as usize;
            if table.len() < end || table[first..end].iter().any(Option::is_none) {
                return Ok(false);
            }
            table.truncate(end);
            table[..first].iter_mut().for_each(|slot| *slot = None);
            state.retired[list] = first as u32;
        }
        snapshot.window = Some(state);
        for id in Self::slots(&snapshot) {
            *self.refs.entry(id).or_insert(1) += 1;
        }
        let snapshot = self.working_sets.insert(snapshot);
        self.keep(snapshot, key, owner);
        Ok(true)
    }

    /// Keeps `snapshot`, a hidden working set no process holds, as `key`.
    pub fn keep(&mut self, snapshot: RsWorkingSetId, key: Hash256, owner: Option<ProcessId>) {
        let kept = RsCheckpoint {
            snapshot,
            owner,
            last_use: Instant::now(),
        };
        if let Some(old) = self.checkpoints.insert(key, kept) {
            self.cache_holds(key, old.snapshot, false);
            self.release_working_set(old.snapshot);
        }
        self.cache_holds(key, snapshot, true);
    }

    /// The checkpoints among `scope` let go (`hold` false) or take again
    /// their cache entries' holds: around a move that renames their rows.
    fn cache_holds_in(&mut self, scope: &HashSet<RsWorkingSetId>, hold: bool) {
        let kept: Vec<(Hash256, RsWorkingSetId)> = self
            .checkpoints
            .iter()
            .filter(|(_, entry)| scope.contains(&entry.snapshot))
            .map(|(&key, entry)| (key, entry.snapshot))
            .collect();
        for (key, snapshot) in kept {
            self.cache_holds(key, snapshot, hold);
        }
    }

    /// The checkpoint `key` takes (`hold`) or lets go its cache entry's
    /// hold on the device rows of `snapshot`.
    fn cache_holds(&mut self, key: Hash256, snapshot: RsWorkingSetId, hold: bool) {
        let Some(entry) = self.working_sets.get(snapshot) else {
            return;
        };
        let ids: BTreeSet<RsSlotId> = Self::slots(entry).filter(|&id| !self.is_host(id)).collect();
        // By pool: state slots, pages (or window list 0), each other list.
        let mut by_pool: BTreeMap<usize, Vec<RsSlotId>> = BTreeMap::new();
        for id in ids {
            let at = match self.list_of(id) {
                Some(list) => list + 1,
                None if self.is_wpage(id) => usize::MAX,
                None => usize::from(self.is_page(id)),
            };
            by_pool.entry(at).or_default().push(id);
        }
        let holder = Holder::CacheEntry(key);
        for (at, ids) in by_pool {
            let pool = match at {
                0 => &mut self.slots,
                usize::MAX => &mut self.wpages,
                at => self.list_pool(at - 1),
            };
            if hold {
                pool.share(holder, &ids);
            } else {
                pool.unshare(holder, &ids);
            }
        }
    }

    /// The host pool `id` is one of.
    fn host_pool_of(&self, id: RsSlotId) -> &ShapePool<RsSlotId> {
        if let Some(pool) = self.host_window(id) {
            pool
        } else if self.host_wpages.contains(id) {
            &self.host_wpages
        } else if self.host_pages.contains(id) {
            &self.host_pages
        } else {
            &self.host_slots
        }
    }

    /// The device pool `id` is one of.
    fn pool_of(&self, id: RsSlotId) -> &ShapePool<RsSlotId> {
        if self.is_wpage(id) {
            return &self.wpages;
        }
        match self.list_of(id) {
            Some(0) => &self.pages,
            Some(list) => &self.window_pools[list - 1],
            None if self.is_page(id) => &self.pages,
            None => &self.slots,
        }
    }

    /// Device rows only checkpoints hold: every reference the store
    /// counts on them is a checkpoint's cache entry's.
    pub fn cache_only(&self) -> RsUnits {
        let mut ids = BTreeSet::new();
        for entry in self.checkpoints.values() {
            if let Some(snapshot) = self.working_sets.get(entry.snapshot) {
                ids.extend(Self::slots(snapshot).filter(|&id| !self.is_host(id)));
            }
        }
        self.units(
            ids.into_iter()
                .filter(|&id| self.pool_of(id).cache_refs(id) == self.ref_count(id)),
        )
    }

    /// Whether any checkpoint is kept: cache a policy could let go.
    pub fn holds_checkpoints(&self) -> bool {
        !self.checkpoints.is_empty()
    }

    /// Not one parked on the host with the set it shares state with, or on
    /// its way there or back: it is read from the device only.
    pub fn has_checkpoint(&self, key: &Hash256) -> bool {
        self.checkpoints.get(key).is_some_and(|entry| {
            !self.unsettled(entry.snapshot)
                && self.working_sets.get(entry.snapshot).is_some_and(|ws| {
                    !self.holds_host(ws) && Self::slots(ws).all(|id| !self.moving.contains(&id))
                })
        })
    }

    /// Sharing its slots until a write copies them.
    pub fn adopt_checkpoint(&mut self, ws: RsWorkingSetId, key: &Hash256) -> Result<bool, RsError> {
        let fresh = {
            let entry = self.entry(ws)?;
            entry.folded.is_none()
                && entry.buffer.iter().all(Option::is_none)
                && entry.wbuffer.iter().all(Option::is_none)
                && entry.windows.iter().flatten().all(Option::is_none)
                && entry.window.is_none_or(|window| window.committed == 0)
        };
        let Some(snapshot) = self
            .checkpoints
            .get(key)
            .map(|entry| entry.snapshot)
            .filter(|_| fresh && self.has_checkpoint(key))
        else {
            return Ok(false);
        };
        let from = self.entry(snapshot)?.clone();
        for id in Self::slots(&from) {
            *self.refs.entry(id).or_insert(1) += 1;
        }
        let entry = self.entry_mut(ws)?;
        entry.folded = from.folded;
        entry.buffer = from.buffer;
        entry.base = from.base;
        entry.frozen = from.frozen;
        entry.windows = from.windows;
        entry.window = from.window;
        entry.wbuffer = from.wbuffer;
        entry.wbase = from.wbase;
        if let Some(entry) = self.checkpoints.get_mut(key) {
            entry.last_use = Instant::now();
        }
        Ok(true)
    }

    pub fn forget_state(&mut self, ws: RsWorkingSetId) -> Result<(), RsError> {
        let (folded, buffer, windows, wbuffer) = {
            let entry = self.entry_mut(ws)?;
            entry.base = 0;
            entry.wbase = 0;
            entry.frozen = 0;
            // No record is left for a frozen length to be behind.
            entry.device_fold_failed = None;
            entry.window = entry.window.map(|_| WindowState::default());
            let lists = entry.windows.len();
            (
                entry.folded.take(),
                std::mem::take(&mut entry.buffer),
                std::mem::replace(&mut entry.windows, vec![Vec::new(); lists]),
                std::mem::take(&mut entry.wbuffer),
            )
        };
        let windows = windows.into_iter().flatten();
        for id in folded
            .into_iter()
            .chain(buffer.into_iter().flatten())
            .chain(wbuffer.into_iter().flatten())
            .chain(windows.flatten())
        {
            self.decref(id);
        }
        Ok(())
    }

    /// The slots of each window list a fire landing rows `rows` of `entry`
    /// writes, `(list, index, held)`; none where the engine keeps no window
    /// pool.
    fn window_targets(
        &self,
        entry: &RsEntry,
        rows: std::ops::Range<u32>,
    ) -> Vec<(usize, u32, Option<RsSlotId>)> {
        if self.live_window(entry).is_none() || rows.is_empty() {
            return Vec::new();
        }
        let mut targets = Vec::new();
        for (list, geom) in self.window.iter().enumerate() {
            let tokens = geom.tokens.max(1);
            let table = &entry.windows[list];
            for index in rows.start / tokens..=(rows.end - 1) / tokens {
                targets.push((list, index, table.get(index as usize).copied().flatten()));
            }
        }
        targets
    }

    /// The pages writing `targets` takes: a fresh one for each the set holds
    /// none for, or shares.
    fn targets_demand(&self, targets: &[(u32, Option<RsSlotId>)]) -> u32 {
        targets
            .iter()
            .filter(|(_, slot)| slot.is_none_or(|id| self.ref_count(id) > 1))
            .count() as u32
    }

    /// Each window list's fresh slots writing `targets` takes: one for each
    /// the set holds none of, or shares.
    fn window_need(&self, targets: &[(usize, u32, Option<RsSlotId>)]) -> [u32; MAX_WINDOW_LISTS] {
        let mut need = [0; MAX_WINDOW_LISTS];
        for &(list, _, held) in targets {
            if held.is_none_or(|id| self.ref_count(id) > 1) {
                need[list] += 1;
            }
        }
        need
    }

    /// The window slots a fire landing rows `rows` of `ws` takes.
    ///
    /// Window-list path: removed by the windowed-state change (the integration
    /// commit deletes it once every engine states `Capabilities::windowed_state`).
    pub fn window_demand(
        &self,
        ws: RsWorkingSetId,
        rows: std::ops::Range<u32>,
    ) -> Result<RsUnits, RsError> {
        let targets = self.window_targets(self.entry(ws)?, rows);
        Ok(RsUnits::of_windows(self.window_need(&targets)))
    }

    /// Prepares a fire that lands rows `rows` of `ws`'s windows, a fresh
    /// slot for each slot of a list it holds none of or shares (a write into
    /// a shared one copies it first: [`RsPreparedWrite::buffer_copy_plan`]),
    /// from `granted` or the pools; [`Self::window_advance`] moves the
    /// windows once the fire's tables have been read.
    ///
    /// Window-list path: removed by the windowed-state change (the integration
    /// commit deletes it once every engine states `Capabilities::windowed_state`).
    pub fn prepare_window(
        &mut self,
        ws: RsWorkingSetId,
        rows: std::ops::Range<u32>,
        granted: Option<&mut RsGrant>,
    ) -> Result<RsPreparedWrite, RsError> {
        let targets = self.window_targets(self.entry(ws)?, rows);
        let need = self.window_need(&targets);
        let mut fresh: Vec<std::vec::IntoIter<RsSlotId>> = match granted {
            Some(granted) => {
                let have = granted.units();
                if !have.covers(RsUnits::of_windows(need)) {
                    return Err(RsError::GrantMismatch {
                        required: RsUnits::of_windows(need),
                        granted: have,
                    });
                }
                granted
                    .windows
                    .iter_mut()
                    .zip(need)
                    .map(|(ids, n)| ids.drain(..n as usize).collect::<Vec<_>>().into_iter())
                    .collect()
            }
            None => self
                .alloc_windows(need)
                .ok_or(RsError::OutOfSlots {
                    requested: RsUnits::of_windows(need),
                    available: self.available(),
                })?
                .into_iter()
                .map(Vec::into_iter)
                .collect(),
        };
        let allocated: Vec<RsSlotId> = fresh.iter().flat_map(|ids| ids.clone()).collect();
        let buffers: Vec<RsBufferTarget> = targets
            .into_iter()
            .map(|(list, index, held)| match held {
                None => RsBufferTarget::Fresh {
                    index,
                    dst: fresh[list].next().expect("allocated covers each list"),
                },
                Some(src) if self.ref_count(src) > 1 => RsBufferTarget::Cow {
                    index,
                    src,
                    dst: fresh[list].next().expect("allocated covers each list"),
                },
                Some(src) => RsBufferTarget::InPlace { index, dst: src },
            })
            .collect();
        self.seq += 1;
        let touch = Touch {
            writes: buffers.iter().map(RsBufferTarget::dst).collect(),
            reads: buffers.iter().filter_map(RsBufferTarget::src).collect(),
        };
        self.touch_ids(self.seq, touch.writes.iter().chain(&touch.reads).copied());
        self.inflight.insert(self.seq, touch);
        Ok(RsPreparedWrite {
            ws,
            state: None,
            buffers,
            allocated,
            seq: self.seq,
            snapshots: Vec::new(),
        })
    }

    /// The window list `id` is a slot of, where the state is windows.
    #[must_use]
    pub fn window_list_of(&self, id: RsSlotId) -> Option<usize> {
        self.list_of(id)
    }

    /// Each window list's table of `ws` as a fire reading rows up to `end`
    /// reads it, in the engine's numbering: slot `i` its rows
    /// `[i * T_w, (i + 1) * T_w)`, 0 where trimmed or not written.
    ///
    /// Window-list path: removed by the windowed-state change (the integration
    /// commit deletes it once every engine states `Capabilities::windowed_state`).
    pub fn window_tables(&self, ws: RsWorkingSetId, end: u32) -> Result<Vec<Vec<u32>>, RsError> {
        let entry = self.entry(ws)?;
        if self.live_window(entry).is_none() {
            return Ok(Vec::new());
        }
        Ok(self
            .window
            .iter()
            .zip(&entry.windows)
            .map(|(geom, table)| {
                (0..end.div_ceil(geom.tokens.max(1)))
                    .map(|index| {
                        table
                            .get(index as usize)
                            .copied()
                            .flatten()
                            .map_or(0, |id| self.engine_id(id))
                    })
                    .collect()
            })
            .collect())
    }

    /// A set's windows' `(committed, head)` positions.
    #[cfg(test)]
    pub fn window_positions(&self, ws: RsWorkingSetId) -> Result<Option<(u32, u32)>, RsError> {
        Ok(self
            .entry(ws)?
            .window
            .map(|window| (window.committed, window.head)))
    }

    /// Moves `ws`'s windows by what its published fire commits: each list's
    /// slots whose last row lies below the committed position less the
    /// list's own window retire, once no fire still reads them.
    ///
    /// Window-list path: removed by the windowed-state change (the integration
    /// commit deletes it once every engine states `Capabilities::windowed_state`).
    pub fn window_advance(&mut self, ws: RsWorkingSetId, advance: WindowAdvance) {
        let lists = self.window.clone();
        let entry = self.entry_mut(ws).expect("batch prevalidated");
        let Some(window) = entry.window.as_mut() else {
            return;
        };
        if advance.fold.is_none() {
            window.committed = window.committed.max(advance.settled.min(window.head));
        }
        if let Some(head) = advance.head {
            window.head = head;
        }
        if let Some(fold) = advance.fold {
            let tail = advance.commit_end.saturating_sub(window.committed);
            window.committed = window.committed.saturating_add(fold.min(tail));
        }
        // The committed position never passes a row the program has not
        // accepted (the lagged fold), so a host-positioned fire retires no
        // slot a stop's rewind still needs. A device-positioned fire commits
        // nothing and is never taken back: it retires behind its reach.
        let reach = if advance.positioned {
            window.committed
        } else {
            advance.reach
        }
        .min(advance.keep.unwrap_or(u32::MAX));
        let mut dropped: Vec<RsSlotId> = Vec::new();
        for (list, geom) in lists.iter().enumerate() {
            let first = reach.saturating_sub(geom.window) / geom.tokens.max(1);
            let retired = &mut window.retired[list];
            if first <= *retired {
                continue;
            }
            let table = &mut entry.windows[list];
            let end = (first as usize).min(table.len());
            let from = (*retired as usize).min(end);
            dropped.extend(table[from..end].iter_mut().filter_map(Option::take));
            *retired = first;
        }
        for id in dropped {
            self.decref(id);
        }
    }

    /// To a fresh slot when `fresh`: other lanes of the fire read the one it
    /// holds.
    #[cfg(test)]
    pub fn group_demand(
        &self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        writes: Option<std::ops::Range<u32>>,
    ) -> Result<RsUnits, RsError> {
        self.group_demand_split(ws, write_state, fresh, writes, None)
    }

    /// [`Self::group_demand`] of a fire buffering `writes` in the rs list
    /// (or a store's one buffer) and `wwrites` in a hybrid's window buffer
    /// ([`Self::split_writes`]).
    pub fn group_demand_split(
        &self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        writes: Option<std::ops::Range<u32>>,
        wwrites: Option<std::ops::Range<u32>>,
    ) -> Result<RsUnits, RsError> {
        let entry = self.entry(ws)?;
        let targets = self.buffer_targets(entry, writes)?;
        let wtargets = self.wbuffer_targets(entry, wwrites)?;
        let state = write_state && (fresh || entry.folded.is_none_or(|id| self.ref_count(id) > 1));
        let wbuf = self.targets_demand(&wtargets);
        Ok(RsUnits {
            slots: u32::from(state),
            pages: self.targets_demand(&targets) + wbuf,
            wbuf,
            ..RsUnits::default()
        })
    }

    /// The rows a hybrid's window-buffer slot holds.
    fn wpage_tokens(&self) -> u32 {
        self.w_tokens.unwrap_or(1).max(1)
    }

    /// The row a hybrid's window buffer's entry 0 holds: where it stands,
    /// or, empty, the folded index rounded down to its slot.
    fn wbuffer_base(&self, entry: &RsEntry) -> u32 {
        if entry.wbuffer.is_empty() {
            entry.frozen - entry.frozen % self.wpage_tokens()
        } else {
            entry.wbase
        }
    }

    /// Where a fire writing `writes` finds a hybrid's window buffer's entry
    /// 0: empty, at the fire's first buffered row rounded down
    /// ([`Self::split_writes`]); else where it stands.
    fn wwrite_base(&self, entry: &RsEntry, writes: &std::ops::Range<u32>) -> u32 {
        if entry.wbuffer.iter().all(Option::is_none) {
            let at = writes.start.max(entry.frozen);
            at - at % self.wpage_tokens()
        } else {
            entry.wbase
        }
    }

    /// The window-buffer slots of a hybrid a fire writing `writes` lands in.
    fn wbuffer_targets(
        &self,
        entry: &RsEntry,
        writes: Option<std::ops::Range<u32>>,
    ) -> Result<Vec<(u32, Option<RsSlotId>)>, RsError> {
        let Some(writes) = writes.filter(|writes| !writes.is_empty() && self.hybrid()) else {
            return Ok(Vec::new());
        };
        if writes.start < entry.frozen {
            return Err(RsError::BelowFrozen {
                slot: writes.start,
                frozen: entry.frozen,
            });
        }
        let page = self.wpage_tokens();
        let base = self.wwrite_base(entry, &writes);
        let first = (writes.start - base) / page;
        let last = (writes.end - 1 - base) / page;
        Ok((first..=last)
            .map(|index| (index, entry.wbuffer.get(index as usize).copied().flatten()))
            .collect())
    }

    /// The rows a buffer page holds: an rs list slot's tokens in a shared
    /// pool, else the set's geometry's.
    fn page_tokens(&self, entry: &RsEntry) -> u32 {
        self.rs_tokens
            .unwrap_or(entry.geom.buffer_page_tokens)
            .max(1)
    }

    /// Where an empty buffer starts: at the state's count, rounded down to
    /// an rs list slot in a shared pool (CONTRACT-BC 18: `base` is aligned
    /// to `T_rs`; the rows below the count in its first slot are folded and
    /// never read).
    fn empty_base(&self, frozen: u32) -> u32 {
        match self.rs_tokens {
            Some(tokens) => frozen - frozen % tokens,
            None => frozen,
        }
    }

    /// The slot the buffer's page 0 starts at.
    fn buffer_base(&self, entry: &RsEntry) -> u32 {
        if entry.buffer.is_empty() {
            self.empty_base(entry.frozen)
        } else {
            entry.base
        }
    }

    /// Where a fire writing `writes` finds the buffer's page 0: an empty
    /// window buffer starts at the fire's first buffered row, rounded down
    /// ([`Self::split_writes`]); any other buffer where it stands.
    fn write_base(&self, entry: &RsEntry, writes: &std::ops::Range<u32>) -> u32 {
        if self.window_buffer && !self.hybrid() && entry.buffer.iter().all(Option::is_none) {
            self.empty_base(writes.start.max(entry.frozen))
        } else {
            self.buffer_base(entry)
        }
    }

    fn buffer_targets(
        &self,
        entry: &RsEntry,
        writes: Option<std::ops::Range<u32>>,
    ) -> Result<Vec<(u32, Option<RsSlotId>)>, RsError> {
        let Some(writes) = writes.filter(|writes| !writes.is_empty()) else {
            return Ok(Vec::new());
        };
        if writes.start < entry.frozen {
            return Err(RsError::BelowFrozen {
                slot: writes.start,
                frozen: entry.frozen,
            });
        }
        let page = self.page_tokens(entry);
        let base = self.write_base(entry, &writes);
        let first = (writes.start - base) / page;
        let last = (writes.end - 1 - base) / page;
        Ok((first..=last)
            .map(|index| (index, entry.buffer.get(index as usize).copied().flatten()))
            .collect())
    }

    #[cfg(test)]
    pub fn prepare_group(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        fold: Option<u32>,
        writes: Option<std::ops::Range<u32>>,
        reserved: Option<&mut RsGrant>,
    ) -> Result<RsPreparedWrite, RsError> {
        self.prepare_group_split(ws, write_state, fresh, fold, writes, None, reserved)
    }

    /// [`Self::prepare_group`] of a fire buffering `writes` in the rs list
    /// (or a store's one buffer) and `wwrites` in a hybrid's window buffer
    /// ([`Self::split_writes`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "the state's write, the fold, and each buffer's rows"
    )]
    pub fn prepare_group_split(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        fold: Option<u32>,
        writes: Option<std::ops::Range<u32>>,
        wwrites: Option<std::ops::Range<u32>>,
        reserved: Option<&mut RsGrant>,
    ) -> Result<RsPreparedWrite, RsError> {
        // Refused (below the state) before the buffer grows.
        self.buffer_targets(self.entry(ws)?, writes.clone())?;
        self.wbuffer_targets(self.entry(ws)?, wwrites.clone())?;
        if let Some(writes) = wwrites
            .as_ref()
            .filter(|writes| !writes.is_empty() && self.hybrid())
        {
            let (base, page) = {
                let entry = self.entry(ws)?;
                (self.wwrite_base(entry, writes), self.wpage_tokens())
            };
            let entry = self.entry_mut(ws)?;
            entry.wbase = base;
            let pages = writes.end.saturating_sub(base).div_ceil(page) as usize;
            if entry.wbuffer.len() < pages {
                entry.wbuffer.resize(pages, None);
            }
        }
        if let Some(writes) = writes.as_ref().filter(|writes| !writes.is_empty()) {
            let (base, page) = {
                let entry = self.entry(ws)?;
                (self.write_base(entry, writes), self.page_tokens(entry))
            };
            let entry = self.entry_mut(ws)?;
            entry.base = base;
            let pages = writes.end.saturating_sub(entry.base).div_ceil(page) as usize;
            if entry.buffer.len() < pages {
                entry.buffer.resize(pages, None);
            }
        }
        let targets = self.buffer_targets(self.entry(ws)?, writes)?;
        let wtargets = self.wbuffer_targets(self.entry(ws)?, wwrites)?;
        self.prepare_targets(ws, write_state, fresh, fold, targets, wtargets, reserved)
    }

    /// Prepares a write of the state (when `write_state`) and of the buffer
    /// `targets`, taking a fresh slot or page for each the set holds none
    /// for or shares, from `reserved` or the pools.
    fn prepare_targets(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        fold_tokens: Option<u32>,
        buffer_targets_src: Vec<(u32, Option<RsSlotId>)>,
        wbuffer_targets_src: Vec<(u32, Option<RsSlotId>)>,
        reserved: Option<&mut RsGrant>,
    ) -> Result<RsPreparedWrite, RsError> {
        let folded = {
            let entry = self.entry(ws)?;
            // Fires lease the kv fence the suspend raised and quiesced, so none reaches here.
            debug_assert!(
                !self.holds_host(entry),
                "an rs write on a suspended working set"
            );
            entry.folded
        };
        let need = RsUnits {
            slots: u32::from(
                write_state && (fresh || folded.is_none_or(|id| self.ref_count(id) > 1)),
            ),
            pages: self.targets_demand(&buffer_targets_src)
                + self.targets_demand(&wbuffer_targets_src),
            wbuf: self.targets_demand(&wbuffer_targets_src),
            ..RsUnits::default()
        };
        let (slots, pages, wpages) = match reserved {
            Some(granted) => {
                if !granted.units().covers(need) {
                    return Err(RsError::GrantMismatch {
                        required: need,
                        granted: granted.units(),
                    });
                }
                (
                    granted.slots.drain(..need.slots as usize).collect(),
                    granted
                        .pages
                        .drain(..(need.pages - need.wbuf) as usize)
                        .collect(),
                    granted.wpages.drain(..need.wbuf as usize).collect(),
                )
            }
            None => self.alloc(need).ok_or(RsError::OutOfSlots {
                requested: need,
                available: self.available(),
            })?,
        };
        let mut fresh_slots = slots.iter().copied();
        let mut fresh_ids = pages.iter().copied();
        let mut fresh_wids = wpages.iter().copied();
        let allocated: Vec<RsSlotId> = slots
            .iter()
            .chain(&pages)
            .chain(&wpages)
            .copied()
            .collect();

        let state = if write_state {
            Some(match folded {
                None => RsStateTarget {
                    slot: fresh_slots.next().expect("allocated for fresh state"),
                    reset: true,
                    copy_from: None,
                    fold_tokens,
                },
                Some(old) if fresh || self.ref_count(old) > 1 => RsStateTarget {
                    slot: fresh_slots.next().expect("allocated for cow state"),
                    reset: false,
                    copy_from: Some(old),
                    fold_tokens,
                },
                Some(old) => RsStateTarget {
                    slot: old,
                    reset: false,
                    copy_from: None,
                    fold_tokens,
                },
            })
        } else {
            None
        };

        let mut buffers: Vec<RsBufferTarget> = buffer_targets_src
            .into_iter()
            .map(|(index, slot)| match slot {
                None => RsBufferTarget::Fresh {
                    index,
                    dst: fresh_ids.next().expect("allocated covers materialize"),
                },
                Some(src) if self.ref_count(src) > 1 => RsBufferTarget::Cow {
                    index,
                    src,
                    dst: fresh_ids.next().expect("allocated covers cow"),
                },
                Some(src) => RsBufferTarget::InPlace { index, dst: src },
            })
            .collect();
        let wbuffers = wbuffer_targets_src
            .into_iter()
            .map(|(index, slot)| match slot {
                None => RsBufferTarget::Fresh {
                    index,
                    dst: fresh_wids.next().expect("allocated covers materialize"),
                },
                Some(src) if self.ref_count(src) > 1 => RsBufferTarget::Cow {
                    index,
                    src,
                    dst: fresh_wids.next().expect("allocated covers cow"),
                },
                Some(src) => RsBufferTarget::InPlace { index, dst: src },
            })
            .collect::<Vec<_>>();
        // A target names its table by its slot's pool (`set_buffer_page`).
        buffers.extend(wbuffers);

        self.seq += 1;
        let touch = Touch {
            writes: state
                .iter()
                .map(|state| state.slot)
                .chain(buffers.iter().map(RsBufferTarget::dst))
                .collect(),
            reads: state
                .iter()
                .filter_map(|state| state.copy_from)
                .chain(buffers.iter().filter_map(RsBufferTarget::src))
                .collect(),
        };
        self.touch_ids(self.seq, touch.writes.iter().chain(&touch.reads).copied());
        self.inflight.insert(self.seq, touch);
        Ok(RsPreparedWrite {
            ws,
            state,
            buffers,
            allocated,
            seq: self.seq,
            snapshots: Vec::new(),
        })
    }

    /// A fresh slot for each of `rows`, where the fire `prepared` lands its
    /// working set's state as of that row; each becomes a hidden working set
    /// on publish ([`RsPublished::snapshots`]).
    pub fn prepare_snapshots(
        &mut self,
        prepared: &mut RsPreparedWrite,
        rows: &[u32],
        reserved: Option<&mut RsGrant>,
    ) -> Result<(), RsError> {
        if rows.is_empty() {
            return Ok(());
        }
        let need = RsUnits {
            slots: rows.len() as u32,
            pages: 0,
            ..RsUnits::default()
        };
        let slots: Vec<RsSlotId> = match reserved {
            Some(granted) if granted.slots.len() >= rows.len() => {
                granted.slots.drain(..rows.len()).collect()
            }
            Some(granted) => {
                return Err(RsError::GrantMismatch {
                    required: need,
                    granted: granted.units(),
                });
            }
            None => {
                self.alloc(need)
                    .ok_or(RsError::OutOfSlots {
                        requested: need,
                        available: self.available(),
                    })?
                    .0
            }
        };
        if let Some(touch) = self.inflight.get_mut(&prepared.seq) {
            touch.writes.extend(&slots);
            self.touch_ids(prepared.seq, slots.iter().copied());
        }
        prepared.allocated.extend(&slots);
        prepared.snapshots = rows.iter().copied().zip(slots).collect();
        Ok(())
    }

    #[cfg(test)]
    pub fn publish_prepared(&mut self, prepared: RsPreparedWrite) -> Result<RsPublished, RsError> {
        let (published, folds) = self.publish_batch(vec![prepared])?;
        self.commit_folds(folds);
        Ok(published)
    }

    pub fn publish_batch(
        &mut self,
        prepared: Vec<RsPreparedWrite>,
    ) -> Result<(RsPublished, RsPendingFolds), RsError> {
        let validation = (|| {
            let mut seen = Vec::with_capacity(prepared.len());
            for write in &prepared {
                self.entry(write.ws)?;
                if seen.contains(&write.ws) {
                    return Err(RsError::DuplicateWorkingSet);
                }
                seen.push(write.ws);
            }
            Ok(())
        })();
        if let Err(error) = validation {
            self.cancel_batch(prepared);
            return Err(error);
        }
        let seqs = prepared
            .iter()
            .map(RsPreparedWrite::seq)
            .collect::<Vec<_>>();
        let mut folds = RsPendingFolds::default();
        let mut snapshots = Vec::new();
        for write in prepared {
            self.publish_prevalidated(write, &mut folds, &mut snapshots);
        }
        Ok((RsPublished::new(seqs, snapshots), folds))
    }

    pub fn commit_folds(&mut self, folds: RsPendingFolds) {
        self.commit_folds_keeping(folds, &[]);
    }

    /// [`RsStore::commit_folds`], the sets of `keep` folding their state
    /// only: a beam group's buffer backs its pool page by page
    /// (`BEAM-PAGE-LISTS.md` §11), whatever slots its state has folded
    /// (§11.3), so none of it goes back while the group runs.
    pub fn commit_folds_keeping(&mut self, folds: RsPendingFolds, keep: &[RsWorkingSetId]) {
        for RsPendingFold { ws, tokens } in folds.0 {
            if keep.contains(&ws) {
                if let Ok(entry) = self.entry_mut(ws) {
                    entry.frozen = entry.frozen.saturating_add(tokens);
                }
            } else {
                self.fold(ws, tokens);
            }
        }
    }

    fn publish_prevalidated(
        &mut self,
        prepared: RsPreparedWrite,
        folds: &mut RsPendingFolds,
        snapshots: &mut Vec<RsSnapshot>,
    ) {
        let ws = prepared.ws;
        let (geom, frozen) = {
            let entry = self.entry(ws).expect("batch prevalidated");
            (entry.geom, entry.frozen)
        };
        for &(row, slot) in &prepared.snapshots {
            self.refs.insert(slot, 1);
            let snapshot = self.working_sets.insert(RsEntry {
                geom,
                folded: Some(slot),
                buffer: Vec::new(),
                base: frozen + row,
                frozen: frozen + row,
                windows: Vec::new(),
                window: None,
                wbuffer: Vec::new(),
                wbase: frozen + row,
                device_folds: 0,
                device_fold_failed: None,
            });
            snapshots.push(RsSnapshot {
                ws,
                row,
                slot,
                snapshot,
            });
        }
        if let Some(state) = &prepared.state {
            let old = self.entry(ws).expect("batch prevalidated").folded;
            if old != Some(state.slot) {
                self.refs.insert(state.slot, 1);
                self.entry_mut(ws).expect("batch prevalidated").folded = Some(state.slot);
                if let Some(old) = old {
                    self.decref(old);
                }
            }
        }

        for target in &prepared.buffers {
            match *target {
                RsBufferTarget::Fresh { index, dst } => {
                    self.refs.insert(dst, 1);
                    self.set_buffer_page(ws, index, dst);
                }
                RsBufferTarget::Cow { index, src, dst } => {
                    self.refs.insert(dst, 1);
                    self.set_buffer_page(ws, index, dst);
                    self.decref(src);
                }
                RsBufferTarget::InPlace { .. } => {}
            }
        }

        if let Some(tokens) = prepared.state.as_ref().and_then(|state| state.fold_tokens) {
            folds.0.push(RsPendingFold { ws, tokens });
        }
    }

    /// A window list's table grows as its sequence does (the list `dst` is
    /// a slot of); a slot buffer was sized by the guest.
    fn set_buffer_page(&mut self, ws: RsWorkingSetId, index: u32, dst: RsSlotId) {
        let list = self.list_of(dst);
        let wpage = self.is_wpage(dst);
        let entry = self.entry_mut(ws).expect("batch prevalidated");
        let table = match list {
            Some(list) => &mut entry.windows[list],
            None if wpage => &mut entry.wbuffer,
            None => &mut entry.buffer,
        };
        let index = index as usize;
        if table.len() <= index {
            table.resize(index + 1, None);
        }
        table[index] = Some(dst);
    }

    pub fn settle(&mut self, published: RsPublished) {
        for seq in published.seqs() {
            self.settle_seq(*seq);
        }
        self.retire_idle();
    }

    pub fn cancel_prepared(&mut self, prepared: RsPreparedWrite) {
        self.recycle(prepared.allocated, prepared.seq);
        self.settle_seq(prepared.seq);
        self.retire_idle();
    }

    pub fn cancel_batch(&mut self, prepared: Vec<RsPreparedWrite>) {
        for write in prepared {
            self.cancel_prepared(write);
        }
    }

    /// The buffer pages wholly below them go.
    pub fn fold(&mut self, ws: RsWorkingSetId, slots: u32) {
        let Ok(page) = self.entry(ws).map(|entry| self.page_tokens(entry)) else {
            return;
        };
        let rs_tokens = self.rs_tokens;
        let wpage = self.wpage_tokens();
        let Ok(entry) = self.entry_mut(ws) else {
            return;
        };
        entry.frozen = entry.frozen.saturating_add(slots);
        let below = (entry.frozen.saturating_sub(entry.base) / page) as usize;
        let drop = below.min(entry.buffer.len());
        entry.base += drop as u32 * page;
        let mut dropped: Vec<RsSlotId> = entry.buffer.drain(..drop).flatten().collect();
        if entry.buffer.is_empty() {
            entry.base = match rs_tokens {
                Some(tokens) => entry.frozen - entry.frozen % tokens,
                None => entry.frozen,
            };
        }
        // A hybrid's window buffer: every slot whose rows are all folded.
        let below = (entry.frozen.saturating_sub(entry.wbase) / wpage) as usize;
        let drop = below.min(entry.wbuffer.len());
        entry.wbase += drop as u32 * wpage;
        dropped.extend(entry.wbuffer.drain(..drop).flatten());
        if entry.wbuffer.is_empty() {
            entry.wbase = entry.frozen - entry.frozen % wpage;
        }
        for id in dropped {
            self.decref(id);
        }
    }

    /// Takes back `ws`'s rows from `cut` on, which only its buffer or its
    /// windows' heads hold: the buffer pages wholly past it go back, and
    /// each window list's slots wholly past it; the head ends at `cut`. A
    /// slot the cut falls inside stays, its rows past the cut stray until
    /// the next write lands over them (copying it first where shared).
    /// Refused where the state, or the windows' committed position, holds
    /// rows past `cut`.
    pub fn truncate(&mut self, ws: RsWorkingSetId, cut: u32) -> Result<(), RsError> {
        if !self.holds_unfolded() {
            self.entry(ws)?;
            return Err(RsError::Unbuffered);
        }
        let page = self.page_tokens(self.entry(ws)?);
        let lists = self.window.clone();
        let base = self.buffer_base(self.entry(ws)?);
        let entry = self.entry_mut(ws)?;
        let committed = entry.window.map_or(entry.frozen, |window| window.committed);
        if committed > cut {
            return Err(RsError::FoldedPastCut { committed, cut });
        }
        let mut dropped: Vec<RsSlotId> = Vec::new();
        let keep = cut.saturating_sub(base).div_ceil(page) as usize;
        if entry.buffer.len() > keep {
            dropped.extend(entry.buffer.drain(keep..).flatten());
        }
        if let Some(window) = entry.window.as_mut() {
            window.head = window.head.min(cut);
            for (geom, table) in lists.iter().zip(entry.windows.iter_mut()) {
                let keep = cut.div_ceil(geom.tokens.max(1)) as usize;
                if table.len() > keep {
                    dropped.extend(table.drain(keep..).flatten());
                }
            }
        }
        for id in dropped {
            self.decref(id);
        }
        Ok(())
    }

    pub fn frozen(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.frozen)
    }

    pub fn base(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.buffer_base(self.entry(ws)?))
    }

    #[cfg(test)]
    pub fn buffer_size(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.buffer.len() as u32)
    }

    pub fn buffer_translation(&self, ws: RsWorkingSetId) -> Result<Vec<u32>, RsError> {
        Ok(self
            .entry(ws)?
            .buffer
            .iter()
            .map(|slot| slot.map_or(RS_TRANSLATION_UNMAPPED, |id| self.engine_id(id)))
            .collect())
    }

    pub fn folded_slot(&self, ws: RsWorkingSetId) -> Result<Option<RsSlotId>, RsError> {
        Ok(self.entry(ws)?.folded)
    }

    /// Each window list's ids `count` gives of its pool.
    fn per_list(&self, count: impl Fn(&ShapePool<RsSlotId>) -> u32) -> RsUnits {
        let mut windows = [0; MAX_WINDOW_LISTS];
        for (list, out) in windows.iter_mut().enumerate().take(self.window.len()) {
            *out = count(if list == 0 {
                &self.pages
            } else {
                &self.window_pools[list - 1]
            });
        }
        RsUnits::of_windows(windows)
    }

    pub fn available(&self) -> RsUnits {
        if self.is_window() {
            return self.per_list(|pool| pool.available() as u32);
        }
        let wbuf = self.wpages.available() as u32;
        RsUnits {
            wbuf,
            ..RsUnits::of(self.slots.available() as u32, self.pages.available() as u32 + wbuf)
        }
    }

    pub fn capacity(&self) -> RsUnits {
        if self.is_window() {
            return self.per_list(ShapePool::capacity);
        }
        let wbuf = self.wpages.capacity();
        RsUnits {
            wbuf,
            ..RsUnits::of(self.slots.capacity(), self.pages.capacity() + wbuf)
        }
    }

    /// One shared by forks counted once.
    pub fn held(&self, working_sets: impl IntoIterator<Item = RsWorkingSetId>) -> RsUnits {
        let mut held = BTreeSet::new();
        for ws in working_sets {
            if let Some(entry) = self.working_sets.get(ws) {
                held.extend(Self::slots(entry).filter(|&id| !self.is_host(id)));
            }
        }
        self.units(held)
    }

    fn is_host(&self, id: RsSlotId) -> bool {
        id.0 >= self.host_slots.base()
    }

    fn holds_host(&self, entry: &RsEntry) -> bool {
        Self::slots(entry).any(|id| self.is_host(id))
    }

    /// Swaps `from` for `to` in every working set of `scope`, returning
    /// how many held it.
    fn swap_within(&mut self, scope: &[RsWorkingSetId], from: RsSlotId, to: RsSlotId) -> u32 {
        let mut held = 0;
        for &ws in scope {
            if let Some(entry) = self.working_sets.get_mut(ws) {
                for id in entry
                    .folded
                    .iter_mut()
                    .chain(entry.buffer.iter_mut().flatten())
                    .chain(entry.wbuffer.iter_mut().flatten())
                    .chain(entry.windows.iter_mut().flatten().flatten())
                {
                    if *id == from {
                        *id = to;
                        held += 1;
                    }
                }
            }
        }
        held
    }

    /// The units every working set holds on every tier, each slot once:
    /// one figure wherever a slot lies, so a move changes no count (what
    /// the capacity gate counts state by, `planner::PoolPort::capacity_census`).
    #[must_use]
    pub fn census_units(&self) -> RsUnits {
        let held: BTreeSet<RsSlotId> = self
            .working_sets
            .iter()
            .flat_map(|(_, entry)| Self::slots(entry))
            .collect();
        self.units(held)
    }

    fn held_within(&self, scope: &HashSet<RsWorkingSetId>) -> BTreeMap<RsSlotId, u32> {
        let mut held = BTreeMap::new();
        for entry in scope.iter().filter_map(|&ws| self.working_sets.get(ws)) {
            for id in Self::slots(entry) {
                *held.entry(id).or_insert(0) += 1;
            }
        }
        held
    }

    fn sharing_checkpoints(
        &self,
        working_sets: &HashSet<RsWorkingSetId>,
    ) -> HashSet<RsWorkingSetId> {
        let held = self.held_within(working_sets);
        self.checkpoints
            .values()
            .map(|entry| entry.snapshot)
            .filter(|&snapshot| {
                self.working_sets
                    .get(snapshot)
                    .is_some_and(|entry| Self::slots(entry).any(|id| held.contains_key(&id)))
            })
            .collect()
    }

    /// The device slots of `working_sets` held by no working set outside
    /// them but checkpoints, which a suspend takes along: left holding a
    /// parked set's slot a checkpoint would pin it on the device, and the
    /// runtime does not drop one on its own.
    fn private(&self, working_sets: &HashSet<RsWorkingSetId>) -> Vec<RsSlotId> {
        let mut scope = self.sharing_checkpoints(working_sets);
        scope.extend(working_sets.iter().copied());
        let shared = self.held_within(&scope);
        self.held_within(working_sets)
            .into_keys()
            .filter(|id| !self.is_host(*id) && self.ref_count(*id) == shared[id])
            .collect()
    }

    /// The device slots `working_sets` alone hold (what a suspend of them
    /// takes along), as the engine's slot refs by shape: what holds their
    /// chunks, for a map of what evicting their process empties.
    #[must_use]
    pub fn private_slot_refs(
        &self,
        working_sets: &HashSet<RsWorkingSetId>,
    ) -> Vec<(::engine::chunk::ShapeId, Vec<u32>)> {
        self.private(working_sets)
            .into_iter()
            .flat_map(|id| self.pool_of(id).slot_refs(id))
            .collect()
    }

    pub fn suspendable(&self, working_sets: &HashSet<RsWorkingSetId>) -> RsUnits {
        self.units(self.private(working_sets))
    }

    fn swapped(&self, working_sets: &HashSet<RsWorkingSetId>) -> Vec<RsSlotId> {
        self.held_within(working_sets)
            .into_keys()
            .filter(|&id| self.is_host(id))
            .collect()
    }

    pub fn swapped_units(&self, working_sets: &HashSet<RsWorkingSetId>) -> RsUnits {
        self.units(self.swapped(working_sets))
    }

    /// Host rows: state slots and buffer pages, or a windowed store's list
    /// 0 pages (only a store of one window list parks on the host).
    pub fn host_available(&self) -> RsUnits {
        self.host_units(|pool| pool.available() as u32)
    }

    pub fn host_capacity(&self) -> RsUnits {
        self.host_units(ShapePool::capacity)
    }

    /// `count` of the host pools: the records', and the pages' or, of a
    /// windowed store, each window list's.
    fn host_units(&self, count: impl Fn(&ShapePool<RsSlotId>) -> u32) -> RsUnits {
        if self.is_window() {
            let mut windows = [0; MAX_WINDOW_LISTS];
            for (out, pool) in windows
                .iter_mut()
                .zip(std::iter::once(&self.host_pages).chain(&self.host_windows))
            {
                *out = count(pool);
            }
            return RsUnits::of_windows(windows);
        }
        let wbuf = count(&self.host_wpages);
        RsUnits {
            wbuf,
            ..RsUnits::of(count(&self.host_slots), count(&self.host_pages) + wbuf)
        }
    }

    /// Host rows for each of `ids`, in order, each of its own pool; none
    /// unless all fit.
    fn take_host_rows(&mut self, ids: &[RsSlotId]) -> Option<Vec<RsSlotId>> {
        let mut taken: Vec<RsSlotId> = Vec::with_capacity(ids.len());
        for &id in ids {
            match self.host_pool_for(id).try_alloc_n(1) {
                Some(row) => taken.extend(row),
                None => {
                    self.release_host(taken);
                    return None;
                }
            }
        }
        Some(taken)
    }

    /// Parks the sets' private, settled slots in host rows, when rows for
    /// all of them are free, and the checkpoints sharing them with them;
    /// otherwise they stay on the device, nothing is dropped, and `None`
    /// says so ([`Self::suspend_short`] tells what the host lacks). A set a
    /// fire on its way folds by a count the device computes is not settled:
    /// its slots stay where the engine holds the record
    /// ([`device_folds_settled`] waits for it), as do those of a set such a
    /// fire failed on, whose frozen length the record may stand past.
    pub fn prepare_suspend(
        &mut self,
        working_sets: &HashSet<RsWorkingSetId>,
    ) -> Option<RsResidencyTxn> {
        let busy = self.unsettled_slots(working_sets);
        let slots: Vec<RsSlotId> = self
            .private(working_sets)
            .into_iter()
            .filter(|id| !busy.contains(id))
            .collect();
        if slots.is_empty() {
            return None;
        }
        // Host rows where the host holds them all, else disk rows: with no
        // host tier (unified memory) the state goes straight to disk.
        let host = match self.take_host_rows(&slots) {
            Some(host) => host,
            None => self.take_disk_rows(&slots)?,
        };
        self.moving.extend(slots.iter().copied());
        let scope = self.with_sharing(working_sets);
        Some(self.txn(&scope, slots.into_iter().zip(host).collect()))
    }

    /// The host rows a suspend of `working_sets` would take beyond those
    /// free: what keeps its state on the device. Zero when it fits.
    pub fn suspend_short(&self, working_sets: &HashSet<RsWorkingSetId>) -> RsUnits {
        let busy = self.unsettled_slots(working_sets);
        let need = self.units(
            self.private(working_sets)
                .into_iter()
                .filter(|id| !busy.contains(id)),
        );
        need.sub(self.host_available())
    }

    /// `working_sets` and the checkpoints sharing a slot with them: a move
    /// takes them as one, so a checkpoint reads where its slot went.
    fn with_sharing(&self, working_sets: &HashSet<RsWorkingSetId>) -> HashSet<RsWorkingSetId> {
        let mut scope = self.sharing_checkpoints(working_sets);
        scope.extend(working_sets.iter().copied());
        scope
    }

    fn txn(
        &self,
        working_sets: &HashSet<RsWorkingSetId>,
        moves: Vec<(RsSlotId, RsSlotId)>,
    ) -> RsResidencyTxn {
        // Only a shared pool has host rows to move to (CONTRACT-BC 22).
        debug_assert!(self.is_shared() || moves.is_empty());
        let mut plan = RsCopyPlan::default();
        for &(from, to) in &moves {
            let (device, host) = if self.is_host(from) {
                (to, from)
            } else {
                (from, to)
            };
            let host_refs = self.row_refs(host);
            let into = if self.is_disk(host) {
                &mut plan.disk
            } else {
                &mut plan.shared
            };
            for ((shape, device), (_, host)) in self
                .pool_of(device)
                .slot_refs(device)
                .into_iter()
                .zip(host_refs)
            {
                let pairs: Vec<(u32, u32)> = device.into_iter().zip(host).collect();
                match into.iter_mut().find(|(at, _)| *at == shape) {
                    Some((_, all)) => all.extend(pairs),
                    None => into.push((shape, pairs)),
                }
            }
        }
        RsResidencyTxn {
            scope: working_sets.iter().copied().collect(),
            moves,
            plan,
        }
    }

    /// One shared outside the set or dropped while the copy ran stays as it is.
    pub fn commit_suspend(&mut self, txn: RsResidencyTxn) -> RsUnits {
        let epoch = self.seq;
        let busy = self.busy_slots();
        let scope: HashSet<RsWorkingSetId> = txn.scope.iter().copied().collect();
        let held = self.held_within(&scope);
        self.cache_holds_in(&scope, false);
        let mut freed = Vec::new();
        self.moved.notify_waiters();
        for (device, host) in txn.moves {
            self.moving.remove(&device);
            // A set released while the copy ran holds nothing; what it still
            // holds is as private and settled as the prepare found it, since
            // the process is fenced.
            let within = held.get(&device).copied().unwrap_or(0);
            if within == 0 {
                self.release_host([host]);
                continue;
            }
            debug_assert!(
                self.ref_count(device) == within && !busy.contains(&device),
                "an rs slot went shared or busy under its suspend"
            );
            self.swap_within(&txn.scope, device, host);
            if let Some(count) = self.refs.remove(&device) {
                self.refs.insert(host, count);
            }
            self.recycle(vec![device], epoch);
            freed.push(device);
        }
        self.cache_holds_in(&scope, true);
        self.retire_idle();
        self.units(freed)
    }

    pub fn abort_suspend(&mut self, txn: RsResidencyTxn) {
        for (device, _) in &txn.moves {
            self.moving.remove(device);
        }
        self.moved.notify_waiters();
        self.release_host(txn.moves.into_iter().map(|(_, host)| host));
    }

    pub fn prepare_restore(
        &mut self,
        working_sets: &HashSet<RsWorkingSetId>,
        granted: &mut RsGrant,
    ) -> Result<RsResidencyTxn, RsError> {
        let swapped = self.swapped(working_sets);
        let need = self.units(swapped.iter().copied());
        if !granted.units().covers(need) {
            return Err(RsError::GrantMismatch {
                required: need,
                granted: granted.units(),
            });
        }
        // Each host row comes back as a slot of its own pool: a window
        // list's from that list's grant.
        let windowed = self.is_window();
        let mut slots: std::vec::IntoIter<RsSlotId> = granted
            .slots
            .drain(..need.slots as usize)
            .collect::<Vec<_>>()
            .into_iter();
        let mut pages: std::vec::IntoIter<RsSlotId> = if windowed {
            Vec::new().into_iter()
        } else {
            granted
                .pages
                .drain(..need.pages.saturating_sub(need.wbuf) as usize)
                .collect::<Vec<_>>()
                .into_iter()
        };
        let mut wpages: std::vec::IntoIter<RsSlotId> = granted
            .wpages
            .drain(..need.wbuf as usize)
            .collect::<Vec<_>>()
            .into_iter();
        let mut lists: Vec<std::vec::IntoIter<RsSlotId>> = granted
            .windows
            .iter_mut()
            .zip(need.windows)
            .map(|(list, n)| list.drain(..n as usize).collect::<Vec<_>>().into_iter())
            .collect();
        let mut moves = Vec::with_capacity(swapped.len());
        for &host in &swapped {
            let device = match self.list_of(host) {
                Some(list) if windowed => lists.get_mut(list).and_then(Iterator::next),
                _ if self.is_wpage(host) => wpages.next(),
                _ if self.is_page(host) => pages.next(),
                _ => slots.next(),
            };
            moves.push((host, device.expect("the grant covers what was swapped")));
        }
        self.moving.extend(swapped.iter().copied());
        let scope = self.with_sharing(working_sets);
        Ok(self.txn(&scope, moves))
    }

    /// Swaps each host slot back for its device copy, returning the slots restored.
    pub fn commit_restore(&mut self, txn: RsResidencyTxn) -> usize {
        let scope: HashSet<RsWorkingSetId> = txn.scope.iter().copied().collect();
        self.cache_holds_in(&scope, false);
        let mut restored = 0;
        self.moved.notify_waiters();
        for (host, device) in txn.moves {
            self.moving.remove(&host);
            let within = self.swap_within(&txn.scope, host, device);
            if within == 0 {
                self.release_device([device]);
                continue;
            }
            restored += 1;
            // A fork made at rest reads the host copy still.
            let others = self.ref_count(host) - within;
            if others > 0 {
                self.refs.insert(host, others);
                self.refs.insert(device, within);
                continue;
            }
            if let Some(count) = self.refs.remove(&host) {
                self.refs.insert(device, count);
            }
            self.release_host([host]);
        }
        self.cache_holds_in(&scope, true);
        restored
    }

    pub fn abort_restore(&mut self, txn: RsResidencyTxn) {
        for (host, _) in &txn.moves {
            self.moving.remove(host);
        }
        self.moved.notify_waiters();
        self.release_device(txn.moves.into_iter().map(|(_, device)| device));
    }

    /// `need`'s state slots, rs pages and (a hybrid's) window-buffer
    /// slots; none unless all fit.
    fn alloc(&mut self, need: RsUnits) -> Option<(Vec<RsSlotId>, Vec<RsSlotId>, Vec<RsSlotId>)> {
        let slots = self.slots.try_alloc_n(need.slots as usize)?;
        let Some(pages) = self
            .pages
            .try_alloc_n(need.pages.saturating_sub(need.wbuf) as usize)
        else {
            self.slots.release_reserved(slots);
            return None;
        };
        let Some(wpages) = self.wpages.try_alloc_n(need.wbuf as usize) else {
            self.slots.release_reserved(slots);
            self.pages.release_reserved(pages);
            return None;
        };
        Some((slots, pages, wpages))
    }

    /// `need[l]` fresh slots of each window list `l`; none unless all fit.
    fn alloc_windows(&mut self, need: [u32; MAX_WINDOW_LISTS]) -> Option<Vec<Vec<RsSlotId>>> {
        let mut taken: Vec<Vec<RsSlotId>> = Vec::with_capacity(self.window.len());
        for (list, &n) in need.iter().enumerate().take(self.window.len()) {
            match self.list_pool(list).try_alloc_n(n as usize) {
                Some(ids) => taken.push(ids),
                None => {
                    for (list, ids) in taken.into_iter().enumerate() {
                        self.list_pool(list).release_reserved(ids);
                    }
                    return None;
                }
            }
        }
        Some(taken)
    }

    pub fn reserve(&mut self, need: RsUnits) -> Option<RsGrant> {
        if self.is_window() {
            return self.alloc_windows(need.windows).map(|windows| RsGrant {
                windows,
                ..RsGrant::default()
            });
        }
        self.alloc(need).map(|(slots, pages, wpages)| RsGrant {
            slots,
            pages,
            windows: Vec::new(),
            wpages,
        })
    }

    pub fn release_reservation(&mut self, grant: RsGrant) {
        self.slots.release_reserved(grant.slots);
        self.pages.release_reserved(grant.pages);
        self.wpages.release_reserved(grant.wpages);
        for (list, ids) in grant.windows.into_iter().enumerate() {
            self.list_pool(list).release_reserved(ids);
        }
    }

    /// A beam search's finish on a hybrid (`BEAM-PAGE-LISTS.md` §6.6,
    /// §11): the set's buffered rows become its chosen beam's, slot by slot.
    /// The buffer's entry holding slot `s` (`f0 S <= s < end`) becomes the
    /// entry that held slot `(f0 + order[s / S - f0]) S + s mod S`; entries
    /// on the frozen pages stay; every entry no longer named is let go once
    /// the fires that may read it retire. Nothing is copied.
    pub fn relist_beam(
        &mut self,
        ws: RsWorkingSetId,
        f0: u32,
        page: u32,
        order: &[u32],
        end: u32,
    ) -> Result<(), RsError> {
        let epoch = self.seq;
        let (base, tokens) = {
            let entry = self.entry(ws)?;
            (self.buffer_base(entry), self.page_tokens(entry))
        };
        let entry = self.entry_mut(ws)?;
        let old = entry.buffer.clone();
        let count = end.saturating_sub(base).div_ceil(tokens) as usize;
        let mut new = Vec::with_capacity(count);
        for e in 0..count as u32 {
            let slot = base + e * tokens;
            if slot < f0 * page {
                new.push(old.get(e as usize).copied().flatten());
                continue;
            }
            let column = (slot / page - f0) as usize;
            let q = *order.get(column).ok_or(RsError::BeamFinish(
                "the chosen beam's list is shorter than its buffered rows",
            ))?;
            let sigma = (f0 + q) * page + slot % page;
            let from = ((sigma - base) / tokens) as usize;
            new.push(old.get(from).copied().flatten());
        }
        let freed: Vec<RsSlotId> = old
            .iter()
            .flatten()
            .copied()
            .filter(|id| !new.contains(&Some(*id)))
            .collect();
        entry.buffer = new;
        self.recycle(freed, epoch);
        Ok(())
    }

    fn entry(&self, ws: RsWorkingSetId) -> Result<&RsEntry, RsError> {
        self.working_sets.get(ws).ok_or(RsError::UnknownWorkingSet)
    }

    fn entry_mut(&mut self, ws: RsWorkingSetId) -> Result<&mut RsEntry, RsError> {
        self.working_sets
            .get_mut(ws)
            .ok_or(RsError::UnknownWorkingSet)
    }

    fn ref_count(&self, id: RsSlotId) -> u32 {
        self.refs.get(&id).copied().unwrap_or(1)
    }

    /// [`Self::decref`] of each of `ids`, the newest unsettled fire touching
    /// each found in one walk over the fires in flight rather than one walk
    /// a slot (a full card's lanes ending together, a windowed ring's dozens
    /// of pages each, held the store's lock for seconds). Returns how many
    /// it was the last holder of.
    fn decref_all(&mut self, ids: &[RsSlotId]) -> usize {
        let mut freed = 0;
        let mut last: Vec<RsSlotId> = Vec::new();
        for &id in ids {
            let count = self.refs.entry(id).or_insert(1);
            *count -= 1;
            if *count != 0 {
                continue;
            }
            self.refs.remove(&id);
            freed += 1;
            if self.is_host(id) {
                self.release_host([id]);
            } else {
                last.push(id);
            }
        }
        if last.is_empty() {
            return freed;
        }
        let settled = self
            .inflight
            .keys()
            .next()
            .map_or(self.seq, |&oldest| oldest - 1);
        for id in last {
            let epoch = self.newest_touch(id).unwrap_or(settled);
            self.recycle(vec![id], epoch);
        }
        // What no fire in flight reads is free now, its chunk with it once
        // empty: left waiting for the next fire's settle, it held its chunk
        // for its shape while nothing fired (a head short of whole chunks,
        // every owner moved, the scheduler silent).
        self.retire_idle();
        freed
    }

    fn touch_ids(&mut self, seq: u64, ids: impl IntoIterator<Item = RsSlotId>) {
        for id in ids {
            let at = id.0 as usize;
            if at >= self.last_touch.len() {
                self.last_touch.resize(at + 1, 0);
                self.pending_touch.resize(at + 1, 0);
            }
            self.last_touch[at] = self.last_touch[at].max(seq);
            self.pending_touch[at] += 1;
        }
    }

    fn settle_seq(&mut self, seq: u64) {
        let Some(touch) = self.inflight.remove(&seq) else {
            return;
        };
        for id in touch.writes.iter().chain(&touch.reads) {
            if let Some(count) = self.pending_touch.get_mut(id.0 as usize) {
                *count = count.saturating_sub(1);
            }
        }
    }

    /// The seq `id` is recycled behind while a fire may still touch it:
    /// none once no unsettled fire touches it, else the newest fire that
    /// touched it. That is the newest unsettled one, unless it settled before
    /// an older one that touches it too, when it is later than needed.
    fn newest_touch(&self, id: RsSlotId) -> Option<u64> {
        let at = id.0 as usize;
        if self.pending_touch.get(at).copied().unwrap_or(0) == 0 {
            return None;
        }
        self.last_touch.get(at).copied()
    }

    /// Drops a reference to `id`, recycling it once none remains, behind
    /// the newest unsettled fire that writes or reads it: no fire still
    /// reading a slot sees it reused, and none that never did holds it.
    fn decref(&mut self, id: RsSlotId) -> bool {
        let count = self.refs.entry(id).or_insert(1);
        *count -= 1;
        if *count != 0 {
            return false;
        }
        self.refs.remove(&id);
        if self.is_host(id) {
            self.release_host([id]);
            return true;
        }
        let epoch = self.newest_touch(id).unwrap_or_else(|| {
            self.inflight
                .keys()
                .next()
                .map_or(self.seq, |&oldest| oldest - 1)
        });
        self.recycle(vec![id], epoch);
        self.retire_idle();
        true
    }
}

/// Returns once no fire on its way folds any of `sets` by a count the
/// device computes: woken by each such fold settling
/// ([`RsStore::end_device_fold`]), which every way a fire ends reaches, and
/// by a set's release. Whether a fold failed is the caller's to read next
/// ([`RsStore::device_fold_gate`]).
pub async fn device_folds_settled(store: &std::sync::Mutex<RsStore>, sets: &[RsWorkingSetId]) {
    let ended = store.lock().unwrap().device_folds_ended();
    loop {
        let notified = ended.notified();
        let mut notified = std::pin::pin!(notified);
        notified.as_mut().enable();
        let unsettled = {
            let store = store.lock().unwrap();
            sets.iter().any(|&ws| {
                store
                    .working_sets
                    .get(ws)
                    .is_some_and(|entry| entry.device_folds > 0)
            })
        };
        if !unsettled {
            return;
        }
        notified.await;
    }
}

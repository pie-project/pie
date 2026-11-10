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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowGeometry {
    pub window: u32,
    pub tokens: u32,
}

/// A set's windows: `committed` is the position they are anchored at and
/// `head` the position written up to, in tokens; `retired[l]` counts the
/// leading slots of list `l`'s table already given back.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowAdvance {
    pub(crate) positioned: bool,
    pub(crate) settled: u32,
    pub(crate) reach: u32,
    pub(crate) fold: Option<u32>,
    pub(crate) commit_end: u32,
    pub(crate) head: Option<u32>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RsError {
    #[error("unknown rs working set")]
    UnknownWorkingSet,
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
        }
    }

    /// Window list slots, `windows[l]` of list `l`; `pages` their sum.
    #[must_use]
    pub fn of_windows(windows: [u32; MAX_WINDOW_LISTS]) -> RsUnits {
        RsUnits {
            slots: 0,
            pages: windows.iter().fold(0u32, |sum, &n| sum.saturating_add(n)),
            windows,
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
        }
    }
}

/// Reserved ids: state slots, buffer pages, and each window list's slots.
#[derive(Debug, Default)]
pub struct RsGrant {
    pub slots: Vec<RsSlotId>,
    pub pages: Vec<RsSlotId>,
    pub windows: Vec<Vec<RsSlotId>>,
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
            pages: (self.pages.len() + window_slots) as u32,
            windows,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty() && self.pages.is_empty() && self.windows.iter().all(Vec::is_empty)
    }
}

/// In the engine's numbering: each kind from 0.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RsCopyPlan {
    pub slots: Vec<(u32, u32)>,
    pub pages: Vec<(u32, u32)>,
}

/// What a set holds, in its own numbering: see [`RsEntry`].
#[derive(Debug, Clone)]
pub struct RsExport {
    pub folded: Option<RsSlotId>,
    pub buffer: Vec<Option<RsSlotId>>,
    pub base: u32,
    pub frozen: u32,
    /// Each window list's table, when the state is windows.
    pub windows: Vec<Vec<Option<RsSlotId>>>,
    /// The windows' position, when the state is windows.
    pub window: Option<WindowState>,
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
}

#[derive(Debug)]
pub struct RsResidencyTxn {
    scope: Vec<RsWorkingSetId>,
    moves: Vec<(RsSlotId, RsSlotId)>,
    plan: RsCopyPlan,
    window: bool,
}

impl RsResidencyTxn {
    pub fn copy_plan(&self) -> &RsCopyPlan {
        &self.plan
    }

    /// The moves are windowed kv pages, copied through the kv planes.
    pub fn is_window(&self) -> bool {
        self.window
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
    /// Where state gains memory as it is handed out, how far it has.
    extent: Option<StateExtent>,
    /// The tokens an rs list slot holds, where the buffer pages are a
    /// shared pool's rs list (CONTRACT-BC 18): a page holds that many rows,
    /// whatever a set's geometry says, and the buffer starts at a multiple
    /// of it.
    rs_tokens: Option<u32>,
}

/// How far the state slots and buffer pages have memory under them; both
/// pools move as one, a slot and a page at each number.
#[derive(Debug, Clone, Copy, Default)]
struct StateExtent {
    target: u32,
    mapped: u32,
    short: u32,
}

#[derive(Debug, Clone, Copy)]
struct RsCheckpoint {
    snapshot: RsWorkingSetId,
    owner: Option<ProcessId>,
    last_use: Instant,
}

impl RsStore {
    #[cfg(test)]
    pub fn new(slots: u32, pages: u32, host_slots: u32, host_pages: u32) -> Self {
        Self::fixed(0, slots, 0, pages, host_slots, host_pages)
    }

    /// The fixed shapes' pools: state slots of `slot_bytes` and buffer
    /// pages of `page_bytes`, on the device and the host.
    fn fixed(
        slot_bytes: u64,
        slots: u32,
        page_bytes: u64,
        pages: u32,
        host_slots: u32,
        host_pages: u32,
    ) -> Self {
        let host = slots + pages;
        Self::build(
            ShapePool::new(ShapeKind::Fixed, slot_bytes, 0, slots),
            ShapePool::new(ShapeKind::Fixed, page_bytes, slots, pages),
            ShapePool::new(ShapeKind::Fixed, slot_bytes, host, host_slots),
            ShapePool::new(ShapeKind::Fixed, page_bytes, host + host_slots, host_pages),
            slots,
            Vec::new(),
            Vec::new(),
        )
    }

    /// A store for a model whose state is one window list of `geom`:
    /// `pages` window pages (0 the null page every page behind a window
    /// reads) and `host_pages` host ones a suspended window parks on.
    #[cfg(test)]
    pub fn new_window(geom: WindowGeometry, pages: u32, host_pages: u32) -> Self {
        Self::windowed(geom, 0, pages, host_pages)
    }

    fn windowed(geom: WindowGeometry, page_bytes: u64, pages: u32, host_pages: u32) -> Self {
        let host = pages.max(1);
        Self::build(
            ShapePool::new(ShapeKind::Fixed, 0, 0, 0),
            ShapePool::new(ShapeKind::List, page_bytes, 1, pages.saturating_sub(1)),
            ShapePool::new(ShapeKind::Fixed, 0, host, 0),
            ShapePool::new(ShapeKind::List, page_bytes, host, host_pages),
            0,
            vec![geom],
            Vec::new(),
        )
    }

    /// The store `pools` states: a window list where there is a window,
    /// else the state slots and buffer pages.
    pub fn of(pools: &StorePools) -> Self {
        if pools.window_tokens > 0 {
            Self::windowed(
                WindowGeometry {
                    window: pools.window_tokens,
                    tokens: pools.page_size,
                },
                pools.window.slot_bytes,
                pools.window_pages(),
                pools.window.host,
            )
        } else {
            Self::fixed(
                pools.state.slot_bytes,
                pools.state.device,
                pools.buffer.slot_bytes,
                pools.buffer.device,
                pools.state.host,
                pools.buffer.host,
            )
        }
    }

    /// The store an engine's shared pool backs: where there are window lists,
    /// each list's slot refs (ref 0, in the null chunk, is never handed
    /// out), list 0's as the pages and each other's numbered past the one
    /// before, every list trimmed by its own window (CONTRACT-BC 19, 21);
    /// else the pool's fixed records as state slots
    /// and the rs list's slot refs as buffer pages (ids from past the
    /// records', an engine id the slot ref; CONTRACT-BC 18). A pool with no
    /// rs list buffers nothing.
    pub fn shared(shared: &Shared) -> Self {
        let (windows, fixed, rs_list) = {
            let pool = shared.lock().unwrap();
            (
                pool.windows().to_vec(),
                !pool.fixed().is_empty(),
                pool.rs_list(),
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
            let host = end.max(1);
            return Self::build(
                ShapePool::new(ShapeKind::Fixed, 0, 0, 0),
                pages,
                ShapePool::new(ShapeKind::Fixed, 0, host, 0),
                ShapePool::new(ShapeKind::List, first.slot_bytes, host, 0),
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
        let host = pages.ids_end();
        let mut store = Self::build(
            slots,
            pages,
            ShapePool::new(ShapeKind::Fixed, 0, host, 0),
            ShapePool::new(ShapeKind::List, 0, host, 0),
            records,
            Vec::new(),
            Vec::new(),
        );
        store.rs_tokens = rs_list.map(|list| list.tokens.max(1));
        store
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
            extent: None,
            rs_tokens: None,
        }
    }

    /// State starts with no memory under it and gains it as it is asked for.
    pub fn commit_on_demand(&mut self) {
        self.extent = Some(StateExtent {
            target: self.slots.capacity().max(self.pages.capacity()),
            ..StateExtent::default()
        });
        self.set_limit(0);
    }

    fn set_limit(&mut self, limit: u32) {
        self.slots.set_limit(limit);
        self.pages.set_limit(limit);
    }

    fn limit(&self) -> u32 {
        self.slots.limit().max(self.pages.limit())
    }

    /// In a shared pool, records that waiters are short `need` of each of
    /// this store's shapes (window lists each their own), so the pool's
    /// segment walker commits for them; `RsUnits::default()` clears it.
    pub fn note_short_shared(&mut self, need: RsUnits) {
        self.slots.wait(u64::from(need.slots));
        if self.is_window() {
            self.pages.wait(u64::from(need.windows[0]));
            for (at, pool) in self.window_pools.iter_mut().enumerate() {
                pool.wait(u64::from(need.windows.get(at + 1).copied().unwrap_or(0)));
            }
        } else {
            self.pages.wait(u64::from(need.pages));
        }
    }

    /// Asks for `need` more than stand free; whether committing can add any.
    pub fn note_short(&mut self, need: RsUnits) -> bool {
        let limit = self.limit();
        let Some(extent) = &mut self.extent else {
            return false;
        };
        extent.short = extent.short.max(need.slots.max(need.pages));
        let can = limit < extent.target;
        self.slots.wait(u64::from(need.slots));
        self.pages.wait(u64::from(need.pages));
        can
    }

    /// No waiter is short any more.
    fn unwait(&mut self) {
        self.slots.wait(0);
        self.pages.wait(0);
    }

    /// The numbers to commit up to next, keeping `step` free and covering
    /// what a waiter is short of; `None` while enough stand free.
    pub fn next_extent(&self, step: u32) -> Option<u32> {
        let extent = self.extent?;
        let free = self.slots.available().min(self.pages.available()) as u32;
        let lacking = step.max(extent.short).saturating_sub(free);
        let limit = self.limit();
        (lacking > 0 && limit < extent.target)
            .then(|| (limit + lacking.next_multiple_of(step.max(1))).min(extent.target))
    }

    pub fn set_committed(&mut self, limit: u32) {
        let committed = limit;
        let limit = self.extent.map_or(limit, |extent| limit.min(extent.target));
        let gained = limit.saturating_sub(self.limit());
        self.set_limit(limit);
        if let Some(extent) = &mut self.extent {
            extent.mapped = extent.mapped.max(committed);
            extent.short = extent.short.saturating_sub(gained);
            if extent.short == 0 {
                self.unwait();
            }
        }
    }

    /// A refused commit stands what is committed as the most there is.
    pub fn hold_extent(&mut self) {
        let limit = self.limit();
        if let Some(extent) = &mut self.extent {
            extent.target = limit;
            extent.short = 0;
            self.unwait();
        }
    }

    /// Moves the target; a fall stops handing out numbers past it at once.
    pub fn set_target(&mut self, target: u32) {
        let limit = self.limit();
        if let Some(extent) = &mut self.extent {
            extent.target = target;
            if target < limit {
                self.set_limit(target);
            }
        }
    }

    pub fn mapped(&self) -> u32 {
        self.extent.map_or(0, |extent| extent.mapped)
    }

    /// What stands free, with what the target allows and nothing has
    /// committed: the planner plans as over a whole pool, as it does for kv.
    pub fn free(&self) -> RsUnits {
        let room = self
            .extent
            .map_or(0, |extent| extent.target.saturating_sub(self.limit()));
        self.available().add(RsUnits::of(room, room))
    }

    pub fn note_released(&mut self, mapped: u32) {
        if let Some(extent) = &mut self.extent {
            extent.mapped = mapped;
        }
    }

    /// Where memory can stop now: past the highest number still out.
    pub fn held_end(&mut self) -> u32 {
        self.retire_idle();
        self.slots.held_end().max(self.pages.held_end())
    }

    /// What is asked for now and, of that, what checkpoints do not merely
    /// keep: as far as the highest number out, so that holding it cuts
    /// nothing off, and a waiter's on top.
    pub fn demand(&self) -> (u32, u32) {
        let extent = self.extent.unwrap_or_default();
        let reach = self.slots.high_end().max(self.pages.high_end());
        let out = |pool: &ShapePool<RsSlotId>| pool.limit() - pool.available() as u32;
        let wanted = out(&self.slots).max(out(&self.pages));
        let kept: HashSet<RsSlotId> = self
            .checkpoints
            .values()
            .filter_map(|entry| self.working_sets.get(entry.snapshot))
            .flat_map(Self::slots)
            .filter(|&id| !self.is_page(id))
            .collect();
        let live = kept.len() as u32;
        (
            reach.max(wanted).saturating_add(extent.short),
            wanted.saturating_sub(live).saturating_add(extent.short),
        )
    }

    /// Drops the checkpoints reading a number past the limit, where their
    /// key lets the runtime drop them unasked (`pressure::may_drop`).
    /// Answers how many went, and the slots past the limit those that stay
    /// read: room for them is the policy's to make.
    pub fn reclaim_tail(&mut self, keys: impl crate::policy::retention::Keys) -> (usize, u32) {
        // A shared pool moves by its segments, not by a limit: no id is
        // past one (an rs list's slot refs run over every chunk).
        if self.is_shared() {
            return (0, 0);
        }
        let limit = self.limit();
        let past = |id: RsSlotId, store: &Self| store.engine_id(id) >= limit && !store.is_host(id);
        let mut kept: HashSet<RsSlotId> = HashSet::new();
        let mut droppable: HashSet<RsWorkingSetId> = HashSet::new();
        for entry in self.checkpoints.values() {
            let Some(ws) = self.working_sets.get(entry.snapshot) else {
                continue;
            };
            let tail: Vec<RsSlotId> = Self::slots(ws).filter(|&id| past(id, self)).collect();
            if tail.is_empty() {
                continue;
            }
            let at = u64::from(Self::position(ws)).saturating_sub(1);
            if crate::policy::pressure::may_drop(keys.key_at(entry.owner, at)) {
                droppable.insert(entry.snapshot);
            } else {
                kept.extend(tail);
            }
        }
        let dropped = self.drop_checkpoints(|_, snapshot| droppable.contains(&snapshot));
        (dropped, kept.len() as u32)
    }

    /// The state is window lists of kv rows, drawn from the pages (list 0)
    /// and the window pools.
    pub fn is_window(&self) -> bool {
        !self.window.is_empty()
    }

    /// The window lists, in their order.
    pub fn window_lists(&self) -> &[WindowGeometry] {
        &self.window
    }

    /// What is held and wanted of the device's state slots and buffer (or
    /// window) pages, in bytes.
    pub fn device_demand(&self) -> Demand {
        let mut demand = Demand::default();
        for pool in [&self.slots, &self.pages]
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
        for pool in &mut self.window_pools {
            pool.retire_through(completed);
        }
    }

    /// The window list `id` is a slot of, where the state is windows.
    fn list_of(&self, id: RsSlotId) -> Option<usize> {
        if self.window.is_empty() {
            return None;
        }
        if self.pages.owns(id) || self.host_pages.contains(id) {
            return Some(0);
        }
        self.window_pools
            .iter()
            .position(|pool| pool.owns(id))
            .map(|at| at + 1)
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
        if self.host_pages.contains(id) {
            id.0 - self.host_pages.base()
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
        (id.0 >= self.pages_at && id.0 < self.host_slots.base()) || self.host_pages.contains(id)
    }

    fn units(&self, ids: impl IntoIterator<Item = RsSlotId>) -> RsUnits {
        let mut units = RsUnits::default();
        for id in ids {
            if self.is_page(id) {
                units.pages += 1;
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
        if let Some(list) = self.list_of(id).filter(|&list| list > 0) {
            &mut self.window_pools[list - 1]
        } else if self.is_page(id) {
            &mut self.pages
        } else {
            &mut self.slots
        }
    }

    fn host_pool(&mut self, id: RsSlotId) -> &mut ShapePool<RsSlotId> {
        if self.is_page(id) {
            &mut self.host_pages
        } else {
            &mut self.host_slots
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
            self.host_pool(id).release_reserved(vec![id]);
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
        })
    }

    /// What a snapshot keeps of `ws`, on the rows it stands on now.
    pub fn export(&self, ws: RsWorkingSetId) -> Result<RsExport, RsError> {
        let entry = self.entry(ws)?;
        Ok(RsExport {
            folded: entry.folded,
            buffer: entry.buffer.clone(),
            base: entry.base,
            frozen: entry.frozen,
            windows: entry.windows.clone(),
            window: entry.window,
        })
    }

    /// Whether `id` is a host row already.
    pub fn on_host(&self, id: RsSlotId) -> bool {
        self.is_host(id)
    }

    /// Host rows for a read or a write that is none of the planner's: state
    /// slots first, then buffer pages. `None` if the host lacks either.
    pub fn take_host(&mut self, slots: usize, pages: usize) -> Option<Vec<RsSlotId>> {
        let mut rows = self.host_slots.try_alloc_n(slots)?;
        let Some(buffer) = self.host_pages.try_alloc_n(pages) else {
            self.host_slots.release_reserved(rows);
            return None;
        };
        rows.extend(buffer);
        Some(rows)
    }

    pub fn give_host(&mut self, rows: Vec<RsSlotId>) {
        self.release_host(rows);
    }

    /// A set of its own on host rows, as `export` said one stood.
    pub fn import(&mut self, geom: RsGeometry, set: RsExport) -> RsWorkingSetId {
        self.working_sets.insert(RsEntry {
            geom,
            folded: set.folded,
            buffer: set.buffer,
            base: set.base,
            frozen: set.frozen,
            windows: set.windows,
            window: set.window,
        })
    }

    pub fn fork(&mut self, ws: RsWorkingSetId) -> Result<RsWorkingSetId, RsError> {
        let entry = self.entry(ws)?.clone();
        for id in Self::slots(&entry) {
            *self.refs.entry(id).or_insert(1) += 1;
        }
        Ok(self.working_sets.insert(entry))
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
    /// folded state.
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
                None => usize::from(self.is_page(id)),
            };
            by_pool.entry(at).or_default().push(id);
        }
        let holder = Holder::CacheEntry(key);
        for (at, ids) in by_pool {
            let pool = match at {
                0 => &mut self.slots,
                at => self.list_pool(at - 1),
            };
            if hold {
                pool.share(holder, &ids);
            } else {
                pool.unshare(holder, &ids);
            }
        }
    }

    /// The device pool `id` is one of.
    fn pool_of(&self, id: RsSlotId) -> &ShapePool<RsSlotId> {
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
        if let Some(entry) = self.checkpoints.get_mut(key) {
            entry.last_use = Instant::now();
        }
        Ok(true)
    }

    pub fn forget_state(&mut self, ws: RsWorkingSetId) -> Result<(), RsError> {
        let (folded, buffer, windows) = {
            let entry = self.entry_mut(ws)?;
            entry.base = 0;
            entry.frozen = 0;
            entry.window = entry.window.map(|_| WindowState::default());
            let lists = entry.windows.len();
            (
                entry.folded.take(),
                std::mem::take(&mut entry.buffer),
                std::mem::replace(&mut entry.windows, vec![Vec::new(); lists]),
            )
        };
        let windows = windows.into_iter().flatten();
        for id in folded
            .into_iter()
            .chain(buffer.into_iter().flatten())
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
        let reach = if advance.positioned {
            window.committed
        } else {
            advance.reach
        };
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
    pub fn group_demand(
        &self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        writes: Option<std::ops::Range<u32>>,
    ) -> Result<RsUnits, RsError> {
        let entry = self.entry(ws)?;
        let targets = self.buffer_targets(entry, writes)?;
        let state = write_state && (fresh || entry.folded.is_none_or(|id| self.ref_count(id) > 1));
        Ok(RsUnits {
            slots: u32::from(state),
            pages: self.targets_demand(&targets),
            ..RsUnits::default()
        })
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
        let base = self.buffer_base(entry);
        let first = (writes.start - base) / page;
        let last = (writes.end - 1 - base) / page;
        Ok((first..=last)
            .map(|index| (index, entry.buffer.get(index as usize).copied().flatten()))
            .collect())
    }

    pub fn prepare_group(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        fold: Option<u32>,
        writes: Option<std::ops::Range<u32>>,
        reserved: Option<&mut RsGrant>,
    ) -> Result<RsPreparedWrite, RsError> {
        // Refused (below the state) before the buffer grows.
        self.buffer_targets(self.entry(ws)?, writes.clone())?;
        if let Some(writes) = writes.as_ref().filter(|writes| !writes.is_empty()) {
            let (base, page) = {
                let entry = self.entry(ws)?;
                (self.buffer_base(entry), self.page_tokens(entry))
            };
            let entry = self.entry_mut(ws)?;
            entry.base = base;
            let pages = writes.end.saturating_sub(entry.base).div_ceil(page) as usize;
            if entry.buffer.len() < pages {
                entry.buffer.resize(pages, None);
            }
        }
        let targets = self.buffer_targets(self.entry(ws)?, writes)?;
        self.prepare_targets(ws, write_state, fresh, fold, targets, reserved)
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
            pages: self.targets_demand(&buffer_targets_src),
            ..RsUnits::default()
        };
        let (slots, pages) = match reserved {
            Some(granted) => {
                if !granted.units().covers(need) {
                    return Err(RsError::GrantMismatch {
                        required: need,
                        granted: granted.units(),
                    });
                }
                (
                    granted.slots.drain(..need.slots as usize).collect(),
                    granted.pages.drain(..need.pages as usize).collect(),
                )
            }
            None => self.alloc(need).ok_or(RsError::OutOfSlots {
                requested: need,
                available: self.available(),
            })?,
        };
        let mut fresh_slots = slots.iter().copied();
        let mut fresh_ids = pages.iter().copied();
        let allocated: Vec<RsSlotId> = slots.iter().chain(&pages).copied().collect();

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

        let buffers: Vec<RsBufferTarget> = buffer_targets_src
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
        for RsPendingFold { ws, tokens } in folds.0 {
            self.fold(ws, tokens);
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
        let entry = self.entry_mut(ws).expect("batch prevalidated");
        let table = match list {
            Some(list) => &mut entry.windows[list],
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
        let Ok(entry) = self.entry_mut(ws) else {
            return;
        };
        entry.frozen = entry.frozen.saturating_add(slots);
        let below = (entry.frozen.saturating_sub(entry.base) / page) as usize;
        let drop = below.min(entry.buffer.len());
        entry.base += drop as u32 * page;
        let dropped: Vec<RsSlotId> = entry.buffer.drain(..drop).flatten().collect();
        if entry.buffer.is_empty() {
            entry.base = match rs_tokens {
                Some(tokens) => entry.frozen - entry.frozen % tokens,
                None => entry.frozen,
            };
        }
        for id in dropped {
            self.decref(id);
        }
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
        RsUnits::of(self.slots.available() as u32, self.pages.available() as u32)
    }

    pub fn capacity(&self) -> RsUnits {
        if self.is_window() {
            return self.per_list(ShapePool::capacity);
        }
        RsUnits::of(self.slots.capacity(), self.pages.capacity())
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
        self.host_units(
            self.host_slots.available() as u32,
            self.host_pages.available() as u32,
        )
    }

    pub fn host_capacity(&self) -> RsUnits {
        self.host_units(self.host_slots.capacity(), self.host_pages.capacity())
    }

    fn host_units(&self, slots: u32, pages: u32) -> RsUnits {
        let mut units = RsUnits::of(slots, pages);
        if self.is_window() {
            units.windows[0] = pages;
        }
        units
    }

    /// Parks the sets' private, settled slots in host rows, when rows for
    /// all of them are free, and the checkpoints sharing them with them;
    /// otherwise they stay on the device, nothing is dropped, and `None`
    /// says so ([`Self::suspend_short`] tells what the host lacks).
    pub fn prepare_suspend(
        &mut self,
        working_sets: &HashSet<RsWorkingSetId>,
    ) -> Option<RsResidencyTxn> {
        let busy = self.busy_slots();
        let slots: Vec<RsSlotId> = self
            .private(working_sets)
            .into_iter()
            .filter(|id| !busy.contains(id))
            .collect();
        if slots.is_empty() {
            return None;
        }
        let (pages, slots): (Vec<RsSlotId>, Vec<RsSlotId>) =
            slots.into_iter().partition(|&id| self.is_page(id));
        let host_slots = self.host_slots.try_alloc_n(slots.len())?;
        let Some(host_pages) = self.host_pages.try_alloc_n(pages.len()) else {
            self.host_slots.release_reserved(host_slots);
            return None;
        };
        let slots: Vec<RsSlotId> = slots.into_iter().chain(pages).collect();
        let host: Vec<RsSlotId> = host_slots.into_iter().chain(host_pages).collect();
        self.moving.extend(slots.iter().copied());
        let scope = self.with_sharing(working_sets);
        Some(self.txn(&scope, slots.into_iter().zip(host).collect()))
    }

    /// The host rows a suspend of `working_sets` would take beyond those
    /// free: what keeps its state on the device. Zero when it fits.
    pub fn suspend_short(&self, working_sets: &HashSet<RsWorkingSetId>) -> RsUnits {
        let busy = self.busy_slots();
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
        let mut plan = RsCopyPlan::default();
        for &(from, to) in &moves {
            let (device, host) = if self.is_host(from) {
                (to, from)
            } else {
                (from, to)
            };
            let pair = (self.engine_id(device), self.engine_id(host));
            if self.is_page(device) {
                plan.pages.push(pair);
            } else {
                plan.slots.push(pair);
            }
        }
        RsResidencyTxn {
            scope: working_sets.iter().copied().collect(),
            moves,
            plan,
            window: self.is_window(),
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
        let mut slots = granted.slots.drain(..need.slots as usize);
        // A windowed store parks list 0 only: its pages come back as it.
        let pages = match granted.windows.first_mut() {
            Some(list) if self.window.len() == 1 => list,
            _ => &mut granted.pages,
        };
        let mut pages = pages.drain(..need.pages as usize);
        self.moving.extend(swapped.iter().copied());
        let scope = self.with_sharing(working_sets);
        let moves = swapped
            .into_iter()
            .map(|host| {
                let device = if self.is_page(host) {
                    pages.next()
                } else {
                    slots.next()
                };
                (host, device.expect("the grant covers what was swapped"))
            })
            .collect();
        drop((slots, pages));
        Ok(self.txn(&scope, moves))
    }

    /// Swaps each host slot back for its device copy, returning the slots restored.
    pub fn commit_restore(&mut self, txn: RsResidencyTxn) -> usize {
        let scope: HashSet<RsWorkingSetId> = txn.scope.iter().copied().collect();
        self.cache_holds_in(&scope, false);
        let mut restored = 0;
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
        self.release_device(txn.moves.into_iter().map(|(_, device)| device));
    }

    fn alloc(&mut self, need: RsUnits) -> Option<(Vec<RsSlotId>, Vec<RsSlotId>)> {
        let slots = self.slots.try_alloc_n(need.slots as usize)?;
        let Some(pages) = self.pages.try_alloc_n(need.pages as usize) else {
            self.slots.release_reserved(slots);
            return None;
        };
        Some((slots, pages))
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
        self.alloc(need).map(|(slots, pages)| RsGrant {
            slots,
            pages,
            windows: Vec::new(),
        })
    }

    pub fn release_reservation(&mut self, grant: RsGrant) {
        self.slots.release_reserved(grant.slots);
        self.pages.release_reserved(grant.pages);
        for (list, ids) in grant.windows.into_iter().enumerate() {
            self.list_pool(list).release_reserved(ids);
        }
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
        true
    }
}

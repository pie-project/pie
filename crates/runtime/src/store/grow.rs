//! How an engine's slot pools grow and shrink: the governor's
//! [`PoolTarget`].
//!
//! An elastic pool hands out only the ids of committed segments (see
//! [`super::chunks::ShapePool`]), and commits ahead of them here, off the
//! frame path: a grant never waits on a mapping unless it outruns this task,
//! no frame ever finds its pages unbacked, and a commit the device refuses
//! costs a waiter an eviction instead of a frame.
//!
//! The governor's byte target is split over the shapes by what each is
//! asked, and each shape's pool stops handing out ids past its share. In
//! the degenerate mode nothing can be evacuated: a segment past the target
//! goes once its holders let it go, and until then the pool reports it held.
//!
//! Memory the target allows but nothing has committed reads as free to the
//! planner, so it plans as over a whole pool and evicts only past the target.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::Notify;

use crate::governor::{Demand, Growth, PoolTarget, Retarget};
use crate::scheduler::worker::PoolOp;
use crate::store::kv::KvStore;
use crate::store::registry;

/// What a pool keeps committed ahead of what it has handed out, in bytes:
/// a mapping's cost on the lane against how often the lane pays it.
pub const COMMIT_AHEAD_BYTES: u64 = 64 * 1024 * 1024;

/// The state slots kept committed ahead of those handed out: what a burst of
/// lanes and their snapshots takes, so it does not stop the lane to map them
/// a few at a time.
const STATE_AHEAD: u32 = 16;

/// A pool that moves by copying grows by this part of itself at a time.
const WHOLE_STEPS: u32 = 8;

/// How often a shrink looks at the tail while its holders move out.
const VACATE_POLL: Duration = Duration::from_millis(50);

#[derive(Clone)]
pub struct Grow {
    model: usize,
    engine: usize,
    wake: Arc<Notify>,
    ask: Arc<Notify>,
    can_rise: Arc<AtomicBool>,
    refused: Arc<AtomicBool>,
    step: u32,
    page_bytes: u64,
    whole: bool,
    slot_bytes: u64,
    state_step: u32,
    /// What the governor's last turn read, and the split it last set.
    turn: Arc<Mutex<(Option<Asked>, Option<Split>)>>,
}

/// Where a byte target puts each pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Split {
    kv_pages: u32,
    state_slots: u32,
}

/// What is asked of the pools now, in bytes, and of that the state's.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Asked {
    pub wanted: u64,
    pub live: u64,
    state: u64,
}

impl Grow {
    /// The extent of `engine`'s pool of pages of `page_bytes`: from nothing,
    /// or `whole` when memory only moves by copying the pool, which is worth
    /// doing only when the system asks. `slot_bytes` is what a state slot
    /// takes where it gains memory as it is handed out, else zero.
    pub fn new(model: usize, engine: usize, page_bytes: u64, whole: bool, slot_bytes: u64) -> Self {
        let stores = registry::get(model, engine);
        let pages = registry::with_kv_lock(&stores.kv, "grow", |kv| {
            if !whole {
                kv.commit_on_demand();
            }
            kv.capacity_pages()
        });
        if slot_bytes > 0 {
            stores.rs.lock().unwrap().commit_on_demand();
        }
        let ahead = if whole {
            pages / WHOLE_STEPS
        } else {
            u32::try_from(COMMIT_AHEAD_BYTES / page_bytes.max(1)).unwrap_or(u32::MAX)
        };
        Self {
            model,
            engine,
            wake: Arc::new(Notify::new()),
            ask: Arc::new(Notify::new()),
            can_rise: Arc::new(AtomicBool::new(false)),
            refused: Arc::new(AtomicBool::new(false)),
            step: ahead.max(1),
            page_bytes,
            whole,
            slot_bytes,
            state_step: STATE_AHEAD,
            turn: Arc::new(Mutex::new((None, None))),
        }
    }

    /// Starts committing; the scheduler that takes the commits is up.
    pub fn start(&self) {
        crate::rt::spawn(self.clone().run());
        self.wake();
    }

    pub(crate) fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    /// What is asked for now and, of that, what no idle cache merely keeps;
    /// each with the step kept in hand ahead of it.
    pub(crate) fn asked(&self) -> Asked {
        // The page pool's demand: what is held, and a waiter's shortfall on
        // top; never below the highest page held, which no limit cuts off.
        let (held, wanted, reach, kept) = self.kv(|kv| {
            let pool = kv.page_pool();
            let slots = |bytes: u64| bytes / pool.slot_bytes().max(1);
            let demand = pool.demand();
            (
                slots(demand.live),
                slots(demand.wanted),
                pool.high_end(),
                kv.reclaimable_cache_pages(),
            )
        });
        let (held, wanted) = (
            u32::try_from(held).unwrap_or(u32::MAX),
            u32::try_from(wanted).unwrap_or(u32::MAX),
        );
        let pages =
            |pages: u32| u64::from(pages.saturating_add(self.step)).saturating_mul(self.page_bytes);
        let (slots, in_use) = self.rs(|rs| rs.demand());
        let state = |slots: u32| {
            if self.slot_bytes == 0 {
                return 0;
            }
            u64::from(slots.saturating_add(self.state_step)).saturating_mul(self.slot_bytes)
        };
        Asked {
            wanted: pages(wanted.max(reach)).saturating_add(state(slots)),
            live: pages(held.saturating_sub(kept)).saturating_add(state(in_use)),
            state: state(slots),
        }
    }

    /// The whole of what the pools can take, in bytes.
    pub(crate) fn hard_limit(&self) -> u64 {
        let pages = u64::from(self.kv(|kv| kv.capacity_pages()));
        pages
            .saturating_mul(self.page_bytes)
            .saturating_add(self.state_bytes(self.rs(|rs| rs.capacity().slots)))
    }

    /// What one lane cannot run under, in bytes: one context's pages and its
    /// state beside them.
    pub(crate) fn floor(&self) -> u64 {
        self.context_pages()
            .saturating_mul(self.page_bytes)
            .saturating_add(self.state_bytes(2))
    }

    /// What the pools hold now, in bytes.
    pub(crate) fn held(&self) -> u64 {
        u64::from(self.kv(|kv| kv.committed_pages()))
            .saturating_mul(self.page_bytes)
            .saturating_add(self.state_bytes(self.rs(|rs| rs.mapped())))
    }

    fn state_bytes(&self, slots: u32) -> u64 {
        u64::from(slots).saturating_mul(self.slot_bytes)
    }

    /// What `target` bytes come to in each pool: each takes its share of
    /// what is asked, and what the target leaves over stands as kv room.
    pub(crate) fn split(&self, target: u64, asked: Asked) -> Split {
        let share = u128::from(asked.state) * u128::from(target.min(asked.wanted))
            / u128::from(asked.wanted.max(1));
        let state_slots =
            u32::try_from(share.div_ceil(u128::from(self.slot_bytes.max(1)))).unwrap_or(u32::MAX);
        let kv = target.saturating_sub(self.state_bytes(state_slots));
        Split {
            kv_pages: u32::try_from(kv / self.page_bytes.max(1)).unwrap_or(u32::MAX),
            state_slots,
        }
    }

    pub(crate) const fn state_step(&self) -> u32 {
        self.state_step
    }

    pub(crate) const fn state_on_demand(&self) -> bool {
        self.slot_bytes > 0
    }

    pub(crate) fn context_pages(&self) -> u64 {
        registry::get(self.model, self.engine).context_pages
    }

    pub(crate) fn step(&self) -> u32 {
        self.step
    }

    /// Whether memory moves by copying the pool: a cut then goes all the way
    /// at once, since each stop on the way would be a copy of what remains.
    pub(crate) const fn moves_by_copy(&self) -> bool {
        self.whole
    }

    pub(crate) fn wake(&self) {
        self.wake.notify_one();
    }

    /// Asks the Governor for room past the target; whether it can give any.
    /// With no Governor, or one held down, the answer is an eviction.
    pub(crate) fn ask(&self) -> bool {
        let can = self.can_rise.load(Ordering::Acquire);
        if can {
            self.ask.notify_one();
        }
        can
    }

    pub(crate) async fn waiter(&self) {
        self.ask.notified().await;
    }

    /// A waiter told to wait for a rise that will not come goes back to the
    /// planner, which funds it some other way.
    pub(crate) fn set_can_rise(&self, can: bool) {
        if self.can_rise.swap(can, Ordering::AcqRel) && !can {
            self.note_supply();
        }
    }

    /// Moves each pool to its share: a rise commits as memory is asked for,
    /// a fall empties the tail and gives its memory back.
    pub(crate) fn retarget(&self, split: Split) {
        self.kv(|kv| kv.set_target(split.kv_pages));
        if self.state_on_demand() {
            self.rs(|rs| rs.set_target(split.state_slots));
        }
        self.wake();
    }

    /// Waves stay narrow while the system is short of memory, and what wider
    /// ones left of the scratch goes back as that starts.
    pub(crate) async fn yield_scratch(&self, yielding: bool) {
        crate::scheduler::narrow(self.engine, yielding);
        if yielding {
            self.release_scratch().await;
        }
    }

    /// What wider waves left of the scratch goes back; the next one takes
    /// what it needs again.
    pub(crate) async fn release_scratch(&self) {
        if let Err(error) = crate::scheduler::pool(self.engine, PoolOp::ReleaseScratch).await {
            tracing::warn!(%error, "scratch release refused");
        }
    }

    async fn run(self) {
        loop {
            self.wake.notified().await;
            self.vacate().await;
        }
    }

    fn kv<T>(&self, read: impl FnOnce(&mut KvStore) -> T) -> T {
        let stores = registry::get(self.model, self.engine);
        registry::with_kv_lock(&stores.kv, "grow", read)
    }

    fn rs<T>(&self, read: impl FnOnce(&mut crate::store::rs::RsStore) -> T) -> T {
        let stores = registry::get(self.model, self.engine);
        let mut store = stores.rs.lock().unwrap();
        read(&mut store)
    }

    fn note_supply(&self) {
        if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
            planner.pages_freed();
        }
    }

    async fn commit(&self) {
        while let Some(pages) = self.kv(|kv| kv.next_extent(self.step)) {
            // Memory a shrink has not given back yet has only to be taken up.
            if pages <= self.kv(|kv| kv.mapped_pages()) {
                self.kv(|kv| kv.set_committed(pages));
                self.note_supply();
                continue;
            }
            let committed = crate::scheduler::pool(self.engine, PoolOp::Commit(pages)).await;
            match committed {
                Ok(()) => self.kv(|kv| kv.set_committed(pages)),
                Err(error) => {
                    self.commit_failed(&error);
                    self.kv(KvStore::hold_extent);
                }
            }
            self.note_supply();
        }
        while let Some(slots) = self.rs(|rs| rs.next_extent(self.state_step)) {
            if slots > self.rs(|rs| rs.mapped()) {
                let committed =
                    crate::scheduler::pool(self.engine, PoolOp::CommitState(slots)).await;
                if let Err(error) = committed {
                    self.commit_failed(&error);
                    self.rs(|rs| rs.hold_extent());
                    self.note_supply();
                    continue;
                }
            }
            self.rs(|rs| rs.set_committed(slots));
            self.note_supply();
        }
    }

    /// What is committed stands as all there is. Memory the system would not
    /// back is the system pushing back, and the Governor answers it as such;
    /// anything else means the pool's sums are wrong or the device is gone,
    /// and no later attempt mends either.
    fn commit_failed(&self, error: &anyhow::Error) {
        if let Some(::engine::Error::Exhausted { .. }) = error.downcast_ref() {
            tracing::warn!(%error, "the system refused memory");
            self.refused.store(true, Ordering::Release);
        } else {
            tracing::error!(%error, "a pool commit failed");
        }
    }

    /// Whether a commit met a refusal since this last asked.
    pub(crate) fn take_refused(&self) -> bool {
        self.refused.swap(false, Ordering::AcqRel)
    }

    /// Hands back the state past the highest slot out. Those a running
    /// process holds go when it ends: nothing here moves them. What idle
    /// cache keeps past the limits the planner lets go or asks the policy
    /// for ([`crate::planner::ResidencyPlanner::reclaim_tail`]).
    async fn release_state(&self) {
        if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
            planner.reclaim_tail();
        }
        let (end, mapped) = self.rs(|rs| (rs.held_end(), rs.mapped()));
        if mapped > end && mapped - end >= self.state_step {
            match crate::scheduler::pool(self.engine, PoolOp::ReleaseState(end)).await {
                Ok(()) => self.rs(|rs| rs.note_released(end)),
                Err(error) => tracing::warn!(%error, end, "state release refused"),
            }
        }
    }

    /// Hands back the memory above the highest page out as the tail empties,
    /// committing as it goes: a tail waits on its holders, and a target that
    /// rises meanwhile takes what is mapped back and ends the wait.
    async fn vacate(&self) {
        loop {
            self.commit().await;
            self.release_state().await;
            let tail = self.kv(|kv| {
                (kv.mapped_pages() > kv.committed_pages()).then(|| {
                    (
                        kv.committed_pages(),
                        kv.tail_held(),
                        kv.held_end(),
                        kv.mapped_pages(),
                    )
                })
            });
            let Some((limit, held, end, mapped)) = tail else {
                return;
            };
            tracing::debug!(limit, held, end, mapped, "kv tail");
            // A copy holds both slabs for a moment: one that keeps more than
            // it frees costs the system more than it gives back.
            let worth = if self.whole {
                mapped - end >= end
            } else {
                held == 0 || mapped - end >= self.step
            };
            if end < mapped && worth {
                match crate::scheduler::pool(self.engine, PoolOp::Release(end)).await {
                    Ok(()) => self.kv(|kv| kv.note_released(end)),
                    Err(error) => tracing::warn!(%error, end, "pool release refused"),
                }
            }
            if held == 0 && end == limit {
                return;
            }
            if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
                planner.vacate_tail();
            }
            // The holders move on their own tasks; this only looks again.
            let _ = crate::rt::time::timeout(VACATE_POLL, self.wake.notified()).await;
        }
    }
}

/// Today's pools as one target: kv pages and state slots split a byte
/// target by what each is asked, so a segment here is one byte. Their tails
/// empty on the grow task; a retarget only moves the targets.
impl PoolTarget for Grow {
    fn demand(&mut self) -> Demand {
        let asked = self.asked();
        self.turn.lock().unwrap().0 = Some(asked);
        Demand {
            wanted: asked.wanted,
            live: asked.live,
        }
    }

    fn segment_bytes(&self) -> u64 {
        1
    }

    fn floor(&self) -> u64 {
        // One context's pages and its state, so one lane can always run.
        Grow::floor(self)
    }

    fn hard_limit(&self) -> u64 {
        Grow::hard_limit(self)
    }

    fn held(&self) -> u64 {
        Grow::held(self)
    }

    fn growth(&self) -> Growth {
        Growth::Elastic {
            least_cut: if self.moves_by_copy() {
                u64::MAX
            } else {
                u64::from(self.step()).saturating_mul(self.page_bytes())
            },
        }
    }

    fn retarget(&mut self, segments: u64) -> Retarget {
        let asked = self.turn.lock().unwrap().0;
        let asked = asked.unwrap_or_else(|| self.asked());
        let split = self.split(segments, asked);
        let changed = {
            let mut turn = self.turn.lock().unwrap();
            let changed = turn.1 != Some(split);
            turn.1 = Some(split);
            changed
        };
        if changed {
            tracing::debug!(target = segments, ?split, "pool target");
            Grow::retarget(self, split);
        }
        Retarget::Reached
    }

    fn take_refused(&mut self) -> bool {
        Grow::take_refused(self)
    }

    /// Narrowing rides the yield: a critical tick yields, and the loop
    /// narrows the waves as it does.
    fn shrink_rung(&mut self) {}

    fn restore_rung(&mut self) {}
}

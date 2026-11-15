//! The governor's [`PoolTarget`] for an engine whose stores draw from a
//! [`SharedPool`]: the target in segments, moved toward off the fire path
//! by a task that speaks to the engine through its scheduler lane, so each
//! commit, evacuation and release is ordered with the fires around it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use tokio::sync::Notify;

use super::{RungStep, RungWalk, SegmentMove, Shared};
use crate::governor::{Demand, Growth, PATIENCE, PoolTarget, Retarget};
use crate::scheduler::worker::PoolOp;
use crate::store::registry;

#[derive(Clone)]
pub struct SharedGrow {
    model: usize,
    engine: usize,
    shared: Shared,
    /// The governor's target, in mapped segments.
    target: Arc<AtomicU32>,
    wake: Arc<Notify>,
    refused: Arc<AtomicBool>,
    /// The last shrink found no room.
    no_room: Arc<AtomicBool>,
    /// The walk of the activation rung toward what the passes still to run
    /// ask of a fire (SPEC decision 6, by demand): the smallest rung while
    /// none asks past it.
    rung_walk: Arc<std::sync::Mutex<RungWalk>>,
    /// The governor's ticks, counted ([`PoolTarget::ticked`]).
    ticks: Arc<AtomicU64>,
    /// The lanes live now that may decode ([`Self::walk_rows`]).
    live_lanes: Arc<dyn Fn() -> u32 + Send + Sync>,
    /// The rows asked within the governor's patience ([`Stood`]): a cap
    /// down or a lower goes no further than the rung they take.
    stood: Arc<std::sync::Mutex<Stood>>,
    /// The last raise found no room: none is tried again until room is
    /// freed ([`Self::room_freed`]), the one event that can change that.
    raise_blocked: Arc<AtomicBool>,
}

/// The rows asked of an engine, kept for the governor's patience
/// ([`PATIENCE`] ticks): the walk brings fires' cap and the rung down only
/// to the rung of the most rows asked within it. So the patience before a
/// cut to a rung runs from the last moment an ask stood above that rung,
/// for each rung, not from when the pool first stood past what was asked:
/// asks that come and go (a decode fire's, dropped as it settles) hold the
/// rung they ask while they keep coming, and cut nothing under themselves.
#[derive(Debug)]
struct Stood {
    /// The rows asked at the last look: they stood until some moment since.
    /// Before the first look, every row: the rung the pool was loaded at
    /// stands as asked until the walk first looks, so it is held for the
    /// patience from there, as one an ask left is.
    last: u32,
    /// The tick and rows of looks within the patience, the rows descending:
    /// each the most asked from its tick on.
    within: std::collections::VecDeque<(u64, u32)>,
}

impl Default for Stood {
    fn default() -> Self {
        Self {
            last: u32::MAX,
            within: std::collections::VecDeque::new(),
        }
    }
}

impl Stood {
    /// A look at `tick` finding `rows` asked, `peak` the most asked since
    /// the last look: the most rows that stood within the patience. What
    /// stood at the last look counts as standing until this one, so the
    /// patience is never shorter than [`PATIENCE`] ticks from an ask's end.
    fn most(&mut self, tick: u64, rows: u32, peak: u32) -> u32 {
        let stood = rows.max(peak).max(self.last);
        self.last = rows;
        while self.within.back().is_some_and(|&(_, back)| back <= stood) {
            self.within.pop_back();
        }
        self.within.push_back((tick, stood));
        while self
            .within
            .front()
            .is_some_and(|&(at, _)| at.saturating_add(u64::from(PATIENCE)) <= tick)
        {
            self.within.pop_front();
        }
        self.within.front().map_or(rows, |&(_, most)| most)
    }
}

impl SharedGrow {
    /// Steers `shared`, engine `engine`'s, from the segments mapped at load.
    #[must_use]
    pub fn new(model: usize, engine: usize, shared: Shared) -> Self {
        let (mapped, start_cap) = {
            let pool = shared.lock().unwrap();
            (pool.mapped_segments(), pool.start_cap())
        };
        // Fires start cut to the rung the pool starts at: on a span backed
        // on demand, the smallest, until a raise reaches the engine.
        super::set_fire_cap(engine, start_cap);
        Self {
            model,
            engine,
            shared,
            target: Arc::new(AtomicU32::new(mapped)),
            wake: Arc::new(Notify::new()),
            refused: Arc::new(AtomicBool::new(false)),
            no_room: Arc::new(AtomicBool::new(false)),
            rung_walk: Arc::new(std::sync::Mutex::new(RungWalk::default())),
            ticks: Arc::new(AtomicU64::new(0)),
            stood: Arc::new(std::sync::Mutex::new(Stood::default())),
            live_lanes: Arc::new(move || super::live_lanes(engine)),
            raise_blocked: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A governor tick; for a test.
    #[cfg(test)]
    pub(crate) fn ticked_for_test(&self) {
        self.ticks.fetch_add(1, Ordering::AcqRel);
    }

    /// The lanes live now, in place of the engine's ([`super::live_lanes`]);
    /// for a test.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_live_lanes(
        mut self,
        live: impl Fn() -> u32 + Send + Sync + 'static,
    ) -> Self {
        self.live_lanes = Arc::new(live);
        self
    }

    /// The rows the walk holds the rung to ([`super::asked_rows`]): what
    /// queued and flying fires ask ([`super::wanted_rows`]), and never
    /// fewer than the row each live lane stages at its next step
    /// ([`super::live_lanes`]: the seats of working sets not waiting for
    /// room). Lanes, not processes: one process decoding 64 sequences
    /// holds the rung of 64 between its steps, so its decode fires never
    /// wait on a raise; a process waiting for room holds none, so the span
    /// its lanes would fire in can be lent to the state it waits for, and
    /// its pass asks again once served. Not the engine's whole lane count:
    /// a pool with one live lane rests at a small rung, and one with none
    /// at the smallest.
    pub(crate) fn walk_rows(&self) -> u32 {
        super::wanted_rows(self.engine).max((self.live_lanes)())
    }

    /// Steps the activation rung toward the rung the rows asked take
    /// ([`Self::walk_rows`]), the smallest while none is asked past it,
    /// until there or until a step has to wait: caps fires, lowers the
    /// pool's rung once no wider fire is in flight, or raises it, the
    /// engine first copying the state out of the larger span. A raise short
    /// of room waits on the room the asking pass has the planner free; one
    /// to a rung the engine has not armed waits on its capture
    /// ([`super::rung_goal`]), the walk looking again each poll.
    ///
    /// The walk's own goal is that rung cut to the widest that can be
    /// seated now ([`super::rung_seatable`]): it raises as far as fits and
    /// never waits on a raise that does not; freed or committed room wakes
    /// it and the cut rises.
    pub(crate) async fn walk_rung(&self) {
        loop {
            let (current, retired) = self.epochs();
            let (step, goal) = {
                let mut pool = self.shared.lock().unwrap();
                pool.pool_mut().retire_through(retired);
                // A raise's copy is still the engine's to take: nothing
                // moves until it has.
                if pool.copying().is_some() {
                    return;
                }
                // What was asked since the last look first: an ask made
                // from here on is the next look's.
                let peak = super::asked_peak(self.engine);
                let rows = self.walk_rows();
                let asked = super::rung_seatable(&pool, self.engine, rows);
                // The pool stands past what is asked: it comes down (a cap
                // down, a lower) only to the rung of the most rows asked
                // within the governor's patience, of those it holds now
                // (nothing is raised for an ask that is gone). So a pause
                // between batches moves no rung, and the patience before a
                // cut to a rung runs from the last ask above it ([`Stood`]).
                // On a span backed on demand an idle pool always comes down
                // after it: the carve held costs resident memory there.
                // Fires wider than the widest rung that fits with the
                // programs' fire terms are cut at once whatever was asked
                // (the step's own first move): the span holds no wider fire.
                let tick = self.ticks.load(Ordering::Acquire);
                let most = self.stood.lock().unwrap().most(tick, rows, peak);
                let goal = asked.max(pool.standing(super::rung_goal(&pool, self.engine, most)));
                let step = self.rung_walk.lock().unwrap().step(
                    &pool,
                    goal,
                    super::fire_cap(self.engine),
                    super::rung_granted(self.engine),
                    current,
                    retired,
                );
                // A raise is taken in the hold it was planned in: no slot is
                // handed out between, and the engine's copy follows.
                let blocked = self.raise_blocked.load(Ordering::Acquire);
                let step = match step {
                    // A raise refused for room is not tried again until
                    // room is freed.
                    Ok(RungStep::Raise { .. }) | Err(_) if blocked => Ok(RungStep::Wait),
                    Ok(RungStep::Raise { rung, evacuation }) => pool
                        .raise_applied(&evacuation)
                        .map(|()| RungStep::Raise { rung, evacuation }),
                    step => step,
                };
                (step, goal)
            };
            let step = match step {
                Ok(step) => step,
                Err(error) => {
                    // Not for want of room (the goal is cut to what fits):
                    // the pool refused the step itself. The walk tries
                    // again once room is freed, not on its next wake.
                    self.raise_blocked.store(true, Ordering::Release);
                    tracing::debug!(engine = self.engine, %error, goal, "the activation rung waits");
                    return;
                }
            };

            match step {
                RungStep::Wait => {
                    // A lower waits on the scheduler's grant, which it
                    // publishes each turn: an idle scheduler takes none
                    // until woken.
                    if super::rung_granted(self.engine) > super::fire_cap(self.engine) {
                        crate::scheduler::nudge(self.engine);
                    }
                    return;
                }
                RungStep::Cap(tokens) => {
                    {
                        // Under the pool's lock, as a pass's `raise_now`
                        // lifts it: a cap down past what a pass asks since
                        // is not taken.
                        let pool = self.shared.lock().unwrap();
                        let now = super::fire_cap(self.engine);
                        if tokens < now && self.walk_rows().min(pool.fit_cap()) > tokens {
                            continue;
                        }
                        super::move_fire_cap(&pool, self.engine, tokens);
                    }
                    crate::scheduler::nudge(self.engine);
                }
                RungStep::Lower(rung) => {
                    let (lowered, pages) = {
                        let mut pool = self.shared.lock().unwrap();
                        // A pass lifted the cap since the walk capped it.
                        if super::fire_cap(self.engine) > rung {
                            continue;
                        }
                        let lowered = pool.lower_rung(rung);
                        (lowered, pool.pages_at_rung())
                    };
                    let notice = match lowered {
                        Ok(notice) => notice,
                        Err(error) => {
                            tracing::warn!(engine = self.engine, %error, "the activation rung stays");
                            return;
                        }
                    };
                    tracing::info!(
                        engine = self.engine,
                        rung,
                        pages,
                        "the activation rung is lowered"
                    );
                    // Past the retire fence: the engine may give back the
                    // carve above the new rung. A failed notice is no
                    // fault.
                    if let Some((from, to)) = notice
                        && let Err(error) =
                            crate::scheduler::pool(self.engine, PoolOp::Rung { from, to }).await
                    {
                        tracing::warn!(engine = self.engine, %error, from, to, "a lower's notice was refused");
                    }
                    if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
                        planner.pages_freed();
                    }
                }
                RungStep::Raise { rung, evacuation } => {
                    // The engine backs its carve (on demand) and copies what
                    // the raise moves; refused, the raise is taken back,
                    // fires stay cut to the rung in hand, and no raise is
                    // tried again until room is freed.
                    let mover = super::programs::LaneMover {
                        engine: self.engine,
                        epoch_now: || self.epochs().0,
                    };
                    if let Err(refused) =
                        super::programs::carry_raise(&self.shared, &evacuation, &mover).await
                    {
                        tracing::debug!(engine = self.engine, %refused, "the walk's raise waits for room");
                        self.raise_blocked.store(true, Ordering::Release);
                        super::rung_changed(self.engine);
                        return;
                    }
                    tracing::info!(
                        engine = self.engine,
                        rung,
                        moved = evacuation.moves.len(),
                        "the activation rung is raised"
                    );
                    super::rung_changed(self.engine);
                }
            }
        }
    }

    /// Holds the segments mapped now as the target: what the load
    /// committed for the weights is not given back before the governor
    /// first says otherwise.
    pub fn hold_mapped(&self) {
        let mapped = self.shared.lock().unwrap().mapped_segments();
        self.target.fetch_max(mapped, Ordering::AcqRel);
    }

    #[must_use]
    pub fn shared(&self) -> &Shared {
        &self.shared
    }

    /// Room was freed (a fire retired, a slot let go): a raise refused for
    /// room may be tried again, and the walk looks now.
    pub fn room_freed(&self) {
        self.raise_blocked.store(false, Ordering::Release);
        self.shared.lock().unwrap().block_span_commit(false);
        self.wake.notify_one();
    }

    /// Whether a raise was refused and none is tried until room is freed;
    /// for a test.
    #[cfg(test)]
    pub(crate) fn raise_blocked_for_test(&self) -> bool {
        self.raise_blocked.load(Ordering::Acquire)
    }

    /// The walk's last raise was refused for room; for a test.
    #[cfg(test)]
    pub(crate) fn block_raise_for_test(&self) {
        self.raise_blocked.store(true, Ordering::Release);
    }

    /// Looks again now: a waiter is short.
    pub fn wake(&self) {
        self.wake.notify_one();
    }

    /// Whether a segment below the top span is still uncommitted, under the
    /// governor's target.
    #[must_use]
    pub fn can_grow(&self) -> bool {
        let pool = self.shared.lock().unwrap();
        let top = pool.pool().top_segment();
        let next = pool.pool().prefix_segments();
        // The next segment must be one the engine binds: one at or past
        // what it holds back never commits, and a waiter told the pool
        // grows would wait on it for good.
        (next < top && pool.pool().backs(next) || pool.pool().span_to_commit().is_some())
            && pool.mapped_segments() < self.target.load(Ordering::Acquire)
    }

    /// Starts moving; the scheduler that takes the moves is up.
    pub fn start(&self) {
        crate::rt::spawn(self.clone().run());
        self.wake.notify_one();
    }

    /// The fire epochs of the engine's stores: the last handed out, and
    /// the last retired.
    fn epochs(&self) -> (u64, u64) {
        registry::try_get(self.model, self.engine).map_or((0, u64::MAX), |stores| {
            registry::with_kv_lock(&stores.kv, "shared-grow", |kv| {
                (kv.current_epoch(), kv.retired_through())
            })
        })
    }

    /// The segments to be at: the governor's target, and on the way up no
    /// further than what stands ready ahead of the waiters asks.
    /// The segments the walk moves toward now; for a test that walks the
    /// segments itself.
    #[cfg(test)]
    pub(crate) fn goal_for_test(&self) -> u32 {
        self.goal()
    }

    fn goal(&self) -> u32 {
        let pinned = pinned_segments();
        let target = pinned.unwrap_or_else(|| self.target.load(Ordering::Acquire));
        let pool = self.shared.lock().unwrap();
        let mapped = pool.mapped_segments();
        if pinned.is_some() {
            return target;
        }
        if target <= mapped {
            return target;
        }
        let short = pool
            .pool()
            .demand()
            .wanted
            .saturating_sub(pool.pool().demand().live);
        let ahead = pool.ahead_target(short);
        tracing::trace!(
            target,
            mapped,
            short,
            ahead,
            headroom = pool.pool().headroom(),
            "segment goal"
        );
        target.min(ahead)
    }

    /// Walks each time something it waits on happens: a retarget or a
    /// governor tick, a waiter short of room, room freed or a fire retired
    /// (the planner's `pages_freed`: what drains and fences wait on), an ask
    /// for a rung the call could not take ([`super::ask_walk`]), never on a
    /// clock of its own.
    async fn run(self) {
        loop {
            tokio::select! {
                () = self.wake.notified() => {}
                () = super::walk_asked(self.engine) => {}
            }
            self.walk().await;
        }
    }

    /// Steps toward the goal until there, or until a move has to wait.
    async fn walk(&self) {
        self.walk_rung().await;
        loop {
            let goal = self.goal();
            let (current, retired) = self.epochs();
            let step = {
                let mut pool = self.shared.lock().unwrap();
                pool.pool_mut().retire_through(retired);
                // What an evacuation moves is read last by the next fire.
                pool.next_move(goal, current.saturating_add(1))
            };
            let op = match step {
                None | Some(SegmentMove::Wait) => return,
                Some(SegmentMove::NoRoom) => {
                    self.no_room.store(true, Ordering::Release);
                    return;
                }
                Some(SegmentMove::Commit { prefix }) => PoolOp::CommitSegments(prefix),
                Some(SegmentMove::Evacuate { evacuation }) => PoolOp::Evacuate(evacuation),
                Some(SegmentMove::Release { prefix }) => PoolOp::ReleaseSegments(prefix),
                Some(SegmentMove::CommitSpan { n }) => PoolOp::CommitSpan(n),
                Some(SegmentMove::ReleaseSpan { n }) => PoolOp::ReleaseSpan(n),
            };
            self.no_room.store(false, Ordering::Release);
            tracing::debug!(engine = self.engine, goal, ?op, "a segment move");
            let commit = match &op {
                PoolOp::CommitSegments(prefix) => Some(*prefix),
                _ => None,
            };
            let span = match &op {
                PoolOp::CommitSpan(n) => Some(*n),
                _ => None,
            };
            let evacuating = matches!(op, PoolOp::Evacuate(_));
            match crate::scheduler::pool(self.engine, op).await {
                Ok(()) => {
                    if let Some(n) = span {
                        self.shared.lock().unwrap().span_committed(n);
                        if let Some(planner) = crate::planner::planner_for(self.model, self.engine)
                        {
                            planner.pages_freed();
                        }
                    }
                    if let Some(prefix) = commit {
                        self.shared.lock().unwrap().committed(prefix);
                        if let Some(planner) = crate::planner::planner_for(self.model, self.engine)
                        {
                            planner.pages_freed();
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(engine = self.engine, %error, "a segment move was refused");
                    if commit.is_some() || span.is_some() {
                        self.refused.store(true, Ordering::Release);
                    }
                    // A refused lend is not asked again on the next wake,
                    // only once room is freed.
                    if span.is_some() {
                        self.shared.lock().unwrap().block_span_commit(true);
                    }
                    return;
                }
            }
            if evacuating {
                // The segment drains until the first fire handed out after
                // the engine took the copy retires.
                let (current, _) = self.epochs();
                self.shared.lock().unwrap().evacuated(current);
                return;
            }
        }
    }
}

/// A test hook: while the file `PIE_SHARED_SEGMENTS_FILE` names holds a
/// number, that many mapped segments stand in for the governor's target, so
/// a test can force a grow or a shrink (and its evacuation) mid-run.
fn pinned_segments() -> Option<u32> {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    let path = PATH
        .get_or_init(|| std::env::var_os("PIE_SHARED_SEGMENTS_FILE").map(Into::into))
        .as_ref()?;
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

impl PoolTarget for SharedGrow {
    fn demand(&mut self) -> Demand {
        self.shared.lock().unwrap().demand()
    }

    fn segment_bytes(&self) -> u64 {
        self.shared.lock().unwrap().segment_bytes()
    }

    fn floor(&self) -> u64 {
        self.shared.lock().unwrap().floor()
    }

    fn hard_limit(&self) -> u64 {
        self.shared.lock().unwrap().hard_limit()
    }

    fn held(&self) -> u64 {
        self.shared.lock().unwrap().held()
    }

    fn growth(&self) -> Growth {
        self.shared.lock().unwrap().growth()
    }

    fn retarget(&mut self, segments: u64) -> Retarget {
        let segments = u32::try_from(segments).unwrap_or(u32::MAX);
        self.target.store(segments, Ordering::Release);
        self.wake.notify_one();
        let mapped = self.shared.lock().unwrap().mapped_segments();
        if self.no_room.load(Ordering::Acquire) && segments < mapped {
            Retarget::NoRoom {
                held: u64::from(mapped),
            }
        } else {
            Retarget::Reached
        }
    }

    fn take_refused(&mut self) -> bool {
        self.refused.swap(false, Ordering::AcqRel)
    }

    /// A governor tick: counted for the rung's patience, and the walk looks
    /// again.
    fn ticked(&mut self) {
        self.ticks.fetch_add(1, Ordering::AcqRel);
        self.wake.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pool says it can grow only where the next segment is one it
    /// may commit: a waiter told it grows into what the engine holds back
    /// would wait on a commit that never comes.
    #[test]
    fn the_pool_grows_only_into_a_segment_the_engine_binds() {
        let mut pool = super::super::tests::pool();
        let top = pool.pool().total_chunks() - pool.pool().act_max();
        let next = pool.pool().prefix_segments();
        let next_end = pool.pool().segments().range(next).end;
        // The ceiling one chunk short of the next segment's end.
        let held = top - (next_end - 1);
        pool.hold_back(u64::from(held) * pool.pool().chunk_bytes());
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let mut grow = SharedGrow::new(0, 58, shared.clone());
        let _ = grow.retarget(u64::from(u32::MAX));
        assert!(!grow.can_grow(), "the next segment straddles the ceiling");
        shared.lock().unwrap().hold_back(0);
        assert!(grow.can_grow(), "held back no more: it grows");
    }

    /// A pool that stands past what is asked comes down only once it has
    /// for the governor's patience: a pause shorter than that moves no rung.
    #[tokio::test(flavor = "current_thread")]
    async fn the_rung_comes_down_only_after_the_governor_s_patience() {
        const ENGINE: usize = 59;
        let pool = super::super::tests::pool();
        assert_eq!(pool.rung(), 2048);
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let mut grow = SharedGrow::new(0, ENGINE, shared);
        super::super::set_fire_cap_for_test(ENGINE, u32::MAX);
        // Nothing asked: the walk would cap fires to the smallest rung.
        grow.walk_rung().await;
        assert_eq!(
            super::super::fire_cap(ENGINE),
            u32::MAX,
            "not before a tick"
        );
        for tick in 1..PATIENCE {
            grow.ticked();
            grow.walk_rung().await;
            assert_eq!(super::super::fire_cap(ENGINE), u32::MAX, "tick {tick}");
        }
        grow.ticked();
        grow.walk_rung().await;
        assert!(
            super::super::fire_cap(ENGINE) < u32::MAX,
            "past the patience: fires are cut toward the rung asked"
        );
    }

    /// A program's fire term the largest rung cannot hold cuts fires to
    /// the widest rung that fits at once, without the governor's patience:
    /// a fire wider would take more than the span.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fire_term_past_the_largest_rung_cuts_fires_without_patience() {
        const ENGINE: usize = 57;
        let mut pool = super::super::tests::pool();
        let key = super::super::programs::FireKey {
            program: ::engine::ProgramId(1),
            extents: ::engine::BindExtents::default(),
        };
        pool.fire_facts_learned(
            key,
            &::engine::FireFacts {
                fixed: 55 * pool.pool().chunk_bytes(),
                ..::engine::FireFacts::default()
            },
        );
        pool.instance_bound(::engine::InstanceId(1), key);
        assert_eq!(pool.fit_cap(), 256);
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let grow = SharedGrow::new(0, ENGINE, shared);
        super::super::set_fire_cap_for_test(ENGINE, u32::MAX);
        grow.walk_rung().await;
        assert!(
            super::super::fire_cap(ENGINE) <= 256,
            "cut before any tick: {}",
            super::super::fire_cap(ENGINE)
        );
    }

    /// A raise the walk found no room for is not tried again on its next
    /// wake (a governor tick), only once room is freed.
    #[tokio::test(flavor = "current_thread")]
    async fn a_raise_refused_for_room_waits_for_room_freed_not_the_next_walk() {
        const ENGINE: usize = 52;
        let mut pool = super::super::tests::pool();
        pool.lower_rung(16).unwrap();
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let mut grow = SharedGrow::new(0, ENGINE, shared.clone());
        let _want = super::super::want_rows(ENGINE, 2048);
        grow.block_raise_for_test();
        grow.ticked();
        grow.walk_rung().await;
        assert_eq!(
            shared.lock().unwrap().rung(),
            16,
            "blocked: not tried again"
        );
        grow.room_freed();
        grow.walk_rung().await;
        assert_eq!(shared.lock().unwrap().rung(), 2048, "room freed: raised");
    }

    /// The walk runs on its wakes only: an ask made while it sleeps waits
    /// for the next event (here room freed), with no poll of its own.
    #[tokio::test(flavor = "current_thread")]
    async fn the_walk_wakes_on_room_freed_not_on_a_clock() {
        const ENGINE: usize = 51;
        let mut pool = super::super::tests::pool();
        pool.lower_rung(16).unwrap();
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let grow = SharedGrow::new(0, ENGINE, shared.clone());
        grow.start();
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        assert_eq!(shared.lock().unwrap().rung(), 16, "nothing asked");
        let _want = super::super::want_rows(ENGINE, 2048);
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            shared.lock().unwrap().rung(),
            16,
            "asked while it sleeps: no poll"
        );
        grow.room_freed();
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        assert_eq!(shared.lock().unwrap().rung(), 2048, "woken: it raises");
    }
}

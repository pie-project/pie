//! The governor's [`PoolTarget`] for an engine whose stores draw from a
//! [`SharedPool`]: the target in segments, moved toward off the fire path
//! by a task that speaks to the engine through its scheduler lane, so each
//! commit, evacuation and release is ordered with the fires around it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use tokio::sync::Notify;

use super::{GrowFor, RungStep, RungWalk, SegmentMove, Shared, SharedPool};
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
    /// The tick since which the rung has stood past what is asked: the
    /// lower that frees its chunks, and the cap down before it, wait the
    /// governor's patience from it.
    idle_since: Arc<std::sync::Mutex<Option<u64>>>,
    /// The last raise found no room: none is tried again until room is
    /// freed ([`Self::room_freed`]), the one event that can change that.
    raise_blocked: Arc<AtomicBool>,
    /// The planner waits on a grow this walk was told to make
    /// ([`Self::promise_grow`]): a walk that ends with the promise standing
    /// tells it no grow is coming ([`Self::no_grow`]).
    grow_waiter: Arc<AtomicBool>,
    /// What the walk tells when a promised grow does not come: the
    /// planner's `no_grow`, whose plan then goes on to demote, sweep and
    /// evict.
    no_grow: Arc<dyn Fn() + Send + Sync>,
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
            idle_since: Arc::new(std::sync::Mutex::new(None)),
            live_lanes: Arc::new(move || {
                let most = crate::engine::get_spec(engine)
                    .map_or(usize::MAX, |spec| spec.limits.max_forward_requests);
                let live = crate::inferlet::process::live_count().min(most);
                u32::try_from(live).unwrap_or(u32::MAX)
            }),
            raise_blocked: Arc::new(AtomicBool::new(false)),
            grow_waiter: Arc::new(AtomicBool::new(false)),
            no_grow: Arc::new(move || {
                if let Some(planner) = crate::planner::planner_for(model, engine) {
                    planner.no_grow();
                }
            }),
        }
    }

    /// What the walk tells when a promised grow does not come; for a test.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_no_grow(mut self, no_grow: impl Fn() + Send + Sync + 'static) -> Self {
        self.no_grow = Arc::new(no_grow);
        self
    }

    /// A governor tick; for a test.
    #[cfg(test)]
    pub(crate) fn ticked_for_test(&self) {
        self.ticks.fetch_add(1, Ordering::AcqRel);
    }

    /// The lanes live now that may decode, as the walk's floor counts them;
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

    /// Steps the activation rung toward the rung the passes still to run
    /// ask of a fire ([`super::want_rows`]), the smallest while none asks
    /// past it, until there or until a step has to wait: caps fires, lowers
    /// the pool's rung once no wider fire is in flight, or raises it, the
    /// engine first copying the state out of the larger span. A raise short
    /// of room waits on the room the asking pass has the planner free, the
    /// walk looking again each poll.
    ///
    /// Fires are cut narrower only for what the cut is needed for: the
    /// chunks a lower after it frees, or the fit. A cut within a chunk
    /// class frees nothing, so none is taken there however long nothing
    /// asks the width: the cap stays where the last pass left it, and the
    /// pause between two passes of one process moves no cap.
    /// The rows the walk holds the rung to: what passes and decode batches
    /// ask ([`super::wanted_rows`]), and never fewer than the lanes live now
    /// (a live process each, at most the engine's lane count), whose decode
    /// frames the scheduler holds rather than cuts. The rung never lowers
    /// under the decode fire the live lanes take, so their decodes never
    /// wait on a raise nobody funds; a lane joins through its prompt's pass,
    /// whose funded ask counts the live lanes beside it
    /// (`pipeline::fire::prefill_want`), and a batch's ask raises at once
    /// where nothing moves. Not the engine's whole lane count: a pool with
    /// one live lane rests at a small rung.
    fn walk_rows(&self) -> u32 {
        super::wanted_rows(self.engine).max((self.live_lanes)())
    }

    pub(crate) async fn walk_rung(&self) {
        loop {
            let (current, retired) = self.epochs();
            // What a raise taken below owes the engine, and whether the
            // rung stood under it ([`SharedPool::raise_applied`]).
            let mut owed = (None, false);
            let (step, goal) = {
                let mut pool = self.shared.lock().unwrap();
                pool.pool_mut().retire_through(retired);
                // A raise's copy is still the engine's to take: nothing
                // moves until it has.
                if pool.copying().is_some() {
                    return;
                }
                let goal = pool.rung_for(self.walk_rows());
                let cap = super::fire_cap(self.engine);
                // The pool stands past what is asked: it comes down only
                // once it has for the governor's patience, so a pause
                // between batches moves no rung. On a span backed on
                // demand an idle pool always comes down after it: the
                // carve held costs resident memory there.
                //
                // Interim: the patience decides this lower, the one move
                // down that frees chunks, only until the governor's `yield`
                // does; it goes with the governor's other debounce
                // constants then.
                let past = pool.pool().act_chunks(goal) < pool.pool().act_reserved();
                let tick = self.ticks.load(Ordering::Acquire);
                // An engine holding a rung the terms no longer fit is
                // brought down at once too: a bind waits on it.
                let forced = cap > pool.fit_cap() || pool.told_unfit();
                let waited = forced || {
                    let mut since = self.idle_since.lock().unwrap();
                    if past {
                        tick >= since
                            .get_or_insert(tick)
                            .saturating_add(u64::from(PATIENCE))
                    } else {
                        *since = None;
                        true
                    }
                };
                let step = self.rung_walk.lock().unwrap().step(
                    &pool,
                    goal,
                    cap,
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
                    Ok(RungStep::Raise { rung, evacuation }) => {
                        let stood = pool.stands_at(rung);
                        pool.raise_applied(&evacuation).map(|notice| {
                            owed = (notice, stood);
                            RungStep::Raise { rung, evacuation }
                        })
                    }
                    // A lower before the patience is out.
                    Ok(RungStep::Lower(_)) if !waited => Ok(RungStep::Wait),
                    // A cap down is taken for the chunks the lower after it
                    // frees, once the patience is out, and at once for the
                    // fit (the span holds no fire wider than the widest
                    // rung that fits with the programs' fire terms). Within
                    // a class it frees nothing and is not taken, however
                    // long the cap stands above what is asked.
                    Ok(RungStep::Cap(tokens)) if tokens < cap && !(forced || (past && waited)) => {
                        Ok(RungStep::Wait)
                    }
                    step => step,
                };
                (step, goal)
            };
            let step = match step {
                Ok(step) => step,
                Err(error) => {
                    // The pass asking funds the room as a planner waiter
                    // (`pipeline::fire`'s raise funding); the walk tries
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
                        // is not taken. What it asks as fires take it: the
                        // rung those rows take.
                        let pool = self.shared.lock().unwrap();
                        let now = super::fire_cap(self.engine);
                        let asked = pool.rung_for(self.walk_rows());
                        if tokens < now && pool.lifted(asked).min(pool.fit_cap()) > tokens {
                            continue;
                        }
                        super::set_fire_cap(self.engine, tokens);
                    }
                    tracing::debug!(
                        engine = self.engine,
                        tokens,
                        "fires cut to the activation rung"
                    );
                    crate::scheduler::nudge(self.engine);
                }
                RungStep::Lower(rung) => {
                    let lowered = {
                        let mut pool = self.shared.lock().unwrap();
                        // A pass lifted the cap since the walk capped it.
                        if super::fire_cap(self.engine) > rung {
                            continue;
                        }
                        pool.lower_notice(rung)
                    };
                    let notice = match lowered {
                        Ok(notice) => notice,
                        Err(error) => {
                            tracing::warn!(engine = self.engine, %error, "the activation rung stays");
                            return;
                        }
                    };
                    // Past the retire fence: the engine may give back the
                    // carve above the new rung, where it holds another (a
                    // lower that returns a dropped term's room at the rung
                    // it holds sends nothing, and was taken). What it held
                    // is given back to state only once the notice landed:
                    // until then the engine lays its guests' tables and
                    // covers a bind's term at the rung it held.
                    if let Some((from, to)) = notice {
                        // No lane: no engine holds a carve to come down from.
                        let landed = match crate::scheduler::scheduler_handle(self.engine) {
                            Ok(lane) => lane.pool(PoolOp::Rung { from, to }).await,
                            Err(_) => Ok(()),
                        };
                        let mut pool = self.shared.lock().unwrap();
                        match landed {
                            Ok(()) => {
                                if let Err(error) = pool.lower_landed(rung) {
                                    tracing::warn!(engine = self.engine, %error, "the activation rung stays");
                                }
                            }
                            Err(error) => {
                                pool.lower_refused();
                                tracing::warn!(engine = self.engine, %error, from, to, "a lower's notice was refused");
                            }
                        }
                    }
                    tracing::info!(engine = self.engine, rung, "the activation rung is lowered");
                    // A bind waiting on the engine to come down looks again.
                    super::rung_changed(self.engine);
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
                        super::programs::carry_raise(&self.shared, &evacuation, owed.0, &mover)
                            .await
                    {
                        tracing::debug!(engine = self.engine, %refused, "the walk's raise waits for room");
                        self.raise_blocked.store(true, Ordering::Release);
                        super::rung_changed(self.engine);
                        return;
                    }
                    super::raise_said(self.engine, rung, evacuation.moves.len(), owed.1);
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

    /// The engine answered the weight residency the load starts with: it
    /// holds `held` bytes back from here on
    /// ([`super::SharedPool::hold_back`]). Where that opens segments the
    /// engine binds and the load did not back, the prefix for the engine
    /// to back before the load returns ([`Self::backed`]): the pool then
    /// starts as the one of a load that never held more, with nothing
    /// left for the walk to commit ahead of a waiter once fires run.
    #[must_use]
    pub fn held_back(&self, held: u64) -> Option<u32> {
        let mut pool = self.shared.lock().unwrap();
        if !pool.hold_back(held) {
            return None;
        }
        let prefix = pool.bound_prefix();
        (prefix > pool.pool().prefix_segments()).then_some(prefix)
    }

    /// The engine backed segments `0..prefix` for [`Self::held_back`]:
    /// their chunks go out, and they are held as the load's own are
    /// ([`Self::hold_mapped`]).
    pub fn backed(&self, prefix: u32) {
        self.shared.lock().unwrap().committed(prefix);
        self.hold_mapped();
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

    /// The walk's last raise was refused for room; for a test.
    #[cfg(test)]
    pub(crate) fn block_raise_for_test(&self) {
        self.raise_blocked.store(true, Ordering::Release);
    }

    /// Looks again now: a waiter is short.
    pub fn wake(&self) {
        self.wake.notify_one();
    }

    /// Whether the walk's next step for a live waiter maps a segment: the
    /// same step the walk takes ([`SharedPool::grow_step`] toward
    /// [`Self::goal`]), so a waiter told the pool grows never waits on a
    /// grow the walk does not make. A span the lowered rung lends is no
    /// such step (live state never opens in the top rung's span), nor a
    /// segment the engine holds back, nor one past the goal.
    #[must_use]
    pub fn can_grow(&self) -> bool {
        let pool = self.shared.lock().unwrap();
        pool.grow_step(self.goal_in(&pool), GrowFor::Live).is_some()
    }

    /// The walk's next grow step for a live waiter; for a test.
    #[cfg(test)]
    pub(crate) fn live_step_for_test(&self) -> Option<super::GrowStep> {
        let pool = self.shared.lock().unwrap();
        pool.grow_step(self.goal_in(&pool), GrowFor::Live)
    }

    /// Promises a waiter a grow where [`Self::can_grow`]: the walk is woken
    /// to make it, and a walk that ends without mapping a segment since
    /// tells the planner no grow is coming. False, promising nothing,
    /// where the next step maps nothing.
    #[must_use]
    pub fn promise_grow(&self) -> bool {
        if !self.can_grow() {
            return false;
        }
        self.grow_waiter.store(true, Ordering::Release);
        self.wake();
        true
    }

    /// Whether a promised grow stands; for a test.
    #[cfg(test)]
    pub(crate) fn grow_promised(&self) -> bool {
        self.grow_waiter.load(Ordering::Acquire)
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
        let pool = self.shared.lock().unwrap();
        self.goal_in(&pool)
    }

    /// [`Self::goal`] of `pool`, held: the test pin as it stands, else the
    /// governor's target as [`SharedPool::walk_goal`] meets it.
    fn goal_in(&self, pool: &SharedPool) -> u32 {
        pinned_segments().unwrap_or_else(|| pool.walk_goal(self.target.load(Ordering::Acquire)))
    }

    /// Walks each time something it waits on happens: a retarget or a
    /// governor tick, a waiter short of room, room freed or a fire retired
    /// (the planner's `pages_freed`: what drains and fences wait on), or a
    /// bind or a raise that waits on a step of the rung's
    /// ([`super::ask_walk`]); never on a clock of its own.
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
    /// A grow promised a waiter and not made by the walk's end (its next
    /// step maps nothing, or the engine refused it) is told to the planner
    /// as no grow: its plan goes on to demote, sweep and evict on that
    /// event, never on a clock.
    pub(crate) async fn walk(&self) {
        self.walk_rung().await;
        self.walk_segments().await;
        if self.grow_waiter.swap(false, Ordering::AcqRel) {
            tracing::debug!(engine = self.engine, "a promised grow maps nothing");
            (self.no_grow)();
        }
    }

    async fn walk_segments(&self) {
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
                    // A segment mapped: what was promised a waiter is
                    // kept, and `pages_freed` wakes the planner.
                    if commit.is_some() || span.is_some() {
                        self.grow_waiter.store(false, Ordering::Release);
                    }
                    if let Some(n) = span {
                        self.shared.lock().unwrap().span_committed(n);
                        if let Some(planner) = crate::planner::planner_for(self.model, self.engine)
                        {
                            planner.pages_freed();
                        }
                    }
                    if let Some(prefix) = commit {
                        self.shared.lock().unwrap().committed(prefix);
                        // A guest bind short of a run waits on this
                        // (`roles::grow_for_guest`).
                        super::rung_changed(self.engine);
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
        // A waiter short of room, as `can_grow` is asked by one.
        pool.pool_mut().wait(1, 0, 1024);
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let mut grow = SharedGrow::new(0, 58, shared.clone());
        let _ = grow.retarget(u64::from(u32::MAX));
        assert!(!grow.can_grow(), "the next segment straddles the ceiling");
        shared.lock().unwrap().hold_back(0);
        assert!(grow.can_grow(), "held back no more: it grows");
    }

    /// A pool seeded with what a load that had armed nothing held back,
    /// told at the starting residency that the engine holds less, stands
    /// once the opened segments are backed as the pool of a load that never
    /// held more: the same chunks open, the same segments mapped and held
    /// as its target, the same bound. An answer that repeats what is held
    /// moves nothing.
    #[test]
    fn a_hold_back_lifted_at_the_starting_residency_is_a_pool_that_never_held_more() {
        let chunk = super::super::tests::pool().pool().chunk_bytes();
        // The load's fit held 28 chunks under the activation (the ceiling
        // at chunk 24 of 52); settled, it holds 4 (the ceiling at 48).
        let (unsettled, settled) = (28 * chunk, 4 * chunk);
        // The load that never held more: it backed every segment wholly
        // under what it binds before it returned.
        let never = {
            let mut pool = super::super::tests::pool();
            pool.hold_back(settled);
            let prefix = pool.bound_prefix();
            pool.committed(prefix);
            pool
        };
        let mut pool = super::super::tests::pool();
        pool.hold_back(unsettled);
        let loaded = pool.pool().prefix_segments();
        assert!(pool.bound_bytes() < never.bound_bytes());
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let grow = SharedGrow::new(0, 49, shared.clone());
        assert_eq!(grow.held_back(unsettled), None, "what the load stated");
        let prefix = grow.held_back(settled).expect("the lift opens segments");
        assert_eq!(prefix, never.pool().prefix_segments());
        assert!(prefix > loaded);
        {
            let pool = shared.lock().unwrap();
            assert_eq!(pool.pool().ceiling(), never.pool().ceiling());
            assert_eq!(pool.bound_bytes(), never.bound_bytes());
            assert_eq!(
                pool.pool().prefix_segments(),
                loaded,
                "no chunk goes out of a segment before the engine backed it"
            );
        }
        grow.backed(prefix);
        {
            let pool = shared.lock().unwrap();
            assert_eq!(
                pool.pool().prefix_segments(),
                never.pool().prefix_segments()
            );
            assert_eq!(pool.mapped_segments(), never.mapped_segments());
            assert_eq!(pool.held(), never.held());
            assert_eq!(pool.pool().headroom(), never.pool().headroom());
            assert_eq!(
                grow.target.load(Ordering::Acquire),
                never.mapped_segments(),
                "held as the load's own segments are"
            );
        }
        assert_eq!(grow.held_back(settled), None, "said again: nothing moves");
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

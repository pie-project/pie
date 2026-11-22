//! The governor's [`PoolTarget`] for an engine whose stores draw from a
//! [`SharedPool`]: the target in segments, moved toward off the fire path
//! by a task that speaks to the engine through its scheduler lane, so each
//! commit, evacuation and release is ordered with the fires around it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use tokio::sync::Notify;

use super::{RungStep, RungWalk, SegmentMove, Shared, SharedPool};
use crate::governor::{Growth, Mapped, PoolTarget, Retarget};
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
    /// The lanes live now that may decode ([`Self::walk_rows`]).
    live_lanes: Arc<dyn Fn() -> u32 + Send + Sync>,
    /// The last raise found no room: none is tried again until room is
    /// freed ([`Self::room_freed`]), the one event that can change that.
    raise_blocked: Arc<AtomicBool>,
    /// A raise to the frames the scheduler holds out under the cap
    /// ([`super::owed_cut_rows`]) short of room is being funded in the
    /// planner ([`Self::fund_owed`]): one at a time.
    funding: Arc<AtomicBool>,
    /// The planner waits on a grow this walk was told to make
    /// ([`Self::promise_grow`]): a walk that ends with the promise standing
    /// tells it no grow is coming ([`Self::no_grow`]).
    grow_waiter: Arc<AtomicBool>,
    /// What the walk tells when a promised grow does not come: the
    /// planner's `no_grow`, whose plan then goes on to demote, sweep and
    /// evict.
    no_grow: Arc<dyn Fn() + Send + Sync>,
    /// The engine's lane the walk's moves go through: its scheduler's.
    lane: Lane,
}

/// The engine's lane a segment move goes through, ordered with its fires.
type Lane = Arc<
    dyn Fn(usize, PoolOp) -> std::pin::Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>
        + Send
        + Sync,
>;

impl SharedGrow {
    /// Steers `shared`, engine `engine`'s, from the segments mapped at load.
    #[must_use]
    pub fn new(model: usize, engine: usize, shared: Shared) -> Self {
        let (mapped, start_cap, target) = {
            let mut pool = shared.lock().unwrap();
            pool.steers(engine);
            let mapped = pool.mapped_segments();
            // The pool reads the same target: a bind short of room waits on
            // a commit only where the walk would make one.
            let target = Arc::new(AtomicU32::new(mapped));
            pool.walked_to(Arc::clone(&target));
            (mapped, pool.start_cap(), target)
        };
        let _ = mapped;
        // Fires start cut to the rung the pool starts at: on a span backed
        // on demand, the smallest, until a raise reaches the engine.
        super::set_fire_cap(engine, start_cap);
        Self {
            model,
            engine,
            shared,
            target,
            wake: Arc::new(Notify::new()),
            refused: Arc::new(AtomicBool::new(false)),
            no_room: Arc::new(AtomicBool::new(false)),
            rung_walk: Arc::new(std::sync::Mutex::new(RungWalk::default())),
            live_lanes: Arc::new(move || {
                let most = crate::engine::get_spec(engine)
                    .map_or(usize::MAX, |spec| spec.limits.max_forward_requests);
                let live = crate::inferlet::process::live_count().min(most);
                u32::try_from(live).unwrap_or(u32::MAX)
            }),
            raise_blocked: Arc::new(AtomicBool::new(false)),
            funding: Arc::new(AtomicBool::new(false)),
            grow_waiter: Arc::new(AtomicBool::new(false)),
            no_grow: Arc::new(move || {
                if let Some(planner) = crate::planner::planner_for(model, engine) {
                    planner.no_grow();
                }
            }),
            lane: Arc::new(|engine, op| Box::pin(crate::scheduler::pool(engine, op))),
        }
    }

    /// The lane the walk's moves go through, for a test that plays the
    /// engine.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_lane(
        mut self,
        lane: impl Fn(PoolOp) -> anyhow::Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.lane = Arc::new(move |_, op| Box::pin(std::future::ready(lane(op))));
        self
    }

    /// What the walk tells when a promised grow does not come; for a test.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_no_grow(mut self, no_grow: impl Fn() + Send + Sync + 'static) -> Self {
        self.no_grow = Arc::new(no_grow);
        self
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
        super::wanted_rows(self.engine)
            .max((self.live_lanes)())
            .max(super::owed_cut_rows(self.engine))
    }

    /// The rows the walk holds the rung to while a process's demand waits
    /// for room: the live lanes' decode alone, the narrowest rung the
    /// admitted work needs; a throughput ask (a cuttable pass's) waits
    /// behind every process demand.
    fn waiting_rows(&self) -> u32 {
        (self.live_lanes)()
            .max(1)
            .max(super::owed_cut_rows(self.engine))
    }

    pub(crate) async fn walk_rung(&self) {
        loop {
            // Read before the pool's lock: the planner reads the pool under
            // its own.
            let planner_waits = super::planner_demand_waits(self.model, self.engine);
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
                // While a process's demand waits for room the rung walks
                // down to what the admitted work needs, at once, and takes
                // no room for throughput: each chunk it gives back tells the
                // pool's room signal, and the planner quotes its head again.
                let waits = pool.demand_waits(planner_waits);
                let rows = if waits {
                    self.waiting_rows()
                } else {
                    self.walk_rows()
                };
                let goal = pool.rung_for(rows);
                let cap = super::fire_cap(self.engine);
                // The pool stands past what is asked: the lower that frees
                // its chunks, and the cap down before it, are taken only
                // once something needs those chunks, so a pause between
                // batches moves no rung however long it lasts. Two things
                // do: a waiter parked short of a slot, and the governor's
                // target under what is mapped where the segment walk found
                // no room to evacuate into. The rung comes down last in
                // that shed: the pool first sheds its weight cache and
                // asks PLEX for idle prefixes (tier pressure). On a span
                // backed on demand the carve held is resident memory
                // within the budget, and comes down the same way: under
                // the system's pressure, as the yield says.
                let past = pool.pool().act_chunks(goal) < pool.pool().act_reserved();
                let needed = pool.pool().waiting()
                    || (self.target.load(Ordering::Acquire) < pool.mapped_segments()
                        && self.no_room.load(Ordering::Acquire));
                // An engine holding a rung the terms no longer fit is
                // brought down at once too: a bind waits on it. So is a
                // rung past what the admitted work needs while a process's
                // demand waits for room.
                let forced = cap > pool.fit_cap() || pool.told_unfit() || (waits && past);
                let waited = forced || !past || needed;
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
                    // room is freed. One that takes no more room (a rung of
                    // the class set aside, told the engine) is taken.
                    Ok(RungStep::Raise { ref evacuation, .. })
                        if blocked
                            && pool.pool().act_chunks(evacuation.rung)
                                > pool.pool().act_reserved() =>
                    {
                        Ok(RungStep::Wait)
                    }
                    Err(_) if blocked => Ok(RungStep::Wait),
                    // No throughput raise past what is set aside while a
                    // process's demand waits; a raise to the frames held
                    // out under the cap is admitted work, and goes.
                    Ok(RungStep::Raise { ref evacuation, .. })
                        if waits
                            && pool.pool().act_chunks(evacuation.rung)
                                > pool.pool().act_reserved()
                            && pool.rung_for(super::owed_cut_rows(self.engine))
                                < evacuation.rung =>
                    {
                        Ok(RungStep::Wait)
                    }
                    Ok(RungStep::Raise { rung, evacuation }) => {
                        let stood = pool.stands_at(rung);
                        pool.raise_applied(&evacuation).map(|notice| {
                            owed = (notice, stood);
                            RungStep::Raise { rung, evacuation }
                        })
                    }
                    // A lower nothing needs yet.
                    Ok(RungStep::Lower(_)) if !waited => Ok(RungStep::Wait),
                    // A cap down is taken for the chunks the lower after it
                    // frees, once something needs them, and at once for
                    // the fit (the span holds no fire wider than the widest
                    // rung that fits with the programs' fire terms). Within
                    // a class it frees nothing and is not taken, however
                    // long the cap stands above what is asked.
                    Ok(RungStep::Cap(tokens)) if tokens < cap && !(forced || (past && needed)) => {
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
                    // (`pipeline::fire`'s raise funding); frames held out
                    // under the cap have no pass asking, and the walk
                    // funds their raise itself (`fund_owed`). It tries
                    // again once room is freed, not on its next wake.
                    self.raise_blocked.store(true, Ordering::Release);
                    tracing::debug!(engine = self.engine, %error, goal, "the activation rung waits");
                    self.fund_owed();
                    return;
                }
            };

            match step {
                RungStep::Wait => {
                    self.lift_to_owed();
                    // A lower waits on the scheduler's grant, which it
                    // publishes each turn: an idle scheduler takes none
                    // until woken.
                    if super::rung_granted(self.engine) > super::fire_cap(self.engine) {
                        crate::scheduler::nudge(self.engine);
                    }
                    // A raise to held frames waiting for room is funded,
                    // never left to a room freed nothing is making.
                    if self.raise_blocked.load(Ordering::Acquire) {
                        self.fund_owed();
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
                        let rows = if pool.demand_waits(planner_waits) {
                            self.waiting_rows()
                        } else {
                            self.walk_rows()
                        };
                        let asked = pool.rung_for(rows);
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
                    let mover = WalkMover(self);
                    if let Err(refused) =
                        super::programs::carry_raise(&self.shared, &evacuation, owed.0, &mover)
                            .await
                    {
                        tracing::debug!(engine = self.engine, %refused, "the walk's raise waits for room");
                        self.raise_blocked.store(true, Ordering::Release);
                        super::rung_changed(self.engine);
                        // Taken back, the room it set aside is free again:
                        // the planner quotes its head now.
                        if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
                            planner.pages_freed();
                        }
                        return;
                    }
                    super::raise_said(self.engine, rung, evacuation.moves.len(), owed.1);
                    super::rung_changed(self.engine);
                }
            }
        }
    }

    /// Lifts the fire cap to the frames the scheduler holds out under it
    /// ([`super::owed_cut_rows`]) where the rung the engine holds covers
    /// them now (a raise taken, by the walk or by a pass's or a bind's
    /// funding): they seal at once. A cap under them with their rung held
    /// waited on a wake nothing sent, nothing in flight (the soak's stall:
    /// raised to 512, the cap left at 128). None while a raise's copy or a
    /// lower's notice is under way.
    fn lift_to_owed(&self) {
        let owed = super::owed_cut_rows(self.engine);
        if owed == 0 {
            return;
        }
        let pool = self.shared.lock().unwrap();
        if pool.copying().is_some() || pool.lowering() {
            return;
        }
        let rung = pool.rung_for(owed);
        if pool.raise_owed(rung) {
            return;
        }
        let cap = pool.lifted(rung).min(pool.fit_cap());
        if cap > super::fire_cap(self.engine) {
            super::set_fire_cap(self.engine, cap);
            tracing::debug!(engine = self.engine, cap, owed, "fires lifted to the frames held out");
        }
    }

    /// Funds the raise to the frames the scheduler holds out under the cap
    /// ([`super::owed_cut_rows`]) where it is short of room: admitted work
    /// no pass asks for, so no pass funds it. Its room is asked of the
    /// planner as a head keyed by the process whose frame it is
    /// ([`crate::planner::ResidencyPlanner::fund_rise`]), funded exactly
    /// as a bind's room, and once there the raise is taken with the queue
    /// held behind it; the walk then lifts the cap (room freed wakes it).
    /// One funding at a time; none where nothing is owed or no planner
    /// steers the engine.
    fn fund_owed(&self) {
        if super::owed_cut_rows(self.engine) == 0 || super::owed_by(self.engine).is_none() {
            return;
        }
        let Some(planner) = crate::planner::planner_for(self.model, self.engine) else {
            return;
        };
        if self.funding.swap(true, Ordering::AcqRel) {
            return;
        }
        let grow = self.clone();
        crate::rt::spawn(async move {
            grow.fund_owed_with(&planner).await;
            grow.funding.store(false, Ordering::Release);
        });
    }

    /// [`Self::fund_owed`]'s funding through `planner`, until the owed raise
    /// is taken, nothing is owed, or the planner refuses it by name.
    pub(crate) async fn fund_owed_with(&self, planner: &Arc<crate::planner::ResidencyPlanner>) {
        let engine = self.engine;
        loop {
            let rows = super::owed_cut_rows(engine);
            let Some(owner) = super::owed_by(engine).filter(|_| rows > 0) else {
                return;
            };
            let rung = {
                let pool = self.shared.lock().unwrap();
                let Some(at) = pool.raise_at(pool.rung_for(rows)) else {
                    // The engine comes down first: the walk's.
                    drop(pool);
                    super::ask_walk(engine);
                    return;
                };
                if !pool.raise_owed(at) || pool.copying().is_some() {
                    // Raised, or being raised: the walk lifts the cap.
                    drop(pool);
                    self.room_freed();
                    return;
                }
                at
            };
            tracing::debug!(engine, rung, %owner, "a raise to held frames asks the planner for room");
            let mover = WalkMover(self);
            let taken = planner
                .fund_rise(owner, rung, || {
                    super::programs::take_raise(&self.shared, move |_| rung, &mover)
                })
                .await;
            match taken {
                // Taken: the walk lifts the cap to it.
                Ok(Some(Ok(Some(_)))) => {
                    self.room_freed();
                    return;
                }
                // Short again (the pool moved since), or nothing left to
                // raise: asked again as it stands.
                Ok(Some(Ok(None))) => {}
                Ok(Some(Err(refused))) => {
                    tracing::warn!(engine, rung, %refused, "the engine refused a raise to held frames");
                    return;
                }
                // Its process is being evicted: its frames go with it, and
                // room freed brings the walk back.
                Ok(None) => return,
                Err(error) => {
                    tracing::warn!(engine, rung, %owner, %error, "a raise to held frames is refused");
                    return;
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
    /// grow the walk does not make. The span the rung lends is such a step
    /// once the prefix is whole (span lending); a segment the engine holds
    /// back, or one past the goal, is none.
    #[must_use]
    pub fn can_grow(&self) -> bool {
        let pool = self.shared.lock().unwrap();
        pool.grow_step(self.goal_in(&pool)).is_some()
    }

    /// The walk's next grow step for a live waiter; for a test.
    #[cfg(test)]
    pub(crate) fn live_step_for_test(&self) -> Option<super::GrowStep> {
        let pool = self.shared.lock().unwrap();
        pool.grow_step(self.goal_in(&pool))
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
            let mut evacuating = None;
            let mut compacting = None;
            let step = {
                let mut pool = self.shared.lock().unwrap();
                pool.pool_mut().retire_through(retired);
                // What an evacuation moves is read last by the next fire.
                pool.next_move(goal, current.saturating_add(1))
            };
            let op = match step {
                Some(SegmentMove::Wait) => return,
                // There, or past it with nothing of the prefix left to
                // give back: what stands is the rung's span, which comes
                // down only with no room, as the last of the shed.
                None => {
                    if self.shared.lock().unwrap().mapped_segments() > goal {
                        self.no_room.store(true, Ordering::Release);
                    }
                    return;
                }
                Some(SegmentMove::NoRoom) => {
                    self.no_room.store(true, Ordering::Release);
                    return;
                }
                Some(SegmentMove::Commit { prefix }) => PoolOp::CommitSegments(prefix),
                Some(SegmentMove::Evacuate {
                    evacuation,
                    applied,
                }) => {
                    evacuating = Some(applied);
                    PoolOp::Evacuate(evacuation)
                }
                Some(SegmentMove::Compact {
                    evacuation,
                    applied,
                    epoch,
                }) => {
                    compacting = Some((applied, epoch));
                    PoolOp::Evacuate(evacuation)
                }
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
            match (self.lane)(self.engine, op).await {
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
                    // A refused evacuation copied nothing: the pool takes it
                    // back, and what the segment holds has no room to go to
                    // until the engine takes one.
                    if let Some(applied) = &evacuating {
                        self.shared.lock().unwrap().evacuation_refused(applied);
                        self.no_room.store(true, Ordering::Release);
                    }
                    // A refused compaction copied nothing either: the bind
                    // it was for is funded as before, by the planner.
                    if let Some((applied, _)) = &compacting {
                        self.shared.lock().unwrap().compaction_refused(applied);
                        if let Some(planner) = crate::planner::planner_for(self.model, self.engine)
                        {
                            planner.pages_freed();
                        }
                    }
                    return;
                }
            }
            if evacuating.is_some() {
                // The engine took the copy and waits it out before it
                // gives the segment back: the segment drains once the
                // fires handed out by now retire, at once on an idle pool.
                let (current, _) = self.epochs();
                self.shared.lock().unwrap().evacuated(current);
            }
            if let Some((_, planned)) = compacting {
                // The engine took the copy on its stream ahead of every
                // later fire: the chunks moved off open once the fires
                // handed out by now retire (the loop's next step reads
                // that), and the run they clear is room given back.
                let (current, _) = self.epochs();
                self.shared.lock().unwrap().compaction_copied(planned, current);
                // Fires held for a refused chunk's move go again now.
                crate::scheduler::nudge(self.engine);
            }
        }
    }
}

/// A test hook: while the file `PIE_SHARED_SEGMENTS_FILE` names holds a
/// number, that many mapped segments stand in for the governor's target, so
/// a test can force a grow or a shrink (and its evacuation) mid-run.
pub(crate) fn pinned_segments() -> Option<u32> {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    let path = PATH
        .get_or_init(|| std::env::var_os("PIE_SHARED_SEGMENTS_FILE").map(Into::into))
        .as_ref()?;
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

impl PoolTarget for SharedGrow {
    fn segment_bytes(&self) -> u64 {
        self.shared.lock().unwrap().segment_bytes()
    }

    fn floor(&self) -> u64 {
        self.shared.lock().unwrap().floor()
    }

    fn least_mapped(&self) -> u64 {
        self.shared.lock().unwrap().least_mapped_bytes()
    }

    fn hard_limit(&self) -> u64 {
        self.shared.lock().unwrap().hard_limit()
    }

    fn held(&self) -> u64 {
        self.shared.lock().unwrap().held()
    }

    fn mapped(&self) -> Mapped {
        let pool = self.shared.lock().unwrap();
        Mapped {
            segments: u64::from(pool.mapped_segments()),
            draining: u64::from(pool.draining_segments()),
        }
    }

    fn growth(&self) -> Growth {
        self.shared.lock().unwrap().growth()
    }

    /// Judged on what the pool maps, the measure the governor's target is
    /// in. Past the target, the walk's last word decides: while it
    /// evacuates and releases on its own (or has yet to look),
    /// [`Retarget::Moving`]; once it found what is left with no room to
    /// move into, [`Retarget::NoRoom`], so the policy's tier pressure sheds
    /// idle cache for the difference.
    fn retarget(&mut self, segments: u64) -> Retarget {
        let could = self.shared.lock().unwrap().commit_can_come();
        self.target.store(
            u32::try_from(segments).unwrap_or(u32::MAX),
            Ordering::Release,
        );
        self.wake.notify_one();
        // Risen past what the pool maps: room the target held back can come.
        // A head that waits on it (short of room no move makes) has no
        // other event to plan again on.
        if !could
            && self.shared.lock().unwrap().commit_can_come()
            && let Some(planner) = crate::planner::planner_for(self.model, self.engine)
        {
            planner.target_raised();
        }
        let held = u64::from(self.shared.lock().unwrap().mapped_segments());
        if held <= segments {
            Retarget::Reached
        } else if self.no_room.load(Ordering::Acquire) {
            Retarget::NoRoom { held }
        } else {
            Retarget::Moving { held }
        }
    }

    fn take_refused(&mut self) -> bool {
        self.refused.swap(false, Ordering::AcqRel)
    }

    /// A governor tick: the walk looks again.
    fn ticked(&mut self) {
        self.wake.notify_one();
    }
}

/// The engine's side of the walk's raise: its own lane (the scheduler's,
/// or a test's engine), and its stores' fire epoch.
struct WalkMover<'a>(&'a SharedGrow);

impl super::programs::Mover for WalkMover<'_> {
    fn engine(&self) -> usize {
        self.0.engine
    }

    async fn evacuate(&self, evacuation: ::engine::chunk::Evacuation) -> Result<(), String> {
        (self.0.lane)(self.0.engine, PoolOp::Evacuate(evacuation))
            .await
            .map_err(|error| format!("{error:#}"))
    }

    async fn set_rung(&self, from: u32, to: u32) -> Result<(), String> {
        (self.0.lane)(self.0.engine, PoolOp::Rung { from, to })
            .await
            .map_err(|error| format!("{error:#}"))
    }

    fn epoch_now(&self) -> u64 {
        self.0.epochs().0
    }

    async fn check_table(&self, runtime: Vec<u32>) -> Result<(), String> {
        (self.0.lane)(self.0.engine, PoolOp::CheckTable(runtime))
            .await
            .map_err(|error| format!("{error:#}"))
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

    /// A pool that stands past what is asked comes down only once its
    /// chunks are needed: however many ticks pass with nothing short, no
    /// rung moves; a waiter parked short of a slot cuts fires toward the
    /// rung asked at once.
    #[tokio::test(flavor = "current_thread")]
    async fn the_rung_comes_down_only_once_its_chunks_are_needed() {
        const ENGINE: usize = 59;
        let pool = super::super::tests::pool();
        assert_eq!(pool.rung(), 2048);
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let mut grow = SharedGrow::new(0, ENGINE, shared.clone());
        super::super::set_fire_cap_for_test(ENGINE, u32::MAX);
        for tick in 0..64 {
            grow.ticked();
            grow.walk_rung().await;
            assert_eq!(super::super::fire_cap(ENGINE), u32::MAX, "tick {tick}");
        }
        shared.lock().unwrap().pool_mut().wait(1, 0, 1);
        grow.walk_rung().await;
        assert!(
            super::super::fire_cap(ENGINE) < u32::MAX,
            "a waiter is short: fires are cut toward the rung asked"
        );
    }

    /// The governor's target under what is mapped, where the segment walk
    /// found no room to evacuate into, needs the rung's chunks too: the
    /// rung is the last thing the shed takes.
    #[tokio::test(flavor = "current_thread")]
    async fn a_target_the_walk_has_no_room_for_brings_the_rung_down() {
        const ENGINE: usize = 56;
        let pool = super::super::tests::pool();
        let mapped = pool.mapped_segments();
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let grow = SharedGrow::new(0, ENGINE, shared);
        super::super::set_fire_cap_for_test(ENGINE, u32::MAX);
        grow.target.store(mapped.saturating_sub(1), Ordering::Release);
        grow.walk_rung().await;
        assert_eq!(
            super::super::fire_cap(ENGINE),
            u32::MAX,
            "under the target, the evacuation goes first"
        );
        grow.no_room.store(true, Ordering::Release);
        grow.walk_rung().await;
        assert!(super::super::fire_cap(ENGINE) < u32::MAX, "no room: the rung");
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

    /// A rung lowered to the smallest with only wider work queued is not
    /// left there: the work's ask raises it (a rung of the class set
    /// aside at once, even while a raise waits for room), and the cap
    /// lifts to the width the work takes, so it is never held for a raise
    /// nobody asks for.
    #[tokio::test(flavor = "current_thread")]
    async fn a_lowered_rung_with_only_wider_work_queued_is_raised_for_it() {
        const ENGINE: usize = 9211;
        let mut facts = super::super::tests::facts();
        facts.act_chunks = vec![(16, 1), (256, 4), (512, 4), (2048, 12)];
        let mut pool = super::super::SharedPool::new(&facts, 256).unwrap();
        pool.lower_rung(16).unwrap();
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let mut grow = SharedGrow::new(0, ENGINE, shared.clone());
        assert_eq!(super::super::fire_cap(ENGINE), 16);
        // Queued: a fire of 200 rows, wider than the rung held.
        let want = super::super::want_rows(ENGINE, 200);
        grow.ticked();
        grow.walk_rung().await;
        assert!(
            super::super::fire_cap(ENGINE) >= 200,
            "raised for the queued work"
        );
        assert_eq!(super::super::engine_rows(ENGINE), 256);
        drop(want);
        // Lowered under the class again, the told rung 256 with 512's
        // span set aside; a raise last refused for room.
        shared.lock().unwrap().lower_rung(256).unwrap();
        let _want = super::super::want_rows(ENGINE, 300);
        grow.block_raise_for_test();
        grow.walk_rung().await;
        assert_eq!(
            super::super::fire_cap(ENGINE),
            512,
            "the class's rung needs no room"
        );
        assert_eq!(super::super::engine_rows(ENGINE), 512);
    }

    /// Mac, yield 1, after a burst, every prompt finished: a pool full of
    /// idle cache, the system's pressure bringing the target under what it
    /// maps, and no fire at all. The walk evacuates the top segment into
    /// the room below; the engine takes the copy and waits it out itself,
    /// so the segment goes with no fire to retire, and while it does the
    /// pool answers Moving, asking nothing of the policy. The next segment
    /// has no room to evacuate into: NoRoom, tier pressure, and once the
    /// policy shed the idle cache the pool stands at its target. No answer
    /// is ever Reached past the target.
    #[tokio::test(flavor = "current_thread")]
    async fn a_target_under_what_idle_cache_holds_is_given_back_with_no_fire() {
        use crate::governor::run::Steer;
        use crate::governor::{Signal, Source, Yield};

        // The engine's stores, idle: no fire handed out, none outstanding.
        const ENGINE: usize = 65;
        let engines = vec![registry::EngineStores::default(); ENGINE + 1];
        let model = registry::register_model_with_chunks(16, &engines);
        let mut pool = super::super::tests::pool();
        pool.committed(6);
        // The prefix cache, idle: every record the pool holds, no fire
        // reading any. Those in the lowest chunks went, leaving room for
        // one segment's evacuation and no more.
        let records = pool
            .take_records(pool.records_available() as usize)
            .unwrap();
        let (gone, idle) = records.split_at(16);
        pool.give_records(gone);
        assert_eq!(pool.mapped_segments(), 8);
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        // The engine takes every move it is told.
        let moves = Arc::new(std::sync::Mutex::new(Vec::new()));
        let told = moves.clone();
        let mut grow = SharedGrow::new(model, ENGINE, shared.clone()).with_lane(move |op| {
            told.lock().unwrap().push(format!("{op:?}"));
            Ok(())
        });

        let dir = std::env::temp_dir().join(format!("pie-govr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pressure");
        std::fs::write(&path, "0").unwrap();
        let source = Source::File(path.clone());
        let mut reader = crate::governor::Reader::new(source.clone());
        let mut steer = Steer::new(&grow, Yield::new(1.0).unwrap()).with_source(source.name());
        let segment = grow.segment_bytes();
        let mut turn = |steer: &mut Steer, grow: &mut SharedGrow| {
            let mut signals = Vec::new();
            let turn = steer.turn(grow, reader.read(), &mut |signal| signals.push(signal));
            let held = grow.held();
            assert_eq!(turn.mapped.segments * segment, held, "the measure is held's");
            assert!(
                turn.answer != Retarget::Reached || held <= turn.segments * segment,
                "reached with {held} held over a target of {} segments",
                turn.segments
            );
            (turn, signals)
        };
        let (first, _) = turn(&mut steer, &mut grow);
        assert_eq!(first.answer, Retarget::Reached);

        // An app takes memory: the floor is the fewest segments the pool
        // can map (the reserved chunk's and the span's, 384 MiB, over its
        // byte floor of 272), and 384 + 640 * 0.55 MiB comes to 6 segments
        // of the 8 mapped. The walk has yet to look: Moving, nothing raised.
        std::fs::write(&path, "0.45").unwrap();
        let (pressed, signals) = turn(&mut steer, &mut grow);
        assert_eq!(pressed.segments, 6);
        assert_eq!(pressed.answer, Retarget::Moving { held: 8 });
        assert!(signals.is_empty(), "{signals:?}");
        // The walk evacuates segment 5 into the room below; the engine took
        // the copy, and with no fire handed out the segment goes at once.
        // Segment 4 has no room left to evacuate into.
        grow.walk().await;
        let told: Vec<String> = moves.lock().unwrap().clone();
        assert!(told[0].starts_with("Evacuate"), "{told:?}");
        assert_eq!(told[1], "ReleaseSegments(5)", "{told:?}");
        assert_eq!(shared.lock().unwrap().draining_segments(), 0);
        let (short, signals) = turn(&mut steer, &mut grow);
        assert_eq!(short.answer, Retarget::NoRoom { held: 7 });
        assert_eq!(
            signals,
            vec![Signal::TierPressure {
                held_bytes: 7 * segment,
                target_bytes: 6 * segment,
            }]
        );

        // The policy sheds the idle cache for the tier pressure: the walk
        // gives segment 4 back, and the pool stands at its target.
        shared.lock().unwrap().give_records(idle);
        grow.walk().await;
        let (eased, signals) = turn(&mut steer, &mut grow);
        assert_eq!(eased.answer, Retarget::Reached);
        assert_eq!(eased.mapped.segments, 6);
        assert_eq!(signals, vec![Signal::Eased]);
        assert_eq!(
            registry::get(model, ENGINE).kv.lock().current_epoch(),
            0,
            "no fire at all"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Yield 1 at full pressure on an empty pool whose byte floor (one
    /// lane, the reserved chunk, the activation span) is under the fewest
    /// segments it can map: the reserved chunk's segment and the span's.
    /// The governor's floor is those segments, so the target is what the
    /// pool can reach: the walk releases the rest and the pool answers
    /// Reached, never NoRoom with nothing left to shed.
    #[tokio::test(flavor = "current_thread")]
    async fn full_pressure_targets_the_fewest_segments_the_pool_can_map() {
        use crate::governor::run::Steer;
        use crate::governor::{Signal, Source, Yield};

        const ENGINE: usize = 67;
        let engines = vec![registry::EngineStores::default(); ENGINE + 1];
        let model = registry::register_model_with_chunks(16, &engines);
        let mut pool = super::super::tests::pool();
        pool.committed(6);
        let (byte_floor, least) = (pool.floor(), pool.least_mapped_bytes());
        let segment = pool.segment_bytes();
        assert!(
            least > byte_floor,
            "the layout's least ({least}) is over the byte floor ({byte_floor})"
        );
        assert_eq!(least % segment, 0, "whole segments");
        assert!(pool.held() > least, "more is mapped than the least");
        let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
        let mut grow = SharedGrow::new(model, ENGINE, shared.clone()).with_lane(move |_| Ok(()));

        let dir = std::env::temp_dir().join(format!("pie-govl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pressure");
        std::fs::write(&path, "1").unwrap();
        let source = Source::File(path.clone());
        let mut reader = crate::governor::Reader::new(source.clone());
        let mut steer = Steer::new(&grow, Yield::new(1.0).unwrap()).with_source(source.name());
        assert_eq!(steer.governor().limits().floor, least);

        let mut answers = Vec::new();
        for _ in 0..16 {
            let mut signals = Vec::new();
            let turn = steer.turn(&mut grow, reader.read(), &mut |signal| signals.push(signal));
            assert_eq!(turn.target, least, "the target is the least the pool maps");
            assert!(
                !signals
                    .iter()
                    .any(|signal| matches!(signal, Signal::TierPressure { .. })),
                "nothing is asked of the policy: {signals:?}"
            );
            answers.push(turn.answer);
            if turn.answer == Retarget::Reached {
                break;
            }
            grow.walk().await;
        }
        assert_eq!(answers.last(), Some(&Retarget::Reached), "{answers:?}");
        assert_eq!(grow.held(), least);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The soak's stall: the walk lowered the rung 512 -> 128 under
    /// prompts cut at 512 (the scheduler holds them out, owed), then a
    /// process demand waits for room and nothing is in flight. The walk
    /// raises the rung back to what is owed (512, nothing moved) and the
    /// fire cap rises with it in the same walk: the held frame seals at
    /// once under the new cap, never on a later wake that does not come.
    #[tokio::test(flavor = "current_thread")]
    async fn a_raise_back_to_frames_held_under_a_lowered_cap_lifts_the_cap_at_once() {
        for (at, on_demand) in [false, true].into_iter().enumerate() {
            for waits in [false, true] {
                let engine = 9300 + at * 2 + usize::from(waits);
                let mut facts = super::super::tests::facts();
                facts.act_chunks = vec![(16, 1), (128, 2), (512, 6), (2048, 12)];
                facts.span_backed_on_demand = on_demand;
                let mut pool = super::super::SharedPool::new(&facts, 256).unwrap();
                let evacuation = pool.plan_raise(512).unwrap();
                pool.raise_applied(&evacuation).unwrap();
                pool.raise_copied(0);
                // The walk lowered it to 128 under the prompts held out.
                pool.lower_rung(128).unwrap();
                if waits {
                    pool.set_guest_head(Some((1, 1)));
                }
                let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
                let told = Arc::new(std::sync::Mutex::new(Vec::new()));
                let seen = told.clone();
                let grow = SharedGrow::new(0, engine, shared.clone())
                    .with_live_lanes(|| 1)
                    .with_lane(move |op| {
                        seen.lock().unwrap().push(format!("{op:?}"));
                        Ok(())
                    });
                super::super::set_fire_cap_for_test(engine, 128);
                super::super::grant_rung(engine, 128);
                // The scheduler holds prompts cut at 512 out, owed.
                super::super::owe_cut_rows(engine, 512, None);
                grow.walk_rung().await;
                let rung = shared.lock().unwrap().rung();
                assert_eq!(rung, 512, "on_demand={on_demand} waits={waits}: raised to the owed");
                assert!(
                    super::super::fire_cap(engine) >= 512,
                    "on_demand={on_demand} waits={waits}: the cap rises with the raise, \
                     now {} (told {:?})",
                    super::super::fire_cap(engine),
                    told.lock().unwrap()
                );
                assert!(super::super::engine_rows(engine) >= 512, "the engine holds it");
                super::super::owe_cut_rows(engine, 0, None);
            }
        }
    }

    /// The same stall with the rung raised back outside the walk (a pass's
    /// or a bind's funding takes the raise, `take_raise`, and lifts no cap
    /// to the frames held out): the raise asks the walk to look, and the
    /// walk lifts the cap to what is owed at once, nothing in flight.
    #[tokio::test(flavor = "current_thread")]
    async fn a_raise_taken_outside_the_walk_lifts_the_cap_to_frames_held_out() {
        for (at, on_demand) in [false, true].into_iter().enumerate() {
            for waits in [false, true] {
                let engine = 9310 + at * 2 + usize::from(waits);
                let mut facts = super::super::tests::facts();
                facts.act_chunks = vec![(16, 1), (128, 2), (512, 6), (2048, 12)];
                facts.span_backed_on_demand = on_demand;
                let mut pool = super::super::SharedPool::new(&facts, 256).unwrap();
                let evacuation = pool.plan_raise(512).unwrap();
                pool.raise_applied(&evacuation).unwrap();
                pool.raise_copied(0);
                pool.lower_rung(128).unwrap();
                if waits {
                    pool.set_guest_head(Some((1, 1)));
                }
                let shared: Shared = Arc::new(std::sync::Mutex::new(pool));
                let grow = SharedGrow::new(0, engine, shared.clone())
                    .with_live_lanes(|| 1)
                    .with_lane(|_| Ok(()));
                super::super::set_fire_cap_for_test(engine, 128);
                super::super::grant_rung(engine, 128);
                super::super::owe_cut_rows(engine, 512, None);
                // Drain the ask the hold itself made.
                let _ = futures::FutureExt::now_or_never(super::super::walk_asked(engine));
                let taken =
                    super::super::programs::take_raise(&shared, |_| 512, &WalkMover(&grow))
                        .await
                        .expect("taken");
                assert!(taken.is_some(), "raised outside the walk");
                assert_eq!(super::super::fire_cap(engine), 128, "no cap lifted by it");
                assert!(
                    futures::FutureExt::now_or_never(super::super::walk_asked(engine)).is_some(),
                    "on_demand={on_demand} waits={waits}: the raise asks the walk to look"
                );
                grow.walk_rung().await;
                assert!(
                    super::super::fire_cap(engine) >= 512,
                    "on_demand={on_demand} waits={waits}: the walk lifts the cap to the held \
                     frames, now {}",
                    super::super::fire_cap(engine)
                );
                super::super::owe_cut_rows(engine, 0, None);
            }
        }
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

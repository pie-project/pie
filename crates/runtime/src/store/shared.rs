//! The runtime on one shared chunk pool: what an engine that states real
//! (unpinned) [`ChunkFacts`] runs on.
//!
//! One [`SlotPool`] holds every slot shape the engine states, kv list, window
//! lists and fixed alike, with the streamable weights' shapes after them; a
//! chunk holds slots of one shape while it is open, and any shape opens any
//! free chunk. The stores draw from it through
//! [`super::chunks::ShapePool::shared`]:
//!
//! - the kv store's pages are slot refs of the list shape;
//! - a working set's window pages (CONTRACT-BC 19, 21) are slot refs of
//!   each window list (one a distinct window, `T_w` tokens a slot of its
//!   own), refcounted by the rs store and dropped list by list once a slot
//!   lies wholly behind that list's window; a lane's table of each window
//!   list is in its own token order, ref 0 (the null chunk, never handed
//!   out) where it holds none;
//! - the rs store's state slots are **records**: one working set's fixed
//!   state, a slot of every fixed shape of role [`SlotRole::State`] for each
//!   of its pieces;
//! - the fixed shapes of roles [`SlotRole::Instance`],
//!   [`SlotRole::Channel`] and [`SlotRole::Adapter`]
//!   (`new-memory/STATE-ROLES.md`) are no part of a
//!   record nor of a working set: their slots are taken on demand as
//!   instances bind and channels register ([`super::roles`]), and are left
//!   out of every record and working-set count here;
//! - the rs store's buffer pages are slot refs of the **rs list** (a list
//!   shape of role [`SlotRole::RsBuffer`], CONTRACT-BC 18): taken as
//!   unfolded rows accumulate, refcounted by the store so forks share their
//!   common unfolded prefix, and given back as rows fold.
//!
//! A lane names its slots in [`::engine::fire::Lane::tables`] (kv list and
//! window list refs in token order) and [`::engine::fire::Lane::fixed`] (one `[run, cow]`
//! a piece). `cow == run` writes the record in place; a lane whose record is
//! shared (a fork, or a snapshot restored, in that fire) writes a fresh
//! record the engine copies `run` onto first. The copy-on-write record is
//! part of that fire's demand, granted before it runs as any slot is: none
//! is kept back up front, and one held counts against the planner's state
//! seats (`ResidencyPlanner::state_seats`).
//!
//! Position (CONTRACT-BC item 2): state lies below `total - act_chunks(rung)`;
//! raising the rung first moves state out of the new activation span
//! ([`RungWalk`], [`SharedPool::plan_raise`]), the engine copying it by
//! [`::engine::Engine::evacuate`] before the next fire
//! ([`SharedPool::apply_rung`] once it has). The top
//! `act_chunks_max` span is committed from the load and counts as held from
//! then on. Segments commit as a prefix (`commit_segments(n)`) ahead of need,
//! off the fire path; a shrink evacuates the highest committed segment of
//! the prefix (the only one `release_segments` can give back without the
//! ones above it) and releases it once the fire carrying the copy retires.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use ::engine::chunk::{
    ChunkFacts, Evacuation as EngineEvacuation, Growth as SegmentGrowth, ShapeId as EngineShape,
    SlotKind, SlotRole,
};

use super::chunks::STORE;
use super::segments::{Evacuation, SegmentState};
use super::slots::{
    Demand, Holder, RungEvacuation, ShapeId, ShapeKind, Slot, SlotError, SlotPool, SlotPoolFacts,
    SlotShape,
};
use crate::governor::Growth;

/// The pool behind an engine's stores, shared by every store and the
/// governor; each call takes the lock for itself, innermost of every store
/// lock.
pub type Shared = Arc<Mutex<SharedPool>>;

/// What a pool keeps committed ahead of what it has handed out, in bytes:
/// a mapping's cost on the lane against how often the lane pays it.
pub const COMMIT_AHEAD_BYTES: u64 = 64 * 1024 * 1024;

/// One engine's rung gates.
struct RungGate {
    /// The tokens a fire is cut and stepped to (SPEC decision 6): the
    /// activation rung the pool is at or going to, `u32::MAX` at its
    /// largest. Set by the pool's walker, read by the pipeline's cuts and
    /// the scheduler's steps.
    cap: std::sync::atomic::AtomicU32,
    /// The widest fire the scheduler may still have in flight: lowered to
    /// the cap once no step past it is (the scheduler holds back a launch
    /// past the cap), so the walker may lower the pool's rung behind it.
    grant: std::sync::atomic::AtomicU32,
    /// The rows of the rung the engine holds ([`SharedPool::engine_rung`],
    /// `u32::MAX` at the largest and where no pool steers the engine): no
    /// step wider reaches it ([`engine_rows`]).
    held: std::sync::atomic::AtomicU32,
    /// The widest rung rows of a frame the scheduler holds out under the
    /// cap ([`owe_cut_rows`]): fires cut under a wider cap before it came
    /// down, queued and no longer cuttable. Admitted work, not throughput:
    /// the walk holds the rung to it even while a process's demand waits,
    /// or the frame waited on a raise nothing asked for, nothing in flight
    /// to wake it. 0 where none is held.
    owed: std::sync::atomic::AtomicU32,
    /// The process whose lane holds that widest frame: the key the walk's
    /// raise to it is funded under in the planner, short of room
    /// ([`crate::planner::ResidencyPlanner::fund_rise`]).
    owed_by: Mutex<Option<uuid::Uuid>>,
    /// A raise to held frames refused by name (it never fits the pool
    /// beside its owner's own state, [`crate::planner::PlannerError::PastRung`]):
    /// the owner and why, for the scheduler to fail the frames held for it
    /// rather than hold them for good ([`refuse_owed`]).
    refused: Mutex<Option<(uuid::Uuid, String)>>,
    /// The event every wait on the activation rung waits on: the fire cap
    /// moved, a raise was applied, copied or refused, the rung was lowered,
    /// segments were committed, or room was freed (a fire retired).
    event: tokio::sync::Notify,
    /// The pool's walker is asked to look now ([`ask_walk`]): a bind or a
    /// raise waits on a cut or a lower that is the walk's to take.
    walk: tokio::sync::Notify,
    /// What the pool's walker awaits now and since when (a lane op's reply:
    /// a segment move, an evacuation, a rung's notice), `None` between: a
    /// stall's dump says it ([`walk_waiting`]).
    walk_waiting: Mutex<Option<(String, std::time::Instant)>>,
}

/// What `engine`'s walker awaits now and for how long, for a stall's dump:
/// `None` where it awaits nothing (it waits for its next wake, or runs).
#[must_use]
pub fn walk_waiting(engine: usize) -> Option<(String, std::time::Duration)> {
    rung_gate(engine)
        .walk_waiting
        .lock()
        .unwrap()
        .as_ref()
        .map(|(what, since)| (what.clone(), since.elapsed()))
}

/// Marks `engine`'s walker awaiting `what` until the guard drops.
pub(crate) fn walk_awaits(engine: usize, what: String) -> WalkAwait {
    *rung_gate(engine).walk_waiting.lock().unwrap() = Some((what, std::time::Instant::now()));
    WalkAwait(engine)
}

/// See [`walk_awaits`].
pub(crate) struct WalkAwait(usize);

impl Drop for WalkAwait {
    fn drop(&mut self) {
        *rung_gate(self.0).walk_waiting.lock().unwrap() = None;
    }
}

/// Every engine's rung gate by its id: one for each engine named here,
/// made the first time it is (uncapped, as an engine no walker has set is)
/// and kept for the process, as engine ids are (an id is never handed out
/// twice). No bound on how many.
static RUNG_GATES: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<usize, &'static RungGate>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

fn rung_gate(engine: usize) -> &'static RungGate {
    if let Some(gate) = RUNG_GATES.read().unwrap().get(&engine) {
        return gate;
    }
    RUNG_GATES
        .write()
        .unwrap()
        .entry(engine)
        .or_insert_with(|| {
            Box::leak(Box::new(RungGate {
                cap: std::sync::atomic::AtomicU32::new(u32::MAX),
                grant: std::sync::atomic::AtomicU32::new(u32::MAX),
                held: std::sync::atomic::AtomicU32::new(u32::MAX),
                owed: std::sync::atomic::AtomicU32::new(0),
                owed_by: Mutex::new(None),
                refused: Mutex::new(None),
                event: tokio::sync::Notify::new(),
                walk: tokio::sync::Notify::new(),
                walk_waiting: Mutex::new(None),
            }))
        })
}

/// The scheduler of `engine` holds frames out under the cap whose widest
/// fire takes `rows` rung rows (0: none), in the lane of process `by`:
/// what the walk raises the rung to as admitted work ([`RungGate::owed`]),
/// funded short of room under `by`'s key. A change asks the walk to look,
/// so the raise is triggered by the hold itself, not by a fire that may
/// never come.
pub fn owe_cut_rows(engine: usize, rows: u32, by: Option<uuid::Uuid>) {
    let gate = rung_gate(engine);
    *gate.owed_by.lock().unwrap() = by.filter(|_| rows > 0);
    let was = gate
        .owed
        .swap(rows, std::sync::atomic::Ordering::AcqRel);
    if rows > 0 && rows != was {
        ask_walk(engine);
    }
}

/// The raise to `by`'s held frames is refused by name (`why`): it never
/// fits. The scheduler fails those frames' fires with it
/// ([`take_refused_owed`]); the gate's event wakes what waits on the rung.
pub fn refuse_owed(engine: usize, by: uuid::Uuid, why: String) {
    let gate = rung_gate(engine);
    *gate.refused.lock().unwrap() = Some((by, why));
    gate.event.notify_waiters();
}

/// The owner and reason of a raise refused by name ([`refuse_owed`]),
/// taken: told once.
#[must_use]
pub fn take_refused_owed(engine: usize) -> Option<(uuid::Uuid, String)> {
    rung_gate(engine).refused.lock().unwrap().take()
}

/// The process whose held frame the walk's owed raise is for
/// ([`owe_cut_rows`]); `None` where none is held.
#[must_use]
pub fn owed_by(engine: usize) -> Option<uuid::Uuid> {
    *rung_gate(engine).owed_by.lock().unwrap()
}

/// See [`owe_cut_rows`].
#[must_use]
pub fn owed_cut_rows(engine: usize) -> u32 {
    rung_gate(engine)
        .owed
        .load(std::sync::atomic::Ordering::Acquire)
}

/// The engine refused a fire for the state in logical chunks `chunks`
/// (`engine::Error::Placement`): `engine`'s pool moves it down, its walk
/// asked to look now. A pool that keeps no chunks moves nothing.
pub fn evacuate_refused(engine: usize, chunks: &[u32]) {
    if let Some(shared) = crate::store::registry::try_get(0, engine).and_then(|stores| stores.shared) {
        shared.lock().unwrap().refuse_chunks(chunks);
    }
    ask_walk(engine);
}

/// How many of `chunks` `engine`'s pool found no free chunk below for
/// ([`SharedPool::refusals_stuck`]).
#[must_use]
pub fn refusals_stuck(engine: usize, chunks: &[u32]) -> u32 {
    crate::store::registry::try_get(0, engine)
        .and_then(|stores| stores.shared)
        .map_or(0, |shared| shared.lock().unwrap().refusals_stuck(chunks))
}

/// Whether any of `chunks` still waits for (or is in) its move down on
/// `engine`'s pool ([`evacuate_refused`]).
#[must_use]
pub fn refusals_pending(engine: usize, chunks: &[u32]) -> bool {
    crate::store::registry::try_get(0, engine)
        .and_then(|stores| stores.shared)
        .is_some_and(|shared| shared.lock().unwrap().refusals_pending(chunks))
}

/// Asks `engine`'s walker to look now: what waits on it (a bind whose term
/// the engine's rung no longer fits, fires cut wider than the widest rung
/// that fits) waits on its next step, not on the governor's next tick. Kept
/// where no walker waits yet: the next wait takes it.
pub fn ask_walk(engine: usize) {
    rung_gate(engine).walk.notify_one();
}

/// The next [`ask_walk`] of `engine`: what its walker waits on beside its
/// own events.
pub(crate) fn walk_asked(engine: usize) -> tokio::sync::futures::Notified<'static> {
    rung_gate(engine).walk.notified()
}

/// Wakes every wait on `engine`'s rung: what it waits on may have changed.
pub fn rung_changed(engine: usize) {
    rung_gate(engine).event.notify_waiters();
}

/// The next [`rung_changed`] of `engine`, armed before the caller looks at
/// what it waits on, so a change between the look and the wait is not
/// lost: pin it, `enable` it, look, then await it.
pub fn rung_watch(engine: usize) -> tokio::sync::futures::Notified<'static> {
    rung_gate(engine).event.notified()
}

/// Waits for the next [`rung_changed`] of `engine` after `check` reads
/// `false`; at once when it reads `true`. A change between the look and the
/// wait is not lost.
pub async fn rung_wait_until(engine: usize, mut check: impl FnMut() -> bool) {
    loop {
        let watch = rung_watch(engine);
        tokio::pin!(watch);
        watch.as_mut().enable();
        if check() {
            return;
        }
        watch.await;
    }
}

/// The tokens a fire on `engine` is cut and stepped to now.
#[must_use]
pub fn fire_cap(engine: usize) -> u32 {
    rung_gate(engine)
        .cap
        .load(std::sync::atomic::Ordering::Acquire)
}

/// Sets `engine`'s cap as the walker would: for a test of what waits on it.
#[cfg(test)]
pub(crate) fn set_fire_cap_for_test(engine: usize, tokens: u32) {
    set_fire_cap(engine, tokens);
}

/// Publishes the rows of the rung `engine` holds as a steering pool
/// would: for a test of what the lane refuses.
#[cfg(test)]
pub(crate) fn set_engine_rows_for_test(engine: usize, rows: u32) {
    rung_gate(engine)
        .held
        .store(rows, std::sync::atomic::Ordering::Release);
}

fn set_fire_cap(engine: usize, tokens: u32) {
    let was = rung_gate(engine)
        .cap
        .swap(tokens, std::sync::atomic::Ordering::AcqRel);
    rung_changed(engine);
    // A cap risen: the scheduler looks again now, nothing in flight being
    // sure to wake it, whether or not it has told the frames it holds out
    // under the cap yet (its report may be on its way).
    if tokens > was {
        crate::scheduler::nudge(engine);
    }
}

/// Whether `engine`'s fire cap is another than `used`, the one a scheduler
/// turn sealed under: that turn's holds are planned again under the cap as
/// it stands.
#[must_use]
pub fn cap_moved(engine: usize, used: u32) -> bool {
    fire_cap(engine) != used
}

/// The rows a step on `engine` may take: the rung the engine holds
/// ([`SharedPool::engine_rung`]), `u32::MAX` at the largest rung and where
/// no pool steers it. The lane refuses a step wider by name before the
/// engine stages it.
#[must_use]
pub fn engine_rows(engine: usize) -> u32 {
    rung_gate(engine)
        .held
        .load(std::sync::atomic::Ordering::Acquire)
}

/// The scheduler of `engine` has no step wider than `cap` in flight, and
/// submits none from here.
pub fn grant_rung(engine: usize, cap: u32) {
    rung_gate(engine)
        .grant
        .store(cap, std::sync::atomic::Ordering::Release);
}

fn rung_granted(engine: usize) -> u32 {
    rung_gate(engine)
        .grant
        .load(std::sync::atomic::Ordering::Acquire)
}

/// Per engine, the rows each pass still to run asks of one fire, by the
/// pass's key: what moves the activation rung (a pool sits at its smallest
/// rung while none asks past it).
static RUNG_WANTS: std::sync::LazyLock<
    Mutex<std::collections::HashMap<usize, std::collections::BTreeMap<u64, u32>>>,
> = std::sync::LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

static RUNG_WANT_KEY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A pass's ask that fires on its engine carry `rows`: held until the pass
/// is through (dropped), the walker raising the rung toward it meanwhile.
#[must_use = "the ask stands while it is held"]
pub struct RungWant {
    engine: usize,
    key: u64,
}

/// Asks that fires on `engine` carry `rows`; the pool's walker raises the
/// rung that holds them where room allows ([`RungWalk`]).
pub fn want_rows(engine: usize, rows: u32) -> RungWant {
    let key = RUNG_WANT_KEY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    RUNG_WANTS
        .lock()
        .unwrap()
        .entry(engine)
        .or_default()
        .insert(key, rows);
    RungWant { engine, key }
}

impl Drop for RungWant {
    fn drop(&mut self) {
        if let Some(wants) = RUNG_WANTS.lock().unwrap().get_mut(&self.engine) {
            wants.remove(&self.key);
        }
    }
}

/// Per process, the ask its last pass made: held while the process lives
/// and its passes stay wide, so the gaps between the passes of one prompt
/// (an inferlet feeding a prompt in several passes) lower nothing.
static PROCESS_WANTS: std::sync::LazyLock<Mutex<std::collections::HashMap<uuid::Uuid, RungWant>>> =
    std::sync::LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

/// Process `pid`'s pass on `engine` asks fires of `rows` (`None`: a pass of
/// a few rows, a decode, asks nothing): what its process holds the rung
/// for until its next pass says otherwise, or it ends ([`forget_process`]).
pub fn hold_for_process(pid: uuid::Uuid, engine: usize, rows: Option<u32>) {
    let mut wants = PROCESS_WANTS.lock().unwrap();
    match rows {
        Some(rows) => {
            wants.insert(pid, want_rows(engine, rows));
        }
        None => {
            wants.remove(&pid);
        }
    }
}

/// Process `pid` ended: its ask goes.
pub fn forget_process(pid: uuid::Uuid) {
    PROCESS_WANTS.lock().unwrap().remove(&pid);
}

/// The width a fire may take at the rung that holds `rows`: the cap the
/// walker lifts to there (`u32::MAX` at the largest rung).
#[must_use]
pub fn cap_for(shared: &Shared, rows: u32) -> u32 {
    let pool = shared.lock().unwrap();
    let rungs = pool.rungs();
    let goal = pool.rung_for(rows);
    if rungs.last() == Some(&goal) {
        u32::MAX
    } else {
        goal
    }
}

/// One line for a raise taken: at debug where the rung stood (a guest term
/// that grew under it, once a bind), at info where the rung moved.
pub(crate) fn raise_said(engine: usize, rung: u32, moved: usize, stood: bool) {
    if stood {
        tracing::debug!(
            engine,
            rung,
            moved,
            "the activation at a standing rung is raised"
        );
    } else {
        tracing::info!(engine, rung, moved, "the activation rung is raised");
    }
}

/// Raises `shared`'s rung at once to the one fires of `rows` take, where
/// nothing lies in the larger span and the room is there (a raise that must
/// move state is the walker's, through the engine's copy): the pass asking
/// is cut at the raised width from its first piece. A cap the walker
/// lowered behind a rung still up is lifted back to it.
pub fn raise_now(shared: &Shared, engine: usize, rows: u32) {
    // Read before the pool's lock: the planner reads the pool under its own.
    let waits = planner_demand_waits(0, engine);
    let mut pool = shared.lock().unwrap();
    let rungs = pool.rungs();
    let goal = pool.rung_for(rows);
    // A throughput raise is ordered after every process demand waiting for
    // room: while one waits it takes no room ([`SharedPool::demand_waits`]).
    if pool.demand_waits(waits) && pool.pool().act_chunks(goal) > pool.pool().act_reserved() {
        return;
    }
    let lifted = |rung: u32| {
        if rungs.last() == Some(&rung) {
            u32::MAX
        } else {
            rung
        }
    };
    if pool.raise_owed(goal) {
        let Some(at) = pool.raise_at(goal) else {
            return;
        };
        let Ok(evacuation) = pool.plan_raise(at) else {
            return;
        };
        let stood = pool.stands_at(at);
        if pool.raise_waits(&evacuation) {
            // The engine hears it first: the walk's to take.
            drop(pool);
            ask_walk(engine);
            return;
        }
        if pool.raise_applied(&evacuation).is_err() {
            return;
        }
        raise_said(engine, at, 0, stood);
        rung_changed(engine);
    }
    // The rung holds `goal` now, and the engine was told it: fires may
    // take its width, once a raise's copy is the engine's and no lower's
    // narrow is under way.
    if pool.copying().is_some() || pool.lowering() {
        return;
    }
    let cap = lifted(goal);
    if cap > fire_cap(engine) {
        set_fire_cap(engine, cap);
    }
    drop(pool);
}

/// Whether a process's demand waits for room on the planner of `model`'s
/// `engine` (a kv allocation, a restore or a guest bind short of room; a
/// rise is ordered among them by its own key, not counted). Read before
/// the pool's lock is taken: the planner reads the pool under its own.
#[must_use]
pub fn planner_demand_waits(model: usize, engine: usize) -> bool {
    crate::planner::planner_for(model, engine).is_some_and(|planner| planner.demand_waits())
}

/// The most rows a pass still to run on `engine` asks of one fire; 0 where
/// none asks.
#[must_use]
pub fn wanted_rows(engine: usize) -> u32 {
    RUNG_WANTS
        .lock()
        .unwrap()
        .get(&engine)
        .and_then(|wants| wants.values().copied().max())
        .unwrap_or(0)
}

/// One step of moving the activation rung toward a goal ([`RungWalk`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RungStep {
    /// Cut and step fires to this many tokens (`u32::MAX`: no cap).
    Cap(u32),
    /// Let state take the span above this rung's activation: no fire wider
    /// is in flight.
    Lower(u32),
    /// Raise the activation to this rung, the engine first copying what
    /// the evacuation moves out of its span.
    Raise {
        rung: u32,
        evacuation: RungEvacuation,
    },
    /// Nothing more until a fire retires or the scheduler grants.
    Wait,
}

/// The walk of the activation rung (SPEC decision 6). Down: cap fires
/// first, then, once the scheduler has no wider step in flight and every
/// fire handed out by then has retired, lower the pool's rung. Up: move
/// state out of the larger rung's span (the engine copying it before the
/// next fire), take the rung, then lift the cap.
#[derive(Debug, Default)]
pub struct RungWalk {
    lowering: Option<(u32, u64)>,
}

impl RungWalk {
    /// The next step toward `goal` (a rung's tokens) for `pool`, fires cut
    /// to `cap` and the scheduler granting `granted`; `current` the last
    /// fire epoch handed out, `retired` the last retired.
    pub fn step(
        &mut self,
        pool: &SharedPool,
        goal: u32,
        cap: u32,
        granted: u32,
        current: u64,
        retired: u64,
    ) -> Result<RungStep, RungError> {
        let rungs = pool.rungs();
        let Some(&largest) = rungs.last() else {
            return Ok(RungStep::Wait);
        };
        let goal = rungs
            .iter()
            .copied()
            .filter(|&rung| rung <= goal.min(pool.fit_rung()))
            .max()
            .unwrap_or(rungs[0]);
        // Fires cut wider than the widest rung that fits with the programs'
        // fire terms are cut to it first, whatever else the walk does: no
        // span holds a wider fire with the terms.
        let fit_cap = pool.fit_cap();
        if cap > fit_cap {
            self.lowering = None;
            return Ok(RungStep::Cap(fit_cap));
        }
        // The walk moves by what the goal's activation takes against what
        // is set aside now. Rungs whose activation takes as many chunks are
        // one class: the pool sets aside the same span for each, so a move
        // down within a class only caps fires, never lowers (and a cap
        // down there, freeing nothing, is one the walker leaves:
        // `SharedGrow::walk_rung`). A move up past the rung the engine was
        // told is a raise even within a class: the engine lays a fire's
        // inputs for the rung it holds, so it hears the wider one before
        // any fire takes it ([`SharedPool::raise_owed`]). A
        // program's fire term registered or dropped since moves the span a
        // rung takes ([`programs`]).
        let want = pool.pool().act_chunks(goal);
        let have = pool.pool().act_reserved();
        let lifted = if goal == largest { u32::MAX } else { goal };
        // The engine holds a rung the terms no longer fit
        // ([`SharedPool::told_unfit`]): it is told the goal as a lower is,
        // behind the retire fence, whatever the goal's activation takes
        // against what is set aside; a raise at the goal follows where it
        // takes more.
        if want < have || pool.told_unfit() {
            if cap > lifted {
                self.lowering = None;
                return Ok(RungStep::Cap(goal));
            }
            if granted > lifted {
                return Ok(RungStep::Wait);
            }
            return Ok(match self.lowering {
                Some((rung, epoch)) if rung == goal => {
                    if retired >= epoch {
                        self.lowering = None;
                        RungStep::Lower(goal)
                    } else {
                        RungStep::Wait
                    }
                }
                _ => {
                    self.lowering = Some((goal, current));
                    RungStep::Wait
                }
            });
        }
        self.lowering = None;
        if pool.raise_owed(goal) {
            // Never under the rung the engine holds: only a lower, above,
            // brings it down ([`SharedPool::raise_at`]).
            let rung = pool.raise_at(goal).unwrap_or(goal);
            let evacuation = pool.plan_raise(rung)?;
            return Ok(RungStep::Raise { rung, evacuation });
        }
        Ok(if cap == lifted {
            RungStep::Wait
        } else {
            RungStep::Cap(lifted)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SharedError {
    #[error("chunk facts: {0}")]
    Facts(String),
    #[error("shape {0} is pinned; a shared pool takes unpinned shapes only")]
    Pinned(u32),
    #[error("the reserved chunks are not one run from chunk 0")]
    Reserved,
    #[error(
        "the {tier} tier's {chunks} chunks of shape {shape}'s {slots_per_chunk} slots a chunk are more slots than a u32 slot ref names"
    )]
    Tier {
        tier: &'static str,
        shape: u32,
        chunks: u32,
        slots_per_chunk: u32,
    },
    #[error(
        "{kind} shapes {shapes:?}: the stores hold one kv list, one rs list, one window list a distinct window, and one one-piece instance and channel shape"
    )]
    Shapes {
        kind: &'static str,
        shapes: Vec<u32>,
    },
}

/// A shared pool's chunks as they divide for good ([`SharedPool::room`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolRoom {
    pub chunk_bytes: u64,
    pub total_chunks: u32,
    pub reserved_chunks: u32,
    /// The activation's span: the largest rung's, which a rise takes back
    /// whatever a lowered rung lent of it.
    pub act_chunks: u32,
    /// The span's chunks the smallest rung lends to state where they can
    /// be backed: room a grant may use while the rung is down, never room
    /// the pool promises, since a rise takes them back.
    pub lent_chunks: u32,
    /// The chunks the weights' pinned slots take.
    pub weight_chunks: u32,
    /// What is left for kv and state below the span: no move frees more,
    /// and no rung takes it.
    pub state_chunks: u32,
    /// `state_chunks` as kv list slots (pages).
    pub list_slots: u32,
    /// Tokens a kv list slot holds.
    pub list_tokens: u32,
}

impl PoolRoom {
    /// Says how the pool divides, for an ask of `need` kv pages it can
    /// never meet.
    #[must_use]
    pub fn describe(&self, need: u32) -> String {
        const MIB: u64 = 1 << 20;
        let mib = |chunks: u32| u64::from(chunks) * self.chunk_bytes / MIB;
        let short = need.saturating_sub(self.list_slots);
        let per_chunk = u64::from(self.list_slots / self.state_chunks.max(1)).max(1);
        let short_mib = u64::from(short).div_ceil(per_chunk) * self.chunk_bytes / MIB;
        format!(
            "the device pool holds {} MiB ({} chunks of {} MiB): weights pinned on it take \
             {} MiB, activation {} MiB, reserved {} MiB, which leaves {} MiB ({} kv pages, \
             {} tokens) for kv and state; this ask needs {need} kv pages ({} tokens), so no \
             eviction can make room — give the device at least {short_mib} MiB more \
             (`[engine.memory] budget_device`, or what else holds it)",
            mib(self.total_chunks),
            self.total_chunks,
            self.chunk_bytes / MIB,
            mib(self.weight_chunks),
            mib(self.act_chunks),
            mib(self.reserved_chunks),
            mib(self.state_chunks),
            self.list_slots,
            u64::from(self.list_slots) * u64::from(self.list_tokens),
            u64::from(need) * u64::from(self.list_tokens),
        )
    }
}

/// A per-token shape the stores take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenShape {
    pub id: EngineShape,
    pub pool: ShapeId,
    pub tokens: u32,
    /// The rows a window list retains; 0 for the kv list and the rs list.
    pub window: u32,
    pub slot_bytes: u64,
    pub slots_per_chunk: u32,
}

/// A fixed shape: `pieces` slots of it a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedShape {
    pub id: EngineShape,
    pub pool: ShapeId,
    pub pieces: u32,
    pub slot_bytes: u64,
}

/// The fixed shapes of an instance's tables and a channel's ring, where the
/// engine states them: one slot a piece, taken on demand
/// ([`super::roles::Roles`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoleShapes {
    pub instance: Option<FixedShape>,
    pub channel: Option<FixedShape>,
    pub adapter: Option<FixedShape>,
}

impl RoleShapes {
    /// The shape of role `role`, where the pool has it.
    #[must_use]
    pub fn of(&self, role: SlotRole) -> Option<FixedShape> {
        match role {
            SlotRole::Instance => self.instance,
            SlotRole::Channel => self.channel,
            SlotRole::Adapter => self.adapter,
            SlotRole::State | SlotRole::RsBuffer => None,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instance.is_none() && self.channel.is_none() && self.adapter.is_none()
    }

    fn iter(&self) -> impl Iterator<Item = &FixedShape> {
        self.instance
            .iter()
            .chain(&self.channel)
            .chain(&self.adapter)
    }
}

/// The tiers below an engine's shared pool (CONTRACT-BC 22): its host
/// chunks, a pool of every state shape of their own
/// ([`SharedPool::tier`]), and the kv list's slots in the engine's slot
/// file. Every move between them and the device is the engine's
/// `copy_slots`.
#[derive(Clone, Debug, Default)]
pub struct Tiers {
    pub host: Option<Shared>,
    pub disk_slots: u32,
    /// The kv list slots a disk chunk holds: a disk chunk is laid out as
    /// the device's, a role shape's slots in a chunk of their own.
    pub disk_chunk_slots: u32,
}

impl Tiers {
    /// The tiers `facts` state: `host_chunks` host chunks, and
    /// `disk_chunks` chunks of kv list slots on disk.
    pub fn of(facts: &ChunkFacts) -> Result<Self, SharedError> {
        let host =
            SharedPool::tier(facts, facts.host_chunks)?.map(|pool| Arc::new(Mutex::new(pool)));
        let list = facts.shapes.iter().find(|shape| {
            shape.role == SlotRole::State
                && matches!(shape.kind, SlotKind::List { window: None, .. })
        });
        // The disk tier's slots are counted in a `u32` and its refs are
        // below the count: a tier past that is refused, never cut short.
        let disk_slots = match list {
            None => 0,
            Some(list) => {
                u32::try_from(u64::from(facts.disk_chunks) * u64::from(list.slots_per_chunk))
                    .map_err(|_| SharedError::Tier {
                        tier: "disk",
                        shape: list.id.0,
                        chunks: facts.disk_chunks,
                        slots_per_chunk: list.slots_per_chunk,
                    })?
            }
        };
        let disk_chunk_slots = list.map_or(0, |list| list.slots_per_chunk);
        Ok(Self {
            host,
            disk_slots,
            disk_chunk_slots,
        })
    }
}

/// One step toward a segment target, as [`SharedPool::next_move`] plans it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentMove {
    /// Back segments `0..prefix` (`commit_segments(prefix)`); the pool hands
    /// out their chunks once the engine has, [`SharedPool::committed`].
    Commit { prefix: u32 },
    /// Copy `evacuation` (`applied` in the pool already), ahead of the
    /// next fire; [`SharedPool::evacuated`] once the engine took it,
    /// [`SharedPool::evacuation_refused`] where it refused it.
    Evacuate {
        evacuation: EngineEvacuation,
        applied: Evacuation,
    },
    /// Give back segments `prefix..` (`release_segments(prefix)`), released
    /// in the pool already.
    Release { prefix: u32 },
    /// Copy `evacuation`, chunk moves that stand a free run for a bind
    /// waiting on one (`applied` in the pool already, the chunks moved off
    /// fenced until the fire at `epoch`): [`SharedPool::compaction_copied`]
    /// once the engine took it, [`SharedPool::compaction_refused`] where it
    /// refused it.
    Compact {
        evacuation: EngineEvacuation,
        applied: Vec<super::segments::ChunkMove>,
        epoch: u64,
    },
    /// Back the top `n` span segments (`commit_span(n)`), where the span is
    /// backed on demand: the rung lends a chunk there no segment backs. The
    /// pool opens their chunks once the engine has, [`SharedPool::span_committed`].
    CommitSpan { n: u32 },
    /// Give back the span segments below the top `n` (`release_span(n)`),
    /// released in the pool already: no state chunk is open there and the
    /// rung no longer lends them.
    ReleaseSpan { n: u32 },
    /// The highest committed segment cannot be emptied into the rest.
    NoRoom,
    /// A drained segment waits for the fire carrying its copy.
    Wait,
}

/// A step of the walk that maps a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowStep {
    /// Back segments `0..prefix` ([`SegmentMove::Commit`]).
    Commit { prefix: u32 },
    /// Back the top `n` span segments ([`SegmentMove::CommitSpan`]).
    CommitSpan { n: u32 },
}

impl From<GrowStep> for SegmentMove {
    fn from(step: GrowStep) -> Self {
        match step {
            GrowStep::Commit { prefix } => Self::Commit { prefix },
            GrowStep::CommitSpan { n } => Self::CommitSpan { n },
        }
    }
}

/// See the module docs.
#[derive(Debug)]
pub struct SharedPool {
    pool: SlotPool,
    growth: SegmentGrowth,
    list: Option<TokenShape>,
    /// The rs buffer's list shape (CONTRACT-BC 18).
    rs_list: Option<TokenShape>,
    /// The window lists, narrowest window first (CONTRACT-BC 19).
    windows: Vec<TokenShape>,
    /// The fixed shapes of role [`SlotRole::State`]: a record's.
    fixed: Vec<FixedShape>,
    /// The fixed shapes taken on demand, outside every record.
    roles: RoleShapes,
    /// The first weight shape's pool id; the weight shapes follow.
    weights_first: ShapeId,
    /// Record id -> its slots, each fixed shape's pieces in order.
    records: Vec<Option<Vec<Vec<Slot>>>>,
    free_records: BTreeSet<u32>,
    /// The records the state chunks hold under the smallest rung: the most
    /// record ids there may be.
    records_most: u32,
    /// Chunks one working set takes at the longest context.
    least_chunks: u32,
    /// That longest context, in tokens.
    max_context: u32,
    /// A raise is applied here and its chunks' copy is not yet the
    /// engine's: no fire is let past the rung held before it
    /// ([`Self::raise_applied`]).
    copying: Option<u32>,
    /// The activation set aside before the raise whose copy is the
    /// engine's to take: what a refused raise goes back to.
    raise_from: Option<u32>,
    /// The rung the engine holds: the one it was last told
    /// (`Engine::set_rung`), set in the hold of the pool that decides the
    /// move, so whatever is decided next reads it, and put back where the
    /// engine refuses a raise. A notice goes only where a move leaves it
    /// another rung ([`Self::raise_applied`], [`Self::lower_rung`]).
    told: u32,
    /// [`Self::told`] before the raise whose copy is the engine's to take:
    /// what a refused raise goes back to.
    told_from: Option<u32>,
    /// The rung a lower has told the engine and the engine has not yet
    /// come down to ([`Self::lower_notice`]): until its narrow lands the
    /// engine holds [`Self::told`], so that rung's activation stays set
    /// aside, no raise is taken past it and no cap is lifted.
    lowering: Option<u32>,
    /// The engine refused the last `commit_span`: no span segment is asked
    /// again until room is freed ([`Self::block_span_commit`]).
    span_commit_blocked: bool,
    /// The head of the guest binds waiting for room, its need as `(run,
    /// pieces)`: runs of `run` adjacent chunks inside one segment. The walk
    /// commits while they do not stand free ([`Self::walk_goal`]), not only
    /// while their slot count is short.
    guest_head: Option<(u32, u32)>,
    /// The runs a bind waiting in the planner asks, `(run, pieces)`, where
    /// free chunks are enough but no run of them stands
    /// ([`Self::compaction_comes`]): the walk moves chunks until they do.
    compact_for: Option<(u32, u32)>,
    /// A compaction asked found no window ([`Self::compact_step`]): the
    /// walk tells the planner, whose head then goes on.
    compaction_answered: bool,
    /// The rung a raise waits on a compaction below its activation for
    /// ([`Self::compaction_for_raise`]).
    compact_raise: Option<u32>,
    /// Logical chunks the engine refused a fire for, their state where it
    /// will not read or write it (`engine::Error::Placement`): the walk
    /// moves them down ([`Self::refused_step`]); the scheduler holds the
    /// refused fires until none of theirs is here or moving
    /// ([`Self::refusals_pending`]).
    refused: BTreeSet<u32>,
    /// Of those, the ones a move planned for them carries now, until the
    /// engine took its copy.
    refused_moving: BTreeSet<u32>,
    /// Of those, the ones the last step found no free chunk below for: room
    /// the planner is asked to make ([`Self::refusals_stuck`]).
    refused_stuck: BTreeSet<u32>,
    /// The governor's segment target the walk steers this pool to, shared
    /// with it ([`target::SharedGrow`]); `None` where no walk steers it.
    walk_target: Option<Arc<std::sync::atomic::AtomicU32>>,
    /// The registered programs' fire terms ([`programs`]).
    programs: programs::ProgramFires,
    /// The engine whose rung gate this pool publishes the rung it holds
    /// to ([`Self::steers`], [`engine_rows`]).
    engine: Option<usize>,
}

/// The fence epoch a raise's moved-off chunks wait on until the engine has
/// taken its copy ([`SharedPool::raise_copied`]): no fire retires it.
const UNTIL_COPIED: u64 = u64::MAX - 1;

/// The engine's moves expressed as `(logical, new physical)`.
fn engine_moves(moves: &[super::segments::ChunkMove]) -> EngineEvacuation {
    EngineEvacuation {
        moves: moves.iter().map(|step| (step.logical, step.to)).collect(),
    }
}

impl SharedPool {
    /// Whether `facts` state a shared pool: shapes, none of them pinned.
    #[must_use]
    pub fn states_shared(facts: &ChunkFacts) -> bool {
        !facts.shapes.is_empty() && facts.shapes.iter().all(|shape| shape.pinned.is_none())
    }

    /// What a shared pool's units take, in bytes, as the planner counts
    /// them: a kv page is a kv list slot, a state slot one record (a slot of
    /// each fixed shape a piece). `None` where the pool is not shared. The
    /// engine's own page and slot bytes are its pinned pools', which a
    /// shared pool's slots are not.
    #[must_use]
    pub fn scale_of(facts: &ChunkFacts) -> Option<crate::policy::pressure::Scale> {
        if !Self::states_shared(facts) {
            return None;
        }
        let kv_page_bytes = facts
            .shapes
            .iter()
            .find(|shape| {
                shape.role == SlotRole::State
                    && matches!(shape.kind, SlotKind::List { window: None, .. })
            })
            .map_or(0, |shape| shape.slot_bytes);
        let state_slot_bytes = facts
            .shapes
            .iter()
            .filter(|shape| shape.role.replayable())
            .filter_map(|shape| match shape.kind {
                SlotKind::Fixed { pieces } => Some(u64::from(pieces) * shape.slot_bytes),
                SlotKind::List { .. } => None,
            })
            .sum();
        Some(crate::policy::pressure::Scale {
            kv_page_bytes,
            state_slot_bytes,
        })
    }

    /// The pool `facts` state, and `max_context` tokens the longest a
    /// working set holds. The rung starts at the largest: state never takes
    /// the top span until a rung is lowered. Where the span is backed on
    /// demand, at the smallest, the rung the engine backs at load
    /// ([`Self::start_cap`]).
    pub fn new(facts: &ChunkFacts, max_context: u32) -> Result<Self, SharedError> {
        facts
            .validate()
            .map_err(|error| SharedError::Facts(error.to_string()))?;
        if let Some(shape) = facts.shapes.iter().find(|shape| shape.pinned.is_some()) {
            return Err(SharedError::Pinned(shape.id.0));
        }
        let mut reserved = 0u32;
        for run in &facts.reserved {
            if run.first != reserved {
                return Err(SharedError::Reserved);
            }
            reserved += run.chunks;
        }
        Self::build(facts, reserved, max_context)
    }

    /// A tier below the device (CONTRACT-BC 22): `chunks` chunks of the
    /// device pool's size on the host, every slot shape of `facts` but the
    /// weights', any shape taking any chunk. Whole from the load (one
    /// committed segment, no activation, no reserved chunk: host refs are
    /// physical, ref 0 a slot like any other). `None` for no chunks.
    ///
    /// Checked as the device pool is ([`Self::new`]): the facts hang
    /// together, and every slot ref of the tier's own chunk count is a
    /// `u32` (the device's count says nothing of the tier's).
    pub fn tier(facts: &ChunkFacts, chunks: u32) -> Result<Option<Self>, SharedError> {
        if chunks == 0 || facts.shapes.is_empty() {
            return Ok(None);
        }
        facts
            .validate()
            .map_err(|error| SharedError::Facts(error.to_string()))?;
        if let Some(shape) = facts.shapes.iter().find(|shape| shape.pinned.is_some()) {
            return Err(SharedError::Pinned(shape.id.0));
        }
        if let Some(shape) = facts.shapes.iter().find(|shape| {
            u64::from(chunks) * u64::from(shape.slots_per_chunk) > u64::from(u32::MAX) + 1
        }) {
            return Err(SharedError::Tier {
                tier: "host",
                shape: shape.id.0,
                chunks,
                slots_per_chunk: shape.slots_per_chunk,
            });
        }
        let tier = ChunkFacts {
            total_chunks: chunks,
            segment_chunks: chunks,
            segments: ::engine::chunk::SegmentFacts {
                committed: 1,
                max: 1,
                growth: SegmentGrowth::Fixed,
            },
            reserved: Vec::new(),
            act_chunks: Vec::new(),
            eager_rungs: Vec::new(),
            held_bytes: 0,
            outside_bytes: 0,
            host_chunks: 0,
            disk_chunks: 0,
            weights: Vec::new(),
            ..facts.clone()
        };
        Self::build(&tier, 0, 0).map(Some)
    }

    fn build(facts: &ChunkFacts, reserved: u32, max_context: u32) -> Result<Self, SharedError> {
        let mut shapes = Vec::with_capacity(facts.shapes.len() + facts.weights.len());
        let (mut lists, mut rs_lists, mut windows, mut fixed) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut instances, mut channels, mut adapters) = (Vec::new(), Vec::new(), Vec::new());
        for shape in &facts.shapes {
            let pool = ShapeId::try_from(shapes.len()).expect("shapes fit a shape id");
            let kind = match shape.kind {
                SlotKind::List { tokens, window, .. } => {
                    let found = match window {
                        Some(_) => &mut windows,
                        None if shape.role == SlotRole::RsBuffer => &mut rs_lists,
                        None => &mut lists,
                    };
                    found.push(TokenShape {
                        id: shape.id,
                        pool,
                        tokens,
                        window: window.unwrap_or(0),
                        slot_bytes: shape.slot_bytes,
                        slots_per_chunk: shape.slots_per_chunk,
                    });
                    ShapeKind::List
                }
                SlotKind::Fixed { pieces } => {
                    let found = match shape.role {
                        SlotRole::Instance => &mut instances,
                        SlotRole::Channel => &mut channels,
                        SlotRole::Adapter => &mut adapters,
                        SlotRole::State | SlotRole::RsBuffer => &mut fixed,
                    };
                    found.push(FixedShape {
                        id: shape.id,
                        pool,
                        pieces,
                        slot_bytes: shape.slot_bytes,
                    });
                    ShapeKind::Fixed
                }
            };
            shapes.push(SlotShape {
                kind,
                slot_bytes: shape.slot_bytes,
                slots_per_chunk: shape.slots_per_chunk,
            });
        }
        for (kind, found) in [("kv list", &lists), ("rs list", &rs_lists)] {
            if found.len() > 1 {
                return Err(SharedError::Shapes {
                    kind,
                    shapes: found.iter().map(|shape| shape.id.0).collect(),
                });
            }
        }
        for (kind, found) in [("instance", &instances), ("channel", &channels)] {
            if found.len() > 1 || found.iter().any(|shape| shape.pieces != 1) {
                return Err(SharedError::Shapes {
                    kind,
                    shapes: found.iter().map(|shape| shape.id.0).collect(),
                });
            }
        }
        windows.sort_by_key(|shape: &TokenShape| shape.window);
        if windows.len() > crate::store::rs::MAX_WINDOW_LISTS {
            return Err(SharedError::Shapes {
                kind: "window list",
                shapes: windows.iter().map(|shape| shape.id.0).collect(),
            });
        }
        if let Some(pair) = windows.windows(2).find(|w| w[0].window == w[1].window) {
            return Err(SharedError::Shapes {
                kind: "window list",
                shapes: pair.iter().map(|shape| shape.id.0).collect(),
            });
        }
        let weights_first = ShapeId::try_from(shapes.len()).expect("shapes fit a shape id");
        shapes.extend(super::weights::pool_shapes(&facts.weights));
        let mut pool = SlotPool::new(SlotPoolFacts {
            chunk_bytes: facts.chunk_bytes,
            total_chunks: facts.total_chunks,
            segment_chunks: facts.segment_chunks,
            committed_segments: facts.segments.committed,
            reserved_chunks: reserved,
            shapes,
            act_chunks: facts.act_chunks.clone(),
            span_on_demand: facts.span_backed_on_demand,
            act_span: facts.act_span_chunks,
        });
        // The span backed from the load: the pool starts at the largest
        // rung. Backed on demand: at the smallest, the one the engine backs
        // at load; passes raise it through `Engine::set_rung` as fires need.
        if !facts.span_backed_on_demand
            && let Some(&(rung, _)) = facts.act_chunks.last()
        {
            pool.set_rung(rung)
                .expect("an empty pool takes its largest rung");
        }
        let mut shared = Self {
            pool,
            growth: facts.segments.growth,
            list: lists.first().copied(),
            rs_list: rs_lists.first().copied(),
            windows,
            fixed,
            roles: RoleShapes {
                instance: instances.first().copied(),
                channel: channels.first().copied(),
                adapter: adapters.first().copied(),
            },
            weights_first,
            records: Vec::new(),
            free_records: BTreeSet::new(),
            records_most: 0,
            least_chunks: 0,
            max_context,
            copying: None,
            raise_from: None,
            told: 0,
            told_from: None,
            lowering: None,
            span_commit_blocked: false,
            guest_head: None,
            compact_for: None,
            compaction_answered: false,
            compact_raise: None,
            refused: BTreeSet::new(),
            refused_moving: BTreeSet::new(),
            refused_stuck: BTreeSet::new(),
            walk_target: None,
            programs: programs::ProgramFires::default(),
            engine: None,
        };
        // The engine holds the rung the pool starts at.
        shared.told = shared.pool.rung();
        // Under the smallest rung state may take every chunk but the
        // reserved ones and that rung's activation.
        let state_chunks = facts
            .total_chunks
            .saturating_sub(reserved)
            .saturating_sub(facts.act_chunks.first().map_or(0, |&(_, chunks)| chunks));
        shared.records_most = shared.records_fitting(state_chunks);
        shared.least_chunks = shared.working_set_chunks(max_context);
        tracing::debug!(
            chunk_bytes = facts.chunk_bytes,
            total_chunks = facts.total_chunks,
            segment_chunks = facts.segment_chunks,
            committed = facts.segments.committed,
            shapes = facts.shapes.len(),
            records = shared.records_most,
            rungs = ?facts.act_chunks,
            act_span = ?facts.act_span_chunks,
            "a shared pool"
        );
        Ok(shared)
    }

    #[must_use]
    pub fn pool(&self) -> &SlotPool {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut SlotPool {
        &mut self.pool
    }

    #[must_use]
    pub fn list(&self) -> Option<TokenShape> {
        self.list
    }

    /// The rs buffer's list shape, where the model buffers rs rows.
    #[must_use]
    pub fn rs_list(&self) -> Option<TokenShape> {
        self.rs_list
    }

    /// The window lists, narrowest window first.
    #[must_use]
    pub fn windows(&self) -> &[TokenShape] {
        &self.windows
    }

    /// The fixed shapes of a record: role [`SlotRole::State`].
    #[must_use]
    pub fn fixed(&self) -> &[FixedShape] {
        &self.fixed
    }

    /// Whether the pool lays recurrent-state records: the model holds state.
    #[must_use]
    pub fn holds_records(&self) -> bool {
        !self.fixed.is_empty()
    }

    /// The fixed shapes taken on demand: an instance's tables, a ring.
    #[must_use]
    pub fn roles(&self) -> RoleShapes {
        self.roles
    }

    /// The kv list slots that, moved out whole, free `chunks` chunks: what
    /// a guest bind short of room is short, in pages
    /// (`planner::PoolPort::chunks_as_pages`).
    #[must_use]
    pub fn list_slots_for_chunks(&self, chunks: u32) -> u64 {
        self.list.map_or(0, |list| {
            u64::from(chunks) * u64::from(list.slots_per_chunk)
        })
    }

    /// The kv list slots `pages` kv pages take, as a restore gathers them:
    /// `None` without a kv list or with no pages.
    #[must_use]
    pub fn kv_takes(&self, pages: u32) -> Option<(ShapeId, u64)> {
        let list = self.list?;
        (pages > 0).then_some((list.pool, u64::from(pages)))
    }

    /// The single slots of each state shape `rs` takes: the rs list's
    /// (`pages`), each window list's (`windows`) and whole records'
    /// (`slots`, each fixed shape's pieces), as a restore's serve takes
    /// them before its unpark ([`SlotPool::opens_for`] places them).
    #[must_use]
    pub fn state_takes(&self, rs: crate::store::rs::RsUnits) -> Vec<(ShapeId, u64)> {
        let mut takes = Vec::new();
        if rs.windows.iter().any(|&n| n > 0) {
            for (window, &n) in self.windows.iter().zip(&rs.windows) {
                takes.push((window.pool, u64::from(n)));
            }
        } else if let Some(rs_list) = self.rs_list {
            takes.push((rs_list.pool, u64::from(rs.pages)));
        }
        for shape in &self.fixed {
            takes.push((shape.pool, u64::from(rs.slots) * u64::from(shape.pieces)));
        }
        takes.retain(|&(_, slots)| slots > 0);
        takes
    }

    /// The kv list slots whose chunks, let go whole, hold `rs` of the state
    /// shapes: the rs list's slots (`pages`), each window list's
    /// (`windows`, a store of windows) and whole records (`slots`). Any
    /// chunk opens for any shape, so kv that parks makes the room state
    /// lacks; 0 without a kv list.
    #[must_use]
    pub fn list_slots_for(&self, rs: crate::store::rs::RsUnits) -> u64 {
        let Some(list) = self.list else {
            return 0;
        };
        let chunks = |shape: ShapeId, slots: u64| -> u64 {
            let spc = self
                .pool
                .shape(shape)
                .map_or(1, |shape| shape.slots_per_chunk.max(1));
            slots.div_ceil(u64::from(spc))
        };
        let mut wanted = 0u64;
        if rs.windows.iter().any(|&n| n > 0) {
            for (window, &n) in self.windows.iter().zip(&rs.windows) {
                wanted += chunks(window.pool, u64::from(n));
            }
        } else if let Some(rs_list) = self.rs_list {
            wanted += chunks(rs_list.pool, u64::from(rs.pages));
        }
        for shape in &self.fixed {
            wanted += chunks(shape.pool, u64::from(rs.slots) * u64::from(shape.pieces));
        }
        wanted * u64::from(list.slots_per_chunk)
    }

    /// The engine's id of pool shape `pool`, a state shape's.
    #[must_use]
    pub fn engine_shape(&self, pool: ShapeId) -> Option<EngineShape> {
        self.list
            .iter()
            .chain(&self.rs_list)
            .chain(&self.windows)
            .find(|shape| shape.pool == pool)
            .map(|shape| shape.id)
            .or_else(|| {
                self.fixed
                    .iter()
                    .chain(self.roles.iter())
                    .find(|shape| shape.pool == pool)
                    .map(|shape| shape.id)
            })
    }

    /// A tier's bytes: every chunk but the reserved ones, the chunks opened
    /// for a shape, and the slots held in them.
    #[must_use]
    pub fn tier_bytes(&self) -> (u64, u64, u64) {
        let chunk = self.pool.chunk_bytes();
        let total = u64::from(self.pool.total_chunks() - self.pool.reserved_chunks()) * chunk;
        let opened = u64::from(self.pool.state_chunks()) * chunk;
        (total, opened, self.pool.demand().live)
    }

    #[must_use]
    pub fn weights_first(&self) -> ShapeId {
        self.weights_first
    }

    /// How the device's chunks divide: what the reserved chunks, the
    /// activation and the weights' pinned slots (held, every atom;
    /// streaming, the staging floor) hold for good, and what is left
    /// for kv and state. No eviction reaches below that. What is left is
    /// what can be backed, not the range: while the engine holds part of
    /// the range back, the chunks it binds none of and the segments that
    /// cannot be mapped whole are no room, so the planner's page total, its
    /// past-the-pool refusal and its seating count only what can be backed.
    #[must_use]
    pub fn room(&self) -> PoolRoom {
        let pool = &self.pool;
        let weight_chunks: u32 = (self.weights_first..)
            .map_while(|shape| {
                pool.shape(shape)
                    .map(|facts| (shape, facts.slots_per_chunk))
            })
            .map(|(shape, spc)| {
                let slots = pool.held_by(&Holder::Weights {
                    shape,
                    pinned: true,
                });
                u32::try_from(slots.div_ceil(u64::from(spc.max(1)))).unwrap_or(u32::MAX)
            })
            .sum();
        // What state can ever be backed in, not the range: below the span,
        // the chunks under what the engine binds that lie in segments the
        // pool may map ([`Self::backable_below_span`]). The kv facts count
        // the same (`ChunkFacts::kv_device_pages`): what the smallest rung
        // leaves of the span, where it is backed, is lent, and a rise takes
        // it back.
        let span_backable = !pool.span_on_demand() || pool.ceiling() >= pool.total_chunks();
        let lent_chunks = if span_backable {
            pool.act_max().saturating_sub(pool.act_chunks(0))
        } else {
            0
        };
        let state_chunks = self
            .backable_below_span()
            .saturating_sub(pool.reserved_chunks())
            .saturating_sub(weight_chunks);
        PoolRoom {
            chunk_bytes: pool.chunk_bytes(),
            total_chunks: pool.total_chunks(),
            reserved_chunks: pool.reserved_chunks(),
            act_chunks: pool.act_max(),
            lent_chunks,
            weight_chunks,
            state_chunks,
            list_slots: self.list.map_or(0, |list| {
                u32::try_from(u64::from(state_chunks) * u64::from(list.slots_per_chunk))
                    .unwrap_or(u32::MAX)
            }),
            list_tokens: self.list.map_or(0, |list| list.tokens),
        }
    }

    /// The chunks below the activation's span state can ever be backed in:
    /// under the ceiling the engine binds below while it holds part of the
    /// range back ([`Self::hold_back`]), and of those only the ones in
    /// segments the pool may map, as [`Self::bound_bytes`] bounds them: a
    /// segment under the top span that straddles the ceiling is never
    /// committed, so its chunks under the ceiling count for nothing. The
    /// whole of it up to the span where the engine holds none.
    fn backable_below_span(&self) -> u32 {
        let pool = &self.pool;
        let act_start = pool.total_chunks().saturating_sub(pool.act_max());
        let ceiling = pool.ceiling().min(act_start);
        let seg = pool.segments().segment_chunks().max(1);
        let top_start = pool.top_segment() * seg;
        if ceiling >= top_start {
            ceiling
        } else {
            ceiling / seg * seg
        }
    }

    /// The chunks `n` records take from empty chunks.
    fn record_chunks(&self, n: u64) -> u64 {
        self.fixed
            .iter()
            .map(|shape| {
                let spc = u64::from(self.pool.shape(shape.pool).map_or(1, |s| s.slots_per_chunk));
                (n * u64::from(shape.pieces)).div_ceil(spc.max(1))
            })
            .sum()
    }

    /// The most records `chunks` empty chunks hold.
    fn records_fitting(&self, chunks: u32) -> u32 {
        if self.fixed.is_empty() {
            return 0;
        }
        let (mut low, mut high) = (0u64, u64::from(u32::MAX));
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if self.record_chunks(mid) <= u64::from(chunks) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        u32::try_from(low).unwrap_or(u32::MAX)
    }

    /// Chunks one working set takes at `context` tokens: its list slots, a
    /// slots of each window list its window covers and the one its oldest
    /// row retires from, its record, and an rs list slot for the rows a fire
    /// leaves unfolded.
    fn working_set_chunks(&self, context: u32) -> u32 {
        let per_chunk = |slots: u64, spc: u32| slots.div_ceil(u64::from(spc.max(1)));
        let list = self.list.map_or(0, |list| {
            per_chunk(
                u64::from(context.div_ceil(list.tokens.max(1))),
                list.slots_per_chunk,
            )
        });
        let windows: u64 = self
            .windows
            .iter()
            .map(|list| {
                let tokens = list.tokens.max(1);
                let slots = context
                    .div_ceil(tokens)
                    .min(list.window.div_ceil(tokens) + 1);
                per_chunk(u64::from(slots), list.slots_per_chunk)
            })
            .sum();
        let rs = u64::from(self.rs_list.is_some());
        u32::try_from(list + windows + rs + self.record_chunks(1)).unwrap_or(u32::MAX)
    }

    // ---- slot refs of the list shapes ----

    /// The slot slot ref `slot_ref` of pool shape `shape` names.
    #[must_use]
    pub fn slot(&self, shape: ShapeId, slot_ref: u32) -> Slot {
        let spc = self
            .pool
            .shape(shape)
            .map_or(1, |s| s.slots_per_chunk.max(1));
        Slot {
            chunk: slot_ref / spc,
            index: slot_ref % spc,
        }
    }

    /// The slots a chunk of pool shape `shape` holds (1 where it is none).
    #[must_use]
    pub fn chunk_slots(&self, shape: ShapeId) -> u32 {
        self.pool
            .shape(shape)
            .map_or(1, |s| s.slots_per_chunk.max(1))
    }

    /// `f` run with `ticket`'s held chunks open to its takes
    /// ([`SlotPool::claim_open`]): a served restore's unpark and bind take
    /// what was reserved for them, and no one else's take can.
    pub fn claiming<R>(&mut self, ticket: Option<u64>, f: impl FnOnce(&mut Self) -> R) -> R {
        let opened = self.pool.claim_open(ticket);
        let out = f(self);
        self.pool.claim_close(opened);
        out
    }

    /// `n` fresh slots of `shape` for the store, as slot refs; none when
    /// they do not all fit.
    pub fn take_refs(&mut self, shape: ShapeId, n: u64) -> Option<Vec<u32>> {
        let slots = self.pool.acquire(STORE, shape, n).ok()?;
        Some(slots.iter().map(|&slot| self.pool.slot_ref(slot)).collect())
    }

    /// `n` adjacent slots of `shape` in one chunk for the store, as slot
    /// refs in order ([`SlotPool::acquire_run`]).
    pub fn take_run(&mut self, shape: ShapeId, n: u64) -> Result<Vec<u32>, SlotError> {
        let slots = self.pool.acquire_run(STORE, shape, n)?;
        Ok(slots.iter().map(|&slot| self.pool.slot_ref(slot)).collect())
    }

    /// `pieces` runs of `run` adjacent whole chunks of `shape` for the
    /// store, each inside one segment, as slot refs in order
    /// ([`SlotPool::acquire_chunk_runs`]).
    pub fn take_chunk_runs(
        &mut self,
        shape: ShapeId,
        run: u32,
        pieces: u32,
    ) -> Result<Vec<u32>, SlotError> {
        let slots = self.pool.acquire_chunk_runs(STORE, shape, run, pieces)?;
        Ok(slots.iter().map(|&slot| self.pool.slot_ref(slot)).collect())
    }

    /// The chunks of `shape`'s single slots and the slots a chunk holds
    /// ([`SlotPool::single_chunks`]).
    #[must_use]
    pub fn single_chunks(&self, shape: ShapeId) -> (u32, Vec<super::slots::SingleChunk>) {
        let spc = self.pool.shape(shape).map_or(0, |s| s.slots_per_chunk);
        (spc, self.pool.single_chunks(shape))
    }

    /// `n` slots of `shape` for the store outside logical chunks
    /// `outside`, as slot refs ([`SlotPool::acquire_outside`]).
    pub fn take_refs_outside(
        &mut self,
        shape: ShapeId,
        n: u64,
        outside: &std::collections::BTreeSet<u32>,
    ) -> Option<Vec<u32>> {
        let slots = self.pool.acquire_outside(STORE, shape, n, outside)?;
        Some(slots.iter().map(|&slot| self.pool.slot_ref(slot)).collect())
    }

    /// The store lets `refs` of `shape` go; no fire reads them any more.
    pub fn give_refs(&mut self, shape: ShapeId, refs: &[u32]) {
        let slots: Vec<Slot> = refs.iter().map(|&r| self.slot(shape, r)).collect();
        self.pool.release(STORE, &slots, 0);
    }

    // ---- records ----

    /// Records the pool's state chunks hold under the current rung: a
    /// smaller rung's room counts once the rung is lowered, not before.
    #[must_use]
    pub fn records_capacity(&self) -> u32 {
        self.records_fitting(self.state_chunks_now())
            .min(self.records_most)
    }

    /// The most record ids there may be: what the state chunks hold under
    /// the smallest rung.
    #[must_use]
    pub fn records_most(&self) -> u32 {
        self.records_most
    }

    /// The device's room for state at its narrowest rung: every chunk but
    /// the reserved and the smallest rung's activation, the span the rung
    /// lends included. One figure, whatever the rung now: what a need past
    /// it can never be held in (the never-fits check), on every path.
    /// Less what never holds state: the weights' pinned slots and what
    /// below the span the engine binds none of
    /// ([`Self::held_for_good`]). Counted whole, a bind needing more than
    /// the weights left waited for room no move could make (a Qwen3-30B-A3B
    /// ROCm boot whose pool was almost all weights: the first requests
    /// waited at bind 45 minutes with nothing running).
    #[must_use]
    pub fn state_chunks_most(&self) -> u32 {
        self.pool
            .total_chunks()
            .saturating_sub(self.pool.reserved_chunks())
            .saturating_sub(self.pool.act_chunks(0))
            .saturating_sub(self.held_for_good())
    }

    /// The chunks below the activation's span that never hold state: the
    /// weights' pinned slots, and what the engine binds none of
    /// ([`Self::room`]'s reading).
    fn held_for_good(&self) -> u32 {
        let room = self.room();
        let below_span = self
            .pool
            .total_chunks()
            .saturating_sub(self.pool.act_max())
            .saturating_sub(self.pool.reserved_chunks());
        below_span.saturating_sub(room.state_chunks)
    }

    /// The chunks state may take under the current rung: every chunk but
    /// the reserved ones and the activation set aside now.
    #[must_use]
    pub fn state_chunks_now(&self) -> u32 {
        self.pool
            .total_chunks()
            .saturating_sub(self.pool.reserved_chunks())
            .saturating_sub(self.pool.act_reserved())
            .saturating_sub(self.held_for_good())
    }

    /// Slots of pool shape `shape` the state chunks hold under the current
    /// rung, were every one that shape's.
    #[must_use]
    pub fn shape_capacity(&self, shape: ShapeId) -> u32 {
        let spc = self.pool.shape(shape).map_or(0, |s| s.slots_per_chunk);
        u32::try_from(u64::from(self.state_chunks_now()) * u64::from(spc)).unwrap_or(u32::MAX)
    }

    /// Records that fit now, each fixed shape's pieces from its partly
    /// filled chunks first.
    #[must_use]
    pub fn records_available(&self) -> u32 {
        if self.fixed.is_empty() {
            return 0;
        }
        let fits = |n: u64| {
            let needs: Vec<(ShapeId, u64)> = self
                .fixed
                .iter()
                .map(|shape| (shape.pool, n * u64::from(shape.pieces)))
                .collect();
            self.pool.fits(&needs)
        };
        let capacity = self.records_capacity();
        let (mut low, mut high) = (0u64, u64::from(capacity));
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if fits(mid) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let free = (capacity as usize).saturating_sub(self.live_records());
        (low as u32).min(free as u32)
    }

    fn live_records(&self) -> usize {
        self.records.len() - self.free_records.len()
    }

    /// `n` fresh records, ids lowest first; none when they do not all fit.
    pub fn take_records(&mut self, n: usize) -> Option<Vec<u32>> {
        if n == 0 {
            return Some(Vec::new());
        }
        if self.live_records() + n > self.records_capacity() as usize {
            return None;
        }
        let needs: Vec<(ShapeId, u64)> = self
            .fixed
            .iter()
            .map(|shape| (shape.pool, n as u64 * u64::from(shape.pieces)))
            .collect();
        if !self.pool.fits(&needs) {
            return None;
        }
        let mut taken: Vec<Vec<Slot>> = Vec::with_capacity(self.fixed.len());
        for &(shape, slots) in &needs {
            taken.push(
                self.pool
                    .acquire(STORE, shape, slots)
                    .expect("the records were checked to fit"),
            );
        }
        let mut ids = Vec::with_capacity(n);
        for at in 0..n {
            let record: Vec<Vec<Slot>> = self
                .fixed
                .iter()
                .zip(&taken)
                .map(|(shape, slots)| {
                    let pieces = shape.pieces as usize;
                    slots[at * pieces..(at + 1) * pieces].to_vec()
                })
                .collect();
            let id = match self.free_records.pop_first() {
                Some(id) => {
                    self.records[id as usize] = Some(record);
                    id
                }
                None => {
                    self.records.push(Some(record));
                    (self.records.len() - 1) as u32
                }
            };
            ids.push(id);
        }
        Some(ids)
    }

    /// The store lets records `ids` go; no fire reads them any more.
    pub fn give_records(&mut self, ids: &[u32]) {
        for &id in ids {
            let record = self
                .records
                .get_mut(id as usize)
                .and_then(Option::take)
                .unwrap_or_else(|| panic!("record {id} is not out"));
            for slots in &record {
                self.pool.release(STORE, slots, 0);
            }
            self.free_records.insert(id);
        }
    }

    /// Each fixed shape's slot refs record `id` holds, pieces in order.
    #[must_use]
    pub fn record_refs(&self, id: u32) -> Vec<(EngineShape, Vec<u32>)> {
        let Some(Some(record)) = self.records.get(id as usize) else {
            return Vec::new();
        };
        self.fixed
            .iter()
            .zip(record)
            .map(|(shape, slots)| {
                (
                    shape.id,
                    slots.iter().map(|&slot| self.pool.slot_ref(slot)).collect(),
                )
            })
            .collect()
    }

    /// The slots record `id` holds, every shape's.
    #[must_use]
    pub fn record_slots(&self, id: u32) -> Vec<Slot> {
        self.records
            .get(id as usize)
            .and_then(Option::as_ref)
            .map_or_else(Vec::new, |record| {
                record.iter().flatten().copied().collect()
            })
    }

    // ---- lane inputs ----

    /// A lane's [`::engine::fire::Lane::fixed`]: one `[run, cow]` a piece of
    /// each fixed shape. `cow == run` writes in place.
    #[must_use]
    pub fn lane_fixed(&self, run: u32, cow: u32) -> Vec<(EngineShape, Vec<[u32; 2]>)> {
        let cow_refs = self.record_refs(cow);
        self.record_refs(run)
            .into_iter()
            .zip(cow_refs)
            .map(|((shape, run), (_, cow))| {
                (
                    shape,
                    run.into_iter()
                        .zip(cow)
                        .map(|(run, cow)| [run, cow])
                        .collect(),
                )
            })
            .collect()
    }

    /// A lane's [`::engine::fire::Lane::tables`]: its list slot refs, and
    /// each window list's own table `windows[l]` (slot refs in its token
    /// order, 0 where trimmed: the null chunk); a shape the pool lacks or
    /// the lane does not read is left out.
    #[must_use]
    pub fn lane_tables(&self, list: &[u32], windows: &[Vec<u32>]) -> Vec<(EngineShape, Vec<u32>)> {
        let mut tables = Vec::new();
        if let Some(shape) = self.list.filter(|_| !list.is_empty()) {
            tables.push((shape.id, list.to_vec()));
        }
        for (shape, table) in self.windows.iter().zip(windows) {
            if !table.is_empty() {
                tables.push((shape.id, table.clone()));
            }
        }
        tables
    }

    /// Whole-slot copies `[src, dst]` of each window list, from slot refs
    /// copied `(list, src, dst)`, as [`::engine::fire::Lane::slot_copies`]
    /// states them; a list with none left out.
    #[must_use]
    pub fn window_copies(&self, copies: &[(usize, u32, u32)]) -> Vec<(EngineShape, Vec<[u32; 2]>)> {
        self.windows
            .iter()
            .enumerate()
            .map(|(at, shape)| {
                (
                    shape.id,
                    copies
                        .iter()
                        .filter(|&&(list, ..)| list == at)
                        .map(|&(_, src, dst)| [src, dst])
                        .collect::<Vec<_>>(),
                )
            })
            .filter(|(_, pairs)| !pairs.is_empty())
            .collect()
    }

    /// The slot ref a snapshot of record `id` is named by in
    /// [`::engine::fire::RsSnapshot::slot`]: its first fixed shape's first
    /// piece.
    #[must_use]
    pub fn snapshot_ref(&self, id: u32) -> u32 {
        self.record_refs(id)
            .first()
            .and_then(|(_, refs)| refs.first().copied())
            .unwrap_or(0)
    }

    /// Every piece of record `id`'s first fixed shape, in piece order, for
    /// [`::engine::fire::RsSnapshot::pieces`] (CONTRACT-BC 16); empty where
    /// the record is one piece, which [`Self::snapshot_ref`] names alone.
    #[must_use]
    pub fn snapshot_pieces(&self, id: u32) -> Vec<u32> {
        self.record_refs(id)
            .into_iter()
            .next()
            .map(|(_, refs)| refs)
            .filter(|refs| refs.len() > 1)
            .unwrap_or_default()
    }

    // ---- rungs ----

    /// Carries out `evacuation` (planned by [`Self::plan_raise`], copied by
    /// the engine) and takes its rung; the chunks moved off stay closed
    /// until the fire at `epoch` retires.
    pub fn apply_rung(&mut self, evacuation: &RungEvacuation, epoch: u64) -> Result<(), RungError> {
        self.pool
            .apply_rung(evacuation, epoch)
            .map_err(RungError::Pool)
    }

    /// Takes `evacuation`'s rung here, in the same hold of the pool it was
    /// planned in, so no slot is handed out between plan and apply: the
    /// chunks it moves off stay fenced, and no fire is let past the rung
    /// held before, until the engine has the copy ([`Self::raise_copied`]).
    /// The engine's copy follows ([`Self::rung_moves`]); a raise that moves
    /// nothing needs none.
    ///
    /// The engine may refuse the copy ([`Self::raise_refused`]).
    ///
    /// The notice the engine is sent first (`Engine::set_rung(from, to)`),
    /// where the raise leaves it another rung than it holds and it is told
    /// rungs ([`Self::tells_rungs`]): decided here, in the hold that takes
    /// the raise. A raise at the rung the engine holds (a guest term grew
    /// under it) owes none.
    pub fn raise_applied(
        &mut self,
        evacuation: &RungEvacuation,
    ) -> Result<Option<(u32, u32)>, RungError> {
        let before = self.pool.rung();
        let act = self.pool.act_reserved();
        if !self.raise_waits(evacuation) {
            self.apply_rung(evacuation, 0)?;
            self.tell(evacuation.rung);
            return Ok(None);
        }
        self.apply_rung(evacuation, UNTIL_COPIED)?;
        self.copying = Some(self.copying.map_or(before, |held| held.min(before)));
        self.raise_from = Some(self.raise_from.map_or(act, |held| held.min(act)));
        self.told_from.get_or_insert(self.told);
        Ok(self.tell(evacuation.rung))
    }

    /// Whether `evacuation`'s raise goes through the engine before fires
    /// take it: it moves state, which the engine copies first, or it
    /// leaves an engine that is told rungs (the span backed on demand)
    /// another rung than it holds, where the engine backs its carve for
    /// the wider rung first (`Engine::set_rung`). A move-free raise makes
    /// no lane trip with the span backed from the load, nor at the rung
    /// the engine holds: a guest term that grew under it is covered by the
    /// engine inside the bind that grew it.
    #[must_use]
    pub fn raise_waits(&self, evacuation: &RungEvacuation) -> bool {
        !evacuation.moves.is_empty()
            || (self.tells_rungs() && self.pool.rung_of(evacuation.rung) != self.told)
    }

    /// The rung the engine holds: the one it was last told.
    #[must_use]
    pub fn told(&self) -> u32 {
        self.told
    }

    /// Whether the engine holds the rung fires of `rung` take: a raise to
    /// it moves no rung.
    #[must_use]
    pub fn stands_at(&self, rung: u32) -> bool {
        self.pool.rung_of(rung) == self.told
    }

    /// Whether an engine that is told rungs holds one wider than the
    /// widest that fits with the programs' guest terms
    /// ([`Self::widest_fit`]): a term grew past it. The walk brings it
    /// down at once ([`RungWalk::step`]), and a bind that grew the term is
    /// not sent before ([`programs::fund_span`]): the engine covers a
    /// bind's term at the rung it holds, so it is never asked to at one
    /// whose term was not held against its span here.
    #[must_use]
    pub fn told_unfit(&self) -> bool {
        self.tells_rungs() && self.told > self.fit_rung()
    }

    /// The rung a raise for fires of `rung` is taken at: `rung`, and on an
    /// engine that is told rungs no narrower than the one it holds. A
    /// raise never brings the engine down (it would give back a carve
    /// fires in flight still take; that is the walk's lower, behind the
    /// retire fence), and the engine covers a bind's term at the rung it
    /// holds, so that rung's activation is what is set aside. `None` where
    /// the rung it holds no longer fits ([`Self::told_unfit`]): the walk
    /// brings it down first.
    #[must_use]
    pub fn raise_at(&self, rung: u32) -> Option<u32> {
        if !self.tells_rungs() {
            return Some(rung);
        }
        if self.told_unfit() {
            return None;
        }
        // A lower's narrow under way: the engine holds the rung it was told
        // before, and a raise past it would tell it a rung while it comes
        // down; it waits for the narrow to land.
        if self.lowering.is_some() && self.pool.rung_of(rung) > self.told {
            return None;
        }
        Some(if self.told > self.pool.rung_of(rung) {
            self.told
        } else {
            rung
        })
    }

    /// The engine holds the rung of `rung` from here; the notice it is
    /// owed (`Engine::set_rung(from, to)`) where that is another rung than
    /// it held and it is told rungs.
    fn tell(&mut self, rung: u32) -> Option<(u32, u32)> {
        let to = self.pool.rung_of(rung);
        let from = std::mem::replace(&mut self.told, to);
        self.publish_held();
        (from != to && self.tells_rungs()).then_some((from, to))
    }

    /// The rung a step may take on the engine now: the one it was told and
    /// has taken. A raise counts once the engine has it (its notice landed
    /// and its copy taken, [`Self::raise_copied`]); a lower from the moment
    /// its notice is sent ([`Self::lower_notice`]), so a fire handed out
    /// after it never passes the narrower rung.
    #[must_use]
    pub fn engine_rung(&self) -> u32 {
        let taken = self.told_from.unwrap_or(self.told);
        self.lowering.map_or(taken, |to| to.min(taken))
    }

    /// Whether fires of `rung` wait on a raise: its activation takes more
    /// than is set aside, or the engine was told a narrower rung (a rung of
    /// the same chunk class too: the engine lays a fire's inputs for the
    /// rung it holds, and hears a wider one before any fire takes it).
    #[must_use]
    pub fn raise_owed(&self, rung: u32) -> bool {
        self.pool.act_chunks(rung) > self.pool.act_reserved() || self.pool.rung_of(rung) > self.told
    }

    /// This pool steers `engine`: the rung it holds is published to the
    /// engine's lane from here ([`engine_rows`]).
    pub fn steers(&mut self, engine: usize) {
        self.engine = Some(engine);
        self.publish_held();
    }

    fn publish_held(&self) {
        if let Some(engine) = self.engine {
            rung_gate(engine).held.store(
                self.lifted(self.engine_rung()),
                std::sync::atomic::Ordering::Release,
            );
        }
    }

    /// The cap fires start at: the rung the pool starts at, lifted. The
    /// largest's (no cap) unless the span is backed on demand, where fires
    /// start cut to the smallest rung until a raise reaches the engine.
    #[must_use]
    pub fn start_cap(&self) -> u32 {
        self.lifted(self.rung())
    }

    /// Whether every rung change is told the engine (`Engine::set_rung`):
    /// the span is backed on demand. A change is a move to another rung
    /// than the engine holds ([`Self::told`]).
    #[must_use]
    pub fn tells_rungs(&self) -> bool {
        self.pool.span_on_demand()
    }

    /// The engine refused `evacuation`'s raise (its carve for the wider
    /// rung, or the copy): the raise is taken back, the state where it lay
    /// and the activation the rung in hand's. Fires stay cut to it; a later
    /// raise may try again. Nothing faults.
    pub fn raise_refused(&mut self, evacuation: &RungEvacuation) {
        if let Some(act) = self.raise_from.take() {
            self.pool.undo_rung(evacuation, act);
        }
        if let Some(told) = self.told_from.take() {
            self.told = told;
        }
        self.copying = None;
        self.publish_held();
    }

    /// The engine took a raise's copy (the state moved down, its table
    /// changed) and then refused the carve for the wider rung: the moved
    /// chunks stay where they went (their fence lifts once the fire at
    /// `epoch` retires), and the activation and the rung the engine holds
    /// are the rung in hand's again. Fires stay cut to it.
    pub fn raise_carve_refused(&mut self, epoch: u64) {
        self.pool.refence(UNTIL_COPIED, epoch);
        if let Some(act) = self.raise_from.take() {
            self.pool.restore_act(act);
        }
        if let Some(told) = self.told_from.take() {
            self.told = told;
        }
        self.copying = None;
        self.publish_held();
    }

    /// The engine has taken a raise's copy, carried by the fire at `epoch`
    /// (the next it is handed): its chunks moved off reopen once that fire
    /// retires, and fires may take the raised rung.
    pub fn raise_copied(&mut self, epoch: u64) {
        self.pool.refence(UNTIL_COPIED, epoch);
        self.copying = None;
        self.raise_from = None;
        self.told_from = None;
        self.publish_held();
    }

    /// The rung fires may not pass while a raise's copy is the engine's to
    /// take; `None` when none is.
    #[must_use]
    pub fn copying(&self) -> Option<u32> {
        self.copying
    }

    /// What raising the activation to `rung` moves: the state in its span,
    /// each chunk to a free one below (empty when nothing is in the way).
    pub fn plan_raise(&self, rung: u32) -> Result<RungEvacuation, RungError> {
        self.pool.plan_rung(rung).map_err(RungError::Pool)
    }

    /// `evacuation`'s moves as the engine takes them (`Engine::evacuate`).
    #[must_use]
    pub fn rung_moves(evacuation: &RungEvacuation) -> EngineEvacuation {
        engine_moves(&evacuation.moves)
    }

    /// Where a raise to `rung` stands for the planner
    /// ([`crate::planner::PoolPort::rise_room`]): past the pool beside the
    /// `own` chunks its owner's state takes at that rung, short of the
    /// chunks it lacks (its pieces those free now and those to empty), or
    /// fitting, a raise's copy under way included (nothing to fund).
    #[must_use]
    pub fn rise_room(&self, rung: u32, own: u32) -> crate::planner::RiseRoom {
        if let Some((need, total)) = self.pool.rung_past(rung, own) {
            return crate::planner::RiseRoom::Past { need, own, total };
        }
        if self.copying.is_some() {
            return crate::planner::RiseRoom::Fits;
        }
        match self.pool.rung_short(rung) {
            0 => crate::planner::RiseRoom::Fits,
            short => crate::planner::RiseRoom::Short {
                pieces: self.pool.free_chunk_runs(1).saturating_add(short),
            },
        }
    }

    /// The chunks raising the activation to `rung` lacks now, and the kv
    /// pages that many chunks hold: what must be freed before it fits.
    #[must_use]
    ///
    /// The pages are what a grant of them leaves that many chunks freed past
    /// what stands free now: every free slot of the kv list (the span the
    /// smaller rung lends included) comes first, so it counts on top (a
    /// grant it covers frees nothing).
    pub fn raise_short(&self, rung: u32) -> (u32, u32) {
        let chunks = self.pool.rung_short(rung);
        if chunks == 0 {
            return (0, 0);
        }
        let Some(list) = self.list else {
            return (chunks, chunks);
        };
        let spc = self
            .pool
            .shape(list.pool)
            .map_or(1, |shape| shape.slots_per_chunk.max(1));
        let free = self.pool.available(list.pool);
        let pages = u64::from(chunks) * u64::from(spc) + free;
        (chunks, u32::try_from(pages).unwrap_or(u32::MAX))
    }

    /// The first of `tables`' slot refs (engine shape, refs) that names no
    /// slot handed out: a chunk not open for its shape, or a free slot in
    /// one; the null chunk's refs (a window's empty slot) pass.
    #[must_use]
    pub fn unassigned(&self, tables: &[(EngineShape, Vec<u32>)]) -> Option<(EngineShape, u32)> {
        let reserved = self.pool.reserved_chunks();
        for (engine, refs) in tables {
            let Some(shape) = self.pool_shape(*engine) else {
                continue;
            };
            for &slot_ref in refs {
                let slot = self.slot(shape, slot_ref);
                if slot.chunk >= reserved && !self.pool.is_held(shape, slot) {
                    return Some((*engine, slot_ref));
                }
            }
        }
        None
    }

    /// The pool shape of engine shape `engine`, a state shape's or a
    /// role's: the inverse of [`Self::engine_shape`].
    #[must_use]
    pub fn pool_shape(&self, engine: EngineShape) -> Option<ShapeId> {
        self.list
            .iter()
            .chain(&self.rs_list)
            .chain(&self.windows)
            .find(|shape| shape.id == engine)
            .map(|shape| shape.pool)
            .or_else(|| {
                self.fixed
                    .iter()
                    .chain(self.roles.iter())
                    .find(|shape| shape.id == engine)
                    .map(|shape| shape.pool)
            })
    }

    /// The engine holds `bytes` of the range it stated back
    /// ([`::engine::chunk::ChunkFacts::held_bytes`], a load fact the pool
    /// is seeded with, and what the engine answers the weight residency
    /// the load starts with, where it settled its fit there; nothing says
    /// it again once the load returned): no state chunk is opened in the
    /// top `bytes` of the span below the activation, which the engine
    /// binds none of; all of it where it holds none.
    /// Whether the ceiling rose: room opened that a planner waiter may be
    /// short of.
    pub fn hold_back(&mut self, bytes: u64) -> bool {
        let ceiling = self.ceiling_for(bytes);
        let rose = ceiling > self.pool.ceiling();
        if ceiling != self.pool.ceiling() {
            self.pool.set_ceiling(ceiling);
        }
        rose
    }

    /// The chunk the engine binds state below while it holds `bytes` back:
    /// the held span lies directly under the activation span
    /// ([fitted, stated)); the whole range when it holds none.
    fn ceiling_for(&self, bytes: u64) -> u32 {
        if bytes == 0 {
            return self.pool.total_chunks();
        }
        let act_start = self.pool.total_chunks() - self.pool.act_max();
        let held =
            u32::try_from(bytes.div_ceil(self.pool.chunk_bytes().max(1))).unwrap_or(u32::MAX);
        act_start
            .saturating_sub(held)
            .max(self.pool.reserved_chunks())
    }

    /// The most the pool may map under what the engine holds back
    /// ([`Self::hold_back`]): the prefix segments wholly under what it
    /// binds ([`crate::store::slots::SlotPool::backs`]) and the top span,
    /// committed from the load; the whole range where it holds none.
    #[must_use]
    pub fn bound_bytes(&self) -> u64 {
        let total = self.pool.total_chunks();
        let seg = self.pool.segments().segment_chunks().max(1);
        let top_start = self.pool.top_segment() * seg;
        let prefix_end = self.bound_prefix() * seg;
        u64::from(prefix_end + (total - top_start)) * self.pool.chunk_bytes()
    }

    /// The segments below the top span that lie wholly under what the
    /// engine binds ([`Self::hold_back`]): every one of them where it holds
    /// none back. What `commit_segments(n)` may back as a prefix.
    #[must_use]
    pub fn bound_prefix(&self) -> u32 {
        let top = self.pool.top_segment();
        let ceiling = self.pool.ceiling();
        if ceiling >= self.pool.total_chunks() {
            return top;
        }
        let seg = self.pool.segments().segment_chunks().max(1);
        (ceiling / seg).min(top)
    }

    /// The smallest rung whose fires carry `rows`, the largest past every
    /// rung's; the smallest where `rows` is 0.
    #[must_use]
    ///
    /// No wider than the widest rung that fits with the programs' fire
    /// terms ([`Self::widest_fit`]): past it, fires are cut to it.
    pub fn rung_for(&self, rows: u32) -> u32 {
        let rungs = self.pool.rungs();
        let goal = rungs
            .iter()
            .copied()
            .find(|&rung| rung >= rows)
            .or_else(|| rungs.last().copied())
            .unwrap_or(u32::MAX);
        goal.min(self.fit_rung())
    }

    /// Every activation rung's tokens, ascending.
    #[must_use]
    pub fn rungs(&self) -> Vec<u32> {
        self.pool.rungs()
    }

    /// The rung the activation is set aside for now.
    #[must_use]
    pub fn rung(&self) -> u32 {
        self.pool.rung()
    }

    /// Lets state take the span above `rung`'s activation again. Where the
    /// span is backed on demand and the engine holds another rung, the
    /// notice it is sent (`Engine::set_rung(from, to)`), after the walk's
    /// retire fence: every fire wider than `rung` has retired, so it may
    /// give back the carve above it. A lower that only gives back what a
    /// dropped guest term took, the rung standing, sends none.
    ///
    /// Where `rung`'s activation takes more than is set aside, only an
    /// engine holding a wider rung is brought down to it (a guest term
    /// grew past the rung it holds, [`Self::told_unfit`]): what is set
    /// aside stays, and a raise at `rung` follows.
    /// The first half of a lower that brings the engine down: the notice it
    /// is sent (`Engine::set_rung(from, to)`), with nothing given back yet.
    /// The engine holds [`Self::told`] until that notice lands, laying its
    /// guests' tables and covering a bind's term at that rung, so what is
    /// set aside for it stays until [`Self::lower_landed`] (a bind funded
    /// meanwhile is funded at the rung the engine holds). A lower that
    /// tells the engine nothing is taken at once ([`Self::lower_rung`]).
    pub fn lower_notice(&mut self, rung: u32) -> Result<Option<(u32, u32)>, RungError> {
        let to = self.pool.rung_of(rung);
        let tells = self.tells_rungs() && self.told != to;
        if !tells {
            return self.lower_rung(rung);
        }
        if self.pool.act_chunks(rung) > self.pool.act_reserved() && self.told <= to {
            return Err(RungError::Pool(SlotError::Occupied { rung, chunks: 0 }));
        }
        self.lowering = Some(to);
        self.publish_held();
        Ok(Some((self.told, to)))
    }

    /// The engine took a lower's notice ([`Self::lower_notice`]) and holds
    /// `rung`'s from here: what it set aside above is given back. Where
    /// the pool cannot lower now, what is set aside stays; the engine's
    /// rung is still the one it holds.
    pub fn lower_landed(&mut self, rung: u32) -> Result<(), RungError> {
        self.lowering = None;
        let lowered = self.lower_rung(rung).map(|_| ());
        if lowered.is_err() {
            let _ = self.tell(rung);
        } else {
            // The engine gave back its carve above the rung it holds now:
            // the span there is the state's to take.
            self.pool.engine_narrowed();
        }
        lowered
    }

    /// The engine refused a lower's notice: it holds the rung it held, and
    /// nothing was given back.
    pub fn lower_refused(&mut self) {
        self.lowering = None;
        self.publish_held();
    }

    /// Whether a lower's notice is under way ([`Self::lower_notice`]).
    #[must_use]
    pub fn lowering(&self) -> bool {
        self.lowering.is_some()
    }

    pub fn lower_rung(&mut self, rung: u32) -> Result<Option<(u32, u32)>, RungError> {
        if self.pool.act_chunks(rung) > self.pool.act_reserved() {
            if !self.tells_rungs() || self.told <= self.pool.rung_of(rung) {
                return Err(RungError::Pool(SlotError::Occupied { rung, chunks: 0 }));
            }
            return Ok(self.tell(rung));
        }
        self.pool.set_rung(rung).map_err(RungError::Pool)?;
        let notice = self.tell(rung);
        // Told the lower (its notice rides the engine's lane ahead of any
        // copy or fire of state placed after it), the engine gives back its
        // carve above the rung: the span there is lent. Not told (the rung
        // standing, a guest term given back), its carve stays as wide, and
        // no state opens in it ([`SlotPool::engine_narrowed`]).
        if notice.is_some() {
            self.pool.engine_narrowed();
        }
        Ok(notice)
    }

    // ---- segments ----

    #[must_use]
    pub fn segment_bytes(&self) -> u64 {
        u64::from(self.pool.segments().segment_chunks()) * self.pool.chunk_bytes()
    }

    /// Segments with memory under them: the prefix (draining ones in it
    /// included) and the top span.
    #[must_use]
    pub fn mapped_segments(&self) -> u32 {
        self.pool.segments().mapped_segments()
    }

    /// Mapped segments an evacuation drained, waiting on the fire that
    /// carries its copy to retire.
    #[must_use]
    pub fn draining_segments(&self) -> u32 {
        let segments = self.pool.segments();
        (0..segments.count())
            .filter(|&s| matches!(segments.state(s), SegmentState::Draining { .. }))
            .count() as u32
    }

    /// The next step toward `target` mapped segments, the top span counted;
    /// `None` once there. A grow commits the lowest uncommitted segment, the
    /// commit ahead of need included; a shrink evacuates the highest
    /// committed segment of the prefix (applied here, for the engine to
    /// copy before the next fire; its chunks stay out until `epoch`
    /// retires), and releases it once drained.
    pub fn next_move(&mut self, target: u32, epoch: u64) -> Option<SegmentMove> {
        let segments = self.pool.segments();
        let top = self.pool.top_segment();
        let prefix = (0..top)
            .take_while(|&s| segments.state(s) != SegmentState::Uncommitted)
            .count() as u32;
        // A span segment backed on demand goes once the rung no longer
        // lends it and it holds nothing.
        if let Some(n) = self.pool.span_to_release() {
            self.pool.release_span(n);
            return Some(SegmentMove::ReleaseSpan { n });
        }
        if let Some(step) = self.grow_step(target) {
            return Some(step.into());
        }
        if let Some(step) = self.refused_step(epoch) {
            return Some(step);
        }
        if let Some(step) = self.compact_step(epoch) {
            return Some(step);
        }
        let mapped = self.mapped_segments();
        if mapped <= target || prefix == 0 {
            return None;
        }
        let last = prefix - 1;
        let segments = self.pool.segments();
        match segments.state(last) {
            SegmentState::Draining { .. } => Some(match self.pool.release_segment(last) {
                Ok(()) => SegmentMove::Release { prefix: last },
                Err(error) => {
                    tracing::trace!(%error, epoch, "a draining segment waits");
                    SegmentMove::Wait
                }
            }),
            SegmentState::Committed if segments.used(last) == 0 => {
                Some(match self.pool.release_segment(last) {
                    Ok(()) => SegmentMove::Release { prefix: last },
                    Err(_) => SegmentMove::NoRoom,
                })
            }
            SegmentState::Committed => {
                let Some(evacuation) = self.pool.plan_evacuation(last) else {
                    return Some(SegmentMove::NoRoom);
                };
                // Refused by the pool's own check (stale, or a run it
                // would split): said and not moved, never a panic (the M1
                // on f623134fc: "an evacuation planned under the lock
                // applies: Stale(11)", then a poisoned lock everywhere).
                if let Err(error) = self.pool.apply_evacuation(&evacuation, epoch) {
                    tracing::warn!(%error, segment = last, "store: a segment's evacuation refused by the pool's check");
                    return Some(SegmentMove::NoRoom);
                }
                Some(SegmentMove::Evacuate {
                    evacuation: engine_moves(&evacuation.moves),
                    applied: evacuation,
                })
            }
            SegmentState::Uncommitted => None,
        }
    }

    /// A compaction for the bind waiting on runs that free chunks would
    /// stand, if one is due: each step clears one run, the next planned
    /// once its moves' fence lifts. Over once the runs stand.
    fn compact_step(&mut self, epoch: u64) -> Option<SegmentMove> {
        let moves = match self.compact_for {
            Some((run, pieces)) if self.pool.free_chunk_runs(run) < pieces => {
                if self.pool.compacting() {
                    // One on its way: the next is planned once its fence
                    // lifts.
                    return None;
                }
                let Some(moves) = self.pool.plan_compaction_for(run, pieces) else {
                    // Asked and no window clears now: the ask is answered,
                    // never left pending (the M1's run 10: a restore's plan
                    // ended "compaction asked" every pass for ten minutes,
                    // the walk sending none). The head's next pass reads the
                    // pool as it stands and goes to its eviction.
                    tracing::debug!(
                        run,
                        pieces,
                        free_runs = self.pool.free_chunk_runs(run),
                        "store: a compaction asked finds no window to clear; answered no room"
                    );
                    self.compact_for = None;
                    self.compaction_answered = true;
                    return None;
                };
                moves
            }
            Some(_) => {
                self.compact_for = None;
                return None;
            }
            None => {
                let rung = self.compact_raise?;
                let Some((run, pieces)) = self.pool.raise_runs_short(rung) else {
                    self.compact_raise = None;
                    return None;
                };
                let below = self.pool.total_chunks() - self.pool.act_chunks(rung);
                self.pool.plan_compaction_below(run, pieces, below)?
            }
        };
        self.pool
            .apply_compaction(&moves, epoch)
            .expect("a compaction planned under the lock applies");
        Some(SegmentMove::Compact {
            evacuation: engine_moves(&moves),
            applied: moves,
            epoch,
        })
    }

    /// A raise to `rung` the pool cannot plan as it stands, its way holding
    /// runs of chunks that find no free run below its activation: the walk
    /// compacts below that limit until they do
    /// ([`SlotPool::raise_runs_short`]), and the raise goes once its moves
    /// plan. Whether such a compaction is plannable now.
    pub fn compaction_for_raise(&mut self, rung: u32) -> bool {
        let Some((run, pieces)) = self.pool.raise_runs_short(rung) else {
            return false;
        };
        let below = self.pool.total_chunks() - self.pool.act_chunks(rung);
        if !self.pool.compacting() && self.pool.plan_compaction_below(run, pieces, below).is_none() {
            return false;
        }
        self.compact_raise = Some(rung);
        true
    }

    /// Whether a compaction asked was answered no room since last read.
    pub fn take_compaction_answered(&mut self) -> bool {
        std::mem::take(&mut self.compaction_answered)
    }

    /// Whether moving chunks stands `pieces` free runs of `run` for a bind
    /// waiting in the planner, where no run stands now: one planned (the
    /// walk asked to make it), or one on its way. Where it is, the planner
    /// moves no resident for the bind; where none can (free chunks too few
    /// whatever moves, or no walk steers the pool), it is no answer, and
    /// the planner goes on.
    pub fn compaction_comes(&mut self, run: u32, pieces: u32) -> bool {
        // No walk steers this pool: nothing would carry a compaction out.
        if self.walk_target.is_none() || self.pool.free_chunk_runs(run) >= pieces {
            return false;
        }
        if self.compact_for.is_some() && self.pool.compacting() {
            return true;
        }
        if self.pool.plan_compaction_for(run, pieces).is_none() {
            return false;
        }
        self.compact_for = Some((run, pieces));
        true
    }

    /// Whether the compaction [`Self::compaction_comes`] asked is on its
    /// way: its moves fenced, or its runs still short and a plan for them
    /// there for the walk to make. One that no plan can make any more is
    /// not (its ask stays, read again by the next).
    #[must_use]
    pub fn compaction_under_way(&self) -> bool {
        let Some((run, pieces)) = self.compact_for else {
            return false;
        };
        self.walk_target.is_some()
            && self.pool.free_chunk_runs(run) < pieces
            && (self.pool.compacting() || self.pool.plan_compaction_for(run, pieces).is_some())
    }

    /// The engine took a compaction's copy (planned to ride the fire at
    /// `planned`) and waits it out: its chunks' fence lifts once the fires
    /// handed out by `current` retire, at once on an idle pool.
    pub fn compaction_copied(&mut self, planned: u64, current: u64) {
        self.pool.compaction_copied(planned, current);
        self.refused_moving.clear();
    }

    /// The engine refused a fire for the state in logical chunks `chunks`
    /// (`engine::Error::Placement`): the walk moves it down.
    pub fn refuse_chunks(&mut self, chunks: &[u32]) {
        self.refused.extend(chunks.iter().copied());
    }

    /// How many of `chunks` the last step found no free chunk below for:
    /// the free chunks the planner must make for them.
    #[must_use]
    pub fn refusals_stuck(&self, chunks: &[u32]) -> u32 {
        u32::try_from(chunks.iter().filter(|chunk| self.refused_stuck.contains(chunk)).count())
            .unwrap_or(u32::MAX)
    }

    /// Whether any of `chunks` still waits for its move down, or is being
    /// moved: a fire refused for it is held until none is.
    #[must_use]
    pub fn refusals_pending(&self, chunks: &[u32]) -> bool {
        chunks
            .iter()
            .any(|chunk| self.refused.contains(chunk) || self.refused_moving.contains(chunk))
    }

    /// A move of the refused chunks down below the state limit, applied
    /// here as a compaction is, for the engine to copy before the next
    /// fire (`SegmentMove::Compact`); `None` where none waits or another
    /// compaction is under way.
    fn refused_step(&mut self, epoch: u64) -> Option<SegmentMove> {
        if self.refused.is_empty() || self.pool.compacting() {
            return None;
        }
        let chunks: Vec<u32> = std::mem::take(&mut self.refused).into_iter().collect();
        let moves = self.pool.plan_moves_of(&chunks);
        // What found no free chunk below (and is still held) waits for the
        // room the planner is asked to make ([`Self::refusals_stuck`]);
        // what is no longer held is no longer waited for.
        let moved: BTreeSet<u32> = moves.iter().map(|step| step.logical).collect();
        let stuck: BTreeSet<u32> = chunks
            .iter()
            .copied()
            .filter(|chunk| !moved.contains(chunk) && self.pool.holds_chunk(*chunk))
            .collect();
        self.refused.extend(stuck.iter().copied());
        self.refused_stuck = stuck;
        if moves.is_empty() {
            return None;
        }
        self.pool
            .apply_compaction(&moves, epoch)
            .expect("a move planned under the lock applies");
        self.refused_moving = moved;
        Some(SegmentMove::Compact {
            evacuation: engine_moves(&moves),
            applied: moves,
            epoch,
        })
    }

    /// The engine refused a compaction before its copy: the pool takes it
    /// back, and no other is planned for that bind.
    pub fn compaction_refused(&mut self, moves: &[super::segments::ChunkMove]) {
        self.pool.undo_compaction(moves);
        self.compact_for = None;
        // The engine refused the move of refused chunks: they wait for the
        // next step, never dropped while a fire waits on them.
        self.refused.extend(std::mem::take(&mut self.refused_moving));
    }

    /// The walk's next grow step toward `goal` mapped segments, `None`
    /// where none maps a segment: the one place a grow is decided.
    /// [`Self::next_move`] takes it, and `SharedGrow::can_grow` tells a
    /// waiter the pool grows exactly where it is `Some`, so the planner
    /// never waits on a grow the walk does not make.
    ///
    /// The lowest uncommitted prefix segment the engine binds first. The
    /// prefix whole, the span the rung lends is backed next (span lending,
    /// [`SlotPool::span_lent`]), for a waiter short of live state as
    /// for the target alone: a fire whose rung needs it takes it back
    /// first. Not while a refused commit waits for room to be freed.
    #[must_use]
    pub fn grow_step(&self, goal: u32) -> Option<GrowStep> {
        if self.mapped_segments() >= goal {
            return None;
        }
        let prefix = self.pool.prefix_segments();
        if prefix < self.pool.top_segment() && self.pool.backs(prefix) {
            return Some(GrowStep::Commit { prefix: prefix + 1 });
        }
        if self.span_commit_blocked() {
            return None;
        }
        let n = self.pool.span_to_commit()?;
        Some(GrowStep::CommitSpan { n })
    }

    /// The segments the walk moves toward for the governor's `target`: on
    /// the way up no further than what commits ahead of what the waiters
    /// (and the cache) ask; at or under what is mapped, the target itself.
    #[must_use]
    pub fn walk_goal(&self, target: u32) -> u32 {
        let mapped = self.mapped_segments();
        if target <= mapped {
            return target;
        }
        let short = self
            .pool
            .demand()
            .wanted
            .saturating_sub(self.pool.demand().live);
        let mut ahead = self.ahead_target(short);
        // The head guest bind's runs: free chunks enough for its slots may
        // still hold no run of its length, scattered by the binds before
        // it; a fresh segment holds them whole, so the walk commits one
        // more while they do not stand free.
        if let Some((run, pieces)) = self.guest_head
            && self.pool.free_chunk_runs(run) < pieces
        {
            ahead = ahead.max(mapped + 1);
        }
        tracing::trace!(
            target,
            mapped,
            short,
            ahead,
            headroom = self.pool.headroom(),
            "segment goal"
        );
        target.min(ahead)
    }

    /// The engine took the evacuation [`Self::next_move`] planned, with
    /// fires through `current` (the kv store's per-fire epoch, CONTRACT-BC
    /// 17) handed out by then. Any of those may still read the old
    /// location; every fire after reads the new one. The copy itself the
    /// engine waits out before it gives the segment back
    /// (`Engine::release_segments`), so the segment drains once the fires
    /// through `current` retire, with no fire to wait for after them: on
    /// an idle pool, at once.
    pub fn evacuated(&mut self, current: u64) {
        self.pool.drains_through(current);
    }

    /// The engine refused `evacuation` ([`Self::next_move`]'s, applied in
    /// the pool already) before its copy: its table never changed, so the
    /// pool's goes back to it. Every moved chunk lies where it lay, and the
    /// segment is committed and mapped again; nothing of it is released.
    pub fn evacuation_refused(&mut self, evacuation: &Evacuation) {
        self.pool.undo_evacuation(evacuation);
    }

    /// A `commit_span` was refused (`true`): none is asked again until room
    /// is freed, which lifts it (`false`), the one event that can change the
    /// answer.
    pub fn block_span_commit(&mut self, blocked: bool) {
        self.span_commit_blocked = blocked;
    }

    /// The head guest bind's need, `(run, pieces)`; `None` where none waits
    /// ([`crate::store::roles::Roles::take_guest`]).
    pub fn set_guest_head(&mut self, need: Option<(u32, u32)>) {
        self.guest_head = need;
    }

    /// The walk's target, shared ([`Self::commit_can_come`]).
    pub fn walked_to(&mut self, target: Arc<std::sync::atomic::AtomicU32>) {
        self.walk_target = Some(target);
    }

    /// Whether the walk would commit a segment now: its goal (a pinned
    /// count, else the governor's target) is past what is mapped. At the
    /// target nothing commits, however many segments the pool's own
    /// limits leave uncommitted. Without a walk (a pool none steers), as
    /// the pool's limits say.
    #[must_use]
    pub fn commit_can_come(&self) -> bool {
        let Some(target) = &self.walk_target else {
            return true;
        };
        let goal = target::pinned_segments()
            .unwrap_or_else(|| target.load(std::sync::atomic::Ordering::Acquire));
        goal > self.mapped_segments()
    }

    /// The chunks the governor's target holds back from state: segments
    /// below the activation the engine backs and the pool could map, left
    /// unmapped while the walk commits nothing at the target
    /// ([`Self::commit_can_come`]). 0 where a commit can come, or every
    /// such segment is mapped. A waiter short of room these would give
    /// waits on the target, an outside event as a policy's answer is: the
    /// target's rise wakes the planner ([`target::SharedGrow`]'s retarget).
    #[must_use]
    pub fn held_back_chunks(&self) -> u32 {
        if self.commit_can_come() {
            return 0;
        }
        let pool = &self.pool;
        let segments = pool.segments();
        (0..pool.top_segment())
            .filter(|&segment| {
                segments.state(segment) == SegmentState::Uncommitted && pool.backs(segment)
            })
            .map(|segment| {
                let range = segments.range(segment);
                range.end - range.start
            })
            .sum()
    }

    /// The head guest bind's need, `(run, pieces)`, where one waits.
    #[must_use]
    pub fn guest_head(&self) -> Option<(u32, u32)> {
        self.guest_head
    }

    /// Whether a process's demand waits for room on this pool: a guest
    /// bind's head here, or (`planner`, read before this pool's lock,
    /// [`planner_demand_waits`]) a kv allocation or restore. A throughput
    /// raise (one only cuttable passes want) takes no room while one does,
    /// and the rung walk lowers toward the narrowest rung the admitted work
    /// needs, giving room back.
    #[must_use]
    pub fn demand_waits(&self, planner: bool) -> bool {
        planner || self.guest_head.is_some()
    }

    /// Whether a refused `commit_span` waits for room to be freed.
    #[must_use]
    pub fn span_commit_blocked(&self) -> bool {
        self.span_commit_blocked
    }

    /// The engine backed the top `n` span segments: the chunks of theirs the
    /// rung lends go out now.
    pub fn span_committed(&mut self, n: u32) {
        self.pool.commit_span(n);
    }

    /// The engine backed segments `0..prefix`: their chunks go out now.
    pub fn committed(&mut self, prefix: u32) {
        for segment in 0..prefix.min(self.pool.top_segment()) {
            if self.pool.segments().state(segment) == SegmentState::Uncommitted {
                self.pool
                    .commit_segment(segment)
                    .expect("an uncommitted segment commits");
            }
        }
    }

    /// The segments to commit ahead of need: enough that `bytes` (what
    /// waiters lack, plus [`COMMIT_AHEAD_BYTES`]) stand ready; the mapped
    /// target that takes, never past the pool.
    #[must_use]
    pub fn ahead_target(&self, bytes: u64) -> u32 {
        let wanted = self
            .pool
            .commit_ahead(bytes.saturating_add(COMMIT_AHEAD_BYTES));
        self.mapped_segments() + wanted.len() as u32
    }

    // ---- governor ----

    /// What is asked of the pool, in bytes: the slots held and wanted, the
    /// reserved chunks, and the top span, committed from the load.
    #[must_use]
    pub fn demand(&self) -> Demand {
        let Demand { live, wanted } = self.pool.demand();
        let fixed = self.fixed_bytes();
        Demand {
            wanted: wanted + fixed,
            live: live + fixed,
        }
    }

    /// The reserved chunks and the top span, held from the load; past it,
    /// what the programs' fire term has the activation take.
    fn fixed_bytes(&self) -> u64 {
        let act = self.pool.act_max().max(self.pool.act_reserved());
        u64::from(self.pool.reserved_chunks() + act) * self.pool.chunk_bytes()
    }

    /// The bytes of the fewest segments this pool can map, as its layout
    /// has them: every segment holding a reserved chunk (no evacuation
    /// moves them) and every segment under the activation span's chunks,
    /// the largest rung's as [`Self::floor`] counts it. However much is
    /// shed, no shrink maps fewer.
    #[must_use]
    pub fn least_mapped_bytes(&self) -> u64 {
        let segments = self.pool.segments();
        let total = self.pool.total_chunks();
        let act = self.pool.act_max().max(self.pool.act_reserved()).min(total);
        let reserved = self.pool.reserved_chunks().min(total);
        let mut least = std::collections::BTreeSet::new();
        if reserved > 0 {
            least.extend(0..=segments.of(reserved - 1));
        }
        if act > 0 {
            least.extend(segments.of(total - act)..=segments.of(total - 1));
        }
        least
            .into_iter()
            .map(|segment| u64::from(segments.len(segment)) * self.pool.chunk_bytes())
            .sum()
    }

    /// What [`Self::floor`] is made of, for the load log.
    #[must_use]
    pub fn floor_parts(&self) -> FloorParts {
        let chunk = self.pool.chunk_bytes();
        let act = self.pool.act_max().max(self.pool.act_reserved());
        FloorParts {
            lane_bytes: u64::from(self.least_chunks) * chunk,
            max_context: self.max_context,
            reserved_bytes: u64::from(self.pool.reserved_chunks()) * chunk,
            act_bytes: u64::from(act) * chunk,
            least_mapped_bytes: self.least_mapped_bytes(),
        }
    }

    /// Bytes below which one lane cannot run: one working set at the longest
    /// context, the reserved chunks and the top span.
    #[must_use]
    pub fn floor(&self) -> u64 {
        u64::from(self.least_chunks) * self.pool.chunk_bytes() + self.fixed_bytes()
    }

    #[must_use]
    pub fn hard_limit(&self) -> u64 {
        u64::from(self.pool.total_chunks()) * self.pool.chunk_bytes()
    }

    /// Bytes under the mapped segments, the top span's included.
    #[must_use]
    pub fn held(&self) -> u64 {
        let segments = self.pool.segments();
        (0..segments.count())
            .filter(|&s| segments.state(s) != SegmentState::Uncommitted)
            .map(|s| u64::from(segments.len(s)) * self.pool.chunk_bytes())
            .sum()
    }

    #[must_use]
    pub fn growth(&self) -> Growth {
        match self.growth {
            SegmentGrowth::Fixed => Growth::Fixed,
            SegmentGrowth::Bottom | SegmentGrowth::TwoEnded => Growth::Elastic,
        }
    }
}

/// The parts of a shared pool's floor: one lane's state at the longest
/// context, the reserved chunks and the activation span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloorParts {
    pub lane_bytes: u64,
    pub max_context: u32,
    pub reserved_bytes: u64,
    pub act_bytes: u64,
    /// The fewest segments the pool can map ([`SharedPool::least_mapped_bytes`]).
    pub least_mapped_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RungError {
    #[error("{0}")]
    Pool(SlotError),
}

#[cfg(test)]
pub(crate) mod tests;

pub mod programs;
mod target;
pub use target::SharedGrow;

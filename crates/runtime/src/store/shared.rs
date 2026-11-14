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
//! - the fixed shapes of roles [`SlotRole::Instance`] and
//!   [`SlotRole::Channel`] (`new-memory/STATE-ROLES.md`) are no part of a
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
//! record the engine copies `run` onto first. The copy-on-write records come
//! out of a reserve of one set per lane a fire runs, which the planner keeps
//! back from admission (`ResidencyPlanner::cow_lanes`).
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
use crate::governor::{Demand as Asked, Growth};

/// The pool behind an engine's stores, shared by every store and the
/// governor; each call takes the lock for itself, innermost of every store
/// lock.
pub type Shared = Arc<Mutex<SharedPool>>;

/// What a pool keeps committed ahead of what it has handed out, in bytes:
/// a mapping's cost on the lane against how often the lane pays it.
pub const COMMIT_AHEAD_BYTES: u64 = 64 * 1024 * 1024;

/// Engines a process drives at most, for the per-engine rung gates.
const RUNG_ENGINES: usize = 64;

/// Per engine, the tokens a fire is cut and stepped to (SPEC decision 6):
/// the activation rung the pool is at or going to, `u32::MAX` at its
/// largest. Set by the pool's walker, read by the pipeline's cuts and the
/// scheduler's steps.
static FIRE_CAP: [std::sync::atomic::AtomicU32; RUNG_ENGINES] =
    [const { std::sync::atomic::AtomicU32::new(u32::MAX) }; RUNG_ENGINES];

/// Per engine, the widest fire the scheduler may still have in flight:
/// lowered to the cap once no step past it is (the scheduler holds back a
/// launch past the cap), so the walker may lower the pool's rung behind it.
static RUNG_GRANT: [std::sync::atomic::AtomicU32; RUNG_ENGINES] =
    [const { std::sync::atomic::AtomicU32::new(u32::MAX) }; RUNG_ENGINES];

/// Per engine, the event every wait on the activation rung waits on: the
/// fire cap moved, a raise was applied, copied or refused, the rung was
/// lowered, segments were committed, or room was freed (a fire retired).
static RUNG_EVENTS: [tokio::sync::Notify; RUNG_ENGINES] =
    [const { tokio::sync::Notify::const_new() }; RUNG_ENGINES];

/// Wakes every wait on `engine`'s rung: what it waits on may have changed.
pub fn rung_changed(engine: usize) {
    if let Some(event) = RUNG_EVENTS.get(engine) {
        event.notify_waiters();
    }
}

/// The next [`rung_changed`] of `engine`, armed before the caller looks at
/// what it waits on, so a change between the look and the wait is not
/// lost: pin it, `enable` it, look, then await it.
#[must_use]
pub fn rung_watch(engine: usize) -> tokio::sync::futures::Notified<'static> {
    RUNG_EVENTS
        .get(engine)
        .unwrap_or(&RUNG_EVENTS[RUNG_ENGINES - 1])
        .notified()
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
    FIRE_CAP.get(engine).map_or(u32::MAX, |cap| {
        cap.load(std::sync::atomic::Ordering::Acquire)
    })
}

/// Sets `engine`'s cap as the walker would: for a test of what waits on it.
#[cfg(test)]
pub(crate) fn set_fire_cap_for_test(engine: usize, tokens: u32) {
    set_fire_cap(engine, tokens);
}

fn set_fire_cap(engine: usize, tokens: u32) {
    if let Some(cap) = FIRE_CAP.get(engine) {
        cap.store(tokens, std::sync::atomic::Ordering::Release);
    }
    rung_changed(engine);
}

/// The scheduler of `engine` has no step wider than `cap` in flight, and
/// submits none from here.
pub fn grant_rung(engine: usize, cap: u32) {
    if let Some(grant) = RUNG_GRANT.get(engine) {
        grant.store(cap, std::sync::atomic::Ordering::Release);
    }
}

fn rung_granted(engine: usize) -> u32 {
    RUNG_GRANT.get(engine).map_or(u32::MAX, |grant| {
        grant.load(std::sync::atomic::Ordering::Acquire)
    })
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

/// The rows past which a pass asks the activation rung for wider fires
/// whatever the cap: a decode's few rows never do.
pub const PASS_ASKS_FROM: usize = 16;

/// Whether a pass of `rows` rows, its fires `wanted` rows wide beside the
/// live lanes' decodes, asks the rung for them, fires now cut to `cap`:
/// past [`PASS_ASKS_FROM`] rows, or where the cap cuts its fire narrower
/// (a pool resting at a small rung). It asks the smallest rung that
/// covers `wanted` ([`SharedPool::rung_for`]), not past it.
#[must_use]
pub fn pass_asks(rows: usize, wanted: u32, cap: u32) -> bool {
    rows > PASS_ASKS_FROM || wanted > cap
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

/// The rung fires of `rows` take on `engine` now: the narrowest that holds
/// them, and no wider than the widest rung the engine has armed
/// ([`::engine::live`]). Every serving fire replays a captured body, so a
/// raise to a rung whose bodies are still being captured waits on them,
/// and fires serve at the rungs armed meanwhile.
#[must_use]
pub fn rung_goal(pool: &SharedPool, engine: usize, rows: u32) -> u32 {
    pool.armed_rung_for(rows, crate::engine::live(engine).armed_through())
}

/// Raises `shared`'s rung at once to the one fires of `rows` take, where
/// nothing lies in the larger span and the room is there (a raise that must
/// move state is the walker's, through the engine's copy): the pass asking
/// is cut at the raised width from its first piece. A cap the walker
/// lowered behind a rung still up is lifted back to it.
pub fn raise_now(shared: &Shared, engine: usize, rows: u32) {
    let mut pool = shared.lock().unwrap();
    let rungs = pool.rungs();
    let goal = rung_goal(&pool, engine, rows);
    let lifted = |rung: u32| {
        if rungs.last() == Some(&rung) {
            u32::MAX
        } else {
            rung
        }
    };
    if pool.pool().act_chunks(goal) > pool.pool().act_reserved() {
        let Ok(evacuation) = pool.plan_raise(goal) else {
            return;
        };
        if pool.raise_waits(&evacuation) || pool.apply_rung(&evacuation, 0).is_err() {
            return;
        }
        tracing::info!(
            engine,
            rung = goal,
            moved = 0,
            "the activation rung is raised"
        );
        rung_changed(engine);
    }
    // The rung holds `goal` now: fires may take its width, once a raise's
    // copy is the engine's.
    if pool.copying().is_some() {
        return;
    }
    let cap = lifted(goal);
    if cap > fire_cap(engine) {
        set_fire_cap(engine, cap);
    }
    drop(pool);
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
        // within a class only caps fires, never lowers or raises. A
        // program's fire term registered or dropped since moves the span a
        // rung takes ([`programs`]).
        let want = pool.pool().act_chunks(goal);
        let have = pool.pool().act_reserved();
        let lifted = if goal == largest { u32::MAX } else { goal };
        if want < have {
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
        if want > have {
            let evacuation = pool.plan_raise(goal)?;
            return Ok(RungStep::Raise {
                rung: goal,
                evacuation,
            });
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
    /// The activation set aside under the smallest rung.
    pub act_chunks: u32,
    /// The chunks the weights' pinned slots take.
    pub weight_chunks: u32,
    /// What is left for kv and state: no move frees more.
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
}

impl RoleShapes {
    /// The shape of role `role`, where the pool has it.
    #[must_use]
    pub fn of(&self, role: SlotRole) -> Option<FixedShape> {
        match role {
            SlotRole::Instance => self.instance,
            SlotRole::Channel => self.channel,
            SlotRole::State | SlotRole::RsBuffer => None,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instance.is_none() && self.channel.is_none()
    }

    fn iter(&self) -> impl Iterator<Item = &FixedShape> {
        self.instance.iter().chain(&self.channel)
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
        let disk_slots = list.map_or(0, |list| {
            u32::try_from(u64::from(facts.disk_chunks) * u64::from(list.slots_per_chunk))
                .unwrap_or(u32::MAX)
        });
        Ok(Self { host, disk_slots })
    }
}

/// One step toward a segment target, as [`SharedPool::next_move`] plans it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentMove {
    /// Back segments `0..prefix` (`commit_segments(prefix)`); the pool hands
    /// out their chunks once the engine has, [`SharedPool::committed`].
    Commit { prefix: u32 },
    /// Copy `evacuation` (applied in the pool already), ahead of the next
    /// fire; [`SharedPool::moved`] once the engine has it.
    Evacuate { evacuation: EngineEvacuation },
    /// Give back segments `prefix..` (`release_segments(prefix)`), released
    /// in the pool already.
    Release { prefix: u32 },
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
    /// A raise is applied here and its chunks' copy is not yet the
    /// engine's: no fire is let past the rung held before it
    /// ([`Self::raise_applied`]).
    copying: Option<u32>,
    /// The activation set aside before the raise whose copy is the
    /// engine's to take: what a refused raise goes back to.
    raise_from: Option<u32>,
    /// The registered programs' fire terms ([`programs`]).
    programs: programs::ProgramFires,
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
    pub fn tier(facts: &ChunkFacts, chunks: u32) -> Result<Option<Self>, SharedError> {
        if chunks == 0 || facts.shapes.is_empty() {
            return Ok(None);
        }
        if let Some(shape) = facts.shapes.iter().find(|shape| shape.pinned.is_some()) {
            return Err(SharedError::Pinned(shape.id.0));
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
        let (mut instances, mut channels) = (Vec::new(), Vec::new());
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
            },
            weights_first,
            records: Vec::new(),
            free_records: BTreeSet::new(),
            records_most: 0,
            least_chunks: 0,
            copying: None,
            raise_from: None,
            programs: programs::ProgramFires::default(),
        };
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

    /// The fixed shapes taken on demand: an instance's tables, a ring.
    #[must_use]
    pub fn roles(&self) -> RoleShapes {
        self.roles
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
    /// for kv and state. No eviction reaches below that.
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
        let act_chunks = pool.act_chunks(0);
        let state_chunks = pool
            .total_chunks()
            .saturating_sub(pool.reserved_chunks())
            .saturating_sub(act_chunks)
            .saturating_sub(weight_chunks);
        PoolRoom {
            chunk_bytes: pool.chunk_bytes(),
            total_chunks: pool.total_chunks(),
            reserved_chunks: pool.reserved_chunks(),
            act_chunks,
            weight_chunks,
            state_chunks,
            list_slots: self.list.map_or(0, |list| {
                u32::try_from(u64::from(state_chunks) * u64::from(list.slots_per_chunk))
                    .unwrap_or(u32::MAX)
            }),
            list_tokens: self.list.map_or(0, |list| list.tokens),
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

    /// The chunks state may take under the current rung: every chunk but
    /// the reserved ones and the activation set aside now.
    #[must_use]
    pub fn state_chunks_now(&self) -> u32 {
        self.pool
            .total_chunks()
            .saturating_sub(self.pool.reserved_chunks())
            .saturating_sub(self.pool.act_reserved())
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
    pub fn raise_applied(&mut self, evacuation: &RungEvacuation) -> Result<(), RungError> {
        let before = self.pool.rung();
        let act = self.pool.act_reserved();
        if !self.raise_waits(evacuation) {
            return self.apply_rung(evacuation, 0);
        }
        self.apply_rung(evacuation, UNTIL_COPIED)?;
        self.copying = Some(self.copying.map_or(before, |held| held.min(before)));
        self.raise_from = Some(self.raise_from.map_or(act, |held| held.min(act)));
        Ok(())
    }

    /// Whether `evacuation`'s raise goes through the engine before fires
    /// take it: it moves state, which the engine copies first, or the span
    /// is backed on demand, where the engine backs its carve for the wider
    /// rung first (`Engine::set_rung`). With the span backed from the load,
    /// a move-free raise makes no lane trip.
    #[must_use]
    pub fn raise_waits(&self, evacuation: &RungEvacuation) -> bool {
        !evacuation.moves.is_empty() || self.pool.span_on_demand()
    }

    /// The cap fires start at: the rung the pool starts at, lifted. The
    /// largest's (no cap) unless the span is backed on demand, where fires
    /// start cut to the smallest rung until a raise reaches the engine.
    #[must_use]
    pub fn start_cap(&self) -> u32 {
        self.lifted(self.rung())
    }

    /// Whether every rung change is told the engine (`Engine::set_rung`):
    /// the span is backed on demand.
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
        self.copying = None;
    }

    /// The engine has taken a raise's copy, carried by the fire at `epoch`
    /// (the next it is handed): its chunks moved off reopen once that fire
    /// retires, and fires may take the raised rung.
    pub fn raise_copied(&mut self, epoch: u64) {
        self.pool.refence(UNTIL_COPIED, epoch);
        self.copying = None;
        self.raise_from = None;
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

    /// The pool shape of engine shape `engine`, a state shape's.
    fn pool_shape(&self, engine: EngineShape) -> Option<ShapeId> {
        self.list
            .iter()
            .chain(&self.rs_list)
            .chain(&self.windows)
            .find(|shape| shape.id == engine)
            .map(|shape| shape.pool)
            .or_else(|| {
                self.fixed
                    .iter()
                    .find(|shape| shape.id == engine)
                    .map(|shape| shape.pool)
            })
    }

    /// Whether every slot of a lease lies below the activation span: kv
    /// list refs `pages`, then a record's pieces `records` (fixed shapes
    /// in order). A lowered rung lends the span's chunks to state, and a
    /// raise takes them back while a step's capture may run on them.
    #[must_use]
    pub fn below_span(&self, pages: &[u32], records: &[u32]) -> bool {
        let span = self.pool.total_chunks() - self.pool.act_max();
        let under = |shape: ShapeId, slot_ref: u32| {
            let chunk = self.slot(shape, slot_ref).chunk;
            self.pool
                .physical(chunk)
                .is_some_and(|physical| physical < span)
        };
        let list = self.list.map(|list| list.pool);
        let pages_under = list.is_none_or(|list| pages.iter().all(|&r| under(list, r)));
        let mut pieces = records.iter();
        let records_under = self.fixed.iter().all(|shape| {
            (0..shape.pieces).all(|_| pieces.next().is_none_or(|&r| under(shape.pool, r)))
        });
        pages_under && records_under
    }

    /// The engine holds `bytes` of the range it stated back
    /// ([`::engine::live::Live::held`]): no state chunk is opened in the
    /// top `bytes` of the span below the activation, which the engine binds
    /// none of yet; all of it again once it holds none.
    /// Whether the ceiling rose: room opened that a planner waiter may be
    /// short of, so the caller wakes them.
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

    /// The most the pool may map while the engine holds `bytes` back: the
    /// prefix segments wholly under what it binds
    /// ([`crate::store::slots::SlotPool::backs`]) and the top span,
    /// committed from the load; the whole range when it holds none.
    #[must_use]
    pub fn bound_bytes(&self, bytes: u64) -> u64 {
        let total = self.pool.total_chunks();
        if bytes == 0 {
            return u64::from(total) * self.pool.chunk_bytes();
        }
        let seg = self.pool.segments().segment_chunks().max(1);
        let top_start = self.pool.top_segment() * seg;
        let prefix_end = (self.ceiling_for(bytes) / seg * seg).min(top_start);
        u64::from(prefix_end + (total - top_start)) * self.pool.chunk_bytes()
    }

    /// [`Self::rung_for`], no wider than `armed` (a rung's tokens): the
    /// widest rung armed at or under it, else the smallest.
    #[must_use]
    pub fn armed_rung_for(&self, rows: u32, armed: u32) -> u32 {
        let goal = self.rung_for(rows);
        if goal <= armed {
            return goal;
        }
        let rungs = self.pool.rungs();
        rungs
            .iter()
            .copied()
            .filter(|&rung| rung <= armed)
            .max()
            .or_else(|| rungs.first().copied())
            .unwrap_or(goal)
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
    /// span is backed on demand, the notice the engine is sent
    /// (`Engine::set_rung(from, to)`), after the walk's retire fence: every
    /// fire wider than `rung` has retired, so it may give back the carve
    /// above it.
    pub fn lower_rung(&mut self, rung: u32) -> Result<Option<(u32, u32)>, RungError> {
        if self.pool.act_chunks(rung) > self.pool.act_reserved() {
            return Err(RungError::Pool(SlotError::Occupied { rung, chunks: 0 }));
        }
        let from = self.pool.rung();
        self.pool.set_rung(rung).map_err(RungError::Pool)?;
        Ok(self.tells_rungs().then_some((from, rung)))
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
        let mapped = self.mapped_segments();
        if mapped < target && prefix < top && self.pool.backs(prefix) {
            return Some(SegmentMove::Commit { prefix: prefix + 1 });
        }
        // The prefix is whole: the span a lowered rung lends is backed next,
        // before state opens a chunk there.
        if mapped < target
            && let Some(n) = self.pool.span_to_commit()
        {
            return Some(SegmentMove::CommitSpan { n });
        }
        if mapped <= target || prefix == 0 {
            return None;
        }
        let last = prefix - 1;
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
                self.apply_evacuation(&evacuation, epoch);
                Some(SegmentMove::Evacuate {
                    evacuation: engine_moves(&evacuation.moves),
                })
            }
            SegmentState::Uncommitted => None,
        }
    }

    /// The engine took the evacuation [`Self::next_move`] planned, with
    /// fires through `current` (the kv store's per-fire epoch, CONTRACT-BC
    /// 17) handed out by then. Any of those may run ahead of the copy and
    /// read the old location; the first fire after them runs after it. The
    /// segment drains until that one, `current + 1`, retires.
    pub fn evacuated(&mut self, current: u64) {
        self.pool.fence_drains(current.saturating_add(1));
    }

    fn apply_evacuation(&mut self, evacuation: &Evacuation, epoch: u64) {
        self.pool
            .apply_evacuation(evacuation, epoch)
            .expect("an evacuation planned under the lock applies");
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
    pub fn demand(&self) -> Asked {
        let Demand { live, wanted } = self.pool.demand();
        let fixed = self.fixed_bytes();
        Asked {
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
            SegmentGrowth::Bottom | SegmentGrowth::TwoEnded => Growth::Elastic {
                least_cut: self.segment_bytes(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RungError {
    #[error("{0}")]
    Pool(SlotError),
}

#[cfg(test)]
mod tests;

pub mod programs;
mod target;
pub use target::SharedGrow;

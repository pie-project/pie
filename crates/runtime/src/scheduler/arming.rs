//! The lane time and the slots an engine arms its rungs with (REDESIGN 2.5,
//! [`::engine::Engine::arm_step`]).
//!
//! The engine lane gives arming one composition at a time: while the lane
//! is idle, and between fires while a raise waits on a rung not armed. A
//! raise gets one composition between every two requests the lane serves,
//! so it cannot starve under any load. Each step is lent the room the
//! engine stated ([`::engine::Engine::arming_room`]) from free slots, and
//! gives it back when it returns; with no room there is no step. Idle, the
//! room is taken only where it stands free; a waiting raise funds it as a
//! planner waiter under its pass's process ([`fund`]), so PLEX frees it.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use ::engine::live::{ArmingLease, ArmingRoom, Live};

use crate::engine::EngineBox;
use crate::planner::{AllocationGrant, Demand};

/// What the lane of each engine arms with: its room, while it arms, and
/// the room a raise funded for its next step.
#[derive(Default)]
struct Engine {
    room: ArmingRoom,
    arming: bool,
    funded: Option<AllocationGrant>,
}

static ENGINES: LazyLock<Mutex<HashMap<usize, Engine>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The planner's ask that lends `room`.
#[must_use]
pub fn demand(room: ArmingRoom) -> Demand {
    Demand {
        kv_pages: room.pages,
        // `records` counts one record's pieces: one record lends them all.
        rs: crate::store::rs::RsUnits::of(u32::from(room.records > 0), 0),
    }
}

/// The room a raise on `engine` must fund before the lane can arm the rung
/// that holds `rows`: `None` where it needs none (every rung that holds
/// them armed, arming over, a step's room funded and not yet used, or an
/// engine whose arming writes no slot).
#[must_use]
pub fn unfunded(engine: usize, rows: u32) -> Option<ArmingRoom> {
    if rows <= crate::engine::live(engine).armed_through() {
        return None;
    }
    let engines = ENGINES.lock().unwrap();
    let state = engines.get(&engine)?;
    let empty = state.room == ArmingRoom::default();
    (state.arming && !empty && state.funded.is_none()).then_some(state.room)
}

/// A raise's grant of `engine`'s arming room, lent to the lane's next step.
/// Given back at once where the lane no longer arms or holds one.
pub fn fund(engine: usize, grant: AllocationGrant) {
    let mut engines = ENGINES.lock().unwrap();
    if let Some(state) = engines.get_mut(&engine)
        && state.arming
        && state.funded.is_none()
    {
        state.funded = Some(grant);
    }
}

fn take_funded(engine: usize) -> Option<AllocationGrant> {
    ENGINES
        .lock()
        .unwrap()
        .get_mut(&engine)
        .and_then(|state| state.funded.take())
}

fn set(engine: usize, room: ArmingRoom, arming: bool) {
    let mut engines = ENGINES.lock().unwrap();
    let state = engines.entry(engine).or_default();
    state.room = room;
    state.arming = arming;
    if !arming {
        state.funded = None;
    }
}

/// The slots one step is lent; they go back to free when it drops.
pub(crate) struct Lent {
    pub(crate) lease: ArmingLease,
    _held: Option<AllocationGrant>,
}

impl Lent {
    /// `grant`'s slots as the engine names them: kv pages are list slot
    /// refs as a lane's are; a record is named by its pieces' refs, through
    /// the store that handed it out (`rs`), never by its store id.
    fn of(grant: AllocationGrant, rs: Option<&crate::store::rs::RsStore>) -> Self {
        let records = match rs {
            Some(rs) => rs.device_refs(grant.rs_slots()),
            None => grant.rs_slots().iter().map(|slot| slot.0).collect(),
        };
        Self {
            lease: ArmingLease {
                pages: grant.kv_pages().iter().map(|page| page.0).collect(),
                records,
            },
            _held: Some(grant),
        }
    }

    /// [`Self::of`] through engine `engine`'s registered stores.
    fn on_engine(grant: AllocationGrant, engine: usize) -> Self {
        match crate::store::registry::try_get(0, engine) {
            Some(stores) => {
                let rs = stores.rs.lock().unwrap();
                Self::of(grant, Some(&rs))
            }
            None => Self::of(grant, None),
        }
    }

    fn empty() -> Self {
        Self {
            lease: ArmingLease::default(),
            _held: None,
        }
    }
}

/// Where a step's room comes from.
pub(crate) trait Lender: Send {
    /// `room` of free slots, `idle` when no raise waits; `None`: no step.
    fn lend(&mut self, engine: usize, room: ArmingRoom, idle: bool) -> Option<Lent>;
}

/// The runtime's: a raise's funded grant first, else what stands free on
/// the engine's planner.
pub(crate) struct PlannerLender;

impl Lender for PlannerLender {
    fn lend(&mut self, engine: usize, room: ArmingRoom, idle: bool) -> Option<Lent> {
        let _ = idle;
        if room == ArmingRoom::default() {
            return Some(Lent::empty());
        }
        let shared = crate::store::registry::try_get(0, engine).and_then(|s| s.shared);
        let lent = match take_funded(engine) {
            Some(grant) => Lent::on_engine(grant, engine),
            None => {
                // What the engine holds back now is no lease's either.
                if let Some(shared) = &shared {
                    let held = crate::engine::live(engine).held();
                    shared.lock().unwrap().hold_back(held);
                }
                let planner = crate::planner::planner_for(0, engine)?;
                planner
                    .try_free(demand(room))
                    .map(|grant| Lent::on_engine(grant, engine))?
            }
        };
        // Never the activation span's chunks a lowered rung lends state: a
        // raise takes them back, and the step's own capture runs on them.
        // A lease that reaches them goes back whole, and no step runs.
        let below = shared.is_none_or(|shared| {
            shared
                .lock()
                .unwrap()
                .below_span(&lent.lease.pages, &lent.lease.records)
        });
        below.then_some(lent)
    }
}

/// A lender over one planner: what the lane's tests lend from.
#[cfg(test)]
pub(crate) struct OnPlanner(pub std::sync::Arc<crate::planner::ResidencyPlanner>);

#[cfg(test)]
impl Lender for OnPlanner {
    fn lend(&mut self, engine: usize, room: ArmingRoom, idle: bool) -> Option<Lent> {
        let _ = (engine, idle);
        self.0
            .try_free(demand(room))
            .map(|grant| Lent::of(grant, None))
    }
}

/// What one [`Arming::step`] came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    /// A composition captured; more remain.
    Stepped,
    /// Its room is not free (or not yet looked for again): call later.
    Waiting,
    /// Arming is over: armed, faulted, or no engine.
    Over,
}

/// The lane time a gap between two requests gives a waiting raise's steps
/// past the first one it runs.
pub(crate) const GAP_BUDGET: std::time::Duration = std::time::Duration::from_millis(20);

/// How long a step short of room waits before its room is looked for
/// again, so an idle lane does not take the planner's lock each spin.
const ROOM_RETRY: std::time::Duration = std::time::Duration::from_millis(1);

enum State {
    Arming,
    /// `arm_step` answered `false`: never called again for this load.
    Done,
    /// `arm_step` failed: a load fault every later fire fails with.
    Faulted(String),
}

/// The lane's arming of one engine's rungs.
pub(crate) struct Arming {
    engine_idx: usize,
    live: Live,
    room: ArmingRoom,
    state: State,
    lender: Box<dyn Lender>,
    /// No room was free: not looked for again before then.
    retry_at: Option<crate::rt::Instant>,
    /// While the lane serves, no step before then: a step's time is owed
    /// back to the requests as much again, so arming takes at most half of
    /// a busy lane however long one composition's capture runs.
    owed_until: Option<crate::rt::Instant>,
}

impl Arming {
    pub(crate) fn new(
        engine_idx: usize,
        engine: Option<&EngineBox>,
        lender: Box<dyn Lender>,
    ) -> Self {
        let live = engine.map_or_else(Live::armed, |engine| engine.live());
        let room = engine.map_or_else(ArmingRoom::default, |engine| engine.arming_room());
        // Every rung armed at load: the lane never asks (engines that do
        // not arm in lane time serve exactly as before).
        let state = if live.armed_through() == u32::MAX {
            State::Done
        } else {
            State::Arming
        };
        let arming = matches!(state, State::Arming);
        if arming {
            set(engine_idx, room, true);
            tracing::info!(
                engine = engine_idx,
                armed_through = live.armed_through(),
                pages = room.pages,
                records = room.records,
                "the lane arms the rest of the rungs, a composition at a time"
            );
        }
        Self {
            engine_idx,
            live,
            room,
            state,
            lender,
            retry_at: None,
            owed_until: None,
        }
    }

    pub(crate) fn fault(&self) -> Option<&str> {
        match &self.state {
            State::Faulted(why) => Some(why),
            _ => None,
        }
    }

    /// A pass waits on a rung above the widest armed: rungs are the
    /// narrowest that hold the rows asked, so it waits iff it asks more
    /// rows than the widest armed holds.
    pub(crate) fn raise_waits(&self) -> bool {
        matches!(self.state, State::Arming)
            && crate::store::shared::wanted_rows(self.engine_idx) > self.live.armed_through()
    }

    /// No raise waits: room a raise funded and the lane did not use goes
    /// back to free.
    pub(crate) fn unfund(&self) {
        if matches!(self.state, State::Arming) {
            drop(take_funded(self.engine_idx));
        }
    }

    /// One composition, while any remain and its room can be lent.
    /// `busy`: the lane has work on hand (a frame in flight, a raise
    /// waiting); arming then takes half its time at most.
    pub(crate) fn step(&mut self, engine: &mut Option<EngineBox>, idle: bool, busy: bool) -> Step {
        if !matches!(self.state, State::Arming) {
            return Step::Over;
        }
        let Some(engine) = engine.as_mut() else {
            self.finish(State::Done);
            return Step::Over;
        };
        let now = crate::rt::Instant::now();
        if self.retry_at.is_some_and(|at| now < at) {
            return Step::Waiting;
        }
        // A busy lane gives arming half its time at most.
        if busy && self.owed_until.is_some_and(|at| now < at) {
            return Step::Waiting;
        }
        let Some(lent) = self.lender.lend(self.engine_idx, self.room, idle) else {
            self.retry_at = Some(crate::rt::Instant::now() + ROOM_RETRY);
            return Step::Waiting;
        };
        self.retry_at = None;
        tracing::debug!(
            engine = self.engine_idx,
            idle,
            pages = ?lent.lease.pages,
            records = ?lent.lease.records,
            "an arming step is lent"
        );
        let armed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.arm_step(&lent.lease)
        }));
        // The room goes back to free before anything else runs.
        drop(lent);
        let after = crate::rt::Instant::now();
        let took = after.saturating_duration_since(now);
        self.owed_until = busy.then(|| after + took);
        match armed {
            Ok(Ok(true)) => Step::Stepped,
            Ok(Ok(false)) => {
                tracing::info!(
                    engine = self.engine_idx,
                    armed_through = self.live.armed_through(),
                    "every rung's bodies are armed"
                );
                self.finish(State::Done);
                Step::Over
            }
            Ok(Err(error)) => self.faulted(format!("arming a rung's bodies failed: {error}")),
            Err(_) => self.faulted("arming a rung's bodies panicked".to_string()),
        }
    }

    fn faulted(&mut self, why: String) -> Step {
        tracing::error!(engine = self.engine_idx, %why, "the engine's load is faulted");
        self.finish(State::Faulted(why));
        Step::Over
    }

    fn finish(&mut self, state: State) {
        self.state = state;
        set(self.engine_idx, self.room, false);
    }
}

impl Drop for Arming {
    fn drop(&mut self) {
        if matches!(self.state, State::Arming) {
            set(self.engine_idx, self.room, false);
        }
    }
}

#[cfg(test)]
mod tests;

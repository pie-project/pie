//! The loop, on a pool: read the system, move the target.

use super::steered::{self, Shared, Steered};
use super::{Governor, Growth, Limits, Mapped, PoolTarget, Reader, Retarget, Signal, TICK, Yield};

/// Steers `pool`, an engine's: one that does not yield, or cannot move,
/// keeps its size, but its loop still ticks, since the weight cache in
/// front of it settles its windows on the governor's tick. Where the engine
/// states a reader of the host's room (`room`, its share of the host's one
/// ledger), every pool that can move is held within it.
/// `source` is the pressure source the load's yield needs
/// ([`super::source_for`]), read where the pool moves.
pub fn start(pool: Shared<Steered>, yields: Yield, room: Option<PoolRoom>, source: Option<super::Source>) {
    crate::rt::spawn(watch(pool, yields, room, source));
}

/// Whether `yield` moves `pool` at all: one its engine states can shrink,
/// and shared. XLA's and wgpu's are laid whole at load.
#[must_use]
pub fn yield_applies(pool: &Shared<Steered>) -> bool {
    pool.growth() != Growth::Fixed && pool.with(|pool| pool.inner().is_shared())
}

/// One host's room, shared by every pool on it. A reading is what the host
/// can still give the process now, which already counts what each pool
/// holds; what it does not count is what a pool was granted and has not yet
/// taken, or was told to give back and has not yet given. So each pool's
/// grant (its ceiling less what it held, negative where it must shrink) is
/// debited from the reading before the next pool reads it, under one lock:
/// N pools together are granted the host's room once, not N times. Once a
/// pool has moved toward its ceiling, its grant is settled against what it
/// holds then, which the host's next reading already counts.
pub struct HostShare {
    reader: ::engine::HostRoom,
    grants: std::sync::Mutex<Vec<Option<Grant>>>,
}

/// One pool's standing in its host's ledger.
#[derive(Clone, Copy)]
struct Grant {
    ceiling: u64,
    /// What it was granted and has not taken (negative: has not given
    /// back), as of its last settling.
    outstanding: i128,
}

impl HostShare {
    #[must_use]
    pub fn new(reader: ::engine::HostRoom) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            reader,
            grants: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// A pool's share of this host's room.
    #[must_use]
    pub fn pool(self: &std::sync::Arc<Self>) -> PoolRoom {
        let mut grants = self
            .grants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        grants.push(Some(Grant {
            ceiling: 0,
            outstanding: 0,
        }));
        PoolRoom {
            share: self.clone(),
            slot: grants.len() - 1,
        }
    }
}

/// One pool's share of its host's room ([`HostShare`]).
pub struct PoolRoom {
    share: std::sync::Arc<HostShare>,
    slot: usize,
}

impl PoolRoom {
    /// A pool alone on its host.
    #[must_use]
    pub fn alone(reader: ::engine::HostRoom) -> Self {
        HostShare::new(reader).pool()
    }

    /// Reads the host's room less what the other pools were granted and have
    /// not taken (or must still give back), lets `ceiling_of` turn it into
    /// this pool's ceiling, and records this pool's grant before the lock
    /// is let go.
    fn grant(
        &self,
        held: u64,
        ceiling_of: impl FnOnce(std::result::Result<i64, String>) -> u64,
    ) -> u64 {
        let mut grants = self
            .share
            .grants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let others: i128 = grants
            .iter()
            .enumerate()
            .filter(|(at, _)| *at != self.slot)
            .filter_map(|(_, grant)| grant.map(|grant| grant.outstanding))
            .sum();
        let read = (self.share.reader)().map(|room| {
            let left = i128::from(room) - others;
            i64::try_from(left).unwrap_or(if left < 0 { i64::MIN } else { i64::MAX })
        });
        let ceiling = ceiling_of(read);
        grants[self.slot] = Some(Grant {
            ceiling,
            outstanding: i128::from(ceiling) - i128::from(held),
        });
        ceiling
    }

    /// Settles this pool's grant against what it holds once it has moved.
    fn settle(&self, held: u64) {
        let mut grants = self
            .share
            .grants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(grant) = grants[self.slot].as_mut() {
            grant.outstanding = i128::from(grant.ceiling) - i128::from(held);
        }
    }
}

impl Drop for PoolRoom {
    fn drop(&mut self) {
        let mut grants = self
            .share
            .grants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        grants[self.slot] = None;
    }
}

/// One governor on one pool, a turn at a time: the loop without its clock,
/// so a test can drive it.
pub struct Steer {
    governor: Governor,
    /// The segments held short of the target, while a shrink has no room.
    short: Option<u64>,
    /// The pool's share of what the host can still give the process, for a
    /// pool that can move.
    room: Option<PoolRoom>,
    /// The most the pool may hold, as the last tick read the host's room:
    /// what the pool held then plus the room (less, where the host is short),
    /// within the floor and the load's hard limit. A clamp on the governor's
    /// target, whatever the yield: the host's room is a fact of the
    /// platform, not pressure to yield to.
    ceiling: Option<u64>,
    /// The regime and segment count the log last named.
    noted: Option<(Regime, u64)>,
    /// Whether the pool held its hard limit at the last turn that moved it,
    /// so reaching it and leaving it are each named once.
    at_limit: Option<bool>,
    /// Whether the yield's target stood under what the pool held at the
    /// last turn, so each crossing is named once, never a tick.
    under: Option<bool>,
    /// The pressure reading's source, for that line.
    source: &'static str,
}

/// Where the host's room leaves a pool: what the log names at info when it
/// changes, the ceiling's moves within one regime being only at debug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// The room exceeds what the pool may ever hold: held to the hard limit.
    AtLimit,
    /// The pool may grow into the room.
    Room,
    /// The ceiling is under what the pool holds: it must give some back.
    Short,
    /// The room leaves the pool only its floor.
    AtFloor,
    /// The room cannot be read: the pool holds the last good ceiling, no
    /// higher than what it holds, and grows no further.
    Unreadable,
}

impl Regime {
    const fn of(ceiling: u64, held: u64, limits: &Limits) -> Self {
        if ceiling <= limits.floor {
            Regime::AtFloor
        } else if ceiling >= limits.hard_limit {
            Regime::AtLimit
        } else if ceiling < held {
            Regime::Short
        } else {
            Regime::Room
        }
    }
}

impl Steer {
    /// A fixed pool has nothing to give back, so no yield applies to it
    /// whatever the deployment asked: the load says so once, where it asked
    /// for one.
    #[must_use]
    pub fn new(pool: &impl PoolTarget, yields: Yield) -> Self {
        let yields = if pool.growth() == Growth::Fixed {
            if yields.yields() {
                tracing::info!(
                    yields = yields.share(),
                    "the pool is laid whole at load; engine.memory.yield = {} is a no-op here",
                    yields.share()
                );
            }
            Yield::NONE
        } else {
            yields
        };
        Self {
            governor: Governor::new(Limits::of(pool), yields),
            short: None,
            room: None,
            ceiling: None,
            noted: None,
            at_limit: None,
            under: None,
            source: "none read",
        }
    }

    /// Holds the pool within the host's room as `room` reads it each tick.
    /// A fixed pool cannot move, so it takes no reader.
    #[must_use]
    pub fn with_room(mut self, pool: &impl PoolTarget, room: Option<PoolRoom>) -> Self {
        self.room = room.filter(|_| pool.growth() != Growth::Fixed);
        self
    }

    /// Names the pressure reading's source in the crossing line.
    #[must_use]
    pub const fn with_source(mut self, source: &'static str) -> Self {
        self.source = source;
        self
    }

    /// Whether the yield's target stood under what the pool held at the
    /// last turn.
    #[must_use]
    pub const fn under(&self) -> Option<bool> {
        self.under
    }

    /// Names, at info, the yield's target crossing what the pool holds: an
    /// edge each way, never a tick.
    fn note_target(&mut self, pressure: f64, target: u64, held: u64) {
        let under = target < held;
        if self.under == Some(under) {
            return;
        }
        let first = self.under.is_none();
        self.under = Some(under);
        if first && !under {
            return;
        }
        let limits = self.governor.limits();
        let (floor, budget) = (limits.floor, limits.hard_limit);
        if under {
            tracing::info!(
                pressure,
                source = self.source,
                target_mib = target >> 20,
                held_mib = held >> 20,
                floor_mib = floor >> 20,
                budget_mib = budget >> 20,
                "the system's memory pressure brings the device target under what the pool holds: it gives the difference back"
            );
        } else {
            tracing::info!(
                pressure,
                source = self.source,
                target_mib = target >> 20,
                held_mib = held >> 20,
                floor_mib = floor >> 20,
                budget_mib = budget >> 20,
                "the device target is at or over what the pool holds again"
            );
        }
    }

    /// Whether the pool is held within the host's room.
    #[must_use]
    pub const fn has_room(&self) -> bool {
        self.room.is_some()
    }

    /// The ceiling the last tick's reading of the host's room set.
    #[must_use]
    pub const fn ceiling(&self) -> Option<u64> {
        self.ceiling
    }

    /// Reads the host's room: the pool may hold what it holds now plus its
    /// share of it, or less by what the host is short.
    ///
    /// A reading that cannot be taken is named once and holds the last good
    /// ceiling, no higher than what the pool holds: nothing grows on a
    /// reading it does not have, and nothing is shed for want of one.
    ///
    /// Only a change of [`Regime`] is named at info (the first reading
    /// among them); a ceiling that crosses a segment within one regime is
    /// named at debug, and a tick that crosses none names nothing.
    fn read_room(&mut self, pool: &impl PoolTarget) {
        let Some(room) = &self.room else {
            return;
        };
        let limits = *self.governor.limits();
        let held = pool.held();
        let last = self.ceiling;
        let mut read = None;
        let mut failed = None;
        let ceiling = room.grant(held, |reading| match reading {
            Ok(room) => {
                read = Some(room);
                let ceiling = i128::from(held) + i128::from(room);
                u64::try_from(ceiling.max(0))
                    .unwrap_or(u64::MAX)
                    .clamp(limits.floor, limits.hard_limit)
            }
            Err(why) => {
                failed = Some(why);
                last.map_or(held, |last| last.min(held))
                    .clamp(limits.floor, limits.hard_limit)
            }
        });
        self.ceiling = Some(ceiling);
        let regime = if failed.is_some() {
            Regime::Unreadable
        } else {
            Regime::of(ceiling, held, &limits)
        };
        let segments = limits.segments(ceiling);
        let was = self.noted.map(|(regime, _)| regime);
        if was == Some(regime) && self.noted.map(|(_, at)| at) == Some(segments) {
            return;
        }
        self.noted = Some((regime, segments));
        if let Some(why) = failed {
            if was != Some(Regime::Unreadable) {
                tracing::error!(
                    %why,
                    ceiling_mib = ceiling >> 20,
                    held_mib = held >> 20,
                    "the host's room could not be read: the pool holds what it has and grows no further until it can"
                );
            }
            return;
        }
        let room = read.unwrap_or(0);
        if was == Some(regime) {
            tracing::debug!(
                ?regime,
                ceiling_mib = ceiling >> 20,
                held_mib = held >> 20,
                room_mib = room >> 20,
                "the pool's ceiling follows the host's room"
            );
        } else {
            tracing::info!(
                ?regime,
                ceiling_mib = ceiling >> 20,
                held_mib = held >> 20,
                room_mib = room >> 20,
                floor_mib = limits.floor >> 20,
                hard_limit_mib = limits.hard_limit >> 20,
                "the pool's ceiling follows the host's room"
            );
        }
    }

    /// Names, at info, the pool reaching its hard limit and leaving it: an
    /// edge each, never a tick. A pool that starts under it names nothing
    /// until it reaches it; one that starts at it says so once.
    fn note_limit(&mut self, held: u64, limits: &Limits) {
        let full = held >= limits.hard_limit;
        if self.at_limit == Some(full) || (self.at_limit.is_none() && !full) {
            self.at_limit = Some(full);
            return;
        }
        self.at_limit = Some(full);
        if full {
            tracing::info!(
                held_mib = held >> 20,
                hard_limit_mib = limits.hard_limit >> 20,
                "the pool holds its hard limit"
            );
        } else {
            tracing::info!(
                held_mib = held >> 20,
                hard_limit_mib = limits.hard_limit >> 20,
                "the pool is under its hard limit again"
            );
        }
    }

    /// Whether the pool held its hard limit at its last turn.
    #[must_use]
    pub const fn at_limit(&self) -> Option<bool> {
        self.at_limit
    }

    /// The regime the log last named.
    #[must_use]
    pub fn regime(&self) -> Option<Regime> {
        self.noted.map(|(regime, _)| regime)
    }

    #[must_use]
    pub const fn governor(&self) -> &Governor {
        &self.governor
    }

    /// One turn on one reading of the system's memory pressure, in `0..=1`
    /// (`0` where it is not read). A pool that does not yield keeps its
    /// size but for the host's room; its tick still comes.
    pub fn turn(
        &mut self,
        pool: &mut impl PoolTarget,
        pressure: f64,
        signal: &mut impl FnMut(Signal),
    ) -> Turn {
        self.read_room(pool);
        let yields = self.governor.yields().yields();
        if !yields && self.ceiling.is_none() {
            pool.ticked();
            return Turn {
                target: self.governor.target(),
                segments: within(self.governor.limits(), self.governor.target()),
                mapped: pool.mapped(),
                answer: Retarget::Reached,
            };
        }
        // The system refusing memory is its own word that it has none to
        // give: full pressure, for a pool that yields to it.
        let pressure = if yields && pool.take_refused() {
            1.0
        } else {
            pressure
        };
        let held_before = pool.held();
        let target = self.governor.tick(pressure);
        if yields {
            self.note_target(pressure, target, held_before);
        }
        let target = self.ceiling.map_or(target, |ceiling| target.min(ceiling));
        let limits = *self.governor.limits();
        let segments = within(&limits, target);
        let answer = pool.retarget(segments);
        let mapped = pool.mapped();
        match answer {
            Retarget::Reached | Retarget::Moving { .. } => {
                if self.short.take().is_some() {
                    signal(Signal::Eased);
                }
            }
            Retarget::NoRoom { held } => {
                if self.short.is_none() {
                    // One measure with the read and crossing lines: the
                    // bytes held and the rule's target; segments apart.
                    tracing::info!(
                        held_mib = pool.held() >> 20,
                        target_mib = target >> 20,
                        held_segments = held,
                        target_segments = segments,
                        draining_segments = mapped.draining,
                        "the pool maps more than its target and has no room to evacuate into: what is left is live or idle cache; the stores let idle cache go by its retention rows"
                    );
                }
                // Said every turn it stands, not only as it changes: what
                // the policy discarded may free slots inside chunks other
                // sets still use and give back no segment, so the
                // shortfall stands at the same `held`. The gauge reports a
                // standing shortfall again on its own period
                // ([`crate::policy::pressure::AGAIN`]), and the stores
                // let more idle cache go each time, lowest retention
                // first, until the pool can give the segments back.
                self.short = Some(held);
                signal(Signal::TierPressure {
                    held_bytes: held.saturating_mul(limits.segment_bytes),
                    target_bytes: segments.saturating_mul(limits.segment_bytes),
                });
            }
        }
        let held = pool.held();
        if let Some(room) = &self.room {
            room.settle(held);
        }
        self.note_limit(held, &limits);
        pool.ticked();
        Turn {
            target,
            segments,
            mapped,
            answer,
        }
    }
}

/// What one turn asked of the pool and how it answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Turn {
    pub target: u64,
    /// The target in the pool's segments, what it was asked.
    pub segments: u64,
    /// What the pool mapped once asked: what its answer was judged on.
    pub mapped: Mapped,
    pub answer: Retarget,
}

/// The segments `target` comes to, the floor's at least.
fn within(limits: &Limits, target: u64) -> u64 {
    limits.segments(target.min(limits.hard_limit))
}

async fn watch(
    mut pool: Shared<Steered>,
    yields: Yield,
    room: Option<PoolRoom>,
    source: Option<super::Source>,
) {
    let named = source.as_ref().map_or("none read", super::Source::name);
    let mut steer = Steer::new(&pool, yields)
        .with_room(&pool, room)
        .with_source(named);
    let (shared, engine) = pool.with(|pool| (pool.inner().is_shared(), pool.inner().locus().1));
    let mut system = said(&pool, engine, yields, &steer, shared, source);
    // A shared pool that does not yield may take every segment its range
    // holds: its walker commits them ahead of the waiters that need them,
    // off the fire path, and never gives them back. Held within the host's
    // room, it is moved there by the loop below, from its first tick.
    if shared && !steer.governor().yields().yields() && !steer.has_room() {
        let _ = pool.retarget(within(steer.governor().limits(), pool.hard_limit()));
    }
    // Chunks are handed out of mapped segments only: a pool mapped past its
    // limit is what lets a sequence onto chunks the engine holds back. The
    // limit is the load's (this loop starts after the engine answered the
    // residency it starts with) and the pool is never moved past it, so it
    // is read once.
    // `mapped`: the bytes under the pool's mapped segments.
    let (mapped, hard_limit) = (pool.held(), steer.governor().limits().hard_limit);
    if shared && mapped > hard_limit {
        tracing::warn!(mapped, hard_limit, "the pool is mapped past its hard limit");
    }
    // A shared pool that yields moves its own segments off the fire path:
    // the loop reads the system each tick and hands it the rule's target.
    // One that does not, or a pool that is not shared, keeps its size but
    // for the host's room: the loop only ticks.
    let mut interval = crate::rt::time::interval(TICK);
    loop {
        interval.tick().await;
        let pressure = system.as_mut().map_or(0.0, |system| system.read());
        let held = pool.held();
        let turn = steer.turn(&mut pool, pressure, &mut |signal| {
            steered::signal(engine, signal);
        });
        // Once per read that reached the source: its own value, the reading,
        // and what the pool made of the target, with the segments its answer
        // was judged on.
        if let Some(raw) = system.as_mut().and_then(Reader::take_fresh) {
            tracing::debug!(
                engine,
                raw,
                pressure,
                target_mib = turn.target >> 20,
                held_mib = held >> 20,
                target_segments = turn.segments,
                mapped_segments = turn.mapped.segments,
                draining_segments = turn.mapped.draining,
                answer = ?turn.answer,
                "pressure read"
            );
        }
        read_ledger(&pool, engine);
    }
}

/// States in the load log what the device target follows, and opens the
/// system's pressure reader where the pool yields to it. The load refused
/// a yield with no source before this runs ([`super::source_for`]).
fn said(
    pool: &Shared<Steered>,
    engine: usize,
    yields: Yield,
    steer: &Steer,
    shared: bool,
    source: Option<super::Source>,
) -> Option<Reader> {
    if !yields.yields() {
        tracing::info!(
            engine,
            "engine.memory.yield = 0: the device target is the budget; the system's memory pressure is not read"
        );
        return None;
    }
    // A fixed pool's Steer said so already.
    if pool.growth() == Growth::Fixed {
        return None;
    }
    if !steer.governor().yields().yields() || !shared {
        tracing::info!(
            engine,
            yields = yields.share(),
            "yield does not apply: this engine's device pool is sized once at load and does not move"
        );
        return None;
    }
    let mut system = source.map(Reader::new);
    let limits = steer.governor().limits();
    match &mut system {
        Some(system) => {
            let pressure = system.read();
            // The floor is the load's own promise: one lane at the full
            // declared context, the reserved chunks, the activation span
            // and the held dense weights. A long max_context raises it,
            // and with it lowers how far yield can press.
            let (parts, weights) = pool.with(|pool| {
                (pool.inner().floor_parts(), pool.cache().pinned_bytes())
            });
            let mib = |bytes: Option<u64>| bytes.map_or(0, |bytes| bytes >> 20);
            tracing::info!(
                engine,
                yields = yields.share(),
                source = system.source().name(),
                pressure,
                floor_mib = limits.floor >> 20,
                lane_kv_mib = mib(parts.map(|parts| parts.lane_bytes)),
                max_context = parts.map_or(0, |parts| parts.max_context),
                reserved_mib = mib(parts.map(|parts| parts.reserved_bytes)),
                activation_mib = mib(parts.map(|parts| parts.act_bytes)),
                dense_weights_mib = weights >> 20,
                least_mapped_mib = mib(parts.map(|parts| parts.least_mapped_bytes)),
                budget_mib = limits.hard_limit >> 20,
                target_mib = limits.target(yields, pressure) >> 20,
                "the device target follows the system's memory pressure: floor + (budget - floor) * (1 - yield * pressure); the floor is one lane at max_context, the reserved chunks, the activation span and the dense weights, and no less than the fewest segments the pool can map (the reserved chunks' and the activation span's)"
            );
        }
        None => tracing::error!(
            engine,
            yields = yields.share(),
            "the pool yields but was handed no pressure source: the load should have refused"
        ),
    }
    system
}

/// Puts `engine`'s moving ledger term where the policy reads it: its weight
/// cache's bytes above the floor. What its driver holds outside is a load
/// fact, set once at bootstrap.
fn read_ledger(pool: &Shared<Steered>, engine: usize) {
    let model = pool.with(|pool| pool.cache().cache_bytes());
    crate::policy::pressure::set_model(engine, model);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

    use super::*;

    const MIB: u64 = 1 << 20;

    /// A host of memory a pool shares with other work: what it has left over
    /// once both have taken theirs, less what is defended and the floor, is
    /// the room its reader reports, negative where it is short.
    struct Host {
        /// What the host could give the pool and the others, in MiB.
        free: AtomicI64,
        /// What the others hold, in MiB.
        others: AtomicI64,
        /// What the pool holds, in segments of a MiB.
        pool: AtomicU64,
    }

    impl Host {
        fn new(free: i64) -> Arc<Self> {
            Arc::new(Self {
                free: AtomicI64::new(free),
                others: AtomicI64::new(0),
                pool: AtomicU64::new(0),
            })
        }

        fn reader(self: &Arc<Self>) -> ::engine::HostRoom {
            let host = self.clone();
            Arc::new(move || {
                let used = host.others.load(Ordering::Relaxed)
                    + i64::try_from(host.pool.load(Ordering::Relaxed)).unwrap();
                Ok((host.free.load(Ordering::Relaxed) - used) * MIB as i64)
            })
        }
    }

    /// A pool of 1 MiB segments, 100 MiB its floor and 1000 its limit, on
    /// `host`. A shrink below `pinned` segments finds no room, and no
    /// retarget ever fails.
    struct Pool {
        host: Arc<Host>,
        growth: Growth,
        pinned: u64,
    }

    impl Pool {
        fn on(host: &Arc<Host>, segments: u64, growth: Growth) -> Self {
            host.pool.store(segments, Ordering::Relaxed);
            Self {
                host: host.clone(),
                growth,
                pinned: 0,
            }
        }

        fn segments(&self) -> u64 {
            self.host.pool.load(Ordering::Relaxed)
        }
    }

    impl PoolTarget for Pool {
        fn segment_bytes(&self) -> u64 {
            MIB
        }
        fn floor(&self) -> u64 {
            100 * MIB
        }
        fn hard_limit(&self) -> u64 {
            1000 * MIB
        }
        fn held(&self) -> u64 {
            self.segments() * MIB
        }
        fn growth(&self) -> Growth {
            self.growth
        }
        fn retarget(&mut self, segments: u64) -> Retarget {
            if segments < self.pinned {
                // It evacuates down to the segments it has room for.
                let held = self.segments().min(self.pinned);
                self.host.pool.store(held, Ordering::Relaxed);
                return Retarget::NoRoom { held };
            }
            self.host.pool.store(segments, Ordering::Relaxed);
            Retarget::Reached
        }
    }

    const ELASTIC: Growth = Growth::Elastic;
    fn half() -> Yield {
        Yield::new(0.5).unwrap()
    }

    fn tick(steer: &mut Steer, pool: &mut Pool) -> Vec<Signal> {
        pressed(steer, pool, 0.0)
    }

    fn pressed(steer: &mut Steer, pool: &mut Pool, pressure: f64) -> Vec<Signal> {
        let mut signals = Vec::new();
        steer.turn(pool, pressure, &mut |signal| signals.push(signal));
        signals
    }

    #[test]
    fn a_yielding_pool_follows_the_rule_and_sheds_through_tier_pressure() {
        let host = Host::new(5000);
        let mut pool = Pool::on(&host, 1000, ELASTIC);
        pool.pinned = 700;
        let mut steer = Steer::new(&pool, half());
        assert!(pressed(&mut steer, &mut pool, 0.0).is_empty());
        assert_eq!(pool.segments(), 1000, "no pressure: the budget");
        // Full pressure at half a yield: 100 + 900 * 0.5 = 550, under the
        // 700 the pool has room to evacuate to.
        assert_eq!(
            pressed(&mut steer, &mut pool, 1.0),
            vec![Signal::TierPressure {
                held_bytes: 700 * MIB,
                target_bytes: 550 * MIB,
            }]
        );
        assert_eq!(pool.segments(), 700, "nothing was dropped");
        // The policy sheds; the shrink lands, and the pressure going grows
        // the pool back to its budget at once.
        pool.pinned = 0;
        assert_eq!(pressed(&mut steer, &mut pool, 1.0), vec![Signal::Eased]);
        assert_eq!(pool.segments(), 550);
        assert!(pressed(&mut steer, &mut pool, 0.2).is_empty());
        assert_eq!(pool.segments(), 100 + 810);
        assert!(pressed(&mut steer, &mut pool, 0.0).is_empty());
        assert_eq!(pool.segments(), 1000);
    }

    /// Two pools on one host: a reader that counts what both hold.
    fn two_pools(
        free: i64,
        a: u64,
        b: u64,
    ) -> (Arc<HostShare>, Arc<Host>, Arc<Host>, Arc<AtomicI64>) {
        let (ha, hb) = (Host::new(0), Host::new(0));
        ha.pool.store(a, Ordering::Relaxed);
        hb.pool.store(b, Ordering::Relaxed);
        let free = Arc::new(AtomicI64::new(free));
        let (ra, rb, rf) = (ha.clone(), hb.clone(), free.clone());
        let reader: ::engine::HostRoom = Arc::new(move || {
            let held =
                i64::try_from(ra.pool.load(Ordering::Relaxed) + rb.pool.load(Ordering::Relaxed))
                    .unwrap();
            Ok((rf.load(Ordering::Relaxed) - held) * MIB as i64)
        });
        (HostShare::new(reader), ha, hb, free)
    }

    #[test]
    fn pools_on_one_host_are_granted_its_room_once() {
        let (share, ha, hb, _free) = two_pools(1400, 500, 500);
        let (mut a, mut b) = (Pool::on(&ha, 500, ELASTIC), Pool::on(&hb, 500, ELASTIC));
        let dedicated = |pool: &Pool| {
            Steer::new(pool, Yield::NONE).with_room(pool, Some(share.pool()))
        };
        let (mut sa, mut sb) = (dedicated(&a), dedicated(&b));
        tick(&mut sa, &mut a);
        tick(&mut sb, &mut b);
        // 400 MiB of room: the first pool is granted it, the second none of
        // it, not 400 each.
        assert_eq!(sa.ceiling(), Some(900 * MIB));
        assert_eq!(sb.ceiling(), Some(500 * MIB));
        let ceilings = sa.ceiling().unwrap() + sb.ceiling().unwrap();
        assert!(ceilings <= 1400 * MIB, "together within the host");
    }

    #[test]
    fn an_unreadable_room_holds_what_the_pool_has_and_grows_nothing() {
        let host = Host::new(1000);
        let mut pool = Pool::on(&host, 600, ELASTIC);
        let broken = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = host.reader();
        let flag = broken.clone();
        let flaky: ::engine::HostRoom = Arc::new(move || {
            if flag.load(Ordering::Relaxed) {
                Err("no /proc/meminfo".to_string())
            } else {
                reader()
            }
        });
        let mut steer =
            Steer::new(&pool, half()).with_room(&pool, Some(PoolRoom::alone(flaky)));
        // Another app holds 300: the pool may hold 700, and grows into it.
        host.others.store(300, Ordering::Relaxed);
        for _ in 0..4 {
            tick(&mut steer, &mut pool);
        }
        assert_eq!(pool.segments(), 700);
        // The reading fails: the pool keeps its 700, not its floor, and
        // grows no further though the app goes.
        broken.store(true, Ordering::Relaxed);
        host.others.store(0, Ordering::Relaxed);
        for _ in 0..4 {
            tick(&mut steer, &mut pool);
        }
        assert_eq!(steer.regime(), Some(Regime::Unreadable));
        assert_eq!(steer.ceiling(), Some(700 * MIB));
        assert_eq!(pool.segments(), 700);
        // It reads again: the room is there and the pool grows into it.
        broken.store(false, Ordering::Relaxed);
        for _ in 0..4 {
            tick(&mut steer, &mut pool);
        }
        assert_eq!(pool.segments(), 1000);
    }

}

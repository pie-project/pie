//! The loop, on a pool: read the system, move the target.

use super::steered::{self, Shared, Steered};
use super::{
    Governor, Growth, Limits, PoolTarget, Pressure, Retarget, Signal, System, TICK, Tolerance,
};

/// Steers `pool`, an engine's: a dedicated or fixed pool keeps its size,
/// but its loop still ticks, since the weight cache in front of it settles
/// its windows on the governor's tick. Where the engine states a reader of
/// the host's room (`room`, its share of the host's one ledger), every pool
/// that can move is held within it.
pub fn start(pool: Shared<Steered>, tolerance: Tolerance, room: Option<PoolRoom>) {
    crate::rt::spawn(watch(pool, tolerance, room));
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

/// What one calm tick may add. A unified memory machine has no budget to
/// stop a burst but this: the system's complaint arrives after the damage.
const GROWTH_STEP: u64 = 512 * 1024 * 1024;

/// One governor on one pool, a turn at a time: the loop without its clock,
/// so a test can drive it.
pub struct Steer {
    governor: Governor,
    /// The segments held short of the target, while a shrink has no room.
    short: Option<u64>,
    yielded: bool,
    idled: bool,
    /// The pool's share of what the host can still give the process, for a
    /// pool that can move.
    room: Option<PoolRoom>,
    /// The most the pool may hold, as the last tick read the host's room:
    /// what the pool held then plus the room (less, where the host is short),
    /// within the floor and the load's hard limit. A clamp on the governor's target, whatever the
    /// tolerance: the host's room is a fact of the platform, not pressure
    /// to yield to.
    ceiling: Option<u64>,
    /// The regime and segment count the log last named.
    noted: Option<(Regime, u64)>,
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

/// What one turn leaves the caller to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Turn {
    pub target: u64,
    /// The governor started (`true`) or stopped yielding.
    pub yielding: Option<bool>,
    /// The process just went idle.
    pub idled: bool,
}

impl Steer {
    /// A fixed pool has nothing to give back, so it is steered as dedicated
    /// whatever the deployment asked.
    #[must_use]
    pub fn new(pool: &impl PoolTarget, tolerance: Tolerance, step: u64) -> Self {
        let tolerance = if pool.growth() == Growth::Fixed {
            Tolerance::DEDICATED
        } else {
            tolerance
        };
        Self {
            governor: Governor::new(Limits::of(pool, step), tolerance, pool.held()),
            short: None,
            yielded: false,
            idled: false,
            room: None,
            ceiling: None,
            noted: None,
        }
    }

    /// Holds the pool within the host's room as `room` reads it each tick.
    /// A fixed pool cannot move, so it takes no reader.
    #[must_use]
    pub fn with_room(mut self, pool: &impl PoolTarget, room: Option<PoolRoom>) -> Self {
        self.room = room.filter(|_| pool.growth() != Growth::Fixed);
        self
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

    /// The regime the log last named.
    #[must_use]
    pub fn regime(&self) -> Option<Regime> {
        self.noted.map(|(regime, _)| regime)
    }

    #[must_use]
    pub const fn governor(&self) -> &Governor {
        &self.governor
    }

    /// One turn: `read` is the system's state on a tick, or `None` when a
    /// waiter asked between ticks. A dedicated pool's size is left as it
    /// is; its tick still comes.
    pub fn turn(
        &mut self,
        pool: &mut impl PoolTarget,
        read: Option<Pressure>,
        signal: &mut impl FnMut(Signal),
    ) -> Turn {
        if read.is_some() {
            self.read_room(pool);
        }
        let dedicated = self.governor.tolerance().dedicated();
        if dedicated && self.ceiling.is_none() {
            if read.is_some() {
                pool.ticked();
            }
            return Turn {
                target: self.governor.target(),
                yielding: None,
                idled: false,
            };
        }
        let target = if dedicated {
            self.governor.target()
        } else {
            let demand = pool.demand();
            match read {
                None => self.governor.asked(demand),
                Some(read) => {
                    let pressure = if pool.take_refused() {
                        Pressure::Critical
                    } else {
                        read
                    };
                    self.governor.tick(pressure, demand)
                }
            }
        };
        let target = self.ceiling.map_or(target, |ceiling| target.min(ceiling));
        let limits = *self.governor.limits();
        let segments = within(&limits, target);
        match pool.retarget(segments) {
            Retarget::Reached => {
                if self.short.take().is_some() {
                    signal(Signal::Eased);
                }
            }
            Retarget::NoRoom { held } => {
                if self.short != Some(held) {
                    self.short = Some(held);
                    signal(Signal::TierPressure {
                        held_bytes: held.saturating_mul(limits.segment_bytes),
                        target_bytes: segments.saturating_mul(limits.segment_bytes),
                    });
                }
            }
        }
        if let Some(room) = &self.room {
            room.settle(pool.held());
        }
        if read.is_some() {
            pool.ticked();
        }
        let mut turn = Turn {
            target,
            yielding: None,
            idled: false,
        };
        if self.governor.yielding() != self.yielded {
            self.yielded = self.governor.yielding();
            turn.yielding = Some(self.yielded);
        }
        if self.governor.idle() != self.idled {
            self.idled = self.governor.idle();
            turn.idled = self.idled;
        }
        turn
    }
}

/// The segments `target` comes to, the floor's at least.
fn within(limits: &Limits, target: u64) -> u64 {
    limits.segments(target.min(limits.hard_limit))
}

async fn watch(mut pool: Shared<Steered>, tolerance: Tolerance, room: Option<PoolRoom>) {
    let mut steer = Steer::new(&pool, tolerance, GROWTH_STEP).with_room(&pool, room);
    let (shared, engine) = pool.with(|pool| (pool.inner().is_shared(), pool.inner().locus().1));
    let mut interval = crate::rt::time::interval(TICK);
    // A shared pool moves its own segments off the fire path; the loop reads
    // the system each tick and hands it the target.
    if shared && !steer.governor().tolerance().dedicated() {
        let mut system: Box<dyn System> = super::system();
        loop {
            interval.tick().await;
            let read = Some(system.read());
            steer.turn(&mut pool, read, &mut |signal| {
                steered::signal(engine, signal);
            });
            read_ledger(&pool, engine);
        }
    }
    // A dedicated shared pool may take every segment its range holds: its
    // walker commits them ahead of the waiters that need them, off the fire
    // path, and never gives them back.
    // Held within the host's room, it is moved there by the loop below,
    // from its first tick.
    if shared && steer.governor().tolerance().dedicated() && !steer.has_room() {
        use crate::governor::PoolTarget;
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
    // A dedicated pool's size stands but for the host's room, and a pool
    // that is not shared never moves: the loop only ticks, reading nothing
    // of the system's pressure and answering no waiter.
    loop {
        interval.tick().await;
        steer.turn(&mut pool, Some(Pressure::Calm), &mut |signal| {
            steered::signal(engine, signal);
        });
        read_ledger(&pool, engine);
    }
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
    use crate::governor::Demand;

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
        fn demand(&mut self) -> Demand {
            Demand {
                wanted: 1000 * MIB,
                live: 1000 * MIB,
            }
        }
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

    const ELASTIC: Growth = Growth::Elastic { least_cut: MIB };
    const BALANCED: Tolerance = Tolerance {
        backoff: 0.8,
        release_idle: true,
    };

    fn tick(steer: &mut Steer, pool: &mut Pool) -> Vec<Signal> {
        let mut signals = Vec::new();
        steer.turn(pool, Some(Pressure::Calm), &mut |signal| {
            signals.push(signal)
        });
        signals
    }

    #[test]
    fn the_ceiling_follows_a_falling_and_recovering_room() {
        let host = Host::new(1200);
        let mut pool = Pool::on(&host, 1000, ELASTIC);
        let mut steer = Steer::new(&pool, BALANCED, 512 * MIB)
            .with_room(&pool, Some(PoolRoom::alone(host.reader())));
        tick(&mut steer, &mut pool);
        assert_eq!(steer.ceiling(), Some(1000 * MIB));
        assert_eq!(pool.segments(), 1000);
        // Another app takes 800 MiB: the host is 600 short, and the pool
        // gives it back on the next tick.
        host.others.store(800, Ordering::Relaxed);
        tick(&mut steer, &mut pool);
        assert_eq!(steer.ceiling(), Some(400 * MIB));
        assert_eq!(pool.segments(), 400);
        // And past the floor, never under it.
        host.others.store(1150, Ordering::Relaxed);
        tick(&mut steer, &mut pool);
        assert_eq!(pool.segments(), 100);
        // The app goes: the room comes back and the pool grows into it as
        // its demand asks, a step a tick.
        host.others.store(0, Ordering::Relaxed);
        for _ in 0..8 {
            tick(&mut steer, &mut pool);
        }
        assert_eq!(steer.ceiling(), Some(1000 * MIB));
        assert_eq!(pool.segments(), 1000);
    }

    #[test]
    fn the_log_names_a_change_of_regime() {
        let host = Host::new(900);
        let mut pool = Pool::on(&host, 600, ELASTIC);
        let mut steer = Steer::new(&pool, BALANCED, 512 * MIB)
            .with_room(&pool, Some(PoolRoom::alone(host.reader())));
        tick(&mut steer, &mut pool);
        assert_eq!(steer.regime(), Some(Regime::Room));
        // Another app takes 500 MiB: the ceiling falls under what is held.
        host.others.store(500, Ordering::Relaxed);
        tick(&mut steer, &mut pool);
        assert_eq!(steer.regime(), Some(Regime::Short));
        // It goes: room again.
        host.others.store(0, Ordering::Relaxed);
        tick(&mut steer, &mut pool);
        assert_eq!(steer.regime(), Some(Regime::Room));
        // Nearly all of it is taken: the pool is left its floor.
        host.others.store(880, Ordering::Relaxed);
        tick(&mut steer, &mut pool);
        assert_eq!(steer.regime(), Some(Regime::AtFloor));
        // More room than the pool may ever hold: its hard limit.
        let roomy = Host::new(5000);
        let mut big = Pool::on(&roomy, 600, ELASTIC);
        let mut held = Steer::new(&big, BALANCED, 512 * MIB)
            .with_room(&big, Some(PoolRoom::alone(roomy.reader())));
        tick(&mut held, &mut big);
        assert_eq!(held.regime(), Some(Regime::AtLimit));
    }

    #[test]
    fn regimes_are_read_against_the_floor_the_limit_and_what_is_held() {
        let limits = Limits {
            floor: 100,
            hard_limit: 1000,
            step: 10,
            shrink_step: 10,
            segment_bytes: 1,
        };
        assert_eq!(Regime::of(100, 500, &limits), Regime::AtFloor);
        assert_eq!(Regime::of(1000, 500, &limits), Regime::AtLimit);
        assert_eq!(Regime::of(400, 500, &limits), Regime::Short);
        assert_eq!(Regime::of(600, 500, &limits), Regime::Room);
        assert_eq!(Regime::of(500, 500, &limits), Regime::Room);
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
            Steer::new(pool, Tolerance::DEDICATED, 512 * MIB).with_room(pool, Some(share.pool()))
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
    fn pools_on_one_host_give_back_its_shortfall_once() {
        let (share, ha, hb, free) = two_pools(1000, 500, 500);
        let (mut a, mut b) = (Pool::on(&ha, 500, ELASTIC), Pool::on(&hb, 500, ELASTIC));
        let (mut sa, mut sb) = (
            Steer::new(&a, BALANCED, 512 * MIB).with_room(&a, Some(share.pool())),
            Steer::new(&b, BALANCED, 512 * MIB).with_room(&b, Some(share.pool())),
        );
        tick(&mut sa, &mut a);
        tick(&mut sb, &mut b);
        // The host loses 300 MiB to another app: 300 short, given back once.
        free.store(700, Ordering::Relaxed);
        tick(&mut sa, &mut a);
        tick(&mut sb, &mut b);
        assert_eq!(a.segments() + b.segments(), 700, "300 given back, not 600");
    }

    #[test]
    fn a_pool_that_goes_leaves_its_grant() {
        let (share, _ha, _hb, _free) = two_pools(1400, 500, 500);
        let (a, b) = (share.pool(), share.pool());
        // The first is granted all 400 MiB of room and has not taken it.
        assert_eq!(
            a.grant(500 * MIB, |room| 500 * MIB + room.unwrap() as u64),
            900 * MIB
        );
        assert_eq!(
            b.grant(500 * MIB, |room| 500 * MIB + room.unwrap().max(0) as u64),
            500 * MIB
        );
        // It goes: what it was granted is the host's again.
        drop(a);
        assert_eq!(
            b.grant(500 * MIB, |room| 500 * MIB + room.unwrap().max(0) as u64),
            900 * MIB
        );
    }

    #[test]
    fn a_shrink_without_room_sheds_through_tier_pressure() {
        let host = Host::new(1000);
        let mut pool = Pool::on(&host, 1000, ELASTIC);
        pool.pinned = 700;
        let mut steer = Steer::new(&pool, BALANCED, 512 * MIB)
            .with_room(&pool, Some(PoolRoom::alone(host.reader())));
        tick(&mut steer, &mut pool);
        host.others.store(500, Ordering::Relaxed);
        let signals = tick(&mut steer, &mut pool);
        assert_eq!(
            signals,
            vec![Signal::TierPressure {
                held_bytes: 700 * MIB,
                target_bytes: 500 * MIB,
            }]
        );
        // What it holds stays: nothing was dropped, nothing failed.
        assert_eq!(pool.segments(), 700);
        // Once the policy has shed, the shrink lands and the pressure eases.
        pool.pinned = 0;
        assert_eq!(tick(&mut steer, &mut pool), vec![Signal::Eased]);
        assert_eq!(pool.segments(), 500);
    }

    /// One episode end to end: the host's room drops below what the pool
    /// holds, the governor raises tier pressure for exactly the shortfall,
    /// PLEX's default responder sheds lowest retention key first (a pinned
    /// set never), the shrink lands with no retarget failing, and once the
    /// room recovers the pool grows back to its limit with nothing raised.
    #[test]
    fn a_room_under_the_pool_sheds_in_plex_order_and_recovers() {
        use crate::governor::steered::short_bytes;
        use crate::policy::pressure::{Candidate, PINNED, choose};

        let set = |id: &str, key: i64, last_use_ms: u64, mib: u64| Candidate {
            id: id.into(),
            key,
            last_use_ms,
            bytes: Some(mib * MIB),
        };
        let sets = vec![
            set("kept", 5, 10, 200),
            set("droppable", -1, 30, 150),
            set("older", 2, 10, 100),
            set("newer", 2, 20, 100),
            set("pinned", PINNED, 0, 300),
        ];
        let live: u64 = 850;

        let host = Host::new(1000);
        let mut pool = Pool::on(&host, 1000, ELASTIC);
        pool.pinned = live;
        let mut steer = Steer::new(&pool, BALANCED, 512 * MIB)
            .with_room(&pool, Some(PoolRoom::alone(host.reader())));
        assert!(tick(&mut steer, &mut pool).is_empty());
        assert_eq!(steer.ceiling(), Some(1000 * MIB));

        // Another app takes 500 MiB: the ceiling falls to 500, under the
        // 850 the live sets hold, so the pool evacuates to them and asks.
        host.others.store(500, Ordering::Relaxed);
        let signals = tick(&mut steer, &mut pool);
        assert_eq!(steer.ceiling(), Some(500 * MIB));
        assert_eq!(
            signals,
            vec![Signal::TierPressure {
                held_bytes: live * MIB,
                target_bytes: 500 * MIB,
            }]
        );
        assert_eq!(pool.segments(), live, "nothing live was dropped");
        let short = short_bytes(signals[0]);
        assert_eq!(short, 350 * MIB);

        // The same reading again raises nothing new while it waits.
        assert!(tick(&mut steer, &mut pool).is_empty());

        // PLEX order: the droppable key first, then key 2 least recently
        // used first, stopping once the shortfall is covered; key 5 and the
        // pinned set stay.
        let chosen = choose(sets.clone(), short);
        assert_eq!(chosen, ["droppable", "older", "newer"]);
        let freed: u64 = sets
            .iter()
            .filter(|c| chosen.contains(&c.id))
            .map(|c| c.bytes.unwrap() / MIB)
            .sum();
        assert!(freed * MIB >= short);

        // The stores discard them; the shrink lands and the pressure eases.
        pool.pinned = live - freed;
        assert_eq!(tick(&mut steer, &mut pool), vec![Signal::Eased]);
        assert_eq!(pool.segments(), 500);

        // The app goes: the room comes back and the pool grows into it,
        // raising nothing on the way.
        host.others.store(0, Ordering::Relaxed);
        for _ in 0..8 {
            assert!(tick(&mut steer, &mut pool).is_empty());
        }
        assert_eq!(steer.ceiling(), Some(1000 * MIB));
        assert_eq!(pool.segments(), 1000);
    }

    #[test]
    fn a_dedicated_pool_is_held_within_the_room_too() {
        let host = Host::new(1000);
        let mut pool = Pool::on(&host, 1000, ELASTIC);
        let mut steer = Steer::new(&pool, Tolerance::DEDICATED, 512 * MIB)
            .with_room(&pool, Some(PoolRoom::alone(host.reader())));
        tick(&mut steer, &mut pool);
        assert_eq!(pool.segments(), 1000);
        host.others.store(300, Ordering::Relaxed);
        tick(&mut steer, &mut pool);
        assert_eq!(pool.segments(), 700);
        host.others.store(0, Ordering::Relaxed);
        tick(&mut steer, &mut pool);
        assert_eq!(pool.segments(), 1000);
    }

    #[test]
    fn a_dedicated_pool_with_no_room_reader_keeps_its_size() {
        let host = Host::new(1000);
        let mut pool = Pool::on(&host, 1000, ELASTIC);
        let mut steer = Steer::new(&pool, Tolerance::DEDICATED, 512 * MIB).with_room(&pool, None);
        host.others.store(600, Ordering::Relaxed);
        tick(&mut steer, &mut pool);
        assert_eq!(steer.ceiling(), None);
        assert_eq!(pool.segments(), 1000);
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
            Steer::new(&pool, BALANCED, 512 * MIB).with_room(&pool, Some(PoolRoom::alone(flaky)));
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

    #[test]
    fn an_unreadable_first_reading_holds_what_the_pool_has() {
        let host = Host::new(1000);
        let mut pool = Pool::on(&host, 600, ELASTIC);
        let unreadable: ::engine::HostRoom = Arc::new(|| Err("no /proc/meminfo".to_string()));
        let mut steer = Steer::new(&pool, BALANCED, 512 * MIB)
            .with_room(&pool, Some(PoolRoom::alone(unreadable)));
        tick(&mut steer, &mut pool);
        assert_eq!(steer.ceiling(), Some(600 * MIB));
        assert_eq!(pool.segments(), 600);
    }

    #[test]
    fn a_fixed_pool_takes_no_room_reader() {
        let host = Host::new(1000);
        let pool = Pool::on(&host, 1000, Growth::Fixed);
        let steer = Steer::new(&pool, BALANCED, 512 * MIB)
            .with_room(&pool, Some(PoolRoom::alone(host.reader())));
        assert!(!steer.has_room());
    }
}

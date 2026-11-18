//! The loop, on a pool: read the system, move the target.

use super::steered::{self, Shared, Steered};
use super::{
    Governor, Growth, Limits, PoolTarget, Pressure, Retarget, Signal, System, TICK, Tolerance,
};

/// Steers `pool`, an engine's: a dedicated or fixed pool keeps its size,
/// but its loop still ticks, since the weight cache in front of it settles
/// its windows on the governor's tick. Where the engine states a reader of
/// the host's room (`room`), every pool that can move is held within it.
pub fn start(pool: Shared<Steered>, tolerance: Tolerance, room: Option<::engine::HostRoom>) {
    crate::rt::spawn(watch(pool, tolerance, room));
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
    /// The engine's reader of what the host can still give the process,
    /// for a pool that can move.
    room: Option<::engine::HostRoom>,
    /// The most the pool may hold, as the last tick read the host's room:
    /// what the pool held then plus the room (less, where the host is short),
    /// within the floor and the load's hard limit. A clamp on the governor's target, whatever the
    /// tolerance: the host's room is a fact of the platform, not pressure
    /// to yield to.
    ceiling: Option<u64>,
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
        }
    }

    /// Holds the pool within the host's room as `room` reads it each tick.
    /// A fixed pool cannot move, so it takes no reader.
    #[must_use]
    pub fn with_room(mut self, pool: &impl PoolTarget, room: Option<::engine::HostRoom>) -> Self {
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

    /// Reads the host's room: the pool may hold what it holds now plus it,
    /// or less by what the host is short.
    /// A reading that cannot be taken is named, and holds the pool to its
    /// floor until one can.
    fn read_room(&mut self, pool: &impl PoolTarget) {
        let Some(room) = &self.room else {
            return;
        };
        let limits = *self.governor.limits();
        let ceiling = match room() {
            Ok(room) => {
                let ceiling = i128::from(pool.held()) + i128::from(room);
                u64::try_from(ceiling.max(0)).unwrap_or(u64::MAX)
            }
            Err(why) => {
                tracing::error!(
                    %why,
                    "the host's room could not be read: the pool holds to its floor until it can"
                );
                0
            }
        };
        self.ceiling = Some(ceiling.clamp(limits.floor, limits.hard_limit));
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

async fn watch(mut pool: Shared<Steered>, tolerance: Tolerance, room: Option<::engine::HostRoom>) {
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
        let mut steer =
            Steer::new(&pool, BALANCED, 512 * MIB).with_room(&pool, Some(host.reader()));
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
    fn a_shrink_without_room_sheds_through_tier_pressure() {
        let host = Host::new(1000);
        let mut pool = Pool::on(&host, 1000, ELASTIC);
        pool.pinned = 700;
        let mut steer =
            Steer::new(&pool, BALANCED, 512 * MIB).with_room(&pool, Some(host.reader()));
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

    #[test]
    fn a_dedicated_pool_is_held_within_the_room_too() {
        let host = Host::new(1000);
        let mut pool = Pool::on(&host, 1000, ELASTIC);
        let mut steer = Steer::new(&pool, Tolerance::DEDICATED, 512 * MIB)
            .with_room(&pool, Some(host.reader()));
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
    fn an_unreadable_room_holds_the_pool_to_its_floor_by_name() {
        let host = Host::new(1000);
        let mut pool = Pool::on(&host, 1000, ELASTIC);
        let unreadable: ::engine::HostRoom = Arc::new(|| Err("no /proc/meminfo".to_string()));
        let mut steer = Steer::new(&pool, BALANCED, 512 * MIB).with_room(&pool, Some(unreadable));
        tick(&mut steer, &mut pool);
        assert_eq!(steer.ceiling(), Some(100 * MIB));
        assert_eq!(pool.segments(), 100);
    }

    #[test]
    fn a_fixed_pool_takes_no_room_reader() {
        let host = Host::new(1000);
        let pool = Pool::on(&host, 1000, Growth::Fixed);
        let steer = Steer::new(&pool, BALANCED, 512 * MIB).with_room(&pool, Some(host.reader()));
        assert!(!steer.has_room());
    }
}

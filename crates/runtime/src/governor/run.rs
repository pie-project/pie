//! The loop, on a pool: read the system, move the target.

use super::steered::{self, Shared, Steered};
use super::{
    Governor, Growth, Limits, PoolTarget, Pressure, Retarget, Signal, System, TICK, Tolerance,
};

/// Steers `pool`, an engine's: a dedicated or fixed pool keeps its size,
/// but its loop still ticks, since the weight cache in front of it settles
/// its windows on the governor's tick.
pub fn start(pool: Shared<Steered>, tolerance: Tolerance) {
    crate::rt::spawn(watch(pool, tolerance));
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
            Tolerance::Dedicated
        } else {
            tolerance
        };
        Self {
            governor: Governor::new(Limits::of(pool, step), tolerance, pool.held()),
            short: None,
            yielded: false,
            idled: false,
        }
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
        if self.governor.tolerance() == Tolerance::Dedicated {
            if read.is_some() {
                pool.ticked();
            }
            return Turn {
                target: self.governor.target(),
                yielding: None,
                idled: false,
            };
        }
        let demand = pool.demand();
        let target = match read {
            None => self.governor.asked(demand),
            Some(read) => {
                let pressure = if pool.take_refused() {
                    Pressure::Critical
                } else {
                    read
                };
                let target = self.governor.tick(pressure, demand);
                if pressure == Pressure::Critical {
                    pool.shrink_rung();
                }
                target
            }
        };
        let limits = *self.governor.limits();
        let segments = limits.segments(target);
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
            if !self.yielded {
                pool.restore_rung();
            }
            turn.yielding = Some(self.yielded);
        }
        if self.governor.idle() != self.idled {
            self.idled = self.governor.idle();
            turn.idled = self.idled;
        }
        turn
    }
}

async fn watch(mut pool: Shared<Steered>, tolerance: Tolerance) {
    let mut steer = Steer::new(&pool, tolerance, GROWTH_STEP);
    let (grow, shared, engine) = pool.with(|pool| {
        (
            pool.inner().grow().cloned(),
            pool.inner().is_shared(),
            pool.inner().locus().1,
        )
    });
    let mut interval = crate::rt::time::interval(TICK);
    // A shared pool moves its own segments off the fire path; the loop reads
    // the system each tick and hands it the target.
    if shared && steer.governor().tolerance() != Tolerance::Dedicated {
        let mut system: Box<dyn System> = super::system();
        loop {
            interval.tick().await;
            let read = Some(system.read());
            steer.turn(&mut pool, read, &mut |signal| {
                steered::signal(engine, signal);
            });
        }
    }
    // A dedicated pool's size stands: the loop only ticks, reading nothing
    // and answering no waiter.
    let Some(grow) = grow.filter(|_| steer.governor().tolerance() != Tolerance::Dedicated) else {
        loop {
            interval.tick().await;
            steer.turn(&mut pool, Some(Pressure::Calm), &mut |_| {});
        }
    };
    let mut system: Box<dyn System> = super::system();
    loop {
        let waiter = tokio::select! {
            _ = interval.tick() => false,
            () = grow.waiter() => true,
        };
        let read = (!waiter).then(|| system.read());
        let turn = steer.turn(&mut pool, read, &mut |signal| {
            steered::signal(engine, signal);
        });
        grow.set_can_rise(steer.governor().can_rise());
        if let Some(yielding) = turn.yielding {
            grow.yield_scratch(yielding).await;
        }
        if turn.idled {
            grow.release_scratch().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CUT_EVERY, Demand, IDLE_AFTER, KEEP_CACHE, SETTLED};
    use super::*;

    const MIB: u64 = 1 << 20;
    const SEGMENT: u64 = 64 * MIB;

    /// A pool of `SEGMENT`s that records what it was asked and holds a
    /// count of slots in use that only its holders could change.
    struct Fake {
        growth: Growth,
        held: u64,
        demand: u64,
        room: bool,
        rung: u32,
        slots_in_use: u32,
        retargets: Vec<u64>,
    }

    impl Fake {
        fn new(growth: Growth, held: u64) -> Self {
            Self {
                growth,
                held,
                demand: 0,
                room: true,
                rung: 3,
                slots_in_use: 40,
                retargets: Vec::new(),
            }
        }
    }

    impl PoolTarget for Fake {
        fn demand(&mut self) -> Demand {
            Demand {
                wanted: self.demand,
                live: self.demand,
            }
        }

        fn segment_bytes(&self) -> u64 {
            SEGMENT
        }

        fn floor(&self) -> u64 {
            100 * MIB
        }

        fn hard_limit(&self) -> u64 {
            16 * SEGMENT
        }

        fn held(&self) -> u64 {
            self.held * SEGMENT
        }

        fn growth(&self) -> Growth {
            self.growth
        }

        fn retarget(&mut self, segments: u64) -> Retarget {
            self.retargets.push(segments);
            if segments < self.held && !self.room {
                return Retarget::NoRoom { held: self.held };
            }
            self.held = segments;
            Retarget::Reached
        }

        fn shrink_rung(&mut self) {
            self.rung = self.rung.saturating_sub(1);
        }

        fn restore_rung(&mut self) {
            self.rung = 3;
        }
    }

    const ELASTIC: Growth = Growth::Elastic { least_cut: 0 };

    fn turns(steer: &mut Steer, pool: &mut Fake, read: Pressure, n: u32) -> Vec<Signal> {
        let mut signals = Vec::new();
        for _ in 0..n {
            steer.turn(pool, Some(read), &mut |signal| signals.push(signal));
        }
        signals
    }

    #[test]
    fn a_cut_is_never_less_than_a_segment() {
        let pool = Fake::new(ELASTIC, 16);
        let steer = Steer::new(&pool, Tolerance::Balanced, 512 * MIB);
        assert_eq!(steer.governor().limits().shrink_step, SEGMENT);
    }

    #[test]
    fn an_idle_pool_shrinks_by_whole_segments_down_to_the_floor_and_no_further() {
        let mut pool = Fake::new(ELASTIC, 16);
        let mut steer = Steer::new(&pool, Tolerance::Balanced, 512 * MIB);
        turns(&mut steer, &mut pool, Pressure::Calm, KEEP_CACHE + 400);
        let floor = steer.governor().limits().segments(0);
        assert_eq!(floor, 2);
        assert!(pool.retargets.iter().all(|&segments| segments >= floor));
        assert!(pool.retargets.windows(2).all(|pair| pair[1] <= pair[0]));
        let first_cut = pool.retargets.iter().position(|&s| s < 16).unwrap();
        assert_eq!(
            u32::try_from(first_cut).unwrap(),
            IDLE_AFTER - 1,
            "nothing goes before the idle"
        );
        assert_eq!(pool.held, floor);
        assert_eq!(pool.slots_in_use, 40);
    }

    #[test]
    fn critical_takes_the_floor_and_a_smaller_rung_then_calm_restores_the_rung() {
        let mut pool = Fake::new(ELASTIC, 16);
        let mut steer = Steer::new(&pool, Tolerance::Balanced, 512 * MIB);
        let turn = steer.turn(&mut pool, Some(Pressure::Critical), &mut |_| {});
        assert_eq!(turn.yielding, Some(true));
        assert_eq!(pool.rung, 2);
        assert_eq!(pool.held, 2, "the floor's segments");
        steer.turn(&mut pool, Some(Pressure::Critical), &mut |_| {});
        assert_eq!(pool.rung, 1, "each critical tick takes one rung more");
        turns(&mut steer, &mut pool, Pressure::Calm, SETTLED);
        assert_eq!(pool.rung, 3);
    }

    #[test]
    fn a_refused_commit_reads_as_critical() {
        struct Refusing(Fake);
        impl PoolTarget for Refusing {
            fn demand(&mut self) -> Demand {
                self.0.demand()
            }
            fn segment_bytes(&self) -> u64 {
                self.0.segment_bytes()
            }
            fn floor(&self) -> u64 {
                self.0.floor()
            }
            fn hard_limit(&self) -> u64 {
                self.0.hard_limit()
            }
            fn held(&self) -> u64 {
                self.0.held()
            }
            fn growth(&self) -> Growth {
                self.0.growth()
            }
            fn retarget(&mut self, segments: u64) -> Retarget {
                self.0.retarget(segments)
            }
            fn take_refused(&mut self) -> bool {
                true
            }
            fn shrink_rung(&mut self) {
                self.0.shrink_rung();
            }
            fn restore_rung(&mut self) {
                self.0.restore_rung();
            }
        }
        let mut pool = Refusing(Fake::new(ELASTIC, 16));
        let mut steer = Steer::new(&pool, Tolerance::Aggressive, 512 * MIB);
        steer.turn(&mut pool, Some(Pressure::Calm), &mut |_| {});
        assert_eq!(pool.0.rung, 2);
        assert_eq!(pool.0.held, 2);
    }

    #[test]
    fn a_shrink_without_room_keeps_the_segment_and_signals_pressure_once() {
        let mut pool = Fake::new(ELASTIC, 16);
        pool.room = false;
        let mut steer = Steer::new(&pool, Tolerance::Balanced, 512 * MIB);
        let signals = turns(&mut steer, &mut pool, Pressure::Critical, 8);
        assert_eq!(
            signals,
            [Signal::TierPressure {
                held_bytes: 16 * SEGMENT,
                target_bytes: 2 * SEGMENT,
            }]
        );
        assert_eq!(pool.held, 16, "the segments stay");
        assert_eq!(pool.slots_in_use, 40, "nothing was dropped");
        assert!(pool.retargets.len() >= 8, "the shrink is tried again");

        pool.room = true;
        let signals = turns(&mut steer, &mut pool, Pressure::Critical, 1);
        assert_eq!(signals, [Signal::Eased]);
        assert_eq!(pool.held, 2);
    }

    #[test]
    fn a_fixed_pool_never_shrinks() {
        let mut pool = Fake::new(Growth::Fixed, 16);
        let mut steer = Steer::new(&pool, Tolerance::Polite, 512 * MIB);
        assert_eq!(steer.governor().tolerance(), Tolerance::Dedicated);
        assert_eq!(steer.governor().limits().shrink_step, u64::MAX);
        turns(&mut steer, &mut pool, Pressure::Critical, 4);
        turns(&mut steer, &mut pool, Pressure::Congested, 16);
        turns(
            &mut steer,
            &mut pool,
            Pressure::Calm,
            IDLE_AFTER + 64 * CUT_EVERY,
        );
        assert!(pool.retargets.is_empty());
        assert_eq!(pool.held, 16);
        assert_eq!(pool.rung, 3);
    }

    #[test]
    fn a_waiter_turn_rises_without_reading_the_system() {
        let mut pool = Fake::new(ELASTIC, 2);
        pool.demand = 6 * SEGMENT;
        let mut steer = Steer::new(&pool, Tolerance::Balanced, 512 * MIB);
        let turn = steer.turn(&mut pool, None, &mut |_| {});
        assert_eq!(turn.target, 6 * SEGMENT);
        assert_eq!(pool.held, 6);
    }
}

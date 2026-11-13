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
            Tolerance::DEDICATED
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
        // What the engine holds beside the pool moves the limit: the pool
        // follows it down (and back up) in whole segments.
        let hard_limit = pool.hard_limit();
        let limits = *self.governor.limits();
        let moved = limits.segments(hard_limit) != limits.segments(limits.hard_limit);
        if moved {
            self.governor.set_hard_limit(hard_limit);
            tracing::info!(
                hard_limit,
                held = pool.held(),
                "the pool's hard limit moves with what the engine holds beside it"
            );
        }
        if self.governor.tolerance().dedicated() {
            if moved && pool.growth() != Growth::Fixed {
                let _ = pool.retarget(within(self.governor.limits(), self.governor.target()));
            }
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
                self.governor.tick(pressure, demand)
            }
        };
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

async fn watch(mut pool: Shared<Steered>, tolerance: Tolerance) {
    let mut steer = Steer::new(&pool, tolerance, GROWTH_STEP);
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
    if shared && steer.governor().tolerance().dedicated() {
        use crate::governor::PoolTarget;
        let _ = pool.retarget(within(steer.governor().limits(), pool.hard_limit()));
    }
    // A dedicated pool's size stands, and a pool that is not shared never
    // moves: the loop only ticks, reading nothing and answering no waiter.
    let mut warned = false;
    loop {
        interval.tick().await;
        steer.turn(&mut pool, Some(Pressure::Calm), &mut |_| {});
        read_ledger(&pool, engine);
        // Chunks are handed out of mapped segments only: a pool mapped
        // past its limit is what lets a sequence or a lease onto chunks the
        // engine still holds.
        let (held, hard_limit) = (pool.held(), steer.governor().limits().hard_limit);
        let past = shared && held > hard_limit;
        if past && !warned {
            tracing::warn!(held, hard_limit, "the pool is mapped past its hard limit");
        }
        if !past && warned {
            tracing::info!(held, hard_limit, "the pool is mapped within its hard limit again");
        }
        warned = past;
    }
}

/// Puts `engine`'s ledger terms where the policy reads them: its weight
/// cache's bytes above the floor, and what its driver holds outside.
fn read_ledger(pool: &Shared<Steered>, engine: usize) {
    let model = pool.with(|pool| pool.cache().cache_bytes());
    crate::policy::pressure::set_ledger(
        engine,
        crate::policy::pressure::Ledger {
            model,
            outside: crate::engine::live(engine).outside(),
        },
    );
}

//! The loop, on a pool: read the system, move the target.

use super::steered::{Shared, Steered};
use super::{Give, Governor, Growth, Limits, PoolTarget, Reading, Retarget, TICK};

/// Steers `pool`, an engine's, at the deployment's `yield`: a fixed pool
/// keeps its size, but its loop still ticks, since the weight cache in
/// front of it settles its windows on the governor's tick.
pub fn start(pool: Shared<Steered>, share: f64) {
    crate::rt::spawn(watch(pool, share));
}

/// One governor on one pool, a turn at a time: the loop without its clock,
/// so a test can drive it.
pub struct Steer {
    governor: Governor,
}

impl Steer {
    /// A fixed pool has nothing to give back, so it is steered at `yield = 0`
    /// whatever the deployment asked.
    #[must_use]
    pub fn new(pool: &impl PoolTarget, share: f64) -> Self {
        let share = if pool.growth() == Growth::Fixed {
            0.0
        } else {
            share
        };
        Self {
            governor: Governor::new(Limits::of(pool), share),
        }
    }

    #[must_use]
    pub const fn governor(&self) -> &Governor {
        &self.governor
    }

    /// One turn at `pressure`: the target, which the pool is moved to. A
    /// commit the system refused holds it at what the pool has; a shrink
    /// with no room has the pool give back what it can ([`give_back`]). At
    /// `yield = 0` the pool is left as it is and the target is its ceiling.
    pub fn turn(&mut self, pool: &mut impl PoolTarget, pressure: f64) -> u64 {
        let limits = *self.governor.limits();
        if !self.governor.yields() {
            return limits.hard_limit;
        }
        let mut target = self.governor.target(pool.floor(), pressure);
        if pool.refused() {
            target = target.min(pool.held());
        }
        let segments = limits.segments(target);
        if let Retarget::NoRoom { held } = pool.retarget(segments) {
            let short = held
                .saturating_sub(segments)
                .saturating_mul(limits.segment_bytes);
            give_back(pool, short);
        }
        target
    }
}

/// One event of the loop: a tick (`ticked`), or the reading changed.
/// `pressure` is the reading, `None` where nothing is read (`yield = 0`, a
/// fixed pool, a system with no reading): the pool is then left as it is
/// and only ticks. A change turns at once; a tick turns only while the
/// pressure stands or a refusal does, re-reading the floor. The target the
/// step leaves.
pub fn step(
    steer: &mut Steer,
    pool: &mut impl PoolTarget,
    pressure: Option<f64>,
    ticked: bool,
) -> u64 {
    let mut target = steer.governor().limits().hard_limit;
    if let Some(pressure) = pressure
        && (!ticked || pressure > 0.0 || pool.refused())
    {
        target = steer.turn(pool, pressure);
    }
    if ticked {
        pool.ticked();
    }
    target
}

/// Has `pool` give back `short` bytes in [`Give::ORDER`], each kind only
/// past what the ones before it gave: the bytes given. Past the last kind
/// the pool keeps what it holds: what running requests hold is never
/// given back, and nothing is raised for it.
pub fn give_back(pool: &mut impl PoolTarget, short: u64) -> u64 {
    let mut given = 0;
    for what in Give::ORDER {
        if given >= short {
            break;
        }
        given += pool.give(what, short - given);
    }
    given
}

/// What `yield` can give back on this engine, and what it cannot, said once
/// at load.
fn say_what_yield_sheds(share: f64, fixed: bool, span_on_demand: bool, reading: bool) {
    if share <= 0.0 {
        return;
    }
    if fixed {
        tracing::warn!(
            share,
            "yield: this engine's pool is fixed from the load; yield sheds nothing here \
             (no idle cache, weight cache, demotion or rung)"
        );
    } else if !reading {
        tracing::warn!(
            share,
            "yield: this system offers no memory pressure reading; yield sheds nothing here"
        );
    } else if !span_on_demand {
        tracing::info!(
            share,
            "yield: this engine backs its activation span from the load; yield sheds idle \
             cache (prefixes, checkpoints), the weight cache and demotes prefixes keyed \
             above the floor, never the rung"
        );
    } else {
        tracing::info!(
            share,
            "yield: this engine backs its activation span on demand; yield sheds idle cache \
             (prefixes, checkpoints), the weight cache, demotes prefixes keyed above the \
             floor, then narrows the rung"
        );
    }
}

async fn watch(mut pool: Shared<Steered>, share: f64) {
    let mut steer = Steer::new(&pool, share);
    let (shared, engine, span_on_demand) = pool.with(|pool| {
        let inner = pool.inner();
        (inner.is_shared(), inner.locus().1, inner.span_on_demand())
    });
    let mut reading: Option<Reading> = if shared && steer.governor().yields() {
        super::pressure()
    } else {
        None
    };
    let fixed = !shared || pool.growth() == Growth::Fixed;
    say_what_yield_sheds(share, fixed, span_on_demand, reading.is_some());
    // A shared pool may take every segment its range holds: its walker
    // commits them ahead of the waiters that need them, off the fire path,
    // and gives them back only under pressure.
    if shared {
        let _ = pool.retarget(steer.governor().limits().segments(pool.hard_limit()));
    }
    // Chunks are handed out of mapped segments only: a pool mapped past its
    // limit is what lets a sequence onto chunks the engine holds back. The
    // limit is the load's and the pool is never moved past it, so it is
    // read once.
    // `mapped`: the bytes under the pool's mapped segments.
    let (mapped, hard_limit) = (pool.held(), steer.governor().limits().hard_limit);
    if shared && mapped > hard_limit {
        tracing::warn!(mapped, hard_limit, "the pool is mapped past its hard limit");
    }
    let mut interval = crate::rt::time::interval(TICK);
    loop {
        // A tick settles the weight cache and the rung walk; a pressed
        // pool's floor moves with its live state, so a tick re-reads it
        // while the pressure stands. A change in the reading turns at once.
        let ticked = match reading.as_mut() {
            None => {
                interval.tick().await;
                true
            }
            Some(watched) => tokio::select! {
                _ = interval.tick() => true,
                changed = watched.changed() => {
                    if changed.is_err() {
                        reading = None;
                    }
                    false
                }
            },
        };
        let pressure = reading.as_mut().map(|watched| *watched.borrow_and_update());
        step(&mut steer, &mut pool, pressure, ticked);
        if ticked {
            read_ledger(&pool, engine);
        }
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
mod tests;

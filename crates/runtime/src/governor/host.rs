//! The host kv tier as its governor steers it. Its ceiling is what the host
//! can give the load (`engine::host::host_kv_ceiling`, stated as its chunks
//! by the engine); its target follows the host's memory pressure by the
//! device's rule and `yield` key:
//!
//! ```text
//! target = floor + (ceiling - floor) * (1 - yield * pressure)
//! ```
//!
//! The tier commits as it fills, so a target is a cap on the chunks it
//! opens: no chunk at or past the target is opened from then on
//! ([`crate::store::slots::SlotPool::set_ceiling`]). A tier holding chunks
//! past its target with no room to give them back tells the policy
//! `tier-pressure` on the host, as the device's governor does on the
//! device; the policy decides what leaves.

use super::{Governor, Growth, Limits, PoolTarget, Reader, Retarget, Signal, TICK, Yield};
use crate::policy::pressure::{self, Tier};
use crate::store::shared::Shared;

/// One engine's host tier: its shared pool of host chunks.
pub struct HostTier {
    pool: Shared,
}

impl HostTier {
    #[must_use]
    pub const fn new(pool: Shared) -> Self {
        Self { pool }
    }
}

impl PoolTarget for HostTier {
    /// A chunk: the tier opens and commits a chunk at a time.
    fn segment_bytes(&self) -> u64 {
        self.pool.lock().unwrap().pool().chunk_bytes()
    }

    /// None: the tier may be empty, every suspended sequence going to the
    /// disk.
    fn floor(&self) -> u64 {
        0
    }

    /// Its chunks: what the host could give the load when it was seated.
    fn hard_limit(&self) -> u64 {
        let pool = self.pool.lock().unwrap();
        u64::from(pool.pool().total_chunks()) * pool.pool().chunk_bytes()
    }

    /// The chunks it has opened.
    fn held(&self) -> u64 {
        let pool = self.pool.lock().unwrap();
        u64::from(pool.pool().state_chunks()) * pool.pool().chunk_bytes()
    }

    fn growth(&self) -> Growth {
        Growth::Elastic
    }

    /// Opens no chunk at or past `segments`; short where a chunk past it is
    /// still open.
    fn retarget(&mut self, segments: u64) -> Retarget {
        let mut pool = self.pool.lock().unwrap();
        let slots = pool.pool_mut();
        let cap = u32::try_from(segments)
            .unwrap_or(u32::MAX)
            .min(slots.total_chunks());
        slots.set_ceiling(cap);
        let past = (cap..slots.total_chunks())
            .filter(|&chunk| slots.chunk_shape(chunk).is_some())
            .count();
        if past == 0 {
            Retarget::Reached
        } else {
            Retarget::NoRoom {
                held: u64::from(slots.state_chunks()),
            }
        }
    }
}

/// The gauge entry the host tier's governor of `engine` speaks under:
/// apart from the device's ([`super::steered::source`]) and the planner's.
#[must_use]
pub const fn source(engine: usize) -> usize {
    usize::MAX / 2 - engine
}

/// One turn: the rule's target for `pressure`, the tier moved toward it,
/// and what it could not give back, as a signal.
pub fn turn(governor: &mut Governor, tier: &mut impl PoolTarget, pressure: f64) -> Signal {
    let target = governor.tick(pressure);
    let segments = governor.limits().segments(target);
    match tier.retarget(segments) {
        Retarget::Reached => Signal::Eased,
        Retarget::NoRoom { held } => Signal::TierPressure {
            held_bytes: held * governor.limits().segment_bytes,
            target_bytes: target,
        },
    }
}

/// Steers `engine`'s host tier by `yields` from now on. With yield 0 the
/// target is the ceiling and nothing is read, so no loop runs.
pub fn start(engine: usize, pool: Shared, yields: Yield) {
    let tier = HostTier::new(pool);
    let limits = Limits::of(&tier);
    if !yields.yields() {
        tracing::info!(
            engine,
            ceiling_mib = limits.hard_limit >> 20,
            "engine.memory.yield = 0: the host kv tier's target is its ceiling; the system's \
             memory pressure is not read"
        );
        return;
    }
    let Some(reader) = Reader::of_this_system() else {
        tracing::info!(
            engine,
            ceiling_mib = limits.hard_limit >> 20,
            "this system states no memory pressure: it reads 0, and the host kv tier's target \
             is its ceiling"
        );
        return;
    };
    tracing::info!(
        engine,
        yields = yields.share(),
        source = reader.source().name(),
        ceiling_mib = limits.hard_limit >> 20,
        "the host kv tier's target follows the system's memory pressure: floor + (ceiling - \
         floor) * (1 - yield * pressure)"
    );
    crate::rt::spawn(watch(engine, tier, Governor::new(limits, yields), reader));
}

async fn watch(engine: usize, mut tier: HostTier, mut governor: Governor, mut reader: Reader) {
    let mut interval = crate::rt::time::interval(TICK);
    let mut short = false;
    loop {
        interval.tick().await;
        match turn(&mut governor, &mut tier, reader.read()) {
            Signal::TierPressure {
                held_bytes,
                target_bytes,
            } => {
                short = true;
                pressure::raise(
                    Tier::Host,
                    source(engine),
                    held_bytes.saturating_sub(target_bytes).max(1),
                );
            }
            Signal::Eased if short => {
                short = false;
                pressure::relieve(Tier::Host, source(engine));
            }
            Signal::Eased => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tier of `chunks` chunks of `chunk` bytes, its target followed with
    /// `yields`.
    fn governed(chunks: u64, chunk: u64, yields: f64) -> Governor {
        Governor::new(
            Limits {
                floor: 0,
                hard_limit: chunks * chunk,
                segment_bytes: chunk,
            },
            Yield::new(yields).unwrap(),
        )
    }

    /// Yield 0.5 at pressure 1 targets the midpoint of the floor (none) and
    /// the ceiling; yield 0 or no pressure, the ceiling.
    #[test]
    fn half_the_yield_at_full_pressure_targets_the_midpoint() {
        let chunk = 4 << 20;
        let mut governor = governed(1000, chunk, 0.5);
        assert_eq!(governor.tick(1.0), 500 * chunk);
        assert_eq!(governor.limits().segments(governor.target()), 500);
        assert_eq!(governor.tick(0.0), 1000 * chunk);
        let mut held = governed(1000, chunk, 0.0);
        assert_eq!(held.tick(1.0), 1000 * chunk);
    }

    /// A tier that reaches its target eases; one holding chunks past it
    /// says by how much, for the policy's `tier-pressure` on the host.
    #[test]
    fn a_tier_past_its_target_says_what_it_holds() {
        struct Tier {
            chunks: u64,
            open: u64,
            cap: u64,
        }
        impl PoolTarget for Tier {
            fn segment_bytes(&self) -> u64 {
                1
            }
            fn floor(&self) -> u64 {
                0
            }
            fn hard_limit(&self) -> u64 {
                self.chunks
            }
            fn held(&self) -> u64 {
                self.open
            }
            fn growth(&self) -> Growth {
                Growth::Elastic
            }
            fn retarget(&mut self, segments: u64) -> Retarget {
                self.cap = segments;
                if self.open > segments {
                    Retarget::NoRoom { held: self.open }
                } else {
                    Retarget::Reached
                }
            }
        }
        let mut governor = Governor::new(
            Limits {
                floor: 0,
                hard_limit: 100,
                segment_bytes: 1,
            },
            Yield::new(0.5).unwrap(),
        );
        let mut tier = Tier {
            chunks: 100,
            open: 80,
            cap: 100,
        };
        assert_eq!(
            turn(&mut governor, &mut tier, 1.0),
            Signal::TierPressure {
                held_bytes: 80,
                target_bytes: 50
            }
        );
        assert_eq!(tier.cap, 50);
        tier.open = 40;
        assert_eq!(turn(&mut governor, &mut tier, 1.0), Signal::Eased);
    }
}

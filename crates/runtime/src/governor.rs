//! How much of its device pool this process holds while the system around
//! it is pressed for memory.
//!
//! The pool is a balloon held taut against its ceiling (`[engine.memory]
//! budget_device`, what the load measured): with demand it takes all of
//! it, and the system's pressure presses it in as far as the deployment's
//! `yield` lets, one continuous figure:
//!
//! ```text
//! target = floor + (ceiling - floor) * (1 - yield * pressure)
//! ```
//!
//! `pressure` is the system's own reading in `[0, 1]` ([`pressure`]),
//! read when the system says it changed; `floor` is what the running
//! requests need to make progress ([`PoolTarget::floor`]). Nothing goes
//! back on its own: memory leaves only under pressure, and what leaves goes
//! in the order [`Give::ORDER`] states, the policy's keys ordering it
//! within each. What running requests hold is never given back.
//!
//! The loop decides how much, never what: the pool behind [`PoolTarget`]
//! decides how its segments move, and which entries go is the policy's.
//! At `yield = 0`, and wherever nothing reads the system (a browser tab),
//! the target is the ceiling and the loop decides nothing.

#[cfg(target_os = "macos")]
pub mod macos;
pub mod psi;
pub mod run;
pub mod steered;

use std::sync::OnceLock;
use std::time::Duration;

/// The system's memory pressure as it last said it, in `[0, 1]`: 0 where
/// nothing stalls, 1 where the system can give no more.
pub type Reading = tokio::sync::watch::Receiver<f64>;

/// This platform's pressure reading, one for every engine's loop; `None`
/// where the platform offers none (a browser tab, a kernel without PSI),
/// so pressure reads 0 there.
#[must_use]
pub fn pressure() -> Option<Reading> {
    static READING: OnceLock<Option<Reading>> = OnceLock::new();
    READING.get_or_init(watch).clone()
}

#[cfg(target_os = "linux")]
fn watch() -> Option<Reading> {
    psi::watch()
}

#[cfg(target_os = "macos")]
fn watch() -> Option<Reading> {
    macos::watch()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn watch() -> Option<Reading> {
    None
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// The pool's ceiling: what the load may take of the device.
    pub hard_limit: u64,
    /// The pool's grow and shrink unit.
    pub segment_bytes: u64,
}

impl Limits {
    /// The limits of `pool`.
    #[must_use]
    pub fn of(pool: &impl PoolTarget) -> Self {
        Self {
            hard_limit: pool.hard_limit(),
            segment_bytes: pool.segment_bytes().max(1),
        }
    }

    /// The segments `target` bytes come to, rounded up, never past the
    /// ceiling's.
    #[must_use]
    pub const fn segments(&self, target: u64) -> u64 {
        let target = if target < self.hard_limit {
            target
        } else {
            self.hard_limit
        };
        target.div_ceil(self.segment_bytes)
    }
}

/// How a pool's memory can move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Growth {
    /// All of it is there from load and stays: nothing can be given back.
    Fixed,
    /// Segments commit and release.
    Elastic,
}

/// What a pool made of a retarget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retarget {
    /// It holds the segments asked for, or is on its way there.
    Reached,
    /// A shrink found no room to evacuate its emptiest segment into: it
    /// keeps `held` segments, and nothing it holds was dropped.
    NoRoom { held: u64 },
}

/// What a pressed pool gives back, in [`Give::ORDER`]: the runtime fixes
/// the order of the kinds, the policy's keys the order within each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Give {
    /// Idle sealed prefixes and checkpoint rows whose keys say they may go
    /// (at the retention floor), lowest key first; with no policy, least
    /// recently hit first, decoded prefixes before prompt-only ones.
    Cache,
    /// The weight cache above its staging floor.
    Weights,
    /// Idle prefixes keyed above the floor, moved down a tier rather than
    /// dropped.
    Demote,
    /// The activation rung: fires narrower, where the engine backs its span
    /// on demand. Where the span is backed from the load, a narrower rung
    /// frees nothing.
    Rung,
}

impl Give {
    pub const ORDER: [Give; 4] = [Give::Cache, Give::Weights, Give::Demote, Give::Rung];
}

/// The pool a governor steers: it says what it can and must hold and
/// takes a number of segments; how they move is its own.
pub trait PoolTarget {
    /// The pool's grow and shrink unit, in bytes.
    fn segment_bytes(&self) -> u64;
    /// Bytes the running requests need to make progress: the reserved
    /// chunks, the least the activation can come down to, the live state
    /// (and never less than one working set at the declared context, so a
    /// request can always start), and the weights' pinned staging bytes.
    /// Idle cache is no part of it.
    fn floor(&self) -> u64;
    /// The whole of what the pool can take, in bytes.
    fn hard_limit(&self) -> u64;
    /// What the pool holds now, in bytes.
    fn held(&self) -> u64;
    fn growth(&self) -> Growth;
    /// Moves the pool toward `segments`. A shrink evacuates the emptiest
    /// segment and then releases it, and stops where there is no room.
    fn retarget(&mut self, segments: u64) -> Retarget;
    /// Whether a commit the system refused stands: until room is freed, the
    /// target stays at what the pool holds.
    fn refused(&self) -> bool {
        false
    }
    /// Gives back up to `bytes` of `what`: the bytes freed, or on their way.
    fn give(&mut self, _what: Give, _bytes: u64) -> u64 {
        0
    }
    /// A tick's turn is done: what changes only once a tick, as which
    /// weights are resident, changes now.
    fn ticked(&mut self) {}
}

/// One pool's governor: the target its ceiling, floor, `yield` and the
/// pressure come to.
#[derive(Debug, Clone, Copy)]
pub struct Governor {
    limits: Limits,
    share: f64,
}

impl Governor {
    /// `share` is the deployment's `yield`, in `[0, 1]`.
    #[must_use]
    pub const fn new(limits: Limits, share: f64) -> Self {
        Self { limits, share }
    }

    #[must_use]
    pub const fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Whether pressure moves the target at all.
    #[must_use]
    pub fn yields(&self) -> bool {
        self.share > 0.0
    }

    /// `floor + (ceiling - floor) * (1 - yield * pressure)`, never under the
    /// floor nor past the ceiling.
    #[must_use]
    pub fn target(&self, floor: u64, pressure: f64) -> u64 {
        let ceiling = self.limits.hard_limit;
        let floor = floor.min(ceiling);
        let kept = 1.0 - self.share * pressure.clamp(0.0, 1.0);
        let room = (ceiling - floor) as f64 * kept;
        floor + (room as u64).min(ceiling - floor)
    }
}

/// How often the loop ticks: what settles on it is the weight cache's
/// windows, the rung walk's patience and the ledger's reading; a yielding
/// pool re-reads its floor on it while pressed. No yield is paced by it.
pub const TICK: Duration = Duration::from_millis(250);

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const LIMITS: Limits = Limits {
        hard_limit: 1000 * MIB,
        segment_bytes: MIB,
    };
    const FLOOR: u64 = 200 * MIB;

    #[test]
    fn the_target_is_the_floor_plus_what_yield_and_pressure_leave_of_the_room() {
        // (yield, pressure, target): the room above the floor is 800 MiB.
        for (share, pressure, target) in [
            (0.0, 0.0, 1000.0),
            (0.0, 0.5, 1000.0),
            (0.0, 1.0, 1000.0),
            (0.3, 0.0, 1000.0),
            (0.3, 0.5, 200.0 + 800.0 * 0.85),
            (0.3, 1.0, 200.0 + 800.0 * 0.7),
            (1.0, 0.0, 1000.0),
            (1.0, 0.5, 600.0),
            (1.0, 1.0, 200.0),
        ] {
            let governor = Governor::new(LIMITS, share);
            let got = governor.target(FLOOR, pressure);
            // A byte either way is the float's rounding.
            assert!(
                got.abs_diff((target * MIB as f64) as u64) <= 1,
                "yield {share}, pressure {pressure}: {got}"
            );
        }
    }

    #[test]
    fn a_reading_past_its_range_is_its_bound() {
        let governor = Governor::new(LIMITS, 1.0);
        assert_eq!(governor.target(FLOOR, 2.0), FLOOR);
        assert_eq!(governor.target(FLOOR, -1.0), LIMITS.hard_limit);
    }

    #[test]
    fn a_floor_past_the_ceiling_is_the_ceiling() {
        let governor = Governor::new(LIMITS, 1.0);
        assert_eq!(governor.target(2000 * MIB, 1.0), LIMITS.hard_limit);
    }

    #[test]
    fn the_segments_round_up_and_stop_at_the_ceiling() {
        assert_eq!(LIMITS.segments(MIB + 1), 2);
        assert_eq!(LIMITS.segments(u64::MAX), 1000);
    }
}

//! How much of its kv pool this process may hold while the system around it
//! is under load.
//!
//! One congestion loop: the system is watched, not estimated. A process
//! cannot read how much memory other work would like, only whether the
//! system is struggling, so it probes upward while the system is calm and
//! backs off when it is not. It watches demand as well: memory the process
//! stopped asking for goes back a little at a time, with or without anyone
//! else wanting it. The loop answers in bytes alone; which pages go is the
//! planner's to decide.
//!
//! The loop decides how much, never what: the pool behind [`PoolTarget`]
//! decides how its segments move, and what stays is the policy's. Short of
//! room to give a segment back, the pool keeps it and the loop raises
//! [`Signal::TierPressure`]; nothing here drops state.
//!
//! A platform supplies two things and no policy: the hard limit, and whether
//! the system is congested ([`Pressure`]). A platform that cannot tell reads
//! as calm, so demand alone moves the target there. Every threshold here was
//! measured on one 16 GB machine; none is claimed for a larger one.

#[cfg(target_os = "macos")]
pub mod macos;
pub mod psi;
pub mod run;
pub mod steered;

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// What the platform says about the system right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pressure {
    /// Nothing is struggling.
    Calm,
    /// The system is working to make room, or what it pushed out is coming
    /// straight back.
    Congested,
    /// Hand memory back now: the next step is the system killing something.
    Critical,
}

/// A platform's reader of how the system is doing, a tick at a time.
pub trait System: Send {
    fn read(&mut self) -> Pressure;
}

/// Where nothing says how the system is doing, as in a browser tab.
#[cfg(not(target_os = "macos"))]
struct Silent;

#[cfg(not(target_os = "macos"))]
impl System for Silent {
    fn read(&mut self) -> Pressure {
        Pressure::Calm
    }
}

/// A test hook: while the file `PIE_PRESSURE_FILE` names says `critical`,
/// `congested` or `calm`, that is how the system is read, so a test can
/// induce pressure mid-run; anything else reads calm.
struct Told(std::path::PathBuf);

impl System for Told {
    fn read(&mut self) -> Pressure {
        match std::fs::read_to_string(&self.0).as_deref().map(str::trim) {
            Ok("critical") => Pressure::Critical,
            Ok("congested") => Pressure::Congested,
            _ => Pressure::Calm,
        }
    }
}

/// This platform's reader.
#[must_use]
pub fn system() -> Box<dyn System> {
    if let Some(path) = std::env::var_os("PIE_PRESSURE_FILE") {
        return Box::new(Told(path.into()));
    }
    #[cfg(target_os = "macos")]
    return Box::<macos::Watch>::default();
    #[cfg(not(target_os = "macos"))]
    psi::Watch::open().map_or_else(
        || Box::new(Silent) as Box<dyn System>,
        |watch| Box::new(watch) as Box<dyn System>,
    )
}

/// How much congestion this process puts up with, the one axis a deployment
/// sets. Whether one axis is enough is unproven.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Tolerance {
    /// The machine is this process's: the loop never lowers the target and
    /// growth is unlimited. What a server has always done.
    #[default]
    Dedicated,
    /// Backs off only as the system nears critical.
    Aggressive,
    Balanced,
    /// Yields at the first sign of congestion.
    Polite,
}

impl Tolerance {
    /// The share of the target kept on a congested tick, and the ticks of
    /// congestion in a row it takes to act.
    const fn backoff(self) -> (f64, u32) {
        match self {
            Self::Dedicated => (1.0, u32::MAX),
            Self::Aggressive => (0.9, 4),
            Self::Balanced => (0.8, 2),
            Self::Polite => (0.7, 1),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Below this the process cannot make progress.
    pub floor: u64,
    pub hard_limit: u64,
    /// The most one calm tick adds.
    pub step: u64,
    /// The least one idle cut takes: a pool that moves by copying pays for
    /// every cut, so small ones are not worth making. Never less than a
    /// segment, since a cut smaller than one gives nothing back.
    pub shrink_step: u64,
    /// The pool's grow and shrink unit.
    pub segment_bytes: u64,
}

impl Limits {
    /// The limits `pool` puts on a loop that adds at most `step` a tick. A
    /// fixed pool never cuts.
    #[must_use]
    pub fn of(pool: &impl PoolTarget, step: u64) -> Self {
        let segment_bytes = pool.segment_bytes().max(1);
        Self {
            floor: pool.floor(),
            hard_limit: pool.hard_limit(),
            step,
            shrink_step: match pool.growth() {
                Growth::Fixed => u64::MAX,
                Growth::Elastic { least_cut } => least_cut.max(segment_bytes),
            },
            segment_bytes,
        }
    }

    /// The segments `target` bytes come to: rounded up, and never fewer than
    /// the floor takes.
    #[must_use]
    pub const fn segments(&self, target: u64) -> u64 {
        let target = if target > self.floor {
            target
        } else {
            self.floor
        };
        target.div_ceil(self.segment_bytes)
    }
}

/// How a pool's memory can move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Growth {
    /// All of it is there from load and stays: the process is dedicated.
    Fixed,
    /// Segments commit and release; a cut takes at least `least_cut` bytes,
    /// and at least a segment whatever this says.
    Elastic { least_cut: u64 },
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

/// The pool a governor steers: it says how much is asked of it and takes a
/// number of segments; how they move is its own. It offers no way to drop
/// state, and the governor asks for none.
pub trait PoolTarget {
    /// What is asked of the pool now, in bytes.
    fn demand(&mut self) -> Demand;
    /// The pool's grow and shrink unit, in bytes.
    fn segment_bytes(&self) -> u64;
    /// Bytes below which one lane cannot run: one working set at the
    /// declared context, and the activation of the smallest rung.
    fn floor(&self) -> u64;
    /// The whole of what the pool can take, in bytes.
    fn hard_limit(&self) -> u64;
    /// What the pool holds now, in bytes.
    fn held(&self) -> u64;
    fn growth(&self) -> Growth;
    /// Moves the pool toward `segments`. A shrink evacuates the emptiest
    /// segment and then releases it, and stops where there is no room.
    fn retarget(&mut self, segments: u64) -> Retarget;
    /// Whether the system refused memory since this last asked: the system
    /// pushing back, read as [`Pressure::Critical`].
    fn take_refused(&mut self) -> bool {
        false
    }
    /// The system is critical: fires take the next smaller activation rung.
    fn shrink_rung(&mut self);
    /// The system has settled: fires may take their full rung again.
    fn restore_rung(&mut self);
    /// A tick's turn is done (not a waiter's): what changes only once a
    /// tick, as which weights are resident, changes now.
    fn ticked(&mut self) {}
}

/// What the loop tells the runtime, which maps it to the policy's
/// `tier-pressure`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// The pool holds `held_bytes` against a target of `target_bytes` and
    /// has no room to give the difference back.
    TierPressure { held_bytes: u64, target_bytes: u64 },
    /// The pool reached its target again.
    Eased,
}

/// What the process could use right now, in bytes.
#[derive(Debug, Clone, Copy)]
pub struct Demand {
    /// All of it, what it merely keeps included: the target grows this far.
    pub wanted: u64,
    /// Without the idle cache: the target comes back this far.
    pub live: u64,
}

/// Calm ticks to wait after a backoff before probing up again. A push-back
/// quadruples it and a calm stretch halves it, so a probe that keeps failing
/// waits longer each round and a run of calm forgets it.
const FIRST_HOLD: u32 = 4;
const LONGEST_HOLD: u32 = 256;

/// Calm ticks a hold of [`FIRST_HOLD`] takes to count as settled; a longer
/// hold takes as many times longer.
const SETTLED: u32 = 64;

/// Calm ticks with `live` not rising before the target starts back down:
/// long enough that a pause between requests keeps its memory.
const IDLE_AFTER: u32 = 120;

/// Ticks between cuts once idle, and the share of the gap each one takes.
const CUT_EVERY: u32 = 4;
const CUT_SHARE: u64 = 8;

/// Calm ticks with `live` not rising before the idle cache goes too. What it
/// keeps costs a prompt's prefill to lose, which a pause is no reason to pay.
const KEEP_CACHE: u32 = 2400;

pub struct Governor {
    limits: Limits,
    tolerance: Tolerance,
    target: u64,
    congested_ticks: u32,
    hold: u32,
    held: u32,
    calm_ticks: u32,
    yielding: bool,
    idle_ticks: u32,
    last_live: u64,
}

impl Governor {
    /// `held` is what the process has now: a pool that starts whole starts
    /// the target there, and one that starts empty at the floor.
    #[must_use]
    ///
    /// # Panics
    ///
    /// If a cut can be smaller than a segment.
    pub fn new(limits: Limits, tolerance: Tolerance, held: u64) -> Self {
        assert!(
            limits.shrink_step >= limits.segment_bytes && limits.segment_bytes > 0,
            "a cut of {} bytes is smaller than a segment of {}",
            limits.shrink_step,
            limits.segment_bytes,
        );
        let target = match tolerance {
            Tolerance::Dedicated => limits.hard_limit,
            _ => held.clamp(limits.floor, limits.hard_limit),
        };
        Self {
            limits,
            tolerance,
            target,
            congested_ticks: 0,
            hold: 0,
            held: 0,
            calm_ticks: 0,
            yielding: false,
            idle_ticks: 0,
            last_live: 0,
        }
    }

    #[must_use]
    pub const fn target(&self) -> u64 {
        self.target
    }

    #[must_use]
    pub const fn limits(&self) -> &Limits {
        &self.limits
    }

    #[must_use]
    pub const fn tolerance(&self) -> Tolerance {
        self.tolerance
    }

    /// Whether the system has pushed back and not yet stayed calm. While it
    /// is, the process holds to less than it could use.
    #[must_use]
    pub const fn yielding(&self) -> bool {
        self.yielding
    }

    /// Whether the system is calm and nothing has asked for more in a while.
    #[must_use]
    pub const fn idle(&self) -> bool {
        self.idle_ticks >= IDLE_AFTER
    }

    /// One tick. The target rises only toward demand, so memory nothing asked
    /// for is not taken on the strength of a calm system, and it falls back
    /// toward demand once that stops rising.
    pub fn tick(&mut self, pressure: Pressure, demand: Demand) -> u64 {
        if self.tolerance == Tolerance::Dedicated {
            return self.target;
        }
        if pressure != Pressure::Calm {
            self.idle_ticks = 0;
        }
        match pressure {
            Pressure::Critical => {
                self.target = self.limits.floor;
                self.backed_off();
            }
            Pressure::Congested => {
                self.calm_ticks = 0;
                self.congested_ticks += 1;
                let (keep, patience) = self.tolerance.backoff();
                if self.congested_ticks >= patience {
                    self.congested_ticks = 0;
                    self.target = ((self.target as f64 * keep) as u64).max(self.limits.floor);
                    self.backed_off();
                }
            }
            Pressure::Calm => self.calm(demand),
        }
        self.target
    }

    /// One stretch of pressure is one failure, however many ticks it lasts;
    /// only a push-back after calm has been counted makes the wait longer.
    fn backed_off(&mut self) {
        if self.held > 0 || self.hold == 0 {
            self.hold = (self.hold * 4).clamp(FIRST_HOLD, LONGEST_HOLD);
        }
        self.held = 0;
        self.calm_ticks = 0;
        self.yielding = true;
    }

    fn calm(&mut self, demand: Demand) {
        self.congested_ticks = 0;
        self.calm_ticks = self.calm_ticks.saturating_add(1);
        if self.calm_ticks >= SETTLED.saturating_mul(self.hold / FIRST_HOLD).max(SETTLED) {
            self.hold /= 2;
            self.calm_ticks = 0;
            self.yielding = false;
        }
        self.idle_ticks = if demand.live > self.last_live {
            0
        } else {
            self.idle_ticks.saturating_add(1)
        };
        self.last_live = demand.live;
        let kept = self.kept(demand);
        if self.idle() && (self.idle_ticks - IDLE_AFTER).is_multiple_of(CUT_EVERY) {
            self.cut(kept);
        }
        if self.held < self.hold {
            self.held += 1;
            return;
        }
        self.rise(kept);
    }

    /// Whether a waiter can be served by a rise instead of an eviction.
    #[must_use]
    pub const fn can_rise(&self) -> bool {
        !self.yielding && self.held >= self.hold && self.target < self.limits.hard_limit
    }

    /// A waiter is short now: the rise the next calm tick would make, without
    /// waiting for it. A tick reads the system; this only spends what the
    /// last one allowed.
    pub fn asked(&mut self, demand: Demand) -> u64 {
        if self.can_rise() {
            self.idle_ticks = 0;
            self.rise(demand.wanted);
        }
        self.target
    }

    /// How far the target stays up: what the cache keeps is on its way out
    /// only once it has gone unused for long.
    const fn kept(&self, demand: Demand) -> u64 {
        if self.idle_ticks >= KEEP_CACHE {
            demand.live
        } else {
            demand.wanted
        }
    }

    fn rise(&mut self, demand: u64) {
        if demand > self.target {
            let room = demand
                .min(self.limits.hard_limit)
                .saturating_sub(self.target);
            self.target += room.min(self.limits.step);
        }
    }

    fn cut(&mut self, kept: u64) {
        let gap = self.target.saturating_sub(kept.max(self.limits.floor));
        self.target -= (gap / CUT_SHARE).max(self.limits.shrink_step).min(gap);
    }
}

/// How often the loop looks. Critical waits for no more than this.
pub const TICK: Duration = Duration::from_millis(250);

/// Whether what the system pushed out is coming back: decompressions and
/// swap-ins against compressions and swap-outs over a tick. Paging out alone
/// is health, memory going idle to make room; paging back in what just went
/// out is congestion. Below `MIN_EVICTED` pages a tick the ratio is noise.
#[must_use]
pub fn refaulting(evicted: u64, returned: u64) -> bool {
    const MIN_EVICTED: u64 = 2_000;
    const THRASHING: f64 = 0.5;
    evicted >= MIN_EVICTED && returned as f64 / evicted as f64 >= THRASHING
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const LIMITS: Limits = Limits {
        floor: 100 * MIB,
        hard_limit: 1000 * MIB,
        step: 50 * MIB,
        shrink_step: 10 * MIB,
        segment_bytes: MIB,
    };

    const fn all(bytes: u64) -> Demand {
        Demand {
            wanted: bytes,
            live: bytes,
        }
    }

    fn calm_to(governor: &mut Governor, ticks: u32, demand: u64) {
        for _ in 0..ticks {
            governor.tick(Pressure::Calm, all(demand));
        }
    }

    #[test]
    fn a_dedicated_process_holds_the_hard_limit_whatever_happens() {
        let mut governor = Governor::new(LIMITS, Tolerance::Dedicated, 0);
        assert_eq!(governor.tick(Pressure::Critical, all(0)), LIMITS.hard_limit);
        assert_eq!(
            governor.tick(Pressure::Congested, all(0)),
            LIMITS.hard_limit
        );
        assert_eq!(governor.tick(Pressure::Calm, all(0)), LIMITS.hard_limit);
    }

    #[test]
    fn a_pool_that_starts_whole_starts_the_target_there() {
        let governor = Governor::new(LIMITS, Tolerance::Balanced, LIMITS.hard_limit);
        assert_eq!(governor.target(), LIMITS.hard_limit);
    }

    #[test]
    fn calm_grows_by_a_step_toward_demand_and_no_further() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, 0);
        assert_eq!(governor.tick(Pressure::Calm, all(400 * MIB)), 150 * MIB);
        calm_to(&mut governor, 100, 400 * MIB);
        assert_eq!(governor.target(), 400 * MIB);
        calm_to(&mut governor, 10, 300 * MIB);
        assert_eq!(governor.target(), 400 * MIB, "low demand does not cut");
    }

    #[test]
    fn growth_stops_at_the_hard_limit() {
        let mut governor = Governor::new(LIMITS, Tolerance::Aggressive, 0);
        calm_to(&mut governor, 100, u64::MAX);
        assert_eq!(governor.target(), LIMITS.hard_limit);
    }

    #[test]
    fn critical_cuts_to_the_floor_at_once() {
        let mut governor = Governor::new(LIMITS, Tolerance::Aggressive, 0);
        calm_to(&mut governor, 100, u64::MAX);
        assert_eq!(
            governor.tick(Pressure::Critical, all(u64::MAX)),
            LIMITS.floor
        );
    }

    #[test]
    fn congestion_backs_off_by_the_tolerance_and_stops_at_the_floor() {
        let mut governor = Governor::new(LIMITS, Tolerance::Polite, 0);
        calm_to(&mut governor, 100, u64::MAX);
        assert_eq!(
            governor.tick(Pressure::Congested, all(u64::MAX)),
            700 * MIB,
            "a polite process acts on the first tick"
        );
        for _ in 0..64 {
            governor.tick(Pressure::Congested, all(u64::MAX));
        }
        assert_eq!(governor.target(), LIMITS.floor);
    }

    #[test]
    fn a_balanced_process_waits_out_one_congested_tick() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, 0);
        calm_to(&mut governor, 100, u64::MAX);
        assert_eq!(
            governor.tick(Pressure::Congested, all(0)),
            LIMITS.hard_limit
        );
        governor.tick(Pressure::Calm, all(0));
        assert_eq!(
            governor.tick(Pressure::Congested, all(0)),
            LIMITS.hard_limit
        );
        assert_eq!(governor.tick(Pressure::Congested, all(0)), 800 * MIB);
    }

    #[test]
    fn a_probe_that_fails_soon_waits_longer_before_the_next() {
        let mut governor = Governor::new(LIMITS, Tolerance::Polite, 0);
        calm_to(&mut governor, 100, u64::MAX);
        let held_for = |governor: &mut Governor| {
            let backed = governor.tick(Pressure::Congested, all(u64::MAX));
            let mut ticks = 0;
            while governor.tick(Pressure::Calm, all(u64::MAX)) == backed {
                ticks += 1;
            }
            ticks
        };
        let first = held_for(&mut governor);
        let second = held_for(&mut governor);
        assert!(second > first);
    }

    #[test]
    fn one_stretch_of_pressure_is_one_failure() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, LIMITS.hard_limit);
        for _ in 0..8 {
            governor.tick(Pressure::Critical, all(u64::MAX));
        }
        assert_eq!(governor.hold, FIRST_HOLD);
    }

    #[test]
    fn yielding_ends_only_after_a_calm_as_long_as_the_failures_earned() {
        let mut governor = Governor::new(LIMITS, Tolerance::Polite, 0);
        governor.tick(Pressure::Congested, all(0));
        assert!(governor.yielding());
        calm_to(&mut governor, SETTLED - 1, 0);
        assert!(governor.yielding());
        calm_to(&mut governor, 1, 0);
        assert!(!governor.yielding(), "one round settles in one stretch");

        governor.tick(Pressure::Congested, all(0));
        calm_to(&mut governor, SETTLED, 0);
        assert!(governor.yielding(), "a second round in a row takes longer");
    }

    #[test]
    fn a_long_calm_forgets_the_backoff() {
        let mut governor = Governor::new(LIMITS, Tolerance::Polite, 0);
        governor.tick(Pressure::Congested, all(u64::MAX));
        calm_to(&mut governor, 1000, u64::MAX);
        assert_eq!(governor.hold, 0);
    }

    /// A process that kept `wanted`, of which only `live` is in use, over
    /// `ticks` calm ticks.
    fn kept(governor: &mut Governor, ticks: u32, wanted: u64, live: u64) -> u64 {
        for _ in 0..ticks {
            governor.tick(Pressure::Calm, Demand { wanted, live });
        }
        governor.target()
    }

    #[test]
    fn a_pause_keeps_its_memory() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, LIMITS.hard_limit);
        assert_eq!(kept(&mut governor, IDLE_AFTER - 1, 0, 0), LIMITS.hard_limit);
    }

    #[test]
    fn idle_memory_goes_back_a_little_at_a_time_as_far_as_the_floor() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, LIMITS.hard_limit);
        kept(&mut governor, IDLE_AFTER, 0, 0);
        let first = governor.target();
        assert_eq!(first, LIMITS.hard_limit - 900 * MIB / CUT_SHARE);
        let next = kept(&mut governor, CUT_EVERY, 0, 0);
        assert!(next < first && first - next <= 900 * MIB / CUT_SHARE);
        assert_eq!(kept(&mut governor, 2000, 0, 0), LIMITS.floor);
    }

    #[test]
    fn a_cut_stops_at_what_the_cache_keeps_until_it_has_gone_unused_for_long() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, LIMITS.hard_limit);
        let (keeps, in_use) = (600 * MIB, 300 * MIB);
        assert_eq!(kept(&mut governor, KEEP_CACHE - 1, keeps, in_use), keeps);
        assert_eq!(kept(&mut governor, 400, keeps, in_use), in_use);
    }

    #[test]
    fn a_pool_that_moves_by_copying_cuts_once() {
        let limits = Limits {
            shrink_step: u64::MAX,
            ..LIMITS
        };
        let mut governor = Governor::new(limits, Tolerance::Balanced, limits.hard_limit);
        assert_eq!(kept(&mut governor, IDLE_AFTER, 0, 0), limits.floor);
    }

    #[test]
    fn a_waiter_gets_the_rise_at_once_and_ends_the_idle() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, LIMITS.floor);
        // The first tick sees `live` rise from nothing.
        kept(&mut governor, IDLE_AFTER + 1, LIMITS.floor, LIMITS.floor);
        assert!(governor.idle());
        assert_eq!(
            governor.asked(all(LIMITS.floor + 20 * MIB)),
            LIMITS.floor + 20 * MIB
        );
        assert!(!governor.idle());
    }

    #[test]
    fn a_waiter_gets_no_rise_while_the_system_pushes_back() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, LIMITS.hard_limit);
        governor.tick(Pressure::Critical, all(0));
        assert!(!governor.can_rise());
        assert_eq!(governor.asked(all(900 * MIB)), LIMITS.floor);
    }

    #[test]
    fn a_dedicated_process_is_never_asked() {
        let governor = Governor::new(LIMITS, Tolerance::Dedicated, 0);
        assert!(!governor.can_rise());
    }

    #[test]
    fn a_request_ends_the_idle_and_the_target_grows_again() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, LIMITS.hard_limit);
        kept(&mut governor, 2000, 0, 0);
        assert!(governor.idle());
        governor.tick(Pressure::Calm, all(400 * MIB));
        assert!(!governor.idle());
        assert_eq!(governor.target(), LIMITS.floor + LIMITS.step);
    }

    #[test]
    fn pressure_ends_the_idle() {
        let mut governor = Governor::new(LIMITS, Tolerance::Balanced, LIMITS.hard_limit);
        kept(&mut governor, IDLE_AFTER, 0, 0);
        assert!(governor.idle());
        governor.tick(Pressure::Congested, all(0));
        assert!(!governor.idle());
    }

    #[test]
    fn a_dedicated_process_gives_nothing_back_however_idle() {
        let mut governor = Governor::new(LIMITS, Tolerance::Dedicated, 0);
        assert_eq!(kept(&mut governor, 2000, 0, 0), LIMITS.hard_limit);
    }

    #[test]
    #[should_panic(expected = "smaller than a segment")]
    fn a_cut_smaller_than_a_segment_is_refused_at_construction() {
        let limits = Limits {
            segment_bytes: 64 * MIB,
            ..LIMITS
        };
        let _ = Governor::new(limits, Tolerance::Balanced, 0);
    }

    #[test]
    fn a_target_comes_to_whole_segments_and_never_under_the_floor() {
        let limits = Limits {
            segment_bytes: 64 * MIB,
            shrink_step: 64 * MIB,
            ..LIMITS
        };
        assert_eq!(limits.segments(0), 2, "the floor's 100 MiB takes two");
        assert_eq!(limits.segments(128 * MIB), 2);
        assert_eq!(limits.segments(128 * MIB + 1), 3);
        assert_eq!(limits.segments(LIMITS.hard_limit), 16);
    }

    #[test]
    fn refault_needs_real_eviction_and_real_return() {
        assert!(!refaulting(0, 0));
        assert!(!refaulting(100, 100), "too few pages to mean anything");
        assert!(!refaulting(10_000, 1_000), "pushed out and stays out");
        assert!(refaulting(10_000, 8_000));
    }
}

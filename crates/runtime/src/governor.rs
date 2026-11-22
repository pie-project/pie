//! How much of its device pool this process holds while the system around
//! it is under memory pressure.
//!
//! One rule, no loop state: the target is
//!
//! ```text
//! target = floor + (budget - floor) * (1 - yield * pressure)
//! ```
//!
//! `floor` is what one lane needs at least ([`PoolTarget::floor`]), `budget`
//! the device budget the load holds ([`PoolTarget::hard_limit`]), `yield`
//! the deployment's `[engine.memory] yield` in `[0, 1]` ([`Yield`]) and
//! `pressure` the operating system's own memory-pressure reading in `0..=1`
//! ([`Reader`] over `engine::pressure`'s [`Source`]s, which the host tier
//! shares), read at the cadence the system computes it at. `yield = 0`
//! (the default) is a dedicated machine: the target is the budget whatever
//! the system says, and nothing is read.
//!
//! The rule decides how much, never what: the pool behind [`PoolTarget`]
//! decides how its segments move, and what stays is the policy's. Short of
//! room to give a segment back, the pool sheds its weight cache and then
//! raises [`Signal::TierPressure`] for PLEX, whose keys say which idle
//! prefixes go; the activation rung comes down last, once the walk finds the
//! target under what it holds and no room to evacuate into. Nothing here
//! drops state, and nothing here decides how long idle cache stays: that is
//! the policy's retention keys' (`policy::retention`).

pub mod run;
pub mod steered;

use std::time::Duration;

pub use ::engine::pressure::{Source, Yield, YieldOutOfRange};

/// Reads this system's memory pressure ([`::engine::pressure::Source`]) on
/// the governor's tick, no more often than the system recomputes it (PSI's
/// 2 s); a source that states no period (macOS's available memory, which
/// has no notification of its own) is read every tick, the loop's [`TICK`]
/// being its only clock. A reading that fails reads `0` and is named once.
pub struct Reader {
    source: Source,
    last: f64,
    /// The source's own value at the last read that reached it, until
    /// [`Reader::take_fresh`] takes it.
    fresh: Option<f64>,
    at: Option<crate::rt::Instant>,
    failed: bool,
}

/// The pressure source a load's `yield` needs: none at a yield of 0, where
/// nothing is read. A yield above 0 that has no source to follow is refused
/// by name, never left to yield nothing.
///
/// # Errors
///
/// `yield` > 0 and `source` (this system's, [`Source::of_this_system`])
/// did not answer.
pub fn source_for(yields: Yield, source: impl FnOnce() -> Result<Source, String>) -> Result<Option<Source>, String> {
    if !yields.yields() {
        return Ok(None);
    }
    source().map(Some).map_err(|why| {
        format!(
            "engine.memory.yield = {} needs the system's memory pressure, and {why}",
            yields.share()
        )
    })
}

/// The yield a load runs at, and the line its load log states it in. A
/// deployment's `[engine.memory] yield` is kept as configured. Unset, it is
/// `1` where an engine the governor steers states unified memory (the pool
/// and the host share one RAM, so a pool that never yields starves the
/// system and the device's own command buffers with it), and `0` (a
/// dedicated machine) on separate device memory.
#[must_use]
pub fn yield_for(configured: Option<Yield>, unified_memory: bool) -> (Yield, String) {
    match configured {
        Some(yields) => (
            yields,
            format!("engine.memory.yield = {}: as configured", yields.share()),
        ),
        None if unified_memory => (
            Yield::FULL,
            "engine.memory.yield = 1: unified memory, the pool and the host share RAM".to_string(),
        ),
        None => (
            Yield::NONE,
            "engine.memory.yield = 0: separate device memory, the pool holds its budget".to_string(),
        ),
    }
}

impl Reader {
    #[must_use]
    pub const fn new(source: Source) -> Self {
        Self {
            source,
            last: 0.0,
            fresh: None,
            at: None,
            failed: false,
        }
    }

    #[must_use]
    pub const fn source(&self) -> &Source {
        &self.source
    }

    /// The pressure now, in `0..=1`.
    pub fn read(&mut self) -> f64 {
        if let (Some(at), Some(period)) = (self.at, self.source.period())
            && at.elapsed() < period
        {
            return self.last;
        }
        self.at = Some(crate::rt::Instant::now());
        match self.source.sample() {
            Ok(sample) => {
                self.failed = false;
                self.last = sample.pressure;
                self.fresh = Some(sample.raw);
            }
            Err(why) => {
                if !self.failed {
                    tracing::error!(
                        source = self.source.name(),
                        %why,
                        "the memory pressure could not be read: it reads 0 until it can"
                    );
                }
                self.failed = true;
                self.last = 0.0;
            }
        }
        self.last
    }

    /// The source's own value, where the last [`Self::read`] reached the
    /// source rather than its period's cache; once.
    pub fn take_fresh(&mut self) -> Option<f64> {
        self.fresh.take()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Below this the process cannot make progress.
    pub floor: u64,
    /// The device budget the load holds: the most the pool may take.
    pub hard_limit: u64,
    /// The pool's grow and shrink unit.
    pub segment_bytes: u64,
}

impl Limits {
    /// The limits `pool` states. The floor is its byte floor, or the
    /// fewest segments it can map where those come to more: a target under
    /// them is one no shed reaches.
    #[must_use]
    pub fn of(pool: &impl PoolTarget) -> Self {
        Self {
            floor: pool.floor().max(pool.least_mapped()).min(pool.hard_limit()),
            hard_limit: pool.hard_limit(),
            segment_bytes: pool.segment_bytes().max(1),
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

    /// The rule ([`::engine::memory::target`]) on this pool: its floor,
    /// and its hard limit as the ceiling.
    #[must_use]
    pub fn target(&self, yields: Yield, pressure: f64) -> u64 {
        ::engine::memory::target(self.floor, self.hard_limit, yields.share(), pressure)
    }
}

/// How a pool's memory can move: what the engine states of its device
/// pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Growth {
    /// All of it is there from load and stays (one buffer sized at load, as
    /// XLA's): no yield applies.
    Fixed,
    /// Segments commit and release.
    Elastic,
}

/// What a pool made of a retarget, judged on what it maps
/// ([`PoolTarget::mapped`]) against the segments asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retarget {
    /// It maps no more than the segments asked (a grow may still be on its
    /// way).
    Reached,
    /// It maps `held` segments, more than asked, and gives the rest back
    /// on its own: an evacuation or a release under way, or a walk yet to
    /// take its next step. Nothing is asked of the policy.
    Moving { held: u64 },
    /// It maps `held` segments, more than asked, and cannot give the rest
    /// back unless the policy sheds: live state, or idle cache, that no
    /// evacuation has room for. Nothing it holds was dropped; the policy's
    /// tier pressure sheds idle cache for the difference.
    NoRoom { held: u64 },
}

/// What a pool maps: the measure its [`Retarget`] answer is judged on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mapped {
    /// Segments with memory under them.
    pub segments: u64,
    /// Of those, the ones an evacuation drained, waiting on the fire that
    /// moved their chunks to land and the fires that read them there to
    /// retire before they go.
    pub draining: u64,
}

/// The pool a governor steers: it takes a number of segments; how they
/// move is its own. It offers no way to drop state, and the governor asks
/// for none.
pub trait PoolTarget {
    /// The pool's grow and shrink unit, in bytes.
    fn segment_bytes(&self) -> u64;
    /// Bytes below which one lane cannot run: one working set at the
    /// declared context, and the activation of the smallest rung.
    fn floor(&self) -> u64;
    /// The bytes of the fewest segments the pool can map, as its own
    /// layout states them (what no shed brings it under); 0 where its
    /// layout sets no such bound.
    fn least_mapped(&self) -> u64 {
        0
    }
    /// The whole of what the pool can take, in bytes.
    fn hard_limit(&self) -> u64;
    /// What the pool holds now, in bytes.
    fn held(&self) -> u64;
    /// What the pool maps now, in segments: [`Self::held`]'s, where it
    /// names no draining segments of its own.
    fn mapped(&self) -> Mapped {
        Mapped {
            segments: self.held().div_ceil(self.segment_bytes().max(1)),
            draining: 0,
        }
    }
    fn growth(&self) -> Growth;
    /// Moves the pool toward `segments`. A shrink evacuates the emptiest
    /// segment and then releases it, and stops where there is no room.
    /// [`Retarget::Reached`] only where it maps no more than `segments`
    /// ([`Self::mapped`]); short of that, [`Retarget::Moving`] while it
    /// gives the rest back on its own, [`Retarget::NoRoom`] once it cannot.
    fn retarget(&mut self, segments: u64) -> Retarget;
    /// Whether the system refused memory since this last asked: the system
    /// saying it can give no more, read as pressure 1.
    fn take_refused(&mut self) -> bool {
        false
    }
    /// A tick's turn is done: what changes only once a tick, as which
    /// weights are resident, changes now.
    fn ticked(&mut self) {}
}

/// What the loop tells the runtime, which maps it to the policy's
/// `tier-pressure`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// The pool holds `held_bytes` against a target of `target_bytes` and
    /// has no room to give the difference back.
    TierPressure { held_bytes: u64, target_bytes: u64 },
    /// Nothing is asked of the policy any more: the pool reached its
    /// target, or gives the rest back on its own ([`Retarget::Moving`]).
    Eased,
}

/// The rule on one pool.
pub struct Governor {
    limits: Limits,
    yields: Yield,
    target: u64,
}

impl Governor {
    /// A pool that cannot move ([`Growth::Fixed`]) takes no yield: its size
    /// is the load's.
    #[must_use]
    pub fn new(limits: Limits, yields: Yield) -> Self {
        Self {
            limits,
            yields,
            target: limits.hard_limit,
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
    pub const fn yields(&self) -> Yield {
        self.yields
    }

    /// One reading of the system's pressure, in `0..=1`.
    pub fn tick(&mut self, pressure: f64) -> u64 {
        self.target = self.limits.target(self.yields, pressure);
        self.target
    }
}

/// How often the loop looks: the host's room, the weight cache's tick, the
/// pool's turn. The pressure [`Reader`] keeps the system's own cadence
/// beneath it ([`Source::period`]).
pub const TICK: Duration = Duration::from_millis(250);

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const LIMITS: Limits = Limits {
        floor: 100 * MIB,
        hard_limit: 1000 * MIB,
        segment_bytes: MIB,
    };

    fn share(yields: f64) -> Yield {
        Yield::new(yields).unwrap()
    }

    #[test]
    fn the_device_target_is_the_rule_on_the_pool_s_floor_and_budget() {
        let mut governor = Governor::new(LIMITS, Yield::NONE);
        assert_eq!(governor.tick(1.0), LIMITS.hard_limit, "yield 0: the budget");
        let mut governor = Governor::new(LIMITS, share(1.0));
        assert_eq!(governor.tick(1.0), LIMITS.floor, "yield 1, pressure 1: the floor");
        let mut governor = Governor::new(LIMITS, share(0.5));
        assert_eq!(governor.tick(0.5), 100 * MIB + 675 * MIB, "in between");
    }

    #[test]
    fn idle_cache_is_not_the_governor_s_to_time_out() {
        // However long nothing is asked, the target is what the rule says:
        // how long idle cache stays is the policy's keys', and with none
        // installed it stays until a demand needs its room.
        let mut governor = Governor::new(LIMITS, share(0.5));
        for _ in 0..100_000 {
            assert_eq!(governor.tick(0.0), LIMITS.hard_limit);
        }
    }

    #[test]
    fn the_reader_reads_its_source_and_a_failure_reads_zero() {
        let dir = std::env::temp_dir().join(format!("pie-reader-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pressure");
        std::fs::write(&path, "0.5").unwrap();
        let mut reader = Reader::new(Source::File(path.clone()));
        assert_eq!(reader.read(), 0.5);
        std::fs::write(&path, "0.25").unwrap();
        assert_eq!(reader.read(), 0.25, "a source with no period is current");
        std::fs::write(&path, "lots").unwrap();
        assert_eq!(reader.read(), 0.0, "a failed reading is 0, named");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_yield_with_no_pressure_to_follow_is_refused_by_name() {
        let gone = Source::File("/nonexistent/pie-pressure".into());
        let refused = source_for(share(0.5), || gone.clone().checked()).expect_err("refused");
        assert!(
            refused.starts_with(
                "engine.memory.yield = 0.5 needs the system's memory pressure, and PIE_PRESSURE_FILE reads "
            ),
            "{refused}"
        );
        let none = || Err("nothing reads it on this platform (windows)".to_string());
        let refused = source_for(share(1.0), none).expect_err("refused");
        assert!(refused.ends_with("nothing reads it on this platform (windows)"), "{refused}");
    }

    #[test]
    fn a_yield_of_zero_reads_nothing_and_loads_without_a_source() {
        let none = || -> Result<Source, String> { panic!("not read at yield 0") };
        assert_eq!(source_for(Yield::NONE, none), Ok(None));
    }

    #[test]
    fn an_unset_yield_is_full_on_unified_memory_and_none_on_separate() {
        let (yields, line) = yield_for(None, true);
        assert_eq!(yields, Yield::FULL);
        assert_eq!(
            line,
            "engine.memory.yield = 1: unified memory, the pool and the host share RAM"
        );
        let (yields, line) = yield_for(Some(Yield::NONE), true);
        assert_eq!(yields, Yield::NONE, "a configured 0 is kept on unified memory");
        assert_eq!(line, "engine.memory.yield = 0: as configured");
        let (yields, line) = yield_for(Some(share(0.5)), false);
        assert_eq!(yields, share(0.5));
        assert_eq!(line, "engine.memory.yield = 0.5: as configured");
        let (yields, _) = yield_for(None, false);
        assert_eq!(yields, Yield::NONE, "separate device memory: the default stays 0");
    }

    #[test]
    fn the_unified_default_reads_the_system_s_pressure() {
        let (yields, _) = yield_for(None, true);
        let file = Source::File("/pie-pressure".into());
        let read = std::cell::Cell::new(false);
        let source = source_for(yields, || {
            read.set(true);
            Ok(file.clone())
        });
        assert!(read.get(), "the system's source is asked");
        assert_eq!(source, Ok(Some(file)));
        let (yields, _) = yield_for(None, false);
        let none = || -> Result<Source, String> { panic!("not read on separate memory") };
        assert_eq!(source_for(yields, none), Ok(None));
    }

    #[test]
    fn a_psi_reader_waits_out_the_kernel_s_period() {
        let mut reader = Reader::new(Source::Psi);
        reader.at = Some(crate::rt::Instant::now());
        reader.last = 0.75;
        assert_eq!(reader.read(), 0.75, "within the period: the last reading");
    }
}

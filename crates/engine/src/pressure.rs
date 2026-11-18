//! The operating system's own memory-pressure reading, and the one rule
//! every tier's target takes from it (user ruling, 2026-11-15):
//!
//! ```text
//! target = floor + (ceiling - floor) * (1 - yield * pressure)
//! ```
//!
//! `floor` is what the tier needs at least, `ceiling` the most it may hold
//! (the device budget the load holds, the host room the host reader
//! leaves), `yield` the deployment's `[engine.memory] yield` in `[0, 1]`
//! ([`Yield`]) and `pressure` a reading in `0..=1` of a [`Source`] the
//! system states ([`host`]). `yield = 0` holds the ceiling whatever the
//! system says. The rule itself is [`crate::memory::target`].
//!
//! Each source is the system's own, named with its documentation; none is
//! thresholded or smoothed here. Where the system states none, there is no
//! [`Source`], and the caller says so and reads `0`, never a guess. This
//! module keeps no clock and logs nothing: a caller reads a source at most
//! once a [`Source::period`] and names a failed reading itself.

use std::path::PathBuf;
use std::time::Duration;

/// How much of a tier's room above its floor goes back to the system's
/// memory pressure (`[engine.memory] yield`): at pressure 1, a yield of 1
/// holds the floor and 0 the whole ceiling.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Yield(f64);

/// A yield outside `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error(
    "engine.memory.yield = {0} is outside [0, 1]: 0 holds the whole budget whatever the \
     system's pressure, 1 gives all of it above the floor back at full pressure"
)]
pub struct YieldOutOfRange(pub f64);

impl Yield {
    /// The machine is this process's: every target is its ceiling.
    pub const NONE: Yield = Yield(0.0);

    /// # Errors
    ///
    /// `share` is not a number in `[0, 1]`.
    pub fn new(share: f64) -> Result<Self, YieldOutOfRange> {
        if (0.0..=1.0).contains(&share) {
            Ok(Self(share))
        } else {
            Err(YieldOutOfRange(share))
        }
    }

    #[must_use]
    pub const fn share(self) -> f64 {
        self.0
    }

    /// Whether the system's pressure moves a target at all.
    #[must_use]
    pub fn yields(self) -> bool {
        self.0 > 0.0
    }
}

/// The OS's host-memory pressure now, in `0..=1`, from this system's
/// [`Source`]; `None` where it states none or the reading failed: the caller
/// states `0` and logs it. The device tier reads the same: the rule's
/// pressure is the OS's, whichever tier yields to it. A caller that reads
/// on a clock keeps a [`Source`] and reads it no more often than
/// [`Source::period`].
#[must_use]
pub fn host() -> Option<f64> {
    Source::of_this_system()?.read().ok()
}

/// Where a reading of the system's memory pressure comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Linux and Android: pressure stall information,
    /// `/proc/pressure/memory` (Documentation/accounting/psi.rst,
    /// <https://docs.kernel.org/accounting/psi.html>). The reading is the
    /// `some avg10` share of wall time in which at least one task stalled
    /// on memory, a percentage over a hundred ([`psi_some_avg10`]).
    Psi,
    /// macOS: `kern.memorystatus_vm_pressure_level`, the level the
    /// `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` dispatch source reports
    /// (<https://developer.apple.com/documentation/dispatch/dispatch_source_type_memorypressure>),
    /// on its own three-step scale ([`macos_level`]).
    MacosLevel,
    /// A test hook: the file `PIE_PRESSURE_FILE` names holds the reading,
    /// a number in `0..=1`, so a test can induce pressure mid-run.
    File(PathBuf),
}

impl Source {
    const PSI: &str = "/proc/pressure/memory";

    /// This system's source: the test hook where `PIE_PRESSURE_FILE` is
    /// set, else the OS's own where it states one (a Linux or Android
    /// kernel with PSI, macOS). `None` where it states none (no PSI,
    /// Windows, iOS, a browser tab): the pressure is then `0`, and the
    /// caller says so.
    #[must_use]
    pub fn of_this_system() -> Option<Source> {
        if let Some(path) = std::env::var_os("PIE_PRESSURE_FILE") {
            return Some(Source::File(path.into()));
        }
        if cfg!(target_os = "macos") {
            return Some(Source::MacosLevel);
        }
        Source::Psi.read().is_ok().then_some(Source::Psi)
    }

    /// The source, named for the load log.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Source::Psi => "Linux PSI /proc/pressure/memory, some avg10",
            Source::MacosLevel => "macOS kern.memorystatus_vm_pressure_level",
            Source::File(_) => "PIE_PRESSURE_FILE",
        }
    }

    /// How often the system recomputes the reading, where it says: a
    /// caller reads no more often. PSI's averages are recomputed every
    /// `PSI_FREQ`, `(2*HZ+1)` jiffies, "2 sec intervals"
    /// (kernel/sched/psi.c). `None`: the reading is current at each read
    /// (macOS's level changes as the kernel's state does; the test file).
    #[must_use]
    pub const fn period(&self) -> Option<Duration> {
        match self {
            Source::Psi => Some(Duration::from_secs(2)),
            Source::MacosLevel | Source::File(_) => None,
        }
    }

    /// The pressure now, in `0..=1`.
    ///
    /// # Errors
    ///
    /// The source did not answer, or answered something that is not a
    /// reading; the error says what.
    pub fn read(&self) -> Result<f64, String> {
        match self {
            Source::Psi => {
                let text = std::fs::read_to_string(Self::PSI)
                    .map_err(|e| format!("{}: {e}", Self::PSI))?;
                psi_some_avg10(&text)
                    .ok_or_else(|| format!("{} states no `some avg10`", Self::PSI))
            }
            Source::MacosLevel => macos::level().and_then(|level| {
                macos_level(level).ok_or_else(|| {
                    format!("kern.memorystatus_vm_pressure_level is {level}, not a level")
                })
            }),
            Source::File(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                text.trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|share| (0.0..=1.0).contains(share))
                    .ok_or_else(|| format!("{} holds no reading in [0, 1]", path.display()))
            }
        }
    }
}

/// The `some avg10` of a PSI pressure file, in `0..=1`; `None` for text
/// that is not one. The `full` line is not the reading.
#[must_use]
pub fn psi_some_avg10(text: &str) -> Option<f64> {
    let percent: f64 = text
        .lines()
        .find(|line| line.starts_with("some "))?
        .split_whitespace()
        .find_map(|field| field.strip_prefix("avg10="))?
        .parse()
        .ok()?;
    (percent.is_finite() && percent >= 0.0).then(|| (percent / 100.0).min(1.0))
}

/// `DISPATCH_MEMORYPRESSURE_NORMAL`, `_WARN`, `_CRITICAL` (dispatch/source.h).
const NORMAL: u32 = 1;
const WARN: u32 = 2;
const CRITICAL: u32 = 4;

/// A macOS pressure level as a reading on the kernel's own three-step
/// scale: normal `0`, warn half way, critical `1`; `None` for a value the
/// kernel does not state.
#[must_use]
pub const fn macos_level(level: u32) -> Option<f64> {
    match level {
        NORMAL => Some(0.0),
        WARN => Some(0.5),
        CRITICAL => Some(1.0),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod macos {
    /// `kern.memorystatus_vm_pressure_level`, read without a dispatch
    /// queue to run on: the value the dispatch source's events carry.
    pub(super) fn level() -> Result<u32, String> {
        let mut level: u32 = 0;
        let mut size = std::mem::size_of::<u32>();
        // SAFETY: the name is a nul-terminated literal and `level` holds
        // `size` bytes; the kernel writes at most that.
        let status = unsafe {
            libc::sysctlbyname(
                c"kern.memorystatus_vm_pressure_level".as_ptr(),
                (&raw mut level).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if status == 0 {
            Ok(level)
        } else {
            Err(format!(
                "kern.memorystatus_vm_pressure_level: {}",
                std::io::Error::last_os_error()
            ))
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod macos {
    pub(super) fn level() -> Result<u32, String> {
        Err("kern.memorystatus_vm_pressure_level is macOS's".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_yield_outside_zero_to_one_is_refused_by_name() {
        for bad in [-0.1, 1.01, f64::NAN, f64::INFINITY] {
            let error = Yield::new(bad).expect_err("out of range");
            assert!(error.to_string().contains("engine.memory.yield"), "{error}");
        }
        assert!(Yield::new(0.0).is_ok() && Yield::new(1.0).is_ok());
    }

    #[test]
    fn psi_reads_the_share_of_time_some_task_stalled() {
        let file = "some avg10=12.50 avg60=3.00 avg300=1.00 total=123456\n\
                    full avg10=4.00 avg60=1.00 avg300=0.50 total=6543\n";
        assert_eq!(psi_some_avg10(file), Some(0.125));
        assert_eq!(
            psi_some_avg10("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"),
            Some(0.0)
        );
        assert_eq!(
            psi_some_avg10("some avg10=100.00 avg60=0.00 avg300=0.00 total=1\n"),
            Some(1.0)
        );
    }

    #[test]
    fn psi_s_full_line_is_not_the_reading() {
        let file = "full avg10=90.00 avg60=0.00 avg300=0.00 total=1\n\
                    some avg10=5.00 avg60=0.00 avg300=0.00 total=1\n";
        assert_eq!(psi_some_avg10(file), Some(0.05));
    }

    #[test]
    fn anything_else_is_not_a_psi_file() {
        assert_eq!(psi_some_avg10(""), None);
        assert_eq!(psi_some_avg10("some avg10=lots\n"), None);
        assert_eq!(psi_some_avg10("some avg60=1.00 total=1\n"), None);
        assert_eq!(psi_some_avg10("some avg10=-1.00 avg60=0.00\n"), None);
    }

    #[test]
    fn macos_levels_read_on_the_kernel_s_scale() {
        assert_eq!(macos_level(NORMAL), Some(0.0));
        assert_eq!(macos_level(WARN), Some(0.5));
        assert_eq!(macos_level(CRITICAL), Some(1.0));
        assert_eq!(macos_level(3), None);
    }

    #[test]
    fn the_test_file_is_a_source_of_its_own() {
        let dir = std::env::temp_dir().join(format!("pie-pressure-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pressure");
        let source = Source::File(path.clone());
        std::fs::write(&path, "0.25\n").unwrap();
        assert_eq!(source.read(), Ok(0.25));
        std::fs::write(&path, "2").unwrap();
        assert!(source.read().is_err(), "out of range is no reading");
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(source.period(), None);
        assert_eq!(Source::Psi.period(), Some(Duration::from_secs(2)));
    }
}

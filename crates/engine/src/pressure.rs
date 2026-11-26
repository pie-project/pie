//! The operating system's own memory-pressure reading, and the one rule
//! every tier's target takes from it:
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
//! Each source is the system's own continuous reading, named with its
//! documentation; none is thresholded, mapped or smoothed here. Where the
//! system states none, or its source does not read, a yield above 0 has
//! nothing to follow: [`Source::of_this_system`] says why, and the load is
//! refused by name rather than yielding nothing. This module keeps no clock
//! and logs nothing: a caller reads a source at most once a
//! [`Source::period`] and names a failed reading itself.

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

    /// All of the pool above the floor goes back at full pressure.
    pub const FULL: Yield = Yield(1.0);

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
/// [`Source`]; `None` where it states none or the reading failed. A tier
/// that yields to it refuses its load where [`Source::of_this_system`]
/// does not answer, as the device governor does. The device tier reads the same: the rule's
/// pressure is the OS's, whichever tier yields to it. A caller that reads
/// on a clock keeps a [`Source`] and reads it no more often than
/// [`Source::period`].
#[must_use]
pub fn host() -> Option<f64> {
    Source::of_this_system().ok()?.read().ok()
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
    /// macOS: `kern.memorystatus_level`, the kernel's own continuous
    /// reading of the memory available, a percentage (xnu
    /// bsd/kern/kern_memorystatus.c, `memorystatus_level`; what Activity
    /// Monitor's memory pressure graph shows as its complement), readable
    /// unprivileged on every macOS pie supports. The reading is
    /// `1 - level / 100` ([`macos_available`]).
    MacosAvailable,
    /// macOS, beside this process: `kern.memorystatus_level` with this
    /// process's own physical footprint counted as available (its pool,
    /// weights and guests), so the reading is the pressure the rest of the
    /// system puts on the machine ([`macos_available_beside`]). For a load
    /// whose engine already holds its pool within the room its own host
    /// side leaves (`Engine::host_room`): the pool yields to other
    /// processes, never to its own load.
    MacosBesideOwn,
    /// Windows: `GlobalMemoryStatusEx`'s physical memory, the share of it
    /// available now (`ullAvailPhys` over `ullTotalPhys`), the level Task
    /// Manager's memory graph shows; read when asked, as macOS's level is.
    /// The reading is `1 - available / total` ([`windows_available`]).
    WindowsAvailable,
    /// A test hook: the file `PIE_PRESSURE_FILE` names holds the reading,
    /// a number in `0..=1`, so a test can induce pressure mid-run.
    File(PathBuf),
}

impl Source {
    const PSI: &str = "/proc/pressure/memory";

    /// This system's source, read once to show it reads: the test hook
    /// where `PIE_PRESSURE_FILE` is set, else the OS's own (macOS's or
    /// Windows' available memory; a Linux or Android kernel's PSI).
    ///
    /// # Errors
    ///
    /// The source does not read (`"<source> reads <error>"`), or this
    /// platform has none (iOS, a browser tab).
    pub fn of_this_system() -> Result<Source, String> {
        let source = if let Some(path) = std::env::var_os("PIE_PRESSURE_FILE") {
            Source::File(path.into())
        } else if cfg!(target_os = "macos") {
            Source::MacosAvailable
        } else if cfg!(any(target_os = "linux", target_os = "android")) {
            Source::Psi
        } else if cfg!(windows) {
            Source::WindowsAvailable
        } else {
            return Err(format!(
                "nothing reads it on this platform ({})",
                std::env::consts::OS
            ));
        };
        source.checked()
    }

    /// This system's source with this process's own memory discounted
    /// ([`Source::MacosBesideOwn`]), where the OS's signal allows it (a
    /// level of available memory, which a footprint can be added back to);
    /// the test hook where `PIE_PRESSURE_FILE` is set.
    ///
    /// # Errors
    ///
    /// This system's signal cannot discount the process (Linux PSI is a
    /// share of stalled time, not a level), or it does not read.
    pub fn of_this_system_beside_own() -> Result<Source, String> {
        let source = if let Some(path) = std::env::var_os("PIE_PRESSURE_FILE") {
            Source::File(path.into())
        } else if cfg!(target_os = "macos") {
            Source::MacosBesideOwn
        } else {
            return Err(format!(
                "this system's memory pressure cannot discount the process's own memory ({})",
                std::env::consts::OS
            ));
        };
        source.checked()
    }

    /// This source if it reads now; else `"<source> reads <error>"`.
    ///
    /// # Errors
    ///
    /// It does not read.
    pub fn checked(self) -> Result<Source, String> {
        match self.read() {
            Ok(_) => Ok(self),
            Err(why) => Err(format!("{} reads {why}", self.name())),
        }
    }

    /// The source, named for the load log.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Source::Psi => "Linux PSI /proc/pressure/memory, some avg10",
            Source::MacosAvailable => "macOS kern.memorystatus_level, 1 - available / 100",
            Source::MacosBesideOwn => {
                "macOS kern.memorystatus_level beside this process: 1 - (available + its \
                 footprint) / 100"
            }
            Source::WindowsAvailable => {
                "Windows GlobalMemoryStatusEx, 1 - available physical / total physical"
            }
            Source::File(_) => "PIE_PRESSURE_FILE",
        }
    }

    /// How often the system recomputes the reading, where it says: a
    /// caller reads no more often. PSI's averages are recomputed every
    /// `PSI_FREQ`, `(2*HZ+1)` jiffies, "2 sec intervals"
    /// (kernel/sched/psi.c). `None`: the reading is current at each read
    /// and the caller's own tick is the only clock. macOS states no
    /// notification for its continuous `kern.memorystatus_level` (the
    /// dispatch memory-pressure source fires only on the three-step level's
    /// changes, which a tick sees anyway), and a sysctl read is cheap, so it
    /// is read on each tick of the governor's loop; Windows' available
    /// physical memory is current at each `GlobalMemoryStatusEx` call, and
    /// read likewise; the test file likewise.
    #[must_use]
    pub const fn period(&self) -> Option<Duration> {
        match self {
            Source::Psi => Some(Duration::from_secs(2)),
            Source::MacosAvailable
            | Source::MacosBesideOwn
            | Source::WindowsAvailable
            | Source::File(_) => None,
        }
    }

    /// The pressure now, in `0..=1`.
    ///
    /// # Errors
    ///
    /// The source did not answer, or answered something that is not a
    /// reading; the error says what.
    pub fn read(&self) -> Result<f64, String> {
        self.sample().map(|sample| sample.pressure)
    }

    /// The pressure now, beside the source's own value it came from (PSI's
    /// `some avg10` percentage, macOS's or Windows' percent available, the
    /// test file's number), for a log that shows both.
    ///
    /// # Errors
    ///
    /// As [`Self::read`].
    pub fn sample(&self) -> Result<Sample, String> {
        match self {
            Source::Psi => {
                let text = std::fs::read_to_string(Self::PSI)
                    .map_err(|e| format!("{}: {e}", Self::PSI))?;
                let pressure = psi_some_avg10(&text)
                    .ok_or_else(|| format!("{} states no `some avg10`", Self::PSI))?;
                Ok(Sample {
                    raw: pressure * 100.0,
                    pressure,
                })
            }
            Source::MacosAvailable => {
                let level = macos::sysctl_u32(macos::AVAILABLE)?;
                let pressure = macos_available(level).ok_or_else(|| {
                    format!("kern.memorystatus_level is {level}, not a percentage")
                })?;
                Ok(Sample {
                    raw: f64::from(level),
                    pressure,
                })
            }
            Source::MacosBesideOwn => {
                let level = macos::sysctl_u32(macos::AVAILABLE)?;
                let total = macos::physical_memory()?;
                let own = macos::footprint()?;
                let pressure = macos_available_beside(level, own, total).ok_or_else(|| {
                    format!("kern.memorystatus_level is {level}, not a percentage")
                })?;
                Ok(Sample {
                    raw: f64::from(level),
                    pressure,
                })
            }
            Source::WindowsAvailable => {
                let (available, total) = windows::physical_memory()?;
                let pressure = windows_available(available, total).ok_or_else(|| {
                    format!("GlobalMemoryStatusEx states {available} of {total} bytes available")
                })?;
                Ok(Sample {
                    raw: (1.0 - pressure) * 100.0,
                    pressure,
                })
            }
            Source::File(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let pressure = text
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|share| (0.0..=1.0).contains(share))
                    .ok_or_else(|| format!("{} holds no reading in [0, 1]", path.display()))?;
                Ok(Sample {
                    raw: pressure,
                    pressure,
                })
            }
        }
    }
}

/// One reading of a [`Source`]: the source's own value and the pressure in
/// `0..=1` it reads as.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// What the source states: PSI's `some avg10` percent, macOS's or
    /// Windows' percent available, the test file's number.
    pub raw: f64,
    pub pressure: f64,
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

/// `kern.memorystatus_level` (percent available) as a reading: `1 -
/// level / 100`; `None` past 100.
#[must_use]
pub fn macos_available(level: u32) -> Option<f64> {
    (level <= 100).then(|| 1.0 - f64::from(level) / 100.0)
}

/// [`macos_available`] with `own` bytes of a `total`-byte machine counted as
/// available too (this process's footprint): `1 - min(100, level + own /
/// total * 100) / 100`; `None` past 100 or with no total.
#[must_use]
pub fn macos_available_beside(level: u32, own: u64, total: u64) -> Option<f64> {
    if level > 100 || total == 0 {
        return None;
    }
    let available = f64::from(level) + own as f64 / total as f64 * 100.0;
    Some(1.0 - available.min(100.0) / 100.0)
}

/// Windows' available physical memory as a reading: `1 - available /
/// total`; `None` with no total or more available than the total.
#[must_use]
pub fn windows_available(available: u64, total: u64) -> Option<f64> {
    (total > 0 && available <= total).then(|| 1.0 - available as f64 / total as f64)
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows {
    /// `MEMORYSTATUSEX`.
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        _memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        _total_page_file: u64,
        _avail_page_file: u64,
        _total_virtual: u64,
        _avail_virtual: u64,
        _avail_extended_virtual: u64,
    }

    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    /// The machine's available and total physical memory, in bytes.
    pub(super) fn physical_memory() -> Result<(u64, u64), String> {
        // SAFETY: a zeroed `MEMORYSTATUSEX` is valid; its length is set
        // before the call, as the call requires.
        let mut status: MemoryStatusEx = unsafe { std::mem::zeroed() };
        status.length = std::mem::size_of::<MemoryStatusEx>() as u32;
        // SAFETY: `status` is a `MEMORYSTATUSEX` with its length set, owned
        // here for the call.
        if unsafe { GlobalMemoryStatusEx(&raw mut status) } == 0 {
            return Err(format!(
                "GlobalMemoryStatusEx: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok((status.avail_phys, status.total_phys))
    }
}

#[cfg(not(windows))]
mod windows {
    pub(super) fn physical_memory() -> Result<(u64, u64), String> {
        Err("GlobalMemoryStatusEx is Windows'".to_string())
    }
}

#[cfg(test)]
mod beside_tests {
    use super::macos_available_beside;

    /// The process's own footprint counts as available: 29% available
    /// with a 14 GiB process on a 32 GiB machine reads as 72.75% available,
    /// a pressure of 0.2725 where the whole-system reading was 0.71.
    #[test]
    fn the_process_s_own_footprint_reads_as_available() {
        let gib = 1u64 << 30;
        let p = macos_available_beside(29, 14 * gib, 32 * gib).expect("a reading");
        assert!((p - 0.2725).abs() < 1e-9, "{p}");
        assert_eq!(macos_available_beside(29, 0, 32 * gib), Some(0.71));
        assert_eq!(
            macos_available_beside(90, 32 * gib, 32 * gib),
            Some(0.0),
            "capped at 100%"
        );
        assert_eq!(macos_available_beside(101, 0, 32 * gib), None);
        assert_eq!(macos_available_beside(50, 0, 0), None);
    }
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod macos {
    pub(super) const AVAILABLE: &std::ffi::CStr = c"kern.memorystatus_level";

    /// A `u32` sysctl by name.
    pub(super) fn sysctl_u32(name: &std::ffi::CStr) -> Result<u32, String> {
        let mut value: u32 = 0;
        let mut size = std::mem::size_of::<u32>();
        // SAFETY: `name` is nul-terminated and `value` holds `size` bytes;
        // the kernel writes at most that, and says how much it did.
        let status = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&raw mut value).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != 0 {
            return Err(format!(
                "{}: {}",
                name.to_string_lossy(),
                std::io::Error::last_os_error()
            ));
        }
        if size != std::mem::size_of::<u32>() {
            return Err(format!(
                "{}: {size} bytes, not a u32",
                name.to_string_lossy()
            ));
        }
        Ok(value)
    }

    /// The machine's physical memory (`hw.memsize`).
    pub(super) fn physical_memory() -> Result<u64, String> {
        let mut value: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        // SAFETY: as `sysctl_u32`, for the u64 `hw.memsize`.
        let status = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut value).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != 0 || size != std::mem::size_of::<u64>() {
            return Err(format!("hw.memsize: {}", std::io::Error::last_os_error()));
        }
        Ok(value)
    }

    /// This process's physical footprint (`proc_pid_rusage`,
    /// `ri_phys_footprint`): what the kernel charges it, its GPU memory
    /// included on unified memory.
    pub(super) fn footprint() -> Result<u64, String> {
        // SAFETY: `info` is the `rusage_info_v2` the RUSAGE_INFO_V2 flavor
        // writes, owned here for the call.
        let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        let status = unsafe {
            libc::proc_pid_rusage(libc::getpid(), libc::RUSAGE_INFO_V2, (&raw mut info).cast())
        };
        if status != 0 {
            return Err(format!(
                "proc_pid_rusage: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(info.ri_phys_footprint)
    }
}

#[cfg(not(target_os = "macos"))]
mod macos {
    pub(super) const AVAILABLE: &str = "kern.memorystatus_level";

    pub(super) fn sysctl_u32(name: &str) -> Result<u32, String> {
        Err(format!("{name} is macOS's"))
    }

    pub(super) fn physical_memory() -> Result<u64, String> {
        Err("hw.memsize is macOS's".to_string())
    }

    pub(super) fn footprint() -> Result<u64, String> {
        Err("a process footprint is read on macOS only".to_string())
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
}

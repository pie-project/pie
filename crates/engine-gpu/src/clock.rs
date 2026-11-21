//! The GPU's clock table, where the platform states it: the floor a fire
//! may run at under a stated submit bound (`crate::submit`).
//!
//! A phone's GPU caps its clock as it heats, in steps its thermal governor
//! picks: an Adreno 840's cap fell 1300 to 578 MHz in one step, 0.6 s into
//! a fire admitted at its full-clock estimate times every ratio measured
//! before it, and the fire ran past KGSL's 2 s. No ratio measured before a
//! fire sees a drop inside it. So an admitted fire is held to the bound at
//! the lowest clock the device's own table states: its estimate (timed at
//! full clock at load) times the table's greatest over its least
//! ([`ClockRange::floor_factor`]). Read once, at load.
//!
//! Read where the platform states it, unprivileged:
//!
//! - Android's KGSL (Qualcomm's GPU driver), under `/sys/class/kgsl/
//!   kgsl-3d0/`: `gpu_available_frequencies` (Hz), else
//!   `devfreq/available_frequencies` (Hz), else `freq_table_mhz` (MHz);
//!   readable by adb's shell uid;
//! - Linux devfreq, for a GPU no KGSL drives: each `/sys/class/devfreq/*`
//!   device naming a GPU (its name says `gpu`, `kgsl` or `mali`, as the
//!   SoCs' GPU nodes are named): `available_frequencies`, else `min_freq`
//!   and `max_freq`.
//!
//! A discrete GPU states none there, nor does a sandbox that denies the
//! read (an app uid's SELinux may): the load says so and keeps the
//! measured ratios alone.

use std::path::{Path, PathBuf};

/// The GPU's clock table as the platform states it: its least and greatest
/// clock, Hz, and where it was read.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockRange {
    pub min_hz: u64,
    pub max_hz: u64,
    /// Where the table was read.
    pub source: PathBuf,
}

impl std::fmt::Display for ClockRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}-{} MHz ({})",
            self.min_hz / 1_000_000,
            self.max_hz / 1_000_000,
            self.source.display()
        )
    }
}

impl ClockRange {
    /// A table of `min_hz` to `max_hz` read nowhere: a test's.
    #[must_use]
    pub fn stated(min_hz: u64, max_hz: u64) -> ClockRange {
        ClockRange {
            min_hz,
            max_hz,
            source: PathBuf::from("stated"),
        }
    }

    /// The table the platform states, else why none is read.
    pub fn read() -> Result<ClockRange, String> {
        Self::read_under(Path::new("/sys/class"))
    }

    /// [`ClockRange::read`] under `class` (`/sys/class`, or a test's tree).
    pub fn read_under(class: &Path) -> Result<ClockRange, String> {
        let kgsl = class.join("kgsl/kgsl-3d0");
        let kgsl_lists = [
            (kgsl.join("gpu_available_frequencies"), 1),
            (kgsl.join("devfreq/available_frequencies"), 1),
            (kgsl.join("freq_table_mhz"), 1_000_000),
        ];
        for (path, unit) in &kgsl_lists {
            if let Some((min, max)) = read_list(path, *unit) {
                return Ok(ClockRange {
                    min_hz: min,
                    max_hz: max,
                    source: path.clone(),
                });
            }
        }
        let devfreq = class.join("devfreq");
        let mut gpus: Vec<PathBuf> = std::fs::read_dir(&devfreq)
            .map(|dir| {
                dir.filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .filter(|path| {
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(names_a_gpu)
                    })
                    .collect()
            })
            .unwrap_or_default();
        gpus.sort();
        for dir in &gpus {
            if let Some((min, max)) = read_list(&dir.join("available_frequencies"), 1) {
                return Ok(ClockRange {
                    min_hz: min,
                    max_hz: max,
                    source: dir.join("available_frequencies"),
                });
            }
            if let (Some(min), Some(max)) =
                (read_hz(&dir.join("min_freq")), read_hz(&dir.join("max_freq")))
                && max >= min
            {
                return Ok(ClockRange {
                    min_hz: min,
                    max_hz: max,
                    source: dir.clone(),
                });
            }
        }
        Err(format!(
            "no GPU clock table is readable here: neither {}'s gpu_available_frequencies, \
             devfreq/available_frequencies nor freq_table_mhz, nor a GPU's node under {}",
            kgsl.display(),
            devfreq.display()
        ))
    }

    /// The table's greatest clock over its least: how much longer a fire
    /// timed at full clock runs at the floor.
    #[must_use]
    pub fn floor_factor(&self) -> f64 {
        self.max_hz as f64 / self.min_hz.max(1) as f64
    }
}

/// Whether a devfreq device's name says it is a GPU.
fn names_a_gpu(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    ["gpu", "kgsl", "mali"].iter().any(|word| name.contains(word))
}

/// The least and greatest of a list of clocks (whitespace-separated, any
/// order, each `unit` Hz); `None` where it holds no positive one.
#[must_use]
pub fn parse_frequencies(text: &str, unit: u64) -> Option<(u64, u64)> {
    let hz: Vec<u64> = text
        .split_whitespace()
        .filter_map(|word| word.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .map(|v| v.saturating_mul(unit))
        .collect();
    Some((*hz.iter().min()?, *hz.iter().max()?))
}

fn read_list(path: &Path, unit: u64) -> Option<(u64, u64)> {
    parse_frequencies(&std::fs::read_to_string(path).ok()?, unit)
}

fn read_hz(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|&hz| hz > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Adreno 840's KGSL table, fastest first: its least to its
    /// greatest; a table in MHz reads in Hz.
    #[test]
    fn a_frequency_list_reads_its_least_and_greatest() {
        let levels = "1300000000 1200000000 1100000000 1050000000 1000000000 902000000 \
                      826000000 726000000 646000000 578000000 500000000 160000000\n";
        assert_eq!(parse_frequencies(levels, 1), Some((160_000_000, 1_300_000_000)));
        assert_eq!(parse_frequencies("1300 578 160", 1_000_000), Some((160_000_000, 1_300_000_000)));
        assert_eq!(parse_frequencies("", 1), None);
        assert_eq!(parse_frequencies("0 x", 1), None);
    }

    /// KGSL's table first, then a devfreq device naming a GPU; nothing
    /// readable: why, by name.
    #[test]
    fn the_table_is_read_from_kgsl_or_a_gpu_devfreq_node() {
        let root = std::env::temp_dir().join(format!("pie-clock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let write = |path: &Path, text: &str| {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        };
        std::fs::create_dir_all(&root).unwrap();
        let why = ClockRange::read_under(&root).unwrap_err();
        assert!(why.contains("no GPU clock table"), "{why}");
        // A devfreq GPU node with min / max only; a memory node ignored.
        write(&root.join("devfreq/ddr/available_frequencies"), "100 200\n");
        write(&root.join("devfreq/fb000000.gpu/min_freq"), "200000000\n");
        write(&root.join("devfreq/fb000000.gpu/max_freq"), "800000000\n");
        let range = ClockRange::read_under(&root).unwrap();
        assert_eq!((range.min_hz, range.max_hz), (200_000_000, 800_000_000));
        assert_eq!(range.floor_factor(), 4.0);
        // KGSL's MHz table over it, and its Hz table over that.
        write(&root.join("kgsl/kgsl-3d0/freq_table_mhz"), "1300 578 160");
        assert_eq!(ClockRange::read_under(&root).unwrap().min_hz, 160_000_000);
        write(
            &root.join("kgsl/kgsl-3d0/gpu_available_frequencies"),
            "1300000000 578000000 160000000",
        );
        let range = ClockRange::read_under(&root).unwrap();
        assert_eq!((range.min_hz, range.max_hz), (160_000_000, 1_300_000_000));
        assert!(range.source.ends_with("gpu_available_frequencies"));
        assert!((range.floor_factor() - 8.125).abs() < 1e-9);
        let _ = std::fs::remove_dir_all(&root);
    }
}

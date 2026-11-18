//! What the host can give a load: one rule for every engine that keeps a
//! host tier.
//!
//! The host gives the smaller of two things:
//! - what the kernel counts available (`MemAvailable` in `/proc/meminfo`);
//! - what the process's memory cgroup leaves under its limit.
//!
//! A cgroup's usage counts the page cache its reads left behind, which
//! reclaim hands back on demand, so its reclaimable file pages are netted
//! out of the usage, as `MemAvailable` does for the host. Both cgroup
//! hierarchies are read: v2's `memory.max` / `memory.current` /
//! `memory.stat`, else v1's `memory.limit_in_bytes` /
//! `memory.usage_in_bytes` / `memory.stat`. A limit stated as `max` (v2), or
//! as v1's near-2^63 "unlimited", leaves the host to decide.
//!
//! An engine that has a host tier and is told no budget for it seats
//! [`host_kv_ceiling`] of this. A phone's or an integrated GPU's device
//! memory is the host's and is sized by its own rule (lmkd's defended line,
//! `engine-gpu`'s `host_memory`). This one is for a tier beside a device
//! that has memory of its own.

use crate::memory::SAFETY_FLOOR_BYTES;

/// A memory cgroup's limit, usage and the reclaimable file pages in that
/// usage, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cgroup {
    pub max: u64,
    pub current: u64,
    pub reclaimable: u64,
}

impl Cgroup {
    /// What the group leaves under its limit: the limit less what it holds
    /// past its reclaimable page cache.
    #[must_use]
    pub fn room(&self) -> u64 {
        self.max
            .saturating_sub(self.current.saturating_sub(self.reclaimable))
    }
}

/// The host's memory as a load reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemory {
    /// `MemAvailable`, in bytes.
    pub available: u64,
    /// The process's memory cgroup, where it states a limit.
    pub cgroup: Option<Cgroup>,
}

impl HostMemory {
    /// What the host can give: the smaller of what it has available and
    /// what the cgroup leaves.
    #[must_use]
    pub fn free(&self) -> u64 {
        self.cgroup
            .map_or(self.available, |group| self.available.min(group.room()))
    }

    /// The host's memory from the texts it is read from: `/proc/meminfo`,
    /// and the cgroup's limit, usage and `memory.stat` where it has them.
    ///
    /// # Errors
    ///
    /// Names the reading the rule needed and could not take: no
    /// `MemAvailable`, or a stated limit with no usage beside it.
    pub fn of(
        meminfo: &str,
        max: Option<&str>,
        current: Option<&str>,
        stat: Option<&str>,
    ) -> Result<HostMemory, String> {
        let available = field(meminfo, "MemAvailable:")
            .and_then(|kib| kib.checked_mul(1024))
            .ok_or_else(|| "/proc/meminfo states no MemAvailable".to_string())?;
        let number = |text: Option<&str>| -> Option<u64> { text?.trim().parse().ok() };
        // `max` (v2) states no limit; a number is one.
        let cgroup = match number(max) {
            None => None,
            Some(max) => {
                let current = number(current).ok_or_else(|| {
                    format!(
                        "the memory cgroup states a limit of {max} bytes and no usage beside it \
                         (memory.current / memory.usage_in_bytes)"
                    )
                })?;
                let stat = stat.unwrap_or("");
                let reclaimable = field(stat, "inactive_file ")
                    .unwrap_or(0)
                    .saturating_add(field(stat, "active_file ").unwrap_or(0));
                Some(Cgroup {
                    max,
                    current,
                    reclaimable,
                })
            }
        };
        Ok(HostMemory { available, cgroup })
    }

    /// The host's memory now.
    ///
    /// # Errors
    ///
    /// Names the reading that could not be taken ([`HostMemory::of`]), or
    /// a host with no `/proc/meminfo`.
    pub fn read() -> Result<HostMemory, String> {
        let read = |path: &str| std::fs::read_to_string(path).ok();
        let meminfo = read("/proc/meminfo")
            .ok_or_else(|| "the host has no /proc/meminfo to read its memory from".to_string())?;
        // cgroup v2 names the limit `memory.max`; a v1 hierarchy (many docker
        // hosts still) names it `memory.limit_in_bytes`, its usage
        // `memory.usage_in_bytes`, and keeps its own counters under `memory/`.
        let (max, current, stat) = match read("/sys/fs/cgroup/memory.max") {
            Some(max) => (
                Some(max),
                read("/sys/fs/cgroup/memory.current"),
                read("/sys/fs/cgroup/memory.stat"),
            ),
            None => (
                read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                read("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
                read("/sys/fs/cgroup/memory/memory.stat"),
            ),
        };
        HostMemory::of(
            &meminfo,
            max.as_deref(),
            current.as_deref(),
            stat.as_deref(),
        )
    }
}

/// `key`'s number in `text` (`MemAvailable:   123 kB`, `inactive_file 123`):
/// the line's own key, not a `total_` one.
fn field(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?;
        rest.trim().trim_end_matches("kB").trim().parse().ok()
    })
}

/// The host kv tier's ceiling: what the host can give
/// ([`HostMemory::free`]) less the floor ([`SAFETY_FLOOR_BYTES`]), shared
/// by `ranks`. A tier is seated at its ceiling and commits as it fills
/// (every engine's tier is reserved, never touched ahead); its live target
/// under host-memory pressure is the governor's, by the same `yield` rule
/// as the device's (`new-memory/DEVICE-MEMORY-INVENTORY.md`).
#[must_use]
pub fn host_kv_ceiling(host: &HostMemory, ranks: u32) -> u64 {
    host.free().saturating_sub(SAFETY_FLOOR_BYTES) / u64::from(ranks.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn with_no_cgroup_the_host_decides() {
        let host = HostMemory::of("MemAvailable:   16777216 kB\n", None, None, None).unwrap();
        assert_eq!(host.free(), 16 * GIB);
        // v2 states `max`: no limit.
        let host =
            HostMemory::of("MemAvailable:   16777216 kB\n", Some("max\n"), None, None).unwrap();
        assert_eq!(host.cgroup, None);
        assert_eq!(host.free(), 16 * GIB);
    }

    /// This box: 251 GB of RAM, a 62 GB cgroup holding 40 GB of which 30 GB
    /// is page cache. The cgroup decides, its page cache given back.
    #[test]
    fn a_cgroup_under_the_hosts_available_decides_net_of_its_page_cache() {
        let meminfo = "MemTotal:       263000000 kB\nMemAvailable:   230000000 kB\n";
        let stat = "anon 9000000000\nfile 30000000000\ninactive_file 20000000000\n\
                    active_file 10000000000\ntotal_inactive_file 7\n";
        let host = HostMemory::of(
            meminfo,
            Some("62000000000\n"),
            Some("40000000000\n"),
            Some(stat),
        )
        .unwrap();
        let room = 62_000_000_000 - (40_000_000_000 - 30_000_000_000);
        assert_eq!(host.cgroup.unwrap().room(), room);
        assert_eq!(host.free(), room);
    }

    /// A docker container on a v1 hierarchy (engine-cuda's case): its limit
    /// caps the host's; v1's "unlimited" leaves the host to decide.
    #[test]
    fn a_v1_cgroup_limit_caps_the_host_and_its_unlimited_does_not() {
        let meminfo = "MemAvailable:   1383000000 kB\n";
        let stat = "cache 0\ninactive_file 0\nactive_file 0\ntotal_inactive_file 99\n";
        let host = HostMemory::of(
            meminfo,
            Some("376999997440\n"),
            Some("1476259840\n"),
            Some(stat),
        )
        .unwrap();
        assert_eq!(host.free(), 376_999_997_440 - 1_476_259_840);
        let host = HostMemory::of(
            meminfo,
            Some("9223372036854771712\n"),
            Some("1476259840\n"),
            Some(stat),
        )
        .unwrap();
        assert_eq!(host.free(), 1_383_000_000 * 1024);
    }

    #[test]
    fn a_reading_that_cannot_be_taken_is_refused_by_name() {
        let none = HostMemory::of("MemTotal: 1 kB\n", None, None, None).unwrap_err();
        assert!(none.contains("MemAvailable"), "{none}");
        let usage =
            HostMemory::of("MemAvailable: 1 kB\n", Some("100\n"), None, None).unwrap_err();
        assert!(usage.contains("no usage"), "{usage}");
    }

    #[test]
    fn the_ceiling_is_the_hosts_room_past_the_floor_shared_by_ranks() {
        let host = HostMemory {
            available: 100 * GIB + SAFETY_FLOOR_BYTES,
            cgroup: None,
        };
        assert_eq!(host_kv_ceiling(&host, 1), 100 * GIB);
        assert_eq!(host_kv_ceiling(&host, 2), 50 * GIB);
        // A cgroup under the host's available decides.
        let boxed = HostMemory {
            available: 230 * GIB,
            cgroup: Some(Cgroup {
                max: 62 * GIB,
                current: 40 * GIB,
                reclaimable: 30 * GIB,
            }),
        };
        assert_eq!(host_kv_ceiling(&boxed, 1), 52 * GIB - SAFETY_FLOOR_BYTES);
    }
}

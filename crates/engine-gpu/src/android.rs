//! What an Android device has for a load: what its memory has available,
//! less what lmkd keeps free past that, less the floor.
//!
//! A phone's GPU takes its memory from system RAM, and a GPU reservation is
//! part of the process like any other. The heap the driver reports is all
//! of RAM, so it says nothing of what is free. And no Android driver reports
//! `VK_EXT_memory_budget`. What this process may take is what the kernel
//! counts available (`MemAvailable` in `/proc/meminfo`), less any reserve
//! lmkd (the low-memory killer daemon) defends beyond it. Taking that
//! reserve would make lmkd kill processes to win it back: the user's apps,
//! or this process.
//!
//! Each reserve counts once. The kernel's `MemAvailable` already leaves out
//! its `totalreserve`: every zone's high watermark plus its largest
//! lowmem reserve, capped at what the zone manages. lmkd defends one of
//! two lines:
//! - With `ro.lmk.use_minfree_levels` set, it kills at the `minfree` levels
//!   `sys.lmk.minfree_levels` names, as `pages:adj` pairs. The line it
//!   defends is the level guarding the lowest adj stated, the foreground
//!   app's: below it, lmkd kills what the user sees. Only what that level
//!   exceeds `totalreserve` by is subtracted, read from `/proc/zoneinfo`.
//! - Otherwise (the default since Android 10, and an SM8850's), lmkd kills
//!   on memory pressure (PSI) once free memory falls under the zones' high
//!   watermarks: the line `MemAvailable` has already left out. Nothing more
//!   is subtracted.
//!
//! The floor is the driver's safety floor every backend keeps.
//!
//! These are read once, when the device binds. A reading that cannot be
//! taken is refused by name.
#![allow(unsafe_code)]

#[cfg(target_os = "android")]
use crate::error::{Fault, Result};

/// The driver's safety floor under the measured capacity: one size on
/// every backend.
pub const SAFETY_FLOOR_BYTES: u64 = 64 << 20;

/// What the system had for a load when it bound, and how that was reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Room {
    /// `MemAvailable` from `/proc/meminfo`.
    pub available: u64,
    /// The bytes lmkd keeps free by killing beyond what `available`
    /// already leaves out.
    pub defended: u64,
    /// What a load may take: `available` less `defended`, less the floor.
    pub room: u64,
}

impl Room {
    #[must_use]
    pub fn of(available: u64, defended: u64) -> Room {
        Room {
            available,
            defended,
            room: available
                .saturating_sub(defended)
                .saturating_sub(SAFETY_FLOOR_BYTES),
        }
    }
}

/// `MemAvailable` in bytes, from `/proc/meminfo`'s text.
#[cfg(any(target_os = "android", test))]
fn mem_available(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        let kib: u64 = rest.trim().strip_suffix("kB")?.trim().parse().ok()?;
        kib.checked_mul(1024)
    })
}

/// The kernel's `totalreserve`, in pages, from `/proc/zoneinfo`'s text:
/// for every zone, its high watermark plus its largest lowmem reserve
/// (`protection`), capped at the pages it manages, as
/// `calculate_totalreserve_pages` sums it. `MemAvailable` leaves it out.
#[cfg(any(target_os = "android", test))]
fn total_reserve(zoneinfo: &str) -> Option<u64> {
    #[derive(Default)]
    struct Zone {
        high: Option<u64>,
        managed: Option<u64>,
        protection: Option<u64>,
    }
    impl Zone {
        fn reserve(&self) -> Option<u64> {
            let held = self.high?.checked_add(self.protection?)?;
            Some(held.min(self.managed?))
        }
    }
    let mut zones: Vec<Zone> = Vec::new();
    for line in zoneinfo.lines() {
        if line.starts_with("Node ") {
            zones.push(Zone::default());
            continue;
        }
        let Some(zone) = zones.last_mut() else {
            continue;
        };
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("protection:") {
            let most = rest
                .trim()
                .trim_start_matches('(')
                .trim_end_matches(')')
                .split(',')
                .map(|n| n.trim().parse::<u64>().ok())
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            zone.protection = Some(most);
            continue;
        }
        let mut words = line.split_whitespace();
        let slot = match words.next() {
            Some("high") => &mut zone.high,
            Some("managed") => &mut zone.managed,
            _ => continue,
        };
        if slot.is_none() {
            *slot = Some(words.next()?.parse().ok()?);
        }
    }
    if zones.is_empty() {
        return None;
    }
    zones
        .iter()
        .try_fold(0u64, |sum, zone| sum.checked_add(zone.reserve()?))
}

/// The `minfree` level, in pages, that guards the lowest adj in a
/// `sys.lmk.minfree_levels` value (`pages:adj,...`).
#[cfg(any(target_os = "android", test))]
fn foreground_minfree(levels: &str) -> Option<u64> {
    levels
        .split(',')
        .map(|level| {
            let (pages, adj) = level.trim().split_once(':')?;
            Some((
                adj.trim().parse::<i32>().ok()?,
                pages.trim().parse::<u64>().ok()?,
            ))
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .min_by_key(|&(adj, _)| adj)
        .map(|(_, pages)| pages)
}

/// What the system has for a load now, read as the module doc says.
///
/// # Errors
///
/// Refuses by name when a reading the rule needs cannot be taken.
#[cfg(target_os = "android")]
pub fn room() -> Result<Room> {
    let refuse = |why: String| Fault::Device {
        call: "reading Android's memory for the load's capacity",
        why,
    };
    let read =
        |path: &str| std::fs::read_to_string(path).map_err(|e| refuse(format!("{path}: {e}")));
    let meminfo = read("/proc/meminfo")?;
    let available = mem_available(&meminfo)
        .ok_or_else(|| refuse("/proc/meminfo states no MemAvailable".into()))?;
    // SAFETY: `sysconf` reads a constant of the running system.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page = u64::try_from(page)
        .ok()
        .filter(|&page| page > 0)
        .ok_or_else(|| refuse(format!("the page size reads {page}")))?;
    let defended_pages = if property("ro.lmk.use_minfree_levels").as_deref() == Some("true") {
        let levels = property("sys.lmk.minfree_levels").ok_or_else(|| {
            refuse(
                "lmkd kills at minfree levels (ro.lmk.use_minfree_levels), and \
                 sys.lmk.minfree_levels states none"
                    .into(),
            )
        })?;
        let minfree = foreground_minfree(&levels).ok_or_else(|| {
            refuse(format!(
                "sys.lmk.minfree_levels is `{levels}`, not pages:adj pairs"
            ))
        })?;
        let zoneinfo = read("/proc/zoneinfo")?;
        let reserve = total_reserve(&zoneinfo).ok_or_else(|| {
            refuse(
                "/proc/zoneinfo does not state every zone's high watermark, protection and \
                 managed pages"
                    .into(),
            )
        })?;
        minfree.saturating_sub(reserve)
    } else {
        // lmkd's pressure line is the zones' high watermarks, which
        // `MemAvailable` already leaves out.
        0
    };
    let defended = defended_pages
        .checked_mul(page)
        .ok_or_else(|| refuse(format!("{defended_pages} pages of {page} bytes overflow")))?;
    Ok(Room::of(available, defended))
}

/// A system property's value, `None` where it is unset.
#[cfg(target_os = "android")]
fn property(name: &str) -> Option<String> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut value = [0u8; 92]; // PROP_VALUE_MAX
    // SAFETY: `name` is NUL-terminated and `value` holds PROP_VALUE_MAX
    // bytes, the most the call writes.
    let len = unsafe { libc::__system_property_get(name.as_ptr(), value.as_mut_ptr().cast()) };
    let len = usize::try_from(len).ok().filter(|&len| len > 0)?;
    Some(String::from_utf8_lossy(&value[..len]).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_available_reads_kib() {
        let meminfo = "MemTotal:       11350976 kB\nMemFree:          569636 kB\n\
                       MemAvailable:    3692416 kB\nCached:          2551384 kB\n";
        assert_eq!(mem_available(meminfo), Some(3_692_416 * 1024));
        assert_eq!(mem_available("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn total_reserve_sums_high_and_protection_capped_at_managed() {
        // An SM8850's, abridged: only Normal manages pages.
        let zoneinfo = "Node 0, zone    DMA32\n  pages free     0\n        min      0\n\
                        low      0\n        high     0\n        managed  0\n\
                        protection: (0, 0, 0)\nNode 0, zone   Normal\n\
                        min      2896\n        low      17023\n        high     31150\n\
                        managed  2716912\n        protection: (0, 0, 0)\n\
                        Node 0, zone  Movable\n        min      32\n        low      32\n\
                        high     32\n        managed  0\n        protection: (0, 0, 0)\n";
        assert_eq!(total_reserve(zoneinfo), Some(31_150));
        let reserved = "Node 0, zone DMA32\n high 100\n managed 5000\n protection: (0, 700, 900)\n\
                        Node 0, zone Normal\n high 200\n managed 9000\n protection: (0, 0, 0)\n";
        assert_eq!(total_reserve(reserved), Some(1_000 + 200));
        assert_eq!(total_reserve("Node 0, zone Normal\n high 5\n"), None);
        assert_eq!(total_reserve(""), None);
    }

    #[test]
    fn foreground_minfree_is_the_lowest_adj_level() {
        let levels = "18432:0,23040:100,27648:200,32256:250,55296:900,80640:950";
        assert_eq!(foreground_minfree(levels), Some(18_432));
        assert_eq!(foreground_minfree("80640:950, 18432:0"), Some(18_432));
        assert_eq!(foreground_minfree("18432"), None);
    }

    #[test]
    fn room_leaves_the_reserve_and_the_floor() {
        let room = Room::of(3_692_416 * 1024, 0);
        assert_eq!(room.room, 3_692_416 * 1024 - SAFETY_FLOOR_BYTES);
        let room = Room::of(3_692_416 * 1024, 4096 * 4096);
        assert_eq!(
            room.room,
            3_692_416 * 1024 - 4096 * 4096 - SAFETY_FLOOR_BYTES
        );
        assert_eq!(Room::of(1 << 20, 1 << 30).room, 0);
    }
}

//! What a device that shares the host's memory has for a load: what the
//! host's memory has available, less what Android's lmkd keeps free past
//! that, capped at what the process's cgroups leave it, less the floor.
//!
//! A phone's GPU, an integrated GPU, or one whose largest device-local heap
//! the host maps takes its memory from the host's RAM: a reservation is part
//! of the process like any other. The heap such a driver reports is RAM, so
//! it says nothing of what is free, and without `VK_EXT_memory_budget` (no
//! Android driver reports it) nothing else does. What this process may take
//! is what the kernel counts available (`MemAvailable` in `/proc/meminfo`),
//! less any reserve lmkd (Android's low-memory killer daemon) defends beyond
//! it. Taking that reserve would make lmkd kill processes to win it back:
//! the user's apps, or this process. A cgroup capping the process below
//! the host (a systemd slice's `MemoryMax`, a container) kills it at that
//! cap however much the host has free, so what this process may take is
//! also no more than what each capped cgroup ancestor leaves
//! ([`engine::host_memory`]).
//!
//! Each reserve counts once. The kernel's `MemAvailable` already leaves out
//! its `totalreserve`: every zone's high watermark plus its largest lowmem
//! reserve, capped at what the zone manages. So:
//! - Off Android there is no lmkd, and nothing past `MemAvailable` is
//!   defended.
//! - On Android with `ro.lmk.use_minfree_levels` unset (the default since
//!   Android 10, and an SM8850's), lmkd kills on memory pressure (PSI) once
//!   free memory falls under the zones' high watermarks: the line
//!   `MemAvailable` has already left out. Nothing more is subtracted.
//! - On Android with it set, lmkd kills at the `minfree` levels
//!   `sys.lmk.minfree_levels` names, as `pages:adj` pairs. The line it
//!   defends is the level guarding the lowest adj stated, the foreground
//!   app's: below it, lmkd kills what the user sees. Only what that level
//!   exceeds `totalreserve` by is subtracted, read from `/proc/zoneinfo`.
//!
//! The floor is the driver's safety floor every backend keeps. A reading
//! that cannot be taken is refused by name.
#![allow(unsafe_code)]

#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::error::{Fault, Result};

/// The user's stated safety margin ([`engine::SAFETY_FLOOR_BYTES`]).
pub use engine::SAFETY_FLOOR_BYTES;
/// What the host has available to the process ([`engine::host_memory`]).
pub use engine::host_memory::Available;

/// What the host had for a load when read, and how that was reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Room {
    /// `MemAvailable` from `/proc/meminfo`.
    pub available: u64,
    /// The least any capped cgroup ancestor of the process leaves, `None`
    /// where none is capped.
    pub cgroup: Option<u64>,
    /// The bytes lmkd keeps free by killing beyond what `available`
    /// already leaves out.
    pub defended: u64,
    /// What a load may take: `available` less `defended`, capped at
    /// `cgroup`, less the floor.
    pub room: u64,
}

impl Room {
    /// What the host can still give, signed: negative by what it is short
    /// once what it has available (past the defended line, under the
    /// cgroups' caps) falls under the floor, which a process holding memory
    /// gives back to keep it there.
    #[must_use]
    pub fn net(&self) -> i64 {
        let host = i128::from(self.available) - i128::from(self.defended);
        let open = self
            .cgroup
            .map_or(host, |cgroup| host.min(i128::from(cgroup)));
        let net = open - i128::from(SAFETY_FLOOR_BYTES);
        i64::try_from(net).unwrap_or(if net < 0 { i64::MIN } else { i64::MAX })
    }
}

/// Whether, and how, a low-memory killer defends memory past
/// `MemAvailable`, with the texts its line is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Killer<'a> {
    /// No lmkd: a host off Android.
    None,
    /// lmkd in its pressure (PSI) mode.
    Pressure,
    /// lmkd at minfree levels: `sys.lmk.minfree_levels` and
    /// `/proc/zoneinfo`'s text.
    Minfree { levels: &'a str, zoneinfo: &'a str },
}

/// The room what the host has `available` leaves under `killer`, in pages
/// of `page` bytes: the one rule every host that shares its memory sizes
/// by.
///
/// # Errors
///
/// Names the reading the rule needed and could not take.
pub fn room_of(
    available: Available,
    killer: Killer<'_>,
    page: u64,
) -> std::result::Result<Room, String> {
    Ok(room_within(available, defended_by(killer, page)?))
}

/// The bytes `killer` defends past `MemAvailable`, in pages of `page`
/// bytes. Its levels and the zones' reserve do not change while a process
/// serves, so a reader takes this once.
///
/// # Errors
///
/// Names the reading the rule needed and could not take.
pub fn defended_by(killer: Killer<'_>, page: u64) -> std::result::Result<u64, String> {
    Ok(match killer {
        Killer::None | Killer::Pressure => 0,
        Killer::Minfree { levels, zoneinfo } => {
            let minfree = foreground_minfree(levels).ok_or_else(|| {
                format!("sys.lmk.minfree_levels is `{levels}`, not pages:adj pairs")
            })?;
            let reserve = total_reserve(zoneinfo).ok_or_else(|| {
                "/proc/zoneinfo does not state every zone's high watermark, protection and \
                 managed pages"
                    .to_string()
            })?;
            let pages = minfree.saturating_sub(reserve);
            pages
                .checked_mul(page)
                .ok_or_else(|| format!("{pages} pages of {page} bytes overflow"))?
        }
    })
}

/// The room what the host has `available` leaves past `defended` bytes:
/// lmkd defends its line against the host's `MemAvailable`, and the
/// cgroups' caps bind past that.
#[must_use]
pub fn room_within(available: Available, defended: u64) -> Room {
    let host = available.host.saturating_sub(defended);
    let open = available.cgroup.map_or(host, |cgroup| cgroup.min(host));
    Room {
        available: available.host,
        cgroup: available.cgroup,
        defended,
        room: open.saturating_sub(SAFETY_FLOOR_BYTES),
    }
}

/// The kernel's `totalreserve`, in pages, from `/proc/zoneinfo`'s text:
/// for every zone, its high watermark plus its largest lowmem reserve
/// (`protection`), capped at the pages it manages, as
/// `calculate_totalreserve_pages` sums it. `MemAvailable` leaves it out.
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

/// What the host has for a load now, read as the module doc says.
///
/// # Errors
///
/// Refuses by name when a reading the rule needs cannot be taken.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn room() -> Result<Room> {
    reader()?.read()
}

/// A reader of the host's room: what lmkd defends past `MemAvailable` is
/// read once, when it is made; each reading after takes only
/// `/proc/meminfo` and the process's cgroups.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[derive(Clone, Copy, Debug)]
pub struct Reader {
    defended: u64,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn refuse(why: String) -> Fault {
    Fault::Device {
        call: "reading the host's memory for the load's capacity",
        why,
    }
}

#[cfg(target_os = "android")]
fn read(path: &str) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| refuse(format!("{path}: {e}")))
}

/// Makes a [`Reader`], reading the low-memory killer's line once.
///
/// # Errors
///
/// Refuses by name when a reading the rule needs cannot be taken.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn reader() -> Result<Reader> {
    // SAFETY: `sysconf` reads a constant of the running system.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page = u64::try_from(page)
        .ok()
        .filter(|&page| page > 0)
        .ok_or_else(|| refuse(format!("the page size reads {page}")))?;
    #[cfg(target_os = "android")]
    let (levels, zoneinfo);
    #[cfg(target_os = "android")]
    let killer = if property("ro.lmk.use_minfree_levels").as_deref() == Some("true") {
        levels = property("sys.lmk.minfree_levels").ok_or_else(|| {
            refuse(
                "lmkd kills at minfree levels (ro.lmk.use_minfree_levels), and \
                 sys.lmk.minfree_levels states none"
                    .into(),
            )
        })?;
        zoneinfo = read("/proc/zoneinfo")?;
        Killer::Minfree {
            levels: &levels,
            zoneinfo: &zoneinfo,
        }
    } else {
        Killer::Pressure
    };
    #[cfg(not(target_os = "android"))]
    let killer = Killer::None;
    Ok(Reader {
        defended: defended_by(killer, page).map_err(refuse)?,
    })
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl Reader {
    /// The host's room now: `/proc/meminfo` and the process's cgroups,
    /// past the line read once.
    ///
    /// # Errors
    ///
    /// Refuses by name when `/proc/meminfo` cannot be read or states no
    /// `MemAvailable`, or a capped cgroup's usage cannot be read.
    pub fn read(&self) -> Result<Room> {
        let available = engine::host_memory::available().map_err(refuse)?;
        Ok(room_within(available, self.defended))
    }
}

/// What the host can still give the process now ([`Room::net`]), for a
/// device that shares the host's memory: lmkd's line is read once, here,
/// and each call reads only `/proc/meminfo` and the process's cgroups.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[must_use]
pub fn room_reader() -> engine::HostRoom {
    match reader() {
        Ok(reader) => std::sync::Arc::new(move || {
            reader
                .read()
                .map(|room| room.net())
                .map_err(|fault| fault.to_string())
        }),
        Err(fault) => {
            let why = fault.to_string();
            std::sync::Arc::new(move || Err(why.clone()))
        }
    }
}

/// A host with no reader of its available memory refuses each reading.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
#[must_use]
pub fn room_reader() -> engine::HostRoom {
    std::sync::Arc::new(|| Err("this host has no reader of its available memory".to_string()))
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

    const PAGE: u64 = 4096;
    const MIB: u64 = 1 << 20;

    /// An SM8850's, abridged.
    const AVAILABLE: u64 = 3_692_416 * 1024;
    const MEMINFO: Available = Available {
        host: AVAILABLE,
        cgroup: None,
    };

    /// An SM8850's, abridged: only Normal manages pages; its reserve is
    /// 31150 pages.
    const ZONEINFO: &str = "Node 0, zone    DMA32\n  pages free     0\n        min      0\n\
                            low      0\n        high     0\n        managed  0\n\
                            protection: (0, 0, 0)\nNode 0, zone   Normal\n\
                            min      2896\n        low      17023\n        high     31150\n\
                            managed  2716912\n        protection: (0, 0, 0)\n\
                            Node 0, zone  Movable\n        min      32\n        low      32\n\
                            high     32\n        managed  0\n        protection: (0, 0, 0)\n";

    #[test]
    fn off_android_the_room_is_available_less_the_floor() {
        let room = room_of(MEMINFO, Killer::None, PAGE).unwrap();
        assert_eq!(room.defended, 0);
        assert_eq!(room.room, AVAILABLE - SAFETY_FLOOR_BYTES);
    }

    #[test]
    fn lmkd_pressure_mode_subtracts_nothing_available_left_out() {
        let room = room_of(MEMINFO, Killer::Pressure, PAGE).unwrap();
        assert_eq!(room.defended, 0);
        assert_eq!(room.room, AVAILABLE - SAFETY_FLOOR_BYTES);
    }

    #[test]
    fn lmkd_minfree_mode_subtracts_only_the_level_past_totalreserve() {
        // The foreground level (adj 0) is 81150 pages, 50000 past the
        // zones' reserve.
        let levels = "81150:0,90000:100,120000:900";
        let room = room_of(
            MEMINFO,
            Killer::Minfree {
                levels,
                zoneinfo: ZONEINFO,
            },
            PAGE,
        )
        .unwrap();
        assert_eq!(room.defended, 50_000 * PAGE);
        assert_eq!(room.room, AVAILABLE - 50_000 * PAGE - SAFETY_FLOOR_BYTES);
        // A level under the reserve defends nothing past it.
        let room = room_of(
            MEMINFO,
            Killer::Minfree {
                levels: "18432:0",
                zoneinfo: ZONEINFO,
            },
            PAGE,
        )
        .unwrap();
        assert_eq!(room.defended, 0);
    }

    #[test]
    fn a_reading_the_rule_needs_is_refused_by_name() {
        let err = room_of(
            MEMINFO,
            Killer::Minfree {
                levels: "18432",
                zoneinfo: ZONEINFO,
            },
            PAGE,
        )
        .unwrap_err();
        assert!(err.contains("pages:adj"), "{err}");
        let err = room_of(
            MEMINFO,
            Killer::Minfree {
                levels: "18432:0",
                zoneinfo: "Node 0, zone Normal\n high 5\n",
            },
            PAGE,
        )
        .unwrap_err();
        assert!(err.contains("zoneinfo"), "{err}");
    }

    #[test]
    fn the_room_saturates_and_its_net_says_the_shortfall() {
        let tight = Available {
            host: 1024 * 1024,
            cgroup: None,
        };
        let room = room_of(tight, Killer::None, PAGE).unwrap();
        assert!(room.available < 64 * MIB);
        assert_eq!(room.room, 0);
        assert_eq!(room.net(), -(63 * MIB as i64));
        let room = room_of(MEMINFO, Killer::Pressure, PAGE).unwrap();
        assert_eq!(room.net(), (AVAILABLE - SAFETY_FLOOR_BYTES) as i64);
    }

    #[test]
    fn a_cgroup_cap_binds_past_what_lmkd_defends() {
        let killer = Killer::Minfree {
            levels: "81150:0",
            zoneinfo: ZONEINFO,
        };
        let open = AVAILABLE - 50_000 * PAGE;
        // A cap tighter than the host past lmkd's line binds alone: lmkd's
        // line is the host's, not the cgroup's.
        let capped = Available {
            host: AVAILABLE,
            cgroup: Some(open / 2),
        };
        let room = room_of(capped, killer, PAGE).unwrap();
        assert_eq!(room.cgroup, Some(open / 2));
        assert_eq!(room.room, open / 2 - SAFETY_FLOOR_BYTES);
        assert_eq!(room.net(), (open / 2 - SAFETY_FLOOR_BYTES) as i64);
        // A looser cap leaves the host's room.
        let loose = Available {
            host: AVAILABLE,
            cgroup: Some(AVAILABLE),
        };
        let room = room_of(loose, killer, PAGE).unwrap();
        assert_eq!(room.room, open - SAFETY_FLOOR_BYTES);
        // A cap its group has reached leaves the floor short.
        let full = Available {
            host: AVAILABLE,
            cgroup: Some(0),
        };
        let room = room_of(full, Killer::None, PAGE).unwrap();
        assert_eq!(room.room, 0);
        assert_eq!(room.net(), -(SAFETY_FLOOR_BYTES as i64));
    }

    #[test]
    fn total_reserve_sums_high_and_protection_capped_at_managed() {
        assert_eq!(total_reserve(ZONEINFO), Some(31_150));
        let reserved = "Node 0, zone DMA32\n high 100\n managed 5000\n protection: (0, 700, 900)\n\
                        Node 0, zone Normal\n high 200\n managed 9000\n protection: (0, 0, 0)\n\
                        Node 0, zone Tiny\n high 300\n managed 50\n protection: (0, 0, 0)\n";
        assert_eq!(total_reserve(reserved), Some(1_000 + 200 + 50));
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
}

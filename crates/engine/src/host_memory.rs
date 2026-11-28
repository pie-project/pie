//! What the host's memory has available to this process: the least of what
//! the kernel counts available (`MemAvailable` in `/proc/meminfo`) and what
//! each cgroup v2 ancestor of the process with a memory limit leaves under
//! it.
//!
//! `MemAvailable` is the host's. A process in a cgroup whose `memory.max`
//! sits below the host's RAM (a systemd slice, a container) is OOM-killed
//! at that cap however much the host has free. Every ancestor's cap binds,
//! so each from the process's own group (`0::<path>` in
//! `/proc/self/cgroup`) up to the root `/sys/fs/cgroup` is read: where it
//! states a numeric `memory.max`, it leaves `memory.max` less its
//! `memory.current` (less its inactive file pages, `memory.stat`, which the
//! kernel reclaims before a charge fails). `max` there means no limit, and a group with no
//! `memory.max` (the root, or one without the memory controller) caps
//! nothing. A process with no cgroup v2 path is capped by none.
//!
//! A reading that cannot be taken is refused by name: no `MemAvailable`,
//! or a group whose `memory.max` or `memory.current` cannot be read as a
//! number. `/proc/meminfo` and cgroups are Linux's (and Android's); on
//! another platform the reading is refused by name, and a load states its
//! host budget instead.

#[cfg(any(target_os = "linux", target_os = "android"))]
use std::path::{Path, PathBuf};

/// Where the cgroup v2 hierarchy is mounted.
#[cfg(any(target_os = "linux", target_os = "android"))]
const CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// What the host had available to the process when read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Available {
    /// `MemAvailable` from `/proc/meminfo`, in bytes.
    pub host: u64,
    /// The least any capped cgroup ancestor leaves (`memory.max` less
    /// `memory.current`), `None` where none is capped.
    pub cgroup: Option<u64>,
}

impl Available {
    /// What the process may take: the least of [`Self::host`] and
    /// [`Self::cgroup`].
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.cgroup
            .map_or(self.host, |cgroup| cgroup.min(self.host))
    }
}

/// What the host has available to the process now, read as the module doc
/// says.
///
/// # Errors
///
/// Names the reading that could not be taken.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn available() -> Result<Available, String> {
    let meminfo =
        std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
    // No `/proc/self/cgroup` (a host without cgroups) caps nothing.
    let own = std::fs::read_to_string("/proc/self/cgroup").ok();
    available_of(&meminfo, own.as_deref(), Path::new(CGROUP_ROOT), |path| {
        std::fs::read_to_string(path)
    })
}

/// What the host has available to the process: not read on this platform.
///
/// # Errors
///
/// Always: the reading is `/proc/meminfo`'s and the cgroups', Linux's.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub fn available() -> Result<Available, String> {
    Err("host memory isn't read on this platform; state budget_host".to_string())
}

/// What `/proc/meminfo`'s text and the cgroups under `root` leave the
/// process whose `/proc/self/cgroup` reads `own`, each group's files read
/// through `read` (an absent file is a group that states nothing).
///
/// # Errors
///
/// Names the reading that could not be taken.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn available_of(
    meminfo: &str,
    own: Option<&str>,
    root: &Path,
    read: impl Fn(&Path) -> std::io::Result<String>,
) -> Result<Available, String> {
    let host =
        mem_available(meminfo).ok_or_else(|| "/proc/meminfo states no MemAvailable".to_string())?;
    Ok(Available {
        host,
        cgroup: cgroup_room(own, root, read)?,
    })
}

/// `MemAvailable` in bytes, from `/proc/meminfo`'s text.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn mem_available(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        let kib: u64 = rest.trim().strip_suffix("kB")?.trim().parse().ok()?;
        kib.checked_mul(1024)
    })
}

/// The least any capped cgroup v2 ancestor leaves, walking from the group
/// `own` names up to `root`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn cgroup_room(
    own: Option<&str>,
    root: &Path,
    read: impl Fn(&Path) -> std::io::Result<String>,
) -> Result<Option<u64>, String> {
    let Some(path) = own.and_then(|own| own.lines().find_map(|l| l.strip_prefix("0::"))) else {
        return Ok(None);
    };
    let mut dir: PathBuf = root.join(path.trim().trim_start_matches('/'));
    let mut least: Option<u64> = None;
    loop {
        if let Some(room) = group_room(&dir, &read)? {
            least = Some(least.map_or(room, |least| least.min(room)));
        }
        if dir.as_path() == root || !dir.starts_with(root) || !dir.pop() {
            break;
        }
    }
    Ok(least)
}

/// What the group at `dir` leaves: `None` where it states no limit.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn group_room(
    dir: &Path,
    read: &impl Fn(&Path) -> std::io::Result<String>,
) -> Result<Option<u64>, String> {
    let file = |name: &str| {
        let path = dir.join(name);
        match read(&path) {
            Ok(text) => Ok(Some(text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    };
    let number = |name: &str, text: &str| {
        text.parse::<u64>().map_err(|_| {
            format!(
                "{} reads `{text}`, not a number of bytes",
                dir.join(name).display()
            )
        })
    };
    // The group's limit: `memory.max`, where its charges fail, or below it
    // `memory.high`, where the kernel throttles and reclaims the group hard;
    // either reads `max` where unset.
    let mut max: Option<u64> = None;
    for name in ["memory.max", "memory.high"] {
        if let Some(text) = file(name)?
            && text.trim() != "max"
        {
            let limit = number(name, text.trim())?;
            max = Some(max.map_or(limit, |max| max.min(limit)));
        }
    }
    let Some(max) = max else {
        return Ok(None);
    };
    let current = file("memory.current")?.ok_or_else(|| {
        format!(
            "{} states a memory.max or memory.high and no memory.current",
            dir.display()
        )
    })?;
    let current = number("memory.current", current.trim())?;
    // `memory.current` counts the group's page cache; its inactive file
    // pages are what the kernel reclaims first, before any charge fails,
    // so they are room (the working set, as container runtimes count it).
    // A group stating no `memory.stat` reclaims nothing counted here.
    let reclaimable = match file("memory.stat")? {
        Some(stat) => stat
            .lines()
            .find_map(|line| line.strip_prefix("inactive_file "))
            .map(|n| number("memory.stat", n.trim()))
            .transpose()?
            .unwrap_or(0),
        None => 0,
    };
    Ok(Some(
        max.saturating_sub(current.saturating_sub(reclaimable)),
    ))
}

#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const GIB: u64 = 1 << 30;
    /// 100 GiB available.
    const MEMINFO: &str = "MemTotal:       172000000 kB\nMemAvailable:   104857600 kB\n";
    const HOST: u64 = 100 * GIB;
    const OWN: &str = "0::/user.slice/user-0.slice/session-1.scope\n";

    /// Reads fake files under `/cg`, absent ones `NotFound`.
    #[allow(clippy::type_complexity)]
    fn files(texts: &[(&str, String)]) -> Box<dyn Fn(&Path) -> std::io::Result<String>> {
        let texts: HashMap<PathBuf, String> = texts
            .iter()
            .map(|(path, text)| (PathBuf::from(path), text.clone()))
            .collect();
        Box::new(move |path| {
            texts
                .get(path)
                .cloned()
                .ok_or_else(|| std::io::ErrorKind::NotFound.into())
        })
    }

    fn read(own: Option<&str>, texts: &[(&str, String)]) -> Result<Available, String> {
        available_of(MEMINFO, own, Path::new("/cg"), files(texts))
    }

    #[test]
    fn no_cgroup_leaves_mem_available() {
        let none = read(None, &[]).unwrap();
        assert_eq!(
            none,
            Available {
                host: HOST,
                cgroup: None
            }
        );
        assert_eq!(none.bytes(), HOST);
        // A v1-only `/proc/self/cgroup` names no v2 group.
        let v1 = read(Some("4:memory:/docker/abc\n"), &[]).unwrap();
        assert_eq!(v1.bytes(), HOST);
        // A v2 group with no memory controller caps nothing.
        assert_eq!(read(Some(OWN), &[]).unwrap().cgroup, None);
    }

    #[test]
    fn a_cgroup_max_below_available_caps_it() {
        let texts = [
            ("/cg/user.slice/memory.max", format!("{}\n", 150 * GIB)),
            ("/cg/user.slice/memory.current", format!("{}\n", 90 * GIB)),
        ];
        let got = read(Some(OWN), &texts).unwrap();
        assert_eq!(got.cgroup, Some(60 * GIB));
        assert_eq!(got.bytes(), 60 * GIB);
        // A group over its cap leaves nothing.
        let over = [
            ("/cg/user.slice/memory.max", format!("{}", 10 * GIB)),
            ("/cg/user.slice/memory.current", format!("{}", 11 * GIB)),
        ];
        assert_eq!(read(Some(OWN), &over).unwrap().bytes(), 0);
    }

    #[test]
    fn a_cgroups_memory_high_below_its_max_caps_it() {
        // where the kernel throttles the group, below where its charges fail
        let texts = [
            ("/cg/user.slice/memory.max", format!("{}\n", 150 * GIB)),
            ("/cg/user.slice/memory.high", format!("{}\n", 120 * GIB)),
            ("/cg/user.slice/memory.current", format!("{}\n", 90 * GIB)),
        ];
        assert_eq!(read(Some(OWN), &texts).unwrap().cgroup, Some(30 * GIB));
        // a high alone, its max unset
        let high = [
            ("/cg/user.slice/memory.max", "max\n".to_string()),
            ("/cg/user.slice/memory.high", format!("{}\n", 120 * GIB)),
            ("/cg/user.slice/memory.current", format!("{}\n", 90 * GIB)),
        ];
        assert_eq!(read(Some(OWN), &high).unwrap().cgroup, Some(30 * GIB));
    }

    #[test]
    fn a_cgroups_inactive_page_cache_is_room() {
        // v6e: 150 GiB capped, 115 GiB charged of which 96 GiB is file cache,
        // 60 GiB of it inactive: 95 GiB of room, not 35.
        let texts = [
            ("/cg/user.slice/memory.max", format!("{}\n", 150 * GIB)),
            ("/cg/user.slice/memory.current", format!("{}\n", 115 * GIB)),
            (
                "/cg/user.slice/memory.stat",
                format!(
                    "anon {}\nfile {}\ninactive_file {}\nactive_file {}\n",
                    19 * GIB,
                    96 * GIB,
                    60 * GIB,
                    36 * GIB
                ),
            ),
        ];
        assert_eq!(read(Some(OWN), &texts).unwrap().cgroup, Some(95 * GIB));
    }

    #[test]
    fn a_cgroup_leaving_more_than_available_leaves_available() {
        let texts = [
            ("/cg/user.slice/memory.max", format!("{}", 400 * GIB)),
            ("/cg/user.slice/memory.current", format!("{}", 10 * GIB)),
        ];
        assert_eq!(read(Some(OWN), &texts).unwrap().bytes(), HOST);
    }

    #[test]
    fn the_tightest_ancestor_binds_inner_or_outer() {
        let scope = "/cg/user.slice/user-0.slice/session-1.scope";
        let slice = "/cg/user.slice";
        let nested = |inner: u64, outer: u64| {
            [
                (format!("{scope}/memory.max"), format!("{}", inner + GIB)),
                (format!("{scope}/memory.current"), format!("{GIB}")),
                (
                    format!("{slice}/memory.max"),
                    format!("{}", outer + 5 * GIB),
                ),
                (format!("{slice}/memory.current"), format!("{}", 5 * GIB)),
            ]
        };
        fn texts(n: &[(String, String); 4]) -> Vec<(&str, String)> {
            n.iter().map(|(p, t)| (p.as_str(), t.clone())).collect()
        }
        // The inner (own) group is tighter.
        let inner = nested(20 * GIB, 60 * GIB);
        assert_eq!(read(Some(OWN), &texts(&inner)).unwrap().bytes(), 20 * GIB);
        // The outer (slice) group is tighter.
        let outer = nested(60 * GIB, 20 * GIB);
        assert_eq!(read(Some(OWN), &texts(&outer)).unwrap().bytes(), 20 * GIB);
    }

    #[test]
    fn max_means_no_limit() {
        let texts = [
            (
                "/cg/user.slice/user-0.slice/session-1.scope/memory.max",
                "max\n".to_string(),
            ),
            (
                "/cg/user.slice/user-0.slice/session-1.scope/memory.current",
                format!("{GIB}"),
            ),
            ("/cg/user.slice/memory.max", "max\n".to_string()),
            ("/cg/user.slice/memory.current", format!("{}", 2 * GIB)),
        ];
        let got = read(Some(OWN), &texts).unwrap();
        assert_eq!(got.cgroup, None);
        assert_eq!(got.bytes(), HOST);
        // An unlimited inner group under a capped outer one: the outer binds.
        let mut capped = texts.to_vec();
        capped[2].1 = format!("{}", 12 * GIB);
        assert_eq!(read(Some(OWN), &capped).unwrap().bytes(), 10 * GIB);
    }

    #[test]
    fn the_root_group_is_read_and_the_walk_stops_there() {
        let texts = [
            ("/cg/memory.max", format!("{}", 8 * GIB)),
            ("/cg/memory.current", format!("{}", 3 * GIB)),
            ("/memory.max", format!("{GIB}")),
            ("/memory.current", "0".to_string()),
        ];
        assert_eq!(read(Some("0::/\n"), &texts).unwrap().bytes(), 5 * GIB);
    }

    #[test]
    fn a_reading_that_cannot_be_taken_is_refused_by_name() {
        let err = available_of("MemTotal: 1 kB\n", None, Path::new("/cg"), files(&[])).unwrap_err();
        assert!(err.contains("MemAvailable"), "{err}");
        let err = read(
            Some(OWN),
            &[("/cg/user.slice/memory.max", "lots\n".to_string())],
        )
        .unwrap_err();
        assert!(err.contains("/cg/user.slice/memory.max"), "{err}");
        let err = read(
            Some(OWN),
            &[("/cg/user.slice/memory.max", format!("{GIB}"))],
        )
        .unwrap_err();
        assert!(err.contains("memory.current"), "{err}");
        let refused = |path: &Path| -> std::io::Result<String> {
            if path.ends_with("memory.max") {
                Err(std::io::ErrorKind::PermissionDenied.into())
            } else {
                Err(std::io::ErrorKind::NotFound.into())
            }
        };
        let err = available_of(MEMINFO, Some(OWN), Path::new("/cg"), refused).unwrap_err();
        assert!(err.contains("memory.max"), "{err}");
    }
}

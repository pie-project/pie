//! What pie writes to disk, in one place.
//!
//! Every entry below is somewhere pie creates files under `$PIE_HOME`. The
//! list exists so that asking "what is here?" and "reclaim it" read from the
//! same source: a cache that only one of them knows about is a cache that
//! either cannot be reclaimed or is reclaimed by surprise.
//!
//! This is deliberately a description of the tree, not of the code that writes
//! it. The one coupling that matters is the driver cache root: this module
//! names `cache/`, and `embedded_driver::set_cache_dir` is what tells the
//! drivers to write there. Its *contents* are enumerated from disk rather than
//! listed here, because the subdirectory names (`ptir-cuda`, the GEMM tuning
//! files, `cuda_memory_profiles.json`) are chosen on the C++ side and a list
//! here would be a second copy free to drift from them.

use std::path::PathBuf;

/// The root every driver-side disk cache derives from.
///
/// Defined here rather than at the call site so the registry below and
/// `embedded_driver::set_cache_dir` -- the thing that actually tells the
/// drivers where to write -- cannot drift. A `pie cache clear` that looked in
/// a directory nothing writes to would report success and reclaim nothing.
pub fn driver_cache_dir() -> PathBuf {
    crate::paths::pie_home().join("cache")
}

/// Whether an entry may be deleted to reclaim space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reclaim {
    /// Derived and re-derivable. Deleting costs rebuild time, never data.
    Safe,
    /// Reclaimable, but not without being asked: deleting it destroys
    /// something a person may still need.
    OnRequest,
    /// Authored by a person. Never a reclaim target, listed so that a reader
    /// can see the whole tree and know what is off limits.
    Never,
}

/// One place pie writes.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Stable selector, e.g. for a `--what` flag.
    pub name: &'static str,
    pub path: PathBuf,
    /// What it holds, and what deleting it costs.
    pub what: &'static str,
    pub reclaim: Reclaim,
}

/// Everything pie writes under `$PIE_HOME`, whether or not it exists yet.
///
/// Order is roughly "cheapest to lose" first, so a listing reads top-down as
/// increasing regret.
pub fn entries(hf_cache: Option<PathBuf>) -> Vec<Entry> {
    let home = crate::paths::pie_home();
    let mut entries = vec![
        Entry {
            name: "launch",
            path: crate::embedded_driver::launch_state_root(),
            what: "Per-launch driver startup TOMLs. Dead as soon as the drivers \
                   they configured are down; swept at boot for pids that are gone.",
            reclaim: Reclaim::Safe,
        },
        Entry {
            name: "driver",
            path: driver_cache_dir(),
            what: "Driver-side disk caches: compiled PTIR modules, GEMM \
                   autotuning results, planner profiles. All keyed and \
                   self-invalidating; deleting costs one cold rebuild.",
            reclaim: Reclaim::Safe,
        },
        Entry {
            name: "programs",
            path: home.join("programs"),
            what: "Inferlet programs fetched from the registry. Re-fetched on \
                   demand.",
            reclaim: Reclaim::Safe,
        },
        Entry {
            name: "py-runtime",
            path: home.join("py-runtime"),
            what: "The embedded Python-WASM runtime. Re-provisioned by the \
                   next `pie serve`.",
            reclaim: Reclaim::Safe,
        },
        Entry {
            name: "models",
            path: home.join("models"),
            // The artifact store, not a cache. It used to be the driver's
            // materialized-weight cache, which was re-derived on the next
            // load; `.zt` artifacts are not. Losing one costs a download and a
            // conversion, and the file is portable in a way device weights
            // never are -- a TP layout and an ABI version are baked into
            // those, and into nothing here.
            what: "Converted `.zt` artifacts -- the models pie serves. Losing \
                   one costs a re-download and a re-convert, not a reload.",
            reclaim: Reclaim::OnRequest,
        },
        Entry {
            name: "logs",
            path: home.join("logs"),
            what: "Engine logs. Reclaimable, but deleting them mid-investigation \
                   is its own kind of loss.",
            reclaim: Reclaim::OnRequest,
        },
        Entry {
            name: "weights",
            path: driver_cache_dir().join("weights"),
            what: "Materialized device weights, keyed by checkpoint + config + \
                   quant scheme + TP layout + ABI version. One cold load to \
                   rebuild; valid only for this build.",
            reclaim: Reclaim::Safe,
        },
        Entry {
            name: "config",
            path: home.join("config.toml"),
            what: "The config file. Authored, not derived.",
            reclaim: Reclaim::Never,
        },
    ];
    // Outside `$PIE_HOME`, and the only path this layer cannot resolve -- the
    // HuggingFace cache location is `bin/pie`'s to know. Passed in rather than
    // re-derived here, so there is still one description and one reclaim
    // policy per thing pie writes.
    //
    // It is in the list at all because the artifact store demoted it: a
    // snapshot is now the source a `.zt` was converted FROM, kept so a
    // re-convert costs no download.
    if let Some(hf) = hf_cache {
        entries.push(Entry {
            name: "snapshots",
            path: hf,
            what: "HuggingFace downloads, kept so a re-convert needs no \
                   network. Not needed to serve an artifact that already \
                   exists.",
            reclaim: Reclaim::OnRequest,
        });
    }
    entries
}

/// Bytes under `path`, following the tree. Returns 0 for a path that does not
/// exist, and skips anything it cannot read: a size report must never be the
/// reason a command fails.
pub fn disk_usage(path: &std::path::Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if !meta.is_dir() {
        return meta.len();
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| disk_usage(&e.path()))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_one_entry_outside_pie_home_is_the_one_that_was_passed_in() {
        // The guard below exists so a typo cannot reclaim something that is
        // not ours. HuggingFace snapshots ARE outside `$PIE_HOME` and are
        // still pie's to offer -- so they arrive as a parameter rather than
        // being derived here, and this pins that as the only way in.
        let outside = PathBuf::from("/somewhere/else/huggingface");
        let with = entries(Some(outside.clone()));
        let without = entries(None);
        assert_eq!(with.len(), without.len() + 1);
        let extra: Vec<&Entry> = with
            .iter()
            .filter(|e| !e.path.starts_with(crate::paths::pie_home()))
            .collect();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].path, outside);
        // And never `Never`: an entry pie cannot reach should not be one it
        // refuses to reclaim on principle.
        assert_ne!(extra[0].reclaim, Reclaim::Never);
    }

    #[test]
    fn names_are_unique_and_paths_stay_under_pie_home() {
        // A `--what` selector that matches two entries would delete more than
        // it names; a path outside $PIE_HOME would let a typo here reclaim
        // something that is not ours.
        let home = crate::paths::pie_home();
        let mut seen = std::collections::HashSet::new();
        for entry in entries(None) {
            assert!(seen.insert(entry.name), "duplicate name {}", entry.name);
            assert!(
                entry.path.starts_with(&home),
                "{} escapes $PIE_HOME: {:?}",
                entry.name,
                entry.path
            );
        }
    }

    #[test]
    fn the_authored_files_are_never_reclaimable() {
        // The failure this guards is silent and total: a `pie cache clear`
        // that takes config.toml with it looks like it worked.
        for entry in entries(None) {
            if entry.name == "config" {
                assert_eq!(
                    entry.reclaim,
                    Reclaim::Never,
                    "{} must never be reclaimable",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn the_registry_names_the_paths_the_code_actually_writes() {
        // Both of these are defined once and referenced twice; this asserts the
        // reference, so that factoring one apart later fails here rather than
        // silently giving `pie cache` a directory nothing writes to.
        let by_name = |n: &str| {
            entries(None)
                .into_iter()
                .find(|e| e.name == n)
                .unwrap_or_else(|| panic!("no {n} entry"))
                .path
        };
        assert_eq!(by_name("driver"), driver_cache_dir());
        assert_eq!(
            by_name("launch"),
            crate::embedded_driver::launch_state_root()
        );
    }

    #[test]
    fn disk_usage_sums_a_tree_and_tolerates_absence() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(disk_usage(&dir.path().join("nothing-here")), 0);
        std::fs::create_dir_all(dir.path().join("a/b")).unwrap();
        std::fs::write(dir.path().join("a/one"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.path().join("a/b/two"), vec![0u8; 23]).unwrap();
        assert_eq!(disk_usage(dir.path()), 123);
    }
}

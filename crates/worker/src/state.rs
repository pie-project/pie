//! What pie writes to disk, in one place.
//!
//! Every entry below is somewhere pie creates files under `$PIE_HOME`. The
//! list exists so that asking "what is here?" and "reclaim it" read from the
//! same source: a cache that only one of them knows about is a cache that
//! either cannot be reclaimed or is reclaimed by surprise.
//!
//! This is deliberately a description of the tree, not of the code that writes
//! it. The one coupling that matters is the engine cache root: this module
//! names `cache/`, and the boot path hands it to every engine as a
//! `DeviceBoot` field. Its *contents* are enumerated from disk rather than
//! listed here, because the subdirectory names (`cubins`, `gemm-algos`, and
//! whatever the C++ side chooses) belong to the crates that write them, and a
//! list here would be a second copy free to drift from them.
//!
//! That enumeration is what made bringing the kernel caches home a change with
//! no line in this file: `kernels-cuda` used to resolve `~/.cache/pie` for
//! itself, which put its cubins and its measured cuBLASLt table outside
//! `$PIE_HOME` and so outside everything below. They take their subdirectory
//! off the same root now and are counted and reclaimed with the rest.
//!
//! **Entries do not nest.** Two entries whose paths contain one another would
//! double-count in a size report and delete each other in a reclaim, so the
//! ones carved out of `cache/` are subtracted from it via [`Entry::keep`] and
//! the `entries_are_disjoint` test pins that.

use std::path::PathBuf;

// `PLANNER_PROFILE_FILE` and its lock STOOD HERE — `cuda_memory_profiles.json`,
// the one thing under `cache/` a boot did not re-derive, written only by a
// calibration boot (`store/planner_profile_cache.cpp`, C++). Writer and reader
// both left with that engine, so a leftover file is one more derived-looking
// thing the `engine` entry sweeps, not a measurement to protect.

/// The root every engine-side disk cache derives from.
///
/// Defined here rather than at the call site so the registry below and the
/// boot path that actually tells the engines where to write -- a `DeviceBoot`
/// field now -- cannot drift. A `pie cache clear` that looked in a directory
/// nothing writes to would report success and reclaim nothing.
pub fn engine_cache_dir() -> PathBuf {
    crate::paths::pie_home().join("cache")
}

/// Where the engine keeps materialized device weights.
///
/// Defined once for the same reason `engine_cache_dir` is: this and
/// `::runtime::boot` both need it, and when they were two expressions they landed
/// on two directories -- one of them the artifact store, which is not a cache
/// at all.
pub fn weight_cache_dir() -> PathBuf {
    engine_cache_dir().join("weights")
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
    /// Children of `path` that belong to a *different* entry, and so are
    /// neither counted in this entry's size nor removed with it.
    ///
    /// This is what keeps the registry a flat list over a nested tree. `cache/`
    /// physically contains the weight cache, which has its own reclaim policy —
    /// so an `engine` entry that swallowed it would report its bytes twice and
    /// delete it while claiming to reclaim something else.
    pub keep: &'static [&'static str],
}

/// Everything pie writes under `$PIE_HOME`, whether or not it exists yet.
///
/// Order is roughly "cheapest to lose" first, so a listing reads top-down as
/// increasing regret.
pub fn entries(hf_cache: Option<PathBuf>) -> Vec<Entry> {
    let home = crate::paths::pie_home();
    let mut entries = vec![
        // The "launch" entry STOOD FIRST HERE -- `$PIE_HOME/standalone/<pid>`,
        // the per-launch bootstrap TOMLs with their pid sweep. The boot config
        // crosses as a typed struct now, so nothing writes that tree at all.
        Entry {
            name: "engine",
            path: engine_cache_dir(),
            // The GEMM half of this sentence was FALSE until the cache roots
            // came together: `kernels-cuda` resolved its own path from
            // `$XDG_CACHE_HOME`, so its cubins and its measured cuBLASLt table
            // sat in `~/.cache/pie` — outside `$PIE_HOME`, and outside
            // everything this module can see or reclaim. They are `cubins/`
            // and `gemm-algos/` under this root now, enumerated from disk with
            // the rest.
            what: "Engine-side disk caches: compiled ETA modules, kernel \
                   cubins, GEMM autotuning results. All keyed and \
                   self-invalidating; deleting costs one cold rebuild.",
            reclaim: Reclaim::Safe,
            keep: &["weights"],
        },
        Entry {
            name: "programs",
            path: home.join("programs"),
            what: "Inferlet programs fetched from the registry. Re-fetched on \
                   demand.",
            reclaim: Reclaim::Safe,
            keep: &[],
        },
        Entry {
            name: "py-runtime",
            path: home.join("py-runtime"),
            what: "The embedded Python-WASM runtime. Re-provisioned by the \
                   next `pie serve`.",
            reclaim: Reclaim::Safe,
            keep: &[],
        },
        Entry {
            name: "models",
            path: home.join("models"),
            // The artifact store, not a cache. It used to be the engine's
            // materialized-weight cache, which was re-derived on the next
            // load; `.zt` artifacts are not. Losing one costs a download and a
            // conversion, and the file is portable in a way device weights
            // never are -- a TP layout and an ABI version are baked into
            // those, and into nothing here.
            what: "Converted `.zt` artifacts -- the models pie serves. Losing \
                   one costs a re-download and a re-convert, not a reload.",
            reclaim: Reclaim::OnRequest,
            keep: &[],
        },
        Entry {
            name: "logs",
            path: home.join("logs"),
            what: "Engine logs. Reclaimable, but deleting them mid-investigation \
                   is its own kind of loss.",
            reclaim: Reclaim::OnRequest,
            keep: &[],
        },
        Entry {
            name: "weights",
            path: weight_cache_dir(),
            what: "Materialized device weights, keyed by checkpoint + config + \
                   quant scheme + TP layout + ABI version. One cold load to \
                   rebuild; valid only for this build.",
            reclaim: Reclaim::Safe,
            keep: &[],
        },
        Entry {
            name: "config",
            path: home.join("config.toml"),
            what: "The config file. Authored, not derived.",
            reclaim: Reclaim::Never,
            keep: &[],
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
            keep: &[],
        });
    }
    entries
}

impl Entry {
    /// The bytes this entry alone accounts for -- its tree, minus whatever
    /// belongs to another entry.
    pub fn size(&self) -> u64 {
        disk_usage(&self.path).saturating_sub(
            self.keep
                .iter()
                .map(|child| disk_usage(&self.path.join(child)))
                .sum::<u64>(),
        )
    }

    /// Delete what this entry accounts for, preserving its carve-outs.
    ///
    /// A file is removed outright. A directory with no carve-outs goes as a
    /// whole; one with carve-outs is emptied child by child, so what it holds
    /// on behalf of another entry survives -- and the directory itself is
    /// kept, since it is where those carve-outs live.
    pub fn remove(&self) -> std::io::Result<()> {
        if !self.path.is_dir() {
            return std::fs::remove_file(&self.path);
        }
        if self.keep.is_empty() {
            return std::fs::remove_dir_all(&self.path);
        }
        for child in std::fs::read_dir(&self.path)? {
            let child = child?;
            let name = child.file_name();
            if self.keep.iter().any(|kept| name == **kept) {
                continue;
            }
            let path = child.path();
            if path.is_dir() {
                std::fs::remove_dir_all(&path)?;
            } else {
                std::fs::remove_file(&path)?;
            }
        }
        Ok(())
    }
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
    entries.flatten().map(|e| disk_usage(&e.path())).sum()
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

    /// `size` and `remove` agree about what a carve-out means.
    ///
    /// Built over a scratch tree rather than `$PIE_HOME`, so it asserts the
    /// mechanism -- the registry above is what points the mechanism at the
    /// planner profile.
    #[test]
    fn a_carve_out_is_neither_counted_nor_deleted() {
        let root = std::env::temp_dir().join(format!(
            "pie-state-carve-out-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("derived")).unwrap();
        std::fs::write(root.join("derived/module.bin"), [0u8; 100]).unwrap();
        std::fs::write(root.join("measured.json"), [0u8; 30]).unwrap();

        let entry = Entry {
            name: "scratch",
            path: root.clone(),
            what: "",
            reclaim: Reclaim::Safe,
            keep: &["measured.json"],
        };
        assert_eq!(disk_usage(&root), 130, "the tree holds both");
        assert_eq!(entry.size(), 100, "the carve-out is another entry's bytes");

        entry.remove().unwrap();
        assert!(!root.join("derived").exists(), "the derived tree goes");
        assert!(
            root.join("measured.json").exists(),
            "the carve-out survives a clear that did not name it",
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// No entry's bytes are also another entry's bytes.
    ///
    /// `cache/` physically contains both the weight cache and the planner
    /// profile, and while all three were entries over the same tree, `pie
    /// cache list` counted those bytes twice and `pie cache clear` deleted
    /// `engine` -- taking `weights` with it -- and then reported an error for
    /// failing to delete a directory it had just removed.
    ///
    /// Stated over `keep` rather than over the raw paths, because nesting is
    /// the physical truth here: `weights` IS inside `cache/`. What must hold is
    /// that exactly one entry claims it.
    #[test]
    fn entries_are_disjoint() {
        let all = entries(Some(PathBuf::from("/somewhere/else/huggingface")));
        for entry in &all {
            for other in &all {
                if entry.name == other.name || !other.path.starts_with(&entry.path) {
                    continue;
                }
                let carved = other
                    .path
                    .strip_prefix(&entry.path)
                    .expect("just checked the prefix");
                assert!(
                    entry
                        .keep
                        .iter()
                        .any(|kept| std::path::Path::new(kept) == carved),
                    "{} contains {} ({}) but does not carve it out; both would \
                     count its bytes and the first to be cleared would take the \
                     second with it",
                    entry.name,
                    other.name,
                    carved.display(),
                );
            }
        }
    }

    /// Every carve-out names something real.
    ///
    /// A `keep` entry for a path no entry claims would quietly make `pie cache
    /// clear` leave litter behind forever -- the inverse failure, and just as
    /// silent.
    #[test]
    fn every_carve_out_belongs_to_some_entry() {
        let all = entries(None);
        for entry in &all {
            for kept in entry.keep {
                let path = entry.path.join(kept);
                assert!(
                    all.iter().any(|other| other.path == path),
                    "{}'s carve-out {kept} is claimed by no entry, so nothing \
                     will ever reclaim it",
                    entry.name,
                );
            }
        }
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
        assert_eq!(by_name("engine"), engine_cache_dir());
        assert_eq!(by_name("weights"), weight_cache_dir());
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

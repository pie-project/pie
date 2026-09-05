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
//! That enumeration is why the kernel caches need no line of their own here:
//! `kernels-cuda` takes its subdirectory off the same root, so its cubins and
//! its measured cuBLASLt table are counted and reclaimed with the rest.
//!
//! **Entries do not nest.** Two entries whose paths contain one another would
//! double-count in a size report and delete each other in a reclaim, so the
//! ones carved out of `cache/` are subtracted from it via [`Entry::keep`] and
//! the `entries_are_disjoint` test pins that.

use std::path::PathBuf;

/// The root every engine-side disk cache derives from.
///
/// Defined here rather than at the call site so the registry below and the
/// boot path that actually tells the engines where to write -- a `DeviceBoot`
/// field -- cannot drift. A `pie cache clear` that looked in a directory
/// nothing writes to would report success and reclaim nothing.
pub fn engine_cache_dir() -> PathBuf {
    bootstrap::paths::pie_home().join("cache")
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
    let home = bootstrap::paths::pie_home();
    let mut entries = vec![
        Entry {
            name: "engine",
            path: engine_cache_dir(),
            // `kernels-cuda` writes `cubins/` and `gemm-algos/` under this
            // root rather than resolving `$XDG_CACHE_HOME` for itself, so its
            // cubins and its measured cuBLASLt table are enumerated from disk
            // with the rest and are reclaimable with them.
            what: "Engine-side disk caches: compiled ETA modules, kernel \
                   cubins, GEMM autotuning results. All keyed and \
                   self-invalidating; deleting costs one cold rebuild.",
            reclaim: Reclaim::Safe,
            // **NOTHING IS CARVED OUT OF IT.** The model's own `.zt` is
            // where this machine holds its weights; it is not under `cache/`
            // and it is not reclaimable.
            keep: &[],
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
            // The artifact store, not a cache: a `.zt` is not re-derived by
            // the next load. Losing one costs a download and a conversion,
            // and the file is portable in a way device weights are not -- no
            // TP layout and no ABI version are baked into it.
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
}

use std::path::{Path, PathBuf};

pub fn engine_cache_dir(home: &Path) -> PathBuf {
    home.join("cache")
}

/// Where a load's kv slot files live when the config names no path.
///
/// A persistent tier holds kv computed from these exact weights and is read
/// again by the next run: one directory per checkpoint content digest, so
/// no run reads kv another checkpoint wrote. The digest reads each file once
/// and is memoized beside the directories by each file's stats
/// ([`crate::weights::model_artifact_digest`]). A directory named by an
/// earlier digest is not found under the new name and is left to cache
/// cleanup.
///
/// A scratch tier (the default) never outlives its process, so no run can
/// read another's kv: its directory is named by the checkpoint's path, and
/// no byte of the checkpoint is read for it. Two live processes serving one
/// path share it exactly as they shared the digest-named directory before:
/// each slot file is locked at open (a second server is refused by name) and
/// unlinked once locked, so a later process creates its own.
pub fn kv_dir(
    engine_cache_dir: &Path,
    snapshot_dir: &Path,
    persistent: bool,
) -> anyhow::Result<PathBuf> {
    let kv = engine_cache_dir.join("kv");
    if !persistent {
        let path = std::fs::canonicalize(snapshot_dir).unwrap_or_else(|_| snapshot_dir.into());
        let named = blake3::hash(path.as_os_str().as_encoded_bytes());
        return Ok(kv.join(format!("scratch-{}", named.to_hex())));
    }
    let digest = crate::weights::model_artifact_digest(snapshot_dir, &kv.join("digests"))?;
    Ok(kv.join(blake3::Hash::from(digest).to_hex().as_str()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reclaim {
    Safe,
    OnRequest,
    Never,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: &'static str,
    pub path: PathBuf,
    pub what: &'static str,
    pub reclaim: Reclaim,
    pub keep: &'static [&'static str],
}

pub fn entries(home: &Path, hf_cache: Option<PathBuf>) -> Vec<Entry> {
    let mut entries = vec![
        Entry {
            name: "engine",
            path: engine_cache_dir(home),
            what: "Engine-side disk caches: compiled ETA modules, kernel \
                   cubins, GEMM autotuning results. All keyed and \
                   self-invalidating; deleting costs one cold rebuild.",
            reclaim: Reclaim::Safe,
            keep: &[],
        },
        Entry {
            name: "inferlets",
            path: home.join("inferlets"),
            what: "Installed inferlets. Deleting one means `pie inferlet \
                   install` again; the built-in ones need no install.",
            reclaim: Reclaim::Safe,
            keep: &[],
        },
        Entry {
            name: "languages",
            path: home.join("languages"),
            what: "The language components script inferlets run under \
                   (`python.wasm`, `javascript.wasm`). `pie language install` \
                   puts one back from a release archive.",
            reclaim: Reclaim::Safe,
            keep: &[],
        },
        Entry {
            name: "artifacts",
            path: home.join("artifacts"),
            what: "Imported `.zt` artifacts -- the models pie serves. Losing \
                   one costs a re-download and a re-import, not a reload.",
            reclaim: Reclaim::OnRequest,
            keep: &[],
        },
        Entry {
            name: "models",
            path: home.join("models"),
            what: "Model packages that shadow or add to the ones built in. \
                   Deleting one means the built-in package serves again, or \
                   nothing does.",
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
    pub fn size(&self) -> u64 {
        disk_usage(&self.path).saturating_sub(
            self.keep
                .iter()
                .map(|child| disk_usage(&self.path.join(child)))
                .sum::<u64>(),
        )
    }

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
        for entry in entries(Path::new("/home"), None) {
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

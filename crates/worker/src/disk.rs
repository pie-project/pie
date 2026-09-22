use std::path::PathBuf;

pub fn engine_cache_dir() -> PathBuf {
    bootstrap::paths::pie_home().join("cache")
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

pub fn entries(hf_cache: Option<PathBuf>) -> Vec<Entry> {
    let home = bootstrap::paths::pie_home();
    let mut entries = vec![
        Entry {
            name: "engine",
            path: engine_cache_dir(),
            what: "Engine-side disk caches: compiled ETA modules, kernel \
                   cubins, GEMM autotuning results. All keyed and \
                   self-invalidating; deleting costs one cold rebuild.",
            reclaim: Reclaim::Safe,
            keep: &[],
        },
        Entry {
            name: "inferlets",
            path: bootstrap::paths::inferlets_dir(),
            what: "Installed inferlets. Deleting one means `pie inferlet \
                   install` again; the built-in ones need no install.",
            reclaim: Reclaim::Safe,
            keep: &[],
        },
        Entry {
            name: "languages",
            path: bootstrap::paths::languages_dir(),
            what: "The language components script inferlets run under \
                   (`python.wasm`, `javascript.wasm`). `pie language install` \
                   puts one back from a release archive.",
            reclaim: Reclaim::Safe,
            keep: &[],
        },
        Entry {
            name: "models",
            path: home.join("models"),
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

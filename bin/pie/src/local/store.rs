//! `PIE_HOME/models/` — the artifacts pie serves.
//!
//! A flat directory of `.zt` files, and the only thing that makes it a *store*
//! rather than a directory is the two questions asked of it: what is in it, and
//! which files make up each entry. Both are answered by opening the artifacts,
//! not by parsing their names.
//!
//! That matters for the second question. A sharded artifact is a root beside
//! `<stem>-00001.zt`, `<stem>-00002.zt`, … and it would be easy to decide by
//! filename which of those is a root — until someone converts a model actually
//! called `llama-00001`. So the store opens each file and lets the checkpoint
//! reader resolve the set: the files it reports beyond the first are shards,
//! and a shard is never an entry of its own.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use pie_loader::checkpoint::read::parse_checkpoint_metadata;

// Same names `convert` writes under, taken from the same place rather than
// mirrored here.
use pie_loader::checkpoint::meta::{SOURCE_KEY, VERSION_KEY};

/// `$PIE_HOME/models/`.
pub fn dir() -> PathBuf {
    startup::paths::pie_home().join("models")
}

/// One artifact: its root, the files it is made of, and what it says about
/// itself.
pub struct Entry {
    /// The store name, without `.zt`.
    pub name: String,
    /// The root file. For a single-file artifact this is the whole of it.
    pub root: PathBuf,
    /// Root plus shards, in the order the checkpoint reports them.
    pub files: Vec<PathBuf>,
    pub bytes: u64,
    pub tensors: usize,
    /// `pie_version` from the artifact's attributes, when it has one.
    pub written_by: Option<String>,
    /// `pie_source` from the artifact's attributes, when it has one.
    pub source: Option<String>,
}

impl Entry {
    pub fn shards(&self) -> usize {
        self.files.len().saturating_sub(1)
    }
}

/// Every artifact in the store, by name.
///
/// Files that turn out to be shards of another artifact are not entries.
/// A `.zt` that cannot be opened is skipped rather than fatal: a store with one
/// corrupt file should still list the rest, and `convert --force` is the fix.
pub fn entries() -> Result<Vec<Entry>> {
    let dir = dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|err| anyhow!("cannot read {}: {err}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "zt"))
        .collect();
    candidates.sort();

    Ok(entries_from(candidates))
}

/// The scan itself, over an explicit file list.
fn entries_from(candidates: Vec<PathBuf>) -> Vec<Entry> {
    let mut parsed = Vec::new();
    let mut claimed: BTreeSet<PathBuf> = BTreeSet::new();
    for path in candidates {
        let Ok(metadata) = parse_checkpoint_metadata(&path) else {
            continue;
        };
        let files: Vec<PathBuf> = metadata
            .files
            .iter()
            .map(|file| PathBuf::from(&file.path))
            .collect();
        // Everything after the first is a shard, and a shard is not an entry.
        for shard in files.iter().skip(1) {
            claimed.insert(shard.canonicalize().unwrap_or_else(|_| shard.clone()));
        }
        let attributes = pie_loader::checkpoint::zt::read_attributes(&path).unwrap_or_default();
        parsed.push(Entry {
            name: path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            bytes: files
                .iter()
                .filter_map(|f| std::fs::metadata(f).ok())
                .map(|m| m.len())
                .sum(),
            tensors: metadata.weights().count(),
            written_by: attributes.get(VERSION_KEY).cloned(),
            source: attributes.get(SOURCE_KEY).cloned(),
            root: path,
            files,
        });
    }
    parsed.retain(|entry| {
        let key = entry
            .root
            .canonicalize()
            .unwrap_or_else(|_| entry.root.clone());
        !claimed.contains(&key)
    });
    parsed
}

/// The artifact stored under `name`, if there is one.
pub fn find(name: &str) -> Result<Option<Entry>> {
    Ok(entries()?.into_iter().find(|entry| entry.name == name))
}

/// Deletes an artifact: the root last, so an interrupted removal leaves a
/// root whose shards are missing — which fails loudly on open — rather than
/// orphan shards no command would ever mention again.
pub fn remove(entry: &Entry) -> Result<()> {
    for shard in entry.files.iter().skip(1) {
        std::fs::remove_file(shard)
            .map_err(|err| anyhow!("cannot delete {}: {err}", shard.display()))?;
    }
    std::fs::remove_file(&entry.root)
        .map_err(|err| anyhow!("cannot delete {}: {err}", entry.root.display()))?;
    Ok(())
}

/// The HF staging snapshot for a repo, when one is present.
///
/// The cache demotes to a staging area once artifacts exist: it is what
/// `convert` reads, not what serving reads.
pub fn staging_dir(repo_id: &str) -> Option<PathBuf> {
    let dir = hf_hub::resolve_cache_dir().join(format!("models--{}", repo_id.replace('/', "--")));
    dir.is_dir().then_some(dir)
}

/// Bytes a staging snapshot occupies, following no symlinks (the HF cache
/// points snapshot entries at `blobs/`, and counting both double-counts).
pub fn staging_bytes(dir: &Path) -> u64 {
    fn walk(dir: &Path) -> std::io::Result<u64> {
        let mut total = 0;
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                total += walk(&entry.path())?;
            } else if metadata.is_file() {
                total += metadata.len();
            }
        }
        Ok(total)
    }
    walk(dir).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_loader::checkpoint::write::CheckpointWriter;
    use pie_loader::types::{DType, Encoding, TensorDecl, TensorId, Visibility};

    fn decl(name: &str) -> TensorDecl {
        TensorDecl {
            id: TensorId(0),
            name: name.to_string(),
            shape: vec![32_000],
            encoding: Encoding::Raw(DType::U8),
            alignment: 1,
            visibility: Visibility::default(),
        }
    }

    /// A sharded artifact is one entry, not one per file.
    ///
    /// The shards are named `<stem>-00001.zt`, which a listing could mistake
    /// for models of their own — and a filename rule would also mistake a
    /// model actually *called* `llama-00001`. So the store opens each file and
    /// lets the reader say which files belong to which artifact.
    #[test]
    fn shards_are_not_entries_of_their_own() {
        let dir = tempfile::tempdir().unwrap();
        let payload = vec![7u8; 32_000];

        let single = dir.path().join("solo.zt");
        let mut writer = CheckpointWriter::create(&single, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();

        let sharded = dir.path().join("split.zt");
        let mut writer =
            CheckpointWriter::create_sharded(&sharded, &Default::default(), 40_000).unwrap();
        for i in 0..3 {
            writer
                .add_tensor(&decl(&format!("w{i}")), &payload)
                .unwrap();
        }
        writer.finish().unwrap();

        // Five `.zt` files on disk; two artifacts.
        let on_disk = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(
            on_disk, 5,
            "expected a root plus three shards plus the solo"
        );

        let mut found: Vec<Entry> = {
            let mut candidates: Vec<std::path::PathBuf> = std::fs::read_dir(dir.path())
                .unwrap()
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "zt"))
                .collect();
            candidates.sort();
            entries_from(candidates)
        };
        found.sort_by(|a, b| a.name.cmp(&b.name));

        let names: Vec<&str> = found.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["solo", "split"]);
        assert_eq!(found[0].shards(), 0);
        assert_eq!(found[1].shards(), 3);
        assert_eq!(found[1].tensors, 3);
        // The reported size is the whole set, not just the root.
        assert!(found[1].bytes > 3 * 32_000);
    }
}

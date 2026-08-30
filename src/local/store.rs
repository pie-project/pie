//! `PIE_HOME/models/` — the artifacts pie serves.
//!
//! Two layers, one directory per model:
//!
//! ```text
//! models/<name>/archive.zt          the import output
//! models/<name>/archive-00001.zt    its shards, when it has any
//! models/<name>/runtime/<key>.zt    a build output, one per target
//! ```
//!
//! The **archive** is general form: whatever the source said, expressed in
//! pie's own vocabulary, and servable on its own. It is the narrow waist, and
//! there is exactly one of it per model.
//!
//! A **runtime** is that same model already laid out for one target — a
//! particular backend, quantization and MoE lowering. There can be many, they
//! are all derivable from the archive, and deleting one costs a rebuild and
//! nothing else. `<key>` named the target the artifact was built for, so two
//! targets could not land on the same file.
//!
//! NOTHING IN THIS BUILD WRITES ONE. The only producer was `pie model build`,
//! deleted in R3, and the key it stemmed the filename from was deleted with
//! this crate's last reader of it. What survives here is the *reading* half —
//! a store an older pie wrote still lists its runtimes rather than going
//! partly invisible — and it is why the vocabulary is documented at all.
//!
//! This replaced a flat directory of `<name>.zt` beside `<name>-optimized.zt`,
//! where the relationship between an archive and its build was a name suffix,
//! only one build could exist at a time, and nothing recorded which backend it
//! was for. Flat entries are still *listed* so that a store written by an
//! older pie does not go invisible, but nothing writes them any more.
//!
//! The only thing that makes this a *store* rather than a directory tree is
//! the two questions asked of it: what is in it, and which files make up each
//! entry. The second is answered by opening the artifacts, not by parsing
//! their names. A sharded artifact is a root beside `<stem>-00001.zt`,
//! `<stem>-00002.zt`, … and it would be easy to decide by filename which of
//! those is a root — until someone converts a model actually called
//! `llama-00001`. So the store opens each file and lets the checkpoint reader
//! resolve the set: the files it reports beyond the first are shards, and a
//! shard is never an entry of its own.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use checkpoint::file::read::parse_metadata;

// Same names `convert` writes under, taken from the same place rather than
// mirrored here.
use checkpoint::file::meta::{SOURCE_KEY, VERSION_KEY};

/// The archive's filename inside a model directory.
///
/// A fixed name rather than `<name>.zt` because the directory already says
/// which model this is, and repeating it there is a second place for the two
/// to disagree.
pub const ARCHIVE_FILE: &str = "archive.zt";

/// The subdirectory holding per-target builds.
pub const RUNTIME_DIR: &str = "runtime";

/// `$PIE_HOME/models/`.
pub fn dir() -> PathBuf {
    bootstrap::paths::pie_home().join("models")
}

/// `$PIE_HOME/models/<name>/` — everything pie holds for one model.
pub fn model_dir(name: &str) -> PathBuf {
    dir().join(name)
}

/// `$PIE_HOME/models/<name>/archive.zt` — where `pie model import` writes.
pub fn archive_path(name: &str) -> PathBuf {
    model_dir(name).join(ARCHIVE_FILE)
}

// `runtime_dir(name)` and `runtime_path(name, key)` STOOD HERE: the two path
// constructors a `pie model build` used to write `<name>/runtime/<key>.zt`
// with. That command is deleted, so nothing in this build constructs either
// path, and the cache key the second one spelled its stem from went with it.
// [`RUNTIME_DIR`] and [`read_runtimes`] stay, because reading is the half that
// still has a job: an operator may hold a directory an older pie wrote, and
// `pie model list` reports whatever is actually there -- which, in this build,
// is nothing.

/// One per-target build of a model.
pub struct Runtime {
    /// The cache key, which is also the file stem.
    pub key: String,
    /// Root plus shards.
    pub files: Vec<PathBuf>,
    pub bytes: u64,
    /// `pie_runtime_quant` from the artifact's attributes, when it has one.
    pub runtime_quant: Option<String>,
}

/// One artifact: its root, the files it is made of, and what it says about
/// itself.
pub struct Entry {
    /// The store name, without `.zt`.
    pub name: String,
    /// The model directory, for an entry in the two-layer layout. `None` for a
    /// flat `<name>.zt` written by an older pie.
    pub dir: Option<PathBuf>,
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
    /// Per-target builds derived from this archive, by key.
    pub runtimes: Vec<Runtime>,
}

impl Entry {
    pub fn shards(&self) -> usize {
        self.files.len().saturating_sub(1)
    }

    /// Bytes of the archive and every runtime built from it.
    pub fn total_bytes(&self) -> u64 {
        self.bytes + self.runtimes.iter().map(|r| r.bytes).sum::<u64>()
    }
}

/// Every artifact in the store, by name.
///
/// Files that turn out to be shards of another artifact are not entries.
/// A `.zt` that cannot be opened is skipped rather than fatal: a store with one
/// corrupt file should still list the rest, and `convert --force` is the fix.
pub fn entries() -> Result<Vec<Entry>> {
    entries_in(&dir())
}

/// The scan itself, over an explicit store root.
fn entries_in(dir: &Path) -> Result<Vec<Entry>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut children: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|err| anyhow!("cannot read {}: {err}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect();
    children.sort();

    let mut found = Vec::new();
    for child in children.iter().filter(|path| path.is_dir()) {
        let name = child
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Some(mut entry) = read_entry(&child.join(ARCHIVE_FILE), name) else {
            continue;
        };
        entry.dir = Some(child.clone());
        entry.runtimes = read_runtimes(&child.join(RUNTIME_DIR));
        found.push(entry);
    }

    // Flat `.zt` at the top level are the older layout. Read but never
    // written — and only when nothing has taken the name, because a
    // re-import writes `<name>/archive.zt` beside the `<name>.zt` it does
    // not delete. Listing both showed one model twice, and `find` returning
    // the older of the two meant `pie model remove` deleted the file nobody
    // serves and reported success while the model stayed where it was. The
    // worker already prefers the archive (`crates/worker/src/weights.rs`);
    // this is the store agreeing with it.
    let taken: BTreeSet<String> = found.iter().map(|entry| entry.name.clone()).collect();
    let flat: Vec<PathBuf> = children
        .iter()
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "zt"))
        .cloned()
        .collect();
    found.extend(
        entries_from(flat)
            .into_iter()
            .filter(|entry| !taken.contains(&entry.name)),
    );

    found.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(found)
}

/// Reads one artifact root into an entry, or nothing if it cannot be opened.
fn read_entry(root: &Path, name: String) -> Option<Entry> {
    let metadata = parse_metadata(root).ok()?;
    let files: Vec<PathBuf> = metadata
        .files
        .iter()
        .map(|file| PathBuf::from(&file.path))
        .collect();
    let attributes = checkpoint::file::zt::read_attributes(root).unwrap_or_default();
    Some(Entry {
        name,
        dir: None,
        bytes: files
            .iter()
            .filter_map(|f| std::fs::metadata(f).ok())
            .map(|m| m.len())
            .sum(),
        tensors: metadata.weights().count(),
        written_by: attributes.get(VERSION_KEY).cloned(),
        source: attributes.get(SOURCE_KEY).cloned(),
        root: root.to_path_buf(),
        files,
        runtimes: Vec::new(),
    })
}

/// Every build under `<name>/runtime/`, shards folded into their roots.
///
/// Shares [`entries_from`]'s shard folding rather than repeating it: a runtime
/// shards exactly the way an archive does, and two implementations of "which
/// of these files is a root" is one more than can stay in agreement.
fn read_runtimes(dir: &Path) -> Vec<Runtime> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut candidates: Vec<PathBuf> = read
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "zt"))
        .collect();
    candidates.sort();

    entries_from(candidates)
        .into_iter()
        .map(|entry| Runtime {
            key: entry.name,
            bytes: entry.bytes,
            runtime_quant: checkpoint::file::zt::read_attributes(&entry.root)
                .unwrap_or_default()
                .get(checkpoint::file::meta::RUNTIME_QUANT_KEY)
                .cloned(),
            files: entry.files,
        })
        .collect()
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// The scan itself, over an explicit file list.
fn entries_from(candidates: Vec<PathBuf>) -> Vec<Entry> {
    let mut parsed = Vec::new();
    let mut claimed: BTreeSet<PathBuf> = BTreeSet::new();
    for path in candidates {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Some(entry) = read_entry(&path, name) else {
            continue;
        };
        // Everything after the first is a shard, and a shard is not an entry.
        for shard in entry.files.iter().skip(1) {
            claimed.insert(canonical(shard));
        }
        parsed.push(entry);
    }
    parsed.retain(|entry| !claimed.contains(&canonical(&entry.root)));
    parsed
}

/// The artifact stored under `name`, if there is one.
pub fn find(name: &str) -> Result<Option<Entry>> {
    Ok(entries()?.into_iter().find(|entry| entry.name == name))
}

/// Deletes an artifact: the root last, so an interrupted removal leaves a
/// root whose shards are missing — which fails loudly on open — rather than
/// orphan shards no command would ever mention again.
///
/// For a two-layer entry this takes the runtimes with it. They are derived, so
/// there is nothing to keep once the archive they came from is gone, and
/// leaving them would leave files no command names.
pub fn remove(entry: &Entry) -> Result<()> {
    for runtime in &entry.runtimes {
        for file in runtime.files.iter().skip(1) {
            std::fs::remove_file(file)
                .map_err(|err| anyhow!("cannot delete {}: {err}", file.display()))?;
        }
        std::fs::remove_file(&runtime.files[0])
            .map_err(|err| anyhow!("cannot delete {}: {err}", runtime.files[0].display()))?;
    }
    for shard in entry.files.iter().skip(1) {
        std::fs::remove_file(shard)
            .map_err(|err| anyhow!("cannot delete {}: {err}", shard.display()))?;
    }
    std::fs::remove_file(&entry.root)
        .map_err(|err| anyhow!("cannot delete {}: {err}", entry.root.display()))?;
    if let Some(dir) = &entry.dir {
        // Best effort, and only if it is now empty: a file this store does not
        // know about is a reason to leave the directory, not to delete it.
        let _ = std::fs::remove_dir(dir.join(RUNTIME_DIR));
        let _ = std::fs::remove_dir(dir);
    }
    Ok(())
}

/// The HF staging snapshot for a repo, when one is present.
///
/// The cache demotes to a staging area once artifacts exist: it is what
/// `convert` reads, not what serving reads.
pub fn staging_dir(repo_id: &str) -> Option<PathBuf> {
    let dir = crate::local::hf::resolve_cache_dir()
        .join(format!("models--{}", repo_id.replace('/', "--")));
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
    use checkpoint::file::write::Writer;
    use checkpoint::types::{DType, Encoding, TensorDecl, TensorId, Visibility};

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
        let mut writer = Writer::create(&single, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();

        let sharded = dir.path().join("split.zt");
        let mut writer = Writer::create_sharded(&sharded, &Default::default(), 40_000).unwrap();
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

    /// A model directory is one entry, and its builds hang off it.
    ///
    /// The point of the two-layer store: `models/<name>/archive.zt` is the one
    /// artifact per model, and `models/<name>/runtime/<key>.zt` are derived
    /// from it. The old flat layout could not express this — a build was a
    /// sibling of the archive with a `-optimized` suffix, so a listing showed
    /// two models where there is one, and only one build could exist at a
    /// time because they all shared a name.
    ///
    /// The runtimes here are checked to fold their own shards, because that is
    /// the mistake this layout makes easy to write twice: the archive scan and
    /// the runtime scan are both "which of these files is a root", and two
    /// implementations of that cannot stay in agreement.
    #[test]
    fn a_model_directory_is_one_entry_with_its_builds_beneath_it() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("qwen--qwen3-0.6b");
        let payload = vec![7u8; 32_000];

        let archive = model.join(ARCHIVE_FILE);
        let mut writer = Writer::create(&archive, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();

        // Two builds of the same model, one of them sharded.
        let plain = model.join(RUNTIME_DIR).join("0123456789abcdef.zt");
        let mut writer = Writer::create(&plain, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();

        let split = model.join(RUNTIME_DIR).join("fedcba9876543210.zt");
        let mut writer = Writer::create_sharded(&split, &Default::default(), 40_000).unwrap();
        for i in 0..3 {
            writer
                .add_tensor(&decl(&format!("w{i}")), &payload)
                .unwrap();
        }
        writer.finish().unwrap();

        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 1, "one model, not one per file");
        let entry = &found[0];
        assert_eq!(entry.name, "qwen--qwen3-0.6b");

        assert_eq!(entry.tensors, 1, "the archive is one tensor");
        // Five `.zt` under runtime/; two builds.
        let on_disk = std::fs::read_dir(model.join(RUNTIME_DIR)).unwrap().count();
        assert_eq!(
            on_disk, 5,
            "a root plus three shards plus the unsharded one"
        );
        let keys: Vec<&str> = entry.runtimes.iter().map(|r| r.key.as_str()).collect();
        assert_eq!(keys, ["0123456789abcdef", "fedcba9876543210"]);
        assert_eq!(entry.runtimes[1].files.len(), 4, "root plus three shards");

        // The size a listing reports for the model is everything under it, and
        // the builds are the bulk of it — a runtime artifact is about as large
        // as the archive it came from, so a model with two of them occupies
        // three times what the archive line alone would say.
        assert!(entry.total_bytes() > 3 * entry.bytes);
    }

    /// The two-layer entry wins when a flat file of the same name is still there.
    ///
    /// This is the ordinary upgrade path, not a corner case: a store written by
    /// an older pie holds `qwen.zt`, and re-importing writes `qwen/archive.zt`
    /// beside it without deleting anything. Listing both showed one model
    /// twice; worse, `find` returning the older of the two made `pie model
    /// remove` delete the file nobody serves and report success while the
    /// model stayed exactly where it was.
    ///
    /// The worker resolves the same way round
    /// (`crates/worker/src/weights.rs`), and the two must not disagree — a
    /// store where serving reads one file and every `pie model` command reads
    /// another is worse than either rule alone.
    #[test]
    fn a_leftover_flat_file_does_not_shadow_the_archive_that_replaced_it() {
        let root = tempfile::tempdir().unwrap();
        let payload = vec![7u8; 32_000];

        // The older layout, with one tensor.
        let flat = root.path().join("qwen.zt");
        let mut writer = Writer::create(&flat, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();

        // The re-import, with two, so the entries are told apart by content.
        let archive = root.path().join("qwen").join(ARCHIVE_FILE);
        let mut writer = Writer::create(&archive, &Default::default()).unwrap();
        writer.add_tensor(&decl("a"), &payload).unwrap();
        writer.add_tensor(&decl("b"), &payload).unwrap();
        writer.finish().unwrap();

        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 1, "one model, not two");
        assert_eq!(found[0].name, "qwen");
        assert_eq!(found[0].tensors, 2, "the archive, not the flat leftover");
        assert_eq!(found[0].root, archive);

        // A flat file whose name nothing has taken is still an entry.
        let other = root.path().join("legacy.zt");
        let mut writer = Writer::create(&other, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();
        let names: Vec<String> = entries_in(root.path())
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, ["legacy", "qwen"]);
    }

    /// A store written by an older pie still lists.
    ///
    /// Flat `<name>.zt` is read but never written. Dropping the old shape
    /// outright would have made every previously imported model report as
    /// missing, which reads as data loss whether or not the files are still
    /// there.
    #[test]
    fn a_flat_artifact_from_an_older_pie_is_still_an_entry() {
        let dir = tempfile::tempdir().unwrap();
        let flat = dir.path().join("legacy.zt");
        let mut writer = Writer::create(&flat, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &vec![7u8; 32_000]).unwrap();
        writer.finish().unwrap();

        let found = entries_from(vec![flat]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "legacy");
        assert!(
            found[0].dir.is_none(),
            "a flat artifact has no model directory, and `remove` must not \
             delete the store root looking for one"
        );
    }

    /// The store's paths agree with each other.
    ///
    /// Three functions name places under one model directory, and the layout
    /// is only a layout if they nest. Stated as a test rather than by deriving
    /// each from the last, so that a change to any one of them has to be a
    /// change here too.
    #[test]
    fn every_store_path_is_under_the_model_directory() {
        let model = model_dir("qwen--qwen3-0.6b");
        assert_eq!(model.parent().unwrap(), dir());
        assert_eq!(archive_path("qwen--qwen3-0.6b"), model.join("archive.zt"));
        assert_eq!(model.join(RUNTIME_DIR), model.join("runtime"));
    }
}

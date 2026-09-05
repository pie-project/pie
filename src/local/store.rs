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
//! A directory holds ONE archive and as many SERVING artifacts as have been
//! imported into it, each named `<slug>.<sku>.<backend>.zt`:
//!
//! ```text
//! models/<name>/<name>.<sku>.cuda.zt
//! models/<name>/<name>.<sku>.vulkan.zt
//! ```
//!
//! That is the case the naming exists for — one model, two shells or two
//! rows — so it is one *entry* per artifact and not one per directory, and
//! the name that reaches each of them is its whole filename
//! ([`Entry::address`]). A bare model name still reaches the artifact under
//! it when there is only one, which is every config written before a second
//! import.
//!
//! The **archive** is general form: whatever the source said, expressed in
//! pie's own vocabulary, and servable on its own. It is the narrow waist, and
//! there is exactly one of it per model.
//!
//! A **runtime** is that same model already laid out for one target — a
//! particular backend, quantization and MoE lowering. There can be many, they
//! are all derivable from the archive, and deleting one costs a rebuild and
//! nothing else. `<key>` names the target the artifact was built for, so two
//! targets cannot land on the same file.
//!
//! NOTHING IN THIS BUILD WRITES ONE: this build ships no command that lays a
//! model out per target. Only the *reading* half is here, so that a store
//! another pie wrote still lists its runtimes rather than going partly
//! invisible — which is why the vocabulary is documented at all. Flat entries
//! (`<name>.zt` beside `<name>-optimized.zt`) are listed for the same reason.
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

// Nothing in this build constructs a `<name>/runtime/<key>.zt` path, so there
// are no constructors for one. [`RUNTIME_DIR`] and [`read_runtimes`] stay
// because reading is the half that still has a job: an operator may hold a
// directory another pie wrote, and `pie model list` reports whatever is
// actually there.

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
    /// The store name, without `.zt`: the model directory for a two-layer
    /// entry, the file stem for a flat one. **NOT UNIQUE** — a model
    /// imported for two shells is two entries under one name, which is what
    /// [`qualified`](Entry::qualified) and [`address`](Entry::address) are
    /// for.
    pub name: String,
    /// `<slug>.<sku>.<backend>` — the whole of a specialized artifact's
    /// filename, which is the one string that names it and no other. `None`
    /// for `archive.zt` and for a flat file, which their `name` already
    /// names alone.
    pub qualified: Option<String>,
    /// How many artifacts this entry's model directory holds, itself
    /// included. More than one is the ordinary case for a model imported for
    /// two backends, and the reason [`address`](Entry::address) exists.
    pub siblings: usize,
    /// The catalog row this artifact was compiled for, from its own serving
    /// stamp. `None` for a `.zt` carrying none, which is a checkpoint rather
    /// than a serving artifact.
    pub sku: Option<String>,
    /// The engine whose kernels its bytes are landed for, from the same
    /// stamp. This is what a `--features vulkan` build matches on.
    pub backend: Option<String>,
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

    /// The name to type at this artifact: its `name` when that names it
    /// alone, and its whole `<slug>.<sku>.<backend>` when the model
    /// directory holds siblings.
    ///
    /// What a listing prints, and what [`find`] resolves — a listing that
    /// printed three rows all called `google--gemma-4-E4B-it` would name
    /// nothing an operator could act on.
    pub fn address(&self) -> &str {
        match &self.qualified {
            Some(qualified) if self.siblings > 1 => qualified,
            _ => &self.name,
        }
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
        // **EVERY** artifact in the directory, not the one it holds when it
        // holds one. `archive.zt` when that is the name, and otherwise all
        // of the specialized ones the import states a servable artifact's
        // SKU and target in — a model imported for cuda and for vulkan is
        // two files here, and asking for a single root found neither of
        // them and showed an empty shelf over 200 GB of artifacts.
        //
        // Asked through the checkpoint reader's own discovery rather than
        // spelled again here — see `discover_zt_files`, and the note below
        // about the worker and this scan not being allowed to disagree.
        let roots = checkpoint::file::read::discover_zt_files(child);
        // Through the same shard folding the flat scan uses: a specialized
        // artifact shards exactly the way an archive does.
        let mut in_dir = entries_from(roots);
        let siblings = in_dir.len();
        for entry in &mut in_dir {
            entry.qualified = qualified_name(&entry.root);
            entry.name = name.clone();
            entry.siblings = siblings;
            entry.dir = Some(child.clone());
            // Only when the directory holds one artifact. A build under
            // `<name>/runtime/` is derived from *the* archive, and the
            // layout has no way to say which of several it came from —
            // hanging the same set off each would report its bytes N times
            // and let `pie model remove` of one artifact delete builds of
            // another.
            if siblings == 1 {
                entry.runtimes = read_runtimes(&child.join(RUNTIME_DIR));
            }
        }
        found.extend(in_dir);
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

/// `<slug>.<sku>.<backend>`, when that is what the file is called.
///
/// Read off the name rather than rebuilt from the stamp: this is the string
/// an operator types and a directory listing shows, so it has to be the one
/// on disk. `checkpoint::serving::Name` is the same parser the import names
/// with, so a name it refuses is not one pie wrote.
fn qualified_name(root: &Path) -> Option<String> {
    let file_name = root.file_name()?.to_str()?;
    checkpoint::serving::Name::parse(file_name).ok()?;
    Some(file_name.strip_suffix(".zt")?.to_string())
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
    // What the artifact says it is for. From the stamp, not from the name:
    // the name is how it is addressed, this is what it holds. A `.zt` with
    // no stamp is an ordinary checkpoint and answers neither.
    let stamp = checkpoint::file::serve::stamp_of(root).ok().flatten();
    Some(Entry {
        name,
        qualified: qualified_name(root),
        siblings: 1,
        sku: stamp.as_ref().map(|stamp| stamp.sku.clone()),
        backend: stamp.as_ref().map(|stamp| stamp.backend.clone()),
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

/// What the store holds under a name.
///
/// Three answers and not two, because a model directory holds as many
/// artifacts as were imported into it and a bare model name is then a
/// question with several answers. Guessing one would serve, or delete, a
/// file the operator did not name.
pub enum Resolved {
    /// Nothing in the store answers to this name.
    Missing,
    One(Box<Entry>),
    /// Several artifacts share the name; each one's [`Entry::address`], in
    /// listing order, so the refusal can name what to type instead.
    Ambiguous(Vec<String>),
}

/// The artifact stored under `name`.
///
/// Two spellings resolve, and this is the whole of the rule:
///
/// * `<slug>.<sku>.<backend>` — a specialized artifact's own filename —
///   names exactly one file, always, whatever else shares its directory.
/// * a model name (the directory, or a flat file's stem) names the artifact
///   under it when there is one, and is ambiguous when there are several.
///
/// The same order the worker resolves `[model] model` in
/// (`crates/worker/src/weights.rs`), so a name that serves is a name this
/// command answers about — with the one difference that a running worker can
/// break a tie its own engine flavor settles, and a listing has no engine.
pub fn find(name: &str) -> Result<Resolved> {
    Ok(resolve_in(entries()?, name))
}

/// [`find`]'s rule, over an explicit listing: the scan is the slow half and
/// the rule is the half worth testing.
fn resolve_in(entries: Vec<Entry>, name: &str) -> Resolved {
    if let Some(exact) = entries
        .iter()
        .position(|entry| entry.qualified.as_deref() == Some(name))
    {
        return Resolved::One(Box::new(
            entries
                .into_iter()
                .nth(exact)
                .expect("the position just found is in the vector it was found in"),
        ));
    }
    let mut named: Vec<Entry> = entries
        .into_iter()
        .filter(|entry| entry.name == name)
        .collect();
    match named.len() {
        0 => Resolved::Missing,
        1 => Resolved::One(Box::new(named.remove(0))),
        _ => Resolved::Ambiguous(
            named
                .iter()
                .map(|entry| entry.address().to_string())
                .collect(),
        ),
    }
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

    /// **A STAMPED IMPORT'S OWN FILENAME IS AN ENTRY.**
    ///
    /// A servable artifact is named for what it is —
    /// `<slug>.<sku>.<backend>.zt` — so that one model at two
    /// quantizations can share a directory. A scan that only opened
    /// `<name>/archive.zt` would find nothing and skip the entry: a
    /// `--features metal` import of the DeepSeek-V4 mini wrote
    /// `mini-l5-e16.dsv4-flash-u4g64-u2g64-kv-bf16.metal.zt` into the
    /// store, exited zero, and `pie model list` showed an empty shelf. Three
    /// artifacts were invisible on this box when the fix went in.
    ///
    /// The scan asks `checkpoint`'s own discovery, which is the same answer the
    /// worker resolves with — the two are not allowed to disagree, and this is
    /// the disagreement they had.
    #[test]
    fn an_artifact_named_for_its_specialization_is_listed() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("deepseek");
        let specialized = model.join("deepseek.dsv4-flash-full-u4g64-u2g64-kv-bf16.metal.zt");
        let mut writer = Writer::create(&specialized, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &vec![7u8; 32_000]).unwrap();
        writer.finish().unwrap();

        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 1, "the store holds one model and shows it");
        assert_eq!(
            found[0].name, "deepseek",
            "the entry is named for its directory, as every entry is"
        );
        assert_eq!(found[0].root, specialized);
        assert_eq!(found[0].tensors, 1);
    }

    /// Writes a servable artifact under the name a stamped import gives it.
    fn serving(dir: &Path, slug: &str, sku: &str, backend: &str) -> std::path::PathBuf {
        let stamp = checkpoint::serving::Stamp::of(backend, sku);
        let path = dir.join(checkpoint::serving::Name::of(&stamp, slug).render());
        let mut writer = Writer::create_serving(&path, &Default::default(), stamp).unwrap();
        writer.add_tensor(&decl("w"), &vec![7u8; 32_000]).unwrap();
        writer.finish().unwrap();
        path
    }

    /// **ONE DIRECTORY, THREE ARTIFACTS, THREE ENTRIES.**
    ///
    /// The bug this file's whole naming scheme existed to make impossible and
    /// the scan brought back anyway: `<slug>.<sku>.<backend>.zt` is spelled
    /// out so a model imported for three shells can live in one directory,
    /// and the scan asked for the directory's *single* artifact — which a
    /// directory with three of them does not have. `pie model list` printed
    /// "(none)" over 200 GB of artifacts, and `pie model info <slug>` said
    /// there was no such artifact.
    #[test]
    fn a_directory_of_three_backends_is_three_entries() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("gemma");
        let sku = "gemma4-e4b-bf16-kv-bf16";
        for backend in ["cuda", "vulkan", "wgpu"] {
            serving(&model, "gemma", sku, backend);
        }

        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 3, "three artifacts, three entries");
        for entry in &found {
            assert_eq!(entry.name, "gemma", "every one is an artifact of gemma");
            assert_eq!(entry.siblings, 3);
            assert_eq!(entry.sku.as_deref(), Some(sku), "read off the stamp");
            assert_eq!(entry.tensors, 1);
        }
        // And each addresses itself, because a listing of three identical
        // names names nothing.
        let addresses: Vec<&str> = found.iter().map(Entry::address).collect();
        assert_eq!(
            addresses,
            [
                "gemma.gemma4-e4b-bf16-kv-bf16.cuda",
                "gemma.gemma4-e4b-bf16-kv-bf16.vulkan",
                "gemma.gemma4-e4b-bf16-kv-bf16.wgpu",
            ]
        );
        let backends: Vec<&str> = found
            .iter()
            .filter_map(|entry| entry.backend.as_deref())
            .collect();
        assert_eq!(backends, ["cuda", "vulkan", "wgpu"]);
    }

    /// A lone artifact keeps addressing itself by its model name.
    ///
    /// The common case, and the one every config on every box already
    /// states: `[model] model = "<name>"`. The long name is what siblings
    /// cost, not what an import costs.
    #[test]
    fn a_lone_artifact_is_still_addressed_by_its_model_name() {
        let root = tempfile::tempdir().unwrap();
        serving(
            &root.path().join("gemma"),
            "gemma",
            "gemma4-e4b-bf16-kv-bf16",
            "cuda",
        );
        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].address(), "gemma");
        assert_eq!(
            found[0].qualified.as_deref(),
            Some("gemma.gemma4-e4b-bf16-kv-bf16.cuda"),
            "the file still names itself, whether or not anything needs it to"
        );
    }

    /// The resolution rule, in the three answers it has.
    ///
    /// `find` is what `pie model info` and `pie model remove` ask, and
    /// removing the wrong one of three artifacts is not a mistake an operator
    /// can undo without a re-import — so an ambiguous name is answered with
    /// the candidates and nothing else.
    #[test]
    fn a_name_resolves_exactly_or_names_what_it_could_not_choose_between() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("gemma");
        let sku = "gemma4-e4b-bf16-kv-bf16";
        for backend in ["cuda", "vulkan"] {
            serving(&model, "gemma", sku, backend);
        }
        serving(
            &root.path().join("qwen"),
            "qwen",
            "qwen3-0-6b-bf16-kv-bf16",
            "cuda",
        );

        // The real rule, over the real scan.
        let resolve = |name: &str| resolve_in(entries_in(root.path()).unwrap(), name);

        // A model name with one artifact under it resolves.
        let Resolved::One(qwen) = resolve("qwen") else {
            panic!("one artifact, one answer");
        };
        assert_eq!(qwen.backend.as_deref(), Some("cuda"));

        // A model name with two under it names both rather than picking.
        let Resolved::Ambiguous(candidates) = resolve("gemma") else {
            panic!("two artifacts of one model is not a pick");
        };
        assert_eq!(
            candidates,
            [
                "gemma.gemma4-e4b-bf16-kv-bf16.cuda",
                "gemma.gemma4-e4b-bf16-kv-bf16.vulkan"
            ]
        );

        // And either candidate, typed back in full, resolves to itself.
        let Resolved::One(vulkan) = resolve("gemma.gemma4-e4b-bf16-kv-bf16.vulkan") else {
            panic!("a fully specified name names one file");
        };
        assert_eq!(vulkan.backend.as_deref(), Some("vulkan"));

        assert!(matches!(resolve("llama"), Resolved::Missing));
    }
}

//! The one place in the loader that opens a checkpoint: file discovery,
//! header parsing, and the `config.json` read. Everything below it is pure.
//!
//! Parity with the C++ path it replaced is intentional: file discovery
//! mirrors `discover_safetensors_manifest`, and every checkpoint tensor is
//! emitted as [`crate::types::Encoding::Raw`] with the storage dtype.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::file::zt;
use crate::file::{Attributes, Metadata, TokenizerTables};
use crate::error::Error;

/// Discover the safetensors shard files for a snapshot directory, matching
/// the C++ `discover_safetensors_manifest` (`SingleFile` preference).
///
/// Returns shard paths in C++ loader order: a lone `model.safetensors`,
/// else sorted unique shard names from `model.safetensors.index.json`'s
/// `weight_map`.
pub fn discover_safetensors_files(snapshot_dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let single = snapshot_dir.join("model.safetensors");
    let index = snapshot_dir.join("model.safetensors.index.json");

    // SingleFile preference: a lone `model.safetensors` wins even with an
    // index present.
    if single.is_file() {
        return Ok(vec![single]);
    }

    if index.is_file() {
        let text = std::fs::read_to_string(&index)
            .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", index.display())))?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
            Error::Checkpoint(format!("{} is not valid JSON: {err}", index.display()))
        })?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                Error::Checkpoint(format!("{} missing 'weight_map'", index.display()))
            })?;
        // Unique shard names, sorted — a BTreeSet reproduces the C++ dedup+sort.
        let mut shard_names = BTreeSet::new();
        for shard in weight_map.values() {
            let shard = shard.as_str().ok_or_else(|| {
                Error::Checkpoint(format!(
                    "{} weight_map has a non-string shard",
                    index.display()
                ))
            })?;
            shard_names.insert(shard.to_string());
        }
        return Ok(shard_names
            .into_iter()
            .map(|s| snapshot_dir.join(s))
            .collect());
    }

    Err(Error::Checkpoint(format!(
        "no model.safetensors[.index.json] in {}",
        snapshot_dir.display()
    )))
}

/// The GGUF checkpoint files for a snapshot directory, in shard order.
///
/// A snapshot may hold several independent quantizations side by side
/// (e.g. Qwen ships q4_0, q4_k_m, q5_k_m) — reading them together would
/// splice different quantizations into one artifact, so the first file in
/// sorted order is what a bare directory means.
///
/// llama.cpp also *splits* one checkpoint across files, named
/// `<stem>-00001-of-00002.gguf`; only the first shard carries the
/// key-value block, so a split is recognized by filename alone.
///
/// An incomplete set is refused rather than imported: reading only one
/// shard yields a model with holes, not a smaller import.
fn discover_gguf_files(snapshot_dir: &Path) -> Result<Option<Vec<PathBuf>>, Error> {
    let named = snapshot_dir.join("model.gguf");
    if named.is_file() {
        return Ok(Some(vec![named]));
    }
    let Ok(entries) = std::fs::read_dir(snapshot_dir) else {
        return Ok(None);
    };
    let mut ggufs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    ggufs.sort();
    match ggufs.first() {
        Some(first) => gguf_shard_set(first).map(Some),
        None => Ok(None),
    }
}

/// Every shard of the split `path` belongs to, or `path` alone when it is
/// not a shard. Read off the names beside it (not the directory listing),
/// so an unrelated GGUF in the same directory is never drawn in.
fn gguf_shard_set(path: &Path) -> Result<Vec<PathBuf>, Error> {
    let Some((prefix, own, count)) = split_shard_name(path) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut shards = Vec::with_capacity(count as usize);
    for index in 1..=count {
        let shard = dir.join(format!("{prefix}-{index:05}-of-{count:05}.gguf"));
        if !shard.is_file() {
            return Err(Error::Checkpoint(format!(
                "{} is shard {own} of {count}, and shard {index} is not beside \
                 it ({} is missing); a split GGUF is one checkpoint, and \
                 importing the shards that happen to be present would write a \
                 model with holes",
                path.display(),
                shard.display()
            )));
        }
        shards.push(shard);
    }
    Ok(shards)
}

/// The `(prefix, index, count)` of a llama.cpp split shard name, or
/// `None` for a name that is not one. Matches `SPLIT_PATH_FORMAT`
/// (`%s-%05d-of-%05d.gguf`); width is a floor, not enforced.
fn split_shard_name(path: &Path) -> Option<(String, u32, u32)> {
    let stem = path.file_stem()?.to_str()?;
    let (head, count) = stem.rsplit_once("-of-")?;
    let (prefix, index) = head.rsplit_once('-')?;
    let count: u32 = count.parse().ok()?;
    let index: u32 = index.parse().ok()?;
    if count == 0 || index == 0 || index > count {
        return None;
    }
    Some((prefix.to_string(), index, count))
}

/// The single `.zt` checkpoint for a snapshot directory, if present.
///
/// Two spellings: `model.zt` is this reader's own name for a converted
/// snapshot; `archive.zt` is what `pie model import` writes
/// (`$PIE_HOME/models/<name>/archive.zt`). `model.zt` is tried first — a
/// directory holding both was hand-converted beside a store entry, and
/// the hand-made name is the more specific statement.
const ZT_NAMES: [&str; 2] = ["model.zt", "archive.zt"];

/// Which `.zt` in this directory is the artifact — fixed names first,
/// then the specialization import named for what it is.
///
/// Public because the store asks the same question
/// (`src/local/store.rs`, `crates/worker/src/weights.rs`); two
/// implementations of "which of these files is a root" is one too many.
pub fn discover_zt_file(snapshot_dir: &Path) -> Option<PathBuf> {
    if snapshot_dir.is_file()
        && snapshot_dir
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("zt"))
    {
        return Some(snapshot_dir.to_path_buf());
    }
    if let Some(named) = ZT_NAMES
        .iter()
        .map(|name| snapshot_dir.join(name))
        .find(|named| named.is_file())
    {
        return Some(named);
    }
    // Then the specialized names: `pie model import` names a servable
    // artifact `<slug>.<sku>.<backend>.zt`, since one
    // model at two quantizations or shells needs two files in one
    // directory. Exactly one match, or none — coexistence is the point,
    // so ambiguity is refused rather than guessed at.
    let mut found = specialized_zt_files(snapshot_dir);
    (found.len() == 1).then(|| found.remove(0))
}

/// Every `*.zt` in `snapshot_dir` that is not one of [`ZT_NAMES`], sorted.
///
/// Split out from [`discover_zt_file`] so the ambiguous case (two
/// specializations present) can be named rather than fallen through to a
/// misleading "no model.safetensors".
fn specialized_zt_files(snapshot_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(snapshot_dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("zt"))
        })
        .collect();
    found.sort();
    found
}

/// Parse a checkpoint's headers into a [`Metadata`]. Only headers are
/// read; bulk tensor bytes are never mapped.
///
/// Every format is read through [`zt`], which projects safetensors, GGUF,
/// `.npz`, `.pt`, `.h5` and `.onnx` into one object model. This module is
/// about layout — which files make up one checkpoint — not format.
///
/// Tried in the order a snapshot is likely to hold: `.zt` (what `pie
/// model import` writes), else HF safetensors, else GGUF.
/// The bytes of the metadata object named `path`, or `None` if the
/// checkpoint has no such object.
///
/// Lives here (not on [`Metadata`]) because this module is where the
/// filesystem is allowed to exist; resolving the object and reading its
/// span directly would reimplement addressing the loader already does.
pub fn read_meta(metadata: &Metadata, path: &str) -> Result<Option<Vec<u8>>, Error> {
    use std::io::{Read, Seek, SeekFrom};

    let Some(object) = metadata.meta_object(path) else {
        return Ok(None);
    };
    let file = metadata
        .files
        .iter()
        .find(|file| file.id == object.file_id)
        .ok_or_else(|| {
            Error::Checkpoint(format!(
                "{} points at a file the checkpoint lacks",
                object.name
            ))
        })?;
    let mut handle = std::fs::File::open(&file.path)
        .map_err(|err| Error::Checkpoint(format!("cannot open {}: {err}", file.path)))?;
    handle
        .seek(SeekFrom::Start(object.file_offset))
        .map_err(|err| Error::Checkpoint(format!("cannot seek in {}: {err}", file.path)))?;
    let mut bytes = vec![0u8; object.span_bytes as usize];
    handle.read_exact(&mut bytes).map_err(|err| {
        Error::Checkpoint(format!(
            "cannot read {} from {}: {err}",
            object.name, file.path
        ))
    })?;
    Ok(Some(bytes))
}

/// Which files hold this checkpoint, and whether the format names them
/// as a set. Kept as a distinction (not flattened to a list) because a
/// lone safetensors file is still a member of a set, and `index_all`
/// refuses a name appearing in two files where `index` has no second
/// file to refuse against.
enum Discovered {
    One(PathBuf),
    Set(Vec<PathBuf>),
}

fn discover(snapshot_dir: &Path) -> Result<Discovered, Error> {
    if let Some(zt) = discover_zt_file(snapshot_dir) {
        return Ok(Discovered::One(zt));
    }
    // Two specializations is the case the naming exists for: `pie model
    // import` writes `<slug>.<sku>.<backend>.zt` so two
    // artifacts of one model can share a directory. Ambiguity is refused
    // rather than answered with "no model.safetensors".
    let specialized = specialized_zt_files(snapshot_dir);
    if specialized.len() > 1 {
        return Err(Error::Checkpoint(format!(
            "{} holds {} serving artifacts and this load names none of them: {}. \
             Two artifacts of one model is what the naming is for — a different \
             backend, degree or precision — so `[model] model` has to name the \
             file rather than the directory.",
            snapshot_dir.display(),
            specialized.len(),
            specialized
                .iter()
                .filter_map(|path| path.file_name())
                .map(|name| format!("`{}`", name.to_string_lossy()))
                .collect::<Vec<_>>()
                .join(", "),
        )));
    }
    if snapshot_dir.is_file() {
        // A file names itself, except a split GGUF: the suffix is the only
        // thing that knows shard one names the whole checkpoint.
        return Ok(one_or_set(gguf_shard_set(snapshot_dir)?));
    }
    // Safetensors takes precedence — it is the canonical HF snapshot format and
    // the C++ loader opens it first.
    match discover_safetensors_files(snapshot_dir) {
        Ok(files) => Ok(Discovered::Set(files)),
        Err(safetensors_err) => match discover_gguf_files(snapshot_dir)? {
            Some(files) => Ok(one_or_set(files)),
            None => Err(safetensors_err),
        },
    }
}

/// A lone file stays `One`; more than one is a `Set` — not cosmetic:
/// `One` reads through `index`, `Set` through `index_all`, which refuses
/// a tensor name appearing in two files.
fn one_or_set(mut files: Vec<PathBuf>) -> Discovered {
    if files.len() == 1 {
        Discovered::One(files.remove(0))
    } else {
        Discovered::Set(files)
    }
}

pub fn parse_metadata(snapshot_dir: &Path) -> Result<Metadata, Error> {
    match discover(snapshot_dir)? {
        Discovered::One(path) => zt::parse(&path),
        Discovered::Set(paths) => zt::parse_files(&paths),
    }
}

/// The objects [`parse_metadata`] split into planes, as `(object, plane
/// names in canonical order)`. Empty for a source whose objects are all one
/// plane, which is every format but `.zt`.
pub fn parse_groups(snapshot_dir: &Path) -> Result<Vec<(String, Vec<String>)>, Error> {
    let paths = match discover(snapshot_dir)? {
        Discovered::One(path) => vec![path],
        Discovered::Set(paths) => paths,
    };
    let mut groups = Vec::new();
    for path in &paths {
        groups.extend(zt::parse_groups(path)?);
    }
    Ok(groups)
}

/// What this checkpoint says about itself, as opposed to its tensors.
///
/// A separate call, not a field on [`Metadata`]: almost nobody asks it,
/// and a GGUF's key-values are read only by conversion. Reads a header,
/// not a payload — empty for safetensors, the whole key-value block for GGUF.
///
/// # Errors
///
/// The snapshot holds no checkpoint this loader can open.
pub fn parse_attributes(snapshot_dir: &Path) -> Result<Attributes, Error> {
    match discover(snapshot_dir)? {
        Discovered::One(path) => zt::parse_attributes(&path),
        Discovered::Set(paths) => zt::parse_attributes_files(&paths),
    }
}

/// The `tokenizer.ggml.*` tables a GGUF snapshot carries, whole.
///
/// One file only (the first), not the whole set: only shard one of a
/// split GGUF carries a key-value block, so a later shard has no
/// vocabulary to merge.
///
/// # Errors
///
/// The snapshot holds no checkpoint this loader can open.
pub fn parse_tokenizer(snapshot_dir: &Path) -> Result<TokenizerTables, Error> {
    let path = match discover(snapshot_dir)? {
        Discovered::One(path) => path,
        Discovered::Set(mut paths) => {
            paths.sort();
            paths.remove(0)
        }
    };
    zt::parse_tokenizer_tables(&path)
}

/// Check that the files a compiled plan declares are on disk, at the
/// size the plan read them at.
///
/// Lives here (not beside `plan::compile`) because compiling is a pure
/// function of the metadata and the contract; touching the filesystem is
/// this module's job.
///
/// # Errors
///
/// A file the plan declares is missing, unreadable, or a different size.
pub fn verify_declared_files(
    plan: &crate::plan::LoadPlan,
    snapshot_dir: &Path,
) -> Result<(), Error> {
    for file in &plan.files {
        let path = snapshot_dir.join(&file.path);
        match std::fs::metadata(&path) {
            Ok(meta) if meta.len() == file.size_bytes => {}
            Ok(meta) => {
                return Err(Error::Checkpoint(format!(
                    "{} is {} bytes on disk, the plan declares {}",
                    path.display(),
                    meta.len(),
                    file.size_bytes
                )));
            }
            Err(err) => {
                return Err(Error::Checkpoint(format!("{}: {err}", path.display())));
            }
        }
    }
    Ok(())
}


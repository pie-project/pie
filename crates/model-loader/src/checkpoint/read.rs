//! The one place in the loader that opens a checkpoint.
//!
//! `compile` is a function of three values — what is in the file, what the
//! caller wants out, and what the device can do (architecture.md §12 row 12) —
//! and the first of those is a [`CheckpointMetadata`], not a directory. This
//! module is what turns the directory into the value: file discovery, header
//! parsing, and the `config.json` read. Everything below it is pure, and
//! `standalone.rs` is the test that keeps it that way.
//!
//! Parity with the C++ path it replaced is intentional and load-bearing: file
//! discovery mirrors `discover_safetensors_manifest` (`SingleFile` preference:
//! prefer `model.safetensors`, else the sharded `model.safetensors.index.json`
//! `weight_map`, else the single file), and every checkpoint tensor is emitted
//! as [`crate::types::Encoding::Raw`] with the storage dtype — MXFP4
//! recognition is by *name*, inside the compiler.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::checkpoint::zt;
use crate::checkpoint::{Attributes, CheckpointMetadata, TokenizerTables};
use crate::error::Error;

/// Discover the safetensors shard files for a snapshot directory, matching the
/// C++ `discover_safetensors_manifest` with a `SingleFile` layout preference.
///
/// Returns the shard paths in the order the C++ loader assigns file ids:
/// a lone `model.safetensors`, otherwise the sorted unique shard names from
/// `model.safetensors.index.json`'s `weight_map`.
pub fn discover_safetensors_files(snapshot_dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let single = snapshot_dir.join("model.safetensors");
    let index = snapshot_dir.join("model.safetensors.index.json");

    // SingleFile preference: a lone `model.safetensors` wins even when an index
    // is also present.
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
/// A snapshot holding GGUFs is not one checkpoint per directory. Qwen's
/// `Qwen2.5-0.5B-Instruct-GGUF` ships `q4_0`, `q4_k_m` and `q5_k_m` side by
/// side, all of the same model — reading them together would splice three
/// different quantizations into one artifact. So the first file in sorted
/// order is still what a bare directory means, exactly as before.
///
/// What is new is that llama.cpp also *splits* one checkpoint across files,
/// and says so in the name: `<stem>-00001-of-00002.gguf`. Only the first
/// shard carries the key-value block — the second holds `split.no`,
/// `split.count`, `split.tensors.count` and nothing else, no architecture and
/// no tokenizer — so a split is recognizable by its filename and by nothing
/// inside it. That is why the pattern is the test here rather than a header
/// key: a shard whose siblings are missing cannot introduce itself.
///
/// An incomplete set is refused rather than imported. Taking the first file
/// of a split is not a smaller import, it is a model with holes: reading only
/// shard one of `Qwen2.5-7B-Instruct-GGUF` yields 293 of its 339 tensors,
/// which is every layer and no final norm — an artifact that gets written,
/// fails to identify against any row, and reports it as `missing norm`.
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

/// Every shard of the split `path` belongs to, or `path` alone when it is not
/// a shard.
///
/// The set is read off the names beside it rather than off the directory
/// listing, so an unrelated GGUF in the same directory — a second
/// quantization of the same model — is never drawn in.
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

/// The `(prefix, index, count)` of a llama.cpp split shard name, or `None` for
/// a name that is not one.
///
/// llama.cpp writes `SPLIT_PATH_FORMAT` as `%s-%05d-of-%05d.gguf` and finds
/// the siblings by the same spelling, so matching it is not a heuristic — it
/// is the convention both sides agree on. The width is not enforced, because
/// a five-digit format is a floor rather than a ceiling: a hundred-thousandth
/// shard would spell six.
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
fn discover_zt_file(snapshot_dir: &Path) -> Option<PathBuf> {
    if snapshot_dir.is_file()
        && snapshot_dir
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("zt"))
    {
        return Some(snapshot_dir.to_path_buf());
    }
    let named = snapshot_dir.join("model.zt");
    named.is_file().then_some(named)
}

/// Parse a checkpoint's headers into a [`CheckpointMetadata`]. Only headers
/// are read; bulk tensor bytes are never mapped.
///
/// Every format is read the same way, through [`zt`]: `ztensor-compat`
/// projects safetensors, GGUF, `.npz`, `.pt`, `.h5` and `.onnx` into one
/// object model, and `zt` translates that model into the loader's. This module
/// is what remains of format knowledge here, and it is not about *formats* --
/// it is about **layout**: which files on disk make up one checkpoint. That is
/// a question no format answers about itself. A safetensors snapshot states it
/// in `model.safetensors.index.json`, a convention beside the format; GGUF and
/// `.zt` are single-file and state it by being one file.
///
/// The order below is the order a snapshot is likely to hold: a `.zt`
/// artifact (what `pie model import` writes), else the canonical HF
/// safetensors layout, else GGUF.
/// The bytes of the metadata object named `path`, or `None` if the checkpoint
/// has no such object.
///
/// Lives here rather than on [`CheckpointMetadata`] because this module is
/// where the filesystem is allowed to exist: everything below the reader takes
/// values, and `standalone.rs` enforces it. Consumers that instead resolved
/// the object to a file, seeked to its offset and read its span were
/// reimplementing addressing the loader already does — and would keep working
/// while silently disagreeing with it if, say, a sharded artifact put the
/// object somewhere other than the root.
pub fn read_meta(metadata: &CheckpointMetadata, path: &str) -> Result<Option<Vec<u8>>, Error> {
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

/// Which files hold this checkpoint, and whether the format names them as a
/// set.
///
/// Extracted so the two questions asked of a snapshot — where are the tensors,
/// and what does the file say about itself — find the files by one rule
/// instead of two that can drift apart. The distinction is kept rather than
/// flattened to a list because a lone safetensors file is still a member of a
/// set, and `index_all` refuses a name in two files where `index` has no
/// second file to refuse against.
enum Discovered {
    One(PathBuf),
    Set(Vec<PathBuf>),
}

fn discover(snapshot_dir: &Path) -> Result<Discovered, Error> {
    if let Some(zt) = discover_zt_file(snapshot_dir) {
        return Ok(Discovered::One(zt));
    }
    if snapshot_dir.is_file() {
        // A file names itself; detection is the projections' job, not a
        // suffix's. The exception is a split GGUF, where the suffix is the
        // only thing that knows: naming shard one names the whole checkpoint,
        // and the shards after it carry no header to say so.
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

/// A lone file stays `One`; more than one is a `Set`.
///
/// Not a cosmetic distinction: `One` reads through `index`, `Set` through
/// `index_all`, which refuses a tensor name that appears in two files.
fn one_or_set(mut files: Vec<PathBuf>) -> Discovered {
    if files.len() == 1 {
        Discovered::One(files.remove(0))
    } else {
        Discovered::Set(files)
    }
}

pub fn parse_checkpoint_metadata(snapshot_dir: &Path) -> Result<CheckpointMetadata, Error> {
    match discover(snapshot_dir)? {
        Discovered::One(path) => zt::parse_checkpoint(&path),
        Discovered::Set(paths) => zt::parse_checkpoint_files(&paths),
    }
}

/// What this checkpoint says about ITSELF, as opposed to about its tensors.
///
/// A separate call and not a field on [`CheckpointMetadata`], because it
/// answers a separate question and almost nobody asks it. `CheckpointMetadata`
/// is the addressing information a plan is compiled from — which bytes live
/// where — and every caller needs all of it. A GGUF's key-values are a
/// description of the model that only conversion reads, and threading them
/// through would have put an unused field in the hundred-odd literals that
/// build this type in tests.
///
/// Reads a header, not a payload. For safetensors that header is a JSON
/// preamble and this comes back empty; for GGUF it is where the whole
/// key-value block lives.
///
/// # Errors
///
/// The snapshot holds no checkpoint this loader can open.
pub fn parse_checkpoint_attributes(snapshot_dir: &Path) -> Result<Attributes, Error> {
    match discover(snapshot_dir)? {
        Discovered::One(path) => zt::parse_attributes(&path),
        Discovered::Set(paths) => zt::parse_attributes_files(&paths),
    }
}

/// The `tokenizer.ggml.*` tables a GGUF snapshot carries, whole.
///
/// One file only, where [`parse_checkpoint_attributes`] takes a set — and the
/// first, not any. Only shard one of a split GGUF carries a key-value block;
/// every shard after it holds `split.no`, `split.count` and
/// `split.tensors.count`, so there is nothing to merge and asking a later
/// shard would find no vocabulary at all. Measured on
/// `qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf`: three keys, against 26 in
/// shard one.
///
/// # Errors
///
/// The snapshot holds no checkpoint this loader can open.
pub fn parse_checkpoint_tokenizer(snapshot_dir: &Path) -> Result<TokenizerTables, Error> {
    let path = match discover(snapshot_dir)? {
        Discovered::One(path) => path,
        Discovered::Set(mut paths) => {
            paths.sort();
            paths.remove(0)
        }
    };
    zt::parse_tokenizer_tables(&path)
}

/// Check that the files a compiled plan declares are on disk, at the size the
/// plan read them at.
///
/// Lives here rather than beside `plan::compile` because compiling is a pure
/// function of the metadata and the contract — `tests/standalone.rs` holds
/// that as a property, and reaching for `std::fs` in `plan.rs` is exactly what
/// it refuses. Touching the filesystem is this module's job, so the check that
/// touches it is this module's too.
///
/// Both engines carried this block, bit for bit, after their own
/// `plan::compile`. It is not an engine's question: the plan names the files
/// and states their sizes, so a snapshot that moved under a plan compiled
/// against it is a fact about the checkpoint. Catching it here is the
/// difference between a named file with two byte counts and a fault in the
/// middle of a 39 GB read.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("load_planner_inproc_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn discovers_single_file_over_index() {
        let dir = tmpdir("single");
        write(&dir, "model.safetensors", "x");
        write(&dir, "model.safetensors.index.json", "{}");
        let files = discover_safetensors_files(&dir).unwrap();
        assert_eq!(files, vec![dir.join("model.safetensors")]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discovers_sorted_unique_shards_from_index() {
        let dir = tmpdir("sharded");
        write(
            &dir,
            "model.safetensors.index.json",
            r#"{"weight_map":{"a":"model-00002.safetensors","b":"model-00001.safetensors","c":"model-00001.safetensors"}}"#,
        );
        let files = discover_safetensors_files(&dir).unwrap();
        assert_eq!(
            files,
            vec![
                dir.join("model-00001.safetensors"),
                dir.join("model-00002.safetensors"),
            ]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn direct_gguf_paths_are_discovered() {
        let dir = tmpdir("direct_gguf");
        let path = dir.join("model.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        assert_eq!(gguf_shard_set(&path).unwrap(), vec![path.clone()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Only llama.cpp's own spelling counts, because only llama.cpp's own
    /// writer produces a split. The near-misses matter more than the hit: a
    /// model whose name merely contains `-of-` must not be mistaken for a
    /// shard, or discovery would start looking for siblings that were never
    /// written.
    #[test]
    fn a_split_is_recognized_only_by_llama_cpps_own_spelling() {
        let split = Path::new("/m/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf");
        assert_eq!(
            split_shard_name(split),
            Some(("qwen2.5-7b-instruct-q4_0".to_string(), 1, 2))
        );
        for lookalike in [
            "/m/model.gguf",
            "/m/best-of-breed.gguf",
            "/m/model-1-of-.gguf",
            "/m/model--of-2.gguf",
            "/m/model-00003-of-00002.gguf",
            "/m/model-00000-of-00002.gguf",
        ] {
            assert_eq!(
                split_shard_name(Path::new(lookalike)),
                None,
                "{lookalike} is not a shard name"
            );
        }
    }

    /// The regression this whole path exists for. Reading shard one alone
    /// yielded 293 of a 7B model's 339 tensors — every layer and no final
    /// norm — and wrote a 12.7 GiB artifact that matched no catalog row.
    /// Silence is the failure mode, so the gap has to be an error.
    #[test]
    fn a_split_gguf_is_gathered_whole_or_refused() {
        let dir = tmpdir("gguf_split");
        let first = dir.join("m-00001-of-00002.gguf");
        let second = dir.join("m-00002-of-00002.gguf");
        std::fs::write(&first, b"GGUF").unwrap();

        let refusal = gguf_shard_set(&first).unwrap_err().to_string();
        assert!(refusal.contains("m-00002-of-00002.gguf"), "{refusal}");

        std::fs::write(&second, b"GGUF").unwrap();
        // Named by either member, the checkpoint is the same set, in order.
        //
        // In order is not cosmetic. Only shard one carries a key-value block,
        // and `zt::parse_attributes_files` reads the set's first member that
        // has one -- it sorts for itself now, so this is no longer the only
        // thing standing between a reordering here and a checkpoint that
        // cannot say what architecture it is.
        for named in [&first, &second] {
            assert_eq!(
                gguf_shard_set(named).unwrap(),
                vec![first.clone(), second.clone()]
            );
        }
        assert_eq!(
            discover_gguf_files(&dir).unwrap(),
            Some(vec![first.clone(), second.clone()])
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A HF GGUF snapshot holds several *independent* quantizations of one
    /// model — `Qwen2.5-0.5B-Instruct-GGUF` ships q4_0, q4_k_m and q5_k_m
    /// side by side. Gathering every `.gguf` in the directory would splice
    /// three different models together, so a directory still means one file
    /// unless the names say otherwise.
    #[test]
    fn independent_quantizations_in_one_directory_are_not_a_split() {
        let dir = tmpdir("gguf_quants");
        for quant in ["q4_0", "q4_k_m", "q5_k_m"] {
            std::fs::write(dir.join(format!("qwen2.5-0.5b-{quant}.gguf")), b"GGUF").unwrap();
        }
        assert_eq!(
            discover_gguf_files(&dir).unwrap(),
            Some(vec![dir.join("qwen2.5-0.5b-q4_0.gguf")])
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

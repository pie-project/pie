//! Rewriting a checkpoint so a streamer can map its weights where they lie.
//!
//! A weight that is streamed is not copied into device memory; the device
//! reads it out of the checkpoint file, through a mapping. A mapping begins on
//! a page, so a streamed tensor has to begin on a page too -- and it has to be
//! a page of its *own*, because two tensors sharing one means faulting either
//! drags in the other, and the whole point of streaming a mixture-of-experts is
//! to fault in the experts that routed and no others. Paging granularity has to
//! equal tensor granularity or it buys nothing.
//!
//! Published checkpoints pack their tensors back to back, so essentially none
//! of them begin on a page: of the 400 expert tensors in GPT-OSS-20B, zero are
//! even 32-byte aligned. This module is the one-time rewrite that fixes that,
//! and the thing to understand about it is how little it changes. Every tensor
//! keeps its name, its dtype, its shape and its bytes. What changes is where in
//! the file those bytes sit, which is not something a checkpoint's *meaning*
//! depends on.
//!
//! # Why the padding is a tensor
//!
//! The obvious way to move a tensor onto a page is to leave a gap in front of
//! it. safetensors can describe that -- every tensor carries explicit
//! `data_offsets` -- and this loader accepts it, because it validates each
//! tensor against the file and never against its neighbour.
//!
//! The reference implementation does not. `safetensors` 0.8 refuses such a file
//! outright, `invalid offset for tensor b`, because it requires every tensor to
//! begin exactly where the last one ended. A file only this repository can read
//! would be a new format wearing a familiar extension, and then every tool that
//! reads checkpoints would need teaching about it.
//!
//! So the padding is named instead. The run of bytes ahead of a tensor that
//! needs a page is *itself a tensor* -- `U8`, under [`PAD_PREFIX`] -- and the
//! file stays contiguous. Both readers accept the result, and what a model
//! contract does about the fillers is that it never mentions them, so nothing
//! downstream has to know they exist.
//!
//! # What this module does not decide
//!
//! Which tensors to align. Alignment costs a page apiece, which rounds off
//! against a 132 MB expert bank and is ruinous for a 180 KB bias, so aligning
//! everything is the one clearly wrong policy. The caller passes the set, and
//! the caller derives it from the compiled plan's groups -- the tensors a
//! contract has declared interchangeable, which is exactly the set a driver can
//! stream. That is a fact the contract states, not a guess about names.

use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::Error;

/// The alignment to write. 16 KiB is Apple silicon's page; x86-64 Linux uses 4
/// KiB, which divides it, so one artifact is streamable on both. Padding to the
/// larger costs at most 12 KiB per aligned tensor, against the megabytes that
/// make a tensor worth streaming in the first place.
pub const STREAM_PAGE_BYTES: u64 = 16384;

/// Names the filler tensors take. Chosen to be something no checkpoint
/// publishes and no contract claims; the leading and trailing underscores match
/// safetensors' own `__metadata__` convention for "not a weight".
pub const PAD_PREFIX: &str = "__pie_pad__.";

/// Key under `__metadata__` holding the run of characters that pushes the data
/// section itself onto a page. A tensor's position in the *file* is the data
/// section's start plus its offset within it, so both have to be aligned.
const PAD_METADATA_KEY: &str = "pie_align_pad";

/// Key under `__metadata__` recording what this file was aligned to, so a
/// reader can tell an aligned checkpoint from an ordinary one without measuring
/// every tensor.
pub const ALIGN_METADATA_KEY: &str = "pie_align_bytes";

const LEN_PREFIX: u64 = 8;

/// What aligning one file changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileAlignment {
    pub path: PathBuf,
    /// Tensors moved onto a page.
    pub aligned: usize,
    /// Filler tensors introduced.
    pub fillers: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
}

impl FileAlignment {
    /// What alignment cost, which is entirely padding.
    pub fn overhead_bytes(&self) -> u64 {
        self.bytes_after.saturating_sub(self.bytes_before)
    }
}

/// What aligning a checkpoint changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Alignment {
    pub files: Vec<FileAlignment>,
    /// Names the caller asked to align that the checkpoint does not contain.
    ///
    /// Reported rather than ignored: a set derived from a plan should name only
    /// tensors the plan resolved, so anything here means the caller and the
    /// checkpoint disagree about what is in it.
    pub missing: Vec<String>,
}

impl Alignment {
    pub fn aligned(&self) -> usize {
        self.files.iter().map(|f| f.aligned).sum()
    }

    pub fn overhead_bytes(&self) -> u64 {
        self.files.iter().map(FileAlignment::overhead_bytes).sum()
    }
}

/// One tensor's entry in the header, kept as written.
///
/// The spec is carried as opaque JSON rather than as a parsed dtype and shape
/// because the parse is lossy: safetensors' `F4_E2M1` and `F8_E8M0` both read
/// as `U8` here, so a header rebuilt from what this crate understands would
/// silently relabel an MXFP4 checkpoint. Only `data_offsets` is ever rewritten,
/// and every other field -- including ones this crate has never heard of --
/// survives byte for byte.
struct Entry {
    name: String,
    spec: serde_json::Map<String, serde_json::Value>,
    begin: u64,
    end: u64,
}

impl Entry {
    fn bytes(&self) -> u64 {
        self.end - self.begin
    }
}

/// Rewrite `src`'s safetensors so every tensor named in `align` begins on a
/// `page` boundary, writing the result under `dst`.
///
/// `dst` must not exist: a half-written checkpoint that looks like a whole one
/// is the failure mode worth designing out, and the tool is not in a position
/// to know whether an existing directory is a previous run of itself or
/// something the user wanted.
pub fn align_checkpoint(
    src: &Path,
    dst: &Path,
    align: &BTreeSet<String>,
    page: u64,
) -> Result<Alignment, Error> {
    if page == 0 || !page.is_power_of_two() {
        return Err(Error::Checkpoint(format!(
            "alignment must be a power of two, got {page}"
        )));
    }
    if dst.exists() {
        return Err(Error::Checkpoint(format!(
            "{} already exists; aligning writes a fresh checkpoint",
            dst.display()
        )));
    }
    let shards = crate::checkpoint::read::discover_safetensors_files(src)?;

    std::fs::create_dir_all(dst)
        .map_err(|err| Error::Checkpoint(format!("cannot create {}: {err}", dst.display())))?;

    let mut report = Alignment::default();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for shard in &shards {
        let name = shard
            .file_name()
            .ok_or_else(|| Error::Checkpoint(format!("{} has no file name", shard.display())))?;
        let out = dst.join(name);
        report
            .files
            .push(align_file(shard, &out, align, page, &mut seen)?);
    }
    report.missing = align.difference(&seen).cloned().collect();

    copy_sidecars(src, dst, &shards)?;
    Ok(report)
}

/// Everything in the snapshot that is not a shard: config, tokenizer, the
/// shard index. Copied so the output is a checkpoint someone can load, not a
/// pile of weights that needs the original directory beside it.
fn copy_sidecars(src: &Path, dst: &Path, shards: &[PathBuf]) -> Result<(), Error> {
    let shard_names: BTreeSet<_> = shards.iter().filter_map(|p| p.file_name()).collect();
    let entries = std::fs::read_dir(src)
        .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", src.display())))?;
    for entry in entries {
        let entry = entry
            .map_err(|err| Error::Checkpoint(format!("cannot walk {}: {err}", src.display())))?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let name = entry.file_name();
        if shard_names.contains(&name.as_os_str()) {
            continue;
        }
        std::fs::copy(entry.path(), dst.join(&name)).map_err(|err| {
            Error::Checkpoint(format!("cannot copy {}: {err}", entry.path().display()))
        })?;
    }
    Ok(())
}

fn align_file(
    src: &Path,
    dst: &Path,
    align: &BTreeSet<String>,
    page: u64,
    seen: &mut BTreeSet<String>,
) -> Result<FileAlignment, Error> {
    let (prefix, bytes_before) =
        crate::checkpoint::safetensors::read_safetensors_header_prefix(src)?;
    let header_json = &prefix[LEN_PREFIX as usize..];
    let value: serde_json::Value = serde_json::from_slice(header_json).map_err(|err| {
        Error::Checkpoint(format!(
            "safetensors header of {} is not valid JSON: {err}",
            src.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        Error::Checkpoint(format!(
            "safetensors header of {} is not an object",
            src.display()
        ))
    })?;

    let mut metadata = object
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    metadata.remove(PAD_METADATA_KEY);

    let mut entries = Vec::with_capacity(object.len());
    for (name, spec) in object {
        if name == "__metadata__" {
            continue;
        }
        if name.starts_with(PAD_PREFIX) {
            return Err(Error::Checkpoint(format!(
                "{} already contains filler tensor {name}; it has been aligned already",
                src.display()
            )));
        }
        let spec = spec.as_object().cloned().ok_or_else(|| {
            Error::Checkpoint(format!("safetensors entry {name} is not an object"))
        })?;
        let offsets = spec
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                Error::Checkpoint(format!("safetensors entry {name} lacks data_offsets"))
            })?;
        let read = |i: usize| -> Result<u64, Error> {
            offsets
                .get(i)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    Error::Checkpoint(format!("safetensors entry {name} has bad data_offsets"))
                })
        };
        let (begin, end) = (read(0)?, read(1)?);
        if end < begin {
            return Err(Error::Checkpoint(format!(
                "safetensors entry {name} ends at {end} before it begins at {begin}"
            )));
        }
        entries.push(Entry {
            name: name.clone(),
            spec,
            begin,
            end,
        });
    }

    // Laid out in the order the source already uses. Preserving it keeps the
    // rewrite a sequential read of the input, and keeps a diff of the two
    // headers to the offsets that had to move.
    entries.sort_by_key(|e| (e.begin, e.end));

    let mut placed: Vec<(Option<Entry>, u64, u64)> = Vec::new(); // entry (None = filler), begin, bytes
    let mut at = 0u64;
    let mut fillers = 0usize;
    let mut aligned = 0usize;
    for entry in entries {
        let wanted = align.contains(&entry.name);
        if wanted {
            seen.insert(entry.name.clone());
            if !at.is_multiple_of(page) {
                let fill = page - (at % page);
                placed.push((None, at, fill));
                at += fill;
                fillers += 1;
            }
            aligned += 1;
        }
        let bytes = entry.bytes();
        placed.push((Some(entry), at, bytes));
        at += bytes;
    }
    let data_bytes = at;

    // The data section has to start on a page too, and the only lever is the
    // header's length. Padding is a string under `__metadata__`, so adding one
    // character adds exactly one byte to the header -- which makes the fixed
    // point a subtraction rather than a search.
    let mut header = build_header(&metadata, &placed, page, 0)?;
    let short = (page - ((LEN_PREFIX + header.len() as u64) % page)) % page;
    if short != 0 {
        header = build_header(&metadata, &placed, page, short as usize)?;
        // Growing the pad grew the header by exactly `short`, so this holds by
        // construction. It is asserted because if it ever stopped holding, the
        // symptom would be a checkpoint that loads correctly and streams
        // slowly, which is not a symptom anyone would trace back to here.
        debug_assert_eq!((LEN_PREFIX + header.len() as u64) % page, 0);
    }
    if !(LEN_PREFIX + header.len() as u64).is_multiple_of(page) {
        return Err(Error::Checkpoint(format!(
            "cannot pad {}'s header onto a {page}-byte boundary",
            src.display()
        )));
    }

    write_shard(src, dst, &header, &placed)?;

    Ok(FileAlignment {
        path: dst.to_path_buf(),
        aligned,
        fillers,
        bytes_before,
        bytes_after: LEN_PREFIX + header.len() as u64 + data_bytes,
    })
}

fn build_header(
    metadata: &serde_json::Map<String, serde_json::Value>,
    placed: &[(Option<Entry>, u64, u64)],
    page: u64,
    pad: usize,
) -> Result<Vec<u8>, Error> {
    let mut out = serde_json::Map::new();

    let mut metadata = metadata.clone();
    metadata.insert(
        ALIGN_METADATA_KEY.to_string(),
        serde_json::Value::String(page.to_string()),
    );
    // Always present, even empty. The caller sizes the pad by measuring a
    // header built with `pad = 0` and then rebuilding; that subtraction is only
    // exact if the second header differs from the first by the pad characters
    // alone, and a key that appears between the two measurements would add its
    // own name, quotes, colon and comma on top.
    metadata.insert(
        PAD_METADATA_KEY.to_string(),
        serde_json::Value::String("0".repeat(pad)),
    );
    out.insert("__metadata__".into(), serde_json::Value::Object(metadata));

    let mut filler = 0usize;
    for (entry, begin, bytes) in placed {
        let offsets = serde_json::json!([begin, begin + bytes]);
        match entry {
            Some(entry) => {
                let mut spec = entry.spec.clone();
                spec.insert("data_offsets".into(), offsets);
                out.insert(entry.name.clone(), serde_json::Value::Object(spec));
            }
            None => {
                out.insert(
                    format!("{PAD_PREFIX}{filler}"),
                    serde_json::json!({
                        "dtype": "U8",
                        "shape": [bytes],
                        "data_offsets": offsets,
                    }),
                );
                filler += 1;
            }
        }
    }
    let _ = page;
    serde_json::to_vec(&serde_json::Value::Object(out))
        .map_err(|err| Error::Checkpoint(format!("cannot serialize aligned header: {err}")))
}

fn write_shard(
    src: &Path,
    dst: &Path,
    header: &[u8],
    placed: &[(Option<Entry>, u64, u64)],
) -> Result<(), Error> {
    let mut input = std::fs::File::open(src)
        .map_err(|err| Error::Checkpoint(format!("cannot open {}: {err}", src.display())))?;
    let (prefix, _) = crate::checkpoint::safetensors::read_safetensors_header_prefix(src)?;
    let src_data_start = prefix.len() as u64;

    let out = std::fs::File::create(dst)
        .map_err(|err| Error::Checkpoint(format!("cannot create {}: {err}", dst.display())))?;
    let mut out = std::io::BufWriter::with_capacity(1 << 20, out);

    let io =
        |err: std::io::Error| Error::Checkpoint(format!("cannot write {}: {err}", dst.display()));
    out.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(io)?;
    out.write_all(header).map_err(io)?;

    let mut buf = vec![0u8; 1 << 20];
    for (entry, _, bytes) in placed {
        match entry {
            None => {
                let mut left = *bytes;
                let zeros = vec![0u8; std::cmp::min(left, 1 << 16) as usize];
                while left > 0 {
                    let take = std::cmp::min(left, zeros.len() as u64) as usize;
                    out.write_all(&zeros[..take]).map_err(io)?;
                    left -= take as u64;
                }
            }
            Some(entry) => {
                input
                    .seek(SeekFrom::Start(src_data_start + entry.begin))
                    .map_err(|err| {
                        Error::Checkpoint(format!("cannot seek {}: {err}", src.display()))
                    })?;
                let mut left = *bytes;
                while left > 0 {
                    let take = std::cmp::min(left, buf.len() as u64) as usize;
                    input.read_exact(&mut buf[..take]).map_err(|err| {
                        Error::Checkpoint(format!(
                            "cannot read {} of {}: {err}",
                            entry.name,
                            src.display()
                        ))
                    })?;
                    out.write_all(&buf[..take]).map_err(io)?;
                    left -= take as u64;
                }
            }
        }
    }
    // Buffered writes that fail on flush are the reason a 0-byte checkpoint can
    // look like a successful run, so the flush is checked rather than dropped.
    out.flush().map_err(io)?;
    out.into_inner()
        .map_err(|err| Error::Checkpoint(format!("cannot flush {}: {err}", dst.display())))?
        .sync_all()
        .map_err(io)?;
    Ok(())
}

/// How a group's instances sit in the checkpoint, which decides whether moving
/// tensors can align them.
///
/// A group says its instances are interchangeable; streaming holds one
/// instance's weights instead of another's at run time. Both storage shapes a
/// contract can express are groups, but only one of them can be aligned by a
/// rewrite that preserves the checkpoint's tensors:
///
/// * **Whole tensors.** Instance `i` reads its own tensor, named for `i`
///   (DeepSeek, Mixtral). Moving that tensor onto a page aligns the instance,
///   because the instance *is* the tensor.
/// * **A band inside one tensor.** Every instance reads a slice of one fused
///   bank, `[E, ...]` (GPT-OSS, Qwen). Moving the bank aligns instance 0 and no
///   other, since the rest begin at `base + i * stride` and the stride is
///   whatever the shape makes it -- in GPT-OSS-20B, 4147200 bytes, which no
///   page size divides.
///
/// The distinction is reported rather than papered over, because the failure it
/// prevents is silent: a banded checkpoint would align, load and stream
/// correctly, and fault in a whole bank to read one expert.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Streamable {
    /// Tensors a group instance reads in full. Alignable.
    pub whole: BTreeSet<String>,
    /// Fused banks, which a rewrite cannot align without splitting them into
    /// one tensor per instance -- a different set of tensors, and so a decision
    /// above this module.
    pub banded: BTreeSet<String>,
}

/// Classify what a compiled plan's groups read.
///
/// The classification is a fact the plan already states: a binding that starts
/// where its tensor starts reads the whole of it, and one that starts inside it
/// is a band. Nothing here inspects names.
pub fn streamable_tensors(
    plan: &crate::plan::LoadPlan,
    metadata: &crate::checkpoint::CheckpointMetadata,
) -> Streamable {
    let by_id: HashMap<_, _> = metadata.tensors.iter().map(|t| (t.id, t)).collect();
    let mut out = Streamable::default();
    for group in &plan.groups {
        for instance in &group.bindings {
            for binding in instance {
                let Some(tensor) = by_id.get(&binding.tensor_id) else {
                    continue;
                };
                if binding.file_offset == tensor.file_offset {
                    out.whole.insert(tensor.name.clone());
                } else {
                    out.banded.insert(tensor.name.clone());
                }
            }
        }
    }
    // A bank whose instance 0 binds its base looks whole from that one binding.
    // What settles it is any sibling binding that does not.
    for name in &out.banded {
        out.whole.remove(name);
    }
    out
}

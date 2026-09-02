//! Writing a `pie.serving/1` artifact. [`write`] puts a servable `.zt` on
//! disk: tensors in caller order, a [`Stamp`] under [`serving::PROFILE`], and
//! a per-object block table; deleting those two keys leaves an ordinary
//! checkpoint of the same weights.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ztensor::DType as ZDType;
use ztensor::format::cbor::Value;

use crate::error::Error;
use crate::serving::{self, PROFILE, Stamp};
use crate::types::TensorDecl;

/// The placement alignment this build writes at — a policy, not a format
/// fact. Payloads are read into page-locked memory on huge-page boundaries,
/// so a blob beginning on 2 MiB sizes a pinned buffer cleanly. Not stored in
/// the file; a reader recovers it with [`serving::alignment`].
pub const SERVING_ALIGN: u64 = 2 << 20;

/// Free space a write must leave behind after the artifact fits — a
/// filesystem filled to its last block by a large import is a machine
/// nothing else on it can run.
pub const MARGIN: u64 = 256 << 20;

/// Where one part's bytes come from — a choice the caller makes per part.
/// [`Streamed`](Payload::Streamed) is the default and the one every large
/// plane must take: the writer declares the length, bytes arrive through
/// [`write`]'s `fill` in chunks, residency is one block rather than one
/// plane.
///
/// [`Whole`](Payload::Whole) exists so identical parts can share one blob —
/// only the bytes-in-hand path can compare and dedupe them, confirming the
/// match by reading the candidate back. Use it for a plane that is tied or
/// replicated, `Streamed` for everything else.
pub enum Payload<'a> {
    /// The bytes, in hand. Shares a blob with any identical part already
    /// written.
    Whole(&'a [u8]),
    /// The part's decoded size. The bytes arrive through [`write`]'s `fill`.
    Streamed(u64),
}

impl Payload<'_> {
    /// The part's decoded size, whichever way its bytes arrive.
    #[must_use]
    pub fn len(&self) -> u64 {
        match self {
            Payload::Whole(bytes) => bytes.len() as u64,
            Payload::Streamed(length) => *length,
        }
    }

    /// Whether this part has no bytes at all. A zero-length part is a part.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One part of a serving object: a name, how its elements are stored, and
/// where the bytes come from.
pub struct Part<'a> {
    /// `data`, `scales`, `zeros`, or `r0`…`r<n>` under `pie.banded/1`. The
    /// container sorts parts by name; nothing here does.
    pub name: &'a str,
    /// The storage type.
    pub dtype: ZDType,
    /// The logical type laid over `dtype` when there is one — `f4_e2m1`,
    /// `f8_e4m3fn`. `None` means the logical type IS the storage type.
    pub logical: Option<&'a str>,
    /// Where the bytes come from. Serving parts are raw, so the length
    /// either way is also the decoded size the blocks tile.
    pub payload: Payload<'a>,
}

/// One object of a serving artifact, in the shape the container states it.
/// A quantized plane group is one object with several parts: both parts sit
/// inside one object's span and land in the sequence at that position.
pub struct Object<'a> {
    /// The serving object's name — this SKU's plane name, or a `__meta__/`
    /// metadata name.
    pub name: &'a str,
    /// The whole tensor's shape, in elements.
    pub shape: Vec<u64>,
    /// The layout profile id: `dense`, `zt.mx/1`, `zt.quant_group/1`,
    /// `gguf.<type>/1`. This profile adds none of its own at `tp_size == 1`.
    pub layout: &'a str,
    /// What the layout profile needs to be read back — `axis`, `block_size`,
    /// `bits`. The block table is not written here: [`write`] computes it and
    /// merges it in under [`serving::PROFILE`].
    pub attributes: Option<Value>,
    /// The object's parts. At least one.
    pub parts: Vec<Part<'a>>,
}

impl<'a> Object<'a> {
    /// The object a [`TensorDecl`] describes, under the same layout and
    /// storage type `file/write.rs` gives it. Single-part and named `data`;
    /// companions (scales, zeros) are their own declarations. A caller with a
    /// genuinely multi-part object builds [`Object`] itself.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] for an encoding with no zTensor layout profile,
    /// for a declaration naming a sub-byte code with no element width to
    /// store, for a negative extent, or for a name in the reserved `__meta__/`
    /// namespace.
    pub fn of(decl: &'a TensorDecl, bytes: &'a [u8]) -> Result<Object<'a>, Error> {
        crate::file::meta::reject_reserved(&decl.name)?;
        let (dtype, logical) = super::write::storage_of(decl.encoding.dtype(), &decl.encoding)?;
        let (layout, attributes) = super::write::profile_of(&decl.encoding)?;
        let shape = decl
            .shape
            .iter()
            .map(|&extent| {
                u64::try_from(extent).map_err(|_| {
                    Error::Checkpoint(format!("tensor {} has negative extent {extent}", decl.name))
                })
            })
            .collect::<Result<Vec<u64>, Error>>()?;
        Ok(Object {
            name: &decl.name,
            shape,
            layout,
            attributes,
            parts: vec![Part {
                name: "data",
                dtype,
                logical,
                payload: Payload::Whole(bytes),
            }],
        })
    }
}

/// Write the serving artifact at `path`, atomically.
///
/// `objects` are written in the order given, except metadata objects
/// (`__meta__/…`) go first regardless of position — the payload run must be
/// uninterrupted, and a metadata object between two planes would break a
/// boot's contiguous read.
///
/// `fill` delivers every [`Payload::Streamed`] part's bytes, once per part,
/// in the order the container writes them (metadata first, then each
/// object's parts sorted by name), and must deliver exactly [`Payload::len`]
/// bytes. A [`Payload::Whole`] part is never passed to `fill`. An object may
/// not mix the two kinds.
///
/// `align` is the placement policy ([`SERVING_ALIGN`] is what this build
/// passes); a file written at one alignment still verifies under a reader
/// expecting another, since alignment is recovered from the offsets.
///
/// `provenance` stays flat, beside the serving key rather than inside it —
/// those keys are file-general and true of a checkpoint with no profile at
/// all. A key that collides with the serving key is refused.
///
/// # The publish is a temp file, an fsync and a rename
///
/// Bytes land in `.<name>.<pid>.part` beside the target and are renamed at
/// the end, so a process that dies mid-write leaves an unnamed partial file
/// rather than a corrupt one under the artifact's own name.
///
/// # Errors
///
/// [`Error::Checkpoint`] when the filesystem has less than the artifact plus
/// [`MARGIN`] free; when a provenance key collides with a stamp key; when an
/// object has no parts; and when the manifest this call produced does not
/// tile its own payload run ([`serving::tiling_fault`]).
pub fn write(
    path: &Path,
    stamp: &Stamp,
    provenance: &BTreeMap<String, String>,
    align: u64,
    objects: &[Object<'_>],
    fill: impl FnMut(&str, &str, &mut Chunks<'_>) -> Result<(), Error>,
) -> Result<PathBuf, Error> {
    let directory = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(directory).map_err(|why| {
        Error::Checkpoint(format!("cannot create {}: {why}", directory.display()))
    })?;
    refuse_for_space(directory, path, objects)?;

    let temp = partial_path(path);
    if let Err(why) = emit(&temp, stamp, provenance, align, objects, fill) {
        let _ = std::fs::remove_file(&temp);
        return Err(why);
    }
    // The writer checks its own output before anybody can open it: read back
    // the manifest and run the tiling rule with the alignment recovered from
    // it, the way a reader would.
    if let Err(why) = check_tiling(&temp, path) {
        let _ = std::fs::remove_file(&temp);
        return Err(why);
    }
    std::fs::rename(&temp, path).map_err(|why| {
        let _ = std::fs::remove_file(&temp);
        Error::Checkpoint(format!("publishing {}: {why}", path.display()))
    })?;
    Ok(path.to_path_buf())
}

/// The whole file, written to one path, without the temp name, the rename or
/// the space refusal — split out so a test can name a file the real path
/// never would and still get exactly the bytes the real path produces.
fn emit(
    target: &Path,
    stamp: &Stamp,
    provenance: &BTreeMap<String, String>,
    align: u64,
    objects: &[Object<'_>],
    mut fill: impl FnMut(&str, &str, &mut Chunks<'_>) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut writer = ztensor::Writer::options()
        .canonical(false)
        .align(align)
        .create(target)
        .map_err(Error::from)?;

    // Metadata first, then the serving run, so the run is uninterrupted by
    // construction rather than by a caller's care.
    let metadata = objects.iter().filter(|it| !serving::is_serving(it.name));
    let served = objects.iter().filter(|it| serving::is_serving(it.name));
    let mut tables: BTreeMap<String, BTreeMap<String, Vec<u8>>> = BTreeMap::new();
    for object in metadata.chain(served) {
        if let Err(why) = add(&mut writer, stamp, object, &mut tables, &mut fill) {
            writer.abandon();
            return Err(why);
        }
    }

    // Attributes are set last: the manifest is written at `finish`, so the
    // block tables (a fold over bytes already streamed) can be handed over
    // only after every object is closed.
    let attributes = match file_attributes(stamp, &tables, provenance) {
        Ok(attributes) => attributes,
        Err(why) => {
            writer.abandon();
            return Err(why);
        }
    };
    writer.set_attributes(attributes);
    writer.finish().map_err(Error::from)?;

    // A serving artifact is the model, so unlike the container's own
    // `publish`, a failed fsync here is reported rather than forgiven.
    std::fs::File::open(target)
        .and_then(|file| file.sync_all())
        .map_err(|why| Error::Checkpoint(format!("syncing {}: {why}", target.display())))
}

/// Where a streamed part's bytes go, handed to [`write`]'s `fill` one part at
/// a time. Owns the block fold as well as the sink, so the table it produces
/// describes what was actually written, not what was meant.
pub struct Chunks<'w> {
    writer: &'w mut ztensor::Writer,
    sink: &'w mut ztensor::write::Sink,
    fold: serving::BlockFold,
    written: u64,
    expect: u64,
}

impl Chunks<'_> {
    /// Appends `chunk` to the open part.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] when the chunk would carry the part past its
    /// declared length, and whatever the container refuses.
    pub fn put(&mut self, chunk: &[u8]) -> Result<(), Error> {
        self.written = self.written.saturating_add(chunk.len() as u64);
        if self.written > self.expect {
            return Err(Error::Checkpoint(format!(
                "a streamed part was declared {} bytes and has been handed {}",
                self.expect, self.written,
            )));
        }
        self.fold.eat(chunk);
        self.sink.write(self.writer, chunk).map_err(Error::from)
    }
}

/// One object, written and — when it is served — its block tables folded into
/// `tables` from the same bytes the container was handed.
fn add(
    writer: &mut ztensor::Writer,
    stamp: &Stamp,
    object: &Object<'_>,
    tables: &mut BTreeMap<String, BTreeMap<String, Vec<u8>>>,
    fill: &mut impl FnMut(&str, &str, &mut Chunks<'_>) -> Result<(), Error>,
) -> Result<(), Error> {
    if object.parts.is_empty() {
        return Err(Error::Checkpoint(format!(
            "serving object {:?} has no parts, so it has nothing to serve",
            object.name,
        )));
    }
    let streamed = object
        .parts
        .iter()
        .filter(|part| matches!(part.payload, Payload::Streamed(_)))
        .count();
    if streamed != 0 && streamed != object.parts.len() {
        return Err(Error::Checkpoint(format!(
            "serving object {:?} mixes parts whose bytes are in hand with parts that \
             are streamed, and the container declares a streamed object's parts all \
             at once: give the object one kind or the other",
            object.name,
        )));
    }
    let served = serving::is_serving(object.name);
    if streamed == 0 {
        return add_whole(writer, stamp, object, tables, served);
    }
    // Parts in name order: the order the container's sink walks them.
    let mut order: Vec<&Part<'_>> = object.parts.iter().collect();
    order.sort_by_key(|part| part.name);

    let shape = object.shape.clone();
    let layout = object.layout.to_string();
    let attributes = object.attributes.clone();
    let declared: Vec<(String, ZDType, Option<String>, u64)> = order
        .iter()
        .map(|part| {
            (
                part.name.to_string(),
                part.dtype,
                part.logical.map(str::to_string),
                part.payload.len(),
            )
        })
        .collect();
    let mut sink = writer
        .stream(object.name, move |described| {
            let mut described = described.shape(shape).layout(layout);
            if let Some(attributes) = attributes {
                described = described.attributes(attributes);
            }
            for (name, dtype, logical, length) in declared {
                described = described.part(name, move |built| {
                    let mut built = built.dtype(dtype);
                    if let Some(logical) = logical {
                        built = built.logical(logical);
                    }
                    built.length(length)
                });
            }
            described
        })
        .map_err(Error::from)?;

    let mut folded: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for part in order {
        let expect = part.payload.len();
        let mut chunks = Chunks {
            writer,
            sink: &mut sink,
            fold: serving::BlockFold::new(stamp.block_algorithm, stamp.block_bytes),
            written: 0,
            expect,
        };
        fill(object.name, part.name, &mut chunks)?;
        if chunks.written != expect {
            return Err(Error::Checkpoint(format!(
                "{}'s part {:?} was declared {expect} bytes and {} arrived",
                object.name, part.name, chunks.written,
            )));
        }
        let table = chunks.fold.finish();
        if served {
            folded.insert(part.name.to_string(), table);
        }
    }
    sink.close(writer).map_err(Error::from)?;
    if served {
        tables.insert(object.name.to_string(), folded);
    }
    Ok(())
}

/// [`add`], for an object whose every part's bytes are in hand — the path
/// that shares a blob with an identical part already written.
fn add_whole(
    writer: &mut ztensor::Writer,
    stamp: &Stamp,
    object: &Object<'_>,
    tables: &mut BTreeMap<String, BTreeMap<String, Vec<u8>>>,
    served: bool,
) -> Result<(), Error> {
    if served {
        let mut folded: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for part in &object.parts {
            let Payload::Whole(bytes) = part.payload else {
                unreachable!("`add` routes here only when every part is whole");
            };
            let mut fold =
                serving::BlockFold::new(stamp.block_algorithm, stamp.block_bytes);
            fold.eat(bytes);
            folded.insert(part.name.to_string(), fold.finish());
        }
        tables.insert(object.name.to_string(), folded);
    }
    let attributes = object.attributes.clone();
    writer
        .object(object.name, |described| {
            let mut described = described
                .shape(object.shape.clone())
                .layout(object.layout.to_string());
            if let Some(attributes) = attributes {
                described = described.attributes(attributes);
            }
            for part in &object.parts {
                let (name, dtype, logical) = (part.name, part.dtype, part.logical);
                let Payload::Whole(bytes) = part.payload else {
                    unreachable!("`add` routes here only when every part is whole");
                };
                described = described.part(name.to_string(), move |built| {
                    let mut built = built.dtype(dtype);
                    if let Some(logical) = logical {
                        built = built.logical(logical.to_string());
                    }
                    built.bytes(bytes)
                });
            }
            described
        })
        .map_err(Error::from)
}

/// The file's `attributes` map: the stamp under its own key, and the flat
/// provenance keys beside it. A collision with [`serving::PROFILE`] (or any
/// other version of it) is refused rather than resolved.
fn file_attributes(
    stamp: &Stamp,
    tables: &BTreeMap<String, BTreeMap<String, Vec<u8>>>,
    provenance: &BTreeMap<String, String>,
) -> Result<Value, Error> {
    let Value::Map(mut entries) = serving::file_block(stamp, tables) else {
        return Err(Error::Internal(
            "the stamp encoded to something that is not a map".to_string(),
        ));
    };
    for (key, value) in provenance {
        if key.starts_with(serving::PROFILE_FAMILY) {
            return Err(Error::Checkpoint(format!(
                "the provenance key {key:?} is the key the stamp itself is written \
                 under, so this artifact would carry two answers for it; the serving \
                 facts live under {PROFILE:?} and the provenance keys are the flat ones \
                 that say where the weights came from"
            )));
        }
        entries.push((Value::Text(key.clone()), Value::Text(value.clone())));
    }
    Ok(Value::Map(entries))
}

/// Where the partial file lives while it is being written. Beside the
/// target, so the rename is atomic; dot-prefixed and pid-stamped, so two
/// concurrent imports cannot collide.
fn partial_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|it| it.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact.zt".to_string());
    path.with_file_name(format!(".{name}.{}.part", std::process::id()))
}

/// Refuse the write before it starts if the disk cannot hold it. The
/// estimate is the sum of every part's bytes; padding and the manifest are
/// what [`MARGIN`] covers.
fn refuse_for_space(directory: &Path, path: &Path, objects: &[Object<'_>]) -> Result<(), Error> {
    let total: u64 = objects
        .iter()
        .flat_map(|object| object.parts.iter())
        .map(|part| part.payload.len())
        .sum();
    let need = total.saturating_add(MARGIN);
    let free = available_bytes(directory)?;
    if free < need {
        return Err(Error::Checkpoint(format!(
            "{} has {:.1} GiB free and the serving artifact {} wants {:.1} GiB \
             ({} GiB of planes plus a {} GiB margin); point the model store at a disk \
             with more space",
            directory.display(),
            free as f64 / (1u64 << 30) as f64,
            path.display(),
            need as f64 / (1u64 << 30) as f64,
            total >> 30,
            MARGIN >> 30,
        )));
    }
    Ok(())
}

/// Bytes an unprivileged process may still write under `directory`. `libc`
/// directly: one `statvfs` and a multiply, and std has no answer for it.
fn available_bytes(directory: &Path) -> Result<u64, Error> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(directory.as_os_str().as_bytes()).map_err(|_| {
        Error::Checkpoint(format!(
            "{} is not a path this platform can state",
            directory.display()
        ))
    })?;
    // SAFETY: `path` is a NUL-terminated C string that outlives the call, and
    // `stat` is a plain out-parameter the call fully initializes on success.
    let mut stat = unsafe { std::mem::zeroed::<libc::statvfs>() };
    let rc = unsafe { libc::statvfs(path.as_ptr(), &raw mut stat) };
    if rc != 0 {
        return Err(Error::Checkpoint(format!(
            "{}: cannot read the filesystem's free space",
            directory.display()
        )));
    }
    Ok((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

/// Does what was just written tile its own payload run? The manifest is
/// read back from the partial file rather than accumulated while writing, so
/// it's a statement about the bytes actually there. `path`, not the
/// soon-deleted partial, is named in the refusal.
fn check_tiling(temp: &Path, path: &Path) -> Result<(), Error> {
    let manifest = ztensor::read::manifest_of(temp)
        .map_err(Error::from)?
        .ok_or_else(|| {
            Error::Internal(format!(
                "{} was written without a manifest",
                path.display()
            ))
        })?;
    let spans = serving::spans(&manifest);
    if let Some(fault) = serving::tiling_fault(&spans, serving::alignment(&spans)) {
        return Err(Error::Checkpoint(format!(
            "the serving artifact {} {fault}",
            path.display(),
        )));
    }
    Ok(())
}


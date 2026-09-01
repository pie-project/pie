//! **Writing a `pie.serving/1` artifact** — the import-only half of the pair.
//!
//! One function, [`write`], and the two value types it takes its tensors in.
//! It puts a servable `.zt` on the disk: the tensors laid out in the order the
//! caller hands them, one file attribute keyed [`serving::PROFILE`] holding a
//! [`Stamp`], and one attribute of the same name on each serving object
//! holding that object's block tables, computed over its own bytes.
//!
//! **Everything this profile adds is TWO KEYS**, one shape at file level and
//! one per object, each holding a map whose key states its own version. That
//! is what makes the owner's rule cheap to state and cheap to check: delete
//! them and an ordinary checkpoint of the same weights remains.
//!
//! # It is under `file/`, and that is the whole of why it may open anything
//!
//! `tests/standalone.rs`'s `nothing_below_the_reader_opens_a_file` exempts this
//! directory wholesale, by prefix, so nothing was added to that test's allow
//! list to land this module. The rule it states is unchanged: the compiler
//! computes over values, and `file/` is where a path becomes one.
//! [`serving`](crate::serving) — every definition this module spells its
//! agreement in — is deliberately outside it, and is scanned.
//!
//! # What this writer does NOT know
//!
//! **It does not compute the ranking.** `Ranking::images` order is what the
//! caller hands it, and a caller that hands it any other order writes a file
//! this build reads perfectly and merely reads unranked — which is
//! [`serving::sequence`]'s own admitted limit, and the reason order can be an
//! argument rather than a writer's opinion.
//!
//! **And it does not know what a rung, a budget or a device is.** No parameter
//! here is a function of one. `c1` and `c2` are chosen at a boot from that
//! boot's budgets; nothing this writer stores can contradict a pair it never
//! saw, which is what makes one artifact serve any of them (§5.6).
//!
//! # `.canonical(false)`, and what that actually costs
//!
//! Canonical form requires ascending insertion — `ztensor`'s writer refuses
//! otherwise, with `InvalidInput("canonical form requires sorted insertion:
//! …")` — and ascending name order is precisely the order a serving artifact
//! must not use. So the label is turned off.
//!
//! **The label is all it costs**, and this crate has already written the
//! paragraph: `file/write.rs`'s `finish_sharded` says of its own
//! non-canonical root that *"since zTensor 2.1.0 a non-canonical writer still
//! places on 64 KiB … Nothing on disk records the label and no reader checks
//! it."* `ztensor::write::Options::canonical`'s own doc says the same from the
//! other side: *"Placement is not part of what you give up."* And digests are
//! unconditional on the bytes-in-hand path — the container fills a part's
//! `digest` for every part it is handed bytes for — so §3's
//! every-serving-part-carries-a-digest MUST is met by the container's default
//! behaviour rather than by anything this module remembers to do.
//!
//! # Why the blocks tile the DECODED SIZE and not the padded span
//!
//! Profile departure #1, and it is forced rather than chosen. §2.4 makes
//! padding a writer policy (*"4096 is a floor, not a ceiling"*), so a digest
//! that covered the padding would make two files with the same tensors and a
//! different [`write`] `align` fail each other's verification — which is what
//! that argument would otherwise buy at the price of interchange.
//! [`serving::block_count`] already tiles a decoded size; this module feeds it
//! `part.bytes.len()` and nothing else.
//!
//! Nothing is lost. The padding's content is a spec MUST (`0x00`) and is
//! checkable against zero at no hashing cost.
//!
//! # Aliasing is permitted, and is the replication path
//!
//! §2.4: references are valid iff they have **exactly equal** `(offset,
//! length)` or do not overlap at all. `ztensor`'s writer already shares a blob
//! when two parts' bytes are byte-identical, and confirms the match by reading
//! the candidate back, so a hash collision can never alias two different
//! tensors onto one span. That is where M-4f's replicated planes come from:
//! compile the plane ONCE, hand the same slice twice, and the two objects
//! share one blob rather than agreeing by luck.
//!
//! Partial overlap is impossible by construction — every blob is placed by the
//! container, never by this module — and [`write`] still runs
//! [`serving::tiling_fault`] over the manifest it produced, before the file is
//! published. A writer that checks its own output is the only kind whose
//! invariant is a fact rather than a hope.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ztensor::DType as ZDType;
use ztensor::format::cbor::Value;

use crate::error::Error;
use crate::serving::{self, PROFILE, Stamp};
use crate::types::TensorDecl;

/// **The placement alignment this build writes at**, and it is a policy rather
/// than a format fact.
///
/// `weight_cache/tier.rs`'s `TIER_ALIGN`, carried across with its reason: the
/// payload is read into page-locked memory an allocator hands out on huge-page
/// boundaries, so a serving blob that begins on 2 MiB is one a pinned buffer
/// can be sized for. §2.4 permits any coarser power of two and forbids storing
/// the number — *"because alignment is observable from the offsets themselves,
/// the actual alignment used by a writer is not stored in the file"* — so this
/// is a default for [`write`]'s argument and never an attribute. A reader
/// recovers it with [`serving::alignment`].
pub const SERVING_ALIGN: u64 = 2 << 20;

/// **Free space a write must leave behind after the artifact fits.**
///
/// `engine-cuda/src/weight_cache.rs`'s `MARGIN`, re-aimed: the number was
/// chosen against `[model] weight_cache_dir` and the disk it guards is now the
/// model store, because the store is where a serving artifact lives once there
/// is only one file. The quantity is unchanged and so is the argument — a
/// filesystem filled to its last block by a hundred-gigabyte import is a
/// machine nothing else on it can run.
pub const MARGIN: u64 = 256 << 20;

/// **Where one part's bytes come from, and it is a CHOICE the caller makes
/// per part.**
///
/// [`Streamed`](Payload::Streamed) is the default and the one every large
/// plane must take: the writer declares the length, the bytes arrive through
/// [`write`]'s `fill` in chunks, and residency is one block rather than one
/// plane. The catalog says why that is not a preference — the largest single
/// plane it ships is `qwen38-flash-bf16`'s `ple.table` at **95.4 GiB**, and
/// three more rows exceed 2 GiB.
///
/// [`Whole`](Payload::Whole) exists for exactly one property, and it is worth
/// stating plainly because it is the only reason two variants are better than
/// one. **Blob sharing is reachable no other way.** `ztensor`'s
/// `write_or_share_blob` is private and only the bytes-in-hand path calls it:
/// hand two parts identical bytes and the container writes one blob and points
/// both at it, confirming the match by reading the candidate back so a hash
/// collision can never alias two different tensors. That is §2.4's tying case
/// and §M-4f's replication path, and a streamed part cannot have it, because
/// nothing holds the earlier bytes to compare against.
///
/// So the rule for a caller is short: **`Whole` for a plane that is tied or
/// replicated, `Streamed` for everything else.** The tied set is small and
/// bounded — §M-4f measured the vocabulary pair at 97.7–99.9% of the
/// replicated planes in five of six SKUs — so the memory this costs is one
/// embedding, not one model.
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
    /// `data`, `scales`, `zeros` — §5.1's part schema — or `r0`…`r<n>` under
    /// `pie.banded/1`. The container sorts parts by name; nothing here does.
    pub name: &'a str,
    /// The storage type. `file/write.rs`'s `storage_of` answers it when the
    /// object came from a [`TensorDecl`] through [`Object::of`].
    pub dtype: ZDType,
    /// The logical type laid over `dtype` when there is one — `f4_e2m1`,
    /// `f8_e4m3fn`. `None` means the logical type IS the storage type.
    pub logical: Option<&'a str>,
    /// Where the bytes come from. Serving parts are raw (§3), so the length
    /// either way is also the decoded size the blocks tile.
    pub payload: Payload<'a>,
}

/// One object of a serving artifact, in the shape the container states it.
///
/// A quantized plane group is ONE object with several parts, which is the
/// format's own answer to `tier::Group`'s `plane` field: the `0`/`1` that used
/// to tell a split-plane bank's codes from its scales has no referent here,
/// because both parts sit inside one object's span and land in the sequence at
/// that object's position.
pub struct Object<'a> {
    /// The serving object's name — this SKU's plane name after M-4a, or a
    /// `__meta__/` metadata name.
    pub name: &'a str,
    /// The whole tensor's shape, in elements.
    pub shape: Vec<u64>,
    /// The layout profile id: `dense`, `zt.mx/1`, `zt.quant_group/1`,
    /// `gguf.<type>/1`. This profile adds none of its own at `tp_size == 1`.
    pub layout: &'a str,
    /// What the layout profile needs to be read back — `axis`, `block_size`,
    /// `bits`. The block table is NOT written here: [`write`] computes it and
    /// merged in under [`serving::PROFILE`], because it is a fact about bytes
    /// this value does not yet have.
    pub attributes: Option<Value>,
    /// The object's parts. At least one.
    pub parts: Vec<Part<'a>>,
}

impl<'a> Object<'a> {
    /// **The object a [`TensorDecl`] describes**, under the same layout and
    /// the same storage type `file/write.rs` gives it.
    ///
    /// Single-part and named `data`, because that is what a pie declaration
    /// is: a plane with one payload, whose companions — scales, zeros — are
    /// their own declarations under their own names. A caller with a genuinely
    /// multi-part object builds [`Object`] itself.
    ///
    /// Routing through `file/write.rs` rather than deciding again here is the
    /// point: a serving artifact's layouts must be the SAME function of an
    /// encoding as an ordinary checkpoint's. Two functions would be two files
    /// claiming one layout id for different bytes.
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

/// **Write the serving artifact at `path`, atomically.**
///
/// `objects` are written in the order given, except that metadata objects
/// (`__meta__/…`, [`serving::is_serving`]'s complement) go FIRST whatever
/// position they were handed in. That is not a re-ranking: §3 requires the
/// payload run be uninterrupted — no metadata blob and no unreferenced blob
/// inside it — and a metadata object written between two planes is exactly the
/// foreign blob that breaks a boot's one contiguous read. Putting them before
/// the run makes the MUST structural instead of a rule a caller has to
/// remember; the manifest blob lands after the run, where the container puts
/// it.
///
/// `fill` delivers every [`Payload::Streamed`] part's bytes, and is called
/// once per such part, **in the order the container writes them**: objects in
/// [`write`]'s own order (metadata first), and within an object its parts
/// SORTED BY NAME, because that is the order `ztensor`'s sink walks a
/// multi-part object's parts and this module cannot see it. It is handed the
/// object's name, the part's name and a [`Chunks`] to put bytes into, and it
/// must deliver exactly [`Payload::len`] bytes — the container refuses a sink
/// closed short or long, and so does this.
///
/// A [`Payload::Whole`] part is never passed to `fill`; its bytes are already
/// here. An object may not MIX the two, because the container's `stream`
/// requires every part of a streamed object to be declared with a length.
///
/// `align` is the placement policy — [`SERVING_ALIGN`] is what this build
/// passes — and it is a policy in the strong sense: a file written at one
/// alignment verifies under a reader expecting another, because the blocks
/// tile decoded sizes and the alignment is recovered from the offsets rather
/// than believed from a field.
///
/// `provenance` is `file/meta.rs`'s existing flat text vocabulary —
/// `pie_version`, `pie_source`, `pie_source_encoding` — and it **stays flat,
/// beside the serving key rather than inside it**. That mixture is a decision:
/// those three are file-general provenance, true of an artifact whatever
/// profile it does or does not carry, and folding them into this profile's
/// block would mean an ordinary checkpoint could no longer say where it came
/// from. A key that collides with the serving key is refused rather than
/// silently dropped or silently winning.
///
/// # The publish is a temp file, an fsync and a rename
///
/// `tier::write_out`'s discipline, kept whole: the bytes land in
/// `.<name>.<pid>.part` beside the target and are renamed at the end, so a
/// process that dies mid-write leaves a partial file nobody will ever name
/// rather than a corrupt one under the artifact's own name. The reason it is
/// the container's `create` into a path of this module's choosing, rather than
/// its `publish` (which does temp-fsync-rename by itself), is that the tiling
/// check below has to read back the manifest of what was written, and a
/// `publish` has already renamed by the time there is a manifest to read.
///
/// # Errors
///
/// [`Error::Checkpoint`] when the filesystem has less than the artifact plus
/// [`MARGIN`] free — **named with the artifact**, because the disk cost is per
/// deployment and an operator deleting one needs to know which; when a
/// provenance key collides with a stamp key; when an object has no parts; and
/// when the manifest this call produced does not tile its own payload run
/// ([`serving::tiling_fault`]). Whatever the container refuses arrives through
/// [`Error::from`], which is what keeps a container-version refusal
/// [`Error::Unsupported`] rather than "malformed".
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
    // **THE WRITER CHECKS ITS OWN OUTPUT, BEFORE ANYBODY CAN OPEN IT.** Read
    // back the manifest the container just wrote and run the profile's own
    // tiling rule over it. The alignment is RECOVERED rather than assumed,
    // because that is what a reader will do, and a check that used the
    // writer's private number would be checking a different file from the one
    // on the disk.
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
/// the space refusal — `tier::emit`'s split, for `tier::emit`'s reason: a test
/// can name a file the real path never would and still get exactly the bytes
/// the real path produces.
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

    // **THE ATTRIBUTES ARE SET LAST, AND THAT IS WHAT LETS THE WRITER
    // STREAM.** `Writer::set_attributes` stores into the manifest and the
    // manifest is written at `finish`, so the block tables — which are a fold
    // over bytes that have already gone past — can be handed over after every
    // object is closed. See `serving::BLOCKS_KEY` for the measurement that
    // made this necessary rather than merely tidy.
    let attributes = match file_attributes(stamp, &tables, provenance) {
        Ok(attributes) => attributes,
        Err(why) => {
            writer.abandon();
            return Err(why);
        }
    };
    writer.set_attributes(attributes);
    writer.finish().map_err(Error::from)?;

    // The container attempts an fsync on this path and forgives a filesystem
    // that cannot do one, because durability is its `publish`'s promise and
    // this is not one. A serving artifact IS the model, so the promise is
    // wanted here: ask again, and this time say so if it fails.
    std::fs::File::open(target)
        .and_then(|file| file.sync_all())
        .map_err(|why| Error::Checkpoint(format!("syncing {}: {why}", target.display())))
}

/// **Where a streamed part's bytes go**, handed to [`write`]'s `fill` one part
/// at a time.
///
/// It owns the block fold as well as the sink, so a caller cannot deliver
/// bytes without their digests being taken from the same slices: the table
/// this produces describes what was written and not what was meant, which is
/// the property the whole verify path rests on.
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
    // Parts in NAME order, because that is the order the container's sink
    // walks them and it does not publish that order for this module to read.
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

/// [`add`], for an object whose every part's bytes are in hand — the path that
/// shares a blob with an identical part already written, which is the tying
/// and replication case and is reachable no other way.
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
/// provenance keys beside it.
///
/// A collision is refused rather than resolved. There is exactly one key to
/// collide with now — [`serving::PROFILE`], or any other version of it — and a
/// caller that passes one in `provenance` is a caller with two beliefs about
/// the serving block, either of which this function could pick and both of
/// which would make the file say something nobody wrote.
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


/// Where the partial file lives while it is being written.
///
/// Beside the target, so the rename is within one filesystem and is therefore
/// atomic; dot-prefixed and pid-stamped, so two imports of two deployments in
/// one directory cannot collide and neither leaves something an `ls` mistakes
/// for a model.
fn partial_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|it| it.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact.zt".to_string());
    path.with_file_name(format!(".{name}.{}.part", std::process::id()))
}

/// **Refuse the write before it starts if the disk cannot hold it.**
///
/// `tier::write_out`'s refusal, re-aimed at the model store. The estimate is
/// the sum of every part's bytes: padding and the manifest are what [`MARGIN`]
/// covers, and an estimate that tried to predict them would be a second copy
/// of the container's placement arithmetic kept in step by hand.
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

/// Bytes an unprivileged process may still write under `directory`.
///
/// `libc` rather than a crate, and it is
/// `engine-cuda/src/weight_cache.rs`'s own reasoning: this is one `statvfs`
/// and a multiply, and the standard library has no answer for it at all.
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

/// **Does what was just written tile its own payload run?**
///
/// The manifest is read back from the partial file rather than accumulated
/// while writing, for `file/write.rs`'s own reason about shard identities: it
/// is a statement about the bytes that are actually there, which is the only
/// version of it worth having. `path` is named in the refusal because the
/// partial is about to be deleted and its name would tell an operator nothing.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::BlockAlgorithm;

    /// A block size at the profile's floor, so a few kilobytes of fixture is
    /// several blocks and the tiling is exercised rather than assumed.
    const BLOCK: u64 = serving::MIN_BLOCK_BYTES;

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pie_emit_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn stamp() -> Stamp {
        Stamp {
            serving: serving::PROFILE.to_string(),
            backend: "cuda".to_string(),
            tp_size: 1,
            sku: "qwen_3".to_string(),
            precision: "bf16".to_string(),
            layout_revision: 1,
            block_bytes: BLOCK,
            block_algorithm: BlockAlgorithm::Xxh3,
            adapters_zeroed: true,
            model_id: Some("qwen/qwen3-0.6b".to_string()),
            recipe_digest: None,
        }
    }

    fn plane(seed: u8, len: usize) -> Vec<u8> {
        (0..len).map(|at| seed.wrapping_add(at as u8)).collect()
    }

    /// Every fixture that hands its bytes in passes this: nothing is streamed,
    /// so it must never be reached, and it says so rather than succeeding
    /// quietly if the routing ever changes.
    fn no_fill(object: &str, part: &str, _: &mut Chunks<'_>) -> Result<(), Error> {
        panic!("{object}/{part} asked to be filled and this fixture streams nothing")
    }

    fn dense<'a>(name: &'a str, bytes: &'a [u8]) -> Object<'a> {
        Object {
            name,
            shape: vec![bytes.len() as u64],
            layout: "dense",
            attributes: None,
            parts: vec![Part {
                name: "data",
                dtype: ZDType::U8,
                logical: None,
                payload: Payload::Whole(bytes),
            }],
        }
    }

    /// **The order is the caller's, and the manifest is what records it.**
    ///
    /// The three planes go in reverse name order on purpose: a canonical
    /// writer would refuse that outright, and the whole reason for
    /// `.canonical(false)` is that `Ranking::images` order is not name order.
    #[test]
    fn the_sequence_is_the_order_it_was_handed_and_not_the_names() {
        let dir = tmpdir("order");
        let (c, b, a) = (plane(1, 9000), plane(2, 5000), plane(3, 3000));
        let objects = [dense("c", &c), dense("b", &b), dense("a", &a)];
        write(
            &dir.join("m.zt"),
            &stamp(),
            &BTreeMap::new(),
            4096,
            &objects,
            no_fill,
        )
        .unwrap();

        let manifest = ztensor::read::manifest_of(dir.join("m.zt"))
            .unwrap()
            .unwrap();
        assert_eq!(serving::sequence(&manifest), vec!["c", "b", "a"]);
    }

    /// **The blocks tile the DECODED size**, which is departure #1: the table
    /// length is a function of the part's bytes and of nothing the writer
    /// chose about placement.
    #[test]
    fn a_block_table_is_as_long_as_the_bytes_and_not_as_the_span() {
        let dir = tmpdir("blocks");
        let bytes = plane(7, 3 * BLOCK as usize + 11);
        for (at, align) in [4096u64, 65536, SERVING_ALIGN].iter().enumerate() {
            let path = dir.join(format!("m{at}.zt"));
            write(
                &path,
                &stamp(),
                &BTreeMap::new(),
                *align,
                &[dense("w", &bytes)],
                no_fill,
            )
            .unwrap();
            let manifest = ztensor::read::manifest_of(&path).unwrap().unwrap();
            let table = table_of(&manifest, "w", "data");
            assert_eq!(
                table.len(),
                serving::table_len(bytes.len() as u64, BLOCK, BlockAlgorithm::Xxh3),
                "the table at align {align} is not the decoded size's",
            );
            // And the digests themselves are the same at every alignment,
            // which is the property a padded-span digest would have lost.
            assert_eq!(
                table,
                serving::encode_blocks(
                    BlockAlgorithm::Xxh3,
                    (0..serving::block_count(bytes.len() as u64, BLOCK))
                        .map(|which| {
                            let span = serving::block_span(bytes.len() as u64, BLOCK, which)
                                .expect("a counted block has a span");
                            BlockAlgorithm::Xxh3
                                .digest(&bytes[span.start as usize..span.end as usize])
                        })
                        .collect::<Vec<_>>()
                        .iter()
                        .map(Vec::as_slice),
                ),
            );
        }
    }

    /// **Two objects whose bytes are identical share ONE blob** — §2.4's
    /// blessed aliasing, which is M-4f's replication path. The check is
    /// exactly equal `(offset, length)`, because that is the only sharing the
    /// spec permits and the only kind [`serving::tiling_fault`] waves through.
    #[test]
    fn identical_planes_are_one_span_named_twice() {
        let dir = tmpdir("alias");
        let shared = plane(5, 7000);
        let other = plane(6, 7000);
        let objects = [
            dense("embed", &shared),
            dense("head", &shared),
            dense("norm", &other),
        ];
        write(
            &dir.join("m.zt"),
            &stamp(),
            &BTreeMap::new(),
            4096,
            &objects,
            no_fill,
        )
        .unwrap();

        let manifest = ztensor::read::manifest_of(dir.join("m.zt"))
            .unwrap()
            .unwrap();
        let blob = |name: &str| {
            let part = &manifest.objects[name].parts["data"];
            (part.blob.offset, part.blob.length)
        };
        assert_eq!(blob("embed"), blob("head"));
        assert_ne!(blob("embed"), blob("norm"));

        // And the sharing is a saving rather than a coincidence: the same
        // three objects with three different payloads cost one plane more.
        let apart = plane(7, 7000);
        write(
            &dir.join("apart.zt"),
            &stamp(),
            &BTreeMap::new(),
            4096,
            &[
                dense("embed", &shared),
                dense("head", &apart),
                dense("norm", &other),
            ],
            no_fill,
            )
        .unwrap();
        let size = |name: &str| std::fs::metadata(dir.join(name)).unwrap().len();
        assert!(size("m.zt") + 7000 <= size("apart.zt"));
    }

    /// **A metadata object never lands inside the payload run**, whatever
    /// position it was handed in — §3's uninterrupted-run MUST, made
    /// structural.
    #[test]
    fn the_descriptor_sits_outside_the_run_it_was_handed_into() {
        let dir = tmpdir("meta");
        let (a, b, descriptor) = (plane(1, 6000), plane(2, 6000), b"{\"sku\":\"qwen_3\"}");
        let objects = [
            dense("a", &a),
            dense("__meta__/model/descriptor", descriptor),
            dense("b", &b),
        ];
        write(
            &dir.join("m.zt"),
            &stamp(),
            &BTreeMap::new(),
            4096,
            &objects,
            no_fill,
        )
        .unwrap();

        let manifest = ztensor::read::manifest_of(dir.join("m.zt"))
            .unwrap()
            .unwrap();
        let spans = serving::spans(&manifest);
        assert_eq!(
            spans.iter().map(|span| span.object).collect::<Vec<_>>(),
            vec!["a", "b"],
        );
        let run = serving::payload_at(&spans).expect("the file serves something");
        let descriptor = manifest.objects["__meta__/model/descriptor"].parts["data"]
            .blob
            .offset;
        assert!(descriptor < run, "the descriptor is inside the payload run");
        // And it carries no `pie_blocks`: it is not served, so nothing
        // verifies a prefix of it.
        assert!(
            manifest.objects["__meta__/model/descriptor"]
                .attributes
                .is_none()
        );
    }

    /// **The writer runs the profile's own tiling check on what it wrote**,
    /// and the file it publishes is one the check passed.
    #[test]
    fn what_was_published_tiles_its_own_payload_run() {
        let dir = tmpdir("tiling");
        let (a, b) = (plane(1, 20_000), plane(2, 30_000));
        write(
            &dir.join("m.zt"),
            &stamp(),
            &BTreeMap::new(),
            SERVING_ALIGN,
            &[dense("a", &a), dense("b", &b)],
            no_fill,
            )
        .unwrap();
        let manifest = ztensor::read::manifest_of(dir.join("m.zt"))
            .unwrap()
            .unwrap();
        let spans = serving::spans(&manifest);
        assert_eq!(serving::alignment(&spans), SERVING_ALIGN);
        assert!(serving::tiling_fault(&spans, serving::alignment(&spans)).is_none());
    }

    /// **The partial file is not the artifact**, and a failed write leaves
    /// neither.
    #[test]
    fn a_refused_write_publishes_nothing_and_leaves_nothing() {
        let dir = tmpdir("partial");
        let path = dir.join("m.zt");
        let empty = Object {
            name: "w",
            shape: vec![0],
            layout: "dense",
            attributes: None,
            parts: Vec::new(),
        };
        let why = write(&path, &stamp(), &BTreeMap::new(), 4096, &[empty], no_fill).unwrap_err();
        assert!(format!("{why}").contains("has no parts"), "{why}");
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    }

    /// **A provenance key that IS the serving key is refused**, because either
    /// answer would make the file state something nobody wrote.
    #[test]
    fn two_beliefs_about_one_key_are_refused_rather_than_resolved() {
        let dir = tmpdir("collide");
        let provenance = BTreeMap::from([(PROFILE.to_string(), "metal".to_string())]);
        let bytes = plane(1, 4096);
        let why = write(
            &dir.join("m.zt"),
            &stamp(),
            &provenance,
            4096,
            &[dense("w", &bytes)],
            no_fill,
        )
        .unwrap_err();
        assert!(format!("{why}").contains(PROFILE), "{why}");
    }

    /// The stamp and the provenance keys are one attribute map, and both come
    /// back out of it.
    #[test]
    fn the_file_states_its_stamp_and_where_the_weights_came_from() {
        let dir = tmpdir("stamp");
        let provenance = BTreeMap::from([
            (crate::file::meta::VERSION_KEY.to_string(), "0.4.0".to_string()),
            (crate::file::meta::SOURCE_KEY.to_string(), "qwen/qwen3".to_string()),
        ]);
        let bytes = plane(1, 4096);
        write(
            &dir.join("m.zt"),
            &stamp(),
            &provenance,
            4096,
            &[dense("w", &bytes)],
            no_fill,
            )
        .unwrap();
        let manifest = ztensor::read::manifest_of(dir.join("m.zt"))
            .unwrap()
            .unwrap();
        let attributes = manifest.attributes.expect("the file states attributes");
        assert_eq!(Stamp::decode(&attributes).unwrap(), stamp());
        assert!(matches!(
            attributes.get(crate::file::meta::SOURCE_KEY),
            Some(Value::Text(it)) if it == "qwen/qwen3",
        ));
    }

    /// **STREAMING A PART PRODUCES THE FILE HANDING ITS BYTES IN PRODUCES** —
    /// byte for byte, at every chunking, tables included.
    ///
    /// This is the property the whole change rests on. The tables moved to the
    /// file's own attribute block so that a writer could declare an object
    /// before it had hashed the object, and the only thing that makes that
    /// safe is that the fold over arriving chunks and the fold over a slice
    /// are the same fold. A test that only checked the digests would miss a
    /// placement difference; one that only compared the files would pass on
    /// two files with no tables at all.
    ///
    /// The chunkings are deliberately hostile: 1 byte (every block boundary
    /// crossed mid-chunk in the worst way), a prime that divides nothing, one
    /// byte under a block, one byte over, and the whole thing at once.
    #[test]
    fn a_streamed_part_writes_what_the_bytes_in_hand_wrote() {
        let dir = tmpdir("streamed");
        let bytes = plane(11, 3 * BLOCK as usize + 137);

        let whole = dir.join("whole.zt");
        write(
            &whole,
            &stamp(),
            &BTreeMap::new(),
            SERVING_ALIGN,
            &[dense("w", &bytes)],
            no_fill,
        )
        .unwrap();
        let want = std::fs::read(&whole).unwrap();

        for chunk in [1usize, 97, BLOCK as usize - 1, BLOCK as usize + 1, bytes.len()] {
            let path = dir.join(format!("streamed-{chunk}.zt"));
            let streamed = Object {
                name: "w",
                shape: vec![bytes.len() as u64],
                layout: "dense",
                attributes: None,
                parts: vec![Part {
                    name: "data",
                    dtype: ZDType::U8,
                    logical: None,
                    payload: Payload::Streamed(bytes.len() as u64),
                }],
            };
            write(
                &path,
                &stamp(),
                &BTreeMap::new(),
                SERVING_ALIGN,
                &[streamed],
                |object, part, chunks| {
                    assert_eq!((object, part), ("w", "data"));
                    for piece in bytes.chunks(chunk) {
                        chunks.put(piece)?;
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&path).unwrap(),
                want,
                "streaming in {chunk}-byte pieces wrote a different file",
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A part that delivers the wrong number of bytes is refused, in both
    /// directions, and the refusal says which part and both numbers.
    ///
    /// The container catches a sink closed short on its own; this catches it
    /// EARLIER and with the object's name, and it is the only thing that
    /// catches a `fill` that returns having written nothing — the case a
    /// caller reaches by forgetting a branch, which would otherwise write a
    /// hole full of zeros that hashes and verifies perfectly.
    #[test]
    fn a_streamed_part_that_is_short_or_long_is_refused_by_name() {
        let dir = tmpdir("short");
        let bytes = plane(3, 4096);
        let object = |length: u64| Object {
            name: "w",
            shape: vec![4096],
            layout: "dense",
            attributes: None,
            parts: vec![Part {
                name: "data",
                dtype: ZDType::U8,
                logical: None,
                payload: Payload::Streamed(length),
            }],
        };

        let short = write(
            &dir.join("short.zt"),
            &stamp(),
            &BTreeMap::new(),
            4096,
            &[object(4096)],
            |_, _, chunks| chunks.put(&bytes[..100]),
        )
        .unwrap_err();
        assert!(
            format!("{short}").contains("4096") && format!("{short}").contains("100"),
            "{short}"
        );

        let long = write(
            &dir.join("long.zt"),
            &stamp(),
            &BTreeMap::new(),
            4096,
            &[object(100)],
            |_, _, chunks| chunks.put(&bytes),
        )
        .unwrap_err();
        assert!(format!("{long}").contains("100"), "{long}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An object may not mix bytes-in-hand parts with streamed ones, and the
    /// refusal says why rather than leaving the container to say it.
    #[test]
    fn an_object_that_mixes_the_two_payloads_is_refused() {
        let dir = tmpdir("mixed");
        let data = plane(5, 4096);
        let why = write(
            &dir.join("m.zt"),
            &stamp(),
            &BTreeMap::new(),
            4096,
            &[Object {
                name: "w",
                shape: vec![8192],
                layout: "dense",
                attributes: None,
                parts: vec![
                    Part {
                        name: "data",
                        dtype: ZDType::U8,
                        logical: None,
                        payload: Payload::Whole(&data),
                    },
                    Part {
                        name: "scales",
                        dtype: ZDType::U8,
                        logical: None,
                        payload: Payload::Streamed(4096),
                    },
                ],
            }],
            |_, _, chunks| chunks.put(&data),
        )
        .unwrap_err();
        assert!(format!("{why}").contains("one kind or the other"), "{why}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn table_of(manifest: &ztensor::Manifest, object: &str, part: &str) -> Vec<u8> {
        let attributes = manifest
            .attributes
            .as_ref()
            .expect("a serving artifact carries its serving block");
        serving::stated_blocks(attributes, object, part)
            .unwrap_or_else(|| panic!("the file states no block table for {object}/{part}"))
            .to_vec()
    }
}

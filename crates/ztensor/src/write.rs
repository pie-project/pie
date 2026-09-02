//! `.zt` writing.
//!
//! Append-only: magic, then blobs at aligned offsets, then the manifest blob,
//! then the footer. The default mode produces **canonical form** (spec §6.4):
//! 64 KiB placement, sorted insertion, an xxh3 digest on every object, and
//! blob sharing for byte-identical objects.
//!
//! [`Writer::object`] builds any object: a leaf or a group type, bytes in
//! hand or planes handed over one by one, local or a reference into another
//! file. [`Writer::stream`] takes the bytes a chunk at a time, and
//! [`Writer::add`] is a shorthand over `object` for the common case.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result};
use crate::format::cbor;
use crate::format::{
    align_up, check_attributes, check_name, check_shape, check_shard_name, Blob, Blocks, Digest,
    DigestAlgorithm, Hasher, Leaf, Manifest, Object, Shard, Term, ALIGN_CANONICAL, ALIGN_FLOOR,
    FOOTER_LEN, MAGIC, MAX_MANIFEST_LEN, MIN_FILE_LEN, PLANE_ALIGN, VERSION,
};
use crate::read::Source;
use crate::vocab::Vocabulary;

/// Hands out a fresh ticket for every [`Sink`] ever opened, in this process,
/// so a sink can only drive the writer that opened it.
static NEXT_SINK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Writer-side violations of reader rules surface as `InvalidInput` carrying
/// the rule's own message.
fn invalid(e: Error) -> Error {
    match e {
        Error::Reject { detail, .. } => Error::InvalidInput(detail),
        other => other,
    }
}

fn check_alignment(align: u64) -> Result<()> {
    if !align.is_power_of_two() || align < ALIGN_FLOOR {
        return Err(Error::InvalidInput(format!(
            "alignment must be a power of two >= {ALIGN_FLOOR}, got {align}"
        )));
    }
    Ok(())
}

// =======================================================================
// options
// =======================================================================

/// How to write.
pub struct Options {
    canonical: bool,
    align: Option<u64>,
    blocks: Option<u64>,
    vocab: Option<Arc<Vocabulary>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            canonical: true,
            align: None,
            blocks: None,
            vocab: None,
        }
    }
}

impl Options {
    /// Canonical form (the default): 64 KiB placement, ascending insertion,
    /// a digest on every object, raw blobs only, no block digests, single
    /// file.
    ///
    /// Turn it off to insert in any order, encode blobs, add block digests,
    /// or reference other files. Placement is *not* part of what you give
    /// up: a non-canonical writer still defaults to 64 KiB. Use
    /// [`align`](Self::align) to choose something else.
    pub fn canonical(mut self, canonical: bool) -> Self {
        self.canonical = canonical;
        self
    }

    /// Placement alignment: a power of two ≥ 4096. Defaults to 64 KiB.
    pub fn align(mut self, align: u64) -> Self {
        self.align = Some(align);
        self
    }

    /// Block digests (spec §6.2) on every object this writer writes, in
    /// windows of `size` bytes. Needs `.canonical(false)`.
    pub fn blocks(mut self, size: u64) -> Self {
        self.blocks = Some(size);
        self
    }

    pub fn vocabulary(mut self, vocab: &Vocabulary) -> Self {
        self.vocab = Some(Arc::new(vocab.clone()));
        self
    }

    /// Writes to `path` directly.
    pub fn create(self, path: impl AsRef<Path>) -> Result<Writer> {
        self.build(path.as_ref().to_path_buf(), None)
    }

    /// Writes to a sibling partial file and moves it into place on
    /// [`Writer::finish`]. See [`Writer::publish`].
    pub fn publish(self, path: impl AsRef<Path>) -> Result<Writer> {
        let final_path = path.as_ref().to_path_buf();
        let partial = partial_path(&final_path);
        self.build(partial, Some(final_path))
    }

    /// Adds to an existing `.zt` without rewriting the blobs already in it.
    ///
    /// Writing starts past the end of the file, as spec §2.5 requires. New
    /// blobs are placed at the alignment the file already uses, read back
    /// from its offsets; `.align()` overrides it. This is not atomic: until
    /// [`finish`](Writer::finish) puts a footer at the new end, no reader
    /// will open the file. Canonical form forbids unreferenced blobs, so
    /// this needs `.canonical(false)`.
    pub fn append(self, path: impl AsRef<Path>) -> Result<Writer> {
        if self.canonical {
            return Err(Error::InvalidInput(
                "canonical form is written in one pass; add .canonical(false)".into(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        let vocab = self.vocab.clone().unwrap_or_else(Vocabulary::shared);
        let store = crate::provide::store::Store::index(&path, "zt")?;
        let parsed = crate::format::validate::store(&store, &vocab)?;
        let end = store.len();
        let Some((manifest, placement)) = parsed.manifest else {
            return Err(Error::InvalidInput(format!(
                "{}: a data shard carries no manifest, so there is nothing to add to",
                path.display()
            )));
        };
        drop(store);

        let align = match self.align {
            Some(a) => {
                check_alignment(a)?;
                a
            }
            None => inherited_alignment(&manifest, placement.manifest_at),
        };
        let file = File::options().read(true).write(true).open(&path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.seek(SeekFrom::Start(end))?;
        Ok(Writer {
            out: Some(out),
            path,
            publish_to: None,
            offset: end,
            align,
            canonical: false,
            blocks: self.blocks,
            manifest,
            dedup: HashMap::new(),
            last_name: None,
            open_sink: None,
            vocab,
            appending: true,
        })
    }

    fn build(self, path: PathBuf, publish_to: Option<PathBuf>) -> Result<Writer> {
        let align = match (self.canonical, self.align) {
            (true, None) => ALIGN_CANONICAL,
            (true, Some(a)) if a == ALIGN_CANONICAL => ALIGN_CANONICAL,
            (true, Some(a)) => {
                return Err(Error::InvalidInput(format!(
                    "canonical form places blobs at {ALIGN_CANONICAL}; got align({a}). \
                     Add .canonical(false) to choose your own alignment."
                )))
            }
            (false, Some(a)) => {
                check_alignment(a)?;
                a
            }
            (false, None) => ALIGN_CANONICAL,
        };
        if self.canonical && self.blocks.is_some() {
            return Err(Error::InvalidInput(
                "canonical form carries no block digests; add .canonical(false)".into(),
            ));
        }
        if let Some(0) = self.blocks {
            return Err(Error::InvalidInput("block size must be at least 1".into()));
        }
        let file = File::create(&path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&MAGIC)?;
        Ok(Writer {
            out: Some(out),
            path,
            publish_to,
            offset: MAGIC.len() as u64,
            align,
            canonical: self.canonical,
            blocks: self.blocks,
            manifest: Manifest::default(),
            dedup: HashMap::new(),
            last_name: None,
            open_sink: None,
            vocab: self.vocab.unwrap_or_else(Vocabulary::shared),
            appending: false,
        })
    }
}

/// The alignment a file was written at, read back from its offsets: the
/// largest power of two dividing all of them, clamped to the range the
/// format uses.
fn inherited_alignment(manifest: &Manifest, manifest_at: u64) -> u64 {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let mut common = manifest_at;
    for object in manifest.objects.values() {
        if object.blob.shard.is_none() && object.blob.offset > 0 {
            common = gcd(common, object.blob.offset);
        }
    }
    if common == 0 {
        return ALIGN_FLOOR;
    }
    let power_of_two = 1u64 << common.trailing_zeros().min(63);
    power_of_two.clamp(ALIGN_FLOOR, ALIGN_CANONICAL)
}

fn partial_path(final_path: &Path) -> PathBuf {
    let name = final_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out.zt".to_string());
    final_path.with_file_name(format!(".{name}.{}.partial", std::process::id()))
}

// =======================================================================
// writer
// =======================================================================

pub struct Writer {
    out: Option<BufWriter<File>>,
    path: PathBuf,
    publish_to: Option<PathBuf>,
    offset: u64,
    align: u64,
    canonical: bool,
    blocks: Option<u64>,
    manifest: Manifest,
    /// (digest, length) -> offset of a previously written blob. Hash matches
    /// are confirmed byte-for-byte before sharing.
    dedup: HashMap<(Digest, u64), u64>,
    last_name: Option<String>,
    /// The ticket of the [`Sink`] currently open on this writer, if any.
    open_sink: Option<u64>,
    /// Set when adding to a file this writer did not create.
    appending: bool,
    vocab: Arc<Vocabulary>,
}

impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer")
            .field("path", &self.path)
            .field("canonical", &self.canonical)
            .field("align", &self.align)
            .field("objects", &self.manifest.objects.len())
            .finish()
    }
}

impl Writer {
    /// A canonical-form writer over `path`.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Options::default().create(path)
    }

    /// A canonical-form writer that publishes atomically: bytes go to a
    /// partial file beside `path`, and [`finish`](Self::finish) renames it
    /// into place.
    pub fn publish(path: impl AsRef<Path>) -> Result<Self> {
        Options::default().publish(path)
    }

    /// Adds to an existing `.zt`. See [`Options::append`].
    pub fn append(path: impl AsRef<Path>) -> Result<Self> {
        Options::default().canonical(false).append(path)
    }

    pub fn options() -> Options {
        Options::default()
    }

    /// Sets file-level attributes.
    pub fn set_attributes(&mut self, attributes: cbor::Value) {
        self.manifest.attributes = Some(attributes);
    }

    /// The one-liner: a tensor of one leaf, bytes in hand.
    ///
    /// `data` must be exactly `⌈product(shape) × bits(leaf) / 8⌉` bytes.
    pub fn add(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Vec<u64>>,
        leaf: Leaf,
        data: &[u8],
    ) -> Result<()> {
        self.object(name, |o| o.shape(shape).term(leaf).bytes(data))
    }

    /// Registers an external shard under `name`.
    ///
    /// The name is a label you choose, never a path; see spec §7.1 for its
    /// character set. The identity comes from
    /// [`shard_identity`](crate::read::shard_identity). Canonical form is
    /// single-file, so this needs `.canonical(false)`.
    pub fn add_shard(&mut self, name: impl Into<String>, shard: &Shard) -> Result<()> {
        if self.canonical {
            return Err(Error::InvalidInput(
                "canonical form is single-file; add .canonical(false)".into(),
            ));
        }
        let name = name.into();
        check_shard_name(&name).map_err(invalid)?;
        shard.digest.check().map_err(invalid)?;
        if shard.size < MIN_FILE_LEN {
            return Err(Error::InvalidInput(format!(
                "shard size {} below minimum file size",
                shard.size
            )));
        }
        if let Some(existing) = self.manifest.shards.get(&name) {
            if existing != shard {
                return Err(Error::InvalidInput(format!(
                    "shard {name:?} is already registered with a different identity"
                )));
            }
        }
        self.manifest.shards.insert(name, shard.clone());
        Ok(())
    }

    /// Overlay convenience: references `object` (taken from another file's
    /// manifest) through the shard registered under `shard`, writing
    /// nothing. The object's blob must be local in the source manifest.
    pub fn link(&mut self, name: impl Into<String>, object: &Object, shard: &str) -> Result<()> {
        if object.blob.shard.is_some() {
            return Err(Error::InvalidInput(
                "the object is itself a foreign reference; only local blobs can be linked".into(),
            ));
        }
        let mut blob = object.blob.clone();
        blob.shard = Some(shard.to_string());
        let linked = Object {
            blob,
            ..object.clone()
        };
        self.object(name, |o| o.linked(linked))
    }

    /// Copies every tensor of a [`Source`] into this file. This is the
    /// conversion path for every supported format.
    ///
    /// Reads decoded bytes and writes them raw, so a canonical writer turns
    /// *any* source into a canonical, bit-reproducible `.zt` file. File
    /// attributes are copied unless already set.
    pub fn ingest(&mut self, source: &Source) -> Result<()> {
        if self.manifest.attributes.is_none() {
            self.manifest.attributes = source.attributes().cloned();
        }
        for tensor in source.tensors() {
            let bytes = tensor.bytes()?;
            self.object(tensor.name(), |mut o| {
                o = o.shape(tensor.shape().to_vec());
                if let Some(term) = tensor.term() {
                    o = o.term(term.clone());
                }
                if let Some(layout) = tensor.layout() {
                    o = o.layout(layout);
                }
                if let Some(attrs) = tensor.attributes() {
                    o = o.attributes(attrs.clone());
                }
                o.bytes(&bytes)
            })?;
        }
        Ok(())
    }

    /// Writes the manifest blob and footer, then flushes. A publishing writer
    /// also fsyncs and renames into place. Returns the file size.
    pub fn finish(mut self) -> Result<u64> {
        if self.open_sink.is_some() {
            return Err(Error::InvalidInput(
                "a streamed object is still open; close its sink before finishing".into(),
            ));
        }
        let manifest_bytes = cbor::encode(&self.manifest.to_value())?;
        if manifest_bytes.len() as u64 > MAX_MANIFEST_LEN {
            return Err(Error::InvalidInput("manifest exceeds 1 GiB".into()));
        }
        let manifest_offset = self.aligned_offset()?;
        self.pad_to(manifest_offset)?;
        self.write_bytes(&manifest_bytes)?;

        let mut footer = [0u8; FOOTER_LEN as usize];
        footer[0..8].copy_from_slice(&manifest_offset.to_le_bytes());
        footer[8..16].copy_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
        footer[16..24].copy_from_slice(&xxh3_64(&manifest_bytes).to_le_bytes());
        footer[24..28].copy_from_slice(&VERSION.to_le_bytes());
        footer[32..40].copy_from_slice(&MAGIC);
        self.write_bytes(&footer)?;

        let out = self.out.take().expect("writer is open");
        let file = out.into_inner().map_err(|e| Error::Io(e.into_error()))?;
        file.sync_all().or_else(|e| {
            if self.publish_to.is_some() {
                Err(Error::Io(e))
            } else {
                Ok(())
            }
        })?;
        drop(file);

        if let Some(final_path) = self.publish_to.take() {
            std::fs::rename(&self.path, &final_path)?;
        }
        Ok(self.offset)
    }

    /// Abandons the file: a publishing writer removes its partial, a plain one
    /// removes what it has written, an appending one leaves the file alone.
    pub fn abandon(mut self) {
        if self.appending {
            self.discard_buffer();
            return;
        }
        self.out = None;
        let _ = std::fs::remove_file(&self.path);
        self.publish_to = None;
    }

    fn discard_buffer(&mut self) {
        if let Some(out) = self.out.take() {
            let (_file, _buffered) = out.into_parts();
        }
    }

    // ---- blob placement ----

    fn out(&mut self) -> Result<&mut BufWriter<File>> {
        self.out
            .as_mut()
            .ok_or_else(|| Error::InvalidInput("writer is already finished".into()))
    }

    /// Writes a blob, or shares an existing one when the bytes are identical.
    fn write_or_share_blob(
        &mut self,
        data: &[u8],
        blocks: Option<u64>,
    ) -> Result<(u64, Digest, Option<Blocks>)> {
        self.write_or_share_segments(&[(0, data)], blocks)
    }

    /// Writes planes back to back with canonical padding, hashing as it goes.
    /// Shares an identical blob already written, as `bytes` would.
    fn write_planes(
        &mut self,
        planes: &[&[u8]],
        blocks: Option<u64>,
    ) -> Result<(u64, Digest, Option<Blocks>)> {
        let mut segments = Vec::with_capacity(planes.len());
        let mut cursor = 0u64;
        for plane in planes {
            let at = align_up(cursor, PLANE_ALIGN)
                .ok_or_else(|| Error::InvalidInput("blob size overflow".into()))?;
            segments.push(((at - cursor) as usize, *plane));
            cursor = at + plane.len() as u64;
        }
        self.write_or_share_segments(&segments, blocks)
    }

    /// A blob given as segments, each so many zero bytes then a slice.
    fn write_or_share_segments(
        &mut self,
        segments: &[(usize, &[u8])],
        blocks: Option<u64>,
    ) -> Result<(u64, Digest, Option<Blocks>)> {
        let mut hasher = BlobHasher::new(blocks);
        let mut length = 0u64;
        for &(gap, data) in segments {
            hasher.update(&ZEROS[..gap]);
            hasher.update(data);
            length += (gap + data.len()) as u64;
        }
        let (digest, blocks) = hasher.finish();
        let key = (digest.clone(), length);
        if let Some(&offset) = self.dedup.get(&key) {
            if self.segments_equal(offset, segments)? {
                return Ok((offset, digest, blocks));
            }
        }
        let target = self.reserve_blob()?;
        for &(gap, data) in segments {
            self.write_bytes(&ZEROS[..gap])?;
            self.write_bytes(data)?;
        }
        self.dedup.entry(key).or_insert(target);
        Ok((target, digest, blocks))
    }

    fn segments_equal(&mut self, offset: u64, segments: &[(usize, &[u8])]) -> Result<bool> {
        self.out()?.flush()?;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = Vec::new();
        for &(gap, data) in segments {
            buf.resize(gap + data.len(), 0);
            file.read_exact(&mut buf)?;
            if buf[..gap].iter().any(|&b| b != 0) || buf[gap..] != *data {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether two ranges of the file hold the same bytes.
    fn ranges_equal(&mut self, a: u64, b: u64, len: u64) -> Result<bool> {
        self.out()?.flush()?;
        let mut file = File::open(&self.path)?;
        let chunk = len.min(1 << 20) as usize;
        let (mut x, mut y) = (vec![0u8; chunk], vec![0u8; chunk]);
        let mut done = 0u64;
        while done < len {
            let n = (len - done).min(chunk as u64) as usize;
            file.seek(SeekFrom::Start(a + done))?;
            file.read_exact(&mut x[..n])?;
            file.seek(SeekFrom::Start(b + done))?;
            file.read_exact(&mut y[..n])?;
            if x[..n] != y[..n] {
                return Ok(false);
            }
            done += n as u64;
        }
        Ok(true)
    }

    /// Cuts the file back to `to`, which must be at or past the last
    /// committed blob.
    fn truncate(&mut self, to: u64) -> Result<()> {
        let out = self.out()?;
        out.flush()?;
        out.get_mut().set_len(to)?;
        out.seek(SeekFrom::Start(to))?;
        self.offset = to;
        Ok(())
    }

    fn write_blob(&mut self, data: &[u8]) -> Result<u64> {
        let target = self.reserve_blob()?;
        self.write_bytes(data)?;
        Ok(target)
    }

    /// Advances to the next aligned offset without writing anything.
    fn reserve_blob(&mut self) -> Result<u64> {
        let target = self.aligned_offset()?;
        self.pad_to(target)?;
        Ok(target)
    }

    fn aligned_offset(&self) -> Result<u64> {
        align_up(self.offset, self.align)
            .ok_or_else(|| Error::InvalidInput("file offset overflow".into()))
    }

    fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.out()?.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(())
    }

    fn pad_to(&mut self, target: u64) -> Result<()> {
        let mut gap = target - self.offset;
        while gap > 0 {
            let n = gap.min(ZEROS.len() as u64) as usize;
            self.out()?.write_all(&ZEROS[..n])?;
            gap -= n as u64;
        }
        self.offset = target;
        Ok(())
    }

    // ---- shared checks ----

    fn check_canonical_name(&self, name: &str) -> Result<()> {
        if self.canonical && !name.is_ascii() && !unicode_normalization::is_nfc(name) {
            return Err(Error::InvalidInput(format!(
                "canonical form requires NFC-normalized names, got {name:?}"
            )));
        }
        Ok(())
    }

    fn check_new_object(&self, name: &str, shape: &[u64]) -> Result<()> {
        if self.open_sink.is_some() {
            return Err(Error::InvalidInput(format!(
                "object {name:?} cannot be added while a streamed object is open"
            )));
        }
        check_name(name).map_err(invalid)?;
        self.check_canonical_name(name)?;
        if self.manifest.objects.contains_key(name) {
            return Err(Error::InvalidInput(format!("duplicate object {name:?}")));
        }
        check_shape(shape).map_err(invalid)?;
        if self.canonical {
            if let Some(last) = &self.last_name {
                if name <= last.as_str() {
                    return Err(Error::InvalidInput(format!(
                        "canonical form requires sorted insertion: {name:?} after {last:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn commit(&mut self, name: String, object: Object) {
        self.manifest.objects.insert(name.clone(), object);
        self.last_name = Some(name);
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if self.publish_to.is_some() {
            self.out = None;
            let _ = std::fs::remove_file(&self.path);
            return;
        }
        if self.appending {
            self.discard_buffer();
        }
    }
}

const ZEROS: [u8; 4096] = [0u8; 4096];

// =======================================================================
// object builder
// =======================================================================

enum Payload<'d> {
    /// Bytes in hand, decoded.
    Bytes(&'d [u8]),
    /// One slice per plane of the type, to be laid out canonically.
    Planes(Vec<&'d [u8]>),
    /// Bytes already put through their encoding profile.
    Encoded(Vec<u8>),
    /// A length to be streamed later.
    Length(u64),
    /// A byte range of a registered shard.
    External { shard: String, at: Range<u64> },
    /// A whole object taken from another file's manifest ([`Writer::link`]).
    Linked(Object),
    Missing,
}

impl Payload<'_> {
    fn setter(&self) -> &'static str {
        match self {
            Payload::Bytes(_) | Payload::Encoded(_) => "bytes",
            Payload::Planes(_) => "planes",
            Payload::Length(_) => "length",
            Payload::External { .. } => "external",
            Payload::Linked(_) => "link",
            Payload::Missing => "nothing",
        }
    }
}

/// Describes one object.
///
/// A plain description with no writer behind it: it is handed to
/// [`Writer::object`] or [`Writer::stream`], which is what turns it into
/// bytes.
///
/// ```no_run
/// # use ztensor::{Leaf, Term, Writer};
/// # fn f(w: &mut Writer, codes: &[u8], scales: &[u8], biases: &[u8]) -> ztensor::Result<()> {
/// let term = Term::parse("g64_u4_bf16_b_bf16")?;
/// w.object("q", |o| {
///     o.shape([4096u64, 4096])
///         .term(term)
///         .planes([codes, scales, biases])
/// })
/// # }
/// ```
pub struct ObjectBuilder<'d> {
    shape: Vec<u64>,
    term: Option<Term>,
    layout: Option<String>,
    pairs: Vec<(cbor::Value, cbor::Value)>,
    attributes: Option<cbor::Value>,
    encoding: Option<String>,
    digest: Option<Digest>,
    payload: Payload<'d>,
    conflict: Option<String>,
}

impl<'d> ObjectBuilder<'d> {
    fn new() -> Self {
        ObjectBuilder {
            shape: Vec::new(),
            term: None,
            layout: None,
            pairs: Vec::new(),
            attributes: None,
            encoding: None,
            digest: None,
            payload: Payload::Missing,
            conflict: None,
        }
    }

    pub fn shape(mut self, shape: impl Into<Vec<u64>>) -> Self {
        self.shape = shape.into();
        self
    }

    /// The type: a [`Leaf`] or a parsed [`Term`].
    pub fn term(mut self, term: impl Into<Term>) -> Self {
        self.term = Some(term.into());
        self
    }

    /// A named layout. Absent means the canonical layout.
    pub fn layout(mut self, layout: impl Into<String>) -> Self {
        self.layout = Some(layout.into());
        self
    }

    /// One object-level attribute. Not to be mixed with
    /// [`attributes`](Self::attributes).
    pub fn attr(mut self, key: impl Into<String>, value: impl Into<cbor::Value>) -> Self {
        self.pairs.push((cbor::Value::Text(key.into()), value.into()));
        self
    }

    /// Object-level attributes, wholesale. Not to be mixed with
    /// [`attr`](Self::attr).
    pub fn attributes(mut self, attributes: cbor::Value) -> Self {
        self.attributes = Some(attributes);
        self
    }

    /// Stores the blob through an encoding profile. Canonical form is raw by
    /// definition, so this needs `.canonical(false)`.
    pub fn encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = Some(encoding.into());
        self
    }

    /// The digest of an [`external`](Self::external) blob whose bytes this
    /// writer will not see. Everything this writer writes is hashed as it
    /// goes.
    pub fn digest(mut self, digest: Digest) -> Self {
        self.digest = Some(digest);
        self
    }

    /// The decoded bytes, laid out already.
    pub fn bytes(self, data: &'d [u8]) -> Self {
        self.payload(Payload::Bytes(data))
    }

    /// One slice per plane of the type, in canonical order; the writer lays
    /// them out with the canonical padding between them. Needs a term and no
    /// named layout.
    pub fn planes(self, planes: impl IntoIterator<Item = &'d [u8]>) -> Self {
        self.payload(Payload::Planes(planes.into_iter().collect()))
    }

    /// The blob's decoded byte length, to be streamed. See [`Writer::stream`].
    pub fn length(self, length: u64) -> Self {
        self.payload(Payload::Length(length))
    }

    /// A byte range of a registered shard: nothing is written.
    pub fn external(self, shard: impl Into<String>, bytes: Range<u64>) -> Self {
        self.payload(Payload::External {
            shard: shard.into(),
            at: bytes,
        })
    }

    fn linked(self, object: Object) -> Self {
        self.payload(Payload::Linked(object))
    }

    fn payload(mut self, payload: Payload<'d>) -> Self {
        match &self.payload {
            Payload::Missing => self.payload = payload,
            already => {
                let (first, second) = (already.setter(), payload.setter());
                self.conflict.get_or_insert_with(|| {
                    format!("payload already set by `{first}`, then `{second}`")
                });
            }
        }
        self
    }
}

/// Validates a description against the writer and produces the object with
/// its blob offset left at zero, plus the payload to write.
fn build<'d>(writer: &Writer, name: &str, builder: ObjectBuilder<'d>) -> Result<(Object, Payload<'d>)> {
    let ObjectBuilder {
        shape,
        term,
        layout,
        pairs,
        attributes,
        encoding,
        digest,
        payload,
        conflict,
    } = builder;
    let bad = |detail: String| Error::InvalidInput(format!("object {name:?}: {detail}"));
    if let Some(conflict) = conflict {
        return Err(bad(conflict));
    }
    if let Payload::Linked(object) = payload {
        writer.check_new_object(name, &object.shape)?;
        validate_external(&writer.manifest, &object.blob)?;
        validate_object(writer, name, &object)?;
        return Ok((object, Payload::Missing));
    }
    writer.check_new_object(name, &shape)?;
    let attributes = match attributes {
        Some(_) if !pairs.is_empty() => {
            return Err(bad("attributes given both one by one and wholesale".into()))
        }
        Some(cbor::Value::Map(pairs)) => pairs,
        Some(other) => return Err(bad(format!("attributes must be a map, got {other:?}"))),
        None => pairs,
    };
    let attributes = (!attributes.is_empty()).then_some(cbor::Value::Map(attributes));
    if digest.is_some() && !matches!(payload, Payload::External { .. }) {
        return Err(bad(
            "only an external blob takes a digest; this writer hashes what it writes".into(),
        ));
    }
    if encoding.is_some() && writer.canonical {
        return Err(bad("canonical form forbids encoded blobs; add .canonical(false)".into()));
    }

    let (blob, payload) = match payload {
        Payload::Missing => return Err(bad("no bytes, planes, length, or external blob".into())),
        Payload::Linked(_) | Payload::Encoded(_) => unreachable!("never set by a builder"),
        Payload::External { shard, at } => {
            if encoding.is_some() {
                return Err(bad(
                    "an encoded external blob carries a decoded length only its own manifest \
                     knows; link it with `Writer::link`"
                        .into(),
                ));
            }
            let Some(length) = at.end.checked_sub(at.start) else {
                return Err(bad(format!("range {}..{} ends before it starts", at.start, at.end)));
            };
            let blob = Blob {
                shard: Some(shard),
                offset: at.start,
                length,
                encoding: None,
                decoded_length: None,
                digest,
                blocks: None,
            };
            validate_external(&writer.manifest, &blob)?;
            (blob, Payload::Missing)
        }
        Payload::Length(length) => {
            if encoding.is_some() {
                return Err(bad("a streamed blob is written raw; encode bytes in hand".into()));
            }
            (Blob::local(0, length), Payload::Length(length))
        }
        Payload::Planes(planes) => {
            if layout.is_some() {
                return Err(bad("planes are laid out canonically; a named layout takes bytes".into()));
            }
            if encoding.is_some() {
                return Err(bad("planes are written raw; encode bytes in hand".into()));
            }
            let Some(term) = &term else {
                return Err(bad("planes need a type to lay them out".into()));
            };
            let expected = term.planes(&shape).map_err(invalid)?;
            if expected.len() != planes.len() {
                return Err(bad(format!(
                    "the type has {} planes, {} given",
                    expected.len(),
                    planes.len()
                )));
            }
            for (plane, data) in expected.iter().zip(&planes) {
                if data.len() as u64 != plane.len {
                    return Err(bad(format!(
                        "plane {:?} is {} bytes, {} given",
                        plane.path,
                        plane.len,
                        data.len()
                    )));
                }
            }
            let length = term.canonical_size(&shape).map_err(invalid)?;
            (Blob::local(0, length), Payload::Planes(planes))
        }
        Payload::Bytes(data) => match encoding {
            None => (Blob::local(0, data.len() as u64), Payload::Bytes(data)),
            Some(id) => {
                let profile = writer.vocab.encoding(&id).ok_or_else(|| {
                    Error::Unsupported(format!("encoding profile {id:?} is not registered"))
                })?;
                let stored = profile.encode(data)?;
                let mut hasher = BlobHasher::new(writer.blocks);
                hasher.update(data);
                let (digest, blocks) = hasher.finish();
                let blob = Blob {
                    shard: None,
                    offset: 0,
                    length: stored.len() as u64,
                    encoding: Some(id),
                    decoded_length: Some(data.len() as u64),
                    digest: Some(digest),
                    blocks,
                };
                (blob, Payload::Encoded(stored))
            }
        },
    };

    let object = Object {
        shape,
        term,
        layout,
        attributes,
        blob,
    };
    validate_object(writer, name, &object)?;
    Ok((object, payload))
}

/// What a reader would check of the object: it says what its bytes are, its
/// attributes are well-formed, and the size equation and layout rules hold.
fn validate_object(writer: &Writer, name: &str, object: &Object) -> Result<()> {
    if object.term.is_none() && object.layout.is_none() {
        return Err(Error::InvalidInput(format!(
            "object {name:?}: no type and no layout to define its values"
        )));
    }
    if let Some(attributes) = &object.attributes {
        check_attributes(attributes).map_err(invalid)?;
    }
    match &object.layout {
        None => {
            let expected = object.canonical_size().map_err(invalid)?;
            if object.blob.decoded_size() != expected {
                return Err(Error::InvalidInput(format!(
                    "object {name:?}: {} bytes given, its shape and type take {expected}",
                    object.blob.decoded_size()
                )));
            }
        }
        Some(layout) => {
            if let Some(profile) = writer.vocab.layout(layout) {
                profile.validate(name, object).map_err(invalid)?;
            }
        }
    }
    Ok(())
}

fn validate_external(manifest: &Manifest, blob: &Blob) -> Result<()> {
    let Some(sname) = &blob.shard else {
        return Err(Error::InvalidInput(
            "an external blob names no shard".into(),
        ));
    };
    let shard = manifest.shards.get(sname).ok_or_else(|| {
        Error::InvalidInput(format!("unregistered shard {sname:?}"))
    })?;
    if !blob.offset.is_multiple_of(ALIGN_FLOOR) || blob.offset < ALIGN_FLOOR {
        return Err(Error::InvalidInput(format!(
            "offset {} violates the {ALIGN_FLOOR} floor",
            blob.offset
        )));
    }
    let region_end = shard.size - FOOTER_LEN;
    if blob
        .offset
        .checked_add(blob.length)
        .is_none_or(|e| e > region_end)
    {
        return Err(Error::InvalidInput(format!(
            "blob outside shard {sname:?}'s data region"
        )));
    }
    blob.check().map_err(invalid)
}

impl Writer {
    /// Writes one object, described by `describe`. The bytes must be in hand
    /// (or external); see [`stream`](Self::stream) otherwise.
    pub fn object<'d>(
        &mut self,
        name: impl Into<String>,
        describe: impl FnOnce(ObjectBuilder<'d>) -> ObjectBuilder<'d>,
    ) -> Result<()> {
        let name = name.into();
        let (mut object, payload) = build(self, &name, describe(ObjectBuilder::new()))?;
        let written = match payload {
            Payload::Bytes(data) => Some(self.write_or_share_blob(data, self.blocks)?),
            Payload::Planes(planes) => Some(self.write_planes(&planes, self.blocks)?),
            Payload::Encoded(stored) => {
                object.blob.offset = self.write_blob(&stored)?;
                None
            }
            Payload::Length(_) => {
                return Err(Error::InvalidInput(format!(
                    "object {name:?} declares a length; use .stream() instead"
                )));
            }
            Payload::Missing | Payload::External { .. } | Payload::Linked(_) => None,
        };
        if let Some((offset, digest, blocks)) = written {
            object.blob.offset = offset;
            object.blob.digest = Some(digest);
            object.blob.blocks = blocks;
        }
        self.commit(name, object);
        Ok(())
    }

    /// Opens an object for streaming. The description must give the blob's
    /// [`length`](ObjectBuilder::length).
    ///
    /// The returned [`Sink`] is a token, not a borrow: it is passed back to
    /// [`Sink::write`] and consumed by [`Sink::close`]. For an
    /// [`io::Write`](std::io::Write), see [`Sink::attach`].
    pub fn stream<'d>(
        &mut self,
        name: impl Into<String>,
        describe: impl FnOnce(ObjectBuilder<'d>) -> ObjectBuilder<'d>,
    ) -> Result<Sink> {
        let name = name.into();
        let (object, payload) = build(self, &name, describe(ObjectBuilder::new()))?;
        let Payload::Length(length) = payload else {
            return Err(Error::InvalidInput(format!(
                "object {name:?}: streaming declares the blob with .length(); \
                 use .object() for bytes in hand"
            )));
        };
        let ticket = NEXT_SINK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.open_sink = Some(ticket);
        Ok(Sink {
            ticket,
            name,
            object,
            declared: length,
            written: 0,
            hasher: BlobHasher::new(self.blocks),
            started: false,
        })
    }
}

/// Hashes a blob as it is written: the whole-blob xxh3 and, when asked, one
/// xxh3 per block window.
struct BlobHasher {
    whole: Hasher,
    block: Option<(u64, Hasher, u64, Vec<Vec<u8>>)>,
}

impl BlobHasher {
    fn new(blocks: Option<u64>) -> Self {
        BlobHasher {
            whole: Hasher::new(DigestAlgorithm::Xxh3),
            block: blocks.map(|size| (size, Hasher::new(DigestAlgorithm::Xxh3), 0, Vec::new())),
        }
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.whole.update(bytes);
        let Some((size, hasher, filled, digests)) = &mut self.block else {
            return;
        };
        while !bytes.is_empty() {
            let room = (*size - *filled) as usize;
            let take = room.min(bytes.len());
            hasher.update(&bytes[..take]);
            *filled += take as u64;
            bytes = &bytes[take..];
            if *filled == *size {
                let full = std::mem::replace(hasher, Hasher::new(DigestAlgorithm::Xxh3));
                digests.push(full.finish().value);
                *filled = 0;
            }
        }
    }

    fn finish(self) -> (Digest, Option<Blocks>) {
        let blocks = self.block.map(|(size, hasher, filled, mut digests)| {
            if filled > 0 {
                digests.push(hasher.finish().value);
            }
            Blocks { size, digests }
        });
        (self.whole.finish(), blocks)
    }
}

// =======================================================================
// streaming
// =======================================================================

/// An open streamed object.
///
/// Dropping without [`close`](Self::close) leaves the object out of the
/// manifest and the writer refusing further objects.
pub struct Sink {
    ticket: u64,
    name: String,
    object: Object,
    declared: u64,
    written: u64,
    hasher: BlobHasher,
    started: bool,
}

impl std::fmt::Debug for Sink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sink")
            .field("object", &self.name)
            .field("written", &self.written)
            .field("declared", &self.declared)
            .finish()
    }
}

impl Sink {
    /// Bytes written so far.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Bytes still to write.
    pub fn remaining(&self) -> u64 {
        self.declared - self.written
    }

    /// Borrows this sink and its writer together as an
    /// [`io::Write`](std::io::Write).
    pub fn attach<'a>(&'a mut self, writer: &'a mut Writer) -> Attached<'a> {
        Attached { sink: self, writer }
    }

    /// Appends bytes. The first call places the blob at the next aligned
    /// offset. Writing past the declared length is an error.
    pub fn write(&mut self, writer: &mut Writer, chunk: &[u8]) -> Result<()> {
        self.check_owner(writer)?;
        if !self.started {
            self.object.blob.offset = writer.reserve_blob()?;
            self.started = true;
        }
        let end = self
            .written
            .checked_add(chunk.len() as u64)
            .filter(|&e| e <= self.declared)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "object {:?}: {} bytes written into a blob declared as {}",
                    self.name,
                    self.written + chunk.len() as u64,
                    self.declared
                ))
            })?;
        writer.write_bytes(chunk)?;
        self.hasher.update(chunk);
        self.written = end;
        Ok(())
    }

    /// Completes the object and adds it to the manifest. A blob identical to
    /// one already written is dropped in favour of sharing it, as `bytes`
    /// would have.
    pub fn close(self, writer: &mut Writer) -> Result<()> {
        self.check_owner(writer)?;
        if self.written < self.declared {
            return Err(Error::InvalidInput(format!(
                "object {:?}: {} of {} bytes written",
                self.name, self.written, self.declared
            )));
        }
        let Sink {
            name,
            mut object,
            declared,
            hasher,
            started,
            ..
        } = self;
        if !started {
            object.blob.offset = writer.reserve_blob()?;
        }
        let (digest, blocks) = hasher.finish();
        let key = (digest.clone(), declared);
        match writer.dedup.get(&key).copied() {
            Some(prev) if writer.ranges_equal(prev, object.blob.offset, declared)? => {
                writer.truncate(object.blob.offset)?;
                object.blob.offset = prev;
            }
            Some(_) => {}
            None => {
                writer.dedup.insert(key, object.blob.offset);
            }
        }
        object.blob.digest = Some(digest);
        object.blob.blocks = blocks;
        writer.commit(name, object);
        writer.open_sink = None;
        Ok(())
    }

    fn check_owner(&self, writer: &Writer) -> Result<()> {
        if writer.open_sink != Some(self.ticket) {
            return Err(Error::InvalidInput(format!(
                "object {:?}: this sink is not the one open on that writer",
                self.name
            )));
        }
        Ok(())
    }
}

/// A [`Sink`] and its [`Writer`], borrowed together as an
/// [`io::Write`](std::io::Write).
pub struct Attached<'a> {
    sink: &'a mut Sink,
    writer: &'a mut Writer,
}

impl std::io::Write for Attached<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.sink
            .write(self.writer, buf)
            .map(|()| buf.len())
            .map_err(std::io::Error::other)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

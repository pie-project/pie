//! The frozen layers: L0 container and L1 manifest.
//!
//! This is the half of zTensor that a `.zt` file is, as opposed to the half
//! that reads or writes one. The container frame (magic, footer, the alignment
//! floor), the manifest schema and its CBOR mapping, the type grammar, and the
//! rules that decide whether bytes are a conforming file — spec §2, §3, §4 and
//! §6.4 — all live here and nowhere else.
//!
//! Nothing in this module opens a file or knows what a mapping is. That is
//! deliberate: these are the definitions a second implementation would have to
//! agree with, so they are kept where nothing about *this* implementation can
//! leak into them.
//!
//! Everything here is the on-disk structure, unresolved. A [`Blob`]'s `shard`
//! names *this file's* shard table, and an offset means an offset into
//! whichever file that name resolves to. The manifest is a claim about one
//! container, not an address a consumer can use directly.
//!
//! Turning those claims into addresses is
//! [`Catalog`](crate::provide::Catalog)'s job, and it is the reason the two are
//! different types: a catalog can span files that never heard of each other,
//! which no single manifest could honestly describe.

pub mod cbor;
pub mod term;
pub mod validate;

use std::collections::BTreeMap;

use crate::error::{Error, Result, Rule};
use crate::format::cbor::Value;
pub use crate::format::term::{Group, Leaf, Offset, Plane, Term, PLANE_ALIGN};

/// Magic bytes at offset 0 and at the end of the footer (spec §2.2).
pub const MAGIC: [u8; 8] = [0x89, b'Z', b'T', b'2', 0x0d, 0x0a, 0x1a, 0x0a];
/// Footer version integer defined by this implementation (spec §2.3).
pub const VERSION: u32 = 3;
/// Fixed footer size in bytes.
pub const FOOTER_LEN: u64 = 40;
/// Alignment floor: every blob offset is a multiple of this (spec §2.4).
pub const ALIGN_FLOOR: u64 = 4096;
/// Canonical placement alignment (spec §6.4).
pub const ALIGN_CANONICAL: u64 = 65536;
/// Manifest size cap (spec §3.1).
pub const MAX_MANIFEST_LEN: u64 = 1 << 30;
/// Maximum name length in bytes (spec §3.5).
pub const MAX_NAME_LEN: usize = 1024;
/// Maximum shard name length in bytes (spec §7.1).
pub const MAX_SHARD_NAME: usize = 64;
/// Maximum shape rank (spec §3.3).
pub const MAX_RANK: usize = 64;
/// Minimum container size: header magic plus footer (spec §2.1).
pub const MIN_FILE_LEN: u64 = MAGIC.len() as u64 + FOOTER_LEN;

// =======================================================================
// digests
// =======================================================================

/// A digest algorithm this implementation can compute (spec §6.1).
///
/// `Xxh3` is the minimum every reader must have and is what canonical form
/// uses: it detects corruption at memory speed. `Sha256` costs more and buys
/// something different: it is not invertible, so a root manifest whose shard
/// digests are `Sha256` commits to every shard byte, and one signature over
/// that root covers the whole model (§6.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DigestAlgorithm {
    Xxh3,
    Sha256,
}

impl DigestAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            DigestAlgorithm::Xxh3 => "xxh3",
            DigestAlgorithm::Sha256 => "sha256",
        }
    }

    /// Digest width in bytes.
    pub fn width(self) -> usize {
        match self {
            DigestAlgorithm::Xxh3 => 8,
            DigestAlgorithm::Sha256 => 32,
        }
    }

    /// The algorithm a manifest names, or `Unsupported` for one this build
    /// cannot compute: the file may be valid and simply newer than the reader.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "xxh3" => Ok(DigestAlgorithm::Xxh3),
            "sha256" => Ok(DigestAlgorithm::Sha256),
            _ => Err(Error::Unsupported(format!(
                "digest algorithm {name:?}; this build computes xxh3 and sha256"
            ))),
        }
    }

    /// The digest of `bytes`.
    pub fn digest(self, bytes: &[u8]) -> Digest {
        let mut hasher = Hasher::new(self);
        hasher.update(bytes);
        hasher.finish()
    }
}

/// A digest as the manifest stores it: an algorithm name and the raw bytes
/// of its width (spec §6.1). Displays as `algo:hex`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Digest {
    pub algorithm: String,
    pub value: Vec<u8>,
}

impl Digest {
    pub fn new(algorithm: DigestAlgorithm, value: Vec<u8>) -> Self {
        Digest {
            algorithm: algorithm.as_str().to_string(),
            value,
        }
    }

    /// The algorithm, when this build knows it.
    pub fn algorithm(&self) -> Result<DigestAlgorithm> {
        DigestAlgorithm::parse(&self.algorithm)
    }

    pub fn hex(&self) -> String {
        self.value.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Schema rules: a lowercase-alphanumeric algorithm name, a non-empty
    /// value, and the right width when the algorithm is one this build knows.
    pub(crate) fn check(&self) -> Result<()> {
        if self.algorithm.is_empty()
            || !self
                .algorithm
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
            || self.value.is_empty()
        {
            return Err(Error::reject(
                Rule::Schema,
                format!("malformed digest {self}"),
            ));
        }
        if let Ok(algo) = DigestAlgorithm::parse(&self.algorithm) {
            if self.value.len() != algo.width() {
                return Err(Error::reject(
                    Rule::Schema,
                    format!(
                        "{} digest must be {} bytes, got {}",
                        self.algorithm,
                        algo.width(),
                        self.value.len()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Recomputes this digest over `bytes` and reports whether it matches.
    pub fn matches(&self, bytes: &[u8]) -> Result<bool> {
        Ok(self.algorithm()?.digest(bytes).value == self.value)
    }

    fn to_value(&self) -> Value {
        Value::Map(vec![
            (text("algorithm"), text(&self.algorithm)),
            (text("value"), Value::Bytes(self.value.clone())),
        ])
    }

    fn from_value(v: &Value) -> Result<Self> {
        let m = v.map_or("digest")?;
        let mut algorithm = None;
        let mut value = None;
        for (k, val) in m {
            match k.as_text() {
                Some("algorithm") => algorithm = Some(val.text_or("digest.algorithm")?.to_string()),
                Some("value") => {
                    value = Some(match val {
                        Value::Bytes(b) => b.clone(),
                        _ => return Err(Error::reject(Rule::Schema, "digest.value must be bytes")),
                    })
                }
                _ => {}
            }
        }
        let (Some(algorithm), Some(value)) = (algorithm, value) else {
            return missing("digest", "algorithm/value");
        };
        let d = Digest { algorithm, value };
        d.check()?;
        Ok(d)
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.algorithm, self.hex())
    }
}

impl std::fmt::Debug for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Block digests of a blob's decoded bytes (spec §6.2): one per `size`-byte
/// window, under the blob's own digest algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blocks {
    pub size: u64,
    pub digests: Vec<Vec<u8>>,
}

impl Blocks {
    /// How many blocks `decoded` bytes make.
    pub fn count(size: u64, decoded: u64) -> u64 {
        decoded.div_ceil(size)
    }

    /// The byte range of block `which` within `decoded` bytes.
    pub fn span(&self, which: u64, decoded: u64) -> Option<std::ops::Range<u64>> {
        let start = which.checked_mul(self.size)?;
        (start < decoded).then(|| start..start.saturating_add(self.size).min(decoded))
    }

    fn check(&self, digest: &Digest, decoded: u64) -> Result<()> {
        if self.size == 0 {
            return Err(Error::reject(Rule::Schema, "blocks.size must be at least 1"));
        }
        let expected = Self::count(self.size, decoded);
        if self.digests.len() as u64 != expected {
            return Err(Error::reject(
                Rule::Schema,
                format!(
                    "blocks holds {} digests, {expected} blocks of {} cover {decoded} bytes",
                    self.digests.len(),
                    self.size
                ),
            ));
        }
        if let Ok(algo) = digest.algorithm() {
            if self.digests.iter().any(|d| d.len() != algo.width()) {
                return Err(Error::reject(
                    Rule::Schema,
                    format!("every block digest must be {} bytes", algo.width()),
                ));
            }
        }
        Ok(())
    }

    fn to_value(&self) -> Value {
        Value::Map(vec![
            (text("size"), Value::Uint(self.size)),
            (
                text("digests"),
                Value::Array(self.digests.iter().cloned().map(Value::Bytes).collect()),
            ),
        ])
    }

    fn from_value(v: &Value) -> Result<Self> {
        let m = v.map_or("blocks")?;
        let mut size = None;
        let mut digests = None;
        for (k, val) in m {
            match k.as_text() {
                Some("size") => size = Some(val.u64_or("blocks.size")?),
                Some("digests") => {
                    digests = Some(
                        val.array_or("blocks.digests")?
                            .iter()
                            .map(|d| match d {
                                Value::Bytes(b) => Ok(b.clone()),
                                _ => Err(Error::reject(Rule::Schema, "block digests must be bytes")),
                            })
                            .collect::<Result<Vec<_>>>()?,
                    )
                }
                _ => {}
            }
        }
        let (Some(size), Some(digests)) = (size, digests) else {
            return missing("blocks", "size/digests");
        };
        Ok(Blocks { size, digests })
    }
}

/// Computes a digest a chunk at a time, so a whole-file digest never needs the
/// whole file in memory.
pub(crate) enum Hasher {
    Xxh3(Box<xxhash_rust::xxh3::Xxh3>),
    Sha256(sha2::Sha256),
}

impl Hasher {
    pub(crate) fn new(algo: DigestAlgorithm) -> Self {
        match algo {
            DigestAlgorithm::Xxh3 => Hasher::Xxh3(Box::default()),
            DigestAlgorithm::Sha256 => Hasher::Sha256(<sha2::Sha256 as sha2::Digest>::new()),
        }
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        match self {
            Hasher::Xxh3(h) => h.update(bytes),
            Hasher::Sha256(h) => sha2::Digest::update(h, bytes),
        }
    }

    pub(crate) fn finish(self) -> Digest {
        match self {
            Hasher::Xxh3(h) => Digest::new(DigestAlgorithm::Xxh3, h.digest().to_be_bytes().to_vec()),
            Hasher::Sha256(h) => {
                Digest::new(DigestAlgorithm::Sha256, sha2::Digest::finalize(h).to_vec())
            }
        }
    }
}

// =======================================================================
// manifest
// =======================================================================

/// An object's bytes (spec §3.4): one range of one file, plus how they are
/// stored and what they hash to.
///
/// `shard` names an entry in the containing manifest's shard table. `None`
/// means the containing file, which is the common case and the one that costs
/// nothing to say. A name is only a label; turning it into bytes is the
/// transport's job, and only after that is there a
/// [`StoreId`](crate::StoreId).
#[derive(Debug, Clone, PartialEq)]
pub struct Blob {
    pub shard: Option<String>,
    pub offset: u64,
    pub length: u64,
    /// Encoding profile id; `None` means raw.
    pub encoding: Option<String>,
    /// Required iff `encoding` is present.
    pub decoded_length: Option<u64>,
    /// Over the decoded bytes.
    pub digest: Option<Digest>,
    /// Never without `digest`.
    pub blocks: Option<Blocks>,
}

impl Blob {
    /// A raw, local blob.
    pub fn local(offset: u64, length: u64) -> Self {
        Blob {
            shard: None,
            offset,
            length,
            encoding: None,
            decoded_length: None,
            digest: None,
            blocks: None,
        }
    }

    /// Decoded byte size: `length` for raw, `decoded_length` when encoded.
    pub fn decoded_size(&self) -> u64 {
        match self.encoding {
            None => self.length,
            Some(_) => self.decoded_length.unwrap_or(0),
        }
    }

    /// Schema rules that need no file: the encoding/decoded_length pairing,
    /// the blocks/digest pairing, digest widths.
    pub(crate) fn check(&self) -> Result<()> {
        if self.encoding.is_some() != self.decoded_length.is_some() {
            return Err(Error::reject(
                Rule::Schema,
                "'decoded_length' is required iff 'encoding' is present",
            ));
        }
        if let Some(d) = &self.digest {
            d.check()?;
        }
        match (&self.blocks, &self.digest) {
            (Some(_), None) => {
                return Err(Error::reject(Rule::Schema, "'blocks' requires 'digest'"))
            }
            (Some(blocks), Some(digest)) => blocks.check(digest, self.decoded_size())?,
            _ => {}
        }
        Ok(())
    }

    fn to_value(&self) -> Value {
        let mut m = vec![
            (text("offset"), Value::Uint(self.offset)),
            (text("length"), Value::Uint(self.length)),
        ];
        if let Some(shard) = &self.shard {
            m.push((text("shard"), text(shard)));
        }
        if let Some(enc) = &self.encoding {
            m.push((text("encoding"), text(enc)));
        }
        if let Some(dl) = self.decoded_length {
            m.push((text("decoded_length"), Value::Uint(dl)));
        }
        if let Some(d) = &self.digest {
            m.push((text("digest"), d.to_value()));
        }
        if let Some(b) = &self.blocks {
            m.push((text("blocks"), b.to_value()));
        }
        Value::Map(m)
    }

    fn from_value(v: &Value) -> Result<Blob> {
        let entries = v.map_or("blob")?;
        let mut blob = Blob::local(0, 0);
        let (mut offset, mut length) = (None, None);
        for (k, val) in entries {
            match k.as_text() {
                Some("offset") => offset = Some(val.u64_or("offset")?),
                Some("length") => length = Some(val.u64_or("length")?),
                Some("shard") => {
                    let name = val.text_or("shard")?;
                    check_shard_name(name)?;
                    blob.shard = Some(name.to_string());
                }
                Some("encoding") => blob.encoding = Some(val.text_or("encoding")?.to_string()),
                Some("decoded_length") => {
                    blob.decoded_length = Some(val.u64_or("decoded_length")?)
                }
                Some("digest") => blob.digest = Some(Digest::from_value(val)?),
                Some("blocks") => blob.blocks = Some(Blocks::from_value(val)?),
                _ => {}
            }
        }
        let (Some(offset), Some(length)) = (offset, length) else {
            return missing("blob", "offset/length");
        };
        blob.offset = offset;
        blob.length = length;
        blob.check()?;
        Ok(blob)
    }
}

/// A named object (spec §3.3): one tensor, one blob.
#[derive(Debug, Clone, PartialEq)]
pub struct Object {
    pub shape: Vec<u64>,
    /// The `type` field. Absent only under a layout that defines the values
    /// itself (§5.2).
    pub term: Option<Term>,
    /// Absent ⇒ canonical layout (§5.1).
    pub layout: Option<String>,
    pub attributes: Option<Value>,
    pub blob: Blob,
}

impl Object {
    /// Element count: product of dimensions; empty shape is a scalar (1).
    pub fn num_elements(&self) -> Result<u64> {
        check_shape(&self.shape)
    }

    /// The term, or `Unsupported` when the object has none.
    pub fn term(&self) -> Result<&Term> {
        self.term.as_ref().ok_or_else(|| {
            Error::Unsupported(format!(
                "no type: layout {:?} defines the values itself",
                self.layout.as_deref().unwrap_or("?")
            ))
        })
    }

    /// The planes of this object under the canonical layout. `Unsupported`
    /// when a named layout places the bytes instead.
    pub fn planes(&self) -> Result<Vec<Plane>> {
        canonical_term(self.term.as_ref(), self.layout.as_deref())?.planes(&self.shape)
    }

    /// The decoded size the canonical layout requires (§5.1 rule 4).
    pub fn canonical_size(&self) -> Result<u64> {
        canonical_term(self.term.as_ref(), self.layout.as_deref())?.canonical_size(&self.shape)
    }
}

/// The term that lays out an object's bytes, when the canonical layout
/// applies: `Unsupported` under a named layout or without a type.
pub(crate) fn canonical_term<'a>(term: Option<&'a Term>, layout: Option<&str>) -> Result<&'a Term> {
    if let Some(layout) = layout {
        return Err(Error::Unsupported(format!(
            "layout {layout:?} places the planes; the canonical rule does not apply"
        )));
    }
    term.ok_or_else(|| Error::Unsupported("no type".into()))
}

/// Shape rules (spec §3.3): rank within the limit and a product that fits
/// `u64`. Returns the element count; an empty shape is a scalar (1).
pub fn check_shape(shape: &[u64]) -> Result<u64> {
    if shape.len() > MAX_RANK {
        return Err(Error::reject(
            Rule::Shape,
            format!("rank {} exceeds {MAX_RANK}", shape.len()),
        ));
    }
    shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::reject(Rule::Shape, "shape product overflows u64"))
}

/// `n` rounded up to a multiple of `align`, a power of two; `None` past
/// the end of `u64`.
pub(crate) fn align_up(n: u64, align: u64) -> Option<u64> {
    n.checked_add(align - 1).map(|v| v & !(align - 1))
}

/// Shard identity: size and digest, never a location (spec §7.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Shard {
    pub size: u64,
    pub digest: Digest,
}

/// Root manifest (spec §3.2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Manifest {
    pub attributes: Option<Value>,
    /// Keyed by shard name. The containing file is never a key: it is named by
    /// the absence of a name (see [`Blob::shard`]).
    pub shards: BTreeMap<String, Shard>,
    pub objects: BTreeMap<String, Object>,
}

impl Manifest {
    /// Looks up an object by name.
    pub fn object(&self, name: &str) -> Result<&Object> {
        self.objects
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("object {name:?}")))
    }

    /// The content digest of this model (spec §6.5).
    ///
    /// A whole-file hash identifies an *artifact*: those bytes, that
    /// placement. This identifies the *model*, and is defined so that
    /// placement cannot reach it: offsets, lengths, alignment, padding, blob
    /// sharing, encodings, block digests and the shard table are all absent.
    ///
    /// # Errors
    ///
    /// `Unsupported` when an object carries no digest: with none there is
    /// nothing to stand for its content. Canonical form guarantees they are
    /// present (§6.4 rule 4).
    pub fn content_digest(&self, algo: DigestAlgorithm) -> Result<Digest> {
        let mut objects = Vec::with_capacity(self.objects.len());
        for (name, object) in &self.objects {
            let Some(digest) = &object.blob.digest else {
                return Err(Error::Unsupported(format!(
                    "{name:?} carries no digest, so this model has no content digest (spec §6.5)"
                )));
            };
            let mut fields = vec![(
                text("shape"),
                Value::Array(object.shape.iter().copied().map(Value::Uint).collect()),
            )];
            if let Some(term) = &object.term {
                fields.push((text("type"), text(&term.to_string())));
            }
            if let Some(layout) = &object.layout {
                fields.push((text("layout"), text(layout)));
            }
            if let Some(attributes) = &object.attributes {
                fields.push((text("attributes"), attributes.clone()));
            }
            fields.push((text("digest"), digest.to_value()));
            objects.push((text(name), Value::Map(fields)));
        }
        let value = Value::Array(vec![text("zt.content/2"), Value::Map(objects)]);
        Ok(algo.digest(&cbor::encode(&value)?))
    }
}

/// Checks a shard name against §7.1.
///
/// The character set is narrow on purpose. A resolver turns a name into a
/// location, and the conventional ones (Appendix B) use it as a single path
/// component; if the format let a name be `../../etc/passwd`, every consumer
/// would have to sanitize it, and one of them would forget.
pub fn check_shard_name(name: &str) -> Result<()> {
    let bad = |msg: &str| {
        Err(Error::reject(
            Rule::ShardName,
            format!("shard name {name:?}: {msg}"),
        ))
    };
    if name.is_empty() {
        return bad("must not be empty");
    }
    if name.len() > MAX_SHARD_NAME {
        return bad(&format!("longer than {MAX_SHARD_NAME} bytes"));
    }
    if name.starts_with('.') {
        return bad("must not start with '.'");
    }
    if let Some(c) = name
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
    {
        return bad(&format!("contains {c:?}; allowed: A-Z a-z 0-9 . _ -"));
    }
    Ok(())
}

/// Name rules (spec §3.5): non-empty UTF-8 (guaranteed by CBOR decode),
/// ≤ 1024 bytes, no NUL. NFC is a writer duty, not a reader check.
pub(crate) fn check_name(s: &str) -> Result<()> {
    if s.is_empty() || s.len() > MAX_NAME_LEN || s.contains('\0') {
        return Err(Error::reject(Rule::Name, format!("invalid name {s:?}")));
    }
    Ok(())
}

/// Attributes rules (spec §3.1/§3.5): the value MUST be a map whose
/// top-level keys are text obeying the name rules. Nested values are free
/// within the §3.1 type set (the codec already enforces that).
pub(crate) fn check_attributes(v: &Value) -> Result<()> {
    let entries = v
        .as_map()
        .ok_or_else(|| Error::reject(Rule::Schema, "'attributes' must be a map"))?;
    for (k, _) in entries {
        let key = k
            .as_text()
            .ok_or_else(|| Error::reject(Rule::Schema, "attribute keys must be text"))?;
        check_name(key)?;
    }
    Ok(())
}

// =======================================================================
// Manifest <-> CBOR
// =======================================================================

fn missing<T>(what: &str, field: &str) -> Result<T> {
    Err(Error::reject(
        Rule::Schema,
        format!("{what} missing {field:?}"),
    ))
}

impl Value {
    fn text_or(&self, field: &str) -> Result<&str> {
        self.as_text()
            .ok_or_else(|| Error::reject(Rule::Schema, format!("{field:?} must be text")))
    }

    fn u64_or(&self, field: &str) -> Result<u64> {
        self.as_u64().ok_or_else(|| {
            Error::reject(Rule::Schema, format!("{field:?} must be an unsigned int"))
        })
    }

    fn map_or(&self, field: &str) -> Result<&[(Value, Value)]> {
        self.as_map()
            .ok_or_else(|| Error::reject(Rule::Schema, format!("{field:?} must be a map")))
    }

    fn array_or(&self, field: &str) -> Result<&[Value]> {
        self.as_array()
            .ok_or_else(|| Error::reject(Rule::Schema, format!("{field:?} must be an array")))
    }

    fn uints_or(&self, field: &str) -> Result<Vec<u64>> {
        self.array_or(field)?
            .iter()
            .map(|v| v.u64_or(field))
            .collect()
    }
}

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

impl Manifest {
    pub(crate) fn to_value(&self) -> Value {
        let mut root = Vec::new();
        if let Some(attrs) = &self.attributes {
            root.push((text("attributes"), attrs.clone()));
        }
        if !self.shards.is_empty() {
            let shards = self
                .shards
                .iter()
                .map(|(name, s)| {
                    (
                        text(name),
                        Value::Map(vec![
                            (text("size"), Value::Uint(s.size)),
                            (text("digest"), s.digest.to_value()),
                        ]),
                    )
                })
                .collect();
            root.push((text("shards"), Value::Map(shards)));
        }
        let objects = self
            .objects
            .iter()
            .map(|(name, obj)| (text(name), obj.to_value()))
            .collect();
        root.push((text("objects"), Value::Map(objects)));
        Value::Map(root)
    }

    pub(crate) fn from_value(v: Value) -> Result<Manifest> {
        let entries = match v {
            Value::Map(m) => m,
            _ => return Err(Error::reject(Rule::Schema, "manifest root must be a map")),
        };
        let mut manifest = Manifest::default();
        let mut has_objects = false;
        for (k, val) in entries {
            let Some(key) = k.as_text() else {
                continue;
            };
            match key {
                "attributes" => {
                    check_attributes(&val)?;
                    manifest.attributes = Some(val);
                }
                "shards" => manifest.shards = parse_shards(val)?,
                "objects" => {
                    has_objects = true;
                    manifest.objects = parse_objects(val)?;
                }
                _ => {} // unknown fields are ignored (spec §3.1)
            }
        }
        if !has_objects {
            return Err(Error::reject(Rule::Schema, "manifest missing 'objects'"));
        }
        Ok(manifest)
    }
}

fn parse_shards(v: Value) -> Result<BTreeMap<String, Shard>> {
    let entries = v.map_or("shards")?;
    let mut shards = BTreeMap::new();
    for (k, val) in entries {
        let name = k
            .as_text()
            .ok_or_else(|| Error::reject(Rule::Schema, "shard key must be text"))?
            .to_string();
        check_shard_name(&name)?;
        let m = val.map_or("shard entry")?;
        let mut size = None;
        let mut digest = None;
        for (fk, fv) in m {
            match fk.as_text() {
                Some("size") => size = Some(fv.u64_or("shard.size")?),
                Some("digest") => digest = Some(Digest::from_value(fv)?),
                _ => {}
            }
        }
        let (Some(size), Some(digest)) = (size, digest) else {
            return missing("shard entry", "size/digest");
        };
        shards.insert(name, Shard { size, digest });
    }
    Ok(shards)
}

fn parse_objects(v: Value) -> Result<BTreeMap<String, Object>> {
    let entries = v.map_or("objects")?;
    let mut objects = BTreeMap::new();
    for (k, val) in entries {
        let name = k.text_or("object name")?;
        check_name(name)?;
        objects.insert(name.to_string(), Object::from_value(val)?);
    }
    Ok(objects)
}

impl Object {
    fn to_value(&self) -> Value {
        let mut m = vec![(
            text("shape"),
            Value::Array(self.shape.iter().map(|&d| Value::Uint(d)).collect()),
        )];
        if let Some(term) = &self.term {
            m.push((text("type"), text(&term.to_string())));
        }
        if let Some(layout) = &self.layout {
            m.push((text("layout"), text(layout)));
        }
        if let Some(attrs) = &self.attributes {
            m.push((text("attributes"), attrs.clone()));
        }
        m.push((text("blob"), self.blob.to_value()));
        Value::Map(m)
    }

    fn from_value(v: &Value) -> Result<Object> {
        let entries = v.map_or("object")?;
        let mut shape = None;
        let mut term = None;
        let mut layout = None;
        let mut attributes = None;
        let mut blob = None;
        for (k, val) in entries {
            match k.as_text() {
                Some("shape") => shape = Some(val.uints_or("shape")?),
                Some("type") => term = Some(Term::parse(val.text_or("type")?)?),
                Some("layout") => layout = Some(val.text_or("layout")?.to_string()),
                Some("attributes") => {
                    check_attributes(val)?;
                    attributes = Some(val.clone());
                }
                Some("blob") => blob = Some(Blob::from_value(val)?),
                _ => {}
            }
        }
        let (Some(shape), Some(blob)) = (shape, blob) else {
            return missing("object", "shape/blob");
        };
        if term.is_none() && layout.is_none() {
            return Err(Error::reject(
                Rule::Schema,
                "object has no 'type' and no 'layout' to define its values",
            ));
        }
        Ok(Object {
            shape,
            term,
            layout,
            attributes,
            blob,
        })
    }
}

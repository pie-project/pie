//! `pie.serving/1`: the serving-artifact format layer — a stamp in the
//! file's attributes and a placement policy, over an ordinary zTensor v3 file.
//! Every object states its own type and carries its own block digests; the
//! stamp says only which deployment the placement was chosen for.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;

use ztensor::format::cbor::Value;
pub use ztensor::DigestAlgorithm;
use ztensor::{Manifest, Object};

use crate::error::Error;
use crate::file::meta::META_PREFIX;
pub use crate::term::{plane_name, MMA_TILED};

/// The file profile id and version this build implements: the attribute key
/// the stamp lives under. Its presence is what makes a `.zt` a serving
/// artifact rather than an ordinary checkpoint.
pub const PROFILE: &str = "pie.serving/1";

/// Shared prefix of every version of this profile.
pub const PROFILE_FAMILY: &str = "pie.serving/";

/// The revision a human bumps by hand — [`Stamp::layout_revision`]'s value
/// for artifacts this build writes. Bumped when a change to an authored
/// contract or compiled plan moves bytes without moving [`PROFILE`].
pub const LAYOUT_REVISION: u64 = 1;

/// The block this build's digests tile, 64 MiB: the unit a refill reads and
/// hashes. A writer policy, read back per blob from the file.
pub const BLOCK_BYTES: u64 = 64 << 20;

/// The file-level serving facts, as one value (on disk: one attribute,
/// keyed [`PROFILE`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stamp {
    /// The profile id this file states — on disk, the key these live under.
    pub serving: String,
    /// The engine whose kernels these bytes are landed for.
    pub backend: String,
    /// The recipe the planes were compiled under. Object names in the file
    /// are this SKU's weight names.
    pub sku: String,
    /// A judgement a human moves, see [`LAYOUT_REVISION`].
    pub layout_revision: u64,
    /// Every plane this SKU declares a registered adapter bank is absent
    /// from the file, served as `Buffer::zeroed`.
    pub adapters_zeroed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Field {
    Serving,
    Backend,
    Sku,
    LayoutRevision,
    AdaptersZeroed,
}

impl Field {
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Field::Serving => PROFILE,
            Field::Backend => "backend",
            Field::Sku => "sku",
            Field::LayoutRevision => "layout_revision",
            Field::AdaptersZeroed => "adapters_zeroed",
        }
    }

    #[must_use]
    pub fn required() -> &'static [Field] {
        &[
            Field::Serving,
            Field::Backend,
            Field::Sku,
            Field::LayoutRevision,
            Field::AdaptersZeroed,
        ]
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.key())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    pub field: Field,
    pub artifact: String,
    pub deployment: String,
}

impl Mismatch {
    #[must_use]
    pub fn refuse(&self, artifact: &str) -> String {
        format!(
            "checkpoint: the serving artifact {artifact:?} states {} {:?} and this \
             deployment is {:?}. This file is how this machine holds the model, not a \
             cache of a boot, so nothing here rewrites it and nothing here deletes it — \
             run `{}` to write it again from the checkpoint this load names.",
            self.field,
            self.artifact,
            self.deployment,
            rebuild(None),
        )
    }
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} is {:?} in the artifact and {:?} in this deployment",
            self.field, self.artifact, self.deployment
        )
    }
}

#[must_use]
pub fn rebuild(source: Option<&str>) -> String {
    match source {
        Some(source) => format!("pie model import --force {source}"),
        None => "pie model import --force <this deployment's checkpoint>".to_string(),
    }
}

impl Stamp {
    /// The file's [`PROFILE`] attribute: the stamp's members under its key.
    #[must_use]
    pub fn encode(&self) -> Value {
        Value::Map(vec![(
            text(PROFILE),
            Value::Map(vec![
                (text(Field::Backend.key()), text(&self.backend)),
                (text(Field::Sku.key()), text(&self.sku)),
                (
                    text(Field::LayoutRevision.key()),
                    Value::Uint(self.layout_revision),
                ),
                (
                    text(Field::AdaptersZeroed.key()),
                    Value::Bool(self.adapters_zeroed),
                ),
            ]),
        )])
    }

    #[must_use]
    pub fn of(backend: &str, sku: &str) -> Stamp {
        Stamp {
            serving: PROFILE.to_string(),
            backend: backend.to_string(),
            sku: sku.to_string(),
            layout_revision: LAYOUT_REVISION,
            adapters_zeroed: true,
        }
    }

    pub fn decode(attributes: &Value) -> Result<Stamp, Error> {
        let Some(attributes) = attributes.get(PROFILE) else {
            return Err(match stated_profile(attributes) {
                Some(stated) => Error::Unsupported(format!(
                    "the serving artifact states {stated:?} and this build implements \
                     {PROFILE:?}; re-import it rather than repairing it"
                )),
                None => Error::Checkpoint(format!(
                    "this file carries no {PROFILE:?} attribute, so it is an ordinary \
                     checkpoint rather than a serving artifact; {}",
                    rebuild(None),
                )),
            });
        };
        let adapters_zeroed = match attributes.get(Field::AdaptersZeroed.key()) {
            Some(Value::Bool(it)) => *it,
            Some(_) => return Err(malformed(Field::AdaptersZeroed, "is not a boolean")),
            None => return Err(missing(Field::AdaptersZeroed)),
        };
        if !adapters_zeroed {
            return Err(malformed(
                Field::AdaptersZeroed,
                "is false, so a restore would seat an adapter's contents as though they \
                 were a weight",
            ));
        }
        Ok(Stamp {
            serving: PROFILE.to_string(),
            backend: required_text(attributes, Field::Backend)?.to_string(),
            sku: required_text(attributes, Field::Sku)?.to_string(),
            layout_revision: required_uint(attributes, Field::LayoutRevision)?,
            adapters_zeroed,
        })
    }

    pub fn check(&self, deployment: &Stamp) -> Result<(), Mismatch> {
        for field in Field::required() {
            let (artifact, wanted) = (self.say(*field), deployment.say(*field));
            if artifact != wanted {
                return Err(Mismatch {
                    field: *field,
                    artifact,
                    deployment: wanted,
                });
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn say(&self, field: Field) -> String {
        match field {
            Field::Serving => self.serving.clone(),
            Field::Backend => self.backend.clone(),
            Field::Sku => self.sku.clone(),
            Field::LayoutRevision => self.layout_revision.to_string(),
            Field::AdaptersZeroed => self.adapters_zeroed.to_string(),
        }
    }
}

/// `pie.mma_tiled/1`: the canonical planes of a band-padded affine weight,
/// in mma fragment order. The object's shape is the padded rectangle, so the
/// size equation is the canonical one.
struct MmaTiled;

impl ztensor::vocab::Layout for MmaTiled {
    fn id(&self) -> &str {
        MMA_TILED
    }

    fn validate(&self, name: &str, obj: &Object) -> Result<(), ztensor::Error> {
        let fail = |detail: String| {
            Err(ztensor::Error::reject(
                ztensor::Rule::LayoutRule,
                format!("{name:?}: {detail}"),
            ))
        };
        let Some(term) = &obj.term else {
            return fail(format!("{MMA_TILED} requires a type"));
        };
        let attr = |key: &str| obj.attributes.as_ref()?.get(key)?.as_u64();
        if attr("band") != Some(u64::from(dtype::TILED_BAND))
            || attr("step") != Some(u64::from(dtype::TILED_STEP))
        {
            return fail(format!(
                "{MMA_TILED} requires band {} and step {}",
                dtype::TILED_BAND,
                dtype::TILED_STEP
            ));
        }
        let [rows, cols] = obj.shape[..] else {
            return fail(format!("{MMA_TILED} tiles a rank-2 weight"));
        };
        if rows % u64::from(dtype::TILED_BAND) != 0 || cols % u64::from(dtype::TILED_STEP) != 0 {
            return fail(format!(
                "{MMA_TILED} holds {rows}×{cols}, which is not whole bands by whole steps"
            ));
        }
        let expected = term.canonical_size(&obj.shape)?;
        if obj.blob.decoded_size() != expected {
            return Err(ztensor::Error::reject(
                ztensor::Rule::Size,
                format!(
                    "{name:?}: decoded size {} != the {expected} its padded shape and type require",
                    obj.blob.decoded_size()
                ),
            ));
        }
        Ok(())
    }
}

/// The registry a pie reader opens files under: the standard one plus
/// `MmaTiled`.
#[must_use]
pub fn vocabulary() -> ztensor::Vocabulary {
    ztensor::Vocabulary::standard().with_layout(MmaTiled)
}

/// One object's block digests, borrowed from its manifest entry.
#[derive(Debug, Clone, Copy)]
pub struct Blocks<'a> {
    algorithm: DigestAlgorithm,
    blocks: &'a ztensor::Blocks,
    size: u64,
}

impl<'a> Blocks<'a> {
    /// The blocks an object states, or why it states none this build reads.
    pub fn of(name: &str, object: &'a Object) -> Result<Blocks<'a>, Error> {
        let Some(digest) = &object.blob.digest else {
            return Err(Error::Checkpoint(format!(
                "{name:?} carries no digest, so its blocks cannot be checked"
            )));
        };
        let algorithm = digest.algorithm().map_err(Error::from)?;
        let Some(blocks) = &object.blob.blocks else {
            return Err(Error::Checkpoint(format!(
                "{name:?} states no block digests, so no prefix of it can be verified \
                 without hashing the rest"
            )));
        };
        Ok(Blocks {
            algorithm,
            blocks,
            size: object.blob.decoded_size(),
        })
    }

    #[must_use]
    pub fn algorithm(&self) -> DigestAlgorithm {
        self.algorithm
    }

    #[must_use]
    pub fn block_bytes(&self) -> u64 {
        self.blocks.size
    }

    /// The object's decoded size, which the blocks tile exactly.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    #[must_use]
    pub fn count(&self) -> u64 {
        self.blocks.digests.len() as u64
    }

    #[must_use]
    pub fn digest(&self, which: u64) -> Option<&'a [u8]> {
        self.blocks
            .digests
            .get(usize::try_from(which).ok()?)
            .map(Vec::as_slice)
    }

    #[must_use]
    pub fn span(&self, which: u64) -> Option<Range<u64>> {
        self.blocks.span(which, self.size)
    }

    /// Every block, as its blob-local range and its stated digest.
    pub fn iter(&self) -> impl Iterator<Item = (Range<u64>, &'a [u8])> + '_ {
        (0..self.count()).filter_map(|which| Some((self.span(which)?, self.digest(which)?)))
    }
}

/// A block digest computed piecewise, for a reader whose block lands in
/// several destinations.
pub enum Digesting {
    Xxh3(Box<xxhash_rust::xxh3::Xxh3>),
    Sha256(Box<sha2::Sha256>),
}

impl Digesting {
    #[must_use]
    pub fn new(algorithm: DigestAlgorithm) -> Digesting {
        match algorithm {
            DigestAlgorithm::Xxh3 => Digesting::Xxh3(Box::default()),
            DigestAlgorithm::Sha256 => Digesting::Sha256(Box::new(<sha2::Sha256 as sha2::Digest>::new())),
        }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        match self {
            Digesting::Xxh3(h) => h.update(bytes),
            Digesting::Sha256(h) => sha2::Digest::update(h.as_mut(), bytes),
        }
    }

    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        match self {
            Digesting::Xxh3(h) => h.digest().to_be_bytes().to_vec(),
            Digesting::Sha256(h) => sha2::Digest::finalize(*h).to_vec(),
        }
    }
}

/// One serving object's blob: where it is and how long.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span<'a> {
    pub object: &'a str,
    pub shard: Option<&'a str>,
    pub offset: u64,
    pub length: u64,
}

impl Span<'_> {
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.length)
    }
}

/// Whether an object name is a serving object rather than a metadata one.
#[must_use]
pub fn is_serving(name: &str) -> bool {
    !name.starts_with(META_PREFIX)
}

/// The serving sequence, derived by sorting: `(shard index, blob offset)`,
/// ties by name. Position *i* is hotter than *i + 1*.
#[must_use]
pub fn sequence(manifest: &Manifest) -> Vec<&str> {
    spans(manifest).into_iter().map(|span| span.object).collect()
}

/// Every serving object, in sequence order.
#[must_use]
pub fn spans(manifest: &Manifest) -> Vec<Span<'_>> {
    let shards: BTreeMap<&str, usize> = manifest
        .shards
        .keys()
        .enumerate()
        .map(|(at, name)| (name.as_str(), at + 1))
        .collect();
    let mut out: Vec<(usize, u64, Span<'_>)> = manifest
        .objects
        .iter()
        .filter(|(name, _)| is_serving(name))
        .map(|(name, object)| {
            let shard = object.blob.shard.as_deref();
            let index = shard.map_or(0, |it| shards.get(it).copied().unwrap_or(usize::MAX));
            (
                index,
                object.blob.offset,
                Span {
                    object: name.as_str(),
                    shard,
                    offset: object.blob.offset,
                    length: object.blob.length,
                },
            )
        })
        .collect();
    out.sort_unstable_by_key(|(index, offset, span)| (*index, *offset, span.object));
    out.into_iter().map(|(_, _, span)| span).collect()
}

/// The alignment a writer used, read off the offsets: the largest power of
/// two dividing their gcd. `0` with no serving spans.
#[must_use]
pub fn alignment(spans: &[Span<'_>]) -> u64 {
    let mut gcd = 0u64;
    for span in spans {
        gcd = gcd_of(gcd, span.offset);
    }
    if gcd == 0 {
        return 0;
    }
    1u64 << gcd.trailing_zeros()
}

/// What a sequence of spans got wrong; gap and overlap are distinct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    Gap {
        after: String,
        at: u64,
        before: String,
        starts: u64,
    },
    Overlap {
        held: String,
        upto: u64,
        by: String,
        starts: u64,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Gap {
                after,
                at,
                before,
                starts,
            } => write!(
                f,
                "leaves a gap in its payload run: {after} ends at byte {at} and {before} \
                 starts at {starts}, and the bytes between belong to no serving object, so \
                 the run is not one read"
            ),
            Fault::Overlap {
                held,
                upto,
                by,
                starts,
            } => write!(
                f,
                "overlaps in its payload run: {held} runs to byte {upto} and {by} starts \
                 at {starts}, which is inside it without being the same span, so two \
                 objects claim one stretch of bytes"
            ),
        }
    }
}

/// The tiling check: at or after the reached point (up to `align` bytes of
/// padding) is fine, an exact alias is fine, a partial overlap or a larger
/// gap is a fault.
#[must_use]
pub fn tiling_fault(spans: &[Span<'_>], align: u64) -> Option<Fault> {
    let step = align.max(1);
    let mut seen: BTreeSet<(Option<&str>, u64, u64)> = BTreeSet::new();
    let mut reached: Option<(&Span<'_>, u64)> = None;
    for span in spans {
        let key = (span.shard, span.offset, span.length);
        if seen.contains(&key) {
            continue;
        }
        if let Some((previous, at)) = reached
            && previous.shard == span.shard
        {
            if span.offset < at {
                return Some(Fault::Overlap {
                    held: previous.object.to_string(),
                    upto: at,
                    by: span.object.to_string(),
                    starts: span.offset,
                });
            }
            let padded = at.div_ceil(step).saturating_mul(step);
            if span.offset > padded {
                return Some(Fault::Gap {
                    after: previous.object.to_string(),
                    at: padded,
                    before: span.object.to_string(),
                    starts: span.offset,
                });
            }
        }
        seen.insert(key);
        reached = Some((span, span.end()));
    }
    None
}

/// A serving artifact's filename, taken apart: `<model-slug>.<sku>.<backend>.zt`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Name {
    pub slug: String,
    pub sku: String,
    pub backend: String,
}

impl Name {
    #[must_use]
    pub fn render(&self) -> String {
        format!("{}.{}.{}.zt", self.slug, self.sku, self.backend)
    }

    pub fn parse(name: &str) -> Result<Name, Error> {
        let bad = |why: &str| Error::Checkpoint(format!("serving artifact name {name:?} {why}"));
        let (rest, extension) = name.rsplit_once('.').ok_or_else(|| bad("has no extension"))?;
        if extension != "zt" {
            return Err(bad("does not end in `.zt`"));
        }
        let (rest, backend) = rest.rsplit_once('.').ok_or_else(|| bad("names no backend"))?;
        let (slug, sku) = rest.rsplit_once('.').ok_or_else(|| bad("names no sku"))?;
        for field in [slug, sku, backend] {
            if !is_field(field) {
                return Err(bad(&format!(
                    "holds the field {field:?}, which is not `[a-z0-9][a-z0-9_-]*`"
                )));
            }
        }
        Ok(Name {
            slug: slug.to_string(),
            sku: sku.to_string(),
            backend: backend.to_string(),
        })
    }

    #[must_use]
    pub fn of(stamp: &Stamp, slug: &str) -> Name {
        Name {
            slug: slugify(slug),
            sku: slugify(&stamp.sku),
            backend: slugify(&stamp.backend),
        }
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

#[must_use]
pub fn slugify(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    for ch in id.chars() {
        match ch {
            '/' => out.push_str("--"),
            'a'..='z' | '0'..='9' | '_' | '-' => out.push(ch),
            'A'..='Z' => out.push(ch.to_ascii_lowercase()),
            _ => out.push('-'),
        }
    }
    let head = |ch: char| ch.is_ascii_lowercase() || ch.is_ascii_digit();
    while !out.is_empty() && !out.starts_with(head) {
        out.remove(0);
    }
    out
}

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

fn is_field(field: &str) -> bool {
    let mut chars = field.chars();
    chars
        .next()
        .is_some_and(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
        && chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-')
}

/// Which `pie.serving/<n>` key a file states, if any.
#[must_use]
pub(crate) fn stated_profile(attributes: &Value) -> Option<&str> {
    let Value::Map(entries) = attributes else {
        return None;
    };
    entries.iter().find_map(|(key, _)| match key {
        Value::Text(key) if key.starts_with(PROFILE_FAMILY) => Some(key.as_str()),
        _ => None,
    })
}

fn missing(field: Field) -> Error {
    Error::Checkpoint(format!(
        "the serving artifact's `{PROFILE}` carries no {field}, which it requires"
    ))
}

fn malformed(field: Field, why: &str) -> Error {
    Error::Checkpoint(format!(
        "the serving artifact's `{PROFILE}` {field} {why}"
    ))
}

fn required_text(attributes: &Value, field: Field) -> Result<&str, Error> {
    match attributes.get(field.key()) {
        Some(Value::Text(it)) => Ok(it),
        Some(_) => Err(malformed(field, "is not text")),
        None => Err(missing(field)),
    }
}

fn required_uint(attributes: &Value, field: Field) -> Result<u64, Error> {
    match attributes.get(field.key()) {
        Some(Value::Uint(it)) => Ok(*it),
        Some(_) => Err(malformed(field, "is not an unsigned integer")),
        None => Err(missing(field)),
    }
}

fn gcd_of(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd_of(b, a % b) }
}

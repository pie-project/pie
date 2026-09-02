//! `pie.serving/1`: the serving-artifact format layer — attributes layered on top of zTensor, defining no new container.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;

use ztensor::format::cbor::{self, Value};
use ztensor::{DigestAlgorithm, Manifest};

use crate::error::Error;
use crate::file::meta::META_PREFIX;

/// The file profile id and version this build implements — the attribute
/// key every serving fact lives under, and what makes a `.zt` a serving
/// artifact rather than an ordinary checkpoint.
pub const PROFILE: &str = "pie.serving/1";

/// Shared prefix of every version, telling "unreadable" from "not one".
pub const PROFILE_FAMILY: &str = "pie.serving/";

/// The smallest `block_bytes` the profile permits.
pub const MIN_BLOCK_BYTES: u64 = 4096;

/// The revision a human bumps by hand — [`Stamp::layout_revision`]'s value
/// for artifacts this build writes, bumped when a contract or plan change
/// moves bytes without moving [`PROFILE`].
pub const LAYOUT_REVISION: u64 = 1;

/// The block size this build's digest tables tile.
pub const BLOCK_BYTES: u64 = 64 << 20;

/// The file-level [`PROFILE`] block's member holding every serving
/// object's block tables: object name -> part name -> digest table.
///
/// Lives at file level, not per-object: `ztensor` freezes an object's
/// attributes at declaration, so a per-object table would need a writer
/// holding the whole part in memory (some planes exceed 90 GiB).
pub const BLOCKS_KEY: &str = "blocks";

/// The file-level serving facts (on disk: one attribute, keyed
/// [`PROFILE`]). [`Stamp::check`] compares field by field so a mismatch
/// names the field ([`Mismatch::refuse`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stamp {
    /// The profile id this file states — the key these live under, not a
    /// member. An unimplemented version is `Unsupported`, never "malformed".
    pub serving: String,
    /// `backend`: the engine whose kernels these bytes are landed for.
    pub backend: String,
    /// `sku`: the recipe the planes were compiled under.
    pub sku: String,
    /// `layout_revision`: a judgement a human moves, bumped when a contract
    /// or compiled-plan change moves bytes without moving [`PROFILE`].
    pub layout_revision: u64,
    /// `block_bytes`: the size [`Blocks`] are computed at. Stated, not
    /// fixed.
    pub block_bytes: u64,
    /// `block_algorithm`.
    pub block_algorithm: BlockAlgorithm,
    /// `adapters_zeroed`: asserts every registered-adapter bank this SKU
    /// declares is absent, served as `Buffer::zeroed`. Must be `true`.
    pub adapters_zeroed: bool,
}

/// One member of the [`PROFILE`] block, so a refusal can name the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Field {
    Serving,
    Backend,
    Sku,
    LayoutRevision,
    BlockBytes,
    BlockAlgorithm,
    AdaptersZeroed,
}

impl Field {
    /// The name this field is stored under. [`Field::Serving`]'s key is
    /// the block's own key (`PROFILE`).
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Field::Serving => PROFILE,
            Field::Backend => "backend",
            Field::Sku => "sku",
            Field::LayoutRevision => "layout_revision",
            Field::BlockBytes => "block_bytes",
            Field::BlockAlgorithm => "block_algorithm",
            Field::AdaptersZeroed => "adapters_zeroed",
        }
    }

    /// Every required key, cheapest and most-likely-wrong first.
    #[must_use]
    pub fn required() -> &'static [Field] {
        &[
            Field::Serving,
            Field::Backend,
            Field::Sku,
            Field::LayoutRevision,
            Field::BlockBytes,
            Field::BlockAlgorithm,
            Field::AdaptersZeroed,
        ]
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.key())
    }
}

/// The one field that disagreed, and what each side said about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    /// Which fact differs.
    pub field: Field,
    /// What the file on the disk says.
    pub artifact: String,
    /// What the deployment being booted wants.
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

/// The command that writes a serving artifact, spelled for `source`.
#[must_use]
pub fn rebuild(source: Option<&str>) -> String {
    match source {
        Some(source) => format!("pie model import --force {source}"),
        None => "pie model import --force <this deployment's checkpoint>".to_string(),
    }
}

impl Stamp {
    /// The stamp as the file-attribute fragment a manifest carries, keyed
    /// [`PROFILE`] — an inverse of [`Stamp::decode`].
    #[must_use]
    pub fn encode(&self) -> Value {
        Value::Map(vec![(text(PROFILE), self.block())])
    }

    /// The members, without the key they live under.
    fn block(&self) -> Value {
        let entries = vec![
            (text(Field::Backend.key()), text(&self.backend)),
            (text(Field::Sku.key()), text(&self.sku)),
            (
                text(Field::LayoutRevision.key()),
                Value::Uint(self.layout_revision),
            ),
            (text(Field::BlockBytes.key()), Value::Uint(self.block_bytes)),
            (
                text(Field::BlockAlgorithm.key()),
                text(self.block_algorithm.as_str()),
            ),
            (
                text(Field::AdaptersZeroed.key()),
                Value::Bool(self.adapters_zeroed),
            ),
        ];
        Value::Map(entries)
    }

    /// The stamp this build writes and expects, so a constant spelled
    /// twice can't disagree with itself.
    #[must_use]
    pub fn of(backend: &str, sku: &str) -> Stamp {
        Stamp {
            serving: PROFILE.to_string(),
            backend: backend.to_string(),
            sku: sku.to_string(),
            layout_revision: LAYOUT_REVISION,
            block_bytes: BLOCK_BYTES,
            block_algorithm: BlockAlgorithm::Xxh3,
            // Every adapter plane an import writes is zeroed: the artifact
            // holds the base model; a deployment mounting adapters brings its own.
            adapters_zeroed: true,
        }
    }

    /// Reads a stamp out of a manifest's `attributes`. Three outcomes:
    /// [`PROFILE`] present — read it; another `pie.serving/<n>` present —
    /// [`Error::Unsupported`]; neither — [`Error::Checkpoint`].
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for an unimplemented `pie.serving/<n>` — never
    /// "malformed", which would tell an operator with an older file that it
    /// is corrupt. [`Error::Checkpoint`] for anything else wrong with the block.
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
        let serving = PROFILE.to_string();
        let block_bytes = required_uint(attributes, Field::BlockBytes)?;
        if !block_bytes.is_power_of_two() || block_bytes < MIN_BLOCK_BYTES {
            return Err(malformed(
                Field::BlockBytes,
                &format!(
                    "is {block_bytes}, and must be a power of two at or above \
                     {MIN_BLOCK_BYTES}"
                ),
            ));
        }
        let block_algorithm =
            BlockAlgorithm::parse(required_text(attributes, Field::BlockAlgorithm)?)?;
        let adapters_zeroed = match attributes.get(Field::AdaptersZeroed.key()) {
            Some(Value::Bool(it)) => *it,
            Some(_) => return Err(malformed(Field::AdaptersZeroed, "is not a boolean")),
            None => return Err(missing(Field::AdaptersZeroed)),
        };
        if !adapters_zeroed {
            // Refused rather than believed: a default would let an old file
            // silently pass.
            return Err(malformed(
                Field::AdaptersZeroed,
                "is false, so this file was written by something that moved the snapshot \
                 out of the constructor and a restore would seat an adapter's contents as \
                 though they were a weight",
            ));
        }
        Ok(Stamp {
            serving,
            backend: required_text(attributes, Field::Backend)?.to_string(),
            sku: required_text(attributes, Field::Sku)?.to_string(),
            layout_revision: required_uint(attributes, Field::LayoutRevision)?,
            block_bytes,
            block_algorithm,
            adapters_zeroed,
        })
    }

    /// The boot's check: field by field, refusing on the first
    /// disagreement. Not a hash: a one-way fold can only say *different*.
    ///
    /// # Errors
    ///
    /// The first [`Field`] that differs, with both values.
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

    /// What this stamp says about one field, as the text a refusal prints.
    #[must_use]
    pub fn say(&self, field: Field) -> String {
        match field {
            Field::Serving => self.serving.clone(),
            Field::Backend => self.backend.clone(),
            Field::Sku => self.sku.clone(),
            Field::LayoutRevision => self.layout_revision.to_string(),
            Field::BlockBytes => self.block_bytes.to_string(),
            Field::BlockAlgorithm => self.block_algorithm.as_str().to_string(),
            Field::AdaptersZeroed => self.adapters_zeroed.to_string(),
        }
    }
}

/// The digest algorithm the `blocks` tables use — from zTensor's own
/// registered set; `weight_cache::Fnv` does not survive here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockAlgorithm {
    /// 64-bit xxh3, 8 bytes little-endian per block; what this build writes.
    Xxh3,
    /// SHA-256, 32 raw bytes per block.
    Sha256,
}

impl BlockAlgorithm {
    /// The registered name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            BlockAlgorithm::Xxh3 => "xxh3",
            BlockAlgorithm::Sha256 => "sha256",
        }
    }

    /// Bytes one block's digest occupies in a `blocks` byte string.
    #[must_use]
    pub fn width(self) -> usize {
        match self {
            BlockAlgorithm::Xxh3 => 8,
            BlockAlgorithm::Sha256 => 32,
        }
    }

    /// The algorithm a `block_algorithm` value names.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a name this build cannot compute — the
    /// file may be valid and simply newer than the reader.
    pub fn parse(name: &str) -> Result<BlockAlgorithm, Error> {
        match name {
            "xxh3" => Ok(BlockAlgorithm::Xxh3),
            "sha256" => Ok(BlockAlgorithm::Sha256),
            other => Err(Error::Unsupported(format!(
                "block digest algorithm {other:?}; this build computes xxh3 and sha256"
            ))),
        }
    }

    /// One block's digest over the bytes it covers, in the fixed width
    /// above. Pure: the caller (via [`block_span`]) decides which bytes.
    #[must_use]
    pub fn digest(self, bytes: &[u8]) -> Vec<u8> {
        let algo = match self {
            BlockAlgorithm::Xxh3 => DigestAlgorithm::Xxh3,
            BlockAlgorithm::Sha256 => DigestAlgorithm::Sha256,
        };
        // zTensor spells a digest "<algo>:<hex>"; unhexing it is the whole
        // difference from this table's form.
        let spelled = algo.digest(bytes);
        let hex = spelled.split_once(':').map_or("", |(_, hex)| hex);
        let mut out = unhex(hex);
        if matches!(self, BlockAlgorithm::Xxh3) {
            out.reverse();
        }
        out.resize(self.width(), 0);
        out
    }
}

/// How many blocks a part of `size` bytes is divided into (decoded size,
/// not a padded span).
#[must_use]
pub fn block_count(size: u64, block_bytes: u64) -> u64 {
    match block_bytes {
        0 => 0,
        step => size.div_ceil(step),
    }
}

/// Block `which`'s byte range within a part; blocks tile it exactly.
/// `None` past the end.
#[must_use]
pub fn block_span(size: u64, block_bytes: u64, which: u64) -> Option<Range<u64>> {
    if block_bytes == 0 || which >= block_count(size, block_bytes) {
        return None;
    }
    let from = which.saturating_mul(block_bytes);
    Some(from..from.saturating_add(block_bytes).min(size))
}

/// The length a `blocks` byte string must have for a part of `size` bytes.
#[must_use]
pub fn table_len(size: u64, block_bytes: u64, algorithm: BlockAlgorithm) -> usize {
    block_count(size, block_bytes) as usize * algorithm.width()
}

/// One part's block digest table, borrowed from the attribute that holds
/// it. A byte string, cheaper than an array of integers for a ~1600-block
/// bank. Redundant with the per-part `digest` on purpose: `digest` anchors
/// identity, this table is the *working* check a boot verifies a subset of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blocks<'a> {
    algorithm: BlockAlgorithm,
    block_bytes: u64,
    size: u64,
    digests: &'a [u8],
}

impl<'a> Blocks<'a> {
    /// Borrows a table, checking its length against the part it describes.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] when the byte string is not exactly
    /// [`table_len`] long.
    pub fn decode(
        algorithm: BlockAlgorithm,
        block_bytes: u64,
        size: u64,
        digests: &'a [u8],
    ) -> Result<Blocks<'a>, Error> {
        let want = table_len(size, block_bytes, algorithm);
        if digests.len() != want {
            return Err(Error::Checkpoint(format!(
                "a blocks table for a {size}-byte part at {block_bytes}-byte blocks is \
                 {want} bytes of {} digests and this one is {} bytes",
                algorithm.as_str(),
                digests.len(),
            )));
        }
        Ok(Blocks {
            algorithm,
            block_bytes,
            size,
            digests,
        })
    }

    /// The algorithm every digest in this table was computed with.
    #[must_use]
    pub fn algorithm(&self) -> BlockAlgorithm {
        self.algorithm
    }

    /// The part's decoded size, which the blocks tile exactly.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// How many blocks the part has.
    #[must_use]
    pub fn count(&self) -> u64 {
        block_count(self.size, self.block_bytes)
    }

    /// Whether the part has no blocks at all — the zero-length case.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Block `which`'s stored digest, in the algorithm's fixed width.
    #[must_use]
    pub fn digest(&self, which: u64) -> Option<&'a [u8]> {
        let width = self.algorithm.width();
        let at = usize::try_from(which).ok()?.checked_mul(width)?;
        self.digests.get(at..at.checked_add(width)?)
    }

    /// Block `which`'s byte range within the part.
    #[must_use]
    pub fn span(&self, which: u64) -> Option<Range<u64>> {
        block_span(self.size, self.block_bytes, which)
    }

    /// Every block, as its part-local range and its stored digest.
    pub fn iter(&self) -> impl Iterator<Item = (Range<u64>, &'a [u8])> + '_ {
        (0..self.count()).filter_map(|which| Some((self.span(which)?, self.digest(which)?)))
    }

    /// The prefix property: the first `blocks` digests, as raw bytes — a
    /// boot serving `[0, c1)` verifies exactly that and nothing else.
    #[must_use]
    pub fn prefix(&self, blocks: u64) -> &'a [u8] {
        let upto = usize::try_from(blocks.min(self.count()))
            .unwrap_or(usize::MAX)
            .saturating_mul(self.algorithm.width())
            .min(self.digests.len());
        &self.digests[..upto]
    }

    /// The whole table, as the attribute stores it.
    #[must_use]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.digests
    }
}

/// Concatenates block digests into the byte string an attribute holds (the
/// writer's half of [`Blocks::decode`]), each truncated or zero-extended to
/// the algorithm's fixed width.
#[must_use]
pub fn encode_blocks<'a>(
    algorithm: BlockAlgorithm,
    digests: impl IntoIterator<Item = &'a [u8]>,
) -> Vec<u8> {
    let width = algorithm.width();
    let mut out = Vec::new();
    for digest in digests {
        let at = out.len();
        out.extend_from_slice(digest);
        out.truncate(at + width);
        out.resize(at + width, 0);
    }
    out
}

/// The file's [`PROFILE`] fragment: the stamp's members and every serving
/// object's block tables, in one map. [`Stamp::encode`] is the same
/// fragment without the tables, for a caller that hasn't finished streaming.
///
/// `tables` is object name -> part name -> the byte string [`encode_blocks`]
/// produced. An object with no entry simply has no tables.
#[must_use]
pub fn file_block(
    stamp: &Stamp,
    tables: &BTreeMap<String, BTreeMap<String, Vec<u8>>>,
) -> Value {
    let Value::Map(mut members) = stamp.block() else {
        unreachable!("`Stamp::block` is a map by construction");
    };
    if !tables.is_empty() {
        members.push((
            text(BLOCKS_KEY),
            Value::Map(
                tables
                    .iter()
                    .map(|(object, parts)| {
                        (
                            text(object),
                            Value::Map(
                                parts
                                    .iter()
                                    .map(|(part, table)| (text(part), Value::Bytes(table.clone())))
                                    .collect(),
                            ),
                        )
                    })
                    .collect(),
            ),
        ));
    }
    Value::Map(vec![(text(PROFILE), Value::Map(members))])
}

/// One part's digest table, out of a file's [`PROFILE`] block (the
/// reader's half of [`file_block`]). `None` for every way the table can be
/// absent — this part can't be verified without hashing the whole of it.
#[must_use]
pub fn stated_blocks<'a>(attributes: &'a Value, object: &str, part: &str) -> Option<&'a [u8]> {
    match attributes
        .get(PROFILE)?
        .get(BLOCKS_KEY)?
        .get(object)?
        .get(part)?
    {
        Value::Bytes(digests) => Some(digests),
        _ => None,
    }
}

/// A part's block digests, taken as the bytes go past (the writer's side
/// of [`Blocks`]).
///
/// One block is buffered at a time and hashed when it fills, so residency
/// costs only `block_bytes`, not the part's length. The tail block is
/// hashed at [`BlockFold::finish`].
pub struct BlockFold {
    algorithm: BlockAlgorithm,
    block_bytes: usize,
    pending: Vec<u8>,
    digests: Vec<Vec<u8>>,
}

impl BlockFold {
    /// A fold at the file's own algorithm and block size.
    #[must_use]
    pub fn new(algorithm: BlockAlgorithm, block_bytes: u64) -> BlockFold {
        BlockFold {
            algorithm,
            block_bytes: usize::try_from(block_bytes).unwrap_or(usize::MAX),
            pending: Vec::new(),
            digests: Vec::new(),
        }
    }

    /// Takes the next bytes of the part, in order. Chunk boundaries need
    /// not be block boundaries; a streamed table matches the whole-slice one.
    pub fn eat(&mut self, mut chunk: &[u8]) {
        while !chunk.is_empty() {
            let room = self.block_bytes.saturating_sub(self.pending.len());
            let take = room.min(chunk.len());
            self.pending.extend_from_slice(&chunk[..take]);
            chunk = &chunk[take..];
            if self.pending.len() == self.block_bytes {
                self.digests.push(self.algorithm.digest(&self.pending));
                self.pending.clear();
            }
        }
    }

    /// The table, with the short tail block hashed.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if !self.pending.is_empty() {
            self.digests.push(self.algorithm.digest(&self.pending));
        }
        encode_blocks(self.algorithm, self.digests.iter().map(Vec::as_slice))
    }
}

/// Each entry's first digest in a concatenated table, plus a sentinel:
/// entry `i`'s digests are `table[first[i]..first[i + 1]]`.
#[must_use]
pub fn first_blocks(sizes: &[u64], block_bytes: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(sizes.len() + 1);
    let mut at = 0u64;
    for size in sizes {
        out.push(at);
        at = at.saturating_add(block_count(*size, block_bytes));
    }
    out.push(at);
    out
}

/// One serving part, addressed where it lies. Borrowed from the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span<'a> {
    /// The serving object this part belongs to.
    pub object: &'a str,
    /// The part's name within it — `data`, `scales`, `zeros`.
    pub part: &'a str,
    /// The shard the blob lives in; `None` is the containing file.
    pub shard: Option<&'a str>,
    /// The blob's offset within that file.
    pub offset: u64,
    /// The blob's length; also the decoded size the blocks tile.
    pub length: u64,
}

impl Span<'_> {
    /// One past the last byte this part occupies.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.length)
    }
}

/// Whether an object name is a serving object rather than a metadata one
/// (the compiled tokenizer and model descriptor, under `__meta__/`).
#[must_use]
pub fn is_serving(name: &str) -> bool {
    !name.starts_with(META_PREFIX)
}

/// The serving sequence, derived by sorting: `(shard index, min blob
/// offset)`, ties broken by object name. Position *i* is hotter than *i +
/// 1*. Nothing here is stored.
#[must_use]
pub fn sequence(manifest: &Manifest) -> Vec<&str> {
    let shards: BTreeMap<&str, usize> = manifest
        .shards
        .keys()
        .enumerate()
        .map(|(at, name)| (name.as_str(), at + 1))
        .collect();
    let mut out: Vec<(usize, u64, &str)> = manifest
        .objects
        .iter()
        .filter(|(name, _)| is_serving(name))
        .filter_map(|(name, object)| {
            let head = object.parts.values().min_by_key(|part| {
                (
                    part.blob
                        .shard
                        .as_deref()
                        .map_or(0, |it| shards.get(it).copied().unwrap_or(usize::MAX)),
                    part.blob.offset,
                )
            })?;
            let index = head
                .blob
                .shard
                .as_deref()
                .map_or(0, |it| shards.get(it).copied().unwrap_or(usize::MAX));
            Some((index, head.blob.offset, name.as_str()))
        })
        .collect();
    out.sort_unstable();
    out.into_iter().map(|(_, _, name)| name).collect()
}

/// Every serving part, in [`sequence`] order — the list [`tiling_fault`] checks.
#[must_use]
pub fn spans(manifest: &Manifest) -> Vec<Span<'_>> {
    let mut out = Vec::new();
    for name in sequence(manifest) {
        let Some((object_name, object)) = manifest.objects.get_key_value(name) else {
            continue;
        };
        let mut parts: Vec<Span<'_>> = object
            .parts
            .iter()
            .map(|(part_name, part)| Span {
                object: object_name.as_str(),
                part: part_name.as_str(),
                shard: part.blob.shard.as_deref(),
                offset: part.blob.offset,
                length: part.blob.length,
            })
            .collect();
        parts.sort_unstable_by_key(|span| (span.offset, span.part));
        out.extend(parts);
    }
    out
}

/// The padded span of each part: the next offset minus this one, or (for
/// the last part in a shard) its own end.
#[must_use]
pub fn padded_spans(spans: &[Span<'_>]) -> Vec<u64> {
    let mut out = vec![0u64; spans.len()];
    for (at, span) in spans.iter().enumerate() {
        let next = spans
            .get(at + 1)
            .filter(|next| next.shard == span.shard && next.offset >= span.end());
        out[at] = match next {
            Some(next) => next.offset - span.offset,
            None => span.length,
        };
    }
    out
}

/// The alignment a writer used: the largest power of two dividing the
/// offsets' gcd, or `0` with none.
#[must_use]
pub fn alignment(spans: &[Span<'_>]) -> u64 {
    let mut gcd = 0u64;
    for span in spans {
        gcd = binary_gcd(gcd, span.offset);
    }
    if gcd == 0 {
        return 0;
    }
    // gcd(2 MiB, 3 MiB) is 1 MiB, an alignment; a raw gcd could carry an
    // odd factor, which is not.
    1u64 << gcd.trailing_zeros()
}

/// Where the payload run begins: the first serving blob's offset.
/// `None` when the file serves nothing.
#[must_use]
pub fn payload_at(spans: &[Span<'_>]) -> Option<u64> {
    spans
        .iter()
        .filter(|span| span.shard.is_none())
        .map(|span| span.offset)
        .min()
}

/// How long the payload run is, from [`payload_at`] to the last serving
/// blob's end. Must be uninterrupted — a boot pumps a prefix as one read.
#[must_use]
pub fn payload_total(spans: &[Span<'_>]) -> u64 {
    let local = || spans.iter().filter(|span| span.shard.is_none());
    match (payload_at(spans), local().map(Span::end).max()) {
        (Some(at), Some(end)) => end.saturating_sub(at),
        _ => 0,
    }
}

/// What a sequence of spans got wrong; gap and overlap are distinct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// Bytes inside the payload run that no serving part claims, beyond what
    /// the writer's alignment can account for.
    Gap {
        /// The part the run reached.
        after: String,
        /// Where that part's padding ends.
        at: u64,
        /// The part that starts too late.
        before: String,
        /// Where it starts.
        starts: u64,
    },
    /// Two parts claiming the same bytes without claiming exactly the same.
    Overlap {
        /// The part already occupying them.
        held: String,
        /// One past its last byte.
        upto: u64,
        /// The part that reaches back into it.
        by: String,
        /// Where it starts.
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
                 starts at {starts}, and the bytes between belong to no serving part, so \
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
                 parts claim one stretch of bytes"
            ),
        }
    }
}

/// The tiling check. Three-way: at/after the reached point is fine (up to
/// `align` bytes of padding); exactly equal to a span already seen is fine
/// (blessed blob sharing); partially overlapping is refused.
#[must_use]
pub fn tiling_fault(spans: &[Span<'_>], align: u64) -> Option<Fault> {
    let step = align.max(1);
    let mut seen: BTreeSet<(Option<&str>, u64, u64)> = BTreeSet::new();
    let mut reached: Option<(&Span<'_>, u64)> = None;
    for span in spans {
        let key = (span.shard, span.offset, span.length);
        if seen.contains(&key) {
            // An alias: exactly this stretch is already served.
            continue;
        }
        if let Some((previous, at)) = reached
            && previous.shard == span.shard
        {
            if span.offset < at {
                return Some(Fault::Overlap {
                    held: name_of(previous),
                    upto: at,
                    by: name_of(span),
                    starts: span.offset,
                });
            }
            // Padding carries the run up to the next boundary, no further.
            let padded = at.div_ceil(step).saturating_mul(step);
            if span.offset > padded {
                return Some(Fault::Gap {
                    after: name_of(previous),
                    at: padded,
                    before: name_of(span),
                    starts: span.offset,
                });
            }
        }
        seen.insert(key);
        reached = Some((span, span.end()));
    }
    None
}

/// The artifact's key: `xxh3` over deterministic-CBOR of the stamp and
/// every part digest, computed and never stored. The stamp is folded in
/// even though the model's content digest excludes layout, so
/// CUDA-tp1-bf16 and Metal-tp2-int4 share the latter but not this key.
///
/// # Errors
///
/// [`Error::Checkpoint`] when a serving part carries no digest: nothing
/// would stand for that part's content, and skipping it could collide
/// across files that differ there.
pub fn identity(stamp: &Stamp, manifest: &Manifest) -> Result<String, Error> {
    let value = Value::Array(vec![
        text(PROFILE),
        stamp.encode(),
        objects_reduction(manifest)?,
    ]);
    let encoded = cbor::encode(&value).map_err(Error::from)?;
    Ok(DigestAlgorithm::Xxh3.digest(&encoded))
}

/// The objects reduction, plus the shard table: the content half of
/// [`identity`], blind to offsets/lengths/alignment/padding/blob
/// sharing/encodings.
///
/// # Errors
///
/// [`Error::Checkpoint`] for a part with no digest.
pub fn objects_reduction(manifest: &Manifest) -> Result<Value, Error> {
    let mut objects = Vec::with_capacity(manifest.objects.len());
    for (name, object) in &manifest.objects {
        let mut parts = Vec::with_capacity(object.parts.len());
        for (part_name, part) in &object.parts {
            let Some(digest) = &part.digest else {
                return Err(Error::Checkpoint(format!(
                    "serving part {name:?}/{part_name:?} carries no digest, so this \
                     artifact has no key; a serving part must carry one"
                )));
            };
            let mut fields = vec![(text("dtype"), text(part.dtype.as_str()))];
            if let Some(logical) = &part.logical {
                fields.push((text("type"), text(logical)));
            }
            fields.push((text("digest"), text(digest)));
            parts.push((text(part_name), Value::Map(fields)));
        }
        let mut fields = vec![
            (
                text("shape"),
                Value::Array(object.shape.iter().copied().map(Value::Uint).collect()),
            ),
            (text("layout"), text(&object.layout)),
        ];
        if let Some(attributes) = &object.attributes {
            fields.push((text("attributes"), attributes.clone()));
        }
        fields.push((text("parts"), Value::Map(parts)));
        objects.push((text(name), Value::Map(fields)));
    }
    let shards = manifest
        .shards
        .iter()
        .map(|(name, shard)| {
            (
                text(name),
                Value::Map(vec![
                    (text("digest"), text(&shard.digest)),
                    (text("size"), Value::Uint(shard.size)),
                ]),
            )
        })
        .collect();
    Ok(Value::Map(vec![
        (text("objects"), Value::Map(objects)),
        (text("shards"), Value::Map(shards)),
    ]))
}

/// A serving artifact's filename, taken apart: `<model-slug>.<sku>.<backend>.zt`.
/// Every field is a field of [`Stamp`]. [`Stamp::layout_revision`] is
/// deliberately not in it — renaming on every bump would orphan a store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Name {
    /// The model id, slugged. The only field allowed to be long.
    pub slug: String,
    /// [`Stamp::sku`].
    pub sku: String,
    /// [`Stamp::backend`].
    pub backend: String,
}

impl Name {
    /// Renders the name.
    #[must_use]
    pub fn render(&self) -> String {
        format!("{}.{}.{}.zt", self.slug, self.sku, self.backend)
    }

    /// Parses by splitting on `.` from the right: extension, backend, sku,
    /// and everything remaining is the slug (which can't contain `.`).
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] when the name has too few components, does
    /// not end in `zt`, or holds a field outside `[a-z0-9][a-z0-9_-]*`.
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

    /// The name a stamp projects to, given a slug it doesn't itself carry.
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

/// A model id as a name field: lowercased, `/` becomes `--`, else outside
/// `[a-z0-9_-]` becomes `-`; a leading non-`[a-z0-9]` is dropped.
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

fn name_of(span: &Span<'_>) -> String {
    format!("{}/{}", span.object, span.part)
}

/// Which `pie.serving/<n>` a file states — separates "not a serving
/// artifact" from "a broken one". `Some` regardless of member integrity.
#[must_use]
pub fn stated_profile(attributes: &Value) -> Option<&str> {
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
    Error::Checkpoint(format!("the serving artifact's `{PROFILE}` {field} {why}"))
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

fn binary_gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

fn unhex(hex: &str) -> Vec<u8> {
    let bytes = hex.as_bytes();
    bytes
        .chunks_exact(2)
        .filter_map(|pair| {
            let hi = (pair[0] as char).to_digit(16)?;
            let lo = (pair[1] as char).to_digit(16)?;
            u8::try_from(hi * 16 + lo).ok()
        })
        .collect()
}


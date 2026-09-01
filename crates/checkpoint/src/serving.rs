//! `pie.serving/1` — the serving-artifact format layer, and nothing else.
//!
//! A pie serving artifact is a `.zt` holding one deployment's landed weights in
//! the order a boot wants to read them, with enough per-block integrity that
//! any *prefix* of it can be verified without hashing the rest. Everything the
//! profile adds to zTensor is **attributes** — one key, [`PROFILE`], at file
//! level and one on each serving object: delete them and the file is still a
//! valid checkpoint of the same weights, and only stops being servable. So
//! there is no new container here, and this module defines none.
//!
//! # Why this module sits outside `file/`
//!
//! `file/` is where this crate is allowed to touch a disk, and
//! `tests/standalone.rs::nothing_below_the_reader_opens_a_file` exempts it
//! wholesale for that reason. This module is deliberately NOT under it, so
//! that the same test scans this source and passes: a format layer that
//! needed an exemption would be a format layer that had already stopped being
//! one. Every item below is a total function or a type over borrowed data.
//! Nothing here opens anything, and nothing here knows what a mapping is.
//!
//! That is the same standard zTensor holds its own `format/` module to, and
//! the reason is the same one it gives: *these are the definitions a second
//! implementation would have to agree with*. A definition that could only be
//! reached through a reader would be a definition only this reader has.
//!
//! The reader that runs these checks against a real artifact, and the writer
//! that produces one, are M-4b-2 and M-4d. This module is what they will both
//! spell their agreement in.
//!
//! # What is DERIVED here rather than stored
//!
//! Four facts format 3 kept in its header are computed from the manifest
//! instead, and each is a total function of what the manifest already holds:
//!
//! - **The rung order** — [`sequence`], sorting by `(shard, offset)` with the
//!   name breaking ties. §6.3 constrains payload order only for *canonical*
//!   files and a serving artifact is deliberately not one, so the manifest's
//!   offsets are the only statement of order there is. Recovering it rather
//!   than asserting it is what lets a boot read a ranking it did not choose.
//! - **The padded span** — [`padded_spans`], the next offset minus this one.
//!   §2.4 owns the bytes between blobs (they MUST be `0x00`) and floors every
//!   offset at 4096. `Group::reserved` was `bytes.next_multiple_of(256)`, and
//!   4096 % 256 == 0, so the container's own floor already grants everything
//!   `reserved` was asserting.
//! - **The payload alignment** — [`alignment`], the gcd of the serving
//!   offsets. §2.4, verbatim: *"Because alignment is observable from the
//!   offsets themselves, the actual alignment used by a writer is not stored
//!   in the file."* `TIER_ALIGN = 2 MiB` survives as a writer policy and dies
//!   as a constant a reader believes.
//! - **The payload run** — [`payload_at`] and [`payload_total`]: the first
//!   serving blob to the end of the last, both already in the manifest. A
//!   stored total is a claim the offsets can contradict.
//!
//! A fifth, the artifact's own key ([`identity`]), is computed and **MUST NOT
//! be stored in the file**, on §6.4's rule: *a stored value is a claim that
//! can be false; a computed one cannot be.*
//!
//! And one fact is emphatically NOT here at all: **the cut**. `c1` and `c2`
//! are chosen from a boot's budgets. No attribute this module reads or writes
//! is a function of a budget, which is what makes one artifact serve any pair.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;

use ztensor::format::cbor::{self, Value};
use ztensor::{DigestAlgorithm, Manifest};

use crate::error::Error;
use crate::file::meta::META_PREFIX;

/// **The file profile id and version this build implements — and the
/// attribute key every serving fact lives under.**
///
/// One key at file level and one key on each serving object, each holding a
/// map. Its PRESENCE is what makes a `.zt` a serving artifact rather than an
/// ordinary checkpoint, which is how `pie model import` tells its own output
/// from its input, and there is no separate marker key saying so.
///
/// # The version is in the KEY, which is how the spec versions everything else
///
/// `zt.sparse_csr/1`, `gguf.q4_k/1`, and §5.2's own vendor example
/// `pie.paged_kv/1` all carry their version in the id. A flat vocabulary
/// cannot: it would need a member saying which version the other members are
/// in, and a reader would have to parse a block to discover it could not read
/// it. With the profile id as the key, **a v2 reader ignores a v1 file's block
/// wholesale and a v1 reader ignores a v2's** — §3.1's *"readers MUST ignore
/// map keys they do not recognize"* doing exactly the work the spec's L2 layer
/// describes, where a version once published is immutable and a new one is a
/// new name.
///
/// # And it makes the ignorable unit exactly ONE key
///
/// Which is the owner's rule made cheap: *delete every serving fact and the
/// file is still a valid checkpoint of the same weights* is one deletion at
/// file level and one per object, rather than nine and one. A test of that
/// property becomes structural instead of enumerative — and a property nobody
/// has to enumerate is a property nobody can enumerate incompletely.
///
/// Keys obey §3.1's general name rules — non-empty UTF-8, at most 1024 bytes,
/// no `U+0000` — so `.` and `/` are both legal here. Only SHARD names are
/// restricted to `[A-Za-z0-9._-]`, and this is not one.
pub const PROFILE: &str = "pie.serving/1";

/// Every version of this profile shares this prefix, which is what lets a
/// reader tell *"a serving artifact this build cannot read"* from *"not a
/// serving artifact"*.
///
/// The first is [`Error::Unsupported`] — re-import — and the second is an
/// ordinary checkpoint, which is not an error at all until somebody tries to
/// serve it. Without the prefix the two would be one absent key.
pub const PROFILE_FAMILY: &str = "pie.serving/";

/// The smallest `block_bytes` the profile permits (§4.1).
///
/// §2.4's blob-offset floor, reused as the block floor: a block smaller than
/// the alignment quantum would spend a digest on padding policy.
pub const MIN_BLOCK_BYTES: u64 = 4096;

/// **THE REVISION A HUMAN MOVES** — [`Stamp::layout_revision`]'s value for
/// artifacts this build writes.
///
/// Bumped when a change to an authored contract or a compiled plan moves bytes
/// without moving [`PROFILE`]. What it catches and nothing else does is the
/// engine's INTERPRETATION of identical bytes under identical declared facts:
/// how experts sit within a bank's plane, which tile order a repack leaves
/// behind, whether a fold runs at import or at load. Each of those writes an
/// artifact whose objects, shapes, layouts, dtypes and part digests are
/// exactly what a new pie would have written, and whose bytes mean something
/// else.
///
/// Hand-maintained and not hashed, for the reason [`Stamp::layout_revision`]
/// records: hashing the pie version invalidates every artifact on every
/// rebuild, and hashing nothing serves wrong bytes silently.
pub const LAYOUT_REVISION: u64 = 1;

/// **The block this build's digest tables tile.**
///
/// `weight_cache/tier.rs`'s `TIER_BLOCK`, carried across with its number and
/// its argument: eight bytes of table per block, so 64 MiB is ~13 KiB of
/// manifest per 100 GiB of payload, and the granularity was chosen for the
/// READER — a boot verifies the prefix it is about to serve, and one chain
/// over a whole bank would put a minute of one core in front of the first
/// token.
pub const BLOCK_BYTES: u64 = 64 << 20;

/// **The member of the FILE'S [`PROFILE`] block that holds every serving
/// object's block tables (§5.5)**, spelled once so the writer and the reader
/// cannot disagree about it.
///
/// Its value is a map from object name to a map from part name to that part's
/// digest table — which is what [`Blocks::decode`] borrows and
/// [`encode_blocks`] produces.
///
/// ```cbor
/// "pie.serving/1": {
///     "backend": "cuda", … ,
///     "blocks": {
///         "embed":              { "data": h'…' },
///         "layer.0.qg_proj":    { "data": h'…', "scales": h'…' },
///     },
/// }
/// ```
///
/// # It was per OBJECT, and it moved for one measured reason
///
/// A table is a fold over its part's bytes, and `ztensor` freezes an object's
/// attributes AT DECLARATION — so a per-object table can only be written by a
/// writer holding the whole part in memory. The catalog says what that costs:
/// the largest single plane is `qwen38-flash-bf16`'s `ple.table` at **95.4
/// GiB**, and three more rows exceed 2 GiB. That is not a peak to accept; it
/// is a writer that cannot write the models this tree ships.
///
/// `Writer::set_attributes` only stores into the manifest, and the manifest is
/// written at `finish()` — so FILE attributes may be set after every object
/// has been streamed. Measured, not read: two objects streamed in 97-byte
/// chunks with a fold accumulated as the bytes went by, `set_attributes`
/// after both closed, reopened, every value back. The tables therefore live
/// where a one-pass streaming writer can still put them.
///
/// **And the owner's rule gets simpler by it.** It was *delete two keys, one
/// at file level and one on every serving object*; it is now *delete one
/// key*. The mechanical test — delete it and an ordinary checkpoint of the
/// same weights remains — became a single removal.
///
/// It is not a [`Field`]: a [`Field`] is a fact [`Stamp::check`] compares
/// against a deployment, and a digest table is not something a deployment has
/// an opinion about. It is a sibling member of the same block, and
/// [`Stamp::decode`] passes over it the way §3.1 requires a reader to pass
/// over any key it does not recognize.
///
/// The cost is 8 bytes per 64 MiB of payload, by [`Blocks`]'s own arithmetic:
/// ~45 KiB of manifest for a 329 GiB model, ~70 KiB for a 530 GiB one.
pub const BLOCKS_KEY: &str = "blocks";

// ── §4 the stamp ────────────────────────────────────────────────────────────

/// **The file-level serving facts, as one value** — and on the disk, as ONE
/// attribute, keyed [`PROFILE`], whose members are the fields below.
///
/// **The point of the stamp is that its comparison names a field.** The thing
/// it replaces, `tier::Identity::key`, was a `u64`, and a `u64` can only ever
/// answer *same or not*; "this artifact is `tp_size` 1, this deployment is
/// 2" is a sentence an operator can act on. [`Stamp::check`] is that
/// comparison and [`Mismatch::refuse`] is that sentence.
///
/// **The three provenance keys stay FLAT beside it, and that mixture is a
/// decision rather than drift.** `pie_version`, `pie_source` and
/// `pie_source_encoding` (`file/meta.rs`) are file-general provenance — they
/// say where the weights came from, which is true of an artifact whatever
/// profile it does or does not carry — and they predate this profile. Folding
/// them in would claim they are this profile's vocabulary, and a checkpoint
/// that is not a serving artifact would lose the ability to state where it
/// came from. Nothing in this type is about a source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stamp {
    /// The profile id this file states, which on the disk is the KEY the rest
    /// of these live under rather than a member beside them. Carried as a
    /// field so that a version disagreement can name itself like any other,
    /// though §9 checks it at step 2 — *before* the comparison — because an
    /// unimplemented version is `Unsupported` ("re-import"), never
    /// "malformed".
    pub serving: String,
    /// `backend`: the engine whose kernels these bytes are landed for.
    pub backend: String,
    /// `tp_size`: the tensor-parallel **degree** these planes were cut
    /// for. Degree changes bytes, so it is baked; rank does not appear,
    /// because rank is a cut.
    pub tp_size: u64,
    /// `sku`: the contract the planes were compiled under. Load-bearing —
    /// the object names in the file are this SKU's plane names, not the source
    /// checkpoint's, so a reader that does not know the SKU cannot bind them.
    pub sku: String,
    /// `precision`: the served numeric form the operator chose. The field
    /// that makes *one model at two quantizations* two artifacts. It states
    /// the choice, not the per-tensor storage — that is each object's own
    /// `layout` and `dtype`, where the format already says it.
    pub precision: String,
    /// `layout_revision`: **a judgement a human moves**, bumped when a change
    /// to the authored contract or the compiled plan moves bytes without
    /// moving [`PROFILE`]. Hand-maintained and not hashed, for the recorded
    /// reason: hashing the pie version invalidates every artifact on every
    /// rebuild, and hashing nothing serves wrong bytes silently.
    ///
    /// # What it catches that nothing else here does
    ///
    /// Three neighbours cover most of the ground and none of them covers this
    /// one:
    ///
    /// - **A format change** is a different [`PROFILE`], and the version is
    ///   the attribute KEY, so such a file is not read at all.
    /// - **A declared change** — shape, layout, object attributes, dtype,
    ///   logical type — moves [`identity`], which folds all five. But
    ///   [`identity`] answers *"is this the same artifact?"* for a store, not
    ///   *"is this artifact for this deployment?"* for a boot: a boot has no
    ///   independent second copy of the key to compare against, because
    ///   computing one would mean re-running the import.
    /// - **An ordering change** invalidates nothing, by the profile's own
    ///   safety property: [`sequence`] is derived, and a reader that ignores
    ///   hotness performs a correct, merely unranked, load.
    ///
    /// What is left is **the engine's INTERPRETATION of identical bytes under
    /// identical declared facts** — how experts sit within one bank's plane,
    /// which tile order a repack leaves behind, whether a fold happens at
    /// import or at load. Every one of those produces an artifact whose
    /// objects, shapes, layouts, dtypes and part digests are exactly what a
    /// new pie would have written, and whose bytes mean something else. There
    /// is no digest of that fact, because the fact is in the reader.
    /// [`recipe_digest`](Stamp::recipe_digest) covers a change to the compiled
    /// LOAD PLAN and is the strictly better check where it exists — but it is
    /// optional, and a plan digest does not move when the plan is unchanged
    /// and the kernel's reading of the plan's output is not.
    ///
    /// **And it is the revision `model_id`'s belief is paired with.**
    /// `file/meta.rs`'s first ruling — an identity that is BELIEVED rather
    /// than checked is safe only when paired with a revision — applies
    /// verbatim, because [`Stamp::model_id`] is the one field
    /// [`Stamp::check`] never compares. The stamp being checked field by
    /// field does not rescue that: `model_id` is deliberately outside the
    /// comparison, so without this number beside it an artifact from a pie
    /// that laid planes out differently would be believed about its own
    /// catalog row.
    pub layout_revision: u64,
    /// `block_bytes`: the size [`Blocks`] are computed at. Stated rather
    /// than fixed, so that a re-measurement of read concurrency never
    /// invalidates an artifact — the mistake format 2 made by putting the
    /// stripe count in the header.
    pub block_bytes: u64,
    /// `block_algorithm`.
    pub block_algorithm: BlockAlgorithm,
    /// `adapters_zeroed`: asserts that every plane this SKU declares as a
    /// registered adapter bank is **absent from the file**, and that its
    /// correct serving state is what `Buffer::zeroed` leaves.
    ///
    /// MUST be present and MUST be `true`. Absence is not "false": absence
    /// means the file was written by something that moved the snapshot out of
    /// the constructor, and restoring it would seat whatever an adapter held
    /// as though it were a weight.
    pub adapters_zeroed: bool,
    /// `model_id`: the catalog row this artifact was imported for.
    /// **BELIEVED, NOT CHECKED**, and therefore never compared by
    /// [`Stamp::check`] — a boot cannot check it, because identification works
    /// by holding a checkpoint's tensor names against a row's manifest and
    /// these names are post-transform SKU plane names by construction.
    /// Believing it is safe only because [`Stamp::layout_revision`] sits
    /// beside it: an artifact from a pie that laid planes out differently is
    /// believed about nothing.
    pub model_id: Option<String>,
    /// `recipe_digest`: `"<algo>:<hex>"` over the compiled load plan.
    /// CHECKED when both sides have one; see [`Stamp::recipe_unchecked`] for
    /// the case where one does not.
    pub recipe_digest: Option<String>,
}

/// One member of the [`PROFILE`] block, as a value, so a refusal can name the
/// field it compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Field {
    Serving,
    Backend,
    TpSize,
    Sku,
    Precision,
    LayoutRevision,
    BlockBytes,
    BlockAlgorithm,
    AdaptersZeroed,
    ModelId,
    RecipeDigest,
}

impl Field {
    /// **The name this field is stored under, inside the [`PROFILE`] block.**
    ///
    /// [`Field::Serving`] is the one exception and it is not an inconsistency:
    /// its key is the block's own key, because that is where the version now
    /// lives. A file states its profile version by the name of the attribute
    /// and not by a member of it, so "which key says the version" and "which
    /// key holds the block" are one answer.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Field::Serving => PROFILE,
            Field::Backend => "backend",
            Field::TpSize => "tp_size",
            Field::Sku => "sku",
            Field::Precision => "precision",
            Field::LayoutRevision => "layout_revision",
            Field::BlockBytes => "block_bytes",
            Field::BlockAlgorithm => "block_algorithm",
            Field::AdaptersZeroed => "adapters_zeroed",
            Field::ModelId => "model_id",
            Field::RecipeDigest => "recipe_digest",
        }
    }

    /// Every required key of §4.1, in the order [`Stamp::check`] compares
    /// them: cheapest and most-likely-wrong first.
    #[must_use]
    pub fn required() -> &'static [Field] {
        &[
            Field::Serving,
            Field::Backend,
            Field::TpSize,
            Field::Sku,
            Field::Precision,
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
    /// Which fact differs. This is the whole reason the stamp is typed fields
    /// rather than a fold.
    pub field: Field,
    /// What the file on the disk says.
    pub artifact: String,
    /// What the deployment being booted wants.
    pub deployment: String,
}

impl Mismatch {
    /// **The refusal, in `tier::refuse`'s three parts**: what is wrong, that
    /// nothing here rewrote or deleted it, and the command that would fix it.
    ///
    /// The middle part is not politeness. A serving artifact *is* the model —
    /// it was produced from a source that may not still be on this machine —
    /// so a refusal that quietly rebuilt or removed one would be destroying
    /// the only copy on a disagreement about a text field.
    ///
    /// `source` is the checkpoint the refusing load was pointed at, so the
    /// command can be copied rather than adapted. `None` leaves the argument
    /// as a slot, for the one refusal that is not a load's.
    #[must_use]
    pub fn refuse(&self, artifact: &str, source: Option<&str>) -> String {
        format!(
            "checkpoint: the serving artifact {artifact:?} states {} {:?} and this \
             deployment is {:?}. This file is how this machine holds the model, not a \
             cache of a boot, so nothing here rewrites it and nothing here deletes it — \
             run `{}` to write it again from the checkpoint this load names.",
            self.field,
            self.artifact,
            self.deployment,
            rebuild(source),
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

/// **The command that writes a serving artifact, spelled for `source`.**
///
/// One string, in one place, because it is a promise: every refusal above
/// prints it and the command has to be a thing that actually works on the
/// argument it is handed.
#[must_use]
pub fn rebuild(source: Option<&str>) -> String {
    match source {
        Some(source) => format!("pie model import --force {source}"),
        None => "pie model import --force <this deployment's checkpoint>".to_string(),
    }
}

impl Stamp {
    /// **The stamp as the file-attribute fragment a manifest carries**: one
    /// entry, keyed [`PROFILE`], holding the members.
    ///
    /// A fragment rather than a bare block, so that this and
    /// [`Stamp::decode`] are inverses over the same value — the thing a
    /// manifest's `attributes` map is made of — and a caller that wants the
    /// members alone reads them out of the one key.
    ///
    /// Optional fields are absent rather than null: §3.1 says readers ignore
    /// keys they do not recognize, and a present-but-null key is a key.
    #[must_use]
    pub fn encode(&self) -> Value {
        Value::Map(vec![(text(PROFILE), self.block())])
    }

    /// The members, without the key they live under.
    fn block(&self) -> Value {
        let mut entries = vec![
            (text(Field::Backend.key()), text(&self.backend)),
            (text(Field::TpSize.key()), Value::Uint(self.tp_size)),
            (text(Field::Sku.key()), text(&self.sku)),
            (text(Field::Precision.key()), text(&self.precision)),
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
        if let Some(model_id) = &self.model_id {
            entries.push((text(Field::ModelId.key()), text(model_id)));
        }
        if let Some(recipe) = &self.recipe_digest {
            entries.push((text(Field::RecipeDigest.key()), text(recipe)));
        }
        Value::Map(entries)
    }

    /// **THE STAMP THIS BUILD WRITES, AND THE ONE IT EXPECTS** — one
    /// constructor, called by the import that writes an artifact and by every
    /// boot that checks one.
    ///
    /// The five arguments are the facts that differ between a deployment and
    /// an artifact and are therefore worth comparing. Everything else is what
    /// THIS BUILD does, and it is filled in here rather than at each call site
    /// for the reason [`check`](Stamp::check) exists at all: a boot compares
    /// field by field, so a constant spelled twice is a field that can
    /// disagree with itself. An import writing `block_bytes` of 64 MiB while a
    /// boot expected 16 would refuse every artifact this build had ever
    /// written, and the refusal would name a field neither side chose.
    ///
    /// `recipe_digest` is left absent. It is the strictly better check where
    /// it exists — a digest of the compiled load plan — and no caller computes
    /// one yet, so the stamp says nothing rather than something weaker under a
    /// stronger name. A caller that has one sets it afterwards.
    #[must_use]
    pub fn of(
        backend: &str,
        tp_size: u64,
        sku: &str,
        precision: &str,
        model_id: Option<String>,
    ) -> Stamp {
        Stamp {
            serving: PROFILE.to_string(),
            backend: backend.to_string(),
            tp_size,
            sku: sku.to_string(),
            precision: precision.to_string(),
            layout_revision: LAYOUT_REVISION,
            block_bytes: BLOCK_BYTES,
            block_algorithm: BlockAlgorithm::Xxh3,
            // Every adapter plane an import writes is zeroed: the artifact
            // holds the base model, and a deployment that mounts adapters
            // brings its own.
            adapters_zeroed: true,
            model_id,
            recipe_digest: None,
        }
    }

    /// **Reads a stamp out of a manifest's `attributes`, by the key first.**
    ///
    /// §9 steps 2 and 3, in that order and for that reason: the version is the
    /// KEY, so it is answered before a single member is looked at. Three
    /// outcomes, and they are three different sentences —
    ///
    /// - [`PROFILE`] is present: this build reads it.
    /// - another `pie.serving/<n>` is present: [`Error::Unsupported`], which
    ///   means *re-import*. The file is not broken; it was written by another
    ///   build, faithfully.
    /// - neither: an ordinary checkpoint, which is [`Error::Checkpoint`] only
    ///   because somebody asked it to serve.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a `pie.serving/<n>` this build does not
    /// implement — §9's ruling, and the one place the distinction matters:
    /// calling it "malformed" tells every operator with an older file that it
    /// is corrupt. [`Error::Checkpoint`] for anything else: no serving key at
    /// all, a missing required member, a member of the wrong CBOR type, a
    /// `block_bytes` that is not a power of two at or above
    /// [`MIN_BLOCK_BYTES`], a `tp_size` of zero, an unregistered block
    /// algorithm, or an `adapters_zeroed` that is not `true`.
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
        let tp_size = required_uint(attributes, Field::TpSize)?;
        if tp_size == 0 {
            return Err(malformed(Field::TpSize, "must be at least 1"));
        }
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
            // Refused rather than believed. A default would make an old file
            // silently pass, which is the failure this key exists to prevent.
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
            tp_size,
            sku: required_text(attributes, Field::Sku)?.to_string(),
            precision: required_text(attributes, Field::Precision)?.to_string(),
            layout_revision: required_uint(attributes, Field::LayoutRevision)?,
            block_bytes,
            block_algorithm,
            adapters_zeroed,
            model_id: optional_text(attributes, Field::ModelId)?.map(str::to_string),
            recipe_digest: optional_text(attributes, Field::RecipeDigest)?.map(str::to_string),
        })
    }

    /// **The boot's check: field by field, refusing on the first
    /// disagreement.** `self` is the artifact, `deployment` is what is being
    /// asked for.
    ///
    /// Not a hash, and the reason is refusal quality alone. A one-way fold can
    /// only say *different*; this says which fact, which is the difference
    /// between an error code and a sentence. It costs one manifest read and no
    /// payload I/O.
    ///
    /// [`Stamp::model_id`] is never compared — it is believed, not checked
    /// (§4.3). [`Stamp::recipe_digest`] is compared only when both sides carry
    /// one; see [`Stamp::recipe_unchecked`].
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
        if let (Some(artifact), Some(wanted)) = (&self.recipe_digest, &deployment.recipe_digest)
            && artifact != wanted
        {
            return Err(Mismatch {
                field: Field::RecipeDigest,
                artifact: artifact.clone(),
                deployment: wanted.clone(),
            });
        }
        Ok(())
    }

    /// Whether [`Stamp::check`] fell back to the weaker check because one side
    /// carries no `recipe_digest`.
    ///
    /// A caller SHOULD warn: without it, agreement on the stamp is agreement
    /// on a description, and two imports of two different source revisions
    /// under identical settings produce identical stamps.
    #[must_use]
    pub fn recipe_unchecked(&self, deployment: &Stamp) -> bool {
        self.recipe_digest.is_none() || deployment.recipe_digest.is_none()
    }

    /// What this stamp says about one field, as the text a refusal prints.
    #[must_use]
    pub fn say(&self, field: Field) -> String {
        match field {
            Field::Serving => self.serving.clone(),
            Field::Backend => self.backend.clone(),
            Field::TpSize => self.tp_size.to_string(),
            Field::Sku => self.sku.clone(),
            Field::Precision => self.precision.clone(),
            Field::LayoutRevision => self.layout_revision.to_string(),
            Field::BlockBytes => self.block_bytes.to_string(),
            Field::BlockAlgorithm => self.block_algorithm.as_str().to_string(),
            Field::AdaptersZeroed => self.adapters_zeroed.to_string(),
            Field::ModelId => self.model_id.clone().unwrap_or_default(),
            Field::RecipeDigest => self.recipe_digest.clone().unwrap_or_default(),
        }
    }
}

// ── §5.5 the block tables ───────────────────────────────────────────────────

/// The digest algorithm the `blocks` tables use, named the way §3.4's
/// `digest` names its own.
///
/// `weight_cache::Fnv` does not survive here, and the paragraph that made
/// the block table an attribute is the paragraph that decides this: an attribute
/// this profile defines carries its own algorithm from the *registered* set,
/// and putting a name zTensor's registry does not know into a digest-shaped
/// field would be inventing registry vocabulary in a vendor profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockAlgorithm {
    /// 64-bit xxh3, **8 bytes little-endian** per block. What this build
    /// writes, and what canonical zTensor uses.
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
    /// [`Error::Unsupported`] for a name this build cannot compute — the file
    /// may be perfectly valid and simply newer than the reader, which is
    /// zTensor's own rule for the same question.
    pub fn parse(name: &str) -> Result<BlockAlgorithm, Error> {
        match name {
            "xxh3" => Ok(BlockAlgorithm::Xxh3),
            "sha256" => Ok(BlockAlgorithm::Sha256),
            other => Err(Error::Unsupported(format!(
                "block digest algorithm {other:?}; this build computes xxh3 and sha256"
            ))),
        }
    }

    /// One block's digest over the bytes it covers, in the fixed width above.
    ///
    /// Pure over a borrowed slice: the caller has already decided which bytes
    /// a block is, which is what [`block_span`] answers.
    #[must_use]
    pub fn digest(self, bytes: &[u8]) -> Vec<u8> {
        let algo = match self {
            BlockAlgorithm::Xxh3 => DigestAlgorithm::Xxh3,
            BlockAlgorithm::Sha256 => DigestAlgorithm::Sha256,
        };
        // zTensor spells a digest `"<algo>:<big-endian lowercase hex>"`, which
        // is the manifest's form and not this table's. Unhexing it is the
        // whole of the difference, and going through the registered
        // implementation is what keeps the two the same function.
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

/// **How many blocks a part of `size` bytes is divided into.**
///
/// `tier::Head::blocks_of`, with one deliberate change: `size` is the part's
/// **decoded size**, not a padded span. §2.4 makes padding a writer policy
/// (*"4096 is a floor, not a ceiling"*), so a digest that covered it would
/// make two files with the same tensors and different alignment fail each
/// other's verification. Nothing is lost: the padding's content is a spec MUST
/// (zero) and is checkable by comparing against zero, at no hashing cost.
#[must_use]
pub fn block_count(size: u64, block_bytes: u64) -> u64 {
    match block_bytes {
        0 => 0,
        step => size.div_ceil(step),
    }
}

/// **Block `which`'s byte range within a part of `size` bytes.**
///
/// Part-local: engine-cuda's tier.rs states the same rule with "entry" spelled
/// "part". The last block carries whatever the division left over, so the
/// blocks of a part tile it exactly: every byte is covered, and none twice.
///
/// `None` past the end, so a caller cannot be handed a span nobody bounded.
#[must_use]
pub fn block_span(size: u64, block_bytes: u64, which: u64) -> Option<Range<u64>> {
    if block_bytes == 0 || which >= block_count(size, block_bytes) {
        return None;
    }
    let from = which.saturating_mul(block_bytes);
    Some(from..from.saturating_add(block_bytes).min(size))
}

/// The length a `blocks` byte string must have for a part of `size` bytes.
///
/// §9 step 8's check, as one expression.
#[must_use]
pub fn table_len(size: u64, block_bytes: u64, algorithm: BlockAlgorithm) -> usize {
    block_count(size, block_bytes) as usize * algorithm.width()
}

/// One part's block digest table, borrowed from the attribute that holds it.
///
/// A byte string rather than an array of integers: a 100 GiB routed bank has
/// ~1600 blocks, and as a CBOR array that is ~14 KB of heads and 1600
/// per-element decodes, against one length and one slice.
///
/// This is redundant with §3.4's per-part `digest`, and both are carried on
/// purpose. `digest` anchors identity — §6.4 is undefined without it and
/// [`identity`] folds it — while the `blocks` table is the *working* check, because
/// a boot verifies the subset it is about to serve and one chain over a
/// tens-of-gigabytes bank puts a minute of one core in front of the first
/// token. The cost of carrying both is 8 bytes per 64 MiB: 13 KiB for 100 GiB.
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
    /// [`table_len`] long. A zero-length part has zero digests and an empty
    /// byte string, which is a table and not an absence.
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

    /// The **prefix property**: the first `blocks` digests, as raw bytes.
    ///
    /// This is the only reason the granularity is what it is. Since the
    /// serving parts tile the payload run in sequence order, the blocks of a
    /// *prefix* of the sequence are a prefix of the concatenated tables — so a
    /// boot that is about to serve `[0, c1)` verifies exactly that and touches
    /// nothing else.
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

/// Concatenates block digests into the byte string an attribute holds.
///
/// The writer's half of [`Blocks::decode`]. Each digest is truncated or
/// zero-extended to the algorithm's fixed width, so the result is always a
/// well-formed table for the count it was given.
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

/// **The file's [`PROFILE`] fragment: the stamp's members and every serving
/// object's block tables, in one map.**
///
/// [`Stamp::encode`] is the same fragment without the tables, and the two are
/// deliberately separate calls: a caller that has not finished streaming does
/// not yet have the tables, and a caller reading the file back wants the
/// stamp whether or not it cares about digests.
///
/// `tables` is object name → part name → the byte string [`encode_blocks`]
/// produced. `BTreeMap` because the container sorts a map by encoded key bytes
/// anyway and an ordered input makes the artifact byte-reproducible without
/// depending on that.
///
/// An object with no entry here simply has no tables — which is what every
/// `__meta__/` object is, since none of them is served and none appears in
/// the sequence.
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

/// **One part's digest table, out of a file's [`PROFILE`] block.**
///
/// The reader's half of [`file_block`], and it answers `None` for every way
/// the table can be absent — no serving key, no [`BLOCKS_KEY`] member, no
/// entry for this object, no entry for this part, or an entry that is not a
/// byte string. One `None` because the caller's next sentence is the same for
/// all five: this part cannot be verified without hashing the whole of it.
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

/// **A part's block digests, taken as the bytes go past.**
///
/// The writer's side of [`Blocks`], and it exists as a value rather than a
/// loop because TWO writers compute these tables — `file/emit.rs`, which
/// writes a serving artifact from objects handed to it, and
/// `file/write.rs`'s streaming `Writer`, which is what `pie model import`
/// drives — and a table is a claim about bytes that a reader will check. Two
/// folds would be two functions producing one attribute, which is the hazard
/// `emit::Object::of` already routes around for layouts by asking
/// `write.rs::profile_of` rather than deciding again.
///
/// One block is buffered at a time and hashed when it fills, so the residency
/// this costs is `block_bytes` — the stamp's own number, floored at
/// [`MIN_BLOCK_BYTES`] — and not the part's length. That is the whole reason
/// a writer can stream at all: see [`BLOCKS_KEY`] for the 95.4 GiB plane that
/// made it necessary. The tail block is short by construction and is hashed
/// at [`BlockFold::finish`], which is why [`block_count`] rounds up.
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

    /// Takes the next bytes of the part, in order.
    ///
    /// Chunk boundaries are not block boundaries and are not required to be:
    /// what is buffered is whatever has not yet completed a block, so the
    /// table a stream produces is the table the whole slice would have.
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

/// **Each entry's first digest in a concatenated table, plus a sentinel** — so
/// entry `i`'s digests are `table[first[i]..first[i + 1]]` for every `i`, with
/// no special case for the last.
///
/// `tier::first_blocks`, over decoded sizes rather than reserved spans.
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

// ── §5.2/§5.3/§5.4 the derivations ──────────────────────────────────────────

/// One serving part, addressed where it lies.
///
/// Borrowed from the manifest, because that is where all of it already is.
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
    /// The blob's length. Serving parts are raw, so this is also the decoded
    /// size the blocks tile.
    pub length: u64,
}

impl Span<'_> {
    /// One past the last byte this part occupies.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.length)
    }
}

/// Whether an object name is a serving object rather than a metadata one.
///
/// The compiled tokenizer and model descriptor live under `__meta__/`, are
/// `dense` `u8` exactly as today, are not served, and never appear in the
/// sequence.
#[must_use]
pub fn is_serving(name: &str) -> bool {
    !name.starts_with(META_PREFIX)
}

/// **The serving sequence, derived by sorting.**
///
/// Ordered by `(shard index, min over the object's parts of blob offset)`,
/// ties broken by bytewise object name. Position *i* is hotter than position
/// *i + 1*.
///
/// Nothing about this is stored. A non-canonical file imposes no payload
/// order, so laying blobs out in ranking order is spec-legal and the manifest
/// already records every offset — so the order is *recovered* rather than
/// asserted. That is what lets a boot read a ranking a different pie chose.
///
/// Shard index orders the containing file first, then the shard table's own
/// key order. A single-file artifact is the recommended shape and has one
/// index.
///
/// # The limit, admitted rather than hidden
///
/// This pins how any file is READ — the derivation is total, deterministic and
/// needs nothing outside the manifest — and it does not pin the *ranking
/// function*, which belongs to a SKU's plan and not to a container profile.
/// Two implementations therefore agree exactly on how to read every file and
/// may disagree on which one they would write. What makes that acceptable: **a
/// reader that ignores hotness entirely and pumps in file order performs a
/// correct, merely unranked, load.** Order is a performance fact; nothing in
/// correctness rests on it.
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

/// Every serving part, in [`sequence`] order, parts of one object ordered by
/// part name.
///
/// The list [`tiling_fault`] checks and the writer's own output, in one shape.
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

/// **The padded span of each part: the next offset minus this one.**
///
/// `Group::reserved`'s successor, and it is derived rather than stored twice
/// over. §2.4 floors every blob offset at 4096 and 4096 % 256 == 0, so every
/// part in any conforming file already begins on the 256-byte boundary the
/// store's reinterpretation premise wanted; and the bytes between blobs are a
/// spec MUST to be `0x00`, so the padding's content is known without being
/// recorded.
///
/// The last part in a shard has no successor and pads to its own end, because
/// nothing after it is inside the run.
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

/// **The alignment a writer used, read off the offsets themselves.**
///
/// §2.4: *"Because alignment is observable from the offsets themselves, the
/// actual alignment used by a writer is not stored in the file."* `TIER_ALIGN`
/// dies as a pie constant and its reason survives as a writer policy — the
/// payload is read into page-locked memory the allocator hands out on
/// huge-page boundaries — so a reader that wants to size a pinned buffer takes
/// the gcd of the serving offsets, which is this.
///
/// `0` when there are no serving spans, and the largest power of two dividing
/// the gcd otherwise, because an alignment is a power of two and a gcd need
/// not be one.
#[must_use]
pub fn alignment(spans: &[Span<'_>]) -> u64 {
    let mut gcd = 0u64;
    for span in spans {
        gcd = binary_gcd(gcd, span.offset);
    }
    if gcd == 0 {
        return 0;
    }
    // The gcd of {2 MiB, 3 MiB} is 1 MiB, which is an alignment; the gcd of
    // {2 MiB, 3 * 2 MiB + 2 MiB} could carry an odd factor, which is not.
    1u64 << gcd.trailing_zeros()
}

/// **Where the payload run begins**: the first serving blob's offset in the
/// containing file. `None` when the file serves nothing.
#[must_use]
pub fn payload_at(spans: &[Span<'_>]) -> Option<u64> {
    spans
        .iter()
        .filter(|span| span.shard.is_none())
        .map(|span| span.offset)
        .min()
}

/// **How long the payload run is**: from [`payload_at`] to the end of the last
/// serving blob in the containing file.
///
/// The run MUST be uninterrupted — no metadata blob and no unreferenced blob
/// may lie inside it — which is the tiling invariant restated in the
/// container's vocabulary, and the reason it is a MUST: a boot pumps a prefix
/// of the sequence as **one contiguous read**, and a foreign blob in the
/// middle either gets pumped as though it were a weight or breaks the read
/// into two.
#[must_use]
pub fn payload_total(spans: &[Span<'_>]) -> u64 {
    let local = || spans.iter().filter(|span| span.shard.is_none());
    match (payload_at(spans), local().map(Span::end).max()) {
        (Some(at), Some(end)) => end.saturating_sub(at),
        _ => 0,
    }
}

// ── §9.7 the tiling check ───────────────────────────────────────────────────

/// What a sequence of spans got wrong. **Gap and overlap are distinct**, which
/// is the change this makes to the check it replaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// Bytes inside the payload run that no serving part claims, beyond what
    /// the writer's alignment can account for. Either a metadata blob, an
    /// unreferenced blob, or a hole — all three break the contiguous read.
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
    /// Two parts claiming the same bytes without claiming exactly the same
    /// bytes.
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

/// **The tiling check, in the container's vocabulary.**
///
/// `spans` is [`spans`]' output — sequence order — and `align` is
/// [`alignment`]'s, which is what says how large a stretch of padding can
/// legitimately sit between two parts.
///
/// The rule this replaces was a single equality, `group.offset != at`, which
/// read a gap and an overlap as one refusal *and* rejected an exact-equal
/// alias. So the rule here is three-way:
///
/// - **at or after** where the run has reached: fine, up to `align` bytes of
///   padding, which §2.4 already requires be zero.
/// - **exactly equal to a span already seen**: fine, and deliberately so.
///   §2.4 blesses blob sharing for weight tying — *valid iff they have exactly
///   equal `(offset, length)` or do not overlap at all* — and the replication
///   optimisation needs it. Aliased objects occupy adjacent positions,
///   distinguished by the name tiebreak, and are served from one span.
/// - **partially overlapping**: refused. This is the case §2.4 forbids
///   globally, and the one a serving reader must never wave through, because
///   both parts would be handed bytes the other also owns.
///
/// A gap and an overlap get different sentences because they have different
/// causes and different remedies, and the old check could say neither.
///
/// `align` of `0` or `1` means no padding policy, so any gap at all is a gap.
#[must_use]
pub fn tiling_fault(spans: &[Span<'_>], align: u64) -> Option<Fault> {
    let step = align.max(1);
    let mut seen: BTreeSet<(Option<&str>, u64, u64)> = BTreeSet::new();
    let mut reached: Option<(&Span<'_>, u64)> = None;
    for span in spans {
        let key = (span.shard, span.offset, span.length);
        if seen.contains(&key) {
            // An alias: exactly this stretch is already served. It advances
            // nothing, because it adds nothing to the run.
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
            // Padding may carry the run up to the next boundary and no
            // further; anything past that is bytes nothing accounts for.
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

// ── §6.2 identity ───────────────────────────────────────────────────────────

/// **The artifact's key: a fold over the stamp and every part digest,
/// computed and never stored.**
///
/// ```text
/// xxh3 over deterministic-CBOR([ "pie.serving/1", <stamp>, <objects> ])
/// ```
///
/// It **MUST NOT be written into the file**, on §6.4's own rule: *a stored
/// value is a claim that can be false; a computed one cannot be.* The thing it
/// replaces mixed a source checkpoint's mtime into the key, and an artifact
/// keyed by an mtime has no fixed point — the file's own metadata would key
/// the file, a copy would be a different model, and a `touch` would orphan a
/// hundred gigabytes. Content answers all three.
///
/// The stamp is folded in even though §6.4's content digest pointedly excludes
/// everything but content, and the reason is that the two answer different
/// questions: §6.4 identifies *the model*, deliberately blind to layout, so
/// the same weights for CUDA-tp1-bf16 and Metal-tp2-int4 share it. That is
/// right for §6.4 and wrong here, because a store must key on *this
/// specialization of this model* when the two have to coexist.
///
/// # The ORDER is deliberately out of the key
///
/// Format 3 mixed its whole image sequence in, on the argument that a file
/// whose order differs is a file this boot cannot cut. Under zTensor that
/// argument no longer holds: a boot **reads** the order from the manifest
/// instead of assuming it ([`sequence`]), so an artifact written by a pie with
/// a different prefetch ranking is still perfectly servable — merely ranked
/// differently. Leaving order out means a prefetch-heuristic change no longer
/// orphans every artifact on the disk, and [`Stamp::layout_revision`] remains
/// the door for a change that moves *bytes*.
///
/// # Errors
///
/// [`Error::Checkpoint`] when a serving part carries no digest. §3 makes a
/// digest a MUST for exactly this reason: with none there is nothing to stand
/// for the part's content, and a key that skipped it would collide across
/// files that differ there.
pub fn identity(stamp: &Stamp, manifest: &Manifest) -> Result<String, Error> {
    let value = Value::Array(vec![
        text(PROFILE),
        stamp.encode(),
        objects_reduction(manifest)?,
    ]);
    let encoded = cbor::encode(&value).map_err(Error::from)?;
    Ok(DigestAlgorithm::Xxh3.digest(&encoded))
}

/// §6.4's objects reduction, plus the shard table — the content half of
/// [`identity`].
///
/// `name -> {shape, layout, attributes, parts: {name -> {dtype, type,
/// digest}}}`, which is deliberately blind to offsets, lengths, alignment,
/// padding, blob sharing and encodings, so the same tensors give the same
/// answer whether they sit in one file or fifty. The shard table joins it
/// because its entries are whole-file digests, which is what makes the fold a
/// claim about a sharded artifact and not only about its root.
///
/// Metadata objects are folded in with everything else: a different tokenizer
/// is a different artifact.
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

// ── §8 the filename ─────────────────────────────────────────────────────────

/// A serving artifact's filename, taken apart.
///
/// ```text
/// <model-slug>.<sku>.<backend>-tp<n>.<precision>.zt
/// qwen--qwen3-30b-a3b.qwen_3.cuda-tp1.mxfp4.zt
/// ```
///
/// **The whole design in one line: name to find, stamp to verify.** Every
/// field here is a field of [`Stamp`], no field of the stamp is *only* in the
/// name, and so the name is a lossy, human-and-`ls`-readable projection of the
/// stamp. A boot that finds a file by name learns nothing it did not already
/// believe; a boot that reads the stamp refuses with the field that disagreed.
///
/// **[`Stamp::layout_revision`] is deliberately not in the name.** It moves on
/// a judgement, and a judgement that renamed files would orphan a store on
/// every bump. It is exactly the fact the stamp exists to carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Name {
    /// The model id, slugged. The only field allowed to be long, and the only
    /// one that may contain the separator's neighbours (`-`, `_`) freely.
    pub slug: String,
    /// [`Stamp::sku`].
    pub sku: String,
    /// [`Stamp::backend`].
    pub backend: String,
    /// [`Stamp::tp_size`].
    pub tp_size: u64,
    /// [`Stamp::precision`].
    pub precision: String,
}

impl Name {
    /// Renders the name.
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "{}.{}.{}-tp{}.{}.zt",
            self.slug, self.sku, self.backend, self.tp_size, self.precision
        )
    }

    /// **Parses by splitting on `.` from the right**: extension, precision,
    /// backend-and-degree, sku, and everything remaining is the slug.
    ///
    /// A slug cannot contain `.`, which is what makes that unambiguous — a
    /// slug that happens to contain `cuda-tp1` or a trailing `-tp8` is still
    /// read as a slug, because the split counts from the end and stops.
    ///
    /// The last component is the extension `zt`, so `Path::extension()` still
    /// answers `"zt"` and the existing extension test is unaffected. A future
    /// `pie.serving/2` may append a field, and only before the extension.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] when the name has too few components, does not
    /// end in `zt`, has no `-tp<n>` in the third-from-last, or holds a field
    /// outside `[a-z0-9][a-z0-9_-]*`.
    pub fn parse(name: &str) -> Result<Name, Error> {
        let bad = |why: &str| Error::Checkpoint(format!("serving artifact name {name:?} {why}"));
        let (rest, extension) = name.rsplit_once('.').ok_or_else(|| bad("has no extension"))?;
        if extension != "zt" {
            return Err(bad("does not end in `.zt`"));
        }
        let (rest, precision) = rest
            .rsplit_once('.')
            .ok_or_else(|| bad("names no precision"))?;
        let (rest, engine) = rest.rsplit_once('.').ok_or_else(|| bad("names no backend"))?;
        let (slug, sku) = rest.rsplit_once('.').ok_or_else(|| bad("names no sku"))?;
        let (backend, degree) = engine
            .rsplit_once("-tp")
            .ok_or_else(|| bad("names a backend with no `-tp<n>` degree"))?;
        let tp_size: u64 = degree
            .parse()
            .map_err(|_| bad("names a tensor-parallel degree that is not a number"))?;
        if tp_size == 0 {
            return Err(bad("names a tensor-parallel degree of zero"));
        }
        for field in [slug, sku, backend, precision] {
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
            tp_size,
            precision: precision.to_string(),
        })
    }

    /// The name a stamp projects to, for a model id it does not itself have to
    /// carry.
    ///
    /// `model_id` falls back to [`Stamp::model_id`], which is the field the
    /// slug comes from; an artifact with neither has no name to be found by,
    /// only a stamp to be checked against.
    #[must_use]
    pub fn of(stamp: &Stamp, model_id: Option<&str>) -> Option<Name> {
        let id = model_id.or(stamp.model_id.as_deref())?;
        Some(Name {
            slug: slugify(id),
            sku: slugify(&stamp.sku),
            backend: slugify(&stamp.backend),
            tp_size: stamp.tp_size,
            precision: slugify(&stamp.precision),
        })
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

/// A model id as a name field: lowercased, `/` becomes `--`, and every other
/// character outside `[a-z0-9_-]` becomes `-`.
///
/// A leading character outside `[a-z0-9]` is dropped rather than kept, because
/// the field grammar forbids one and a name that cannot be parsed back is not
/// a name.
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

// ── small pure helpers ──────────────────────────────────────────────────────

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

/// Which `pie.serving/<n>` a file states, when it is not the one this build
/// implements.
///
/// The only reason this scans rather than reading a member: the version is the
/// KEY now, so *"which version is this"* and *"is there a serving block at
/// all"* are one question over the key set. A file carrying two of them is
/// answered by the first in map order, which is deterministic and is a file
/// nothing this tree writes.
/// **WHICH SERVING PROFILE A FILE CLAIMS, IF ANY** — the predicate that keeps
/// "not a serving artifact" apart from "a serving artifact that is broken".
///
/// [`Stamp::decode`] answers three outcomes and returns two error variants, so
/// a caller reading only the `Result` cannot tell a plain checkpoint from an
/// artifact whose stamp has a rotted member: both arrive as
/// [`Error::Checkpoint`]. A boot that treats that as "no stamp, proceed" then
/// SERVES a file that claims to be servable and is not — which is the same
/// collapse `file/zt.rs`'s `affine_group_scheme` had when two schemes shared a
/// discriminator, one quiet notch further along.
///
/// So the claim is asked separately from the reading. `Some` means the file
/// states a `pie.serving/<n>` key — whether or not this build implements that
/// version, and whether or not its members are intact — and a caller that
/// gets `Some` and then a decode error is looking at a broken artifact, not
/// at an ordinary checkpoint.
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

fn optional_text(attributes: &Value, field: Field) -> Result<Option<&str>, Error> {
    match attributes.get(field.key()) {
        Some(Value::Text(it)) => Ok(Some(it)),
        Some(_) => Err(malformed(field, "is not text")),
        None => Ok(None),
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

#[cfg(test)]
mod tests {
    use super::*;

    const B: u64 = 64 * 1024 * 1024;

    fn stamp() -> Stamp {
        Stamp {
            serving: PROFILE.to_string(),
            backend: "cuda".to_string(),
            tp_size: 1,
            sku: "qwen_3".to_string(),
            precision: "mxfp4".to_string(),
            layout_revision: 7,
            block_bytes: B,
            block_algorithm: BlockAlgorithm::Xxh3,
            adapters_zeroed: true,
            model_id: Some("qwen--qwen3-30b-a3b".to_string()),
            recipe_digest: Some("xxh3:0123456789abcdef".to_string()),
        }
    }

    // ── the stamp ───────────────────────────────────────────────────────────

    #[test]
    fn a_stamp_round_trips_through_the_attribute_map() {
        let it = stamp();
        assert_eq!(Stamp::decode(&it.encode()).expect("decodes"), it);

        // And the optional pair is genuinely optional, absent rather than null.
        let bare = Stamp {
            model_id: None,
            recipe_digest: None,
            ..stamp()
        };
        let encoded = bare.encode();
        // One key at file level, and the members live inside it.
        assert!(matches!(&encoded, Value::Map(entries) if entries.len() == 1));
        let block = encoded.get(PROFILE).expect("the profile block");
        assert!(block.get(Field::ModelId.key()).is_none());
        assert!(block.get(Field::RecipeDigest.key()).is_none());
        assert!(block.get(Field::Backend.key()).is_some());
        assert_eq!(Stamp::decode(&encoded).expect("decodes"), bare);
    }

    /// The central argument of the profile, as a test: a fold can only say
    /// *different*, and the stamp says WHICH FACT.
    #[test]
    fn every_field_that_differs_names_itself() {
        let artifact = stamp();
        let moved: Vec<(Field, Stamp)> = vec![
            (
                Field::Backend,
                Stamp {
                    backend: "metal".into(),
                    ..stamp()
                },
            ),
            (Field::TpSize, Stamp { tp_size: 2, ..stamp() }),
            (
                Field::Sku,
                Stamp {
                    sku: "dsv4_flash".into(),
                    ..stamp()
                },
            ),
            (
                Field::Precision,
                Stamp {
                    precision: "bf16".into(),
                    ..stamp()
                },
            ),
            (
                Field::LayoutRevision,
                Stamp {
                    layout_revision: 8,
                    ..stamp()
                },
            ),
            (
                Field::BlockBytes,
                Stamp {
                    block_bytes: B * 2,
                    ..stamp()
                },
            ),
            (
                Field::BlockAlgorithm,
                Stamp {
                    block_algorithm: BlockAlgorithm::Sha256,
                    ..stamp()
                },
            ),
            (
                Field::RecipeDigest,
                Stamp {
                    recipe_digest: Some("xxh3:ffffffffffffffff".into()),
                    ..stamp()
                },
            ),
        ];
        for (field, deployment) in moved {
            let fault = artifact
                .check(&deployment)
                .expect_err("the stamps disagree about {field}");
            assert_eq!(fault.field, field, "named the wrong field");
            assert_eq!(fault.artifact, artifact.say(field));
            assert_eq!(fault.deployment, deployment.say(field));
            // The refusal is the deliverable: it names the field, both values,
            // that nothing was destroyed, and the command.
            let sentence = fault.refuse("/store/qwen.zt", Some("/snapshots/qwen"));
            assert!(sentence.contains(field.key()), "{sentence}");
            assert!(sentence.contains(&artifact.say(field)), "{sentence}");
            assert!(sentence.contains(&deployment.say(field)), "{sentence}");
            assert!(sentence.contains("nothing here deletes it"), "{sentence}");
            assert!(
                sentence.contains("pie model import --force /snapshots/qwen"),
                "{sentence}"
            );
        }
        assert!(artifact.check(&stamp()).is_ok(), "a stamp matches itself");
    }

    /// `model_id` is BELIEVED, not checked. A boot cannot check it, so a
    /// comparison that refused on it would refuse on a fact nobody verified.
    #[test]
    fn the_believed_field_is_never_compared() {
        let artifact = stamp();
        let deployment = Stamp {
            model_id: Some("someone-else--other-model".into()),
            ..stamp()
        };
        assert!(artifact.check(&deployment).is_ok());
        assert!(!artifact.recipe_unchecked(&deployment));
    }

    #[test]
    fn a_missing_recipe_digest_is_the_weaker_check_and_says_so() {
        let artifact = Stamp {
            recipe_digest: None,
            ..stamp()
        };
        assert!(artifact.check(&stamp()).is_ok(), "no digest, no disagreement");
        assert!(artifact.recipe_unchecked(&stamp()), "and the caller is told");
    }

    /// **The version is the KEY, so a newer file is answered by its name.**
    ///
    /// The block below is byte-for-byte a valid v1 block; only the key moved.
    /// Nothing in it is read, which is the whole point of putting the version
    /// where the reader looks first.
    #[test]
    fn a_future_profile_is_unsupported_and_not_malformed() {
        let mut entries = match stamp().encode() {
            Value::Map(entries) => entries,
            _ => unreachable!("a stamp encodes as a map"),
        };
        for (key, _) in &mut entries {
            *key = Value::Text("pie.serving/2".into());
        }
        let err = Stamp::decode(&Value::Map(entries)).expect_err("a version this build lacks");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "a newer file is re-imported, never called corrupt: {err}"
        );
        assert!(err.to_string().contains("re-import"), "{err}");
        assert!(err.to_string().contains("pie.serving/2"), "{err}");

        // And a file with no serving key at all is an ordinary checkpoint,
        // which is a different sentence: nothing to re-import, nothing broken.
        let err = Stamp::decode(&Value::Map(vec![(
            text("pie_source"),
            text("qwen/qwen3"),
        )]))
        .expect_err("not a serving artifact");
        assert!(matches!(err, Error::Checkpoint(_)), "{err}");
        assert!(err.to_string().contains("ordinary checkpoint"), "{err}");
    }

    #[test]
    fn the_required_keys_are_required_and_checked() {
        // A field whose value is out of range names itself and its range.
        for (broken, needle) in [
            (Stamp { tp_size: 0, ..stamp() }, "at least 1"),
            (
                Stamp {
                    block_bytes: 2048,
                    ..stamp()
                },
                "power of two",
            ),
            (
                Stamp {
                    block_bytes: B + 1,
                    ..stamp()
                },
                "power of two",
            ),
        ] {
            let err = Stamp::decode(&broken.encode()).expect_err("out of range");
            assert!(err.to_string().contains(needle), "{err}");
        }
        // Absence is not "false" — for every member of the block, and for the
        // block's own key, whose absence is the "not a serving artifact" arm.
        let block = match stamp().encode().get(PROFILE) {
            Some(Value::Map(entries)) => entries.clone(),
            other => unreachable!("the profile block is a map, not {other:?}"),
        };
        for field in Field::required() {
            let attributes = if *field == Field::Serving {
                Value::Map(Vec::new())
            } else {
                Value::Map(vec![(
                    text(PROFILE),
                    Value::Map(
                        block
                            .iter()
                            .filter(|(key, _)| key.as_text() != Some(field.key()))
                            .cloned()
                            .collect(),
                    ),
                )])
            };
            let err = Stamp::decode(&attributes).expect_err("{field} is required");
            assert!(err.to_string().contains(field.key()), "{field}: {err}");
        }
        // And a false one is refused rather than believed.
        let lying = Stamp {
            adapters_zeroed: false,
            ..stamp()
        };
        let err = Stamp::decode(&lying.encode()).expect_err("false is refused");
        assert!(err.to_string().contains("adapter"), "{err}");
    }

    // ── the block tables ────────────────────────────────────────────────────

    /// The adversarial sizes engine-cuda's tier.rs already used, over decoded
    /// sizes rather than reserved spans.
    const SIZES: [u64; 8] = [0, 1, 255, B - 1, B, B + 1, B * 8, B * 8 + 4097];

    #[test]
    fn the_blocks_of_a_part_tile_it_exactly_once() {
        for size in SIZES {
            let count = block_count(size, B);
            assert_eq!(count, size.div_ceil(B), "{size} bytes");
            let mut at = 0u64;
            for which in 0..count {
                let span = block_span(size, B, which).expect("inside the part");
                assert_eq!(span.start, at, "{size} bytes: block {which} starts late");
                assert!(
                    span.end > span.start && span.end - span.start <= B,
                    "{size} bytes: block {which} is {span:?}"
                );
                at = span.end;
            }
            assert_eq!(at, size, "{size} bytes: the blocks end where the part does");
            assert!(
                block_span(size, B, count).is_none(),
                "{size} bytes: there is no block past the end"
            );
        }
    }

    #[test]
    fn a_block_table_round_trips_at_every_adversarial_size() {
        for algorithm in [BlockAlgorithm::Xxh3, BlockAlgorithm::Sha256] {
            for size in SIZES {
                let count = block_count(size, B);
                // A distinct digest per block, so a misindexed read shows.
                let digests: Vec<Vec<u8>> = (0..count)
                    .map(|which| algorithm.digest(&which.to_le_bytes()))
                    .collect();
                let table = encode_blocks(algorithm, digests.iter().map(Vec::as_slice));
                assert_eq!(table.len(), table_len(size, B, algorithm), "{size} bytes");

                let blocks = Blocks::decode(algorithm, B, size, &table).expect("well formed");
                assert_eq!(blocks.count(), count);
                assert_eq!(blocks.is_empty(), count == 0);
                for which in 0..count {
                    assert_eq!(
                        blocks.digest(which).expect("in the table"),
                        digests[which as usize].as_slice(),
                        "{size} bytes: block {which}"
                    );
                    assert_eq!(blocks.digest(which).expect("width").len(), algorithm.width());
                }
                assert!(blocks.digest(count).is_none(), "no digest past the end");
                assert_eq!(blocks.iter().count() as u64, count);

                // The prefix property: a prefix of the sequence checks a
                // prefix of the table and touches nothing else.
                for take in 0..=count {
                    let prefix = blocks.prefix(take);
                    assert_eq!(prefix.len(), take as usize * algorithm.width());
                    assert_eq!(prefix, &table[..prefix.len()]);
                }
                assert_eq!(blocks.prefix(count + 5), table.as_slice());

                // A table of the wrong length is refused, and says both.
                let mut short = table.clone();
                short.push(0);
                let err = Blocks::decode(algorithm, B, size, &short).expect_err("too long");
                assert!(err.to_string().contains(&short.len().to_string()), "{err}");
            }
        }
    }

    #[test]
    fn first_blocks_indexes_a_concatenated_table_with_no_last_case() {
        let sizes = [B, 0, B * 8 + 4097, 1];
        let first = first_blocks(&sizes, B);
        assert_eq!(first.len(), sizes.len() + 1);
        for (at, size) in sizes.iter().enumerate() {
            assert_eq!(
                first[at + 1] - first[at],
                block_count(*size, B),
                "entry {at}'s slice is its own block count"
            );
        }
        assert_eq!(
            *first.last().expect("a sentinel"),
            sizes.iter().map(|size| block_count(*size, B)).sum::<u64>()
        );
    }

    #[test]
    fn a_block_digest_is_the_registered_algorithm_in_the_stated_width() {
        // xxh3 is 8 bytes LITTLE-endian, where the manifest spells it big.
        let spelled = DigestAlgorithm::Xxh3.digest(b"pie");
        let hex = spelled.split_once(':').expect("algo:hex").1;
        let value = u64::from_str_radix(hex, 16).expect("16 hex digits");
        assert_eq!(BlockAlgorithm::Xxh3.digest(b"pie"), value.to_le_bytes());
        assert_eq!(BlockAlgorithm::Sha256.digest(b"pie").len(), 32);
        // Different bytes, different digest — the table is worth carrying.
        assert_ne!(
            BlockAlgorithm::Xxh3.digest(b"pie"),
            BlockAlgorithm::Xxh3.digest(b"pi")
        );
    }

    // ── the tiling check ────────────────────────────────────────────────────

    fn span<'a>(object: &'a str, part: &'a str, offset: u64, length: u64) -> Span<'a> {
        Span {
            object,
            part,
            shard: None,
            offset,
            length,
        }
    }

    #[test]
    fn an_exactly_equal_alias_is_served_from_one_span() {
        // Weight tying: two objects, one blob, exactly equal (offset, length).
        let spans = [
            span("embed", "data", 4096, 4096),
            span("lm_head", "data", 4096, 4096),
            span("norm", "data", 8192, 4096),
        ];
        assert_eq!(tiling_fault(&spans, 4096), None, "§2.4 blesses this");
    }

    #[test]
    fn a_partial_overlap_and_a_gap_are_different_sentences() {
        let overlapping = [
            span("a", "data", 4096, 8192),
            span("b", "data", 8192, 4096), // starts inside `a`, not equal to it
        ];
        let fault = tiling_fault(&overlapping, 4096).expect("a partial overlap is refused");
        assert!(matches!(fault, Fault::Overlap { .. }), "{fault:?}");
        let overlap_says = fault.to_string();
        assert!(overlap_says.contains("overlaps"), "{overlap_says}");
        assert!(overlap_says.contains("a/data"), "{overlap_says}");
        assert!(overlap_says.contains("b/data"), "{overlap_says}");

        let gapped = [
            span("a", "data", 4096, 4096),
            span("b", "data", 4096 * 100, 4096), // a stranger's blob fits between
        ];
        let fault = tiling_fault(&gapped, 4096).expect("a gap is refused");
        assert!(matches!(fault, Fault::Gap { .. }), "{fault:?}");
        let gap_says = fault.to_string();
        assert!(gap_says.contains("gap"), "{gap_says}");

        assert_ne!(
            overlap_says, gap_says,
            "the single equality this replaces read both as one refusal"
        );
    }

    #[test]
    fn padding_up_to_the_alignment_is_not_a_gap() {
        // 2 MiB writer policy: a part of 4097 bytes pads to the next boundary.
        let align = 2 * 1024 * 1024;
        let spans = [
            span("a", "data", align, 4097),
            span("b", "data", align * 2, 16),
        ];
        assert_eq!(tiling_fault(&spans, align), None, "§2.4 owns those bytes");
        // One more boundary along and it is a hole, not padding.
        let holed = [
            span("a", "data", align, 4097),
            span("b", "data", align * 3, 16),
        ];
        assert!(matches!(
            tiling_fault(&holed, align),
            Some(Fault::Gap { .. })
        ));
        // With no padding policy at all, any gap is a gap.
        assert!(matches!(
            tiling_fault(&spans, 1),
            Some(Fault::Gap { .. })
        ));
    }

    #[test]
    fn a_tiled_run_with_no_padding_is_clean_and_its_derivations_agree() {
        let spans = [
            span("a", "data", 4096, 4096),
            span("b", "data", 8192, 100),
            span("b", "scales", 12288, 8),
        ];
        assert_eq!(tiling_fault(&spans, 4096), None);
        assert_eq!(payload_at(&spans), Some(4096));
        assert_eq!(payload_total(&spans), 12288 + 8 - 4096);
        assert_eq!(alignment(&spans), 4096);
        // The padded span is the next offset minus this one; the last part
        // pads to its own end, because nothing after it is inside the run.
        assert_eq!(padded_spans(&spans), vec![4096, 4096, 8]);
        assert_eq!(payload_at(&[]), None);
        assert_eq!(payload_total(&[]), 0);
        assert_eq!(alignment(&[]), 0);
    }

    #[test]
    fn the_alignment_is_a_power_of_two_read_off_the_offsets() {
        let two_mib = 2 * 1024 * 1024;
        let spans = [
            span("a", "data", two_mib, 16),
            span("b", "data", two_mib * 3, 16),
        ];
        assert_eq!(alignment(&spans), two_mib);
        // A gcd that carries an odd factor is not an alignment; the largest
        // power of two dividing it is.
        let odd = [span("a", "data", 3 * 4096, 16), span("b", "data", 9 * 4096, 16)];
        assert_eq!(alignment(&odd), 4096);
    }

    // ── the filename ────────────────────────────────────────────────────────

    #[test]
    fn a_filename_round_trips_including_a_slug_full_of_neighbours() {
        for name in [
            "qwen--qwen3-30b-a3b.qwen_3.cuda-tp1.mxfp4.zt",
            "deepseek-ai--dsv4.dsv4_flash.cuda-tp8.fp8_e4m3.zt",
            // The separator's neighbours, and a slug that impersonates every
            // field after it. Right-split makes it unambiguous.
            "cuda-tp4--qwen_3-mxfp4.qwen_3.metal-tp2.bf16.zt",
            "a.b.c-tp1.d.zt",
        ] {
            let parsed = super::Name::parse(name).expect("parses");
            assert_eq!(parsed.render(), name, "round trip");
        }

        let parsed = super::Name::parse("cuda-tp4--qwen_3-mxfp4.qwen_3.metal-tp2.bf16.zt")
            .expect("parses");
        assert_eq!(parsed.slug, "cuda-tp4--qwen_3-mxfp4");
        assert_eq!(parsed.sku, "qwen_3");
        assert_eq!(parsed.backend, "metal");
        assert_eq!(parsed.tp_size, 2);
        assert_eq!(parsed.precision, "bf16");
        assert_eq!(
            std::path::Path::new(&parsed.render())
                .extension()
                .and_then(|it| it.to_str()),
            Some("zt"),
            "`Path::extension` still answers `zt`"
        );
    }

    /// **A SKU WITH A DOT IN IT, WHICH IS EIGHT OF THE CATALOG'S ROWS.**
    ///
    /// `qwen35-d0.8b-mlxu4-kv-bf16` and its seven siblings hold a `.`, and `.`
    /// is what [`Name::parse`] splits on. The round-trip test above never
    /// caught it because every SKU it uses is invented and none has one — a
    /// gate passing because its fixture is unlike the thing it guards.
    ///
    /// [`Name::of`] is the answer and this is what says so: it slugs every
    /// field, so the filename carries `qwen35-d0-8b-…` where the stamp carries
    /// `qwen35-d0.8b-…`. The name is for a human reading a directory listing;
    /// the stamp is what a boot compares, and it keeps the exact string.
    ///
    /// **Building a `Name` field by field out of a stamp's raw values renders
    /// something `parse` cannot read**, which is exactly what `pie model
    /// import` did on its first run and what the round trip caught.
    #[test]
    fn a_sku_with_a_dot_survives_because_the_constructor_slugs_it() {
        let stamp = Stamp::of(
            "cuda",
            1,
            "qwen35-d0.8b-mlxu4-kv-bf16",
            "mlxu4",
            Some("Qwen/Qwen3.5-0.8B".to_string()),
        );
        let name = super::Name::of(&stamp, None).expect("a stamp with a model id names a file");
        let rendered = name.render();
        assert_eq!(
            rendered,
            "qwen--qwen3-5-0-8b.qwen35-d0-8b-mlxu4-kv-bf16.cuda-tp1.mlxu4.zt"
        );
        let parsed = super::Name::parse(&rendered).expect("what `of` renders, `parse` reads");
        assert_eq!(parsed.render(), rendered);
        assert_eq!(parsed.sku, "qwen35-d0-8b-mlxu4-kv-bf16");
        // And the raw sku, which is what a hand-built `Name` would have used,
        // does NOT survive — which is why `of` exists.
        let raw = super::Name {
            sku: stamp.sku.clone(),
            ..parsed
        };
        assert!(
            super::Name::parse(&raw.render()).is_err(),
            "a name holding the unslugged sku parses, so this test guards nothing"
        );
    }

    #[test]
    fn a_name_that_is_not_one_is_refused_with_what_is_missing() {
        for (name, needle) in [
            ("qwen.qwen_3.cuda-tp1.mxfp4.safetensors", "does not end"),
            ("qwen_3.cuda-tp1.mxfp4.zt", "names no sku"),
            ("qwen.qwen_3.cuda.mxfp4.zt", "-tp"),
            ("qwen.qwen_3.cuda-tpX.mxfp4.zt", "not a number"),
            ("qwen.qwen_3.cuda-tp0.mxfp4.zt", "degree of zero"),
            ("qwen.QWEN.cuda-tp1.mxfp4.zt", "not `[a-z0-9]"),
        ] {
            let err = super::Name::parse(name).expect_err("{name} is not a serving name");
            assert!(err.to_string().contains(needle), "{name}: {err}");
        }
    }

    /// Every field of the name is a field of the stamp, and no field of the
    /// stamp is only in the name — `layout_revision` above all.
    #[test]
    fn the_name_is_a_projection_of_the_stamp_and_omits_the_revision() {
        let it = stamp();
        let name = super::Name::of(&it, None).expect("a model id to slug");
        assert_eq!(name.sku, it.sku);
        assert_eq!(name.backend, it.backend);
        assert_eq!(name.tp_size, it.tp_size);
        assert_eq!(name.precision, it.precision);
        assert_eq!(name.slug, "qwen--qwen3-30b-a3b");
        assert_eq!(super::Name::parse(&name.render()).expect("parses"), name);

        let bumped = Stamp {
            layout_revision: it.layout_revision + 1,
            ..stamp()
        };
        assert_eq!(
            super::Name::of(&bumped, None).expect("a name").render(),
            name.render(),
            "a judgement that renamed files would orphan a store on every bump"
        );
        assert!(
            it.check(&bumped).is_err(),
            "and the stamp is what carries it"
        );

        assert_eq!(slugify("Qwen/Qwen3-30B-A3B"), "qwen--qwen3-30b-a3b");
        assert_eq!(slugify("-leading"), "leading");
        assert!(super::Name::of(&Stamp { model_id: None, ..stamp() }, None).is_none());
    }

    // ── identity ────────────────────────────────────────────────────────────

    fn manifest() -> Manifest {
        use ztensor::format::{BlobRef, Part};
        let mut objects = BTreeMap::new();
        let part = |offset: u64, digest: &str| Part {
            dtype: ztensor::DType::U8,
            logical: None,
            blob: BlobRef::local(offset, 4096),
            encoding: None,
            decoded_length: None,
            digest: Some(digest.to_string()),
        };
        for (name, offset, digest) in [
            ("layer.0.w", 4096u64, "xxh3:0000000000000001"),
            ("layer.1.w", 8192, "xxh3:0000000000000002"),
            ("__meta__/tokenizer", 12288, "xxh3:0000000000000003"),
        ] {
            objects.insert(
                name.to_string(),
                ztensor::Object {
                    shape: vec![4096],
                    layout: "dense".to_string(),
                    attributes: None,
                    parts: BTreeMap::from([("data".to_string(), part(offset, digest))]),
                },
            );
        }
        Manifest {
            attributes: Some(stamp().encode()),
            shards: BTreeMap::new(),
            objects,
        }
    }

    #[test]
    fn the_key_is_computed_and_moves_with_the_stamp_and_with_content() {
        let manifest = manifest();
        let key = identity(&stamp(), &manifest).expect("every part has a digest");
        assert!(key.starts_with("xxh3:"), "{key}");
        assert_eq!(
            identity(&stamp(), &manifest).expect("deterministic"),
            key,
            "a computed key cannot be false, so it must be a function"
        );

        // The stamp is folded in: the same weights for another specialization
        // are another artifact, which is what a store must key on.
        let other = Stamp {
            precision: "bf16".into(),
            ..stamp()
        };
        assert_ne!(identity(&other, &manifest).expect("keys"), key);

        // And content moves it too: the stamp alone is a description.
        let mut moved = manifest.clone();
        moved
            .objects
            .get_mut("layer.0.w")
            .expect("present")
            .parts
            .get_mut("data")
            .expect("present")
            .digest = Some("xxh3:00000000000000ff".into());
        assert_ne!(identity(&stamp(), &moved).expect("keys"), key);

        // The ORDER is deliberately out: moving a blob without moving a byte
        // of content leaves the key alone, so a prefetch-heuristic change no
        // longer orphans every artifact on the disk.
        let mut reordered = manifest.clone();
        for (at, object) in reordered.objects.values_mut().enumerate() {
            object.parts.get_mut("data").expect("present").blob.offset =
                4096 * (9 - at as u64);
        }
        assert_eq!(
            identity(&stamp(), &reordered).expect("keys"),
            key,
            "the boot reads the order from the manifest instead of assuming it"
        );
    }

    #[test]
    fn a_part_with_no_digest_has_no_key() {
        let mut manifest = manifest();
        manifest
            .objects
            .get_mut("layer.0.w")
            .expect("present")
            .parts
            .get_mut("data")
            .expect("present")
            .digest = None;
        let err = identity(&stamp(), &manifest).expect_err("nothing stands for the content");
        assert!(err.to_string().contains("layer.0.w"), "{err}");
    }

    #[test]
    fn the_sequence_is_recovered_by_sorting_and_skips_the_metadata() {
        let manifest = manifest();
        assert_eq!(sequence(&manifest), vec!["layer.0.w", "layer.1.w"]);
        let spans = spans(&manifest);
        assert_eq!(spans.len(), 2, "the tokenizer is not served");
        assert_eq!(spans[0].object, "layer.0.w");
        assert_eq!(spans[0].part, "data");
        assert_eq!(spans[0].end(), 8192);
        assert!(is_serving("layer.0.w"));
        assert!(!is_serving("__meta__/tokenizer"));

        // Laid out hottest-last, the derivation reads it back that way: the
        // ranking is in the file's offsets and nowhere else.
        let mut reversed = manifest.clone();
        reversed
            .objects
            .get_mut("layer.0.w")
            .expect("present")
            .parts
            .get_mut("data")
            .expect("present")
            .blob
            .offset = 65536;
        assert_eq!(sequence(&reversed), vec!["layer.1.w", "layer.0.w"]);
    }
}

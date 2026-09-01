//! **Reading a `pie.serving/1` artifact** — the lean half of the pair.
//!
//! [`Artifact`] is a serving artifact, open and mapped. It answers what a cut
//! needs: what this file is, which planes it carries in which order, where one
//! of them lies, and — only when asked — whether the bytes are the bytes that
//! were written.
//!
//! # AN OPEN IS NOT A VERIFICATION, AND THAT IS THE WHOLE DESIGN
//!
//! [`Artifact::open`] reads the manifest and the stamp. It hashes nothing.
//! Verification is asked for — [`Artifact::verify`] over the objects a boot is
//! about to serve, [`Artifact::verify_prefix`] over `[0, c1)` of the sequence,
//! [`Artifact::verify_all`] over the file — and it is never implied.
//!
//! That split is not an optimisation; it is what makes partial verify possible
//! at all. Format 3 had it (`Artifact::open` validated the header and the
//! index and left the payload alone; only `verify_all` closed the fold), and
//! it is preserved exactly, with one change forced by §6.4: **there is no
//! stored fold to close.** The artifact's key is computed and never written
//! down ([`serving::identity`]), so the whole-file check is every serving
//! part's blocks plus §3.4's per-part `digest` — the anchor identity rests on
//! — and [`Artifact::verify_all`] is the only door that reaches it.
//!
//! A boot that opens a hundred gigabytes and pumps `[0, c1)` pays for `c1`'s
//! blocks and for nothing else. An open that verified would put a minute of
//! one core in front of the first token, every time, for a file the operator
//! already asked to serve.
//!
//! # Two doors, and they are different doors on purpose
//!
//! [`read_head`] opens **no mapping at all**: it wants the manifest and the
//! stamp, and a caller asking only "what is this file for" should not pay a
//! mapping to find out. [`Artifact`] maps, because spans are served where they
//! lie — `Held::Mapped`'s whole premise is that the bytes on the disk ARE the
//! bytes the kernel reads, and a `&[u8]` handed out by [`Artifact::part`]
//! points into the mapping with nothing copied.
//!
//! `weight_cache/tier.rs` kept the same two doors for the same reason, and the
//! one thing that changes here is who owns the mapping: the container's store
//! does, so this module writes no second `mmap` somebody would have to keep in
//! step with the first.
//!
//! [`read_spans_into`] is the third, and it is neither: positioned reads
//! straight into a caller's destinations, verifying each block from the bytes
//! that landed there rather than from a second copy.
//!
//! # ONE FORMAT, AND A NAMED REFUSAL FOR EVERYTHING ELSE
//!
//! This reader reads `pie.serving/1` or it refuses. There is no version
//! dispatch, no translation arm and no shim, because there is nothing to be
//! compatible with: a `.zt` is rebuildable from its source at any time, so an
//! artifact this build cannot read is not a migration and is a re-import. A
//! code path whose only caller would be a file this build did not write is a
//! path nothing tests and nothing needs.
//!
//! **Refusing well is not compatibility**, and it is required. A file whose
//! serving key is a version this build does not read, and a file that carries
//! no serving key at all, are two different sentences, and both are said in
//! `tier::refuse`'s three parts: what is wrong, that nothing here rewrote or
//! deleted it, and the command that would fix it. That is [`refuse`], and it
//! is the whole of what this module does about other versions.
//!
//! # The version is refused before any other field is believed
//!
//! Two versions, two doors, one discipline — `weight_cache/tier.rs`'s
//! `read_head`, whose comment says it in one line: *"every field after the
//! format word means whatever the format says it means."*
//!
//! 1. **The container's.** A footer this build cannot read arrives as
//!    `Reject { rule: Rule::Version }` and `error.rs` maps exactly that rule to
//!    [`Error::Unsupported`] — landed in `563e5e74d`, before any container
//!    change, so that an operator holding an older file is told to re-import
//!    and never that it is corrupt.
//! 2. **The profile's.** The profile version is the attribute KEY —
//!    [`serving::PROFILE`] — so [`Stamp::decode`] answers it before a single
//!    member is looked at, and cannot do otherwise: a `pie.serving/2` block is
//!    a key this build does not have, and it answers [`Error::Unsupported`]
//!    rather than reading members it has no contract for. Nothing in this
//!    module reads an attribute ahead of it.
//!
//! # The stamp check is field by field, and the refusal says which field
//!
//! [`Artifact::check`] is [`Stamp::check`], which answers a [`Mismatch`]
//! naming the field and both values; [`Mismatch::refuse`] turns it into
//! `tier::refuse`'s three-part sentence — what is wrong, that nothing here
//! rewrote or deleted it, and the command that would fix it. A `u64` key could
//! only ever have answered *same or not*.

use std::ops::Range;
use std::path::{Path, PathBuf};

use ztensor::Manifest;

use crate::error::Error;
use crate::serving::{self, BLOCKS_KEY, Blocks, Mismatch, PROFILE, Span, Stamp};

/// **Concurrent readers on the verify and fill paths.**
///
/// `weight_cache/tier.rs`'s `TIER_READERS`, with its measurement: eight
/// concurrent block-sized positioned reads is where the box's filesystem stops
/// answering faster (5.41 GB/s). It is a property of the machine and not of
/// the file, which is why `block_bytes` is stated in the artifact and this
/// number is not — the mistake format 2 made was putting the stripe count in
/// the header, and re-measuring this must never invalidate a file.
pub const READERS: usize = 8;

/// **A serving artifact, open and mapped.**
///
/// Holds the container's own source — and therefore its mapping — for as long
/// as it lives. **Nothing is copied.** A `&[u8]` handed out here points into
/// the mapping, and the first touch of each page is the NVMe read.
pub struct Artifact {
    path: PathBuf,
    stamp: Stamp,
    source: ztensor::Source,
}

impl std::fmt::Debug for Artifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Artifact")
            .field("path", &self.path)
            .field("stamp", &self.stamp)
            .finish_non_exhaustive()
    }
}

impl Artifact {
    /// **Open the serving artifact at `path` and map it.**
    ///
    /// Reads the manifest — the container checks its own footer, its version
    /// and its manifest hash on the way — and then the stamp, which is read
    /// BY ITS KEY, so the profile version is answered before a member of it is
    /// believed.
    ///
    /// **It hashes nothing.** See this module's own doc: an open is not a
    /// verification, and the three verify doors below are where bytes are
    /// checked.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a container version or a `pie.serving/<n>`
    /// this build does not implement — both mean *re-import*, never
    /// *malformed*. [`Error::Checkpoint`] for a file that is not a serving
    /// artifact at all (an ordinary `.zt` carries no [`serving::PROFILE`]
    /// attribute), for a required member missing or of the wrong type, and for
    /// whatever the container refuses.
    pub fn open(path: &Path) -> Result<Artifact, Error> {
        let source = ztensor::Source::open(path).map_err(Error::from)?;
        let attributes = source.attributes().ok_or_else(|| {
            Error::Checkpoint(format!(
                "{} states no file attributes, so it carries no {PROFILE:?} block \
                 and is not a serving artifact",
                path.display(),
            ))
        })?;
        let stamp = Stamp::decode(attributes).map_err(|why| refuse(path, why))?;
        if source.provenance().as_root().is_none() {
            return Err(Error::Checkpoint(format!(
                "{} carries no manifest of its own, so nothing in it says where a \
                 plane lies",
                path.display(),
            )));
        }
        Ok(Artifact {
            path: path.to_path_buf(),
            stamp,
            source,
        })
    }

    /// Where this artifact came from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// **What this file says it is for** — the `pie_*` stamp, decoded.
    #[must_use]
    pub fn stamp(&self) -> &Stamp {
        &self.stamp
    }

    /// The manifest, which is where every derivation below reads from.
    ///
    /// Exposed rather than mirrored: [`serving`]'s derivations are total
    /// functions of this value, and a second copy of any of them here would be
    /// a second definition to keep in step.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        self.source
            .provenance()
            .as_root()
            .expect("open refused a source with no manifest")
    }

    /// **Is this artifact for this deployment?** — field by field, refusing on
    /// the first disagreement.
    ///
    /// The [`Mismatch`] names the field and both values; [`Mismatch::refuse`]
    /// spells the operator's sentence, and [`Artifact::path`] is the artifact
    /// name it wants.
    ///
    /// # Errors
    ///
    /// The first [`serving::Field`] that differs.
    pub fn check(&self, deployment: &Stamp) -> Result<(), Mismatch> {
        self.stamp.check(deployment)
    }

    /// **The serving sequence** — [`serving::sequence`], recovered from the
    /// manifest's offsets rather than believed from a stored order.
    ///
    /// Position *i* is hotter than position *i + 1*, and a reader that ignored
    /// this entirely would perform a correct, merely unranked, load.
    #[must_use]
    pub fn sequence(&self) -> Vec<&str> {
        serving::sequence(self.manifest())
    }

    /// Every serving part, in [`Artifact::sequence`] order.
    #[must_use]
    pub fn spans(&self) -> Vec<Span<'_>> {
        serving::spans(self.manifest())
    }

    /// **The padded span of each part** — [`serving::padded_spans`], the next
    /// offset minus this one, which is what `Group::reserved` became.
    #[must_use]
    pub fn padded_spans(&self) -> Vec<u64> {
        serving::padded_spans(&self.spans())
    }

    /// **The alignment its writer used**, read off the offsets — §2.4's
    /// *"observable from the offsets themselves"*, which is what a caller
    /// sizing a pinned buffer wants and what `TIER_ALIGN` stopped being.
    #[must_use]
    pub fn alignment(&self) -> u64 {
        serving::alignment(&self.spans())
    }

    /// **The payload run**: the first serving blob in the containing file to
    /// the end of the last. Empty when the file serves nothing.
    #[must_use]
    pub fn payload(&self) -> Range<u64> {
        let spans = self.spans();
        let at = serving::payload_at(&spans).unwrap_or(0);
        at..at.saturating_add(serving::payload_total(&spans))
    }

    /// **The artifact's key** — [`serving::identity`], computed here and
    /// stored nowhere, on §6.4's rule that a stored value is a claim that can
    /// be false.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] when a part carries no digest, which §3 makes a
    /// MUST for exactly this reason.
    pub fn identity(&self) -> Result<String, Error> {
        serving::identity(&self.stamp, self.manifest())
    }

    /// **One part's PUBLISHED bytes, borrowed from the mapping.**
    ///
    /// The blob's own length and not its padded span: the bytes past the
    /// published length are the spec-zero padding no reader should be handed,
    /// which is `tier::Artifact::plane`'s distinction between `bytes` and
    /// `reserved`, kept.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] for a name this file does not carry;
    /// [`Error::Unsupported`] when the bytes have no zero-copy view — an
    /// encoded part, which §3 forbids in a serving artifact, or a store this
    /// build could not map.
    pub fn part(&self, object: &str, part: &str) -> Result<&[u8], Error> {
        self.source
            .tensor(object)
            .and_then(|tensor| tensor.part(part))
            .and_then(|part| part.map())
            .map_err(Error::from)
    }

    /// **One part's block digest table**, borrowed from the attribute that
    /// holds it, with its length checked against the part it describes.
    ///
    /// §9 step 8, as one call: inside the FILE's [`serving::PROFILE`] block,
    /// [`BLOCKS_KEY`] is a map from object name to a map from part name to a
    /// byte string of `ceil(decoded_size / block_bytes)` digests of the file's
    /// own algorithm.
    ///
    /// The tables are at file level and not on the objects because an
    /// object's attributes are frozen at declaration, and a writer that had to
    /// know a table before declaring the object could not stream — see
    /// [`serving::BLOCKS_KEY`] for the plane that made that a blocker rather
    /// than an inconvenience. The read costs nothing extra: this artifact
    /// already holds the file attributes, because that is where its stamp is.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] when the object or part is absent, when the file
    /// states no table for them, when the table is not a byte string, or when
    /// it is not exactly as long as the part's decoded size says.
    pub fn blocks(&self, object: &str, part: &str) -> Result<Blocks<'_>, Error> {
        let entry = self.manifest().objects.get(object).ok_or_else(|| {
            Error::Checkpoint(format!(
                "the serving artifact {} carries no object {object:?}",
                self.path.display(),
            ))
        })?;
        let size = entry
            .parts
            .get(part)
            .ok_or_else(|| {
                Error::Checkpoint(format!(
                    "the serving artifact's {object:?} carries no part {part:?}"
                ))
            })?
            .decoded_size();
        let digests = self
            .manifest()
            .attributes
            .as_ref()
            .and_then(|it| serving::stated_blocks(it, object, part))
            .ok_or_else(|| {
                Error::Checkpoint(format!(
                    "the serving artifact states no `{PROFILE}` {BLOCKS_KEY} for \
                     {object:?}'s {part:?}, so no prefix of it can be verified without \
                     hashing the rest"
                ))
            })?;
        Blocks::decode(self.stamp.block_algorithm, self.stamp.block_bytes, size, digests)
    }

    /// **Hash the blocks of these objects and compare them to their tables.**
    ///
    /// The subset verify: a boot checks the planes it is about to serve and
    /// touches nothing else. Every part of every named object, [`READERS`]
    /// blocks at a time, and the refusal names the object, the part, the block
    /// within that part and its byte range — "the digest is wrong" about a
    /// hundred gigabytes says less than it could when the payload is already
    /// divided into an answer per block.
    ///
    /// It reads the pages it hashes, which is what makes it also the thing
    /// that warms the page cache the first fires would otherwise fault in one
    /// page at a time.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] naming the first block whose bytes disagree with
    /// the table, or an object this file does not carry.
    pub fn verify(&self, objects: &[&str]) -> Result<(), Error> {
        let mut work = Vec::new();
        for object in objects {
            self.gather(object, &mut work)?;
        }
        self.hash(&work)
    }

    /// **Verify a PREFIX of the sequence and nothing after it.**
    ///
    /// The property the whole granularity exists for. The serving parts tile
    /// the payload run in sequence order, so the blocks of a prefix of the
    /// sequence are a prefix of the concatenated tables — a boot about to
    /// serve `[0, c1)` verifies exactly that.
    ///
    /// `objects` past the end of the sequence verifies the whole sequence
    /// rather than refusing: a prefix longer than the list is the list.
    ///
    /// # Errors
    ///
    /// As [`Artifact::verify`].
    pub fn verify_prefix(&self, objects: usize) -> Result<(), Error> {
        let sequence = self.sequence();
        let prefix: Vec<&str> = sequence.into_iter().take(objects).collect();
        self.verify(&prefix)
    }

    /// **Hash every block of every serving part, AND check the anchor.**
    ///
    /// [`Artifact::verify`] over the whole sequence, plus §3.4's per-part
    /// `digest` — which is the one thing the block tables do not stand for.
    /// The two are carried on purpose and answer different questions:
    /// the block table is the working check a cut runs on a subset, and
    /// `digest` is what [`serving::identity`] folds, so a file whose blocks
    /// agree and whose part digests do not is a file with two accounts of
    /// itself.
    ///
    /// There is no single number to compare instead, and there must not be:
    /// §6.4 forbids storing the fold, on the rule that a stored value is a
    /// claim that can be false. So the whole-file check is the two things that
    /// ARE in the file, both of them checks against bytes.
    ///
    /// It reads the whole payload twice over — once per block table, once per
    /// part digest — so a caller that means to serve lazily out of the mapping
    /// pays for it deliberately. `--delete-source` is the path that should.
    ///
    /// # Errors
    ///
    /// The first block that disagrees, or the first part whose own digest
    /// does.
    pub fn verify_all(&self) -> Result<(), Error> {
        let sequence = self.sequence();
        self.verify(&sequence)?;
        for name in &sequence {
            let tensor = self.source.tensor(name).map_err(Error::from)?;
            tensor.verify().map_err(Error::from)?;
        }
        Ok(())
    }

    /// One object's work items: every part, every block, with the bytes and
    /// the stored digest side by side.
    fn gather<'a>(&'a self, object: &'a str, into: &mut Vec<Work<'a>>) -> Result<(), Error> {
        let entry = self.manifest().objects.get(object).ok_or_else(|| {
            Error::Checkpoint(format!(
                "the serving artifact {} carries no object {object:?}, so there is \
                 nothing here to verify",
                self.path.display(),
            ))
        })?;
        if !serving::is_serving(object) {
            return Err(Error::Checkpoint(format!(
                "{object:?} is a metadata object, which is not served and carries no \
                 `{PROFILE}` block"
            )));
        }
        for part in entry.parts.keys() {
            let blocks = self.blocks(object, part)?;
            let bytes = self.part(object, part)?;
            for (span, stated) in blocks.iter() {
                let from = usize::try_from(span.start).unwrap_or(usize::MAX);
                let upto = usize::try_from(span.end).unwrap_or(usize::MAX);
                into.push(Work {
                    object,
                    part,
                    which: span.start / self.stamp.block_bytes.max(1),
                    span,
                    bytes: bytes.get(from..upto).unwrap_or(&[]),
                    stated,
                });
            }
        }
        Ok(())
    }

    /// The work list, hashed [`READERS`] lanes wide and compared in order.
    ///
    /// One item per BLOCK rather than per part, for `tier::verify_at`'s
    /// reason: the parts are as uneven as the model's planes are, and a lane
    /// that took a part each would spend its last minutes with seven idle
    /// cores behind the largest bank.
    fn hash(&self, work: &[Work<'_>]) -> Result<(), Error> {
        let algorithm = self.stamp.block_algorithm;
        let found: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let width = READERS.min(work.len().max(1));
            let per = work.len().div_ceil(width.max(1));
            let mut hashing = Vec::with_capacity(width);
            for lane in 0..width {
                let mine = work
                    .get(lane * per..((lane + 1) * per).min(work.len()))
                    .unwrap_or(&[]);
                hashing.push(scope.spawn(move || {
                    mine.iter()
                        .map(|item| algorithm.digest(item.bytes))
                        .collect::<Vec<Vec<u8>>>()
                }));
            }
            hashing
                .into_iter()
                .flat_map(|thread| thread.join().unwrap_or_default())
                .collect()
        });
        for (item, found) in work.iter().zip(&found) {
            if found.as_slice() != item.stated {
                return Err(Error::Checkpoint(item.refusal(
                    self.path.as_path(),
                    algorithm,
                    found,
                )));
            }
        }
        Ok(())
    }
}

/// One block of one part, and the two answers about it.
struct Work<'a> {
    object: &'a str,
    part: &'a str,
    /// The block's place IN ITS OWN PART and not in the table: "block 3 of
    /// `embed`'s data" about a part with one block is a sentence an operator
    /// cannot act on, and the part-local ordinal beside a byte range is the
    /// address they can.
    which: u64,
    span: Range<u64>,
    bytes: &'a [u8],
    stated: &'a [u8],
}

impl Work<'_> {
    fn refusal(&self, path: &Path, algorithm: serving::BlockAlgorithm, found: &[u8]) -> String {
        format!(
            "the serving artifact {} states {} {} for block {} of {:?}/{:?} (part bytes \
             {}..{}), whose bytes hash to {}",
            path.display(),
            algorithm.as_str(),
            hex(self.stated),
            self.which,
            self.object,
            self.part,
            self.span.start,
            self.span.end,
            hex(found),
        )
    }
}

/// **Read a serving artifact's manifest and stamp WITHOUT mapping it.**
///
/// The positioned-read door. `weight_cache/tier.rs`'s `read_head` exists
/// because a caller that only wants to know whether a deployment is on the
/// disk should not pay a mapping to find out, and that is still true: the
/// container indexes the file, reads its footer and its manifest blob with
/// positioned reads, and opens no mapping at all.
///
/// It is not [`Artifact::open`]'s first step, and deliberately not: under
/// zTensor the manifest read IS the container's, so calling this from `open`
/// would parse the same bytes twice for nothing. What differs between the two
/// doors is the mapping, and that is all that differs.
///
/// # Errors
///
/// As [`Artifact::open`], minus anything only a mapping can produce.
/// **THE THREE OUTCOMES, KEPT APART** — is this file a serving artifact, and
/// if it says it is, does its stamp read?
///
/// `Ok(None)` is an ordinary checkpoint: no `pie.serving/<n>` key at all. A
/// caller may serve it by whatever path it had before this profile existed.
///
/// `Ok(Some)` is a serving artifact whose stamp this build reads.
///
/// `Err` is a file that CLAIMS to be one and is not readable as one — a
/// version this build does not implement ([`Error::Unsupported`], meaning
/// re-import), or a member that is missing, mistyped or out of range. A path
/// that is not a zTensor container at all — a snapshot directory, a GGUF, a
/// `.safetensors` — claims nothing and is `Ok(None)`, not an error: most of
/// what a boot is handed is one of those.
///
/// # Why this exists beside [`read_head`]
///
/// [`read_head`] returns `Result<(Stamp, Manifest)>` and [`Stamp::decode`]
/// spends two error variants on three outcomes, so a caller reading only that
/// `Result` cannot tell "no serving key" from "serving key, broken member" —
/// both are [`Error::Checkpoint`]. Every boot written against `read_head`
/// therefore treats a rotted stamp as an unstamped checkpoint and serves it,
/// which is precisely the silent failure the stamp exists to end.
///
/// Found by the mac-engine session reading the split from the other shell, on
/// a call site of mine that had it wrong.
///
/// # Errors
///
/// As above. A file that cannot be opened or has no manifest is an error
/// under every reading, and arrives here as one.
pub fn stamp_of(path: &Path) -> Result<Option<Stamp>, Error> {
    // **A PATH THAT IS NOT A ZTENSOR CONTAINER CLAIMS NOTHING**, and that is
    // most of what a boot is handed: a safetensors SNAPSHOT DIRECTORY, a
    // GGUF, a single `.safetensors`. `manifest_of` answers those with an io
    // error (`Is a directory`) or a rejected magic, and reading that as a
    // refusal makes every ordinary checkpoint unbootable — which is exactly
    // what it did, caught by `cuda_boot_smoke` on a device and by nothing
    // host-side, because every host fixture in this crate is already a
    // container.
    //
    // The distinction this function exists for is NARROWER than "did the read
    // succeed": it is between a file that says nothing about serving and one
    // that SAYS IT IS SERVABLE and is not readable as such. Only the second is
    // an error here. A corrupt container that claims nothing is refused by
    // whatever opens it next, with a better sentence than this one could give.
    let Ok(Some(manifest)) = ztensor::read::manifest_of(path) else {
        return Ok(None);
    };
    let Some(attributes) = manifest.attributes.as_ref() else {
        return Ok(None);
    };
    if serving::stated_profile(attributes).is_none() {
        return Ok(None);
    }
    Stamp::decode(attributes).map(Some)
}

pub fn read_head(path: &Path) -> Result<(Stamp, Manifest), Error> {
    let manifest = ztensor::read::manifest_of(path)
        .map_err(Error::from)?
        .ok_or_else(|| {
            Error::Checkpoint(format!(
                "{} carries no manifest, so nothing in it says what it is for",
                path.display(),
            ))
        })?;
    let attributes = manifest.attributes.as_ref().ok_or_else(|| {
        Error::Checkpoint(format!(
            "{} states no file attributes, so it carries no {PROFILE:?} block and is \
             not a serving artifact",
            path.display(),
        ))
    })?;
    let stamp = Stamp::decode(attributes).map_err(|why| refuse(path, why))?;
    Ok((stamp, manifest))
}

/// **One part, and where its bytes are to be read** — [`read_spans_into`]'s
/// unit of work.
///
/// A WHOLE PART and never a piece of one, because the block digests are
/// part-local: a partial read could not close the block it stopped inside, and
/// a reader that skipped that block's check would be verifying less than the
/// bytes it used.
pub struct Fill<'a> {
    /// The serving object.
    pub object: &'a str,
    /// The part within it.
    pub part: &'a str,
    /// Where the bytes go. Exactly the part's published length; the padding is
    /// not read, because the blocks do not cover it (departure #1).
    pub into: *mut u8,
}

/// One block on its way into a caller's buffer: which file it comes from,
/// where in that file, which part-local bytes it covers, what it should hash
/// to, and where it goes.
struct Landing<'a> {
    /// Index into the open files, so a sharded artifact costs one descriptor
    /// per shard rather than one per block.
    file: usize,
    /// The absolute offset in that file, which is the blob's plus the block's.
    at: u64,
    /// The block's range within its part, which is what a refusal prints.
    span: Range<u64>,
    /// The digest the artifact states for it.
    stated: &'a [u8],
    into: Carried,
}

/// A destination, as something a scope thread may carry. A raw pointer is not
/// `Send` and must not be; what makes these sound to move is
/// [`read_spans_into`]'s own safety clause, which says the destinations are
/// disjoint and nobody else names a byte of them.
struct Carried(*mut u8);

// SAFETY: as above — one address per block, moved once, into a thread that is
// the sole writer of the bytes it names for as long as the scope is open.
unsafe impl Send for Carried {}
// SAFETY: the work list is shared by reference and sliced DISJOINTLY, one
// slice per lane, so no two threads ever reach the same `Carried` — and the
// caller's clause says no two fills name the same byte either.
unsafe impl Sync for Carried {}

/// **READ THESE PARTS STRAIGHT INTO THEIR DESTINATIONS, VERIFYING THEM AS THEY
/// ARRIVE.**
///
/// [`READERS`] threads over the work, each reading its own blocks with
/// positioned reads a `block_bytes` at a time and closing that block's
/// digest from the bytes as they land. **No staging buffer exists anywhere on
/// this path**: the read target IS the destination, and the digest is taken
/// over what is now in that destination rather than over a second copy from
/// the mapping. That is the stronger claim of the two — it verifies the bytes
/// the caller will USE — and it is also the only one that costs a single pass
/// over the disk.
///
/// **WHY THE BLOCK IS THE UNIT AND NOT A CHUNK OF ONE.** A block's digest is
/// serial over its own span, so a reader may only feed the block it is
/// walking, in order. Splitting a block further would buy queue depth and lose
/// the overlap.
///
/// **EVERY BYTE OF EVERY FILL IS WRITTEN BEFORE THIS RETURNS `Ok`**, because
/// the blocks tile the part exactly. On `Err` the destinations hold an
/// indeterminate mixture and the caller owes them a zeroing before anything
/// reads them.
///
/// # Safety
///
/// Each fill's `into` must be valid for writes of its part's decoded length,
/// the fills' destinations must be pairwise disjoint, and no other agent — no
/// thread, no kernel, no guest — may read or write any of them for the
/// duration of the call.
///
/// # Errors
///
/// [`Error::Checkpoint`] for the filesystem's own words, which is a machine
/// failure and not a claim about the file, and for a block that does not hash
/// to its own table, which IS one.
pub unsafe fn read_spans_into(artifact: &Artifact, fills: &[Fill<'_>]) -> Result<(), Error> {
    use std::os::unix::fs::FileExt;

    if fills.is_empty() {
        return Ok(());
    }
    // One work item per BLOCK, for the same reason the verify has one: the
    // parts are as uneven as the model's planes are.
    //
    // The names are kept BESIDE the work rather than inside it, and that is
    // not tidiness: `Fill` holds a raw pointer, so a `&Fill` is neither `Send`
    // nor `Sync` and a work item carrying one could not cross into a lane at
    // all. What crosses is [`Carried`], whose safety clause is this function's.
    let mut files: Vec<std::fs::File> = Vec::new();
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut work: Vec<Landing<'_>> = Vec::new();
    let mut named: Vec<(&str, &str)> = Vec::new();
    for fill in fills {
        let located = artifact
            .source
            .tensor(fill.object)
            .and_then(|tensor| tensor.part(fill.part))
            .and_then(|part| part.locate())
            .map_err(Error::from)?;
        let path = artifact.source.store(located.store).path().to_path_buf();
        let which = match paths.iter().position(|known| *known == path) {
            Some(which) => which,
            None => {
                let file = std::fs::File::open(&path)
                    .map_err(|why| Error::Checkpoint(format!("{}: {why}", path.display())))?;
                paths.push(path);
                files.push(file);
                files.len() - 1
            }
        };
        for (span, stated) in artifact.blocks(fill.object, fill.part)?.iter() {
            // SAFETY: `span.end <= blocks.size()`, which is the part's decoded
            // length, and the caller states `into` is valid for that many
            // bytes.
            let into = Carried(unsafe {
                fill.into
                    .add(usize::try_from(span.start).unwrap_or(usize::MAX))
            });
            work.push(Landing {
                file: which,
                at: located.offset + span.start,
                span,
                stated,
                into,
            });
            named.push((fill.object, fill.part));
        }
    }

    let algorithm = artifact.stamp.block_algorithm;
    let outcomes: Vec<Result<(usize, Vec<u8>), String>> = std::thread::scope(|scope| {
        let width = READERS.min(work.len().max(1));
        let per = work.len().div_ceil(width.max(1));
        let mut reading = Vec::with_capacity(width);
        for lane in 0..width {
            let from = lane * per;
            let mine = work.get(from..(from + per).min(work.len())).unwrap_or(&[]);
            let files = &files;
            reading.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(mine.len());
                for (at, landing) in mine.iter().enumerate() {
                    let len =
                        usize::try_from(landing.span.end - landing.span.start).unwrap_or(0);
                    // SAFETY: the caller's clause, plus the disjointness the
                    // block walk above preserves — no two work items name the
                    // same byte of any destination.
                    let destination =
                        unsafe { core::slice::from_raw_parts_mut(landing.into.0, len) };
                    let read = files[landing.file].read_exact_at(destination, landing.at);
                    out.push(match read {
                        Ok(()) => Ok((from + at, algorithm.digest(destination))),
                        Err(why) => Err(format!("{why}")),
                    });
                }
                out
            }));
        }
        reading
            .into_iter()
            .flat_map(|thread| {
                thread
                    .join()
                    .unwrap_or_else(|_| vec![Err("a read worker panicked".to_string())])
            })
            .collect()
    });

    let mut failure: Option<String> = None;
    let mut rotten: Option<(usize, Vec<u8>)> = None;
    for outcome in outcomes {
        match outcome {
            Ok((at, digest)) => {
                if work[at].stated != digest.as_slice() && rotten.is_none() {
                    rotten = Some((at, digest));
                }
            }
            Err(why) => failure = failure.or(Some(why)),
        }
    }
    // The filesystem's answer first: a read that did not happen leaves a
    // destination nobody wrote, and calling that a corruption would name the
    // file for the machine's failure.
    if let Some(why) = failure {
        return Err(Error::Checkpoint(format!(
            "reading the serving artifact {}: {why}",
            artifact.path.display(),
        )));
    }
    if let Some((at, found)) = rotten {
        let landing = &work[at];
        let (object, part) = named[at];
        return Err(Error::Checkpoint(format!(
            "the serving artifact {} states {} {} for block {} of {object:?}/{part:?} \
             (part bytes {}..{}), which read back as {}",
            artifact.path.display(),
            algorithm.as_str(),
            hex(landing.stated),
            landing.span.start / artifact.stamp.block_bytes.max(1),
            landing.span.start,
            landing.span.end,
            hex(&found),
        )));
    }
    Ok(())
}

/// **A refusal about the file itself, in `tier::refuse`'s three parts**: what
/// is wrong, that nothing here rewrote or deleted it, and the command that
/// would fix it.
///
/// [`Mismatch::refuse`] says this for a stamp that disagrees; this says it for
/// a stamp that cannot be read at all — a `pie.serving/<n>` this build does
/// not implement, or no serving key at all. The middle part is not politeness:
/// a serving artifact IS the model, produced from a source that may not still
/// be on this machine, so a refusal that quietly rebuilt or removed one would
/// be destroying the only copy over a version number.
///
/// The variant is preserved, which is the distinction `563e5e74d` landed:
/// [`Error::Unsupported`] means *this build cannot read that*, and
/// [`Error::Checkpoint`] means *this file did not deliver*. Only the sentence
/// is added.
fn refuse(path: &Path, why: Error) -> Error {
    let say = |what: String| {
        format!(
            "{what} This file is how this machine holds the model, not a cache of a \
             boot, so nothing here rewrites it and nothing here deletes it — run `{}` \
             to write it again.",
            serving::rebuild(Some(&path.to_string_lossy())),
        )
    };
    match why {
        Error::Unsupported(what) => Error::Unsupported(say(what)),
        Error::Checkpoint(what) => Error::Checkpoint(say(what)),
        other => other,
    }
}

/// A digest, as the lowercase hex a refusal prints.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

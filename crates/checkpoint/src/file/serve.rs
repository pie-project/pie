//! Reading a `pie.serving/1` artifact. [`Artifact`] is a serving artifact,
//! open and mapped, answering what this file is, which planes it carries in
//! which order, where one lies, and — only when asked — whether the bytes
//! match what was written ([`Artifact::verify`]/`verify_prefix`/`verify_all`;
//! [`Artifact::open`] itself hashes nothing). [`read_head`] answers "what is
//! this file for" without mapping; [`read_spans_into`] reads straight into a
//! caller's destinations, verifying each block as it lands. This reader reads
//! `pie.serving/1` or refuses — no version dispatch or shim.

use std::ops::Range;
use std::path::{Path, PathBuf};

use ztensor::Manifest;

use crate::error::Error;
use crate::serving::{self, BLOCKS_KEY, Blocks, Mismatch, PROFILE, Span, Stamp};

/// Concurrent readers on the verify and fill paths. A property of the
/// machine, not the file — unlike `block_bytes`, not stated in the header.
pub const READERS: usize = 8;

/// A serving artifact, open and mapped. Holds the container's own source —
/// and therefore its mapping — for as long as it lives. Nothing is copied: a
/// `&[u8]` handed out here points into the mapping, and the first touch of
/// each page is the NVMe read.
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
    /// Open the serving artifact at `path` and map it. Reads the manifest and
    /// then the stamp; hashes nothing.
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

    /// What this file says it is for — the `pie_*` stamp, decoded.
    #[must_use]
    pub fn stamp(&self) -> &Stamp {
        &self.stamp
    }

    /// The manifest every derivation below reads from.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        self.source
            .provenance()
            .as_root()
            .expect("open refused a source with no manifest")
    }

    /// Is this artifact for this deployment? Field by field, refusing on the
    /// first disagreement.
    ///
    /// # Errors
    ///
    /// The first [`serving::Field`] that differs.
    pub fn check(&self, deployment: &Stamp) -> Result<(), Mismatch> {
        self.stamp.check(deployment)
    }

    /// The serving sequence, recovered from the manifest's offsets rather
    /// than believed from a stored order. Position *i* is hotter than *i+1*.
    #[must_use]
    pub fn sequence(&self) -> Vec<&str> {
        serving::sequence(self.manifest())
    }

    /// Every serving part, in [`Artifact::sequence`] order.
    #[must_use]
    pub fn spans(&self) -> Vec<Span<'_>> {
        serving::spans(self.manifest())
    }

    /// The padded span of each part: the next offset minus this one.
    #[must_use]
    pub fn padded_spans(&self) -> Vec<u64> {
        serving::padded_spans(&self.spans())
    }

    /// The alignment its writer used, read off the offsets, for a caller
    /// sizing a pinned buffer.
    #[must_use]
    pub fn alignment(&self) -> u64 {
        serving::alignment(&self.spans())
    }

    /// The payload run: the first serving blob in the containing file to the
    /// end of the last. Empty when the file serves nothing.
    #[must_use]
    pub fn payload(&self) -> Range<u64> {
        let spans = self.spans();
        let at = serving::payload_at(&spans).unwrap_or(0);
        at..at.saturating_add(serving::payload_total(&spans))
    }

    /// The artifact's key, computed here and stored nowhere: a stored value
    /// would be a claim that can go stale.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] when a part carries no digest.
    pub fn identity(&self) -> Result<String, Error> {
        serving::identity(&self.stamp, self.manifest())
    }

    /// One part's published bytes, borrowed from the mapping. The blob's own
    /// length and not its padded span: bytes past the published length are
    /// zero padding no reader should be handed.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] for a name this file does not carry;
    /// [`Error::Unsupported`] when the bytes have no zero-copy view — an
    /// encoded part (forbidden in a serving artifact) or a store this build
    /// could not map.
    pub fn part(&self, object: &str, part: &str) -> Result<&[u8], Error> {
        self.source
            .tensor(object)
            .and_then(|tensor| tensor.part(part))
            .and_then(|part| part.map())
            .map_err(Error::from)
    }

    /// A plane's padded extent, for a reader that needs the span a layout
    /// tiles with rather than the bytes the tensor publishes. The padding
    /// after a plane is real, readable and zero — every byte between blobs
    /// is required to be `0x00`. Keyed by name, not offset: a `(offset, len)`
    /// door is one typo from returning a neighbour's bytes.
    ///
    /// # Errors
    ///
    /// [`Error::Checkpoint`] for an object or part this file does not carry,
    /// and for a `len` reaching past what the file holds after the blob:
    /// asking for more padding than was written is refused rather than
    /// answered with the next object.
    pub fn span(&self, object: &str, part: &str, len: u64) -> Result<&[u8], Error> {
        let bytes = self.part(object, part)?;
        let published = bytes.len() as u64;
        if len <= published {
            let upto = usize::try_from(len).unwrap_or(usize::MAX);
            return Ok(&bytes[..upto]);
        }
        // Past the published bytes: must land within this blob's padding
        // room, the distance to whatever the writer placed next.
        let spans = self.spans();
        let padded = serving::padded_spans(&spans);
        let at = spans
            .iter()
            .position(|span| span.object == object && span.part == part)
            .ok_or_else(|| {
                Error::Checkpoint(format!(
                    "the serving artifact {} carries no {object:?}/{part:?} span",
                    self.path.display(),
                ))
            })?;
        let room = padded[at];
        if len > room {
            return Err(Error::Checkpoint(format!(
                "{object:?}/{part:?} publishes {published} bytes and the writer left it                  {room} before the next blob; a reader asking for {len} is asking for                  padding this file does not have",
            )));
        }
        // SAFETY: the blob is mapped and `len <= room` (the distance to the
        // next blob), so bytes between `published` and `len` are inside the
        // same mapping and are zero padding.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr(), usize::try_from(len).unwrap_or(0)) })
    }

    /// One part's block digest table, borrowed from the attribute that holds
    /// it, with its length checked against the part it describes.
    /// [`BLOCKS_KEY`] maps object name to part name to a byte string of
    /// `ceil(decoded_size / block_bytes)` digests. Tables live at file
    /// level, not on the objects, since an object's attributes are frozen at
    /// declaration and a writer needs to stream before it can hash.
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

    /// Hash the blocks of these objects and compare them to their tables. A
    /// boot checks only the planes it is about to serve, [`READERS`] blocks
    /// at a time; the refusal names the object, part, block and byte range.
    /// Also warms the page cache.
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

    /// Verify a prefix of the sequence and nothing after it. `objects` past
    /// the end of the sequence verifies the whole sequence rather than
    /// refusing.
    ///
    /// # Errors
    ///
    /// As [`Artifact::verify`].
    pub fn verify_prefix(&self, objects: usize) -> Result<(), Error> {
        let sequence = self.sequence();
        let prefix: Vec<&str> = sequence.into_iter().take(objects).collect();
        self.verify(&prefix)
    }

    /// Hash every block of every serving part, and check each part's own
    /// `digest` too — the one thing the block tables don't stand for. Reads
    /// the whole payload twice (once per block table, once per part digest).
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
    /// One item per block, not per part: parts are as uneven as the model's
    /// planes, and a lane per part would idle behind the largest bank.
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
    /// The block's place in its own part, not in the table.
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

/// Read a serving artifact's manifest and stamp without mapping it.
///
/// # Errors
///
/// As [`Artifact::open`], minus anything only a mapping can produce. Three
/// outcomes, kept apart: `Ok(None)` is an ordinary checkpoint (no
/// `pie.serving/<n>` key); `Ok(Some)` is a serving artifact whose stamp this
/// build reads; `Err` is a file that claims to be one and is not readable as
/// one. A path that is not a zTensor container at all is `Ok(None)`, not an
/// error.
///
/// # Why this exists beside [`read_head`]
///
/// [`read_head`]'s `Result<(Stamp, Manifest)>` cannot tell "no serving key"
/// from "serving key, broken member" — a boot written against it would treat
/// a rotted stamp as an unstamped checkpoint and serve it.
pub fn stamp_of(path: &Path) -> Result<Option<Stamp>, Error> {
    // A non-zTensor path claims nothing; only a file that says it IS servable
    // and is not readable as such is an error here.
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

/// One part, and where its bytes are to be read — [`read_spans_into`]'s unit
/// of work. A whole part, never a piece of one, since block digests are
/// part-local.
pub struct Fill<'a> {
    /// The serving object.
    pub object: &'a str,
    /// The part within it.
    pub part: &'a str,
    /// Where the bytes go. Exactly the part's published length; padding is
    /// not read.
    pub into: *mut u8,
}

/// One block on its way into a caller's buffer: which file it comes from,
/// where in that file, which part-local bytes it covers, what it should hash
/// to, and where it goes.
struct Landing<'a> {
    /// Index into the open files: one descriptor per shard, not per block.
    file: usize,
    /// The absolute offset in that file, which is the blob's plus the block's.
    at: u64,
    /// The block's range within its part, which is what a refusal prints.
    span: Range<u64>,
    /// The digest the artifact states for it.
    stated: &'a [u8],
    into: Carried,
}

/// A destination, as something a scope thread may carry. A raw pointer is
/// not `Send`; [`read_spans_into`]'s safety clause (disjoint destinations)
/// is what makes moving these sound.
struct Carried(*mut u8);

// SAFETY: one address per block, moved once, into the thread that is the
// sole writer of those bytes for as long as the scope is open.
unsafe impl Send for Carried {}
// SAFETY: the work list is sliced disjointly, one slice per lane, so no two
// threads reach the same `Carried`, and the caller's clause forbids overlap.
unsafe impl Sync for Carried {}

/// Read these parts straight into their destinations, verifying them as they
/// arrive. [`READERS`] threads over the work, each reading its own blocks
/// with positioned reads and closing that block's digest from the bytes as
/// they land — no staging buffer, so this verifies the exact bytes the
/// caller will use, in a single pass over the disk.
///
/// Every byte of every fill is written before this returns `Ok`. On `Err`
/// the destinations hold an indeterminate mixture and the caller owes them a
/// zeroing before anything reads them.
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
    // One work item per block, as in verify. Names are kept beside the work
    // rather than inside it: `Fill` holds a raw pointer, so a `&Fill` is
    // neither `Send` nor `Sync`; what crosses into a lane is `Carried`.
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
    // Filesystem failure first: a read that didn't happen isn't a corruption.
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

/// A refusal about the file itself: what is wrong, that nothing here rewrote
/// or deleted it, and the command that would fix it. The variant is
/// preserved: [`Error::Unsupported`] means this build cannot read that;
/// [`Error::Checkpoint`] means this file did not deliver.
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

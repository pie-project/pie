//! **The warm-boot weight artifact cache** — alto design §7's T2 tier, ported
//! from `origin/dev`'s `driver/cuda/src/model/weight_artifact_cache.hpp`.
//!
//! # What it is
//!
//! The materialized device weight table is a DETERMINISTIC FUNCTION of three
//! things: which checkpoint, which load contract compiled against it, and
//! which shell laid the result out. Nothing else varies. So the second boot
//! of the same deployment recomputes, byte for byte, what the first one
//! already produced — every dequant, every cast, every repack, every
//! per-tensor copy — and then uploads the identical bytes.
//!
//! This module makes that second boot a file read. After a cold load the
//! device store is snapshotted beside a key; on a warm boot with a matching
//! key the blob goes straight to the device and the whole host-side transform
//! pipeline ([`checkpoint::executor::Execution`]) never runs.
//!
//! # Why the key is a hash of the recipe and not a name
//!
//! dev's rule, kept: **a false miss costs one re-materialization; a false hit
//! puts silently wrong weights on the device.** So the key errs toward
//! missing. It mixes the whole [`LoadPlan`](checkpoint::plan::LoadPlan)
//! through `serde_json` — the plan IS the recipe, so hashing it whole is the
//! only formulation that cannot go stale when a plan grows a field — plus the
//! checkpoint's own identity, plus the device layout this shell chose, plus a
//! format version. A key or format mismatch is a MISS, not an error: the
//! shell recomputes and rewrites.
//!
//! # Why the checksum is always verified
//!
//! Also dev's, verbatim in spirit: *"a silently-corrupt weight artifact
//! produces garbage tokens with no error, which is not a trade any operator
//! should be offered for a few seconds of load time."* There is no
//! `verify = false`. A blob whose digest does not match is counted as
//! [`Stat::Corrupt`], the half-filled store is thrown away, and the full load
//! runs — loudly, through a named counter, never as a silent retry.
//!
//! # Why the write can decline
//!
//! An artifact is the size of the weights — tens to hundreds of gigabytes —
//! and a cache that fills the disk it lives on has cost the operator more
//! than it saved. dev refuses a write when the filesystem's available space
//! is under the blob plus a margin, names both figures, and carries on with
//! the load. So does this. **A declined write never breaks a load.** Neither
//! does a failed one: every error in here is counted and swallowed, because
//! the cache is an optimization and the cold path is always correct.
//!
//! # How the bytes cross
//!
//! Through [`crate::staged_h2d`] — dev's four-lane pinned double-buffered H2D,
//! whose measured argument is that one staging lane already outruns NVMe by
//! 1.6×, so GDS buys nothing under ~7 GB/s per reader. The artifact is mmap'd
//! and the lanes stream it, so the read and the copy overlap instead of
//! alternating; the digest runs beside them over the same mapping, which is
//! what keeps "always verified" from becoming the new floor. The blocking
//! [`Buffer::write`] loop that came first is still here behind
//! [`restore_through_the_pump`], because the gate that justifies the pump has
//! to measure it against something.
//!
//! # And what the file has become
//!
//! **A SERVING-TIME SOURCE, NOT ONLY A BOOT ACCELERATOR** (alto streaming §0).
//! The stored weights need no load-time conversion, and that format already
//! existed: this one. So the artifact now carries a **plane-group index** — the
//! device store's own `(offset, bytes, reserved)` per plane, transcribed, not
//! computed — ahead of a page-aligned blob, and [`Artifact`] opens, maps and
//! resolves it without copying a byte. That index is what the T2 pointer class
//! of streaming §2 will point through; the HMM arm itself is a later wave.
//!
//! The version bump that carried it is the one place the two doors disagree.
//! On the boot path a stale artifact stays dev's MISS — the shell recomputes
//! and overwrites, loudly — because the cost of being wrong is one
//! re-materialization. On the serving door it is a [`Refused::StaleFormat`]
//! with both numbers in it, because a caller that opened a file to serve out
//! of has no recipe to fall back to.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::device::alloc::Buffer;
use crate::error::Result;

/// **The read side of the promotion** (alto streaming §0 and build-order item
/// 2): the artifact opened, mapped, and resolved plane-group by plane-group
/// without a copy. Separate file because it is a different verb on the same
/// bytes — this module WRITES the format and restores from it; that one serves
/// out of it.
mod mapped;

pub use mapped::{Artifact, Refused};

/// The artifact format's own version.
///
/// **A DIFFERENT ONE IS A MISS ON THE RESTORE PATH, AND A REFUSAL ON THE
/// SERVING PATH.** dev's `SCHEMA_VERSION` rule — the shell recomputes and
/// overwrites — still governs [`restore`], because there a stale artifact
/// costs one re-materialization and nothing else. It does NOT govern
/// [`Artifact::open`]: a caller that asked to SERVE from this file cannot be
/// handed silence, so an old version comes back as
/// [`Refused::StaleFormat`] by name. Bump this whenever the bytes on disk
/// mean something different — a header field, a layout rule, an alignment.
///
/// **2** was the promotion of alto streaming §0: the same blob, now preceded
/// by a plane-group index, so the file is a mmap-able serving-time source and
/// not only a boot accelerator.
///
/// **3 is the striped digest** (W-4's follow-on). The header still carries one
/// `digest`, but it now folds [`STRIPES`] independent FNV chains instead of
/// being one chain over every byte, and the chains themselves are written
/// beside it. The bytes of the blob did not move; what a reader must COMPUTE
/// from them did, which is exactly the kind of change this number exists to
/// announce — a v2 file's digest would disagree with a v3 reader's arithmetic
/// and be indicted as corruption if the version did not speak first.
const FORMAT: u32 = 3;

/// What every artifact starts with, so a file that is not one is a miss
/// rather than a parse.
///
/// **UNCHANGED ACROSS THE VERSION BUMP, ON PURPOSE.** The magic says "this is
/// a weight artifact"; [`FORMAT`] says which one. Keeping them separate is
/// what lets a version-1 file be RECOGNIZED and refused by name rather than
/// mistaken for somebody else's file.
const MAGIC: [u8; 8] = *b"PIEWCAC1";

/// Header bytes. The layout, which is the file's whole contract:
///
/// ```text
///    0..8   magic                       "PIEWCAC1"
///    8..12  format          u32          FORMAT
///   12..16  groups          u32          how many index entries follow
///   16..24  key             u64          Identity::key
///   24..32  total           u64          blob bytes = the device store's size
///   32..40  digest          u64          FNV-1a over the blob, and ONLY it
///   40..48  index_digest    u64          FNV-1a over the index entries
///   48..56  index_at        u64          file offset of the first entry
///   56..64  blob_at         u64          file offset of the first blob byte
///   64..96  stripes         u64 x 4      one FNV-1a per stripe of the blob
/// ```
///
/// **`digest` covers the blob alone**, deliberately: it is the same number
/// [`digest_of`] computes from what is resident on the device, so a gate can
/// compare a file and a card without knowing this header exists. The index
/// gets its own digest rather than being folded into that one.
///
/// Since v3 that number is [`fold`] of the four `stripes` rather than a single
/// chain over the blob. **The stripes are written as well as folded**, and not
/// for the restore's benefit — the restore checks the fold and would be
/// satisfied without them. They are here because a serving-time reader
/// (streaming §2) faults in one plane, not the file, and a per-stripe digest is
/// the coarsest thing it can check without hashing bytes it never mapped.
const HEADER: usize = 64 + STRIPES * 8;

/// One index entry on disk: `id`, `plane`, `offset`, `bytes`, `reserved`.
const ENTRY: usize = 4 + 4 + 8 + 8 + 8;

/// What the blob's first byte is aligned to.
///
/// A page, because the whole point of the promotion is that this file is
/// mmap'd and a plane's bytes are resolved to an offset a device may fault on
/// (streaming §2's third pointer class). A blob that starts mid-page would put
/// every plane's alignment at the mercy of how many params the model has.
const BLOB_ALIGN: u64 = 4096;

/// How much room the write wants BEYOND the blob before it will use the disk.
///
/// dev's 256 MiB, unchanged. It is a statute, not a constitution (alto §1):
/// the number is here so that a full disk is a declined write rather than a
/// deployment that cannot boot, and it moves with a measurement.
const MARGIN: u64 = 256 << 20;

/// How much of the blob crosses at a time, in either direction.
///
/// dev's `kChunkBytes`: the host never holds more than one chunk of a store
/// that is tens of gigabytes.
const CHUNK: usize = 64 << 20;

// ── the counters ────────────────────────────────────────────────────────────

/// What happened to the cache on one load.
///
/// **PROCESS-GLOBAL AND NOT A FIELD**, for the reason the retired
/// `record::FOLD_OBSERVED` was: a gate at the runtime level holds the engine
/// behind a `Box<dyn Engine>`
/// on a lane thread and cannot ask the instance anything. The per-load answer
/// a CALLER wants rides home on
/// [`LoadFacts::weights_from_cache`](engine::load::LoadFacts);
/// this is what a test counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Observed {
    /// Loads that read the whole table off the artifact and skipped every
    /// host-side transform.
    pub restored: u64,
    /// Loads that ran the full pipeline because no artifact matched the key.
    pub missed: u64,
    /// Artifacts written after a full load.
    pub stored: u64,
    /// Artifacts whose digest did not match their bytes. **Counted, named,
    /// and followed by the full load** — never by a silent retry.
    pub corrupt: u64,
    /// Writes declined for want of disk space, or that failed outright.
    pub declined: u64,
}

/// Which counter a load moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stat {
    Restored = 0,
    Missed = 1,
    Stored = 2,
    Corrupt = 3,
    Declined = 4,
}

static COUNTS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn bump(stat: Stat) {
    COUNTS[stat as usize].fetch_add(1, Ordering::Relaxed);
}

/// Everything this process has seen the weight artifact cache do.
#[must_use]
pub fn observed() -> Observed {
    let at = |stat: Stat| COUNTS[stat as usize].load(Ordering::Relaxed);
    Observed {
        restored: at(Stat::Restored),
        missed: at(Stat::Missed),
        stored: at(Stat::Stored),
        corrupt: at(Stat::Corrupt),
        declined: at(Stat::Declined),
    }
}

// ── the key ─────────────────────────────────────────────────────────────────

/// FNV-1a, 64-bit, with **length-prefixed** mixing.
///
/// The length prefix is dev's and it is load-bearing: without it `"ab"` then
/// `"c"` collides with `"a"` then `"bc"`, and the key is exactly a sequence of
/// variable-length fields.
#[derive(Debug, Clone, Copy)]
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    /// Mix raw bytes, with no length prefix — for a blob whose length is
    /// mixed separately (the store's, which the header states).
    pub fn raw(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// Mix a field: its length, then its bytes.
    pub fn field(&mut self, bytes: &[u8]) {
        self.raw(&(bytes.len() as u64).to_le_bytes());
        self.raw(bytes);
    }

    /// Mix a number as a field.
    pub fn number(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    /// The digest.
    #[must_use]
    pub fn finish(self) -> u64 {
        self.0
    }
}

// ── the striped digest ──────────────────────────────────────────────────────

/// **How many independent FNV chains cover the blob.**
///
/// The digest used to be ONE chain over every byte, and W-4's measurement
/// found that this — not the disk, and not the PCIe bus — was what a warm boot
/// waited on. Each byte's multiply depends on the previous byte's, so the
/// chain runs on exactly one core at ~0.93 GB/s, while the same filesystem
/// feeds four parallel readers at 3.7 GB/s. The pump had already hidden the
/// read and the upload behind that hash and could go no faster: 1.6 GiB
/// restored in 2.08s against a 1.84s digest floor.
///
/// **Four, and the arithmetic picks the number — not the pump's lane count.**
/// With `N` stripes a pumped restore costs `max(T, H/N)`: `T` for the transfer
/// the lanes overlap, `H/N` for the hash they cannot. It therefore stops
/// improving the moment `H/N` drops under `T`. Measured on the L40S box over
/// 1.6 GiB: `H = 1.84s`, `T ≈ 0.6s`, so the knee is at `N = H/T ≈ 3` and four
/// is the first power of two past it. **Eight would not make a restore
/// faster** — the pump is transfer-bound by then — it would only make the
/// blocking arm cheaper and the measured advantage smaller.
///
/// The 8-thread figure that box turns in for *reading* (5.41 GB/s) is not the
/// number that sets this one. The hash is CPU-bound, so what governs is cores
/// spent per byte, not queue depth against the disk.
///
/// It equals [`crate::staged_h2d::LANES`] by arithmetic rather than by
/// coincidence — one hash thread per pump lane is a tidy result, not a
/// requirement — and nothing here reads that constant, so the two may diverge
/// the day either measurement moves.
pub const STRIPES: usize = 4;

/// **Where each stripe starts and how long it is.**
///
/// Contiguous and page-aligned: stripe `i` covers `[i*span, (i+1)*span)` for a
/// `span` rounded DOWN to [`BLOB_ALIGN`], and the last stripe carries whatever
/// the rounding left over. The alignment is not decoration — the parallel hash
/// walks a mapping, and a boundary mid-page would put two threads on one page
/// and make them fault against each other.
///
/// A blob too small to give every stripe a page puts all of it in stripe 0 and
/// leaves the rest empty. Empty stripes hash to the FNV basis and fold
/// deterministically, so a tiny artifact needs no special case anywhere else.
#[must_use]
fn stripe_spans(total: u64) -> [(u64, u64); STRIPES] {
    let span = total / STRIPES as u64 / BLOB_ALIGN * BLOB_ALIGN;
    let mut out = [(0u64, 0u64); STRIPES];
    if span == 0 {
        out[0] = (0, total);
        return out;
    }
    for (which, slot) in out.iter_mut().enumerate() {
        let at = which as u64 * span;
        *slot = if which == STRIPES - 1 {
            (at, total - at)
        } else {
            (at, span)
        };
    }
    out
}

/// **The one number the stripes fold to** — [`Head::digest`], and what
/// [`digest_of`] answers.
///
/// So that a gate can still compare a file against a card with a single
/// `assert_eq!` and never learn that the digest acquired a shape.
#[must_use]
pub fn fold(stripes: &[u64; STRIPES]) -> u64 {
    let mut hash = Fnv::default();
    for digest in stripes {
        hash.number(*digest);
    }
    hash.finish()
}

/// **Hash every stripe of `blob` at once**, one thread per stripe.
///
/// The whole point of the striping: [`STRIPES`] independent chains over
/// disjoint spans of one mapping, so the digest costs a core-second per stripe
/// rather than a wall-second per blob. The spans are disjoint by construction,
/// which is what lets this be safe code — each thread gets its own `&[u8]`.
#[must_use]
pub fn stripe_digests(blob: &[u8]) -> [u64; STRIPES] {
    let mut out = [0u64; STRIPES];
    std::thread::scope(|scope| {
        let mut hashing = Vec::with_capacity(STRIPES);
        for (at, len) in stripe_spans(blob.len() as u64) {
            let part = blob
                .get(usize::try_from(at).unwrap_or(usize::MAX)..)
                .and_then(|rest| rest.get(..usize::try_from(len).unwrap_or(0)))
                .unwrap_or(&[]);
            hashing.push(scope.spawn(move || {
                let mut hash = Fnv::default();
                hash.raw(part);
                hash.finish()
            }));
        }
        for (slot, thread) in out.iter_mut().zip(hashing) {
            *slot = thread.join().unwrap_or(0);
        }
    });
    out
}

/// **The streaming twin of [`stripe_digests`]**, for the paths that never hold
/// the whole blob.
///
/// The write side reads the device a chunk at a time and the device-side
/// digest reads it back the same way; neither can map what it is hashing. So
/// bytes arrive here IN ORDER, in chunks of any size, and each is routed to
/// whichever stripe spans hold it — a chunk that straddles a boundary is split.
///
/// It answers exactly what the parallel version answers over the same bytes,
/// and that agreement is not left to inspection:
/// [`seed`] writes an artifact through this and [`Artifact::verify`] checks it
/// through the other, so `the_artifact_says_where_every_plane_lives` goes red
/// the day the two arithmetics part company.
struct Striper {
    spans: [(u64, u64); STRIPES],
    hash: [Fnv; STRIPES],
    /// How many blob bytes have already been fed — the absolute offset the
    /// next slice starts at, which is what makes routing possible without
    /// holding anything.
    at: u64,
}

impl Striper {
    fn new(total: u64) -> Striper {
        Striper {
            spans: stripe_spans(total),
            hash: [Fnv::default(); STRIPES],
            at: 0,
        }
    }

    /// Mix the next `bytes` of the blob into whichever stripes they land in.
    fn feed(&mut self, bytes: &[u8]) {
        let from = self.at;
        let to = from.saturating_add(bytes.len() as u64);
        for (span, hash) in self.spans.iter().zip(self.hash.iter_mut()) {
            let (at, len) = *span;
            let start = at.max(from);
            let end = at.saturating_add(len).min(to);
            if start < end {
                let lo = usize::try_from(start - from).unwrap_or(0);
                let hi = usize::try_from(end - from).unwrap_or(0);
                hash.raw(&bytes[lo..hi]);
            }
        }
        self.at = to;
    }

    fn finish(self) -> [u64; STRIPES] {
        self.hash.map(Fnv::finish)
    }
}

/// **Everything the materialized table is a function of.**
///
/// Three groups, and the module header argues each: which checkpoint, which
/// recipe, which layout. A field that belongs here and is missing is a false
/// HIT — the dangerous direction — so this struct is deliberately over-mixed:
/// the whole plan, not a summary of it.
#[derive(Clone, Copy)]
pub struct Identity<'a> {
    /// The checkpoint, as a path. Its bytes are mixed, and so is what
    /// [`stat_identity`] can learn about the files behind it.
    pub checkpoint: &'a Path,
    /// The plan's name, as the model text declared it.
    pub trace_name: &'a str,
    /// The load plan — the RECIPE — serialized whole.
    pub plan_json: &'a [u8],
    /// The device layout this shell chose: `(offset, bytes, reserved)` per
    /// param, in param order. What the blob physically IS.
    pub layout: &'a [(u64, u64, u64)],
    /// Total device bytes.
    pub total: u64,
}

impl Identity<'_> {
    /// This identity as one number.
    #[must_use]
    pub fn key(&self) -> u64 {
        let mut hash = Fnv::default();
        hash.number(u64::from(FORMAT));
        hash.field(self.checkpoint.as_os_str().as_encoded_bytes());
        hash.field(self.trace_name.as_bytes());
        hash.field(self.plan_json);
        hash.number(self.total);
        hash.number(self.layout.len() as u64);
        for &(offset, bytes, reserved) in self.layout {
            hash.number(offset);
            hash.number(bytes);
            hash.number(reserved);
        }
        hash.number(stat_identity(self.checkpoint));
        hash.finish()
    }
}

/// What can be learned about a checkpoint without reading it: the size and
/// modification time of every file under it, in sorted order.
///
/// dev's `mix_snapshot_stat`, with dev's two rules kept. **Stat follows
/// symlinks** — an HF snapshot is a link farm into `blobs/`, and `lstat`
/// would hash the length of a link string instead of the weights. **A stat
/// that fails mixes a sentinel** rather than being skipped, so an
/// unreadable file is a different key rather than the same one.
fn stat_identity(path: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;

    let mut hash = Fnv::default();
    let mut entries: Vec<PathBuf> = Vec::new();
    if path.is_dir() {
        match fs::read_dir(path) {
            Ok(dir) => {
                for entry in dir.flatten() {
                    entries.push(entry.path());
                }
            }
            Err(_) => return 0xdead_beef_dead_beef,
        }
        entries.sort();
    } else {
        entries.push(path.to_path_buf());
    }
    for entry in entries {
        hash.field(entry.as_os_str().as_encoded_bytes());
        // `metadata` follows symlinks; `symlink_metadata` would not.
        match fs::metadata(&entry) {
            Ok(meta) => {
                hash.number(meta.size());
                hash.number(meta.mtime() as u64);
                hash.number(meta.mtime_nsec() as u64);
            }
            Err(_) => hash.number(0xdead_beef_dead_beef),
        }
    }
    hash.finish()
}

// ── the artifact ────────────────────────────────────────────────────────────

/// **What every artifact says about itself before its bytes.**
///
/// The whole of [`HEADER`], decoded. It is a public type because the mmap side
/// ([`Artifact`]) answers with it and a gate reads it: the file's identity, its
/// two digests, and where its two payloads start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Head {
    /// [`FORMAT`] as this file states it. A different one is the version
    /// refusal.
    pub format: u32,
    /// How many [`Group`] entries the index holds. Zero is legal and means an
    /// artifact written by a caller that had no layout to declare.
    pub groups: u32,
    /// [`Identity::key`] — which deployment's weights these are.
    pub key: u64,
    /// The blob's length, which is the device store's length.
    pub total: u64,
    /// FNV-1a over the blob and only the blob — the same number
    /// [`digest_of`] computes from the device.
    pub digest: u64,
    /// FNV-1a over the index entries as they are written.
    pub index_digest: u64,
    /// Where the first index entry starts.
    pub index_at: u64,
    /// Where the blob's first byte starts.
    pub blob_at: u64,
    /// **One FNV-1a per stripe of the blob**, in stripe order, which
    /// [`fold`] folds into `digest`.
    ///
    /// Written since format 3. The restore checks the fold and would be
    /// satisfied without ever reading these; they are on disk for the serving
    /// reader that maps one plane and wants to check something narrower than
    /// the whole file.
    pub stripes: [u64; STRIPES],
}

impl Head {
    /// The bytes that go on the front of the file.
    #[must_use]
    fn encode(&self) -> [u8; HEADER] {
        let mut out = [0u8; HEADER];
        out[..8].copy_from_slice(&MAGIC);
        out[8..12].copy_from_slice(&self.format.to_le_bytes());
        out[12..16].copy_from_slice(&self.groups.to_le_bytes());
        out[16..24].copy_from_slice(&self.key.to_le_bytes());
        out[24..32].copy_from_slice(&self.total.to_le_bytes());
        out[32..40].copy_from_slice(&self.digest.to_le_bytes());
        out[40..48].copy_from_slice(&self.index_digest.to_le_bytes());
        out[48..56].copy_from_slice(&self.index_at.to_le_bytes());
        out[56..64].copy_from_slice(&self.blob_at.to_le_bytes());
        for (which, digest) in self.stripes.iter().enumerate() {
            let at = Head::STRIPES_AT as usize + which * 8;
            out[at..at + 8].copy_from_slice(&digest.to_le_bytes());
        }
        out
    }

    /// Read one back. `None` for anything that is not an artifact at all —
    /// too short, or the wrong magic — which is the only outcome this function
    /// judges. **The version is decoded, not checked**, because the two
    /// callers disagree about what a stale one means.
    #[must_use]
    fn decode(bytes: &[u8]) -> Option<Head> {
        if bytes.len() < HEADER || bytes[..8] != MAGIC {
            return None;
        }
        // Every slice below is inside the length just checked, so the
        // fallbacks are unreachable rather than lenient.
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or([0; 4]));
        let long = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or([0; 8]));
        Some(Head {
            format: word(8),
            groups: word(12),
            key: long(16),
            total: long(24),
            digest: long(32),
            index_digest: long(40),
            index_at: long(48),
            blob_at: long(56),
            stripes: core::array::from_fn(|which| long(Head::STRIPES_AT as usize + which * 8)),
        })
    }

    /// Where `digest` sits. The header is written twice — once with the digest
    /// fields blank, once whole over the top when the blob has been read — so
    /// these two offsets are part of the format rather than literals somebody
    /// has to keep in step by hand.
    pub const DIGEST_AT: u64 = 32;

    /// Where the stripe digests start.
    pub const STRIPES_AT: u64 = 64;
}

/// **One plane group, and where the store keeps it.**
///
/// The index entry of alto streaming §3's item 2. The three numbers are the
/// DEVICE STORE'S OWN — `weights.rs`' `Place`, verbatim — which is the whole
/// reason the promotion is a formalization rather than a format design: the
/// artifact is a snapshot of the store, so the store's offsets already address
/// it. Nothing here is computed; it is transcribed.
///
/// `id` and `plane` name the group rather than a byte range so that a
/// split-plane bank (one weight id, two device planes —
/// [`WeightRow::Planes`](crate::run::WeightRow)) is two entries under one id,
/// and a dense row is one entry with `plane: 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Group {
    /// The param's ordinal in the plan's own order — the number `places`
    /// indexes by, which is what makes this an index and not a name table.
    pub id: u32,
    /// Which plane of that group: `0` for a dense row, `0`/`1` for a
    /// split-plane bank.
    pub plane: u32,
    /// Where the plane starts in the store, and therefore in the blob.
    pub offset: u64,
    /// The plane's own bytes — what the checkpoint publishes.
    pub bytes: u64,
    /// What the store gives it, aligned up. Equal to `bytes` for every plane
    /// of a fully-resident load, and larger for a streamed routed bank whose
    /// slab seats only its resident experts.
    pub reserved: u64,
}

impl Group {
    /// The entry as it goes on disk.
    #[must_use]
    fn encode(&self) -> [u8; ENTRY] {
        let mut out = [0u8; ENTRY];
        out[..4].copy_from_slice(&self.id.to_le_bytes());
        out[4..8].copy_from_slice(&self.plane.to_le_bytes());
        out[8..16].copy_from_slice(&self.offset.to_le_bytes());
        out[16..24].copy_from_slice(&self.bytes.to_le_bytes());
        out[24..32].copy_from_slice(&self.reserved.to_le_bytes());
        out
    }

    /// Read one back.
    #[must_use]
    fn decode(bytes: &[u8]) -> Option<Group> {
        if bytes.len() < ENTRY {
            return None;
        }
        // As `Head::decode`: the length is checked above.
        Some(Group {
            id: u32::from_le_bytes(bytes[..4].try_into().unwrap_or([0; 4])),
            plane: u32::from_le_bytes(bytes[4..8].try_into().unwrap_or([0; 4])),
            offset: u64::from_le_bytes(bytes[8..16].try_into().unwrap_or([0; 8])),
            bytes: u64::from_le_bytes(bytes[16..24].try_into().unwrap_or([0; 8])),
            reserved: u64::from_le_bytes(bytes[24..32].try_into().unwrap_or([0; 8])),
        })
    }
}

/// The index's own digest, over the entries exactly as they are written.
fn index_digest(groups: &[Group]) -> u64 {
    let mut hash = Fnv::default();
    hash.number(groups.len() as u64);
    for group in groups {
        hash.raw(&group.encode());
    }
    hash.finish()
}

/// Where the blob starts, given how many groups precede it.
fn blob_at(groups: usize) -> u64 {
    ((HEADER + groups * ENTRY) as u64).next_multiple_of(BLOB_ALIGN)
}

/// **Every index entry is inside the blob it indexes.**
///
/// The one structural claim a reader can check without the bytes, and the
/// reason it is checked: an entry that points past the end would hand a device
/// address to a caller that resolved it, and `Buffer`'s door is not on that
/// path. A `Some` is the reason it failed.
fn index_fault(groups: &[Group], total: u64) -> Option<String> {
    for group in groups {
        // An overflow is a fault of its own and not a `None`: a group whose
        // end does not fit a `u64` is exactly the entry a bounds check exists
        // to catch.
        let Some(end) = group.offset.checked_add(group.reserved) else {
            return Some(format!(
                "plane group {}/{} states a span that overflows: {} + {}",
                group.id, group.plane, group.offset, group.reserved
            ));
        };
        if end > total {
            return Some(format!(
                "plane group {}/{} states bytes {}..{} of a {total}-byte table",
                group.id, group.plane, group.offset, end
            ));
        }
    }
    None
}

// ── the restore path's one switch ───────────────────────────────────────────

/// Whether a restore goes through [`crate::staged_h2d`] or through the
/// blocking read-then-`Buffer::write` loop that shipped before it.
///
/// **A CODE-LEVEL SWITCH AND NOT A DEPLOYMENT ONE** (article 9: shells read no
/// environment). It exists for exactly one reason — the W-4 checkpoint gate
/// measures the pump against what it replaced, and a measurement needs both
/// arms in one process — so the setter is a `probe` hook, dropped by a serving
/// binary the same way [`ProgramSession::skew_prediction`](crate::ProgramSession)
/// is. Serving always takes the pump.
static PUMPED: AtomicBool = AtomicBool::new(true);

/// **Send the next restores through the blocking path instead of the pump.**
///
/// The W-4 gate's other arm. Gate-only: this is behind `probe`, which a
/// serving binary drops with `default-features = false`.
#[cfg(feature = "probe")]
pub fn restore_through_the_pump(pumped: bool) {
    PUMPED.store(pumped, Ordering::Relaxed);
}

/// Which arm the next restore will take.
#[cfg(feature = "probe")]
#[must_use]
pub fn restore_is_pumped() -> bool {
    PUMPED.load(Ordering::Relaxed)
}

// ── reading and writing one ─────────────────────────────────────────────────

/// Where this key's artifact lives under `dir`.
///
/// **PUBLIC BECAUSE THE SERVING SIDE HAS TO NAME THE SAME FILE** (alto
/// streaming §2, wave W-1). `weights::spill_source` opens this path to map a
/// load's T2 tier out of it, and a second spelling of the filename in another
/// module is exactly the drift that makes a cache miss look like a corruption.
/// One home for one string.
#[must_use]
pub fn artifact_path(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.weights"))
}

/// **Try to fill `store` from the artifact for `key`.**
///
/// Answers `true` when the whole table came off the disk and the caller may
/// skip the transform pipeline entirely. Every other outcome — no directory,
/// no file, a stale format, a key that does not match, a truncated blob, a
/// digest that does not match — answers `false`, and the caller runs the full
/// load. The store is left ZEROED on a `false`, whatever was written into it,
/// so the full load starts from the same state a cold boot would.
///
/// # How the bytes cross
///
/// Through [`crate::staged_h2d`]: the file is mmap'd and four pinned lanes
/// stream it to the device, so the read and the H2D copy overlap instead of
/// alternating. The digest runs beside them on a fifth thread over the same
/// mapping, which is what keeps "always verified" from becoming the new floor.
/// [`restore_through_the_pump`] selects the old blocking arm for the gate that
/// measures the difference.
///
/// # Errors
///
/// Only a device failure while zeroing a store this function had already
/// written into. Disk failures are not errors here: they are misses.
pub fn restore(dir: Option<&Path>, key: u64, store: &mut Buffer) -> Result<bool> {
    let Some(dir) = dir else {
        // The deployment told the shell nothing. The feature is off: no
        // reads, no writes, and no counter — a load that was never offered a
        // cache did not miss one.
        return Ok(false);
    };
    let path = artifact_path(dir, key);
    match read_into(&path, key, store) {
        Ok(true) => {
            bump(Stat::Restored);
            Ok(true)
        }
        Ok(false) => {
            bump(Stat::Missed);
            Ok(false)
        }
        Err(why) => {
            // **THE ONE LOUD OUTCOME.** A digest that does not match its
            // bytes is not a miss: something wrote or rotted an artifact that
            // claims to be this deployment's weights. It is counted under its
            // own name, said out loud, the half-filled store is thrown away,
            // and the full load runs.
            bump(Stat::Corrupt);
            eprintln!(
                "engine-cuda: weight artifact {path:?} is corrupt ({why}); \
                 discarding it and running the full load"
            );
            let _ = fs::remove_file(&path);
            store.zero_span(0, store.bytes())?;
            Ok(false)
        }
    }
}

/// `Ok(true)` restored, `Ok(false)` no artifact for this key, `Err` corrupt.
fn read_into(path: &Path, key: u64, store: &mut Buffer) -> std::result::Result<bool, String> {
    let (head, groups) = match mapped::read_head(path) {
        Ok(read) => read,
        // Not a file, not an artifact, or a version this build does not read.
        // All three are MISSES on this path — dev's rule, and the cheap
        // direction: the shell recomputes and overwrites. The version is the
        // one that is said out loud, because an operator who upgraded a build
        // over a populated cache should learn why the first boot was slow.
        Err(Refused::StaleFormat { states, reads }) => {
            eprintln!(
                "engine-cuda: weight artifact {path:?} states format {states} and this build \
                 reads {reads}; recomputing the table and overwriting it"
            );
            return Ok(false);
        }
        Err(Refused::Unreadable { .. } | Refused::NotAnArtifact) => return Ok(false),
        Err(other) => return Err(other.to_string()),
    };
    // A key from another deployment is a MISS, not a fault.
    if head.key != key {
        return Ok(false);
    }
    if head.total != store.bytes() as u64 {
        // The layout is part of the key, so this cannot happen without the
        // key having been reused for a different table. Treat it as
        // corruption rather than as a miss: a file that lies about its own
        // size is the case the digest exists to catch.
        return Err(format!(
            "states {} bytes for a table of {}",
            head.total,
            store.bytes()
        ));
    }
    if let Some(why) = index_fault(&groups, head.total) {
        return Err(why);
    }

    // **THE CHECKSUM IS NOT A MODE, AND STRIPING DID NOT MAKE IT ONE.** Both
    // arms below hash EVERY byte they move, across all `STRIPES` chains, and
    // compare the lot to the header before answering `true`. The switch changes
    // how the bytes cross — four pinned lanes or one blocking loop — and
    // nothing about what is verified.
    //
    // Both arms also pay the SAME digest: `stripe_digests` on `STRIPES`
    // threads. That is deliberate and it is what makes the W-4 gate's ratio
    // mean something — an arm that hashed serially while the other hashed in
    // parallel would be losing a race it was entered into carrying weight.
    let stripes = if PUMPED.load(Ordering::Relaxed) {
        pumped_into(path, &head, store)?
    } else {
        blocking_into(path, &head, store)?
    };
    // Named per stripe, because "the digest is wrong" about a 1.6 GiB file says
    // less than it could when the file is already divided into four answers.
    for (which, (found, stated)) in stripes.iter().zip(head.stripes.iter()).enumerate() {
        if found != stated {
            let (at, len) = stripe_spans(head.total)[which];
            return Err(format!(
                "stripe {which} of {STRIPES} (bytes {at}..{}) hashes to {found:016x} \
                 where the header states {stated:016x}",
                at + len,
            ));
        }
    }
    if fold(&stripes) != head.digest {
        return Err(format!(
            "the stripes agree one by one but fold to {:016x} where the header states \
             {:016x} over {} bytes",
            fold(&stripes),
            head.digest,
            head.total,
        ));
    }
    Ok(true)
}

/// **The pump's arm**: mmap, four staging lanes, and the striped digest
/// beside them.
///
/// Answers the stripe digests of what it moved, so the caller does the
/// comparing — one place, both arms.
///
/// **What this arm pays**: `max(transfer, digest)`. The lanes and the hash
/// threads run over the same mapping at the same time, so the read, the
/// staging memcpy, the DMA and all `STRIPES` chains are one wall-clock cost
/// rather than four.
fn pumped_into(
    path: &Path,
    head: &Head,
    store: &mut Buffer,
) -> std::result::Result<[u64; STRIPES], String> {
    let artifact = Artifact::open(path).map_err(|why| why.to_string())?;
    let blob = artifact.blob();
    if blob.len() as u64 != head.total {
        return Err(format!(
            "maps {} blob bytes where its header states {}",
            blob.len(),
            head.total
        ));
    }
    // **THE ONE BOUNDS CHECK ON THIS PATH.** The pump takes device addresses,
    // so the span meets the store's length HERE rather than at
    // `Buffer::write`'s door, which this arm does not go through.
    let base = store.at(0).map_err(|why| why.to_string())?;
    let _ = store
        .at(head.total)
        .map_err(|why| format!("the artifact does not fit the store: {why}"))?;

    let mut lanes = crate::staged_h2d::Lanes::standard().map_err(|why| why.to_string())?;
    let transfer = [crate::staged_h2d::Transfer {
        dst: base,
        src: blob.as_ptr(),
        len: head.total,
    }];
    // **THE DIGEST RUNS BESIDE THE LANES, AND NOW IT ALSO RUNS BESIDE
    // ITSELF.** Taking it off the copy's critical path was the first half of
    // the answer and it was not enough: one FNV chain over 1.6 GiB is 1.84s of
    // one core, which simply became the new floor once the transfer went
    // parallel. `stripe_digests` spawns `STRIPES` chains over disjoint spans of
    // this same mapping, so the hash finishes in a quarter of that and the
    // transfer is the thing in front again.
    let (moved, stripes) = std::thread::scope(|scope| {
        let hashing = scope.spawn(|| stripe_digests(blob));
        let moved = lanes.pump(&transfer);
        let stripes = hashing.join().unwrap_or([0; STRIPES]);
        (moved, stripes)
    });
    moved.map_err(|why| format!("staged upload failed: {why}"))?;
    Ok(stripes)
}

/// **The arm the pump replaced**, kept whole for the measurement that
/// justifies the pump: one chunk of host memory, read then uploaded, in
/// series. Selected by [`restore_through_the_pump`] and by nothing else.
///
/// **What this arm pays**: `read + upload + digest`, the three of them end to
/// end. That is the ONE property that makes it the right control — it overlaps
/// nothing — and it is why the digest below runs after the transfer rather
/// than beside it.
///
/// **But the digest itself is the same parallel one the pump uses**, and that
/// is a deliberate correction to an earlier shape of this function, which
/// hashed inline on one core. Leaving it serial would have handed the pump a
/// four-fold head start on a cost that has nothing to do with how bytes cross,
/// and the gate's ratio would have been measuring the striping instead of the
/// overlap. Both arms are charged `H/STRIPES`; only one of them gets to hide it.
///
/// The hash reads the file a second time, through a mapping, rather than
/// hashing each chunk as it passes. A single in-order reader CANNOT feed
/// `STRIPES` parallel chains over contiguous spans — it produces byte 0 before
/// byte `total/4`, so three of the four chains would sit idle waiting for
/// bytes that have not been read yet. The second pass costs a walk of the page
/// cache this loop just filled, which is the cheapest way to buy the arm the
/// same digest the pump gets.
fn blocking_into(
    path: &Path,
    head: &Head,
    store: &mut Buffer,
) -> std::result::Result<[u64; STRIPES], String> {
    use std::io::Seek;

    // ── the transfer, strictly in series
    let mut file = fs::File::open(path).map_err(|why| format!("{why}"))?;
    file.seek(std::io::SeekFrom::Start(head.blob_at))
        .map_err(|why| format!("{why}"))?;
    let mut at = 0u64;
    let mut chunk = vec![0u8; CHUNK.min(store.bytes().max(1))];
    while at < head.total {
        let want = usize::try_from(head.total - at)
            .unwrap_or(usize::MAX)
            .min(chunk.len());
        let slice = &mut chunk[..want];
        file.read_exact(slice)
            .map_err(|why| format!("truncated at byte {at}: {why}"))?;
        // Uploaded as it is read: the host never holds more than one chunk.
        // The digest is checked AFTER, which is why a mismatch has to throw
        // the store away rather than leave it half-written (dev's
        // `loaded_model.cpp` hit the same ordering and answered it the same
        // way).
        store
            .write(at, slice)
            .map_err(|why| format!("upload failed at byte {at}: {why}"))?;
        at += want as u64;
    }
    drop(chunk);
    drop(file);

    // ── then the digest, striped and parallel — the same cost the pump pays,
    //    and the only thing this arm does not overlap it with.
    let artifact = Artifact::open(path).map_err(|why| why.to_string())?;
    let blob = artifact.blob();
    if blob.len() as u64 != head.total {
        return Err(format!(
            "maps {} blob bytes where its header states {}",
            blob.len(),
            head.total
        ));
    }
    Ok(stripe_digests(blob))
}

// ── writing one ─────────────────────────────────────────────────────────────

/// **Snapshot `store` beside `key`, if there is room.**
///
/// Best-effort in every direction: no directory means the feature is off, a
/// filesystem with less than the blob plus [`MARGIN`] free means a declined
/// write with both figures named, and any failure at all means a declined
/// write. None of them fails the load, because the load already succeeded —
/// the store this is reading from is the answer.
///
/// **WITH NO INDEX.** The layout lives in the loader, which forms the key from
/// it and has never had to hand it any further; this is the call it makes
/// today. The format carries the index either way, so the day the loader
/// passes its `places` across, that is one argument at one call site and no
/// change here — and until then a v2 artifact with zero groups restores
/// exactly as it always did.
pub fn store(dir: Option<&Path>, key: u64, store: &Buffer) {
    store_indexed(dir, key, &[], store);
}

/// **Snapshot `store` beside `key` WITH its plane-group index.**
///
/// [`store`] plus the thing that makes the file a serving-time source rather
/// than only a boot accelerator (alto streaming §0): the groups are written
/// ahead of the blob, so a later open can resolve a plane to an offset without
/// the load plan that produced it.
///
/// The groups are transcribed, not computed — they are the device store's own
/// `(offset, bytes, reserved)`, which is exactly what the key already mixes,
/// so an index that disagreed with the blob would be an index for a different
/// key.
pub fn store_indexed(dir: Option<&Path>, key: u64, groups: &[Group], store: &Buffer) {
    let Some(dir) = dir else {
        return;
    };
    if let Err(why) = write_out(dir, key, groups, store) {
        bump(Stat::Declined);
        eprintln!("engine-cuda: declined to cache this load's weights: {why}");
        return;
    }
    bump(Stat::Stored);
}

fn write_out(
    dir: &Path,
    key: u64,
    groups: &[Group],
    store: &Buffer,
) -> std::result::Result<(), String> {
    fs::create_dir_all(dir).map_err(|why| format!("{dir:?}: {why}"))?;
    let total = store.bytes() as u64;
    let blob_at = blob_at(groups.len());
    let need = total.saturating_add(blob_at).saturating_add(MARGIN);
    let free = available_bytes(dir)?;
    if free < need {
        return Err(format!(
            "{dir:?} has {:.1} GiB free and this artifact wants {:.1} GiB \
             (a {} GiB blob plus a {} GiB margin); point `[model] \
             weight_cache_dir` at a disk with more space",
            free as f64 / (1u64 << 30) as f64,
            need as f64 / (1u64 << 30) as f64,
            total >> 30,
            MARGIN >> 30,
        ));
    }

    // Published atomically: the bytes land beside the target and are moved
    // into place at the end, so a boot that dies mid-write leaves a partial
    // file nobody will ever name rather than a corrupt one under the key.
    let final_path = artifact_path(dir, key);
    let temp_path = dir.join(format!("{key:016x}.weights.{}.part", std::process::id()));
    let outcome = (|| -> std::result::Result<(), String> {
        let mut file = fs::File::create(&temp_path).map_err(|why| format!("{temp_path:?}: {why}"))?;
        let head = Head {
            format: FORMAT,
            groups: u32::try_from(groups.len())
                .map_err(|_| format!("{} plane groups is more than a header states", groups.len()))?,
            key,
            total,
            // Written last, over these placeholders, once the blob has been
            // read; a rewind is cheaper than holding the store.
            digest: 0,
            stripes: [0; STRIPES],
            index_digest: index_digest(groups),
            index_at: HEADER as u64,
            blob_at,
        };
        file.write_all(&head.encode()).map_err(|why| format!("{why}"))?;
        for group in groups {
            file.write_all(&group.encode()).map_err(|why| format!("{why}"))?;
        }
        // The gap that puts the blob on a page boundary. Written rather than
        // seeked over, so the file has no hole a later mmap would read as
        // zeros it did not intend.
        let pad = blob_at - (HEADER + groups.len() * ENTRY) as u64;
        file.write_all(&vec![0u8; usize::try_from(pad).unwrap_or(0)])
            .map_err(|why| format!("{why}"))?;

        // Streamed, because the host never holds more than one chunk of a
        // store that may be tens of gigabytes — so the stripes are accumulated
        // by routing rather than by mapping.
        let mut striper = Striper::new(total);
        let mut at = 0u64;
        let mut chunk = vec![0u8; CHUNK.min(store.bytes().max(1))];
        while at < total {
            let want = usize::try_from(total - at).unwrap_or(usize::MAX).min(chunk.len());
            let slice = &mut chunk[..want];
            store
                .read(at, slice)
                .map_err(|why| format!("reading the store at {at}: {why}"))?;
            striper.feed(slice);
            file.write_all(slice).map_err(|why| format!("{why}"))?;
            at += want as u64;
        }
        // The whole header goes back over the top, rather than two seeks to
        // two fields: the digests are the only things that changed, and
        // re-encoding what was already computed cannot get them out of step.
        use std::io::Seek;
        let stripes = striper.finish();
        let head = Head {
            digest: fold(&stripes),
            stripes,
            ..head
        };
        file.seek(std::io::SeekFrom::Start(0))
            .map_err(|why| format!("{why}"))?;
        file.write_all(&head.encode()).map_err(|why| format!("{why}"))?;
        file.sync_all().map_err(|why| format!("{why}"))?;
        Ok(())
    })();
    if let Err(why) = outcome {
        let _ = fs::remove_file(&temp_path);
        return Err(why);
    }
    fs::rename(&temp_path, &final_path).map_err(|why| {
        let _ = fs::remove_file(&temp_path);
        format!("publishing {final_path:?}: {why}")
    })
}

/// **Write an artifact from host bytes** — the seed a gate synthesizes.
///
/// The device-free twin of [`store_indexed`], for the tests that have to
/// assert what the format promises on a machine with no GPU in it: the same
/// header, the same index, the same digest discipline, a blob that came from a
/// `Vec` instead of from a card.
///
/// Gate-only (`probe`), like every other hook in this crate that exists so a
/// test can state something a serving path never would.
///
/// # Errors
///
/// The filesystem's own words. Nothing here declines for space: a synthetic
/// blob is kilobytes.
#[cfg(feature = "probe")]
pub fn seed(path: &Path, key: u64, groups: &[Group], blob: &[u8]) -> std::result::Result<(), String> {
    use std::io::Seek;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|why| format!("{parent:?}: {why}"))?;
    }
    let blob_at = blob_at(groups.len());
    let mut file = fs::File::create(path).map_err(|why| format!("{path:?}: {why}"))?;
    let head = Head {
        format: FORMAT,
        groups: u32::try_from(groups.len()).map_err(|_| "too many plane groups".to_string())?,
        key,
        total: blob.len() as u64,
        digest: 0,
        stripes: [0; STRIPES],
        index_digest: index_digest(groups),
        index_at: HEADER as u64,
        blob_at,
    };
    file.write_all(&head.encode()).map_err(|why| format!("{why}"))?;
    for group in groups {
        file.write_all(&group.encode()).map_err(|why| format!("{why}"))?;
    }
    let pad = blob_at - (HEADER + groups.len() * ENTRY) as u64;
    file.write_all(&vec![0u8; usize::try_from(pad).unwrap_or(0)])
        .map_err(|why| format!("{why}"))?;
    file.write_all(blob).map_err(|why| format!("{why}"))?;
    // Through the STREAMING striper even though the whole blob is in hand, so
    // that the seed exercises the arithmetic the write path uses and
    // `Artifact::verify` — which reads through the parallel one — is checking
    // that the two agree.
    let mut striper = Striper::new(blob.len() as u64);
    striper.feed(blob);
    let stripes = striper.finish();
    let head = Head {
        digest: fold(&stripes),
        stripes,
        ..head
    };
    file.seek(std::io::SeekFrom::Start(0))
        .map_err(|why| format!("{why}"))?;
    file.write_all(&head.encode()).map_err(|why| format!("{why}"))?;
    file.sync_all().map_err(|why| format!("{why}"))?;
    Ok(())
}

/// **Overwrite an artifact's stated format** — the only way a gate can hold a
/// file from a build that no longer exists.
///
/// Gate-only, and it writes nothing but the four version bytes: everything
/// else about the file stays true, which is what makes the refusal it provokes
/// a refusal about the VERSION and not about a corruption.
///
/// # Errors
///
/// The filesystem's own words.
#[cfg(feature = "probe")]
pub fn restate_format(path: &Path, format: u32) -> std::result::Result<(), String> {
    use std::io::Seek;

    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|why| format!("{path:?}: {why}"))?;
    file.seek(std::io::SeekFrom::Start(8))
        .map_err(|why| format!("{why}"))?;
    file.write_all(&format.to_le_bytes())
        .map_err(|why| format!("{why}"))?;
    file.sync_all().map_err(|why| format!("{why}"))
}

/// Bytes available to an unprivileged writer under `dir`.
///
/// `statvfs`, because the standard library has no answer and dev's refusal is
/// stated in these terms (`std::filesystem::space(dir).available`).
fn available_bytes(dir: &Path) -> std::result::Result<u64, String> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(dir.as_os_str().as_bytes())
        .map_err(|_| format!("{dir:?} is not a path this platform can state"))?;
    // SAFETY: `path` is a NUL-terminated C string that outlives the call, and
    // `stat` is a plain out-parameter the call fully initializes on success.
    let mut stat = unsafe { std::mem::zeroed::<libc::statvfs>() };
    let rc = unsafe { libc::statvfs(path.as_ptr(), &raw mut stat) };
    if rc != 0 {
        return Err(format!("{dir:?}: cannot read the filesystem's free space"));
    }
    Ok((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

/// The digest of what is actually resident on the device.
///
/// What a gate compares between two loads: not "the same size" and not "the
/// same source", but the same bytes.
///
/// **[`fold`] of the stripes**, which is the same number [`Head::digest`]
/// carries — so a gate still compares a card against a file with one
/// `assert_eq!` and never has to learn that the digest is four chains. Serial,
/// because the bytes come back over PCIe a chunk at a time and this is a gate's
/// helper, not a load path.
///
/// # Errors
///
/// A device failure reading the store back.
pub fn digest_of(store: &Buffer) -> Result<u64> {
    let mut striper = Striper::new(store.bytes() as u64);
    let total = store.bytes() as u64;
    let mut chunk = vec![0u8; CHUNK.min(store.bytes().max(1))];
    let mut at = 0u64;
    while at < total {
        let want = usize::try_from(total - at).unwrap_or(usize::MAX).min(chunk.len());
        let slice = &mut chunk[..want];
        store.read(at, slice)?;
        striper.feed(slice);
        at += want as u64;
    }
    Ok(fold(&striper.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The length prefix is what makes a key a sequence of fields rather than
    /// a concatenation of them.
    #[test]
    fn two_field_splits_of_the_same_bytes_do_not_collide() {
        let mut one = Fnv::default();
        one.field(b"ab");
        one.field(b"c");
        let mut other = Fnv::default();
        other.field(b"a");
        other.field(b"bc");
        assert_ne!(one.finish(), other.finish());
    }

    /// **THE KEY IS A FUNCTION OF THE RECIPE, AND OF EVERY PART OF IT.**
    ///
    /// A false miss costs one re-materialization; a false hit puts silently
    /// wrong weights on the device. So each field is checked to MOVE the key,
    /// which is the only direction that matters.
    #[test]
    fn every_part_of_the_recipe_moves_the_key() {
        let layout = [(0u64, 100u64, 256u64), (256, 40, 256)];
        let base = Identity {
            checkpoint: Path::new("/models/qwen"),
            trace_name: "qwen35-d0.8b",
            plan_json: b"{\"steps\":[]}",
            layout: &layout,
            total: 512,
        };
        let key = base.key();

        let other_layout = [(0u64, 100u64, 256u64), (256, 44, 256)];
        for moved in [
            Identity {
                checkpoint: Path::new("/models/other"),
                ..base
            },
            Identity {
                trace_name: "qwen35-d1.7b",
                ..base
            },
            Identity {
                plan_json: b"{\"steps\":[1]}",
                ..base
            },
            Identity {
                layout: &other_layout,
                ..base
            },
            Identity { total: 768, ..base },
        ] {
            assert_ne!(key, moved.key(), "a changed recipe kept its key");
        }

        // And the same recipe is the same key, which is the whole point.
        assert_eq!(key, base.key());
    }

    /// **THE HEADER ROUND-TRIPS**, field for field, through the bytes it
    /// writes. Every field is given a different value, because a header whose
    /// fields are all zero would round-trip through an encoder that dropped
    /// half of them.
    #[test]
    fn a_header_round_trips_through_the_bytes_it_writes() {
        let head = Head {
            format: FORMAT,
            groups: 3,
            key: 0x0123_4567_89ab_cdef,
            total: 1 << 30,
            digest: 0xfedc_ba98_7654_3210,
            index_digest: 0x1111_2222_3333_4444,
            index_at: HEADER as u64,
            blob_at: 4096,
            stripes: [
                0xaaaa_0000_0000_0001,
                0xbbbb_0000_0000_0002,
                0xcccc_0000_0000_0003,
                0xdddd_0000_0000_0004,
            ],
        };
        let bytes = head.encode();
        assert_eq!(bytes.len(), HEADER, "the header is exactly its own length");
        assert_eq!(bytes[..8], MAGIC, "and it starts with the magic");
        assert_eq!(Head::decode(&bytes), Some(head), "every field came back");

        // The header is written twice — once with the digest fields blank,
        // once whole over the top — so both offsets are part of the format and
        // not literals somebody has to keep in step by hand.
        let mut patched = bytes;
        patched[Head::DIGEST_AT as usize..Head::DIGEST_AT as usize + 8]
            .copy_from_slice(&0u64.to_le_bytes());
        let read = Head::decode(&patched).expect("still an artifact");
        assert_eq!(read.digest, 0, "DIGEST_AT names the digest field");
        assert_eq!(
            read.stripes, head.stripes,
            "and nothing else — blanking the fold left the stripes alone"
        );

        let mut patched = bytes;
        let at = Head::STRIPES_AT as usize;
        patched[at..at + STRIPES * 8].fill(0);
        let read = Head::decode(&patched).expect("still an artifact");
        assert_eq!(read.stripes, [0; STRIPES], "STRIPES_AT names the stripe vector");
        assert_eq!(
            read.digest, head.digest,
            "and nothing else — blanking the stripes left the fold alone"
        );
        assert_eq!(
            HEADER,
            Head::STRIPES_AT as usize + STRIPES * 8,
            "the stripe vector is the last thing in the header"
        );

        // Anything that is not one answers `None` rather than a guess.
        assert_eq!(Head::decode(&bytes[..HEADER - 1]), None, "too short");
        let mut foreign = bytes;
        foreign[0] = b'X';
        assert_eq!(Head::decode(&foreign), None, "not the magic");
    }

    /// **AN INDEX ENTRY ROUND-TRIPS**, and it is exactly [`ENTRY`] bytes —
    /// which is what lets a reader `chunks_exact` its way through the index
    /// without a per-entry length.
    #[test]
    fn a_plane_group_round_trips_through_a_fixed_width_entry() {
        let group = Group {
            id: 41,
            plane: 1,
            offset: 1 << 20,
            bytes: 300,
            reserved: 512,
        };
        let bytes = group.encode();
        assert_eq!(bytes.len(), ENTRY);
        assert_eq!(Group::decode(&bytes), Some(group));
        assert_eq!(Group::decode(&bytes[..ENTRY - 1]), None, "a short entry");
    }

    /// The blob starts on a page, whatever the index costs — the alignment
    /// the mmap side depends on.
    #[test]
    fn the_blob_starts_on_a_page_however_many_groups_precede_it() {
        for groups in [0usize, 1, 7, 128, 4096] {
            let at = blob_at(groups);
            assert_eq!(at % BLOB_ALIGN, 0, "{groups} groups did not land on a page");
            assert!(
                at >= (HEADER + groups * ENTRY) as u64,
                "{groups} groups do not fit before their blob"
            );
        }
    }

    /// **AN ENTRY THAT POINTS OUTSIDE THE TABLE IS A FAULT WITH A SENTENCE.**
    ///
    /// The one structural claim a reader can check without the bytes, and it
    /// is checked because resolving such an entry would hand a caller a span
    /// `Buffer`'s door never saw.
    #[test]
    fn an_index_entry_past_the_end_of_the_table_is_named() {
        let inside = [Group {
            id: 0,
            plane: 0,
            offset: 0,
            bytes: 100,
            reserved: 256,
        }];
        assert_eq!(index_fault(&inside, 256), None, "a group that fits");
        assert!(
            index_fault(&inside, 255).is_some_and(|why| why.contains("plane group 0/0")),
            "a group one byte too long names itself"
        );

        let overflowing = [Group {
            id: 4,
            plane: 1,
            offset: u64::MAX,
            bytes: 8,
            reserved: 8,
        }];
        assert!(
            index_fault(&overflowing, u64::MAX).is_some_and(|why| why.contains("overflows")),
            "a span that does not fit a u64 is a fault and not a wrap"
        );
    }

    /// The index's digest is a function of every entry, in order — so a
    /// reordered index is a different index.
    #[test]
    fn the_index_digest_moves_with_every_entry_and_with_their_order() {
        let one = Group {
            id: 0,
            plane: 0,
            offset: 0,
            bytes: 8,
            reserved: 256,
        };
        let other = Group {
            id: 1,
            plane: 0,
            offset: 256,
            bytes: 8,
            reserved: 256,
        };
        let straight = index_digest(&[one, other]);
        assert_ne!(straight, index_digest(&[other, one]), "order counts");
        assert_ne!(straight, index_digest(&[one]), "length counts");
        assert_ne!(straight, index_digest(&[]), "an empty index is its own digest");
        assert_eq!(straight, index_digest(&[one, other]), "and it is a function");
    }
}

//! **The serving artifact** — §M's budget-polymorphic tier file, format side.
//!
//! # What it is
//!
//! [`super`]'s artifact snapshots one thing: the device store. That is the
//! whole answer for a load that fits, and it is no answer at all for a load
//! that does not — a streamed deployment materializes three images, not one.
//! This file is the same promise for a streamed load, and since §M it is not a
//! cache of a boot but **the model as this machine serves it**: `pie model
//! import` writes it, every boot reads it, and nothing here ever deletes it.
//!
//! # One sequence, not three sections
//!
//! Formats 1 and 2 held THREE images — device, host, mapped — because that is
//! what a boot materializes. It also made the file a function of the BUDGETS:
//! change `device_weight_budget` and every one of the three sections is a
//! different length, so the key mixed the rungs and a different budget was a
//! different file.
//!
//! Format 3 stops doing that, on one measured fact (`experts::Plan::mapped_layout`
//! states it, §M.3 repeats it): **a plane's bytes are identical on all three
//! rungs.** [`weights::ALIGN`](crate::weights::ALIGN) is what makes
//! "reinterpret an mxfp4 code plane as 32-bit words" a valid thing to do out
//! of the store, out of page-locked memory and out of a mapping alike, so the
//! three rungs differ only in the WALK — which plane, which destination, which
//! offset — and never in the bytes. So the file holds each plane's image
//! ONCE:
//!
//! ```text
//!   index      one entry per image, IN PRIORITY ORDER, offsets consecutive
//!   blocks     one FNV digest per TIER_BLOCK of each entry, entry-local
//!   payload    the images, back to back, each at its ALIGN-padded span
//! ```
//!
//! and a boot cuts it with the budget it has:
//!
//! ```text
//!   [0 .. c1)   pumped into the device store
//!   [c1 .. c2)  the pinned tier — verified first, filled in the background
//!   [c2 ..  )   Held::Mapped: served where it lies, never copied
//! ```
//!
//! **The cut is a load-time decision and never a baked one.** One artifact
//! serves any budget pair on this setup, which is the whole of §M.3.
//!
//! # The order, which is the file's own contract
//!
//! ```text
//!   1. the CUT SEQUENCE, hottest first —
//!        every dense plane a budget may spill, in prefetch-schedule order,
//!        then every packed routed group in ascending param order,
//!        each group's planes in ascending param order
//!   2. the DENSE ROUTED BANKS, whole, in ascending param order
//! ```
//!
//! It is [`experts::Ranking`](crate::experts::Ranking) handed to a writer, and
//! it is BUDGET-FREE: `Plan::of` derives the same ranking and then cuts it,
//! so the file states the ranking and the boot states the cut. The banks sit
//! after the sequence because they are the one region with TWO readers at
//! EVERY budget — a device slab takes their prefix and the pinned tier takes
//! them whole — so they belong on neither side of a cut, and keeping them out
//! of the sequence is what leaves `c1` and `c2` both inside one contiguous
//! run.
//!
//! **The registered planes are not in this file at all.** An adapter bank's
//! store bytes are whatever `Buffer::zeroed` left, because the artifact is
//! written from inside `Weights::resident` and `register_adapter` is a method
//! on a `Weights` that has not been returned yet. Writing a region of zeros so
//! a restore could write it back is a hundred megabytes of disk to reproduce
//! `Buffer::zeroed`; [`FLAG_ADAPTERS_ZEROED`] states the omission instead, and
//! a file without the flag is refused.
//!
//! # Why the digests are per-entry blocks
//!
//! A boot verifies what it is about to SERVE, and what it serves is a set of
//! entries the current budget picked — so the digest granularity has to be
//! fine enough that any subset of entries, and therefore any contiguous
//! prefix, can be checked without hashing the rest. One digest per entry would
//! do that and no more: a dense routed bank's image is tens of gigabytes and
//! an FNV chain is serial over its own span, so one chain per entry would put
//! a minute of one core in front of the first token.
//!
//! So each entry is divided into [`TIER_BLOCK`] pieces and each piece gets its
//! own chain. Every unit of verification is then at most 64 MiB of serial
//! hashing, the blocks of any entry are contiguous, and the blocks of a prefix
//! of entries are a prefix of the table. How MANY chains run at once is not in
//! the format — see [`TIER_READERS`] — which is the other thing this buys: the
//! stripe count used to be a header field, so §L's measurement could not be
//! re-taken without a format bump.
//!
//! # Where this file stops
//!
//! At the format. Nothing here reads a device, opens a checkpoint, or knows
//! what a `Plan` is: bytes in, file out, file in, bytes out. That is what lets
//! it lift into `checkpoint::source` whole when the metal twin adopts it
//! (§K.7's condition, still in force).

use std::fs;
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::mapped::Map;
use super::{
    CHUNK, ENTRY, Fnv, Group, MARGIN, Refused, available_bytes, index_digest, stat_identity,
};

/// The serving artifact's own version, and **it is not [`super::FORMAT`]**.
///
/// The two files are different formats under different magics with different
/// lifetimes: the resident artifact's version moves when the device store's
/// bytes change meaning, this one's moves when the sequence, a flag or an
/// alignment changes. Sharing a counter would have made every bump of either
/// invalidate both, which is a cost with no argument behind it.
///
/// **1** was the first: three sections, one index, per-section digests.
///
/// **2** widened the stripe vectors to eight chains a section.
///
/// **3** is §M's, and it is the one bump that changes what the file IS: one
/// budget-free sequence of plane images with per-entry block digests, no
/// sections, no rungs and no budgets in the key. A format-1 or format-2 file
/// is refused BY VERSION before any of its fields is believed, said out loud,
/// and **left where it is** — see [`refuse`].
pub const TIER_FORMAT: u32 = 3;

/// **What every serving artifact starts with.**
///
/// Distinct from [`super::MAGIC`](super) so that neither reader can be handed
/// the other's file and get as far as parsing it — the two live in the same
/// directory under the same key space.
const MAGIC: [u8; 8] = *b"PIEWTIER";

/// Header bytes. The layout, which is the file's whole contract:
///
/// ```text
///     0..8   magic          "PIEWTIER"
///     8..12  format         u32   TIER_FORMAT
///    12..16  flags          u32   FLAG_*
///    16..24  key            u64   Identity::key
///    24..32  index_at       u64   where the first index entry starts
///    32..40  entries        u64   how many images this file holds
///    40..48  index_digest   u64   FNV-1a over the entries, in order
///    48..56  blocks_at      u64   where the block digest table starts
///    56..64  block_bytes    u64   TIER_BLOCK as this file states it
///    64..72  blocks         u64   how many digests the table holds
///    72..80  digest         u64   the fold over all of them
///    80..88  payload_at     u64   where the images begin (TIER_ALIGN)
///    88..96  payload_total  u64   how many bytes they occupy
/// ```
const TIER_HEADER: usize = 96;

/// **What the payload's first byte is aligned to.**
///
/// Two mebibytes, and not [`super::BLOB_ALIGN`](super)'s page. The payload is
/// pumped in whole entries and read straight into page-locked memory the
/// allocator hands out on huge-page boundaries; starting it mid-huge-page
/// would put the alignment of a hundred-gigabyte read at the mercy of how many
/// params the model has. The alignment INSIDE the payload is the store's own
/// ([`weights::ALIGN`](crate::weights::ALIGN), 256), because that is the
/// alignment every rung's arithmetic already rests on and the whole reason one
/// image can serve all three.
pub const TIER_ALIGN: u64 = 2 << 20;

/// **The unit of verification** — how much of one entry a single FNV chain
/// covers.
///
/// Sixty-four mebibytes, and it is [`CHUNK`] on purpose: the writer streams an
/// entry in `CHUNK` pieces and closes exactly one digest per piece, so the
/// bytes that are hashed and the bytes that are written cannot come from two
/// different arithmetics. A reader that wants one entry checked hashes that
/// entry's blocks and nothing else.
///
/// The table costs eight bytes per block — thirteen kibibytes for a hundred
/// gigabytes of payload — which is small enough that the granularity was
/// chosen for the READER and not for the header.
///
/// **Blocks are ENTRY-LOCAL**: entry `e`'s block `b` covers
/// `[e.offset + b*TIER_BLOCK, e.offset + min((b+1)*TIER_BLOCK, e.reserved))`.
/// Since the entries tile the payload consecutively, the blocks of a prefix of
/// entries are a prefix of the table, which is the property the cut needs.
pub const TIER_BLOCK: u64 = CHUNK as u64;

/// **How many block digests are computed at once**, and it is deliberately NOT
/// in the format.
///
/// §L.3's measurement, which is a fact about this box and not about the file:
/// the warm boot's host verify hides under the device pump, an FNV chain runs
/// at about 0.93 GB/s on one core and this filesystem answers eight concurrent
/// readers at 5.41 GB/s, so eight is where the verify stops being the term in
/// the way and sixteen would only idle on the NVMe.
///
/// Format 2 spent a header field on that number, which meant a re-measurement
/// invalidated every artifact on the disk. Per-entry blocks make the count a
/// scheduling decision instead: whichever way the workers are handed out, they
/// answer the same digests over the same bytes.
pub const TIER_READERS: usize = 8;

/// **The registered planes are not in this file.**
///
/// Stated rather than implied. A registered plane is an adapter bank, and the
/// artifact is written from inside `Weights::resident` — before
/// `register_adapter` can have run — so every one of them holds what
/// `Buffer::zeroed` left. The file omits them and says so; a restore leaves
/// the store's zeros where they are.
///
/// A file WITHOUT the flag was written by something that moved the snapshot
/// out of the constructor, and restoring it would seat whatever an adapter
/// held as though it were a weight. Refused rather than believed.
pub const FLAG_ADAPTERS_ZEROED: u32 = 1 << 0;

// ── the counters ────────────────────────────────────────────────────────────

/// **What the serving artifact has done, process-wide.**
///
/// Its OWN register, on `experts::observed`'s precedent and not by widening
/// [`super::Observed`]: the two files answer different questions and a
/// deployment can hit one and miss the other in the same boot, so folding them
/// into one set of counters would make "restored" mean neither. Nothing reads
/// these back — no path branches on them (design §14).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Observed {
    /// Serving artifacts written after a full streamed load.
    pub stored: u64,
    /// Loads that filled their tiers from one and skipped the cold branch.
    pub restored: u64,
    /// Entries whose bytes did not hash to what the block table states.
    /// **Counted, named per block, and followed by the full load.**
    pub corrupt: u64,
    /// Writes declined for want of disk space, or that failed outright.
    pub declined: u64,
    /// **Writes skipped because this key's file was already on the disk.**
    ///
    /// A second boot at the same seat, which is the case the key exists to
    /// make cheap — and since §M, the case `pie model import` makes ordinary.
    /// Counted separately from `stored` because the two are different
    /// sentences about the same directory.
    pub skipped: u64,
    /// **Loads that served their pinned tier out of the file's own mapping
    /// while a background thread built the page-locked copy** (§L, phase L-1).
    ///
    /// The warm boot's other shape: the entries the cut puts on T1 are
    /// verified where they lie and the kernels read them over HMM, so the two
    /// terms that dominate a warm streamed boot — `cudaHostAlloc` over tens of
    /// gigabytes, and the read that fills it — are moved off the road to the
    /// first token.
    pub deferred: u64,
    /// **Deferred loads whose page-locked image arrived and was installed.**
    ///
    /// `deferred - promoted` is the honest residue: a background fill that
    /// found the file rotted under it (counted in `corrupt`), one the machine
    /// refused, and one whose load ended before the window closed. None of
    /// them is a wrong answer — a seat that never promotes serves the same
    /// bytes at the mapping's speed for its whole life — so this is a
    /// PERFORMANCE counter and the design says so (§14).
    pub promoted: u64,
    /// **How long the last window that closed was open**, in milliseconds.
    ///
    /// A GAUGE AND NOT A TOTAL — the last one, not the sum — because a process
    /// boots one seat far more often than several and an operator asking "how
    /// long did it serve degraded?" wants the number, not an average over one.
    /// Zero until a window closes.
    pub window_ms: u64,
}

/// Which counter moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stat {
    Stored = 0,
    Restored = 1,
    Corrupt = 2,
    Declined = 3,
    Skipped = 4,
    Deferred = 5,
    Promoted = 6,
    /// **Stored, not added to.** See [`Observed::window_ms`].
    WindowMs = 7,
}

static COUNTS: [AtomicU64; 8] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn bump(stat: Stat) {
    COUNTS[stat as usize].fetch_add(1, Ordering::Relaxed);
}

/// Everything this process has seen the serving artifact do.
#[must_use]
pub fn observed() -> Observed {
    let at = |stat: Stat| COUNTS[stat as usize].load(Ordering::Relaxed);
    Observed {
        stored: at(Stat::Stored),
        restored: at(Stat::Restored),
        corrupt: at(Stat::Corrupt),
        declined: at(Stat::Declined),
        skipped: at(Stat::Skipped),
        deferred: at(Stat::Deferred),
        promoted: at(Stat::Promoted),
        window_ms: at(Stat::WindowMs),
    }
}

/// **Count a load that filled its tiers out of this file.**
///
/// The one counter this module has no door of its own for: restoring is the
/// boot's verb, and it says so here when the last entry has crossed. A
/// function rather than a field for the reason [`observed`] is process-global
/// — a gate at the runtime level holds the engine behind a `Box<dyn Engine>`
/// and cannot ask the instance anything.
pub fn count_restored() {
    bump(Stat::Restored);
}

/// **Count a file that lied about itself somewhere no digest was taken.**
///
/// [`Artifact::verify_entries`] and [`read_spans_into`] count the corruptions
/// they FIND, because they are the doors where bytes are checked. This is for
/// the ones a reader catches before a byte is hashed: a header whose entries
/// do not tile their own payload, an index that does not describe the images
/// this boot lays out, a name and a header that disagree. §K.5 gives them one
/// counter, not two, because to the operator asking why the warm boot went
/// cold they are one answer.
pub fn count_corrupt() {
    bump(Stat::Corrupt);
}

/// **Count a load that took the deferred seat** (§L, phase L-1).
///
/// Called once, by the arm that serves T1 out of this file's own mapping
/// instead of out of a page-locked copy of it. It says the WINDOW OPENED and
/// nothing about how it closes — [`count_promoted`] is the other half, and a
/// deployment showing this without that spent its whole life on the mapping.
pub fn count_deferred() {
    bump(Stat::Deferred);
}

/// **Count a deferred seat whose page-locked image was installed**, and state
/// how long that took.
///
/// `window_ms` is STORED rather than added to — see [`Observed::window_ms`] —
/// and the two writes are separate atomics, so a reader between them sees the
/// old window beside the new count. That is a §14 register and not a control
/// input: nothing branches on either number, and the pair is a sentence for an
/// operator rather than a transaction.
pub fn count_promoted(window_ms: u64) {
    bump(Stat::Promoted);
    COUNTS[Stat::WindowMs as usize].store(window_ms, Ordering::Relaxed);
}

/// **THE ONE SENTENCE A REFUSED SERVING ARTIFACT GETS** (§M.4, §M-3) — three
/// facts in a fixed order, so that the three refusals a boot can reach read as
/// one message with one variable in it.
///
/// ```text
///   what is wrong   `why`, in the vocabulary of the door that found it
///   what is NOT     nothing is rewritten here and nothing is deleted here
///     done about it
///   the remedy      `rebuild(source)` — the command that writes this file,
///                   spelled against the checkpoint THIS load names
/// ```
///
/// # It used to delete, and then it used to fall through
///
/// It deleted while the file was a CACHE: a cache's contents are reproducible
/// by the thing that reads them, so a corrupt one is best removed before it is
/// re-read, re-hashed and re-refused on every boot. §M.4 stopped that — **it
/// is the model now**, `pie model import` produced it out of a source that may
/// not still be on this machine, and a serving path that deletes a hundred
/// gigabytes on its own judgement is a serving path that can turn a bad block
/// into a lost deployment.
///
/// What §M-3 stops is the OTHER half of the old sentence. "This boot runs the
/// full load instead" was true while a streamed serving load could still
/// stream, transform and write; it cannot (`weights::Intent`), so a message
/// that promised it would have been a lie printed one line above a refusal.
/// The serving path has exactly one road to these bytes now and this names it.
///
/// # It BUILDS the sentence and does not print it
///
/// Because the two callers want it in different places: a serving load puts it
/// in a [`Fault::Residency`](crate::Fault::Residency) — which the runtime logs
/// once, with the load it refused — and a prepare prints it and goes on to
/// write the file the sentence is about. A function that printed would double
/// the first and a function that returned nothing could not feed it.
///
/// The door that FOUND the disagreement is still the door that COUNTS it, so a
/// caller composes: [`count_corrupt`] where nothing else counted, then this.
#[must_use]
pub fn refuse(path: &Path, source: Option<&Path>, why: &str) -> String {
    format!(
        "engine-cuda: the serving artifact {path:?} {why}. This file is how this machine \
         holds the model, not a cache of a boot, so nothing here rewrites it and nothing \
         here deletes it — run `{}` to write it again from the checkpoint this load \
         names. There is no cold serving path left to fall back to.",
        rebuild(source),
    )
}

/// **The command that writes a serving artifact, spelled for `source`.**
///
/// One string, in one place, because it is a PROMISE: every refusal above
/// prints it and `pie model import --prepare-only` has to be a thing that
/// actually works on the argument it is handed. `source` is the checkpoint the
/// refusing load was pointed at — a `.zt` or a snapshot directory — which is
/// exactly what `import`'s own `--prepare-only` resolves, so the line can be
/// copied rather than adapted.
///
/// **`None` IS FOR THE ONE REFUSAL THAT IS NOT A LOAD'S.** §L's background
/// refill runs on a thread the load armed and outlived; it holds the artifact
/// it is reading and not the checkpoint the deployment was pointed at, and
/// inventing a path for it would be worse than a placeholder. So it gets the
/// same sentence with the argument left as a slot — the operator reading it
/// has the deployment's config open, which is where that path is written down.
#[must_use]
pub fn rebuild(source: Option<&Path>) -> String {
    match source {
        Some(source) => format!("pie model import --prepare-only {}", source.display()),
        None => "pie model import --prepare-only <this deployment's checkpoint>".to_string(),
    }
}

/// **The OTHER serving artifacts in this directory, newest first.**
///
/// The census behind §M-3's loudest refusal. [`path`] puts the key in the
/// FILENAME, so a changed plan or a changed recipe is a changed key is a
/// different file — and before this wave that arrived as a silent cold boot
/// standing beside a hundred gigabytes nothing would ever read again. A
/// refusal that can say *"this one is not here and these four are"* turns that
/// into the one sentence an operator needs.
///
/// `key` is this load's own and is excluded, so a non-empty answer is always
/// news. Names only — the caller renders them — and the ordering is by
/// modification time so that a truncated list shows the ones most likely to be
/// the previous deployment.
///
/// An unreadable directory answers empty: this is a diagnostic, and a refusal
/// that failed to enumerate is still a refusal.
#[must_use]
pub fn others(dir: &Path, key: u64) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| key_in(path).is_some_and(|held| held != key))
        .map(|path| {
            let at = path
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (at, path)
        })
        .collect();
    found.sort_by(|left, right| right.0.cmp(&left.0));
    found.into_iter().map(|(_, path)| path).collect()
}

// ── the key ─────────────────────────────────────────────────────────────────

/// **Everything this file's bytes are a function of — AND NOT ONE BUDGET**
/// (§M.3).
///
/// [`super::Identity`]'s fields verbatim — which checkpoint, which recipe,
/// which uncapped layout, how many bytes — plus the one thing this file has
/// that the store's snapshot does not: **the sequence**, stated as the triples
/// its own index carries.
///
/// ```text
///   checkpoint, trace_name, plan_json, total, layout
///                  the deployment: a pure function of the trace and recipe
///   images         (param, bytes, reserved) per image, IN FILE ORDER
/// ```
///
/// Format 2's identity mixed `device_layout`, `host_layout` and every param's
/// RUNG, which made it budget-dependent by construction: a changed budget was
/// a different key, a different file, and a hundred gigabytes rewritten to say
/// the same thing about the same weights. None of the three is here. What
/// replaces them is `images`, which is what the file physically holds — the
/// ranking and every span in it — and is a function of the trace and the
/// recipe alone.
///
/// **A field that belongs here and is missing is a false HIT**, the dangerous
/// direction, so `images` is mixed WHOLE rather than summarized: a file whose
/// order or whose spans differ from what this boot would write is a file this
/// boot cannot cut.
#[derive(Clone, Copy)]
pub struct Identity<'a> {
    /// The checkpoint, as a path. Its bytes are mixed, and so is what
    /// [`super::stat_identity`](super) can learn about the files behind it.
    pub checkpoint: &'a Path,
    /// The plan's name, as the model text declared it.
    pub trace_name: &'a str,
    /// The load plan — the RECIPE — serialized whole.
    pub plan_json: &'a [u8],
    /// What a FULLY RESIDENT load of this deployment lays out, in bytes.
    pub total: u64,
    /// That load's layout: `(offset, bytes, reserved)` per param, in param
    /// order. `resident_key`'s own field, for its own reason — it is the
    /// deployment's identity and it is budget-free.
    pub layout: &'a [(u64, u64, u64)],
    /// **The file's own contents**: `(param, bytes, reserved)` per image, in
    /// the order the payload holds them.
    pub images: &'a [(u64, u64, u64)],
}

impl Identity<'_> {
    /// This identity as one number.
    ///
    /// **Mixes [`TIER_FORMAT`] and not [`super::FORMAT`]**: a key that moved
    /// when the other file's format moved would throw away a valid serving
    /// artifact every time the resident one's bytes changed meaning.
    #[must_use]
    pub fn key(&self) -> u64 {
        let mut hash = Fnv::default();
        hash.number(u64::from(TIER_FORMAT));
        hash.field(self.checkpoint.as_os_str().as_encoded_bytes());
        hash.field(self.trace_name.as_bytes());
        hash.field(self.plan_json);
        hash.number(self.total);
        mix_triples(&mut hash, self.layout);
        mix_triples(&mut hash, self.images);
        hash.number(stat_identity(self.checkpoint));
        hash.finish()
    }
}

/// The length, then every number — so a shorter layout is a different key and
/// not a prefix of a longer one.
fn mix_triples(hash: &mut Fnv, triples: &[(u64, u64, u64)]) {
    hash.number(triples.len() as u64);
    for &(one, two, three) in triples {
        hash.number(one);
        hash.number(two);
        hash.number(three);
    }
}

// ── the header ──────────────────────────────────────────────────────────────

/// **What every serving artifact says about itself before its bytes.**
///
/// The whole of [`TIER_HEADER`], decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Head {
    /// [`TIER_FORMAT`] as this file states it.
    pub format: u32,
    /// What the file says about its own contents — [`FLAG_ADAPTERS_ZEROED`]
    /// and whatever joins it. **Flags describe the BYTES, never the parse**,
    /// so a bit this build does not know cannot change how a field is read; a
    /// reader that needs one tests for it by name.
    pub flags: u32,
    /// [`Identity::key`] — which deployment's images these are.
    pub key: u64,
    /// Where the first index entry starts.
    pub index_at: u64,
    /// How many images this file holds.
    pub entries: u64,
    /// FNV-1a over the entries, written end to end in file order.
    pub index_digest: u64,
    /// Where the block digest table starts.
    pub blocks_at: u64,
    /// [`TIER_BLOCK`] as this file states it. Read back rather than assumed,
    /// because it is what turns a block ordinal into a byte range and a reader
    /// that used its own constant would hash the wrong span of a file written
    /// at another one.
    pub block_bytes: u64,
    /// How many digests the table holds.
    pub blocks: u64,
    /// The fold over every one of them, in order — the one number that stands
    /// for "the whole payload is the payload that was written".
    pub digest: u64,
    /// Where the images begin — a [`TIER_ALIGN`] multiple.
    pub payload_at: u64,
    /// How many bytes they occupy.
    pub payload_total: u64,
}

impl Head {
    /// Where `index_digest` sits. The header is written twice — once with the
    /// digests blank, once whole over the top once the bytes have crossed —
    /// so the offsets a rewriter needs are part of the format.
    pub const INDEX_DIGEST_AT: u64 = 40;

    /// The payload's byte range within the file.
    #[must_use]
    pub fn payload(&self) -> Range<u64> {
        self.payload_at..self.payload_at.saturating_add(self.payload_total)
    }

    /// **How many blocks one entry's span is divided into.**
    #[must_use]
    pub fn blocks_of(&self, reserved: u64) -> u64 {
        match self.block_bytes {
            0 => 0,
            step => reserved.div_ceil(step),
        }
    }

    /// **The byte range of one block of an entry**, relative to the payload.
    ///
    /// Block `which` of an entry at `at` spanning `reserved`. The last block
    /// carries whatever the division left over, so the blocks of an entry tile
    /// its span exactly — which is what [`read_spans_into`]'s "every byte is
    /// written" rests on.
    #[must_use]
    pub fn block_span(&self, at: u64, reserved: u64, which: u64) -> (u64, u64) {
        let step = self.block_bytes.max(1);
        let from = which.saturating_mul(step).min(reserved);
        let upto = from.saturating_add(step).min(reserved);
        (at.saturating_add(from), upto - from)
    }

    /// The bytes that go on the front of the file.
    #[must_use]
    fn encode(&self) -> [u8; TIER_HEADER] {
        let mut out = [0u8; TIER_HEADER];
        out[..8].copy_from_slice(&MAGIC);
        out[8..12].copy_from_slice(&self.format.to_le_bytes());
        out[12..16].copy_from_slice(&self.flags.to_le_bytes());
        for (at, value) in [
            (16, self.key),
            (24, self.index_at),
            (32, self.entries),
            (40, self.index_digest),
            (48, self.blocks_at),
            (56, self.block_bytes),
            (64, self.blocks),
            (72, self.digest),
            (80, self.payload_at),
            (88, self.payload_total),
        ] {
            out[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
        out
    }

    /// Read one back. `None` for anything that is not a serving artifact at
    /// all — too short, or the wrong magic — which is the only outcome this
    /// function judges. **The version is decoded, not checked**, so that the
    /// refusal can name both numbers (§K.5).
    #[must_use]
    fn decode(bytes: &[u8]) -> Option<Head> {
        let format = format_in(bytes)?;
        if bytes.len() < TIER_HEADER {
            return None;
        }
        // Every slice below is inside the length just checked, so the
        // fallbacks are unreachable rather than lenient.
        let long = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or([0; 8]));
        Some(Head {
            format,
            flags: u32::from_le_bytes(bytes[12..16].try_into().unwrap_or([0; 4])),
            key: long(16),
            index_at: long(24),
            entries: long(32),
            index_digest: long(40),
            blocks_at: long(48),
            block_bytes: long(56),
            blocks: long(64),
            digest: long(72),
            payload_at: long(80),
            payload_total: long(88),
        })
    }

    /// **Everything this header claims about itself, checked against itself
    /// and against the file's length.**
    ///
    /// A header is arithmetic before it is data: the index has to fit after
    /// the header, the block table after the index, the payload on its
    /// boundary after both, and the last byte of it inside the file. A header
    /// that fails any of those would hand a caller a span nobody bounded,
    /// which is the one failure mode a serving door must not have.
    fn arithmetic(&self, holds: u64) -> Result<(), Refused> {
        if self.index_at < TIER_HEADER as u64 {
            return Err(Refused::IndexCorrupt {
                why: format!(
                    "starts at byte {} and the header runs to {TIER_HEADER}",
                    self.index_at
                ),
            });
        }
        if self.block_bytes == 0 && self.payload_total > 0 {
            return Err(Refused::IndexCorrupt {
                why: format!(
                    "states a {}-byte payload divided into blocks of no bytes",
                    self.payload_total
                ),
            });
        }
        let index_end = self.index_at.saturating_add(self.entries * ENTRY as u64);
        if self.blocks_at < index_end {
            return Err(Refused::IndexCorrupt {
                why: format!(
                    "runs to byte {index_end} and states a block table at {}, inside \
                     itself",
                    self.blocks_at
                ),
            });
        }
        let blocks_end = self.blocks_at.saturating_add(self.blocks.saturating_mul(8));
        if self.payload_at % TIER_ALIGN != 0 {
            return Err(Refused::IndexCorrupt {
                why: format!(
                    "states a payload at byte {}, which is not a {TIER_ALIGN}-byte \
                     boundary",
                    self.payload_at
                ),
            });
        }
        if self.payload_at < blocks_end {
            return Err(Refused::IndexCorrupt {
                why: format!(
                    "states a payload at byte {}, inside the block table that runs to \
                     {blocks_end}",
                    self.payload_at
                ),
            });
        }
        let Some(end) = self.payload_at.checked_add(self.payload_total) else {
            return Err(Refused::IndexCorrupt {
                why: format!(
                    "states a payload of {} bytes at {}, which overflows",
                    self.payload_total, self.payload_at
                ),
            });
        };
        if end > holds {
            return Err(Refused::Truncated { states: end, holds });
        }
        Ok(())
    }
}

// ── where the file lives ────────────────────────────────────────────────────

/// **Where this key's serving artifact lives under `dir`.**
///
/// [`super::artifact_path`]'s shape under a different extension, and the
/// difference is load-bearing: the two files share a directory and a key
/// space, and a resident artifact that shadowed this one would be opened by a
/// reader expecting an image sequence and refused as a stranger's file rather
/// than as a stale one.
#[must_use]
pub fn path(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.tiers"))
}

/// The key a path NAMES, when the name is one [`path`] would have produced.
///
/// `None` for anything else, because a caller may point [`Artifact::open`] at
/// a file it named itself and that is not a disagreement.
fn key_in(path: &Path) -> Option<u64> {
    let name = path.file_name()?.to_str()?;
    let (stem, rest) = name.split_once('.')?;
    if rest != "tiers" || stem.len() != 16 {
        return None;
    }
    u64::from_str_radix(stem, 16).ok()
}

/// **The format word, out of the front of a file whose magic is ours.**
///
/// `None` for anything else — a stranger's bytes, or too few of them to carry
/// a version at all. The twelve bytes this reads are the only ones every
/// format of this file agrees about, which is what lets the version refusal
/// name a build whose header was a different LENGTH rather than dismissing its
/// file as no artifact.
#[must_use]
fn format_in(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 12 || bytes[..8] != MAGIC {
        return None;
    }
    Some(u32::from_le_bytes(bytes[8..12].try_into().ok()?))
}

// ── what a writer states ────────────────────────────────────────────────────

/// **The entries TILE THE PAYLOAD, consecutively, from zero.**
///
/// The one structural claim about this format that a reader can check without
/// the bytes, and the reason it is checked: the payload is a concatenation and
/// nothing else, so a gap would be bytes no digest covers and an overlap would
/// be two entries claiming one image. Both are files a boot must not cut.
///
/// A `Some` says which entry broke it and how.
fn payload_fault(entries: &[Group], total: u64) -> Option<String> {
    let mut at = 0u64;
    for (which, group) in entries.iter().enumerate() {
        if group.offset != at {
            return Some(format!(
                "does not tile its payload: image {which} (param {}) starts at byte {} \
                 and {at} was next, and the images are a concatenation",
                group.id, group.offset,
            ));
        }
        if group.bytes > group.reserved {
            return Some(format!(
                "does not tile its payload: image {which} (param {}) publishes {} bytes \
                 into a {}-byte span",
                group.id, group.bytes, group.reserved,
            ));
        }
        let Some(end) = at.checked_add(group.reserved) else {
            return Some(format!(
                "does not tile its payload: image {which} (param {}) states a span that \
                 overflows: {at} + {}",
                group.id, group.reserved,
            ));
        };
        at = end;
    }
    if at != total {
        return Some(format!(
            "states a {total}-byte payload and its {} images tile {at} of it",
            entries.len(),
        ));
    }
    None
}

/// **Does this hash move the corruption counter?** — see [`Artifact::intact`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Count {
    /// A serving door: a disagreement here is a load refusing bytes.
    It,
    /// The writer, asking whether it may skip.
    Not,
}

impl Count {
    fn corrupt(self) {
        if self == Count::It {
            bump(Stat::Corrupt);
        }
    }
}

/// **The digest of a run of bytes**, which is one block's answer.
#[must_use]
fn block_digest(bytes: &[u8]) -> u64 {
    let mut hash = Fnv::default();
    hash.raw(bytes);
    hash.finish()
}

/// **The one number a file's blocks fold to** — [`Head::digest`], and what a
/// reader compares when it has checked every block and wants to say so once.
#[must_use]
fn fold(blocks: &[u64]) -> u64 {
    let mut hash = Fnv::default();
    hash.number(blocks.len() as u64);
    for digest in blocks {
        hash.number(*digest);
    }
    hash.finish()
}

// ── reading one ─────────────────────────────────────────────────────────────

/// **A serving artifact, open and mapped.**
///
/// Holds the file's whole mapping for as long as it lives and answers what a
/// cut needs: what this file is, which images it carries, where one of them
/// lives, and whether the bytes are the bytes that were written. **Nothing is
/// copied.** A `&[u8]` handed out here points into the mapping.
#[derive(Debug)]
pub struct Artifact {
    path: PathBuf,
    head: Head,
    entries: Vec<Group>,
    blocks: Vec<u64>,
    /// Entry `i`'s first digest in `blocks`. One extra entry at the end, so
    /// entry `i`'s digests are `blocks[first[i]..first[i + 1]]` for every `i`.
    first: Vec<usize>,
    map: Map,
}

impl Artifact {
    /// **Open the serving artifact at `path` and map it.**
    ///
    /// Validates, in this order: the magic, the version, the header's own
    /// arithmetic against the file's length, the key the filename states, the
    /// index's digest, that the entries tile the payload, and that the block
    /// table is exactly as long as the entries say it should be. Each failure
    /// is a [`Refused`] with what it disagreed about in it — a caller that
    /// asked to serve out of a file has no recipe to fall back to.
    ///
    /// # Errors
    ///
    /// Any [`Refused`]. In particular [`Refused::StaleFormat`] with both
    /// numbers for a file from another build (§K.5), and [`Refused::WrongKey`]
    /// for a file whose name and header disagree.
    pub fn open(path: &Path) -> Result<Artifact, Refused> {
        let (head, entries, blocks) = read_head(path)?;
        let file = fs::File::open(path).map_err(|why| Refused::Unreadable {
            why: format!("{why}"),
        })?;
        let holds = file
            .metadata()
            .map_err(|why| Refused::Unreadable {
                why: format!("{why}"),
            })?
            .len();
        let map = Map::open(&file, usize::try_from(holds).unwrap_or(0))
            .map_err(|why| Refused::Unmappable { why })?;
        let first = first_blocks(&head, &entries);
        Ok(Artifact {
            path: path.to_path_buf(),
            head,
            entries,
            blocks,
            first,
            map,
        })
    }

    /// Where this artifact came from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Everything the file says about itself.
    #[must_use]
    pub fn head(&self) -> Head {
        self.head
    }

    /// Which deployment's images these are — [`Identity::key`].
    #[must_use]
    pub fn key(&self) -> u64 {
        self.head.key
    }

    /// **Every image this file holds, in the order the payload holds them** —
    /// the sequence a cut walks.
    #[must_use]
    pub fn entries(&self) -> &[Group] {
        &self.entries
    }

    /// The block digest table, whole. What [`read_spans_into`] is handed.
    #[must_use]
    pub fn blocks(&self) -> &[u64] {
        &self.blocks
    }

    /// **One image's entry and its first block**, or `None` for a param this
    /// file does not carry.
    ///
    /// The pair every restoring verb needs: where the bytes are, and which
    /// digests answer for them.
    #[must_use]
    pub fn locate(&self, param: u32) -> Option<(Group, u64)> {
        let at = self.entries.iter().position(|group| group.id == param)?;
        Some((self.entries[at], self.first[at] as u64))
    }

    /// Where one image sits in the payload, or `None`.
    #[must_use]
    pub fn resolve(&self, param: u32) -> Option<Group> {
        self.entries.iter().find(|group| group.id == param).copied()
    }

    /// The payload's bytes, borrowed from the mapping.
    ///
    /// **NOTHING IS COPIED AND NOTHING IS READ YET.** The slice is a window on
    /// the mapping; the first touch of each page is the NVMe read.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        let at = usize::try_from(self.head.payload_at).unwrap_or(usize::MAX);
        let len = usize::try_from(self.head.payload_total).unwrap_or(0);
        self.map
            .bytes()
            .get(at..at.saturating_add(len))
            .unwrap_or(&[])
    }

    /// A window on the payload, borrowed from the mapping.
    #[must_use]
    pub fn span(&self, at: u64, len: u64) -> Option<&[u8]> {
        let from = usize::try_from(at).ok()?;
        let want = usize::try_from(len).ok()?;
        self.payload().get(from..from.checked_add(want)?)
    }

    /// **One image's PUBLISHED bytes, borrowed from the mapping.**
    ///
    /// `bytes`, not `reserved`: the reserved span is what the payload GIVES
    /// the image, and the bytes past the published length are the padding no
    /// reader should be handed.
    #[must_use]
    pub fn plane(&self, param: u32) -> Option<&[u8]> {
        let group = self.resolve(param)?;
        self.span(group.offset, group.bytes)
    }

    /// **Hash the blocks of these images and compare them to the table.**
    ///
    /// Per block, [`TIER_READERS`] at a time, and the refusal names the image,
    /// the block and its byte range — "the digest is wrong" about a hundred
    /// gigabytes says less than it could when the payload is already divided
    /// into an answer per sixty-four mebibytes.
    ///
    /// **This is the door that counts a corruption**, because it is the door
    /// where bytes are checked. It reads the pages it hashes, which is what
    /// makes it also the thing that warms the page cache the first fires would
    /// otherwise fault in one page at a time.
    ///
    /// # Errors
    ///
    /// [`Refused::IndexCorrupt`] naming the image, the block, its byte range
    /// and both digests.
    pub fn verify_entries(&self, params: &[u32]) -> Result<(), Refused> {
        let mut want: Vec<usize> = Vec::with_capacity(params.len());
        for param in params {
            let Some(at) = self.entries.iter().position(|group| group.id == *param) else {
                bump(Stat::Corrupt);
                // Not `count`: this arm is reached before any hashing and only
                // from `verify_entries`, whose callers are all serving doors.
                return Err(Refused::IndexCorrupt {
                    why: format!("carries no image for param {param}"),
                });
            };
            want.push(at);
        }
        self.verify_at(&want, Count::It)
    }

    /// **Hash every block of the payload.** [`Artifact::verify_entries`] over
    /// all of them, and the one call that also checks the fold — a reader that
    /// has checked the whole file may compare one number with the header
    /// rather than a table with itself.
    ///
    /// It reads the whole payload, so a caller that means to serve lazily out
    /// of the mapping pays for it deliberately.
    ///
    /// # Errors
    ///
    /// The first block's [`Refused::IndexCorrupt`].
    pub fn verify(&self) -> Result<(), Refused> {
        self.verify_all(Count::It)
    }

    /// **IS THIS FILE THE FILE IT SAYS IT IS?** — [`Artifact::verify`] with
    /// the counter struck out, and the ONE caller is [`store`]'s already-on-
    /// the-disk check.
    ///
    /// The distinction is what the counter means. `corrupt` answers *"how many
    /// times did this process refuse bytes it was about to serve?"*, and every
    /// door that counts is a door where a load was reading. The WRITER is not
    /// reading: it is asking whether it may skip a hundred gigabytes of disk,
    /// and a `false` is an answer to that question rather than a second report
    /// of a fault. Counting here would say `corrupt: 2` about one bad file on
    /// one boot — once where the restore refused it, once where the write that
    /// repaired it looked — which is a number no operator can act on.
    ///
    /// A rot that ONLY this door ever sees — `pie model import` over a file no
    /// boot has read — therefore goes uncounted, and that is the right silence
    /// too: the operator asked for a rebuild and got one.
    #[must_use]
    pub fn intact(&self) -> bool {
        self.verify_all(Count::Not).is_ok()
    }

    /// [`Artifact::verify`]'s body, over every entry and then the fold.
    fn verify_all(&self, count: Count) -> Result<(), Refused> {
        let all: Vec<usize> = (0..self.entries.len()).collect();
        self.verify_at(&all, count)?;
        let folded = fold(&self.blocks);
        if folded != self.head.digest {
            count.corrupt();
            return Err(Refused::IndexCorrupt {
                why: format!(
                    "states digest {:016x} over a block table that folds to {folded:016x}",
                    self.head.digest,
                ),
            });
        }
        Ok(())
    }

    /// [`Artifact::verify_entries`] by index, which is what every caller has
    /// by the time it gets here.
    fn verify_at(&self, which: &[usize], count: Count) -> Result<(), Refused> {
        // One work item per block, so a two-hundred-gigabyte image and a
        // kilobyte one are the same shape of job and the readers do not idle
        // behind whichever entry happens to be the largest.
        let mut work: Vec<(usize, usize, u64, u64)> = Vec::new();
        for at in which {
            let group = self.entries[*at];
            for block in 0..self.head.blocks_of(group.reserved) {
                let (from, len) = self.head.block_span(group.offset, group.reserved, block);
                work.push((*at, self.first[*at] + block as usize, from, len));
            }
        }
        let payload = self.payload();
        let found: Vec<Option<u64>> = std::thread::scope(|scope| {
            let width = TIER_READERS.min(work.len().max(1));
            let per = work.len().div_ceil(width.max(1));
            let mut hashing = Vec::with_capacity(width);
            for lane in 0..width {
                let mine = work
                    .get(lane * per..((lane + 1) * per).min(work.len()))
                    .unwrap_or(&[]);
                hashing.push(scope.spawn(move || {
                    mine.iter()
                        .map(|(_, _, from, len)| {
                            let at = usize::try_from(*from).ok()?;
                            let want = usize::try_from(*len).ok()?;
                            payload.get(at..at.checked_add(want)?).map(block_digest)
                        })
                        .collect::<Vec<Option<u64>>>()
                }));
            }
            hashing
                .into_iter()
                .flat_map(|thread| thread.join().unwrap_or_default())
                .collect()
        });
        for ((entry, block, from, len), found) in work.iter().zip(found) {
            let group = self.entries[*entry];
            let stated = self.blocks.get(*block).copied().unwrap_or(0);
            if found != Some(stated) {
                count.corrupt();
                // The ordinal is the block's place IN ITS OWN IMAGE and not in
                // the table: "block 3 of param 1's image" about an image with
                // one block is a sentence an operator cannot act on, and the
                // payload byte range beside it is the address they can.
                let which = block - self.first[*entry];
                return Err(Refused::IndexCorrupt {
                    why: match found {
                        Some(found) => format!(
                            "states digest {stated:016x} for block {which} of param {}'s \
                             image (payload bytes {from}..{}), whose bytes hash to \
                             {found:016x}",
                            group.id,
                            from + len,
                        ),
                        None => format!(
                            "states a block {which} of param {}'s image at payload bytes \
                             {from}..{} and maps no such bytes",
                            group.id,
                            from + len,
                        ),
                    },
                });
            }
        }
        Ok(())
    }
}

/// Entry `i`'s first digest in the block table, plus a sentinel — so entry
/// `i`'s digests are `blocks[first[i]..first[i + 1]]` for every `i`, with no
/// special case for the last.
fn first_blocks(head: &Head, entries: &[Group]) -> Vec<usize> {
    let mut out = Vec::with_capacity(entries.len() + 1);
    let mut at = 0usize;
    for group in entries {
        out.push(at);
        at += usize::try_from(head.blocks_of(group.reserved)).unwrap_or(0);
    }
    out.push(at);
    out
}

/// **Read a serving artifact's header, its index and its block table without
/// mapping it.**
///
/// What [`Artifact::open`] does first, and what a caller that only wants to
/// know whether a key is on the disk can call on its own.
///
/// # Errors
///
/// Every [`Refused`] except [`Refused::Unmappable`], which only the mapping
/// can produce.
pub fn read_head(path: &Path) -> Result<(Head, Vec<Group>, Vec<u64>), Refused> {
    use std::io::Read;
    use std::os::unix::fs::FileExt;

    let mut file = fs::File::open(path).map_err(|why| Refused::Unreadable {
        why: format!("{why}"),
    })?;
    let holds = file
        .metadata()
        .map_err(|why| Refused::Unreadable {
            why: format!("{why}"),
        })?
        .len();

    // **THE FRONT OF THE FILE, FOR WHATEVER IT HOLDS.** Not `read_exact` over
    // a whole header: a build whose header was a different LENGTH — which is
    // precisely what a [`TIER_FORMAT`] bump leaves on the disk — would answer
    // `NotAnArtifact`, and the boot needs that file refused BY VERSION so it
    // can name `pie model import`. Only the twelve bytes every format of this
    // file agrees about are believed until the version does.
    let mut bytes = [0u8; TIER_HEADER];
    let mut have = 0usize;
    loop {
        match file.read(&mut bytes[have..]) {
            Ok(0) => break,
            Ok(read) => have += read,
            Err(why) if why.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(why) => {
                return Err(Refused::Unreadable {
                    why: format!("{why}"),
                });
            }
        }
    }
    // A short read is not an artifact rather than a truncated one: a file with
    // no header has not made a claim this reader could find it short of.
    let Some(states) = format_in(&bytes[..have]) else {
        return Err(Refused::NotAnArtifact);
    };
    // **THE VERSION, BEFORE ANYTHING ELSE IS BELIEVED.** Every field after the
    // format word means whatever the format says it means.
    if states != TIER_FORMAT {
        return Err(Refused::StaleFormat {
            states,
            reads: TIER_FORMAT,
        });
    }
    let Some(head) = Head::decode(&bytes[..have]) else {
        return Err(Refused::NotAnArtifact);
    };
    head.arithmetic(holds)?;
    // **THE NAME AND THE HEADER ARE ONE CLAIM MADE TWICE.** A file under one
    // key's name holding another key's images is the case that would restore
    // the wrong deployment's weights with every digest agreeing.
    if let Some(names) = key_in(path)
        && names != head.key
    {
        return Err(Refused::WrongKey {
            states: head.key,
            names,
        });
    }

    let entries = usize::try_from(head.entries).unwrap_or(usize::MAX);
    let mut raw = vec![0u8; entries.saturating_mul(ENTRY)];
    if entries > 0 {
        file.read_exact_at(&mut raw, head.index_at)
            .map_err(|why| Refused::Unreadable {
                why: format!("{why}"),
            })?;
    }
    let mut all = Vec::with_capacity(entries);
    for entry in raw.chunks_exact(ENTRY) {
        let Some(group) = Group::decode(entry) else {
            return Err(Refused::IndexCorrupt {
                why: "has an entry that does not decode".to_string(),
            });
        };
        all.push(group);
    }
    if index_digest(&all) != head.index_digest {
        return Err(Refused::IndexCorrupt {
            why: "does not match its digest".to_string(),
        });
    }
    if let Some(why) = payload_fault(&all, head.payload_total) {
        return Err(Refused::IndexCorrupt { why });
    }
    // **THE TABLE IS AS LONG AS THE ENTRIES SAY IT IS.** The block ordinals a
    // reader computes are a function of the index, so a table of another
    // length is a file whose two halves were written from different walks —
    // and every digest lookup after this point would be off by whatever the
    // difference was.
    let want: u64 = all.iter().map(|group| head.blocks_of(group.reserved)).sum();
    if want != head.blocks {
        return Err(Refused::IndexCorrupt {
            why: format!(
                "states {} block digests and its {} images divide into {want}",
                head.blocks,
                all.len(),
            ),
        });
    }
    let mut raw = vec![0u8; usize::try_from(head.blocks).unwrap_or(usize::MAX) * 8];
    if head.blocks > 0 {
        file.read_exact_at(&mut raw, head.blocks_at)
            .map_err(|why| Refused::Unreadable {
                why: format!("{why}"),
            })?;
    }
    let blocks: Vec<u64> = raw
        .chunks_exact(8)
        .map(|word| u64::from_le_bytes(word.try_into().unwrap_or([0; 8])))
        .collect();
    Ok((head, all, blocks))
}

/// **One image, and where its bytes are to be read** — [`read_spans_into`]'s
/// unit of work.
///
/// It is a WHOLE ENTRY and never part of one, because the block digests are
/// entry-local: a partial read could not close the block it stopped inside,
/// and a reader that skipped that block's check would be verifying less than
/// the bytes it used.
pub struct Span {
    /// The image's offset in the payload — [`Group::offset`].
    pub at: u64,
    /// Its whole span — [`Group::reserved`].
    pub len: u64,
    /// Its first digest in the block table — [`Artifact::locate`]'s other
    /// half.
    pub first_block: u64,
    /// Where the bytes go.
    pub into: *mut u8,
}

/// A span's destination, as something a scope thread may carry. A raw pointer
/// is not `Send` and must not be; what makes these sound to move is
/// [`read_spans_into`]'s own safety clause, which says the destinations are
/// disjoint and nobody else names a byte of them.
struct Carried(*mut u8);

// SAFETY: as above — one address per span, moved once, into a thread that is
// the sole writer of the bytes it names for as long as the scope is open.
unsafe impl Send for Carried {}
// SAFETY: the work list is shared by reference and sliced DISJOINTLY, one
// slice per lane, so no two threads ever reach the same `Carried` — and the
// caller's clause says no two spans name the same byte either.
unsafe impl Sync for Carried {}

/// **READ THESE IMAGES STRAIGHT INTO THEIR DESTINATIONS, VERIFYING THEM AS
/// THEY ARRIVE** (§K.4's host restore, at §M's granularity).
///
/// [`TIER_READERS`] threads over the spans, each reading its own with
/// positioned reads a [`TIER_BLOCK`] at a time and closing that block's FNV
/// chain from the bytes as they land. No staging buffer exists anywhere on
/// this path: the read target IS the destination, and the digest is taken over
/// what is now in that destination rather than over a second copy from the
/// mapping. That is the stronger claim of the two — it verifies the bytes the
/// caller will USE — and it is also the only one that costs a single pass over
/// the disk.
///
/// **WHY THE BLOCK IS THE UNIT AND NOT A CHUNK OF ONE.** An FNV chain is
/// serial over its own span, so a reader may only feed the block it is walking,
/// in order. Splitting a block further would buy queue depth and lose the
/// overlap. [`TIER_READERS`] concurrent 64 MiB positioned reads is where this
/// box's filesystem stops answering faster (5.41 GB/s), which is the same
/// measurement that picked the count.
///
/// **EVERY BYTE OF EVERY SPAN IS WRITTEN BEFORE THIS RETURNS `Ok`** — each
/// image's whole `reserved` span, padding included, because the blocks tile it
/// exactly. That is the promise
/// [`Pinned::mapped_uninit`](crate::device::alloc::Pinned::mapped_uninit) asks
/// its caller for, and this function is where it is kept. On `Err` the
/// destinations hold an indeterminate mixture and the caller owes them a
/// zeroing before anything reads them.
///
/// # Safety
///
/// Each span's `into` must be valid for writes of its `len` bytes, the spans'
/// destinations must be pairwise disjoint, and no other agent — no thread, no
/// kernel, no guest — may read or write any of them for the duration of the
/// call.
///
/// # Errors
///
/// [`Refused::Unreadable`] for the filesystem's own words, which is a machine
/// failure and not a claim about the file; [`Refused::IndexCorrupt`] naming
/// the block, its byte range and both digests, which IS a claim about the file
/// and is counted here.
pub unsafe fn read_spans_into(
    path: &Path,
    head: &Head,
    blocks: &[u64],
    spans: &[Span],
) -> Result<(), Refused> {
    use std::os::unix::fs::FileExt;

    if spans.is_empty() {
        return Ok(());
    }
    let file = fs::File::open(path).map_err(|why| Refused::Unreadable {
        why: format!("{why}"),
    })?;

    // One work item per BLOCK for the same reason `verify_at` has one: the
    // spans are as uneven as the model's planes are, and a reader that took a
    // span each would spend its last minutes with seven idle cores behind the
    // largest bank.
    let mut work: Vec<(u64, u64, u64, Carried)> = Vec::new();
    for span in spans {
        for block in 0..head.blocks_of(span.len) {
            let (from, len) = head.block_span(span.at, span.len, block);
            // SAFETY: `from - span.at + len <= span.len`, and the caller
            // states `into` is valid for `span.len` bytes.
            let into = Carried(unsafe {
                span.into
                    .add(usize::try_from(from - span.at).unwrap_or(usize::MAX))
            });
            work.push((span.first_block + block, from, len, into));
        }
    }

    let found: Vec<Result<(u64, u64), String>> = std::thread::scope(|scope| {
        let width = TIER_READERS.min(work.len().max(1));
        let per = work.len().div_ceil(width.max(1));
        let mut reading = Vec::with_capacity(width);
        for lane in 0..width {
            let mine = work
                .get(lane * per..((lane + 1) * per).min(work.len()))
                .unwrap_or(&[]);
            let file = &file;
            let payload_at = head.payload_at;
            reading.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(mine.len());
                for (block, from, len, into) in mine {
                    // SAFETY: the caller's clause, plus the disjointness the
                    // block walk above preserves: no two work items name the
                    // same byte of any destination.
                    let dst = unsafe {
                        core::slice::from_raw_parts_mut(
                            into.0,
                            usize::try_from(*len).unwrap_or(0),
                        )
                    };
                    out.push(match file.read_exact_at(dst, payload_at + from) {
                        Ok(()) => Ok((*block, block_digest(dst))),
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
    let mut rotten: Option<(u64, u64, u64)> = None;
    for outcome in found {
        match outcome {
            Ok((block, digest)) => {
                let stated = blocks.get(usize::try_from(block).unwrap_or(usize::MAX)).copied();
                if stated != Some(digest) && rotten.is_none() {
                    rotten = Some((block, stated.unwrap_or(0), digest));
                }
            }
            Err(why) => failure = failure.or(Some(why)),
        }
    }
    // The filesystem's answer first: a read that did not happen leaves a
    // destination nobody wrote, and calling that a corruption would name the
    // file for the machine's failure.
    if let Some(why) = failure {
        return Err(Refused::Unreadable { why });
    }
    if let Some((block, stated, found)) = rotten {
        bump(Stat::Corrupt);
        return Err(Refused::IndexCorrupt {
            why: format!(
                "does not hash to its own block table: block {block} of the payload is \
                 stated as {stated:016x} and read back as {found:016x}"
            ),
        });
    }
    Ok(())
}

// ── writing one ─────────────────────────────────────────────────────────────

/// **Snapshot this deployment's images beside `key`, if there is room and if
/// the disk does not already hold them.**
///
/// [`write_out`]'s counted wrapper, on [`super::store_indexed`]'s shape: best
/// effort in every direction, and **a declined write never breaks a load** —
/// the load already succeeded, the images this is reading from are the answer.
/// No directory means the feature is off: no write, and no counter, because a
/// load that was never offered one did not decline one.
///
/// # The write this does NOT do
///
/// **A KEY ALREADY ON THE DISK IS ALREADY THESE BYTES.** The key is a function
/// of everything the payload is a function of — the checkpoint's stat, the
/// recipe, the uncapped layout, every image's span (§M.3) — so a file under
/// this key that HASHES TO ITS OWN TABLE holds what this call would have
/// written, and writing it again would spend a hundred gigabytes of disk
/// bandwidth to produce the same file. That boot counts `skipped` and returns.
///
/// **AND THE CHECK IS A FULL VERIFY, WHICH IT DID NOT USED TO BE** (§M.4).
/// Format 2 checked only that the header and index parsed, on the argument
/// that a file lying about its BYTES was the reader's to catch — and the
/// reader caught it by DELETING it, so the next write had nothing to skip.
/// This is the model now and no reader deletes anything, which makes this
/// call the only door that ever replaces a bad file. A skip on a parse alone
/// would leave a deployment cold forever with `pie model import` unable to fix
/// it, which is not a remedy.
///
/// The cost is a re-hash of the payload, and it is charged where it is
/// affordable: this runs only after a COLD load, which means either there was
/// no file or the file was refused. In the first case [`Artifact::open`] fails
/// on the first read and nothing is hashed; in the second the hash is seconds
/// behind a load that took minutes.
///
/// # And THIS IS THE REBUILD DOOR — the only one (§M-3)
///
/// The verify-then-replace above is the whole authority to overwrite a serving
/// artifact, and §M-3 moved it: the one caller left is
/// [`weights::write_tiers`](crate::weights), which a
/// [`Weights::resident`](crate::weights::Weights) reached only under
/// `Intent::Prepare` — that is, only from `Shell::prepare`, only from
/// `pie model import`. A serving boot cannot reach this function at all, which
/// is why the refusals above may promise a command rather than a next boot:
/// the operator running that command IS the caller of this one.
///
/// Naming it here rather than only at the call site because the property this
/// paragraph states is about the WRITER — "an artifact is replaced by an
/// import and by nothing else" — and a second caller appearing under a serving
/// path would break it silently. There is one call. Keep it that way.
///
/// It is [`Artifact::intact`] and not [`Artifact::verify`], and that door's
/// own doc says why: a writer asking whether it may skip is not a load
/// refusing bytes, and counting it would report one bad file twice.
///
/// The replacement itself is atomic either way — a temp file and a rename over
/// whatever was there — so a reader holding the old inode goes on reading it.
pub fn store<Fill>(dir: Option<&Path>, key: u64, entries: &[Group], flags: u32, fill: Fill)
where
    Fill: FnMut(u32, u64, &mut [u8]) -> Result<(), String>,
{
    let Some(dir) = dir else {
        return;
    };
    if Artifact::open(&path(dir, key)).is_ok_and(|artifact| artifact.intact()) {
        bump(Stat::Skipped);
        return;
    }
    if let Err(why) = write_out(dir, key, entries, flags, fill) {
        decline(&why);
        return;
    }
    bump(Stat::Stored);
}

/// **Count a write this shell declined before it had images to write.**
///
/// [`store`]'s own refusal arm, reachable by name — because a caller can know
/// that it cannot honour the format before it can build an entry list. Said
/// out loud and counted, exactly like a write that failed on the filesystem,
/// because to the operator asking why the second boot is still cold they are
/// one answer.
///
/// A function rather than a `Refused`, for [`count_restored`]'s reason: the
/// load succeeded and there is nobody to return an error to.
pub fn decline(why: &str) {
    bump(Stat::Declined);
    eprintln!("engine-cuda: declined to write this load's serving artifact: {why}");
}

/// **Write the serving artifact for `key` under `dir`.**
///
/// `fill` is the only way bytes get in: it is handed a param, an offset within
/// that image, and a buffer to fill — at most [`TIER_BLOCK`] bytes at a time,
/// so the host never holds more than 64 MiB of an image that may be 64 GiB. It
/// is asked only for the image's PUBLISHED bytes; the padding up to the span
/// is written as the deterministic zeros a reader's digest will hash, here,
/// once, rather than by every caller.
///
/// Published atomically: the bytes land in `{key:016x}.tiers.{pid}.part` and
/// are renamed at the end, so a boot that dies mid-write leaves a partial file
/// nobody will ever name rather than a corrupt one under the key.
///
/// # Errors
///
/// An entry list that does not tile its own payload, a filesystem with less
/// than the file plus [`MARGIN`] free — **named with the key**, because the
/// disk cost is per deployment and an operator deleting one needs to know
/// which — or the filesystem's own words. `fill`'s refusals come back
/// verbatim.
pub fn write_out<Fill>(
    dir: &Path,
    key: u64,
    entries: &[Group],
    flags: u32,
    fill: Fill,
) -> Result<PathBuf, String>
where
    Fill: FnMut(u32, u64, &mut [u8]) -> Result<(), String>,
{
    fs::create_dir_all(dir).map_err(|why| format!("{dir:?}: {why}"))?;
    let total: u64 = entries.iter().map(|group| group.reserved).sum();
    let need = starts_of(entries)
        .2
        .saturating_add(total)
        .saturating_add(MARGIN);
    let free = available_bytes(dir)?;
    if free < need {
        return Err(format!(
            "{dir:?} has {:.1} GiB free and the serving artifact for key {key:016x} wants \
             {:.1} GiB ({} GiB of plane images plus a {} GiB margin); point `[model] \
             weight_cache_dir` at a disk with more space",
            free as f64 / (1u64 << 30) as f64,
            need as f64 / (1u64 << 30) as f64,
            total >> 30,
            MARGIN >> 30,
        ));
    }

    let final_path = path(dir, key);
    let temp_path = dir.join(format!("{key:016x}.tiers.{}.part", std::process::id()));
    if let Err(why) = emit(&temp_path, key, entries, flags, fill) {
        let _ = fs::remove_file(&temp_path);
        return Err(why);
    }
    fs::rename(&temp_path, &final_path).map_err(|why| {
        let _ = fs::remove_file(&temp_path);
        format!("publishing {final_path:?}: {why}")
    })?;
    Ok(final_path)
}

/// Where the index, the block table and the payload start, given the entries.
///
/// The one piece of arithmetic both the writer and the reader's validation
/// have to agree on, so it is written once.
fn starts_of(entries: &[Group]) -> (u64, u64, u64) {
    let index_at = TIER_HEADER as u64;
    let blocks_at = index_at + (entries.len() * ENTRY) as u64;
    let blocks: u64 = entries
        .iter()
        .map(|group| match TIER_BLOCK {
            0 => 0,
            step => group.reserved.div_ceil(step),
        })
        .sum();
    let payload_at = blocks_at
        .saturating_add(blocks.saturating_mul(8))
        .next_multiple_of(TIER_ALIGN);
    (index_at, blocks_at, payload_at)
}

/// **The whole format, written to one path** — [`write_out`]'s body without
/// the temp file, the rename or the space refusal, so that the `probe` writer
/// can name a file the real path never would and still produce exactly the
/// bytes the real path produces.
fn emit<Fill>(
    target: &Path,
    key: u64,
    entries: &[Group],
    flags: u32,
    mut fill: Fill,
) -> Result<(), String>
where
    Fill: FnMut(u32, u64, &mut [u8]) -> Result<(), String>,
{
    use std::io::Seek;

    let total: u64 = entries.iter().map(|group| group.reserved).sum();
    if let Some(why) = payload_fault(entries, total) {
        return Err(why);
    }
    let (index_at, blocks_at, payload_at) = starts_of(entries);
    let head = Head {
        format: TIER_FORMAT,
        flags,
        key,
        index_at,
        entries: entries.len() as u64,
        // Written last, over these placeholders, once the bytes have crossed;
        // a rewind is cheaper than holding the payload.
        index_digest: 0,
        blocks_at,
        block_bytes: TIER_BLOCK,
        blocks: 0,
        digest: 0,
        payload_at,
        payload_total: total,
    };
    let counted: u64 = entries.iter().map(|group| head.blocks_of(group.reserved)).sum();
    let head = Head {
        blocks: counted,
        ..head
    };

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|why| format!("{parent:?}: {why}"))?;
    }
    let mut file = fs::File::create(target).map_err(|why| format!("{target:?}: {why}"))?;
    file.write_all(&head.encode()).map_err(|why| format!("{why}"))?;
    for group in entries {
        file.write_all(&group.encode()).map_err(|why| format!("{why}"))?;
    }
    // The table's own placeholder, so the payload starts where the header says
    // it does; the digests go back over the top with the header.
    write_zeros(&mut file, counted * 8)?;
    write_zeros(&mut file, payload_at.saturating_sub(blocks_at + counted * 8))?;

    // Streamed, because the host never holds more than one block of an image
    // that may be tens of gigabytes — and the block IS the chunk, so the
    // digest a reader checks and the bytes a writer wrote come from one pass.
    let widest = entries.iter().map(|group| group.reserved).max().unwrap_or(0);
    let step = usize::try_from(TIER_BLOCK.min(widest.max(1))).unwrap_or(CHUNK);
    let mut chunk = vec![0u8; step];
    let mut blocks = Vec::with_capacity(usize::try_from(counted).unwrap_or(0));
    for group in entries {
        let mut done = 0u64;
        while done < group.reserved {
            let want = usize::try_from(group.reserved - done)
                .unwrap_or(usize::MAX)
                .min(usize::try_from(TIER_BLOCK).unwrap_or(usize::MAX))
                .min(chunk.len());
            let slice = &mut chunk[..want];
            // **THE PADDING IS WRITTEN HERE AND NOWHERE ELSE.** `fill` answers
            // for the published bytes; everything past them is the zeros the
            // store and the pinned tier already hold there, made explicit so
            // that two boots of one deployment write one file.
            let published = usize::try_from(group.bytes.saturating_sub(done))
                .unwrap_or(usize::MAX)
                .min(want);
            slice[published..].fill(0);
            if published > 0 {
                fill(group.id, done, &mut slice[..published])?;
            }
            blocks.push(block_digest(slice));
            file.write_all(slice).map_err(|why| format!("{why}"))?;
            done += want as u64;
        }
    }
    if blocks.len() as u64 != counted {
        return Err(format!(
            "wrote {} block digests for {counted} blocks; the writer and the header \
             counted differently",
            blocks.len(),
        ));
    }

    // The block table and the whole header go back over the top, rather than
    // seven seeks to seven fields: the digests are the only things that
    // changed, and re-encoding what was already computed cannot get them out
    // of step.
    file.seek(std::io::SeekFrom::Start(blocks_at))
        .map_err(|why| format!("{why}"))?;
    for digest in &blocks {
        file.write_all(&digest.to_le_bytes()).map_err(|why| format!("{why}"))?;
    }
    let head = Head {
        index_digest: index_digest(entries),
        digest: fold(&blocks),
        ..head
    };
    file.seek(std::io::SeekFrom::Start(0))
        .map_err(|why| format!("{why}"))?;
    file.write_all(&head.encode()).map_err(|why| format!("{why}"))?;
    file.sync_all().map_err(|why| format!("{why}"))
}

/// `len` zero bytes, a chunk at a time, so a 2 MiB gap costs a 2 MiB buffer
/// and a 2 GiB one does not.
fn write_zeros(file: &mut fs::File, len: u64) -> Result<(), String> {
    if len == 0 {
        return Ok(());
    }
    let step = CHUNK.min(usize::try_from(len).unwrap_or(CHUNK)).max(1);
    let zeros = vec![0u8; step];
    let mut done = 0u64;
    while done < len {
        let want = usize::try_from(len - done).unwrap_or(usize::MAX).min(step);
        file.write_all(&zeros[..want]).map_err(|why| format!("{why}"))?;
        done += want as u64;
    }
    Ok(())
}

// ── the gate's writers ──────────────────────────────────────────────────────

/// **Write a serving artifact from host bytes** — the seed a gate synthesizes.
///
/// The device-free twin of [`store`], for the tests that have to assert what
/// the format promises on a machine with no GPU in it: the same header, the
/// same index, the same block discipline, a payload that came from a `Vec`
/// instead of from a card and a pinned tier. It writes the path it is given
/// rather than [`path`]'s, which is what lets a gate state a file whose name
/// and header disagree.
///
/// It goes through the same [`emit`] the real writer does, so a seed exercises
/// the arithmetic the write path uses.
///
/// Gate-only (`probe`), like every other hook in this crate that exists so a
/// test can state something a serving path never would.
///
/// # Errors
///
/// The filesystem's own words, or an entry list that does not tile the bytes
/// it was handed. Nothing here declines for space: a synthetic image is
/// kilobytes.
#[cfg(feature = "probe")]
pub fn seed(
    target: &Path,
    key: u64,
    entries: &[Group],
    flags: u32,
    payload: &[u8],
) -> Result<(), String> {
    let total: u64 = entries.iter().map(|group| group.reserved).sum();
    if total != payload.len() as u64 {
        return Err(format!(
            "the entries tile {total} bytes and the payload holds {}",
            payload.len()
        ));
    }
    emit(target, key, entries, flags, |param, at, into| {
        let group = entries
            .iter()
            .find(|group| group.id == param)
            .ok_or_else(|| format!("no image for param {param}"))?;
        let from = usize::try_from(group.offset + at).map_err(|_| "an offset past this host")?;
        let span = payload
            .get(from..from.saturating_add(into.len()))
            .ok_or_else(|| format!("the payload is short of {from}"))?;
        into.copy_from_slice(span);
        Ok(())
    })
}

/// **Overwrite one index entry in place** — the only way a gate can hold a
/// file whose index lies about its own payload.
///
/// [`super::restate_format`]'s discipline: it writes the entry and repairs the
/// index digest over it, and nothing else about the file moves. So the refusal
/// it provokes is a refusal about THAT ENTRY and not about a digest.
///
/// Gate-only.
///
/// # Errors
///
/// The filesystem's own words, or an index this file does not have that entry
/// in.
#[cfg(feature = "probe")]
pub fn restate_group(target: &Path, at: usize, group: Group) -> Result<(), String> {
    use std::io::Seek;
    use std::os::unix::fs::FileExt;

    let (head, mut entries, _) = read_head(target).map_err(|why| format!("{why}"))?;
    if at >= entries.len() {
        return Err(format!(
            "the index has {} entries and this asks for {at}",
            entries.len()
        ));
    }
    entries[at] = group;

    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(target)
        .map_err(|why| format!("{target:?}: {why}"))?;
    file.write_at(&group.encode(), head.index_at + at as u64 * ENTRY as u64)
        .map_err(|why| format!("{why}"))?;
    file.seek(std::io::SeekFrom::Start(Head::INDEX_DIGEST_AT))
        .map_err(|why| format!("{why}"))?;
    file.write_all(&index_digest(&entries).to_le_bytes())
        .map_err(|why| format!("{why}"))?;
    file.sync_all().map_err(|why| format!("{why}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(param: u32, at: u64, bytes: u64, reserved: u64) -> Group {
        Group {
            id: param,
            plane: 0,
            offset: at,
            bytes,
            reserved,
        }
    }

    /// **THE HEADER ROUND-TRIPS**, field for field, through the bytes it
    /// writes — with every field a different value, because a header of zeros
    /// would round-trip through an encoder that dropped half of them.
    #[test]
    fn a_tier_header_round_trips_through_the_bytes_it_writes() {
        let head = Head {
            format: TIER_FORMAT,
            flags: FLAG_ADAPTERS_ZEROED,
            key: 0x0123_4567_89ab_cdef,
            index_at: TIER_HEADER as u64,
            entries: 11,
            index_digest: 0x1111_2222_3333_4444,
            blocks_at: 0x2000,
            block_bytes: TIER_BLOCK,
            blocks: 17,
            digest: 0x5555_6666_7777_8888,
            payload_at: TIER_ALIGN,
            payload_total: 3 << 30,
        };
        let bytes = head.encode();
        assert_eq!(bytes.len(), TIER_HEADER, "the header is exactly its length");
        assert_eq!(bytes[..8], MAGIC, "and it starts with the magic");
        assert_eq!(Head::decode(&bytes), Some(head), "every field came back");
        assert_eq!(TIER_HEADER, 96, "format 3's header is 96 bytes, where 2's was 320");

        // The header is rewritten field-wise by `restate_group`, so the one
        // offset it names is part of the format and not a literal somebody has
        // to keep in step by hand.
        let mut patched = bytes;
        let at = Head::INDEX_DIGEST_AT as usize;
        patched[at..at + 8].copy_from_slice(&0u64.to_le_bytes());
        let read = Head::decode(&patched).expect("still an artifact");
        assert_eq!(read.index_digest, 0, "INDEX_DIGEST_AT names that field");
        assert_eq!(read.digest, head.digest, "and nothing else");

        // Anything that is not one answers `None` rather than a guess.
        assert_eq!(Head::decode(&bytes[..TIER_HEADER - 1]), None, "too short");
        let mut foreign = bytes;
        foreign[0] = b'X';
        assert_eq!(Head::decode(&foreign), None, "not the magic");
    }

    /// The two magics are different, which is what keeps either reader from
    /// parsing the other's file.
    #[test]
    fn a_tier_artifact_is_not_a_resident_artifact() {
        assert_ne!(MAGIC, super::super::MAGIC, "two formats, two magics");
        assert_ne!(
            path(Path::new("/cache"), 7),
            super::super::artifact_path(Path::new("/cache"), 7),
            "and the same key names two different files"
        );
    }

    /// **THE IMAGES TILE THE PAYLOAD**, and a list that does not is named for
    /// the entry that broke it.
    #[test]
    fn the_images_are_a_concatenation_and_a_gap_is_named() {
        let good = [image(0, 0, 100, 256), image(1, 256, 300, 512)];
        assert_eq!(payload_fault(&good, 768), None, "they tile");
        let gap = [image(0, 0, 100, 256), image(1, 512, 300, 512)];
        let why = payload_fault(&gap, 1024).expect("a hole in the payload");
        assert!(why.contains("param 1"), "the image is named: {why}");
        let over = [image(0, 0, 300, 256)];
        let why = payload_fault(&over, 256).expect("more published than reserved");
        assert!(why.contains("300"), "both numbers: {why}");
        let short = [image(0, 0, 100, 256)];
        let why = payload_fault(&short, 512).expect("a payload longer than its images");
        assert!(why.contains("512"), "the total is named: {why}");
    }

    /// **THE BLOCKS TILE ONE IMAGE EXACTLY**, at every span an image can have
    /// — which is what [`read_spans_into`]'s "every byte is written" rests on,
    /// and what makes a block's byte range in a refusal an address a reader
    /// can act on.
    #[test]
    fn the_blocks_of_an_image_tile_it_at_any_length() {
        let head = Head {
            format: TIER_FORMAT,
            flags: 0,
            key: 0,
            index_at: TIER_HEADER as u64,
            entries: 0,
            index_digest: 0,
            blocks_at: TIER_HEADER as u64,
            block_bytes: TIER_BLOCK,
            blocks: 0,
            digest: 0,
            payload_at: TIER_ALIGN,
            payload_total: 0,
        };
        for span in [
            0,
            1,
            255,
            TIER_BLOCK - 1,
            TIER_BLOCK,
            TIER_BLOCK + 1,
            TIER_BLOCK * 8,
            TIER_BLOCK * 8 + 4097,
        ] {
            let count = head.blocks_of(span);
            assert_eq!(count, span.div_ceil(TIER_BLOCK), "{span} bytes");
            let mut at = 0u64;
            for block in 0..count {
                let (from, len) = head.block_span(1 << 21, span, block);
                assert_eq!(from, (1 << 21) + at, "{span} bytes: block {block} starts late");
                assert!(len > 0 && len <= TIER_BLOCK, "{span} bytes: block {block}");
                at += len;
            }
            assert_eq!(at, span, "{span} bytes: the blocks end where the image does");
        }
    }

    /// The name a path carries is read back, and only from a name this module
    /// would have written.
    #[test]
    fn a_filename_states_the_key_it_holds() {
        let dir = Path::new("/cache");
        assert_eq!(key_in(&path(dir, 0xdead_beef)), Some(0xdead_beef));
        assert_eq!(key_in(Path::new("/cache/0000000000000001.weights")), None);
        assert_eq!(key_in(Path::new("/cache/mine.tiers")), None);
        assert_eq!(key_in(Path::new("/cache/notahex000000zz.tiers")), None);
    }

    /// **THE READER FILLS EVERY IMAGE WHOLE AND AGREES WITH THE WRITER'S
    /// DIGESTS** — the two claims [`read_spans_into`] is trusted for, and the
    /// pair that lets `Tier::open` skip its memset.
    ///
    /// The destinations are pre-filled with `0xFF`, which neither image
    /// contains — their bytes are counters modulo a prime under 255 — so
    /// "every byte was written" is asserted rather than assumed: a reader that
    /// skipped a block, a padding tail or a whole image would leave that byte
    /// behind.
    #[test]
    fn every_image_is_read_back_whole_and_checked_against_what_was_written() {
        let dir = std::env::temp_dir().join(format!(
            "pie-tier-read-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&dir).expect("a temporary directory");
        let target = path(&dir, 0x51ce);

        // Not a multiple of anything: the padding is real and the last block
        // of each image is short.
        let one: Vec<u8> = (0..3_333u32).map(|at| (at % 251) as u8).collect();
        let two: Vec<u8> = (0..40_009u32).map(|at| (at % 253) as u8).collect();
        let entries = [
            image(0, 0, 3_333, 3_456),
            image(1, 3_456, 40_009, 40_192),
        ];
        let sources = [&one[..], &two[..]];
        emit(&target, 0x51ce, &entries, FLAG_ADAPTERS_ZEROED, |param, at, into| {
            let from = usize::try_from(at).expect("an offset");
            into.copy_from_slice(&sources[param as usize][from..from + into.len()]);
            Ok(())
        })
        .expect("the synthetic artifact writes");

        let artifact = Artifact::open(&target).expect("and reads back");
        artifact.verify().expect("the whole payload hashes to its table");
        for (param, source) in [(0u32, &one), (1u32, &two)] {
            assert_eq!(
                artifact.plane(param).expect("the image resolves"),
                &source[..],
                "param {param} came back as something else"
            );
        }

        let head = artifact.head();
        let blocks = artifact.blocks().to_vec();
        let mut into: Vec<Vec<u8>> = entries
            .iter()
            .map(|group| vec![0xFFu8; usize::try_from(group.reserved).expect("a span")])
            .collect();
        let spans: Vec<Span> = entries
            .iter()
            .zip(into.iter_mut())
            .enumerate()
            .map(|(at, (group, dst))| Span {
                at: group.offset,
                len: group.reserved,
                first_block: artifact.first[at] as u64,
                into: dst.as_mut_ptr(),
            })
            .collect();
        // SAFETY: each destination is a live allocation of exactly its
        // image's span, they are separate `Vec`s, and nothing else names them.
        unsafe { read_spans_into(&target, &head, &blocks, &spans) }
            .expect("the images read back");
        for (at, (group, dst)) in entries.iter().zip(&into).enumerate() {
            let published = usize::try_from(group.bytes).expect("published bytes");
            assert_eq!(&dst[..published], sources[at], "image {at} came back wrong");
            assert!(
                dst[published..].iter().all(|byte| *byte == 0),
                "image {at}'s padding is the zeros the writer wrote"
            );
        }

        // ── AND ONE FLIPPED BYTE IS CAUGHT, NAMED WITH ITS BLOCK.
        {
            use std::os::unix::fs::FileExt;

            let at = head.payload_at + entries[1].offset + 17;
            let file = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&target)
                .expect("the artifact is writable");
            let mut byte = [0u8; 1];
            file.read_exact_at(&mut byte, at).expect("the byte reads");
            byte[0] ^= 0x01;
            file.write_all_at(&byte, at).expect("the byte writes");
        }
        let before = observed();
        let mut dst = vec![0xFFu8; usize::try_from(entries[1].reserved).expect("a span")];
        let spans = [Span {
            at: entries[1].offset,
            len: entries[1].reserved,
            first_block: artifact.first[1] as u64,
            into: dst.as_mut_ptr(),
        }];
        // SAFETY: as above.
        let why = unsafe { read_spans_into(&target, &head, &blocks, &spans) }
            .expect_err("a flipped byte is not the byte that was written");
        let why = format!("{why}");
        assert!(why.contains("block"), "the block is named: {why}");
        assert!(
            observed().corrupt > before.corrupt,
            "the corruption is counted at the door that found it"
        );
        // The bytes are still all there — the destination is filled whether or
        // not the digest agreed, which is what makes the recovery a ZEROING
        // and not a repair.
        assert!(!dst.contains(&0xFF), "and nothing was left unwritten");

        // The mapping's reader finds the same thing, and names the image.
        let artifact = Artifact::open(&target).expect("it still parses");
        let why = format!("{}", artifact.verify_entries(&[1]).expect_err("still rotten"));
        assert!(why.contains("param 1"), "the image is named: {why}");
        artifact
            .verify_entries(&[0])
            .expect("and the OTHER image is untouched, which is the whole point");

        let _ = fs::remove_dir_all(&dir);
    }

    /// **A FORMAT-2 FILE IS REFUSED BY VERSION, NOT BY LENGTH — AND SURVIVES
    /// THE REFUSAL** (§M.4).
    ///
    /// Format 2's header was 320 bytes, this one's is 96, and a reader that
    /// insisted on its own length before reading the version would answer
    /// `NotAnArtifact` for exactly the file an operator needs named.
    #[test]
    fn an_older_header_alone_is_refused_by_its_version_and_left_alone() {
        let dir = std::env::temp_dir().join(format!(
            "pie-tier-format-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&dir).expect("a temporary directory");

        // Format 2's whole header, by hand: the magic, the version, and 308
        // bytes this build has no offsets for.
        let mut old = vec![0u8; 320];
        old[..8].copy_from_slice(&MAGIC);
        old[8..12].copy_from_slice(&2u32.to_le_bytes());
        let target = dir.join("format-two.tiers");
        fs::write(&target, &old).expect("a format-2 header");
        assert!(
            matches!(
                read_head(&target),
                Err(Refused::StaleFormat { states: 2, reads: 3 })
            ),
            "a header of another length is stale, not a stranger: {:?}",
            read_head(&target)
        );
        // `refuse` BUILDS the sentence since §M-3 — a serve puts it in a
        // fault and a prepare prints it — so what is asserted is the sentence
        // and the file, and neither of them is a side effect.
        let said = refuse(
            &target,
            Some(Path::new("/models/some.zt")),
            "states a format this build does not read",
        );
        assert!(
            said.contains("pie model import --prepare-only /models/some.zt"),
            "the remedy is spelled against the checkpoint it was given: {said}"
        );
        assert!(target.exists(), "and NOTHING in this module deletes it");
        assert!(
            refuse(&target, None, "rotted").contains("<this deployment's checkpoint>"),
            "and the one caller with no checkpoint in hand gets a slot, not a guess"
        );

        // Twelve bytes is the least that carries a version; eleven is not a
        // claim about a format at all.
        fs::write(&target, &old[..12]).expect("twelve bytes");
        assert!(matches!(
            read_head(&target),
            Err(Refused::StaleFormat { states: 2, reads: 3 })
        ));
        fs::write(&target, &old[..11]).expect("eleven bytes");
        assert!(
            matches!(read_head(&target), Err(Refused::NotAnArtifact)),
            "and a file too short to state one is no artifact"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}

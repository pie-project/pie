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
//! pipeline ([`model_loader::executor::Execution`]) never runs.
//!
//! # Why the key is a hash of the recipe and not a name
//!
//! dev's rule, kept: **a false miss costs one re-materialization; a false hit
//! puts silently wrong weights on the device.** So the key errs toward
//! missing. It mixes the whole [`LoadPlan`](model_loader::plan::LoadPlan)
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
//! # What is NOT here
//!
//! dev's `staged_h2d.hpp` — the four-lane pinned double-buffered H2D whose
//! measured argument is that one staging lane already outruns NVMe by 1.6×,
//! so GDS buys nothing under ~7 GB/s per reader. The restore path here uses
//! the shell's ordinary blocking [`Buffer::write`], which is correct and
//! slower; the lanes are a throughput refinement with a measurement attached
//! and they belong with the elastic supply, not with the contract.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::alloc::Buffer;
use crate::error::Result;

/// The artifact format's own version.
///
/// **A DIFFERENT ONE IS A MISS, NOT AN ERROR** (dev's `SCHEMA_VERSION`): the
/// shell recomputes and overwrites. Bump it whenever the bytes on disk mean
/// something different — a header field, a layout rule, an alignment.
const FORMAT: u32 = 1;

/// What every artifact starts with, so a file that is not one is a miss
/// rather than a parse.
const MAGIC: [u8; 8] = *b"PIEWCAC1";

/// Header bytes: magic, format, key, blob length, digest, reserved.
const HEADER: usize = 8 + 4 + 8 + 8 + 8 + 4;

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
/// **PROCESS-GLOBAL AND NOT A FIELD**, for the reason `record::FOLD_OBSERVED`
/// is: a gate at the runtime level holds the engine behind a `Box<dyn Engine>`
/// on a lane thread and cannot ask the instance anything. The per-load answer
/// a CALLER wants rides home on
/// [`LoadFacts::weights_from_cache`](engine::engine_api::load::LoadFacts);
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

/// Where this key's artifact lives under `dir`.
fn artifact_path(dir: &Path, key: u64) -> PathBuf {
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
    let Ok(mut file) = fs::File::open(path) else {
        return Ok(false);
    };
    let mut header = [0u8; HEADER];
    if file.read_exact(&mut header).is_err() {
        return Ok(false);
    }
    if header[..8] != MAGIC {
        return Ok(false);
    }
    let format = u32::from_le_bytes(header[8..12].try_into().unwrap_or([0; 4]));
    let stated_key = u64::from_le_bytes(header[12..20].try_into().unwrap_or([0; 8]));
    let total = u64::from_le_bytes(header[20..28].try_into().unwrap_or([0; 8]));
    let digest = u64::from_le_bytes(header[28..36].try_into().unwrap_or([0; 8]));
    // A stale format or a key from another deployment is a MISS. Neither is
    // a fault: the shell recomputes and overwrites.
    if format != FORMAT || stated_key != key {
        return Ok(false);
    }
    if total != store.bytes() as u64 {
        // The layout is part of the key, so this cannot happen without the
        // key having been reused for a different table. Treat it as
        // corruption rather than as a miss: a file that lies about its own
        // size is the case the digest exists to catch.
        return Err(format!(
            "states {total} bytes for a table of {}",
            store.bytes()
        ));
    }

    let mut hash = Fnv::default();
    let mut at = 0u64;
    let mut chunk = vec![0u8; CHUNK.min(store.bytes().max(1))];
    while at < total {
        let want = usize::try_from(total - at).unwrap_or(usize::MAX).min(chunk.len());
        let slice = &mut chunk[..want];
        file.read_exact(slice)
            .map_err(|why| format!("truncated at byte {at}: {why}"))?;
        hash.raw(slice);
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
    if hash.finish() != digest {
        return Err(format!(
            "digest {:016x} does not match its {total} bytes",
            digest
        ));
    }
    Ok(true)
}

/// **Snapshot `store` beside `key`, if there is room.**
///
/// Best-effort in every direction: no directory means the feature is off, a
/// filesystem with less than the blob plus [`MARGIN`] free means a declined
/// write with both figures named, and any failure at all means a declined
/// write. None of them fails the load, because the load already succeeded —
/// the store this is reading from is the answer.
pub fn store(dir: Option<&Path>, key: u64, store: &Buffer) {
    let Some(dir) = dir else {
        return;
    };
    if let Err(why) = write_out(dir, key, store) {
        bump(Stat::Declined);
        eprintln!("engine-cuda: declined to cache this load's weights: {why}");
        return;
    }
    bump(Stat::Stored);
}

fn write_out(dir: &Path, key: u64, store: &Buffer) -> std::result::Result<(), String> {
    fs::create_dir_all(dir).map_err(|why| format!("{dir:?}: {why}"))?;
    let total = store.bytes() as u64;
    let need = total.saturating_add(MARGIN);
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
        let mut header = [0u8; HEADER];
        header[..8].copy_from_slice(&MAGIC);
        header[8..12].copy_from_slice(&FORMAT.to_le_bytes());
        header[12..20].copy_from_slice(&key.to_le_bytes());
        header[20..28].copy_from_slice(&total.to_le_bytes());
        // The digest is written last, over the header's placeholder, once the
        // blob has been read; a rewind is cheaper than holding the store.
        file.write_all(&header).map_err(|why| format!("{why}"))?;

        let mut hash = Fnv::default();
        let mut at = 0u64;
        let mut chunk = vec![0u8; CHUNK.min(store.bytes().max(1))];
        while at < total {
            let want = usize::try_from(total - at).unwrap_or(usize::MAX).min(chunk.len());
            let slice = &mut chunk[..want];
            store
                .read(at, slice)
                .map_err(|why| format!("reading the store at {at}: {why}"))?;
            hash.raw(slice);
            file.write_all(slice).map_err(|why| format!("{why}"))?;
            at += want as u64;
        }
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(28))
            .map_err(|why| format!("{why}"))?;
        file.write_all(&hash.finish().to_le_bytes())
            .map_err(|why| format!("{why}"))?;
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
/// # Errors
///
/// A device failure reading the store back.
pub fn digest_of(store: &Buffer) -> Result<u64> {
    let mut hash = Fnv::default();
    let total = store.bytes() as u64;
    let mut chunk = vec![0u8; CHUNK.min(store.bytes().max(1))];
    let mut at = 0u64;
    while at < total {
        let want = usize::try_from(total - at).unwrap_or(usize::MAX).min(chunk.len());
        let slice = &mut chunk[..want];
        store.read(at, slice)?;
        hash.raw(slice);
        at += want as u64;
    }
    Ok(hash.finish())
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
}

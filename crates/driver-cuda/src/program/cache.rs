//! The cubin disk cache: a compile that survives the process.
//!
//! # Why this exists at all
//!
//! NVRTC compilation of one fused region is hundreds of milliseconds, a
//! program has one region per fusable stage, and a serving process is
//! restarted often — for a config change, for a deploy, for a crash. Without a
//! disk tier every restart pays the whole compile again before it can answer
//! its first token.
//!
//! # The key, and the bug it exists to prevent
//!
//! The in-memory tiers key on [`cache_identity`](driver::cache_identity):
//! backend, device, signature, and four version numbers. The DISK key is that
//! string plus **a fingerprint of the emitted source bytes**, and the extra
//! eight bytes are the whole reason this file is worth reading.
//!
//! Every field of the identity is a number someone has to remember to bump.
//! Editing `tensor-compiler`'s device templates — `fused_block0.cuh`, the
//! runtime body, the emitter's own text — changes none of them. So the cubin
//! compiled from yesterday's source stayed on disk, matched today's key, and
//! was loaded instead of the new one: **kernel edits appeared to do nothing**,
//! and the model answered fluently out of the previous kernel. The C++ records
//! this at `module_cache.hpp:299-305` in the past tense because it happened.
//!
//! Keying the file on the source that produced it removes the remembering.
//!
//! # The format, byte for byte
//!
//! ```text
//! "PTRCUB01"        8 bytes, magic
//! region_index      u32 le
//! key_size          u32 le
//! entry_size        u32 le
//! cubin_size        u64 le
//! key               key_size bytes
//! entry             entry_size bytes
//! cubin             cubin_size bytes
//! ```
//!
//! The key and the entry name are stored in the file and compared on load even
//! though the filename is derived from the key. That is not redundant: the
//! filename is a 64-bit hash, hashes collide, and a collision here would load
//! one program's machine code for another program's launch. The stored copy
//! turns that from a wrong answer into a miss.
//!
//! # Every failure here is a miss, never an error
//!
//! A cache is an optimisation. An unreadable entry, an unwritable directory, a
//! full disk — none of them are reasons to fail a registration, because the
//! compiler is right there. So every function returns `Option` or `()`, and a
//! corrupt entry is additionally *removed* on the way past, so the next run
//! does not pay the same read to reach the same conclusion.

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// The file magic. `01` is a format version: a change to the layout above
/// bumps it, and every older entry becomes a miss rather than a misparse.
const MAGIC: &[u8; 8] = b"PTRCUB01";

/// Header bytes before the variable-length tail: magic + three `u32` + one `u64`.
const HEADER_BYTES: usize = 8 + 4 + 4 + 4 + 8;

/// The largest entry that will be read.
///
/// A cubin is tens to hundreds of kilobytes. Anything past this is a corrupt
/// header claiming a huge length, and the bound is what stops that from
/// becoming an allocation the size of the claim.
const MAX_ENTRY_BYTES: u64 = 128 * 1024 * 1024;

/// Serialises the temp-file names of concurrent writers in one process.
static NONCE: AtomicU64 = AtomicU64::new(0);

/// Where cubins are kept, or `None` when nowhere is writable.
///
/// A cache that cannot find a home is not an error: the driver compiles every
/// time, which is what it would do without this file at all.
#[derive(Clone, Debug)]
pub struct Disk {
    directory: Option<PathBuf>,
}

impl Disk {
    /// Resolve the cache directory from the environment.
    ///
    /// `$PIE_HOME/cache/ptir-cuda`, else `$XDG_CACHE_HOME/pie/ptir-cuda`, else
    /// `$HOME/.cache/pie/ptir-cuda`, else nowhere. The order is the C++'s and
    /// the reason for it is that `PIE_HOME` is the one a deployment sets on
    /// purpose.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            directory: default_directory(),
        }
    }

    /// A cache rooted at an explicit directory. For tests, and for a caller
    /// that has already resolved a home.
    #[must_use]
    pub fn at(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: Some(directory.into()),
        }
    }

    /// A cache that stores nothing. Every load misses and every store is a
    /// no-op — the behaviour when no directory is writable, made explicit so a
    /// caller can ask for it.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { directory: None }
    }

    /// Whether anything will actually be written.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.directory.is_some()
    }

    /// The cubin stored for `(key, region_index, entry)`, if one is and it
    /// still matches.
    ///
    /// A mismatch or a malformed entry is removed before returning `None`, so
    /// a corrupt file is paid for once rather than every launch.
    #[must_use]
    pub fn load(&self, key: &str, region_index: u32, entry: &str) -> Option<Vec<u8>> {
        let path = self.path(key, region_index)?;
        let bytes = fs::read(&path).ok()?;
        match parse(&bytes, key, region_index, entry) {
            Some(cubin) => Some(cubin),
            None => {
                self.invalidate(key, region_index);
                None
            }
        }
    }

    /// Store `cubin` for `(key, region_index, entry)`.
    ///
    /// Written to a per-writer temp file and `rename`d into place, because two
    /// processes calibrating the same program under one `$HOME` is the normal
    /// case rather than the exotic one, and a half-written cubin that another
    /// process loads is a segfault inside the driver. `rename` within a
    /// directory is atomic on every filesystem this runs on.
    pub fn store(&self, key: &str, region_index: u32, entry: &str, cubin: &[u8]) {
        let Some(directory) = self.directory.as_ref() else {
            return;
        };
        if u32::try_from(key.len()).is_err() || u32::try_from(entry.len()).is_err() {
            return;
        }
        if fs::create_dir_all(directory).is_err() {
            return;
        }
        let Some(destination) = self.path(key, region_index) else {
            return;
        };

        let mut bytes = Vec::with_capacity(HEADER_BYTES + key.len() + entry.len() + cubin.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&region_index.to_le_bytes());
        bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(entry.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(cubin.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(entry.as_bytes());
        bytes.extend_from_slice(cubin);

        let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
        let temporary =
            destination.with_extension(format!("cubin.tmp-{}-{nonce}", std::process::id()));
        // The whole write, then the rename. A failure at any point removes the
        // temp file and gives up: this is a cache, and the caller has the
        // compiler.
        let written = fs::File::create(&temporary).and_then(|mut file| {
            file.write_all(&bytes)?;
            file.sync_all()
        });
        if written.is_err() || fs::rename(&temporary, &destination).is_err() {
            let _ = fs::remove_file(&temporary);
        }
    }

    /// Remove whatever is stored for `(key, region_index)`.
    pub fn invalidate(&self, key: &str, region_index: u32) {
        if let Some(path) = self.path(key, region_index) {
            let _ = fs::remove_file(path);
        }
    }

    /// The file a `(key, region_index)` pair maps to.
    fn path(&self, key: &str, region_index: u32) -> Option<PathBuf> {
        let directory = self.directory.as_ref()?;
        Some(directory.join(format!(
            "{:016x}-{region_index}.cubin",
            fnv1a64(key.as_bytes())
        )))
    }
}

/// The eight bytes of source fingerprint the disk key carries beyond the
/// in-memory one.
///
/// Appended to the identity string rather than folded into it, so a reader
/// looking at a key can still see the identity that produced it.
#[must_use]
pub fn disk_key(identity: &str, source: &str) -> String {
    let hash = fnv1a64(source.as_bytes());
    let mut key = String::with_capacity(identity.len() + 16);
    key.push_str(identity);
    for byte in hash.to_le_bytes() {
        // Hex rather than the C++'s raw bytes: the key is stored in a file and
        // compared as bytes either way, and a key that is printable is one a
        // human can read off a stale entry. Nothing else depends on its
        // spelling -- both sides of the comparison are produced here.
        use std::fmt::Write as _;
        let _ = write!(key, "{byte:02x}");
    }
    key
}

/// Validate a stored entry and return its cubin.
///
/// Every field is checked against what was asked for. The lengths are checked
/// against the file's own size *before* the key and entry are compared, so a
/// header claiming lengths the file does not hold is a miss rather than a
/// panic on a slice.
fn parse(bytes: &[u8], key: &str, region_index: u32, entry: &str) -> Option<Vec<u8>> {
    if bytes.len() < HEADER_BYTES || bytes.len() as u64 > MAX_ENTRY_BYTES {
        return None;
    }
    if &bytes[..8] != MAGIC {
        return None;
    }
    let stored_region = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    let key_size = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    let entry_size = u32::from_le_bytes(bytes[16..20].try_into().ok()?) as usize;
    let cubin_size = u64::from_le_bytes(bytes[20..28].try_into().ok()?);

    if stored_region != region_index || key_size != key.len() || entry_size != entry.len() {
        return None;
    }
    // The tail must be exactly the three pieces the header describes. Not
    // "at least": a longer tail means the header and the file disagree, and
    // the header is the thing that would be believed on the next read.
    let tail = bytes.len().checked_sub(HEADER_BYTES)?;
    let claimed = (key_size as u64)
        .checked_add(entry_size as u64)?
        .checked_add(cubin_size)?;
    if tail as u64 != claimed {
        return None;
    }

    let key_at = HEADER_BYTES;
    let entry_at = key_at + key_size;
    let cubin_at = entry_at + entry_size;
    if &bytes[key_at..entry_at] != key.as_bytes() || &bytes[entry_at..cubin_at] != entry.as_bytes()
    {
        return None;
    }
    Some(bytes[cubin_at..].to_vec())
}

/// `$PIE_HOME/cache/ptir-cuda`, else the XDG cache, else `~/.cache`.
fn default_directory() -> Option<PathBuf> {
    let non_empty = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|value| !value.as_os_str().is_empty())
    };
    if let Some(home) = non_empty("PIE_HOME") {
        return Some(home.join("cache").join("ptir-cuda"));
    }
    if let Some(cache) = non_empty("XDG_CACHE_HOME") {
        return Some(cache.join("pie").join("ptir-cuda"));
    }
    non_empty("HOME").map(|home| home.join(".cache").join("pie").join("ptir-cuda"))
}

/// FNV-1a over bytes.
///
/// The same function `tensor_ir::fnv1a64` is, spelled here so this module does
/// not pull the IR crate in for one fold. The constants are the algorithm's.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("pie-ptir-disk-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        path
    }

    /// The fold must match `tensor_ir::fnv1a64`, because the identity string
    /// this key extends was produced with it.
    #[test]
    fn the_fold_is_the_same_fnv1a_the_rest_of_the_workspace_uses() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), driver::tensor_ir::fnv1a64(b"a"));
        assert_eq!(fnv1a64(b"ptir"), driver::tensor_ir::fnv1a64(b"ptir"));
    }

    /// The whole point of the disk key: an edit to the emitted source must
    /// miss, even though every version number and the identity are unchanged.
    /// This is the regression the C++ comment records in the past tense.
    #[test]
    fn editing_the_source_changes_the_disk_key_with_no_version_bump() {
        let identity = "0100000000000000000300000000000000000000-v0003000400000003 00000015";
        let before = disk_key(identity, "__global__ void k() { a(); }");
        let after = disk_key(identity, "__global__ void k() { b(); }");
        assert_ne!(
            before, after,
            "a template edit bumps no version, so the source itself must be in \
             the key -- otherwise yesterday's cubin answers today's launch"
        );
        assert!(before.starts_with(identity), "the identity stays readable");
    }

    /// A stored cubin comes back exactly.
    #[test]
    fn a_stored_cubin_round_trips() {
        let disk = Disk::at(scratch("roundtrip"));
        let cubin = vec![0xdeu8, 0xad, 0xbe, 0xef, 0x00, 0x01];
        disk.store("key-a", 2, "ptir_fused_abc_r2", &cubin);
        assert_eq!(disk.load("key-a", 2, "ptir_fused_abc_r2"), Some(cubin));
    }

    /// Every field of the request is part of the identity, and a request that
    /// differs in any one of them must not be answered by this entry. The
    /// filename covers only the key, so the region and the entry name are
    /// checked from the file's own contents.
    #[test]
    fn an_entry_answers_only_the_exact_request_it_was_stored_for() {
        let disk = Disk::at(scratch("exact"));
        disk.store("key-a", 2, "entry_r2", b"cubin");
        assert_eq!(disk.load("key-a", 2, "entry_r2"), Some(b"cubin".to_vec()));
        assert_eq!(disk.load("key-a", 3, "entry_r2"), None, "wrong region");
        assert_eq!(disk.load("key-a", 2, "entry_r9"), None, "wrong entry name");
        assert_eq!(disk.load("key-b", 2, "entry_r2"), None, "wrong key");
    }

    /// A truncated or corrupt file is a miss AND is removed, so the next run
    /// does not read it again to reach the same conclusion.
    #[test]
    fn a_corrupt_entry_is_a_miss_and_is_deleted() {
        let directory = scratch("corrupt");
        let disk = Disk::at(&directory);
        disk.store("key-a", 0, "entry", b"cubin-bytes");
        let path = disk.path("key-a", 0).expect("enabled");
        let good = fs::read(&path).expect("stored");

        fs::write(&path, &good[..good.len() - 3]).expect("truncate");
        assert_eq!(
            disk.load("key-a", 0, "entry"),
            None,
            "a short tail is a miss"
        );
        assert!(!path.exists(), "and the entry is removed");

        disk.store("key-a", 0, "entry", b"cubin-bytes");
        let mut wrong_magic = fs::read(&path).expect("stored");
        wrong_magic[7] = b'9';
        fs::write(&path, &wrong_magic).expect("write");
        assert_eq!(
            disk.load("key-a", 0, "entry"),
            None,
            "a format bump is a miss"
        );
        assert!(!path.exists());
    }

    /// A header claiming a length the file does not hold must not panic on a
    /// slice, and must not allocate the claim.
    #[test]
    fn a_header_that_lies_about_its_lengths_is_refused_without_panicking() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(b"key-a");
        bytes.extend_from_slice(b"entry");
        assert_eq!(parse(&bytes, "key-a", 0, "entry"), None);
    }

    /// A cache with no home stores nothing and misses everything, rather than
    /// failing. That is the behaviour when no environment variable is set, and
    /// it must not be a registration error.
    #[test]
    fn a_disabled_cache_is_a_miss_and_not_a_failure() {
        let disk = Disk::disabled();
        assert!(!disk.is_enabled());
        disk.store("key", 0, "entry", b"cubin");
        assert_eq!(disk.load("key", 0, "entry"), None);
    }
}

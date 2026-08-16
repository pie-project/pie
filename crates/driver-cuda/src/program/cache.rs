//! The cubin disk cache: a compile that survives the process.
//!
//! The disk key is [`cache_identity`](driver::cache_identity) plus a
//! fingerprint of the emitted source: editing `tensor-compiler`'s device
//! templates bumps no version number, so without the fingerprint a stale cubin
//! matches today's key and kernel edits silently do nothing. Every failure here
//! is a miss, never an error — the compiler is always available, and a corrupt
//! entry is removed on the way past so it is paid for once, not every run.

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// File magic; the `01` is a format version, bumped when the layout changes so
/// older entries miss rather than misparse.
const MAGIC: &[u8; 8] = b"PTRCUB01";

/// Header bytes before the variable-length tail: magic + three `u32` + one `u64`.
const HEADER_BYTES: usize = 8 + 4 + 4 + 4 + 8;

/// The largest entry that will be read: a corrupt header claiming a huge length
/// must not become an allocation the size of the claim.
const MAX_ENTRY_BYTES: u64 = 128 * 1024 * 1024;

/// Serialises the temp-file names of concurrent writers in one process.
static NONCE: AtomicU64 = AtomicU64::new(0);

/// Where cubins are kept, or `None` when nowhere is writable — not an error,
/// the driver just compiles every time.
#[derive(Clone, Debug)]
pub struct Disk {
    directory: Option<PathBuf>,
}

impl Disk {
    /// Resolve the cache directory: `$PIE_HOME/cache/ptir-cuda`, else
    /// `$XDG_CACHE_HOME/pie/ptir-cuda`, else `$HOME/.cache/pie/ptir-cuda`, else
    /// nowhere. `PIE_HOME` wins because a deployment sets it on purpose.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            directory: default_directory(),
        }
    }

    /// A cache rooted at an explicit directory.
    #[must_use]
    pub fn at(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: Some(directory.into()),
        }
    }

    /// A cache that stores nothing: every load misses, every store is a no-op.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { directory: None }
    }

    /// Whether anything will actually be written.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.directory.is_some()
    }

    /// The cubin stored for `(key, region_index, entry)`, if it still matches.
    /// A mismatch or malformed entry is removed before returning `None`.
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
    /// Written to a per-writer temp file and atomically `rename`d in: concurrent
    /// writers are normal, and a half-written cubin another process loads is a
    /// segfault in the driver.
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
        // Write in full, then rename; any failure removes the temp file and
        // gives up — this is a cache.
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

/// The identity string plus an eight-byte fingerprint of the source, appended
/// (not folded in) so the identity stays readable in a key.
#[must_use]
pub fn disk_key(identity: &str, source: &str) -> String {
    let hash = fnv1a64(source.as_bytes());
    let mut key = String::with_capacity(identity.len() + 16);
    key.push_str(identity);
    for byte in hash.to_le_bytes() {
        // Hex so a stored key stays human-readable; both sides of the
        // comparison are produced here, so the spelling is free to choose.
        use std::fmt::Write as _;
        let _ = write!(key, "{byte:02x}");
    }
    key
}

/// Validate a stored entry and return its cubin.
///
/// The filename is only a 64-bit hash of the key, so the key and entry are
/// stored and compared here too: a hash collision would otherwise load one
/// program's machine code for another's launch. Lengths are checked against
/// the file's own size before any slice, so a lying header is a miss, not a panic.
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
    // The tail must be exactly the three pieces the header describes; a longer
    // tail means header and file disagree.
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

/// FNV-1a over bytes; the same as `tensor_ir::fnv1a64`, inlined so this module
/// does not pull in the IR crate for one fold.
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

    /// The fold must match `tensor_ir::fnv1a64`, which produced the identity.
    #[test]
    fn the_fold_is_the_same_fnv1a_the_rest_of_the_workspace_uses() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), driver::tensor_ir::fnv1a64(b"a"));
        assert_eq!(fnv1a64(b"ptir"), driver::tensor_ir::fnv1a64(b"ptir"));
    }

    /// The point of the disk key: a source edit must miss even when every
    /// version number and the identity are unchanged.
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

    /// A request differing in region, entry, or key must not be answered by
    /// this entry, since the filename covers only the key.
    #[test]
    fn an_entry_answers_only_the_exact_request_it_was_stored_for() {
        let disk = Disk::at(scratch("exact"));
        disk.store("key-a", 2, "entry_r2", b"cubin");
        assert_eq!(disk.load("key-a", 2, "entry_r2"), Some(b"cubin".to_vec()));
        assert_eq!(disk.load("key-a", 3, "entry_r2"), None, "wrong region");
        assert_eq!(disk.load("key-a", 2, "entry_r9"), None, "wrong entry name");
        assert_eq!(disk.load("key-b", 2, "entry_r2"), None, "wrong key");
    }

    /// A truncated or corrupt file is a miss and is removed.
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

    /// A header lying about its lengths is refused without panicking or allocating.
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

    /// A cache with no home stores nothing and misses everything, never fails.
    #[test]
    fn a_disabled_cache_is_a_miss_and_not_a_failure() {
        let disk = Disk::disabled();
        assert!(!disk.is_enabled());
        disk.store("key", 0, "entry", b"cubin");
        assert_eq!(disk.load("key", 0, "entry"), None);
    }
}

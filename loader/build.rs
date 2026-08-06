use std::env;
use std::path::{Path, PathBuf};

/// Content-hash every `.rs` file under `dir` (recursively, name-sorted for
/// determinism) into an FNV-1a 64-bit value. Fingerprints the planner's compiler
/// logic so the on-disk LoadPlan cache auto-invalidates when that logic
/// changes — no manual `cache-vN` bump. Content-based (not mtime), so a no-op
/// rebuild keeps the same hash and the cache stays warm.
///
/// Every failure here is fatal. A skipped file hashes to the same value as a
/// tree that never had it, so swallowing an I/O error hands out a fingerprint
/// that says "unchanged" about a compiler that did change — and the cache
/// serves a stale plan. Failing the build is the only outcome that keeps the
/// number meaning what it claims.
fn hash_sources(dir: &Path) -> u64 {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_rs(dir, &mut files);
    files.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in &files {
        let bytes = std::fs::read(f)
            .unwrap_or_else(|err| panic!("hashing compiler sources: cannot read {f:?}: {err}"));
        for b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("hashing compiler sources: cannot list {dir:?}: {err}"));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| panic!("hashing compiler sources: cannot walk {dir:?}: {err}"))
            .path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let compiler_hash = hash_sources(&manifest_dir.join("src"));
    println!("cargo:rustc-env=PIE_LOADER_COMPILER_HASH={compiler_hash}");

    println!("cargo:rerun-if-changed=src");
}

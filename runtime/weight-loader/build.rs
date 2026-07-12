use std::env;
use std::path::{Path, PathBuf};

/// Content-hash every `.rs` file under `dir` (recursively, name-sorted for
/// determinism) into an FNV-1a 64-bit value. Fingerprints the loader's compiler
/// logic so the on-disk storage-program cache auto-invalidates when that logic
/// changes — no manual `cache-vN` bump. Content-based (not mtime), so a no-op
/// rebuild keeps the same hash and the cache stays warm.
fn hash_sources(dir: &Path) -> u64 {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_rs(dir, &mut files);
    files.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in &files {
        if let Ok(bytes) = std::fs::read(f) {
            for b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    h
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
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
    println!("cargo:rustc-env=PIE_WL_COMPILER_HASH={compiler_hash}");

    println!("cargo:rerun-if-changed=src");
}

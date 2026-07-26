use std::env;
use std::path::{Path, PathBuf};

/// Content-hash every `.rs` file under `dir` (recursively, name-sorted for
/// determinism) into an FNV-1a 64-bit value. Fingerprints the tracer: the
/// traced form is a pure function of (declaration code, facts), so this hash
/// plus the facts identifies a plan exactly, and anything caching or
/// comparing traced forms across builds can key on
/// `PieForwardPlan::compiler_version` without a manual version bump. Copied
/// from `loader/build.rs`, which fingerprints the load-plan compiler the same
/// way. Content-based (not mtime), so a no-op rebuild keeps the same hash.
///
/// Every failure here is fatal, for the reason `loader/build.rs` states: a
/// skipped file hashes to the same value as a tree that never had it, so
/// swallowing an I/O error hands out a fingerprint that says "unchanged"
/// about a tracer that did change. Failing the build is the only outcome
/// that keeps the number meaning what it claims.
fn hash_sources(dir: &Path) -> u64 {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_rs(dir, &mut files);
    files.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in &files {
        let bytes = std::fs::read(f)
            .unwrap_or_else(|err| panic!("hashing tracer sources: cannot read {f:?}: {err}"));
        for b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("hashing tracer sources: cannot list {dir:?}: {err}"));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| panic!("hashing tracer sources: cannot walk {dir:?}: {err}"))
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
    println!("cargo:rustc-env=PIE_FORWARD_COMPILER_HASH={compiler_hash}");

    // Hand the committed, cbindgen-generated header directory to downstream
    // build scripts, the way `loader/build.rs` hands out `pie_loader.h`: a
    // driver's build script reads `DEP_PIE_FORWARD_INCLUDE` and adds it to
    // the C++ include path.
    let include = manifest_dir.join("include");
    println!(
        "cargo:rerun-if-changed={}",
        include.join("pie_forward.h").display()
    );
    println!("cargo:include={}", include.display());

    println!("cargo:rerun-if-changed=src");
}

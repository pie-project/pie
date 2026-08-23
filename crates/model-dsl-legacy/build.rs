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
/// The tracer is TWO source trees since the toolchain split: the authoring
/// surface here and the recorder (`TraceBuilder`, the op vocabulary, the seam
/// words) in `model-ir`. A fingerprint that covered only one of them would
/// say "unchanged" about a tracer that did change, which is the exact failure
/// the fatal-on-error rule below exists to prevent -- so both are folded in,
/// in a FIXED order, each tree name-sorted within itself. Chaining rather
/// than combining two independent hashes keeps this a single FNV-1a pass over
/// the concatenation, which is what the number always was.
fn hash_sources(dirs: &[PathBuf]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for dir in dirs {
        hash_tree(dir, &mut h);
    }
    h
}

fn hash_tree(dir: &Path, h: &mut u64) {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_rs(dir, &mut files);
    files.sort();
    for f in &files {
        let bytes = std::fs::read(f)
            .unwrap_or_else(|err| panic!("hashing tracer sources: cannot read {f:?}: {err}"));
        for b in bytes {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
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
    let ir_src = manifest_dir.join("../model-ir/src");
    let dsl_src = manifest_dir.join("src");
    let compiler_hash = hash_sources(&[ir_src, dsl_src]);
    println!("cargo:rustc-env=PIE_FORWARD_COMPILER_HASH={compiler_hash}");

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../model-ir/src");
}

//! The fingerprint of the code that emits a fire's program (this crate's
//! and kernels-xla's sources): a traced fire's record (`Device::trace_record`)
//! holds while it is unchanged, whatever else of pie changes.

use std::path::{Path, PathBuf};

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo states it"));
    let roots = [here.join("src"), here.join("../kernels-xla/src")];
    let mut files = Vec::new();
    for root in &roots {
        println!("cargo:rerun-if-changed={}", root.display());
        sources(root, &mut files);
    }
    files.sort();
    // FNV-1a over each file's path (relative to its root) and bytes
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for file in &files {
        let rel = roots
            .iter()
            .find_map(|r| file.strip_prefix(r).ok())
            .unwrap_or(file);
        eat(rel.to_string_lossy().as_bytes());
        eat(&std::fs::read(file).unwrap_or_default());
    }
    println!("cargo:rustc-env=XLA_LOWERING_FINGERPRINT={hash:016x}");
}

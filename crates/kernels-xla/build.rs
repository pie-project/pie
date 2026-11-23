//! The fingerprint of this crate's sources (`SOURCE_FINGERPRINT`): what
//! the programs its ops lower to depend on, beside the tuning they are
//! laid by. A key on it holds while they are unchanged, whatever else of
//! pie changes.

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
    let root =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo states it")).join("src");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut files = Vec::new();
    sources(&root, &mut files);
    files.sort();
    // FNV-1a over each file's path (relative to src) and bytes
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for file in &files {
        eat(file
            .strip_prefix(&root)
            .unwrap_or(file)
            .to_string_lossy()
            .as_bytes());
        eat(&std::fs::read(file).unwrap_or_default());
    }
    println!("cargo:rustc-env=KERNELS_XLA_SOURCE_FINGERPRINT={hash:016x}");
}

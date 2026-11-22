//! The fingerprint of the code that shapes what a deployment's bodies hold
//! (`serve::measured`): this crate's sources and the kernel crate it builds
//! over (its Rust and its kernels' sources). A record of what the bodies
//! took is kept across pie versions that change none of them, and measured
//! again by one that changes any; nothing of the build's version, commit or
//! file times keys it.

use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let kernels = manifest.join("..").join(if std::env::var_os("CARGO_FEATURE_ROCM").is_some() {
        "kernels-rocm"
    } else {
        "kernels-cuda"
    });
    let roots = [
        manifest.join("src"),
        kernels.join("src"),
        kernels.join("kernels"),
        kernels.join("shim"),
    ];
    let mut paths = Vec::new();
    for root in &roots {
        println!("cargo:rerun-if-changed={}", root.display());
        files(root, &mut paths);
    }
    let base = manifest.join("..");
    let mut named: Vec<(String, PathBuf)> = paths
        .into_iter()
        .map(|path| {
            let name = path
                .strip_prefix(&base)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            (name, path)
        })
        .collect();
    named.sort();
    // FNV-1a over each file's name and bytes, in name order.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (name, path) in &named {
        eat(name.as_bytes());
        eat(&[0]);
        eat(&std::fs::read(path).unwrap_or_default());
        eat(&[0]);
    }
    println!("cargo:rustc-env=PIE_BODIES_BUILD={hash:016x}");

    // The committed bodies records (`profiles/bodies/<key>`, `pie tune
    // export --engine cuda`), carried by key: a load whose cache holds no
    // record of its key takes the committed one.
    let committed = manifest.join("profiles").join("bodies");
    println!("cargo:rerun-if-changed={}", committed.display());
    let mut records: Vec<(String, PathBuf)> = std::fs::read_dir(&committed)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter_map(|path| {
                    let name = path.file_name()?.to_str()?.to_string();
                    Some((name, path))
                })
                .collect()
        })
        .unwrap_or_default();
    records.sort();
    let mut table = String::from("/// The committed bodies records, by key.\npub(crate) const COMMITTED: &[(&str, &str)] = &[\n");
    for (name, path) in &records {
        table.push_str(&format!("    ({name:?}, include_str!({:?})),\n", path.display().to_string()));
    }
    table.push_str("];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets it"));
    std::fs::write(out.join("committed_bodies.rs"), table).expect("OUT_DIR is writable");
}

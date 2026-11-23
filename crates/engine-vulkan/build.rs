//! The hash of the code a Vulkan walk records (`PIE_ARMED_BUILD`): this
//! crate's sources, the shared shell's (`engine-gpu`) and the Slang
//! kernels' (`kernels-slang`), which keys what a load armed; and the armed
//! records the build carries (`profiles/bodies/<key>`, `pie tune export
//! --engine vulkan`).

use std::path::{Path, PathBuf};

fn files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
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
    let base = manifest.join("..");
    let roots = [
        manifest.join("src"),
        base.join("engine-gpu").join("src"),
        base.join("kernels-slang").join("src"),
        base.join("kernels-slang").join("kernels"),
    ];
    let mut paths = Vec::new();
    for root in &roots {
        println!("cargo:rerun-if-changed={}", root.display());
        files(root, &mut paths);
    }
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
    println!("cargo:rustc-env=PIE_ARMED_BUILD={hash:016x}");

    let committed = manifest.join("profiles").join("bodies");
    println!("cargo:rerun-if-changed={}", committed.display());
    let mut records: Vec<(String, PathBuf)> = std::fs::read_dir(&committed)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter_map(|path| Some((path.file_name()?.to_str()?.to_string(), path)))
                .collect()
        })
        .unwrap_or_default();
    records.sort();
    let mut table = String::from(
        "/// The committed armed records, by key.\npub const COMMITTED: &[(&str, &str)] = &[\n",
    );
    for (name, path) in &records {
        table.push_str(&format!(
            "    ({name:?}, include_str!({:?})),\n",
            path.display().to_string()
        ));
    }
    table.push_str("];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets it"));
    std::fs::write(out.join("committed_armed.rs"), table).expect("OUT_DIR is writable");
}

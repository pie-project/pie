//! The hash of the encoder's layout (`PIE_ENCODER_LAYOUT`: the files that
//! lay a recording's commands and its fire block), which a test holds to
//! `armed::ENCODING`; and the armed records the build carries
//! (`profiles/bodies/<key>`, `pie tune export --engine vulkan`).

use std::path::PathBuf;

/// The files that lay a recording's commands and its fire block.
const LAYOUT: [&str; 3] = ["src/encode.rs", "src/device/block.rs", "src/record.rs"];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    // The encoder's layout: the files that lay what a recording's commands
    // carry and what its fire block holds. A change to them that moves the
    // recorded layout bumps `armed::ENCODING` (a test holds the two together).
    let mut layout: Vec<PathBuf> = LAYOUT.iter().map(|file| manifest.join(file)).collect();
    layout.sort();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for path in &layout {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path
            .strip_prefix(&manifest)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        eat(name.as_bytes());
        eat(&[0]);
        eat(&std::fs::read(path).unwrap_or_default());
        eat(&[0]);
    }
    println!("cargo:rustc-env=PIE_ENCODER_LAYOUT={hash:016x}");

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

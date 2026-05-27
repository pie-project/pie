//! Build script — hashes every Rust source file that owns `#[schema]`
//! types to produce `SCHEMA_HASH`. The hash is written into the shmem
//! ring header and compared on connect so a producer and a consumer
//! compiled against different schema versions fail loudly at handshake
//! time.

use std::path::PathBuf;

const SCHEMA_SOURCES: &[&str] = &["src/schema.rs", "src/brle.rs"];

fn main() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let mut bytes = Vec::new();
    for rel in SCHEMA_SOURCES {
        let path = crate_dir.join(rel);
        println!("cargo:rerun-if-changed={}", path.display());
        bytes.extend_from_slice(rel.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&std::fs::read(&path).unwrap_or_else(|e| {
            panic!("read schema source {}: {e}", path.display());
        }));
        bytes.push(0);
    }
    let hash = xxhash_rust::xxh3::xxh3_64(&bytes).to_le_bytes();

    let out_dir = std::env::var_os("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(&out_dir).join("schema_hash.rs");
    std::fs::write(
        out_path,
        format!("pub const SCHEMA_HASH: [u8; 8] = {hash:?};\n"),
    )
    .expect("write schema_hash.rs");

    // Downstream `pie-server/build.rs` reads `DEP_PIE_BRIDGE_INCLUDE` to
    // locate the C header directory it then passes to CMake via
    // `-DPIE_BRIDGE_INCLUDE_DIR=...`. The header lives at
    // `driver/bridge/include/pie_bridge.h` and declares the C ABI emitted
    // by `#[schema]` on each type in the schema sources above.
    let include = crate_dir.join("include");
    println!(
        "cargo:rerun-if-changed={}",
        include.join("pie_bridge.h").display()
    );
    println!("cargo:include={}", include.display());
    let _ = out_dir; // OUT_DIR retained for the schema_hash.rs include.
}

//! pie-schema-bindgen — decoupled generator for the committed C++ header.
//!
//! Emits `protocol/schema/include/pie_schema.h` from `pie-schema`'s repr(C)
//! POD + `cabi` surface via cbindgen. NOT wired into any build graph: the
//! C++/nvcc build consumes the COMMITTED header, so it never needs cbindgen.
//! Devs + CI run `cargo run -p pie-schema-bindgen`; CI then `git diff
//! --exit-code` guards drift (belt) alongside the SCHEMA_HASH handshake
//! (suspenders).
//!
//! Requirements (validated on pie-schema):
//!   * cbindgen >= 0.29 — 0.27 silently drops Rust-2024 `#[unsafe(no_mangle)]`
//!     fns, which would strip the entire `pie_*` function surface.
//!   * RUSTC_BOOTSTRAP=1 — cbindgen's `parse.expand` runs
//!     `cargo rustc -Zunpretty=expanded`, which stable rustc rejects unless
//!     this is set. Set here so the tool "just works" via plain `cargo run`.
//!   * A pinned rustc toolchain in CI keeps the committed-header diff
//!     deterministic (expansion output is toolchain-dependent).

use std::path::PathBuf;

fn main() {
    // Let cbindgen's internal `cargo rustc -Zunpretty=expanded` accept the
    // nightly flag on a stable toolchain. Inherited by child cargo/rustc procs.
    // SAFETY: set before any threads spawn or cbindgen subprocesses launch.
    unsafe {
        std::env::set_var("RUSTC_BOOTSTRAP", "1");
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // protocol/schema-bindgen
    let schema_crate = manifest.join("..").join("schema"); // protocol/schema
    let config_path = manifest.join("cbindgen.toml");
    let out = schema_crate.join("include").join("pie_schema.h");

    let config = cbindgen::Config::from_file(&config_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", config_path.display()));

    cbindgen::Builder::new()
        .with_crate(&schema_crate)
        .with_config(config)
        .generate()
        .expect("generate pie_schema.h (cbindgen parse.expand over pie-schema)")
        .write_to_file(&out);

    eprintln!("wrote {}", out.display());
}

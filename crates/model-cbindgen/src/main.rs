//! model-cbindgen — decoupled generator for the committed C header.
//!
//! Emits `crates/model/include/pie_forward.h` from the `#[repr(C)]` surface in
//! `model::ffi`. The build graph consumes the committed header;
//! developers and CI run this tool to refresh it — the same arrangement as
//! `loader/cbindgen`, and for the same reason: the generated header is the
//! *only* definition of the forward crate's C vocabulary, so a hand-written
//! C++ copy of these enums has nothing to drift from the Rust ones
//! (`loader/architecture.md` §9).

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // A satellite is a SIBLING of its parent, not a subdirectory of it.
    let forward_crate = manifest.join("../model");
    let config_path = manifest.join("cbindgen.toml");
    let out = forward_crate.join("include").join("pie_forward.h");

    let config = cbindgen::Config::from_file(&config_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", config_path.display()));

    cbindgen::Builder::new()
        .with_crate(&forward_crate)
        .with_config(config)
        .generate()
        .expect("generate pie_forward.h from model::ffi")
        .write_to_file(&out);

    eprintln!("wrote {}", out.display());
}

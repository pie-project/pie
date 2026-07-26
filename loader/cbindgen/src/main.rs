//! pie-loader-cbindgen — decoupled generator for the committed C header.
//!
//! Emits `loader/include/pie_loader.h` from the `#[repr(C)]` surface in
//! `pie-loader::ffi`. The build graph consumes the committed header; developers
//! and CI run this tool to refresh it.
//!
//! The header is the *only* definition of the loader's C vocabulary. It replaces
//! the hand-written `PieLoader*` structs that used to live in
//! `driver/common/include/pie_native/load_plan.hpp`, where a second copy of the
//! same enums drifted out of order from the Rust ones without anything noticing
//! (see `loader/architecture.md` §9).

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let loader_crate = manifest.join("..");
    let config_path = manifest.join("cbindgen.toml");
    let out = loader_crate.join("include").join("pie_loader.h");

    let config = cbindgen::Config::from_file(&config_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", config_path.display()));

    cbindgen::Builder::new()
        .with_crate(&loader_crate)
        .with_config(config)
        .generate()
        .expect("generate pie_loader.h from pie-loader::ffi")
        .write_to_file(&out);

    eprintln!("wrote {}", out.display());
}

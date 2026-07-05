//! pie-driver-abi-cbindgen — decoupled generator for the committed C header.
//!
//! Emits `interface/driver/include/pie_driver_abi.h` from `pie-driver-abi`'s
//! plain `local.rs` `#[repr(C)]` surface. The build graph consumes the committed
//! header; developers and CI run this tool to refresh it.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let driver_crate = manifest.join("..");
    let config_path = manifest.join("cbindgen.toml");
    let out = driver_crate.join("include").join("pie_driver_abi.h");

    let config = cbindgen::Config::from_file(&config_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", config_path.display()));

    cbindgen::Builder::new()
        .with_crate(&driver_crate)
        .with_config(config)
        .generate()
        .expect("generate pie_driver_abi.h from pie-driver-abi::local")
        .write_to_file(&out);

    eprintln!("wrote {}", out.display());
}

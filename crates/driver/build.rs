//! Publish `include/` to the two execution shells.
//!
//! No compilation happens here. This crate is headers -- the vocabulary a
//! shell interprets `pie_driver_abi.h` with -- and the only thing a build
//! script can usefully do with headers is tell cargo where they are, so that
//! `driver-cuda` and `driver-metal` read `DEP_PIE_DRIVER_INCLUDE` rather than
//! reaching across the repo by relative path.

fn main() {
    let include = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("include");
    println!("cargo:include={}", include.display());
    println!("cargo:rerun-if-changed=include");
}

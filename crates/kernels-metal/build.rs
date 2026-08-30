//! Stage the RNG preamble beside the shaders that `#include` it.
//!
//! **THE ONLY THING THIS SCRIPT DOES, AND IT IS A COPY.** `eta-compiler` owns
//! the preamble's text and commits it once, under `include/`, where
//! `tests/rng_contract.rs` regenerates and verifies it. Metal's runtime shader
//! compiler resolves a `#include "..."` against the including file's directory
//! and nothing else — `newLibraryWithSource:` has no header search path — so
//! `ptir_m1_runtime.metal` can only reach the preamble if a copy sits in
//! `kernels/ptir/` too. This writes that copy; `.gitignore` keeps it from
//! becoming a second committed source.
//!
//! **IT IS NOT GATED, AND THE PREDECESSOR'S GATE IS WHY IT SAYS SO.** The
//! build script this one restores staged the file only under a `native`
//! feature. This crate no longer has one, and `sources.rs` reaches the staged
//! copy with an unconditional `include_str!`, so a gate here would be a
//! checkout that does not compile rather than a checkout without a device.
//! The copy is cheap and the write is skipped when the bytes already match.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/kernels-metal has a parent")
        .join("eta-compiler/include/ptir_rng.generated.metal");
    println!("cargo:rerun-if-changed={}", src.display());

    let dst = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels/ptir/ptir_rng.generated.metal");

    let text = std::fs::read(&src).unwrap_or_else(|e| {
        panic!(
            "cannot read the generated RNG preamble at {}: {e}. It is emitted \
             by eta-compiler and committed there.",
            src.display()
        )
    });

    // Only when it differs: an unconditional write would restamp the file's
    // mtime every build and make the `rerun-if-changed` above fire on itself.
    if std::fs::read(&dst).ok().as_deref() != Some(text.as_slice()) {
        std::fs::write(&dst, &text)
            .unwrap_or_else(|e| panic!("cannot stage {}: {e}", dst.display()));
    }
}

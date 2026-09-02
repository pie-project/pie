//! Stages the RNG preamble beside the shaders that `#include` it, since
//! Metal's shader compiler resolves `#include "..."` only against the
//! including file's own directory. `eta-compiler` owns the source text
//! (under `include/`); this writes an untracked copy into `kernels/ptir/`.

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

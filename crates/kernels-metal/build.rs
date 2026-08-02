//! Publish `kernels/` to the Metal shell, and stage the one shader that is
//! generated rather than written.
//!
//! There is no compile step here, and that is not an omission. Metal shaders
//! are compiled at RUN time — the driver builds pipeline state objects from
//! `.metal` source it reads out of `PIE_METAL_KERNELS_DIR`, which defaults to
//! this directory. So the CUDA side's `native` feature drives nvcc over a
//! hundred translation units and this one copies a file: the asymmetry is
//! between the two toolchains, not between the two crates' jobs.
//!
//! What `native` gates is the staging below, which reads out of
//! `tensor-compiler`. Without it this crate is the signature table and a
//! directory of shaders, which is what `model-compiler` wants and all it
//! wants.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");

    let kernels = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");

    // Both keys name the same directory and are read for different reasons:
    // `include` is the conventional one a C++ include path is built from (the
    // shell compiles against `affine_format.hpp` and the `*_params.h` a
    // shader and its host caller must agree on), and `kernels_dir` is what
    // becomes `PIE_METAL_KERNELS_DIR_DEFAULT` — a path baked into the binary
    // for the runtime shader compiler to read `.metal` out of.
    println!("cargo:include={}", kernels.display());
    println!("cargo:kernels_dir={}", kernels.display());

    if std::env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }

    stage_rng_preamble(&kernels);
}

/// `ptir_rng.generated.metal` is `tensor-compiler`'s: the host emitter
/// generates it, and it is committed once, over there.
///
/// It has to sit BESIDE the hand-written shaders anyway, because Metal's
/// runtime shader compiler does no filesystem include lookup — the driver
/// splices this preamble into the source text it hands the compiler, so both
/// halves have to be reachable from one directory. Copying is how they stay
/// in lockstep; a second committed copy is how they would stop.
fn stage_rng_preamble(kernels: &Path) {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/kernels-metal has a parent")
        .join("tensor-compiler/include/ptir_rng.generated.metal");
    println!("cargo:rerun-if-changed={}", src.display());

    let dst = kernels.join("ptir_rng.generated.metal");
    let text = std::fs::read(&src).unwrap_or_else(|e| {
        panic!(
            "cannot read the generated RNG preamble at {}: {e}. It is emitted \
             by tensor-compiler and committed there.",
            src.display()
        )
    });
    // Only write on a real difference: an unconditional write would restamp
    // the mtime every build and re-trigger the `rerun-if-changed=kernels`
    // above, which is a build that never settles.
    if std::fs::read(&dst).ok().as_deref() != Some(text.as_slice()) {
        std::fs::write(&dst, &text)
            .unwrap_or_else(|e| panic!("cannot stage {}: {e}", dst.display()));
    }
}

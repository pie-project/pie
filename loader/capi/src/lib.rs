//! The loader's C ABI, as a crate of its own.
//!
//! `pie-loader` is the compiler and knows nothing of C; this crate is where
//! its plans become `#[repr(C)]` and its entry points become symbols a driver
//! links. Splitting the two makes the layering a dependency arrow instead of
//! a module convention — the compiler cannot reach the ABI, and everything
//! that must version with the header lives next to the header.
//!
//! `types` is the published `#[repr(C)]` vocabulary, `arena` turns a compiled
//! [`LoadPlan`](pie_loader::plan::LoadPlan) into it, and `entry` holds the
//! `extern "C"` functions the driver calls. `model` reads the one input back
//! the other way — the facts-and-policy request a driver states — and
//! `checkpoint` says which checkpoint to ask it of. `weight_store` is a
//! surface of its own: the materialized-weight artifact, which is about
//! finished device memory rather than about compiling anything, and carries
//! its own entry points for that reason. The generated header
//! (`loader/capi/include/pie_loader.h`) is the C view of exactly those five
//! files; `view` is Rust-side borrowing helpers over them and publishes
//! nothing. (A driver used to be able to hand over a fully-authored contract
//! graph instead — `contract`, `contract_writer` and the C++ builder in
//! `model_contract.hpp` were that path. It died with the last C++ author,
//! `plan/model-in-rust.md` §8-5: authoring happens on this side now, and a
//! contract is loader-internal IR that never crosses the ABI.)
//!
//! This is the only boundary the design has (`architecture.md` §10). There is
//! no serialized form: a plan is compiled and executed in one process, so a
//! JSON round-trip would be a second representation to keep in step with no
//! reader on the far end.

pub mod arena;
pub mod checkpoint;
pub mod entry;
pub mod model;
pub mod types;
pub mod view;
pub mod weight_store;

pub use entry::{
    PieLoaderDiagnostic, PieLoaderDiagnostics, PieLoaderSeverity, PieLoaderStatus,
    PieLoaderTargetSpec, pie_loader_release, pie_loader_release_diagnostics,
};
pub use types::*;

use pie_loader::plan::StorageTarget;
use pie_loader::types::{BackendKind, DType};

/// Build the compiler's [`StorageTarget`] from the driver's measured spec.
///
/// Every field is copied straight through: the driver measured it, so the
/// loader has nothing to add. What this function does add is refusal — a target
/// that under-states itself is an error, not a default.
fn storage_target(
    spec: &PieLoaderTargetSpec,
    backend: PieLoaderBackendKind,
) -> Result<StorageTarget, String> {
    // A target that reports no tile budget is not saying "no limit"; it is
    // saying it did not measure one, and the loader used to guess 64 MiB on its
    // behalf. Guessing a device constant is exactly what `architecture.md` §9
    // forbids: the number decided how much scratch every Encode allocated, so
    // the guess was a silent performance contract nobody had signed.
    if spec.max_tile_bytes == 0 {
        return Err(
            "request.target.max_tile_bytes is 0; the driver must state its tile \
             budget (there is no safe default for a device the loader cannot measure)"
                .to_string(),
        );
    }
    let kind = match backend {
        PieLoaderBackendKind::Cuda => BackendKind::Cuda,
        PieLoaderBackendKind::Metal => BackendKind::Metal,
        PieLoaderBackendKind::Unknown => BackendKind::Unknown,
    };

    // The driver's `tile_map_mask` and the loader's `plan::passes::tile` masks
    // are two independent statements of the same fact — the C++ constant is
    // written by hand next to the kernels, the Rust one next to the lowering
    // rules that decide when to emit them. `architecture.md` §9 makes the
    // driver the authority, so a driver may implement *fewer* transforms than
    // the loader knows how to lower. What it may not do is claim one the loader
    // has never heard of: that bit would silently pass
    // `validate_target_support` and then fail as an unrecognized kernel
    // dispatch at load time, far from its cause (`architecture.md` §8).
    let known = pie_loader::plan::passes::tile::tile_map_mask(kind);
    if spec.tile_map_mask & !known != 0 {
        return Err(format!(
            "target claims tile map transforms {:#x} that the {} backend model \
             does not define (loader knows {:#x})",
            spec.tile_map_mask & !known,
            backend_name(kind),
            known
        ));
    }

    Ok(StorageTarget {
        backend: kind,
        tp_rank: spec.tp_rank,
        tp_size: spec.tp_size,
        max_tile_bytes: spec.max_tile_bytes,
        preferred_alignment: spec.preferred_alignment,
        tile_map_mask: spec.tile_map_mask,
        native_mxfp4_moe: spec.native_mxfp4_moe,
        fusion_mask: spec.fusion_mask,
        encode_scratch_dtype: PieLoaderDType::try_from(spec.encode_scratch_dtype)
            .map(DType::from)
            .map_err(|v| format!("request.target.encode_scratch_dtype: {v} is not a dtype"))?,
        block_scale_rows: spec.block_scale_rows,
    })
}

#[cfg(test)]
mod tests;

/// The lowercase name a driver would recognize, for messages only.
fn backend_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Cuda => "cuda",
        BackendKind::Metal => "metal",
        BackendKind::Unknown => "host",
    }
}

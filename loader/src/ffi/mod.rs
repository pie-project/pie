//! The loader's FFI boundary.
//!
//! `types` is the published `#[repr(C)]` vocabulary, `arena` turns a compiled
//! [`LoadPlan`](crate::load_plan::LoadPlan) into it, and `entry` holds the
//! `extern "C"` functions the driver calls. The generated header
//! (`loader/include/pie_loader.h`) is the C view of exactly these three files.
//!
//! This module replaces `inproc.rs`: where that offered a Rust-to-Rust compile
//! helper and a JSON serializer, this offers the one boundary the design
//! actually has (§10).

pub mod arena;
pub mod checkpoint;
pub mod contract;
pub mod entry;
pub mod inproc;
pub mod types;

pub use entry::{
    PieLoaderDiagnostic, PieLoaderDiagnostics, PieLoaderSeverity, PieLoaderStatus,
    PieLoaderTargetSpec, pie_loader_release, pie_loader_release_diagnostics,
};
pub use types::*;

use crate::load_plan::StorageTarget;
use crate::types::{BackendKind, DType};

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
    // behalf. Guessing a device constant is exactly what §9 forbids: the number
    // decided how much scratch every Encode allocated, so the guess was a
    // silent performance contract nobody had signed.
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

    // The driver's `tile_map_mask` and the loader's `backend::*::TILE_MAP_MASK`
    // are two independent statements of the same fact — the C++ constant is
    // written by hand next to the kernels, the Rust one next to the lowering
    // rules that decide when to emit them. §9 makes the driver the authority, so
    // a driver may implement *fewer* transforms than the loader knows how to
    // lower. What it may not do is claim one the loader has never heard of:
    // that bit would silently pass `validate_target_support` and then fail as an
    // unrecognized kernel dispatch at load time, far from its cause (§8).
    let known = crate::backend::for_backend(kind).tile_map_mask();
    if spec.tile_map_mask & !known != 0 {
        return Err(format!(
            "target claims tile map transforms {:#x} that the {} backend model \
             does not define (loader knows {:#x})",
            spec.tile_map_mask & !known,
            crate::backend::for_backend(kind).name(),
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

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
    PieLoaderComponent, PieLoaderDiagnostic, PieLoaderDiagnostics, PieLoaderMxfp4MoeRequest,
    PieLoaderRequest, PieLoaderSeverity, PieLoaderStatus, PieLoaderTargetSpec, pie_loader_compile,
    pie_loader_release, pie_loader_release_diagnostics, pie_loader_verify,
};
pub use types::*;

use crate::config::ModelConfig;
use crate::error::CompileError;
use crate::load_plan::StorageTarget;
use crate::types::{BackendKind, DType, Mxfp4MoePolicy};

/// Take the model facts from the request.
///
/// The loader used to read `config.json` here. It no longer does: the driver
/// has already parsed that file to build its model, and `config.json` describes
/// both how to interpret the checkpoint *and* how to configure inference — only
/// the driver can tell the two apart (§10.4). So the driver states the handful
/// of facts the storage compile keys off, and the loader opens no JSON at all.
fn model_config(
    spec: &crate::ffi::entry::PieLoaderModelSpec,
    runtime_quant: &str,
) -> Result<ModelConfig, CompileError> {
    let text = |bytes: &PieLoaderBytes, what: &str| -> Result<String, CompileError> {
        unsafe { crate::ffi::entry::as_str(bytes, what) }
            .map(str::to_string)
            .map_err(CompileError::InvalidInput)
    };
    Ok(ModelConfig {
        model_type: text(&spec.model_type, "request.model.model_type")?,
        quant_method: text(&spec.quant_method, "request.model.quant_method")?,
        runtime_quant: runtime_quant.to_string(),
        num_hidden_layers: spec.num_hidden_layers,
        num_experts: spec.num_experts,
        num_experts_per_tok: spec.num_experts_per_tok,
    })
}

/// Resolve the caller's MoE request against what the device can do.
///
/// Split out of [`storage_target`] because verification needs the same answer:
/// the plan records the resolved policy, so checking a plan against a request
/// means resolving the request first (§8).
fn resolve_mxfp4_moe(
    mxfp4_moe: PieLoaderMxfp4MoeRequest,
    native_mxfp4_moe: bool,
) -> Result<Mxfp4MoePolicy, String> {
    match mxfp4_moe {
        PieLoaderMxfp4MoeRequest::RoutedDecode => Ok(Mxfp4MoePolicy::RoutedDecode),
        PieLoaderMxfp4MoeRequest::EagerBf16 => Ok(Mxfp4MoePolicy::EagerBf16),
        PieLoaderMxfp4MoeRequest::NativeGemm => {
            if !native_mxfp4_moe {
                return Err("mxfp4_moe=NativeGemm requested, but the target reports \
                     native_mxfp4_moe=false"
                    .to_string());
            }
            Ok(Mxfp4MoePolicy::NativeGemm)
        }
        // `Auto` follows the device: a native MXFP4 GEMM is always the better
        // path when the kernels exist, and decoding on the routed path is the
        // fallback that works everywhere.
        PieLoaderMxfp4MoeRequest::Auto => {
            if native_mxfp4_moe {
                Ok(Mxfp4MoePolicy::NativeGemm)
            } else {
                Ok(Mxfp4MoePolicy::RoutedDecode)
            }
        }
    }
}

/// Build the compiler's [`StorageTarget`] from the driver's measured spec.
///
/// Everything except the MoE policy is copied straight through: the driver
/// measured it, so the loader has nothing to add.
fn storage_target(
    spec: &PieLoaderTargetSpec,
    backend: PieLoaderBackendKind,
    mxfp4_moe: PieLoaderMxfp4MoeRequest,
) -> Result<StorageTarget, String> {
    let mxfp4_moe = resolve_mxfp4_moe(mxfp4_moe, spec.native_mxfp4_moe)?;
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
        mxfp4_moe,
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

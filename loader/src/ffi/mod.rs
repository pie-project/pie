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
use crate::contract::ModelContract;
use crate::error::CompileError;
use crate::load_plan::StorageTarget;
use crate::types::{BackendKind, Mxfp4MoePolicy};

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
        fused_transcode: spec.fused_transcode,
    })
}

/// Narrow the ABI to the requested component.
///
/// `Full` and `Text` keep everything. `Encode` retains only the multimodal
/// towers, so an encoder-only rank never materializes the language model. The
/// guards below are the ones the worker applied before the boundary moved; they
/// live here now because this is where the request arrives.
fn scope_to_component(
    contract: ModelContract,
    component: PieLoaderComponent,
    model: &ModelConfig,
    target: &StorageTarget,
) -> Result<ModelContract, CompileError> {
    if component != PieLoaderComponent::Encode || target.backend != BackendKind::Cuda {
        return Ok(contract);
    }
    if target.tp_size != 1 {
        return Err(CompileError::InvalidInput(
            "encode-scoped CUDA loading does not support tensor parallelism".to_string(),
        ));
    }
    if !matches!(model.model_type.as_str(), "gemma4" | "gemma4_text") {
        return Err(CompileError::InvalidInput(format!(
            "encode-scoped CUDA loading currently supports Gemma-4, got {}",
            model.model_type
        )));
    }
    contract.retain_outputs(|name| {
        name.starts_with("model.vision_tower.")
            || name.starts_with("model.embed_vision.")
            || name.starts_with("model.audio_tower.")
            || name.starts_with("model.embed_audio.")
    })
}

#[cfg(test)]
mod tests;

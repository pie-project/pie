//! What this driver tells the loader, and the plan it gets back.
//!
//! The C++ (`loader/load_plan.hpp`) reached the Rust loader through the C ABI:
//! open a checkpoint handle, marshal a request, get a marshalled plan back,
//! then run it with `load_plan_executor.hpp`. In-process there is no wire, and
//! the executor is `model_loader::executor::host` driven through a
//! [`DeviceArena`](super::arena::DeviceArena) — so what is left here is only
//! what the driver alone knows: which tile transforms it can run, its storage
//! constants, and the loading policy.
//!
//! The policy STATES every field rather than defaulting any, because equal
//! requests must author equal contracts and a default is a decision made
//! somewhere the reader cannot see.

use std::path::Path;

use model::facts::ModelFacts;
use model::policy::{
    Component, FamilyKnobs, Mxfp4MoePolicy, Mxfp4MoeRequest, Naming, Policy, Projections,
    RuntimeQuant,
};
use model_loader::plan::{self, LoadPlan, StorageTarget, TileMapKind};
use model_loader::types::{BackendKind, DType};

/// What the loader may put in a plan for this driver.
///
/// The same three as Metal's, and for the same reason: the transforms are run
/// by `model_loader::executor::host` on their way to the device, so the mask
/// is bounded by what THAT implements
/// ([`CONVERT_TILE_MAP_MASK`](model_loader::plan::CONVERT_TILE_MAP_MASK)) and
/// not by any CUDA kernel.
///
/// `Repack` and `Reblock` are what the C++ `transcode_engine.hpp` had device
/// kernels for and this tree does not. They stay out of the mask, so a
/// checkpoint needing one is refused when its plan is COMPILED — with the
/// tensor named — rather than mis-bound at launch.
pub const CUDA_TILE_MAP_MASK: u32 = TileMapKind::Cast.capability_bit()
    | TileMapKind::Encode.capability_bit()
    | TileMapKind::Scale.capability_bit();

/// Alignment for a tensor's offset inside the arena.
///
/// 256 bytes: what cuBLAS wants for a matrix operand and what `cudaMalloc`
/// itself guarantees, so a view into the arena is as aligned as its own
/// allocation would have been.
pub const CUDA_PREFERRED_ALIGNMENT: u32 = 256;

/// How much host staging one load-time transform may take at once.
pub const CUDA_MAX_TILE_BYTES: u64 = 64 * 1024 * 1024;

/// This device's storage capability, for one rank of a tensor-parallel group.
///
/// `tp_rank`/`tp_size` are the whole of what makes a load SHARDED. Every
/// family's contract states its splits in terms of them
/// (`Builder::local_extent`, `Builder::split`), so a rank compiles a plan that
/// reads only its own bands out of the checkpoint and allocates an arena sized
/// to them. The driver never slices a tensor itself.
///
/// The KV cache is sharded from the same two numbers, one layer down
/// (`store::kv_geometry` divides `num_key_value_heads` by `tp_size`), so the
/// weights and the cache cannot disagree about how wide a rank is.
#[must_use]
pub fn cuda_storage_target(tp_rank: u32, tp_size: u32) -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank,
        tp_size: tp_size.max(1),
        max_tile_bytes: CUDA_MAX_TILE_BYTES,
        preferred_alignment: CUDA_PREFERRED_ALIGNMENT,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        // FALSE, and the name is the trap. `native_mxfp4_moe` does not mean
        // "reads MXFP4"; it means "has a native MXFP4 *GEMM*", which in
        // gpt-oss's contract selects a Marlin REPACK of the expert banks —
        // `transcode_engine.hpp`'s work, which this tree did not port.
        //
        // This driver takes the other branch: `_blocks`/`_scales`/`_bias`
        // pass through as three plain tensors and `quant::mxfp4_moe_*_decode`
        // indexes the stored layout directly. So the honest answer is no.
        native_mxfp4_moe: false,
        // No fused transcode kernels; `PIE_CUDA_DISABLE_FUSED_TRANSCODE` was
        // the C++ knob for the ones that no longer exist.
        fusion_mask: 0,
        encode_scratch_dtype: DType::BF16,
        block_scale_rows: 0,
    }
}

/// Why a plan could not be compiled, with the checkpoint's own words.
#[derive(Debug)]
pub enum LoadPlanError {
    /// The `pie.model/1` document did not parse.
    Descriptor(String),
    /// The contract or the plan did not compile.
    Compile(String),
    /// No author in the `model` registry claims this `model_type`.
    UnknownFamily(String),
    /// A file the plan declares is missing, or is the wrong size on disk.
    DeclaredFile(String),
}

impl std::fmt::Display for LoadPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Descriptor(m) => write!(f, "descriptor: {m}"),
            Self::Compile(m) => write!(f, "compile: {m}"),
            Self::UnknownFamily(m) => write!(f, "no author for model_type {m:?}"),
            Self::DeclaredFile(m) => write!(f, "checkpoint file: {m}"),
        }
    }
}

/// Author the contract this driver wants and compile it into a plan.
///
/// The policy is the CUDA one, and each field is a claim about the forward
/// path rather than a preference:
///
/// - [`Projections::Fused`] because the dense attention and MLP GEMMs take one
///   operand — `layer.N.qkv`, `layer.N.gate_up`. The driver used to build
///   those itself, at load, by reading three tensors BACK off the device and
///   re-uploading their concatenation. The plan states the same joins as
///   `BulkExtentWrite`s into the arena, so the bytes land fused the first and
///   only time they are copied.
/// - [`Naming::Hf`] because this driver binds checkpoint names.
/// - [`RuntimeQuant::None`] because a requantization is a decision about an
///   artifact, made once by `pie model build --quant`, not re-run every boot.
/// - [`Mxfp4MoeRequest::Auto`] with `native_mxfp4_moe`, so gpt-oss binds its
///   expert banks as stored.
///
/// # Errors
///
/// The descriptor does not parse, no author claims its `model_type`, the
/// contract does not compile, or a file the plan declares is missing or the
/// wrong size on disk.
pub fn compile_load_plan(
    snapshot_dir: &Path,
    metadata: &model_loader::checkpoint::CheckpointMetadata,
    target: &StorageTarget,
    descriptor_json: &str,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    let facts = ModelFacts::from_descriptor(descriptor_json.as_bytes())
        .map_err(|e| LoadPlanError::Descriptor(e.to_string()))?;
    let policy = Policy {
        projections: Projections::Fused,
        naming: Naming::Hf,
        runtime_quant: RuntimeQuant::None,
        moe_request: Mxfp4MoeRequest::Auto,
        component: Component::Full,
        stream_routed_experts: false,
        knobs: FamilyKnobs::default(),
    };
    let (contract, resolved_moe) =
        model::contract::author_with_policy(&facts, metadata, target, &policy)
            .map_err(|e| LoadPlanError::Compile(e.to_string()))?
            .ok_or_else(|| LoadPlanError::UnknownFamily(facts.model_type.clone()))?;
    let plan = plan::compile(metadata, &contract, target.clone())
        .map_err(|e| LoadPlanError::Compile(e.to_string()))?;
    for file in &plan.files {
        let path = snapshot_dir.join(&file.path);
        match std::fs::metadata(&path) {
            Ok(m) if m.len() == file.size_bytes => {}
            Ok(m) => {
                return Err(LoadPlanError::DeclaredFile(format!(
                    "{} is {} bytes on disk, the plan declares {}",
                    path.display(),
                    m.len(),
                    file.size_bytes
                )));
            }
            Err(e) => {
                return Err(LoadPlanError::DeclaredFile(format!(
                    "{}: {e}",
                    path.display()
                )));
            }
        }
    }
    Ok((plan, resolved_moe))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rank_states_its_own_place_in_the_group() {
        let t = cuda_storage_target(1, 4);
        assert_eq!(t.tp_rank, 1);
        assert_eq!(t.tp_size, 4);
        // A zero group size is one rank, not a division by zero.
        assert_eq!(cuda_storage_target(0, 0).tp_size, 1);
    }

    #[test]
    fn the_driver_mask_never_claims_a_transform_the_loader_cannot_model() {
        let modelled = model_loader::plan::passes::tile::tile_map_mask(BackendKind::Cuda);
        assert_eq!(
            CUDA_TILE_MAP_MASK & !modelled,
            0,
            "narrower is fine; wider is a claim about kernels the loader \
             cannot reason about"
        );
    }

    #[test]
    fn every_transform_this_driver_claims_has_a_host_implementation() {
        // What lets this driver delegate load-time transcoding to
        // `model_loader::executor::host` instead of porting the C++
        // `transcode_engine.hpp`: the executor's gate is the convert mask,
        // and every transform this mask can put in a plan is inside it.
        assert_eq!(
            CUDA_TILE_MAP_MASK & !model_loader::plan::CONVERT_TILE_MAP_MASK,
            0,
            "a transform the host executor cannot run would fail at load, \
             far from this claim"
        );
    }

    #[test]
    fn the_target_states_the_device_and_nothing_optimistic() {
        let t = cuda_storage_target(0, 1);
        assert_eq!(t.backend, BackendKind::Cuda);
        assert_eq!(t.preferred_alignment, 256);
        assert_eq!(t.fusion_mask, 0, "no fused transcode kernels here");
        assert!(
            !t.native_mxfp4_moe,
            "the routed-decode path reads the stored banks; the native GEMM \
             would want a Marlin repack no kernel here implements"
        );
    }
}

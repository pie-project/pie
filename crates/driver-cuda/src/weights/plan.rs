//! What this driver tells the loader, and the plan it gets back.

use std::path::Path;

use model::boot::Binding;
use model::shared::policy::Mxfp4MoePolicy;
use model_loader::plan::{LoadPlan, StorageTarget};
use model_loader::types::BackendKind;

/// This device's storage capability, for one tensor-parallel rank. `tp_rank`/
/// `tp_size` also shard the KV cache, so weights and cache can't disagree.
#[must_use]
pub fn cuda_storage_target(tp_rank: u32, tp_size: u32) -> StorageTarget {
    StorageTarget::for_backend(BackendKind::Cuda, tp_rank, tp_size)
}

/// Why a plan could not be compiled, with the checkpoint's own words.
pub use model::boot::LoadPlanError;

/// Author the contract this driver wants and compile it into a plan.
/// Contributes only [`Binding::HF_FUSED`]; the rest is [`model::boot::compile_load_plan`]'s.
/// # Errors
/// The checkpoint is unknown, an override is missing, or the contract or a file is invalid.
pub fn compile_load_plan(
    snapshot_dir: &Path,
    metadata: &model_loader::checkpoint::CheckpointMetadata,
    target: &StorageTarget,
    chosen: &model::catalog::Override,
    encoding: &model::encoding::Encoding,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    model::boot::compile_load_plan(
        snapshot_dir,
        metadata,
        target,
        chosen,
        encoding,
        Binding::HF_FUSED,
    )
}

/// The same, for a caller that has already matched its row.
/// # Errors
/// As [`compile_load_plan`], minus the identification.
pub fn compile_load_plan_for(
    snapshot_dir: &Path,
    metadata: &model_loader::checkpoint::CheckpointMetadata,
    target: &StorageTarget,
    row: &dyn model::catalog::Variant,
    encoding: &model::encoding::Encoding,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    model::boot::compile_load_plan_for(
        snapshot_dir,
        metadata,
        target,
        row,
        encoding,
        Binding::HF_FUSED,
    )
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

    /// `native_mxfp4_moe` stays `false`: no kernel here does the Marlin repack.
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

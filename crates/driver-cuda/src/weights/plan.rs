//! What this driver tells the loader, and the plan it gets back.
//!
//! The C++ (`loader/load_plan.hpp`) reached the Rust loader through the C ABI:
//! open a checkpoint handle, marshal a request, get a marshalled plan back,
//! then run it with `load_plan_executor.hpp`. In-process there is no wire, and
//! the executor is `model_loader::executor::host` driven through a
//! [`DeviceArena`](super::arena::DeviceArena) — so what is left here is only
//! what the driver alone knows.
//!
//! Which is less than it was. The loading policy stated all seven
//! [`model::policy::Policy`] fields here, and `driver-metal` stated the same
//! seven, differing in exactly two; the comment below the file check said the
//! other driver "carried the same block, bit for bit". Equal requests have to
//! author equal contracts, and two copies of a policy cannot promise that —
//! a field added to `Policy` gets a considered value on the copy its author
//! was reading and a `Default` on the other, and nothing fails to compile.
//! The five shared answers are [`model::boot`]'s now. What stays here is the
//! two this driver knows, named [`Binding::HF_FUSED`].

use std::path::Path;

use model::boot::Binding;
use model::policy::Mxfp4MoePolicy;
use model_loader::plan::{LoadPlan, StorageTarget};
use model_loader::types::BackendKind;

/// This device's storage capability, for one rank of a tensor-parallel group.
///
/// `tp_rank`/`tp_size` are the whole of what makes a load SHARDED. Every
/// family's contract states its splits in terms of them
/// (`Builder::local_extent`, `Builder::split`), so a rank compiles a plan that
/// reads only its own bands out of the checkpoint and allocates an arena sized
/// to them. The driver never slices a tensor itself.
///
/// The KV cache is sharded from the same two numbers, one layer down
/// (`layout::kv_geometry` divides `num_key_value_heads` by `tp_size`), so the
/// weights and the cache cannot disagree about how wide a rank is.
///
/// Everything else — the alignment, the tile budget, the transform mask — is
/// `StorageTarget::for_backend`'s. This module used to state all of it, and
/// the mask twice over: once here and once in `model_loader::plan::passes::tile`,
/// with a test comparing them. A driver is the authority on what its kernels
/// do only until the loader has to own the fallback for each one, which it
/// does; so there is one statement now, and it is the loader's.
#[must_use]
pub fn cuda_storage_target(tp_rank: u32, tp_size: u32) -> StorageTarget {
    StorageTarget::for_backend(BackendKind::Cuda, tp_rank, tp_size)
}

/// Why a plan could not be compiled, with the checkpoint's own words.
///
/// Re-exported rather than redeclared. This driver had its own three-variant
/// copy — `Descriptor`, `UnknownFamily`, `Compile` — and every one of them
/// described a refusal raised inside the shared load path, not here. Nothing
/// outside this module ever matched on it; the boot site formats it and
/// returns `PIE_STATUS_UNSUPPORTED`. A second enum whose variants a driver
/// cannot itself produce is a name, so the name is all that is kept.
pub use model::boot::LoadPlanError;

/// Author the contract this driver wants and compile it into a plan.
///
/// The two answers this driver contributes are [`Binding::HF_FUSED`], and
/// both are claims about the forward path rather than preferences:
///
/// - [`Projections::Fused`](model::policy::Projections::Fused) because the
///   dense attention and MLP GEMMs take one operand — `layer.N.qkv`,
///   `layer.N.gate_up`. The driver used to build those itself, at load, by
///   reading three tensors BACK off the device and re-uploading their
///   concatenation. The plan states the same joins as `BulkExtentWrite`s into
///   the arena, so the bytes land fused the first and only time they are
///   copied.
/// - [`Naming::Hf`](model::policy::Naming::Hf) because this driver binds
///   checkpoint names.
///
/// The remaining five policy fields, the author call, the compile and the
/// declared-file check are [`model::boot::compile_load_plan`]'s, shared with
/// `driver-metal`.
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
    model::boot::compile_load_plan(
        snapshot_dir,
        metadata,
        target,
        descriptor_json,
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

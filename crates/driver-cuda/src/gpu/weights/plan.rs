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
use model_loader::plan::{self, LoadPlan, StorageTarget};
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
#[derive(Debug)]
pub enum LoadPlanError {
    /// The `pie.model/1` document did not parse.
    Descriptor(String),
    /// The contract or the plan did not compile.
    Compile(String),
    /// No author in the `model` registry claims this `model_type`.
    ///
    /// The only refusal here the loader could not raise itself: the family
    /// registry is `model`'s, and a `model_type` nothing claims is a fact
    /// about that registry rather than about the checkpoint.
    UnknownFamily(String),
}

impl std::fmt::Display for LoadPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Descriptor(m) => write!(f, "descriptor: {m}"),
            Self::Compile(m) => write!(f, "compile: {m}"),
            Self::UnknownFamily(m) => write!(f, "no author for model_type {m:?}"),
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
    // The plan names the files and states their sizes, so a snapshot that
    // moved under a plan compiled against it is the loader's refusal to make,
    // not this module's. `driver-metal` carried the same block, bit for bit.
    model_loader::checkpoint::read::verify_declared_files(&plan, snapshot_dir)
        .map_err(|e| LoadPlanError::Compile(e.to_string()))?;
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

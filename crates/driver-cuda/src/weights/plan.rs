//! What this driver tells the loader, and the plan it gets back.
//!
//! The C++ (`loader/load_plan.hpp`) reached the Rust loader through the C ABI:
//! open a checkpoint handle, marshal a request, get a marshalled plan back,
//! then run it with `load_plan_executor.hpp`. In-process there is no wire, and
//! the executor is `model_loader::executor::Execution` driven through a
//! [`DeviceArena`](super::arena::DeviceArena) — so what is left here is only
//! what the driver alone knows.
//!
//! Which is less than it was. The loading policy stated all seven
//! [`model::shared::policy::Policy`] fields here, and `driver-metal` stated the same
//! seven, differing in exactly two; the comment below the file check said the
//! other driver "carried the same block, bit for bit". Equal requests have to
//! author equal contracts, and two copies of a policy cannot promise that —
//! a field added to `Policy` gets a considered value on the copy its author
//! was reading and a `Default` on the other, and nothing fails to compile.
//! The five shared answers are [`model::boot`]'s now. What stays here is the
//! two this driver knows, named [`Binding::HF_FUSED`].

use std::path::Path;

use model::boot::Binding;
use model::shared::policy::Mxfp4MoePolicy;
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
/// - [`Projections::Fused`](model::shared::policy::Projections::Fused) because the
///   dense attention and MLP GEMMs take one operand — `layer.N.qkv`,
///   `layer.N.gate_up`. The driver used to build those itself, at load, by
///   reading three tensors BACK off the device and re-uploading their
///   concatenation. The plan states the same joins as `BulkExtentWrite`s into
///   the arena, so the bytes land fused the first and only time they are
///   copied.
/// - [`Naming::Hf`](model::shared::policy::Naming::Hf) because this driver binds
///   checkpoint names.
///
/// The remaining five policy fields, the author call, the compile and the
/// declared-file check are [`model::boot::compile_load_plan`]'s, shared with
/// `driver-metal`.
///
/// # Errors
///
/// The checkpoint is no model this build serves, an override named a row
/// that does not exist, the contract does not compile, or a file the plan
/// declares is missing or the wrong size on disk.
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
///
/// The boot path has: it identifies once and then uses the row for the
/// contract, the deployment and the trace, so a second `identify` here
/// would be the same match run twice with the second answer thrown away.
///
/// # Errors
///
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

    /// The device fact this asserts is `false` has a MEASUREMENT behind it,
    /// and this is where it lives now.
    ///
    /// The assertion's message says the native GEMM "would want a Marlin
    /// repack no kernel here implements", which is true and is not the whole
    /// reason. The rest was in `kernels-cuda/csrc/src/kernels_manifest.hpp`'s
    /// sm_100 quarantine block — **the ARCHIVE crate's text, deleted with it
    /// at north star step 6** — whose closing line reads *"`new-horizon.md`
    /// §47 holds the argument and this comment holds the measurement."* That
    /// division is accurate and it is a single point of failure twice over:
    /// §47 held no numbers, and that comment goes with the archive at step 6.
    /// Both halves were on the losing side of the deletion. So it is carried
    /// here, in a crate that survives and a file that is committed.
    ///
    /// (The first draft of this note said `.wiki/` "is not tracked by this
    /// repository at all". Half right and the wrong half: `.wiki/` is its own
    /// git repository — 154 tracked files, its own history — and the parent's
    /// `git add` refuses only because the nested repo owns those paths. The
    /// placement stands on a better reason than the one it was made for: **a
    /// measurement beside the assertion that depends on it is read by the
    /// person who needs it, and a measurement in a document is read by
    /// whoever happens to be reading that document.**)
    ///
    /// THE QUESTION. Do the Marlin MXFP4 MoE kernels produce correct values
    /// on sm_100? Measured on a B200 with gpt-oss-20b: decode under this
    /// lowering emitted uniform garbage (`nasquorashBR @@ Put ShortfacesInte
    /// Imper fmt Ind tass`), a **0% function-word rate against 39%** for the
    /// same prompt on vLLM and SGLang, while the routed-dequant GEMV answered
    /// correctly at **314 tok/s** on the same build. Bisected against CUDA
    /// graph capture: corrupt both captured and eager, correct both ways with
    /// the GEMV, so the lowering was the only variable. The generated kernel
    /// set is sm80-shaped (`sm80_kernel_bfloat16_fe2m1f_bfloat16.cu`) and was
    /// never exercised above sm_90.
    ///
    /// WHY IT IS PROBABLY A NO-QUESTION. The root cause was found afterwards
    /// and it is not the kernel. Two bugs in the lowering ABOVE the GEMM,
    /// both architecture-independent, corrupted sm_80 in exactly the way
    /// described for sm_100. (1) The MXFP4 group scales were transposed
    /// TWICE — the loader already publishes them in Marlin's order and the
    /// mixtral path transposed them again; mean relative error against a host
    /// reference at gpt-oss's shape was **0.0017 correct against 0.9350
    /// double-transposed**, i.e. uncorrelated output. (2) `d_marlin_act`'s
    /// padding tail was never initialised, and `0 * NaN` is NaN, so one NaN
    /// pattern poisoned a whole fp32-accumulated output row. After both fixes
    /// sm_80 measured **0/64** degenerate requests at 32-wide and **0/16** at
    /// concurrency 1, against **8–12/32** and **2/16** before, and the kernel
    /// measured correct on sm_80 (0.0017 to 0.0023 mean relative error across
    /// E ∈ {1, 4, 32}, top_k ∈ {1, 4}).
    ///
    /// HOW TO ANSWER IT NOW. Only by re-vendoring, so it is a precondition on
    /// that work rather than a test anyone can run today: whoever restores a
    /// native MXFP4 MoE lowering re-tests sm_100 FIRST, before re-adding any
    /// quarantine, and only re-adds one if it is actually dirty. The original
    /// instruction — *"RE-TEST sm_100 with `PIE_CUDA_NATIVE_MXFP4_MOE=1` and
    /// drop this if it is clean"* — cannot be followed as written: the
    /// variable is gone with the function that read it and the kernels are
    /// gone with the vendored tree.
    ///
    /// AND THE CHEAPER READING. The quarantine's justification was that the
    /// gate above it answered `true` on Blackwell whether or not Marlin was
    /// built, so without it every sm_100 deployment served gpt-oss as garbage
    /// by default. That gate answers `false` unconditionally now and the
    /// lowering has no kernels behind it, so the failure mode the quarantine
    /// guarded is unreachable by construction. **What is open is not a risk;
    /// it is an unmeasured fact about hardware nobody here has** — which is
    /// why this is a doc comment on an assertion rather than an `#[ignore]`d
    /// test pretending it could ever run.
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

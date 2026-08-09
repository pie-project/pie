//! What this driver tells the loader, and the plan it gets back.
//!
//! The C++ (`loader/load_plan.hpp`) reached the Rust loader through the C
//! ABI: open a checkpoint handle, marshal a request, get a marshalled plan.
//! This port calls the same crates — `model` for the author registry,
//! `model-loader` for the compiler — in-process, so the wire structs and
//! their lifetime rules disappear; what remains is what the driver alone
//! knows and must state.
//!
//! That is one thing now. Which backend this is ([`metal_storage_target`], a
//! call into `StorageTarget::for_backend`).
//!
//! It was three. The mask of transforms this driver's kernels implement was
//! stated here AND in `model_loader::plan::passes::tile`, with a test
//! comparing them; the loader keeps it, because the loader is where the
//! consequence lands — it decides which plans compile, and it owns the host
//! fallback every claimed transform must have.
//!
//! The loading policy went the same way. [`compile_load_plan`] here used to
//! state all seven [`model::policy::Policy`] fields, and
//! `driver-cuda`'s copy stated the same seven, differing in exactly two —
//! its own comment said the block was carried "bit for bit". Two copies of a
//! policy is not a spelling problem: a field added to `Policy` gets a
//! considered value on the copy its author was looking at and a `Default` on
//! the other, and both still compile and both still boot. The five shared
//! answers are [`model::boot`]'s now, and what stays here is the two this
//! driver alone knows — named [`Binding::MLX_IN_PLACE`] — plus the checkpoint
//! parse `model::boot` deliberately does not do.

use std::path::Path;

use model::boot::Binding;
use model::policy::Mxfp4MoePolicy;
use model_loader::checkpoint::read::parse_checkpoint_metadata;
use model_loader::plan::{LoadPlan, StorageTarget};
use model_loader::types::BackendKind;

/// This device's storage capability.
///
/// One definition, two readers: the device facts published at create time,
/// and the target supplied with every compile request.
///
/// The alignment, the tile budget and the transform mask were stated here as
/// three constants and stated again in `model_loader::plan::passes::tile`,
/// with a test comparing the masks. They are `StorageTarget::for_backend`'s
/// now — one statement, on the side that owns the consequence: the loader
/// decides which plans compile and owns the host fallback every claimed
/// transform has to have.
#[must_use]
pub fn metal_storage_target() -> StorageTarget {
    StorageTarget::for_backend(BackendKind::Metal, 0, 1)
}

/// The two or three facts a probe states by hand. See
/// [`descriptor_for_testing`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TestFacts {
    /// Decoder layer count, or zero for the family default.
    pub num_hidden_layers: i32,
    /// Trailing layers whose KV is shared with the last full layer.
    pub num_kv_shared_layers: i32,
    /// The config's tie flag; a shipped `lm_head` still beats it.
    pub tied_embeddings: bool,
    /// Affine width, or zero for unquantized.
    pub quant_bits: i32,
    /// Affine group, or zero for unquantized.
    pub quant_group_size: i32,
}

/// Build a `pie.model/1` document from facts assembled by hand.
///
/// **Test and tool scaffolding.** A serving boot forwards the descriptor it
/// was handed; this exists for the probes and numerics tests that point at
/// a checkpoint directory with no descriptor beside it and state the two or
/// three facts their family needs.
///
/// It writes the schema's own field names, which is the only reason it is
/// tolerable to spell a descriptor here at all: a name that drifts from the
/// schema stops being read rather than being read wrong, because
/// `ModelFacts::from_descriptor` takes what it recognizes and leaves the
/// rest at its default. The round-trip through the real reader is pinned by
/// test.
#[must_use]
pub fn descriptor_for_testing(model_type: &str, facts: TestFacts) -> String {
    serde_json::json!({
        "version": "pie.model/1",
        "model_type": model_type,
        "num_hidden_layers": facts.num_hidden_layers,
        "num_kv_shared_layers": facts.num_kv_shared_layers,
        "tie_word_embeddings": facts.tied_embeddings,
        "quant_bits": facts.quant_bits,
        "quant_group_size": facts.quant_group_size,
    })
    .to_string()
}

/// Why a load plan was not produced.
///
/// Two variants, and the split says which side refused. [`Self::Checkpoint`]
/// is this module's own step — reading the snapshot directory, which
/// [`model::boot`] deliberately leaves to its caller. [`Self::Plan`] is
/// everything the shared load path can refuse: the descriptor, the family
/// registry, the compiler, the file check.
#[derive(Debug)]
pub enum LoadPlanError {
    /// The snapshot directory did not read as a checkpoint.
    Checkpoint(String),
    /// The shared load path refused; the value says why.
    Plan(model::boot::LoadPlanError),
}

impl std::fmt::Display for LoadPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Both arms keep the `load plan: ` prefix the four variants carried,
        // because it is what a `Error::Create { what }` message reads as at
        // the boot site.
        match self {
            LoadPlanError::Checkpoint(err) => write!(f, "load plan: {err}"),
            LoadPlanError::Plan(err) => write!(f, "load plan: {err}"),
        }
    }
}

impl std::error::Error for LoadPlanError {}

impl From<model::boot::LoadPlanError> for LoadPlanError {
    fn from(err: model::boot::LoadPlanError) -> Self {
        LoadPlanError::Plan(err)
    }
}

/// Compile the plan: the descriptor in, plan out.
///
/// This driver reads the snapshot directory itself and then hands the shared
/// load path everything else. The two answers it contributes are
/// [`Binding::MLX_IN_PLACE`] — MLX tensor names, projections left as stored
/// — and they are claims about the lowering rather than preferences: the
/// bind path looks up MLX names, and the attention and MLP kernels here read
/// the separate `q`/`k`/`v` tensors, so a fused request would produce
/// operands this driver cannot find.
///
/// Everything else — the other five policy fields, the author call, the
/// plan compile, and the check that every declared file is still on disk at
/// the size the plan states — is [`model::boot::compile_load_plan`]'s, which
/// `driver-cuda` calls with its own [`Binding`]. That is the point: equal
/// requests author equal contracts because there is one policy, not two that
/// happen to agree today.
///
/// The returned [`Mxfp4MoePolicy`] is the author's resolved answer — a family
/// may override the device rule — handed back rather than recomputed, so the
/// bind path cannot disagree with the contract it binds.
///
/// # Errors
///
/// The snapshot directory does not read as a checkpoint, or the shared load
/// path refuses; see [`LoadPlanError`].
pub fn compile_load_plan(
    snapshot_dir: &Path,
    target: &StorageTarget,
    descriptor_json: &str,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    let metadata = parse_checkpoint_metadata(snapshot_dir)
        .map_err(|err| LoadPlanError::Checkpoint(err.to_string()))?;
    Ok(model::boot::compile_load_plan(
        snapshot_dir,
        &metadata,
        target,
        descriptor_json,
        Binding::MLX_IN_PLACE,
    )?)
}

#[cfg(test)]
mod tests {
    use model::facts::ModelFacts;

    use super::*;

    #[test]
    fn the_target_states_the_device_and_nothing_optimistic() {
        let target = metal_storage_target();
        assert_eq!(target.backend, BackendKind::Metal);
        assert_eq!(target.preferred_alignment, 256);
        assert_eq!(target.max_tile_bytes, 64 * 1024 * 1024);
        assert!(!target.native_mxfp4_moe);
        assert_eq!(target.fusion_mask, 0, "no fused transcode kernels here");
    }

    #[test]
    fn a_testing_descriptor_is_read_back_by_the_real_reader() {
        let json = descriptor_for_testing(
            "qwen3",
            TestFacts {
                num_hidden_layers: 28,
                quant_bits: 4,
                quant_group_size: 64,
                ..TestFacts::default()
            },
        );
        let facts = ModelFacts::from_descriptor(json.as_bytes())
            .expect("the helper writes the schema's own field names");
        assert_eq!(facts.model_type, "qwen3");
        // The names did not drift: what the helper wrote is what the reader
        // read, which is the helper's whole justification.
        assert_eq!(facts.num_hidden_layers, 28);
        assert_eq!(facts.quant_bits, 4);
        assert_eq!(facts.quant_group_size, 64);
    }
}

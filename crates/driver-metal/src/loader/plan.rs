//! What this driver tells the loader, and the plan it gets back.
//!
//! The C++ (`loader/load_plan.hpp`) reached the Rust loader through the C
//! ABI: open a checkpoint handle, marshal a request, get a marshalled plan.
//! This port calls the same crates — `model` for the author registry,
//! `model-loader` for the compiler — in-process, so the wire structs and
//! their lifetime rules disappear; what remains is what the driver alone
//! knows and must state.
//!
//! That is two things now. Which backend this is ([`metal_storage_target`],
//! a call into `StorageTarget::for_backend`), and the loading policy that
//! makes equal requests author equal contracts ([`compile_load_plan`] states
//! every field rather than defaulting any).
//!
//! It was three. The mask of transforms this driver's kernels implement was
//! stated here AND in `model_loader::plan::passes::tile`, with a test
//! comparing them; the loader keeps it, because the loader is where the
//! consequence lands — it decides which plans compile, and it owns the host
//! fallback every claimed transform must have.

use std::path::Path;

use model::facts::ModelFacts;
use model::policy::{
    Component, FamilyKnobs, Mxfp4MoePolicy, Mxfp4MoeRequest, Naming, Policy, Projections,
    RuntimeQuant,
};
use model_loader::checkpoint::read::parse_checkpoint_metadata;
use model_loader::plan::{self, LoadPlan, StorageTarget};
use model_loader::types::{BackendKind, DType};

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

/// Did the contract tie the embedding and the head, or ship two tensors?
///
/// Decided ONCE, by the contract, and read back rather than decided a
/// second time. The rule is that a shipped `lm_head` beats whatever the
/// config says, and the config can be wrong in both directions:
/// Qwen3.5-35B-A3B is a multimodal wrapper spelling `tie_word_embeddings`
/// at the TOP level, outside the `text_config` its family parses, so the
/// facts default to tied; Qwen3-0.6B says `tie_word_embeddings: true` and
/// then ships an `lm_head.weight` anyway. Either way the contract staged
/// `embed_tokens` and `lm_head` while the DAG asked for
/// `shared_embedding`, and the load stopped on "unstaged weight
/// shared_embedding.weight" — two opinions about one fact.
///
/// The plan's own tensor list is the only opinion that cannot be wrong in
/// a way the binding survives, so it is the one every family follows.
#[must_use]
pub fn plan_ties_embeddings(plan: &LoadPlan) -> bool {
    plan.tensors
        .iter()
        .any(|tensor| tensor.name == "shared_embedding.weight")
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
#[derive(Debug)]
pub enum LoadPlanError {
    /// The snapshot directory did not read as a checkpoint.
    Checkpoint(String),
    /// The descriptor document was rejected by the facts reader.
    Descriptor(String),
    /// No author claims this `model_type`; the value names it.
    UnknownFamily(String),
    /// The author or the plan compiler refused; the value says why.
    ///
    /// Includes a file the plan declares being absent or the wrong size on
    /// disk: `plan::compile_checked` raises that, and it is a refusal about
    /// the checkpoint rather than a second kind of failure here.
    Compile(String),
}

impl std::fmt::Display for LoadPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadPlanError::Checkpoint(err) => write!(f, "load plan: {err}"),
            LoadPlanError::Descriptor(err) => write!(f, "load plan: descriptor: {err}"),
            LoadPlanError::UnknownFamily(model_type) => write!(
                f,
                "load plan: no author for model_type '{model_type}'; every family \
                 loads through this entry, so an unknown one needs an author in \
                 model::contract"
            ),
            LoadPlanError::Compile(err) => write!(f, "load plan: {err}"),
        }
    }
}

impl std::error::Error for LoadPlanError {}

/// Compile the plan: the descriptor and policy in, plan out.
///
/// The driver sends what only it can know — the compiled model descriptor
/// it was handed, and that this device wants MLX names with in-place
/// projections — and authoring happens in the same `model::contract`
/// registry the CUDA boot goes through. An unknown `model_type` comes back
/// as [`LoadPlanError::UnknownFamily`] naming it, rather than a plan.
///
/// What the policy does NOT carry is anything this driver has no operator
/// knob for: no MXFP4 MoE lowering choice, no component split, no expert
/// streaming, and none of the CUDA per-family environment knobs — zeros
/// and defaults, stated here rather than defaulted there, so equal
/// requests author equal contracts.
///
/// `runtime_quant` is `None` for a reason of its own. The MLX authors do
/// read it (`Int4` encodes a float weight to affine-U4), but this driver
/// binds what the checkpoint holds: a requantization is a decision about an
/// artifact, made once by `pie model build --quant int4` and written down,
/// not one to re-run over every weight on each boot.
///
/// The returned [`Mxfp4MoePolicy`] is the author's resolved answer — a
/// family may override the device rule — handed back rather than recomputed,
/// so the bind path cannot disagree with the contract it binds.
///
/// The C++ called `verify_model` here: a re-author on the far side of the C
/// ABI, holding the *marshalled* plan to the request — marshalling and
/// author determinism both in scope. In-process there is no marshalling and
/// a same-process re-author is a restatement, not a second opinion, so what
/// survives of it is the part that still checks something real: each file
/// the plan declares is stat'ed against the snapshot directory.
pub fn compile_load_plan(
    snapshot_dir: &Path,
    target: &StorageTarget,
    descriptor_json: &str,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    let metadata = parse_checkpoint_metadata(snapshot_dir)
        .map_err(|err| LoadPlanError::Checkpoint(err.to_string()))?;
    let facts = ModelFacts::from_descriptor(descriptor_json.as_bytes())
        .map_err(|err| LoadPlanError::Descriptor(err.to_string()))?;
    let policy = Policy {
        projections: Projections::InPlace, // the MLX lowering
        naming: Naming::Mlx,               // what this bind path reads
        runtime_quant: RuntimeQuant::None,
        moe_request: Mxfp4MoeRequest::Auto,
        component: Component::Full,
        stream_routed_experts: false,
        knobs: FamilyKnobs::default(),
    };
    let (contract, resolved_moe) =
        model::contract::author_with_policy(&facts, &metadata, target, &policy)
            .map_err(|err| LoadPlanError::Compile(err.to_string()))?
            .ok_or_else(|| LoadPlanError::UnknownFamily(facts.model_type.clone()))?;
    let plan = plan::compile(&metadata, &contract, target.clone())
        .map_err(|err| LoadPlanError::Compile(err.to_string()))?;
    // The plan names the files and states their sizes, so a snapshot that
    // moved under a plan compiled against it is the loader's refusal to make.
    // `driver-cuda` carried the same block, bit for bit.
    model_loader::checkpoint::read::verify_declared_files(&plan, snapshot_dir)
        .map_err(|err| LoadPlanError::Compile(err.to_string()))?;
    Ok((plan, resolved_moe))
}

#[cfg(test)]
mod tests {
    use model_loader::types::{Encoding, TensorDecl, TensorId, Visibility};

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

    #[test]
    fn tied_embeddings_are_read_off_the_plan_not_the_config() {
        let mut plan = LoadPlan::empty(metal_storage_target());
        assert!(!plan_ties_embeddings(&plan));
        plan.tensors.push(TensorDecl {
            id: TensorId(0),
            name: "shared_embedding.weight".to_string(),
            shape: vec![32, 8],
            encoding: Encoding::Raw(DType::BF16),
            alignment: 256,
            visibility: Visibility::Public,
        });
        assert!(plan_ties_embeddings(&plan));
    }
}

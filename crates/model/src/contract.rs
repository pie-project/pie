//! The contract aspect's registry: `model_type` in, authored contract out.
//!
//! One row per `model_type`, every N:1 reuse written out. Like
//! `instruct::create`, dispatch is on the *model type*; the family
//! directories only organize the implementations.
//!
//! The rows are a table and not a `match`, so the supported set is a value
//! something can iterate. Two other tables in this repo answer the same
//! question from the other side of the ABI — the CUDA arch table
//! (`crates/driver-cuda/csrc/src/model/registry.cpp`) and Metal's `model_family_of`
//! (`crates/driver-metal/csrc/src/model/facts.hpp`) — and `tests/registry_agreement.rs`
//! reads all three and refuses a disagreement. Before it, a `model_type`
//! added to one side and not the other surfaced as two unrelated errors
//! ("unsupported model_type" from a driver, "no author" from here) whose
//! common cause nothing named.
//!
//! `author` is the whole aspect: a pure function from
//! `(facts, checkpoint, target, policy)` to the contract the loader
//! compiles. Nothing here opens a file or asks a device — both of those are
//! the caller's, which is what lets the same row serve a driver boot, an
//! offline `pie model build`, and a test that authors against a fixture.

use model_loader::checkpoint::CheckpointMetadata;
use model_loader::contract::ModelContract;
use model_loader::error::Error;
use model_loader::plan::StorageTarget;

use crate::builder::Builder;
use crate::facts::ModelFacts;
use crate::policy::{Mxfp4MoePolicy, Naming, Policy};

/// What a row dispatches to: the family's authoring pass.
pub type Author = fn(&mut Builder<'_>) -> Result<(), Error>;

/// The HF-naming rows: what a CUDA boot, `pie model optimize` and the
/// goldens all dispatch through.
///
/// A table rather than a `match` arm so the row set is a *value*. Three
/// tables in this repo answer "which `model_type` is supported" — this one,
/// `crates/driver-cuda/csrc/src/model/registry.cpp`'s arch table, and Metal's
/// `model_family_of` — and `tests/registry_agreement.rs` holds them to each
/// other. That check can only exist if at least this one can be enumerated.
pub const HF_ROWS: &[(&str, Author)] = &[
    // ── llama lineage: dense/GQA decoders sharing one storage schema.
    ("qwen3", crate::llama_3::contract::author_llama_like),
    ("qwen2", crate::llama_3::contract::author_llama_like),
    ("llama", crate::llama_3::contract::author_llama_like),
    ("llama3", crate::llama_3::contract::author_llama_like),
    ("mistral", crate::llama_3::contract::author_llama_like),
    ("mistral3", crate::llama_3::contract::author_dense),
    ("ministral3", crate::llama_3::contract::author_dense),
    ("olmo2", crate::llama_3::contract::author_dense),
    ("olmo3", crate::llama_3::contract::author_dense),
    ("phi3", crate::phi_3::contract::author_phi3),
    // ── Mixtral family: plain Mixtral needs nothing beyond the dense
    //    rules; GPT-OSS is its own author (MXFP4 expert triplets).
    ("mixtral", crate::llama_3::contract::author_dense),
    ("gpt_oss", crate::gpt_oss::contract::author_gpt_oss),
    // ── Gemma-4: nested decoder, router-scale fold, encode scoping.
    ("gemma4", crate::gemma_4::contract::author_gemma4),
    ("gemma4_text", crate::gemma_4::contract::author_gemma4),
    // ── GLM-5.1: FP8 kv_b_proj dequant + per-expert stacks.
    ("glm_moe_dsa", crate::glm_5::contract::author_glm5),
    // ── CSM: fp32 checkpoints, bf16 kernels.
    ("csm", crate::csm::contract::author_csm),
    // ── Qwen3.5 hybrids: GDN blocked shards; the MoE members add the
    //    shared-expert join and per-expert stacks.
    ("qwen3_5", crate::qwen_3_5::contract::author_qwen3_5),
    ("qwen3_5_text", crate::qwen_3_5::contract::author_qwen3_5),
    // Qwen3-VL binds the plain Qwen3 text tower; its contract is the
    // generic dense one, not the hybrid's.
    ("qwen3_vl", crate::llama_3::contract::author_dense),
    ("qwen3_vl_text", crate::llama_3::contract::author_dense),
    ("qwen3_moe", crate::qwen_3_5::contract::author_qwen3_5_moe),
    ("qwen3_5_moe", crate::qwen_3_5::contract::author_qwen3_5_moe),
    ("qwen3_5_moe_text", crate::qwen_3_5::contract::author_qwen3_5_moe),
    // ── Nemotron-H: Mamba2 mixers sharded band-by-band, packed experts.
    ("nemotron_h", crate::nemotron_h::contract::author_nemotron_h),
    // ── MLA lineage: DeepSeek-V2/V3 plain; Kimi-K2 adds the prefix,
    //    the embed/lm_head memory trade, and the W4A16 expert stacks.
    ("deepseek_v2", crate::kimi_k2::contract::author_deepseek_mla),
    ("deepseek_v3", crate::kimi_k2::contract::author_deepseek_mla),
    ("kimi_k2", crate::kimi_k2::contract::author_kimi),
    ("kimi_k3", crate::kimi_k3::contract::author_kimi_k3),
    // ── DeepSeek-V4: own shard rule, MXFP4 expert stacks or groups.
    ("deepseek_v4", crate::deepseek_v4::contract::author_deepseek_v4),
    // ── Gemma 2/3/3n bind the generic dense contract.
    ("gemma2", crate::llama_3::contract::author_dense),
    ("gemma3", crate::llama_3::contract::author_dense),
    ("gemma3_text", crate::llama_3::contract::author_dense),
    ("gemma3n", crate::llama_3::contract::author_dense),
    ("gemma3n_text", crate::llama_3::contract::author_dense),
];

/// The MLX-naming rows: the same families at a different point in policy
/// space, which is what the Metal driver boots through.
///
/// These mirror Metal's `model_family_of` (`crates/driver-metal/csrc/src/model/facts.hpp`)
/// — the driver decides which decode DAG to run from the same string this
/// dispatches an author on, so the two lists have to name the same set.
pub const MLX_ROWS: &[(&str, Author)] = &[
    ("llama", crate::llama_3::contract::author_llama_mlx),
    ("llama3", crate::llama_3::contract::author_llama_mlx),
    ("mistral", crate::llama_3::contract::author_llama_mlx),
    ("qwen2", crate::llama_3::contract::author_llama_mlx),
    ("qwen3", crate::llama_3::contract::author_llama_mlx),
    ("qwen3_moe", crate::llama_3::contract::author_llama_mlx),
    ("qwen2_moe", crate::llama_3::contract::author_llama_mlx),
    ("qwen3_5", crate::qwen_3_5::contract::author_qwen3_5_mlx),
    ("qwen3_5_text", crate::qwen_3_5::contract::author_qwen3_5_mlx),
    ("qwen3_5_moe", crate::qwen_3_5::contract::author_qwen3_5_mlx),
    ("qwen3_5_moe_text", crate::qwen_3_5::contract::author_qwen3_5_mlx),
    ("qwen3_next", crate::qwen_3_5::contract::author_qwen3_5_mlx),
    ("qwen3_next_text", crate::qwen_3_5::contract::author_qwen3_5_mlx),
    ("qwen3_6", crate::qwen_3_5::contract::author_qwen3_5_mlx),
    ("gemma4", crate::gemma_4::contract::author_gemma4_mlx),
    ("gemma4_text", crate::gemma_4::contract::author_gemma4_mlx),
    ("gpt_oss", crate::gpt_oss::contract::author_gpt_oss_mlx),
];

/// The rows this naming dispatches through.
pub fn rows(naming: Naming) -> &'static [(&'static str, Author)] {
    match naming {
        Naming::Mlx => MLX_ROWS,
        Naming::Hf => HF_ROWS,
    }
}

/// Author the load contract for one model type.
///
/// `Ok(None)` when no family here authors this `model_type` yet — during the
/// migration that answer means "ask the C++ author", and afterwards it means
/// "unsupported model".
pub fn author(
    facts: &ModelFacts,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    policy: &Policy,
) -> Result<Option<ModelContract>, Error> {
    Ok(author_with_policy(facts, metadata, target, policy)?.map(|(contract, _)| contract))
}

/// [`author`], also answering how the author resolved the MXFP4 MoE request.
///
/// The bind path branches on the *author's* answer — a family may override
/// the device rule (DeepSeek-V4 does) — so a caller that authored on this
/// side of a boundary has to be handed the answer back rather than
/// recomputing it and disagreeing.
pub fn author_with_policy(
    facts: &ModelFacts,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    policy: &Policy,
) -> Result<Option<(ModelContract, Mxfp4MoePolicy)>, Error> {
    let Some((_, author)) = rows(policy.naming)
        .iter()
        .find(|(model_type, _)| *model_type == facts.model_type)
    else {
        return Ok(None);
    };
    let mut builder = Builder::new(metadata, facts, target, policy);
    author(&mut builder)?;
    let resolved = builder.mxfp4_moe();
    builder.finish().map(|contract| Some((contract, resolved)))
}

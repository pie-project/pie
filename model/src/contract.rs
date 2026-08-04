//! The contract aspect's registry: `model_type` in, authored contract out.
//!
//! The rows mirror `driver/cuda/src/model/registry.cpp` while the migration
//! runs — one row per `model_type`, every N:1 reuse written out — and shrink
//! that file as families move here. Like `instruct::create`, the match
//! dispatches on the *model type*; the family directories only organize the
//! implementations.
//!
//! `author` is the whole aspect: a pure function from
//! `(facts, checkpoint, target, policy)` to the contract the loader
//! compiles. Nothing here opens a file or asks a device — both of those are
//! the caller's, which is what lets the same row serve a driver boot, an
//! offline `pie model optimize`, and a test that authors against a fixture.

use pie_loader::checkpoint::CheckpointMetadata;
use pie_loader::contract::ModelContract;
use pie_loader::error::Error;
use pie_loader::plan::StorageTarget;

use crate::common::builder::Builder;
use crate::common::facts::ModelFacts;
use crate::common::policy::{Mxfp4MoePolicy, Policy};

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
    // The Metal lowering: same families, a different point in policy space.
    // The rows mirror each Metal schema's `is_supported_model_type` list.
    if policy.naming == crate::common::policy::Naming::Mlx {
        let author = match facts.model_type.as_str() {
            "llama" | "llama3" | "mistral" | "qwen2" | "qwen3" | "qwen3_moe" | "qwen2_moe" => {
                crate::llama::contract::author_llama_mlx
            }
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_next"
            | "qwen3_next_text" | "qwen3_6" => crate::qwen3_5::contract::author_qwen3_5_mlx,
            "gemma4" | "gemma4_text" => crate::gemma::contract::author_gemma4_mlx,
            "gpt_oss" => crate::gptoss::contract::author_gpt_oss_mlx,
            _ => return Ok(None),
        };
        let mut builder = Builder::new(metadata, facts, target, policy);
        author(&mut builder)?;
        let resolved = builder.mxfp4_moe();
        return builder.finish().map(|contract| Some((contract, resolved)));
    }

    let author = match facts.model_type.as_str() {
        // ── llama lineage: dense/GQA decoders sharing one storage schema.
        "qwen3" | "qwen2" | "llama" | "llama3" | "mistral" => {
            crate::llama::contract::author_llama_like
        }
        "mistral3" | "ministral3" | "olmo2" | "olmo3" => crate::llama::contract::author_dense,
        "phi3" => crate::llama::contract::author_phi3,
        // ── Mixtral family: plain Mixtral needs nothing beyond the dense
        //    rules; GPT-OSS is its own author (MXFP4 expert triplets).
        "mixtral" => crate::llama::contract::author_dense,
        "gpt_oss" => crate::gptoss::contract::author_gpt_oss,
        // ── Gemma-4: nested decoder, router-scale fold, encode scoping.
        "gemma4" | "gemma4_text" => crate::gemma::contract::author_gemma4,
        // ── GLM-5.1: FP8 kv_b_proj dequant + per-expert stacks.
        "glm_moe_dsa" => crate::glm5::contract::author_glm5,
        // ── CSM: fp32 checkpoints, bf16 kernels.
        "csm" => crate::csm::contract::author_csm,
        // ── Qwen3.5 hybrids: GDN blocked shards; the MoE members add the
        //    shared-expert join and per-expert stacks.
        "qwen3_5" | "qwen3_5_text" => crate::qwen3_5::contract::author_qwen3_5,
        // Qwen3-VL binds the plain Qwen3 text tower; its contract is the
        // generic dense one, not the hybrid's.
        "qwen3_vl" | "qwen3_vl_text" => crate::llama::contract::author_dense,
        "qwen3_moe" | "qwen3_5_moe" | "qwen3_5_moe_text" => {
            crate::qwen3_5::contract::author_qwen3_5_moe
        }
        // ── Nemotron-H: Mamba2 mixers sharded band-by-band, packed experts.
        "nemotron_h" => crate::nemotron_h::contract::author_nemotron_h,
        // ── MLA lineage: DeepSeek-V2/V3 plain; Kimi-K2 adds the prefix,
        //    the embed/lm_head memory trade, and the W4A16 expert stacks.
        "deepseek_v2" | "deepseek_v3" => crate::kimi::contract::author_deepseek_mla,
        "kimi_k2" => crate::kimi::contract::author_kimi,
        "kimi_k3" => crate::kimi::contract_k3::author_kimi_k3,
        // ── DeepSeek-V4: own shard rule, MXFP4 expert stacks or groups.
        "deepseek_v4" => crate::deepseek_v4::contract::author_deepseek_v4,
        // ── Gemma 2/3/3n bind the generic dense contract.
        "gemma2" | "gemma3" | "gemma3_text" | "gemma3n" | "gemma3n_text" => {
            crate::llama::contract::author_dense
        }
        _ => return Ok(None),
    };
    let mut builder = Builder::new(metadata, facts, target, policy);
    author(&mut builder)?;
    let resolved = builder.mxfp4_moe();
    builder.finish().map(|contract| Some((contract, resolved)))
}

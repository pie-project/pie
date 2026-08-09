//! Reading `pie.model/1`, the compiled model descriptor — gate-hf-config's
//! read side.
//!
//! Ports `model/descriptor.cpp`. The descriptor is exhaustive by
//! construction — `config.json`'s quirks were normalized once, in Rust, at
//! `pie model convert` — so a missing key is a bug in the WRITER rather
//! than a case to default around, and this reader refuses instead of
//! zero-filling ("reading it as zero would turn that bug into a model that
//! loads and answers wrongly").
//!
//! serde could deserialize [`HfConfig`] directly, and deliberately does
//! not: the C++ reader's behaviour is the contract — which keys are
//! demanded, which sub-configs may be null or absent, the version refusal's
//! exact message — and a hand walk states it where a derive would bury it.
//!
//! `head_dim_kernel` is the one field not read from the descriptor: it
//! rounds `head_dim` up to a head dim THIS BUILD instantiated
//! (`kernels.def`), so it is a build property recomputed here, not a fact
//! about the checkpoint.

use serde_json::Value;

use crate::config::schema::{
    CsmConfig, CsmDepthDecoderConfig, GemmaAudioConfig, GemmaVisionConfig,
    HfConfig, MimiCodecConfig, Qwen3VLVisionConfig, RopeScaling,
};

/// Why the reader refused, carrying the C++ `what()` text verbatim — the
/// message is part of the behaviour the parity transcript pins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorError(pub String);

impl std::fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DescriptorError {}

type Result<T> = std::result::Result<T, DescriptorError>;

/// The attention head dims this build instantiates — `kernels.def`'s
/// `PIE_ATTN_HEAD_DIM` rows. A test in `tests/hf_descriptor_parity.rs`
/// parses the `.def` and asserts this list has not drifted.
pub const ATTN_HEAD_DIMS: &[i32] = &[64, 128, 256, 512];

/// Smallest instantiated head_dim that can hold `head_dim`, or `head_dim`
/// itself when none can — callers then surface the dispatch error rather
/// than silently mis-sizing. Ports `round_up_attn_head_dim`
/// (`kernels_manifest.hpp`).
#[must_use]
pub fn round_up_attn_head_dim(head_dim: i32) -> i32 {
    let mut best = 0;
    for &hd in ATTN_HEAD_DIMS {
        if hd >= head_dim && (best == 0 || hd < best) {
            best = hd;
        }
    }
    if best != 0 { best } else { head_dim }
}

fn missing(key: &str) -> DescriptorError {
    DescriptorError(format!("pie.model/1 descriptor: missing key '{key}'"))
}

fn need<'a>(j: &'a Value, key: &str) -> Result<&'a Value> {
    j.get(key).ok_or_else(|| missing(key))
}

fn wrong_type(key: &str, wanted: &str) -> DescriptorError {
    DescriptorError(format!(
        "pie.model/1 descriptor: key '{key}' is not {wanted}"
    ))
}

fn need_i32(j: &Value, key: &str) -> Result<i32> {
    need(j, key)?
        .as_i64()
        .and_then(|v| i32::try_from(v).ok())
        .ok_or_else(|| wrong_type(key, "an integer"))
}

/// The C++ reads floats through `get<float>` — a JSON double narrowed to
/// `float`. `as f32` is that same conversion.
#[allow(clippy::cast_possible_truncation)]
fn need_f32(j: &Value, key: &str) -> Result<f32> {
    need(j, key)?
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| wrong_type(key, "a number"))
}

fn need_bool(j: &Value, key: &str) -> Result<bool> {
    need(j, key)?.as_bool().ok_or_else(|| wrong_type(key, "a bool"))
}

fn need_string(j: &Value, key: &str) -> Result<String> {
    Ok(need(j, key)?
        .as_str()
        .ok_or_else(|| wrong_type(key, "a string"))?
        .to_owned())
}

fn need_vec_i32(j: &Value, key: &str) -> Result<Vec<i32>> {
    need(j, key)?
        .as_array()
        .ok_or_else(|| wrong_type(key, "an array"))?
        .iter()
        .map(|v| {
            v.as_i64()
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| wrong_type(key, "an integer array"))
        })
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn need_vec_f32(j: &Value, key: &str) -> Result<Vec<f32>> {
    need(j, key)?
        .as_array()
        .ok_or_else(|| wrong_type(key, "an array"))?
        .iter()
        .map(|v| {
            v.as_f64().map(|v| v as f32).ok_or_else(|| wrong_type(key, "a number array"))
        })
        .collect()
}

fn need_vec_string(j: &Value, key: &str) -> Result<Vec<String>> {
    need(j, key)?
        .as_array()
        .ok_or_else(|| wrong_type(key, "an array"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| wrong_type(key, "a string array"))
        })
        .collect()
}

fn rope_scaling_of(name: &str) -> Result<RopeScaling> {
    match name {
        "none" => Ok(RopeScaling::None),
        "llama3" => Ok(RopeScaling::Llama3),
        "original_yarn" => Ok(RopeScaling::OriginalYarn),
        other => Err(DescriptorError(format!(
            "pie.model/1 descriptor: unknown rope scaling {other}"
        ))),
    }
}

fn read_gemma_vision(j: &Value) -> Result<GemmaVisionConfig> {
    Ok(GemmaVisionConfig {
        hidden_size: need_i32(j, "hidden_size")?,
        intermediate_size: need_i32(j, "intermediate_size")?,
        num_hidden_layers: need_i32(j, "num_hidden_layers")?,
        num_attention_heads: need_i32(j, "num_attention_heads")?,
        num_key_value_heads: need_i32(j, "num_key_value_heads")?,
        head_dim: need_i32(j, "head_dim")?,
        patch_size: need_i32(j, "patch_size")?,
        rms_norm_eps: need_f32(j, "rms_norm_eps")?,
        rope_theta: need_f32(j, "rope_theta")?,
        pooling_kernel_size: need_i32(j, "pooling_kernel_size")?,
        soft_tokens_per_image: need_i32(j, "soft_tokens_per_image")?,
        use_clipped_linears: need_bool(j, "use_clipped_linears")?,
    })
}

fn read_gemma_audio(j: &Value) -> Result<GemmaAudioConfig> {
    Ok(GemmaAudioConfig {
        hidden_size: need_i32(j, "hidden_size")?,
        num_attention_heads: need_i32(j, "num_attention_heads")?,
        num_hidden_layers: need_i32(j, "num_hidden_layers")?,
        conv_kernel_size: need_i32(j, "conv_kernel_size")?,
        subsampling_conv_channels0: need_i32(j, "subsampling_conv_channels0")?,
        subsampling_conv_channels1: need_i32(j, "subsampling_conv_channels1")?,
        output_proj_dims: need_i32(j, "output_proj_dims")?,
        attention_chunk_size: need_i32(j, "attention_chunk_size")?,
        attention_context_left: need_i32(j, "attention_context_left")?,
        attention_context_right: need_i32(j, "attention_context_right")?,
        feature_size: need_i32(j, "feature_size")?,
        attention_logit_cap: need_f32(j, "attention_logit_cap")?,
        residual_weight: need_f32(j, "residual_weight")?,
        rms_norm_eps: need_f32(j, "rms_norm_eps")?,
        use_clipped_linears: need_bool(j, "use_clipped_linears")?,
    })
}

fn read_csm_depth(j: &Value) -> Result<CsmDepthDecoderConfig> {
    Ok(CsmDepthDecoderConfig {
        hidden_size: need_i32(j, "hidden_size")?,
        backbone_hidden_size: need_i32(j, "backbone_hidden_size")?,
        num_hidden_layers: need_i32(j, "num_hidden_layers")?,
        num_attention_heads: need_i32(j, "num_attention_heads")?,
        num_key_value_heads: need_i32(j, "num_key_value_heads")?,
        head_dim: need_i32(j, "head_dim")?,
        intermediate_size: need_i32(j, "intermediate_size")?,
        num_codebooks: need_i32(j, "num_codebooks")?,
        vocab_size: need_i32(j, "vocab_size")?,
        max_position_embeddings: need_i32(j, "max_position_embeddings")?,
        rms_norm_eps: need_f32(j, "rms_norm_eps")?,
        rope_theta: need_f32(j, "rope_theta")?,
        rope_factor: need_f32(j, "rope_factor")?,
        rope_low_freq_factor: need_f32(j, "rope_low_freq_factor")?,
        rope_high_freq_factor: need_f32(j, "rope_high_freq_factor")?,
        rope_original_max_position: need_i32(j, "rope_original_max_position")?,
    })
}

fn read_mimi(j: &Value) -> Result<MimiCodecConfig> {
    Ok(MimiCodecConfig {
        hidden_size: need_i32(j, "hidden_size")?,
        codebook_dim: need_i32(j, "codebook_dim")?,
        codebook_size: need_i32(j, "codebook_size")?,
        num_quantizers: need_i32(j, "num_quantizers")?,
        num_semantic_quantizers: need_i32(j, "num_semantic_quantizers")?,
        num_filters: need_i32(j, "num_filters")?,
        upsampling_ratios: need_vec_i32(j, "upsampling_ratios")?,
        xf_num_attention_heads: need_i32(j, "xf_num_attention_heads")?,
        xf_num_key_value_heads: need_i32(j, "xf_num_key_value_heads")?,
        xf_head_dim: need_i32(j, "xf_head_dim")?,
        xf_intermediate_size: need_i32(j, "xf_intermediate_size")?,
        xf_num_hidden_layers: need_i32(j, "xf_num_hidden_layers")?,
        xf_sliding_window: need_i32(j, "xf_sliding_window")?,
        xf_rope_theta: need_f32(j, "xf_rope_theta")?,
        norm_eps: need_f32(j, "norm_eps")?,
        sampling_rate: need_i32(j, "sampling_rate")?,
        use_causal_conv: need_bool(j, "use_causal_conv")?,
        upsample_groups: need_i32(j, "upsample_groups")?,
        residual_kernel_size: need_i32(j, "residual_kernel_size")?,
        kernel_size: need_i32(j, "kernel_size")?,
        last_kernel_size: need_i32(j, "last_kernel_size")?,
    })
}

fn read_csm(j: &Value) -> Result<CsmConfig> {
    Ok(CsmConfig {
        text_vocab_size: need_i32(j, "text_vocab_size")?,
        audio_vocab_size: need_i32(j, "audio_vocab_size")?,
        num_codebooks: need_i32(j, "num_codebooks")?,
        codebook_eos_token_id: need_i32(j, "codebook_eos_token_id")?,
        audio_eos_token_id: need_i32(j, "audio_eos_token_id")?,
        audio_token_id: need_i32(j, "audio_token_id")?,
        depth: read_csm_depth(need(j, "depth")?)?,
        codec: read_mimi(need(j, "codec")?)?,
    })
}

fn read_qwen3_vl_vision(j: &Value) -> Result<Qwen3VLVisionConfig> {
    Ok(Qwen3VLVisionConfig {
        hidden_size: need_i32(j, "hidden_size")?,
        intermediate_size: need_i32(j, "intermediate_size")?,
        depth: need_i32(j, "depth")?,
        num_heads: need_i32(j, "num_heads")?,
        patch_size: need_i32(j, "patch_size")?,
        temporal_patch_size: need_i32(j, "temporal_patch_size")?,
        spatial_merge_size: need_i32(j, "spatial_merge_size")?,
        in_channels: need_i32(j, "in_channels")?,
        out_hidden_size: need_i32(j, "out_hidden_size")?,
        num_position_embeddings: need_i32(j, "num_position_embeddings")?,
        deepstack_visual_indexes: need_vec_i32(j, "deepstack_visual_indexes")?,
    })
}

/// A sub-config that may be absent OR null — both mean "not present".
fn optional<'a>(j: &'a Value, key: &str) -> Option<&'a Value> {
    match j.get(key) {
        Some(v) if !v.is_null() => Some(v),
        _ => None,
    }
}

#[allow(clippy::too_many_lines)]
fn read_config(j: &Value) -> Result<HfConfig> {
    Ok(HfConfig {
        arch_name: need_string(j, "arch_name")?,
        model_type: need_string(j, "model_type")?,
        hidden_size: need_i32(j, "hidden_size")?,
        intermediate_size: need_i32(j, "intermediate_size")?,
        num_hidden_layers: need_i32(j, "num_hidden_layers")?,
        num_attention_heads: need_i32(j, "num_attention_heads")?,
        num_key_value_heads: need_i32(j, "num_key_value_heads")?,
        head_dim: need_i32(j, "head_dim")?,
        // Recomputed by the caller; see the module docs.
        head_dim_kernel: 0,
        vocab_size: need_i32(j, "vocab_size")?,
        max_position_embeddings: need_i32(j, "max_position_embeddings")?,
        rms_norm_eps: need_f32(j, "rms_norm_eps")?,
        hidden_act: need_string(j, "hidden_act")?,
        mlp_hidden_act: need_string(j, "mlp_hidden_act")?,
        rope_theta: need_f32(j, "rope_theta")?,
        rope_scaling_kind: rope_scaling_of(&need_string(j, "rope_scaling_kind")?)?,
        rope_factor: need_f32(j, "rope_factor")?,
        rope_low_freq_factor: need_f32(j, "rope_low_freq_factor")?,
        rope_high_freq_factor: need_f32(j, "rope_high_freq_factor")?,
        rope_original_max_position: need_i32(j, "rope_original_max_position")?,
        rope_beta_fast: need_f32(j, "rope_beta_fast")?,
        rope_beta_slow: need_f32(j, "rope_beta_slow")?,
        rope_attention_factor: need_f32(j, "rope_attention_factor")?,
        rope_mla_softmax_mscale: need_f32(j, "rope_mla_softmax_mscale")?,
        has_rope_scaling: need_bool(j, "has_rope_scaling")?,
        sliding_window: need_i32(j, "sliding_window")?,
        layer_types: need_vec_string(j, "layer_types")?,
        rope_local_base_freq: need_f32(j, "rope_local_base_freq")?,
        tie_word_embeddings: need_bool(j, "tie_word_embeddings")?,
        attention_bias: need_bool(j, "attention_bias")?,
        use_qk_norm: need_bool(j, "use_qk_norm")?,
        num_experts: need_i32(j, "num_experts")?,
        num_experts_per_tok: need_i32(j, "num_experts_per_tok")?,
        gemma4_enable_moe: need_bool(j, "gemma4_enable_moe")?,
        gemma4_attention_k_eq_v: need_bool(j, "gemma4_attention_k_eq_v")?,
        gemma4_num_global_key_value_heads: need_i32(j, "gemma4_num_global_key_value_heads")?,
        gemma4_global_head_dim: need_i32(j, "gemma4_global_head_dim")?,
        gemma4_double_wide_mlp: need_bool(j, "gemma4_double_wide_mlp")?,
        swiglu_limit: need_f32(j, "swiglu_limit")?,
        mlp_has_bias: need_bool(j, "mlp_has_bias")?,
        router_has_bias: need_bool(j, "router_has_bias")?,
        attention_has_sinks: need_bool(j, "attention_has_sinks")?,
        gemma_query_pre_attn_scalar: need_f32(j, "gemma_query_pre_attn_scalar")?,
        gemma_final_logit_softcap: need_f32(j, "gemma_final_logit_softcap")?,
        gemma_attn_logit_softcap: need_f32(j, "gemma_attn_logit_softcap")?,
        gemma_hidden_size_per_layer_input: need_i32(j, "gemma_hidden_size_per_layer_input")?,
        num_kv_shared_layers: need_i32(j, "num_kv_shared_layers")?,
        gemma4_use_ordered_embeddings: need_bool(j, "gemma4_use_ordered_embeddings")?,
        gemma4_num_centroids: need_i32(j, "gemma4_num_centroids")?,
        gemma4_centroid_intermediate_top_k: need_i32(j, "gemma4_centroid_intermediate_top_k")?,
        gemma_per_layer_rope_theta: need_vec_f32(j, "gemma_per_layer_rope_theta")?,
        gemma_per_layer_partial_rotary_factor: need_vec_f32(
            j,
            "gemma_per_layer_partial_rotary_factor",
        )?,
        moe_intermediate_size: need_i32(j, "moe_intermediate_size")?,
        shared_expert_intermediate_size: need_i32(j, "shared_expert_intermediate_size")?,
        routed_scaling_factor: need_f32(j, "routed_scaling_factor")?,
        n_group: need_i32(j, "n_group")?,
        topk_group: need_i32(j, "topk_group")?,
        norm_topk_prob: need_bool(j, "norm_topk_prob")?,
        mamba_num_heads: need_i32(j, "mamba_num_heads")?,
        mamba_head_dim: need_i32(j, "mamba_head_dim")?,
        mamba_state_size: need_i32(j, "mamba_state_size")?,
        mamba_n_groups: need_i32(j, "mamba_n_groups")?,
        mamba_conv_kernel: need_i32(j, "mamba_conv_kernel")?,
        mamba_chunk_size: need_i32(j, "mamba_chunk_size")?,
        mamba_time_step_min: need_f32(j, "mamba_time_step_min")?,
        q_lora_rank: need_i32(j, "q_lora_rank")?,
        kv_lora_rank: need_i32(j, "kv_lora_rank")?,
        qk_nope_head_dim: need_i32(j, "qk_nope_head_dim")?,
        qk_rope_head_dim: need_i32(j, "qk_rope_head_dim")?,
        v_head_dim: need_i32(j, "v_head_dim")?,
        first_k_dense_replace: need_i32(j, "first_k_dense_replace")?,
        n_shared_experts: need_i32(j, "n_shared_experts")?,
        dsv4_o_lora_rank: need_i32(j, "dsv4_o_lora_rank")?,
        dsv4_o_groups: need_i32(j, "dsv4_o_groups")?,
        dsv4_index_head_dim: need_i32(j, "dsv4_index_head_dim")?,
        dsv4_index_n_heads: need_i32(j, "dsv4_index_n_heads")?,
        dsv4_index_topk: need_i32(j, "dsv4_index_topk")?,
        dsv4_hc_mult: need_i32(j, "dsv4_hc_mult")?,
        dsv4_num_hash_layers: need_i32(j, "dsv4_num_hash_layers")?,
        dsv4_sliding_window: need_i32(j, "dsv4_sliding_window")?,
        dsv4_hc_eps: need_f32(j, "dsv4_hc_eps")?,
        dsv4_compress_rope_theta: need_f32(j, "dsv4_compress_rope_theta")?,
        dsv4_compress_ratios: need_vec_i32(j, "dsv4_compress_ratios")?,
        dsv4_scoring_func: need_string(j, "dsv4_scoring_func")?,
        dsv4_expert_dtype: need_string(j, "dsv4_expert_dtype")?,
        linear_num_value_heads: need_i32(j, "linear_num_value_heads")?,
        linear_num_key_heads: need_i32(j, "linear_num_key_heads")?,
        linear_key_head_dim: need_i32(j, "linear_key_head_dim")?,
        linear_value_head_dim: need_i32(j, "linear_value_head_dim")?,
        linear_conv_kernel_dim: need_i32(j, "linear_conv_kernel_dim")?,
        attn_output_gate: need_bool(j, "attn_output_gate")?,
        partial_rotary_factor: need_f32(j, "partial_rotary_factor")?,
        mtp_num_hidden_layers: need_i32(j, "mtp_num_hidden_layers")?,
        mtp_use_dedicated_embeddings: need_bool(j, "mtp_use_dedicated_embeddings")?,
        altup_num_inputs: need_i32(j, "altup_num_inputs")?,
        altup_active_idx: need_i32(j, "altup_active_idx")?,
        altup_correct_scale: need_bool(j, "altup_correct_scale")?,
        altup_coef_clip: need_f32(j, "altup_coef_clip")?,
        laurel_rank: need_i32(j, "laurel_rank")?,
        vocab_size_per_layer_input: need_i32(j, "vocab_size_per_layer_input")?,
        gemma3n_rope_local_base_freq: need_f32(j, "gemma3n_rope_local_base_freq")?,
        gemma3n_per_layer_intermediate: need_vec_i32(j, "gemma3n_per_layer_intermediate")?,
        gemma3n_activation_sparsity: need_vec_f32(j, "gemma3n_activation_sparsity")?,
        kda_gate_lower_bound: need_f32(j, "kda_gate_lower_bound")?,
        kda_full_rank_gate: need_bool(j, "kda_full_rank_gate")?,
        mla_use_nope: need_bool(j, "mla_use_nope")?,
        mla_output_gate: need_bool(j, "mla_output_gate")?,
        routed_expert_hidden_size: need_i32(j, "routed_expert_hidden_size")?,
        latent_moe_use_norm: need_bool(j, "latent_moe_use_norm")?,
        attn_res_block_size: need_i32(j, "attn_res_block_size")?,
        situ_beta: need_f32(j, "situ_beta")?,
        situ_linear_beta: need_f32(j, "situ_linear_beta")?,
        moe_router_activation_func: need_string(j, "moe_router_activation_func")?,
        index_topk: need_i32(j, "index_topk")?,
        index_head_dim: need_i32(j, "index_head_dim")?,
        index_n_heads: need_i32(j, "index_n_heads")?,
        indexer_types: need_vec_string(j, "indexer_types")?,
        torch_dtype: need_string(j, "torch_dtype")?,
        quant_method: need_string(j, "quant_method")?,
        quant_bits: need_i32(j, "quant_bits")?,
        quant_group_size: need_i32(j, "quant_group_size")?,
        quant_desc_act: need_bool(j, "quant_desc_act")?,
        quant_sym: need_bool(j, "quant_sym")?,
        quant_zero_point: need_bool(j, "quant_zero_point")?,
        kv_cache_scheme_present: need_bool(j, "kv_cache_scheme_present")?,
        mm_lm_strip_prefix: need_string(j, "mm_lm_strip_prefix")?,
        mm_skip_prefixes: need_vec_string(j, "mm_skip_prefixes")?,
        gemma_vision: optional(j, "gemma_vision").map(read_gemma_vision).transpose()?,
        gemma_audio: optional(j, "gemma_audio").map(read_gemma_audio).transpose()?,
        qwen3_vl_vision: optional(j, "qwen3_vl_vision")
            .map(read_qwen3_vl_vision)
            .transpose()?,
        csm: optional(j, "csm").map(read_csm).transpose()?,
        qwen3_vl_mrope_section: need_vec_i32(j, "qwen3_vl_mrope_section")?,
        qwen3_vl_mrope_interleaved: need_bool(j, "qwen3_vl_mrope_interleaved")?,
        qwen3_vl_image_token_id: need_i32(j, "qwen3_vl_image_token_id")?,
        qwen3_vl_vision_start_token_id: need_i32(j, "qwen3_vl_vision_start_token_id")?,
        qwen3_vl_vision_end_token_id: need_i32(j, "qwen3_vl_vision_end_token_id")?,
    })
}

/// Parse a `pie.model/1` descriptor into the same struct the normalizer
/// wrote out. Refuses a version this build does not read, or a field the
/// descriptor should carry and does not.
///
/// Ports `parse_pie_model_descriptor`.
pub fn parse_pie_model_descriptor(json: &str) -> Result<HfConfig> {
    let j: Value = serde_json::from_str(json)
        .map_err(|e| DescriptorError(format!("pie.model/1 descriptor: {e}")))?;
    let version = need_string(&j, "version")?;
    if version != "pie.model/1" {
        return Err(DescriptorError(format!(
            "this artifact's model descriptor is {version}, and this build \
             reads pie.model/1; regenerate it with `pie model convert --force`"
        )));
    }
    let mut cfg = read_config(&j)?;
    // Not a fact about the checkpoint: the set of instantiated head dims is
    // a property of this build (kernels.def), so it is derived here.
    cfg.head_dim_kernel = round_up_attn_head_dim(cfg.head_dim);
    Ok(cfg)
}

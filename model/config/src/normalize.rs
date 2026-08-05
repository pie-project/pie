//! HuggingFace `config.json` → the normalized shape, once, at import.
//!
//! A transliteration of `parse_hf_config`
//! (`driver/cuda/src/model/config.cpp`), in the same order, with the same
//! rules. Not a reimplementation: the order matters, because several fields
//! are written more than once and the last write wins, and the defaults matter
//! because most of them are "what HF omits when it means the common case".
//!
//! Two consequences of being a transliteration rather than a rewrite:
//!
//! - **Bugs are reproduced, not fixed.** `attention_has_sinks` is set to `true`
//!   inside the `deepseek_v4` block and then overwritten unconditionally a few
//!   lines later, so DeepSeek-V4 loses its sinks. That is what this does too.
//!   Fixing it is a deliberate change to both sides at once, with the goldens
//!   regenerated — not something the port decides on its own.
//! - **Field order in the source is load-bearing.** `norm_topk_prob` has an
//!   in-class default of `true` and is then read with a default of `false`,
//!   so the effective default is `false`. Reordering to group the MoE fields
//!   would quietly change that.
//!
//! `head_dim_kernel` is the one field here that is not a fact about the
//! checkpoint: it rounds `head_dim` up to a head dim the *driver build*
//! instantiated. It is computed for the differential's sake and excluded from
//! the descriptor `pie.model/1` writes.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::json::*;
use crate::schema::*;

/// Model types whose query/key width is the sum of the two MLA halves.
const MLA_HEAD_DIM_ARCHS: [&str; 5] = [
    "kimi_k2",
    "deepseek_v2",
    "deepseek_v3",
    "glm_moe_dsa",
    "kimi_k3",
];

/// Model types whose HF config omits `tie_word_embeddings` but means `true`.
const TIE_BY_DEFAULT: [&str; 8] = [
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
    "gemma4",
    "gemma4_text",
    "gemma3n",
    "gemma3n_text",
];

/// The head dims the CUDA attention kernels are instantiated for
/// (`driver/cuda/src/kernels.def`).
const ATTN_HEAD_DIMS: [i32; 4] = [64, 128, 256, 512];

/// Smallest instantiated head dim that can hold `head_dim`, else `head_dim`.
///
/// Mirrors `round_up_attn_head_dim`. Kept here only so the differential can
/// compare the field; it is a property of a driver build, not of a checkpoint,
/// and never reaches an artifact.
fn round_up_attn_head_dim(head_dim: i32) -> i32 {
    ATTN_HEAD_DIMS
        .iter()
        .copied()
        .filter(|&hd| hd >= head_dim)
        .min()
        .unwrap_or(head_dim)
}

/// `use_qk_norm`: explicit if present, else inferred from the model type.
///
/// Qwen3 RMS-normalises q and k per head; llama, mistral and qwen2 do not.
/// There is no `config.json` key for it — it is implied by the architecture.
/// This list must stay in step with the driver's own inference
/// (`driver/metal/src/model_facts.cpp`, `ll_qk_norm`) and with the family
/// table in `driver/metal/src/model/facts.hpp`; a checkpoint that silently
/// runs without q/k norm produces fluent, plausible, wrong tokens.
fn infer_qk_norm(model_type: &str, text: &Value, path: &str) -> Result<bool> {
    if text.get("use_qk_norm").is_some() {
        return opt_bool(text, "use_qk_norm", false, path);
    }
    Ok(matches!(
        model_type,
        "qwen3"
            | "qwen3_moe"
            | "qwen3_5"
            | "qwen3_5_text"
            | "qwen3_5_moe"
            | "qwen3_5_moe_text"
            | "qwen3_next"
            | "qwen3_next_text"
            | "qwen3_6"
    ))
}

/// Normalizes a parsed `config.json`.
///
/// `path` appears in error messages only.
pub fn normalize(root: &Value, path: &str) -> Result<HfConfig> {
    let mut cfg = HfConfig::default();

    let architectures = match root.get("architectures") {
        Some(Value::Array(items)) if !items.is_empty() => items,
        _ => bail!("config.json: missing or empty 'architectures'"),
    };
    cfg.arch_name = match &architectures[0] {
        Value::String(name) => name.clone(),
        _ => bail!("config.json: 'architectures[0]' is not a string"),
    };

    // Both levels of the config, and both `model_type`s, resolved once.
    let j = text_config_view(root);
    let outer_model_type = model_type(root);
    let text_model_type = model_type(j);
    cfg.model_type = if text_model_type.is_empty() {
        outer_model_type.clone()
    } else {
        text_model_type
    };
    // Kimi-K3's text tower calls itself `kimi_linear`, which is also the
    // model_type of the standalone Kimi-Linear checkpoints — a different
    // architecture. The outer shell is the only thing that tells them apart.
    if outer_model_type == "kimi_k3" {
        cfg.model_type = "kimi_k3".to_string();
    }

    cfg.hidden_size = require_i32(j, "hidden_size", path)?;
    // Normally a scalar; Gemma-3n stores a per-layer list. Take either, and
    // mirror the list's first element into the scalar.
    match opt_i32_array(j, "intermediate_size", path)? {
        Some(list) if list.is_empty() => {
            bail!("config.json ({path}): intermediate_size list is empty")
        }
        Some(list) => {
            cfg.intermediate_size = list[0];
            cfg.gemma3n_per_layer_intermediate = list;
        }
        // Pure-MoE configs omit it and use the MoE-specific widths instead.
        None => cfg.intermediate_size = opt_i32(j, "intermediate_size", 0, path)?,
    }
    cfg.num_hidden_layers = require_i32(j, "num_hidden_layers", path)?;
    cfg.num_attention_heads = require_i32(j, "num_attention_heads", path)?;
    cfg.num_key_value_heads = opt_i32(j, "num_key_value_heads", cfg.num_attention_heads, path)?;
    let implied_head_dim = cfg
        .hidden_size
        .checked_div(cfg.num_attention_heads)
        .unwrap_or(0);
    cfg.head_dim = opt_i32(j, "head_dim", implied_head_dim, path)?;
    cfg.q_lora_rank = opt_i32(j, "q_lora_rank", 0, path)?;
    cfg.kv_lora_rank = opt_i32(j, "kv_lora_rank", 0, path)?;
    cfg.qk_nope_head_dim = opt_i32(j, "qk_nope_head_dim", 0, path)?;
    cfg.qk_rope_head_dim = opt_i32(j, "qk_rope_head_dim", 0, path)?;
    cfg.v_head_dim = opt_i32(j, "v_head_dim", 0, path)?;
    if MLA_HEAD_DIM_ARCHS.contains(&cfg.model_type.as_str())
        && cfg.qk_nope_head_dim > 0
        && cfg.qk_rope_head_dim > 0
    {
        // MLA's query/key width is independent of both the value width and
        // hidden_size / num_heads. `head_dim` is the QK width; `v_head_dim`
        // carries the output-value width.
        cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    }
    cfg.head_dim_kernel = round_up_attn_head_dim(cfg.head_dim);
    cfg.vocab_size = require_i32(j, "vocab_size", path)?;
    cfg.max_position_embeddings = require_i32(j, "max_position_embeddings", path)?;

    let eps_fallback = opt_f32(j, "norm_eps", 1e-5, path)?;
    let eps_fallback = opt_f32(j, "layer_norm_epsilon", eps_fallback, path)?;
    cfg.rms_norm_eps = opt_f32(j, "rms_norm_eps", eps_fallback, path)?;
    cfg.hidden_act = opt_string(j, "hidden_act", "silu", path)?;
    cfg.mlp_hidden_act = opt_string(j, "mlp_hidden_act", &cfg.hidden_act, path)?;

    cfg.rope_theta = opt_f32(j, "rope_theta", 10000.0, path)?;

    // Defaults are inert (`RopeScaling::None`) so plain-RoPE checkpoints take
    // the unscaled path.
    cfg.rope_factor = 1.0;
    cfg.rope_low_freq_factor = 1.0;
    cfg.rope_high_freq_factor = 4.0;
    cfg.rope_beta_fast = 32.0;
    cfg.rope_beta_slow = 1.0;
    cfg.rope_attention_factor = 1.0;
    cfg.rope_original_max_position = cfg.max_position_embeddings;
    cfg.rope_scaling_kind = RopeScaling::None;
    cfg.has_rope_scaling = false;
    if let Some(s) = flat_rope_config(j) {
        cfg.rope_theta = opt_f32(s, "rope_theta", cfg.rope_theta, path)?;
        let alias = opt_string(s, "type", "", path)?;
        let rope_type = opt_string(s, "rope_type", &alias, path)?;
        let has_llama3_keys =
            s.get("low_freq_factor").is_some() || s.get("high_freq_factor").is_some();
        if rope_type == "llama3" || has_llama3_keys {
            cfg.rope_scaling_kind = RopeScaling::Llama3;
            cfg.has_rope_scaling = true;
            cfg.rope_factor = opt_f32(s, "factor", 8.0, path)?;
            cfg.rope_low_freq_factor = opt_f32(s, "low_freq_factor", 1.0, path)?;
            cfg.rope_high_freq_factor = opt_f32(s, "high_freq_factor", 4.0, path)?;
            cfg.rope_original_max_position = opt_i32(
                s,
                "original_max_position_embeddings",
                cfg.max_position_embeddings,
                path,
            )?;
        } else if rope_type == "yarn" {
            cfg.rope_scaling_kind = RopeScaling::OriginalYarn;
            cfg.has_rope_scaling = true;
            cfg.rope_factor = opt_f32(s, "factor", 1.0, path)?;
            cfg.rope_beta_fast = opt_f32(s, "beta_fast", 32.0, path)?;
            cfg.rope_beta_slow = opt_f32(s, "beta_slow", 1.0, path)?;
            // vLLM `rotary_embedding.py`'s `yarn_get_mscale(scale, mscale)`.
            let factor = cfg.rope_factor;
            let yarn_mscale = |m: f32| {
                if factor <= 1.0 {
                    1.0f32
                } else {
                    0.1 * m * factor.ln() + 1.0
                }
            };
            // HF's DeepSeek configs default these to 1 and 0.
            let mscale = opt_f32(s, "mscale", 1.0, path)?;
            let mscale_all_dim = opt_f32(s, "mscale_all_dim", 0.0, path)?;
            let default_attn_factor = yarn_mscale(mscale) / yarn_mscale(mscale_all_dim);
            cfg.rope_attention_factor = opt_f32(s, "attention_factor", default_attn_factor, path)?;
            cfg.rope_mla_softmax_mscale = yarn_mscale(mscale_all_dim);
            cfg.rope_original_max_position = opt_i32(
                s,
                "original_max_position_embeddings",
                cfg.max_position_embeddings,
                path,
            )?;
        } else if (rope_type == "linear" || rope_type == "default")
            && s.get("factor").is_some()
        {
            // Linear scaling is the plain geometric series with positions
            // divided by `factor` — there is no separate schedule, so the kind
            // stays `None`, which also keeps the descriptor's enum wire-
            // compatible with the drivers' strict parsers. But the factor is a
            // real numeric fact and has to survive: dropping it makes an
            // imported artifact differ from the same checkpoint read as a raw
            // `config.json` (which does honour it), and the difference only
            // shows past the original context length.
            //
            // Gated on `factor` being present: a bare `rope_type: "default"`
            // (which Qwen3.5 ships) declares no scaling at all.
            cfg.has_rope_scaling = true;
            cfg.rope_factor = opt_f32(s, "factor", 1.0, path)?;
        }
    }

    cfg.sliding_window = opt_i32(j, "sliding_window", -1, path)?;
    cfg.rope_local_base_freq = opt_f32(j, "rope_local_base_freq", 0.0, path)?;

    // Per-layer attention type, from four sources in precedence order.
    if let Some(types) = opt_string_array(j, "layer_types", path)? {
        cfg.layer_types = types;
    } else if cfg.model_type == "nemotron_h" && j.get("hybrid_override_pattern").is_some() {
        let pattern = opt_string(j, "hybrid_override_pattern", "", path)?;
        if pattern.chars().count() as i32 != cfg.num_hidden_layers {
            bail!("config.json ({path}): hybrid_override_pattern size != num_hidden_layers");
        }
        for c in pattern.chars() {
            cfg.layer_types.push(
                match c {
                    'M' => "mamba",
                    '*' => "attention",
                    'E' => "moe",
                    '-' => "mlp",
                    _ => {
                        bail!("config.json ({path}): unsupported hybrid_override_pattern character")
                    }
                }
                .to_string(),
            );
        }
    } else if (cfg.model_type == "gemma3" || cfg.model_type == "gemma3_text")
        && j.get("sliding_window_pattern").is_some()
    {
        let pat = opt_i32(j, "sliding_window_pattern", 0, path)?;
        for i in 0..cfg.num_hidden_layers {
            // HF puts the full layer at the *end* of each pattern block.
            let full = pat != 0 && (i + 1) % pat == 0;
            cfg.layer_types.push(
                if full {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
                .to_string(),
            );
        }
    } else if cfg.model_type == "gemma2" && cfg.sliding_window > 0 {
        // Gemma-2 hardcodes `is_sliding = (layer_idx % 2 == 0)`.
        for i in 0..cfg.num_hidden_layers {
            cfg.layer_types.push(
                if i % 2 == 0 {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
                .to_string(),
            );
        }
    }

    let tie_default = TIE_BY_DEFAULT.contains(&cfg.model_type.as_str());
    cfg.tie_word_embeddings = opt_bool(j, "tie_word_embeddings", tie_default, path)?;
    // Qwen-2 always ships biased QKV but its config omits the flag.
    cfg.attention_bias = opt_bool(j, "attention_bias", cfg.model_type == "qwen2", path)?;
    cfg.use_qk_norm = infer_qk_norm(&cfg.model_type, j, path)?;

    let experts = opt_i32(j, "n_routed_experts", 0, path)?;
    let experts = opt_i32(j, "num_experts", experts, path)?;
    cfg.num_experts = opt_i32(j, "num_local_experts", experts, path)?;
    let top_k = opt_i32(j, "top_k_experts", 0, path)?;
    cfg.num_experts_per_tok = opt_i32(j, "num_experts_per_tok", top_k, path)?;
    cfg.first_k_dense_replace = opt_i32(j, "first_k_dense_replace", 0, path)?;
    cfg.n_shared_experts = opt_i32(j, "n_shared_experts", 0, path)?;
    // Read with a default of `false`, which overrides the schema's `true`.
    cfg.norm_topk_prob = opt_bool(j, "norm_topk_prob", false, path)?;
    cfg.routed_scaling_factor = opt_f32(j, "routed_scaling_factor", 1.0, path)?;

    if cfg.model_type == "deepseek_v4" {
        cfg.dsv4_o_lora_rank = opt_i32(j, "o_lora_rank", 0, path)?;
        cfg.dsv4_o_groups = opt_i32(j, "o_groups", 0, path)?;
        cfg.dsv4_index_head_dim = opt_i32(j, "index_head_dim", 0, path)?;
        cfg.dsv4_index_n_heads = opt_i32(j, "index_n_heads", 0, path)?;
        cfg.dsv4_index_topk = opt_i32(j, "index_topk", 0, path)?;
        cfg.dsv4_hc_mult = opt_i32(j, "hc_mult", 0, path)?;
        cfg.dsv4_hc_eps = opt_f32(j, "hc_eps", 1e-6, path)?;
        cfg.dsv4_num_hash_layers = opt_i32(j, "num_hash_layers", 0, path)?;
        cfg.dsv4_sliding_window = opt_i32(j, "sliding_window", 0, path)?;
        cfg.dsv4_compress_rope_theta = opt_f32(j, "compress_rope_theta", 0.0, path)?;
        cfg.dsv4_scoring_func = opt_string(j, "scoring_func", "", path)?;
        cfg.dsv4_expert_dtype = opt_string(j, "expert_dtype", "", path)?;
        // Dead: overwritten unconditionally below. Kept because the port
        // reproduces behavior, and behavior today is that it is dead.
        cfg.attention_has_sinks = true;
        if let Some(ratios) = opt_i32_array(j, "compress_ratios", path)? {
            cfg.dsv4_compress_ratios = ratios;
        }
    }

    cfg.gemma4_enable_moe = opt_bool(j, "enable_moe_block", false, path)?;
    cfg.gemma4_attention_k_eq_v = opt_bool(j, "attention_k_eq_v", false, path)?;
    cfg.gemma4_num_global_key_value_heads = opt_i32(
        j,
        "num_global_key_value_heads",
        cfg.num_key_value_heads,
        path,
    )?;
    // Carried for the Metal driver, which reads them; nothing on the CUDA side
    // does. They are in the descriptor because `pie.model/1` is the artifact's
    // model config, not one driver's.
    cfg.gemma4_global_head_dim = opt_i32(j, "global_head_dim", 0, path)?;
    cfg.gemma4_double_wide_mlp = opt_bool(j, "use_double_wide_mlp", false, path)?;

    cfg.swiglu_limit = opt_f32(j, "swiglu_limit", 0.0, path)?;
    cfg.mlp_has_bias = cfg.model_type == "gpt_oss";
    cfg.router_has_bias = cfg.model_type == "gpt_oss";
    // This is the assignment that kills the `deepseek_v4` one above.
    cfg.attention_has_sinks = cfg.model_type == "gpt_oss";

    cfg.gemma_query_pre_attn_scalar =
        opt_f32(j, "query_pre_attn_scalar", cfg.head_dim as f32, path)?;
    // `null` means "disabled", which is also the default — so both spellings
    // land on 0 and `opt_f32`'s null-is-absent rule does the work.
    cfg.gemma_final_logit_softcap = opt_f32(j, "final_logit_softcapping", 0.0, path)?;
    cfg.gemma_attn_logit_softcap = opt_f32(j, "attn_logit_softcapping", 0.0, path)?;
    cfg.gemma_hidden_size_per_layer_input = opt_i32(j, "hidden_size_per_layer_input", 0, path)?;
    cfg.num_kv_shared_layers = opt_i32(j, "num_kv_shared_layers", 0, path)?;
    // These three read the *root* first, then the text config.
    let nested = opt_bool(j, "use_ordered_embeddings", false, path)?;
    cfg.gemma4_use_ordered_embeddings = opt_bool(root, "use_ordered_embeddings", nested, path)?;
    let nested = opt_i32(j, "num_centroids", 0, path)?;
    cfg.gemma4_num_centroids = opt_i32(root, "num_centroids", nested, path)?;
    let nested = opt_i32(j, "centroid_intermediate_top_k", 0, path)?;
    cfg.gemma4_centroid_intermediate_top_k =
        opt_i32(root, "centroid_intermediate_top_k", nested, path)?;

    // Gemma-4 keys RoPE by attention type; pre-expand to per-layer vectors so
    // the forward does not have to consult `layer_types` twice.
    if let Some(rope_params) = object(j, "rope_parameters")
        && !cfg.layer_types.is_empty()
    {
        cfg.gemma_per_layer_rope_theta = vec![cfg.rope_theta; cfg.layer_types.len()];
        cfg.gemma_per_layer_partial_rotary_factor = vec![1.0; cfg.layer_types.len()];
        for (i, layer_type) in cfg.layer_types.iter().enumerate() {
            if let Some(e) = object(rope_params, layer_type) {
                cfg.gemma_per_layer_rope_theta[i] = opt_f32(e, "rope_theta", cfg.rope_theta, path)?;
                cfg.gemma_per_layer_partial_rotary_factor[i] =
                    opt_f32(e, "partial_rotary_factor", 1.0, path)?;
            }
        }
    }

    cfg.moe_intermediate_size = opt_i32(j, "moe_intermediate_size", 0, path)?;
    let aliased = opt_i32(j, "moe_shared_expert_intermediate_size", 0, path)?;
    cfg.shared_expert_intermediate_size =
        opt_i32(j, "shared_expert_intermediate_size", aliased, path)?;
    if cfg.shared_expert_intermediate_size == 0
        && cfg.n_shared_experts > 0
        && cfg.moe_intermediate_size > 0
    {
        cfg.shared_expert_intermediate_size = cfg.n_shared_experts * cfg.moe_intermediate_size;
    }
    let groups = opt_i32(j, "n_groups", 1, path)?;
    cfg.n_group = opt_i32(j, "n_group", groups, path)?;
    cfg.topk_group = opt_i32(j, "topk_group", 1, path)?;

    if cfg.model_type == "nemotron_h" {
        cfg.mamba_num_heads = opt_i32(j, "mamba_num_heads", 0, path)?;
        cfg.mamba_head_dim = opt_i32(j, "mamba_head_dim", 0, path)?;
        cfg.mamba_state_size = opt_i32(j, "ssm_state_size", 0, path)?;
        let alias = opt_i32(j, "mamba_n_groups", 0, path)?;
        cfg.mamba_n_groups = opt_i32(j, "n_groups", alias, path)?;
        let alias = opt_i32(j, "mamba_d_conv", 0, path)?;
        cfg.mamba_conv_kernel = opt_i32(j, "conv_kernel", alias, path)?;
        cfg.mamba_chunk_size = opt_i32(j, "chunk_size", 0, path)?;
        let alias = opt_f32(j, "mamba_dt_min", 0.001, path)?;
        cfg.mamba_time_step_min = opt_f32(j, "time_step_min", alias, path)?;
    }

    cfg.linear_num_value_heads = opt_i32(j, "linear_num_value_heads", 0, path)?;
    cfg.linear_num_key_heads = opt_i32(j, "linear_num_key_heads", 0, path)?;
    cfg.linear_key_head_dim = opt_i32(j, "linear_key_head_dim", 0, path)?;
    cfg.linear_value_head_dim = opt_i32(j, "linear_value_head_dim", 0, path)?;
    cfg.linear_conv_kernel_dim = opt_i32(j, "linear_conv_kernel_dim", 0, path)?;
    cfg.attn_output_gate = opt_bool(
        j,
        "attn_output_gate",
        cfg.model_type == "qwen3_5" || cfg.model_type == "qwen3_5_text",
        path,
    )?;

    if cfg.model_type == "kimi_k3" {
        // K3 spells its expert counts differently from everything above.
        cfg.num_experts_per_tok =
            opt_i32(j, "num_experts_per_token", cfg.num_experts_per_tok, path)?;
        cfg.n_group = opt_i32(j, "num_expert_group", cfg.n_group, path)?;
        cfg.n_shared_experts = opt_i32(j, "num_shared_experts", cfg.n_shared_experts, path)?;
        cfg.norm_topk_prob = opt_bool(j, "moe_renormalize", cfg.norm_topk_prob, path)?;
        cfg.moe_router_activation_func =
            opt_string(j, "moe_router_activation_func", "sigmoid", path)?;
        cfg.routed_expert_hidden_size = opt_i32(j, "routed_expert_hidden_size", 0, path)?;
        cfg.latent_moe_use_norm = opt_bool(j, "latent_moe_use_norm", false, path)?;
        cfg.attn_res_block_size = opt_i32(j, "attn_res_block_size", 0, path)?;
        cfg.mla_use_nope = opt_bool(j, "mla_use_nope", false, path)?;
        cfg.mla_output_gate = opt_bool(j, "mla_use_output_gate", false, path)?;
        if opt_string(j, "hidden_act", "", path)? == "situ" {
            cfg.situ_beta = opt_f32(j, "activation_situ_beta", 1.0, path)?;
            cfg.situ_linear_beta = opt_f32(j, "activation_situ_linear_beta", 0.0, path)?;
        }
        // `shared_experts` is one MLP `num_shared_experts` wide.
        if cfg.n_shared_experts > 0 && cfg.moe_intermediate_size > 0 {
            cfg.shared_expert_intermediate_size = cfg.n_shared_experts * cfg.moe_intermediate_size;
        }

        if let Some(la) = object(j, "linear_attn_config") {
            let heads = opt_i32(la, "num_heads", 0, path)?;
            let dim = opt_i32(la, "head_dim", 0, path)?;
            // KDA is not grouped: Q, K and V all carry `num_heads` heads.
            cfg.linear_num_value_heads = heads;
            cfg.linear_num_key_heads = heads;
            cfg.linear_key_head_dim = dim;
            cfg.linear_value_head_dim = dim;
            cfg.linear_conv_kernel_dim = opt_i32(la, "short_conv_kernel_size", 0, path)?;
            cfg.kda_gate_lower_bound = opt_f32(la, "gate_lower_bound", 0.0, path)?;
            cfg.kda_full_rank_gate = opt_bool(la, "use_full_rank_gate", false, path)?;

            // `kda_layers` is **1-indexed** — HF tests `layer_idx + 1`.
            // Reading it 0-indexed shifts the hybrid pattern by one layer,
            // which still loads and still runs.
            if cfg.layer_types.is_empty()
                && let Some(layers) = opt_i32_array(la, "kda_layers", path)?
            {
                let mut is_kda = vec![false; cfg.num_hidden_layers.max(0) as usize];
                for entry in layers {
                    let zero_based = entry - 1;
                    if zero_based < 0 || zero_based >= cfg.num_hidden_layers {
                        bail!(
                            "config.json ({path}): linear_attn_config.kda_layers entry \
                                 {entry} is out of range for {} layers (the list is 1-indexed)",
                            cfg.num_hidden_layers
                        );
                    }
                    is_kda[zero_based as usize] = true;
                }
                for i in 0..cfg.num_hidden_layers {
                    cfg.layer_types.push(
                        if is_kda[i as usize] {
                            "linear_attention"
                        } else {
                            "full_attention"
                        }
                        .to_string(),
                    );
                }
            }
        }
    }

    // Partial RoPE (Qwen3.5): under `rope_parameters` there, top level
    // elsewhere.
    cfg.partial_rotary_factor = 1.0;
    match flat_rope_parameters(j) {
        Some(rp) if rp.get("partial_rotary_factor").is_some() => {
            cfg.partial_rotary_factor = opt_f32(rp, "partial_rotary_factor", 1.0, path)?;
        }
        _ => cfg.partial_rotary_factor = opt_f32(j, "partial_rotary_factor", 1.0, path)?,
    }
    if let Some(rp) = flat_rope_parameters(j)
        && rp.get("rope_theta").is_some_and(Value::is_number)
    {
        cfg.rope_theta = opt_f32(rp, "rope_theta", cfg.rope_theta, path)?;
    }
    cfg.mtp_num_hidden_layers = opt_i32(j, "mtp_num_hidden_layers", 0, path)?;
    cfg.mtp_use_dedicated_embeddings = opt_bool(j, "mtp_use_dedicated_embeddings", false, path)?;

    cfg.altup_num_inputs = opt_i32(j, "altup_num_inputs", 0, path)?;
    cfg.altup_active_idx = opt_i32(j, "altup_active_idx", 0, path)?;
    cfg.altup_correct_scale = opt_bool(j, "altup_correct_scale", false, path)?;
    cfg.altup_coef_clip = opt_f32(j, "altup_coef_clip", 0.0, path)?;
    cfg.laurel_rank = opt_i32(j, "laurel_rank", 0, path)?;
    cfg.vocab_size_per_layer_input = opt_i32(j, "vocab_size_per_layer_input", 0, path)?;
    cfg.gemma3n_rope_local_base_freq =
        opt_f32(j, "rope_local_base_freq", cfg.rope_local_base_freq, path)?;
    if let Some(sparsity) = opt_f32_array(j, "activation_sparsity_pattern", path)? {
        cfg.gemma3n_activation_sparsity = sparsity;
    }

    cfg.index_topk = opt_i32(j, "index_topk", 0, path)?;
    cfg.index_head_dim = opt_i32(j, "index_head_dim", 0, path)?;
    cfg.index_n_heads = opt_i32(j, "index_n_heads", 0, path)?;
    if let Some(types) = opt_string_array(j, "indexer_types", path)? {
        cfg.indexer_types = types;
    } else if cfg.index_topk > 0 && cfg.num_hidden_layers > 0 {
        let freq = opt_i32(j, "index_topk_freq", 1, path)?;
        for i in 0..cfg.num_hidden_layers {
            let is_full = freq != 0 && ((i - 1).max(0) % freq) == 0;
            cfg.indexer_types
                .push(if is_full { "full" } else { "shared" }.to_string());
        }
    }

    cfg.torch_dtype = opt_string(j, "torch_dtype", "bfloat16", path)?;

    // Multimodal text-tower extraction: the driver runs the LLM forward only,
    // so vision-side tensors are skipped and the LM prefix is stripped.
    let has_vision_config = object(root, "vision_config").is_some();
    let is_kimi_k25_wrapper =
        outer_model_type == "kimi_k25" && root.get("text_config").is_some();
    if (has_vision_config && root.get("text_config").is_some()) || is_kimi_k25_wrapper {
        cfg.mm_lm_strip_prefix = "language_model.".to_string();
        cfg.mm_skip_prefixes = [
            "vision_tower.",
            "vision_model.",
            "visual.",
            "multi_modal_projector.",
            "mm_projector.",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
    }
    if cfg.model_type == "nemotron_h" {
        cfg.mm_skip_prefixes = [
            "vision_model.",
            "mlp1.",
            "sound_encoder.",
            "sound_projection.",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
    }

    if let Some(v) = object(root, "vision_config")
        && opt_string(v, "model_type", "", path)? == "gemma4_vision"
    {
        let mut gv = GemmaVisionConfig::default();
        gv.hidden_size = opt_i32(v, "hidden_size", gv.hidden_size, path)?;
        gv.intermediate_size = opt_i32(v, "intermediate_size", gv.intermediate_size, path)?;
        gv.num_hidden_layers = opt_i32(v, "num_hidden_layers", gv.num_hidden_layers, path)?;
        gv.num_attention_heads = opt_i32(v, "num_attention_heads", gv.num_attention_heads, path)?;
        gv.num_key_value_heads = opt_i32(v, "num_key_value_heads", gv.num_attention_heads, path)?;
        gv.head_dim = opt_i32(v, "head_dim", gv.head_dim, path)?;
        gv.patch_size = opt_i32(v, "patch_size", gv.patch_size, path)?;
        gv.rms_norm_eps = opt_f32(v, "rms_norm_eps", gv.rms_norm_eps, path)?;
        gv.pooling_kernel_size = opt_i32(v, "pooling_kernel_size", gv.pooling_kernel_size, path)?;
        gv.use_clipped_linears = opt_bool(v, "use_clipped_linears", gv.use_clipped_linears, path)?;
        if let Some(rp) = object(v, "rope_parameters") {
            gv.rope_theta = opt_f32(rp, "rope_theta", gv.rope_theta, path)?;
        }
        gv.soft_tokens_per_image = opt_i32(
            root,
            "vision_soft_tokens_per_image",
            gv.soft_tokens_per_image,
            path,
        )?;
        cfg.gemma_vision = Some(gv);
    }

    if let Some(a) = object(root, "audio_config")
        && opt_string(a, "model_type", "", path)? == "gemma4_audio"
    {
        let mut ga = GemmaAudioConfig::default();
        ga.hidden_size = opt_i32(a, "hidden_size", ga.hidden_size, path)?;
        ga.num_attention_heads = opt_i32(a, "num_attention_heads", ga.num_attention_heads, path)?;
        ga.num_hidden_layers = opt_i32(a, "num_hidden_layers", ga.num_hidden_layers, path)?;
        ga.conv_kernel_size = opt_i32(a, "conv_kernel_size", ga.conv_kernel_size, path)?;
        ga.output_proj_dims = opt_i32(a, "output_proj_dims", ga.output_proj_dims, path)?;
        ga.attention_chunk_size =
            opt_i32(a, "attention_chunk_size", ga.attention_chunk_size, path)?;
        ga.attention_context_left =
            opt_i32(a, "attention_context_left", ga.attention_context_left, path)?;
        ga.attention_context_right = opt_i32(
            a,
            "attention_context_right",
            ga.attention_context_right,
            path,
        )?;
        ga.feature_size = opt_i32(a, "feature_size", ga.feature_size, path)?;
        ga.attention_logit_cap = opt_f32(a, "attention_logit_cap", ga.attention_logit_cap, path)?;
        ga.residual_weight = opt_f32(a, "residual_weight", ga.residual_weight, path)?;
        ga.rms_norm_eps = opt_f32(a, "rms_norm_eps", ga.rms_norm_eps, path)?;
        ga.use_clipped_linears = opt_bool(a, "use_clipped_linears", ga.use_clipped_linears, path)?;
        if let Some(channels) = opt_i32_array(a, "subsampling_conv_channels", path)?
            && channels.len() >= 2
        {
            ga.subsampling_conv_channels0 = channels[0];
            ga.subsampling_conv_channels1 = channels[1];
        }
        cfg.gemma_audio = Some(ga);
    }

    if let Some(v) = object(root, "vision_config")
        && opt_string(v, "model_type", "", path)? == "qwen3_vl"
    {
        let mut qv = Qwen3VLVisionConfig::default();
        qv.hidden_size = opt_i32(v, "hidden_size", qv.hidden_size, path)?;
        qv.intermediate_size = opt_i32(v, "intermediate_size", qv.intermediate_size, path)?;
        qv.depth = opt_i32(v, "depth", qv.depth, path)?;
        qv.num_heads = opt_i32(v, "num_heads", qv.num_heads, path)?;
        qv.patch_size = opt_i32(v, "patch_size", qv.patch_size, path)?;
        qv.temporal_patch_size = opt_i32(v, "temporal_patch_size", qv.temporal_patch_size, path)?;
        qv.spatial_merge_size = opt_i32(v, "spatial_merge_size", qv.spatial_merge_size, path)?;
        qv.in_channels = opt_i32(v, "in_channels", qv.in_channels, path)?;
        qv.out_hidden_size = opt_i32(v, "out_hidden_size", qv.out_hidden_size, path)?;
        qv.num_position_embeddings = opt_i32(
            v,
            "num_position_embeddings",
            qv.num_position_embeddings,
            path,
        )?;
        if let Some(indexes) = opt_i32_array(v, "deepstack_visual_indexes", path)? {
            qv.deepstack_visual_indexes = indexes;
        }
        cfg.qwen3_vl_vision = Some(qv);

        // The text tower is Qwen3 (per-head q/k norm) but its config has
        // no explicit flag — the tensors carry q_norm/k_norm.
        cfg.use_qk_norm = true;

        if let Some(rs) = object(j, "rope_scaling") {
            if let Some(section) = opt_i32_array(rs, "mrope_section", path)? {
                cfg.qwen3_vl_mrope_section = section;
            }
            cfg.qwen3_vl_mrope_interleaved = opt_bool(rs, "mrope_interleaved", false, path)?;
        }
        // M-RoPE is not a frequency-scaling variant — keep plain RoPE.
        cfg.rope_scaling_kind = RopeScaling::None;
        cfg.has_rope_scaling = false;

        cfg.qwen3_vl_image_token_id = opt_i32(root, "image_token_id", -1, path)?;
        cfg.qwen3_vl_vision_start_token_id = opt_i32(root, "vision_start_token_id", -1, path)?;
        cfg.qwen3_vl_vision_end_token_id = opt_i32(root, "vision_end_token_id", -1, path)?;
    }

    // Multimodal configs put `quantization_config` at the root; text-only
    // ones under the text view. Resolved here, applied after the CSM block —
    // the order the C++ side happens to use.
    //
    // The bare `quantization` key is `mlx_lm`'s spelling of the same block,
    // and it is last because it is the only one `parse_hf_config` never read:
    // the C++ normalizer served CUDA, and MLX checkpoints reach Metal. Metal
    // read them with a parser of its own, which is the parser this crate
    // replaced — so leaving the key unread here would move an MLX checkpoint's
    // declared width out of reach entirely, and `push_mlx_affine_declared`
    // answers a missing width with 4 bits. An 8-bit checkpoint would author
    // silently wrong (twice the logical columns), which is the failure mode
    // `pie.model/1` exists to remove rather than relocate.
    //
    // No corpus config carries it, so no golden moves: this is additive for
    // every input the differential covers.
    let quantization = object(j, "quantization_config")
        .or_else(|| object(root, "quantization_config"))
        .or_else(|| object(j, "quantization"))
        .or_else(|| object(root, "quantization"));

    if cfg.model_type == "csm" {
        let mut csm = CsmConfig::default();
        csm.text_vocab_size = opt_i32(root, "text_vocab_size", csm.text_vocab_size, path)?;
        // The top-level vocab_size is the per-codebook audio vocabulary.
        csm.audio_vocab_size = cfg.vocab_size;
        csm.num_codebooks = opt_i32(root, "num_codebooks", csm.num_codebooks, path)?;
        csm.codebook_eos_token_id = opt_i32(
            root,
            "codebook_eos_token_id",
            csm.codebook_eos_token_id,
            path,
        )?;
        csm.audio_eos_token_id = opt_i32(root, "audio_eos_token_id", csm.audio_eos_token_id, path)?;
        csm.audio_token_id = opt_i32(root, "audio_token_id", csm.audio_token_id, path)?;

        if let Some(d) = object(root, "depth_decoder_config") {
            let dd = &mut csm.depth;
            dd.hidden_size = opt_i32(d, "hidden_size", dd.hidden_size, path)?;
            dd.backbone_hidden_size =
                opt_i32(d, "backbone_hidden_size", dd.backbone_hidden_size, path)?;
            dd.num_hidden_layers = opt_i32(d, "num_hidden_layers", dd.num_hidden_layers, path)?;
            dd.num_attention_heads =
                opt_i32(d, "num_attention_heads", dd.num_attention_heads, path)?;
            dd.num_key_value_heads =
                opt_i32(d, "num_key_value_heads", dd.num_key_value_heads, path)?;
            dd.head_dim = opt_i32(d, "head_dim", dd.head_dim, path)?;
            dd.intermediate_size = opt_i32(d, "intermediate_size", dd.intermediate_size, path)?;
            dd.num_codebooks = opt_i32(d, "num_codebooks", dd.num_codebooks, path)?;
            dd.vocab_size = opt_i32(d, "vocab_size", dd.vocab_size, path)?;
            dd.max_position_embeddings = opt_i32(
                d,
                "max_position_embeddings",
                dd.max_position_embeddings,
                path,
            )?;
            dd.rms_norm_eps = opt_f32(d, "rms_norm_eps", dd.rms_norm_eps, path)?;
            dd.rope_theta = opt_f32(d, "rope_theta", dd.rope_theta, path)?;
            if let Some(s) = object(d, "rope_scaling") {
                dd.rope_factor = opt_f32(s, "factor", dd.rope_factor, path)?;
                dd.rope_low_freq_factor =
                    opt_f32(s, "low_freq_factor", dd.rope_low_freq_factor, path)?;
                dd.rope_high_freq_factor =
                    opt_f32(s, "high_freq_factor", dd.rope_high_freq_factor, path)?;
                dd.rope_original_max_position = opt_i32(
                    s,
                    "original_max_position_embeddings",
                    dd.rope_original_max_position,
                    path,
                )?;
            }
        }

        if let Some(c) = object(root, "codec_config") {
            let mc = &mut csm.codec;
            mc.hidden_size = opt_i32(c, "hidden_size", mc.hidden_size, path)?;
            let alias = opt_i32(c, "codebook_dim", mc.codebook_dim, path)?;
            mc.codebook_dim = opt_i32(c, "vector_quantization_hidden_dimension", alias, path)?;
            mc.codebook_size = opt_i32(c, "codebook_size", mc.codebook_size, path)?;
            mc.num_quantizers = opt_i32(c, "num_quantizers", mc.num_quantizers, path)?;
            mc.num_semantic_quantizers = opt_i32(
                c,
                "num_semantic_quantizers",
                mc.num_semantic_quantizers,
                path,
            )?;
            mc.num_filters = opt_i32(c, "num_filters", mc.num_filters, path)?;
            mc.xf_num_attention_heads =
                opt_i32(c, "num_attention_heads", mc.xf_num_attention_heads, path)?;
            mc.xf_num_key_value_heads =
                opt_i32(c, "num_key_value_heads", mc.xf_num_key_value_heads, path)?;
            mc.xf_head_dim = opt_i32(c, "head_dim", mc.xf_head_dim, path)?;
            mc.xf_intermediate_size =
                opt_i32(c, "intermediate_size", mc.xf_intermediate_size, path)?;
            mc.xf_num_hidden_layers =
                opt_i32(c, "num_hidden_layers", mc.xf_num_hidden_layers, path)?;
            mc.xf_sliding_window = opt_i32(c, "sliding_window", mc.xf_sliding_window, path)?;
            mc.xf_rope_theta = opt_f32(c, "rope_theta", mc.xf_rope_theta, path)?;
            mc.norm_eps = opt_f32(c, "norm_eps", mc.norm_eps, path)?;
            mc.sampling_rate = opt_i32(c, "sampling_rate", mc.sampling_rate, path)?;
            mc.use_causal_conv = opt_bool(c, "use_causal_conv", mc.use_causal_conv, path)?;
            mc.upsample_groups = opt_i32(c, "upsample_groups", mc.upsample_groups, path)?;
            mc.residual_kernel_size =
                opt_i32(c, "residual_kernel_size", mc.residual_kernel_size, path)?;
            mc.kernel_size = opt_i32(c, "kernel_size", mc.kernel_size, path)?;
            mc.last_kernel_size = opt_i32(c, "last_kernel_size", mc.last_kernel_size, path)?;
            if let Some(ratios) = opt_i32_array(c, "upsampling_ratios", path)? {
                mc.upsampling_ratios = ratios;
            }
        }
        cfg.csm = Some(csm);
    }

    if let Some(q) = quantization {
        cfg.quant_method = opt_string(q, "quant_method", "", path)?;
        cfg.quant_bits = opt_i32(q, "bits", 0, path)?;
        cfg.quant_group_size = opt_i32(q, "group_size", 0, path)?;
        cfg.quant_desc_act = opt_bool(q, "desc_act", false, path)?;
        cfg.quant_sym = opt_bool(q, "sym", true, path)?;
        // GPTQ's `sym=true` implies no zero points; AWQ sets `zero_point`
        // explicitly even when `sym` is unset. Independent signals.
        cfg.quant_zero_point = opt_bool(q, "zero_point", !cfg.quant_sym, path)?;
        if q.get("kv_cache_scheme").is_some_and(|v| !v.is_null()) {
            cfg.kv_cache_scheme_present = true;
        }
    }

    Ok(cfg)
}

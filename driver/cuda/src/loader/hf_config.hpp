#pragma once

// Parsed HuggingFace `config.json`. We keep the dataclass narrow — only the
// fields the forward pass and capability handshake need. Add fields here as
// new architectures land.

#include <filesystem>
#include <string>
#include <vector>

namespace pie_cuda_driver {

struct HfConfig {
    // Architecture discriminator (e.g. "Qwen3ForCausalLM"). The first entry
    // of HF's `architectures` list — we keep the exact string for
    // DriverCapabilities.arch_name and for our own model-registry dispatch.
    std::string arch_name;

    // Lower-case `model_type` ("qwen3", "llama", …). Used for registry lookup
    // because some HF configs ship with multiple architectures.
    std::string model_type;

    // ── Transformer dimensions ────────────────────────────────────────
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;   // GQA. Equal to num_attention_heads if MHA.
    int head_dim;              // Some configs imply head_dim = hidden / heads;
                               // Qwen3 sets it explicitly.
    // The HEAD_DIM the attention kernel actually runs at. Equal to
    // `head_dim` for every model whose head_dim is in flashinfer's
    // dispatch set ({64, 128, 256, 512}). For Phi-3-mini (head_dim=96)
    // we round up to 128 and pad Q/K/V/O with zeros — flashinfer's
    // tensor-core prefill produces NaN at HEAD_DIM=96 (a swizzle/
    // alignment bug in 0.6.x).
    int head_dim_kernel;
    int vocab_size;
    int max_position_embeddings;

    // ── Norm / activation ─────────────────────────────────────────────
    float rms_norm_eps;
    std::string hidden_act;    // "silu" — only one supported for now.
    std::string mlp_hidden_act; // Nemotron-H uses "relu2".

    // ── RoPE ──────────────────────────────────────────────────────────
    float rope_theta;
    // RoPE scaling variant — `None` means plain RoPE (no scaling).
    // `Llama3` uses the smoothed-interpolation YaRN ramp (low/high
    // freq factors); `OriginalYaRN` uses the dim-index ramp from the
    // YaRN paper (beta_fast/beta_slow + attention_factor mscale) and
    // is what OLMo-3 / gpt-oss / DeepSeek-V3 ship.
    enum class RopeScaling { None, Llama3, OriginalYaRN };
    RopeScaling rope_scaling_kind = RopeScaling::None;
    // Llama-3 YaRN params. Inert under `RopeScaling::OriginalYaRN`.
    float rope_factor;
    float rope_low_freq_factor;
    float rope_high_freq_factor;
    int   rope_original_max_position;
    // Original-YaRN params. Inert under `RopeScaling::Llama3`.
    float rope_beta_fast        = 32.f;
    float rope_beta_slow        = 1.f;
    float rope_attention_factor = 1.f;
    // Backwards-compatible accessor: `has_rope_scaling` is true iff the
    // ckpt uses Llama-3 YaRN (the only variant that took the YaRN code
    // path before the OriginalYaRN dispatch was added). Per-arch code
    // that branches on YaRN vs plain RoPE keeps its existing semantics.
    bool has_rope_scaling = false;

    // ── Sliding-window attention ──────────────────────────────────────
    // -1 means full causal. Positive = `window_left` per request to
    // pass through to flashinfer's prefill plan.
    int sliding_window;

    // Per-layer attention type. Empty when the model uses a single
    // attention type across all layers (homogeneous Llama / Mistral /
    // Phi-3). Populated for OLMo-3 (HF `layer_types`) and Gemma-2/3
    // (Gemma-2 hardcodes `i%2==0` = sliding in `modeling_gemma2.py`;
    // Gemma-3 uses `sliding_window_pattern` — every Nth layer is
    // full, others sliding). Values are kept as the same strings HF
    // stores: "sliding_attention" or "full_attention".
    std::vector<std::string> layer_types;

    // Gemma-3-style dual-RoPE base. Sliding layers in Gemma-3 use a
    // *separate* rope_theta (`rope_local_base_freq`) while full-attention
    // layers use the standard `rope_theta`. Zero means "fall back to
    // `rope_theta` for sliding layers too" (every other model).
    float rope_local_base_freq;

    // ── Architectural quirks ──────────────────────────────────────────
    bool tie_word_embeddings;  // If true, lm_head shares weight with embed.
    bool attention_bias;       // QKV/O biases (Qwen2 yes, Qwen3 no).
    bool use_qk_norm;          // Qwen3 / Gemma-3 / OLMo-3.

    // ── Sparse MoE (Mixtral / GPT-OSS / Qwen-3.5 hybrid) ─────────────
    // Zero on dense models. `num_experts` is HF's `num_local_experts`;
    // `num_experts_per_tok` is the top-K used by the router. (Gemma-4
    // calls these `num_experts` / `top_k_experts` — same fields.)
    int num_experts;
    int num_experts_per_tok;
    // Gemma-4 26B-A4B runs **both** dense MLP and MoE in parallel per
    // layer; the dense `intermediate_size` and `moe_intermediate_size`
    // both apply when this is true. Inert on every other arch.
    bool gemma4_enable_moe = false;
    // Gemma-4 26B-A4B's "k_eq_v" mode: full-attention layers ship with
    // no `v_proj.weight` (V is derived from raw k_proj output, then
    // v-norm) and use `gemma4_num_global_key_value_heads` instead of
    // `num_key_value_heads`. Sliding-attention layers stay on the
    // standard `num_key_value_heads` and have their own v_proj.
    bool gemma4_attention_k_eq_v = false;
    int  gemma4_num_global_key_value_heads = 0;

    // GPT-OSS-specific knobs. Inert on every other model.
    //   * `swiglu_limit` — clipping threshold applied to gate values
    //     before SiLU (`x = clamp(x, -limit, limit)`). 0 = no clip.
    //   * `mlp_has_bias` — gate_up / down each carry an additive bias
    //     vector per expert.
    //   * `router_has_bias` — router projection has a bias term added
    //     before the top-K softmax.
    //   * `attention_has_sinks` — per-head learnable sink scalar that
    //     extends the softmax denominator (`Z + exp(sink_h)`). The
    //     forward path applies this as a post-attention rescale based
    //     on the kernel's log-sum-exp output.
    float swiglu_limit;
    bool  mlp_has_bias;
    bool  router_has_bias;
    bool  attention_has_sinks;

    // ── Gemma family ─────────────────────────────────────────────────
    // `query_pre_attn_scalar` defaults to `head_dim` so non-Gemma models
    // that don't set the field still get the standard `1/sqrt(head_dim)`
    // attention scale. `gemma_final_logit_softcap` is 0 when no cap is
    // requested (the kernel checks `> 0` and skips the call). The
    // attention-logit cap (Gemma-2 only — `attn_logit_softcapping=50`)
    // routes through flashinfer's `use_logits_soft_cap=true` variant.
    float gemma_query_pre_attn_scalar;
    float gemma_final_logit_softcap;
    float gemma_attn_logit_softcap;

    // Gemma-4 — Per-Layer Embeddings + KV-cache sharing. `ple_dim` is
    // 0 on every other model (no PLE residual block runs). When
    // `num_kv_shared_layers > 0` the last N layers reuse K/V from a
    // source layer of the same `layer_types[i]`.
    int gemma_hidden_size_per_layer_input;
    int num_kv_shared_layers;
    bool gemma4_use_ordered_embeddings = false;
    int gemma4_num_centroids = 0;
    int gemma4_centroid_intermediate_top_k = 0;

    // Gemma-4 per-layer rope_theta (HF nests under `rope_parameters`),
    // including `partial_rotary_factor` for full-attention layers
    // (typically 0.25 — only the first 25% of head_dim is rotated).
    // Empty for non-Gemma-4 models. Indexed by layer.
    std::vector<float> gemma_per_layer_rope_theta;
    std::vector<float> gemma_per_layer_partial_rotary_factor;

    // ── Qwen3.6-MoE specific ─────────────────────────────────────────
    // Sparse-MoE block dims (zero on non-MoE archs). `moe_intermediate_size`
    // is the per-expert hidden width; `shared_expert_intermediate_size`
    // is the (always-on) shared expert's MLP width.
    int moe_intermediate_size;
    int shared_expert_intermediate_size;
    float routed_scaling_factor = 1.f;
    int n_group = 1;
    int topk_group = 1;
    bool norm_topk_prob = true;

    // ── Nemotron-H hybrid Mamba2/attention/MoE ─────────────────────
    // `layer_types` stores "mamba", "attention", or "moe" for this
    // architecture. These dimensions are zero on non-Nemotron models.
    int mamba_num_heads = 0;
    int mamba_head_dim = 0;
    int mamba_state_size = 0;
    int mamba_n_groups = 0;
    int mamba_conv_kernel = 0;
    int mamba_chunk_size = 0;
    float mamba_time_step_min = 0.f;

    // ── DeepSeek/Kimi MLA + MoE specific ────────────────────────────
    // Kimi K2.6 exposes the language tower as `model_type=kimi_k2` inside
    // a `kimi_k25` wrapper and uses DeepSeek-V3-style MLA attention.
    // These are zero/inert for standard MHA/GQA models.
    int q_lora_rank = 0;
    int kv_lora_rank = 0;
    int qk_nope_head_dim = 0;
    int qk_rope_head_dim = 0;
    int v_head_dim = 0;
    int first_k_dense_replace = 0;
    int n_shared_experts = 0;

    // ── DeepSeek V4 specific ────────────────────────────────────────
    int dsv4_o_lora_rank = 0;
    int dsv4_o_groups = 0;
    int dsv4_index_head_dim = 0;
    int dsv4_index_n_heads = 0;
    int dsv4_index_topk = 0;
    int dsv4_hc_mult = 0;
    int dsv4_num_hash_layers = 0;
    int dsv4_sliding_window = 0;
    float dsv4_hc_eps = 1e-6f;
    float dsv4_compress_rope_theta = 0.f;
    std::vector<int> dsv4_compress_ratios;
    std::string dsv4_scoring_func;
    std::string dsv4_expert_dtype;

    // ── Qwen3.5 hybrid (linear-attention SSM + full attention) ──────
    // Per-layer attention type is in `layer_types` (values
    // "linear_attention" / "full_attention"). The linear layers run a
    // Gated DeltaNet recurrence; the full layers run standard scaled-
    // dot-product attention with a per-token output gate (a' = a *
    // sigmoid(gate), where gate is the second half of `q_proj`).
    //
    // Linear-attention dimensions. Inert (zero) on every other model.
    int   linear_num_value_heads;     // 32 on Qwen3.5-4B
    int   linear_num_key_heads;       // 16 on Qwen3.5-4B
    int   linear_key_head_dim;        // 128 on Qwen3.5-4B
    int   linear_value_head_dim;      // 128 on Qwen3.5-4B
    int   linear_conv_kernel_dim;     // 4 (depthwise causal conv kernel)
    // Full-attention output gating: q_proj output is split (query, gate);
    // attn output is multiplied by sigmoid(gate) before o_proj.
    bool  attn_output_gate;
    // Partial RoPE: only the first `partial_rotary_factor * head_dim`
    // dimensions are rotated. Defaults to 1.0 (full rotation).
    float partial_rotary_factor;

    // Qwen3.5 / Qwen3.6 MTP (multi-token prediction) auxiliary head.
    int  mtp_num_hidden_layers = 0;
    bool mtp_use_dedicated_embeddings = false;

    // Gemma-3n (E2B / E4B "Nano") additions on top of Gemma-4.
    // Gemma-3n is a *different* architecture from Gemma-4 (despite the
    // overlapping E2B / E4B naming) — it adds three new building blocks:
    //
    //   * AltUp ("Alternating Updates"): each layer maintains
    //     `altup_num_inputs` (4 by default) parallel residual streams; the
    //     active one (idx `altup_active_idx`) flows through attention +
    //     MLP, the others are updated via per-layer prediction /
    //     correction matmuls routed by a per-token "modality" vector.
    //   * Laurel ("Learned Augmented Residual Layer"): a per-layer
    //     low-rank skip — `linear_left` (H → laurel_rank), `linear_right`
    //     (laurel_rank → H), then RMSNorm and residual-add.
    //   * Activation sparsity: per-layer hard sparsity gate on the SwiGLU
    //     gate via Gaussian-quantile cutoff (only nonzero on the early
    //     layers of E2B per `activation_sparsity_pattern`).
    //
    // Plus `intermediate_size` is per-layer (HF stores it as a list).
    // We populate `gemma3n_per_layer_intermediate` from that list and
    // mirror the first element into the scalar `intermediate_size` for
    // back-compat with code that reads it as a scalar.
    int   altup_num_inputs;       // 4 on E2B/E4B
    int   altup_active_idx;       // 0 on E2B/E4B
    bool  altup_correct_scale;    // true on E2B/E4B
    float altup_coef_clip;        // 120.0 on E2B/E4B (training-only clip)
    int   laurel_rank;            // 64 on E2B/E4B
    int   vocab_size_per_layer_input;  // 262144 on E2B
    float gemma3n_rope_local_base_freq; // sliding-layer rope theta
    std::vector<int>   gemma3n_per_layer_intermediate;
    std::vector<float> gemma3n_activation_sparsity;

    // ── GLM-5.1 DSA (Differential Sparse Attention) indexer ────────
    // Per-layer indexer selects top-k tokens for sparse attention.
    // Zero/empty on non-GLM models.
    int index_topk = 0;         // 2048 on GLM-5.1
    int index_head_dim = 0;     // 128 on GLM-5.1
    int index_n_heads = 0;      // 32 on GLM-5.1
    // Per-layer indexer type: "full" (computes indices from scratch)
    // or "shared" (reuses previous layer's indices). Empty on non-GLM.
    std::vector<std::string> indexer_types;

    // ── Storage dtype as declared on disk (for the safetensors loader).
    std::string torch_dtype;   // "bfloat16", "float16", "float32".

    // ── Offline-quantized checkpoint metadata ────────────────────────
    // Empty `quant_method` means an unquantized (bf16/fp16/fp32) ckpt.
    // Recognised values:
    //   * "gptq"  — GPTQ INT4/INT8 with packed `qweight`/`qzeros`/
    //               `scales` (+ optional `g_idx` for desc_act).
    //   * "awq"   — AWQ INT4 with similar tensors but distinct packing
    //               and asymmetric (zero_point) by default.
    //   * "fp8"   — compressed-tensors / static FP8 (handled elsewhere).
    //
    // `quant_bits` is 4 or 8 for gptq/awq.
    // `quant_group_size` is the per-group axis-K stride (typical 128;
    // -1 = per-channel, no group dim).
    // `quant_desc_act` (GPTQ act-order) and `quant_sym` (symmetric vs
    // zero-point) gate the dispatch path; v1 supports `desc_act=false`
    // and `sym=true` only.
    std::string quant_method;
    int   quant_bits = 0;
    int   quant_group_size = 0;
    bool  quant_desc_act = false;
    bool  quant_sym = true;
    bool  quant_zero_point = false;
    // True when a compressed-tensors ckpt sets a non-null
    // `quantization_config.kv_cache_scheme` (FP8 / INT8 KV cache). We
    // don't yet store K/V quantized — surface the mismatch loudly so
    // the user knows generation may drift slightly vs the calibrated
    // reference. Wiring quantized KV is its own milestone.
    bool  kv_cache_scheme_present = false;

    // ── Multimodal text-tower extraction ────────────────────────────
    // For multimodal checkpoints (Mistral3ForConditionalGeneration,
    // LlavaForConditionalGeneration, …) HF stores the LLM weights under
    // a top-level prefix ("language_model.") alongside vision tower /
    // projector weights. The CUDA driver currently only runs the LLM
    // forward, so the loader strips `mm_lm_strip_prefix` from each name
    // and skips any tensor whose name starts with one of the entries in
    // `mm_skip_prefixes`. Empty for text-only checkpoints.
    std::string mm_lm_strip_prefix;
    std::vector<std::string> mm_skip_prefixes;
};

// Parse `<snapshot_dir>/config.json`. Throws on missing required fields.
HfConfig parse_hf_config(const std::filesystem::path& path);

}  // namespace pie_cuda_driver

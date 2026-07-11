#pragma once

// Parser for HuggingFace `config.json` plus arch-name canonicalization.
//
// Pie's reference loader at `pie/src/pie_driver/loader.py` reads the same
// fields; we mirror its surface so the C++ driver and the Python driver
// agree on hyperparameter interpretation. See `model/<arch>.py::ModelConfig.from_dict`
// for the per-arch field expectations.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace pie_portable_driver {

// Canonical Pie architecture identifiers. Must match the strings the
// runtime accepts in `bootstrap::ModelConfig::arch_name` and the inferlet
// instruct-template registry at `runtime/src/model/instruct.rs`.
enum class PieArch {
    Qwen3,
    Qwen2,
    Llama3,
    Gemma2,
    Gemma3,
    Gemma4,
    Gemma3n,     // Gemma 3n (E2B / E4B); HF model_type "gemma3n" /
                 // "gemma3n_text". Dense per-layer attention with the
                 // Gemma 3 sliding/full mix. v1 implementation skips
                 // AltUp / PLE / Laurel (the parameter-efficient
                 // tricks); the runtime executes a vanilla Gemma 3-
                 // style graph and produces approximately-correct text.
                 // Full architectural fidelity is a follow-up.
    Mistral3,
    Olmo3,
    GptOss,
    Phi3,
    Mixtral,
    Qwen3Moe,    // Qwen3-MoE family (Qwen3-30B-A3B etc.); HF model_type "qwen3_moe"
    Qwen3_5,     // Qwen 3.5 / 3.6 hybrid (gated delta + GQA);
                 // HF model_type "qwen3_5" (dense) or "qwen3_5_moe" (MoE)
    Phi3Small,   // Phi-3-small (microsoft); HF model_type "phi3small". Uses
                 // LayerNorm with bias, fused query_key_value, packed
                 // mlp.up_proj (gate||up), self_attn.dense, mup parameter-
                 // ization. v1 graph treats blocksparse as causal (correct
                 // for prompts up to num_local_blocks * block_size = 1024
                 // tokens).
    Phi3_5Moe,   // Phi-3.5-MoE (microsoft); HF model_type "phimoe". Mixtral-
                 // style sparse MoE (per-expert w1/w2/w3) with LayerNorm-
                 // with-bias norms, Q/K/V/O biases, lm_head bias.
    GlmMoeDsa,   // GLM-5.1 (zai-org); HF model_type "glm_moe_dsa".
                 // DeepSeek-V3/Kimi-style MLA attention with LoRA-compressed
                 // Q/KV projections. MoE with 256 routed experts + 1 shared
                 // expert; first 3 layers are dense. Sigmoid routing with
                 // routed_scaling_factor. v1 skips the DSA (Differential
                 // Sparse Attention) indexer.
    Qwen3VL,     // Qwen3-VL (Qwen/Qwen3-VL-2B-Instruct); HF model_type
                 // "qwen3_vl". Qwen3 text decoder + a ViT vision tower
                 // (2D-RoPE, learned abs pos-embed, 2x2 spatial merge) plus
                 // 3 deepstack mergers injected into LLM layers 0/1/2 on image
                 // rows. The `vision_*` Hparams fields below describe the tower.
    Csm,         // CSM-1B (sesame/csm-1b); HF model_type "csm". Native audio
                 // OUTPUT (TTS): Llama-3.2-1B backbone + a 4-layer depth decoder
                 // (RVQ codebooks) + a Mimi codec vocoder. Driven via
                 // pie:core/audio-out (GENERATE_AUDIO), not a text forward.
};

// ── CSM-1B sub-configs (model_type "csm") ─────────────────────────────────────
// The backbone Llama-3.2-1B hparams live at the top level of config.json (parsed
// into the main Hparams fields). These two nested pieces describe the depth
// decoder (samples RVQ codebooks 1..31 of a frame) and the Mimi codec (codes ->
// 24 kHz waveform). Mirrors the CUDA loader's CsmDepthDecoderConfig / MimiCodecConfig.
struct CsmDepthDecoderConfig {
    int hidden_size = 1024;
    int backbone_hidden_size = 2048;
    int num_hidden_layers = 4;
    int num_attention_heads = 8;
    int num_key_value_heads = 2;
    int head_dim = 128;
    int intermediate_size = 8192;
    int num_codebooks = 32;
    int vocab_size = 2051;
    int max_position_embeddings = 33;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 500000.0f;
    float rope_factor = 32.0f;
    float rope_low_freq_factor = 0.001953125f;
    float rope_high_freq_factor = 0.0078125f;
    int rope_original_max_position = 16;
};

struct MimiCodecConfig {
    int hidden_size = 512;
    int codebook_dim = 256;          // vector_quantization_hidden_dimension
    int codebook_size = 2048;
    int num_quantizers = 32;
    int num_semantic_quantizers = 1;
    int num_filters = 64;
    std::vector<int> upsampling_ratios{8, 6, 5, 4};
    int xf_num_attention_heads = 8;
    int xf_num_key_value_heads = 8;
    int xf_head_dim = 64;
    int xf_intermediate_size = 2048;
    int xf_num_hidden_layers = 8;
    int xf_sliding_window = 250;
    float xf_rope_theta = 10000.0f;
    float norm_eps = 1e-5f;
    int sampling_rate = 24000;
    bool use_causal_conv = true;
    int upsample_groups = 512;
    int residual_kernel_size = 3;
    int kernel_size = 7;
    int last_kernel_size = 3;
};

struct CsmConfig {
    int text_vocab_size = 128256;     // embed_text_tokens rows
    int audio_vocab_size = 2051;      // per-codebook vocab (== top-level vocab_size)
    int num_codebooks = 32;
    int codebook_eos_token_id = 0;    // all-EOS frame == stop
    int audio_eos_token_id = 128003;
    int audio_token_id = 128002;
    CsmDepthDecoderConfig depth;
    MimiCodecConfig codec;
};

const char* pie_arch_name(PieArch a);

// Maps an HF `model_type` (e.g. "qwen3", "llama") to its Pie counterpart.
// Throws on unknown types.
PieArch hf_model_type_to_pie_arch(const std::string& hf_model_type);

// Subset of `config.json` we use. Optional fields have std::optional values
// so per-arch builders can branch on presence.
struct Hparams {
    PieArch arch;
    std::string hf_model_type;     // raw "qwen3"
    std::string torch_dtype;       // "bfloat16" / "float16" / "float32"

    // Common transformer hparams.
    std::int32_t num_hidden_layers = 0;
    std::int32_t num_attention_heads = 0;
    std::int32_t num_key_value_heads = 0;
    // Gemma 4 alternative attention: full_attention layers use a
    // SMALLER kv_heads count than sliding layers (e.g., 4 vs 16 on 31B).
    // 0 = not provided / not applicable.
    std::int32_t num_global_key_value_heads = 0;
    // Gemma 4 26B-A4B: parallel sparse-MoE block alongside the dense MLP.
    // When true, the loader expects per-layer experts.gate_up_proj /
    // experts.down_proj / router.{proj,scale,per_expert_scale} plus
    // pre_feedforward_layernorm_2 + post_feedforward_layernorm_{1,2}.
    bool         gemma4_enable_moe          = false;
    // Phi-3-small mup parameterization.
    float        mup_attn_multiplier         = 0.0f;  // 0 = unused
    float        mup_embedding_multiplier    = 0.0f;
    float        mup_width_multiplier        = 0.0f;
    // Phi-3-small blocksparse attention parameters.
    std::int32_t phi3small_block_size                  = 0;
    std::int32_t phi3small_num_local_blocks            = 0;
    std::int32_t phi3small_vert_stride                 = 0;
    std::int32_t phi3small_dense_attention_every_n_layers = 0;
    std::int32_t gemma4_moe_intermediate_size = 0;
    std::int32_t hidden_size = 0;
    std::int32_t intermediate_size = 0;
    std::int32_t head_dim = 0;             // computed if missing
    std::int32_t vocab_size = 0;
    std::int32_t max_position_embeddings = 0;

    // Norm / RoPE.
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1e6f;
    // Gemma3 / Gemma4: separate base frequency for sliding-window layers.
    // 0.0 = no override (fall back to rope_theta).
    float rope_local_base_freq = 0.0f;

    // Tied embeddings (qwen3 default true; llama3 typically false).
    bool tie_word_embeddings = true;

    // Sliding window attention. Some archs use it; populated when present.
    std::optional<std::int32_t> sliding_window;
    bool use_sliding_window = false;

    // RoPE scaling (LLaMA-3.1 NTK-by-parts, YARN, linear). Stored raw —
    // the per-arch graph builder interprets the structure.
    bool has_rope_scaling = false;
    std::string rope_scaling_type;             // "llama3" / "yarn" / "linear" / ""
    float rope_scaling_factor = 1.0f;
    float rope_scaling_low_freq_factor = 1.0f;
    float rope_scaling_high_freq_factor = 4.0f;
    std::int32_t rope_scaling_original_max_position = 0;
    // YaRN-specific (olmo3, Ministral 3, gpt-oss). Defaults match HF.
    float rope_yarn_attention_factor = 0.0f;   // 0 = use 0.1*ln(factor)+1
    float rope_yarn_beta_fast = 32.0f;
    float rope_yarn_beta_slow = 1.0f;

    // Per-layer attention type list (olmo3, gpt-oss, gemma4 etc.). One char
    // per layer: 's' (sliding_attention) or 'g' (full_attention). Empty if
    // the config doesn't specify a per-layer pattern.
    std::vector<char> layer_types;

    // ── Gemma 4 / 3n ──
    // Per-attention-type RoPE (gemma4 uses a `rope_parameters` dict keyed by
    // "full_attention" / "sliding_attention" with per-entry rope_theta and
    // partial_rotary_factor). For non-gemma4 archs these stay defaulted and
    // are unused.
    float gemma4_rope_theta_full = 1e6f;
    float gemma4_rope_theta_sliding = 1e4f;
    float gemma4_rope_partial_factor_full = 1.0f;     // 1.0 = full rotation
    float gemma4_rope_partial_factor_sliding = 1.0f;
    // Per-layer-type head_dim. Sliding head_dim lives in `head_dim`; full
    // attention reads `gemma4_head_dim_global`. 0 = same as head_dim.
    std::int32_t gemma4_head_dim_global = 0;
    // KV-cache sharing: last `gemma4_num_kv_shared_layers` layers reuse
    // upstream non-shared layer's K/V (matched by attention type). 0 = no
    // sharing. The shared layers still have k_proj/v_proj weights in the
    // safetensors; the driver loads them but never feeds them at inference.
    std::int32_t gemma4_num_kv_shared_layers = 0;
    // Per-Layer Embeddings: extra ple_dim residual injected after MLP each
    // layer. 0 = PLE disabled (gemma4 always >0 in shipped checkpoints).
    std::int32_t gemma4_ple_dim = 0;
    std::int32_t gemma4_ple_vocab = 0;
    bool         gemma4_use_double_wide_mlp = false;

    // ── Gemma 3n: AltUp / Laurel ──
    // altup_num_inputs (=4): number of parallel hidden streams.
    // altup_active_idx (=0): which stream the transformer block actually
    //   reads/writes; the others are predicted/corrected from the active
    //   stream's behavior via small router + 4×4 / 4×16 coefficient
    //   matrices learned per layer.
    // altup_correct_scale (=true): apply correct_output_scale to the
    //   active stream after each layer's correction step.
    // laurel_rank: low-rank dim for the Laurel residual MLP (=64 in E2B/E4B).
    std::int32_t altup_num_inputs    = 0;   // 0 = arch doesn't use AltUp
    std::int32_t altup_active_idx    = 0;
    bool         altup_correct_scale = false;
    std::int32_t laurel_rank         = 0;
    // Per-layer "Gaussian top-k" pre-activation sparsity. Layers where this
    // is > 0 zero out everything below `mean + icdf(p)*std` in the gate
    // projection (E2B/E4B: 0.95 on layers 0-9, 0.0 on the rest). Empty
    // vector = activation sparsity disabled.
    std::vector<float> activation_sparsity_pattern;

    // Logit softcap (gemma2; future gemma).
    std::optional<float> attn_logit_softcapping;
    std::optional<float> final_logit_softcapping;
    // Custom Q scaling — gemma2/3 use 1/sqrt(query_pre_attn_scalar)
    // instead of 1/sqrt(head_dim). 0 = use head_dim default.
    std::optional<float> query_pre_attn_scalar;

    // ── Mixture-of-Experts (Mixtral, Qwen2/3-MoE, GPT-OSS, etc.) ──
    // num_local_experts (Mixtral) / num_experts (Qwen-MoE) — total experts
    // per layer. 0 = not an MoE model.
    std::int32_t num_experts = 0;
    // num_experts_per_tok — top-k routing.
    std::int32_t num_experts_per_tok = 0;
    // moe_intermediate_size — per-expert FFN hidden size. Defaults to
    // intermediate_size when not specified separately.
    std::int32_t moe_intermediate_size = 0;
    // norm_topk_prob (qwen-moe / mixtral) — renormalize the selected
    // top-k expert weights so they sum to 1.
    bool norm_topk_prob = true;
    // Qwen 3.6 (qwen3_5_moe): shared dense expert that runs alongside the
    // routed experts. 0 disables (Qwen-MoE / Mixtral / gpt-oss don't have it).
    std::int32_t shared_expert_intermediate_size = 0;

    // ── Qwen 3.5 / 3.6 ──
    // Hybrid arch: 3-of-4 layers are gated-delta-rule "linear attention"
    // with recurrent matrix state, every 4th layer is standard GQA with
    // mrope + output gate. Vision tower + multi-token-prediction head
    // are present in checkpoints but ignored by the driver.
    bool         qwen35_attn_output_gate     = false;
    // GGUF stores the linear-attention V heads in *tiled* order (the
    // llama.cpp qwen35 converter permutes grouped->tiled when
    // num_v_heads != num_k_heads so ggml's tiled broadcast aligns); HF
    // safetensors keep the original grouped order. The graph expands Q/K
    // to match V using a tiled `ggml_repeat` when set, else a grouped
    // `repeat_interleave`.
    bool         qwen35_linear_v_tiled       = false;
    std::int32_t qwen35_full_attn_interval   = 0;
    std::int32_t qwen35_linear_num_k_heads   = 0;
    std::int32_t qwen35_linear_num_v_heads   = 0;
    std::int32_t qwen35_linear_k_head_dim    = 0;
    std::int32_t qwen35_linear_v_head_dim    = 0;
    std::int32_t qwen35_linear_conv_kernel   = 0;
    bool         qwen35_mrope_interleaved    = false;
    std::int32_t qwen35_mrope_section[3]     = {0, 0, 0};
    float        qwen35_partial_rotary_factor = 1.0f;
    // When true, the dense Qwen graph applies multimodal RoPE (ggml_rope_multi)
    // over a 4×-wide pos_input carrying per-token [t,h,w] axes (image/video
    // tokens get spatial/temporal positions; text tokens get [p,p,p], which
    // reduces mrope to plain RoPE). Set for Qwen3-VL. The reusable mrope fields
    // above (section/interleaved) drive it.
    bool         use_mrope                   = false;

    // ---- Vision tower (multimodal) -------------------------------------
    // Populated from the checkpoint's `vision_config` sub-dict when present.
    // `vision_hidden_size == 0` means "no vision tower" (text-only); every
    // vision code path is gated on it so text models are unaffected. Defaults
    // match Qwen3-VL-2B; other vision archs override via vision_config.
    std::int32_t vision_hidden_size        = 0;   // 0 = no tower
    std::int32_t vision_num_layers         = 0;
    std::int32_t vision_num_heads          = 0;
    std::int32_t vision_head_dim           = 0;   // hidden/heads if 0
    std::int32_t vision_intermediate_size  = 0;
    std::int32_t vision_patch_size         = 16;
    std::int32_t vision_temporal_patch_size = 2;
    std::int32_t vision_spatial_merge_size  = 2;  // 2x2 -> 4 patches/token
    std::int32_t vision_in_channels        = 3;
    std::int32_t vision_out_hidden         = 0;   // projector out dim (= text hidden)
    std::int32_t vision_num_pos_embed      = 0;   // learned abs pos-embed table rows
    float        vision_rope_theta         = 10000.0f;
    float        vision_ln_eps             = 1e-6f;
    // Deepstack merger source layers (Qwen3-VL: {5,11,17}); injected into the
    // text decoder at LLM layers 0..n-1 on image rows. -1 = unused slot.
    std::int32_t vision_deepstack_layers[3] = {-1, -1, -1};
    std::int32_t vision_num_deepstack       = 0;
    // Gemma-4 vision: 2D average-pool kernel applied to the patch grid (the
    // SigLIP merger). 0 for archs (Qwen3-VL) that use a 2x2 token-merge merger.
    std::int32_t vision_pool_kernel         = 0;
    bool         vision_clipped_linears     = false;  // Gemma-4 QK-clip linears

    // ── Audio tower (Gemma-4 Conformer encoder) ───────────────────────────────
    // Populated from the checkpoint's `audio_config` sub-dict when present.
    // `audio_hidden_size == 0` means "no audio tower". Defaults match
    // google/gemma-4-E2B/E4B `audio_config`.
    std::int32_t audio_hidden_size          = 0;     // 0 = no tower
    std::int32_t audio_num_layers           = 0;     // Conformer blocks
    std::int32_t audio_num_heads            = 0;
    std::int32_t audio_conv_kernel          = 5;     // depthwise causal conv1d
    std::int32_t audio_sscp_channels0       = 128;   // subsampling_conv_channels[0]
    std::int32_t audio_sscp_channels1       = 32;    // subsampling_conv_channels[1]
    std::int32_t audio_out_proj_dims        = 1536;  // output_proj out dim
    std::int32_t audio_feature_size         = 128;   // mel bins (SSCP input freq)
    std::int32_t audio_chunk_size           = 12;    // attention_chunk_size
    std::int32_t audio_context_left         = 13;    // attention_context_left
    std::int32_t audio_context_right        = 0;     // attention_context_right
    float        audio_logit_cap            = 50.0f; // attention_logit_cap
    float        audio_residual_weight      = 0.5f;
    float        audio_ln_eps               = 1e-6f;

    // ── CSM-1B audio output (model_type "csm") ────────────────────────────────
    // Present only for CSM; the backbone Llama hparams use the main fields above.
    std::optional<CsmConfig> csm;
};

Hparams parse_hf_config(const std::filesystem::path& config_json_path);

}  // namespace pie_portable_driver

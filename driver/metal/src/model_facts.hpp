#pragma once

/// The driver's one read of `config.json`, and the mapping from what it says to
/// the geometry a family is set up with.
///
/// This lived inside `context.cpp` while the runtime was its only caller. It is
/// its own translation unit now because it stopped being: a benchmark or a test
/// pointed at a real checkpoint needs the SAME answer the runtime gets, and the
/// alternative -- each caller spelling out a model's shape as literals -- is the
/// restatement pattern that has produced most of this driver's shape bugs. A
/// config the runtime accepts and a bench transcribes by hand agree only until
/// someone edits one of them.

#include <cstdint>
#include <optional>
#include <string>

// `SetupConfig` is forward-declared rather than included. This header used to
// pull in `batch/forward.hpp` -- the whole executor -- for one function
// signature, which meant anything that wanted to know a checkpoint's
// `model_type` had to be able to link the driver. Reading config.json is the
// smaller question and now has the smaller dependency.
namespace pie::metal::batch { struct SetupConfig; }

namespace pie::metal {

// Slice member `m`'s forward fields out of the batch-wide CSR view (§5.2).
// `member_count` is `members.size()` for THIS launch — every CSR indptr
// slice must carry exactly `member_count + 1` entries when present.
//
// `resolved` (Phase 2, C3): when non-null, the member's token/position/KV-
// page/readout fields come from the descriptor-resolved `FireGeometry`
// INSTEAD of the wire CSR slices — a device-geometry program's wire span is
// an empty placeholder (`pie::driver::fire::LaunchView`'s own doc comment); the
// channel-resolved geometry is the only truth for those fields. The
// recurrent-state slot bookkeeping (rs_slot_id/rs_reset) is unrelated to
// device-geometry classification and always comes from the wire.
struct ModelFacts {
    /// The `quantization` block's width and group, which the SHAPES cannot
    /// recover: a weight quantized 8-bit in groups of 64 and one quantized
    /// 4-bit in groups of 128 pack to exactly the same u32 count against
    /// exactly the same number of scales. Assuming one of them is how an 8-bit
    /// llama checkpoint was refused as "g128/b4" -- a quantization it does not
    /// have and nobody ships.
    ///
    /// Zero means the config declared none, which is a checkpoint whose tensors
    /// are dense.
    int quant_bits = 0;
    int quant_group_size = 0;

    std::uint32_t vocab_size = 32000;
    std::uint32_t max_model_len = 8192;
    std::string arch_name = "llama";
    bool has_linear_attn = false;
    // Qwen3.5/3.6 carry the RoPE hyperparameters in a nested `rope_parameters`
    // object rather than at the top level, so a reader that only knows the flat
    // key finds nothing and silently keeps its default. Nothing fails: the
    // rotated channels just come out wrong, and the error compounds layer over
    // layer until the activations saturate. tests/mlx/loader/hf_config.cpp
    // reads the nested form; this must agree with it.
    float rope_theta = 1.0e7f;
    float partial_rotary_factor = 0.25f;
    // ── Gemma 4 ──
    // Its shape is per attention type, and `rope_parameters` is nested one
    // level deeper again: `{full_attention: {...}, sliding_attention: {...}}`
    // rather than one object for the whole stack. Defaults are E2B's.
    // ── GPT-OSS ──
    // Non-zero marks "this config was read as gpt-oss". Its shape is flat --
    // no nested text_config -- and its rope is YaRN, whose four parameters sit
    // in `rope_scaling`.
    int go_num_hidden_layers = 0;
    int go_hidden_size = 0;
    int go_vocab_size = 0;
    int go_num_attention_heads = 0;
    int go_num_key_value_heads = 0;
    int go_head_dim = 0;
    int go_sliding_window = 0;
    int go_num_local_experts = 0;
    int go_num_experts_per_tok = 0;
    int go_intermediate_size = 0;
    int go_rope_original_max_position = 4096;
    float go_rms_norm_eps = 1e-5f;
    float go_swiglu_limit = 7.0f;
    float go_rope_theta = 150000.0f;
    float go_rope_factor = 32.0f;
    float go_rope_beta_fast = 32.0f;
    float go_rope_beta_slow = 1.0f;

    int g4_num_hidden_layers = 0;   // non-zero marks "this config was read as gemma4"
    int g4_hidden_size = 0;
    int g4_intermediate_size = 0;
    int g4_num_attention_heads = 0;
    int g4_num_key_value_heads = 0;
    int g4_head_dim = 0;            // sliding layers
    int g4_global_head_dim = 0;     // full layers
    int g4_sliding_window = 0;
    int g4_num_kv_shared_layers = 0;
    int g4_per_layer_emb_dim = 0;   // hidden_size_per_layer_input
    int g4_full_attn_interval = 0;  // derived from `layer_types`
    bool g4_double_wide_mlp = false;
    float g4_final_softcap = 0.0f;
    float g4_rope_theta_full = 1.0e6f;
    float g4_rope_theta_sliding = 1.0e4f;
    float g4_full_partial_rotary = 0.25f;
    // The mixture (gemma-4-26B-A4B) and its k-eq-V attention.
    bool g4_enable_moe = false;
    int g4_num_experts = 0;
    int g4_experts_per_token = 0;
    int g4_moe_intermediate = 0;
    bool g4_attention_k_eq_v = false;
    int g4_num_global_kv_heads = 0;
    // ── The llama-shaped families ──
    // `llama`, `llama3`, `mistral`, `qwen2`, `qwen3` and the two MoE variants.
    // One set of fields, because they differ in VALUES and not in shape: a
    // routed model sets the three expert fields, a Qwen3 sets `qk_norm`, and
    // the rest of the config is read identically. Non-zero `ll_num_hidden_layers`
    // marks "this config was read as one of them".
    int ll_num_hidden_layers = 0;
    int ll_hidden_size = 0;
    int ll_vocab_size = 0;
    int ll_num_attention_heads = 0;
    int ll_num_key_value_heads = 0;
    int ll_head_dim = 0;
    int ll_intermediate_size = 0;
    int ll_num_experts = 0;
    int ll_num_experts_per_tok = 0;
    int ll_moe_intermediate_size = 0;
    float ll_rms_norm_eps = 1e-5f;
    float ll_rope_theta = 500000.0f;
    float ll_rope_scale = 1.0f;
    std::string ll_rope_scaling_kind;
    // Llama 3.1's piecewise schedule. `factor` is `ll_rope_scale`, shared with
    // the linear kind; these three are its own. Defaults are HF's, so a config
    // that names the type but omits a knob gets the schedule everyone else
    // gets rather than a division by zero.
    float ll_rope_low_freq_factor = 1.0f;
    float ll_rope_high_freq_factor = 4.0f;
    int ll_rope_original_max_position = 8192;
    bool ll_qk_norm = false;
    bool ll_tied_embeddings = true;
    bool ll_norm_topk_prob = true;
    // ── Qwen3.5 / Qwen3-Next (the GDN hybrid) ──
    // Its shape used to be a struct full of defaults compiled into the driver,
    // which meant the family ran exactly one checkpoint and mis-ran any other
    // silently. These are the keys that shape actually comes from.
    int q35_num_hidden_layers = 0;
    int q35_hidden_size = 0;
    int q35_vocab_size = 0;
    int q35_num_attention_heads = 0;
    int q35_num_key_value_heads = 0;
    int q35_head_dim = 0;
    int q35_intermediate_size = 0;
    // The linear-attention block. `conv_dim` and the value total are DERIVED
    // from these rather than read, because a config cannot state them
    // inconsistently with the head counts and this driver should not be able
    // to either.
    int q35_linear_key_heads = 0;
    int q35_linear_value_heads = 0;
    int q35_linear_key_head_dim = 0;
    int q35_linear_value_head_dim = 0;
    int q35_linear_conv_kernel = 0;
    /// One full-attention layer every `interval`; -1 for an irregular pattern
    /// the geometry refuses rather than approximates.
    int q35_full_attn_interval = 0;
    // The routed FFN.
    int q35_num_experts = 0;
    int q35_num_experts_per_tok = 0;
    int q35_moe_intermediate_size = 0;
    int q35_shared_expert_intermediate = 0;
    int q35_decoder_sparse_step = 1;
    /// Layers the config exempts from routing. Carried as a COUNT because the
    /// geometry refuses any non-empty one; the number is for the message.
    int q35_mlp_only_layer_count = 0;
    float q35_rms_norm_eps = 1e-6f;
    bool q35_tied_embeddings = true;
    bool q35_norm_topk_prob = true;

    // Which storage schema this driver authors against. Parsed here because
    // this is already the driver's one read of `config.json`; the loader no
    // longer opens it (`loader/architecture.md` §10.4).
    std::string model_type;
};

/// Parse `<hf_path>/config.json`. A missing or unparseable file yields the
/// defaults, which every caller then refuses on: there is no configuration
/// worth guessing at.
ModelFacts read_model_facts(const std::string& hf_path);

/// The same facts, read out of a `pie.model/1` descriptor instead of derived
/// from `config.json`. `nullopt` when there is no descriptor to read, and the
/// caller falls back to [`read_model_facts`].
std::optional<ModelFacts> read_model_facts_from_descriptor(
    const std::string& descriptor_path);

/// Fill the family-specific half of a `SetupConfig` from those facts.
void fill_family_geometry(pie::metal::batch::SetupConfig& cfg, const ModelFacts& facts);

}  // namespace pie::metal

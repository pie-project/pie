#pragma once

// Model: HF safetensors → ggml backend tensors.
//
// Owns the Hparams, the safetensors mmap, the ggml metadata context, and the
// backend buffer. Per-arch tensor name tables live in `model.cpp` and select
// from the HF-canonical names (e.g. "model.layers.N.self_attn.q_proj.weight")
// — no GGUF involved.
//
// For v1 the only supported backend is CPU. CUDA / Metal will plug in via the
// same `ggml_backend_t` API in M11.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "hf_config.hpp"
#include "safetensors.hpp"

namespace pie_portable_driver {

// Per-layer weights, raw split-projection layout (matches HF; no QKV/MLP
// fusion). The graph builder is free to fuse on the fly. Optional tensors
// (biases, QK-norm) are nullptr when the architecture doesn't use them.
struct LayerWeights {
    // Norms.
    ggml_tensor* attn_norm     = nullptr;   // input_layernorm (always)
    ggml_tensor* ffn_norm      = nullptr;   // post_attention_layernorm (llama)
                                            // OR pre_feedforward_layernorm (gemma)
    // Gemma family: extra norms for the post-attn / post-FFN residual paths.
    ggml_tensor* post_attn_norm = nullptr;  // gemma2/3/4
    ggml_tensor* post_ffn_norm  = nullptr;  // gemma2/3/4

    // Self-attention projections.
    ggml_tensor* q_proj   = nullptr;
    ggml_tensor* k_proj   = nullptr;
    ggml_tensor* v_proj   = nullptr;
    ggml_tensor* o_proj   = nullptr;
    // Optional QKV biases (qwen2 has them; qwen3/llama3 don't).
    ggml_tensor* q_proj_b = nullptr;
    ggml_tensor* k_proj_b = nullptr;
    ggml_tensor* v_proj_b = nullptr;

    // QK normalization (qwen3 only).
    ggml_tensor* q_norm = nullptr;
    ggml_tensor* k_norm = nullptr;

    // MLP — gated SwiGLU. For dense (non-MoE) archs, these hold the
    // single-expert weights. For MoE archs they're nullptr and the
    // stacked-expert tensors below carry per-layer weights instead.
    ggml_tensor* gate_proj = nullptr;
    ggml_tensor* up_proj   = nullptr;
    ggml_tensor* down_proj = nullptr;

    // ── MoE: per-layer router + stacked expert tensors ──
    ggml_tensor* moe_router    = nullptr;  // [hidden, n_experts]
    ggml_tensor* moe_gate_exps = nullptr;  // [hidden, ff, n_experts]
    ggml_tensor* moe_up_exps   = nullptr;  // [hidden, ff, n_experts]
    ggml_tensor* moe_down_exps = nullptr;  // [ff, hidden, n_experts]
};

struct ModelWeights {
    ggml_tensor* tok_embd     = nullptr;  // model.embed_tokens.weight
    ggml_tensor* output_norm  = nullptr;  // model.norm.weight
    ggml_tensor* output_head  = nullptr;  // lm_head.weight (nullptr if tied)
    // RoPE per-dim frequency factors (LLaMA-3.1+ NTK-by-parts). nullptr
    // for models that use plain θ-only RoPE.
    ggml_tensor* freq_factors = nullptr;
    std::vector<LayerWeights> layers;
};

class Model {
public:
    // Loads from a HuggingFace snapshot directory. Expects:
    //   <snapshot_dir>/config.json
    //   <snapshot_dir>/model.safetensors  OR  <...>/model.safetensors.index.json
    //
    // `prefer_gpu`: if true, picks the best registered GPU backend
    // (CUDA / Metal / Vulkan, depending on what was compiled into ggml).
    // Falls back to CPU if no GPU backend is available.
    explicit Model(const std::filesystem::path& snapshot_dir,
                   bool prefer_gpu = false);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Hparams&       hparams() const noexcept { return hparams_; }
    const ModelWeights&  weights() const noexcept { return weights_; }
    const std::filesystem::path& snapshot_dir() const noexcept { return snapshot_dir_; }
    std::size_t          buffer_size() const noexcept;
    ggml_backend_t       backend() const noexcept { return backend_; }
    std::string          backend_name() const noexcept;

    // Friendly arch string for the READY handshake.
    std::string arch_name_pie() const { return pie_arch_name(hparams_.arch); }

    // Best-effort dtype string for `DriverCapabilities::activation_dtype`.
    // Reflects the model's stored weight dtype, not graph activations.
    std::string activation_dtype_str() const;

private:
    // Per-arch loader description. Captures the structural variations across
    // archs (extra norms, fused/biased QKV, MoE naming) so a single set of
    // loader helpers can drive all of them. Specialized cases that don't fit
    // (phi3's fused QKV, llama3's rope freq factors) are layered on top.
    struct LoaderSpec {
        // Suffix of the per-layer norm sitting on the FFN side. Most archs
        // reuse "post_attention_layernorm"; the gemma family puts a
        // pre-FFN norm at "pre_feedforward_layernorm" and adds extras.
        std::string ffn_norm_name = "post_attention_layernorm";

        // Gemma family: extra norms on the post-attention and post-FFN
        // residual paths.
        bool has_post_attn_norm = false;
        bool has_post_ffn_norm  = false;

        // QKV biases (qwen2 only) and per-head Q/K RMSNorm (qwen3, gemma3).
        bool has_qkv_bias = false;
        bool has_qk_norm  = false;

        // MoE naming family. None = dense MLP path. The two MoE flavors
        // differ in tensor naming only:
        //   MlpExperts:    mlp.gate.weight, mlp.experts.E.{gate,up,down}_proj
        //   BlockSparseMoe: block_sparse_moe.gate.weight,
        //                   block_sparse_moe.experts.E.{w1,w2,w3}
        enum class MoeNaming { None, MlpExperts, BlockSparseMoe };
        MoeNaming moe = MoeNaming::None;
    };

    void build_qwen3_();
    void build_qwen2_();
    void build_llama3_();
    void build_mistral3_();
    void build_phi3_();
    void build_gemma2_();
    void build_gemma3_();
    void build_gemma4_();
    void build_gpt_oss_();
    void build_qwen3_5_();   // qwen3-moe family
    void build_mixtral_();

    // Load tok_embd / output_norm / (optional) output_head and resize the
    // per-layer vector. Common to all archs.
    void load_top_level_();
    // Per-layer dense or MoE weight loader, driven by `spec`.
    void load_layer_(std::int32_t i, const LoaderSpec& spec);
    // Per-layer MoE branch (router + 3 stacked-expert tensors).
    void load_moe_layer_(std::int32_t i,
                         const std::string& path_prefix,
                         LoaderSpec::MoeNaming kind);
    // LLaMA-3.1 NTK-by-parts: synthesize the rope freq_factors tensor if
    // `hparams_.rope_scaling_type == "llama3"`. No-op otherwise.
    void synth_llama3_freq_factors_();

    // Helper: create a ggml tensor mirroring the HF tensor's shape (with
    // dims reversed) and dtype, then load the safetensor data into the
    // backend buffer once allocated.
    ggml_tensor* declare_(const std::string& hf_name);

    // Load-time slice from a fused safetensor. Used by phi3 (qkv_proj
    // → q/k/v_proj, gate_up_proj → gate/up_proj). The new tensor's shape
    // is `[in_dim, out_dim]` (ggml convention, dim-reversed from HF). At
    // load time we copy `out_dim * in_dim * dtype_size` bytes from
    // `src_offset_bytes` of the fused safetensor.
    ggml_tensor* declare_slice_(const std::string& fused_hf_name,
                                const std::string& dbg_name,
                                std::int64_t in_dim,
                                std::int64_t out_dim,
                                std::size_t  src_offset_bytes,
                                ggml_type    dtype);
    void         load_into_backend_();

    std::filesystem::path snapshot_dir_;
    Hparams               hparams_;

    std::unique_ptr<SafetensorsArchive> archive_;
    ggml_backend_t        backend_ = nullptr;
    ggml_context*         ctx_     = nullptr;
    ggml_backend_buffer_t buf_     = nullptr;

    // Each entry tells `load_into_backend_` where to copy data from.
    // `src_offset_bytes` and `copy_bytes` are 0 for full-tensor loads;
    // non-zero `copy_bytes` indicates a sliced load from a fused
    // safetensor (phi3 qkv_proj / gate_up_proj).
    // `dst_offset_bytes` non-zero indicates the source's data is copied
    // into a slot of an already-declared destination tensor (stacked
    // MoE experts).
    struct DeclaredTensor {
        std::string  hf_name;
        ggml_tensor* tensor;
        std::size_t  src_offset_bytes = 0;
        std::size_t  copy_bytes       = 0;
        std::size_t  dst_offset_bytes = 0;
    };
    std::vector<DeclaredTensor> declared_;

    // Helper: allocate a stacked experts tensor `[in_dim, out_dim, n_experts]`
    // and queue per-expert source-name copies. Each source must contain a
    // tensor of HF shape [out_dim, in_dim] (i.e. ggml ne [in_dim, out_dim]).
    // At load time the i-th source's bytes are copied to slot i along the
    // 3rd dim. Returns the allocated tensor.
    ggml_tensor* declare_stacked_experts_(
        const std::string& dbg_name,
        std::int64_t in_dim, std::int64_t out_dim, std::int32_t n_experts,
        const std::vector<std::string>& per_expert_hf_names,
        ggml_type dtype);

    // Synthesized constant tensors (e.g. RoPE freq_factors) computed at
    // load time; allocated alongside weights, copied in afterwards.
    struct SynthTensor {
        ggml_tensor* tensor;
        std::vector<std::uint8_t> data;
    };
    std::vector<SynthTensor> synth_;

    ModelWeights weights_;
};

}  // namespace pie_portable_driver

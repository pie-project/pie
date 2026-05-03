#pragma once

// Model: HF safetensors → ggml backend tensors.
//
// Owns the Hparams, the safetensors mmap, the ggml metadata context, and
// the backend buffer. Per-arch builders live in model.cpp; they resolve
// HF-canonical tensor names (e.g. "model.layers.N.self_attn.q_proj.weight")
// — no GGUF involved. Multimodal-wrapper checkpoints (Ministral 3, Gemma 4)
// substitute the canonical "model." prefix via Model::tname_().
//
// Backend selection follows ggml's registered devices: prefer_gpu picks
// CUDA / Metal / Vulkan via ggml_backend_init_best() and falls back to
// CPU when no GPU device is available.

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
    // Optional QKV biases (qwen2 has them; qwen3/llama3 don't). gpt-oss
    // also has an o_proj bias.
    ggml_tensor* q_proj_b = nullptr;
    ggml_tensor* k_proj_b = nullptr;
    ggml_tensor* v_proj_b = nullptr;
    ggml_tensor* o_proj_b = nullptr;

    // QK normalization (qwen3 only).
    ggml_tensor* q_norm = nullptr;
    ggml_tensor* k_norm = nullptr;

    // gpt-oss attention sinks. One learned scalar per Q-head; passed to
    // ggml_flash_attn_ext_add_sinks at the per-layer attention call. Null
    // for archs without sinks. Shape: [num_q_heads], stored F32 after the
    // BF16→F32 conversion (ggml's flash_attn_ext_add_sinks expects F32).
    ggml_tensor* attn_sinks = nullptr;

    // ── Gemma 4 ──
    // layer_scalar: per-layer learnable scalar applied to the layer's
    // output residual (shape [1] in shipped E2B/E4B). Null for non-gemma4.
    ggml_tensor* layer_scalar = nullptr;
    // PLE (Per-Layer Embeddings): each layer carries a small auxiliary
    // residual block that mixes a per-layer per-token signal back into the
    // hidden stream. Shapes (E2B): ple_gate [ple_dim, hidden],
    // ple_proj [hidden, ple_dim], ple_norm [hidden].
    ggml_tensor* ple_gate = nullptr;
    ggml_tensor* ple_proj = nullptr;
    ggml_tensor* ple_norm = nullptr;

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
    // Gemma 4 26B-A4B (MoE): the router's pre-projection rmsnorm uses
    // its own per-channel weight (router.scale, [hidden]) and a per-
    // expert gain after top-K renorm (router.per_expert_scale,
    // [n_experts]).
    ggml_tensor* moe_router_scale          = nullptr;
    ggml_tensor* moe_router_per_expert_scale = nullptr;
    // Gemma 4 26B-A4B parallel-MoE block: dense and MoE branches each
    // have their own post-norm; their sum is fed into the layer's
    // existing `post_ffn_norm`. The MoE branch also has its own pre-norm.
    ggml_tensor* gemma4_moe_pre_ffn_norm_2  = nullptr;
    ggml_tensor* gemma4_moe_post_ffn_norm_1 = nullptr;
    ggml_tensor* gemma4_moe_post_ffn_norm_2 = nullptr;

    // gpt-oss MoE biases (router + per-expert per-output-dim biases). Null
    // on Mixtral / Qwen3-MoE.
    ggml_tensor* moe_router_b   = nullptr;  // [n_experts]
    ggml_tensor* moe_gate_exps_b = nullptr; // [ff, n_experts]
    ggml_tensor* moe_up_exps_b   = nullptr; // [ff, n_experts]
    ggml_tensor* moe_down_exps_b = nullptr; // [hidden, n_experts]

    // ── Qwen 3.6 (qwen3_5_moe) shared-expert path ──
    // Per HF Qwen3_5MoeSparseMoeBlock: top-K routed experts run in parallel
    // with one always-active dense MLP, gated by sigmoid(shared_gate(x)).
    ggml_tensor* moe_shared_gate_proj = nullptr; // [hidden, shared_ff]
    ggml_tensor* moe_shared_up_proj   = nullptr; // [hidden, shared_ff]
    ggml_tensor* moe_shared_down_proj = nullptr; // [shared_ff, hidden]
    ggml_tensor* moe_shared_gate      = nullptr; // [hidden, 1]   sigmoid mixer

    // ── Qwen 3.5 / 3.6 linear-attention layer (gated-delta-rule) ──
    // Populated only for layers where hparams.layer_types[il] == 'l'.
    // Tensor names mirror modeling_qwen3_5.py::Qwen3_5GatedDeltaNet.
    ggml_tensor* lin_in_proj_qkv = nullptr;  // [hidden, 2*key_dim+value_dim]
    ggml_tensor* lin_in_proj_z   = nullptr;  // [hidden, value_dim]
    ggml_tensor* lin_in_proj_a   = nullptr;  // [hidden, num_v_heads]
    ggml_tensor* lin_in_proj_b   = nullptr;  // [hidden, num_v_heads]
    ggml_tensor* lin_conv1d      = nullptr;  // [conv_kernel, conv_dim] after squeeze
    ggml_tensor* lin_A_log       = nullptr;  // [num_v_heads]
    ggml_tensor* lin_dt_bias     = nullptr;  // [num_v_heads]
    ggml_tensor* lin_norm        = nullptr;  // [head_v_dim] — per-head RMSNorm
    ggml_tensor* lin_out_proj    = nullptr;  // [value_dim, hidden]

    // Qwen 3.5 full-attention layer with attn_output_gate=true: q_proj
    // output is 2× expected (split [Q, gate]). Loader uses the existing
    // q_proj field; the graph builder slices it.

    // ── Gemma 3n: AltUp (Alternative Updates) ──
    // Maintains `altup_num_inputs` (=4) parallel hidden streams. Each
    // layer predicts the streams from one "active" stream, runs the
    // standard transformer block on the active prediction, then
    // corrects all four predictions using the actual block output.
    // See graph_gemma3n.cpp.
    ggml_tensor* altup_predict_coefs = nullptr;  // [altup_num_inputs**2, altup_num_inputs] = [16, 4]
    ggml_tensor* altup_correct_coefs = nullptr;  // [altup_num_inputs, altup_num_inputs] = [4, 4]
    ggml_tensor* altup_router        = nullptr;  // [altup_num_inputs, hidden] = [4, hidden]
    ggml_tensor* altup_router_norm   = nullptr;  // [hidden] RMSNorm before router
    ggml_tensor* altup_correct_scale = nullptr;  // [hidden] scaling vector for active stream

    // ── Gemma 3n: Laurel (low-rank residual through MLPs) ──
    // y = post_laurel_norm(linear_right(linear_left(x))); the layer's
    // MLP output is mlp(x) + laurel(active_pre_mlp).
    ggml_tensor* laurel_left  = nullptr;  // [laurel_rank, hidden]
    ggml_tensor* laurel_right = nullptr;  // [hidden, laurel_rank]
    ggml_tensor* laurel_norm  = nullptr;  // [hidden] post_laurel_norm

    // ── Gemma 3n: per-layer PLE gate/projection ──
    // Each layer projects its slice of the global PLE table into hidden_size
    // and adds (gated) into the active stream. Distinct from Gemma 4 PLE
    // (different shapes / wiring); Gemma 3n's per-layer PLE is laid out as
    //   per_layer_input_gate     [hidden_per_layer_input, hidden]
    //   per_layer_projection     [hidden, hidden_per_layer_input]
    //   post_per_layer_input_norm[hidden]
    ggml_tensor* gemma3n_ple_gate = nullptr;
    ggml_tensor* gemma3n_ple_proj = nullptr;
    ggml_tensor* gemma3n_ple_norm = nullptr;
};

struct ModelWeights {
    ggml_tensor* tok_embd     = nullptr;  // model.embed_tokens.weight
    ggml_tensor* output_norm  = nullptr;  // model.norm.weight
    ggml_tensor* output_head  = nullptr;  // lm_head.weight (nullptr if tied)
    // RoPE per-dim frequency factors (LLaMA-3.1+ NTK-by-parts). nullptr
    // for models that use plain θ-only RoPE.
    ggml_tensor* freq_factors = nullptr;

    // ── Gemma 4 ──
    // Proportional-RoPE freq_factors for full-attention layers. Shape
    // [head_dim_global/2] F32. Null on archs without proportional RoPE.
    ggml_tensor* gemma4_rope_full_factors = nullptr;

    // ── Gemma 4 PLE — top-level ──
    // embed_tokens_per_layer: token-identity component, shape
    // [num_layers * ple_dim, vocab]. Null when PLE is disabled.
    ggml_tensor* ple_token_embed   = nullptr;
    // per_layer_model_projection: context component projection from main
    // hidden state to all layers' PLE inputs. Shape [num_layers*ple_dim, hidden].
    ggml_tensor* ple_model_proj    = nullptr;
    // per_layer_projection_norm: RMSNorm scale [ple_dim].
    ggml_tensor* ple_model_norm    = nullptr;

    // ── Gemma 3n: AltUp top-level projections ──
    // 3 projections each (one per non-active stream, indices 0..2). Used
    // to initialize the alt streams from the embedding and to combine the
    // 4 streams at the end of the stack.
    ggml_tensor* altup_proj_0 = nullptr;
    ggml_tensor* altup_proj_1 = nullptr;
    ggml_tensor* altup_proj_2 = nullptr;
    ggml_tensor* altup_unembed_proj_0 = nullptr;
    ggml_tensor* altup_unembed_proj_1 = nullptr;
    ggml_tensor* altup_unembed_proj_2 = nullptr;

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

    // CPU-side companion backend, used by `ForwardEngine`'s
    // `ggml_backend_sched` as the fallback when the primary backend
    // (e.g. ggml-vulkan, ggml-metal) doesn't implement an op the graph
    // requires. Returns null when the primary backend already IS the
    // CPU backend (no fallback needed). Lifetime is tied to this Model.
    ggml_backend_t       cpu_fallback() const noexcept { return cpu_fallback_; }

    // True iff the primary backend can run `ggml_argsort` on the full
    // vocab. ggml-vulkan caps argsort at 1024 cols which is far below
    // any modern LLM vocab (Qwen3 = 152k). When this returns false,
    // ForwardEngine should drive the slow-only path that downloads
    // raw logits and samples host-side, avoiding `ggml_top_k` in the
    // graph entirely. Probed once at model load time. Returns true on
    // CPU/CUDA/Metal (full coverage) and on Vulkan only for tiny
    // vocabs (which we don't care about).
    bool supports_in_graph_topk() const noexcept { return supports_in_graph_topk_; }

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
        // Per-layer "input_layernorm" present (true for L4MA / Gemma /
        // Qwen / Phi / Mistral). Olmo3 is post-norm only — set to false to
        // skip loading it.
        bool has_input_layernorm = true;

        // Suffix of the per-layer norm sitting on the FFN side. Most archs
        // reuse "post_attention_layernorm"; the gemma family puts a
        // pre-FFN norm at "pre_feedforward_layernorm" and adds extras.
        // Empty string = no pre-FFN norm (olmo3 post-norm-only).
        std::string ffn_norm_name = "post_attention_layernorm";

        // Gemma family + Olmo3: extra norms on the post-attention and
        // post-FFN residual paths. For Gemma these sandwich the existing
        // pre-norms; for Olmo3 they're the only norms (input + pre-FFN
        // norms are absent).
        bool has_post_attn_norm = false;
        bool has_post_ffn_norm  = false;

        // QKV biases (qwen2 only) and per-head Q/K RMSNorm (qwen3, gemma3,
        // olmo3). Olmo3 stores its q_norm/k_norm with full hidden-dim
        // weights (not per-head); both shapes load identically here.
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

    // Shared dense / dense+MoE skeleton: load_top_level_() then
    // load_layer_(i, spec) for every layer. Used by every arch that
    // doesn't need a custom per-layer loop (qwen2/qwen3/llama3/mistral3/
    // gemma2/gemma3/olmo3/qwen3_moe/mixtral). phi3, gemma4, gpt-oss have
    // their own builders below.
    void build_dense_(LoaderSpec spec);

    void build_qwen3_();
    void build_qwen2_();
    void build_llama3_();
    void build_mistral3_();
    void build_phi3_();
    void build_gemma2_();
    void build_gemma3_();
    void build_gemma4_();
    void build_gemma3n_();
    void build_olmo3_();
    // Gemma 4 proportional-RoPE: synthesize a freq_factors tensor that
    // makes ggml's rope_ext rotate only the first `rope_angles` dim-pairs
    // while pairing them at the model's head_dim/2 offset (not the rotary
    // dim's natural offset). Stored in synth_ so it lives alongside weights.
    ggml_tensor* synth_proportional_rope_factors_(std::int32_t head_dim,
                                                   std::int32_t rope_angles);
    void build_gpt_oss_();

    // Synth-allocate a stacked-experts tensor whose data comes from MXFP4-
    // quantized source tensors, dequanted to BF16 at load time. Optionally
    // de-interleaves rows (for gpt-oss's fused gate_up: gate at even rows,
    // up at odd rows) via row_stride/row_offset.
    //
    //   in_dim, out_dim, n_experts: result tensor ne = [in_dim, out_dim, n_experts]
    //   blocks_hf_name: U8 tensor [n_experts, out_dim_full, in_dim/32, 16]
    //   scales_hf_name: U8 tensor [n_experts, out_dim_full, in_dim/32]
    //   row_stride=1 row_offset=0: take every row (out_dim == out_dim_full)
    //   row_stride=2 row_offset=0: take even rows (gate from gate_up)
    //   row_stride=2 row_offset=1: take odd  rows (up from gate_up)
    ggml_tensor* declare_synth_dequanted_mxfp4_experts_(
        const std::string& dbg_name,
        std::int64_t in_dim, std::int64_t out_dim, std::int32_t n_experts,
        const std::string& blocks_hf_name,
        const std::string& scales_hf_name,
        int row_stride, int row_offset);
    // Synth-allocate a per-expert F32 bias by selecting strided rows from
    // a source [n_experts, out_dim_full] BF16 tensor. Used for gpt-oss's
    // de-interleaved gate / up biases. The dtype switch (F32 vs BF16)
    // hoists the per-batch ggml_cast that build_moe_ffn would otherwise
    // emit — ggml_add_id requires an F32 src1 on the CUDA backend.
    ggml_tensor* declare_synth_strided_bias_(
        const std::string& dbg_name,
        std::int64_t out_dim, std::int32_t n_experts,
        const std::string& bias_hf_name,
        int row_stride, int row_offset);

    // Synth-allocate an F32 tensor whose values are the BF16 source's
    // values converted at load time. Used for tensors that are consumed
    // by CUDA kernels asserting / preferring F32 (gpt-oss attn_sinks,
    // gemma4 layer_scalar). Avoids a per-batch in-graph ggml_cast.
    ggml_tensor* declare_synth_f32_from_bf16_(const std::string& hf_name);
    void build_qwen3_moe_();   // qwen3-moe family
    void build_mixtral_();
    void build_qwen3_5_();     // Qwen 3.5 / 3.6 hybrid (gated delta + GQA)

    // Qwen 3.5 conv1d.weight ships as [conv_dim, 1, conv_kernel] (HF
    // convention: [out_ch, in_ch, kernel]). ggml_ssm_conv wants
    // [conv_kernel, conv_dim]. Synth-load with a flat copy + reshape
    // since the underlying byte layout matches modulo the squeeze.
    ggml_tensor* declare_qwen3_5_conv1d_(const std::string& hf_name,
                                         std::int32_t conv_dim,
                                         std::int32_t conv_kernel);

    // Load tok_embd / output_norm / (optional) output_head and resize the
    // per-layer vector. Common to all archs.
    void load_top_level_();
    // Per-layer dense or MoE weight loader, driven by `spec`.
    void load_layer_(std::int32_t i, const LoaderSpec& spec);
    // Per-layer MoE branch (router + 3 stacked-expert tensors).
    void load_moe_layer_(std::int32_t i,
                         const std::string& path_prefix,
                         LoaderSpec::MoeNaming kind);
    // Qwen 3.6 MoE layer: fused 3D `gate_up_proj` + 3D `down_proj`
    // expert tensors plus a shared dense expert path. Different from the
    // qwen3_moe / mixtral / gpt-oss layouts handled by `load_moe_layer_`.
    void load_qwen3_5_moe_layer_(std::int32_t i,
                                 const std::string& path_prefix);
    // Helper: declare a stacked-expert F32/BF16 tensor backed by ONE
    // 3D HF safetensor source (shape [n_exp, out_dim, in_dim]), with an
    // optional intra-expert byte offset (for splitting fused gate_up).
    ggml_tensor* declare_stacked_experts_from_3d_(
        const std::string& dbg_name,
        const std::string& src_hf_name,
        std::int64_t in_dim, std::int64_t out_dim,
        std::int32_t n_experts,
        std::size_t  src_per_expert_bytes,
        std::size_t  src_intra_offset_bytes,
        ggml_type    dtype);
    // LLaMA-3.1 NTK-by-parts: synthesize the rope freq_factors tensor if
    // `hparams_.rope_scaling_type == "llama3"`. No-op otherwise.
    void synth_llama3_freq_factors_();

    // Resolve the per-arch HF tensor-name prefix once, right after
    // hparams_ are parsed. Multimodal wrappers nest the text decoder
    // under a sub-namespace; canonical "model.foo" tensor names get
    // the `model.` → `tensor_prefix_` substitution applied via tname_().
    // Text-only checkpoints keep tensor_prefix_ empty.
    void resolve_tensor_prefix_();

    // Apply tensor_prefix_ to a canonical "model.foo" name, producing
    // the actual HF tensor name to look up in the safetensors archive.
    // Empty prefix → returns `name` unchanged. Non-empty prefix → strips
    // the leading "model." (when present) and prepends `tensor_prefix_`.
    std::string tname_(const std::string& name) const;

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

    // Prefix prepended to every HF tensor name. Empty for plain text
    // checkpoints (qwen, llama, olmo, gemma2/3, etc.). Multimodal
    // checkpoints that wrap the text decoder (Ministral 3 ⇒
    // "language_model.", Gemma 4 ⇒ "model.language_model." stripping
    // — the per-arch builder is responsible for setting this before
    // calling the standard load_*_ helpers).
    std::string           tensor_prefix_;

    // Underlying weight archive — either SafetensorsArchive (HF
    // snapshot dir) or GGUFArchive (single .gguf file).
    std::unique_ptr<WeightArchive> archive_;
    ggml_backend_t        backend_     = nullptr;
    // CPU companion for the multi-backend scheduler. Created only when
    // `backend_` is a non-CPU backend (i.e. GPU). Null otherwise. The
    // scheduler in `ForwardEngine` uses it to absorb ops the GPU
    // backend can't dispatch (e.g. ggml-vulkan's missing CPY pipelines
    // for some bf16 source types). For Qwen3 family + Part A (norm
    // weights upcast to f32), no actual fallback fires in steady state
    // — the CPU backend is just a safety net so new arches don't
    // hard-abort the runtime.
    ggml_backend_t        cpu_fallback_ = nullptr;
    // Probed in the constructor once the primary backend exists. Drives
    // the slow_only sampling path in ForwardEngine when false.
    bool                  supports_in_graph_topk_ = true;
    ggml_threadpool_t     threadpool_  = nullptr;
    ggml_context*         ctx_         = nullptr;
    ggml_backend_buffer_t buf_         = nullptr;

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
        // When set, `load_into_backend_` materializes the source bf16
        // bytes into a temporary f32 buffer before uploading. Used for
        // small parameters (norm scales, projection biases) that we
        // promote to f32 at load time so backends that lack a bf16→f32
        // CPY kernel (notably ggml-vulkan, b6993) don't see an implicit
        // cast in the graph. Memory cost is trivial — norm + bias
        // weights together are <1 MB on a 1.7 B model. See
        // `is_small_weight_for_upcast()` in model.cpp for the predicate.
        bool         upcast_bf16_to_f32 = false;
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

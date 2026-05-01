#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <ggml-cpu.h>

namespace pie_portable_driver {

namespace {

ggml_type st_to_ggml_dtype(StDtype dt, const std::string& tensor_name) {
    switch (dt) {
        case StDtype::F32:  return GGML_TYPE_F32;
        case StDtype::F16:  return GGML_TYPE_F16;
        case StDtype::BF16: return GGML_TYPE_BF16;
        default:
            throw std::runtime_error(
                "model: unsupported safetensors dtype for '" + tensor_name +
                "': " + std::string(st_dtype_name(dt)) +
                " (v1 supports F32 / F16 / BF16; quantized HF formats land later)");
    }
}

std::string layer_path(std::int32_t i) {
    return "model.layers." + std::to_string(i) + ".";
}

// HF/PyTorch shape `[d_n, ..., d_0]` → ggml ne `[d_0, d_1, ..., d_n]`.
// Memory layout is row-major in both, so reversing the dimension list
// produces the correct ggml tensor without touching bytes.
ggml_tensor* new_tensor_from_st(ggml_context* ctx,
                                const StTensor& t,
                                const std::string& dbg_name) {
    if (t.shape.empty() || t.shape.size() > 4) {
        throw std::runtime_error(
            "model: tensor '" + dbg_name + "' has unsupported rank " +
            std::to_string(t.shape.size()));
    }
    const auto type = st_to_ggml_dtype(t.dtype, dbg_name);

    std::int64_t ne[4] = {1, 1, 1, 1};
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
        // reverse: hf[i] → ne[shape.size()-1-i]
        ne[t.shape.size() - 1 - i] = t.shape[i];
    }

    ggml_tensor* out = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
    if (!out) {
        throw std::runtime_error(
            "model: ggml_new_tensor failed for '" + dbg_name + "'");
    }
    ggml_set_name(out, dbg_name.c_str());
    return out;
}

}  // namespace

Model::Model(const std::filesystem::path& snapshot_dir, bool prefer_gpu)
    : snapshot_dir_(snapshot_dir) {
    if (!std::filesystem::is_directory(snapshot_dir_)) {
        throw std::runtime_error(
            "model: not a directory: " + snapshot_dir_.string());
    }

    hparams_ = parse_hf_config(snapshot_dir_ / "config.json");
    archive_ = std::make_unique<SafetensorsArchive>(snapshot_dir_);

    // Backend selection. With `GGML_CUDA=ON` (or Metal / Vulkan) at build
    // time, `ggml_backend_load_all()` registers them as device providers and
    // `ggml_backend_init_best()` picks GPU > CPU. Falls back to CPU if no
    // GPU backend was compiled in or no device is available.
    if (prefer_gpu) {
        backend_ = ggml_backend_init_best();
        if (!backend_) {
            std::cerr << "[model] ggml_backend_init_best() returned null; "
                         "falling back to CPU\n";
        }
    }
    if (!backend_) {
        backend_ = ggml_backend_cpu_init();
    }
    if (!backend_) {
        throw std::runtime_error("model: backend init failed");
    }

    // Generously size the metadata context. Each tensor consumes
    // `ggml_tensor_overhead()` bytes of metadata; bump if we ever exceed.
    constexpr std::size_t MAX_TENSORS = 4096;
    const std::size_t mem_size = ggml_tensor_overhead() * MAX_TENSORS;
    ggml_init_params ip{
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_ = ggml_init(ip);
    if (!ctx_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
        throw std::runtime_error("model: ggml_init failed");
    }

    try {
        switch (hparams_.arch) {
            case PieArch::Qwen3:
                build_qwen3_();
                break;
            case PieArch::Qwen2:
                build_qwen2_();
                break;
            case PieArch::Llama3:
                build_llama3_();
                break;
            case PieArch::Mistral3:
                build_mistral3_();
                break;
            case PieArch::Phi3:
                build_phi3_();
                break;
            case PieArch::Gemma2:
                build_gemma2_();
                break;
            case PieArch::Gemma3:
                build_gemma3_();
                break;
            case PieArch::Gemma4:
                build_gemma4_();
                break;
            case PieArch::GptOss:
                build_gpt_oss_();
                break;
            case PieArch::Qwen3_5:
                build_qwen3_5_();
                break;
            case PieArch::Mixtral:
                build_mixtral_();
                break;
            default:
                throw std::runtime_error(
                    "model: arch '" + std::string(pie_arch_name(hparams_.arch)) +
                    "' not implemented yet (M12 expansion)");
        }

        load_into_backend_();
    } catch (...) {
        if (buf_) {
            ggml_backend_buffer_free(buf_);
            buf_ = nullptr;
        }
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        if (backend_) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
        throw;
    }
}

Model::~Model() {
    if (buf_)     ggml_backend_buffer_free(buf_);
    if (ctx_)     ggml_free(ctx_);
    if (backend_) ggml_backend_free(backend_);
}

ggml_tensor* Model::declare_(const std::string& hf_name) {
    const auto& t = archive_->at(hf_name);
    auto* tensor = new_tensor_from_st(ctx_, t, hf_name);
    DeclaredTensor d;
    d.hf_name = hf_name;
    d.tensor = tensor;
    declared_.push_back(std::move(d));
    return tensor;
}

ggml_tensor* Model::declare_slice_(const std::string& fused_hf_name,
                                   const std::string& dbg_name,
                                   std::int64_t in_dim,
                                   std::int64_t out_dim,
                                   std::size_t  src_offset_bytes,
                                   ggml_type    dtype) {
    auto* tensor = ggml_new_tensor_2d(ctx_, dtype, in_dim, out_dim);
    ggml_set_name(tensor, dbg_name.c_str());

    DeclaredTensor d;
    d.hf_name = fused_hf_name;
    d.tensor = tensor;
    d.src_offset_bytes = src_offset_bytes;
    d.copy_bytes = ggml_nbytes(tensor);

    // Sanity: the source must be at least src_offset_bytes + copy_bytes.
    const auto& src = archive_->at(fused_hf_name);
    if (src_offset_bytes + d.copy_bytes > src.nbytes) {
        throw std::runtime_error(
            "model: slice " + dbg_name + " of " + fused_hf_name +
            " exceeds source size (" +
            std::to_string(src_offset_bytes + d.copy_bytes) + " > " +
            std::to_string(src.nbytes) + ")");
    }
    declared_.push_back(std::move(d));
    return tensor;
}

ggml_tensor* Model::declare_stacked_experts_(
        const std::string& dbg_name,
        std::int64_t in_dim, std::int64_t out_dim, std::int32_t n_experts,
        const std::vector<std::string>& per_expert_hf_names,
        ggml_type dtype) {
    if (static_cast<std::int32_t>(per_expert_hf_names.size()) != n_experts) {
        throw std::runtime_error(
            "model: stacked experts source count mismatch for " + dbg_name);
    }
    auto* tensor = ggml_new_tensor_3d(ctx_, dtype, in_dim, out_dim, n_experts);
    ggml_set_name(tensor, dbg_name.c_str());

    // Per-expert byte size — must match each source safetensor's nbytes.
    const std::size_t expert_bytes =
        ggml_nbytes(tensor) / static_cast<std::size_t>(n_experts);

    for (std::int32_t e = 0; e < n_experts; ++e) {
        const auto& name = per_expert_hf_names[e];
        const auto& src = archive_->at(name);
        if (src.nbytes != expert_bytes) {
            throw std::runtime_error(
                "model: stacked-expert source size mismatch for '" + name +
                "': expected " + std::to_string(expert_bytes) +
                ", got " + std::to_string(src.nbytes));
        }

        DeclaredTensor d;
        d.hf_name          = name;
        d.tensor           = tensor;
        d.src_offset_bytes = 0;
        d.copy_bytes       = expert_bytes;
        d.dst_offset_bytes = static_cast<std::size_t>(e) * expert_bytes;
        declared_.push_back(std::move(d));
    }
    return tensor;
}

void Model::load_into_backend_() {
    // One backend buffer for the whole model. After this call all declared
    // tensors have valid `data` pointers in the backend's address space.
    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
    if (!buf_) {
        throw std::runtime_error(
            "model: ggml_backend_alloc_ctx_tensors failed (out of memory?)");
    }

    for (const auto& d : declared_) {
        const auto& src = archive_->at(d.hf_name);
        if (d.copy_bytes > 0) {
            // Sliced load (phi3 fused QKV / gate_up) OR stacked-expert
            // load (one source written into a 3D dest at dst_offset).
            ggml_backend_tensor_set(d.tensor,
                                    src.data + d.src_offset_bytes,
                                    d.dst_offset_bytes,
                                    d.copy_bytes);
        } else {
            const std::size_t need = ggml_nbytes(d.tensor);
            if (need != src.nbytes) {
                throw std::runtime_error(
                    "model: byte-size mismatch for '" + d.hf_name + "': ggml=" +
                    std::to_string(need) + " safetensors=" + std::to_string(src.nbytes));
            }
            ggml_backend_tensor_set(d.tensor, src.data, 0, src.nbytes);
        }
    }
    for (const auto& s : synth_) {
        ggml_backend_tensor_set(s.tensor, s.data.data(), 0, s.data.size());
    }
}

std::size_t Model::buffer_size() const noexcept {
    return buf_ ? ggml_backend_buffer_get_size(buf_) : 0;
}

std::string Model::backend_name() const noexcept {
    if (!backend_) return "<null>";
    auto* dev = ggml_backend_get_device(backend_);
    if (!dev) return ggml_backend_name(backend_);
    const char* desc = ggml_backend_dev_description(dev);
    return desc ? std::string(desc) : ggml_backend_name(backend_);
}

std::string Model::activation_dtype_str() const {
    // HF's torch_dtype is the ground truth for the stored weight dtype.
    // (Pie's runtime treats this field as informational.)
    if (hparams_.torch_dtype == "bfloat16") return "bfloat16";
    if (hparams_.torch_dtype == "float16")  return "float16";
    if (hparams_.torch_dtype == "float32")  return "float32";
    return hparams_.torch_dtype;
}

// -----------------------------------------------------------------------------
// Per-arch builders
//
// All dense (non-MoE) archs share a common skeleton; the variations are
// captured by `LoaderSpec`. MoE archs add stacked-expert tensors driven by
// `LoaderSpec::moe`. Phi3 keeps its own per-layer loop because its fused
// QKV / gate_up tensors require offset-based slicing.
// -----------------------------------------------------------------------------

void Model::load_top_level_() {
    const auto& h = hparams_;
    weights_.tok_embd    = declare_("model.embed_tokens.weight");
    weights_.output_norm = declare_("model.norm.weight");
    if (!h.tie_word_embeddings) {
        weights_.output_head = declare_("lm_head.weight");
    }
    weights_.layers.resize(h.num_hidden_layers);
}

void Model::load_layer_(std::int32_t i, const LoaderSpec& spec) {
    const std::string p = layer_path(i);
    auto& L = weights_.layers[i];

    L.attn_norm = declare_(p + "input_layernorm.weight");
    L.ffn_norm  = declare_(p + spec.ffn_norm_name + ".weight");
    if (spec.has_post_attn_norm) {
        L.post_attn_norm = declare_(p + "post_attention_layernorm.weight");
    }
    if (spec.has_post_ffn_norm) {
        L.post_ffn_norm  = declare_(p + "post_feedforward_layernorm.weight");
    }

    L.q_proj = declare_(p + "self_attn.q_proj.weight");
    L.k_proj = declare_(p + "self_attn.k_proj.weight");
    L.v_proj = declare_(p + "self_attn.v_proj.weight");
    L.o_proj = declare_(p + "self_attn.o_proj.weight");

    if (spec.has_qkv_bias) {
        L.q_proj_b = declare_(p + "self_attn.q_proj.bias");
        L.k_proj_b = declare_(p + "self_attn.k_proj.bias");
        L.v_proj_b = declare_(p + "self_attn.v_proj.bias");
    }
    if (spec.has_qk_norm) {
        L.q_norm = declare_(p + "self_attn.q_norm.weight");
        L.k_norm = declare_(p + "self_attn.k_norm.weight");
    }

    if (spec.moe == LoaderSpec::MoeNaming::None) {
        L.gate_proj = declare_(p + "mlp.gate_proj.weight");
        L.up_proj   = declare_(p + "mlp.up_proj.weight");
        L.down_proj = declare_(p + "mlp.down_proj.weight");
    } else {
        load_moe_layer_(i, p, spec.moe);
    }
}

void Model::load_moe_layer_(std::int32_t i,
                            const std::string& p,
                            LoaderSpec::MoeNaming kind) {
    const auto& h = hparams_;
    auto& L = weights_.layers[i];

    const std::int32_t n_exp  = h.num_experts;
    const std::int64_t hidden = h.hidden_size;
    const std::int64_t ff     = h.moe_intermediate_size > 0
        ? h.moe_intermediate_size : h.intermediate_size;

    std::vector<std::string> gate_names, up_names, down_names;
    gate_names.reserve(n_exp);
    up_names.reserve(n_exp);
    down_names.reserve(n_exp);

    if (kind == LoaderSpec::MoeNaming::MlpExperts) {
        // Qwen-MoE / GPT-OSS layout.
        L.moe_router = declare_(p + "mlp.gate.weight");
        for (std::int32_t e = 0; e < n_exp; ++e) {
            const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
            gate_names.push_back(ep + "gate_proj.weight");
            up_names  .push_back(ep + "up_proj.weight");
            down_names.push_back(ep + "down_proj.weight");
        }
    } else {
        // Mixtral layout: w1=gate, w2=down, w3=up, router under
        // block_sparse_moe.gate.
        L.moe_router = declare_(p + "block_sparse_moe.gate.weight");
        for (std::int32_t e = 0; e < n_exp; ++e) {
            const std::string ep = p + "block_sparse_moe.experts." + std::to_string(e) + ".";
            gate_names.push_back(ep + "w1.weight");
            up_names  .push_back(ep + "w3.weight");
            down_names.push_back(ep + "w2.weight");
        }
    }

    const ggml_type w_dtype =
        st_to_ggml_dtype(archive_->at(gate_names[0]).dtype, gate_names[0]);
    L.moe_gate_exps = declare_stacked_experts_(
        p + "moe_gate", hidden, ff, n_exp, gate_names, w_dtype);
    L.moe_up_exps   = declare_stacked_experts_(
        p + "moe_up",   hidden, ff, n_exp, up_names,   w_dtype);
    L.moe_down_exps = declare_stacked_experts_(
        p + "moe_down", ff, hidden, n_exp, down_names, w_dtype);
}

void Model::build_qwen3_() {
    LoaderSpec s;
    s.has_qk_norm = true;
    load_top_level_();
    for (std::int32_t i = 0; i < hparams_.num_hidden_layers; ++i) {
        load_layer_(i, s);
    }
}

void Model::build_qwen2_() {
    LoaderSpec s;
    s.has_qkv_bias = true;
    load_top_level_();
    for (std::int32_t i = 0; i < hparams_.num_hidden_layers; ++i) {
        load_layer_(i, s);
    }
}

void Model::build_mistral3_() {
    // Mistral / Mistral3: same tensor layout as Llama3 (no biases, no
    // QK-norm, no rope_scaling.llama3). All sliding-window attention is
    // encoded in the graph mask, not the weight set.
    LoaderSpec s;
    load_top_level_();
    for (std::int32_t i = 0; i < hparams_.num_hidden_layers; ++i) {
        load_layer_(i, s);
    }
}

namespace {

// Compute LLaMA-3.1 NTK-by-parts RoPE frequency factors. Mirrors
// transformers' `_compute_llama3_parameters`. Returns a [head_dim/2] F32
// vector; entry i is the divisor applied to the base frequency at dim i.
//
// Default (no scaling) base inv_freq[i] = 1 / theta^(2i/head_dim).
// LLaMA-3.1 scaling depends on wavelength = 2π / inv_freq[i]:
//   - wavelen >= old_ctx / low_freq_factor   → inv_freq /= factor
//   - wavelen <= old_ctx / high_freq_factor  → unchanged
//   - else (medium-freq band) → smooth interpolation
//
// In ggml's `c` (freq_factors) tensor convention, c[i] divides the base
// frequency, so c[i] = inv_freq_default / inv_freq_llama3.
std::vector<float> compute_llama3_freq_factors(
        std::int32_t head_dim,
        float rope_theta,
        float factor,
        float low_freq_factor,
        float high_freq_factor,
        std::int32_t old_context_len) {
    const std::int32_t n = head_dim / 2;
    std::vector<float> c(n, 1.0f);
    if (factor <= 0.0f || old_context_len <= 0) return c;

    const float low_freq_wavelen  = static_cast<float>(old_context_len) / low_freq_factor;
    const float high_freq_wavelen = static_cast<float>(old_context_len) / high_freq_factor;
    const float two_pi = 6.28318530717958647692f;

    for (std::int32_t i = 0; i < n; ++i) {
        const float exponent = (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim);
        const float inv_freq = 1.0f / std::pow(rope_theta, exponent);
        const float wavelen = two_pi / inv_freq;
        if (wavelen < high_freq_wavelen) {
            c[i] = 1.0f;  // unchanged
        } else if (wavelen > low_freq_wavelen) {
            c[i] = factor;  // long wavelength → divide by factor
        } else {
            // Smoothed band.
            const float smooth =
                (static_cast<float>(old_context_len) / wavelen - low_freq_factor) /
                (high_freq_factor - low_freq_factor);
            // 1/c = smooth + (1 - smooth) / factor
            const float inv_c = smooth + (1.0f - smooth) / factor;
            c[i] = 1.0f / inv_c;
        }
    }
    return c;
}

}  // namespace

// LLaMA-3.1+: NTK-by-parts scaling synthesizes a freq_factors tensor.
// Llama 3.0 has no rope_scaling; this is a no-op there. Other archs
// don't call this.
void Model::synth_llama3_freq_factors_() {
    const auto& h = hparams_;
    if (!h.has_rope_scaling || h.rope_scaling_type != "llama3") return;

    const auto factors = compute_llama3_freq_factors(
        h.head_dim,
        h.rope_theta,
        h.rope_scaling_factor,
        h.rope_scaling_low_freq_factor,
        h.rope_scaling_high_freq_factor,
        h.rope_scaling_original_max_position);
    weights_.freq_factors =
        ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, h.head_dim / 2);
    ggml_set_name(weights_.freq_factors, "rope_freq_factors");
    SynthTensor s;
    s.tensor = weights_.freq_factors;
    s.data.resize(factors.size() * sizeof(float));
    std::memcpy(s.data.data(), factors.data(), s.data.size());
    synth_.push_back(std::move(s));
}

void Model::build_llama3_() {
    LoaderSpec s;
    load_top_level_();
    for (std::int32_t i = 0; i < hparams_.num_hidden_layers; ++i) {
        load_layer_(i, s);
    }
    synth_llama3_freq_factors_();
}

// Phi3: HF stores fused QKV as `qkv_proj.weight` of shape
//   [(n_q + 2*n_kv) * head_dim, hidden_size]
// and fused gate+up as `gate_up_proj.weight` of shape
//   [2 * intermediate_size, hidden_size]
// At load time we slice each into the per-projection tensors the graph
// builder expects. PyTorch row-major + ggml dim-reversal mean each
// projection occupies a contiguous byte range of the fused tensor.
void Model::build_phi3_() {
    const auto& h = hparams_;
    load_top_level_();

    const std::int64_t hidden = h.hidden_size;
    const std::int64_t n_q    = static_cast<std::int64_t>(h.num_attention_heads) * h.head_dim;
    const std::int64_t n_kv   = static_cast<std::int64_t>(h.num_key_value_heads) * h.head_dim;
    const std::int64_t n_ff   = h.intermediate_size;

    auto fused_dtype = [&](const std::string& name) {
        const auto& t = archive_->at(name);
        return std::pair{st_to_ggml_dtype(t.dtype, name), st_dtype_size(t.dtype)};
    };

    for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
        const std::string p = layer_path(i);
        auto& L = weights_.layers[i];

        L.attn_norm = declare_(p + "input_layernorm.weight");
        L.ffn_norm  = declare_(p + "post_attention_layernorm.weight");

        const std::string qkv_name = p + "self_attn.qkv_proj.weight";
        auto [qkv_type, qkv_dsz] = fused_dtype(qkv_name);
        // Q, K, V rows live back-to-back in the fused tensor. Each row is
        // `hidden` elements; bytes per row = hidden * dtype_size.
        const std::size_t row_bytes = static_cast<std::size_t>(hidden) * qkv_dsz;
        L.q_proj = declare_slice_(qkv_name, p + "q_proj", hidden, n_q,
                                  /*offset=*/ 0, qkv_type);
        L.k_proj = declare_slice_(qkv_name, p + "k_proj", hidden, n_kv,
                                  /*offset=*/ static_cast<std::size_t>(n_q) * row_bytes,
                                  qkv_type);
        L.v_proj = declare_slice_(qkv_name, p + "v_proj", hidden, n_kv,
                                  /*offset=*/ static_cast<std::size_t>(n_q + n_kv) * row_bytes,
                                  qkv_type);
        L.o_proj = declare_(p + "self_attn.o_proj.weight");

        const std::string gu_name = p + "mlp.gate_up_proj.weight";
        auto [gu_type, gu_dsz] = fused_dtype(gu_name);
        const std::size_t gu_row_bytes = static_cast<std::size_t>(hidden) * gu_dsz;
        L.gate_proj = declare_slice_(gu_name, p + "gate_proj", hidden, n_ff,
                                     /*offset=*/ 0, gu_type);
        L.up_proj   = declare_slice_(gu_name, p + "up_proj", hidden, n_ff,
                                     /*offset=*/ static_cast<std::size_t>(n_ff) * gu_row_bytes,
                                     gu_type);
        L.down_proj = declare_(p + "mlp.down_proj.weight");
    }
}

// Gemma2: extra norms (post_attention_layernorm + pre_feedforward_layernorm
// + post_feedforward_layernorm), softcaps, alternating SWA layers.
void Model::build_gemma2_() {
    LoaderSpec s;
    s.ffn_norm_name      = "pre_feedforward_layernorm";
    s.has_post_attn_norm = true;
    s.has_post_ffn_norm  = true;
    load_top_level_();
    for (std::int32_t i = 0; i < hparams_.num_hidden_layers; ++i) {
        load_layer_(i, s);
    }
}

// Gemma3: like Gemma2 but adds QK-norm.
void Model::build_gemma3_() {
    LoaderSpec s;
    s.ffn_norm_name      = "pre_feedforward_layernorm";
    s.has_post_attn_norm = true;
    s.has_post_ffn_norm  = true;
    s.has_qk_norm        = true;
    load_top_level_();
    for (std::int32_t i = 0; i < hparams_.num_hidden_layers; ++i) {
        load_layer_(i, s);
    }
}

// Gemma4: per-layer head_dim (sliding vs global), partial RoPE,
// layer_types, and Per-Layer Embeddings (PLE). v1 supports the layer
// sequencing + dual RoPE structurally; PLE is not yet wired.
void Model::build_gemma4_() {
    throw std::runtime_error(
        "model: gemma4 has Per-Layer Embeddings (PLE) and dual head_dim per "
        "layer that are structurally different from Gemma3. The skeleton is "
        "in place; full implementation is post-v1.");
}

// MoE family. Mixtral / Qwen2-3 MoE / GPT-OSS share the same overall
// structure (attention + 3 stacked-expert tensors + 1 router weight per
// layer). Tensor naming differs and is captured by `LoaderSpec::moe`:
//   - MlpExperts:    qwen-moe / gpt-oss layout
//   - BlockSparseMoe: mixtral layout (w1=gate, w2=down, w3=up)
// The graph builder picks `is_moe` and routes to `build_moe_ffn` instead
// of the dense SwiGLU path.
void Model::build_qwen3_5_() {
    const auto& h = hparams_;
    if (h.num_experts <= 0 || h.num_experts_per_tok <= 0) {
        throw std::runtime_error(
            "model qwen3_5: missing num_experts / num_experts_per_tok in config");
    }
    LoaderSpec s;
    s.moe = LoaderSpec::MoeNaming::MlpExperts;
    load_top_level_();
    for (std::int32_t i = 0; i < hparams_.num_hidden_layers; ++i) {
        load_layer_(i, s);
    }
}

void Model::build_mixtral_() {
    const auto& h = hparams_;
    if (h.num_experts <= 0 || h.num_experts_per_tok <= 0) {
        throw std::runtime_error(
            "model mixtral: missing num_experts / num_experts_per_tok");
    }
    LoaderSpec s;
    s.moe = LoaderSpec::MoeNaming::BlockSparseMoe;
    load_top_level_();
    for (std::int32_t i = 0; i < hparams_.num_hidden_layers; ++i) {
        load_layer_(i, s);
    }
}

void Model::build_gpt_oss_() {
    // GPT-OSS uses the same layout as Qwen3-MoE for routing + expert
    // tensor naming. The arch-specific differences (attention sinks,
    // YARN-extended RoPE) are handled at the graph layer via ArchSpec
    // and the existing flash_attn_ext sink support.
    build_qwen3_5_();
}

}  // namespace pie_portable_driver

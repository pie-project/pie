#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <ggml-cpu.h>
#include <omp.h>

#include "gguf_archive.hpp"
#include "gguf_hparams.hpp"

namespace pie_portable_driver {

namespace {

// True if the tensor is a small parameter that we want to coerce to
// F32 at load time. Norm scales and per-channel biases are tiny
// (~hidden_size floats per layer each) so the bf16 → f32 promotion
// is cheap, but it eliminates an implicit `ggml_cast` in the graph
// per use:
//
//   - `norm_scale` (graph_common.cpp) casts norm weights to the
//      activation dtype (always F32 after rms_norm).
//   - `add_with_cast` (graph_common.cpp) casts bias weights to the
//      activation dtype before the residual add — used for attention
//      biases (q/k/v_proj.bias) on archs that have them: Qwen2,
//      Phi3, GPT-OSS, etc.
//
// Why this matters: every `ggml_cast` is a `GGML_OP_CPY` of type
// (bf16 → f32). ggml-vulkan (as of llama.cpp b6993) doesn't have a
// CPY pipeline for that pair, so each cast becomes an off-primary
// graph segment under `ggml_backend_sched`. With Qwen2's bias terms
// the split count blew up to 145 per forward — the model still ran
// (sched fell back to CPU per cast) but threw away the GPU's batched
// command-buffer optimization. Promoting these tensors to f32 at load
// makes `w->type == x->type` so the cast is skipped entirely; the
// graph stays on a single backend (n_splits=1).
//
// CPU and CUDA paths handle the cast natively but also benefit from
// the simpler graph topology (one fewer node per use).
bool is_small_weight_for_upcast(const std::string& hf_name) {
    // Match suffixes:
    //   `*norm.weight` — any RMSNorm / LayerNorm gain (input_layernorm,
    //                    q_norm, k_norm, post_*_layernorm, model.norm,
    //                    pre_feedforward_layernorm, etc.)
    //   `*.bias`       — any attention / projection bias
    static constexpr std::string_view kNormSuffix = "norm.weight";
    static constexpr std::string_view kBiasSuffix = ".bias";
    auto ends_with = [&](std::string_view suf) {
        return hf_name.size() >= suf.size() &&
               hf_name.compare(hf_name.size() - suf.size(),
                               suf.size(), suf) == 0;
    };
    return ends_with(kNormSuffix) || ends_with(kBiasSuffix);
}

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

// MXFP4 dequant table (E2M1 values, NOT doubled — matches HF gpt-oss's
// FP4_VALUES from pie/src/pie_driver/model/gpt_oss_utils.py:20).
constexpr float kFP4Values[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

inline std::uint16_t bf16_from_f32(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}

// BF16 → F32 zero-extend. BF16 stores the top 16 bits of an IEEE-754
// binary32; placing them back at that offset reproduces the original
// rounded value exactly.
inline float bf16_to_f32(std::uint16_t bf16_bits) {
    const std::uint32_t bits = static_cast<std::uint32_t>(bf16_bits) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Dequantize MXFP4 (HF gpt-oss layout) into BF16. Each block is 32
// elements: 16 packed-nibble bytes + 1 E8M0 scale byte. HF stores nibbles
// in interleaved-pair convention: byte j → out[2j] = lut[lo nibble],
// out[2j+1] = lut[hi nibble]. E8M0 scale: e → 2^(e-127); e==0 → 0.0;
// e==255 (NaN) → 0.0 to avoid Inf.
//
// FUTURE: dequant-to-BF16 inflates gpt-oss-20b's expert weights ~4× in
// GPU memory (13 GB MXFP4 source → ~52 GB BF16 stacked-experts). Loading
// natively into ggml's GGML_TYPE_MXFP4 (which stores 17 bytes per 32
// elements as `[E8M0, qs[16]]`) would keep the original size on-device
// and let ggml's MXFP4 mul_mat kernels do the heavy lifting. The byte
// layout differs from HF's: ggml interleaves 1 scale byte before each
// 16-byte qs run, AND uses lo-nibbles[0..15], hi-nibbles[16..31] strided
// (vs HF's [lo,hi,lo,hi,...] interleaved). Implementation requires (1) a
// per-block byte-rewrite pass at load time (~free) and (2) verifying
// ggml's mul_mat_id supports MXFP4 src0 on the CUDA backend.
void dequant_mxfp4_to_bf16(const std::uint8_t* blocks_u8,
                           const std::uint8_t* scales_u8,
                           std::uint16_t* dst_bf16,
                           std::size_t n_blocks) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const int e = scales_u8[b];
        const float scale = (e == 0 || e == 0xFF)
            ? 0.0f
            : std::ldexp(1.0f, e - 127);
        const std::uint8_t* qs = blocks_u8 + b * 16;
        std::uint16_t* dst = dst_bf16 + b * 32;
        for (int j = 0; j < 16; ++j) {
            const std::uint8_t byte = qs[j];
            const float v_lo = kFP4Values[byte & 0x0F] * scale;
            const float v_hi = kFP4Values[byte >> 4]   * scale;
            dst[2*j + 0] = bf16_from_f32(v_lo);
            dst[2*j + 1] = bf16_from_f32(v_hi);
        }
    }
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
    // GGUF path: ggml_type_override carries the actual ggml type
    // (Q4_K, Q5_K, F16, ...). Safetensors path: derive from StDtype.
    const auto type = (t.ggml_type_override >= 0)
        ? static_cast<ggml_type>(t.ggml_type_override)
        : st_to_ggml_dtype(t.dtype, dbg_name);

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
    // The "snapshot" can be either:
    //   * an HF snapshot directory (config.json + *.safetensors)
    //   * a single .gguf file (llama.cpp-style, includes Q-formats)
    const bool is_gguf_file =
        std::filesystem::is_regular_file(snapshot_dir_) &&
        snapshot_dir_.extension() == ".gguf";
    const bool is_hf_dir = std::filesystem::is_directory(snapshot_dir_);
    if (!is_gguf_file && !is_hf_dir) {
        throw std::runtime_error(
            "model: expected an HF snapshot dir or a .gguf file, got: " +
            snapshot_dir_.string());
    }

    if (is_gguf_file) {
        auto gguf = std::make_unique<GGUFArchive>(snapshot_dir_);
        hparams_ = parse_gguf_hparams(gguf->meta());
        archive_ = std::move(gguf);
    } else {
        hparams_ = parse_hf_config(snapshot_dir_ / "config.json");
        archive_ = std::make_unique<SafetensorsArchive>(snapshot_dir_);
    }
    resolve_tensor_prefix_();

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

    // Pin the CPU backend to all available HW threads. ggml's default is
    // 4 (sometimes 1 depending on backend init path), which leaves a
    // 13900K idling. The static-lib build path in particular gets
    // initialized from a Rust-spawned thread that may not inherit the
    // process-level OpenMP affinity, so be explicit.
    auto pin_cpu_threads = [](ggml_backend_t b) {
        unsigned n = std::thread::hardware_concurrency();
        if (n == 0) n = 4;
        ggml_backend_cpu_set_n_threads(b, static_cast<int>(n));
        return n;
    };
    if (ggml_backend_is_cpu(backend_)) {
        const unsigned n = pin_cpu_threads(backend_);
        std::cerr << "[model] ggml CPU backend pinned to "
                  << n << " thread(s)\n";
    } else {
        // Primary is a GPU backend. Set up a CPU companion so the
        // ForwardEngine's `ggml_backend_sched` has somewhere to route
        // ops the GPU can't handle. Cost is minimal: the CPU backend
        // doesn't allocate anything until sched actually splits work
        // to it. With Part A's norm-weight upcast, the steady-state
        // graph for Qwen3 / Llama / Gemma stays entirely on GPU and
        // the CPU backend remains idle — but the safety net matters
        // for arches that emit casts / ops the GPU lacks.
        cpu_fallback_ = ggml_backend_cpu_init();
        if (cpu_fallback_) {
            const unsigned n = pin_cpu_threads(cpu_fallback_);
            std::cerr << "[model] CPU fallback backend pinned to "
                      << n << " thread(s) (sched safety net for "
                      << ggml_backend_name(backend_) << ")\n";
        } else {
            // Continue without fallback: graphs that would have used
            // it will hard-abort instead of falling back. Better than
            // refusing to start the model entirely.
            std::cerr << "[model] WARNING: failed to init CPU fallback "
                         "backend; sched will run primary-only\n";
        }

        // Probe: can the primary backend run argsort on a vocab-sized
        // input? ggml-vulkan caps at 1024 cols (max_argsort_cols), so
        // Qwen3 (152k vocab) etc. fail. When this returns false the
        // ForwardEngine forces the `slow_only` sampling path: download
        // raw logits, sample host-side, no in-graph `ggml_top_k`. This
        // keeps Vulkan steady-state decode at GPU speed instead of
        // round-tripping each token through CPU for the argsort.
        ggml_init_params probe_ip{
            /*.mem_size   =*/ ggml_tensor_overhead() * 16,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        if (auto* probe_ctx = ggml_init(probe_ip)) {
            ggml_tensor* probs =
                ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F32,
                                   hparams_.vocab_size, 1);
            ggml_tensor* sorted =
                ggml_argsort(probe_ctx, probs, GGML_SORT_ORDER_DESC);
            supports_in_graph_topk_ =
                ggml_backend_supports_op(backend_, sorted);
            std::cerr << "[model] in-graph top-k support on "
                      << ggml_backend_name(backend_) << ": "
                      << (supports_in_graph_topk_ ? "yes" : "no — using slow_only sampling path")
                      << "\n";

            // ggml_paged_attn_ext probe. Use the model's real attention
            // geometry: FlashInfer supports only specific head_dim/GQA
            // combinations, so a synthetic group-1 probe can incorrectly
            // enable paged attention for models like Qwen2-0.5B
            // (14 Q heads / 2 KV heads => group size 7).
            // Q must be BF16 to match the FlashInfer wrapper's expected dtype.
            const int probe_head_dim   = hparams_.head_dim;
            const int probe_n_q_heads  = hparams_.num_attention_heads;
            const int probe_n_kv_heads = hparams_.num_key_value_heads;
            const int probe_page_size  = 16;
            const int probe_n_req      = 1;
            ggml_tensor* probe_q = ggml_new_tensor_4d(
                probe_ctx, GGML_TYPE_BF16,
                probe_head_dim, /*n_q_tokens=*/ 1,
                /*n_q_heads=*/  probe_n_q_heads,
                /*n_req=*/      probe_n_req);
            ggml_tensor* probe_kv_pool = ggml_new_tensor_2d(
                probe_ctx, GGML_TYPE_F16,
                probe_head_dim * probe_n_kv_heads, probe_page_size);
            ggml_tensor* probe_page_indices = ggml_new_tensor_1d(
                probe_ctx, GGML_TYPE_I32, probe_n_req);
            ggml_tensor* probe_page_indptr = ggml_new_tensor_1d(
                probe_ctx, GGML_TYPE_I32, probe_n_req + 1);
            ggml_tensor* probe_last_page_lens = ggml_new_tensor_1d(
                probe_ctx, GGML_TYPE_I32, probe_n_req);
            ggml_tensor* probe_pa = ggml_paged_attn_ext(
                probe_ctx, probe_q, probe_kv_pool, probe_kv_pool,
                probe_page_indices, probe_page_indptr, probe_last_page_lens,
                probe_page_size, probe_head_dim, probe_n_kv_heads,
                /*sliding_window=*/ -1,
                /*scale=*/ 1.0f / 8.0f, /*softcap=*/ 0.0f);
            supports_paged_attn_ext_ =
                ggml_backend_supports_op(backend_, probe_pa);
            std::cerr << "[model] paged_attn_ext support on "
                      << ggml_backend_name(backend_) << ": "
                      << (supports_paged_attn_ext_
                          ? "yes"
                          : "no — using materialize+flash_attn_ext fallback")
                      << "\n";

            ggml_free(probe_ctx);
        }
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
            case PieArch::Phi3Small:
                build_phi3small_();
                break;
            case PieArch::Phi3_5Moe:
                build_phi3_5moe_();
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
            case PieArch::Gemma3n:
                build_gemma3n_();
                break;
            case PieArch::Olmo3:
                build_olmo3_();
                break;
            case PieArch::GptOss:
                build_gpt_oss_();
                break;
            case PieArch::Qwen3Moe:
                build_qwen3_moe_();
                break;
            case PieArch::Mixtral:
                build_mixtral_();
                break;
            case PieArch::Qwen3_5:
                build_qwen3_5_();
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
        if (cpu_fallback_) {
            ggml_backend_free(cpu_fallback_);
            cpu_fallback_ = nullptr;
        }
        if (backend_) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
        throw;
    }
}

Model::~Model() {
    if (buf_)          ggml_backend_buffer_free(buf_);
    if (ctx_)          ggml_free(ctx_);
    if (cpu_fallback_) ggml_backend_free(cpu_fallback_);
    if (backend_)      ggml_backend_free(backend_);
}

std::string Model::tname_(const std::string& name) const {
    if (tensor_prefix_.empty()) return name;
    constexpr std::string_view kModelPrefix = "model.";
    if (name.size() >= kModelPrefix.size() &&
        std::string_view(name).substr(0, kModelPrefix.size()) == kModelPrefix) {
        return tensor_prefix_ + std::string(name.substr(kModelPrefix.size()));
    }
    return tensor_prefix_ + name;
}

void Model::resolve_tensor_prefix_() {
    // Multimodal wrappers nest the text decoder one level under the
    // canonical `model.` namespace. tname_() strips the canonical
    // "model." prefix and substitutes `tensor_prefix_` in its place.
    switch (hparams_.arch) {
        case PieArch::Gemma4:
        case PieArch::Gemma3n:
        case PieArch::Qwen3_5:
            // Gemma 4 / 3n and Qwen 3.5 / 3.6 all ship as multimodal
            // ForConditionalGeneration wrappers; the text decoder lives
            // under `model.language_model.`.
            tensor_prefix_ = "model.language_model.";
            break;
        case PieArch::Gemma3:
            // Gemma 3 4B+ multimodal (Gemma3ForConditionalGeneration) puts
            // the text decoder under `language_model.model.`. The 270M/1B
            // text-only checkpoints keep canonical names; detect from the
            // archive.
            if (archive_->find("model.embed_tokens.weight") == nullptr &&
                archive_->find("language_model.model.embed_tokens.weight") != nullptr) {
                tensor_prefix_ = "language_model.model.";
            }
            break;
        case PieArch::Mistral3:
            // Plain Mistral / Mistral 7B v0.3 keeps canonical "model."
            // names. Multimodal Mistral3 wrappers (Ministral 3 8B/14B Base,
            // Mistral-Small-3.1, etc.) put the text decoder under
            // `language_model.model.`. Detect from the archive rather than
            // from text_config.model_type, since the inner type can be
            // either "mistral" or "ministral3".
            if (archive_->find("model.embed_tokens.weight") == nullptr &&
                archive_->find("language_model.model.embed_tokens.weight") != nullptr) {
                tensor_prefix_ = "language_model.model.";
            }
            break;
        default:
            break;
    }
}

ggml_tensor* Model::declare_(const std::string& hf_name) {
    const auto& t = archive_->at(hf_name);

    // Norm-style weights stored in bf16 get promoted to f32 here. The
    // tensor we hand back to the graph builder is typed F32, so the
    // implicit `ggml_cast` in `norm_scale` becomes a no-op. The actual
    // bf16 → f32 byte conversion happens in load_into_backend_().
    const bool upcast = t.dtype == StDtype::BF16
                     && is_small_weight_for_upcast(hf_name);
    // Some HF releases (notably Gemma-2-2b) store weights as F32. The F32
    // mul_mat path is ~2x slower than BF16 on tensor-core hardware and
    // doubles VRAM. Downcast large matmul weights at load time, matching
    // llama.cpp's GGUF default. Norms / biases stay F32 (precision).
    const bool downcast = t.dtype == StDtype::F32
                       && !is_small_weight_for_upcast(hf_name);

    ggml_tensor* tensor;
    if (upcast || downcast) {
        // Build the target tensor manually with the same shape as the
        // safetensor (HF dim order reversed for ggml). Type differs
        // from the source.
        std::int64_t ne[4] = {1, 1, 1, 1};
        for (std::size_t i = 0; i < t.shape.size(); ++i) {
            ne[t.shape.size() - 1 - i] = t.shape[i];
        }
        const ggml_type tgt = upcast ? GGML_TYPE_F32 : GGML_TYPE_BF16;
        tensor = ggml_new_tensor_4d(ctx_, tgt, ne[0], ne[1], ne[2], ne[3]);
        if (!tensor) {
            throw std::runtime_error(
                std::string("model: ggml_new_tensor_4d (")
                + (upcast ? "upcast" : "downcast")
                + ") failed for '" + hf_name + "'");
        }
        ggml_set_name(tensor, hf_name.c_str());
    } else {
        tensor = new_tensor_from_st(ctx_, t, hf_name);
    }

    DeclaredTensor d;
    d.hf_name              = hf_name;
    d.tensor               = tensor;
    d.upcast_bf16_to_f32   = upcast;
    d.downcast_f32_to_bf16 = downcast;
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
        if (d.upcast_bf16_to_f32) {
            // Promote bf16 source bytes to f32 in a temp buffer, then
            // upload. ggml provides `ggml_bf16_to_fp32_row` for this.
            // Norm weights are small enough that the temp allocation
            // is irrelevant (a few KB per call).
            const std::size_t n = static_cast<std::size_t>(ggml_nelements(d.tensor));
            if (src.nbytes != n * sizeof(ggml_bf16_t)) {
                throw std::runtime_error(
                    "model: bf16 upcast size mismatch for '" + d.hf_name +
                    "': expected=" + std::to_string(n * sizeof(ggml_bf16_t)) +
                    " safetensors=" + std::to_string(src.nbytes));
            }
            std::vector<float> tmp(n);
            ggml_bf16_to_fp32_row(
                reinterpret_cast<const ggml_bf16_t*>(src.data),
                tmp.data(), n);
            ggml_backend_tensor_set(d.tensor, tmp.data(), 0,
                                    n * sizeof(float));
        } else if (d.downcast_f32_to_bf16) {
            // Demote f32 source bytes to bf16 in a temp buffer, then
            // upload. Matmul weights only — norms & biases keep F32 via
            // is_small_weight_for_upcast().
            const std::size_t n = static_cast<std::size_t>(ggml_nelements(d.tensor));
            if (src.nbytes != n * sizeof(float)) {
                throw std::runtime_error(
                    "model: f32 downcast size mismatch for '" + d.hf_name +
                    "': expected=" + std::to_string(n * sizeof(float)) +
                    " safetensors=" + std::to_string(src.nbytes));
            }
            std::vector<ggml_bf16_t> tmp(n);
            ggml_fp32_to_bf16_row(
                reinterpret_cast<const float*>(src.data),
                tmp.data(), n);
            ggml_backend_tensor_set(d.tensor, tmp.data(), 0,
                                    n * sizeof(ggml_bf16_t));
        } else if (d.copy_bytes > 0) {
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
    auto& h = hparams_;
    weights_.tok_embd    = declare_(tname_("model.embed_tokens.weight"));
    weights_.output_norm = declare_(tname_("model.norm.weight"));
    // Probe the archive for an explicit lm_head. Two layouts:
    //   - "<wrapper>lm_head.weight" — multimodal-wrapped (Ministral 3
    //     8B/14B, Mistral-Small-3.1: "language_model.lm_head.weight").
    //   - "lm_head.weight" — plain causal LMs.
    // If neither exists, use tied embeddings. The HF config flag
    // `tie_word_embeddings` is unreliable (often null in newer Mistral
    // configs and the HF default differs by arch), so we drive this off
    // the actual safetensors index and override the hparam to match.
    constexpr std::string_view kModelSuffix = "model.";
    std::string wrapper = tensor_prefix_;
    if (wrapper.size() >= kModelSuffix.size() &&
        std::string_view(wrapper).substr(wrapper.size() - kModelSuffix.size()) == kModelSuffix) {
        wrapper = wrapper.substr(0, wrapper.size() - kModelSuffix.size());
    }
    const std::string lm_wrapped = wrapper + "lm_head.weight";
    const std::string lm_plain   = "lm_head.weight";
    const std::string* found = nullptr;
    if (archive_->find(lm_wrapped) != nullptr) {
        found = &lm_wrapped;
    } else if (lm_wrapped != lm_plain && archive_->find(lm_plain) != nullptr) {
        found = &lm_plain;
    }
    if (found) {
        weights_.output_head = declare_(*found);
        h.tie_word_embeddings = false;
    } else {
        h.tie_word_embeddings = true;
    }
    weights_.layers.resize(h.num_hidden_layers);
}

void Model::load_layer_(std::int32_t i, const LoaderSpec& spec) {
    const std::string p = tname_(layer_path(i));
    auto& L = weights_.layers[i];

    if (spec.has_input_layernorm) {
        L.attn_norm = declare_(p + "input_layernorm.weight");
    }
    if (!spec.ffn_norm_name.empty()) {
        L.ffn_norm  = declare_(p + spec.ffn_norm_name + ".weight");
    }
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

ggml_tensor* Model::declare_stacked_experts_from_3d_(
        const std::string& dbg_name,
        const std::string& src_hf_name,
        std::int64_t in_dim, std::int64_t out_dim,
        std::int32_t n_experts,
        std::size_t  src_per_expert_bytes,
        std::size_t  src_intra_offset_bytes,
        ggml_type    dtype) {
    auto* tensor = ggml_new_tensor_3d(ctx_, dtype, in_dim, out_dim, n_experts);
    ggml_set_name(tensor, dbg_name.c_str());

    const std::size_t expert_bytes =
        ggml_nbytes(tensor) / static_cast<std::size_t>(n_experts);

    // Sanity-check the source size: each expert occupies at least
    // intra_offset + expert_bytes within its src_per_expert_bytes slab.
    if (src_intra_offset_bytes + expert_bytes > src_per_expert_bytes) {
        throw std::runtime_error(
            "model: stacked-expert source slice out of range for '" +
            dbg_name + "'");
    }
    const auto& src = archive_->at(src_hf_name);
    const std::size_t expected =
        src_per_expert_bytes * static_cast<std::size_t>(n_experts);
    if (src.nbytes != expected) {
        throw std::runtime_error(
            "model: stacked-expert source nbytes mismatch for '" +
            src_hf_name + "': expected " + std::to_string(expected) +
            ", got " + std::to_string(src.nbytes));
    }

    for (std::int32_t e = 0; e < n_experts; ++e) {
        DeclaredTensor d;
        d.hf_name          = src_hf_name;
        d.tensor           = tensor;
        d.src_offset_bytes = static_cast<std::size_t>(e) * src_per_expert_bytes
                           + src_intra_offset_bytes;
        d.copy_bytes       = expert_bytes;
        d.dst_offset_bytes = static_cast<std::size_t>(e) * expert_bytes;
        declared_.push_back(std::move(d));
    }
    return tensor;
}

void Model::load_qwen3_5_moe_layer_(std::int32_t i, const std::string& p) {
    const auto& h = hparams_;
    auto& L = weights_.layers[i];

    const std::int32_t n_exp  = h.num_experts;
    const std::int64_t hidden = h.hidden_size;
    const std::int64_t ff     = h.moe_intermediate_size;

    // Router: HF naming `mlp.gate.weight`, shape [num_experts, hidden].
    L.moe_router = declare_(p + "mlp.gate.weight");

    // Fused experts. Source `mlp.experts.gate_up_proj` is one 3D safetensor
    // [n_exp, 2*ff, hidden]; we split into separate stacked gate / up
    // tensors at load time so the existing build_moe_ffn helper applies.
    const std::string gate_up_name = p + "mlp.experts.gate_up_proj";
    const std::string down_name    = p + "mlp.experts.down_proj";
    const ggml_type w_dtype =
        st_to_ggml_dtype(archive_->at(gate_up_name).dtype, gate_up_name);
    const std::size_t row_bytes = static_cast<std::size_t>(hidden) *
                                   ggml_type_size(w_dtype);
    const std::size_t per_expert_bytes =
        static_cast<std::size_t>(2 * ff) * row_bytes;
    const std::size_t up_intra_offset =
        static_cast<std::size_t>(ff) * row_bytes;

    L.moe_gate_exps = declare_stacked_experts_from_3d_(
        p + "moe_gate", gate_up_name, hidden, ff, n_exp,
        per_expert_bytes, /*intra_off=*/0, w_dtype);
    L.moe_up_exps   = declare_stacked_experts_from_3d_(
        p + "moe_up",   gate_up_name, hidden, ff, n_exp,
        per_expert_bytes, /*intra_off=*/up_intra_offset, w_dtype);
    // down_proj source is [n_exp, hidden, ff] → per-expert [hidden, ff]
    // with bytes ff * hidden * dtype_size.
    const std::size_t down_per_expert_bytes =
        static_cast<std::size_t>(hidden) *
        static_cast<std::size_t>(ff) * ggml_type_size(w_dtype);
    L.moe_down_exps = declare_stacked_experts_from_3d_(
        p + "moe_down", down_name, ff, hidden, n_exp,
        down_per_expert_bytes, 0, w_dtype);

    // Shared expert (always-active dense path).
    if (h.shared_expert_intermediate_size > 0) {
        L.moe_shared_gate_proj = declare_(p + "mlp.shared_expert.gate_proj.weight");
        L.moe_shared_up_proj   = declare_(p + "mlp.shared_expert.up_proj.weight");
        L.moe_shared_down_proj = declare_(p + "mlp.shared_expert.down_proj.weight");
        L.moe_shared_gate      = declare_(p + "mlp.shared_expert_gate.weight");
    }
}

// Shared dense / dense+MoE skeleton used by every arch that doesn't need a
// custom per-layer loop. phi3, gemma4, gpt_oss have their own builders.
void Model::build_dense_(LoaderSpec spec) {
    load_top_level_();
    for (std::int32_t i = 0; i < hparams_.num_hidden_layers; ++i) {
        load_layer_(i, spec);
    }
}

void Model::build_qwen3_() { build_dense_({.has_qk_norm = true}); }

void Model::build_qwen2_() { build_dense_({.has_qkv_bias = true}); }

void Model::build_mistral3_() {
    // Mistral / Mistral3: same tensor layout as Llama3 (no biases, no
    // QK-norm). All sliding-window attention is encoded in the graph
    // mask, not the weight set. YaRN scaling (Ministral 3) is handled
    // via ArchSpec at the graph layer; no weight-space synth needed.
    // Multimodal-wrapper prefix is set up in resolve_tensor_prefix_().
    build_dense_({});
}

// Synthesize a [head_dim/2] freq_factors tensor that suppresses (zeros)
// inv_freq for the dims beyond `rope_angles`. ggml_rope_ext computes
// inv_freq[k] = 1/(theta^(2k/n_dims) * c[k]); we set c[k] to a huge value
// for k >= rope_angles, which makes inv_freq[k] ~ 0 (no rotation) for
// those dims. Used by Gemma 4's "proportional" RoPE on full-attention
// layers, where Python rotates only the first rope_angles dim-pairs but
// pairs them at offset head_dim/2 (NOT n_rotated_dims/2 like ggml's
// default). Caller passes n_dims=head_dim along with this factor tensor.
ggml_tensor* Model::synth_proportional_rope_factors_(std::int32_t head_dim,
                                                     std::int32_t rope_angles) {
    const std::int32_t half = head_dim / 2;
    if (rope_angles >= half) return nullptr;
    auto* t = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, half);
    ggml_set_name(t, "rope_partial_factors");
    SynthTensor s;
    s.tensor = t;
    s.data.resize(static_cast<std::size_t>(half) * sizeof(float));
    auto* fp = reinterpret_cast<float*>(s.data.data());
    for (std::int32_t k = 0; k < half; ++k) {
        fp[k] = (k < rope_angles) ? 1.0f : 1e30f;
    }
    synth_.push_back(std::move(s));
    return t;
}

void Model::build_olmo3_() {
    // Olmo3 is post-norm-only:
    //   hidden = residual + post_attention_layernorm(attention(hidden))
    //   hidden = residual + post_feedforward_layernorm(mlp(hidden))
    // No `input_layernorm` and no pre-FFN norm. QK-norm is on by default.
    // YaRN RoPE + per-layer attention pattern (layer_types) plumb through
    // ArchSpec.
    build_dense_({
        .has_input_layernorm = false,
        .ffn_norm_name       = "",     // no pre-FFN norm
        .has_post_attn_norm  = true,
        .has_post_ffn_norm   = true,
        .has_qkv_bias        = false,
        .has_qk_norm         = true,
    });
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
    build_dense_({});
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
        const std::string p = tname_(layer_path(i));
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

// Phi-3-small: LayerNorm with bias, fused `query_key_value` ([Q|K|V]
// with biases on each), `self_attn.dense` as the output projection,
// packed `mlp.up_proj` ([gate || up] split for GeGLU). All norms +
// projections carry a learnable bias.
void Model::build_phi3small_() {
    const auto& h = hparams_;

    // Top-level: embed + final LayerNorm (with bias) + lm_head.
    weights_.tok_embd      = declare_(tname_("model.embed_tokens.weight"));
    weights_.output_norm   = declare_(tname_("model.final_layernorm.weight"));
    weights_.output_norm_b = declare_(tname_("model.final_layernorm.bias"));
    if (!h.tie_word_embeddings) {
        weights_.output_head = declare_("lm_head.weight");
    }
    weights_.layers.resize(h.num_hidden_layers);

    const std::int64_t hidden = h.hidden_size;
    const std::int64_t n_q  = static_cast<std::int64_t>(h.num_attention_heads) * h.head_dim;
    const std::int64_t n_kv = static_cast<std::int64_t>(h.num_key_value_heads) * h.head_dim;
    const std::int64_t n_ff = h.intermediate_size;

    auto fused_dtype = [&](const std::string& name) {
        const auto& t = archive_->at(name);
        return std::pair{st_to_ggml_dtype(t.dtype, name), st_dtype_size(t.dtype)};
    };

    for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
        const std::string p = tname_(layer_path(i));
        auto& L = weights_.layers[i];

        // Pre-attn / pre-FFN LayerNorm (weight + bias).
        L.attn_norm   = declare_(p + "input_layernorm.weight");
        L.attn_norm_b = declare_(p + "input_layernorm.bias");
        L.ffn_norm    = declare_(p + "post_attention_layernorm.weight");
        L.ffn_norm_b  = declare_(p + "post_attention_layernorm.bias");

        // Fused query_key_value [Q | K | V] split into separate slices.
        const std::string qkv_name = p + "self_attn.query_key_value.weight";
        auto [qkv_type, qkv_dsz] = fused_dtype(qkv_name);
        const std::size_t row_bytes =
            static_cast<std::size_t>(hidden) * qkv_dsz;
        L.q_proj = declare_slice_(qkv_name, p + "q_proj", hidden, n_q,
                                  /*offset=*/0, qkv_type);
        L.k_proj = declare_slice_(qkv_name, p + "k_proj", hidden, n_kv,
                                  static_cast<std::size_t>(n_q) * row_bytes, qkv_type);
        L.v_proj = declare_slice_(qkv_name, p + "v_proj", hidden, n_kv,
                                  static_cast<std::size_t>(n_q + n_kv) * row_bytes,
                                  qkv_type);
        // Per-projection biases. The fused query_key_value.bias has
        // shape [n_q + n_kv + n_kv]; slice into Q/K/V parts.
        const std::string qkv_b_name = p + "self_attn.query_key_value.bias";
        auto [qkvb_type, qkvb_dsz] = fused_dtype(qkv_b_name);
        L.q_proj_b = declare_slice_(qkv_b_name, p + "q_proj.bias",
                                     n_q, /*out_dim=*/1, /*offset=*/0, qkvb_type);
        L.k_proj_b = declare_slice_(qkv_b_name, p + "k_proj.bias",
                                     n_kv, /*out_dim=*/1,
                                     static_cast<std::size_t>(n_q) * qkvb_dsz, qkvb_type);
        L.v_proj_b = declare_slice_(qkv_b_name, p + "v_proj.bias",
                                     n_kv, /*out_dim=*/1,
                                     static_cast<std::size_t>(n_q + n_kv) * qkvb_dsz, qkvb_type);

        // self_attn.dense (= o_proj) with bias.
        L.o_proj   = declare_(p + "self_attn.dense.weight");
        L.o_proj_b = declare_(p + "self_attn.dense.bias");

        // Packed mlp.up_proj [gate || up] split. Output dim is 2 * n_ff.
        const std::string up_name = p + "mlp.up_proj.weight";
        auto [up_type, up_dsz] = fused_dtype(up_name);
        const std::size_t up_row_bytes =
            static_cast<std::size_t>(hidden) * up_dsz;
        L.gate_proj = declare_slice_(up_name, p + "gate_proj", hidden, n_ff,
                                      /*offset=*/0, up_type);
        L.up_proj   = declare_slice_(up_name, p + "up_proj", hidden, n_ff,
                                      static_cast<std::size_t>(n_ff) * up_row_bytes,
                                      up_type);
        // Packed up_proj.bias [2 * n_ff] split.
        const std::string up_b_name = p + "mlp.up_proj.bias";
        auto [upb_type, upb_dsz] = fused_dtype(up_b_name);
        L.gate_proj_b = declare_slice_(up_b_name, p + "gate_proj.bias",
                                        n_ff, 1, /*offset=*/0, upb_type);
        L.up_proj_b   = declare_slice_(up_b_name, p + "up_proj.bias",
                                        n_ff, 1,
                                        static_cast<std::size_t>(n_ff) * upb_dsz,
                                        upb_type);

        // mlp.down_proj weight + bias.
        L.down_proj   = declare_(p + "mlp.down_proj.weight");
        L.down_proj_b = declare_(p + "mlp.down_proj.bias");
    }
}

// Phi-3.5-MoE (phi3_5moe; HF model_type "phimoe"): Mixtral-style
// per-expert w1/w2/w3 layout + block_sparse_moe.gate router, plus
// LayerNorm-with-bias on input / post-attn / final norms, biases on
// Q/K/V/O, and bias on lm_head.
void Model::build_phi3_5moe_() {
    const auto& h = hparams_;
    if (h.num_experts <= 0 || h.num_experts_per_tok <= 0) {
        throw std::runtime_error(
            "model phi3_5moe: missing num_experts / num_experts_per_tok");
    }

    // Top-level: embed + final LayerNorm (with bias) + lm_head (with bias).
    weights_.tok_embd      = declare_(tname_("model.embed_tokens.weight"));
    weights_.output_norm   = declare_(tname_("model.norm.weight"));
    weights_.output_norm_b = declare_(tname_("model.norm.bias"));
    if (!h.tie_word_embeddings) {
        weights_.output_head = declare_("lm_head.weight");
        // Phi-3.5-MoE: lm_head also has a bias.
        if (archive_->find("lm_head.bias") != nullptr) {
            weights_.output_head_b = declare_("lm_head.bias");
        }
    }
    weights_.layers.resize(h.num_hidden_layers);

    const std::int64_t hidden = h.hidden_size;
    const std::int64_t n_q  = static_cast<std::int64_t>(h.num_attention_heads) * h.head_dim;
    const std::int64_t n_kv = static_cast<std::int64_t>(h.num_key_value_heads) * h.head_dim;
    const std::int32_t n_exp = h.num_experts;
    const std::int64_t ff   = h.moe_intermediate_size > 0
                              ? h.moe_intermediate_size
                              : h.intermediate_size;

    for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
        const std::string p = tname_(layer_path(i));
        auto& L = weights_.layers[i];

        // LayerNorm weights + biases.
        L.attn_norm   = declare_(p + "input_layernorm.weight");
        L.attn_norm_b = declare_(p + "input_layernorm.bias");
        L.ffn_norm    = declare_(p + "post_attention_layernorm.weight");
        L.ffn_norm_b  = declare_(p + "post_attention_layernorm.bias");

        // Q/K/V/O projections (split — not fused on phimoe) + biases.
        L.q_proj   = declare_(p + "self_attn.q_proj.weight");
        L.q_proj_b = declare_(p + "self_attn.q_proj.bias");
        L.k_proj   = declare_(p + "self_attn.k_proj.weight");
        L.k_proj_b = declare_(p + "self_attn.k_proj.bias");
        L.v_proj   = declare_(p + "self_attn.v_proj.weight");
        L.v_proj_b = declare_(p + "self_attn.v_proj.bias");
        L.o_proj   = declare_(p + "self_attn.o_proj.weight");
        L.o_proj_b = declare_(p + "self_attn.o_proj.bias");

        // MoE: block_sparse_moe.gate router + per-expert w1/w2/w3 (mixtral
        // naming). Stack into the standard moe_gate_exps / moe_up_exps /
        // moe_down_exps tensors via declare_stacked_experts_.
        L.moe_router = declare_(p + "block_sparse_moe.gate.weight");
        std::vector<std::string> gate_names, up_names, down_names;
        gate_names.reserve(n_exp); up_names.reserve(n_exp); down_names.reserve(n_exp);
        for (std::int32_t e = 0; e < n_exp; ++e) {
            const std::string ep =
                p + "block_sparse_moe.experts." + std::to_string(e) + ".";
            gate_names.push_back(ep + "w1.weight");
            up_names  .push_back(ep + "w3.weight");
            down_names.push_back(ep + "w2.weight");
        }
        const ggml_type w_dtype = st_to_ggml_dtype(
            archive_->at(gate_names[0]).dtype, gate_names[0]);
        L.moe_gate_exps = declare_stacked_experts_(
            p + "moe_gate", hidden, ff, n_exp, gate_names, w_dtype);
        L.moe_up_exps   = declare_stacked_experts_(
            p + "moe_up",   hidden, ff, n_exp, up_names,   w_dtype);
        L.moe_down_exps = declare_stacked_experts_(
            p + "moe_down", ff, hidden, n_exp, down_names, w_dtype);
    }
}

// Gemma2: extra norms (post_attention_layernorm + pre_feedforward_layernorm
// + post_feedforward_layernorm), softcaps, alternating SWA layers.
void Model::build_gemma2_() {
    build_dense_({
        .ffn_norm_name       = "pre_feedforward_layernorm",
        .has_post_attn_norm  = true,
        .has_post_ffn_norm   = true,
    });
}

// Gemma3: like Gemma2 but adds QK-norm.
void Model::build_gemma3_() {
    build_dense_({
        .ffn_norm_name       = "pre_feedforward_layernorm",
        .has_post_attn_norm  = true,
        .has_post_ffn_norm   = true,
        .has_qkv_bias        = false,
        .has_qk_norm         = true,
    });
}

// Gemma 3n: Gemma 3-style attention + GeGLU MLP, augmented with
//   - AltUp: 4 parallel "alt" hidden streams updated each layer via
//     learned prediction/correction coefs. The trained weights expect
//     this routing — applying it correctly is what produces fluent
//     output. See graph_gemma3n.cpp for the per-layer logic.
//   - PLE (Per-Layer Embeddings): a secondary 262 144 × 7 680 embedding
//     table whose per-layer slices add a token-conditioned signal to
//     the active stream after each transformer block.
//   - Laurel: low-rank residual through the MLP (linear_left → right →
//     post_laurel_norm), summed with the standard MLP output.
//
// Tensor names mirror google/gemma-3n-E2B-it / E4B-it under the
// `model.language_model.` prefix (set up in resolve_tensor_prefix_).
// Activation sparsity (`activation_sparsity_pattern`) is read from the
// config but not yet applied; the v1 graph uses standard GeGLU.
void Model::build_gemma3n_() {
    const auto& h = hparams_;

    weights_.tok_embd    = declare_(tname_("model.embed_tokens.weight"));
    weights_.output_norm = declare_(tname_("model.norm.weight"));
    if (!h.tie_word_embeddings) {
        weights_.output_head = declare_("lm_head.weight");
    }

    // PLE top-level: token-identity table + context projection + norm.
    // Always present on Gemma 3n (gemma4_ple_dim > 0 holds for both
    // E2B and E4B).
    if (h.gemma4_ple_dim > 0) {
        weights_.ple_token_embed = declare_(
            tname_("model.embed_tokens_per_layer.weight"));
        weights_.ple_model_proj  = declare_(
            tname_("model.per_layer_model_projection.weight"));
        weights_.ple_model_norm  = declare_(
            tname_("model.per_layer_projection_norm.weight"));
    }

    // AltUp: top-level alternative-input projections (3 each direction).
    // Used to initialize the 3 inactive streams from the embedding and
    // to combine the 4 streams into the final output before the LM head.
    weights_.altup_proj_0         = declare_(tname_("model.altup_projections.0.weight"));
    weights_.altup_proj_1         = declare_(tname_("model.altup_projections.1.weight"));
    weights_.altup_proj_2         = declare_(tname_("model.altup_projections.2.weight"));
    weights_.altup_unembed_proj_0 = declare_(tname_("model.altup_unembed_projections.0.weight"));
    weights_.altup_unembed_proj_1 = declare_(tname_("model.altup_unembed_projections.1.weight"));
    weights_.altup_unembed_proj_2 = declare_(tname_("model.altup_unembed_projections.2.weight"));

    weights_.layers.resize(h.num_hidden_layers);
    // v1: declare K/V on every layer even when the config says the last
    // `num_kv_shared_layers` should reuse upstream K/V. Pie's generic
    // dense graph builder (build_qwen3_graph) — which Gemma 3n routes
    // through for now — doesn't yet implement KV-sharing; calling it
    // with null L.k_proj segfaults inside ggml_mul_mat. Wasting a
    // little memory on the unused projections beats either forking
    // the graph or porting Gemma 4's KV-share logic. Follow-up: skip
    // K/V mul_mat on shared layers and feed the upstream cache slot
    // directly, mirroring graph_gemma4.cpp.

    for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
        const std::string p = tname_(layer_path(i));
        auto& L = weights_.layers[i];

        // Standard Gemma-3 four-norm layout.
        L.attn_norm      = declare_(p + "input_layernorm.weight");
        L.post_attn_norm = declare_(p + "post_attention_layernorm.weight");
        L.ffn_norm       = declare_(p + "pre_feedforward_layernorm.weight");
        L.post_ffn_norm  = declare_(p + "post_feedforward_layernorm.weight");

        // Self-attention. Q/K/V/O on every layer (KV-share NYI).
        L.q_proj = declare_(p + "self_attn.q_proj.weight");
        L.q_norm = declare_(p + "self_attn.q_norm.weight");
        L.o_proj = declare_(p + "self_attn.o_proj.weight");
        L.k_proj = declare_(p + "self_attn.k_proj.weight");
        L.v_proj = declare_(p + "self_attn.v_proj.weight");
        L.k_norm = declare_(p + "self_attn.k_norm.weight");

        // MLP — standard GeGLU.
        L.gate_proj = declare_(p + "mlp.gate_proj.weight");
        L.up_proj   = declare_(p + "mlp.up_proj.weight");
        L.down_proj = declare_(p + "mlp.down_proj.weight");

        // AltUp per-layer: prediction + correction coef tables, the
        // small 4×4 router, and the active-stream output scale.
        L.altup_predict_coefs = declare_(p + "altup.prediction_coefs.weight");
        L.altup_correct_coefs = declare_(p + "altup.correction_coefs.weight");
        L.altup_router        = declare_(p + "altup.modality_router.weight");
        L.altup_router_norm   = declare_(p + "altup.router_norm.weight");
        L.altup_correct_scale = declare_(p + "altup.correct_output_scale");

        // Laurel: low-rank residual that bypasses the MLP via two thin
        // projections + a post-norm.
        L.laurel_left  = declare_(p + "laurel.linear_left.weight");
        L.laurel_right = declare_(p + "laurel.linear_right.weight");
        L.laurel_norm  = declare_(p + "laurel.post_laurel_norm.weight");

        // Per-layer PLE: project the per-layer slice of the PLE table
        // back into the hidden stream, gated by an input-derived norm.
        if (h.gemma4_ple_dim > 0) {
            L.gemma3n_ple_gate = declare_(p + "per_layer_input_gate.weight");
            L.gemma3n_ple_proj = declare_(p + "per_layer_projection.weight");
            L.gemma3n_ple_norm = declare_(p + "post_per_layer_input_norm.weight");
        }
    }
}

// Gemma4: per-layer head_dim (sliding vs global), partial RoPE,
// layer_types, Per-Layer Embeddings (PLE), V-norm, per-layer scalar.
// Multimodal wrapper prefix `model.language_model.`. Tensor declarations
// here mirror the schema in pie/src/pie_driver/model/gemma4.py.
void Model::build_gemma4_() {
    const auto& h = hparams_;
    if (h.layer_types.empty() ||
        static_cast<std::int32_t>(h.layer_types.size()) != h.num_hidden_layers) {
        throw std::runtime_error(
            "model gemma4: layer_types missing or wrong length (need " +
            std::to_string(h.num_hidden_layers) + ")");
    }
    // tensor_prefix_ ("model.language_model.") is set up in
    // resolve_tensor_prefix_().

    // ----- proportional-RoPE freq_factors (full layers only) -----
    // partial_factor < 1 means proportional RoPE: rotate first
    // (head_dim_global * partial)/2 dim-pairs, leaving the rest untouched.
    if (h.gemma4_rope_partial_factor_full < 1.0f
        && h.gemma4_rope_partial_factor_full > 0.0f) {
        const std::int32_t hd = h.gemma4_head_dim_global > 0
            ? h.gemma4_head_dim_global : h.head_dim;
        std::int32_t rotary_pairs =
            static_cast<std::int32_t>(hd * h.gemma4_rope_partial_factor_full) / 2;
        weights_.gemma4_rope_full_factors =
            synth_proportional_rope_factors_(hd, rotary_pairs);
    }

    // ----- top level -----
    weights_.tok_embd    = declare_(tname_("model.embed_tokens.weight"));
    weights_.output_norm = declare_(tname_("model.norm.weight"));
    if (!h.tie_word_embeddings) {
        // lm_head sits at the file root, not under language_model.
        weights_.output_head = declare_("lm_head.weight");
    }
    if (h.gemma4_ple_dim > 0) {
        weights_.ple_token_embed = declare_(
            tname_("model.embed_tokens_per_layer.weight"));
        weights_.ple_model_proj  = declare_(
            tname_("model.per_layer_model_projection.weight"));
        weights_.ple_model_norm  = declare_(
            tname_("model.per_layer_projection_norm.weight"));
    }

    weights_.layers.resize(h.num_hidden_layers);
    const std::int32_t first_shared =
        h.gemma4_num_kv_shared_layers > 0
            ? h.num_hidden_layers - h.gemma4_num_kv_shared_layers
            : h.num_hidden_layers;

    for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
        const std::string p = tname_(layer_path(i));
        auto& L = weights_.layers[i];

        // Four RMSNorms (gemma family): pre/post for both attention and FFN.
        L.attn_norm      = declare_(p + "input_layernorm.weight");
        L.post_attn_norm = declare_(p + "post_attention_layernorm.weight");
        L.ffn_norm       = declare_(p + "pre_feedforward_layernorm.weight");
        L.post_ffn_norm  = declare_(p + "post_feedforward_layernorm.weight");

        // Per-layer scalar (BF16 [1] in checkpoint; converted to F32 at
        // load time so the per-batch elementwise mul against an F32
        // residual stream skips the in-graph cast).
        L.layer_scalar = declare_synth_f32_from_bf16_(p + "layer_scalar");

        // Q always exists. Per-layer head_dim is encoded in the tensor's
        // shape via num_attention_heads × (head_dim or global_head_dim).
        L.q_proj = declare_(p + "self_attn.q_proj.weight");
        L.q_norm = declare_(p + "self_attn.q_norm.weight");
        L.o_proj = declare_(p + "self_attn.o_proj.weight");

        // K/V projections + K-norm exist on all layers in the safetensors,
        // but the python reference only loads them on non-shared layers.
        // We follow the same convention: shared layers (last N) skip these
        // tensors entirely so the graph won't accidentally compute K/V from
        // unused weights and corrupt the reused upstream KV.
        if (i < first_shared) {
            L.k_proj = declare_(p + "self_attn.k_proj.weight");
            // Gemma 4 large variants (31B, 26B-A4B) omit v_proj on
            // full_attention layers (alternative attention: V is the
            // same projection result as K). Probe per-layer.
            if (archive_->find(p + "self_attn.v_proj.weight") != nullptr) {
                L.v_proj = declare_(p + "self_attn.v_proj.weight");
            }
            L.k_norm = declare_(p + "self_attn.k_norm.weight");
        }

        // MLP — gemma's GeGLU. intermediate_size doubles when
        // use_double_wide_mlp is set AND the layer is shared (not E2B).
        L.gate_proj = declare_(p + "mlp.gate_proj.weight");
        L.up_proj   = declare_(p + "mlp.up_proj.weight");
        L.down_proj = declare_(p + "mlp.down_proj.weight");

        // Sparse-MoE block (Gemma 4 26B-A4B only). Loaded alongside
        // the dense MLP — at inference, both run in parallel and their
        // post-normed outputs are summed before the layer's
        // post_feedforward_layernorm. Mirrors HF Gemma4TextDecoderLayer
        // and upstream driver/cuda/src/model/gemma4.cpp commit 59f45df4.
        if (h.gemma4_enable_moe) {
            const std::int32_t n_exp  = h.num_experts;
            const std::int64_t hidden = h.hidden_size;
            const std::int64_t ff     = h.gemma4_moe_intermediate_size;

            L.moe_router                   = declare_(p + "router.proj.weight");
            L.moe_router_scale             = declare_(p + "router.scale");
            L.moe_router_per_expert_scale  = declare_(p + "router.per_expert_scale");
            L.gemma4_moe_pre_ffn_norm_2    = declare_(p + "pre_feedforward_layernorm_2.weight");
            L.gemma4_moe_post_ffn_norm_1   = declare_(p + "post_feedforward_layernorm_1.weight");
            L.gemma4_moe_post_ffn_norm_2   = declare_(p + "post_feedforward_layernorm_2.weight");

            // Packed experts.gate_up_proj is one 3D safetensor
            // [n_exp, 2*ff, hidden]; split into separate stacked
            // gate/up tensors at load so build_moe_ffn helpers apply.
            // Mirrors load_qwen3_5_moe_layer_.
            const std::string gate_up_name = p + "experts.gate_up_proj";
            const std::string down_name    = p + "experts.down_proj";
            const ggml_type w_dtype =
                st_to_ggml_dtype(archive_->at(gate_up_name).dtype, gate_up_name);
            const std::size_t row_bytes  =
                static_cast<std::size_t>(hidden) * ggml_type_size(w_dtype);
            const std::size_t per_expert_bytes =
                static_cast<std::size_t>(2 * ff) * row_bytes;
            const std::size_t up_intra_offset =
                static_cast<std::size_t>(ff) * row_bytes;

            L.moe_gate_exps = declare_stacked_experts_from_3d_(
                p + "moe_gate", gate_up_name, hidden, ff, n_exp,
                per_expert_bytes, /*intra_off=*/0, w_dtype);
            L.moe_up_exps   = declare_stacked_experts_from_3d_(
                p + "moe_up",   gate_up_name, hidden, ff, n_exp,
                per_expert_bytes, /*intra_off=*/up_intra_offset, w_dtype);
            const std::size_t down_per_expert_bytes =
                static_cast<std::size_t>(hidden) *
                static_cast<std::size_t>(ff) * ggml_type_size(w_dtype);
            L.moe_down_exps = declare_stacked_experts_from_3d_(
                p + "moe_down", down_name, ff, hidden, n_exp,
                down_per_expert_bytes, 0, w_dtype);
        }

        // PLE per-layer.
        if (h.gemma4_ple_dim > 0) {
            L.ple_gate = declare_(p + "per_layer_input_gate.weight");
            L.ple_proj = declare_(p + "per_layer_projection.weight");
            L.ple_norm = declare_(p + "post_per_layer_input_norm.weight");
        }
    }
}

// MoE family. Mixtral / Qwen2-3 MoE / GPT-OSS share the same overall
// structure (attention + 3 stacked-expert tensors + 1 router weight per
// layer). Tensor naming differs and is captured by `LoaderSpec::moe`:
//   - MlpExperts:    qwen-moe / gpt-oss layout
//   - BlockSparseMoe: mixtral layout (w1=gate, w2=down, w3=up)
// The graph builder picks `is_moe` and routes to `build_moe_ffn` instead
// of the dense SwiGLU path.
void Model::build_qwen3_moe_() {
    const auto& h = hparams_;
    if (h.num_experts <= 0 || h.num_experts_per_tok <= 0) {
        throw std::runtime_error(
            "model qwen3_moe: missing num_experts / num_experts_per_tok in config");
    }
    // Qwen3-MoE keeps the qwen3 attention block (Q/K-norm per head) and
    // swaps the MLP for top-k routed experts. Without has_qk_norm the
    // graph builder leaves q_norm / k_norm null and per-head norms are
    // skipped at attention time → garbage logits.
    LoaderSpec s;
    s.has_qk_norm = true;
    s.moe = LoaderSpec::MoeNaming::MlpExperts;
    build_dense_(s);
}

void Model::build_mixtral_() {
    const auto& h = hparams_;
    if (h.num_experts <= 0 || h.num_experts_per_tok <= 0) {
        throw std::runtime_error(
            "model mixtral: missing num_experts / num_experts_per_tok");
    }
    LoaderSpec s;
    s.moe = LoaderSpec::MoeNaming::BlockSparseMoe;
    build_dense_(s);
}

ggml_tensor* Model::declare_synth_dequanted_mxfp4_experts_(
        const std::string& dbg_name,
        std::int64_t in_dim, std::int64_t out_dim, std::int32_t n_experts,
        const std::string& blocks_hf_name,
        const std::string& scales_hf_name,
        int row_stride, int row_offset) {
    const auto& blocks_st = archive_->at(blocks_hf_name);
    const auto& scales_st = archive_->at(scales_hf_name);
    if (blocks_st.dtype != StDtype::U8 || scales_st.dtype != StDtype::U8) {
        throw std::runtime_error(
            "model: MXFP4 source tensors must be U8 (" + blocks_hf_name + ")");
    }
    if (in_dim % 32 != 0) {
        throw std::runtime_error(
            "model: MXFP4 in_dim must be a multiple of 32 (" + dbg_name + ")");
    }
    const std::int64_t n_blocks_per_row = in_dim / 32;
    // HF layout: blocks [n_experts, out_dim_full, n_blocks_per_row, 16]
    //            scales [n_experts, out_dim_full, n_blocks_per_row]
    if (blocks_st.shape.size() != 4 ||
        blocks_st.shape[0] != n_experts ||
        blocks_st.shape[2] != n_blocks_per_row ||
        blocks_st.shape[3] != 16) {
        throw std::runtime_error(
            "model: MXFP4 blocks shape mismatch for " + blocks_hf_name);
    }
    if (scales_st.shape.size() != 3 ||
        scales_st.shape[0] != n_experts ||
        scales_st.shape[1] != blocks_st.shape[1] ||
        scales_st.shape[2] != n_blocks_per_row) {
        throw std::runtime_error(
            "model: MXFP4 scales shape mismatch for " + scales_hf_name);
    }
    const std::int64_t out_dim_full = blocks_st.shape[1];
    if (row_offset + (out_dim - 1) * row_stride >= out_dim_full) {
        throw std::runtime_error(
            "model: MXFP4 row strided slice OOB for " + dbg_name);
    }

    auto* tensor = ggml_new_tensor_3d(
        ctx_, GGML_TYPE_BF16, in_dim, out_dim, n_experts);
    ggml_set_name(tensor, dbg_name.c_str());

    SynthTensor s;
    s.tensor = tensor;
    s.data.resize(static_cast<std::size_t>(in_dim) * out_dim * n_experts *
                  sizeof(std::uint16_t));
    auto* dst_bf16 = reinterpret_cast<std::uint16_t*>(s.data.data());

    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(out_dim_full) * n_blocks_per_row;
    const std::size_t blocks_bytes_per_row =
        static_cast<std::size_t>(n_blocks_per_row) * 16;

    for (std::int32_t e = 0; e < n_experts; ++e) {
        const std::uint8_t* blocks_e =
            blocks_st.data + e * blocks_per_expert * 16;
        const std::uint8_t* scales_e =
            scales_st.data + e * blocks_per_expert;
        for (std::int64_t r = 0; r < out_dim; ++r) {
            const std::int64_t r_src = row_offset + r * row_stride;
            const std::uint8_t* blocks_row =
                blocks_e + r_src * blocks_bytes_per_row;
            const std::uint8_t* scales_row =
                scales_e + r_src * n_blocks_per_row;
            std::uint16_t* dst_row = dst_bf16
                + (static_cast<std::size_t>(e) * out_dim + r) * in_dim;
            dequant_mxfp4_to_bf16(blocks_row, scales_row, dst_row,
                                  static_cast<std::size_t>(n_blocks_per_row));
        }
    }
    synth_.push_back(std::move(s));
    return tensor;
}

ggml_tensor* Model::declare_synth_strided_bias_(
        const std::string& dbg_name,
        std::int64_t out_dim, std::int32_t n_experts,
        const std::string& bias_hf_name,
        int row_stride, int row_offset) {
    const auto& src = archive_->at(bias_hf_name);
    if (src.dtype != StDtype::BF16) {
        throw std::runtime_error(
            "model: bias source must be BF16 (" + bias_hf_name + ")");
    }
    if (src.shape.size() != 2 || src.shape[0] != n_experts) {
        throw std::runtime_error(
            "model: bias shape mismatch for " + bias_hf_name);
    }
    const std::int64_t out_dim_full = src.shape[1];
    if (row_offset + (out_dim - 1) * row_stride >= out_dim_full) {
        throw std::runtime_error(
            "model: bias strided slice OOB for " + dbg_name);
    }
    // Emit F32 — ggml_add_id (the consumer in build_moe_ffn) requires an
    // F32 src1 on the CUDA backend; doing the BF16→F32 conversion at load
    // time avoids a per-batch in-graph ggml_cast.
    auto* tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, out_dim, n_experts);
    ggml_set_name(tensor, dbg_name.c_str());

    SynthTensor s;
    s.tensor = tensor;
    s.data.resize(static_cast<std::size_t>(out_dim) * n_experts *
                  sizeof(float));
    auto* dst_f32 = reinterpret_cast<float*>(s.data.data());
    const auto* src_bf16 = reinterpret_cast<const std::uint16_t*>(src.data);

    for (std::int32_t e = 0; e < n_experts; ++e) {
        const std::uint16_t* src_row = src_bf16 + e * out_dim_full;
        float* dst_row = dst_f32 + e * out_dim;
        for (std::int64_t r = 0; r < out_dim; ++r) {
            dst_row[r] = bf16_to_f32(src_row[row_offset + r * row_stride]);
        }
    }
    synth_.push_back(std::move(s));
    return tensor;
}

ggml_tensor* Model::declare_synth_f32_from_bf16_(const std::string& hf_name) {
    const auto& src = archive_->at(hf_name);
    if (src.dtype != StDtype::BF16) {
        throw std::runtime_error(
            "model: declare_synth_f32_from_bf16 source must be BF16 ("
            + hf_name + ")");
    }
    // Allocate a tensor with the same shape as the source but F32 dtype.
    std::int64_t ne[4] = {1, 1, 1, 1};
    for (std::size_t i = 0; i < src.shape.size(); ++i) {
        ne[src.shape.size() - 1 - i] = src.shape[i];
    }
    auto* tensor = ggml_new_tensor_4d(
        ctx_, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    ggml_set_name(tensor, hf_name.c_str());

    const std::size_t n_elems = src.nbytes / sizeof(std::uint16_t);
    SynthTensor s;
    s.tensor = tensor;
    s.data.resize(n_elems * sizeof(float));
    auto* dst = reinterpret_cast<float*>(s.data.data());
    const auto* src_bf16 = reinterpret_cast<const std::uint16_t*>(src.data);
    for (std::size_t i = 0; i < n_elems; ++i) {
        dst[i] = bf16_to_f32(src_bf16[i]);
    }
    synth_.push_back(std::move(s));
    return tensor;
}

void Model::build_gpt_oss_() {
    const auto& h = hparams_;
    if (h.num_experts <= 0 || h.num_experts_per_tok <= 0) {
        throw std::runtime_error(
            "model gpt-oss: missing num_experts / num_experts_per_tok");
    }
    const std::int32_t n_experts = h.num_experts;
    const std::int64_t hidden    = h.hidden_size;
    const std::int64_t intermediate = h.intermediate_size;

    load_top_level_();

    for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
        const std::string p = tname_(layer_path(i));
        auto& L = weights_.layers[i];

        L.attn_norm = declare_(p + "input_layernorm.weight");
        L.ffn_norm  = declare_(p + "post_attention_layernorm.weight");

        // gpt-oss attention has Q/K/V/O biases AND attention sinks.
        L.q_proj = declare_(p + "self_attn.q_proj.weight");
        L.k_proj = declare_(p + "self_attn.k_proj.weight");
        L.v_proj = declare_(p + "self_attn.v_proj.weight");
        L.o_proj = declare_(p + "self_attn.o_proj.weight");
        L.q_proj_b = declare_(p + "self_attn.q_proj.bias");
        L.k_proj_b = declare_(p + "self_attn.k_proj.bias");
        L.v_proj_b = declare_(p + "self_attn.v_proj.bias");
        L.o_proj_b = declare_(p + "self_attn.o_proj.bias");
        // Sinks (BF16 [n_q_heads] in checkpoint) → F32 at load time.
        // ggml_flash_attn_ext_add_sinks asserts F32; converting up front
        // skips the per-batch in-graph cast.
        L.attn_sinks = declare_synth_f32_from_bf16_(p + "self_attn.sinks");

        // MoE — fused per-layer MXFP4 experts, dequanted to BF16 on host.
        L.moe_router   = declare_(p + "mlp.router.weight");
        L.moe_router_b = declare_(p + "mlp.router.bias");

        // gate_up_proj is the fused gate+up: even rows are gate, odd are
        // up (per HF gpt_oss_utils.py convention `[:, 0::2, :]` / `[:, 1::2, :]`).
        L.moe_gate_exps = declare_synth_dequanted_mxfp4_experts_(
            p + "moe_gate", hidden, intermediate, n_experts,
            p + "mlp.experts.gate_up_proj_blocks",
            p + "mlp.experts.gate_up_proj_scales",
            /*row_stride=*/2, /*row_offset=*/0);
        L.moe_up_exps = declare_synth_dequanted_mxfp4_experts_(
            p + "moe_up",   hidden, intermediate, n_experts,
            p + "mlp.experts.gate_up_proj_blocks",
            p + "mlp.experts.gate_up_proj_scales",
            /*row_stride=*/2, /*row_offset=*/1);
        L.moe_down_exps = declare_synth_dequanted_mxfp4_experts_(
            p + "moe_down", intermediate, hidden, n_experts,
            p + "mlp.experts.down_proj_blocks",
            p + "mlp.experts.down_proj_scales",
            /*row_stride=*/1, /*row_offset=*/0);

        // Per-expert biases.
        L.moe_gate_exps_b = declare_synth_strided_bias_(
            p + "moe_gate_b", intermediate, n_experts,
            p + "mlp.experts.gate_up_proj_bias",
            /*row_stride=*/2, /*row_offset=*/0);
        L.moe_up_exps_b = declare_synth_strided_bias_(
            p + "moe_up_b",   intermediate, n_experts,
            p + "mlp.experts.gate_up_proj_bias",
            /*row_stride=*/2, /*row_offset=*/1);
        L.moe_down_exps_b = declare_synth_strided_bias_(
            p + "moe_down_b", hidden, n_experts,
            p + "mlp.experts.down_proj_bias",
            /*row_stride=*/1, /*row_offset=*/0);
    }
}

// Qwen 3.5 / 3.6 conv1d weight: HF stores depthwise conv as
// `[conv_dim, 1, conv_kernel]` BF16; ggml_ssm_conv requires the weight
// to be 2D F32 (`[conv_kernel, conv_dim]`). Squeeze + convert at load
// time using a synth tensor.
ggml_tensor* Model::declare_qwen3_5_conv1d_(const std::string& hf_name,
                                            std::int32_t conv_dim,
                                            std::int32_t conv_kernel) {
    const auto& src = archive_->at(hf_name);
    if (src.dtype != StDtype::BF16) {
        throw std::runtime_error(
            "model qwen3_5: conv1d weight must be BF16 for " + hf_name);
    }
    auto* t = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, conv_kernel, conv_dim);
    ggml_set_name(t, hf_name.c_str());

    const std::size_t n_elems = src.nbytes / sizeof(std::uint16_t);
    if (static_cast<std::int32_t>(n_elems) != conv_dim * conv_kernel) {
        throw std::runtime_error(
            "model qwen3_5: conv1d weight size mismatch for " + hf_name);
    }
    SynthTensor s;
    s.tensor = t;
    s.data.resize(n_elems * sizeof(float));
    auto* dst_f32  = reinterpret_cast<float*>(s.data.data());
    const auto* src_bf16 = reinterpret_cast<const std::uint16_t*>(src.data);
    for (std::size_t i = 0; i < n_elems; ++i) {
        dst_f32[i] = bf16_to_f32(src_bf16[i]);
    }
    synth_.push_back(std::move(s));
    return t;
}

void Model::build_qwen3_5_() {
    const auto& h = hparams_;
    if (h.layer_types.empty() ||
        static_cast<std::int32_t>(h.layer_types.size()) != h.num_hidden_layers) {
        throw std::runtime_error(
            "model qwen3_5: layer_types missing or wrong length (need " +
            std::to_string(h.num_hidden_layers) + ")");
    }
    if (h.qwen35_linear_num_v_heads <= 0 || h.qwen35_linear_k_head_dim <= 0 ||
        h.qwen35_linear_v_head_dim <= 0 || h.qwen35_linear_conv_kernel <= 1) {
        throw std::runtime_error(
            "model qwen3_5: missing/invalid linear-attention dims");
    }

    // ── top level ── (multimodal-wrapped via tensor_prefix_)
    weights_.tok_embd    = declare_(tname_("model.embed_tokens.weight"));
    weights_.output_norm = declare_(tname_("model.norm.weight"));
    if (!h.tie_word_embeddings) {
        weights_.output_head = declare_("lm_head.weight");
    }

    weights_.layers.resize(h.num_hidden_layers);

    const std::int32_t n_k_heads = h.qwen35_linear_num_k_heads;
    const std::int32_t n_v_heads = h.qwen35_linear_num_v_heads;
    const std::int32_t hk        = h.qwen35_linear_k_head_dim;
    const std::int32_t hv        = h.qwen35_linear_v_head_dim;
    const std::int32_t key_dim   = n_k_heads * hk;
    const std::int32_t value_dim = n_v_heads * hv;
    const std::int32_t conv_dim  = 2 * key_dim + value_dim;

    const bool is_moe = h.num_experts > 0;
    for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
        const std::string p = tname_(layer_path(i));
        auto& L = weights_.layers[i];

        // RMSNorms (same names on both layer types).
        L.attn_norm = declare_(p + "input_layernorm.weight");
        L.ffn_norm  = declare_(p + "post_attention_layernorm.weight");

        // FFN: dense SwiGLU on Qwen 3.5; fused-stacked MoE + shared expert
        // on Qwen 3.6 (qwen3_5_moe). The MoE layout matches HF's
        // `Qwen3_5MoeExperts`:
        //   mlp.experts.gate_up_proj  shape [E, 2*ff, hidden] (fused)
        //   mlp.experts.down_proj     shape [E, hidden, ff]
        //   mlp.gate.weight           [num_experts, hidden]   (router)
        //   mlp.shared_expert.{gate,up,down}_proj.weight      (dense MLP)
        //   mlp.shared_expert_gate.weight                     [1, hidden]
        if (is_moe) {
            load_qwen3_5_moe_layer_(i, p);
        } else {
            L.gate_proj = declare_(p + "mlp.gate_proj.weight");
            L.up_proj   = declare_(p + "mlp.up_proj.weight");
            L.down_proj = declare_(p + "mlp.down_proj.weight");
        }

        const char kind = h.layer_types[i];
        if (kind == 'l') {
            // ── linear-attention layer ──
            L.lin_in_proj_qkv = declare_(p + "linear_attn.in_proj_qkv.weight");
            L.lin_in_proj_z   = declare_(p + "linear_attn.in_proj_z.weight");
            L.lin_in_proj_a   = declare_(p + "linear_attn.in_proj_a.weight");
            L.lin_in_proj_b   = declare_(p + "linear_attn.in_proj_b.weight");
            L.lin_A_log       = declare_(p + "linear_attn.A_log");
            L.lin_dt_bias     = declare_(p + "linear_attn.dt_bias");
            L.lin_norm        = declare_(p + "linear_attn.norm.weight");
            L.lin_out_proj    = declare_(p + "linear_attn.out_proj.weight");
            L.lin_conv1d      = declare_qwen3_5_conv1d_(
                p + "linear_attn.conv1d.weight", conv_dim,
                h.qwen35_linear_conv_kernel);
        } else if (kind == 'g') {
            // ── full-attention layer (GQA + q/k_norm + mrope + output gate) ──
            L.q_proj = declare_(p + "self_attn.q_proj.weight");
            L.k_proj = declare_(p + "self_attn.k_proj.weight");
            L.v_proj = declare_(p + "self_attn.v_proj.weight");
            L.o_proj = declare_(p + "self_attn.o_proj.weight");
            L.q_norm = declare_(p + "self_attn.q_norm.weight");
            L.k_norm = declare_(p + "self_attn.k_norm.weight");
        } else {
            throw std::runtime_error(
                "model qwen3_5: unexpected layer_type '" +
                std::string(1, kind) + "' at layer " + std::to_string(i));
        }
    }
}

}  // namespace pie_portable_driver

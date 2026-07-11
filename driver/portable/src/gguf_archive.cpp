#include "gguf_archive.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <gguf.h>
#include <ggml.h>

namespace pie_portable_driver {

namespace {

#ifdef _WIN32
std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("gguf: open failed for " + path.string());
    }
    const auto end = in.tellg();
    if (end < 0) {
        throw std::runtime_error("gguf: tellg failed for " + path.string());
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(end));
    in.seekg(0, std::ios::beg);
    if (!data.empty() &&
        !in.read(reinterpret_cast<char*>(data.data()),
                 static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("gguf: read failed for " + path.string());
    }
    return data;
}
#endif

// `ggml_get_rows` on the CUDA backend supports a fixed set of source
// dtypes. tok_embd.weight is consumed by `ggml_get_rows`, so when the
// GGUF stores it in an unsupported quant (Q4_K / Q5_K / Q6_K / IQ*) we
// must dequant at load time. Returns true for types get_rows handles
// natively (no dequant needed).
bool get_rows_supports(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

// GGUF tensor name → HF-canonical name. Currently covers the
// Llama-family layout used by qwen3 / qwen2 / llama3 / mistral3 / olmo3
// and Qwen3-MoE / Mixtral / gpt-oss MoE variants. Returns empty string
// when a tensor has no canonical mapping (the caller keeps the raw
// name; per-arch loaders that look up by HF name will simply not find
// it, which is what we want for unknown-but-harmless extras).
std::string gguf_to_hf_name(const std::string& g) {
    // Top-level tensors.
    if (g == "token_embd.weight")  return "model.embed_tokens.weight";
    if (g == "output_norm.weight") return "model.norm.weight";
    if (g == "output.weight")      return "lm_head.weight";

    // Per-block: `blk.{N}.<part>` → `model.layers.{N}.<...>`.
    const std::string blk = "blk.";
    if (g.compare(0, blk.size(), blk) != 0) return {};
    auto dot = g.find('.', blk.size());
    if (dot == std::string::npos) return {};
    const std::string n = g.substr(blk.size(), dot - blk.size());
    const std::string suffix = g.substr(dot + 1);
    const std::string layer = "model.layers." + n + ".";

    // Norms.
    if (suffix == "attn_norm.weight")        return layer + "input_layernorm.weight";
    if (suffix == "ffn_norm.weight")         return layer + "post_attention_layernorm.weight";
    if (suffix == "post_attention_norm.weight") return layer + "post_attention_layernorm.weight";
    if (suffix == "post_ffw_norm.weight")    return layer + "post_feedforward_layernorm.weight";
    // Attention projections.
    if (suffix == "attn_q.weight")           return layer + "self_attn.q_proj.weight";
    if (suffix == "attn_k.weight")           return layer + "self_attn.k_proj.weight";
    if (suffix == "attn_v.weight")           return layer + "self_attn.v_proj.weight";
    if (suffix == "attn_output.weight")      return layer + "self_attn.o_proj.weight";
    if (suffix == "attn_q.bias")             return layer + "self_attn.q_proj.bias";
    if (suffix == "attn_k.bias")             return layer + "self_attn.k_proj.bias";
    if (suffix == "attn_v.bias")             return layer + "self_attn.v_proj.bias";
    if (suffix == "attn_output.bias")        return layer + "self_attn.o_proj.bias";
    // Per-head q/k norms (Qwen 3, Olmo 3, Gemma 3).
    if (suffix == "attn_q_norm.weight")      return layer + "self_attn.q_norm.weight";
    if (suffix == "attn_k_norm.weight")      return layer + "self_attn.k_norm.weight";
    // Dense FFN.
    if (suffix == "ffn_gate.weight")         return layer + "mlp.gate_proj.weight";
    if (suffix == "ffn_up.weight")           return layer + "mlp.up_proj.weight";
    if (suffix == "ffn_down.weight")         return layer + "mlp.down_proj.weight";
    // MoE: stacked-expert tensors land directly on the same canonical
    // names our existing model.cpp uses for `moe_gate_exps` etc. We
    // include both the qwen3_moe / mixtral / gpt-oss style stacked
    // tensors and the legacy per-expert form (some converters still
    // emit `blk.N.ffn_gate.E.weight`).
    if (suffix == "ffn_gate_inp.weight")     return layer + "mlp.gate.weight";
    if (suffix == "ffn_gate_exps.weight")    return layer + "mlp.experts.gate_proj.weight";
    if (suffix == "ffn_up_exps.weight")      return layer + "mlp.experts.up_proj.weight";
    if (suffix == "ffn_down_exps.weight")    return layer + "mlp.experts.down_proj.weight";
    // Qwen 3.5 / 3.6 shared expert (qwen35moe). `ffn_gate_inp_shexp` is the
    // 1-D sigmoid gate over the shared expert (HF `mlp.shared_expert_gate`),
    // distinct from the per-token router `ffn_gate_inp`.
    if (suffix == "ffn_gate_shexp.weight")     return layer + "mlp.shared_expert.gate_proj.weight";
    if (suffix == "ffn_up_shexp.weight")       return layer + "mlp.shared_expert.up_proj.weight";
    if (suffix == "ffn_down_shexp.weight")     return layer + "mlp.shared_expert.down_proj.weight";
    if (suffix == "ffn_gate_inp_shexp.weight") return layer + "mlp.shared_expert_gate.weight";
    // Qwen 3.5 / 3.6 hybrid gated-delta-rule linear attention. The GGUF
    // (llama.cpp qwen35 convention) packs the linear-attn input projection
    // as `attn_qkv` (the fused q/k/v of the delta net) + `attn_gate` (the
    // output Z gate), and the gated-delta params as `ssm_*`. These names
    // only ever appear on the *linear* layers; full-attention layers carry
    // the ordinary `attn_q/k/v` handled above. The convert-time value
    // transforms (A_log = -exp(raw); folded norm.weight +1) are inverted at
    // load in `model.cpp::build_qwen3_5_`, not here.
    if (suffix == "attn_qkv.weight")         return layer + "linear_attn.in_proj_qkv.weight";
    if (suffix == "attn_gate.weight")        return layer + "linear_attn.in_proj_z.weight";
    if (suffix == "ssm_alpha.weight")        return layer + "linear_attn.in_proj_a.weight";
    if (suffix == "ssm_beta.weight")         return layer + "linear_attn.in_proj_b.weight";
    if (suffix == "ssm_a")                   return layer + "linear_attn.A_log";
    if (suffix == "ssm_dt.bias")             return layer + "linear_attn.dt_bias";
    if (suffix == "ssm_conv1d.weight")       return layer + "linear_attn.conv1d.weight";
    if (suffix == "ssm_norm.weight")         return layer + "linear_attn.norm.weight";
    if (suffix == "ssm_out.weight")          return layer + "linear_attn.out_proj.weight";
    return {};
}

}  // namespace

void GGUFArchive::close_mmap_() noexcept {
#ifdef _WIN32
    owned_data_.clear();
    owned_data_.shrink_to_fit();
#else
    if (mmap_base_ != nullptr && mmap_size_ > 0) {
        ::munmap(const_cast<std::uint8_t*>(mmap_base_), mmap_size_);
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
    mmap_base_ = nullptr;
    mmap_size_ = 0;
}

GGUFArchive::GGUFArchive(const std::filesystem::path& gguf_file) {
    if (!std::filesystem::is_regular_file(gguf_file)) {
        throw std::runtime_error(
            "gguf: not a regular file: " + gguf_file.string());
    }

    // gguf_init_from_file reads metadata via fread. With no_alloc=true
    // it skips loading tensor data — we mmap separately so the data
    // pointers can be kept stable and inexpensive.
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx      = nullptr;
    gctx_ = gguf_init_from_file(gguf_file.string().c_str(), p);
    if (!gctx_) {
        throw std::runtime_error(
            "gguf: gguf_init_from_file failed for " + gguf_file.string());
    }

    // Now mmap the whole file so we can hand stable data pointers to
    // the loader. Tensor data starts at gguf_get_data_offset(gctx).
#ifdef _WIN32
    owned_data_ = read_file(gguf_file);
    mmap_size_ = owned_data_.size();
    mmap_base_ = owned_data_.data();
#else
    fd_ = ::open(gguf_file.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        gguf_free(gctx_); gctx_ = nullptr;
        throw std::runtime_error("gguf: open failed for " + gguf_file.string());
    }
    struct stat sb{};
    if (::fstat(fd_, &sb) != 0) {
        ::close(fd_); fd_ = -1; gguf_free(gctx_); gctx_ = nullptr;
        throw std::runtime_error("gguf: fstat failed");
    }
    mmap_size_ = static_cast<std::size_t>(sb.st_size);
    void* base = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (base == MAP_FAILED) {
        ::close(fd_); fd_ = -1; gguf_free(gctx_); gctx_ = nullptr;
        throw std::runtime_error("gguf: mmap failed");
    }
    mmap_base_ = static_cast<const std::uint8_t*>(base);
#endif

    const std::size_t data_offset = gguf_get_data_offset(gctx_);

    // ---- Parse KV metadata into GgufMeta -------------------------------
    const std::int64_t n_kv = gguf_get_n_kv(gctx_);
    for (std::int64_t i = 0; i < n_kv; ++i) {
        const std::string key = gguf_get_key(gctx_, i);
        const auto t = gguf_get_kv_type(gctx_, i);
        GgufMeta::KV kv;
        kv.key = key;
        kv.type = static_cast<std::int32_t>(t);
        switch (t) {
            case GGUF_TYPE_UINT8:   kv.num_value = gguf_get_val_u8 (gctx_, i); break;
            case GGUF_TYPE_INT8:    kv.num_value = gguf_get_val_i8 (gctx_, i); break;
            case GGUF_TYPE_UINT16:  kv.num_value = gguf_get_val_u16(gctx_, i); break;
            case GGUF_TYPE_INT16:   kv.num_value = gguf_get_val_i16(gctx_, i); break;
            case GGUF_TYPE_UINT32:  kv.num_value = gguf_get_val_u32(gctx_, i); break;
            case GGUF_TYPE_INT32:   kv.num_value = gguf_get_val_i32(gctx_, i); break;
            case GGUF_TYPE_UINT64:  kv.num_value = static_cast<double>(
                                          gguf_get_val_u64(gctx_, i)); break;
            case GGUF_TYPE_INT64:   kv.num_value = static_cast<double>(
                                          gguf_get_val_i64(gctx_, i)); break;
            case GGUF_TYPE_FLOAT32: kv.num_value = gguf_get_val_f32(gctx_, i); break;
            case GGUF_TYPE_FLOAT64: kv.num_value = gguf_get_val_f64(gctx_, i); break;
            case GGUF_TYPE_BOOL:    kv.bool_value = gguf_get_val_bool(gctx_, i); break;
            case GGUF_TYPE_STRING:  kv.str_value = gguf_get_val_str(gctx_, i); break;
            case GGUF_TYPE_ARRAY:   /* arrays handled on demand (tokens, etc.) */ break;
            default: break;
        }
        meta_.kv.emplace(key, std::move(kv));
    }
    if (auto it = meta_.kv.find("general.architecture"); it != meta_.kv.end()) {
        meta_.general_architecture = it->second.str_value;
    }
    if (auto it = meta_.kv.find("general.name"); it != meta_.kv.end()) {
        meta_.general_name = it->second.str_value;
    }

    // Capture the tokens-array length for vocab_size fallback. Most GGUFs
    // omit `<arch>.vocab_size` but always store `tokenizer.ggml.tokens`.
    {
        const std::int64_t tok_id =
            gguf_find_key(gctx_, "tokenizer.ggml.tokens");
        if (tok_id >= 0 &&
            gguf_get_kv_type(gctx_, tok_id) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(gctx_, tok_id) == GGUF_TYPE_STRING) {
            meta_.tokens_count = gguf_get_arr_n(gctx_, tok_id);
        }
    }

    // ---- Build the StTensor table keyed by HF canonical name -----------
    const std::int64_t n_tensors = gguf_get_n_tensors(gctx_);
    raw_names_.reserve(static_cast<std::size_t>(n_tensors));

    // We need tensor shapes (ne[]) and the public gguf API doesn't
    // expose them directly (only type, offset, size). Workaround: pass
    // ctx=&meta_ggml_ctx to gguf_init_from_file with no_alloc=true and
    // walk the resulting ggml_tensors. But that path also reads (and
    // would refuse with no_alloc=true...) — so instead we re-init JUST
    // for shape extraction with no_alloc=true + ctx, which DOES populate
    // a metadata-only ggml_context with named tensors.
    ggml_context* shape_ctx = nullptr;
    gguf_init_params p2{};
    p2.no_alloc = true;
    p2.ctx      = &shape_ctx;
    gguf_context* shape_gctx = gguf_init_from_file(gguf_file.string().c_str(), p2);
    if (!shape_gctx || !shape_ctx) {
        if (shape_gctx) gguf_free(shape_gctx);
        close_mmap_();
        gguf_free(gctx_); gctx_ = nullptr;
        throw std::runtime_error(
            "gguf: shape pass init failed for " + gguf_file.string());
    }

    for (std::int64_t i = 0; i < n_tensors; ++i) {
        const char* raw_name = gguf_get_tensor_name(gctx_, i);
        const std::string raw{raw_name};
        raw_names_.push_back(raw);

        // Pull shape from the shape-pass ggml context.
        ggml_tensor* meta_t = ggml_get_tensor(shape_ctx, raw_name);
        if (!meta_t) continue;

        StTensor st{};
        st.dtype = StDtype::F32;  // placeholder; ignored when override set.
        // GGUF stores ne[0] innermost. Reverse to HF order so the loader's
        // existing `ne[size-1-i] = shape[i]` reversal lands correctly.
        for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) {
            if (meta_t->ne[d] > 1 || st.shape.size() > 0) {
                st.shape.push_back(static_cast<std::int64_t>(meta_t->ne[d]));
            }
        }
        if (st.shape.empty()) {
            st.shape.push_back(1);  // scalar
        }
        st.ggml_type_override = static_cast<std::int32_t>(meta_t->type);
        st.nbytes = gguf_get_tensor_size(gctx_, i);
        const std::size_t local_offset = gguf_get_tensor_offset(gctx_, i);
        st.data = mmap_base_ + data_offset + local_offset;

        if (raw == "output.weight") meta_.has_output_weight = true;

        // tok_embd is read by ggml_get_rows. If the GGUF stored it in
        // an unsupported quant (Q4_K / Q5_K / Q6_K / IQ*), dequant to
        // F16 at load time. F16 keeps memory low; ggml_get_rows
        // supports it natively.
        if (raw == "token_embd.weight" &&
            !get_rows_supports(static_cast<ggml_type>(meta_t->type))) {
            const auto src_type = static_cast<ggml_type>(meta_t->type);
            const std::int64_t nrows = ggml_nrows(meta_t);
            const std::int64_t ncols = meta_t->ne[0];
            const std::size_t  f16_bytes =
                static_cast<std::size_t>(nrows) * ncols * sizeof(uint16_t);

            dequant_buffers_.emplace_back();
            auto& buf = dequant_buffers_.back();
            buf.resize(f16_bytes);
            std::vector<float> row_f32(static_cast<std::size_t>(ncols));

            const auto* src_bytes = reinterpret_cast<const char*>(st.data);
            const std::size_t src_row_bytes = ggml_row_size(src_type, ncols);
            const auto* traits = ggml_get_type_traits(src_type);
            if (!traits || !traits->to_float) {
                throw std::runtime_error(
                    std::string("gguf: no dequant for token_embd type ") +
                    ggml_type_name(src_type));
            }
            auto* dst_f16 = reinterpret_cast<uint16_t*>(buf.data());
            for (std::int64_t r = 0; r < nrows; ++r) {
                traits->to_float(src_bytes + r * src_row_bytes,
                                 row_f32.data(), ncols);
                ggml_fp32_to_fp16_row(row_f32.data(),
                                      dst_f16 + r * ncols, ncols);
            }
            st.data    = buf.data();
            st.nbytes  = f16_bytes;
            st.ggml_type_override = static_cast<std::int32_t>(GGML_TYPE_F16);
        }

        std::string canonical = gguf_to_hf_name(raw);
        if (canonical.empty()) {
            // Unmapped tensor — keep under the raw name so callers that
            // know about it can still find it (e.g. arch-specific
            // diagnostic dumps).
            canonical = raw;
        }
        tensors_.emplace(std::move(canonical), st);
    }

    gguf_free(shape_gctx);
    ggml_free(shape_ctx);
}

GGUFArchive::~GGUFArchive() {
    if (gctx_) { gguf_free(gctx_); gctx_ = nullptr; }
    close_mmap_();
}

const StTensor* GGUFArchive::find(const std::string& name) const noexcept {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const StTensor& GGUFArchive::at(const std::string& name) const {
    const auto* t = find(name);
    if (!t) throw std::runtime_error("gguf: missing tensor '" + name + "'");
    return *t;
}

// ---------------------------------------------------------------------------
// GGUFArchive::open_sharded — multi-file / split GGUF loader
// ---------------------------------------------------------------------------
// Discovers sibling shards from the first shard path using the canonical
// *-NNNNN-of-MMMMM.gguf naming scheme produced by llama.cpp's gguf-split.
// Reads split.count from the first shard's KV metadata to know how many
// siblings to expect, resolves their paths, constructs a GGUFArchive for
// each, and merges all tensors into the first archive.  Falls back to a
// plain single-file open when split.count is absent or equals 1.

namespace {

// Given a shard filename like "model-00001-of-00003.gguf", return the
// stem prefix ("model-"), the total shard count (3), and the zero-based
// index of this shard (0).  Returns false when the filename doesn't match
// the pattern.
bool parse_shard_filename(const std::filesystem::path& p,
                          std::string& prefix_out,
                          int& index_out,
                          int& total_out) {
    const std::string stem = p.stem().string();  // e.g. "model-00001-of-00003"
    // Pattern: <prefix>-<NNNNN>-of-<MMMMM>
    const std::string of_marker = "-of-";
    const auto of_pos = stem.rfind(of_marker);
    if (of_pos == std::string::npos) return false;

    const std::string total_str = stem.substr(of_pos + of_marker.size());
    // Find the index field: last '-' before of_pos
    const auto dash_pos = stem.rfind('-', of_pos - 1);
    if (dash_pos == std::string::npos) return false;

    const std::string index_str = stem.substr(dash_pos + 1, of_pos - dash_pos - 1);
    prefix_out = stem.substr(0, dash_pos + 1);  // includes trailing '-'

    try {
        index_out = std::stoi(index_str) - 1;  // 1-based → 0-based
        total_out = std::stoi(total_str);
    } catch (...) {
        return false;
    }
    return total_out > 0 && index_out >= 0 && index_out < total_out;
}

// Build the Nth shard path (1-based n) given the directory, prefix, total,
// and extension.  Pads to 5 digits matching llama.cpp convention.
std::filesystem::path shard_path(const std::filesystem::path& dir,
                                 const std::string& prefix,
                                 int n, int total,
                                 const std::string& ext) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%05d-of-%05d", n, total);
    return dir / (prefix + buf + ext);
}

}  // namespace

std::unique_ptr<GGUFArchive>
GGUFArchive::open_sharded(const std::filesystem::path& first_shard) {
    // Always open the given file first.
    auto primary = std::make_unique<GGUFArchive>(first_shard);

    // Check for split.count in KV metadata.
    int n_shards = 1;
    if (auto it = primary->meta_.kv.find("split.count"); it != primary->meta_.kv.end()) {
        n_shards = static_cast<int>(it->second.num_value);
    }

    if (n_shards <= 1) {
        // Single file or metadata says only one shard — nothing to merge.
        return primary;
    }

    // Parse the filename to discover sibling paths.
    std::string prefix;
    int this_index = 0, total_from_name = 0;
    const bool parsed = parse_shard_filename(first_shard, prefix,
                                             this_index, total_from_name);
    if (parsed && this_index != 0) {
        throw std::runtime_error(
            "gguf: open_sharded() requires the first shard (got shard " +
            std::to_string(this_index + 1) + " of " +
            std::to_string(total_from_name) + ")");
    }

    if (!parsed) {
        // Filename doesn't match the pattern but split.count > 1 — warn and
        // return what we have; callers will get a missing-tensor error later
        // which is more informative than a hard crash here.
        std::cerr << "[gguf] warning: split.count=" << n_shards
                  << " but filename '" << first_shard.filename().string()
                  << "' doesn't match *-NNNNN-of-MMMMM.gguf — "
                     "loading first shard only\n";
        return primary;
    }

    if (total_from_name != n_shards) {
        std::cerr << "[gguf] warning: split.count=" << n_shards
                  << " but filename implies " << total_from_name
                  << " shards — trusting filename\n";
        n_shards = total_from_name;
    }

    const auto dir = first_shard.parent_path();
    const std::string ext = first_shard.extension().string();  // ".gguf"

    // Load and merge shards 2..N into the primary archive.
    for (int i = 2; i <= n_shards; ++i) {
        const auto path = shard_path(dir, prefix, i, n_shards, ext);
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error(
                "gguf: shard " + std::to_string(i) + "/" +
                std::to_string(n_shards) + " not found: " + path.string());
        }

        auto shard = std::make_unique<GGUFArchive>(path);

        // Merge tensors — move each StTensor entry into the primary map.
        // The shard's mmap must remain alive as long as we use its data
        // pointers, so we adopt ownership of its mmap region.
        for (auto& [name, st] : shard->tensors_) {
            if (primary->tensors_.count(name)) {
                throw std::runtime_error(
                    "gguf: duplicate tensor '" + name +
                    "' found in shard " + std::to_string(i));
            }
            primary->tensors_.emplace(name, st);
        }
        // Transfer raw names for diagnostics.
        for (auto& rn : shard->raw_names_) {
            primary->raw_names_.push_back(std::move(rn));
        }
        // Transfer dequant buffers ownership so their data pointers stay valid.
        for (auto& buf : shard->dequant_buffers_) {
            primary->dequant_buffers_.push_back(std::move(buf));
        }
        // Adopt mmap ownership: move shard's owned_data / mmap pointers into
        // extra_mmaps_ so they stay alive for the lifetime of the primary.
        primary->extra_mmaps_.push_back(std::move(shard));
    }

    return primary;
}

}  // namespace pie_portable_driver

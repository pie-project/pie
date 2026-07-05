#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>


#include "loader/rust_quant_attachment.hpp"
#include "loader/rust_loader_input.hpp"
#include "loader/rust_storage_program.hpp"
#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

inline std::string rust_loader_bytes_to_string(
    pie_weight_loader::PieLoaderBytes bytes)
{
    if (bytes.ptr == nullptr || bytes.len == 0) return {};
    return std::string(
        reinterpret_cast<const char*>(bytes.ptr),
        reinterpret_cast<const char*>(bytes.ptr) + bytes.len);
}

struct RustLoaderCompileResult {
    RustStorageProgram program;
    std::vector<std::string> source_tensor_names;
    std::vector<RustQuantAttachment> quant_attachments;
    std::size_t source_tensor_count = 0;
    std::size_t covered_contract_count = 0;
    std::size_t runtime_tensor_count = 0;
    // Authoritative content key for the (deterministic) materialized weights:
    // checkpoint files + config + quant scheme + TP layout + ABI version. Reused
    // by the materialized-artifact cache so a warm reload can skip compile +
    // materialize entirely. Same value that keys the compile-plan disk cache.
    std::string cache_key;
};

struct RustLoaderSourceIndex {
    std::unordered_map<std::string, std::uint32_t> tensor_ids;
    std::vector<std::string> tensor_names;
};

class RustLoaderFnv1a128 {
public:
    void update_bytes(const void* data, std::size_t len) noexcept
    {
        const auto* p = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < len; ++i) {
            h1_ ^= p[i];
            h1_ *= 1099511628211ull;
            h2_ ^= static_cast<std::uint64_t>(p[i]) + 0x9e3779b97f4a7c15ull +
                   (h2_ << 6) + (h2_ >> 2);
            h2_ *= 1099511628211ull;
        }
    }

    template <typename T>
    void update_scalar(T value) noexcept
    {
        update_bytes(&value, sizeof(value));
    }

    void update_loader_bytes(pie_weight_loader::PieLoaderBytes bytes) noexcept
    {
        update_scalar<std::uint64_t>(bytes.len);
        if (bytes.ptr != nullptr && bytes.len != 0) {
            update_bytes(bytes.ptr, bytes.len);
        }
    }

    std::string hex() const
    {
        std::ostringstream out;
        out << std::hex;
        out.width(16);
        out.fill('0');
        out << h1_;
        out.width(16);
        out.fill('0');
        out << h2_;
        return out.str();
    }

private:
    std::uint64_t h1_ = 1469598103934665603ull;
    std::uint64_t h2_ = 1099511628211ull;
};

inline void rust_loader_hash_i64_slice(
    RustLoaderFnv1a128& h,
    pie_weight_loader::PieLoaderI64Slice slice)
{
    h.update_scalar<std::uint64_t>(slice.len);
    for (std::size_t i = 0; i < slice.len; ++i) {
        h.update_scalar<std::int64_t>(slice.ptr[i]);
    }
}

inline void rust_loader_hash_u32_slice(
    RustLoaderFnv1a128& h,
    pie_weight_loader::PieLoaderU32Slice slice)
{
    h.update_scalar<std::uint64_t>(slice.len);
    for (std::size_t i = 0; i < slice.len; ++i) {
        h.update_scalar<std::uint32_t>(slice.ptr[i]);
    }
}

inline void rust_loader_hash_byte_spans(
    RustLoaderFnv1a128& h,
    pie_weight_loader::PieLoaderRuntimeByteSpanSlice slice)
{
    h.update_scalar<std::uint64_t>(slice.len);
    for (std::size_t i = 0; i < slice.len; ++i) {
        const auto& s = slice.ptr[i];
        h.update_scalar(s.source_tensor_id);
        h.update_scalar(s.source_offset_bytes);
        h.update_scalar(s.dest_offset_bytes);
        h.update_scalar(s.span_bytes);
    }
}

inline std::string rust_loader_compile_cache_key(
    const pie_weight_loader::PieLoaderCompileInput& input)
{
    RustLoaderFnv1a128 h;
    // Manual override knob — bump only to force a flush by hand. Compiler-logic
    // changes are now tracked automatically by the FFI hash below (build.rs
    // content-hashes the loader source), so this no longer needs bumping per
    // code change.
    constexpr const char* cache_version =
        "pie-cuda-rust-storage-program-cache-v10";
    h.update_bytes(cache_version, std::char_traits<char>::length(cache_version));
    // Auto-invalidate when the Rust compiler logic changes (build.rs content-
    // hashes the loader source into this weak FFI symbol). Weak-guarded so the
    // header still compiles where the staticlib isn't linked (cxx_compat probe).
    h.update_scalar(::pie_loader_compiler_version != nullptr
                        ? ::pie_loader_compiler_version()
                        : std::uint64_t{0});
    h.update_scalar(input.version);
    h.update_loader_bytes(input.model.model_type);
    h.update_loader_bytes(input.model.quant_method);
    h.update_loader_bytes(input.model.runtime_quant);
    h.update_scalar(input.model.num_hidden_layers);
    h.update_scalar(input.model.num_experts);
    h.update_scalar(input.model.num_experts_per_tok);
    h.update_scalar(static_cast<std::uint32_t>(input.target.backend));
    h.update_scalar(input.target.tp_rank);
    h.update_scalar(input.target.tp_size);
    h.update_scalar(input.target.max_tile_bytes);
    h.update_scalar(input.target.preferred_alignment);
    h.update_scalar(static_cast<std::uint32_t>(input.target.mxfp4_moe));
    h.update_scalar(static_cast<std::uint8_t>(input.target.native_mxfp4_moe));
    h.update_loader_bytes(input.runtime_abi.name);
    h.update_scalar(input.runtime_abi.version);

    h.update_scalar<std::uint64_t>(input.files.len);
    for (std::size_t i = 0; i < input.files.len; ++i) {
        const auto& f = input.files.ptr[i];
        h.update_scalar(f.id);
        h.update_loader_bytes(f.path);
        h.update_scalar(f.size_bytes);
        h.update_scalar(static_cast<std::uint32_t>(f.format));
    }

    h.update_scalar<std::uint64_t>(input.tensors.len);
    for (std::size_t i = 0; i < input.tensors.len; ++i) {
        const auto& t = input.tensors.ptr[i];
        h.update_scalar(t.id);
        h.update_loader_bytes(t.name);
        h.update_scalar(t.file_id);
        h.update_scalar(t.file_offset);
        h.update_scalar(t.span_bytes);
        h.update_scalar(static_cast<std::uint32_t>(t.dtype));
        h.update_scalar(static_cast<std::uint32_t>(t.encoding_kind));
        h.update_scalar(static_cast<std::uint32_t>(t.quant_scheme));
        rust_loader_hash_i64_slice(h, t.shape);
        h.update_scalar(t.quant_bits_per_element);
        h.update_scalar(t.quant_group_size);
        h.update_scalar(t.quant_channel_axis);
        h.update_scalar(static_cast<std::uint8_t>(t.quant_has_scale_dtype));
        h.update_scalar(static_cast<std::uint32_t>(t.quant_scale_dtype));
        h.update_scalar(static_cast<std::uint8_t>(t.quant_has_zero_point_dtype));
        h.update_scalar(static_cast<std::uint32_t>(t.quant_zero_point_dtype));
        rust_loader_hash_i64_slice(h, t.quant_block_shape);
    }

    h.update_scalar<std::uint64_t>(input.runtime_abi.tensors.len);
    for (std::size_t i = 0; i < input.runtime_abi.tensors.len; ++i) {
        const auto& t = input.runtime_abi.tensors.ptr[i];
        h.update_loader_bytes(t.output_name);
        h.update_scalar(static_cast<std::uint32_t>(t.source_kind));
        h.update_scalar(t.source_tensor_id);
        rust_loader_hash_u32_slice(h, t.source_tensor_ids);
        rust_loader_hash_byte_spans(h, t.byte_spans);
        rust_loader_hash_u32_slice(h, t.metadata_tensor_ids);
        h.update_scalar(t.source_contract_id);
        h.update_scalar(static_cast<std::uint32_t>(t.semantic_role));
        h.update_scalar(t.layer);
        h.update_scalar(static_cast<std::uint8_t>(t.has_layer));
        h.update_scalar(t.expert);
        h.update_scalar(static_cast<std::uint8_t>(t.has_expert));
        h.update_scalar(t.axis);
        h.update_scalar(t.start);
        h.update_scalar(t.length);
        h.update_scalar(static_cast<std::uint32_t>(t.dtype));
        h.update_scalar(static_cast<std::uint32_t>(t.encoding_kind));
        h.update_scalar(static_cast<std::uint32_t>(t.quant_scheme));
        rust_loader_hash_i64_slice(h, t.shape);
        h.update_scalar(t.alignment);
        h.update_scalar(t.shard_axis);
        h.update_scalar(t.quant_bits_per_element);
        h.update_scalar(t.quant_group_size);
        h.update_scalar(t.quant_channel_axis);
        h.update_scalar(static_cast<std::uint8_t>(t.quant_has_scale_dtype));
        h.update_scalar(static_cast<std::uint32_t>(t.quant_scale_dtype));
        h.update_scalar(static_cast<std::uint8_t>(t.quant_has_zero_point_dtype));
        h.update_scalar(static_cast<std::uint32_t>(t.quant_zero_point_dtype));
        rust_loader_hash_i64_slice(h, t.quant_block_shape);
    }
    if (const char* tag = std::getenv("PIE_WEIGHT_LOADER_COMPILE_CACHE_TAG");
        tag != nullptr && tag[0] != '\0') {
        h.update_bytes(tag, std::char_traits<char>::length(tag));
    }
    return h.hex();
}

inline bool rust_loader_compile_cache_enabled()
{
    if (const char* value = std::getenv("PIE_WEIGHT_LOADER_COMPILE_CACHE");
        value != nullptr) {
        return std::string(value) != "0" && std::string(value) != "false";
    }
    return true;
}

inline bool rust_loader_compile_cache_debug()
{
    return std::getenv("PIE_WEIGHT_LOADER_DEBUG") != nullptr ||
           std::getenv("PIE_WEIGHT_LOADER_COMPILE_CACHE_DEBUG") != nullptr;
}

inline std::filesystem::path rust_loader_compile_cache_dir()
{
    if (const char* dir = std::getenv("PIE_WEIGHT_LOADER_COMPILE_CACHE_DIR");
        dir != nullptr && dir[0] != '\0') {
        return std::filesystem::path(dir);
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME");
        xdg != nullptr && xdg[0] != '\0') {
        return std::filesystem::path(xdg) / "pie" / "weight-loader" /
               "compile-v1";
    }
    if (const char* home = std::getenv("HOME");
        home != nullptr && home[0] != '\0') {
        return std::filesystem::path(home) / ".cache" / "pie" /
               "weight-loader" / "compile-v1";
    }
    return std::filesystem::temp_directory_path() / "pie-weight-loader" /
           "compile-v1";
}

inline bool rust_loader_read_file(
    const std::filesystem::path& path,
    std::vector<std::uint8_t>& bytes)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const auto size = in.tellg();
    if (size < 0) return false;
    bytes.resize(static_cast<std::size_t>(size));
    in.seekg(0);
    if (!bytes.empty()) {
        in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    }
    return static_cast<bool>(in);
}

inline void rust_loader_write_file_atomic(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& bytes)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return;
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto tmp = path.string() + ".tmp." + std::to_string(now);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return;
        if (!bytes.empty()) {
            out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        if (!out) {
            std::filesystem::remove(tmp, ec);
            return;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
    }
}

inline RustStorageProgram compile_rust_storage_program_cached(
    const pie_weight_loader::PieLoaderCompileInput& input)
{
    if (!rust_loader_compile_cache_enabled()) {
        return compile_rust_storage_program(input);
    }

    const auto key = rust_loader_compile_cache_key(input);
    const auto path = rust_loader_compile_cache_dir() / (key + ".bin");
    if (std::getenv("PIE_WEIGHT_LOADER_COMPILE_CACHE_RESET") == nullptr) {
        std::vector<std::uint8_t> bytes;
        if (rust_loader_read_file(path, bytes)) {
            try {
                RustStorageProgram program =
                    deserialize_rust_storage_program(bytes);
                if (rust_loader_compile_cache_debug()) {
                    std::cerr << "[pie-weight-loader] compile cache hit "
                              << path << " bytes=" << bytes.size() << "\n";
                }
                return program;
            } catch (const std::exception& e) {
                if (rust_loader_compile_cache_debug()) {
                    std::cerr << "[pie-weight-loader] compile cache ignored "
                              << path << ": " << e.what() << "\n";
                }
            }
        }
    }

    if (rust_loader_compile_cache_debug()) {
        std::cerr << "[pie-weight-loader] compile cache miss " << path << "\n";
    }
    RustStorageProgram program = compile_rust_storage_program(input);
    try {
        std::vector<std::uint8_t> bytes =
            serialize_rust_storage_program(program);
        rust_loader_write_file_atomic(path, bytes);
        if (rust_loader_compile_cache_debug()) {
            std::cerr << "[pie-weight-loader] compile cache stored " << path
                      << " bytes=" << bytes.size() << "\n";
        }
    } catch (const std::exception& e) {
        if (rust_loader_compile_cache_debug()) {
            std::cerr << "[pie-weight-loader] compile cache store failed "
                      << path << ": " << e.what() << "\n";
        }
    }
    return program;
}

inline RustLoaderSourceIndex add_checkpoint_metadata_to_rust_input(
    RustLoaderInputBuilder& input,
    const SafetensorsCheckpointSource& loader,
    const HfConfig& hf)
{
    std::map<std::uint32_t, std::filesystem::path> files;
    std::map<std::uint32_t, std::uint64_t> file_sizes;

    auto names = loader.tensor_names();
    std::sort(names.begin(), names.end());

    RustLoaderSourceIndex index;
    index.tensor_names.reserve(names.size());
    std::uint32_t next_tensor_id = 0;
    for (const auto& name : names) {
        bool skip = false;
        for (const auto& prefix : hf.mm_skip_prefixes) {
            if (!prefix.empty() && name.rfind(prefix, 0) == 0) {
                skip = true;
                break;
            }
        }
        if (skip) continue;
        const TensorInfo& info = loader.info(name);
        const TensorStorageInfo storage = loader.storage_info(name);
        index.tensor_ids.emplace(name, next_tensor_id);
        files.emplace(storage.shard_id, storage.path);
        if (!storage.path.empty()) {
            std::error_code ec;
            const auto size = std::filesystem::file_size(storage.path, ec);
            if (!ec) {
                file_sizes[storage.shard_id] = size;
            }
        }
        input.add_tensor(
            next_tensor_id,
            name,
            storage.shard_id,
            storage.file_offset,
            storage.nbytes,
            info);
        index.tensor_names.push_back(name);
        ++next_tensor_id;
    }
    for (const auto& [file_id, path] : files) {
        input.add_file(
            file_id,
            path.string(),
            file_sizes[file_id],
            pie_weight_loader::PieLoaderCheckpointFormat::Safetensors);
    }
    return index;
}

inline bool rust_loader_ends_with(
    const std::string& value,
    const std::string& suffix)
{
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::vector<RustQuantAttachment> infer_rust_quant_attachments(
    const HfConfig& hf,
    const pie_weight_loader::PieLoaderStorageProgramView& view)
{
    std::vector<std::string> runtime_names;
    runtime_names.reserve(view.tensors.len);
    std::unordered_map<std::string, bool> present;
    present.reserve(view.tensors.len);
    for (std::size_t i = 0; i < view.tensors.len; ++i) {
        const auto name = rust_loader_bytes_to_string(view.tensors.ptr[i].name);
        runtime_names.push_back(name);
        present.emplace(name, true);
    }

    std::vector<RustQuantAttachment> attachments;
    for (std::size_t i = 0; i < view.tensors.len; ++i) {
        const auto& tensor = view.tensors.ptr[i];
        if (tensor.encoding_kind != pie_weight_loader::PieLoaderEncodingKind::Quant) {
            continue;
        }
        if (tensor.quant_scheme != pie_weight_loader::PieLoaderQuantScheme::Fp8E4M3 &&
            tensor.quant_scheme != pie_weight_loader::PieLoaderQuantScheme::Int8Symmetric) {
            continue;
        }
        const std::string name = rust_loader_bytes_to_string(tensor.name);
        const std::string scale = name + "_scale_inv";
        if (!present.contains(scale)) continue;
        attachments.push_back(RustQuantAttachment{
            .tensor_name = name,
            .scale_tensor_name = scale,
            .granularity = QuantGranularity::PerChannel,
            .group_size = 0,
            .channel_axis = 0,
        });
    }

    // DeepSeek V4: pair block-scaled FP8 (E4M3 weight + E8M0 scale).
    // Weight names end with `.weight`, scales end with `.scale`.
    // The scale tensor is E8M0 with shape [N/128, K/128] — each 128x128
    // block shares one scale (block-scaled quantization).
    if (hf.model_type == "deepseek_v4") {
        for (const auto& name : runtime_names) {
            if (!rust_loader_ends_with(name, ".weight")) continue;
            const std::string scale =
                name.substr(0, name.size() - std::string(".weight").size()) +
                ".scale";
            if (!present.contains(scale)) continue;
            attachments.push_back(RustQuantAttachment{
                .tensor_name = name,
                .scale_tensor_name = scale,
                .granularity = QuantGranularity::PerGroup,
                .group_size = 128,
                .channel_axis = 0,
            });
        }
        return attachments;
    }

    const bool is_gpt_oss =
        hf.model_type == "gpt_oss" || hf.model_type == "gpt-oss" ||
        hf.model_type == "gptoss";
    if (is_gpt_oss) {
        for (const auto& name : runtime_names) {
            if (!rust_loader_ends_with(name, ".weight")) continue;
            const std::string scale =
                name.substr(0, name.size() - std::string(".weight").size()) +
                ".weight_scale";
            if (!present.contains(scale)) continue;
            attachments.push_back(RustQuantAttachment{
                .tensor_name = name,
                .scale_tensor_name = scale,
                .granularity = QuantGranularity::PerGroup,
                .group_size = 32,
                .channel_axis = 1,
            });
        }
        return attachments;
    }

    // Dense pre-quantized block-FP8 (e.g. Qwen3-FP8, Llama-FP8): every linear
    // weight ships as raw F8_E4M3 paired with a [N/128, K/128] `weight_scale_inv`
    // block scale (weight_block_size [128,128]). Same handling as GLM/V4 block
    // FP8 — PerGroup/128 — but keyed off the FP8 quant_method so it covers any
    // dense arch. (deepseek_v4/glm/gpt_oss have their own scale conventions and
    // returned above.)
    if (hf.quant_method == "fp8" && hf.model_type != "deepseek_v4" &&
        hf.model_type != "glm_moe_dsa" && !is_gpt_oss) {
        std::unordered_map<std::string, pie_weight_loader::PieLoaderDType> dtype_for;
        dtype_for.reserve(view.tensors.len);
        for (std::size_t i = 0; i < view.tensors.len; ++i) {
            dtype_for.emplace(
                rust_loader_bytes_to_string(view.tensors.ptr[i].name),
                view.tensors.ptr[i].dtype);
        }
        for (const auto& name : runtime_names) {
            if (!rust_loader_ends_with(name, ".weight")) continue;
            auto it = dtype_for.find(name);
            if (it == dtype_for.end() ||
                it->second != pie_weight_loader::PieLoaderDType::F8E4M3) {
                continue;
            }
            const std::string scale = name + "_scale_inv";
            if (!present.contains(scale)) continue;
            attachments.push_back(RustQuantAttachment{
                .tensor_name = name,
                .scale_tensor_name = scale,
                .granularity = QuantGranularity::PerGroup,
                .group_size = 128,
                .channel_axis = 0,
            });
        }
        return attachments;
    }

    // GLM-5.1 ships scales as 2D block tensors (128x128 blocks), e.g.
    // weight [2048, 6144] pairs with weight_scale_inv [16, 48]. Use
    // PerGroup with group_size=128 — same handling as V4's block FP8.
    //
    // When runtime_quant=fp4 is enabled the expert weights are transcoded
    // to MXFP4 at materialize time (the FP8 source + its _scale_inv get
    // consumed by the encode TileMap). In that case the runtime contract
    // is no longer FP8 and must not get an FP8 PerGroup attachment — the
    // emitted MXFP4 weight pairs with its own E8M0 `_scale` tensor that
    // downstream code reads directly.
    if (hf.model_type == "glm_moe_dsa") {
        // Map runtime tensor name → final dtype. Attach PerGroup FP8
        // QuantMeta only to weights that end up as FP8 in the runtime
        // contract; the runtime fp4 path produces MXFP4 contracts whose
        // dtype is something else (UINT8 byte buffer) — those weights
        // pair with their own E8M0 `_scale` and must NOT get an FP8
        // PerGroup attachment here.
        // Look up dtype + quant_scheme per runtime tensor name.
        struct TensorInfo {
            pie_weight_loader::PieLoaderDType dtype;
            pie_weight_loader::PieLoaderEncodingKind encoding_kind;
            pie_weight_loader::PieLoaderQuantScheme scheme;
        };
        std::unordered_map<std::string, TensorInfo> info_for;
        info_for.reserve(view.tensors.len);
        for (std::size_t i = 0; i < view.tensors.len; ++i) {
            const auto& t = view.tensors.ptr[i];
            info_for.emplace(
                rust_loader_bytes_to_string(t.name),
                TensorInfo{t.dtype, t.encoding_kind, t.quant_scheme});
        }
        for (const auto& name : runtime_names) {
            if (!rust_loader_ends_with(name, ".weight")) continue;
            auto it = info_for.find(name);
            if (it == info_for.end()) continue;

            // MXFP4 runtime-quant output: encoding is Quant with the MXFP4
            // scheme. Pair with E8M0 `_scale` sibling and emit PerGroup.
            if (it->second.encoding_kind ==
                    pie_weight_loader::PieLoaderEncodingKind::Quant &&
                it->second.scheme ==
                    pie_weight_loader::PieLoaderQuantScheme::Mxfp4E2M1E8M0) {
                const std::string scale = name + "_scale";
                if (!present.contains(scale)) continue;
                attachments.push_back(RustQuantAttachment{
                    .tensor_name = name,
                    .scale_tensor_name = scale,
                    .granularity = QuantGranularity::PerGroup,
                    .group_size = 32,
                    .channel_axis = 1,
                });
                continue;
            }

            // Raw FP8 weight kept as-is: attach PerGroup 128 to pair with
            // its `_scale_inv` sibling.
            if (it->second.dtype ==
                pie_weight_loader::PieLoaderDType::F8E4M3) {
                const std::string scale = name + "_scale_inv";
                if (!present.contains(scale)) continue;
                attachments.push_back(RustQuantAttachment{
                    .tensor_name = name,
                    .scale_tensor_name = scale,
                    .granularity = QuantGranularity::PerGroup,
                    .group_size = 128,
                    .channel_axis = 0,
                });
            }
        }
    }
    return attachments;
}

// Read a runtime-compiled StorageProgram (weight-loader Variant A) from
// `path` — the serialized `bincode` program the runtime wrote in-process and
// handed off via `[model].storage_program_path` in the boot TOML. The **bulk
// weight bytes never cross**: the program records only tensor locations; the
// driver reads payloads from its own copy of the checkpoint during execution.
inline RustStorageProgram read_storage_program_from_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error(
            "engine: cannot open runtime storage-program file '" + path +
            "' ([model].storage_program_path). Bulk weights never cross this "
            "boundary — only the compiled StorageProgram IR is shipped.");
    }
    const std::streamsize size = in.tellg();
    if (size < 0) {
        throw std::runtime_error(
            "engine: cannot size runtime storage-program file '" + path + "'");
    }
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (size > 0 && !in.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error(
            "engine: failed to read runtime storage-program file '" + path + "'");
    }
    return deserialize_rust_storage_program(bytes);
}

inline RustLoaderCompileResult compile_rust_loader_plan_from_metadata(
    const HfConfig& hf,
    const SafetensorsCheckpointSource& loader,
    const std::string& runtime_quant,
    int tp_rank,
    int tp_size,
    std::uint64_t max_tile_bytes,
    std::uint32_t preferred_alignment,
    const BackendTarget& backend_target,
    const std::string& snapshot_dir,
    const std::string& storage_program_path)
{
    RustLoaderInputBuilder input;
    input.set_model(hf, runtime_quant);
    input.set_target(
        tp_rank,
        tp_size,
        max_tile_bytes,
        preferred_alignment,
        backend_target.mxfp4_moe,
        backend_target.mxfp4_native_gemm);
    input.set_runtime_abi_name("pie-cuda", /*version=*/1);

    RustLoaderSourceIndex source_index =
        add_checkpoint_metadata_to_rust_input(input, loader, hf);
    const std::string cache_key = rust_loader_compile_cache_key(input.view());
    (void)snapshot_dir;
    // Locality switch (weight-loader Variant A): an embedded/in-process driver's
    // runtime compiles the StorageProgram in-process and hands it off as a file
    // named in `[model].storage_program_path`; we only deserialize + execute it
    // here. Remote/out-of-process (and standalone) drivers leave the path empty
    // and keep the C++ FFI compile below. Either way the source index, quant
    // attachments, coverage counts, and cache key are computed identically from
    // the checkpoint metadata + HF config.
    RustStorageProgram program =
        !storage_program_path.empty()
            ? read_storage_program_from_file(storage_program_path)
            : compile_rust_storage_program_cached(input.view());
    const auto view = program.view();
    const std::size_t runtime_tensor_count = view.tensors.len;
    std::vector<RustQuantAttachment> attachments =
        infer_rust_quant_attachments(hf, view);
    return RustLoaderCompileResult{
        .program = std::move(program),
        .source_tensor_names = std::move(source_index.tensor_names),
        .quant_attachments = std::move(attachments),
        .source_tensor_count = source_index.tensor_ids.size(),
        .covered_contract_count = runtime_tensor_count,
        .runtime_tensor_count = runtime_tensor_count,
        .cache_key = cache_key,
    };
}

inline std::string describe_rust_storage_program(
    const pie_weight_loader::PieLoaderStorageProgramView& view,
    std::size_t source_tensor_count,
    std::size_t covered_contract_count,
    std::size_t runtime_tensor_count)
{
    std::uint64_t optimizer_rewrites = 0;
    for (std::size_t i = 0; i < view.optimizer.passes.len; ++i) {
        optimizer_rewrites += view.optimizer.passes.ptr[i].rewrites;
    }
    std::ostringstream out;
    out << "rust_storage_program(version=" << view.version
        << ", source_tensors=" << source_tensor_count
        << ", contracts=" << covered_contract_count << "/" << runtime_tensor_count
        << ", tensors=" << view.tensors.len
        << ", buffers=" << view.buffers.len
        << ", instrs=" << view.instrs.len
        << ", schedule=" << view.schedule.len
        << ", optimizer_passes=" << view.optimizer.passes.len
        << ", optimizer_rewrites=" << optimizer_rewrites
        << ", persistent_bytes=" << view.memory.persistent_bytes
        << ", read_bytes=" << view.memory.checkpoint_read_bytes
        << ", write_bytes=" << view.memory.device_write_bytes
        << ")";
    return out.str();
}

inline const char* rust_storage_instr_kind_name(
    pie_weight_loader::PieLoaderStorageInstrKind kind) noexcept
{
    switch (kind) {
    case pie_weight_loader::PieLoaderStorageInstrKind::Allocate:
        return "Allocate";
    case pie_weight_loader::PieLoaderStorageInstrKind::ExtentWrite:
        return "ExtentWrite";
    case pie_weight_loader::PieLoaderStorageInstrKind::TileMap:
        return "TileMap";
    case pie_weight_loader::PieLoaderStorageInstrKind::CreateView:
        return "CreateView";
    case pie_weight_loader::PieLoaderStorageInstrKind::Attach:
        return "Attach";
    case pie_weight_loader::PieLoaderStorageInstrKind::Release:
        return "Release";
    case pie_weight_loader::PieLoaderStorageInstrKind::Finalize:
        return "Finalize";
    case pie_weight_loader::PieLoaderStorageInstrKind::BulkExtentWrite:
        return "BulkExtentWrite";
    case pie_weight_loader::PieLoaderStorageInstrKind::SlabScatter:
        return "SlabScatter";
    }
    return "Unknown";
}

inline const char* rust_tile_map_kind_name(
    pie_weight_loader::PieLoaderTileMapKind kind) noexcept
{
    switch (kind) {
    case pie_weight_loader::PieLoaderTileMapKind::Cast:
        return "Cast";
    case pie_weight_loader::PieLoaderTileMapKind::Decode:
        return "Decode";
    case pie_weight_loader::PieLoaderTileMapKind::Encode:
        return "Encode";
    case pie_weight_loader::PieLoaderTileMapKind::Transcode:
        return "Transcode";
    case pie_weight_loader::PieLoaderTileMapKind::Reblock:
        return "Reblock";
    case pie_weight_loader::PieLoaderTileMapKind::Reorder:
        return "Reorder";
    case pie_weight_loader::PieLoaderTileMapKind::Repack:
        return "Repack";
    case pie_weight_loader::PieLoaderTileMapKind::None:
        return "None";
    }
    return "Unknown";
}

inline void dump_rust_count_map(
    std::ostringstream& out,
    const char* key,
    const std::map<std::string, std::size_t>& counts,
    const char* suffix)
{
    out << "  \"" << key << "\": {";
    bool first = true;
    for (const auto& [name, count] : counts) {
        if (!first) out << ", ";
        out << "\"" << name << "\": " << count;
        first = false;
    }
    out << "}" << suffix << "\n";
}

inline void dump_rust_optimizer_report(
    std::ostringstream& out,
    const pie_weight_loader::PieLoaderOptimizerReportView& optimizer,
    const char* suffix)
{
    out << "  \"optimizer\": {\n"
        << "    \"passes\": [\n";
    for (std::size_t i = 0; i < optimizer.passes.len; ++i) {
        const auto& pass = optimizer.passes.ptr[i];
        out << "      {\"name\": \""
            << rust_loader_bytes_to_string(pass.name)
            << "\", \"exprs_before\": " << pass.exprs_before
            << ", \"exprs_after\": " << pass.exprs_after
            << ", \"rewrites\": " << pass.rewrites << "}";
        if (i + 1 < optimizer.passes.len) out << ",";
        out << "\n";
    }
    out << "    ]\n"
        << "  }" << suffix << "\n";
}

inline std::string dump_rust_storage_program_json(
    const pie_weight_loader::PieLoaderStorageProgramView& view,
    std::size_t source_tensor_count,
    std::size_t covered_contract_count,
    std::size_t runtime_tensor_count)
{
    std::map<std::string, std::size_t> instruction_kinds;
    std::map<std::string, std::size_t> tile_map_kinds;
    for (std::size_t i = 0; i < view.instrs.len; ++i) {
        const auto& instr = view.instrs.ptr[i];
        instruction_kinds[rust_storage_instr_kind_name(instr.kind)] += 1;
        if (instr.kind ==
            pie_weight_loader::PieLoaderStorageInstrKind::TileMap) {
            tile_map_kinds[rust_tile_map_kind_name(instr.tile_kind)] += 1;
        }
    }
    std::ostringstream out;
    out << "{\n"
        << "  \"summary\": \""
        << describe_rust_storage_program(
               view, source_tensor_count, covered_contract_count,
               runtime_tensor_count)
        << "\",\n"
        << "  \"version\": " << view.version << ",\n"
        << "  \"source_tensor_count\": " << source_tensor_count << ",\n"
        << "  \"covered_contract_count\": " << covered_contract_count << ",\n"
        << "  \"runtime_tensor_count\": " << runtime_tensor_count << ",\n"
        << "  \"tensor_count\": " << view.tensors.len << ",\n"
        << "  \"buffer_count\": " << view.buffers.len << ",\n"
        << "  \"instruction_count\": " << view.instrs.len << ",\n"
        << "  \"schedule_count\": " << view.schedule.len << ",\n"
        << "  \"memory\": {\n"
        << "    \"persistent_bytes\": " << view.memory.persistent_bytes << ",\n"
        << "    \"temporary_peak_bytes\": "
        << view.memory.temporary_peak_bytes << ",\n"
        << "    \"transform_scratch_peak_bytes\": "
        << view.memory.transform_scratch_peak_bytes << ",\n"
        << "    \"checkpoint_read_bytes\": "
        << view.memory.checkpoint_read_bytes << ",\n"
        << "    \"device_write_bytes\": "
        << view.memory.device_write_bytes << "\n"
        << "  },\n";
    dump_rust_optimizer_report(out, view.optimizer, ",");
    dump_rust_count_map(out, "instruction_kinds", instruction_kinds, ",");
    dump_rust_count_map(out, "tile_map_kinds", tile_map_kinds, "");
    out
        << "}\n";
    return out.str();
}

}  // namespace pie_cuda_driver

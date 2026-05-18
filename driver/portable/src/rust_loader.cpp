#include "rust_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <weight_loader.h>

extern "C" {
#if defined(__GNUC__) || defined(__clang__)
#define PIE_PORTABLE_RUST_LOADER_WEAK __attribute__((weak))
#else
#define PIE_PORTABLE_RUST_LOADER_WEAK
#endif

pie_weight_loader::PieLoaderStatus pie_loader_compile(
    const pie_weight_loader::PieLoaderCompileInput* input,
    pie_weight_loader::PieLoaderProgramHandle** out_program,
    pie_weight_loader::PieLoaderError* out_error) PIE_PORTABLE_RUST_LOADER_WEAK;
pie_weight_loader::PieLoaderStorageProgramView pie_loader_program_view(
    const pie_weight_loader::PieLoaderProgramHandle* program)
    PIE_PORTABLE_RUST_LOADER_WEAK;
void pie_loader_program_free(
    pie_weight_loader::PieLoaderProgramHandle* program)
    PIE_PORTABLE_RUST_LOADER_WEAK;
void pie_loader_error_free(pie_weight_loader::PieLoaderError* error)
    PIE_PORTABLE_RUST_LOADER_WEAK;
}

#undef PIE_PORTABLE_RUST_LOADER_WEAK

namespace pie_portable_driver {

namespace {

enum class PlannerMode {
    Cpp,
    Rust,
    Dual,
};

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

PlannerMode parse_mode(const char* value) {
    const std::string mode = lowercase(value == nullptr ? std::string{} : value);
    if (mode.empty() || mode == "cpp") return PlannerMode::Cpp;
    if (mode == "rust") return PlannerMode::Rust;
    if (mode == "dual") return PlannerMode::Dual;
    throw std::runtime_error(
        "portable rust loader: PIE_PORTABLE_LOADER_PLANNER must be one of "
        "{cpp,rust,dual}");
}

const char* mode_name(PlannerMode mode) noexcept {
    switch (mode) {
        case PlannerMode::Cpp: return "cpp";
        case PlannerMode::Rust: return "rust";
        case PlannerMode::Dual: return "dual";
    }
    return "?";
}

std::string bytes_to_string(pie_weight_loader::PieLoaderBytes bytes) {
    if (bytes.ptr == nullptr || bytes.len == 0) return {};
    return std::string(
        reinterpret_cast<const char*>(bytes.ptr),
        reinterpret_cast<const char*>(bytes.ptr) + bytes.len);
}

std::vector<std::int64_t> shape_from_ggml(const ggml_tensor* tensor) {
    std::vector<std::int64_t> shape;
    for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) {
        if (tensor->ne[d] > 1 || !shape.empty()) {
            shape.push_back(static_cast<std::int64_t>(tensor->ne[d]));
        }
    }
    if (shape.empty()) shape.push_back(1);
    return shape;
}

std::string tensor_name(const ggml_tensor* tensor, const std::string& fallback) {
    const char* name = ggml_get_name(tensor);
    if (name != nullptr && name[0] != '\0') return std::string(name);
    return fallback;
}

std::optional<pie_weight_loader::PieLoaderDType> dtype_from_st(StDtype dtype) {
    switch (dtype) {
        case StDtype::F32:  return pie_weight_loader::PieLoaderDType::F32;
        case StDtype::F16:  return pie_weight_loader::PieLoaderDType::F16;
        case StDtype::BF16: return pie_weight_loader::PieLoaderDType::BF16;
        case StDtype::I32:  return pie_weight_loader::PieLoaderDType::I32;
        case StDtype::I16:  return pie_weight_loader::PieLoaderDType::I16;
        case StDtype::I8:   return pie_weight_loader::PieLoaderDType::I8;
        case StDtype::U32:  return pie_weight_loader::PieLoaderDType::U32;
        case StDtype::U16:  return pie_weight_loader::PieLoaderDType::U16;
        case StDtype::U8:   return pie_weight_loader::PieLoaderDType::U8;
        case StDtype::BOOL: return pie_weight_loader::PieLoaderDType::Bool;
        case StDtype::F8_E4M3:
            return pie_weight_loader::PieLoaderDType::F8E4M3;
        case StDtype::F8_E5M2:
            return pie_weight_loader::PieLoaderDType::F8E5M2;
        case StDtype::F64:
        case StDtype::I64:
        case StDtype::U64:
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<pie_weight_loader::PieLoaderDType> dtype_from_ggml(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return pie_weight_loader::PieLoaderDType::F32;
        case GGML_TYPE_F16:  return pie_weight_loader::PieLoaderDType::F16;
        case GGML_TYPE_BF16: return pie_weight_loader::PieLoaderDType::BF16;
        case GGML_TYPE_I32:  return pie_weight_loader::PieLoaderDType::I32;
        case GGML_TYPE_I16:  return pie_weight_loader::PieLoaderDType::I16;
        case GGML_TYPE_I8:   return pie_weight_loader::PieLoaderDType::I8;
        default:             return std::nullopt;
    }
}

bool same_dtype(StDtype src, ggml_type dst) {
    const auto src_dt = dtype_from_st(src);
    const auto dst_dt = dtype_from_ggml(dst);
    return src_dt && dst_dt && *src_dt == *dst_dt;
}

bool compact_extent(const pie_weight_loader::PieLoaderStridedExtentView& extent) {
    std::int64_t stride = static_cast<std::int64_t>(extent.element_bytes);
    for (std::size_t i = extent.dims.len; i > 0; --i) {
        const auto& dim = extent.dims.ptr[i - 1];
        if (dim.src_stride != stride || dim.dst_stride != stride) {
            return false;
        }
        stride *= dim.count;
    }
    return true;
}

class RustProgram {
public:
    explicit RustProgram(pie_weight_loader::PieLoaderProgramHandle* handle) noexcept
        : handle_(handle) {}

    RustProgram(const RustProgram&) = delete;
    RustProgram& operator=(const RustProgram&) = delete;

    RustProgram(RustProgram&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    ~RustProgram() {
        if (handle_ != nullptr && ::pie_loader_program_free != nullptr) {
            ::pie_loader_program_free(handle_);
        }
    }

    pie_weight_loader::PieLoaderStorageProgramView view() const {
        if (::pie_loader_program_view == nullptr || handle_ == nullptr) {
            return pie_weight_loader::PieLoaderStorageProgramView{};
        }
        return ::pie_loader_program_view(handle_);
    }

private:
    pie_weight_loader::PieLoaderProgramHandle* handle_ = nullptr;
};

class RustError {
public:
    ~RustError() {
        if (::pie_loader_error_free != nullptr) {
            ::pie_loader_error_free(&error_);
        }
    }

    pie_weight_loader::PieLoaderError* out() noexcept { return &error_; }

    std::string message() const {
        return error_.message == nullptr ? std::string{} : std::string(error_.message);
    }

private:
    pie_weight_loader::PieLoaderError error_{};
};

class InputBuilder {
public:
    pie_weight_loader::PieLoaderCompileInput view() const noexcept {
        return pie_weight_loader::PieLoaderCompileInput{
            .version = 1,
            .files = {.ptr = files_.data(), .len = files_.size()},
            .tensors = {.ptr = tensors_.data(), .len = tensors_.size()},
            .model = model_,
            .runtime_abi = runtime_abi_,
            .target = target_,
        };
    }

    std::uint32_t intern_string(std::string value) {
        strings_.push_back(std::move(value));
        return static_cast<std::uint32_t>(strings_.size() - 1);
    }

    pie_weight_loader::PieLoaderBytes bytes(std::uint32_t id) const noexcept {
        const auto& value = strings_[id];
        return pie_weight_loader::PieLoaderBytes{
            .ptr = reinterpret_cast<const std::uint8_t*>(value.data()),
            .len = value.size(),
        };
    }

    std::uint32_t intern_shape(std::vector<std::int64_t> shape) {
        shapes_.push_back(std::move(shape));
        return static_cast<std::uint32_t>(shapes_.size() - 1);
    }

    pie_weight_loader::PieLoaderI64Slice shape(std::uint32_t id) const noexcept {
        const auto& value = shapes_[id];
        return pie_weight_loader::PieLoaderI64Slice{
            .ptr = value.data(),
            .len = value.size(),
        };
    }

    std::uint32_t intern_u32_slice(std::vector<std::uint32_t> values) {
        u32_slices_.push_back(std::move(values));
        return static_cast<std::uint32_t>(u32_slices_.size() - 1);
    }

    pie_weight_loader::PieLoaderU32Slice u32_slice(std::uint32_t id) const noexcept {
        const auto& value = u32_slices_[id];
        return pie_weight_loader::PieLoaderU32Slice{
            .ptr = value.data(),
            .len = value.size(),
        };
    }

    void set_model(const Hparams& hp) {
        model_type_ = intern_string(hp.hf_model_type);
        quant_method_ = intern_string(std::string{});
        model_ = pie_weight_loader::PieLoaderModelConfigView{
            .model_type = bytes(model_type_),
            .quant_method = bytes(quant_method_),
            .num_hidden_layers =
                static_cast<std::uint32_t>(std::max(hp.num_hidden_layers, 0)),
            .num_experts = static_cast<std::uint32_t>(std::max(hp.num_experts, 0)),
            .num_experts_per_tok =
                static_cast<std::uint32_t>(std::max(hp.num_experts_per_tok, 0)),
        };
    }

    void set_target() {
        target_ = pie_weight_loader::PieLoaderBackendTargetView{
            .backend = pie_weight_loader::PieLoaderBackendKind::Portable,
            .tp_rank = 0,
            .tp_size = 1,
            .max_tile_bytes = 64ull * 1024ull * 1024ull,
            .preferred_alignment = 1,
            .mxfp4_moe =
                pie_weight_loader::PieLoaderMxfp4MoePolicy::RoutedDecode,
            .native_mxfp4_moe = false,
        };
    }

    void set_runtime_abi_name(std::string name, std::uint32_t version) {
        runtime_abi_name_ = intern_string(std::move(name));
        runtime_abi_.name = bytes(runtime_abi_name_);
        runtime_abi_.version = version;
        refresh_contract_slice();
    }

    void add_file(std::uint32_t id,
                  std::string path,
                  std::uint64_t size_bytes,
                  pie_weight_loader::PieLoaderCheckpointFormat format) {
        const auto path_id = intern_string(std::move(path));
        files_.push_back(pie_weight_loader::PieLoaderCheckpointFileView{
            .id = id,
            .path = bytes(path_id),
            .size_bytes = size_bytes,
            .format = format,
        });
    }

    void add_tensor(std::uint32_t id,
                    std::string name,
                    const StTensor& tensor,
                    pie_weight_loader::PieLoaderDType dtype) {
        add_raw_tensor(id, std::move(name), dtype, tensor.shape,
                       static_cast<std::uint64_t>(tensor.nbytes));
    }

    void add_raw_tensor(std::uint32_t id,
                        std::string name,
                        pie_weight_loader::PieLoaderDType dtype,
                        std::vector<std::int64_t> tensor_shape,
                        std::uint64_t span_bytes) {
        add_checkpoint_tensor(
            id,
            std::move(name),
            dtype,
            pie_weight_loader::PieLoaderEncodingKind::Raw,
            pie_weight_loader::PieLoaderQuantScheme::None,
            std::move(tensor_shape),
            span_bytes);
    }

    void add_quant_tensor(std::uint32_t id,
                          std::string name,
                          pie_weight_loader::PieLoaderDType logical_dtype,
                          pie_weight_loader::PieLoaderQuantScheme scheme,
                          std::vector<std::int64_t> tensor_shape,
                          std::uint64_t span_bytes) {
        add_checkpoint_tensor(
            id,
            std::move(name),
            logical_dtype,
            pie_weight_loader::PieLoaderEncodingKind::Quant,
            scheme,
            std::move(tensor_shape),
            span_bytes);
    }

    void add_checkpoint_tensor(std::uint32_t id,
                               std::string name,
                               pie_weight_loader::PieLoaderDType dtype,
                               pie_weight_loader::PieLoaderEncodingKind encoding_kind,
                               pie_weight_loader::PieLoaderQuantScheme quant_scheme,
                               std::vector<std::int64_t> tensor_shape,
                               std::uint64_t span_bytes) {
        const auto name_id = intern_string(std::move(name));
        const auto shape_id = intern_shape(std::move(tensor_shape));
        tensors_.push_back(pie_weight_loader::PieLoaderCheckpointTensorView{
            .id = id,
            .name = bytes(name_id),
            .file_id = 0,
            .file_offset = 0,
            .span_bytes = span_bytes,
            .dtype = dtype,
            .encoding_kind = encoding_kind,
            .quant_scheme = quant_scheme,
            .shape = shape(shape_id),
        });
    }

    void add_direct_contract(std::string output_name,
                             std::uint32_t source_tensor_id,
                             std::vector<std::uint32_t> metadata_tensor_ids,
                             pie_weight_loader::PieLoaderDType dtype,
                             std::vector<std::int64_t> tensor_shape) {
        const auto name_id = intern_string(std::move(output_name));
        const auto shape_id = intern_shape(std::move(tensor_shape));
        const auto metadata_id =
            intern_u32_slice(std::move(metadata_tensor_ids));
        contracts_.push_back(pie_weight_loader::PieLoaderRuntimeTensorContractView{
            .output_name = bytes(name_id),
            .source_kind = pie_weight_loader::PieLoaderRuntimeSourceKind::DirectTensor,
            .source_tensor_id = source_tensor_id,
            .source_tensor_ids = {},
            .byte_spans = {},
            .metadata_tensor_ids = u32_slice(metadata_id),
            .source_contract_id = UINT32_MAX,
            .semantic_role = pie_weight_loader::PieLoaderSemanticRole::DirectTensor,
            .layer = 0,
            .has_layer = false,
            .expert = 0,
            .has_expert = false,
            .axis = -1,
            .start = 0,
            .length = 0,
            .dtype = dtype,
            .encoding_kind = pie_weight_loader::PieLoaderEncodingKind::Raw,
            .quant_scheme = pie_weight_loader::PieLoaderQuantScheme::None,
            .shape = shape(shape_id),
            .alignment = 1,
            .shard_axis = -1,
        });
        refresh_contract_slice();
    }

    void add_byte_span_contract(
            std::string output_name,
            std::vector<pie_weight_loader::PieLoaderRuntimeByteSpanView> spans,
            pie_weight_loader::PieLoaderDType dtype,
            std::vector<std::int64_t> tensor_shape) {
        byte_spans_.push_back(std::move(spans));
        const auto name_id = intern_string(std::move(output_name));
        const auto shape_id = intern_shape(std::move(tensor_shape));
        const auto& stored_spans = byte_spans_.back();
        contracts_.push_back(pie_weight_loader::PieLoaderRuntimeTensorContractView{
            .output_name = bytes(name_id),
            .source_kind = pie_weight_loader::PieLoaderRuntimeSourceKind::ByteSpans,
            .source_tensor_id = UINT32_MAX,
            .source_tensor_ids = {},
            .byte_spans = {
                .ptr = stored_spans.data(),
                .len = stored_spans.size(),
            },
            .metadata_tensor_ids = {},
            .source_contract_id = UINT32_MAX,
            .semantic_role = pie_weight_loader::PieLoaderSemanticRole::DirectTensor,
            .layer = 0,
            .has_layer = false,
            .expert = 0,
            .has_expert = false,
            .axis = -1,
            .start = 0,
            .length = 0,
            .dtype = dtype,
            .encoding_kind = pie_weight_loader::PieLoaderEncodingKind::Raw,
            .quant_scheme = pie_weight_loader::PieLoaderQuantScheme::None,
            .shape = shape(shape_id),
            .alignment = 1,
            .shard_axis = -1,
        });
        refresh_contract_slice();
    }

private:
    void refresh_contract_slice() noexcept {
        runtime_abi_.tensors = pie_weight_loader::PieLoaderRuntimeTensorContractSlice{
            .ptr = contracts_.data(),
            .len = contracts_.size(),
        };
    }

    std::deque<std::string> strings_;
    std::deque<std::vector<std::int64_t>> shapes_;
    std::deque<std::vector<std::uint32_t>> u32_slices_;
    std::vector<pie_weight_loader::PieLoaderCheckpointFileView> files_;
    std::vector<pie_weight_loader::PieLoaderCheckpointTensorView> tensors_;
    std::deque<std::vector<pie_weight_loader::PieLoaderRuntimeByteSpanView>> byte_spans_;
    std::vector<pie_weight_loader::PieLoaderRuntimeTensorContractView> contracts_;
    std::uint32_t model_type_ = 0;
    std::uint32_t quant_method_ = 0;
    std::uint32_t runtime_abi_name_ = 0;
    pie_weight_loader::PieLoaderModelConfigView model_{};
    pie_weight_loader::PieLoaderRuntimeAbiView runtime_abi_{};
    pie_weight_loader::PieLoaderBackendTargetView target_{};
};

struct CompileResult {
    RustProgram program;
    std::vector<std::string> source_names;
    std::unordered_map<std::string, ggml_tensor*> targets;
    std::size_t source_tensor_count = 0;
    std::size_t emitted_contracts = 0;
    std::size_t required_contracts = 0;
};

struct ContractCandidate {
    std::string hf_name;
    ggml_tensor* tensor = nullptr;
    std::size_t src_offset_bytes = 0;
    std::size_t copy_bytes = 0;
    std::size_t dst_offset_bytes = 0;
    bool decode_fp8_to_bf16 = false;
    std::string fp8_scale_hf_name;
};

RustProgram compile_program(const pie_weight_loader::PieLoaderCompileInput& input) {
    if (::pie_loader_compile == nullptr) {
        throw std::runtime_error(
            "portable rust loader: pie-weight-loader symbols are not linked");
    }
    pie_weight_loader::PieLoaderProgramHandle* handle = nullptr;
    RustError error;
    const auto status = ::pie_loader_compile(&input, &handle, error.out());
    if (status != pie_weight_loader::PieLoaderStatus::Ok) {
        throw std::runtime_error(
            "portable rust loader compile failed: " + error.message());
    }
    return RustProgram(handle);
}

CompileResult compile_from_candidates(
        const Hparams& hparams,
        const std::filesystem::path& snapshot_dir,
        WeightArchive& archive,
        const std::vector<ContractCandidate>& candidates,
        std::size_t /*synth_count*/) {
    InputBuilder input;
    input.set_model(hparams);
    input.set_target();
    input.set_runtime_abi_name("pie-portable", /*version=*/1);

    const bool is_gguf_file =
        std::filesystem::is_regular_file(snapshot_dir) &&
        snapshot_dir.extension() == ".gguf";
    std::uint64_t file_size = 0;
    if (is_gguf_file) {
        std::error_code ec;
        const auto size = std::filesystem::file_size(snapshot_dir, ec);
        if (!ec) file_size = static_cast<std::uint64_t>(size);
    }
    input.add_file(
        0,
        snapshot_dir.string(),
        file_size,
        is_gguf_file
            ? pie_weight_loader::PieLoaderCheckpointFormat::Gguf
            : pie_weight_loader::PieLoaderCheckpointFormat::Safetensors);

    std::unordered_map<std::string, std::uint32_t> source_ids;
    std::vector<std::string> source_names;
    std::unordered_map<std::string, ggml_tensor*> targets;
    std::size_t emitted = 0;

    enum class SourceMode {
        Typed,
        Bytes,
        QuantFp8,
    };

    auto source_key = [](const std::string& name, SourceMode mode) {
        switch (mode) {
            case SourceMode::Typed:
                return name + "\n typed";
            case SourceMode::Bytes:
                return name + "\n bytes";
            case SourceMode::QuantFp8:
                return name + "\n quant-fp8";
        }
        return name;
    };

    auto source_id_for = [&](const std::string& name,
                             SourceMode mode) -> std::optional<std::uint32_t> {
        const auto key = source_key(name, mode);
        if (const auto it = source_ids.find(key); it != source_ids.end()) {
            return it->second;
        }
        const auto* tensor = archive.find(name);
        if (tensor == nullptr) {
            return std::nullopt;
        }
        if (mode == SourceMode::Typed) {
            if (tensor->ggml_type_override >= 0) return std::nullopt;
            if (!dtype_from_st(tensor->dtype)) return std::nullopt;
        } else if (mode == SourceMode::QuantFp8) {
            if (tensor->ggml_type_override >= 0 ||
                tensor->dtype != StDtype::F8_E4M3) {
                return std::nullopt;
            }
        }
        const auto id = static_cast<std::uint32_t>(source_ids.size());
        source_ids.emplace(key, id);
        source_names.push_back(name);
        if (mode == SourceMode::Bytes) {
            input.add_raw_tensor(
                id,
                name,
                pie_weight_loader::PieLoaderDType::U8,
                {static_cast<std::int64_t>(tensor->nbytes)},
                static_cast<std::uint64_t>(tensor->nbytes));
        } else if (mode == SourceMode::QuantFp8) {
            input.add_quant_tensor(
                id,
                name,
                pie_weight_loader::PieLoaderDType::BF16,
                pie_weight_loader::PieLoaderQuantScheme::Fp8E4M3,
                tensor->shape,
                static_cast<std::uint64_t>(tensor->nbytes));
        } else {
            const auto dtype = dtype_from_st(tensor->dtype);
            input.add_tensor(id, name, *tensor, *dtype);
        }
        return id;
    };

    std::unordered_map<std::string, std::vector<const ContractCandidate*>> groups;
    std::vector<std::string> group_order;
    for (const auto& d : candidates) {
        if (d.tensor == nullptr) continue;
        const auto output = tensor_name(d.tensor, d.hf_name);
        if (!groups.contains(output)) group_order.push_back(output);
        groups[output].push_back(&d);
    }

    for (const auto& output : group_order) {
        const auto& group = groups.at(output);
        if (group.empty()) continue;
        auto* target = group.front()->tensor;
        if (target == nullptr || targets.contains(output)) continue;

        if (group.size() == 1) {
            const auto& d = *group.front();
            if (d.copy_bytes == 0 && d.dst_offset_bytes == 0 &&
                d.src_offset_bytes == 0) {
                const auto* src = archive.find(d.hf_name);
                const auto dtype = dtype_from_ggml(target->type);
                if (src != nullptr && dtype && d.decode_fp8_to_bf16 &&
                    target->type == GGML_TYPE_BF16) {
                    const auto source_id =
                        source_id_for(d.hf_name, SourceMode::QuantFp8);
                    std::vector<std::uint32_t> metadata_ids;
                    if (!d.fp8_scale_hf_name.empty()) {
                        const auto scale_id =
                            source_id_for(d.fp8_scale_hf_name, SourceMode::Typed);
                        if (!scale_id) {
                            throw std::runtime_error(
                                "portable rust loader: missing FP8 scale tensor '" +
                                d.fp8_scale_hf_name + "'");
                        }
                        metadata_ids.push_back(*scale_id);
                    }
                    if (source_id) {
                        input.add_direct_contract(
                            output,
                            *source_id,
                            std::move(metadata_ids),
                            *dtype,
                            shape_from_ggml(target));
                        targets.emplace(output, target);
                        ++emitted;
                        continue;
                    }
                }
                const auto source_id = source_id_for(d.hf_name, SourceMode::Typed);
                if (src != nullptr && source_id && dtype) {
                    const bool cast_ok =
                        same_dtype(src->dtype, target->type) ||
                        (src->dtype == StDtype::BF16 && target->type == GGML_TYPE_F32) ||
                        (src->dtype == StDtype::F32 && target->type == GGML_TYPE_BF16);
                    if (cast_ok) {
                        input.add_direct_contract(
                            output,
                            *source_id,
                            {},
                            *dtype,
                            shape_from_ggml(target));
                        targets.emplace(output, target);
                        ++emitted;
                        continue;
                    }
                }
            }
        }

        const std::uint64_t target_bytes =
            static_cast<std::uint64_t>(ggml_nbytes(target));
        const auto typed_output = dtype_from_ggml(target->type);
        const auto output_dtype =
            typed_output.value_or(pie_weight_loader::PieLoaderDType::U8);
        const auto output_shape = typed_output
            ? shape_from_ggml(target)
            : std::vector<std::int64_t>{static_cast<std::int64_t>(target_bytes)};
        std::vector<pie_weight_loader::PieLoaderRuntimeByteSpanView> spans;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
        bool byte_spans_ok = true;
        for (const auto* candidate : group) {
            const auto& d = *candidate;
            const auto* src = archive.find(d.hf_name);
            if (src == nullptr) {
                byte_spans_ok = false;
                break;
            }
            const std::uint64_t span_bytes = static_cast<std::uint64_t>(
                d.copy_bytes == 0 ? src->nbytes : d.copy_bytes);
            const std::uint64_t src_offset =
                static_cast<std::uint64_t>(d.src_offset_bytes);
            const std::uint64_t dst_offset =
                static_cast<std::uint64_t>(d.dst_offset_bytes);
            if (span_bytes == 0 ||
                src_offset + span_bytes > static_cast<std::uint64_t>(src->nbytes) ||
                dst_offset + span_bytes > target_bytes) {
                byte_spans_ok = false;
                break;
            }
            const auto source_id = source_id_for(d.hf_name, SourceMode::Bytes);
            if (!source_id) {
                byte_spans_ok = false;
                break;
            }
            spans.push_back(pie_weight_loader::PieLoaderRuntimeByteSpanView{
                .source_tensor_id = *source_id,
                .source_offset_bytes = src_offset,
                .dest_offset_bytes = dst_offset,
                .span_bytes = span_bytes,
            });
            intervals.emplace_back(dst_offset, dst_offset + span_bytes);
        }

        if (!byte_spans_ok || spans.empty()) continue;
        std::sort(intervals.begin(), intervals.end());
        std::uint64_t cursor = 0;
        for (const auto& [begin, end] : intervals) {
            if (begin != cursor || end < begin) {
                byte_spans_ok = false;
                break;
            }
            cursor = end;
        }
        if (!byte_spans_ok || cursor != target_bytes) continue;

        input.add_byte_span_contract(output, std::move(spans), output_dtype, output_shape);
        targets.emplace(output, target);
        ++emitted;
    }

    CompileResult result{
        .program = compile_program(input.view()),
        .source_names = std::move(source_names),
        .targets = std::move(targets),
        .source_tensor_count = source_ids.size(),
        .emitted_contracts = emitted,
        .required_contracts = group_order.size(),
    };
    return result;
}

const pie_weight_loader::PieLoaderStorageInstrView& instruction(
    const pie_weight_loader::PieLoaderStorageProgramView& program,
    std::uint32_t id) {
    for (std::size_t i = 0; i < program.instrs.len; ++i) {
        if (program.instrs.ptr[i].id == id) return program.instrs.ptr[i];
    }
    throw std::runtime_error("portable rust loader: instruction id out of range");
}

const pie_weight_loader::PieLoaderBufferDeclView& buffer_decl(
    const pie_weight_loader::PieLoaderStorageProgramView& program,
    std::uint32_t id) {
    for (std::size_t i = 0; i < program.buffers.len; ++i) {
        if (program.buffers.ptr[i].id == id) return program.buffers.ptr[i];
    }
    throw std::runtime_error("portable rust loader: buffer id out of range");
}

const pie_weight_loader::PieLoaderTensorDeclView& tensor_decl(
    const pie_weight_loader::PieLoaderStorageProgramView& program,
    std::uint32_t id) {
    for (std::size_t i = 0; i < program.tensors.len; ++i) {
        if (program.tensors.ptr[i].id == id) return program.tensors.ptr[i];
    }
    throw std::runtime_error("portable rust loader: tensor id out of range");
}

std::uint16_t bf16_from_f32(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

float bf16_to_f32(std::uint16_t bits) {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &widened, sizeof(out));
    return out;
}

float fp16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000)) << 16;
    int exp = (h >> 10) & 0x1f;
    std::uint32_t mant = h & 0x03ff;
    std::uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ff;
            bits = sign |
                (static_cast<std::uint32_t>(exp + (127 - 15)) << 23) |
                (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign |
            (static_cast<std::uint32_t>(exp + (127 - 15)) << 23) |
            (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float f8_e4m3fn_to_f32(std::uint8_t v) {
    const float sign = (v & 0x80) ? -1.0f : 1.0f;
    const int exp = (v >> 3) & 0x0f;
    const int mant = v & 0x07;
    if (exp == 0 && mant == 0) return sign * 0.0f;
    if (exp == 0) {
        return sign * std::ldexp(static_cast<float>(mant) / 8.0f, -6);
    }
    if (exp == 0x0f && mant == 0x07) return 0.0f;
    return sign * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

float scalar_to_f32(const std::uint8_t* data, std::size_t index, StDtype dtype) {
    if (dtype == StDtype::F32) {
        return reinterpret_cast<const float*>(data)[index];
    }
    if (dtype == StDtype::BF16) {
        return bf16_to_f32(reinterpret_cast<const std::uint16_t*>(data)[index]);
    }
    if (dtype == StDtype::F16) {
        return fp16_to_f32(reinterpret_cast<const std::uint16_t*>(data)[index]);
    }
    throw std::runtime_error("portable rust loader: unsupported scale element size");
}

class Executor {
public:
    Executor(WeightArchive& archive,
             std::vector<std::string> source_names,
             std::unordered_map<std::string, ggml_tensor*> targets)
        : archive_(archive),
          source_names_(std::move(source_names)),
          targets_(std::move(targets)) {}

    void execute(const pie_weight_loader::PieLoaderStorageProgramView& program) {
        for (std::size_t i = 0; i < program.schedule.len; ++i) {
            const auto& instr = instruction(program, program.schedule.ptr[i]);
            switch (instr.kind) {
                case pie_weight_loader::PieLoaderStorageInstrKind::Allocate:
                    allocate(program, instr);
                    break;
                case pie_weight_loader::PieLoaderStorageInstrKind::ExtentWrite:
                    extent_write(instr);
                    break;
                case pie_weight_loader::PieLoaderStorageInstrKind::TileMap:
                    tile_map(instr);
                    break;
                case pie_weight_loader::PieLoaderStorageInstrKind::Finalize:
                    finalize(instr);
                    break;
                case pie_weight_loader::PieLoaderStorageInstrKind::Release:
                    buffers_.erase(instr.buffer_id);
                    break;
                case pie_weight_loader::PieLoaderStorageInstrKind::CreateView:
                    create_view(program, instr);
                    break;
                case pie_weight_loader::PieLoaderStorageInstrKind::Attach:
                    throw std::runtime_error(
                        "portable rust loader: instruction is not executable in "
                        "the portable direct/cast path");
            }
        }
    }

private:
    struct BufferEntry {
        ggml_tensor* tensor = nullptr;
        std::vector<std::uint8_t> bytes;
        std::optional<StDtype> source_dtype;
        std::vector<std::int64_t> source_shape;
    };

    const StTensor& source_tensor(std::uint32_t id) const {
        if (id >= source_names_.size()) {
            throw std::runtime_error("portable rust loader: source tensor id out of range");
        }
        return archive_.at(source_names_[id]);
    }

    const BufferEntry& buffer_entry(std::uint32_t buffer) const {
        const auto it = buffers_.find(buffer);
        if (it == buffers_.end()) {
            throw std::runtime_error("portable rust loader: buffer missing");
        }
        return it->second;
    }

    BufferEntry& buffer_entry(std::uint32_t buffer) {
        const auto it = buffers_.find(buffer);
        if (it == buffers_.end()) {
            throw std::runtime_error("portable rust loader: buffer missing");
        }
        return it->second;
    }

    std::size_t buffer_bytes(const BufferEntry& entry) const {
        if (entry.tensor != nullptr) return ggml_nbytes(entry.tensor);
        return entry.bytes.size();
    }

    static std::uint64_t extent_bytes(
        const pie_weight_loader::PieLoaderStridedExtentView& extent) {
        std::uint64_t elements = 1;
        for (std::size_t i = 0; i < extent.dims.len; ++i) {
            const auto count = extent.dims.ptr[i].count;
            if (count < 0) {
                throw std::runtime_error(
                    "portable rust loader: negative extent dimension");
            }
            const auto ucount = static_cast<std::uint64_t>(count);
            if (ucount != 0 &&
                elements > UINT64_MAX / ucount) {
                throw std::runtime_error(
                    "portable rust loader: extent element count overflow");
            }
            elements *= ucount;
        }
        if (extent.element_bytes != 0 &&
            elements > UINT64_MAX / extent.element_bytes) {
            throw std::runtime_error(
                "portable rust loader: extent byte count overflow");
        }
        return elements * extent.element_bytes;
    }

    std::size_t buffer_elements_bf16(const BufferEntry& entry) const {
        if (entry.tensor != nullptr) return static_cast<std::size_t>(ggml_nelements(entry.tensor));
        if (entry.bytes.size() % sizeof(std::uint16_t) != 0) {
            throw std::runtime_error("portable rust loader: BF16 buffer byte size mismatch");
        }
        return entry.bytes.size() / sizeof(std::uint16_t);
    }

    void write_buffer(std::uint32_t buffer,
                      const void* data,
                      std::uint64_t offset,
                      std::uint64_t nbytes) {
        auto& entry = buffer_entry(buffer);
        if (offset + nbytes > buffer_bytes(entry)) {
            throw std::runtime_error("portable rust loader: write exceeds buffer");
        }
        if (entry.tensor != nullptr) {
            ggml_backend_tensor_set(entry.tensor, data, offset, nbytes);
        } else {
            std::memcpy(entry.bytes.data() + offset, data, nbytes);
        }
    }

    void read_buffer(std::uint32_t buffer,
                     void* data,
                     std::uint64_t offset,
                     std::uint64_t nbytes) {
        const auto& entry = buffer_entry(buffer);
        if (offset + nbytes > buffer_bytes(entry)) {
            throw std::runtime_error("portable rust loader: read exceeds buffer");
        }
        if (entry.tensor != nullptr) {
            ggml_backend_tensor_get(entry.tensor, data, offset, nbytes);
        } else {
            std::memcpy(data, entry.bytes.data() + offset, nbytes);
        }
    }

    template <class CopyFn>
    void for_each_strided_span(
        const pie_weight_loader::PieLoaderStridedExtentView& src,
        const pie_weight_loader::PieLoaderStridedExtentView& dst,
        CopyFn&& copy) {
        if (src.element_bytes != dst.element_bytes) {
            throw std::runtime_error(
                "portable rust loader: strided copy element size mismatch");
        }
        if (src.dims.len != dst.dims.len) {
            throw std::runtime_error(
                "portable rust loader: strided copy rank mismatch");
        }
        const std::size_t rank = src.dims.len;
        for (std::size_t i = 0; i < rank; ++i) {
            if (src.dims.ptr[i].count != dst.dims.ptr[i].count) {
                throw std::runtime_error(
                    "portable rust loader: strided copy shape mismatch");
            }
        }
        const auto elem = static_cast<std::uint64_t>(src.element_bytes);
        if (rank == 0) {
            copy(src.base_offset, dst.base_offset, elem);
            return;
        }

        std::function<void(std::size_t, std::uint64_t, std::uint64_t)> walk =
            [&](std::size_t dim, std::uint64_t src_offset, std::uint64_t dst_offset) {
                if (dim == rank) {
                    copy(src_offset, dst_offset, elem);
                    return;
                }
                const auto& src_dim = src.dims.ptr[dim];
                const auto& dst_dim = dst.dims.ptr[dim];
                if (src_dim.count < 0) {
                    throw std::runtime_error(
                        "portable rust loader: negative strided copy count");
                }
                const auto count = static_cast<std::uint64_t>(src_dim.count);
                if (dim + 1 == rank &&
                    src_dim.src_stride == static_cast<std::int64_t>(elem) &&
                    dst_dim.dst_stride == static_cast<std::int64_t>(elem)) {
                    copy(src_offset, dst_offset, count * elem);
                    return;
                }
                for (std::uint64_t i = 0; i < count; ++i) {
                    walk(
                        dim + 1,
                        src_offset + i * static_cast<std::uint64_t>(src_dim.src_stride),
                        dst_offset + i * static_cast<std::uint64_t>(dst_dim.dst_stride));
                }
            };
        walk(0, src.base_offset, dst.base_offset);
    }

    const std::uint8_t* temp_data(std::uint32_t buffer) const {
        const auto& entry = buffer_entry(buffer);
        if (entry.tensor != nullptr) {
            throw std::runtime_error(
                "portable rust loader: expected CPU temporary buffer");
        }
        return entry.bytes.data();
    }

    void allocate(const pie_weight_loader::PieLoaderStorageProgramView& program,
                  const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        const auto& buffer = buffer_decl(program, instr.buffer_id);
        if (!buffer.has_tensor) {
            buffers_[buffer.id] = BufferEntry{
                .tensor = nullptr,
                .bytes = std::vector<std::uint8_t>(
                    static_cast<std::size_t>(buffer.bytes)),
            };
            return;
        }
        const auto& tensor = tensor_decl(program, buffer.tensor_id);
        const auto name = bytes_to_string(tensor.name);
        const auto target = targets_.find(name);
        if (target == targets_.end()) {
            throw std::runtime_error(
                "portable rust loader: no ggml tensor for runtime tensor '" + name + "'");
        }
        const auto nbytes = static_cast<std::uint64_t>(ggml_nbytes(target->second));
        if (buffer.bytes != nbytes) {
            throw std::runtime_error(
                "portable rust loader: buffer byte size mismatch for '" + name +
                "' (rust=" + std::to_string(buffer.bytes) +
                ", ggml=" + std::to_string(nbytes) + ")");
        }
        buffers_[buffer.id] = BufferEntry{.tensor = target->second};
    }

    void extent_write(const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        if (!instr.has_source || !instr.has_dest) {
            throw std::runtime_error("portable rust loader: ExtentWrite missing extent");
        }
        const auto& src = source_tensor(instr.source.tensor_id);
        if (instr.source.file_offset + instr.source.span_bytes >
            static_cast<std::uint64_t>(src.nbytes)) {
            throw std::runtime_error("portable rust loader: source extent out of range");
        }
        if (auto& entry = buffer_entry(instr.dest.buffer_id); entry.tensor == nullptr) {
            entry.source_dtype = src.dtype;
            entry.source_shape = src.shape;
        }
        if (compact_extent(instr.source.stride) && compact_extent(instr.dest.stride)) {
            write_buffer(
                instr.dest.buffer_id,
                src.data + instr.source.file_offset + instr.source.stride.base_offset,
                instr.dest.offset + instr.dest.stride.base_offset,
                instr.source.span_bytes);
            return;
        }
        for_each_strided_span(
            instr.source.stride,
            instr.dest.stride,
            [&](std::uint64_t src_offset,
                std::uint64_t dst_offset,
                std::uint64_t nbytes) {
                if (instr.source.file_offset + src_offset + nbytes >
                    static_cast<std::uint64_t>(src.nbytes)) {
                    throw std::runtime_error(
                        "portable rust loader: strided source span out of range");
                }
                write_buffer(
                    instr.dest.buffer_id,
                    src.data + instr.source.file_offset + src_offset,
                    instr.dest.offset + dst_offset,
                    nbytes);
            });
    }

    void tile_map(const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        switch (instr.tile_kind) {
            case pie_weight_loader::PieLoaderTileMapKind::Cast:
                cast_tile_map(instr);
                return;
            case pie_weight_loader::PieLoaderTileMapKind::Decode:
                decode_tile_map(instr);
                return;
            case pie_weight_loader::PieLoaderTileMapKind::Reblock:
                reblock_tile_map(instr);
                return;
            case pie_weight_loader::PieLoaderTileMapKind::Reorder:
                reorder_tile_map(instr);
                return;
            default:
                throw std::runtime_error(
                    "portable rust loader: unsupported TileMap kind");
        }
    }

    std::uint64_t tile_dest_offset(
        const pie_weight_loader::PieLoaderStorageInstrView& instr) const {
        return instr.has_dest ? instr.dest.offset + instr.dest.stride.base_offset : 0;
    }

    std::uint64_t tile_dest_bytes(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        std::uint32_t output) const {
        if (instr.has_dest) return extent_bytes(instr.dest.stride);
        return static_cast<std::uint64_t>(buffer_bytes(buffer_entry(output)));
    }

    void cast_tile_map(const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        if (!instr.has_source || instr.output_buffers.len != 1) {
            throw std::runtime_error(
                "portable rust loader: Cast TileMap expects a source and one output");
        }
        const auto& src = source_tensor(instr.source.tensor_id);
        if (instr.source.file_offset + instr.source.span_bytes > src.nbytes) {
            throw std::runtime_error("portable rust loader: cast source extent out of range");
        }
        const auto output = instr.output_buffers.ptr[0];
        const auto& dst = buffer_entry(output);
        const auto n = dst.tensor != nullptr
            ? static_cast<std::size_t>(ggml_nelements(dst.tensor))
            : buffer_bytes(dst) / sizeof(std::uint16_t);
        const auto* data = src.data + instr.source.file_offset;
        const ggml_type dst_type = dst.tensor != nullptr ? dst.tensor->type : GGML_TYPE_BF16;
        if (src.dtype == StDtype::BF16 && dst_type == GGML_TYPE_F32) {
            if (instr.source.span_bytes != n * sizeof(ggml_bf16_t)) {
                throw std::runtime_error(
                    "portable rust loader: bf16->f32 cast size mismatch");
            }
            std::vector<float> tmp(n);
            ggml_bf16_to_fp32_row(
                reinterpret_cast<const ggml_bf16_t*>(data),
                tmp.data(),
                n);
            write_buffer(output, tmp.data(), tile_dest_offset(instr), n * sizeof(float));
        } else if (src.dtype == StDtype::F32 && dst_type == GGML_TYPE_BF16) {
            if (instr.source.span_bytes != n * sizeof(float)) {
                throw std::runtime_error(
                    "portable rust loader: f32->bf16 cast size mismatch");
            }
            std::vector<ggml_bf16_t> tmp(n);
            ggml_fp32_to_bf16_row(
                reinterpret_cast<const float*>(data),
                tmp.data(),
                n);
            write_buffer(
                output,
                tmp.data(),
                tile_dest_offset(instr),
                n * sizeof(ggml_bf16_t));
        } else if (dst.tensor != nullptr && same_dtype(src.dtype, dst.tensor->type)) {
            if (instr.source.span_bytes != ggml_nbytes(dst.tensor)) {
                throw std::runtime_error(
                    "portable rust loader: identity cast byte size mismatch");
            }
            write_buffer(output, data, tile_dest_offset(instr), instr.source.span_bytes);
        } else {
            throw std::runtime_error(
                "portable rust loader: unsupported Cast TileMap dtype pair");
        }
    }

    void decode_tile_map(const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        if (!instr.has_source || instr.output_buffers.len != 1) {
            throw std::runtime_error(
                "portable rust loader: Decode TileMap expects a source and one output");
        }
        if (instr.transform_from != pie_weight_loader::PieLoaderQuantScheme::Fp8E4M3) {
            throw std::runtime_error(
                "portable rust loader: only FP8_E4M3 Decode TileMap is implemented");
        }
        const auto& src = source_tensor(instr.source.tensor_id);
        if (src.dtype != StDtype::F8_E4M3) {
            throw std::runtime_error("portable rust loader: Decode source is not F8_E4M3");
        }
        if (instr.source.file_offset + instr.source.span_bytes > src.nbytes) {
            throw std::runtime_error("portable rust loader: decode source extent out of range");
        }
        const auto output = instr.output_buffers.ptr[0];
        const auto& dst = buffer_entry(output);
        const auto n = buffer_elements_bf16(dst);
        if (instr.source.span_bytes != n) {
            throw std::runtime_error("portable rust loader: FP8 decode byte size mismatch");
        }
        const std::uint8_t* scale_data = nullptr;
        std::size_t scale_bytes = 0;
        if (instr.input_buffers.len > 0) {
            const auto& scale = buffer_entry(instr.input_buffers.ptr[0]);
            if (scale.tensor != nullptr) {
                throw std::runtime_error(
                    "portable rust loader: Decode scale must be a temporary buffer");
            }
            scale_data = scale.bytes.data();
            scale_bytes = scale.bytes.size();
        }
        const std::size_t rows = src.shape.empty()
            ? 1
            : static_cast<std::size_t>(src.shape.front());
        const std::size_t cols = rows == 0 ? 0 : n / rows;
        if (rows == 0 || rows * cols != n) {
            throw std::runtime_error("portable rust loader: invalid FP8 source shape");
        }
        std::size_t scale_count = 0;
        std::size_t scale_elem = 0;
        StDtype scale_dtype = StDtype::F32;
        if (scale_data != nullptr) {
            const auto& scale = buffer_entry(instr.input_buffers.ptr[0]);
            if (!scale.source_dtype) {
                throw std::runtime_error(
                    "portable rust loader: Decode scale dtype is unknown");
            }
            scale_dtype = *scale.source_dtype;
            switch (scale_dtype) {
                case StDtype::F32:
                    scale_elem = sizeof(float);
                    break;
                case StDtype::BF16:
                case StDtype::F16:
                    scale_elem = sizeof(std::uint16_t);
                    break;
                default:
                    throw std::runtime_error(
                        "portable rust loader: unsupported Decode scale dtype");
            }
            if (scale_bytes % scale_elem != 0) {
                throw std::runtime_error(
                    "portable rust loader: FP8 scale byte size is not aligned");
            }
            scale_count = scale_bytes / scale_elem;
            if (scale_count != 1 && scale_count != rows) {
                throw std::runtime_error(
                    "portable rust loader: FP8 scale must be scalar or per-row");
            }
        }
        std::vector<std::uint16_t> tmp(n);
        const auto* src8 = src.data + instr.source.file_offset;
        for (std::size_t i = 0; i < n; ++i) {
            float scale = 1.0f;
            if (scale_count == 1) {
                scale = scalar_to_f32(scale_data, 0, scale_dtype);
            } else if (scale_count == rows) {
                scale = scalar_to_f32(scale_data, i / cols, scale_dtype);
            }
            tmp[i] = bf16_from_f32(f8_e4m3fn_to_f32(src8[i]) * scale);
        }
        write_buffer(
            output,
            tmp.data(),
            tile_dest_offset(instr),
            tmp.size() * sizeof(std::uint16_t));
    }

    void reblock_tile_map(const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        if (instr.input_buffers.len != 1 || instr.output_buffers.len != 1) {
            throw std::runtime_error(
                "portable rust loader: Reblock TileMap expects one input and one output");
        }
        const auto input = instr.input_buffers.ptr[0];
        const auto output = instr.output_buffers.ptr[0];
        const auto nbytes = tile_dest_bytes(instr, output);
        std::vector<std::uint8_t> tmp(static_cast<std::size_t>(nbytes));
        read_buffer(input, tmp.data(), 0, nbytes);
        write_buffer(output, tmp.data(), tile_dest_offset(instr), nbytes);
    }

    void reorder_tile_map(const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        // The v1 FFI does not yet carry a permutation payload. Treat
        // identity reorders as a byte reblock and fail other cases in the
        // compiler before they reach a production executor.
        reblock_tile_map(instr);
    }

    void create_view(const pie_weight_loader::PieLoaderStorageProgramView& program,
                     const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        if (instr.input_buffers.len != 1 || instr.output_buffers.len != 1 ||
            !instr.has_dest) {
            throw std::runtime_error(
                "portable rust loader: CreateView expects one input, one output, and a view");
        }
        const auto input = instr.input_buffers.ptr[0];
        const auto output = instr.output_buffers.ptr[0];
        const auto nbytes = extent_bytes(instr.dest.stride);
        BufferEntry entry;
        const auto& output_buffer = buffer_decl(program, output);
        if (!output_buffer.has_tensor) {
            entry.bytes.resize(static_cast<std::size_t>(nbytes));
        } else if (const auto target = targets_.find(bytes_to_string(
                       tensor_decl(program, output_buffer.tensor_id).name));
            target != targets_.end()) {
            entry.tensor = target->second;
            if (nbytes > ggml_nbytes(entry.tensor)) {
                throw std::runtime_error(
                    "portable rust loader: CreateView target is smaller than view");
            }
        } else {
            entry.bytes.resize(static_cast<std::size_t>(nbytes));
        }
        buffers_[output] = std::move(entry);
        std::vector<std::uint8_t> tmp(static_cast<std::size_t>(nbytes));
        read_buffer(input, tmp.data(), instr.dest.offset + instr.dest.stride.base_offset, nbytes);
        write_buffer(output, tmp.data(), 0, nbytes);
    }

    void finalize(const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        if (instr.output_buffers.len == 1) {
            buffers_.erase(instr.output_buffers.ptr[0]);
        } else {
            buffers_.erase(instr.buffer_id);
        }
    }

    WeightArchive& archive_;
    std::vector<std::string> source_names_;
    std::unordered_map<std::string, ggml_tensor*> targets_;
    std::unordered_map<std::uint32_t, BufferEntry> buffers_;
};

std::string describe_program(const pie_weight_loader::PieLoaderStorageProgramView& view,
                             const CompileResult& result) {
    std::ostringstream out;
    out << "rust_storage_program(version=" << view.version
        << ", source_tensors=" << result.source_tensor_count
        << ", contracts=" << result.emitted_contracts << "/"
        << result.required_contracts
        << ", tensors=" << view.tensors.len
        << ", buffers=" << view.buffers.len
        << ", instrs=" << view.instrs.len
        << ", schedule=" << view.schedule.len
        << ", persistent_bytes=" << view.memory.persistent_bytes
        << ", read_bytes=" << view.memory.checkpoint_read_bytes
        << ", write_bytes=" << view.memory.device_write_bytes
        << ")";
    return out.str();
}

const char* storage_instr_kind_name(
    pie_weight_loader::PieLoaderStorageInstrKind kind) {
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
    }
    return "Unknown";
}

const char* tile_map_kind_name(pie_weight_loader::PieLoaderTileMapKind kind) {
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
        case pie_weight_loader::PieLoaderTileMapKind::None:
            return "None";
    }
    return "Unknown";
}

void dump_count_map(std::ostringstream& out,
                    const char* key,
                    const std::map<std::string, std::size_t>& counts,
                    const char* suffix) {
    out << "  \"" << key << "\": {";
    bool first = true;
    for (const auto& [name, count] : counts) {
        if (!first) out << ", ";
        out << "\"" << name << "\": " << count;
        first = false;
    }
    out << "}" << suffix << "\n";
}

std::string dump_program_json(const pie_weight_loader::PieLoaderStorageProgramView& view,
                              const CompileResult& result) {
    std::map<std::string, std::size_t> instruction_kinds;
    std::map<std::string, std::size_t> tile_map_kinds;
    for (std::size_t i = 0; i < view.instrs.len; ++i) {
        const auto& instr = view.instrs.ptr[i];
        instruction_kinds[storage_instr_kind_name(instr.kind)] += 1;
        if (instr.kind == pie_weight_loader::PieLoaderStorageInstrKind::TileMap) {
            tile_map_kinds[tile_map_kind_name(instr.tile_kind)] += 1;
        }
    }
    std::ostringstream out;
    out << "{\n"
        << "  \"summary\": \"" << describe_program(view, result) << "\",\n"
        << "  \"version\": " << view.version << ",\n"
        << "  \"source_tensor_count\": " << result.source_tensor_count << ",\n"
        << "  \"direct_contract_count\": " << result.emitted_contracts << ",\n"
        << "  \"required_contract_count\": " << result.required_contracts << ",\n"
        << "  \"tensor_count\": " << view.tensors.len << ",\n"
        << "  \"buffer_count\": " << view.buffers.len << ",\n"
        << "  \"instruction_count\": " << view.instrs.len << ",\n"
        << "  \"schedule_count\": " << view.schedule.len << ",\n"
        << "  \"memory\": {\n"
        << "    \"persistent_bytes\": " << view.memory.persistent_bytes << ",\n"
        << "    \"temporary_peak_bytes\": " << view.memory.temporary_peak_bytes << ",\n"
        << "    \"transform_scratch_peak_bytes\": "
        << view.memory.transform_scratch_peak_bytes << ",\n"
        << "    \"checkpoint_read_bytes\": "
        << view.memory.checkpoint_read_bytes << ",\n"
        << "    \"device_write_bytes\": " << view.memory.device_write_bytes << "\n"
        << "  },\n";
    dump_count_map(out, "instruction_kinds", instruction_kinds, ",");
    dump_count_map(out, "tile_map_kinds", tile_map_kinds, "");
    out
        << "}\n";
    return out.str();
}

void maybe_dump_program(const pie_weight_loader::PieLoaderStorageProgramView& view,
                        const CompileResult& result) {
    const char* dump_path = std::getenv("PIE_PORTABLE_RUST_LAYOUT_PLAN_DUMP");
    if (dump_path == nullptr || dump_path[0] == '\0') return;
    std::ofstream out(dump_path);
    if (!out) {
        throw std::runtime_error(
            "portable rust loader: failed to open PIE_PORTABLE_RUST_LAYOUT_PLAN_DUMP path: " +
            std::string(dump_path));
    }
    out << dump_program_json(view, result);
}

}  // namespace

bool try_load_with_rust_storage_program(Model& model, const char* planner_mode) {
    const PlannerMode mode = parse_mode(planner_mode);
    if (mode == PlannerMode::Cpp) return false;

    std::vector<ContractCandidate> candidates;
    candidates.reserve(model.declared_.size());
    for (const auto& d : model.declared_) {
        candidates.push_back(ContractCandidate{
            .hf_name = d.hf_name,
            .tensor = d.tensor,
            .src_offset_bytes = d.src_offset_bytes,
            .copy_bytes = d.copy_bytes,
            .dst_offset_bytes = d.dst_offset_bytes,
            .decode_fp8_to_bf16 = d.decode_fp8_to_bf16,
            .fp8_scale_hf_name = d.fp8_scale_hf_name,
        });
    }

    CompileResult result = compile_from_candidates(
        model.hparams_,
        model.snapshot_dir_,
        *model.archive_,
        candidates,
        model.synth_.size());
    const auto view = result.program.view();
    maybe_dump_program(view, result);
    std::cerr << "[model] " << describe_program(view, result)
              << " (planner=" << mode_name(mode) << ")\n";

    const bool complete = result.emitted_contracts == result.required_contracts;
    if (!complete) {
        if (mode == PlannerMode::Dual) return false;
        throw std::runtime_error(
            "portable rust loader: incomplete contract coverage (" +
            std::to_string(result.emitted_contracts) + "/" +
            std::to_string(result.required_contracts) + ")");
    }

    if (mode == PlannerMode::Dual) return false;

    Executor executor(*model.archive_, std::move(result.source_names), std::move(result.targets));
    executor.execute(view);
    for (const auto& s : model.synth_) {
        ggml_backend_tensor_set(s.tensor, s.data.data(), 0, s.data.size());
    }
    return true;
}

}  // namespace pie_portable_driver

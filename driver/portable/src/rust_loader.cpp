#include "rust_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
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
        const auto name_id = intern_string(std::move(name));
        const auto shape_id = intern_shape(tensor.shape);
        tensors_.push_back(pie_weight_loader::PieLoaderCheckpointTensorView{
            .id = id,
            .name = bytes(name_id),
            .file_id = 0,
            .file_offset = 0,
            .span_bytes = static_cast<std::uint64_t>(tensor.nbytes),
            .dtype = dtype,
            .encoding_kind = pie_weight_loader::PieLoaderEncodingKind::Raw,
            .quant_scheme = pie_weight_loader::PieLoaderQuantScheme::None,
            .shape = shape(shape_id),
        });
    }

    void add_direct_contract(std::string output_name,
                             std::uint32_t source_tensor_id,
                             pie_weight_loader::PieLoaderDType dtype,
                             std::vector<std::int64_t> tensor_shape) {
        const auto name_id = intern_string(std::move(output_name));
        const auto shape_id = intern_shape(std::move(tensor_shape));
        contracts_.push_back(pie_weight_loader::PieLoaderRuntimeTensorContractView{
            .output_name = bytes(name_id),
            .source_kind = pie_weight_loader::PieLoaderRuntimeSourceKind::DirectTensor,
            .source_tensor_id = source_tensor_id,
            .source_tensor_ids = {},
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
    std::vector<pie_weight_loader::PieLoaderCheckpointFileView> files_;
    std::vector<pie_weight_loader::PieLoaderCheckpointTensorView> tensors_;
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
    std::size_t copy_bytes = 0;
    std::size_t dst_offset_bytes = 0;
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
        std::size_t synth_count) {
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

    auto source_id_for = [&](const std::string& name) -> std::optional<std::uint32_t> {
        if (const auto it = source_ids.find(name); it != source_ids.end()) {
            return it->second;
        }
        const auto* tensor = archive.find(name);
        if (tensor == nullptr || tensor->ggml_type_override >= 0) {
            return std::nullopt;
        }
        const auto dtype = dtype_from_st(tensor->dtype);
        if (!dtype) return std::nullopt;
        const auto id = static_cast<std::uint32_t>(source_ids.size());
        source_ids.emplace(name, id);
        source_names.push_back(name);
        input.add_tensor(id, name, *tensor, *dtype);
        return id;
    };

    for (const auto& d : candidates) {
        if (d.copy_bytes != 0 || d.dst_offset_bytes != 0) {
            continue;
        }
        const auto* src = archive.find(d.hf_name);
        if (src == nullptr) continue;
        if (src->ggml_type_override >= 0) continue;
        const auto source_id = source_id_for(d.hf_name);
        const auto dtype = dtype_from_ggml(d.tensor->type);
        if (!source_id || !dtype) continue;
        const auto output = tensor_name(d.tensor, d.hf_name);
        if (targets.contains(output)) {
            continue;
        }
        const bool cast_ok =
            same_dtype(src->dtype, d.tensor->type) ||
            (src->dtype == StDtype::BF16 && d.tensor->type == GGML_TYPE_F32) ||
            (src->dtype == StDtype::F32 && d.tensor->type == GGML_TYPE_BF16);
        if (!cast_ok) continue;
        input.add_direct_contract(output, *source_id, *dtype, shape_from_ggml(d.tensor));
        targets.emplace(output, d.tensor);
        ++emitted;
    }

    CompileResult result{
        .program = compile_program(input.view()),
        .source_names = std::move(source_names),
        .targets = std::move(targets),
        .source_tensor_count = source_ids.size(),
        .emitted_contracts = emitted,
        .required_contracts = candidates.size() + synth_count,
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
                case pie_weight_loader::PieLoaderStorageInstrKind::Attach:
                    throw std::runtime_error(
                        "portable rust loader: instruction is not executable in "
                        "the portable direct/cast path");
            }
        }
    }

private:
    const StTensor& source_tensor(std::uint32_t id) const {
        if (id >= source_names_.size()) {
            throw std::runtime_error("portable rust loader: source tensor id out of range");
        }
        return archive_.at(source_names_[id]);
    }

    ggml_tensor* target_for_buffer(std::uint32_t buffer) const {
        const auto it = buffers_.find(buffer);
        if (it == buffers_.end()) {
            throw std::runtime_error("portable rust loader: target buffer missing");
        }
        return it->second;
    }

    void allocate(const pie_weight_loader::PieLoaderStorageProgramView& program,
                  const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        const auto& buffer = buffer_decl(program, instr.buffer_id);
        if (!buffer.has_tensor) {
            throw std::runtime_error(
                "portable rust loader: temporary allocation is not supported");
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
        buffers_[buffer.id] = target->second;
    }

    void extent_write(const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        if (!instr.has_source || !instr.has_dest) {
            throw std::runtime_error("portable rust loader: ExtentWrite missing extent");
        }
        if (!compact_extent(instr.source.stride) || !compact_extent(instr.dest.stride)) {
            throw std::runtime_error(
                "portable rust loader: non-compact ExtentWrite is not implemented");
        }
        const auto& src = source_tensor(instr.source.tensor_id);
        if (instr.source.file_offset + instr.source.span_bytes > src.nbytes) {
            throw std::runtime_error("portable rust loader: source extent out of range");
        }
        auto* dst = target_for_buffer(instr.dest.buffer_id);
        ggml_backend_tensor_set(
            dst,
            src.data + instr.source.file_offset,
            instr.dest.offset,
            instr.source.span_bytes);
    }

    void tile_map(const pie_weight_loader::PieLoaderStorageInstrView& instr) {
        if (instr.tile_kind != pie_weight_loader::PieLoaderTileMapKind::Cast) {
            throw std::runtime_error(
                "portable rust loader: only Cast TileMap is implemented");
        }
        if (!instr.has_source || instr.output_buffers.len != 1) {
            throw std::runtime_error(
                "portable rust loader: Cast TileMap expects a source and one output");
        }
        const auto& src = source_tensor(instr.source.tensor_id);
        if (instr.source.file_offset + instr.source.span_bytes > src.nbytes) {
            throw std::runtime_error("portable rust loader: cast source extent out of range");
        }
        auto* dst = target_for_buffer(instr.output_buffers.ptr[0]);
        const auto n = static_cast<std::size_t>(ggml_nelements(dst));
        const auto* data = src.data + instr.source.file_offset;
        if (src.dtype == StDtype::BF16 && dst->type == GGML_TYPE_F32) {
            if (instr.source.span_bytes != n * sizeof(ggml_bf16_t)) {
                throw std::runtime_error(
                    "portable rust loader: bf16->f32 cast size mismatch");
            }
            std::vector<float> tmp(n);
            ggml_bf16_to_fp32_row(
                reinterpret_cast<const ggml_bf16_t*>(data),
                tmp.data(),
                n);
            ggml_backend_tensor_set(dst, tmp.data(), 0, n * sizeof(float));
        } else if (src.dtype == StDtype::F32 && dst->type == GGML_TYPE_BF16) {
            if (instr.source.span_bytes != n * sizeof(float)) {
                throw std::runtime_error(
                    "portable rust loader: f32->bf16 cast size mismatch");
            }
            std::vector<ggml_bf16_t> tmp(n);
            ggml_fp32_to_bf16_row(
                reinterpret_cast<const float*>(data),
                tmp.data(),
                n);
            ggml_backend_tensor_set(dst, tmp.data(), 0, n * sizeof(ggml_bf16_t));
        } else if (same_dtype(src.dtype, dst->type)) {
            if (instr.source.span_bytes != ggml_nbytes(dst)) {
                throw std::runtime_error(
                    "portable rust loader: identity cast byte size mismatch");
            }
            ggml_backend_tensor_set(dst, data, 0, instr.source.span_bytes);
        } else {
            throw std::runtime_error(
                "portable rust loader: unsupported Cast TileMap dtype pair");
        }
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
    std::unordered_map<std::uint32_t, ggml_tensor*> buffers_;
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

std::string dump_program_json(const pie_weight_loader::PieLoaderStorageProgramView& view,
                              const CompileResult& result) {
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
        << "  }\n"
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
            .copy_bytes = d.copy_bytes,
            .dst_offset_bytes = d.dst_offset_bytes,
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
    return true;
}

}  // namespace pie_portable_driver

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace pie_load_planner {

enum class PieLoaderBackendKind { Cuda = 0, Metal = 1, Unknown = 255 };
enum class PieLoaderDType {
    F32, F16, BF16, F8E4M3, F8E5M2, I32, I16, I8, U32, U16, U8, Bool,
};
enum class PieLoaderEncodingKind { Raw, Quant };
enum class PieLoaderMxfp4MoePolicy { RoutedDecode, NativeGemm, EagerBf16 };
enum class PieLoaderQuantScheme {
    None,
    Fp8E4M3,
    Fp8E5M2,
    Int8Symmetric,
    Int8Asymmetric,
    AwqInt4,
    GptqInt4,
    Mxfp4E2M1E8M0,
    GgufQ4_0,
    GgufQ4K,
    GgufQ5_0,
    GgufQ5K,
    GgufQ8_0,
    MlxAffineU4,
};
enum class PieLoaderRepackLayout {
    None, MarlinMxfp4Weight, MarlinMxfp4Scale, DenseRowGather,
};
enum class PieLoaderRowMap { Identity, Even, Odd };
enum class PieLoaderStorageInstrKind {
    Allocate,
    ExtentWrite,
    TileMap,
    CreateView,
    Attach,
    Release,
    Finalize,
    BulkExtentWrite,
    SlabScatter,
};
enum class PieLoaderTileMapKind {
    Cast, Decode, Encode, Transcode, Reblock, Reorder, Repack, None,
};
inline constexpr std::uint32_t kTileMapCast = 1u << 0;
inline constexpr std::uint32_t kTileMapDecode = 1u << 1;
inline constexpr std::uint32_t kTileMapEncode = 1u << 2;
inline constexpr std::uint32_t kTileMapTranscode = 1u << 3;
inline constexpr std::uint32_t kTileMapReblock = 1u << 4;
inline constexpr std::uint32_t kTileMapReorder = 1u << 5;
inline constexpr std::uint32_t kTileMapRepack = 1u << 6;
inline constexpr std::uint32_t kCudaTileMapMask =
    kTileMapCast | kTileMapEncode | kTileMapReblock | kTileMapReorder |
    kTileMapRepack;
inline constexpr std::uint32_t kMetalTileMapMask = 0;

struct PieLoaderBytes {
    const std::uint8_t* ptr = nullptr;
    std::size_t len = 0;
};
struct PieLoaderU32Slice {
    const std::uint32_t* ptr = nullptr;
    std::size_t len = 0;
};
struct PieLoaderI64Slice {
    const std::int64_t* ptr = nullptr;
    std::size_t len = 0;
};
struct PieLoaderDimSpecView {
    std::int64_t count = 0;
    std::int64_t src_stride = 0;
    std::int64_t dst_stride = 0;
};
struct PieLoaderDimSpecSlice {
    const PieLoaderDimSpecView* ptr = nullptr;
    std::size_t len = 0;
};
using PieLoaderBufferIdSlice = PieLoaderU32Slice;
struct PieLoaderStridedExtentView {
    std::uint64_t base_offset = 0;
    std::uint32_t element_bytes = 0;
    PieLoaderDimSpecSlice dims{};
};
struct PieLoaderSourceExtentView {
    std::uint32_t file_id = 0;
    std::uint32_t tensor_id = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t span_bytes = 0;
    PieLoaderStridedExtentView stride{};
};
struct PieLoaderDestExtentView {
    std::uint32_t buffer_id = 0;
    std::uint64_t offset = 0;
    PieLoaderStridedExtentView stride{};
};
struct PieLoaderTensorDeclView {
    std::uint32_t id = 0;
    PieLoaderBytes name{};
    PieLoaderDType dtype = PieLoaderDType::BF16;
    PieLoaderEncodingKind encoding_kind = PieLoaderEncodingKind::Raw;
    PieLoaderQuantScheme quant_scheme = PieLoaderQuantScheme::None;
    std::uint8_t quant_bits_per_element = 0;
    std::uint32_t quant_group_size = 0;
    PieLoaderI64Slice shape{};
    std::uint32_t alignment = 1;
};
struct PieLoaderSourceTensorView {
    std::uint32_t id = 0;
    PieLoaderBytes name{};
    std::uint32_t file_id = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t span_bytes = 0;
    PieLoaderDType dtype = PieLoaderDType::BF16;
    PieLoaderEncodingKind encoding_kind = PieLoaderEncodingKind::Raw;
    PieLoaderQuantScheme quant_scheme = PieLoaderQuantScheme::None;
    std::uint8_t quant_bits_per_element = 0;
    std::uint32_t quant_group_size = 0;
    PieLoaderI64Slice shape{};
};
struct PieLoaderSourceTensorSlice {
    const PieLoaderSourceTensorView* ptr = nullptr;
    std::size_t len = 0;
};
struct PieLoaderTensorDeclSlice {
    const PieLoaderTensorDeclView* ptr = nullptr;
    std::size_t len = 0;
};
struct PieLoaderBufferDeclView {
    std::uint32_t id = 0;
    std::uint32_t tensor_id = std::numeric_limits<std::uint32_t>::max();
    bool has_tensor = false;
    std::uint64_t bytes = 0;
    std::uint32_t alignment = 1;
    bool temporary = false;
    bool has_persistent_offset = false;
    std::uint64_t persistent_offset = 0;
};
struct PieLoaderBufferDeclSlice {
    const PieLoaderBufferDeclView* ptr = nullptr;
    std::size_t len = 0;
};
struct PieLoaderSlabPlacementView {
    std::uint64_t src_offset = 0;
    std::uint64_t dest_offset = 0;
    std::uint64_t bytes = 0;
};
struct PieLoaderSlabPlacementSlice {
    const PieLoaderSlabPlacementView* ptr = nullptr;
    std::size_t len = 0;
};
struct PieLoaderStorageInstrView {
    std::uint32_t id = 0;
    PieLoaderStorageInstrKind kind = PieLoaderStorageInstrKind::Allocate;
    std::uint32_t buffer_id = std::numeric_limits<std::uint32_t>::max();
    PieLoaderSourceExtentView source{};
    bool has_source = false;
    PieLoaderDestExtentView dest{};
    bool has_dest = false;
    PieLoaderBufferIdSlice input_buffers{};
    PieLoaderBufferIdSlice output_buffers{};
    PieLoaderTileMapKind tile_kind = PieLoaderTileMapKind::None;
    std::uint64_t max_tile_bytes = 0;
    PieLoaderQuantScheme transform_from = PieLoaderQuantScheme::None;
    PieLoaderQuantScheme transform_to = PieLoaderQuantScheme::None;
    PieLoaderRepackLayout repack_layout = PieLoaderRepackLayout::None;
    PieLoaderRowMap row_map = PieLoaderRowMap::Identity;
    std::uint32_t transform_batch = 0;
    std::uint32_t transform_source_rows = 0;
    std::uint32_t transform_source_row_offset = 0;
    std::uint32_t transform_target_rows = 0;
    std::uint32_t transform_valid_rows = 0;
    std::uint32_t transform_source_stride_cols = 0;
    std::uint32_t transform_source_col_offset = 0;
    std::uint32_t transform_source_cols = 0;
    std::uint32_t transform_target_cols = 0;
    std::uint64_t transform_scratch_bytes = 0;
    PieLoaderBytes name{};
    std::uint32_t slab_file_id = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t slab_file_offset = 0;
    std::uint64_t slab_span_bytes = 0;
    PieLoaderSlabPlacementSlice slab_placements{};
};
struct PieLoaderStorageInstrSlice {
    const PieLoaderStorageInstrView* ptr = nullptr;
    std::size_t len = 0;
};
struct PieLoaderMemoryPlanView {
    std::uint64_t persistent_bytes = 0;
    std::uint64_t temporary_peak_bytes = 0;
    std::uint64_t transform_scratch_peak_bytes = 0;
    std::uint64_t checkpoint_read_bytes = 0;
    std::uint64_t device_write_bytes = 0;
};
struct PieLoaderOptimizerPassStatsView {
    PieLoaderBytes name{};
    std::uint64_t exprs_before = 0;
    std::uint64_t exprs_after = 0;
    std::uint64_t rewrites = 0;
};
struct PieLoaderOptimizerPassStatsSlice {
    const PieLoaderOptimizerPassStatsView* ptr = nullptr;
    std::size_t len = 0;
};
struct PieLoaderOptimizerReportView {
    PieLoaderOptimizerPassStatsSlice passes{};
};
struct LoadPlanView {
    std::uint32_t version = 0;
    PieLoaderSourceTensorSlice sources{};
    PieLoaderTensorDeclSlice tensors{};
    PieLoaderBufferDeclSlice buffers{};
    PieLoaderStorageInstrSlice instrs{};
    PieLoaderU32Slice schedule{};
    PieLoaderMemoryPlanView memory{};
    PieLoaderOptimizerReportView optimizer{};
};

namespace detail {

inline PieLoaderBytes bytes(const std::string& value) {
    return {
        reinterpret_cast<const std::uint8_t*>(value.data()),
        value.size(),
    };
}

template <typename T>
inline std::uint32_t id(const T& value, const char* field) {
    std::uint64_t narrowed = 0;
    if (value.is_number_unsigned()) {
        narrowed = value.template get<std::uint64_t>();
    } else if (value.is_number_integer()) {
        const std::int64_t signed_value =
            value.template get<std::int64_t>();
        if (signed_value < 0) {
            throw std::runtime_error(
                std::string("load plan: ") + field +
                " must be non-negative");
        }
        narrowed = static_cast<std::uint64_t>(signed_value);
    } else {
        throw std::runtime_error(std::string("load plan: ") + field + " is not an id");
    }
    if (narrowed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            std::string("load plan: ") + field +
            " exceeds uint32 range");
    }
    return static_cast<std::uint32_t>(narrowed);
}

inline PieLoaderDType dtype(const std::string& value) {
    static const std::unordered_map<std::string, PieLoaderDType> map = {
        {"F32", PieLoaderDType::F32}, {"F16", PieLoaderDType::F16},
        {"BF16", PieLoaderDType::BF16}, {"F8E4M3", PieLoaderDType::F8E4M3},
        {"F8E5M2", PieLoaderDType::F8E5M2}, {"I32", PieLoaderDType::I32},
        {"I16", PieLoaderDType::I16}, {"I8", PieLoaderDType::I8},
        {"U32", PieLoaderDType::U32}, {"U16", PieLoaderDType::U16},
        {"U8", PieLoaderDType::U8}, {"Bool", PieLoaderDType::Bool},
    };
    const auto it = map.find(value);
    if (it == map.end()) throw std::runtime_error("load plan: unknown dtype " + value);
    return it->second;
}

inline PieLoaderQuantScheme quant(const std::string& value) {
    static const std::unordered_map<std::string, PieLoaderQuantScheme> map = {
        {"None", PieLoaderQuantScheme::None},
        {"Fp8E4M3", PieLoaderQuantScheme::Fp8E4M3},
        {"Fp8E5M2", PieLoaderQuantScheme::Fp8E5M2},
        {"Int8Symmetric", PieLoaderQuantScheme::Int8Symmetric},
        {"Int8Asymmetric", PieLoaderQuantScheme::Int8Asymmetric},
        {"AwqInt4", PieLoaderQuantScheme::AwqInt4},
        {"GptqInt4", PieLoaderQuantScheme::GptqInt4},
        {"Mxfp4E2M1E8M0", PieLoaderQuantScheme::Mxfp4E2M1E8M0},
        {"GgufQ4_0", PieLoaderQuantScheme::GgufQ4_0},
        {"GgufQ4K", PieLoaderQuantScheme::GgufQ4K},
        {"GgufQ5_0", PieLoaderQuantScheme::GgufQ5_0},
        {"GgufQ5K", PieLoaderQuantScheme::GgufQ5K},
        {"GgufQ8_0", PieLoaderQuantScheme::GgufQ8_0},
        {"MlxAffineU4", PieLoaderQuantScheme::MlxAffineU4},
    };
    const auto it = map.find(value);
    if (it == map.end()) {
        throw std::runtime_error("load plan: unknown quant scheme " + value);
    }
    return it->second;
}

inline PieLoaderTileMapKind tile_kind(const std::string& value) {
    static const std::unordered_map<std::string, PieLoaderTileMapKind> map = {
        {"Cast", PieLoaderTileMapKind::Cast}, {"Decode", PieLoaderTileMapKind::Decode},
        {"Encode", PieLoaderTileMapKind::Encode},
        {"Transcode", PieLoaderTileMapKind::Transcode},
        {"Reblock", PieLoaderTileMapKind::Reblock},
        {"Reorder", PieLoaderTileMapKind::Reorder},
        {"Repack", PieLoaderTileMapKind::Repack},
    };
    const auto it = map.find(value);
    if (it == map.end()) throw std::runtime_error("load plan: unknown TileMap " + value);
    return it->second;
}

inline PieLoaderRepackLayout repack(const std::string& value) {
    if (value == "None") return PieLoaderRepackLayout::None;
    if (value == "MarlinMxfp4Weight") return PieLoaderRepackLayout::MarlinMxfp4Weight;
    if (value == "MarlinMxfp4Scale") return PieLoaderRepackLayout::MarlinMxfp4Scale;
    if (value == "DenseRowGather") return PieLoaderRepackLayout::DenseRowGather;
    throw std::runtime_error("load plan: unknown repack layout " + value);
}

inline PieLoaderRowMap row_map(const std::string& value) {
    if (value == "Identity") return PieLoaderRowMap::Identity;
    if (value == "Even") return PieLoaderRowMap::Even;
    if (value == "Odd") return PieLoaderRowMap::Odd;
    throw std::runtime_error("load plan: unknown row map " + value);
}

}  // namespace detail

class LoadPlan {
  public:
    LoadPlan() = default;
    LoadPlan(const LoadPlan&) = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;
    LoadPlan(LoadPlan&&) noexcept = default;
    LoadPlan& operator=(LoadPlan&&) noexcept = default;

    static LoadPlan deserialize(
        std::span<const std::uint8_t> bytes,
        std::uint64_t expected_compiler_version = 0) {
        if (bytes.empty()) throw std::runtime_error("load plan is empty");
        const nlohmann::json root = nlohmann::json::parse(bytes.begin(), bytes.end());
        LoadPlan plan;
        plan.parse(root, expected_compiler_version);
        return plan;
    }

    LoadPlanView view() const noexcept {
        return {
            version_,
            {sources_.data(), sources_.size()},
            {tensors_.data(), tensors_.size()},
            {buffers_.data(), buffers_.size()},
            {instrs_.data(), instrs_.size()},
            {schedule_.data(), schedule_.size()},
            memory_,
            {{optimizer_.data(), optimizer_.size()}},
        };
    }

    PieLoaderBackendKind backend() const noexcept { return backend_; }
    PieLoaderMxfp4MoePolicy mxfp4_moe() const noexcept { return mxfp4_moe_; }
    bool native_mxfp4_moe() const noexcept { return native_mxfp4_moe_; }
    std::uint32_t preferred_alignment() const noexcept {
        return preferred_alignment_;
    }
    std::uint64_t max_tile_bytes() const noexcept { return max_tile_bytes_; }
    std::uint32_t tile_map_mask() const noexcept { return tile_map_mask_; }
    std::uint64_t compiler_version() const noexcept { return compiler_version_; }

  private:
    PieLoaderBytes store_string(std::string value) {
        strings_.push_back(std::move(value));
        return detail::bytes(strings_.back());
    }

    PieLoaderI64Slice store_i64(const nlohmann::json& values) {
        i64_slices_.emplace_back();
        auto& out = i64_slices_.back();
        out.reserve(values.size());
        for (const auto& value : values) out.push_back(value.get<std::int64_t>());
        return {out.data(), out.size()};
    }

    PieLoaderU32Slice store_ids(const nlohmann::json& values) {
        u32_slices_.emplace_back();
        auto& out = u32_slices_.back();
        out.reserve(values.size());
        for (const auto& value : values) out.push_back(detail::id(value, "buffer id"));
        return {out.data(), out.size()};
    }

    PieLoaderStridedExtentView extent(const nlohmann::json& value) {
        dim_slices_.emplace_back();
        auto& dims = dim_slices_.back();
        for (const auto& dim : value.at("dims")) {
            dims.push_back({
                dim.at("count").get<std::int64_t>(),
                dim.at("src_stride").get<std::int64_t>(),
                dim.at("dst_stride").get<std::int64_t>(),
            });
        }
        return {
            value.at("base_offset").get<std::uint64_t>(),
            value.at("element_bytes").get<std::uint32_t>(),
            {dims.data(), dims.size()},
        };
    }

    PieLoaderSourceExtentView source(const nlohmann::json& value) {
        return {
            detail::id(value.at("file_id"), "source.file_id"),
            detail::id(value.at("tensor_id"), "source.tensor_id"),
            value.at("file_offset").get<std::uint64_t>(),
            value.at("span_bytes").get<std::uint64_t>(),
            extent(value.at("stride")),
        };
    }

    PieLoaderDestExtentView dest(const nlohmann::json& value) {
        return {
            detail::id(value.at("buffer"), "dest.buffer"),
            value.at("offset").get<std::uint64_t>(),
            extent(value.at("stride")),
        };
    }

    static std::pair<PieLoaderEncodingKind, PieLoaderDType>
    encoding(const nlohmann::json& value, PieLoaderQuantScheme& scheme) {
        if (value.contains("Raw")) {
            scheme = PieLoaderQuantScheme::None;
            return {
                PieLoaderEncodingKind::Raw,
                detail::dtype(value.at("Raw").get<std::string>()),
            };
        }
        const auto& quant = value.at("Quant");
        scheme = detail::quant(quant.at("scheme").get<std::string>());
        return {
            PieLoaderEncodingKind::Quant,
            detail::dtype(quant.at("logical_dtype").get<std::string>()),
        };
    }

    PieLoaderStorageInstrView instruction(const nlohmann::json& tagged) {
        if (tagged.size() != 1) {
            throw std::runtime_error("load plan: instruction must have one tag");
        }
        const std::string tag = tagged.begin().key();
        const auto& value = tagged.begin().value();
        PieLoaderStorageInstrView out;
        out.id = detail::id(value.at("id"), "instruction.id");
        if (tag == "Allocate") {
            out.kind = PieLoaderStorageInstrKind::Allocate;
            out.buffer_id = detail::id(value.at("buffer"), "allocate.buffer");
        } else if (tag == "ExtentWrite") {
            out.kind = PieLoaderStorageInstrKind::ExtentWrite;
            out.source = source(value.at("source"));
            out.has_source = true;
            out.dest = dest(value.at("dest"));
            out.has_dest = true;
            out.buffer_id = out.dest.buffer_id;
        } else if (tag == "BulkExtentWrite") {
            out.kind = PieLoaderStorageInstrKind::BulkExtentWrite;
            out.source = source(value.at("source"));
            out.has_source = true;
            const std::uint64_t offset = value.at("dest_offset").get<std::uint64_t>();
            dim_slices_.push_back({{
                static_cast<std::int64_t>(out.source.span_bytes), 1, 1,
            }});
            const auto& dims = dim_slices_.back();
            out.dest = {
                std::numeric_limits<std::uint32_t>::max(),
                offset,
                {0, 1, {dims.data(), dims.size()}},
            };
            out.has_dest = true;
        } else if (tag == "SlabScatter") {
            out.kind = PieLoaderStorageInstrKind::SlabScatter;
            out.slab_file_id = detail::id(value.at("file_id"), "slab.file_id");
            out.slab_file_offset = value.at("file_offset").get<std::uint64_t>();
            out.slab_span_bytes = value.at("span_bytes").get<std::uint64_t>();
            slab_slices_.emplace_back();
            auto& placements = slab_slices_.back();
            for (const auto& placement : value.at("placements")) {
                placements.push_back({
                    placement.at("src_offset").get<std::uint64_t>(),
                    placement.at("dest_offset").get<std::uint64_t>(),
                    placement.at("bytes").get<std::uint64_t>(),
                });
            }
            out.slab_placements = {placements.data(), placements.size()};
        } else if (tag == "TileMap") {
            out.kind = PieLoaderStorageInstrKind::TileMap;
            out.tile_kind = detail::tile_kind(value.at("kind").get<std::string>());
            if (!value.at("source").is_null()) {
                out.source = source(value.at("source"));
                out.has_source = true;
            }
            if (!value.at("dest").is_null()) {
                out.dest = dest(value.at("dest"));
                out.has_dest = true;
            }
            out.input_buffers = store_ids(value.at("inputs"));
            out.output_buffers = store_ids(value.at("outputs"));
            if (out.output_buffers.len != 0) out.buffer_id = out.output_buffers.ptr[0];
            out.max_tile_bytes = value.at("tile").at("max_tile_bytes").get<std::uint64_t>();
            const auto& transform = value.at("transform");
            if (!transform.at("from").is_null()) {
                out.transform_from =
                    detail::quant(transform.at("from").get<std::string>());
            }
            if (!transform.at("to").is_null()) {
                out.transform_to =
                    detail::quant(transform.at("to").get<std::string>());
            }
            const auto& repack = transform.at("repack");
            out.repack_layout =
                detail::repack(repack.at("layout").get<std::string>());
            out.row_map = detail::row_map(repack.at("row_map").get<std::string>());
            out.transform_batch = repack.at("batch").get<std::uint32_t>();
            out.transform_source_rows = repack.at("source_rows").get<std::uint32_t>();
            out.transform_source_row_offset =
                repack.at("source_row_offset").get<std::uint32_t>();
            out.transform_target_rows = repack.at("target_rows").get<std::uint32_t>();
            out.transform_valid_rows = repack.at("valid_rows").get<std::uint32_t>();
            out.transform_source_stride_cols =
                repack.at("source_stride_cols").get<std::uint32_t>();
            out.transform_source_col_offset =
                repack.at("source_col_offset").get<std::uint32_t>();
            out.transform_source_cols = repack.at("source_cols").get<std::uint32_t>();
            out.transform_target_cols = repack.at("target_cols").get<std::uint32_t>();
            out.transform_scratch_bytes =
                transform.at("scratch_bytes").get<std::uint64_t>();
        } else if (tag == "CreateView") {
            out.kind = PieLoaderStorageInstrKind::CreateView;
            const std::uint32_t input = detail::id(value.at("input"), "view.input");
            const std::uint32_t output = detail::id(value.at("output"), "view.output");
            u32_slices_.push_back({input});
            out.input_buffers = {u32_slices_.back().data(), 1};
            u32_slices_.push_back({output});
            out.output_buffers = {u32_slices_.back().data(), 1};
            out.buffer_id = output;
            out.dest = dest(value.at("view"));
            out.has_dest = true;
        } else if (tag == "Attach") {
            out.kind = PieLoaderStorageInstrKind::Attach;
            out.buffer_id = detail::id(value.at("tensor"), "attach.tensor");
            out.input_buffers = store_ids(value.at("metadata"));
            u32_slices_.push_back({out.buffer_id});
            out.output_buffers = {u32_slices_.back().data(), 1};
        } else if (tag == "Release") {
            out.kind = PieLoaderStorageInstrKind::Release;
            out.buffer_id = detail::id(value.at("buffer"), "release.buffer");
        } else if (tag == "Finalize") {
            out.kind = PieLoaderStorageInstrKind::Finalize;
            out.buffer_id = detail::id(value.at("tensor"), "finalize.tensor");
            u32_slices_.push_back({out.buffer_id});
            out.output_buffers = {u32_slices_.back().data(), 1};
            out.name = store_string(value.at("name").get<std::string>());
        } else {
            throw std::runtime_error("load plan: unknown instruction " + tag);
        }
        return out;
    }

    void parse(
        const nlohmann::json& root,
        std::uint64_t expected_compiler_version) {
        version_ = root.at("version").get<std::uint32_t>();
        if (version_ != 5) {
            throw std::runtime_error(
                "load plan version " + std::to_string(version_) +
                " does not match executor version 5");
        }
        compiler_version_ = root.at("compiler_version").get<std::uint64_t>();
        if (expected_compiler_version != 0 &&
            compiler_version_ != expected_compiler_version) {
            throw std::runtime_error(
                "storage compiler version mismatch: program=" +
                std::to_string(compiler_version_) +
                " executor=" + std::to_string(expected_compiler_version));
        }

        const auto& target = root.at("target");
        const std::string backend = target.at("backend").get<std::string>();
        if (backend == "Cuda") {
            backend_ = PieLoaderBackendKind::Cuda;
        } else if (backend == "Metal") {
            backend_ = PieLoaderBackendKind::Metal;
        } else if (backend == "Unknown") {
            backend_ = PieLoaderBackendKind::Unknown;
        } else {
            throw std::runtime_error(
                "load plan: unknown backend " + backend);
        }
        const std::string policy = target.at("mxfp4_moe").get<std::string>();
        if (policy == "NativeGemm") {
            mxfp4_moe_ = PieLoaderMxfp4MoePolicy::NativeGemm;
        } else if (policy == "EagerBf16") {
            mxfp4_moe_ = PieLoaderMxfp4MoePolicy::EagerBf16;
        } else if (policy == "RoutedDecode") {
            mxfp4_moe_ = PieLoaderMxfp4MoePolicy::RoutedDecode;
        } else {
            throw std::runtime_error(
                "load plan: unknown mxfp4_moe policy " + policy);
        }
        native_mxfp4_moe_ = target.at("native_mxfp4_moe").get<bool>();
        preferred_alignment_ =
            target.at("preferred_alignment").get<std::uint32_t>();
        max_tile_bytes_ = target.at("max_tile_bytes").get<std::uint64_t>();
        tile_map_mask_ = target.at("tile_map_mask").get<std::uint32_t>();

        const auto& tensor_json = root.at("tensors");
        tensors_.reserve(tensor_json.size());
        for (const auto& tensor : tensor_json) {
            PieLoaderTensorDeclView view;
            view.id = detail::id(tensor.at("id"), "tensor.id");
            view.name = store_string(tensor.at("name").get<std::string>());
            auto [kind, dtype] =
                encoding(tensor.at("encoding"), view.quant_scheme);
            view.encoding_kind = kind;
            view.dtype = dtype;
            if (kind == PieLoaderEncodingKind::Quant) {
                const auto& quant = tensor.at("encoding").at("Quant");
                view.quant_bits_per_element =
                    quant.at("bits_per_element").get<std::uint8_t>();
                view.quant_group_size =
                    quant.at("group_size").get<std::uint32_t>();
            }
            view.shape = store_i64(tensor.at("shape"));
            view.alignment = tensor.at("alignment").get<std::uint32_t>();
            tensors_.push_back(view);
        }

        const auto& source_json = root.at("sources");
        sources_.reserve(source_json.size());
        for (const auto& source : source_json) {
            PieLoaderSourceTensorView view;
            view.id = detail::id(source.at("id"), "source tensor.id");
            view.name = store_string(source.at("name").get<std::string>());
            view.file_id = detail::id(source.at("file_id"), "source tensor.file_id");
            view.file_offset = source.at("file_offset").get<std::uint64_t>();
            view.span_bytes = source.at("span_bytes").get<std::uint64_t>();
            auto [kind, dtype] =
                encoding(source.at("encoding"), view.quant_scheme);
            view.encoding_kind = kind;
            view.dtype = dtype;
            if (kind == PieLoaderEncodingKind::Quant) {
                const auto& quant = source.at("encoding").at("Quant");
                view.quant_bits_per_element =
                    quant.at("bits_per_element").get<std::uint8_t>();
                view.quant_group_size =
                    quant.at("group_size").get<std::uint32_t>();
            }
            view.shape = store_i64(source.at("shape"));
            sources_.push_back(view);
        }

        const auto& buffer_json = root.at("buffers");
        buffers_.reserve(buffer_json.size());
        for (const auto& buffer : buffer_json) {
            PieLoaderBufferDeclView view;
            view.id = detail::id(buffer.at("id"), "buffer.id");
            if (!buffer.at("tensor").is_null()) {
                view.has_tensor = true;
                view.tensor_id = detail::id(buffer.at("tensor"), "buffer.tensor");
            }
            view.bytes = buffer.at("bytes").get<std::uint64_t>();
            view.alignment = buffer.at("alignment").get<std::uint32_t>();
            view.temporary = buffer.at("temporary").get<bool>();
            if (!buffer.at("persistent_offset").is_null()) {
                view.has_persistent_offset = true;
                view.persistent_offset =
                    buffer.at("persistent_offset").get<std::uint64_t>();
            }
            buffers_.push_back(view);
        }

        const auto& instr_json = root.at("instrs");
        instrs_.reserve(instr_json.size());
        for (const auto& instr : instr_json) instrs_.push_back(instruction(instr));
        for (const auto& id : root.at("schedule")) {
            schedule_.push_back(detail::id(id, "schedule"));
        }

        const auto& memory = root.at("memory");
        memory_ = {
            memory.at("persistent_bytes").get<std::uint64_t>(),
            memory.at("temporary_peak_bytes").get<std::uint64_t>(),
            memory.at("transform_scratch_peak_bytes").get<std::uint64_t>(),
            memory.at("checkpoint_read_bytes").get<std::uint64_t>(),
            memory.at("device_write_bytes").get<std::uint64_t>(),
        };
        const auto& passes = root.at("optimizer").at("passes");
        optimizer_.reserve(passes.size());
        for (const auto& pass : passes) {
            optimizer_.push_back({
                store_string(pass.at("name").get<std::string>()),
                pass.at("exprs_before").get<std::uint64_t>(),
                pass.at("exprs_after").get<std::uint64_t>(),
                pass.at("rewrites").get<std::uint64_t>(),
            });
        }
    }

    std::uint32_t version_ = 0;
    std::uint64_t compiler_version_ = 0;
    PieLoaderBackendKind backend_ = PieLoaderBackendKind::Unknown;
    PieLoaderMxfp4MoePolicy mxfp4_moe_ =
        PieLoaderMxfp4MoePolicy::RoutedDecode;
    bool native_mxfp4_moe_ = false;
    std::uint32_t preferred_alignment_ = 1;
    std::uint64_t max_tile_bytes_ = 0;
    std::uint32_t tile_map_mask_ = 0;
    std::deque<std::string> strings_;
    std::deque<std::vector<std::int64_t>> i64_slices_;
    std::deque<std::vector<std::uint32_t>> u32_slices_;
    std::deque<std::vector<PieLoaderDimSpecView>> dim_slices_;
    std::deque<std::vector<PieLoaderSlabPlacementView>> slab_slices_;
    std::vector<PieLoaderTensorDeclView> tensors_;
    std::vector<PieLoaderSourceTensorView> sources_;
    std::vector<PieLoaderBufferDeclView> buffers_;
    std::vector<PieLoaderStorageInstrView> instrs_;
    std::vector<std::uint32_t> schedule_;
    PieLoaderMemoryPlanView memory_{};
    std::vector<PieLoaderOptimizerPassStatsView> optimizer_;
};

namespace cpp {

inline std::string bytes_to_string(PieLoaderBytes bytes) {
    if (bytes.ptr == nullptr || bytes.len == 0) return {};
    return std::string(
        reinterpret_cast<const char*>(bytes.ptr),
        reinterpret_cast<const char*>(bytes.ptr) + bytes.len);
}

inline std::vector<std::int64_t> i64_slice_to_vector(PieLoaderI64Slice shape) {
    if (shape.ptr == nullptr || shape.len == 0) return {};
    return {shape.ptr, shape.ptr + shape.len};
}

inline std::vector<std::uint32_t> buffer_id_slice_to_vector(
    PieLoaderBufferIdSlice ids) {
    if (ids.ptr == nullptr || ids.len == 0) return {};
    return {ids.ptr, ids.ptr + ids.len};
}

inline std::vector<std::int64_t> extent_shape(
    const PieLoaderStridedExtentView& extent) {
    std::vector<std::int64_t> shape;
    shape.reserve(extent.dims.len);
    for (std::size_t i = 0; i < extent.dims.len; ++i) {
        shape.push_back(extent.dims.ptr[i].count);
    }
    return shape;
}

inline bool compact_extent(const PieLoaderStridedExtentView& extent) {
    std::int64_t stride = static_cast<std::int64_t>(extent.element_bytes);
    for (std::size_t i = extent.dims.len; i > 0; --i) {
        const auto& dim = extent.dims.ptr[i - 1];
        if (dim.src_stride != stride || dim.dst_stride != stride) return false;
        stride *= dim.count;
    }
    return true;
}

inline std::uint64_t extent_bytes(
    const PieLoaderStridedExtentView& extent,
    const char* context) {
    std::uint64_t elements = 1;
    for (std::size_t i = 0; i < extent.dims.len; ++i) {
        const auto count = extent.dims.ptr[i].count;
        if (count < 0) {
            throw std::runtime_error(std::string(context) + ": negative extent");
        }
        const auto ucount = static_cast<std::uint64_t>(count);
        if (ucount != 0 &&
            elements > std::numeric_limits<std::uint64_t>::max() / ucount) {
            throw std::runtime_error(std::string(context) + ": extent overflow");
        }
        elements *= ucount;
    }
    if (extent.element_bytes != 0 &&
        elements > std::numeric_limits<std::uint64_t>::max() /
                       extent.element_bytes) {
        throw std::runtime_error(std::string(context) + ": extent overflow");
    }
    return elements * extent.element_bytes;
}

class LoadPlanIndex {
  public:
    explicit LoadPlanIndex(std::string context)
        : context_(std::move(context)) {}

    void reset(const LoadPlanView& plan) {
        instr_by_id_.clear();
        buffer_by_id_.clear();
        tensor_by_id_.clear();
        source_by_id_.clear();
        source_by_name_.clear();
        for (std::size_t i = 0; i < plan.instrs.len; ++i) {
            instr_by_id_.emplace(plan.instrs.ptr[i].id, &plan.instrs.ptr[i]);
        }
        for (std::size_t i = 0; i < plan.buffers.len; ++i) {
            buffer_by_id_.emplace(plan.buffers.ptr[i].id, &plan.buffers.ptr[i]);
        }
        for (std::size_t i = 0; i < plan.tensors.len; ++i) {
            tensor_by_id_.emplace(plan.tensors.ptr[i].id, &plan.tensors.ptr[i]);
        }
        for (std::size_t i = 0; i < plan.sources.len; ++i) {
            const auto* source = &plan.sources.ptr[i];
            source_by_id_.emplace(source->id, source);
            source_by_name_.emplace(bytes_to_string(source->name), source);
        }
    }

    const PieLoaderStorageInstrView& instruction(std::uint32_t id) const {
        const auto it = instr_by_id_.find(id);
        if (it == instr_by_id_.end()) {
            throw std::runtime_error(context_ + ": instruction id out of range");
        }
        return *it->second;
    }
    const PieLoaderBufferDeclView& buffer(std::uint32_t id) const {
        const auto it = buffer_by_id_.find(id);
        if (it == buffer_by_id_.end()) {
            throw std::runtime_error(context_ + ": buffer id out of range");
        }
        return *it->second;
    }
    const PieLoaderTensorDeclView& tensor(std::uint32_t id) const {
        const auto it = tensor_by_id_.find(id);
        if (it == tensor_by_id_.end()) {
            throw std::runtime_error(context_ + ": tensor id out of range");
        }
        return *it->second;
    }
    const PieLoaderSourceTensorView& source(std::uint32_t id) const {
        const auto it = source_by_id_.find(id);
        if (it == source_by_id_.end()) {
            throw std::runtime_error(context_ + ": source tensor id out of range");
        }
        return *it->second;
    }
    const PieLoaderSourceTensorView* find_source(const std::string& name) const {
        const auto it = source_by_name_.find(name);
        return it == source_by_name_.end() ? nullptr : it->second;
    }

  private:
    std::string context_;
    std::unordered_map<std::uint32_t, const PieLoaderStorageInstrView*> instr_by_id_;
    std::unordered_map<std::uint32_t, const PieLoaderBufferDeclView*> buffer_by_id_;
    std::unordered_map<std::uint32_t, const PieLoaderTensorDeclView*> tensor_by_id_;
    std::unordered_map<std::uint32_t, const PieLoaderSourceTensorView*> source_by_id_;
    std::unordered_map<std::string, const PieLoaderSourceTensorView*> source_by_name_;
};

}  // namespace cpp
}  // namespace pie_load_planner

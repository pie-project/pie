#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "loader/rust_quant_attachment.hpp"
#include "loader/rust_storage_program.hpp"

#include "model/config.hpp"

namespace pie_cuda_driver {

inline std::string rust_loader_bytes_to_string(
    pie_weight_loader::PieLoaderBytes bytes) {
    return pie_weight_loader::cpp::bytes_to_string(bytes);
}

struct RustLoaderCompileResult {
    RustStorageProgram program;
    std::vector<RustQuantAttachment> quant_attachments;
    std::size_t source_tensor_count = 0;
    std::size_t covered_contract_count = 0;
    std::size_t runtime_tensor_count = 0;
    std::string cache_key;
};

inline bool rust_loader_ends_with(
    const std::string& value,
    const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::vector<RustQuantAttachment> infer_rust_quant_attachments(
    const HfConfig& hf,
    const pie_weight_loader::PieLoaderStorageProgramView& view) {
    (void)hf;
    std::unordered_set<std::string> present;
    present.reserve(view.tensors.len);
    for (std::size_t i = 0; i < view.tensors.len; ++i) {
        present.insert(rust_loader_bytes_to_string(view.tensors.ptr[i].name));
    }

    std::vector<RustQuantAttachment> attachments;
    for (std::size_t i = 0; i < view.tensors.len; ++i) {
        const auto& tensor = view.tensors.ptr[i];
        const std::string name = rust_loader_bytes_to_string(tensor.name);
        auto attach = [&](const std::string& scale,
                          QuantGranularity granularity,
                          int group_size,
                          int channel_axis) {
            if (!present.contains(scale)) return false;
            attachments.push_back({
                .tensor_name = name,
                .scale_tensor_name = scale,
                .granularity = granularity,
                .group_size = group_size,
                .channel_axis = channel_axis,
            });
            return true;
        };

        if (tensor.encoding_kind == pie_weight_loader::PieLoaderEncodingKind::Quant) {
            if (tensor.quant_scheme ==
                    pie_weight_loader::PieLoaderQuantScheme::Fp8E4M3 ||
                tensor.quant_scheme ==
                    pie_weight_loader::PieLoaderQuantScheme::Int8Symmetric) {
                attach(name + "_scale_inv", QuantGranularity::PerChannel, 0, 0);
            } else if (tensor.quant_scheme ==
                       pie_weight_loader::PieLoaderQuantScheme::Mxfp4E2M1E8M0) {
                if (!attach(name + "_scale", QuantGranularity::PerGroup, 32, 1) &&
                    rust_loader_ends_with(name, ".weight")) {
                    attach(
                        name.substr(0, name.size() - 7) + ".weight_scale",
                        QuantGranularity::PerGroup,
                        32,
                        1);
                }
            }
            continue;
        }
        if (tensor.dtype != pie_weight_loader::PieLoaderDType::F8E4M3 ||
            !rust_loader_ends_with(name, ".weight")) {
            continue;
        }
        if (attach(name + "_scale_inv", QuantGranularity::PerGroup, 128, 0)) {
            continue;
        }
        attach(
            name.substr(0, name.size() - 7) + ".scale",
            QuantGranularity::PerGroup,
            128,
            0);
    }
    return attachments;
}

inline std::string storage_program_cache_key(std::span<const std::uint8_t> bytes) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const std::uint8_t byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

inline RustLoaderCompileResult load_rust_storage_program(
    const HfConfig& hf,
    RustStorageProgram program,
    std::span<const std::uint8_t> program_bytes,
    std::uint64_t compiler_version) {
    if (program.compiler_version() != compiler_version) {
        throw std::runtime_error("engine: StorageProgram compiler version mismatch");
    }
    if (program.backend() != pie_weight_loader::PieLoaderBackendKind::Cuda) {
        throw std::runtime_error("engine: CUDA received a non-CUDA StorageProgram");
    }
    const auto view = program.view();
    const std::size_t runtime_tensor_count = view.tensors.len;
    return {
        .program = std::move(program),
        .quant_attachments = infer_rust_quant_attachments(hf, view),
        .source_tensor_count = view.sources.len,
        .covered_contract_count = runtime_tensor_count,
        .runtime_tensor_count = runtime_tensor_count,
        .cache_key = storage_program_cache_key(program_bytes),
    };
}

inline std::string describe_rust_storage_program(
    const pie_weight_loader::PieLoaderStorageProgramView& view,
    std::size_t source_tensor_count,
    std::size_t covered_contract_count,
    std::size_t runtime_tensor_count) {
    std::uint64_t optimizer_rewrites = 0;
    for (std::size_t i = 0; i < view.optimizer.passes.len; ++i) {
        optimizer_rewrites += view.optimizer.passes.ptr[i].rewrites;
    }
    std::ostringstream out;
    out << "storage_program(version=" << view.version
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
    pie_weight_loader::PieLoaderStorageInstrKind kind) noexcept {
    using K = pie_weight_loader::PieLoaderStorageInstrKind;
    switch (kind) {
    case K::Allocate: return "Allocate";
    case K::ExtentWrite: return "ExtentWrite";
    case K::TileMap: return "TileMap";
    case K::CreateView: return "CreateView";
    case K::Attach: return "Attach";
    case K::Release: return "Release";
    case K::Finalize: return "Finalize";
    case K::BulkExtentWrite: return "BulkExtentWrite";
    case K::SlabScatter: return "SlabScatter";
    }
    return "Unknown";
}

inline const char* rust_tile_map_kind_name(
    pie_weight_loader::PieLoaderTileMapKind kind) noexcept {
    using K = pie_weight_loader::PieLoaderTileMapKind;
    switch (kind) {
    case K::Cast: return "Cast";
    case K::Decode: return "Decode";
    case K::Encode: return "Encode";
    case K::Transcode: return "Transcode";
    case K::Reblock: return "Reblock";
    case K::Reorder: return "Reorder";
    case K::Repack: return "Repack";
    case K::None: return "None";
    }
    return "Unknown";
}

inline std::string dump_rust_storage_program_json(
    const pie_weight_loader::PieLoaderStorageProgramView& view,
    std::size_t source_tensor_count,
    std::size_t covered_contract_count,
    std::size_t runtime_tensor_count) {
    std::map<std::string, std::size_t> instruction_kinds;
    std::map<std::string, std::size_t> tile_map_kinds;
    for (std::size_t i = 0; i < view.instrs.len; ++i) {
        const auto& instr = view.instrs.ptr[i];
        ++instruction_kinds[rust_storage_instr_kind_name(instr.kind)];
        if (instr.kind == pie_weight_loader::PieLoaderStorageInstrKind::TileMap) {
            ++tile_map_kinds[rust_tile_map_kind_name(instr.tile_kind)];
        }
    }
    nlohmann::json out = {
        {"summary", describe_rust_storage_program(
            view, source_tensor_count, covered_contract_count,
            runtime_tensor_count)},
        {"version", view.version},
        {"source_tensor_count", source_tensor_count},
        {"covered_contract_count", covered_contract_count},
        {"runtime_tensor_count", runtime_tensor_count},
        {"tensor_count", view.tensors.len},
        {"buffer_count", view.buffers.len},
        {"instruction_count", view.instrs.len},
        {"schedule_count", view.schedule.len},
        {"instruction_kinds", instruction_kinds},
        {"tile_map_kinds", tile_map_kinds},
    };
    return out.dump(2);
}

}  // namespace pie_cuda_driver

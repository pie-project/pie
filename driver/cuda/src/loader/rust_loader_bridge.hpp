#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
};

struct RustLoaderSourceIndex {
    std::unordered_map<std::string, std::uint32_t> tensor_ids;
    std::vector<std::string> tensor_names;
};

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

    const bool is_gpt_oss =
        hf.model_type == "gpt_oss" || hf.model_type == "gpt-oss" ||
        hf.model_type == "gptoss";
    if (!is_gpt_oss) {
        return attachments;
    }
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

inline RustLoaderCompileResult compile_rust_loader_plan_from_metadata(
    const HfConfig& hf,
    const SafetensorsCheckpointSource& loader,
    const std::string& runtime_quant,
    int tp_rank,
    int tp_size,
    std::uint64_t max_tile_bytes,
    std::uint32_t preferred_alignment,
    const BackendTarget& backend_target)
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
    RustStorageProgram program = compile_rust_storage_program(input.view());
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

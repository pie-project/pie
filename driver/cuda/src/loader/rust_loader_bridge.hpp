#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "loader/layout_plan.hpp"
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
    std::size_t direct_contract_count = 0;
    std::size_t runtime_tensor_count = 0;
};

inline bool rust_loader_contract_is_direct(
    const TensorDecl& spec,
    const TensorInfo& info)
{
    const bool dtype_supported =
        spec.dtype == info.dtype ||
        (info.dtype == DType::FP16 && spec.dtype == DType::BF16) ||
        (info.dtype == DType::FP32 && spec.dtype == DType::BF16) ||
        (info.dtype == DType::BF16 && spec.dtype == DType::FP32);
    return spec.shape == info.shape &&
           dtype_supported &&
           spec.ownership == TensorOwnershipKind::Owned &&
           (spec.layout == TensorLayoutKind::Dense ||
            spec.layout == TensorLayoutKind::Grouped ||
            spec.layout == TensorLayoutKind::QuantPacked);
}

inline std::uint64_t rust_loader_checked_mul(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const std::string& context)
{
    if (rhs != 0 && lhs > UINT64_MAX / rhs) {
        throw std::runtime_error(
            "rust loader compiler: byte-size overflow while lowering " +
            context);
    }
    return lhs * rhs;
}

inline std::uint64_t rust_loader_tensor_nbytes(
    const TensorInfo& info,
    const std::string& name)
{
    std::uint64_t elements = 1;
    for (const auto dim : info.shape) {
        if (dim < 0) {
            throw std::runtime_error(
                "rust loader compiler: negative dimension for '" + name + "'");
        }
        elements = rust_loader_checked_mul(
            elements,
            static_cast<std::uint64_t>(dim),
            name);
    }
    return rust_loader_checked_mul(
        elements,
        static_cast<std::uint64_t>(dtype_bytes(info.dtype)),
        name);
}

inline RustLoaderCompileResult compile_rust_loader_plan(
    const HfConfig& hf,
    const LayoutPlan& layout_plan,
    const SafetensorsCheckpointSource& loader,
    int tp_rank,
    int tp_size,
    std::uint64_t max_tile_bytes,
    std::uint32_t preferred_alignment)
{
    RustLoaderInputBuilder input;
    input.set_model(hf);
    input.set_target(
        tp_rank,
        tp_size,
        max_tile_bytes,
        preferred_alignment);
    input.set_runtime_abi_name("pie-cuda", /*version=*/1);

    std::unordered_map<std::string, std::uint32_t> tensor_ids;
    std::map<std::uint32_t, std::filesystem::path> files;
    std::map<std::uint32_t, std::uint64_t> file_sizes;

    auto names = loader.tensor_names();
    std::sort(names.begin(), names.end());
    std::vector<std::string> tensor_names;
    tensor_names.reserve(names.size());
    std::uint32_t next_tensor_id = 0;
    for (const auto& name : names) {
        const TensorInfo& info = loader.info(name);
        const TensorStorageInfo storage = loader.storage_info(name);
        tensor_ids.emplace(name, next_tensor_id);
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
        tensor_names.push_back(name);
        ++next_tensor_id;
    }
    for (const auto& [file_id, path] : files) {
        input.add_file(
            file_id,
            path.string(),
            file_sizes[file_id],
            pie_weight_loader::PieLoaderCheckpointFormat::Safetensors);
    }

    std::unordered_map<LayoutExprId, std::uint32_t> contract_for_expr;
    std::size_t emitted_contracts = 0;
    for (const auto& binding : layout_plan.algebra.bindings) {
        if (binding.root >= layout_plan.algebra.exprs.size()) continue;
        const LayoutExpr& root = layout_plan.algebra.exprs[binding.root];
        const auto spec_it = layout_plan.tensors.find(binding.runtime_name);
        if (spec_it == layout_plan.tensors.end()) continue;
        const TensorDecl& spec = spec_it->second;

        auto emit_direct_source = [&](const LayoutExpr& source_expr) -> bool {
            if (source_expr.kind != LayoutExprKind::Source ||
                !loader.contains(source_expr.raw_name)) {
                return false;
            }
            const TensorInfo& info = loader.info(source_expr.raw_name);
            if (!rust_loader_contract_is_direct(spec, info)) return false;
            const auto tensor_it = tensor_ids.find(source_expr.raw_name);
            if (tensor_it == tensor_ids.end()) return false;
            input.add_direct_contract(
                binding.runtime_name,
                tensor_it->second,
                spec.dtype,
                spec.shape,
                preferred_alignment,
                /*shard_axis=*/-1);
            return true;
        };

        auto emit_join = [&](LayoutExprId join_id, const LayoutExpr& join_expr) -> bool {
            if (join_expr.kind != LayoutExprKind::Join || join_expr.axis < 0) {
                return false;
            }
            std::vector<std::uint32_t> source_ids;
            source_ids.reserve(join_expr.inputs.size());
            for (const auto source_expr_id : join_expr.inputs) {
                if (source_expr_id >= layout_plan.algebra.exprs.size()) return false;
                const LayoutExpr& source_expr =
                    layout_plan.algebra.exprs[source_expr_id];
                if (source_expr.kind != LayoutExprKind::Source ||
                    !loader.contains(source_expr.raw_name)) {
                    return false;
                }
                const auto tensor_it = tensor_ids.find(source_expr.raw_name);
                if (tensor_it == tensor_ids.end()) return false;
                source_ids.push_back(tensor_it->second);
            }
            if (source_ids.empty()) return false;
            input.add_join_contract(
                binding.runtime_name,
                std::move(source_ids),
                join_expr.axis,
                spec.dtype,
                spec.shape,
                preferred_alignment);
            contract_for_expr[join_id] =
                static_cast<std::uint32_t>(emitted_contracts);
            return true;
        };

        auto emit_expert_stack_byte_spans =
            [&](const LayoutExpr& stack_expr) -> bool {
                if (stack_expr.kind != LayoutExprKind::Stack ||
                    stack_expr.inputs.empty() ||
                    stack_expr.inputs.size() % 3 != 0) {
                    return false;
                }
                const bool wants_gate_up =
                    binding.runtime_name == stack_expr.runtime_name;
                const bool wants_down =
                    binding.runtime_name == stack_expr.secondary_runtime_name;
                if (!wants_gate_up && !wants_down) return false;
                if (spec.dtype != DType::BF16 || spec.shape.size() != 3) {
                    return false;
                }

                const std::size_t expert_count =
                    stack_expr.inputs.size() / 3;
                if (static_cast<std::uint64_t>(spec.shape[0]) !=
                    expert_count) {
                    return false;
                }

                std::vector<pie_weight_loader::PieLoaderRuntimeByteSpanView>
                    spans;
                spans.reserve(wants_gate_up ? expert_count * 2 : expert_count);
                for (std::size_t expert = 0; expert < expert_count;
                     ++expert) {
                    const std::size_t base = expert * 3;
                    const std::size_t first = wants_gate_up ? base : base + 2;
                    const std::size_t count = wants_gate_up ? 2 : 1;
                    std::int64_t row_offset = 0;
                    for (std::size_t j = 0; j < count; ++j) {
                        const LayoutExprId source_expr_id =
                            stack_expr.inputs[first + j];
                        if (source_expr_id >=
                            layout_plan.algebra.exprs.size()) {
                            return false;
                        }
                        const LayoutExpr& source_expr =
                            layout_plan.algebra.exprs[source_expr_id];
                        if (source_expr.kind != LayoutExprKind::Source ||
                            !loader.contains(source_expr.raw_name)) {
                            return false;
                        }
                        const TensorInfo& info =
                            loader.info(source_expr.raw_name);
                        if (info.dtype != spec.dtype ||
                            info.shape.size() != 2 ||
                            info.shape[1] != spec.shape[2]) {
                            return false;
                        }
                        if (row_offset + info.shape[0] > spec.shape[1]) {
                            return false;
                        }
                        const auto tensor_it =
                            tensor_ids.find(source_expr.raw_name);
                        if (tensor_it == tensor_ids.end()) return false;
                        const std::uint64_t rows_before =
                            static_cast<std::uint64_t>(expert) *
                                static_cast<std::uint64_t>(spec.shape[1]) +
                            static_cast<std::uint64_t>(row_offset);
                        const std::uint64_t dest_offset =
                            rust_loader_checked_mul(
                                rust_loader_checked_mul(
                                    rows_before,
                                    static_cast<std::uint64_t>(spec.shape[2]),
                                    binding.runtime_name),
                                static_cast<std::uint64_t>(
                                    dtype_bytes(spec.dtype)),
                                binding.runtime_name);
                        spans.push_back(
                            pie_weight_loader::
                                PieLoaderRuntimeByteSpanView{
                                    .source_tensor_id = tensor_it->second,
                                    .source_offset_bytes = 0,
                                    .dest_offset_bytes = dest_offset,
                                    .span_bytes =
                                        rust_loader_tensor_nbytes(
                                            info,
                                            source_expr.raw_name),
                                });
                        row_offset += info.shape[0];
                    }
                    if (row_offset != spec.shape[1]) return false;
                }
                input.add_byte_span_contract(
                    binding.runtime_name,
                    std::move(spans),
                    spec.dtype,
                    spec.shape,
                    preferred_alignment);
                return true;
            };

        bool emitted = false;
        if (root.kind == LayoutExprKind::Realize && !root.inputs.empty()) {
            const LayoutExprId input_expr_id = root.inputs.front();
            if (input_expr_id < layout_plan.algebra.exprs.size()) {
                const LayoutExpr& input_expr =
                    layout_plan.algebra.exprs[input_expr_id];
                emitted = emit_direct_source(input_expr);
                if (!emitted) {
                    emitted = emit_join(input_expr_id, input_expr);
                }
                if (!emitted) {
                    emitted = emit_expert_stack_byte_spans(input_expr);
                }
            }
        } else if (root.kind == LayoutExprKind::View && !root.inputs.empty()) {
            const LayoutExprId input_expr_id = root.inputs.front();
            const auto source_contract = contract_for_expr.find(input_expr_id);
            if (source_contract != contract_for_expr.end()) {
                if (spec.view_axis < 0) {
                    throw std::runtime_error(
                        "rust loader compiler: view tensor '" +
                        binding.runtime_name +
                        "' does not declare ABI-owned view range");
                }
                if (root.axis >= 0 &&
                    (root.axis != spec.view_axis ||
                     root.start != spec.view_start ||
                     root.length != spec.view_length)) {
                    throw std::runtime_error(
                        "rust loader compiler: view tensor '" +
                        binding.runtime_name +
                        "' algebra range disagrees with RuntimeABI contract");
                }
                input.add_select_contract(
                    binding.runtime_name,
                    source_contract->second,
                    spec.view_axis,
                    spec.view_start,
                    spec.view_length,
                    spec.dtype,
                    spec.shape,
                    preferred_alignment);
                emitted = true;
            }
        } else if (root.kind == LayoutExprKind::Source) {
            emitted = emit_direct_source(root);
        }

        if (emitted) {
            const auto contract_id =
                static_cast<std::uint32_t>(emitted_contracts);
            contract_for_expr[binding.root] = contract_id;
            ++emitted_contracts;
        }
    }

    RustLoaderCompileResult result{
        .program = compile_rust_storage_program(input.view()),
        .source_tensor_names = std::move(tensor_names),
        .quant_attachments = {},
        .source_tensor_count = tensor_ids.size(),
        .direct_contract_count = emitted_contracts,
        .runtime_tensor_count = layout_plan.algebra.bindings.size(),
    };
    for (const auto& [name, spec] : layout_plan.tensors) {
        if (spec.quant.format == QuantFormat::None ||
            spec.quant.scale_tensor.empty()) {
            continue;
        }
        result.quant_attachments.push_back(
            RustQuantAttachment{
                .tensor_name = name,
                .scale_tensor_name = spec.quant.scale_tensor,
                .granularity = spec.quant.granularity,
                .group_size = spec.quant.group_size,
                .channel_axis = spec.quant.channel_axis,
            });
    }
    return result;
}

inline std::string describe_rust_storage_program(
    const pie_weight_loader::PieLoaderStorageProgramView& view,
    std::size_t source_tensor_count,
    std::size_t direct_contract_count,
    std::size_t runtime_tensor_count)
{
    std::uint64_t optimizer_rewrites = 0;
    for (std::size_t i = 0; i < view.optimizer.passes.len; ++i) {
        optimizer_rewrites += view.optimizer.passes.ptr[i].rewrites;
    }
    std::ostringstream out;
    out << "rust_storage_program(version=" << view.version
        << ", source_tensors=" << source_tensor_count
        << ", contracts=" << direct_contract_count << "/" << runtime_tensor_count
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
    std::size_t direct_contract_count,
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
               view, source_tensor_count, direct_contract_count,
               runtime_tensor_count)
        << "\",\n"
        << "  \"version\": " << view.version << ",\n"
        << "  \"source_tensor_count\": " << source_tensor_count << ",\n"
        << "  \"direct_contract_count\": " << direct_contract_count << ",\n"
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

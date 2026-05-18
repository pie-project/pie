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
#include "loader/rust_loader_input.hpp"
#include "loader/rust_storage_program.hpp"
#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

enum class RustLoaderPlannerMode {
    Cpp,
    Rust,
    Dual,
};

inline RustLoaderPlannerMode parse_rust_loader_planner_mode(
    const std::string& value)
{
    if (value.empty() || value == "cpp") return RustLoaderPlannerMode::Cpp;
    if (value == "rust") return RustLoaderPlannerMode::Rust;
    if (value == "dual") return RustLoaderPlannerMode::Dual;
    throw std::runtime_error(
        "rust loader planner: PIE_CUDA_LOADER_PLANNER must be one of "
        "{cpp,rust,dual}");
}

inline const char* rust_loader_planner_mode_name(
    RustLoaderPlannerMode mode) noexcept
{
    switch (mode) {
    case RustLoaderPlannerMode::Cpp: return "cpp";
    case RustLoaderPlannerMode::Rust: return "rust";
    case RustLoaderPlannerMode::Dual: return "dual";
    }
    return "?";
}

struct RustLoaderCompileResult {
    RustStorageProgram program;
    std::vector<std::string> source_tensor_names;
    std::size_t source_tensor_count = 0;
    std::size_t direct_contract_count = 0;
    std::size_t cpp_tensor_count = 0;
};

inline bool rust_loader_contract_is_direct(
    const TensorDecl& spec,
    const TensorInfo& info)
{
    return spec.shape == info.shape &&
           spec.dtype == info.dtype &&
           spec.ownership == TensorOwnershipKind::Owned &&
           spec.layout == TensorLayoutKind::Dense;
}

inline RustLoaderCompileResult compile_rust_loader_plan_from_cpp_plan(
    const HfConfig& hf,
    const LayoutPlan& cpp_plan,
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
    for (const auto& binding : cpp_plan.algebra.bindings) {
        if (binding.root >= cpp_plan.algebra.exprs.size()) continue;
        const LayoutExpr& root = cpp_plan.algebra.exprs[binding.root];
        const auto spec_it = cpp_plan.tensors.find(binding.runtime_name);
        if (spec_it == cpp_plan.tensors.end()) continue;
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
                if (source_expr_id >= cpp_plan.algebra.exprs.size()) return false;
                const LayoutExpr& source_expr =
                    cpp_plan.algebra.exprs[source_expr_id];
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

        bool emitted = false;
        if (root.kind == LayoutExprKind::Realize && !root.inputs.empty()) {
            const LayoutExprId input_expr_id = root.inputs.front();
            if (input_expr_id < cpp_plan.algebra.exprs.size()) {
                const LayoutExpr& input_expr =
                    cpp_plan.algebra.exprs[input_expr_id];
                emitted = emit_direct_source(input_expr);
                if (!emitted) {
                    emitted = emit_join(input_expr_id, input_expr);
                }
            }
        } else if (root.kind == LayoutExprKind::View && !root.inputs.empty()) {
            const LayoutExprId input_expr_id = root.inputs.front();
            const auto source_contract = contract_for_expr.find(input_expr_id);
            if (source_contract != contract_for_expr.end() && root.axis >= 0) {
                input.add_select_contract(
                    binding.runtime_name,
                    source_contract->second,
                    root.axis,
                    root.start,
                    root.length,
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
        .source_tensor_count = tensor_ids.size(),
        .direct_contract_count = emitted_contracts,
        .cpp_tensor_count = cpp_plan.algebra.bindings.size(),
    };
    return result;
}

inline std::string describe_rust_storage_program(
    const pie_weight_loader::PieLoaderStorageProgramView& view,
    std::size_t source_tensor_count,
    std::size_t direct_contract_count,
    std::size_t cpp_tensor_count)
{
    std::ostringstream out;
    out << "rust_storage_program(version=" << view.version
        << ", source_tensors=" << source_tensor_count
        << ", contracts=" << direct_contract_count << "/" << cpp_tensor_count
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

inline std::string dump_rust_storage_program_json(
    const pie_weight_loader::PieLoaderStorageProgramView& view,
    std::size_t source_tensor_count,
    std::size_t direct_contract_count,
    std::size_t cpp_tensor_count)
{
    std::ostringstream out;
    out << "{\n"
        << "  \"summary\": \""
        << describe_rust_storage_program(
               view, source_tensor_count, direct_contract_count,
               cpp_tensor_count)
        << "\",\n"
        << "  \"version\": " << view.version << ",\n"
        << "  \"source_tensor_count\": " << source_tensor_count << ",\n"
        << "  \"direct_contract_count\": " << direct_contract_count << ",\n"
        << "  \"cpp_tensor_count\": " << cpp_tensor_count << ",\n"
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
        << "  }\n"
        << "}\n";
    return out.str();
}

}  // namespace pie_cuda_driver

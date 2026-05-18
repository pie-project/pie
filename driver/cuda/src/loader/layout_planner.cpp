#include "loader/layout_planner.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace pie_cuda_driver {

namespace {

bool can_pack_2d_bf16_group(
    const CheckpointSource& source,
    const std::vector<std::string>& raw_names)
{
    if (raw_names.empty()) return false;
    std::int64_t cols = -1;
    for (const auto& raw : raw_names) {
        if (!source.contains(raw)) return false;
        const auto& info = source.info(raw);
        if (info.dtype != DType::BF16 || info.shape.size() != 2) return false;
        if (cols < 0) {
            cols = info.shape[1];
        } else if (info.shape[1] != cols) {
            return false;
        }
    }
    return true;
}

TensorParallelKind parallel_kind_from_axis(int axis) {
    if (axis == 0) return TensorParallelKind::Column;
    if (axis == 1) return TensorParallelKind::Row;
    return TensorParallelKind::Replicated;
}

std::vector<std::int64_t> sharded_shape(
    std::vector<std::int64_t> shape,
    int shard_axis,
    int tp_size,
    const std::string& name)
{
    if (tp_size <= 1 || shard_axis < 0) return shape;
    if (shard_axis >= static_cast<int>(shape.size())) {
        throw std::runtime_error(
            "layout planner: shard axis out of range for '" + name + "'");
    }
    if (shape[shard_axis] % tp_size != 0) {
        throw std::runtime_error(
            "layout planner: dimension " + std::to_string(shard_axis) +
            " of '" + name + "' is not divisible by tp_size=" +
            std::to_string(tp_size));
    }
    shape[shard_axis] /= tp_size;
    return shape;
}

std::uint64_t tensor_bytes(
    DType dtype,
    const std::vector<std::int64_t>& shape)
{
    std::uint64_t numel = 1;
    for (const auto dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("layout planner: negative tensor dimension");
        }
        numel *= static_cast<std::uint64_t>(dim);
    }
    return numel * static_cast<std::uint64_t>(dtype_bytes(dtype));
}

void register_tensor(LayoutPlan& plan, TensorDecl spec) {
    const std::string name = spec.name;
    auto [it, inserted] = plan.tensors.emplace(name, std::move(spec));
    if (!inserted) {
        throw std::runtime_error(
            "layout planner: duplicate runtime tensor '" + it->first + "'");
    }

    if (it->second.ownership == TensorOwnershipKind::Owned) {
        plan.memory.persistent_bytes += tensor_bytes(
            it->second.dtype, it->second.shape);
    } else if (it->second.ownership == TensorOwnershipKind::Temporary) {
        plan.memory.max_temporary_bytes = std::max(
            plan.memory.max_temporary_bytes,
            tensor_bytes(it->second.dtype, it->second.shape));
    }
    plan.memory.estimated_peak_bytes =
        plan.memory.persistent_bytes + plan.memory.max_temporary_bytes;
}

LayoutExprId add_expr(LayoutPlan& plan, LayoutExpr expr) {
    const LayoutExprId id = plan.algebra.exprs.size();
    plan.algebra.exprs.push_back(std::move(expr));
    return id;
}

LayoutExpr make_expr(LayoutExprKind kind, TensorDecl decl) {
    LayoutExpr expr;
    expr.kind = kind;
    expr.dtype = decl.dtype;
    expr.encoding = decl.quant;
    expr.decl = std::move(decl);
    return expr;
}

LayoutExprId source_expr(
    LayoutPlan& plan,
    const std::string& raw_name,
    TensorDecl decl)
{
    LayoutExpr expr = make_expr(LayoutExprKind::Source, std::move(decl));
    expr.raw_name = raw_name;
    return add_expr(plan, std::move(expr));
}

LayoutExprId maybe_partition_expr(
    LayoutPlan& plan,
    LayoutExprId input,
    TensorDecl decl,
    int shard_axis,
    int tp_size)
{
    if (tp_size <= 1 || shard_axis < 0) return input;
    LayoutExpr expr = make_expr(LayoutExprKind::Partition, std::move(decl));
    expr.inputs = {input};
    expr.axis = shard_axis;
    expr.partitions = tp_size;
    return add_expr(plan, std::move(expr));
}

LayoutExprId realize_expr(
    LayoutPlan& plan,
    std::string runtime_name,
    LayoutExprId input,
    TensorDecl decl)
{
    LayoutExpr expr = make_expr(LayoutExprKind::Realize, std::move(decl));
    expr.inputs = {input};
    expr.runtime_name = runtime_name;
    const LayoutExprId root = add_expr(plan, std::move(expr));
    plan.algebra.bindings.push_back(LayoutBinding{
        .runtime_name = std::move(runtime_name),
        .root = root,
    });
    return root;
}

void add_dense_source_tensor(
    LayoutPlan& plan,
    const RuntimeABI& runtime_abi,
    const SemanticTensor& tensor,
    const TensorInfo& info,
    int tp_size)
{
    const int shard_axis = tp_size > 1 ? tensor.shard_axis : -1;
    const auto final_shape = sharded_shape(
        info.shape, shard_axis, tp_size, tensor.runtime_name);
    auto contract = runtime_abi.tensor_contract(
        tensor.runtime_name,
        info.dtype,
        final_shape,
        TensorLayoutKind::Dense,
        TensorOwnershipKind::Owned,
        parallel_kind_from_axis(shard_axis));
    TensorDecl decl{
        .name = contract.name,
        .dtype = contract.dtype,
        .shape = contract.shape,
        .layout = contract.layout,
        .ownership = contract.ownership,
        .parallel = contract.parallel,
        .quant = contract.quant,
        .backing_tensor = contract.backing_tensor,
    };
    register_tensor(plan, decl);

    LayoutExprId source = source_expr(plan, tensor.raw_name, decl);
    source = maybe_partition_expr(plan, source, decl, shard_axis, tp_size);
    const std::string runtime_name = decl.name;
    realize_expr(plan, runtime_name, source, std::move(decl));
}

bool add_packed_axis_group(
    LayoutPlan& plan,
    const RuntimeABI& runtime_abi,
    const CheckpointSource& source,
    const SemanticGroup& group,
    std::unordered_set<std::string>& consumed_raw,
    int tp_size)
{
    const bool is_qkv = group.kind == SemanticGroupKind::PackedQkv;
    const bool is_gate_up = group.kind == SemanticGroupKind::PackedGateUp;
    if (!is_qkv && !is_gate_up) return false;

    const std::size_t expected = is_qkv ? 3 : 2;
    if (group.raw_names.size() != expected ||
        group.runtime_names.size() != expected) {
        throw std::runtime_error(
            "layout planner: packed group has wrong arity at '" +
            group.runtime_base + "'");
    }
    for (const auto& raw : group.raw_names) {
        if (consumed_raw.contains(raw)) return true;
    }
    if (!can_pack_2d_bf16_group(source, group.raw_names)) return false;

    const auto packed = runtime_abi.packed_projection(
        is_qkv
            ? RuntimeProjectionPackKind::AttentionQkvRows
            : RuntimeProjectionPackKind::MlpGateUpRows,
        group.runtime_base);

    std::int64_t rows = 0;
    std::int64_t cols = -1;
    std::vector<LayoutExprId> inputs;
    inputs.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        const auto& info = source.info(group.raw_names[i]);
        if (cols < 0) cols = info.shape[1];
        auto local_shape = sharded_shape(
            info.shape, 0, tp_size, group.raw_names[i]);
        rows += local_shape[0];

        TensorDecl source_decl{
            .name = group.runtime_names[i],
            .dtype = info.dtype,
            .shape = local_shape,
            .layout = TensorLayoutKind::Dense,
            .ownership = TensorOwnershipKind::Owned,
            .parallel = TensorParallelKind::Column,
            .quant = {},
            .backing_tensor = {},
        };
        LayoutExprId expr = source_expr(plan, group.raw_names[i], source_decl);
        expr = maybe_partition_expr(plan, expr, source_decl, 0, tp_size);
        inputs.push_back(expr);
    }

    auto packed_contract = runtime_abi.tensor_contract(
        packed.storage_name,
        source.info(group.raw_names.front()).dtype,
        {rows, cols},
        packed.storage_layout,
        TensorOwnershipKind::Owned,
        TensorParallelKind::Column);
        TensorDecl packed_decl{
            .name = packed_contract.name,
            .dtype = packed_contract.dtype,
            .shape = packed_contract.shape,
            .layout = packed_contract.layout,
            .ownership = packed_contract.ownership,
            .parallel = packed_contract.parallel,
            .quant = packed_contract.quant,
            .backing_tensor = packed_contract.backing_tensor,
            .view_axis = packed_contract.view_axis,
            .view_start = packed_contract.view_start,
            .view_length = packed_contract.view_length,
        };
    register_tensor(plan, packed_decl);

    LayoutExpr join = make_expr(LayoutExprKind::Join, packed_decl);
    join.inputs = std::move(inputs);
    join.axis = 0;
    const LayoutExprId joined = add_expr(plan, std::move(join));
    const std::string packed_name = packed_decl.name;
    realize_expr(plan, packed_name, joined, packed_decl);

    std::int64_t row_offset = 0;
    for (std::size_t i = 0; i < expected; ++i) {
        const auto& info = source.info(group.raw_names[i]);
        const auto local_shape = sharded_shape(
            info.shape, 0, tp_size, group.runtime_names[i]);
        auto view_contract = runtime_abi.view_contract(
            group.runtime_names[i],
            info.dtype,
            local_shape,
            packed_decl.name,
            /*axis=*/0,
            row_offset,
            local_shape[0],
            TensorParallelKind::Column);
        TensorDecl view_decl{
            .name = view_contract.name,
            .dtype = view_contract.dtype,
            .shape = view_contract.shape,
            .layout = view_contract.layout,
            .ownership = view_contract.ownership,
            .parallel = view_contract.parallel,
            .quant = {},
            .backing_tensor = view_contract.backing_tensor,
            .view_axis = view_contract.view_axis,
            .view_start = view_contract.view_start,
            .view_length = view_contract.view_length,
        };
        const std::string view_name = view_decl.name;
        register_tensor(plan, view_decl);

        LayoutExpr view = make_expr(LayoutExprKind::View, std::move(view_decl));
        view.inputs = {joined};
        view.runtime_name = view_name;
        view.axis = 0;
        view.start = row_offset;
        view.length = local_shape[0];
        const LayoutExprId view_root = add_expr(plan, std::move(view));
        plan.algebra.bindings.push_back(LayoutBinding{
            .runtime_name = view_name,
            .root = view_root,
        });

        consumed_raw.insert(group.raw_names[i]);
        row_offset += local_shape[0];
    }
    ++plan.axis_concat_groups;
    return true;
}

}  // namespace

LayoutPlanner::LayoutPlanner(const RuntimeABI& runtime_abi) noexcept
    : runtime_abi_(runtime_abi)
{}

LayoutPlan LayoutPlanner::build_dense_algebra_plan(
    const SemanticGraph& graph,
    const CheckpointSource& source,
    int tp_size) const
{
    LayoutPlan plan;
    std::unordered_set<std::string> consumed_raw;
    consumed_raw.reserve(graph.tensors.size());

    for (const auto& group : graph.groups) {
        (void)add_packed_axis_group(
            plan, runtime_abi_, source, group, consumed_raw, tp_size);
    }

    for (const auto& tensor : graph.tensors) {
        if (consumed_raw.contains(tensor.raw_name)) continue;
        const auto& info = source.info(tensor.raw_name);
        add_dense_source_tensor(plan, runtime_abi_, tensor, info, tp_size);
        consumed_raw.insert(tensor.raw_name);
    }

    validate_layout_plan(plan);
    return plan;
}

LayoutPlan build_native_dense_algebra_plan(
    const SemanticGraph& graph,
    const CheckpointSource& source,
    int tp_size,
    const RuntimeABI& runtime_abi)
{
    return LayoutPlanner(runtime_abi).build_dense_algebra_plan(
        graph, source, tp_size);
}

}  // namespace pie_cuda_driver

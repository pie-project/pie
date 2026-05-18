#include "loader/model_schema.hpp"

#include <algorithm>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "loader/model_adapter.hpp"
#include "loader/model_family.hpp"
#include "loader/layout_planner.hpp"
#include "loader/runtime_abi.hpp"
#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

namespace {

bool ends_with(const std::string& s, const char* suffix) {
    const auto n = std::char_traits<char>::length(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool can_pack_2d_bf16_group(
    const CheckpointSource& loader,
    const std::vector<std::string>& raw_names)
{
    if (raw_names.empty()) return false;
    std::int64_t cols = -1;
    for (const auto& raw : raw_names) {
        if (!loader.contains(raw)) return false;
        const auto& info = loader.info(raw);
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
            "load schema: shard axis out of range for '" + name + "'");
    }
    if (shape[shard_axis] % tp_size != 0) {
        throw std::runtime_error(
            "load schema: dimension " + std::to_string(shard_axis) +
            " of '" + name + "' is not divisible by tp_size=" +
            std::to_string(tp_size));
    }
    shape[shard_axis] /= tp_size;
    return shape;
}

void register_tensor_spec(
    LayoutPlan& plan,
    std::string name,
    DType dtype,
    std::vector<std::int64_t> shape,
    TensorLayoutKind layout,
    TensorOwnershipKind ownership,
    TensorParallelKind parallel,
    std::string backing_tensor = {},
    QuantSpec quant = {},
    int view_axis = -1,
    std::int64_t view_start = 0,
    std::int64_t view_length = 0)
{
    TensorDecl spec;
    spec.name = std::move(name);
    spec.dtype = dtype;
    spec.shape = std::move(shape);
    spec.layout = layout;
    spec.ownership = ownership;
    spec.parallel = parallel;
    spec.quant = std::move(quant);
    spec.backing_tensor = std::move(backing_tensor);
    spec.view_axis = view_axis;
    spec.view_start = view_start;
    spec.view_length = view_length;
    const std::string key = spec.name;
    auto [it, inserted] = plan.tensors.emplace(key, std::move(spec));
    if (!inserted) {
        throw std::runtime_error(
            "load schema: duplicate runtime tensor spec '" + it->first + "'");
    }

    if (it->second.ownership == TensorOwnershipKind::Owned ||
        it->second.ownership == TensorOwnershipKind::Temporary) {
        std::uint64_t numel = 1;
        for (const auto dim : it->second.shape) {
            numel *= static_cast<std::uint64_t>(dim);
        }
        const std::uint64_t bytes =
            numel * static_cast<std::uint64_t>(dtype_bytes(it->second.dtype));
        if (it->second.ownership == TensorOwnershipKind::Owned) {
            plan.memory.persistent_bytes += bytes;
        } else {
            plan.memory.max_temporary_bytes =
                std::max(plan.memory.max_temporary_bytes, bytes);
        }
        plan.memory.estimated_peak_bytes = std::max(
            plan.memory.estimated_peak_bytes,
            plan.memory.persistent_bytes + plan.memory.max_temporary_bytes);
    }
}

LayoutExprId add_expr(LayoutPlan& plan, LayoutExpr expr) {
    const LayoutExprId id = plan.algebra.exprs.size();
    plan.algebra.exprs.push_back(std::move(expr));
    return id;
}

LayoutExpr make_expr(LayoutExprKind kind, TensorDecl decl) {
    LayoutExpr expr;
    expr.kind = kind;
    expr.decl = std::move(decl);
    expr.dtype = expr.decl.dtype;
    expr.encoding = expr.decl.quant;
    return expr;
}

TensorDecl tensor_decl_for(LayoutPlan& plan, const std::string& name) {
    const auto it = plan.tensors.find(name);
    if (it == plan.tensors.end()) {
        throw std::runtime_error(
            "load schema: missing TensorDecl for algebra tensor '" + name + "'");
    }
    return it->second;
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

LayoutExprId source_expr_from_info(
    LayoutPlan& plan,
    const std::string& raw_name,
    const TensorInfo& info)
{
    TensorDecl decl;
    decl.name = raw_name;
    decl.dtype = info.dtype;
    decl.shape = info.shape;
    decl.layout = TensorLayoutKind::Dense;
    decl.ownership = TensorOwnershipKind::Temporary;
    decl.parallel = TensorParallelKind::Replicated;
    return source_expr(plan, raw_name, std::move(decl));
}

LayoutExprId partition_expr(
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

LayoutExprId select_expr(
    LayoutPlan& plan,
    LayoutExprId input,
    TensorDecl decl,
    int axis,
    std::int64_t start,
    std::int64_t length,
    int shard_axis = -1)
{
    LayoutExpr expr = make_expr(LayoutExprKind::Select, std::move(decl));
    expr.inputs = {input};
    expr.axis = axis;
    expr.start = start;
    expr.length = length;
    expr.partitions = shard_axis;
    return add_expr(plan, std::move(expr));
}

LayoutExprId realize_expr(
    LayoutPlan& plan,
    const std::string& runtime_name,
    LayoutExprId input,
    TensorDecl decl)
{
    LayoutExpr expr = make_expr(LayoutExprKind::Realize, std::move(decl));
    expr.inputs = {input};
    expr.runtime_name = runtime_name;
    const LayoutExprId root = add_expr(plan, std::move(expr));
    plan.algebra.bindings.push_back(LayoutBinding{
        .runtime_name = runtime_name,
        .root = root,
    });
    return root;
}

LayoutExprId source_realize_expr(
    LayoutPlan& plan,
    const std::string& raw_name,
    const std::string& output_name,
    int shard_axis,
    int tp_size)
{
    TensorDecl decl = tensor_decl_for(plan, output_name);
    LayoutExprId expr = source_expr(plan, raw_name, decl);
    expr = partition_expr(plan, expr, decl, shard_axis, tp_size);
    return realize_expr(plan, output_name, expr, std::move(decl));
}

LayoutExprId source_realize_expr_from_info(
    LayoutPlan& plan,
    const std::string& raw_name,
    const TensorInfo& info,
    const std::string& output_name,
    int shard_axis,
    int tp_size)
{
    TensorDecl decl = tensor_decl_for(plan, output_name);
    LayoutExprId expr = source_expr_from_info(plan, raw_name, info);
    expr = partition_expr(plan, expr, decl, shard_axis, tp_size);
    return realize_expr(plan, output_name, expr, std::move(decl));
}

LayoutExprId row_range_realize_expr_from_info(
    LayoutPlan& plan,
    const std::string& raw_name,
    const TensorInfo& info,
    const std::string& output_name,
    std::int64_t row_offset,
    std::int64_t rows,
    int tp_size)
{
    TensorDecl output_decl = tensor_decl_for(plan, output_name);
    LayoutExprId expr = source_expr_from_info(plan, raw_name, info);
    TensorDecl selected_decl = output_decl;
    selected_decl.shape[0] = rows;
    LayoutExpr select = make_expr(LayoutExprKind::Select, selected_decl);
    select.inputs = {expr};
    select.axis = 0;
    select.start = row_offset;
    select.length = rows;
    const LayoutExprId selected = add_expr(plan, std::move(select));
    LayoutExprId value = selected;
    if (tp_size > 1) {
        value = partition_expr(
            plan, selected, output_decl, 0, tp_size);
    }
    return realize_expr(plan, output_name, value, std::move(output_decl));
}

LayoutExprId cast_realize_expr(
    LayoutPlan& plan,
    const std::string& output_name,
    LayoutExprId input)
{
    TensorDecl decl = tensor_decl_for(plan, output_name);
    LayoutExpr cast = make_expr(LayoutExprKind::Cast, decl);
    cast.inputs = {input};
    cast.runtime_name = output_name;
    const LayoutExprId casted = add_expr(plan, std::move(cast));
    return realize_expr(plan, output_name, casted, std::move(decl));
}

LayoutExprId encode_realize_expr(
    LayoutPlan& plan,
    const std::string& output_name,
    const std::string& secondary_output_name,
    LayoutExprId input)
{
    TensorDecl decl = tensor_decl_for(plan, output_name);
    LayoutExpr encode = make_expr(LayoutExprKind::Encode, decl);
    encode.inputs = {input};
    encode.runtime_name = output_name;
    encode.secondary_runtime_name = secondary_output_name;
    const LayoutExprId encoded = add_expr(plan, std::move(encode));
    return realize_expr(plan, output_name, encoded, std::move(decl));
}

LayoutExprId reorder_realize_expr(
    LayoutPlan& plan,
    const std::string& output_name,
    const std::string& secondary_output_name,
    std::vector<LayoutExprId> inputs,
    int shard_axis)
{
    TensorDecl decl = tensor_decl_for(plan, output_name);
    LayoutExpr reorder = make_expr(LayoutExprKind::Reorder, decl);
    reorder.inputs = std::move(inputs);
    reorder.runtime_name = output_name;
    reorder.secondary_runtime_name = secondary_output_name;
    reorder.axis = shard_axis;
    const LayoutExprId reordered = add_expr(plan, std::move(reorder));
    realize_expr(plan, output_name, reordered, decl);
    if (!secondary_output_name.empty()) {
        TensorDecl secondary_decl = tensor_decl_for(plan, secondary_output_name);
        realize_expr(plan, secondary_output_name, reordered, std::move(secondary_decl));
    }
    return reordered;
}

LayoutExprId decode_realize_expr(
    LayoutPlan& plan,
    const std::string& output_name,
    std::vector<LayoutExprId> inputs)
{
    TensorDecl decl = tensor_decl_for(plan, output_name);
    LayoutExpr decode = make_expr(LayoutExprKind::Decode, decl);
    decode.inputs = std::move(inputs);
    decode.runtime_name = output_name;
    const LayoutExprId decoded = add_expr(plan, std::move(decode));
    return realize_expr(plan, output_name, decoded, std::move(decl));
}

LayoutExprId unzip_realize_expr(
    LayoutPlan& plan,
    const std::string& first_output,
    const std::string& second_output,
    LayoutExprId input,
    int shard_axis)
{
    TensorDecl decl = tensor_decl_for(plan, first_output);
    LayoutExpr unzip = make_expr(LayoutExprKind::Unzip, decl);
    unzip.inputs = {input};
    unzip.runtime_name = first_output;
    unzip.secondary_runtime_name = second_output;
    unzip.axis = shard_axis;
    const LayoutExprId unzipped = add_expr(plan, std::move(unzip));
    realize_expr(plan, first_output, unzipped, decl);
    TensorDecl second_decl = tensor_decl_for(plan, second_output);
    return realize_expr(plan, second_output, unzipped, std::move(second_decl));
}

LayoutExprId attach_metadata_expr(
    LayoutPlan& plan,
    const std::string& output_name,
    LayoutExprId input)
{
    TensorDecl decl = tensor_decl_for(plan, output_name);
    LayoutExpr attach = make_expr(LayoutExprKind::Attach, std::move(decl));
    attach.inputs = {input};
    attach.runtime_name = output_name;
    return add_expr(plan, std::move(attach));
}

void release_expr(LayoutPlan& plan, std::string name, std::vector<LayoutExprId> inputs) {
    TensorDecl decl;
    decl.name = std::move(name);
    LayoutExpr release = make_expr(LayoutExprKind::Release, std::move(decl));
    release.inputs = std::move(inputs);
    release.runtime_name = release.decl.name;
    (void)add_expr(plan, std::move(release));
}

std::uint64_t tensor_nbytes(
    DType dtype,
    const std::vector<std::int64_t>& shape);
bool normalizes_to_bf16(DType dtype) noexcept;
void estimate_temporary_bytes(LayoutPlan& plan, std::uint64_t bytes);

LayoutExprId add_owned_source_tensor(
    LayoutPlan& plan,
    const std::string& raw_name,
    const std::string& output_name,
    const TensorInfo& info,
    const std::vector<std::int64_t>& shape,
    TensorLayoutKind layout,
    TensorParallelKind parallel,
    int shard_axis,
    int tp_size)
{
    if (!normalizes_to_bf16(info.dtype) || ends_with(output_name, "_scale_inv")) {
        register_tensor_spec(
            plan, output_name, info.dtype, shape, layout,
            TensorOwnershipKind::Owned, parallel);
        return source_realize_expr_from_info(
            plan, raw_name, info, output_name, shard_axis, tp_size);
    }

    const std::string tmp_name = output_name + ".__dtype_source";
    register_tensor_spec(
        plan, tmp_name, info.dtype, shape, layout,
        TensorOwnershipKind::Temporary, parallel);
    const LayoutExprId tmp = source_realize_expr_from_info(
        plan, raw_name, info, tmp_name, shard_axis, tp_size);

    register_tensor_spec(
        plan, output_name, DType::BF16, shape, layout,
        TensorOwnershipKind::Owned, parallel);
    const LayoutExprId out = cast_realize_expr(plan, output_name, tmp);
    release_expr(plan, tmp_name + ".__drop", {tmp});
    estimate_temporary_bytes(plan, tensor_nbytes(info.dtype, shape));
    return out;
}

LayoutExprId add_owned_row_range_tensor(
    LayoutPlan& plan,
    const std::string& raw_name,
    const std::string& output_name,
    const TensorInfo& info,
    std::int64_t row_offset,
    std::int64_t rows,
    const std::vector<std::int64_t>& shape,
    TensorLayoutKind layout,
    TensorParallelKind parallel,
    int tp_size)
{
    if (!normalizes_to_bf16(info.dtype) || ends_with(output_name, "_scale_inv")) {
        register_tensor_spec(
            plan, output_name, info.dtype, shape, layout,
            TensorOwnershipKind::Owned, parallel);
        return row_range_realize_expr_from_info(
            plan, raw_name, info, output_name, row_offset, rows, tp_size);
    }

    const std::string tmp_name = output_name + ".__dtype_source";
    register_tensor_spec(
        plan, tmp_name, info.dtype, shape, layout,
        TensorOwnershipKind::Temporary, parallel);
    const LayoutExprId tmp = row_range_realize_expr_from_info(
        plan, raw_name, info, tmp_name, row_offset, rows, tp_size);
    register_tensor_spec(
        plan, output_name, DType::BF16, shape, layout,
        TensorOwnershipKind::Owned, parallel);
    const LayoutExprId out = cast_realize_expr(plan, output_name, tmp);
    release_expr(plan, tmp_name + ".__drop", {tmp});
    estimate_temporary_bytes(plan, tensor_nbytes(info.dtype, shape));
    return out;
}

void register_tensor_contract(
    LayoutPlan& plan,
    RuntimeTensorContract contract)
{
    register_tensor_spec(
        plan,
        std::move(contract.name),
        contract.dtype,
        std::move(contract.shape),
        contract.layout,
        contract.ownership,
        contract.parallel,
        std::move(contract.backing_tensor),
        std::move(contract.quant),
        contract.view_axis,
        contract.view_start,
        contract.view_length);
}

void estimate_temporary_bytes(LayoutPlan& plan, std::uint64_t bytes) {
    plan.memory.max_temporary_bytes =
        std::max(plan.memory.max_temporary_bytes, bytes);
    plan.memory.estimated_peak_bytes = std::max(
        plan.memory.estimated_peak_bytes,
        plan.memory.persistent_bytes + plan.memory.max_temporary_bytes);
}

std::uint64_t tensor_nbytes(
    DType dtype,
    const std::vector<std::int64_t>& shape);

bool normalizes_to_bf16(DType dtype) noexcept {
    return dtype == DType::FP16 || dtype == DType::FP32;
}

bool try_add_packed_axis_group(
    LayoutPlan& plan,
    const CheckpointSource& loader,
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
            "load schema: packed semantic group has wrong arity at '" +
            group.runtime_base + "'");
    }
    for (const auto& raw : group.raw_names) {
        if (consumed_raw.contains(raw)) return true;
    }
    if (!can_pack_2d_bf16_group(loader, group.raw_names)) return false;

    const auto& runtime_abi = pie_cuda_runtime_abi();
    const auto packed = runtime_abi.packed_projection(
        is_qkv
            ? RuntimeProjectionPackKind::AttentionQkvRows
            : RuntimeProjectionPackKind::MlpGateUpRows,
        group.runtime_base);
    const std::string& packed_name = packed.storage_name;

    std::int64_t rows = 0;
    std::int64_t cols = -1;
    std::vector<LayoutExprId> inputs;
    inputs.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        const auto& info = loader.info(group.raw_names[i]);
        if (cols < 0) cols = info.shape[1];
        const auto local_shape = sharded_shape(
            info.shape, 0, tp_size, group.raw_names[i]);
        rows += local_shape[0];
        TensorDecl local_decl;
        local_decl.name = group.runtime_names[i];
        local_decl.dtype = info.dtype;
        local_decl.shape = local_shape;
        local_decl.layout = TensorLayoutKind::Dense;
        local_decl.ownership = TensorOwnershipKind::Owned;
        local_decl.parallel = TensorParallelKind::Column;
        LayoutExprId expr = source_expr_from_info(
            plan, group.raw_names[i], info);
        expr = partition_expr(plan, expr, local_decl, 0, tp_size);
        inputs.push_back(expr);
    }

    register_tensor_contract(
        plan,
        runtime_abi.tensor_contract(
            packed_name,
            loader.info(group.raw_names[0]).dtype,
            {rows, cols},
            packed.storage_layout,
            TensorOwnershipKind::Owned,
            TensorParallelKind::Column));
    TensorDecl packed_decl = tensor_decl_for(plan, packed_name);
    LayoutExpr join = make_expr(LayoutExprKind::Join, packed_decl);
    join.inputs = std::move(inputs);
    join.axis = 0;
    const LayoutExprId joined = add_expr(plan, std::move(join));
    realize_expr(plan, packed_name, joined, packed_decl);

    for (std::size_t i = 0; i < expected; ++i) {
        const auto& info = loader.info(group.raw_names[i]);
        const auto local_shape = sharded_shape(
            info.shape, 0, tp_size, group.runtime_names[i]);
        std::int64_t view_start = 0;
        for (std::size_t j = 0; j < i; ++j) {
            view_start += loader.info(group.raw_names[j]).shape[0] / tp_size;
        }
        const std::int64_t view_length = local_shape[0];
        register_tensor_contract(
            plan,
            runtime_abi.view_contract(
                group.runtime_names[i],
                info.dtype,
                local_shape,
                packed_name,
                /*axis=*/0,
                view_start,
                view_length,
                TensorParallelKind::Column));
        TensorDecl view_decl = tensor_decl_for(plan, group.runtime_names[i]);
        LayoutExpr view = make_expr(LayoutExprKind::View, view_decl);
        view.inputs = {joined};
        view.runtime_name = group.runtime_names[i];
        view.axis = 0;
        view.start = view_start;
        view.length = view_length;
        const LayoutExprId view_root = add_expr(plan, std::move(view));
        plan.algebra.bindings.push_back(LayoutBinding{
            .runtime_name = group.runtime_names[i],
            .root = view_root,
        });
        consumed_raw.insert(group.raw_names[i]);
    }
    ++plan.axis_concat_groups;
    return true;
}

const std::string& group_runtime_name_for_role(
    const SemanticGroup& group,
    SemanticRole role)
{
    for (std::size_t i = 0; i < group.runtime_roles.size(); ++i) {
        if (group.runtime_roles[i] == role && i < group.runtime_names.size()) {
            return group.runtime_names[i];
        }
    }
    throw std::runtime_error(
        "load schema: semantic group '" + group.runtime_base +
        "' does not declare expected runtime role");
}

bool try_add_row_range_split_group(
    LayoutPlan& plan,
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticGroup& group,
    std::unordered_set<std::string>& consumed_raw,
    int tp_size)
{
    if (group.kind != SemanticGroupKind::RowRangeSplit) return false;
    if (group.raw_names.size() != 1 || group.raw_roles.size() != 1) {
        throw std::runtime_error(
            "load schema: row-range split group has wrong source arity at '" +
            group.runtime_base + "'");
    }
    const std::string& raw_name = group.raw_names[0];
    if (consumed_raw.contains(raw_name)) return true;
    const TensorInfo& info = loader.info(raw_name);
    const SemanticRole fused_role = group.raw_roles[0];

    struct SplitPart {
        SemanticRole role;
        std::int64_t offset;
        std::int64_t rows;
    };
    std::vector<SplitPart> parts;
    if (fused_role == SemanticRole::AttentionQkv) {
        const std::int64_t Hq =
            static_cast<std::int64_t>(hf.num_attention_heads) * hf.head_dim;
        const std::int64_t Hk =
            static_cast<std::int64_t>(hf.num_key_value_heads) * hf.head_dim;
        parts = {
            {SemanticRole::AttentionQ, 0, Hq},
            {SemanticRole::AttentionK, Hq, Hk},
            {SemanticRole::AttentionV, Hq + Hk, Hk},
        };
    } else if (fused_role == SemanticRole::MlpGateUp) {
        const std::int64_t I = hf.intermediate_size;
        parts = {
            {SemanticRole::MlpGate, 0, I},
            {SemanticRole::MlpUp, I, I},
        };
    } else {
        return false;
    }

    if (info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: row-range split expects 2-D source at '" +
            raw_name + "'");
    }
    for (const auto& part : parts) {
        if (part.rows % tp_size != 0) {
            throw std::runtime_error(
                "load schema: row-range split output is not divisible by "
                "tp_size at '" + raw_name + "'");
        }
        const std::string& output_name =
            group_runtime_name_for_role(group, part.role);
        add_owned_row_range_tensor(
            plan, raw_name, output_name, info, part.offset, part.rows,
            {part.rows / tp_size, info.shape[1]},
            TensorLayoutKind::Dense, TensorParallelKind::Column, tp_size);
    }

    consumed_raw.insert(raw_name);
    return true;
}

void add_copy(LayoutPlan& plan, const std::string& raw_name,
              const std::string& output_name, const TensorInfo& info,
              int shard_axis, int tp_size)
{
    add_owned_source_tensor(
        plan, raw_name, output_name, info,
        sharded_shape(info.shape, shard_axis, tp_size, output_name),
        TensorLayoutKind::Dense, parallel_kind_from_axis(shard_axis),
        shard_axis, tp_size);
}

bool runtime_quant_model_supported(const std::string& mt) {
    return mt == "qwen3"
        || mt == "qwen2"
        || mt == "llama" || mt == "llama3"
        || mt == "mistral"
        || mt == "qwen3_5" || mt == "qwen3_5_text";
}

bool runtime_quantizable_role(SemanticRole role) {
    switch (role) {
    case SemanticRole::AttentionQ:
    case SemanticRole::AttentionK:
    case SemanticRole::AttentionV:
    case SemanticRole::AttentionO:
    case SemanticRole::MlpGate:
    case SemanticRole::MlpUp:
    case SemanticRole::MlpDown:
        return true;
    default:
        return false;
    }
}

bool runtime_quant_enabled_for_plan(
    const HfConfig& hf,
    const Config& boot_cfg,
    bool fp8_native)
{
    const auto& mode = boot_cfg.model.runtime_quant;
    if (mode.empty()) return false;
    const bool is_fp8 = (mode == "fp8");
    const bool is_int8 = (mode == "int8");
    if (!is_fp8 && !is_int8) {
        throw std::runtime_error(
            "load schema: unsupported runtime_quant '" + mode +
            "'. Currently supported: 'fp8' or 'int8'.");
    }
    if (!runtime_quant_model_supported(hf.model_type)) {
        throw std::runtime_error(
            "load schema: runtime_quant=" + mode + " is currently wired "
            "for {qwen2, qwen3, qwen3_5, qwen3_5_text, llama, llama3, "
            "mistral} (got '" + hf.model_type + "').");
    }
    if (!hf.quant_method.empty()) return false;
    if (is_fp8 && !fp8_native) return false;
    return true;
}

void add_runtime_quantized_copy(
    LayoutPlan& plan,
    const SemanticTensor& semantic,
    const TensorInfo& info,
    const Config& boot_cfg,
    int shard_axis,
    int tp_size)
{
    if (info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: runtime quant source is not 2-D: " +
            semantic.runtime_name);
    }

    const std::string tmp_name =
        semantic.runtime_name + ".__runtime_quant_source";

    auto final_shape =
        sharded_shape(info.shape, shard_axis, tp_size, semantic.runtime_name);
    const bool is_int8 = boot_cfg.model.runtime_quant == "int8";
    const DType q_dtype = is_int8 ? DType::INT8 : DType::FP8_E4M3;
    const QuantFormat q_format = is_int8
        ? QuantFormat::RuntimeInt8
        : QuantFormat::RuntimeFp8E4M3;
    const std::string scale_name =
        pie_cuda_runtime_abi().quant_scale_inv_name(semantic.runtime_name);
    const TensorParallelKind parallel = parallel_kind_from_axis(shard_axis);

    register_tensor_spec(
        plan, tmp_name, info.dtype, final_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        parallel);
    const LayoutExprId source = source_realize_expr_from_info(
        plan, semantic.raw_name, info, tmp_name, shard_axis, tp_size);

    QuantSpec quant;
    quant.format = q_format;
    quant.granularity = QuantGranularity::PerChannel;
    quant.group_size = 0;
    quant.channel_axis = 0;
    quant.scale_tensor = scale_name;

    register_tensor_spec(
        plan, semantic.runtime_name, q_dtype, final_shape,
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        parallel, /*backing_tensor=*/{}, std::move(quant));
    register_tensor_spec(
        plan, scale_name, DType::FP32, {final_shape[0]},
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        parallel);

    const LayoutExprId encoded = encode_realize_expr(
        plan, semantic.runtime_name, scale_name, source);
    (void)attach_metadata_expr(plan, semantic.runtime_name, encoded);
    release_expr(plan, tmp_name + ".__drop", {source});

    std::uint64_t source_numel = 1;
    for (const auto dim : final_shape) {
        source_numel *= static_cast<std::uint64_t>(dim);
    }
    const std::uint64_t source_bytes =
        source_numel * static_cast<std::uint64_t>(dtype_bytes(info.dtype));
    const std::uint64_t bf16_scratch_bytes =
        (info.dtype == DType::BF16)
            ? 0
            : source_numel * static_cast<std::uint64_t>(dtype_bytes(DType::BF16));
    estimate_temporary_bytes(plan, source_bytes + bf16_scratch_bytes);
}

bool is_compressed_fp8_scale_companion(
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic)
{
    if (hf.quant_method != "compressed-tensors") return false;
    constexpr const char* scale_suffix = "_scale";
    if (!ends_with(semantic.raw_name, scale_suffix)) return false;
    const std::string raw_weight =
        semantic.raw_name.substr(0, semantic.raw_name.size() -
                                      std::strlen(scale_suffix));
    return loader.contains(raw_weight) &&
           loader.info(raw_weight).dtype == DType::FP8_E4M3;
}

bool is_compressed_quant_companion(
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic)
{
    if (is_compressed_fp8_scale_companion(hf, loader, semantic)) {
        return true;
    }
    if (hf.quant_method != "compressed-tensors") return false;
    constexpr const char* scale_suffix = "_scale";
    constexpr const char* zero_suffix = "_zero_point";
    const char* matched = nullptr;
    if (ends_with(semantic.raw_name, scale_suffix)) matched = scale_suffix;
    if (ends_with(semantic.raw_name, zero_suffix)) matched = zero_suffix;
    if (matched == nullptr) return false;

    const std::string raw_weight =
        semantic.raw_name.substr(0, semantic.raw_name.size() -
                                      std::strlen(matched));
    return loader.contains(raw_weight) &&
           (loader.info(raw_weight).dtype == DType::FP8_E4M3 ||
            loader.info(raw_weight).dtype == DType::INT8);
}

bool is_fp8_scale_inv_companion(
    const CheckpointSource& loader,
    const SemanticTensor& semantic)
{
    constexpr const char* scale_suffix = "_scale_inv";
    if (!ends_with(semantic.raw_name, scale_suffix)) return false;
    const std::string raw_weight =
        semantic.raw_name.substr(0, semantic.raw_name.size() -
                                      std::strlen(scale_suffix));
    return loader.contains(raw_weight) &&
           loader.info(raw_weight).dtype == DType::FP8_E4M3;
}

bool try_add_compressed_fp8_weight(
    LayoutPlan& plan,
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic,
    const TensorInfo& info,
    int shard_axis,
    int tp_size,
    bool fp8_native)
{
    if (hf.quant_method != "compressed-tensors" ||
        info.dtype != DType::FP8_E4M3 ||
        !ends_with(semantic.runtime_name, ".weight")) {
        return false;
    }

    const std::string scale_raw = semantic.raw_name + "_scale";
    if (!loader.contains(scale_raw)) return false;
    const TensorInfo& scale_info = loader.info(scale_raw);
    if (scale_info.dtype != DType::BF16 && scale_info.dtype != DType::FP32) {
        throw std::runtime_error(
            "load schema: compressed-tensors FP8 scale '" + scale_raw +
            "' has unsupported dtype " +
            std::string(dtype_name(scale_info.dtype)));
    }

    const auto final_shape =
        sharded_shape(info.shape, shard_axis, tp_size, semantic.runtime_name);
    const int scale_axis = (tp_size > 1)
        ? llama_like_shard_axis(semantic.runtime_name + "_scale")
        : -1;
    const auto scale_shape = sharded_shape(
        scale_info.shape, scale_axis, tp_size, semantic.runtime_name + "_scale");
    std::uint64_t scale_numel = 1;
    for (const auto dim : scale_shape) {
        scale_numel *= static_cast<std::uint64_t>(dim);
    }

    if (fp8_native) {
        const std::string scale_name =
            pie_cuda_runtime_abi().quant_scale_inv_name(semantic.runtime_name);

        QuantSpec quant;
        quant.format = QuantFormat::CompressedFp8E4M3;
        quant.granularity = (scale_numel == 1)
            ? QuantGranularity::PerTensor
            : QuantGranularity::PerChannel;
        quant.group_size = 0;
        quant.channel_axis = 0;
        quant.scale_tensor = scale_name;

        register_tensor_spec(
            plan, semantic.runtime_name, DType::FP8_E4M3, final_shape,
            TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
            parallel_kind_from_axis(shard_axis),
            /*backing_tensor=*/{}, std::move(quant));
        const LayoutExprId weight = source_realize_expr_from_info(
            plan, semantic.raw_name, info, semantic.runtime_name,
            shard_axis, tp_size);

        register_tensor_spec(
            plan, scale_name, DType::FP32, scale_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
            parallel_kind_from_axis(scale_axis));
        if (scale_info.dtype == DType::FP32) {
            (void)source_realize_expr_from_info(
                plan, scale_raw, scale_info, scale_name, scale_axis, tp_size);
        } else {
            const std::string scale_tmp = scale_name + ".__source";

            register_tensor_spec(
                plan, scale_tmp, scale_info.dtype, scale_shape,
                TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
                parallel_kind_from_axis(scale_axis));
            const LayoutExprId tmp = source_realize_expr_from_info(
                plan, scale_raw, scale_info, scale_tmp, scale_axis, tp_size);

            (void)cast_realize_expr(plan, scale_name, tmp);
            release_expr(plan, scale_tmp + ".__drop", {tmp});
        }

        (void)attach_metadata_expr(plan, semantic.runtime_name, weight);
    } else {
        const std::string weight_tmp =
            semantic.runtime_name + ".__compressed_fp8_source";
        const std::string scale_tmp =
            semantic.runtime_name + ".__compressed_fp8_scale";

        register_tensor_spec(
            plan, weight_tmp, DType::FP8_E4M3, final_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
            parallel_kind_from_axis(shard_axis));
        register_tensor_spec(
            plan, scale_tmp, scale_info.dtype, scale_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
            parallel_kind_from_axis(scale_axis));

        register_tensor_spec(
            plan, semantic.runtime_name, DType::BF16, final_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
            parallel_kind_from_axis(shard_axis));

        const LayoutExprId weight = source_realize_expr_from_info(
            plan, semantic.raw_name, info, weight_tmp, shard_axis, tp_size);
        const LayoutExprId scale = source_realize_expr_from_info(
            plan, scale_raw, scale_info, scale_tmp, scale_axis, tp_size);
        (void)decode_realize_expr(
            plan, semantic.runtime_name, {weight, scale});
        release_expr(
            plan, semantic.runtime_name + ".__compressed_fp8_drop",
            {weight, scale});

        std::uint64_t source_numel = 1;
        for (const auto dim : final_shape) {
            source_numel *= static_cast<std::uint64_t>(dim);
        }
        const std::uint64_t scale_bytes =
            scale_numel * static_cast<std::uint64_t>(dtype_bytes(scale_info.dtype));
        const std::uint64_t fp32_scale_scratch =
            scale_info.dtype == DType::BF16
                ? scale_numel * static_cast<std::uint64_t>(dtype_bytes(DType::FP32))
                : 0;
        estimate_temporary_bytes(
            plan,
            source_numel * static_cast<std::uint64_t>(dtype_bytes(DType::FP8_E4M3)) +
            scale_bytes + fp32_scale_scratch);
    }
    return true;
}

bool try_add_fp8_scale_inv_weight(
    LayoutPlan& plan,
    const CheckpointSource& loader,
    const SemanticTensor& semantic,
    const TensorInfo& info,
    int shard_axis,
    int tp_size)
{
    if (info.dtype != DType::FP8_E4M3 ||
        !ends_with(semantic.runtime_name, ".weight")) {
        return false;
    }

    const std::string scale_raw = semantic.raw_name + "_scale_inv";
    if (!loader.contains(scale_raw)) return false;
    const TensorInfo& scale_info = loader.info(scale_raw);
    if (scale_info.dtype != DType::BF16 && scale_info.dtype != DType::FP32) {
        throw std::runtime_error(
            "load schema: FP8 scale_inv '" + scale_raw +
            "' has unsupported dtype " +
            std::string(dtype_name(scale_info.dtype)));
    }

    const auto final_shape =
        sharded_shape(info.shape, shard_axis, tp_size, semantic.runtime_name);
    const std::string scale_name =
        pie_cuda_runtime_abi().quant_scale_inv_name(semantic.runtime_name);
    int scale_axis = -1;
    if (scale_info.shape.size() == 1 && final_shape.size() >= 1 &&
        scale_info.shape[0] == info.shape[0]) {
        scale_axis = shard_axis == 0 ? 0 : -1;
    }
    const auto scale_shape =
        sharded_shape(scale_info.shape, scale_axis, tp_size, scale_name);

    std::uint64_t scale_numel = 1;
    for (const auto dim : scale_shape) {
        scale_numel *= static_cast<std::uint64_t>(dim);
    }
    if (scale_numel != 1 &&
        (final_shape.empty() ||
         scale_numel != static_cast<std::uint64_t>(final_shape[0]))) {
        throw std::runtime_error(
            "load schema: FP8 scale_inv '" + scale_raw +
            "' must be scalar or one scale per output row for '" +
            semantic.runtime_name + "'");
    }

    const std::string weight_tmp =
        semantic.runtime_name + ".__fp8_scale_inv_source";
    const std::string scale_tmp =
        semantic.runtime_name + ".__fp8_scale_inv_scale";

    register_tensor_spec(
        plan, weight_tmp, DType::FP8_E4M3, final_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        parallel_kind_from_axis(shard_axis));
    register_tensor_spec(
        plan, scale_tmp, scale_info.dtype, scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        parallel_kind_from_axis(scale_axis));
    register_tensor_spec(
        plan, semantic.runtime_name, DType::BF16, final_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        parallel_kind_from_axis(shard_axis));

    const LayoutExprId weight = source_realize_expr_from_info(
        plan, semantic.raw_name, info, weight_tmp, shard_axis, tp_size);
    const LayoutExprId scale = source_realize_expr_from_info(
        plan, scale_raw, scale_info, scale_tmp, scale_axis, tp_size);
    (void)decode_realize_expr(plan, semantic.runtime_name, {weight, scale});
    release_expr(
        plan, semantic.runtime_name + ".__fp8_scale_inv_drop",
        {weight, scale});

    estimate_temporary_bytes(
        plan,
        tensor_nbytes(DType::FP8_E4M3, final_shape) +
        tensor_nbytes(scale_info.dtype, scale_shape) +
        (scale_info.dtype == DType::BF16
            ? tensor_nbytes(DType::FP32, scale_shape)
            : 0));
    return true;
}

bool try_add_compressed_int8_weight(
    LayoutPlan& plan,
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic,
    const TensorInfo& info,
    int shard_axis,
    int tp_size)
{
    if (hf.quant_method != "compressed-tensors" ||
        info.dtype != DType::INT8 ||
        !ends_with(semantic.runtime_name, ".weight")) {
        return false;
    }

    const std::string scale_raw = semantic.raw_name + "_scale";
    const std::string zero_raw = semantic.raw_name + "_zero_point";
    if (!loader.contains(scale_raw)) return false;
    if (loader.contains(zero_raw)) {
        throw std::runtime_error(
            "load schema: compressed-tensors INT8 weight '" +
            semantic.raw_name + "' has a zero-point companion. The scheduled "
            "INT8 runtime backend supports symmetric per-channel INT8; "
            "asymmetric compressed-tensors INT8 should lower through an "
            "explicit Dequantize op once that kernel is registered.");
    }

    const TensorInfo& scale_info = loader.info(scale_raw);
    if (scale_info.dtype != DType::BF16 && scale_info.dtype != DType::FP32) {
        throw std::runtime_error(
            "load schema: compressed-tensors INT8 scale '" + scale_raw +
            "' has unsupported dtype " +
            std::string(dtype_name(scale_info.dtype)));
    }

    const auto final_shape =
        sharded_shape(info.shape, shard_axis, tp_size, semantic.runtime_name);
    if (final_shape.size() != 2) {
        throw std::runtime_error(
            "load schema: compressed-tensors INT8 weight is not 2-D: " +
            semantic.runtime_name);
    }

    const int scale_axis = (tp_size > 1)
        ? llama_like_shard_axis(semantic.runtime_name + "_scale")
        : -1;
    const auto scale_shape = sharded_shape(
        scale_info.shape, scale_axis, tp_size, semantic.runtime_name + "_scale");
    std::uint64_t scale_numel = 1;
    for (const auto dim : scale_shape) {
        scale_numel *= static_cast<std::uint64_t>(dim);
    }
    if (scale_numel != static_cast<std::uint64_t>(final_shape[0])) {
        throw std::runtime_error(
            "load schema: compressed-tensors INT8 currently requires one "
            "scale per output row for '" + semantic.runtime_name + "'");
    }

    const std::string scale_name =
        pie_cuda_runtime_abi().quant_scale_inv_name(semantic.runtime_name);

    QuantSpec quant;
    quant.format = QuantFormat::CompressedInt8;
    quant.granularity = QuantGranularity::PerChannel;
    quant.group_size = 0;
    quant.channel_axis = 0;
    quant.scale_tensor = scale_name;

    register_tensor_spec(
        plan, semantic.runtime_name, DType::INT8, final_shape,
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        parallel_kind_from_axis(shard_axis),
        /*backing_tensor=*/{}, std::move(quant));
    const LayoutExprId weight = source_realize_expr_from_info(
        plan, semantic.raw_name, info, semantic.runtime_name,
        shard_axis, tp_size);

    register_tensor_spec(
        plan, scale_name, DType::FP32, scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        parallel_kind_from_axis(scale_axis));
    if (scale_info.dtype == DType::FP32) {
        (void)source_realize_expr_from_info(
            plan, scale_raw, scale_info, scale_name, scale_axis, tp_size);
    } else {
        const std::string scale_tmp = scale_name + ".__source";

        register_tensor_spec(
            plan, scale_tmp, scale_info.dtype, scale_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
            parallel_kind_from_axis(scale_axis));
        const LayoutExprId tmp = source_realize_expr_from_info(
            plan, scale_raw, scale_info, scale_tmp, scale_axis, tp_size);

        (void)cast_realize_expr(plan, scale_name, tmp);
        release_expr(plan, scale_tmp + ".__drop", {tmp});
    }

    (void)attach_metadata_expr(plan, semantic.runtime_name, weight);
    return true;
}

bool can_lower_gptq_marlin_repack_to_plan(
    const HfConfig& hf,
    int tp_size)
{
    return hf.quant_method == "gptq" &&
           hf.quant_bits == 4 &&
           hf.quant_group_size > 0 &&
           !hf.quant_desc_act &&
           hf.quant_sym &&
           !hf.quant_zero_point &&
           tp_size >= 1;
}

bool can_lower_awq_dequant_to_plan(const HfConfig& hf) {
    return hf.quant_method == "awq" &&
           hf.quant_bits == 4 &&
           hf.quant_group_size > 0;
}

bool can_lower_awq_marlin_repack_to_plan(const HfConfig& hf) {
    return hf.quant_method == "awq" &&
           hf.quant_bits == 4 &&
           hf.quant_group_size > 0;
}

bool can_lower_gptq_dequant_to_plan(
    const HfConfig& hf,
    int tp_size)
{
    (void)tp_size;
    if (hf.quant_method != "gptq" ||
        hf.quant_bits != 4 ||
        hf.quant_group_size <= 0) {
        return false;
    }
    const bool needs_bf16_fallback =
        hf.quant_desc_act || hf.quant_zero_point || !hf.quant_sym;
    return needs_bf16_fallback;
}

enum class OfflineInt4Format {
    Awq,
    Gptq,
};

struct OfflineInt4ShardPlan {
    int canonical_axis = -1;
    int qweight_axis = -1;
    int qzeros_axis = -1;
    int scale_axis = -1;
    int gidx_axis = -1;
    TensorParallelKind parallel = TensorParallelKind::Replicated;
};

OfflineInt4ShardPlan offline_int4_shard_plan_for(
    OfflineInt4Format format,
    const std::string& canonical_weight_name,
    int tp_size)
{
    OfflineInt4ShardPlan out;
    if (tp_size <= 1) return out;

    out.canonical_axis = llama_like_shard_axis(canonical_weight_name);
    out.parallel = parallel_kind_from_axis(out.canonical_axis);
    if (out.canonical_axis == 0) {
        // Canonical dense weight is [N, K]. Column-parallel sharding cuts N.
        // GPTQ qweight is [K/8, N]; AWQ qweight/qzeros pack N by 8.
        out.qweight_axis = 1;
        out.qzeros_axis = 1;
        out.scale_axis = 1;
    } else if (out.canonical_axis == 1) {
        // Row-parallel sharding cuts K. GPTQ packs K by 8 on qweight axis 0;
        // AWQ stores K directly on qweight axis 0. Both use K-derived scale
        // groups, and GPTQ g_idx follows K when present.
        out.qweight_axis = 0;
        out.qzeros_axis = 0;
        out.scale_axis = 0;
        if (format == OfflineInt4Format::Gptq) {
            out.gidx_axis = 0;
        }
    }
    return out;
}

OfflineInt4ShardPlan gptq_shard_plan_for(
    const std::string& canonical_weight_name,
    int tp_size)
{
    return offline_int4_shard_plan_for(
        OfflineInt4Format::Gptq, canonical_weight_name, tp_size);
}

void add_awq_marlin_repack_ops(
    LayoutPlan& plan,
    const HfConfig& hf,
    const TensorInfo& qweight_info,
    const TensorInfo& qzeros_info,
    const TensorInfo& scale_info,
    const std::string& raw_qweight,
    const std::string& raw_qzeros,
    const std::string& raw_scales,
    const std::string& canonical_w,
    const OfflineInt4ShardPlan& shard_plan,
    int tp_size)
{
    const auto local_qweight_shape = sharded_shape(
        qweight_info.shape, shard_plan.qweight_axis, tp_size, raw_qweight);
    const auto local_qzeros_shape = sharded_shape(
        qzeros_info.shape, shard_plan.qzeros_axis, tp_size, raw_qzeros);
    const auto local_scale_shape = sharded_shape(
        scale_info.shape, shard_plan.scale_axis, tp_size, raw_scales);

    const std::int64_t k_local = local_qweight_shape[0];
    const std::int64_t n_local = local_qweight_shape[1] * 8;
    if (k_local <= 0 || n_local <= 0 ||
        k_local % 16 != 0 ||
        n_local % 64 != 0 ||
        k_local % hf.quant_group_size != 0) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + raw_qweight +
            "' local shape is not compatible with Marlin repack");
    }
    const std::int64_t groups_local = k_local / hf.quant_group_size;
    if (local_qzeros_shape !=
            std::vector<std::int64_t>{groups_local, n_local / 8} ||
        local_scale_shape !=
            std::vector<std::int64_t>{groups_local, n_local}) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales TP slices do not match "
            "qweight/group_size for Marlin repack at '" + raw_qweight + "'");
    }

    const std::string canonical_s =
        pie_cuda_runtime_abi().quant_scale_inv_name(canonical_w);
    const std::string canonical_z = canonical_w + "_zero_point";
    const std::string tmp_qw = canonical_w + ".__awq_qweight";
    const std::string tmp_qz = canonical_w + ".__awq_qzeros";
    const std::string tmp_scales = canonical_w + ".__awq_scales";

    register_tensor_spec(
        plan, tmp_qw, qweight_info.dtype, local_qweight_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, tmp_qz, qzeros_info.dtype, local_qzeros_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, tmp_scales, scale_info.dtype, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);

    QuantSpec quant;
    quant.format = QuantFormat::AwqInt4;
    quant.granularity = QuantGranularity::PerGroup;
    quant.group_size = hf.quant_group_size;
    quant.channel_axis = 0;
    quant.scale_tensor = canonical_s;
    quant.zero_point_tensor = canonical_z;

    register_tensor_spec(
        plan, canonical_w, DType::INT4_PACKED,
        {k_local / 16, n_local * 8},
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        shard_plan.parallel,
        /*backing_tensor=*/{}, std::move(quant));
    register_tensor_spec(
        plan, canonical_s, DType::BF16, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        shard_plan.parallel);
    register_tensor_spec(
        plan, canonical_z, DType::INT32, local_qzeros_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        shard_plan.parallel);
    const LayoutExprId qw = source_realize_expr_from_info(
        plan, raw_qweight, qweight_info, tmp_qw,
        shard_plan.qweight_axis, tp_size);
    const LayoutExprId qz = source_realize_expr_from_info(
        plan, raw_qzeros, qzeros_info, tmp_qz,
        shard_plan.qzeros_axis, tp_size);
    const LayoutExprId scales = source_realize_expr_from_info(
        plan, raw_scales, scale_info, tmp_scales,
        shard_plan.scale_axis, tp_size);
    const LayoutExprId packed = reorder_realize_expr(
        plan, canonical_w, canonical_s, {qw, qz, scales},
        shard_plan.canonical_axis);
    (void)attach_metadata_expr(plan, canonical_w, packed);
    release_expr(
        plan, canonical_w + ".__drop_awq_sources",
        {qw, qz, scales});
    estimate_temporary_bytes(
        plan,
        tensor_nbytes(qweight_info.dtype, local_qweight_shape) +
        tensor_nbytes(qzeros_info.dtype, local_qzeros_shape) +
        tensor_nbytes(scale_info.dtype, local_scale_shape));
}

bool is_gptq_repack_companion(
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic,
    bool lowering_enabled)
{
    if (!lowering_enabled || hf.quant_method != "gptq") return false;

    constexpr const char* scale_suffix = ".scales";
    constexpr const char* zero_suffix = ".qzeros";
    constexpr const char* gidx_suffix = ".g_idx";

    const char* matched = nullptr;
    if (ends_with(semantic.raw_name, scale_suffix)) matched = scale_suffix;
    if (ends_with(semantic.raw_name, zero_suffix)) matched = zero_suffix;
    if (ends_with(semantic.raw_name, gidx_suffix)) matched = gidx_suffix;
    if (matched == nullptr) return false;

    const std::string prefix =
        semantic.raw_name.substr(0, semantic.raw_name.size() -
                                      std::strlen(matched));
    return loader.contains(prefix + ".qweight");
}

bool is_offline_int4_dequant_companion(
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic,
    bool lowering_enabled)
{
    if (!lowering_enabled) return false;
    if (hf.quant_method != "awq" && hf.quant_method != "gptq") return false;

    constexpr const char* scale_suffix = ".scales";
    constexpr const char* zero_suffix = ".qzeros";
    constexpr const char* gidx_suffix = ".g_idx";

    const char* matched = nullptr;
    if (ends_with(semantic.raw_name, scale_suffix)) matched = scale_suffix;
    if (ends_with(semantic.raw_name, zero_suffix)) matched = zero_suffix;
    if (hf.quant_method == "gptq" &&
        ends_with(semantic.raw_name, gidx_suffix)) {
        matched = gidx_suffix;
    }
    if (matched == nullptr) return false;

    const std::string prefix =
        semantic.raw_name.substr(0, semantic.raw_name.size() -
                                      std::strlen(matched));
    return loader.contains(prefix + ".qweight");
}

bool try_add_gptq_marlin_repack_weight(
    LayoutPlan& plan,
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic,
    const TensorInfo& info,
    int tp_size,
    std::unordered_set<std::string>& consumed_raw)
{
    constexpr const char* qweight_suffix = ".qweight";
    if (!ends_with(semantic.runtime_name, qweight_suffix)) return false;

    if (info.dtype != DType::INT32 || info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + semantic.raw_name +
            "' must be a 2-D int32 tensor");
    }

    const std::string raw_prefix =
        semantic.raw_name.substr(0, semantic.raw_name.size() -
                                      std::strlen(qweight_suffix));
    const std::string runtime_prefix =
        semantic.runtime_name.substr(0, semantic.runtime_name.size() -
                                          std::strlen(qweight_suffix));
    const std::string canonical_w = runtime_prefix + ".weight";
    const OfflineInt4ShardPlan shard_plan =
        gptq_shard_plan_for(canonical_w, tp_size);
    const std::string raw_scales = raw_prefix + ".scales";
    if (!loader.contains(raw_scales)) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + semantic.raw_name +
            "' is missing matching '.scales' tensor");
    }
    const TensorInfo& scale_info = loader.info(raw_scales);
    if (scale_info.dtype != DType::FP16 || scale_info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: GPTQ scales '" + raw_scales +
            "' must be a 2-D fp16 tensor for Marlin repack");
    }

    const std::int64_t k_full = info.shape[0] * 8;
    const std::int64_t n_full = info.shape[1];
    if (k_full <= 0 || n_full <= 0 || k_full % 16 != 0) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + semantic.raw_name +
            "' has unsupported packed shape");
    }
    if (k_full % hf.quant_group_size != 0) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + semantic.raw_name +
            "' K dimension is not divisible by group_size=" +
            std::to_string(hf.quant_group_size));
    }
    const std::int64_t expected_groups = k_full / hf.quant_group_size;
    if (scale_info.shape[0] != expected_groups ||
        scale_info.shape[1] != n_full) {
        throw std::runtime_error(
            "load schema: GPTQ scales '" + raw_scales +
            "' shape does not match qweight/group_size");
    }

    const std::string canonical_s =
        pie_cuda_runtime_abi().quant_scale_inv_name(runtime_prefix + ".weight");
    const auto local_qweight_shape = sharded_shape(
        info.shape, shard_plan.qweight_axis, tp_size, semantic.raw_name);
    const auto local_scale_shape = sharded_shape(
        scale_info.shape, shard_plan.scale_axis, tp_size, raw_scales);
    const std::int64_t k_local = local_qweight_shape[0] * 8;
    const std::int64_t n_local = local_qweight_shape[1];
    if (local_scale_shape[1] != n_local) {
        throw std::runtime_error(
            "load schema: GPTQ scale/qweight TP slices disagree for '" +
            semantic.raw_name + "'");
    }
    if (k_local % hf.quant_group_size != 0 ||
        local_scale_shape[0] != k_local / hf.quant_group_size) {
        throw std::runtime_error(
            "load schema: GPTQ row-parallel slice for '" + semantic.raw_name +
            "' does not preserve whole quantization groups");
    }
    const std::string tmp_qw = canonical_w + ".__gptq_qweight";
    const std::string tmp_scales = canonical_w + ".__gptq_scales";

    register_tensor_spec(
        plan, tmp_qw, info.dtype, local_qweight_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, tmp_scales, scale_info.dtype, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);

    QuantSpec quant;
    quant.format = QuantFormat::GptqInt4;
    quant.granularity = QuantGranularity::PerGroup;
    quant.group_size = hf.quant_group_size;
    quant.channel_axis = 0;
    quant.scale_tensor = canonical_s;

    register_tensor_spec(
        plan, canonical_w, DType::INT4_PACKED,
        {k_local / 16, n_local * 8},
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        shard_plan.parallel,
        /*backing_tensor=*/{}, std::move(quant));
    register_tensor_spec(
        plan, canonical_s, DType::BF16, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        shard_plan.parallel);
    const LayoutExprId qw = source_realize_expr_from_info(
        plan, semantic.raw_name, info, tmp_qw,
        shard_plan.qweight_axis, tp_size);
    const LayoutExprId scales = source_realize_expr_from_info(
        plan, raw_scales, scale_info, tmp_scales,
        shard_plan.scale_axis, tp_size);
    const LayoutExprId packed = reorder_realize_expr(
        plan, canonical_w, canonical_s, {qw, scales},
        shard_plan.canonical_axis);
    (void)attach_metadata_expr(plan, canonical_w, packed);
    release_expr(
        plan, canonical_w + ".__drop_gptq_sources",
        {qw, scales});
    estimate_temporary_bytes(
        plan,
        tensor_nbytes(info.dtype, local_qweight_shape) +
        tensor_nbytes(scale_info.dtype, local_scale_shape));

    consumed_raw.insert(semantic.raw_name);
    consumed_raw.insert(raw_scales);
    if (loader.contains(raw_prefix + ".qzeros")) {
        consumed_raw.insert(raw_prefix + ".qzeros");
    }
    if (loader.contains(raw_prefix + ".g_idx")) {
        consumed_raw.insert(raw_prefix + ".g_idx");
    }
    return true;
}

bool scale_dtype_supported_for_int4_dequant(DType dtype) {
    return dtype == DType::BF16 || dtype == DType::FP16 || dtype == DType::FP32;
}

void add_int4_dequant_ops(
    LayoutPlan& plan,
    const HfConfig& hf,
    const TensorInfo& qweight_info,
    const TensorInfo& qzeros_info,
    const TensorInfo& scale_info,
    const TensorInfo* gidx_info,
    const std::string& raw_qweight,
    const std::string& raw_qzeros,
    const std::string& raw_scales,
    const std::string& raw_gidx,
    const std::string& canonical_w,
    OfflineInt4Format format,
    const OfflineInt4ShardPlan& shard_plan,
    int tp_size)
{
    const bool is_awq = format == OfflineInt4Format::Awq;
    const QuantFormat quant_format =
        is_awq ? QuantFormat::AwqInt4 : QuantFormat::GptqInt4;
    const std::string quant_prefix = is_awq ? "awq" : "gptq";

    const auto local_qweight_shape = sharded_shape(
        qweight_info.shape, shard_plan.qweight_axis, tp_size, raw_qweight);
    const auto local_qzeros_shape = sharded_shape(
        qzeros_info.shape, shard_plan.qzeros_axis, tp_size, raw_qzeros);
    const auto local_scale_shape = sharded_shape(
        scale_info.shape, shard_plan.scale_axis, tp_size, raw_scales);

    const std::int64_t k_local =
        is_awq ? local_qweight_shape[0] : local_qweight_shape[0] * 8;
    const std::int64_t n_local =
        is_awq ? local_qweight_shape[1] * 8 : local_qweight_shape[1];
    if (k_local <= 0 || n_local <= 0 ||
        k_local % hf.quant_group_size != 0) {
        throw std::runtime_error(
            "load schema: " + quant_prefix + " qweight '" + raw_qweight +
            "' local shape is not compatible with group_size=" +
            std::to_string(hf.quant_group_size));
    }
    const std::int64_t groups_local = k_local / hf.quant_group_size;
    const std::vector<std::int64_t> expected_scale_shape =
        {groups_local, n_local};
    const std::vector<std::int64_t> expected_qzeros_shape =
        {groups_local, n_local / 8};
    if (n_local % 8 != 0) {
        throw std::runtime_error(
            "load schema: " + quant_prefix + " local N is not divisible "
            "by 8 for '" + raw_qweight + "'");
    }
    if (gidx_info == nullptr) {
        if (local_scale_shape != expected_scale_shape ||
            local_qzeros_shape != expected_qzeros_shape) {
            throw std::runtime_error(
                "load schema: " + quant_prefix + " qzeros/scales TP slices "
                "do not match qweight/group_size for '" + raw_qweight + "'");
        }
    } else {
        if (local_scale_shape.size() != 2 || local_qzeros_shape.size() != 2 ||
            local_scale_shape[1] != n_local ||
            local_qzeros_shape[1] != n_local / 8 ||
            local_scale_shape[0] != local_qzeros_shape[0] ||
            local_scale_shape[0] < groups_local) {
            throw std::runtime_error(
                "load schema: GPTQ act-order qzeros/scales slices do not "
                "cover local g_idx groups for '" + raw_qweight + "'");
        }
    }

    const std::string tmp_qw =
        canonical_w + ".__" + quant_prefix + "_qweight";
    const std::string tmp_qz =
        canonical_w + ".__" + quant_prefix + "_qzeros";
    const std::string tmp_scales =
        canonical_w + ".__" + quant_prefix + "_scales";
    const std::string tmp_gidx =
        canonical_w + ".__" + quant_prefix + "_g_idx";

    std::vector<LayoutExprId> dequant_inputs;
    std::vector<LayoutExprId> drop_inputs;
    if (gidx_info != nullptr) {
        const auto local_gidx_shape = sharded_shape(
            gidx_info->shape, shard_plan.gidx_axis, tp_size, raw_gidx);
        if (local_gidx_shape != std::vector<std::int64_t>{k_local}) {
            throw std::runtime_error(
                "load schema: GPTQ g_idx slice does not match local K for '" +
                raw_qweight + "'");
        }

        register_tensor_spec(
            plan, tmp_gidx, gidx_info->dtype, local_gidx_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
            shard_plan.parallel);
    }

    QuantSpec source_quant;
    source_quant.format = quant_format;
    source_quant.granularity = QuantGranularity::PerGroup;
    source_quant.group_size = hf.quant_group_size;
    source_quant.channel_axis = 0;
    source_quant.scale_tensor = tmp_scales;
    source_quant.zero_point_tensor = tmp_qz;

    register_tensor_spec(
        plan, tmp_qw, qweight_info.dtype, local_qweight_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel, /*backing_tensor=*/{}, std::move(source_quant));
    register_tensor_spec(
        plan, tmp_qz, qzeros_info.dtype, local_qzeros_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, tmp_scales, scale_info.dtype, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, canonical_w, DType::BF16, {n_local, k_local},
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        shard_plan.parallel);

    const LayoutExprId qw = source_realize_expr_from_info(
        plan, raw_qweight, qweight_info, tmp_qw,
        shard_plan.qweight_axis, tp_size);
    const LayoutExprId qz = source_realize_expr_from_info(
        plan, raw_qzeros, qzeros_info, tmp_qz,
        shard_plan.qzeros_axis, tp_size);
    const LayoutExprId scales = source_realize_expr_from_info(
        plan, raw_scales, scale_info, tmp_scales,
        shard_plan.scale_axis, tp_size);
    dequant_inputs = {qw, qz, scales};
    drop_inputs = dequant_inputs;
    if (gidx_info != nullptr) {
        const LayoutExprId gidx = source_realize_expr_from_info(
            plan, raw_gidx, *gidx_info, tmp_gidx,
            shard_plan.gidx_axis, tp_size);
        dequant_inputs.push_back(gidx);
        drop_inputs.push_back(gidx);
    }
    (void)decode_realize_expr(
        plan, canonical_w, std::move(dequant_inputs));
    release_expr(
        plan, canonical_w + ".__drop_" + quant_prefix + "_sources",
        std::move(drop_inputs));

    const std::uint64_t scale_cast_scratch =
        scale_info.dtype == DType::BF16
            ? 0
            : tensor_nbytes(DType::BF16, local_scale_shape);
    std::uint64_t gidx_bytes = 0;
    if (gidx_info != nullptr) {
        gidx_bytes = tensor_nbytes(
            gidx_info->dtype,
            sharded_shape(gidx_info->shape, shard_plan.gidx_axis,
                          tp_size, raw_gidx));
    }
    estimate_temporary_bytes(
        plan,
        tensor_nbytes(qweight_info.dtype, local_qweight_shape) +
        tensor_nbytes(qzeros_info.dtype, local_qzeros_shape) +
        tensor_nbytes(scale_info.dtype, local_scale_shape) +
        scale_cast_scratch + gidx_bytes);
}

bool try_add_awq_dequant_weight(
    LayoutPlan& plan,
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic,
    const TensorInfo& info,
    int tp_size,
    std::unordered_set<std::string>& consumed_raw)
{
    constexpr const char* qweight_suffix = ".qweight";
    if (!ends_with(semantic.runtime_name, qweight_suffix)) return false;
    if (info.dtype != DType::INT32 || info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + semantic.raw_name +
            "' must be a 2-D int32 tensor");
    }

    const std::string raw_prefix =
        semantic.raw_name.substr(0, semantic.raw_name.size() -
                                      std::strlen(qweight_suffix));
    const std::string runtime_prefix =
        semantic.runtime_name.substr(0, semantic.runtime_name.size() -
                                          std::strlen(qweight_suffix));
    const std::string raw_qzeros = raw_prefix + ".qzeros";
    const std::string raw_scales = raw_prefix + ".scales";
    if (!loader.contains(raw_qzeros) || !loader.contains(raw_scales)) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + semantic.raw_name +
            "' is missing qzeros/scales companions");
    }
    const TensorInfo& qzeros_info = loader.info(raw_qzeros);
    const TensorInfo& scale_info = loader.info(raw_scales);
    if (qzeros_info.dtype != DType::INT32 || qzeros_info.shape.size() != 2 ||
        scale_info.shape.size() != 2 ||
        !scale_dtype_supported_for_int4_dequant(scale_info.dtype)) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales for '" + semantic.raw_name +
            "' have unsupported dtype or rank");
    }

    const std::int64_t k_full = info.shape[0];
    const std::int64_t n_full = info.shape[1] * 8;
    if (k_full <= 0 || n_full <= 0 ||
        k_full % hf.quant_group_size != 0) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + semantic.raw_name +
            "' shape is incompatible with group_size=" +
            std::to_string(hf.quant_group_size));
    }
    const std::int64_t groups = k_full / hf.quant_group_size;
    if (qzeros_info.shape != std::vector<std::int64_t>{groups, n_full / 8} ||
        scale_info.shape != std::vector<std::int64_t>{groups, n_full}) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales shape does not match qweight for '" +
            semantic.raw_name + "'");
    }

    const std::string canonical_w = runtime_prefix + ".weight";
    const OfflineInt4ShardPlan shard_plan =
        offline_int4_shard_plan_for(
            OfflineInt4Format::Awq, canonical_w, tp_size);
    add_int4_dequant_ops(
        plan, hf, info, qzeros_info, scale_info, nullptr,
        semantic.raw_name, raw_qzeros, raw_scales, {},
        canonical_w, OfflineInt4Format::Awq, shard_plan, tp_size);

    consumed_raw.insert(semantic.raw_name);
    consumed_raw.insert(raw_qzeros);
    consumed_raw.insert(raw_scales);
    return true;
}

bool try_add_awq_marlin_repack_weight(
    LayoutPlan& plan,
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic,
    const TensorInfo& info,
    int tp_size,
    std::unordered_set<std::string>& consumed_raw)
{
    constexpr const char* qweight_suffix = ".qweight";
    if (!ends_with(semantic.runtime_name, qweight_suffix)) return false;
    if (info.dtype != DType::INT32 || info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + semantic.raw_name +
            "' must be a 2-D int32 tensor");
    }

    const std::string raw_prefix =
        semantic.raw_name.substr(0, semantic.raw_name.size() -
                                      std::strlen(qweight_suffix));
    const std::string runtime_prefix =
        semantic.runtime_name.substr(0, semantic.runtime_name.size() -
                                          std::strlen(qweight_suffix));
    const std::string raw_qzeros = raw_prefix + ".qzeros";
    const std::string raw_scales = raw_prefix + ".scales";
    if (!loader.contains(raw_qzeros) || !loader.contains(raw_scales)) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + semantic.raw_name +
            "' is missing qzeros/scales companions");
    }
    const TensorInfo& qzeros_info = loader.info(raw_qzeros);
    const TensorInfo& scale_info = loader.info(raw_scales);
    if (qzeros_info.dtype != DType::INT32 || qzeros_info.shape.size() != 2 ||
        scale_info.shape.size() != 2 ||
        !scale_dtype_supported_for_int4_dequant(scale_info.dtype)) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales for '" + semantic.raw_name +
            "' have unsupported dtype or rank");
    }

    const std::int64_t k_full = info.shape[0];
    const std::int64_t n_full = info.shape[1] * 8;
    if (k_full <= 0 || n_full <= 0 ||
        k_full % hf.quant_group_size != 0 ||
        k_full % 16 != 0 || n_full % 64 != 0) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + semantic.raw_name +
            "' shape is incompatible with Marlin repack");
    }
    const std::int64_t groups = k_full / hf.quant_group_size;
    if (qzeros_info.shape != std::vector<std::int64_t>{groups, n_full / 8} ||
        scale_info.shape != std::vector<std::int64_t>{groups, n_full}) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales shape does not match qweight for '" +
            semantic.raw_name + "'");
    }

    const std::string canonical_w = runtime_prefix + ".weight";
    const OfflineInt4ShardPlan shard_plan =
        offline_int4_shard_plan_for(
            OfflineInt4Format::Awq, canonical_w, tp_size);
    add_awq_marlin_repack_ops(
        plan, hf, info, qzeros_info, scale_info,
        semantic.raw_name, raw_qzeros, raw_scales,
        canonical_w, shard_plan, tp_size);

    consumed_raw.insert(semantic.raw_name);
    consumed_raw.insert(raw_qzeros);
    consumed_raw.insert(raw_scales);
    return true;
}

bool try_add_gptq_dequant_weight(
    LayoutPlan& plan,
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticTensor& semantic,
    const TensorInfo& info,
    int tp_size,
    std::unordered_set<std::string>& consumed_raw)
{
    constexpr const char* qweight_suffix = ".qweight";
    if (!ends_with(semantic.runtime_name, qweight_suffix)) return false;
    if (info.dtype != DType::INT32 || info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + semantic.raw_name +
            "' must be a 2-D int32 tensor");
    }

    const std::string raw_prefix =
        semantic.raw_name.substr(0, semantic.raw_name.size() -
                                      std::strlen(qweight_suffix));
    const std::string runtime_prefix =
        semantic.runtime_name.substr(0, semantic.runtime_name.size() -
                                          std::strlen(qweight_suffix));
    const std::string raw_qzeros = raw_prefix + ".qzeros";
    const std::string raw_scales = raw_prefix + ".scales";
    const std::string raw_gidx = raw_prefix + ".g_idx";
    if (!loader.contains(raw_qzeros) || !loader.contains(raw_scales)) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + semantic.raw_name +
            "' is missing qzeros/scales companions");
    }
    const TensorInfo& qzeros_info = loader.info(raw_qzeros);
    const TensorInfo& scale_info = loader.info(raw_scales);
    if (qzeros_info.dtype != DType::INT32 || qzeros_info.shape.size() != 2 ||
        scale_info.shape.size() != 2 ||
        !scale_dtype_supported_for_int4_dequant(scale_info.dtype)) {
        throw std::runtime_error(
            "load schema: GPTQ qzeros/scales for '" + semantic.raw_name +
            "' have unsupported dtype or rank");
    }

    const std::int64_t k_full = info.shape[0] * 8;
    const std::int64_t n_full = info.shape[1];
    if (k_full <= 0 || n_full <= 0 ||
        k_full % hf.quant_group_size != 0 ||
        n_full % 8 != 0) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + semantic.raw_name +
            "' shape is incompatible with group_size=" +
            std::to_string(hf.quant_group_size));
    }
    const std::int64_t groups = k_full / hf.quant_group_size;
    if (qzeros_info.shape != std::vector<std::int64_t>{groups, n_full / 8} ||
        scale_info.shape != std::vector<std::int64_t>{groups, n_full}) {
        throw std::runtime_error(
            "load schema: GPTQ qzeros/scales shape does not match qweight for '" +
            semantic.raw_name + "'");
    }

    const TensorInfo* gidx_info = nullptr;
    if (hf.quant_desc_act) {
        if (!loader.contains(raw_gidx)) {
            throw std::runtime_error(
                "load schema: GPTQ desc_act qweight '" + semantic.raw_name +
                "' is missing g_idx companion");
        }
        gidx_info = &loader.info(raw_gidx);
        if (gidx_info->dtype != DType::INT32 ||
            gidx_info->shape != std::vector<std::int64_t>{k_full}) {
            throw std::runtime_error(
                "load schema: GPTQ g_idx shape does not match qweight for '" +
                semantic.raw_name + "'");
        }
    }

    const std::string canonical_w = runtime_prefix + ".weight";
    OfflineInt4ShardPlan shard_plan =
        offline_int4_shard_plan_for(
            OfflineInt4Format::Gptq, canonical_w, tp_size);
    if (hf.quant_desc_act && shard_plan.canonical_axis == 1) {
        // Act-order row shards carry local qweight/g_idx, but g_idx values
        // are global group ids. Keep qzeros/scales unsharded so the generic
        // dequant kernel can dereference those global ids directly.
        shard_plan.qzeros_axis = -1;
        shard_plan.scale_axis = -1;
    }
    add_int4_dequant_ops(
        plan, hf, info, qzeros_info, scale_info, gidx_info,
        semantic.raw_name, raw_qzeros, raw_scales, raw_gidx,
        canonical_w, OfflineInt4Format::Gptq, shard_plan, tp_size);

    consumed_raw.insert(semantic.raw_name);
    consumed_raw.insert(raw_qzeros);
    consumed_raw.insert(raw_scales);
    if (loader.contains(raw_gidx)) {
        consumed_raw.insert(raw_gidx);
    }
    return true;
}

std::uint64_t tensor_nbytes(DType dtype, const std::vector<std::int64_t>& shape) {
    std::uint64_t numel = 1;
    for (const auto dim : shape) {
        numel *= static_cast<std::uint64_t>(dim);
    }
    return numel * static_cast<std::uint64_t>(dtype_bytes(dtype));
}

std::uint64_t tensor_nbytes(const TensorDecl& spec) {
    return tensor_nbytes(spec.dtype, spec.shape);
}

bool try_add_per_expert_moe_fusion(
    LayoutPlan& plan,
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticGroup& group,
    std::unordered_set<std::string>& consumed_raw,
    int tp_size)
{
    if (group.kind != SemanticGroupKind::PerExpertMoe) return false;
    if (group.raw_names.empty() || group.raw_names.size() % 3 != 0 ||
        group.raw_roles.size() != group.raw_names.size() ||
        group.runtime_names.size() != 2) {
        throw std::runtime_error(
            "load schema: per-expert MoE group has wrong arity at '" +
            group.runtime_base + "'");
    }
    const std::string& raw_gate0 = group.raw_names[0];
    if (consumed_raw.contains(raw_gate0)) return true;

    int expert_count = static_cast<int>(group.raw_names.size() / 3);
    if (expert_count <= 0) {
        throw std::runtime_error(
            "load schema: could not determine MoE expert count for '" +
            group.runtime_base + "'");
    }
    if (hf.num_experts > 0 && hf.num_experts != expert_count) {
        throw std::runtime_error(
            "load schema: per-expert MoE group count disagrees with config at '" +
            group.runtime_base + "'");
    }

    std::vector<LayoutExprId> fuse_sources;
    fuse_sources.reserve(static_cast<std::size_t>(expert_count) * 3);

    std::vector<std::int64_t> gate_shape0;
    std::vector<std::int64_t> down_shape0;
    for (int e = 0; e < expert_count; ++e) {
        const std::size_t base = static_cast<std::size_t>(e) * 3;
        if (group.raw_roles[base] != SemanticRole::MoeExpertGate ||
            group.raw_roles[base + 1] != SemanticRole::MoeExpertUp ||
            group.raw_roles[base + 2] != SemanticRole::MoeExpertDown) {
            throw std::runtime_error(
                "load schema: per-expert MoE source roles are not gate/up/down "
                "triples at '" + group.runtime_base + "'");
        }
        const std::string& raw_gate = group.raw_names[base];
        const std::string& raw_up = group.raw_names[base + 1];
        const std::string& raw_down = group.raw_names[base + 2];
        if (!loader.contains(raw_gate) ||
            !loader.contains(raw_up) ||
            !loader.contains(raw_down)) {
            throw std::runtime_error(
                "load schema: incomplete per-expert MoE group under '" +
                group.runtime_base + "' at expert " + std::to_string(e));
        }

        const TensorInfo& gate_info = loader.info(raw_gate);
        const TensorInfo& up_info = loader.info(raw_up);
        const TensorInfo& down_info = loader.info(raw_down);
        if (gate_info.shape.size() != 2 ||
            up_info.shape.size() != 2 ||
            down_info.shape.size() != 2) {
            throw std::runtime_error(
                "load schema: per-expert MoE fusion expects 2-D expert "
                "weights under '" + group.runtime_base + "'");
        }

        const auto local_gate_shape =
            sharded_shape(gate_info.shape, 0, tp_size, raw_gate);
        const auto local_up_shape =
            sharded_shape(up_info.shape, 0, tp_size, raw_up);
        const auto local_down_shape =
            sharded_shape(down_info.shape, 1, tp_size, raw_down);
        if (local_gate_shape != local_up_shape ||
            local_down_shape[0] != local_gate_shape[1] ||
            local_down_shape[1] != local_gate_shape[0]) {
            throw std::runtime_error(
                "load schema: per-expert MoE gate/up/down shapes do not "
                "align under '" + group.runtime_base + "'");
        }
        if (e == 0) {
            gate_shape0 = local_gate_shape;
            down_shape0 = local_down_shape;
        } else if (local_gate_shape != gate_shape0 ||
                   local_down_shape != down_shape0) {
            throw std::runtime_error(
                "load schema: per-expert MoE shapes are not uniform under '" +
                group.runtime_base + "'");
        }

        if (gate_info.dtype != DType::BF16 ||
            up_info.dtype != DType::BF16 ||
            down_info.dtype != DType::BF16) {
            throw std::runtime_error(
                "load schema: direct MoE expert fusion currently requires "
                "bf16 sources under '" + group.runtime_base + "'");
        }

        TensorDecl gate_decl;
        gate_decl.name = raw_gate;
        gate_decl.dtype = gate_info.dtype;
        gate_decl.shape = local_gate_shape;
        gate_decl.layout = TensorLayoutKind::Dense;
        gate_decl.ownership = TensorOwnershipKind::Temporary;
        gate_decl.parallel = TensorParallelKind::Column;
        LayoutExprId gate = source_expr_from_info(plan, raw_gate, gate_info);
        gate = partition_expr(plan, gate, gate_decl, 0, tp_size);

        TensorDecl up_decl = gate_decl;
        up_decl.name = raw_up;
        LayoutExprId up = source_expr_from_info(plan, raw_up, up_info);
        up = partition_expr(plan, up, up_decl, 0, tp_size);

        TensorDecl down_decl;
        down_decl.name = raw_down;
        down_decl.dtype = down_info.dtype;
        down_decl.shape = local_down_shape;
        down_decl.layout = TensorLayoutKind::Dense;
        down_decl.ownership = TensorOwnershipKind::Temporary;
        down_decl.parallel = TensorParallelKind::Row;
        LayoutExprId down = source_expr_from_info(plan, raw_down, down_info);
        down = partition_expr(plan, down, down_decl, 1, tp_size);
        fuse_sources.push_back(gate);
        fuse_sources.push_back(up);
        fuse_sources.push_back(down);

        consumed_raw.insert(raw_gate);
        consumed_raw.insert(raw_up);
        consumed_raw.insert(raw_down);
    }

    const std::string& gate_up_name =
        group_runtime_name_for_role(group, SemanticRole::MoeExpertsGateUp);
    const std::string& down_name =
        group_runtime_name_for_role(group, SemanticRole::MoeExpertsDown);
    const std::int64_t E = expert_count;
    register_tensor_spec(
        plan, gate_up_name, DType::BF16,
        {E, 2 * gate_shape0[0], gate_shape0[1]},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, down_name, DType::BF16,
        {E, down_shape0[0], down_shape0[1]},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);

    TensorDecl gate_up_decl = tensor_decl_for(plan, gate_up_name);
    LayoutExpr stack = make_expr(LayoutExprKind::Stack, gate_up_decl);
    stack.inputs = std::move(fuse_sources);
    stack.runtime_name = gate_up_name;
    stack.secondary_runtime_name = down_name;
    stack.axis = 0;
    const LayoutExprId stacked = add_expr(plan, std::move(stack));
    realize_expr(plan, gate_up_name, stacked, gate_up_decl);
    TensorDecl down_decl = tensor_decl_for(plan, down_name);
    realize_expr(plan, down_name, stacked, std::move(down_decl));
    return true;
}

void finalize_memory_plan(LayoutPlan& plan) {
    const std::uint64_t conservative_temp_floor =
        plan.memory.max_temporary_bytes;

    std::uint64_t persistent_bytes = 0;
    std::uint64_t max_temp_tensor_bytes = 0;
    for (const auto& [name, spec] : plan.tensors) {
        (void)name;
        if (spec.ownership == TensorOwnershipKind::Owned) {
            persistent_bytes += tensor_nbytes(spec);
        } else if (spec.ownership == TensorOwnershipKind::Temporary) {
            max_temp_tensor_bytes =
                std::max(max_temp_tensor_bytes, tensor_nbytes(spec));
        }
    }

    plan.memory.persistent_bytes = persistent_bytes;
    plan.memory.max_temporary_bytes =
        std::max(conservative_temp_floor, max_temp_tensor_bytes);
    plan.memory.estimated_peak_bytes =
        plan.memory.persistent_bytes + plan.memory.max_temporary_bytes;
}

bool can_use_native_dense_algebra_plan(
    const HfConfig& hf,
    const Config& boot_cfg,
    const CheckpointSource& loader,
    const SemanticGraph& semantic_graph,
    const ModelSchema& schema)
{
    if (!schema.pack_dense_qkv_and_gate_up ||
        schema.unfuse_phi3_for_tp ||
        schema.shard_fused_moe_experts_for_tp ||
        schema.fuse_per_expert_moe_after_load) {
        return false;
    }
    if (!hf.quant_method.empty() ||
        !boot_cfg.model.runtime_quant.empty()) {
        return false;
    }
    for (const auto& tensor : semantic_graph.tensors) {
        if (!loader.contains(tensor.raw_name) ||
            loader.info(tensor.raw_name).dtype != DType::BF16) {
            return false;
        }
    }
    for (const auto& group : semantic_graph.groups) {
        if (group.kind != SemanticGroupKind::PackedQkv &&
            group.kind != SemanticGroupKind::PackedGateUp) {
            return false;
        }
    }
    return true;
}

enum class Mxfp4ExpertProjection {
    GateUpInterleaved,
    Down,
};

struct Mxfp4ExpertGroup {
    Mxfp4ExpertProjection projection = Mxfp4ExpertProjection::Down;
    std::string raw_base;
    std::string runtime_base;
    std::string stem;
    std::string raw_blocks;
    std::string raw_scales;
    std::string raw_bias;
    std::string group_prefix;
    TensorInfo blocks;
    TensorInfo scales;
    TensorInfo bias;
    std::int64_t experts = 0;
    std::int64_t out_dim = 0;
    std::int64_t in_dim = 0;
    int shard_axis = -1;

    bool is_gate_up() const noexcept {
        return projection == Mxfp4ExpertProjection::GateUpInterleaved;
    }

    std::vector<std::int64_t> full_bf16_shape() const {
        return {experts, out_dim, in_dim};
    }
};

bool try_describe_gpt_oss_mxfp4_group(
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticGroup& semantic_group,
    int tp_size,
    Mxfp4ExpertGroup& out)
{
    if (hf.model_type != "gpt_oss" || hf.quant_method != "mxfp4") return false;
    if (semantic_group.kind != SemanticGroupKind::GptOssMxfp4) return false;
    if (semantic_group.raw_names.size() != 3 ||
        semantic_group.runtime_names.size() != 3 ||
        semantic_group.raw_roles.size() != 3 ||
        semantic_group.runtime_roles.size() != 3 ||
        semantic_group.raw_roles[0] != SemanticRole::QuantPackedData ||
        semantic_group.raw_roles[1] != SemanticRole::QuantScale ||
        semantic_group.raw_roles[2] != SemanticRole::Bias) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 group has wrong declaration at '" +
            semantic_group.runtime_base + "'");
    }

    const bool is_gate_up =
        semantic_group.runtime_base.find(".gate_up_proj") != std::string::npos;
    const bool is_down =
        semantic_group.runtime_base.find(".down_proj") != std::string::npos;
    if (!is_gate_up && !is_down) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 group has unknown projection at '" +
            semantic_group.runtime_base + "'");
    }

    Mxfp4ExpertGroup group;
    group.projection = is_gate_up
        ? Mxfp4ExpertProjection::GateUpInterleaved
        : Mxfp4ExpertProjection::Down;
    group.raw_base = semantic_group.raw_names[0].substr(
        0, semantic_group.raw_names[0].size() - std::strlen("_blocks"));
    const std::string marker =
        is_gate_up ? ".mlp.experts.gate_up_proj" : ".mlp.experts.down_proj";
    const auto marker_pos = semantic_group.runtime_base.rfind(marker);
    if (marker_pos == std::string::npos) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 group base is inconsistent at '" +
            semantic_group.runtime_base + "'");
    }
    group.runtime_base = semantic_group.runtime_base.substr(0, marker_pos);
    group.stem = is_gate_up ? "gate_up_proj" : "down_proj";
    group.raw_blocks = semantic_group.raw_names[0];
    group.raw_scales = semantic_group.raw_names[1];
    group.raw_bias = semantic_group.raw_names[2];
    group.group_prefix = semantic_group.runtime_base;

    if (!loader.contains(group.raw_blocks) ||
        !loader.contains(group.raw_scales) ||
        !loader.contains(group.raw_bias)) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 expert tensor group is incomplete at '" +
            group.raw_base + ".mlp.experts." + group.stem + "'");
    }

    group.blocks = loader.info(group.raw_blocks);
    group.scales = loader.info(group.raw_scales);
    group.bias = loader.info(group.raw_bias);
    if (group.blocks.dtype != DType::UINT8 ||
        group.scales.dtype != DType::UINT8 ||
        group.bias.dtype != DType::BF16) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 group has unexpected dtypes at '" +
            group.raw_base + ".mlp.experts." + group.stem + "'");
    }
    if (group.blocks.shape.size() != 4 || group.scales.shape.size() != 3 ||
        group.blocks.shape[0] != group.scales.shape[0] ||
        group.blocks.shape[1] != group.scales.shape[1] ||
        group.blocks.shape[2] != group.scales.shape[2] ||
        group.blocks.shape[3] != 16) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 packed/scales shape mismatch at '" +
            group.raw_blocks + "'");
    }

    group.experts = group.blocks.shape[0];
    group.out_dim = group.blocks.shape[1];
    group.in_dim = group.blocks.shape[2] * 32;
    group.shard_axis = (tp_size > 1) ? (group.is_gate_up() ? 1 : 2) : -1;
    out = std::move(group);
    return true;
}

void validate_gpt_oss_mxfp4_runtime_shape(
    const HfConfig& hf,
    const Mxfp4ExpertGroup& group,
    int tp_size,
    const char* lowering_name)
{
    if (group.is_gate_up()) {
        if (group.out_dim % 2 != 0 || (group.out_dim / 2) % tp_size != 0) {
            throw std::runtime_error(
                "load schema: GPT-OSS " + std::string(lowering_name) +
                " gate/up intermediate axis is not divisible by tp_size at '" +
                group.raw_blocks + "'");
        }
        if (group.bias.shape !=
            std::vector<std::int64_t>{group.experts, group.out_dim}) {
            throw std::runtime_error(
                "load schema: GPT-OSS gate/up bias shape mismatch at '" +
                group.raw_bias + "'");
        }
    } else {
        if (group.out_dim != hf.hidden_size || group.in_dim % tp_size != 0) {
            throw std::runtime_error(
                "load schema: GPT-OSS " + std::string(lowering_name) +
                " down projection shape cannot be tensor-parallel sharded at '" +
                group.raw_blocks + "'");
        }
        if (group.bias.shape !=
            std::vector<std::int64_t>{group.experts, group.out_dim}) {
            throw std::runtime_error(
                "load schema: GPT-OSS down bias shape mismatch at '" +
                group.raw_bias + "'");
        }
    }
}

void lower_mxfp4_expert_group_routed_dequant(
    LayoutPlan& plan,
    const HfConfig& hf,
    const Mxfp4ExpertGroup& group,
    int tp_size)
{
    validate_gpt_oss_mxfp4_runtime_shape(
        hf, group, tp_size, "routed MXFP4 dequant");

    const std::string weight_name = group.group_prefix + ".weight";
    const std::string scale_name = group.group_prefix + ".weight_scale";
    const std::string bias_name = group.group_prefix + ".bias";
    const int bias_shard_axis = (tp_size > 1 && group.is_gate_up()) ? 1 : -1;

    QuantSpec quant;
    quant.format = QuantFormat::Mxfp4E2M1E8M0;
    quant.granularity = QuantGranularity::PerGroup;
    quant.group_size = 32;
    quant.channel_axis = 2;
    quant.scale_tensor = scale_name;

    register_tensor_spec(
        plan, weight_name, DType::UINT8,
        sharded_shape(group.blocks.shape, group.shard_axis, tp_size, weight_name),
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert, /*backing_tensor=*/{}, std::move(quant));
    register_tensor_spec(
        plan, scale_name, DType::UINT8,
        sharded_shape(group.scales.shape, group.shard_axis, tp_size, scale_name),
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, bias_name, DType::BF16,
        sharded_shape(group.bias.shape, bias_shard_axis, tp_size, bias_name),
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);

    const LayoutExprId weight =
        source_realize_expr(plan, group.raw_blocks, weight_name,
                            group.shard_axis, tp_size);
    (void)source_realize_expr(plan, group.raw_scales, scale_name,
                              group.shard_axis, tp_size);
    (void)source_realize_expr(plan, group.raw_bias, bias_name,
                              bias_shard_axis, tp_size);
    (void)attach_metadata_expr(plan, weight_name, weight);
}

void lower_mxfp4_gate_up_group_to_bf16(
    LayoutPlan& plan,
    const Mxfp4ExpertGroup& group,
    int tp_size,
    LayoutExprId blocks_expr,
    LayoutExprId scales_expr)
{
    const std::vector<std::int64_t> full_bf16_shape = group.full_bf16_shape();
    const std::int64_t I_local = (group.out_dim / 2) / tp_size;
    const std::vector<std::int64_t> expert_w_shape =
        {group.experts, I_local, group.in_dim};
    const std::vector<std::int64_t> expert_b_shape =
        {group.experts, I_local};
    const std::string gate_w =
        group.runtime_base + ".mlp.experts.gate_proj.weight";
    const std::string up_w =
        group.runtime_base + ".mlp.experts.up_proj.weight";
    const std::string gate_b =
        group.runtime_base + ".mlp.experts.gate_proj.bias";
    const std::string up_b =
        group.runtime_base + ".mlp.experts.up_proj.bias";
    const std::string bf16_tmp = group.group_prefix + ".__mxfp4_bf16";

    register_tensor_spec(
        plan, gate_w, DType::BF16, expert_w_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, up_w, DType::BF16, expert_w_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, bf16_tmp, DType::BF16, full_bf16_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Temporary,
        TensorParallelKind::Expert);

    const LayoutExprId bf16_expr =
        decode_realize_expr(plan, bf16_tmp, {blocks_expr, scales_expr});
    (void)unzip_realize_expr(
        plan, gate_w, up_w, bf16_expr, group.shard_axis);

    const std::string bias_tmp = group.group_prefix + ".__gate_up_bias";
    register_tensor_spec(
        plan, bias_tmp, group.bias.dtype, group.bias.shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, gate_b, DType::BF16, expert_b_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, up_b, DType::BF16, expert_b_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    const LayoutExprId bias_expr =
        source_realize_expr(plan, group.raw_bias, bias_tmp, -1, tp_size);
    (void)unzip_realize_expr(
        plan, gate_b, up_b, bias_expr, group.shard_axis);

    release_expr(
        plan, group.group_prefix + ".__drop",
        {blocks_expr, scales_expr, bf16_expr, bias_expr});
    estimate_temporary_bytes(
        plan,
        group.blocks.nbytes + group.scales.nbytes + group.bias.nbytes +
        tensor_nbytes(DType::BF16, full_bf16_shape));
}

void lower_mxfp4_down_group_to_bf16(
    LayoutPlan& plan,
    const Mxfp4ExpertGroup& group,
    int tp_size,
    LayoutExprId blocks_expr,
    LayoutExprId scales_expr)
{
    const std::vector<std::int64_t> full_bf16_shape = group.full_bf16_shape();
    const std::int64_t I_local = group.in_dim / tp_size;
    const std::string down_w =
        group.runtime_base + ".mlp.experts.down_proj.weight";
    const std::string down_b =
        group.runtime_base + ".mlp.experts.down_proj.bias";

    register_tensor_spec(
        plan, down_w, DType::BF16, {group.experts, group.out_dim, I_local},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    constexpr LayoutExprId invalid_expr = static_cast<LayoutExprId>(-1);
    LayoutExprId bf16_expr = invalid_expr;
    if (tp_size == 1) {
        (void)decode_realize_expr(plan, down_w, {blocks_expr, scales_expr});
    } else {
        const std::string bf16_tmp = group.group_prefix + ".__mxfp4_bf16";
        register_tensor_spec(
            plan, bf16_tmp, DType::BF16, full_bf16_shape,
            TensorLayoutKind::Grouped, TensorOwnershipKind::Temporary,
            TensorParallelKind::Expert);
        bf16_expr =
            decode_realize_expr(plan, bf16_tmp, {blocks_expr, scales_expr});
        TensorDecl down_decl = tensor_decl_for(plan, down_w);
        const LayoutExprId selected = select_expr(
            plan, bf16_expr, down_decl, /*axis=*/2,
            /*start=*/0, I_local, group.shard_axis);
        (void)realize_expr(plan, down_w, selected, std::move(down_decl));
    }

    register_tensor_spec(
        plan, down_b, DType::BF16, group.bias.shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    (void)source_realize_expr(plan, group.raw_bias, down_b, -1, tp_size);

    std::vector<LayoutExprId> drop_inputs = {blocks_expr, scales_expr};
    if (bf16_expr != invalid_expr) {
        drop_inputs.push_back(bf16_expr);
    }
    release_expr(plan, group.group_prefix + ".__drop", std::move(drop_inputs));
    estimate_temporary_bytes(
        plan,
        group.blocks.nbytes + group.scales.nbytes +
        (tp_size > 1 ? tensor_nbytes(DType::BF16, full_bf16_shape) : 0));
}

void lower_mxfp4_expert_group_to_bf16(
    LayoutPlan& plan,
    const HfConfig& hf,
    const Mxfp4ExpertGroup& group,
    int tp_size)
{
    validate_gpt_oss_mxfp4_runtime_shape(
        hf, group, tp_size, "BF16 fallback");

    const std::string blocks_tmp = group.group_prefix + ".__mxfp4_blocks";
    const std::string scales_tmp = group.group_prefix + ".__mxfp4_scales";
    register_tensor_spec(
        plan, blocks_tmp, group.blocks.dtype, group.blocks.shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, scales_tmp, group.scales.dtype, group.scales.shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        TensorParallelKind::Expert);
    const LayoutExprId blocks_expr =
        source_realize_expr(plan, group.raw_blocks, blocks_tmp, -1, tp_size);
    const LayoutExprId scales_expr =
        source_realize_expr(plan, group.raw_scales, scales_tmp, -1, tp_size);

    if (group.is_gate_up()) {
        lower_mxfp4_gate_up_group_to_bf16(
            plan, group, tp_size, blocks_expr, scales_expr);
    } else {
        lower_mxfp4_down_group_to_bf16(
            plan, group, tp_size, blocks_expr, scales_expr);
    }
}

void mark_mxfp4_group_consumed(
    const Mxfp4ExpertGroup& group,
    std::unordered_set<std::string>& consumed_raw)
{
    consumed_raw.insert(group.raw_blocks);
    consumed_raw.insert(group.raw_scales);
    consumed_raw.insert(group.raw_bias);
}

bool try_add_gpt_oss_mxfp4_expert(
    LayoutPlan& plan,
    const HfConfig& hf,
    const CheckpointSource& loader,
    const SemanticGroup& semantic_group,
    std::unordered_set<std::string>& consumed_raw,
    int tp_size,
    Mxfp4MoeLowering lowering)
{
    Mxfp4ExpertGroup group;
    if (!try_describe_gpt_oss_mxfp4_group(
            hf, loader, semantic_group, tp_size, group)) {
        return false;
    }
    if (consumed_raw.contains(group.raw_blocks)) return true;

    if (lowering == Mxfp4MoeLowering::RoutedDequant) {
        lower_mxfp4_expert_group_routed_dequant(plan, hf, group, tp_size);
    } else if (lowering == Mxfp4MoeLowering::NativeGemm) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 native GEMM target was selected, "
            "but no native MXFP4 MoE GEMM backend is registered yet");
    } else {
        lower_mxfp4_expert_group_to_bf16(plan, hf, group, tp_size);
    }

    mark_mxfp4_group_consumed(group, consumed_raw);
    return true;
}

}  // namespace

ModelSchema resolve_model_schema(
    const HfConfig& hf,
    const Config& boot_cfg,
    int tp_size)
{
    ModelSchema schema;
    schema.name = hf.model_type.empty() ? hf.arch_name : hf.model_type;
    schema.pack_dense_qkv_and_gate_up =
        supports_dense_llama_packed_load(hf, boot_cfg);
    schema.unfuse_phi3_for_tp = (hf.model_type == "phi3");
    schema.shard_fused_moe_experts_for_tp =
        (tp_size > 1) && is_qwen_moe_model_type(hf.model_type);
    schema.fuse_per_expert_moe_after_load =
        is_qwen_moe_model_type(hf.model_type);
    schema.family = model_schema_family_for_type(hf.model_type);

    return schema;
}

LayoutPlan build_model_layout_plan(
    const HfConfig& hf,
    const Config& boot_cfg,
    const CheckpointSource& loader,
    int tp_size,
    const BackendTarget& target)
{
    LayoutPlan plan;
    std::unordered_set<std::string> consumed_raw;

    const ModelSchema schema = resolve_model_schema(hf, boot_cfg, tp_size);
    const SemanticGraph semantic_graph =
        build_model_semantic_graph(hf, boot_cfg, loader);
    if (can_use_native_dense_algebra_plan(
            hf, boot_cfg, loader, semantic_graph, schema)) {
        return build_native_dense_algebra_plan(
            semantic_graph, loader, tp_size);
    }
    std::unordered_map<std::string, const SemanticGroup*> packed_group_by_raw;
    packed_group_by_raw.reserve(semantic_graph.groups.size() * 3);
    std::unordered_map<std::string, const SemanticGroup*> row_split_group_by_raw;
    row_split_group_by_raw.reserve(semantic_graph.groups.size());
    std::unordered_map<std::string, const SemanticGroup*> per_expert_group_by_raw;
    per_expert_group_by_raw.reserve(semantic_graph.groups.size() * 3);
    std::unordered_map<std::string, const SemanticGroup*> mxfp4_group_by_raw;
    mxfp4_group_by_raw.reserve(semantic_graph.groups.size() * 3);
    for (const auto& group : semantic_graph.groups) {
        if (group.kind == SemanticGroupKind::PackedQkv ||
            group.kind == SemanticGroupKind::PackedGateUp) {
            for (const auto& raw : group.raw_names) {
                packed_group_by_raw.emplace(raw, &group);
            }
        } else if (group.kind == SemanticGroupKind::RowRangeSplit) {
            for (const auto& raw : group.raw_names) {
                row_split_group_by_raw.emplace(raw, &group);
            }
        } else if (group.kind == SemanticGroupKind::PerExpertMoe) {
            for (const auto& raw : group.raw_names) {
                per_expert_group_by_raw.emplace(raw, &group);
            }
        } else if (group.kind == SemanticGroupKind::GptOssMxfp4) {
            for (const auto& raw : group.raw_names) {
                mxfp4_group_by_raw.emplace(raw, &group);
            }
        }
    }
    const bool has_per_expert_moe_sources =
        schema.fuse_per_expert_moe_after_load &&
        semantic_graph_has_group(
            semantic_graph, SemanticGroupKind::PerExpertMoe);
    const bool lower_runtime_quant =
        runtime_quant_enabled_for_plan(hf, boot_cfg, target.fp8_native);
    const bool can_repack_gptq_marlin =
        can_lower_gptq_marlin_repack_to_plan(hf, tp_size);
    const bool lower_gptq_marlin_repack =
        can_repack_gptq_marlin && target.gptq_marlin_int4;
    const bool can_repack_awq_marlin =
        can_lower_awq_marlin_repack_to_plan(hf);
    const bool lower_awq_marlin_repack =
        can_repack_awq_marlin && target.gptq_marlin_int4;
    const bool lower_awq_dequant =
        can_lower_awq_dequant_to_plan(hf) && !lower_awq_marlin_repack;
    const bool lower_gptq_dequant =
        can_lower_gptq_dequant_to_plan(hf, tp_size) ||
        (can_repack_gptq_marlin && !target.gptq_marlin_int4);
    const bool lower_offline_int4_dequant =
        lower_awq_dequant || lower_gptq_dequant;
    bool lowered_compressed_quant = false;
    bool lowered_gptq_marlin_repack = false;
    bool lowered_offline_int4_dequant = false;
    bool lowered_per_expert_moe_fusion = false;

    for (const auto& semantic : semantic_graph.tensors) {
        const std::string& raw_name = semantic.raw_name;
        const std::string& name = semantic.runtime_name;
        if (consumed_raw.contains(raw_name)) continue;
        if (is_compressed_quant_companion(hf, loader, semantic)) {
            continue;
        }
        if (is_fp8_scale_inv_companion(loader, semantic)) {
            continue;
        }
        if (is_gptq_repack_companion(
                hf, loader, semantic, lower_gptq_marlin_repack)) {
            continue;
        }
        if (is_offline_int4_dequant_companion(
                hf, loader, semantic, lower_offline_int4_dequant)) {
            continue;
        }
        if (is_offline_int4_dequant_companion(
                hf, loader, semantic, lower_awq_marlin_repack)) {
            continue;
        }
        if (mxfp4_group_by_raw.contains(raw_name) &&
            raw_name != mxfp4_group_by_raw.at(raw_name)->raw_names.front()) {
            continue;
        }

        if (schema.shard_fused_moe_experts_for_tp &&
            semantic.role == SemanticRole::MoeExpertsGateUp) {
            const auto& info = loader.info(raw_name);
            auto shape = info.shape;
            if (shape.size() != 3 || shape[1] % (2 * tp_size) != 0) {
                throw std::runtime_error(
                    "load schema: MoE gate/up tensor has unsupported shape: " +
                    name);
            }
            const std::int64_t full_i = shape[1] / 2;
            shape[1] /= tp_size;
            register_tensor_spec(
                plan, name, info.dtype, shape,
                TensorLayoutKind::Grouped,
                TensorOwnershipKind::Owned,
                TensorParallelKind::Expert);
            const LayoutExprId source =
                source_expr_from_info(plan, raw_name, info);
            TensorDecl half_decl;
            half_decl.name = name + ".__half";
            half_decl.dtype = info.dtype;
            half_decl.shape = {shape[0], full_i, shape[2]};
            half_decl.layout = TensorLayoutKind::Grouped;
            half_decl.ownership = TensorOwnershipKind::Temporary;
            half_decl.parallel = TensorParallelKind::Expert;
            TensorDecl local_half_decl = half_decl;
            local_half_decl.shape[1] /= tp_size;
            LayoutExprId gate = select_expr(
                plan, source, half_decl, 1, 0, full_i);
            gate = partition_expr(plan, gate, local_half_decl, 1, tp_size);
            LayoutExprId up = select_expr(
                plan, source, half_decl, 1, full_i, full_i);
            up = partition_expr(plan, up, local_half_decl, 1, tp_size);
            TensorDecl out_decl = tensor_decl_for(plan, name);
            LayoutExpr join = make_expr(LayoutExprKind::Join, out_decl);
            join.inputs = {gate, up};
            join.axis = 1;
            const LayoutExprId joined = add_expr(plan, std::move(join));
            (void)realize_expr(plan, name, joined, std::move(out_decl));
            consumed_raw.insert(raw_name);
            continue;
        }
        if (schema.shard_fused_moe_experts_for_tp &&
            semantic.role == SemanticRole::MoeExpertsDown) {
            const auto& info = loader.info(raw_name);
            auto shape = info.shape;
            if (shape.size() != 3 || shape[2] % tp_size != 0) {
                throw std::runtime_error(
                    "load schema: MoE down tensor has unsupported shape: " +
                    name);
            }
            shape[2] /= tp_size;
            register_tensor_spec(
                plan, name, info.dtype, shape,
                TensorLayoutKind::Grouped,
                TensorOwnershipKind::Owned,
                TensorParallelKind::Expert);
            TensorDecl out_decl = tensor_decl_for(plan, name);
            LayoutExprId source =
                source_expr_from_info(plan, raw_name, info);
            source = partition_expr(plan, source, out_decl, 2, tp_size);
            (void)realize_expr(plan, name, source, std::move(out_decl));
            consumed_raw.insert(raw_name);
            continue;
        }

        if (schema.unfuse_phi3_for_tp) {
            const auto split_it = row_split_group_by_raw.find(raw_name);
            if (split_it != row_split_group_by_raw.end() &&
                try_add_row_range_split_group(
                    plan, hf, loader, *split_it->second,
                    consumed_raw, tp_size)) {
                continue;
            }
        }

        if (const auto mxfp4_it = mxfp4_group_by_raw.find(raw_name);
            mxfp4_it != mxfp4_group_by_raw.end()) {
            if (try_add_gpt_oss_mxfp4_expert(
                    plan, hf, loader, *mxfp4_it->second, consumed_raw, tp_size,
                    target.mxfp4_moe)) {
                continue;
            }
        }

        if (schema.pack_dense_qkv_and_gate_up) {
            const auto group_it = packed_group_by_raw.find(raw_name);
            if (group_it != packed_group_by_raw.end() &&
                try_add_packed_axis_group(
                    plan, loader, *group_it->second, consumed_raw, tp_size)) {
                continue;
            }
        }

        if (schema.fuse_per_expert_moe_after_load &&
            per_expert_group_by_raw.contains(raw_name) &&
            try_add_per_expert_moe_fusion(
                plan, hf, loader, *per_expert_group_by_raw.at(raw_name),
                consumed_raw, tp_size)) {
            lowered_per_expert_moe_fusion = true;
            continue;
        }

        const int axis = (tp_size > 1) ? semantic.shard_axis : -1;
        if (try_add_compressed_fp8_weight(
                plan, hf, loader, semantic, loader.info(raw_name),
                axis, tp_size, target.fp8_native)) {
            consumed_raw.insert(raw_name);
            consumed_raw.insert(raw_name + "_scale");
            lowered_compressed_quant = true;
            continue;
        }
        if (try_add_compressed_int8_weight(
                plan, hf, loader, semantic, loader.info(raw_name),
                axis, tp_size)) {
            consumed_raw.insert(raw_name);
            consumed_raw.insert(raw_name + "_scale");
            lowered_compressed_quant = true;
            continue;
        }
        if (try_add_fp8_scale_inv_weight(
                plan, loader, semantic, loader.info(raw_name),
                axis, tp_size)) {
            consumed_raw.insert(raw_name);
            consumed_raw.insert(raw_name + "_scale_inv");
            continue;
        }
        if (lower_awq_dequant &&
            try_add_awq_dequant_weight(
                plan, hf, loader, semantic, loader.info(raw_name),
                tp_size, consumed_raw)) {
            lowered_offline_int4_dequant = true;
            continue;
        }
        if (lower_awq_marlin_repack &&
            try_add_awq_marlin_repack_weight(
                plan, hf, loader, semantic, loader.info(raw_name),
                tp_size, consumed_raw)) {
            lowered_gptq_marlin_repack = true;
            continue;
        }
        if (lower_gptq_marlin_repack &&
            try_add_gptq_marlin_repack_weight(
                plan, hf, loader, semantic, loader.info(raw_name),
                tp_size, consumed_raw)) {
            lowered_gptq_marlin_repack = true;
            continue;
        }
        if (lower_gptq_dequant &&
            try_add_gptq_dequant_weight(
                plan, hf, loader, semantic, loader.info(raw_name),
                tp_size, consumed_raw)) {
            lowered_offline_int4_dequant = true;
            continue;
        }
        if (lower_runtime_quant && runtime_quantizable_role(semantic.role)) {
            add_runtime_quantized_copy(
                plan, semantic, loader.info(raw_name), boot_cfg,
                axis, tp_size);
            consumed_raw.insert(raw_name);
            continue;
        }
        add_copy(plan, raw_name, name, loader.info(raw_name), axis, tp_size);
        consumed_raw.insert(raw_name);
    }
    if (hf.quant_method == "compressed-tensors" && !lowered_compressed_quant) {
        throw std::runtime_error(
            "load schema: compressed-tensors checkpoint did not match any "
            "scheduled FP8/INT8 quantized weight layout");
    }
    if ((hf.quant_method == "awq" || hf.quant_method == "gptq") &&
        !lowered_gptq_marlin_repack && !lowered_offline_int4_dequant) {
        throw std::runtime_error(
            "load schema: " + hf.quant_method +
            " checkpoint did not match any scheduled INT4 dequant/repack "
            "layout");
    }
    if (schema.fuse_per_expert_moe_after_load &&
        has_per_expert_moe_sources && !lowered_per_expert_moe_fusion) {
        throw std::runtime_error(
            "load schema: per-expert MoE checkpoint layout requires "
            "StackGroups IR lowering, but no complete expert group was "
            "scheduled");
    }
    finalize_memory_plan(plan);
    return plan;
}

}  // namespace pie_cuda_driver

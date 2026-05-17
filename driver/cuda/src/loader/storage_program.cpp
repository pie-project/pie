#include "loader/storage_program.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <pie_driver_common/shard_plan.hpp>

#include "tensor.hpp"

namespace pie_cuda_driver {

namespace {

std::uint64_t tensor_nbytes(DType dtype, const std::vector<std::int64_t>& shape) {
    std::uint64_t n = 1;
    for (const auto dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("storage program: negative tensor dimension");
        }
        n *= static_cast<std::uint64_t>(dim);
    }
    return n * static_cast<std::uint64_t>(dtype_bytes(dtype));
}

std::uint64_t tensor_nbytes(const TensorDecl& spec) {
    return tensor_nbytes(spec.dtype, spec.shape);
}

std::uint64_t source_linear_element(
    const TensorInfo& info,
    const std::vector<std::int64_t>& index)
{
    if (info.shape.empty()) return 0;
    std::uint64_t linear = 0;
    std::uint64_t stride = 1;
    for (std::size_t rev = 0; rev < info.shape.size(); ++rev) {
        const std::size_t axis = info.shape.size() - 1 - rev;
        linear += static_cast<std::uint64_t>(index[axis]) * stride;
        stride *= static_cast<std::uint64_t>(info.shape[axis]);
    }
    return linear;
}

void source_bounds_for_slices(
    const TensorInfo& info,
    const std::vector<TensorSlice>& slices,
    std::uint64_t& source_delta_bytes,
    std::uint64_t& source_span_bytes)
{
    const std::uint64_t elem = dtype_bytes(info.dtype);
    if (info.shape.empty() || slices.empty()) {
        source_delta_bytes = 0;
        source_span_bytes = info.nbytes;
        return;
    }
    std::vector<std::int64_t> first(info.shape.size(), 0);
    std::vector<std::int64_t> last = info.shape;
    for (auto& dim : last) {
        dim -= 1;
    }
    for (const auto& slice : slices) {
        if (slice.axis < 0 ||
            slice.axis >= static_cast<int>(info.shape.size()) ||
            slice.length <= 0) {
            throw std::runtime_error(
                "storage program: invalid source slice bounds");
        }
        const auto axis = static_cast<std::size_t>(slice.axis);
        first[axis] = slice.start;
        last[axis] = slice.start + slice.length - 1;
    }
    const std::uint64_t first_linear = source_linear_element(info, first);
    const std::uint64_t last_linear = source_linear_element(info, last);
    source_delta_bytes = first_linear * elem;
    source_span_bytes = (last_linear - first_linear + 1) * elem;
}

const TensorDecl& tensor_spec_for(
    const LayoutPlan& plan,
    const std::string& name)
{
    const auto it = plan.tensors.find(name);
    if (it == plan.tensors.end()) {
        throw std::runtime_error(
            "storage program: missing TensorDecl for '" + name + "'");
    }
    return it->second;
}

const TensorDecl* find_tensor_spec(
    const LayoutPlan& plan,
    const std::string& name)
{
    const auto it = plan.tensors.find(name);
    return it == plan.tensors.end() ? nullptr : &it->second;
}

struct AlgebraBindingRef {
    std::size_t binding_index = kInvalidStorageId;
    LayoutExprId root = kInvalidStorageId;
    const LayoutExpr* root_expr = nullptr;
};

AlgebraBindingRef realize_binding_for_output(
    const LayoutPlan& plan,
    const std::string& output_name)
{
    for (std::size_t i = plan.algebra.bindings.size(); i > 0; --i) {
        const std::size_t binding_index = i - 1;
        const auto& binding = plan.algebra.bindings[binding_index];
        if (binding.runtime_name != output_name ||
            binding.root >= plan.algebra.exprs.size()) {
            continue;
        }
        const LayoutExpr& root = plan.algebra.exprs[binding.root];
        if (root.kind == LayoutExprKind::Realize ||
            root.kind == LayoutExprKind::View) {
            return AlgebraBindingRef{
                .binding_index = binding_index,
                .root = binding.root,
                .root_expr = &root,
            };
        }
    }
    return {};
}

void attach_algebra_binding(
    std::vector<ExtentWrite>& writes,
    const AlgebraBindingRef& binding)
{
    if (binding.root_expr == nullptr) return;
    for (auto& write : writes) {
        write.expr_id = binding.root;
        write.binding_index = binding.binding_index;
    }
}

bool is_raw_copy_kind(LayoutOpKind kind) noexcept {
    return kind == LayoutOpKind::Read ||
           kind == LayoutOpKind::Copy ||
           kind == LayoutOpKind::Shard ||
           kind == LayoutOpKind::RowRangeShard;
}

bool op_allocates_tensor(LayoutOpKind kind) noexcept {
    return kind != LayoutOpKind::Drop &&
           kind != LayoutOpKind::View &&
           kind != LayoutOpKind::Alias &&
           kind != LayoutOpKind::AttachMetadata;
}

bool is_raw_copy_cast_drop_sequence(
    const LayoutPlan& plan,
    std::size_t producer_index)
{
    if (producer_index + 2 >= plan.ops.size()) return false;
    const LayoutOp& producer = plan.ops[producer_index];
    const LayoutOp& cast = plan.ops[producer_index + 1];
    const LayoutOp& drop = plan.ops[producer_index + 2];
    if (!is_raw_copy_kind(producer.kind) ||
        cast.kind != LayoutOpKind::Cast ||
        drop.kind != LayoutOpKind::Drop) {
        return false;
    }
    if (layout_op_inputs(cast).size() != 1 ||
        layout_op_inputs(cast)[0] != layout_op_output(producer)) {
        return false;
    }
    return std::find(
        layout_op_inputs(drop).begin(),
        layout_op_inputs(drop).end(),
        layout_op_output(producer)) != layout_op_inputs(drop).end();
}

std::uint64_t leading_tile_scratch_bytes(
    const TensorDecl& spec,
    std::uint64_t tile_bytes)
{
    if (spec.shape.empty()) return dtype_bytes(spec.dtype);
    std::vector<std::int64_t> inner_shape(
        spec.shape.begin() + 1, spec.shape.end());
    std::uint64_t row_bytes = tensor_nbytes(spec.dtype, inner_shape);
    if (spec.shape.size() == 1) {
        row_bytes = static_cast<std::uint64_t>(dtype_bytes(spec.dtype));
    }
    const std::int64_t rows_per_tile = std::max<std::int64_t>(
        1,
        static_cast<std::int64_t>(
            tile_bytes / std::max<std::uint64_t>(row_bytes, 1)));
    const std::int64_t tile_rows = std::min(spec.shape[0], rows_per_tile);
    return static_cast<std::uint64_t>(tile_rows) * row_bytes;
}

std::uint64_t transform_scratch_bytes_for_op(
    const LayoutPlan& plan,
    std::size_t op_index,
    std::uint64_t tile_bytes)
{
    const LayoutOp& op = plan.ops[op_index];
    if (op.kind == LayoutOpKind::Cast) {
        if (op_index > 0 &&
            is_raw_copy_cast_drop_sequence(plan, op_index - 1)) {
            return leading_tile_scratch_bytes(
                tensor_spec_for(plan, layout_op_output(plan.ops[op_index - 1])),
                tile_bytes);
        }
        return 0;
    }
    if (op.kind != LayoutOpKind::Dequantize || layout_op_inputs(op).size() < 2) {
        return 0;
    }

    const TensorDecl* source = find_tensor_spec(plan, layout_op_inputs(op)[0]);
    const TensorDecl* scale = find_tensor_spec(plan, layout_op_inputs(op)[1]);
    if (source == nullptr || scale == nullptr) return 0;

    if ((source->quant.format == QuantFormat::AwqInt4 ||
         source->quant.format == QuantFormat::GptqInt4) &&
        scale->dtype != DType::BF16) {
        return tensor_nbytes(DType::BF16, scale->shape);
    }
    if (source->dtype == DType::FP8_E4M3 && scale->dtype == DType::BF16) {
        return tensor_nbytes(DType::FP32, scale->shape);
    }
    return 0;
}

struct StorageTempUsage {
    std::uint64_t max_resident_temp_bytes = 0;
    std::uint64_t max_transform_scratch_bytes = 0;
    std::uint64_t max_total_temporary_bytes = 0;
};

StorageTempUsage compute_storage_temp_usage(
    const LayoutPlan& plan,
    std::uint64_t tile_bytes)
{
    std::unordered_map<std::string, std::uint64_t> temp_bytes;
    temp_bytes.reserve(plan.tensors.size());
    for (const auto& [name, spec] : plan.tensors) {
        if (spec.ownership == TensorOwnershipKind::Temporary) {
            temp_bytes.emplace(name, tensor_nbytes(spec));
        }
    }

    StorageTempUsage usage;
    std::uint64_t live_temp_bytes = 0;
    auto add_temp = [&](const std::string& name) {
        const auto it = temp_bytes.find(name);
        if (it == temp_bytes.end()) return;
        live_temp_bytes += it->second;
        usage.max_resident_temp_bytes =
            std::max(usage.max_resident_temp_bytes, live_temp_bytes);
        usage.max_total_temporary_bytes =
            std::max(usage.max_total_temporary_bytes, live_temp_bytes);
    };
    auto release_temp = [&](const std::string& name) {
        const auto it = temp_bytes.find(name);
        if (it == temp_bytes.end()) return;
        live_temp_bytes -= std::min(live_temp_bytes, it->second);
    };
    auto account_scratch = [&](std::uint64_t scratch) {
        usage.max_transform_scratch_bytes =
            std::max(usage.max_transform_scratch_bytes, scratch);
        usage.max_total_temporary_bytes =
            std::max(usage.max_total_temporary_bytes,
                     live_temp_bytes + scratch);
    };

    for (std::size_t i = 0; i < plan.ops.size(); ++i) {
        if (is_raw_copy_cast_drop_sequence(plan, i)) {
            const LayoutOp& cast = plan.ops[i + 1];
            add_temp(layout_op_output(cast));
            add_temp(layout_op_secondary_output(cast));
            account_scratch(transform_scratch_bytes_for_op(
                plan, i + 1, tile_bytes));
            i += 2;
            continue;
        }

        const LayoutOp& op = plan.ops[i];
        if (op_allocates_tensor(op.kind)) {
            add_temp(layout_op_output(op));
            add_temp(layout_op_secondary_output(op));
        }
        account_scratch(transform_scratch_bytes_for_op(plan, i, tile_bytes));

        if (op.kind == LayoutOpKind::Drop) {
            if (layout_op_inputs(op).empty()) {
                release_temp(layout_op_output(op));
            } else {
                for (const auto& input : layout_op_inputs(op)) {
                    release_temp(input);
                }
            }
        }
    }
    return usage;
}

std::uint64_t range_count_for(
    const TensorInfo& info,
    const std::vector<TensorSlice>& slices,
    const std::vector<std::int64_t>& dst_shape)
{
    if (info.shape.empty() || slices.empty()) return 1;
    bool leading_only = true;
    for (const auto& slice : slices) {
        if (slice.axis != 0) {
            leading_only = false;
            break;
        }
    }
    if (leading_only) return 1;
    std::uint64_t ranges = 1;
    for (std::size_t i = 0; i + 1 < dst_shape.size(); ++i) {
        ranges *= static_cast<std::uint64_t>(dst_shape[i]);
    }
    return std::max<std::uint64_t>(ranges, 1);
}

ExtentWrite make_write_record(
    std::size_t op_index,
    LayoutExprId expr_id,
    std::size_t binding_index,
    std::string op_kind,
    const CheckpointSource& metadata,
    std::string output_name,
    std::string raw_name,
    std::vector<TensorSlice> slices,
    std::vector<std::int64_t> dst_shape,
    std::uint64_t dst_offset_bytes)
{
    const auto& info = metadata.info(raw_name);
    const std::uint64_t bytes = tensor_nbytes(info.dtype, dst_shape);
    const std::uint64_t ranges = range_count_for(info, slices, dst_shape);
    const TensorStorageInfo storage = metadata.storage_info(raw_name);
    std::uint64_t source_delta = 0;
    std::uint64_t source_span = 0;
    source_bounds_for_slices(info, slices, source_delta, source_span);
    return ExtentWrite{
        .op_index = op_index,
        .expr_id = expr_id,
        .binding_index = binding_index,
        .op_kind = std::move(op_kind),
        .raw_name = std::move(raw_name),
        .output_name = std::move(output_name),
        .slices = std::move(slices),
        .dst_shape = std::move(dst_shape),
        .dst_offset_bytes = dst_offset_bytes,
        .source_path = storage.path.string(),
        .source_shard_id = storage.shard_id,
        .source_offset_bytes = storage.file_offset + source_delta,
        .source_span_bytes = source_span,
        .bytes = bytes,
        .range_count = ranges,
        .contiguous = ranges == 1,
    };
}

ExtentWrite make_write(
    const LayoutOp& op,
    std::size_t op_index,
    const CheckpointSource& metadata,
    std::string output_name,
    std::string raw_name,
    std::vector<TensorSlice> slices,
    std::vector<std::int64_t> dst_shape,
    std::uint64_t dst_offset_bytes)
{
    return make_write_record(
        op_index, kInvalidStorageId, kInvalidStorageId,
        layout_op_kind_name(op.kind), metadata, std::move(output_name),
        std::move(raw_name), std::move(slices), std::move(dst_shape),
        dst_offset_bytes);
}

std::vector<TensorSlice> shard_slices_for(
    const TensorInfo& info,
    int shard_axis,
    int tp_rank,
    int tp_size,
    std::vector<std::int64_t>& shape,
    const std::string& raw_name)
{
    shape = info.shape;
    if (tp_size <= 1 || shard_axis < 0) return {};
    const auto shard = pie_driver_common::plan_axis_shard(
        info.shape, shard_axis, tp_rank, tp_size,
        "storage program shard: " + raw_name);
    shape = shard.output_shape;
    return {TensorSlice{shard_axis, shard.offset, shard.shard_dim}};
}

std::vector<ExtentWrite> lower_raw_like_op(
    const LayoutOp& op,
    std::size_t op_index,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorDecl& out = tensor_spec_for(plan, layout_op_output(op));
    const auto& info = metadata.info(layout_op_raw_name(op));
    if (info.dtype != out.dtype) {
        throw std::runtime_error(
            "storage program: raw extent write dtype mismatch for '" +
            out.name + "'");
    }
    std::vector<std::int64_t> shape;
    auto slices = shard_slices_for(
        info, layout_op_shard_axis(op), tp_rank, tp_size, shape,
        layout_op_raw_name(op));
    if (shape != out.shape) {
        throw std::runtime_error(
            "storage program: raw extent write shape mismatch for '" +
            out.name + "'");
    }
    return {make_write(
        op, op_index, metadata, out.name, layout_op_raw_name(op),
        std::move(slices), out.shape, 0)};
}

std::vector<ExtentWrite> lower_row_range_shard_op(
    const LayoutOp& op,
    std::size_t op_index,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorDecl& out = tensor_spec_for(plan, layout_op_output(op));
    const auto& info = metadata.info(layout_op_raw_name(op));
    std::int64_t rows = layout_op_rows(op);
    std::int64_t offset = layout_op_row_offset(op);
    if (rows <= 0) {
        throw std::runtime_error(
            "storage program: row-range shard has non-positive rows for '" +
            layout_op_raw_name(op) + "'");
    }
    if (tp_size > 1) {
        if (rows % tp_size != 0) {
            throw std::runtime_error(
                "storage program: row range is not divisible by tp_size for '" +
                layout_op_raw_name(op) + "'");
        }
        rows /= tp_size;
        offset += static_cast<std::int64_t>(tp_rank) * rows;
    }
    std::vector<std::int64_t> shape = info.shape;
    shape[0] = rows;
    if (info.dtype != out.dtype || shape != out.shape) {
        throw std::runtime_error(
            "storage program: row-range output mismatch for '" +
            out.name + "'");
    }
    return {make_write(
        op, op_index, metadata, out.name, layout_op_raw_name(op),
        {TensorSlice{0, offset, rows}}, out.shape, 0)};
}

std::vector<ExtentWrite> lower_axis_concat_op(
    const LayoutOp& op,
    std::size_t op_index,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorDecl& out = tensor_spec_for(plan, layout_op_output(op));
    if (out.shape.size() != 2) {
        throw std::runtime_error(
            "storage program: AxisConcat output must be 2-D for '" +
            out.name + "'");
    }

    std::vector<ExtentWrite> writes;
    std::uint64_t dst_offset = 0;
    std::int64_t rows = 0;
    for (const auto& src : layout_op_sources(op)) {
        const auto& info = metadata.info(src.raw_name);
        if (info.dtype != out.dtype || info.shape.size() != 2) {
            throw std::runtime_error(
                "storage program: AxisConcat source mismatch for '" +
                out.name + "'");
        }
        std::vector<std::int64_t> shape;
        auto slices = shard_slices_for(
            info, layout_op_shard_axis(op), tp_rank, tp_size, shape,
            src.raw_name);
        if (shape.size() != 2 || shape[1] != out.shape[1]) {
            throw std::runtime_error(
                "storage program: AxisConcat source shape mismatch for '" +
                src.raw_name + "'");
        }
        writes.push_back(make_write(
            op, op_index, metadata, out.name, src.raw_name,
            std::move(slices), shape, dst_offset));
        rows += shape[0];
        dst_offset += tensor_nbytes(out.dtype, shape);
    }
    if (rows != out.shape[0]) {
        throw std::runtime_error(
            "storage program: AxisConcat row count mismatch for '" +
            out.name + "'");
    }
    return writes;
}

std::vector<ExtentWrite> lower_grouped_slice_concat_op(
    const LayoutOp& op,
    std::size_t op_index,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorDecl& out = tensor_spec_for(plan, layout_op_output(op));
    const auto& info = metadata.info(layout_op_raw_name(op));
    if (info.dtype != out.dtype || info.shape.size() != 3 ||
        out.shape.size() != 3) {
        throw std::runtime_error(
            "storage program: GroupedSliceConcat expects rank-3 tensors for '" +
            out.name + "'");
    }
    const std::int64_t E = info.shape[0];
    const std::int64_t two_I = info.shape[1];
    const std::int64_t H = info.shape[2];
    if (two_I % 2 != 0 || (two_I / 2) % tp_size != 0) {
        throw std::runtime_error(
            "storage program: GroupedSliceConcat intermediate axis mismatch for '" +
            out.name + "'");
    }
    const std::int64_t I = two_I / 2;
    const std::int64_t I_local = I / tp_size;
    if (out.shape != std::vector<std::int64_t>{E, 2 * I_local, H}) {
        throw std::runtime_error(
            "storage program: GroupedSliceConcat output shape mismatch for '" +
            out.name + "'");
    }

    std::vector<ExtentWrite> writes;
    const std::uint64_t half_bytes =
        tensor_nbytes(out.dtype, {1, I_local, H});
    const std::uint64_t expert_bytes = 2 * half_bytes;
    const std::int64_t gate_start =
        static_cast<std::int64_t>(tp_rank) * I_local;
    const std::int64_t up_start = I + gate_start;
    for (std::int64_t e = 0; e < E; ++e) {
        const std::uint64_t expert_offset =
            static_cast<std::uint64_t>(e) * expert_bytes;
        writes.push_back(make_write(
            op, op_index, metadata, out.name, layout_op_raw_name(op),
            {TensorSlice{0, e, 1}, TensorSlice{1, gate_start, I_local}},
            {1, I_local, H}, expert_offset));
        writes.push_back(make_write(
            op, op_index, metadata, out.name, layout_op_raw_name(op),
            {TensorSlice{0, e, 1}, TensorSlice{1, up_start, I_local}},
            {1, I_local, H}, expert_offset + half_bytes));
    }
    return writes;
}

std::vector<ExtentWrite> lower_grouped_slice_op(
    const LayoutOp& op,
    std::size_t op_index,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorDecl& out = tensor_spec_for(plan, layout_op_output(op));
    const auto& info = metadata.info(layout_op_raw_name(op));
    if (info.dtype != out.dtype || info.shape.size() != 3 ||
        out.shape.size() != 3) {
        throw std::runtime_error(
            "storage program: GroupedSlice expects rank-3 tensors for '" +
            out.name + "'");
    }
    const std::int64_t I = info.shape[2];
    if (I % tp_size != 0) {
        throw std::runtime_error(
            "storage program: GroupedSlice axis mismatch for '" +
            out.name + "'");
    }
    const std::int64_t I_local = I / tp_size;
    if (out.shape != std::vector<std::int64_t>{info.shape[0], info.shape[1], I_local}) {
        throw std::runtime_error(
            "storage program: GroupedSlice output shape mismatch for '" +
            out.name + "'");
    }
    return {make_write(
        op, op_index, metadata, out.name, layout_op_raw_name(op),
        {TensorSlice{2, static_cast<std::int64_t>(tp_rank) * I_local, I_local}},
        out.shape, 0)};
}

std::vector<ExtentWrite> lower_stack_groups_op(
    const LayoutOp& op,
    std::size_t op_index,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size)
{
    if (layout_op_sources(op).empty()) return {};
    const TensorDecl& gate_up = tensor_spec_for(plan, layout_op_output(op));
    const TensorDecl& down = tensor_spec_for(plan, layout_op_secondary_output(op));
    if (layout_op_sources(op).size() % 3 != 0 ||
        gate_up.shape.size() != 3 || down.shape.size() != 3) {
        throw std::runtime_error(
            "storage program: StackGroups expects expert triples for '" +
            gate_up.name + "'");
    }
    const std::int64_t E =
        static_cast<std::int64_t>(layout_op_sources(op).size() / 3);
    const std::int64_t I = gate_up.shape[1] / 2;
    const std::int64_t H = gate_up.shape[2];
    const std::int64_t I_down = down.shape[2];
    if (gate_up.shape != std::vector<std::int64_t>{E, 2 * I, H} ||
        down.shape != std::vector<std::int64_t>{E, H, I_down}) {
        throw std::runtime_error(
            "storage program: StackGroups output shape mismatch for '" +
            gate_up.name + "'");
    }

    std::vector<ExtentWrite> writes;
    const std::uint64_t proj_bytes = tensor_nbytes(gate_up.dtype, {I, H});
    const std::uint64_t gate_up_expert_bytes = 2 * proj_bytes;
    const std::uint64_t down_expert_bytes = tensor_nbytes(down.dtype, {H, I_down});
    for (std::int64_t e = 0; e < E; ++e) {
        const auto& gate_src = layout_op_sources(op)[static_cast<std::size_t>(e) * 3];
        const auto& up_src = layout_op_sources(op)[static_cast<std::size_t>(e) * 3 + 1];
        const auto& down_src = layout_op_sources(op)[static_cast<std::size_t>(e) * 3 + 2];
        const std::vector<std::int64_t> full_gate_shape{
            I * static_cast<std::int64_t>(tp_size), H};
        const std::vector<std::int64_t> full_down_shape{
            H, I_down * static_cast<std::int64_t>(tp_size)};
        const auto& gate_info = metadata.info(gate_src.raw_name);
        const auto& up_info = metadata.info(up_src.raw_name);
        const auto& down_info = metadata.info(down_src.raw_name);
        if (gate_info.dtype != gate_up.dtype ||
            up_info.dtype != gate_up.dtype ||
            down_info.dtype != down.dtype ||
            gate_info.shape != full_gate_shape ||
            up_info.shape != full_gate_shape ||
            down_info.shape != full_down_shape) {
            throw std::runtime_error(
                "storage program: StackGroups source metadata mismatch for '" +
                gate_up.name + "'");
        }
        const std::uint64_t gate_up_offset =
            static_cast<std::uint64_t>(e) * gate_up_expert_bytes;
        const std::uint64_t down_offset =
            static_cast<std::uint64_t>(e) * down_expert_bytes;
        const std::vector<TensorSlice> gate_slices =
            tp_size > 1
                ? std::vector<TensorSlice>{
                      TensorSlice{0, static_cast<std::int64_t>(tp_rank) * I, I}}
                : std::vector<TensorSlice>{};
        const std::vector<TensorSlice> down_slices =
            tp_size > 1
                ? std::vector<TensorSlice>{
                      TensorSlice{1, static_cast<std::int64_t>(tp_rank) * I_down, I_down}}
                : std::vector<TensorSlice>{};
        writes.push_back(make_write(
            op, op_index, metadata, gate_up.name, gate_src.raw_name,
            gate_slices, {I, H}, gate_up_offset));
        writes.push_back(make_write(
            op, op_index, metadata, gate_up.name, up_src.raw_name,
            gate_slices, {I, H}, gate_up_offset + proj_bytes));
        writes.push_back(make_write(
            op, op_index, metadata, down.name, down_src.raw_name,
            down_slices, {H, I_down}, down_offset));
    }
    return writes;
}

bool lower_source_expr_to_write(
    const LayoutExpr& expr,
    std::size_t op_index,
    LayoutExprId expr_id,
    std::size_t binding_index,
    const std::string& op_kind,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    const std::string& output_name,
    const std::vector<std::int64_t>& output_shape,
    std::uint64_t dst_offset_bytes,
    int tp_rank,
    int tp_size,
    std::vector<ExtentWrite>& writes)
{
    if (expr.kind == LayoutExprKind::Source) {
        const auto& info = metadata.info(expr.raw_name);
        if (info.dtype != tensor_spec_for(plan, output_name).dtype ||
            info.shape != output_shape) {
            return false;
        }
        writes.push_back(make_write_record(
            op_index, expr_id, binding_index, op_kind, metadata,
            output_name, expr.raw_name, {}, output_shape, dst_offset_bytes));
        return true;
    }

    if ((expr.kind == LayoutExprKind::Partition ||
         expr.kind == LayoutExprKind::Select) &&
        expr.inputs.size() == 1 &&
        expr.inputs.front() < plan.algebra.exprs.size() &&
        plan.algebra.exprs[expr.inputs.front()].kind == LayoutExprKind::Source) {
        const LayoutExpr& source = plan.algebra.exprs[expr.inputs.front()];
        const auto& info = metadata.info(source.raw_name);
        std::vector<TensorSlice> slices;
        std::vector<std::int64_t> shape = info.shape;
        if (expr.kind == LayoutExprKind::Partition) {
            slices = shard_slices_for(
                info, expr.axis, tp_rank, tp_size, shape, source.raw_name);
        } else {
            if (expr.axis < 0 ||
                expr.axis >= static_cast<int>(info.shape.size()) ||
                expr.length <= 0) {
                return false;
            }
            std::int64_t start = expr.start;
            std::int64_t length = expr.length;
            if (tp_size > 1 && output_shape.size() == info.shape.size() &&
                expr.axis == 0 && length % tp_size == 0 &&
                output_shape[0] == length / tp_size) {
                length /= tp_size;
                start += static_cast<std::int64_t>(tp_rank) * length;
            }
            shape[static_cast<std::size_t>(expr.axis)] = length;
            slices.push_back(TensorSlice{expr.axis, start, length});
        }
        if (info.dtype != tensor_spec_for(plan, output_name).dtype ||
            shape != output_shape) {
            return false;
        }
        writes.push_back(make_write_record(
            op_index, expr_id, binding_index, op_kind, metadata,
            output_name, source.raw_name, std::move(slices), output_shape,
            dst_offset_bytes));
        return true;
    }
    return false;
}

bool try_lower_join_expr_to_writes(
    const LayoutExpr& join,
    std::size_t op_index,
    LayoutExprId expr_id,
    std::size_t binding_index,
    const std::string& op_kind,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    const std::string& output_name,
    int tp_rank,
    int tp_size,
    std::vector<ExtentWrite>& writes)
{
    if (join.kind != LayoutExprKind::Join) return false;
    const TensorDecl& out = tensor_spec_for(plan, output_name);
    if (join.axis != 0 || out.shape.empty()) return false;

    std::uint64_t dst_offset = 0;
    std::vector<std::int64_t> total_shape = out.shape;
    total_shape[0] = 0;
    for (const auto input_id : join.inputs) {
        if (input_id >= plan.algebra.exprs.size()) return false;
        const LayoutExpr& child = plan.algebra.exprs[input_id];
        const LayoutExpr* source = &child;
        int shard_axis = join.axis;
        if (child.kind == LayoutExprKind::Partition &&
            child.inputs.size() == 1 &&
            child.inputs.front() < plan.algebra.exprs.size()) {
            source = &plan.algebra.exprs[child.inputs.front()];
            shard_axis = child.axis;
        }
        if (source->kind != LayoutExprKind::Source) return false;
        const auto& info = metadata.info(source->raw_name);
        if (info.dtype != out.dtype ||
            info.shape.size() != out.shape.size()) {
            return false;
        }
        std::vector<std::int64_t> shape;
        auto slices = shard_slices_for(
            info, shard_axis, tp_rank, tp_size, shape, source->raw_name);
        if (shape.size() != out.shape.size()) return false;
        for (std::size_t axis = 1; axis < shape.size(); ++axis) {
            if (shape[axis] != out.shape[axis]) return false;
        }
        writes.push_back(make_write_record(
            op_index, expr_id, binding_index, op_kind, metadata,
            output_name, source->raw_name, std::move(slices), shape,
            dst_offset));
        dst_offset += tensor_nbytes(out.dtype, shape);
        total_shape[0] += shape[0];
    }
    return total_shape == out.shape;
}

std::vector<ExtentWrite> lower_extent_writes_for_algebra_root(
    const LayoutOp& op,
    std::size_t op_index,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size)
{
    const std::string& output_name = layout_op_output(op);
    if (output_name.empty()) return {};
    const AlgebraBindingRef binding =
        realize_binding_for_output(plan, output_name);
    const LayoutExpr* root = binding.root_expr;
    if (root == nullptr || root->inputs.size() != 1) return {};
    const LayoutExpr& value = plan.algebra.exprs[root->inputs.front()];
    std::vector<ExtentWrite> writes;
    const std::string op_kind = std::string("Algebra.") +
        layout_expr_kind_name(root->kind);
    if (try_lower_join_expr_to_writes(
            value, op_index, binding.root, binding.binding_index, op_kind,
            plan, metadata, output_name,
            tp_rank, tp_size, writes)) {
        return writes;
    }
    writes.clear();
    if (lower_source_expr_to_write(
            value, op_index, binding.root, binding.binding_index, op_kind,
            plan, metadata, output_name,
            tensor_spec_for(plan, output_name).shape, 0,
            tp_rank, tp_size, writes)) {
        return writes;
    }
    return {};
}

void json_string(std::ostream& out, const std::string& value) {
    out << '"';
    for (const char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    out << '"';
}

void json_shape(std::ostream& out, const std::vector<std::int64_t>& shape) {
    out << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    out << ']';
}

void json_slices(std::ostream& out, const std::vector<TensorSlice>& slices) {
    out << '[';
    for (std::size_t i = 0; i < slices.size(); ++i) {
        if (i) out << ',';
        out << "{\"axis\":" << slices[i].axis
            << ",\"start\":" << slices[i].start
            << ",\"length\":" << slices[i].length << '}';
    }
    out << ']';
}

void json_id_field(
    std::ostream& out,
    const char* name,
    std::size_t value)
{
    out << ",\"" << name << "\":";
    if (value == kInvalidStorageId) {
        out << "null";
    } else {
        out << value;
    }
}

void json_id_value(std::ostream& out, std::size_t value)
{
    if (value == kInvalidStorageId) {
        out << "null";
    } else {
        out << value;
    }
}

void json_payload(std::ostream& out, const StorageInstrPayload& payload)
{
    out << "{\"kind\":";
    if (std::holds_alternative<std::monostate>(payload)) {
        json_string(out, "None");
        out << '}';
        return;
    }
    if (const auto* slice = std::get_if<StorageSlicePayload>(&payload)) {
        json_string(out, "Slice");
        out << ",\"axis\":" << slice->axis
            << ",\"start\":" << slice->start
            << ",\"length\":" << slice->length
            << ",\"shard_axis\":" << slice->shard_axis
            << '}';
        return;
    }
    if (const auto* view = std::get_if<StorageViewPayload>(&payload)) {
        json_string(out, "View");
        out << ",\"axis\":" << view->axis
            << ",\"start\":" << view->start
            << ",\"length\":" << view->length
            << '}';
        return;
    }
    if (const auto* axis = std::get_if<StorageAxisPayload>(&payload)) {
        json_string(out, "Axis");
        out << ",\"shard_axis\":" << axis->shard_axis
            << '}';
        return;
    }
    json_string(out, "?");
    out << '}';
}

StorageActionKind classify_unwritten_op(LayoutOpKind kind) noexcept {
    switch (kind) {
    case LayoutOpKind::Cast:
    case LayoutOpKind::Dequantize:
        return StorageActionKind::TileMap;
    case LayoutOpKind::View:
    case LayoutOpKind::Alias:
        return StorageActionKind::CreateView;
    case LayoutOpKind::Drop:
        return StorageActionKind::Release;
    case LayoutOpKind::AttachMetadata:
        return StorageActionKind::Attach;
    default:
        return StorageActionKind::Transform;
    }
}

StorageTransformKind transform_kind_for_op(LayoutOpKind kind) noexcept
{
    switch (kind) {
    case LayoutOpKind::Slice:
        return StorageTransformKind::Slice;
    case LayoutOpKind::Concat:
        return StorageTransformKind::Concat;
    case LayoutOpKind::QuantizeRuntime:
        return StorageTransformKind::Quantize;
    case LayoutOpKind::Materialize:
        return StorageTransformKind::Materialize;
    case LayoutOpKind::Dequantize:
        return StorageTransformKind::Decode;
    case LayoutOpKind::Deinterleave:
        return StorageTransformKind::Deinterleave;
    case LayoutOpKind::RepackLayout:
        return StorageTransformKind::Repack;
    case LayoutOpKind::StackGroups:
        return StorageTransformKind::Stack;
    default:
        return StorageTransformKind::None;
    }
}

StorageInstrPayload payload_for_op(const LayoutOp& op)
{
    if (op.kind == LayoutOpKind::Slice) {
        return StorageSlicePayload{
            .axis = layout_op_slice_axis(op),
            .start = layout_op_slice_start(op),
            .length = layout_op_slice_length(op),
            .shard_axis = layout_op_shard_axis(op),
        };
    }
    if (op.kind == LayoutOpKind::Deinterleave) {
        return StorageAxisPayload{
            .shard_axis = layout_op_shard_axis(op),
        };
    }
    if (op.kind == LayoutOpKind::View ||
        op.kind == LayoutOpKind::Alias) {
        return StorageViewPayload{
            .axis = (layout_op_rows(op) > 0 ||
                     layout_op_row_offset(op) > 0) ? 0 : -1,
            .start = layout_op_row_offset(op),
            .length = layout_op_rows(op),
        };
    }
    return std::monostate{};
}

bool can_coalesce_adjacent_extent_writes(
    const ExtentWrite& a,
    const ExtentWrite& b)
{
    if (!a.contiguous || !b.contiguous) return false;
    if (a.op_index != b.op_index ||
        a.expr_id != b.expr_id ||
        a.binding_index != b.binding_index ||
        a.raw_name != b.raw_name ||
        a.output_name != b.output_name ||
        a.source_path != b.source_path ||
        a.source_shard_id != b.source_shard_id ||
        a.source_offset_bytes + a.source_span_bytes !=
            b.source_offset_bytes ||
        a.dst_offset_bytes + a.bytes != b.dst_offset_bytes) {
        return false;
    }
    if (a.slices.size() != 1 || b.slices.size() != 1 ||
        a.slices[0].axis != 0 || b.slices[0].axis != 0) {
        return false;
    }
    if (a.slices[0].start + a.slices[0].length != b.slices[0].start) {
        return false;
    }
    if (a.dst_shape.size() != b.dst_shape.size() || a.dst_shape.empty()) {
        return false;
    }
    for (std::size_t i = 1; i < a.dst_shape.size(); ++i) {
        if (a.dst_shape[i] != b.dst_shape[i]) return false;
    }
    return true;
}

void optimize_extent_writes(std::vector<ExtentWrite>& writes)
{
    if (writes.size() < 2) return;
    std::vector<ExtentWrite> optimized;
    optimized.reserve(writes.size());
    for (const auto& write : writes) {
        if (!optimized.empty() &&
            can_coalesce_adjacent_extent_writes(optimized.back(), write)) {
            auto& prev = optimized.back();
            prev.slices[0].length += write.slices[0].length;
            prev.dst_shape[0] += write.dst_shape[0];
            prev.bytes += write.bytes;
            prev.source_span_bytes += write.source_span_bytes;
            prev.range_count = 1;
            prev.contiguous = true;
            continue;
        }
        optimized.push_back(write);
    }
    writes.swap(optimized);
}

StorageInstrKind schedule_kind_for_action(
    StorageActionKind action) noexcept
{
    switch (action) {
    case StorageActionKind::Allocate:
        return StorageInstrKind::Allocate;
    case StorageActionKind::ExtentWrite:
        return StorageInstrKind::ExtentWrite;
    case StorageActionKind::TileMap:
        return StorageInstrKind::TileMap;
    case StorageActionKind::Transform:
        return StorageInstrKind::Transform;
    case StorageActionKind::Attach:
        return StorageInstrKind::Attach;
    case StorageActionKind::CreateView:
        return StorageInstrKind::CreateView;
    case StorageActionKind::Release:
        return StorageInstrKind::Release;
    case StorageActionKind::Finalize:
        return StorageInstrKind::Finalize;
    }
    return StorageInstrKind::Transform;
}

void append_unique(std::vector<std::size_t>& dst, std::size_t value)
{
    if (std::find(dst.begin(), dst.end(), value) == dst.end()) {
        dst.push_back(value);
    }
}

void append_unique(
    std::vector<std::string>& dst,
    const std::string& value)
{
    if (!value.empty() &&
        std::find(dst.begin(), dst.end(), value) == dst.end()) {
        dst.push_back(value);
    }
}

void append_producer_dependencies(
    const std::unordered_map<std::string, std::vector<std::size_t>>& producers,
    const std::string& name,
    std::vector<std::size_t>& deps)
{
    const auto it = producers.find(name);
    if (it == producers.end()) return;
    for (const auto dep : it->second) append_unique(deps, dep);
}

std::vector<std::string> op_dependency_names(
    const LayoutPlan& plan,
    const LayoutOp& op)
{
    std::vector<std::string> inputs = layout_op_inputs(op);
    if (op.kind == LayoutOpKind::Drop && inputs.empty()) {
        append_unique(inputs, layout_op_output(op));
    }
    if (op.kind == LayoutOpKind::AttachMetadata) {
        append_unique(inputs, layout_op_output(op));
        const TensorDecl& spec = tensor_spec_for(plan, layout_op_output(op));
        append_unique(inputs, spec.quant.scale_tensor);
        append_unique(inputs, spec.quant.zero_point_tensor);
    }
    return inputs;
}

std::vector<std::string> op_output_names(const LayoutOp& op)
{
    std::vector<std::string> outputs;
    if (op.kind != LayoutOpKind::Drop) {
        append_unique(outputs, layout_op_output(op));
        append_unique(outputs, layout_op_secondary_output(op));
    }
    return outputs;
}

AlgebraBindingRef binding_for_outputs(
    const LayoutPlan& plan,
    const std::vector<std::string>& outputs)
{
    for (const auto& output : outputs) {
        AlgebraBindingRef binding = realize_binding_for_output(plan, output);
        if (binding.root_expr != nullptr) return binding;
    }
    return {};
}

bool is_extent_write_step(const StorageInstr& step) noexcept
{
    return !step.extent_write_indices.empty();
}

bool source_order_before(
    const StorageProgram& plan,
    const StorageInstr& a,
    const StorageInstr& b)
{
    const auto& wa = plan.extent_writes[a.extent_write_indices.front()];
    const auto& wb = plan.extent_writes[b.extent_write_indices.front()];
    if (wa.source_path != wb.source_path) {
        return wa.source_path < wb.source_path;
    }
    if (wa.source_shard_id != wb.source_shard_id) {
        return wa.source_shard_id < wb.source_shard_id;
    }
    if (wa.source_offset_bytes != wb.source_offset_bytes) {
        return wa.source_offset_bytes < wb.source_offset_bytes;
    }
    return a.step_index < b.step_index;
}

std::vector<StorageInstr> schedule_steps_file_ordered(
    const StorageProgram& plan,
    std::vector<StorageInstr> steps)
{
    std::vector<std::vector<std::size_t>> dependents(steps.size());
    std::vector<std::size_t> indegree(steps.size(), 0);
    for (const auto& step : steps) {
        if (step.step_index >= steps.size()) {
            throw std::runtime_error(
                "storage program: schedule step index out of range");
        }
        indegree[step.step_index] = step.dependencies.size();
        for (const auto dep : step.dependencies) {
            if (dep >= steps.size()) {
                throw std::runtime_error(
                    "storage program: schedule dependency out of range");
            }
            dependents[dep].push_back(step.step_index);
        }
    }

    std::vector<std::size_t> ready;
    for (const auto& step : steps) {
        if (indegree[step.step_index] == 0) {
            ready.push_back(step.step_index);
        }
    }

    std::vector<StorageInstr> ordered;
    ordered.reserve(steps.size());
    while (!ready.empty()) {
        auto chosen = ready.begin();
        for (auto it = ready.begin(); it != ready.end(); ++it) {
            const auto& cur = steps[*it];
            const auto& best = steps[*chosen];
            const bool cur_byte = is_extent_write_step(cur);
            const bool best_byte = is_extent_write_step(best);
            if (cur_byte && best_byte) {
                if (source_order_before(plan, cur, best)) chosen = it;
            } else if (cur_byte != best_byte) {
                if (cur_byte) chosen = it;
            } else if (cur.step_index < best.step_index) {
                chosen = it;
            }
        }
        const std::size_t step_index = *chosen;
        ready.erase(chosen);
        ordered.push_back(steps[step_index]);
        for (const auto dependent : dependents[step_index]) {
            if (--indegree[dependent] == 0) {
                ready.push_back(dependent);
            }
        }
    }
    if (ordered.size() != steps.size()) {
        throw std::runtime_error(
            "storage program: dependency cycle in storage schedule");
    }
    return ordered;
}

void build_storage_schedule(
    const LayoutPlan& layout_plan,
    StorageProgram& storage_program)
{
    std::vector<std::vector<std::size_t>> writes_by_op(
        layout_plan.ops.size());
    for (std::size_t i = 0; i < storage_program.extent_writes.size(); ++i) {
        const auto op_index = storage_program.extent_writes[i].op_index;
        if (op_index >= layout_plan.ops.size()) {
            throw std::runtime_error(
                "storage program: extent write op index out of range");
        }
        writes_by_op[op_index].push_back(i);
    }
    std::vector<std::vector<std::size_t>> transforms_by_op(
        layout_plan.ops.size());
    for (std::size_t i = 0; i < storage_program.tile_maps.size(); ++i) {
        const auto op_index = storage_program.tile_maps[i].op_index;
        if (op_index >= layout_plan.ops.size()) {
            throw std::runtime_error(
                "storage program: tile map op index out of range");
        }
        transforms_by_op[op_index].push_back(i);
    }

    std::vector<StorageInstr> steps;
    std::unordered_map<std::string, std::vector<std::size_t>> producers;
    auto dependencies_for = [&](const std::vector<std::string>& names) {
        std::vector<std::size_t> deps;
        for (const auto& name : names) {
            append_producer_dependencies(producers, name, deps);
        }
        std::sort(deps.begin(), deps.end());
        return deps;
    };
    auto add_step = [&](StorageInstr step) {
        step.step_index = steps.size();
        steps.push_back(std::move(step));
        return steps.back().step_index;
    };

    for (std::size_t op_index = 0; op_index < layout_plan.ops.size(); ++op_index) {
        if (is_raw_copy_cast_drop_sequence(layout_plan, op_index)) {
            const std::size_t cast_index = op_index + 1;
            const LayoutOp& producer = layout_plan.ops[op_index];
            if (writes_by_op[op_index].empty() ||
                transforms_by_op[cast_index].empty()) {
                throw std::runtime_error(
                    "storage program: fused raw-cast sequence is missing "
                    "byte extents or tile map");
            }
            auto deps = dependencies_for(
                op_dependency_names(layout_plan, producer));
            for (const auto transform_index : transforms_by_op[cast_index]) {
                const auto& transform =
                    storage_program.tile_maps[transform_index];
                const std::size_t step_index = add_step(StorageInstr{
                    .kind = StorageInstrKind::TileMap,
                    .transform_kind = StorageTransformKind::None,
                    .op_index = cast_index,
                    .expr_id = transform.expr_id,
                    .binding_index = transform.binding_index,
                    .extent_write_indices = writes_by_op[op_index],
                    .tile_map_index = transform_index,
                    .inputs = transform.inputs,
                    .outputs = {transform.output_name},
                    .dependencies = deps,
                });
                producers[transform.output_name] = {step_index};
            }
            op_index += 2;
            continue;
        }

        const LayoutOp& op = layout_plan.ops[op_index];
        const auto op_inputs = op_dependency_names(layout_plan, op);
        auto deps = dependencies_for(op_inputs);
        std::unordered_map<std::string, std::vector<std::size_t>> op_producers;

        if (!writes_by_op[op_index].empty()) {
            for (const auto write_index : writes_by_op[op_index]) {
                const auto& write = storage_program.extent_writes[write_index];
                const std::size_t step_index = add_step(StorageInstr{
                    .kind = StorageInstrKind::ExtentWrite,
                    .transform_kind = StorageTransformKind::None,
                    .op_index = op_index,
                    .expr_id = write.expr_id,
                    .binding_index = write.binding_index,
                    .extent_write_indices = {write_index},
                    .tile_map_index = kInvalidStorageId,
                    .inputs = {write.raw_name},
                    .outputs = {write.output_name},
                    .dependencies = deps,
                });
                op_producers[write.output_name].push_back(step_index);
            }

            for (auto& [name, produced_by] : op_producers) {
                std::sort(produced_by.begin(), produced_by.end());
                producers[name] = std::move(produced_by);
            }

            if (op.kind == LayoutOpKind::AxisConcat) {
                std::vector<std::size_t> view_deps;
                append_producer_dependencies(
                    producers, layout_op_output(op), view_deps);
                std::vector<std::string> outputs;
                for (const auto& source : layout_op_sources(op)) {
                    append_unique(outputs, source.view_name);
                }
                if (!outputs.empty()) {
                    const AlgebraBindingRef view_binding =
                        binding_for_outputs(layout_plan, outputs);
                    const std::size_t view_step = add_step(StorageInstr{
                        .kind = StorageInstrKind::CreateView,
                        .transform_kind = StorageTransformKind::None,
                        .op_index = op_index,
                        .expr_id = view_binding.root,
                        .binding_index = view_binding.binding_index,
                        .extent_write_indices = {},
                        .tile_map_index = kInvalidStorageId,
                        .inputs = {layout_op_output(op)},
                        .outputs = outputs,
                        .dependencies = view_deps,
                    });
                    for (const auto& output : outputs) {
                        producers[output] = {view_step};
                    }
                }
            }
            continue;
        }

        if (!transforms_by_op[op_index].empty()) {
            for (const auto transform_index : transforms_by_op[op_index]) {
                const auto& transform =
                    storage_program.tile_maps[transform_index];
                const std::size_t step_index = add_step(StorageInstr{
                    .kind = StorageInstrKind::TileMap,
                    .transform_kind = StorageTransformKind::None,
                    .op_index = op_index,
                    .expr_id = transform.expr_id,
                    .binding_index = transform.binding_index,
                    .extent_write_indices = {},
                    .tile_map_index = transform_index,
                    .inputs = transform.inputs,
                    .outputs = {transform.output_name},
                    .dependencies = deps,
                });
                producers[transform.output_name] = {step_index};
            }
            continue;
        }

        const StorageInstrKind kind = schedule_kind_for_action(
            classify_unwritten_op(op.kind));
        const std::vector<std::string> outputs = op_output_names(op);
        const AlgebraBindingRef binding =
            binding_for_outputs(layout_plan, outputs);
        const std::size_t step_index = add_step(StorageInstr{
            .kind = kind,
            .transform_kind = transform_kind_for_op(op.kind),
            .op_index = op_index,
            .expr_id = binding.root,
            .binding_index = binding.binding_index,
            .extent_write_indices = {},
            .tile_map_index = kInvalidStorageId,
            .inputs = op_inputs,
            .outputs = outputs,
            .dependencies = deps,
            .payload = payload_for_op(op),
        });
        if (op.kind == LayoutOpKind::Drop) {
            if (layout_op_inputs(op).empty()) {
                producers.erase(layout_op_output(op));
            } else {
                for (const auto& input : layout_op_inputs(op)) {
                    producers.erase(input);
                }
            }
        } else {
            for (const auto& output : outputs) {
                producers[output] = {step_index};
            }
        }
    }

    storage_program.schedule =
        schedule_steps_file_ordered(storage_program, std::move(steps));
    storage_program.scheduled_extent_writes.clear();
    storage_program.scheduled_extent_writes.reserve(
        storage_program.extent_writes.size());
    for (const auto& step : storage_program.schedule) {
        if (!is_extent_write_step(step)) continue;
        for (const auto write_index : step.extent_write_indices) {
            storage_program.scheduled_extent_writes.push_back(write_index);
        }
    }
    storage_program.memory.scheduled_step_count =
        static_cast<std::uint64_t>(storage_program.schedule.size());
    storage_program.memory.file_ordered_extent_write_count =
        static_cast<std::uint64_t>(
            storage_program.scheduled_extent_writes.size());
}

}  // namespace

std::vector<ExtentWrite> lower_extent_writes_for_step(
    const LayoutOp& op,
    std::size_t op_index,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size)
{
    switch (op.kind) {
    case LayoutOpKind::Read:
    case LayoutOpKind::Copy:
    case LayoutOpKind::Shard:
        return lower_raw_like_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LayoutOpKind::RowRangeShard:
        return lower_row_range_shard_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LayoutOpKind::AxisConcat:
        return lower_axis_concat_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LayoutOpKind::GroupedSliceConcat:
        return lower_grouped_slice_concat_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LayoutOpKind::GroupedSlice:
        return lower_grouped_slice_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LayoutOpKind::StackGroups:
        return lower_stack_groups_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    default:
        return {};
    }
}

StorageProgram build_storage_program_from_algebra_only(
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size,
    std::uint64_t transform_tile_bytes,
    StorageOptimizerConfig optimizer)
{
    validate_layout_plan(plan);
    StorageProgram storage;
    storage.memory.persistent_bytes = plan.memory.persistent_bytes;
    storage.memory.layout_max_temporary_bytes = plan.memory.max_temporary_bytes;
    storage.memory.max_extent_temporary_bytes = plan.memory.max_temporary_bytes;

    std::vector<StorageInstr> steps;
    std::unordered_map<std::string, std::vector<std::size_t>> producers;
    std::unordered_map<LayoutExprId, std::vector<std::string>> materialized;

    auto expr_output_name = [&](LayoutExprId id) -> std::string {
        if (id >= plan.algebra.exprs.size()) return {};
        const auto& expr = plan.algebra.exprs[id];
        if (!expr.runtime_name.empty()) return expr.runtime_name;
        if (!expr.decl.name.empty()) return expr.decl.name;
        return expr.raw_name;
    };

    auto dependencies_for = [&](const std::vector<std::string>& names) {
        std::vector<std::size_t> deps;
        for (const auto& name : names) {
            append_producer_dependencies(producers, name, deps);
        }
        std::sort(deps.begin(), deps.end());
        return deps;
    };

    auto add_step = [&](StorageInstr step) {
        step.step_index = steps.size();
        steps.push_back(std::move(step));
        return steps.back().step_index;
    };

    auto remember_materialized =
        [&](LayoutExprId id, const std::vector<std::string>& names) {
            auto& out = materialized[id];
            for (const auto& name : names) append_unique(out, name);
        };

    auto already_materialized =
        [&](LayoutExprId id, const std::string& requested) -> std::string {
            const auto it = materialized.find(id);
            if (it == materialized.end()) return {};
            if (requested.empty()) {
                return it->second.empty() ? std::string{} : it->second.front();
            }
            return std::find(it->second.begin(), it->second.end(), requested) !=
                    it->second.end()
                ? requested
                : std::string{};
        };

    auto record_writes =
        [&](std::vector<ExtentWrite> writes,
            const std::vector<std::size_t>& deps) {
        const std::uint64_t unoptimized_write_count =
            static_cast<std::uint64_t>(writes.size());
        if (optimizer.enabled && optimizer.coalesce_adjacent) {
            optimize_extent_writes(writes);
        }
        const std::uint64_t write_count =
            static_cast<std::uint64_t>(writes.size());
        if (unoptimized_write_count > write_count) {
            storage.memory.coalesced_extent_write_count +=
                unoptimized_write_count - write_count;
        }
        for (const auto& write : writes) {
            storage.memory.checkpoint_read_bytes += write.bytes;
            storage.memory.device_write_bytes += write.bytes;
            storage.memory.extent_write_count += 1;
            storage.memory.algebra_extent_write_count += 1;
            storage.memory.extent_range_count += write.range_count;
        }
        for (auto& write : writes) {
            const std::size_t write_index = storage.extent_writes.size();
            const std::string output_name = write.output_name;
            const std::string raw_name = write.raw_name;
            storage.extent_writes.push_back(std::move(write));
            const std::size_t step_index = add_step(StorageInstr{
                .kind = StorageInstrKind::ExtentWrite,
                .transform_kind = StorageTransformKind::None,
                .op_index = kInvalidStorageId,
                .expr_id = storage.extent_writes[write_index].expr_id,
                .binding_index = storage.extent_writes[write_index].binding_index,
                .extent_write_indices = {write_index},
                .tile_map_index = kInvalidStorageId,
                .inputs = {raw_name},
                .outputs = {output_name},
                .dependencies = deps,
            });
            producers[output_name].push_back(step_index);
        }
    };

    auto lower_direct_writes =
        [&](LayoutExprId value_id,
            std::size_t binding_index,
            const std::string& output_name) {
            if (value_id >= plan.algebra.exprs.size()) {
                return std::vector<ExtentWrite>{};
            }
            const LayoutExpr& value = plan.algebra.exprs[value_id];
            std::vector<ExtentWrite> writes;
            const std::string op_kind = std::string("Algebra.") +
                layout_expr_kind_name(value.kind);
            const LayoutExprId write_expr_id =
                binding_index != kInvalidStorageId &&
                    binding_index < plan.algebra.bindings.size()
                ? plan.algebra.bindings[binding_index].root
                : value_id;
            if (try_lower_join_expr_to_writes(
                    value, kInvalidStorageId, write_expr_id, binding_index,
                    op_kind, plan, metadata, output_name,
                    tp_rank, tp_size, writes)) {
                return writes;
            }
            writes.clear();
            if (lower_source_expr_to_write(
                    value, kInvalidStorageId, write_expr_id, binding_index,
                    op_kind, plan, metadata, output_name,
                    tensor_spec_for(plan, output_name).shape, 0,
                    tp_rank, tp_size, writes)) {
                return writes;
            }
            return std::vector<ExtentWrite>{};
        };

    std::function<std::string(LayoutExprId, std::string, std::size_t)>
        materialize_expr;
    materialize_expr =
        [&](LayoutExprId id,
            std::string requested_name,
            std::size_t binding_index) -> std::string {
            if (id >= plan.algebra.exprs.size()) {
                throw std::runtime_error(
                    "storage program: algebra expr id out of range");
            }
            const LayoutExpr& expr = plan.algebra.exprs[id];
            if (requested_name.empty()) requested_name = expr_output_name(id);
            if (const auto existing =
                    already_materialized(id, requested_name);
                !existing.empty()) {
                return existing;
            }

            if (expr.kind == LayoutExprKind::Realize) {
                if (expr.inputs.size() != 1) {
                    throw std::runtime_error(
                        "storage program: Realize expects one input");
                }
                const std::string name = materialize_expr(
                    expr.inputs.front(),
                    requested_name.empty() ? expr.runtime_name : requested_name,
                    binding_index);
                remember_materialized(id, {name});
                return name;
            }

            if (expr.kind == LayoutExprKind::View) {
                if (expr.inputs.size() != 1) {
                    throw std::runtime_error(
                        "storage program: View expects one input");
                }
                const std::string input =
                    materialize_expr(expr.inputs.front(), {}, binding_index);
                const auto deps = dependencies_for({input});
                const std::string output =
                    requested_name.empty() ? expr.runtime_name : requested_name;
                const std::size_t step_index = add_step(StorageInstr{
                    .kind = StorageInstrKind::CreateView,
                    .transform_kind = StorageTransformKind::None,
                    .op_index = kInvalidStorageId,
                    .expr_id = id,
                    .binding_index = binding_index,
                    .extent_write_indices = {},
                    .tile_map_index = kInvalidStorageId,
                    .inputs = {input},
                    .outputs = {output},
                    .dependencies = deps,
                    .payload = StorageViewPayload{
                        .axis = expr.axis,
                        .start = expr.start,
                        .length = expr.length,
                    },
                });
                producers[output] = {step_index};
                remember_materialized(id, {output});
                return output;
            }

            if (!requested_name.empty()) {
                auto writes = lower_direct_writes(id, binding_index, requested_name);
                if (!writes.empty()) {
                    record_writes(std::move(writes), {});
                    remember_materialized(id, {requested_name});
                    return requested_name;
                }
            }

            auto materialize_inputs = [&]() {
                std::vector<std::string> inputs;
                inputs.reserve(expr.inputs.size());
                for (const auto input_id : expr.inputs) {
                    inputs.push_back(materialize_expr(input_id, {}, binding_index));
                }
                return inputs;
            };

            const std::string output =
                requested_name.empty() ? expr_output_name(id) : requested_name;
            switch (expr.kind) {
            case LayoutExprKind::Source:
            case LayoutExprKind::Partition:
            case LayoutExprKind::Join:
                throw std::runtime_error(
                    "storage program: algebra expr '" +
                    std::string(layout_expr_kind_name(expr.kind)) +
                    "' for '" + output + "' cannot lower to storage");
            case LayoutExprKind::Select: {
                const auto inputs = materialize_inputs();
                const auto deps = dependencies_for(inputs);
                const std::size_t step_index = add_step(StorageInstr{
                    .kind = StorageInstrKind::Transform,
                    .transform_kind = StorageTransformKind::Slice,
                    .op_index = kInvalidStorageId,
                    .expr_id = id,
                    .binding_index = binding_index,
                    .extent_write_indices = {},
                    .tile_map_index = kInvalidStorageId,
                    .inputs = inputs,
                    .outputs = {output},
                    .dependencies = deps,
                    .payload = StorageSlicePayload{
                        .axis = expr.axis,
                        .start = expr.start,
                        .length = expr.length,
                        .shard_axis = expr.partitions,
                    },
                });
                producers[output] = {step_index};
                remember_materialized(id, {output});
                return output;
            }
            case LayoutExprKind::Stack:
            case LayoutExprKind::Reorder: {
                const auto inputs = materialize_inputs();
                const auto deps = dependencies_for(inputs);
                const StorageTransformKind transform =
                    expr.kind == LayoutExprKind::Stack
                        ? StorageTransformKind::Stack
                        : StorageTransformKind::Repack;
                const std::vector<std::string> outputs =
                    expr.secondary_runtime_name.empty()
                        ? std::vector<std::string>{output}
                        : std::vector<std::string>{output,
                              expr.secondary_runtime_name};
                const std::size_t step_index = add_step(StorageInstr{
                    .kind = StorageInstrKind::Transform,
                    .transform_kind = transform,
                    .op_index = kInvalidStorageId,
                    .expr_id = id,
                    .binding_index = binding_index,
                    .extent_write_indices = {},
                    .tile_map_index = kInvalidStorageId,
                    .inputs = inputs,
                    .outputs = outputs,
                    .dependencies = deps,
                });
                for (const auto& name : outputs) producers[name] = {step_index};
                remember_materialized(id, outputs);
                return output;
            }
            case LayoutExprKind::Unzip: {
                const auto inputs = materialize_inputs();
                const auto deps = dependencies_for(inputs);
                const std::vector<std::string> outputs{
                    expr.runtime_name.empty() ? output : expr.runtime_name,
                    expr.secondary_runtime_name,
                };
                const std::size_t step_index = add_step(StorageInstr{
                    .kind = StorageInstrKind::Transform,
                    .transform_kind = StorageTransformKind::Deinterleave,
                    .op_index = kInvalidStorageId,
                    .expr_id = id,
                    .binding_index = binding_index,
                    .extent_write_indices = {},
                    .tile_map_index = kInvalidStorageId,
                    .inputs = inputs,
                    .outputs = outputs,
                    .dependencies = deps,
                    .payload = StorageAxisPayload{.shard_axis = expr.axis},
                });
                for (const auto& name : outputs) producers[name] = {step_index};
                remember_materialized(id, outputs);
                if (!requested_name.empty() &&
                    std::find(outputs.begin(), outputs.end(), requested_name) !=
                        outputs.end()) {
                    return requested_name;
                }
                return outputs.front();
            }
            case LayoutExprKind::Cast:
            case LayoutExprKind::Encode:
            case LayoutExprKind::Decode:
            case LayoutExprKind::Transcode: {
                const auto inputs = materialize_inputs();
                const auto deps = dependencies_for(inputs);
                const auto out_it = plan.tensors.find(output);
                const TensorDecl& out =
                    out_it == plan.tensors.end() ? expr.decl : out_it->second;
                std::uint64_t input_bytes = 0;
                for (const auto& input : inputs) {
                    const auto spec_it = plan.tensors.find(input);
                    if (spec_it != plan.tensors.end()) {
                        input_bytes += tensor_nbytes(spec_it->second);
                    }
                }
                const std::uint64_t scratch = 0;
                const std::size_t tile_index = storage.tile_maps.size();
                storage.tile_maps.push_back(TileMap{
                    .op_index = kInvalidStorageId,
                    .expr_id = id,
                    .binding_index = binding_index,
                    .kind = expr.kind == LayoutExprKind::Cast
                        ? TileMapKind::Cast
                        : expr.kind == LayoutExprKind::Encode
                            ? TileMapKind::Encode
                            : expr.kind == LayoutExprKind::Transcode
                                ? TileMapKind::Transcode
                                : TileMapKind::Decode,
                    .output_name = output,
                    .inputs = inputs,
                    .input_bytes = input_bytes,
                    .output_bytes = tensor_nbytes(out),
                    .tile_bytes = transform_tile_bytes,
                    .scratch_bytes = scratch,
                });
                storage.memory.tile_map_count += 1;
                storage.memory.max_transform_scratch_bytes =
                    std::max(storage.memory.max_transform_scratch_bytes, scratch);
                const std::size_t step_index = add_step(StorageInstr{
                    .kind = StorageInstrKind::TileMap,
                    .transform_kind = StorageTransformKind::None,
                    .op_index = kInvalidStorageId,
                    .expr_id = id,
                    .binding_index = binding_index,
                    .extent_write_indices = {},
                    .tile_map_index = tile_index,
                    .inputs = inputs,
                    .outputs = {output},
                    .dependencies = deps,
                });
                producers[output] = {step_index};
                remember_materialized(id, {output});
                return output;
            }
            case LayoutExprKind::Attach:
            case LayoutExprKind::Release:
                throw std::runtime_error(
                    "storage program: side-effect algebra expr cannot be materialized");
            case LayoutExprKind::View:
            case LayoutExprKind::Realize:
                break;
            }
            throw std::runtime_error(
                "storage program: unhandled algebra expr for '" + output + "'");
        };

    for (std::size_t binding_index = 0;
         binding_index < plan.algebra.bindings.size();
         ++binding_index) {
        const auto& binding = plan.algebra.bindings[binding_index];
        (void)materialize_expr(
            binding.root, binding.runtime_name, binding_index);
    }

    for (LayoutExprId id = 0; id < plan.algebra.exprs.size(); ++id) {
        const auto& expr = plan.algebra.exprs[id];
        if (expr.kind == LayoutExprKind::Attach) {
            const auto deps = dependencies_for({expr.runtime_name});
            const std::size_t step_index = add_step(StorageInstr{
                .kind = StorageInstrKind::Attach,
                .transform_kind = StorageTransformKind::None,
                .op_index = kInvalidStorageId,
                .expr_id = id,
                .binding_index = kInvalidStorageId,
                .extent_write_indices = {},
                .tile_map_index = kInvalidStorageId,
                .inputs = {expr.runtime_name},
                .outputs = {expr.runtime_name},
                .dependencies = deps,
            });
            producers[expr.runtime_name] = {step_index};
        } else if (expr.kind == LayoutExprKind::Release) {
            std::vector<std::string> inputs;
            for (const auto input_id : expr.inputs) {
                const auto name = materialize_expr(input_id, {}, kInvalidStorageId);
                if (!name.empty()) inputs.push_back(name);
            }
            const auto deps = dependencies_for(inputs);
            (void)add_step(StorageInstr{
                .kind = StorageInstrKind::Release,
                .transform_kind = StorageTransformKind::None,
                .op_index = kInvalidStorageId,
                .expr_id = id,
                .binding_index = kInvalidStorageId,
                .extent_write_indices = {},
                .tile_map_index = kInvalidStorageId,
                .inputs = inputs,
                .outputs = {},
                .dependencies = deps,
            });
            for (const auto& input : inputs) producers.erase(input);
        }
    }

    storage.memory.optimized_extent_write_count =
        static_cast<std::uint64_t>(storage.extent_writes.size());
    storage.memory.max_temporary_bytes = std::max(
        storage.memory.max_extent_temporary_bytes,
        storage.memory.max_transform_scratch_bytes);
    storage.memory.estimated_peak_bytes =
        storage.memory.persistent_bytes + storage.memory.max_temporary_bytes;
    storage.schedule = schedule_steps_file_ordered(storage, std::move(steps));
    storage.scheduled_extent_writes.clear();
    storage.scheduled_extent_writes.reserve(storage.extent_writes.size());
    for (const auto& step : storage.schedule) {
        if (!is_extent_write_step(step)) continue;
        for (const auto write_index : step.extent_write_indices) {
            storage.scheduled_extent_writes.push_back(write_index);
        }
    }
    storage.memory.scheduled_step_count =
        static_cast<std::uint64_t>(storage.schedule.size());
    storage.memory.file_ordered_extent_write_count =
        static_cast<std::uint64_t>(storage.scheduled_extent_writes.size());
    validate_storage_program(plan, storage);
    return storage;
}

StorageProgram build_storage_program(
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size,
    std::uint64_t transform_tile_bytes,
    StorageOptimizerConfig optimizer)
{
    validate_layout_plan(plan);
    if (plan.ops.empty() && !plan.algebra.bindings.empty()) {
        return build_storage_program_from_algebra_only(
            plan, metadata, tp_rank, tp_size,
            transform_tile_bytes, optimizer);
    }
    StorageProgram storage;
    storage.memory.persistent_bytes = plan.memory.persistent_bytes;
    storage.memory.layout_max_temporary_bytes = plan.memory.max_temporary_bytes;

    for (std::size_t i = 0; i < plan.ops.size(); ++i) {
        const LayoutOp& op = plan.ops[i];
        const AlgebraBindingRef op_binding =
            realize_binding_for_output(plan, layout_op_output(op));
        auto writes = lower_extent_writes_for_algebra_root(
            op, i, plan, metadata, tp_rank, tp_size);
        const bool lowered_from_algebra = !writes.empty();
        if (writes.empty()) {
            writes = lower_extent_writes_for_step(
                op, i, plan, metadata, tp_rank, tp_size);
            attach_algebra_binding(writes, op_binding);
        }
        const std::uint64_t unoptimized_op_write_count =
            static_cast<std::uint64_t>(writes.size());
        if (optimizer.enabled && optimizer.coalesce_adjacent) {
            optimize_extent_writes(writes);
        }
        const std::uint64_t op_write_count =
            static_cast<std::uint64_t>(writes.size());
        if (unoptimized_op_write_count > op_write_count) {
            storage.memory.coalesced_extent_write_count +=
                unoptimized_op_write_count - op_write_count;
        }
        for (const auto& write : writes) {
            storage.memory.checkpoint_read_bytes += write.bytes;
            storage.memory.device_write_bytes += write.bytes;
            storage.memory.extent_write_count += 1;
            if (lowered_from_algebra) {
                storage.memory.algebra_extent_write_count += 1;
            } else {
                storage.memory.planner_extent_write_count += 1;
            }
            storage.memory.extent_range_count += write.range_count;
        }
        storage.extent_writes.insert(
            storage.extent_writes.end(),
            std::make_move_iterator(writes.begin()),
            std::make_move_iterator(writes.end()));

        if (op.kind == LayoutOpKind::Cast || op.kind == LayoutOpKind::Dequantize) {
            const TensorDecl& out = tensor_spec_for(plan, layout_op_output(op));
            std::uint64_t input_bytes = 0;
            for (const auto& input : layout_op_inputs(op)) {
                const auto spec_it = plan.tensors.find(input);
                if (spec_it != plan.tensors.end()) {
                    input_bytes += tensor_nbytes(spec_it->second);
                }
            }
            const std::uint64_t output_bytes = tensor_nbytes(out);
            const std::uint64_t scratch = transform_scratch_bytes_for_op(
                plan, i, transform_tile_bytes);
            storage.tile_maps.push_back(TileMap{
                .op_index = i,
                .expr_id = op_binding.root,
                .binding_index = op_binding.binding_index,
                .kind = op.kind == LayoutOpKind::Cast
                    ? TileMapKind::Cast
                    : TileMapKind::Decode,
                .output_name = layout_op_output(op),
                .inputs = layout_op_inputs(op),
                .input_bytes = input_bytes,
                .output_bytes = output_bytes,
                .tile_bytes = transform_tile_bytes,
                .scratch_bytes = scratch,
            });
            storage.memory.max_transform_scratch_bytes =
                std::max(storage.memory.max_transform_scratch_bytes, scratch);
            storage.memory.tile_map_count += 1;
        }

        StorageOpCoverage coverage{
            .op_index = i,
            .expr_id = op_binding.root,
            .binding_index = op_binding.binding_index,
            .op_kind = layout_op_kind_name(op.kind),
            .action = op_write_count > 0
                ? StorageActionKind::ExtentWrite
                : classify_unwritten_op(op.kind),
            .extent_writes = op_write_count,
            .tile_maps =
                (op.kind == LayoutOpKind::Cast ||
                 op.kind == LayoutOpKind::Dequantize)
                    ? 1u
                    : 0u,
        };
        storage.coverage.push_back(std::move(coverage));
    }

    storage.memory.optimized_extent_write_count =
        static_cast<std::uint64_t>(storage.extent_writes.size());
    const StorageTempUsage temp_usage =
        compute_storage_temp_usage(plan, transform_tile_bytes);
    storage.memory.max_extent_temporary_bytes =
        temp_usage.max_resident_temp_bytes;
    storage.memory.max_transform_scratch_bytes =
        temp_usage.max_transform_scratch_bytes;
    storage.memory.max_temporary_bytes = temp_usage.max_total_temporary_bytes;
    storage.memory.estimated_peak_bytes =
        storage.memory.persistent_bytes + storage.memory.max_temporary_bytes;
    build_storage_schedule(plan, storage);
    validate_storage_program(plan, storage);
    return storage;
}

const char* storage_action_kind_name(StorageActionKind kind) noexcept {
    switch (kind) {
    case StorageActionKind::Allocate: return "Allocate";
    case StorageActionKind::ExtentWrite: return "ExtentWrite";
    case StorageActionKind::TileMap: return "TileMap";
    case StorageActionKind::Transform: return "Transform";
    case StorageActionKind::Attach: return "Attach";
    case StorageActionKind::CreateView: return "CreateView";
    case StorageActionKind::Release: return "Release";
    case StorageActionKind::Finalize: return "Finalize";
    }
    return "?";
}

const char* storage_instr_kind_name(
    StorageInstrKind kind) noexcept
{
    switch (kind) {
    case StorageInstrKind::Allocate: return "Allocate";
    case StorageInstrKind::ExtentWrite: return "ExtentWrite";
    case StorageInstrKind::TileMap: return "TileMap";
    case StorageInstrKind::Transform: return "Transform";
    case StorageInstrKind::Attach: return "Attach";
    case StorageInstrKind::CreateView: return "CreateView";
    case StorageInstrKind::Release: return "Release";
    case StorageInstrKind::Finalize: return "Finalize";
    }
    return "?";
}

const char* storage_transform_kind_name(
    StorageTransformKind kind) noexcept
{
    switch (kind) {
    case StorageTransformKind::None: return "None";
    case StorageTransformKind::Slice: return "Slice";
    case StorageTransformKind::Concat: return "Concat";
    case StorageTransformKind::Quantize: return "Quantize";
    case StorageTransformKind::Materialize: return "Materialize";
    case StorageTransformKind::Decode: return "Decode";
    case StorageTransformKind::Deinterleave: return "Deinterleave";
    case StorageTransformKind::Repack: return "Repack";
    case StorageTransformKind::Stack: return "Stack";
    }
    return "?";
}

const char* tile_map_kind_name(TileMapKind kind) noexcept {
    switch (kind) {
    case TileMapKind::Cast: return "Cast";
    case TileMapKind::Decode: return "Decode";
    case TileMapKind::Encode: return "Encode";
    case TileMapKind::Transcode: return "Transcode";
    case TileMapKind::Reblock: return "Reblock";
    case TileMapKind::Reorder: return "Reorder";
    }
    return "?";
}

void validate_storage_program(
    const LayoutPlan& layout_plan,
    const StorageProgram& storage_program)
{
    auto expr_reaches =
        [&](std::size_t root, std::size_t needle) {
            if (root >= layout_plan.algebra.exprs.size() ||
                needle >= layout_plan.algebra.exprs.size()) {
                return false;
            }
            std::vector<std::size_t> stack{root};
            std::unordered_set<std::size_t> seen;
            while (!stack.empty()) {
                const auto id = stack.back();
                stack.pop_back();
                if (!seen.insert(id).second) continue;
                if (id == needle) return true;
                for (const auto input : layout_plan.algebra.exprs[id].inputs) {
                    if (input < layout_plan.algebra.exprs.size()) {
                        stack.push_back(input);
                    }
                }
            }
            return false;
        };

    auto validate_algebra_ref =
        [&](std::size_t expr_id,
            std::size_t binding_index,
            const std::string& context) {
            if (expr_id != kInvalidStorageId &&
                expr_id >= layout_plan.algebra.exprs.size()) {
                throw std::runtime_error(
                    "storage program: " + context +
                    " expr id out of range");
            }
            if (binding_index != kInvalidStorageId) {
                if (binding_index >= layout_plan.algebra.bindings.size()) {
                    throw std::runtime_error(
                        "storage program: " + context +
                        " binding index out of range");
                }
                const auto& binding =
                    layout_plan.algebra.bindings[binding_index];
                if (expr_id != kInvalidStorageId &&
                    !expr_reaches(binding.root, expr_id)) {
                    throw std::runtime_error(
                        "storage program: " + context +
                        " binding does not reach expr id");
                }
            }
        };

    const bool has_layout_ops = !layout_plan.ops.empty();
    if (has_layout_ops) {
        if (storage_program.coverage.size() != layout_plan.ops.size()) {
            throw std::runtime_error(
                "storage program: coverage table does not match layout op count");
        }
        std::unordered_set<std::size_t> covered;
        covered.reserve(storage_program.coverage.size());
        for (const auto& coverage : storage_program.coverage) {
            if (coverage.op_index >= layout_plan.ops.size()) {
                throw std::runtime_error(
                    "storage program: coverage op index out of range");
            }
            validate_algebra_ref(
                coverage.expr_id, coverage.binding_index, "coverage");
            if (!covered.insert(coverage.op_index).second) {
                throw std::runtime_error(
                    "storage program: duplicate coverage for op " +
                    std::to_string(coverage.op_index));
            }
            const auto& layout_op = layout_plan.ops[coverage.op_index];
            if (coverage.op_kind != layout_op_kind_name(layout_op.kind)) {
                throw std::runtime_error(
                    "storage program: coverage kind mismatch for op " +
                    std::to_string(coverage.op_index));
            }
            switch (coverage.action) {
            case StorageActionKind::ExtentWrite:
                if (coverage.extent_writes == 0) {
                    throw std::runtime_error(
                        "storage program: extent-write op has no extent writes");
                }
                break;
            case StorageActionKind::TileMap:
                if (coverage.tile_maps == 0) {
                    throw std::runtime_error(
                        "storage program: tile map op has no transform record");
                }
                break;
            case StorageActionKind::Allocate:
            case StorageActionKind::Transform:
            case StorageActionKind::Attach:
            case StorageActionKind::CreateView:
            case StorageActionKind::Release:
            case StorageActionKind::Finalize:
                break;
            }
        }
    } else if (!storage_program.coverage.empty()) {
        throw std::runtime_error(
            "storage program: algebra-only program must not carry planner-op coverage");
    }

    for (const auto& write : storage_program.extent_writes) {
        if (has_layout_ops) {
            if (write.op_index >= layout_plan.ops.size()) {
                throw std::runtime_error(
                    "storage program: extent write op index out of range");
            }
        } else if (write.op_index != kInvalidStorageId) {
            throw std::runtime_error(
                "storage program: algebra-only extent write has layout op index");
        }
        validate_algebra_ref(write.expr_id, write.binding_index, "extent write");
    }
    for (const auto& tile : storage_program.tile_maps) {
        if (has_layout_ops) {
            if (tile.op_index >= layout_plan.ops.size()) {
                throw std::runtime_error(
                    "storage program: tile map op index out of range");
            }
        } else if (tile.op_index != kInvalidStorageId) {
            throw std::runtime_error(
                "storage program: algebra-only tile map has layout op index");
        }
        validate_algebra_ref(tile.expr_id, tile.binding_index, "tile map");
    }

    if (storage_program.scheduled_extent_writes.size() !=
        storage_program.extent_writes.size()) {
        throw std::runtime_error(
            "storage program: scheduled extent-write count mismatch");
    }
    std::unordered_set<std::size_t> scheduled_writes;
    scheduled_writes.reserve(storage_program.scheduled_extent_writes.size());
    for (const auto write_index : storage_program.scheduled_extent_writes) {
        if (write_index >= storage_program.extent_writes.size()) {
            throw std::runtime_error(
                "storage program: scheduled extent-write index out of range");
        }
        if (!scheduled_writes.insert(write_index).second) {
            throw std::runtime_error(
                "storage program: duplicate scheduled extent write");
        }
    }

    std::unordered_set<std::size_t> scheduled_steps;
    scheduled_steps.reserve(storage_program.schedule.size());
    std::unordered_set<std::size_t> scheduled_transforms;
    scheduled_transforms.reserve(storage_program.tile_maps.size());
    for (const auto& step : storage_program.schedule) {
        if (has_layout_ops) {
            if (step.op_index >= layout_plan.ops.size()) {
                throw std::runtime_error(
                    "storage program: schedule op index out of range");
            }
        } else if (step.op_index != kInvalidStorageId) {
            throw std::runtime_error(
                "storage program: algebra-only schedule has layout op index");
        }
        validate_algebra_ref(step.expr_id, step.binding_index, "schedule");
        for (const auto dep : step.dependencies) {
            if (!scheduled_steps.contains(dep)) {
                throw std::runtime_error(
                    "storage program: schedule dependency does not "
                    "precede dependent step");
            }
        }
        if (!scheduled_steps.insert(step.step_index).second) {
            throw std::runtime_error(
                "storage program: duplicate schedule step");
        }
        if (step.kind == StorageInstrKind::Transform &&
            step.transform_kind == StorageTransformKind::None) {
            throw std::runtime_error(
                "storage program: Transform instruction is missing typed payload");
        }
        if (step.kind != StorageInstrKind::Transform &&
            step.transform_kind != StorageTransformKind::None) {
            throw std::runtime_error(
                "storage program: non-Transform instruction carries transform payload");
        }
        if (step.kind == StorageInstrKind::ExtentWrite) {
            if (step.extent_write_indices.empty()) {
                throw std::runtime_error(
                    "storage program: extent-write step has no writes");
            }
        } else if (!step.extent_write_indices.empty() &&
                   step.kind != StorageInstrKind::TileMap) {
            throw std::runtime_error(
                "storage program: only ExtentWrite and TileMap instructions "
                "may own extent writes");
        }
        if (step.kind == StorageInstrKind::TileMap) {
            if (step.tile_map_index >=
                storage_program.tile_maps.size()) {
                throw std::runtime_error(
                    "storage program: tile map index out of range");
            }
            scheduled_transforms.insert(step.tile_map_index);
        }
    }
    if (scheduled_transforms.size() != storage_program.tile_maps.size()) {
        throw std::runtime_error(
            "storage program: tile map schedule coverage mismatch");
    }

    if (!layout_plan.algebra.bindings.empty()) {
        std::unordered_set<std::string> scheduled_outputs;
        for (const auto& step : storage_program.schedule) {
            for (const auto& output : step.outputs) {
                scheduled_outputs.insert(output);
            }
        }
        for (const auto& binding : layout_plan.algebra.bindings) {
            if (binding.root >= layout_plan.algebra.exprs.size()) {
                throw std::runtime_error(
                    "storage program: algebra binding root out of range for '" +
                    binding.runtime_name + "'");
            }
            const auto& root = layout_plan.algebra.exprs[binding.root];
            if (root.kind != LayoutExprKind::Realize &&
                root.kind != LayoutExprKind::View) {
                throw std::runtime_error(
                    "storage program: algebra binding for '" +
                    binding.runtime_name + "' is not a realized/view root");
            }
            if (!scheduled_outputs.contains(binding.runtime_name)) {
                const auto spec_it = layout_plan.tensors.find(binding.runtime_name);
                const bool temporary =
                    spec_it != layout_plan.tensors.end() &&
                    spec_it->second.ownership == TensorOwnershipKind::Temporary;
                if (!temporary) {
                    throw std::runtime_error(
                        "storage program: algebra binding '" +
                        binding.runtime_name +
                        "' is not covered by the storage schedule");
                }
            }
        }
    }
}

std::string describe_storage_program(const StorageProgram& plan) {
    std::ostringstream out;
    out << plan.memory.extent_write_count << " extent writes"
        << ", ranges=" << plan.memory.extent_range_count
        << ", algebra_writes=" << plan.memory.algebra_extent_write_count
        << ", planner_writes=" << plan.memory.planner_extent_write_count
        << ", checkpoint_read="
        << (plan.memory.checkpoint_read_bytes / (1024 * 1024)) << " MiB"
        << ", storage_temp<="
        << (plan.memory.max_temporary_bytes / (1024 * 1024)) << " MiB"
        << ", storage_peak~="
        << (plan.memory.estimated_peak_bytes / (1024 * 1024)) << " MiB";
    if (plan.memory.coalesced_extent_write_count > 0) {
        out << ", coalesced_writes=" << plan.memory.coalesced_extent_write_count;
    }
    if (plan.memory.tile_map_count > 0) {
        out << ", tile_maps=" << plan.memory.tile_map_count
            << ", transform_scratch<="
            << (plan.memory.max_transform_scratch_bytes / (1024 * 1024)) << " MiB";
    }
    if (plan.memory.scheduled_step_count > 0) {
        out << ", scheduled_steps=" << plan.memory.scheduled_step_count
            << ", file_ordered_writes="
            << plan.memory.file_ordered_extent_write_count;
    }
    return out.str();
}

std::string dump_storage_program_json(const StorageProgram& plan) {
    std::ostringstream out;
    out << "{";
    out << "\"program_kind\":\"StorageProgram\","
        << "\"instr_kind\":\"StorageInstr\","
        << "\"write_kind\":\"ExtentWrite\","
        << "\"tile_kind\":\"TileMap\",";
    out << "\"summary\":";
    json_string(out, describe_storage_program(plan));
    out << ",\"memory\":{"
        << "\"persistent_bytes\":" << plan.memory.persistent_bytes << ','
        << "\"layout_max_temporary_bytes\":"
        << plan.memory.layout_max_temporary_bytes << ','
        << "\"max_extent_temporary_bytes\":"
        << plan.memory.max_extent_temporary_bytes << ','
        << "\"max_transform_scratch_bytes\":"
        << plan.memory.max_transform_scratch_bytes << ','
        << "\"max_temporary_bytes\":" << plan.memory.max_temporary_bytes << ','
        << "\"estimated_peak_bytes\":" << plan.memory.estimated_peak_bytes << ','
        << "\"checkpoint_read_bytes\":" << plan.memory.checkpoint_read_bytes << ','
        << "\"device_write_bytes\":" << plan.memory.device_write_bytes << ','
        << "\"extent_write_count\":" << plan.memory.extent_write_count << ','
        << "\"algebra_extent_write_count\":"
        << plan.memory.algebra_extent_write_count << ','
        << "\"planner_extent_write_count\":"
        << plan.memory.planner_extent_write_count << ','
        << "\"extent_range_count\":" << plan.memory.extent_range_count << ','
        << "\"optimized_extent_write_count\":"
        << plan.memory.optimized_extent_write_count << ','
        << "\"coalesced_extent_write_count\":"
        << plan.memory.coalesced_extent_write_count << ','
        << "\"tile_map_count\":"
        << plan.memory.tile_map_count << ','
        << "\"scheduled_step_count\":"
        << plan.memory.scheduled_step_count << ','
        << "\"file_ordered_extent_write_count\":"
        << plan.memory.file_ordered_extent_write_count
        << "},\"extent_writes\":[";
    for (std::size_t i = 0; i < plan.extent_writes.size(); ++i) {
        const auto& write = plan.extent_writes[i];
        if (i) out << ',';
        out << "{\"op_index\":";
        json_id_value(out, write.op_index);
        out << ",\"op_kind\":";
        json_string(out, write.op_kind);
        json_id_field(out, "expr_id", write.expr_id);
        json_id_field(out, "binding_index", write.binding_index);
        out << ",\"raw_name\":";
        json_string(out, write.raw_name);
        out << ",\"output_name\":";
        json_string(out, write.output_name);
        out << ",\"dst_shape\":";
        json_shape(out, write.dst_shape);
        out << ",\"dst_offset_bytes\":" << write.dst_offset_bytes
            << ",\"source_path\":";
        json_string(out, write.source_path);
        out << ",\"source_shard_id\":" << write.source_shard_id
            << ",\"source_offset_bytes\":" << write.source_offset_bytes
            << ",\"source_span_bytes\":" << write.source_span_bytes
            << ",\"bytes\":" << write.bytes
            << ",\"range_count\":" << write.range_count
            << ",\"contiguous\":" << (write.contiguous ? "true" : "false")
            << ",\"slices\":";
        json_slices(out, write.slices);
        out << '}';
    }
    out << "],\"tile_maps\":[";
    for (std::size_t i = 0; i < plan.tile_maps.size(); ++i) {
        const auto& tx = plan.tile_maps[i];
        if (i) out << ',';
        out << "{\"op_index\":";
        json_id_value(out, tx.op_index);
        out << ",\"kind\":";
        json_string(out, tile_map_kind_name(tx.kind));
        json_id_field(out, "expr_id", tx.expr_id);
        json_id_field(out, "binding_index", tx.binding_index);
        out << ",\"output_name\":";
        json_string(out, tx.output_name);
        out << ",\"inputs\":[";
        for (std::size_t j = 0; j < tx.inputs.size(); ++j) {
            if (j) out << ',';
            json_string(out, tx.inputs[j]);
        }
        out << "],\"input_bytes\":" << tx.input_bytes
            << ",\"output_bytes\":" << tx.output_bytes
            << ",\"tile_bytes\":" << tx.tile_bytes
            << ",\"scratch_bytes\":" << tx.scratch_bytes
            << '}';
    }
    out << "],\"schedule\":[";
    for (std::size_t i = 0; i < plan.schedule.size(); ++i) {
        const auto& step = plan.schedule[i];
        if (i) out << ',';
        out << "{\"step_index\":" << step.step_index
            << ",\"kind\":";
        json_string(out, storage_instr_kind_name(step.kind));
        out << ",\"transform_kind\":";
        json_string(out, storage_transform_kind_name(step.transform_kind));
        out << ",\"op_index\":";
        json_id_value(out, step.op_index);
        json_id_field(out, "expr_id", step.expr_id);
        json_id_field(out, "binding_index", step.binding_index);
        out << ",\"extent_write_indices\":[";
        for (std::size_t j = 0; j < step.extent_write_indices.size(); ++j) {
            if (j) out << ',';
            out << step.extent_write_indices[j];
        }
        out << "],\"tile_map_index\":";
        json_id_value(out, step.tile_map_index);
        out << ",\"inputs\":[";
        for (std::size_t j = 0; j < step.inputs.size(); ++j) {
            if (j) out << ',';
            json_string(out, step.inputs[j]);
        }
        out << "],\"outputs\":[";
        for (std::size_t j = 0; j < step.outputs.size(); ++j) {
            if (j) out << ',';
            json_string(out, step.outputs[j]);
        }
        out << "],\"dependencies\":[";
        for (std::size_t j = 0; j < step.dependencies.size(); ++j) {
            if (j) out << ',';
            out << step.dependencies[j];
        }
        out << "],\"payload\":";
        json_payload(out, step.payload);
        out << "}";
    }
    out << "],\"scheduled_extent_writes\":[";
    for (std::size_t i = 0; i < plan.scheduled_extent_writes.size(); ++i) {
        if (i) out << ',';
        out << plan.scheduled_extent_writes[i];
    }
    out << "],\"coverage\":[";
    for (std::size_t i = 0; i < plan.coverage.size(); ++i) {
        const auto& coverage = plan.coverage[i];
        if (i) out << ',';
        out << "{\"op_index\":" << coverage.op_index
            << ",\"op_kind\":";
        json_string(out, coverage.op_kind);
        json_id_field(out, "expr_id", coverage.expr_id);
        json_id_field(out, "binding_index", coverage.binding_index);
        out << ",\"action\":";
        json_string(out, storage_action_kind_name(coverage.action));
        out << ",\"extent_writes\":" << coverage.extent_writes
            << ",\"tile_maps\":" << coverage.tile_maps
            << '}';
    }
    out << "]}";
    return out.str();
}

std::string dump_layout_plan_json(
    const LayoutPlan& plan,
    const StorageProgram& storage_program)
{
    std::string layout_json = dump_layout_plan_json(plan);
    const auto insert = layout_json.rfind("\n}");
    if (insert == std::string::npos) return layout_json;
    std::ostringstream out;
    out << layout_json.substr(0, insert)
        << ",\n  \"storage\": "
        << dump_storage_program_json(storage_program)
        << "\n}\n";
    return out.str();
}

}  // namespace pie_cuda_driver

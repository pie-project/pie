#include "loader/physical_load_plan.hpp"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <pie_driver_common/shard_plan.hpp>

#include "tensor.hpp"

namespace pie_cuda_driver {

namespace {

std::uint64_t tensor_nbytes(DType dtype, const std::vector<std::int64_t>& shape) {
    std::uint64_t n = 1;
    for (const auto dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("physical load plan: negative tensor dimension");
        }
        n *= static_cast<std::uint64_t>(dim);
    }
    return n * static_cast<std::uint64_t>(dtype_bytes(dtype));
}

std::uint64_t tensor_nbytes(const TensorSpec& spec) {
    return tensor_nbytes(spec.dtype, spec.shape);
}

const TensorSpec& tensor_spec_for(
    const LoadPlan& plan,
    const std::string& name)
{
    const auto it = plan.tensors.find(name);
    if (it == plan.tensors.end()) {
        throw std::runtime_error(
            "physical load plan: missing TensorSpec for '" + name + "'");
    }
    return it->second;
}

const TensorSpec* find_tensor_spec(
    const LoadPlan& plan,
    const std::string& name)
{
    const auto it = plan.tensors.find(name);
    return it == plan.tensors.end() ? nullptr : &it->second;
}

bool is_raw_copy_kind(LoadOpKind kind) noexcept {
    return kind == LoadOpKind::Read ||
           kind == LoadOpKind::Copy ||
           kind == LoadOpKind::Shard ||
           kind == LoadOpKind::RowRangeShard;
}

bool op_allocates_tensor(LoadOpKind kind) noexcept {
    return kind != LoadOpKind::Drop &&
           kind != LoadOpKind::View &&
           kind != LoadOpKind::Alias &&
           kind != LoadOpKind::BindMetadata;
}

bool is_raw_copy_cast_drop_sequence(
    const LoadPlan& plan,
    std::size_t producer_index)
{
    if (producer_index + 2 >= plan.ops.size()) return false;
    const LoadOp& producer = plan.ops[producer_index];
    const LoadOp& cast = plan.ops[producer_index + 1];
    const LoadOp& drop = plan.ops[producer_index + 2];
    if (!is_raw_copy_kind(producer.kind) ||
        cast.kind != LoadOpKind::Cast ||
        drop.kind != LoadOpKind::Drop) {
        return false;
    }
    if (load_op_inputs(cast).size() != 1 ||
        load_op_inputs(cast)[0] != load_op_output(producer)) {
        return false;
    }
    return std::find(
        load_op_inputs(drop).begin(),
        load_op_inputs(drop).end(),
        load_op_output(producer)) != load_op_inputs(drop).end();
}

std::uint64_t leading_tile_scratch_bytes(
    const TensorSpec& spec,
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
    const LoadPlan& plan,
    std::size_t op_index,
    std::uint64_t tile_bytes)
{
    const LoadOp& op = plan.ops[op_index];
    if (op.kind == LoadOpKind::Cast) {
        if (op_index > 0 &&
            is_raw_copy_cast_drop_sequence(plan, op_index - 1)) {
            return leading_tile_scratch_bytes(
                tensor_spec_for(plan, load_op_output(plan.ops[op_index - 1])),
                tile_bytes);
        }
        return 0;
    }
    if (op.kind != LoadOpKind::Dequantize || load_op_inputs(op).size() < 2) {
        return 0;
    }

    const TensorSpec* source = find_tensor_spec(plan, load_op_inputs(op)[0]);
    const TensorSpec* scale = find_tensor_spec(plan, load_op_inputs(op)[1]);
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

struct PhysicalTempUsage {
    std::uint64_t max_resident_temp_bytes = 0;
    std::uint64_t max_transform_scratch_bytes = 0;
    std::uint64_t max_total_temporary_bytes = 0;
};

PhysicalTempUsage compute_physical_temp_usage(
    const LoadPlan& plan,
    std::uint64_t tile_bytes)
{
    std::unordered_map<std::string, std::uint64_t> temp_bytes;
    temp_bytes.reserve(plan.tensors.size());
    for (const auto& [name, spec] : plan.tensors) {
        if (spec.ownership == TensorOwnershipKind::Temporary) {
            temp_bytes.emplace(name, tensor_nbytes(spec));
        }
    }

    PhysicalTempUsage usage;
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
            const LoadOp& cast = plan.ops[i + 1];
            add_temp(load_op_output(cast));
            add_temp(load_op_secondary_output(cast));
            account_scratch(transform_scratch_bytes_for_op(
                plan, i + 1, tile_bytes));
            i += 2;
            continue;
        }

        const LoadOp& op = plan.ops[i];
        if (op_allocates_tensor(op.kind)) {
            add_temp(load_op_output(op));
            add_temp(load_op_secondary_output(op));
        }
        account_scratch(transform_scratch_bytes_for_op(plan, i, tile_bytes));

        if (op.kind == LoadOpKind::Drop) {
            if (load_op_inputs(op).empty()) {
                release_temp(load_op_output(op));
            } else {
                for (const auto& input : load_op_inputs(op)) {
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

ByteRangeWrite make_write(
    const LoadOp& op,
    std::size_t op_index,
    const TensorMetadataSource& metadata,
    std::string output_name,
    std::string raw_name,
    std::vector<TensorSlice> slices,
    std::vector<std::int64_t> dst_shape,
    std::uint64_t dst_offset_bytes)
{
    const auto& info = metadata.info(raw_name);
    const std::uint64_t bytes = tensor_nbytes(info.dtype, dst_shape);
    const std::uint64_t ranges = range_count_for(info, slices, dst_shape);
    return ByteRangeWrite{
        .op_index = op_index,
        .op_kind = load_op_kind_name(op.kind),
        .raw_name = std::move(raw_name),
        .output_name = std::move(output_name),
        .slices = std::move(slices),
        .dst_shape = std::move(dst_shape),
        .dst_offset_bytes = dst_offset_bytes,
        .bytes = bytes,
        .range_count = ranges,
        .contiguous = ranges == 1,
    };
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
        "physical load plan shard: " + raw_name);
    shape = shard.output_shape;
    return {TensorSlice{shard_axis, shard.offset, shard.shard_dim}};
}

std::vector<ByteRangeWrite> lower_raw_like_op(
    const LoadOp& op,
    std::size_t op_index,
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out = tensor_spec_for(plan, load_op_output(op));
    const auto& info = metadata.info(load_op_raw_name(op));
    if (info.dtype != out.dtype) {
        throw std::runtime_error(
            "physical load plan: raw byte write dtype mismatch for '" +
            out.name + "'");
    }
    std::vector<std::int64_t> shape;
    auto slices = shard_slices_for(
        info, load_op_shard_axis(op), tp_rank, tp_size, shape,
        load_op_raw_name(op));
    if (shape != out.shape) {
        throw std::runtime_error(
            "physical load plan: raw byte write shape mismatch for '" +
            out.name + "'");
    }
    return {make_write(
        op, op_index, metadata, out.name, load_op_raw_name(op),
        std::move(slices), out.shape, 0)};
}

std::vector<ByteRangeWrite> lower_row_range_shard_op(
    const LoadOp& op,
    std::size_t op_index,
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out = tensor_spec_for(plan, load_op_output(op));
    const auto& info = metadata.info(load_op_raw_name(op));
    std::int64_t rows = load_op_rows(op);
    std::int64_t offset = load_op_row_offset(op);
    if (rows <= 0) {
        throw std::runtime_error(
            "physical load plan: row-range shard has non-positive rows for '" +
            load_op_raw_name(op) + "'");
    }
    if (tp_size > 1) {
        if (rows % tp_size != 0) {
            throw std::runtime_error(
                "physical load plan: row range is not divisible by tp_size for '" +
                load_op_raw_name(op) + "'");
        }
        rows /= tp_size;
        offset += static_cast<std::int64_t>(tp_rank) * rows;
    }
    std::vector<std::int64_t> shape = info.shape;
    shape[0] = rows;
    if (info.dtype != out.dtype || shape != out.shape) {
        throw std::runtime_error(
            "physical load plan: row-range output mismatch for '" +
            out.name + "'");
    }
    return {make_write(
        op, op_index, metadata, out.name, load_op_raw_name(op),
        {TensorSlice{0, offset, rows}}, out.shape, 0)};
}

std::vector<ByteRangeWrite> lower_axis_concat_op(
    const LoadOp& op,
    std::size_t op_index,
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out = tensor_spec_for(plan, load_op_output(op));
    if (out.shape.size() != 2) {
        throw std::runtime_error(
            "physical load plan: AxisConcat output must be 2-D for '" +
            out.name + "'");
    }

    std::vector<ByteRangeWrite> writes;
    std::uint64_t dst_offset = 0;
    std::int64_t rows = 0;
    for (const auto& src : load_op_sources(op)) {
        const auto& info = metadata.info(src.raw_name);
        if (info.dtype != out.dtype || info.shape.size() != 2) {
            throw std::runtime_error(
                "physical load plan: AxisConcat source mismatch for '" +
                out.name + "'");
        }
        std::vector<std::int64_t> shape;
        auto slices = shard_slices_for(
            info, load_op_shard_axis(op), tp_rank, tp_size, shape,
            src.raw_name);
        if (shape.size() != 2 || shape[1] != out.shape[1]) {
            throw std::runtime_error(
                "physical load plan: AxisConcat source shape mismatch for '" +
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
            "physical load plan: AxisConcat row count mismatch for '" +
            out.name + "'");
    }
    return writes;
}

std::vector<ByteRangeWrite> lower_grouped_slice_concat_op(
    const LoadOp& op,
    std::size_t op_index,
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out = tensor_spec_for(plan, load_op_output(op));
    const auto& info = metadata.info(load_op_raw_name(op));
    if (info.dtype != out.dtype || info.shape.size() != 3 ||
        out.shape.size() != 3) {
        throw std::runtime_error(
            "physical load plan: GroupedSliceConcat expects rank-3 tensors for '" +
            out.name + "'");
    }
    const std::int64_t E = info.shape[0];
    const std::int64_t two_I = info.shape[1];
    const std::int64_t H = info.shape[2];
    if (two_I % 2 != 0 || (two_I / 2) % tp_size != 0) {
        throw std::runtime_error(
            "physical load plan: GroupedSliceConcat intermediate axis mismatch for '" +
            out.name + "'");
    }
    const std::int64_t I = two_I / 2;
    const std::int64_t I_local = I / tp_size;
    if (out.shape != std::vector<std::int64_t>{E, 2 * I_local, H}) {
        throw std::runtime_error(
            "physical load plan: GroupedSliceConcat output shape mismatch for '" +
            out.name + "'");
    }

    std::vector<ByteRangeWrite> writes;
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
            op, op_index, metadata, out.name, load_op_raw_name(op),
            {TensorSlice{0, e, 1}, TensorSlice{1, gate_start, I_local}},
            {1, I_local, H}, expert_offset));
        writes.push_back(make_write(
            op, op_index, metadata, out.name, load_op_raw_name(op),
            {TensorSlice{0, e, 1}, TensorSlice{1, up_start, I_local}},
            {1, I_local, H}, expert_offset + half_bytes));
    }
    return writes;
}

std::vector<ByteRangeWrite> lower_grouped_slice_op(
    const LoadOp& op,
    std::size_t op_index,
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out = tensor_spec_for(plan, load_op_output(op));
    const auto& info = metadata.info(load_op_raw_name(op));
    if (info.dtype != out.dtype || info.shape.size() != 3 ||
        out.shape.size() != 3) {
        throw std::runtime_error(
            "physical load plan: GroupedSlice expects rank-3 tensors for '" +
            out.name + "'");
    }
    const std::int64_t I = info.shape[2];
    if (I % tp_size != 0) {
        throw std::runtime_error(
            "physical load plan: GroupedSlice axis mismatch for '" +
            out.name + "'");
    }
    const std::int64_t I_local = I / tp_size;
    if (out.shape != std::vector<std::int64_t>{info.shape[0], info.shape[1], I_local}) {
        throw std::runtime_error(
            "physical load plan: GroupedSlice output shape mismatch for '" +
            out.name + "'");
    }
    return {make_write(
        op, op_index, metadata, out.name, load_op_raw_name(op),
        {TensorSlice{2, static_cast<std::int64_t>(tp_rank) * I_local, I_local}},
        out.shape, 0)};
}

std::vector<ByteRangeWrite> lower_stack_groups_op(
    const LoadOp& op,
    std::size_t op_index,
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size)
{
    if (load_op_sources(op).empty()) return {};
    const TensorSpec& gate_up = tensor_spec_for(plan, load_op_output(op));
    const TensorSpec& down = tensor_spec_for(plan, load_op_secondary_output(op));
    if (load_op_sources(op).size() % 3 != 0 ||
        gate_up.shape.size() != 3 || down.shape.size() != 3) {
        throw std::runtime_error(
            "physical load plan: StackGroups expects expert triples for '" +
            gate_up.name + "'");
    }
    const std::int64_t E =
        static_cast<std::int64_t>(load_op_sources(op).size() / 3);
    const std::int64_t I = gate_up.shape[1] / 2;
    const std::int64_t H = gate_up.shape[2];
    const std::int64_t I_down = down.shape[2];
    if (gate_up.shape != std::vector<std::int64_t>{E, 2 * I, H} ||
        down.shape != std::vector<std::int64_t>{E, H, I_down}) {
        throw std::runtime_error(
            "physical load plan: StackGroups output shape mismatch for '" +
            gate_up.name + "'");
    }

    std::vector<ByteRangeWrite> writes;
    const std::uint64_t proj_bytes = tensor_nbytes(gate_up.dtype, {I, H});
    const std::uint64_t gate_up_expert_bytes = 2 * proj_bytes;
    const std::uint64_t down_expert_bytes = tensor_nbytes(down.dtype, {H, I_down});
    for (std::int64_t e = 0; e < E; ++e) {
        const auto& gate_src = load_op_sources(op)[static_cast<std::size_t>(e) * 3];
        const auto& up_src = load_op_sources(op)[static_cast<std::size_t>(e) * 3 + 1];
        const auto& down_src = load_op_sources(op)[static_cast<std::size_t>(e) * 3 + 2];
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
                "physical load plan: StackGroups source metadata mismatch for '" +
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

}  // namespace

std::vector<ByteRangeWrite> lower_byte_writes_for_op(
    const LoadOp& op,
    std::size_t op_index,
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size)
{
    switch (op.kind) {
    case LoadOpKind::Read:
    case LoadOpKind::Copy:
    case LoadOpKind::Shard:
        return lower_raw_like_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LoadOpKind::RowRangeShard:
        return lower_row_range_shard_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LoadOpKind::AxisConcat:
        return lower_axis_concat_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LoadOpKind::GroupedSliceConcat:
        return lower_grouped_slice_concat_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LoadOpKind::GroupedSlice:
        return lower_grouped_slice_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    case LoadOpKind::StackGroups:
        return lower_stack_groups_op(
            op, op_index, plan, metadata, tp_rank, tp_size);
    default:
        return {};
    }
}

PhysicalLoadPlan build_physical_load_plan(
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size,
    std::uint64_t transform_tile_bytes)
{
    validate_load_plan(plan);
    PhysicalLoadPlan physical;
    physical.memory.persistent_bytes = plan.memory.persistent_bytes;
    physical.memory.semantic_max_temporary_bytes = plan.memory.max_temporary_bytes;

    for (std::size_t i = 0; i < plan.ops.size(); ++i) {
        const LoadOp& op = plan.ops[i];
        auto writes = lower_byte_writes_for_op(
            op, i, plan, metadata, tp_rank, tp_size);
        for (const auto& write : writes) {
            physical.memory.checkpoint_read_bytes += write.bytes;
            physical.memory.device_write_bytes += write.bytes;
            physical.memory.byte_write_count += 1;
            physical.memory.byte_range_count += write.range_count;
        }
        physical.byte_writes.insert(
            physical.byte_writes.end(),
            std::make_move_iterator(writes.begin()),
            std::make_move_iterator(writes.end()));

        if (op.kind == LoadOpKind::Cast || op.kind == LoadOpKind::Dequantize) {
            const TensorSpec& out = tensor_spec_for(plan, load_op_output(op));
            std::uint64_t input_bytes = 0;
            for (const auto& input : load_op_inputs(op)) {
                const auto spec_it = plan.tensors.find(input);
                if (spec_it != plan.tensors.end()) {
                    input_bytes += tensor_nbytes(spec_it->second);
                }
            }
            const std::uint64_t output_bytes = tensor_nbytes(out);
            const std::uint64_t scratch = transform_scratch_bytes_for_op(
                plan, i, transform_tile_bytes);
            physical.tiled_transforms.push_back(TiledTransform{
                .op_index = i,
                .kind = op.kind == LoadOpKind::Cast
                    ? PhysicalTransformKind::Cast
                    : PhysicalTransformKind::Dequantize,
                .output_name = load_op_output(op),
                .inputs = load_op_inputs(op),
                .input_bytes = input_bytes,
                .output_bytes = output_bytes,
                .tile_bytes = transform_tile_bytes,
                .scratch_bytes = scratch,
            });
            physical.memory.max_transform_scratch_bytes =
                std::max(physical.memory.max_transform_scratch_bytes, scratch);
            physical.memory.tiled_transform_count += 1;
        }
    }

    const PhysicalTempUsage temp_usage =
        compute_physical_temp_usage(plan, transform_tile_bytes);
    physical.memory.max_copy_temporary_bytes =
        temp_usage.max_resident_temp_bytes;
    physical.memory.max_transform_scratch_bytes =
        temp_usage.max_transform_scratch_bytes;
    physical.memory.max_temporary_bytes = temp_usage.max_total_temporary_bytes;
    physical.memory.estimated_peak_bytes =
        physical.memory.persistent_bytes + physical.memory.max_temporary_bytes;
    return physical;
}

const char* physical_transform_kind_name(PhysicalTransformKind kind) noexcept {
    switch (kind) {
    case PhysicalTransformKind::Cast: return "Cast";
    case PhysicalTransformKind::Dequantize: return "Dequantize";
    }
    return "?";
}

std::string describe_physical_load_plan(const PhysicalLoadPlan& plan) {
    std::ostringstream out;
    out << plan.memory.byte_write_count << " byte writes"
        << ", ranges=" << plan.memory.byte_range_count
        << ", checkpoint_read="
        << (plan.memory.checkpoint_read_bytes / (1024 * 1024)) << " MiB"
        << ", physical_temp<="
        << (plan.memory.max_temporary_bytes / (1024 * 1024)) << " MiB"
        << ", physical_peak~="
        << (plan.memory.estimated_peak_bytes / (1024 * 1024)) << " MiB";
    if (plan.memory.tiled_transform_count > 0) {
        out << ", tiled_transforms=" << plan.memory.tiled_transform_count
            << ", transform_scratch<="
            << (plan.memory.max_transform_scratch_bytes / (1024 * 1024)) << " MiB";
    }
    return out.str();
}

std::string dump_physical_load_plan_json(const PhysicalLoadPlan& plan) {
    std::ostringstream out;
    out << "{";
    out << "\"summary\":";
    json_string(out, describe_physical_load_plan(plan));
    out << ",\"memory\":{"
        << "\"persistent_bytes\":" << plan.memory.persistent_bytes << ','
        << "\"semantic_max_temporary_bytes\":"
        << plan.memory.semantic_max_temporary_bytes << ','
        << "\"max_copy_temporary_bytes\":"
        << plan.memory.max_copy_temporary_bytes << ','
        << "\"max_transform_scratch_bytes\":"
        << plan.memory.max_transform_scratch_bytes << ','
        << "\"max_temporary_bytes\":" << plan.memory.max_temporary_bytes << ','
        << "\"estimated_peak_bytes\":" << plan.memory.estimated_peak_bytes << ','
        << "\"checkpoint_read_bytes\":" << plan.memory.checkpoint_read_bytes << ','
        << "\"device_write_bytes\":" << plan.memory.device_write_bytes << ','
        << "\"byte_write_count\":" << plan.memory.byte_write_count << ','
        << "\"byte_range_count\":" << plan.memory.byte_range_count << ','
        << "\"tiled_transform_count\":"
        << plan.memory.tiled_transform_count
        << "},\"byte_writes\":[";
    for (std::size_t i = 0; i < plan.byte_writes.size(); ++i) {
        const auto& write = plan.byte_writes[i];
        if (i) out << ',';
        out << "{\"op_index\":" << write.op_index
            << ",\"op_kind\":";
        json_string(out, write.op_kind);
        out << ",\"raw_name\":";
        json_string(out, write.raw_name);
        out << ",\"output_name\":";
        json_string(out, write.output_name);
        out << ",\"dst_shape\":";
        json_shape(out, write.dst_shape);
        out << ",\"dst_offset_bytes\":" << write.dst_offset_bytes
            << ",\"bytes\":" << write.bytes
            << ",\"range_count\":" << write.range_count
            << ",\"contiguous\":" << (write.contiguous ? "true" : "false")
            << ",\"slices\":";
        json_slices(out, write.slices);
        out << '}';
    }
    out << "],\"tiled_transforms\":[";
    for (std::size_t i = 0; i < plan.tiled_transforms.size(); ++i) {
        const auto& tx = plan.tiled_transforms[i];
        if (i) out << ',';
        out << "{\"op_index\":" << tx.op_index
            << ",\"kind\":";
        json_string(out, physical_transform_kind_name(tx.kind));
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
    out << "]}";
    return out.str();
}

std::string dump_load_plan_json(
    const LoadPlan& plan,
    const PhysicalLoadPlan& physical_plan)
{
    std::string semantic = dump_load_plan_json(plan);
    const auto insert = semantic.rfind("\n}");
    if (insert == std::string::npos) return semantic;
    std::ostringstream out;
    out << semantic.substr(0, insert)
        << ",\n  \"physical\": "
        << dump_physical_load_plan_json(physical_plan)
        << "\n}\n";
    return out.str();
}

}  // namespace pie_cuda_driver

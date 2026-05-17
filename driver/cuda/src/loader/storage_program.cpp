#include "loader/storage_program.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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

struct SourceSlicePlan {
    std::string raw_name;
    DType dtype = DType::BF16;
    std::vector<std::int64_t> shape;
    std::vector<TensorSlice> slices;
};

void apply_source_slice(
    SourceSlicePlan& plan,
    int axis,
    std::int64_t start,
    std::int64_t length)
{
    if (axis < 0 || axis >= static_cast<int>(plan.shape.size()) ||
        start < 0 || length <= 0 ||
        start + length > plan.shape[static_cast<std::size_t>(axis)]) {
        throw std::runtime_error(
            "storage program: invalid algebra source slice for '" +
            plan.raw_name + "'");
    }
    for (auto& slice : plan.slices) {
        if (slice.axis == axis) {
            slice.start += start;
            slice.length = length;
            plan.shape[static_cast<std::size_t>(axis)] = length;
            return;
        }
    }
    plan.slices.push_back(TensorSlice{axis, start, length});
    plan.shape[static_cast<std::size_t>(axis)] = length;
}

bool lower_expr_to_source_slice(
    LayoutExprId id,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size,
    SourceSlicePlan& out)
{
    if (id >= plan.algebra.exprs.size()) return false;
    const LayoutExpr& expr = plan.algebra.exprs[id];
    switch (expr.kind) {
    case LayoutExprKind::Source: {
        const auto& info = metadata.info(expr.raw_name);
        out.raw_name = expr.raw_name;
        out.dtype = info.dtype;
        out.shape = info.shape;
        out.slices.clear();
        return true;
    }
    case LayoutExprKind::Realize:
        return expr.inputs.size() == 1 &&
               lower_expr_to_source_slice(
                   expr.inputs.front(), plan, metadata, tp_rank, tp_size, out);
    case LayoutExprKind::Select: {
        if (expr.inputs.size() != 1 ||
            !lower_expr_to_source_slice(
                expr.inputs.front(), plan, metadata, tp_rank, tp_size, out)) {
            return false;
        }
        std::int64_t start = expr.start;
        std::int64_t length = expr.length;
        if (tp_size > 1 &&
            expr.axis >= 0 &&
            expr.axis < static_cast<int>(expr.decl.shape.size()) &&
            length % tp_size == 0 &&
            expr.decl.shape[static_cast<std::size_t>(expr.axis)] ==
                length / tp_size) {
            length /= tp_size;
            start += static_cast<std::int64_t>(tp_rank) * length;
        }
        apply_source_slice(out, expr.axis, start, length);
        return true;
    }
    case LayoutExprKind::Partition: {
        if (expr.inputs.size() != 1 ||
            !lower_expr_to_source_slice(
                expr.inputs.front(), plan, metadata, tp_rank, tp_size, out)) {
            return false;
        }
        if (tp_size <= 1 || expr.axis < 0) return true;
        if (expr.axis >= static_cast<int>(out.shape.size())) return false;
        const auto axis = static_cast<std::size_t>(expr.axis);
        if (out.shape[axis] % tp_size != 0) return false;
        const std::int64_t length = out.shape[axis] / tp_size;
        const std::int64_t start =
            static_cast<std::int64_t>(tp_rank) * length;
        apply_source_slice(out, expr.axis, start, length);
        return true;
    }
    default:
        return false;
    }
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

bool lower_source_expr_to_write(
    LayoutExprId value_id,
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
    SourceSlicePlan source;
    (void)expr;
    if (!lower_expr_to_source_slice(
            value_id, plan, metadata, tp_rank, tp_size, source)) {
        return false;
    }
    if (source.dtype != tensor_spec_for(plan, output_name).dtype ||
        source.shape != output_shape) {
        return false;
    }
    writes.push_back(make_write_record(
        op_index, expr_id, binding_index, op_kind, metadata,
        output_name, source.raw_name, std::move(source.slices), output_shape,
        dst_offset_bytes));
    return true;
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
    if (join.axis < 0 ||
        join.axis >= static_cast<int>(out.shape.size()) ||
        out.shape.empty()) {
        return false;
    }

    std::vector<std::int64_t> total_shape = out.shape;
    total_shape[static_cast<std::size_t>(join.axis)] = 0;
    std::int64_t axis_cursor = 0;
    const std::size_t axis = static_cast<std::size_t>(join.axis);
    for (const auto input_id : join.inputs) {
        if (input_id >= plan.algebra.exprs.size()) return false;
        SourceSlicePlan source;
        if (!lower_expr_to_source_slice(
                input_id, plan, metadata, tp_rank, tp_size, source)) {
            return false;
        }
        if (source.dtype != out.dtype ||
            source.shape.size() != out.shape.size()) {
            return false;
        }
        for (std::size_t dim = 0; dim < source.shape.size(); ++dim) {
            if (dim == axis) continue;
            if (source.shape[dim] != out.shape[dim]) return false;
        }

        if (axis == 0) {
            const std::uint64_t dst_offset =
                tensor_nbytes(out.dtype, total_shape);
            writes.push_back(make_write_record(
                op_index, expr_id, binding_index, op_kind, metadata,
                output_name, source.raw_name, std::move(source.slices),
                source.shape, dst_offset));
        } else {
            std::uint64_t prefix_count = 1;
            for (std::size_t dim = 0; dim < axis; ++dim) {
                prefix_count *= static_cast<std::uint64_t>(out.shape[dim]);
            }
            std::uint64_t suffix_count = 1;
            for (std::size_t dim = axis + 1; dim < out.shape.size(); ++dim) {
                suffix_count *= static_cast<std::uint64_t>(out.shape[dim]);
            }
            std::vector<std::int64_t> dst_shape = source.shape;
            for (std::size_t dim = 0; dim < axis; ++dim) {
                dst_shape[dim] = 1;
            }
            for (std::uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
                std::uint64_t rem = prefix;
                auto slices = source.slices;
                for (std::size_t dim = axis; dim-- > 0;) {
                    const auto coord = static_cast<std::int64_t>(
                        rem % static_cast<std::uint64_t>(out.shape[dim]));
                    rem /= static_cast<std::uint64_t>(out.shape[dim]);
                    slices.push_back(TensorSlice{
                        static_cast<int>(dim), coord, 1});
                }
                const std::uint64_t element_offset =
                    (prefix * static_cast<std::uint64_t>(out.shape[axis]) *
                         suffix_count +
                     static_cast<std::uint64_t>(axis_cursor) * suffix_count);
                writes.push_back(make_write_record(
                    op_index, expr_id, binding_index, op_kind, metadata,
                    output_name, source.raw_name, std::move(slices),
                    dst_shape,
                    element_offset *
                        static_cast<std::uint64_t>(dtype_bytes(out.dtype))));
            }
        }

        total_shape[axis] += source.shape[axis];
        axis_cursor += source.shape[axis];
    }
    return total_shape == out.shape;
}

bool try_lower_stack_expr_to_writes(
    const LayoutExpr& stack,
    std::size_t op_index,
    LayoutExprId expr_id,
    std::size_t binding_index,
    const std::string& op_kind,
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size,
    std::vector<ExtentWrite>& writes)
{
    if (stack.kind != LayoutExprKind::Stack ||
        stack.secondary_runtime_name.empty() ||
        stack.inputs.empty() ||
        stack.inputs.size() % 3 != 0) {
        return false;
    }
    const TensorDecl& gate_up = tensor_spec_for(plan, stack.runtime_name);
    const TensorDecl& down = tensor_spec_for(plan, stack.secondary_runtime_name);
    if (gate_up.shape.size() != 3 || down.shape.size() != 3) return false;

    const std::int64_t E = static_cast<std::int64_t>(stack.inputs.size() / 3);
    const std::int64_t I = gate_up.shape[1] / 2;
    const std::int64_t H = gate_up.shape[2];
    const std::int64_t I_down = down.shape[2];
    if (gate_up.shape != std::vector<std::int64_t>{E, 2 * I, H} ||
        down.shape != std::vector<std::int64_t>{E, H, I_down}) {
        return false;
    }

    const std::uint64_t proj_bytes = tensor_nbytes(gate_up.dtype, {I, H});
    const std::uint64_t gate_up_expert_bytes = 2 * proj_bytes;
    const std::uint64_t down_expert_bytes =
        tensor_nbytes(down.dtype, {H, I_down});
    for (std::int64_t e = 0; e < E; ++e) {
        SourceSlicePlan gate;
        SourceSlicePlan up;
        SourceSlicePlan down_source;
        if (!lower_expr_to_source_slice(
                stack.inputs[static_cast<std::size_t>(e) * 3],
                plan, metadata, tp_rank, tp_size, gate) ||
            !lower_expr_to_source_slice(
                stack.inputs[static_cast<std::size_t>(e) * 3 + 1],
                plan, metadata, tp_rank, tp_size, up) ||
            !lower_expr_to_source_slice(
                stack.inputs[static_cast<std::size_t>(e) * 3 + 2],
                plan, metadata, tp_rank, tp_size, down_source)) {
            return false;
        }
        if (gate.dtype != gate_up.dtype ||
            up.dtype != gate_up.dtype ||
            down_source.dtype != down.dtype ||
            gate.shape != std::vector<std::int64_t>{I, H} ||
            up.shape != std::vector<std::int64_t>{I, H} ||
            down_source.shape != std::vector<std::int64_t>{H, I_down}) {
            return false;
        }
        (void)tp_rank;
        (void)tp_size;

        const std::uint64_t gate_up_offset =
            static_cast<std::uint64_t>(e) * gate_up_expert_bytes;
        const std::uint64_t down_offset =
            static_cast<std::uint64_t>(e) * down_expert_bytes;
        writes.push_back(make_write_record(
            op_index, expr_id, binding_index, op_kind, metadata,
            gate_up.name, gate.raw_name, std::move(gate.slices),
            {I, H}, gate_up_offset));
        writes.push_back(make_write_record(
            op_index, expr_id, binding_index, op_kind, metadata,
            gate_up.name, up.raw_name, std::move(up.slices),
            {I, H}, gate_up_offset + proj_bytes));
        writes.push_back(make_write_record(
            op_index, expr_id, binding_index, op_kind, metadata,
            down.name, down_source.raw_name, std::move(down_source.slices),
            {H, I_down}, down_offset));
    }
    return true;
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

}  // namespace

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

    auto write_output_names =
        [](const std::vector<ExtentWrite>& writes) {
            std::vector<std::string> names;
            for (const auto& write : writes) {
                append_unique(names, write.output_name);
            }
            return names;
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
            if (try_lower_stack_expr_to_writes(
                    value, kInvalidStorageId, write_expr_id, binding_index,
                    op_kind, plan, metadata, tp_rank, tp_size, writes)) {
                return writes;
            }
            writes.clear();
            if (try_lower_join_expr_to_writes(
                    value, kInvalidStorageId, write_expr_id, binding_index,
                    op_kind, plan, metadata, output_name,
                    tp_rank, tp_size, writes)) {
                return writes;
            }
            writes.clear();
            if (lower_source_expr_to_write(
                    value_id, value, kInvalidStorageId, write_expr_id, binding_index,
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
                    const auto outputs = write_output_names(writes);
                    record_writes(std::move(writes), {});
                    remember_materialized(id, outputs);
                    if (std::find(outputs.begin(), outputs.end(), requested_name) !=
                        outputs.end()) {
                        return requested_name;
                    }
                    return outputs.empty() ? requested_name : outputs.front();
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
                const std::vector<std::string> outputs =
                    expr.secondary_runtime_name.empty()
                        ? std::vector<std::string>{output}
                        : std::vector<std::string>{output,
                              expr.secondary_runtime_name};
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
                    .outputs = outputs,
                    .dependencies = deps,
                });
                for (const auto& name : outputs) producers[name] = {step_index};
                remember_materialized(id, outputs);
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
            std::vector<std::string> inputs{expr.runtime_name};
            if (const auto spec = find_tensor_spec(plan, expr.runtime_name);
                spec != nullptr) {
                append_unique(inputs, spec->quant.scale_tensor);
                append_unique(inputs, spec->quant.zero_point_tensor);
            }
            const auto deps = dependencies_for(inputs);
            const std::size_t step_index = add_step(StorageInstr{
                .kind = StorageInstrKind::Attach,
                .transform_kind = StorageTransformKind::None,
                .op_index = kInvalidStorageId,
                .expr_id = id,
                .binding_index = kInvalidStorageId,
                .extent_write_indices = {},
                .tile_map_index = kInvalidStorageId,
                .inputs = inputs,
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
    return build_storage_program_from_algebra_only(
        plan, metadata, tp_rank, tp_size, transform_tile_bytes, optimizer);
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

    for (const auto& write : storage_program.extent_writes) {
        if (write.op_index != kInvalidStorageId) {
            throw std::runtime_error(
                "storage program: algebra extent write has layout op index");
        }
        validate_algebra_ref(write.expr_id, write.binding_index, "extent write");
    }
    for (const auto& tile : storage_program.tile_maps) {
        if (tile.op_index != kInvalidStorageId) {
            throw std::runtime_error(
                "storage program: algebra tile map has layout op index");
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
        if (step.op_index != kInvalidStorageId) {
            throw std::runtime_error(
                "storage program: algebra schedule has layout op index");
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

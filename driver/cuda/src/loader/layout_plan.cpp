#include "loader/layout_plan.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "loader/layout_typecheck.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

const char* layout_op_kind_name(LayoutOpKind kind) noexcept {
    switch (kind) {
    case LayoutOpKind::Read: return "Read";
    case LayoutOpKind::Copy: return "Copy";
    case LayoutOpKind::Slice: return "Slice";
    case LayoutOpKind::Shard: return "Shard";
    case LayoutOpKind::RowRangeShard: return "RowRangeShard";
    case LayoutOpKind::GroupedSliceConcat: return "GroupedSliceConcat";
    case LayoutOpKind::GroupedSlice: return "GroupedSlice";
    case LayoutOpKind::Cast: return "Cast";
    case LayoutOpKind::Concat: return "Concat";
    case LayoutOpKind::AxisConcat: return "AxisConcat";
    case LayoutOpKind::View: return "View";
    case LayoutOpKind::Alias: return "Alias";
    case LayoutOpKind::Drop: return "Drop";
    case LayoutOpKind::QuantizeRuntime: return "QuantizeRuntime";
    case LayoutOpKind::Dequantize: return "Dequantize";
    case LayoutOpKind::Deinterleave: return "Deinterleave";
    case LayoutOpKind::RepackLayout: return "RepackLayout";
    case LayoutOpKind::StackGroups: return "StackGroups";
    case LayoutOpKind::AttachMetadata: return "AttachMetadata";
    case LayoutOpKind::Materialize: return "Materialize";
    }
    return "?";
}

const char* layout_expr_kind_name(LayoutExprKind kind) noexcept {
    switch (kind) {
    case LayoutExprKind::Source: return "Source";
    case LayoutExprKind::Select: return "Select";
    case LayoutExprKind::Partition: return "Partition";
    case LayoutExprKind::Join: return "Join";
    case LayoutExprKind::Stack: return "Stack";
    case LayoutExprKind::Unzip: return "Unzip";
    case LayoutExprKind::Reorder: return "Reorder";
    case LayoutExprKind::View: return "View";
    case LayoutExprKind::Cast: return "Cast";
    case LayoutExprKind::Encode: return "Encode";
    case LayoutExprKind::Decode: return "Decode";
    case LayoutExprKind::Transcode: return "Transcode";
    case LayoutExprKind::Attach: return "Attach";
    case LayoutExprKind::Release: return "Release";
    case LayoutExprKind::Realize: return "Realize";
    }
    return "?";
}

const char* tensor_layout_kind_name(TensorLayoutKind kind) noexcept {
    switch (kind) {
    case TensorLayoutKind::Dense: return "dense";
    case TensorLayoutKind::RowPacked: return "row-packed";
    case TensorLayoutKind::AxisConcatenated: return "axis-concatenated";
    case TensorLayoutKind::Grouped: return "grouped";
    case TensorLayoutKind::QuantPacked: return "quant-packed";
    case TensorLayoutKind::View: return "view";
    }
    return "?";
}

const char* quant_format_name(QuantFormat format) noexcept {
    switch (format) {
    case QuantFormat::None: return "none";
    case QuantFormat::RuntimeFp8E4M3: return "runtime-fp8-e4m3";
    case QuantFormat::RuntimeInt8: return "runtime-int8";
    case QuantFormat::GptqInt4: return "gptq-int4";
    case QuantFormat::AwqInt4: return "awq-int4";
    case QuantFormat::CompressedFp8E4M3: return "compressed-fp8-e4m3";
    case QuantFormat::CompressedInt8: return "compressed-int8";
    case QuantFormat::Mxfp4E2M1E8M0: return "mxfp4-e2m1-e8m0";
    }
    return "?";
}

LayoutOp make_raw_load_op(
    LayoutOpKind kind,
    std::string output_name,
    std::string raw_name,
    int shard_axis)
{
    return LayoutOp{
        .kind = kind,
        .payload = RawLoadPayload{
            .output_name = std::move(output_name),
            .raw_name = std::move(raw_name),
            .shard_axis = shard_axis,
        },
    };
}

LayoutOp make_row_range_shard_op(
    std::string output_name,
    std::string raw_name,
    std::int64_t row_offset,
    std::int64_t rows)
{
    return LayoutOp{
        .kind = LayoutOpKind::RowRangeShard,
        .payload = RowRangeShardPayload{
            .output_name = std::move(output_name),
            .raw_name = std::move(raw_name),
            .row_offset = row_offset,
            .rows = rows,
        },
    };
}

LayoutOp make_tensor_op(
    LayoutOpKind kind,
    std::string output_name,
    std::vector<std::string> inputs,
    std::string secondary_output_name,
    int shard_axis)
{
    return LayoutOp{
        .kind = kind,
        .payload = TensorOpPayload{
            .output_name = std::move(output_name),
            .secondary_output_name = std::move(secondary_output_name),
            .inputs = std::move(inputs),
            .shard_axis = shard_axis,
        },
    };
}

LayoutOp make_slice_op(
    std::string output_name,
    std::string input,
    int slice_axis,
    std::int64_t slice_start,
    std::int64_t slice_length,
    int shard_axis)
{
    return LayoutOp{
        .kind = LayoutOpKind::Slice,
        .payload = SlicePayload{
            .output_name = std::move(output_name),
            .inputs = {std::move(input)},
            .slice_axis = slice_axis,
            .slice_start = slice_start,
            .slice_length = slice_length,
            .shard_axis = shard_axis,
        },
    };
}

LayoutOp make_axis_concat_op(
    std::string output_name,
    int shard_axis,
    std::vector<TensorSourceRef> sources)
{
    return LayoutOp{
        .kind = LayoutOpKind::AxisConcat,
        .payload = AxisConcatPayload{
            .output_name = std::move(output_name),
            .shard_axis = shard_axis,
            .sources = std::move(sources),
        },
    };
}

LayoutOp make_stack_groups_op(
    std::string output_name,
    std::string secondary_output_name,
    std::vector<std::string> inputs,
    std::vector<TensorSourceRef> sources)
{
    return LayoutOp{
        .kind = LayoutOpKind::StackGroups,
        .payload = StackGroupsPayload{
            .output_name = std::move(output_name),
            .secondary_output_name = std::move(secondary_output_name),
            .inputs = std::move(inputs),
            .sources = std::move(sources),
        },
    };
}

namespace {

const std::string kEmptyString;
const std::vector<std::string> kEmptyInputs;
const std::vector<TensorSourceRef> kEmptySources;

template <typename Payload>
const std::string& output_of(const Payload& payload) {
    return payload.output_name;
}

const char* ownership_kind_name(TensorOwnershipKind kind) noexcept {
    switch (kind) {
    case TensorOwnershipKind::Owned: return "owned";
    case TensorOwnershipKind::BorrowedView: return "borrowed-view";
    case TensorOwnershipKind::Alias: return "alias";
    case TensorOwnershipKind::Temporary: return "temporary";
    }
    return "?";
}

const char* parallel_kind_name(TensorParallelKind kind) noexcept {
    switch (kind) {
    case TensorParallelKind::Replicated: return "replicated";
    case TensorParallelKind::Column: return "column";
    case TensorParallelKind::Row: return "row";
    case TensorParallelKind::Expert: return "expert";
    case TensorParallelKind::Custom: return "custom";
    }
    return "?";
}

const char* quant_granularity_name(QuantGranularity granularity) noexcept {
    switch (granularity) {
    case QuantGranularity::None: return "none";
    case QuantGranularity::PerTensor: return "per-tensor";
    case QuantGranularity::PerChannel: return "per-channel";
    case QuantGranularity::PerGroup: return "per-group";
    }
    return "?";
}

bool op_produces_tensor(LayoutOpKind kind) noexcept {
    return kind != LayoutOpKind::Drop &&
           kind != LayoutOpKind::AttachMetadata;
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

}  // namespace

const std::string& layout_op_output(const LayoutOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::string& {
            return payload.output_name;
        },
        op.payload);
}

const std::string& layout_op_secondary_output(const LayoutOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::string& {
            if constexpr (requires { payload.secondary_output_name; }) {
                return payload.secondary_output_name;
            } else {
                return kEmptyString;
            }
        },
        op.payload);
}

const std::string& layout_op_raw_name(const LayoutOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::string& {
            if constexpr (requires { payload.raw_name; }) {
                return payload.raw_name;
            } else {
                return kEmptyString;
            }
        },
        op.payload);
}

const std::vector<std::string>& layout_op_inputs(const LayoutOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::vector<std::string>& {
            if constexpr (requires { payload.inputs; }) {
                return payload.inputs;
            } else {
                return kEmptyInputs;
            }
        },
        op.payload);
}

int layout_op_shard_axis(const LayoutOp& op) {
    return std::visit(
        [](const auto& payload) -> int {
            if constexpr (requires { payload.shard_axis; }) {
                return payload.shard_axis;
            } else {
                return -1;
            }
        },
        op.payload);
}

int layout_op_slice_axis(const LayoutOp& op) {
    if (const auto* payload = std::get_if<SlicePayload>(&op.payload)) {
        return payload->slice_axis;
    }
    return -1;
}

std::int64_t layout_op_row_offset(const LayoutOp& op) {
    if (const auto* payload = std::get_if<RowRangeShardPayload>(&op.payload)) {
        return payload->row_offset;
    }
    return 0;
}

std::int64_t layout_op_rows(const LayoutOp& op) {
    if (const auto* payload = std::get_if<RowRangeShardPayload>(&op.payload)) {
        return payload->rows;
    }
    return 0;
}

std::int64_t layout_op_slice_start(const LayoutOp& op) {
    if (const auto* payload = std::get_if<SlicePayload>(&op.payload)) {
        return payload->slice_start;
    }
    return 0;
}

std::int64_t layout_op_slice_length(const LayoutOp& op) {
    if (const auto* payload = std::get_if<SlicePayload>(&op.payload)) {
        return payload->slice_length;
    }
    return 0;
}

const std::vector<TensorSourceRef>& layout_op_sources(const LayoutOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::vector<TensorSourceRef>& {
            if constexpr (requires { payload.sources; }) {
                return payload.sources;
            } else {
                return kEmptySources;
            }
        },
        op.payload);
}

void set_layout_op_output(LayoutOp& op, std::string output_name) {
    std::visit(
        [&](auto& payload) {
            payload.output_name = output_name;
        },
        op.payload);
}

void validate_layout_plan(const LayoutPlan& plan) {
    validate_layout_algebra(plan.algebra);

    for (const auto& [name, spec] : plan.tensors) {
        if (name.empty() || spec.name.empty()) {
            throw std::runtime_error("layout plan: tensor spec has empty name");
        }
        if (name != spec.name) {
            throw std::runtime_error(
                "layout plan: tensor spec key/name mismatch for '" + name + "'");
        }
        for (const auto dim : spec.shape) {
            if (dim < 0) {
                throw std::runtime_error(
                    "layout plan: tensor spec '" + name +
                    "' has a negative dimension");
            }
        }
        if (spec.ownership == TensorOwnershipKind::BorrowedView ||
            spec.ownership == TensorOwnershipKind::Alias) {
            if (spec.backing_tensor.empty()) {
                throw std::runtime_error(
                    "layout plan: view/alias spec '" + name +
                    "' has no backing tensor");
            }
            if (!plan.tensors.contains(spec.backing_tensor)) {
                throw std::runtime_error(
                    "layout plan: view/alias spec '" + name +
                    "' references missing backing tensor '" +
                    spec.backing_tensor + "'");
            }
        }
        if (spec.quant.format != QuantFormat::None) {
            if (spec.quant.scale_tensor.empty()) {
                throw std::runtime_error(
                    "layout plan: quant tensor spec '" + name +
                    "' has no scale tensor");
            }
            if (!plan.tensors.contains(spec.quant.scale_tensor)) {
                throw std::runtime_error(
                    "layout plan: quant tensor spec '" + name +
                    "' references missing scale tensor '" +
                    spec.quant.scale_tensor + "'");
            }
            if (!spec.quant.zero_point_tensor.empty() &&
                !plan.tensors.contains(spec.quant.zero_point_tensor)) {
                throw std::runtime_error(
                    "layout plan: quant tensor spec '" + name +
                    "' references missing zero-point tensor '" +
                    spec.quant.zero_point_tensor + "'");
            }
        }
    }

    std::unordered_set<std::string> produced;
    produced.reserve(plan.ops.size() + plan.tensors.size());

    for (const auto& op : plan.ops) {
        if (layout_op_output(op).empty() &&
            (op.kind != LayoutOpKind::Drop || layout_op_inputs(op).empty())) {
            throw std::runtime_error(
                std::string("layout plan: ") + layout_op_kind_name(op.kind) +
                " op has empty output name");
        }
        for (const auto& input : layout_op_inputs(op)) {
            if (input.empty()) {
                throw std::runtime_error(
                    "layout plan: " +
                    std::string(layout_op_kind_name(op.kind)) +
                    " op for '" + layout_op_output(op) + "' has an empty input");
            }
            if (!produced.contains(input)) {
                throw std::runtime_error(
                    "layout plan: " +
                    std::string(layout_op_kind_name(op.kind)) +
                    " op for '" + layout_op_output(op) +
                    "' reads tensor before it is produced: '" + input + "'");
            }
        }
        if (op_produces_tensor(op.kind) &&
            !plan.tensors.contains(layout_op_output(op))) {
            throw std::runtime_error(
                "layout plan: " + std::string(layout_op_kind_name(op.kind)) +
                " op produces tensor without TensorDecl: '" +
                layout_op_output(op) + "'");
        }
        if (op_produces_tensor(op.kind) &&
            !produced.insert(layout_op_output(op)).second) {
            throw std::runtime_error(
                "layout plan: duplicate runtime tensor output '" +
                layout_op_output(op) + "'");
        }
        if (!layout_op_secondary_output(op).empty() &&
            !plan.tensors.contains(layout_op_secondary_output(op))) {
            throw std::runtime_error(
                "layout plan: " + std::string(layout_op_kind_name(op.kind)) +
                " op produces secondary tensor without TensorDecl: '" +
                layout_op_secondary_output(op) + "'");
        }
        if (!layout_op_secondary_output(op).empty() &&
            !produced.insert(layout_op_secondary_output(op)).second) {
            throw std::runtime_error(
                "layout plan: duplicate secondary runtime tensor output '" +
                layout_op_secondary_output(op) + "'");
        }
        if ((op.kind == LayoutOpKind::Read ||
             op.kind == LayoutOpKind::Copy ||
             op.kind == LayoutOpKind::Shard ||
             op.kind == LayoutOpKind::RowRangeShard ||
             op.kind == LayoutOpKind::GroupedSliceConcat ||
             op.kind == LayoutOpKind::GroupedSlice) &&
            layout_op_raw_name(op).empty()) {
            throw std::runtime_error(
                "layout plan: op for '" + layout_op_output(op) +
                "' has empty raw tensor name");
        }
        if (op.kind == LayoutOpKind::AxisConcat) {
            if (layout_op_sources(op).empty()) {
                throw std::runtime_error(
                    "layout plan: AxisConcat op for '" + layout_op_output(op) +
                    "' has no sources");
            }
            std::unordered_set<std::string> view_names;
            for (const auto& src : layout_op_sources(op)) {
                if (src.raw_name.empty() || src.view_name.empty()) {
                    throw std::runtime_error(
                        "layout plan: AxisConcat op for '" + layout_op_output(op) +
                        "' has an empty raw/view source");
                }
                if (!view_names.insert(src.view_name).second) {
                    throw std::runtime_error(
                        "layout plan: AxisConcat op for '" + layout_op_output(op) +
                        "' has duplicate view '" + src.view_name + "'");
                }
                if (!plan.tensors.contains(src.view_name)) {
                    throw std::runtime_error(
                        "layout plan: AxisConcat op for '" + layout_op_output(op) +
                        "' has no TensorDecl for view '" + src.view_name + "'");
                }
                if (!produced.insert(src.view_name).second) {
                    throw std::runtime_error(
                        "layout plan: duplicate AxisConcat view output '" +
                        src.view_name + "'");
                }
            }
        }
        if ((op.kind == LayoutOpKind::Slice ||
             op.kind == LayoutOpKind::Cast ||
             op.kind == LayoutOpKind::Concat ||
             op.kind == LayoutOpKind::View ||
             op.kind == LayoutOpKind::Alias ||
             op.kind == LayoutOpKind::QuantizeRuntime ||
             op.kind == LayoutOpKind::Dequantize ||
             op.kind == LayoutOpKind::RepackLayout ||
             op.kind == LayoutOpKind::Deinterleave ||
             op.kind == LayoutOpKind::Materialize) &&
            layout_op_inputs(op).empty()) {
            throw std::runtime_error(
                "layout plan: " + std::string(layout_op_kind_name(op.kind)) +
                " op for '" + layout_op_output(op) + "' has no inputs");
        }
        if (op.kind == LayoutOpKind::Dequantize && layout_op_inputs(op).size() < 2) {
            throw std::runtime_error(
                "layout plan: Dequantize op for '" + layout_op_output(op) +
                "' requires weight and scale inputs");
        }
        if (op.kind == LayoutOpKind::Deinterleave &&
            layout_op_secondary_output(op).empty()) {
            throw std::runtime_error(
                "layout plan: Deinterleave op for '" + layout_op_output(op) +
                "' requires a secondary output name");
        }
        if (op.kind == LayoutOpKind::RepackLayout) {
            if (layout_op_inputs(op).size() < 2) {
                throw std::runtime_error(
                    "layout plan: RepackLayout op for '" + layout_op_output(op) +
                    "' requires weight and metadata inputs");
            }
            if (layout_op_secondary_output(op).empty()) {
                throw std::runtime_error(
                    "layout plan: RepackLayout op for '" + layout_op_output(op) +
                    "' requires a secondary output name");
            }
            const auto spec_it = plan.tensors.find(layout_op_output(op));
            if (spec_it != plan.tensors.end() &&
                !spec_it->second.quant.zero_point_tensor.empty() &&
                !produced.insert(spec_it->second.quant.zero_point_tensor).second) {
                throw std::runtime_error(
                    "layout plan: duplicate RepackLayout zero-point output '" +
                    spec_it->second.quant.zero_point_tensor + "'");
            }
        }
        if (op.kind == LayoutOpKind::StackGroups) {
            if (layout_op_secondary_output(op).empty()) {
                throw std::runtime_error(
                    "layout plan: StackGroups op for '" + layout_op_output(op) +
                    "' requires a secondary output name");
            }
            const bool has_input_triples =
                !layout_op_inputs(op).empty() && layout_op_inputs(op).size() % 3 == 0;
            const bool has_raw_triples =
                !layout_op_sources(op).empty() && layout_op_sources(op).size() % 3 == 0;
            if (has_input_triples == has_raw_triples) {
                throw std::runtime_error(
                    "layout plan: StackGroups op for '" + layout_op_output(op) +
                    "' expects either input tensor triples or raw source triples");
            }
            if (has_raw_triples) {
                for (const auto& src : layout_op_sources(op)) {
                    if (src.raw_name.empty() || src.view_name.empty()) {
                        throw std::runtime_error(
                            "layout plan: StackGroups op for '" +
                            layout_op_output(op) + "' has empty raw source metadata");
                    }
                }
            }
        }
        if (op.kind == LayoutOpKind::AttachMetadata) {
            const auto spec_it = plan.tensors.find(layout_op_output(op));
            if (spec_it == plan.tensors.end()) {
                throw std::runtime_error(
                    "layout plan: AttachMetadata has no TensorDecl for '" +
                    layout_op_output(op) + "'");
            }
            const auto& spec = spec_it->second;
            if (!produced.contains(layout_op_output(op))) {
                throw std::runtime_error(
                    "layout plan: AttachMetadata runs before weight tensor '" +
                    layout_op_output(op) + "' is produced");
            }
            if (!produced.contains(spec.quant.scale_tensor)) {
                throw std::runtime_error(
                    "layout plan: AttachMetadata for '" + layout_op_output(op) +
                    "' runs before scale tensor '" +
                    spec.quant.scale_tensor + "' is produced");
            }
            if (!spec.quant.zero_point_tensor.empty() &&
                !produced.contains(spec.quant.zero_point_tensor)) {
                throw std::runtime_error(
                    "layout plan: AttachMetadata for '" + layout_op_output(op) +
                    "' runs before zero-point tensor '" +
                    spec.quant.zero_point_tensor + "' is produced");
            }
        }
    }
}

void build_layout_algebra_from_ops(LayoutPlan& plan) {
    plan.algebra = LayoutAlgebra{};
    std::unordered_map<std::string, LayoutExprId> latest;
    latest.reserve(plan.tensors.size());

    auto find_decl = [&](const std::string& name) -> TensorDecl {
        const auto it = plan.tensors.find(name);
        if (it != plan.tensors.end()) return it->second;
        TensorDecl decl;
        decl.name = name;
        return decl;
    };
    auto add_expr = [&](LayoutExpr expr) -> LayoutExprId {
        const LayoutExprId id = plan.algebra.exprs.size();
        plan.algebra.exprs.push_back(std::move(expr));
        return id;
    };
    auto make_expr = [](LayoutExprKind kind, TensorDecl decl) {
        LayoutExpr expr;
        expr.kind = kind;
        expr.decl = std::move(decl);
        expr.dtype = expr.decl.dtype;
        expr.encoding = expr.decl.quant;
        return expr;
    };
    auto source_expr = [&](const std::string& raw_name,
                           const TensorDecl& decl) -> LayoutExprId {
        LayoutExpr expr = make_expr(LayoutExprKind::Source, decl);
        expr.raw_name = raw_name;
        return add_expr(std::move(expr));
    };
    auto realize_expr = [&](const std::string& output_name,
                            LayoutExprId input,
                            TensorDecl decl) {
        decl.name = output_name;
        LayoutExpr expr = make_expr(LayoutExprKind::Realize, std::move(decl));
        expr.inputs = {input};
        expr.runtime_name = output_name;
        const LayoutExprId root = add_expr(std::move(expr));
        latest[output_name] = root;
        plan.algebra.bindings.push_back(LayoutBinding{
            .runtime_name = output_name,
            .root = root,
        });
    };
    auto input_exprs = [&](const LayoutOp& step) {
        std::vector<LayoutExprId> inputs;
        for (const auto& input : layout_op_inputs(step)) {
            const auto it = latest.find(input);
            if (it != latest.end()) {
                inputs.push_back(it->second);
            } else {
                inputs.push_back(source_expr(input, find_decl(input)));
            }
        }
        return inputs;
    };

    for (const auto& step : plan.ops) {
        const std::string& output = layout_op_output(step);
        switch (step.kind) {
        case LayoutOpKind::Read:
        case LayoutOpKind::Copy:
        case LayoutOpKind::Shard:
        case LayoutOpKind::GroupedSliceConcat:
        case LayoutOpKind::GroupedSlice: {
            TensorDecl decl = find_decl(output);
            LayoutExprId expr = source_expr(layout_op_raw_name(step), decl);
            if (layout_op_shard_axis(step) >= 0) {
                LayoutExpr partition =
                    make_expr(LayoutExprKind::Partition, decl);
                partition.inputs = {expr};
                partition.axis = layout_op_shard_axis(step);
                expr = add_expr(std::move(partition));
            }
            realize_expr(output, expr, std::move(decl));
            break;
        }
        case LayoutOpKind::RowRangeShard: {
            TensorDecl decl = find_decl(output);
            LayoutExprId expr = source_expr(layout_op_raw_name(step), decl);
            LayoutExpr select = make_expr(LayoutExprKind::Select, decl);
            select.inputs = {expr};
            select.axis = 0;
            select.start = layout_op_row_offset(step);
            select.length = layout_op_rows(step);
            expr = add_expr(std::move(select));
            realize_expr(output, expr, std::move(decl));
            break;
        }
        case LayoutOpKind::AxisConcat: {
            TensorDecl decl = find_decl(output);
            std::vector<LayoutExprId> inputs;
            inputs.reserve(layout_op_sources(step).size());
            for (const auto& source : layout_op_sources(step)) {
                inputs.push_back(source_expr(source.raw_name, decl));
            }
            LayoutExpr join = make_expr(LayoutExprKind::Join, decl);
            join.inputs = std::move(inputs);
            join.axis = layout_op_shard_axis(step) >= 0
                ? layout_op_shard_axis(step)
                : 0;
            const LayoutExprId joined = add_expr(std::move(join));
            realize_expr(output, joined, decl);
            for (const auto& source : layout_op_sources(step)) {
                TensorDecl view_decl = find_decl(source.view_name);
                LayoutExpr view_expr = make_expr(LayoutExprKind::View, view_decl);
                view_expr.inputs = {joined};
                view_expr.runtime_name = source.view_name;
                const LayoutExprId view = add_expr(std::move(view_expr));
                realize_expr(source.view_name, view, std::move(view_decl));
            }
            break;
        }
        case LayoutOpKind::Slice: {
            TensorDecl decl = find_decl(output);
            auto inputs = input_exprs(step);
            if (inputs.empty()) break;
            LayoutExpr select = make_expr(LayoutExprKind::Select, decl);
            select.inputs = {inputs.front()};
            select.axis = layout_op_slice_axis(step);
            select.start = layout_op_slice_start(step);
            select.length = layout_op_slice_length(step);
            const LayoutExprId selected = add_expr(std::move(select));
            realize_expr(output, selected, std::move(decl));
            break;
        }
        case LayoutOpKind::Concat: {
            TensorDecl decl = find_decl(output);
            LayoutExpr join = make_expr(LayoutExprKind::Join, decl);
            join.inputs = input_exprs(step);
            join.axis = 0;
            const LayoutExprId joined = add_expr(std::move(join));
            realize_expr(output, joined, std::move(decl));
            break;
        }
        case LayoutOpKind::StackGroups: {
            TensorDecl decl = find_decl(output);
            std::vector<LayoutExprId> inputs;
            if (!layout_op_sources(step).empty()) {
                inputs.reserve(layout_op_sources(step).size());
                for (const auto& source : layout_op_sources(step)) {
                    inputs.push_back(source_expr(source.raw_name, decl));
                }
            } else {
                inputs = input_exprs(step);
            }
            LayoutExpr stack = make_expr(LayoutExprKind::Stack, decl);
            stack.inputs = std::move(inputs);
            stack.axis = 0;
            const LayoutExprId stacked = add_expr(std::move(stack));
            realize_expr(output, stacked, decl);
            const auto& secondary = layout_op_secondary_output(step);
            if (!secondary.empty()) {
                TensorDecl secondary_decl = find_decl(secondary);
                realize_expr(secondary, stacked, std::move(secondary_decl));
            }
            break;
        }
        case LayoutOpKind::Cast:
        case LayoutOpKind::QuantizeRuntime: {
            TensorDecl decl = find_decl(output);
            auto inputs = input_exprs(step);
            if (inputs.empty()) break;
            const LayoutExprKind kind = step.kind == LayoutOpKind::Cast
                ? LayoutExprKind::Cast
                : LayoutExprKind::Encode;
            LayoutExpr converted_expr = make_expr(kind, decl);
            converted_expr.inputs = {inputs.front()};
            converted_expr.runtime_name = output;
            const LayoutExprId converted = add_expr(std::move(converted_expr));
            realize_expr(output, converted, std::move(decl));
            break;
        }
        case LayoutOpKind::Dequantize: {
            TensorDecl decl = find_decl(output);
            LayoutExpr decode = make_expr(LayoutExprKind::Decode, decl);
            decode.inputs = input_exprs(step);
            decode.runtime_name = output;
            const LayoutExprId decoded = add_expr(std::move(decode));
            realize_expr(output, decoded, std::move(decl));
            break;
        }
        case LayoutOpKind::RepackLayout:
            {
                TensorDecl decl = find_decl(output);
                LayoutExpr reorder = make_expr(LayoutExprKind::Reorder, decl);
                reorder.inputs = input_exprs(step);
                reorder.runtime_name = output;
                reorder.secondary_runtime_name = layout_op_secondary_output(step);
                const LayoutExprId reblocked = add_expr(std::move(reorder));
                realize_expr(output, reblocked, decl);
                const auto& secondary = layout_op_secondary_output(step);
                if (!secondary.empty()) {
                    TensorDecl secondary_decl = find_decl(secondary);
                    realize_expr(secondary, reblocked, std::move(secondary_decl));
                }
                break;
            }
        case LayoutOpKind::Deinterleave: {
            TensorDecl decl = find_decl(output);
            LayoutExpr unzip = make_expr(LayoutExprKind::Unzip, decl);
            unzip.inputs = input_exprs(step);
            unzip.runtime_name = output;
            unzip.secondary_runtime_name = layout_op_secondary_output(step);
            unzip.axis = layout_op_shard_axis(step);
            const LayoutExprId unzipped = add_expr(std::move(unzip));
            realize_expr(output, unzipped, decl);
            const auto& secondary = layout_op_secondary_output(step);
            if (!secondary.empty()) {
                TensorDecl secondary_decl = find_decl(secondary);
                realize_expr(secondary, unzipped, std::move(secondary_decl));
            }
            break;
        }
        case LayoutOpKind::View:
        case LayoutOpKind::Alias:
        case LayoutOpKind::Materialize: {
            TensorDecl decl = find_decl(output);
            auto inputs = input_exprs(step);
            if (inputs.empty()) break;
            LayoutExpr view_expr = make_expr(
                step.kind == LayoutOpKind::Materialize
                    ? LayoutExprKind::Realize
                    : LayoutExprKind::View,
                decl);
            view_expr.inputs = {inputs.front()};
            view_expr.runtime_name = output;
            const LayoutExprId view = add_expr(std::move(view_expr));
            latest[output] = view;
            plan.algebra.bindings.push_back(LayoutBinding{
                .runtime_name = output,
                .root = view,
            });
            break;
        }
        case LayoutOpKind::AttachMetadata: {
            TensorDecl decl = find_decl(output);
            const auto it = latest.find(output);
            if (it == latest.end()) break;
            LayoutExpr attach = make_expr(LayoutExprKind::Attach, decl);
            attach.inputs = {it->second};
            attach.runtime_name = output;
            const LayoutExprId attached = add_expr(std::move(attach));
            latest[output] = attached;
            break;
        }
        case LayoutOpKind::Drop:
            {
                std::vector<LayoutExprId> release_inputs;
                for (const auto& input : layout_op_inputs(step)) {
                    const auto it = latest.find(input);
                    if (it != latest.end()) {
                        release_inputs.push_back(it->second);
                    }
                }
                if (layout_op_inputs(step).empty()) {
                    const auto it = latest.find(output);
                    if (it != latest.end()) {
                        release_inputs.push_back(it->second);
                    }
                }
                if (!release_inputs.empty()) {
                    TensorDecl decl = find_decl(output);
                    LayoutExpr release =
                        make_expr(LayoutExprKind::Release, std::move(decl));
                    release.inputs = std::move(release_inputs);
                    release.runtime_name = output;
                    (void)add_expr(std::move(release));
                }
            }
            for (const auto& input : layout_op_inputs(step)) {
                latest.erase(input);
            }
            if (layout_op_inputs(step).empty()) {
                latest.erase(output);
            }
            break;
        }
    }
}

std::string describe_layout_plan(const LayoutPlan& plan) {
    std::ostringstream out;
    out << plan.ops.size() << " layout ops, "
        << plan.tensors.size() << " runtime tensor specs"
        << ", algebra_exprs=" << plan.algebra.exprs.size()
        << ", algebra_roots=" << plan.algebra.bindings.size()
        << ", persistent=" << (plan.memory.persistent_bytes / (1024 * 1024))
        << " MiB"
        << ", temp<=" << (plan.memory.max_temporary_bytes / (1024 * 1024))
        << " MiB"
        << ", peak~=" << (plan.memory.estimated_peak_bytes / (1024 * 1024))
        << " MiB";
    if (plan.axis_concat_groups > 0) {
        out << ", axis_concat_groups=" << plan.axis_concat_groups;
    }
    return out.str();
}

std::string dump_layout_plan_json(const LayoutPlan& plan) {
    validate_layout_plan(plan);

    std::ostringstream out;
    out << "{\n";
    out << "  \"plan_kind\": \"LayoutPlan\",\n";
    out << "  \"expr_kind\": \"LayoutExpr\",\n";
    out << "  \"summary\": ";
    json_string(out, describe_layout_plan(plan));
    out << ",\n";
    out << "  \"memory\": {"
        << "\"persistent_bytes\":" << plan.memory.persistent_bytes << ','
        << "\"max_temporary_bytes\":" << plan.memory.max_temporary_bytes << ','
        << "\"estimated_peak_bytes\":" << plan.memory.estimated_peak_bytes
        << "},\n";

    out << "  \"ops\": [";
    for (std::size_t i = 0; i < plan.ops.size(); ++i) {
        const LayoutOp& op = plan.ops[i];
        if (i) out << ',';
        out << "\n    {\"kind\":";
        json_string(out, layout_op_kind_name(op.kind));
        out << ",\"output_name\":";
        json_string(out, layout_op_output(op));
        out << ",\"secondary_output_name\":";
        json_string(out, layout_op_secondary_output(op));
        out << ",\"raw_name\":";
        json_string(out, layout_op_raw_name(op));
        out << ",\"inputs\":[";
        for (std::size_t j = 0; j < layout_op_inputs(op).size(); ++j) {
            if (j) out << ',';
            json_string(out, layout_op_inputs(op)[j]);
        }
        out << "],\"shard_axis\":" << layout_op_shard_axis(op)
            << ",\"slice_axis\":" << layout_op_slice_axis(op)
            << ",\"row_offset\":" << layout_op_row_offset(op)
            << ",\"rows\":" << layout_op_rows(op)
            << ",\"slice_start\":" << layout_op_slice_start(op)
            << ",\"slice_length\":" << layout_op_slice_length(op);
        if (!layout_op_sources(op).empty()) {
            out << ",\"sources\":[";
            for (std::size_t j = 0; j < layout_op_sources(op).size(); ++j) {
                if (j) out << ',';
                out << "{\"raw_name\":";
                json_string(out, layout_op_sources(op)[j].raw_name);
                out << ",\"view_name\":";
                json_string(out, layout_op_sources(op)[j].view_name);
                out << '}';
            }
            out << ']';
        }
        out << '}';
    }
    if (!plan.ops.empty()) out << '\n';
    out << "  ],\n";

    out << "  \"algebra\": {\"expr_kind\":\"LayoutExpr\",";
    out << "\"exprs\":[";
    for (std::size_t i = 0; i < plan.algebra.exprs.size(); ++i) {
        const LayoutExpr& expr = plan.algebra.exprs[i];
        if (i) out << ',';
        out << "\n    {\"id\":" << i << ",\"kind\":";
        json_string(out, layout_expr_kind_name(expr.kind));
        out << ",\"inputs\":[";
        for (std::size_t j = 0; j < expr.inputs.size(); ++j) {
            if (j) out << ',';
            out << expr.inputs[j];
        }
        out << "],\"raw_name\":";
        json_string(out, expr.raw_name);
        out << ",\"runtime_name\":";
        json_string(out, expr.runtime_name);
        out << ",\"secondary_runtime_name\":";
        json_string(out, expr.secondary_runtime_name);
        out << ",\"axis\":" << expr.axis
            << ",\"start\":" << expr.start
            << ",\"length\":" << expr.length
            << ",\"dtype\":";
        json_string(out, dtype_name(expr.dtype));
        out << ",\"decl\":{\"name\":";
        json_string(out, expr.decl.name);
        out << ",\"dtype\":";
        json_string(out, dtype_name(expr.decl.dtype));
        out << ",\"shape\":";
        json_shape(out, expr.decl.shape);
        out << ",\"layout\":";
        json_string(out, tensor_layout_kind_name(expr.decl.layout));
        out << "}}";
    }
    if (!plan.algebra.exprs.empty()) out << '\n';
    out << "  ],\"bindings\":[";
    for (std::size_t i = 0; i < plan.algebra.bindings.size(); ++i) {
        const auto& binding = plan.algebra.bindings[i];
        if (i) out << ',';
        out << "{\"runtime_name\":";
        json_string(out, binding.runtime_name);
        out << ",\"root\":" << binding.root << '}';
    }
    out << "]},\n";

    out << "  \"tensors\": [";
    bool first = true;
    for (const auto& [name, spec] : plan.tensors) {
        if (!first) out << ',';
        first = false;
        out << "\n    {\"name\":";
        json_string(out, name);
        out << ",\"dtype\":";
        json_string(out, dtype_name(spec.dtype));
        out << ",\"shape\":";
        json_shape(out, spec.shape);
        out << ",\"layout\":";
        json_string(out, tensor_layout_kind_name(spec.layout));
        out << ",\"ownership\":";
        json_string(out, ownership_kind_name(spec.ownership));
        out << ",\"parallel\":";
        json_string(out, parallel_kind_name(spec.parallel));
        out << ",\"backing_tensor\":";
        json_string(out, spec.backing_tensor);
        out << ",\"quant\":{";
        out << "\"format\":";
        json_string(out, quant_format_name(spec.quant.format));
        out << ",\"granularity\":";
        json_string(out, quant_granularity_name(spec.quant.granularity));
        out << ",\"group_size\":" << spec.quant.group_size
            << ",\"channel_axis\":" << spec.quant.channel_axis
            << ",\"scale_tensor\":";
        json_string(out, spec.quant.scale_tensor);
        out << ",\"zero_point_tensor\":";
        json_string(out, spec.quant.zero_point_tensor);
        out << "}}";
    }
    if (!plan.tensors.empty()) out << '\n';
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

}  // namespace pie_cuda_driver

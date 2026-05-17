#include "loader/layout_plan.hpp"

#include <sstream>
#include <stdexcept>

#include "loader/layout_typecheck.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

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

namespace {

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
}

std::string describe_layout_plan(const LayoutPlan& plan) {
    std::ostringstream out;
    out << plan.tensors.size() << " runtime tensor specs"
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

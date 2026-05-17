#include "loader/layout_typecheck.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_cuda_driver {

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("layout algebra: " + message);
    }
}

const LayoutExpr& input_expr(
    const LayoutAlgebra& algebra,
    const LayoutExpr& expr,
    std::size_t input_index,
    std::size_t expr_index)
{
    require(
        input_index < expr.inputs.size(),
        "missing input " + std::to_string(input_index) +
            " for expr " + std::to_string(expr_index));
    const LayoutExprId id = expr.inputs[input_index];
    require(
        id < expr_index,
        "expr " + std::to_string(expr_index) +
            " has non-topological input " + std::to_string(id));
    return algebra.exprs[id];
}

void validate_axis(const LayoutExpr& expr, std::size_t expr_index) {
    require(expr.axis >= 0, "negative axis on expr " + std::to_string(expr_index));
    require(
        expr.axis < static_cast<int>(expr.decl.shape.size()),
        "axis out of range on expr " + std::to_string(expr_index));
}

void validate_same_rank_join(
    const LayoutAlgebra& algebra,
    const LayoutExpr& expr,
    std::size_t expr_index)
{
    validate_axis(expr, expr_index);
    require(!expr.inputs.empty(), "empty join/stack at expr " +
        std::to_string(expr_index));
    const auto rank = input_expr(algebra, expr, 0, expr_index).decl.shape.size();
    for (std::size_t i = 1; i < expr.inputs.size(); ++i) {
        require(
            input_expr(algebra, expr, i, expr_index).decl.shape.size() == rank,
            "rank mismatch in join/stack at expr " + std::to_string(expr_index));
    }
}

std::string kind_name(LayoutExprKind kind) {
    return layout_expr_kind_name(kind);
}

void require_decl_basics(const TensorDecl& decl, std::size_t expr_index) {
    require(!decl.name.empty(), "empty TensorDecl name at expr " +
        std::to_string(expr_index));
    for (const auto dim : decl.shape) {
        require(dim >= 0, "negative dimension in TensorDecl at expr " +
            std::to_string(expr_index));
    }
}

void require_same_shape(
    const std::vector<std::int64_t>& a,
    const std::vector<std::int64_t>& b,
    const std::string& message)
{
    require(a == b, message);
}

void require_same_type(
    const TensorDecl& inferred,
    const TensorDecl& declared,
    std::size_t expr_index,
    const char* context)
{
    require(
        inferred.dtype == declared.dtype,
        std::string(context) + " dtype mismatch at expr " +
            std::to_string(expr_index) + ": inferred " +
            dtype_name(inferred.dtype) + ", declared " +
            dtype_name(declared.dtype));
    require_same_shape(
        inferred.shape, declared.shape,
        std::string(context) + " shape mismatch at expr " +
            std::to_string(expr_index));
}

TensorDecl infer_unary_passthrough(
    const LayoutAlgebra& algebra,
    const LayoutExpr& expr,
    std::size_t expr_index)
{
    TensorDecl inferred = input_expr(algebra, expr, 0, expr_index).decl;
    inferred.name = expr.decl.name;
    inferred.layout = expr.decl.layout;
    inferred.ownership = expr.decl.ownership;
    inferred.parallel = expr.decl.parallel;
    inferred.quant = expr.decl.quant;
    inferred.backing_tensor = expr.decl.backing_tensor;
    return inferred;
}

void validate_fp8_decode_decl(
    const LayoutAlgebra& algebra,
    const LayoutExpr& expr,
    std::size_t expr_index,
    const char* context,
    bool requires_bf16_output)
{
    const TensorDecl& src = input_expr(algebra, expr, 0, expr_index).decl;
    const TensorDecl& scale = input_expr(algebra, expr, 1, expr_index).decl;
    if (requires_bf16_output) {
        require(
            expr.decl.dtype == DType::BF16,
            std::string(context) + " FP8 output must be BF16 at expr " +
                std::to_string(expr_index));
    }
    require_same_shape(
        src.shape, expr.decl.shape,
        std::string(context) + " FP8 shape mismatch at expr " +
            std::to_string(expr_index));
    require(
        scale.dtype == DType::FP32 || scale.dtype == DType::BF16,
        std::string(context) + " FP8 scale must be FP32/BF16 at expr " +
            std::to_string(expr_index));
    require(
        scale.shape == std::vector<std::int64_t>{1} ||
            (!expr.decl.shape.empty() &&
             scale.shape == std::vector<std::int64_t>{expr.decl.shape[0]}),
        std::string(context) + " FP8 scale shape mismatch at expr " +
            std::to_string(expr_index));
}

void validate_mxfp4_decode_decl(
    const LayoutAlgebra& algebra,
    const LayoutExpr& expr,
    std::size_t expr_index,
    const char* context,
    bool requires_bf16_output)
{
    const TensorDecl& src = input_expr(algebra, expr, 0, expr_index).decl;
    const TensorDecl& scale = input_expr(algebra, expr, 1, expr_index).decl;
    if (requires_bf16_output) {
        require(
            expr.decl.dtype == DType::BF16,
            std::string(context) + " MXFP4 output must be BF16 at expr " +
                std::to_string(expr_index));
    }
    require(
        src.dtype == DType::UINT8 && scale.dtype == DType::UINT8,
        std::string(context) + " MXFP4 source/scale dtype mismatch at expr " +
            std::to_string(expr_index));
    require(
        !scale.shape.empty() && src.shape.size() == scale.shape.size() + 1 &&
            !src.shape.empty() && src.shape.back() == 16,
        std::string(context) + " MXFP4 packed rank mismatch at expr " +
            std::to_string(expr_index));
    auto decoded_shape = scale.shape;
    decoded_shape.back() *= 32;
    require_same_shape(
        decoded_shape, expr.decl.shape,
        std::string(context) + " MXFP4 shape mismatch at expr " +
            std::to_string(expr_index));
}

void validate_encoding_expr_decl(
    const LayoutAlgebra& algebra,
    const LayoutExpr& expr,
    std::size_t expr_index,
    const char* context)
{
    require(
        expr.inputs.size() >= 2,
        std::string(context) + " expects encoded tensor and metadata inputs at expr " +
            std::to_string(expr_index));
    const TensorDecl& src = input_expr(algebra, expr, 0, expr_index).decl;
    const bool requires_bf16_output = expr.kind == LayoutExprKind::Decode;
    for (std::size_t j = 1; j < expr.inputs.size(); ++j) {
        (void)input_expr(algebra, expr, j, expr_index);
    }
    if (src.dtype == DType::FP8_E4M3) {
        validate_fp8_decode_decl(
            algebra, expr, expr_index, context, requires_bf16_output);
    } else if (src.dtype == DType::UINT8 &&
               input_expr(algebra, expr, 1, expr_index).decl.dtype ==
                   DType::UINT8) {
        validate_mxfp4_decode_decl(
            algebra, expr, expr_index, context, requires_bf16_output);
    } else if (src.quant.format == QuantFormat::AwqInt4) {
        require(
            expr.inputs.size() == 3,
            std::string(context) + " AWQ INT4 expects qweight/qzeros/scales at expr " +
                std::to_string(expr_index));
        if (requires_bf16_output) {
            require(
                expr.decl.dtype == DType::BF16,
                std::string(context) + " INT4 output must be BF16 at expr " +
                    std::to_string(expr_index));
        }
    } else if (src.quant.format == QuantFormat::GptqInt4) {
        require(
            expr.inputs.size() == 3 || expr.inputs.size() == 4,
            std::string(context) + " GPTQ INT4 expects qweight/qzeros/scales[/g_idx] at expr " +
                std::to_string(expr_index));
        if (requires_bf16_output) {
            require(
                expr.decl.dtype == DType::BF16,
                std::string(context) + " INT4 output must be BF16 at expr " +
                    std::to_string(expr_index));
        }
    } else {
        require(
            false,
            std::string(context) + " unsupported encoded source at expr " +
                std::to_string(expr_index));
    }
}

}  // namespace

void validate_layout_algebra(const LayoutAlgebra& algebra) {
    if (algebra.exprs.empty()) {
        require(
            algebra.bindings.empty(),
            "bindings cannot exist without expressions");
        return;
    }

    for (std::size_t i = 0; i < algebra.exprs.size(); ++i) {
        const LayoutExpr& expr = algebra.exprs[i];
        require_decl_basics(expr.decl, i);
        switch (expr.kind) {
        case LayoutExprKind::Source:
            require(expr.inputs.empty(), "Source has inputs at expr " +
                std::to_string(i));
            require(!expr.raw_name.empty(), "Source missing raw name at expr " +
                std::to_string(i));
            break;
        case LayoutExprKind::Select:
            require(expr.inputs.size() == 1, "Select expects one input at expr " +
                std::to_string(i));
            {
                const TensorDecl& input = input_expr(algebra, expr, 0, i).decl;
                require(
                    input.shape.size() == expr.decl.shape.size(),
                    "Select rank mismatch at expr " + std::to_string(i));
                validate_axis(expr, i);
                require(expr.length > 0, "Select has non-positive length at expr " +
                    std::to_string(i));
                require(
                    expr.start >= 0 &&
                        expr.start + expr.length <=
                            input.shape[static_cast<std::size_t>(expr.axis)],
                    "Select range out of input bounds at expr " +
                        std::to_string(i));
                TensorDecl inferred = input;
                inferred.name = expr.decl.name;
                inferred.shape[static_cast<std::size_t>(expr.axis)] =
                    expr.length;
                inferred.layout = expr.decl.layout;
                inferred.ownership = expr.decl.ownership;
                inferred.parallel = expr.decl.parallel;
                require_same_type(inferred, expr.decl, i, "Select");
            }
            break;
        case LayoutExprKind::Partition:
            require(expr.inputs.size() == 1, "Partition expects one input at expr " +
                std::to_string(i));
            {
                const TensorDecl& input = input_expr(algebra, expr, 0, i).decl;
                validate_axis(expr, i);
                require(expr.partitions > 0, "Partition missing partition count at expr " +
                    std::to_string(i));
                const auto axis = static_cast<std::size_t>(expr.axis);
                require(
                    input.shape[axis] % expr.partitions == 0,
                    "Partition axis is not divisible at expr " +
                        std::to_string(i));
                TensorDecl inferred = input;
                inferred.name = expr.decl.name;
                inferred.shape[axis] /= expr.partitions;
                inferred.layout = expr.decl.layout;
                inferred.ownership = expr.decl.ownership;
                inferred.parallel = expr.decl.parallel;
                require_same_type(inferred, expr.decl, i, "Partition");
            }
            break;
        case LayoutExprKind::Join: {
            validate_same_rank_join(algebra, expr, i);
            const TensorDecl& first = input_expr(algebra, expr, 0, i).decl;
            TensorDecl inferred = first;
            inferred.name = expr.decl.name;
            inferred.layout = expr.decl.layout;
            inferred.ownership = expr.decl.ownership;
            inferred.parallel = expr.decl.parallel;
            const auto axis = static_cast<std::size_t>(expr.axis);
            inferred.shape[axis] = 0;
            for (std::size_t j = 0; j < expr.inputs.size(); ++j) {
                const TensorDecl& input = input_expr(algebra, expr, j, i).decl;
                require(input.dtype == first.dtype, "Join dtype mismatch at expr " +
                    std::to_string(i));
                for (std::size_t dim = 0; dim < input.shape.size(); ++dim) {
                    if (dim == axis) continue;
                    require(
                        input.shape[dim] == first.shape[dim],
                        "Join non-axis shape mismatch at expr " +
                            std::to_string(i));
                }
                inferred.shape[axis] += input.shape[axis];
            }
            require_same_type(inferred, expr.decl, i, "Join");
            break;
        }
        case LayoutExprKind::Stack: {
            validate_same_rank_join(algebra, expr, i);
            if (expr.secondary_runtime_name.empty() && expr.axis >= 0) {
                const TensorDecl& first = input_expr(algebra, expr, 0, i).decl;
                TensorDecl inferred = first;
                inferred.name = expr.decl.name;
                inferred.shape.insert(
                    inferred.shape.begin() + expr.axis,
                    static_cast<std::int64_t>(expr.inputs.size()));
                inferred.layout = expr.decl.layout;
                inferred.ownership = expr.decl.ownership;
                inferred.parallel = expr.decl.parallel;
                if (inferred.shape == expr.decl.shape) {
                    require_same_type(inferred, expr.decl, i, "Stack");
                } else {
                    TensorDecl group_inferred = first;
                    group_inferred.name = expr.decl.name;
                    group_inferred.layout = expr.decl.layout;
                    group_inferred.ownership = expr.decl.ownership;
                    group_inferred.parallel = expr.decl.parallel;
                    require_same_type(group_inferred, expr.decl, i, "Stack");
                }
            } else {
                require(
                    expr.inputs.size() % 3 == 0,
                    "multi-output Stack expects expert triples at expr " +
                        std::to_string(i));
                require(
                    expr.decl.shape.size() == 3,
                    "multi-output Stack expects rank-3 output at expr " +
                        std::to_string(i));
                for (std::size_t j = 0; j < expr.inputs.size(); j += 3) {
                    const auto& gate = input_expr(algebra, expr, j, i).decl;
                    const auto& up = input_expr(algebra, expr, j + 1, i).decl;
                    const auto& down = input_expr(algebra, expr, j + 2, i).decl;
                    require(
                        gate.dtype == expr.decl.dtype &&
                            up.dtype == expr.decl.dtype &&
                            down.dtype == expr.decl.dtype,
                        "multi-output Stack dtype mismatch at expr " +
                            std::to_string(i));
                    require(
                        gate.shape == up.shape &&
                            gate.shape.size() == 2 &&
                            down.shape.size() == 2,
                        "multi-output Stack input shape mismatch at expr " +
                            std::to_string(i));
                }
            }
            break;
        }
        case LayoutExprKind::Unzip:
            require(expr.inputs.size() == 1, "Unzip expects one input at expr " +
                std::to_string(i));
            (void)input_expr(algebra, expr, 0, i);
            require(!expr.runtime_name.empty(), "Unzip missing primary output at expr " +
                std::to_string(i));
            require(!expr.secondary_runtime_name.empty(), "Unzip missing secondary output at expr " +
                std::to_string(i));
            break;
        case LayoutExprKind::View: {
            require(expr.inputs.size() == 1, "View expects one input at expr " +
                std::to_string(i));
            const TensorDecl& input = input_expr(algebra, expr, 0, i).decl;
            require(input.dtype == expr.decl.dtype, "View dtype mismatch at expr " +
                std::to_string(i));
            if (expr.axis >= 0) {
                require(
                    expr.axis < static_cast<int>(input.shape.size()) &&
                        input.shape.size() == expr.decl.shape.size(),
                    "View axis/rank mismatch at expr " + std::to_string(i));
                auto inferred_shape = input.shape;
                inferred_shape[static_cast<std::size_t>(expr.axis)] =
                    expr.length;
                require_same_shape(
                    inferred_shape, expr.decl.shape,
                    "View shape mismatch at expr " + std::to_string(i));
            }
            break;
        }
        case LayoutExprKind::Reorder:
            for (std::size_t j = 0; j < expr.inputs.size(); ++j) {
                (void)input_expr(algebra, expr, j, i);
            }
            break;
        case LayoutExprKind::Cast: {
            require(expr.inputs.size() == 1, "unary " + kind_name(expr.kind) +
                " expr expects one input at expr " + std::to_string(i) +
                " runtime='" + expr.runtime_name + "' raw='" + expr.raw_name + "'");
            TensorDecl inferred = input_expr(algebra, expr, 0, i).decl;
            inferred.name = expr.decl.name;
            inferred.dtype = expr.decl.dtype;
            inferred.layout = expr.decl.layout;
            inferred.ownership = expr.decl.ownership;
            inferred.parallel = expr.decl.parallel;
            require_same_type(inferred, expr.decl, i, "Cast");
            break;
        }
        case LayoutExprKind::Encode:
            require(expr.inputs.size() == 1, "unary " + kind_name(expr.kind) +
                " expr expects one input at expr " + std::to_string(i) +
                " runtime='" + expr.runtime_name + "' raw='" + expr.raw_name + "'");
            {
                TensorDecl inferred = input_expr(algebra, expr, 0, i).decl;
                inferred.name = expr.decl.name;
                inferred.dtype = expr.decl.dtype;
                inferred.layout = expr.decl.layout;
                inferred.ownership = expr.decl.ownership;
                inferred.parallel = expr.decl.parallel;
                inferred.quant = expr.decl.quant;
                require(
                    inferred.shape == expr.decl.shape,
                    "Encode shape mismatch at expr " + std::to_string(i));
                require(
                    expr.decl.quant.format != QuantFormat::None,
                    "Encode output missing quant encoding at expr " +
                        std::to_string(i));
            }
            break;
        case LayoutExprKind::Decode:
            validate_encoding_expr_decl(algebra, expr, i, "Decode");
            break;
        case LayoutExprKind::Transcode:
            validate_encoding_expr_decl(algebra, expr, i, "Transcode");
            require(
                expr.decl.quant.format != QuantFormat::None,
                "Transcode output missing quant encoding at expr " +
                    std::to_string(i));
            break;
        case LayoutExprKind::Attach:
            require(!expr.inputs.empty(), "metadata/encoding expr missing inputs at expr " +
                std::to_string(i));
            for (std::size_t j = 0; j < expr.inputs.size(); ++j) {
                (void)input_expr(algebra, expr, j, i);
            }
            if (expr.kind == LayoutExprKind::Attach) {
                TensorDecl inferred =
                    infer_unary_passthrough(algebra, expr, i);
                require_same_type(inferred, expr.decl, i, "Attach");
            }
            break;
        case LayoutExprKind::Release:
            require(!expr.inputs.empty(), "Release expr missing inputs at expr " +
                std::to_string(i));
            for (std::size_t j = 0; j < expr.inputs.size(); ++j) {
                (void)input_expr(algebra, expr, j, i);
            }
            break;
        case LayoutExprKind::Realize:
            require(expr.inputs.size() == 1, "Realize expects one input at expr " +
                std::to_string(i));
            {
                const LayoutExpr& input_expr_ref = input_expr(algebra, expr, 0, i);
                const bool secondary_output =
                    !input_expr_ref.secondary_runtime_name.empty() &&
                    input_expr_ref.secondary_runtime_name == expr.runtime_name;
                if (!secondary_output) {
                    const TensorDecl& input = input_expr_ref.decl;
                    require(
                        input.dtype == expr.decl.dtype &&
                            input.shape == expr.decl.shape,
                        "Realize input does not match target TensorDecl at expr " +
                            std::to_string(i));
                }
            }
            require(!expr.runtime_name.empty(), "Realize missing runtime name at expr " +
                std::to_string(i));
            break;
        }
    }

    for (const auto& binding : algebra.bindings) {
        require(
            binding.root < algebra.exprs.size(),
            "binding root out of range for '" + binding.runtime_name + "'");
        require(
            !binding.runtime_name.empty(),
            "binding with empty runtime name");
    }
}

}  // namespace pie_cuda_driver

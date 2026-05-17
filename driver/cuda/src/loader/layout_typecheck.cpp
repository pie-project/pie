#include "loader/layout_typecheck.hpp"

#include <stdexcept>
#include <string>

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
            (void)input_expr(algebra, expr, 0, i);
            validate_axis(expr, i);
            require(expr.length > 0, "Select has non-positive length at expr " +
                std::to_string(i));
            break;
        case LayoutExprKind::Partition:
            require(expr.inputs.size() == 1, "Partition expects one input at expr " +
                std::to_string(i));
            (void)input_expr(algebra, expr, 0, i);
            validate_axis(expr, i);
            break;
        case LayoutExprKind::Join:
        case LayoutExprKind::Stack:
            validate_same_rank_join(algebra, expr, i);
            break;
        case LayoutExprKind::Unzip:
            require(expr.inputs.size() == 1, "Unzip expects one input at expr " +
                std::to_string(i));
            (void)input_expr(algebra, expr, 0, i);
            require(!expr.runtime_name.empty(), "Unzip missing primary output at expr " +
                std::to_string(i));
            require(!expr.secondary_runtime_name.empty(), "Unzip missing secondary output at expr " +
                std::to_string(i));
            break;
        case LayoutExprKind::View:
        case LayoutExprKind::Reorder:
            for (std::size_t j = 0; j < expr.inputs.size(); ++j) {
                (void)input_expr(algebra, expr, j, i);
            }
            break;
        case LayoutExprKind::Cast:
        case LayoutExprKind::Encode:
            require(expr.inputs.size() == 1, "unary " + kind_name(expr.kind) +
                " expr expects one input at expr " + std::to_string(i) +
                " runtime='" + expr.runtime_name + "' raw='" + expr.raw_name + "'");
            (void)input_expr(algebra, expr, 0, i);
            break;
        case LayoutExprKind::Decode:
        case LayoutExprKind::Transcode:
        case LayoutExprKind::Attach:
            require(!expr.inputs.empty(), "metadata/encoding expr missing inputs at expr " +
                std::to_string(i));
            for (std::size_t j = 0; j < expr.inputs.size(); ++j) {
                (void)input_expr(algebra, expr, j, i);
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
            (void)input_expr(algebra, expr, 0, i);
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

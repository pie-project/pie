#include "loader/layout_optimizer.hpp"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "loader/layout_typecheck.hpp"

namespace pie_cuda_driver {

namespace {

LayoutOptimizerPassStats run_dead_binding_elimination(LayoutAlgebra& algebra) {
    LayoutOptimizerPassStats stats;
    stats.name = "dead-binding-elimination";
    stats.exprs_before = algebra.exprs.size();

    std::unordered_set<std::string> seen;
    std::vector<LayoutBinding> bindings;
    bindings.reserve(algebra.bindings.size());
    for (auto it = algebra.bindings.rbegin(); it != algebra.bindings.rend(); ++it) {
        if (seen.insert(it->runtime_name).second) {
            bindings.push_back(*it);
        }
    }
    std::reverse(bindings.begin(), bindings.end());
    algebra.bindings = std::move(bindings);

    stats.exprs_after = algebra.exprs.size();
    return stats;
}

LayoutExprId add_expr(LayoutAlgebra& algebra, LayoutExpr expr)
{
    const LayoutExprId id = algebra.exprs.size();
    algebra.exprs.push_back(std::move(expr));
    return id;
}

std::int64_t dim_at(
    const LayoutExpr& expr,
    int axis)
{
    if (axis < 0 || axis >= static_cast<int>(expr.decl.shape.size())) {
        return -1;
    }
    return expr.decl.shape[static_cast<std::size_t>(axis)];
}

LayoutExpr make_select_expr(
    const LayoutExpr& source,
    LayoutExprId source_id,
    int axis,
    std::int64_t start,
    std::int64_t length)
{
    LayoutExpr select;
    select.kind = LayoutExprKind::Select;
    select.inputs = {source_id};
    select.decl = source.decl;
    if (axis >= 0 && axis < static_cast<int>(select.decl.shape.size())) {
        select.decl.shape[static_cast<std::size_t>(axis)] = length;
    }
    select.axis = axis;
    select.start = start;
    select.length = length;
    select.dtype = select.decl.dtype;
    select.encoding = select.decl.quant;
    return select;
}

LayoutExprId push_select_through_join(
    LayoutAlgebra& dst,
    const LayoutExpr& select,
    const LayoutExpr& join,
    LayoutOptimizerPassStats& stats)
{
    const int axis = select.axis;
    if (axis < 0 || axis >= static_cast<int>(join.decl.shape.size()) ||
        select.length <= 0) {
        return add_expr(dst, select);
    }

    const std::int64_t select_begin = select.start;
    const std::int64_t select_end = select.start + select.length;
    std::int64_t cursor = 0;
    std::vector<LayoutExprId> pushed_inputs;
    pushed_inputs.reserve(join.inputs.size());

    for (const auto child_id : join.inputs) {
        const LayoutExpr& child = dst.exprs.at(child_id);
        const std::int64_t child_dim = dim_at(child, axis);
        if (child_dim < 0) return add_expr(dst, select);
        const std::int64_t child_begin = cursor;
        const std::int64_t child_end = cursor + child_dim;
        cursor = child_end;
        const std::int64_t overlap_begin =
            std::max(select_begin, child_begin);
        const std::int64_t overlap_end =
            std::min(select_end, child_end);
        if (overlap_begin >= overlap_end) continue;
        if (overlap_begin == child_begin && overlap_end == child_end) {
            pushed_inputs.push_back(child_id);
            continue;
        }
        pushed_inputs.push_back(add_expr(
            dst,
            make_select_expr(
                child,
                child_id,
                axis,
                overlap_begin - child_begin,
                overlap_end - overlap_begin)));
    }

    if (pushed_inputs.empty()) {
        return add_expr(dst, select);
    }
    if (pushed_inputs.size() == 1 &&
        dst.exprs[pushed_inputs.front()].decl.shape == select.decl.shape) {
        ++stats.rewrites;
        return pushed_inputs.front();
    }

    LayoutExpr replacement = join;
    replacement.inputs = std::move(pushed_inputs);
    replacement.decl = select.decl;
    replacement.axis = axis;
    replacement.dtype = select.decl.dtype;
    replacement.encoding = select.decl.quant;
    ++stats.rewrites;
    return add_expr(dst, std::move(replacement));
}

LayoutExprId sink_cast_through_variadic(
    LayoutAlgebra& dst,
    const LayoutExpr& cast,
    const LayoutExpr& input,
    LayoutOptimizerPassStats& stats)
{
    std::vector<LayoutExprId> cast_inputs;
    cast_inputs.reserve(input.inputs.size());
    for (const auto child_id : input.inputs) {
        const LayoutExpr& child = dst.exprs.at(child_id);
        LayoutExpr child_cast = cast;
        child_cast.inputs = {child_id};
        child_cast.decl = child.decl;
        child_cast.decl.dtype = cast.decl.dtype;
        child_cast.decl.quant = cast.decl.quant;
        child_cast.dtype = child_cast.decl.dtype;
        child_cast.encoding = child_cast.decl.quant;
        cast_inputs.push_back(add_expr(dst, std::move(child_cast)));
    }

    LayoutExpr replacement = input;
    replacement.inputs = std::move(cast_inputs);
    replacement.decl = cast.decl;
    replacement.dtype = cast.decl.dtype;
    replacement.encoding = cast.decl.quant;
    ++stats.rewrites;
    return add_expr(dst, std::move(replacement));
}

LayoutExprId push_select_through_decode(
    LayoutAlgebra& dst,
    const LayoutExpr& select,
    const LayoutExpr& decode,
    LayoutOptimizerPassStats& stats)
{
    if (decode.inputs.empty()) {
        return add_expr(dst, select);
    }
    const LayoutExprId input_id = decode.inputs.front();
    const LayoutExpr& input = dst.exprs.at(input_id);
    if (select.axis < 0 ||
        select.axis >= static_cast<int>(input.decl.shape.size()) ||
        select.start < 0 ||
        select.length <= 0 ||
        select.start + select.length > dim_at(input, select.axis)) {
        return add_expr(dst, select);
    }

    std::vector<LayoutExprId> selected_inputs;
    selected_inputs.reserve(decode.inputs.size());
    for (std::size_t i = 0; i < decode.inputs.size(); ++i) {
        const LayoutExprId child_id = decode.inputs[i];
        const LayoutExpr& child = dst.exprs.at(child_id);
        const bool axis_selectable =
            select.axis >= 0 &&
            select.axis < static_cast<int>(child.decl.shape.size()) &&
            select.start + select.length <= dim_at(child, select.axis);
        const bool side_tensor_tracks_axis =
            axis_selectable &&
            select.axis < static_cast<int>(decode.decl.shape.size()) &&
            child.decl.shape[static_cast<std::size_t>(select.axis)] ==
                decode.decl.shape[static_cast<std::size_t>(select.axis)];
        if (i == 0 || side_tensor_tracks_axis) {
            selected_inputs.push_back(add_expr(
                dst,
                make_select_expr(
                    child,
                    child_id,
                    select.axis,
                    select.start,
                    select.length)));
        } else {
            selected_inputs.push_back(child_id);
        }
    }
    LayoutExpr replacement = decode;
    replacement.inputs = std::move(selected_inputs);
    replacement.decl = select.decl;
    replacement.runtime_name = select.runtime_name;
    replacement.dtype = select.decl.dtype;
    replacement.encoding = select.decl.quant;
    ++stats.rewrites;
    return add_expr(dst, std::move(replacement));
}

LayoutExprId push_encode_through_select(
    LayoutAlgebra& dst,
    const LayoutExpr& encode,
    const LayoutExpr& select,
    LayoutOptimizerPassStats& stats)
{
    if (select.inputs.size() != 1) {
        return add_expr(dst, encode);
    }
    const LayoutExprId input_id = select.inputs.front();
    const LayoutExpr& input = dst.exprs.at(input_id);
    if (select.axis < 0 ||
        select.axis >= static_cast<int>(input.decl.shape.size()) ||
        select.start < 0 ||
        select.length <= 0 ||
        select.start + select.length > dim_at(input, select.axis)) {
        return add_expr(dst, encode);
    }

    LayoutExpr pushed_encode = encode;
    pushed_encode.inputs = {input_id};
    pushed_encode.decl = input.decl;
    pushed_encode.decl.name = input.decl.name + ".__encoded";
    pushed_encode.decl.dtype = encode.decl.dtype;
    pushed_encode.decl.quant = encode.decl.quant;
    pushed_encode.runtime_name.clear();
    pushed_encode.dtype = pushed_encode.decl.dtype;
    pushed_encode.encoding = pushed_encode.decl.quant;
    const LayoutExprId encoded_input =
        add_expr(dst, std::move(pushed_encode));

    LayoutExpr replacement = select;
    replacement.inputs = {encoded_input};
    replacement.decl = encode.decl;
    replacement.runtime_name = encode.runtime_name;
    replacement.dtype = encode.decl.dtype;
    replacement.encoding = encode.decl.quant;
    ++stats.rewrites;
    return add_expr(dst, std::move(replacement));
}

LayoutExprId fuse_encode_decode_to_transcode(
    LayoutAlgebra& dst,
    const LayoutExpr& encode,
    const LayoutExpr& decode,
    LayoutOptimizerPassStats& stats)
{
    LayoutExpr transcode = encode;
    transcode.kind = LayoutExprKind::Transcode;
    transcode.inputs = decode.inputs;
    transcode.decl = encode.decl;
    transcode.runtime_name = encode.runtime_name;
    transcode.dtype = encode.decl.dtype;
    transcode.encoding = encode.decl.quant;
    ++stats.rewrites;
    return add_expr(dst, std::move(transcode));
}

LayoutExprId cancel_partition_join(
    LayoutAlgebra& dst,
    const LayoutExpr& partition,
    const LayoutExpr& join,
    LayoutOptimizerPassStats& stats)
{
    if (partition.partitions <= 0 ||
        partition.partition_index < 0 ||
        partition.partition_index >= partition.partitions) {
        return add_expr(dst, partition);
    }
    const std::int64_t total = dim_at(join, partition.axis);
    if (total <= 0 || total % partition.partitions != 0) {
        return add_expr(dst, partition);
    }
    LayoutExpr select = partition;
    select.kind = LayoutExprKind::Select;
    select.start = (total / partition.partitions) * partition.partition_index;
    select.length = total / partition.partitions;
    const LayoutExprId replacement =
        push_select_through_join(dst, select, join, stats);
    if (replacement != std::numeric_limits<LayoutExprId>::max()) {
        ++stats.rewrites;
        return replacement;
    }
    return add_expr(dst, partition);
}

LayoutOptimizerPassStats run_algebra_rewrite_pass(LayoutAlgebra& algebra)
{
    LayoutOptimizerPassStats stats;
    stats.name = "algebra-normalization";
    stats.exprs_before = algebra.exprs.size();

    LayoutAlgebra rewritten;
    rewritten.exprs.reserve(algebra.exprs.size());
    std::vector<LayoutExprId> id_map(
        algebra.exprs.size(),
        std::numeric_limits<LayoutExprId>::max());

    for (std::size_t old_id = 0; old_id < algebra.exprs.size(); ++old_id) {
        LayoutExpr expr = algebra.exprs[old_id];
        for (auto& input : expr.inputs) {
            if (input >= id_map.size() ||
                id_map[input] == std::numeric_limits<LayoutExprId>::max()) {
                throw std::runtime_error(
                    "layout optimizer: non-topological algebra input");
            }
            input = id_map[input];
        }

        LayoutExprId new_id = std::numeric_limits<LayoutExprId>::max();
        if ((expr.kind == LayoutExprKind::Select ||
             expr.kind == LayoutExprKind::Partition) &&
            expr.inputs.size() == 1) {
            const LayoutExpr& input = rewritten.exprs[expr.inputs.front()];
            if (input.kind == LayoutExprKind::Join) {
                new_id = expr.kind == LayoutExprKind::Select
                    ? push_select_through_join(rewritten, expr, input, stats)
                    : cancel_partition_join(rewritten, expr, input, stats);
            } else if (expr.kind == LayoutExprKind::Select &&
                       input.kind == LayoutExprKind::Decode) {
                new_id = push_select_through_decode(
                    rewritten, expr, input, stats);
            }
        } else if (expr.kind == LayoutExprKind::Cast &&
                   expr.inputs.size() == 1) {
            const LayoutExpr& input = rewritten.exprs[expr.inputs.front()];
            if (input.kind == LayoutExprKind::Join ||
                input.kind == LayoutExprKind::Stack) {
                new_id = sink_cast_through_variadic(
                    rewritten, expr, input, stats);
            } else if (input.kind == LayoutExprKind::Decode) {
                LayoutExpr fused = input;
                fused.decl = expr.decl;
                fused.runtime_name = expr.runtime_name;
                fused.dtype = expr.decl.dtype;
                fused.encoding = expr.decl.quant;
                new_id = add_expr(rewritten, std::move(fused));
                ++stats.rewrites;
            }
        } else if (expr.kind == LayoutExprKind::Encode &&
                   expr.inputs.size() == 1) {
            const LayoutExpr& input = rewritten.exprs[expr.inputs.front()];
            if (input.kind == LayoutExprKind::Decode) {
                new_id = fuse_encode_decode_to_transcode(
                    rewritten, expr, input, stats);
            } else if (input.kind == LayoutExprKind::Select) {
                new_id = push_encode_through_select(
                    rewritten, expr, input, stats);
            }
        }
        if (new_id == std::numeric_limits<LayoutExprId>::max()) {
            new_id = add_expr(rewritten, std::move(expr));
        }
        id_map[old_id] = new_id;
    }

    rewritten.bindings = algebra.bindings;
    for (auto& binding : rewritten.bindings) {
        if (binding.root >= id_map.size() ||
            id_map[binding.root] == std::numeric_limits<LayoutExprId>::max()) {
            throw std::runtime_error(
                "layout optimizer: binding root was not rewritten");
        }
        binding.root = id_map[binding.root];
    }
    algebra = std::move(rewritten);
    stats.exprs_after = algebra.exprs.size();
    return stats;
}

}  // namespace

LayoutOptimizerResult optimize_layout_algebra(LayoutPlan& plan) {
    validate_layout_algebra(plan.algebra);

    LayoutOptimizerResult result;
    result.passes.push_back(run_algebra_rewrite_pass(plan.algebra));
    result.passes.push_back(run_dead_binding_elimination(plan.algebra));

    validate_layout_algebra(plan.algebra);
    return result;
}

}  // namespace pie_cuda_driver

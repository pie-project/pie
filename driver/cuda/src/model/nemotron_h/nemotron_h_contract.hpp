#pragma once

/// What Nemotron-H binds (`model/registry.cpp` row).
///
/// The Mamba2/attention/MoE hybrid keeps its decoder under
/// `language_model.backbone.`, and its MoE GEMM addresses all experts of a
/// layer as one contiguous slab. The contract declares the slab and then
/// declares each expert as a slice of it, so no byte is copied twice.

#include <cstdlib>
#include <string>
#include <vector>

#include "model/contract.hpp"

namespace pie_cuda_driver::model {

/// Whether this rank splits the Mamba mixer, rather than replicating it.
///
/// Read by the contract, by bind and by the forward, which have to agree: the
/// contract decides what shape the weights arrive in, and the forward's head
/// and group counts are derived from the same answer. The environment variable
/// is a kill switch for bisecting a numerical regression; because it changes
/// the contract, and the cache key is the contract's, flipping it re-plans
/// rather than reusing a cached plan for the other layout.
inline bool nemotron_h_tp_mamba_sharding_enabled(int tp_size) {
    if (tp_size <= 1) {
        return false;
    }
    const char* disabled = std::getenv("PIE_NEMOTRON_DISABLE_TP_MAMBA_SHARD");
    return disabled == nullptr || disabled[0] == '\0' || disabled[0] == '0';
}

namespace contract_detail {

/// This rank's share of a leading axis that is `units` equal blocks.
///
/// [`ContractBuilder::split`] divides an extent, and an extent cannot say what
/// it is a concatenation *of*: `[heads * head_dim, cols]` divides by a
/// `tp_size` that does not divide `heads`, and the split then cuts a head in
/// half. Reshaping the axis to `[units, block]` first puts the divisibility
/// question on the value that has to answer it, so the loader rejects an
/// indivisible world naming this tensor instead of loading a wrong one.
///
/// Both transmutes are pure renames, so the composition still compiles to one
/// contiguous run per band -- `a_head_boundary_shard_is_one_contiguous_run` in
/// `loader/tests/storage_compiler.rs` pins that.
///
/// `out_shape` is the shape the band is published as, with `-1` where the split
/// axis was, and `enc` is the encoding it keeps throughout -- neither transmute
/// changes the element type, only how the elements are grouped.
inline Node split_by_unit(ContractBuilder& b, Node rows, std::int64_t units, std::int64_t block,
                          std::vector<std::int64_t> out_shape, PieLoaderEncodingSpec enc) {
    const Node folded = b.contract().transmute(rows, {units, block}, enc);
    return b.contract().transmute(b.split(folded, 0), std::move(out_shape), enc);
}

/// Elements per row of `shape`, i.e. everything but the leading extent.
inline std::int64_t row_elems(const std::vector<std::int64_t>& shape) {
    std::int64_t elems = 1;
    for (std::size_t axis = 1; axis < shape.size(); ++axis) {
        elems *= shape[axis];
    }
    return elems;
}

/// `shape` with its leading extent replaced by `-1`.
inline std::vector<std::int64_t> rows_inferred(const std::vector<std::int64_t>& shape) {
    std::vector<std::int64_t> out = shape;
    out[0] = -1;
    return out;
}

/// Declare this rank's slice of one Mamba mixer.
///
/// The mixer's `in_proj` is five bands stacked on one axis -- `z` and `x` of
/// `heads * head_dim` rows, `B` and `C` of `groups * state`, and `dt` of
/// `heads` -- and TP splits each band by its own unit. So the local tensor is
/// the concatenation of five independently sharded bands, which is a sentence
/// the algebra can say and a row range cannot. `conv1d` is the same in three
/// bands; the per-head vectors split directly; `out_proj` splits by column
/// because it is the row-parallel half of the pair.
///
/// This used to run after the load: bind allocated a second copy of every
/// weight and `cudaMemcpy2D`'d the rank's rows into it. Every rank therefore
/// read, transferred and kept resident the *whole* mixer -- tensor parallelism
/// that saved no Mamba memory at all, since nothing freed the originals --
/// and the copies were invisible to the plan, so they were redone on every
/// load and had to stay out of the cache key by hand.
inline void nemotron_h_layer_mamba_tp(ContractBuilder& b, const std::string& mp,
                                      std::int64_t groups) {
    const SourceTensor* in_proj = b.find(mp + "in_proj.weight");
    const SourceTensor* conv_w = b.find(mp + "conv1d.weight");
    const SourceTensor* conv_b = b.find(mp + "conv1d.bias");
    const SourceTensor* a_log = b.find(mp + "A_log");
    const SourceTensor* d = b.find(mp + "D");
    const SourceTensor* dt_bias = b.find(mp + "dt_bias");
    const SourceTensor* norm_w = b.find(mp + "norm.weight");
    const SourceTensor* out_proj = b.find(mp + "out_proj.weight");
    if (in_proj == nullptr || conv_w == nullptr || conv_b == nullptr || a_log == nullptr ||
        d == nullptr || dt_bias == nullptr || norm_w == nullptr || out_proj == nullptr) {
        return;  // an attention or MoE layer; it has no mixer to split.
    }

    // Every extent below is read off a tensor. Only `groups` cannot be: see
    // `ModelFacts::mamba_groups`.
    const std::vector<std::int64_t> in_shape = shape_of(*in_proj);
    const std::vector<std::int64_t> conv_shape = shape_of(*conv_w);
    const std::vector<std::int64_t> out_shape = shape_of(*out_proj);
    if (in_shape.size() != 2 || conv_shape.size() < 2 || out_shape.size() != 2 ||
        shape_of(*a_log).size() != 1 || shape_of(*norm_w).size() != 1) {
        fail("nemotron_h: mamba mixer '" + mp + "' has an unexpected tensor rank");
    }
    const std::int64_t heads = shape_of(*a_log)[0];
    const std::int64_t intermediate = shape_of(*norm_w)[0];
    const std::int64_t hidden = in_shape[1];
    const std::int64_t conv_dim = conv_shape[0];
    const std::int64_t group_state = (conv_dim - intermediate) / 2;
    if (heads <= 0 || intermediate % heads != 0 || conv_dim <= intermediate ||
        (conv_dim - intermediate) % 2 != 0 || groups <= 0 || group_state % groups != 0 ||
        in_shape[0] != 2 * intermediate + 2 * group_state + heads ||
        out_shape[0] != hidden || out_shape[1] != intermediate) {
        fail("nemotron_h: mamba mixer '" + mp + "' does not match heads=" +
             std::to_string(heads) + " groups=" + std::to_string(groups));
    }
    const std::int64_t head_dim = intermediate / heads;
    const std::int64_t state = group_state / groups;

    // The five bands of `in_proj`, in the order the mixer reads them.
    const std::vector<std::int64_t> band_shape{-1, hidden};
    const std::vector<Node> bands{
        split_by_unit(b,
                      b.contract().slice(b.contract().src(std::string(in_proj->name)), 0, 0,
                                         intermediate),
                      heads, head_dim * hidden, band_shape, in_proj->encoding),
        split_by_unit(b,
                      b.contract().slice(b.contract().src(std::string(in_proj->name)), 0, intermediate,
                                         intermediate),
                      heads, head_dim * hidden, band_shape, in_proj->encoding),
        split_by_unit(b,
                      b.contract().slice(b.contract().src(std::string(in_proj->name)), 0, 2 * intermediate,
                                         group_state),
                      groups, state * hidden, band_shape, in_proj->encoding),
        split_by_unit(b,
                      b.contract().slice(b.contract().src(std::string(in_proj->name)), 0,
                                         2 * intermediate + group_state, group_state),
                      groups, state * hidden, band_shape, in_proj->encoding),
        split_by_unit(b,
                      b.contract().slice(b.contract().src(std::string(in_proj->name)), 0,
                                         2 * intermediate + 2 * group_state, heads),
                      heads, hidden, band_shape, in_proj->encoding),
    };
    const std::int64_t local_heads = b.local_extent(heads);
    const std::int64_t local_groups = b.local_extent(groups);
    const std::int64_t local_intermediate = local_heads * head_dim;
    const std::int64_t local_group_state = local_groups * state;
    b.define(b.output_name(in_proj->name), b.contract().concat(bands, 0), in_proj->encoding,
             std::vector<std::int64_t>{
                 2 * local_intermediate + 2 * local_group_state + local_heads, hidden});
    b.consume(in_proj->id);

    // `conv1d` carries the same bands minus `z` and `dt`, over whatever trailing
    // extents the checkpoint gave it (`[conv_dim, k]` or `[conv_dim, 1, k]`).
    for (const SourceTensor* raw : {conv_w, conv_b}) {
        const std::vector<std::int64_t> shape = shape_of(*raw);
        const std::int64_t cols = row_elems(shape);
        const std::vector<std::int64_t> published = rows_inferred(shape);
        const std::vector<Node> conv_bands{
            split_by_unit(b,
                          b.contract().slice(b.contract().src(std::string(raw->name)), 0, 0,
                                             intermediate),
                          heads, head_dim * cols, published, raw->encoding),
            split_by_unit(b,
                          b.contract().slice(b.contract().src(std::string(raw->name)), 0, intermediate,
                                             group_state),
                          groups, state * cols, published, raw->encoding),
            split_by_unit(b,
                          b.contract().slice(b.contract().src(std::string(raw->name)), 0,
                                             intermediate + group_state, group_state),
                          groups, state * cols, published, raw->encoding),
        };
        std::vector<std::int64_t> local_shape = shape;
        local_shape[0] = local_intermediate + 2 * local_group_state;
        b.define(b.output_name(raw->name), b.contract().concat(conv_bands, 0), raw->encoding,
                 std::move(local_shape));
        b.consume(raw->id);
    }

    // One entry per head, so the head axis is already the leading extent.
    for (const SourceTensor* raw : {a_log, d, dt_bias}) {
        b.define(b.output_name(raw->name), b.split(b.contract().src(std::string(raw->name)), 0), raw->encoding,
                 std::vector<std::int64_t>{local_heads});
        b.consume(raw->id);
    }
    b.define(b.output_name(norm_w->name),
             split_by_unit(b, b.contract().src(std::string(norm_w->name)), heads, head_dim,
                           std::vector<std::int64_t>{-1}, norm_w->encoding),
             norm_w->encoding, std::vector<std::int64_t>{local_intermediate});
    b.consume(norm_w->id);

    // `out_proj` is `[hidden, heads * head_dim]`: the reduction axis is the
    // columns, so the unit fold goes on axis 1 and the split with it.
    b.define(b.output_name(out_proj->name),
             b.contract().transmute(
                 b.split(b.contract().transmute(b.contract().src(std::string(out_proj->name)),
                                             {hidden, heads, head_dim}, out_proj->encoding),
                         1),
                 {hidden, -1}, out_proj->encoding),
             out_proj->encoding, std::vector<std::int64_t>{hidden, local_intermediate});
    b.consume(out_proj->id);
}

/// Split every Mamba mixer across the tensor-parallel world.
inline void nemotron_h_mamba_tp_shards(ContractBuilder& b) {
    if (!nemotron_h_tp_mamba_sharding_enabled(b.target().tp_size)) {
        return;
    }
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        nemotron_h_layer_mamba_tp(
            b,
            b.source_name("language_model.backbone.layers." + std::to_string(layer) + ".mixer."),
            static_cast<std::int64_t>(b.facts().mamba_groups));
    }
}

inline void nemotron_h_layer_packed_experts(ContractBuilder& b, const std::string& base,
                                         const std::vector<const SourceTensor*>& up,
                                         const std::vector<const SourceTensor*>& down) {
    if (up.empty() || down.empty()) {
        return;
    }
    const SourceTensor& first_up = *up.front();
    const SourceTensor& first_down = *down.front();
    if (first_up.shape.size() != 2 || first_down.shape.size() != 2 ||
        !is_raw(first_up.encoding, PieLoaderDType::BF16) ||
        !is_raw(first_down.encoding, PieLoaderDType::BF16)) {
        return;
    }
    const std::int64_t full_intermediate = first_up.shape[0];
    const std::int64_t hidden = first_up.shape[1];
    if (first_down.shape[0] != hidden || first_down.shape[1] != full_intermediate) {
        return;
    }
    for (const SourceTensor* raw : up) {
        if (raw->shape.size() != 2 || raw->shape[0] != full_intermediate ||
            raw->shape[1] != hidden || !same_encoding(raw->encoding, first_up.encoding)) {
            return;
        }
    }
    for (const SourceTensor* raw : down) {
        if (raw->shape.size() != 2 || raw->shape[0] != hidden ||
            raw->shape[1] != full_intermediate ||
            !same_encoding(raw->encoding, first_down.encoding)) {
            return;
        }
    }

    const std::int64_t local_intermediate = b.local_extent(full_intermediate);
    const std::int64_t expert_count = static_cast<std::int64_t>(up.size());

    // Each expert contributes its local row band; the pack is their
    // concatenation. The sharding is in the expression, not in a flag.
    const std::string up_name = base + ".up_proj.packed.weight";
    std::vector<Node> up_parts;
    up_parts.reserve(up.size());
    for (const SourceTensor* raw : up) {
        up_parts.push_back(b.split(b.contract().src(std::string(raw->name)), 0));
    }
    b.define(up_name, b.contract().concat(up_parts, 0), pie_loader::raw(PieLoaderDType::BF16),
           std::vector<std::int64_t>{expert_count * local_intermediate, hidden});

    std::vector<Node> down_parts;
    down_parts.reserve(down.size());
    for (const SourceTensor* raw : down) {
        down_parts.push_back(b.contract().src(std::string(raw->name)));
    }
    auto [down_expr, down_shape] =
        b.shard(b.contract().concat(down_parts, 0),
              {expert_count * hidden, full_intermediate}, std::uint8_t{1});
    const std::string down_name = base + ".down_proj.packed.weight";
    b.define(down_name, down_expr, pie_loader::raw(PieLoaderDType::BF16), std::move(down_shape));

    for (std::size_t expert = 0; expert < up.size(); ++expert) {
        const std::int64_t index = static_cast<std::int64_t>(expert);
        b.define(std::string(up[expert]->name),
               b.contract().slice(b.contract().out(up_name), 0, index * local_intermediate,
                                local_intermediate),
               pie_loader::raw(PieLoaderDType::BF16),
               std::vector<std::int64_t>{local_intermediate, hidden});
        b.consume(up[expert]->id);
    }
    for (std::size_t expert = 0; expert < down.size(); ++expert) {
        const std::int64_t index = static_cast<std::int64_t>(expert);
        b.define(std::string(down[expert]->name),
               b.contract().slice(b.contract().out(down_name), 0, index * hidden, hidden),
               pie_loader::raw(PieLoaderDType::BF16),
               std::vector<std::int64_t>{hidden, local_intermediate});
        b.consume(down[expert]->id);
    }
}

/// Publish each layer's experts as one packed slab plus per-expert views into
/// it, which is what the Nemotron-H MoE GEMM addresses.
inline void nemotron_h_packed_expert_views(ContractBuilder& b) {
    if (b.facts().num_experts == 0) {
        return;
    }
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string base =
            "language_model.backbone.layers." + std::to_string(layer) + ".mixer.experts";
        if (b.find(base + ".up_proj.packed.weight") != nullptr ||
            b.find(base + ".down_proj.packed.weight") != nullptr) {
            continue;
        }
        std::vector<const SourceTensor*> up;
        std::vector<const SourceTensor*> down;
        bool complete = true;
        for (std::uint32_t expert = 0; expert < b.facts().num_experts; ++expert) {
            const std::string tag = base + "." + std::to_string(expert) + ".";
            const SourceTensor* u = b.find(tag + "up_proj.weight");
            const SourceTensor* d = b.find(tag + "down_proj.weight");
            if (u == nullptr || d == nullptr) {
                complete = false;
                break;
            }
            up.push_back(u);
            down.push_back(d);
        }
        if (complete) {
            nemotron_h_layer_packed_experts(b, base, up, down);
        }
    }
}

}  // namespace contract_detail

inline void author_nemotron_h_contract(ContractBuilder& b) {
    contract_detail::nemotron_h_mamba_tp_shards(b);
    b.fused_moe_gate_up_tp_slices();
    contract_detail::nemotron_h_packed_expert_views(b);
    b.dense_fused_projection_joins();
    b.publish_remaining();
}
}  // namespace pie_cuda_driver::model

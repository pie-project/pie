#pragma once

// LinearStateCache — the recurrent/conv state store for qwen3.6's Gated-
// DeltaNet linear-attention layers (delta-owned, separate from the paged-KV
// cache). Unlike paged-KV, linear-attn state is *fixed-size per request* (it
// summarises the whole history into a constant-shape state), so it is indexed
// by request slot, not by token/page:
//
//   conv_state[linear_layer]      : [num_slots, conv_kernel, conv_dim]   (bf16)
//   recurrent_state[linear_layer] : [num_slots, v_heads, k_head_dim, v_head_dim] (fp32)
//
//   conv_dim = 2*k_heads*k_head_dim + v_heads*v_head_dim   (the in_proj qkv width)
//
// Like PagedKvCache, MLX arrays are immutable graph values, so writes do not
// mutate in place: `write_*` builds a scatter node over the active request
// slots and rebinds the stored buffer, ordering the next read after the write
// within the lazy graph. `linear_layer` is the *ordinal among linear layers*
// (0..n_linear_layers-1); the graph maps decoder-layer → linear ordinal, since
// qwen3.6 interleaves linear-attn and full-attn layers.

#include <stdexcept>
#include <string>
#include <vector>

#include "ops/tensor.hpp"

namespace pie_metal_driver {

struct LinearStateGeometry {
    int n_linear_layers = 0;  // number of Gated-DeltaNet layers
    int num_slots       = 0;  // max concurrent requests (KV/batch slots)
    int conv_kernel     = 0;  // depthwise causal conv width (linear_conv_kernel_dim, ~4)
    int conv_dim        = 0;  // 2*k_heads*k_head_dim + v_heads*v_head_dim
    int v_heads         = 0;  // linear_num_value_heads
    int k_head_dim      = 0;  // linear_key_head_dim
    int v_head_dim      = 0;  // linear_value_head_dim

    bool valid() const {
        return n_linear_layers > 0 && num_slots > 0 && conv_kernel > 0 &&
               conv_dim > 0 && v_heads > 0 && k_head_dim > 0 && v_head_dim > 0;
    }
};

class LinearStateCache final {
public:
    explicit LinearStateCache(const LinearStateGeometry& geo) : geo_(geo) {
        if (!geo_.valid()) {
            throw std::invalid_argument(
                "LinearStateCache: all geometry dimensions must be positive");
        }
        conv_.reserve(geo_.n_linear_layers);
        rstate_.reserve(geo_.n_linear_layers);
        for (int l = 0; l < geo_.n_linear_layers; ++l) {
            conv_.push_back(make_conv_buffer());
            rstate_.push_back(make_recurrent_buffer());
        }
    }

    // ── batched buffer views (whole-slot tensors) ───────────────────────────
    // Beta gathers the active rows itself: `take(conv_state(l), slot_ids, 0)`
    // → [R, conv_kernel, conv_dim]; likewise recurrent_state.
    const Tensor& conv_state(int linear_layer) const {
        check_layer(linear_layer);
        return conv_[linear_layer];
    }
    const Tensor& recurrent_state(int linear_layer) const {
        check_layer(linear_layer);
        return rstate_[linear_layer];
    }

    // Convenience gathers for the active request slots (i32/u32 [R]).
    Tensor gather_conv_state(int linear_layer, const Tensor& slot_ids) const {
        return mlx::core::take(conv_state(linear_layer),
                               mlx::core::astype(slot_ids, mlx::core::uint32), 0);
    }
    Tensor gather_recurrent_state(int linear_layer, const Tensor& slot_ids) const {
        return mlx::core::take(recurrent_state(linear_layer),
                               mlx::core::astype(slot_ids, mlx::core::uint32), 0);
    }

    // ── writeback (scatter the new per-request states at slot_ids) ───────────
    // new_conv:      [R, conv_kernel, conv_dim]
    // new_recurrent: [R, v_heads, k_head_dim, v_head_dim]
    void write_conv_state(int linear_layer, const Tensor& slot_ids,
                          const Tensor& new_conv) {
        check_layer(linear_layer);
        conv_[linear_layer] =
            scatter_slots(conv_[linear_layer], new_conv, slot_ids,
                          {geo_.conv_kernel, geo_.conv_dim});
    }
    void write_recurrent_state(int linear_layer, const Tensor& slot_ids,
                               const Tensor& new_recurrent) {
        check_layer(linear_layer);
        rstate_[linear_layer] =
            scatter_slots(rstate_[linear_layer], new_recurrent, slot_ids,
                          {geo_.v_heads, geo_.k_head_dim, geo_.v_head_dim});
    }

    // ── geometry / lifecycle ────────────────────────────────────────────────
    const LinearStateGeometry& geometry() const { return geo_; }
    int n_linear_layers() const { return geo_.n_linear_layers; }
    int num_slots()       const { return geo_.num_slots; }

    // Zero a single request's state across all linear layers (call on sequence
    // start / slot reuse — linear-attn state is not implicitly cleared like KV).
    void reset_slot(int slot) {
        if (slot < 0 || slot >= geo_.num_slots) {
            throw std::out_of_range("LinearStateCache: slot out of range");
        }
        namespace mx = mlx::core;
        const Tensor id = mx::array({slot}, {1}, mx::uint32);
        const Tensor zc = mx::zeros({1, geo_.conv_kernel, geo_.conv_dim}, mx::bfloat16);
        const Tensor zr = mx::zeros(
            {1, geo_.v_heads, geo_.k_head_dim, geo_.v_head_dim}, mx::float32);
        for (int l = 0; l < geo_.n_linear_layers; ++l) {
            write_conv_state(l, id, zc);
            write_recurrent_state(l, id, zr);
        }
    }

    // Zero every buffer (between independent batches / on full reset).
    void reset() {
        for (int l = 0; l < geo_.n_linear_layers; ++l) {
            conv_[l]   = make_conv_buffer();
            rstate_[l] = make_recurrent_buffer();
        }
    }

    void eval() {
        std::vector<Tensor> all;
        all.reserve(conv_.size() + rstate_.size());
        for (const auto& t : conv_)   all.push_back(t);
        for (const auto& t : rstate_) all.push_back(t);
        mlx::core::eval(std::move(all));
    }

private:
    Tensor make_conv_buffer() const {
        // conv state carries a short token history → bf16 is sufficient.
        return mlx::core::zeros(
            {geo_.num_slots, geo_.conv_kernel, geo_.conv_dim}, mlx::core::bfloat16);
    }
    Tensor make_recurrent_buffer() const {
        // recurrent state accumulates over the whole sequence → keep fp32.
        return mlx::core::zeros(
            {geo_.num_slots, geo_.v_heads, geo_.k_head_dim, geo_.v_head_dim},
            mlx::core::float32);
    }

    void check_layer(int l) const {
        if (l < 0 || l >= geo_.n_linear_layers) {
            throw std::out_of_range("LinearStateCache: linear layer " +
                                    std::to_string(l) + " out of range [0," +
                                    std::to_string(geo_.n_linear_layers) + ")");
        }
    }

    // Scatter `src` [R, tail...] into `buf` [num_slots, tail...] at rows
    // `slot_ids`, returning the rebound buffer. Mirrors PagedKvCache.
    Tensor scatter_slots(const Tensor& buf, const Tensor& src,
                         const Tensor& slot_ids,
                         const std::vector<int>& tail) const {
        namespace mx = mlx::core;
        const int R = src.shape(0);
        // updates carry a leading index axis + a size-1 placeholder for the
        // scattered axis: [R, 1, tail...].
        mx::Shape upd_shape;
        upd_shape.push_back(R);
        upd_shape.push_back(1);
        for (int d : tail) upd_shape.push_back(d);
        Tensor upd = mx::reshape(mx::astype(src, buf.dtype()), upd_shape);
        return mx::scatter(buf,
                           std::vector<Tensor>{mx::astype(slot_ids, mx::uint32)},
                           upd, std::vector<int>{0});
    }

    LinearStateGeometry geo_;
    std::vector<Tensor> conv_;    // [n_linear_layers] × [num_slots,conv_K,conv_dim]
    std::vector<Tensor> rstate_;  // [n_linear_layers] × [num_slots,V_h,K_d,V_d]
};

}  // namespace pie_metal_driver

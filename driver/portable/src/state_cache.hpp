#pragma once

// Recurrent-state cache for Qwen 3.5 / 3.6 gated-delta-rule (linear-
// attention) layers. Each linear-attention layer maintains, per request:
//
//   1. A SSM state matrix [head_v_dim, head_k_dim] per head, F32.
//   2. A causal-conv1d state [conv_dim, conv_kernel-1] (the trailing
//      kernel-1 columns of the conv input window), in activation dtype.
//
// Unlike KvCachePaged, state isn't paged — it's a fixed-size slot per
// request per layer that persists across compute_() calls within one
// generation. The driver allocates `n_slots` slots at startup; the
// runtime (or the test harness) assigns each context a slot index.
//
// Layout per layer (one backend tensor each):
//   ssm_states_[il]:  F32 [head_k_dim, head_v_dim, n_heads, n_slots]
//   conv_states_[il]: F32 [conv_kernel-1, conv_dim, n_slots]
//
// (F32 throughout — the python ref keeps SSM math in F32, and the conv
// state is small enough that the type conversion isn't worth bf16
// storage.)

#include <cstdint>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

namespace pie_portable_driver {

class StateCache {
public:
    // `linear_layer_indices` lists the layer indices that are linear-
    // attention. Other layers don't get state slots. `n_slots` is the
    // maximum number of concurrent contexts the cache can hold.
    StateCache(ggml_backend_t backend,
               std::int32_t n_slots,
               const std::vector<std::int32_t>& linear_layer_indices,
               std::int32_t n_heads,
               std::int32_t head_k_dim,
               std::int32_t head_v_dim,
               std::int32_t conv_dim,
               std::int32_t conv_kernel);
    ~StateCache();

    StateCache(const StateCache&) = delete;
    StateCache& operator=(const StateCache&) = delete;

    // Per-layer SSM state [head_k_dim, head_v_dim, n_heads, n_slots] F32.
    // Returns nullptr for non-linear layers.
    ggml_tensor* ssm(std::int32_t layer) const noexcept;

    // Per-layer conv state [conv_kernel-1, conv_dim, n_slots] F32.
    ggml_tensor* conv(std::int32_t layer) const noexcept;

    // Zero the state for a specific slot across every linear layer.
    // Used at the start of a fresh context (since the test harness
    // doesn't have explicit "begin context" RPC plumbing).
    void zero_slot(std::int32_t slot);

    // Copy one slot to another across every linear-attention layer.
    void copy_slot(std::int32_t src_slot, std::int32_t dst_slot);

    std::int32_t n_slots()     const noexcept { return n_slots_; }
    std::int32_t n_heads()     const noexcept { return n_heads_; }
    std::int32_t head_k_dim()  const noexcept { return head_k_dim_; }
    std::int32_t head_v_dim()  const noexcept { return head_v_dim_; }
    std::int32_t conv_dim()    const noexcept { return conv_dim_; }
    std::int32_t conv_kernel() const noexcept { return conv_kernel_; }
    std::size_t  buffer_size() const noexcept;

    // True iff layer `il` is a linear-attention layer (has state).
    bool is_linear_layer(std::int32_t il) const noexcept;

private:
    void allocate_();

    ggml_backend_t backend_;
    std::int32_t   n_slots_;
    std::int32_t   n_heads_;
    std::int32_t   head_k_dim_;
    std::int32_t   head_v_dim_;
    std::int32_t   conv_dim_;
    std::int32_t   conv_kernel_;

    std::vector<bool> is_linear_;          // size = n_layers (max layer_idx + 1)

    ggml_context*         ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    // One entry per layer index; nullptr for non-linear layers.
    std::vector<ggml_tensor*> ssm_states_;
    std::vector<ggml_tensor*> conv_states_;
};

}  // namespace pie_portable_driver

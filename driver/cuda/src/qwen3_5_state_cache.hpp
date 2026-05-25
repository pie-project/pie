#pragma once

// Per-(layer, rs_cache-slot) state slabs for Qwen3.5's linear-attention layers.
// Each linear-attention layer carries two persistent per-request tensors,
// indexed by a request slot id in [0, max_slots):
//
//   * conv_state     : [max_slots, K, conv_dim] bf16
//                      Trailing K-window of the in-projection output for
//                      each active request, oldest-first.
//
//   * recurrent_state: [max_slots, V_h, K_d, V_d] fp32
//                      Linear-attention running state per request. The
//                      recurrence accumulates in fp32 and keeps the persisted
//                      state fp32 for the current CUDA kernels.
//
// Slot assignment is owned by the runtime and arrives on each forward as
// runtime-managed rs_cache slot ids. The CUDA storage itself is "dumb": it just
// hands out raw pointers offset by the slot.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "device_buffer.hpp"

namespace pie_cuda_driver {

class Qwen3_5StateCache {
public:
    // Allocate state for `num_linear_layers` linear-attention layers,
    // each holding `max_slots` independent slabs.
    //
    //     conv_dim = 2 * (num_k_heads * head_k_dim) + (num_v_heads * head_v_dim)
    //     K        = conv_kernel_size
    //     V_h      = num_v_heads, K_d = head_k_dim, V_d = head_v_dim
    //
    // Single-slot allocation (`max_slots == 1`) is the legacy
    // bring-up layout — semantics for callers that only ever pass
    // slot=0 are unchanged.
    static Qwen3_5StateCache allocate(
        const std::vector<bool>& layer_is_linear,
        int conv_dim,
        int conv_kernel,
        int v_heads,
        int head_k_dim,
        int head_v_dim,
        int hidden_size = 0,
        int max_slots = 1);

    // Same slot/indexing contract as `allocate`, but recurrent slabs are
    // unconditionally bf16 (ignoring PIE_QWEN35_RS_STATE_DTYPE). Used by
    // Nemotron-H's Mamba2 cache, where state is defined in activation dtype
    // and fp32 storage is too large at serving request counts.
    static Qwen3_5StateCache allocate_bf16_recurrent(
        const std::vector<bool>& layer_is_linear,
        int conv_dim,
        int conv_kernel,
        int v_heads,
        int head_k_dim,
        int head_v_dim,
        int max_slots = 1);

    // Zero the state of every slot of every linear-attention layer.
    // Called at the start of each fresh prefill — no cross-prefill
    // state continuity in this driver's batching model.
    void reset(cudaStream_t stream = 0);

    // Zero a single slot across every linear-attention layer. Called
    // when the runtime marks a slot as reset for a fresh recurrent
    // replay/prefill.
    void reset_slot(int slot, cudaStream_t stream = 0);

    // Copy one state slot to another across every linear-attention layer.
    // Used by runtime-managed fork/snapshot paths.
    void copy_slot_d2d(int src_slot, int dst_slot, cudaStream_t stream = 0);

    // Copy only the linear-attention state slabs. MTP pending-hidden is
    // intentionally left alone; speculative verifier rollback can restore
    // recurrent/conv state after the accepted prefix while preserving the MTP
    // state that was rebuilt from the accepted tokens.
    void copy_linear_state_slot_d2d(
        int src_slot, int dst_slot, cudaStream_t stream = 0);

    // Per-(layer, slot) accessors. `slot` defaults to 0 to keep the
    // legacy single-request callsites compiling unchanged.
    void*  conv_state(int layer, int slot = 0);
    float* recurrent_state(int layer, int slot = 0);
    void*  recurrent_state_raw(int layer, int slot = 0);
    void*  mtp_pending_hidden(int slot = 0);

    // Strides in bytes / fp32 elements between consecutive slots.
    std::size_t conv_slot_stride_bytes() const noexcept {
        return static_cast<std::size_t>(conv_kernel_) * conv_dim_ *
               sizeof(std::uint16_t);
    }
    std::size_t recurrent_slot_stride_floats() const noexcept {
        return static_cast<std::size_t>(v_heads_) * head_k_dim_ * head_v_dim_;
    }
    std::size_t recurrent_slot_stride_bytes() const noexcept {
        return recurrent_slot_stride_floats() *
               (recurrent_state_bf16_ ? sizeof(std::uint16_t) : sizeof(float));
    }

    int num_layers() const noexcept { return static_cast<int>(layer_is_linear_.size()); }
    int max_slots()  const noexcept { return max_slots_; }
    bool is_linear(int layer) const noexcept { return layer_is_linear_[layer]; }

    int conv_dim()    const noexcept { return conv_dim_; }
    int conv_kernel() const noexcept { return conv_kernel_; }
    int v_heads()     const noexcept { return v_heads_; }
    int head_k_dim()  const noexcept { return head_k_dim_; }
    int head_v_dim()  const noexcept { return head_v_dim_; }
    int hidden_size() const noexcept { return hidden_size_; }
    bool recurrent_state_bf16() const noexcept {
        return recurrent_state_bf16_;
    }

    // Speculative verifier prefix snapshots. When enabled, small prefill
    // linear-attention kernels write state after prefix length p into
    // slot `spec_snapshot_base_slot() + p - 1`.
    void set_spec_snapshot_slots(int base_slot, int count) noexcept {
        spec_snapshot_base_slot_ = base_slot;
        spec_snapshot_count_ = count;
    }
    void clear_spec_snapshot_slots() noexcept {
        spec_snapshot_base_slot_ = -1;
        spec_snapshot_count_ = 0;
    }
    int spec_snapshot_base_slot() const noexcept {
        return spec_snapshot_base_slot_;
    }
    int spec_snapshot_count() const noexcept {
        return spec_snapshot_count_;
    }

    Qwen3_5StateCache() = default;

private:
    std::vector<bool> layer_is_linear_;
    // Mapping from transformer layer index to compact linear-attention layer
    // index. Full-attention layers carry -1.
    std::vector<int> linear_layer_index_;
    int num_linear_layers_ = 0;
    // Flat buffers laid out as [linear_layer, slot, state]. This preserves the
    // existing per-layer slot stride seen by kernels while allowing slot-wide
    // reset/copy across all linear layers via pitched 2D CUDA operations.
    DeviceBuffer<std::uint16_t> conv_states_;
    DeviceBuffer<float>         recurrent_states_;
    DeviceBuffer<std::uint16_t> recurrent_states_bf16_;
    DeviceBuffer<std::uint16_t>              mtp_pending_hidden_;
    int max_slots_   = 1;
    int conv_dim_    = 0;
    int conv_kernel_ = 0;
    int v_heads_     = 0;
    int head_k_dim_  = 0;
    int head_v_dim_  = 0;
    int hidden_size_  = 0;
    bool recurrent_state_bf16_ = false;
    int spec_snapshot_base_slot_ = -1;
    int spec_snapshot_count_ = 0;
};

}  // namespace pie_cuda_driver

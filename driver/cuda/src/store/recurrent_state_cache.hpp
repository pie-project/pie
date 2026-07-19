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

#include "../device_buffer.hpp"

namespace pie_cuda_driver {

class RecurrentStateCache {
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
    // Whether `allocate` stores recurrent state in bf16 (the default)
    // vs fp32, per the PIE_QWEN35_RS_STATE_DTYPE env. Exposed so the
    // memory planner and verbose logging size rs_cache slots with the
    // same dtype the cache will actually use (otherwise they assume
    // fp32 and over-reserve ~2x the recurrent footprint).
    static bool recurrent_state_bf16_default();

    static RecurrentStateCache allocate(
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
    static RecurrentStateCache allocate_bf16_recurrent(
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

    // Device-predicated reset used by fixed envelopes. A negative slot id is
    // an invalid row and leaves every recurrent-state surface unchanged.
    void reset_slots_if_fresh(
        const std::int32_t* slot_ids,
        const std::uint8_t* is_fresh,
        int request_count,
        cudaStream_t stream = 0);

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

    // Frozen-verify mode. When set, the GDN verify kernels walk the recurrent
    // state in registers to produce correct speculative draft outputs but
    // persist NOTHING, leaving each committed slot at its pre-verify value
    // (the implicit speculative snapshot). The executor then advances each
    // committed slot to the confirmed prefix via a batched repair forward over
    // [input | accepted]. The committed slot only ever reflects committed
    // tokens — no per-request snapshot buffer, no concurrency cap. Set by the
    // executor around the verify forward; cleared for the repair forward
    // (which writes state normally).
    void set_verify_frozen(bool frozen) noexcept { verify_frozen_ = frozen; }
    bool verify_frozen() const noexcept { return verify_frozen_; }

    // Per-linear-layer stash of the verify forward's normed hidden input
    // (ws.norm_x), captured during a frozen verify. The recurrent-only
    // commit-advance reuses it to re-run just the linear-attn block (conv +
    // recurrence) over the accepted tokens, skipping attention / MLP /
    // non-linear layers / lm_head — instead of re-running the whole backbone.
    // Sized [num_linear_layers, max_tokens, hidden_size] bf16.
    void configure_verify_hidden_stash(int max_tokens, int hidden_size);
    bool verify_hidden_stash_enabled() const noexcept {
        return verify_stash_max_tokens_ > 0 && verify_stash_hidden_ > 0;
    }
    int verify_stash_hidden() const noexcept { return verify_stash_hidden_; }
    int verify_stash_max_tokens() const noexcept {
        return verify_stash_max_tokens_;
    }
    // Base pointer of compact linear-layer `linear_idx`'s stash region
    // (the [max_tokens, hidden_size] bf16 slab). Null if not configured.
    void* verify_hidden_stash_layer(int linear_idx);

    // Persistent slot-indexed buffered-activation pool (Ph7 RS working-set
    // fold-from-buffer). Mirrors the verify stash's per-token activation layout
    // ([ mixed_qkv (conv_dim) | a (V_h) | b (V_h) ] bf16, token-major within
    // each region) but PERSISTENT and indexed by a buffered-slab `slot`, so a
    // forward's `rs-buffer-output` write (W10, write_state=false) can stash its
    // in-proj activations into a buffered slab and a later `fold-buffered(n)`
    // can gather + replay them into the folded recurrent_state — vs the verify
    // stash, which only holds the same-pass forward's own activations.
    // Sized [num_linear_layers, num_slots, page_tokens, hidden_size] bf16, with
    // hidden_size = conv_dim + 2*v_heads (== the verify stash width). Runtime
    // owns slot assignment (arena RsSlab object id = buffered-pool slot id),
    // exactly mirroring the recurrent_state's "dumb pointer offset by slot".
    void configure_rs_buffer_pool(int page_tokens, int hidden_size, int num_slots);
    bool rs_buffer_pool_enabled() const noexcept {
        return rs_buffer_page_tokens_ > 0 && rs_buffer_hidden_ > 0 &&
               rs_buffer_num_slots_ > 0;
    }
    int rs_buffer_page_tokens() const noexcept { return rs_buffer_page_tokens_; }
    int rs_buffer_hidden() const noexcept { return rs_buffer_hidden_; }
    int rs_buffer_num_slots() const noexcept { return rs_buffer_num_slots_; }
    // Base pointer of compact linear-layer `linear_idx`, buffered slab `slot`'s
    // [page_tokens, hidden_size] bf16 region. Null if not configured / OOB.
    void* rs_buffer_slab(int linear_idx, int slot);

    RecurrentStateCache() = default;

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
    bool verify_frozen_ = false;
    DeviceBuffer<std::uint16_t> verify_hidden_stash_;
    int verify_stash_max_tokens_ = 0;
    int verify_stash_hidden_ = 0;
    // Persistent slot-indexed buffered-activation pool (Ph7). See
    // configure_rs_buffer_pool. [num_linear_layers, num_slots, page_tokens,
    // hidden_size] bf16.
    DeviceBuffer<std::uint16_t> rs_buffer_pool_;
    int rs_buffer_page_tokens_ = 0;
    int rs_buffer_hidden_ = 0;
    int rs_buffer_num_slots_ = 0;
};

}  // namespace pie_cuda_driver

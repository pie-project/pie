#pragma once

// LRU-bounded mapping from runtime context_id (u64) → driver state-cache
// slot index (int). Used by the linear-attention path in qwen3_5 /
// qwen3_5_moe forwards: each request needs a private chunk of conv_state
// + recurrent_state, and the runtime guarantees a stable context_id
// across a single request's lifetime (prefill → all decodes → free).
//
// Eviction is LRU within the configured capacity (max_slots, set to the
// driver's max_batch_size). When acquire() reassigns a slot, it returns
// is_fresh=true so the caller can zero that slot's state via
// Qwen3_5StateCache::reset_slot before the layer body consumes it. A
// returning evicted request will, by the runtime's own invariants, fire
// a fresh prefill — so the zero-then-prefill path is always correct.
//
// Lives on rank 0 only; slot decisions are broadcast to TP followers as
// part of the per-fire payload.

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace pie_cuda_driver {

class SlotAllocator {
public:
    struct Acquired {
        int  slot;
        bool is_fresh;  // true = slot was just (re)assigned to this ctx
    };

    SlotAllocator() = default;

    // Capacity = max number of concurrent contexts (must be ≥ 1).
    explicit SlotAllocator(int max_slots);

    // Reset capacity. Drops all mappings. Used when switching engine.
    void reset(int max_slots);

    // Acquire (or reuse) a slot for `ctx_id`. Returns {slot, is_fresh}.
    // - Existing context: returns its current slot, is_fresh=false.
    //   Bumps it to the back of the LRU.
    // - New context with free slot: returns it, is_fresh=true.
    // - New context, no free slot: evicts the least-recently-used
    //   non-`in_use_this_fire` slot, returns it, is_fresh=true.
    //
    // Throws on `max_slots == 0` or "every slot already in use this
    // fire" (the caller should size capacity ≥ max_batch_size so this
    // never happens — but the throw makes the invariant violation
    // visible rather than silently corrupting state).
    Acquired acquire(std::uint64_t ctx_id);

    // Mark every "in_use_this_fire" flag false. Call once at the end of
    // each fire. Slots that were touched this fire become eligible for
    // eviction next fire (they remain mapped — eviction is by LRU
    // order, not by activity within a single fire).
    void end_of_fire();

    // Forget `ctx_id` — its slot becomes immediately reusable. Optional;
    // not required for correctness (LRU will eventually evict).
    void release(std::uint64_t ctx_id);

    int max_slots() const noexcept { return max_slots_; }
    int active_count() const noexcept {
        return static_cast<int>(by_ctx_.size());
    }

private:
    int max_slots_ = 0;
    // ctx_id → slot index in [0, max_slots_).
    std::unordered_map<std::uint64_t, int> by_ctx_;
    // Reverse: slot → ctx_id (or 0 if unmapped). 0 is reserved as
    // "unmapped" sentinel — runtime ContextId allocation begins at 1
    // (see runtime/src/context.rs).
    std::vector<std::uint64_t> by_slot_;
    // Slot ids ordered least-recently-used first, most-recently at back.
    // Length always == active mappings; missing slots are "free."
    std::vector<int> lru_order_;
    // Per-slot guard: cleared at end_of_fire(), set when acquire()
    // touches the slot. Prevents acquire() from evicting a slot we
    // already handed out earlier in the same fire.
    std::vector<bool> in_use_this_fire_;
    // Free-slot stack (slot ids not currently mapped to any context).
    std::vector<int> free_slots_;
};

}  // namespace pie_cuda_driver

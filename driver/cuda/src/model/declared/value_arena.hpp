#pragma once

// The SSA VALUE ARENA — where a traced value's buffer comes from.
//
// The trace is pure SSA: `rmsnorm(x: &Val) -> Val`, and every op carries
// its `inputs` / `outputs` as value ids. What it deliberately does NOT
// carry is a buffer, because choosing one is a BACKEND job. Both family
// executors did that job as convention — "the normed activation lives in
// `ws.norm_y`" here, "in `ws.norm_x`" there — and since an arm then has to
// know whose convention it is serving, the executor could not be one file.
// (Measured at the start of this merge: the SSA edges are read 1 and 0
// times across the two executors; workspace fields are read 82 and 110.)
//
// This is the other answer: a value gets a slot, the arm asks for the slot
// by value id, and which physical bytes those are stops being anyone's
// convention.
//
// SIZING is a three-case fold, because the ABI's extent vocabulary is
// exactly {Tokens, Requests, Const} and its dtypes {BF16, F32, I32}.
//
// CAPTURE SAFETY is why this is a BUMP arena over a block the workspace
// already owns, reset per fire: a decode body runs inside
// `cudaStreamBeginCapture`, so it must allocate nothing. Same plan, same
// op order, same asks — so the same value lands at the same address on
// every fire, which is what a replayed graph requires. Growth is not
// allowed here for the same reason; an overflow throws and names the
// value, rather than quietly handing back a pointer a captured graph
// would not recognise.
//
// MIGRATION: arms move onto the arena one dataflow island at a time. An
// arm that has not moved keeps its convention, and the two coexist —
// which is only sound while every producer and consumer of a given value
// have moved TOGETHER. That is why the islands are converted whole.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pie_forward.h"
#include "pie_forward/plan.hpp"

using pie_forward::PieForwardOp;
using pie_forward::PieForwardValue;
using pie_forward::PieForwardDim;
using pie_forward::PieForwardDimKind;
using pie_forward::PieForwardDType;

namespace pie_cuda_driver::model::declared {

// Bytes one traced value occupies at this fire's extents.
inline std::size_t value_bytes(const PieForwardValue& value, int n_fire,
                               int r_fire) {
    std::size_t elements = 1;
    for (std::uint32_t d = 0; d < value.rank; ++d) {
        const PieForwardDim& dim = value.dims[d];
        switch (dim.kind) {
        case PieForwardDimKind::Tokens:
            elements *= static_cast<std::size_t>(n_fire);
            break;
        case PieForwardDimKind::Requests:
            elements *= static_cast<std::size_t>(r_fire);
            break;
        case PieForwardDimKind::Const:
            elements *= static_cast<std::size_t>(dim.value);
            break;
        }
    }
    std::size_t width = 2;  // BF16
    switch (value.dtype) {
    case PieForwardDType::BF16: width = 2; break;
    case PieForwardDType::F32:  width = 4; break;
    case PieForwardDType::I32:  width = 4; break;
    }
    return elements * width;
}

class ValueArena {
   public:
    // `block` is workspace-owned memory, allocated once, outside capture.
    // The plan is walked ONCE here for last-use, which is what lets slots
    // be reused: the trace is layer-unrolled, so a 28-layer plan names 28
    // distinct "normed activation" values whose live ranges never overlap.
    void reset(void* block, std::size_t capacity,
               const pie_forward::ForwardPlan& plan, int n_fire, int r_fire) {
        block_ = static_cast<std::uint8_t*>(block);
        capacity_ = capacity;
        used_ = 0;
        n_fire_ = n_fire;
        r_fire_ = r_fire;
        op_ = 0;
        free_.clear();
        pinned_.clear();
        const std::size_t values = plan.value_count();
        slots_.assign(values, nullptr);
        sizes_.assign(values, 0);
        last_use_.assign(values, 0);
        const std::size_t ops = plan.op_count();
        for (std::size_t i = 0; i < ops; ++i) {
            const PieForwardOp& op = plan.op(i);
            for (const std::uint32_t id : plan.inputs(op)) {
                if (id < last_use_.size()) last_use_[id] = i;
            }
            for (const std::uint32_t id : plan.outputs(op)) {
                if (id < last_use_.size() && last_use_[id] < i) {
                    last_use_[id] = i;
                }
            }
        }
    }

    // Advances the walk. Slots whose value was last read BEFORE this op
    // return to the free list — the reuse that keeps a layer-unrolled
    // plan inside one buffer's worth of block.
    void begin_op(std::size_t index) {
        op_ = index;
        for (std::size_t id = 0; id < slots_.size(); ++id) {
            if (slots_[id] == nullptr || last_use_[id] >= index) continue;
            if (last_use_[id] == kNeverFreed) continue;  // pinned
            free_.push_back(Block{
                static_cast<std::size_t>(
                    static_cast<std::uint8_t*>(slots_[id]) - block_),
                sizes_[id]});
            slots_[id] = nullptr;
        }
    }

    // PIN a value to memory the arena does not own: the buffer some
    // machinery OUTSIDE the traced ops reaches by name — LoRA captures the
    // normed activation's pointer at fire setup, hook sites observe the
    // query buffer, the sampler reads the logits. Declared once per family
    // by a pass over the plan, so an ARM still just asks by value id and
    // stays family-blind; the convention lives in the pass, not in 82
    // scattered arm sites. A pinned value never allocates and never frees.
    void pin(std::uint32_t value_id, void* ptr) {
        if (value_id >= slots_.size() || ptr == nullptr) return;
        slots_[value_id] = ptr;
        pinned_.push_back(value_id);
        last_use_[value_id] = kNeverFreed;
    }

    // The slot for one value, allocated on first ask and stable until its
    // last reader has run. Deterministic in ask order (the op order), so a
    // value keeps its address across fires of the same plan.
    void* slot(std::uint32_t value_id, const PieForwardValue& value) {
        if (value_id >= slots_.size()) {
            throw std::runtime_error(
                "declared value arena: value id " +
                std::to_string(value_id) + " is outside the plan's table");
        }
        if (slots_[value_id] != nullptr) return slots_[value_id];
        const std::size_t bytes = value_bytes(value, n_fire_, r_fire_);
        for (std::size_t f = 0; f < free_.size(); ++f) {
            if (free_[f].size < bytes) continue;
            slots_[value_id] = block_ + free_[f].offset;
            sizes_[value_id] = free_[f].size;
            free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(f));
            return slots_[value_id];
        }
        constexpr std::size_t kAlign = 256;
        const std::size_t at = (used_ + kAlign - 1) / kAlign * kAlign;
        if (block_ == nullptr || at + bytes > capacity_) {
            throw std::runtime_error(
                "declared value arena: value " + std::to_string(value_id) +
                " needs " + std::to_string(bytes) + " bytes at offset " +
                std::to_string(at) + ", past the workspace block's " +
                std::to_string(capacity_) +
                " — the arena may not grow inside a captured body");
        }
        used_ = at + bytes;
        slots_[value_id] = block_ + at;
        sizes_[value_id] = bytes;
        return slots_[value_id];
    }

    // Peak bytes handed out — what the workspace block must hold.
    std::size_t used() const { return used_; }

   private:
    static constexpr std::size_t kNeverFreed =
        static_cast<std::size_t>(-1);

    struct Block {
        std::size_t offset;
        std::size_t size;
    };

    std::uint8_t* block_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t used_ = 0;
    int n_fire_ = 0;
    int r_fire_ = 0;
    std::size_t op_ = 0;
    std::vector<void*> slots_;
    std::vector<std::size_t> sizes_;
    std::vector<std::size_t> last_use_;
    std::vector<Block> free_;
    std::vector<std::uint32_t> pinned_;
};

}  // namespace pie_cuda_driver::model::declared

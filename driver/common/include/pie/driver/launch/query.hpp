#pragma once

// Pure queries over an adopted `launch::Trace`. Nothing here touches device
// memory, channel state, or any backend type — these are structural walks of
// the launch package, so both drivers must answer them identically or they
// disagree about what the same program means. They lived as hand-copies in
// driver/{cuda,metal}/src/pipeline/descriptor_resolve.hpp until the copies
// started to drift stylistically; a shared definition is the only thing that
// can keep them from drifting behaviourally next.

#include <cstddef>

#include "pie/driver/fire/geometry.hpp"
#include "pie/driver/launch/program.hpp"

namespace pie::driver::launch {

// The op that defines `value`, or null if it is a root (const / intrinsic /
// channel take-read). Ops define a contiguous run of `result_count` ids
// starting at `result_id` (top_k defines two: value then index).
inline const Op* producer(const Trace& trace, ValueId value) {
    for (const Stage& stage : trace.stages) {
        for (const Op& op : stage.ops) {
            if (value >= op.result_id && value < op.result_id + op.result_count) {
                return &op;
            }
        }
    }
    return nullptr;
}

// Classify the mask a program puts on `mask_channel` by walking back from the
// last put to its defining op, seeing through reshapes. A structured verdict
// lets the forward use the attention kernel's own mask instead of materializing
// one; anything unrecognized returns an empty descriptor, which is the safe
// answer (the caller falls back to the dense path).
inline fire::StructuredMaskDescriptor structured_mask_descriptor(const Trace& trace,
                                                           ChannelId mask_channel) {
    const ChannelPut* selected = nullptr;
    for (const Stage& stage : trace.stages) {
        for (const ChannelPut& put : stage.puts) {
            if (put.channel == mask_channel) selected = &put;
        }
    }
    if (selected == nullptr) return {};

    ValueId value = selected->value;
    // Bounded by the value count: a reshape chain cannot be longer than the
    // number of values without being a cycle, which adoption already rejects.
    for (std::size_t depth = 0; depth <= trace.values.size(); ++depth) {
        const Op* op = producer(trace, value);
        if (op == nullptr) break;
        if (op->code == OpCode::Reshape && !op->args.empty()) {
            value = op->args[0];
            continue;
        }
        fire::StructuredMaskDescriptor descriptor;
        switch (op->code) {
            case OpCode::CausalMask:
                descriptor.kind = fire::StructuredMaskKind::Causal;
                descriptor.key_len = op->imm;
                return descriptor;
            case OpCode::SlidingWindowMask:
                descriptor.kind = fire::StructuredMaskKind::SlidingWindow;
                descriptor.key_len = op->imm;
                descriptor.window = op->imm2;
                return descriptor;
            case OpCode::SinkWindowMask:
                descriptor.kind = fire::StructuredMaskKind::SinkWindow;
                descriptor.key_len = op->imm;
                descriptor.sink = op->imm2;
                descriptor.window = op->imm3;
                return descriptor;
            default:
                return {};
        }
    }
    return {};
}

}  // namespace pie::driver::launch

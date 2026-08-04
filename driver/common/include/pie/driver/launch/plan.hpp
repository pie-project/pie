#pragma once

// The per-stage **launch plans** — the compiled, region-partitioned form of a
// stage, adopted from `PieLaunchStagePlan`.
//
// `plans` is parallel to `program.hpp`'s `Trace::stages`: plan `i` is the
// compiled form of stage `i`. The host owns every decision recorded here --
// normalization, region partitioning, symbolic extents, graph identity, and the
// grouped-launch envelope -- so `adopt()` is a field copy.

#include <cstdint>
#include <string>
#include <vector>

#include <pie_driver_abi.h>

#include "pie/driver/launch/program.hpp"

namespace pie::driver::launch::plan {

// One dimension of a value's type: either a trace-known extent or a symbolic
// one (`PTIR_EXTENT_*`) resolved at fire time.
struct Dimension {
    bool symbolic = false;
    std::uint32_t value = 0;
};

struct ValueType {
    std::uint8_t dtype = 0;
    std::vector<Dimension> dims;
    std::uint8_t domain = 0;
};

// One op in a normalized stage body. Field names match the op's flat encoding:
// only the fields an op's tag defines are meaningful.
struct PlanOp {
    std::uint8_t  tag = 0;
    std::int64_t  chan = -1;        // chan_take/read/put target, else -1
    std::uint16_t name_idx = 0;     // kernel_call / sink_call name table index
    std::vector<std::uint32_t> args;
    std::uint32_t results = 1;      // SSA ids defined
    std::uint8_t  lit_dtype = 0;    // const literal dtype
    std::uint32_t lit_bits = 0;     // const literal raw bits / cast dtype target
    std::uint16_t intr = 0;         // intrinsic_val id
    std::uint8_t  dtype = 0;        // intrinsic_val / cast / rng element dtype
    Shape         shape;            // broadcast/reshape/rng/intrinsic target shape
    std::uint32_t imm = 0;          // top_k k / iota len / rng stream
    std::uint32_t imm2 = 0;
    std::uint32_t imm3 = 0;
    std::uint8_t  kind = 0;         // rng kind (0 uniform, 1 gumbel)
    std::uint8_t  pred_tag = 0;     // pivot_threshold predicate tag
    std::uint32_t pred_payload = 0; // pivot_threshold predicate payload (value id)
};

// A normalized op plus the pre-normalization op indices it came from, which is
// how a region maps back onto the stage body.
struct NormalizedOp {
    PlanOp op;
    std::vector<std::uint32_t> source_ops;
};

struct ChannelSink {
    std::uint32_t channel_slot = 0;
    std::uint32_t value = 0;
};

struct Region {
    bool library = false;
    std::uint8_t library_op = 0;
    std::uint8_t schedule = 0;
    std::vector<std::uint32_t> nodes;
    // Library inputs are role ordered. Nucleus is [logits, top_p, rng_state],
    // independent of numeric value-id order.
    std::vector<std::uint32_t> inputs;
    std::vector<std::uint32_t> outputs;
    std::vector<ChannelSink> sinks;
};

struct Partition {
    std::uint8_t kind = 0;
    bool whole_stage_fallback = false;
    std::vector<Region> regions;
};

// The grouped-launch envelope: which symbolic extents a stage's shapes use, how
// its channel ops bind, and what the fire needs available. Derived on the host
// so a driver only has to read it.
struct GroupedEnvelope {
    bool valid = false;
    std::uint32_t flags = 0;
    std::uint32_t mtp_rows = 0;
    std::vector<std::uint8_t> used_extents;
    // Parallel to the stage's channel ops, in body order.
    std::vector<PieLaunchChannelRule> channel_rules;
    // Why the stage cannot be grouped, when `valid` is false. The fused path
    // still reads `flags`.
    std::string error;

    bool requires_query() const { return (flags & PIE_STAGE_REQUIRES_QUERY) != 0; }
    bool requires_layer() const { return (flags & PIE_STAGE_REQUIRES_LAYER) != 0; }
    bool requires_attn_score() const {
        return (flags & PIE_STAGE_REQUIRES_ATTN_SCORE) != 0;
    }
    bool requires_kernel_call() const {
        return (flags & PIE_STAGE_REQUIRES_KERNEL_CALL) != 0;
    }
    bool requires_page_mask() const {
        return (flags & PIE_STAGE_REQUIRES_PAGE_MASK) != 0;
    }
    bool requires_lora() const {
        return (flags & PIE_STAGE_REQUIRES_LORA) != 0;
    }
    bool requires_mtp_rows() const {
        return (flags & PIE_STAGE_REQUIRES_MTP_ROWS) != 0;
    }
};

struct StagePlan {
    std::uint8_t stage = 0;
    std::uint64_t signature_hash = 0;
    // The graph-cache identity for this stage. A wrong key silently reuses
    // another program's CUDA graph, so it is shipped rather than re-derived.
    std::uint64_t identity = 0;
    std::vector<std::uint32_t> channel_bindings;
    std::vector<std::string> names;
    std::vector<NormalizedOp> ops;
    std::vector<ValueType> value_types;
    Partition singleton;
    Partition fused;
    GroupedEnvelope grouped;
};

namespace detail {

inline Shape shape_from(const PieU32Slice& dims) {
    Shape shape;
    shape.dims.assign(dims.ptr, dims.ptr + dims.len);
    return shape;
}

inline PlanOp op_from(const PieLaunchOp& raw) {
    PlanOp op;
    op.tag = static_cast<std::uint8_t>(raw.code);
    op.chan = raw.channel == PIE_NO_CHANNEL ? -1
                                            : static_cast<std::int64_t>(raw.channel);
    op.name_idx = static_cast<std::uint16_t>(raw.name_index);
    op.args.assign(raw.args.ptr, raw.args.ptr + raw.args.len);
    op.results = raw.result_count;
    op.lit_dtype = raw.lit_dtype;
    op.lit_bits = raw.lit_bits;
    op.intr = raw.intrinsic;
    op.dtype = raw.dtype;
    op.shape = shape_from(raw.shape);
    op.imm = raw.imm;
    op.imm2 = raw.imm2;
    op.imm3 = raw.imm3;
    op.kind = raw.rng_kind;
    op.pred_tag = raw.pred_tag;
    op.pred_payload = raw.pred_payload;
    return op;
}

inline Partition partition_from(const PieLaunchRegionSlice& regions) {
    Partition partition;
    partition.regions.reserve(regions.len);
    for (std::size_t i = 0; i < regions.len; ++i) {
        const PieLaunchRegion& src = regions.ptr[i];
        Region region;
        region.library = src.kind == PIE_REGION_LIBRARY;
        region.library_op = src.library;
        region.schedule = src.schedule;
        region.nodes.assign(src.nodes.ptr, src.nodes.ptr + src.nodes.len);
        region.inputs.assign(src.inputs.ptr, src.inputs.ptr + src.inputs.len);
        region.outputs.assign(src.outputs.ptr, src.outputs.ptr + src.outputs.len);
        region.sinks.reserve(src.sinks.len);
        for (std::size_t s = 0; s < src.sinks.len; ++s) {
            region.sinks.push_back(
                ChannelSink{src.sinks.ptr[s].channel, src.sinks.ptr[s].value});
        }
        partition.regions.push_back(std::move(region));
    }
    return partition;
}

}  // namespace detail

inline StagePlan adopt(std::uint8_t stage, const PieLaunchStagePlan& src) {
    StagePlan plan;
    plan.stage = stage;
    plan.signature_hash = src.signature_hash;
    plan.identity = src.identity;
    plan.channel_bindings.assign(
        src.channel_bindings.ptr,
        src.channel_bindings.ptr + src.channel_bindings.len);

    plan.names.reserve(src.names.len);
    for (std::size_t i = 0; i < src.names.len; ++i) {
        const PieBytes& name = src.names.ptr[i];
        plan.names.emplace_back(reinterpret_cast<const char*>(name.ptr), name.len);
    }

    // `source_ops` is the concatenation of every op's source list, cut by
    // `source_op_counts`.
    plan.ops.reserve(src.ops.len);
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < src.ops.len; ++i) {
        NormalizedOp op;
        op.op = detail::op_from(src.ops.ptr[i]);
        const std::size_t count = src.source_op_counts.ptr[i];
        op.source_ops.assign(src.source_ops.ptr + cursor,
                             src.source_ops.ptr + cursor + count);
        cursor += count;
        plan.ops.push_back(std::move(op));
    }

    plan.value_types.reserve(src.value_types.len);
    for (std::size_t i = 0; i < src.value_types.len; ++i) {
        const PieLaunchPlanValue& raw = src.value_types.ptr[i];
        ValueType type;
        type.dtype = raw.dtype;
        type.dims.reserve(raw.dims.len);
        for (std::size_t d = 0; d < raw.dims.len; ++d) {
            const std::uint8_t extent = raw.extents.ptr[d];
            type.dims.push_back(extent == PIE_EXTENT_STATIC
                                    ? Dimension{false, raw.dims.ptr[d]}
                                    : Dimension{true, extent});
        }
        plan.value_types.push_back(std::move(type));
    }

    plan.singleton = detail::partition_from(src.singleton);
    plan.fused = detail::partition_from(src.fused);

    plan.grouped.valid = (src.flags & PIE_STAGE_GROUPED_VALID) != 0;
    plan.grouped.flags = src.flags;
    plan.grouped.mtp_rows = src.mtp_rows;
    plan.grouped.used_extents.assign(
        src.used_extents.ptr, src.used_extents.ptr + src.used_extents.len);
    plan.grouped.channel_rules.assign(
        src.channel_rules.ptr, src.channel_rules.ptr + src.channel_rules.len);
    plan.grouped.error =
        std::string(reinterpret_cast<const char*>(src.error.ptr), src.error.len);

    return plan;
}

}  // namespace pie::driver::launch::plan

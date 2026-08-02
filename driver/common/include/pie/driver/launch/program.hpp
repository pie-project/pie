#pragma once

// The **launch package**, adopted into the driver's execution model.
//
// This is what a driver receives instead of PTIR. The host compiles the
// program and ships `PieLaunchPackage` — flat typed POD records — so there is
// no wire format to parse, no plan to re-derive, and no structural validation
// to duplicate. `adopt()` below is a field copy, not a decode: it can fail only
// if the host sent something the ABI validator already rejected.
//
// The structs are the driver's own execution state (the same status as
// `fire/descriptor.hpp` and `fire/fire_geometry.hpp`), which is why they live
// under `driver/common/` rather than being shared with the compiler.
//
// This is a value -> value graph over trace-known shapes; nothing here touches
// memory. The only stateful construct is the Channel: a bounded queue of cells
// with full/empty bits.

#include <cstdint>
#include <string>
#include <vector>

#include <pie_driver_abi.h>

#include "pie/driver/launch/op_table.hpp"

namespace pie::driver::launch {

using ValueId   = std::uint32_t;   // SSA value id, unique within a program
using ChannelId = std::uint32_t;   // channel index within a program
using ProgramId = std::uint32_t;

// Trace-known logical shape: an ordered dim list (row-major). Empty = scalar.
// Ranks go up to [B, P, page]; tier-0 treats the leading dim as the row (CTA)
// axis and the trailing dims flattened as the per-row length.
struct Shape {
    std::vector<std::uint32_t> dims;

    bool is_scalar() const { return dims.empty(); }
    std::uint32_t rank() const { return static_cast<std::uint32_t>(dims.size()); }

    // Total element count (1 for scalar).
    std::uint64_t numel() const {
        std::uint64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    // Row count = leading dim (1 for scalar/vector-as-one-row is caller policy;
    // here a 1-D shape is a single row of `dims[0]`, matching row-local reduce).
    std::uint32_t rows() const { return dims.size() >= 2 ? dims[0] : 1; }
    // Per-row length = product of the trailing dims (the last/reduce axis span).
    std::uint32_t row_len() const {
        if (dims.empty()) return 1;
        std::uint64_t n = 1;
        for (std::size_t i = (dims.size() >= 2 ? 1 : 0); i < dims.size(); ++i) n *= dims[i];
        return static_cast<std::uint32_t>(n);
    }

    static Shape scalar() { return {}; }
    static Shape vec(std::uint32_t len) { return {{len}}; }
    static Shape mat(std::uint32_t rows, std::uint32_t len) { return {{rows, len}}; }
};

struct TensorType {
    Shape shape;
    DType dtype = DType::F32;
};

enum class PredTag : std::uint8_t { RankLe = 0, CummassLe = 1, ProbGe = 2 };

// pivot_threshold predicate. The payload is ALWAYS a ValueId (never an
// immediate) for all three tags -- RankLe's `k` is a U32 scalar/per-row value
// id, CummassLe/ProbGe's threshold is an F32 scalar/per-row value id. The host
// remaps it to a program-global id like any other operand.
struct Predicate {
    PredTag       tag = PredTag::RankLe;
    std::uint32_t payload = 0;
};

enum class RngKind : std::uint8_t { Uniform = 0, Gumbel = 1 };

// A value's producer. Constants and intrinsics are trace-time roots; channel
// reads are the stateful roots; op results are interior nodes.
enum class ValueSource : std::uint8_t {
    Const       = PIE_VALUE_CONST,         // trace-known literal
    Intrinsic   = PIE_VALUE_INTRINSIC,     // stage-scoped intrinsic (logits/hidden/…)
    ChannelTake = PIE_VALUE_CHANNEL_TAKE,  // consumes a full cell, marks it empty
    ChannelRead = PIE_VALUE_CHANNEL_READ,  // peeks a full cell without consuming
    OpResult    = PIE_VALUE_OP_RESULT,     // produced by an op in this stage's DAG
};

enum class Intrinsic : std::uint8_t {
    Logits = PTIR_INTR_LOGITS,
    MtpLogits = PTIR_INTR_MTP_LOGITS,
    Hidden = PTIR_INTR_HIDDEN,
    Query = PTIR_INTR_QUERY,
    ValueHead = PTIR_INTR_VALUE_HEAD,
    Layer = PTIR_INTR_LAYER,
    MtpDrafts = PTIR_INTR_MTP_DRAFTS,
    AttnScore = PTIR_INTR_ATTN_SCORE,
};

// A literal payload (dtype + 4 raw bytes, interpreted per dtype).
struct Literal {
    DType dtype = DType::F32;
    std::uint32_t bits = 0;

    float as_f32() const { float f; __builtin_memcpy(&f, &bits, 4); return f; }
    std::int32_t as_i32() const { return static_cast<std::int32_t>(bits); }
    std::uint32_t as_u32() const { return bits; }
    bool as_bool() const { return bits != 0; }

    static Literal f32(float v) { Literal l; l.dtype = DType::F32; __builtin_memcpy(&l.bits, &v, 4); return l; }
    static Literal i32(std::int32_t v) { Literal l; l.dtype = DType::I32; l.bits = static_cast<std::uint32_t>(v); return l; }
    static Literal u32(std::uint32_t v) { Literal l; l.dtype = DType::U32; l.bits = v; return l; }
};

// A declared SSA value: its type + where it comes from. Interior op results have
// source == OpResult and are defined by the Op that lists them in `result_id`.
struct Value {
    ValueId     id = 0;
    TensorType  type;
    ValueSource source = ValueSource::OpResult;

    // Source-specific payload:
    Literal          lit;                                   // Const
    Intrinsic        intrinsic = Intrinsic::Logits;         // Intrinsic
    ChannelId        channel = 0;                           // ChannelTake / ChannelRead
};

// One op in a stage DAG. Operand fields carry value ids; `result_id` is the SSA
// value this op defines (top_k defines two: value then index). Op-specific
// immediates ride alongside.
struct Op {
    OpCode  code = OpCode::Add;

    std::vector<ValueId> args;      // operand value ids, in op-defined order

    TensorType result_type;         // trace-known result type
    Predicate  predicate;           // PivotThreshold / RankLe
    RngKind    rng_kind = RngKind::Gumbel;   // Gumbel
    std::uint32_t imm = 0;          // TopK k, Transpose axis pair, Iota len, …
    std::uint32_t imm2 = 0;
    std::uint32_t imm3 = 0;
    std::uint32_t name_index = 0;   // KernelCall / SinkCall → Trace::names

    ValueId       result_id = 0;    // first SSA id this op defines
    std::uint32_t result_count = 1; // top_k / sort_desc → 2 (value, index)
};

// op_result_count(OpCode) is provided by op_table.hpp (sort_desc/top_k → 2).

// A channel effect: this stage `put`s `value` into `channel` at pass end
// (predicated on the commit flag). Ordered within the stage (register
// semantics: double put = last wins within a pass).
struct ChannelPut {
    ChannelId channel = 0;
    ValueId   value = 0;
};

// The four attachment points. Boundary stages (Prologue/Epilogue) run once per
// pass; the anatomical taps (OnAttnProj/OnAttn) run once per layer.
enum class StageKind : std::uint8_t {
    Prologue = PTIR_STAGE_PROLOGUE,
    OnAttnProj = PTIR_STAGE_ON_ATTN_PROJ,
    OnAttn = PTIR_STAGE_ON_ATTN,
    Epilogue = PTIR_STAGE_EPILOGUE,
};

// One stage program: a straight-line SSA op DAG plus its channel effects. Its
// readiness predicate is the AND over the first-op direction of every channel
// it touches.
struct Stage {
    StageKind kind = StageKind::Epilogue;
    std::vector<Op>         ops;
    std::vector<ChannelPut> puts;
    // Channels consumed by a chan_take / peeked by a chan_read in this stage --
    // including takes whose result is unused. Readiness needs them full.
    std::vector<ChannelId>  takes;
    std::vector<ChannelId>  reads;
};

// Which way a fire's readiness gate points for one channel: the direction the
// FIRST op to touch it in pass order requires.
//
// Shipped, not derived. A channel that is taken and then put back -- a counter,
// a beam state, a DFA cursor -- appears in both a stage's `takes` and its
// `puts`, so a driver that ORs the two sets asks for a ring that is at once
// non-empty and non-full, which `capacity == 1` can never be. Only the order
// resolves it, and the order is a fact about the program.
enum class Readiness : std::uint8_t {
    Untouched = PIE_READINESS_UNTOUCHED,
    NeedsFull = PIE_READINESS_NEEDS_FULL,
    NeedsEmpty = PIE_READINESS_NEEDS_EMPTY,
};

// A channel declaration: a bounded queue of `capacity + 1` cells (ring), each of
// shape/dtype. `has_seed` marks a Channel::from(v) -- a cell pre-put full at
// instantiation. `host_visible` channels always keep the full ring.
struct Channel {
    ChannelId  id = 0;
    TensorType type;
    std::uint32_t capacity = 1;     // logical capacity; ring holds capacity+1 cells
    bool       has_seed = false;    // Channel::from(seed) — first cell put full
    bool       host_visible = false;
    bool       host_reader = false; // host HARVESTS (output)
    std::int8_t extern_dir = -1;    // -1 private, 0 import, 1 export
    Readiness  readiness = Readiness::Untouched;
    std::string extern_name;
};

// A descriptor-port binding: a forward-pass input port fed by a channel. The
// token family (embed_tokens/positions/w_slot/w_off) CONSUMES (takes) its
// channel at the descriptor phase; geometry/masks PEEK (read).
struct PortBinding {
    std::uint8_t port = 0;      // PtirPort tag
    ChannelId    channel = 0;
    bool         is_const = false;   // const-folded port (no channel consumption)
    TensorType   const_type;
    std::vector<std::uint8_t> const_data;
};

// True if a port CONSUMES (takes) its channel at the descriptor phase (advances
// the ring), vs peeks (reads). Token family takes; geometry/masks peek.
inline bool port_consumes(std::uint8_t port) {
    // PtirPort: 0 embed_tokens, 2 positions, 6 w_slot, 7 w_off take.
    return port == 0 || port == 2 || port == 6 || port == 7;
}

// A full program: its value table, channels, and stage programs. The batching
// identity is the tuple of stage traces; program identity is `program_hash`.
struct Trace {
    std::vector<Value>       values;    // SSA value table (indexed by ValueId)
    std::vector<Channel>     channels;
    // Program-wide name table for second-party kernels and sinks. An
    // `OpCode::KernelCall` or `OpCode::SinkCall` carries its index here in
    // `Op::name_index`.
    std::vector<std::string> names;
    std::vector<PortBinding> ports;     // descriptor-port channel bindings
    std::vector<Stage>       stages;    // in attachment order

    const Value* value(ValueId id) const {
        for (const auto& v : values) if (v.id == id) return &v;
        return nullptr;
    }
    const Channel* channel(ChannelId id) const {
        for (const auto& c : channels) if (c.id == id) return &c;
        return nullptr;
    }
};

namespace detail {

inline Shape shape_from(const PieU32Slice& dims) {
    Shape shape;
    shape.dims.assign(dims.ptr, dims.ptr + dims.len);
    return shape;
}

inline std::string string_from(const PieBytes& bytes) {
    return std::string(reinterpret_cast<const char*>(bytes.ptr), bytes.len);
}

}  // namespace detail

// Adopts the shipped package into the driver's execution model. A field copy:
// every decision was made on the host.
inline Trace adopt(const PieLaunchPackage& package) {
    Trace trace;

    trace.values.reserve(package.values.len);
    for (std::size_t i = 0; i < package.values.len; ++i) {
        const PieLaunchValue& src = package.values.ptr[i];
        Value value;
        value.id = src.id;
        value.type.shape = detail::shape_from(src.shape);
        value.type.dtype = static_cast<DType>(src.dtype);
        value.source = static_cast<ValueSource>(src.source);
        value.lit.dtype = static_cast<DType>(src.dtype);
        value.lit.bits = src.literal_bits;
        value.intrinsic = static_cast<Intrinsic>(src.intrinsic);
        value.channel = src.channel;
        trace.values.push_back(std::move(value));
    }

    trace.channels.reserve(package.channels.len);
    for (std::size_t i = 0; i < package.channels.len; ++i) {
        const PieLaunchChannel& src = package.channels.ptr[i];
        Channel channel;
        channel.id = src.id;
        channel.type.shape = detail::shape_from(src.shape);
        channel.type.dtype = static_cast<DType>(src.dtype);
        channel.capacity = src.capacity;
        channel.has_seed = (src.flags & PIE_CHANNEL_SEEDED) != 0;
        channel.host_visible = (src.flags & PIE_CHANNEL_HOST_VISIBLE) != 0;
        channel.host_reader = (src.flags & PIE_CHANNEL_HOST_READER) != 0;
        channel.extern_dir = src.extern_dir;
        channel.readiness = static_cast<Readiness>(src.readiness);
        channel.extern_name = detail::string_from(src.extern_name);
        trace.channels.push_back(std::move(channel));
    }

    trace.names.reserve(package.names.len);
    for (std::size_t i = 0; i < package.names.len; ++i) {
        trace.names.push_back(detail::string_from(package.names.ptr[i]));
    }

    trace.ports.reserve(package.ports.len);
    for (std::size_t i = 0; i < package.ports.len; ++i) {
        const PieLaunchPort& src = package.ports.ptr[i];
        PortBinding port;
        port.port = src.port;
        port.channel = src.channel;
        port.is_const = src.is_const != 0;
        port.const_type.shape = detail::shape_from(src.const_shape);
        port.const_type.dtype = static_cast<DType>(src.const_dtype);
        const auto* data = static_cast<const std::uint8_t*>(src.const_data.ptr);
        port.const_data.assign(data, data + src.const_data.len);
        trace.ports.push_back(std::move(port));
    }

    trace.stages.reserve(package.stages.len);
    for (std::size_t i = 0; i < package.stages.len; ++i) {
        const PieLaunchStage& src = package.stages.ptr[i];
        Stage stage;
        stage.kind = static_cast<StageKind>(src.kind);
        stage.ops.reserve(src.ops.len);
        for (std::size_t o = 0; o < src.ops.len; ++o) {
            const PieLaunchOp& raw = src.ops.ptr[o];
            Op op;
            op.code = static_cast<OpCode>(raw.code);
            op.args.assign(raw.args.ptr, raw.args.ptr + raw.args.len);
            op.result_type.shape = detail::shape_from(raw.shape);
            op.result_type.dtype = static_cast<DType>(raw.dtype);
            op.predicate.tag = static_cast<PredTag>(raw.pred_tag);
            op.predicate.payload = raw.pred_payload;
            op.rng_kind = static_cast<RngKind>(raw.rng_kind);
            op.imm = raw.imm;
            op.imm2 = raw.imm2;
            op.imm3 = raw.imm3;
            op.name_index = raw.name_index;
            op.result_id = raw.result_id;
            op.result_count = raw.result_count;
            stage.ops.push_back(std::move(op));
        }
        stage.puts.reserve(src.puts.len);
        for (std::size_t p = 0; p < src.puts.len; ++p) {
            stage.puts.push_back(
                ChannelPut{src.puts.ptr[p].channel, src.puts.ptr[p].value});
        }
        stage.takes.assign(src.takes.ptr, src.takes.ptr + src.takes.len);
        stage.reads.assign(src.reads.ptr, src.reads.ptr + src.reads.len);
        trace.stages.push_back(std::move(stage));
    }

    return trace;
}

}  // namespace pie::driver::launch

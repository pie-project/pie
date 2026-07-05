#pragma once

// In-memory PTIR trace model — the CUDA driver's decoded view of a traced
// program (docs/ptir/overview.md §1–§4). Generalizes the Sampling-IR `Program`
// (sampling_ir/ir.hpp) from "one epilogue slot" to "stage programs over
// channels": a trace carries per-stage op DAGs, channel declarations, value
// definitions, and stage-tagged attachment.
//
// This is a value → value graph over trace-known shapes (§5.1); nothing here
// touches memory (§5.2 — there is no memory op in the IR). The only stateful
// construct is the Channel (§1): a bounded queue of cells with full/empty bits.
//
// PROVISIONAL: the binary trace-container encoding is co-owned with echo
// (op-table) and delta (SDK emitter). This header is the decoded shape the
// tier-0/1/2 backends consume; reader.cpp fills it from the wire.

#include <cstdint>
#include <string>
#include <vector>

#include "ptir/op_table.hpp"

namespace pie_cuda_driver::ptir {

using ValueId   = std::uint32_t;   // SSA value id, unique within a trace
using ChannelId = std::uint32_t;   // channel index within a trace
using InputKey  = std::uint32_t;   // host-input selector into the request table
using ProgramId = std::uint32_t;

// Trace-known logical shape: an ordered dim list (row-major). Empty = scalar.
// The overview uses ranks up to [B, P, page]; tier-0 treats the leading dim as
// the row (CTA) axis and the trailing dims flattened as the per-row length.
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

// pivot_threshold / rank_le predicate. For RankLe the payload is the immediate
// k; for CummassLe/ProbGe it is a Scalar value id (§ order family).
struct Predicate {
    PredTag       tag = PredTag::RankLe;
    std::uint32_t payload = 0;
};

enum class RngKind : std::uint8_t { Uniform = 0, Gumbel = 1 };

// A value's producer. Constants and intrinsics are trace-time roots; channel
// reads are the stateful roots; op results are interior nodes.
enum class ValueSource : std::uint8_t {
    Const,        // trace-known literal (Broadcast/Const-fold; codegen may inline)
    Intrinsic,    // stage-scoped intrinsic value (logits/hidden/…)
    HostInput,    // a submit- or late-bound host tensor (readiness input, §1)
    ChannelTake,  // consumes a full cell, marks it empty (readiness: needs full)
    ChannelRead,  // peeks a full cell without consuming (readiness: needs full)
    OpResult,     // produced by an op in this stage's DAG
};

enum class Intrinsic : std::uint8_t {
    Logits = 0, MtpLogits = 1, Hidden = 2, ValueHead = 3, Query = 4,
};

enum class HostAvailability : std::uint8_t { SubmitBound = 0, LateBound = 1 };

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
    InputKey         host_key = 0;                          // HostInput
    HostAvailability host_avail = HostAvailability::SubmitBound;  // HostInput
    ChannelId        channel = 0;                           // ChannelTake / ChannelRead
};

// One op in a stage DAG. Operand fields carry value ids; `result_id` is the SSA
// value this op defines (top_k defines two: value then index). Op-specific
// immediates ride alongside (predicate/rng/imm), mirroring sampling_ir::Op.
struct Op {
    OpCode  code = OpCode::Add;

    std::vector<ValueId> args;      // operand value ids, in op-defined order

    TensorType result_type;         // trace-known result type
    Predicate  predicate;           // PivotThreshold / RankLe
    RngKind    rng_kind = RngKind::Gumbel;   // Gumbel
    std::uint32_t imm = 0;          // TopK k, Transpose axis pair, Iota len, …

    ValueId       result_id = 0;    // first SSA id this op defines
    std::uint32_t result_count = 1; // top_k / sort_desc → 2 (value, index)
};

// op_result_count(OpCode) is provided by op_table.hpp (sort_desc/top_k → 2).

// A channel effect: this stage `put`s `value` into `channel` at pass end
// (predicated on the commit flag, §7.1). Ordered within the stage (register
// semantics: double put = last wins within a pass, T4).
struct ChannelPut {
    ChannelId channel = 0;
    ValueId   value = 0;
};

// A declared program output slot (host-facing publication metadata). `kind`
// mirrors the SDK OutputKind.
enum class OutputKind : std::uint8_t {
    Token = 0, Distribution = 1, Logits = 2, Logprobs = 3,
    Entropy = 4, Scalar = 5, Embedding = 6,
};

struct Output {
    ValueId    value = 0;
    OutputKind kind = OutputKind::Token;
};

// The four attachment points (overview §5.3), matching the wire encoding
// (interface/sampling-ir/src/ptir/registry.rs + PTIR_STAGE_* in ptir_abi.h:
// Prologue=0, OnAttnProj=1, OnAttn=2, Epilogue=3). Boundary stages (Prologue/
// Epilogue) run once per pass; the anatomical taps (OnAttnProj/OnAttn) run once
// per layer. A spurious `AttnMask=2` here previously shifted OnAttn→3/Epilogue→4,
// so a wire Epilogue byte (3) decoded to OnAttn → the Epilogue ran PER-LAYER
// instead of once (masked for single-layer goldens; BROKE multi-layer programs
// like the pentathlon composition). Must match the wire byte-for-byte — the
// decode casts the wire byte directly (bound.hpp `(StageKind)cs.stage`).
enum class StageKind : std::uint8_t {
    Prologue = 0, OnAttnProj = 1, OnAttn = 2, Epilogue = 3,
};

// One stage program: a straight-line SSA op DAG plus its channel effects and
// declared outputs. Its readiness predicate is the AND over the first-op
// direction of every channel it touches (§1/T3), computed by the validator.
struct Stage {
    StageKind kind = StageKind::Epilogue;
    std::vector<Op>         ops;
    std::vector<ChannelPut> puts;
    std::vector<Output>     outputs;
    // Channels consumed by a chan_take / peeked by a chan_read in this stage —
    // including takes whose result is unused (e.g. §6.2's klen/kvm drain). The
    // container translator fills these; readiness needs them full.
    std::vector<ChannelId>  takes;
    std::vector<ChannelId>  reads;
};

// A channel declaration (§1): a bounded queue of `capacity + 1` cells (ring),
// each of shape/dtype. `seed` marks a Channel::from(v) — a cell pre-put full at
// instantiation. `host_visible` channels always keep the full ring (§7.1).
struct Channel {
    ChannelId  id = 0;
    TensorType type;
    std::uint32_t capacity = 1;     // logical capacity; ring holds capacity+1 cells
    bool       has_seed = false;    // Channel::from(seed) — first cell put full
    bool       host_visible = false;
    bool       host_reader = false; // PtirHostRole::READER — host HARVESTS (output)
    std::int8_t extern_dir = -1;     // -1 private, 0 import, 1 export
    std::string extern_name;
};

// A descriptor-port binding: a forward-pass input port fed by a channel (C1).
// The token family (embed_tokens/positions/w_slot/w_off) CONSUMES (takes) its
// channel at the descriptor phase; geometry/masks PEEK (read). echo's pin §5.1.
struct PortBinding {
    std::uint8_t port = 0;      // PtirPort tag
    ChannelId    channel = 0;
    bool         is_const = false;   // const-folded port (no channel consumption)
};

// True if a port CONSUMES (takes) its channel at the descriptor phase (advances
// the ring), vs peeks (reads). Token family takes; geometry/masks peek.
inline bool port_consumes(std::uint8_t port) {
    // PtirPort: 0 embed_tokens, 2 positions, 6 w_slot, 7 w_off take.
    return port == 0 || port == 2 || port == 6 || port == 7;
}

// A full traced program: its value table, channels, and stage programs. The
// batching identity is the tuple of stage traces (T5); program identity is a
// hash over the canonical encoding (C3).
struct Trace {
    std::vector<Value>       values;    // SSA value table (indexed by ValueId)
    std::vector<Channel>     channels;
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

}  // namespace pie_cuda_driver::ptir

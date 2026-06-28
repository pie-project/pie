#pragma once

// In-memory Sampling-IR graph — the C++ mirror of the `pie-sampling-ir` typed
// IR (interface/sampling-ir/src/types.rs), populated by the bytecode reader
// (reader.hpp) from a PSIR v1 buffer (interface/sampling-ir/BYTECODE.md).
//
// The driver only ever receives *validated* programs (the host runs
// pie-sampling-ir's validator before submit), so this model carries no
// validation logic — just the decoded shape. The reader does structural
// bounds-checking only (magic / version / EOF / unknown tags).

#include <cstdint>
#include <vector>

namespace pie_cuda_driver::sampling_ir {

using ValueId = std::uint32_t;
using InputId = std::uint32_t;
using InputKey = std::uint32_t;
using ProgramId = std::uint32_t;

// Element dtype. Wire tags: F32=0, I32=1, U32=2, Bool=3 (BYTECODE.md §1).
enum class DType : std::uint8_t { F32 = 0, I32 = 1, U32 = 2, Bool = 3 };

inline std::uint32_t dtype_size(DType d) {
    switch (d) {
        case DType::F32: return 4;
        case DType::I32: return 4;
        case DType::U32: return 4;
        case DType::Bool: return 1;
    }
    return 4;
}

inline bool dtype_is_float(DType d) { return d == DType::F32; }
inline bool dtype_is_int(DType d) { return d == DType::I32 || d == DType::U32; }

// Logical shape. Wire = `tag:u8 | a:u32 | b:u32` (BYTECODE.md §1).
enum class ShapeTag : std::uint8_t { Scalar = 0, Vector = 1, Matrix = 2, Indices = 3 };

struct Shape {
    ShapeTag tag = ShapeTag::Scalar;
    std::uint32_t a = 0;   // Vector/Indices: len ; Matrix: rows
    std::uint32_t b = 0;   // Matrix: len

    bool is_scalar() const { return tag == ShapeTag::Scalar; }
    // Length of the last (reduce/scan) axis, 1 for Scalar.
    std::uint32_t last_len() const {
        switch (tag) {
            case ShapeTag::Scalar: return 1;
            case ShapeTag::Vector: return a;
            case ShapeTag::Matrix: return b;
            case ShapeTag::Indices: return a;
        }
        return 1;
    }
    std::uint32_t rows() const { return tag == ShapeTag::Matrix ? a : 1; }

    static Shape scalar() { return {ShapeTag::Scalar, 0, 0}; }
    static Shape vector(std::uint32_t len) { return {ShapeTag::Vector, len, 0}; }
    static Shape matrix(std::uint32_t rows, std::uint32_t len) { return {ShapeTag::Matrix, rows, len}; }
    static Shape indices(std::uint32_t len) { return {ShapeTag::Indices, len, 0}; }
};

struct ValueType {
    Shape shape;
    DType dtype = DType::F32;
};

// Op tag bytes (BYTECODE.md §4). Values are the wire constants.
enum class OpCode : std::uint8_t {
    Exp = 0x01, Log = 0x02, Neg = 0x03, Recip = 0x04, Abs = 0x05, Sign = 0x06,
    Add = 0x10, Sub = 0x11, Mul = 0x12, Div = 0x13, MaxElem = 0x14, MinElem = 0x15,
    Gt = 0x16, Ge = 0x17, Eq = 0x18,
    Select = 0x20,
    ReduceSum = 0x30, ReduceMax = 0x31, ReduceMin = 0x32, ReduceArgmax = 0x33,
    Broadcast = 0x38,
    CumSum = 0x40, CumProd = 0x41,
    SortDesc = 0x50,
    PivotThreshold = 0x58,
    Gather = 0x60, GatherRow = 0x61, ScatterAdd = 0x62, ScatterSet = 0x63,
    GatherCols = 0x64,   // (v3) src Matrix{rows,len}, idx Vector{rows} → Vector{rows}: out[i]=src[i,idx[i]]
    MaskApply = 0x65,    // (v4) mask_apply(logits, packed-mask): out[j] = bit_j(mask) ? logits[j] : -inf
    Rng = 0x70,
    RowBroadcast = 0x39,   // (v3) per_row Vector{rows} → Matrix{rows, len}
};

enum class PredTag : std::uint8_t { RankLe = 0, CummassLe = 1, ProbGe = 2 };

// Predicate (BYTECODE.md §4): `pred_tag:u8 | payload:u32`. For RankLe the
// payload is the immediate k; for CummassLe/ProbGe it is a Scalar-F32 value id.
struct Predicate {
    PredTag tag = PredTag::RankLe;
    std::uint32_t payload = 0;
};

enum class RngKind : std::uint8_t { Uniform = 0, Gumbel = 1 };

// A decoded op. Operand fields carry value ids; which are meaningful depends on
// `code`. `result_id` / `result_count` are assigned by the reader from op order
// (SSA counter rule, BYTECODE.md §3) — SortDesc defines 2 ids (value-first),
// every other op exactly 1.
struct Op {
    OpCode code = OpCode::Exp;

    // up to three value-id operands (unary: a; binary: a,b; ternary: a,b,c).
    ValueId a = 0;
    ValueId b = 0;
    ValueId c = 0;

    Shape shape;                 // Broadcast (target), Rng (output)
    Predicate predicate;         // PivotThreshold
    RngKind rng_kind = RngKind::Gumbel;
    std::uint32_t imm = 0;       // RowBroadcast: the target row length `len`
    std::uint32_t stream = 0;    // Rng (v4): static per-op stream salt (Model B, no seed operand)

    ValueId result_id = 0;       // first SSA id this op defines
    std::uint32_t result_count = 1;
};

inline std::uint32_t op_result_count(OpCode c) {
    return c == OpCode::SortDesc ? 2u : 1u;
}

enum class BindingTag : std::uint8_t { Const = 0, Intrinsic = 1, Host = 2, Output = 3 };
// Which `ws.logits` row an Intrinsic binding resolves to. `MtpLogits` = the
// speculator DRAFT row (#21 phase-2): identical bytecode, manifest-only
// (foxtrot's additive `ir::Binding::MtpLogits`) — no opcode/version bump. The
// executor maps this to the runtime `IntrinsicKind` (Logits → the sampled row,
// MtpLogits → `ws.logits[mtp_draft_row]`); delta bridges at jit_backend.cpp:282.
enum class Intrinsic : std::uint8_t { Logits = 0, MtpLogits = 1 };
enum class HostAvailability : std::uint8_t { SubmitBound = 0, LateBound = 1 };

// Semantic kind of a declared slot output (PSIR v2). The host marshals each
// output into the matching WIT slot-output variant; the driver reads it to know
// how to write the value into ForwardResponse. Wire tags are frozen (mirror the
// SDK `OutputKind`): Token=0, Distribution=1, Logits=2, Logprobs=3, Entropy=4,
// Scalar=5, Embedding=6.
enum class OutputKind : std::uint8_t {
    Token = 0,
    Distribution = 1,
    Logits = 2,
    Logprobs = 3,
    Entropy = 4,
    Scalar = 5,
    Embedding = 6,
};

// Raw literal: dtype + 4-byte payload (interpret per dtype, BYTECODE.md §1).
struct Literal {
    DType dtype = DType::F32;
    std::uint32_t bits = 0;

    float as_f32() const { float f; __builtin_memcpy(&f, &bits, 4); return f; }
    std::int32_t as_i32() const { return static_cast<std::int32_t>(bits); }
    std::uint32_t as_u32() const { return bits; }
    bool as_bool() const { return bits != 0; }
};

struct Binding {
    BindingTag tag = BindingTag::Intrinsic;
    Literal lit;                                 // Const
    Intrinsic intrinsic = Intrinsic::Logits;     // Intrinsic
    InputKey host_key = 0;                        // Host
    HostAvailability host_avail = HostAvailability::SubmitBound;  // Host
    ProgramId out_program = 0;                    // Output
    ValueId out_value = 0;                        // Output
};

struct Input {
    InputId id = 0;
    ValueType ty;
    Binding binding;
};

// A declared slot output: the SSA value id it exposes + its semantic kind (v2).
struct Output {
    ValueId value = 0;
    OutputKind kind = OutputKind::Token;
};

struct Slot {
    std::vector<Op> ops;
    std::vector<Output> outputs;
};

struct Program {
    std::vector<Input> inputs;
    std::vector<Slot> slots;

    // v4 (Model B): RNG seed is ambient (per-row sample_seed S), not an Op::Rng
    // operand. The codegen emits a RowSeed buffer + seed_eff_stream(S[r], stream).
    // False for v2/v3 programs (seed is a Host input operand of Op::Rng).
    bool ambient_seed = false;

    std::uint32_t n_inputs() const { return static_cast<std::uint32_t>(inputs.size()); }
};

}  // namespace pie_cuda_driver::sampling_ir

#pragma once

// Sampling-IR codegen → JIT hand-off contract (Lane L2 / charlie → L3 / delta).
//
// The codegen (W2) lowers validated IR bytecode into a DAG of CUDA-C kernels.
// This header is the *seam* it hands to delta's JIT: per-kernel source + entry
// name + launch formula + ordered argument sources, plus the intermediate
// device buffers that carry cross-kernel values. delta compiles each kernel's
// `source` via NVRTC (every kernel's source already embeds primitive_prelude()),
// caches by program hash, and binds + launches the DAG via the driver API.
//
// Layering (per echo's L4 map): the runtime never sees these types — it talks
// only to delta's abstract `IProgramBackend`. This header is the *internal*
// charlie↔delta contract behind that backend; delta maps a KernelDAG onto its
// ProgramHandle and projects the declared inputs/outputs onto echo's
// ProgramInterface.
//
// STATUS: DRAFT pending alpha's BYTECODE.md. The launch formulas and the
// prelude are firm (W1, tested); the struct field set may widen once the
// frozen bytecode lands. Coordinated in #programmable-sampler.

#include <cstdint>
#include <string>
#include <vector>

#include "sampling_ir/ir.hpp"
#include "sampling_ir/primitives_src.hpp"

namespace pie_cuda_driver::sampling_ir {

// DType / dtype_size are defined in ir.hpp (the single canonical mirror of the
// pie-sampling-ir DType). codegen reuses them directly.

// ---------------------------------------------------------------------------
// Buffers — every device buffer the program touches carries a BufferId so the
// JIT can bind echo's resolved pointers (external inputs / output_ptrs) and
// allocate scratch (intermediates). BufferClass tags how delta/echo source it.
// ---------------------------------------------------------------------------
using BufferId = std::uint32_t;

enum class BufferClass : std::uint8_t {
    IntrinsicLogits,   // ws.logits rows (bf16 [rows, vocab]); echo passes base+row*vocab
    HostSubmit,        // stable device buffer, contents refreshed each fire
    HostLate,          // late-bound device buffer; miss = skip the program
    Intermediate,      // device scratch produced by an earlier kernel (delta allocates)
    Output,            // a declared program output slot (echo provides output_ptr)
    RowSeed,           // ambient per-row RNG seed S (u32 [num_rows]); external, bind-only
                       // (Model B: Op::Rng has no seed operand; echo binds
                       // row_seeds[r] = pi.sample_seed[r]). Like IntrinsicLogits,
                       // the JIT never allocates it — delta binds LaunchArgs.row_seeds.
};

struct BufferDecl {
    BufferId id = 0;
    BufferClass cls = BufferClass::Intermediate;
    DType dtype = DType::F32;
    std::uint32_t elem_count = 0;   // per-row element count (row-major); 1 = scalar

    // External inputs (Intrinsic/HostSubmit/HostLate): the IR input id echo
    // binds from. Output: which program output slot (0-based).
    std::uint32_t input_id = 0;
    std::uint32_t output_index = 0;

    // Output buffers only: the declared semantic kind (PSIR v2), so echo knows
    // how to marshal the value into ForwardResponse. Ignored for non-outputs.
    OutputKind output_kind = OutputKind::Token;

    // Late-bound buffers only (HostLate / cross-program Output-ref input): the
    // index of the first kernel in the DAG that consumes this buffer — the
    // inject-before barrier for delta's split-launch (launch_until/launch_from).
    // This is the C++ device-side mirror of alpha's Rust `late_values()`
    // first-use analysis (one definition, two surfaces — like output_kind).
    // UINT32_MAX if the buffer is not late-bound or is unused.
    std::uint32_t first_consuming_kernel = 0xFFFFFFFFu;

    // Batched (M>1) lowering: `elem_count` is the PER-ROW element count; the
    // buffer physically holds `num_rows * elem_count` elements (row-major), so
    // the JIT must allocate / bind `num_rows * elem_count` at fire time. False
    // for single-row (M=1) lowering, where elem_count is the whole size.
    bool batched = false;

    // IntrinsicLogits buffers only: which ws.logits row the binding resolves to
    // (Logits = the sampled row; MtpLogits = the speculator draft row, #21
    // phase-2 mtp-logits). Manifest-only (identical bytecode for both). delta
    // bridges this to the runtime `IntrinsicKind` at jit_backend.cpp:282
    // (in.intrinsic), which echo's resolver reads. Ignored for non-intrinsic.
    Intrinsic intrinsic_kind = Intrinsic::Logits;

    std::uint64_t byte_size() const {
        return static_cast<std::uint64_t>(elem_count) * dtype_size(dtype);
    }
};

// ---------------------------------------------------------------------------
// Kernel arguments — delta's frozen taxonomy: Buffer | Scalar | Param.
//   * Buffer — a device pointer resolved by BufferId (external or intermediate).
//   * Scalar — a literal baked into the launch (const folded by codegen).
//   * Param  — a launch-context slot: 0 = num_rows, 1 = vocab_size,
//              2 = prng_offset (RESERVED / no-op at MVP — must NOT be folded
//              into the Gumbel noise; folding it regresses sample_temp parity).
// ---------------------------------------------------------------------------
enum class KernelArgKind : std::uint8_t { Buffer, Scalar, Param };

enum class ParamSlot : std::uint32_t { NumRows = 0, VocabSize = 1, PrngOffset = 2 };

struct KernelArg {
    KernelArgKind kind = KernelArgKind::Buffer;

    BufferId buffer = 0;                 // kind == Buffer
    std::uint32_t param_index = 0;       // kind == Param (see ParamSlot)

    DType scalar_dtype = DType::F32;      // kind == Scalar
    union {
        std::int32_t  i32;
        float         f32;
        std::uint32_t u32;
        std::uint64_t u64;
    } scalar = {0};

    static KernelArg Buf(BufferId id) { KernelArg a; a.kind = KernelArgKind::Buffer; a.buffer = id; return a; }
    static KernelArg Par(ParamSlot s) { KernelArg a; a.kind = KernelArgKind::Param; a.param_index = static_cast<std::uint32_t>(s); return a; }
    static KernelArg I32(std::int32_t v) { KernelArg a; a.kind = KernelArgKind::Scalar; a.scalar_dtype = DType::I32; a.scalar.i32 = v; return a; }
    static KernelArg F32(float v) { KernelArg a; a.kind = KernelArgKind::Scalar; a.scalar_dtype = DType::F32; a.scalar.f32 = v; return a; }
    static KernelArg U32(std::uint32_t v) { KernelArg a; a.kind = KernelArgKind::Scalar; a.scalar_dtype = DType::U32; a.scalar.u32 = v; return a; }
};

// ---------------------------------------------------------------------------
// Launch shape — grid/block as a formula over the launch context, so delta and
// codegen share ONE definition (compute_launch_dims below). At M=1 decode delta
// bakes the dims (num_rows = 1); the formula is the single source of truth.
// Block width is always PIE_IR_BLOCK (256); the prelude's block primitives
// (reduce/scan/pivot/sort) require exactly 256 threads.
// ---------------------------------------------------------------------------
enum class LaunchShape : std::uint8_t {
    OneBlockPerRow,       // grid = num_rows,                block = 256
    GridStrideOverVocab,  // grid = ceil(num_rows*vocab/256),block = 256  (elementwise map)
    GridStrideOverLen,    // grid = ceil(len/256),           block = 256  (gather/scatter)
    Custom,               // explicit grid_x / block_x
};

struct LaunchDims {
    std::uint32_t grid_x;
    std::uint32_t block_x;
};

inline constexpr std::uint32_t kSamplingIrBlock = 256;

// Single source of truth for grid/block. `len` is only used by GridStrideOverLen
// (the kernel's element count). delta calls this with the per-fire context.
inline LaunchDims compute_launch_dims(LaunchShape shape,
                                      std::uint32_t num_rows,
                                      std::uint32_t vocab_size,
                                      std::uint32_t len,
                                      std::uint32_t custom_grid = 0,
                                      std::uint32_t custom_block = 0) {
    const std::uint32_t B = kSamplingIrBlock;
    auto ceil_div = [](std::uint64_t a, std::uint64_t b) -> std::uint32_t {
        return static_cast<std::uint32_t>((a + b - 1) / b);
    };
    switch (shape) {
        case LaunchShape::OneBlockPerRow:
            return {num_rows == 0 ? 1u : num_rows, B};
        case LaunchShape::GridStrideOverVocab:
            return {ceil_div(static_cast<std::uint64_t>(num_rows) * vocab_size, B), B};
        case LaunchShape::GridStrideOverLen:
            return {ceil_div(len == 0 ? 1u : len, B), B};
        case LaunchShape::Custom:
            return {custom_grid, custom_block};
    }
    return {1u, B};
}

// ---------------------------------------------------------------------------
// One generated kernel.
// ---------------------------------------------------------------------------
struct KernelDesc {
    std::string entry_name;          // extern "C" __global__ symbol inside `source`
    std::string source;              // self-contained CUDA-C (embeds primitive_prelude())
    LaunchShape shape = LaunchShape::OneBlockPerRow;
    std::uint32_t shared_bytes = 0;  // dynamic shared mem (0 = static only)
    std::uint32_t len_buffer = 0;    // GridStrideOverLen: buffer whose elem_count = len
    std::uint32_t custom_grid = 0;   // LaunchShape::Custom
    std::uint32_t custom_block = 0;  // LaunchShape::Custom
    std::vector<KernelArg> args;     // ordered to match cuLaunchKernel param order
};

// The full hand-off: a topologically ordered kernel DAG + all buffers (external
// inputs, intermediates, outputs). delta compiles each kernel's `source`,
// allocates the Intermediate buffers, binds External/Output buffers from echo
// by BufferId, then launches in order resolving each KernelArg per fire.
struct KernelDAG {
    std::vector<KernelDesc> kernels;
    std::vector<BufferDecl> buffers;
};

// delta only needs this header to obtain the device prelude that must prefix
// every kernel source it NVRTC-compiles: call primitive_prelude() (declared in
// primitives_src.hpp, same namespace).

// ---------------------------------------------------------------------------
// Lowering — the W2 entrypoint delta wires the backend to.
//
//   lower(program) → KernelDAG :  decode'd IR (ir.hpp) → fused CUDA-C kernels.
//   lower(bytecode) :             decode then lower (convenience).
//
// MVP scope (design §3.3): single slot, single position, M=1 decode. Shapes are
// static (baked from the IR). Each emitted kernel is one block of 256 threads
// (LaunchShape::OneBlockPerRow); intermediate values are materialized to typed
// device buffers; a kernel boundary is cut at a data-dependent barrier (a
// reduction result consumed as a gather index). On success `ok` is true and the
// DAG is populated; otherwise `error` explains why (e.g. an unsupported op or
// shape for the MVP). Each kernel's `source` already embeds primitive_prelude().
// ---------------------------------------------------------------------------
struct LowerResult {
    bool ok = false;
    std::string error;
    KernelDAG dag;
};

// Lowering options.
//
// `batched` (M>1): lower the (single-row) program for a whole batch of rows in
// ONE launch — grid = num_rows (one block per batch row, `blockIdx.x` = row),
// matching the hardwired sampler shape (e.g. sample_temp.cu). Every value gains
// an implicit leading batch dim: a program-scalar → one-per-row, a vector{L} →
// num_rows×L. All intermediate/output/external `BufferDecl`s are marked
// `batched` with a PER-ROW `elem_count` (the JIT sizes num_rows×elem_count).
// The intrinsic logits is bound as `[num_rows, V]`. This is the batch-throughput
// path (flat in B); the default (M=1) path is one block total. Only programs
// whose ops are all per-row (the core samplers — no cross-row data dependence)
// are batchable; a cross-row barrier in batched mode is rejected.
struct LowerOptions {
    bool batched = false;
};

// v4 binding manifest. v4 bytecode is binding-free (InputDecl = dtype|shape only);
// the binding — which slot is the logits intrinsic vs a host tensor, and its
// readiness — is supplied at attach/registration alongside the bytecode (sourced
// from the EDSL `Built.bindings`). `get_or_compile(bytecode, manifest)` folds it
// into the cache key. One entry per input slot, in slot order.
enum class BindKind : std::uint8_t { Logits, HostTensor };
struct InputBind {
    BindKind kind = BindKind::Logits;
    std::uint32_t host_key = 0;                                  // HostTensor only
    HostAvailability ready = HostAvailability::SubmitBound;      // HostTensor only
    // Logits kind only: which ws.logits row this intrinsic resolves to. The host
    // sets MtpLogits for foxtrot's `Binding::MtpLogits` (the speculator draft
    // row); identical bytecode, manifest-only. Flows to BufferDecl.intrinsic_kind.
    Intrinsic intrinsic_kind = Intrinsic::Logits;
};
using ProgramManifest = std::vector<InputBind>;

LowerResult lower(const Program& program);
LowerResult lower(const Program& program, const LowerOptions& opts);
LowerResult lower_bytecode(const std::uint8_t* data, std::size_t len);
LowerResult lower_bytecode(const std::uint8_t* data, std::size_t len, const LowerOptions& opts);

// v4 entry: lower binding-free v4 bytecode + its attach manifest. The manifest
// supplies per-slot bindings the v4 bytecode omits; everything else (fusion,
// batched, geometry) is identical to lower_bytecode.
LowerResult lower_bytecode_v4(const std::uint8_t* data, std::size_t len,
                              const ProgramManifest& manifest, const LowerOptions& opts);

}  // namespace pie_cuda_driver::sampling_ir

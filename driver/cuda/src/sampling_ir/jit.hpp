#pragma once

// Sampling-IR NVRTC JIT + driver-API launch infrastructure (lane L3 / delta).
//
// This is the codegen->JIT seam for the CUDA sampling-IR backend. charlie's
// codegen (L2) lowers a validated bytecode program to a `KernelDAG`: a
// topologically-ordered list of self-contained CUDA-C kernels plus the device
// buffers that carry values between them. This module:
//
//   * NVRTC-compiles each kernel's source to PTX for the live device arch
//     (e.g. sm_89 on the RTX 4090), `cuModuleLoadData`s the PTX, and resolves
//     the `extern "C" __global__` entry to a `CUfunction`.
//   * Allocates the DAG's intermediate/IO device buffers once and launches the
//     kernel DAG with the driver API (`cuLaunchKernel`). Cross-kernel values
//     (e.g. `j = reduce-sum(accept)` consumed by `gather-row(resid, j)`) flow
//     through a device buffer at a fixed captured pointer.
//   * Caches the compiled program keyed by the bytecode hash so a program is
//     compiled once and replayed across fires.
//
// The model forward keeps using the CUDA *runtime* API (`<<<>>>`); only the
// sampling-IR DAG uses the driver API. This translation unit is plain C++
// (driver API + NVRTC headers) and needs no nvcc — keep it separable.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda.h>

// All JIT types live in the `::jit` sub-namespace so they never collide with
// charlie's codegen types (`sampling_ir::KernelDAG`/`BufferDecl`/`KernelArg`)
// or echo's runtime types in the parent `sampling_ir` namespace. The
// SamplingIrBackend facade (W3) includes both headers and adapts codegen ->
// jit at that single translation point.
namespace pie_cuda_driver::sampling_ir::jit {

// ─────────────────────────── codegen → JIT seam ───────────────────────────

using BufferId = std::uint32_t;

// Scalar argument / buffer element types the JIT understands. The JIT itself is
// dtype-agnostic for buffers (it only needs byte sizes); `dtype` on BufferDecl
// is a hint carried for codegen/executor bookkeeping.
enum class ScalarType : std::uint8_t { U32, I32, F32, U64 };

enum class ArgKind : std::uint8_t {
    Buffer,  // a device pointer bound from the DAG's buffer table (by BufferId)
    Scalar,  // a compile-time immediate baked into the launch (Const inputs)
    Param,   // a per-fire scalar supplied at launch (prng offset, num_rows, …);
             // refreshed every fire with no recompile
};

// One ordered kernel parameter. Buffer args resolve to a CUdeviceptr at launch;
// Scalar args are compile-time immediates; Param args are per-fire scalars
// patched in from `JitEngine::launch`'s param table (so submit-bound values like
// the RNG offset change each fire without recompiling).
struct KernelArg {
    ArgKind kind = ArgKind::Buffer;
    BufferId buffer = 0;                       // valid iff kind == Buffer
    std::uint32_t param = 0;                   // valid iff kind == Param (param id)
    ScalarType scalar_type = ScalarType::U32;  // width for Scalar/Param
    std::uint64_t scalar_bits = 0;             // raw bits of the immediate (Scalar)

    static KernelArg buffer_arg(BufferId id) {
        KernelArg a;
        a.kind = ArgKind::Buffer;
        a.buffer = id;
        return a;
    }
    // Per-fire scalar resolved from the launch param table by `id`.
    static KernelArg param_arg(std::uint32_t id, ScalarType ty) {
        KernelArg a;
        a.kind = ArgKind::Param;
        a.param = id;
        a.scalar_type = ty;
        return a;
    }
    static KernelArg u32(std::uint32_t v) { return scalar(ScalarType::U32, v); }
    static KernelArg i32(std::int32_t v) {
        return scalar(ScalarType::I32, static_cast<std::uint32_t>(v));
    }
    static KernelArg u64(std::uint64_t v) { return scalar(ScalarType::U64, v); }
    static KernelArg f32(float v) {
        std::uint32_t bits;
        static_assert(sizeof(bits) == sizeof(v));
        __builtin_memcpy(&bits, &v, sizeof(bits));
        return scalar(ScalarType::F32, bits);
    }

  private:
    static KernelArg scalar(ScalarType ty, std::uint64_t bits) {
        KernelArg a;
        a.kind = ArgKind::Scalar;
        a.scalar_type = ty;
        a.scalar_bits = bits;
        return a;
    }
};

struct Dim3 {
    unsigned x = 1, y = 1, z = 1;
};

// How a kernel's grid.x is derived. For batched (M>1) programs the row count is
// dynamic per fire (Param 0), so the JIT recomputes grid.x at launch instead of
// baking it. Mirrors charlie's codegen `LaunchShape`; the codegen→jit adapter
// maps one to the other. `Fixed` keeps the baked `KernelDef::grid` (used when a
// program's grid never varies with the launch context).
//
// Convention (frozen ParamSlot contract): launch param 0 = num_rows, 1 = vocab.
enum class GridShape : std::uint8_t {
    Fixed,                // use KernelDef::grid verbatim
    OneBlockPerRow,       // grid.x = num_rows
    GridStrideOverVocab,  // grid.x = ceil(num_rows * vocab / block.x)
    GridStrideOverLen,    // grid.x = ceil(num_rows * per_row_len / block.x)
};

// One kernel node of the DAG. `source` is self-contained CUDA-C defining a
// single `extern "C" __global__ void name(...)` whose signature matches `args`
// positionally (buffer args -> pointer params; scalar args -> by-value params).
struct KernelDef {
    std::string name;    // entry point; unique within the DAG (no name mangling)
    std::string source;  // self-contained CUDA-C translation unit
    Dim3 grid;           // used when grid_shape == Fixed (else recomputed/fire)
    Dim3 block;
    unsigned shared_bytes = 0;
    std::vector<KernelArg> args;
    GridShape grid_shape = GridShape::Fixed;
    std::uint32_t per_row_len = 0;  // GridStrideOverLen: per-row element count
};

// A device buffer carrying a value into/out of or between kernels. `external`
// buffers (logits inputs, response outputs) are bound by the executor via
// `bind_buffer`; non-external buffers are allocated and owned by the JIT. For
// `batched` (M>1) intermediates, `size_bytes` is the PER-ROW size and the JIT
// allocates `capacity * size_bytes` (capacity = max num_rows seen, grown on
// demand). External buffers are caller-sized regardless of `batched`.
struct BufferDecl {
    BufferId id = 0;
    std::size_t size_bytes = 0;
    ScalarType dtype = ScalarType::F32;  // hint only; JIT allocates raw bytes
    bool external = false;
    bool batched = false;  // size_bytes is per-row; allocate capacity * size_bytes
};

// The whole compiled-once unit handed from codegen to the JIT.
struct KernelDAG {
    std::uint64_t hash = 0;            // bytecode hash — program-cache key
    std::vector<BufferDecl> buffers;  // intermediate + IO device buffers
    std::vector<KernelDef> kernels;   // topologically ordered
};

// ─────────────────────────── compiled program ────────────────────────────

// A compiled, allocated, ready-to-launch program. Owned by JitEngine's cache.
// Buffer pointers live in `buffer_ptrs` at stable addresses; per-kernel
// `cuLaunchKernel` param arrays point into that table and into per-kernel
// scalar backing storage, so `bind_buffer` overrides are picked up with no
// re-marshaling.
class CompiledProgram {
  public:
    CompiledProgram() = default;
    ~CompiledProgram();
    CompiledProgram(const CompiledProgram&) = delete;
    CompiledProgram& operator=(const CompiledProgram&) = delete;

    std::uint64_t hash() const { return hash_; }
    // Current device pointer bound to `id` (JIT-owned allocation or external).
    CUdeviceptr device_ptr(BufferId id) const;

  private:
    friend class JitEngine;

    struct LaunchItem {
        CUfunction fn = nullptr;
        Dim3 grid, block;
        unsigned shared_bytes = 0;
        std::vector<KernelArg> args;
        std::vector<std::uint64_t> scalar_store;  // stable backing for immediates
        std::vector<void*> params;                // cuLaunchKernel kernelParams
        // Param args to patch from the launch param table each fire:
        // (param id, index into scalar_store).
        std::vector<std::pair<std::uint32_t, std::size_t>> param_patches;
        GridShape grid_shape = GridShape::Fixed;  // grid.x recompute mode
        std::uint32_t per_row_len = 0;            // GridStrideOverLen
    };

    std::uint64_t hash_ = 0;
    std::vector<CUmodule> modules_;            // owned; cuModuleUnload on destroy
    std::vector<CUdeviceptr> buffer_ptrs_;     // stable; do not resize post-build
    std::vector<std::size_t> buffer_sizes_;    // per-row bytes if buffer_batched_
    std::vector<bool> buffer_owned_;           // true => cuMemFree on destroy
    std::vector<bool> buffer_batched_;         // size scales with num_rows
    std::size_t capacity_ = 1;                 // num_rows the batched bufs fit
    std::unordered_map<BufferId, std::size_t> buffer_slot_;  // id -> index
    std::vector<LaunchItem> items_;

    // Optional phase-2 CUDA-Graph capture of the kernel DAG (spec §7.2).
    CUgraph graph_ = nullptr;          // owned; cuGraphDestroy on destroy
    CUgraphExec graph_exec_ = nullptr; // owned; cuGraphExecDestroy on destroy
};

// ───────────────────────────── JIT engine ────────────────────────────────

class JitEngine {
  public:
    // Resolves the target architecture (e.g. "compute_89") from the device of
    // the current CUDA context. A context must be current on the calling
    // thread (the executor makes the device's primary context current).
    JitEngine();

    // Compile (or return cached) the program keyed by `dag.hash`. The first
    // call compiles every kernel and allocates non-external buffers; later
    // calls with the same hash are O(1).
    CompiledProgram& get_or_compile(const KernelDAG& dag);

    // Bind an externally-owned device buffer (logits in / response outputs) to
    // a BufferId, overriding any JIT allocation. Must be called for every
    // `external` buffer before `launch`.
    void bind_buffer(CompiledProgram& prog, BufferId id, CUdeviceptr ptr);

    // Launch the whole DAG on `stream` (one `cuLaunchKernel` per node, in DAG
    // order). Cross-kernel values flow through the bound device buffers.
    // `param_values` (indexed by Param-arg id) supplies per-fire scalars such
    // as the RNG offset, num_rows, and vocab_size — refreshed each fire with no
    // recompile.
    void launch(CompiledProgram& prog, CUstream stream,
                const std::vector<std::uint64_t>& param_values = {});

    // ── Phase 2: CUDA-Graph capture (spec §7.2) ──────────────────────────
    // Capture one full DAG launch on `capture_stream` into a CUDA graph and
    // instantiate it for single-launch replay. Buffer bindings and the supplied
    // `param_values` are baked into the captured graph; rebind/re-capture if the
    // device pointers or per-fire scalars must change. `capture_stream` must be
    // a real (non-default/legacy-0) stream. Idempotent: re-instantiation
    // replaces any prior graph.
    void instantiate_graph(CompiledProgram& prog, CUstream capture_stream,
                           const std::vector<std::uint64_t>& param_values = {});

    // Replay the instantiated graph on `stream` with a single cuGraphLaunch.
    // Requires a prior instantiate_graph.
    void launch_graph(CompiledProgram& prog, CUstream stream);

    const std::string& arch() const { return arch_; }

  private:
    void compile_dag(const KernelDAG& dag, CompiledProgram& prog);
    // Verify all referenced buffers are bound, then patch per-fire Param scalars
    // and issue one cuLaunchKernel per DAG node on `stream` (shared by direct
    // launch and graph capture). `num_rows`/`vocab` (from the param table) drive
    // the per-fire grid recompute for batched (M>1) kernels.
    void issue_kernels(CompiledProgram& prog, CUstream stream,
                       const std::vector<std::uint64_t>& param_values);
    // Grow the JIT-owned batched intermediate buffers so they fit `num_rows`
    // rows. No-op if capacity already suffices. Reallocates in place (the
    // pointer-to-pointer the launch params hold stays valid).
    void ensure_capacity(CompiledProgram& prog, std::size_t num_rows);

    std::string arch_;  // NVRTC --gpu-architecture, e.g. "compute_89"
    std::unordered_map<std::uint64_t, std::unique_ptr<CompiledProgram>> cache_;
};

// Launch-param slot convention (frozen with charlie's ParamSlot): the JIT reads
// these to recompute batched grids per fire.
inline constexpr std::uint32_t kParamNumRows = 0;
inline constexpr std::uint32_t kParamVocab = 1;

// FNV-1a 64-bit hash, handy for keying the program cache off raw bytecode.
std::uint64_t fnv1a64(const void* data, std::size_t len);

}  // namespace pie_cuda_driver::sampling_ir::jit

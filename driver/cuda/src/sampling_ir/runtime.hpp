#pragma once

// Sampling-IR executor runtime (lane L4 / echo).
//
// This is the *separable seam* that wires the Sampling-IR path into the CUDA
// executor without colliding with charlie (codegen, `sampling_ir/codegen.*`)
// or delta (NVRTC JIT + launch, `sampling_ir/jit.*`). It owns three things:
//
//   1. INPUT BINDING — mapping each IR program input to a concrete device
//      source per fire (`intrinsic(logits)` / `const` / `host(submit-bound)` /
//      `host(late-bound)`), with the **late-bind miss = skip** policy.
//   2. OUTPUT MARSHALING — mapping each declared program output value to a
//      `ForwardResponse` slot (reusing the response-builder / response-subpass
//      marshaling already in the executor).
//   3. MODE SELECT + ORCHESTRATION — at the executor insertion point
//      (`executor.cpp`, after `run_forward_dispatch` writes `ws.logits`,
//      parallel to the legacy `single_gpu_greedy_argmax` / `launch_sampling_
//      kernel` ladder, before the `pi.sampled` D2H + response build): if the
//      request carries a sampling program, take the IR path; else legacy.
//
// The actual compile + launch is delegated to an `IProgramBackend` that
// delta's JIT layer implements (NVRTC compile of charlie's per-kernel CUDA-C
// source → CUfunction, program cache keyed by bytecode hash, driver-API
// `cuLaunchKernel` of the kernel DAG). Until a backend is registered the
// runtime is inert (`try_run_sampling_program` returns `NoProgram`), so this
// module compiles and links standalone during W1.
//
// MVP scope (per design §3.3): single-slot, single-position, M=1 decode.
// Matrix / speculative-verify is phase 2.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "device_buffer.hpp"
#include "sampling_ir/codegen.hpp"  // ProgramManifest (v4 attach-binding manifest)

namespace pie_cuda_driver {
struct PersistentInputs;
}

namespace pie_cuda_driver::sampling_ir {

// ───────────────────────────── Input binding ─────────────────────────────
//
// Mirrors `pie_sampling_ir::Binding` (alpha's IR). The driver reads the
// binding class out of the compiled program's declared interface and the
// runtime resolves each input to a device pointer per fire.

enum class BindingClass : std::uint8_t {
    // Baked literal — no per-fire host source; codegen may inline the value
    // directly into the kernel, so the runtime never binds a buffer.
    Const = 0,
    // LM-head output rows for this fire (`ws.logits`). The runtime resolves
    // the row pointer from the workspace + the sample row index and hands it to
    // the backend as a raw `const void*`. The buffer is **bf16** — charlie's
    // codegen takes it as a `const unsigned short*` and converts in-kernel via
    // `__uint_as_float(h << 16)` (no cuda_bf16.h / NVRTC dep), so the runtime
    // does NOT up-convert to f32. The IR declares this input F32 logically
    // (BYTECODE.md §2.2); the bf16 storage is a codegen-side detail.
    Intrinsic = 1,
    // Kernel arg refreshed from the request payload each fire. No recompile —
    // the value rides in a stable device buffer whose *contents* are copied
    // per fire (same discipline as `PersistentInputs::sample_*`).
    HostSubmit = 2,
    // Supplied AFTER submit, before the first consuming kernel (e.g. mirostat
    // µ, grammar mask). If absent at launch time → SKIP (see RunStatus).
    HostLate = 3,
};

enum class IntrinsicKind : std::uint8_t {
    Logits = 0,     // bf16 [vocab] row from `ws.logits` at `sample_row`
    MtpLogits = 1,  // bf16 [vocab] row from `ws.logits` at the speculator DRAFT
                    // row (`mtp_draft_row`) — the MTP head's next-token logits.
    // #31 self-spec verify draft INPUT marker (NOT a ws.logits intrinsic): a
    // HostLate `[k]` i32 draft binding the resolver redirects to the refed
    // `pi.tokens + sample_row + 1`. Reuses this `intrinsic` field as the distinct
    // marker (cls==HostLate && intrinsic==SelfSpecDraftInput) so it survives the
    // BufferDecl→InputDecl projection to echo's resolver without folding to Late.
    SelfSpecDraftInput = 2,
};

// One resolved per-fire binding. Produced by the runtime from the program's
// declared interface + the live request payload + the sample plan.
struct ResolvedInput {
    std::uint32_t  input_id = 0;
    BindingClass   cls = BindingClass::Const;
    IntrinsicKind  intrinsic = IntrinsicKind::Logits;

    // Device pointer the kernel DAG should read for this input. For
    // `Intrinsic` it points into `ws.logits`; for `HostSubmit`/`HostLate` it
    // points into a stable scratch device buffer the runtime filled this
    // fire; for `Const` it is null (value inlined by codegen).
    const void*    device_ptr = nullptr;
    std::size_t    elem_count = 0;   // length in elements (not bytes)

    // False only for a `HostLate` input that was never supplied → the runtime
    // raises `SkippedLateBindMiss` and discards this program for the fire.
    bool           present = true;
};

// ───────────────────────────── Output binding ────────────────────────────
//
// Maps an IR output value to the response shape the executor already knows how
// to marshal. Reuses the `PerRequestOutput` fields / response-subpass layout.
//
// The enum values are the **frozen v2 `OutputKind` wire tags** (BYTECODE.md §1)
// so the backend can pass charlie's parsed `BufferDecl.output_kind` byte through
// to the runtime verbatim (`static_cast<OutputClass>(kind)`). Codegen/kernels
// are kind-agnostic (they emit by shape/dtype); the executor marshaling layer
// (this lane, L4) routes each output buffer into its `ForwardResponse` channel
// by `OutputClass`, and the host then picks the WIT `slot-output` variant.
enum class OutputClass : std::uint8_t {
    Token = 0,         // int (I32/U32) sampled id     → tokens slot (pi.sampled)
    Distribution = 1,  // (ids,probs) top-k            → dists slot
    Logits = 2,        // f32 [vocab]                  → logits_bytes slot
    Logprobs = 3,      // f32 [k] (scalar or per-row)  → logprobs slot
    Entropy = 4,       // f32 scalar                   → entropies slot
    Scalar = 5,        // f32 scalar                   → logprobs-shaped slot
    Embedding = 6,     // f32 [hidden] (reserved)      → not yet marshaled
};

struct DeclaredOutput {
    std::uint32_t value_id = 0;   // IR value id producing this output
    OutputClass   cls = OutputClass::Token;
    std::size_t   elem_count = 0; // declared length (1 for Token/Entropy, k/vocab otherwise)
};

// ─────────────────────────── Program interface ───────────────────────────
//
// The declared shape of a compiled program, surfaced by the backend (filled
// from charlie's codegen artifact). The runtime reads it to drive binding and
// marshaling. This is the *contract* echo consumes; charlie/delta own how it
// is produced.

struct InputDecl {
    std::uint32_t input_id = 0;
    BindingClass  cls = BindingClass::Const;
    IntrinsicKind intrinsic = IntrinsicKind::Logits;
    std::uint32_t host_key = 0;   // selector into the request's host-input table
    std::size_t   elem_count = 0; // expected length in elements
};

struct ProgramInterface {
    std::vector<InputDecl>      inputs;
    std::vector<DeclaredOutput> outputs;
};

// ──────────────────────── Backend (delta's JIT) ──────────────────────────
//
// Opaque handle to a compiled+cached program. The runtime never inspects it.
using ProgramHandle = std::uint64_t;
constexpr ProgramHandle kInvalidProgram = 0;

// Per-fire launch arguments resolved by the runtime and handed to the backend.
struct LaunchArgs {
    // Resolved inputs in `input_id` order. The backend binds each
    // `device_ptr` as the corresponding kernel argument.
    std::span<const ResolvedInput> inputs;
    // Device pointers the backend should write each declared output into,
    // in `outputs` order. Sized by the runtime from the program interface.
    std::span<void* const>         output_ptrs;
    // Logit row count being processed (MVP = 1). Drives grid sizing.
    int                            num_rows = 1;
    int                            vocab_size = 0;
    // PRNG offset for parity with the legacy sampler (`handled` counter).
    std::uint64_t                  prng_offset = 0;
    // Ambient per-row RNG seed S (Model B): device pointer to a [num_rows] u32
    // array = pi.sample_seed[sampling_row[r]]. Bound to the program's
    // BufferClass::RowSeed buffer, which Op::Rng reads at rowseed[r] (via
    // seed_eff_stream). Delivered out-of-band — the ambient seed is NOT an
    // Op::Input — so it decouples from the program's HostSubmit inputs and the
    // program is identical seeded-or-not. null/unused for programs with no
    // Op::Rng (e.g. argmax).
    const void*                    row_seeds = nullptr;
};

// Implemented by delta's JIT layer. Compile is lazy + cached by bytecode hash.
class IProgramBackend {
public:
    virtual ~IProgramBackend() = default;

    // Compile-or-fetch the program for `bytecode`. Returns `kInvalidProgram`
    // on a codegen/NVRTC failure (the runtime then fails loud).
    virtual ProgramHandle get_or_compile(std::span<const std::uint8_t> bytecode) = 0;

    // v4 compile-or-fetch: binding-free `bytecode` + its attach `manifest`
    // (which slot is the logits intrinsic vs a host tensor). An empty manifest
    // is the v3 path (delegates to the 1-arg form). Non-pure so mock backends
    // need no change; delta's `SamplingIrBackend` overrides it for real v4
    // (decode_v4 + lower_bytecode_v4, manifest folded into the cache key).
    virtual ProgramHandle get_or_compile(std::span<const std::uint8_t> bytecode,
                                         const ProgramManifest& /*manifest*/) {
        return get_or_compile(bytecode);
    }

    // Fire-and-forget prefetch (#11 TTFT): kick off the async NVRTC compile for
    // `bytecode` + `manifest` at admission — before the consuming fire — so the
    // PTX-gen overlaps the in-flight run-ahead steps off the critical path. The
    // consuming `get_or_compile` then finds it Ready (or waits on the in-flight
    // `shared_future`). Idempotent: dedup'd downstream by the program-cache key
    // (`program_identity_hash(bytecode, manifest)`), the SAME key as the compile
    // cache and #10's distinct-count, so prefetch ≡ compile ≡ dedup — no
    // divergence. Default no-op: backends without async compile (mocks) inherit
    // it unchanged; delta's `SamplingIrBackend` overrides it →
    // `JitEngine::prefetch_compile`. `bytecode + manifest` in / nothing out keeps
    // `IProgramBackend` backend-agnostic (no jit types leak).
    virtual void prefetch_compile(std::span<const std::uint8_t> /*bytecode*/,
                                  const ProgramManifest& /*manifest*/) {}

    // The program's declared input/output interface (from codegen).
    virtual const ProgramInterface& interface(ProgramHandle program) = 0;

    // Launch the kernel DAG on `stream`. Pure device-side work; no host sync.
    virtual void launch(ProgramHandle program,
                        const LaunchArgs& args,
                        cudaStream_t stream) = 0;

    // Human-readable reason for the most recent `kInvalidProgram` (decode /
    // lower / NVRTC compile failure), for diagnostics. Default: empty.
    virtual const std::string& last_error() const {
        static const std::string empty;
        return empty;
    }
};

// ─────────────────────────── Executor entry point ────────────────────────

enum class RunStatus : std::uint8_t {
    // No sampling program on the request → caller runs the legacy path.
    NoProgram = 0,
    // Program ran; outputs are in the runtime's per-fire scratch ready for
    // marshaling into `ForwardResponse`.
    Handled = 1,
    // A `HostLate` input was missing at launch time. Per spec §7.4 the fire is
    // discarded (no block, no default) and the host retries; fail loud.
    SkippedLateBindMiss = 2,
    // Compile/launch failed (codegen reject, NVRTC error, shape overflow).
    Failed = 3,
};

// One submit-bound host input for this fire: raw bytes pulled from the request
// carrier (`sampling_input_blob` + key/offset/len index). The runtime stages
// these into its own stable device buffer before binding — the WS1a sequential
// path. `host(submit-bound)` values (e.g. mirostat µ, a grammar mask) refresh
// per fire with no recompile; the inferlet supplies the next step's value as an
// ordinary submit-bound input on the following fire.
struct SubmitInput {
    std::uint32_t       key = 0;
    const std::uint8_t* data = nullptr;   // host pointer into the carrier blob
    std::size_t         len_bytes = 0;
};

// One late-bound input already resident on device this fire: either a CPU value
// written into a fixed buffer (`host(late-bound)`) or a prior program's
// on-device output (`output(output-ref)`) — the WS1b/WS4 async path. Matched to
// a program input by `key`; absent at launch → SkippedLateBindMiss (spec §7.4).
struct LateInput {
    std::uint32_t key = 0;
    const void*   device_ptr = nullptr;
    std::size_t   elem_count = 0;
};

// What the executor hands the runtime for one fire. References live workspace
// and request state; nothing is owned here.
struct FireContext {
    // Raw sampling-program bytecode (empty → NoProgram, legacy path).
    std::span<const std::uint8_t> program_bytecode;

    // v4 attach-binding manifest for `program_bytecode` (one InputBind per slot:
    // logits-intrinsic vs host-tensor). Empty = v3 self-binding bytecode (the
    // backend's 1-arg path). For a driver-baked standard sampler this is
    // `standard_sampler_program(kind, vocab).manifest`.
    ProgramManifest manifest;

    // Submit-bound host inputs for this fire (host bytes; the runtime stages
    // them to device). Empty when the program declares none.
    std::span<const SubmitInput> submit_inputs;

    // Host-late VALUES supplied as bytes for this fire (the WS1b correctness
    // path: value known by submit time, carried in the late-value table). The
    // runtime stages these like submit inputs; a `HostLate` binding resolves to
    // the staged bytes when present. Disjoint keys from `submit_inputs`.
    std::span<const SubmitInput> late_value_inputs;

    // Late-bound inputs already on device this fire (output-ref device alias).
    // A declared `HostLate` input with neither a staged value nor a device
    // entry here → SkippedLateBindMiss (true-async not yet arrived).
    std::span<const LateInput> late_inputs;

    // LM-head output for this fire: bf16 [rows, vocab] row-major base pointer
    // (i.e. `ws.logits.data()`). The runtime offsets by `sample_row * vocab` for
    // the `Intrinsic(Logits)` binding. Kept as an opaque pointer so the runtime
    // stays decoupled from the model workspace header. Null when the program has
    // no Intrinsic input.
    const void*                   logits = nullptr;
    // Stable per-fire device buffers; the Token output is written into
    // `pi->sampled[sample_row]` for the existing response marshaling.
    PersistentInputs*             pi = nullptr;

    int                           vocab_size = 0;
    // The single logit row to sample (MVP single-row), OR the FIRST of
    // `num_rows` contiguous sampling rows for the batched de-hardwiring path.
    int                           sample_row = 0;
    // #21 phase-2 mtp-logits: the `ws.logits` row an `IntrinsicKind::MtpLogits`
    // binding reads — the speculator DRAFT position (MTP head output), vs
    // `sample_row` for plain `Logits`. The MTP draft logits live in DRAFT ROWS
    // of `ws.logits` (not a separate buffer), so this is a row-select. -1 = unset
    // (no mtp-logits program this fire); the executor sets it from the MTP draft
    // layout. When unset the resolver falls back to `sample_row` (safe default).
    int                           mtp_draft_row = -1;
    // Number of contiguous sampling rows processed in one batched launch
    // (Task #4 [N,V] primitive). 1 = single-row (custom-program / MVP). The
    // bindings are base pointers (Intrinsic = block base at `sample_row`,
    // HostSubmit/output = `[num_rows]` bases); the batched kernel strides per
    // row internally, so only `num_rows` drives the grid.
    int                           num_rows = 1;
    std::uint64_t                 prng_offset = 0;
    // Ambient per-row RNG seed S (Model B): device ptr to a `[num_rows]` u32
    // block (= `pi->sample_seed[sampling_row[r]]`). Bound to `LaunchArgs.row_seeds`
    // for programs that emit `Op::Rng{stream}` over a `BufferClass::RowSeed`
    // (temp/min-p). Null for argmax / no-RNG programs (they ignore it).
    const void*                   row_seeds = nullptr;

    cudaStream_t                  stream = nullptr;
};

// The runtime facade. One instance per executor; holds the backend and the
// per-fire output scratch. Thread-compat matches the executor (single fire at
// a time per device).
class SamplingIrRuntime {
public:
    SamplingIrRuntime() = default;

    // Register delta's JIT backend. Until set, `try_run` returns NoProgram so
    // the executor always falls back to the legacy sampler (W1 inert state).
    void set_backend(IProgramBackend* backend) noexcept { backend_ = backend; }
    bool has_backend() const noexcept { return backend_ != nullptr; }

    // Mode-select + bind + launch. Does NOT do the D2H copy or response build;
    // on `Handled` the caller reads `last_outputs()` and marshals them with the
    // existing response path. Pure device-side launch on `ctx.stream`.
    RunStatus try_run(const FireContext& ctx);

    // Device pointers for each declared output of the program that last ran,
    // in interface order. Valid until the next `try_run`.
    std::span<void* const> last_output_ptrs() const noexcept { return out_ptrs_; }
    const ProgramInterface* last_interface() const noexcept { return last_iface_; }

private:
    // Stage all submit-bound inputs for one fire into `host_stage_` (a stable,
    // grown-on-demand device blob) and record each key's staged device pointer
    // in `staged_`. Async H2D on `stream`; ordered before the kernel launch.
    void stage_submit_inputs(const FireContext& ctx);

    IProgramBackend*          backend_ = nullptr;
    std::vector<ResolvedInput> resolved_;
    std::vector<void*>         out_ptrs_;
    const ProgramInterface*    last_iface_ = nullptr;

    // Submit-bound host-input device staging. One contiguous blob refreshed per
    // fire; `staged_` maps input key → (device offset, byte length) within it.
    DeviceBuffer<std::uint8_t> host_stage_;
    struct StagedSpan { std::size_t offset; std::size_t len; };
    std::unordered_map<std::uint32_t, StagedSpan> staged_;

    // Runtime-owned output scratch for multi-element outputs (e.g. spec-verify's
    // `-1`-sentinel `Vector<k>` Token). Single-token outputs write straight to
    // `pi->sampled`; vector/non-token outputs land here and are read back by the
    // executor via `last_output_ptrs()` + `last_interface()->outputs[i].elem_count`.
    DeviceBuffer<std::int32_t> out_scratch_;
};

}  // namespace pie_cuda_driver::sampling_ir

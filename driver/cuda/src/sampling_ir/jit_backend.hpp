#pragma once

// SamplingIrBackend — the CUDA-pod facade (lane L3 / delta).
//
// Implements echo's abstract `IProgramBackend` (runtime.hpp) by composing
// charlie's codegen (`lower()` / `lower_bytecode()`, codegen.hpp) with delta's
// NVRTC JIT (`jit::JitEngine`, jit.hpp). This is the single translation point
// between the two type families:
//
//   bytecode ──decode──► Program ──lower──► sampling_ir::KernelDAG
//             (reader)              (codegen)        │  adapt
//                                                    ▼
//                                       sampling_ir::jit::KernelDAG
//                                                    │  JitEngine
//                                                    ▼
//                                       NVRTC compile + cuLaunchKernel
//
// The runtime (echo) only ever sees `IProgramBackend`; codegen/jit types stay
// behind this facade. The executor registers an instance via
// `SamplingIrRuntime::set_backend(&backend)` at engine init.
//
// Cache: keyed by the program IDENTITY `fnv1a64(bytecode) ^ manifest_hash(manifest)`
// (v4 bytecode is binding-free, so distinct bindings ⇒ distinct program), XOR a
// process-constant bit for M=1-vs-M-batch lowering — i.e. `program_identity_hash`
// (program_identity.hpp) plus the batched bit. A program is
// decoded+lowered+compiled once; later fires reuse the compiled DAG and only
// rebind per-fire device pointers + launch-context scalars.
//
// MVP scope (design §3.3): single slot, single position, M=1 decode.

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "sampling_ir/codegen.hpp"
#include "sampling_ir/jit.hpp"
#include "sampling_ir/runtime.hpp"

namespace pie_cuda_driver::sampling_ir {

class SamplingIrBackend final : public IProgramBackend {
  public:
    // Constructs the JIT engine, which resolves the device arch from the
    // current CUDA context (the executor makes the primary context current
    // before registering the backend). `batched_lowering` selects charlie's
    // M>1 batched codegen (one grid=num_rows launch for the whole batch); the
    // default M=1 lowering processes one row per launch. Batched lowering at
    // num_rows=1 is equivalent to M=1, so it is safe to enable for batches.
    explicit SamplingIrBackend(bool batched_lowering = false);
    ~SamplingIrBackend() override = default;

    SamplingIrBackend(const SamplingIrBackend&) = delete;
    SamplingIrBackend& operator=(const SamplingIrBackend&) = delete;

    // Decode + lower + NVRTC-compile `bytecode`, cached by its hash. Returns
    // `kInvalidProgram` on a decode/lower/compile failure (see `last_error()`).
    // This v3/v2 (self-binding) form equals the manifest-aware overload with an
    // empty manifest.
    ProgramHandle get_or_compile(std::span<const std::uint8_t> bytecode) override;

    // v4 (binding-free) entry: `manifest` carries the per-slot bindings the v4
    // bytecode omits — which slot is the intrinsic logits vs a host tensor, plus
    // its host key/readiness — sourced from the EDSL bake (pie_standard_samplers.h).
    // An empty manifest selects the v3 self-binding decode, so this is a superset
    // of the 1-arg form; the manifest is folded into the program cache key.
    // Overrides echo's IProgramBackend 2-arg virtual (runtime.hpp).
    ProgramHandle get_or_compile(std::span<const std::uint8_t> bytecode,
                                 const ProgramManifest& manifest) override;

    // The compiled program's declared input/output interface (binding classes,
    // host keys, output kinds) for the runtime to drive binding + marshaling.
    const ProgramInterface& interface(ProgramHandle program) override;

    // Bind the per-fire resolved input/output device pointers + launch-context
    // scalars and launch the kernel DAG on `stream` (driver-API, no host sync).
    void launch(ProgramHandle program, const LaunchArgs& args,
                cudaStream_t stream) override;

    // #11 prefetch (ANY THREAD, fire-and-forget): warm the off-context PTX cache
    // for `(bytecode, manifest)` so a later `get_or_compile` finds it Ready (cache
    // hit) — the TTFT win. Idempotent + dedup'd by the SAME program_identity_hash
    // key as the compile cache / #10. Touches NO mutable backend state
    // (programs_/last_error_) — only the pure lowering + the mu_-guarded
    // `JitEngine::prefetch_compile` — so it races nothing on the context thread.
    void prefetch_compile(std::span<const std::uint8_t> bytecode,
                          const ProgramManifest& manifest) override;

    // Diagnostics: reason for the most recent `kInvalidProgram`, the NVRTC
    // target arch (e.g. "compute_89"), and whether a compiled program is on the
    // batched fast path or fell back to per-row M=1 (a batched-unsupported op
    // like Gather/Scatter/SortDesc — e.g. mirostat). Returns false for unknown
    // handles.
    const std::string& last_error() const override { return last_error_; }
    const std::string& arch() const { return engine_.arch(); }
    // #11 dedup metric: total NVRTC compiles run. A prefetch-warmed program and
    // its later on-demand get_or_compile share the off-context cache, so it
    // compiles exactly once — delta's load-test asserts cache-HIT via this.
    std::uint64_t compiles_run() const { return engine_.compiles_run(); }
    bool program_is_batched(ProgramHandle program) const;

  private:
    // Everything the runtime needs to bind + launch one compiled program.
    struct Entry {
        jit::CompiledProgram* prog = nullptr;  // owned by engine_'s cache
        ProgramInterface iface;
        // IR input id -> JIT BufferId, for external (logits/host) inputs.
        std::unordered_map<std::uint32_t, jit::BufferId> input_to_buffer;
        // JIT BufferId per declared output, indexed by output_index.
        std::vector<jit::BufferId> output_buffers;
        // Per-row byte stride of each external buffer (logits/host/output/row-seed),
        // for slicing an M=1 program to row r in the custom-batch replay. Logits is
        // bf16 (elem_count*2); the rest use the codegen BufferDecl byte size.
        std::unordered_map<jit::BufferId, std::uint64_t> row_stride_bytes;
        // True = batched fast path (one grid=N launch). False = M=1 fallback
        // (program has a batched-unsupported op; runs per-row, MVP num_rows==1).
        bool batched_mode = false;
        // Ambient RowSeed buffer (Model B Op::Rng): external, bound per-fire from
        // LaunchArgs.row_seeds (not an Op::Input, so not in `iface`/`input_to_buffer`).
        // Absent for programs with no Op::Rng (e.g. argmax).
        jit::BufferId row_seed_buffer = 0;
        bool has_row_seed = false;
    };

    // Shared compile path used by both get_or_compile overloads: lower (batched
    // with M=1 fallback) → adapt to the JIT DAG → NVRTC-compile → build the
    // declared interface + binding maps, for an already-decoded `program` under
    // cache key `keyed`. Returns the resulting (cached) handle.
    ProgramHandle compile_decoded(const Program& program, std::uint64_t keyed);

    // Decode bytecode → Program: route by the authoritative PSIR version (v4
    // binding-free, bindings from `manifest`; v2/v3 self-bind). `const` and writes
    // any error to `*err` (NOT last_error_) so the thread-safe prefetch path reuses
    // it without touching mutable backend state. Shared by get_or_compile + prefetch.
    bool decode_program(std::span<const std::uint8_t> bytecode,
                        const ProgramManifest& manifest, Program& out,
                        std::string* err) const;

    // Lower with the batched→M=1 fallback (codegen). Pure (reads batched_lowering_,
    // calls the free `lower()`); shared by compile_decoded + prefetch_compile.
    LowerResult lower_with_fallback(const Program& program, bool& batched_mode) const;

    // Adapt the codegen KernelDAG → jit::KernelDAG under compile-cache hash `keyed`.
    // Pure; shared by compile_decoded (→get_or_compile) + prefetch_compile.
    jit::KernelDAG build_jdag(const LowerResult& lr, std::uint64_t keyed) const;

    // Bind every external buffer (logits / host inputs / outputs / ambient
    // row-seed) for one fire, each sliced to logical row `row` (base + row*stride;
    // row 0 = the buffer base). Used both for the normal single launch (row 0) and
    // the custom-batch per-row replay of an M=1 program.
    void bind_external(Entry& e, const LaunchArgs& args, int row);

    jit::JitEngine engine_;
    std::unordered_map<ProgramHandle, Entry> programs_;
    std::string last_error_;
    bool batched_lowering_ = false;
};

}  // namespace pie_cuda_driver::sampling_ir

// C-ABI trampoline for the #11 in-proc prefetch FFI (alpha's InProcVTable
// `PrefetchFn`). Reconstructs a `ProgramManifest` from the `(kind, key)` wire
// with `ready = SubmitBound` — IDENTICAL to the submit path (executor.cpp:3244,
// the carrier conveys only `(kind, key)`) — so the prefetch-warm hash MATCHES the
// real submit-fire hash ⇒ compile-cache HIT ⇒ the TTFT win. `backend_ctx` is the
// `IProgramBackend*`; echo registers `&this` via `InProcVTable.register_prefetch`
// once at backend-ready. kind: 0=Logits, 1=Tensor(host_key), 2=MtpLogits.
extern "C" void pie_sampling_ir_prefetch_trampoline(
    void* backend_ctx, const std::uint8_t* bytecode, std::size_t bytecode_len,
    const std::uint8_t* binds_kind, const std::uint32_t* binds_key,
    std::size_t binds_len);

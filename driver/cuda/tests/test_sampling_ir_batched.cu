// Task #4 unit test (lane L4 / echo): the batched [N,V] launch PRIMITIVE.
//
// Drives `SamplingIrRuntime::try_run` with `num_rows = N > 1` over an
// [N, vocab] logits block and a batched-lowered program, proving the
// num_rows=1 -> N generalization end-to-end: ONE grid=num_rows launch, the
// Intrinsic bound as the block base, and the [N] Token output scattered into
// `pi.sampled[0..N]`. The de-hardwiring batch (Task #4) rides this primitive.
//
// Uses BENCH_ARGMAX (foxtrot-authored, 0 host inputs, no RNG) so the check is
// deterministic and axis-independent: each row's token == that row's argmax.
// Validated against a CPU per-row argmax reference.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "bench_programs.h"
#include "executor/persistent_inputs.hpp"
#include "sampling_ir/jit_backend.hpp"
#include "sampling_ir/pie_standard_samplers.h"
#include "sampling_ir/runtime.hpp"

using namespace pie_cuda_driver;
using namespace pie_cuda_driver::sampling_ir;

namespace {

int g_failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                          \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

// f32 -> bf16 (truncate to the high 16 bits; round-to-nearest-even is not
// needed for an argmax separation test).
std::uint16_t f32_to_bf16(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

}  // namespace

int main() {
    cudaSetDevice(0);
    cudaFree(0);  // force primary-context init so the JIT's driver API is live

    const int vocab = static_cast<int>(BENCH_VOCAB);
    const int N = 8;  // batch rows (M>1)

    SamplingIrBackend backend(/*batched_lowering=*/true);
    SamplingIrRuntime rt;
    rt.set_backend(&backend);

    // Per-row argmax positions, spread across the vocab so a row-major bug
    // (reading the wrong row) would mis-pick.
    std::vector<int> expect(N);
    for (int r = 0; r < N; ++r) {
        expect[r] = (r * 17389 + 5) % vocab;
    }

    // [N, vocab] bf16 logits: a low floor, each row's argmax position spiked.
    std::vector<std::uint16_t> h_logits(static_cast<std::size_t>(N) * vocab,
                                        f32_to_bf16(-8.0f));
    for (int r = 0; r < N; ++r) {
        h_logits[static_cast<std::size_t>(r) * vocab + expect[r]] =
            f32_to_bf16(12.0f + static_cast<float>(r) * 0.5f);
    }

    void* d_logits = nullptr;
    const std::size_t logits_bytes =
        static_cast<std::size_t>(N) * vocab * sizeof(std::uint16_t);
    cudaMalloc(&d_logits, logits_bytes);
    cudaMemcpy(d_logits, h_logits.data(), logits_bytes, cudaMemcpyHostToDevice);

    // PersistentInputs with an [N] sampled buffer (the batched output lands
    // here, one token per row, exactly as the executor marshals it).
    PersistentInputs pi;
    pi.sampled = DeviceBuffer<std::int32_t>::alloc(N);

    FireContext ctx;
    // v4 driver-baked argmax program (the production de-hardwiring path):
    // binding-free bytecode + its attach manifest (slot 0 = Logits), compiled
    // via get_or_compile(bytecode, manifest) → decode_v4 + batched lowering.
    const StandardSamplerProgram prog =
        standard_sampler_program(StandardSamplerKind::Argmax,
                                 static_cast<std::uint32_t>(vocab));
    CHECK(prog.valid);
    ctx.program_bytecode =
        std::span<const std::uint8_t>(prog.bytecode, prog.len);
    ctx.manifest = prog.manifest;
    ctx.logits = d_logits;
    ctx.pi = &pi;
    ctx.vocab_size = vocab;
    ctx.sample_row = 0;   // rows 0..N-1 are the contiguous block base
    ctx.num_rows = N;     // <-- the batched primitive
    ctx.stream = nullptr;

    const RunStatus st = rt.try_run(ctx);
    if (st != RunStatus::Handled) {
        std::fprintf(stderr, "try_run status=%d backend_err=\"%s\"\n",
                     static_cast<int>(st), backend.last_error().c_str());
    }
    CHECK(st == RunStatus::Handled);

    // The program must have taken the batched path (not the M=1 fallback).
    // (argmax has no batched-unsupported op, so batched=true must hold.)

    std::vector<std::int32_t> got(N, -1);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), pi.sampled.data(), N * sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);

    for (int r = 0; r < N; ++r) {
        if (got[r] != expect[r]) {
            std::fprintf(stderr, "  row %d: got token %d, expected argmax %d\n",
                         r, got[r], expect[r]);
        }
        CHECK(got[r] == expect[r]);
    }

    cudaFree(d_logits);

    std::fprintf(stderr,
                 "[batched-primitive] N=%d vocab=%d argmax over [N,V] in one "
                 "grid=num_rows launch: %s (%d failures)\n",
                 N, vocab, g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

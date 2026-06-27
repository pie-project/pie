// WS4 part-1 test (lane L4 / echo): drive charlie's greedy speculative-verify
// program through `SamplingIrRuntime::try_run` — the executor capability for
// binding a matrix `Intrinsic(Logits)` verify block + a submit-bound draft
// vector and reading the `-1`-sentinel `Vector<k+1>` Token accept-prefix output.
//
// Proves the runtime's matrix-logits binding + multi-element output handling
// end-to-end against a CPU greedy verifier (the same reference charlie's
// codegen test uses), across accept/reject-at-0/mid/last cases. Uses the frozen
// `GV_SPECGREEDY` golden bytecode; all shapes/keys are read from the compiled
// program interface (no hardcoding).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "executor/persistent_inputs.hpp"
#include "sampling_ir/jit_backend.hpp"
#include "sampling_ir/runtime.hpp"
#include "sampling_ir_golden_bytecode.h"

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

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}
float bf16_round(float f) {
    return __bfloat162float(__float2bfloat16(f));
}

// CPU greedy verifier (mirrors charlie's reference): per-row argmax over the
// bf16-rounded verify block, accept run vs drafts, emit prefix + -1 sentinels.
std::vector<std::int32_t> cpu_greedy(int kp1, int k, int V,
                                     const std::vector<float>& tlog,
                                     const std::vector<std::int32_t>& draft) {
    std::vector<std::int32_t> ttok(kp1);
    for (int r = 0; r < kp1; ++r) {
        int ai = 0; float mv = -1e30f;
        for (int j = 0; j < V; ++j) {
            float v = bf16_round(tlog[r * V + j]);
            if (v > mv) { mv = v; ai = j; }
        }
        ttok[r] = ai;
    }
    int alen = 0;
    while (alen < k && ttok[alen] == draft[alen]) ++alen;
    std::vector<std::int32_t> out(kp1, -1);
    for (int i = 0; i <= alen; ++i) out[i] = ttok[i];
    return out;
}

}  // namespace

int main() {
    cudaSetDevice(0);
    cudaFree(nullptr);

    {
        SamplingIrBackend backend;
        SamplingIrRuntime rt;
        rt.set_backend(&backend);

        PersistentInputs pi;
        pi.sampled = DeviceBuffer<std::int32_t>::alloc(1);

        std::span<const std::uint8_t> bytecode(
            reinterpret_cast<const std::uint8_t*>(GV_SPECGREEDY), sizeof(GV_SPECGREEDY));

        // Derive shapes + the draft submit key from the compiled interface.
        ProgramHandle h = backend.get_or_compile(bytecode);
        if (h == kInvalidProgram)
            std::fprintf(stderr, "compile failed: %s\n", backend.last_error().c_str());
        CHECK(h != kInvalidProgram);
        const ProgramInterface& iface = backend.interface(h);

        std::size_t mat_elems = 0, draft_elems = 0, out_elems = 0;
        std::uint32_t draft_key = 0;
        for (const InputDecl& in : iface.inputs) {
            if (in.cls == BindingClass::Intrinsic) mat_elems = in.elem_count;
            if (in.cls == BindingClass::HostSubmit) { draft_elems = in.elem_count; draft_key = in.host_key; }
        }
        for (const DeclaredOutput& o : iface.outputs)
            if (o.cls == OutputClass::Token) out_elems = o.elem_count;

        const int kp1 = static_cast<int>(out_elems);
        const int V = kp1 > 0 ? static_cast<int>(mat_elems / kp1) : 0;
        const int k = kp1 - 1;
        std::fprintf(stderr, "spec program: kp1=%d V=%d k=%d draft_elems=%zu key=%u\n",
                     kp1, V, k, draft_elems, draft_key);
        CHECK(kp1 >= 2 && V > 0);
        CHECK(static_cast<int>(draft_elems) >= k);

        CUdeviceptr d_logits = 0;
        cuMemAlloc(&d_logits, static_cast<std::size_t>(kp1) * V * sizeof(std::uint16_t));

        std::mt19937 rng(53);
        std::uniform_real_distribution<float> d(-5.f, 5.f);

        auto run_case = [&](const char* name, int reject_at) {
            std::vector<float> tlog(static_cast<std::size_t>(kp1) * V);
            for (auto& v : tlog) v = d(rng);
            // Per-row argmax (bf16-rounded) → the "target" tokens.
            std::vector<std::int32_t> ttok(kp1);
            for (int r = 0; r < kp1; ++r) {
                int ai = 0; float mv = -1e30f;
                for (int j = 0; j < V; ++j) {
                    float v = bf16_round(tlog[r * V + j]);
                    if (v > mv) { mv = v; ai = j; }
                }
                ttok[r] = ai;
            }
            // Drafts = target tokens (all-accept), then force a reject point.
            std::vector<std::int32_t> draft(draft_elems, 0);
            for (int i = 0; i < k; ++i) draft[i] = ttok[i];
            if (reject_at >= 0 && reject_at < k) draft[reject_at] = (ttok[reject_at] + 1) % V;

            // Upload the bf16 verify-block matrix.
            std::vector<std::uint16_t> hb(static_cast<std::size_t>(kp1) * V);
            for (std::size_t i = 0; i < hb.size(); ++i) hb[i] = f32_to_bf16(tlog[i]);
            cuMemcpyHtoD(d_logits, hb.data(), hb.size() * sizeof(std::uint16_t));

            SubmitInput si;
            si.key = draft_key;
            si.data = reinterpret_cast<const std::uint8_t*>(draft.data());
            si.len_bytes = draft.size() * sizeof(std::int32_t);

            FireContext ctx;
            ctx.program_bytecode = bytecode;
            ctx.submit_inputs = std::span<const SubmitInput>(&si, 1);
            ctx.logits = reinterpret_cast<const void*>(d_logits);  // matrix base
            ctx.pi = &pi;
            ctx.vocab_size = V;
            ctx.sample_row = 0;
            ctx.prng_offset = 0;
            ctx.stream = nullptr;

            const RunStatus st = rt.try_run(ctx);
            CHECK(st == RunStatus::Handled);
            cudaDeviceSynchronize();

            // Read the Vector{kp1} Token output from the runtime's scratch.
            std::span<void* const> outs = rt.last_output_ptrs();
            CHECK(!outs.empty());
            std::vector<std::int32_t> got(kp1, -99);
            cudaMemcpy(got.data(), outs[0], kp1 * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost);

            std::vector<std::int32_t> ref = cpu_greedy(kp1, k, V, tlog, draft);
            bool ok = (got == ref);
            if (!ok) {
                std::fprintf(stderr, "  %s: got=[", name);
                for (int x : got) std::fprintf(stderr, "%d ", x);
                std::fprintf(stderr, "] ref=[");
                for (int x : ref) std::fprintf(stderr, "%d ", x);
                std::fprintf(stderr, "]\n");
            } else {
                std::fprintf(stderr, "  %s: OK (accept_prefix matches)\n", name);
            }
            CHECK(ok);
        };

        run_case("all-accepted", -1);
        run_case("reject-at-0", 0);
        run_case("reject-mid", k / 2);
        run_case("reject-at-last", k - 1);

        cuMemFree(d_logits);
    }

    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_spec_verify: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_spec_verify: %d failure(s)\n", g_failures);
    return 1;
}

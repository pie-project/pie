// WS5 removal-gate parity (lane L7 / hotel): the **IR-through-the-real-backend ≡
// the hardwired kernel** gate that must be green BEFORE charlie/echo delete any
// `sample_temp.cu` path.
//
// echo's `test_sampling_ir_executor_parity` proves IR-argmax == `launch_argmax_bf16`.
// This extends that to the **temperature + min-p** sampler: the GV_TEMP / GV_MINP
// programs, compiled + launched through echo's `SamplingIrBackend` exactly as the
// executor binds them, must produce the SAME token as the production
// `launch_sample_temp_bf16` kernel over the same bf16 logits + seed. They share the
// SplitMix64 + seed×column Gumbel-max RNG (the frozen parity contract), so the match
// is **token-exact**.
//
// Gate semantics: green here ⇒ removing `sample_temp.cu` is behavior-preserving for
// temperature / min-p decode. (top-k/top-p vs FlashInfer is a separate gate —
// FlashInfer uses a different PRNG, so that removal is distribution-equivalent, not
// token-identical; tracked separately.)
//
// Self-contained: echo's backend (codegen/reader/jit) + the production sample_temp
// kernel + driver API. Reuses charlie's authoritative golden bytecode (GV_TEMP,
// GV_MINP) so the IR path is the real encoder output, not hand-rolled.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels/sample_temp.hpp"
#include "sampling_ir/jit_backend.hpp"
#include "sampling_ir/runtime.hpp"
#include "sampling_ir_golden_bytecode.h"
#include "sampler_reference.hpp"

using namespace pie_cuda_driver::sampling_ir;
namespace ref = pie_sampler_ref;

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

void cu_check(CUresult res, const char* expr, int line) {
    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(res, &name);
        std::fprintf(stderr, "FATAL: %s:%d: %s -> %s\n", __FILE__, line, expr,
                     name ? name : "?");
        std::exit(2);
    }
}
#define CU(expr) cu_check((expr), #expr, __LINE__)

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}
float bf16_to_f32(std::uint16_t b) {
    std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Bind a program's Intrinsic logits row + single HostSubmit seed and launch one
// row through the backend (M=1, exactly as the executor binds it). Returns the
// IR token. Shared by the temp/min-p (vs kernel) and top-k/top-p (vs golden) gates.
std::int32_t launch_ir_row(SamplingIrBackend& backend, ProgramHandle h,
                           int logits_input_id, int seed_input_id, int vocab,
                           CUdeviceptr d_logits_row, CUdeviceptr d_seed,
                           CUdeviceptr d_token_ir) {
    ResolvedInput rins[2];
    rins[0].input_id = static_cast<std::uint32_t>(logits_input_id);
    rins[0].cls = BindingClass::Intrinsic;
    rins[0].intrinsic = IntrinsicKind::Logits;
    rins[0].device_ptr = reinterpret_cast<const void*>(d_logits_row);
    rins[0].elem_count = static_cast<std::size_t>(vocab);
    rins[0].present = true;
    rins[1].input_id = static_cast<std::uint32_t>(seed_input_id);
    rins[1].cls = BindingClass::HostSubmit;
    rins[1].device_ptr = reinterpret_cast<const void*>(d_seed);
    rins[1].elem_count = 1;
    rins[1].present = true;

    void* outs[1] = {reinterpret_cast<void*>(d_token_ir)};
    LaunchArgs args;
    args.inputs = std::span<const ResolvedInput>(rins, 2);
    args.output_ptrs = std::span<void* const>(outs, 1);
    args.num_rows = 1;
    args.vocab_size = vocab;
    args.prng_offset = 0;
    backend.launch(h, args, /*stream=*/nullptr);
    CU(cuCtxSynchronize());
    std::int32_t tok = -1;
    CU(cuMemcpyDtoH(&tok, d_token_ir, sizeof(std::int32_t)));
    return tok;
}

bool resolve_io_ids(SamplingIrBackend& backend, ProgramHandle h,
                    int* logits_id, int* seed_id) {
    const ProgramInterface& iface = backend.interface(h);
    *logits_id = -1;
    *seed_id = -1;
    for (const InputDecl& in : iface.inputs) {
        if (in.cls == BindingClass::Intrinsic) *logits_id = static_cast<int>(in.input_id);
        if (in.cls == BindingClass::HostSubmit) *seed_id = static_cast<int>(in.input_id);
    }
    return *logits_id >= 0 && *seed_id >= 0;
}

// Run one sampler program through SamplingIrBackend over `num_rows` bf16 logit
// rows (one launch per row, M=1, exactly as the executor invokes it), binding
// the single HostSubmit input as the per-row seed. Returns the IR tokens.
//
// `compare` runs the hardwired `launch_sample_temp_bf16` over the same rows with
// the given T/min_p and asserts every token matches.
void run_gate(const char* name,
              const unsigned char* bytecode, std::size_t bytecode_len,
              int vocab, float temperature, float min_p) {
    SamplingIrBackend backend;
    ProgramHandle h = backend.get_or_compile(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(bytecode), bytecode_len));
    if (h == kInvalidProgram) {
        std::fprintf(stderr, "[%s] get_or_compile failed: %s\n", name,
                     backend.last_error().c_str());
    }
    CHECK(h != kInvalidProgram);
    if (h == kInvalidProgram) return;

    // Locate the single HostSubmit input id (the seed) from the declared
    // interface, so we bind by class rather than guessing the index.
    const ProgramInterface& iface = backend.interface(h);
    int seed_input_id = -1;
    int logits_input_id = -1;
    for (const InputDecl& in : iface.inputs) {
        if (in.cls == BindingClass::Intrinsic) logits_input_id = static_cast<int>(in.input_id);
        if (in.cls == BindingClass::HostSubmit) seed_input_id = static_cast<int>(in.input_id);
    }
    CHECK(logits_input_id >= 0);
    CHECK(seed_input_id >= 0);
    if (logits_input_id < 0 || seed_input_id < 0) return;

    const int num_rows = 128;
    std::mt19937 rng(0x5A11D00Du);
    std::normal_distribution<float> nd(0.f, 3.f);

    // Random bf16 logits, distinct per row.
    std::vector<std::uint16_t> h_logits(static_cast<std::size_t>(num_rows) * vocab);
    std::vector<std::uint32_t> h_seeds(num_rows);
    for (int r = 0; r < num_rows; ++r) {
        h_seeds[r] = 0x1234abcdu * static_cast<std::uint32_t>(r + 1) + 0x9E3779B9u;
        for (int j = 0; j < vocab; ++j) {
            h_logits[static_cast<std::size_t>(r) * vocab + j] = f32_to_bf16(nd(rng));
        }
    }

    CUdeviceptr d_logits = 0;
    const std::size_t logits_bytes =
        static_cast<std::size_t>(num_rows) * vocab * sizeof(std::uint16_t);
    CU(cuMemAlloc(&d_logits, logits_bytes));
    CU(cuMemcpyHtoD(d_logits, h_logits.data(), logits_bytes));

    CUdeviceptr d_seed = 0;   // single u32 seed buffer, refreshed per row
    CU(cuMemAlloc(&d_seed, sizeof(std::uint32_t)));
    CUdeviceptr d_token_ir = 0;
    CU(cuMemAlloc(&d_token_ir, sizeof(std::int32_t)));

    // ── IR path: SamplingIrBackend per row ──
    std::vector<std::int32_t> ir(num_rows);
    for (int r = 0; r < num_rows; ++r) {
        CU(cuMemcpyHtoD(d_seed, &h_seeds[r], sizeof(std::uint32_t)));

        ResolvedInput rins[2];
        rins[0].input_id = static_cast<std::uint32_t>(logits_input_id);
        rins[0].cls = BindingClass::Intrinsic;
        rins[0].intrinsic = IntrinsicKind::Logits;
        rins[0].device_ptr = reinterpret_cast<const void*>(
            d_logits + static_cast<std::size_t>(r) * vocab * sizeof(std::uint16_t));
        rins[0].elem_count = static_cast<std::size_t>(vocab);
        rins[0].present = true;

        rins[1].input_id = static_cast<std::uint32_t>(seed_input_id);
        rins[1].cls = BindingClass::HostSubmit;
        rins[1].device_ptr = reinterpret_cast<const void*>(d_seed);
        rins[1].elem_count = 1;
        rins[1].present = true;

        void* outs[1] = {reinterpret_cast<void*>(d_token_ir)};
        LaunchArgs args;
        args.inputs = std::span<const ResolvedInput>(rins, 2);
        args.output_ptrs = std::span<void* const>(outs, 1);
        args.num_rows = 1;
        args.vocab_size = vocab;
        args.prng_offset = 0;
        backend.launch(h, args, /*stream=*/nullptr);
        CU(cuCtxSynchronize());
        CU(cuMemcpyDtoH(&ir[r], d_token_ir, sizeof(std::int32_t)));
    }

    // ── Legacy path: launch_sample_temp_bf16 over all rows at once ──
    std::vector<float> h_T(num_rows, temperature), h_minp(num_rows, min_p);
    CUdeviceptr d_T = 0, d_minp = 0, d_seeds_arr = 0;
    std::int32_t* d_tokens_legacy = nullptr;
    CU(cuMemAlloc(&d_T, num_rows * sizeof(float)));
    CU(cuMemAlloc(&d_minp, num_rows * sizeof(float)));
    CU(cuMemAlloc(&d_seeds_arr, num_rows * sizeof(std::uint32_t)));
    cudaMalloc(&d_tokens_legacy, num_rows * sizeof(std::int32_t));
    CU(cuMemcpyHtoD(d_T, h_T.data(), num_rows * sizeof(float)));
    CU(cuMemcpyHtoD(d_minp, h_minp.data(), num_rows * sizeof(float)));
    CU(cuMemcpyHtoD(d_seeds_arr, h_seeds.data(), num_rows * sizeof(std::uint32_t)));

    pie_cuda_driver::kernels::launch_sample_temp_bf16(
        reinterpret_cast<const void*>(d_logits),
        reinterpret_cast<const float*>(d_T),
        reinterpret_cast<const float*>(d_minp),
        reinterpret_cast<const std::uint32_t*>(d_seeds_arr),
        d_tokens_legacy, num_rows, vocab, /*stream=*/nullptr);
    cudaDeviceSynchronize();
    std::vector<std::int32_t> legacy(num_rows);
    cudaMemcpy(legacy.data(), d_tokens_legacy, num_rows * sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);

    int mism = 0;
    for (int r = 0; r < num_rows; ++r) {
        if (ir[r] != legacy[r]) {
            if (mism < 8) {
                std::fprintf(stderr, "  [%s] MISMATCH row %d: ir=%d legacy=%d\n",
                             name, r, ir[r], legacy[r]);
            }
            ++mism;
        }
    }
    std::fprintf(stderr,
                 "[%s] IR-backend vs launch_sample_temp_bf16: rows=%d vocab=%d "
                 "T=%.2f min_p=%.2f mismatches=%d\n",
                 name, num_rows, vocab, temperature, min_p, mism);
    CHECK(mism == 0);

    cudaFree(d_tokens_legacy);
    CU(cuMemFree(d_T)); CU(cuMemFree(d_minp)); CU(cuMemFree(d_seeds_arr));
    CU(cuMemFree(d_token_ir)); CU(cuMemFree(d_seed)); CU(cuMemFree(d_logits));
}

// top-k / top-p removal gate. FlashInfer uses a different PRNG (philox), so the
// IR path is NOT token-identical to `sample_flashinfer` — the removal of that
// kernel is distribution-equivalent, not behavior-preserving. The authoritative
// oracle is therefore the **golden** (`sampler_reference.hpp`), which shares the
// IR's SplitMix64 Gumbel scheme → token-exact. This gate proves the IR top-k /
// top-p sampler is correct per spec; the distributional-equivalence-vs-FlashInfer
// check + Oracle sign-off on the token-stream change is tracked separately.
void run_gate_vs_golden(const char* name,
                        const unsigned char* bytecode, std::size_t bytecode_len,
                        int vocab, bool is_topk, int k, float p, float T) {
    SamplingIrBackend backend;
    ProgramHandle h = backend.get_or_compile(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(bytecode), bytecode_len));
    if (h == kInvalidProgram) {
        std::fprintf(stderr, "[%s] get_or_compile failed: %s\n", name,
                     backend.last_error().c_str());
    }
    CHECK(h != kInvalidProgram);
    if (h == kInvalidProgram) return;

    int logits_id = -1, seed_id = -1;
    CHECK(resolve_io_ids(backend, h, &logits_id, &seed_id));
    if (logits_id < 0 || seed_id < 0) return;

    const int num_rows = 128;
    std::mt19937 rng(0xBEEF1234u);
    std::normal_distribution<float> nd(0.f, 3.f);

    CUdeviceptr d_logits = 0;
    const std::size_t logits_bytes =
        static_cast<std::size_t>(num_rows) * vocab * sizeof(std::uint16_t);
    CU(cuMemAlloc(&d_logits, logits_bytes));
    CUdeviceptr d_seed = 0, d_token_ir = 0;
    CU(cuMemAlloc(&d_seed, sizeof(std::uint32_t)));
    CU(cuMemAlloc(&d_token_ir, sizeof(std::int32_t)));

    int mism = 0;
    std::vector<std::uint16_t> h_row(vocab);
    std::vector<float> ref_row(vocab);
    for (int r = 0; r < num_rows; ++r) {
        const std::uint32_t seed = 0x1234abcdu * static_cast<std::uint32_t>(r + 1) + 0x9E3779B9u;
        for (int j = 0; j < vocab; ++j) {
            // Distinct logits (small per-index ramp) so bf16 rounding can't
            // collide two probabilities at the nucleus boundary — matching the
            // codegen test's methodology. Removes tie-break ambiguity between
            // the golden's prefix-cumsum and the IR's CummassLe pivot.
            const std::uint16_t b = f32_to_bf16(nd(rng) + 1e-3f * static_cast<float>(j));
            h_row[j] = b;
            ref_row[j] = bf16_to_f32(b);
        }
        CU(cuMemcpyHtoD(d_logits, h_row.data(), vocab * sizeof(std::uint16_t)));
        CU(cuMemcpyHtoD(d_seed, &seed, sizeof(std::uint32_t)));

        const std::int32_t ir_tok = launch_ir_row(
            backend, h, logits_id, seed_id, vocab, d_logits, d_seed, d_token_ir);

        // Golden mask, then Gumbel-max selection at temperature T.
        //  * top-k: rank over logits (temperature-invariant).
        //  * top-p: nucleus over the **un-temperatured** softmax (T=1) — the IR
        //    program (GV_TOPP) computes the nucleus on raw softmax and applies T
        //    only in the final score, matching charlie's codegen reference.
        const std::vector<bool> keep =
            is_topk ? ref::top_k_mask(ref_row, k)
                    : ref::top_p_mask(ref_row, p, /*temperature=*/1.0f);
        const int golden_tok = ref::gumbel_argmax_masked(ref_row, keep, T, seed).token;

        if (ir_tok != golden_tok) {
            if (mism < 8) {
                std::fprintf(stderr, "  [%s] MISMATCH row %d: ir=%d golden=%d\n",
                             name, r, ir_tok, golden_tok);
            }
            ++mism;
        }
    }
    std::fprintf(stderr,
                 "[%s] IR-backend vs golden (Gumbel): rows=%d vocab=%d %s mismatches=%d\n",
                 name, num_rows, vocab, is_topk ? "k" : "p", mism);
    CHECK(mism == 0);

    CU(cuMemFree(d_token_ir)); CU(cuMemFree(d_seed)); CU(cuMemFree(d_logits));
}

}  // namespace

int main() {
    CU(cuInit(0));
    CUdevice dev = 0;
    CU(cuDeviceGet(&dev, 0));
    CUcontext ctx = nullptr;
    CU(cuCtxCreate(&ctx, nullptr, 0, dev));

    {
        // GV_TEMP: temperature=0.8, no min-p (baked consts in the program).
        run_gate("temp", GV_TEMP, sizeof(GV_TEMP), /*vocab=*/128,
                 /*T=*/0.8f, /*min_p=*/0.0f);
        // GV_MINP: temperature=0.7, min_p=0.1 (logit-space, sample_temp parity).
        run_gate("min-p", GV_MINP, sizeof(GV_MINP), /*vocab=*/256,
                 /*T=*/0.7f, /*min_p=*/0.1f);

        // top-k / top-p: oracle is the golden (FlashInfer's PRNG differs, so not
        // token-comparable). GV_TOPK: k=10, T=0.9; GV_TOPP: p=0.9, T=0.9.
        run_gate_vs_golden("top-k", GV_TOPK, sizeof(GV_TOPK), /*vocab=*/256,
                           /*is_topk=*/true, /*k=*/10, /*p=*/0.0f, /*T=*/0.9f);
        run_gate_vs_golden("top-p", GV_TOPP, sizeof(GV_TOPP), /*vocab=*/256,
                           /*is_topk=*/false, /*k=*/0, /*p=*/0.9f, /*T=*/0.9f);
    }

    CU(cuCtxDestroy(ctx));
    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_removal_gate: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_removal_gate: %d failure(s)\n", g_failures);
    return 1;
}

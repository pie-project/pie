// Capstone cross-check (lane L7 / hotel): the **eval-mock ≡ real-driver** proof.
//
// The 4090-free eval-mock executor (runtime/tests/common/mock_device.rs) runs a
// sampling-program through the CPU `eval` interpreter over deterministic
// `synthetic_logits(req_id, vocab)`. This test feeds the EXACT same synthetic
// rows to the REAL GPU executor (echo's `SamplingIrBackend`, bound exactly as
// executor.cpp binds it) and asserts the GPU token equals the CPU golden's —
// closing the chain:
//
//   GPU IR executor  ≡  C++ golden (sampler_reference.hpp)  ≡  Rust eval
//   (eval_parity)    ≡  eval-mock     — all over identical synthetic logits.
//
// Consequence: CI can validate IR-sampler correctness on CPU (no GPU) and
// trust it matches the GPU bit-for-bit at M=1 decode argmax.
//
// Self-contained: echo's backend (codegen/reader/jit) + hotel's golden + driver
// API. Mirrors test_sampling_ir_executor_parity.cu's binding exactly.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "sampling_ir/jit_backend.hpp"
#include "sampling_ir/runtime.hpp"
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

void put_u8(std::vector<std::uint8_t>& b, std::uint8_t v) { b.push_back(v); }
void put_u16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(v & 0xff);
    b.push_back((v >> 8) & 0xff);
}
void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xff);
}

// PSIR v2 greedy argmax over intrinsic logits Vector{vocab} → Token (identical
// to the executor-parity test's bytecode).
std::vector<std::uint8_t> build_argmax_v2(std::uint32_t vocab) {
    std::vector<std::uint8_t> b;
    b.insert(b.end(), {'P', 'S', 'I', 'R'});
    put_u16(b, 2);
    put_u16(b, 0);
    put_u32(b, 1);
    put_u32(b, 1);
    put_u32(b, 0);
    put_u8(b, 0);
    put_u8(b, 1);
    put_u32(b, vocab);
    put_u32(b, 0);
    put_u8(b, 1);
    put_u8(b, 0);
    put_u32(b, 1);
    put_u8(b, 0x33);  // ReduceArgmax
    put_u32(b, 0);
    put_u32(b, 1);
    put_u32(b, 1);
    put_u8(b, 0);  // Token
    return b;
}

// f32 → bf16 by truncation — matches test_sampling_ir_executor_parity.cu so the
// GPU and the golden see byte-identical inputs.
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

// EXACT C++ port of the eval-mock's synthetic_logits(req_id, vocab):
// SplitMix64(seed = req_id ^ 0xA5A5A5A5, column j), high-24-bit uniform → [-4, 4].
float synth_logit(std::uint64_t req_id, std::uint32_t j) {
    std::uint64_t seed = req_id ^ 0xA5A5'A5A5ull;
    std::uint64_t x =
        seed + 0x9E37'79B9'7F4A'7C15ull * (static_cast<std::uint64_t>(j) + 1);
    x ^= x >> 27; x *= 0x3C79'AC49'2BA7'B653ull;
    x ^= x >> 33; x *= 0x1C69'B3F7'4AC4'AE35ull;
    x ^= x >> 27;
    const std::uint32_t bits = static_cast<std::uint32_t>(x >> 40);
    const float u = (static_cast<float>(bits) + 0.5f) * (1.0f / 16'777'216.0f);
    return (u * 2.0f - 1.0f) * 4.0f;
}

}  // namespace

int main() {
    CU(cuInit(0));
    CUdevice dev = 0;
    CU(cuDeviceGet(&dev, 0));
    CUcontext ctx = nullptr;
    CU(cuCtxCreate(&ctx, nullptr, 0, dev));

    {
        SamplingIrBackend backend;
        const int num_rows = 64;
        const std::uint32_t vocab = 4096;

        ProgramHandle h = backend.get_or_compile(build_argmax_v2(vocab));
        if (h == kInvalidProgram) {
            std::fprintf(stderr, "get_or_compile failed: %s\n",
                         backend.last_error().c_str());
        }
        CHECK(h != kInvalidProgram);

        // Per row r = req_id: build the eval-mock's synthetic logits, round to
        // bf16 (as the GPU reads them), keep the bf16→f32 expansion for the
        // golden so both sides see identical values.
        std::vector<std::uint16_t> h_logits(static_cast<std::size_t>(num_rows) * vocab);
        std::vector<std::vector<float>> ref_rows(num_rows, std::vector<float>(vocab));
        for (int r = 0; r < num_rows; ++r) {
            for (std::uint32_t j = 0; j < vocab; ++j) {
                const std::uint16_t b = f32_to_bf16(synth_logit(static_cast<std::uint64_t>(r), j));
                h_logits[static_cast<std::size_t>(r) * vocab + j] = b;
                ref_rows[r][j] = bf16_to_f32(b);
            }
        }

        CUdeviceptr d_logits = 0;
        const std::size_t logits_bytes =
            static_cast<std::size_t>(num_rows) * vocab * sizeof(std::uint16_t);
        CU(cuMemAlloc(&d_logits, logits_bytes));
        CU(cuMemcpyHtoD(d_logits, h_logits.data(), logits_bytes));

        CUdeviceptr d_token_ir = 0;
        CU(cuMemAlloc(&d_token_ir, sizeof(std::int32_t)));

        int mism = 0;
        for (int r = 0; r < num_rows; ++r) {
            // GPU IR executor over row r, bound exactly as executor.cpp binds it.
            ResolvedInput rin;
            rin.input_id = 0;
            rin.cls = BindingClass::Intrinsic;
            rin.intrinsic = IntrinsicKind::Logits;
            rin.device_ptr = reinterpret_cast<const void*>(
                d_logits + static_cast<std::size_t>(r) * vocab * sizeof(std::uint16_t));
            rin.elem_count = vocab;
            rin.present = true;

            void* outs[1] = {reinterpret_cast<void*>(d_token_ir)};
            LaunchArgs args;
            args.inputs = std::span<const ResolvedInput>(&rin, 1);
            args.output_ptrs = std::span<void* const>(outs, 1);
            args.num_rows = 1;
            args.vocab_size = static_cast<int>(vocab);
            args.prng_offset = 0;
            backend.launch(h, args, /*stream=*/nullptr);
            CU(cuCtxSynchronize());

            std::int32_t ir_tok = -1;
            CU(cuMemcpyDtoH(&ir_tok, d_token_ir, sizeof(std::int32_t)));

            // CPU golden (== eval == eval-mock) over the identical synthetic row.
            const int golden_tok = ref::argmax(ref_rows[r]).token;

            if (ir_tok != golden_tok) {
                if (mism < 8) {
                    std::fprintf(stderr,
                                 "  MISMATCH row %d: gpu_ir=%d golden=%d\n",
                                 r, ir_tok, golden_tok);
                }
                ++mism;
            }
        }
        std::fprintf(stderr,
                     "eval-mock ≡ real-driver (argmax): rows=%d vocab=%u mismatches=%d\n",
                     num_rows, vocab, mism);
        CHECK(mism == 0);

        CU(cuMemFree(d_token_ir));
        CU(cuMemFree(d_logits));
    }

    CU(cuCtxDestroy(ctx));
    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_eval_mock_xcheck: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_eval_mock_xcheck: %d failure(s)\n", g_failures);
    return 1;
}

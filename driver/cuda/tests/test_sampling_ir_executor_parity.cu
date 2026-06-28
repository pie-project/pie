// Driver-level parity gate (lane L4 / echo): the Sampling-IR argmax program,
// run through echo's `IProgramBackend` exactly as the executor invokes it, must
// produce the SAME token as the legacy `launch_argmax_bf16` kernel the executor
// uses today (executor.cpp single_gpu_greedy_argmax path) — over the same bf16
// `ws.logits` rows. This is the on-GPU proof that the IR path is a drop-in for
// the legacy argmax sampler at M=1 decode.
//
// Self-contained: the backend (codegen/reader/jit) + the production argmax
// kernel + driver API. Failures abort non-zero so CTest catches them.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels/argmax.hpp"
#include "sampling_ir/jit_backend.hpp"
#include "sampling_ir/runtime.hpp"

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

// PSIR v2: greedy argmax over intrinsic logits Vector{vocab} → Token.
std::vector<std::uint8_t> build_argmax_v2(std::uint32_t vocab) {
    std::vector<std::uint8_t> b;
    b.insert(b.end(), {'P', 'S', 'I', 'R'});
    put_u16(b, 2);          // version
    put_u16(b, 0);          // flags
    put_u32(b, 1);          // n_inputs
    put_u32(b, 1);          // n_slots
    put_u32(b, 0);          // Input[0].id
    put_u8(b, 0);           // dtype F32
    put_u8(b, 1);           // shape Vector
    put_u32(b, vocab);      // len
    put_u32(b, 0);
    put_u8(b, 1);           // binding Intrinsic
    put_u8(b, 0);           // intrinsic Logits
    put_u32(b, 1);          // n_ops
    put_u8(b, 0x33);        // ReduceArgmax
    put_u32(b, 0);          // operand id 0
    put_u32(b, 1);          // n_outputs
    put_u32(b, 1);          // output value id
    put_u8(b, 0);           // kind Token
    return b;
}

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<std::uint16_t>(bits >> 16);
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
        std::fprintf(stderr, "backend arch: %s\n", backend.arch().c_str());

        const int num_rows = 64;
        const std::uint32_t vocab = 4096;
        std::vector<std::uint8_t> bytecode = build_argmax_v2(vocab);
        ProgramHandle h = backend.get_or_compile(bytecode);
        if (h == kInvalidProgram) {
            std::fprintf(stderr, "get_or_compile failed: %s\n",
                         backend.last_error().c_str());
        }
        CHECK(h != kInvalidProgram);

        // Random bf16 logits, one distinct max per row — the shape of a real
        // ws.logits decode block (num_rows × vocab, row-major bf16).
        std::mt19937 rng(20260626);
        std::normal_distribution<float> nd(0.f, 3.f);
        std::vector<std::uint16_t> h_logits(static_cast<std::size_t>(num_rows) * vocab);
        for (int r = 0; r < num_rows; ++r) {
            for (std::uint32_t j = 0; j < vocab; ++j) {
                h_logits[static_cast<std::size_t>(r) * vocab + j] =
                    f32_to_bf16(nd(rng));
            }
        }

        CUdeviceptr d_logits = 0;
        const std::size_t logits_bytes =
            static_cast<std::size_t>(num_rows) * vocab * sizeof(std::uint16_t);
        CU(cuMemAlloc(&d_logits, logits_bytes));
        CU(cuMemcpyHtoD(d_logits, h_logits.data(), logits_bytes));

        // ── Legacy path: the production argmax kernel over all rows ──
        std::int32_t* d_tokens_legacy = nullptr;
        cudaMalloc(&d_tokens_legacy, sizeof(std::int32_t) * num_rows);
        pie_cuda_driver::kernels::launch_argmax_bf16(
            reinterpret_cast<const void*>(d_logits), d_tokens_legacy,
            num_rows, static_cast<int>(vocab), /*stream=*/nullptr);
        cudaDeviceSynchronize();
        std::vector<std::int32_t> legacy(num_rows);
        cudaMemcpy(legacy.data(), d_tokens_legacy,
                   sizeof(std::int32_t) * num_rows, cudaMemcpyDeviceToHost);

        // ── IR path: SamplingIrBackend per row, bound exactly as the executor
        // binds it (Intrinsic logits row ptr → output token ptr, M=1). ──
        CUdeviceptr d_token_ir = 0;
        CU(cuMemAlloc(&d_token_ir, sizeof(std::int32_t)));
        std::vector<std::int32_t> ir(num_rows);
        for (int r = 0; r < num_rows; ++r) {
            ResolvedInput rin;
            rin.input_id = 0;
            rin.cls = BindingClass::Intrinsic;
            rin.intrinsic = IntrinsicKind::Logits;
            rin.device_ptr = reinterpret_cast<const void*>(
                d_logits + static_cast<std::size_t>(r) * vocab *
                               sizeof(std::uint16_t));
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
            CU(cuMemcpyDtoH(&ir[r], d_token_ir, sizeof(std::int32_t)));
        }

        int mism = 0;
        for (int r = 0; r < num_rows; ++r) {
            if (ir[r] != legacy[r]) {
                if (mism < 8) {
                    std::fprintf(stderr,
                                 "  MISMATCH row %d: ir=%d legacy=%d\n",
                                 r, ir[r], legacy[r]);
                }
                ++mism;
            }
        }
        std::fprintf(stderr,
                     "IR-argmax vs legacy-argmax: rows=%d vocab=%u mismatches=%d\n",
                     num_rows, vocab, mism);
        CHECK(mism == 0);

        cudaFree(d_tokens_legacy);
        CU(cuMemFree(d_token_ir));
        CU(cuMemFree(d_logits));
    }

    CU(cuCtxDestroy(ctx));

    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_executor_parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_executor_parity: %d failure(s)\n",
                 g_failures);
    return 1;
}

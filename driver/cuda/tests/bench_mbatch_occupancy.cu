// #10 M-batch occupancy/sync micro-bench (echo) — the land-vs-defer measurement.
//
// Question: does collapsing N single-row grammar fires (the LANDED sequential
// scatter: N × `num_rows=1` launch + N `cudaStreamSynchronize`) into ONE
// `num_rows=N` launch + 1 sync actually win? The fire's occupancy-critical stage
// is the argmax reduction, codegen'd as `LaunchShape::OneBlockPerRow`
// (grid = num_rows). So a single-row fire is **grid=1 block** — ~1/SM_count of an
// RTX 4090 (~1% occupancy on 128 SMs); the M-batch is **grid=N blocks**.
//
// delta's #11 load-test C3 (~1.4µs) is trivial-kernel LAUNCH overhead and CANNOT
// show this — it needs the real argmax-over-vocab kernel at num_rows=1 vs N, which
// is what this measures. Reports per-fire host wall-clock (includes the real
// sync cost the production scatter pays) + the speedup `seq_us / batched_us`.
//
// If the speedup is material at high N → the M-batch earns its multi-file
// complexity (per-row mask binding + kernel striding) → LAND. If marginal →
// DEFER (dedup + the sequential scatter already capture the dominant behavior;
// per delta's coefficients the compile wall ≫ fire by 10³–10⁴, so the M-batch is
// secondary). Build-to-measure, decide on data.
//
// Reuses the WS6 bench infra (SamplingIrBackend batched-lowering + BENCH_ARGMAX,
// the OneBlockPerRow program). Manual GPU perf run (not a ctest):
//   cmake --build build --target bench_mbatch_occupancy
//   ./build/bin/bench_mbatch_occupancy

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "sampling_ir/jit_backend.hpp"
#include "sampling_ir/runtime.hpp"

#include "bench_programs.h"

using namespace pie_cuda_driver::sampling_ir;

namespace {

void cu_check(CUresult res, const char* expr, int line) {
    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(res, &name);
        std::fprintf(stderr, "FATAL cu: %s:%d: %s -> %s\n", __FILE__, line, expr,
                     name ? name : "?");
        std::exit(2);
    }
}
void rt_check(cudaError_t e, const char* expr, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL rt: %s:%d: %s -> %s\n", __FILE__, line, expr,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define CU(x) cu_check((x), #x, __LINE__)
#define RT(x) rt_check((x), #x, __LINE__)

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return static_cast<std::uint16_t>(b >> 16);
}

double us_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

// Row counts spanning under-occupancy (N < SM_count: single-row leaves SMs idle)
// through saturation (N ≫ SM_count: the M-batch fills the GPU) — that's where the
// occupancy win, if real, shows.
const int kRows[] = {1, 2, 8, 32, 64, 128, 256, 512};
constexpr int kWarmup = 10;
constexpr int kIters = 50;

}  // namespace

int main() {
    CU(cuInit(0));
    CUdevice dev = 0;
    CU(cuDeviceGet(&dev, 0));
    CUcontext ctx = nullptr;
    CU(cuCtxCreate(&ctx, nullptr, 0, dev));

    char gpu[256] = {0};
    CU(cuDeviceGetName(gpu, sizeof(gpu), dev));
    int sm_count = 0;
    CU(cuDeviceGetAttribute(&sm_count,
                            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));

    const int vocab = static_cast<int>(BENCH_VOCAB);
    const int maxN = kRows[sizeof(kRows) / sizeof(kRows[0]) - 1];

    // bf16 logits [maxN, vocab] with a distinct max per row.
    std::vector<std::uint16_t> h_logits(static_cast<std::size_t>(maxN) * vocab);
    for (int r = 0; r < maxN; ++r)
        for (int j = 0; j < vocab; ++j)
            h_logits[static_cast<std::size_t>(r) * vocab + j] =
                f32_to_bf16(j == (r % vocab) ? 10.0f : 0.0f);
    CUdeviceptr d_logits = 0, d_out = 0;
    CU(cuMemAlloc(&d_logits, h_logits.size() * sizeof(std::uint16_t)));
    CU(cuMemcpyHtoD(d_logits, h_logits.data(),
                    h_logits.size() * sizeof(std::uint16_t)));
    CU(cuMemAlloc(&d_out, static_cast<std::size_t>(maxN) * sizeof(int)));

    // Batched-lowering backend + the bare argmax program (OneBlockPerRow). v3
    // self-binding bytecode ⇒ empty manifest.
    SamplingIrBackend backend(/*batched_lowering=*/true);
    ProgramHandle h = backend.get_or_compile(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(BENCH_ARGMAX),
            sizeof(BENCH_ARGMAX)),
        ProgramManifest{});
    if (h == kInvalidProgram) {
        std::fprintf(stderr, "argmax compile FAILED: %s\n",
                     backend.last_error().c_str());
        return 1;
    }
    const ProgramInterface& iface = backend.interface(h);

    // One batched launch at num_rows=N (argmax binds only the intrinsic logits).
    auto launch = [&](int N, cudaStream_t st) {
        std::vector<ResolvedInput> ri;
        for (const InputDecl& in : iface.inputs) {
            ResolvedInput r;
            r.input_id = in.input_id;
            r.cls = in.cls;
            r.intrinsic = in.intrinsic;
            r.elem_count = in.elem_count;
            r.present = true;
            r.device_ptr = reinterpret_cast<const void*>(d_logits);
            ri.push_back(r);
        }
        void* outp = reinterpret_cast<void*>(d_out);
        LaunchArgs a;
        a.inputs = std::span<const ResolvedInput>(ri.data(), ri.size());
        a.output_ptrs = std::span<void* const>(&outp, 1);
        a.num_rows = N;
        a.vocab_size = vocab;
        a.prng_offset = 0;
        a.row_seeds = nullptr;
        backend.launch(h, a, st);
    };

    cudaStream_t stream = nullptr;
    RT(cudaStreamCreate(&stream));

    std::fprintf(stderr,
                 "GPU=%s SMs=%d vocab=%d — argmax OneBlockPerRow (grid=num_rows): "
                 "seq=N×(num_rows=1 launch + sync), batched=1×(num_rows=N + sync)\n",
                 gpu, sm_count, vocab);
    std::printf("gpu,sm_count,vocab,N,seqA_us,asyncB_us,mbatchC_us,"
                "sync_win_AoverB,occ_win_BoverC,total_win_AoverC\n");

    for (int N : kRows) {
        // CONFIG A — SEQUENTIAL (the LANDED scatter): N × num_rows=1 launch + N
        // per-fire syncs (the sync is there to read out_scratch before the next
        // fire overwrites it).
        for (int w = 0; w < kWarmup; ++w) {
            launch(1, stream);
            RT(cudaStreamSynchronize(stream));
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < kIters; ++it)
            for (int r = 0; r < N; ++r) {
                launch(1, stream);
                RT(cudaStreamSynchronize(stream));
            }
        const double seq_us = us_since(t0) / kIters;

        // CONFIG B — ASYNC sequential: N × num_rows=1 launch with NO per-fire
        // sync + ONE sync at the end. Isolates the SYNC-ELIMINATION win (CHEAP:
        // give each program its own out_scratch slot ⇒ no per-fire sync, no kernel
        // change) from the occupancy win. Kernels still run GPU-serial (one
        // stream, grid=1 block each) — so A→B = sync overhead removed, B→C =
        // pure occupancy. (Timing only: outputs clobber d_out[0], irrelevant.)
        for (int w = 0; w < kWarmup; ++w) {
            for (int r = 0; r < N; ++r) launch(1, stream);
            RT(cudaStreamSynchronize(stream));
        }
        auto tb = std::chrono::steady_clock::now();
        for (int it = 0; it < kIters; ++it) {
            for (int r = 0; r < N; ++r) launch(1, stream);
            RT(cudaStreamSynchronize(stream));
        }
        const double async_us = us_since(tb) / kIters;

        // CONFIG C — M-BATCH: 1 × num_rows=N launch (grid=N blocks) + 1 sync.
        for (int w = 0; w < kWarmup; ++w) {
            launch(N, stream);
            RT(cudaStreamSynchronize(stream));
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int it = 0; it < kIters; ++it) {
            launch(N, stream);
            RT(cudaStreamSynchronize(stream));
        }
        const double batched_us = us_since(t1) / kIters;

        const double sync_win = async_us > 0.0 ? seq_us / async_us : 0.0;
        const double occ_win = batched_us > 0.0 ? async_us / batched_us : 0.0;
        const double total_win = batched_us > 0.0 ? seq_us / batched_us : 0.0;
        std::printf("%s,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n", gpu, sm_count,
                    vocab, N, seq_us, async_us, batched_us, sync_win, occ_win,
                    total_win);
        std::fprintf(stderr,
                     "N=%4d  seqA=%10.3f  asyncB=%10.3f  mbatchC=%9.3f us/fire  | "
                     "sync(A/B)=%.1fx  occ(B/C)=%.1fx  total=%.1fx\n",
                     N, seq_us, async_us, batched_us, sync_win, occ_win,
                     total_win);
    }

    RT(cudaStreamDestroy(stream));
    CU(cuMemFree(d_logits));
    CU(cuMemFree(d_out));
    CU(cuCtxDestroy(ctx));
    return 0;
}

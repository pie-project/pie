// Standalone microbenchmark for the bf16 decode GEMV.
//
// Why this exists: a survey of what vLLM and SGLang actually launch on
// sm_100 puts cuBLAS's `nvjet_sm100_*` dense GEMM/GEMV at 25-81% of decode
// time on every one of the five benchmark models -- the largest single job
// in all five, and the only one that is the SAME job on all of them:
//
//     Qwen3.6-27B      81.3%      gemma-4-26B-A4B   33.2%
//     gemma-4-31B      77.2%      gpt-oss-20b       25.4%
//     Qwen3.6-35B-A3B  41.7%      (+ 4-11% splitKreduce on each)
//
// pie's kernel for this is one output row per warp (or per block with the
// K-split variant), accumulating in FP32 through `__bfloat162float` -- pure
// scalar FMA on CUDA cores, no tensor core. cuBLAS runs tiles like
// `64x8_64x16` and pays a separate `splitKreduce` pass. This harness times
// pie's two kernels against cuBLASLt at the shapes the models actually
// decode through, so a replacement can be judged in seconds instead of
// through a 12-minute engine build.
//
// Build (~15s):
//   nvcc -O3 -std=c++20 -arch=sm_100 --expt-relaxed-constexpr \
//        -I driver/cuda/src -o /tmp/gemv_bench \
//        driver/cuda/bench/gemv_bench.cu driver/cuda/src/kernels/gemv.cu \
//        -lcublas -lcublasLt
//
// Run:
//   /tmp/gemv_bench                # every shape in the table below
//   /tmp/gemv_bench 4096 2880      # one N,K
//
// Reports per-call microseconds and the effective bandwidth each variant
// reaches against the weight read, which is the only term that matters at
// M=1: the weight matrix is N*K*2 bytes and everything else is noise.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/gemv.hpp"

namespace {

constexpr int kWarmup = 20;
constexpr int kIters = 100;

#define CHECK(x)                                                            \
    do {                                                                    \
        cudaError_t e_ = (x);                                               \
        if (e_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(e_));                           \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

struct Shape {
    const char* what;
    int n;
    int k;
};

// Every projection the five benchmark models decode through, from their own
// configs. N is the output width, K the reduction. M is 1 (one decode row).
const Shape kShapes[] = {
    // gpt-oss-20b: hidden 2880, 64 q heads x 64, 8 kv heads x 64
    {"gptoss q_proj",      4096,   2880},
    {"gptoss k/v_proj",     512,   2880},
    {"gptoss o_proj",      2880,   4096},
    {"gptoss lm_head",   201088,   2880},
    {"qwen35 lm_head",   151936,   4096},
    // gemma-4-31B: hidden 5376, 32 q x 256, 16 kv x 256, inter 21504
    {"gemma31 q_proj",     8192,   5376},
    {"gemma31 kv_proj",    4096,   5376},
    {"gemma31 gate/up",   21504,   5376},
    {"gemma31 down",       5376,  21504},
    {"gemma31 lm_head",  262144,   5376},
    // Qwen3.6-27B: hidden 5120, 24 q x 256, 4 kv x 256, inter 17408
    {"qwen27 q_proj",      6144,   5120},
    {"qwen27 kv_proj",     1024,   5120},
    {"qwen27 gate/up",    17408,   5120},
    {"qwen27 down",        5120,  17408},
};

float time_ms(void (*fn)(void*), void* ctx) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < kWarmup; ++i) fn(ctx);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < kIters; ++i) fn(ctx);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return ms / kIters;
}

struct Ctx {
    // B200's L2 is 132.6 MB. Most shapes below are SMALLER than that (gptoss
    // q_proj is 23.6 MB), so timing a loop over one weight buffer measures L2
    // bandwidth, not HBM -- and the engine never gets that reuse, since a
    // token streams ~3.6 GB of gpt-oss weights past the cache. Rotate enough
    // buffers to exceed 2x L2, then the read is cold every time.
    std::vector<__nv_bfloat16*> w;
    int it;
    __nv_bfloat16* x;
    __nv_bfloat16* y;
    int n, k;
    cublasLtHandle_t lt;
    void* ws;
    size_t ws_bytes;
    // Built ONCE per shape. Creating these inside the timed loop -- which is
    // what this harness did originally -- puts four cublasLtMatrixLayoutCreate
    // and four destroys on the host between every launch, and at ~10 us per
    // GEMM that is enough host work to leave the GPU idle waiting for the next
    // one. It made cuBLAS look slower than it is in every shape reported here.
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulAlgo_t algo;
    bool have_algo;
};

void run_pie(void* p) {
    Ctx* c = static_cast<Ctx*>(p);
    __nv_bfloat16* w = c->w[c->it++ % c->w.size()];
    // The driver's own entry point: picks the row-per-warp or the K-split
    // form by `PIE_GEMV_SPLITK_MAX_ROWS`, exactly as the engine would.
    pie_cuda_driver::kernels::launch_gemv_bf16(
        w, c->x, /*bias=*/nullptr, c->y, c->n, c->k, /*stream=*/nullptr,
        /*beta=*/0.f);
}

void cublas_setup(Ctx* c) {
    cublasLtMatmulDescCreate(&c->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t tn = CUBLAS_OP_T, nn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(c->op, CUBLASLT_MATMUL_DESC_TRANSA, &tn,
                                   sizeof(tn));
    cublasLtMatmulDescSetAttribute(c->op, CUBLASLT_MATMUL_DESC_TRANSB, &nn,
                                   sizeof(nn));
    // W is [N, K] row-major = [K, N] column-major, so op(A)=A^T gives N rows.
    cublasLtMatrixLayoutCreate(&c->la, CUDA_R_16BF, c->k, c->n, c->k);
    cublasLtMatrixLayoutCreate(&c->lb, CUDA_R_16BF, c->k, 1, c->k);
    cublasLtMatrixLayoutCreate(&c->lc, CUDA_R_16BF, c->n, 1, c->n);
    c->have_algo = false;
}

void cublas_teardown(Ctx* c) {
    cublasLtMatrixLayoutDestroy(c->la);
    cublasLtMatrixLayoutDestroy(c->lb);
    cublasLtMatrixLayoutDestroy(c->lc);
    cublasLtMatmulDescDestroy(c->op);
}

void run_cublas(void* p) {
    Ctx* c = static_cast<Ctx*>(p);
    __nv_bfloat16* w = c->w[c->it++ % c->w.size()];
    const float alpha = 1.f, beta = 0.f;
    cublasLtMatmul(c->lt, c->op, &alpha, w, c->la, c->x, c->lb, &beta, c->y,
                   c->lc, c->y, c->lc, c->have_algo ? &c->algo : nullptr,
                   c->ws, c->ws_bytes, nullptr);
}

void bench(const Shape& s) {
    Ctx c{};
    c.n = s.n;
    c.k = s.k;
    const size_t wn = static_cast<size_t>(s.n) * s.k;
    const size_t wbytes = wn * sizeof(__nv_bfloat16);
    // 300 MB of distinct weights is > 2x L2; one buffer already suffices once
    // the shape itself exceeds the cache (gemma31 lm_head is 2.8 GB).
    int bufs = static_cast<int>((300u << 20) / wbytes) + 1;
    if (bufs > 8) bufs = 8;
    c.w.resize(bufs);
    for (int i = 0; i < bufs; ++i) {
        CHECK(cudaMalloc(&c.w[i], wbytes));
        CHECK(cudaMemset(c.w[i], 0x3c + i, wbytes));
    }
    CHECK(cudaMalloc(&c.x, s.k * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&c.y, s.n * sizeof(__nv_bfloat16)));
    CHECK(cudaMemset(c.x, 0x3c, s.k * sizeof(__nv_bfloat16)));
    c.ws_bytes = 32u << 20;
    CHECK(cudaMalloc(&c.ws, c.ws_bytes));
    cublasLtCreate(&c.lt);

    const double bytes = static_cast<double>(wn) * 2.0;
    cublas_setup(&c);
    c.it = 0;
    const float pie_ms = time_ms(run_pie, &c);
    c.it = 0;
    const float cub_ms = time_ms(run_cublas, &c);   // default heuristic
    // Then ask for candidates explicitly and time each: passing a null algo
    // takes cuBLASLt's first heuristic pick, which is not always its best at
    // M=1, and SGLang reaches shapes this harness could not.
    float best_ms = cub_ms;
    int best_i = -1;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c.ws_bytes,
        sizeof(c.ws_bytes));
    constexpr int kCand = 12;
    cublasLtMatmulHeuristicResult_t heur[kCand];
    int found = 0;
    cublasLtMatmulAlgoGetHeuristic(c.lt, c.op, c.la, c.lb, c.lc, c.lc, pref,
                                   kCand, heur, &found);
    for (int i = 0; i < found; ++i) {
        c.algo = heur[i].algo;
        c.have_algo = true;
        c.it = 0;
        const float ms = time_ms(run_cublas, &c);
        if (ms < best_ms) { best_ms = ms; best_i = i; }
    }
    c.have_algo = false;
    cublasLtMatmulPreferenceDestroy(pref);
    // Staging must not change the answer. Compare the two forms bit-for-bit
    // on the same inputs before either number is reported.
    std::printf("%-18s N=%6d K=%6d %4.0fMB | pie %7.1f %4.2fTB/s | cuBLASLt "
                "dflt %7.1f | best(%2d/%2d) %7.1f %4.2fTB/s | pie/best %5.2fx\n",
                s.what, s.n, s.k, wbytes / 1e6,
                pie_ms * 1e3, bytes / (pie_ms * 1e-3) / 1e12,
                cub_ms * 1e3, best_i, found,
                best_ms * 1e3, bytes / (best_ms * 1e-3) / 1e12,
                pie_ms / best_ms);

    cublas_teardown(&c);
    cublasLtDestroy(c.lt);
    CHECK(cudaFree(c.ws));
    CHECK(cudaFree(c.y));
    CHECK(cudaFree(c.x));
    for (__nv_bfloat16* w : c.w) CHECK(cudaFree(w));
}

}  // namespace

// Sweep the row-per-warp GEMV's two compile-time constants at one shape.
// They were picked on an H100; B200 has 148 SMs and a different memory
// system, and this kernel is 78% of Qwen3.6-27B's decode and 77% of
// gemma-4-31B's, so the constants are worth measuring rather than inheriting.
void sweep(const Shape& sh) {
    Ctx c{};
    c.n = sh.n; c.k = sh.k;
    const size_t wn = static_cast<size_t>(sh.n) * sh.k;
    const size_t wbytes = wn * sizeof(__nv_bfloat16);
    int bufs = static_cast<int>((300u << 20) / wbytes) + 1;
    if (bufs > 8) bufs = 8;
    c.w.resize(bufs);
    for (int i = 0; i < bufs; ++i) {
        CHECK(cudaMalloc(&c.w[i], wbytes));
        CHECK(cudaMemset(c.w[i], 0x3c + i, wbytes));
    }
    CHECK(cudaMalloc(&c.x, sh.k * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&c.y, sh.n * sizeof(__nv_bfloat16)));
    CHECK(cudaMemset(c.x, 0x3c, sh.k * sizeof(__nv_bfloat16)));
    const double bytes = static_cast<double>(wn) * 2.0;
    std::printf("%-18s N=%6d K=%6d  %4.0f MB\n", sh.what, sh.n, sh.k,
                wbytes / 1e6);
    struct Cfg { int w, u; };
    const Cfg cfgs[] = {{1,4},{2,4},{4,4},{8,4},{16,4},
                        {4,2},{4,8},{2,8},{8,8},{1,8},{2,2},{8,2}};
    for (const Cfg& cf : cfgs) {
        c.it = 0;
        struct Arg { Ctx* c; int w, u; } arg{&c, cf.w, cf.u};
        auto fn = [](void* p) {
            Arg* a = static_cast<Arg*>(p);
            __nv_bfloat16* w = a->c->w[a->c->it++ % a->c->w.size()];
            pie_cuda_driver::kernels::launch_gemv_bf16_tuned(
                w, a->c->x, nullptr, a->c->y, a->c->n, a->c->k, a->w, a->u,
                nullptr);
        };
        const float ms = time_ms(fn, &arg);
        std::printf("   warps=%-3d unroll=%-2d %8.1f us  %5.2f TB/s%s\n",
                    cf.w, cf.u, ms * 1e3, bytes / (ms * 1e-3) / 1e12,
                    (cf.w == 4 && cf.u == 4) ? "   <- shipping" : "");
    }
    for (__nv_bfloat16* w : c.w) CHECK(cudaFree(w));
    CHECK(cudaFree(c.x)); CHECK(cudaFree(c.y));
}

void sweep_splitk(const Shape& sh) {
    Ctx c{};
    c.n = sh.n; c.k = sh.k;
    const size_t wn = static_cast<size_t>(sh.n) * sh.k;
    const size_t wbytes = wn * sizeof(__nv_bfloat16);
    int bufs = static_cast<int>((300u << 20) / wbytes) + 1;
    if (bufs > 8) bufs = 8;
    c.w.resize(bufs);
    for (int i = 0; i < bufs; ++i) {
        CHECK(cudaMalloc(&c.w[i], wbytes));
        CHECK(cudaMemset(c.w[i], 0x3c + i, wbytes));
    }
    CHECK(cudaMalloc(&c.x, sh.k * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&c.y, sh.n * sizeof(__nv_bfloat16)));
    CHECK(cudaMemset(c.x, 0x3c, sh.k * sizeof(__nv_bfloat16)));
    const double bytes = static_cast<double>(wn) * 2.0;
    std::printf("%-18s N=%6d K=%6d  %4.0f MB  [split-K]\n", sh.what, sh.n,
                sh.k, wbytes / 1e6);
    for (int w : {2, 4, 8, 16, 32}) {
        c.it = 0;
        struct Arg { Ctx* c; int w; } arg{&c, w};
        auto fn = [](void* p) {
            Arg* a = static_cast<Arg*>(p);
            __nv_bfloat16* wt = a->c->w[a->c->it++ % a->c->w.size()];
            pie_cuda_driver::kernels::launch_gemv_splitk_tuned(
                wt, a->c->x, nullptr, a->c->y, a->c->n, a->c->k, a->w, nullptr);
        };
        const float ms = time_ms(fn, &arg);
        std::printf("   warps=%-3d %8.1f us  %5.2f TB/s%s\n", w, ms * 1e3,
                    bytes / (ms * 1e-3) / 1e12, w == 8 ? "   <- shipping" : "");
    }
    for (__nv_bfloat16* w : c.w) CHECK(cudaFree(w));
    CHECK(cudaFree(c.x)); CHECK(cudaFree(c.y));
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "sweep") {
        // The shapes where this kernel dominates: Qwen3.6-27B's MLP (78% of
        // its decode) and gemma-4-31B's (77%), plus a vocab projection.
        // Split-K form: the shapes small enough in N to take it (N <= 4096).
        const Shape sk[] = {{"gptoss k/v_proj", 512, 2880},
                            {"qwen27 kv_proj", 1024, 5120},
                            {"gemma31 kv_proj", 4096, 5376},
                            {"gptoss q_proj", 4096, 2880}};
        for (const Shape& x : sk) sweep_splitk(x);
        const Shape sw[] = {{"qwen27 gate/up", 17408, 5120},
                            {"qwen27 down", 5120, 17408},
                            {"gemma31 gate/up", 21504, 5376},
                            {"gemma31 down", 5376, 21504},
                            {"gptoss lm_head", 201088, 2880}};
        for (const Shape& x : sw) sweep(x);
        return 0;
    }
    if (argc == 3) {
        Shape s{"custom", std::atoi(argv[1]), std::atoi(argv[2])};
        bench(s);
        return 0;
    }
    for (const Shape& s : kShapes) bench(s);
    return 0;
}

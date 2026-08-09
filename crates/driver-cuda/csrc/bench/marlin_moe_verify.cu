// Does the Marlin MoE GEMM actually compute the right answer on this device?
//
// Nothing in the tree asserts this. `tests/test_mxfp4_marlin.cu` is named for
// the MXFP4 Marlin path but only checks the layout kernels against a
// re-implementation of their own index formula in the test body -- so a
// misunderstood layout contract makes kernel and test wrong together and the
// suite still passes. It never launches a GEMM, never calls
// `launch_gptq_repack_w4_no_perm`, and makes no numeric assertion at all.
// `bench/moe_bench.cu` only times, and says so: "Marlin wants its own repacked
// layout that this does not build."
//
// So this builds it. E=1 and top_k=1 put every route on one expert, which
// makes the reference an ordinary dense matmul that can be computed on the
// host with no reference kernel to trust. The weights go through the REAL
// repack chain the loader uses (`transcode_engine.hpp:1328-1336`):
//
//     raw MXFP4 --kernels::quant::mxfp4_weight_to_gptq_w4--> GPTQ staging
//               --launch_gptq_repack_w4_no_perm--> Marlin tiles
//     raw E8M0  --kernels::quant::mxfp4_scales_to_marlin_e8m0--> Marlin scales
//
// Build (~40 s):
//   nvcc -O3 -std=c++20 -arch=sm_80 --expt-relaxed-constexpr \
//        -I driver/cuda/src -I driver/cuda/third_party/marlin_moe \
//        -I driver/cuda/third_party/marlin \
//        -o /tmp/marlin_verify driver/cuda/bench/marlin_moe_verify.cu \
//        driver/cuda/src/quant/mxfp4_marlin.cu \
//        driver/cuda/src/moe/moe_dispatch.cu \
//        driver/cuda/third_party/marlin_moe/ops.cu \
//        driver/cuda/third_party/marlin_moe/marlin_moe_wrapper.cpp \
//        driver/cuda/third_party/marlin/gptq_marlin_repack.cu \
//        driver/cuda/third_party/marlin/marlin_wrapper.cpp
//
// Run: /tmp/marlin_verify [M] [N] [K]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "quant/mxfp4_marlin.hpp"
#include "moe/moe_dispatch.hpp"
#include "marlin_moe_wrapper.hpp"

// Declared rather than pulled in via `marlin_wrapper.hpp`: that header's other
// entry points drag in the dense Marlin GEMM TU, which this test does not use.
namespace marlin {
void pie_gptq_marlin_repack_w4_no_perm(const std::uint32_t* b_q_weight,
                                       std::uint32_t* out, int size_k,
                                       int size_n, cudaStream_t stream);
}  // namespace marlin

#define CHECK(x)                                                              \
    do {                                                                      \
        cudaError_t e_ = (x);                                                 \
        if (e_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,  \
                         cudaGetErrorString(e_));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// The same table `kernels/dequant_fp4.cu` decodes with, and the same
// convention: byte i of a row holds k = 2i in its LOW nibble and k = 2i+1 in
// its HIGH nibble, with one E8M0 scale per 32 contiguous k.
static const float kFp4Lut[16] = {
     0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

static void* dalloc(std::size_t n) {
    void* p = nullptr;
    CHECK(cudaMalloc(&p, n));
    CHECK(cudaMemset(p, 0, n));
    return p;
}

template <typename T>
static void* dupload(const std::vector<T>& v) {
    void* p = dalloc(v.size() * sizeof(T));
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

static float bf16_to_float(std::uint16_t b) {
    std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static std::uint16_t float_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // round-to-nearest-even, matching __float2bfloat16
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(u >> 16);
}

int main(int argc, char** argv) {
    const int M     = argc > 1 ? std::atoi(argv[1]) : 32;    // tokens
    const int N     = argc > 2 ? std::atoi(argv[2]) : 256;   // output rows
    const int K     = argc > 3 ? std::atoi(argv[3]) : 256;   // reduction dim
    const int E     = argc > 4 ? std::atoi(argv[4]) : 1;     // experts
    const int top_k = argc > 5 ? std::atoi(argv[5]) : 1;
    // scale_mode 0 = the loader's repack (kernels::quant::mxfp4_scales_to_marlin_e8m0,
    // which applies Marlin's 64-wide + 4-lane permutation).
    // scale_mode 1 = a PLAIN [N, groups] -> [groups, N] transpose, which is
    // what `kernels::moe::transpose_expert_scales_u8` does at runtime in
    // mixtral.cpp. If 0 passes and 1 fails, the runtime path is feeding
    // Marlin unpermuted scales and that is the gpt-oss native bug.
    const int scale_mode = argc > 6 ? std::atoi(argv[6]) : 0;
    // route_mode picks the ROUTING DISTRIBUTION. Uniform random (0) spreads
    // routes evenly and leaves every expert with a full-ish block, which is
    // exactly the case least likely to reproduce a rare metadata bug. The
    // others deliberately produce the shapes moe_align_decode has to pad
    // around: a single hot expert with every other one EMPTY, one route per
    // expert so every block is partial, and a skew that leaves most experts
    // empty while one is saturated.
    const int route_mode = argc > 7 ? std::atoi(argv[7]) : 0;

    if (K % 32 != 0 || N % 64 != 0) {
        std::fprintf(stderr, "need K %% 32 == 0 and N %% 64 == 0\n");
        return 2;
    }

    int dev = 0, ccmaj = 0, ccmin = 0;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&ccmaj, cudaDevAttrComputeCapabilityMajor, dev));
    CHECK(cudaDeviceGetAttribute(&ccmin, cudaDevAttrComputeCapabilityMinor, dev));
    const int routes = M * top_k;
    std::printf("device sm_%d%d   M=%d N=%d K=%d  E=%d top_k=%d  routes=%d  scales=%s\n",
                ccmaj, ccmin, M, N, K, E, top_k, routes,
                scale_mode == 0 ? "marlin-permuted (loader repack)"
                : scale_mode == 1 ? "plain transpose, no permutation"
                : "loader repack THEN engine transpose (mixtral.cpp path)");
    static const char* kRouteName[4] = {
        "uniform random", "all on expert 0", "round robin (partial blocks)",
        "skewed (most experts empty)"};
    std::printf("  routing: %s\n", kRouteName[route_mode & 3]);

    std::mt19937 rng(20260808);
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_int_distribution<int> expert_dist(0, E - 1);
    std::uniform_real_distribution<float> actd(-1.f, 1.f);

    // --- weights: [E, N, K] MXFP4 nibbles + [E, N, K/32] E8M0 scales ------
    const int groups = K / 32;
    const std::size_t w_stride = std::size_t(N) * (K / 2);
    const std::size_t s_stride = std::size_t(N) * groups;
    std::vector<std::uint8_t> hw(std::size_t(E) * w_stride);
    std::vector<std::uint8_t> hs(std::size_t(E) * s_stride);
    for (auto& b : hw) b = static_cast<std::uint8_t>(nib(rng) | (nib(rng) << 4));
    // Scales spread around 1.0 so a mis-indexed scale cannot hide behind a
    // uniform table -- 127 everywhere would make the permutation untestable.
    {
        std::uniform_int_distribution<int> sd(125, 129);
        for (auto& b : hs) b = static_cast<std::uint8_t>(sd(rng));
    }

    // --- activations: [M, K] bf16 -----------------------------------------
    std::vector<std::uint16_t> ha(std::size_t(M) * K);
    for (auto& x : ha) x = float_to_bf16(actd(rng));

    // --- routing: route r = token r/top_k, expert h_topk[r] ---------------
    std::vector<std::int32_t> h_topk(routes);
    switch (route_mode) {
        case 1:  // every route on expert 0; all other experts EMPTY
            for (auto& x : h_topk) x = 0;
            break;
        case 2:  // round robin: every expert gets a PARTIAL block
            for (int i = 0; i < routes; ++i) h_topk[i] = i % E;
            break;
        case 3: { // saturate expert 0, sprinkle the rest, most experts empty
            for (int i = 0; i < routes; ++i)
                h_topk[i] = (i % 10 == 0) ? (1 + (i / 10) % (E > 1 ? E - 1 : 1)) : 0;
            break;
        }
        default:
            for (auto& x : h_topk) x = expert_dist(rng);
    }

    // --- host reference, per ROUTE (which is how Marlin indexes `out`) ----
    std::vector<float> ref(std::size_t(routes) * N, 0.f);
    for (int r = 0; r < routes; ++r) {
        const int tok = r / top_k;
        const int e = h_topk[r];
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int g = 0; g < groups; ++g) {
                const float scale = std::exp2f(
                    static_cast<float>(hs[e * s_stride + std::size_t(n) * groups + g]) -
                    127.f);
                for (int i = 0; i < 16; ++i) {
                    const std::uint8_t byte =
                        hw[e * w_stride + std::size_t(n) * (K / 2) + g * 16 + i];
                    const int klo = g * 32 + 2 * i;
                    acc += bf16_to_float(ha[std::size_t(tok) * K + klo]) *
                               kFp4Lut[byte & 0xF] * scale +
                           bf16_to_float(ha[std::size_t(tok) * K + klo + 1]) *
                               kFp4Lut[byte >> 4] * scale;
                }
            }
            ref[std::size_t(r) * N + n] = acc;
        }
    }

    // --- device: run the real repack chain --------------------------------
    using namespace pie_cuda_driver;
    void* d_raw_w = dupload(hw);
    void* d_raw_s = dupload(hs);
    void* d_act   = dupload(ha);

    // Per-expert marlin tile block and scale block, tightly packed, exactly as
    // `transcode_engine.hpp` writes them: it loops over the expert axis and
    // repacks each slab independently into a contiguous stride.
    const std::size_t marl_w_stride =
        std::size_t(K / 16) * (N * 16 / 8) * sizeof(std::uint32_t);
    const std::size_t marl_s_stride = std::size_t(groups) * N;

    void* d_gptq   = dalloc(std::size_t(K / 8) * N * sizeof(std::uint32_t));
    void* d_marl_w = dalloc(std::size_t(E) * marl_w_stride);
    void* d_marl_s = dalloc(std::size_t(E) * marl_s_stride);

    for (int e = 0; e < E; ++e) {
        kernels::quant::mxfp4_weight_to_gptq_w4(
            static_cast<const std::uint8_t*>(d_raw_w) + e * w_stride, d_gptq,
            /*source_rows=*/N, /*source_row_offset=*/0,
            /*selected_rows=*/N, /*valid_rows=*/N, /*source_stride_k=*/K,
            /*source_col_offset=*/0, /*source_k=*/K, /*target_k=*/K,
            kernels::quant::Mxfp4RowSelect::Identity, /*stream=*/0);
        ::marlin::pie_gptq_marlin_repack_w4_no_perm(
            static_cast<const std::uint32_t*>(d_gptq),
            reinterpret_cast<std::uint32_t*>(
                static_cast<std::uint8_t*>(d_marl_w) + e * marl_w_stride),
            /*size_k=*/K, /*size_n=*/N, /*stream=*/0);
        if (scale_mode == 0) {
            kernels::quant::mxfp4_scales_to_marlin_e8m0(
                static_cast<const std::uint8_t*>(d_raw_s) + e * s_stride,
                static_cast<std::uint8_t*>(d_marl_s) + e * marl_s_stride,
                /*source_rows=*/N, /*source_row_offset=*/0,
                /*selected_rows=*/N, /*valid_rows=*/N,
                /*source_stride_groups=*/groups,
                /*source_group_offset=*/0, /*source_groups=*/groups,
                /*target_groups=*/groups, kernels::quant::Mxfp4RowSelect::Identity,
                /*stream=*/0);
        } else if (scale_mode == 1) {
            // Plain [N, groups] -> [groups, N], no Marlin lane permutation --
            // the shape `kernels::moe::transpose_expert_scales_u8` produces.
            std::vector<std::uint8_t> t(marl_s_stride);
            for (int n = 0; n < N; ++n)
                for (int g = 0; g < groups; ++g)
                    t[std::size_t(g) * N + n] = hs[e * s_stride + std::size_t(n) * groups + g];
            CHECK(cudaMemcpy(static_cast<std::uint8_t*>(d_marl_s) + e * marl_s_stride,
                             t.data(), t.size(), cudaMemcpyHostToDevice));
        } else {
            // Mode 2 reproduces the ENGINE's actual sequence: the loader
            // repack already wrote [groups, N] (permuted, K-major), the
            // contract mislabelled it [Ip, groups], and mixtral.cpp then
            // transposes it AGAIN believing the label. If mode 0 passes and
            // this fails, that double transform is the gpt-oss native bug.
            void* tmp = dalloc(marl_s_stride);
            kernels::quant::mxfp4_scales_to_marlin_e8m0(
                static_cast<const std::uint8_t*>(d_raw_s) + e * s_stride, tmp,
                /*source_rows=*/N, /*source_row_offset=*/0,
                /*selected_rows=*/N, /*valid_rows=*/N,
                /*source_stride_groups=*/groups,
                /*source_group_offset=*/0, /*source_groups=*/groups,
                /*target_groups=*/groups, kernels::quant::Mxfp4RowSelect::Identity,
                /*stream=*/0);
            CHECK(cudaDeviceSynchronize());
            std::vector<std::uint8_t> packed(marl_s_stride), t(marl_s_stride);
            CHECK(cudaMemcpy(packed.data(), tmp, marl_s_stride, cudaMemcpyDeviceToHost));
            for (int n = 0; n < N; ++n)
                for (int g = 0; g < groups; ++g)
                    t[std::size_t(g) * N + n] = packed[std::size_t(n) * groups + g];
            CHECK(cudaMemcpy(static_cast<std::uint8_t*>(d_marl_s) + e * marl_s_stride,
                             t.data(), t.size(), cudaMemcpyHostToDevice));
            CHECK(cudaFree(tmp));
        }
    }
    CHECK(cudaDeviceSynchronize());

    // --- MoE metadata -----------------------------------------------------
    const int block = 16;
    const int max_blocks = (routes + E * (block - 1) + block - 1) / block;
    void* d_topk    = dupload(h_topk);
    void* d_sorted  = dalloc(std::size_t(max_blocks) * block * 4);
    void* d_experts = dalloc(std::size_t(max_blocks) * 4);
    void* d_npast   = dalloc(sizeof(std::int32_t));
    kernels::moe::moe_align_decode(
        static_cast<const std::int32_t*>(d_topk),
        static_cast<std::int32_t*>(d_sorted),
        static_cast<std::int32_t*>(d_experts), nullptr, routes, E, block,
        max_blocks, static_cast<std::int32_t*>(d_npast), /*stream=*/0);
    CHECK(cudaDeviceSynchronize());

    void* d_out = dalloc(std::size_t(routes) * N * sizeof(std::uint16_t));
    void* d_ws  = dalloc(marlin_moe::marlin_moe_workspace_bytes(N, block));
    // Poison the output so a kernel that never writes is caught as garbage
    // rather than silently reading as zeros that happen to look plausible.
    CHECK(cudaMemset(d_out, 0x5A, std::size_t(routes) * N * sizeof(std::uint16_t)));

    marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16(
        d_act, d_marl_w, d_marl_s, /*bias=*/nullptr, d_out,
        /*reduce_scratch=*/nullptr, d_ws,
        static_cast<const std::int32_t*>(d_sorted),
        static_cast<const std::int32_t*>(d_experts),
        static_cast<const std::int32_t*>(d_npast),
        // prob_m is TOKENS, not routes -- the kernel expands by top_k itself.
        // Passing routes here walks top_k times too far and hangs.
        /*topk_weights=*/nullptr, block, E, top_k,
        /*mul_topk_weights=*/false, M, N, K, /*stream=*/0);
    CHECK(cudaDeviceSynchronize());

    std::vector<std::uint16_t> hout(std::size_t(routes) * N);
    CHECK(cudaMemcpy(hout.data(), d_out, hout.size() * sizeof(std::uint16_t),
                     cudaMemcpyDeviceToHost));

    // --- compare -----------------------------------------------------------
    double max_rel = 0.0, sum_abs = 0.0, ref_mag = 0.0;
    int worst_m = 0, worst_n = 0;
    for (int m = 0; m < routes; ++m) {
        for (int n = 0; n < N; ++n) {
            const double got = bf16_to_float(hout[std::size_t(m) * N + n]);
            const double want = ref[std::size_t(m) * N + n];
            const double denom = std::max(1e-3, std::fabs(want));
            const double rel = std::fabs(got - want) / denom;
            sum_abs += std::fabs(got - want);
            ref_mag += std::fabs(want);
            if (rel > max_rel) { max_rel = rel; worst_m = m; worst_n = n; }
        }
    }
    const double mean_rel = ref_mag > 0 ? sum_abs / ref_mag : 0.0;

    std::printf("  max rel err  %.4f  at (route=%d, n=%d): got %.5f  want %.5f\n",
                max_rel, worst_m, worst_n,
                bf16_to_float(hout[std::size_t(worst_m) * N + worst_n]),
                ref[std::size_t(worst_m) * N + worst_n]);
    std::printf("  mean rel err %.4f\n", mean_rel);
    for (int n = 0; n < 4 && n < N; ++n) {
        std::printf("    [0][%d] got %10.5f  want %10.5f\n", n,
                    bf16_to_float(hout[n]), ref[n]);
    }
    // bf16 accumulate over K=256 leaves real slack; 2% mean is generous for
    // "computing the right thing" and nowhere near what a layout bug produces
    // (those land at O(1) relative error, i.e. uncorrelated output).
    const bool ok = mean_rel < 0.02;
    std::printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

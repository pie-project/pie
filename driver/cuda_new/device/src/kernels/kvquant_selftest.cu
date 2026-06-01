// Standalone round-trip selftest for the quantizing KV append +
// dequant-active kernels. NOT part of the CMake target.
//
// Build:
//   /usr/local/cuda/bin/nvcc -arch=sm_90 -std=c++20 -O2 -o /tmp/kvquant_selftest \
//       kv_append_quant.cu dequant_kv_active.cu kvquant_selftest.cu && \
//   LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/kvquant_selftest
//
// Flow: random bf16 K/V [T, h_kv, d] -> write_kv_to_pages_quant into a fresh
// page pool (prefill: one request, contiguous pages) -> dequant_kv_layer_to_bf16_active
// over the touched pages -> compare the dequantized values at each token/head/dim
// against the original bf16 inputs. We compute the same destination mapping on
// the host to gather the dequantized scratch back to [T, h_kv, d] order.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kv_append_quant.cuh"
#include "dequant_kv_active.cuh"

using namespace pie_cuda_device::kernels;

#define CK(call)                                                        \
    do {                                                                \
        cudaError_t e = (call);                                         \
        if (e != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, \
                         __FILE__, __LINE__, cudaGetErrorString(e));    \
            std::exit(2);                                               \
        }                                                               \
    } while (0)

struct Config {
    int total_tokens;
    int num_kv_heads;
    int head_dim;
    int page_size;
};

// Run one round-trip for a given scheme. Returns false on a tolerance failure.
static bool run_case(const char* name, KvQuantScheme scheme,
                     int storage_bytes_per_elem, double rel_tol, double abs_tol,
                     const Config& cfg) {
    const int T = cfg.total_tokens;
    const int h_kv = cfg.num_kv_heads;
    const int d = cfg.head_dim;
    const int page_size = cfg.page_size;

    // Prefill: single request, T tokens starting at absolute position 0.
    const int num_requests = 1;
    const int num_pages = (T + page_size - 1) / page_size;
    const int last_page_len = T - (num_pages - 1) * page_size;

    std::vector<std::uint32_t> qo_indptr = {0, (std::uint32_t)T};
    std::vector<std::uint32_t> kv_page_indptr = {0, (std::uint32_t)num_pages};
    std::vector<std::uint32_t> kv_last_page_lens = {(std::uint32_t)last_page_len};
    // Use a non-trivial physical page mapping to exercise the indices: active
    // page p maps to physical page (num_pages - 1 - p), so the scratch index
    // and the linear active index differ.
    std::vector<std::uint32_t> kv_page_indices(num_pages);
    for (int p = 0; p < num_pages; ++p)
        kv_page_indices[p] = (std::uint32_t)(num_pages - 1 - p);

    const long long row = (long long)h_kv * d;
    const long long n_curr = (long long)T * row;

    // Random bf16 K/V.
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.f, 1.0f);
    std::vector<__nv_bfloat16> k_host(n_curr), v_host(n_curr);
    std::vector<float> k_ref(n_curr), v_ref(n_curr);
    for (long long i = 0; i < n_curr; ++i) {
        float kf = dist(rng);
        float vf = dist(rng);
        k_host[i] = __float2bfloat16(kf);
        v_host[i] = __float2bfloat16(vf);
        // Reference is the bf16-rounded value (what the writer actually reads).
        k_ref[i] = __bfloat162float(k_host[i]);
        v_ref[i] = __bfloat162float(v_host[i]);
    }

    // Device buffers.
    __nv_bfloat16 *d_k_curr, *d_v_curr;
    CK(cudaMalloc(&d_k_curr, n_curr * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_v_curr, n_curr * sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(d_k_curr, k_host.data(), n_curr * sizeof(__nv_bfloat16),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v_curr, v_host.data(), n_curr * sizeof(__nv_bfloat16),
                  cudaMemcpyHostToDevice));

    const long long page_elems = (long long)page_size * h_kv * d;
    const long long pool_elems = (long long)num_pages * page_elems;

    void *d_k_pages, *d_v_pages;
    CK(cudaMalloc(&d_k_pages, pool_elems * storage_bytes_per_elem));
    CK(cudaMalloc(&d_v_pages, pool_elems * storage_bytes_per_elem));
    CK(cudaMemset(d_k_pages, 0, pool_elems * storage_bytes_per_elem));
    CK(cudaMemset(d_v_pages, 0, pool_elems * storage_bytes_per_elem));

    // Per-token-head scales: [num_pages * page_size * h_kv].
    const long long n_scales = (long long)num_pages * page_size * h_kv;
    float *d_k_scales = nullptr, *d_v_scales = nullptr;
    const bool has_scales = (scheme == KvQuantScheme::Int8PerTokenHead ||
                             scheme == KvQuantScheme::Fp8PerTokenHead);
    if (has_scales) {
        CK(cudaMalloc(&d_k_scales, n_scales * sizeof(float)));
        CK(cudaMalloc(&d_v_scales, n_scales * sizeof(float)));
        CK(cudaMemset(d_k_scales, 0, n_scales * sizeof(float)));
        CK(cudaMemset(d_v_scales, 0, n_scales * sizeof(float)));
    }

    // Index buffers.
    std::uint32_t *d_qo, *d_kpi, *d_kppi, *d_klpl;
    CK(cudaMalloc(&d_qo, qo_indptr.size() * 4));
    CK(cudaMalloc(&d_kpi, kv_page_indices.size() * 4));
    CK(cudaMalloc(&d_kppi, kv_page_indptr.size() * 4));
    CK(cudaMalloc(&d_klpl, kv_last_page_lens.size() * 4));
    CK(cudaMemcpy(d_qo, qo_indptr.data(), qo_indptr.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_kpi, kv_page_indices.data(), kv_page_indices.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_kppi, kv_page_indptr.data(), kv_page_indptr.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_klpl, kv_last_page_lens.data(), kv_last_page_lens.size() * 4, cudaMemcpyHostToDevice));

    // --- quantizing append ---
    write_kv_to_pages_quant(
        d_k_pages, d_v_pages, d_k_scales, d_v_scales,
        d_k_curr, d_v_curr, d_qo, d_kpi, d_kppi, d_klpl,
        T, num_requests, page_size, h_kv, d, scheme,
        KvQuantFp8Kind::E4M3, /*stream=*/0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // --- dequant active back to bf16 scratch ---
    __nv_bfloat16 *d_k_bf16, *d_v_bf16;
    CK(cudaMalloc(&d_k_bf16, pool_elems * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_v_bf16, pool_elems * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_k_bf16, 0, pool_elems * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_v_bf16, 0, pool_elems * sizeof(__nv_bfloat16)));

    dequant_kv_layer_to_bf16_active(
        d_k_pages, d_v_pages, d_k_scales, d_v_scales,
        d_k_bf16, d_v_bf16, d_kpi, num_pages,
        page_size, h_kv, d, scheme, KvQuantFp8Kind::E4M3, /*stream=*/0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> k_bf16(pool_elems), v_bf16(pool_elems);
    CK(cudaMemcpy(k_bf16.data(), d_k_bf16, pool_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(v_bf16.data(), d_v_bf16, pool_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Compare. For token t (abs pos t, prefill), head h, dim j:
    //   actual_page = kv_page_indices[t / page_size]
    //   offset_in_page = t % page_size
    //   scratch index (NHD) = ((actual_page*page_size + offset)*h_kv + h)*d + j
    //
    // Validation is per-element against the *principled* quantization bound,
    // not a single global relative error (a global rel error is meaningless for
    // small elements: a symmetric int8 grid has a fixed step independent of the
    // element's own magnitude, so a near-zero element trivially has huge rel
    // error while its abs error is still half a step). For each (token,head)
    // row we recompute the row amax exactly as the kernel does, derive the
    // expected per-element error budget, and assert every element is within it.
    //   int8 per-token-head: step = amax/127, |err| <= step/2 + slack
    //   fp8 e4m3 per-token-head: q = round(x/scale) in e4m3 (3-bit mantissa),
    //       relative step ~ 2^-3, so |err| <= |x| * 2^-3 + scale*slack
    // `rel_tol`/`abs_tol` here are interpreted as fractional/absolute slack on
    // top of that exact budget. We also report the observed worst abs error and
    // worst ratio (err / budget) for visibility.
    const bool is_int8 = (scheme == KvQuantScheme::Int8PerTokenHead);
    double max_abs = 0.0;
    double max_budget_ratio = 0.0;  // err / per-element budget; PASS iff <= 1
    for (int t = 0; t < T; ++t) {
        const int page_in_req = t / page_size;
        const int off = t % page_size;
        const int actual_page = (int)kv_page_indices[page_in_req];
        for (int h = 0; h < h_kv; ++h) {
            // Row amax (matches the kernel: max over dim of |bf16 value|).
            float k_amax = 0.f, v_amax = 0.f;
            for (int j = 0; j < d; ++j) {
                const long long src = ((long long)t * h_kv + h) * d + j;
                k_amax = std::max(k_amax, std::fabs(k_ref[src]));
                v_amax = std::max(v_amax, std::fabs(v_ref[src]));
            }
            for (int j = 0; j < d; ++j) {
                const long long src = ((long long)t * h_kv + h) * d + j;
                const long long dst =
                    (((long long)actual_page * page_size + off) * h_kv + h) * d + j;
                for (int kv = 0; kv < 2; ++kv) {
                    const float ref = kv ? v_ref[src] : k_ref[src];
                    const float amax = kv ? v_amax : k_amax;
                    const float got = kv ? __bfloat162float(v_bf16[dst])
                                         : __bfloat162float(k_bf16[dst]);
                    const double abs_err = std::fabs((double)got - ref);
                    max_abs = std::max(max_abs, abs_err);

                    // bf16 has an 8-bit mantissa => the final __float2bfloat16
                    // of the reconstructed value adds up to |value|*2^-8 of
                    // rounding error on top of the quant grid error.
                    const double bf16_round = std::fabs((double)got) * (1.0 / 256.0);
                    double budget;
                    if (is_int8) {
                        const double step = (amax > 0.f) ? (double)amax / 127.0 : 1.0;
                        // half-step quant grid + bf16 output rounding + slack.
                        budget = 0.5 * step * (1.0 + rel_tol) + bf16_round + abs_tol;
                    } else {
                        const double scale = (amax > 0.f) ? (double)amax / 448.0 : 1.0;
                        // e4m3: 3-bit mantissa => half-ULP rel 2^-4 on |x|, plus
                        // one scale-quantum, bf16 output rounding, and slack.
                        budget = (1.0 / 16.0) * std::fabs((double)ref) * (1.0 + rel_tol) +
                                 scale + bf16_round + abs_tol;
                    }
                    max_budget_ratio = std::max(max_budget_ratio, abs_err / budget);
                }
            }
        }
    }

    const bool ok = (max_budget_ratio <= 1.0);
    std::printf("[%-18s] %s  max_abs=%.5f  worst(err/budget)=%.3f  (PASS iff <=1)\n",
                name, ok ? "PASS" : "FAIL", max_abs, max_budget_ratio);

    cudaFree(d_k_curr); cudaFree(d_v_curr);
    cudaFree(d_k_pages); cudaFree(d_v_pages);
    if (has_scales) { cudaFree(d_k_scales); cudaFree(d_v_scales); }
    cudaFree(d_qo); cudaFree(d_kpi); cudaFree(d_kppi); cudaFree(d_klpl);
    cudaFree(d_k_bf16); cudaFree(d_v_bf16);
    return ok;
}

int main() {
    Config cfg{/*total_tokens=*/300, /*num_kv_heads=*/4, /*head_dim=*/128,
               /*page_size=*/16};
    std::printf("KV-quant round-trip selftest: T=%d h_kv=%d d=%d page_size=%d\n",
                cfg.total_tokens, cfg.num_kv_heads, cfg.head_dim, cfg.page_size);

    bool ok = true;
    // int8 per-token-head: assert each element is within half a quant step
    // (step = row_amax/127) plus a small bf16-rounding slack. This is the
    // correct, magnitude-independent bound for symmetric int8.
    ok &= run_case("Int8PerTokenHead", KvQuantScheme::Int8PerTokenHead,
                   /*bytes=*/1, /*rel_slack=*/0.15, /*abs_slack=*/0.002, cfg);
    // fp8 e4m3 per-token-head: 3-bit mantissa => ~half-ULP relative budget plus
    // one scale-quantum of slack for the per-row rescaling.
    ok &= run_case("Fp8PerTokenHead", KvQuantScheme::Fp8PerTokenHead,
                   /*bytes=*/1, /*rel_slack=*/0.10, /*abs_slack=*/0.002, cfg);

    std::printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Quest envelope kernels: correctness vs the CPU golden.
//
// The CPU references here MIRROR `pie_sampling_ir::eval::envelope_dot_reference`
// (the canonical golden, verified in `interface/sampling-ir` with the SAME
// hand-computed vector this file cross-checks). Two checks:
//   1. `envelope_recompute` == per-(page,kv_head,dim) min/max over live keys.
//   2. `envelope_dot` == Σ_group Σ_dim max(q·min, q·max), −inf beyond live.
//
// Requires a CUDA device.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kernels/envelope.hpp"

using pie_cuda_driver::kernels::launch_envelope_dot_f32;
using pie_cuda_driver::kernels::launch_envelope_recompute_bf16;

namespace {

int g_fail = 0;

void rt(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL %s:%d: %s -> %s\n", __FILE__, line, what,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define RT(e) rt((e), #e, __LINE__)

// bf16 round-trip (so the CPU reference sees the exact bytes the kernel reads).
std::uint16_t f2b(float f) {
    __nv_bfloat16 h = __float2bfloat16(f);
    std::uint16_t u;
    __builtin_memcpy(&u, &h, sizeof(u));
    return u;
}
float b2f(std::uint16_t u) {
    __nv_bfloat16 h;
    __builtin_memcpy(&h, &u, sizeof(h));
    return __bfloat162float(h);
}

// CPU golden — mirrors envelope_dot_reference (per-kv_head SUM over the GQA group).
std::vector<float> cpu_envelope_dot(
    const std::vector<float>& q, const std::vector<float>& emin,
    const std::vector<float>& emax, int nqh, int nkvh, int hd, int p_max,
    int live) {
    const int group = nqh / nkvh;
    std::vector<float> out(static_cast<std::size_t>(nkvh) * p_max,
                           -INFINITY);
    for (int kh = 0; kh < nkvh; ++kh)
        for (int p = 0; p < live && p < p_max; ++p) {
            const long eb = (static_cast<long>(p) * nkvh + kh) * hd;
            float acc = 0.f;
            for (int g = 0; g < group; ++g) {
                const long qb = static_cast<long>(kh * group + g) * hd;
                for (int d = 0; d < hd; ++d) {
                    const float qd = q[qb + d];
                    const float lo = qd * emin[eb + d];
                    const float hi = qd * emax[eb + d];
                    acc += (lo > hi) ? lo : hi;
                }
            }
            out[static_cast<long>(kh) * p_max + p] = acc;
        }
    return out;
}

void check_recompute(const char* name, int num_pages, int page_size, int nkvh,
                     int hd, const std::vector<int>& live) {
    const long tok = static_cast<long>(nkvh) * hd;
    const long n = static_cast<long>(num_pages) * page_size * tok;
    std::vector<std::uint16_t> k(n);
    for (long g = 0; g < n; ++g)
        k[g] = f2b(std::sin(0.01f * static_cast<float>(g)) * 5.0f);

    std::uint16_t* dk;
    std::int32_t* dlive;
    float *dmin, *dmax;
    const long envn = static_cast<long>(num_pages) * nkvh * hd;
    RT(cudaMalloc(&dk, n * 2));
    RT(cudaMalloc(&dlive, num_pages * sizeof(std::int32_t)));
    RT(cudaMalloc(&dmin, envn * sizeof(float)));
    RT(cudaMalloc(&dmax, envn * sizeof(float)));
    RT(cudaMemcpy(dk, k.data(), n * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dlive, live.data(), num_pages * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice));

    launch_envelope_recompute_bf16(dk, dlive, dmin, dmax, num_pages, page_size,
                                   nkvh, hd, nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<float> gmin(envn), gmax(envn);
    RT(cudaMemcpy(gmin.data(), dmin, envn * sizeof(float),
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(gmax.data(), dmax, envn * sizeof(float),
                  cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int p = 0; p < num_pages && ok; ++p)
        for (int kh = 0; kh < nkvh && ok; ++kh)
            for (int d = 0; d < hd && ok; ++d) {
                float mn = INFINITY, mx = -INFINITY;
                for (int t = 0; t < live[p]; ++t) {
                    const long idx =
                        ((static_cast<long>(p) * page_size + t) * nkvh + kh) *
                            hd + d;
                    const float v = b2f(k[idx]);
                    mn = std::fmin(mn, v);
                    mx = std::fmax(mx, v);
                }
                const long e = (static_cast<long>(p) * nkvh + kh) * hd + d;
                ok = (gmin[e] == mn) && (gmax[e] == mx);
            }
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", name);
    if (!ok) ++g_fail;
    cudaFree(dk); cudaFree(dlive); cudaFree(dmin); cudaFree(dmax);
}

void check_dot(const char* name, int nqh, int nkvh, int hd, int p_max,
               int live) {
    const long envn = static_cast<long>(p_max) * nkvh * hd;
    std::vector<float> q(static_cast<std::size_t>(nqh) * hd);
    std::vector<float> emin(envn), emax(envn);
    for (std::size_t i = 0; i < q.size(); ++i)
        q[i] = std::cos(0.017f * static_cast<float>(i)) * 2.0f;
    for (long i = 0; i < envn; ++i) {
        const float a = std::sin(0.013f * static_cast<float>(i));
        emin[i] = a - 1.5f;
        emax[i] = a + 2.0f;
    }

    float *dq, *dmin, *dmax, *dscore;
    const long sn = static_cast<long>(nkvh) * p_max;
    RT(cudaMalloc(&dq, q.size() * sizeof(float)));
    RT(cudaMalloc(&dmin, envn * sizeof(float)));
    RT(cudaMalloc(&dmax, envn * sizeof(float)));
    RT(cudaMalloc(&dscore, sn * sizeof(float)));
    RT(cudaMemcpy(dq, q.data(), q.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dmin, emin.data(), envn * sizeof(float),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dmax, emax.data(), envn * sizeof(float),
                  cudaMemcpyHostToDevice));

    launch_envelope_dot_f32(dq, dmin, dmax, dscore, nqh, nkvh, hd, p_max, live,
                            nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<float> gs(sn);
    RT(cudaMemcpy(gs.data(), dscore, sn * sizeof(float),
                  cudaMemcpyDeviceToHost));
    const std::vector<float> ref =
        cpu_envelope_dot(q, emin, emax, nqh, nkvh, hd, p_max, live);

    bool ok = true;
    for (long i = 0; i < sn && ok; ++i) {
        if (std::isinf(ref[i])) {
            ok = std::isinf(gs[i]) && (gs[i] < 0);  // both −inf beyond live
        } else {
            // The reduction order differs (grid-stride tree vs sequential), so
            // allow a tiny f32 rounding tolerance; not a bit-exact contract.
            ok = std::fabs(gs[i] - ref[i]) <=
                 1e-3f * (1.0f + std::fabs(ref[i]));
        }
    }
    std::printf("[%s] %s (nqh=%d nkvh=%d hd=%d P_MAX=%d live=%d)\n",
                ok ? " ok " : "FAIL", name, nqh, nkvh, hd, p_max, live);
    if (!ok) ++g_fail;
    cudaFree(dq); cudaFree(dmin); cudaFree(dmax); cudaFree(dscore);
}

// Cross-check the exact hand-computed vector from the Rust golden test
// (envelope_dot_reference_quest_math): score[0,0]=11, [0,1]=3, [0,2]=−inf.
void check_dot_golden_vector() {
    const std::vector<float> q{1.0f, -1.0f, 2.0f, 0.5f};
    const std::vector<float> emin{0.0f, 0.0f, -2.0f, 1.0f, 9.9f, 9.9f};
    const std::vector<float> emax{3.0f, 4.0f, 1.0f, 2.0f, 9.9f, 9.9f};
    float *dq, *dmin, *dmax, *ds;
    RT(cudaMalloc(&dq, 4 * sizeof(float)));
    RT(cudaMalloc(&dmin, 6 * sizeof(float)));
    RT(cudaMalloc(&dmax, 6 * sizeof(float)));
    RT(cudaMalloc(&ds, 3 * sizeof(float)));
    RT(cudaMemcpy(dq, q.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dmin, emin.data(), 6 * sizeof(float), cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dmax, emax.data(), 6 * sizeof(float), cudaMemcpyHostToDevice));
    launch_envelope_dot_f32(dq, dmin, dmax, ds, 2, 1, 2, 3, 2, nullptr);
    RT(cudaDeviceSynchronize());
    float s[3];
    RT(cudaMemcpy(s, ds, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    const bool ok = (s[0] == 11.0f) && (s[1] == 3.0f) &&
                    (std::isinf(s[2]) && s[2] < 0);
    std::printf("[%s] golden vector cross-check (11, 3, -inf) got (%.1f, %.1f, %.1f)\n",
                ok ? " ok " : "FAIL", s[0], s[1], s[2]);
    if (!ok) ++g_fail;
    cudaFree(dq); cudaFree(dmin); cudaFree(dmax); cudaFree(ds);
}

}  // namespace

int main() {
    RT(cudaSetDevice(0));

    check_recompute("recompute: full + partial pages", 6, 16, 8, 128,
                    {16, 1, 9, 16, 5, 0});
    check_recompute("recompute: head_dim 64", 4, 16, 4, 64, {16, 3, 16, 8});

    check_dot_golden_vector();
    check_dot("dot: GQA 16q/8kv, P_MAX 32", 16, 8, 128, 32, 20);
    check_dot("dot: MHA 8/8", 8, 8, 128, 16, 16);
    check_dot("dot: live < P_MAX (−inf tail)", 16, 8, 128, 64, 7);

    std::printf(g_fail ? "\nENVELOPE FAILED (%d)\n" : "\nALL PASS (0 failures)\n",
                g_fail);
    return g_fail ? 1 : 0;
}

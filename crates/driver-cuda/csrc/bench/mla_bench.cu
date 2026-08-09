// Standalone microbenchmark + reference check for the naive paged MLA kernel.
//
// Build (no CMake, no engine rebuild -- seconds, not minutes):
//   nvcc -O3 -std=c++17 -arch=sm_100 -I crates/driver-cuda/csrc/src \
//        -o /tmp/mla_bench crates/driver-cuda/csrc/bench/mla_bench.cu
//
// Run:
//   /tmp/mla_bench decode  128 64 51   # tokens heads kv_len
//   /tmp/mla_bench prefill 869 64 1    # qlen heads requests (causal)
//   /tmp/mla_bench check   16  8  33   # vs CPU reference
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "attn/attention_mla_naive.cuh"

using pie_cuda_driver::kernels::attn::mla_naive::launch_mla_naive_paged_raw;

namespace {

constexpr int kCkv = 512;
constexpr int kKpe = 64;
constexpr int kPageSize = 16;

#define CHK(x)                                                              \
    do {                                                                    \
        cudaError_t e_ = (x);                                               \
        if (e_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(e_));                           \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

struct Case {
    int total_tokens = 0;
    int num_requests = 0;
    int num_heads = 64;
    bool causal = false;
    std::vector<std::uint32_t> qo_indptr;
    std::vector<std::uint32_t> kv_page_indptr;
    std::vector<std::uint32_t> kv_page_indices;
    std::vector<std::uint32_t> kv_last_page_lens;
    std::vector<int> kv_len;
    int total_pages = 0;
};

Case make_case(bool prefill, int reqs, int qlen, int kv, int heads) {
    Case c;
    c.num_requests = reqs;
    c.num_heads = heads;
    c.causal = prefill;
    c.qo_indptr.push_back(0);
    c.kv_page_indptr.push_back(0);
    for (int r = 0; r < reqs; ++r) {
        const int q = prefill ? qlen : 1;
        const int klen = prefill ? qlen : kv;
        c.total_tokens += q;
        c.qo_indptr.push_back(static_cast<std::uint32_t>(c.total_tokens));
        c.kv_len.push_back(klen);
        const int pages = (klen + kPageSize - 1) / kPageSize;
        for (int p = 0; p < pages; ++p)
            c.kv_page_indices.push_back(static_cast<std::uint32_t>(c.total_pages++));
        c.kv_page_indptr.push_back(
            static_cast<std::uint32_t>(c.kv_page_indices.size()));
        c.kv_last_page_lens.push_back(
            static_cast<std::uint32_t>(klen - (pages - 1) * kPageSize));
    }
    return c;
}

template <typename T>
T* to_dev(const std::vector<T>& v) {
    T* d = nullptr;
    CHK(cudaMalloc(&d, v.size() * sizeof(T)));
    CHK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

float ref_max_abs_diff(const Case& c, const std::vector<float>& q_nope,
                       const std::vector<float>& q_pe,
                       const std::vector<float>& ckv,
                       const std::vector<float>& kpe,
                       const std::vector<float>& got, float sm_scale) {
    float worst = 0.f;
    std::vector<float> acc(kCkv);
    for (int r = 0; r < c.num_requests; ++r) {
        const int qlo = static_cast<int>(c.qo_indptr[r]);
        const int qn = static_cast<int>(c.qo_indptr[r + 1]) - qlo;
        const int klen = c.kv_len[r];
        const int pfirst = static_cast<int>(c.kv_page_indptr[r]);
        for (int ti = 0; ti < qn; ++ti) {
            const int t = qlo + ti;
            const int abs_q = klen - qn + ti;
            const int jend = c.causal ? abs_q + 1 : klen;
            for (int h = 0; h < c.num_heads; ++h) {
                float m = -INFINITY, l = 0.f;
                std::fill(acc.begin(), acc.end(), 0.f);
                const float* qnr =
                    &q_nope[(static_cast<long long>(t) * c.num_heads + h) * kCkv];
                const float* qpr =
                    &q_pe[(static_cast<long long>(t) * c.num_heads + h) * kKpe];
                for (int j = 0; j < jend; ++j) {
                    const long long page = c.kv_page_indices[pfirst + j / kPageSize];
                    const long long off = j % kPageSize;
                    const float* kv_r = &ckv[(page * kPageSize + off) * kCkv];
                    const float* kp_r = &kpe[(page * kPageSize + off) * kKpe];
                    float d = 0.f;
                    for (int i = 0; i < kCkv; ++i) d += qnr[i] * kv_r[i];
                    for (int i = 0; i < kKpe; ++i) d += qpr[i] * kp_r[i];
                    const float s = d * sm_scale;
                    const float mn = std::fmax(m, s);
                    const float corr = std::exp(m - mn);
                    const float p = std::exp(s - mn);
                    l = l * corr + p;
                    for (int i = 0; i < kCkv; ++i)
                        acc[i] = acc[i] * corr + p * kv_r[i];
                    m = mn;
                }
                const float inv = l > 0.f ? 1.f / l : 0.f;
                const float* g =
                    &got[(static_cast<long long>(t) * c.num_heads + h) * kCkv];
                for (int i = 0; i < kCkv; ++i)
                    worst = std::fmax(worst, std::fabs(acc[i] * inv - g[i]));
            }
        }
    }
    return worst;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "decode";
    const bool check = mode.rfind("check", 0) == 0;
    const bool prefill = mode == "prefill" || mode == "checkp";
    const int a1 = argc > 2 ? std::atoi(argv[2]) : (prefill ? 869 : 128);
    const int a2 = argc > 3 ? std::atoi(argv[3]) : 64;
    const int a3 = argc > 4 ? std::atoi(argv[4]) : (prefill ? 1 : 51);

    Case c = make_case(prefill, prefill ? a3 : a1, prefill ? a1 : 1, a3, a2);

    const long long qelems = static_cast<long long>(c.total_tokens) * c.num_heads;
    const long long kelems = static_cast<long long>(c.total_pages) * kPageSize;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto mk = [&](long long n) {
        std::vector<float> v(static_cast<size_t>(n));
        for (auto& x : v) x = __bfloat162float(__float2bfloat16(nd(rng) * 0.05f));
        return v;
    };
    std::vector<float> q_nope_f = mk(qelems * kCkv);
    std::vector<float> q_pe_f = mk(qelems * kKpe);
    std::vector<float> ckv_f = mk(kelems * kCkv);
    std::vector<float> kpe_f = mk(kelems * kKpe);

    auto pack = [](const std::vector<float>& f) {
        std::vector<__nv_bfloat16> b(f.size());
        for (size_t i = 0; i < f.size(); ++i) b[i] = __float2bfloat16(f[i]);
        return b;
    };
    __nv_bfloat16* d_qn = to_dev(pack(q_nope_f));
    __nv_bfloat16* d_qp = to_dev(pack(q_pe_f));
    __nv_bfloat16* d_ckv = to_dev(pack(ckv_f));
    __nv_bfloat16* d_kpe = to_dev(pack(kpe_f));
    __nv_bfloat16* d_o = nullptr;
    CHK(cudaMalloc(&d_o, qelems * kCkv * sizeof(__nv_bfloat16)));
    auto* d_qoi = to_dev(c.qo_indptr);
    auto* d_kpi = to_dev(c.kv_page_indptr);
    auto* d_kpx = to_dev(c.kv_page_indices);
    auto* d_klp = to_dev(c.kv_last_page_lens);

    const float sm_scale = 1.f / std::sqrt(static_cast<float>(kCkv + kKpe));
    auto run = [&] {
        launch_mla_naive_paged_raw(d_qn, d_qp, d_ckv, d_kpe, kCkv, kKpe,
                                   kPageSize, d_o, d_qoi, d_kpx, d_kpi, d_klp,
                                   c.total_tokens, c.num_requests, c.num_heads,
                                   sm_scale, c.causal, nullptr, nullptr, 0);
    };

    run();
    CHK(cudaDeviceSynchronize());

    if (check) {
        std::vector<__nv_bfloat16> ob(static_cast<size_t>(qelems) * kCkv);
        CHK(cudaMemcpy(ob.data(), d_o, ob.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost));
        std::vector<float> of(ob.size());
        for (size_t i = 0; i < ob.size(); ++i) of[i] = __bfloat162float(ob[i]);
        const float d =
            ref_max_abs_diff(c, q_nope_f, q_pe_f, ckv_f, kpe_f, of, sm_scale);
        std::printf("check mode=%s tokens=%d reqs=%d heads=%d max_abs_diff=%.3e %s\n",
                    mode.c_str(), c.total_tokens, c.num_requests, c.num_heads, d,
                    d < 6e-3f ? "OK" : "FAIL");
        return d < 6e-3f ? 0 : 1;
    }

    const int iters = 20;
    for (int i = 0; i < 3; ++i) run();
    CHK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CHK(cudaEventCreate(&e0));
    CHK(cudaEventCreate(&e1));
    CHK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) run();
    CHK(cudaEventRecord(e1));
    CHK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CHK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= iters;

    long long keyvisits = 0;
    for (int r = 0; r < c.num_requests; ++r) {
        const int qn = static_cast<int>(c.qo_indptr[r + 1] - c.qo_indptr[r]);
        const int klen = c.kv_len[r];
        if (c.causal) {
            for (int i = 0; i < qn; ++i) keyvisits += klen - qn + i + 1;
        } else {
            keyvisits += static_cast<long long>(qn) * klen;
        }
    }
    const double flops = 4.0 * static_cast<double>(keyvisits) * c.num_heads *
                         static_cast<double>(kCkv) +
                         2.0 * static_cast<double>(keyvisits) * c.num_heads * kKpe;
    const double bytes = static_cast<double>(keyvisits) * (kCkv + kKpe) * 2.0;
    std::printf(
        "mode=%-7s tokens=%6d reqs=%4d heads=%3d kv=%5d  %8.3f ms  %6.1f TFLOP/s  "
        "%7.1f TB/s-eff\n",
        mode.c_str(), c.total_tokens, c.num_requests, c.num_heads,
        prefill ? a1 : a3, ms, flops / (ms * 1e-3) / 1e12,
        bytes / (ms * 1e-3) / 1e12);
    return 0;
}

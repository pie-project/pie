// The fused QKV postprocess used to be pure-decode only, so a prefill fire ran
// the unfused chain instead: split_qkv -> qk_rmsnorm_rope -> write_kv_to_pages.
// Those three kernels were 16% of GPU time on a prefill-heavy run. The fused
// kernel now accepts `qo_indptr` and serves prefill too.
//
// That generalisation is only safe if the fused kernel picks the SAME KV cell
// the unfused chain would have. Multi-token requests make that non-trivial:
// a row's slot is `total_kv_after - new_tokens_r + offset_in_new`, not simply
// "the last occupied slot". This test runs both paths over a ragged fire whose
// requests differ in token count, pre-existing KV length, and page count, and
// asserts they agree on Q, on every written K/V cell, and on which cells were
// left alone.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/kv_paged.hpp"
#include "kernels/rope.hpp"
#include "kernels/split_packed.hpp"

namespace k = pie_cuda_driver::kernels;

namespace {

void rt_check(cudaError_t e, const char* expr, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL rt %s:%d: %s -> %s\n", __FILE__, line, expr,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define RT(expr) rt_check((expr), #expr, __LINE__)

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    const std::uint32_t lsb = (b >> 16) & 1u;
    b += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(b >> 16);
}
float bf16_to_f32(std::uint16_t h) {
    std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

constexpr int HQ = 16, HKV = 8, D = 128, PAGE = 16;
constexpr int Q_DIM = HQ * D, KV_DIM = HKV * D, PACKED = Q_DIM + 2 * KV_DIM;
constexpr float THETA = 1.0e6f, EPS = 1.0e-6f;
// A sentinel no norm+rope output can land on, so an untouched cell is
// distinguishable from a written one.
constexpr float CANARY = -123.5f;

// A ragged fire. `new_tokens` is what this fire appends; `pre_kv` is what the
// request already had cached. The mix deliberately covers: a request whose
// span crosses a page boundary, a single-token (decode-shaped) request inside
// a prefill fire, a request starting exactly on a page boundary, and a
// request whose pre-existing KV is zero.
struct ReqShape {
    int new_tokens;
    int pre_kv;
};
const ReqShape kShapes[] = {
    {5, 14},   // crosses the page-0 -> page-1 boundary
    {1, 31},   // decode-shaped row, lands on the last slot of page 1
    {19, 0},   // fresh prefill spanning two page boundaries
    {3, 16},   // starts exactly on a page boundary
    {7, 45},   // deep in page 2
};
constexpr int R = static_cast<int>(sizeof(kShapes) / sizeof(kShapes[0]));

struct Geometry {
    std::vector<std::uint32_t> qo_indptr;        // R+1
    std::vector<std::uint32_t> page_indptr;      // R+1
    std::vector<std::uint32_t> page_indices;     // page_indptr[R]
    std::vector<std::uint32_t> last_page_lens;   // R
    std::vector<std::int32_t> positions;         // N
    int n_tokens = 0;
    int total_pages = 0;
};

Geometry build_geometry() {
    Geometry g;
    g.qo_indptr.push_back(0);
    g.page_indptr.push_back(0);
    for (int r = 0; r < R; ++r) {
        const int total_kv = kShapes[r].pre_kv + kShapes[r].new_tokens;
        const int pages = (total_kv + PAGE - 1) / PAGE;
        for (int p = 0; p < pages; ++p) {
            // Deliberately non-contiguous physical pages: a fused kernel that
            // assumed page ids were `pages_first + p` would pass with an
            // identity mapping and corrupt real fires.
            g.page_indices.push_back(
                static_cast<std::uint32_t>(g.total_pages + p) * 2u + 1u);
        }
        g.total_pages += pages;
        g.page_indptr.push_back(static_cast<std::uint32_t>(g.page_indices.size()));
        int last_len = total_kv - (pages - 1) * PAGE;
        g.last_page_lens.push_back(static_cast<std::uint32_t>(last_len));
        for (int t = 0; t < kShapes[r].new_tokens; ++t) {
            g.positions.push_back(kShapes[r].pre_kv + t);
        }
        g.n_tokens += kShapes[r].new_tokens;
        g.qo_indptr.push_back(static_cast<std::uint32_t>(g.n_tokens));
    }
    return g;
}

struct Buffers {
    void* packed = nullptr;
    void* qw = nullptr;
    void* kw = nullptr;
    std::int32_t* positions = nullptr;
    std::uint32_t* qo_indptr = nullptr;
    std::uint32_t* page_indices = nullptr;
    std::uint32_t* page_indptr = nullptr;
    std::uint32_t* last_page_lens = nullptr;
};

struct Result {
    std::vector<float> q_out;
    std::vector<float> k_pages;
    std::vector<float> v_pages;
};

// Physical page ids are sparse (2p+1), so the page pool must cover the max id.
int page_pool_size(const Geometry& g) {
    std::uint32_t hi = 0;
    for (auto p : g.page_indices) hi = p > hi ? p : hi;
    return static_cast<int>(hi) + 1;
}

Result run(const Geometry& g, const Buffers& b, bool fused) {
    const int N = g.n_tokens;
    const int pages = page_pool_size(g);
    const std::size_t page_stride = static_cast<std::size_t>(PAGE) * HKV * D;
    const std::size_t kv_elems = static_cast<std::size_t>(pages) * page_stride;

    void *d_qout = nullptr, *d_k = nullptr, *d_v = nullptr;
    void *d_q_split = nullptr, *d_k_split = nullptr, *d_v_split = nullptr;
    RT(cudaMalloc(&d_qout, static_cast<std::size_t>(N) * Q_DIM * 2));
    RT(cudaMalloc(&d_k, kv_elems * 2));
    RT(cudaMalloc(&d_v, kv_elems * 2));

    std::vector<std::uint16_t> canary(kv_elems, f32_to_bf16(CANARY));
    RT(cudaMemcpy(d_k, canary.data(), kv_elems * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_v, canary.data(), kv_elems * 2, cudaMemcpyHostToDevice));
    RT(cudaMemset(d_qout, 0, static_cast<std::size_t>(N) * Q_DIM * 2));

    if (fused) {
        k::launch_qkv_qk_norm_rope_write_kv_bf16(
            b.packed, d_qout, d_k, d_v, b.qw, b.kw, b.positions,
            /*rope_table=*/nullptr, b.qo_indptr, b.page_indices, b.page_indptr,
            b.last_page_lens, /*w_page=*/nullptr, /*w_off=*/nullptr,
            /*row_valid=*/nullptr, N, R, HQ, HKV, D, PAGE,
            /*hnd_layout=*/false, THETA, EPS, /*stream=*/nullptr);
    } else {
        RT(cudaMalloc(&d_q_split, static_cast<std::size_t>(N) * Q_DIM * 2));
        RT(cudaMalloc(&d_k_split, static_cast<std::size_t>(N) * KV_DIM * 2));
        RT(cudaMalloc(&d_v_split, static_cast<std::size_t>(N) * KV_DIM * 2));
        k::launch_split_qkv_bf16(b.packed, d_q_split, d_k_split, d_v_split,
                                 N, Q_DIM, KV_DIM, /*stream=*/nullptr);
        k::launch_qk_rmsnorm_rope_bf16(d_q_split, d_k_split, b.qw, b.kw,
                                       b.positions, N, HQ, HKV, D, THETA, EPS,
                                       /*stream=*/nullptr);
        k::launch_write_kv_to_pages_bf16(
            d_k, d_v, d_k_split, d_v_split, b.qo_indptr, b.page_indices,
            b.page_indptr, b.last_page_lens, N, R, PAGE, HKV, D,
            /*hnd_layout=*/false, /*stream=*/nullptr, /*row_valid=*/nullptr);
        RT(cudaMemcpy(d_qout, d_q_split,
                      static_cast<std::size_t>(N) * Q_DIM * 2,
                      cudaMemcpyDeviceToDevice));
    }
    RT(cudaDeviceSynchronize());
    RT(cudaGetLastError());

    std::vector<std::uint16_t> qo(static_cast<std::size_t>(N) * Q_DIM);
    std::vector<std::uint16_t> kp(kv_elems), vp(kv_elems);
    RT(cudaMemcpy(qo.data(), d_qout, qo.size() * 2, cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(kp.data(), d_k, kp.size() * 2, cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(vp.data(), d_v, vp.size() * 2, cudaMemcpyDeviceToHost));

    Result res;
    res.q_out.resize(qo.size());
    res.k_pages.resize(kp.size());
    res.v_pages.resize(vp.size());
    for (std::size_t i = 0; i < qo.size(); ++i) res.q_out[i] = bf16_to_f32(qo[i]);
    for (std::size_t i = 0; i < kp.size(); ++i) res.k_pages[i] = bf16_to_f32(kp[i]);
    for (std::size_t i = 0; i < vp.size(); ++i) res.v_pages[i] = bf16_to_f32(vp[i]);

    cudaFree(d_qout);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q_split);
    cudaFree(d_k_split);
    cudaFree(d_v_split);
    return res;
}

float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

std::size_t count_written(const std::vector<float>& v) {
    std::size_t n = 0;
    for (float x : v)
        if (x != CANARY) ++n;
    return n;
}

}  // namespace

int main() {
    const Geometry g = build_geometry();
    const int N = g.n_tokens;

    std::mt19937 rng(0x9E3779B9u);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::uniform_real_distribution<float> w(0.5f, 1.5f);

    std::vector<std::uint16_t> packed(static_cast<std::size_t>(N) * PACKED);
    for (auto& x : packed) x = f32_to_bf16(u(rng));
    std::vector<std::uint16_t> qw(D), kw(D);
    for (auto& x : qw) x = f32_to_bf16(w(rng));
    for (auto& x : kw) x = f32_to_bf16(w(rng));

    Buffers b;
    RT(cudaMalloc(&b.packed, packed.size() * 2));
    RT(cudaMalloc(&b.qw, qw.size() * 2));
    RT(cudaMalloc(&b.kw, kw.size() * 2));
    RT(cudaMalloc(&b.positions, static_cast<std::size_t>(N) * 4));
    RT(cudaMalloc(&b.qo_indptr, g.qo_indptr.size() * 4));
    RT(cudaMalloc(&b.page_indices, g.page_indices.size() * 4));
    RT(cudaMalloc(&b.page_indptr, g.page_indptr.size() * 4));
    RT(cudaMalloc(&b.last_page_lens, g.last_page_lens.size() * 4));
    RT(cudaMemcpy(b.packed, packed.data(), packed.size() * 2,
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(b.qw, qw.data(), qw.size() * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(b.kw, kw.data(), kw.size() * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(b.positions, g.positions.data(),
                  static_cast<std::size_t>(N) * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(b.qo_indptr, g.qo_indptr.data(), g.qo_indptr.size() * 4,
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(b.page_indices, g.page_indices.data(),
                  g.page_indices.size() * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(b.page_indptr, g.page_indptr.data(), g.page_indptr.size() * 4,
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(b.last_page_lens, g.last_page_lens.data(),
                  g.last_page_lens.size() * 4, cudaMemcpyHostToDevice));

    const Result reference = run(g, b, /*fused=*/false);
    const Result fused = run(g, b, /*fused=*/true);

    // bf16 quantum at magnitude 1 is 2^-8 ≈ 3.9e-3. The two paths round at the
    // same points but accumulate the rope rotation in a different fp32 order
    // (warp-shuffle pairing vs direct indexing), so allow two quanta. A wrong
    // KV cell would differ by O(1), not by ulps.
    const float tol = 8e-3f;
    const float dq = maxdiff(reference.q_out, fused.q_out);
    const float dk = maxdiff(reference.k_pages, fused.k_pages);
    const float dv = maxdiff(reference.v_pages, fused.v_pages);
    const std::size_t wk_ref = count_written(reference.k_pages);
    const std::size_t wk_fus = count_written(fused.k_pages);
    const std::size_t expect = static_cast<std::size_t>(N) * KV_DIM;

    bool ok = dq <= tol && dk <= tol && dv <= tol;
    std::printf("[%s] fused prefill == unfused chain  "
                "q Δ=%.5g  K Δ=%.5g  V Δ=%.5g\n",
                ok ? " ok " : "FAIL", dq, dk, dv);

    // A path that wrote too few cells (a row silently dropped) or too many
    // (a row landing on a neighbour's slot) is a correctness bug even when the
    // cells it did write happen to match.
    const bool counts_ok = wk_ref == expect && wk_fus == expect;
    ok = ok && counts_ok;
    std::printf("[%s] written K cells: reference %zu, fused %zu, expected %zu "
                "(%d rows x %d)\n",
                counts_ok ? " ok " : "FAIL", wk_ref, wk_fus, expect, N, KV_DIM);

    cudaFree(b.packed);
    cudaFree(b.qw);
    cudaFree(b.kw);
    cudaFree(b.positions);
    cudaFree(b.qo_indptr);
    cudaFree(b.page_indices);
    cudaFree(b.page_indptr);
    cudaFree(b.last_page_lens);

    if (!ok) {
        std::printf("\nFUSED prefill postprocess DIVERGES from the unfused "
                    "split/rope/write chain.\n");
        return 1;
    }
    std::printf("\nFused prefill postprocess matches the unfused chain over a "
                "ragged %d-token / %d-request fire.\n", N, R);
    return 0;
}

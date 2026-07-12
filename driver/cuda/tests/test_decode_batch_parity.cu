// Bug#2 reproducer + permanent guard: batched (R>1) paged decode attention must
// match the per-request R=1 reference bit-for-bit.
//
// The concurrent-decode corruption (reported on a 4090: R=1 perfect, R>=2
// cross-scrambled/garbage logits despite correct geometry) is per-request KV
// mis-attribution INSIDE the R>1 BatchDecode kernel/dispatch. This test isolates
// the attention READ path from all model/scheduler/KV-write machinery: it
// hand-builds R distinct requests' paged KV (disjoint pages, distinct random
// content + distinct seq-lens), runs ONE batched `dispatch_attention_flashinfer_decode_bf16`
// (R>1), and compares each request's output to the SAME request run alone (R=1).
//
// A correct kernel gives request r the identical output whether batched or
// alone. A divergence pins the bug to the attention read/dispatch (this test),
// vs the fused KV-write kernel (would pass here). Runs the exact plan+dispatch
// the production sm_89 pure-decode path uses.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "ops/attention_flashinfer.hpp"

using pie_cuda_driver::AttentionWorkspace;
namespace ops = pie_cuda_driver::ops;

namespace {

int g_failures = 0;

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

// Qwen3-0.6B decode shape: 16 q heads / 8 kv heads (GQA group 2), head_dim 128,
// page_size 16 — the exact geometry of the failing repro.
constexpr int HQ = 16, HKV = 8, D = 128, PAGE = 16;
const std::size_t PAGE_STRIDE = static_cast<std::size_t>(PAGE) * HKV * D;  // NHD

// One request's hand-built decode problem: `kv_len` prior tokens of KV + 1 query.
struct Req {
    int kv_len;
    std::vector<std::uint16_t> q_bf;    // [HQ, D] bf16 (the single decode query)
    std::vector<std::uint16_t> k_bf;    // [kv_len, HKV, D] bf16 (token-major)
    std::vector<std::uint16_t> v_bf;    // [kv_len, HKV, D] bf16
};

Req make_req(std::mt19937& rng, int kv_len) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    Req r;
    r.kv_len = kv_len;
    r.q_bf.resize(static_cast<std::size_t>(HQ) * D);
    for (auto& x : r.q_bf) x = f32_to_bf16(u(rng));
    r.k_bf.resize(static_cast<std::size_t>(kv_len) * HKV * D);
    r.v_bf.resize(static_cast<std::size_t>(kv_len) * HKV * D);
    for (auto& x : r.k_bf) x = f32_to_bf16(u(rng));
    for (auto& x : r.v_bf) x = f32_to_bf16(u(rng));
    return r;
}

// Run a batched decode over `reqs[first..first+R)`. Fills `out` with the R
// per-request output rows ([R, HQ, D] f32). Pages are laid out disjointly, one
// contiguous page block per request, mirroring the runtime's page assignment.
std::vector<float> run_batch(const std::vector<Req>& reqs, int first, int R, bool full_attn) {
    // Geometry (host).
    std::vector<std::uint32_t> kv_page_indptr_h(R + 1, 0);
    std::vector<std::uint32_t> kv_last_page_lens_h(R);
    for (int i = 0; i < R; ++i) {
        const int kv = reqs[first + i].kv_len;
        const int pages = (kv + PAGE - 1) / PAGE;
        kv_page_indptr_h[i + 1] = kv_page_indptr_h[i] + pages;
        kv_last_page_lens_h[i] = static_cast<std::uint32_t>(kv - (pages - 1) * PAGE);
    }
    const int total_pages = static_cast<int>(kv_page_indptr_h[R]);
    std::vector<std::uint32_t> kv_page_indices_h(total_pages);
    for (int p = 0; p < total_pages; ++p) kv_page_indices_h[p] = static_cast<std::uint32_t>(p);

    // Pack Q [R,HQ,D] and paged K/V [total_pages, PAGE, HKV, D] (NHD, zero-filled).
    std::vector<std::uint16_t> q_bf(static_cast<std::size_t>(R) * HQ * D);
    std::vector<std::uint16_t> kbuf(static_cast<std::size_t>(total_pages) * PAGE_STRIDE, 0);
    std::vector<std::uint16_t> vbuf(static_cast<std::size_t>(total_pages) * PAGE_STRIDE, 0);
    for (int i = 0; i < R; ++i) {
        const Req& rq = reqs[first + i];
        std::memcpy(q_bf.data() + static_cast<std::size_t>(i) * HQ * D, rq.q_bf.data(),
                    rq.q_bf.size() * 2);
        const int base_page = static_cast<int>(kv_page_indptr_h[i]);
        for (int j = 0; j < rq.kv_len; ++j) {
            const int pg = base_page + j / PAGE, slot = j % PAGE;
            for (int h = 0; h < HKV; ++h)
                for (int d = 0; d < D; ++d) {
                    const std::size_t dst = static_cast<std::size_t>(pg) * PAGE_STRIDE +
                                            static_cast<std::size_t>(slot) * HKV * D +
                                            static_cast<std::size_t>(h) * D + d;
                    const std::size_t src = (static_cast<std::size_t>(j) * HKV + h) * D + d;
                    kbuf[dst] = rq.k_bf[src];
                    vbuf[dst] = rq.v_bf[src];
                }
        }
    }

    void *d_q=nullptr,*d_k=nullptr,*d_v=nullptr,*d_o=nullptr;
    std::uint32_t *d_kpi=nullptr,*d_kpp=nullptr,*d_klpl=nullptr;
    RT(cudaMalloc(&d_q, q_bf.size()*2));
    RT(cudaMalloc(&d_k, kbuf.size()*2));
    RT(cudaMalloc(&d_v, vbuf.size()*2));
    RT(cudaMalloc(&d_o, static_cast<std::size_t>(R)*HQ*D*2));
    RT(cudaMalloc(&d_kpi, kv_page_indices_h.size()*4));
    RT(cudaMalloc(&d_kpp, kv_page_indptr_h.size()*4));
    RT(cudaMalloc(&d_klpl, kv_last_page_lens_h.size()*4));
    RT(cudaMemcpy(d_q, q_bf.data(), q_bf.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_k, kbuf.data(), kbuf.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_v, vbuf.data(), vbuf.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpi, kv_page_indices_h.data(), kv_page_indices_h.size()*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpp, kv_page_indptr_h.data(), kv_page_indptr_h.size()*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_klpl, kv_last_page_lens_h.data(), kv_last_page_lens_h.size()*4, cudaMemcpyHostToDevice));

    auto ws = AttentionWorkspace::allocate(256ull*1024*1024, 32ull*1024*1024);
    auto plan = ops::make_decode_plan();
    ops::plan_attention_flashinfer_decode(
        *plan, kv_page_indptr_h.data(), R, HQ, HKV, D, PAGE,
        ws, /*stream=*/nullptr, /*enable_cuda_graph=*/false,
        /*full_attention_variant=*/full_attn, /*hnd_layout=*/false);
    ops::dispatch_attention_flashinfer_decode_bf16(
        *plan, d_q, d_k, d_v, d_o,
        d_kpi, d_kpp, d_klpl, ws, /*stream=*/nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<std::uint16_t> o_bf(static_cast<std::size_t>(R)*HQ*D);
    RT(cudaMemcpy(o_bf.data(), d_o, o_bf.size()*2, cudaMemcpyDeviceToHost));
    std::vector<float> out(o_bf.size());
    for (std::size_t i = 0; i < o_bf.size(); ++i) out[i] = bf16_to_f32(o_bf[i]);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
    cudaFree(d_kpi); cudaFree(d_kpp); cudaFree(d_klpl);
    return out;
}

float max_abs_diff(const float* a, const float* b, std::size_t n) {
    float m = 0.f;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace

int main() {
    const std::size_t row = static_cast<std::size_t>(HQ) * D;
    const float tol = 3e-3f;  // bf16 accumulation slack

    // Mixed-length regimes — the reported failing regime is co-batched
    // rows at MIXED KV-len (short kv_len=2 bit-exact, kv_len>=4 diverge 5-7%).
    // A uniform-length batch can hide a per-request length-attribution bug (a
    // mis-index still reads an equal length), so sweep mixed + edge lengths.
    const std::vector<std::vector<int>> regimes = {
        {2, 4, 5},           // exact reported failing mix
        {2, 3, 4, 5},        // sweep across the kv_len=4 boundary
        {2, 5},              // minimal short+long pair
        {2, 2, 2, 2},        // uniform short (control)
        {5, 5, 5, 5},        // uniform long (control)
        {2, 4, 5, 3, 8, 16}, // wide mix incl page boundary
        {7, 20, 4, 29},      // the original (previously green)
    };

    bool all_ok = true;
    for (const auto& kv_lens : regimes) {
        const int R = static_cast<int>(kv_lens.size());
        std::mt19937 rng(0xB2C0DEu);  // reset per regime for reproducibility
        std::vector<Req> reqs;
        reqs.reserve(R);
        for (int r = 0; r < R; ++r) reqs.push_back(make_req(rng, kv_lens[r]));

        std::printf("\n===== regime {");
        for (int r = 0; r < R; ++r) std::printf("%d%s", kv_lens[r], r + 1 < R ? "," : "");
        std::printf("}  (R=%d) =====\n", R);

        for (int fa = 0; fa <= 1; ++fa) {
            const bool full_attn = (fa == 1);
            const std::vector<float> batched = run_batch(reqs, 0, R, full_attn);
            for (int r = 0; r < R; ++r) {
                const std::vector<float> alone = run_batch(reqs, r, 1, full_attn);
                const float diff = max_abs_diff(batched.data() + static_cast<std::size_t>(r) * row,
                                                alone.data(), row);
                const bool ok = diff <= tol;
                all_ok = all_ok && ok;
                std::printf("[%s] full_attn=%d r=%d (kv_len=%d)  batched-vs-alone max|Δ|=%.5g\n",
                            ok ? " ok " : "FAIL", full_attn, r, kv_lens[r], diff);
            }
        }
    }

    if (!all_ok) {
        ++g_failures;
        std::printf("\nR>1 BATCHED DECODE DIVERGES from the R=1 per-request reference "
                    "in a mixed-length regime — reproduced in the attention "
                    "read/dispatch (isolated from KV-write + model loop).\n");
    } else {
        std::printf("\nAll regimes (incl. the reported mixed {2,4,5}), both variants: "
                    "batched == R=1 reference. R>1 decode attention is per-request "
                    "correct across mixed KV-lengths.\n");
    }
    return g_failures == 0 ? 0 : 1;
}

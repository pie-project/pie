// PTIR M2b — fork-geometry masked-attention GPU parity test.
//
// The M2b exit gate: randomized fork-geometry attention (mask lowering) matches
// the reference oracle within fp32-accumulation tolerance (overview §6.4). Each
// case builds a fork geometry — prompt-page aliasing, mid-chain frozen page,
// designated-child tail, within-page fork, plus a multi-row causal prefill —
// as (Q, paged K/V, BRLE mask), decodes the BRLE through the REAL src/brle.cpp
// into FlashInfer's packed `qo×kv` bitmap, runs the production masked path
// (`launch_attention_flashinfer_prefill_custom_bf16`, MaskMode::kCustom), and
// diffs the device output against `masked_attention_reference.hpp`. A masked
// position is exactly prefix truncation (W6/W11): NO attention-kernel change.
//
// bf16 fairness: Q/K/V are rounded float→bf16→float once and shared with both
// the device and the oracle, so the residual is only the kernel's bf16
// intermediate + output rounding — bounded well under the tolerance.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "attention_workspace.hpp"
#include "brle.hpp"
#include "ops/attention_flashinfer.hpp"

#include "masked_attention_reference.hpp"

namespace {

using pie_cuda_driver::AttentionWorkspace;
using pie_cuda_driver::brle::DecodedMasks;

int g_failures = 0;

void rt_check(cudaError_t e, const char* expr, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL rt %s:%d: %s -> %s\n", __FILE__, line, expr,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define RT(expr) rt_check((expr), #expr, __LINE__)

// bf16 round-trip helpers (round-to-nearest-even not needed; truncation matches
// the driver's `__float2bfloat16_rn`? we use RNE to be safe on both sides).
std::uint16_t f32_to_bf16(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    const std::uint32_t lsb = (b >> 16) & 1u;
    b += 0x7fffu + lsb;  // round to nearest even
    return static_cast<std::uint16_t>(b >> 16);
}
float bf16_to_f32(std::uint16_t h) {
    std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
float bf16_rt(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// Config: GQA, head_dim=64 (a FlashInfer-instantiated dim), page_size=4.
constexpr int HQ = 4, HKV = 2, D = 64, PAGE = 4;

struct Case {
    const char* name;
    int qo_len;
    int kv_len;
    std::vector<std::uint32_t> brle;         // flattened runs
    std::vector<std::uint32_t> mask_indptr;  // [qo_len+1]
    std::vector<std::uint8_t>  want;         // intended-valid mask [qo_len*kv_len]
};

// Unpack brle::decode's packed bits for request 0 → oracle byte mask.
std::vector<std::uint8_t> unpack(const DecodedMasks& dm, int qo_len, int kv_len) {
    std::vector<std::uint8_t> m(static_cast<std::size_t>(qo_len) * kv_len, 0);
    const std::uint8_t* base = dm.packed.data() + dm.mask_indptr[0];
    for (int q = 0; q < qo_len; ++q)
        for (int j = 0; j < kv_len; ++j) {
            const long bit = static_cast<long>(q) * kv_len + j;
            m[static_cast<std::size_t>(q) * kv_len + j] =
                (base[bit / 8] >> (bit % 8)) & 1u;
        }
    return m;
}

bool run_case(const Case& c, float tol) {
    const int pages = (c.kv_len + PAGE - 1) / PAGE;
    const int last_len = c.kv_len - (pages - 1) * PAGE;
    const std::vector<std::uint32_t> qo_indptr_h{0, static_cast<std::uint32_t>(c.qo_len)};
    const std::vector<std::uint32_t> kv_page_indptr_h{0, static_cast<std::uint32_t>(pages)};
    const std::vector<std::uint32_t> kv_last_page_lens_h{static_cast<std::uint32_t>(last_len)};
    std::vector<std::uint32_t> kv_page_indices_h(pages);
    for (int p = 0; p < pages; ++p) kv_page_indices_h[p] = static_cast<std::uint32_t>(p);

    // 1. Decode the BRLE through the real driver decoder; sanity-check vs intent.
    DecodedMasks dm = pie_cuda_driver::brle::decode(
        c.brle, c.mask_indptr, qo_indptr_h, kv_page_indptr_h, kv_last_page_lens_h, PAGE);
    auto decoded = unpack(dm, c.qo_len, c.kv_len);
    for (std::size_t i = 0; i < decoded.size(); ++i) {
        if ((decoded[i] != 0) != (c.want[i] != 0)) {
            std::printf("    %s: BRLE decode != intent at %zu\n", c.name, i);
            return false;
        }
    }

    // 2. Random bf16-rounded Q/K/V (shared with the oracle).
    std::mt19937_64 rng(0xA11CE + c.kv_len);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> q(static_cast<std::size_t>(c.qo_len) * HQ * D);
    std::vector<float> k(static_cast<std::size_t>(c.kv_len) * HKV * D);
    std::vector<float> v(static_cast<std::size_t>(c.kv_len) * HKV * D);
    for (auto& x : q) x = bf16_rt(u(rng));
    for (auto& x : k) x = bf16_rt(u(rng));
    for (auto& x : v) x = bf16_rt(u(rng));

    // 3. Reference (fp32 over the bf16-rounded inputs + decoded mask).
    pie_attn_ref::Problem prob;
    prob.qo_len = c.qo_len; prob.kv_len = c.kv_len;
    prob.num_qo_heads = HQ; prob.num_kv_heads = HKV; prob.head_dim = D;
    prob.scale = 1.0f / std::sqrt(static_cast<float>(D));
    prob.q = q.data(); prob.k = k.data(); prob.v = v.data(); prob.mask = decoded.data();
    auto ref = pie_attn_ref::attention(prob);

    // 4. Device buffers. Q/O: [tokens, HQ, D] bf16. Paged K/V: NHD
    //    [pages, PAGE, HKV, D] bf16 (page-major, then token, head, dim).
    const int tokens = c.qo_len;
    std::vector<std::uint16_t> q_bf(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) q_bf[i] = f32_to_bf16(q[i]);
    const std::size_t page_stride = static_cast<std::size_t>(PAGE) * HKV * D;
    std::vector<std::uint16_t> kbuf(static_cast<std::size_t>(pages) * page_stride, 0);
    std::vector<std::uint16_t> vbuf(static_cast<std::size_t>(pages) * page_stride, 0);
    for (int j = 0; j < c.kv_len; ++j) {
        const int pg = j / PAGE, slot = j % PAGE;
        for (int h = 0; h < HKV; ++h)
            for (int d = 0; d < D; ++d) {
                const std::size_t dst = static_cast<std::size_t>(pg) * page_stride +
                                        static_cast<std::size_t>(slot) * HKV * D +
                                        static_cast<std::size_t>(h) * D + d;
                const std::size_t src = (static_cast<std::size_t>(j) * HKV + h) * D + d;
                kbuf[dst] = f32_to_bf16(k[src]);
                vbuf[dst] = f32_to_bf16(v[src]);
            }
    }

    void *d_q=nullptr,*d_k=nullptr,*d_v=nullptr,*d_o=nullptr;
    std::uint32_t *d_qo=nullptr,*d_kpi=nullptr,*d_kpp=nullptr,*d_klpl=nullptr;
    std::uint8_t* d_mask=nullptr; std::int32_t* d_mip=nullptr;
    RT(cudaMalloc(&d_q, q_bf.size()*2));
    RT(cudaMalloc(&d_k, kbuf.size()*2));
    RT(cudaMalloc(&d_v, vbuf.size()*2));
    RT(cudaMalloc(&d_o, static_cast<std::size_t>(tokens)*HQ*D*2));
    RT(cudaMalloc(&d_qo, qo_indptr_h.size()*4));
    RT(cudaMalloc(&d_kpi, kv_page_indices_h.size()*4));
    RT(cudaMalloc(&d_kpp, kv_page_indptr_h.size()*4));
    RT(cudaMalloc(&d_klpl, kv_last_page_lens_h.size()*4));
    RT(cudaMalloc(&d_mask, dm.packed.size()));
    RT(cudaMalloc(&d_mip, dm.mask_indptr.size()*4));
    RT(cudaMemcpy(d_q, q_bf.data(), q_bf.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_k, kbuf.data(), kbuf.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_v, vbuf.data(), vbuf.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_qo, qo_indptr_h.data(), qo_indptr_h.size()*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpi, kv_page_indices_h.data(), kv_page_indices_h.size()*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpp, kv_page_indptr_h.data(), kv_page_indptr_h.size()*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_klpl, kv_last_page_lens_h.data(), kv_last_page_lens_h.size()*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_mask, dm.packed.data(), dm.packed.size(), cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_mip, dm.mask_indptr.data(), dm.mask_indptr.size()*4, cudaMemcpyHostToDevice));

    // 5. Workspace + the production masked kernel.
    auto ws = AttentionWorkspace::allocate(128ull*1024*1024, 16ull*1024*1024);
    pie_cuda_driver::ops::launch_attention_flashinfer_prefill_custom_bf16(
        d_q, d_k, d_v, d_o,
        d_qo, d_kpi, d_kpp, d_klpl,
        d_mask, d_mip,
        qo_indptr_h.data(), kv_page_indptr_h.data(),
        tokens, /*num_requests=*/1, HQ, HKV, D, PAGE,
        ws, /*stream=*/nullptr);
    RT(cudaDeviceSynchronize());

    // 6. Download, compare.
    std::vector<std::uint16_t> o_bf(static_cast<std::size_t>(tokens)*HQ*D);
    RT(cudaMemcpy(o_bf.data(), d_o, o_bf.size()*2, cudaMemcpyDeviceToHost));
    std::vector<float> dev(o_bf.size());
    for (std::size_t i = 0; i < o_bf.size(); ++i) dev[i] = bf16_to_f32(o_bf[i]);
    const float diff = pie_attn_ref::max_abs_diff(dev, ref);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o); cudaFree(d_qo);
    cudaFree(d_kpi); cudaFree(d_kpp); cudaFree(d_klpl); cudaFree(d_mask); cudaFree(d_mip);

    const bool ok = diff <= tol;
    std::printf("[%s] %-42s max|Δ|=%.4g (tol %.3g)\n", ok?" ok ":"FAIL", c.name, diff, tol);
    return ok;
}

}  // namespace

// ── Beam-shaped microbench (M2b): masked-variant overhead vs dense decode ─────
// B=8 decode-shaped lanes (qo=1 each, fork every step ⇒ each lane a full-KV
// custom mask). Masked path = flashinfer prefill-custom (kCustom); dense
// baseline = flashinfer causal prefill at qo=1 (which attends the whole KV, so
// it's the same effective attention set — the delta is purely the kCustom
// variant + packed-mask reads). Reports mean us/fire + the overhead ratio; this
// is the "recorded as direction" number (perf escape gate), not a pass/fail.
namespace {

double ms_ev(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.f; RT(cudaEventElapsedTime(&ms, a, b)); return ms;
}

void beam_microbench() {
    constexpr int B = 8, QH = 32, KVH = 8, HD = 128, PG = 16, KVLEN = 512;
    const int pages_per = (KVLEN + PG - 1) / PG;      // 32
    const int total_pages = B * pages_per;
    const int last_len = KVLEN - (pages_per - 1) * PG;

    std::vector<std::uint32_t> qo_h(B + 1), kvpp_h(B + 1), klpl_h(B), kvpi_h(total_pages);
    for (int r = 0; r <= B; ++r) { qo_h[r] = r; kvpp_h[r] = r * pages_per; }
    for (int r = 0; r < B; ++r) klpl_h[r] = last_len;
    for (int p = 0; p < total_pages; ++p) kvpi_h[p] = p;

    // Full-KV BRLE per lane (attend all KVLEN): {0, KVLEN} per row.
    std::vector<std::uint32_t> brle(2 * B), mip(B + 1, 0);
    for (int r = 0; r < B; ++r) { brle[2*r] = 0; brle[2*r+1] = KVLEN; mip[r+1] = mip[r] + 2; }
    DecodedMasks dm = pie_cuda_driver::brle::decode(brle, mip, qo_h, kvpp_h, klpl_h, PG);

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<std::uint16_t> q_bf(static_cast<std::size_t>(B) * QH * HD);
    for (auto& x : q_bf) x = f32_to_bf16(u(rng));
    const std::size_t kv_elems = static_cast<std::size_t>(total_pages) * PG * KVH * HD;
    std::vector<std::uint16_t> kbuf(kv_elems), vbuf(kv_elems);
    for (auto& x : kbuf) x = f32_to_bf16(u(rng));
    for (auto& x : vbuf) x = f32_to_bf16(u(rng));

    void *d_q,*d_k,*d_v,*d_o; std::uint32_t *d_qo,*d_kpi,*d_kpp,*d_klpl;
    std::uint8_t* d_mask; std::int32_t* d_mip;
    RT(cudaMalloc(&d_q, q_bf.size()*2)); RT(cudaMalloc(&d_k, kbuf.size()*2));
    RT(cudaMalloc(&d_v, vbuf.size()*2)); RT(cudaMalloc(&d_o, q_bf.size()*2));
    RT(cudaMalloc(&d_qo,(B+1)*4)); RT(cudaMalloc(&d_kpi,total_pages*4));
    RT(cudaMalloc(&d_kpp,(B+1)*4)); RT(cudaMalloc(&d_klpl,B*4));
    RT(cudaMalloc(&d_mask, dm.packed.size())); RT(cudaMalloc(&d_mip, dm.mask_indptr.size()*4));
    RT(cudaMemcpy(d_q,q_bf.data(),q_bf.size()*2,cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_k,kbuf.data(),kbuf.size()*2,cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_v,vbuf.data(),vbuf.size()*2,cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_qo,qo_h.data(),(B+1)*4,cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpi,kvpi_h.data(),total_pages*4,cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpp,kvpp_h.data(),(B+1)*4,cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_klpl,klpl_h.data(),B*4,cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_mask,dm.packed.data(),dm.packed.size(),cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_mip,dm.mask_indptr.data(),dm.mask_indptr.size()*4,cudaMemcpyHostToDevice));

    auto ws = AttentionWorkspace::allocate(256ull*1024*1024, 32ull*1024*1024);
    cudaEvent_t e0, e1; RT(cudaEventCreate(&e0)); RT(cudaEventCreate(&e1));
    constexpr int WARM = 10, ITER = 50;

    auto masked = [&]{ pie_cuda_driver::ops::launch_attention_flashinfer_prefill_custom_bf16(
        d_q,d_k,d_v,d_o,d_qo,d_kpi,d_kpp,d_klpl,d_mask,d_mip,qo_h.data(),kvpp_h.data(),
        B,B,QH,KVH,HD,PG,ws,nullptr); };
    auto dense = [&]{ pie_cuda_driver::ops::launch_attention_flashinfer_prefill_bf16(
        d_q,d_k,d_v,d_o,d_qo,d_kpi,d_kpp,d_klpl,qo_h.data(),kvpp_h.data(),
        B,B,QH,KVH,HD,PG,ws,nullptr); };

    for (int i=0;i<WARM;++i){ masked(); dense(); } RT(cudaDeviceSynchronize());
    RT(cudaEventRecord(e0)); for(int i=0;i<ITER;++i) masked(); RT(cudaEventRecord(e1));
    RT(cudaEventSynchronize(e1)); const double t_masked = ms_ev(e0,e1)/ITER*1000.0;
    RT(cudaEventRecord(e0)); for(int i=0;i<ITER;++i) dense(); RT(cudaEventRecord(e1));
    RT(cudaEventSynchronize(e1)); const double t_dense = ms_ev(e0,e1)/ITER*1000.0;

    std::printf("\n[beam microbench] B=%d qo=1/lane kv_len=%d h_q=%d h_kv=%d d=%d\n",
                B, KVLEN, QH, KVH, HD);
    std::printf("  masked (prefill-custom kCustom): %.2f us/fire\n", t_masked);
    std::printf("  dense  (causal prefill,   qo=1): %.2f us/fire\n", t_dense);
    std::printf("  masked-variant overhead: %.2fx (+%.2f us)\n",
                t_masked / t_dense, t_masked - t_dense);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_q);cudaFree(d_k);cudaFree(d_v);cudaFree(d_o);cudaFree(d_qo);
    cudaFree(d_kpi);cudaFree(d_kpp);cudaFree(d_klpl);cudaFree(d_mask);cudaFree(d_mip);
}

}  // namespace

int main() {
    RT(cudaSetDevice(0));

    std::vector<Case> cases;
    // A. prompt-page aliasing — decode row attends a full 8-key shared prefix.
    cases.push_back({"A prompt-page aliasing", 1, 8, {0,8}, {0,2}, std::vector<std::uint8_t>(8,1)});
    // B. mid-chain frozen page — residual [6,8) excluded; valid {0..5, 8..11}.
    {   std::vector<std::uint8_t> w(12,0); for(int j=0;j<6;++j)w[j]=1; for(int j=8;j<12;++j)w[j]=1;
        cases.push_back({"B mid-chain frozen page", 1, 12, {0,6,2,4}, {0,4}, w}); }
    // C. designated-child tail — sibling excludes [6,8); valid {0..5}.
    {   std::vector<std::uint8_t> w(8,0); for(int j=0;j<6;++j)w[j]=1;
        cases.push_back({"C designated-child tail", 1, 8, {0,6}, {0,2}, w}); }
    // D. within-page fork at offset 5 — sibling slot 5 masked; valid {0..4}.
    {   std::vector<std::uint8_t> w(6,0); for(int j=0;j<5;++j)w[j]=1;
        cases.push_back({"D within-page fork", 1, 6, {0,5}, {0,2}, w}); }
    // E. multi-row causal prefill — row q attends [0,q].
    {   std::vector<std::uint8_t> w(16,0); for(int q=0;q<4;++q) for(int j=0;j<=q;++j) w[q*4+j]=1;
        cases.push_back({"E multi-row causal prefill", 4, 4, {0,1,0,2,0,3,0,4}, {0,2,4,6,8}, w}); }

    const float tol = 1e-2f;  // bf16 output rounding slack (observed max ~2.4e-3)
    for (const auto& c : cases) if (!run_case(c, tol)) ++g_failures;

    std::printf(g_failures ? "\nM2b PARITY FAILED (%d)\n" : "\nALL PASS (0 failures)\n", g_failures);

    // Overhead number (recorded-as-direction; does not gate pass/fail).
    if (g_failures == 0) beam_microbench();
    return g_failures ? 1 : 0;
}

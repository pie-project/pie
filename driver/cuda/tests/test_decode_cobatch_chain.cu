// Bug#2 DEFINITIVE splitter: chained {prefill + fused-decode-write + attention}
// co-batched R>1 vs alone, checking BOTH the written K/V CONTENT and the attn_out.
//
// The metadata dump proved the kernel receives correct indptr/last_page_len/
// pages under R>1, and my two isolated guards proved the attention read (on
// hand-built KV) and the fused write (to fresh pages) are each per-request
// correct. This test chains them through a NON-FRESH prefill page — the exact
// real decode invocation — and splits the remaining two candidates:
//   K/V CONTENT differs co-batched-vs-alone  → the fused decode-write stores
//       wrong bytes under co-batch (race/ordering) → fix the WRITE.
//   CONTENT identical but attn_out diverges  → genuine kernel bug the fresh-page
//       guard missed → bisect the kernel.
//
// Regime = the failing shape: R=2 co-batched, kv_len {3,4} (both attend a
// decode-written token; len-2 pure-prefill is the bit-exact control).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/split_packed.hpp"
#include "ops/attention_flashinfer.hpp"

using pie_cuda_driver::AttentionWorkspace;
namespace ops = pie_cuda_driver::ops;
namespace k = pie_cuda_driver::kernels;

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

constexpr int HQ = 16, HKV = 8, D = 128, PAGE = 16;
constexpr int Q_DIM = HQ * D, KV_DIM = HKV * D, PACKED = Q_DIM + 2 * KV_DIM;
const std::size_t PAGE_STRIDE = static_cast<std::size_t>(PAGE) * HKV * D;  // NHD

struct Req {
    int prefill_len;                        // hand-built prefill tokens (slots 0..prefill_len)
    std::vector<std::uint16_t> prefill_k;   // [prefill_len, HKV, D]
    std::vector<std::uint16_t> prefill_v;
    std::vector<std::vector<std::uint16_t>> packed;  // per decode step: [PACKED]
    // kv_len at step s = prefill_len + s + 1
};

Req make_req(std::mt19937& rng, int prefill_len, int num_steps) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    Req r;
    r.prefill_len = prefill_len;
    r.prefill_k.resize(static_cast<std::size_t>(prefill_len) * HKV * D);
    r.prefill_v.resize(static_cast<std::size_t>(prefill_len) * HKV * D);
    for (auto& x : r.prefill_k) x = f32_to_bf16(u(rng));
    for (auto& x : r.prefill_v) x = f32_to_bf16(u(rng));
    r.packed.resize(num_steps);
    for (int s = 0; s < num_steps; ++s) {
        r.packed[s].resize(PACKED);
        for (auto& x : r.packed[s]) x = f32_to_bf16(u(rng));
    }
    return r;
}

struct ChainResult {
    std::vector<float> attn_out;  // [R, HQ, D]  — kernel output
    std::vector<float> decode_k;  // [R, HKV, D] — K bytes written at each decode slot
    std::vector<float> decode_v;
};

// Chain: pre-fill prefill KV → fused decode write (adds one token at slot
// prefill_len) → BatchDecode attention over [0, prefill_len+1). Returns attn_out
// + the K/V bytes the kernel reads at each request's decode slot.
ChainResult run_chain(const std::vector<Req>& reqs, int first, int R, int num_steps,
                      const std::vector<std::uint16_t>& qw,
                      const std::vector<std::uint16_t>& kw) {
    std::vector<std::uint32_t> kv_page_indptr_h(R + 1, 0);
    std::vector<std::uint32_t> kv_last_page_lens_h(R);
    std::vector<std::int32_t> positions_h(R);
    std::vector<int> prefill(R), base_page(R);
    for (int i = 0; i < R; ++i) {
        prefill[i] = reqs[first + i].prefill_len;
        const int kv_len_final = prefill[i] + num_steps;
        const int pages = (kv_len_final + PAGE - 1) / PAGE;  // single page for our shapes
        base_page[i] = static_cast<int>(kv_page_indptr_h[i]);
        kv_page_indptr_h[i + 1] = kv_page_indptr_h[i] + pages;
    }
    const int total_pages = static_cast<int>(kv_page_indptr_h[R]);
    std::vector<std::uint32_t> kv_page_indices_h(total_pages);
    for (int p = 0; p < total_pages; ++p) kv_page_indices_h[p] = static_cast<std::uint32_t>(p);

    // Host KV pool: pre-fill each request's prefill K/V into slots [0, prefill_len).
    std::vector<std::uint16_t> kbuf(static_cast<std::size_t>(total_pages) * PAGE_STRIDE, 0);
    std::vector<std::uint16_t> vbuf(static_cast<std::size_t>(total_pages) * PAGE_STRIDE, 0);
    for (int i = 0; i < R; ++i) {
        const Req& rq = reqs[first + i];
        for (int j = 0; j < rq.prefill_len; ++j)
            for (int h = 0; h < HKV; ++h)
                for (int d = 0; d < D; ++d) {
                    const std::size_t dst = static_cast<std::size_t>(base_page[i]) * PAGE_STRIDE +
                                            static_cast<std::size_t>(j) * HKV * D +
                                            static_cast<std::size_t>(h) * D + d;
                    const std::size_t src = (static_cast<std::size_t>(j) * HKV + h) * D + d;
                    kbuf[dst] = rq.prefill_k[src];
                    vbuf[dst] = rq.prefill_v[src];
                }
    }
    std::vector<std::uint16_t> packed(static_cast<std::size_t>(R) * PACKED);

    void *d_k=nullptr,*d_v=nullptr,*d_packed=nullptr,*d_qout=nullptr,*d_o=nullptr,*d_qw=nullptr,*d_kw=nullptr;
    std::int32_t* d_pos=nullptr;
    std::uint32_t *d_kpi=nullptr,*d_kpp=nullptr,*d_klpl=nullptr;
    RT(cudaMalloc(&d_k, kbuf.size()*2));
    RT(cudaMalloc(&d_v, vbuf.size()*2));
    RT(cudaMalloc(&d_packed, packed.size()*2));
    RT(cudaMalloc(&d_qout, static_cast<std::size_t>(R)*Q_DIM*2));
    RT(cudaMalloc(&d_o, static_cast<std::size_t>(R)*Q_DIM*2));
    RT(cudaMalloc(&d_qw, qw.size()*2));
    RT(cudaMalloc(&d_kw, kw.size()*2));
    RT(cudaMalloc(&d_pos, R*4));
    RT(cudaMalloc(&d_kpi, total_pages*4));
    RT(cudaMalloc(&d_kpp, (R+1)*4));
    RT(cudaMalloc(&d_klpl, R*4));
    RT(cudaMemcpy(d_k, kbuf.data(), kbuf.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_v, vbuf.data(), vbuf.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_qw, qw.data(), qw.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kw, kw.data(), kw.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpi, kv_page_indices_h.data(), total_pages*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpp, kv_page_indptr_h.data(), (R+1)*4, cudaMemcpyHostToDevice));

    // Decode STEPS: each step s is a separate fused write appending request i's
    // token at slot prefill[i]+s onto the SAME (accumulating, non-fresh) page —
    // exactly the step0/step1/... regime (step1+ = attends a prior step's
    // decode-written token). Per-step geometry (position + last_page_len).
    for (int s = 0; s < num_steps; ++s) {
        for (int i = 0; i < R; ++i) {
            positions_h[i] = prefill[i] + s;
            // Mechanism repro: PIE_INJECT_WRONG_POS simulates the upstream bug —
            // a non-first (r>=1) co-batched row receiving the wrong position
            // (here, row 0's). If this yields the wrong-decode-K/attn_out
            // for r>=1 (co-batched) vs correct alone, the mechanism is confirmed.
            static const bool inject = std::getenv("PIE_INJECT_WRONG_POS") != nullptr;
            if (inject && R > 1 && i >= 1) positions_h[i] = prefill[0] + s;
            kv_last_page_lens_h[i] = static_cast<std::uint32_t>(prefill[i] + s + 1);  // single page
            std::memcpy(packed.data() + static_cast<std::size_t>(i) * PACKED,
                        reqs[first + i].packed[s].data(), PACKED * 2);
        }
        RT(cudaMemcpy(d_pos, positions_h.data(), R*4, cudaMemcpyHostToDevice));
        RT(cudaMemcpy(d_klpl, kv_last_page_lens_h.data(), R*4, cudaMemcpyHostToDevice));
        RT(cudaMemcpy(d_packed, packed.data(), packed.size()*2, cudaMemcpyHostToDevice));
        k::launch_qkv_decode_qk_norm_rope_write_kv_bf16(
            d_packed, d_qout, d_k, d_v, d_qw, d_kw, d_pos, /*rope_table=*/nullptr,
            d_kpi, d_kpp, d_klpl, R, HQ, HKV, D, PAGE, /*hnd_layout=*/false,
            /*theta=*/1.0e6f, /*eps=*/1.0e-6f, /*stream=*/nullptr);
        RT(cudaDeviceSynchronize());
    }

    // Final BatchDecode attention over prefill + all num_steps decode tokens.
    for (int i = 0; i < R; ++i)
        kv_last_page_lens_h[i] = static_cast<std::uint32_t>(prefill[i] + num_steps);
    RT(cudaMemcpy(d_klpl, kv_last_page_lens_h.data(), R*4, cudaMemcpyHostToDevice));
    auto ws = AttentionWorkspace::allocate(256ull*1024*1024, 32ull*1024*1024);
    auto plan = ops::make_decode_plan();
    ops::plan_attention_flashinfer_decode(*plan, kv_page_indptr_h.data(), R, HQ, HKV, D, PAGE,
        ws, nullptr, false, /*full_attention_variant=*/true, false);
    ops::dispatch_attention_flashinfer_decode_bf16(
        *plan, d_qout, d_k, d_v, d_o, d_kpi, d_kpp, d_klpl, ws, nullptr);
    RT(cudaDeviceSynchronize());

    // Download the KV pool + o; extract each request's LAST decode-slot K/V + attn_out.
    std::vector<std::uint16_t> kpg(kbuf.size()), vpg(vbuf.size()), o(static_cast<std::size_t>(R)*Q_DIM);
    RT(cudaMemcpy(kpg.data(), d_k, kpg.size()*2, cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(vpg.data(), d_v, vpg.size()*2, cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(o.data(), d_o, o.size()*2, cudaMemcpyDeviceToHost));

    ChainResult res;
    res.attn_out.resize(static_cast<std::size_t>(R)*Q_DIM);
    res.decode_k.resize(static_cast<std::size_t>(R)*KV_DIM);
    res.decode_v.resize(static_cast<std::size_t>(R)*KV_DIM);
    for (std::size_t i = 0; i < o.size(); ++i) res.attn_out[i] = bf16_to_f32(o[i]);
    for (int i = 0; i < R; ++i) {
        const int last_slot = prefill[i] + num_steps - 1;
        const std::size_t slot_base = static_cast<std::size_t>(base_page[i]) * PAGE_STRIDE +
                                       static_cast<std::size_t>(last_slot) * HKV * D;
        for (int j = 0; j < KV_DIM; ++j) {
            res.decode_k[static_cast<std::size_t>(i)*KV_DIM + j] = bf16_to_f32(kpg[slot_base + j]);
            res.decode_v[static_cast<std::size_t>(i)*KV_DIM + j] = bf16_to_f32(vpg[slot_base + j]);
        }
    }

    cudaFree(d_k); cudaFree(d_v); cudaFree(d_packed); cudaFree(d_qout); cudaFree(d_o);
    cudaFree(d_qw); cudaFree(d_kw); cudaFree(d_pos);
    cudaFree(d_kpi); cudaFree(d_kpp); cudaFree(d_klpl);
    return res;
}

float maxdiff(const float* a, const float* b, std::size_t n) {
    float m = 0.f;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace

int main() {
    std::mt19937 rng(0xC0BA7Cu);
    constexpr int R = 2;
    // the failing shape: R=2 co-batched, prefill_len {2,3}. Sweep decode
    // steps: step0 attends only prefill (the bit-exact control), step1/step2
    // attend PRIOR decode-written tokens (the diverging regime).
    const int prefill_lens[R] = {2, 3};

    std::uniform_real_distribution<float> u(0.5f, 1.5f);
    std::vector<std::uint16_t> qw(D), kw(D);
    for (auto& x : qw) x = f32_to_bf16(u(rng));
    for (auto& x : kw) x = f32_to_bf16(u(rng));

    const std::size_t qrow = Q_DIM, kvrow = KV_DIM;
    const float tol = 3e-3f;
    bool all_ok = true;

    for (int num_steps = 1; num_steps <= 3; ++num_steps) {
        std::mt19937 r2(0xC0BA7Cu);  // same reqs across step counts
        std::vector<Req> reqs;
        reqs.reserve(R);
        for (int r = 0; r < R; ++r) reqs.push_back(make_req(r2, prefill_lens[r], num_steps));

        std::printf("\n===== num_decode_steps=%d (step%d attends %d prior decode tokens) =====\n",
                    num_steps, num_steps - 1, num_steps - 1);
        const ChainResult batched = run_chain(reqs, 0, R, num_steps, qw, kw);
        for (int r = 0; r < R; ++r) {
            const ChainResult alone = run_chain(reqs, r, 1, num_steps, qw, kw);
            const float dk = maxdiff(batched.decode_k.data() + static_cast<std::size_t>(r)*kvrow,
                                     alone.decode_k.data(), kvrow);
            const float dv = maxdiff(batched.decode_v.data() + static_cast<std::size_t>(r)*kvrow,
                                     alone.decode_v.data(), kvrow);
            const float da = maxdiff(batched.attn_out.data() + static_cast<std::size_t>(r)*qrow,
                                     alone.attn_out.data(), qrow);
            const bool content_ok = dk <= tol && dv <= tol;
            const bool attn_ok = da <= tol;
            all_ok = all_ok && content_ok && attn_ok;
            std::printf("[%s] r=%d (kv_len=%d)  WRITE last-K Δ=%.5g last-V Δ=%.5g | KERNEL attn_out Δ=%.5g\n",
                        (content_ok && attn_ok) ? " ok " : "FAIL", r,
                        reqs[r].prefill_len + num_steps, dk, dv, da);
        }
    }

    if (!all_ok) {
        std::printf("\nSPLIT: WRITE Δ>0 → fused decode-write corrupts content under co-batch (fix "
                    "write); WRITE Δ=0 but KERNEL Δ>0 → attention kernel bug on the multi-step "
                    "real invocation.\n");
        ++g_failures;
    } else {
        std::printf("\nAll step counts: chain co-batched == alone for BOTH written K/V content AND "
                    "attn_out. The multi-step real decode invocation is per-request correct in "
                    "isolation — bug is in concurrent-fires / multi-layer real-forward context.\n");
    }
    return g_failures == 0 ? 0 : 1;
}

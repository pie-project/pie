// Bug#2: the FUSED decode kernel (qk-norm + rope + KV-write + q_out) must be
// per-request correct for R>1.
//
// The standalone BatchDecode attention parity test proved the R>1 attention
// read is per-request correct given correct KV/Q. So the concurrent-decode
// corruption is in the KV/Q *production* — and the Qwen3 decode produces both
// via `launch_qkv_decode_qk_norm_rope_write_kv_bf16` (fused_decode_qkv_post),
// NOT the `resolve_dst` path already verified. This test isolates that fused
// kernel: run it for R=4 requests (distinct packed QKV + distinct rope
// positions + disjoint pages), then run each request ALONE (R=1), and assert
// the produced q_out row AND the written K/V page slot are bit-identical.
//
// A divergence pins the concurrent-decode bug to the fused kernel's R>1 path
// (unit/`total_qk_heads` decomposition, per-request position/page indexing).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/split_packed.hpp"

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
    int position;                    // decode token's sequence position (rope + write slot)
    std::vector<std::uint16_t> packed;  // [PACKED] bf16 (projected q|k|v for this token)
};

Req make_req(std::mt19937& rng, int position) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    Req r;
    r.position = position;
    r.packed.resize(PACKED);
    for (auto& x : r.packed) x = f32_to_bf16(u(rng));
    return r;
}

struct Result {
    std::vector<float> q_out;   // [R, HQ, D]
    std::vector<float> k_slot;  // [R, HKV, D] — the K written at each request's decode slot
    std::vector<float> v_slot;  // [R, HKV, D]
};

// Run the fused decode kernel over `reqs[first..first+R)`. Each request gets its
// own 1 KV page; its decode token is written at slot = position (last_page_len =
// position+1). Returns the produced q_out + the written K/V at each slot.
Result run_fused(const std::vector<Req>& reqs, int first, int R,
                 const std::vector<std::uint16_t>& qw,
                 const std::vector<std::uint16_t>& kw,
                 bool explicit_write = false) {
    std::vector<std::uint32_t> kv_page_indptr_h(R + 1);
    std::vector<std::uint32_t> kv_last_page_lens_h(R);
    std::vector<std::uint32_t> kv_page_indices_h(R);
    std::vector<std::int32_t> positions_h(R);
    for (int i = 0; i < R; ++i) {
        kv_page_indptr_h[i] = static_cast<std::uint32_t>(i);
        kv_page_indices_h[i] = static_cast<std::uint32_t>(i);   // disjoint: request i -> page i
        kv_last_page_lens_h[i] = static_cast<std::uint32_t>(reqs[first + i].position + 1);
        positions_h[i] = reqs[first + i].position;
    }
    kv_page_indptr_h[R] = static_cast<std::uint32_t>(R);

    std::vector<std::uint16_t> packed(static_cast<std::size_t>(R) * PACKED);
    for (int i = 0; i < R; ++i)
        std::memcpy(packed.data() + static_cast<std::size_t>(i) * PACKED,
                    reqs[first + i].packed.data(), PACKED * 2);

    void *d_packed=nullptr,*d_qout=nullptr,*d_k=nullptr,*d_v=nullptr,*d_qw=nullptr,*d_kw=nullptr;
    std::int32_t* d_pos=nullptr;
    std::uint32_t *d_kpi=nullptr,*d_kpp=nullptr,*d_klpl=nullptr;
    std::uint32_t *d_w_page=nullptr,*d_w_off=nullptr;
    std::uint8_t* d_valid=nullptr;
    const std::size_t kv_bytes = static_cast<std::size_t>(R) * PAGE_STRIDE * 2;
    RT(cudaMalloc(&d_packed, packed.size()*2));
    RT(cudaMalloc(&d_qout, static_cast<std::size_t>(R)*Q_DIM*2));
    RT(cudaMalloc(&d_k, kv_bytes));
    RT(cudaMalloc(&d_v, kv_bytes));
    RT(cudaMalloc(&d_qw, qw.size()*2));
    RT(cudaMalloc(&d_kw, kw.size()*2));
    RT(cudaMalloc(&d_pos, R*4));
    RT(cudaMalloc(&d_kpi, R*4));
    RT(cudaMalloc(&d_kpp, (R+1)*4));
    RT(cudaMalloc(&d_klpl, R*4));
    RT(cudaMemset(d_k, 0, kv_bytes));
    RT(cudaMemset(d_v, 0, kv_bytes));
    RT(cudaMemcpy(d_packed, packed.data(), packed.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_qw, qw.data(), qw.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kw, kw.data(), kw.size()*2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_pos, positions_h.data(), R*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpi, kv_page_indices_h.data(), R*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpp, kv_page_indptr_h.data(), (R+1)*4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_klpl, kv_last_page_lens_h.data(), R*4, cudaMemcpyHostToDevice));
    if (explicit_write) {
        std::vector<std::uint32_t> w_page(R), w_off(R);
        std::vector<std::uint8_t> valid(R, 1);
        for (int i = 0; i < R; ++i) {
            w_page[i] = static_cast<std::uint32_t>(i);
            w_off[i] = static_cast<std::uint32_t>(
                reqs[first + i].position);
        }
        RT(cudaMalloc(&d_w_page, R * sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_w_off, R * sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_valid, R));
        RT(cudaMemcpy(
            d_w_page, w_page.data(), R * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice));
        RT(cudaMemcpy(
            d_w_off, w_off.data(), R * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice));
        RT(cudaMemcpy(
            d_valid, valid.data(), R, cudaMemcpyHostToDevice));
    }

    k::launch_qkv_decode_qk_norm_rope_write_kv_bf16(
        d_packed, d_qout, d_k, d_v, d_qw, d_kw, d_pos, /*rope_table=*/nullptr,
        d_kpi, d_kpp, d_klpl,
        d_w_page, d_w_off, d_valid,
        R, HQ, HKV, D, PAGE, /*hnd_layout=*/false,
        /*theta=*/1.0e6f, /*eps=*/1.0e-6f, /*stream=*/nullptr);
    RT(cudaDeviceSynchronize());

    // Download q_out + the K/V page contents; extract each request's decode slot.
    std::vector<std::uint16_t> qo(static_cast<std::size_t>(R)*Q_DIM);
    std::vector<std::uint16_t> kpg(static_cast<std::size_t>(R)*PAGE_STRIDE);
    std::vector<std::uint16_t> vpg(static_cast<std::size_t>(R)*PAGE_STRIDE);
    RT(cudaMemcpy(qo.data(), d_qout, qo.size()*2, cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(kpg.data(), d_k, kpg.size()*2, cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(vpg.data(), d_v, vpg.size()*2, cudaMemcpyDeviceToHost));

    Result res;
    res.q_out.resize(static_cast<std::size_t>(R)*Q_DIM);
    res.k_slot.resize(static_cast<std::size_t>(R)*KV_DIM);
    res.v_slot.resize(static_cast<std::size_t>(R)*KV_DIM);
    for (std::size_t i = 0; i < qo.size(); ++i) res.q_out[i] = bf16_to_f32(qo[i]);
    for (int i = 0; i < R; ++i) {
        const int slot = reqs[first + i].position;  // page i (each request its own page)
        const std::size_t page_base = static_cast<std::size_t>(i) * PAGE_STRIDE +
                                       static_cast<std::size_t>(slot) * HKV * D;
        for (int j = 0; j < KV_DIM; ++j) {
            res.k_slot[static_cast<std::size_t>(i)*KV_DIM + j] = bf16_to_f32(kpg[page_base + j]);
            res.v_slot[static_cast<std::size_t>(i)*KV_DIM + j] = bf16_to_f32(vpg[page_base + j]);
        }
    }

    cudaFree(d_packed); cudaFree(d_qout); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_qw); cudaFree(d_kw); cudaFree(d_pos);
    cudaFree(d_kpi); cudaFree(d_kpp); cudaFree(d_klpl);
    cudaFree(d_w_page); cudaFree(d_w_off); cudaFree(d_valid);
    return res;
}

float maxdiff(const float* a, const float* b, std::size_t n) {
    float m = 0.f;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace

int main() {
    std::mt19937 rng(0xFA5EDu);
    constexpr int R = 4;
    const int positions[R] = {3, 10, 1, 7};  // distinct rope positions + write slots

    std::uniform_real_distribution<float> u(0.5f, 1.5f);
    std::vector<std::uint16_t> qw(D), kw(D);
    for (auto& x : qw) x = f32_to_bf16(u(rng));
    for (auto& x : kw) x = f32_to_bf16(u(rng));

    std::vector<Req> reqs;
    reqs.reserve(R);
    for (int r = 0; r < R; ++r) reqs.push_back(make_req(rng, positions[r]));

    const Result batched = run_fused(reqs, 0, R, qw, kw);
    const Result explicit_batched =
        run_fused(reqs, 0, R, qw, kw, true);

    const std::size_t qrow = Q_DIM, kvrow = KV_DIM;
    const float tol = 3e-3f;
    bool all_ok = true;
    const bool explicit_ok =
        maxdiff(
            batched.q_out.data(), explicit_batched.q_out.data(),
            batched.q_out.size()) <= tol &&
        maxdiff(
            batched.k_slot.data(), explicit_batched.k_slot.data(),
            batched.k_slot.size()) <= tol &&
        maxdiff(
            batched.v_slot.data(), explicit_batched.v_slot.data(),
            batched.v_slot.size()) <= tol;
    all_ok = all_ok && explicit_ok;
    std::printf(
        "[%s] explicit WSlot/WOff matches page-derived fused write\n",
        explicit_ok ? " ok " : "FAIL");
    for (int r = 0; r < R; ++r) {
        const Result alone = run_fused(reqs, r, 1, qw, kw);
        const float dq = maxdiff(batched.q_out.data() + static_cast<std::size_t>(r)*qrow,
                                 alone.q_out.data(), qrow);
        const float dk = maxdiff(batched.k_slot.data() + static_cast<std::size_t>(r)*kvrow,
                                 alone.k_slot.data(), kvrow);
        const float dv = maxdiff(batched.v_slot.data() + static_cast<std::size_t>(r)*kvrow,
                                 alone.v_slot.data(), kvrow);
        const bool ok = dq <= tol && dk <= tol && dv <= tol;
        all_ok = all_ok && ok;
        std::printf("[%s] r=%d (pos=%d)  q_out Δ=%.5g  K Δ=%.5g  V Δ=%.5g\n",
                    ok ? " ok " : "FAIL", r, positions[r], dq, dk, dv);
    }

    if (!all_ok) {
        ++g_failures;
        std::printf("\nFUSED decode kernel DIVERGES R=4 vs R=1 — the concurrent-decode "
                    "bug is in launch_qkv_decode_qk_norm_rope_write_kv_bf16.\n");
    } else {
        std::printf("\nFused decode kernel: R=4 == R=1 for q_out + written K/V. "
                    "Per-request correct.\n");
    }
    return g_failures == 0 ? 0 : 1;
}

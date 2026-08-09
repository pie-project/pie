//===-- test_attn_score_capture.cu ------------------------------*- CUDA -*-===//
//
// Parity test for score-observing decode (design doc §3).
//
// The claim under test is narrow and total: running decode through
// `PieScoreCapture` must (a) leave the attention output bit-identical to the
// uncaptured path, and (b) deposit the exact softmax attention weights
// `p[head, kv_idx]`.
//
// (a) matters as much as (b). The capture variant composes with the stock
// `DefaultAttention` rather than replacing it, so if the wrapper ever perturbs
// the value flowing back into the softmax, every model running with
// observation enabled would silently drift. Bit-identity is the only check
// that catches that, so this test demands exact equality, not a tolerance.
//
// (b) is checked against a naive `softmax(q·k * sm_scale)` computed on the
// host from the same bf16 inputs the kernel read.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "attention_workspace.hpp"
#include "attn/attention_flashinfer.hpp"

using pie_cuda_driver::AttentionWorkspace;
namespace kernels = pie_cuda_driver::kernels;

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

// Qwen3-0.6B decode geometry: 16 q heads / 8 kv heads (GQA group 2),
// head_dim 128, page_size 16.
constexpr int HQ = 16, HKV = 8, D = 128, PAGE = 16;
const std::size_t PAGE_STRIDE = static_cast<std::size_t>(PAGE) * HKV * D;

struct Req {
    int kv_len;
    std::vector<std::uint16_t> q_bf;  // [HQ, D]
    std::vector<std::uint16_t> k_bf;  // [kv_len, HKV, D]
    std::vector<std::uint16_t> v_bf;  // [kv_len, HKV, D]
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

struct Run {
    std::vector<float> out;     // [R, HQ, D]
    std::vector<float> scores;  // ragged, [sum_r HQ*kv_len_r]
};

// When non-zero, `run_batch` re-launches its dispatch this many times behind
// CUDA events and reports the mean. Isolating the two entry points against an
// identical plan is the only way to attribute tap cost to the kernel rather
// than to the PTIR stage machinery wrapped around it.
int g_bench_iters = 0;
double g_bench_ms = 0.0;

// `capture` selects the entry point; both take the identical plan.
Run run_batch(const std::vector<Req>& reqs, bool full_attn, bool capture) {
    const int R = static_cast<int>(reqs.size());
    std::vector<std::uint32_t> kv_page_indptr_h(R + 1, 0);
    std::vector<std::uint32_t> kv_last_page_lens_h(R);
    std::vector<std::int32_t> score_indptr_h(R + 1, 0);
    for (int i = 0; i < R; ++i) {
        const int kv = reqs[i].kv_len;
        const int pages = (kv + PAGE - 1) / PAGE;
        kv_page_indptr_h[i + 1] = kv_page_indptr_h[i] + pages;
        kv_last_page_lens_h[i] =
            static_cast<std::uint32_t>(kv - (pages - 1) * PAGE);
        score_indptr_h[i + 1] = score_indptr_h[i] + HQ * kv;
    }
    const int total_pages = static_cast<int>(kv_page_indptr_h[R]);
    std::vector<std::uint32_t> kv_page_indices_h(total_pages);
    for (int p = 0; p < total_pages; ++p) {
        kv_page_indices_h[p] = static_cast<std::uint32_t>(p);
    }

    std::vector<std::uint16_t> q_bf(static_cast<std::size_t>(R) * HQ * D);
    std::vector<std::uint16_t> kbuf(
        static_cast<std::size_t>(total_pages) * PAGE_STRIDE, 0);
    std::vector<std::uint16_t> vbuf(
        static_cast<std::size_t>(total_pages) * PAGE_STRIDE, 0);
    for (int i = 0; i < R; ++i) {
        const Req& rq = reqs[i];
        std::memcpy(q_bf.data() + static_cast<std::size_t>(i) * HQ * D,
                    rq.q_bf.data(), rq.q_bf.size() * 2);
        const int base_page = static_cast<int>(kv_page_indptr_h[i]);
        for (int j = 0; j < rq.kv_len; ++j) {
            const int pg = base_page + j / PAGE, slot = j % PAGE;
            for (int h = 0; h < HKV; ++h) {
                for (int d = 0; d < D; ++d) {
                    const std::size_t dst =
                        static_cast<std::size_t>(pg) * PAGE_STRIDE +
                        static_cast<std::size_t>(slot) * HKV * D +
                        static_cast<std::size_t>(h) * D + d;
                    const std::size_t src =
                        (static_cast<std::size_t>(j) * HKV + h) * D + d;
                    kbuf[dst] = rq.k_bf[src];
                    vbuf[dst] = rq.v_bf[src];
                }
            }
        }
    }

    void *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    std::uint32_t *d_kpi = nullptr, *d_kpp = nullptr, *d_klpl = nullptr;
    float* d_scores = nullptr;
    std::int32_t* d_sindptr = nullptr;
    const std::size_t score_elems =
        static_cast<std::size_t>(score_indptr_h[R]);
    RT(cudaMalloc(&d_q, q_bf.size() * 2));
    RT(cudaMalloc(&d_k, kbuf.size() * 2));
    RT(cudaMalloc(&d_v, vbuf.size() * 2));
    RT(cudaMalloc(&d_o, static_cast<std::size_t>(R) * HQ * D * 2));
    RT(cudaMalloc(&d_kpi, kv_page_indices_h.size() * 4));
    RT(cudaMalloc(&d_kpp, kv_page_indptr_h.size() * 4));
    RT(cudaMalloc(&d_klpl, kv_last_page_lens_h.size() * 4));
    RT(cudaMalloc(&d_scores, score_elems * sizeof(float)));
    RT(cudaMalloc(&d_sindptr, score_indptr_h.size() * 4));
    RT(cudaMemcpy(d_q, q_bf.data(), q_bf.size() * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_k, kbuf.data(), kbuf.size() * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_v, vbuf.data(), vbuf.size() * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpi, kv_page_indices_h.data(),
                  kv_page_indices_h.size() * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpp, kv_page_indptr_h.data(), kv_page_indptr_h.size() * 4,
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_klpl, kv_last_page_lens_h.data(),
                  kv_last_page_lens_h.size() * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_sindptr, score_indptr_h.data(), score_indptr_h.size() * 4,
                  cudaMemcpyHostToDevice));
    // Poison the sink so an unwritten slot cannot masquerade as a zero weight.
    RT(cudaMemset(d_scores, 0x7f, score_elems * sizeof(float)));

    auto ws = AttentionWorkspace::allocate(256ull * 1024 * 1024,
                                           32ull * 1024 * 1024);
    auto plan = kernels::attn::make_decode_plan();
    kernels::attn::plan_attention_flashinfer_decode(
        *plan, kv_page_indptr_h.data(), R, HQ, HKV, D, PAGE, ws.view(),
        /*stream=*/nullptr, /*enable_cuda_graph=*/false,
        /*full_attention_variant=*/full_attn, /*hnd_layout=*/false);
    if (capture) {
        kernels::attn::dispatch_attention_flashinfer_decode_capture_bf16(
            *plan, d_q, d_k, d_v, d_o, d_kpi, d_kpp, d_klpl, ws.view(),
            /*stream=*/nullptr, d_scores, d_sindptr);
    } else {
        kernels::attn::dispatch_attention_flashinfer_decode_bf16(
            *plan, d_q, d_k, d_v, d_o, d_kpi, d_kpp, d_klpl, ws.view(),
            /*stream=*/nullptr);
    }
    RT(cudaDeviceSynchronize());

    if (g_bench_iters > 0) {
        cudaEvent_t beg = nullptr, end = nullptr;
        RT(cudaEventCreate(&beg));
        RT(cudaEventCreate(&end));
        for (int warm = 0; warm < 3; ++warm) {
            if (capture) {
                kernels::attn::dispatch_attention_flashinfer_decode_capture_bf16(
                    *plan, d_q, d_k, d_v, d_o, d_kpi, d_kpp, d_klpl, ws.view(),
                    nullptr, d_scores, d_sindptr);
            } else {
                kernels::attn::dispatch_attention_flashinfer_decode_bf16(
                    *plan, d_q, d_k, d_v, d_o, d_kpi, d_kpp, d_klpl, ws.view(),
                    nullptr);
            }
        }
        RT(cudaDeviceSynchronize());
        RT(cudaEventRecord(beg));
        for (int it = 0; it < g_bench_iters; ++it) {
            if (capture) {
                kernels::attn::dispatch_attention_flashinfer_decode_capture_bf16(
                    *plan, d_q, d_k, d_v, d_o, d_kpi, d_kpp, d_klpl, ws.view(),
                    nullptr, d_scores, d_sindptr);
            } else {
                kernels::attn::dispatch_attention_flashinfer_decode_bf16(
                    *plan, d_q, d_k, d_v, d_o, d_kpi, d_kpp, d_klpl, ws.view(),
                    nullptr);
            }
        }
        RT(cudaEventRecord(end));
        RT(cudaEventSynchronize(end));
        float ms = 0.f;
        RT(cudaEventElapsedTime(&ms, beg, end));
        g_bench_ms = static_cast<double>(ms) / g_bench_iters;
        cudaEventDestroy(beg);
        cudaEventDestroy(end);
    }

    Run result;
    std::vector<std::uint16_t> o_bf(static_cast<std::size_t>(R) * HQ * D);
    RT(cudaMemcpy(o_bf.data(), d_o, o_bf.size() * 2, cudaMemcpyDeviceToHost));
    result.out.resize(o_bf.size());
    for (std::size_t i = 0; i < o_bf.size(); ++i) {
        result.out[i] = bf16_to_f32(o_bf[i]);
    }
    result.scores.resize(score_elems);
    RT(cudaMemcpy(result.scores.data(), d_scores, score_elems * sizeof(float),
                  cudaMemcpyDeviceToHost));

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_kpi);
    cudaFree(d_kpp);
    cudaFree(d_klpl);
    cudaFree(d_scores);
    cudaFree(d_sindptr);
    return result;
}

// softmax(q·k / sqrt(D)) computed on the host from the same bf16 values.
std::vector<float> reference_scores(const Req& r) {
    std::vector<float> out(static_cast<std::size_t>(HQ) * r.kv_len);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    const int group = HQ / HKV;
    for (int h = 0; h < HQ; ++h) {
        const int kvh = h / group;
        std::vector<float> logits(r.kv_len);
        for (int j = 0; j < r.kv_len; ++j) {
            float dot = 0.f;
            for (int d = 0; d < D; ++d) {
                dot += bf16_to_f32(r.q_bf[static_cast<std::size_t>(h) * D + d]) *
                       bf16_to_f32(
                           r.k_bf[(static_cast<std::size_t>(j) * HKV + kvh) * D +
                                  d]);
            }
            logits[j] = dot * scale;
        }
        const float m = *std::max_element(logits.begin(), logits.end());
        float total = 0.f;
        for (int j = 0; j < r.kv_len; ++j) {
            logits[j] = std::exp(logits[j] - m);
            total += logits[j];
        }
        for (int j = 0; j < r.kv_len; ++j) {
            out[static_cast<std::size_t>(h) * r.kv_len + j] = logits[j] / total;
        }
    }
    return out;
}

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

}  // namespace

int main() {
    if (std::getenv("PIE_ATTN_SCORE_BENCH") != nullptr) {
        g_bench_iters = 50;
        std::printf("%8s %12s %12s %8s\n", "kv_len", "plain_ms", "capture_ms",
                    "ratio");
        for (const int kv : {512, 2048, 6400, 16384}) {
            std::mt19937 rng(0x5C0DEu);
            std::vector<Req> reqs{make_req(rng, kv)};
            run_batch(reqs, /*full_attn=*/false, /*capture=*/false);
            const double plain_ms = g_bench_ms;
            run_batch(reqs, /*full_attn=*/false, /*capture=*/true);
            const double cap_ms = g_bench_ms;
            std::printf("%8d %12.5f %12.5f %8.2fx\n", kv, plain_ms, cap_ms,
                        cap_ms / plain_ms);
        }
        g_bench_iters = 0;
        return 0;
    }

    // Mixed lengths, page boundaries, and a length long enough that the
    // scheduler may choose to split KV across CTAs — the case where "each
    // (head, kv_idx) is written exactly once" has to hold across blocks.
    const std::vector<std::vector<int>> regimes = {
        {2, 4, 5}, {2, 3, 4, 5}, {16, 17}, {1, 64, 33}, {7, 20, 4, 29}, {512},
    };

    for (const auto& kv_lens : regimes) {
        for (const bool full_attn : {true, false}) {
            std::mt19937 rng(0x5C0DEu);
            std::vector<Req> reqs;
            reqs.reserve(kv_lens.size());
            for (const int kv : kv_lens) reqs.push_back(make_req(rng, kv));

            const Run plain = run_batch(reqs, full_attn, /*capture=*/false);
            const Run observed = run_batch(reqs, full_attn, /*capture=*/true);

            char label[128];
            std::snprintf(label, sizeof(label), "R=%zu full_attn=%d",
                          kv_lens.size(), static_cast<int>(full_attn));

            // (a) The observation must be free of side effects on the output.
            bool identical = plain.out.size() == observed.out.size();
            for (std::size_t i = 0; identical && i < plain.out.size(); ++i) {
                identical = std::memcmp(&plain.out[i], &observed.out[i],
                                        sizeof(float)) == 0;
            }
            if (!identical) {
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                              "%s: capture perturbed attention output", label);
                check(false, msg);
            }

            // (b) The captured weights must be the true softmax.
            std::size_t base = 0;
            float worst = 0.f;
            float worst_rowsum = 0.f;
            for (const Req& r : reqs) {
                const std::vector<float> want = reference_scores(r);
                for (std::size_t i = 0; i < want.size(); ++i) {
                    worst = std::max(
                        worst, std::fabs(want[i] - observed.scores[base + i]));
                }
                for (int h = 0; h < HQ; ++h) {
                    float total = 0.f;
                    for (int j = 0; j < r.kv_len; ++j) {
                        total += observed.scores[base +
                                                 static_cast<std::size_t>(h) *
                                                     r.kv_len +
                                                 j];
                    }
                    worst_rowsum =
                        std::max(worst_rowsum, std::fabs(total - 1.f));
                }
                base += want.size();
            }
            // bf16 inputs accumulated in f32; the kernel and the reference
            // sum the head_dim dot product in a different order.
            const bool ok = worst < 2e-3f;
            const bool rows_ok = worst_rowsum < 1e-4f;
            if (!ok || !rows_ok) {
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                              "%s: max|p-ref|=%.3e rowsum err=%.3e", label,
                              worst, worst_rowsum);
                check(false, msg);
            } else {
                std::printf("ok  %-24s max|p-ref|=%.3e rowsum=%.1e\n", label,
                            worst, worst_rowsum);
            }
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_attn_score_capture: PASS\n");
    return 0;
}

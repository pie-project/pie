//===-- test_attn_prefill_score_capture.cu ----------------------*- CUDA -*-===//
//
// Parity test for score-observing prefill (design doc §12, SnapKV).
//
// Three claims, all exact:
//
//  (a) The attention output is BIT-identical to the uncaptured prefill. The
//      capture variant wraps `DefaultAttention` rather than replacing it, so
//      if the wrapper ever perturbed the logit flowing into the softmax, every
//      model running with observation enabled would silently drift. A
//      tolerance would hide exactly the bug worth catching.
//
//  (b) The captured rows are the true CAUSAL softmax. This is the claim most
//      likely to break: `LogitsMask` runs after `LogitsTransform`, so the hook
//      sees real dot products at positions the softmax later discards, and the
//      normalisation pass has to zero them before renormalising. Checked
//      against a naive host `softmax(q·k * sm_scale)` over `[0, limit)`.
//
//  (c) ONLY the last `min(window, qo_len)` query rows are written. The sink is
//      poisoned rather than zeroed, so any stray write outside the observation
//      window shows up as a value that is no longer the poison pattern. This
//      is what makes "the caller must zero the buffer" a safe contract rather
//      than a hope.
//
// Prefill capture is the interesting case precisely because it is NOT decode:
// the decode hook can rely on every `bdx` lane holding the same reduced score
// and elect thread 0 to write, but a prefill thread holds a distinct
// `(q_idx, kv_idx, head)` element of an MMA fragment. Carrying the decode
// guard over would drop 31 of every 32 scores, and that error is invisible
// unless the reference is exact -- a dropped score reads as a smaller
// denominator, i.e. a plausible-looking distribution. Hence (b).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
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

// Qwen3-0.6B geometry: 16 q heads / 8 kv heads (GQA group 2), head_dim 128,
// page_size 16.
constexpr int HQ = 16, HKV = 8, D = 128, PAGE = 16;
constexpr int GQA = HQ / HKV;
const std::size_t PAGE_STRIDE = static_cast<std::size_t>(PAGE) * HKV * D;

// The poison pattern the sink is filled with (via `cudaMemset(0x7f)`). Chosen
// so that a stray write of any plausible score (finite, or even an infinity) is
// distinguishable.
constexpr std::uint32_t kPoisonBits = 0x7f7f7f7fu;
bool is_poison(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return b == kPoisonBits;
}

// One-shot prefill: the prompt is the whole sequence, so kv_len == qo_len.
struct Req {
    int qo_len;
    std::vector<std::uint16_t> q_bf;  // [qo_len, HQ, D]
    std::vector<std::uint16_t> k_bf;  // [qo_len, HKV, D]
    std::vector<std::uint16_t> v_bf;  // [qo_len, HKV, D]
};

Req make_req(std::mt19937& rng, int qo_len) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    Req r;
    r.qo_len = qo_len;
    r.q_bf.resize(static_cast<std::size_t>(qo_len) * HQ * D);
    for (auto& x : r.q_bf) x = f32_to_bf16(u(rng));
    r.k_bf.resize(static_cast<std::size_t>(qo_len) * HKV * D);
    r.v_bf.resize(static_cast<std::size_t>(qo_len) * HKV * D);
    for (auto& x : r.k_bf) x = f32_to_bf16(u(rng));
    for (auto& x : r.v_bf) x = f32_to_bf16(u(rng));
    return r;
}

struct Run {
    std::vector<float> out;     // [total_tokens, HQ, D]
    std::vector<float> scores;  // ragged, [sum_r HQ*window*kv_len_r]
    std::vector<float> folded;  // ragged, [sum_r kv_len_r]
};

// When non-zero, `run_batch` re-launches its dispatch this many times behind
// CUDA events and reports the mean. Isolating the two entry points against an
// identical plan is the only way to attribute tap cost to the kernel rather
// than to the PTIR stage machinery wrapped around it -- and on a shared host
// wall-clock end-to-end numbers are worthless.
int g_bench_iters = 0;
double g_bench_ms = 0.0;

// `window` sizes the sink; `window_arg` is what the dispatch is told. They
// differ only in the refusal cases, where the point is to hand the dispatch a
// value it must reject while still giving it well-formed buffers -- otherwise
// the throw could come from the null-pointer check and the test would pass for
// the wrong reason.
Run run_batch(const std::vector<Req>& reqs, int window, bool capture,
              int window_arg = -1, float logits_soft_cap = 0.f) {
    if (window_arg < 0) window_arg = window;
    const int R = static_cast<int>(reqs.size());
    std::vector<std::uint32_t> qo_indptr_h(R + 1, 0);
    std::vector<std::uint32_t> kv_page_indptr_h(R + 1, 0);
    std::vector<std::uint32_t> kv_last_page_lens_h(R);
    std::vector<std::int32_t> score_indptr_h(R + 1, 0);
    for (int i = 0; i < R; ++i) {
        const int qo = reqs[i].qo_len;
        const int pages = (qo + PAGE - 1) / PAGE;
        qo_indptr_h[i + 1] = qo_indptr_h[i] + static_cast<std::uint32_t>(qo);
        kv_page_indptr_h[i + 1] = kv_page_indptr_h[i] + pages;
        kv_last_page_lens_h[i] =
            static_cast<std::uint32_t>(qo - (pages - 1) * PAGE);
        score_indptr_h[i + 1] = score_indptr_h[i] + HQ * window * qo;
    }
    const int total_tokens = static_cast<int>(qo_indptr_h[R]);
    const int total_pages = static_cast<int>(kv_page_indptr_h[R]);
    std::vector<std::uint32_t> kv_page_indices_h(total_pages);
    for (int p = 0; p < total_pages; ++p) {
        kv_page_indices_h[p] = static_cast<std::uint32_t>(p);
    }

    std::vector<std::uint16_t> q_bf(static_cast<std::size_t>(total_tokens) * HQ * D);
    std::vector<std::uint16_t> kbuf(
        static_cast<std::size_t>(total_pages) * PAGE_STRIDE, 0);
    std::vector<std::uint16_t> vbuf(
        static_cast<std::size_t>(total_pages) * PAGE_STRIDE, 0);
    for (int i = 0; i < R; ++i) {
        const Req& rq = reqs[i];
        std::memcpy(q_bf.data() +
                        static_cast<std::size_t>(qo_indptr_h[i]) * HQ * D,
                    rq.q_bf.data(), rq.q_bf.size() * 2);
        const int base_page = static_cast<int>(kv_page_indptr_h[i]);
        for (int j = 0; j < rq.qo_len; ++j) {
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
    std::uint32_t *d_qoi = nullptr, *d_kpi = nullptr, *d_kpp = nullptr,
                  *d_klpl = nullptr;
    float *d_scores = nullptr, *d_folded = nullptr;
    std::int32_t* d_sindptr = nullptr;
    const std::size_t score_elems =
        static_cast<std::size_t>(score_indptr_h[R]);
    const std::size_t folded_elems =
        score_elems / (static_cast<std::size_t>(HQ) * window);
    RT(cudaMalloc(&d_q, q_bf.size() * 2));
    RT(cudaMalloc(&d_k, kbuf.size() * 2));
    RT(cudaMalloc(&d_v, vbuf.size() * 2));
    RT(cudaMalloc(&d_o, static_cast<std::size_t>(total_tokens) * HQ * D * 2));
    RT(cudaMalloc(&d_qoi, qo_indptr_h.size() * 4));
    RT(cudaMalloc(&d_kpi, kv_page_indices_h.size() * 4));
    RT(cudaMalloc(&d_kpp, kv_page_indptr_h.size() * 4));
    RT(cudaMalloc(&d_klpl, kv_last_page_lens_h.size() * 4));
    RT(cudaMalloc(&d_scores, score_elems * sizeof(float)));
    RT(cudaMalloc(&d_folded, folded_elems * sizeof(float)));
    RT(cudaMalloc(&d_sindptr, score_indptr_h.size() * 4));
    RT(cudaMemcpy(d_q, q_bf.data(), q_bf.size() * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_k, kbuf.data(), kbuf.size() * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_v, vbuf.data(), vbuf.size() * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_qoi, qo_indptr_h.data(), qo_indptr_h.size() * 4,
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpi, kv_page_indices_h.data(),
                  kv_page_indices_h.size() * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_kpp, kv_page_indptr_h.data(), kv_page_indptr_h.size() * 4,
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_klpl, kv_last_page_lens_h.data(),
                  kv_last_page_lens_h.size() * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_sindptr, score_indptr_h.data(), score_indptr_h.size() * 4,
                  cudaMemcpyHostToDevice));
    // Deliberately NOT zeroed: claim (c) is that the kernel only ever touches
    // the observation window, and poison is the only fill that can prove it.
    RT(cudaMemset(d_scores, 0x7f, score_elems * sizeof(float)));
    RT(cudaMemset(d_folded, 0x7f, folded_elems * sizeof(float)));

    auto ws = AttentionWorkspace::allocate(256ull * 1024 * 1024,
                                           64ull * 1024 * 1024);
    auto plan = kernels::attn::make_prefill_plan();
    kernels::attn::plan_attention_flashinfer_prefill_bf16(
        *plan, qo_indptr_h.data(), kv_page_indptr_h.data(),
        kv_last_page_lens_h.data(), total_tokens, R, HQ, HKV, D, PAGE, ws.view(),
        /*stream=*/nullptr, /*enable_cuda_graph=*/false, /*window_left=*/-1,
        /*full_attention_variant=*/true, /*hnd_layout=*/false,
        /*causal_mask=*/true, /*custom_mask=*/false,
        /*wants_prefill_score=*/capture);
    if (capture) {
        kernels::attn::dispatch_attention_flashinfer_prefill_capture_bf16(
            *plan, d_q, d_k, d_v, d_o, d_qoi, d_kpi, d_kpp, d_klpl, ws.view(),
            /*stream=*/nullptr, d_scores, d_folded, d_sindptr, window_arg,
            logits_soft_cap);
    } else {
        kernels::attn::dispatch_attention_flashinfer_prefill_bf16(
            *plan, d_q, d_k, d_v, d_o, d_qoi, d_kpi, d_kpp, d_klpl, ws.view(),
            /*stream=*/nullptr);
    }
    RT(cudaDeviceSynchronize());

    if (g_bench_iters > 0) {
        // The capture entry point also runs `normalize` and `fold`, so what is
        // being timed is the whole tap, not just the variant's extra store.
        auto launch = [&] {
            if (capture) {
                kernels::attn::dispatch_attention_flashinfer_prefill_capture_bf16(
                    *plan, d_q, d_k, d_v, d_o, d_qoi, d_kpi, d_kpp, d_klpl, ws.view(),
                    nullptr, d_scores, d_folded, d_sindptr, window_arg,
                    logits_soft_cap);
            } else {
                kernels::attn::dispatch_attention_flashinfer_prefill_bf16(
                    *plan, d_q, d_k, d_v, d_o, d_qoi, d_kpi, d_kpp, d_klpl, ws.view(),
                    nullptr);
            }
        };
        cudaEvent_t beg = nullptr, end = nullptr;
        RT(cudaEventCreate(&beg));
        RT(cudaEventCreate(&end));
        for (int warm = 0; warm < 3; ++warm) launch();
        RT(cudaDeviceSynchronize());
        RT(cudaEventRecord(beg));
        for (int it = 0; it < g_bench_iters; ++it) launch();
        RT(cudaEventRecord(end));
        RT(cudaEventSynchronize(end));
        float ms = 0.f;
        RT(cudaEventElapsedTime(&ms, beg, end));
        g_bench_ms = static_cast<double>(ms) / g_bench_iters;
        cudaEventDestroy(beg);
        cudaEventDestroy(end);
    }

    Run result;
    std::vector<std::uint16_t> o_bf(
        static_cast<std::size_t>(total_tokens) * HQ * D);
    RT(cudaMemcpy(o_bf.data(), d_o, o_bf.size() * 2, cudaMemcpyDeviceToHost));
    result.out.resize(o_bf.size());
    for (std::size_t i = 0; i < o_bf.size(); ++i) {
        result.out[i] = bf16_to_f32(o_bf[i]);
    }
    result.scores.resize(score_elems);
    result.folded.resize(folded_elems);
    RT(cudaMemcpy(result.scores.data(), d_scores, score_elems * sizeof(float),
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(result.folded.data(), d_folded, folded_elems * sizeof(float),
                  cudaMemcpyDeviceToHost));

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_qoi);
    cudaFree(d_kpi);
    cudaFree(d_kpp);
    cudaFree(d_klpl);
    cudaFree(d_scores);
    cudaFree(d_folded);
    cudaFree(d_sindptr);
    return result;
}

// Naive host reference for one (request, head, window row): the causal softmax
// over `[0, limit)`.
std::vector<float> host_row(const Req& r, int head, int w, int rows) {
    const int kv_len = r.qo_len;
    const int first_qo = r.qo_len - rows;
    const int q_idx = first_qo + w;
    const int limit = q_idx + 1;  // one-shot prefill: abs pos == q_idx
    const int kvh = head / GQA;
    const float sm_scale = 1.f / std::sqrt(static_cast<float>(D));

    std::vector<float> logits(limit);
    for (int k = 0; k < limit; ++k) {
        float dot = 0.f;
        for (int d = 0; d < D; ++d) {
            dot += bf16_to_f32(
                       r.q_bf[(static_cast<std::size_t>(q_idx) * HQ + head) * D + d]) *
                   bf16_to_f32(
                       r.k_bf[(static_cast<std::size_t>(k) * HKV + kvh) * D + d]);
        }
        logits[k] = dot * sm_scale;
    }
    float mx = -INFINITY;
    for (float x : logits) mx = std::max(mx, x);
    float sum = 0.f;
    for (float& x : logits) {
        x = std::exp(x - mx);
        sum += x;
    }
    std::vector<float> p(kv_len, 0.f);
    for (int k = 0; k < limit; ++k) p[k] = logits[k] / sum;
    return p;
}

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", what);
        ++g_failures;
    }
}

void case_capture(const char* name, std::vector<int> lens, int window) {
    std::fprintf(stderr, "[case] %s (window=%d)\n", name, window);
    std::mt19937 rng(0xc0ffeeu);
    std::vector<Req> reqs;
    reqs.reserve(lens.size());
    for (int L : lens) reqs.push_back(make_req(rng, L));

    const Run plain = run_batch(reqs, window, /*capture=*/false);
    const Run capt = run_batch(reqs, window, /*capture=*/true);

    // (a) bit-identical output.
    expect(plain.out.size() == capt.out.size(), "output sizes match");
    const bool bit_identical =
        std::memcmp(plain.out.data(), capt.out.data(),
                    plain.out.size() * sizeof(float)) == 0;
    expect(bit_identical, "capture leaves attention output bit-identical");

    // (b) captured rows are the true causal softmax, and
    // (c) nothing outside the observation window was touched.
    const int R = static_cast<int>(reqs.size());
    std::size_t base = 0;
    std::size_t folded_base = 0;
    double worst = 0.0;
    for (int i = 0; i < R; ++i) {
        const Req& r = reqs[i];
        const int kv_len = r.qo_len;
        const int rows = std::min(window, r.qo_len);
        std::vector<double> fold_ref(kv_len, 0.0);
        for (int h = 0; h < HQ; ++h) {
            for (int w = 0; w < window; ++w) {
                const std::size_t off =
                    base + (static_cast<std::size_t>(h) * window + w) * kv_len;
                if (w >= rows) {
                    bool untouched = true;
                    for (int k = 0; k < kv_len; ++k) {
                        if (!is_poison(capt.scores[off + k])) untouched = false;
                    }
                    expect(untouched,
                           "rows past min(window, qo_len) are never written");
                    continue;
                }
                const std::vector<float> ref = host_row(r, h, w, rows);
                double sum = 0.0;
                for (int k = 0; k < kv_len; ++k) {
                    const float got = capt.scores[off + k];
                    worst = std::max(worst,
                                     static_cast<double>(std::fabs(got - ref[k])));
                    sum += got;
                    fold_ref[k] += ref[k];
                }
                if (std::fabs(sum - 1.0) > 2e-3) {
                    std::fprintf(stderr,
                                 "  row r=%d h=%d w=%d sums to %.6f\n", i, h, w,
                                 sum);
                    expect(false, "captured row is a distribution");
                }
            }
        }
        const double inv = 1.0 / (static_cast<double>(HQ) * rows);
        for (int k = 0; k < kv_len; ++k) {
            const double want = fold_ref[k] * inv;
            const double got = capt.folded[folded_base + k];
            if (std::fabs(got - want) > 2e-3) {
                std::fprintf(stderr,
                             "  fold r=%d k=%d got %.6f want %.6f\n", i, k, got,
                             want);
                expect(false, "folded row matches the host mean");
                break;
            }
        }
        base += static_cast<std::size_t>(HQ) * window * kv_len;
        folded_base += static_cast<std::size_t>(kv_len);
    }
    std::fprintf(stderr, "  worst |p_gpu - p_host| = %.3e\n", worst);
    expect(worst < 3e-3, "captured probabilities match the host reference");
}

// The refusals are part of the contract: a configuration where the captured
// score would not mean what SnapKV assumes must fail loudly, because the
// failure mode of a silent wrong score is "evict the wrong half of the
// prompt", which no downstream test can distinguish from a bad policy.
void case_refusals() {
    std::fprintf(stderr, "[case] refusals\n");
    std::mt19937 rng(7u);
    const std::vector<Req> reqs{make_req(rng, 40)};

    struct Bad {
        const char* what;
        int window_arg;
        float soft_cap;
    };
    const Bad bad[] = {
        {"a non-positive window is refused", 0, 0.f},
        {"logits_soft_cap is refused", 8, 30.f},
    };
    for (const Bad& b : bad) {
        bool threw = false;
        try {
            run_batch(reqs, /*window=*/8, /*capture=*/true, b.window_arg,
                      b.soft_cap);
        } catch (const std::exception&) {
            threw = true;
        }
        expect(threw, b.what);
    }
}

}  // namespace

int main() {
    if (std::getenv("PIE_ATTN_PREFILL_SCORE_BENCH") != nullptr) {
        // Prefill is compute-bound and the tap adds one f32 store per
        // (q_row_in_window, kv, head) evaluated. Rows outside the window take a
        // register compare and nothing else, so the tap's relative cost falls as
        // the prompt grows past the window -- the opposite of the decode
        // capture, where every fire is one row and the tap is always paid.
        g_bench_iters = 20;
        std::printf("%8s %8s %12s %12s %8s\n", "qo_len", "window", "plain_ms",
                    "capture_ms", "ratio");
        for (const int qo : {512, 2048, 8192}) {
            for (const int win : {32, 128}) {
                std::mt19937 rng(0x5C0DEu);
                const std::vector<Req> reqs{make_req(rng, qo)};
                run_batch(reqs, win, /*capture=*/false);
                const double plain_ms = g_bench_ms;
                run_batch(reqs, win, /*capture=*/true);
                const double cap_ms = g_bench_ms;
                std::printf("%8d %8d %12.5f %12.5f %8.2fx\n", qo, win, plain_ms,
                            cap_ms, cap_ms / plain_ms);
            }
        }
        g_bench_iters = 0;
        return 0;
    }

    // A single short prompt: exercises rows < window and a partial last page.
    case_capture("single short prompt", {40}, 8);
    // Multi-request, ragged, crossing the 128-row q-tile boundary so the last
    // MMA tile is padded -- the shape that needs the q_idx bounds check the
    // decode capture never needed.
    case_capture("ragged batch across a q-tile boundary", {200, 5, 137, 64}, 8);
    // A window wider than the shortest prompt, and wide enough to span more
    // than one MMA row group.
    case_capture("window wider than a prompt", {12, 300}, 32);
    case_refusals();

    if (g_failures == 0) {
        std::fprintf(stderr, "test_attn_prefill_score_capture: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_attn_prefill_score_capture: %d FAILURE(S)\n",
                 g_failures);
    return 1;
}

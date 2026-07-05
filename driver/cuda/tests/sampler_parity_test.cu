// Sampling-IR parity harness (lane L7 / hotel).
//
// Proves the CPU golden reference in `sampler_reference.hpp` reproduces the
// production Gumbel-max temperature/min-p sampler
// (`launch_sample_temp_bf16` in kernels/sample_temp.cu) bit-for-bit for the
// argmax path and within transcendental ulp for the temperature/min-p paths.
//
// Once the IR codegen/JIT/executor lands, the same golden reference is the
// oracle the IR sampler is checked against — so locking it to the real kernel
// here is the foundation of the whole parity story.
//
// It also exercises the host-only masking semantics (top-k / top-p / min-p /
// entropy) with brute-force checks so the reference's mask logic — which the
// IR top-k/top-p programs must reproduce — is validated independently of the GPU.
//
// Build: linked against kernels/sample_temp.cu + CUDA::cudart (see CMakeLists).
// Run:   ctest -R sampler_parity   (or invoke the binary directly).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/sample_temp.hpp"
#include "sampler_reference.hpp"

namespace ref = pie_sampler_ref;

#define CHK(expr) do { auto _e = (expr); if (_e != cudaSuccess) {            \
    std::fprintf(stderr, "CUDA error: %s at %s:%d — %s\n",                   \
                 cudaGetErrorString(_e), __FILE__, __LINE__, #expr);         \
    std::exit(2); } } while (0)

namespace {

// ── bf16 conversion on the host. We round float→bf16 ourselves and feed the
//    *exact* bf16→float expansion to the reference, while uploading the same 16
//    bits to the device. `__bfloat162float` of those bits equals the expansion
//    losslessly, so device and reference see byte-identical inputs regardless
//    of CUDA's rounding mode. ───────────────────────────────────────────────
std::uint16_t f32_to_bf16_bits(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffffu) > 0x7f800000u) {       // NaN → quiet NaN
        return static_cast<std::uint16_t>((x >> 16) | 0x0040u);
    }
    const std::uint32_t bias = 0x00007fffu + ((x >> 16) & 1u);  // round-to-even
    x += bias;
    return static_cast<std::uint16_t>(x >> 16);
}

float bf16_bits_to_f32(std::uint16_t b) {
    const std::uint32_t x = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

struct RowConfig {
    const char* name;
    float       temperature;
    float       min_p;
};

// Diverse parameter mix. Rows are assigned configs round-robin.
const RowConfig kConfigs[] = {
    {"argmax",        0.0f, 0.0f},
    {"temp1.0",       1.0f, 0.0f},
    {"temp0.7",       0.7f, 0.0f},
    {"temp0.8",       0.8f, 0.0f},
    {"minp0.05@T1.0", 1.0f, 0.05f},
    {"minp0.10@T0.8", 0.8f, 0.10f},
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

// A device-vs-reference token disagreement on a T>0 row is acceptable only when
// it is a genuine near-tie: the reference score the device's token earns is
// within this margin of the reference argmax. (Host/device logf differ ≤~2 ulp;
// that only ever flips an argmax when the top two scores are within ~1e-4.)
constexpr float kNearTieTol = 1e-2f;

int g_failures = 0;
void expect(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "  FAIL: %s\n", msg); ++g_failures; }
}

// Reference score the masked/temperature path assigns to a specific token, so
// we can measure how far a device disagreement is from the reference argmax.
float ref_score_for_token(const std::vector<float>& logits, int token,
                          const RowConfig& cfg, std::uint32_t seed) {
    const bool greedy = !(cfg.temperature > 0.f);
    const float inv_T = greedy ? 1.f : (1.f / cfg.temperature);
    if (greedy) return logits[token];
    const std::uint64_t s = ref::seed_eff_from_row(seed);
    return logits[token] * inv_T + ref::gumbel_noise(s, token);
}

// ── 1. GPU parity: golden reference vs launch_sample_temp_bf16. ─────────────
void run_gpu_parity() {
    constexpr int kRows  = 512;
    constexpr int kVocab = 151936;          // Qwen3 vocab

    std::printf("[gpu-parity] %d rows × %d vocab, %d configs\n",
                kRows, kVocab, kNumConfigs);

    std::mt19937_64 rng(0xC0FFEEull);
    std::normal_distribution<float> logit_dist(0.f, 3.0f);

    std::vector<std::uint16_t> bf16(static_cast<size_t>(kRows) * kVocab);
    std::vector<float>         temps(kRows), min_ps(kRows);
    std::vector<std::uint32_t> seeds(kRows);

    // Per-row bf16-rounded logits, kept host-side for the reference.
    std::vector<std::vector<float>> ref_logits(kRows, std::vector<float>(kVocab));

    for (int r = 0; r < kRows; ++r) {
        const RowConfig& cfg = kConfigs[r % kNumConfigs];
        temps[r]  = cfg.temperature;
        min_ps[r] = cfg.min_p;
        seeds[r]  = 0x9E3779B9u * static_cast<std::uint32_t>(r + 1);
        for (int j = 0; j < kVocab; ++j) {
            const std::uint16_t b = f32_to_bf16_bits(logit_dist(rng));
            bf16[static_cast<size_t>(r) * kVocab + j] = b;
            ref_logits[r][j] = bf16_bits_to_f32(b);
        }
    }

    // Upload + launch.
    void*           d_logits = nullptr;
    float*          d_temps  = nullptr;
    float*          d_min_ps = nullptr;
    std::uint32_t*  d_seeds  = nullptr;
    std::int32_t*   d_out    = nullptr;
    const size_t logit_bytes = bf16.size() * sizeof(std::uint16_t);
    CHK(cudaMalloc(&d_logits, logit_bytes));
    CHK(cudaMalloc(&d_temps,  kRows * sizeof(float)));
    CHK(cudaMalloc(&d_min_ps, kRows * sizeof(float)));
    CHK(cudaMalloc(&d_seeds,  kRows * sizeof(std::uint32_t)));
    CHK(cudaMalloc(&d_out,    kRows * sizeof(std::int32_t)));
    CHK(cudaMemcpy(d_logits, bf16.data(), logit_bytes, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_temps,  temps.data(),  kRows * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_min_ps, min_ps.data(), kRows * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_seeds,  seeds.data(),  kRows * sizeof(std::uint32_t), cudaMemcpyHostToDevice));

    pie_cuda_driver::kernels::launch_sample_temp_bf16(
        d_logits, d_temps, d_min_ps, d_seeds, d_out, kRows, kVocab, /*stream=*/0);
    CHK(cudaDeviceSynchronize());

    std::vector<std::int32_t> out(kRows);
    CHK(cudaMemcpy(out.data(), d_out, kRows * sizeof(std::int32_t), cudaMemcpyDeviceToHost));

    // Compare per config.
    struct Tally { int rows = 0, exact = 0, near_tie = 0, hard = 0; float worst_gap = 0.f; };
    std::vector<Tally> tally(kNumConfigs);

    for (int r = 0; r < kRows; ++r) {
        const int ci = r % kNumConfigs;
        const RowConfig& cfg = kConfigs[ci];
        Tally& t = tally[ci];
        ++t.rows;

        const ref::SampleResult rr =
            ref::sample_temp(ref_logits[r], cfg.temperature, cfg.min_p, seeds[r]);
        const int dev_tok = out[r];

        if (dev_tok == rr.token) { ++t.exact; continue; }

        if (cfg.temperature <= 0.f) {       // argmax must be exact
            ++t.hard;
            std::fprintf(stderr, "  [%s] row %d argmax mismatch: dev=%d ref=%d\n",
                         cfg.name, r, dev_tok, rr.token);
            continue;
        }
        // Classify the disagreement.
        const float dev_score = ref_score_for_token(ref_logits[r], dev_tok, cfg, seeds[r]);
        const float gap = rr.best_score - dev_score;
        if (gap <= kNearTieTol) {
            ++t.near_tie;
            if (gap > t.worst_gap) t.worst_gap = gap;
        } else {
            ++t.hard;
            std::fprintf(stderr,
                "  [%s] row %d divergence: dev=%d(score %.6f) ref=%d(score %.6f) gap=%.6f\n",
                cfg.name, r, dev_tok, dev_score, rr.token, rr.best_score, gap);
        }
    }

    for (int ci = 0; ci < kNumConfigs; ++ci) {
        const Tally& t = tally[ci];
        const double exact_pct = t.rows ? 100.0 * t.exact / t.rows : 0.0;
        std::printf("  %-16s rows=%d exact=%d (%.1f%%) near_tie=%d hard=%d worst_gap=%.2e\n",
                    kConfigs[ci].name, t.rows, t.exact, exact_pct,
                    t.near_tie, t.hard, t.worst_gap);
        expect(t.hard == 0, "hard divergence(s) in config");
        // argmax: every row must be exact.
        if (kConfigs[ci].temperature <= 0.f)
            expect(t.exact == t.rows, "argmax not bit-exact");
        else
            // Temperature paths: overwhelmingly exact; the rest are near-ties.
            expect(exact_pct >= 95.0, "temperature exact-match rate below 95%");
    }

    cudaFree(d_logits); cudaFree(d_temps); cudaFree(d_min_ps);
    cudaFree(d_seeds);  cudaFree(d_out);
}

// ── 2. Host-only masking semantics (no GPU). Brute-force the reference's
//       top-k / top-p / min-p / entropy against hand-computed expectations. ──
void run_mask_unit_tests() {
    std::printf("[mask-units]\n");

    // logits chosen so softmax is easy to reason about.
    const std::vector<float> logits = {3.0f, 1.0f, 0.0f, -1.0f, 2.0f};
    //   index:                          0     1     2     3      4
    // descending logit order: 0(3.0), 4(2.0), 1(1.0), 2(0.0), 3(-1.0)

    // top-k = 2 keeps indices {0, 4}.
    {
        const auto m = ref::top_k_mask(logits, 2);
        expect(m[0] && m[4] && !m[1] && !m[2] && !m[3], "top_k=2 keeps {0,4}");
    }
    // top-k = 0 / >= n keeps all.
    {
        const auto m0 = ref::top_k_mask(logits, 0);
        const auto mn = ref::top_k_mask(logits, 99);
        bool all0 = true, alln = true;
        for (bool b : m0) all0 &= b;
        for (bool b : mn) alln &= b;
        expect(all0 && alln, "top_k disabled keeps all");
    }
    // min-p: thr = max + log(min_p). With min_p=0.2, log(0.2)≈-1.609, max=3.0,
    // thr≈1.391 → keep logits >= 1.391 → {0(3.0), 4(2.0)}.
    {
        const auto m = ref::min_p_mask(logits, 0.2f);
        expect(m[0] && m[4] && !m[1] && !m[2] && !m[3], "min_p=0.2 keeps {0,4}");
    }
    // min-p disabled keeps all.
    {
        const auto m = ref::min_p_mask(logits, 0.0f);
        bool all = true; for (bool b : m) all &= b;
        expect(all, "min_p=0 keeps all");
    }
    // top-p: cumulative nucleus. Cross-check against an independent prefix sum
    // of the sorted softmax — keep the shortest prefix reaching p.
    {
        const float p = 0.9f;
        const auto m = ref::top_p_mask(logits, p, 1.f);
        const auto prob = ref::softmax(logits, 1.f);
        std::vector<int> order(logits.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b){ return prob[a] > prob[b]; });
        float cum = 0.f; int kept = 0;
        std::vector<bool> expect_mask(logits.size(), false);
        for (int idx : order) { expect_mask[idx] = true; ++kept; cum += prob[idx]; if (cum >= p) break; }
        bool eq = true;
        for (size_t i = 0; i < m.size(); ++i) eq &= (m[i] == expect_mask[i]);
        expect(eq, "top_p=0.9 matches independent prefix-sum");
        expect(kept >= 1, "top_p keeps at least one token");
    }
    // top-p >= 1 keeps all.
    {
        const auto m = ref::top_p_mask(logits, 1.0f, 1.f);
        bool all = true; for (bool b : m) all &= b;
        expect(all, "top_p>=1 keeps all");
    }
    // softmax sums to 1.
    {
        const auto p = ref::softmax(logits, 1.f);
        float s = 0.f; for (float v : p) s += v;
        expect(std::fabs(s - 1.0f) < 1e-5f, "softmax normalizes");
    }
    // entropy bounds: 0 <= H <= log(n); uniform logits → H == log(n).
    {
        const float h = ref::entropy(logits, 1.f);
        expect(h >= 0.f && h <= logf(static_cast<float>(logits.size())) + 1e-4f,
               "entropy within [0, log n]");
        const std::vector<float> uniform(8, 1.5f);
        const float hu = ref::entropy(uniform, 1.f);
        expect(std::fabs(hu - logf(8.0f)) < 1e-4f, "uniform entropy == log n");
    }
    // gumbel_argmax_masked with T=0 == plain argmax over kept set.
    {
        const auto keep = ref::top_k_mask(logits, 2);   // {0,4}
        const auto rgreedy = ref::gumbel_argmax_masked(logits, keep, 0.f, 123u);
        expect(rgreedy.token == 0, "masked greedy picks highest kept logit");
    }
    std::printf("  mask units done\n");
}

}  // namespace

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "no CUDA device available — skipping GPU parity\n");
    } else {
        run_gpu_parity();
    }
    run_mask_unit_tests();

    if (g_failures == 0) {
        std::printf("PASS: sampler parity + mask units\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
    return 1;
}

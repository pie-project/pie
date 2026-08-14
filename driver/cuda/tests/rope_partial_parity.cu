// Parity guard for `launch_rope_partial_bf16` against a CPU reference.
//
// WHY THIS EXISTS. Partial rotary rotates only the first `rotary_dim`
// channels of each head and leaves `[rotary_dim, head_dim)` untouched. The
// CUDA kernel had BOTH halves of that contract wrong — it used `head_dim` as
// the frequency denominator and `head_dim/2` as the pair offset — and the
// error survived because nothing tested this launcher. It survived twice: a
// comment above the kernel asserted the incorrect form was right and called
// the correct form "the previous draft [that] got it wrong", so the code had
// been changed INTO the bug and documented confidently.
//
// For Qwen3.6-27B (head_dim 256, partial_rotary_factor 0.25, rotary_dim 64)
// the consequences were:
//
//   * dims 32..63   left UNROTATED  — they are the second half of each pair
//   * dims 128..159 OVERWRITTEN     — they are pass-through
//   * frequency denominator 4x too large, angle off by up to 1.2e5 at j=31
//
// Observably that produced a systematic ~2.2 nat logit disagreement against
// vLLM on the same checkpoint, which is exactly zero at relative distance 0
// and grows with |m-n| — so a decode agreed for its first several tokens and
// then diverged, fluently, in a way that read as "the model is worse".
//
// The reference below is the HF definition, written out longhand rather than
// factored, so that a future edit cannot make reference and kernel wrong in
// the same direction. THE POINT OF THIS TEST IS THE THREE ASSERTIONS BELOW,
// not the tolerance:
//
//   1. channels [0, rotary_dim) rotate with pair offset rotary_dim/2
//   2. channels [rotary_dim, head_dim) are BYTE-IDENTICAL to the input
//   3. the frequency denominator is rotary_dim, not head_dim
//
// Assertion 2 is the cheap one that would have caught this on day one: the
// buggy kernel wrote to dims 128..159, and no tolerance is needed to see it.
//
// The kernel accumulates in fp32 and stores bf16, so rotated channels are
// compared with a tolerance. bf16 carries ~8 mantissa bits; a correct
// rotation of unit-scale values lands well inside 5e-3, while either of the
// two historical bugs lands orders of magnitude outside it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/rope.hpp"

namespace kernels = pie_cuda_driver::kernels;

namespace {

int g_failures = 0;

std::uint16_t float_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

float bf16_to_float(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// HF partial rotary, longhand. `rotate_half` is applied to the slice
// x[..., :rotary_dim], so the pair partner is rotary_dim/2 away and the
// frequency denominator is rotary_dim.
void reference_rope_partial(
    std::vector<float>& head, int head_dim, int rotary_dim, int pos, float theta) {
    const int angles = rotary_dim / 2;
    const std::vector<float> in = head;
    for (int j = 0; j < angles; ++j) {
        const double freq = std::pow(static_cast<double>(theta),
                                     -2.0 * static_cast<double>(j) /
                                         static_cast<double>(rotary_dim));
        const double ang = static_cast<double>(pos) * freq;
        const double c = std::cos(ang), s = std::sin(ang);
        const double a = in[j], b = in[j + angles];
        head[j] = static_cast<float>(a * c - b * s);
        head[j + angles] = static_cast<float>(b * c + a * s);
    }
    // [rotary_dim, head_dim) deliberately untouched.
}

struct Case {
    const char* label;
    int head_dim;
    int rotary_dim;
    int num_q_heads;
    int num_kv_heads;
    int num_tokens;
    float theta;
};

void run_case(const Case& c) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    const int total_q = c.num_tokens * c.num_q_heads * c.head_dim;
    const int total_k = c.num_tokens * c.num_kv_heads * c.head_dim;
    std::vector<std::uint16_t> q(total_q), k(total_k);
    std::vector<float> q_ref(total_q), k_ref(total_k);
    for (int i = 0; i < total_q; ++i) {
        const float v = dist(rng);
        q[i] = float_to_bf16(v);
        q_ref[i] = bf16_to_float(q[i]);
    }
    for (int i = 0; i < total_k; ++i) {
        const float v = dist(rng);
        k[i] = float_to_bf16(v);
        k_ref[i] = bf16_to_float(k[i]);
    }
    const std::vector<std::uint16_t> q_in = q, k_in = k;

    std::vector<std::int32_t> positions(c.num_tokens);
    for (int n = 0; n < c.num_tokens; ++n) positions[n] = 3 + 5 * n;

    // CPU reference, per head.
    for (int n = 0; n < c.num_tokens; ++n) {
        for (int h = 0; h < c.num_q_heads; ++h) {
            std::vector<float> head(q_ref.begin() + (n * c.num_q_heads + h) * c.head_dim,
                                    q_ref.begin() + (n * c.num_q_heads + h + 1) * c.head_dim);
            reference_rope_partial(head, c.head_dim, c.rotary_dim, positions[n], c.theta);
            std::copy(head.begin(), head.end(),
                      q_ref.begin() + (n * c.num_q_heads + h) * c.head_dim);
        }
        for (int h = 0; h < c.num_kv_heads; ++h) {
            std::vector<float> head(k_ref.begin() + (n * c.num_kv_heads + h) * c.head_dim,
                                    k_ref.begin() + (n * c.num_kv_heads + h + 1) * c.head_dim);
            reference_rope_partial(head, c.head_dim, c.rotary_dim, positions[n], c.theta);
            std::copy(head.begin(), head.end(),
                      k_ref.begin() + (n * c.num_kv_heads + h) * c.head_dim);
        }
    }

    void *dq = nullptr, *dk = nullptr;
    std::int32_t* dpos = nullptr;
    cudaMalloc(&dq, total_q * sizeof(std::uint16_t));
    cudaMalloc(&dk, total_k * sizeof(std::uint16_t));
    cudaMalloc(&dpos, c.num_tokens * sizeof(std::int32_t));
    cudaMemcpy(dq, q.data(), total_q * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), total_k * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, positions.data(), c.num_tokens * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);

    kernels::launch_rope_partial_bf16(dq, dk, dpos, c.num_tokens, c.num_q_heads,
                                      c.num_kv_heads, c.head_dim, c.rotary_dim,
                                      c.theta, /*stream=*/nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(q.data(), dq, total_q * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(k.data(), dk, total_k * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(dpos);

    const double tol = 5e-3;
    double max_rot = 0.0;
    int passthrough_violations = 0;

    auto check = [&](const std::vector<std::uint16_t>& got,
                     const std::vector<std::uint16_t>& in,
                     const std::vector<float>& ref, int heads) {
        for (int n = 0; n < c.num_tokens; ++n) {
            for (int h = 0; h < heads; ++h) {
                const int base = (n * heads + h) * c.head_dim;
                for (int d = 0; d < c.head_dim; ++d) {
                    if (d < c.rotary_dim) {
                        max_rot = std::fmax(max_rot,
                            std::fabs(bf16_to_float(got[base + d]) - ref[base + d]));
                    } else if (got[base + d] != in[base + d]) {
                        // Assertion 2: pass-through channels must be BIT-identical.
                        // The historical bug wrote dims 128..159 here.
                        ++passthrough_violations;
                    }
                }
            }
        }
    };
    check(q, q_in, q_ref, c.num_q_heads);
    check(k, k_in, k_ref, c.num_kv_heads);

    const bool ok = (max_rot <= tol) && (passthrough_violations == 0);
    if (!ok) ++g_failures;
    std::printf("[%s] %-28s head_dim=%d rotary_dim=%d  max_rot=%.3e  passthrough_violations=%d\n",
                ok ? "ok" : "FAIL", c.label, c.head_dim, c.rotary_dim, max_rot,
                passthrough_violations);
}

}  // namespace

int main() {
    const Case cases[] = {
        // The shape that was broken in production.
        {"qwen3.6-27b", 256, 64, 8, 2, 4, 1e7f},
        // A different partial factor, so a fix that hardcodes 64 is caught.
        {"partial-half", 128, 64, 4, 2, 3, 1e6f},
        {"partial-eighth", 256, 32, 4, 1, 2, 1e7f},
        // rotary_dim == head_dim: both historical bugs vanish here, which is
        // exactly why every full-rotary model stayed correct and this went
        // unnoticed. Kept so the fix cannot regress the full case.
        {"full-rotary", 128, 128, 4, 2, 3, 1e6f},
    };
    for (const auto& c : cases) run_case(c);

    if (g_failures != 0) {
        std::printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}

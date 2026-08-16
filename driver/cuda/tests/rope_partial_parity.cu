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

// ── vLLM cos/sin-table emulation (PIE_ROPE_VLLM_TABLE / the explicit
//    `launch_rope_partial_vllm_table_bf16` launcher) ─────────────────────────
//
// This reference is deliberately NOT the accurate one above. vLLM computes its
// cos/sin cache once on the host in fp32 and then rounds the whole table to
// bf16 before storing it; the `triton_mrope` path indexes that bf16 table with
// no fp32 cast and casts q/k down to match, so the rotate is bf16 too. A Pie
// that computed accurate fp32 trig per token would be MORE accurate than the
// reference and still mismatch it, so what is pinned here is the ROUNDING
// STRUCTURE, not the accuracy:
//
//   4. inv_freq = 1 / theta^(2j/rotary_dim)  -- reciprocal AFTER the power,
//      positive exponent; not the `theta^(-2j/d)` the default path uses
//   5. cos/sin rounded to bf16 (RNE) BEFORE they are used at all
//   6. the rotate itself in bf16: three separately-rounded operations per
//      output, not one fp32 expression rounded at the end
//
// Numbering continues the three assertions listed at the top of this file;
// 1-3 (which channels rotate, which pass through, the denominator) apply to
// this path unchanged and are checked for it as well.

float vllm_inv_freq(float theta, int j, int rotary_dim) {
    const float exponent = (2.f * static_cast<float>(j)) /
                           static_cast<float>(rotary_dim);
    const float p = static_cast<float>(
        std::pow(static_cast<double>(theta), static_cast<double>(exponent)));
    return 1.f / p;
}

// A lane whose accurate fp32 trig value sits within `kTieMargin` fp32 ulp of a
// bf16 rounding boundary can round either way depending on which correctly-
// rounded-to-2-ulp implementation evaluated it (device `sincosf` vs host
// `std::cos`). Those lanes are excluded from the bit-exact comparison and
// counted instead. The exclusion is deterministic -- fixed seed, fixed
// positions -- so this cannot make the test flaky; it is reported so that a
// change which silently pushes many lanes into the excluded set is visible.
constexpr std::uint32_t kTieMargin = 16;

bool near_bf16_boundary(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const std::uint32_t r = bits & 0xffffu;
    return r + kTieMargin >= 0x8000u && r <= 0x8000u + kTieMargin;
}

// Rotates in place over bf16 bit patterns, so every rounding the kernel does is
// reproduced here rather than approximated.
void reference_rope_partial_vllm(
    std::vector<std::uint16_t>& head, int rotary_dim, int pos, float theta,
    std::vector<char>& lane_excluded) {
    const int angles = rotary_dim / 2;
    const std::vector<std::uint16_t> in = head;
    for (int j = 0; j < angles; ++j) {
        const float ang = static_cast<float>(pos) *
                          vllm_inv_freq(theta, j, rotary_dim);
        const float c32 = static_cast<float>(std::cos(static_cast<double>(ang)));
        const float s32 = static_cast<float>(std::sin(static_cast<double>(ang)));
        lane_excluded[j] =
            near_bf16_boundary(c32) || near_bf16_boundary(s32) ? 1 : 0;

        const float c = bf16_to_float(float_to_bf16(c32));
        const float s = bf16_to_float(float_to_bf16(s32));
        const float a = bf16_to_float(in[j]);
        const float b = bf16_to_float(in[j + angles]);

        // Three roundings per output, in Triton's operand order.
        const float ac = bf16_to_float(float_to_bf16(a * c));
        const float bs = bf16_to_float(float_to_bf16(b * s));
        const float bc = bf16_to_float(float_to_bf16(b * c));
        const float as = bf16_to_float(float_to_bf16(a * s));
        head[j] = float_to_bf16(ac - bs);
        head[j + angles] = float_to_bf16(bc + as);
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

// Bit-exact guard for the vLLM-table path. Positions are LONG on purpose:
// `inv_freq[0]` is exactly 1.0 at rotary_dim=64/theta=1e7, so lane 0's angle in
// radians is the token position itself, and everything this path exists to fix
// only becomes visible past a few hundred.
void run_vllm_case(const Case& c) {
    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    const int total_q = c.num_tokens * c.num_q_heads * c.head_dim;
    const int total_k = c.num_tokens * c.num_kv_heads * c.head_dim;
    std::vector<std::uint16_t> q(total_q), k(total_k);
    for (int i = 0; i < total_q; ++i) q[i] = float_to_bf16(dist(rng));
    for (int i = 0; i < total_k; ++i) k[i] = float_to_bf16(dist(rng));
    const std::vector<std::uint16_t> q_in = q, k_in = k;
    std::vector<std::uint16_t> q_ref = q, k_ref = k;

    std::vector<std::int32_t> positions(c.num_tokens);
    for (int n = 0; n < c.num_tokens; ++n) positions[n] = 13000 + 1777 * n;

    const int angles = c.rotary_dim / 2;
    // excluded[token][j]: set by the reference when lane j of that token's row
    // sits on a bf16 rounding boundary.
    std::vector<std::vector<char>> excluded(
        c.num_tokens, std::vector<char>(angles, 0));

    auto ref_heads = [&](std::vector<std::uint16_t>& buf, int heads) {
        for (int n = 0; n < c.num_tokens; ++n) {
            for (int h = 0; h < heads; ++h) {
                const int base = (n * heads + h) * c.head_dim;
                std::vector<std::uint16_t> head(buf.begin() + base,
                                                buf.begin() + base + c.head_dim);
                reference_rope_partial_vllm(head, c.rotary_dim, positions[n],
                                            c.theta, excluded[n]);
                std::copy(head.begin(), head.end(), buf.begin() + base);
            }
        }
    };
    ref_heads(q_ref, c.num_q_heads);
    ref_heads(k_ref, c.num_kv_heads);

    void *dq = nullptr, *dk = nullptr;
    std::int32_t* dpos = nullptr;
    cudaMalloc(&dq, total_q * sizeof(std::uint16_t));
    cudaMalloc(&dk, total_k * sizeof(std::uint16_t));
    cudaMalloc(&dpos, c.num_tokens * sizeof(std::int32_t));
    cudaMemcpy(dq, q.data(), total_q * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), total_k * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, positions.data(), c.num_tokens * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);

    kernels::launch_rope_partial_vllm_table_bf16(
        dq, dk, dpos, c.num_tokens, c.num_q_heads, c.num_kv_heads, c.head_dim,
        c.rotary_dim, c.theta, /*stream=*/nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(q.data(), dq, total_q * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(k.data(), dk, total_k * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(dpos);

    int bit_mismatches = 0, passthrough_violations = 0, skipped = 0;
    auto check = [&](const std::vector<std::uint16_t>& got,
                     const std::vector<std::uint16_t>& in,
                     const std::vector<std::uint16_t>& ref, int heads) {
        for (int n = 0; n < c.num_tokens; ++n) {
            for (int h = 0; h < heads; ++h) {
                const int base = (n * heads + h) * c.head_dim;
                for (int d = 0; d < c.head_dim; ++d) {
                    if (d >= c.rotary_dim) {
                        // Assertion 2: pass-through stays bit-identical here too.
                        if (got[base + d] != in[base + d]) ++passthrough_violations;
                        continue;
                    }
                    if (excluded[n][d % angles]) { ++skipped; continue; }
                    // Assertions 1, 3, 4, 5, 6: every rounding reproduced.
                    if (got[base + d] != ref[base + d]) ++bit_mismatches;
                }
            }
        }
    };
    check(q, q_in, q_ref, c.num_q_heads);
    check(k, k_in, k_ref, c.num_kv_heads);

    const bool ok = bit_mismatches == 0 && passthrough_violations == 0;
    if (!ok) ++g_failures;
    std::printf("[%s] %-28s head_dim=%d rotary_dim=%d  bit_mismatches=%d  "
                "passthrough_violations=%d  tie_lanes_skipped=%d\n",
                ok ? "ok" : "FAIL", c.label, c.head_dim, c.rotary_dim,
                bit_mismatches, passthrough_violations, skipped);
}

// Non-vacuity, and the sign of the fix. At position 20000 the default path's
// `__sincosf` has lost its range reduction -- lane 0's angle is 20000 rad --
// while the vLLM-table path should land within a couple of bf16 steps of true
// math. If this ever reports the two paths as comparable, either the knob is
// not routing or `sincosf` has been mapped onto the intrinsic by a fast-math
// flag, and the whole change is inert.
void run_long_position_probe() {
    constexpr int head_dim = 256, rotary_dim = 64, heads = 4;
    constexpr float theta = 1e7f;
    // Several long positions, not one: `__sincosf`'s error at a single position
    // could land small by luck, and a probe that can pass by luck is not a probe.
    const std::vector<std::int32_t> positions = {13000, 16384, 20000, 24000};
    const int tokens = static_cast<int>(positions.size());
    const int total = tokens * heads * head_dim;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<std::uint16_t> src(total);
    for (int i = 0; i < total; ++i) src[i] = float_to_bf16(dist(rng));

    std::vector<float> ref(total);
    for (int i = 0; i < total; ++i) ref[i] = bf16_to_float(src[i]);
    for (int n = 0; n < tokens; ++n) {
        for (int h = 0; h < heads; ++h) {
            const int base = (n * heads + h) * head_dim;
            std::vector<float> head(ref.begin() + base,
                                    ref.begin() + base + head_dim);
            reference_rope_partial(head, head_dim, rotary_dim, positions[n], theta);
            std::copy(head.begin(), head.end(), ref.begin() + base);
        }
    }

    auto max_err = [&](bool vllm_table) {
        std::vector<std::uint16_t> buf = src;
        void *d = nullptr, *dk = nullptr;   // q and k stay distinct: both are
        std::int32_t* dpos = nullptr;       // `__restrict__` on the kernel.
        cudaMalloc(&d, total * sizeof(std::uint16_t));
        cudaMalloc(&dk, sizeof(std::uint16_t));
        cudaMalloc(&dpos, tokens * sizeof(std::int32_t));
        cudaMemcpy(d, buf.data(), total * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dpos, positions.data(), tokens * sizeof(std::int32_t),
                   cudaMemcpyHostToDevice);
        if (vllm_table) {
            kernels::launch_rope_partial_vllm_table_bf16(
                d, dk, dpos, tokens, heads, 0, head_dim, rotary_dim, theta, nullptr);
        } else {
            kernels::launch_rope_partial_bf16(
                d, dk, dpos, tokens, heads, 0, head_dim, rotary_dim, theta, nullptr);
        }
        cudaDeviceSynchronize();
        cudaMemcpy(buf.data(), d, total * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        cudaFree(d); cudaFree(dk); cudaFree(dpos);
        double m = 0.0;
        for (int n = 0; n < tokens; ++n)
            for (int h = 0; h < heads; ++h)
                for (int j = 0; j < rotary_dim; ++j) {
                    const int i = (n * heads + h) * head_dim + j;
                    m = std::fmax(m, std::fabs(bf16_to_float(buf[i]) - ref[i]));
                }
        return m;
    };

    const double err_default = max_err(false);
    const double err_table = max_err(true);
    // One bf16 step at unit scale is ~7.8e-3 and a bf16 rotate rounds three
    // times, so 4e-2 is a loose ceiling on a correct table path. Measured on
    // the host, bf16(cos(fp32 angle)) is within 1.8e-3 of true math across
    // every lane at position 20000.
    const bool ok = err_table <= 4e-2 && err_default > 10.0 * err_table;
    if (!ok) ++g_failures;
    std::printf("[%s] %-28s positions=13000..24000  err_default=%.3e  "
                "err_vllm_table=%.3e\n",
                ok ? "ok" : "FAIL", "long-position-probe",
                err_default, err_table);
}

}  // namespace

int main() {
    // Without this, a run on a host with no CUDA device does not skip and does
    // not error -- every `cudaMalloc` fails silently, the kernels never launch,
    // and the checks compare uninitialized memory and print FAIL. A build host
    // and a genuine numeric regression then look identical in the log, which is
    // exactly the confusion this whole file exists to prevent.
    int devices = 0;
    const cudaError_t err = cudaGetDeviceCount(&devices);
    if (err != cudaSuccess || devices == 0) {
        std::printf("no CUDA device (%s) -- this test requires a GPU and did "
                    "NOT run\n", cudaGetErrorString(err));
        return 77;  // ctest's conventional "skipped"
    }

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

    // Same channel contract, vLLM's rounding structure.
    const Case vllm_cases[] = {
        {"vllm-table qwen3.5-9b", 256, 64, 8, 2, 4, 1e7f},
        {"vllm-table partial-half", 128, 64, 4, 2, 3, 1e6f},
        {"vllm-table full-rotary", 128, 128, 4, 2, 3, 1e6f},
    };
    for (const auto& c : vllm_cases) run_vllm_case(c);
    run_long_position_probe();

    if (g_failures != 0) {
        std::printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}

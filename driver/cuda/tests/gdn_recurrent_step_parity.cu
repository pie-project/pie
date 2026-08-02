// Parity guard for the GDN recurrent DECODE step kernels, against a CPU
// reference.
//
// Three kernels implement the same one-token recurrence and the launcher
// picks between them from head counts and env flags:
//
//   * `recurrent_step_batched_kernel`          — legacy, non-GQA (K_h == V_h)
//   * `recurrent_step_batched_gqa_kernel`      — legacy, GQA (V_h % K_h == 0)
//   * `recurrent_step_batched_gqa_smem_kernel` — the tuned SMEM path, which
//     the GQA launcher takes whenever V_d == K_d == 128 and the state is
//     V-last. That is the production default for every shape that reaches
//     the GQA launcher, so a divergence here is silent wrong output on a
//     real model, not a benchmark artifact.
//
// The env flags that select between them are read into function-local
// statics, so a single process cannot A/B them. This test therefore compares
// whichever kernels the current environment selects against a CPU reference
// of the recurrence, and the caller flips `PIE_QWEN35_GDN_SMEM_STEP` between
// runs. Both launchers are checked in the same run:
//
//   non-GQA launcher   — only expressible when K_h == V_h
//   GQA launcher       — every shape, SMEM path subject to the env flag
//
// The kernels accumulate in fp32 and keep the state in bf16, so the
// comparison is a tolerance rather than bit-equality. bf16 carries ~8
// mantissa bits, and a 128-term reduction over values of order 0.1 lands
// well inside 5e-2; an indexing or synchronisation bug lands orders of
// magnitude outside it.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/gated_delta_net.hpp"

namespace kernels = pie_cuda_driver::kernels;

namespace {

int g_failures = 0;

void rt_check(cudaError_t e, const char* expr, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL rt %s:%d: %s -> %s\n", __FILE__, line,
                     expr, cudaGetErrorString(e));
        std::exit(2);
    }
}
#define RT(expr) rt_check((expr), #expr, __LINE__)

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    const std::uint32_t rounded = b + 0x7fffu + ((b >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// The recurrence, in fp32 over a V-last state tile ([k][v] per head):
//
//   s   <- state * g
//   d   <- (v - Σ_k s[k][j] * k[k]) * beta
//   s'  <- s + k ⊗ d
//   out <- Σ_k s'[k][j] * q[k]
//
// This mirrors the LEGACY kernel's rounding exactly, which is the behaviour
// every implementation must reproduce: `state * g` is rounded to bf16 before
// delta is added (the legacy kernel stores it to HBM between its two phases
// and reloads it), while the kv_mem reduction consumes the unrounded value.
// The rounding point is not cosmetic — under greedy decoding it decides the
// token whenever two logits are close, so a kernel that carries full fp32
// through phase 2 is *more* accurate and still produces a different
// trajectory. Holding every kernel to this reference keeps kernel selection
// unobservable in model output.
void cpu_reference(
    const std::vector<float>& q, const std::vector<float>& k,
    const std::vector<float>& v, const std::vector<float>& g_log,
    const std::vector<float>& beta, std::vector<std::uint16_t>& state,
    const std::vector<std::int32_t>& slots, long long slot_stride,
    std::vector<float>& out, int R, int K_h, int V_h, int K_d, int V_d)
{
    const int repeat = V_h / K_h;
    for (int r = 0; r < R; ++r) {
        const int slot = slots[r];
        if (slot < 0) continue;
        for (int h = 0; h < V_h; ++h) {
            const int h_k = h / repeat;
            const std::size_t qk_off =
                (static_cast<std::size_t>(r) * K_h + h_k) * K_d;
            const std::size_t vh =
                static_cast<std::size_t>(r) * V_h + h;
            const float g = std::exp(g_log[vh]);
            const float b = beta[vh];
            std::uint16_t* st = state.data()
                + static_cast<std::size_t>(slot) * slot_stride
                + static_cast<std::size_t>(h) * K_d * V_d;

            for (int j = 0; j < V_d; ++j) {
                float kv_mem = 0.0f;
                for (int i = 0; i < K_d; ++i) {
                    const float s = bf16_to_f32(st[i * V_d + j]) * g;
                    kv_mem += s * k[qk_off + i];
                }
                const float delta = (v[vh * V_d + j] - kv_mem) * b;
                float out_v = 0.0f;
                for (int i = 0; i < K_d; ++i) {
                    const float sg =
                        bf16_to_f32(f32_to_bf16(
                            bf16_to_f32(st[i * V_d + j]) * g));
                    const float s = sg + k[qk_off + i] * delta;
                    out_v += s * q[qk_off + i];
                    st[i * V_d + j] = f32_to_bf16(s);
                }
                out[vh * V_d + j] = out_v;
            }
        }
    }
}

struct Case {
    int R;
    int K_h;
    int V_h;
};

struct Buffers {
    float *q = nullptr, *k = nullptr, *v = nullptr, *g = nullptr, *beta = nullptr;
    float* out = nullptr;
    void* state = nullptr;
    std::int32_t* slots = nullptr;
};

// Compare one launcher's result against the CPU reference. `label` names the
// launcher so a failure says which of the three kernels diverged.
void check(
    const char* label, const Case& c, int K_d, int V_d,
    const std::vector<float>& o_ref, const std::vector<std::uint16_t>& s_ref,
    const std::vector<float>& o_got, const std::vector<std::uint16_t>& s_got)
{
    double max_out = 0.0, max_state = 0.0;
    std::size_t worst_out = 0, worst_state = 0;
    for (std::size_t i = 0; i < o_ref.size(); ++i) {
        const double d = std::fabs(o_ref[i] - o_got[i]);
        if (d > max_out) { max_out = d; worst_out = i; }
    }
    for (std::size_t i = 0; i < s_ref.size(); ++i) {
        const double d =
            std::fabs(bf16_to_f32(s_ref[i]) - bf16_to_f32(s_got[i]));
        if (d > max_state) { max_state = d; worst_state = i; }
    }
    const double tol = 5e-2;
    const bool ok = (max_out <= tol) && (max_state <= tol);
    std::printf(
        "  %-4s %-10s R=%-3d K_h=%-3d V_h=%-3d repeat=%-2d "
        "max|dout|=%.6f (i=%zu) max|dstate|=%.6f (i=%zu)\n",
        ok ? "ok" : "FAIL", label, c.R, c.K_h, c.V_h, c.V_h / c.K_h,
        max_out, worst_out, max_state, worst_state);
    if (!ok) ++g_failures;
    (void)K_d;
    (void)V_d;
}

void run_case(const Case& c, int K_d, int V_d, unsigned seed) {
    const int R = c.R;
    const int K_h = c.K_h;
    const int V_h = c.V_h;
    const long long slot_stride = static_cast<long long>(V_h) * K_d * V_d;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> pos(0.0f, 1.0f);

    const std::size_t qk_n = static_cast<std::size_t>(R) * K_h * K_d;
    const std::size_t v_n = static_cast<std::size_t>(R) * V_h * V_d;
    const std::size_t gb_n = static_cast<std::size_t>(R) * V_h;
    const std::size_t st_n = static_cast<std::size_t>(R) * slot_stride;

    std::vector<float> h_q(qk_n), h_k(qk_n), h_v(v_n);
    std::vector<float> h_g(gb_n), h_beta(gb_n);
    for (auto& x : h_q) x = unit(rng);
    for (auto& x : h_k) x = unit(rng);
    for (auto& x : h_v) x = unit(rng);
    // g_log is a log-decay, so g = exp(g_log) lands in (0, 1).
    for (auto& x : h_g) x = -pos(rng);
    for (auto& x : h_beta) x = pos(rng);

    std::vector<std::uint16_t> h_state0(st_n);
    for (auto& x : h_state0) x = f32_to_bf16(unit(rng) * 0.1f);
    std::vector<std::int32_t> h_slots(R);
    for (int r = 0; r < R; ++r) h_slots[r] = r;

    std::vector<std::uint16_t> s_ref = h_state0;
    std::vector<float> o_ref(v_n, 0.0f);
    cpu_reference(h_q, h_k, h_v, h_g, h_beta, s_ref, h_slots, slot_stride,
                  o_ref, R, K_h, V_h, K_d, V_d);

    Buffers d;
    RT(cudaMalloc(&d.q, qk_n * sizeof(float)));
    RT(cudaMalloc(&d.k, qk_n * sizeof(float)));
    RT(cudaMalloc(&d.v, v_n * sizeof(float)));
    RT(cudaMalloc(&d.g, gb_n * sizeof(float)));
    RT(cudaMalloc(&d.beta, gb_n * sizeof(float)));
    RT(cudaMalloc(&d.out, v_n * sizeof(float)));
    RT(cudaMalloc(&d.state, st_n * sizeof(std::uint16_t)));
    RT(cudaMalloc(&d.slots, static_cast<std::size_t>(R) * sizeof(std::int32_t)));

    RT(cudaMemcpy(d.q, h_q.data(), qk_n * sizeof(float), cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d.k, h_k.data(), qk_n * sizeof(float), cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d.v, h_v.data(), v_n * sizeof(float), cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d.g, h_g.data(), gb_n * sizeof(float), cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d.beta, h_beta.data(), gb_n * sizeof(float),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d.slots, h_slots.data(),
                  static_cast<std::size_t>(R) * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice));

    std::vector<float> o_got(v_n);
    std::vector<std::uint16_t> s_got(st_n);

    auto readback = [&] {
        RT(cudaMemcpy(o_got.data(), d.out, v_n * sizeof(float),
                      cudaMemcpyDeviceToHost));
        RT(cudaMemcpy(s_got.data(), d.state, st_n * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost));
    };
    auto reset = [&] {
        RT(cudaMemcpy(d.state, h_state0.data(), st_n * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice));
        RT(cudaMemset(d.out, 0, v_n * sizeof(float)));
    };

    // The non-GQA launcher wants Q/K already expanded to V_h heads, which at
    // repeat=1 is exactly the compact K_h buffer. At repeat>1 it cannot
    // express the shape without a materialised expansion, so skip it.
    if (K_h == V_h) {
        reset();
        kernels::launch_recurrent_gated_delta_step_batched_state_bf16(
            d.q, d.k, d.v, d.g, d.beta, d.state, d.slots, slot_stride,
            d.out, R, V_h, K_d, V_d, nullptr);
        RT(cudaDeviceSynchronize());
        readback();
        check("non-gqa", c, K_d, V_d, o_ref, s_ref, o_got, s_got);
    }

    reset();
    kernels::launch_recurrent_gated_delta_step_batched_gqa_state_bf16(
        d.q, d.k, d.v, d.g, d.beta, d.state, d.slots, slot_stride,
        d.out, R, K_h, V_h, K_d, V_d, nullptr);
    RT(cudaDeviceSynchronize());
    readback();
    check("gqa", c, K_d, V_d, o_ref, s_ref, o_got, s_got);

    RT(cudaFree(d.q));
    RT(cudaFree(d.k));
    RT(cudaFree(d.v));
    RT(cudaFree(d.g));
    RT(cudaFree(d.beta));
    RT(cudaFree(d.out));
    RT(cudaFree(d.state));
    RT(cudaFree(d.slots));
}

}  // namespace

int main() {
    // 128 is the production decode head dim for every Qwen3.5 GDN variant,
    // and the only shape the SMEM path claims.
    const int K_d = 128;
    const int V_d = 128;

    const char* smem = std::getenv("PIE_QWEN35_GDN_SMEM_STEP");
    std::printf("PIE_QWEN35_GDN_SMEM_STEP=%s\n",
                (smem == nullptr || smem[0] == '\0') ? "(unset -> on)" : smem);

    const Case cases[] = {
        // repeat = 1 — Qwen3.5-0.8B has 16 linear key heads and 16 value heads.
        {1, 16, 16}, {2, 16, 16}, {8, 16, 16}, {64, 16, 16},
        // repeat > 1 — the grouped shapes the SMEM path was tuned on.
        {1, 16, 32}, {8, 16, 32}, {64, 16, 32},
        {1, 8, 32},  {8, 8, 32},
    };

    unsigned seed = 1234;
    for (const auto& c : cases) run_case(c, K_d, V_d, seed++);

    if (g_failures != 0) {
        std::printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nall checks ok\n");
    return 0;
}

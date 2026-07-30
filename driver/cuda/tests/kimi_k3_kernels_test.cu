// Numerical check for the Kimi-K3 kernels against a scalar fp64 reference.
//
// Each case is written straight from Moonshot's `modeling_kimi_linear.py` /
// the KDA Triton kernels, so a mismatch here is a mismatch with the reference
// implementation and not with another of our own kernels.
//
// Build and run (no CMake, no driver, no checkpoint):
//
//   nvcc -std=c++17 -arch=sm_89 -I driver/cuda/src \
//        driver/cuda/tests/kimi_k3_kernels_test.cu \
//        driver/cuda/src/kernels/kda.cu \
//        driver/cuda/src/kernels/attn_res.cu \
//        driver/cuda/src/kernels/swiglu.cu -o /tmp/k3_kernels && /tmp/k3_kernels

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernels/attn_res.hpp"
#include "kernels/kda.hpp"
#include "kernels/swiglu.hpp"

using pie_cuda_driver::kernels::launch_attn_res_blend_bf16;
using pie_cuda_driver::kernels::launch_kda_gate_beta_bf16;
using pie_cuda_driver::kernels::launch_kda_o_norm_gated_bf16;
using pie_cuda_driver::kernels::launch_kda_prefill_batched;
using pie_cuda_driver::kernels::launch_kda_recurrent_step_batched;
using pie_cuda_driver::kernels::launch_situ_bf16;

namespace {

int g_failures = 0;

#define CUDA_OK(expr)                                                        \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e),  \
                        __FILE__, __LINE__);                                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// bf16 round-trip, so the reference sees exactly the values the kernel does.
float bf16(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

template <typename T>
T* upload(const std::vector<T>& host) {
    T* dev = nullptr;
    CUDA_OK(cudaMalloc(&dev, host.size() * sizeof(T)));
    CUDA_OK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T),
                       cudaMemcpyHostToDevice));
    return dev;
}

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& x) {
    std::vector<__nv_bfloat16> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = __float2bfloat16(x[i]);
    return out;
}

void report(const char* name, double max_rel, double tol) {
    const bool ok = max_rel <= tol;
    if (!ok) ++g_failures;
    std::printf("%-28s max_rel %.3e  tol %.0e  %s\n", name, max_rel, tol,
                ok ? "ok" : "FAIL");
}

std::mt19937 rng(1234);
float uniform(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
}

double rel_error(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double denom = std::max(1e-3, std::abs(want[i]));
        worst = std::max(worst, std::abs(got[i] - want[i]) / denom);
    }
    return worst;
}

// ── SiTU ────────────────────────────────────────────────────────────
//
//   situ(gate) = beta * tanh(gate / beta) * sigmoid(gate)
//   y          = situ(gate) * linear_beta * tanh(up / linear_beta)
void test_situ() {
    constexpr int N = 4096;
    constexpr float kBeta = 4.f, kLinearBeta = 25.f;
    std::vector<float> gate(N), up(N);
    for (int i = 0; i < N; ++i) {
        gate[i] = bf16(uniform(-40.f, 40.f));
        up[i] = bf16(uniform(-60.f, 60.f));
    }
    auto* d_gate = upload(to_bf16(gate));
    auto* d_up = upload(to_bf16(up));
    __nv_bfloat16* d_y = nullptr;
    CUDA_OK(cudaMalloc(&d_y, N * sizeof(__nv_bfloat16)));

    launch_situ_bf16(d_gate, d_up, d_y, N, kBeta, kLinearBeta, nullptr);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_y(N);
    CUDA_OK(cudaMemcpy(h_y.data(), d_y, N * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost));

    std::vector<float> got(N);
    std::vector<double> want(N);
    for (int i = 0; i < N; ++i) {
        got[i] = __bfloat162float(h_y[i]);
        const double g = gate[i];
        const double situ = kBeta * std::tanh(g / kBeta) / (1.0 + std::exp(-g));
        const double u = kLinearBeta * std::tanh(up[i] / kLinearBeta);
        want[i] = situ * u;
    }
    // bf16 output carries ~3 decimal digits.
    report("situ", rel_error(got, want), 1e-2);

    CUDA_OK(cudaFree(d_gate)); CUDA_OK(cudaFree(d_up)); CUDA_OK(cudaFree(d_y));
}

// ── AttnRes blend ───────────────────────────────────────────────────
void test_attn_res() {
    constexpr int T = 7, B = 3, H = 512;
    constexpr float kEps = 1e-5f;
    std::vector<float> prefix(T * H), blocks(B * T * H), nw(H), pw(H);
    for (auto& v : prefix) v = bf16(uniform(-2.f, 2.f));
    for (auto& v : blocks) v = bf16(uniform(-2.f, 2.f));
    for (auto& v : nw) v = bf16(uniform(0.5f, 1.5f));
    for (auto& v : pw) v = bf16(uniform(-0.1f, 0.1f));

    auto* d_prefix = upload(to_bf16(prefix));
    auto* d_blocks = upload(to_bf16(blocks));
    auto* d_nw = upload(to_bf16(nw));
    auto* d_pw = upload(to_bf16(pw));
    __nv_bfloat16* d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_out, T * H * sizeof(__nv_bfloat16)));

    launch_attn_res_blend_bf16(d_prefix, d_blocks, d_nw, d_pw, d_out, T, B, H,
                               /*block_rows=*/T, kEps, nullptr);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out(T * H);
    CUDA_OK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost));

    std::vector<float> got(T * H);
    std::vector<double> want(T * H);
    for (int t = 0; t < T; ++t) {
        // v = [blocks..., prefix]; the scores read the RMS-normalised rows and
        // the output reads the raw ones.
        auto row = [&](int j, int h) -> double {
            return j < B ? blocks[((std::size_t)j * T + t) * H + h]
                         : prefix[(std::size_t)t * H + h];
        };
        std::vector<double> score(B + 1);
        for (int j = 0; j <= B; ++j) {
            double ss = 0.0;
            for (int h = 0; h < H; ++h) ss += row(j, h) * row(j, h);
            const double scale = 1.0 / std::sqrt(ss / H + kEps);
            double dot = 0.0;
            for (int h = 0; h < H; ++h) dot += row(j, h) * scale * nw[h] * pw[h];
            score[j] = dot;
        }
        double m = score[0];
        for (int j = 1; j <= B; ++j) m = std::max(m, score[j]);
        double sum = 0.0;
        for (int j = 0; j <= B; ++j) { score[j] = std::exp(score[j] - m); sum += score[j]; }
        for (int j = 0; j <= B; ++j) score[j] /= sum;
        for (int h = 0; h < H; ++h) {
            double acc = 0.0;
            for (int j = 0; j <= B; ++j) acc += score[j] * row(j, h);
            want[(std::size_t)t * H + h] = acc;
            got[(std::size_t)t * H + h] = __bfloat162float(h_out[(std::size_t)t * H + h]);
        }
    }
    report("attn_res blend", rel_error(got, want), 1e-2);

    CUDA_OK(cudaFree(d_prefix)); CUDA_OK(cudaFree(d_blocks));
    CUDA_OK(cudaFree(d_nw)); CUDA_OK(cudaFree(d_pw)); CUDA_OK(cudaFree(d_out));
}

// ── KDA gate/beta + recurrence ──────────────────────────────────────
//
// gate[t,h,d] = lower_bound * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h,d]))
// beta[t,h]   = sigmoid(raw_beta[t,h])
//
// then, per (request, head), for each token in order:
//   state[k,v] *= exp(gate[k]); delta = (v - Σ_k state[k,v]*k[k]) * beta
//   state[k,v] += k[k]*delta;   out[v] = Σ_k state[k,v]*q[k]
void test_kda() {
    constexpr int R = 2, T_PER = 5, H = 3, D = 64;
    constexpr int T = R * T_PER;
    constexpr float kLowerBound = -5.f;

    std::vector<float> raw_g(T * H * D), raw_beta(T * H), A_log(H), dt_bias(H * D);
    for (auto& v : raw_g) v = bf16(uniform(-3.f, 3.f));
    for (auto& v : raw_beta) v = bf16(uniform(-3.f, 3.f));
    for (auto& v : A_log) v = uniform(0.f, 2.f);
    for (auto& v : dt_bias) v = uniform(-1.f, 1.f);

    auto* d_raw_g = upload(to_bf16(raw_g));
    auto* d_raw_beta = upload(to_bf16(raw_beta));
    auto* d_A = upload(A_log);
    auto* d_dt = upload(dt_bias);
    float *d_gate = nullptr, *d_beta = nullptr;
    CUDA_OK(cudaMalloc(&d_gate, (std::size_t)T * H * D * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_beta, (std::size_t)T * H * sizeof(float)));

    launch_kda_gate_beta_bf16(d_raw_g, d_raw_beta, d_A, d_dt, d_gate, d_beta, T,
                              H, D, kLowerBound, nullptr);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> gate(T * H * D), beta(T * H);
    CUDA_OK(cudaMemcpy(gate.data(), d_gate, gate.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(beta.data(), d_beta, beta.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));

    {
        std::vector<double> want(gate.size());
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    const std::size_t i = ((std::size_t)t * H + h) * D + d;
                    const double a = std::exp((double)A_log[h]);
                    const double g = (double)raw_g[i] + dt_bias[(std::size_t)h * D + d];
                    want[i] = kLowerBound / (1.0 + std::exp(-a * g));
                }
        report("kda gate", rel_error(gate, want), 1e-5);

        std::vector<double> want_beta(beta.size());
        for (std::size_t i = 0; i < beta.size(); ++i)
            want_beta[i] = 1.0 / (1.0 + std::exp(-(double)raw_beta[i]));
        report("kda beta", rel_error(beta, want_beta), 1e-5);
    }

    // Prefill: q/k arrive already l2-normalised and q pre-scaled, which is
    // what the caller does, so feed the reference the same tensors.
    std::vector<float> q(T * H * D), k(T * H * D), v(T * H * D);
    for (auto& x : q) x = uniform(-1.f, 1.f);
    for (auto& x : k) x = uniform(-1.f, 1.f);
    for (auto& x : v) x = uniform(-1.f, 1.f);
    const float scale = 1.f / std::sqrt((float)D);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const std::size_t base = ((std::size_t)t * H + h) * D;
            double nq = 0.0, nk = 0.0;
            for (int d = 0; d < D; ++d) { nq += q[base+d]*q[base+d]; nk += k[base+d]*k[base+d]; }
            nq = 1.0 / std::sqrt(nq + 1e-6); nk = 1.0 / std::sqrt(nk + 1e-6);
            for (int d = 0; d < D; ++d) { q[base+d] *= (float)nq * scale; k[base+d] *= (float)nk; }
        }

    std::vector<std::int32_t> slots{0, 1};
    std::vector<std::uint32_t> indptr{0, T_PER, 2 * T_PER};
    const long long slot_stride = (long long)H * D * D;
    std::vector<float> state_init((std::size_t)2 * slot_stride, 0.f);
    for (auto& x : state_init) x = uniform(-0.05f, 0.05f);

    auto* d_q = upload(q); auto* d_k = upload(k); auto* d_v = upload(v);
    auto* d_slots = upload(slots); auto* d_indptr = upload(indptr);
    auto* d_state = upload(state_init);
    float* d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_out, (std::size_t)T * H * D * sizeof(float)));

    launch_kda_prefill_batched(d_q, d_k, d_v, d_gate, d_beta, d_state,
                               d_slots, d_indptr, slot_stride, d_out, R, H, D,
                               nullptr);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> out(T * H * D), state_out(state_init.size());
    CUDA_OK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(state_out.data(), d_state, state_out.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));

    std::vector<double> want(out.size());
    std::vector<double> ref_state(state_init.begin(), state_init.end());
    for (int r = 0; r < R; ++r)
        for (int h = 0; h < H; ++h) {
            double* st = ref_state.data() + (std::size_t)slots[r] * slot_stride +
                         (std::size_t)h * D * D;
            for (int ti = indptr[r]; ti < (int)indptr[r + 1]; ++ti) {
                const std::size_t base = ((std::size_t)ti * H + h) * D;
                for (int vi = 0; vi < D; ++vi) {
                    double mem = 0.0;
                    for (int ki = 0; ki < D; ++ki) {
                        double& s = st[(std::size_t)vi * D + ki];
                        s *= std::exp((double)gate[base + ki]);
                        mem += s * k[base + ki];
                    }
                    const double delta = ((double)v[base + vi] - mem) * beta[(std::size_t)ti * H + h];
                    double acc = 0.0;
                    for (int ki = 0; ki < D; ++ki) {
                        double& s = st[(std::size_t)vi * D + ki];
                        s += (double)k[base + ki] * delta;
                        acc += s * q[base + ki];
                    }
                    want[base + vi] = acc;
                }
            }
        }
    report("kda prefill (out)", rel_error(out, want), 1e-4);
    report("kda prefill (state)", rel_error(state_out, ref_state), 1e-4);

    // One more token per request through the decode kernel, continuing from
    // the state the prefill left behind.
    std::vector<float> dq(R * H * D), dk(R * H * D), dv(R * H * D),
        dgate(R * H * D), dbeta(R * H);
    for (auto& x : dq) x = uniform(-1.f, 1.f);
    for (auto& x : dk) x = uniform(-1.f, 1.f);
    for (auto& x : dv) x = uniform(-1.f, 1.f);
    for (auto& x : dgate) x = uniform(-4.f, -0.1f);
    for (auto& x : dbeta) x = uniform(0.f, 1.f);
    for (int r = 0; r < R; ++r)
        for (int h = 0; h < H; ++h) {
            const std::size_t base = ((std::size_t)r * H + h) * D;
            double nq = 0.0, nk = 0.0;
            for (int d = 0; d < D; ++d) { nq += dq[base+d]*dq[base+d]; nk += dk[base+d]*dk[base+d]; }
            nq = 1.0 / std::sqrt(nq + 1e-6); nk = 1.0 / std::sqrt(nk + 1e-6);
            for (int d = 0; d < D; ++d) { dq[base+d] *= (float)nq * scale; dk[base+d] *= (float)nk; }
        }
    auto* d_dq = upload(dq); auto* d_dk = upload(dk); auto* d_dv = upload(dv);
    auto* d_dgate = upload(dgate); auto* d_dbeta = upload(dbeta);
    float* d_dout = nullptr;
    CUDA_OK(cudaMalloc(&d_dout, (std::size_t)R * H * D * sizeof(float)));

    launch_kda_recurrent_step_batched(d_dq, d_dk, d_dv, d_dgate, d_dbeta,
                                      d_state, d_slots, slot_stride, d_dout, R,
                                      H, D, nullptr);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> dout(R * H * D);
    CUDA_OK(cudaMemcpy(dout.data(), d_dout, dout.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));

    std::vector<double> dwant(dout.size());
    for (int r = 0; r < R; ++r)
        for (int h = 0; h < H; ++h) {
            double* st = ref_state.data() + (std::size_t)slots[r] * slot_stride +
                         (std::size_t)h * D * D;
            const std::size_t base = ((std::size_t)r * H + h) * D;
            for (int vi = 0; vi < D; ++vi) {
                double mem = 0.0;
                for (int ki = 0; ki < D; ++ki) {
                    double& s = st[(std::size_t)vi * D + ki];
                    s *= std::exp((double)dgate[base + ki]);
                    mem += s * dk[base + ki];
                }
                const double delta = ((double)dv[base + vi] - mem) * dbeta[(std::size_t)r * H + h];
                double acc = 0.0;
                for (int ki = 0; ki < D; ++ki) {
                    double& s = st[(std::size_t)vi * D + ki];
                    s += (double)dk[base + ki] * delta;
                    acc += s * dq[base + ki];
                }
                dwant[base + vi] = acc;
            }
        }
    report("kda decode step", rel_error(dout, dwant), 1e-4);
}

// ── Gated output RMSNorm ────────────────────────────────────────────
void test_kda_o_norm() {
    constexpr int T = 5, H = 3, D = 128;
    constexpr float kEps = 1e-5f;
    std::vector<float> o(T * H * D), g(T * H * D), w(D);
    for (auto& x : o) x = uniform(-2.f, 2.f);
    for (auto& x : g) x = bf16(uniform(-3.f, 3.f));
    for (auto& x : w) x = uniform(0.5f, 1.5f);

    auto* d_o = upload(o);
    auto* d_g = upload(to_bf16(g));
    auto* d_w = upload(w);
    __nv_bfloat16* d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_out, (std::size_t)T * H * D * sizeof(__nv_bfloat16)));

    launch_kda_o_norm_gated_bf16(d_o, d_g, d_w, d_out, T, H, D, kEps, nullptr);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out(T * H * D);
    CUDA_OK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost));

    std::vector<float> got(h_out.size());
    std::vector<double> want(h_out.size());
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const std::size_t base = ((std::size_t)t * H + h) * D;
            double ss = 0.0;
            for (int d = 0; d < D; ++d) ss += (double)o[base + d] * o[base + d];
            const double sc = 1.0 / std::sqrt(ss / D + kEps);
            for (int d = 0; d < D; ++d) {
                want[base + d] = o[base + d] * sc * w[d] /
                                 (1.0 + std::exp(-(double)g[base + d]));
                got[base + d] = __bfloat162float(h_out[base + d]);
            }
        }
    report("kda o_norm gated", rel_error(got, want), 1e-2);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device; skipping\n");
        return 0;
    }
    test_situ();
    test_attn_res();
    test_kda();
    test_kda_o_norm();
    std::printf(g_failures == 0 ? "\nALL PASS\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

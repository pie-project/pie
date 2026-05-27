#include "kernels/causal_conv1d.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <span>
#include <vector>

#include <cuda_runtime.h>

#define CHK(expr)                                                            \
    do {                                                                     \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",           \
                         __FILE__, __LINE__, cudaGetErrorString(_e), #expr); \
            return false;                                                    \
        }                                                                    \
    } while (0)

namespace {

std::uint16_t f32_to_bf16(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1u;
    const std::uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

float bf16_to_f32(std::uint16_t x) {
    const std::uint32_t bits = static_cast<std::uint32_t>(x) << 16;
    float out = 0.f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<std::uint16_t> bf16_vec(std::span<const float> src) {
    std::vector<std::uint16_t> out;
    out.reserve(src.size());
    for (float x : src) {
        out.push_back(f32_to_bf16(x));
    }
    return out;
}

template <class T>
bool upload(T*& dst, std::span<const T> src) {
    CHK(cudaMalloc(&dst, src.size() * sizeof(T)));
    CHK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                   cudaMemcpyHostToDevice));
    return true;
}

template <class T>
bool download(std::vector<T>& dst, const T* src) {
    CHK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T),
                   cudaMemcpyDeviceToHost));
    return true;
}

bool nearly_equal_bf16(std::span<const std::uint16_t> a,
                       std::span<const std::uint16_t> b,
                       const char* label) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "%s size mismatch: %zu != %zu\n",
                     label, a.size(), b.size());
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float af = bf16_to_f32(a[i]);
        const float bf = bf16_to_f32(b[i]);
        if (std::fabs(af - bf) > 1e-2f) {
            std::fprintf(stderr,
                         "%s mismatch at %zu: got %.6f expected %.6f "
                         "(0x%04x vs 0x%04x)\n",
                         label, i, af, bf, a[i], b[i]);
            return false;
        }
    }
    return true;
}

bool run_decode_reference(
    const std::vector<std::uint16_t>& x,
    const std::vector<std::uint16_t>& weight,
    const std::vector<std::uint16_t>& bias,
    std::vector<std::uint16_t> state,
    int T, int C, int K,
    std::vector<std::uint16_t>& y_out,
    std::vector<std::uint16_t>& state_out) {
    std::uint16_t* d_weight = nullptr;
    std::uint16_t* d_bias = nullptr;
    std::uint16_t* d_state = nullptr;
    std::uint16_t* d_x = nullptr;
    std::uint16_t* d_y = nullptr;

    if (!upload(d_weight, std::span<const std::uint16_t>(weight))) return false;
    if (!upload(d_bias, std::span<const std::uint16_t>(bias))) return false;
    if (!upload(d_state, std::span<const std::uint16_t>(state))) return false;
    CHK(cudaMalloc(&d_y, static_cast<std::size_t>(T) * C * sizeof(std::uint16_t)));

    for (int t = 0; t < T; ++t) {
        const std::span<const std::uint16_t> x_t(
            x.data() + static_cast<std::size_t>(t) * C,
            static_cast<std::size_t>(C));
        if (!upload(d_x, x_t)) return false;
        pie_cuda_driver::kernels::launch_causal_conv1d_update_bf16(
            d_x, d_weight, d_bias, d_state,
            d_y + static_cast<std::size_t>(t) * C,
            C, K, nullptr);
        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());
        CHK(cudaFree(d_x));
        d_x = nullptr;
    }

    y_out.resize(static_cast<std::size_t>(T) * C);
    state_out.resize(static_cast<std::size_t>(K) * C);
    if (!download(y_out, d_y)) return false;
    if (!download(state_out, d_state)) return false;

    CHK(cudaFree(d_weight));
    CHK(cudaFree(d_bias));
    CHK(cudaFree(d_state));
    CHK(cudaFree(d_y));
    return true;
}

bool test_single_prefill_t1_matches_decode_with_prior_state() {
    constexpr int T = 1;
    constexpr int C = 7;
    constexpr int K = 4;

    const auto x = bf16_vec(std::vector<float>{
        -0.70f, -0.45f, -0.20f, 0.05f, 0.30f, 0.55f, 0.80f,
    });
    std::vector<float> w_f;
    std::vector<float> b_f;
    std::vector<float> state_f;
    for (int c = 0; c < C; ++c) {
        b_f.push_back(0.01f * static_cast<float>(c - 3));
        for (int k = 0; k < K; ++k) {
            w_f.push_back(0.03f * static_cast<float>((c + 1) * (k + 2)));
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            state_f.push_back(0.02f * static_cast<float>(1 + k * C + c));
        }
    }
    const auto weight = bf16_vec(w_f);
    const auto bias = bf16_vec(b_f);
    const auto state = bf16_vec(state_f);

    std::vector<std::uint16_t> ref_y, ref_state;
    if (!run_decode_reference(x, weight, bias, state, T, C, K, ref_y, ref_state)) {
        return false;
    }

    std::uint16_t* d_x = nullptr;
    std::uint16_t* d_weight = nullptr;
    std::uint16_t* d_bias = nullptr;
    std::uint16_t* d_state = nullptr;
    std::uint16_t* d_y = nullptr;
    if (!upload(d_x, std::span<const std::uint16_t>(x))) return false;
    if (!upload(d_weight, std::span<const std::uint16_t>(weight))) return false;
    if (!upload(d_bias, std::span<const std::uint16_t>(bias))) return false;
    if (!upload(d_state, std::span<const std::uint16_t>(state))) return false;
    CHK(cudaMalloc(&d_y, x.size() * sizeof(std::uint16_t)));

    pie_cuda_driver::kernels::launch_causal_conv1d_prefill_bf16(
        d_x, d_weight, d_bias, d_y, d_state, T, C, K, nullptr);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    std::vector<std::uint16_t> got_y(x.size());
    std::vector<std::uint16_t> got_state(state.size());
    if (!download(got_y, d_y)) return false;
    if (!download(got_state, d_state)) return false;

    CHK(cudaFree(d_x));
    CHK(cudaFree(d_weight));
    CHK(cudaFree(d_bias));
    CHK(cudaFree(d_state));
    CHK(cudaFree(d_y));

    return nearly_equal_bf16(got_y, ref_y, "single prefill T=1 y") &&
           nearly_equal_bf16(got_state, ref_state, "single prefill T=1 state");
}

bool test_batched_prefill_mixed_t_matches_decode_with_prior_state() {
    constexpr int R = 3;
    constexpr int C = 5;
    constexpr int K = 4;
    constexpr int SLOTS = 6;
    const std::vector<std::int32_t> slot_ids{4, 1, 5};
    const std::vector<std::uint32_t> qo{0, 1, 3, 4};
    constexpr int N = 4;

    std::vector<float> x_f;
    for (int t = 0; t < N; ++t) {
        for (int c = 0; c < C; ++c) {
            x_f.push_back(-0.35f + 0.11f * static_cast<float>(t * C + c));
        }
    }
    std::vector<float> w_f;
    std::vector<float> b_f;
    std::vector<float> state_f(static_cast<std::size_t>(SLOTS) * K * C);
    for (int c = 0; c < C; ++c) {
        b_f.push_back(0.02f * static_cast<float>(c - 2));
        for (int k = 0; k < K; ++k) {
            w_f.push_back(-0.05f + 0.015f * static_cast<float>(c * K + k));
        }
    }
    for (int s = 0; s < SLOTS; ++s) {
        for (int k = 0; k < K; ++k) {
            for (int c = 0; c < C; ++c) {
                state_f[static_cast<std::size_t>(s) * K * C + k * C + c] =
                    0.01f * static_cast<float>(100 * s + 10 * k + c);
            }
        }
    }

    const auto x = bf16_vec(x_f);
    const auto weight = bf16_vec(w_f);
    const auto bias = bf16_vec(b_f);
    const auto state = bf16_vec(state_f);

    std::vector<std::uint16_t> expected_y(static_cast<std::size_t>(N) * C);
    std::vector<std::uint16_t> expected_state = state;
    for (int r = 0; r < R; ++r) {
        const int t0 = static_cast<int>(qo[r]);
        const int t1 = static_cast<int>(qo[r + 1]);
        const int T = t1 - t0;
        const int slot = slot_ids[r];
        std::vector<std::uint16_t> req_x(
            x.begin() + static_cast<std::ptrdiff_t>(t0 * C),
            x.begin() + static_cast<std::ptrdiff_t>(t1 * C));
        std::vector<std::uint16_t> req_state(
            expected_state.begin() + static_cast<std::ptrdiff_t>(slot * K * C),
            expected_state.begin() + static_cast<std::ptrdiff_t>((slot + 1) * K * C));
        std::vector<std::uint16_t> ref_y, ref_state;
        if (!run_decode_reference(req_x, weight, bias, req_state, T, C, K,
                                  ref_y, ref_state)) {
            return false;
        }
        std::copy(ref_y.begin(), ref_y.end(),
                  expected_y.begin() + static_cast<std::ptrdiff_t>(t0 * C));
        std::copy(ref_state.begin(), ref_state.end(),
                  expected_state.begin() + static_cast<std::ptrdiff_t>(slot * K * C));
    }

    std::uint16_t* d_x = nullptr;
    std::uint16_t* d_weight = nullptr;
    std::uint16_t* d_bias = nullptr;
    std::uint16_t* d_state = nullptr;
    std::uint16_t* d_y = nullptr;
    std::int32_t* d_slots = nullptr;
    std::uint32_t* d_qo = nullptr;
    if (!upload(d_x, std::span<const std::uint16_t>(x))) return false;
    if (!upload(d_weight, std::span<const std::uint16_t>(weight))) return false;
    if (!upload(d_bias, std::span<const std::uint16_t>(bias))) return false;
    if (!upload(d_state, std::span<const std::uint16_t>(state))) return false;
    if (!upload(d_slots, std::span<const std::int32_t>(slot_ids))) return false;
    if (!upload(d_qo, std::span<const std::uint32_t>(qo))) return false;
    CHK(cudaMalloc(&d_y, x.size() * sizeof(std::uint16_t)));

    pie_cuda_driver::kernels::launch_causal_conv1d_prefill_batched_bf16(
        d_x, d_weight, d_bias, d_y, d_state, d_slots, d_qo,
        static_cast<long long>(K * C), R, C, K, nullptr);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    std::vector<std::uint16_t> got_y(x.size());
    std::vector<std::uint16_t> got_state(state.size());
    if (!download(got_y, d_y)) return false;
    if (!download(got_state, d_state)) return false;

    CHK(cudaFree(d_x));
    CHK(cudaFree(d_weight));
    CHK(cudaFree(d_bias));
    CHK(cudaFree(d_state));
    CHK(cudaFree(d_y));
    CHK(cudaFree(d_slots));
    CHK(cudaFree(d_qo));

    return nearly_equal_bf16(got_y, expected_y, "batched mixed prefill y") &&
           nearly_equal_bf16(got_state, expected_state, "batched mixed prefill state");
}

}  // namespace

int main() {
    bool ok = true;
    ok = test_single_prefill_t1_matches_decode_with_prior_state() && ok;
    ok = test_batched_prefill_mixed_t_matches_decode_with_prior_state() && ok;
    if (!ok) {
        return 1;
    }
    std::puts("test_causal_conv1d OK");
    return 0;
}

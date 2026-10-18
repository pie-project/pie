#pragma once

// What the decode-batch and routed kernels share: the 16-byte buffer loads
// and their waits, and the split-K finalize, which sums the f32 partials in
// slice order (deterministic) and rounds. SWIGLU reads a packed [gate | up]
// result and lands silu(gate)·up.

#include "prelude/mfma.hip"

namespace pie::linear {

// As the unfused `mlp_swiglu` spells it, so the two agree bit for bit.
template <int V>
struct Const {
    static constexpr int value = V;
};

// f(Const<I>, base + I) for I in [I0, N), unconditionally.
template <int N, int I>
struct Unroll {
    template <class F>
    __device__ __forceinline__ static void run(F&& f, int base) {
        if constexpr (I < N) {
            f(Const<I>{}, base + I);
            Unroll<N, I + 1>::run(f, base);
        }
    }
};

typedef int i32x4 __attribute__((ext_vector_type(4)));

// A 16-byte global read the compiler neither counts nor moves (NT: a
// streamed read, the weight, which no later read finds in cache).
template <bool NT = false>
__device__ __forceinline__ void gload16(i32x4& dst, const void* p) {
    if constexpr (NT) {
        asm volatile("global_load_dwordx4 %0, %1, off nt" : "=v"(dst) : "v"(p) : "memory");
    } else {
        asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(dst) : "v"(p) : "memory");
    }
}

// Waits until at most N global reads are pending.
template <int N>
__device__ __forceinline__ void vm_wait() {
    static_assert(N < 64, "vmcnt counts to 63");
    asm volatile("s_waitcnt vmcnt(%0)" ::"n"(N) : "memory");
}

// Ties registers a hand-issued read fills to this point: volatile asm keeps
// its order, so placed after a wait, no use of them moves above the wait.
template <int COUNT>
__device__ __forceinline__ void tie(i32x4 (&regs)[COUNT]) {
#pragma unroll
    for (int i = 0; i < COUNT; ++i) asm volatile("" : "+v"(regs[i]));
}

__device__ __forceinline__ float mgemm_silu(float x) { return x * (1.f / (1.f + __expf(-x))); }

// The split launches' sum in slice order, rounded: y[m, n] = bf16(Σ ws[s,
// m, n]), or under SWIGLU h[m, n] = silu(bf16(Σ gate)) · bf16(Σ up).
template <bool SWIGLU>
__global__ __launch_bounds__(256) void mgemm_finalize(const float* __restrict__ ws, bf16* __restrict__ y, int M,
                                                     int N, int splits) {
    const long long total = static_cast<long long>(M) * N;
    const int width = SWIGLU ? 2 * N : N;
    const long long slice = static_cast<long long>(M) * width;
    for (long long i = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 4; i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x * 4) {
        const long long m = i / N;
        const int n = static_cast<int>(i % N);
        float av[4] = {0.f, 0.f, 0.f, 0.f}, uv[4] = {0.f, 0.f, 0.f, 0.f};
        for (int s = 0; s < splits; ++s) {
            const float4 a = *reinterpret_cast<const float4*>(ws + s * slice + m * width + n);
            av[0] += a.x;
            av[1] += a.y;
            av[2] += a.z;
            av[3] += a.w;
            if constexpr (SWIGLU) {
                const float4 u = *reinterpret_cast<const float4*>(ws + s * slice + m * width + N + n);
                uv[0] += u.x;
                uv[1] += u.y;
                uv[2] += u.z;
                uv[3] += u.w;
            }
        }
        bf16 v[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            if constexpr (SWIGLU) {
                v[t] = from_f32<bf16>(mgemm_silu(to_f32(from_f32<bf16>(av[t]))) * to_f32(from_f32<bf16>(uv[t])));
            } else {
                v[t] = from_f32<bf16>(av[t]);
            }
        }
        *reinterpret_cast<uint2*>(y + i) = *reinterpret_cast<const uint2*>(v);
    }
}

}  // namespace pie::linear

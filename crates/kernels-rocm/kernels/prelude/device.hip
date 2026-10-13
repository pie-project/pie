#pragma once

// What every unit stands on: the element types the Rust side names in its
// instantiations (`::pie::bf16`, `::pie::i32`, `::pie::true_type::value`),
// the wavefront, and the reductions over it.
//
// Device text here is HIP as hiprtc compiles it: the HIP device runtime is in
// scope already, so no unit includes `hip/hip_runtime.h`.

namespace pie {

using bf16 = __bf16;
using f16 = _Float16;

using i8 = signed char;
using u8 = unsigned char;
using i16 = short;
using u16 = unsigned short;
using i32 = int;
using u32 = unsigned int;
using i64 = long long;
using u64 = unsigned long long;
using usize = decltype(sizeof(0));

static_assert(sizeof(bf16) == 2 && sizeof(f16) == 2);
static_assert(sizeof(i64) == 8 && sizeof(usize) == sizeof(void*));

struct true_type {
    static constexpr bool value = true;
};
struct false_type {
    static constexpr bool value = false;
};

template <class A, class B>
struct is_same {
    static constexpr bool value = false;
};
template <class A>
struct is_same<A, A> {
    static constexpr bool value = true;
};

// Lanes per wavefront: 64 on CDNA (gfx9), 32 on RDNA.
#if defined(__GFX9__)
constexpr int kWave = 64;
#else
constexpr int kWave = 32;
#endif

__device__ __forceinline__ int lane() { return static_cast<int>(__lane_id()); }

// A workgroup barrier that orders LDS only. `__syncthreads()` also drains
// every outstanding global load and store (vmcnt 0) before it waits, which
// stalls a loop on the operands it prefetched; this one leaves them in
// flight. Global memory a block writes and then reads across it still needs
// `__syncthreads()`.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Storage and arithmetic types meet at f32: every kernel loads, works in f32,
// and rounds once on the store (round to nearest even for bf16 and f16).
template <class T>
__device__ __forceinline__ float to_f32(T v) {
    return static_cast<float>(v);
}
template <class T>
__device__ __forceinline__ T from_f32(float v) {
    return static_cast<T>(v);
}

__device__ __forceinline__ float neg_inf() { return -__builtin_huge_valf(); }

// Reductions over the whole wavefront; every lane gets the answer.
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
    return v;
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, kWave));
    return v;
}

// Reductions over the block, for a block of whole wavefronts. `smem` holds
// one float per wavefront; every thread gets the answer.
__device__ __forceinline__ float block_sum(float v, float* smem) {
    const int waves = static_cast<int>(blockDim.x) / kWave;
    const int w = static_cast<int>(threadIdx.x) / kWave;
    v = wave_sum(v);
    if (lane() == 0) smem[w] = v;
    __syncthreads();
    float total = 0.f;
    for (int i = 0; i < waves; ++i) total += smem[i];
    __syncthreads();
    return total;
}

__device__ __forceinline__ float block_max(float v, float* smem) {
    const int waves = static_cast<int>(blockDim.x) / kWave;
    const int w = static_cast<int>(threadIdx.x) / kWave;
    v = wave_max(v);
    if (lane() == 0) smem[w] = v;
    __syncthreads();
    float best = neg_inf();
    for (int i = 0; i < waves; ++i) best = fmaxf(best, smem[i]);
    __syncthreads();
    return best;
}

// The live window a captured launch reads its extent from: the engine pads
// a launch to a bucket of rows and stages the run's true extent in four
// words, {rows, row_offset, lanes, lane_offset}. Absent (null), the launch's
// own extent is the true one and it starts at row 0.
struct Window {
    const u32* w;

    // Whether row `row` of a launch over `per` rows per stated row is live.
    __device__ __forceinline__ bool live(long long row, int per = 1) const {
        return w == nullptr || row / per < static_cast<long long>(w[0]);
    }
    // The row of the plane the launch's row `row` lands on.
    __device__ __forceinline__ long long plane(long long row, int per = 1) const {
        return w == nullptr ? row : row + static_cast<long long>(w[1]) * per;
    }
    // The launch's live lane count, capped at `stated`.
    __device__ __forceinline__ int lanes(int stated) const {
        return w == nullptr ? stated : min(stated, static_cast<int>(w[2]));
    }
};

// Eight 2-byte elements moved as one 16-byte access, when the row allows it.
template <class T>
struct alignas(16) Pack8 {
    static_assert(sizeof(T) == 2, "a pack is eight 2-byte elements");
    T v[8];
};

// Sixteen opaque bytes: the unit a row copy moves when both ends allow it.
struct alignas(16) b128 {
    u32 w[4];
};

__device__ __forceinline__ bool aligned16(const void* p) {
    return (reinterpret_cast<usize>(p) & 15u) == 0;
}

}  // namespace pie

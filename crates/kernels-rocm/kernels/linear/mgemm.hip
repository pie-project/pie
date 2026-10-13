#pragma once

// y[m, n] = act[m, k] · w[n, k]ᵀ for a decode batch of 1..256 rows: the
// weight streams once from memory into registers, the activation is staged
// through LDS a K-chunk at a time and shared by the block's four waves, and
// the matrix cores multiply. A wave owns NBW 16-row blocks of the weight and
// every row of the batch, so the weight is read once per launch and the
// activation (small, L2-resident) once per block.
//
// SPLIT slices K across blockIdx.y; each slice stores an f32 partial and
// the finalize kernel sums them in slice order (deterministic) and rounds. SWIGLU reads a packed [gate | up]
// bank of 2·N rows and lands silu(gate)·up: waves 0-1 take the gate rows,
// waves 2-3 the matching up rows, and the two meet in LDS at the end.

#include "prelude/mfma.hip"

namespace pie::linear {

constexpr int kMgemmWaves = 4;
constexpr int kMgemmPad = 8;

template <int MB, int KC>
struct MgemmLds {
    static constexpr int kRows = 16 * MB;
    static constexpr int kStride = KC + kMgemmPad;
    static constexpr int kStage = kRows * kStride;
};

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

// MB: the batch in 16-row blocks (M <= 16·MB). NBW: weight blocks a wave.
// KC: K a stage, a multiple of 32. DEPTH: stages of weight in flight ahead
// of the one multiplied. SPLIT: K sliced over blockIdx.y.
template <int MB, int NBW, int KC, int DEPTH, bool SPLIT, bool SWIGLU>
__global__ __launch_bounds__(kWave* kMgemmWaves) void mgemm_bf16(
    const bf16* __restrict__ w, const bf16* __restrict__ act, bf16* __restrict__ y,
    float* __restrict__ ws, int M, int N, int K, int chunks_per_split) {
    using L = MgemmLds<MB, KC>;
    extern __shared__ __align__(16) unsigned char mgemm_smem[];
    bf16* stage = reinterpret_cast<bf16*>(mgemm_smem);

    const int l = lane(), wave = threadIdx.y;
    const int r16 = l % 16, g = l / 16;
    // This wave's first weight row.
    int n0;
    if constexpr (SWIGLU) {
        const int half = wave / 2;
        n0 = half * N + blockIdx.x * (2 * NBW * 16) + (wave % 2) * (NBW * 16);
    } else {
        n0 = (blockIdx.x * kMgemmWaves + wave) * (NBW * 16);
    }
    const int rows_w = SWIGLU ? 2 * N : N;

    const int chunks = K / KC;
    const int c_begin = blockIdx.y * chunks_per_split;
    const int c_end = min(chunks, c_begin + chunks_per_split);
    if (c_begin >= c_end) return;

    constexpr int KP = KC / 32;  // 16-byte weight reads a block a stage
    constexpr int R = DEPTH + 1;  // register sets, a ring
    static_assert(DEPTH >= 2, "the activation lands a stage before its use, DEPTH stages after its load");

    // The activation stage: 16·MB rows × KC, as 16-byte packs over the block.
    constexpr int kPacks = L::kRows * KC / 8;
    constexpr int kThreads = kWave * kMgemmWaves;
    constexpr int NA = (kPacks + kThreads - 1) / kThreads;  // activation reads a lane a stage
    constexpr int NW = NBW * KP;                            // weight reads a lane a stage
    const int tid = wave * kWave + l;

    i32x4 wreg[R][NW];
    i32x4 areg[R][NA];
    mfma_cd acc[NBW][MB];
#pragma unroll
    for (int nb = 0; nb < NBW; ++nb)
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) acc[nb][mb] = mfma_zero();

    // Every global read of the loop is issued here by hand and waited by
    // hand: the compiler, counting them, sinks each read to its use and
    // waits it out there along with every read behind it, which drains the
    // weight in flight. A stage issues the activation for DEPTH stages on,
    // then the weight for DEPTH stages on; the count waits in issue order,
    // so the activation that lands next is always the oldest read pending
    // short of DEPTH - 1 stages of both and one stage of weight. Reads are
    // unconditional (a row past the bank reads the last row, a chunk past
    // the slice rereads the slice's last) so the counts hold to the end.
    auto issue = [&](auto slot, int c) {
        constexpr int S = decltype(slot)::value;
        const int cc = min(c, c_end - 1);
#pragma unroll
        for (int i = 0; i < NA; ++i) {
            const int pack = min(tid + i * kThreads, kPacks - 1);
            const int row = min(pack / (KC / 8), M - 1), col = (pack % (KC / 8)) * 8;
            gload16(areg[S][i], act + static_cast<long long>(row) * K + cc * KC + col);
        }
#pragma unroll
        for (int nb = 0; nb < NBW; ++nb) {
            const int row = min(n0 + nb * 16 + r16, rows_w - 1);
            // Lane group g reads KP·16 contiguous bytes, so a row's stage is
            // whole cache lines read back to back.
            const bf16* p = w + static_cast<long long>(row) * K + cc * KC + 8 * KP * g;
#pragma unroll
            for (int kp = 0; kp < KP; ++kp) gload16(wreg[S][nb * KP + kp], p + kp * 8);
        }
    };
    // Rows past M hold a copy of the last row, a pack past the stage
    // rewrites the last pack with its own value: neither reaches a store. A
    // chunk past the slice lands zeros, so its stage adds nothing.
    auto store_act = [&](auto slot, int buf, int c) {
        constexpr int S = decltype(slot)::value;
        bf16* st = stage + buf * L::kStage;
        const bool live = c < c_end;
#pragma unroll
        for (int i = 0; i < NA; ++i) {
            const int pack = min(tid + i * kThreads, kPacks - 1);
            const int row = pack / (KC / 8), col = (pack % (KC / 8)) * 8;
            *reinterpret_cast<i32x4*>(st + row * L::kStride + col) = live ? areg[S][i] : i32x4{0, 0, 0, 0};
        }
    };

    Unroll<DEPTH, 0>::run([&](auto slot, int c) { issue(slot, c); }, c_begin);
    // The first chunk's activation: younger than it are its own weight and
    // the DEPTH - 1 stages behind.
    vm_wait<NW + (DEPTH - 1) * (NA + NW)>();
    tie(areg[0]);

    store_act(Const<0>{}, 0, c_begin);
    lds_barrier();

    auto stage_at = [&](auto cur, int c) {
        constexpr int C = decltype(cur)::value;
        constexpr int NEXT = (C + 1) % R;
        const int i = c - c_begin;
        // Chunk c + 1's activation, and so (older still) chunk c's weight.
        vm_wait<NW + (DEPTH - 2) * (NA + NW)>();
        tie(areg[NEXT]);
        tie(wreg[C]);

        store_act(Const<NEXT>{}, (i + 1) & 1, c + 1);
        issue(Const<(C + DEPTH) % R>{}, c + DEPTH);
        const bf16* st = stage + (i & 1) * L::kStage;
#pragma unroll
        for (int kp = 0; kp < KP; ++kp) {
            mfma_ab2 b[MB];
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) b[mb] = load8(st + (mb * 16 + r16) * L::kStride + 8 * (KP * g + kp));
#pragma unroll
            for (int mb = 0; mb < MB; ++mb)
#pragma unroll
                for (int nb = 0; nb < NBW; ++nb)
                    acc[nb][mb] = mfma16x2(__builtin_bit_cast(mfma_ab2, wreg[C][nb * KP + kp]), b[mb], acc[nb][mb]);
        }
        lds_barrier();
    };
    const int rings = (c_end - c_begin + R - 1) / R;
    for (int i = 0; i < rings; ++i) Unroll<R, 0>::run(stage_at, c_begin + i * R);
    // The reads past the end land before their registers are given up.
    vm_wait<0>();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        tie(wreg[r]);
        tie(areg[r]);
    }


    if constexpr (SPLIT) {
        // This slice's partial, ws[slice, m, row] (rows of the whole bank,
        // 2·N wide under SWIGLU); the finalize sums the slices in order.
        float* part = ws + static_cast<long long>(blockIdx.y) * M * rows_w;
#pragma unroll
        for (int nb = 0; nb < NBW; ++nb)
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                const int m = mb * 16 + r16;
                const int row = n0 + nb * 16 + 4 * g;
                if (m >= M || row >= rows_w) continue;
                *reinterpret_cast<float4*>(part + static_cast<long long>(m) * rows_w + row) =
                    float4{acc[nb][mb][0], acc[nb][mb][1], acc[nb][mb][2], acc[nb][mb][3]};
            }
    } else if constexpr (SWIGLU) {
        // The up waves hand their cells to the gate waves lane for lane.
        float* xch = reinterpret_cast<float*>(mgemm_smem);
#pragma unroll
        for (int nb = 0; nb < NBW; ++nb) {
            if (wave >= 2) {
#pragma unroll
                for (int mb = 0; mb < MB; ++mb)
                    *reinterpret_cast<mfma_cd*>(xch + (((wave - 2) * MB + mb) * kWave + l) * 4) = acc[nb][mb];
            }
            lds_barrier();
            if (wave < 2) {
#pragma unroll
                for (int mb = 0; mb < MB; ++mb) {
                    const mfma_cd up = *reinterpret_cast<const mfma_cd*>(xch + ((wave * MB + mb) * kWave + l) * 4);
                    const int m = mb * 16 + r16;
                    const int col = n0 + nb * 16 + 4 * g;
                    if (m < M && col < N) {
                        bf16 v[4];
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            // The gate and up products round to bf16 first, as
                            // the unfused projection lands them.
                            const float gt = to_f32(from_f32<bf16>(acc[nb][mb][t]));
                            const float u = to_f32(from_f32<bf16>(up[t]));
                            v[t] = from_f32<bf16>(mgemm_silu(gt) * u);
                        }
                        *reinterpret_cast<uint2*>(y + static_cast<long long>(m) * N + col) =
                            *reinterpret_cast<const uint2*>(v);
                    }
                }
            }
            lds_barrier();
        }
    } else {
#pragma unroll
        for (int nb = 0; nb < NBW; ++nb)
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                const int m = mb * 16 + r16;
                const int col = n0 + nb * 16 + 4 * g;
                if (m >= M || col >= N) continue;
                bf16 v[4];
#pragma unroll
                for (int t = 0; t < 4; ++t) v[t] = from_f32<bf16>(acc[nb][mb][t]);
                *reinterpret_cast<uint2*>(y + static_cast<long long>(m) * N + col) = *reinterpret_cast<const uint2*>(v);
            }
    }
}

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

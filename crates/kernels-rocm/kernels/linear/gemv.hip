#pragma once

// out[n] = act · weight[n] (+ beta·out[n]) (+ bias[n]): the one-row
// projection a decode step runs. Weight rows are K bf16, K a multiple of 8,
// loaded 16 bytes per lane; f32 accumulation.

#include "prelude/device.hip"

namespace pie::linear {

constexpr int kGemvUnroll = 4;

// The dot product of `vectors` 8-element packs starting at lane `first`,
// stepping `stride` packs: this lane's share of one row.
__device__ __forceinline__ float dot_packs(const Pack8<bf16>* __restrict__ w,
                                           const Pack8<bf16>* __restrict__ x, int vectors,
                                           int first, int stride) {
    float acc = 0.f;
    int i = first;
    for (; i + stride * (kGemvUnroll - 1) < vectors; i += stride * kGemvUnroll) {
        Pack8<bf16> wv[kGemvUnroll], xv[kGemvUnroll];
#pragma unroll
        for (int u = 0; u < kGemvUnroll; ++u) {
            wv[u] = w[i + stride * u];
            xv[u] = x[i + stride * u];
        }
#pragma unroll
        for (int u = 0; u < kGemvUnroll; ++u)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc += to_f32(wv[u].v[j]) * to_f32(xv[u].v[j]);
    }
    for (; i < vectors; i += stride) {
        const Pack8<bf16> wv = w[i], xv = x[i];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += to_f32(wv.v[j]) * to_f32(xv.v[j]);
    }
    return acc;
}

__device__ __forceinline__ void gemv_store(bf16* out, const bf16* bias, int row, float acc, float beta) {
    if (beta != 0.f) acc += beta * to_f32(out[row]);
    float v = to_f32(from_f32<bf16>(acc));
    if (bias != nullptr) v += to_f32(bias[row]);
    out[row] = from_f32<bf16>(v);
}

// Block (kWave, WAVES). SPLIT = false: each wavefront owns a row, WAVES rows
// a block. SPLIT = true: the block owns a row and its WAVES wavefronts split
// K, for a short N that would not fill the device a wavefront per row.
template <int WAVES, bool SPLIT>
__global__ __launch_bounds__(kWave* WAVES) void gemv_bf16(const bf16* __restrict__ weight,
                                                          const bf16* __restrict__ act,
                                                          const bf16* __restrict__ bias,
                                                          bf16* __restrict__ out, int N, int K,
                                                          float beta) {
    const int wave = threadIdx.y;
    const int row = SPLIT ? blockIdx.x : blockIdx.x * WAVES + wave;
    if (row >= N) return;
    const auto* w = reinterpret_cast<const Pack8<bf16>*>(weight + static_cast<long long>(row) * K);
    const auto* x = reinterpret_cast<const Pack8<bf16>*>(act);
    const int vectors = K / 8;
    if constexpr (SPLIT) {
        __shared__ float partial[WAVES];
        const float acc = wave_sum(dot_packs(w, x, vectors, wave * kWave + lane(), WAVES * kWave));
        if (lane() == 0) partial[wave] = acc;
        __syncthreads();
        if (wave == 0 && lane() == 0) {
            float total = 0.f;
#pragma unroll
            for (int i = 0; i < WAVES; ++i) total += partial[i];
            gemv_store(out, bias, row, total, beta);
        }
    } else {
        const float acc = wave_sum(dot_packs(w, x, vectors, lane(), kWave));
        if (lane() == 0) gemv_store(out, bias, row, acc, beta);
    }
}

// ------------------------------------------------------------- skinny --

// y[r][n] = act[r] · weight[n] for ROWS ≤ 8 activation rows: the decode
// projection, bound by streaming the weight once. The activation sits in LDS
// for the block's life; the weight streams past it in non-temporal 16-byte
// reads, UNRL deep, YT rows a wavefront. The products run on the 4 x 4 x 4
// matrix core: with a lane's activation chunk as A's row and its weight
// chunk as B's column, block b's diagonal D[i][i] (lane 4b+i, register i) is
// lane 4b+i's own dot of four, and the off-diagonal cells are discarded.
// Persistent: gridDim.x blocks (one a CU) of `waves` wavefronts walk the
// weight rows.
typedef short skinny_ab __attribute__((ext_vector_type(4)));
typedef float skinny_cd __attribute__((ext_vector_type(4)));
typedef int skinny_v4 __attribute__((ext_vector_type(4)));

template <int CTRL>
__device__ __forceinline__ float dpp_add(float acc, float v) {
    return acc + __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xf, 0xf, true));
}

// The sum of the lanes' diagonal cells, landed in lane 63.
__device__ __forceinline__ float skinny_reduce(skinny_cd d) {
    float a = d[0];
    a = dpp_add<0x101>(a, d[1]);  // row_shl:1
    a = dpp_add<0x102>(a, d[2]);  // row_shl:2
    a = dpp_add<0x103>(a, d[3]);  // row_shl:3  → lane 4b holds block b
    a = dpp_add<0x104>(a, a);     // row_shl:4
    a = dpp_add<0x108>(a, a);     // row_shl:8  → lane 16r holds row r
    a = __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, a), 0x11f, 0xf, 0xf, true));  // row_shr:15
    a = dpp_add<0x142>(a, a);     // row_bcast:15
    a = dpp_add<0x143>(a, a);     // row_bcast:31
    return a;
}

// The skinny kernel's epilogue. SWIGLU and GEGLU read a packed [gate | up]
// bank of 2N rows, output column n being act(g)·u of rows n and N + n, each
// rounded to bf16 first as the unfused projection would store them;
// SOFTCAP lands cap·tanh(v / cap) of the rounded product.
constexpr int kSkinnyStore = 0;
constexpr int kSkinnySwiglu = 1;
constexpr int kSkinnyGeglu = 2;
constexpr int kSkinnySoftcap = 3;

template <int ROWS, int YT, int UNRL, int EPI = kSkinnyStore>
__global__ __launch_bounds__(1024) void skinny_bf16(const bf16* __restrict__ weight, const bf16* __restrict__ act,
                                                    bf16* __restrict__ out, int N, int K, int waves, float cap) {
    constexpr bool SWIGLU = EPI == kSkinnySwiglu || EPI == kSkinnyGeglu;  // two weight rows a column
    constexpr int WR = SWIGLU ? 2 * YT : YT;  // weight rows a wavefront streams
    extern __shared__ b128 act_s[];
    const int wave = threadIdx.y, l = lane();
    const int packs = K / 8;
    for (int i = wave * kWave + l; i < ROWS * packs; i += blockDim.y * kWave)
        act_s[i] = reinterpret_cast<const b128*>(act)[i];
    __syncthreads();
    if (wave >= waves) return;
    constexpr int STEP = kWave * UNRL;  // packs a wavefront walks a round
    for (int n0 = (blockIdx.x * waves + wave) * YT; n0 < N; n0 += gridDim.x * waves * YT) {
        skinny_cd acc[ROWS][WR];
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int y = 0; y < WR; ++y) acc[r][y] = skinny_cd{0.f, 0.f, 0.f, 0.f};
        const skinny_v4* wr[WR];
#pragma unroll
        for (int y = 0; y < WR; ++y) {
            const int row = min(n0 + y % YT, N - 1) + (y >= YT ? N : 0);
            wr[y] = reinterpret_cast<const skinny_v4*>(weight + static_cast<long long>(row) * K);
        }
        for (int p0 = 0; p0 < packs; p0 += STEP) {
            skinny_v4 wv[WR][UNRL];
            b128 xv[ROWS][UNRL];
#pragma unroll
            for (int u = 0; u < UNRL; ++u) {
                const int p = min(p0 + u * kWave + l, packs - 1);
#pragma unroll
                for (int y = 0; y < WR; ++y) wv[y][u] = __builtin_nontemporal_load(wr[y] + p);
            }
#pragma unroll
            for (int u = 0; u < UNRL; ++u) {
                const int p = p0 + u * kWave + l;
#pragma unroll
                for (int r = 0; r < ROWS; ++r) xv[r][u] = p < packs ? act_s[r * packs + p] : b128{};
            }
#pragma unroll
            for (int u = 0; u < UNRL; ++u)
#pragma unroll
                for (int r = 0; r < ROWS; ++r)
#pragma unroll
                    for (int y = 0; y < WR; ++y) {
                        const skinny_ab* x4 = reinterpret_cast<const skinny_ab*>(&xv[r][u]);
                        const skinny_ab* w4 = reinterpret_cast<const skinny_ab*>(&wv[y][u]);
                        acc[r][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(x4[0], w4[0], acc[r][y], 0, 0, 0);
                        acc[r][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(x4[1], w4[1], acc[r][y], 0, 0, 0);
                    }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int y = 0; y < YT; ++y) {
                float v = skinny_reduce(acc[r][y]);
                if constexpr (SWIGLU) {
                    const float g = to_f32(from_f32<bf16>(v));
                    const float u = to_f32(from_f32<bf16>(skinny_reduce(acc[r][YT + y])));
                    if constexpr (EPI == kSkinnyGeglu) {
                        v = 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) * u;
                    } else {
                        v = g / (1.f + __expf(-g)) * u;
                    }
                } else if constexpr (EPI == kSkinnySoftcap) {
                    v = cap * tanhf(to_f32(from_f32<bf16>(v)) * (1.f / cap));
                }
                if (l == kWave - 1 && n0 + y < N) out[static_cast<long long>(r) * N + n0 + y] = from_f32<bf16>(v);
            }
    }
}

}  // namespace pie::linear

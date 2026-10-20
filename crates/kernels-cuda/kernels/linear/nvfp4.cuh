#pragma once

// NVFP4 weights read at bf16: e2m1 codes two to a byte (low nibble first),
// one e4m3 scale per 16 codes, one f32 tensor scale per row. A code times its
// e4m3 scale is exactly a bf16, so the products below round nothing until the
// accumulator; the tensor scale multiplies a row's sum once, in f32.

#include "linear/quant.cuh"

namespace pie::linear {

__device__ __forceinline__ float nvfp4_scale(u8 b) {
    const unsigned e = (b >> 3) & 0xFu;
    const unsigned m = b & 0x7u;
    const float v = e != 0u ? __int_as_float(static_cast<int>(((e + 120u) << 23) | (m << 20)))
                            : static_cast<float>(m) * 0.001953125f;
    return (b & 0x80u) != 0u ? -v : v;
}

template <class T>
__global__ void dequant_nvfp4(
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const float* __restrict__ tensor_scales,
    T* __restrict__ out,
    int rows,
    int k)
{
    const long long pairs = static_cast<long long>(rows) * (k / 2);
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= pairs) return;
    const int row = static_cast<int>(at / (k / 2));
    const int col = static_cast<int>(at % (k / 2)) * 2;
    const float gain = nvfp4_scale(scales[static_cast<long long>(row) * (k / 16) + col / 16])
        * tensor_scales[row];
    const u8 b = codes[at];
    out[static_cast<long long>(row) * k + col] = Elem<T>::from_f32(kFp4Lut[b & 0xF] * gain);
    out[static_cast<long long>(row) * k + col + 1] = Elem<T>::from_f32(kFp4Lut[b >> 4] * gain);
}

// One block a route and `kRowsT` rows a warp, the route's expert read once a
// route: the decode arm. A null `routes` reads one expert, route `r` being
// activation row `r`, which is the dense product. With kLean (A100's
// profile) one 32-code chunk a row in flight a lane and the codes widened by
// integer ops as `moe_matmul_select_mxfp4`'s lean arm widens them.
template <class T, int kRowsT, bool kLean = false>
__global__ void moe_matmul_select_nvfp4(
    const T* __restrict__ act,
    const i32* __restrict__ routes,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const float* __restrict__ tensor_scales,
    T* __restrict__ out,
    int top_k,
    int act_div,
    int n,
    int k,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    const int route = blockIdx.x;

    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;

    const int plane_route = win != nullptr ? route + static_cast<int>(win[1]) * top_k : route;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;
    const int expert = routes != nullptr ? routes[plane_route] : 0;
    T* y = out + static_cast<long long>(plane_route) * n;
    if (expert < 0) {
        if (lane_id == 0) {
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                if (row0 + r < n) y[row0 + r] = Elem<T>::from_f32(0.f);
        }
        return;
    }

    moe_note_group(group_hits);

    // A 16-byte load is 32 codes, two groups of a row.
    const int chunks = k / 32;
    const int groups = k / 16;

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const float* tensor_at = tensor_scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        tensor_at = reinterpret_cast<const float*>(seat.biases);
    }

    const uint4* wq = reinterpret_cast<const uint4*>(
        codes_at + static_cast<long long>(expert) * n * (k / 2));
    const unsigned short* s2 = reinterpret_cast<const unsigned short*>(
        scales_at + static_cast<long long>(expert) * n * groups);
    const float* g = tensor_at + static_cast<long long>(expert) * n;

    const T* x = act + static_cast<long long>(plane_route / act_div) * k;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    constexpr int kU = kLean ? 1 : 2;
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int c0 = lane_id; c0 < chunks; c0 += 32 * kU) {
        uint4 ww[kU][kRows];
        unsigned short sc[kU][kRows];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int c = min(c0 + 32 * u, chunks - 1);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const long long at = static_cast<long long>(row_of[r]) * chunks + c;
                ww[u][r] = wq[at];
                sc[u][r] = s2[at];
            }
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int c = c0 + 32 * u;
            if (c >= chunks) break;
            float part[2][kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[0][r] = part[1][r] = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 xq = x4[c * 4 + q];
                const T* xe = reinterpret_cast<const T*>(&xq);
                float xv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) xv[j] = Elem<T>::to_f32(xe[j]) * (kLean ? 0x1p100f : 1.f);
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    if constexpr (kLean) {
                        const unsigned wd = (&ww[u][r].x)[q];
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            // Codes 3 - p and 7 - p, 2^-126 times their value.
                            const unsigned t = wd << (4 * p);
                            const unsigned b2 = (t & 0x80008000u) | ((t >> 6) & 0x01C001C0u);
                            part[q >> 1][r] = fmaf(__uint_as_float(b2 << 16), xv[3 - p], part[q >> 1][r]);
                            part[q >> 1][r] = fmaf(__uint_as_float(b2 & 0xFFFF0000u), xv[7 - p], part[q >> 1][r]);
                        }
                    } else {
                        __half2 qd[4];
                        mxfp4_unpack8((&ww[u][r].x)[q], qd);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float2 f = __half22float2(qd[j]);
                            part[q >> 1][r] = fmaf(f.x, xv[2 * j], part[q >> 1][r]);
                            part[q >> 1][r] = fmaf(f.y, xv[2 * j + 1], part[q >> 1][r]);
                        }
                    }
                }
            }
            // The lean arm's products are 2^-26 short: the scale takes it back.
            constexpr float kBack = kLean ? 0x1p26f : 1.f;
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                acc[r] = fmaf(part[0][r], nvfp4_scale(static_cast<u8>(sc[u][r] & 0xFFu)) * kBack, acc[r]);
                acc[r] = fmaf(part[1][r], nvfp4_scale(static_cast<u8>(sc[u][r] >> 8)) * kBack, acc[r]);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row >= n) break;
            y[row] = Elem<T>::from_f32(acc[r] * g[row]);
        }
    }
}

// The grouped arm: one block a (expert, slice of up to 32 of its routes, 128
// rows) work item from `moe_route_order`, the expert's tile decoded to bf16
// in shared memory once and multiplied by mma.sync against every route of
// the slice. K is whole 128-code tiles.
__global__ void moe_matmul_select_nvfp4_mma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const float* __restrict__ tensor_scales,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits)
{
    constexpr int kTileN = 128;
    constexpr int kTileK = 128;
    constexpr int kBatch = 16;
    constexpr int kBatches = 2;
    constexpr int kRoutes = kBatch * kBatches;
    constexpr int kLd = kTileK + 8;
    constexpr int kWarps = kTileN / 16;
    constexpr int kChunkCodes = 32;
    constexpr int kChunksPerTileK = kTileK / kChunkCodes;
    constexpr int kChunksPerThread = kTileN * kChunksPerTileK / 256;
    constexpr int kVec = 8;
    constexpr int kVecsPerTileK = kTileK / kVec;
    constexpr int kVecsPerThread = kRoutes * kVecsPerTileK / 256;
    static_assert(kRoutes * kLd * 2 >= kWarps * 16 * 16 * 4, "the C tiles fit where the activations were");
    static_assert(kTileN * kChunksPerTileK % 256 == 0 && kRoutes * kVecsPerTileK % 256 == 0, "even split");

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item / 65536;
    const int slice = item % 65536;
    const int row0 = blockIdx.y * kTileN;
    if (row0 >= n) return;
    const int begin = offsets[expert] + slice * kRoutes;
    const int end = min(offsets[expert + 1], begin + kRoutes);
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));

    const int chunks = k / kChunkCodes;
    const int groups = k / 16;
    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const float* tensor_at = tensor_scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        tensor_at = reinterpret_cast<const float*>(seat.biases);
    }
    const uint4* wq = reinterpret_cast<const uint4*>(
        codes_at + static_cast<long long>(expert) * n * (k / 2));
    const unsigned short* s2 = reinterpret_cast<const unsigned short*>(
        scales_at + static_cast<long long>(expert) * n * groups);
    const float* g = tensor_at + static_cast<long long>(expert) * n;

    __shared__ __align__(32) bf16 w_tile[kTileN][kLd];
    __shared__ __align__(32) bf16 x_tile[kRoutes][kLd];
    float* c_tile = reinterpret_cast<float*>(&x_tile[0][0]);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    struct Staged {
        uint4 chunk[kChunksPerThread];
        unsigned short scale[kChunksPerThread];
        bool live[kChunksPerThread];
        uint4 vec[kVecsPerThread];
    };

    const int group = end - begin;
    const int batches = (group + kBatch - 1) / kBatch;

    float acc[kBatches][2][4];
#pragma unroll
    for (int b = 0; b < kBatches; ++b)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[b][t][e] = 0.f;

    const bf16* vec_src[kVecsPerThread];
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
        const int t = (tid + i * 256) / kVecsPerTileK;
        vec_src[i] = nullptr;
        if (t < group) {
            const int plane_route = order[begin + t];
            vec_src[i] = act + static_cast<long long>(plane_route / act_div) * k;
        }
    }

    auto fetch = [&](int k0, Staged& st) {
#pragma unroll
        for (int i = 0; i < kChunksPerThread; ++i) {
            const int idx = tid + i * 256;
            const int r = idx / kChunksPerTileK;
            const int q = idx % kChunksPerTileK;
            const int rr = row0 + r;
            st.live[i] = rr < n;
            if (st.live[i]) {
                const long long at = static_cast<long long>(rr) * chunks + k0 / kChunkCodes + q;
                st.chunk[i] = pie_ld_line(wq + at);
                st.scale[i] = s2[at];
            }
        }
#pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int v = (tid + i * 256) % kVecsPerTileK;
            st.vec[i] = make_uint4(0u, 0u, 0u, 0u);
            if (vec_src[i] != nullptr) {
                st.vec[i] = *reinterpret_cast<const uint4*>(vec_src[i] + k0 + v * kVec);
            }
        }
    };

    auto store = [&](const Staged& st) {
#pragma unroll
        for (int i = 0; i < kChunksPerThread; ++i) {
            const int idx = tid + i * 256;
            const int r = idx / kChunksPerTileK;
            const int q = idx % kChunksPerTileK;
            unsigned* dst = reinterpret_cast<unsigned*>(&w_tile[r][q * kChunkCodes]);
            if (st.live[i]) {
                const float lo = nvfp4_scale(static_cast<u8>(st.scale[i] & 0xFFu));
                const float hi = nvfp4_scale(static_cast<u8>(st.scale[i] >> 8));
                const unsigned words[4] = {st.chunk[i].x, st.chunk[i].y, st.chunk[i].z, st.chunk[i].w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const float sv = w < 2 ? lo : hi;
                    __half2 qd[4];
                    mxfp4_unpack8(words[w], qd);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(qd[j]);
                        dst[w * 4 + j] = pack_bf16x2(f.x * sv, f.y * sv);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < kChunkCodes / 2; ++j) dst[j] = 0u;
            }
        }
#pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int idx = tid + i * 256;
            const int t = idx / kVecsPerTileK;
            const int v = idx % kVecsPerTileK;
            *reinterpret_cast<uint4*>(&x_tile[t][v * kVec]) = st.vec[i];
        }
    };

    Staged staged;
    fetch(0, staged);
    for (int k0 = 0; k0 < k; k0 += kTileK) {
        store(staged);
        __syncthreads();
        if (k0 + kTileK < k) fetch(k0 + kTileK, staged);
#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
            unsigned bfrag[4];
            pie_ldmatrix_x4(bfrag, &w_tile[warp * 16 + (lane & 7) + ((lane >> 4) << 3)][kk + (((lane >> 3) & 1) << 3)]);
#pragma unroll
            for (int bt = 0; bt < kBatches; ++bt) {
                unsigned afrag[4];
                pie_ldmatrix_x4(afrag, &x_tile[bt * kBatch + (lane & 15)][kk + ((lane >> 4) << 3)]);
                pie_mma_bf16_16816(acc[bt][0], afrag, bfrag[0], bfrag[1]);
                pie_mma_bf16_16816(acc[bt][1], afrag, bfrag[2], bfrag[3]);
            }
        }
        __syncthreads();
    }

    for (int bt = 0; bt < batches; ++bt) {
        {
            float* mine = c_tile + warp * 16 * 16;
            const int m = lane >> 2;
            const int col = (lane & 3) << 1;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                mine[m * 16 + t * 8 + col] = acc[bt][t][0];
                mine[m * 16 + t * 8 + col + 1] = acc[bt][t][1];
                mine[(m + 8) * 16 + t * 8 + col] = acc[bt][t][2];
                mine[(m + 8) * 16 + t * 8 + col + 1] = acc[bt][t][3];
            }
        }
        __syncthreads();
        for (int idx = lane; idx < 16 * 16; idx += 32) {
            const int m = idx / 16;
            const int nn = idx % 16;
            const int t = bt * kBatch + m;
            const int row = row0 + warp * 16 + nn;
            if (t < group && row < n) {
                const int plane_route = order[begin + t];
                out[static_cast<long long>(plane_route) * n + row] =
                    f32_to_bf16(c_tile[warp * 16 * 16 + idx] * g[row]);
            }
        }
        __syncthreads();
    }
}

}  // namespace pie::linear

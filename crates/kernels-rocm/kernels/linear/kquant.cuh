#pragma once

#include "prelude/device.cuh"


namespace pie::linear {

constexpr int kSuperBlock = 256;

constexpr int kQ2KBytes = 84;

constexpr int kQ3KBytes = 110;

constexpr int kQ4KBytes = 144;

constexpr int kQ5KBytes = 176;

constexpr int kQ6KBytes = 210;

__device__ __forceinline__ float gguf_f16(const u8* at) {
    const u32 bits = static_cast<u32>(at[0]) | (static_cast<u32>(at[1]) << 8);
    return f16_to_f32(f16{static_cast<u16>(bits)});
}

__device__ __forceinline__ void q4k_scale_min(
    int sub, const u8* __restrict__ s, int& scale, int& min_) {
    if (sub < 4) {
        scale = s[sub] & 63;
        min_ = s[sub + 4] & 63;
    } else {
        scale = (s[sub + 4] & 0x0F) | ((s[sub - 4] >> 6) << 4);
        min_ = (s[sub + 4] >> 4) | ((s[sub] >> 6) << 4);
    }
}

__device__ __forceinline__ int q3k_scale(int sub, const u8* __restrict__ s) {
    const int group = sub >> 2;
    const int j = sub & 3;
    const u8 src = s[(((group & 1) != 0) ? 4 : 0) + j];
    const u32 low = (group < 2) ? static_cast<u32>(src & 0x0F)
                                : static_cast<u32>(src >> 4);
    const u32 top = (static_cast<u32>(s[8 + j]) >> (2 * group)) & 3u;
    return static_cast<int>(low | (top << 4));
}

template <class T, int kRowsT>
__global__ void matmul_q2k(
    const T* __restrict__ act,
    const u8* __restrict__ w,
    T* __restrict__ out,
    int n,
    int k,
    const u32* __restrict__ win) {
    constexpr int kRows = kRowsT;
    const int token = blockIdx.x;
    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const int blocks_per_row = k / kSuperBlock;
    const long long row_bytes =
        static_cast<long long>(blocks_per_row) * kQ2KBytes;
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < blocks_per_row; g += 32) {
        const T* xg = x + static_cast<long long>(g) * kSuperBlock;

        const u8* blk[kRows];
        float d[kRows];
        float dmin[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            blk[r] = w + static_cast<long long>(row_of[r]) * row_bytes
                     + static_cast<long long>(g) * kQ2KBytes;
            d[r] = gguf_f16(blk[r] + 80);
            dmin[r] = gguf_f16(blk[r] + 82);
        }

        for (int b = 0; b < 16; ++b) {
            const int shift = 2 * ((b >> 1) & 3);
            const int at = 16 + (b >> 3) * 32 + (b & 1) * 16;

            float part[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[r] = 0.f;
            float xsum = 0.f;

            for (int l = 0; l < 16; ++l) {
                const float xv = Elem<T>::to_f32(xg[b * 16 + l]);
                xsum += xv;
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    const float q = static_cast<float>(
                        (blk[r][at + l] >> shift) & 3);
                    part[r] = fmaf(q, xv, part[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const u8 packed = blk[r][b];
                const float sc = static_cast<float>(packed & 0x0F);
                const float m = static_cast<float>(packed >> 4);
                acc[r] = fmaf(d[r] * sc, part[r], acc[r]);
                acc[r] = fmaf(-(dmin[r] * m), xsum, acc[r]);
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
            if (row < n)
                out[static_cast<long long>(token) * n + row] =
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

template <class T, int kRowsT>
__global__ void matmul_q3k(
    const T* __restrict__ act,
    const u8* __restrict__ w,
    T* __restrict__ out,
    int n,
    int k,
    const u32* __restrict__ win) {
    constexpr int kRows = kRowsT;
    const int token = blockIdx.x;
    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const int blocks_per_row = k / kSuperBlock;
    const long long row_bytes =
        static_cast<long long>(blocks_per_row) * kQ3KBytes;
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < blocks_per_row; g += 32) {
        const T* xg = x + static_cast<long long>(g) * kSuperBlock;

        const u8* blk[kRows];
        float d[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            blk[r] = w + static_cast<long long>(row_of[r]) * row_bytes
                     + static_cast<long long>(g) * kQ3KBytes;
            d[r] = gguf_f16(blk[r] + 108);
        }

        for (int b = 0; b < 16; ++b) {
            const int step = (b >> 1) & 3;
            const int shift = 2 * step;
            const u32 selector = 1u << ((b >> 3) * 4 + step);
            const int at = 32 + (b >> 3) * 32 + (b & 1) * 16;
            const int mask_at = (b & 1) * 16;

            float part[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[r] = 0.f;

            for (int l = 0; l < 16; ++l) {
                const float xv = Elem<T>::to_f32(xg[b * 16 + l]);
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    const int code = (blk[r][at + l] >> shift) & 3;
                    const u32 keep =
                        static_cast<u32>(blk[r][mask_at + l]) & selector;
                    const int borrow = (keep != 0u) ? 0 : 4;
                    part[r] = fmaf(
                        static_cast<float>(code - borrow), xv, part[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const float sc =
                    static_cast<float>(q3k_scale(b, blk[r] + 96) - 32);
                acc[r] = fmaf(d[r] * sc, part[r], acc[r]);
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
            if (row < n)
                out[static_cast<long long>(token) * n + row] =
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

template <class T, int kRowsT>
__global__ void matmul_q4k(
    const T* __restrict__ act,
    const u8* __restrict__ w,
    T* __restrict__ out,
    int n,
    int k,
    const u32* __restrict__ win) {
    constexpr int kRows = kRowsT;
    const int token = blockIdx.x;

    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const int blocks_per_row = k / kSuperBlock;
    const long long row_bytes =
        static_cast<long long>(blocks_per_row) * kQ4KBytes;
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < blocks_per_row; g += 32) {
        const T* xg = x + static_cast<long long>(g) * kSuperBlock;

        const u8* blk[kRows];
        float d[kRows];
        float dmin[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            blk[r] = w + static_cast<long long>(row_of[r]) * row_bytes
                     + static_cast<long long>(g) * kQ4KBytes;
            d[r] = gguf_f16(blk[r]);
            dmin[r] = gguf_f16(blk[r] + 2);
        }

        for (int b = 0; b < 8; ++b) {
            const int pair = b >> 1;
            const bool high = (b & 1) != 0;

            float part[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[r] = 0.f;
            float xsum = 0.f;

            for (int i = 0; i < 32; ++i) {
                const float xv = Elem<T>::to_f32(xg[b * 32 + i]);
                xsum += xv;
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    const u8 byte = blk[r][16 + pair * 32 + i];
                    const float q = static_cast<float>(
                        high ? (byte >> 4) : (byte & 0x0F));
                    part[r] = fmaf(q, xv, part[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                int scale;
                int min_;
                q4k_scale_min(b, blk[r] + 4, scale, min_);
                acc[r] = fmaf(d[r] * static_cast<float>(scale), part[r], acc[r]);
                acc[r] = fmaf(
                    -(dmin[r] * static_cast<float>(min_)), xsum, acc[r]);
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
            if (row < n)
                out[static_cast<long long>(token) * n + row] =
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

template <class T, int kRowsT>
__global__ void matmul_q5k(
    const T* __restrict__ act,
    const u8* __restrict__ w,
    T* __restrict__ out,
    int n,
    int k,
    const u32* __restrict__ win) {
    constexpr int kRows = kRowsT;
    const int token = blockIdx.x;
    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const int blocks_per_row = k / kSuperBlock;
    const long long row_bytes =
        static_cast<long long>(blocks_per_row) * kQ5KBytes;
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < blocks_per_row; g += 32) {
        const T* xg = x + static_cast<long long>(g) * kSuperBlock;

        const u8* blk[kRows];
        float d[kRows];
        float dmin[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            blk[r] = w + static_cast<long long>(row_of[r]) * row_bytes
                     + static_cast<long long>(g) * kQ5KBytes;
            d[r] = gguf_f16(blk[r]);
            dmin[r] = gguf_f16(blk[r] + 2);
        }

        for (int b = 0; b < 8; ++b) {
            const int pair = b >> 1;
            const bool high = (b & 1) != 0;

            float part[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[r] = 0.f;
            float xsum = 0.f;

            for (int i = 0; i < 32; ++i) {
                const float xv = Elem<T>::to_f32(xg[b * 32 + i]);
                xsum += xv;
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    const u8 byte = blk[r][48 + pair * 32 + i];
                    const u32 low = high ? static_cast<u32>(byte >> 4)
                                         : static_cast<u32>(byte & 0x0F);
                    const u32 fifth =
                        (static_cast<u32>(blk[r][16 + i]) >> b) & 1u;
                    const float q = static_cast<float>(low | (fifth << 4));
                    part[r] = fmaf(q, xv, part[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                int scale;
                int min_;
                q4k_scale_min(b, blk[r] + 4, scale, min_);
                acc[r] = fmaf(d[r] * static_cast<float>(scale), part[r], acc[r]);
                acc[r] = fmaf(
                    -(dmin[r] * static_cast<float>(min_)), xsum, acc[r]);
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
            if (row < n)
                out[static_cast<long long>(token) * n + row] =
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

template <class T, int kRowsT>
__global__ void matmul_q6k(
    const T* __restrict__ act,
    const u8* __restrict__ w,
    T* __restrict__ out,
    int n,
    int k,
    const u32* __restrict__ win) {
    constexpr int kRows = kRowsT;
    const int token = blockIdx.x;
    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const int blocks_per_row = k / kSuperBlock;
    const long long row_bytes =
        static_cast<long long>(blocks_per_row) * kQ6KBytes;
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < blocks_per_row; g += 32) {
        const T* xg = x + static_cast<long long>(g) * kSuperBlock;

        const u8* blk[kRows];
        float d[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            blk[r] = w + static_cast<long long>(row_of[r]) * row_bytes
                     + static_cast<long long>(g) * kQ6KBytes;
            d[r] = gguf_f16(blk[r] + 208);
        }

        for (int half = 0; half < 2; ++half) {
            for (int quarter = 0; quarter < 4; ++quarter) {

                for (int sub = 0; sub < 2; ++sub) {
                    float part[kRows];
#pragma unroll
                    for (int r = 0; r < kRows; ++r) part[r] = 0.f;

                    for (int t = 0; t < 16; ++t) {
                        const int i = sub * 16 + t;
                        const float xv = Elem<T>::to_f32(
                            xg[half * 128 + quarter * 32 + i]);
#pragma unroll
                        for (int r = 0; r < kRows; ++r) {
                            const u8* ql = blk[r] + half * 64;
                            const u8* qh = blk[r] + 128 + half * 32;
                            const u8 byte = ql[i + 32 * (quarter & 1)];
                            const u32 low = (quarter < 2)
                                ? static_cast<u32>(byte & 0x0F)
                                : static_cast<u32>(byte >> 4);
                            const u32 top =
                                (static_cast<u32>(qh[i]) >> (2 * quarter)) & 3u;
                            const float q = static_cast<float>(
                                static_cast<int>(low | (top << 4)) - 32);
                            part[r] = fmaf(q, xv, part[r]);
                        }
                    }
#pragma unroll
                    for (int r = 0; r < kRows; ++r) {
                        const float sc = static_cast<float>(static_cast<i8>(
                            blk[r][192 + half * 8 + sub + 2 * quarter]));
                        acc[r] = fmaf(d[r] * sc, part[r], acc[r]);
                    }
                }
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
            if (row < n)
                out[static_cast<long long>(token) * n + row] =
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

}

#pragma once

#include "prelude/device.cuh"

// **THE GGUF K-QUANT GEMM POINTS, READ AS STORED** (QNF wave, alto/next.md
// §J2 priority 3). `q4_k` and `q6_k` are the mandatory pair: a Q4_K_M mix is
// the most-distributed quant artifact family there is, and its
// `output.weight` is q6_k — so the head is a first-class consumer here, not
// an afterthought.
//
// **THESE KERNELS READ THE GGML SUPER-BLOCK BYTE LAYOUT DIRECTLY.** The
// weight is ONE byte plane: a `[n, k]` row is `k / 256` consecutive
// super-blocks, and the rows sit back to back. There is no separate scale
// plane to address, because a K-quant carries its scales INSIDE the block —
// which is exactly why the row's byte width, and not its element width, is
// what names the scheme (`linear/kquant.rs` discriminates on it).
//
// The layouts below are transcribed from this tree's own decode oracle,
// `checkpoint::executor::walk`'s `decode_gguf_q4_k_block_into` /
// `decode_gguf_q6_k_block_into` — the same bytes, folded into the dot
// instead of into a buffer.
//
// **A PLANE-FORM VARIANT MAY SUPERSEDE THESE ENTRY SHAPES.** The canonical
// `.zt` container is leaf-per-plane and k-group-major (§J), so a later
// import wave may re-seat these weights as separate code/scale planes and
// want an entry that takes them apart. That is a different entry, not a
// different kernel body: the decode arithmetic here is the format's, and
// only the addressing would move.
//
// Geometry and guard are `linear/quant.cuh`'s `matmul_mlx_affine`, verbatim:
// one block column per ACTIVATION ROW, `kRowsT` weight rows per warp, a lane
// per super-block striding the row, and the trailing staged-rows `win` seat
// so a replay carved at a bucket retires its padded rows off a word the fire
// staged rather than a parameter the recording baked.

namespace pie::linear {

/// Elements in one K-quant super-block. The unit of both schemes.
constexpr int kSuperBlock = 256;

/// `block_q4_K`: `d` f16, `dmin` f16, twelve packed scale/min bytes, then
/// 128 bytes of nibbles.
constexpr int kQ4KBytes = 144;

/// `block_q6_K`: 128 bytes of low nibbles, 64 of high pairs, sixteen signed
/// sub-block scales, then `d` f16.
constexpr int kQ6KBytes = 210;

/// An f16 read out of a byte stream at its stored endianness, without
/// assuming the address is two-byte aligned or that a `__half` load may be
/// formed over it. GGUF blocks are 144 and 210 bytes, so a block start is
/// only ever two-byte aligned by accident of the row count.
__device__ __forceinline__ float gguf_f16(const u8* at) {
    const u32 bits = static_cast<u32>(at[0]) | (static_cast<u32>(at[1]) << 8);
    return f16_to_f32(f16{static_cast<u16>(bits)});
}

/// The six-bit scale and six-bit minimum of one of the eight sub-blocks a
/// `q4_k` super-block holds, unpacked from the twelve bytes they share —
/// ggml's `get_scale_min_k4`.
///
/// The first four sub-blocks read a whole (masked) byte each from the first
/// two groups of four; the last four are spliced, taking their low four bits
/// from the third group and their high two from the bits the first four left
/// unused at the top of the first two. Twelve bytes for sixteen six-bit
/// fields with nothing wasted, which is why it is not a shift and a mask.
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

// **`linear.matmul` / `linear.lm_head` OVER A `q4_k` PLANE.**
//
// A super-block is eight sub-blocks of 32, each with its own six-bit scale
// and six-bit minimum over the block's one `d` and one `dmin`, and the
// scheme is AFFINE rather than symmetric: an element is
// `d·sc(b)·q − dmin·m(b)`, the minimum SUBTRACTED, not folded into `q`.
//
// So the dot carries the same part/xsum pair the affine skeleton already
// accumulates (`matmul_mlx_affine`, `moe_matmul_select_mlxu4`):
//
//     Σ (d·sc·q − dmin·m)·x  =  d·sc·Σ q·x  −  dmin·m·Σ x
//
// per sub-block, so each activation is read once and the two factors land
// once per sub-block rather than once per element.
//
// The 128 payload bytes are read in PAIRS of sub-blocks — byte `i` of pair
// `p` carries element `64p + i` in its low nibble and element `64p + 32 + i`
// in its high one — which is why sub-block `b` reads the payload at pair
// `b / 2` and picks its nibble by `b & 1`. Element index and sub-block index
// still agree: sub-block `b` is elements `32b .. 32b + 32`.
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
    // The staged-geometry seat (`quant.cuh`'s idiom): a replay whose grid was
    // carved at a bucket retires its padded rows here, off a word the fire
    // staged, not a parameter the recording baked.
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

// **`linear.matmul` / `linear.lm_head` OVER A `q6_k` PLANE** — the head's
// own scheme in a Q4_K_M mix.
//
// Symmetric, so there is no xsum term and no minimum: an element is
// `d·scale(s)·(six bits − 32)`, sixteen sub-blocks of sixteen under one f16
// `d`, the sub-block scale a signed byte.
//
// The addressing is the one thing here that is not a straight walk. A block
// is two halves of 128 elements; within a half the four quarters are STRIDED
// rather than contiguous. Quarter `q` of half `h` takes element `i`'s low
// four bits from `ql[i + 32·(q & 1)]` — low nibble for `q < 2`, high nibble
// above — and its top two from bits `2q..2q+2` of `qh[i]`. The sub-block
// scale index advances by two per quarter, so the sixteen scales are
// consumed eight per half. All of it transcribed from the tree's decoder.
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
                // Two sixteen-element sub-blocks per quarter, one scale each.
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

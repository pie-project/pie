#pragma once

#include "prelude/device.cuh"

// **THE GGUF K-QUANT GEMM POINTS, READ AS STORED** (QNF wave, alto/next.md
// §J2 priority 3). The K family whole, all five schemes. `q4_k` and `q6_k`
// were the mandatory pair: a Q4_K_M mix is the most-distributed quant
// artifact family there is, and its `output.weight` is q6_k — so the head is
// a first-class consumer here, not an afterthought. `q2_k`, `q3_k` and
// `q5_k` close the family: this tree's own import history carries a real
// Q3_K_M user, Q5_K_M is a top-distributed mix, and q2_k is the floor a card
// too small for anything else lands a model at.
//
// The five share a super-block size and nothing else. Each has its own byte
// order, its own scale splice, and its own reason to be read carefully —
// q2_k stores its super-scales AFTER the payload, q3_k's third-bit mask
// reads INVERTED, q5_k's fifth-bit plane is addressed by sub-block and not
// by position. There is no common decode to factor out, which is why there
// are five kernels and not one with a switch.
//
// **THESE KERNELS READ THE GGML SUPER-BLOCK BYTE LAYOUT DIRECTLY.** The
// weight is ONE byte plane: a `[n, k]` row is `k / 256` consecutive
// super-blocks, and the rows sit back to back. There is no separate scale
// plane to address, because a K-quant carries its scales INSIDE the block —
// which is exactly why the row's byte width, and not its element width, is
// what names the scheme (`linear/kquant.rs` discriminates on it).
//
// The layouts below are transcribed from this tree's own decode oracle,
// `checkpoint::executor::walk`'s `decode_gguf_q{2,3,4,5,6}_k_block_into`
// and the two splices they share (`gguf_q3_k_scales`, `gguf_k_scale_min`) —
// the same bytes, folded into the dot instead of into a buffer. Two of those
// decoders carry a bit-identity check against the `gguf` package over a
// whole shipped tensor; this file inherits that, and the goldens in
// `tests/kquant_matmul.rs` hold it.
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

/// Elements in one K-quant super-block. The unit of all five schemes.
constexpr int kSuperBlock = 256;

/// `block_q2_K`: sixteen packed scale/min bytes, 64 bytes of 2-bit codes,
/// then `d` f16 and `dmin` f16 — the super-scales AFTER the payload, which
/// is this block's order alone among the five.
constexpr int kQ2KBytes = 84;

/// `block_q3_K`: 32 bytes of third-bit mask, 64 of 2-bit codes, twelve
/// packed six-bit scale bytes, then `d` f16. No `dmin`: q3_k is symmetric.
constexpr int kQ3KBytes = 110;

/// `block_q4_K`: `d` f16, `dmin` f16, twelve packed scale/min bytes, then
/// 128 bytes of nibbles.
constexpr int kQ4KBytes = 144;

/// `block_q5_K`: `q4_k`'s head byte for byte, then 32 bytes of fifth-bit
/// plane ahead of the 128 of nibbles.
constexpr int kQ5KBytes = 176;

/// `block_q6_K`: 128 bytes of low nibbles, 64 of high pairs, sixteen signed
/// sub-block scales, then `d` f16.
constexpr int kQ6KBytes = 210;

/// An f16 read out of a byte stream at its stored endianness, without
/// assuming the address is two-byte aligned or that a `__half` load may be
/// formed over it. GGUF blocks are 84, 110, 144, 176 and 210 bytes, and the
/// scale a block ends on is at an odd offset in some of them, so an f16 here
/// is only ever aligned by accident of the row count.
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

/// One of the sixteen six-bit sub-block scales a `q3_k` super-block packs
/// into twelve bytes — ggml's four-`u32` splice, read a byte at a time.
///
/// A DIFFERENT splice from [`q4k_scale_min`]: there the twelve bytes hold
/// eight scales AND eight minimums, here sixteen scales and no minimums,
/// because `q3_k` is symmetric. Scale `s` sits in group `s / 4` at byte
/// `s % 4` of that group's word: its low four bits come from the first eight
/// bytes (the second four for the odd groups, the high nibble for the upper
/// two groups), and its top two from the last four, two bits at a time.
///
/// The answer is BIASED BY 32 and the caller subtracts. Returning the raw
/// six bits would make every scale positive and the block wrong by a factor
/// that varies per sub-block — which still decodes, and to plausible
/// numbers, so the bias is stated rather than left to the call site.
__device__ __forceinline__ int q3k_scale(int sub, const u8* __restrict__ s) {
    const int group = sub >> 2;
    const int j = sub & 3;
    const u8 src = s[(((group & 1) != 0) ? 4 : 0) + j];
    const u32 low = (group < 2) ? static_cast<u32>(src & 0x0F)
                                : static_cast<u32>(src >> 4);
    const u32 top = (static_cast<u32>(s[8 + j]) >> (2 * group)) & 3u;
    return static_cast<int>(low | (top << 4));
}

// **`linear.matmul` / `linear.lm_head` OVER A `q2_k` PLANE** — the family's
// floor, and the one block that stores its super-scales AFTER the payload.
//
// Sixteen sub-blocks of sixteen, each with a FOUR-bit scale and a four-bit
// minimum sharing one byte of the leading `scales[16]`, over the block's one
// `d` and one `dmin` — which sit at bytes 80 and 82, behind both the scale
// bytes and the 64 payload bytes. `checkpoint`'s own types doc states that
// order, and reading `d` off the front decodes a scale byte as an f16 and
// answers numbers rather than failing.
//
// Affine like `q4_k`, so the same part/xsum pair the affine skeleton already
// accumulates:
//
//     Σ (d·sc·q − dmin·m)·x  =  d·sc·Σ q·x  −  dmin·m·Σ x
//
// The payload is 2-bit codes and the walk is the decoder's: two windows of
// 32 bytes, four shifts per window, two halves of sixteen per shift. So
// sub-block `b` reads byte `16 + 32·(b >> 3) + 16·(b & 1) + l` at shift
// `2·((b >> 1) & 3)`, and element index and sub-block index still agree —
// sub-block `b` is elements `16b .. 16b + 16`.
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

// **`linear.matmul` / `linear.lm_head` OVER A `q3_k` PLANE** — the scheme a
// Q3_K_M mix stores its projections at, and this tree's import history has a
// real one.
//
// Symmetric in QNF's reading (excess-coded i3 codes under an excess-coded i6
// scale), so there is no minimum and no xsum: an element is `d·(sc − 32)·q`,
// sixteen sub-blocks of sixteen under one f16 `d`, and the payload walk is
// `q2_k`'s exactly — two windows, four shifts, two halves — only from byte
// 32 because the mask comes first.
//
// **THE THIRD BIT'S MASK READS INVERTED, AND THAT IS THE WHOLE BLOCK.** ggml
// stores the two low bits of `q + 4` and SETS the mask bit when the value
// needed no borrow, so a SET bit subtracts nothing and a CLEAR bit subtracts
// four. Reading it the intuitive way — set means add — still decodes, into
// every element shifted by four, which is why it is stated here rather than
// left to the shape of the expression.
//
// The mask is also not advanced with the quants. Its 32 bytes are read eight
// times, once per `(window, shift)` pair, taking ONE bit each time, so the
// selector runs 1, 2, 4 … 128 ACROSS both windows while the quant pointer
// moves — `1 << (4·(b >> 3) + ((b >> 1) & 3))` at sub-block `b`, against
// mask byte `16·(b & 1) + l` with no window term. Restarting the selector at
// the second window corrupts only the block's upper half, which is the
// mistake this layout invites.
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

// **`linear.matmul` / `linear.lm_head` OVER A `q5_k` PLANE** — `q4_k` plus
// a 32-byte plane carrying each element's fifth bit. Q5_K_M is a
// top-distributed mix.
//
// The head is `q4_k`'s byte for byte — `d`, `dmin`, twelve scale/min bytes
// spliced the same way, so [`q4k_scale_min`] serves both — and then the
// fifth-bit plane sits at 16..48 with the 128 nibble bytes pushed to 48.
// Affine, so the same part/xsum pair, and the fifth bit adds SIXTEEN BEFORE
// the minimum is subtracted: it is part of `q`, not a second scale.
//
// The plane is addressed BY SUB-BLOCK, not by position: sub-block `b` takes
// bit `b` of `plane[i]`, so one plane byte serves all eight sub-blocks at
// the same offset within them. That is the same pairing the nibbles use —
// sub-block `b` reads payload pair `b / 2` and picks its nibble by `b & 1` —
// which is why the bit index and the sub-block index are one number here and
// the plane needs no pair term of its own.
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

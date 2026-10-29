#include "linear/codec.cuh"
#include "linear/grouped_fp8.cuh"
#include "linear/quant.cuh"

namespace pie::linear {

/// An fp8b128 plane (codes `[rows, k]`, each 128x128 block's exponent in
/// tcgen05's atoms) as bf16, each value exactly: what `quantize_fp8b128`
/// read. 16 codes a thread; `k` whole 128s.
__global__ void __launch_bounds__(256) dequantize_fp8b128(const u8* __restrict__ codes,
                                                          const u8* __restrict__ atoms,
                                                          bf16* __restrict__ out, int rows,
                                                          int k) {
    const long long at = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    if (at >= static_cast<long long>(rows) * k) return;
    const int r = static_cast<int>(at / k);
    const int c = static_cast<int>(at % k);
    const float s = __uint_as_float(static_cast<unsigned>(atoms[atom_offset(r, c / 128, k / 128)]) << 23);
    const uint4 q = *reinterpret_cast<const uint4*>(codes + at);
    const unsigned qs[4] = {q.x, q.y, q.z, q.w};
    unsigned w[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 lo = gfp8::e4m3x2_f32(static_cast<unsigned short>(qs[i] & 0xFFFFu));
        const float2 hi = gfp8::e4m3x2_f32(static_cast<unsigned short>(qs[i] >> 16));
        w[2 * i] = pack_bf16x2(lo.x * s, lo.y * s);
        w[2 * i + 1] = pack_bf16x2(hi.x * s, hi.y * s);
    }
    *reinterpret_cast<uint4*>(out + at) = make_uint4(w[0], w[1], w[2], w[3]);
    *reinterpret_cast<uint4*>(out + at + 8) = make_uint4(w[4], w[5], w[6], w[7]);
}

/// An MXFP4 plane (codes `[rows, k / 2]`, an e8m0 scale a 32 codes) as
/// bf16, each value exactly. 8 codes a thread; `count` whole 32s.
__global__ void __launch_bounds__(256) dequantize_mxfp4(const u8* __restrict__ codes,
                                                        const u8* __restrict__ scales,
                                                        bf16* __restrict__ out, long long count) {
    const long long at = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (at >= count) return;
    const float s = mxfp4_block_scale(scales[at / 32]);
    __half2 qd[4];
    mxfp4_unpack8(reinterpret_cast<const unsigned*>(codes)[at / 8], qd);
    unsigned w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(qd[j]);
        w[j] = pack_bf16x2(f.x * s, f.y * s);
    }
    *reinterpret_cast<uint4*>(out + at) = make_uint4(w[0], w[1], w[2], w[3]);
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

template <class F, int kBits, int kOffset, int kGroup>
__global__ void dequant_affine(
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    bf16* __restrict__ out,
    int n,
    int k)
{
    constexpr int kPerWord = 32 / kBits;
    constexpr unsigned kMask = (1u << kBits) - 1u;

    constexpr float kExcess = static_cast<float>(1 << (kBits - 1));

    const int words_per_row = k / kPerWord;
    const long long words = static_cast<long long>(n) * words_per_row;
    const long long at =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= words) return;

    const int row = static_cast<int>(at / words_per_row);
    const int word_in_row = static_cast<int>(at % words_per_row);

    const int groups_per_row = k / kGroup;
    const long long fx = static_cast<long long>(row) * groups_per_row
        + (word_in_row * kPerWord) / kGroup;

    const F* sf = reinterpret_cast<const F*>(scales);
    const F* bf = reinterpret_cast<const F*>(biases);
    const u8* zb = biases;
    const float sv = Elem<F>::to_f32(sf[fx]);
    float off;
    if constexpr (kOffset == kOffPost || kOffset == kOffPreReal) {
        off = Elem<F>::to_f32(bf[fx]);
    } else if constexpr (kOffset == kOffPreInt) {
        off = static_cast<float>(zb[fx]);
    } else {
        off = kExcess;
    }

    const unsigned word = reinterpret_cast<const unsigned*>(codes)[at];
    bf16* dst = out + static_cast<long long>(row) * k + word_in_row * kPerWord;

    unsigned int* dst2 = reinterpret_cast<unsigned int*>(dst);
#pragma unroll
    for (int j = 0; j < kPerWord; j += 2) {
        float v[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float code = static_cast<float>((word >> (kBits * (j + h))) & kMask);
            if constexpr (kOffset == kOffPost) {
                v[h] = fmaf(code, sv, off);
            } else {
                v[h] = sv * (code - off);
            }
        }
        dst2[j / 2] = pack_bf16x2(v[0], v[1]);
    }
}

}

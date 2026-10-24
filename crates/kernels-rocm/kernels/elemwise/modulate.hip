#pragma once

// The DiT modulations: a row's `x` scaled and shifted by its token's (or its
// lane's) modulation vectors, a residual's gated sum, and the two fused with
// the row norm before them. One block of `BLOCK` threads (whole wavefronts)
// a row, f32 throughout, one rounding on the store. The modulation plane is
// the activation's element or f32 (`TM`); `lane_of_row`, when present, maps
// a row to the modulation row it reads.

#include "prim/device.hip"

namespace pie::elemwise {

constexpr int kModScaleShift = 0;
constexpr int kModScale = 1;
constexpr int kModTanhGate = 2;

__device__ __forceinline__ int modulation_row(const i32* __restrict__ lane_of_row, long long row) {
    return lane_of_row != nullptr ? lane_of_row[row] : static_cast<int>(row);
}

// `o = x * (1 + scale) + shift` (FORM 0), `x * (1 + scale)` (1), or
// `tanh(gate) * x` (2); the vectors are `m`'s row, `scale` then `shift`.
template <class T, class TM, int FORM>
__global__ void modulate(const T* __restrict__ x, const TM* __restrict__ m,
                         const i32* __restrict__ lane_of_row, T* __restrict__ o, int width,
                         int m_width, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const T* xr = x + row * width;
    T* orow = o + row * width;
    const TM* mr = m + static_cast<long long>(modulation_row(lane_of_row, row)) * m_width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float xv = to_f32(xr[i]);
        float v;
        if constexpr (FORM == kModScaleShift) {
            v = fmaf(xv, 1.f + to_f32(mr[i]), to_f32(mr[i + width]));
        } else if constexpr (FORM == kModScale) {
            v = xv * (1.f + to_f32(mr[i]));
        } else {
            v = tanhf(to_f32(mr[i])) * xv;
        }
        orow[i] = from_f32<T>(v);
    }
}

// `r_out = r + g * y`, `g` the row's gate vector.
template <class T, class TM>
__global__ void gated_residual_add(const T* __restrict__ r, const TM* __restrict__ g,
                                   const T* __restrict__ y, const i32* __restrict__ lane_of_row,
                                   T* __restrict__ r_out, int width, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const T* rr = r + row * width;
    const T* yr = y + row * width;
    T* orow = r_out + row * width;
    const TM* gr = g + static_cast<long long>(modulation_row(lane_of_row, row)) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        orow[i] = from_f32<T>(fmaf(to_f32(gr[i]), to_f32(yr[i]), to_f32(rr[i])));
    }
}

constexpr int kNormLayer = 0;
constexpr int kNormRmsNoScale = 1;
constexpr int kNormRmsWeight = 2;

// A row normed (layer norm without affine, RMS norm unscaled, or RMS norm
// under `weight`) and then scale-shift modulated. Under GATED the row is
// first the gated residual sum `src + g * y`, rounded to T and landed in
// `residual`, and the norm reads that.
template <class T, class TM, int BLOCK, int NORM, bool GATED>
__device__ __forceinline__ void norm_modulate_row(
    const T* __restrict__ src, const TM* __restrict__ g, const T* __restrict__ y,
    T* __restrict__ residual, const T* __restrict__ weight, const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row, T* __restrict__ o, int width, int m_width,
    float norm_eps, const u32* __restrict__ win) {
    static_assert(BLOCK % kWave == 0, "a block is whole wavefronts");
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int tid = threadIdx.x;
    const int mrow = modulation_row(lane_of_row, row);
    const TM* mr = m + static_cast<long long>(mrow) * m_width;
    T* orow = o + row * width;
    const T* normed = GATED ? residual + row * width : src + row * width;

    __shared__ float smem[BLOCK / kWave];

    float local = 0.f;
    if constexpr (GATED) {
        const T* rr = src + row * width;
        const T* yr = y + row * width;
        const TM* gr = g + static_cast<long long>(mrow) * width;
        T* rout = residual + row * width;
        for (int i = tid; i < width; i += BLOCK) {
            const T summed = from_f32<T>(fmaf(to_f32(gr[i]), to_f32(yr[i]), to_f32(rr[i])));
            rout[i] = summed;
            const float v = to_f32(summed);
            local += NORM == kNormLayer ? v : v * v;
        }
        // Each thread rereads only the columns it wrote, so no barrier.
    } else {
        for (int i = tid; i < width; i += BLOCK) {
            const float v = to_f32(normed[i]);
            local += NORM == kNormLayer ? v : v * v;
        }
    }

    float mean = 0.f;
    float inv = 0.f;
    if constexpr (NORM == kNormLayer) {
        mean = block_sum(local, smem) / static_cast<float>(width);
        float spread = 0.f;
        for (int i = tid; i < width; i += BLOCK) {
            const float c = to_f32(normed[i]) - mean;
            spread += c * c;
        }
        inv = rsqrtf(block_sum(spread, smem) / static_cast<float>(width) + norm_eps);
    } else {
        inv = rsqrtf(block_sum(local, smem) / static_cast<float>(width) + norm_eps);
    }

    for (int i = tid; i < width; i += BLOCK) {
        float c = (to_f32(normed[i]) - mean) * inv;
        if constexpr (NORM == kNormRmsWeight) c *= to_f32(weight[i]);
        orow[i] = from_f32<T>(fmaf(c, 1.f + to_f32(mr[i]), to_f32(mr[i + width])));
    }
}

template <class T, class TM, int BLOCK, int NORM>
__global__ __launch_bounds__(BLOCK) void norm_modulate(
    const T* __restrict__ x, const T* __restrict__ weight, const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row, T* __restrict__ o, int width, int m_width,
    float norm_eps, const u32* __restrict__ win) {
    norm_modulate_row<T, TM, BLOCK, NORM, false>(x, nullptr, nullptr, nullptr, weight, m,
                                                 lane_of_row, o, width, m_width, norm_eps, win);
}

template <class T, class TM, int BLOCK, int NORM>
__global__ __launch_bounds__(BLOCK) void gated_residual_norm_modulate(
    const T* __restrict__ r, const TM* __restrict__ g, const T* __restrict__ y,
    T* __restrict__ r_out, const T* __restrict__ weight, const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row, T* __restrict__ o, int width, int m_width,
    float norm_eps, const u32* __restrict__ win) {
    norm_modulate_row<T, TM, BLOCK, NORM, true>(r, g, y, r_out, weight, m, lane_of_row, o, width,
                                                m_width, norm_eps, win);
}

}  // namespace pie::elemwise

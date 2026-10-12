#pragma once

#include "prelude/device.cuh"
#include "elemwise/norm.cuh"

namespace pie::elemwise {

constexpr int kModScaleShift = 0;
constexpr int kModScale = 1;
constexpr int kModTanhGate = 2;

__device__ __forceinline__ int modulation_row(
    const i32* __restrict__ lane_of_row, int row)
{
    return lane_of_row != nullptr ? lane_of_row[row] : row;
}

template <class T, class TM, int FORM>
__global__ void modulate(
    const T* __restrict__ x,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T* xr = x + static_cast<long long>(row) * width;
    T* orow = o + static_cast<long long>(row) * width;
    const TM* mr = m + static_cast<long long>(modulation_row(lane_of_row, row)) * m_width;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float xv = Elem<T>::to_f32(xr[i]);
        float v;
        if constexpr (FORM == kModScaleShift) {
            v = fmaf(xv, 1.f + Elem<TM>::to_f32(mr[i]), Elem<TM>::to_f32(mr[i + width]));
        } else if constexpr (FORM == kModScale) {
            v = xv * (1.f + Elem<TM>::to_f32(mr[i]));
        } else {
            v = tanhf(Elem<TM>::to_f32(mr[i])) * xv;
        }
        orow[i] = Elem<T>::from_f32(v);
    }
}

template <class T, class TM>
__global__ void gated_residual_add(
    const T* __restrict__ r,
    const TM* __restrict__ g,
    const T* __restrict__ y,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ r_out,
    int width,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T* rr = r + static_cast<long long>(row) * width;
    const T* yr = y + static_cast<long long>(row) * width;
    T* orow = r_out + static_cast<long long>(row) * width;
    const TM* gr = g + static_cast<long long>(modulation_row(lane_of_row, row)) * width;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        orow[i] = Elem<T>::from_f32(fmaf(Elem<TM>::to_f32(gr[i]),
                                         Elem<T>::to_f32(yr[i]),
                                         Elem<T>::to_f32(rr[i])));
    }
}

constexpr int kNormLayer = 0;
constexpr int kNormRmsNoScale = 1;
constexpr int kNormRmsWeight = 2;

template <class T, class TM, int BLOCK, int NORM, bool GATED>
__device__ __forceinline__ void norm_modulate_row(
    const T* __restrict__ src,
    const TM* __restrict__ g,
    const T* __restrict__ y,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    float norm_eps,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;
    const int mrow = modulation_row(lane_of_row, row);
    const TM* mr = m + static_cast<long long>(mrow) * m_width;
    T* orow = o + static_cast<long long>(row) * width;

    const T* normed = GATED ? residual + static_cast<long long>(row) * width
                            : src + static_cast<long long>(row) * width;

    __shared__ float buf[BLOCK];

    float local = 0.f;
    if constexpr (GATED) {
        const T* rr = src + static_cast<long long>(row) * width;
        const T* yr = y + static_cast<long long>(row) * width;
        const TM* gr = g + static_cast<long long>(mrow) * width;
        T* rout = residual + static_cast<long long>(row) * width;
        for (int i = tid; i < width; i += BLOCK) {
            const T summed = Elem<T>::from_f32(fmaf(Elem<TM>::to_f32(gr[i]),
                                                    Elem<T>::to_f32(yr[i]),
                                                    Elem<T>::to_f32(rr[i])));
            rout[i] = summed;
            const float v = Elem<T>::to_f32(summed);
            local += NORM == kNormLayer ? v : v * v;
        }
    } else {
        for (int i = tid; i < width; i += BLOCK) {
            const float v = Elem<T>::to_f32(normed[i]);
            local += NORM == kNormLayer ? v : v * v;
        }
    }

    float mean = 0.f;
    float inv = 0.f;
    if constexpr (NORM == kNormLayer) {
        mean = block_reduce_sum_fast<BLOCK>(local, buf) / static_cast<float>(width);
        __syncthreads();
        float spread = 0.f;
        for (int i = tid; i < width; i += BLOCK) {
            const float c = Elem<T>::to_f32(normed[i]) - mean;
            spread += c * c;
        }
        inv = rsqrtf(block_reduce_sum_fast<BLOCK>(spread, buf) /
                         static_cast<float>(width) +
                     norm_eps);
    } else {
        inv = rsqrtf(block_reduce_sum_fast<BLOCK>(local, buf) /
                         static_cast<float>(width) +
                     norm_eps);
    }

    for (int i = tid; i < width; i += BLOCK) {
        float c = (Elem<T>::to_f32(normed[i]) - mean) * inv;
        if constexpr (NORM == kNormRmsWeight) c *= Elem<T>::to_f32(weight[i]);
        orow[i] = Elem<T>::from_f32(
            fmaf(c, 1.f + Elem<TM>::to_f32(mr[i]), Elem<TM>::to_f32(mr[i + width])));
    }
}

template <class T, class TM, int BLOCK, int NORM>
__global__ void norm_modulate(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    float norm_eps,
    const u32* __restrict__ win)
{
    norm_modulate_row<T, TM, BLOCK, NORM, false>(
        x, nullptr, nullptr, nullptr, weight, m, lane_of_row, o, width, m_width,
        norm_eps, win);
}

template <class T, class TM, int BLOCK, int NORM>
__global__ void gated_residual_norm_modulate(
    const T* __restrict__ r,
    const TM* __restrict__ g,
    const T* __restrict__ y,
    T* __restrict__ r_out,
    const T* __restrict__ weight,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    float norm_eps,
    const u32* __restrict__ win)
{
    norm_modulate_row<T, TM, BLOCK, NORM, true>(
        r, g, y, r_out, weight, m, lane_of_row, o, width, m_width, norm_eps, win);
}

}

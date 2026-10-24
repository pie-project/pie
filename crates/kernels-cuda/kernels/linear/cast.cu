#include "linear/codec.cuh"
#include "linear/quant.cuh"

namespace pie::linear {

template <class T>
struct Cast {
    static __device__ __forceinline__ float to_f32(T v) { return Elem<T>::to_f32(v); }
    static __device__ __forceinline__ T from_f32(float v) { return Elem<T>::from_f32(v); }
};

template <>
struct Cast<f32> {
    static __device__ __forceinline__ float to_f32(f32 v) { return v; }
    static __device__ __forceinline__ f32 from_f32(float v) { return v; }
};

template <class T>
__global__ void cast_f32_to(const float* __restrict__ src, T* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(src[i]);
}

template <class T>
__global__ void scale(
    const T* __restrict__ src, T* __restrict__ dst, usize n, float factor) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(Cast<T>::to_f32(src[i]) * factor);
}

template <class T>
__global__ void scale_rows(T* buf, const T* l, int width) {
    const int row = blockIdx.x;
    T* row_buf = buf + static_cast<usize>(row) * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        row_buf[c] =
            Cast<T>::from_f32(Cast<T>::to_f32(row_buf[c]) * Cast<T>::to_f32(l[c]));
    }
}

}

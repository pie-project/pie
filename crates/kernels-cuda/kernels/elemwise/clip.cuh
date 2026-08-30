#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

/// **THE CLIPPED LINEAR'S CLAMP** (`.wiki/alto/multimodal.md` §6.5).
///
/// `x = min(max(x, lo), hi)`, in place, one thread per element. gemma4's
/// `vision_config.use_clipped_linears: true` publishes
/// `{input,output}_{min,max}` as scalars beside every vision projection, so a
/// text clamps what a matmul reads and what it writes; the bounds are the
/// checkpoint's own numbers and arrive stated.
///
/// **THE BOUNDS ARE ROUNDED THROUGH `T` FIRST**, the way `mul_scalar` next
/// door rounds its scalar: the clamp's output is a `T`, so comparing against
/// an f32 bound the element cannot represent would let a value land one
/// rounding past the bound the text stated.
template <class T>
__global__ void clamp(T* __restrict__ x, float lo, float hi, usize n)
{
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float lo_r = Elem<T>::to_f32(Elem<T>::from_f32(lo));
    const float hi_r = Elem<T>::to_f32(Elem<T>::from_f32(hi));
    const float v = Elem<T>::to_f32(x[i]);
    x[i] = Elem<T>::from_f32(fminf(fmaxf(v, lo_r), hi_r));
}


/// **THE SAME CLAMP, WITH THE BOUNDS ON THE DEVICE**
/// (`.wiki/alto/multimodal.md` §12.2).
///
/// `lo` and `hi` are one-element planes rather than launch arguments, because
/// gemma4's are 448 learned scalars the CHECKPOINT ships — saturating bounds
/// from quantization-aware training, one pair per side of every linear — and
/// a text that stated them would be a checkpoint transcribed into a `const`.
/// `elemwise::scale` reads its scalar the same way and for the same reason.
///
/// The bounds are already in `T`, so there is no rounding to do: the plain
/// form rounds its `float` arguments through the element before comparing,
/// and this one reads elements that were rounded at import.
template <class T>
__global__ void clamp_learned(
    T* __restrict__ x,
    const T* __restrict__ lo,
    const T* __restrict__ hi,
    usize n)
{
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float lo_v = Elem<T>::to_f32(lo[0]);
    const float hi_v = Elem<T>::to_f32(hi[0]);
    const float v = Elem<T>::to_f32(x[i]);
    x[i] = Elem<T>::from_f32(fminf(fmaxf(v, lo_v), hi_v));
}

}

#pragma once

#include <cuda_bf16.h>
#include "prelude/device.cuh"

namespace pie::ssm {

using f32 = float;
using state_bf16 = __nv_bfloat16;

__device__ __forceinline__ float silu_f(float z) {
    return z / (1.f + __expf(-z));
}

// Whether lane `r` folds this fire's boundary: a fire that buffers its rows
// for a later commit asks every arm not to write the state, and a
// predicated fire names the lanes that do.
__device__ __forceinline__ bool row_persists(
    const u8* __restrict__ mask, int r) {
    return mask == nullptr || mask[r] != 0;
}

__device__ __forceinline__ bool folds(
    bool write_state, const u8* __restrict__ mask, int r) {
    return write_state && row_persists(mask, r);
}

// A lane's state snapshots: entries snap[rl]..snap[rl + 1] of snap_rows (rows
// inside the lane, ascending) and snap_slots, each the state after that many
// rows landed in a slot of its own. None where snap is null.
struct Snaps {
    int e0;
    int e1;
};

__device__ __forceinline__ Snaps snaps_of(const int* snap, int rl) {
    return snap == nullptr ? Snaps{0, 0} : Snaps{snap[rl], snap[rl + 1]};
}

template <typename StateT>
__device__ __forceinline__ float state_load(const StateT* p) {
    return static_cast<float>(*p);
}

template <>
__device__ __forceinline__ float state_load<__nv_bfloat16>(
    const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename StateT>
__device__ __forceinline__ void state_store(StateT* p, float v) {
    *p = static_cast<StateT>(v);
}

template <>
__device__ __forceinline__ void state_store<__nv_bfloat16>(
    __nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

// What a value reads back as once it has been through the state's cell: the
// unfolded step rounds where the folded one stores, so both answer alike.
template <typename StateT>
__device__ __forceinline__ float state_round(float v) {
    StateT cell;
    state_store(&cell, v);
    return state_load(&cell);
}

template <bool KLast>
__device__ __forceinline__ long long state_offset(
    int k_idx, int v_idx, int K_d, int V_d) {
    if constexpr (KLast) {
        return (long long)v_idx * K_d + k_idx;
    } else {
        return (long long)k_idx * V_d + v_idx;
    }
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return __shfl_sync(0xffffffffu, x, 0);
}

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.f / (1.f + __expf(-x));
}

}

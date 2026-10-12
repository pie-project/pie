#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

template <class T>
using Elem = ::pie::Elem<T>;

template <class T>
__global__ void rope_mrope(
    T* __restrict__ q,
    T* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    int s0, int s1, int s2,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total_heads = num_q_heads + num_kv_heads;
    // The rotated prefix is its own neox head: pairs (i, i + rotary_dim/2)
    // at theta^(-2i/rotary_dim), as `rope_partial` and upstream turn it.
    const int half = rotary_dim / 2;

    const int pos_t = positions[3 * row + 0];
    const int pos_h = positions[3 * row + 1];
    const int pos_w = positions[3 * row + 2];
    (void)s0;

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        int axis_pos;
        const int m = dim_pair % 3;
        if (m == 1 && dim_pair < 3 * s1)      axis_pos = pos_h;
        else if (m == 2 && dim_pair < 3 * s2) axis_pos = pos_w;
        else                                  axis_pos = pos_t;

        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) /
                   static_cast<float>(rotary_dim));
        const float ang = static_cast<float>(axis_pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            T* qp = q +
                (static_cast<long long>(row) * num_q_heads + head_idx) * head_dim;
            const float a = Elem<T>::to_f32(qp[dim_pair]);
            const float b = Elem<T>::to_f32(qp[dim_pair + half]);
            qp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            qp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            T* kp = k +
                (static_cast<long long>(row) * num_kv_heads + kv_h) * head_dim;
            const float a = Elem<T>::to_f32(kp[dim_pair]);
            const float b = Elem<T>::to_f32(kp[dim_pair + half]);
            kp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            kp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
    }
}


template <class T>
__global__ void rope_mrope_blocked(
    T* __restrict__ q,
    T* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    int s0, int s1, int s2,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int rope_angles = rotary_dim / 2;
    const int total = s0 + s1 + s2;

    const int pos[3] = { positions[3 * row + 0],
                         positions[3 * row + 1],
                         positions[3 * row + 2] };

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        if (dim_pair >= rope_angles) continue;
        if (dim_pair >= total) continue;

        int axis;
        int within;
        if (dim_pair < s0)           { axis = 0; within = dim_pair; }
        else if (dim_pair < s0 + s1) { axis = 1; within = dim_pair - s0; }
        else                         { axis = 2; within = dim_pair - s0 - s1; }

        const float freq = powf(theta,
            -2.f * static_cast<float>(within) / static_cast<float>(total));
        const float ang = static_cast<float>(pos[axis]) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            T* qp = q +
                (static_cast<long long>(row) * num_q_heads + head_idx) * head_dim;
            const float a = Elem<T>::to_f32(qp[dim_pair]);
            const float b = Elem<T>::to_f32(qp[dim_pair + half]);
            qp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            qp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            T* kp = k +
                (static_cast<long long>(row) * num_kv_heads + kv_h) * head_dim;
            const float a = Elem<T>::to_f32(kp[dim_pair]);
            const float b = Elem<T>::to_f32(kp[dim_pair + half]);
            kp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            kp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
    }
}

}

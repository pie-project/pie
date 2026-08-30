#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

template <class T>
using Elem = ::pie::Elem<T>;

/// **MULTIMODAL ROTARY: ONE ROW, THREE POSITIONS** (`.wiki/alto/multimodal.md`
/// §2's second op).
///
/// `rope_partial`'s arithmetic, unchanged, over a position that is a TRIPLE
/// instead of a scalar. An image lane's row does not sit at one place in one
/// sequence; it sits at `(t, h, w)` — time, and the patch's row and column in
/// its grid — and the sections say which of the three each frequency pair
/// turns by. Everything else is the scalar kernel: the same `powf` frequency,
/// the same `__sincosf`, the same pair `(dim_pair, dim_pair + half)`, so a row
/// whose triple is `(p, p, p)` comes out where `rope_partial` at `p` would have
/// put it, to the last bit the two expressions can share.
///
/// **The split is INTERLEAVED, and this is the same formula
/// `qk_rmsnorm_rotate_mrope` already carries** — written once more here rather
/// than shared, because the two kernels differ in everything else and a shared
/// helper would tie a fused norm to a plain rotate. Frequency pairs alternate
/// `t, h, w, t, h, w, ...` for as far as the sections reach: pair `p` turns by
/// `h` when `p % 3 == 1` and `p < 3 * s1`, by `w` when `p % 3 == 2` and
/// `p < 3 * s2`, and by `t` otherwise — so a checkpoint stating `[11, 11, 10]`
/// over a 64-wide head gets exactly 11 `h` pairs, 10 `w` pairs, and the
/// remaining 11 of the interleaved prefix plus every pair above it turning by
/// `t`. `s0` is read by the shape of that "otherwise" and never by name; it is
/// taken anyway so the trace's three numbers arrive as three numbers.
///
/// Pairs at or above `rotary_dim / 2` are left alone, exactly as in the
/// partial scalar kernel — the tail of a head that this checkpoint does not
/// rotate.
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
    int s0, int s1, int s2)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int rope_angles = rotary_dim / 2;

    const int pos_t = positions[3 * n + 0];
    const int pos_h = positions[3 * n + 1];
    const int pos_w = positions[3 * n + 2];
    (void)s0;

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        if (dim_pair >= rope_angles) continue;

        int axis_pos;
        const int m = dim_pair % 3;
        if (m == 1 && dim_pair < 3 * s1)      axis_pos = pos_h;
        else if (m == 2 && dim_pair < 3 * s2) axis_pos = pos_w;
        else                                  axis_pos = pos_t;

        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) /
                   static_cast<float>(head_dim));
        const float ang = static_cast<float>(axis_pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            T* qp = q +
                (static_cast<long long>(n) * num_q_heads + head_idx) * head_dim;
            const float a = Elem<T>::to_f32(qp[dim_pair]);
            const float b = Elem<T>::to_f32(qp[dim_pair + half]);
            qp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            qp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            T* kp = k +
                (static_cast<long long>(n) * num_kv_heads + kv_h) * head_dim;
            const float a = Elem<T>::to_f32(kp[dim_pair]);
            const float b = Elem<T>::to_f32(kp[dim_pair + half]);
            kp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            kp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
    }
}


/// **THE TOWER'S ROTATION: CONTIGUOUS SECTIONS, AND EACH RESTARTS THE
/// LADDER** (`.wiki/alto/multimodal.md` §6.3).
///
/// The same pairing as `rope_mrope` above — `(dim_pair, dim_pair + half)`,
/// which is `rotate_half` — over the same `[rows, 3]` position stream. What
/// differs is which pair takes which axis, and at what frequency:
///
/// - the sections are CONTIGUOUS BLOCKS. Pairs `[0, s0)` turn by `t`,
///   `[s0, s0+s1)` by `h`, `[s0+s1, s0+s1+s2)` by `w`. The tower states
///   `[0, head_dim/4, head_dim/4]`, so it turns by `(h, w)` and reads no `t`
///   at all — `s0 == 0` is how a two-axis rotation is spelled here, rather
///   than by a second position shape.
/// - and each block RESTARTS the frequency ladder. The `i`-th pair OF ITS
///   BLOCK turns at `theta^(-2i / total)` where `total = s0 + s1 + s2`.
///
/// That second half is the part nobody would guess and the part a wrong
/// kernel still looks plausible under.
/// `Qwen3_5VisionRotaryEmbedding(head_dim / 2)` builds `head_dim/4`
/// frequencies over a `head_dim/2`-wide ladder, and
/// `freqs[pos_ids].flatten(1)` indexes that ONE ladder once per axis before
/// concatenating — so the exponent's numerator counts within the block and
/// its denominator is the ladder's width, which is `total` exactly when the
/// sections tile the rotated pairs, as the tower's do.
///
/// Pairs at or above `rotary_dim / 2`, and pairs past `total`, are left
/// alone: the tail of a head this checkpoint does not rotate.
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
    int s0, int s1, int s2)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int rope_angles = rotary_dim / 2;
    const int total = s0 + s1 + s2;

    const int pos[3] = { positions[3 * n + 0],
                         positions[3 * n + 1],
                         positions[3 * n + 2] };

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
                (static_cast<long long>(n) * num_q_heads + head_idx) * head_dim;
            const float a = Elem<T>::to_f32(qp[dim_pair]);
            const float b = Elem<T>::to_f32(qp[dim_pair + half]);
            qp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            qp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            T* kp = k +
                (static_cast<long long>(n) * num_kv_heads + kv_h) * head_dim;
            const float a = Elem<T>::to_f32(kp[dim_pair]);
            const float b = Elem<T>::to_f32(kp[dim_pair + half]);
            kp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            kp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
    }
}

}

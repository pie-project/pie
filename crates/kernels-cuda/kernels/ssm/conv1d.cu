#include "ssm/state.cuh"

namespace pie::ssm {

template <bool SILU, bool RESIDUAL>
__device__ __forceinline__ float conv_out(float acc, float x_t) {
    if constexpr (RESIDUAL) {
        return acc + x_t;
    } else if constexpr (SILU) {
        return silu_f(acc);
    } else {
        return acc;
    }
}

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void conv1d_chunked(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    (void)lanes_stated;
    const int c = blockIdx.x;
    const int r = blockIdx.y;
    if (c >= C) return;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;

    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;

    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;

    if (begin_at != nullptr) {
        int b = begin_at[rl];
        if (b > Nr) b = Nr;
        if (b > 0) { t0 += b; Nr -= b; }
    }
    if (commit_len != nullptr) {
        const int c = commit_len[rl];
        if (c < Nr) Nr = c;
    }
    if (Nr <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const T* x_r = x + (long long)(t0 + row0) * C;
    T* y_r = y + (long long)(t0 + row0) * C;
    T* state = state_out_base + (long long)slot * slot_stride_elems;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;

    for (int t = tid; t < Nr; t += block_size) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= K) break;
            const int src_t = t - (K - 1 - k) * dil;
            float xv = 0.f;
            if (src_t < 0) {
                xv = Elem<T>::to_f32(state[((K - 1) * dil + 1 + src_t) * C + c]);
            } else {
                xv = Elem<T>::to_f32(x_r[src_t * C + c]);
            }
            const float wv = Elem<T>::to_f32(weight[c * K + k]);
            acc += wv * xv;
        }
        y_r[t * C + c] = Elem<T>::from_f32(
            conv_out<SILU, RESIDUAL>(acc, Elem<T>::to_f32(x_r[t * C + c])));
    }

    __syncthreads();

    if (state_out_base && write_state &&
        (write_state_mask == nullptr || write_state_mask[rl] != 0) &&
        tid == 0) {
        const int span = (K - 1) * dil + 1;
        for (int s = 0; s < span; ++s) {
            const int src_t = Nr - span + s;
            const float v = (src_t < 0)
                ? Elem<T>::to_f32(state[(span + src_t) * C + c])
                : Elem<T>::to_f32(x_r[src_t * C + c]);
            state[s * C + c] = Elem<T>::from_f32(v);
        }
    }
}

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void conv1d_chunked_channel_tile(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    (void)lanes_stated;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= C) return;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;

    if (begin_at != nullptr) {
        int b = begin_at[rl];
        if (b > Nr) b = Nr;
        if (b > 0) { t0 += b; Nr -= b; }
    }
    if (commit_len != nullptr) {
        const int c = commit_len[rl];
        if (c < Nr) Nr = c;
    }
    if (Nr <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const T* x_r = x + static_cast<long long>(t0 + row0) * C;
    T* y_r = y + static_cast<long long>(t0 + row0) * C;
    T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    float wv[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        wv[k] = (k < K) ? Elem<T>::to_f32(weight[c * K + k]) : 0.f;
    }

    for (int t = 0; t < Nr; ++t) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= K) break;
            const int src_t = t - (K - 1 - k) * dil;
            float xv = 0.f;
            if (src_t < 0) {
                xv = Elem<T>::to_f32(state[((K - 1) * dil + 1 + src_t) * C + c]);
            } else {
                xv = Elem<T>::to_f32(x_r[src_t * C + c]);
            }
            acc += wv[k] * xv;
        }
        y_r[static_cast<long long>(t) * C + c] = Elem<T>::from_f32(
            conv_out<SILU, RESIDUAL>(acc, Elem<T>::to_f32(x_r[static_cast<long long>(t) * C + c])));
    }

    if (state_out_base && write_state &&
        (write_state_mask == nullptr || write_state_mask[rl] != 0)) {
        const int span = (K - 1) * dil + 1;
        for (int s = 0; s < span; ++s) {
            const int src_t = Nr - span + s;
            const float v = (src_t < 0)
                ? Elem<T>::to_f32(state[(span + src_t) * C + c])
                : Elem<T>::to_f32(x_r[src_t * C + c]);
            state[s * C + c] = Elem<T>::from_f32(v);
        }
    }
}

constexpr int CONV_TT = 32;

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void conv1d_chunked_row_tile(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(qo_indptr[lanes]);
    const int tile0 = static_cast<int>(blockIdx.y) * CONV_TT;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + CONV_TT, total);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int span = (K - 1) * dil + 1;
    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    float wv[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        wv[k] = (k < K) ? Elem<T>::to_f32(weight[c * K + k]) : 0.f;
    }
    int r = 0;
    {
        int lo = 0, hi = lanes - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (static_cast<int>(qo_indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
        }
        r = lo;
    }
    for (; r < lanes; ++r) {
        int t0 = static_cast<int>(qo_indptr[r]);
        int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
        if (t0 >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        if (begin_at != nullptr) {
            int b = begin_at[rl];
            if (b > Nr) b = Nr;
            if (b > 0) { t0 += b; Nr -= b; }
        }
        if (commit_len != nullptr) {
            const int cl = commit_len[rl];
            if (cl < Nr) Nr = cl;
        }
        if (Nr <= 0) continue;
        const int slot = slot_ids[rl];
        if (slot < 0) continue;
        const T* x_r = x + static_cast<long long>(t0 + row0) * C;
        T* y_r = y + static_cast<long long>(t0 + row0) * C;
        T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems;
        const int t_lo = max(tile0, t0) - t0;
        const int t_hi = min(tile_end, t0 + Nr) - t0;
        for (int t = t_lo; t < t_hi; ++t) {
            float acc = bias_v;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (k >= K) break;
                const int src_t = t - (K - 1 - k) * dil;
                float xv = 0.f;
                if (src_t < 0) {
                    xv = Elem<T>::to_f32(state[(span + src_t) * C + c]);
                } else {
                    xv = Elem<T>::to_f32(x_r[static_cast<long long>(src_t) * C + c]);
                }
                acc += wv[k] * xv;
            }
            y_r[static_cast<long long>(t) * C + c] = Elem<T>::from_f32(
                conv_out<SILU, RESIDUAL>(acc, Elem<T>::to_f32(x_r[static_cast<long long>(t) * C + c])));
        }
    }
}

// The row-tiled conv eight channels a thread for a four-tap, undilated
// kernel: a thread reads each of its rows as one 16-byte vector, carries the
// three rows before the current one in registers, and reads a tile's rows
// eight at a time so their loads are in flight together.
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void conv1d_chunked_row_tile_vec8(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    constexpr int TAPS = 4;
    constexpr int VEC = 8;
    constexpr int RUN = 8;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(qo_indptr[lanes]);
    const int tile0 = static_cast<int>(blockIdx.y) * CONV_TT;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + CONV_TT, total);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    auto unpack8 = [](const uint4& raw, float (&f)[VEC]) {
        const T* e = reinterpret_cast<const T*>(&raw);
        #pragma unroll
        for (int u = 0; u < VEC; ++u) f[u] = Elem<T>::to_f32(e[u]);
    };
    float wv[TAPS][VEC];
    {
        const uint4* wq = reinterpret_cast<const uint4*>(weight + static_cast<long long>(c) * TAPS);
        #pragma unroll
        for (int q = 0; q < TAPS; ++q) {
            float f[VEC];
            unpack8(wq[q], f);
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const int flat = q * VEC + e;
                wv[flat % TAPS][flat / TAPS] = f[e];
            }
        }
    }
    float bias_v[VEC];
    #pragma unroll
    for (int u = 0; u < VEC; ++u) bias_v[u] = bias ? Elem<T>::to_f32(bias[c + u]) : 0.f;
    int r = 0;
    {
        int lo = 0, hi = lanes - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (static_cast<int>(qo_indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
        }
        r = lo;
    }
    for (; r < lanes; ++r) {
        int t0 = static_cast<int>(qo_indptr[r]);
        int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
        if (t0 >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        if (begin_at != nullptr) {
            int b = begin_at[rl];
            if (b > Nr) b = Nr;
            if (b > 0) { t0 += b; Nr -= b; }
        }
        if (commit_len != nullptr) {
            const int cl = commit_len[rl];
            if (cl < Nr) Nr = cl;
        }
        if (Nr <= 0) continue;
        const int slot = slot_ids[rl];
        if (slot < 0) continue;
        const T* x_r = x + static_cast<long long>(t0 + row0) * C + c;
        T* y_r = y + static_cast<long long>(t0 + row0) * C + c;
        const T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems + c;
        const int t_lo = max(tile0, t0) - t0;
        const int t_hi = min(tile_end, t0 + Nr) - t0;
        if (t_lo >= t_hi) continue;
        float hist[TAPS - 1][VEC];
        #pragma unroll
        for (int j = 0; j < TAPS - 1; ++j) {
            const int src = t_lo - (TAPS - 1) + j;
            const uint4 raw = src < 0
                ? *reinterpret_cast<const uint4*>(state + static_cast<long long>(TAPS + src) * C)
                : *reinterpret_cast<const uint4*>(x_r + static_cast<long long>(src) * C);
            unpack8(raw, hist[j]);
        }
        for (int tb = t_lo; tb < t_hi; tb += RUN) {
            uint4 run[RUN];
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                const int t = min(tb + j, t_hi - 1);
                run[j] = *reinterpret_cast<const uint4*>(x_r + static_cast<long long>(t) * C);
            }
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                const int t = tb + j;
                if (t >= t_hi) break;
                float cur[VEC];
                unpack8(run[j], cur);
                T out[VEC];
                #pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    float acc = bias_v[u];
                    acc += wv[0][u] * hist[0][u];
                    acc += wv[1][u] * hist[1][u];
                    acc += wv[2][u] * hist[2][u];
                    acc += wv[3][u] * cur[u];
                    out[u] = Elem<T>::from_f32(conv_out<SILU, RESIDUAL>(acc, cur[u]));
                    hist[0][u] = hist[1][u];
                    hist[1][u] = hist[2][u];
                    hist[2][u] = cur[u];
                }
                *reinterpret_cast<uint4*>(y_r + static_cast<long long>(t) * C) =
                    *reinterpret_cast<const uint4*>(out);
            }
        }
    }
}

// The row-tiled conv four channels a thread (one 8-byte vector a row): a
// 64-register bound keeps eight blocks resident on sm_100. Same sums, same
// order, same bits as the 16-byte form.
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void __launch_bounds__(128, 8) conv1d_chunked_row_tile_vec4(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    using Run = uint2;
    constexpr int TAPS = 4;
    constexpr int VEC = 4;
    constexpr int RUN = 8;
    constexpr int TT = CONV_TT;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(qo_indptr[lanes]);
    const int tile0 = static_cast<int>(blockIdx.y) * TT;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + TT, total);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    // The taps, channel-major: element (u * TAPS + q) is channel c + u's tap q.
    T wraw[VEC * TAPS];
    {
        const Run* wq = reinterpret_cast<const Run*>(weight + static_cast<long long>(c) * TAPS);
        #pragma unroll
        for (int q = 0; q < TAPS; ++q) reinterpret_cast<Run*>(wraw)[q] = wq[q];
    }
    float wf[VEC * TAPS];
    #pragma unroll
    for (int i = 0; i < VEC * TAPS; ++i) wf[i] = Elem<T>::to_f32(wraw[i]);
    auto wt = [&](int q, int u) -> float { return wf[u * TAPS + q]; };
    float bias_v[VEC];
    #pragma unroll
    for (int u = 0; u < VEC; ++u) bias_v[u] = bias ? Elem<T>::to_f32(bias[c + u]) : 0.f;
    int r = 0;
    {
        int lo = 0, hi = lanes - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (static_cast<int>(qo_indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
        }
        r = lo;
    }
    for (; r < lanes; ++r) {
        int t0 = static_cast<int>(qo_indptr[r]);
        int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
        if (t0 >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        if (begin_at != nullptr) {
            int b = begin_at[rl];
            if (b > Nr) b = Nr;
            if (b > 0) { t0 += b; Nr -= b; }
        }
        if (commit_len != nullptr) {
            const int cl = commit_len[rl];
            if (cl < Nr) Nr = cl;
        }
        if (Nr <= 0) continue;
        const int slot = slot_ids[rl];
        if (slot < 0) continue;
        const T* x_r = x + static_cast<long long>(t0 + row0) * C + c;
        T* y_r = y + static_cast<long long>(t0 + row0) * C + c;
        const T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems + c;
        const int t_lo = max(tile0, t0) - t0;
        const int t_hi = min(tile_end, t0 + Nr) - t0;
        if (t_lo >= t_hi) continue;
        Run hist[TAPS - 1];
        #pragma unroll
        for (int j = 0; j < TAPS - 1; ++j) {
            const int src = t_lo - (TAPS - 1) + j;
            hist[j] = src < 0
                ? *reinterpret_cast<const Run*>(state + static_cast<long long>(TAPS + src) * C)
                : *reinterpret_cast<const Run*>(x_r + static_cast<long long>(src) * C);
        }
        for (int tb = t_lo; tb < t_hi; tb += RUN) {
            Run run[RUN];
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                const int t = min(tb + j, t_hi - 1);
                run[j] = *reinterpret_cast<const Run*>(x_r + static_cast<long long>(t) * C);
            }
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                const int t = tb + j;
                if (t >= t_hi) break;
                const T* h0 = reinterpret_cast<const T*>(&hist[0]);
                const T* h1 = reinterpret_cast<const T*>(&hist[1]);
                const T* h2 = reinterpret_cast<const T*>(&hist[2]);
                const T* cu = reinterpret_cast<const T*>(&run[j]);
                T out[VEC];
                #pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    const float cur = Elem<T>::to_f32(cu[u]);
                    float acc = bias_v[u];
                    acc += wt(0, u) * Elem<T>::to_f32(h0[u]);
                    acc += wt(1, u) * Elem<T>::to_f32(h1[u]);
                    acc += wt(2, u) * Elem<T>::to_f32(h2[u]);
                    acc += wt(3, u) * cur;
                    out[u] = Elem<T>::from_f32(conv_out<SILU, RESIDUAL>(acc, cur));
                }
                hist[0] = hist[1];
                hist[1] = hist[2];
                hist[2] = run[j];
                *reinterpret_cast<Run*>(y_r + static_cast<long long>(t) * C) =
                    *reinterpret_cast<const Run*>(out);
            }
        }
    }
}


template <class T>
__global__ void conv1d_state_commit(
    const T* __restrict__ x,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const int* __restrict__ snap,
    const int* __restrict__ snap_rows,
    const int* __restrict__ snap_slots,
    const u32* __restrict__ win,
    int lanes_stated)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    const int r = static_cast<int>(blockIdx.y);
    if (r >= lanes) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    if (write_state_mask != nullptr && write_state_mask[rl] == 0) return;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (begin_at != nullptr) {
        int b = begin_at[rl];
        if (b > Nr) b = Nr;
        if (b > 0) { t0 += b; Nr -= b; }
    }
    if (commit_len != nullptr) {
        const int cl = commit_len[rl];
        if (cl < Nr) Nr = cl;
    }
    if (Nr <= 0) return;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int span = (K - 1) * dil + 1;
    const T* x_r = x + static_cast<long long>(t0 + row0) * C;
    T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems;
    // Each snapshot's window, from the rows and the window the lane started
    // with, before the lane's own overwrites it.
    const Snaps sn = snaps_of(snap, rl);
    for (int e = sn.e0; e < sn.e1 && snap_rows[e] < Nr; ++e) {
        T* at = state_out_base + static_cast<long long>(snap_slots[e]) * slot_stride_elems;
        for (int s = 0; s < span; ++s) {
            const int src_t = snap_rows[e] - span + s;
            at[s * C + c] = src_t < 0 ? state[(span + src_t) * C + c]
                                      : x_r[static_cast<long long>(src_t) * C + c];
        }
    }
    for (int s = 0; s < span; ++s) {
        const int src_t = Nr - span + s;
        const float v = (src_t < 0)
            ? Elem<T>::to_f32(state[(span + src_t) * C + c])
            : Elem<T>::to_f32(x_r[static_cast<long long>(src_t) * C + c]);
        state[s * C + c] = Elem<T>::from_f32(v);
    }
}

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void conv1d_update(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    long long slot_stride_elems,
    T* __restrict__ y,
    int R, int C, int K, int dil,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    const int r = blockIdx.y;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R || c >= C) return;

    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    T* state = state_base + (long long)slot * slot_stride_elems;
    const T* x_r = x + (long long)r_row * C;
    T* y_r = y + (long long)r_row * C;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    const float new_x  = Elem<T>::to_f32(x_r[c]);

    float acc = bias_v;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K) break;
        float xv;
        if (k < K - 1) {
            xv = Elem<T>::to_f32(state[(k * dil + 1) * C + c]);
        } else {
            xv = new_x;
        }
        const float wv = Elem<T>::to_f32(weight[c * K + k]);
        acc += wv * xv;
    }
    y_r[c] = Elem<T>::from_f32(conv_out<SILU, RESIDUAL>(acc, new_x));

    if (!folds(write_state, write_state_mask, rl)) return;
    const int span = (K - 1) * dil;
    for (int k = 0; k < span; ++k) {
        state[k * C + c] = state[(k + 1) * C + c];
    }
    state[span * C + c] = Elem<T>::from_f32(new_x);
}

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void conv1d_update_vec8(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    long long slot_stride_elems,
    T* __restrict__ y,
    int R, int C, int K,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    constexpr int VEC = 8;
    constexpr int K_MAX = 8;
    union Vec { uint4 raw; T e[VEC]; };

    const int r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (r >= R || c0 >= C) return;

    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    T* state = state_base + (long long)slot * slot_stride_elems;
    const int span = K - 1;

    Vec xv;
    xv.raw = *reinterpret_cast<const uint4*>(x + (long long)r_row * C + c0);
    Vec bv;
    bv.raw = make_uint4(0u, 0u, 0u, 0u);
    if (bias != nullptr) {
        bv.raw = *reinterpret_cast<const uint4*>(bias + c0);
    }

    Vec wv[K_MAX];
    const uint4* w = reinterpret_cast<const uint4*>(weight + (long long)c0 * K);
    #pragma unroll
    for (int t = 0; t < K_MAX; ++t) {
        if (t < K) wv[t].raw = w[t];
    }
    Vec window[K_MAX - 1];
    #pragma unroll
    for (int k = 0; k < K_MAX - 1; ++k) {
        if (k < span) {
            window[k].raw = *reinterpret_cast<const uint4*>(state + (long long)(k + 1) * C + c0);
        }
    }
    const T* wflat = &wv[0].e[0];

    float acc[VEC];
    #pragma unroll
    for (int u = 0; u < VEC; ++u) {
        acc[u] = bias != nullptr ? Elem<T>::to_f32(bv.e[u]) : 0.f;
    }
    #pragma unroll
    for (int k = 0; k < K_MAX - 1; ++k) {
        if (k >= span) break;
        #pragma unroll
        for (int u = 0; u < VEC; ++u) {
            acc[u] += Elem<T>::to_f32(wflat[u * K + k]) * Elem<T>::to_f32(window[k].e[u]);
        }
    }
    #pragma unroll
    for (int u = 0; u < VEC; ++u) {
        acc[u] += Elem<T>::to_f32(wflat[u * K + span]) * Elem<T>::to_f32(xv.e[u]);
    }
    Vec yv;
    #pragma unroll
    for (int u = 0; u < VEC; ++u) {
        yv.e[u] = Elem<T>::from_f32(conv_out<SILU, RESIDUAL>(acc[u], Elem<T>::to_f32(xv.e[u])));
    }
    *reinterpret_cast<uint4*>(y + (long long)r_row * C + c0) = yv.raw;

    if (!folds(write_state, write_state_mask, rl)) return;
    #pragma unroll
    for (int k = 0; k < K_MAX - 1; ++k) {
        if (k < span) {
            *reinterpret_cast<uint4*>(state + (long long)k * C + c0) = window[k].raw;
        }
    }
    *reinterpret_cast<uint4*>(state + (long long)span * C + c0) = xv.raw;
}

}

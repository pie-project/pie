#include "elemwise/row.cuh"

namespace pie::elemwise {

template <class T, int BLOCK, bool WEIGHT_PLUS_ONE>
__global__ void residual_add_rmsnorm(
    const T* __restrict__ x,
    T* __restrict__ y,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;
    T* outr = out + static_cast<long long>(plane_row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const T summed = Elem<T>::from_f32(Elem<T>::to_f32(yr[i]) + Elem<T>::to_f32(xr[i]));
        yr[i] = summed;
        const float v = Elem<T>::to_f32(summed);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = Elem<T>::to_f32(yr[i]);
        float wv = Elem<T>::to_f32(weight[i]);
        if constexpr (WEIGHT_PLUS_ONE) wv += 1.f;
        outr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

template <int BLOCK, bool WEIGHT_PLUS_ONE>
__global__ void residual_add_rmsnorm_vec8(
    const bf16* __restrict__ x,
    bf16* __restrict__ y,
    const bf16* __restrict__ weight,
    bf16* __restrict__ out,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const int nvec = hidden / 8;
    const long long base = static_cast<long long>(plane_row) * hidden;

    const uint4* xr = reinterpret_cast<const uint4*>(x + base);
    uint4* yr = reinterpret_cast<uint4*>(y + base);
    uint4* outr = reinterpret_cast<uint4*>(out + base);
    const uint4* wr = reinterpret_cast<const uint4*>(weight);

    float local = 0.f;
    for (int i = tid; i < nvec; i += BLOCK) {
        uint4 yv = yr[i];
        const uint4 xv = xr[i];
        bf16x2* yh = reinterpret_cast<bf16x2*>(&yv);
        const bf16x2* xh = reinterpret_cast<const bf16x2*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = bf16x2_to_f32(yh[j]);
            const float2 b = bf16x2_to_f32(xh[j]);
            yh[j] = f32_to_bf16x2(a.x + b.x, a.y + b.y);
            const float2 f = bf16x2_to_f32(yh[j]);
            local += f.x * f.x + f.y * f.y;
        }
        yr[i] = yv;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < nvec; i += BLOCK) {
        const uint4 yv = yr[i];
        const uint4 wv = wr[i];
        uint4 ov;
        const bf16x2* yh = reinterpret_cast<const bf16x2*>(&yv);
        const bf16x2* wh = reinterpret_cast<const bf16x2*>(&wv);
        bf16x2* oh = reinterpret_cast<bf16x2*>(&ov);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 v = bf16x2_to_f32(yh[j]);
            float2 w = bf16x2_to_f32(wh[j]);
            if constexpr (WEIGHT_PLUS_ONE) {
                w.x += 1.f;
                w.y += 1.f;
            }
            oh[j] = f32_to_bf16x2(v.x * inv_rms * w.x, v.y * inv_rms * w.y);
        }
        outr[i] = ov;
    }
}

// residual_add_rmsnorm_vec8 in one pass: a thread keeps its PER vectors of
// the summed row (and its weight vectors) in registers across the block's
// reduction, all its loads issued up front, so the normed row is written
// without reading the stream back. The sums run in the same order, so it
// lands the same bits.
template <int PER, bool WEIGHT_PLUS_ONE>
__global__ void __launch_bounds__(256) residual_add_rmsnorm_reg(
    const bf16* __restrict__ x,
    bf16* __restrict__ y,
    const bf16* __restrict__ weight,
    bf16* __restrict__ out,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    constexpr int BLOCK = 256;
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const int nvec = hidden / 8;
    const long long base = static_cast<long long>(plane_row) * hidden;

    const uint4* xr = reinterpret_cast<const uint4*>(x + base);
    uint4* yr = reinterpret_cast<uint4*>(y + base);
    uint4* outr = reinterpret_cast<uint4*>(out + base);
    const uint4* wr = reinterpret_cast<const uint4*>(weight);

    uint4 yv[PER], xv[PER], wv[PER];
    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int i = tid + p * BLOCK;
        if (i < nvec) {
            yv[p] = yr[i];
            xv[p] = xr[i];
            wv[p] = wr[i];
        }
    }
    float local = 0.f;
    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int i = tid + p * BLOCK;
        if (i < nvec) {
            bf16x2* yh = reinterpret_cast<bf16x2*>(&yv[p]);
            const bf16x2* xh = reinterpret_cast<const bf16x2*>(&xv[p]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 a = bf16x2_to_f32(yh[j]);
                const float2 b = bf16x2_to_f32(xh[j]);
                yh[j] = f32_to_bf16x2(a.x + b.x, a.y + b.y);
                const float2 f = bf16x2_to_f32(yh[j]);
                local += f.x * f.x + f.y * f.y;
            }
            yr[i] = yv[p];
        }
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int i = tid + p * BLOCK;
        if (i < nvec) {
            uint4 ov;
            const bf16x2* yh = reinterpret_cast<const bf16x2*>(&yv[p]);
            const bf16x2* wh = reinterpret_cast<const bf16x2*>(&wv[p]);
            bf16x2* oh = reinterpret_cast<bf16x2*>(&ov);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 v = bf16x2_to_f32(yh[j]);
                float2 w = bf16x2_to_f32(wh[j]);
                if constexpr (WEIGHT_PLUS_ONE) {
                    w.x += 1.f;
                    w.y += 1.f;
                }
                oh[j] = f32_to_bf16x2(v.x * inv_rms * w.x, v.y * inv_rms * w.y);
            }
            outr[i] = ov;
        }
    }
}

template <class T, int BLOCK, int PER_THREAD, bool SCALE, bool POST, bool POST_PLUS_ONE>
__global__ void rmsnorm_residual_add(
    const T* __restrict__ x,
    const T* __restrict__ w0,
    T* __restrict__ t,
    T* __restrict__ y,
    const T* __restrict__ s,
    T* __restrict__ scaled,
    const T* __restrict__ w1,
    T* __restrict__ out,
    int hidden,
    float eps0,
    float eps1,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const long long base = static_cast<long long>(plane_row) * hidden;

    __shared__ float buf[BLOCK];
    __shared__ float buf2[BLOCK];

    float xv[PER_THREAD];
    float yv[PER_THREAD];
    float local = 0.f;
#pragma unroll
    for (int k = 0; k < PER_THREAD; ++k) {
        const int i = tid + k * BLOCK;
        xv[k] = 0.f;
        yv[k] = 0.f;
        if (i < hidden) {
            xv[k] = Elem<T>::to_f32(x[base + i]);
            yv[k] = Elem<T>::to_f32(y[base + i]);
            local += xv[k] * xv[k];
        }
    }
    const float sum0 = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv0 = rsqrtf(sum0 / static_cast<float>(hidden) + eps0);

    const float sf = SCALE ? Elem<T>::to_f32(s[0]) : 1.f;

    float last[PER_THREAD];
    float local2 = 0.f;
#pragma unroll
    for (int k = 0; k < PER_THREAD; ++k) {
        const int i = tid + k * BLOCK;
        if (i < hidden) {
            const float wv = Elem<T>::to_f32(w0[i]);
            const T tv = Elem<T>::from_f32(xv[k] * inv0 * wv);
            t[base + i] = tv;
            const T folded = Elem<T>::from_f32(yv[k] + Elem<T>::to_f32(tv));
            y[base + i] = folded;
            T final_v = folded;
            if constexpr (SCALE) {
                final_v = Elem<T>::from_f32(Elem<T>::to_f32(folded) * sf);
                scaled[base + i] = final_v;
            }
            last[k] = Elem<T>::to_f32(final_v);
            local2 += last[k] * last[k];
        } else {
            last[k] = 0.f;
        }
    }
    if constexpr (POST) {
        const float sum1 = block_reduce_sum_fast<BLOCK>(local2, buf2);
        const float inv1 = rsqrtf(sum1 / static_cast<float>(hidden) + eps1);
#pragma unroll
        for (int k = 0; k < PER_THREAD; ++k) {
            const int i = tid + k * BLOCK;
            if (i < hidden) {
                float wv = Elem<T>::to_f32(w1[i]);
                if constexpr (POST_PLUS_ONE) wv += 1.f;
                out[base + i] = Elem<T>::from_f32(last[k] * inv1 * wv);
            }
        }
    }
}

// KEEP_T / KEEP_Y off: the normed row t, or the stream y it folds into (in
// place), has no reader past the chain and is not written back.
template <int BLOCK, int CHUNKS, bool SCALE, bool POST, bool POST_PLUS_ONE, bool KEEP_T = true,
          bool KEEP_Y = true>
__global__ __launch_bounds__(BLOCK) void rmsnorm_residual_add_vec8(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w0,
    bf16* __restrict__ t,
    bf16* __restrict__ y,
    const bf16* __restrict__ s,
    bf16* __restrict__ scaled,
    const bf16* __restrict__ w1,
    bf16* __restrict__ out,
    int hidden,
    float eps0,
    float eps1,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const int nvec = hidden / 8;
    const long long base = static_cast<long long>(plane_row) * hidden;

    const uint4* xr = reinterpret_cast<const uint4*>(x + base);
    uint4* yr = reinterpret_cast<uint4*>(y + base);
    uint4* tr = reinterpret_cast<uint4*>(t + base);
    uint4* sr = SCALE ? reinterpret_cast<uint4*>(scaled + base) : nullptr;
    uint4* outr = POST ? reinterpret_cast<uint4*>(out + base) : nullptr;
    const uint4* w0r = reinterpret_cast<const uint4*>(w0);
    const uint4* w1r = POST ? reinterpret_cast<const uint4*>(w1) : nullptr;

    __shared__ float buf[BLOCK / 32 + 1];
    __shared__ float buf2[BLOCK / 32 + 1];

    float xv[CHUNKS][8];
    float yv[CHUNKS][8];
    float local = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int i = tid + c * BLOCK;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            xv[c][j] = 0.f;
            yv[c][j] = 0.f;
        }
        if (i < nvec) {
            const uint4 xq = xr[i];
            const uint4 yq = yr[i];
            const bf16x2* xh = reinterpret_cast<const bf16x2*>(&xq);
            const bf16x2* yh = reinterpret_cast<const bf16x2*>(&yq);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 a = bf16x2_to_f32(xh[j]);
                const float2 b = bf16x2_to_f32(yh[j]);
                xv[c][2 * j] = a.x;
                xv[c][2 * j + 1] = a.y;
                yv[c][2 * j] = b.x;
                yv[c][2 * j + 1] = b.y;
                local += a.x * a.x + a.y * a.y;
            }
        }
    }
    const float sum0 = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv0 = rsqrtf(sum0 / static_cast<float>(hidden) + eps0);
    const float sf = SCALE ? Elem<bf16>::to_f32(s[0]) : 1.f;

    float last[CHUNKS][8];
    float local2 = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int i = tid + c * BLOCK;
#pragma unroll
        for (int j = 0; j < 8; ++j) last[c][j] = 0.f;
        if (i < nvec) {
            const uint4 wq = w0r[i];
            const bf16x2* wh = reinterpret_cast<const bf16x2*>(&wq);
            uint4 tq;
            uint4 yq;
            uint4 sq;
            bf16x2* th = reinterpret_cast<bf16x2*>(&tq);
            bf16x2* yh = reinterpret_cast<bf16x2*>(&yq);
            bf16x2* sh = reinterpret_cast<bf16x2*>(&sq);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 w = bf16x2_to_f32(wh[j]);
                th[j] = f32_to_bf16x2(xv[c][2 * j] * inv0 * w.x, xv[c][2 * j + 1] * inv0 * w.y);
                const float2 tv = bf16x2_to_f32(th[j]);
                yh[j] = f32_to_bf16x2(yv[c][2 * j] + tv.x, yv[c][2 * j + 1] + tv.y);
                float2 fin = bf16x2_to_f32(yh[j]);
                if constexpr (SCALE) {
                    sh[j] = f32_to_bf16x2(fin.x * sf, fin.y * sf);
                    fin = bf16x2_to_f32(sh[j]);
                }
                last[c][2 * j] = fin.x;
                last[c][2 * j + 1] = fin.y;
                local2 += fin.x * fin.x + fin.y * fin.y;
            }
            if constexpr (KEEP_T) tr[i] = tq;
            if constexpr (KEEP_Y) yr[i] = yq;
            if constexpr (SCALE) sr[i] = sq;
        }
    }
    if constexpr (POST) {
        const float sum1 = block_reduce_sum_fast<BLOCK>(local2, buf2);
        const float inv1 = rsqrtf(sum1 / static_cast<float>(hidden) + eps1);
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            const int i = tid + c * BLOCK;
            if (i < nvec) {
                const uint4 wq = w1r[i];
                const bf16x2* wh = reinterpret_cast<const bf16x2*>(&wq);
                uint4 oq;
                bf16x2* oh = reinterpret_cast<bf16x2*>(&oq);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 w = bf16x2_to_f32(wh[j]);
                    if constexpr (POST_PLUS_ONE) {
                        w.x += 1.f;
                        w.y += 1.f;
                    }
                    oh[j] = f32_to_bf16x2(last[c][2 * j] * inv1 * w.x, last[c][2 * j + 1] * inv1 * w.y);
                }
                outr[i] = oq;
            }
        }
    }
}

}

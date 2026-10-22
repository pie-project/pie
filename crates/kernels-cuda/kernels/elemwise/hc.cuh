#pragma once

#include "prim/device.cuh"
#include "elemwise/norm.cuh"
#include "linear/act_quant.cuh"

namespace pie::elemwise {

constexpr int MAX_HC_MULT = 8;

template <int BLOCK = 256>
__global__ void hc_project(
    const float* __restrict__ normed,
    const float* __restrict__ hc_fn,
    float* __restrict__ mixes,
    int fan_in,
    int mix_hc,
    const u32* __restrict__ win)
{
    const int o = static_cast<int>(blockIdx.x) % mix_hc;
    const int n = static_cast<int>(blockIdx.x) / mix_hc;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;
    const float* src = normed + static_cast<long long>(row) * fan_in;
    const float* w = hc_fn + static_cast<long long>(o) * fan_in;

    float local_sum = 0.f;
    for (int d = tid; d < fan_in; d += blockDim.x) {
        local_sum += src[d] * w[d];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float warp_sums[BLOCK / 32];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.f;
        const int nwarps = (blockDim.x + 31) / 32;
        for (int i = 0; i < nwarps; ++i) total += warp_sums[i];
        mixes[static_cast<long long>(row) * mix_hc + o] = total;
    }
}

template <class T, int BLOCK = 256>
__global__ void hc_gates(
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const T* __restrict__ residual,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    T* __restrict__ layer_input,
    int M,
    int H,
    float hc_eps,
    float hc_post_alpha,
    int sinkhorn_iters,
    bool combined,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;

    // Without a combine (the independent form) the row is the pre and post
    // logits alone, the post gate carries the eps the pre gate does, and the
    // combine is the identity.
    const int mix_hc = combined ? M * 2 + M * M : M * 2;
    const float* row = mixes + static_cast<long long>(plane_row) * mix_hc;

    __shared__ float pre[MAX_HC_MULT];
    __shared__ float post[MAX_HC_MULT];
    __shared__ float comb[MAX_HC_MULT * MAX_HC_MULT];

    if (tid < M) {

        const float logit = row[tid] * scale[0] + base[tid];
        pre[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
    }
    if (tid < M) {

        const float logit = row[M + tid] * scale[1] + base[M + tid];
        post[tid] = 1.f / (1.f + expf(-logit)) * hc_post_alpha + (combined ? 0.f : hc_eps);
        post_mix[static_cast<long long>(plane_row) * M + tid] = post[tid];
    }
    __syncthreads();

    if (!combined) {
        if (tid < M * M) {
            comb_mix[static_cast<long long>(plane_row) * M * M + tid] =
                (tid / M == tid % M) ? 1.f : 0.f;
        }
        const T* res_n = residual + static_cast<long long>(plane_row) * M * H;
        T* out = layer_input + static_cast<long long>(plane_row) * H;
        for (int h = tid; h < H; h += blockDim.x) {
            float acc = 0.f;
            for (int i = 0; i < M; ++i) {
                acc += pre[i] * Elem<T>::to_f32(res_n[i * H + h]);
            }
            out[h] = Elem<T>::from_f32(acc);
        }
        return;
    }

    if (tid < M * M) {
        const float logit = row[2 * M + tid] * scale[2] + base[2 * M + tid];
        comb[tid] = logit;
    }
    __syncthreads();

    if (tid < M) {
        float max_v = -flt_max();
        for (int j = 0; j < M; ++j)
            max_v = fmaxf(max_v, comb[tid * M + j]);
        float sum = 0.f;
        for (int j = 0; j < M; ++j) {
            comb[tid * M + j] = expf(comb[tid * M + j] - max_v);
            sum += comb[tid * M + j];
        }
        for (int j = 0; j < M; ++j)
            comb[tid * M + j] = comb[tid * M + j] / sum + hc_eps;
    }
    __syncthreads();

    if (tid < M) {
        float col_sum = 0.f;
        for (int i = 0; i < M; ++i) col_sum += comb[i * M + tid];
        col_sum += hc_eps;
        for (int i = 0; i < M; ++i)
            comb[i * M + tid] = comb[i * M + tid] / col_sum;
    }
    __syncthreads();

    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {

        if (tid < M) {
            float row_sum = 0.f;
            for (int j = 0; j < M; ++j) row_sum += comb[tid * M + j];
            row_sum += hc_eps;
            for (int j = 0; j < M; ++j)
                comb[tid * M + j] = comb[tid * M + j] / row_sum;
        }
        __syncthreads();

        if (tid < M) {
            float col_sum = 0.f;
            for (int i = 0; i < M; ++i) col_sum += comb[i * M + tid];
            col_sum += hc_eps;
            for (int i = 0; i < M; ++i)
                comb[i * M + tid] = comb[i * M + tid] / col_sum;
        }
        __syncthreads();
    }

    if (tid < M * M) {
        comb_mix[static_cast<long long>(plane_row) * M * M + tid] = comb[tid];
    }
    __syncthreads();

    const T* res_n = residual + static_cast<long long>(plane_row) * M * H;
    T* out = layer_input + static_cast<long long>(plane_row) * H;

    for (int h = tid; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < M; ++i) {
            acc += pre[i] * Elem<T>::to_f32(res_n[i * H + h]);
        }
        out[h] = Elem<T>::from_f32(acc);
    }
}

// The combine's Sinkhorn walk in one warp, lane `i * M + j` holding entry
// (i, j) from its logit `c`: the row softmax plus the eps, then column,
// then (row, column) `iters - 1` times, every sum by shuffles.
__device__ __forceinline__ float hc_sinkhorn(float c, int tid, int M, float hc_eps, int iters) {
    if (M == 4) {
        // Four streams: a row is four adjacent lanes and a column lanes four
        // apart, each sum two butterfly steps.
        auto rows = [](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            return v + __shfl_xor_sync(0xffffffffu, v, 2);
        };
        auto cols = [](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            return v + __shfl_xor_sync(0xffffffffu, v, 8);
        };
        float max_v = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffffu, max_v, 2));
        c = tid < 16 ? expf(c - max_v) : 1.f;
        c = c / rows(c) + hc_eps;
        c = c / (cols(c) + hc_eps);
        for (int iter = 0; iter < iters - 1; ++iter) {
            c = c / (rows(c) + hc_eps);
            c = c / (cols(c) + hc_eps);
        }
        return c;
    }
    const int MM = M * M, i = tid / M, j = tid % M;
    auto at = [&](int lane) { return __shfl_sync(0xffffffffu, c, lane < MM ? lane : 0); };
    float max_v = -flt_max();
    for (int k = 0; k < M; ++k) max_v = fmaxf(max_v, at(i * M + k));
    c = expf(c - max_v);
    float sum = 0.f;
    for (int k = 0; k < M; ++k) sum += at(i * M + k);
    c = c / sum + hc_eps;
    auto col = [&]() {
        float t = 0.f;
        for (int k = 0; k < M; ++k) t += at(k * M + j);
        c = c / (t + hc_eps);
    };
    auto rowf = [&]() {
        float t = 0.f;
        for (int k = 0; k < M; ++k) t += at(i * M + k);
        c = c / (t + hc_eps);
    };
    col();
    for (int iter = 0; iter < iters - 1; ++iter) {
        rowf();
        col();
    }
    return c;
}

// The pre-mixed row eight columns a thread, 16 bytes at a time: runs
// `[v, v1)` stepping `BLOCK`, each the `pre`-weighted sum of the `M` streams.
template <class T, int BLOCK>
__device__ __forceinline__ void hc_premix8(const uint4* __restrict__ r4, uint4* __restrict__ o4,
                                           const float* pre, int M, int runs, int v, int v1) {
    for (; v < v1; v += BLOCK) {
        float acc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.f;
        for (int k = 0; k < M; ++k) {
            const uint4 rw = r4[k * runs + v];
            const T* r8 = reinterpret_cast<const T*>(&rw);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += pre[k] * Elem<T>::to_f32(r8[e]);
        }
        uint4 ow;
        T* o8 = reinterpret_cast<T*>(&ow);
#pragma unroll
        for (int e = 0; e < 8; ++e) o8[e] = Elem<T>::from_f32(acc[e]);
        o4[v] = ow;
    }
}

// `hc_gates`' combined form with the Sinkhorn walk in one warp (an entry a
// lane, `M * M <= 32`, sums by shuffles in the shared walk's order) and the
// row's `H` over `gridDim.y` blocks; block `y == 0` lands the post and
// combine mixes.
template <class T, int BLOCK = 256>
__global__ void __launch_bounds__(BLOCK) hc_gates_warp(
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const T* __restrict__ residual,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    T* __restrict__ layer_input,
    int M,
    int H,
    float hc_eps,
    float hc_post_alpha,
    int sinkhorn_iters,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = threadIdx.x;
    const int mix_hc = M * 2 + M * M;
    const float* row = mixes + static_cast<long long>(plane_row) * mix_hc;
    const bool lands = blockIdx.y == 0;

    __shared__ float pre[MAX_HC_MULT];
    if (tid < 32) {
        if (tid < M) {
            const float logit = row[tid] * scale[0] + base[tid];
            pre[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
            const float pl = row[M + tid] * scale[1] + base[M + tid];
            if (lands)
                post_mix[static_cast<long long>(plane_row) * M + tid] =
                    1.f / (1.f + expf(-pl)) * hc_post_alpha;
        }
        // The combine: lane `i * M + j` holds entry (i, j).
        const int MM = M * M;
        float c = tid < MM ? row[2 * M + tid] * scale[2] + base[2 * M + tid] : 0.f;
        c = hc_sinkhorn(c, tid, M, hc_eps, sinkhorn_iters);
        if (lands && tid < MM)
            comb_mix[static_cast<long long>(plane_row) * MM + tid] = c;
    }
    __syncthreads();

    const T* res_n = residual + static_cast<long long>(plane_row) * M * H;
    T* out = layer_input + static_cast<long long>(plane_row) * H;
    const int per = (H + gridDim.y - 1) / gridDim.y;
    const int h0 = blockIdx.y * per;
    const int h1 = min(h0 + per, H);
    for (int h = h0 + tid; h < h1; h += BLOCK) {
        float acc = 0.f;
        for (int k = 0; k < M; ++k) acc += pre[k] * Elem<T>::to_f32(res_n[k * H + h]);
        out[h] = Elem<T>::from_f32(acc);
    }
}

// `hc_gates_warp` with its halves on blocks of their own: blocks
// `y < gridDim.y - 1` land the pre-mix eight columns a thread, the last
// walks the Sinkhorn. Every value is `hc_gates_warp`'s.
template <class T, int BLOCK = 256>
__global__ void __launch_bounds__(BLOCK) hc_gates_split(
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const T* __restrict__ residual,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    T* __restrict__ layer_input,
    int M,
    int H,
    float hc_eps,
    float hc_post_alpha,
    int sinkhorn_iters,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = threadIdx.x;
    const int mix_hc = M * 2 + M * M;
    const float* row = mixes + static_cast<long long>(plane_row) * mix_hc;

    if (blockIdx.y == gridDim.y - 1) {
        if (tid >= 32) return;
        if (tid < M) {
            const float pl = row[M + tid] * scale[1] + base[M + tid];
            post_mix[static_cast<long long>(plane_row) * M + tid] =
                1.f / (1.f + expf(-pl)) * hc_post_alpha;
        }
        const int MM = M * M;
        float c = tid < MM ? row[2 * M + tid] * scale[2] + base[2 * M + tid] : 0.f;
        c = hc_sinkhorn(c, tid, M, hc_eps, sinkhorn_iters);
        if (tid < MM) comb_mix[static_cast<long long>(plane_row) * MM + tid] = c;
        return;
    }

    __shared__ float pre[MAX_HC_MULT];
    if (tid < M) {
        const float logit = row[tid] * scale[0] + base[tid];
        pre[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
    }
    __syncthreads();
    const T* res_n = residual + static_cast<long long>(plane_row) * M * H;
    T* out = layer_input + static_cast<long long>(plane_row) * H;
    const int runs = H / 8;
    const int spread = gridDim.y - 1;
    const int per = (runs + spread - 1) / spread;
    const int v0 = blockIdx.y * per;
    const int v1 = min(v0 + per, runs);
    const uint4* r4 = reinterpret_cast<const uint4*>(res_n);
    uint4* o4 = reinterpret_cast<uint4*>(out);
    hc_premix8<T, BLOCK>(r4, o4, pre, M, runs, v0 + tid, v1);
}

// The combined gates and the RMS norm over the pre-mixed row in one launch:
// blocks `y == 1` walk the Sinkhorn as `hc_gates_split` does; blocks
// `y == 0` pre-mix the row (`PER` columns a thread, rounded as the gates
// land it) and norm it as `rmsnorm` does. Every value is the pair's.
template <class T, int PER, bool PARTIAL, int THREADS = 256, bool EMIT = false>
__global__ void __launch_bounds__(THREADS) hc_gates_norm(
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const T* __restrict__ residual,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int M,
    float hc_eps,
    float hc_post_alpha,
    int sinkhorn_iters,
    float norm_eps,
    int slices,
    int fan_in,
    float mix_eps,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    constexpr int BLOCK = 256;
    constexpr int H = PER * BLOCK;
    static_assert(H % THREADS == 0, "whole columns a thread");
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = threadIdx.x;
    const int mix_hc = M * 2 + M * M;
    // Every load the fold does not feed is issued before it: at a decode they
    // come cold from memory.
    constexpr int COLS = H / THREADS;
    const int MM = M * M;
    float gate_s = 0.f, gate_b = 0.f, comb_s = 0.f, comb_b = 0.f;
    float wv[COLS];
    // The first four streams' columns are held; any past them are read late.
    constexpr int HELD = 4;
    float rv[COLS][HELD];
    if (blockIdx.y == 1) {
        if (tid < M) {
            gate_s = scale[1];
            gate_b = base[M + tid];
        }
        if (tid < MM) {
            comb_s = scale[2];
            comb_b = base[2 * M + tid];
        }
    } else {
        if (tid < M) {
            gate_s = scale[0];
            gate_b = base[tid];
        }
        const T* res_n = residual + static_cast<long long>(plane_row) * M * H;
#pragma unroll
        for (int p = 0; p < COLS; ++p) {
            const int h = tid + p * THREADS;
            wv[p] = Elem<T>::to_f32(weight[h]);
#pragma unroll
            for (int k = 0; k < HELD; ++k)
                rv[p][k] = k < M ? Elem<T>::to_f32(res_n[k * H + h]) : 0.f;
        }
    }
    // Under `PARTIAL` the row's mixes are the projection's slices
    // (`mixes[slice][row][mix_hc + 1]`, the dots then the sum of squares),
    // summed as `hc_prenorm_project_fold` sums them: a warp a mix, its lanes
    // over the slices, the dots scaled by the norm.
    __shared__ float folded[2 * MAX_HC_MULT + MAX_HC_MULT * MAX_HC_MULT + 1];
    const float* row = mixes + static_cast<long long>(plane_row) * mix_hc;
    if constexpr (PARTIAL) {
        const int warp = tid >> 5;
        const int lane = tid & 31;
        const int w = mix_hc + 1;
        // A warp a slice group (slices `g`, `g + 32`, ...), its lanes the run's
        // adjacent values; the groups' sums then fold in a column warp's order.
        if (THREADS == 1024 && w <= 32) {
            __shared__ float red[32][33];
            float sum = 0.f;
            if (lane < w)
                for (int s = warp; s < slices; s += 32)
                    sum += mixes[(static_cast<long long>(s) * gridDim.x + n) * w + lane];
            red[warp][lane] = sum;
            __syncthreads();
            if (warp < w) {
                float v = red[lane][warp];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
                if (lane == 0) folded[warp] = v;
            }
        } else {
            for (int c = warp; c <= mix_hc; c += THREADS / 32) {
                float sum = 0.f;
                for (int s = lane; s < slices; s += 32)
                    sum += mixes[(static_cast<long long>(s) * gridDim.x + n) * w + c];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
                if (lane == 0) folded[c] = sum;
            }
        }
        __syncthreads();
        const float sq = folded[mix_hc];
        __syncthreads();
        if (tid < mix_hc) folded[tid] = folded[tid] * rsqrtf(sq / static_cast<float>(fan_in) + mix_eps);
        __syncthreads();
        row = folded;
    }

    if (blockIdx.y == 1) {
        if (tid >= 32) return;
        if (tid < M) {
            const float pl = row[M + tid] * gate_s + gate_b;
            post_mix[static_cast<long long>(plane_row) * M + tid] =
                1.f / (1.f + expf(-pl)) * hc_post_alpha;
        }
        float c = tid < MM ? row[2 * M + tid] * comb_s + comb_b : 0.f;
        c = hc_sinkhorn(c, tid, M, hc_eps, sinkhorn_iters);
        if (tid < MM) comb_mix[static_cast<long long>(plane_row) * MM + tid] = c;
        return;
    }

    __shared__ float pre[MAX_HC_MULT];
    __shared__ float buf[BLOCK / 32 + 1];
    __shared__ float xsm[H];
    if (tid < M) {
        const float logit = row[tid] * gate_s + gate_b;
        pre[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
    }
    __syncthreads();
    // The pre-mixed row `THREADS` columns at a time, each sum in the gates'
    // order rounded to the element, into shared memory.
    const T* res_n = residual + static_cast<long long>(plane_row) * M * H;
#pragma unroll
    for (int p = 0; p < COLS; ++p) {
        const int h = tid + p * THREADS;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < HELD; ++k)
            if (k < M) acc += pre[k] * rv[p][k];
        for (int k = HELD; k < M; ++k) acc += pre[k] * Elem<T>::to_f32(res_n[k * H + h]);
        xsm[h] = Elem<T>::to_f32(Elem<T>::from_f32(acc));
    }
    __syncthreads();
    // The squares as `rmsnorm`'s 256 threads take them (thread `t` its
    // columns `t`, `t + 256`, ...), folded as its block folds them.
    if (tid < BLOCK) {
        float local = 0.f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const float v = xsm[tid + p * BLOCK];
            local += v * v;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
        if ((tid & 31) == 0) buf[tid >> 5] = local;
    }
    __syncthreads();
    if (tid < 32) {
        float v = tid < BLOCK / 32 ? buf[tid] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (tid == 0) buf[BLOCK / 32] = v;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(buf[BLOCK / 32] / static_cast<float>(H) + norm_eps);
    T* yr = y + static_cast<long long>(plane_row) * H;
#pragma unroll
    for (int p = 0; p < COLS; ++p) {
        const int h = tid + p * THREADS;
        yr[h] = Elem<T>::from_f32(xsm[h] * inv_rms * wv[p]);
    }
    // Under `EMIT` the normed row also lands as the e4m3 1x128 operand a
    // GEMM's quantize would make of it (codes row-major, ue8m0 atoms): a
    // warp 256 columns, a lane eight, a block's 16 lanes adjacent.
    if constexpr (EMIT) {
        static_assert(sizeof(T) == 2 && H % 256 == 0, "whole 128-wide blocks");
        const int lane = tid & 31;
        for (int chunk = tid >> 5; chunk < H / 256; chunk += THREADS / 32) {
            const int h0 = chunk * 256 + lane * 8;
            float x8[8];
#pragma unroll
            for (int e = 0; e < 8; ++e)
                x8[e] = xsm[h0 + e] * inv_rms * Elem<T>::to_f32(weight[h0 + e]);
            pie::linear::e4m3_block_store(x8, lane, plane_row, h0 / 128, H, 0, codes, atoms);
        }
    }
}

template <class T>
__global__ void hc_fold(
    const T* __restrict__ x,
    const T* residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    T* out,
    int N,
    int M,
    int H,
    const u32* __restrict__ win)
{
    if (M > MAX_HC_MULT) return;
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(N) * H) return;

    const int h = static_cast<int>(idx % H);
    const int n = static_cast<int>(idx / H);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const float* comb_n = comb_mix + static_cast<long long>(row) * M * M;
    const float* post_n = post_mix + static_cast<long long>(row) * M;
    const float x_h = Elem<T>::to_f32(x[static_cast<long long>(row) * H + h]);
    const T* res_n = residual + static_cast<long long>(row) * M * H;

    float r[MAX_HC_MULT];
    for (int i = 0; i < M; ++i) {
        r[i] = Elem<T>::to_f32(res_n[static_cast<long long>(i) * H + h]);
    }

    T* out_n = out + static_cast<long long>(row) * M * H;

    for (int j = 0; j < M; ++j) {
        float acc = post_n[j] * x_h;
        for (int i = 0; i < M; ++i) {
            acc += comb_n[i * M + j] * r[i];
        }
        out_n[static_cast<long long>(j) * H + h] = Elem<T>::from_f32(acc);
    }
}

// One 16-byte run of `hc_fold_vec8`: every load (the streams, the post and
// combine gates) ahead of the first store, `MM` streams unrolled.
template <class T, int MM>
__device__ __forceinline__ void hc_fold_run(
    const T* x8,
    const uint4* res_n,
    const float* post_n,
    const float* comb_n,
    uint4* out_n,
    unsigned runs,
    int v,
    int M = MM)
{
    float r[MM][8];
    float p[MM];
    float c[MM][MM];
#pragma unroll
    for (int i = 0; i < MM; ++i) {
        if (i < M) {
            const uint4 rw = res_n[static_cast<long long>(i) * runs + v];
            const T* r8 = reinterpret_cast<const T*>(&rw);
#pragma unroll
            for (int e = 0; e < 8; ++e) r[i][e] = Elem<T>::to_f32(r8[e]);
            p[i] = post_n[i];
        }
#pragma unroll
        for (int j = 0; j < MM; ++j) c[i][j] = (i < M && j < M) ? comb_n[i * M + j] : 0.f;
    }
    float xf[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) xf[e] = Elem<T>::to_f32(x8[e]);
#pragma unroll
    for (int j = 0; j < MM; ++j) {
        if (j >= M) break;
        uint4 ow;
        T* o8 = reinterpret_cast<T*>(&ow);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float acc = p[j] * xf[e];
#pragma unroll
            for (int i = 0; i < MM; ++i) {
                if (i < M) acc += c[i][j] * r[i][e];
            }
            o8[e] = Elem<T>::from_f32(acc);
        }
        out_n[static_cast<long long>(j) * runs + v] = ow;
    }
}

// `hc_fold` eight columns a thread, 16 bytes at a time: the same sums in
// the same order, the row's gate matrices read once a thread.
template <class T>
__global__ void __launch_bounds__(256) hc_fold_vec8(
    const T* __restrict__ x,
    const T* residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    T* out,
    int N,
    int M,
    int H,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    static_assert(sizeof(T) == 2, "eight elements are one 16-byte run");
    if (M > MAX_HC_MULT) return;
    // 32-bit index arithmetic: the host keeps `N * H / 8` under 2^31.
    const unsigned runs = static_cast<unsigned>(H) / 8u;
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<unsigned>(N) * runs) return;
    const int v = static_cast<int>(idx % runs);
    const int n = static_cast<int>(idx / runs);
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const float* comb_n = comb_mix + static_cast<long long>(row) * M * M;
    const float* post_n = post_mix + static_cast<long long>(row) * M;
    const uint4 xw = reinterpret_cast<const uint4*>(x + static_cast<long long>(row) * H)[v];
    const T* x8 = reinterpret_cast<const T*>(&xw);
    const uint4* res_n = reinterpret_cast<const uint4*>(residual + static_cast<long long>(row) * M * H);
    uint4* out_n = reinterpret_cast<uint4*>(out + static_cast<long long>(row) * M * H);

    if (M == 4) {
        hc_fold_run<T, 4>(x8, res_n, post_n, comb_n, out_n, runs, v);
    } else {
        hc_fold_run<T, MAX_HC_MULT>(x8, res_n, post_n, comb_n, out_n, runs, v, M);
    }
}

template <class T, int BLOCK = 256>
__global__ void hc_head_postprocess(
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const T* __restrict__ residual,
    T* __restrict__ out,
    int M,
    int H,
    float hc_eps,
    const u32* __restrict__ win)
{
    const int block_row = blockIdx.x;

    if (win != nullptr && block_row >= static_cast<int>(win[0])) return;
    const int n = win != nullptr ? block_row + static_cast<int>(win[1]) : block_row;
    const int tid = threadIdx.x;

    __shared__ float gates[MAX_HC_MULT];

    if (tid < M) {
        const float logit = mixes[static_cast<long long>(n) * M + tid] * scale[0] + base[tid];
        gates[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
    }
    __syncthreads();

    const T* res_n = residual + static_cast<long long>(n) * M * H;
    T* out_n = out + static_cast<long long>(n) * H;

    for (int h = tid; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < M; ++i) {
            acc += gates[i] * Elem<T>::to_f32(res_n[i * H + h]);
        }
        out_n[h] = Elem<T>::from_f32(acc);
    }
}

template <class T>
__global__ void hc_expand(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N,
    int M,
    int H,
    const u32* __restrict__ win)
{
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(N) * H) return;
    const int n = static_cast<int>(idx / H);
    const int h = static_cast<int>(idx % H);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T val = input[static_cast<long long>(row) * H + h];
    for (int m = 0; m < M; ++m) {
        output[static_cast<long long>(row) * M * H + m * H + h] = val;
    }
}

template <class T, int BLOCK = 256>
__global__ void hc_rmsnorm_f32(
    const T* __restrict__ input,
    float* __restrict__ output,
    int dim,
    float eps,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;
    const T* row = input + static_cast<long long>(plane_row) * dim;
    float* out = output + static_cast<long long>(plane_row) * dim;

    float local_sum = 0.f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = Elem<T>::to_f32(row[d]);
        local_sum += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float warp_sums[BLOCK / 32];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = local_sum;
    __syncthreads();

    __shared__ float scale;
    if (tid == 0) {
        float total = 0.f;
        const int nwarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < nwarps; ++w) total += warp_sums[w];
        scale = rsqrtf(total / dim + eps);
    }
    __syncthreads();

    const float s = scale;
    for (int d = tid; d < dim; d += blockDim.x) {
        out[d] = Elem<T>::to_f32(row[d]) * s;
    }
}

// `hc_rmsnorm_f32` eight elements a thread (one 16-byte load, two 16-byte
// stores), a 1024-thread block a row: a 16384-wide stream row is two loads a
// thread rather than 64 serial ones. `dim` a whole number of eight.
template <class T, int BLOCK>
__global__ void __launch_bounds__(BLOCK) hc_rmsnorm_f32_vec8(
    const T* __restrict__ input,
    float* __restrict__ output,
    int dim,
    float eps,
    const u32* __restrict__ win)
{
    static_assert(sizeof(T) == 2, "eight elements are one 16-byte load");
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = threadIdx.x;
    const uint4* row = reinterpret_cast<const uint4*>(input + static_cast<long long>(plane_row) * dim);
    float4* out = reinterpret_cast<float4*>(output + static_cast<long long>(plane_row) * dim);
    const int vecs = dim / 8;

    float local_sum = 0.f;
    for (int v = tid; v < vecs; v += BLOCK) {
        const uint4 raw = row[v];
        const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float x = Elem<T>::to_f32(e[j]);
            local_sum += x * x;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    __shared__ float warp_sums[BLOCK / 32];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = local_sum;
    __syncthreads();
    __shared__ float scale;
    if (tid < 32) {
        float t = tid < BLOCK / 32 ? warp_sums[tid] : 0.f;
        for (int offset = 16; offset > 0; offset >>= 1)
            t += __shfl_down_sync(0xFFFFFFFF, t, offset);
        if (tid == 0) scale = rsqrtf(t / dim + eps);
    }
    __syncthreads();
    const float sc = scale;
    for (int v = tid; v < vecs; v += BLOCK) {
        const uint4 raw = row[v];
        const T* e = reinterpret_cast<const T*>(&raw);
        out[2 * v] = make_float4(Elem<T>::to_f32(e[0]) * sc, Elem<T>::to_f32(e[1]) * sc,
                                 Elem<T>::to_f32(e[2]) * sc, Elem<T>::to_f32(e[3]) * sc);
        out[2 * v + 1] = make_float4(Elem<T>::to_f32(e[4]) * sc, Elem<T>::to_f32(e[5]) * sc,
                                     Elem<T>::to_f32(e[6]) * sc, Elem<T>::to_f32(e[7]) * sc);
    }
}

template <class T, int BLOCK = 256>
__global__ void hc_mix(
    const T* __restrict__ gates,
    const T* __restrict__ normed,
    T* __restrict__ y,
    int M,
    int H,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;

    const T* gr = gates + static_cast<long long>(row) * M * H;
    const T* nr = normed + static_cast<long long>(row) * M * H;
    T* yr = y + static_cast<long long>(row) * H;

    for (int h = tid; h < H; h += BLOCK) {
        float acc = 0.f;
        for (int s = 0; s < M; ++s) {
            const float g = Elem<T>::to_f32(gr[s * H + h]);
            const float v = Elem<T>::to_f32(nr[s * H + h]);
            acc += v / (1.f + __expf(-g));
        }
        yr[h] = Elem<T>::from_f32(acc / static_cast<float>(M));
    }
}

// `hc_mix` eight columns a thread, read and written 16 bytes at a time: the
// same sums, element for element. A row's columns spread over `gridDim.y`
// blocks, so a narrow fire still fills the SMs.
template <class T, int BLOCK = 256>
__global__ void hc_mix_vec8(
    const T* __restrict__ gates,
    const T* __restrict__ normed,
    T* __restrict__ y,
    int M,
    int H,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;

    const T* gr = gates + static_cast<long long>(row) * M * H;
    const T* nr = normed + static_cast<long long>(row) * M * H;
    T* yr = y + static_cast<long long>(row) * H;

    for (int h0 = (blockIdx.y * BLOCK + tid) * 8; h0 < H; h0 += gridDim.y * BLOCK * 8) {
        float acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
        for (int s = 0; s < M; ++s) {
            uint4 gw = *reinterpret_cast<const uint4*>(gr + s * H + h0);
            uint4 nw = *reinterpret_cast<const uint4*>(nr + s * H + h0);
            const T* g8 = reinterpret_cast<const T*>(&gw);
            const T* v8 = reinterpret_cast<const T*>(&nw);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float g = Elem<T>::to_f32(g8[i]);
                const float v = Elem<T>::to_f32(v8[i]);
                acc[i] += v / (1.f + __expf(-g));
            }
        }
        uint4 out;
        T* o8 = reinterpret_cast<T*>(&out);
#pragma unroll
        for (int i = 0; i < 8; ++i) o8[i] = Elem<T>::from_f32(acc[i] / static_cast<float>(M));
        *reinterpret_cast<uint4*>(yr + h0) = out;
    }
}

template <class T, int BLOCK = 256>
__global__ void hc_inject(
    const T* __restrict__ o,
    const T* __restrict__ gates,
    T* __restrict__ hyper,
    int M,
    int H,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;

    const T* orow = o + static_cast<long long>(row) * H;
    const T* gr = gates + static_cast<long long>(row) * M;
    T* hr = hyper + static_cast<long long>(row) * M * H;

    __shared__ float g[MAX_HC_MULT];
    if (tid < M) {
        const float logit = Elem<T>::to_f32(gr[tid]) / static_cast<float>(M);
        g[tid] = 2.f / (1.f + __expf(-logit));
    }
    __syncthreads();

    const int total = M * H;
    for (int i = tid; i < total; i += BLOCK) {
        const int s = i / H;
        const int h = i - s * H;
        const float v = Elem<T>::to_f32(hr[i]) + g[s] * Elem<T>::to_f32(orow[h]);
        hr[i] = Elem<T>::from_f32(v);
    }
}

// `hc_inject` eight columns a thread, 16 bytes at a time: the same sums. A
// row's streams spread over `gridDim.y` blocks.
template <class T, int BLOCK = 256>
__global__ void hc_inject_vec8(
    const T* __restrict__ o,
    const T* __restrict__ gates,
    T* __restrict__ hyper,
    int M,
    int H,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;

    const T* orow = o + static_cast<long long>(row) * H;
    const T* gr = gates + static_cast<long long>(row) * M;
    T* hr = hyper + static_cast<long long>(row) * M * H;

    __shared__ float g[MAX_HC_MULT];
    if (tid < M) {
        const float logit = Elem<T>::to_f32(gr[tid]) / static_cast<float>(M);
        g[tid] = 2.f / (1.f + __expf(-logit));
    }
    __syncthreads();

    const int total = M * H;
    for (int i0 = (blockIdx.y * BLOCK + tid) * 8; i0 < total; i0 += gridDim.y * BLOCK * 8) {
        const int s = i0 / H;
        const int h0 = i0 - s * H;
        uint4 hw = *reinterpret_cast<const uint4*>(hr + i0);
        const uint4 ow = *reinterpret_cast<const uint4*>(orow + h0);
        T* h8 = reinterpret_cast<T*>(&hw);
        const T* o8 = reinterpret_cast<const T*>(&ow);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            h8[i] = Elem<T>::from_f32(Elem<T>::to_f32(h8[i]) + g[s] * Elem<T>::to_f32(o8[i]));
        }
        *reinterpret_cast<uint4*>(hr + i0) = hw;
    }
}

template <class T, int BLOCK = 256>
__global__ void ple_gate(
    const T* __restrict__ key,
    const T* __restrict__ query,
    const T* __restrict__ value,
    T* __restrict__ y,
    int M,
    int H,
    const u32* __restrict__ win)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int n = b / M;
    const int s = b - n * M;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T* kr = key + (static_cast<long long>(row) * M + s) * H;
    const T* qr = query + (static_cast<long long>(row) * M + s) * H;
    const T* vr = value + static_cast<long long>(row) * H;
    T* yr = y + (static_cast<long long>(row) * M + s) * H;

    float local = 0.f;
    for (int i = tid; i < H; i += BLOCK) {
        local += Elem<T>::to_f32(kr[i]) * Elem<T>::to_f32(qr[i]);
    }
    __shared__ float buf[BLOCK];
    const float dot = block_reduce_sum_fast<BLOCK>(local, buf) * rsqrtf((float)H);

    float damped = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    damped = dot > 0.f ? damped : (dot < 0.f ? -damped : 0.f);
    const float gate = 1.f / (1.f + __expf(-damped));

    for (int i = tid; i < H; i += BLOCK) {
        yr[i] = Elem<T>::from_f32(gate * Elem<T>::to_f32(vr[i]));
    }
}

// `hc_project` a block `kRows` rows of one of `gridDim.y` column slices: a
// warp a mix column at a time, a row read once for every column. Several
// slices land `partial[slice][row][col]` for `hc_project_fold`.
template <int kRows, int kCols>
__global__ void __launch_bounds__(256) hc_project_split(
    const float* __restrict__ normed,
    const float* __restrict__ hc_fn,
    float* __restrict__ out,
    int fan_in,
    int mix_hc,
    int rows,
    int slice,
    const u32* __restrict__ win)
{
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int n0 = static_cast<int>(blockIdx.x) * kRows;
    const int live = win != nullptr ? min(static_cast<int>(win[0]), rows) : rows;
    if (n0 >= live) return;
    const int base = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int d0 = static_cast<int>(blockIdx.y) * slice;
    const int d1 = min(d0 + slice, fan_in);

    const float4* xr[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r)
        xr[r] = reinterpret_cast<const float4*>(
            normed + static_cast<long long>(base + min(n0 + r, live - 1)) * fan_in);
    const float4* wc[kCols];
    bool on[kCols];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        const int o = warp + 8 * c;
        on[c] = o < mix_hc;
        wc[c] = reinterpret_cast<const float4*>(hc_fn + static_cast<long long>(on[c] ? o : 0) * fan_in);
    }
    float acc[kRows][kCols];
#pragma unroll
    for (int r = 0; r < kRows; ++r)
#pragma unroll
        for (int c = 0; c < kCols; ++c) acc[r][c] = 0.f;
    for (int v = d0 / 4 + lane; v < d1 / 4; v += 32) {
        float4 x[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) x[r] = xr[r][v];
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (!on[c]) continue;
            const float4 w = wc[c][v];
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                acc[r][c] += x[r].x * w.x + x[r].y * w.y + x[r].z * w.z + x[r].w * w.w;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int r = 0; r < kRows; ++r)
#pragma unroll
            for (int c = 0; c < kCols; ++c)
                acc[r][c] += __shfl_xor_sync(0xffffffffu, acc[r][c], off);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int n = n0 + r;
            if (n >= live) break;
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                if (!on[c]) continue;
                const int o = warp + 8 * c;
                if (gridDim.y == 1) {
                    out[static_cast<long long>(base + n) * mix_hc + o] = acc[r][c];
                } else {
                    out[(static_cast<long long>(blockIdx.y) * rows + n) * mix_hc + o] = acc[r][c];
                }
            }
        }
    }
}

// The slices' partial projections summed in slice order into `mixes`.
__global__ void hc_project_fold(
    const float* __restrict__ partial,
    float* __restrict__ mixes,
    int mix_hc,
    int rows,
    int slices,
    const u32* __restrict__ win)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int live = win != nullptr ? min(static_cast<int>(win[0]), rows) : rows;
    if (i >= live * mix_hc) return;
    const int base = win != nullptr ? static_cast<int>(win[1]) : 0;
    float sum = 0.f;
    for (int s = 0; s < slices; ++s) sum += partial[static_cast<long long>(s) * rows * mix_hc + i];
    mixes[static_cast<long long>(base) * mix_hc + i] = sum;
}

// `hc_project` tiled: a block `BR` rows of every mix column (up to 24) over
// a slice of the row, staged `KT` floats at a time by `cp.async` (`STAGES`
// deep); each thread four rows by six columns, the quads split over `KG`
// groups summed in group order. Several slices land `partial` for
// `hc_project_fold`.
__device__ __forceinline__ void hc_cp16(void* smem, const void* gmem)
{
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(at), "l"(gmem) : "memory");
}

template <int BR, int KG, int KT, int STAGES>
__global__ void __launch_bounds__(BR * KG) hc_project_tile(
    const float* __restrict__ normed,
    const float* __restrict__ hc_fn,
    float* __restrict__ out,
    int fan_in,
    int mix_hc,
    int rows,
    int slice,
    const u32* __restrict__ win)
{
    constexpr int S = KT + 4;
    constexpr int NT = BR * KG;
    constexpr int COLS = 24;
    constexpr int Q = KT / 4;
    constexpr int XC = BR * Q;
    constexpr int WC = COLS * Q;
    static_assert(BR % 32 == 0 && Q % KG == 0, "whole quads a group");
    static_assert(STAGES * (BR + COLS) * S >= (KG > 1 ? (KG - 1) * BR * COLS : 0), "the fold fits the tiles");
    extern __shared__ __align__(16) float smem[];
    float* xs = smem;
    float* ws = smem + STAGES * BR * S;

    const int tid = static_cast<int>(threadIdx.x);
    const int kg = tid / BR;
    const int t = tid % BR;
    const int tr = t >> 2;
    const int tc = t & 3;
    const int n0 = static_cast<int>(blockIdx.x) * BR;
    const int live = win != nullptr ? min(static_cast<int>(win[0]), rows) : rows;
    if (n0 >= live) return;
    const int base = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int d0 = static_cast<int>(blockIdx.y) * slice;
    const int d1 = min(d0 + slice, fan_in);
    const int tiles = (d1 - d0) / KT;

    auto issue = [&](int tile) {
        const int k0 = d0 + tile * KT;
        float* xd = xs + (tile % STAGES) * BR * S;
        float* wd = ws + (tile % STAGES) * COLS * S;
        for (int c = tid; c < XC; c += NT) {
            const int row = c / Q;
            const int q = c % Q;
            const int n = min(n0 + row, live - 1);
            hc_cp16(xd + row * S + q * 4, normed + static_cast<long long>(base + n) * fan_in + k0 + q * 4);
        }
        for (int c = tid; c < WC; c += NT) {
            const int col = c / Q;
            const int q = c % Q;
            const int o = col < mix_hc ? col : 0;
            hc_cp16(wd + col * S + q * 4, hc_fn + static_cast<long long>(o) * fan_in + k0 + q * 4);
        }
    };

    float acc[4][6];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 6; ++j) acc[i][j] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < tiles) issue(s);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (int tile = 0; tile < tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");
        __syncthreads();
        if (tile + STAGES - 1 < tiles) issue(tile + STAGES - 1);
        asm volatile("cp.async.commit_group;" ::: "memory");
        const float* xt = xs + (tile % STAGES) * BR * S;
        const float* wt = ws + (tile % STAGES) * COLS * S;
#pragma unroll
        for (int qq = 0; qq < Q / KG; ++qq) {
            const int k = (kg * (Q / KG) + qq) * 4;
            float4 xv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                xv[i] = *reinterpret_cast<const float4*>(xt + (tr + i * (BR / 4)) * S + k);
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                const float4 w = *reinterpret_cast<const float4*>(wt + (tc * 6 + j) * S + k);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float a = acc[i][j];
                    a = fmaf(xv[i].x, w.x, a);
                    a = fmaf(xv[i].y, w.y, a);
                    a = fmaf(xv[i].z, w.z, a);
                    acc[i][j] = fmaf(xv[i].w, w.w, a);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (KG > 1) {
        __syncthreads();
        float* red = smem;
        if (kg > 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 6; ++j)
                    red[((kg - 1) * BR + t) * COLS + i * 6 + j] = acc[i][j];
        }
        __syncthreads();
        if (kg > 0) return;
#pragma unroll
        for (int g = 0; g < KG - 1; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 6; ++j) acc[i][j] += red[(g * BR + t) * COLS + i * 6 + j];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = n0 + tr + i * (BR / 4);
        if (n >= live) continue;
#pragma unroll
        for (int j = 0; j < 6; ++j) {
            const int o = tc * 6 + j;
            if (o >= mix_hc) continue;
            if (gridDim.y == 1) {
                out[static_cast<long long>(base + n) * mix_hc + o] = acc[i][j];
            } else {
                out[(static_cast<long long>(blockIdx.y) * rows + n) * mix_hc + o] = acc[i][j];
            }
        }
    }
}

// The stream norm and the mix projection in one pass over the bf16 streams:
// `mixes = (x . w) * rsqrt(mean(x^2) + eps)`, `hc_project_tile`'s tiles with
// the sums of squares beside the dot. Several slices land
// `partial[slice][row][mix_hc + 1]` for `hc_prenorm_project_fold`.
template <int BR, int KG, int KT, int STAGES>
__global__ void __launch_bounds__(BR * KG) hc_prenorm_project_tile(
    const bf16* __restrict__ streams,
    const float* __restrict__ hc_fn,
    float* __restrict__ out,
    int fan_in,
    int mix_hc,
    int rows,
    int slice,
    float eps,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    constexpr int SX = KT + 8;
    constexpr int SW = KT + 4;
    constexpr int NT = BR * KG;
    constexpr int COLS = 24;
    constexpr int O = KT / 8;
    constexpr int XC = BR * O;
    constexpr int WC = COLS * (KT / 4);
    constexpr int RED = KG > 1 ? (KG - 1) * BR * (COLS + 4) : 0;
    static_assert(BR % 32 == 0 && O % KG == 0, "whole octets a group");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* xs = reinterpret_cast<bf16*>(smem_raw);
    float* ws = reinterpret_cast<float*>(smem_raw + STAGES * BR * SX * sizeof(bf16));

    const int tid = static_cast<int>(threadIdx.x);
    const int kg = tid / BR;
    const int t = tid % BR;
    const int tr = t >> 2;
    const int tc = t & 3;
    const int n0 = static_cast<int>(blockIdx.x) * BR;
    const int live = win != nullptr ? min(static_cast<int>(win[0]), rows) : rows;
    if (n0 >= live) return;
    const int base = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int d0 = static_cast<int>(blockIdx.y) * slice;
    const int d1 = min(d0 + slice, fan_in);
    const int tiles = (d1 - d0) / KT;

    auto issue = [&](int tile) {
        const int k0 = d0 + tile * KT;
        bf16* xd = xs + (tile % STAGES) * BR * SX;
        float* wd = ws + (tile % STAGES) * COLS * SW;
        for (int c = tid; c < XC; c += NT) {
            const int row = c / O;
            const int q = c % O;
            const int n = min(n0 + row, live - 1);
            hc_cp16(xd + row * SX + q * 8, streams + static_cast<long long>(base + n) * fan_in + k0 + q * 8);
        }
        for (int c = tid; c < WC; c += NT) {
            const int col = c / (KT / 4);
            const int q = c % (KT / 4);
            const int o = col < mix_hc ? col : 0;
            hc_cp16(wd + col * SW + q * 4, hc_fn + static_cast<long long>(o) * fan_in + k0 + q * 4);
        }
    };

    float acc[4][6];
    float sq[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sq[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 6; ++j) acc[i][j] = 0.f;
    }

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < tiles) issue(s);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (int tile = 0; tile < tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");
        __syncthreads();
        if (tile + STAGES - 1 < tiles) issue(tile + STAGES - 1);
        asm volatile("cp.async.commit_group;" ::: "memory");
        const bf16* xt = xs + (tile % STAGES) * BR * SX;
        const float* wt = ws + (tile % STAGES) * COLS * SW;
#pragma unroll
        for (int oo = 0; oo < O / KG; ++oo) {
            const int k = (kg * (O / KG) + oo) * 8;
            float xf[4][8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4 raw = *reinterpret_cast<const uint4*>(xt + (tr + i * (BR / 4)) * SX + k);
                const bf16* e = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    xf[i][u] = Elem<bf16>::to_f32(e[u]);
                    sq[i] = fmaf(xf[i][u], xf[i][u], sq[i]);
                }
            }
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                const float4 wa = *reinterpret_cast<const float4*>(wt + (tc * 6 + j) * SW + k);
                const float4 wb = *reinterpret_cast<const float4*>(wt + (tc * 6 + j) * SW + k + 4);
                const float wf[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float a = acc[i][j];
#pragma unroll
                    for (int u = 0; u < 8; ++u) a = fmaf(xf[i][u], wf[u], a);
                    acc[i][j] = a;
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (KG > 1) {
        __syncthreads();
        float* red = reinterpret_cast<float*>(smem_raw);
        if (kg > 0) {
            float* mine = red + ((kg - 1) * BR + t) * (COLS + 4);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 6; ++j) mine[i * 6 + j] = acc[i][j];
                mine[COLS + i] = sq[i];
            }
        }
        __syncthreads();
        if (kg > 0) return;
#pragma unroll
        for (int g = 0; g < KG - 1; ++g) {
            const float* theirs = red + (g * BR + t) * (COLS + 4);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 6; ++j) acc[i][j] += theirs[i * 6 + j];
                sq[i] += theirs[COLS + i];
            }
        }
    }
    static_cast<void>(RED);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = n0 + tr + i * (BR / 4);
        if (n >= live) continue;
        if (gridDim.y == 1) {
            const float s = rsqrtf(sq[i] / static_cast<float>(fan_in) + eps);
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                const int o = tc * 6 + j;
                if (o < mix_hc) out[static_cast<long long>(base + n) * mix_hc + o] = acc[i][j] * s;
            }
        } else {
            float* p = out + (static_cast<long long>(blockIdx.y) * rows + n) * (mix_hc + 1);
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                const int o = tc * 6 + j;
                if (o < mix_hc) p[o] = acc[i][j];
            }
            if (tc == 0) p[mix_hc] = sq[i];
        }
    }
}

// The slices' partial dots and sums of squares summed and the dots scaled
// by the row's norm: a warp a mix, its lanes over the slices.
__global__ void __launch_bounds__(256) hc_prenorm_project_fold(
    const float* __restrict__ partial,
    float* __restrict__ mixes,
    int mix_hc,
    int rows,
    int slices,
    int fan_in,
    float eps,
    const u32* __restrict__ win)
{
    const int i = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int live = win != nullptr ? min(static_cast<int>(win[0]), rows) : rows;
    if (i >= live * mix_hc) return;
    const int n = i / mix_hc;
    const int o = i % mix_hc;
    const int base = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int w = mix_hc + 1;
    float dot = 0.f, sq = 0.f;
    for (int s = lane; s < slices; s += 32) {
        const float* p = partial + (static_cast<long long>(s) * rows + n) * w;
        dot += p[o];
        sq += p[mix_hc];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        dot += __shfl_xor_sync(0xffffffffu, dot, off);
        sq += __shfl_xor_sync(0xffffffffu, sq, off);
    }
    if (lane == 0)
        mixes[static_cast<long long>(base + n) * mix_hc + o] =
            dot * rsqrtf(sq / static_cast<float>(fan_in) + eps);
}

// `hc_fn` (f32, [mix_hc][fan_in]) as two bf16 planes summing to it, for the
// tensor-core projection: rows [0, 24) the high halves, [24, 48) what each
// leaves, rows past mix_hc zero. A thread four columns of a row.
__global__ void __launch_bounds__(256) hc_fn_split(
    const float* __restrict__ hc_fn,
    bf16* __restrict__ split,
    int fan_in,
    int mix_hc)
{
    const long long i = (static_cast<long long>(blockIdx.x) * 256 + threadIdx.x) * 4;
    if (i >= 24LL * fan_in) return;
    const int o = static_cast<int>(i / fan_in);
    const int k = static_cast<int>(i % fan_in);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float w = o < mix_hc ? hc_fn[static_cast<long long>(o) * fan_in + k + u] : 0.f;
        const bf16 hi = Elem<bf16>::from_f32(w);
        split[static_cast<long long>(o) * fan_in + k + u] = hi;
        split[static_cast<long long>(o + 24) * fan_in + k + u] = Elem<bf16>::from_f32(w - Elem<bf16>::to_f32(hi));
    }
}

// `hc_prenorm_project_tile` on mma.sync bf16: the weight's high and low
// bf16 halves (16 bits of each weight, summed in f32) are the mma's 48 rows,
// a warp's 8 NT stream rows its columns; a row's sum of squares comes off its
// B fragments. Several slices land `partial` as the tiled form does.
template <int STAGES, int NT>
__global__ void __launch_bounds__(128) hc_prenorm_project_mma_t(
    const bf16* __restrict__ streams,
    const bf16* __restrict__ split,
    float* __restrict__ out,
    int fan_in,
    int mix_hc,
    int rows,
    int slice,
    float eps,
    const u32* __restrict__ win)
{
    constexpr int BR = 4 * 8 * NT;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;
    constexpr int NC = 48;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* xs = reinterpret_cast<bf16*>(smem_raw);
    bf16* ws = xs + STAGES * BR * LD;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n0 = static_cast<int>(blockIdx.x) * BR;
    const int live = win != nullptr ? min(static_cast<int>(win[0]), rows) : rows;
    if (n0 >= live) return;
    const int base = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int d0 = static_cast<int>(blockIdx.y) * slice;
    const int d1 = min(d0 + slice, fan_in);
    const int tiles = (d1 - d0) / KT;

    auto issue = [&](int tile) {
        const int k0 = d0 + tile * KT;
        bf16* xd = xs + (tile % STAGES) * BR * LD;
        bf16* wd = ws + (tile % STAGES) * NC * LD;
        for (int c = tid; c < BR * (KT / 8); c += 128) {
            const int row = c / (KT / 8);
            const int q = c % (KT / 8);
            const int n = min(n0 + row, live - 1);
            hc_cp16(xd + row * LD + q * 8, streams + static_cast<long long>(base + n) * fan_in + k0 + q * 8);
        }
        for (int c = tid; c < NC * (KT / 8); c += 128) {
            const int col = c / (KT / 8);
            const int q = c % (KT / 8);
            hc_cp16(wd + col * LD + q * 8, split + static_cast<long long>(col) * fan_in + k0 + q * 8);
        }
    };

    float acc[3][NT][4];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;
    float sq[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) sq[nt] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < tiles) issue(s);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (int tile = 0; tile < tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");
        __syncthreads();
        if (tile + STAGES - 1 < tiles) issue(tile + STAGES - 1);
        asm volatile("cp.async.commit_group;" ::: "memory");
        const bf16* xt = xs + (tile % STAGES) * BR * LD + warp * 8 * NT * LD;
        const bf16* wt = ws + (tile % STAGES) * NC * LD;
#pragma unroll
        for (int k0 = 0; k0 < KT; k0 += 16) {
            u32 a[3][4];
#pragma unroll
            for (int mt = 0; mt < 3; ++mt) {
                const bf16* rp = wt + (mt * 16 + (lane & 15)) * LD + k0 + ((lane & 16) ? 8 : 0);
                const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(rp));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(a[mt][0]), "=r"(a[mt][1]), "=r"(a[mt][2]), "=r"(a[mt][3])
                             : "r"(at));
            }
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                // Two row tiles a load: lanes 0-7 / 8-15 tile 2np at k0 / k0 + 8,
                // lanes 16-23 / 24-31 tile 2np + 1.
                u32 b[4];
                const bf16* rp = xt + ((2 * np + (lane >> 4)) * 8 + (lane & 7)) * LD + k0 + ((lane & 8) ? 8 : 0);
                const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(rp));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"(at));
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int nt = 2 * np + h;
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const float x0 = __uint_as_float(b[2 * h + r] << 16);
                        const float x1 = __uint_as_float(b[2 * h + r] & 0xffff0000u);
                        sq[nt] = fmaf(x0, x0, fmaf(x1, x1, sq[nt]));
                    }
#pragma unroll
                    for (int mt = 0; mt < 3; ++mt) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]),
                              "+f"(acc[mt][nt][3])
                            : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]), "r"(b[2 * h]),
                              "r"(b[2 * h + 1]));
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    // A row tile's row g sum of squares over the quad's lanes.
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        sq[nt] += __shfl_xor_sync(0xffffffffu, sq[nt], 1);
        sq[nt] += __shfl_xor_sync(0xffffffffu, sq[nt], 2);
    }
    const int g = lane >> 2;
    const int pc = (lane & 3) * 2;
    // Half-column o (< 24) and its low half o + 24 in this thread's tiles:
    // o = g (mt 0, i 0-1) with 24 + g (mt 1, i 2-3); 8 + g (mt 0, i 2-3)
    // with 32 + g (mt 2, i 0-1); 16 + g (mt 1, i 0-1) with 40 + g (mt 2, i 2-3).
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int n = n0 + warp * 8 * NT + nt * 8 + pc + c;
            const float rsq = __shfl_sync(0xffffffffu, sq[nt], (pc + c) * 4);
            if (n >= live) continue;
            const float v0 = acc[0][nt][c] + acc[1][nt][2 + c];
            const float v1 = acc[0][nt][2 + c] + acc[2][nt][c];
            const float v2 = acc[1][nt][c] + acc[2][nt][2 + c];
            float* o_row;
            float s = 1.f;
            if (gridDim.y == 1) {
                o_row = out + static_cast<long long>(base + n) * mix_hc;
                s = rsqrtf(rsq / static_cast<float>(fan_in) + eps);
            } else {
                o_row = out + (static_cast<long long>(blockIdx.y) * rows + n) * (mix_hc + 1);
                if (g == 0) o_row[mix_hc] = rsq;
            }
            if (g < mix_hc) o_row[g] = v0 * s;
            if (8 + g < mix_hc) o_row[8 + g] = v1 * s;
            if (16 + g < mix_hc) o_row[16 + g] = v2 * s;
        }
    }
}

// `hc_prenorm_project_fold` for a few slices (a prefill's): a thread a row,
// its slices' dots and sums of squares summed in slice order, the dots
// scaled by the row's norm.
__global__ void __launch_bounds__(256) hc_prenorm_project_fold_rows(
    const float* __restrict__ partial,
    float* __restrict__ mixes,
    int mix_hc,
    int rows,
    int slices,
    int fan_in,
    float eps,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    const int live = win != nullptr ? min(static_cast<int>(win[0]), rows) : rows;
    if (n >= live) return;
    const int base = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int w = mix_hc + 1;
    float sq = 0.f;
    for (int s = 0; s < slices; ++s) sq += partial[(static_cast<long long>(s) * rows + n) * w + mix_hc];
    const float scale = rsqrtf(sq / static_cast<float>(fan_in) + eps);
    for (int o = 0; o < mix_hc; ++o) {
        float dot = 0.f;
        for (int s = 0; s < slices; ++s) dot += partial[(static_cast<long long>(s) * rows + n) * w + o];
        mixes[static_cast<long long>(base + n) * mix_hc + o] = dot * scale;
    }
}

// `hc_fold_vec8` a block a row for a prefill (four streams): the same sums
// in the same order, the next run's loads in flight while one stores.
template <class T>
__global__ void __launch_bounds__(128) hc_fold_rows4(
    const T* __restrict__ x,
    const T* residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    T* out,
    int N,
    int H,
    const u32* __restrict__ win)
{
    static_assert(sizeof(T) == 2, "eight elements are one 16-byte run");
    constexpr int M = 4;
    const int n = static_cast<int>(blockIdx.x);
    if (n >= N) return;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const unsigned runs = static_cast<unsigned>(H) / 8u;
    const float4 pw = *reinterpret_cast<const float4*>(post_mix + static_cast<long long>(row) * M);
    const float p[M] = {pw.x, pw.y, pw.z, pw.w};
    float c[M][M];
#pragma unroll
    for (int i = 0; i < M; ++i) {
        const float4 cw = *reinterpret_cast<const float4*>(comb_mix + static_cast<long long>(row) * M * M + i * M);
        c[i][0] = cw.x;
        c[i][1] = cw.y;
        c[i][2] = cw.z;
        c[i][3] = cw.w;
    }
    const uint4* x_n = reinterpret_cast<const uint4*>(x + static_cast<long long>(row) * H);
    const uint4* res_n = reinterpret_cast<const uint4*>(residual + static_cast<long long>(row) * M * H);
    uint4* out_n = reinterpret_cast<uint4*>(out + static_cast<long long>(row) * M * H);
    for (unsigned v = threadIdx.x; v < runs; v += 128) {
        const uint4 xw = x_n[v];
        uint4 rw[M];
#pragma unroll
        for (int i = 0; i < M; ++i) rw[i] = res_n[static_cast<long long>(i) * runs + v];
        const T* x8 = reinterpret_cast<const T*>(&xw);
        float r[M][8];
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const T* r8 = reinterpret_cast<const T*>(&rw[i]);
#pragma unroll
            for (int e = 0; e < 8; ++e) r[i][e] = Elem<T>::to_f32(r8[e]);
        }
#pragma unroll
        for (int j = 0; j < M; ++j) {
            uint4 ow;
            T* o8 = reinterpret_cast<T*>(&ow);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float acc = p[j] * Elem<T>::to_f32(x8[e]);
#pragma unroll
                for (int i = 0; i < M; ++i) acc += c[i][j] * r[i][e];
                o8[e] = Elem<T>::from_f32(acc);
            }
            out_n[static_cast<long long>(j) * runs + v] = ow;
        }
    }
}

// A prefill's RMS norm of a 4096-wide row that also lands its e4m3 1x128
// operand for the next GEMM (`Run::hold_e4m3`), in one pass.
template <class T>
__global__ void __launch_bounds__(256) rmsnorm_emit_e4m3(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    float eps,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    const u32* __restrict__ win)
{
    static_assert(sizeof(T) == 2, "16 values are two 16-byte runs");
    const int n = static_cast<int>(blockIdx.x);
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int e0 = tid * 16;
    const bool live = e0 < hidden;
    const T* xr = x + static_cast<long long>(row) * hidden;
    float v[16];
    float ss = 0.f;
    if (live) {
        const uint4 a = *reinterpret_cast<const uint4*>(xr + e0);
        const uint4 b = *reinterpret_cast<const uint4*>(xr + e0 + 8);
        const T* pa = reinterpret_cast<const T*>(&a);
        const T* pb = reinterpret_cast<const T*>(&b);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = Elem<T>::to_f32(pa[i]);
            v[8 + i] = Elem<T>::to_f32(pb[i]);
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) ss = fmaf(v[i], v[i], ss);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    __shared__ float warp_ss[8];
    if (lane == 0) warp_ss[tid >> 5] = ss;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += warp_ss[w];
    const float inv = rsqrtf(total / static_cast<float>(hidden) + eps);
    // The normed row, rounded to T as the plane holds it.
    float q[16];
    float amax = 0.f;
    if (live) {
        const uint4 wa = *reinterpret_cast<const uint4*>(weight + e0);
        const uint4 wb = *reinterpret_cast<const uint4*>(weight + e0 + 8);
        const T* pa = reinterpret_cast<const T*>(&wa);
        const T* pb = reinterpret_cast<const T*>(&wb);
        uint4 oa, ob;
        T* qa = reinterpret_cast<T*>(&oa);
        T* qb = reinterpret_cast<T*>(&ob);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qa[i] = Elem<T>::from_f32(v[i] * inv * Elem<T>::to_f32(pa[i]));
            qb[i] = Elem<T>::from_f32(v[8 + i] * inv * Elem<T>::to_f32(pb[i]));
            q[i] = Elem<T>::to_f32(qa[i]);
            q[8 + i] = Elem<T>::to_f32(qb[i]);
        }
        T* yr = y + static_cast<long long>(row) * hidden;
        *reinterpret_cast<uint4*>(yr + e0) = oa;
        *reinterpret_cast<uint4*>(yr + e0 + 8) = ob;
#pragma unroll
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(q[i]));
    }
    // A 128-wide block's eight threads (aligned in the warp) share its amax.
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    if (!live) return;
    const int e = pie::linear::e4m3_block_exp(amax);
    const float s = __uint_as_float(static_cast<unsigned>(e + 127) << 23);
    u32 w4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        u32 word = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const u32 c = __nv_cvt_float_to_fp8(fminf(fmaxf(q[4 * j + i] / s, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3);
            word |= c << (8 * i);
        }
        w4[j] = word;
    }
    *reinterpret_cast<uint4*>(codes + static_cast<long long>(row) * hidden + e0) = make_uint4(w4[0], w4[1], w4[2], w4[3]);
    if ((tid & 7) == 0) {
        atoms[pie::linear::atom_offset(row, e0 / 128, hidden / 128)] = static_cast<u8>(e + 127);
    }
}

}

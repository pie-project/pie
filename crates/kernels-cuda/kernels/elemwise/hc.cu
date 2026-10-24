#include "prim/device.cuh"
#include "elemwise/hc.cuh"
#include "elemwise/row.cuh"
#include "linear/act_quant.cuh"

namespace pie::elemwise {

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
                v = ::pie::warp_sum(v);
                if (lane == 0) folded[warp] = v;
            }
        } else {
            for (int c = warp; c <= mix_hc; c += THREADS / 32) {
                float sum = 0.f;
                for (int s = lane; s < slices; s += 32)
                    sum += mixes[(static_cast<long long>(s) * gridDim.x + n) * w + c];
                sum = ::pie::warp_sum(sum);
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
        local = ::pie::warp_sum(local);
        if ((tid & 31) == 0) buf[tid >> 5] = local;
    }
    __syncthreads();
    if (tid < 32) {
        float v = tid < BLOCK / 32 ? buf[tid] : 0.f;
        v = ::pie::warp_sum(v);
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
            ::pie::linear::E4m3Block<128> q;
            q.hold(x8);
            q.store(codes, atoms, plane_row, h0 / 128, H, 0, lane);
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

}

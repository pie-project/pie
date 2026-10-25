#include "elemwise/row.cuh"
#include "prim/voxels.cuh"

namespace pie::elemwise {

using namespace ::pie::voxels;

template <class T, int BLOCK = 256>
__global__ void rmsnorm(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    rmsnorm_row<T, BLOCK, false>(
        x, weight, y, hidden, x_row_stride, y_row_stride, eps, heads, win);
}

template <class T, int BLOCK = 256>
__global__ void rmsnorm_plus_one(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    rmsnorm_row<T, BLOCK, true>(
        x, weight, y, hidden, x_row_stride, y_row_stride, eps, heads, win);
}

// Each row's sum of squares, f32: one rank's term of a row norm whose row
// the ranks split, summed across them before `rmsnorm_by` reads it.
template <class T, int BLOCK = 256>
__global__ void sum_squares(
    const T* __restrict__ x,
    float* __restrict__ ss,
    int width,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const T* xr = x + static_cast<long long>(plane_row) * width;
    float local = 0.f;
    for (int i = threadIdx.x; i < width; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }
    __shared__ float buf[BLOCK];
    const float sum = block_reduce_sum_fast<BLOCK>(local, buf);
    if (threadIdx.x == 0) ss[plane_row] = sum;
}

// `x / sqrt(ss / whole + eps)` times the weight (`WEIGHT` 0: none, 1: the
// weight, 2: one plus it): a row norm over a share `width` wide of a row
// `whole` wide, its sum of squares `ss` stated a row.
template <class T, int WEIGHT, int BLOCK = 256>
__global__ void rmsnorm_by(
    const T* __restrict__ x,
    const float* __restrict__ ss,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int width,
    float whole,
    float eps,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const T* xr = x + static_cast<long long>(plane_row) * width;
    T* yr = y + static_cast<long long>(plane_row) * width;
    const float inv_rms = rsqrtf(ss[plane_row] / whole + eps);
    for (int i = threadIdx.x; i < width; i += BLOCK) {
        float w = 1.f;
        if constexpr (WEIGHT == 1) w = Elem<T>::to_f32(weight[i]);
        if constexpr (WEIGHT == 2) w = 1.f + Elem<T>::to_f32(weight[i]);
        yr[i] = Elem<T>::from_f32(Elem<T>::to_f32(xr[i]) * inv_rms * w);
    }
}

// A wide row's RMS norm (its width a whole number of eights, at most
// `256 * 8 * kRuns`) a block a row: each thread's 16-byte runs loaded at
// once and held through the block's sum, so the row is read once.
template <class T, int kRuns, bool WEIGHT_PLUS_ONE>
__global__ void __launch_bounds__(256) rmsnorm_runs(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps,
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
    const T* xr = x + static_cast<long long>(plane_row) * x_row_stride;
    T* yr = y + static_cast<long long>(plane_row) * y_row_stride;
    const int runs = hidden >> 3;
    uint4 v[kRuns];
#pragma unroll
    for (int r = 0; r < kRuns; ++r) {
        const int at = tid + r * 256;
        v[r] = at < runs ? *reinterpret_cast<const uint4*>(xr + at * 8) : make_uint4(0, 0, 0, 0);
    }
    float local = 0.f;
#pragma unroll
    for (int r = 0; r < kRuns; ++r) {
        const T* e = reinterpret_cast<const T*>(&v[r]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float f = Elem<T>::to_f32(e[j]);
            local += f * f;
        }
    }
    __shared__ float buf[256];
    const float sum = block_reduce_sum_fast<256>(local, buf);
    const float inv_rms = rsqrtf(sum / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int r = 0; r < kRuns; ++r) {
        const int at = tid + r * 256;
        if (at >= runs) continue;
        const uint4 wr = *reinterpret_cast<const uint4*>(weight + at * 8);
        const T* e = reinterpret_cast<const T*>(&v[r]);
        const T* wv = reinterpret_cast<const T*>(&wr);
        uint4 o;
        T* oe = reinterpret_cast<T*>(&o);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float w = Elem<T>::to_f32(wv[j]);
            if constexpr (WEIGHT_PLUS_ONE) w += 1.f;
            oe[j] = Elem<T>::from_f32(Elem<T>::to_f32(e[j]) * inv_rms * w);
        }
        *reinterpret_cast<uint4*>(yr + at * 8) = o;
    }
}

// `rmsnorm` (`rmsnorm_plus_one` under PLUS_ONE) of a whole row of `256 *
// PER`, landing its bits: each thread's values (its elements `t`, `t + 256`,
// ...) and their weights loaded at once and held, the squares summed in the
// same order, not read twice in a loop each.
template <class T, int PER, bool PLUS_ONE>
__global__ void __launch_bounds__(256) rmsnorm_held(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    float eps,
    const u32* __restrict__ win)
{
    constexpr int BLOCK = 256;
    constexpr int hidden = BLOCK * PER;
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;
    float v[PER], w[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        v[k] = Elem<T>::to_f32(xr[tid + k * BLOCK]);
        w[k] = Elem<T>::to_f32(weight[tid + k * BLOCK]);
    }
    float local = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) local += v[k] * v[k];
    __shared__ float buf[BLOCK];
    const float inv_rms = rsqrtf(block_reduce_sum_fast<BLOCK>(local, buf) / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        float wv = w[k];
        if constexpr (PLUS_ONE) wv += 1.f;
        yr[tid + k * BLOCK] = Elem<T>::from_f32(v[k] * inv_rms * wv);
    }
}

template <int PER> struct LaneRun { using type = uint4; };
template <> struct LaneRun<4> { using type = uint2; };

// A 128- or 256-wide RMS norm one warp per (row, head): a lane carries
// `PER` consecutive values in one 8- or 16-byte load, so the sum of squares
// is one butterfly and no block barrier; a block is eight such units.
// `units` counts (row, head) pairs; the rows are dense.
template <class T, int PER, bool WEIGHT_PLUS_ONE>
__global__ void __launch_bounds__(256) rmsnorm_in_warp(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int units,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    static_assert(PER == 4 || PER == 8, "a lane's run is one 8- or 16-byte load");
    constexpr int hidden = 32 * PER;
    using Run = typename LaneRun<PER>::type;
    const int unit = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.x >> 5);
    if (unit >= units) return;
    if (win != nullptr && unit / heads >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x & 31;
    const long long at = static_cast<long long>(plane_row) * hidden + lane * PER;

    T xv[PER];
    T wv[PER];
    *reinterpret_cast<Run*>(xv) = *reinterpret_cast<const Run*>(x + at);
    *reinterpret_cast<Run*>(wv) = *reinterpret_cast<const Run*>(weight + lane * PER);
    float xf[PER];
    float local = 0.f;
    #pragma unroll
    for (int u = 0; u < PER; ++u) {
        xf[u] = Elem<T>::to_f32(xv[u]);
        local += xf[u] * xf[u];
    }
    local = ::pie::warp_sum(local);
    const float inv_rms = rsqrtf(local / static_cast<float>(hidden) + eps);

    T out[PER];
    #pragma unroll
    for (int u = 0; u < PER; ++u) {
        float w = Elem<T>::to_f32(wv[u]);
        if constexpr (WEIGHT_PLUS_ONE) w += 1.f;
        out[u] = Elem<T>::from_f32(xf[u] * inv_rms * w);
    }
    *reinterpret_cast<Run*>(y + at) = *reinterpret_cast<const Run*>(out);
}

// Each `group`-wide run of a row RMS-normed and scaled by its own weights,
// plus one under `PLUS_ONE`.
template <class T, int BLOCK, bool PLUS_ONE>
__global__ void rmsnorm_grouped(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int group,
    int groups,
    float eps,
    const u32* __restrict__ win)
{
    const int b = blockIdx.x;

    if (win != nullptr && b / groups >= static_cast<int>(win[0])) return;

    const int gb = win != nullptr ? b + static_cast<int>(win[1]) * groups : b;

    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(gb) * group;
    const T* wr = weight + static_cast<long long>(b % groups) * group;
    T* yr = y + static_cast<long long>(gb) * group;

    float local = 0.f;
    for (int i = tid; i < group; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(group) + eps);

    for (int i = tid; i < group; i += BLOCK) {
        const float xv = Elem<T>::to_f32(xr[i]);
        const float wv = Elem<T>::to_f32(wr[i]) + (PLUS_ONE ? 1.f : 0.f);
        yr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

template <class T, int BLOCK = 256>
__global__ void rmsnorm_no_scale(
    const T* __restrict__ x,
    T* __restrict__ y,
    int hidden,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row / heads >= static_cast<int>(win[0])) return;

    const int plane_row =
        win != nullptr ? row + static_cast<int>(win[1]) * heads : row;

    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        yr[i] = Elem<T>::from_f32(Elem<T>::to_f32(xr[i]) * inv_rms);
    }
}

// `rmsnorm_no_scale` a warp a (row, head) unit, landing what the 256-thread
// block form lands bit for bit: lane `l` takes the sums of the block's
// threads `l`, `l + 32`, ... (each its elements `t`, `t + 256`, ... in
// order), folds each as the block's warps fold theirs and the eight as its
// first warp folds them. A 512-wide head (DeepSeek-V4's q) is a warp where
// a block of 256 threads walked it two elements a thread.
template <class T, int RUNS>
__global__ void __launch_bounds__(256) rmsnorm_no_scale_warp(
    const T* __restrict__ x,
    T* __restrict__ y,
    int units,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    constexpr int hidden = 256 * RUNS;
    const int unit = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.x >> 5);
    if (unit >= units) return;
    if (win != nullptr && unit / heads >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x & 31;
    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;
    float v[8][RUNS];
    float block = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        float local = 0.f;
#pragma unroll
        for (int r = 0; r < RUNS; ++r) {
            v[w][r] = Elem<T>::to_f32(xr[w * 32 + lane + r * 256]);
            local = fmaf(v[w][r], v[w][r], local);
        }
        local = ::pie::warp_sum(local);
        if (lane == w) block = local;
    }
    block = ::pie::warp_sum(block);
    const float inv_rms = rsqrtf(block / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int w = 0; w < 8; ++w)
#pragma unroll
        for (int r = 0; r < RUNS; ++r)
            yr[w * 32 + lane + r * 256] = Elem<T>::from_f32(v[w][r] * inv_rms);
}

// `rmsnorm_no_scale` `LANES` lanes a (row, head) unit over 16-byte runs:
// lane `l` loads the head's runs `l`, `l + LANES`, ... (eight elements each),
// so a 512-wide head (DeepSeek-V4's q) is two loads a lane of 32 and a
// 64-wide one (Muse Spark's) one a lane of 8. Its sum folds in a different
// order from the block form's, so it lands within rounding of it.
template <class T, int RUNS, int LANES = 32>
__global__ void __launch_bounds__(256) rmsnorm_no_scale_vec(
    const T* __restrict__ x,
    T* __restrict__ y,
    int units,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    constexpr int hidden = 8 * LANES * RUNS;
    const int unit = static_cast<int>(blockIdx.x) * (256 / LANES) + static_cast<int>(threadIdx.x) / LANES;
    if (LANES == 32 && unit >= units) return;
    // A unit shares its warp's shuffles with its neighbours, so a dead one
    // runs them on zeros.
    const bool live = unit < units && (win == nullptr || unit / heads < static_cast<int>(win[0]));
    const int plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x % LANES;
    const T* xr = x + static_cast<long long>(plane_row) * hidden + lane * 8;
    T* yr = y + static_cast<long long>(plane_row) * hidden + lane * 8;
    T v[RUNS][8];
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
        *reinterpret_cast<uint4*>(v[r]) = live ? *reinterpret_cast<const uint4*>(xr + r * 8 * LANES) : uint4{};
    float local = 0.f;
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float f = Elem<T>::to_f32(v[r][u]);
            local = fmaf(f, f, local);
        }
#pragma unroll
    for (int off = LANES / 2; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
    if (!live) return;
    const float inv_rms = rsqrtf(local / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int r = 0; r < RUNS; ++r) {
        T out[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) out[u] = Elem<T>::from_f32(Elem<T>::to_f32(v[r][u]) * inv_rms);
        *reinterpret_cast<uint4*>(yr + r * 8 * LANES) = *reinterpret_cast<const uint4*>(out);
    }
}

// The gated norm over an f32 accumulator, or one landed in bf16 (`X`).
template <class X, class T, int BLOCK = 256>
__global__ void rmsnorm_gated_in(
    const X* __restrict__ x,
    const T* __restrict__ gate,
    const float* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    float eps,
    int sigmoid_gate,
    int heads,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row / heads >= static_cast<int>(win[0])) return;

    const int plane_row =
        win != nullptr ? row + static_cast<int>(win[1]) * heads : row;

    const int tid = threadIdx.x;

    const X* xr = x + static_cast<long long>(plane_row) * hidden;
    const T* gr = gate + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<X>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = Elem<X>::to_f32(xr[i]) * inv_rms;
        const float wv = weight != nullptr ? weight[i] : 1.f;
        const float gv = Elem<T>::to_f32(gr[i]);
        const float sg = sigmoid_gate
            ? 1.f / (1.f + __expf(-gv))
            : gv / (1.f + __expf(-gv));
        yr[i] = Elem<T>::from_f32(wv * xv * sg);
    }
}

// The gated norm `LANES` lanes per (row, head) when the normed width is
// `LANES * PER`: a lane carries `PER` consecutive values, so the sum of
// squares is one butterfly and no block barrier; a block is `256 / LANES`
// such rows. No weight: scaleless.
template <class X, class T, int PER, int LANES = 32>
__global__ void __launch_bounds__(256) rmsnorm_gated_in_warp(
    const X* __restrict__ x,
    const T* __restrict__ gate,
    const float* __restrict__ weight,
    T* __restrict__ y,
    int units,
    float eps,
    int sigmoid_gate,
    int heads,
    const u32* __restrict__ win)
{
    static_assert(PER == 4 || PER == 8, "a lane's gate run is one 8- or 16-byte load");
    constexpr int hidden = LANES * PER;
    const int unit = static_cast<int>(blockIdx.x) * (256 / LANES) + static_cast<int>(threadIdx.x) / LANES;
    if (LANES == 32 && unit >= units) return;
    // A unit shares its warp's shuffles with its neighbours: a dead one runs
    // them on zeros, reading and landing nothing.
    const bool live = unit < units && (win == nullptr || unit / heads < static_cast<int>(win[0]));
    const int plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x % LANES;
    const long long at = live ? static_cast<long long>(plane_row) * hidden + lane * PER : 0;

    float xv[PER];
    float wv[PER];
    T gv[PER];
    #pragma unroll
    for (int q = 0; q < PER / 4; ++q) {
        if constexpr (sizeof(X) == 4) {
            const float4 xf = reinterpret_cast<const float4*>(x + at)[q];
            xv[4 * q] = xf.x;
            xv[4 * q + 1] = xf.y;
            xv[4 * q + 2] = xf.z;
            xv[4 * q + 3] = xf.w;
        } else {
            const uint2 xb = reinterpret_cast<const uint2*>(x + at)[q];
            const float2 x01 = bf16x2_to_f32(*reinterpret_cast<const bf16x2*>(&xb.x));
            const float2 x23 = bf16x2_to_f32(*reinterpret_cast<const bf16x2*>(&xb.y));
            xv[4 * q] = x01.x;
            xv[4 * q + 1] = x01.y;
            xv[4 * q + 2] = x23.x;
            xv[4 * q + 3] = x23.y;
        }
        const float4 wf = weight != nullptr ? reinterpret_cast<const float4*>(weight + lane * PER)[q]
                                            : make_float4(1.f, 1.f, 1.f, 1.f);
        wv[4 * q] = wf.x;
        wv[4 * q + 1] = wf.y;
        wv[4 * q + 2] = wf.z;
        wv[4 * q + 3] = wf.w;
    }
    if constexpr (PER == 4) {
        *reinterpret_cast<uint2*>(gv) = *reinterpret_cast<const uint2*>(gate + at);
    } else {
        *reinterpret_cast<uint4*>(gv) = *reinterpret_cast<const uint4*>(gate + at);
    }
    float local = 0.f;
    #pragma unroll
    for (int u = 0; u < PER; ++u) local += xv[u] * xv[u];
    #pragma unroll
    for (int off = LANES / 2; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
    if (!live) return;
    const float inv_rms = rsqrtf(local / static_cast<float>(hidden) + eps);

    T out[PER];
    #pragma unroll
    for (int u = 0; u < PER; ++u) {
        const float g = Elem<T>::to_f32(gv[u]);
        // The gate on the fast division (vLLM's Triton gate's).
        const float sg = __fdividef(sigmoid_gate ? 1.f : g, 1.f + __expf(-g));
        out[u] = Elem<T>::from_f32(wv[u] * (xv[u] * inv_rms) * sg);
    }
    if constexpr (PER == 4) {
        *reinterpret_cast<uint2*>(y + at) = *reinterpret_cast<const uint2*>(out);
    } else {
        *reinterpret_cast<uint4*>(y + at) = *reinterpret_cast<const uint4*>(out);
    }
}

template <class T>
__global__ void standardize(
    T* __restrict__ out,
    const T* __restrict__ bias,
    const T* __restrict__ scale,
    int dim)
{
    T* __restrict__ row = out + static_cast<long long>(blockIdx.x) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {

        const float v = (Elem<T>::to_f32(row[d]) - Elem<T>::to_f32(bias[d]))
                      * Elem<T>::to_f32(scale[d]);
        row[d] = Elem<T>::from_f32(v);
    }
}

// A warp `kP` (row, head) pairs: a pair's `D` (32 to 256) dims a lane
// `D / 32` of them, held across the norm. At 128 dims a warp's pairs load
// at once.
template <class ElemT, int kP>
__global__ void __launch_bounds__(256) rmsnorm_gated_by(
    const float* __restrict__ o,
    const ElemT* __restrict__ g,
    const float* __restrict__ weight,
    ElemT* __restrict__ out,
    int rows, int H, int D, float eps,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const long long first = (static_cast<long long>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * kP;
    const int lane = threadIdx.x & 31;
    const int live = win != nullptr ? min(static_cast<int>(win[0]), rows) : rows;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    auto base_of = [&](long long pair, bool& in) {
        const int t = static_cast<int>(pair / H);
        in = t < live;
        return (static_cast<long long>(t + row0) * H + pair % H) * D;
    };
    auto sigmoid = [](float x) { return 1.f / (1.f + __expf(-x)); };
    const int per = D / 32;
    if (per == 4) {
        float4 v[kP];
        uint2 gw[kP];
        long long base[kP];
        bool in[kP];
        #pragma unroll
        for (int p = 0; p < kP; ++p) {
            base[p] = base_of(first + p, in[p]);
            if (in[p]) {
                v[p] = reinterpret_cast<const float4*>(o + base[p])[lane];
                gw[p] = reinterpret_cast<const uint2*>(g + base[p])[lane];
            }
        }
        const float4 wv = reinterpret_cast<const float4*>(weight)[lane];
        const float w4[4] = {wv.x, wv.y, wv.z, wv.w};
        #pragma unroll
        for (int p = 0; p < kP; ++p) {
            if (!in[p]) continue;
            const float x[4] = {v[p].x, v[p].y, v[p].z, v[p].w};
            float acc = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
            acc = ::pie::warp_sum(acc);
            const float scale = rsqrtf(acc / static_cast<float>(D) + eps);
            const ElemT* g4 = reinterpret_cast<const ElemT*>(&gw[p]);
            uint2 ow;
            ElemT* o4 = reinterpret_cast<ElemT*>(&ow);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                o4[i] = Elem<ElemT>::from_f32(x[i] * scale * w4[i] * sigmoid(Elem<ElemT>::to_f32(g4[i])));
            reinterpret_cast<uint2*>(out + base[p])[lane] = ow;
        }
        return;
    }
    for (int p = 0; p < kP; ++p) {
        bool in;
        const long long base = base_of(first + p, in);
        if (!in) return;
        float x[8];
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) x[i] = o[base + lane * per + i];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) acc += x[i] * x[i];
        acc = ::pie::warp_sum(acc);
        const float scale = rsqrtf(acc / static_cast<float>(D) + eps);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < per) {
                const int d = lane * per + i;
                out[base + d] = Elem<ElemT>::from_f32(x[i] * scale * weight[d] * sigmoid(Elem<ElemT>::to_f32(g[base + d])));
            }
        }
    }
}

template <class T, int BLOCK, bool AFFINE>
__device__ __forceinline__ void layernorm_row(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
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

    __shared__ float buf[BLOCK];

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        local += Elem<T>::to_f32(xr[i]);
    }
    const float mean = block_reduce_sum_fast<BLOCK>(local, buf) /
                       static_cast<float>(hidden);

    __syncthreads();

    float spread = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float c = Elem<T>::to_f32(xr[i]) - mean;
        spread += c * c;
    }
    const float inv = rsqrtf(block_reduce_sum_fast<BLOCK>(spread, buf) /
                                 static_cast<float>(hidden) +
                             eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float c = (Elem<T>::to_f32(xr[i]) - mean) * inv;
        if constexpr (AFFINE) {
            yr[i] = Elem<T>::from_f32(
                fmaf(c, Elem<T>::to_f32(weight[i]), Elem<T>::to_f32(bias[i])));
        } else {
            yr[i] = Elem<T>::from_f32(c);
        }
    }
}

template <class T, int BLOCK = 256>
__global__ void layernorm_no_scale(
    const T* __restrict__ x,
    T* __restrict__ y,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    layernorm_row<T, BLOCK, false>(x, nullptr, nullptr, y, hidden, eps, win);
}

template <class T, int BLOCK = 256>
__global__ void layernorm(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    layernorm_row<T, BLOCK, true>(x, weight, bias, y, hidden, eps, win);
}

using namespace ::pie::voxels;

__device__ __forceinline__ void welford_merge(
    float& cnt,
    float& mean,
    float& m2,
    float cnt_b,
    float mean_b,
    float m2_b)
{
    const float n = cnt + cnt_b;
    if (n == 0.f) return;
    const float delta = mean_b - mean;
    const float f = cnt_b / n;
    mean += delta * f;
    m2 += m2_b + delta * delta * cnt * f;
    cnt = n;
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void group_norm_stats(
    const bf16* __restrict__ x,
    const int* __restrict__ grid,
    float4* __restrict__ partials,
    int c,
    int groups,
    int splits,
    int base)
{
    __shared__ float s_cnt[BLOCK];
    __shared__ float s_mean[BLOCK];
    __shared__ float s_m2[BLOCK];

    const int split = blockIdx.x;
    const int l = blockIdx.y;
    const Lane g = lane_at(grid, l);
    const int n = g.voxels();
    const int chunk = (n + splits - 1) / splits;
    const int begin = split * chunk;
    const int end = min(n, begin + chunk);
    const int tid = threadIdx.x;

    float cnt = 0.f;
    float mean = 0.f;
    float m2 = 0.f;
    if (tid < c) {
        // `base`: the row `grid` addresses `x`'s row 0 at.
        const bf16* at = x + static_cast<long long>(g.off - base + begin) * c + tid;
        for (int v = begin; v < end; ++v, at += c) {
            const float val = bf16_to_f32(ldg(at));
            cnt += 1.f;
            const float d = val - mean;
            mean += d / cnt;
            m2 += d * (val - mean);
        }
    }
    s_cnt[tid] = cnt;
    s_mean[tid] = mean;
    s_m2[tid] = m2;
    __syncthreads();

    if (tid < groups) {
        const int cg = c / groups;
        float gcnt = 0.f;
        float gmean = 0.f;
        float gm2 = 0.f;
        for (int i = tid * cg; i < (tid + 1) * cg; ++i) {
            welford_merge(gcnt, gmean, gm2, s_cnt[i], s_mean[i], s_m2[i]);
        }
        partials[(static_cast<long long>(l) * splits + split) * groups + tid] =
            make_float4(gcnt, gmean, gm2, 0.f);
    }
}

__global__ __launch_bounds__(32) void group_norm_finalize(
    const float4* __restrict__ partials,
    float2* __restrict__ stats,
    int groups,
    int splits,
    float eps)
{
    const int l = blockIdx.y;
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    float cnt = 0.f;
    float mean = 0.f;
    float m2 = 0.f;
    for (int s = lane; s < splits; s += 32) {
        const float4 p = partials[(static_cast<long long>(l) * splits + s) * groups + group];
        welford_merge(cnt, mean, m2, p.x, p.y, p.z);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float o_cnt = __shfl_xor_sync(0xffffffffu, cnt, off);
        const float o_mean = __shfl_xor_sync(0xffffffffu, mean, off);
        const float o_m2 = __shfl_xor_sync(0xffffffffu, m2, off);
        welford_merge(cnt, mean, m2, o_cnt, o_mean, o_m2);
    }
    if (lane == 0) {
        const float var = cnt > 0.f ? m2 / cnt : 0.f;
        stats[l * groups + group] = make_float2(mean, rsqrtf(var + eps));
    }
}

template <bool SILU>
__global__ __launch_bounds__(256) void group_norm_apply(
    const bf16* __restrict__ x,
    const int* __restrict__ grid,
    const float2* __restrict__ stats,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    bf16* __restrict__ y,
    int c,
    int groups,
    int lanes,
    int base,
    long long total)
{
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c);
    const int col = static_cast<int>(e - static_cast<long long>(row) * c);
    Lane box;
    const int l = lane_of(grid, lanes, row + base, box);
    float v = 0.f;
    if (l >= 0) {
        const int group = col / (c / groups);
        const float2 st = stats[l * groups + group];
        v = fmaf((bf16_to_f32(ldg(x + e)) - st.x) * st.y, __ldg(weight + col), __ldg(bias + col));
        if constexpr (SILU) {
            v = v / (1.f + __expf(-v));
        }
    }
    y[e] = f32_to_bf16(v);
}

}

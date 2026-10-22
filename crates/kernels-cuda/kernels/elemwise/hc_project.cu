#include "prim/device.cuh"
#include "elemwise/row.cuh"

namespace pie::elemwise {

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

}

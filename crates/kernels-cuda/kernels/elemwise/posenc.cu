#include "prim/device.cuh"
#include "prim/voxels.cuh"

namespace pie::elemwise {

__global__ void sinusoid(
    const float* __restrict__ t,
    float* __restrict__ o,
    int dim,
    float max_period,
    int flip_sin_cos,
    float scale,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int half = dim / 2;
    const float tv = t[row];
    const float log_period = logf(max_period);
    float* orow = o + static_cast<long long>(row) * dim;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float freq =
            expf(-log_period * static_cast<float>(i) / static_cast<float>(half));
        float sin_v, cos_v;

        sincosf(scale * (tv * freq), &sin_v, &cos_v);
        orow[i] = flip_sin_cos ? cos_v : sin_v;
        orow[i + half] = flip_sin_cos ? sin_v : cos_v;
    }

    if ((dim & 1) != 0 && threadIdx.x == 0) orow[dim - 1] = 0.f;
}

__device__ __forceinline__ int relative_position_bucket(
    int d, bool bidirectional, int num_buckets, float log_ratio)
{
    int bucket = 0;
    int n;
    if (bidirectional) {
        num_buckets /= 2;
        if (d > 0) bucket += num_buckets;
        n = d < 0 ? -d : d;
    } else {
        n = d < 0 ? -d : 0;
    }
    const int max_exact = num_buckets / 2;
    if (n < max_exact) return bucket + n;
    const float x = logf(static_cast<float>(n) / static_cast<float>(max_exact));
    const float scaled = x / log_ratio * static_cast<float>(num_buckets - max_exact);
    int large = max_exact + static_cast<int>(scaled);
    if (large > num_buckets - 1) large = num_buckets - 1;
    return bucket + large;
}

template <class T>
__global__ void relative_bucket_bias(
    const T* __restrict__ embedding,
    float* __restrict__ y,
    int heads,
    int span,
    int max_len,
    int stride,
    int num_buckets,
    int bidirectional,
    float log_ratio)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(heads) * span) return;
    const int h = static_cast<int>(i / span);
    const int c = static_cast<int>(i % span);
    const int d = c - (max_len - 1);
    const int b = relative_position_bucket(d, bidirectional != 0, num_buckets, log_ratio);
    y[i] = Elem<T>::to_f32(embedding[static_cast<long long>(b) * stride + h]);
}

template <class T>
__global__ void rel_bias(
    const T* __restrict__ r,
    const T* __restrict__ proj,
    float* __restrict__ bias,
    int rows, int heads, int d_rel, int extent,
    const u32* __restrict__ win)
{
    const int row = blockIdx.z;

    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    if (row >= rows) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int h = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads || d >= extent) return;
    const T* rr = r + (static_cast<long long>(plane_row) * heads + h) * d_rel;
    float acc = 0.f;
    for (int j = 0; j < d_rel; ++j) {
        acc += Elem<T>::to_f32(rr[j]) *
               Elem<T>::to_f32(proj[static_cast<long long>(j) * extent + d]);
    }
    bias[(static_cast<long long>(plane_row) * heads + h) * extent + d] = acc;
}

struct RuleGeom {
    int kind;
    int a0, a1, a2;
    int b0, b1, b2;
    int c0, c1, c2;
    int d0, d1, d2;
    int flag;
    int lanes;
};

__device__ __forceinline__ int conv_axis(int n, int k, int s, int front, int back, bool& ok) {
    const int span = n + front + back - k;
    if (span < 0) { ok = false; return 0; }
    return span / (s > 0 ? s : 1) + 1;
}

__global__ void grid_rule(const int* __restrict__ grid, int* __restrict__ out, RuleGeom g)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long off = 0;
    for (int l = 0; l < g.lanes; ++l) {
        const int t = grid[4 * l], h = grid[4 * l + 1], w = grid[4 * l + 2];
        int ot = 0, oh = 0, ow = 0;
        bool ok = true;
        switch (g.kind) {
            case 0: {
                const int back_t = g.flag ? 0 : g.d0;
                ot = conv_axis(t, g.a0, g.b0, g.c0, back_t, ok);
                oh = conv_axis(h, g.a1, g.b1, g.c1, g.d1, ok);
                ow = conv_axis(w, g.a2, g.b2, g.c2, g.d2, ok);
                break;
            }
            case 1:
                ot = (g.flag && t > 0) ? 1 + (t - 1) * g.a0 : t * g.a0;
                oh = h * g.a1;
                ow = w * g.a2;
                break;
            case 2:

                ot = t * g.a0 - g.flag;
                oh = h * g.a1;
                ow = w * g.a2;
                ok = ot > 0;
                break;
            case 3:
                ok = g.a0 > 0 && g.a1 > 0 && g.a2 > 0
                    && t % g.a0 == 0 && h % g.a1 == 0 && w % g.a2 == 0;
                if (ok) { ot = t / g.a0; oh = h / g.a1; ow = w / g.a2; }
                break;
            default:

                ok = g.a0 > 0 && g.a1 > 0 && g.a2 > 0
                    && h % g.a1 == 0 && w % g.a2 == 0;
                if (ok) { ot = (t + g.a0 - 1) / g.a0; oh = h / g.a1; ow = w / g.a2; }
                break;
        }
        if (!ok) { ot = 0; oh = 0; ow = 0; }
        out[4 * l] = ot;
        out[4 * l + 1] = oh;
        out[4 * l + 2] = ow;
        out[4 * l + 3] = static_cast<int>(off);
        off += static_cast<long long>(ot) * oh * ow;
    }
}

__global__ __launch_bounds__(256) void cells(
    const int* __restrict__ tgrid,
    const int* __restrict__ positions,
    float* __restrict__ y,
    int clips,
    int axis,
    int width,
    int rows,
    int base)
{
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float* o = y + static_cast<long long>(width) * row;
    for (int c = 0; c < width; ++c) o[c] = 0.0f;
    // The token grid addresses fire rows.
    ::pie::voxels::Lane g;
    if (::pie::voxels::lane_of(tgrid, clips, row + base, g) < 0) {
        o[axis] = static_cast<float>(positions[row]);
        return;
    }
    const ::pie::voxels::Voxel v = ::pie::voxels::unravel(g, row + base - g.off);
    o[0] = static_cast<float>(v.t);
    o[1] = static_cast<float>(v.h);
    o[2] = static_cast<float>(v.w);
    o[width - 3] = static_cast<float>(g.t);
    o[width - 2] = static_cast<float>(g.h);
    o[width - 1] = static_cast<float>(g.w);
}

// Zeros where the group packs none.
__global__ __launch_bounds__(256) void group_first(
    const float* __restrict__ x,
    const int* __restrict__ csr,
    const int* __restrict__ groups,
    const int* __restrict__ lanes,
    float* __restrict__ y,
    int width,
    int x_rows,
    int lane_count,
    int group_count,
    int rows)
{
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int l = lanes[row];
    const int g = (l >= 0 && l < lane_count) ? groups[l] : -1;
    const int at = (g >= 0 && g < group_count && csr[g + 1] > csr[g]) ? csr[g] : -1;
    float* o = y + static_cast<long long>(width) * row;
    for (int c = 0; c < width; ++c) {
        o[c] = (at >= 0 && at < x_rows) ? x[static_cast<long long>(width) * at + c] : 0.0f;
    }
}

__global__ __launch_bounds__(256) void group_rows(
    const int* __restrict__ csr,
    const int* __restrict__ groups,
    const int* __restrict__ lanes,
    float* __restrict__ y,
    int lane_count,
    int group_count,
    int rows)
{
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int l = lanes[row];
    const int g = (l >= 0 && l < lane_count) ? groups[l] : -1;
    y[row] = (g >= 0 && g < group_count) ? static_cast<float>(csr[g + 1] - csr[g]) : 0.0f;
}

}

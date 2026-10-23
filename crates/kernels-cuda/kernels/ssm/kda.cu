#include "ssm/copy.cuh"

namespace pie::ssm {

template <class ElemT>
__global__ void kda_gate_beta(
    const ElemT* __restrict__ raw_g,
    const ElemT* __restrict__ raw_beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    int T, int H, int D,
    float lower_bound,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    const int h = blockIdx.y;
    if (t >= T || h >= H) return;

    const float a = __expf(A_log[h]);
    const long long base = ((long long)t * H + h) * D;
    const long long src = ((long long)t_row * H + h) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float g = Elem<ElemT>::to_f32(raw_g[src + d]) + dt_bias[(long long)h * D + d];
        float gate;
        if (lower_bound < 0.f) {
            gate = lower_bound * sigmoidf(a * g);
        } else {

            const float sp = (g > 20.f) ? g : __logf(1.f + __expf(g));
            gate = -a * sp;
        }
        gate_out[base + d] = gate;
    }

    if (threadIdx.x == 0) {
        beta_out[(long long)t * H + h] =
            sigmoidf(Elem<ElemT>::to_f32(raw_beta[(long long)t_row * H + h]));
    }
}

// `kda_gate_beta` a row a block: each thread four channels of the
// row's heads at once, the same values (each channel's gate as that kernel
// forms it, each head's beta).
template <class ElemT>
__global__ void kda_gate_beta_rows(
    const ElemT* __restrict__ raw_g,
    const ElemT* __restrict__ raw_beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    int T, int H, int D,
    float lower_bound,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int t = blockIdx.x;
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    if (t >= T) return;
    const int width = H * D;
    const long long base = static_cast<long long>(t) * width;
    const long long src = static_cast<long long>(t_row) * width;
    // `gridDim.y` blocks share the row's channels (a decode's few rows).
    for (int e = (blockIdx.y * blockDim.x + threadIdx.x) * 4; e < width;
         e += gridDim.y * blockDim.x * 4) {
        const float a = __expf(A_log[e / D]);
        float out[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float g = Elem<ElemT>::to_f32(raw_g[src + e + u]) + dt_bias[e + u];
            float gate;
            if (lower_bound < 0.f) {
                gate = lower_bound * sigmoidf(a * g);
            } else {
                const float sp = (g > 20.f) ? g : __logf(1.f + __expf(g));
                gate = -a * sp;
            }
            out[u] = gate;
        }
        *reinterpret_cast<float4*>(gate_out + base + e) = make_float4(out[0], out[1], out[2], out[3]);
    }
    for (int h = threadIdx.x; h < H && blockIdx.y == 0; h += blockDim.x) {
        beta_out[static_cast<long long>(t) * H + h] =
            sigmoidf(Elem<ElemT>::to_f32(raw_beta[static_cast<long long>(t_row) * H + h]));
    }
}

// `kda_qkv_prep` a warp a head: block (row, plane, z) takes heads
// z * BLOCK / 32 .. +BLOCK / 32, each warp its head's sum of squares (folded
// across its lanes) and the head's normalized values, no shared memory
// between: a decode's row over the card rather than one block a plane.
// `width` whole `head_dim`s.
template <class ElemT, int BLOCK>
__global__ void kda_qkv_prep_heads(
    const ElemT* __restrict__ mixed,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int width, int head_dim, float eps,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int plane = blockIdx.y;
    const int lane = threadIdx.x % 32;
    const int h = blockIdx.z * (BLOCK / 32) + static_cast<int>(threadIdx.x) / 32;
    if (h * head_dim >= width) return;
    const ElemT* src = mixed + (long long)n_row * 3 * width + (long long)plane * width + h * head_dim;
    float* dst = (plane == 0 ? q_out : (plane == 1 ? k_out : v_out)) + (long long)n * width + h * head_dim;
    if (plane == 2) {
        for (int d = lane; d < head_dim; d += 32) dst[d] = Elem<ElemT>::to_f32(src[d]);
        return;
    }
    float acc = 0.f;
    for (int d = lane; d < head_dim; d += 32) {
        const float x = Elem<ElemT>::to_f32(src[d]);
        acc += x * x;
    }
    acc = ::pie::warp_sum(acc);
    const float q_scale = (plane == 0) ? rsqrtf(static_cast<float>(head_dim)) : 1.f;
    const float inv = rsqrtf(acc + eps);
    for (int d = lane; d < head_dim; d += 32) dst[d] = Elem<ElemT>::to_f32(src[d]) * inv * q_scale;
}

template <class ElemT, int BLOCK>
__global__ void kda_qkv_prep(
    const ElemT* __restrict__ mixed,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int width, int head_dim, float eps,
    const u32* __restrict__ win)
{

    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int plane = blockIdx.y;
    const int tid = threadIdx.x;
    const ElemT* src =
        mixed + (long long)n_row * 3 * width + (long long)plane * width;
    float* dst =
        (plane == 0 ? q_out : (plane == 1 ? k_out : v_out)) +
        (long long)n * width;
    if (plane == 2) {
        for (int i = tid; i < width; i += BLOCK) {
            dst[i] = Elem<ElemT>::to_f32(src[i]);
        }
        return;
    }
    constexpr int kMaxHeads = 256;
    __shared__ float sums[kMaxHeads];
    const int heads = head_dim > 0 ? width / head_dim : 1;

    {
        constexpr int kWarps = BLOCK / 32;
        const int warp = tid / 32;
        const int lane = tid % 32;
        for (int h = warp; h < heads && h < kMaxHeads; h += kWarps) {
            float acc = 0.f;
            for (int d = lane; d < head_dim; d += 32) {
                const float x = Elem<ElemT>::to_f32(src[h * head_dim + d]);
                acc += x * x;
            }
            acc = ::pie::warp_sum(acc);
            if (lane == 0) sums[h] = acc;
        }
    }
    __syncthreads();
    const float q_scale = (plane == 0) ? rsqrtf(static_cast<float>(head_dim)) : 1.f;
    for (int i = tid; i < width; i += BLOCK) {
        const float inv = rsqrtf(sums[(i / head_dim) % kMaxHeads] + eps);
        dst[i] = Elem<ElemT>::to_f32(src[i]) * inv * q_scale;
    }
}

// The KDA scan walks its `D x D` head state in place, so a row that does not
// fold this fire's boundary (an unfolded buffer, a masked lane, the replayed
// tail past a commit) walks a copy of it in `work` and the slot stays what
// the fold before it left.
__device__ __forceinline__ float* kda_state_or_copy(
    float* __restrict__ st,
    float* __restrict__ work,
    long long at,
    int D,
    bool fold)
{
    if (fold) return st;
    float* copy = work + at * D * D;
    for (int i = threadIdx.x; i < D * D; i += blockDim.x) copy[i] = st[i];
    return copy;
}

// `kda_step` over the packed `[q | k | v]` rows themselves: the
// query and key norms and the gates and beta formed in place, the same
// values without the staged planes and their two launches. A lane holds
// `kPer` consecutive dims of a state row in registers (`D = 32 kPer`), and a
// warp walks four rows at once: one load and one store a row.
template <class ElemT, int kPer>
__global__ void kda_step_fused(
    const ElemT* __restrict__ mixed,
    const ElemT* __restrict__ raw_g,
    const ElemT* __restrict__ raw_beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float lower_bound,
    float eps,
    float* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float* __restrict__ out,
    int H, int D,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    float* __restrict__ work,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h = blockIdx.y;
    const int slot = slot_ids[r];
    if (slot < 0) return;
    const long long rh = (long long)r * H + h;
    const long long width = (long long)H * D;
    const ElemT* row = mixed + (long long)r_row * 3 * width + (long long)h * D;
    // This block's value rows of the state: a `gridDim.z`-th of them, so a
    // decode's few heads spread over the SMs. A row that does not fold its
    // boundary lands its walked rows in `work`, and the slot stays as it was.
    const int per = (D + gridDim.z - 1) / gridDim.z;
    const int v0 = blockIdx.z * per;
    const int v1 = min(D, v0 + per);
    const float* st = state_base + (long long)slot * slot_stride_elems + (long long)h * D * D;
    float* dst = folds(write_state, write_state_mask, r) ? const_cast<float*>(st) : work + rh * D * D;
    float* out_h = out + ((long long)r_row * H + h) * D;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + D;
    float* sg = smem + 2 * D;
    float* sv = smem + 3 * D;
    __shared__ float norms[2];
    __shared__ float beta_s;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    if (warp < 2) {
        const ElemT* src = row + (long long)warp * width;
        float acc = 0.f;
        for (int d = lane; d < D; d += 32) {
            const float x = Elem<ElemT>::to_f32(src[d]);
            acc += x * x;
        }
        acc = ::pie::warp_sum(acc);
        if (lane == 0) norms[warp] = acc;
    }
    if (threadIdx.x == 0) {
        beta_s = sigmoidf(Elem<ElemT>::to_f32(raw_beta[(long long)r_row * H + h]));
    }
    const float a = __expf(A_log[h]);
    const long long gsrc = ((long long)r_row * H + h) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float g = Elem<ElemT>::to_f32(raw_g[gsrc + d]) + dt_bias[(long long)h * D + d];
        float gate;
        if (lower_bound < 0.f) {
            gate = lower_bound * sigmoidf(a * g);
        } else {
            const float sp = (g > 20.f) ? g : __logf(1.f + __expf(g));
            gate = -a * sp;
        }
        sg[d] = __expf(gate);
        sv[d] = Elem<ElemT>::to_f32(row[2 * width + d]);
    }
    __syncthreads();
    {
        const float q_scale = rsqrtf(static_cast<float>(D));
        const float inv_q = rsqrtf(norms[0] + eps);
        const float inv_k = rsqrtf(norms[1] + eps);
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            sq[d] = Elem<ElemT>::to_f32(row[d]) * inv_q * q_scale;
            sk[d] = Elem<ElemT>::to_f32(row[width + d]) * inv_k * 1.f;
        }
    }
    __syncthreads();
    const float beta_h = beta_s;

    constexpr int kRows = 4;
    using Vec = Packed<kPer>;
    const int k0 = lane * kPer;
    float g[kPer], k[kPer], q[kPer];
    #pragma unroll
    for (int i = 0; i < kPer; ++i) {
        g[i] = sg[k0 + i];
        k[i] = sk[k0 + i];
        q[i] = sq[k0 + i];
    }
    for (int base = v0 + warp * kRows; base < v1; base += warps * kRows) {
        float s[kRows][kPer];
        #pragma unroll
        for (int j = 0; j < kRows; ++j) {
            if (base + j < v1) Vec::load(s[j], st + (long long)(base + j) * D + k0);
        }
        float mem[kRows];
        #pragma unroll
        for (int j = 0; j < kRows; ++j) {
            mem[j] = 0.f;
            #pragma unroll
            for (int i = 0; i < kPer; ++i) {
                s[j][i] *= g[i];
                mem[j] += s[j][i] * k[i];
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int j = 0; j < kRows; ++j) mem[j] += __shfl_xor_sync(0xffffffffu, mem[j], off);
        }
        float acc[kRows];
        #pragma unroll
        for (int j = 0; j < kRows; ++j) {
            const float delta = (sv[min(base + j, D - 1)] - mem[j]) * beta_h;
            acc[j] = 0.f;
            #pragma unroll
            for (int i = 0; i < kPer; ++i) {
                s[j][i] += k[i] * delta;
                acc[j] += s[j][i] * q[i];
            }
            if (base + j < v1) Vec::store(dst + (long long)(base + j) * D + k0, s[j]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int j = 0; j < kRows; ++j) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], off);
        }
        if (lane < kRows && base + lane < v1) out_h[base + lane] = acc[lane];
    }
}

// `kda_step`'s value rows `v_lo..v_hi` of a head's `D = 128
// kRuns` state in registers: a lane's float4 of each 128-column run, a
// warp's `kPair` rows loaded together so their reads are in flight at
// once, each row read once and written once.
template <int kRuns, int kPair>
__device__ __forceinline__ void kda_step_rows(float* st, const float* sq, const float* sk,
                                              const float* sg, const float* v_h, float beta_h,
                                              float* out_h, int v_lo, int v_hi, int lane,
                                              int warp, int warps) {
    constexpr int D = 128 * kRuns;
    for (int v0 = v_lo + warp * kPair; v0 < v_hi; v0 += warps * kPair) {
        float4 sv[kPair][kRuns];
        float mem[kPair], acc[kPair];
#pragma unroll
        for (int p = 0; p < kPair; ++p) {
            if (v0 + p < v_hi) {
                const float4* row = reinterpret_cast<const float4*>(st + (long long)(v0 + p) * D);
#pragma unroll
                for (int c = 0; c < kRuns; ++c) sv[p][c] = row[c * 32 + lane];
            }
        }
#pragma unroll
        for (int p = 0; p < kPair; ++p) {
            mem[p] = 0.f;
#pragma unroll
            for (int c = 0; c < kRuns; ++c) {
                const int at = c * 32 + lane;
                const float4 gg = reinterpret_cast<const float4*>(sg)[at];
                const float4 kk = reinterpret_cast<const float4*>(sk)[at];
                float4& x = sv[p][c];
                x = make_float4(x.x * gg.x, x.y * gg.y, x.z * gg.z, x.w * gg.w);
                mem[p] += x.x * kk.x + x.y * kk.y + x.z * kk.z + x.w * kk.w;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
            for (int p = 0; p < kPair; ++p) mem[p] += __shfl_xor_sync(0xffffffffu, mem[p], off);
        }
#pragma unroll
        for (int p = 0; p < kPair; ++p) {
            acc[p] = 0.f;
            if (v0 + p >= v_hi) continue;
            const float delta = (v_h[v0 + p] - mem[p]) * beta_h;
            float4* row = reinterpret_cast<float4*>(st + (long long)(v0 + p) * D);
#pragma unroll
            for (int c = 0; c < kRuns; ++c) {
                const int at = c * 32 + lane;
                const float4 kk = reinterpret_cast<const float4*>(sk)[at];
                const float4 qq = reinterpret_cast<const float4*>(sq)[at];
                float4 x = sv[p][c];
                x.x += kk.x * delta;
                x.y += kk.y * delta;
                x.z += kk.z * delta;
                x.w += kk.w * delta;
                row[at] = x;
                acc[p] += x.x * qq.x + x.y * qq.y + x.z * qq.z + x.w * qq.w;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
            for (int p = 0; p < kPair; ++p) acc[p] += __shfl_xor_sync(0xffffffffu, acc[p], off);
        }
#pragma unroll
        for (int p = 0; p < kPair; ++p) {
            if (lane == p && v0 + p < v_hi) out_h[v0 + p] = acc[p];
        }
    }
}

__global__ void kda_step(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float* __restrict__ out,
    int H, int D,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    float* __restrict__ work,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int r = blockIdx.x;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h = blockIdx.y;

    const long long rh = (long long)r * H + h;
    const float* q_h = q_norm + rh * D;
    const float* k_h = k_norm + rh * D;
    const float* v_h = v      + rh * D;
    const float* g_h = gate   + rh * D;
    const float beta_h = beta[rh];

    const int slot = slot_ids[r];
    if (slot < 0) return;
    // `gridDim.z` blocks share the head's value rows (a decode's few rows):
    // this block's run of them, and only those copied where the row does not
    // fold.
    const int per = (D + static_cast<int>(gridDim.z) - 1) / static_cast<int>(gridDim.z);
    const int v_lo = static_cast<int>(blockIdx.z) * per;
    const int v_hi = min(D, v_lo + per);
    float* st = state_base + (long long)slot * slot_stride_elems + (long long)h * D * D;
    if (!folds(write_state, write_state_mask, r)) {
        float* copy = work + rh * D * D;
        for (int i = v_lo * D + threadIdx.x; i < v_hi * D; i += blockDim.x) copy[i] = st[i];
        st = copy;
    }
    float* out_h = out + ((long long)r_row * H + h) * D;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + D;
    float* sg = smem + 2 * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
        sg[i] = __expf(g_h[i]);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    if (D == 128 || D == 256 || D == 512) {
        if (D == 128) {
            kda_step_rows<1, 8>(st, sq, sk, sg, v_h, beta_h, out_h, v_lo, v_hi, lane, warp, warps);
        } else if (D == 256) {
            kda_step_rows<2, 4>(st, sq, sk, sg, v_h, beta_h, out_h, v_lo, v_hi, lane, warp, warps);
        } else {
            kda_step_rows<4, 2>(st, sq, sk, sg, v_h, beta_h, out_h, v_lo, v_hi, lane, warp, warps);
        }
        return;
    }
    for (int vi = v_lo + warp; vi < v_hi; vi += warps) {
        float* row = st + (long long)vi * D;
        float mem = 0.f;
        for (int ki = lane; ki < D; ki += 32) {
            const float sv = row[ki] * sg[ki];
            row[ki] = sv;
            mem += sv * sk[ki];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            mem += __shfl_down_sync(0xffffffffu, mem, off);
        }
        const float delta = __shfl_sync(0xffffffffu, (v_h[vi] - mem) * beta_h, 0);

        float acc = 0.f;
        for (int ki = lane; ki < D; ki += 32) {
            const float sv = row[ki] + sk[ki] * delta;
            row[ki] = sv;
            acc += sv * sq[ki];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out_h[vi] = acc;
    }
}

__global__ void kda_chunked(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float* __restrict__ out,
    int H, int D,
    bool write_state,
    const int* __restrict__ commit_len,
    const u8* __restrict__ write_state_mask,
    const int* __restrict__ begin_at,
    float* __restrict__ work,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;

    long long begin = qo_indptr[r];
    long long end = qo_indptr[r + 1];
    if (begin_at != nullptr) {
        const long long b = begin + begin_at[rl];
        if (b > begin) begin = b < end ? b : end;
    }
    if (commit_len != nullptr) {
        const long long c = begin + commit_len[rl];
        if (c < end) end = c;
    }
    if (end <= begin) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    float* st = kda_state_or_copy(
        state_base + (long long)slot * slot_stride_elems + (long long)h * D * D,
        work, (long long)r * H + h, D, folds(write_state, write_state_mask, rl));

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + D;
    float* sg = smem + 2 * D;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    for (long long t = begin; t < end; ++t) {
        const long long th = t * H + h;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            sq[i] = q_norm[th * D + i];
            sk[i] = k_norm[th * D + i];
            sg[i] = __expf(gate[th * D + i]);
        }
        __syncthreads();

        const float beta_h = beta[th];
        for (int vi = warp; vi < D; vi += warps) {
            float* row = st + (long long)vi * D;
            float mem = 0.f;
            for (int ki = lane; ki < D; ki += 32) {
                const float sv = row[ki] * sg[ki];
                row[ki] = sv;
                mem += sv * sk[ki];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                mem += __shfl_down_sync(0xffffffffu, mem, off);
            }
            const float delta =
                __shfl_sync(0xffffffffu, (v[th * D + vi] - mem) * beta_h, 0);

            float acc = 0.f;
            for (int ki = lane; ki < D; ki += 32) {
                const float sv = row[ki] + sk[ki] * delta;
                row[ki] = sv;
                acc += sv * sq[ki];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }

            if (lane == 0) out[((t + row0) * H + h) * D + vi] = acc;
        }

        __syncthreads();
    }
}

}

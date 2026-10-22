#include "attn/index/index.cuh"

namespace pie::attn {

// DeepSeek-V4.1's FP4 index keys: a position's 128 dims as e2m1 pairs (64
// bytes, the lower dim in the low nibble) under one ue8m0 exponent a 32 (4
// bytes), a page of `ps` positions its `ps` data rows then its `ps` scale
// rows. The reference's `fp4_act_quant(k, 32, ue8m0)`: the exponent
// 2^ceil(log2(amax / 6)), amax floored at 6 * 2^-126, the codes e2m1 of
// x / 2^e. Its queries are read alike.
namespace index_fp4 {

constexpr int kD = 128;
constexpr int kData = kD / 2;
constexpr int kScale = kD / 32;
constexpr int kRow = kData + kScale;

// The biased exponent of 2^ceil(log2(amax * (1/6))), as the reference's
// `fast_round_scale` takes it off the product's bits.
__device__ __forceinline__ int e8m0_ceil6(float amax) {
    const unsigned bits = __float_as_uint(fmaxf(amax, 6.f * 1.1754943508222875e-38f) * (1.f / 6.f));
    return static_cast<int>((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1 : 0);
}

__device__ __forceinline__ float pow2_of(int e) {
    return e != 0 ? __uint_as_float(static_cast<unsigned>(e) << 23) : __uint_as_float(0x00400000u);
}

// Eight dims of a key from its page plane: `slot` its page * ps + offset,
// `run` which 8 of its 128.
__device__ __forceinline__ uint4 load8(const std::uint8_t* plane, long long slot, int ps, int run) {
    const long long page = slot / ps;
    const int off = static_cast<int>(slot % ps);
    const std::uint8_t* base = plane + page * ps * kRow;
    const std::uint32_t nib = *reinterpret_cast<const std::uint32_t*>(base + off * kData + run * 4);
    const float s = pow2_of(base[ps * kData + off * kScale + run / 4]);
    uint4 out;
    std::uint32_t* o32 = reinterpret_cast<std::uint32_t*>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned byte = (nib >> (8 * i)) & 0xffu;
        const __nv_bfloat162 b = __floats2bfloat162_rn(dsv4::e2m1_value(byte & 0xfu) * s,
                                                       dsv4::e2m1_value(byte >> 4) * s);
        o32[i] = *reinterpret_cast<const std::uint32_t*>(&b);
    }
    return out;
}

}

// namespace index_fp4

// The index keys `entries` ([n, 128] bf16) as FP4 rows at their boundaries'
// positions (`pool_kv_append` onto an FP4 key pool). A warp an entry, a lane
// four dims, eight lanes a 32-block.
__global__ __launch_bounds__(128) void index_store_fp4(
    const bf16* __restrict__ entries,
    std::uint8_t* __restrict__ pages,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ boundary_req,
    const u32* __restrict__ page_indices,
    const u32* __restrict__ page_indptr,
    int n,
    int page_size,
    int div)
{
    using namespace index_fp4;
    const int c = static_cast<int>(blockIdx.x) * 4 + static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (c >= n) return;
    const int pos = boundary_pos[c];
    if (pos < 0) return;
    // A compact pool (`div` its ratio): the key at its index, `page_size /
    // div` rows a page; else at the boundary's position.
    const int cell = pos / div;
    page_size /= div;
    const long long page = page_indices[page_indptr[boundary_req[c]] + cell / page_size];
    const int off = cell % page_size;
    const uint2 raw = *reinterpret_cast<const uint2*>(entries + static_cast<long long>(c) * kD + lane * 4);
    const bf16* x = reinterpret_cast<const bf16*>(&raw);
    float v[4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = Elem<bf16>::to_f32(x[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    const int e = e8m0_ceil6(amax);
    const float inv = __uint_as_float(static_cast<unsigned>(254 - e) << 23);
    unsigned code[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) code[i] = dsv4::e2m1_code(fminf(fmaxf(v[i] * inv, -6.f), 6.f));
    std::uint8_t* base = pages + page * page_size * kRow;
    *reinterpret_cast<std::uint16_t*>(base + off * kData + lane * 2) =
        static_cast<std::uint16_t>(code[0] | (code[1] << 4) | (code[2] << 8) | (code[3] << 12));
    if ((lane & 7) == 0) base[page_size * kData + off * kScale + lane / 8] = static_cast<std::uint8_t>(e);
}

// The index queries ([rows, heads * 128] bf16) read at FP4 as the reference
// reads them: each 32-block quantized and read back, `out` the values. A
// thread eight dims, four threads a block.
__global__ __launch_bounds__(256) void index_q_fp4(
    const bf16* __restrict__ q,
    bf16* __restrict__ out,
    long long runs,
    const u32* __restrict__ win,
    int width)
{
    using namespace index_fp4;
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    // Every lane reaches the shuffles; a 32-block is four lanes of one row.
    const bool live = i < runs && (win == nullptr || i / (width / 8) < static_cast<long long>(win[0]));
    const long long at = (win != nullptr ? static_cast<long long>(win[1]) * (width / 8) : 0) + i;
    float v[8];
    float amax = 0.f;
    if (live) {
        const uint4 raw = *reinterpret_cast<const uint4*>(q + at * 8);
        const bf16* x = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = Elem<bf16>::to_f32(x[k]);
            amax = fmaxf(amax, fabsf(v[k]));
        }
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (!live) return;
    const int e = e8m0_ceil6(amax);
    const float s = pow2_of(e);
    const float inv = __uint_as_float(static_cast<unsigned>(254 - e) << 23);
    uint4 o;
    std::uint32_t* o32 = reinterpret_cast<std::uint32_t*>(&o);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float a = dsv4::e2m1_value(dsv4::e2m1_code(fminf(fmaxf(v[2 * k] * inv, -6.f), 6.f))) * s;
        const float b = dsv4::e2m1_value(dsv4::e2m1_code(fminf(fmaxf(v[2 * k + 1] * inv, -6.f), 6.f))) * s;
        const __nv_bfloat162 pr = __floats2bfloat162_rn(a, b);
        o32[k] = *reinterpret_cast<const std::uint32_t*>(&pr);
    }
    *reinterpret_cast<uint4*>(out + at * 8) = o;
}

// FP4 index-key pages widened into a bf16 plane in table order: page i of
// `out` is page indices[i] of the FP4 plane, out_indices[i] = i. A thread a
// run of 8 dims.
__global__ __launch_bounds__(256) void index_stage_fp4(
    const std::uint8_t* __restrict__ plane,
    bf16* __restrict__ out,
    const u32* __restrict__ indices,
    u32* __restrict__ out_indices,
    int n,
    int page_size)
{
    using namespace index_fp4;
    const int i = static_cast<int>(blockIdx.x);
    if (i >= n) return;
    if (threadIdx.x == 0) out_indices[i] = static_cast<u32>(i);
    const long long src = indices[i];
    for (int c = threadIdx.x; c < page_size * (kD / 8); c += blockDim.x) {
        const int r = c / (kD / 8);
        const int run = c % (kD / 8);
        *reinterpret_cast<uint4*>(out + (static_cast<long long>(i) * page_size + r) * kD + run * 8) =
            load8(plane, src * page_size + r, page_size, run);
    }
}

constexpr int kMaxRopeDim = 256;

__device__ __forceinline__ void rope_interleave_inplace(
    float* v, int rope_dim, int pos, float theta)
{
    const int half = rope_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float freq = powf(theta, -2.f * static_cast<float>(i) /
                                       static_cast<float>(rope_dim));
        const float ang = static_cast<float>(pos) * freq;
        float c, s;
        __sincosf(ang, &s, &c);
        const float a = v[2 * i];
        const float b = v[2 * i + 1];
        v[2 * i]     = a * c - b * s;
        v[2 * i + 1] = b * c + a * s;
    }
}

template <class T>
__global__ void index_knorm_rope(
    T* __restrict__ idx_k,
    const T* __restrict__ w,
    const T* __restrict__ b,
    const i32* __restrict__ positions,
    i32 head_dim, i32 rope_dim, float theta, float eps,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = static_cast<int>(threadIdx.x);
    T* row = idx_k + static_cast<long long>(n_row) * head_dim;

    __shared__ float red[kBlock];
    float s = 0.f;
    for (int d = tid; d < head_dim; d += kBlock) s += Elem<T>::to_f32(row[d]);
    red[tid] = s; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    const float mean = red[0] / head_dim;
    __syncthreads();
    float vv = 0.f;
    for (int d = tid; d < head_dim; d += kBlock) { float x = Elem<T>::to_f32(row[d]) - mean; vv += x * x; }
    red[tid] = vv; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / head_dim + eps);
    __syncthreads();
    for (int d = tid; d < head_dim; d += kBlock) {
        const float x = (Elem<T>::to_f32(row[d]) - mean) * inv;
        row[d] = Elem<T>::from_f32(x * Elem<T>::to_f32(w[d]) + Elem<T>::to_f32(b[d]));
    }
    __syncthreads();
    if (tid == 0) {
        float buf[kMaxRopeDim];
        for (int d = 0; d < rope_dim; ++d) buf[d] = Elem<T>::to_f32(row[d]);
        rope_interleave_inplace(buf, rope_dim, positions[n_row], theta);
        for (int d = 0; d < rope_dim; ++d) row[d] = Elem<T>::from_f32(buf[d]);
    }
}

template <class T>
__global__ void index_q_rope(
    T* __restrict__ idx_q,
    const i32* __restrict__ positions,
    i32 n_heads, i32 head_dim, i32 rope_dim, float theta,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = static_cast<int>(threadIdx.x);
    if (h >= n_heads) return;
    T* row = idx_q + (static_cast<long long>(n_row) * n_heads + h) * head_dim;
    float buf[kMaxRopeDim];
    for (int d = 0; d < rope_dim; ++d) buf[d] = Elem<T>::to_f32(row[d]);
    rope_interleave_inplace(buf, rope_dim, positions[n_row], theta);
    for (int d = 0; d < rope_dim; ++d) row[d] = Elem<T>::from_f32(buf[d]);
}

// `index_knorm_rope` a warp a row (rows_vec): the layer norm's moments by
// shuffles, the normed row rounded to T as the plane holds it, then the
// rotated pairs a lane each, every pair's frequency its own.
template <class T>
__global__ void index_knorm_rope_warp(
    T* __restrict__ idx_k,
    const T* __restrict__ w,
    const T* __restrict__ b,
    const i32* __restrict__ positions,
    i32 head_dim, i32 rope_dim, float theta, float eps,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int lane = static_cast<int>(threadIdx.x);
    T* row = idx_k + static_cast<long long>(n_row) * head_dim;
    float s = 0.f;
    for (int d = lane; d < head_dim; d += 32) s += Elem<T>::to_f32(row[d]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / head_dim;
    float vv = 0.f;
    for (int d = lane; d < head_dim; d += 32) {
        const float x = Elem<T>::to_f32(row[d]) - mean;
        vv += x * x;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) vv += __shfl_xor_sync(0xffffffffu, vv, o);
    const float inv = rsqrtf(vv / head_dim + eps);
    __syncwarp();
    for (int d = lane; d < head_dim; d += 32) {
        const float x = (Elem<T>::to_f32(row[d]) - mean) * inv;
        row[d] = Elem<T>::from_f32(x * Elem<T>::to_f32(w[d]) + Elem<T>::to_f32(b[d]));
    }
    __syncwarp();
    const int pos = positions[n_row];
    for (int i = lane; i < rope_dim / 2; i += 32) {
        const float freq = powf(theta, -2.f * static_cast<float>(i) / static_cast<float>(rope_dim));
        float c, sn;
        __sincosf(static_cast<float>(pos) * freq, &sn, &c);
        const float a = Elem<T>::to_f32(row[2 * i]);
        const float bb = Elem<T>::to_f32(row[2 * i + 1]);
        row[2 * i] = Elem<T>::from_f32(a * c - bb * sn);
        row[2 * i + 1] = Elem<T>::from_f32(bb * c + a * sn);
    }
}

// `index_q_rope` with a thread a (head, pair) (rows_vec): the same
// rotation, every pair at once rather than a head's pairs in turn.
template <class T>
__global__ void index_q_rope_pairs(
    T* __restrict__ idx_q,
    const i32* __restrict__ positions,
    i32 n_heads, i32 head_dim, i32 rope_dim, float theta,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int half = rope_dim / 2;
    const int pos = positions[n_row];
    for (int it = static_cast<int>(threadIdx.x); it < n_heads * half; it += static_cast<int>(blockDim.x)) {
        const int h = it / half;
        const int i = it % half;
        T* row = idx_q + (static_cast<long long>(n_row) * n_heads + h) * head_dim;
        const float freq = powf(theta, -2.f * static_cast<float>(i) / static_cast<float>(rope_dim));
        float c, sn;
        __sincosf(static_cast<float>(pos) * freq, &sn, &c);
        const float a = Elem<T>::to_f32(row[2 * i]);
        const float bb = Elem<T>::to_f32(row[2 * i + 1]);
        row[2 * i] = Elem<T>::from_f32(a * c - bb * sn);
        row[2 * i + 1] = Elem<T>::from_f32(bb * c + a * sn);
    }
}

// A pooled selection read back as tokens: each selected block `b` (of
// `ratio` rows, -1 none) its rows `b * ratio ..`, then the row's own open
// block's rows (past its last closed block, up to and with the row), the
// rest -1; `topk * ratio + ratio - 1` a row.
__global__ void index_pool_tokens(
    const i32* __restrict__ blocks,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    i32* __restrict__ tokens,
    i32 R, i32 page_size, i32 topk, i32 ratio,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    int r = 0;
    for (; r < R - 1; ++r) {
        if (t < static_cast<int>(qo_indptr[r + 1])) break;
    }
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - static_cast<int>(kv_page_indptr[r]);
    const int kv_len = (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int abs_q = kv_len - new_tokens + (t - qo_lo);
    const int open_from = (abs_q + 1) / ratio * ratio;
    const i32* brow = blocks + static_cast<long long>(t_row) * topk;
    const int width = topk * ratio + ratio - 1;
    i32* trow = tokens + static_cast<long long>(t_row) * width;
    for (int n = static_cast<int>(threadIdx.x); n < width; n += static_cast<int>(blockDim.x)) {
        int j = -1;
        if (n < topk * ratio) {
            const int b = brow[n / ratio];
            if (b >= 0) j = b * ratio + n % ratio;
        } else {
            const int at = open_from + (n - topk * ratio);
            if (at <= abs_q) j = at;
        }
        trow[n] = j;
    }
}

template <class T, bool kRadix = false>
__global__ void index_topk_paged(
    const T* __restrict__ idx_q,
    const T* __restrict__ idx_w,
    const T* __restrict__ key_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    float* __restrict__ scores,
    i32* __restrict__ selection,
    i32 R, i32 H, i32 D, i32 page_size, i32 score_stride, i32 topk,
    i32 ratio,
    i32 compact,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int tid = static_cast<int>(threadIdx.x);
    i32* srow = selection + static_cast<long long>(t_row) * topk;

    int r = 0;
    for (; r < R - 1; ++r) {
        if (t < static_cast<int>(qo_indptr[r + 1])) break;
    }
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int abs_q = kv_len - new_tokens + (t - qo_lo);

    const int stride = (ratio > 0) ? ratio : 1;
    const int total = abs_q + 1;
    int nkeys = (total > 0) ? total / stride : 0;
    if (nkeys > score_stride) nkeys = score_stride;

    // A row that sees no more blocks than the budget selects them all: its
    // scores would rank nothing, so none are taken.
    if (nkeys <= topk) {
        for (int n = tid; n < topk; n += kBlock) srow[n] = (n < nkeys) ? n : -1;
        return;
    }

    float* frow = scores + static_cast<long long>(t) * score_stride;
    const T* qi = idx_q + static_cast<long long>(t_row) * H * D;
    const T* wi = (idx_w != nullptr) ? idx_w + static_cast<long long>(t_row) * H : nullptr;
    // The row's query, read once into shared memory where every key's
    // thread reads it (the same values, in the same order).
    constexpr int kStagedQuery = 1024;
    __shared__ float q_s[kStagedQuery];
    const bool staged = H * D <= kStagedQuery;
    if (staged) {
        for (int i = tid; i < H * D; i += kBlock) q_s[i] = Elem<T>::to_f32(qi[i]);
    }
    __syncthreads();
    // A compact pool holds its keys a row each, `page_size / stride` a page;
    // else key j sits at its group's last position.
    const int kps = compact ? page_size / stride : page_size;
    for (int j = tid; j < nkeys; j += kBlock) {
        const int cell = compact ? j : (j + 1) * stride - 1;
        const int page =
            static_cast<int>(kv_page_indices[pages_first + cell / kps]);
        const int off = cell % kps;
        const T* kj =
            key_pages + (static_cast<long long>(page) * kps + off) * D;
        float acc = 0.f;
        for (int h = 0; h < H; ++h) {
            const T* qh = qi + static_cast<long long>(h) * D;
            float dot = 0.f;
            if (staged) {
                for (int d = 0; d < D; ++d) dot += q_s[h * D + d] * Elem<T>::to_f32(kj[d]);
            } else {
                for (int d = 0; d < D; ++d) dot += Elem<T>::to_f32(qh[d]) * Elem<T>::to_f32(kj[d]);
            }
            acc += fmaxf(dot, 0.f) * (idx_w != nullptr ? Elem<T>::to_f32(wi[h]) : 1.f);
        }
        frow[j] = acc;
    }
    __syncthreads();

    if constexpr (kRadix) {
        select_top_radix(frow, nkeys, topk, srow, tid);
    } else {
        select_top(frow, nkeys, topk, srow, tid);
    }
}

// `index_topk_paged` with the scores on the tensor cores: the row's heads
// are an mma's rows, `kTiles` tiles of 16 (`H` <= 16 kTiles, the rest zero),
// and each warp scores its own
// 16-block chunks of the row's keys (`D` = 128), gathered into shared memory
// a chunk at a time. The ranking that follows is the same.
namespace index_mma {

constexpr int kD = 128;
constexpr int kLd = kD + 8;
constexpr int kWarps = kBlock / 32;
constexpr int kKeys = 16;
constexpr int kNt = kKeys / 8;  // a chunk's 8-key mma column tiles

__device__ __forceinline__ std::uint32_t smem_addr(const void* p) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const std::uint32_t (&a)[4],
                                             const std::uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}

// namespace index_mma

template <class T, int kTiles = 1, bool kRadix = false, bool kFp4 = false>
__global__ void index_topk_paged_mma(
    const T* __restrict__ idx_q,
    const T* __restrict__ idx_w,
    const T* __restrict__ key_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    float* __restrict__ scores,
    i32* __restrict__ selection,
    i32 R, i32 H, i32 D, i32 page_size, i32 score_stride, i32 topk,
    i32 ratio,
    i32 compact,
    const i32* __restrict__ turn_pos,
    i32 turn_dim,
    float turn_theta,
    float turn_factor,
    float turn_low,
    float turn_high,
    const u32* __restrict__ win)
{
    using namespace index_mma;
    const int t = static_cast<int>(blockIdx.x);
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    i32* srow = selection + static_cast<long long>(t_row) * topk;

    int r = 0;
    for (; r < R - 1; ++r) {
        if (t < static_cast<int>(qo_indptr[r + 1])) break;
    }
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int abs_q = kv_len - new_tokens + (t - qo_lo);

    const int stride = (ratio > 0) ? ratio : 1;
    const int total = abs_q + 1;
    int nkeys = (total > 0) ? total / stride : 0;
    if (nkeys > score_stride) nkeys = score_stride;

    if (nkeys <= topk) {
        for (int n = tid; n < topk; n += kBlock) srow[n] = (n < nkeys) ? n : -1;
        return;
    }

    float* frow = scores + static_cast<long long>(t) * score_stride;
    const T* qi = idx_q + static_cast<long long>(t_row) * H * kD;
    // The row's query (16 kTiles rows of kLd) and each warp's key chunk, in
    // dynamic shared memory.
    extern __shared__ __align__(16) unsigned char q_raw[];
    T* q_s = reinterpret_cast<T*>(q_raw);
    T* k_s0 = q_s + 16 * kTiles * kLd;
    __shared__ float w_s[16 * kTiles];
    __shared__ long long slot_s[kWarps][2][kKeys];
    // The turn's frequencies, a pair each, worked once a block.
    __shared__ float turn_f[kD / 2];
    if (turn_dim > 0) {
        for (int pair = tid; pair < turn_dim / 2; pair += kBlock) {
            float freq = powf(turn_theta, -2.f * static_cast<float>(pair) / static_cast<float>(turn_dim));
            if (turn_factor > 1.f) freq = ::pie::yarn_original_freq(freq, turn_factor, turn_low, turn_high, pair);
            turn_f[pair] = freq;
        }
        __syncthreads();
    }
    for (int i = tid; i < 16 * kTiles * (kD / 8); i += kBlock) {
        const int row = i / (kD / 8);
        const int d = (i % (kD / 8)) * 8;
        int4 v = make_int4(0, 0, 0, 0);
        if (row < H) v = *reinterpret_cast<const int4*>(qi + row * kD + d);
        // The query's tail turned as it lands (`rope_partial_last`, forward,
        // interleaved, at the row's position), where the ranking holds it.
        if (turn_dim > 0 && d >= kD - turn_dim) {
            T* e = reinterpret_cast<T*>(&v);
            const float pos = static_cast<float>(turn_pos[t_row]);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int pair = (d - (kD - turn_dim)) / 2 + u;
                float sn, cs;
                __sincosf(pos * turn_f[pair], &sn, &cs);
                const float a = Elem<T>::to_f32(e[2 * u]);
                const float b = Elem<T>::to_f32(e[2 * u + 1]);
                e[2 * u] = Elem<T>::from_f32(a * cs - b * sn);
                e[2 * u + 1] = Elem<T>::from_f32(b * cs + a * sn);
            }
        }
        *reinterpret_cast<int4*>(q_s + row * kLd + d) = v;
    }
    for (int h = tid; h < 16 * kTiles; h += kBlock) {
        w_s[h] = h < H
            ? (idx_w != nullptr ? Elem<T>::to_f32(idx_w[static_cast<long long>(t_row) * H + h]) : 1.f)
            : 0.f;
    }
    __syncthreads();

    const int g = lane >> 2, p = (lane & 3) << 1;
    // A compact pool holds its keys a row each, `page_size / stride` a page;
    // else key j sits at its group's last position.
    const int kps = compact ? page_size / stride : page_size;
    // Each warp's chunks two deep: the next chunk's keys copy in
    // (cp.async, zero-filled past the row's keys) while this one's score.
    auto fetch = [&](int base, int buf) {
        if (lane < kKeys) {
            const int j = base + lane;
            long long slot = -1;
            if (j < nkeys) {
                const int cell = compact ? j : (j + 1) * stride - 1;
                slot = static_cast<long long>(kv_page_indices[pages_first + cell / kps]) * kps + cell % kps;
            }
            slot_s[warp][buf][lane] = slot;
        }
        __syncwarp();
        T* kb = k_s0 + (warp * 2 + buf) * kKeys * kLd;
        if constexpr (kFp4) {
            // FP4 keys, widened as they land.
            const std::uint8_t* plane = reinterpret_cast<const std::uint8_t*>(key_pages);
            for (int c = lane; c < kKeys * (kD / 8); c += 32) {
                const int key = c / (kD / 8);
                const int run = c % (kD / 8);
                const long long slot = slot_s[warp][buf][key];
                uint4 v = make_uint4(0, 0, 0, 0);
                if (slot >= 0) v = index_fp4::load8(plane, slot, kps, run);
                *reinterpret_cast<uint4*>(kb + key * kLd + run * 8) = v;
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            return;
        }
        for (int c = lane; c < kKeys * (kD / 8); c += 32) {
            const int key = c / (kD / 8);
            const int d = (c % (kD / 8)) * 8;
            const long long slot = slot_s[warp][buf][key];
            const T* src = key_pages + (slot >= 0 ? slot : 0) * kD + d;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(kb + key * kLd + d)),
                         "l"(src), "r"(slot >= 0 ? 16 : 0)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    int buf = 0;
    if (warp * kKeys < nkeys) fetch(warp * kKeys, 0);
    for (int base = warp * kKeys; base < nkeys; base += kWarps * kKeys, buf ^= 1) {
        if (base + kWarps * kKeys < nkeys) {
            fetch(base + kWarps * kKeys, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncwarp();
        const T* ks = k_s0 + (warp * 2 + buf) * kKeys * kLd;
        // A key's score, this lane's share: each head tile's weighted
        // rectified dots, rows `g` and `g + 8` of the tile.
        float part[kNt][2] = {};
#pragma unroll
        for (int ht = 0; ht < kTiles; ++ht) {
        float acc[kNt][4] = {};
#pragma unroll
        for (int k0 = 0; k0 < kD; k0 += 16) {
            std::uint32_t a[4];
            {
                const T* rp = q_s + (ht * 16 + (lane & 15)) * kLd + k0 + ((lane & 16) ? 8 : 0);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                             : "r"(smem_addr(rp)));
            }
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                std::uint32_t b[2];
                const T* rp = ks + (nt * 8 + (lane & 7)) * kLd + k0 + ((lane & 8) ? 8 : 0);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(b[0]), "=r"(b[1])
                             : "r"(smem_addr(rp)));
                mma_m16n8k16(acc[nt], a, b);
            }
        }
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                part[nt][i] += fmaxf(acc[nt][i], 0.f) * w_s[ht * 16 + g]
                    + fmaxf(acc[nt][2 + i], 0.f) * w_s[ht * 16 + g + 8];
        }
        // A key's score sums its heads' weighted rectified dots, across the
        // eight lane groups.
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                float v = part[nt][i];
#pragma unroll
                for (int sh = 4; sh < 32; sh <<= 1) v += __shfl_xor_sync(0xffffffffu, v, sh);
                const int j = base + nt * 8 + p + i;
                if (g == 0 && j < nkeys) frow[j] = v;
            }
        }
        __syncwarp();
    }
    __syncthreads();

    if constexpr (kRadix) {
        select_top_radix(frow, nkeys, topk, srow, tid);
    } else {
        select_top(frow, nkeys, topk, srow, tid);
    }
}

constexpr int kMaxIndexHeads = 8;

constexpr int kMaxIndexPer = 8;

// A row's ranking of the blocks it sees, one per head: a block scores the
// best `q_h·k` among its keys at or before the row; the row's own block is
// always kept, the other `topk - 1` are the best of the rest, and the run is
// written ascending, `-1` past its end. Warps score the blocks (a lane holds
// `D / 32` of each head's query); warp `h` then ranks head `h`.
template <class T>
__global__ void index_topk_block_max(
    const T* __restrict__ idx_q,
    const T* __restrict__ key_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    float* __restrict__ scores,
    i32* __restrict__ selection,
    i32 R, i32 H, i32 D, i32 page_size, i32 score_stride, i32 topk,
    i32 block,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int kWarps = kBlock / 32;
    i32* srow = selection + static_cast<long long>(t_row) * H * topk;

    int r = 0;
    for (; r < R - 1; ++r) {
        if (t < static_cast<int>(qo_indptr[r + 1])) break;
    }
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int abs_q = kv_len - new_tokens + (t - qo_lo);
    if (abs_q < 0) {
        for (int n = tid; n < H * topk; n += kBlock) srow[n] = -1;
        return;
    }

    int nb = abs_q / block + 1;
    if (nb > score_stride) nb = score_stride;
    if (nb <= topk) {
        for (int n = tid; n < H * topk; n += kBlock) {
            const int at = n % topk;
            srow[n] = (at < nb) ? at : -1;
        }
        return;
    }

    const int per = D / 32;
    float qreg[kMaxIndexHeads][kMaxIndexPer];
    const T* qi = idx_q + static_cast<long long>(t_row) * H * D;
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < per; ++i) {
            qreg[h][i] = Elem<T>::to_f32(qi[h * D + lane + i * 32]);
        }
    }

    float* frow = scores + static_cast<long long>(t) * H * score_stride;
    for (int b = warp; b < nb; b += kWarps) {
        float best[kMaxIndexHeads];
        for (int h = 0; h < H; ++h) best[h] = neg_inf();
        const int lo = b * block;
        const int hi = min(lo + block, abs_q + 1);
        for (int j = lo; j < hi; ++j) {
            const int page =
                static_cast<int>(kv_page_indices[pages_first + j / page_size]);
            const T* kj = key_pages +
                (static_cast<long long>(page) * page_size + j % page_size) * D;
            float kv[kMaxIndexPer];
            for (int i = 0; i < per; ++i) kv[i] = Elem<T>::to_f32(kj[lane + i * 32]);
            for (int h = 0; h < H; ++h) {
                float pd = 0.f;
                for (int i = 0; i < per; ++i) pd += qreg[h][i] * kv[i];
                #pragma unroll
                for (int sh = 16; sh > 0; sh >>= 1) {
                    pd += __shfl_xor_sync(0xffffffffu, pd, sh);
                }
                best[h] = fmaxf(best[h], pd);
            }
        }
        if (lane == 0) {
            for (int h = 0; h < H; ++h) frow[h * score_stride + b] = best[h];
        }
    }
    __syncthreads();

    for (int h = warp; h < H; h += kWarps) {
        float* hrow = frow + h * score_stride;
        i32* out = srow + h * topk;
        const int own = nb - 1;
        if (lane == 0) hrow[own] = neg_inf();
        __syncwarp();
        int picked = 0;
        for (int n = 0; n < topk - 1; ++n) {
            float v = neg_inf();
            int at = nb;
            for (int b = lane; b < nb; b += 32) {
                const float s = hrow[b];
                if (s > v || (s == v && b < at && s > neg_inf())) { v = s; at = b; }
            }
            #pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, sh);
                const int oa = __shfl_xor_sync(0xffffffffu, at, sh);
                if (ov > v || (ov == v && oa < at)) { v = ov; at = oa; }
            }
            if (at >= nb) break;
            if (lane == 0) {
                out[picked] = at;
                hrow[at] = neg_inf();
            }
            ++picked;
            __syncwarp();
        }
        if (lane == 0) {
            out[picked++] = own;
            for (int a = 1; a < picked; ++a) {
                const int key = out[a];
                int c = a - 1;
                while (c >= 0 && out[c] > key) { out[c + 1] = out[c]; --c; }
                out[c + 1] = key;
            }
            for (int n = picked; n < topk; ++n) out[n] = -1;
        }
    }
}

}

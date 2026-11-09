#include <cstdint>
#include <cuda_bf16.h>
#include <math_constants.h>

// The paged latent attention on Hopper's wgmma (`kernel.sdpa.mla_wgmma`):
// `mla_mma_paged_kernel`'s rows, selections, causal bound, windows and
// splits, 64 heads a CTA over a 512-wide bf16 latent and a 64-wide rotated
// part. Q (64 heads x 576) and a ring of two 64-key tiles sit in shared
// memory, 128-byte swizzled K-major; each of two warpgroups scores 32 of a
// tile's keys for every head (m64n32k16), the two halves' maxima meet in
// shared memory, the probabilities land there as bf16, and each warpgroup
// sums half the latent's 512 values over all 64 keys (m64n256k16, the values
// read transposed out of the key tile).

namespace pie::attn::mla::wg {

constexpr int kH = 64;
constexpr int kBK = 64;
constexpr int kCkv = 512;
constexpr int kKpe = 64;
constexpr int kD = kCkv + kKpe;
constexpr int kAtoms = kD / 64;
constexpr int kAtomBytes = 64 * 128;
constexpr int kTileBytes = kAtoms * kAtomBytes;
constexpr int kThreads = 256;

/// The dynamic shared memory the kernel takes: the 1024-byte alignment, Q,
/// two key tiles, P, the stage's slots, the halves' row maxima.
constexpr int kSmem = 1024 + 3 * kTileBytes + kAtomBytes + 2 * kBK * 8 + 2 * kH * 4 + 64;

__device__ __forceinline__ unsigned long long desc_k(const void* smem) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    unsigned long long d = (at & 0x3FFFFu) >> 4;
    d |= 1ull << 16;
    d |= static_cast<unsigned long long>(1024 >> 4) << 32;
    d |= 1ull << 62;
    return d;
}

/// An MN-major (transposed) operand of 128-byte swizzle atoms: 64 elements
/// along MN by 8 along K, the next 64 MN `lbo` bytes on, the next 8 K `sbo`.
__device__ __forceinline__ unsigned long long desc_mn(const void* smem, unsigned lbo, unsigned sbo) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    unsigned long long d = (at & 0x3FFFFu) >> 4;
    d |= static_cast<unsigned long long>((lbo >> 4) & 0x3FFFu) << 16;
    d |= static_cast<unsigned long long>((sbo >> 4) & 0x3FFFu) << 32;
    d |= 1ull << 62;
    return d;
}

__device__ __forceinline__ void mma_s(float (&d)[16], unsigned long long a, unsigned long long b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15])
        : "l"(a), "l"(b));
}

// m64n256k16, B transposed (MN-major): a warpgroup's 64 rows by 256 values.
__device__ __forceinline__ void mma_o(float (&d)[128], unsigned long long a, unsigned long long b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
        "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
        "%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
        "%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,"
        "%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
        " %128, %129, 1, 1, 1, 0, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
          "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
          "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
          "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
          "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
          "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
          "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
          "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
          "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(a), "l"(b));
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool live) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(at), "l"(src), "r"(live ? 16 : 0));
}

__device__ __forceinline__ void bar_all() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }

/// The byte offset of row `r`'s 16-byte chunk `c` (of 72) in a tile.
__device__ __forceinline__ int at_of(int r, int c) {
    return (c >> 3) * kAtomBytes + r * 128 + (((c & 7) ^ (r & 7)) << 4);
}

/// Arguments and grid as `mla_mma_paged_kernel` takes them, but a CTA 64
/// heads (`blockIdx.x` a group of 64) and 256 threads.
__global__ __launch_bounds__(kThreads, 1) void mla_wgmma_paged_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_pages,
    const __nv_bfloat16* __restrict__ kpe_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    __nv_bfloat16* __restrict__ o,
    float* __restrict__ lse,
    __nv_bfloat16* __restrict__ part_o,
    float* __restrict__ part_lse,
    const std::int32_t* __restrict__ selection, int top_k, int dense,
    int R, int H, int page_size, long long ckv_stride, long long kpe_stride,
    float sm_scale, bool causal,
    const std::uint32_t* __restrict__ win)
{
    extern __shared__ __align__(16) unsigned char wg_smem_raw[];
    unsigned char* smem = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<unsigned long long>(wg_smem_raw) + 1023) & ~1023ull);
    unsigned char* sQ = smem;
    unsigned char* sK = sQ + kTileBytes;
    unsigned char* sP = sK + 2 * kTileBytes;
    long long* s_slot = reinterpret_cast<long long*>(sP + kAtomBytes);
    float* s_max = reinterpret_cast<float*>(s_slot + 2 * kBK);
    int* s_misc = reinterpret_cast<int*>(s_max + 2 * kH);
    unsigned long long* s_ok = reinterpret_cast<unsigned long long*>(s_misc + 4);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wg = warp >> 2;
    const int wi = warp & 3;
    const int g = lane >> 2;
    const int c = lane & 3;

    // The rows last first: a causal prefill's late rows read the most tiles,
    // and the light early ones then fill the last wave.
    const int t = static_cast<int>(gridDim.y - 1 - blockIdx.y);
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int h0 = blockIdx.x * kH;

    if (tid == 0) { s_misc[0] = 0; s_misc[1] = 0; }
    __syncthreads();
    for (int i = tid; i < R; i += kThreads) {
        if (t >= static_cast<int>(qo_indptr[i]) && t < static_cast<int>(qo_indptr[i + 1])) s_misc[0] = i;
    }
    __syncthreads();
    const int r = s_misc[0];

    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len = (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int abs_q = kv_len - new_tokens + (t - qo_lo);
    const int j_end = causal ? (abs_q + 1) : kv_len;
    const std::int32_t* srow =
        selection != nullptr ? selection + static_cast<long long>(t_row) * top_k : nullptr;
    if (srow != nullptr && dense != 0 && j_end <= top_k) srow = nullptr;
    if (srow != nullptr) {
        int last = 0;
        for (int n = tid; n < top_k; n += kThreads) {
            if (srow[n] >= 0) last = n + 1;
        }
        if (last > 0) atomicMax(&s_misc[1], last);
        __syncthreads();
    }
    const int steps = srow != nullptr ? s_misc[1] : j_end;
    const int splits = static_cast<int>(gridDim.z);
    const int per = ((steps + kBK - 1) / kBK + splits - 1) / splits * kBK;
    const int j_lo = min(steps, static_cast<int>(blockIdx.z) * per);
    const int j_hi = min(steps, j_lo + per);

    // Q: 64 heads' 576 values, a row a head.
    for (int i = tid; i < kH * (kD / 8); i += kThreads) {
        const int row = i / (kD / 8);
        const int ch = i % (kD / 8);
        const long long qh = static_cast<long long>(t_row) * H + h0 + row;
        const void* src = ch < kCkv / 8
            ? static_cast<const void*>(q_nope + qh * kCkv + ch * 8)
            : static_cast<const void*>(q_pe + qh * kKpe + (ch - kCkv / 8) * 8);
        cp16(sQ + at_of(row, ch), src, true);
    }

    // A tile's slots (and which hold a key), read a tile ahead into the
    // first 64 threads' registers under the scores' MMAs, then published:
    // a key's page above its 16 low bits, its cell in the page below them.
    long long nslot = -1;
    bool nok = false;
    auto fetch = [&](int j0) {
        if (tid < kBK) {
            const int n = j0 + tid;
            int j = n;
            if (srow != nullptr) j = n < j_hi ? srow[n] : -1;
            nok = n < j_hi && j >= 0 && j < j_end;
            nslot = nok
                ? (static_cast<long long>(kv_page_indices[pages_first + j / page_size]) << 16) | (j % page_size)
                : -1;
        }
    };
    auto publish = [&](int stage) {
        if (tid < kBK) {
            s_slot[stage * kBK + tid] = nslot;
            const unsigned bits = __ballot_sync(0xffffffffu, nok);
            if (lane == 0) reinterpret_cast<unsigned*>(&s_ok[stage])[warp] = bits;
        }
    };
    auto issue = [&](int stage) {
        unsigned char* dst = sK + stage * kTileBytes;
        const long long* slots = s_slot + stage * kBK;
#pragma unroll
        for (int u = 0; u < kBK * (kD / 8) / kThreads; ++u) {
            const int i = tid + u * kThreads;
            const int row = i / (kD / 8);
            const int ch = i % (kD / 8);
            const long long slot = slots[row];
            const bool live = slot >= 0;
            const long long page = live ? slot >> 16 : 0;
            const long long cell = live ? slot & 0xffff : 0;
            const void* src = ch < kCkv / 8
                ? static_cast<const void*>(ckv_pages + page * ckv_stride + cell * kCkv + ch * 8)
                : static_cast<const void*>(kpe_pages + page * kpe_stride + cell * kKpe + (ch - kCkv / 8) * 8);
            cp16(dst + at_of(row, ch), src, live);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float oacc[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) oacc[i] = 0.f;
    float m_run[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l_run[2] = {0.f, 0.f};
    const float scale = sm_scale * 1.4426950408889634f;

    const int tiles = (j_hi - j_lo + kBK - 1) / kBK;
    if (tiles > 0) {
        fetch(j_lo);
        publish(0);
        if (tiles > 1) {
            fetch(j_lo + kBK);
            publish(1);
        }
        __syncthreads();
        issue(0);
    } else {
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (int i = 0; i < tiles; ++i) {
        const int st = i & 1;
        __syncthreads();
        if (i + 1 < tiles) {
            issue(st ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();

        const unsigned char* tile = sK + st * kTileBytes;
        float s[16];
#pragma unroll
        for (int e = 0; e < 16; ++e) s[e] = 0.f;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int a = 0; a < kAtoms; ++a) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                mma_s(s, desc_k(sQ + a * kAtomBytes + kk * 32),
                      desc_k(tile + a * kAtomBytes + wg * 32 * 128 + kk * 32));
            }
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        if (i + 2 < tiles) fetch(j_lo + (i + 2) * kBK);
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

        // Row g and g + 8 of the warp's 16: keys 32 wg + 8 j + 2 c (+1).
        const unsigned long long okm = s_ok[st];
        float mloc[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = wg * 32 + 8 * j + 2 * c + (e & 1);
                const float v = (okm >> key) & 1ull ? s[4 * j + e] * scale : -CUDART_INF_F;
                s[4 * j + e] = v;
                mloc[e >> 1] = fmaxf(mloc[e >> 1], v);
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            mloc[h] = fmaxf(mloc[h], __shfl_xor_sync(0xffffffffu, mloc[h], 1));
            mloc[h] = fmaxf(mloc[h], __shfl_xor_sync(0xffffffffu, mloc[h], 2));
        }
        const int row0 = wi * 16 + g;
        if (c == 0) {
            s_max[wg * kH + row0] = mloc[0];
            s_max[wg * kH + row0 + 8] = mloc[1];
        }
        bar_all();
        float corr[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float other = s_max[(wg ^ 1) * kH + row0 + 8 * h];
            const float m_new = fmaxf(m_run[h], fmaxf(mloc[h], other));
            const bool empty = m_new == -CUDART_INF_F;
            corr[h] = empty ? 1.f : (m_run[h] == -CUDART_INF_F ? 0.f : exp2f(m_run[h] - m_new));
            m_run[h] = m_new;
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int e = 2 * h; e < 2 * h + 2; ++e) {
                    const float p = empty ? 0.f : exp2f(s[4 * j + e] - m_new);
                    s[4 * j + e] = p;
                    sum += p;
                }
            }
            l_run[h] = l_run[h] * corr[h] + sum;
        }
        // P as bf16: row, then key 32 wg + 8 j + 2 c, a 128-byte swizzle atom.
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = wg * 32 + 8 * j + 2 * c;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = row0 + 8 * h;
                const __nv_bfloat162 pr = __floats2bfloat162_rn(s[4 * j + 2 * h], s[4 * j + 2 * h + 1]);
                *reinterpret_cast<__nv_bfloat162*>(sP + row * 128 + ((((key >> 3) ^ (row & 7))) << 4) + (key & 7) * 2) = pr;
            }
        }
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            oacc[4 * j + 0] *= corr[0];
            oacc[4 * j + 1] *= corr[0];
            oacc[4 * j + 2] *= corr[1];
            oacc[4 * j + 3] *= corr[1];
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        bar_all();

        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            mma_o(oacc, desc_k(sP + kk * 32),
                  desc_mn(tile + (4 * wg) * kAtomBytes + kk * 16 * 128, kAtomBytes, 8 * 128));
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        // Tile i + 2's slots, read under the scores' MMAs, land where tile
        // i's were (its copies issued a tile ago); the next top barrier
        // publishes them.
        if (i + 2 < tiles) publish(st);
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");

    // The two halves' sums meet; each warpgroup lands its 256 values.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        l_run[h] += __shfl_xor_sync(0xffffffffu, l_run[h], 1);
        l_run[h] += __shfl_xor_sync(0xffffffffu, l_run[h], 2);
    }
    const int row0 = wi * 16 + g;
    __syncthreads();
    if (c == 0) {
        s_max[wg * kH + row0] = l_run[0];
        s_max[wg * kH + row0 + 8] = l_run[1];
    }
    __syncthreads();
    float l_tot[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) l_tot[h] = l_run[h] + s_max[(wg ^ 1) * kH + row0 + 8 * h];

    const bool split = splits > 1;
    float* lse_at = split ? part_lse : lse;
    __nv_bfloat16* o_at = split ? part_o : o;
    const long long unit =
        split ? static_cast<long long>(t_row) * splits + blockIdx.z : static_cast<long long>(t_row);
    if (lse_at != nullptr && wg == 0 && c == 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float l = l_tot[h];
            lse_at[unit * H + h0 + row0 + 8 * h] = l > 0.f ? m_run[h] + log2f(l) : -CUDART_INF_F;
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float inv = l_tot[h] > 0.f ? 1.f / l_tot[h] : 0.f;
        __nv_bfloat16* orow = o_at + (unit * H + h0 + row0 + 8 * h) * kCkv + wg * 256;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const __nv_bfloat162 v =
                __floats2bfloat162_rn(oacc[4 * j + 2 * h] * inv, oacc[4 * j + 2 * h + 1] * inv);
            *reinterpret_cast<__nv_bfloat162*>(orow + 8 * j + 2 * c) = v;
        }
    }
}

}

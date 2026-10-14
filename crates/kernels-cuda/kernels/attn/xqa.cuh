#pragma once

#include "prelude/device.cuh"

// A paged decode on tensor cores, a whole GQA group a tile: one block reads
// one (request, kv head, kv chunk) of the decode plan the FA2 decode kernel
// reads, its warps taking turns over the chunk's key tiles, and lands what
// that kernel lands -- the normalized output and its base-2 log-sum-exp per
// query head, in the output plane where the plan does not split and in the
// split's partial planes where it does -- so the same fold reads it.
//
// Each warp's product is swapped (keys as the MMA's rows, the group's query
// heads, padded to eight, as its columns): S^T = K Q^T and O^T += V^T P^T on
// bf16 m16n8k16 with f32 accumulators, so a group of up to eight heads costs
// one column tile and the output sits in HEAD_DIM / 4 registers a lane. The
// keys stream through shared memory on `cp.async`, STAGES tiles deep a warp.
// A sliding window starts the walk at the window, and where the plan does
// not split the sinks scale the output as `sink_rescale` does.

namespace pie::attn::xqa {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.69314718055994530942f;
// FA2's empty state: a chunk with no keys lands this log-sum-exp.
constexpr float kEmpty = -5e4f;
// A masked score: below the running maximum's floor, so it weighs nothing.
constexpr float kMasked = -3.0e38f;

__device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm4_t(unsigned (&r)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm2(unsigned& r0, unsigned& r1, const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm2_t(unsigned& r0, unsigned& r1, const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool live) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(d), "l"(src), "r"(live ? 16 : 0)
                 : "memory");
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ unsigned pack(float lo, float hi) {
    unsigned bits;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(lo), "f"(hi));
    return bits;
}

template <int HEAD_DIM, int MT, int WARPS, int STAGES>
struct Shape {
    // A 16-byte pad a row puts the eight rows one `ldmatrix` reads in eight
    // bank groups.
    static constexpr int LD = HEAD_DIM + 8;
    static constexpr int TILE = 16 * MT;
    static constexpr int TILE_ELEMS = TILE * LD;
    // K and V a stage, STAGES stages a warp; the query rows; P a warp.
    static constexpr int STAGE_ELEMS = 2 * TILE_ELEMS;
    static constexpr int WARP_ELEMS = STAGES * STAGE_ELEMS;
    static constexpr int Q_ELEMS = 8 * LD;
    static constexpr int P_ELEMS = TILE * 8;
    static constexpr int STAGED_BYTES = WARPS * WARP_ELEMS * 2;
    // The warps' states, folded through the staged region once it drains.
    static constexpr int FOLD_BYTES = WARPS * (8 * HEAD_DIM + 16) * 4;
    static constexpr int REGION = STAGED_BYTES > FOLD_BYTES ? STAGED_BYTES : FOLD_BYTES;
    static constexpr int SMEM_BYTES = REGION + (Q_ELEMS + WARPS * P_ELEMS) * 2;
};

template <int HEAD_DIM, int MT, int WARPS, int STAGES, class S>
__global__ void __launch_bounds__(32 * WARPS) decode(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k_data,
    const bf16* __restrict__ v_data,
    const i32* __restrict__ page_indices,
    const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_len,
    u32 page_size,
    u32 stride_page,
    u32 stride_n,
    u32 stride_h,
    const i32* __restrict__ q_indptr,
    u32 q_stride_n,
    const i32* __restrict__ request_indices,
    const i32* __restrict__ kv_tile_indices,
    const i32* __restrict__ kv_chunk_size_ptr,
    const bool* __restrict__ block_valid_mask,
    bool partition_kv,
    u32 num_qo_heads,
    u32 group,
    i32 window_left,
    float sm_scale,
    bf16* __restrict__ o,
    float* __restrict__ lse,
    const S* __restrict__ sinks)
{
    using Sh = Shape<HEAD_DIM, MT, WARPS, STAGES>;
    constexpr int LD = Sh::LD;
    constexpr int TILE = Sh::TILE;
    constexpr int DT = HEAD_DIM / 16;
    constexpr int CHUNKS = HEAD_DIM / 8;  // 16-byte pieces a row

    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* staged = reinterpret_cast<bf16*>(smem_raw);
    float* fold = reinterpret_cast<float*>(smem_raw);
    bf16* qs = reinterpret_cast<bf16*>(smem_raw + Sh::REGION);
    bf16* ps_all = qs + Sh::Q_ELEMS;

    // A request's kv heads run side by side, their slices of each key row
    // one contiguous run.
    const int bx = blockIdx.y;
    const int kv_head = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    if (block_valid_mask != nullptr && !block_valid_mask[bx]) return;
    const int batch = request_indices[bx];
    const int kv_tile = kv_tile_indices[bx];
    const int first_page = page_indptr[batch];
    const int pages = page_indptr[batch + 1] - first_page;
    const int kv_len =
        pages > 0 ? (pages - 1) * static_cast<int>(page_size) + last_page_len[batch] : 0;
    int window_start = 0;
    if (window_left >= 0 && kv_len > window_left + 1) window_start = kv_len - window_left - 1;
    const int chunk = partition_kv ? *kv_chunk_size_ptr : kv_len - window_start;
    const int chunk_start = window_start + (partition_kv ? kv_tile * chunk : 0);
    const int chunk_end = partition_kv ? min(window_start + (kv_tile + 1) * chunk, kv_len) : kv_len;

    // The group's query rows, zero past the group.
    const long long q_row = static_cast<long long>(q_indptr[batch]) * q_stride_n;
    for (int e = tid; e < 8 * CHUNKS; e += 32 * WARPS) {
        const int i = e / CHUNKS;
        const int c = e % CHUNKS;
        const bool live = i < static_cast<int>(group);
        const bf16* src = q + q_row + static_cast<long long>(kv_head * group + (live ? i : 0)) * HEAD_DIM + c * 8;
        cp16(qs + i * LD + c * 8, src, live);
    }
    cp_commit();

    const int span = max(chunk_end - chunk_start, 0);
    const int tiles = (span + TILE - 1) / TILE;

    bf16* mine = staged + warp * Sh::WARP_ELEMS;
    bf16* ps = ps_all + warp * Sh::P_ELEMS;
    const long long head_off = static_cast<long long>(kv_head) * stride_h;
    // Stage the warp's j-th tile (the chunk's tile warp + j * WARPS): lane
    // r < TILE finds row r's cell once, and the copies read it off a shuffle.
    auto load = [&](int j, int stage) {
        const int tile = warp + j * WARPS;
        bf16* ks = mine + stage * Sh::STAGE_ELEMS;
        bf16* vs = ks + Sh::TILE_ELEMS;
        const int base = chunk_start + tile * TILE;
        long long cell = -1;
        if (lane < TILE) {
            const int tok = base + lane;
            if (tile < tiles && tok < chunk_end) {
                const int page = page_indices[first_page + tok / static_cast<int>(page_size)];
                const int entry = tok % static_cast<int>(page_size);
                cell = static_cast<long long>(page) * stride_page +
                       static_cast<long long>(entry) * stride_n + head_off;
            }
        }
#pragma unroll
        for (int e = lane; e < TILE * CHUNKS; e += 32) {
            const int r = e / CHUNKS;
            const int c = e % CHUNKS;
            long long at = __shfl_sync(0xffffffffu, cell, r & 31);
            const bool live = at >= 0;
            at = live ? at + c * 8 : 0;
            cp16(ks + r * LD + c * 8, k_data + at, live);
            cp16(vs + r * LD + c * 8, v_data + at, live);
        }
    };

    const int mine_tiles = tiles > warp ? (tiles - warp + WARPS - 1) / WARPS : 0;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < mine_tiles) load(s, s);
        cp_commit();
    }

    float o_acc[DT][4];
#pragma unroll
    for (int d = 0; d < DT; ++d) {
#pragma unroll
        for (int x = 0; x < 4; ++x) o_acc[d][x] = 0.f;
    }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const float scale = sm_scale * kLog2e;

    cp_wait<STAGES - 1>();
    __syncthreads();  // the query rows

    for (int j = 0; j < mine_tiles; ++j) {
        const int stage = j % STAGES;
        if (j + STAGES - 1 < mine_tiles) load(j + STAGES - 1, (j + STAGES - 1) % STAGES);
        cp_commit();
        cp_wait<STAGES - 1>();
        __syncwarp();
        const bf16* ks = mine + stage * Sh::STAGE_ELEMS;
        const bf16* vs = ks + Sh::TILE_ELEMS;
        const int base = chunk_start + (warp + j * WARPS) * TILE;

        float s[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            // Two accumulators halve the chain of dependent products.
            float s2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int x = 0; x < 4; ++x) s[mt][x] = 0.f;
#pragma unroll
            for (int kc = 0; kc < DT; ++kc) {
                unsigned a[4], b0, b1;
                ldsm4(a, ks + (mt * 16 + (lane & 15)) * LD + kc * 16 + ((lane >> 4) << 3));
                ldsm2(b0, b1, qs + (lane & 7) * LD + kc * 16 + (((lane >> 3) & 1) << 3));
                if (kc & 1) {
                    mma(s2, a, b0, b1);
                } else {
                    mma(s[mt], a, b0, b1);
                }
            }
#pragma unroll
            for (int x = 0; x < 4; ++x) s[mt][x] += s2[x];
        }
        // Scale and mask; each lane's columns are heads 2t and 2t + 1, its
        // rows keys g and g + 8 of each 16.
        float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int r0 = base + mt * 16 + g;
            const bool v0 = r0 < chunk_end, v1 = r0 + 8 < chunk_end;
            s[mt][0] = v0 ? s[mt][0] * scale : kMasked;
            s[mt][1] = v0 ? s[mt][1] * scale : kMasked;
            s[mt][2] = v1 ? s[mt][2] * scale : kMasked;
            s[mt][3] = v1 ? s[mt][3] * scale : kMasked;
            mx0 = fmaxf(mx0, fmaxf(s[mt][0], s[mt][2]));
            mx1 = fmaxf(mx1, fmaxf(s[mt][1], s[mt][3]));
        }
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
        m0 = n0;
        m1 = n1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int d = 0; d < DT; ++d) {
            o_acc[d][0] *= c0;
            o_acc[d][1] *= c1;
            o_acc[d][2] *= c0;
            o_acc[d][3] *= c1;
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const float p0 = exp2f(s[mt][0] - n0), p1 = exp2f(s[mt][1] - n1);
            const float p2 = exp2f(s[mt][2] - n0), p3 = exp2f(s[mt][3] - n1);
            l0 += p0 + p2;
            l1 += p1 + p3;
            // P as [key][head] bf16, the B operand's storage.
            *reinterpret_cast<unsigned*>(ps + (mt * 16 + g) * 8 + 2 * t) = pack(p0, p1);
            *reinterpret_cast<unsigned*>(ps + (mt * 16 + g + 8) * 8 + 2 * t) = pack(p2, p3);
        }
        __syncwarp();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            unsigned b0, b1;
            ldsm2_t(b0, b1, ps + (mt * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * 8);
#pragma unroll
            for (int d = 0; d < DT; ++d) {
                unsigned a[4];
                // V^T as the A operand: V's rows are keys (k), its columns
                // the value dims (m).
                ldsm4_t(a, vs + (mt * 16 + (lane & 7) + ((lane >> 4) << 3)) * LD + d * 16 +
                                (((lane >> 3) & 1) << 3));
                mma(o_acc[d], a, b0, b1);
            }
        }
        __syncwarp();
    }
    cp_wait<0>();

    // Each lane's partial sums over its key rows, summed across the rows.
#pragma unroll
    for (int off = 4; off < 32; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }

    __syncthreads();  // every warp is done with the staged region
    // The warps' states: m and l a head, then O as [head][dim].
    float* wm = fold + warp * (8 * HEAD_DIM + 16);
    float* wl = wm + 8;
    float* wo = wm + 16;
    if (g == 0) {
        wm[2 * t] = m0;
        wm[2 * t + 1] = m1;
        wl[2 * t] = l0;
        wl[2 * t + 1] = l1;
    }
#pragma unroll
    for (int d = 0; d < DT; ++d) {
        wo[(2 * t) * HEAD_DIM + d * 16 + g] = o_acc[d][0];
        wo[(2 * t + 1) * HEAD_DIM + d * 16 + g] = o_acc[d][1];
        wo[(2 * t) * HEAD_DIM + d * 16 + g + 8] = o_acc[d][2];
        wo[(2 * t + 1) * HEAD_DIM + d * 16 + g + 8] = o_acc[d][3];
    }
    __syncthreads();

    const bool empty = partition_kv && chunk_start >= kv_len;
    for (int e = tid; e < static_cast<int>(group) * HEAD_DIM; e += 32 * WARPS) {
        const int h = e / HEAD_DIM;
        const int d = e % HEAD_DIM;
        float m = -1e30f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float* sw = fold + w * (8 * HEAD_DIM + 16);
            if (sw[8 + h] > 0.f) m = fmaxf(m, sw[h]);
        }
        float l = 0.f, acc = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float* sw = fold + w * (8 * HEAD_DIM + 16);
            const float lw = sw[8 + h];
            if (lw > 0.f) {
                const float f = exp2f(sw[h] - m);
                l += lw * f;
                acc += sw[16 + h * HEAD_DIM + d] * f;
            }
        }
        const int qo_head = kv_head * static_cast<int>(group) + h;
        const long long row = static_cast<long long>(bx) * num_qo_heads + qo_head;
        float out = (l > 0.f && !empty) ? acc / l : 0.f;
        const float lse_v = (l > 0.f && !empty) ? m + log2f(l) : kEmpty;
        if (!partition_kv && sinks != nullptr) {
            const float sink = static_cast<float>(sinks[qo_head]);
            out *= 1.0f / (1.0f + __expf(-(lse_v * kLn2 - sink)));
        }
        o[row * HEAD_DIM + d] = bf16(out);
        if (lse != nullptr && d == 0) lse[row] = lse_v;
    }
}

}

#include "prim/device.cuh"
#include "prim/mma.cuh"

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
//
// With `ranks` above one, a cluster of that many blocks reads one (request,
// kv head) unsplit by the plan, each block a slice of its keys, and the
// cluster folds the slices through distributed shared memory: each block
// lands its share of the rows, sinks and all, with no partial plane and no
// fold launch after it.

namespace pie::attn {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.69314718055994530942f;
// FA2's empty state: a chunk with no keys lands this log-sum-exp.
constexpr float kEmpty = -5e4f;
// A masked score: below the running maximum's floor, so it weighs nothing.
constexpr float kMasked = -3.0e38f;

// A float in cluster block `rank`'s shared memory, at the address `at` holds
// in this block's.
__device__ __forceinline__ float ld_cluster(const float* at, unsigned rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const unsigned local = static_cast<unsigned>(__cvta_generic_to_shared(at));
    unsigned remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(rank));
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
    return v;
#else
    return rank == 0 ? *at : 0.f;
#endif
}

__device__ __forceinline__ void sync_cluster() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::
                     : "memory");
#endif
}

__device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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

template <int HEAD_DIM, int MT, int WARPS, int STAGES, bool FP8 = false, bool SWZ = false>
struct Shape {
    // A 16-byte pad a row puts the eight rows one `ldmatrix` reads in eight
    // bank groups; swizzled, a key row's 16-byte pieces trade places by its
    // row instead (piece c at c ^ (row % 8)), unpadded.
    static constexpr int LD = HEAD_DIM + 8;
    static constexpr int KV_LD = SWZ ? HEAD_DIM : LD;
    static constexpr int TILE = 16 * MT;
    static constexpr int TILE_ELEMS = TILE * KV_LD;
    // K and V a stage, STAGES stages a warp; the query rows; P a warp.
    static constexpr int STAGE_ELEMS = 2 * TILE_ELEMS;
    static constexpr int WARP_ELEMS = STAGES * STAGE_ELEMS;
    static constexpr int Q_ELEMS = 8 * LD;
    static constexpr int P_ELEMS = TILE * 8;
    static constexpr int STAGED_BYTES = WARPS * WARP_ELEMS * 2;
    // The warps' states, folded through the staged region once it drains.
    static constexpr int FOLD_BYTES = WARPS * (8 * HEAD_DIM + 16) * 4;
    static constexpr int REGION = STAGED_BYTES > FOLD_BYTES ? STAGED_BYTES : FOLD_BYTES;
    // An e4m3 pool's K and V land raw first, a stage a warp, and widen to
    // the bf16 tiles the products read.
    static constexpr int RAW_OFF = REGION + (Q_ELEMS + WARPS * P_ELEMS) * 2;
    static constexpr int RAW_STAGE = 2 * TILE * HEAD_DIM;
    static constexpr int SMEM_BYTES = RAW_OFF + (FP8 ? WARPS * STAGES * RAW_STAGE : 0);
};

template <int HEAD_DIM, int MT, int WARPS, int STAGES, class S, class KV, bool SWZ>
__device__ __forceinline__ void decode_body(
    const bf16* __restrict__ q,
    const KV* __restrict__ k_data,
    const KV* __restrict__ v_data,
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
    const S* __restrict__ sinks,
    u32 ranks)
{
    constexpr bool FP8 = sizeof(KV) == 1;
    static_assert(!(SWZ && FP8), "an e4m3 pool's tiles widen into padded rows");
    using Sh = Shape<HEAD_DIM, MT, WARPS, STAGES, FP8, SWZ>;
    constexpr int CHUNKS8 = HEAD_DIM / 16;  // 16-byte pieces an e4m3 row
    constexpr int LD = Sh::LD;
    constexpr int TILE = Sh::TILE;
    constexpr int DT = HEAD_DIM / 16;
    constexpr int CHUNKS = HEAD_DIM / 8;  // 16-byte pieces a row
    constexpr int KLD = Sh::KV_LD;
    // Key row r's 16-byte piece c in a K or V tile.
    auto kv_at = [](int r, int c) { return r * KLD + (SWZ ? (c ^ (r & 7)) : c) * 8; };

    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* staged = reinterpret_cast<bf16*>(smem_raw);
    float* fold = reinterpret_cast<float*>(smem_raw);
    bf16* qs = reinterpret_cast<bf16*>(smem_raw + Sh::REGION);
    bf16* ps_all = qs + Sh::Q_ELEMS;

    // A request's kv heads run side by side, their slices of each key row
    // one contiguous run.
    const int bx = blockIdx.y;
    const int slices = ranks > 1 ? static_cast<int>(ranks) : 1;
    const int kv_head = blockIdx.x / slices;
    const int rank = blockIdx.x % slices;
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
    int chunk_start = window_start + (partition_kv ? kv_tile * chunk : 0);
    int chunk_end = partition_kv ? min(window_start + (kv_tile + 1) * chunk, kv_len) : kv_len;
    if (slices > 1) {
        // The cluster's slices of the unsplit span, whole key tiles each.
        const int per = ((chunk + slices - 1) / slices + TILE - 1) / TILE * TILE;
        chunk_start = min(window_start + rank * per, kv_len);
        chunk_end = min(chunk_start + per, kv_len);
    }

    // The plan above was landed before the step's kernels; as a programmatic
    // dependent (`kernel.pdl`) the query rows and the keys wait for the
    // kernel before (the one landing this token's rows).
    pdl_wait();
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
    auto raw_of = [&](int stage) {
        return smem_raw + Sh::RAW_OFF + (warp * STAGES + stage) * Sh::RAW_STAGE;
    };
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
        if constexpr (FP8) {
            unsigned char* rk = raw_of(stage);
            unsigned char* rv = rk + TILE * HEAD_DIM;
#pragma unroll
            for (int e = lane; e < TILE * CHUNKS8; e += 32) {
                const int r = e / CHUNKS8;
                const int c = e % CHUNKS8;
                long long at = __shfl_sync(0xffffffffu, cell, r & 31);
                const bool live = at >= 0;
                at = live ? at + c * 16 : 0;
                cp16(rk + r * HEAD_DIM + c * 16, k_data + at, live);
                cp16(rv + r * HEAD_DIM + c * 16, v_data + at, live);
            }
            return;
        }
#pragma unroll
        for (int e = lane; e < TILE * CHUNKS; e += 32) {
            const int r = e / CHUNKS;
            const int c = e % CHUNKS;
            long long at = __shfl_sync(0xffffffffu, cell, r & 31);
            const bool live = at >= 0;
            at = live ? at + c * 8 : 0;
            cp16(ks + kv_at(r, c), k_data + at, live);
            cp16(vs + kv_at(r, c), v_data + at, live);
        }
    };
    // An e4m3 stage's raw rows widened into its bf16 tiles (exact).
    auto widen = [&](int stage) {
        const unsigned char* rk = raw_of(stage);
        bf16* ks = mine + stage * Sh::STAGE_ELEMS;
#pragma unroll
        for (int e = lane; e < 2 * TILE * CHUNKS8; e += 32) {
            const int half = e / (TILE * CHUNKS8);
            const int r = (e / CHUNKS8) % TILE;
            const int c = e % CHUNKS8;
            const uint4 w = *reinterpret_cast<const uint4*>(rk + half * TILE * HEAD_DIM + r * HEAD_DIM + c * 16);
            const unsigned words[4] = {w.x, w.y, w.z, w.w};
            unsigned out[8];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned x = words[u];
                out[2 * u] = pack_bf16x2(e4m3_to_f32(x & 0xff), e4m3_to_f32((x >> 8) & 0xff));
                out[2 * u + 1] = pack_bf16x2(e4m3_to_f32((x >> 16) & 0xff), e4m3_to_f32(x >> 24));
            }
            uint4* dst = reinterpret_cast<uint4*>(ks + half * Sh::TILE_ELEMS + r * LD + c * 16);
            dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
            dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
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
        if constexpr (FP8) {
            widen(stage);
            __syncwarp();
        }
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
                unsigned a[4], b[2];
                ldmatrix_x4(a, ks + kv_at(mt * 16 + (lane & 15), kc * 2 + (lane >> 4)));
                ld_b(b, qs + kc * 16, LD, lane);
                if (kc & 1) {
                    mma(s2, a, b[0], b[1]);
                } else {
                    mma(s[mt], a, b[0], b[1]);
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
            *reinterpret_cast<unsigned*>(ps + (mt * 16 + g) * 8 + 2 * t) = pack_bf16x2(p0, p1);
            *reinterpret_cast<unsigned*>(ps + (mt * 16 + g + 8) * 8 + 2 * t) = pack_bf16x2(p2, p3);
        }
        __syncwarp();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            unsigned b[2];
            ld_b_trans(b, ps + mt * 16 * 8, 8, lane);
#pragma unroll
            for (int d = 0; d < DT; ++d) {
                unsigned a[4];
                // V^T as the A operand: V's rows are keys (k), its columns
                // the value dims (m).
                ldmatrix_x4_trans(a, vs + kv_at(mt * 16 + (lane & 7) + ((lane >> 4) << 3), d * 2 + ((lane >> 3) & 1)));
                mma(o_acc[d], a, b[0], b[1]);
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
    // A head's maximum, sum and each warp's factor once, not a cell at a
    // time (a warp with no keys factors nothing in).
    __shared__ float head_m[8], head_l[8], head_f[8][WARPS];
    if (tid < static_cast<int>(group)) {
        const int h = tid;
        float m = -1e30f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float* sw = fold + w * (8 * HEAD_DIM + 16);
            if (sw[8 + h] > 0.f) m = fmaxf(m, sw[h]);
        }
        float l = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float* sw = fold + w * (8 * HEAD_DIM + 16);
            const float lw = sw[8 + h];
            const float f = lw > 0.f ? exp2f(sw[h] - m) : 0.f;
            if (lw > 0.f) l += lw * f;
            head_f[h][w] = f;
        }
        head_m[h] = m;
        head_l[h] = l;
    }
    __syncthreads();
    // A partial lands at its work item; an unsplit row where its query row
    // sits (a whole-window plan's requests are the fire's lanes).
    const long long row_of = partition_kv ? bx : q_indptr[batch];
    // Four columns a thread at a time.
    for (int e = tid; e < static_cast<int>(group) * HEAD_DIM / 4; e += 32 * WARPS) {
        const int h = e / (HEAD_DIM / 4);
        const int d = e % (HEAD_DIM / 4) * 4;
        const float m = head_m[h], l = head_l[h];
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float* sw = fold + w * (8 * HEAD_DIM + 16);
            if (sw[8 + h] > 0.f) {
                const float4 ow = *reinterpret_cast<const float4*>(sw + 16 + h * HEAD_DIM + d);
                const float f = head_f[h][w];
                acc[0] += ow.x * f;
                acc[1] += ow.y * f;
                acc[2] += ow.z * f;
                acc[3] += ow.w * f;
            }
        }
        const int qo_head = kv_head * static_cast<int>(group) + h;
        const long long row = row_of * num_qo_heads + qo_head;
        const bool live = l > 0.f && !empty;
        float out[4];
#pragma unroll
        for (int x = 0; x < 4; ++x) out[x] = live ? acc[x] / l : 0.f;
        const float lse_v = live ? m + log2f(l) : kEmpty;
        if (slices > 1) {
            // A slice's state waits in shared memory for the cluster's fold.
            float* mine_out = reinterpret_cast<float*>(smem_raw + Sh::SMEM_BYTES);
            *reinterpret_cast<float4*>(mine_out + h * HEAD_DIM + d) = make_float4(out[0], out[1], out[2], out[3]);
            if (d == 0) mine_out[8 * HEAD_DIM + h] = lse_v;
            continue;
        }
        if (!partition_kv && sinks != nullptr) {
            const float sink = static_cast<float>(sinks[qo_head]);
            const float sc = 1.0f / (1.0f + __expf(-(lse_v * kLn2 - sink)));
#pragma unroll
            for (int x = 0; x < 4; ++x) out[x] *= sc;
        }
        uint2 packed;
        packed.x = pack_bf16x2(out[0], out[1]);
        packed.y = pack_bf16x2(out[2], out[3]);
        *reinterpret_cast<uint2*>(o + row * HEAD_DIM + d) = packed;
        if (lse != nullptr && d == 0) lse[row] = lse_v;
    }
    if (slices == 1) return;

    // The cluster's fold: each block folds every slice's state for its
    // share of the (head, column) cells, read across the cluster.
    sync_cluster();
    const float* states = reinterpret_cast<const float*>(smem_raw + Sh::SMEM_BYTES);
    const long long out_row = q_indptr[batch];
    for (int e = rank * 32 * WARPS + tid; e < static_cast<int>(group) * HEAD_DIM;
         e += slices * 32 * WARPS) {
        const int h = e / HEAD_DIM;
        const int d = e % HEAD_DIM;
        // Every slice's state out at once (eight ranks at most).
        float lr[8], orr[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < slices) {
                lr[r] = ld_cluster(states + 8 * HEAD_DIM + h, static_cast<unsigned>(r));
                orr[r] = ld_cluster(states + h * HEAD_DIM + d, static_cast<unsigned>(r));
            }
        }
        float m = -1e30f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < slices) m = fmaxf(m, lr[r]);
        }
        float l = 0.f, acc = 0.f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < slices) {
                const float w = exp2f(lr[r] - m);
                l += w;
                acc += w * orr[r];
            }
        }
        float out = l > 0.f ? acc / l : 0.f;
        const float lse_v = l > 0.f ? m + log2f(l) : kEmpty;
        const int qo_head = kv_head * static_cast<int>(group) + h;
        if (sinks != nullptr) {
            const float sink = static_cast<float>(sinks[qo_head]);
            out *= 1.0f / (1.0f + __expf(-(lse_v * kLn2 - sink)));
        }
        const long long row = out_row * num_qo_heads + qo_head;
        o[row * HEAD_DIM + d] = bf16(out);
        if (lse != nullptr && d == 0) lse[row] = lse_v;
    }
    // No block leaves while another may still read its shared memory.
    sync_cluster();
}

template <int HEAD_DIM, int MT, int WARPS, int STAGES, class S, class KV = bf16>
__global__ void __launch_bounds__(32 * WARPS) decode(
    const bf16* __restrict__ q,
    const KV* __restrict__ k_data,
    const KV* __restrict__ v_data,
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
    const S* __restrict__ sinks,
    u32 ranks)
{
    decode_body<HEAD_DIM, MT, WARPS, STAGES, S, KV, false>(
        q, k_data, v_data, page_indices, page_indptr, last_page_len, page_size, stride_page, stride_n,
        stride_h, q_indptr, q_stride_n, request_indices, kv_tile_indices, kv_chunk_size_ptr,
        block_valid_mask, partition_kv, num_qo_heads, group, window_left, sm_scale, o, lse, sinks, ranks);
}

// The swizzled stamp over a bf16 pool, `MIN_CTAS` blocks an SM (as many as
// its shared memory lets share one): its registers held to them.
template <int HEAD_DIM, int MT, int WARPS, int STAGES, int MIN_CTAS, class S>
__global__ void __launch_bounds__(32 * WARPS, MIN_CTAS) decode_deep(
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
    const S* __restrict__ sinks,
    u32 ranks)
{
    decode_body<HEAD_DIM, MT, WARPS, STAGES, S, bf16, true>(
        q, k_data, v_data, page_indices, page_indptr, last_page_len, page_size, stride_page, stride_n,
        stride_h, q_indptr, q_stride_n, request_indices, kv_tile_indices, kv_chunk_size_ptr,
        block_valid_mask, partition_kv, num_qo_heads, group, window_left, sm_scale, o, lse, sinks, ranks);
}

}

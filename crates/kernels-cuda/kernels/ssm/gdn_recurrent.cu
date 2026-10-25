#include "ssm/copy.cuh"

namespace pie::ssm {

constexpr int gqa_smem_bv = 128;

template <typename StateT, bool KLast>
__global__ void gdn_step(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h = blockIdx.y;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const bool fold = folds(write_state, write_state_mask, rl);

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + ((long long)r_row * V_h + h) * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            if (fold) state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float decayed = fold
                ? state_load(state + off)
                : state_round<StateT>(state_load(state + off) * g_h);
            const float s = decayed + sk[k_idx] * delta;
            if (fold) state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out_bh[v_idx] = out_v;
    }
}

template <int BV>
__global__ void gdn_step_smem(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h  = blockIdx.z;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const bool fold = folds(write_state, write_state_mask, rl);

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* v_h = v + vh * V_d;
    const float g_h = __expf(g_log[vh]);
    const float beta_h = beta[vh];

    __nv_bfloat16* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + ((long long)r_row * V_h + h) * V_d;

    extern __shared__ __nv_bfloat16 smem_smem_step[];
    __nv_bfloat16* s_state = smem_smem_step;
    float* sq = (float*)(smem_smem_step + K_d * BV);
    float* sk = sq + K_d;

    const bool vec_tile =
        (BV == V_d) && ((V_d & 7) == 0) &&
        ((reinterpret_cast<usize>(state) & 15) == 0);
    const int n_vec = (K_d * V_d) >> 3;
    if (vec_tile) {
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(state);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(s_state);
        for (int i = threadIdx.x; i < n_vec; i += BV) dst[i] = src[i];
    } else {
        constexpr int kStageTile = 16;
        __nv_bfloat16 staged[kStageTile];
        int k = 0;
        for (; k + kStageTile <= K_d; k += kStageTile) {
            #pragma unroll
            for (int u = 0; u < kStageTile; ++u) {
                staged[u] = state[(long long)(k + u) * V_d + v_idx];
            }
            #pragma unroll
            for (int u = 0; u < kStageTile; ++u) {
                s_state[(k + u) * BV + threadIdx.x] = staged[u];
            }
        }
        for (; k < K_d; ++k) {
            s_state[k * BV + threadIdx.x] =
                state[(long long)k * V_d + v_idx];
        }
    }
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float kv_mem = 0.f;
    for (int k = 0; k < K_d; ++k) {
        float s = __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h;
        kv_mem += s * sk[k];
    }
    const float delta = (v_h[v_idx] - kv_mem) * beta_h;

    float out_v = 0.f;
    for (int k = 0; k < K_d; ++k) {
        const float sg = __bfloat162float(__float2bfloat16(
            __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h));
        float s = sg + sk[k] * delta;
        out_v += s * sq[k];

        if (vec_tile) {
            s_state[k * BV + threadIdx.x] = __float2bfloat16(s);
        } else if (fold) {
            state[(long long)k * V_d + v_idx] = __float2bfloat16(s);
        }
    }
    out_bh[v_idx] = out_v;
    if (vec_tile && fold) {
        __syncthreads();
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(s_state);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(state);
        for (int i = threadIdx.x; i < n_vec; i += BV) dst[i] = src[i];
    }
}

// One decode step of the gated delta rule, streamed: a warp owns a 16-wide
// value tile of one head and keeps its `K_d x 16` state tile in registers
// (lane `l` holds columns `(l & 1) * 8` of key rows `l / 2 + 16 m`), so the
// norms, column sums and update stay in the warp. Persistent warps prefetch
// the next tile's operands into their other shared stage. The host keeps the
// planes 16-byte aligned and the key width a multiple of 16 up to 128.
//
// The step's pieces are shared with the multi-row walk below (`gdn_walk_fused`),
// which runs them row after row on a tile it keeps in registers: a stage
// holds the tile's state (bytes 0..4096), the row's q and k head rows
// (4096..4608), its 16 value columns (4608..4640) and its two gates.
constexpr int GDN_STREAM_STAGE = 4096 + 512 + 32 + 32;
constexpr int GDN_MR = 8;
constexpr int GDN_VEC = 8;

// The tile's state rows into a stage.
__device__ __forceinline__ void gdn_issue_state(
    unsigned char* st, const __nv_bfloat16* cell, int lane, int K_d, int V_d)
{
    const int i0 = lane >> 1;
    #pragma unroll
    for (int m = 0; m < GDN_MR; ++m) {
        if (16 * m < K_d) {
            gdn_cp16(st + (m * 32 + lane) * 16, cell + (long long)(i0 + 16 * m) * V_d, true);
        }
    }
}

// One row's operands into a stage: its q and k head rows, the tile's value
// columns and the head's `[g_log | beta]` pair.
__device__ __forceinline__ void gdn_issue_row(
    unsigned char* st, const bf16* row, const float* gate_row,
    int lane, int h, int h_k, int j0, int K_d, int K_dim, int V_h, int V_d)
{
    const int c = lane & 15;
    if (c * 8 < K_d) {
        const bf16* src = row + (lane < 16 ? 0 : K_dim) + (long long)h_k * K_d + c * 8;
        gdn_cp16(st + 4096 + (lane >> 4) * 256 + c * 16, src, true);
    }
    if (lane < 2) {
        gdn_cp16(st + 4608 + lane * 16, row + 2 * K_dim + (long long)h * V_d + j0, true);
        gdn_cp4(st + 4640 + lane * 4, gate_row + lane * V_h + h, true);
    }
}

// The tile's state out of a stage, into registers.
__device__ __forceinline__ void gdn_unpack_state(
    float (&s)[GDN_MR][GDN_VEC], const unsigned char* st, int lane, int K_d)
{
    #pragma unroll
    for (int m = 0; m < GDN_MR; ++m) {
        const uint4 raw = 16 * m < K_d
            ? *reinterpret_cast<const uint4*>(st + (m * 32 + lane) * 16)
            : make_uint4(0u, 0u, 0u, 0u);
        const unsigned* w32 = reinterpret_cast<const unsigned*>(&raw);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float2 f = gdn_unpack(w32[u]);
            s[m][2 * u] = f.x;
            s[m][2 * u + 1] = f.y;
        }
    }
}

// One step of the rule on the tile: `s` becomes the state after the row the
// stage holds (unrounded, as the step computes it) and `o` this lane's part
// of the row's output columns, summed over the warp by `gdn_rows_sum`.
__device__ __forceinline__ void gdn_step_row(
    float (&s)[GDN_MR][GDN_VEC], float (&o)[GDN_VEC],
    const unsigned char* st, int lane, int K_d, float q_scale)
{
    const int i0 = lane >> 1;
    const bf16* sq = reinterpret_cast<const bf16*>(st + 4096);
    const bf16* sk = reinterpret_cast<const bf16*>(st + 4096 + 256);
    float qi[GDN_MR];
    float ki[GDN_MR];
    float qs = 0.f;
    float ks = 0.f;
    #pragma unroll
    for (int m = 0; m < GDN_MR; ++m) {
        const int i = i0 + 16 * m;
        qi[m] = 16 * m < K_d ? Elem<bf16>::to_f32(sq[i]) : 0.f;
        ki[m] = 16 * m < K_d ? Elem<bf16>::to_f32(sk[i]) : 0.f;
        qs += qi[m] * qi[m];
        ks += ki[m] * ki[m];
    }
    float vv[GDN_VEC];
    {
        const uint4 vraw = *reinterpret_cast<const uint4*>(st + 4608 + (lane & 1) * 16);
        const unsigned* w32 = reinterpret_cast<const unsigned*>(&vraw);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float2 f = gdn_unpack(w32[u]);
            vv[2 * u] = f.x;
            vv[2 * u + 1] = f.y;
        }
    }
    const float* sgate = reinterpret_cast<const float*>(st + 4640);
    const float g = __expf(sgate[0]);
    const float beta = sgate[1];

    // Both lanes of a row pair hold the row, so the warp sums it twice.
    const float nq = rsqrtf(0.5f * warp_sum(qs) + 1e-6f) * q_scale;
    const float nk = rsqrtf(0.5f * warp_sum(ks) + 1e-6f);
    #pragma unroll
    for (int m = 0; m < GDN_MR; ++m) {
        qi[m] *= nq;
        ki[m] *= nk;
    }

    float x[GDN_VEC];
    #pragma unroll
    for (int u = 0; u < GDN_VEC; ++u) x[u] = 0.f;
    #pragma unroll
    for (int m = 0; m < GDN_MR; ++m) {
        #pragma unroll
        for (int u = 0; u < GDN_VEC; ++u) x[u] += s[m][u] * g * ki[m];
    }
    gdn_rows_sum(x);

    #pragma unroll
    for (int u = 0; u < GDN_VEC; ++u) {
        x[u] = (vv[u] - x[u]) * beta;
        o[u] = 0.f;
    }
    #pragma unroll
    for (int m = 0; m < GDN_MR; ++m) {
        #pragma unroll
        for (int u = 0; u < GDN_VEC; ++u) {
            const float sg = __bfloat162float(__float2bfloat16(s[m][u] * g));
            const float sn = sg + ki[m] * x[u];
            s[m][u] = sn;
            o[u] += sn * qi[m];
        }
    }
}

// The tile's state into its slot (or a snapshot's), as bf16.
__device__ __forceinline__ void gdn_store_state(
    __nv_bfloat16* cell, const float (&s)[GDN_MR][GDN_VEC], int lane, int K_d, int V_d)
{
    const int i0 = lane >> 1;
    #pragma unroll
    for (int m = 0; m < GDN_MR; ++m) {
        if (16 * m >= K_d) break;
        uint4 packed;
        unsigned* w32 = reinterpret_cast<unsigned*>(&packed);
        #pragma unroll
        for (int u = 0; u < 4; ++u) w32[u] = pack_bf16x2(s[m][2 * u], s[m][2 * u + 1]);
        *reinterpret_cast<uint4*>(cell + (long long)(i0 + 16 * m) * V_d) = packed;
    }
}

// The row's eight output columns this lane pair lands (lanes 0 and 1 of the
// warp hold them once summed).
template <typename OutT>
__device__ __forceinline__ void gdn_store_out(OutT* dst, const float (&o)[GDN_VEC]);

template <>
__device__ __forceinline__ void gdn_store_out<float>(float* dst, const float (&o)[GDN_VEC]) {
    float4* d = reinterpret_cast<float4*>(dst);
    d[0] = make_float4(o[0], o[1], o[2], o[3]);
    d[1] = make_float4(o[4], o[5], o[6], o[7]);
}

template <>
__device__ __forceinline__ void gdn_store_out<bf16>(bf16* dst, const float (&o)[GDN_VEC]) {
    uint4 packed;
    unsigned* w32 = reinterpret_cast<unsigned*>(&packed);
    #pragma unroll
    for (int u = 0; u < 4; ++u) w32[u] = pack_bf16x2(o[2 * u], o[2 * u + 1]);
    *reinterpret_cast<uint4*>(dst) = packed;
}

template <int W>
__global__ void __launch_bounds__(W * 32) gdn_step_fused(
    const bf16* __restrict__ qkv_post,
    const float* __restrict__ gates,
    __nv_bfloat16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float* __restrict__ out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim,
    float q_scale,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    int tiles,
    const u32* __restrict__ win)
{
    extern __shared__ __align__(16) unsigned char gdn_ring[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned char* ring = gdn_ring + warp * 2 * GDN_STREAM_STAGE;
    const int vts = V_d / 16;
    const int group = V_h / K_h;
    const int K_dim = K_h * K_d;
    const int live_rows = win != nullptr ? static_cast<int>(win[0]) : 0x7fffffff;

    struct Tile {
        bool live;
        bool fold;
        int r_row;
        int h;
        int j0;
        __nv_bfloat16* cell;
    };
    auto plan = [&](int tile) {
        Tile tl;
        tl.live = false;
        if (tile >= tiles) return tl;
        const int vt = tile % vts;
        const int rest = tile / vts;
        tl.h = rest % V_h;
        const int r = rest / V_h;
        if (r >= live_rows) return tl;
        tl.r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const int slot = slot_ids[rl];
        if (slot < 0) return tl;
        tl.live = true;
        tl.fold = folds(write_state, write_state_mask, rl);
        tl.j0 = vt * 16 + (lane & 1) * GDN_VEC;
        tl.cell = state_base + (long long)slot * slot_stride_elems
            + (long long)tl.h * K_d * V_d + tl.j0;
        return tl;
    };
    auto issue = [&](const Tile& tl, int stage) {
        if (!tl.live) return;
        unsigned char* st = ring + stage * GDN_STREAM_STAGE;
        gdn_issue_state(st, tl.cell, lane, K_d, V_d);
        gdn_issue_row(st, qkv_post + (long long)tl.r_row * conv_dim,
                      gates + (long long)tl.r_row * 2 * V_h,
                      lane, tl.h, tl.h / group, tl.j0, K_d, K_dim, V_h, V_d);
    };

    const int stride = gridDim.x * W;
    int t = blockIdx.x * W + warp;
    Tile cur = plan(t);
    issue(cur, 0);
    gdn_cp_commit();
    for (int it = 0; t < tiles; t += stride, ++it) {
        const Tile nxt = plan(t + stride);
        issue(nxt, (it + 1) & 1);
        gdn_cp_commit();
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncwarp();
        if (cur.live) {
            const unsigned char* st = ring + (it & 1) * GDN_STREAM_STAGE;
            float s[GDN_MR][GDN_VEC];
            float o[GDN_VEC];
            gdn_unpack_state(s, st, lane, K_d);
            gdn_step_row(s, o, st, lane, K_d, q_scale);
            if (cur.fold) gdn_store_state(cur.cell, s, lane, K_d, V_d);
            gdn_rows_sum(o);
            if (lane < 2) {
                gdn_store_out<float>(out + ((long long)cur.r_row * V_h + cur.h) * V_d + cur.j0, o);
            }
        }
        __syncwarp();
        cur = nxt;
    }
    gdn_cp_wait_all();
}

// A lane's run of rows through the gated delta rule in one launch: the
// fused step's warp tile walks the run row after row with its state in
// registers, rounded to the state's bf16 between rows as a step that stores
// and the next that loads it would, so the walk lands the bits of as many
// one-row steps. The state lands in its slot after `commit_len` rows (and a
// snapshot's after its own), and the rows past the commit keep walking from
// there, their outputs landed and their state not.
//
// A lane's rows are `indptr[r]..indptr[r + 1]` of `qkv` and `gates` (past
// the stage word's row offset); its outputs land at the same rows of `out`,
// or, given `back`, at `back[row] - back_skip` (none where `back` is
// negative: an extended run's replayed rows answer nothing).
template <int W, typename OutT>
__global__ void __launch_bounds__(W * 32) gdn_walk_fused(
    const bf16* __restrict__ qkv,
    const float* __restrict__ gates,
    __nv_bfloat16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    OutT* __restrict__ out,
    const u32* __restrict__ indptr,
    const int* __restrict__ commit_len,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* __restrict__ snap,
    const int* __restrict__ snap_rows,
    const int* __restrict__ snap_slots,
    const i32* __restrict__ back,
    int back_skip,
    int K_h, int V_h, int K_d, int V_d, int conv_dim,
    float q_scale,
    int lanes,
    const u32* __restrict__ win)
{
    extern __shared__ __align__(16) unsigned char gdn_ring[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned char* ring = gdn_ring + warp * 2 * GDN_STREAM_STAGE;
    const int vts = V_d / 16;
    const int group = V_h / K_h;
    const int K_dim = K_h * K_d;
    const int live_lanes = win != nullptr ? min(lanes, static_cast<int>(win[2])) : lanes;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int tiles = live_lanes * V_h * vts;

    struct Tile {
        bool live;
        bool fold;
        int h;
        int j0;
        int t0;
        int rows;
        int commit;
        Snaps sn;
        __nv_bfloat16* cell;
    };
    auto plan = [&](int tile) {
        Tile tl;
        tl.live = false;
        if (tile >= tiles) return tl;
        const int vt = tile % vts;
        const int rest = tile / vts;
        tl.h = rest % V_h;
        const int r = rest / V_h;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        tl.t0 = static_cast<int>(indptr[r]);
        tl.rows = static_cast<int>(indptr[r + 1]) - tl.t0;
        if (tl.rows <= 0) return tl;
        const int slot = slot_ids[rl];
        if (slot < 0) return tl;
        tl.commit = commit_len != nullptr ? min(commit_len[rl], tl.rows) : tl.rows;
        tl.live = true;
        tl.fold = folds(write_state, write_state_mask, rl);
        tl.sn = snaps_of(snap, rl);
        tl.j0 = vt * 16 + (lane & 1) * GDN_VEC;
        tl.cell = state_base + (long long)slot * slot_stride_elems
            + (long long)tl.h * K_d * V_d + tl.j0;
        return tl;
    };
    const int stride = gridDim.x * W;
    // The first live tile at or past `tile`, stepping by the grid.
    auto live_from = [&](int tile, Tile& tl) {
        for (; tile < tiles; tile += stride) {
            tl = plan(tile);
            if (tl.live) return tile;
        }
        return tiles;
    };
    auto issue = [&](const Tile& tl, int t, int stage) {
        unsigned char* st = ring + stage * GDN_STREAM_STAGE;
        if (t == 0) gdn_issue_state(st, tl.cell, lane, K_d, V_d);
        const long long row = tl.t0 + t + row0;
        gdn_issue_row(st, qkv + row * conv_dim, gates + row * 2 * V_h,
                      lane, tl.h, tl.h / group, tl.j0, K_d, K_dim, V_h, V_d);
    };

    Tile cur;
    int tile = live_from(blockIdx.x * W + warp, cur);
    int t = 0;
    if (tile < tiles) issue(cur, 0, 0);
    gdn_cp_commit();
    float s[GDN_MR][GDN_VEC];
    for (int it = 0; tile < tiles; ++it) {
        Tile nxt = cur;
        int next_tile = tile;
        int next_t = t + 1;
        if (next_t >= cur.rows) {
            next_tile = live_from(tile + stride, nxt);
            next_t = 0;
        }
        if (next_tile < tiles) issue(nxt, next_t, (it + 1) & 1);
        gdn_cp_commit();
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncwarp();
        const unsigned char* st = ring + (it & 1) * GDN_STREAM_STAGE;
        float o[GDN_VEC];
        if (t == 0) gdn_unpack_state(s, st, lane, K_d);
        gdn_step_row(s, o, st, lane, K_d, q_scale);
        const int done = t + 1;
        if (cur.sn.e0 < cur.sn.e1 && snap_rows[cur.sn.e0] == done) {
            if (cur.fold && done <= cur.commit) {
                gdn_store_state(
                    state_base + (long long)snap_slots[cur.sn.e0] * slot_stride_elems
                        + (long long)cur.h * K_d * V_d + cur.j0,
                    s, lane, K_d, V_d);
            }
            ++cur.sn.e0;
        }
        if (cur.fold && done == cur.commit) gdn_store_state(cur.cell, s, lane, K_d, V_d);
        gdn_rows_sum(o);
        const int row = cur.t0 + t;
        const int at = back != nullptr ? (back[row] >= 0 ? back[row] - back_skip : -1)
                                       : row + row0;
        if (lane < 2 && at >= 0) {
            gdn_store_out<OutT>(out + ((long long)at * V_h + cur.h) * V_d + cur.j0, o);
        }
        // The next row reads the state as the step that stored it would.
        #pragma unroll
        for (int m = 0; m < GDN_MR; ++m) {
            #pragma unroll
            for (int u = 0; u < GDN_VEC; ++u) s[m][u] = __bfloat162float(__float2bfloat16(s[m][u]));
        }
        __syncwarp();
        if (next_t == 0) cur = nxt;
        tile = next_tile;
        t = next_t;
    }
    gdn_cp_wait_all();
}

}

#include "linear/codec.cuh"
#include "linear/moe_w4.cuh"
#include "linear/quant.cuh"

namespace pie::linear {

// `kNv`: an NVFP4 bank (an e4m3 scale a 16 codes, `tscale` an f32 tensor
// scale a weight row) in place of mxfp4's e8m0 a 32.
// `kKSplit`: K split across that many groups of the CTA's four warps, each
// walking every `kKSplit`-th chunk, their sums folded through shared memory
// before the first group lands them: a decode's few CTAs keep the card's
// memory busy with more warps, each a share of K.
template <int kNb, int kDepth, int kGlu, bool kNv = false, int kKSplit = 1>
__global__ void __launch_bounds__(128 * kKSplit) moe_select_mxfp4_mma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha,
    const float* __restrict__ tscale)
{
    constexpr int kTiles = 2;
    constexpr int kRowsWarp = kGlu != 0 ? 16 : kTiles * 16;
    constexpr int kRowsCta = 4 * kRowsWarp;
    constexpr int kSlice = 8 * kNb;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int warp = (static_cast<int>(threadIdx.x) >> 5) & 3;
    const int part = static_cast<int>(threadIdx.x) >> 7;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int row_base = static_cast<int>(blockIdx.y) * kRowsCta + warp * kRowsWarp;
    // A tile's first row: the next 16 rows, or a GLU's up rows `inter` on.
    const int span = kGlu != 0 ? inter : n;
    auto tile_row = [&](int t) { return kGlu != 0 ? row_base + t * inter : row_base + 16 * t; };
    // With K split, a warp past the bank's rows still meets the fold.
    if (kKSplit == 1 && row_base >= span) return;
    const bool rows_live = row_base < span;

    const int groups = k >> 5;
    const int chunks = (groups + 3) >> 2;
    const long long row_bytes = k >> 1;
    const u8* w_e = codes + static_cast<long long>(expert) * n * row_bytes;
    // A row's scales: one a 32-code group (mxfp4), two (NVFP4).
    const int sf_row = kNv ? 2 * groups : groups;
    const u8* s_e = scales + static_cast<long long>(expert) * n * sf_row;

    const u8* wrow[kTiles][2];
    const u8* srow[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(tile_row(t) + 8 * h + g, n - 1);
            wrow[t][h] = w_e + r * row_bytes + c * 16;
            srow[t][h] = s_e + static_cast<long long>(r) * sf_row + (kNv ? 2 * c : c);
        }
    const bf16* xrow[kNb];
#pragma unroll
    for (int b = 0; b < kNb; ++b) {
        const int tok = 8 * b + g;
        xrow[b] = tok < count
            ? act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 32
            : nullptr;
    }

    using Chunk = w4mma::Chunk<kTiles, kNb>;
    auto load = [&](int ch, Chunk& cc) {
        const bool live = ch * 4 + c < groups;
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                cc.w[t][h] = live ? w4mma::ld_stream(wrow[t][h] + ch * 64) : make_uint4(0, 0, 0, 0);
                if constexpr (kNv) {
                    cc.s[t][h] = live ? w4mma::ld_scale2(srow[t][h] + ch * 8) : 0u;
                } else {
                    cc.s[t][h] = live ? w4mma::ld_scale(srow[t][h] + ch * 4) : 127u;
                }
            }
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                cc.x[b][q] = live && xrow[b] != nullptr
                    ? w4mma::ld_act(xrow[b] + ch * 128 + q * 8)
                    : make_uint4(0, 0, 0, 0);
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

    // This group's chunks: part, part + kKSplit, ...
    const int mine = rows_live && part < chunks ? (chunks - part + kKSplit - 1) / kKSplit : 0;
    auto chunk_of = [&](int i) { return part + i * kKSplit; };
    Chunk ring[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d)
        if (d < mine) load(chunk_of(d), ring[d]);
    for (int i0 = 0; i0 < mine; i0 += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            const int i = i0 + d;
            if (i < mine) {
                bool wide = false;
#pragma unroll
                for (int t = 0; t < kTiles; ++t)
                    wide |= ring[d].s[t][0] > 128u || ring[d].s[t][1] > 128u;
                if constexpr (kNv) {
                    w4mma::compute<false, true>(ring[d], acc);
                } else if (__any_sync(0xffffffffu, wide)) {
                    w4mma::compute<true, false>(ring[d], acc);
                } else {
                    w4mma::compute<false, false>(ring[d], acc);
                }
                if (i + kDepth < mine) load(chunk_of(i + kDepth), ring[d]);
            }
        }
    }
    if (!w4mma::fold_ksplit<kKSplit>(acc, part, warp, lane, rows_live)) return;

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    // NVFP4: a weight row's tensor scale, and the 2^8 the scales held back.
    const float* t_e = kNv ? tscale + static_cast<long long>(expert) * n : nullptr;
    auto factor = [&](int row) { return kNv ? t_e[row] * 256.f : 1.f; };
    w4mma::w4_land<kGlu, kTiles, kNb>(acc, factor, b_e, order, begin, count, out, n, inter,
                                      row_base, g, c, limit, galpha);
}


/// `moe_select_mxfp4_mma` for a slice of `8 kNb` routes an expert, up to
/// 64: the slice's activations staged a 128-code chunk at a time in shared
/// memory through a `kStages` cp.async ring the CTA's warps share, so a
/// wide fire streams each expert's codes once a 64 routes, not a 32.
template <int kNb, int kStages, int kGlu, bool kNv = false, int kWarps = 4>
__global__ void __launch_bounds__(32 * kWarps) moe_select_mxfp4_mma_staged(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha,
    const float* __restrict__ tscale)
{
    // Four warps a CTA, each two m16 tiles of weight rows (a GLU's gate tile
    // and its up tile).
    constexpr int kTiles = 2;
    constexpr int kRowsWarp = kGlu != 0 ? 16 : kTiles * 16;
    constexpr int kRowsCta = kWarps * kRowsWarp;
    constexpr int kSlice = 8 * kNb;
    constexpr int kStageRuns = kSlice * 16;
    // A staging pass is `kThreads / 16` route rows, a thread a run of each.
    constexpr int kThreads = 32 * kWarps;
    constexpr int kRowStep = kThreads / 16;
    constexpr int kPasses = kSlice / kRowStep;
    static_assert(kPasses * kRowStep == kSlice, "the slice stages in whole passes");

    const int at = static_cast<int>(act_div == 1 ? blockIdx.y : blockIdx.x);
    if (at >= offsets[num_experts + 1]) return;
    const int item = work[at];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int row_base = static_cast<int>(act_div == 1 ? blockIdx.x : blockIdx.y) * kRowsCta + warp * kRowsWarp;
    const int span = kGlu != 0 ? inter : n;
    auto tile_row = [&](int t) { return kGlu != 0 ? row_base + t * inter : row_base + 16 * t; };
    // A warp past the bank's rows still stages its share of the routes.
    const bool rows_live = row_base < span;

    const int groups = k >> 5;
    const int chunks = (groups + 3) >> 2;
    const long long row_bytes = k >> 1;
    const u8* w_e = codes + static_cast<long long>(expert) * n * row_bytes;
    const int sf_row = kNv ? 2 * groups : groups;
    const u8* s_e = scales + static_cast<long long>(expert) * n * sf_row;

    const u8* wrow[kTiles][2];
    const u8* srow[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(max(tile_row(t), 0) + 8 * h + g, n - 1);
            wrow[t][h] = w_e + r * row_bytes + c * 16;
            srow[t][h] = s_e + static_cast<long long>(r) * sf_row + (kNv ? 2 * c : c);
        }

    // The runs this thread stages a chunk: rows tid / 16 + 8 j, run tid % 16.
    const int run = tid & 15;
    const bf16* src[kPasses];
#pragma unroll
    for (int j = 0; j < kPasses; ++j) {
        const int tok = (tid >> 4) + kRowStep * j;
        src[j] = tok < count ? act + static_cast<long long>(order[begin + tok] / act_div) * k + run * 8
                             : nullptr;
    }
    extern __shared__ __align__(16) uint4 staged_x[];
    auto stage = [&](int ch) {
        if (ch < chunks) {
            uint4* dst = staged_x + (ch % kStages) * kStageRuns;
            const int col = ch * 128 + run * 8;
#pragma unroll
            for (int j = 0; j < kPasses; ++j) {
                const int row = (tid >> 4) + kRowStep * j;
                const bool live = src[j] != nullptr && col < k;
                const unsigned saddr = static_cast<unsigned>(__cvta_generic_to_shared(dst + w4mma::x_slot(row, run)));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(saddr),
                             "l"(live ? src[j] + ch * 128 : act), "r"(live ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    auto load_w = [&](int ch, uint4 (&cw)[kTiles][2], unsigned (&cs)[kTiles][2]) {
        const bool live = rows_live && ch < chunks && ch * 4 + c < groups;
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                cw[t][h] = live ? w4mma::ld_stream(wrow[t][h] + ch * 64) : make_uint4(0, 0, 0, 0);
                if constexpr (kNv) {
                    cs[t][h] = live ? w4mma::ld_scale2(srow[t][h] + ch * 8) : 0u;
                } else {
                    cs[t][h] = live ? w4mma::ld_scale(srow[t][h] + ch * 4) : 127u;
                }
            }
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) stage(s);
    uint4 cw[kTiles][2], nw[kTiles][2];
    unsigned cs[kTiles][2], ns[kTiles][2];
    load_w(0, cw, cs);

    for (int ch = 0; ch < chunks; ++ch) {
        load_w(ch + 1, nw, ns);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
        // This thread's own runs of the chunk (visible to it past the wait)
        // relaid as B words before the barrier shares them.
        {
            uint4* xs_w = staged_x + (ch % kStages) * kStageRuns;
#pragma unroll
            for (int j = 0; j < kPasses; ++j) w4mma::lay_staged(xs_w + w4mma::x_slot((tid >> 4) + kRowStep * j, run));
        }
        __syncthreads();
        if (rows_live) {
            const uint4* xs = staged_x + (ch % kStages) * kStageRuns;
            bool wide = false;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) wide |= cs[t][0] > 128u || cs[t][1] > 128u;
            if constexpr (kNv) {
                w4mma::compute_staged<false, true>(cw, cs, xs, g, c, acc);
            } else if (__any_sync(0xffffffffu, wide)) {
                w4mma::compute_staged<true, false>(cw, cs, xs, g, c, acc);
            } else {
                w4mma::compute_staged<false, false>(cw, cs, xs, g, c, acc);
            }
        }
        stage(ch + kStages - 1);
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                cw[t][h] = nw[t][h];
                cs[t][h] = ns[t][h];
            }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    if (!rows_live) return;

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    const float* t_e = kNv ? tscale + static_cast<long long>(expert) * n : nullptr;
    auto factor = [&](int row) { return kNv ? t_e[row] * 256.f : 1.f; };
    w4mma::w4_land<kGlu, kTiles, kNb>(acc, factor, b_e, order, begin, count, out, n, inter,
                                      row_base, g, c, limit, galpha);
}

// The grouped arm: one block a (expert, slice of up to 32 of its routes, 128
// rows) work item from `moe_route_order`, the expert's tile decoded to bf16
// in shared memory once and multiplied by mma.sync against every route of
// the slice. K is whole 128-code tiles.
__global__ void moe_matmul_select_nvfp4_mma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const float* __restrict__ tensor_scales,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits)
{
    constexpr int kTileN = 128;
    constexpr int kTileK = 128;
    constexpr int kBatch = 16;
    constexpr int kBatches = 2;
    constexpr int kRoutes = kBatch * kBatches;
    constexpr int kLd = kTileK + 8;
    constexpr int kWarps = kTileN / 16;
    constexpr int kChunkCodes = 32;
    constexpr int kChunksPerTileK = kTileK / kChunkCodes;
    constexpr int kChunksPerThread = kTileN * kChunksPerTileK / 256;
    constexpr int kVec = 8;
    constexpr int kVecsPerTileK = kTileK / kVec;
    constexpr int kVecsPerThread = kRoutes * kVecsPerTileK / 256;
    static_assert(kRoutes * kLd * 2 >= kWarps * 16 * 16 * 4, "the C tiles fit where the activations were");
    static_assert(kTileN * kChunksPerTileK % 256 == 0 && kRoutes * kVecsPerTileK % 256 == 0, "even split");

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item / 65536;
    const int slice = item % 65536;
    const int row0 = blockIdx.y * kTileN;
    if (row0 >= n) return;
    const int begin = offsets[expert] + slice * kRoutes;
    const int end = min(offsets[expert + 1], begin + kRoutes);
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));

    const int chunks = k / kChunkCodes;
    const int groups = k / 16;
    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const float* tensor_at = tensor_scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        tensor_at = reinterpret_cast<const float*>(seat.biases);
    }
    const uint4* wq = reinterpret_cast<const uint4*>(
        codes_at + static_cast<long long>(expert) * n * (k / 2));
    const unsigned short* s2 = reinterpret_cast<const unsigned short*>(
        scales_at + static_cast<long long>(expert) * n * groups);
    const float* g = tensor_at + static_cast<long long>(expert) * n;

    __shared__ __align__(32) bf16 w_tile[kTileN][kLd];
    __shared__ __align__(32) bf16 x_tile[kRoutes][kLd];
    float* c_tile = reinterpret_cast<float*>(&x_tile[0][0]);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    struct Staged {
        uint4 chunk[kChunksPerThread];
        unsigned short scale[kChunksPerThread];
        bool live[kChunksPerThread];
        uint4 vec[kVecsPerThread];
    };

    const int group = end - begin;
    const int batches = (group + kBatch - 1) / kBatch;

    float acc[kBatches][2][4];
#pragma unroll
    for (int b = 0; b < kBatches; ++b)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[b][t][e] = 0.f;

    const bf16* vec_src[kVecsPerThread];
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
        const int t = (tid + i * 256) / kVecsPerTileK;
        vec_src[i] = nullptr;
        if (t < group) {
            const int plane_route = order[begin + t];
            vec_src[i] = act + static_cast<long long>(plane_route / act_div) * k;
        }
    }

    auto fetch = [&](int k0, Staged& st) {
#pragma unroll
        for (int i = 0; i < kChunksPerThread; ++i) {
            const int idx = tid + i * 256;
            const int r = idx / kChunksPerTileK;
            const int q = idx % kChunksPerTileK;
            const int rr = row0 + r;
            st.live[i] = rr < n;
            if (st.live[i]) {
                const long long at = static_cast<long long>(rr) * chunks + k0 / kChunkCodes + q;
                st.chunk[i] = pie_ld_line(wq + at);
                st.scale[i] = s2[at];
            }
        }
#pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int v = (tid + i * 256) % kVecsPerTileK;
            st.vec[i] = make_uint4(0u, 0u, 0u, 0u);
            if (vec_src[i] != nullptr) {
                st.vec[i] = *reinterpret_cast<const uint4*>(vec_src[i] + k0 + v * kVec);
            }
        }
    };

    auto store = [&](const Staged& st) {
#pragma unroll
        for (int i = 0; i < kChunksPerThread; ++i) {
            const int idx = tid + i * 256;
            const int r = idx / kChunksPerTileK;
            const int q = idx % kChunksPerTileK;
            unsigned* dst = reinterpret_cast<unsigned*>(&w_tile[r][q * kChunkCodes]);
            if (st.live[i]) {
                const float lo = nvfp4_scale(static_cast<u8>(st.scale[i] & 0xFFu));
                const float hi = nvfp4_scale(static_cast<u8>(st.scale[i] >> 8));
                const unsigned words[4] = {st.chunk[i].x, st.chunk[i].y, st.chunk[i].z, st.chunk[i].w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const float sv = w < 2 ? lo : hi;
                    __half2 qd[4];
                    mxfp4_unpack8(words[w], qd);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(qd[j]);
                        dst[w * 4 + j] = pack_bf16x2(f.x * sv, f.y * sv);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < kChunkCodes / 2; ++j) dst[j] = 0u;
            }
        }
#pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int idx = tid + i * 256;
            const int t = idx / kVecsPerTileK;
            const int v = idx % kVecsPerTileK;
            *reinterpret_cast<uint4*>(&x_tile[t][v * kVec]) = st.vec[i];
        }
    };

    Staged staged;
    fetch(0, staged);
    for (int k0 = 0; k0 < k; k0 += kTileK) {
        store(staged);
        __syncthreads();
        if (k0 + kTileK < k) fetch(k0 + kTileK, staged);
#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
            unsigned bfrag[4];
            ldmatrix_x4(bfrag, &w_tile[warp * 16 + (lane & 7) + ((lane >> 4) << 3)][kk + (((lane >> 3) & 1) << 3)]);
#pragma unroll
            for (int bt = 0; bt < kBatches; ++bt) {
                unsigned afrag[4];
                ldmatrix_x4(afrag, &x_tile[bt * kBatch + (lane & 15)][kk + ((lane >> 4) << 3)]);
                mma_m16n8k16(acc[bt][0], afrag, {bfrag[0], bfrag[1]});
                mma_m16n8k16(acc[bt][1], afrag, {bfrag[2], bfrag[3]});
            }
        }
        __syncthreads();
    }

    for (int bt = 0; bt < batches; ++bt) {
        {
            float* mine = c_tile + warp * 16 * 16;
            const int m = lane >> 2;
            const int col = (lane & 3) << 1;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                mine[m * 16 + t * 8 + col] = acc[bt][t][0];
                mine[m * 16 + t * 8 + col + 1] = acc[bt][t][1];
                mine[(m + 8) * 16 + t * 8 + col] = acc[bt][t][2];
                mine[(m + 8) * 16 + t * 8 + col + 1] = acc[bt][t][3];
            }
        }
        __syncthreads();
        for (int idx = lane; idx < 16 * 16; idx += 32) {
            const int m = idx / 16;
            const int nn = idx % 16;
            const int t = bt * kBatch + m;
            const int row = row0 + warp * 16 + nn;
            if (t < group && row < n) {
                const int plane_route = order[begin + t];
                out[static_cast<long long>(plane_route) * n + row] =
                    f32_to_bf16(c_tile[warp * 16 * 16 + idx] * g[row]);
            }
        }
        __syncthreads();
    }
}

}

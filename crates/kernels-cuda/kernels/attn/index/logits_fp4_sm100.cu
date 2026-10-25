#include "attn/index/index.cuh"
#include "prim/mma_sm100.cuh"

// A pooled index's scores over DeepSeek-V4.1's FP4 keys on tcgen05's
// block-scaled MMA (sm_100a), the queries read at FP4 as the reference reads
// them: for each row and each block it sees, `sum_h w_h * relu(q_h . k)`,
// landed in the scores scratch for `index_select_rows` to rank. Neither side
// is widened: e2m1 against e2m1, each 32 under its ue8m0 exponent. The shape
// is DeepGEMM's sm100 FP4 MQA logits (the keys the MMA's 128 rows, the
// queries' heads its columns, so a thread's lane is a key and its heads'
// sum stays in the thread), in this codebase's primitives:
//
//   - A pass is a row, or up to 128 / H rows of one request where the rows
//     fill the machine, their heads B's N rows. The queries land quantized
//     as they load: each 32 of a head's dims under the exponent
//     `index_q_fp4` gives it, the codes into a 64-byte swizzled tile, the
//     exponents into a scale atom, two buffers.
//   - A tile is 128 of the request's keys, landed as their pages lay them
//     (a page's codes, 64 bytes a key, then its exponents, a key's four in
//     a u32) by a bulk copy a page, six tiles a ring: a 64-byte TMA row a
//     key costs the copy engine more than the whole tile's bytes. Two warps
//     move each into the MMA's layout, the codes into the 64-byte swizzle
//     and the exponents into a scale atom, four tiles a ring.
//   - S = K Q^T is two MMAs of K 64 into TMEM (128 lanes by N), three
//     buffers; the scales reach TMEM by tcgen05.cp.
//   - The epilogue's thread is a key: each row's H columns rectified,
//     weighted and summed (packed f32x2 FMAs), its score stored with the
//     warp's 32 keys.
//
// The work is (group of rows, slice of its tiles): a decode's few rows
// slice their keys over the multiprocessors, a CTA a slice; a prefill's
// many take whole groups, a CTA every grid's.
//
// Warps 0-7 run the epilogue (0-3 the even tiles, 4-7 the odd), warp 8
// issues the tiles' copies, warp 9 owns TMEM and issues the MMAs, warps
// 10-13 load and quantize the queries, warps 14-15 move the tiles.
namespace pie::attn::index {

using tc5::u32;
using tc5::u64;

namespace logits_fp4 {

constexpr int kBN = 128;                  // keys a tile
constexpr int kRaw = 6;                   // tiles landing as their pages lay them
constexpr int kSlots = 4;                 // tiles in the MMA's layout
constexpr int kAcc = 3;                   // S buffers, the warpgroups alternating tiles
constexpr int kQThreads = 128;
constexpr int kMoveThreads = 64;
constexpr int kThreads = 320 + kQThreads + kMoveThreads;
constexpr u32 kTile = 128 * 64;           // 128 rows of 64 bytes: a tile's codes
constexpr u32 kAtom = 512;                // a scale atom: 128 rows' four exponents
constexpr u32 kRawBytes = 128 * 68;       // a tile as its pages lay it
constexpr u32 kOffQ = 0;                  // two Q buffers
constexpr u32 kOffK = 2 * kTile;
constexpr u32 kOffKSf = kOffK + kSlots * kTile;
constexpr u32 kOffQSf = kOffKSf + kSlots * kAtom;
constexpr u32 kOffW = kOffQSf + 2 * kAtom;  // [2 buffers][128] f32
constexpr u32 kOffRaw = kOffW + 2 * 128 * 4;
constexpr u32 kOffBar = kOffRaw + kRaw * kRawBytes;
constexpr u32 kCols = 512;
constexpr u32 kColKSf = kAcc * 128;        // a slot's key scales, 4 columns
constexpr u32 kColQSf = kColKSf + kSlots * 4;

__host__ __device__ constexpr u32 smem_bytes() { return kOffBar + 256u + 1024u; }

}  // namespace logits_fp4

struct Fp4Params {
    const unsigned short* q;  // [rows][heads * 128] bf16
    const unsigned short* w;  // [rows][heads] bf16, or null
    const u32* qo_indptr;
    const u32* page_indices;
    const u32* page_indptr;
    const u32* last_page_lens;
    float* scores;            // [rows][score_stride]
    const unsigned char* plane;  // the FP4 key pages
    const u32* win;
    int requests;
    int heads;
    int page_size;            // positions a page
    int score_stride;
    int top_k;
    int ratio;
    int per_shift;            // keys a page, a power of two
    int rows;
};

// A pass: rows `t[0..count)` (local) of one request, its pages, the blocks
// each row scores (0 for a row left to the ranking, which sees no more than
// `top_k`) and the tiles the most of them take.
template <int RP>
struct Fp4Pass {
    int t[RP];
    int nkeys[RP];
    int count;
    int req;
    int pages_first;
    int num_pages;
    int tiles;
};

// Whole-warp: group `group`'s passes, its `rp` rows from `rp * group` split
// where their request changes and those with nothing to score dropped, into
// `out`. A row's request is the count of requests ending at or before it,
// 32 ends a load.
template <int RP>
__device__ __forceinline__ int fp4_passes(const Fp4Params& p, int group, int rp, int live, int R,
                                          Fp4Pass<RP> (&out)[RP]) {
    const int lane = static_cast<int>(threadIdx.x % 32);
    int req[RP];
#pragma unroll
    for (int i = 0; i < RP; ++i) req[i] = 0;
    for (int base = 0; base < R - 1; base += 32) {
        const int i = base + lane;
        const int end = i < R - 1 ? static_cast<int>(p.qo_indptr[i + 1]) : 0x7fffffff;
#pragma unroll
        for (int k = 0; k < RP; ++k) req[k] += __popc(__ballot_sync(0xffffffffu, end <= rp * group + k));
    }
    int n = 0;
#pragma unroll
    for (int k = 0; k < RP; ++k) {
        const int t = rp * group + k;
        if (k >= rp || t >= live) break;
        const int r = req[k];
        const int qo_lo = static_cast<int>(p.qo_indptr[r]);
        const int new_tokens = static_cast<int>(p.qo_indptr[r + 1]) - qo_lo;
        const int first = static_cast<int>(p.page_indptr[r]);
        const int pages = static_cast<int>(p.page_indptr[r + 1]) - first;
        const int kv_len = (pages - 1) * p.page_size + static_cast<int>(p.last_page_lens[r]);
        const int total = kv_len - new_tokens + (t - qo_lo) + 1;
        int nk = total > 0 ? min(total / max(p.ratio, 1), p.score_stride) : 0;
        if (nk <= p.top_k) nk = 0;
        if (n == 0 || out[n - 1].req != r) {
            Fp4Pass<RP>& s = out[n++];
            s.count = 0;
            s.req = r;
            s.pages_first = first;
            s.num_pages = pages;
            s.tiles = 0;
        }
        Fp4Pass<RP>& s = out[n - 1];
        s.t[s.count] = t;
        s.nkeys[s.count] = nk;
        ++s.count;
        s.tiles = max(s.tiles, (nk + logits_fp4::kBN - 1) / logits_fp4::kBN);
    }
    int kept = 0;
    for (int k = 0; k < n; ++k) {
        if (out[k].tiles > 0) out[kept++] = out[k];
    }
    return kept;
}

// Every role walks the same tasks, passes and tiles: `visit(pass, t0, t1)`
// for each pass with tiles [t0, t1) in this CTA's share. A group is a row
// unless its rows fill the grid; a group's tiles slice `grid / groups`
// ways where they do not.
template <int RP, class F>
__device__ __forceinline__ void walk_tasks(const Fp4Params& p, int live, int R, F&& visit) {
    const int grid = static_cast<int>(gridDim.x);
    const int rp = (live + RP - 1) / RP >= grid ? RP : 1;
    const int groups = (live + rp - 1) / rp;
    const int slices = groups >= grid ? 1 : grid / groups;
    for (int task = static_cast<int>(blockIdx.x); task < groups * slices; task += grid) {
        const int group = task / slices;
        const int slice = task % slices;
        Fp4Pass<RP> ps[RP];
        const int np = fp4_passes(p, group, rp, live, R, ps);
        for (int k = 0; k < np; ++k) {
            const int tiles = ps[k].tiles;
            const int t0 = static_cast<int>(static_cast<long long>(tiles) * slice / slices);
            const int t1 = static_cast<int>(static_cast<long long>(tiles) * (slice + 1) / slices);
            if (t0 < t1) visit(ps[k], t0, t1);
        }
    }
}

// H heads a row (32 or 64): up to 128 / H rows a pass.
template <int H>
__global__ __launch_bounds__(logits_fp4::kThreads, 1) void index_logits_fp4(const Fp4Params p) {
    using namespace tc5;
    using namespace logits_fp4;
    constexpr int RP = 128 / H;
    const int live = p.win != nullptr ? min(static_cast<int>(p.win[0]), p.rows) : p.rows;
    const int base_row = p.win != nullptr ? static_cast<int>(p.win[1]) : 0;
    const int R = (p.win != nullptr && static_cast<int>(p.win[2]) < p.requests) ? static_cast<int>(p.win[2])
                                                                               : p.requests;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const u32 sbase = smem_addr(smem);
    float* wsm = reinterpret_cast<float*>(smem + kOffW);
    u64* raw_full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* raw_empty = raw_full + kRaw;
    u64* k_full = raw_empty + kRaw;
    u64* k_empty = k_full + kSlots;
    u64* q_full = k_empty + kSlots;
    u64* q_empty = q_full + 2;
    u64* s_full = q_empty + 2;
    u64* s_free = s_full + kAcc;
    u32* tmem_slot = reinterpret_cast<u32*>(s_free + kAcc);

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
        for (int s = 0; s < kRaw; ++s) {
            bar_init(&raw_full[s], 1);
            bar_init(&raw_empty[s], kMoveThreads);
        }
        for (int s = 0; s < kSlots; ++s) {
            bar_init(&k_full[s], kMoveThreads);
            bar_init(&k_empty[s], 1);
        }
        for (int s = 0; s < 2; ++s) {
            bar_init(&q_full[s], kQThreads);
            // The MMAs' commit and the epilogue's 256 (done with the weights).
            bar_init(&q_empty[s], 257);
        }
        for (int s = 0; s < kAcc; ++s) {
            bar_init(&s_full[s], 1);
            bar_init(&s_free[s], 128);
        }
        bar_init_fence();
    }
    if (warp == 9) {
        tmem_alloc(tmem_slot, kCols);
        tmem_relinquish();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;
    // A tile's keys as their pages lay them, a run of `seg` keys a copy:
    // the run's codes (64 bytes a key), then its exponents (4).
    const int seg_shift = min(p.per_shift, 7);
    const u32 seg_bytes = (1u << seg_shift) * index_fp4::kRow;

    if (warp == 8) {
        // The tiles' loads, a lane a run: a page's codes and exponents one
        // bulk copy where a page is at most a tile, else 128 keys' codes and
        // their exponents two. A lane's page id for the next tile loads
        // while this one issues.
        const int nb = kBN >> seg_shift;
        const long long page_bytes = static_cast<long long>(1 << p.per_shift) * index_fp4::kRow;
        const bool whole = p.per_shift <= 7;
        const int in_page = (1 << p.per_shift) - 1;
        int J = 0;
        walk_tasks<RP>(p, live, R, [&](const Fp4Pass<RP>& s, int t0, int t1) {
            const u32* ids = p.page_indices + s.pages_first;
            const int last = s.num_pages - 1;
            // A lane's run of a whole page's keys; both copies of a wider
            // page's 128 the tile's page.
            const int lane_key = whole ? static_cast<int>(lane) << seg_shift : 0;
            const bool issues = static_cast<int>(lane) < (whole ? nb : 2);
            auto page_of = [&](int jt) { return min((jt * kBN + lane_key) >> p.per_shift, last); };
            u32 id = issues ? __ldg(ids + page_of(t0)) : 0u;
            for (int jt = t0; jt < t1; ++jt, ++J) {
                const int slot = J % kRaw;
                const u32 next = issues && jt + 1 < t1 ? __ldg(ids + page_of(jt + 1)) : 0u;
                bar_wait(&raw_empty[slot], ((J / kRaw) & 1) ^ 1);
                if (lane == 0) bar_arrive_expect_tx(&raw_full[slot], kRawBytes);
                __syncwarp();
                unsigned char* dst = smem + kOffRaw + slot * kRawBytes;
                const unsigned char* page = p.plane + id * page_bytes;
                if (whole) {
                    if (issues) bulk_load(dst + lane * seg_bytes, page, seg_bytes, &raw_full[slot]);
                } else if (issues) {
                    // 128 keys of a wider page: their codes, then their exponents.
                    const int y = (jt * kBN) & in_page;
                    const u32 codes = kBN * index_fp4::kData;
                    if (lane == 0) {
                        bulk_load(dst, page + y * index_fp4::kData, codes, &raw_full[slot]);
                    } else {
                        bulk_load(dst + codes, page + (in_page + 1) * index_fp4::kData + y * index_fp4::kScale,
                                  kBN * index_fp4::kScale, &raw_full[slot]);
                    }
                }
                __syncwarp();
                id = next;
            }
        });
    } else if (warp >= 14) {
        // The tiles' moves, 64 threads: each key's 16-byte chunk c of codes
        // to row k of the 64-byte swizzle (at c ^ (k / 2 % 4)), its four
        // exponents to the scale atom (row k % 32, column k / 32).
        const int mt = static_cast<int>(threadIdx.x) - 448;
        int J = 0;
        walk_tasks<RP>(p, live, R, [&](const Fp4Pass<RP>&, int t0, int t1) {
            for (int jt = t0; jt < t1; ++jt, ++J) {
                const int rs = J % kRaw;
                const int slot = J % kSlots;
                bar_wait(&raw_full[rs], (J / kRaw) & 1);
                const unsigned char* src = smem + kOffRaw + rs * kRawBytes;
                uint4 chunk[kBN * 4 / kMoveThreads];
#pragma unroll
                for (int m = 0; m < kBN * 4 / kMoveThreads; ++m) {
                    const int u = mt + m * kMoveThreads;
                    const int k = u >> 2;
                    const int run = k >> seg_shift;
                    const int in_run = k & ((1 << seg_shift) - 1);
                    chunk[m] = *reinterpret_cast<const uint4*>(src + run * seg_bytes + in_run * index_fp4::kData +
                                                               (u & 3) * 16);
                }
                u32 sf[kBN / kMoveThreads];
#pragma unroll
                for (int m = 0; m < kBN / kMoveThreads; ++m) {
                    const int k = mt + m * kMoveThreads;
                    const int run = k >> seg_shift;
                    const int in_run = k & ((1 << seg_shift) - 1);
                    sf[m] = *reinterpret_cast<const u32*>(src + run * seg_bytes +
                                                          (index_fp4::kData << seg_shift) + in_run * 4);
                }
                bar_arrive(&raw_empty[rs]);
                bar_wait(&k_empty[slot], ((J / kSlots) & 1) ^ 1);
                unsigned char* kd = smem + kOffK + slot * kTile;
                u32* atom = reinterpret_cast<u32*>(smem + kOffKSf + slot * kAtom);
#pragma unroll
                for (int m = 0; m < kBN * 4 / kMoveThreads; ++m) {
                    const int u = mt + m * kMoveThreads;
                    const int k = u >> 2;
                    *reinterpret_cast<uint4*>(kd + k * 64 + (((u & 3) ^ ((k >> 1) & 3)) * 16)) = chunk[m];
                }
#pragma unroll
                for (int m = 0; m < kBN / kMoveThreads; ++m) {
                    const int k = mt + m * kMoveThreads;
                    atom[(k % 32) * 4 + k / 32] = sf[m];
                }
                fence_async_shared();
                bar_arrive(&k_full[slot]);
            }
        });
    } else if (warp == 9) {
        int J = 0, P = 0;
        walk_tasks<RP>(p, live, R, [&](const Fp4Pass<RP>& s, int t0, int t1) {
            const int qb = P & 1;
            const u32 idesc = idesc_scaled(kNvE2M1, kNvE2M1, 128, static_cast<u32>(s.count * H), true);
            bar_wait(&q_full[qb], (P >> 1) & 1);
            tc_fence_after();
            __syncwarp();
            if (elect_one()) tc_cp_sf_atom(tmem + kColQSf + qb * 4, sdesc_sf(sbase + kOffQSf + qb * kAtom));
            __syncwarp();
            for (int jt = t0; jt < t1; ++jt, ++J) {
                const int slot = J % kSlots;
                const int acc = J % kAcc;
                bar_wait(&k_full[slot], (J / kSlots) & 1);
                if (J >= kAcc) bar_wait(&s_free[acc], ((J / kAcc) - 1) & 1);
                tc_fence_after();
                __syncwarp();
                if (elect_one()) {
                    const u32 sfa = tmem + kColKSf + slot * 4;
                    tc_cp_sf_atom(sfa, sdesc_sf(sbase + kOffKSf + slot * kAtom));
                    const u64 ka = sdesc_sw64(sbase + kOffK + slot * kTile);
                    const u64 qa = sdesc_sw64(sbase + kOffQ + qb * kTile);
#pragma unroll
                    for (u32 k = 0; k < 2; ++k) {
                        // K 64 is 32 bytes on, two exponents on.
                        mma_mxf4(tmem + acc * 128, ka + 2 * k, qa + 2 * k, idesc_with_ids(idesc, 2 * k, 2 * k),
                                 k > 0, sfa, tmem + kColQSf + qb * 4);
                    }
                    tc_commit(&k_empty[slot]);
                    tc_commit(&s_full[acc]);
                }
                __syncwarp();
            }
            if (elect_one()) tc_commit(&q_empty[qb]);
            __syncwarp();
            ++P;
        });
    } else if (warp >= 10 && warp < 14) {
        // The queries, 128 threads, two of a head's 32-blocks a thread at a
        // time (their eight loads in flight together): each one's exponent
        // as `index_q_fp4` takes it, its 16 bytes of codes into the swizzled
        // tile (row n's 16-byte chunk c at c ^ (n / 2 % 4)) and its exponent
        // into the atom; then the head weights, a lane each.
        const int qt = static_cast<int>(threadIdx.x) - 320;
        int P = 0;
        walk_tasks<RP>(p, live, R, [&](const Fp4Pass<RP>& s, int, int) {
            const int qb = P & 1;
            bar_wait(&q_empty[qb], ((P >> 1) & 1) ^ 1);
            unsigned char* qd = smem + kOffQ + qb * kTile;
            unsigned char* qs = smem + kOffQSf + qb * kAtom;
            const int blocks = s.count * H * 4;
            for (int i0 = qt; i0 < blocks; i0 += 2 * kQThreads) {
                uint4 raw4[2][4] = {};
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const int i = i0 + b * kQThreads;
                    const int n = i / 4;
                    const int h = n % H;
                    if (i < blocks && h < p.heads) {
                        const uint4* src = reinterpret_cast<const uint4*>(
                            p.q + (static_cast<long long>(base_row + s.t[n / H]) * p.heads + h) * index_fp4::kD +
                            (i % 4) * 32);
#pragma unroll
                        for (int u = 0; u < 4; ++u) raw4[b][u] = src[u];
                    }
                }
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const int i = i0 + b * kQThreads;
                    if (i >= blocks) break;
                    const int n = i / 4;
                    const int c = i % 4;
                    const unsigned short* x = reinterpret_cast<const unsigned short*>(raw4[b]);
                    float amax = 0.f;
#pragma unroll
                    for (int u = 0; u < 32; ++u) amax = fmaxf(amax, fabsf(__uint_as_float(static_cast<u32>(x[u]) << 16)));
                    const int e = index_fp4::e8m0_ceil6(amax);
                    const float inv = __uint_as_float(static_cast<unsigned>(254 - e) << 23);
                    u32 packed[4];
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        float v[8];
#pragma unroll
                        for (int u = 0; u < 8; ++u) v[u] = __uint_as_float(static_cast<u32>(x[g * 8 + u]) << 16) * inv;
                        packed[g] = kv_rows::e2m1x8(v);
                    }
                    *reinterpret_cast<uint4*>(qd + n * 64 + ((c ^ ((n >> 1) & 3)) * 16)) =
                        make_uint4(packed[0], packed[1], packed[2], packed[3]);
                    qs[(n % 32) * 16 + (n / 32) * 4 + c] = static_cast<unsigned char>(e);
                }
            }
            {
                const int rr = qt / H;
                const int h = qt % H;
                float wv = 0.f;
                if (rr < s.count && h < p.heads) {
                    const int t_row = base_row + s.t[rr];
                    wv = p.w != nullptr
                             ? __uint_as_float(static_cast<u32>(p.w[static_cast<long long>(t_row) * p.heads + h]) << 16)
                             : 1.f;
                }
                wsm[qb * 128 + qt] = wv;
            }
            fence_async_shared();
            bar_arrive(&q_full[qb]);
            ++P;
        });
    } else {
        // The epilogue, two warpgroups taking even and odd tiles: thread l
        // of a group is the tile's key l, TMEM lane l; a row's H columns
        // land together, then four running sums meet.
        const int l = static_cast<int>(threadIdx.x) % 128;
        const int wg = static_cast<int>(warp) / 4;
        const u32 lane_base = ((warp % 4) * 32) << 16;
        int J = 0, P = 0;
        walk_tasks<RP>(p, live, R, [&](const Fp4Pass<RP>& s, int t0, int t1) {
            const int qb = P & 1;
            bar_wait(&q_full[qb], (P >> 1) & 1);
            const float4* w4 = reinterpret_cast<const float4*>(wsm + qb * 128);
            for (int jt = t0; jt < t1; ++jt, ++J) {
                const int acc = J % kAcc;
                if ((J & 1) != wg) continue;
                bar_wait(&s_full[acc], (J / kAcc) & 1);
                tc_fence_after();
                const int j = jt * kBN + l;
#pragma unroll
                for (int rr = 0; rr < RP; ++rr) {
                    if (rr >= s.count) break;
                    u32 raw32[H / 32][32];
#pragma unroll
                    for (int c = 0; c < H / 32; ++c) {
                        tmem_ld32(tmem + lane_base + acc * 128 + rr * H + c * 32, raw32[c]);
                    }
                    tmem_wait_ld();
                    float2 lo = make_float2(0.f, 0.f), hi = make_float2(0.f, 0.f);
#pragma unroll
                    for (int c = 0; c < H / 32; ++c) {
#pragma unroll
                        for (int i = 0; i < 32; i += 4) {
                            const float4 wv = w4[(rr * H + c * 32 + i) / 4];
                            const auto relu = [&](int u) { return fmaxf(__uint_as_float(raw32[c][i + u]), 0.f); };
                            lo = fma2(make_float2(relu(0), relu(1)), make_float2(wv.x, wv.y), lo);
                            hi = fma2(make_float2(relu(2), relu(3)), make_float2(wv.z, wv.w), hi);
                        }
                    }
                    if (j < s.nkeys[rr]) {
                        p.scores[static_cast<long long>(s.t[rr]) * p.score_stride + j] = (lo.x + lo.y) + (hi.x + hi.y);
                    }
                }
                tc_fence_before();
                bar_arrive(&s_free[acc]);
            }
            bar_arrive(&q_empty[qb]);
            ++P;
        });
    }
    tc_fence_before();
    __syncthreads();
    if (warp == 9) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
}

}  // namespace pie::attn::index

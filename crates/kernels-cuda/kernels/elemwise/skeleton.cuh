#pragma once

#include "elemwise/row.cuh"
#include <cuda_fp8.h>

namespace pie::elemwise {

// A plane a row skeleton reads or writes: its first element, how far apart
// its rows sit, in elements (0 for one row every row shares, as a norm's
// weight), the ids its rows are picked by (each row `r` reads the plane's
// row `ids[r]`, an embedding's; null for row `r`'s own), the vectors of 8
// it repeats every (a head's weight read by every head; 0 for a plane as
// wide as the row), and the rows an id picks among: one past them (or
// below 0) picks row 0, as the gather kernels pick it.
struct Plane {
    const void* base;
    long long row_stride;
    const int* ids;
    int period;
    int bound;
};

// Where a result's row lands: its vectors of 8 from `from` up to `to` (every
// one where `to` is 0), at its plane's row, from the plane row's first
// element; or, where `page` is set, at the paged cache row `page[r] *
// page_size + slot[r]` (none where the slot lies past the page or `valid[r]`
// is 0), as e4m3 elements where `e4m3`, each `e4m3(x / scale)`; or, where
// `blocks` is set, as a block-scaled row (kernels-ir's `BlockRows`, packed:
// bit 0 set, bit 1 e2m1 codes, bit 2 an e4m3 scale (else a power of two),
// bits 3..11 the codes a scale, 11..19 the coded columns / 8, 19..27 the
// bf16 tail's columns / 8, 27..32 the scale bytes / 4; `scale` its floor),
// a page of `page_size` rows its data rows, then its scale rows.
struct Store {
    const int* page;
    const int* slot;
    const unsigned char* valid;
    int from;
    int to;
    int page_size;
    int e4m3;
    float scale;
    // a head-major window's head and stride, in vectors (0: whole)
    int head;
    int stride;
    unsigned blocks;
};

// A block's scale of its largest magnitude `amax` (kernels-ir's
// `BlockRows::block_scale`): `q = max(amax, floor) / max`, the power of two
// at or above it as its biased exponent, or it rounded to e4m3 as its bits.
__device__ __forceinline__ float block_scale(float amax, float floor, float max, bool e4m3, unsigned char& byte) {
    const float q = __fdiv_rn(fmaxf(amax, floor), max);
    if (e4m3) {
        byte = static_cast<unsigned char>(__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3));
        const unsigned e = (byte >> 3) & 0xFu;
        const unsigned m = byte & 0x7u;
        return e != 0u ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : static_cast<float>(m) * 0.001953125f;
    }
    const unsigned bits = __float_as_uint(q);
    const unsigned biased = min(((bits >> 23) & 0xFFu) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u), 254u);
    byte = static_cast<unsigned char>(biased);
    return __uint_as_float(biased << 23);
}

// `x`'s e2m1 code: the nearest (ties to the even code), saturating at ±6,
// the sign kept on a zero.
__device__ __forceinline__ unsigned block_e2m1(float x) {
    const float a = fabsf(x);
    const unsigned m = static_cast<unsigned>(a > 0.25f) + static_cast<unsigned>(a >= 0.75f) +
                       static_cast<unsigned>(a > 1.25f) + static_cast<unsigned>(a >= 1.75f) +
                       static_cast<unsigned>(a > 2.5f) + static_cast<unsigned>(a >= 3.5f) +
                       static_cast<unsigned>(a > 5.f);
    return signbit(x) ? (m | 8u) : m;
}

// What a row skeleton fires over: its body's IN parameters and OUT results,
// every row `width` wide, `rows` rows; each of the planes' rows cut into
// `parts` of the skeleton's (a row of heads, each head its own row), and
// `table` floats of shared memory the body tables what a column alone
// computes in; then where each result lands. `kernels_cuda::elemwise::skeleton`
// mirrors it.
template <int IN, int OUT>
struct Planes {
    Plane in[IN];
    Plane out[OUT];
    int width;
    int rows;
    int parts;
    int table;
    Store store[OUT];
};

// How parameter `k` is stored, two bits of DT each: a bf16 plane, an f32
// one, or (a scalar's) an i32 element.
constexpr int kBf16 = 0;
constexpr int kF32 = 1;
constexpr int kI32 = 2;

// The sums a skeleton combines over the row in one pass of its barriers
// (`reduce_sums`).
constexpr int kBatch = 8;

// One row of planes held whole in registers: LANES threads a row (BLOCK /
// LANES rows a block), each holding VECS vectors of 8 elements of each
// parameter and each result. A body names an element by its index `e` in
// [0, N), column (lane + (e / 8) * LANES) * 8 + e % 8, and asks the skeleton
// for everything else: its parameters' values, where its results go, and
// the row's sums. Every index a body passes is a constant once its loops
// unroll, so the arrays live in registers. An element past the row
// (`!valid(e)`) reads garbage and is never stored; a body keeps it out of
// its reductions. A parameter whose bit `ONES` sets is one element a row
// (a layer's scale, every row the same, or a row's position), read once
// and named `one(k)`; its plane is never read as a row. With `READ_ONLY` no
// result lands over a parameter, so the parameters are read through the
// read-only path (`ld.global.nc`), as a hand kernel reads its `const
// __restrict__` rows; a skeleton that sums a stream in place (a parameter's
// plane also a result's) reads them as plain loads. A parameter whose bit
// `LATE` sets is a row the body first reads in a later pass: its vectors
// are issued where the body asks (`load<k>()`), not up front, so they hold
// no registers through the passes before it. A row past the fire's
// (`!live`) computes on garbage beside its block's live rows, through
// every barrier they meet, and stores nothing.
// The vector of 8 each thread's slot holds: the slot's own (`lane + p *
// LANES`), or where a body lays its vectors out (`Body::vector`, a
// pair-packed turn's turned vectors on the first threads).
struct InOrder {
    __device__ __forceinline__ static unsigned vector(unsigned s) { return s; }
};

template <class B, class = void>
struct OrderOf {
    using type = InOrder;
};
template <class B>
struct OrderOf<B, decltype(static_cast<void>(B::vector(0u)))> {
    using type = B;
};

template <int BLOCK, int LANES, int VECS, int IN, int OUT, unsigned ONES, unsigned long long DT,
          bool READ_ONLY, unsigned LATE = 0u, class ORDER = InOrder, unsigned STREAM = 0u,
          unsigned VECTORS = 0u, bool SIDE_BLOCKS = false>
struct RowSkeleton {
    static_assert((LANES & (LANES - 1)) == 0 && LANES <= BLOCK && (LANES <= 32 || LANES % 32 == 0),
                  "a row is a power-of-two run of a block's threads");
    static constexpr int PER = VECS;
    static constexpr int THREADS = BLOCK;
    static constexpr int N = VECS * 8;
    // The f32 a row shares (`share`): a row of its lanes' elements.
    static constexpr int SHARED = LANES * N;
    // Whether the grid's blocks `y == 1` sit beside the rows' (`side()`):
    // each of their rows' first warp computes what only lands, reading no
    // row.
    static constexpr bool SIDE = SIDE_BLOCKS;

    int width;
    int lane;
    int group;
    int vectors;
    bool live;
    bool beside;
    float* buf;
    float* row;
    float* tab;
    uint4* dst[OUT];
    // Each result's first f32 a row, where it is a small vector one a row
    // (`st_lane`).
    float* vdst[OUT];
    // Each small-vector parameter's (`VECTORS`) first f32 a row.
    const float* vsrc[IN];
    // Each result's window of vectors and whether its row lands at all (a
    // paged row past its page, or not live, does not).
    int from[OUT];
    int to[OUT];
    int head[OUT];
    int stride[OUT];
    bool lands[OUT];
    bool e4m3[OUT];
    float scale[OUT];
    // A block-scaled result's layout (`Store::blocks`) and its row's scale
    // bytes (its data's at `dst`).
    unsigned blocks[OUT];
    unsigned char* sdst[OUT];
    // A late parameter's first vector, issued by `load<k>()`.
    const uint4* src[IN];
    // The vector of 8 each of this thread's slots holds (`ORDER`).
    unsigned at[VECS];
    uint4 in[IN][VECS];
    // An f32 plane's vector of 8 is two words; its last four elements here.
    uint4 hi[IN][VECS];
    uint4 out[OUT][VECS];
    float ones[IN];

    __host__ __device__ static constexpr int dtype(int k) { return static_cast<int>((DT >> (2 * k)) & 3ull); }

    // Where the skeleton's row, part `part` of the planes' row `of`, starts in
    // `plane`: a row its own (`whole`), or one element.
    __device__ __forceinline__ static long long start_of(const Plane& plane, int of, int part, int width,
                                                         bool whole) {
        if (plane.row_stride == 0) return 0;
        long long src = of;
        if (plane.ids != nullptr) {
            const int id = plane.ids[of];
            src = id >= 0 && id < plane.bound ? id : 0;
        }
        return src * plane.row_stride + (whole ? static_cast<long long>(part) * width : 0);
    }

    // Every parameter's vectors are issued up front, before any body math.
    __device__ __forceinline__ RowSkeleton(const Planes<IN, OUT>& planes, int of, int part,
                                           bool live, bool beside, float* buf, float* shared)
        : width(planes.width),
          lane(static_cast<int>(threadIdx.x) % LANES),
          group(static_cast<int>(threadIdx.x) / LANES),
          vectors(planes.width / 8),
          live(live),
          beside(beside),
          buf(buf),
          row(shared + planes.table + (static_cast<int>(threadIdx.x) / LANES) * SHARED),
          tab(shared) {
#pragma unroll
        for (int p = 0; p < VECS; ++p) at[p] = ORDER::vector(static_cast<unsigned>(lane + p * LANES));
#pragma unroll
        for (int k = 0; k < OUT; ++k) {
            const Store& store = planes.store[k];
            from[k] = store.from;
            head[k] = store.head;
            stride[k] = store.stride;
            to[k] = store.to > 0 ? store.to : planes.width / 8;
            e4m3[k] = store.e4m3 != 0;
            scale[k] = store.scale;
            blocks[k] = store.blocks;
            sdst[k] = nullptr;
            lands[k] = true;
            long long row_at = start_of(planes.out[k], of, part, width, true);
            if (store.page != nullptr) {
                // the paged cache row this row's page and slot name
                const int page = live ? store.page[of] : 0;
                const int slot = live ? store.slot[of] : -1;
                lands[k] = slot >= 0 && slot < store.page_size && page >= 0 &&
                           (store.valid == nullptr || store.valid[of] != 0);
                const long long cell = static_cast<long long>(page) * store.page_size + slot;
                row_at = (lands[k] ? cell : 0) * planes.out[k].row_stride +
                         static_cast<long long>(part) * width;
            }
            // the window's first vector lands at the plane row's start
            if (blocks[k] != 0u) {
                // a page's data rows, then its scale rows
                const unsigned w = blocks[k];
                const long long coded = static_cast<long long>((w >> 11) & 0xFFu) * 8;
                const long long tail = static_cast<long long>((w >> 19) & 0xFFu) * 8;
                const long long data = ((w >> 1) & 1u ? coded / 2 : coded) + 2 * tail;
                const long long scales = static_cast<long long>((w >> 27) & 0x1Fu) * 4;
                const int page = lands[k] ? store.page[of] : 0;
                const int slot = lands[k] ? store.slot[of] : 0;
                unsigned char* base = static_cast<unsigned char*>(const_cast<void*>(planes.out[k].base)) +
                                      static_cast<long long>(page) * store.page_size * (data + scales);
                dst[k] = reinterpret_cast<uint4*>(base + slot * data);
                sdst[k] = base + static_cast<long long>(store.page_size) * data + slot * scales;
            } else if (e4m3[k]) {
                dst[k] = reinterpret_cast<uint4*>(
                    static_cast<unsigned char*>(const_cast<void*>(planes.out[k].base)) + row_at);
            } else {
                dst[k] = reinterpret_cast<uint4*>(
                    static_cast<bf16*>(const_cast<void*>(planes.out[k].base)) + row_at);
            }
        }
#pragma unroll
        for (int k = 0; k < OUT; ++k) {
            vdst[k] = static_cast<float*>(const_cast<void*>(planes.out[k].base)) +
                      start_of(planes.out[k], of, part, width, false);
        }
        if (!live) return;
#pragma unroll
        for (int k = 0; k < IN; ++k) {
            const Plane& plane = planes.in[k];
            if ((VECTORS >> k) & 1u) {
                // a small vector one a row: its row's first f32, read where
                // the body asks
                vsrc[k] = static_cast<const float*>(plane.base) + start_of(plane, of, part, width, false);
                continue;
            }
            const long long at = start_of(plane, of, part, width, !((ONES >> k) & 1u));
            if ((ONES >> k) & 1u) {
                if (dtype(k) == kI32) {
                    ones[k] = static_cast<float>(static_cast<const int*>(plane.base)[at]);
                } else if (dtype(k) == kF32) {
                    ones[k] = static_cast<const float*>(plane.base)[at];
                } else {
                    ones[k] = bf16_to_f32(static_cast<const bf16*>(plane.base)[at]);
                }
                continue;
            }
            // a side block reads no row
            if (beside) continue;
            src[k] = dtype(k) == kF32
                         ? reinterpret_cast<const uint4*>(static_cast<const float*>(plane.base) + at)
                         : reinterpret_cast<const uint4*>(static_cast<const bf16*>(plane.base) + at);
            // the vectors repeat every `period` (a head's weight); else 0
            period[k] = plane.period;
            if (!(((LATE | STREAM) >> k) & 1u)) issue(k);
        }
    }

    int period[IN];

    // Parameter `k`'s vector `p` from `src[k]`.
    __device__ __forceinline__ void issue_one(int k, int p) {
        const int slot = lane + p * LANES;
        if (slot >= vectors) return;
        const int i = static_cast<int>(at[p]);
        const int v = period[k] > 0 ? i % period[k] : i;
        if (dtype(k) == kF32) {
            in[k][p] = READ_ONLY ? __ldg(src[k] + 2 * v) : src[k][2 * v];
            hi[k][p] = READ_ONLY ? __ldg(src[k] + 2 * v + 1) : src[k][2 * v + 1];
        } else {
            in[k][p] = READ_ONLY ? __ldg(src[k] + v) : src[k][v];
        }
    }

    // Parameter `k`'s vectors from `src[k]`.
    __device__ __forceinline__ void issue(int k) {
#pragma unroll
        for (int p = 0; p < VECS; ++p) issue_one(k, p);
    }

    // The streamed parameters of `M` (a pass's reads) at vector `p`, where
    // the pass reaches it: a row too many planes wide for the registers
    // holds one vector of each at a time, read again by each pass.
    template <unsigned M>
    __device__ __forceinline__ void fetch(int p) {
        if constexpr ((M & STREAM) != 0u) {
            if (!live) return;
#pragma unroll
            for (int k = 0; k < IN; ++k)
                if (((M & STREAM) >> k) & 1u) issue_one(k, p);
        }
    }

    // A late parameter's vectors, issued where the body first wants them
    // near (the pass before the one that reads them); a parameter this
    // skeleton loaded up front is already held.
    template <int K>
    __device__ __forceinline__ void load() {
        if constexpr (((LATE >> K) & 1u) != 0u) {
            if (live) issue(K);
        }
    }

    __device__ __forceinline__ bool valid(int e) const { return lane + (e / 8) * LANES < vectors; }

    __device__ __forceinline__ unsigned col(int e) const {
        return at[e / 8] * 8u + static_cast<unsigned>(e % 8);
    }

    __device__ __forceinline__ float one(int k) const { return ones[k]; }

    // Whether slot `q` of this thread is one of the first `k` of its row: a
    // pair-packed turn's turned vectors.
    __device__ __forceinline__ bool turned(int q, unsigned k) const {
        return static_cast<unsigned>(lane + q * LANES) < k;
    }

    __device__ __forceinline__ float ld(int k, int e) const {
        if (dtype(k) == kF32) {
            const uint4& word = e % 8 < 4 ? in[k][e / 8] : hi[k][e / 8];
            return reinterpret_cast<const float*>(&word)[e % 4];
        }
        return bf16_to_f32(reinterpret_cast<const bf16*>(&in[k][e / 8])[e % 8]);
    }

    // `v` as result `k` holds it: rounded to the plane's bf16, which is
    // what a later read of the result sees. It comes back as a permute of
    // the stored pair rather than the rounded f32, so a body that keeps it
    // across a reduction holds the pair packed and unpacks it where it is
    // read: a row of 4 vectors a thread then fits 64 registers, as the hand
    // register kernel does.
    __device__ __forceinline__ float st(int k, int e, float v) {
        reinterpret_cast<bf16*>(&out[k][e / 8])[e % 8] = f32_to_bf16(v);
        const unsigned pair = reinterpret_cast<const unsigned*>(&out[k][e / 8])[e % 8 / 2];
        return __uint_as_float(__byte_perm(pair, 0, e % 2 == 0 ? 0x1044 : 0x3244));
    }

    // `v` rounded to bf16 and back: a value the program a body came from
    // stores between two ops, read on as it stored it.
    __device__ __forceinline__ float round_bf16(float v) const { return bf16_to_f32(f32_to_bf16(v)); }

    // An angle as the hand kernels turn one (`__sincosf`).
    __device__ __forceinline__ float sin(float v) const { return __sinf(v); }
    __device__ __forceinline__ float cos(float v) const { return __cosf(v); }

    // Partials combined over runs of S threads, in place, the total to each
    // of them: a butterfly over a warp's lanes, and past a warp the warps'
    // totals in the butterfly's order too (lane `i` and `i ^ off`), as a
    // block's reduction folds its warps (`block_reduce_sum_fast`, the eight
    // a block's first warp folds with its lanes past them zero).
    template <int S, bool MAX, int K>
    __device__ __forceinline__ void reduce_runs(float (&a)[K]) {
        static_assert((S & (S - 1)) == 0 && S <= BLOCK, "a run is a power of two of a block's threads");
        auto join = [](float x, float y) { return MAX ? fmaxf(x, y) : x + y; };
        constexpr int kLanes = S < 32 ? S : 32;
#pragma unroll
        for (int p = 0; p < K; ++p)
#pragma unroll
            for (int off = kLanes / 2; off > 0; off >>= 1) a[p] = join(a[p], __shfl_xor_sync(0xffffffffu, a[p], off));
        if constexpr (S > 32) {
            constexpr int kWarps = BLOCK / 32;
            constexpr int kSpan = S / 32;
            const int warp = static_cast<int>(threadIdx.x) >> 5;
            if ((threadIdx.x & 31) == 0) {
#pragma unroll
                for (int p = 0; p < K; ++p) buf[p * kWarps + warp] = a[p];
            }
            __syncthreads();
            const int first = warp / kSpan * kSpan;
#pragma unroll
            for (int p = 0; p < K; ++p) {
                float t[kSpan];
#pragma unroll
                for (int w = 0; w < kSpan; ++w) t[w] = buf[p * kWarps + first + w];
#pragma unroll
                for (int off = kSpan / 2; off > 0; off >>= 1)
#pragma unroll
                    for (int i = 0; i < off; ++i) t[i] = join(t[i], t[i + off]);
                a[p] = t[0];
            }
            __syncthreads();
        }
    }

    // `K` sums over the row at once, in place, a batch of up to kBatch
    // through one pass of the block's barriers: a blend's candidates' square
    // sums and dots.
    template <int K>
    __device__ __forceinline__ void reduce_sums(float (&a)[K]) {
#pragma unroll
        for (int at0 = 0; at0 < K; at0 += kBatch) {
            constexpr int kMost = K < kBatch ? K : kBatch;
            float batch[kMost];
#pragma unroll
            for (int j = 0; j < kMost; ++j) batch[j] = at0 + j < K ? a[at0 + j] : 0.f;
            reduce_runs<LANES, false>(batch);
#pragma unroll
            for (int j = 0; j < kMost; ++j)
                if (at0 + j < K) a[at0 + j] = batch[j];
        }
    }

    __device__ __forceinline__ float reduce_sum(float local) {
        if constexpr (LANES == BLOCK) {
            return block_reduce_sum_fast<BLOCK>(local, buf);
        } else {
            float a[1] = {local};
            reduce_runs<LANES, false>(a);
            return a[0];
        }
    }

    __device__ __forceinline__ float reduce_max(float local) {
        if constexpr (LANES == BLOCK) {
            return block_reduce_max_fast<BLOCK>(local, buf);
        } else {
            float a[1] = {local};
            reduce_runs<LANES, true>(a);
            return a[0];
        }
    }

    // Each vector's partial combined over its head of S vectors, in place:
    // a head's lanes sit side by side (S divides LANES).
    template <int S, bool MAX>
    __device__ __forceinline__ void reduce_heads(float (&a)[VECS]) {
        static_assert((S & (S - 1)) == 0 && S <= LANES && LANES % S == 0,
                      "a head is a power-of-two run of a row's lanes");
        reduce_runs<S, MAX>(a);
    }

    template <int S>
    __device__ __forceinline__ void reduce_sum_heads(float (&a)[VECS]) {
        reduce_heads<S, false>(a);
    }

    template <int S>
    __device__ __forceinline__ void reduce_max_heads(float (&a)[VECS]) {
        reduce_heads<S, true>(a);
    }

    // A gather within the row: each element shared at its column, the row
    // complete for every thread (`sync`), then any column read back.
    __device__ __forceinline__ void share(int e, float v) { row[col(e)] = v; }

    // A row within a warp shares its row with that warp's threads alone.
    __device__ __forceinline__ void sync() {
        if constexpr (LANES <= 32) {
            __syncwarp();
        } else {
            __syncthreads();
        }
    }

    // The block's table complete for every thread.
    __device__ __forceinline__ void sync_block() { __syncthreads(); }

    // The block's table complete for every thread by the first read of it,
    // which follows a row's share (`sync()`): that share meets the whole
    // block where a row is more than a warp's threads; a warp's row meets
    // its warp alone, so the table takes a barrier of its own.
    __device__ __forceinline__ void sync_table() {
        if constexpr (LANES <= 32) __syncthreads();
    }

    __device__ __forceinline__ float peer(unsigned c) const { return row[c < SHARED ? c : 0u]; }

    // What a column alone computes, tabled once a block: the body writes
    // entry `c` of its table (each thread its own entries, `f.sync_block()`
    // after) and every row reads it back.
    __device__ __forceinline__ float* table() const { return tab; }

    __device__ __forceinline__ unsigned thread() const { return threadIdx.x; }

    // A small vector one a row lies over the lanes of a warp the row takes
    // (SPAN of them, its lanes past them computing it alike): value `i` in
    // lane `i % SPAN`'s slot `i / SPAN`, a thread holding `(n + SPAN - 1)
    // / SPAN` slots of an `n`-vector. A value one a row (a scalar) is the
    // same in every lane.
    static constexpr int SPAN = LANES < 32 ? LANES : 32;

    // Whether this block is a side block (`SIDE`).
    __device__ __forceinline__ bool side() const { return beside; }

    // This thread's lane among them.
    __device__ __forceinline__ int vlane() const { return lane % SPAN; }

    // Whether this thread is in the row's first warp: the one that computes
    // what only lands (a small-vector result no pass reads) and lands it.
    __device__ __forceinline__ bool lead() const { return lane < SPAN; }

    // Value `i` of small-vector parameter `k` (any lane may read any).
    __device__ __forceinline__ float vec(int k, int i) const { return live ? __ldg(vsrc[k] + i) : 0.f; }

    // `x` from lane `src` of this row's lanes.
    __device__ __forceinline__ static float shfl(float x, int src) {
        return __shfl_sync(0xffffffffu, x, src, SPAN);
    }
    __device__ __forceinline__ static unsigned shfl(unsigned x, int src) {
        return __shfl_sync(0xffffffffu, x, src, SPAN);
    }
    __device__ __forceinline__ static bool shfl(bool x, int src) {
        return __shfl_sync(0xffffffffu, static_cast<int>(x), src, SPAN) != 0;
    }

    // Value `j` of small vector `v` (`j` the same in every lane, a constant
    // once the body's loops unroll), to every lane: a shuffle.
    template <class T, int S>
    __device__ __forceinline__ T pick(const T (&v)[S], int j) const {
        return shfl(v[j / SPAN], j % SPAN);
    }

    // Value `j` of small vector `v`, `j` a lane's own: a shuffle of each
    // slot, the one `j` lies in taken.
    template <class T, int S>
    __device__ __forceinline__ T gather(const T (&v)[S], int j) const {
        T got = shfl(v[0], j % SPAN);
#pragma unroll
        for (int s = 1; s < S; ++s) {
            const T o = shfl(v[s], j % SPAN);
            got = j / SPAN == s ? o : got;
        }
        return got;
    }

    // Entry `i` of a table of up to 36 indices below 32, five bits each,
    // twelve a word (0 past them).
    __device__ __forceinline__ static int index(int i, unsigned long long a, unsigned long long b,
                                                unsigned long long c) {
        const unsigned long long w = i < 12 ? a : (i < 24 ? b : c);
        return i < 36 ? static_cast<int>((w >> (5 * (i % 12))) & 31ull) : 0;
    }

    // Value `i` of small-vector result `K`, one a row, landed by the lane
    // that holds it in the row's first warp.
    template <int K>
    __device__ __forceinline__ void st_lane(int i, float v) {
        if (live && lane < SPAN) vdst[K][i] = v;
    }

    // Result `K`'s row to memory: a body flushes a result once its last
    // element is written, so the store overlaps whatever follows.
    template <int K>
    __device__ __forceinline__ void flush() {
        if (blocks[K] != 0u) {
            flush_blocks<K>();
            return;
        }
        if (!live || !lands[K]) return;
#pragma unroll
        for (int p = 0; p < VECS; ++p) {
            const int slot = lane + p * LANES;
            const int i = static_cast<int>(at[p]);
            if (slot >= vectors || i < from[K] || i >= to[K]) continue;
            // the vector the window lands it at: a head's of every stride
            int dst_at = i - from[K];
            if (head[K] > 0) {
                const int j = dst_at % stride[K];
                if (j >= head[K]) continue;
                dst_at = dst_at / stride[K] * head[K] + j;
            }
            if (e4m3[K]) {
                // the bf16 the result holds, each element to e4m3
                const bf16* held = reinterpret_cast<const bf16*>(&out[K][p]);
                __align__(8) __nv_fp8_e4m3 q[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) q[e] = __nv_fp8_e4m3(bf16_to_f32(held[e]) / scale[K]);
                reinterpret_cast<uint2*>(dst[K])[dst_at] = *reinterpret_cast<const uint2*>(q);
            } else {
                dst[K][dst_at] = out[K][p];
            }
        }
    }

    // Result `K`'s row as a block-scaled cache row: each block's largest
    // magnitude over the lanes that hold it (a block's vectors sit on side
    // by side lanes, a row's vectors in order), its scale, then its codes;
    // the tail's vectors as they are. Every thread takes the max, live or
    // not, as the lanes exchange it.
    template <int K>
    __device__ __forceinline__ void flush_blocks() {
        const unsigned w = blocks[K];
        const bool e2m1 = ((w >> 1) & 1u) != 0u;
        const bool e4m3_scale = ((w >> 2) & 1u) != 0u;
        const int block = static_cast<int>((w >> 3) & 0xFFu);
        const int coded = static_cast<int>((w >> 11) & 0xFFu) * 8;
        const int scales = static_cast<int>((w >> 27) & 0x1Fu) * 4;
        const int per = block / 8;
        const float max = e2m1 ? 6.f : 448.f;
        const bool lands_row = live && lands[K];
        unsigned char* data = reinterpret_cast<unsigned char*>(dst[K]);
        // each vector's largest, then its block's: over side by side lanes
        // where the row's vectors sit in order, else (a pair-packed turn's
        // order) through shared memory, each row's blocks its own
        float amaxes[VECS];
#pragma unroll
        for (int p = 0; p < VECS; ++p) {
            const int i = static_cast<int>(at[p]);
            const bf16* h = reinterpret_cast<const bf16*>(&out[K][p]);
            float amax = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(bf16_to_f32(h[e])));
            amaxes[p] = lane + p * LANES < vectors && i * 8 < coded ? amax : 0.f;
        }
        if constexpr (is_same<ORDER, InOrder>::value) {
#pragma unroll
            for (int p = 0; p < VECS; ++p)
                for (int off = 1; off < per; off <<= 1)
                    amaxes[p] = fmaxf(amaxes[p], __shfl_xor_sync(0xffffffffu, amaxes[p], off));
        } else {
            constexpr int kBlocks = 64;
            __shared__ unsigned blockmax[(BLOCK / LANES) * kBlocks];
            for (int b = static_cast<int>(threadIdx.x); b < (BLOCK / LANES) * kBlocks; b += BLOCK) blockmax[b] = 0u;
            __syncthreads();
#pragma unroll
            for (int p = 0; p < VECS; ++p) {
                const int i = static_cast<int>(at[p]);
                if (lane + p * LANES < vectors && i * 8 < coded)
                    atomicMax(&blockmax[group * kBlocks + i / per], __float_as_uint(amaxes[p]));
            }
            __syncthreads();
#pragma unroll
            for (int p = 0; p < VECS; ++p) {
                const int i = static_cast<int>(at[p]);
                if (lane + p * LANES < vectors && i * 8 < coded)
                    amaxes[p] = __uint_as_float(blockmax[group * kBlocks + i / per]);
            }
        }
#pragma unroll
        for (int p = 0; p < VECS; ++p) {
            const int slot = lane + p * LANES;
            const int i = static_cast<int>(at[p]);
            const bool held = slot < vectors;
            const bf16* h = reinterpret_cast<const bf16*>(&out[K][p]);
            const float amax = amaxes[p];
            if (!lands_row || !held) continue;
            const int c0 = i * 8;
            if (c0 >= coded) {
                // the tail's bf16, as the result holds it
                uint2* t = reinterpret_cast<uint2*>(data + (e2m1 ? coded / 2 : coded) + (c0 - coded) * 2);
                t[0] = make_uint2(out[K][p].x, out[K][p].y);
                t[1] = make_uint2(out[K][p].z, out[K][p].w);
                continue;
            }
            unsigned char byte;
            const float s = block_scale(amax, scale[K], max, e4m3_scale, byte);
            if (i % per == 0) sdst[K][i / per] = byte;
            if (i == 0) {
                for (int b = coded / block; b < scales; ++b) sdst[K][b] = 0;
            }
            if (e2m1) {
                unsigned packed = 0u;
#pragma unroll
                for (int e = 0; e < 8; ++e) packed |= block_e2m1(__fdiv_rn(bf16_to_f32(h[e]), s)) << (4 * e);
                *reinterpret_cast<unsigned*>(data + c0 / 2) = packed;
            } else {
                __align__(8) unsigned char q[8];
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    q[e] = static_cast<unsigned char>(
                        __nv_cvt_float_to_fp8(__fdiv_rn(bf16_to_f32(h[e]), s), __NV_SATFINITE, __NV_E4M3));
                *reinterpret_cast<uint2*>(data + c0) = *reinterpret_cast<const uint2*>(q);
            }
        }
    }
};

// A row skeleton of LANES threads a row: `body(f)` computes the row in hooks
// over `f`, its own fields (a norm's eps) passed by value. The rows past a
// staged window's live count are left as they are. A body that tables its
// columns' values is fired with `planes.table` floats of dynamic shared
// memory, and one that shares its row BLOCK * VECS * 8 floats more. Under
// `SIDE` the grid is two blocks high, the blocks `y == 1` its side blocks.
template <class Body, int BLOCK, int LANES, int VECS, int IN, int OUT, unsigned ONES = 0u,
          unsigned long long DT = 0ull, bool READ_ONLY = false, unsigned LATE = 0u,
          unsigned STREAM = 0u, unsigned VECTORS = 0u, bool SIDE = false>
__global__ void __launch_bounds__(BLOCK) row_skeleton(
    const Planes<IN, OUT> planes,
    const Body body,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    constexpr int ROWS = BLOCK / LANES;
    // A staged window's rows are the planes' rows; the skeleton's are their
    // parts.
    const int live_rows =
        win != nullptr ? min(planes.rows, static_cast<int>(win[0]) * planes.parts) : planes.rows;
    const int first = static_cast<int>(blockIdx.x) * ROWS;
    if (first >= live_rows) return;
    const int row = first + static_cast<int>(threadIdx.x) / LANES;
    // the planes' row and the part of it
    int of = row;
    int part = 0;
    if (planes.parts > 1) {
        of = row / planes.parts;
        part = row - of * planes.parts;
    }
    if (win != nullptr) of += static_cast<int>(win[1]);

    __shared__ float buf[(VECS > kBatch ? VECS : kBatch) * (BLOCK / 32) + 1];
    extern __shared__ float shared[];
    RowSkeleton<BLOCK, LANES, VECS, IN, OUT, ONES, DT, READ_ONLY, LATE,
                typename OrderOf<Body>::type, STREAM, VECTORS, SIDE>
        f(
        planes, of, part, row < live_rows, SIDE && blockIdx.y == 1, buf, shared);
    body(f);
}

}

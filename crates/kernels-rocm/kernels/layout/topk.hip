#pragma once

// The best columns of each row, as greedy sampling and top-k read the
// logits every decode step: one block per row, rows as wide as a vocabulary
// (~250K). A block streams its row 16 bytes per lane. A wave64 VALU op is
// four cycles, so the scans work a chunk at a time: one max per column, and
// a closer look only at the rare chunk that could matter.
//
// Order: the larger value wins and a tie goes to the lower column. A NaN
// never ranks. An empty slot is (-inf, kNone), which every column beats,
// -inf included, so slots need no case of their own.

#include "prelude/device.hip"

namespace pie::layout {

constexpr u32 kNone = 0xFFFFFFFFu;

__device__ __forceinline__ bool topk_beats(float av, u32 ai, float bv, u32 bi) {
    return av > bv || (av == bv && ai < bi);
}

// A row's columns as a block streams them: in rounds of one chunk per
// thread, PER columns a chunk (16 bytes, or one element when the row does
// not allow 16-byte lanes). Whole rounds load unconditionally, UNROLL of
// them in flight at once; only the one ragged last round tests its chunks.
// The round count is the block's, so a visitor runs with every lane active
// and may use wavefront operations.
template <class T, int PER>
struct Chunk;

template <class T>
struct Chunk<T, 1> {
    T raw;
    __device__ __forceinline__ float operator[](int) const { return to_f32(raw); }
};

template <class T, int PER>
struct Chunk {
    static_assert(PER * sizeof(T) == 16, "a chunk is one element or one 16-byte lane");
    b128 raw;
    __device__ __forceinline__ float operator[](int j) const {
        if constexpr (is_same<T, bf16>::value) {
            // Widened by bits: LLVM folds fmaxf of two widened bf16 into a
            // bf16 max, which gfx942 lacks and emulates by rounding to bf16
            // after every max.
            const u32 w = raw.w[j / 2];
            return __builtin_bit_cast(float, j % 2 == 0 ? w << 16 : w & 0xffff0000u);
        } else {
            return to_f32(reinterpret_cast<const T*>(&raw)[j]);
        }
    }
};

template <class T, int PER>
__device__ __forceinline__ Chunk<T, PER> load_chunk(const T* __restrict__ src, int idx) {
    return Chunk<T, PER>{reinterpret_cast<const decltype(Chunk<T, PER>::raw)*>(src)[idx]};
}

// The chunk's largest column value, NaNs passed over (fmaxf keeps the
// number); NaN only when every column is.
template <class T, int PER>
__device__ __forceinline__ float chunk_max(const Chunk<T, PER>& c) {
    float m = c[0];
#pragma unroll
    for (int j = 1; j < PER; ++j) m = fmaxf(m, c[j]);
    return m;
}

// f(chunk, first column, held): held is false for a chunk past the row.
template <class T, int PER, int UNROLL, class F>
__device__ __forceinline__ void scan_chunks(const T* __restrict__ src, int chunks, F&& f) {
    const int n = blockDim.x, t = threadIdx.x;
    const int whole = chunks / n;
    int r = 0;
    for (; r + UNROLL <= whole; r += UNROLL) {
        Chunk<T, PER> c[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) c[u] = load_chunk<T, PER>(src, (r + u) * n + t);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) f(c[u], static_cast<u32>(((r + u) * n + t) * PER), true);
    }
    for (; r < whole; ++r) f(load_chunk<T, PER>(src, r * n + t), static_cast<u32>((r * n + t) * PER), true);
    if (whole * n < chunks) {
        const int idx = whole * n + t;
        const bool held = idx < chunks;
        f(load_chunk<T, PER>(src, held ? idx : chunks - 1), static_cast<u32>(idx * PER), held);
    }
}

// Whether a row streams in 16-byte chunks: its width and base allow them.
template <class T>
__device__ __forceinline__ bool chunked(const T* src, int width) {
    return width % (16 / sizeof(T)) == 0 && aligned16(src);
}

// The pair that beats every other lane's: every lane gets it.
__device__ __forceinline__ void wave_best(float& v, u32& i) {
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) {
        const float ov = __shfl_xor(v, off, kWave);
        const u32 oi = __shfl_xor(i, off, kWave);
        if (topk_beats(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
}

// The thread's best value and the first column holding it. A thread's
// chunks arrive in increasing order, so a strict `>` keeps the first chunk
// of a tie; that chunk is read again at the end for the column within it.
template <class T, int PER, int UNROLL>
__device__ __forceinline__ void argmax_scan(const T* __restrict__ src, int chunks, float& best, u32& at) {
    u32 first = kNone;
    scan_chunks<T, PER, UNROLL>(src, chunks, [&](const Chunk<T, PER>& c, u32 col, bool held) {
        const float m = held ? chunk_max(c) : __builtin_nanf("");
        // Bitwise, not short-circuit: two selects and no branch.
        const bool take = (m > best) | ((first == kNone) & (m == m));
        best = take ? m : best;
        first = take ? col : first;
    });
    at = first;
    if (PER > 1 && first != kNone) {
        const Chunk<T, PER> c = load_chunk<T, PER>(src, first / PER);
#pragma unroll
        for (int j = PER - 1; j >= 0; --j) at = c[j] == best ? first + j : at;
    }
}

// y[row][column] = the row's best column (0 for a row of NaNs).
template <class T>
__global__ __launch_bounds__(1024) void argmax_rows(const T* __restrict__ x, i32* __restrict__ y, int width,
                                                    int depth, int column, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = blockIdx.x;
    const T* src = x + row * width;

    float best = neg_inf();
    u32 at = kNone;
    if (chunked(src, width)) {
        argmax_scan<T, 16 / sizeof(T), 4>(src, width / (16 / sizeof(T)), best, at);
    } else {
        argmax_scan<T, 1, 4>(src, width, best, at);
    }

    __shared__ float sv[16];
    __shared__ u32 si[16];
    const int w = threadIdx.x / kWave;
    wave_best(best, at);
    if (lane() == 0) {
        sv[w] = best;
        si[w] = at;
    }
    lds_barrier();
    if (threadIdx.x == 0) {
        for (int k = 1; k < static_cast<int>(blockDim.x) / kWave; ++k) {
            if (topk_beats(sv[k], si[k], best, at)) {
                best = sv[k];
                at = si[k];
            }
        }
        y[row * depth + column] = static_cast<i32>(at == kNone ? 0u : at);
    }
}

// A column as one 64-bit key whose unsigned order is topk_beats: the
// value's order above the complemented column. No column's low word is 0,
// so key 0 is below every column. -0 is keyed as +0, which it equals.
__device__ __forceinline__ u64 topk_key(float v, u32 col) {
    const u32 b = __builtin_bit_cast(u32, v + 0.f);
    const u32 hi = (b & 0x80000000u) != 0u ? ~b : b | 0x80000000u;
    return static_cast<u64>(hi) << 32 | ~col;
}
__device__ __forceinline__ float key_value(u64 k) {
    const u32 hi = static_cast<u32>(k >> 32);
    return __builtin_bit_cast(float, (hi & 0x80000000u) != 0u ? hi & 0x7fffffffu : ~hi);
}
__device__ __forceinline__ u32 key_column(u64 k) { return ~static_cast<u32>(k); }

template <int V>
struct integral {
    static constexpr int value = V;
};

__device__ __forceinline__ u64 shfl_xor_u64(u64 v, int off) {
    const u32 lo = __shfl_xor(static_cast<u32>(v), off, kWave);
    const u32 hi = __shfl_xor(static_cast<u32>(v >> 32), off, kWave);
    return static_cast<u64>(hi) << 32 | lo;
}

// The wavefront's keys sorted descending across its lanes (bitonic).
__device__ __forceinline__ u64 wave_sort_desc(u64 key) {
    const int l = lane();
#pragma unroll
    for (int k = 2; k <= kWave; k <<= 1) {
#pragma unroll
        for (int j = k / 2; j > 0; j >>= 1) {
            const u64 other = shfl_xor_u64(key, j);
            // Within a run of k, the lower lane of a pair keeps the larger
            // key when the run descends; runs alternate, the last descends.
            const bool lower = (l & j) == 0;
            const bool descending = (l & k) == 0;
            key = lower == descending ? (other > key ? other : key) : (other < key ? other : key);
        }
    }
    return key;
}

// The K best (value, column) pairs of each row, best first, into values
// and indices ([rows, K] each; an unfilled place is (0, 0)). Two passes:
//
//   1. Each thread finds its own best column (the argmax scan). The K-th
//      best of those is a floor: K distinct columns stand at or above it,
//      so no column under it is among the row's K best. And only the K
//      threads whose best reaches it hold any column that does.
//   2. The threads scan again and append every column reaching the floor to
//      `pool` in LDS: at most K threads' worth of columns, which the
//      launcher sizes the dynamic LDS for. Each pooled column's rank among
//      the pool is its place.
//
// For a vocabulary row the pool holds a few dozen columns, so the cost is
// two streaming scans (the second mostly from cache).
template <class T, int K>
__global__ __launch_bounds__(1024) void topk_rows(const T* __restrict__ x, float* __restrict__ values,
                                                  i32* __restrict__ indices, int width,
                                                  const u32* __restrict__ win) {
    static_assert(K <= kWave && (K & (K - 1)) == 0, "a wavefront's sorted keys hold the K best of its threads");
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = blockIdx.x;
    const T* src = x + row * width;
    const int n = blockDim.x, t = threadIdx.x, l = lane(), w = t / kWave;
    constexpr int PER = 16 / sizeof(T);
    const bool vec = chunked(src, width);

    // 1. Every thread's best, then the K-th best of those.
    float best = neg_inf();
    u32 at = kNone;
    if (vec) {
        argmax_scan<T, PER, 4>(src, width / PER, best, at);
    } else {
        argmax_scan<T, 1, 4>(src, width, best, at);
    }
    constexpr int kMaxWaves = 1024 / kWave;
    __shared__ u64 heads[kMaxWaves * K];
    __shared__ u64 floor_key;
    __shared__ int pooled;
    const u64 sorted = wave_sort_desc(at == kNone ? 0ull : topk_key(best, at));
    if (l < K) heads[w * K + l] = sorted;
    if (t == 0) {
        floor_key = 0;
        pooled = 0;
    }
    lds_barrier();
    // A head's rank: its place in its own sorted list plus, per other list,
    // how many beat it (a binary search: K is a power of two).
    const int waves = n / kWave;
    if (t < waves * K) {
        const u64 mine = heads[t];
        int rank = t % K;
#pragma unroll
        for (int v = 0; v < kMaxWaves; ++v) {
            if (v >= waves || v == t / K) continue;
            const u64* list = heads + v * K;
            int pos = 0;
#pragma unroll
            for (int step = K / 2; step > 0; step >>= 1) pos += list[pos + step - 1] > mine ? step : 0;
            rank += pos + (list[pos] > mine ? 1 : 0);
        }
        // Keys are distinct but for the empty 0s; fewer than K columns leave
        // no K-th and the floor at 0, below every column.
        if (rank == K - 1 && mine != 0) floor_key = mine;
    }
    lds_barrier();

    // 2. Pool the columns reaching the floor; a chunk is looked into only
    // when its max could: past the floor's value, or at it from a column
    // not past the floor's (a vocabulary in bf16 ties a lot at the top).
    extern __shared__ u64 pool[];
    // The launcher's bound: K threads' columns, at most ⌈width / n⌉ + 8 each.
    const int capacity = K * ((width + n - 1) / n + 8);
    const u64 floor = floor_key;
    const float floor_v = floor == 0 ? neg_inf() : key_value(floor);
    const u32 floor_c = key_column(floor);
    const auto gather = [&](const auto& c, u32 col, bool held, auto per) {
        constexpr int P = decltype(per)::value;
        const float m = held ? chunk_max(c) : __builtin_nanf("");
        if (__ballot((m > floor_v) | ((m == floor_v) & (col <= floor_c))) == 0) return;
#pragma unroll
        for (int j = 0; j < P; ++j) {
            const float v = c[j];
            const u64 key = topk_key(v, col + j);
            const bool in = held && v == v && key >= floor;
            const u64 mask = __ballot(in);
            if (mask == 0) continue;
            int base = 0;
            if (l == 0) base = atomicAdd(&pooled, __popcll(mask));
            base = __shfl(base, 0, kWave);
            const int at = base + __popcll(mask & ((1ull << l) - 1));
            if (in && at < capacity) pool[at] = key;
        }
    };
    if (vec) {
        scan_chunks<T, PER, 4>(src, width / PER, [&](const Chunk<T, PER>& c, u32 col, bool held) {
            gather(c, col, held, integral<PER>{});
        });
    } else {
        scan_chunks<T, 1, 4>(src, width, [&](const Chunk<T, 1>& c, u32 col, bool held) {
            gather(c, col, held, integral<1>{});
        });
    }
    __syncthreads();

    const int m = min(pooled, capacity);
    for (int p = t; p < m; p += n) {
        const u64 mine = pool[p];
        int rank = 0;
        for (int e = 0; e < m; ++e) rank += pool[e] > mine ? 1 : 0;
        if (rank < K) {
            values[row * K + rank] = key_value(mine);
            indices[row * K + rank] = static_cast<i32>(key_column(mine));
        }
    }
    if (t < K && t >= m) {
        values[row * K + t] = 0.f;
        indices[row * K + t] = 0;
    }
}

}  // namespace pie::layout

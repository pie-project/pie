#pragma once

// Ranks of one process that map one another's memory (peer access on, as
// the tp collectives run): flags at system scope, a timed wait, a barrier
// over signal rows, and 16-byte copies that keep peer traffic out of L1.
// Every rank owns a signal block; a rank writes only into peers' blocks at
// its own column, and reads only its own block.

namespace pie::peer {

constexpr int kMaxRanks = 8;

// A wait whose partner never arrives (the ranks diverged) traps after ten
// minutes, so the fire fails with a device fault instead of spinning for the
// life of the process.
constexpr unsigned long long kPatienceNs = 600ull * 1000000000ull;

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release(unsigned* p, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void add_release(unsigned* p, unsigned v) {
    asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ int ld_acquire_i32(const int* p) {
    int v;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_i32(int* p, int v) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Waits until `*p` has reached `want`: counters only grow, compared modulo
// 2^32 so a long-lived group's wrap is harmless.
__device__ __forceinline__ void wait_reach(const unsigned* p, unsigned want) {
    const unsigned long long start = now_ns();
    while (static_cast<int>(ld_acquire(p) - want) < 0) {
        if (now_ns() - start > kPatienceNs) __trap();
    }
}

// One rank's signal block, as a barrier sees it: `kMaxRanks` arrival
// counters per block slot. Peers add into `arrive[slot][their rank]`;
// `gen[slot]` is this rank's own count of barriers passed at that slot.
struct Signals {
    unsigned arrive[64][kMaxRanks];
    unsigned gen[64];
};

struct Group {
    Signals* signals[kMaxRanks];
};

// Every block `slot` of every rank meets here. Writes made before the
// barrier by this block are visible after it to the same slot on every
// rank (release on arrival, acquire on the wait). Threads `0 .. ranks` do
// the signalling; the block synchronizes on both sides.
__device__ __forceinline__ void barrier(const Group& g, Signals* self, int rank, int ranks,
                                        int slot) {
    __syncthreads();
    const unsigned want = self->gen[slot] + 1;
    if (threadIdx.x < static_cast<unsigned>(ranks)) {
        add_release(&g.signals[threadIdx.x]->arrive[slot][rank], 1u);
        wait_reach(&self->arrive[slot][threadIdx.x], want);
    }
    __syncthreads();
    if (threadIdx.x == 0) self->gen[slot] = want;
}

// 16-byte loads through the non-coherent path, for data no rank writes
// while the kernel runs (a rank's own rows being sent), and stores that do
// not allocate in L1. What a peer wrote is read only after an acquire, by
// plain loads, or in a later launch.
__device__ __forceinline__ int4 ld_nc(const int4* p) {
    int4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void st_na(int4* p, int4 v) {
    asm volatile("st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x),
                 "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// Packs a thread keeps in flight: a copy over the link is bound by the loads
// it waits on, so each thread issues this many before its first store.
constexpr int kInFlight = 4;

// `n` 16-byte packs from `src` (unchanging, as `ld_nc` asks) to each of the
// `ways` rows `dst[0 ..]` by `threads` threads, thread `at` taking the packs
// `at`, `at + threads`, ...: each pack is read once however many rows take it.
__device__ __forceinline__ void fan_copy(int4* const* dst, int ways, const int4* src, int n,
                                         int at, int threads) {
    for (int i = at; i < n; i += threads * kInFlight) {
        int4 v[kInFlight];
#pragma unroll
        for (int u = 0; u < kInFlight; ++u)
            if (i + u * threads < n) v[u] = ld_nc(src + i + u * threads);
        for (int w = 0; w < ways; ++w) {
#pragma unroll
            for (int u = 0; u < kInFlight; ++u)
                if (i + u * threads < n) st_na(dst[w] + i + u * threads, v[u]);
        }
    }
}

}  // namespace pie::peer

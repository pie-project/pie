#pragma once

// Datacenter Blackwell (sm_100a/sm_103a) tensor-core primitives in raw PTX:
// mbarrier, TMA, and tcgen05 over tensor memory (TMEM). A unit including
// this compiles for the arch-specific target only; nothing here has a
// fallback, and nothing here is reached on sm_120, whose block-scaled MMA
// is the warp-level mma.sync.
//
// TMEM is 128 lanes x 512 columns of 32 bits per SM. An address is
// (lane << 16) | column. A warp reaches the 32 lanes of its own quarter,
// warp % 4, and no others.

namespace pie::tc5 {

typedef unsigned int u32;
typedef unsigned long long u64;

// CUtensorMap: 128 opaque bytes the host encodes, passed __grid_constant__.
struct alignas(64) TmaMap {
    u64 raw[16];
};

__device__ __forceinline__ u32 smem_addr(const void* p) {
    return static_cast<u32>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ u32 lane_id() {
    u32 lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

// ---- mbarrier -------------------------------------------------------------

__device__ __forceinline__ void bar_init(u64* bar, u32 count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(count));
}

// Makes initialized barriers visible to the async proxy (TMA, tcgen05).
__device__ __forceinline__ void bar_init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void bar_arrive(u64* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(bar)) : "memory");
}

// One arrival that also tells the phase to wait for `bytes` of transactions.
__device__ __forceinline__ void bar_arrive_expect_tx(u64* bar, u32 bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(bar)),
                 "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ bool bar_try_wait(u32 bar, u32 parity) {
    u32 done;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    return done != 0;
}

// Waits for the phase of parity `parity` to complete.
__device__ __forceinline__ void bar_wait(u64* bar, u32 parity) {
    const u32 at = smem_addr(bar);
    while (!bar_try_wait(at, parity)) {
    }
}

// ---- TMA --------------------------------------------------------------------

__device__ __forceinline__ void tma_prefetch(const TmaMap* map) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<u64>(map)) : "memory");
}

// A 2D box at (x = inner, y = outer) into shared memory, completing `bar`.
__device__ __forceinline__ void tma_load_2d(void* dst, u64* bar, const TmaMap* map, int x, int y) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_addr(dst)),
        "l"(reinterpret_cast<u64>(map)), "r"(smem_addr(bar)), "r"(x), "r"(y)
        : "memory");
}

// The same box landed at the same offset in every CTA of `mask` in the
// cluster, each completing its own barrier at that offset.
__device__ __forceinline__ void tma_load_2d_multicast(void* dst, u64* bar, const TmaMap* map, int x,
                                                      int y, unsigned short mask) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_addr(dst)),
        "l"(reinterpret_cast<u64>(map)), "r"(smem_addr(bar)), "r"(x), "r"(y), "h"(mask)
        : "memory");
}

// A 3D box at (x, y, z), completing `bar`.
__device__ __forceinline__ void tma_load_3d(void* dst, u64* bar, const TmaMap* map, int x, int y,
                                            int z) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(smem_addr(dst)),
        "l"(reinterpret_cast<u64>(map)), "r"(smem_addr(bar)), "r"(x), "r"(y), "r"(z)
        : "memory");
}

// ---- CTA pairs (cta_group::2) -------------------------------------------------
// Two CTAs of a cluster run one MMA of 256 rows: each holds its 128 rows of
// A, half of B's rows and its own half of D. Rank 0 issues; its barriers
// count both CTAs' transactions.

__device__ __forceinline__ u32 cluster_rank() {
    u32 rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::
                     : "memory");
}

// Arrives once on the barrier at `bar`'s offset in CTA `rank` of the cluster.
__device__ __forceinline__ void bar_arrive_at(u64* bar, u32 rank) {
    asm volatile(
        "{\n\t.reg .b32 remote;\n\t"
        "mapa.shared::cluster.u32 remote, %0, %1;\n\t"
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote];\n\t}" ::"r"(smem_addr(bar)),
        "r"(rank)
        : "memory");
}

// The rank-0 twin of a barrier address: TMA issued by either CTA of a pair
// completes its bytes there.
__device__ __forceinline__ u32 leader_bar(u64* bar) { return smem_addr(bar) & 0xFEFFFFFFu; }

__device__ __forceinline__ void tma_load_2d_pair(void* dst, u64* bar, const TmaMap* map, int x,
                                                 int y) {
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_addr(dst)),
        "l"(reinterpret_cast<u64>(map)), "r"(leader_bar(bar)), "r"(x), "r"(y)
        : "memory");
}

__device__ __forceinline__ void tma_load_3d_pair(void* dst, u64* bar, const TmaMap* map, int x,
                                                 int y, int z) {
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(smem_addr(dst)),
        "l"(reinterpret_cast<u64>(map)), "r"(leader_bar(bar)), "r"(x), "r"(y), "r"(z)
        : "memory");
}

// A contiguous run of `bytes` (a multiple of 16) into shared memory.
__device__ __forceinline__ void bulk_load(void* dst, const void* src, u32 bytes, u64* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
            smem_addr(dst)),
        "l"(src), "r"(bytes), "r"(smem_addr(bar))
        : "memory");
}

// A 2D box of shared memory out to (x, y); rows and columns past the
// tensor's extent are dropped. Joins this thread's open bulk group.
__device__ __forceinline__ void tma_store_2d(const TmaMap* map, const void* src, int x, int y) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"(
                     reinterpret_cast<u64>(map)),
                 "r"(smem_addr(src)), "r"(x), "r"(y)
                 : "memory");
}

__device__ __forceinline__ void bulk_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Until at most N of this thread's bulk groups still read shared memory.
template <int N>
__device__ __forceinline__ void bulk_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

// Until all of this thread's bulk groups have landed.
__device__ __forceinline__ void bulk_wait_all() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// An f32 add to global memory that returns nothing.
__device__ __forceinline__ void red_add(float* at, float v) {
    asm volatile("red.global.add.f32 [%0], %1;" ::"l"(at), "f"(v) : "memory");
}

// Four f32 adds to 16 aligned bytes of global memory (sm_90+).
__device__ __forceinline__ void red_add4(float* at, float a, float b, float c, float d) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(at), "f"(a), "f"(b),
                 "f"(c), "f"(d)
                 : "memory");
}

// Makes generic-proxy writes to shared memory visible to TMA.
__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---- tcgen05: allocation and ordering ------------------------------------
// G is the cta_group: 1 for a lone CTA, 2 for a pair, where every op below
// but the fences acts on both CTAs (the allocation is issued by the same
// warp in each).

#define PIE_TC5_GROUP(G, head, tail, ...)              \
    if constexpr ((G) == 1) {                          \
        asm volatile(head "1" tail __VA_ARGS__);       \
    } else {                                           \
        asm volatile(head "2" tail __VA_ARGS__);       \
    }

// Whole-warp: `cols` (a power of two, 32..512) TMEM columns; the base
// address lands in shared memory at `dst`.
template <int G = 1>
__device__ __forceinline__ void tmem_alloc(u32* dst, u32 cols) {
    PIE_TC5_GROUP(G, "tcgen05.alloc.cta_group::", ".sync.aligned.shared::cta.b32 [%0], %1;",
                  ::"r"(smem_addr(dst)), "r"(cols))
}

// Whole-warp: this CTA allocates no more, so a co-resident one may.
template <int G = 1>
__device__ __forceinline__ void tmem_relinquish() {
    PIE_TC5_GROUP(G, "tcgen05.relinquish_alloc_permit.cta_group::", ".sync.aligned;", ::)
}

// Whole-warp, the warp that allocated.
template <int G = 1>
__device__ __forceinline__ void tmem_dealloc(u32 base, u32 cols) {
    PIE_TC5_GROUP(G, "tcgen05.dealloc.cta_group::", ".sync.aligned.b32 %0, %1;", ::"r"(base),
                  "r"(cols))
}

// Orders tcgen05 work around a thread synchronization (barrier wait or
// bar.sync): `before` on the signalling side, `after` on the waiting side.
__device__ __forceinline__ void tc_fence_before() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tc_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// Arrives once on `bar` when every tcgen05 op this thread issued before it
// (mma, cp) has completed. Implies fence::before_thread_sync.
template <int G = 1>
__device__ __forceinline__ void tc_commit(u64* bar) {
    PIE_TC5_GROUP(G, "tcgen05.commit.cta_group::",
                  ".mbarrier::arrive::one.shared::cluster.b64 [%0];", ::"r"(smem_addr(bar))
                  : "memory")
}

// As tc_commit for a pair: one arrival on `bar`'s offset in both CTAs.
__device__ __forceinline__ void tc_commit_pair(u64* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;" ::"r"(smem_addr(bar)),
        "h"(static_cast<unsigned short>(3))
        : "memory");
}

// ---- descriptors ------------------------------------------------------------

// Shared-memory matrix descriptor for a K-major operand in the 128-byte
// swizzle TMA writes: rows of 128 bytes, 8-row atoms 1024 bytes apart. The
// tile must start 1024-aligned; stepping K inside the atom is a plain
// add of (bytes >> 4) to the start address.
__device__ __forceinline__ u64 sdesc_sw128(u32 saddr) {
    u64 d = 0;
    d |= static_cast<u64>((saddr & 0x3FFFFu) >> 4);  // start address, [0,14)
    d |= static_cast<u64>(1) << 16;                   // leading byte offset (unused when swizzled)
    d |= static_cast<u64>(1024 >> 4) << 32;           // stride byte offset: one 8-row atom
    d |= static_cast<u64>(1) << 46;                   // descriptor version 1 (sm_100)
    d |= static_cast<u64>(2) << 61;                   // SWIZZLE_128B
    return d;
}

// Descriptor tcgen05.cp reads a scale atom through: unswizzled rows of 16
// bytes, 8-row core matrices 128 bytes apart.
__device__ __forceinline__ u64 sdesc_sf(u32 saddr) {
    u64 d = 0;
    d |= static_cast<u64>((saddr & 0x3FFFFu) >> 4);
    d |= static_cast<u64>(128 >> 4) << 32;
    d |= static_cast<u64>(1) << 46;
    return d;
}

// Operand formats in the instruction descriptor.
enum : u32 {
    kF16Bf16 = 1,  // kind::f16
    kE4M3 = 0,     // kind::f8f6f4 / mxf8f6f4
    kE5M2 = 1,
    kE2M1 = 5,     // kind::f8f6f4 / mxf8f6f4, unpacked in shared memory
    kNvE2M1 = 1,   // kind::mxf4nvf4, packed
};

// Instruction descriptor, dense kinds (f16, f8f6f4): f32 accumulator,
// both operands K-major.
__host__ __device__ constexpr u32 idesc_dense(u32 fmt_a, u32 fmt_b, u32 m, u32 n) {
    return (1u << 4) | (fmt_a << 7) | (fmt_b << 10) | ((n >> 3) << 17) | ((m >> 4) << 24);
}

// Instruction descriptor, block-scaled kinds: the scale format is ue8m0
// or ue4m3; the scale ids pick the byte of each 32-bit scale column.
__host__ __device__ constexpr u32 idesc_scaled(u32 fmt_a, u32 fmt_b, u32 m, u32 n, bool ue8m0) {
    return (fmt_a << 7) | (fmt_b << 10) | ((n >> 3) << 17) | ((ue8m0 ? 1u : 0u) << 23) |
           ((m >> 4) << 24);
}

__host__ __device__ constexpr u32 idesc_with_ids(u32 desc, u32 sfa_id, u32 sfb_id) {
    return desc | (sfa_id << 29) | (sfb_id << 4);
}

// ---- tcgen05: MMA -----------------------------------------------------------
// D[tmem] (+)= A[smem] . B[smem]^T, one thread issues. `acc` false
// overwrites D.

#define PIE_TC5_MMA_HEAD "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::"

template <int G = 1>
__device__ __forceinline__ void mma_f16(u32 d, u64 a, u64 b, u32 idesc, bool acc) {
    PIE_TC5_GROUP(G, PIE_TC5_MMA_HEAD, ".kind::f16 [%0], %1, %2, %3, p;\n\t}", ::"r"(d), "l"(a),
                  "l"(b), "r"(idesc), "r"(static_cast<u32>(acc)))
}

template <int G = 1>
__device__ __forceinline__ void mma_f8f6f4(u32 d, u64 a, u64 b, u32 idesc, bool acc) {
    PIE_TC5_GROUP(G, PIE_TC5_MMA_HEAD, ".kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}", ::"r"(d),
                  "l"(a), "l"(b), "r"(idesc), "r"(static_cast<u32>(acc)))
}

// One ue8m0 scale per 32 K of each row, read from TMEM at sfa / sfb.
template <int G = 1>
__device__ __forceinline__ void mma_mxf8f6f4(u32 d, u64 a, u64 b, u32 idesc, bool acc, u32 sfa,
                                             u32 sfb) {
    PIE_TC5_GROUP(G, PIE_TC5_MMA_HEAD,
                  ".kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}", ::"r"(d),
                  "l"(a), "l"(b), "r"(idesc), "r"(static_cast<u32>(acc)), "r"(sfa), "r"(sfb))
}

// NVFP4: packed e2m1, one ue4m3 scale per 16 K of each row (four a column).
template <int G = 1>
__device__ __forceinline__ void mma_nvf4(u32 d, u64 a, u64 b, u32 idesc, bool acc, u32 sfa,
                                         u32 sfb) {
    PIE_TC5_GROUP(G, PIE_TC5_MMA_HEAD,
                  ".kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p;\n\t}",
                  ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(static_cast<u32>(acc)), "r"(sfa),
                  "r"(sfb))
}

// ---- tcgen05: data movement -------------------------------------------------

// A 512-byte scale atom (32 rows of 16 bytes) into 4 TMEM columns, copied
// to all four lane quarters: row r of the atom lands in lane r of each.
// A pair copies each CTA's atom at that offset into its own TMEM.
template <int G = 1>
__device__ __forceinline__ void tc_cp_sf_atom(u32 taddr, u64 sdesc) {
    PIE_TC5_GROUP(G, "tcgen05.cp.cta_group::", ".32x128b.warpx4 [%0], %1;", ::"r"(taddr),
                  "l"(sdesc))
}

// Whole-warp: 32 consecutive columns of the warp's 32 lanes, one row a
// thread. Pair with tmem_wait_ld before reading `v`.
__device__ __forceinline__ void tmem_ld32(u32 taddr, u32 (&v)[32]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, "
        "%11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
        "%28, %29, %30, %31}, [%32];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]),
          "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]),
          "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]),
          "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
        : "r"(taddr));
}

__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

}  // namespace pie::tc5

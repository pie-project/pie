#pragma once

// Guest channels: the device half of a program's ring buffers. A fire's
// tickets are checked against the rings' head/tail words (shared with the
// host, so read and written at system scope), host-written cells are pulled
// into device cells, and on commit the rings advance and device-written
// cells are published back.

#include "prelude/device.hip"

namespace pie::channel {

constexpr u32 MAX_RING = 64;

constexpr u32 TICKET_CONSUME = 1u << 0;
constexpr u32 TICKET_PUBLISH = 1u << 1;
constexpr u32 TICKET_HOST_WRITER = 1u << 2;
constexpr u32 TICKET_PACKED_BOOL = 1u << 3;
constexpr u32 TICKET_REQUIRE_INPUT = 1u << 4;
constexpr u32 TICKET_HOST_READER = 1u << 5;
constexpr u32 TICKET_ADVANCE_HEAD = 1u << 6;
constexpr u32 TICKET_ADVANCE_TAIL = 1u << 7;
constexpr u32 TICKET_SHADOW = 1u << 8;

// The layouts below are mirrored field for field by `src/channel.rs`.
struct Ticket {
    u32 slot;
    u32 flags;
    u64 expected_head;
    u64 expected_tail;
    u64* words;
    const u8* mirror;
    u8* cells;
    u32 cap1;
    u32 wire_bytes;
    u32 native_bytes;
};
static_assert(sizeof(Ticket) == 64, "Ticket ABI");

struct PullLane {
    u8* full;
    u32* pass_commit;
    u32 ticket_offset;
    u32 ticket_count;
    u32 initial_commit;
    u32 diagnose;
};
static_assert(sizeof(PullLane) == 32, "PullLane ABI");

struct BumpLane {
    u8* full;
    u32* head;
    u32* tail;
    const u32* cap1;
    const u32* taken;
    u32 taken_count;
    const u32* put;
    u32 put_count;
    const u32* commit;
};
static_assert(sizeof(BumpLane) == 72, "BumpLane ABI");

struct PublishLane {
    const u32* commit;
    u32 ticket_offset;
    u32 ticket_count;
};
static_assert(sizeof(PublishLane) == 16, "PublishLane ABI");

struct SettleLane {
    const u32* commit;
    u32 ticket_offset;
    u32 ticket_count;
};
static_assert(sizeof(SettleLane) == 16, "SettleLane ABI");

__device__ __forceinline__ u64 load_system_acquire(const u64* word) {
    return __hip_atomic_load(word, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void store_system_relaxed(u64* word, u64 value) {
    __hip_atomic_store(word, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Tickets a pull block validates at once; one bit each in `held`.
constexpr u32 PULL_CHUNK = 256;
constexpr u32 PULL_CHUNK_WORDS = PULL_CHUNK / 32;

__device__ __forceinline__ bool ticket_holds(const Ticket& ticket, u64& head, u64& tail) {
    if ((ticket.flags & TICKET_SHADOW) != 0) {
        head = 0;
        tail = 0;
        return true;
    }
    head = load_system_acquire(ticket.words + 0);
    tail = load_system_acquire(ticket.words + 1);
    bool ok = true;
    if ((ticket.flags & TICKET_CONSUME) != 0) ok = head == ticket.expected_head;
    if ((ticket.flags & TICKET_REQUIRE_INPUT) != 0) ok = ok && tail > head;
    if ((ticket.flags & TICKET_PUBLISH) != 0) {
        const u64 same_fire_consume = (ticket.flags & TICKET_CONSUME) != 0 ? 1u : 0u;
        ok = ok && tail == ticket.expected_tail &&
             tail - head < static_cast<u64>(ticket.cap1 - 1) + same_fire_consume;
    }
    return ok;
}

// One cell of `ticket`'s ring, from its host mirror into device cells,
// copied by a team of `width` threads of which this one is `at`.
__device__ __forceinline__ void pull_cell(const Ticket& ticket, u32 ring, u32 at, u32 width) {
    const u8* source = ticket.mirror + static_cast<usize>(ring) * ticket.wire_bytes;
    u8* destination = ticket.cells + static_cast<usize>(ring) * ticket.native_bytes;
    if ((ticket.flags & TICKET_PACKED_BOOL) != 0) {
        for (u32 i = at; i < ticket.native_bytes; i += width) destination[i] = static_cast<u8>((source[i / 8] >> (i % 8)) & 1u);
        return;
    }
    if (ticket.native_bytes % sizeof(b128) == 0 && aligned16(source) && aligned16(destination)) {
        const b128* in = reinterpret_cast<const b128*>(source);
        b128* out = reinterpret_cast<b128*>(destination);
        for (u32 i = at; i < ticket.native_bytes / sizeof(b128); i += width) out[i] = in[i];
        return;
    }
    for (u32 i = at; i < ticket.native_bytes; i += width) destination[i] = source[i];
}

// One block per lane: every ticket checked against its ring (a refusal
// clears the lane's commit word), then each host-written cell it consumes
// pulled, a wavefront per cell.
__global__ void pull_validate(const Ticket* __restrict__ tickets, const PullLane* __restrict__ lanes,
                              u32 lane_count) {
    const u32 lane_index = blockIdx.x;
    if (lane_index >= lane_count) return;
    const PullLane lane = lanes[lane_index];
    __shared__ u32 held[PULL_CHUNK_WORDS];
    __shared__ u32 vetoed;
    if (threadIdx.x == 0) {
        vetoed = 0;
        lane.pass_commit[0] = lane.initial_commit;
        lane.pass_commit[1] = 0;
    }
    const u32 team = blockDim.x < static_cast<u32>(kWave) ? blockDim.x : static_cast<u32>(kWave);
    const u32 teams = blockDim.x / team;
    const u32 mine = threadIdx.x / team;
    const u32 within = threadIdx.x % team;
    const u32 width = blockDim.x < PULL_CHUNK ? blockDim.x : PULL_CHUNK;
    for (u32 base = 0; base < lane.ticket_count; base += width) {
        const u32 span = min(lane.ticket_count - base, width);
        for (u32 word = threadIdx.x; word < PULL_CHUNK_WORDS; word += blockDim.x) held[word] = 0;
        __syncthreads();
        if (threadIdx.x < span) {
            const Ticket ticket = tickets[lane.ticket_offset + base + threadIdx.x];
            u64 head = 0, tail = 0;
            if (ticket_holds(ticket, head, tail)) {
                atomicOr(&held[threadIdx.x / 32], 1u << (threadIdx.x % 32));
            } else {
                if (lane.diagnose != 0) {
                    printf("[kernels-rocm] pull-validate reject: slot=%u flags=0x%x head=%llu tail=%llu "
                           "expected_head=%llu expected_tail=%llu cap1=%u\n",
                           ticket.slot, ticket.flags, static_cast<unsigned long long>(head),
                           static_cast<unsigned long long>(tail),
                           static_cast<unsigned long long>(ticket.expected_head),
                           static_cast<unsigned long long>(ticket.expected_tail), ticket.cap1);
                }
                atomicOr(&vetoed, 1u);
            }
        }
        __syncthreads();
        for (u32 index = mine; index < span; index += teams) {
            const Ticket ticket = tickets[lane.ticket_offset + base + index];
            const bool valid = ((held[index / 32] >> (index % 32)) & 1u) != 0;
            if (!valid || (ticket.flags & TICKET_HOST_WRITER) == 0 || (ticket.flags & TICKET_CONSUME) == 0) continue;
            const u32 ring = static_cast<u32>(ticket.expected_head % ticket.cap1);
            pull_cell(ticket, ring, within, team);
            if (within == 0) lane.full[static_cast<usize>(ticket.slot) * MAX_RING + ring] = 1;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && vetoed != 0) atomicAnd(lane.pass_commit, 0u);
}

// On commit: a cell put fills the tail and advances it, a cell taken
// empties the head and advances it. One thread per lane.
__global__ void commit_bump(const BumpLane* __restrict__ lanes, u32 lane_count) {
    if (blockIdx.x >= lane_count || threadIdx.x != 0) return;
    const BumpLane lane = lanes[blockIdx.x];
    if (!*lane.commit) return;
    for (u32 i = 0; i < lane.put_count; ++i) {
        const u32 slot = lane.put[i];
        lane.full[static_cast<usize>(slot) * MAX_RING + lane.tail[slot]] = 1;
        lane.tail[slot] = (lane.tail[slot] + 1) % lane.cap1[slot];
    }
    for (u32 i = 0; i < lane.taken_count; ++i) {
        const u32 slot = lane.taken[i];
        lane.full[static_cast<usize>(slot) * MAX_RING + lane.head[slot]] = 0;
        lane.head[slot] = (lane.head[slot] + 1) % lane.cap1[slot];
    }
}

// On commit: each device-written cell a host reads, copied (or bit-packed)
// out to its host mirror. One block per lane.
__global__ void scatter_publish(const Ticket* __restrict__ tickets, const PublishLane* __restrict__ lanes,
                                u32 lane_count) {
    if (blockIdx.x >= lane_count) return;
    const PublishLane lane = lanes[blockIdx.x];
    if (lane.commit == nullptr || *lane.commit == 0u) return;
    for (u32 index = 0; index < lane.ticket_count; ++index) {
        const Ticket ticket = tickets[lane.ticket_offset + index];
        const u32 outward = TICKET_PUBLISH | TICKET_HOST_READER;
        if ((ticket.flags & outward) != outward || (ticket.flags & TICKET_SHADOW) != 0) continue;
        if (ticket.mirror == nullptr || ticket.cells == nullptr) continue;
        const u32 ring = static_cast<u32>(ticket.expected_tail % ticket.cap1);
        const u8* source = ticket.cells + static_cast<usize>(ring) * ticket.native_bytes;
        u8* destination = const_cast<u8*>(ticket.mirror) + static_cast<usize>(ring) * ticket.wire_bytes;
        if ((ticket.flags & TICKET_PACKED_BOOL) != 0) {
            for (u32 i = threadIdx.x; i < ticket.wire_bytes; i += blockDim.x) {
                u8 packed = 0;
                for (u32 bit = 0; bit < 8u && i * 8u + bit < ticket.native_bytes; ++bit) {
                    if (source[i * 8u + bit] != 0u) packed |= static_cast<u8>(1u << bit);
                }
                destination[i] = packed;
            }
        } else {
            for (u32 i = threadIdx.x; i < ticket.wire_bytes; i += blockDim.x) destination[i] = source[i];
        }
        __syncthreads();
    }
}

// On commit: the rings' head and tail words advance, where the host reads them.
__global__ void settle(const Ticket* __restrict__ tickets, const SettleLane* __restrict__ lanes,
                       u32 lane_count) {
    if (blockIdx.x >= lane_count) return;
    const SettleLane lane = lanes[blockIdx.x];
    if (lane.commit == nullptr || *lane.commit == 0u) return;
    for (u32 index = threadIdx.x; index < lane.ticket_count; index += blockDim.x) {
        const Ticket ticket = tickets[lane.ticket_offset + index];
        if (ticket.words == nullptr || (ticket.flags & TICKET_SHADOW) != 0) continue;
        if ((ticket.flags & TICKET_ADVANCE_HEAD) != 0) store_system_relaxed(ticket.words + 0, ticket.expected_head + 1);
        if ((ticket.flags & TICKET_ADVANCE_TAIL) != 0) store_system_relaxed(ticket.words + 1, ticket.expected_tail + 1);
    }
}

// Each lane's rows marked with whether its fire committed.
__global__ void mask_from_commit(const u32* const* __restrict__ commits, const i32* __restrict__ indptr,
                                 u8* __restrict__ mask, u32 lane_count) {
    if (blockIdx.x >= lane_count) return;
    const u32* commit = commits[blockIdx.x];
    const u8 byte = (commit != nullptr && *commit != 0u) ? 1 : 0;
    for (i32 row = indptr[blockIdx.x] + static_cast<i32>(threadIdx.x); row < indptr[blockIdx.x + 1];
         row += static_cast<i32>(blockDim.x)) {
        mask[row] = byte;
    }
}

}  // namespace pie::channel

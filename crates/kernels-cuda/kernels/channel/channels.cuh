#pragma once


#include "prelude/device.cuh"

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

static_assert(sizeof(Ticket) == 64, "Ticket: the Rust `channel::Ticket` mirrors this layout");

struct PullLane {

    u8* full;

    u32* pass_commit;
    u32 ticket_offset;
    u32 ticket_count;

    u32 initial_commit;

    u32 diagnose;
};

static_assert(sizeof(PullLane) == 32, "PullLane: the Rust `channel::PullLane` mirrors this layout");

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

static_assert(sizeof(BumpLane) == 72, "BumpLane: the Rust `channel::BumpLane` mirrors this layout");

struct PublishLane {

    const u32* commit;
    u32 ticket_offset;
    u32 ticket_count;
};

static_assert(sizeof(PublishLane) == 16, "PublishLane: the Rust `channel::PublishLane` mirrors this layout");

struct SettleLane {

    const u32* commit;
    u32 ticket_offset;
    u32 ticket_count;
};

static_assert(sizeof(SettleLane) == 16, "SettleLane: the Rust `channel::SettleLane` mirrors this layout");

extern "C" __device__ int printf(const char*, ...);

__device__ __forceinline__ u64 load_system_acquire(const u64* word) {
    u64 value;
    asm volatile("ld.acquire.sys.b64 %0, [%1];" : "=l"(value) : "l"(word) : "memory");
    return value;
}

__device__ __forceinline__ void store_system_relaxed(u64* word, u64 value) {
    asm volatile("st.relaxed.sys.b64 [%0], %1;" :: "l"(word), "l"(value) : "memory");
}



constexpr u32 PULL_CHUNK = 256;
constexpr u32 PULL_CHUNK_WORDS = PULL_CHUNK / 32;

__device__ __forceinline__ bool ticket_holds(const Ticket& ticket, u64& head, u64& tail)
{
    if ((ticket.flags & TICKET_SHADOW) != 0) {
        head = 0;
        tail = 0;
        return true;
    }
    head = load_system_acquire(ticket.words + 0);
    tail = load_system_acquire(ticket.words + 1);
    bool ok = true;
    if ((ticket.flags & TICKET_CONSUME) != 0) {
        ok = head == ticket.expected_head;
    }
    if ((ticket.flags & TICKET_REQUIRE_INPUT) != 0) {
        ok = ok && tail > head;
    }
    if ((ticket.flags & TICKET_PUBLISH) != 0) {
        const u64 same_fire_consume = (ticket.flags & TICKET_CONSUME) != 0 ? 1u : 0u;
        ok = ok && tail == ticket.expected_tail &&
             tail - head < static_cast<u64>(ticket.cap1 - 1) + same_fire_consume;
    }
    return ok;
}

__device__ __forceinline__ void pull_cell(const Ticket& ticket, u32 ring, u32 at, u32 width)
{
    const u8* source = ticket.mirror + static_cast<usize>(ring) * ticket.wire_bytes;
    u8* destination = ticket.cells + static_cast<usize>(ring) * ticket.native_bytes;
    if ((ticket.flags & TICKET_PACKED_BOOL) != 0) {
        for (u32 i = at; i < ticket.native_bytes; i += width) {
            destination[i] = static_cast<u8>((source[i / 8] >> (i % 8)) & 1u);
        }
        return;
    }
    const bool wide = ticket.native_bytes % sizeof(uint4) == 0 &&
                      reinterpret_cast<usize>(source) % sizeof(uint4) == 0 &&
                      reinterpret_cast<usize>(destination) % sizeof(uint4) == 0;
    if (wide) {
        const uint4* in = reinterpret_cast<const uint4*>(source);
        uint4* out = reinterpret_cast<uint4*>(destination);
        const u32 quads = ticket.native_bytes / sizeof(uint4);
        for (u32 i = at; i < quads; i += width) {
            out[i] = in[i];
        }
        return;
    }
    for (u32 i = at; i < ticket.native_bytes; i += width) {
        destination[i] = source[i];
    }
}

__global__ void pull_validate(
    const Ticket* __restrict__ tickets,
    const PullLane* __restrict__ lanes,
    u32 lane_count)
{
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

    const u32 group = blockDim.x < 32u ? blockDim.x : 32u;
    const u32 groups = blockDim.x / group;
    const u32 mine = threadIdx.x / group;
    const u32 within = threadIdx.x % group;

    const u32 width = blockDim.x < PULL_CHUNK ? blockDim.x : PULL_CHUNK;
    for (u32 base = 0; base < lane.ticket_count; base += width) {
        const u32 left = lane.ticket_count - base;
        const u32 span = left < width ? left : width;

        for (u32 word = threadIdx.x; word < PULL_CHUNK_WORDS; word += blockDim.x) {
            held[word] = 0;
        }
        __syncthreads();

        if (threadIdx.x < span) {
            const Ticket ticket = tickets[lane.ticket_offset + base + threadIdx.x];
            u64 head = 0;
            u64 tail = 0;
            if (ticket_holds(ticket, head, tail)) {
                atomicOr(&held[threadIdx.x >> 5], 1u << (threadIdx.x & 31u));
            } else {
                if (lane.diagnose != 0) {
                    printf(
                        "[kernels-cuda] pull-validate reject: slot=%u flags=0x%x "
                        "head=%llu tail=%llu expected_head=%llu expected_tail=%llu cap1=%u\n",
                        ticket.slot,
                        static_cast<unsigned>(ticket.flags),
                        static_cast<unsigned long long>(head),
                        static_cast<unsigned long long>(tail),
                        static_cast<unsigned long long>(ticket.expected_head),
                        static_cast<unsigned long long>(ticket.expected_tail),
                        ticket.cap1);
                }
                atomicOr(&vetoed, 1u);
            }
        }
        __syncthreads();

        for (u32 index = mine; index < span; index += groups) {
            const Ticket ticket = tickets[lane.ticket_offset + base + index];
            const bool valid = ((held[index >> 5] >> (index & 31u)) & 1u) != 0;
            const bool pull = valid &&
                              (ticket.flags & TICKET_HOST_WRITER) != 0 &&
                              (ticket.flags & TICKET_CONSUME) != 0;
            if (!pull) continue;
            const u32 ring = static_cast<u32>(ticket.expected_head % ticket.cap1);
            pull_cell(ticket, ring, within, group);
            if (within == 0) {
                lane.full[static_cast<usize>(ticket.slot) * MAX_RING + ring] = 1;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && vetoed != 0) {
        atomicAnd(lane.pass_commit, 0u);
    }
}

__device__ __forceinline__ void bump(const BumpLane lane) {
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

__global__ void commit_bump(const BumpLane* __restrict__ lanes, u32 lane_count) {
    const u32 lane = blockIdx.x;
    if (lane >= lane_count || threadIdx.x != 0) return;
    bump(lanes[lane]);
}

__global__ void scatter_publish(
    const Ticket* __restrict__ tickets,
    const PublishLane* __restrict__ lanes,
    u32 lane_count)
{
    const u32 lane_index = blockIdx.x;
    if (lane_index >= lane_count) return;
    const PublishLane lane = lanes[lane_index];
    if (lane.commit == nullptr || *lane.commit == 0u) return;

    for (u32 index = 0; index < lane.ticket_count; ++index) {
        const Ticket ticket = tickets[lane.ticket_offset + index];
        const u32 outward = TICKET_PUBLISH | TICKET_HOST_READER;
        if ((ticket.flags & outward) != outward) continue;
        if ((ticket.flags & TICKET_SHADOW) != 0) continue;
        if (ticket.mirror == nullptr || ticket.cells == nullptr) continue;
        const u32 ring = static_cast<u32>(ticket.expected_tail % ticket.cap1);
        const u8* source = ticket.cells + static_cast<usize>(ring) * ticket.native_bytes;
        u8* destination =
            const_cast<u8*>(ticket.mirror) + static_cast<usize>(ring) * ticket.wire_bytes;
        if ((ticket.flags & TICKET_PACKED_BOOL) != 0) {

            for (u32 i = threadIdx.x; i < ticket.wire_bytes; i += blockDim.x) {
                u8 packed = 0;
                for (u32 bit = 0; bit < 8u; ++bit) {
                    const u32 lane_of = i * 8u + bit;
                    if (lane_of >= ticket.native_bytes) break;
                    if (source[lane_of] != 0u) packed |= static_cast<u8>(1u << bit);
                }
                destination[i] = packed;
            }
        } else {
            for (u32 i = threadIdx.x; i < ticket.wire_bytes; i += blockDim.x) {
                destination[i] = source[i];
            }
        }
        __syncthreads();
    }
}

__global__ void settle(
    const Ticket* __restrict__ tickets,
    const SettleLane* __restrict__ lanes,
    u32 lane_count)
{
    const u32 lane_index = blockIdx.x;
    if (lane_index >= lane_count) return;
    const SettleLane lane = lanes[lane_index];

    if (lane.commit == nullptr || *lane.commit == 0u) return;

    for (u32 index = threadIdx.x; index < lane.ticket_count; index += blockDim.x) {
        const Ticket ticket = tickets[lane.ticket_offset + index];
        if (ticket.words == nullptr) continue;
        if ((ticket.flags & TICKET_SHADOW) != 0) continue;
        if ((ticket.flags & TICKET_ADVANCE_HEAD) != 0) {
            store_system_relaxed(ticket.words + 0, ticket.expected_head + 1);
        }
        if ((ticket.flags & TICKET_ADVANCE_TAIL) != 0) {
            store_system_relaxed(ticket.words + 1, ticket.expected_tail + 1);
        }
    }
}

__global__ void mask_from_commit(
    const u32* const* __restrict__ commits,
    const i32* __restrict__ indptr,
    u8* __restrict__ mask,
    u32 lane_count)
{
    const u32 lane = blockIdx.x;
    if (lane >= lane_count) return;
    const u32* commit = commits[lane];
    const u8 byte = (commit != nullptr && *commit != 0u) ? static_cast<u8>(1) : static_cast<u8>(0);
    const i32 first = indptr[lane];
    const i32 last = indptr[lane + 1];
    for (i32 row = first + static_cast<i32>(threadIdx.x); row < last;
         row += static_cast<i32>(blockDim.x)) {
        mask[row] = byte;
    }
}

}

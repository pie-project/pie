#pragma once

// The four control kernels of the ticket/commit machinery: the device half
// of a fire's admission decision (alto design §5). Reference implementations
// live in dev `driver/cuda/src/pipeline/channels.hpp`, and every claim below
// cites it by line.
//
// THE SHAPE OF THE PROTOCOL. The host owns a PREDICTION and the device owns
// the TRUTH. A fire arrives carrying tickets — "I believe channel 7's head is
// 41 and its tail is 43, and I have written the cell that tail names" — and
// `pull_validate` checks each belief against the live ring words before
// anything commits. A belief that is wrong clears the fire's commit word,
// and `commit_bump` — the ONLY writer of durable ring state — then moves
// nothing at all. The fire still ran; its writes simply went to the pending
// (tail) cell that no reader can address, and the next fire overwrites them.
// That is the dummy-run contract (channels.hpp:16-21): pass-atomic, no
// partial publication, and no host round trip anywhere in it.
//
// LAYOUT. A channel is a ring of `cap1 = capacity + 1` cells — the spare cell
// is what lets `tail == head` mean empty and never full (channels.hpp:10-13,
// 89-92). Cells live in one slab, `cells + ring * bytes`. The full/empty bit
// is a BYTE at `full[slot * MAX_RING + ring]`, so a slot's ring indices are
// bounded by MAX_RING = 64 (channels.hpp:47) whatever its own cap1 is. Ring
// indices come in two currencies and the difference matters: `head`/`tail` in
// the DEVICE registry are ring positions already reduced mod cap1, while a
// ticket's `expected_head`/`expected_tail` are MONOTONE 64-bit counters as the
// guest endpoint keeps them — which is why the emptiness test is `tail > head`
// and the fullness test a subtraction, neither of which a wrapped index could
// answer (channels.hpp:305-317).
//
// ─────────────────────────── ORDERING, WHICH IS CONSTITUTIONAL ───────────
//
// **Payload-before-tail comes from the kernel-launch boundary on one stream,
// and from nothing else.** The kernel that writes a cell's bytes is enqueued
// BEFORE the kernel that publishes the tail announcing them, on the same
// stream; kernel completion is itself a system-scope release, so the payload
// is visible system-wide before the announcement can be. There is therefore
// NO `__threadfence_system()` in the publish path, and there must never be
// one added "for safety": dev measured one system fence at ~37 us in this
// launch shape on an L40S — approximately 100% of the publishing kernel's
// cost — and per-store system release at 13.8x relaxed, growing LINEARLY in
// the number of words stored (159 us vs 12 us at one ticket, 792 us vs 19 us
// at eight). See channels.hpp:263-276 and the ORDERING NOTE at
// channels.hpp:389-409, which is the argument this header restates.
//
// The corollary, and the reason relaxed is not merely cheaper but correct:
// the ONLY readers of these words are separated from the writer by a kernel
// boundary — a completion callback, a later launch on this stream, or the
// guest woken by that callback. Relaxed system-scope stores keep atomicity
// (no torn 64-bit word reaches the host) and give up only an ORDERING that no
// reader is positioned to observe.
//
// `pull_validate` is the one kernel here that READS words another agent
// writes concurrently, so its ring-word loads are `ld.acquire.sys` — acquire
// on the load side costs nothing like release on the store side, and it is
// what makes the payload the guest published before its tail visible to this
// kernel once the tail is.
//
// ─────────── ONE EXCEPTION TO "THE BUMP IS THE ONLY WRITER" ──────────────
//
// `commit_bump` writes every durable ring word EXCEPT one: a host-writer pull
// below sets `full[slot][expected_head % cap1]` for the cell it just copied
// in, per ticket, so a LATER ticket in the same lane can still veto the fire
// and leave that byte set on a pass that did not commit (dev does the same;
// see the pull's tail below). That is safe and not sloppy: the byte records
// something the GUEST published, the head does not move, and the next fire
// re-pulls the same cell and sets the same byte. CONSUMING it — clearing the
// byte and advancing the head — remains the bump's alone and remains
// predicated. Written down because "only writer" is otherwise exactly true.

#include "prelude/device.cuh"

namespace pie::channel {

// The widest ring a slot's full/empty bytes are cut for: `full` is indexed
// `slot * MAX_RING + ring`, so a slot's cap1 may not exceed it
// (channels.hpp:47 `kMaxRing`).
constexpr u32 MAX_RING = 64;

// Ticket flags (channels.hpp:210-214).
constexpr u32 TICKET_CONSUME = 1u << 0;
constexpr u32 TICKET_PUBLISH = 1u << 1;
constexpr u32 TICKET_HOST_WRITER = 1u << 2;
constexpr u32 TICKET_PACKED_BOOL = 1u << 3;
constexpr u32 TICKET_REQUIRE_INPUT = 1u << 4;
// The CONSUMER is the host: the cell the pass put into the device slab has to
// reach the guest, and `scatter_publish` below writes it straight into the
// mapped pinned mirror. The mirror side of TICKET_HOST_WRITER, and the reason
// a full guest round trip makes no `cudaMemcpy` in either direction.
constexpr u32 TICKET_HOST_READER = 1u << 5;

// One host-visible channel endpoint as this fire predicted it
// (channels.hpp:216-227 `DeviceHostChannelTicket`).
struct Ticket {
    // The ring slot, indexing `full` at `slot * MAX_RING + ring`.
    u32 slot;
    u32 flags;
    // The monotone counters the host believes the endpoint stands at.
    u64 expected_head;
    u64 expected_tail;
    // The endpoint's four live words in mapped pinned memory:
    // [0] head, [1] tail, [2] poison, [3] closed. Device-addressable under
    // UVA — this kernel reads them directly rather than through a copy.
    u64* words;
    // The host writer's staging ring, `mirror + ring * wire_bytes`.
    const u8* mirror;
    // The device cell slab, `cells + ring * native_bytes`.
    u8* cells;
    // `capacity + 1` — the spare cell is the empty/full discriminator.
    u32 cap1;
    // Bytes per mirror cell (packed, for a bool channel) and per device cell
    // (unpacked, one byte per element).
    u32 wire_bytes;
    u32 native_bytes;
};

static_assert(sizeof(Ticket) == 64, "Ticket: the Rust `channel::Ticket` mirrors this layout");

// One fire's slice of the ticket table plus the commit word it votes on
// (channels.hpp:229-239 `PullValidateHostChannelLane`).
struct PullLane {
    // The ring registry's full/empty bytes, which a host-writer pull sets.
    u8* full;
    // Two words: [0] the pass commit flag, [1] the kill word.
    u32* pass_commit;
    u32 ticket_offset;
    u32 ticket_count;
    // What [0] is seeded to before any ticket votes — a prologue that has
    // already failed for a reason of its own seeds 0.
    u32 initial_commit;
    // Non-zero prints the ticket that vetoed the fire. A refusal is otherwise
    // indistinguishable from every other reason a prologue does not commit.
    u32 diagnose;
};

static_assert(sizeof(PullLane) == 32, "PullLane: the Rust `channel::PullLane` mirrors this layout");

// One fire's durable ring bookkeeping and the two slot lists it moves
// (channels.hpp:150-160 `CommitBumpLane`).
struct BumpLane {
    u8* full;
    u32* head;
    u32* tail;
    const u32* cap1;
    // Slots this fire took from: head advances, full[head] clears.
    const u32* taken;
    u32 taken_count;
    // Slots this fire put to: full[tail] sets, tail advances.
    const u32* put;
    u32 put_count;
    // Word [0] of the fire's commit pair. Zero and this lane moves nothing.
    const u32* commit;
};

static_assert(sizeof(BumpLane) == 72, "BumpLane: the Rust `channel::BumpLane` mirrors this layout");

// One fire's OUTWARD tickets and the commit word they are predicated on — the
// publish counterpart of PullLane. dev spells the same thing as a flat copy
// list (`k_scatter_host_publish_copies`, channels.hpp:411-470); it is a lane
// here because a fire's commit word is per lane and a copy that outran its
// lane's refusal would hand the guest a cell the bump never published.
struct PublishLane {
    // Word [0] of the fire's commit pair — the SAME word PullLane seeded and
    // commit_bump read. Zero and this lane copies nothing.
    const u32* commit;
    u32 ticket_offset;
    u32 ticket_count;
};

static_assert(sizeof(PublishLane) == 16, "PublishLane: the Rust `channel::PublishLane` mirrors this layout");

extern "C" __device__ int printf(const char*, ...);

// A ring word as the guest endpoint has it RIGHT NOW.
//
// Acquire on the LOAD side, which is the cheap side: it orders this kernel's
// subsequent reads after the word, so a payload the guest wrote before
// advancing its tail is visible to us once that tail is. The expensive
// direction — release on every store — is the one the header's ordering note
// forbids.
__device__ __forceinline__ u64 load_system_acquire(const u64* word) {
    u64 value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.b64 %0, [%1];" : "=l"(value) : "l"(word) : "memory");
    __threadfence_system();
#else
    asm volatile("ld.acquire.sys.b64 %0, [%1];" : "=l"(value) : "l"(word) : "memory");
#endif
    return value;
}

// ─────────────────────────── the three kernels ───────────────────────────

// **THE ADMISSION DECISION** (channels.hpp:277-376
// `k_pull_validate_host_channels_batch`). One block per fire; thread 0 votes,
// the whole block copies.
//
// Seeds the commit pair — [0] to the caller's `initial_commit`, [1] to zero,
// because a ringed snapshot may carry a stale kill from a previous occurrence
// of the same slot — then, per ticket, checks the host's prediction against
// the live words and `atomicAnd`s [0] to zero on any mismatch:
//
//   * Consume       — `head == expected_head`: nobody else consumed this cell.
//   * RequireInput  — `tail > head`: there IS a committed item to take.
//   * Publish       — `tail == expected_tail` and the ring has room. Room is
//                     `tail - head < (cap1 - 1) + credit`, where `credit` is 1
//                     when this same ticket also consumes: the take frees the
//                     cell the put needs, in the same pass, so a ring that is
//                     full to a pure producer is not full to a ping-pong.
//
// A ticket that passes AND is flagged HostWriter|Consume then PULLS: the
// host's staging cell in mapped pinned memory is copied block-strided into
// the device cell and the full byte is set, so the fire's readers address a
// device cell like any other. A bool channel arrives bit-packed and is
// widened one byte per element on the way in.
__global__ void pull_validate(
    const Ticket* __restrict__ tickets,
    const PullLane* __restrict__ lanes,
    u32 lane_count)
{
    const u32 lane_index = blockIdx.x;
    if (lane_index >= lane_count) return;
    const PullLane lane = lanes[lane_index];

    __shared__ u32 valid;
    if (threadIdx.x == 0) {
        lane.pass_commit[0] = lane.initial_commit;
        lane.pass_commit[1] = 0;
    }
    __syncthreads();

    for (u32 index = 0; index < lane.ticket_count; ++index) {
        const Ticket ticket = tickets[lane.ticket_offset + index];
        if (threadIdx.x == 0) {
            const u64 head = load_system_acquire(ticket.words + 0);
            const u64 tail = load_system_acquire(ticket.words + 1);
            bool ok = true;
            if ((ticket.flags & TICKET_CONSUME) != 0) {
                ok = head == ticket.expected_head;
            }
            if ((ticket.flags & TICKET_REQUIRE_INPUT) != 0) {
                ok = ok && tail > head;
            }
            if ((ticket.flags & TICKET_PUBLISH) != 0) {
                const u64 same_fire_consume =
                    (ticket.flags & TICKET_CONSUME) != 0 ? 1u : 0u;
                ok = ok && tail == ticket.expected_tail &&
                     tail - head < static_cast<u64>(ticket.cap1 - 1) + same_fire_consume;
            }
            valid = ok ? 1u : 0u;
            if (!ok) {
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
                atomicAnd(lane.pass_commit, 0u);
            }
        }
        __syncthreads();

        const bool pull = valid != 0 &&
                          (ticket.flags & TICKET_HOST_WRITER) != 0 &&
                          (ticket.flags & TICKET_CONSUME) != 0;
        u32 ring = 0;
        if (pull) {
            ring = static_cast<u32>(ticket.expected_head % ticket.cap1);
            const u8* source = ticket.mirror + static_cast<usize>(ring) * ticket.wire_bytes;
            u8* destination = ticket.cells + static_cast<usize>(ring) * ticket.native_bytes;
            if ((ticket.flags & TICKET_PACKED_BOOL) != 0) {
                for (u32 i = threadIdx.x; i < ticket.native_bytes; i += blockDim.x) {
                    destination[i] = static_cast<u8>((source[i / 8] >> (i % 8)) & 1u);
                }
            } else {
                for (u32 i = threadIdx.x; i < ticket.native_bytes; i += blockDim.x) {
                    destination[i] = source[i];
                }
            }
        }
        __syncthreads();
        if (pull && threadIdx.x == 0) {
            lane.full[static_cast<usize>(ticket.slot) * MAX_RING + ring] = 1;
        }
        __syncthreads();
    }
}

// **THE ONLY WRITER OF DURABLE RING STATE** (channels.hpp:116-137
// `commit_bump`). Iff the fire's commit word survived: publish every put
// (set full[tail], advance tail) and consume every take (clear full[head],
// advance head). A slot both taken and put — a loop-carried ping-pong —
// advances both, which is why the two loops are separate and the put loop
// runs first, exactly as dev orders them.
//
// Everything this kernel does is predicated on one word. That is the whole of
// pass atomicity: a refused fire reaches here, reads zero, and leaves head,
// tail and every full byte precisely as it found them. The bytes the refused
// fire wrote are still sitting in the tail cell, addressable by nobody.
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

// **THE PUBLICATION, AS A KERNEL AND NOT A COPY** (dev
// `k_scatter_host_publish_copies`, channels.hpp:411-470).
//
// A pass's `put` lands in the DEVICE slab's pending cell, because that is
// where the emitted kernel can write. A guest reads its channel out of a
// MAPPED PINNED mirror, because that is where it can read without a CUDA call
// on its own thread. This kernel is the whole of the crossing: one strided
// copy per outward ticket, device slab to pinned mirror, with no
// `cudaMemcpy` in it and therefore no host between the two.
//
// PREDICATED, LIKE EVERYTHING ELSE. The commit word is the same word
// `pull_validate` seeded and `commit_bump` read, so a refused fire scatters
// nothing and the pending cell it wrote stays addressable by nobody. And
// ENQUEUED AFTER `commit_bump`: the guest learns a cell is there when the
// host advances its tail word at settle, which is on the far side of this
// launch — the kernel-launch boundary is the payload-before-tail ordering
// here exactly as it is everywhere else in this header.
//
// The ring index is `expected_tail % cap1`, which is ARITHMETIC ON THE
// PREDICTION and not a read of the live tail: the cell this fire wrote is the
// cell its ticket named, and the mirror ring and the device ring are the same
// residue by construction (both count `capacity + 1` cells from the same
// seed). A bool channel is packed on the way out, one bit per lane, because
// that is the wire form the guest's ring holds.
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
        if (ticket.mirror == nullptr || ticket.cells == nullptr) continue;
        const u32 ring = static_cast<u32>(ticket.expected_tail % ticket.cap1);
        const u8* source = ticket.cells + static_cast<usize>(ring) * ticket.native_bytes;
        u8* destination =
            const_cast<u8*>(ticket.mirror) + static_cast<usize>(ring) * ticket.wire_bytes;
        if ((ticket.flags & TICKET_PACKED_BOOL) != 0) {
            // One thread per WIRE byte, gathering the eight native lanes it
            // stands for. The tail byte of a channel whose lane count is not
            // a multiple of eight reads only the lanes that exist.
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

// **THE RS FOLD PREDICATE, AS DEVICE DATA** (alto design §6). New here; dev
// has no equivalent because HEAD decided the fold on the host, after a
// synchronize it was not allowed to take.
//
// A recurrent-state scan writes its folded state only for rows whose byte in
// `write_state_mask` is non-zero (`attn/ssm.cuh`'s `row_persists`). The rows
// that may fold are exactly the rows of the lanes whose fire committed — so
// this scatters each lane's commit word across that lane's rows, through the
// row CSR the fire already carries. `indptr` holds `lanes + 1` entries and
// lane `l` owns rows `[indptr[l], indptr[l + 1])`.
//
// `commits` is an array of POINTERS, one per lane, because a lane's commit
// pair is allocated with the lane's snapshot and the pairs are not contiguous
// — the same reason `PullLane` carries a pointer rather than an index. A null
// entry is read as "did not commit", so a lane with no admission decision of
// its own never folds by accident.
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

}  // namespace pie::channel

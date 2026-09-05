#pragma once

// The five control kernels of the ticket/commit machinery: the device half
// of a fire's admission decision, its publication and its settlement (alto
// design §5). Reference implementations
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
// **THE TWO FLAGS THE SETTLE KERNEL READS, AND THE REASON THEY ARE NOT
// CONSUME/PUBLISH.** dev's settlement predicates its word stores on
// kTicketConsume/kTicketPublish directly (channels.hpp:432-444) because there
// a consuming ticket always consumed. Here CONSUME means "this fire ADDRESSES
// the committed cell" — a `read` that peeks without taking sets it, and so
// does a take whose ring was empty at mint — while what the settlement must
// advance is the endpoint counter the ENGINE owns, and only where the host's
// prediction actually moved. So the mint states the advance separately from
// the address, and the two are set together only in the common case.
//
// They also carry the ownership decision the device cannot see: on a channel
// the host WRITES the guest owns the tail, on one it READS the guest owns the
// head, and the settlement may never store the guest's own counter. The mint
// sets these flags off `Endpoint::engine_owns_head`/`engine_owns_tail`, so a
// word the guest owns simply has no flag naming it.
constexpr u32 TICKET_ADVANCE_HEAD = 1u << 6;
constexpr u32 TICKET_ADVANCE_TAIL = 1u << 7;
// **A FOLLOWER RANK'S TICKET.** Under tensor parallelism every rank runs the
// same pass over the same guest ring, but the ring's words and mirror belong
// to rank 0's device: it alone votes on them, advances them and publishes
// into them. A shadow ticket takes its vote as held (the host gate on the
// same words already admitted the lane, and rank 0's `settle` may have moved
// the word since), still pulls the host writer's cell at the host's predicted
// ring position, and writes NOTHING durable — no word, no mirror.
constexpr u32 TICKET_SHADOW = 1u << 8;

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

// One fire's SETTLEMENT window over the same ticket table — dev
// `HostChannelSettlementLane` (channels.hpp:380-388), less the three fields
// this port does not need. Same three values as `PublishLane` and a distinct
// type on purpose: two kernels reading one array would make a field added to
// either a silent corruption of the other.
//
// WHAT DEV CARRIES HERE AND THIS DOES NOT, and why leaving it out is not a
// gap:
//
//   * `host_commit` — dev's mapped mirror of the commit pair, written so the
//     completion callback can classify a lane without a D2H. Here the commit
//     pair IS mapped pinned memory (`Session::commit`), so the host reads the
//     word the kernels wrote where they wrote it and there is nothing to
//     mirror.
//   * `full`/`head`/`cap1`/`consume` — dev's settlement also clears the
//     consumed cell's full byte and advances the registry head for its
//     "conditional consume" slots. That is `commit_bump`'s `taken` loop here,
//     verbatim and already predicated on the same word, so doing it again
//     would advance every consumed ring TWICE. **The registry is the bump's;
//     the endpoint words are this kernel's.** That line is the whole ownership
//     split and it is why this lane carries no registry pointer at all.
struct SettleLane {
    // Word [0] of the fire's commit pair — the SAME word `pull_validate`
    // seeded, `commit_bump` read and `scatter_publish` read. Zero and this
    // lane advances nothing, which is what makes a refused fire leave the
    // guest's endpoint exactly where it found it.
    const u32* commit;
    u32 ticket_offset;
    u32 ticket_count;
};

static_assert(sizeof(SettleLane) == 16, "SettleLane: the Rust `channel::SettleLane` mirrors this layout");

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

// A ring word as this fire leaves it, stored where the guest will read it.
//
// RELAXED, AND THAT IS THE MEASURED DECISION, NOT AN OVERSIGHT
// (channels.hpp:249-262). System-scope RELEASE compiles to a system release
// fence per store, which on a discrete GPU serialises the write against the
// host interconnect instead of letting the stores pipeline: dev measured 159
// us against 12 us at one ticket and 792 us against 19 us at eight, growing
// linearly in the word count, plus ~37 us for a single explicit
// `__threadfence_system()` at this launch shape. Relaxed keeps the ATOMICITY
// — no torn 64-bit word ever reaches the host — and gives up only an ordering
// between these stores that no reader is positioned to observe, because every
// reader of them is on the far side of this kernel's completion boundary.
__device__ __forceinline__ void store_system_relaxed(u64* word, u64 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    asm volatile("st.volatile.b64 [%0], %1;" :: "l"(word), "l"(value) : "memory");
#else
    asm volatile("st.relaxed.sys.b64 [%0], %1;" :: "l"(word), "l"(value) : "memory");
#endif
}

// ─────────────────────────── the three kernels ───────────────────────────

// ────────────────────────── the admission decision ───────────────────────

// How many of a lane's tickets one pass of the vote covers, and the width of
// the shared bitmap the vote writes into. Never more than the launch's block
// (`channel::PULL_BLOCK`, 256): a fire carries a handful of tickets, so a lane
// takes one pass and the chunking exists only so that a fire with more than a
// block's worth is SLOW rather than wrong.
constexpr u32 PULL_CHUNK = 256;
constexpr u32 PULL_CHUNK_WORDS = PULL_CHUNK / 32;

// **DOES ONE TICKET'S BELIEF SURVIVE CONTACT WITH THE LIVE RING WORDS?**
//
//   * Consume       — `head == expected_head`: nobody else consumed this cell.
//   * RequireInput  — `tail > head`: there IS a committed item to take.
//   * Publish       — `tail == expected_tail` and the ring has room. Room is
//                     `tail - head < (cap1 - 1) + credit`, where `credit` is 1
//                     when this same ticket also consumes: the take frees the
//                     cell the put needs, in the same pass, so a ring that is
//                     full to a pure producer is not full to a ping-pong.
//
// Both loads are PCIe reads out of the guest's mapped pinned memory, and they
// are the entire reason this is a function called by one thread per ticket
// rather than a loop body walked by thread 0 (see the kernel below).
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

// **THE HOST WRITER'S CELL, MIRROR TO SLAB**, strided over ONE GROUP of
// threads — a warp, at the launch's block width — rather than over the whole
// block, so that a fire's several cells cross the aperture at once instead of
// one after another. `at`/`width` are the caller's position in its group; the
// call is group-uniform, never block-uniform.
//
// SIXTEEN BYTES A THREAD WHERE THE CELL ALLOWS IT. The source is mapped pinned
// memory, so each load crosses PCIe; a byte per thread per instruction asks
// the aperture for a quarter of what a `uint4` asks for in the same
// instruction count, and a group of 32 reading `uint4` is still one fully
// coalesced 512-byte request. The guard is the honest one — the widening
// applies only when the cell's width and BOTH addresses are 16-byte aligned,
// which is the common case (a slab base is `cudaMalloc`-aligned and the ring
// stride is the cell width) and never an assumption. A bool channel arrives
// bit-packed and is widened one byte per element on the way in, which has no
// vector form.
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

// **THE ADMISSION DECISION** (channels.hpp:277-376
// `k_pull_validate_host_channels_batch`). One block per fire; ONE THREAD PER
// TICKET votes, and the whole block copies.
//
// Seeds the commit pair — [0] to the caller's `initial_commit`, [1] to zero,
// because a ringed snapshot may carry a stale kill from a previous occurrence
// of the same slot — then checks every ticket's prediction against the live
// words and `atomicAnd`s [0] to zero for each one that is wrong. A ticket that
// passes AND is flagged HostWriter|Consume then PULLS: the host's staging cell
// in mapped pinned memory is copied block-strided into the device cell and the
// full byte is set, so the fire's readers address a device cell like any other.
//
// ───────────────── THE VOTE IS TAKEN IN PARALLEL, AND THAT IS WHERE THIS
//                   KERNEL'S COST LIVES ─────────────────
//
// Every predicate above reads `ticket.words` — the guest's live counters, in
// MAPPED PINNED memory, which means a PCIe read per load and a round trip of
// latency that no arithmetic hides. This kernel used to walk a lane's tickets
// in a `for` loop with thread 0 doing both loads and the rest of the block
// waiting on a `__syncthreads()`, so a fire paid ONE ROUND TRIP PER ENDPOINT
// IT ADDRESSED, in series, in its own prologue. Measured on an L40S at 64
// lanes (`tests/channel_pull_cost.rs`): 6.6 us at one ticket a lane, 29.4 us
// at eight — 3.3 us of pure latency added per ticket. Overlapped, the same
// four points are 6.4 / 6.4 / 6.8 / 7.9 us.
//
// The tickets are INDEPENDENT — each is a claim about a different ring, and
// the vote is an `and` over the answers, which is order-free — so the loads
// belong in flight together. One thread per ticket issues its own pair, the
// answers land in a shared bitmap, and the block waits out one round trip
// instead of `n`. Nothing about the DECISION changes: the same predicates over
// the same words, the same `atomicAnd` per failing ticket, and the same pull
// gated on the same per-ticket validity.
//
// TWO CONSEQUENCES WORTH WRITING DOWN. A ticket's pull now happens after every
// ticket has voted rather than immediately after its own vote — invisible,
// because the pull was never predicated on anything but that ticket's own
// validity (the header's "one exception to the bump is the only writer" says
// exactly this, and still holds word for word). And the `diagnose` printf now
// reports rejects in whatever order the warps finish rather than in ticket
// order; it names the slot in every line, which is what a reader of it needs.
//
// WHAT IS NOT DONE HERE, AND WHY. A fire that PEEKS the same committed cell
// again — a `read` addresses the head without moving it — re-drags mirror
// bytes that are provably identical to the ones already in the slab, and the
// skip would be exact if the kernel could tell "this cell is already pulled"
// from "this cell is full". It cannot: `full[slot][ring]` is also set at BIND
// for a channel declared `from(seed)`, and a host-writer's seed lands in the
// mirror alone (`program::launch::Rings::write_cell` writes no slab for an
// endpoint with a guest end), so a first fire would read a zeroed cell.
// Distinguishing them wants a per-slot "last pulled counter" in the registry,
// which is an ENGINE-side structure — recorded here rather than half-built.
__global__ void pull_validate(
    const Ticket* __restrict__ tickets,
    const PullLane* __restrict__ lanes,
    u32 lane_count)
{
    const u32 lane_index = blockIdx.x;
    if (lane_index >= lane_count) return;
    const PullLane lane = lanes[lane_index];

    // One bit per ticket of the chunk being voted on: set means the ticket's
    // belief held, which is the only thing the pull below is predicated on.
    __shared__ u32 held[PULL_CHUNK_WORDS];
    // Whether ANY ticket of this lane vetoed, across every chunk. The veto is
    // gathered here and spent once, at the end, by the thread that seeded the
    // word — so the commit pair, which is MAPPED PINNED memory in production
    // (the session's own), takes two stores and at most one atomic per fire
    // rather than one atomic per failing ticket across the PCIe aperture.
    // Identical outcome: `atomicAnd(w, 0)` is idempotent and order-free, so
    // "and-ed to zero by k tickets" and "stored zero once because k > 0" are
    // the same word.
    __shared__ u32 vetoed;

    if (threadIdx.x == 0) {
        vetoed = 0;
        lane.pass_commit[0] = lane.initial_commit;
        lane.pass_commit[1] = 0;
    }

    // The block, cut into groups of a warp for the copies below. Written off
    // `blockDim` rather than assuming 256: a group is a warp at every launch
    // this crate makes, and a narrower block degrades to one group rather than
    // to a division by zero.
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

        // **THE PULLS, ONE GROUP PER TICKET** — the same argument as the vote,
        // one level down. A cell copy is a stream of PCIe reads; two cells
        // read by two groups overlap, while two cells read one after another
        // by the whole block do not. At the launch's 256-thread block a group
        // is a warp and eight cells cross at once. Aggregate work per thread
        // is unchanged, so a wide cell loses nothing: 32 threads reading
        // `uint4` is one fully coalesced request either way.
        //
        // NOTHING IN HERE NEEDS A BARRIER. The full byte and the cell bytes
        // are read by LATER KERNELS on this stream, never by this one, and
        // kernel completion is the release that orders them — the header's
        // ordering note, applied to the pull's own two writes.
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

    // ATOMIC AND NOT A STORE, for the one reason dev's was: the word is the
    // fire's, and nothing here may assume it is only ever this block's.
    if (threadIdx.x == 0 && vetoed != 0) {
        atomicAnd(lane.pass_commit, 0u);
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
        if ((ticket.flags & TICKET_SHADOW) != 0) continue;
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

// **THE ENDPOINT'S COUNTERS, ADVANCED BY THE DEVICE** — dev
// `k_settle_host_channels_batch` (channels.hpp:411-455), and the kernel whose
// absence made every frame boundary take a `cudaStreamSynchronize`.
//
// WHAT THE WAIT WAS FOR. These four words are the guest's view of the ring,
// and until this kernel existed the HOST advanced them: synchronize, read the
// pinned commit word, then `bump_head`/`bump_tail` per slot. The next fire's
// mint predicts off exactly those counters, so the wait was not there to make
// the answer correct — it was there because the answer did not exist until a
// host thread wrote it. Written by the device, in stream order, it exists
// when the next kernel that reads it runs, and no host thread is between the
// two. ~826 waits a c64 run, all of them this.
//
// WHAT IT ADVANCES, AND WHAT IT MUST NOT. Exactly the two words a ticket
// names with TICKET_ADVANCE_HEAD / TICKET_ADVANCE_TAIL, to `expected_head + 1`
// / `expected_tail + 1` — the PREDICTION plus one, never a read-modify-write,
// because the prediction is what `pull_validate` already proved the word
// equals and an increment would race the guest's own counter on the other
// end. The registry (`full`, `head`, `tail`) belongs to `commit_bump` and is
// not touched here; the guest's own counter has no flag naming it and is not
// touched here either. See `SettleLane` for the full ownership split.
//
// ORDERING. Enqueued AFTER `scatter_publish` on the same stream, and that
// launch boundary is the ENTIRE payload-before-tail argument: the cells
// `scatter_publish` wrote into the guest's mirror are visible system-wide at
// its completion, which strictly precedes the first store this kernel makes.
// So the tail that announces a cell can never reach the guest ahead of the
// cell. Every store below is therefore relaxed and there is no fence — see
// `store_system_relaxed` for the 13.8x this buys, and the header's ordering
// note for why relaxed gives up nothing any reader can observe.
__global__ void settle(
    const Ticket* __restrict__ tickets,
    const SettleLane* __restrict__ lanes,
    u32 lane_count)
{
    const u32 lane_index = blockIdx.x;
    if (lane_index >= lane_count) return;
    const SettleLane lane = lanes[lane_index];
    // **THE ONE PREDICATE, AS EVERYWHERE ELSE.** A refused fire settles
    // nothing: the guest's endpoint stands exactly where it stood, the bytes
    // the pass wrote are in a cell no counter addresses, and the next fire
    // predicts the same numbers and is admitted.
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

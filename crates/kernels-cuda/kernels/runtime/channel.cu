#include "prim/device.cuh"

namespace pie::runtime::channel {

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
            const bool wide = ticket.wire_bytes % sizeof(uint4) == 0 &&
                              reinterpret_cast<usize>(source) % sizeof(uint4) == 0 &&
                              reinterpret_cast<usize>(destination) % sizeof(uint4) == 0;
            if (wide) {
                const uint4* in = reinterpret_cast<const uint4*>(source);
                uint4* out = reinterpret_cast<uint4*>(destination);
                const u32 quads = ticket.wire_bytes / sizeof(uint4);
                for (u32 i = threadIdx.x; i < quads; i += blockDim.x) {
                    out[i] = in[i];
                }
            } else {
                for (u32 i = threadIdx.x; i < ticket.wire_bytes; i += blockDim.x) {
                    destination[i] = source[i];
                }
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


// Packs `bits` bool bytes at `src` (a channel's native cell) into the
// little-endian bits at `dst` (a lane's row of the staged attention mask),
// a byte a thread: the mask the device carries for a fire staged before
// the fire that computes it has run.
__global__ void gather_bits(const u8* __restrict__ src, u8* __restrict__ dst, u32 bits)
{
    const u32 bytes = (bits + 7u) / 8u;
    for (u32 byte = blockIdx.x * blockDim.x + threadIdx.x; byte < bytes;
         byte += gridDim.x * blockDim.x) {
        u8 packed = 0;
        for (u32 bit = 0; bit < 8u; ++bit) {
            const u32 at = byte * 8u + bit;
            if (at >= bits) break;
            if (src[at] != 0u) packed |= static_cast<u8>(1u << bit);
        }
        dst[byte] = packed;
    }
}

struct BitRun {
    const u8* src;
    u32 dst_byte;
    u32 bits;
};

static_assert(sizeof(BitRun) == 16, "BitRun: the Rust `channel::BitRun` mirrors this layout");

// Packs each of `run_count` runs, a block a run: run r's `bits` bool bytes
// at its `src` (a lane's row of a channel cell) into the little-endian bits
// `dst_byte` bytes past `dst` (that lane's row of the staged attention mask).
__global__ void gather_bit_runs(
    const BitRun* __restrict__ runs,
    u8* __restrict__ dst,
    u32 run_count)
{
    const u32 run = blockIdx.x;
    if (run >= run_count) return;
    const BitRun r = runs[run];
    const u32 bytes = (r.bits + 7u) / 8u;
    for (u32 byte = threadIdx.x; byte < bytes; byte += blockDim.x) {
        u8 packed = 0;
        for (u32 bit = 0; bit < 8u; ++bit) {
            const u32 at = byte * 8u + bit;
            if (at >= r.bits) break;
            if (r.src[at] != 0u) packed |= static_cast<u8>(1u << bit);
        }
        dst[r.dst_byte + byte] = packed;
    }
}

struct WordRun {
    const u32* src;
    u32 dst_word;
    u32 words;
};

static_assert(sizeof(WordRun) == 16, "WordRun: the Rust `channel::WordRun` mirrors this layout");

__global__ void gather_words(
    const WordRun* __restrict__ runs,
    u32* __restrict__ dst,
    u32 run_count)
{
    const u32 run = blockIdx.x;
    if (run >= run_count) return;
    const WordRun r = runs[run];
    u32* out = dst + r.dst_word;
    if ((r.words & 3u) == 0u &&
        (reinterpret_cast<usize>(r.src) & 15u) == 0u &&
        (reinterpret_cast<usize>(out) & 15u) == 0u) {
        const uint4* in4 = reinterpret_cast<const uint4*>(r.src);
        uint4* out4 = reinterpret_cast<uint4*>(out);
        const u32 quads = r.words >> 2;
        for (u32 q = threadIdx.x; q < quads; q += blockDim.x) {
            out4[q] = in4[q];
        }
        return;
    }
    for (u32 word = threadIdx.x; word < r.words; word += blockDim.x) {
        out[word] = r.src[word];
    }
}

// The buffered recurrent rows of a fire, laid as the staged rs tables state
// them (engine-cuda `inputs`, the rs section): a lane each, where its rows
// land in its rs buffer pages and which of those pages' rows it replays
// ahead of its own. Every address a kernel below writes or reads through
// the tables is data, so one recorded body serves any slot assignment.
struct RsLane {
    // The buffer token the lane's first row lands at, RS_NONE: it lands none.
    u32 at;
    // The buffered rows it replays ahead of its own.
    u32 replay;
    // Where its replayed tokens start in the replay table.
    u32 replay_off;
    // Where its pages' slot addresses start in the page table.
    u32 page_off;
};

static_assert(sizeof(RsLane) == 16, "RsLane: the Rust `channel::RsLane` mirrors this layout");

constexpr u32 RS_NONE = 0xffffffffu;

// One row of `row_bytes` (a multiple of 2) from `src` to `dst`, the block's
// threads striding it in words where both ends allow.
__device__ __forceinline__ void rs_copy_row(u8* dst, const u8* src, u32 row_bytes)
{
    if ((row_bytes & 3u) == 0u &&
        (reinterpret_cast<usize>(src) & 3u) == 0u &&
        (reinterpret_cast<usize>(dst) & 3u) == 0u) {
        const u32* s = reinterpret_cast<const u32*>(src);
        u32* d = reinterpret_cast<u32*>(dst);
        for (u32 w = threadIdx.x; w < (row_bytes >> 2); w += blockDim.x) d[w] = s[w];
        return;
    }
    const unsigned short* s = reinterpret_cast<const unsigned short*>(src);
    unsigned short* d = reinterpret_cast<unsigned short*>(dst);
    for (u32 h = threadIdx.x; h < (row_bytes >> 1); h += blockDim.x) d[h] = s[h];
}

// The page row token `tok` of plane (`plane_off`, `plane_stride`) lies at.
__device__ __forceinline__ u8* rs_page_row(
    const u64* pages, RsLane lane, u32 tok, u32 page_tokens, u64 plane_off, u64 plane_stride)
{
    const u64 slot = pages[lane.page_off + tok / page_tokens];
    return reinterpret_cast<u8*>(slot + plane_off + static_cast<u64>(tok % page_tokens) * plane_stride);
}

// Lands each lane's rows of `src` (a block a lane) at its buffer tokens:
// row e of lane r at token `at + e` of its pages. `win` (the window's live
// words, or null) offsets rows and lanes as the recurrent kernels do.
__global__ void rs_scatter(
    const u8* __restrict__ src,
    u32 row_bytes,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ win,
    u32 lanes_stated,
    const RsLane* __restrict__ lanes_tab,
    const u64* __restrict__ pages,
    u32 page_tokens,
    u64 plane_off,
    u64 plane_stride)
{
    const u32 lanes = win != nullptr ? win[2] : lanes_stated;
    const u32 r = blockIdx.x;
    if (r >= lanes) return;
    const u32 row0 = win != nullptr ? win[1] : 0u;
    const u32 rl = win != nullptr ? r + win[3] : r;
    const RsLane lane = lanes_tab[rl];
    if (lane.at == RS_NONE) return;
    const u32 t0 = qo_indptr[r];
    const u32 n = qo_indptr[r + 1] - t0;
    for (u32 e = 0; e < n; ++e) {
        u8* dst = rs_page_row(pages, lane, lane.at + e, page_tokens, plane_off, plane_stride);
        rs_copy_row(dst, src + static_cast<u64>(row0 + t0 + e) * row_bytes, row_bytes);
    }
}

// The window's extended layout: lane r's replayed rows then its own from
// `csr[r]`, padded lanes (to `lanes_cap`) empty; `ext_win` the live words
// the recurrent kernels read the extended rows by (no row offset: the
// extended rows are the window's own).
__global__ void rs_ext_csr(
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ win,
    u32 lanes_stated,
    u32 lanes_cap,
    const RsLane* __restrict__ lanes_tab,
    u32* __restrict__ csr,
    u32* __restrict__ ext_win)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const u32 lanes = win != nullptr ? win[2] : lanes_stated;
    const u32 lane0 = win != nullptr ? win[3] : 0u;
    u32 acc = 0;
    for (u32 r = 0; r < lanes; ++r) {
        csr[r] = acc;
        acc += lanes_tab[lane0 + r].replay + (qo_indptr[r + 1] - qo_indptr[r]);
    }
    for (u32 r = lanes; r <= lanes_cap; ++r) csr[r] = acc;
    ext_win[0] = acc;
    ext_win[1] = 0u;
    ext_win[2] = lanes;
    ext_win[3] = lane0;
}

// Lane r's extended rows (a block a lane): its replayed buffer rows (when
// the value is a buffered plane, `pages` non-null) then its own rows of
// `own`, which land at its buffer tokens as `rs_scatter` lands them.
__global__ void rs_ext_gather(
    const u8* __restrict__ own,
    u8* __restrict__ ext,
    u32 row_bytes,
    const u32* __restrict__ qo_indptr,
    u32* __restrict__ csr,
    u32* __restrict__ ext_win,
    const u32* __restrict__ win,
    u32 lanes_stated,
    const RsLane* __restrict__ lanes_tab,
    const u64* __restrict__ pages,
    const u32* __restrict__ replay_tok,
    u32 page_tokens,
    u64 plane_off,
    u64 plane_stride)
{
    const u32 lanes = win != nullptr ? win[2] : lanes_stated;
    const u32 lane0 = win != nullptr ? win[3] : 0u;
    const u32 r = blockIdx.x;
    // Lane r's extended rows start past every earlier lane's replayed and
    // own rows; the first gather of an op (`ext_win` non-null) lays the
    // window's layout from block 0, padded lanes empty.
    __shared__ u32 begin;
    if (threadIdx.x == 0) {
        const u32 upto = r < lanes ? r : lanes;
        u32 acc = 0;
        for (u32 i = 0; i < upto; ++i) {
            acc += lanes_tab[lane0 + i].replay + (qo_indptr[i + 1] - qo_indptr[i]);
        }
        begin = acc;
        if (ext_win != nullptr && r == 0) {
            u32 total = 0;
            for (u32 i = 0; i < lanes; ++i) {
                csr[i] = total;
                total += lanes_tab[lane0 + i].replay + (qo_indptr[i + 1] - qo_indptr[i]);
            }
            for (u32 i = lanes; i <= lanes_stated; ++i) csr[i] = total;
            ext_win[0] = total;
            ext_win[1] = 0u;
            ext_win[2] = lanes;
            ext_win[3] = lane0;
        }
    }
    __syncthreads();
    if (r >= lanes) return;
    const u32 row0 = win != nullptr ? win[1] : 0u;
    const RsLane lane = lanes_tab[lane0 + r];
    const u32 e0 = begin;
    if (pages != nullptr) {
        for (u32 k = 0; k < lane.replay; ++k) {
            const u32 tok = replay_tok[lane.replay_off + k];
            const u8* src = rs_page_row(pages, lane, tok, page_tokens, plane_off, plane_stride);
            rs_copy_row(ext + static_cast<u64>(e0 + k) * row_bytes, src, row_bytes);
        }
    }
    const u32 t0 = qo_indptr[r];
    const u32 n = qo_indptr[r + 1] - t0;
    for (u32 e = 0; e < n; ++e) {
        const u8* src = own + static_cast<u64>(row0 + t0 + e) * row_bytes;
        rs_copy_row(ext + static_cast<u64>(e0 + lane.replay + e) * row_bytes, src, row_bytes);
        // A buffered plane's own rows land in the lane's buffer too.
        if (pages != nullptr && lane.at != RS_NONE) {
            rs_copy_row(rs_page_row(pages, lane, lane.at + e, page_tokens, plane_off, plane_stride),
                        src, row_bytes);
        }
    }
}

// Lane r's own rows of the extended `ext` back onto `target` (a block a
// lane), past the rows it replayed.
__global__ void rs_ext_land(
    const u8* __restrict__ ext,
    u8* __restrict__ target,
    u32 row_bytes,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ csr,
    const u32* __restrict__ win,
    u32 lanes_stated,
    const RsLane* __restrict__ lanes_tab)
{
    const u32 lanes = win != nullptr ? win[2] : lanes_stated;
    const u32 r = blockIdx.x;
    if (r >= lanes) return;
    const u32 row0 = win != nullptr ? win[1] : 0u;
    const u32 rl = win != nullptr ? r + win[3] : r;
    const u32 skip = csr[r] + lanes_tab[rl].replay;
    const u32 t0 = qo_indptr[r];
    const u32 n = qo_indptr[r + 1] - t0;
    for (u32 e = 0; e < n; ++e) {
        rs_copy_row(target + static_cast<u64>(row0 + t0 + e) * row_bytes,
                    ext + static_cast<u64>(skip + e) * row_bytes, row_bytes);
    }
}

}

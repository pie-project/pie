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
    // Where the host reads the pass's verdict: `commit` lives in the wave's
    // device tables, read by every region; the settle hands it over.
    u32* commit_host;
};

static_assert(sizeof(SettleLane) == 24, "SettleLane: the Rust `channel::SettleLane` mirrors this layout");

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

    if (threadIdx.x == 0 && lane.commit_host != nullptr && lane.commit != nullptr) {
        *reinterpret_cast<volatile u32*>(lane.commit_host) = *lane.commit;
    }
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
    u64 dst_byte;
    u32 bits;
    u32 pad;
};

static_assert(sizeof(BitRun) == 24, "BitRun: the Rust `channel::BitRun` mirrors this layout");

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
    // Where its extended rows start in the fire's extended run: past every
    // earlier lane's replayed and own rows.
    u32 ext;
};

static_assert(sizeof(RsLane) == 20, "RsLane: the Rust `channel::RsLane` mirrors this layout");

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
// the recurrent kernels read the extended rows by, from the window's place
// in the fire's extended run (its first lane's `ext`).
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
    ext_win[1] = lanes != 0u ? lanes_tab[lane0].ext : 0u;
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
    // own rows, from the window's place in the fire's extended run; the
    // first gather of an op (`ext_win` non-null) lays the window's layout
    // from block 0, padded lanes empty.
    __shared__ u32 begin;
    if (threadIdx.x == 0) {
        const u32 upto = r < lanes ? r : lanes;
        const u32 base = lanes != 0u ? lanes_tab[lane0].ext : 0u;
        u32 acc = base;
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
            ext_win[1] = base;
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
// lane), past the rows it replayed, from the window's place in the fire's
// extended run.
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
    const u32 lane0 = win != nullptr ? win[3] : 0u;
    const u32 skip = lanes_tab[lane0].ext + csr[r] + lanes_tab[lane0 + r].replay;
    const u32 t0 = qo_indptr[r];
    const u32 n = qo_indptr[r + 1] - t0;
    for (u32 e = 0; e < n; ++e) {
        rs_copy_row(target + static_cast<u64>(row0 + t0 + e) * row_bytes,
                    ext + static_cast<u64>(skip + e) * row_bytes, row_bytes);
    }
}

// The fault a lane's page list states (its `fault` word): an entry past its
// working set's reserve, a list shorter than the pages the lane reads, a
// row written below the set's frozen slots, a copy out of its bounds.
#define PIE_PAGES_PAST_RESERVE 1u
#define PIE_PAGES_SHORT 2u
#define PIE_PAGES_BELOW_FLOOR 3u
#define PIE_PAGES_COPY_BOUNDS 4u

// A lane's page copy in one page list (the copy `beam_step` left in its
// group's state): `[src, dst, tokens]` at `copy`, reserve pages `f0 + src` and `f0 + dst`
// at `base[src]` and `base[dst]`, its planes `planes` of the plane table.
struct PageCopyRun {
    const u32* copy;
    const u64* base;
    u32* fault;
    u32 reserve;
    u32 page;
    u32 floor_rel;
    u32 plane_at;
    u32 planes;
    u32 pad;
};

static_assert(sizeof(PageCopyRun) == 48, "PageCopyRun: the Rust `channel::PageCopyRun` mirrors this layout");

// Where a plane lies in a page and the bytes a token holds in it.
struct PagePlane {
    u64 offset;
    u64 cell;
};

static_assert(sizeof(PagePlane) == 16, "PagePlane: the Rust `channel::PagePlane` mirrors this layout");

// Copies each run's first `tokens` slots of page `src` to page `dst`, a
// block a (run, plane): ahead of the fire's appends and reads, in stream
// order. A copy out of its bounds (a page past the reserve, more tokens
// than a page, a destination slot below the frozen ones) copies nothing
// and states the lane's fault.
__global__ void copy_pages(
    const PageCopyRun* __restrict__ runs,
    const PagePlane* __restrict__ planes,
    u32 run_count)
{
    const u32 run = blockIdx.x;
    if (run >= run_count) return;
    const PageCopyRun r = runs[run];
    if (blockIdx.y >= r.planes) return;
    const u32 src = r.copy[0];
    const u32 dst = r.copy[1];
    const u32 tokens = r.copy[2];
    if (tokens == 0u || src == dst) return;
    if (src >= r.reserve || dst >= r.reserve || tokens > r.page
        || static_cast<u64>(dst) * r.page < r.floor_rel) {
        if (blockIdx.y == 0 && threadIdx.x == 0) atomicMax(r.fault, PIE_PAGES_COPY_BOUNDS);
        return;
    }
    const PagePlane plane = planes[r.plane_at + blockIdx.y];
    const u64 bytes = static_cast<u64>(tokens) * plane.cell;
    const u8* from = reinterpret_cast<const u8*>(r.base[src] + plane.offset);
    u8* to = reinterpret_cast<u8*>(r.base[dst] + plane.offset);
    if ((bytes & 15u) == 0u && ((r.base[src] + plane.offset) & 15u) == 0u
        && ((r.base[dst] + plane.offset) & 15u) == 0u) {
        const uint4* f = reinterpret_cast<const uint4*>(from);
        uint4* t = reinterpret_cast<uint4*>(to);
        for (u64 at = threadIdx.x; at < bytes / 16u; at += blockDim.x) t[at] = f[at];
    } else {
        for (u64 at = threadIdx.x; at < bytes; at += blockDim.x) to[at] = from[at];
    }
}

// Beam page lists (new-memory/designs/BEAM-PAGE-LISTS.md): one group's step.
// The group's resident state, u32 words: a header (primed, fires run, the
// pool's pages at the last fire), its
// lists twice (the fire's and the one before, by the fire count's parity),
// its free pages, a ring of `frames` pending pages (a page freed by fire t
// joins the free ones at fire t + frames + 1), and this fire's copies.
#define PIE_BEAM_HEADER 4u
#define PIE_PAGES_PARENT 5u
#define PIE_PAGES_EXHAUSTED 6u

// A lane's page table to resolve: its staged indices from entry `f0`, its
// rows' staged write pages and offsets.
struct BeamListRun {
    i32* indices;
    i32* write_page;
    const i32* write_offset;
    u32 lane;
    u32 entries;
    u32 rows;
    u32 pad;
};

static_assert(sizeof(BeamListRun) == 40, "BeamListRun: the Rust `channel::BeamListRun` mirrors this layout");

// A lane's rs page table to resolve (§11, a hybrid's): its `entries`
// entries from `from`, entry e holding the `tokens` buffered rows of slots
// `base + e * tokens` on, each resolved to the set's entry holding the same
// rows of its beam's list's page (`bases`: the set's entries' slot
// addresses, `bases_len` of them, from its entry 0).
struct BeamRsRun {
    u64* pages;
    const u64* bases;
    u32 lane;
    u32 from;
    u32 entries;
    u32 base;
    u32 tokens;
    u32 bases_len;
};

static_assert(sizeof(BeamRsRun) == 40, "BeamRsRun: the Rust `channel::BeamRsRun` mirrors this layout");

// A beam's rs copy (§6.4 on a hybrid): the first `tokens` slots of pool page
// `src` to pool page `dst` (the `[src, dst, tokens]` its group's step left
// at `copy`), row by row of each plane, through the set's rs entries
// (`bases`, as `BeamRsRun`'s): a slot below the buffer (the prompt's tail,
// in the state) has no row and is skipped.
struct RsCopyRun {
    const u32* copy;
    const u64* bases;
    u32* fault;
    u32 reserve;
    u32 page;
    u32 tokens;
    u32 base;
    u32 f0;
    u32 floor_rel;
    u32 plane_at;
    u32 planes;
    u32 bases_len;
    u32 pad;
};

static_assert(sizeof(RsCopyRun) == 64, "RsCopyRun: the Rust `channel::RsCopyRun` mirrors this layout");

// Copies each run's rows, a block a (run, plane), a slot's row a step.
__global__ void copy_rs_rows(
    const RsCopyRun* __restrict__ runs,
    const PagePlane* __restrict__ planes,
    u32 run_count)
{
    const u32 run = blockIdx.x;
    if (run >= run_count) return;
    const RsCopyRun r = runs[run];
    if (blockIdx.y >= r.planes) return;
    const u32 src = r.copy[0];
    const u32 dst = r.copy[1];
    const u32 tokens = r.copy[2];
    if (tokens == 0u || src == dst) return;
    if (src >= r.reserve || dst >= r.reserve || tokens > r.page
        || static_cast<u64>(dst) * r.page < r.floor_rel) {
        if (blockIdx.y == 0 && threadIdx.x == 0) atomicMax(r.fault, PIE_PAGES_COPY_BOUNDS);
        return;
    }
    const PagePlane plane = planes[r.plane_at + blockIdx.y];
    const u64 T = r.tokens == 0u ? 1u : r.tokens;
    for (u32 k = 0u; k < tokens; ++k) {
        const u64 from_slot = (static_cast<u64>(r.f0) + src) * r.page + k;
        const u64 to_slot = (static_cast<u64>(r.f0) + dst) * r.page + k;
        if (from_slot < r.base || to_slot < r.base) continue;
        const u64 fe = (from_slot - r.base) / T, te = (to_slot - r.base) / T;
        if (fe >= r.bases_len || te >= r.bases_len) {
            if (threadIdx.x == 0) atomicMax(r.fault, PIE_PAGES_COPY_BOUNDS);
            return;
        }
        const u8* from = reinterpret_cast<const u8*>(
            r.bases[fe] + plane.offset + ((from_slot - r.base) % T) * plane.cell);
        u8* to = reinterpret_cast<u8*>(r.bases[te] + plane.offset + ((to_slot - r.base) % T) * plane.cell);
        for (u64 at = threadIdx.x; at < plane.cell; at += blockDim.x) to[at] = from[at];
    }
}

// A group's step: its state, the parents its last epilogue wrote, its pool's
// device ids, its lanes' list runs and fault words, the numbers of §3, and
// on a hybrid its lanes' rs page tables (§11).
struct BeamStepRun {
    u32* state;
    const u32* parents;
    const i32* trans;
    const BeamListRun* lists;
    u32* faults;
    u32 lanes;
    u32 cap;
    u32 reserve;
    u32 page;
    u32 folded;
    u32 slot;
    u32 list_count;
    u32 frames;
    u32 backed;
    u32 rs_count;
    const BeamRsRun* rs;
};

static_assert(sizeof(BeamStepRun) == 88, "BeamStepRun: the Rust `channel::BeamStepRun` mirrors this layout");

#define PIE_BEAM_MAX_LANES 64u

// One block a group (blockIdx.x). §5 on the group's first fire, §6.1-6.3
// after, then §6.5 over its lanes' staged tables; leaves this fire's copies
// (§6.4) in its state for `copy_pages`.
__global__ void beam_step(const BeamStepRun* __restrict__ runs, u32 run_count)
{
    if (blockIdx.x >= run_count) return;
    const BeamStepRun r = runs[blockIdx.x];
    const u32 B = r.lanes, cap = r.cap, Q = r.reserve, S = r.page;
    const u32 F = r.frames == 0u ? 1u : r.frames;
    // The pool pages the fire stages (`reserve` is the pool's whole): the
    // lowest-free rule keeps every page the step names below them.
    const u32 K = r.backed < Q ? r.backed : Q;
    if (B == 0u || B > PIE_BEAM_MAX_LANES || Q > cap) return;
    u32* header = r.state;
    const u32 primed = header[0];
    const u32 fired = header[1];
    const u32 parity = fired & 1u;
    u32* map_old = r.state + PIE_BEAM_HEADER + parity * B * cap;
    u32* map_new = r.state + PIE_BEAM_HEADER + (parity ^ 1u) * B * cap;
    u32* free_pages = r.state + PIE_BEAM_HEADER + 2u * B * cap;
    u32* pend = free_pages + cap + (fired % F) * cap;
    u32* copies = free_pages + cap + F * cap;
    const u32 f0 = r.folded / S, tail = r.folded % S;
    const u32 c = r.slot / S - f0, o = r.slot % S;
    const u32 cp = r.slot == 0u ? 0u : (r.slot - 1u) / S - f0;

    __shared__ u32 parent_of[PIE_BEAM_MAX_LANES];
    __shared__ u32 need[PIE_BEAM_MAX_LANES];
    __shared__ u32 rank_of[PIE_BEAM_MAX_LANES];
    __shared__ u32 fresh[PIE_BEAM_MAX_LANES];
    __shared__ u32 hand[PIE_BEAM_MAX_LANES];
    __shared__ u32 counts[256];
    __shared__ u32 needed;
    __shared__ u32 handed;

    if (primed == 0u) {
        // §5: beam b on pool page b; page 0 is the prompt's where it ends
        // mid-page, which beam 0 keeps and every other beam copies.
        for (u32 at = threadIdx.x; at < B * cap; at += blockDim.x)
            map_new[at] = (at % cap) == 0u ? at / cap : 0u;
        for (u32 q = threadIdx.x; q < cap; q += blockDim.x)
            free_pages[q] = (q < Q && q >= B && !(q == 0u && tail > 0u)) ? 1u : 0u;
        for (u32 at = threadIdx.x; at < F * cap; at += blockDim.x) free_pages[cap + at] = 0u;
        if (threadIdx.x == 0u) header[2] = Q;
        for (u32 b = threadIdx.x; b < B; b += blockDim.x) {
            const bool copy = b > 0u && tail > 0u && o > 0u;
            copies[b * 3u + 0u] = 0u;
            copies[b * 3u + 1u] = copy ? b : 0u;
            copies[b * 3u + 2u] = copy ? tail : 0u;
            if (c != 0u) atomicMax(&r.faults[b], PIE_PAGES_SHORT);
            if (B > Q) atomicMax(&r.faults[b], PIE_PAGES_EXHAUSTED);
        }
    } else {
        // §6.1-6.2.
        for (u32 b = threadIdx.x; b < B; b += blockDim.x) {
            const u32 p = r.parents[b];
            if (p >= B) {
                atomicMax(&r.faults[b], PIE_PAGES_PARENT);
                parent_of[b] = b;
            } else {
                parent_of[b] = p;
            }
        }
        // The pool grows a column at a time (§3): pages that joined it since
        // the last fire are free.
        const u32 grown = header[2] < Q ? header[2] : Q;
        for (u32 q = grown + threadIdx.x; q < Q; q += blockDim.x)
            free_pages[q] = (q == 0u && tail > 0u) ? 0u : 1u;
        __syncthreads();
        if (threadIdx.x == 0u) {
            u32 k = 0u;
            for (u32 b = 0u; b < B; ++b) {
                bool first = true;
                for (u32 e = 0u; e < b; ++e) first = first && parent_of[e] != parent_of[b];
                need[b] = (o == 0u || !first) ? 1u : 0u;
                rank_of[b] = k;
                k += need[b];
            }
            needed = k;
        }
        __syncthreads();
        // The lowest free pages, in order: each thread counts its stretch.
        const u32 threads = blockDim.x < 256u ? blockDim.x : 256u;
        const u32 per = (cap + threads - 1u) / threads;
        const u32 lo = threadIdx.x * per, hi = lo + per < cap ? lo + per : cap;
        u32 own = 0u;
        if (threadIdx.x < threads) {
            for (u32 q = lo; q < hi; ++q) own += free_pages[q] != 0u ? 1u : 0u;
            counts[threadIdx.x] = own;
        }
        __syncthreads();
        // The stretches' running counts, a scan over the block.
        for (u32 off = 1u; off < threads; off <<= 1u) {
            u32 add = 0u;
            if (threadIdx.x < threads && threadIdx.x >= off) add = counts[threadIdx.x - off];
            __syncthreads();
            if (threadIdx.x < threads) counts[threadIdx.x] += add;
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            const u32 run = counts[threads - 1u];
            handed = run < needed ? run : needed;
        }
        if (threadIdx.x < threads) counts[threadIdx.x] -= own;
        __syncthreads();
        if (threadIdx.x < threads) {
            u32 k = counts[threadIdx.x];
            for (u32 q = lo; q < hi && k < needed; ++q) {
                if (free_pages[q] != 0u) {
                    hand[k] = q;
                    ++k;
                }
            }
        }
        __syncthreads();
        for (u32 b = threadIdx.x; b < B; b += blockDim.x) {
            const bool gets = need[b] != 0u && rank_of[b] < handed;
            if (need[b] != 0u && !gets) atomicMax(&r.faults[b], PIE_PAGES_EXHAUSTED);
            fresh[b] = gets ? hand[rank_of[b]] : 0xFFFFFFFFu;
        }
        __syncthreads();
        // Only the columns up to this fire's are live (later ones are
        // unspecified, §4).
        const u32 live = c + 1u < cap ? c + 1u : cap;
        for (u32 at = threadIdx.x; at < B * live; at += blockDim.x) {
            const u32 b = at / live, j = at % live;
            const u32 inherited = map_old[parent_of[b] * cap + j];
            map_new[b * cap + j] = (j == c && fresh[b] != 0xFFFFFFFFu) ? fresh[b] : inherited;
        }
        __syncthreads();
        for (u32 b = threadIdx.x; b < B; b += blockDim.x) {
            const bool copy = fresh[b] != 0xFFFFFFFFu && o > 0u;
            copies[b * 3u + 0u] = copy ? map_old[parent_of[b] * cap + c] : 0u;
            copies[b * 3u + 1u] = copy ? fresh[b] : 0u;
            copies[b * 3u + 2u] = copy ? o : 0u;
        }
        // §6.3: the pages freed `frames` fires ago join the free ones; this
        // fire's take their slot of the ring.
        for (u32 q = threadIdx.x; q < cap; q += blockDim.x) {
            bool taken = false;
            for (u32 b = 0u; b < B; ++b) taken = taken || fresh[b] == q;
            u32 f = ((free_pages[q] != 0u && !taken) || pend[q] != 0u) ? 1u : 0u;
            if ((q == 0u && tail > 0u) || q >= Q) f = 0u;
            free_pages[q] = f;
        }
        __syncthreads();
        for (u32 q = threadIdx.x; q < cap; q += blockDim.x) pend[q] = 0u;
        __syncthreads();
        // An old entry no new row names in its column is freed (a lane that
        // faulted PARENT keeps its own row, so its pages stay named).
        const u32 held_cols = cp + 1u < cap ? cp + 1u : cap;
        for (u32 at = threadIdx.x; at < B * held_cols; at += blockDim.x) {
            const u32 p = at / held_cols, j = at % held_cols;
            const u32 page = map_old[p * cap + j];
            bool held = false;
            for (u32 b = 0u; b < B && !held; ++b) held = map_new[b * cap + j] == page;
            if (!held && page < cap && !(page == 0u && tail > 0u)) pend[page] = 1u;
        }
    }
    __syncthreads();
    // §6.5: each lane's staged entries past `f0` and its rows' write pages.
    for (u32 i = 0u; i < r.list_count; ++i) {
        const BeamListRun l = r.lists[i];
        const u32* list = map_new + l.lane * cap;
        if (l.entries > cap) {
            if (threadIdx.x == 0u) atomicMax(&r.faults[l.lane], PIE_PAGES_SHORT);
            continue;
        }
        for (u32 j = threadIdx.x; j < l.entries; j += blockDim.x) {
            const u32 q = list[j];
            if (q >= K) {
                atomicMax(&r.faults[l.lane], q >= Q ? PIE_PAGES_PAST_RESERVE : PIE_PAGES_EXHAUSTED);
                continue;
            }
            l.indices[j] = r.trans[q];
        }
        for (u32 row = threadIdx.x; row < l.rows; row += blockDim.x) {
            const u32 slot = r.slot + row;
            const u32 col = slot / S - f0;
            if (col >= cap) {
                atomicMax(&r.faults[l.lane], PIE_PAGES_SHORT);
                continue;
            }
            const u32 q = list[col];
            const u64 at = (static_cast<u64>(f0) + q) * S + static_cast<u32>(l.write_offset[row]);
            if (q >= K || at < r.folded || at >= (static_cast<u64>(f0) + Q) * S) {
                atomicMax(&r.faults[l.lane], PIE_PAGES_BELOW_FLOOR);
                continue;
            }
            l.write_page[row] = r.trans[q];
        }
    }
    // §6.5 on a hybrid (§11): each lane's rs page table over the same list,
    // slot by slot: entry e (slot `base + e T`) holds the rows of the set's
    // slot (f0 + list[j]) S + (slot mod S), j its column; its write page
    // among them.
    for (u32 i = 0u; i < r.rs_count; ++i) {
        const BeamRsRun l = r.rs[i];
        const u32* list = map_new + l.lane * cap;
        const u64 T = l.tokens == 0u ? 1u : l.tokens;
        for (u32 e = threadIdx.x; e < l.entries; e += blockDim.x) {
            const u64 slot = l.base + e * T;
            if (slot < static_cast<u64>(f0) * S) continue;
            const u64 j = slot / S - f0;
            if (j >= cap) {
                atomicMax(&r.faults[l.lane], PIE_PAGES_SHORT);
                continue;
            }
            const u32 q = list[j];
            if (q >= K) {
                atomicMax(&r.faults[l.lane], q >= Q ? PIE_PAGES_PAST_RESERVE : PIE_PAGES_EXHAUSTED);
                continue;
            }
            const u64 sigma = (static_cast<u64>(f0) + q) * S + slot % S;
            const u64 at = (sigma - l.base) / T;
            if (sigma < l.base || at >= l.bases_len) {
                atomicMax(&r.faults[l.lane], PIE_PAGES_PAST_RESERVE);
                continue;
            }
            l.pages[l.from + e] = l.bases[at];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        header[0] = 1u;
        header[1] = fired + 1u;
        header[2] = Q;
    }
}

}

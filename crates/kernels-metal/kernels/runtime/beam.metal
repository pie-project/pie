// Beam page lists (new-memory/designs/BEAM-PAGE-LISTS.md): a beam group's
// head step, on the device in the fire's own command buffer ahead of its
// first append or read. The Metal port of engine-cuda's `beam_step` and
// `copy_pages` (kernels-cuda/kernels/runtime/channel.cu): the same state
// layout, the same run records (every address a GPU virtual address), the
// same fault codes.
#include <metal_stdlib>
#include "../prim/kv_page.h"
using namespace metal;

#define PIE_PAGES_PAST_RESERVE 1u
#define PIE_PAGES_SHORT 2u
#define PIE_PAGES_BELOW_FLOOR 3u
#define PIE_PAGES_COPY_BOUNDS 4u
#define PIE_PAGES_PARENT 5u
#define PIE_PAGES_EXHAUSTED 6u
#define PIE_PAGES_TOO_WIDE 7u
#define PIE_BEAM_HEADER 4u
#define PIE_BEAM_THREADS 256u

// A lane's page table to resolve: its staged indices from entry `f0`, its
// rows' staged write pages and offsets.
struct BeamListRun {
    ulong indices;
    ulong write_page;
    ulong write_offset;
    uint lane;
    uint entries;
    uint rows;
    uint pad;
};

// A hybrid lane's rs page table to resolve (§11.1): its row's slot entries
// from the one holding buffer entry `first` (`pages`: the row past its
// in-page header), entry `e` the rows of slot `base + e * tokens` on, each
// resolved to the set's entry holding the same rows of its beam's list's
// page (`bases`: the set's entries' slab slots, `bases_len` of them, from
// its entry 0), through `entries` (its last row's entry, plus one).
struct BeamRsRun {
    ulong pages;
    ulong bases;
    uint lane;
    uint first;
    uint entries;
    uint base;
    uint tokens;
    uint bases_len;
};

// A group's step: its state, the parents its last epilogue wrote, its pool's
// device page ids, its lanes' list runs and fault words, the numbers of §3,
// and on a hybrid its lanes' rs page tables (§11).
struct BeamStepRun {
    ulong state;
    ulong parents;
    ulong trans;
    ulong lists;
    ulong faults;
    uint lanes;
    uint cap;
    uint reserve;
    uint page;
    uint folded;
    uint slot;
    uint list_count;
    uint frames;
    uint backed;
    uint rs_count;
    ulong rs;
};

// A lane's page copy (the copy `beam_step` left in its group's state):
// `[src, dst, tokens]` at `copy`, pool pages' base addresses at `base`.
struct PageCopyRun {
    ulong copy;
    ulong base;
    ulong fault;
    uint reserve;
    uint page;
    uint floor_rel;
    uint plane_at;
    uint planes;
    uint pad;
};

// Where a plane lies in a page and the bytes a token holds in it.
struct PagePlane {
    ulong offset;
    ulong cell;
};

static inline void fault_max(ulong faults, uint lane, uint code) {
    device atomic_uint* word = reinterpret_cast<device atomic_uint*>(faults) + lane;
    atomic_fetch_max_explicit(word, code, memory_order_relaxed);
}

// One threadgroup a group. §5 on the group's first fire, §6.1-6.3 after,
// then §6.5 over its lanes' staged tables; leaves this fire's copies (§6.4)
// in its state for `beam_copy_pages`.
kernel void beam_step(
    const device BeamStepRun* runs [[buffer(0)]],
    constant uint& run_count [[buffer(1)]],
    // The step's per-beam arrays, sized at dispatch for the widest group of
    // the dispatch: five of B words, then the B x B common-ancestor table
    // (`beam_step_words`). `tg_words` is how many words the block holds.
    constant uint& tg_words [[buffer(2)]],
    threadgroup uint* tg [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_in [[threads_per_threadgroup]])
{
    if (group >= run_count) return;
    const BeamStepRun r = runs[group];
    const uint B = r.lanes, cap = r.cap, Q = r.reserve, S = r.page;
    const uint F = r.frames == 0u ? 1u : r.frames;
    const uint K = r.backed < Q ? r.backed : Q;
    // A group wider than the dispatch's block holds (the host sizes the block
    // for its widest group and refuses at bind a group past the device's
    // threadgroup memory): every lane faulted by name, never a step that
    // leaves its lists unresolved.
    if (B * (B + 5u) > tg_words) {
        for (uint b = tid; b < B; b += threads_in) fault_max(r.faults, b, PIE_PAGES_TOO_WIDE);
        return;
    }
    if (B == 0u || Q > cap || S == 0u) return;
    device uint* state = reinterpret_cast<device uint*>(r.state);
    const device uint* parents = reinterpret_cast<const device uint*>(r.parents);
    const device int* trans = reinterpret_cast<const device int*>(r.trans);
    const device BeamListRun* lists = reinterpret_cast<const device BeamListRun*>(r.lists);
    device uint* header = state;
    const uint primed = header[0];
    const uint fired = header[1];
    const uint parity = fired & 1u;
    device uint* map_old = state + PIE_BEAM_HEADER + parity * B * cap;
    device uint* map_new = state + PIE_BEAM_HEADER + (parity ^ 1u) * B * cap;
    device uint* free_pages = state + PIE_BEAM_HEADER + 2u * B * cap;
    device uint* pend = free_pages + cap + (fired % F) * cap;
    device uint* copies = free_pages + cap + F * cap;
    const uint f0 = r.folded / S, tail = r.folded % S;
    const uint c = r.slot / S - f0, o = r.slot % S;
    const uint cp = r.slot == 0u ? 0u : (r.slot - 1u) / S - f0;

    threadgroup uint* parent_of = tg;
    threadgroup uint* need = tg + B;
    threadgroup uint* rank_of = tg + 2u * B;
    threadgroup uint* fresh = tg + 3u * B;
    threadgroup uint* hand = tg + 4u * B;
    threadgroup uint counts[PIE_BEAM_THREADS];
    threadgroup uint needed;
    threadgroup uint handed;

    if (primed == 0u) {
        // §5: beam b on pool page b; page 0 is the prompt's where it ends
        // mid-page, which beam 0 keeps and every other beam copies.
        for (uint at = tid; at < B * cap; at += threads_in)
            map_new[at] = (at % cap) == 0u ? at / cap : 0u;
        for (uint q = tid; q < cap; q += threads_in)
            free_pages[q] = (q < Q && q >= B && !(q == 0u && tail > 0u)) ? 1u : 0u;
        for (uint at = tid; at < F * cap; at += threads_in) free_pages[cap + at] = 0u;
        if (tid == 0u) header[2] = Q;
        for (uint b = tid; b < B; b += threads_in) {
            const bool copy = b > 0u && tail > 0u && o > 0u;
            copies[b * 3u + 0u] = 0u;
            copies[b * 3u + 1u] = copy ? b : 0u;
            copies[b * 3u + 2u] = copy ? tail : 0u;
            if (c != 0u) fault_max(r.faults, b, PIE_PAGES_SHORT);
            if (B > Q) fault_max(r.faults, b, PIE_PAGES_EXHAUSTED);
        }
    } else {
        // §6.1: a parent past the beams continues itself.
        for (uint b = tid; b < B; b += threads_in) {
            const uint p = parents[b];
            if (p >= B) {
                fault_max(r.faults, b, PIE_PAGES_PARENT);
                parent_of[b] = b;
            } else {
                parent_of[b] = p;
            }
        }
        // The pool grows a column at a time (§3): pages that joined it since
        // the last fire are free.
        const uint grown = header[2] < Q ? header[2] : Q;
        for (uint q = grown + tid; q < Q; q += threads_in)
            free_pages[q] = (q == 0u && tail > 0u) ? 0u : 1u;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        // §6.2: who needs a fresh page, and its rank among those that do.
        if (tid == 0u) {
            uint k = 0u;
            for (uint b = 0u; b < B; ++b) {
                bool first = true;
                for (uint e = 0u; e < b; ++e) first = first && parent_of[e] != parent_of[b];
                need[b] = (o == 0u || !first) ? 1u : 0u;
                rank_of[b] = k;
                k += need[b];
            }
            needed = k;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The lowest free pages, in order: each thread counts its stretch,
        // a scan gives each stretch's first rank.
        const uint threads = threads_in < PIE_BEAM_THREADS ? threads_in : PIE_BEAM_THREADS;
        const uint per = (cap + threads - 1u) / threads;
        const uint lo = tid * per, hi = lo + per < cap ? lo + per : cap;
        uint own = 0u;
        if (tid < threads) {
            for (uint q = lo; q < hi; ++q) own += free_pages[q] != 0u ? 1u : 0u;
            counts[tid] = own;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1u; off < threads; off <<= 1u) {
            uint add = 0u;
            if (tid < threads && tid >= off) add = counts[tid - off];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < threads) counts[tid] += add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            const uint run = counts[threads - 1u];
            handed = run < needed ? run : needed;
        }
        if (tid < threads) counts[tid] -= own;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < threads) {
            uint k = counts[tid];
            for (uint q = lo; q < hi && k < needed; ++q) {
                if (free_pages[q] != 0u) {
                    hand[k] = q;
                    ++k;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = tid; b < B; b += threads_in) {
            const bool gets = need[b] != 0u && rank_of[b] < handed;
            if (need[b] != 0u && !gets) fault_max(r.faults, b, PIE_PAGES_EXHAUSTED);
            fresh[b] = gets ? hand[rank_of[b]] : 0xFFFFFFFFu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Only the columns up to this fire's are live (later ones are
        // unspecified, §4).
        const uint live = c + 1u < cap ? c + 1u : cap;
        for (uint at = tid; at < B * live; at += threads_in) {
            const uint b = at / live, j = at % live;
            const uint inherited = map_old[parent_of[b] * cap + j];
            map_new[b * cap + j] = (j == c && fresh[b] != 0xFFFFFFFFu) ? fresh[b] : inherited;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint b = tid; b < B; b += threads_in) {
            const bool copy = fresh[b] != 0xFFFFFFFFu && o > 0u;
            copies[b * 3u + 0u] = copy ? map_old[parent_of[b] * cap + c] : 0u;
            copies[b * 3u + 1u] = copy ? fresh[b] : 0u;
            copies[b * 3u + 2u] = copy ? o : 0u;
        }
        // §6.3: the pages freed `frames` fires ago join the free ones; this
        // fire's take their slot of the ring.
        for (uint q = tid; q < cap; q += threads_in) {
            bool taken = false;
            for (uint b = 0u; b < B; ++b) taken = taken || fresh[b] == q;
            uint f = ((free_pages[q] != 0u && !taken) || pend[q] != 0u) ? 1u : 0u;
            if ((q == 0u && tail > 0u) || q >= Q) f = 0u;
            free_pages[q] = f;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint q = tid; q < cap; q += threads_in) pend[q] = 0u;
        threadgroup_barrier(mem_flags::mem_device);
        // An old entry no new row names in its column is freed (a lane that
        // faulted PARENT keeps its own row, so its pages stay named).
        const uint held_cols = cp + 1u < cap ? cp + 1u : cap;
        for (uint at = tid; at < B * held_cols; at += threads_in) {
            const uint p = at / held_cols, j = at % held_cols;
            const uint page = map_old[p * cap + j];
            bool held = false;
            for (uint b = 0u; b < B && !held; ++b) held = map_new[b * cap + j] == page;
            if (!held && page < cap && !(page == 0u && tail > 0u)) pend[page] = 1u;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // The group's common slot (BEAM-PAGE-LISTS §11.3, §11.4): the depth of
    // its live lanes' most recent common ancestor, whatever the guest forks
    // them for. `lca[i][j]` holds, for lanes i != j, the slot below which
    // their rows agree: lanes forked from one parent agree below this fire's
    // row (counted as parting there: a lower bound, exact where their inputs
    // differ); others agree as far as their parents did. Every row below the
    // least of them was written once and reached the other lanes only by
    // copies: the same bytes in every lane. Capped at the slot past this
    // fire's rows (with one lane every row is common), never below the
    // folded slot; monotone, read by the host at harvest.
    device uint* lca = copies + 3u * B;
    threadgroup uint* lca_new = tg + 5u * B;
    threadgroup atomic_uint common_s;
    if (tid == 0u) atomic_store_explicit(&common_s, r.slot + 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint x = tid; x < B * B; x += threads_in) {
        const uint i = x / B, j = x % B;
        uint depth = 0xFFFFFFFFu;
        if (i != j) {
            if (primed == 0u) {
                depth = r.slot;
            } else {
                const uint pi = parent_of[i], pj = parent_of[j];
                depth = pi == pj ? r.slot : lca[pi * B + pj];
            }
            atomic_fetch_min_explicit(&common_s, depth, memory_order_relaxed);
        }
        lca_new[x] = depth;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint x = tid; x < B * B; x += threads_in) lca[x] = lca_new[x];
    // §6.5: each lane's staged entries past `f0` and its rows' write pages.
    for (uint i = 0u; i < r.list_count; ++i) {
        const BeamListRun l = lists[i];
        const device uint* list = map_new + l.lane * cap;
        if (l.entries > cap) {
            if (tid == 0u) fault_max(r.faults, l.lane, PIE_PAGES_SHORT);
            continue;
        }
        device int* indices = reinterpret_cast<device int*>(l.indices);
        device int* write_page = reinterpret_cast<device int*>(l.write_page);
        const device int* write_offset = reinterpret_cast<const device int*>(l.write_offset);
        for (uint j = tid; j < l.entries; j += threads_in) {
            const uint q = list[j];
            if (q >= K) {
                fault_max(r.faults, l.lane, q >= Q ? PIE_PAGES_PAST_RESERVE : PIE_PAGES_EXHAUSTED);
                continue;
            }
            indices[j] = trans[q];
        }
        for (uint row = tid; row < l.rows; row += threads_in) {
            const uint slot = r.slot + row;
            const uint col = slot / S - f0;
            if (col >= cap) {
                fault_max(r.faults, l.lane, PIE_PAGES_SHORT);
                continue;
            }
            const uint q = list[col];
            const ulong at = (ulong(f0) + q) * S + uint(write_offset[row]);
            if (q >= K || at < r.folded || at >= (ulong(f0) + Q) * S) {
                fault_max(r.faults, l.lane, PIE_PAGES_BELOW_FLOOR);
                continue;
            }
            write_page[row] = trans[q];
        }
    }
    // §11.1 on a hybrid: each lane's rs page table over the same list, slot
    // by slot: entry e (slot `base + e T`) holds the rows of the set's slot
    // (f0 + list[j]) S + (slot mod S), j its column; its write entry among
    // them. A frozen entry (below `f0 S`) keeps the set's staged slot.
    const device BeamRsRun* rs = reinterpret_cast<const device BeamRsRun*>(r.rs);
    for (uint i = 0u; i < r.rs_count; ++i) {
        const BeamRsRun l = rs[i];
        const device uint* list = map_new + l.lane * cap;
        const ulong T = l.tokens == 0u ? 1u : l.tokens;
        device int* pages = reinterpret_cast<device int*>(l.pages);
        const device int* bases = reinterpret_cast<const device int*>(l.bases);
        for (uint e = l.first + tid; e < l.entries; e += threads_in) {
            const ulong slot = ulong(l.base) + ulong(e) * T;
            if (slot < ulong(f0) * S) continue;
            const ulong j = slot / S - f0;
            if (j >= cap) {
                fault_max(r.faults, l.lane, PIE_PAGES_SHORT);
                continue;
            }
            const uint q = list[j];
            if (q >= K) {
                fault_max(r.faults, l.lane, q >= Q ? PIE_PAGES_PAST_RESERVE : PIE_PAGES_EXHAUSTED);
                continue;
            }
            const ulong sigma = (ulong(f0) + q) * S + slot % S;
            if (sigma < l.base || (sigma - l.base) / T >= l.bases_len) {
                fault_max(r.faults, l.lane, PIE_PAGES_PAST_RESERVE);
                continue;
            }
            pages[e - l.first] = bases[(sigma - l.base) / T];
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0u) {
        header[0] = 1u;
        header[1] = fired + 1u;
        header[2] = Q;
        // c_t, never below the folded slot.
        const uint common = atomic_load_explicit(&common_s, memory_order_relaxed);
        header[3] = common > r.folded ? common : r.folded;
    }
}

// Copies each run's first `tokens` slots of page `src` to page `dst`, a
// threadgroup a (run, plane): ahead of the fire's appends and reads. A copy
// out of its bounds copies nothing and states the lane's fault.
kernel void beam_copy_pages(
    const device PageCopyRun* runs [[buffer(0)]],
    const device PagePlane* planes [[buffer(1)]],
    constant uint& run_count [[buffer(2)]],
    uint3 at [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]])
{
    const uint tid = tid3.x, threads_in = threads3.x;
    const uint run = at.x;
    if (run >= run_count) return;
    const PageCopyRun r = runs[run];
    if (at.y >= r.planes) return;
    const device uint* copy = reinterpret_cast<const device uint*>(r.copy);
    const uint src = copy[0];
    const uint dst = copy[1];
    const uint tokens = copy[2];
    if (tokens == 0u || src == dst) return;
    if (src >= r.reserve || dst >= r.reserve || tokens > r.page
        || ulong(dst) * r.page < r.floor_rel) {
        if (at.y == 0u && tid == 0u) fault_max(r.fault, 0u, PIE_PAGES_COPY_BOUNDS);
        return;
    }
    const device ulong* base = reinterpret_cast<const device ulong*>(r.base);
    const PagePlane plane = planes[r.plane_at + at.y];
    const ulong bytes = ulong(tokens) * plane.cell;
    const ulong from_at = base[src] + plane.offset;
    const ulong to_at = base[dst] + plane.offset;
    if ((bytes & 15ul) == 0ul && (from_at & 15ul) == 0ul && (to_at & 15ul) == 0ul) {
        const device uint4* f = reinterpret_cast<const device uint4*>(from_at);
        device uint4* t = reinterpret_cast<device uint4*>(to_at);
        for (ulong i = tid; i < bytes / 16ul; i += threads_in) t[i] = f[i];
    } else {
        const device uchar* f = reinterpret_cast<const device uchar*>(from_at);
        device uchar* t = reinterpret_cast<device uchar*>(to_at);
        for (ulong i = tid; i < bytes; i += threads_in) t[i] = f[i];
    }
}

// A beam's rs copy (§11.2): the first `tokens` slots of pool page `src` to
// pool page `dst` (the `[src, dst, tokens]` its group's step left at
// `copy`), row by row of each plane, through the set's rs entries (`bases`:
// slab slots, as `BeamRsRun`'s, each a page of `page_bytes` in `slab`): a
// slot below the buffer (the prompt's tail, in the state) has no row and is
// skipped.
struct RsCopyRun {
    ulong copy;
    ulong bases;
    ulong fault;
    ulong slab;
    uint reserve;
    uint page;
    uint tokens;
    uint base;
    uint f0;
    uint floor_rel;
    uint plane_at;
    uint planes;
    uint bases_len;
    uint page_bytes;
};

// Copies each run's rows, a threadgroup a (run, plane), a slot's row a step:
// after the head step and its kv copies, ahead of the fire's first append or
// read. A copy out of its bounds copies nothing past the fault and states
// the lane's fault.
kernel void copy_rs_rows(
    const device RsCopyRun* runs [[buffer(0)]],
    const device PagePlane* planes [[buffer(1)]],
    constant uint& run_count [[buffer(2)]],
    uint3 at [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]])
{
    const uint tid = tid3.x, threads_in = threads3.x;
    const uint run = at.x;
    if (run >= run_count) return;
    const RsCopyRun r = runs[run];
    if (at.y >= r.planes) return;
    const device uint* copy = reinterpret_cast<const device uint*>(r.copy);
    const uint src = copy[0], dst = copy[1], tokens = copy[2];
    if (tokens == 0u || src == dst) return;
    device atomic_uint* fault = reinterpret_cast<device atomic_uint*>(r.fault);
    if (src >= r.reserve || dst >= r.reserve || tokens > r.page
        || ulong(dst) * r.page < r.floor_rel) {
        if (at.y == 0u && tid == 0u) {
            atomic_fetch_max_explicit(fault, PIE_PAGES_COPY_BOUNDS, memory_order_relaxed);
        }
        return;
    }
    const PagePlane plane = planes[r.plane_at + at.y];
    const ulong T = r.tokens == 0u ? 1u : r.tokens;
    const device int* bases = reinterpret_cast<const device int*>(r.bases);
    device uchar* slab = reinterpret_cast<device uchar*>(r.slab);
    for (uint k = 0u; k < tokens; ++k) {
        const ulong from_slot = (ulong(r.f0) + src) * r.page + k;
        const ulong to_slot = (ulong(r.f0) + dst) * r.page + k;
        if (from_slot < r.base || to_slot < r.base) continue;
        const ulong fe = (from_slot - r.base) / T, te = (to_slot - r.base) / T;
        if (fe >= r.bases_len || te >= r.bases_len) {
            if (tid == 0u) atomic_fetch_max_explicit(fault, PIE_PAGES_COPY_BOUNDS, memory_order_relaxed);
            return;
        }
        const device uchar* from = slab + pie_page_at(uint(bases[fe]), r.page_bytes, 1) + plane.offset
                                   + ((from_slot - r.base) % T) * plane.cell;
        device uchar* to = slab + pie_page_at(uint(bases[te]), r.page_bytes, 1) + plane.offset
                           + ((to_slot - r.base) % T) * plane.cell;
        for (ulong b = tid; b < plane.cell; b += threads_in) to[b] = from[b];
    }
}

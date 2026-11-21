#include "prim/params.slang"
#include "prim/pool.slang"
#include "prim/barrier.slang"

// Beam page lists (new-memory/designs/BEAM-PAGE-LISTS.md), the Vulkan /
// WebGPU port of engine-cuda's `beam_step`, `copy_pages` and `copy_rs_rows`
// (kernels-cuda/kernels/runtime/channel.cu): the same state layout, records
// and fault codes. These backends hold no device addresses, so every place
// is a u32 word index: into the arm's input store (bound whole, from its
// first word: the records, each group's pool page ids, its lanes' staged
// page tables and fault words, the parents cell copied in ahead of the
// step), or into the chunk pool (a group's resident state, the kv and rs
// pages a copy moves).
//
// A group's state, u32 words of the pool: a header (primed, fires run, the
// pool's pages at the last fire, one free word), its lists twice (the
// fire's and the one before, by the fire count's parity), its free pages, a
// ring of `frames` pending pages (a page freed by fire t joins the free ones
// at fire t + frames + 1), and this fire's copies, `[src, dst, tokens]` a
// beam.

[[vk::binding(0)]] RWStructuredBuffer<uint> store;
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, 1)

struct Params {
    uint seg_words;
    // The first record's word in the store, and the records.
    uint runs;
    uint count;
    // The plane table's first word in the store (the copies').
    uint planes;
};
PIE_PARAMS(5, Params, params);
PIE_POOL_READ(uint, params.seg_words)
PIE_POOL_WRITE(uint, params.seg_words)

#define PIE_PAGES_PAST_RESERVE 1u
#define PIE_PAGES_SHORT 2u
#define PIE_PAGES_BELOW_FLOOR 3u
#define PIE_PAGES_COPY_BOUNDS 4u
#define PIE_PAGES_PARENT 5u
#define PIE_PAGES_EXHAUSTED 6u
#define PIE_BEAM_HEADER 4u
#define PIE_BEAM_MAX_LANES 64u
#define PIE_BEAM_NONE 0xFFFFFFFFu

#if defined(PIE_BEAM_STEP)

// A group's step record, 16 words: its state (pool word), its parents
// (store word), its pool's page ids (store word), its list records (store
// word, 8 words each: the staged indices' word at entry f0, the rows' write
// pages' and write offsets' words, the beam, the entries, the rows), its
// lanes' fault words (store word), the numbers of §3, and its rs list
// records (store word, 8 words each: the rs table's first entry word, the
// set's rs entries' word, the beam, the entries, the buffer's base slot,
// its rows a page, the set's entries).
#define REC 16u

groupshared uint g_rec[REC];
groupshared uint parent_of[PIE_BEAM_MAX_LANES];
groupshared uint need[PIE_BEAM_MAX_LANES];
groupshared uint rank_of[PIE_BEAM_MAX_LANES];
groupshared uint fresh[PIE_BEAM_MAX_LANES];
groupshared uint hand[PIE_BEAM_MAX_LANES];
groupshared uint counts[PIE_GROUP_X];
groupshared uint g_needed;
groupshared uint g_handed;
groupshared Atomic<uint> g_fault[PIE_BEAM_MAX_LANES];

void raise(uint b, uint code) {
    if (b < PIE_BEAM_MAX_LANES) {
        g_fault[b].max(code);
    }
}

// One workgroup a group. Every barrier stands in the kernel's top-level
// flow (WGSL holds a barrier to uniform control flow); a group whose
// record does not hold, or a step its state skips, only skips the work
// between them. §5 on the group's first fire, §6.1-6.3 after, then §6.5
// over its lanes' staged tables; leaves this fire's copies (§6.4) in its
// state for `copy_pages` and `copy_rs_rows`.
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID) {
    const uint t = local.x;
    const uint T = PIE_GROUP_X;
    const bool live_group = group.x < params.count;
    if (live_group && t < REC) {
        g_rec[t] = store[params.runs + group.x * REC + t];
    }
    if (t < PIE_BEAM_MAX_LANES) {
        g_fault[t].store(0u);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint state = g_rec[0];
    const uint parents = g_rec[1];
    const uint trans = g_rec[2];
    const uint lists = g_rec[3];
    const uint faults = g_rec[4];
    const uint B = g_rec[5];
    const uint cap = g_rec[6];
    const uint Q = g_rec[7];
    const uint S = max(g_rec[8], 1u);
    const uint folded = g_rec[9];
    const uint slot = g_rec[10];
    const uint list_count = g_rec[11];
    const uint F = max(g_rec[12], 1u);
    const uint backed = g_rec[13];
    const uint rs_count = g_rec[14];
    const uint rs = g_rec[15];
    const bool ok = live_group && B > 0u && B <= PIE_BEAM_MAX_LANES && Q <= cap;
    // The pool pages the fire stages (`reserve` is the pool's whole): the
    // lowest-free rule keeps every page the step names below them.
    const uint K = min(backed, Q);

    const uint primed = ok ? pie_pool_get(state) : 0u;
    const uint fired = ok ? pie_pool_get(state + 1u) : 0u;
    const uint pool_was = ok ? pie_pool_get(state + 2u) : 0u;
    const uint parity = fired & 1u;
    const uint map_old = state + PIE_BEAM_HEADER + parity * B * cap;
    const uint map_new = state + PIE_BEAM_HEADER + (parity ^ 1u) * B * cap;
    const uint free_pages = state + PIE_BEAM_HEADER + 2u * B * cap;
    const uint pend = free_pages + cap + (fired % F) * cap;
    const uint copies = free_pages + cap + F * cap;
    const uint f0 = folded / S;
    const uint tail = folded % S;
    const uint c = slot / S - f0;
    const uint o = slot % S;
    const uint cp = slot == 0u ? 0u : (slot - 1u) / S - f0;
    const bool priming = ok && primed == 0u;
    const bool stepping = ok && primed != 0u;

    if (priming) {
        // §5: beam b on pool page b; page 0 is the prompt's where it ends
        // mid-page, which beam 0 keeps and every other beam copies.
        for (uint at = t; at < B * cap; at += T) {
            pie_pool_set(map_new + at, (at % cap) == 0u ? at / cap : 0u);
        }
        for (uint q = t; q < cap; q += T) {
            pie_pool_set(free_pages + q, (q < Q && q >= B && !(q == 0u && tail > 0u)) ? 1u : 0u);
        }
        for (uint at = t; at < F * cap; at += T) {
            pie_pool_set(free_pages + cap + at, 0u);
        }
        for (uint b = t; b < B; b += T) {
            const bool copy = b > 0u && tail > 0u && o > 0u;
            pie_pool_set(copies + b * 3u + 0u, 0u);
            pie_pool_set(copies + b * 3u + 1u, copy ? b : 0u);
            pie_pool_set(copies + b * 3u + 2u, copy ? tail : 0u);
            if (c != 0u) {
                raise(b, PIE_PAGES_SHORT);
            }
            if (B > Q) {
                raise(b, PIE_PAGES_EXHAUSTED);
            }
        }
    }
    if (stepping) {
        // §6.1.
        for (uint b = t; b < B; b += T) {
            const uint p = store[parents + b];
            if (p >= B) {
                raise(b, PIE_PAGES_PARENT);
                parent_of[b] = b;
            } else {
                parent_of[b] = p;
            }
        }
        // The pool grows a column at a time (§3): pages that joined it
        // since the last fire are free.
        for (uint q = min(pool_was, Q) + t; q < Q; q += T) {
            pie_pool_set(free_pages + q, (q == 0u && tail > 0u) ? 0u : 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // §6.2: which beams take a fresh page, and their order.
    if (stepping && t == 0u) {
        uint k = 0u;
        for (uint b = 0u; b < B; ++b) {
            bool first = true;
            for (uint e = 0u; e < b; ++e) {
                first = first && parent_of[e] != parent_of[b];
            }
            need[b] = (o == 0u || !first) ? 1u : 0u;
            rank_of[b] = k;
            k += need[b];
        }
        g_needed = k;
    }
    // The lowest free pages, in order: each lane counts its stretch.
    const uint per = (cap + T - 1u) / T;
    const uint lo = min(t * per, cap);
    const uint hi = min(lo + per, cap);
    uint own = 0u;
    if (stepping) {
        for (uint q = lo; q < hi; ++q) {
            own += pie_pool_get(free_pages + q) != 0u ? 1u : 0u;
        }
    }
    counts[t] = own;
    GroupMemoryBarrierWithGroupSync();
    // The stretches' running counts, a scan over the workgroup.
    for (uint off = 1u; off < T; off <<= 1u) {
        uint add = 0u;
        if (t >= off) {
            add = counts[t - off];
        }
        GroupMemoryBarrierWithGroupSync();
        counts[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    const uint needed = g_needed;
    if (t == 0u) {
        g_handed = min(counts[T - 1u], needed);
    }
    counts[t] -= own;
    GroupMemoryBarrierWithGroupSync();
    if (stepping) {
        uint k = counts[t];
        for (uint q = lo; q < hi && k < needed; ++q) {
            if (pie_pool_get(free_pages + q) != 0u) {
                if (k < PIE_BEAM_MAX_LANES) {
                    hand[k] = q;
                }
                ++k;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    const uint handed = g_handed;
    if (stepping) {
        for (uint b = t; b < B; b += T) {
            const bool gets = need[b] != 0u && rank_of[b] < handed;
            if (need[b] != 0u && !gets) {
                raise(b, PIE_PAGES_EXHAUSTED);
            }
            fresh[b] = gets ? hand[rank_of[b]] : PIE_BEAM_NONE;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (stepping) {
        // Only the columns up to this fire's are live (later ones are
        // unspecified, §4).
        const uint live = min(c + 1u, cap);
        for (uint at = t; at < B * live; at += T) {
            const uint b = at / live;
            const uint j = at % live;
            const uint inherited = pie_pool_get(map_old + parent_of[b] * cap + j);
            pie_pool_set(map_new + b * cap + j,
                (j == c && fresh[b] != PIE_BEAM_NONE) ? fresh[b] : inherited);
        }
        for (uint b = t; b < B; b += T) {
            const bool copy = fresh[b] != PIE_BEAM_NONE && o > 0u;
            pie_pool_set(copies + b * 3u + 0u,
                copy ? pie_pool_get(map_old + parent_of[b] * cap + c) : 0u);
            pie_pool_set(copies + b * 3u + 1u, copy ? fresh[b] : 0u);
            pie_pool_set(copies + b * 3u + 2u, copy ? o : 0u);
        }
        // §6.3: the pages freed `frames` fires ago join the free ones; this
        // fire's take their slot of the ring.
        for (uint q = t; q < cap; q += T) {
            bool taken = false;
            for (uint b = 0u; b < B; ++b) {
                taken = taken || fresh[b] == q;
            }
            uint f = ((pie_pool_get(free_pages + q) != 0u && !taken)
                || pie_pool_get(pend + q) != 0u) ? 1u : 0u;
            if ((q == 0u && tail > 0u) || q >= Q) {
                f = 0u;
            }
            pie_pool_set(free_pages + q, f);
        }
    }
    pie_storage_barrier();
    if (stepping) {
        for (uint q = t; q < cap; q += T) {
            pie_pool_set(pend + q, 0u);
        }
    }
    pie_storage_barrier();
    if (stepping) {
        // An old entry no new row names in its column is freed (a lane
        // that faulted PARENT keeps its own row, so its pages stay named).
        const uint held_cols = min(cp + 1u, cap);
        for (uint at = t; at < B * held_cols; at += T) {
            const uint p = at / held_cols;
            const uint j = at % held_cols;
            const uint page = pie_pool_get(map_old + p * cap + j);
            bool held = false;
            for (uint b = 0u; b < B && !held; ++b) {
                held = pie_pool_get(map_new + b * cap + j) == page;
            }
            if (!held && page < cap && !(page == 0u && tail > 0u)) {
                pie_pool_set(pend + page, 1u);
            }
        }
    }
    pie_storage_barrier();
    if (ok) {
        // §6.5: each lane's staged entries past `f0` and its rows' write
        // pages.
        for (uint i = 0u; i < list_count; ++i) {
            const uint l = lists + i * 8u;
            const uint indices = store[l + 0u];
            const uint write_page = store[l + 1u];
            const uint write_offset = store[l + 2u];
            const uint lane = store[l + 3u];
            const uint entries = store[l + 4u];
            const uint rows = store[l + 5u];
            const uint list = map_new + lane * cap;
            if (entries > cap) {
                if (t == 0u) {
                    raise(lane, PIE_PAGES_SHORT);
                }
                continue;
            }
            for (uint j = t; j < entries; j += T) {
                const uint q = pie_pool_get(list + j);
                if (q >= K) {
                    raise(lane, q >= Q ? PIE_PAGES_PAST_RESERVE : PIE_PAGES_EXHAUSTED);
                    continue;
                }
                store[indices + j] = store[trans + q];
            }
            for (uint row = t; row < rows; row += T) {
                const uint col = (slot + row) / S - f0;
                if (col >= cap) {
                    raise(lane, PIE_PAGES_SHORT);
                    continue;
                }
                const uint q = pie_pool_get(list + col);
                // In slots of the set's: past u32 only where `f0 + Q` pages
                // are, which a page table of u32 entries never names.
                const uint at = (f0 + q) * S + store[write_offset + row];
                if (q >= K || at < folded || at >= (f0 + Q) * S) {
                    raise(lane, PIE_PAGES_BELOW_FLOOR);
                    continue;
                }
                store[write_page + row] = store[trans + q];
            }
        }
        // §6.5 on a hybrid (§11): each lane's rs page table over the same
        // list, slot by slot: entry e (slot `base + e T`) holds the rows of
        // the set's slot (f0 + list[j]) S + (slot mod S), j its column; its
        // write page among them.
        for (uint i = 0u; i < rs_count; ++i) {
            const uint l = rs + i * 8u;
            const uint pages = store[l + 0u];
            const uint bases = store[l + 1u];
            const uint lane = store[l + 2u];
            const uint entries = store[l + 3u];
            const uint base = store[l + 4u];
            const uint rows_a_page = max(store[l + 5u], 1u);
            const uint bases_len = store[l + 6u];
            const uint list = map_new + lane * cap;
            for (uint e = t; e < entries; e += T) {
                const uint at_slot = base + e * rows_a_page;
                if (at_slot < f0 * S) {
                    continue;
                }
                const uint j = at_slot / S - f0;
                if (j >= cap) {
                    raise(lane, PIE_PAGES_SHORT);
                    continue;
                }
                const uint q = pie_pool_get(list + j);
                if (q >= K) {
                    raise(lane, q >= Q ? PIE_PAGES_PAST_RESERVE : PIE_PAGES_EXHAUSTED);
                    continue;
                }
                const uint sigma = (f0 + q) * S + at_slot % S;
                if (sigma < base || (sigma - base) / rows_a_page >= bases_len) {
                    raise(lane, PIE_PAGES_PAST_RESERVE);
                    continue;
                }
                store[pages + e] = store[bases + (sigma - base) / rows_a_page];
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (ok && t == 0u) {
        pie_pool_set(state + 0u, 1u);
        pie_pool_set(state + 1u, fired + 1u);
        pie_pool_set(state + 2u, Q);
    }
    if (ok && t < B) {
        store[faults + t] = g_fault[t].load();
    }
}

// pie:instantiate runtime_beam_step PIE_BEAM_STEP=1 PIE_GROUP_X=256

#elif defined(PIE_BEAM_COPY)

// A beam's copy over the kv planes a page holds, 8 words: its `[src, dst,
// tokens]` (pool word, in its group's state), its group's pool page ids
// (store word), its fault word (store word), the pool pages staged, the
// page's slots, the frozen slots on pool page 0, the pool words a page id
// counts, and its planes. A plane, 2 words in the plane table: its first
// word in a page and the words a slot holds there.
//
// Copies each run's first `tokens` slots of pool page `src` to `dst`, a
// workgroup a (run, plane): ahead of the fire's appends and reads, in queue
// order. A copy out of its bounds (a page past the staged ones, more slots
// than a page, a destination slot below the frozen ones) copies nothing and
// states the lane's fault.
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID) {
    const uint run = group.x;
    if (run >= params.count) {
        return;
    }
    const uint r = params.runs + run * 8u;
    const uint planes = store[r + 7u];
    if (group.y >= planes) {
        return;
    }
    const uint copy = store[r + 0u];
    const uint ids = store[r + 1u];
    const uint fault = store[r + 2u];
    const uint reserve = store[r + 3u];
    const uint page = store[r + 4u];
    const uint floor_rel = store[r + 5u];
    const uint unit = store[r + 6u];
    const uint src = pie_pool_get(copy + 0u);
    const uint dst = pie_pool_get(copy + 1u);
    const uint tokens = pie_pool_get(copy + 2u);
    if (tokens == 0u || src == dst) {
        return;
    }
    if (src >= reserve || dst >= reserve || tokens > page || dst * page < floor_rel) {
        if (group.y == 0u && local.x == 0u) {
            store[fault] = max(store[fault], PIE_PAGES_COPY_BOUNDS);
        }
        return;
    }
    const uint offset = store[params.planes + 2u * group.y];
    const uint cell = store[params.planes + 2u * group.y + 1u];
    const uint from = store[ids + src] * unit + offset;
    const uint to = store[ids + dst] * unit + offset;
    for (uint at = local.x; at < tokens * cell; at += PIE_GROUP_X) {
        pie_pool_set(to + at, pie_pool_get(from + at));
    }
}

// pie:instantiate runtime_beam_copy_pages PIE_BEAM_COPY=1 PIE_GROUP_X=256

#elif defined(PIE_BEAM_RS)

// A beam's rs copy (§6.4 on a hybrid, §11.2), 16 words: its `[src, dst,
// tokens]` (pool word), the set's rs entries' units (store word), its fault
// word (store word), the pool pages staged, the kv page's slots, the rows an
// rs page holds, the buffer's base slot, `f0`, the frozen slots on pool
// page 0, the pool words an rs unit counts, its planes, the set's entries.
// A plane, 2 words in the plane table: its first word in an rs page and the
// words a row holds there.
//
// Copies each run's rows, a workgroup a (run, plane), a slot's row a step: a
// slot below the buffer (the prompt's tail, in the state) has no row and is
// skipped.
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID) {
    const uint run = group.x;
    if (run >= params.count) {
        return;
    }
    const uint r = params.runs + run * 16u;
    const uint planes = store[r + 10u];
    if (group.y >= planes) {
        return;
    }
    const uint copy = store[r + 0u];
    const uint bases = store[r + 1u];
    const uint fault = store[r + 2u];
    const uint reserve = store[r + 3u];
    const uint page = store[r + 4u];
    const uint rows_a_page = max(store[r + 5u], 1u);
    const uint base = store[r + 6u];
    const uint f0 = store[r + 7u];
    const uint floor_rel = store[r + 8u];
    const uint unit = store[r + 9u];
    const uint bases_len = store[r + 11u];
    const uint src = pie_pool_get(copy + 0u);
    const uint dst = pie_pool_get(copy + 1u);
    const uint tokens = pie_pool_get(copy + 2u);
    if (tokens == 0u || src == dst) {
        return;
    }
    if (src >= reserve || dst >= reserve || tokens > page || dst * page < floor_rel) {
        if (group.y == 0u && local.x == 0u) {
            store[fault] = max(store[fault], PIE_PAGES_COPY_BOUNDS);
        }
        return;
    }
    const uint offset = store[params.planes + 2u * group.y];
    const uint cell = store[params.planes + 2u * group.y + 1u];
    for (uint k = 0u; k < tokens; ++k) {
        const uint from_slot = (f0 + src) * page + k;
        const uint to_slot = (f0 + dst) * page + k;
        if (from_slot < base || to_slot < base) {
            continue;
        }
        const uint fe = (from_slot - base) / rows_a_page;
        const uint te = (to_slot - base) / rows_a_page;
        if (fe >= bases_len || te >= bases_len) {
            if (group.y == 0u && local.x == 0u) {
                store[fault] = max(store[fault], PIE_PAGES_COPY_BOUNDS);
            }
            return;
        }
        const uint from = store[bases + fe] * unit + offset + ((from_slot - base) % rows_a_page) * cell;
        const uint to = store[bases + te] * unit + offset + ((to_slot - base) % rows_a_page) * cell;
        for (uint at = local.x; at < cell; at += PIE_GROUP_X) {
            pie_pool_set(to + at, pie_pool_get(from + at));
        }
    }
}

// pie:instantiate runtime_beam_copy_rs_rows PIE_BEAM_RS=1 PIE_GROUP_X=64

#endif

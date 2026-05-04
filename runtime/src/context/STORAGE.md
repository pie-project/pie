# Tiered Page Storage

Concrete data-structure design for the tiered storage model described in
SCHED.md §6. Companion document — SCHED.md owns the auction mechanics,
this document owns the storage layer.

---

## 1. Page Kinds

| Kind | Mutability | Ownership | Sharing |
|------|-----------|-----------|---------| 
| **Working** | Mutable | Exclusive (one context) | Never shared |
| **Committed** | Immutable | Content-addressed | Shared via refcount |

Working pages are transient buffers for in-progress KV data. Committed
pages are finalized, content-hashed, and shareable across contexts.

---

## 2. Storage Tiers

| Tier | Medium | Restore Cost | Capacity | Data Structure |
|------|--------|-------------|----------|----------------|
| **GPU** | HBM | 0 | `C_gpu` pages | CAS Trie + PagePool |
| **CPU** | Host DRAM | ~0.16ms/page (PCIe) | `C_cpu` pages | FlatMap + PagePool |
| **Recompute** | None | ~10ms/page (replay) | Unlimited | Lineage (always available) |

Recompute is not a storage tier — it is the always-available fallback.
Any page can be regenerated from lineage + working_page_tokens. The
storage layer manages GPU and CPU only.

---

## 3. Data Structures

### 3.1 GPU: CAS Trie (existing `PageStore`)

Compressed Radix Trie with path-inclusive refcounting. Handles:
- Prefix sharing (dedup) across active contexts
- O(depth) insert/release/lookup
- Shapley `effective_pages` for GPU auction

Requires **root-first insertion order** — chains must be inserted from
the root hash outward. This is always satisfied on GPU because contexts
commit pages incrementally via `extend(existing_prefix, new_suffix)`.

**New query**: `would_free(hashes) → Vec<PageHash>` — read-only peek that
returns hashes with rc=1 (pages that *would* reach rc=0 on release). Used
to identify which pages to D2H copy before calling `release()`.

### 3.2 CPU: FlatMap

```rust
struct FlatPageStore {
    /// Hash → (physical CPU page ID, refcount).
    map: HashMap<PageHash, (PhysicalPageId, usize)>,
    /// Physical CPU page allocation pool.
    /// Shared by committed pages, working pages, and snapshot stash.
    pool: PagePool,
}
```

Why FlatMap instead of Trie:
- **Out-of-order insertion.** Pages arrive at CPU in eviction order, not
  chain order. A suffix may be stashed before its prefix (see §4.1).
  The trie requires root-first insertion; FlatMap doesn't care.
- **Same capabilities.** Refcounting, prefix_len, effective_pages, and
  physical_ids all work via hash lookup — no tree structure needed.

### 3.3 Shared Interface

Both tiers expose the same logical operations:

```
insert(hashes, phys_ids)    — add pages (rc=1, or rc++ on dedup)
release(hashes)             — rc--, free pages at rc=0
physical_ids(hashes)        — look up physical page IDs
prefix_len(hashes)          — count consecutive resident hashes
effective_pages(hashes)     — Σ (1 / rc) per hash (Shapley cost)
alloc(n) / free(pages)      — pool operations
```

Implementations differ (trie traversal vs hash lookup) but semantics
are identical. This enables symmetric `move(src, dst)` operations.

### 3.4 Per-Device Layout

```
DeviceStorage {
    gpu: PageStore,       // CAS Trie + GPU PagePool
    cpu: FlatPageStore,    // FlatMap + CPU PagePool (single shared pool)
}
```

The current `PageStore` is refactored to remove its hardcoded `cpu: PagePool`.
Each store manages exactly one pool for its tier. The CPU pool is shared
across all uses: committed page stash, working page stash, snapshot swap.

---

## 4. Page Movement

### 4.1 Committed: GPU → CPU (suspend)

A context excluded from GPU by the auction (SCHED.md §6) descends
through the storage waterfall.

For a context with committed chain `[H0..H99]`:

1. **Peek at evictable pages.** `gpu.would_free(hashes)` returns hashes
   with rc=1 — pages that will reach rc=0 on release and actually leave
   GPU. Shared prefix (rc > 1) stays on GPU, no copy needed.

2. **Stash evictable pages to CPU.**

   **(a) Happy case** — CPU has free pages:
   - `cpu.alloc(n)` → CPU page slots.
   - D2H copy GPU pages → CPU pages.
   - `cpu.insert(hashes, cpu_phys)` — rc=1 (or rc++ if dedup hit).

   **(b) Contended case** — CPU is full:
   - Find CPU eviction victim: iterate all contexts, find the one with
     the **lowest bid** that has pages on CPU (check via
     `cpu.prefix_len(ctx.committed_hashes) > 0`).
   - `cpu.release(victim_hashes)` → rc--, free CPU pages at rc=0.
     Victim falls to recompute (lineage replay on next restore).
   - Retry `cpu.alloc(n)` with freed capacity.
   - If still insufficient: the incoming context also falls to
     recompute (no CPU stash for this suspension).

3. **Release from GPU trie.** `gpu.release(hashes)` — frees rc=0 pages.

Because CPU uses FlatMap, insertion order doesn't matter. A suffix
`[H50..H99]` can be stashed first; its prefix `[H0..H49]` can arrive
later when its GPU refcount drops to zero.

> **Pattern.** The contention flow is the same at every tier boundary:
> try alloc → if full, find lowest-bid victim → evict victim to next
> tier down → retry. GPU→CPU and CPU→recompute use identical logic,
> just different stores and different clearing prices.

### 4.2 Committed: CPU → GPU (restore)

1. `gpu.prefix_len(hashes)` → find GPU-resident prefix length `p`.
2. For missing suffix `[Hp..H99]`:
   - Check `cpu.physical_ids(suffix)` — which pages are on CPU?
   - `gpu.alloc(n)` → fresh GPU page slots.
   - H2D copy from CPU pages to GPU pages.
   - `gpu.extend(prefix, suffix, gpu_pages)` → insert into GPU trie.
   - `cpu.release(suffix)` → rc--, free CPU pages at rc=0.
3. Pages on neither GPU nor CPU → recompute from lineage.

### 4.3 Working: GPU ↔ CPU (suspend/restore)

Working pages are exclusive (not shared), so no trie — just pool ops.
Working and committed pages share the same CPU pool.

**Suspend:**
1. `cpu.alloc(n)` → CPU page slots (from shared pool).
2. D2H copy working GPU pages → CPU pages.
3. `gpu.free(working_gpu)`.
4. Context stores `cpu_working_pages: Vec<PhysicalPageId>`.

**Restore:**
1. `gpu.alloc(n)` → GPU page slots.
2. H2D copy CPU pages → GPU pages.
3. `cpu.free(cpu_working_pages)`.
4. If no CPU stash → recompute from `working_page_tokens` via replay.

---

## 5. Integration with Auction (SCHED.md §6)

### Unified auction

The `tick()` function runs the **same knapsack auction** for both tiers,
parameterized by store and capacity:

```
auction(store, capacity, contexts) → (admitted, excluded, clearing_price)
```

- GPU auction: `auction(gpu, C_gpu, all_contexts)`.
- CPU auction: `auction(cpu, C_cpu, gpu_excluded_contexts)`.
  Uses `cpu.effective_pages(hashes)` for Shapley cost-sharing.

This replaces the current separate GPU auction loop + hand-rolled CPU
auction loop in `sched.rs`.

### CPU bid formula (from SCHED.md §6):

```
cpu_bid_i = (recompute_cost − transfer_cost) / effective_pages_i
```

`effective_pages_i` is computed on the **CPU store** — it reflects
actual CPU sharing.

### Restore priority

Restore queue follows bid only. No preference for CPU-warm contexts
over cold (recompute-only) contexts — the market decides via bids.

---

## 6. Context State Changes

```rust
struct Context {
    // Existing fields (unchanged):
    working_pages: Vec<PhysicalPageId>,      // GPU working pages
    committed_hashes: Vec<PageHash>,          // committed chain (metadata)
    suspended_working_count: usize,           // for recompute restore
    working_page_tokens: Vec<TokenInfo>,      // for recompute restore

    // Replaced:
    //   cpu_cached: bool        → derived from cpu.prefix_len > 0
    //   cpu_page_ids: Vec<..>   → cpu_working_pages (working only)

    // New:
    cpu_working_pages: Vec<PhysicalPageId>,   // CPU stash for working pages
}
```

Committed page CPU residency is tracked by the CPU FlatMap itself
(hash lookup), not by per-context fields.

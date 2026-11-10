# Engine <-> runtime contract for the CUDA pool switch (BC), agreed 2026-11-10

Proposed by the L40S session (branch `sc/l40s-bc`), accepted by New Pie. Runtime side = package RW.

1. **Layout.** One VA of `total_chunks * C`; physical chunk p at `base + p*C` (no per-segment padding). A segment =
   `segment_chunks` chunks; the 2MiB map units a committed segment touches are mapped (a unit on a segment edge stays
   mapped while either neighbour is committed), so C needs no 2MiB alignment.
2. **Activation.** A fire of rung r uses `[(total - act_chunks(r))*C, total*C)` (fixed top; graphs keyed per rung).
   The top `act_chunks_max` chunks are backed from load and never released. `commit_segments(n)` backs segments 0..n
   (prefix), never on the fire path. **Runtime rule:** state chunks live at physical ids
   `< total - act_chunks(current rung)`; raising the rung first evacuates whatever sits in the new activation span
   (SlotPool's `set_rung` / chunk opening must enforce this position rule, not just the count). Governor bytes:
   model the top `act_chunks_max` span as segments committed from load.
3. **kv addressing.** Kernels get `addr = plane_base + q * G`, G = gcd(C, slot_bytes) per token shape,
   q = (physical_chunk*C + index*slot_bytes)/G computed by the engine from the lane's slot refs. List shapes:
   C is a multiple of the slot, so G = slot_bytes and q = chunk*slots_per_chunk + index. Ring shapes use the gcd form.
4. **Lane inputs.** `Lane.tables`: list/ring slot refs in token order (ring ref 0 = null chunk).
   `Lane.fixed: Vec<(ShapeId, Vec<[u32; 2]>)>` — one `[run, cow]` per piece, in piece order.
   `cow == run` is legal and means write in place (the normal decode case); `cow != run` means the engine copies
   run -> cow on the stream before the fire and the fire writes cow (fork / snapshot restore in that fire).
   Snapshots keep `rs_snapshots` with slot refs of the fixed shape. Slot refs are LOGICAL; the engine owns the
   logical -> physical chunk table and applies `Evacuation{moves}` stream-ordered before the next fire.
   Temporary shim (deleted when RW lands): with empty `tables`/`fixed` the engine maps the legacy fields
   (`kv.pages`, `kv.window`, `slot`/`rs_slot`, buffer pages) as slot refs of the matching shape.
5. **Facts.** CUDA reports real `ChunkFacts` (unpinned shapes from `chunk::plan` SERVER_CUDA, `Growth::TwoEnded`,
   `act_chunks` per token bucket). The runtime must accept unpinned/shared shapes (today `StorePools::of` refuses
   them), so BC and RW merge together.

## Amendments (2026-11-10, after RW)
6. **Logical ids:** the engine's `ChunkTable` starts as the identity and changes only through `Evacuation` moves,
   applied in order (`ChunkTable::apply`). The runtime opens a chunk at physical p under the logical id the table maps
   to p.
7. **Segment verbs:** `commit_segments(n)` backs `0..n` (prefix); `release_segments(n)` releases from n up but never
   the top `act_chunks_max` span. The runtime shrinks by evacuating the highest committed prefix segment.
8. **Fixed pieces:** `Lane.fixed` lists every fixed shape with exactly `pieces` pairs. The rs buffer sits inside the
   fixed record (no buffer pages). `cow != run` is the only state copy the runtime asks for; no `StateCopy` for records.
9. **Snapshots:** `RsSnapshot` carries one ref per piece of the snapshot's fixed record (Vec, piece order) — to be
   changed on both sides; today it is a single ref (first piece), valid only for one-piece records.
10. **`sets_per_working_set` = 1** for fixed shapes (one set per working set; the CoW reserve is one record per lane a
    fire runs, held by the runtime). Engine validation that expects 2 must change to 1.
11. **Shapes:** at most one list and one ring shape per load; the list slot's tokens equal the kv page size;
    ring/list refs are slot refs `chunk * slots_per_chunk + index` (ring ref 0 = null chunk).
12. **Multi-lane groups on one working set:** the first lane gets `[src, dst]`, the others `[src, src]`.
    Forked rings still copy through `kv.window_copies` (to fold into `copy_slots` later). Host/disk tiers for shared
    shapes come with `copy_slots`.

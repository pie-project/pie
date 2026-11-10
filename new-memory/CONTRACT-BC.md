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
   C is a multiple of the slot, so G = slot_bytes and q = chunk*slots_per_chunk + index. Ring shapes use the gcd form (removed by 19).
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
12. **Multi-lane groups on one working set:** in shared mode every lane of the group names the record the group
    writes (`[src, dst]` on copy-on-write, `[x, x]` in place). Lanes 1.. read the state from the record lane 0 folds
    into in the same fire, so the engine must fold lane 0 only after the other lanes have read the state.
    Forked rings still copy through `kv.window_copies` (to fold into `copy_slots` later). Host/disk tiers for shared
    shapes come with `copy_slots`.
13. **Shared-mode rs buffer.** `RsVerb::Buffer.pages` ids are refcount bookkeeping only (ignore for addressing).
    `base` moves in whole buffer pages (kv page size), `folded` = frozen; rows written/replayed lie in
    `[folded, writes.end)`. Engine maps buffered row `s` to row `s % 16` of the record's `[16, width]` rs rows
    (cow record if cow != run, else run). The runtime guarantees `writes.end - folded <= 16` (RW follow-up: enforce);
    the engine refuses otherwise. No page copies in shared mode: a buffered write onto a shared record forces record
    CoW, so `Lane.fixed`'s run -> cow copy carries the buffer rows.
14. **Activation segment backing.** The engine backs the whole segment that contains the activation's start (from
    that segment's first chunk), so the runtime's "top segments committed from load" model holds and state opened
    below the activation inside that segment is backed.
15. **Multi-lane group folds (supersedes the in-place part of 12).** CUDA runs a fire's lanes concurrently. A lane
    whose fire folds nothing reads its state from `run` and lands its rows in `written`; a folding lane reads and
    writes `written` after the single run -> written copy. Therefore a multi-lane group in which any lane folds names
    `[src, dst]` with `dst != src` on EVERY lane (a CoW record from the reserve); `[x, x]` on a multi-lane group only
    when no lane folds. Single-lane fires: in place stays in place.
16. **Snapshots per piece.** `RsSnapshot.pieces: Vec<u32>` (serde default; `slot` = first piece) — the runtime fills
    `pieces` when a record has more than one piece.
17. **Epochs.** Draining an evacuated segment and recycling freed slots are fenced by a per-fire epoch that advances
    on every fire (not only on copy-on-write / explicit writes).
18. **rs buffer is an rs-list shape (supersedes 8's "rs buffer inside the record" and 13 entirely; amends 11).**
    Facts may state two list shapes: the kv list and the rs list (`SlotKind::List` with its own tokens per slot T_rs).
    `RsVerb::Buffer { pages, base, folded, fold }`: `pages` are real rs-list slot refs (logical, through the
    ChunkTable) in token order starting at `base` (`base` aligned to T_rs); buffered row s lives in
    `pages[(s - base) / T_rs]` at row `(s - base) % T_rs`. No unfolded-row cap beyond what the runtime can allocate.
    Fixed records hold state only. Forks share rs-list slots by refcount; a write into a shared, partially filled tail
    slot is preceded by a slot copy (runtime asks via `copy_slots`, or the engine copies a `(src, dst)` pair stated in
    the fire). Contract 15 (group CoW of the state record when a lane folds) still applies to the state record.
19. **Sliding-window kv is a trimmed list, not a ring (supersedes every ring rule; amends 3, 4, 11, 12).**
    Facts state window lists as `SlotKind::List` shapes with a `window` (rows retained); one shape per distinct
    window, layers of that window layer-major inside the slot; T = the kv page size. `Lane.tables` for a window list
    has the same token-order length as the full table; slots the runtime has trimmed are ref 0 (null chunk) — today's
    window kernels already skip pages outside the window, so they read it unchanged. The runtime drops a slot from a
    working set once its last row is below `committed_end - window` (rows of a fire still being decided — spec drafts —
    are never counted as committed); freeing goes through refcount and the per-fire epoch (17). Prefill longer than the
    window simply holds more slots until the next trim. Forks / prefix-cache entries share window slots by refcount
    like the kv list; a write into a shared partially filled tail slot follows the kv list's tail rule. No ring cap,
    ring addressing, gcd form or `kv.window_copies` remain; a backend's ring path is deleted, not kept.

20. **rs-list mechanics (fills in 18).** `SlotShape.role: SlotRole` (serde default `State`); the rs list is
    `SlotKind::List { tokens: T_rs, rate: 1 }` with `role: RsBuffer` (refused on non-List). At most one kv list and
    one rs list per load. rs-list refs appear only in `RsVerb::Buffer.pages` (same `pages` on every lane of a group),
    not in `Lane.tables`. Tail copies: `Lane.slot_copies: Vec<(ShapeId, Vec<[u32; 2]>)>` (serde default), whole-slot
    `[src, dst]`, stated on the first writing lane of the group; the engine runs every lane's copies on the stream
    before any lane of the fire reads or writes (after earlier fires); no two copies share a dst; `pages` already
    names dst; `validate_chunks` checks them like table refs. A buffered write no longer forces record CoW. A model
    that buffers rs rows must state an rs-list shape.
21. **Window-list fields (fills in 19; final after WL2).** `SlotKind::List { tokens, rate, window: Option<u32> }`
    (serde default None); `window: Some(w)` retains w rows, role State, `tokens` = its own slot size T_w (<= the kv
    page; compiler picks a power of two >= 16 by waste). `ChunkFacts::validate` refuses two window lists with the
    same w, `Some(0)`, an RsBuffer with a window, and more than 4 window lists. `Lane.tables` holds the kv list's
    table and one table per window list, each in that list's own token order: entry i = slot holding rows
    `[i*T_w, (i+1)*T_w)`, ref 0 (null chunk) where trimmed or not written; `validate_chunks` accepts ref 0 only in a
    window list's table, never in `slot_copies`. Each list trims on its own: slot i dropped once
    `(i+1)*T_w - 1 < committed_end - w` (settled commits only; draft rows never count), freed through refcount and
    the per-fire epoch. Forks / prefix-cache checkpoints share window slots by refcount, list by list. A write into a
    shared window slot is preceded by a whole-slot copy in `Lane.slot_copies` under that list's shape id (degenerate
    pools: `KvCopy.windowed` pairs on the pre-launch copy). `KvDelta::window_copies` is removed.
22. **Host and disk tiers of shared shapes (fills in the tiers part of 12).** `ChunkFacts.host_chunks`: host chunks
    of `chunk_bytes`, ids from 0, none reserved, no ChunkTable (a host ref names its own place); any state shape
    (weights excepted) may open any host chunk; a host ref of shape s is `chunk * s.slots_per_chunk + index` at
    `SlotShape::tier_offset(chunk_bytes, ref)`, laid out exactly as the device slot. `ChunkFacts.disk_chunks`: chunks
    of kv-list slots; disk ref r = slot r of the engine's kv slot file; no other shape goes to disk;
    `disk_persistent` / `disk_kept` keep their meaning and the runtime journals disk refs as it journaled pages.
    `Engine::copy_slots(SlotCopy { shape, src, dst, slots, runs: [] })` moves whole slots of one shape device <->
    HostPinned, device <-> LocalDisk, HostPinned <-> LocalDisk (kv list only); device refs logical; token runs device
    only; a record moves as one copy per fixed shape, pieces paired in order. Ordering: a copy reads what every
    earlier fire wrote and lands before every later fire; copies run in submission order; the runtime may reuse a
    source / name a destination again once the call returns. The runtime parks only settled, private slots (sharing
    checkpoints move together); a parked slot keeps its holders; restore lands in fresh device slots named by the
    lane's tables / Lane.fixed. Byte reads of host refs (read_host_kv / read_host_state) are not stated. copy_kv /
    copy_state to/from host remain only for engines stating degenerate facts. TP: smallest host/disk chunk counts
    across ranks; each rank moves its band.

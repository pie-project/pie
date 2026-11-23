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
    `pieces` when a record has more than one piece. Amended (XLA, `sc/xla2`): `RsSnapshot.fixed: Vec<(ShapeId,
    Vec<u32>)>` (serde default) names the snapshot record as `Lane.fixed` names a lane's — every fixed shape's slot
    refs, a piece each; the runtime fills it from the record (`record_refs`). An engine whose record spans more than
    one fixed shape (XLA: one fixed shape a distinct state block size) reads it; `slot` / `pieces` still name the first
    shape's for the others.
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
19. **A windowed layer is state + buffer, exactly like a recurrent layer (user ruling: "a window model has exactly
    the same mental model as a recurrent model. Use the buffer. The window holds only folded rows."; supersedes the
    trimmed window lists of the earlier 19 and the ring shapes, ring addressing and gcd form of 3, 4, 11, 12).**
    Windowed layers are the sliding-window attention layers (Gemma, gpt-oss, and the window part of a windowed sink
    attention). Attention over rows chosen by score (index-selected or compressed attention) is no window: it stays kv.
    - **State.** Per windowed layer the lane's record holds a ring of W rows (that layer's K and V) holding only
      folded rows, row r at ring place `r mod W`, and, for a layer with S sink rows, those S rows beside it (folded
      once, never dropped). The ring and sinks are record bytes: they copy, park and restore with the record as
      opaque bytes and are costed in the record's tier. Their layout is the engine's (a plane of its own or inside
      the state bytes); the contract states only the bytes per windowed layer, at least (W + S) times that layer's
      K + V row bytes, padded as the engine pads them. W need not be a page multiple.
    - **Buffer.** The windowed layers' K/V rows `[folded, p)` live in window-buffer slots, a slot list like kv: taken
      on demand as rows accrue (no carried-rows bound; a fire too wide for the pool is cut like chunked prefill,
      never refused), named in the lane's tables in token order. A fold frees every buffer slot whose rows are all
      folded. On a hybrid the rs rows keep their own buffer (18, 20); window K/V use this list.
    - **Fold.** The lane's folded index (`KvDelta.folded`, the one `RsVerb::Buffer.folded` states) is the only
      source of truth. Advancing folded from f to f' writes buffer rows `[f, f')` into the ring at `row mod W`,
      dropping the oldest past W, in the same fold step as rs. Lanes sharing a folded window (a beam group: the
      shared common-ancestor lagged fold) fold once; 12 and 15 order a group's fold after every lane read.
    - **Read.** Row p of a windowed layer reads keys `[0, S)` and `[max(S, p - W + 1), p]` (`[max(0, p - W + 1), p]`
      for a plain window): those below folded from the ring and sinks, those in `[folded, p]` from the lane's
      buffer rows in lineage order (the `buffered_reads` lineage the rs replay uses; a lane never reads a sibling's
      divergent rows) and its own rows. One mask; one softmax over the union.
    - **Sharing.** A ring shared with a prefix-cache entry (or any other owner at a different fold point) is copied
      once at its first fold (copy-on-first-fold, bounded by the ring's bytes: the record's `cow != run`), so
      committed state stays immutable. An auto-checkpoint on a windowed model is a snapshot of the ring. A windowed
      lane's snapshot is its record plus its unfolded buffer slots, as an rs lane's; window tables are not saved.
    - **Prefill.** A long prefill streams through the ring in the same fire.
    - **Gone.** Window lists, their slot tables, ref-0 trimming, `slot_copies` of window lists, `KvCopy.windowed`
      and `PoolFacts.window_tokens` go. An engine states that it serves windowed layers as ring + buffer
      (`Capabilities.windowed_state`); until every engine does, the old window-list fields stay in the engine crate
      marked "removed by the windowed-state change; do not use in new code", and the integration commit removes
      them everywhere before anything lands (no dual paths in what lands).

20. **rs-list mechanics (fills in 18).** `SlotShape.role: SlotRole` (serde default `State`); the rs list is
    `SlotKind::List { tokens: T_rs, rate: 1 }` with `role: RsBuffer` (refused on non-List). At most one kv list and
    one rs list per load. rs-list refs appear only in `RsVerb::Buffer.pages` (same `pages` on every lane of a group),
    not in `Lane.tables`. Tail copies: `Lane.slot_copies: Vec<(ShapeId, Vec<[u32; 2]>)>` (serde default), whole-slot
    `[src, dst]`, stated on the first writing lane of the group; the engine runs every lane's copies on the stream
    before any lane of the fire reads or writes (after earlier fires); no two copies share a dst; `pages` already
    names dst; `validate_chunks` checks them like table refs. A buffered write no longer forces record CoW. A model
    that buffers rs rows must state an rs-list shape.
21. **Windowed-state fields (fills in 19).** `ChunkFacts.windowed: Vec<WindowedLayer { layer, window, sinks,
    bytes }>` (serde default; layers ascending and distinct, `window > 0`, `bytes > 0`), the rings and sinks a
    record holds; `ChunkFacts.record_bytes` counts them (validate refuses rings past it). The window buffer is
    `SlotKind::List { tokens: T, rate: 1, window: None }` with `role: SlotRole::WindowBuffer`: a slot holds T
    unfolded rows of every windowed layer's K and V, layer-major as the engine lays it. `ChunkFacts::validate`
    refuses a window buffer of another kind, two window buffers, a window buffer with no windowed layer and windowed
    layers with none, and any window list beside windowed layers. A lane states `Lane.window: Option<WindowFold {
    base, fold, lineage }>` (serde default None), on every seated lane of such a load and on no other:
    `base` is the row entry 0 of the lane's window-buffer table holds (a multiple of T, `<= kv.folded`), `fold` how
    far the fire advances folded (`FoldLen`, read as `RsVerb::Buffer`'s: a device-counted fold's count reaches the
    runtime when the fire settles, and the engine keeps the row its rings stand at with the record), `lineage`
    which buffered rows the windowed layers read. On a hybrid `RsVerb::Buffer` states the same folded, fold and
    lineage (`Lane::validate` refuses otherwise). The window-buffer table sits in `Lane.tables` under its shape id:
    entry i = slot of rows `[base + i*T, base + (i+1)*T)`, every entry a real slot (no ref 0). Forks / prefix-cache
    entries share window-buffer slots by refcount; a write into a shared partly filled slot is preceded by a
    whole-slot copy in `Lane.slot_copies` under the buffer's shape id (as the rs list's, 20). A copy-on-first-fold
    of a shared ring is the record's `[run, cow]` with `cow != run`. `Capabilities.windowed_state` (serde default
    false; `Serves.windowed_state`): a lane stating `window` to an engine without it is refused by name.
22. **Host and disk tiers of shared shapes (fills in the tiers part of 12).** `ChunkFacts.host_chunks`: host chunks
    of `chunk_bytes`, ids from 0, none reserved, no ChunkTable (a host ref names its own place); any state shape
    (weights excepted) may open any host chunk; a host ref of shape s is `chunk * s.slots_per_chunk + index` at
    `SlotShape::tier_offset(chunk_bytes, ref)`, laid out exactly as the device slot. `ChunkFacts.disk_chunks`: disk chunks,
    one id space beside the host's; the disk tier's byte budget covers every state shape, divided into chunks of
    `chunk_bytes` (each rounded up to the slot file's direct I/O boundary), ids from 0, none reserved, no table, any
    state shape (weights excepted) taking any chunk; a disk ref of shape s is `chunk * s.slots_per_chunk + index`,
    laid out in its disk chunk exactly as the device slot (as a host ref is). The engine states the division at load
    (budget, chunks, chunk bytes, each shape's slots a chunk); no per-shape constants.
    `disk_persistent` / `disk_kept` keep their meaning and the runtime journals disk refs as it journaled pages.
    `Engine::copy_slots(SlotCopy { shape, src, dst, slots, runs: [] })` moves whole slots of one shape device <->
    HostPinned, device <-> LocalDisk, HostPinned <-> LocalDisk, every state shape; device refs logical; token runs device
    only; a record moves as one copy per fixed shape, pieces paired in order. Ordering: a copy reads what every
    earlier fire wrote and lands before every later fire; copies run in submission order; the runtime may reuse a
    source / name a destination again once the call returns. The runtime parks only settled, private slots (sharing
    checkpoints move together); a parked slot keeps its holders; restore lands in fresh device slots named by the
    lane's tables / Lane.fixed. Byte reads of host refs (read_host_kv / read_host_state) are not stated. copy_kv /
    copy_state to/from host remain only for engines stating degenerate facts. TP: smallest host/disk chunk counts
    across ranks; each rank moves its band.

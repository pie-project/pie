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
23. **A rung's replay scratch is stated; one function checks a fire against it (replaces `fire::rung_rows`).**
    Each activation rung states what its replay scratch holds, from the engine's own reserve formula:
    `ChunkFacts.act_replay: Vec<RungReplay { own_max: Option<u32>, carried_max: Option<u32>, scratch:
    ScratchBytes { slot_bytes: Vec<u64>, lane_bytes, room_bytes, pad } }>`, one a rung of `act_chunks`, in its
    order; required (no default) of a load with an rs list (`ChunkFacts::validate` refuses one stating none by
    name). A fire of `rows` scratch rows over `lanes` lanes takes `Σ_slots roundup_pad(rows × slot_bytes) +
    roundup_pad(lanes × lane_bytes)` bytes (`ScratchBytes::bytes`), and fits where that is at most `room_bytes`;
    an engine laying no rs scratch past its rung's rows states no slot. A fire's rows are `ReplayRows { own,
    carried, scratch, lanes }`: `own` every row its lanes compute (never its lane count), `carried` each lane's
    by `fire::lane_carried` (a decode lane's none), `scratch` `fire::rs_scratch_rows` (own, and each lane's
    carried rows: on a compact fire `lane_carried`'s, else every one, a decode lane's included). Compactness is
    the fire's, decided only by `fire::scratch_compact(ChunkFacts.rs_compact, ChunkFacts.rs_scratch_per_fire,
    every_lane_one_run, any_lane_snapshots)`: the load's layout compact, and where compactness is a fire's
    (Metal's), every lane's buffered rows one stretch (`fire::lane_one_run(mask, folded, first, lineage)`, from
    its `RsVerb::Buffer`) and no lane landing an rs snapshot (an auto-checkpoint's included); no engine derives
    it otherwise. `fire::rung_holds(&RungReplay, ReplayRows)` returns the bound that binds (`ReplayPast { bound:
    Own | Carried | Scratch, rows, max }`). The runtime runs a fire at the smallest rung that holds it, each
    piece of a cut pass as cut (the uncut pass's own rows are no piece's), and refuses by name a piece no rung
    holds, before any step; each engine's submit check calls the same functions against the rung it stages in.
    Stated: Metal own T, carried T, its rs layout's slots and work term (`Seat::scratch_bytes`), pad 256,
    compactness a fire's; CUDA own T, carried its widest bucket's tokens, the extended run's row bytes over
    `rows_ext(max_tokens)` rows, compact (a decoding lane's rows never take it); engine-gpu and XLA own T, carried
    T, no slot (their replay lies in the rung's own rooms). `FireLimits.replayed_rows` stays the step's bound on
    every lane's replayed rows.
24. **The step compositions a load is sent are stated at load, one function computes them (`engine::shapes`).**
    The runtime states its step builder's rules in the load request (`LoadRequest.steps: StepRules { kinds:
    [Decode, Mixed, Prefill], treatments: [Folds, Buffers] }`, `StepRules::runtime()`); over the facts the load
    states of itself (`StepFacts::stated(caps, readouts)`: the rungs of `ChunkFacts.act_chunks` up to the
    first holding `max_tokens` (one fire's tokens where none is stated), `max_lanes`, `max_tokens`,
    `ReadoutFacts::fire_rows(max_readouts)`, whether a shape is the rs list, and the attachments it serves), `StepShapes::of(rules, facts)`
    is the finite set of compositions the builder can form: per rung, per kind (decode lanes only, prompt lanes
    only, both) and per treatment (every lane folding, or any lane buffering, a decode lane beside a prompt
    included; buffered only where the load has an rs list), a `StepShape { rung, kind, treatment, rows,
    decode_lanes, prompt_lanes, prompt_rows, lane_rows, readout_rows, attachments }` of inclusive spans. A decode
    lane takes one row, a prompt lane two or more; the prompt rows split across the prompt lanes in any way with
    each lane in `lane_rows` (uneven splits such as [10, 6] and odd lane counts occur at any rung: the builder
    joins whole members as they come); readout rows are the sequence lanes' (a kv-less lane's land through the
    readout landing). `attachments: Attachments { mask, adapter, inputs, snapshots, tap, float }` are what the
    shape's lanes may carry, in any subset on any lane (`Attachments::served(has_lora, inputs declared,
    rs_snapshots > 0)`: a mask, a tap and a float lane always); an engine whose bodies key on one records the
    shape with and without it, so a lattice built from plain requests alone is a gap. Lane-word classes are no
    part of a shape (an engine crosses its classes with each). A folded shape's
    rows run from past the rung below to its own; a buffered one's from one (its rung holds its carried rows,
    `fire::rung_holds`). The set is ordered smallest first across kinds: by rung, then decode, mixed, prompt, then
    folded before buffered.
    - **Where it is computed.** The engine computes it once its lattice is fixed and before its arm walk, walks it
      in its order, and states it with those it runs eager (`Capabilities.steps: StatedSteps { shapes, eager }`).
      The runtime computes it again from the stated facts at bootstrap and refuses a load whose stated set is
      another by name (`StatedSteps::agrees`). No setter: nothing changes the set after ready.
    - **The allowance split.** The runtime owns the set and its order; the engine owns its measured allowance.
      The engine records each shape's bodies in the set's order while they fit, and states the shapes whose bodies
      do not fit eager at load (`eager`, indices; under smallest-first the tail of the order). An eager shape still
      fires, uncaptured and tallied (`live::Tally::Eager`), never refused. A shape is left unstated by no engine.
    - **The builder forms nothing outside it.** `StepGroup::accepts` takes a member only where the step with it
      stays in the set (`StepShapes::shape_of`), else it goes in a later step. A member reaching the builder is
      one piece of its pass, already cut to a fire (`pipeline::fire::pass_pieces`): a prompt past every rung is
      cut and served, never refused. Admission refuses against the set only a piece no cut makes smaller and no
      shape holds (its sequence readouts past the rows one fire reads out, a buffered step on a load with no rs
      list, a lane attachment the load does not serve), naming what of the nearest shape it passes
      (`StepShape::misses`); a piece past one fire's tokens or lanes is refused, as before, by those bounds. Under the contract check (debug builds or
      `PIE_CHECK_REPLAY`, which gates both cross-checks: the engines' replay check and this one, `shapes::check_on`), every step formed is checked, a gap failing with the composition and
      the nearest stated shapes (`StepGap`).
    - **Engines, interim.** Each engine computes and states the set after its walk and says what it does with it:
      CUDA names each shape whose padded buckets its lattice walk records no body of that kind and treatment for
      (`Shell::walk_gaps`; today mixed and wide targets are folded only, so buffered mixed shapes are named); Metal,
      engine-gpu (Vulkan, wgpu), XLA log the set and that their walks do not key on it yet; Cerebras computes it and
      records no bodies. Each walk then takes its targets from the set in its own branch: CUDA records the buffered
      treatment of mixed / wide targets; Metal arms each readout-row tier inside a shape's `readout_rows` at its
      rung; engine-gpu records each shape before ready (sc/gpu-arm-before-ready); XLA lays its warm ladder from it
      (its laid mixed rungs, `Capabilities.mixed_lanes`, are narrower than the set's mixed shapes, so it states those
      it cannot lay eager).

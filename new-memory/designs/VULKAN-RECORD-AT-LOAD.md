# Vulkan: record at load

Status: design (GPU's design, checked against the code and measured on the
L40S). Branch `sc/l40s-vkrec-bodykey`, cut from `memory-redesign` 71ad83629.
Implementation is in steps (a) to (d) below, each one gated.

North star: nothing is built after ready. Native Vulkan records a fire's
command buffer at the first serving fire whose makeup matches no earlier
recording, and replays it after that. CUDA instead arms a bounded lattice of
bodies before ready (`engine-cuda serve/arming.rs`, keyed by
`record::BodyKey`). This note moves Vulkan onto the CUDA shape: a key that
does not depend on the fire's makeup, the makeup written as table data that
the recorded body reads, and the whole bounded set recorded before ready.

## 1. What makes a recording fire-specific today (checked)

`engine-vulkan serve.rs` `record_key` hashes:

| part | what varies per fire | where it reaches the commands |
|---|---|---|
| `descriptor.pack()` | lanes × rows of each class, the class runs | every walk: region launches, `FireDescriptor` rows |
| `slots` (`SlotTable`) | each value's rect `rows` / `bytes`, and `Scored` (capturing lanes) | grids and push words of every kernel. Offsets are fixed per rung: `arena.rs` `Span::slots` binds `base + rect.offset` |
| `rung` | the carve's place | already a function of rows |
| `caches` (`CacheTable`) | kv pool views. `page_indices.rows` is zeroed already (sc/l40s-vkrec-key) | sdpa / kv_append args |
| `bindings` (`FireBindings`) | token / position / readout-row tensors sized to the fire's rows. Geometry `indices.rows` is zeroed already | embed, rope, sdpa tables |
| `windows` | each region's runs, gathered windows | one launch per run (`walk_with`: `launches = Σ runs(region)`) |
| `arm` | the per-arm input store and readout seat | every input binding |
| per lane `readout` / `reads` / score room | which rows the host reads, the capturing lanes' room | the readout copies: `enqueue` records one `frame.copy` per readout span (per lane) into the body |

Fires that are never recorded: group masks, `rs_snaps`, split steps
(`SubmitSplitRows`, voxels), a streaming weights plane. Recordings are an
LRU of 64 (`SHAPES_KEPT`). Every `commit_segments` / `release_segments`
drops all of them (`B::forget`), and so does an expert-stage change.

The recurrent copies are a host loop: `dispatch/rs.rs` `rs_extend` /
`rs_land` issue one `copy_words` dispatch per lane and per gather run.

The kernels take the fire's rows in two places: the push words and the
grid. A throwaway census on a 2-layer model (scratch, `census.log`) shows
it for every decode-path family: `attn_kv_append` (groups[2]),
`sdpa_decode` / `split` / `fold` (groups[1], push row count), `norm_rms`,
`residual_add`, `sigmoid_mul`, `glu` (push element count = rows × width),
`rope` (groups[2]), `gather_embed` (groups[1]), `gather_rows` (push word 1).
GEMM picks its kernel by rows: gemv ↔ tiled, the tile's `bm` push word.

Measured on main (section 6): one 400-token decode recorded 45 times before
sc/l40s-vkrec-key, and 1 after it.

## 2. The key

```
VkBodyKey {
    bucket:   u32,         // the narrowest rung that holds the fire's rows (§3)
    classes:  Ladder,      // the classes present, in the class order, each at its rung
    islands:  Box<[u8]>,   // the regions the fire's windows run gathered / split, a bit each
    rs:       RsBody,      // Folds | Buffers | Replays, as CUDA's
    arm:      u8,          // the input store and readout seat the body binds
}
```

This is CUDA's `BodyKey` (`engine-cuda record.rs:434`) plus `arm`. Vulkan
binds each arm's inputs and readout seat directly, while CUDA's graphs read
fixed device addresses. `patch` (the tower axis) joins once Vulkan serves
towers recorded; today a patch fire is never recorded. Lanes, rows per lane,
slot refs, positions, page tables, readout rows and score rows are all data.

The key replaces `record_key`. `KeyBuilder` (the Debug-format hasher) goes,
and with it the per-fire hashing cost on a phone core (F23).

## 3. Bucket against carve: the rung ladder already has both

`Span::rung_of` (`arena.rs:298`) finds the narrowest rung that holds the
fire (`narrowest`), then returns `widest[narrowest]`: the widest rung whose
carve lies in the same stated chunks (`Span::widen`). On a device with room
(the L40S, Qwen3-0.6B) every rung widens to the top, which is why all of
today's recordings sit at one rung.

So two numbers must be kept apart:

- **bucket** = `rungs[narrowest].tokens`: the row bound the body's grids
  are sized to, and the GEMM pick. It goes in the key.
- **carve** = `widest[narrowest]`: where the slots lie. Since it is a
  function of `narrowest`, it adds nothing to the key.

The rungs are the powers of two from 1 up to the top
(`serve.rs` ~1020).

**Decode buckets follow the vector GEMM's row tile, not the rung ladder**
(step (a), measured). A decode body carries M (its lanes) as a push
constant: read from the fire block it cost about 1 µs a GEMV dispatch,
113 a fire (+110 µs at 4 rows on the L40S, per-node timestamps). Padding a
decode to the next power of two makes M the bucket's, but the vector GEMV
then computes the padding rows: 5 lanes padded to 8 ran −4.7% against the
tip (the load's estimate: gemv ×8 2290 µs at 5 rows, 2486 µs at 8). So the
decode buckets are every count of lanes up to the widest vector row tile
(the kernel table's last `kernels_slang::linear::gemm::VECTOR_ROWS`, 16
today), each body's M its live lanes and no padding compute, then each rung
past it up to the lane cap, where a decode pads to the narrowest rung that
holds it and the matmul's cost steps anyway (`Shell::decode_buckets`). A
decode of 9 lanes takes its own body; one of 17 takes 32. Prefill and mixed
buckets stay the rung ladder.

## 4. What goes into tables

There is one makeup table per arm, laid in the arm's input store next to the
tables already there (`inputs.rs`; the CSR tables are already laid at
capacity and padded flat: `csr` repeats the last entry, `rows_table` pads
with −1). It is written once per fire:

| field | size | read by |
|---|---|---|
| `rows` (live token rows), `lanes` | 2 words | every row-bounded kernel's early exit |
| per class run: first row, rows, first lane, lanes | 4 words × ladder length | region launches (each region launches once per class in the ladder; a 0-row run exits) |
| per value: live rows | 1 word × values with a row axis | kernels whose value is not the fire's token axis (readout planes, capture rows) |
| `qo_indptr` | lanes + 1, at the bucket | sdpa, the rs copy kernel, readout gather |
| readout rows | at the bucket | the readout gather (replaces the per-lane `frame.copy` list) |

Positions, rotations, request_of_token, mask words, page indptr / indices /
last-page lens and slot ids already lie in the arm's store. They are written
host-side today; under this design they are written with the makeup table.

**The write is in the step's own command stream** (GPU's review condition).
sc/gpu-serial-fixes adds `Frame::update_regions` (`vkCmdUpdateBuffer` under
one barrier pair, at most 64 KiB a command). The makeup table and the fire's
small tables go through it in `feed()`'s pre-fire frame, which is committed
before the body on the same queue. No host write lands in memory that a
run-ahead frame is reading. The arm is held until its landing as today, so
the write is per arm. Larger tables (page indices at high lane counts, past
64 KiB) are split by `update_regions` already.

Push constants keep only shape constants: widths, heads, head_dim, scale,
and the bucket's row bound. The descriptors are already rung-only on native
Vulkan: `encode.rs` binds each view from its offset to the slab's end
(`storage_view`: `remaining`, not the view's own bytes). So a binding
depends on (rung, value, arm, layer) only. sc/spark-wgpu-atload-2's
capacity-length views (wgpu binds the power of two over what a view holds,
then aims to bind to capacity) are the same property on wgpu. Native Vulkan
needs no change there. Two exceptions remain, both per-lane offsets:
`rs_copy` (`handles.cut` at a lane's own slot) and the score room (a pair's
place in the pool). Both become offsets read from the table: the rs kernel
over `qo_indptr`, and the score room bound at its seat's base with each
lane's row taken from `Scored::table` (already a staged table).

## 5. Kernels

The grid is sized to the bucket. Each kernel reads the live count from the
makeup table, and threads past it exit. The families: sdpa decode / split /
fold / prefill, pool, mla, kv_append, norm, hc, gemm / qmm / qmv (m), argmax,
gather / embed, conv1d / gdn / kda, moe route / select / combine. GEMM picks
its kernel by the bucket. The rs copy loop becomes one kernel over
`qo_indptr`. Regions launch once per class in the ladder.

Two notes from the code:

- **sdpa decode's split count is already the shape's** (sc/l40s-vkrec-key:
  `splits_for` from heads × rows, never the keys). At the bucket's rows it
  is a function of the key.
- **Side effects come first.** A padded row that ran unguarded would compute
  garbage in its own activation row, which nothing reads, for a row-local
  kernel (norm, rope, residual, glu, gemm). But for kv_append (it writes
  pages), the rs copies and stores (they write slots), sdpa (it reads a
  padded lane's page table), moe route / select (they count rows across the
  fire) and the readout gather, a padded row writes or counts. So step (a)
  guards those first. A row-local kernel's early exit only saves its
  padded rows' work, which at decode sizes is a GEMM's memory-bound weight
  read paid once whatever m is. It is guarded for tidiness and measured, not
  required for correctness.

## 6. Measurements (L40S, Qwen3-0.6B, native Vulkan)

Binaries: main 51d58d8c3 and memory-redesign 71ad83629, each with a
throwaway probe that prints every recording's key, launches, walk time and
makeup (`VKREC`; not committed). Every GPU process ran under
`flock /tmp/pie-gpu.lock`.

### 6.1 Distinct recordings made while serving

The runs go one after another in one server life: greedy (32 tokens), top-p
(32), beam 4 (16), then concurrent separate requests of 64 tokens each. The
whole sequence runs twice. "Recordings" means kept frames inserted, which is
also the recorder's `compiled` tally on memory-redesign.

| phase | main r1 | main r2 | memory-redesign r1 | memory-redesign r2 |
|---|---|---|---|---|
| greedy | 7 | 1 | 3 | 1 |
| top-p | 1 | 1 | 1 | 1 |
| beam 4 | 2 (fails: `EmbedTokens is not host-derivable`, a main bug) | 0 | 3 | 0 |
| conc8 | 7 | 1 | 3 | 1 |
| c9 | – | – | 2 | 1 |
| c16 | – | – | 4 | 1 |
| c17 | – | – | 3 | 1 |
| c64 | 13 | 1 | 10 | 1 |
| **total** | **30** | **4** (34) | **29** | **7** (36) |

All of them were recorded after ready, and the second round of the same
workload still records. Each is a new makeup: a prefill's lane split
(5 / 21 rows), a decode of 49, 51, 52, 57, 60, 62, 63 or 64 lanes as c64's
requests finish, and so on. On memory-redesign every one sits at the top
rung's carve (rung 10 of 1..1024) on arms 0 and 1. On main, a single-lane
decode recorded 6 times in its first 32 tokens: the arms, and the first
decodes' makeups. sc/l40s-vkrec-key already stopped re-recording on page
growth.

### 6.2 Load seconds

Boot to "Server ready", Qwen3-0.6B, L40S:

| | ready |
|---|---|
| main 51d58d8c3 | 16.8 / 17.2 / 17.9 s (three loads) |
| memory-redesign 71ad83629, first load on a fresh PIE_HOME | 47.2 s (it measures the roofline at load and saves it to the user profile: the user's ruling "b") |
| memory-redesign, later loads | 16.7 s |

### 6.3 Recording cost per body

The probe times each kept walk (encode only, from plan to the kept frame):
main median ~2.0 ms, range 1.2–6.8 ms over 34 bodies; memory-redesign median
~2.0 ms, range 1.2–5.5 ms over 36. The 202 launches of Qwen3-0.6B are the
same at every row count. On main the very first walk (487 ms) built
pipelines. memory-redesign builds them at load, so its first walk is 5.5 ms.
At load each body is also submitted once (§9). That costs one fire at its
bucket: ≤ 3.4 ms for decode buckets (below) and tens of ms for a 1024-row
prefill.

### 6.4 Decode step time against lane count (main, fire-exact)

One process with n lanes, greedy, 32-token prompts. Each value is the median
of 3 (t(144 tokens) − t(16 tokens)) / 128 steps:

| lanes | 1 | 2 | 4 | 8 | 9 | 16 | 17 | 32 | 33 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms / step | 3.40 | 4.54 | 27.65 | 32.13 | 32.06 | 41.32 | 41.82 | 58.63 | 60.26 | 93.12 |

The 2→4 jump is main's untuned bf16 GEMM leaving the vector kernel at 4 rows
on a derived profile (F25 notes the same `VECTOR_MAX_ROWS = 4` edge on the
Adreno). It bounds what padding can cost: a pad from 3 to 4 lanes on that
profile pays the tiled GEMM. The bound on a padded fire is the full step at
its bucket (padded rows doing real work): 9→16 at most +29%, 17→32 at most
+40%, 33→64 at most +54%. The measured cost (§6.5) is far below that bound,
because most of the step's growth with lanes is per-lane host work (guest
sampling, readout), which a padding lane does not have.

### 6.5 Padding to the bucket (prototype)

A local prototype of step (a)'s padding, not committed:

- `Shell::pad_bucket` gives the narrowest rung ≥ the lane count, on a
  recording backend with no recurrent state, window lists or expert stages.
- `api.rs` appends padding lanes to a one-row, one-class decode. Each copies
  the first lane (reads, word) over page 0, with no feed and no report.
- The padding rows' write entries are −1, and `kv_append` now skips them.
  mla and pool already did.
- The readout spans of padding lanes are copied but not read by the host.
- A replay hands back the fire's own readout layout, not the recording's.

It builds and checks clean (engine-gpu / engine-vulkan / engine-wgpu,
tests included). Its first run padded nothing: the eligibility test wrongly
required an empty `windows` list, and a lane's LIST table is carried there.
The fixed run (census + step times at 1 / 3 / 4 / 9 / 16 / 17 / 33 / 64, both
against memory-redesign) is queued on the GPU lock. Step (a) will be rebuilt
on the reworked fireblock head (§8a) rather than landed from this prototype,
so its numbers come with step (a)'s gate.

## 7. The recorded set and its bound

The set is every (bucket, ladder, islands, rs, arm) a fire of the load can
take, which is CUDA's `lattice_targets` enumeration (decode / prefill /
mixed / wide / tower / idle) on Vulkan's ladder:

- **buckets**: the rung ladder, the powers of two from 1 to the top rung.
  Qwen3-0.6B on the L40S: top 1024, so 11 buckets. A decode-only ladder
  is the decode buckets of §3 (every count of lanes up to the vector row
  tile, then the rungs), up to the most lanes a fire seats.
- **ladders**: the plain classes present, in order. Qwen3-0.6B serves text
  on two plain classes (prefill c0, decode c2) of its 24, so there are three
  ladders: decode alone, prefill alone, and prefill + decode. A beam's mask
  is per-lane data (the user's mask-fold ruling), so it adds no class.
  Adapted and drafting classes add their own ladders where a load has them,
  as CUDA's `Tiers::plain` does.
- **islands**: none on Qwen3-0.6B (no windows). A windowed model adds one
  body per admissibility pattern at a bucket, as CUDA's `with_islands`.
- **rs**: Folds only on a model without recurrent state. On a hybrid (Qwen3.5
  / Qwen3-Next), the rs rungs take Folds + Buffers + Replays: ×3 at those
  rungs, as CUDA's arming does.
- **arms**: the frames in flight. Qwen3-0.6B on the L40S runs 2 (the census
  shows arms 0 and 1).

Qwen3-0.6B: (11 decode + 11 prefill + 10 mixed) × 2 arms = **64 bodies**.
At ~2 ms to encode and one submit each (≤ 3.4 ms at decode buckets, up to
~10–30 ms at the widest prefill buckets), that is **~0.3–0.8 s at load**,
against a 16.7 s ready. A hybrid at the same ladder is ×2–3 at its rs rungs:
~130–190 bodies, ~1–2 s. The recorder holds them all. The fixed count
replaces `SHAPES_KEPT = 64`, which is exactly Qwen3-0.6B's bound, so an LRU
of 64 would evict at the first wider model.

Memory: a body keeps its command buffer and one descriptor set per dispatch
(Qwen3-0.6B: the naming walk states at most 805 dispatches a fire). 64
bodies are at most ~52k sets, about 50 descriptor pools of
`SETS_PER_POOL = 1024`. Their count and bytes go into
the load's books before ready.

## 8. Interaction with sc/gpu-serial-fixes and capacity binding

- serial-fixes (`7ca4bd46c`) returns a dropped frame's command buffer and
  descriptor pools to the device's spares. A kept frame is never dropped
  while its recording stands, so its command buffer and sets stay its own.
  The recorded set is fixed at load, so those pools are allocated once,
  before ready. The spares then only serve the per-step feed, guest and
  readout frames.
- serial-fixes (`d32816fc6`, `7f1f8b8d2`) adds `Frame::update_regions` and
  writes the guest's roots and the readout widen's config in the command
  stream. The makeup table uses the same mechanism (§4), so the step's
  frames are: the feed frame (makeup + tables, then the token feed copies),
  the replayed body, then the guest frame. All three are in queue order.
- The `Recorder` becomes a fixed map filled at load. `SHAPES_KEPT` and the
  LRU go. `get` returning `None` after ready is a counted bug (the tally's
  "should have had a body"), unless the rung is eager by statement (§9).
- `commit_segments` / `release_segments` drop every recording today. After
  ready that would force re-recording, which is compiled > 0. Native
  Vulkan's pool must be laid whole at load, as wgpu's and XLA's are (the
  user's ruling for wgpu: "Same as XLA"; yield a no-op, stated once).
  Otherwise the body must bind segments only through a table, never as
  bindings. Native Vulkan's kv views bind the segment slabs, so the first
  option is the one that fits. Open question 1.
- sc/spark-wgpu-atload-2's `walk_binds` moves the lane synthesis
  (`naming_lane`, `alone`, `seat_rows`) from the naming shell to the
  generic one and walks every class × lane count × rows on every arm into
  a bind-only frame. Step (c) is the same walk with a kept frame and
  `recorder.insert`. It lands after atload-2, reusing its synthesis.

## 8a. sc/l40s-vkrec-fireblock (bd2c21743), and how this note builds on it

What the fire block already is: per-recording chunks of host-mapped memory
(the BAR heap on the L40S). A walk lays in it each fired kernel's changing
scalars (`PIE_PARAMS_FIRED` / `PIE_FIRE(field)`: elementwise / GLU counts,
gather_rows' count, GEMM M, sdpa splits / rows, the mask stride, the readout
copier's region count) and every dispatch's group counts, indirect only
where a family's counts have varied. A *family* is the hash of a walk's
commands without the block: in effect (arm, rung, class subset, GEMM tactic).
A *fill* (block image, groups, readout layout) is kept per shape, the shape
being today's `record_key`. A replay refills the block from the fill's image
and submits.

How the parts of this note map onto it:

| this note | fireblock | left to do |
|---|---|---|
| the key: (bucket, ladder, islands, rs, arm) (§2) | the family is close: (arm, rung, class subset, tactic) by command hash | make the family's key the explicit body key, and its bucket the narrowest rung, not the widened carve (§3) |
| the makeup as data (§4) | the per-dispatch scalars and group counts are data in the block | the block's image is laid **by a walk of each new shape**. That is a host walk (~2 ms) after ready for every new makeup. No recording is counted, but a fire still waits on an encode, and the set of fills is unbounded, so it cannot be made at load |
| padding to the bucket (step (a) here) | – | with padding, a decode shape *is* its bucket: one fill per (arm, bucket), bounded, made at load. The block then holds what is constant per bucket, and the indirect group counts are needed only where padding leaves rows varying |
| per-class-run rows, qo_indptr, readout rows (§4) | the readout copier's regions are in the block (`copy_rows`, a table of rung tokens × layers) | a prefill's lane split (5 / 21 rows), qo_indptr and per-run rows change between makeups of one bucket. Kernels must read them from a makeup table laid per fire (not per walk), so one fill serves every makeup of a bucket |
| window offsets | not covered (F19 / F35: a mixed prefill + decode fire binds each window's views at an offset that moves with the classes before it; 6 of the remaining 10 mixed recordings) | kernels take the window's first row from the table (the per-class-run rows above), and the views bind at the run's carve base |
| class-ladder launches (§5) | not covered | a region launches once per class in the ladder; a 0-row run exits |
| rs as one kernel | not covered | one copy kernel over qo_indptr instead of `rs_extend` / `rs_land`'s per-lane loop |
| recording at load (§9) | not covered | walk each (bucket, ladder, rs, arm) once at load with padded synthetic lanes: one family recording and one fill each |

**GPU's write-path condition.** At bd2c21743 the block is host-mapped and
the host writes it (`FireBlock::refill`) before the submit. GPU's review
found that the refill waits on the previous fire's fence first, so
consecutive replays of one recording run at depth 1. The fireblock branch is
being reworked so that commands at the head of each step's submission write
the block (`vkCmdUpdateBuffer`, or a copy from that frame's own staging
slice laid at load), then a barrier, with no host wait. The rework also
drops the `vkAllocateMemory` churn and the per-step fence-watcher threads.
This note's makeup table goes through that same command-written path,
written in the step's submission ahead of the body. sc/gpu-serial-fixes'
`Frame::update_regions` is the same mechanism. Nothing in the makeup table
is ever written by the host into memory a run-ahead frame may read. Step
(a)'s code is based on the reworked fireblock head.

## 9. Recording at load (step c)

For each (bucket, ladder, rs, arm) of the bounded set, the load synthesizes
lanes with the generic `naming_lane` (as `walk_binds` does: zero tokens,
each class's mask / adapter / inputs). It stages them on the real shell and
walks them through `plan → submit → recorder.insert` into a kept frame,
which is submitted once so that the first replay is not the first
execution. That happens before "Server ready", after the pool is laid.

Eager by statement (the leader's records: the user said a very large,
rarely used rung may simply run eager). An engine states each rung armed or
eager at load, and that statement never changes after ready. A fire at an
eager rung runs a live encode, which is not a bug. The tally separates
"eager by statement" from "should have had a body". The runtime never cuts
a pass down to the widest armed rung because a wider one is eager. Where
the line lies is not yet the user's word. The proposal is structural: rungs
that fit beside state at rest are armed, and rungs reachable only by moving
state out are eager. It is never a size constant. Step (c) builds it as an
option behind that rule, off by default (record everything bounded),
pending the user's confirmation of the line.

## 9a. As built (sc/l40s-vkrec-bodykey)

- **Recorded, not run.** The load records each body into its kept frame and
  submits none of them: a recording's commands need no execution. The fire
  block is written by the head of the first serving fire, and a head is
  skipped only where the staged image's hash and bytes match, so a body
  that never ran replays correctly. Arming took Qwen3-8B 40.9 s when the
  bodies ran, and 1.5 to 2.9 s recorded only.
- **The allowance is host memory.** A body costs what its command buffer
  and descriptor sets take on the host (the driver holds nothing measurable
  on the device for them). Each body is measured as the growth of the
  process's resident memory across its recording. The allowance is the
  host's available memory less the host tier the pool laid. On a device
  that shares the host's memory it is what the device has left, so there
  is one allowance. A body past it runs eager, named at load. On the L40S
  the bodies take 0.75 to 1.5 GB and the dearest about 11 MB.
- **Fed decodes are bodies of their own.** Where the weight cache streams
  the token table by its rows, a decode fed its token on the device binds
  the fire's own row stage, not the arm's ring (`FireBindings::row_ring`).
  The load records every shape that carries decode lanes in that form too.
- **Fixed bodies fill without a walk.** A body whose commands carry their
  rows (the decode buckets, fires within the widest vector-GEMM row tile)
  lays one image for all its fires. Its body key is the record key without
  what the lanes read out, and it maps to the family. A fire of a known
  body key whose readout copies move the regions the recording's do fills
  from the body and its own readout layout (`Shell::readout_copies`). A
  prefill wider than the tile still walks once for its fill (FOLLOWUPS F53).
- **Buffering fires.** A fire whose lanes buffer their recurrent rows (a
  beam's, a draft's) records through the state+buffer form of the
  recurrent kernels (GPU's port, sc/gpu-beam-parents-6): one kernel over
  the lane's carried rows. It does not use a table of per-page copies, and
  the rs copy loop of §4 / §5 is not built here.

## 10. Steps and gates

(a) On the reworked fireblock head: padding to the bucket, the explicit
    body key in place of the shape hash for padded fires, and the makeup
    table (lanes, per-class-run rows, qo_indptr, readout rows) laid in the
    fire block by the step's own commands, with the decode families
    table-driven.
    The decode buckets' bodies are recorded at load in (a) (padded
    synthetic lanes, one per (bucket, arm)), the rest of (c) later.
    Gate: compiled = 0 after ready for decode-only serving (prefill and
    mixed still record at their first fire until (b) / (c)); greedy and
    beam token ids identical to the base; decode ABBA ≤ 2% at 1 / c9 / c16 / c17 / c64 (c9 and c17 are the
    lane counts just past a bucket boundary).
(b) Prefill families, the rs kernel, and ladder launches. Gate: the same,
    plus top-p, beam and mixed prefill+decode with no recording beyond one
    per (bucket, ladder, arm).
(c) Recording at load. Gate: **compiled = 0 after ready over greedy / top-p /
    beam / conc8** (and c64), with built_after_ready = 0. The load time is
    stated against main. It is not gated (user ruling: ready time is out of
    the benchmarks).
(d) The Adreno. Gate: decode ABBA ≤ 2% at 1 / c16 / c64 against main on the
    desktop (L40S) and on the Adreno, and c9 / c17 / c33 reported on both.

## 11. Open questions

1. **Segments after ready.** `commit_segments` / `release_segments` drop
   every recording. Under record-at-load that is compiled > 0 at the next
   fire. Proposal: native Vulkan lays its pool whole at load, as the user
   ruled for wgpu ("Same as XLA"), with yield a no-op stated once. The other
   way is to make the body bind segments only through a table, which the kv
   views' bindings rule out today. This needs the user's word, since it
   changes Vulkan's yield.
2. **The eager line.** The leader's records hold the user's word that a very
   large, rarely used rung may run eager. Where the line lies is not stated.
   The proposal is structural (armed if it fits beside state at rest, eager
   if reachable only by moving state out), never a size constant. The
   default here records every bounded body. Step (c) builds the eager
   statement behind that rule, off, until the user confirms the line.
3. **Padding cost.** Main's L40S profile is derived (untuned). Its step time
   jumps 6× from 2 to 4 lanes (the vector GEMM's row edge), so on that
   profile a 3-lane decode padded to 4 pays the tiled GEMM. The gate's
   c9 / c17 / c33 are measured in §6.5. A low-lane bucket edge that sits on
   a GEMM kernel edge is a profile fact (the `kernel.gemm` crossover), and
   it is visible in the profile, not tuned around in the ladder.
4. **The readout reads.** The content hash `record_key` keeps each lane's
   `reads` (the outputs its program reads on the device). The fixed key
   (§2) makes them data: each read span's offset and rows come from the
   makeup table, and the copies become one gather a plane. Until then a
   fire whose lanes read different outputs records once per mix. In step
   (a) a padding lane copies the first lane's reads, so padding never adds
   a mix.
5. **Beam on main fails on native Vulkan** (`EmbedTokens is not
   host-derivable: channel 1 has no host-known value`). memory-redesign
   serves it. This is noted for the main comparison, not chased here.
6. **Kernels gated by data or by a count.** GPU's outline reads a live
   count in every family. Step (a) uses the table data each kernel already
   reads: a padding row's write entry is −1, which kv_append now skips (mla
   and pool already did, as CUDA's padded rows do), and its page table is
   page 0, read, never written. It costs what §6.5 shows. Whether the
   row-local families also take a count to exit early is decided by that
   number, not up front.

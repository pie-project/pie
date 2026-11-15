# The binding form: params and offsets from a per-fire buffer

Design note, before any code. File:line is at memory-redesign 3e4e78300 (crates/ prefix
dropped). Numbers for Qwen3-0.6B are from the naming pass of sc/gpu-arming f53ebcb38 on
Vulkan, NVIDIA RTX PRO 4500 Blackwell (the tip has no enumerator). "Not read" means not
read, not a guess. Section 9 holds the rulings of 15 Nov; its file:line is at 6af426fb4
(between the two, of the crates cited, only engine-gpu api.rs, serve.rs and settle.rs
moved). Where a section below was open and is now ruled, it says so and points there.

## 0. What is baked today

A kernel call is `Encode::fire(fire, args)`: buffers in binding order, then the scalars of
its `PIE_PARAMS` block (kernels-slang/src/encode.rs:1-4, :93-94); `Fire.groups` is computed
on the host from the fire's rows (encode.rs:160-169).

- Vulkan (engine-vulkan/src/encode.rs:274-540): each buffer becomes a STORAGE_BUFFER
  descriptor `(slab, binding.offset(), range)` (:343-351), scalars become push constants
  (:356-358, :474), counts go to `cmd_dispatch` (:510), a barrier is decided from the spans
  `(buffer, offset, extent)` (:406-419). A recording is the whole primary command buffer
  resubmitted (device/ctx.rs:1726, :1827), so offsets, ranges, scalars, counts and barriers
  are all frozen in it, and the key must pin everything they depend on
  (engine-vulkan/src/serve.rs:372-398). A replay runs no walk at all (engine-gpu
  serve.rs:3898-3916).
- wgpu (engine-wgpu/src/encode.rs:279-558): views `(buffer, offset, size)` (:338-353), the
  scalars in a uniform at the slot after the buffers; a bind group is cached under
  `(pipeline, views, params bytes)` (device/pipelines.rs:162-164), cap 16384 (:174), its
  uniform in an arena chunk (:175, :259); a view whose offset is not a multiple of the
  storage alignment is copied into scratch and bound uncached (encode.rs:347-402, :481-520).
  Commands are encoded every fire (WebGPU command buffers are single use), so wgpu has
  nothing to replay: only bind groups and uniforms are the serving-time builds.
- What already has the target form: the chunk pool. Its segments are bound whole and a
  kernel adds `origin`/`seg_words` from params at the access (kernels/prim/pool.slang:3-21,
  :29-38, :189-205; src/tensor.rs:61-97).
- The fire-varying parts of a non-pool view are its size, rows x stride
  (engine-gpu inputs.rs:996, :1008), and a cut's offset, skip x stride
  (run.rs:396-415 -> handles.rs:129). Its base is fixed per rung (scratch.rs:41-50) or per
  load (weights). No kernel reads a buffer's length (0 uses of GetDimensions/arrayLength in
  kernels/), and kernels bound their rows from params (e.g. linear/gemm.slang write_out).

## 1. Params and offsets in a per-fire buffer

- One block buffer per fire arm. An arm is a frame in flight (engine-gpu settle.rs:60-74,
  serve.rs:1799; default 1, engine/src/runahead.rs:25); inputs and readout are already per
  arm.
- Site = (region, run of the region, call ordinal inside the launch, kernel alternative),
  plus the calls outside regions. The naming pass already walks this per rung
  (`Found {launch, whole, rest}`); slot(site) is a prefix sum laid at load.
- Block of a site: the kernel's scalars as today (they carry the row counts), then one
  element offset per buffer argument: rung base + cut skip. The sink can append the
  offsets itself, since it already resolves every handle to (slab, offset)
  (engine-vulkan encode.rs:325-351, engine-wgpu encode.rs:338-353): no change to the 149
  `.fire(` sites in 40 Rust files of kernels-slang.
- Size, derived at load: arms x sites x stride, stride = widest block rounded up to the
  adapter's min uniform offset alignment. Widest scalars today: 23 (92 bytes,
  conv_conv3d_bf16; regex count over the 530 compiled WGSL modules, vector fields not
  counted; `pipeline.push_bytes` is exact at load); most buffers a kernel binds: 16. So one
  block is under 256 bytes. 0.6B: 1064 sites (Asks.dispatches, pick alternates not in it)
  x 256 = 272 KiB an arm at WebGPU's default alignment.
- How a kernel finds its block: a uniform binding with a dynamic offset, on both targets.
  WGSL has no push constants, so the site index cannot be pushed there; on Vulkan a
  UNIFORM_BUFFER_DYNAMIC offset is a load constant in the recorded bind call, exactly as a
  pushed site index would be, and it keeps `ConstantBuffer<Params>` in the kernel, so
  `PIE_PARAMS` (kernels/prim/params.slang:7-11) becomes one form for both targets.
- Who writes it: not prepare as it stands. The scalars come out of the walk (Run ->
  kernels-slang host fn -> sink). Today's replay skips the walk. The new form needs a walk
  per fire over a sink that only writes blocks and counts. Its cost is NOT MEASURED: today
  a live walk with all Vulkan calls is 1.6-1.7 ms p50 and a replay 131 us (c1) / 243 us
  (c64), pie-m3 = 96902d1ab + prints. This is the first thing to measure (a probe sink,
  no design needed) because it sits on the c1 gate.

## 2. Indirect group counts

- Vulkan: `vkCmdDispatchIndirect` from a sibling per-arm buffer, 12 bytes a site, written
  by the same walk; the workgroup limit check (engine-gpu error.rs:488) moves to the
  writer. A site not taken (absent region, unused run, pick alternative) dispatches zero
  groups: 0.55-0.58 us each on this card (pie-m3 measurement).
- wgpu: not needed. Encoding is per fire anyway, so counts stay direct arguments
  (encode.rs:538, :543) and absent sites are simply not encoded.
  `dispatchWorkgroupsIndirect` exists in WebGPU; Chromium validates each call with an
  injected dispatch (from memory of Dawn, not read here, not measured).

## 3. One binding per buffer, offsets at the access

- After: a descriptor is `(slab, 0, whole)`; rung and cut offsets are in the block. The
  unaligned-offset case disappears on wgpu (offset is always 0).
- Storage buffers per stage: unchanged by this design. Params stay a uniform, the indirect
  buffer is not a binding, pool segments still cost 4 a pool. Today over the 530 WGSL
  modules: worst 16 (ssm_kda_committed_snap_bf16), attn_sdpa_split_lse 15; 167 modules bind
  more than 8, 153 more than 10. The adapter's own limit is requested
  (engine-wgpu device/ctx.rs:380) and each kernel is held to it by name
  (device/pipelines.rs:404-409, engine-gpu error.rs:507). Worst count among 0.6B's
  kernels: 14. Getting under 8 is its own piece, moved up (9b).
- Max binding size: on wgpu a slab is never larger than one binding
  (device/ctx.rs:498-502), so whole-slab binding always fits; a pool larger than one
  binding is the four segments, virtual index -> (segment, local) (pool.slang:6-14).
  Unchanged; the ceiling stays 4 x binding size. Vulkan: see 9, still open (i).
- Isolation (ruled, 9a). Guest stages keep ranged bindings until the lowering clamps
  every data-dependent index to its view's extent and a device-free test shows an
  out-of-range guest index cannot reach another lane's rows. Model kernels on whole-slab
  bindings rely on our own row bounds; one bounds check per view is compiled in debug
  builds under a test feature, not in release.
- Barriers: see 9, still open (ii).

## 4. What bounds the counts; "mix"

Today's key (engine-vulkan serve.rs:372-398):
- packed descriptor (:374; rows, bucket, class windows, every lane's rows,
  poem-exec fire/descriptor.rs:22-37): rows -> block; which regions run -> mix.
- arm (:375): stays.
- readout bytes, each lane's readout and reads (:376, :391-394): they fix the copy regions
  recorded at engine-gpu serve.rs:3955-3992. `vkCmdCopyBuffer` has no indirect form, so
  these leave the recording. Ruled (9.2): a live tail in the per-fire primary.
- slots, bindings, windows (:380, :385, :386): rows -> block; handles are per arm/rung.
- rung (:383): -> block (its offsets), once views are bound whole.
- caches (:384): pool geometry; fixed unless a commit changes segments (see 5).
- Fires not recorded at all today (serve.rs:173-176: custom masks, snapshots, no row ring):
  each needs its buffer laid at its bound at load. Not read further.

Mix, exactly: for each region, how many runs it launches in this fire (0..its run bound).
Rows go in class order, so this is a function of the set of classes with rows and of
whether a masked lane is among them, not of row counts.

0.6B: 15 rungs, 24 classes, 202 regions, 28 of them cuttable (one mask). Single-class
fires make 452-540 dispatches (8 distinct counts) and name 593-681 with pick alternates;
the witness mixed fire makes 568.

- wgpu bind groups = arms x (sites + alternates): about 1064 + (141 or more) an arm. If
  the rung stayed in the binding it would be x 15, i.e. over today's cap, as measured.
- Vulkan recordings, three grains (ruled: R, subject to the probe, 9.1):
  (W) one per arm holding every site: >= 584 zero dispatches on a plain decode, about
      0.33 ms of a 3.2 ms c1 fire. Fails +-1% by arithmetic.
  (C) one per arm x class set: exact, but up to 2^24 - 1 sets.
  (R) one per arm x region-run, executed in order by a per-fire primary that lists the
      regions present: count = arms x sum of run bounds (>= 202; exact sum not read). The
      primary is encoded per fire (no allocation, no pipeline, no descriptor).
  The number of distinct mixes of 0.6B: not read (needs the template's distinct masks; one
  device-free print in the naming pass gives it). The earlier "plain-mix within 2.3%"
  figure is grain C for single-class fires only (141 zero dispatches).

## 5. What is deleted

- Recorder eviction, SHAPES_KEPT (record.rs:74, :143-158): entirely.
- Recording at a serving fire (serve.rs:294-303): entirely.
- Forget on residency change (serve.rs:156-164): entirely, once segments are fixed at
  load (ruled, 9.3; no update-after-bind). Expert-miss cuts (record.rs:14-15): not read.
- wgpu bind cache, BIND_CACHE_CAP, UNIFORM_ARENA_CHUNK (device/pipelines.rs:162-175,
  :229-312): entirely. Segments are fixed at load (9.3), so no group is remade.
- Unaligned-cut path and SCRATCH_CHUNK (encode.rs:347-402, :481-520, device/ctx.rs:63):
  entirely for kernel views, provided no load-fixed slab needs an offset.
- UNIFORM_CHUNK (device/ctx.rs:61): entirely for kernels.
- Vulkan SETS_PER_POOL growth (device/ctx.rs:1281, :1398-1444): replaced by sets counted
  from sites.
- Guest widen's per-pass bind group (engine-wgpu guest/widen.rs:120-127) and regroup on
  rebind (guest/session.rs:238-241): not in this form; guest-span agent's piece.

## 6. kernels-slang

52 kernel files declare 147 `PIE_PARAMS` blocks, each a hand-written `struct Params` in
its own file (e.g. kernels/elemwise/norm.slang:27-31); 529 buffer bindings; 901
instantiations. Every block is a push constant on SPIR-V and a uniform on WGSL through one
macro. Moving the block: one macro edit plus the two sinks. Applying offsets: per kernel,
every buffer declaration (a view macro like `PIE_POOL_VIEW`, pool.slang:189-205, keeps
`x[i]` bodies as they are); whether the shared accessors (prim/bf16.slang,
prim/slot16.slang) can take the offset once for all: not read. Offsets in u32 words,
scaled by the element as pool.slang:13-14; sub-word cuts (engine-wgpu encode.rs:359): not
read.

## 7. The gate

c1 and c64 decode against main d5b8eff26 on Vulkan, paired rounds, zero throttled periods;
browser serving on wgpu; same fires bitwise (model_logits dumps identical before/after on
both targets); after serving: pipelines built, recordings made, bind groups made, device
objects allocated all zero by named counters.

## 8. Pieces in build order

1. Probe only (not committed), and nothing else of the form before its figures (9.1):
   host time of the per-fire walk over a sink that writes blocks and counts and makes no
   Vulkan call, plus executing about 200 secondaries from one primary, c1 and c64, against
   the 131/243 us replay. If c1 loses more than spread against main, the grain comes back.
2. Site table from the naming pass, landed with 3. Check: sites = Asks.dispatches on the
   three miniatures and 0.6B; every serving call maps to a slot.
3. Block per arm at load; `PIE_PARAMS` one uniform form; both sinks write blocks. Lands
   alone, no dual path (one macro, all kernels). The serving-time recorder keeps working
   by holding each recording's block image. Deletes UNIFORM_ARENA_CHUNK, UNIFORM_CHUNK,
   params bytes from the wgpu key. Check: bitwise; A/B.
4. Offsets into the block, views bound whole (52 kernel files); guest stages keep ranged
   bindings; the debug bounds check per view (9a). Lands alone. Deletes the unaligned
   path and SCRATCH_CHUNK. Check: bitwise; the STAGED counter gone.
5. Segments fixed at load, in their own set / bind group, one per pool; the segment count
   a load fact and kernels specialised to it (9.3, 9b: the 8-storage-buffer piece, moved
   up). Check: no buffer made after ready by a named counter; the count of modules past 8
   at the load's segment count.
6. wgpu bind groups at load, cache deleted. Check: groups made after ready = 0; browser.
7. Vulkan indirect counts, per-site sets at load. Lands alone. Check: A/B.
8. Vulkan secondaries at load, one per arm x region-run; the per-fire primary with the
   readout tail; recorder insert, eviction and forget deleted. Check:
   recorded-at-serving = 0; A/B.

## 9. Rulings (15 Nov, Pie Leader)

File:line here is at memory-redesign 6af426fb4.

1. Grain on Vulkan is (R). Secondaries per arm x region-run, all made at load and
   counted; a fire encodes a primary of begin, one execute per region-run present, end.
   That is encoding, of the same standing as wgpu's per-frame encoder: it records and
   allocates nothing new. The probe decides it (piece 1).
2. Readout copies are a live tail in that same per-fire primary. No gather kernel.
3. Segments are fixed at load. No allocation after load, taken strictly: a pool's segment
   buffers exist from load, a commit binds memory under standing buffers. Segments go in
   their own set / bind group, one per pool, not repeated in each site's set. No
   update-after-bind.

   What a commit does today, both targets (one shared path):
   - `Pool::commit_segments` (engine-gpu store/pool.rs:275-304) makes a NEW buffer for
     each segment not yet backed, `B::Buffer::zeroed` (:289). On Vulkan that is
     vkCreateBuffer + vkAllocateMemory + vkBindBufferMemory and a zero fill
     (engine-vulkan device/alloc.rs:86, :101, :124, :334-344); on wgpu,
     `device.create_buffer` (engine-wgpu device/alloc.rs:60). `release_segments` drops the
     buffer (pool.rs:309-313).
   - Around either, the shell drains, rewinds its handles and forgets every recording
     (engine-gpu serve.rs:1656-1675; the engine's entry, api.rs:562-567).
   - At load only the null chunk's segment and the span's are backed (serve.rs:1260); a
     segment not backed is bound absent (pool.rs:259-266). `growth()` answers TwoEnded or
     Bottom on both targets (pool.rs:175-181), never Fixed.
   - So a Vulkan commit is a violation of this ruling as the tree stands, and so is a
     wgpu commit at 6af426fb4: sc/wgpu-fixed-load (548ebee61: `growth()` answers Fixed,
     the pool backed to its last segment at load, a release refused by name) is not an
     ancestor of 6af426fb4. Its diff was read no further than that.
   - Not settled by reading, and it decides how Vulkan meets the ruling: a non-sparse
     VkBuffer takes memory once, cannot give it back, and may not be written into a
     descriptor set before it has memory (Vulkan valid usage, from memory; not read
     here). So 'memory under a standing buffer' at a commit, with sets made at load, is
     either sparse binding (vkQueueBindSparse; no use of it in the tree, support on the
     targets not read) or the pool backed whole at load as on wgpu (then a commit and a
     release move no device object at all). To be chosen before piece 5.

a. Isolation: as written into section 3.

b. The 8-storage-buffer limit moves up: piece 5, right after views are bound whole and
   before wgpu's bind groups at load. Shape: the segment count is a load fact,
   ceil(pool bytes / max binding size), and kernels are specialised to it, possible now
   that pipelines are built at load.

   A commit of this shape exists, unmerged: 863587d55 (a local branch of another
   worktree, on 451233bfd, not on origin). By its message (its diff not read): the pool's
   buffers are a kernel's last arguments, as many as the pool is cut into; SPIR-V binds
   one array of N buffers, N a specialization constant; a WGSL module's N declarations
   and accessor are filled at the load for its N.

   What is known of maxStorageBuffersPerShaderStage:
   - WebGPU's limits table gives it a default of 8; a device gets the default unless
     requiredLimits asks for more, and asking for more than the adapter states fails the
     request (WebGPU, 'Limits'). 8 is the one figure every implementation owes.
   - The tree asks for the adapter's own figure (engine-wgpu device/ctx.rs:380;
     kernels-slang/tools/wgsl-check.html:10-12) and logs it at boot
     (packages/javascript/server-web/rust/engine.rs:49). No file in the tree records a
     figure read from a real browser; the gate of c06b5a7a1 states adapters of 10 and 16
     of its own making.
   - Chrome on Mac and Chrome on Windows, the adapter's limit requested: not verified.

   Counts (compiled WGSL modules; a pool's segments are the `pie_seg0..3` bindings; 174
   of the 530 modules bind a pool, none more than one):

   | modules binding more than 8 | 4 segments | 2 | 1 |
   |---|---|---|---|
   | 0.6B's set (15 of its 17) | 4 | 3 | 3 |
   | all 530 | 167 | 153 | 91 |
   | most a module binds, all 530 | 16 | 14 | 13 |

   0.6B's set is the 17 kernels named on Vulkan (f53ebcb38). 15 have a WGSL module of
   the same key (a `.subgroup` pick read as the plain module where WGSL has none); the
   two gemm picks are SPIR-V only, and all six WGSL gemm modules bind 3. Past 8 at 4
   segments: attn_sdpa_split 14, attn_sdpa_decode 12, attn_sdpa_tiled 12,
   attn_score_capture 10. At 1 segment: split 11, decode 9, tiled 9. So specialising to
   the segment count does not by itself put 0.6B under 8: the three decode kernels bind
   8 to 10 buffers of their own, and those must come down too (which of them can share
   a buffer: not read). The set wgpu itself names for 0.6B: not read (the naming pass
   has not run that load on wgpu).

c. The form on the decode kernels, counted (15 Nov, later). Three local commits on
   863587d55 (branch sc/gpu-pool-segments, not on origin, held until the probe's
   figures; no GPU run of the form at a serving fire): the stub fix, a wasm32 lint fix,
   and the form on attn/sdpa (decode, split, tiled, selected, fold) and attn/kv_append.
   The fact to keep: a kernel in the form binds N pool buffers plus the distinct OTHER
   buffers its tensors lie in, and a decode's other tensors lie in one buffer (the
   fire's inputs: positions, page tables, masks), the split walk's in two (inputs and
   the load's scratch for its partials). Rows in and out are in the activation span,
   which is in the pool.

   Storage buffers a kernel binds in its stage, WGSL:

   | kernel | today, 4 fixed segments | per-tensor views at 1 / 2 / 4 segments | in the form at 1 / 2 / 4 |
   |---|---|---|---|
   | sdpa decode | 12 (8 own + 4) | 9 / 10 / 12 | 2 / 3 / 5 |
   | sdpa decode, lse or selected | 13 (9 + 4) | 10 / 11 / 13 | 2 / 3 / 5 |
   | sdpa tiled (lse) | 12 (13) | 9 / 10 / 12 (10 / 11 / 13) | 2 / 3 / 5 |
   | sdpa split | 14 (10 + 4) | 11 / 12 / 14 | 3 / 4 / 6 |
   | sdpa split, lse | 15 (11 + 4) | 12 / 13 / 15 | 3 / 4 / 6 |
   | sdpa fold (lse) | 3 (4), no pool | 3 (4) | 2, whatever the cut |
   | kv_append | 8 (4 + 4) | 5 / 6 / 8 | 2 / 3 / 5 |
   | score capture | 10 (6 + 4) | 7 / 8 / 10 | not moved (attn/score.slang) |

   What each column rests on:
   - "Today": reflection of the compiled WGSL modules (every `var<storage>` of each
     module), from a dump of the per-tensor form made before 863587d55.
   - "Per-tensor views at 1 / 2": arithmetic, the reflected own count plus N. No module
     was compiled at those counts.
   - "In the form": COUNTED BY READING engine-gpu dispatch/attn.rs and scratch.rs, not
     observed at a serving fire. Not traced to their buffer: lse_out, `selection` and the
     tiled walk's tables; their rows are given the decode's count and are the least
     certain. What reflection does show: every form module reflects with all its tensors
     and its pool in one buffer, and with a pool of three plus each tensor in its own
     buffer, so a module sets no floor above 1 and the count is the fire's distinct
     buffers. The six GPU runs of that branch bound the suite's own buffers, not the
     serving layout; the miniatures, which would confirm it, never ran.
   - If the reading is wrong a fire is refused, not miscounted: the wgpu pipeline build
     holds the buffers a fire reaches to the adapter's limit by name. The load-time hold
     uses a stated `DECODE_HOMES` (1 whole, 2 split), a layout fact written by hand, not
     derived: to be derived in the re-cut.

   Against WebGPU's 8, by these counts: the split fits up to a cut of 6 segments, the
   whole walk up to 7. The branch's tuning tests state adapters of 4 (refused by name),
   5, 6 and 8.

   What the re-cut owes against the rulings (after the probe): N a load fact, not a
   per-fire one; the cut derived from pool bytes / max binding size, not capped at a
   constant 4; pipelines built at load for the load's real N; segments standing from
   load; a gate. The array form on Vulkan needs shaderStorageBufferArrayDynamicIndexing
   and shaderStorageBufferArrayNonUniformIndexing.

d. Device features, read (15 Nov), one device only: "NVIDIA RTX PRO 4500 Blackwell",
   through the engine's own physical-device query (a filtered test, no model, no kernel):
   sparseBinding true; sparseResidencyBuffer true; sparseAddressSpaceSize 1 TiB;
   shaderStorageBufferArrayDynamicIndexing true;
   shaderStorageBufferArrayNonUniformIndexing true; the engine's queue family binds
   sparse memory. Not read: the sparse bind granularity (it needs the feature enabled at
   device creation; the probe prints it), and the cost of a sparse bind, which decides
   9.3. No AMD device was read: the MI300X host has no Vulkan stack (no ICD), so AMD
   hardware under Vulkan is untested. Chrome's maxStorageBuffersPerShaderStage: still
   not read.

Still open (filled when read):
- Exact sum of run bounds for 0.6B, and the number of distinct mixes: not read. Both are
  one print in the naming pass; the probe build prints them.
- (i) Vulkan slab past maxStorageBufferRange. Read: the load's ceiling is the least of
  maxStorageBufferRange, maxMemoryAllocationSize and maxBufferSize (engine-vulkan
  device/ctx.rs:586-592, :696-699); the recipe, the pool's segments and the weight slabs
  are sized under it (engine-gpu serve.rs:796, :1159, :1178; weights.rs:254). Buffer
  creation itself refuses nothing by it (device/alloc.rs:63-86), and the sink still
  carries the case: a view whose slab runs past the range binds the range from its
  offset, and one whose own extent is past it is refused (encode.rs:343, :611-630).
  Whether a scratch or input slab can be larger than the range: not read. If one can,
  that view keeps a load-fixed offset in its descriptor and only the cut goes to the
  block.
- (ii) Barriers at load. Today each dispatch asks whether its read and write spans
  overlap what the frame wrote or read since the last barrier (encode.rs:406-421,
  device/ctx.rs:1320-1335; past 96 spans a barrier is owed regardless, :1308, :1321),
  and the answer is frozen in the recording. Inside a secondary made at load the same
  question is asked of each site's rung-widest spans, which can only add barriers;
  whether it adds any: not read. What ran before a secondary is not known at load, so
  each secondary opens with a barrier unless the primary places one between executes
  only where a pair of region-runs owes it. The probe measures the first, the worse.

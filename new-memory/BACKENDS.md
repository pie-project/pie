# North star on EVERY backend (tracking matrix)

Goal: the north star (SPEC.md) holds on every backend — CUDA, ROCm, Metal, Vulkan, wgpu, XLA/TPU, Cerebras. A feature
done on one backend is not done until it is on all of them (or the backend's spec realization says why not, e.g.
"one pool fixed at load" on XLA). No backend is "later, optional".

Legend: ✅ verified on hardware · 🟡 in progress / compile-only · 🔴 not started · — not applicable by spec

| Feature | CUDA | ROCm | Metal | Vulkan | wgpu | XLA/TPU | Cerebras |
|---|---|---|---|---|---|---|---|
| Real ChunkFacts, slot-ref lanes, fixed pieces | ✅ M34 | ✅ M43 (MI300X) | ✅ M30 (dsv4 compressor fix M46) | ✅ M48 | ✅ M48 | 🟡 M63 (TPU check by team) | 🟡 M43 host pool (no HW) |
| Grow / shrink (segments, evacuation) | ✅ M34 | — flat by spec | ✅ M30 | ✅ M48 | ✅ M48 | — fixed at load | 🟡 M43 (host segments) |
| rs list + slot_copies | ✅ M34 | ✅ golden_buffered (MI300X) | ✅ M37 | ✅ M53 (rs tail fork, engine) | ✅ M53 | 🟡 M63 | 🟡 M43 (host sim) |
| Window lists (own T_w, trimmed) | ✅ tables per list M51 (T_w = page; 2-window untested) | ✅ goldens (one list) | ✅ up to 4 lists M50 | ✅ up to 4 lists, own T_w M53 | ✅ M53 | 🟡 M63 (one list, T_w = page) | 🟡 one list, T_w=page |
| Multi-piece records end to end | ✅ engine (L40S) · ✅ runtime boot (IT r5, 18 pieces) | ✅ records in pieces (chunk goldens); rows cut across pieces kernel-tested | ✅ M50 | ✅ M48 (4 pieces bitwise) | ✅ M48 | 🟡 M63 | 🟡 host sim |
| Weight streaming: PLE rows | ✅ M35/M36/M41 | ✅ M43 golden_rows | ✅ M50 | ✅ M53 (rows, grouped tables) | ✅ M53 | 🔴 design only | — weights on host |
| Weight streaming: experts | 🟡 dense bf16 M40; quantized next | ✅ dense M47; quantized after CUDA | ✅ dense M50; quantized next | ✅ VK3 dense + affine + MXFP4, every arm (engine tier deleted) | ✅ VK3 | 🔴 design only | — weights on host |
| Host/disk tiers of shared shapes (copy_slots) | ✅ engine probe M43 · ✅ e2e M52 | ✅ golden_offload 8/8 (MI300X) | ✅ M50 offload probe | ✅ M53 offload (dense, hybrid, window) | ✅ M53 | 🟡 M63 (CUDA PJRT bitwise; TPU by team) | 🟡 M44 (host sim) |
| Snapshots (rs_snapshots) | ✅ | ✅ | ✅ | ✅ VK3 (4 a lane, per piece; conv, GDN, KDA) | ✅ VK3 | 🔴 XLA | — |
| K2 (no kv spaces) | ✅ M39 | ✅ | ✅ M42 | compile | compile | compile | compile |

Cerebras realization (CB, `sc/cb-chunk-pool`): the column is the host-traced `Cerebras` engine. State lives in
one host chunk arena (shared, unpinned shapes: the kv list, one window list of the widest window with T_w = kv
page, the fixed record cut into equal word-range pieces, the rs list with T_rs = kv page); each fire stages the
slots its lanes name into fire-sized planes under local ids and writes back what it wrote, so the CSL kernels are
unchanged. Segments are host allocations (commit / release / evacuate are real). Weights are whole on the host
(uploaded per fire, or kept by a resident fabric server): there is no partial device copy to stream into, so no
weight shapes are stated and `weight_residency` / `weight_uses` take and report nothing. Host/disk tiers
(contract 22): host chunks are seated only where `host_kv_budget` is stated (the pool is host DRAM already),
allocated on first write, any shape; the kv slot file (`r0.kv`, kept with `disk_kv_persistent`) holds kv list
slots; `copy_slots` moves device <-> host <-> disk; the legacy `copy_kv` / `copy_state` refuse every host move
(this engine never served them). No hardware here: lib tests and a host simulation of staging and parking
against in-place slots; nothing ran on a CS system or the simulator.

The wafer `Spatial` engine keeps degenerate facts, because: (1) its kv pages are wafer-resident at fixed places
(pool page g on grid row g % P as block g / P; a lane reads at most E pages a row), so a page id is a physical
placement the program is built around: a kv-only shared pool (one kv-list shape, chunks of whole P-page rows,
`Growth::Fixed`, identity table, no evacuation) could be stated, nothing more; (2) recurrent state is not slot
memory there: one state is resident on the wafer and the others are rebuilt by replaying the tokens that made
them, so there is no record to cut into pieces, copy `[run, cow]` or park, and a shared pool stating no fixed
shape would stop the runtime seating state for the wafer's hybrid builds; (3) one-sequence builds keep kv as
replayed tokens, not pages; (4) the unmerged `cerebras-model-support` branch (the Spatial owner's) rewrites
`spatial/engine/{fire,mod,posted}.rs`, `paged.rs` and `wafer.rs` (+1149 / -62). Follow-up for that owner after
its merge: the kv-only shared pool of (1) for batched paged builds.

Vulkan / wgpu (VK2, `sc/vk2`): window lists are lists of their own (up to four, each its T_w <= the kv page and
its own page table; trimmed slots ref 0 = the null chunk), verified bitwise trimmed vs untrimmed with scattered and
evacuated slots on gemma4-e4b-mini-l1 / -l6 and a two-window variant. Snapshots per piece (contract 16): served since
VK3 (see below). On-device tests (`tests/*/tiers.rs`,
both engines): a lane parked on the host and back at decode 2 and its kv list through the slot file at decode 6 lands
the never-parked logits bit for bit (qwen3-micro-text, qwen35-tiny, gemma4-e4b-mini-l6 with a trimmed window list);
contract 15 (a group `[run, cow]`, the folding lane writing the copy, the buffering lane reading the run, the run
untouched) = two ungrouped lanes bit for bit; a fork's rs-list and window-list tail slot copied by `slot_copies`
lands its parent's row bit for bit. Each has a negative control that fails (restore skipped, cow unfilled, tail
not copied).

Vulkan / wgpu weight cache (VK2): row tables only. The tables a plain token-id gather alone reads (PLE) stay on
the host (the mapped artifact) and are stated as `WeightKind::Rows` shapes (ids from 8; tables the recipe groups
share an atom, a token id's row of each, so gemma4-e4b's per-layer tables are one shape, unlike CUDA's one table a
shape). `weight_residency` fills newly named slots from the host in batched uploads (one copy of merged regions a
segment), rewrites a u32 block -> pool-unit table; evacuation rewrites it. `gather_embed_tabled` (Slang, SPIR-V and
WGSL) reads a resident block from the pool and a block not resident from rows staged for that fire: the engine
flushes the frame, reads the ids back, copies the missing rows from the host into a host-visible stage both
backends can bind (wgpu: through the queue). Uses: one atomic a row read into a per-block counter (the group's
first table), `weight_uses` reads and zeroes them and reports `WeightUses.held`. Held fast path: with every block
resident no fire flushes or stages, and on Vulkan fires record again (a residency not wholly resident refuses
recording). Seat-addressed fires are refused while a weight slot lies in the seats' chunks (M41). Verified: a
gather copies, so all-resident (held, after an evacuation of every weight chunk), none resident and every other
block resident land the same bits, and all-resident = the build before the cache bit for bit (gemma4-e4b-mini-l1,
both engines; -l6 grouped tables within the build). Routed experts: see the next paragraph.

Vulkan / wgpu routed experts (VK3, `sc/vk3`): every routed bank streams an expert at a time. Every checkpoint plane the
residency census reads an expert at a time, read only by a routed select (bank or expert bias; dense bf16, affine
64/8, 64/4, {32,64,128}/2, MXFP4) or a routed bias sum, is a `WeightKind::Expert` shape from the recipe (ids after
the row shapes): an atom = one expert of one layer, its plane of every bank of the layer back to back (each from a
word) in one slot; the profile no longer cuts experts (`cut_experts: false`: the chunk is never smaller than an
expert, so a table entry names one place). The banks stay on the host (the mapped artifact); `weight_residency` fills
newly named slots in batched uploads and rewrites a u32 atom -> pool-unit table (evacuation rewrites it). Every
routed kernel on every arm (per-route select SPIR-V + WGSL subgroup, sorted tiles baseline + cooperative matrix,
bias sum) has a `PIE_TABLED` twin (`prim/experts.slang`) that picks the expert's segment once and reads the same
elements in the same order. Non-resident entry: STAGING, not a host-visible bank: before the first select of a layer
with a routed expert in no slot, the frame lands, the routes are read back, and each such expert is copied from the
mmap into a seat of a host-visible stage (Vulkan: host memory read in place; wgpu: written through the queue) the
kernels bind. Binding the whole bank host-visible would pin a second host copy of every expert (M49 problem 2) and
wgpu cannot bind mapped memory to a kernel at all. Uses: one atomic a route at the layer's first bank; `weight_uses`
reports them and `WeightUses.held`. Held fast path: every atom resident -> no flush, no stage (the stage buffer is
given back), Vulkan fires record again. M41: a seat-addressed fire is refused while an expert slot lies in the seats'
chunks. The engines' own expert tier is deleted (routed groups in `Plan::tiers`, `Seats` for routed banks,
`Run::staged`, the expert-bias seat kind); what remains of `experts.rs` is the host mapping, the n-gram `Gathered`
seats and the dense-plane pump. Verified (RTX PRO 4500): qwen3-30b-a3b-mini, glm45-air-mini (both engines) and
qwen36-35b-a3b-mini (Vulkan; wgpu does not serve it before or after) held = base (the build before, its own
expert tier, e2971507c) bit for bit, and none resident (every expert staged) / every other expert resident = held bit for bit
(`tests/*/experts.rs`); quantized schemes kernel-level: tabled = contiguous bit for bit for every scheme, a route at
a time and in sorted tiles, slots in two segments + a staged seat (`tests/*/linear/moe/table.rs`). No quantized MoE
mini serves end to end here (the bf16 fixture writer; gpt-oss traces only quantized).

Vulkan / wgpu snapshots (VK3, contract 16): both state `rs_snapshots = 4` where the trace holds state rows and every
op reading one is a causal conv, a gated delta or a KDA scan (else 0, and the runtime cuts the pass). A fire whose
lanes snapshot binds a table (a lane: 4 rows into it, 0 ends; then each snapshot's record as the pool's piece units,
`RsSnapshot.pieces` or `[slot]`; a seat's record for a seat-addressed lane) and its chunked and committed scans run a
`PIE_SNAPS` twin (`prim/pool.slang` `PIE_STATE_SNAPSHOTS`) that writes the state as of each row into the snapshot's
record as it walks (conv: the history rolled to that row; GDN register / memory, KDA: the head state, each lane its
own cells). Fires without snapshots run the kernels unchanged; a snapshot fire is not recorded (its table is the
shell's, rewritten after a drain). Refused: more than 4, a row not interior, a lane split across the fire's runs, a
lane not folding. Verified (`tests/*/snapshots.rs`, both engines, qwen35-tiny: GDN r128 + conv): 3 snapshots of a
150-row prompt = the pass cut at each row (record bytes) bit for bit, in one slot and in 4 pieces (64 KiB chunks),
alone (chunked arms) and beside a lane buffering its row (committed arms); the lane's own record and logits = the
pass without snapshots; a decode from each snapshot = the cut pass's decode. Kernel level (`tests/*/ssm`): GDN k 128
and k 32 (memory walk) and KDA chunked with a snapshot = the f64 reference. Not on device: KDA end to end (no KDA
miniature serves here).

Vulkan / wgpu PLE miss path (VK3): a row table's misses no longer flush when the gather reads the fire's own token
ids (every decode and prefill): the host knows them, so the rows are staged in the frame's order (Vulkan:
`vkCmdUpdateBuffer`, inline in the command buffer behind a compute->transfer barrier, so no fire in flight reads the
stage changed; wgpu: the queue write, ordered by WebGPU's queue timeline). Ids the device writes still flush, settle
the queue (`Frame::flush_settled`) and read back. Measured (release, RTX PRO 4500, gemma4-e4b-mini-l6, 4 lanes, 192
decode fires, best of 3; `rows::the_miss_path_costs_this_much_decode_throughput`, ignored): Vulkan all resident 199.6
/ half 174.3 / none 183.4 tok/s with the synchronous flush; 209.6 / 185.0 / 181.5 with the ordered stage. The flush
was ~1 point of the ~12 % gap; the rest is that a fire with misses is not recorded (the stage is per fire), so it is
re-encoded each fire. wgpu (no recording): 122.6 / 125.0 / 120.3, no gap. l36 Vulkan (sync path): 57.7 / 54.7 /
55.4. Next if it matters: stage through a per-fire binding a recording can replay (a ring of stages indexed by an
input), so misses keep recording.

Pre-existing parity gaps (not memory, but in scope for parity): ROCm synth codegen, ROCm graph replay with >= 2 MoE layers (AMD d); (concat_cols, FP8 m33 fixed M43);
sm_100 tcgen05 W4A16 arms read no expert table; Metal MLA prefill unsupported.

Owners: L40S (CUDA), AMD (ROCm), BM agent (Metal, Mac Studio), VK agent (Vulkan + wgpu), TPU session (XLA/TPU, with hardware; took over sc/xla2), TIER agent (host/disk tiers), CB agent (Cerebras, sc/cb-chunk-pool; no hardware — lib tests + host simulation, rest unverified).

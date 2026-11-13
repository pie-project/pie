# The weight cache on CUDA: what is built, and what a port needs

For whoever ports this to another backend (ROCm first). State after
`sc/l40s-wc-experts`. Everything below was run on one L40S (sm_89); nothing
was run on sm_90 or sm_100.

## What the engine does

A model's streamable weights are atoms of weight shapes the engine states in
`ChunkFacts::weights`. The runtime names the pool slot each resident atom
lies in (`Engine::weight_residency`) and reads how often each atom was read
(`Engine::weight_uses`). An atom with no slot is read from a pinned host copy
the device maps, in place.

Two kinds are built:

| kind | atom | which weights | kernel that reads the table |
| --- | --- | --- | --- |
| `Rows` | a block of a table's rows | a bf16 table read only by the plain `Gather::Embed`, alone in its recipe slot | `gather::embed_tabled` (`RowTable`) |
| `Expert` | one expert of one MoE layer, its plane of every bank of the layer | dense banks read only at the `bank` port of a plain `MoeSelect` (no bias, not quantized) | every dense select, through `ExpertTable` |
| `Expert` | the same, the layer's planes being each bank's codes and scales | MXFP4 banks read only by a plain quantized `MoeSelect` of bf16 activations (no bias, no activation quantization), off tcgen05 | the MXFP4 selects, through an `ExpertTable` of `Planes::Mxfp4` entries |

Everything else stays dense on the device and is not stated.

### Load

- `layout::streamed_experts(trace)` (the trace before the load's rewrites)
  and `layout::streamed_rows(trace)` (the trace after them: a gather a rewrite
  folded into another op reads its table whole) pick the params.
- `experts::Plan::host_rows` / `host_experts` move those params to the pinned
  host tier. The weight store then binds them as `WeightRow::Dense` at their
  mapped host address.
- `Layout::of` turns the recipe's weight slots into `RowsShape` /
  `ExpertShape` (ids from `layout::WEIGHTS`, rows first). An expert slot holds
  the layer's planes back to back in the recipe's param order; atom
  `layer * experts + expert`.
- `Pools::seat_rows` / `seat_experts` open a cache a shape: one device table
  of u64 addresses per table / per bank, a u32 hit counter per atom (a layer's
  reads are counted at its first bank), every entry the host copy's.
- The load's own arming fires read everything from the host copy.

### A fire

`FireTables::row_tables` / `expert_tables` carry each param's table.
`Run::row_table` and `Run::expert_bank` hand them to the dispatch. A table is
`located` only when the runtime holds the set (below) and every expert of the
bank is in a slot; otherwise the kernels crate takes its plain-load arm (the
per-route GEMV), whatever the entries say.

### Residency

`Pools::weight_residency`: for each shape, the atoms newly named a slot are
filled from the host copy (h2d on the stream, behind the fires it holds), then
every table is rewritten on the stream. `WeightResidency::staging` is not read:
a miss is read from host in place, not through a staging slot. Evacuation
restages the tables.

`WeightResidency::held` (new, serde default false; the runtime sets it when
the deployment does not stream) says the set is the load's for life:

1. a bank whose experts are all named becomes `located`;
2. the recorded bodies are dropped and the lattice armed again
   (`Bodies::forget_bodies`, `Shell::arm_bodies`), because a recorded body
   keeps the select it chose when the experts were on host;
3. once every atom of every shape is named, the pinned host copy is freed
   (`Tier::release_host`), provided the host image holds nothing else.

After 3 a residency that names a new slot for an atom is refused, and a
non-held residency after a held one is refused.

`Pools::weight_uses` reads and zeroes the counters and reports
`WeightUses::held` (resident and host atoms per shape).

### Packed banks (MXFP4, `sc/l40s-wc-quant-engine`)

- `layout::packed_planes(trace, codes)` names the planes an expert's table
  entry lists (`[codes, scales, none]`); `streamed_experts` returns the codes
  and the scales together, and nothing where `gemm_sm100::supported()` or
  under the `rocm` feature.
- `Ranking::caching` takes a cached packed bank's group off the sequence the
  plan cuts; each plane is then hosted or filed like a dense bank.
- One table a bank, at the codes' param: expert `e`'s entries at
  `table[e * 3 ..]`, the third 0. The scales have no table of their own.
- `Run::bank_table` hands it to `select_quant` and to the fused select-GLU
  (`routed_bank`); a bank with a table is never staged by the fire
  (`staged_experts`) and keeps no bf16 or fp8 copy.
- No MXFP4 kernel counts an expert's reads. While the set may change (the
  table is not `located`) the dispatch launches `linear::moe::note_routes`
  before the select, which adds the fire's routes to the bank's counters. A
  held set of MXFP4 banks records no use.

## What was measured

`golden_experts` (engine-cuda `tests/engine_cuda/golden_logits.rs`), each
miniature over the probe's 12-step script, against the base engine's hashes
(`new-memory/GOLDEN-LOGITS-BASE-L40S.txt`):

| miniature | cached and held | all on host | every other expert cached |
| --- | --- | --- | --- |
| nemotron3-nano-mini | bitwise base | most abs. diff 0.049 | 0.049 |
| minimax-m3-mini | bitwise base | 0.125 | 0.125 |
| mimo26-flash-mini | bitwise base | 0.051 | 0.051 |
| dots3-note-mini | bitwise base | 0.27 | 0.27 |
| dsv4-flash-mini | bitwise base | 0.61 | 0.61 |

| dsv4-flash-mini-bf16-mxfp4-kv-bf16 | bitwise the dense serve (`8b85965b6c94d3fa`; the base engine was not run on it) | 0.51 | 0.51 |

No step's top token changed in any case. Cached and held runs the kernels the
base engine runs (checked on Nemotron with `PIE_CUDA_TRACE_FIRES=1`: the same
set of kernels). The other two run the per-route GEMV for every route of a
bank, including the cached experts: a bank with one miss is all on the slow
arm, and so is a bank that becomes whole under a set that is not held.

Reading experts from host in place is slow: the five miniatures' probe with
its host and mixed cases took 35 minutes; Nemotron's three cases take a
minute and a half, DeepSeek V4's most of the rest. Not measured case by case.

`golden_rows` covers the row tables (host / cached / mixed, bitwise base).
The older probes load with the diagnostic `dense-weights` (no weight shape
stated), so they still compare the engine's other paths with base.

## Not built, and known gaps

- Quantized banks other than MXFP4 W4A16 (NVFP4, affine u4, block FP8,
  MXFP4 against quantized activations, biased selects): the dispatch passes
  `ExpertTable::RESIDENT` for all of them. They stay dense or on the engine's
  own tier.
- MXFP4 banks were run on sm_89 only. The Hopper wgmma arms take the same
  table argument and were not run.
- sm_100: the grouped tcgen05 GEMM for MXFP4 / affine-u4 / NVFP4 W4A16 reads
  no table (`select_fp4.rs`), and a tabled block-FP8 bank needs whole 128-row
  atom blocks an expert. Those banks must not be stated as weight shapes there
  until the arms take a table. No dense bf16 arm has that limit.
- The engine's own per-expert tier (`device_weight_budget` seating `resident`
  experts in a slab, `Tier::promote`) is still there. Expert shapes are stated
  only when that budget cuts no bank. Its removal is the next step.
- Until a held residency arrives, a load that will not stream runs on the
  host copy and its first-armed bodies; the re-arm at that residency costs a
  second arming.
- The pinned copy is freed only when the host image holds nothing but held
  weights. A load with other host planes (`engine.host_planes`, spilled dense
  planes) keeps it.
- Fills are synchronous and unbatched across shapes; staging slots are unused.

## For a port

What the other backend's kernels need:

1. An embedding gather that reads a row through a u64 block table and bumps a
   u32 counter per block (`kernels/gather/embed.cu`, `5bcc67884`; test
   `gather::embed_table`).
2. Dense routed selects that read an expert through a u64 table and bump a
   counter per expert, on every arm the backend would otherwise take for a
   contiguous bank (`ba9f6b9f2`, `97b89dcdf`; tests `linear::moe_table*`).
   Without that on the fast arms, a held set cannot be bitwise the dense bank.
3. Nothing else: the layout, the caches, the fills and the re-arm are engine
   code shared by both builds. `layout::streamed_rows` and
   `layout::streamed_experts` return nothing under the `rocm` feature today;
   lifting those two early returns turns the cache on.

Reference commits on new-memory: `5bcc67884` (row table kernel), `94759969e`
(row cache), `90af984b5` (`WeightUses::held`), and this branch's expert cache
commits.

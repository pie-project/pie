# Parity ledger: `csrc/src/loader/` → `src/loader/`

Every entity in the C++ loader is listed here as ported, dropped (with the
reason), or missing (with the blocker). Same rules as `PARITY.md`.

## Heap planning — `src/loader/heap.rs`

`loader/heap_layout.hpp` (192 lines), pure offset arithmetic.

| C++ | Rust | |
|---|---|---|
| `align_up(n, a)` | `align_up` | ported |
| `HeapPlan` (7 regions + intermediates) | `HeapPlan` | ported |
| `plan_heap(g, weights, max_ctx, …)` | `plan_heap(g, tuning, weights, HeapParams)` | ported |
| defaulted trailing parameters ×5 | `HeapParams` with `Default` | ported |

What changed and why:

* The five defaulted positional parameters became `HeapParams`. Two of the
  five were both `int` and adjacent (`state_dtype_bytes`,
  `act_dtype_bytes`); a call that swaps them compiles and allocates fp32
  state at bf16 width. Named fields cannot swap.
* `plan_heap` takes `&Tuning` because the scratch slot's width reaches
  `Tuning::moe_tile_rows` through `scratch_slot_elems`. The C++ read a
  process-global tuning singleton from inside the arithmetic, which is why
  its heap plan could not be tested against two devices in one process.
* The scratch-slot derivation this header USED to carry is deleted, not
  ported — that copy had drifted (slot sized 8320 elements where the binder
  laid rows 16384 apart; every row past the halfway point wrote into the
  next colour). `src/batch/sizing.rs` is the one derivation and
  `plan_heap` calls it. The C++ fixed this the same way; the ledger entry
  exists so nobody re-introduces a second copy "for layering reasons".

## Load-plan compilation — `src/loader/plan.rs`

`loader/load_plan.hpp` (160 lines). The C++ reached the Rust loader through
the C ABI; this port calls `model` (author registry) and `model-loader`
(plan compiler) in-process, so the wire structs disappear.

| C++ | Rust | |
|---|---|---|
| `kMetalTileMapMask` | `METAL_TILE_MAP_MASK` | ported; one-sidedness vs the loader's model pinned by test |
| `kMetalPreferredAlignment` / `kMetalMaxTileBytes` | `METAL_PREFERRED_ALIGNMENT` / `METAL_MAX_TILE_BYTES` | ported |
| `metal_device_target()` | `metal_storage_target()` | ported; states the fields the C ABI defaulted (fusion_mask 0, BF16 encode scratch, no native MXFP4) |
| `plan_ties_embeddings` | `plan_ties_embeddings` | ported, with the two-wrong-configs story |
| `descriptor_for_testing` | `descriptor_for_testing` + `TestFacts` | ported; the round-trip through `ModelFacts::from_descriptor` is pinned by test |
| `compile_load_plan` | `compile_load_plan` | ported; returns the author's resolved `Mxfp4MoePolicy` like the C ABI did |
| `Checkpoint::open` + handle lifetime | — | dropped: `parse_checkpoint_metadata` is called in-process; there is no handle to keep alive |
| `plan.verify_model(request)` | file stat loop | dropped in part, with a reason: the verifier existed to hold the MARSHALLED plan to a re-authored contract — marshalling and author determinism both in scope. In-process there is no marshalling, and a same-process re-author is a restatement, not a second opinion. What still checks something real survives: every file the plan declares is stat'ed against the snapshot |
| exceptions | `LoadPlanError` (5 named variants) | ported |

## The expert slab — `src/loader/slab.rs` (+ `model-loader/src/group_slot.rs`)

`loader/expert_slab.hpp` (197) and its dependency
`pie_loader/group_slot_index.hpp` (163), which had no Rust counterpart.

| C++ | Rust | |
|---|---|---|
| `pie_loader::GroupSlotIndex` | `model_loader::group_slot::GroupSlotIndex` | ported — into the LOADER crate, per the header's own argument: two backends deciding residency by two eviction rules is two ways for the same checkpoint to thrash |
| `kAbsent` sentinel / `int32_t` slot | `Option<u32>` | ported |
| all-slots-pinned `runtime_error` | `AllSlotsPinned` (typed) | ported |
| `SlabTensor` (suffix, band, layer pointers) | `SlabTensor<'a>` (byte slices) | ported |
| `ExpertSlab` ctor's thrown strings | `SlabError` (9 named variants) | ported |
| null-pointer layer check | `ShortBank` length check | ported, stronger: a slice carries its length, so the real precondition (`experts * band_bytes` per bank, `slots * band_bytes` per slab) is checked instead of just non-null |
| `ensure_resident` / `end_batch` / stats | same; `ensure_resident` is `unsafe fn` | ported — the GPU-quiescence contract the C++ carried in prose is a `# Safety` section, and out-of-grid (layer, expert) is a typed error because the expert id is the ROUTER's readback: data fails the fire, it does not crash the process |
| `slot_data` pointer accessor | `slab(t)` + `slot_offset(t, slot)` | ported: binding needs the region and the offset separately |

The module keeps the two arguments that justify the design: residency has
to be a wired region whose contents change (`requestResidency` wires every
page — 18.4 GB for a streamed Qwen3-30B-A3B against 1.5 GB at rest, and an
Apple GPU aborts rather than faults on a non-resident touch), and a slot is
every tensor of one expert or nothing (one `expert_ids` buffer indexes every
routed projection, so per-tensor slot numbers cannot exist).

## Transcode — dropped whole, with its receipts

`loader/transcode.hpp` (354) is not ported, because it was already a port:
its own header says the affine encoder "is mirrored in
`loader/src/testkit/host_executor.rs`" — today
`model_loader::executor::host` — and exists only because a C++ driver
could not run the Rust executor's loops over its own heap. The Rust
driver can: `execute_plan_into(plan, snapshot_dir, sink)` streams every
finalized tensor (TileMaps run, peak memory = one tensor's working set)
into a sink that writes heap regions.

Verified quirk-for-quirk against the C++ before dropping, because the
header's stories are load-bearing: `mlx_affine_group_params` in the Rust
executor starts `w_max` at ZERO, negates the scale unless the negative
extreme dominates, snaps the ENDPOINT rather than the scale
(`scale = edge / round(edge / scale)`), and rounds half AWAY FROM ZERO
(`f32::round`) — the one character that was an 8.2% disagreement with
`mx.quantize` on MXFP4-derived banks whose values sit on half-integers.
Codes are picked against the f32 parameters and stored as BF16, as MLX
does. The E2M1 nibble LUT and E8M0 block exponents live in
`decode_mxfp4_elements`.

What the C++ had that the executor does not: `parallel_ranges` threaded
the loops across cores, and transforms wrote DIRECTLY into the wired heap
(no per-tensor allocation). If load time ever regresses on big
checkpoints, the fix is threading inside the executor — where convert
also benefits — not a re-mirrored copy here; this ledger entry is the
argument against the second copy.

The claim is pinned by `every_transform_this_driver_claims_has_a_host_
implementation` in `src/loader/plan.rs`: the Metal tile-map mask is
inside the host executor's convert gate, so every transform a Metal plan
can carry runs there.

## Decode storage — `src/metal/storage.rs` (in progress)

The first arm of `heap_bind.cpp`: `stage_decode_storage` and the weight
staging. Implementation-first per the current directive; the bind pass
(`weight_binds`, `bind_decode_dag`) and the encode half follow.

| C++ | Rust | |
|---|---|---|
| `stage_decode_storage` | `stage_decode_storage` | ported: KV / GDN state / scratch / IO / argmax regions, geometry arithmetic identical |
| `stage_plan_weights` + `run_tile_map` (~900 lines) | `stage_plan_weights` (~40) | ported BY DELEGATION: `execute_plan` runs the plan — checkpoint reads and every TileMap — and the tensors pack 256-aligned into one region, sliced per name. The C++ transform loops were the mirror of that executor (see the transcode entry) |
| `alloc_zeroed(..., initial_commit)` elastic sizing | full-size allocations | deferred: a memory optimization, ledgered so it is not forgotten |
| `resolve_mappable` / zero-copy mapping / stream pack | — | deferred: every checkpoint loads correctly through the copy path, some resident-larger |
| `ExpertSlabRequest` staging arm | — | deferred: `ExpertSlab` exists; wiring needs the paging fire path |
| `weight_binds` (317) / `bind_decode_dag` | — | next: the per-kind weight-name table and the argument-table walk |

## Not yet started

| C++ | lines | blocker |
|---|---|---|
| `heap_bind.cpp` binding half | ~500 | `weight_binds` + `bind_decode_dag`; needs the argument-table surface |

# Storage Refactor & Metal: Review Follow-ups

Scope: commit `7ab03df3` (the storage-refact-and-metal.md implementation:
load-planner crate moved to `runtime/`, create/`load_model` boot split with
device facts, CUDA in-driver compile deleted, Metal ported to LoadPlan
execution). Method: 10 independent finder angles, every survivor adversarially
verified against the tree (CONFIRMED unless marked otherwise). Line numbers
refer to the tree at review time (2026-07-11).

> **Resolution:** Findings 1–8 are resolved in the current tree. Finding 9
> remains intentionally separate in the contention workstream.

The contention-workstream findings (uncommitted `store/` + scheduler diff) are
tracked separately; only one adjacent item (§9) appears here because it was
surfaced in the same review.

Overall: the architecture landed as planned — the plan-executor split, the
facts payload, and the deletion ledger all check out. Every defect below is in
the seams: boot-path error handling, fail-open decode defaults, and the
conformance authority of the host executor.

---

## P1: boot and load correctness

### 1. Unconditional compile hard-fails `.gguf` snapshots; I64 regression; parity gate absent
`worker/src/embedded_driver.rs:484` (accept), `runtime/load-planner/src/inproc.rs:204` (fail)

`validate_snapshot_dir` still accepts a `.gguf` FILE path and
`worker/src/weights.rs:23` resolves one as first-class input, but the
now-mandatory compile path has no GGUF branch: `compile_load_plan` calls
`parse_model_config`, which unconditionally does
`snapshot_dir.join("config.json")` — on a file path that read fails (ENOTDIR)
and, with the old warn-and-fallback deleted, every backend hard-fails boot.
`parse_gguf_checkpoint` IS wired (inproc.rs:122) but only inside
`parse_checkpoint_metadata`, which runs after `parse_model_config` succeeds;
`discover_gguf_file` also `read_dir`s, so it too fails on a file path.

Same mechanism, PLAUSIBLE trigger: `checkpoint_header.rs:76-81` rejects all
64-bit dtypes (`F64 | I64 | U64`) while the deleted C++ parse
(`safetensors.cpp` → `tensor.cpp:37`) accepted I64 — a checkpoint carrying any
I64 tensor that previously loaded via the in-driver compile is now a hard boot
error. No fixture demonstrates one for the supported families, so exposure is
unverified.

Meta-finding: the plan's §6.1 gate ("every checkpoint in the test/bench fleet
compiles through `parse_model_config` before the mask is removed") has no
artifact in the tree — only synthetic config-literal unit tests. The mask was
removed without the audit that was declared load-bearing.

Failure: `pie run` with `snapshot_dir=/models/foo.gguf`: boot aborts with
"cannot read /models/foo.gguf/config.json" on every driver, including dummy,
which never read the checkpoint before.

Fix: teach the compile entry to branch on GGUF before `parse_model_config`
(single-file-aware `discover_gguf_file`), or reject `.gguf` at
`validate_snapshot_dir` with an explicit "GGUF deferred" error until the
plan's GGUF phase lands. Either way, add the fleet-parity test the §6.1 gate
requires; decide I64 policy (map to a supported dtype or document the drop).

### 2. Metal load-time tile bound is not actually implemented
`driver/metal/src/loader/safetensors_view.cpp:196` (madvise), `runtime/load-planner/src/planner.rs:1655` (uncapped merge)

`copy_storage_bytes` tiles the memcpy but issues its single
`madvise(MADV_DONTNEED)` only AFTER the whole extent is copied, so the chunk
loop is functionally a plain memcpy and peak page-cache overhead is one
EXTENT, not one 64 MiB tile. Worse than first flagged: the compiler's
`try_merge_bulk_extent_write` merges source+dest-contiguous bulks with NO size
cap (the 256 MiB `max_slab_bytes` bounds only the SlabScatter pass, and no
pass splits), so if file order matches arena order a merged BulkExtentWrite
can span the checkpoint's whole contiguous run — GBs — before its one
madvise. This defeats §7.3's stated purpose (peak = heap + one tile on
16–32 GB machines). Minor: `sysconf(_SC_PAGESIZE)` is re-queried per extent.

Failure: loading a large model on a small-RAM Mac re-creates the mid-load
swap pressure the tile bound exists to prevent: transient memory = heap copy
+ largest merged extent instead of + 64 MiB.

Fix: madvise the consumed page-aligned source range inside the chunk loop;
hoist the sysconf. Optionally cap bulk merging at `max_tile_bytes` so the
bound holds by construction on every executor.

---

## P2: fail-open contract points

### 3. Shared LoadPlan decoder absorbs skew silently
`driver/common/include/pie_native/load_plan.hpp:561` (enums), `:229` (ids)

Two fail-open points in the JSON contract that replaced the deleted
1,061-line FFI layout tests:

- Unknown `target.mxfp4_moe` strings silently default to RoutedDecode and
  unknown `target.backend` to Unknown, while dtype/quant/tile/repack parsing
  all throw. The compiler-version check does NOT close this: the version is
  an automatic source hash, so a freshly built Rust runtime hands the C++
  parser its own new expected hash and the check passes while the parser
  absorbs the new enum string. Backend skew fails closed downstream (both
  drivers reject a non-matching backend), but an mxfp4_moe skew configures
  RoutedDequant lowering against layouts compiled for something else — wrong
  lowering with no error.
- `detail::id` accepts any JSON integer and converts via nlohmann
  `get<uint32_t>` — an unchecked static_cast. `-1` wraps to 4294967295 (the
  exact no-buffer/no-tensor sentinel the instruction views use) and `2^32+k`
  truncates to `k`, aliasing a live id. Non-colliding wraps fail closed at
  execution with a misleading error; colliding truncation misdirects a weight
  write silently.

Fix: throw on unknown enum strings (match the dtype parser); range-check ids
before narrowing. Both are decode-time one-liners that convert silent
mislowering into attributable parse errors.

### 4. Metal's first RETRY implementation classifies by error-message substring
`driver/metal/src/context.cpp:598`, messages at `driver/metal/src/pipeline/descriptor_resolve.hpp:86,109`

`lm.build_err.find("not ready")` decides RETRY vs member-FAILED. This is new
code (this commit is Metal's first runahead-RETRY support, closing
follow-up.md P1-9), and the magic string itself conflates the two causes it
should separate: `"... not ready (producing fire failed or not yet
produced)"` — a permanently failed producer is classified RETRY. Runtime-side
`permanent_retry_cause` catches only poisoned/closed channels in the fire's
own access set; a permanent cause outside it burns the full 1024-attempt
budget. Conversely, rewording any readiness message silently converts
transient stragglers into immediate member poison. The CUDA driver got a
typed `RetryableLaunchError` for the same classification in the same commit.

Fix: type the classification (mirror CUDA's exception, or a
transient/permanent flag on the descriptor-resolve result); split the
conflated message; add a test coupling the classifier to the producers.

---

## P2: conformance-executor divergences

The host executor is production code for the dummy driver and the S9
conformance reference; where it diverges from the C++ executors, the
conformance story inverts — wrong reference outputs get blessed.

### 5. Host executor semantics diverge from CUDA/Metal on three extent paths
`runtime/load-planner/src/host_executor.rs:212` (base_offset), `:277` (dest), `:330` (tiles)

- `read_extent`/`physical_bytes` ignore `source.stride.base_offset`; both C++
  executors read from `file_offset + base_offset`
  (`strided_copy.hpp:61`, `heap_bind.cpp:82`), and the same crate's
  `strided_physical_source_bytes` (planner.rs:2094) includes it. The
  dest side of the host executor DOES apply it, so the omission reads as
  accidental. Latent: the compiler currently always emits 0.
- `write_extent` stages the full physical dest span zero-filled and
  blanket-copies it, so a non-compact dest would zero the stride gaps,
  destroying interleaved data. CUDA and Metal both REJECT non-compact dests;
  the host executor silently corrupts. Latent: the compiler emits only
  compact dests today.
- The tile bound is computed and then dropped: `for _ in
  output.chunks(tile) {}` is a no-op loop and the write lands in one call; the
  input path also materializes a full `to_vec` copy. The Phase-1 "tile
  bounds" conformance the executor exists to prove is not exercised.

Fix: add `base_offset` at the read (reuse the compiler's helper); reject
non-compact dests exactly like C++ (fail loudly, don't corrupt); delete the
dead loop and actually chunk, or drop the pretense. These are cheap now and
land before the first compiler change that makes them reachable.

### 6. Hand-rolled float casts: truncation and a finite→NaN overflow
`runtime/load-planner/src/host_executor.rs:689` (f16), `:651` (bf16)

`f32_to_f16` truncates the mantissa in both the normal and subnormal paths
(no round-to-nearest-even), and its overflow branch maps large FINITE inputs
with nonzero mantissa bits (e.g. `100000.0`) to NaN instead of ±inf. The BF16
encode (`to_bits() >> 16`) is also truncating while CUDA's fp32→bf16 kernel
rounds — and bf16 casts ARE the common LoadPlan case, so that
divergence is reachable today, not latent. The `half` crate is already
version-locked in the workspace (transitive via ciborium).

Failure: dummy-vs-native parity on bf16/f16 models shows persistent small
diffs that get waved off as float noise (masking real regressions), and an
f32→f16 cast of a large finite weight loads NaN.

Fix: take `half` as a direct dependency for f16 (RNE, correct inf/NaN), and
make the bf16 encode round-to-nearest-even to match the CUDA kernel.

---

## P2: compiler design debt

### 7. The heap_bind_names port landed as a backend fork, not a schema
`runtime/load-planner/src/abi.rs:591` (fork), `runtime/load-planner/src/planner.rs:1729` (capability matrix)

`build()` branches on `backend == BackendKind::Metal`, detects Qwen3.5 by
sniffing checkpoint tensor names (U32 `lm_head.weight` + a
`model.language_model.layers.` tensor) instead of keying off
`cfg.model_type` via the arch_profile table one screen above, hard-errors
every other model_type on Metal, and `metal_qwen35_runtime_name` silently
drops unmatched tensors instead of declaring skips. The `.language_model.`
infix and tied-lm_head facts now live in two independently drifting places
(ArchProfile `source_prefix` vs the Metal literals).

Same altitude problem in the per-backend transform-capability match:
`validate_target_support` says Metal supports TileMap Cast/Reblock/Reorder,
but the Metal executor throws on EVERY TileMap (`heap_bind.cpp:237`) — and
`context.cpp` sets `load_attempted_` before `ensure_executor`, so the first
compile-clean-but-unexecutable plan bricks the driver (retry returns
CLOSED). `BackendKind::Unknown => true` while the host executor rejects
Repack/Decode/Encode/Transcode and silently PASSES THROUGH Reblock/Reorder
(wrong bytes that the conformance suite would bless). All triggers are latent
today (verified: no current Metal compile emits a TileMap; dummy GPT-OSS uses
push_direct), but every one arms on the next contract that needs an encoding
change.

Fix: key Metal contracts off `model_type` through the existing arch_profile
mechanism, with declared skip-lists instead of silent drops; replace the
backend-identity capability match with capability fields on
`StorageTarget`/facts that the executors also assert against, plus a
conformance test that compiler-validated ⊆ executor-supported per backend.

---

## P3: plausible, config-gated

### 8. Facts report `native_mxfp4_moe` without the Marlin build gate
`driver/cuda/src/context.cpp:456` vs `driver/cuda/src/model/loaded_model.cpp:157`

The facts payload computes `native_mxfp4_moe = dev_prop.major >= 10`, but the
executor still gates NativeGemm behind `PIE_CUDA_HAS_MARLIN` and throws on a
NativeGemm plan without it. The compiler's only guard (abi.rs:1661) reads
the same ungated fact. A Marlin-less CUDA build on Blackwell with a GPT-OSS
MXFP4 checkpoint and `mxfp4_moe="auto"` compiles NativeGemm and fails at
load, where the old in-driver compile silently fell back to RoutedDequant.
PLAUSIBLE, not CONFIRMED: Marlin defaults ON in CMake and no shipped CI
config disables it — this bites custom builds only.

Fix: facts must describe this build+device, not the device alone — include
the build gate in the reported fact (`#ifdef PIE_CUDA_HAS_MARLIN`).

### 9. (Adjacent, contention workstream) WS cursor hole survives pipeline death
`runtime/engine/src/store/kv/working_set.rs:158`

Noted here because the review surfaced it alongside the storage items; it
belongs to the uncommitted contention diff. The submitted-token cursor
advances permanently on successful submit; a dispatch-time
`LaunchPreparationError::Failed` then leaves reserved-but-unmapped tokens
with no repair path. The code documents this as intended pipeline poisoning,
but the WorkingSet RESOURCE outlives the pipeline: the same process can bind
the holed WS to a fresh pipeline (`claim_pipeline_scope` is pid-keyed) and
every subsequent prepare dies with an undiagnosable Fatal despite free pages.
The originally-claimed Drop-ordering mechanism was refuted (reserve→commit is
await-free); this sibling path is the real one.

Fix: invalidate or repair the WS mapping when a committed extent's fire fails
at dispatch (rewind on pass failure, or mark the WS poisoned so rebind errors
attributably).

---

## Appendix: claims investigated and REFUTED (do not re-flag)

- **Metal seeded-channel re-bind rewinds a live ring** — the engine cannot
  resend a consumed seed (`MissingSeed` aborts bind; `seed_taken` never
  resets), and CUDA's `seed_cell` writes the same words; a fresh cell gets a
  fresh endpoint.
- **TP group caps taken from the leader mis-size heterogeneous ranks** —
  `tp_min_plan` (memory_planner.cpp:903) barrier-min-reduces kv_pages/
  state_slots/max_forward_* across ranks before caps are reported; launch
  validation rejects out-of-range page ids loudly.
- **Device-geometry batch-wide RETRY kills co-batched RS fires** — prebuilt
  fires batch solo (`LaunchGrouping` rejects any co-member), so the batch-wide
  publish degenerates to the straggler's own cell.
- **Dummy GPT-OSS MXFP4 boot regression via Repack** — Repack is emitted only
  under `Mxfp4MoePolicy::NativeGemm`, which is doubly unreachable for dummy
  (facts report false; "native" is rejected pre-compile).

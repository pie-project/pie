# Weight loader — improvement & refactor plan

Consolidated plan covering **load performance** (Track A) and **architecture /
refactor** (Track B). Perf numbers are measured on a 4×B200 box loading GLM-5.1
(705 GB FP8 → 88 GB MXFP4 runtime quant); file:line refs point at the current
tree. Two independent reviews (the FP4 GLM-5.1 work and a fresh e2e loader audit)
arrived at the same perf diagnosis, so Track A is well-grounded.

---

## Baseline & diagnosis (measured)

`materialize ≈ 235–255 s/rank, total ≈ 390 s (was ~540 s before recent fixes).`

Materialize moves **192 GB FP8 H2D + 106 GB arena write per rank**. At PCIe-Gen5
×16 pinned (~50 GB/s) that's a **~4–6 s hardware floor** — we take 235 s, so this
is **~98 % overhead, not bandwidth.** RAM is 2 TB vs a 705 GB checkpoint, so the
whole thing is page-cached — **disk is not the bottleneck.** 58,596 projections
are each materialized as a serial `copy → FP8-dequant → bf16→MXFP4 encode` chain
on **stream 0** (~4 ms each, fully serial, zero overlap — the stream-0
serialization was added deliberately to fix a forward race).

Current shape: *recompute-everything-every-boot, per-tile-serial transform, then a
separate GPU-touching bind.* Four structural flaws: (1) pageable per-tile H2D, no
overlap; (2) a bf16 round-trip that triples HBM traffic (192→384→106 GB); (3) bind
does serial GPU work outside the parallel phase; (4) all of it re-runs every boot.

---

## North star

A loader that is a **pipelined producer of a cacheable materialized artifact**,
consumed **bind-free via a slot manifest**, with pinned H2D and no per-instruction
syncs. Four multiplicative cold-path wins → ~3× cold, 6–8× warm:

1. **Streaming pipeline** — reader threads → **pinned** staging ring → compute
   streams, event-driven double-buffered prefetch. Overlap turns Σ(stages) into
   max(stage); pinned fixes pageable H2D. 235 s → ~90 s.  *(Track A1+A2)*
2. **Fused FP8→MXFP4 kernel** — one kernel reads FP8 + block scale, writes MXFP4 +
   E8M0 directly; no bf16 intermediate. ~1.5× compute, half the HBM traffic +
   launches.  *(Track A2.2)*
3. **Bind-free / slot manifest** — contract emits *all* tensors incl. derived
   (`kv_b_proj` bf16); model binds by slot index, not name lookup + serial
   `materialise_bf16`. Post-materialize 150 s → ~20 s.  *(Track B1)*
4. **Materialized-artifact cache** — snapshot the 106 GB arena + manifest, keyed by
   checkpoint-hash + scheme + TP + ABI-version. Warm boot skips compile + quantize
   + bind. → ~50–70 s total, **6–8×**.  *(Track A3)*

---

## Validation reality

This box validates **compile + gemma dense bench (no-regress) + weight_loader unit
tests + cxx_compat**. The load-time wins on GLM-5.1 / Kimi / DeepSeek need the
**4×B200 box** — implement here, measure there. Dense-path correctness (alignment,
IR shape, bind) and the refactor items (B1/B4) **are** verifiable locally.

---

## Already landed

- Real `cudaMemcpyBatchAsync` in `flush_batched_copies` (rust_storage_executor.hpp:759).
- Persistent FP8 source-tile, BF16 scratch, scale-local, scale-cache buffers (≈ no
  per-tile `cudaMalloc` on the encode hot path).
- FP8 source + FP8 scale H2D routed to stream 0 (eliminates per-tile `flush_copy_streams`).
- `copy_storage_bytes_to_device_on(...)` helper for stream-specific H2D.
- MXFP4 scales kept as raw E8M0 U8 (no E8M0→F32 expansion) — removed ~30k
  post-materialize `cudaMemcpy`/`cudaMalloc` + ~14–20 GB/rank of redundant F32
  scales. See `attach_quant_metadata` `group_size==32` guard (hpp:2158).
- **Operand-unit alignment fix** — persistent buffers aligned to
  `PERSISTENT_OPERAND_ALIGNMENT=256` in `assign_persistent_offsets`
  (storage_compiler.rs); packed members stay tight *within* their backing buffer,
  so cuBLASLt picks `align8` sm_80 kernels instead of slow `align1` `s1688`.
  Recovered gemma 2925→3160 tok/s. (nemotron test asserts experts tight at 0/12.)

---

# TRACK A — Load performance

## A1 — Remove the obvious waste  *(each <50-line patch, low risk, correctness-preserving)*

### A1.1 Pinned host staging — ❌ TESTED, NOT A WIN, REVERTED  *(was billed Tier-1 #1, "≈30–60 s")*
**Measured on this box (gemma-4-E4B, 15.99 GB dense BF16, page-cache warm) and
reverted.** Flipping `pinned_staging_enabled()` default→ON:
- materialize 3133 ms → 3222 ms (**+3%, net negative**); `copy_flushes` 2 → 31;
  copy pipeline depth (`max_pending`) 512 → 68.
- **Pinned covered only 900 MiB of 15 251 MiB (6%).** The other 94% goes through
  `BulkExtentWrite` → `cudaMemcpyBatchAsync`, which **bypasses the pinned ring
  entirely** — so the flag barely touches the real load path, and where it does it
  only adds slot-busy `flush_copy_streams()` syncs.
- **Root cause the estimate missed:** the load runs at **~4.7 GB/s, far below PCIe
  BW → it is NOT H2D-bandwidth-bound.** Varying every H2D knob confirms it:
  serial/no-parallel = 3372 ms, no-batch = 3360 ms, 8 streams = 3490 ms, 16 streams
  = 3417 ms — **all within noise.** Neither pinning, stream-parallelism, nor
  batching moves the floor. The bottleneck is the **serial host-side read of the
  weight bytes (mmap→host)**, upstream of H2D, so no H2D-side lever can help.
- **Status:** reverted to opt-in (`PIE_CUDA_ENABLE_PINNED_WEIGHT_STAGING`). See the
  comment in `rust_storage_executor.hpp pinned_staging_enabled()`.

**Consequences for this plan (measurement supersedes the bravo estimates):**
- ❌ A1.2 (eager-flush removal), A1.5 stream/batch tuning — **no value on this path**;
  it is already not flush/parallelism/batch-bound (serial == parallel here).
- ✅ **DONE — parallel host-side reader.** The direct `cudaMemcpyBatchAsync` from
  mmap conflates page-faulting + bounce + DMA (~4.4 GiB/s). Replaced with lane
  threads that fault mmap→**pinned** in parallel (parallel faulting) and async-H2D
  from a pinned double-buffer (overlap). `flush_batched_copies` → `parallel_staged_flush`
  in `rust_storage_executor.hpp`. **Measured on gemma (16 GB): 3208 → ~2200 ms,
  ~1.45× (4.4→7.3 GiB/s), serves correctly.** Default 4 lanes
  (`PIE_CUDA_WEIGHT_READER_THREADS`, 0 disables; `_BUF_MB` tunes the 32 MiB buffers).
  4 lanes ≈ 8 (diminishing past 4 — now read-bandwidth/H2D bound, not fault-serial).
- ⚠️ The B200 GLM numbers may differ because materialize there also does FP8
  dequant/transcode *compute*; but a serial host-read floor would compound there too.

### A1.2 `flush_copy_streams()` fires before every TileMap/SlabScatter/Release/Attach (≈ 5–15 s)  *(#2)*
`rust_storage_executor.hpp:128–147`. For GLM-5.1 that's ~58k TileMap + ~30k Release
+ ~30k Attach, each doing `cudaStreamSynchronize` per copy stream (4–8). Even at
50 µs/sync → 5–10 s of pure waiting. Most TileMaps run on stream 0, so the flush is
a no-op anyway.
- **Fix:** record a `cudaEvent_t` per copy enqueue (or per stream); only the
  consuming instruction `cudaStreamWaitEvent`s. Add a `pending_copies_per_stream_`
  counter and make `flush_copy_streams()` a no-op when all-zero.

### A1.3 Kill per-call syncs
- **slab_scatter** `cudaStreamSynchronize` per call (≈ 1–3 s) — hpp:462. Drop it;
  rely on stream ordering / event-gate the next consumer.  *(#4)*
- **materialise_bf16** `cudaDeviceSynchronize` per layer (78×, ≈ 1–2 s) —
  glm5.cpp:89. Interim: `cudaStreamSynchronize(0)`, or launch all 78 then sync once.
  (Superseded by B1.)  *(#5)*
- **TP-sharded FP8 scale slicing** per-row D2D loop (≈ 0.3–1 s) — hpp:1389–1400.
  ~16 rows × ~30k weights = ~480k tiny async D2D. Replace with one
  `cudaMemcpy2DAsync` (src pitch = scale_cols×4, width = local_scale_cols×4).  *(#8)*

### A1.4 Scale-cache single arena (≈ 1 s + per-load `cudaMalloc` churn)  *(#6)*
`fp8_scale_cache_` destructor frees ~30k individual `cudaMalloc`'d entries
(hpp:72–81); `ensure_fp8_scale_loaded` (hpp:1276) does a `cudaMalloc` per weight.
- **Fix:** one contiguous arena sized to Σ scale bytes; one `cudaFree` on teardown.

### A1.5 Cheap cleanups
- **bind_glm5** per-expert string concat + `unordered_map::find` (~30k lookups,
  ≈ 0.3–1 s) — glm5.cpp:214–264. Cache layer prefix, `reserve`, or build an
  `experts[layer][e]` index. (Superseded by B1's slot binding.)  *(#7)*
- **Memory planner candidate search** — `plan_cuda_memory` (entry.cpp) sweeps many
  (N×R×page_size) combos; we set `memory_profile="balanced"` so `score_as_auto`
  work may be skippable. Verify it's not multi-second on glm_moe_dsa.  *(was B3)*
- **verbose `std::cerr`** per-rank duplication (≈ 0.3–0.8 s) — rate-limit ranks 1–3.  *(#9)*
- **`flush_batched_copies` chunk size = 1024** (hpp:736) — probe 256/2048/4096 on B200.  *(#10)*

## A2 — Overlap  *(medium; double-buffer plumbing)*

### A2.1 Look-ahead encode prefetcher  *(#3, the "+30 %" ask)*
`materialize_encode_input_bf16_rows` (hpp:1117) is fully sequential per tile: H2D →
dequant → encode → emit, so disk read for tile N+1 idles while N computes (~58k
tiles).
- **Fix:** walk the schedule one TileMap ahead; kick tile N+1's source onto a
  dedicated `prefetch_stream` into a double-buffered scratch; `cudaStreamWaitEvent`
  when N+1 fires. The storage compiler can annotate each TileMap with its next
  prefetchable source (ties into A3 IR bulk-ops below).

### A2.2 Fused FP8→MXFP4 transcode  *(north-star #2)* — ✅ FRAMEWORK DONE + parity-validated; executor wiring is the remaining step
The fused-`Transcode` IR is correct (frontend.rs:402), but the executor implements
it FP8 → **BF16 scratch** → MXFP4 — a BF16 HBM round-trip that triples traffic.
- **Built (composable, not a one-off):** `kernels/transcode.cuh` — a templated
  `transcode_rowmajor_kernel<Decode, Encode>` with `a` Decode functors + `B` Encode
  functors; the compiler emits the a×B specializations and the `float[32]`
  intermediate stays in **registers**, never HBM. Avoids the a×B hand-written-kernel
  trap. `DecodeFp8E4m3PerGroup` (rounds through BF16 to match existing numerics) +
  `EncodeMxfp4`; launch wrapper `launch_transcode_fp8_e4m3_to_mxfp4_per_group`
  (`transcode.cu`/`.hpp`), in the lib.
- **Validated:** `tests/test_transcode_fused.cu` asserts **bit-identical** output vs
  the two-step (`dequant_fp8...` + `quantize_bf16_to_mxfp4...`) — passes
  packed_mismatch=0 / scale_mismatch=0 across 4 shapes incl. non-multiple rows.
  `ctest -R transcode_fused`.
- **Executor wiring — DONE (B200 end-to-end owed).** `encode_tile_map` (the FP8→MXFP4
  path is an **Encode** TileMap, not Repack) now branches: when source is FP8_E4M3 and
  target is MXFP4 it calls `launch_fused_mxfp4_tile` → `transcode_fp8_tile_to_mxfp4`
  instead of `materialize_encode_input_bf16_rows` + `launch_encode_tile`. Refactored
  for single-source-of-truth: extracted `acquire_encode_source_tile` (FP8 tile on
  device) and `fp8_tile_scale` (scale + group_size + TP slice) from the existing
  dequant path, so the fused path passes **byte-identical args** to the parity-proven
  kernel → bit-identical by construction. Default on; opt out with
  `PIE_CUDA_DISABLE_FUSED_TRANSCODE`.
- **✅ VALIDATED on real GLM-5.1 layout (no B200 needed).** Pulled `zai-org/GLM-5.1-FP8`
  (real `glm_moe_dsa` config + tensor names/shapes/dtypes), reduced to `num_hidden_layers=4`
  + `n_routed_experts=8` (real shapes, only the index/layer-count trimmed), loaded with
  `runtime_quant=mxfp4`. The materialize transcoded **1690 real FP8 projections → MXFP4**.
  A/B via the artifact cache: **fused (default) vs `PIE_CUDA_DISABLE_FUSED_TRANSCODE=1`
  → byte-for-byte identical (sha256 match) on the 16 GB materialized arena.** So the
  executor integration is bit-exact on real GLM-5.1 experts, not just synthetically.
  (The `entry.cpp:314` forward-graph arch gate rejects `glm_moe_dsa` so it can't reach
  "serving", but materialize + cache run before it — which is exactly the A/B point.)
- Also re-confirmed on this real GLM load: the **parallel reader** engages on real FP8
  (`pinned_copies`, full bytes through the ring) and the **artifact cache** writes
  correctly (16 GB) and is deterministic (fused/unfused caches identical).
- gemma (dense BF16) still loads+serves unchanged. The BF16 scratch is still allocated
  (non-MXFP4 FP8 encode targets); harmless.
- **Extensible dispatch (refactored):** the launch is now enum-keyed —
  `launch_transcode(TranscodeSource, TranscodeTarget, TranscodeParams, stream)` with a
  two-level switch (one arm per source × one per target) that the compiler expands to
  the source×target kernel set; the kernel template is generalized on group width
  (`Encode::kGroup`, 32 for MXFP4 / 16 for NVFP4). `transcode_supported()` is the single
  source of truth for registered pairs. **Adding a pair = one functor + one switch arm
  + one `transcode_supported` line.** Proven: added a second pair, **`bf16→mxfp4`**, that
  way — parity test now validates both `fp8→mxfp4` and `bf16→mxfp4` bit-exact.
- **Scope note (re: "add popular paths"):** the framework's win is **quant→quant**
  (eliminating the BF16 HBM intermediate). `bf16→X` paths are already single-pass encode
  kernels (no intermediate to save) — they fit the dispatch for uniformity but gain
  nothing over the existing `quantize_bf16_to_*`. Per-channel INT8/FP8 (one scale per
  row) use a different per-row reduction, not the per-group kernel here. The genuinely
  valuable next pairs are quant→quant: **FP8→NVFP4, INT8→MXFP4** — each one functor +
  one arm. NVFP4 also needs its reference encoder to parity-test before shipping (don't
  add blind).

## A3 — Don't recompute every boot

### A3.1 ⭐ Materialized-artifact cache — ✅ IMPLEMENTED + validated on gemma (reload opt is follow-up)
Today only the **compile plan** is cached; the **materialized weights are recomputed
every boot** (~290 s of FP8→bf16→MXFP4), though materialization is deterministic.
- **Implemented:** `driver/cuda/src/loader/weight_artifact_cache.hpp` +
  `loaded_model.cpp` wiring. After materialize, snapshots every *owned* DeviceTensor's
  bytes + the full WeightStore manifest (TensorDecl specs, view (root,offset) pairs,
  quant_meta) to `<key>.weights`, keyed by the **authoritative compile cache key**
  (now surfaced as `RustLoaderCompileResult.cache_key`). Warm boot reloads straight
  into device memory and **skips compile-consumer + materialize**. Per-owned-blob
  FNV1a64 checksum; opt-in via `PIE_CUDA_WEIGHT_CACHE_DIR`.
- **Validated (gemma-4-E4B, this box):** cold = miss → materialize (3104 ms) → wrote
  15.25 GB cache → serving; warm = `cache hit (skipped materialize)`, checksum-verified
  bit-identical, finalize-validated, serving. The fallback path is proven too (a v1
  spec-serialization bug was caught by `validate_tensor_records()`, fell back to
  materialize, and served — so a bad cache never breaks loading).
- **Perf — measured + fixed (gemma):** first cut reloaded at ~712 MB/s (22 s).
  Per-phase timing (`PIE_WEIGHT_LOADER_PROFILE`) showed the **byte-wise FNV1a64
  checksum was 90%** (read 1.5 s / verify **19.8 s** / h2d 0.8 s). Replaced it with a
  4-lane word-wise hash → verify **0.66 s (~22.5 GiB/s, 30× faster)**; **verified
  reload is now 3.0 s (4.93 GiB/s)**, on par with `_NO_VERIFY`, so integrity is
  effectively free. gemma is still the worst case (3.0 s reload ≈ 3.1 s materialize,
  no transcode to skip). The win is the transcode-heavy target: GLM reloads ~88 GB
  MXFP4 verified at ~5 GiB/s ≈ **~22 s vs ~290 s materialize → ~13×; measure on B200.**
- **Follow-up (optional, diminishing returns):** reload is now read+H2D bound at
  ~5 GiB/s; overlap them (reader threads + pinned double-buffered async H2D) to push
  toward ~8–10 GiB/s. The cold *write* is still single-threaded (~30 s/15 GB) — same
  treatment. `PIE_CUDA_WEIGHT_CACHE_NO_VERIFY` skips the (now-cheap) checksum.
- **Risk / footprint:** **OFF by default** — strictly opt-in via
  `PIE_CUDA_WEIGHT_CACHE_DIR`; unset => zero disk, never reads/writes. Even when
  enabled the write **declines if free space < blob size + 256 MiB margin** (the
  artifact is the size of the materialized weights, ~88 GB/rank for GLM), so it can't
  fill the disk. Key reuses the authoritative compile key (scheme + TP + ABI). The
  FP8-MoE multi-owned-buffer + quant_meta path is implemented but B200-unvalidated
  (gemma exercises only the single-arena dense path).

## A-structural (longer-term)

- **IR bulk ops** — `storage_compiler.rs` emits ~441k per-instruction entries
  (`source_tensors=119028, contracts=177627, instrs=441637`); most are trivial.
  Collapse `Allocate→ExtentWrite→Finalize` → `MaterializeFromCheckpoint`, and runs
  of `Finalize` → `FinalizeRange`. ~3× fewer dispatches + makes A2.1 prefetch
  cleaner.  *(#12)*
- **`program_index_.instruction(instr_id)`** per loop step (hpp:102) — confirm it's
  a vector index, not a hashmap.  *(#13)*
- **cuFile / GDS direct I/O** — DMA NVMe→device, bypass the host bounce; ~30–50 %
  on storage-bound boxes. Needs aligned reads (safetensors offsets aren't always
  4 KiB aligned).  *(#11)*
- **NVFP4 port** — group_size=16 + FP8 E4M3 scales (vs MXFP4 g32 + E8M0); same
  footprint, better accuracy. Plug into `push_runtime_quant` once MXFP4 forward is
  verified.  *(#14)*

---

# TRACK B — Architecture / refactor

### B1 — Bind-free slot manifest  *(keystone)*
`bind_glm5` ran `materialise_bf16(kv_b_proj)` per-layer (glm5.cpp), an FP8→bf16
dequant *outside* the parallel materialize, **each followed by a full
`cudaDeviceSynchronize`** — ~one device barrier per hidden layer (tens of seconds
on GLM). Plus ~30k string-hash `WeightStore::find` lookups (sub-second, low value).
- **✅ Cheap interim DONE (safe, default):** removed the per-layer barrier from
  `materialise_bf16`; `bind_glm5` now syncs **once** after the layer loop (all
  dequants are stream-0-ordered, so this is correct by inspection). Compile-
  validated; GLM bind speedup to confirm on B200 (gemma doesn't exercise glm5.cpp).
- **Full keystone (B200, blind):** the compiler emits the derived bf16 (e.g.
  `…kv_b_proj.weight.bf16`) as a runtime-ABI Decode contract, produced *during*
  parallel materialize; bind just grabs the pointer (zero post-processing). This is
  the "direct file→tensor" goal — but it's GLM-specific, cross-language, and a wrong
  scale/shape/TP silently corrupts GLM with **no local validation path**. Recommend
  implementing it *with* B200 access (or flag-gated, default-off) rather than blind.
  The slot-index binding half is low-value churn (lookups are sub-second).

### B2 — Close the IR fusion gaps  *(helps inference, not load)*
`add_mla_fused_projection_joins` (abi.rs) matches only `kimi_k2|k25|deepseek_v2|v3`
— **DeepSeek-V4 and glm_moe_dsa are excluded**, so their q/kv-a and shared gate/up
aren't fused → more, smaller runtime GEMMs. The joins also force `dtype: BF16` even
where FP8 suffices.
- **Fix:** extend the detector to dsv4 + glm_moe_dsa; preserve FP8 where the kernel
  accepts it instead of forcing a bf16 dequant.

### B3 — Data-drive the ABI detectors — ✅ DONE
`abi.rs` had `matches!(model_type, …)` checks scattered across ~8 sites (detector
gating, source prefix, MLA fusion, TP shard quirks, runtime-quant eligibility).
- **Done:** added a declarative `ArchProfile` registry — `const ARCH_PROFILES`
  table mapping model_type → facts (`source_prefix`, `mla_fused_joins`,
  `phi3_fused_splits`, `nemotron_packed_experts`, `gpt_oss_mxfp4_groups`,
  `shard_embed_tokens`, `replicate_lm_head`, `mxfp4_runtime_quant`,
  `bf16_runtime_quant`) + `arch_profile()` lookup + a `profile()` accessor. Every
  scattered check now reads a profile flag; **detector transform logic is
  unchanged** (gating only), so behavior is preserved. Adding a model = one table
  row. Residual `model_type` checks: the registry and the `deepseek_v4` custom
  shard fn (one localized special; a leftover `shard_axis` debug print was
  removed). All 45 weight_loader tests
  pass (incl. the nemotron-packing / MLA / dense / gpt_oss detector tests, which
  prove the profile reproduces the gating).

### B4 — First-class Pack / allocation-unit invariant — ✅ DONE (checked contract)
The gemma regression came from an *implicit* invariant: "packed members are
internal to one backing buffer, so aligning the buffer base is safe" — previously
only a comment in `assign_persistent_offsets`.
- **Done:** `validate_persistent_layout(program)` runs on the final storage program
  and enforces, fail-fast: (1) every persistent operand buffer base is aligned to its
  contract (`>= PERSISTENT_OPERAND_ALIGNMENT`); (2) operand buffers are disjoint in
  the arena; (3) every `CreateView` reads a single existing backing buffer with the
  window *inside* it (packing stays intra-buffer). 4 unit tests (1 positive + 3
  negative) prove it's non-vacuous; all existing tests pass.
- Deliberately did **not** add a new IR node type — `CreateView` is already
  structurally single-backing, so a checked contract gives the robustness without the
  churn of restructuring `ir.rs`.

### B5 — Auto-version the compile cache — ✅ DONE
`rust_loader_compile_cache_key()` hashed inputs but not compiler logic; the manual
`cache-vN` string was a footgun (stale layouts served after a logic change).
- **Done:** `build.rs` content-hashes the loader's Rust source (`hash_sources`) →
  `cargo:rustc-env=PIE_WL_COMPILER_HASH` → FFI `pie_loader_compiler_version()` (a weak
  symbol, hand-declared in `rust_storage_program.hpp` since cbindgen here emits
  types-only) → folded weak-guarded into the cache key. Compiler-logic changes now
  auto-invalidate the on-disk cache; `cache-vN` demoted to a manual force-flush knob.
  All 45 weight_loader tests pass (incl. cxx_compat); driver builds + gemma serves.

### Forward-graph arch gate (`entry.cpp`) — ✅ FIXED
`entry.cpp`'s `supported` allowlist was stale: it rejected the bravo-merged MLA/MoE
archs (`glm_moe_dsa`, `deepseek_v4/v2/v3`, `kimi_k2`) **before `bind_cuda_model`**,
even though the downstream code + `bound_model.cpp` already handle them — so GLM/DSV4/
Kimi could load+materialize but never serve.
- **Done:** added those arch strings to the allowlist (matching `bound_model.cpp`'s
  dispatch) + a sync comment. They now reach bind + the forward path. Full serve
  end-to-end still owed (re-download; may surface a separate glm5 forward-graph issue).

### B6 — Split the monolithic executor — ✅ DONE
`rust_storage_executor.hpp` went 2389 → 675 lines; it is now just the materialize
VM (the `execute()` dispatch loop + allocate/arena + extent/bulk write + slab-
scatter + create-view + finalize + quant-metadata). The cohesive subsystems are
now their own headers, each a member the executor delegates to:
- **`loader/staged_h2d.hpp`** — `staged_pinned_h2d` + `PinnedLanePool`: the shared
  pinned-pipelined H2D engine (reused by the copy engine *and* the cache restore).
- **`loader/weight_copy_engine.hpp`** — `WeightCopyEngine`: copy streams + pinned
  staging slots + reader-lane pool + pending/batched queue. Narrow interface
  (loader + stats sink + raw dst; `acquire_stream()` for slab-scatter).
- **`loader/transcode_engine.hpp`** — `TranscodeEngine`: the TileMap quant path
  (Cast / Encode / Repack / Reblock + FP8 scratch). Deps injected by ref (loader,
  source names, copy engine, program index, `BufferResolver`).
- **shared primitives:** `dtype_map.hpp` (rust-enum → runtime-type mappers),
  `phase_timer.hpp`, `buffer_resolver.hpp` (buffer-id → DeviceTensor),
  `strided_copy.hpp` (non-compact strided H2D).
Each extraction was gated by build + parity (0 failures / 36 recipes × 3 modes) +
the 45 Rust loader tests, including the standalone `cuda_rust_executor_header_
compiles` check (which caught a real CUDA-guard bug during the work). The artifact
cache was also split into a loader codec (`weight_store_codec.hpp`, owns the
PIEWCAC3 format + a pinned/pipelined restore) and a driver-policy layer
(`model/weight_artifact_cache.hpp`).

### Smell cleanup refactor — ✅ DONE (Phases 1-4; SSOT theme)
Removed scattered hardcoding / duplication across the stack; every phase gated by
the parity suite staying **0 failures / 36 recipes × 3 modes** + 45 loader unit tests.
- **P1 correctness/dead code:** wrapped the unchecked `cudaMemcpy` in
  `convert_e8m0_scale_to_f32` in `CUDA_CHECK` + deduped its 2D/1D E8M0 expansion;
  deleted dead `lower_dense_copies`, a duplicate `o_proj` suffix, dead `inflight_bytes`/
  `pending_pinned_bytes_` members, a no-op try/catch, and a dead `program` param.
- **P2 registry dispatch:** added a `shard_axis_fn` field to `ArchProfile` + a
  `deepseek_v4` registry row, deleting the last `model_type == "..."` string-case
  (`shard_axis`). All arch dispatch is now table-driven — a new arch = one row.
- **P3 dedup + named config:** extracted `instr_by_id` (5 copy-paste sites → 1);
  bundled the 7 positional slab knobs into a `SlabConfig` struct; routed the 3
  `PIE_WEIGHT_LOADER_DEBUG` checks through one `wl_debug_enabled()`; new
  `loader/loader_config.hpp` centralizes the executor's ~12 `PIE_CUDA_*` env knobs
  (3 named convention helpers) + magic numbers (stream/slot/pool/tile/`kE8M0Bias`).
- **P4 test SSOT:** new `load_parity/dtypes.py` is the one dtype table; `gen.py`/
  `parse_cache.py`/`oracle.py` derive from it (was 4 hand-kept copies). `run.py`
  discovers the repo root instead of a hardcoded depth; the oracle reports the
  shard count so the runner drops its `_has_shard` re-derivation.
- **P3b (kernels):** C4 — the int narrows in `fp8_tile_scale`/`encode_tile_map`
  now go through a `checked_int()` that throws on >INT_MAX instead of silently
  truncating (a checked narrow is safer + less invasive than widening the whole
  path, which would re-narrow at the kernel boundary anyway). MXFP4 `32`/`16`/`2`
  literals are named: `EncodeMxfp4::{kGroup,kPackedPerByte,kBytesPerGroup}` +
  device `kE8M0Bias` in `transcode.cuh`, and `loader_config::{kMxfp4Group,
  kMxfp4PackedPerByte}` for the executor's shape math. Gated by
  `test_transcode_fused` (bit-identical, 0 mismatches) + the parity suite.

### E2E weight-loading parity harness — ✅ DONE (`driver/cuda/tests/load_parity/`)
Automated regression net for the C++ materialize path (previously only the manual
32 GB GLM A/B). Tiny synthetic checkpoints (numpy-written safetensors, no torch),
materialized by the real loader, dumped via the artifact cache (PIEWCAC3). The
harness is **composable + generative**, not a set of hand-written per-arch fixtures:
- **`spec.py`** — a module DSL. A model is composed from primitives (`mha`/`mla`/
  `fused_qkv` attention; `dense`/`fused_gate_up`/`moe_qwen`/`moe_mixtral` FFN; DSA
  indexer; block-FP8). `named_recipes()` expresses all 24 supported model_types as
  compositions; `random_recipe(rng)` generates random valid ones (random dims /
  counts / quant, `--seed` reproducible) tagged with a TP-enabled carrier
  model_type matching their structure (dense/MoE map to the GENERIC ArchProfile).
- **`oracle.py`** — a **generic** checker: derives the expected result from the
  source by byte-reconstruction — `direct` (==), `fusion` (ordered concat of the
  consumed source siblings), `split` (slice that must *tile* its source), and
  `skip-quant` (packed weight / derived scale). No hardcoded fusion/prefix map; for
  TP it reassembles shards on their axis then runs the same classifier.
- **`run.py`** — unified runner, `--mode differential|absolute|tp|all`, `--random N`.
  Differential (reader on/off, fused/unfused FP8→MXFP4) + absolute (tp=1 vs oracle)
  + tp (tp=2 reassemble→source) all run for named *and* random recipes. The loader's
  TP slicing is arch-agnostic but the engine only starts a TP group for a known
  model_type, so random recipes carry a TP-enabled carrier (dense→llama/qwen3,
  MoE→mixtral/qwen3_moe, MLA→deepseek_v3); the qwen3.5 named hybrids are tp-skipped
  (engine gates `linear_num_key_heads`). deepseek_v4 is flagged `[WARN: nothing
  sharded]` (HF-named recipe vs dsv4 native `.ffn.experts.w*`).
- Replaced the prior hand-written `templates.py` + 3 separate runners (which
  hardcoded a `FUSIONS` map, prefix list, dim constants and a fixed seed). The
  generic oracle now also *checks* phi3's `split` projections and MLA `q_kv_a`
  fusion under TP (previously skipped). Quant output stays differential +
  `test_transcode_fused.cu`.

---

## Suggested order

1. **A1.1 pinned default-on** — biggest single win, no API change.
2. **A1.2 drop eager `flush_copy_streams`** (per-stream events).
3. **A1.3 / A1.4 sync + alloc cleanups** — correctness-preserving.
4. **A3.1 artifact cache** — unlocks fast iteration for everything after it.
5. **B1 bind-free manifest** — keystone refactor; retire `materialise_bf16` /
   string-lookup bind.
6. **A2.1 / A2.2 overlap + fused kernel** — measure against the clean A1 baseline.
7. **B2–B6** — fusion gaps, data-driven ABI, first-class packing, cache versioning,
   header split (independent, schedule opportunistically).

Items 1–3 are < 50-line patches in `rust_storage_executor.hpp`. Land them and
re-measure on the B200 box before the larger A2/A3/B1 diffs.

---

## Transcode / quantize-on-load perf — empirical scoping

**Landed:** encode-source pooling + stream-0 ordering for the non-FP8
(BF16/FP16/FP32) compact `Encode` source in `transcode_engine.hpp`
(`acquire_encode_source_tile`). Mirrors the FP8 path. Both a perf win (BF16→MXFP4
951→574 ms; copy_flushes 1388→2) **and a silent-corruption fix** — the old
round-robin BF16 source had no sync before the stream-0 encode, so it encoded
*unwritten (zero) buffers*. Untested path (parity only covers FP8→MXFP4), so it
shipped broken. Proven via byte-diff vs old binary + MXFP4 decode (corr 0.9954 vs
source; old = all-zero, corr 0). Parity 0/24×3.

Measured floor (synthetic glm_moe_dsa, L40, page-cache warm):
- FP8→MXFP4 fused (glm/deepseek path), 1.64 GB: transfer=33 transform=205 (294 ms).
- same model RAW (4-lane, no encode): transfer=111 (162 ms). ← the bandwidth floor.
- BF16→MXFP4, 3.25 GB: 432 ms; RAW 187 ms.

Bottleneck decomposition (post-fix):
- transform ≈ expert source H2D (**single-thread ~10 GiB/s** via `queue_on_stream(0)`
  — driver pageable staging from mmap, confirmed by direct_mmap bench 10.3 GiB/s)
  + fused/encode kernels (compute-bound).
- **Per-tile launch overhead is NOT a factor** post-fix: 8×4096 (168 tiles)=218 ms
  vs 32×1024 (672 tiles)=205 ms, same bytes → fewer tiles no faster. So
  **kernel-batching is OFF the table — no real gain.**

### The ONE remaining real lever: multi-stream encode-source pipeline (~1.5×)
Route the expert source H2D through the 4-thread **reader pool** (recovers ~10→~19
GiB/s) instead of single-thread `queue_on_stream(0)`, and overlap the encode kernels
with the next tile's copy. Estimated transform 205→~110 ms, total 294→~190 ms;
generalizes to BF16→MXFP4 (~1.7×) and any transcode-heavy load (glm/kimi/deepseek
quantize-on-load).

**Why it's not a quick patch (hazards a naive version gets wrong):**
- The host memcpy (mmap→pinned) is the bottleneck, *not* the DMA — so merely
  K-laning CUDA streams (`queue_on_stream` per-stream) does NOT recover bandwidth;
  the source must go through the reader-thread pool (`queue()`/pinned-staged).
- That pool *defers* copies → needs **per-tile completion events** (not the old
  per-tile full flush, which was the 1388-flush cost; not no-sync, which was the
  zero-weight bug) so the encode waits only on *its* tile's copy.
- The encode kernel on stream 0 has a **web of stream-0 dependencies** any
  multi-stream version must make stream-aware: FP8 scale-cache load
  (`ensure_fp8_scale_loaded`, stream 0), scale reblock D2D loop (TP-sharded),
  dequant BF16 scratch. Each needs its own event/lane or a per-scale wait.
- Source-tile buffer must become a small **ring** (depth = lanes) with reuse gated
  by the consuming encode's completion.

Do this as a focused pass with the **byte-diff differential harness** (it caught
the last silent-corruption bug): validate FP8→MXFP4 AND BF16→MXFP4 bit-exact +
parity + peak-mem before keeping. Correct-by-construction "same-stream K-lane"
(copy+kernel on the same lane stream, no events) is the elegant target IF the
scale/scratch deps can be cleanly laned; otherwise events.

Everything else (dense / TP / fusion / pre-quantized) is at the hardware wall.

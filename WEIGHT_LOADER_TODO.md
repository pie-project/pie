# Weight loader bottlenecks (post-FP4 GLM-5.1 work)

Baseline (4×B200, GLM-5.1 705 GB FP8 → 88 GB MXFP4 runtime quant):
`materialize ≈ 235–255 s/rank, total ≈ 390 s (vs ~540 s before recent fixes).`

## ARCHITECTURE RETHINK — path to 3× cold / 6–8× warm

**Diagnosis (grounded in this box's numbers):** materialize moves 192 GB FP8 H2D +
106 GB arena write per rank. At PCIe-Gen5 ×16 pinned (~50 GB/s) that's a ~4–6 s
*hardware floor*; we take 235 s → **~98 % overhead, not bandwidth.** RAM is 2 TB vs a
705 GB checkpoint, so the whole thing is page-cached — **disk is not the bottleneck.**
58,596 projections are each materialized as a serial `copy → FP8-dequant → bf16→MXFP4
encode` chain on **stream 0** (≈4 ms each, fully serial, zero overlap — the stream-0
serialization added to fix the forward race).

**Current architecture:** *recompute-everything-every-boot, per-tile-serial transform,
then a separate GPU-touching bind.* Flaws: (1) pageable per-tile H2D, no overlap;
(2) a bf16 round-trip that triples HBM traffic (192→384→106 GB); (3) bind does serial
GPU work outside the parallel phase; (4) all of it re-runs every boot.

**Redesign:** *a pipelined producer of a cacheable materialized artifact, consumed
bind-free via a slot manifest.* Three multiplicative cold-path wins → ~3×:

1. **Streaming pipeline** — reader threads → **pinned** staging ring → compute streams,
   event-driven double-buffered prefetch (undo stream-0 serialization). Overlap turns
   Σ(stages) into max(stage); pinned fixes pageable H2D. 235 s → ~90 s.
2. **Fused FP8→MXFP4 kernel** — one kernel reads FP8 + block scale, writes MXFP4 + E8M0
   directly; no bf16 intermediate. ~1.5× compute, half the HBM traffic + launches.
3. **Bind-free / slot manifest** — contract emits *all* tensors incl. derived
   (`kv_b_proj` bf16); model binds by slot index, not name lookup + serial
   `materialise_bf16`. Post-materialize 150 s → ~20 s.
   → 390 s → ~120–130 s ≈ **3×**.

4. **Materialized-artifact cache (beyond 3×)** — snapshot the 106 GB arena + manifest,
   keyed by checkpoint-hash + scheme + TP + ABI-version. Warm boot skips
   compile+quantize+bind: `mmap` + one pinned batched H2D ≈ a few s; artifact fits the
   2 TB page cache so even cross-process reloads are RAM-speed. → ~50–70 s total, **6–8×**.

The detailed, file-level items below are the concrete sub-tasks behind these four.

Already landed:
- Real `cudaMemcpyBatchAsync` in `flush_batched_copies` (rust_storage_executor.hpp:759)
- Persistent FP8 source-tile, BF16 scratch, scale-local, and scale-cache buffers (≈ no per-tile cudaMalloc on the encode hot path)
- FP8 source + FP8 scale H2D routed to stream 0 (eliminates per-tile `flush_copy_streams`)
- `copy_storage_bytes_to_device_on(...)` helper for stream-specific H2D
- MXFP4 scales kept as raw E8M0 U8 (no E8M0→F32 expansion) — removed ~30k post-materialize
  cudaMemcpy/cudaMalloc + ~14–20 GB/rank of redundant F32 scale tensors (also fixed the
  forward `w=u8` crash). See `attach_quant_metadata` group_size==32 guard.

## POST-MATERIALIZE PHASE (~150–190 s after materialize, before `serving`)

Measured: materialize done ≈ +290 s, `serving on` ≈ +480 s. The gap is
attach-quant-meta + `bind_*` + memory-plan + workspace/KV alloc. The user asked
"why a separate bind phase — the compiler should load storage→model tensors
directly." It largely does (the runtime ABI names exactly the tensors the model
consumes; materialize writes them to their final GPU buffers). `bind` is meant to
be pure pointer-wiring. The avoidable costs:

### B1. Derived tensors materialized serially in `bind` instead of in the contract (≈ tens of s)
`driver/cuda/src/model/glm5.cpp:189` `materialise_bf16(kv_b_proj)` runs an FP8→bf16
dequant **per layer (78×), each followed by a full `cudaDeviceSynchronize`** (line 89).
This is the literal thing the user is pointing at — a transform the model needs that
happens outside the parallel materialize.
- **Fix:** express `kv_b_proj`'s BF16 form as a runtime-ABI contract tensor (a Cast/
  dequant Encode in `abi.rs`, e.g. `...kv_b_proj.weight.bf16`). Then it's produced
  during the parallelized materialize across copy streams and `bind` just grabs the
  pointer. Generalizes to any model-specific derived tensor.
- **Cheap interim:** if kept in bind, drop the per-layer `cudaDeviceSynchronize` —
  launch all 78 dequants on one stream, sync once at the end.

### B2. Slot-based binding instead of name lookups
`bind_*` does ~30k `WeightStore::find` (string hash) to wire the typed struct. CPU,
sub-second, but pure churn.
- **Fix:** have the compiler emit a stable per-arch slot table (or let the model
  request tensors by contract index), so bind is O(1) array indexing. Removes the
  last CPU cost and the string allocations.

### B3. Memory planner candidate search
`plan_cuda_memory` (entry.cpp) sweeps many (N × R × page_size) combos computing
workspace bytes each. Verify it's not a multi-second cost on glm_moe_dsa; if so,
prune the candidate grid or short-circuit once a profile is fixed (we set
`memory_profile="balanced"`, so `score_as_auto` work could be skipped).

## ⭐ BIG TEST-SPEED LEVER: cache materialized (MXFP4) weights to disk

Today only the **compile plan** is cached (`compile_rust_storage_program_cached`
→ `<key>.bin`). The **quantized weights are recomputed every boot** — ~290 s of
FP8→bf16→MXFP4 re-encode — even though runtime quant is deterministic.
- **Fix:** after materialize, serialize the packed MXFP4 weights + E8M0 scales (and any
  derived bf16 like kv_b_proj) to a local cache keyed by
  (checkpoint hash, scheme, tp_rank, tp_size, runtime ABI version). On next boot,
  detect the cache and `mmap` + batched H2D straight into the persistent arena,
  skipping dequant/re-encode entirely.
- **Payoff:** reload drops from ~390 s to roughly the H2D time of ~88 GB/rank
  (~30–60 s with pinned staging) — ~6–8× faster test iteration. This is the single
  most useful change for debugging the remaining forward-correctness issues, since
  those don't require re-quantizing weights.
- **Risk:** cache invalidation (must include scheme + tp layout + ABI version in the
  key); disk space (~88 GB/rank). Gate behind `PIE_CUDA_WEIGHT_CACHE_DIR`.

The remaining items below are ordered by estimated impact on GLM-5.1's load path. Numbers are best-guess from instruction counts in the program summary (`source_tensors=119028, contracts=177627, instrs=441637`) and `runtime_quant=fp4 quantised 58596 projections`.

## TIER 1 — multi-second wins

### 1. Pinned host staging is off by default (≈ 30–60 s)
`rust_storage_executor.hpp:539 pinned_staging_enabled()` returns true only when `PIE_CUDA_ENABLE_PINNED_WEIGHT_STAGING` is set. With it off, the H2D in `safetensors.cpp:290` does `cudaMemcpyAsync` straight from the `mmap`'d shard (pageable memory). For pageable→device, the CUDA driver allocates an internal pinned bounce buffer per call and the "async" copy effectively serializes on the host side.
- **Fix:** flip the default to on, or honour `PIE_CUDA_DISABLE_PINNED_WEIGHT_STAGING` instead. Reuse the existing `pinned_slots_` ring (`enqueue_pinned_staged_copy`, line ~588) — it already handles eviction, pool sizing, and `cudaFreeHost`.
- **Risk:** pool size; default `pinned_staging_pool_bytes()` should cap below KV pool so we don't steal from the model arena.

### 2. `flush_copy_streams()` fires before every TileMap / SlabScatter / Release / Attach (≈ 5–15 s)
`rust_storage_executor.hpp:128–147` calls `flush_copy_streams()` eagerly before each non-Allocate instruction. For GLM-5.1 that's ~58 k TileMap calls + ~30 k Release + 30 k Attach. Each flush does `cudaStreamSynchronize` per copy stream (4–8 streams). Even at 50 µs per sync, that's 5–10 s of pure waiting.
- **Fix:** record a `cudaEvent_t` per copy enqueue (or per stream) and have only the consuming instruction `cudaStreamWaitEvent` on it. Most TileMaps now run on stream 0 only, so the flush is a no-op anyway — keep it strictly behind a per-stream "any pending?" flag.
- **Where:** introduce `pending_copies_per_stream_[stream]` counter, drop `flush_copy_streams()` to a no-op when all-zero.

### 3. Look-ahead Encode prefetcher (Task #12, user's "+30 %" ask)
Today, the FP4 encode loop in `materialize_encode_input_bf16_rows` (`rust_storage_executor.hpp:1117`) is fully sequential per tile: H2D → dequant → encode → emit. Disk read for tile N+1 sits idle while tile N's dequant/encode runs. With ~58 k tiles this is the biggest remaining structural cost.
- **Fix sketch:** walk the schedule one TileMap ahead. For tile N+1's source, kick a copy onto a dedicated `prefetch_stream` into a per-slot scratch buffer (double-buffered: A while B is in compute). When tile N+1 fires, `cudaStreamWaitEvent(stream0, prefetch_event)` and proceed.
- **Compiler help (optional):** the storage compiler could annotate each TileMap with the next prefetchable source. The executor can also do it on its own by scanning the schedule prefix.

### 4. `slab_scatter` does `cudaStreamSynchronize` per call (≈ 1–3 s)
`rust_storage_executor.hpp:462`. Every SlabScatter call ends with `cudaStreamSynchronize(stream)`. For a model with many fused slabs this is N × ~50 µs full-stream syncs.
- **Fix:** drop the sync; rely on stream ordering with the next consumer. If a follower instruction needs the data on a different stream, gate it on an event rather than a host sync.

## TIER 2 — sub-second to a few seconds

### 5. `materialise_bf16` calls `cudaDeviceSynchronize` per layer (≈ 1–2 s)
`driver/cuda/src/model/glm5.cpp:89`. Hit once per layer (78×) inside `bind_glm5`. `cudaDeviceSynchronize` blocks all streams, not just the one used.
- **Fix:** `cudaStreamSynchronize(0)` (or pass the stream explicitly). Better still: launch all 78 dequants, then sync once at the end of `bind_glm5`.

### 6. `fp8_scale_cache_` destructor: ~30 k `cudaFree` calls (≈ 1 s)
`rust_storage_executor.hpp:72–81`. Cache holds one `cudaMalloc`'d entry per FP8 weight; destructor frees each individually after materialize. Same pattern for `fp8_bf16_scratch_ptr_` etc. but those are 1 alloc each.
- **Fix:** allocate the scale cache from a single contiguous arena (one `cudaMalloc` sized to the sum of scale bytes) and just `cudaFree` the arena on teardown.
- **Bonus:** avoids the per-weight `cudaMalloc` *during* load too (currently `ensure_fp8_scale_loaded` line 1276 calls `cudaMalloc(scale_nbytes)` per weight — 30 k calls).

### 7. `bind_glm5` per-expert string concat + hashmap (≈ 0.3–1 s)
`driver/cuda/src/model/glm5.cpp:214–264`. Loop: `cfg.num_experts` × 3 weights × `engine.get(name)` + `engine.quant_meta(name)`. Each iteration does `std::to_string(e)`, string concat, then `unordered_map::find` on `WeightStore`. For 128 experts × 78 layers × 3 = 30 k lookups.
- **Fix:** cache the layer prefix once, do `name.reserve` to avoid realloc, or build an `experts[layer][e]` index up front during weight finalize.

### 8. TP-sharded FP8 scale slicing: per-row D2D loop (≈ 0.3–1 s)
`rust_storage_executor.hpp:1389–1400`. For each FP8 weight on a sharded rank, slices the cached scale row-by-row with N D2D `cudaMemcpyAsync` calls. For ~16 rows × 30 k weights = ~480 k tiny async D2D enqueues.
- **Fix:** issue a single 2-D `cudaMemcpy2DAsync` (src pitch = scale_cols * 4, width = local_scale_cols * 4, height = local_scale_rows) instead of the row loop.

### 9. `verbose=true` adds noticeable `std::cerr` traffic (≈ 0.3–0.8 s)
The per-stage logs in `entry.cpp` are cheap, but `rust_storage_program(version=...)` lines emit big strings and `mem_plan` lines interleave across ranks. Not a hot loop on its own, but it discourages using `verbose` for profiling.
- **Fix:** rate-limit or drop the per-rank duplicates; ranks 1–3 can emit only their `materialize done` line.

### 10. `flush_batched_copies` chunk size = 1024 — tunable
`rust_storage_executor.hpp:736`. We chunk `cudaMemcpyBatchAsync` at 1024 copies/call. Worth probing 256 / 2048 / 4096; the driver may pipeline submit/exec better with larger batches on B200.

## TIER 3 — structural / longer-term

### 11. Direct GPU-direct I/O (cuFile / GDS) instead of mmap+H2D
Today every tensor goes: NVMe → page cache → `mmap`'d host VMA → `cudaMemcpyAsync` → GPU. With cuFile (NVIDIA GDS), the same data can DMA directly from NVMe into device memory, bypassing the host bounce. For 192 GB of FP8 reads on GLM-5.1 this can be a ~30–50 % time cut on storage-bound boxes.
- **Cost:** non-trivial; requires `cuFile` runtime, fs-specific tuning, and the safetensors path becoming buffer-aware.
- **Pre-req:** alignment — safetensors tensor offsets aren't always 4 KiB aligned; cuFile needs aligned reads or a small unaligned-tail fallback.

### 12. Storage compiler IR is per-instruction, ~441 k entries
`storage_compiler.rs`. Most instructions are trivial (Allocate / Finalize / single ExtentWrite). Could collapse runs into "bulk" variants:
- "Allocate-then-ExtentWrite-then-Finalize" → a single `MaterializeFromCheckpoint` instruction
- "N consecutive Finalize of buffer i..i+N" → `FinalizeRange`
That cuts loop iterations and per-instr `program_index_.instruction()` dispatches by ~3×. Modest CPU win (~1–2 s), but it also makes look-ahead prefetching (Tier 1 #3) much cleaner.

### 13. `program_index_.instruction(instr_id)` lookup per loop step
`rust_storage_executor.hpp:102`. Currently fetched per iteration. If it's a vector index lookup, fine; if it's a hashmap, replace with a direct slice. (Need to verify; the `program_index_` wrapper hides it.)

### 14. NVFP4 port (still owed from the FP4 plan)
MXFP4 group_size=32 with E8M0 byte scales; NVFP4 uses group_size=16 with FP8 E4M3 scales — better accuracy and the same memory footprint. Plug into the existing `push_runtime_quant` once MXFP4 forward is verified working.

---

## Suggested order to attack

1. **#1 pinned staging on by default** — biggest single win, no API change.
2. **#2 drop the eager `flush_copy_streams`** — paired with per-stream event tracking.
3. **#4 / #5 / #8 sync removal** — easy correctness-preserving cleanups.
4. **#6 scale-cache arena** — eliminates 60 k `cudaMalloc/Free` calls.
5. **#3 prefetcher** — last because it requires double-buffer plumbing and is the largest diff.

Most of items 1–4 are < 50-line patches in `rust_storage_executor.hpp`. The big architectural items (#3, #11, #12) are worth doing only after the small fixes land — they're easier to measure against a clean baseline.

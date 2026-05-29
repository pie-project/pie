# TODO — Probe rework + perf status

## Current branch state: `agents/alpha` @ `3fbd8589`

### Committed and pushed (working)

1. **`cafde76a`** — Extend pure-decode fast path to chain continuations
2. **`0fe53d70`** — Fire batches synchronously from scheduler thread
   (removed async ceremony, deleted in_flight gate, +1.6% tput)
3. **`3fbd8589`** — Restructure host probes by domain; gate fire probes behind feature
   (`probe/fire.rs`, `profile-fire` feature, `probe_fire!` macro, nested hierarchy)

### Uncommitted work in progress (dirty working tree)

**Task 80: Probe upgrade — driver_cuda + IPC split + per-completion counts (DONE)**

Build-verified and tests pass. Files modified:

- `runtime/src/probe/driver_cuda.rs` — NEW. `DriverCudaProbes` struct +
  thread-local IPC phase timings (`record_ipc_submit`, `record_gpu_wait`,
  `record_ipc_recv`) + `probe_driver_cuda!` macro. Gated by
  `profile-driver-cuda` feature.
- `runtime/src/probe/mod.rs` — added `pub mod driver_cuda;`
- `runtime/Cargo.toml` — added `profile-driver-cuda` feature +
  updated `profile-hot-path` and `profile-all` bundles
- `server/Cargo.toml` — added `profile-driver-cuda` passthrough
- `runtime/src/inference/scheduler.rs`:
  - `SchedulerStats` gains `pub driver_cuda: DriverCudaProbes`
  - `execute_batch` takes `&SchedulerStats` (was `Option<Arc<...>>`)
  - `fire.execute.response_dispatch` is now a nested
    `ResponseDispatchProbes` struct with `total_us` +
    `direct_count/chain_count/chunk_count`
  - IPC phase thread-locals drained inside `probe_fire!(driver_fire_us)`
  - Per-completion-type counters incremented in match arms
- `runtime/src/inference.rs`:
  - `InferenceStats` gains nested `ResponseDispatchStats` and
    `DriverCudaStats`
  - Aggregator updated to walk new hierarchy
- `runtime/src/server/handler.rs` — emits new dotted keys:
  `fire.execute.response_dispatch.{total_us,direct_count,chain_count,chunk_count}`,
  `fire.execute.driver_cuda.{ipc_submit_us,gpu_wait_us,ipc_recv_us,sync_us}`
- `runtime/src/driver/inproc_polling.rs` — `submit_sync_for_state` and
  `wait_response` instrumented with `record_ipc_submit/gpu_wait/ipc_recv`
- `runtime/src/driver/inproc.rs` — same for InProcChannel
- `benches/pie_bench.py` — reads new keys

**Task 81: C++ driver probes via ForwardResponse payload (DONE)**

Implemented. Files modified:

- `driver/bridge/src/schema.rs` — added 6 `u32` probe fields to `ForwardResponse`
  (`probe_wire_parse_us`, `probe_plan_us`, `probe_h2d_us`,
  `probe_kernel_launch_us`, `probe_sync_us`, `probe_response_build_us`)
- `driver/bridge/include/pie_bridge/view.hpp` — added matching fields to
  `PieForwardResponseView` + wired in `build_response_desc`
- `driver/bridge/include/pie_bridge.h` — added fields to `PieForwardResponseDesc`
- `driver/cuda/src/executor/executor.cpp` — added `steady_clock` phase
  boundaries in `handle_fire_batch`: wire_parse (entry→spec expansion),
  plan (KV scan→sample plan), h2d (uploads→prepare→TP), kernel_launch
  (forward dispatch→sampling→D2H), sync (cudaStreamSynchronize),
  response_build (response assembly). `write_probes()` at all 3 return paths.
- `runtime/src/inference/scheduler.rs` — reads `ForwardResponse.probe_*`
  fields after each fire and `fetch_add`s into `stats.driver_cuda.*`
- `runtime/src/server/handler.rs` — emits 5 new keys:
  `driver_cuda.{wire_parse_us,plan_us,h2d_us,kernel_launch_us,response_build_us}`
- `benches/pie_bench.py` — reads the 5 new keys

### Build issue notes

The cmake build step (`pie-server(build)`) clones 5 deps from GitHub via CPM:
flashinfer, tomlplusplus, json, CLI11, zstd. On this machine, git clones
through the tailscale proxy are extremely slow/hanging.

**If cmake hangs on git clone:**
- Do NOT delete the cmake build dir (`target/release/build/pie-server-*/out/cuda/build/`)
- Only delete specific lock/stamp files
- Pre-populated deps live in `_deps/{name}-src/`; cached copies exist at
  `/tmp/pie-cuda-loader-build/_deps/`
- If all else fails, build without cuda: `cargo build --release -p pie -p pie-server`
  to verify Rust code compiles, then retry the full build later

**If cargo hangs at 0% CPU:**
- Check for orphaned `.cargo-lock`: `find target -maxdepth 2 -name .cargo-lock`
- Remove them: `find target -maxdepth 2 -name .cargo-lock -delete`

### Perf status

| Workload | pie tok/s | vLLM tok/s | pie/vLLM |
|---|---:|---:|---:|
| conc=256 (loaded host) | 20,100 | 21,155 | 95.0% |
| unlimited (loaded host) | 21,983 | 22,254 | 98.8% |
| conc=256 (idle host, from memory) | 23,587 | 25,937 | 90.9% |
| unlimited (idle host, from memory) | 26,510 (max 27,996) | 27,326 (max 27,395) | 97.0% |

Bounded host-side optimizations exhausted. Remaining gap requires:
- Pipelined IPC with placeholder tokens (real refactor)
- MTP draft head for Qwen3-0.6B (not available)
- Attention backend swap (FlashInfer -> FlashAttention 2)

### Speculation depth × concurrency interaction (gemma-4-E4B, L40)

**FIXED** by stripping AdaptivePolicy down to two concepts:
`fired_high_water` (monotonic max of fired batch sizes — the one
firing-rule heuristic) and `last_latency` (deadlock watchdog when
active concurrency drops below `fired_high_water`). Removed:
`cohort_high_water` (per-batch ratchet driven by noisy
`resident_count`), `prefill_cohort_grace`, `dense_cohort_wait`
multiplier, `in_flight` gate, `PREFILL_COHORT_TARGET`.

Diagnosis: the cohort_high_water ratchet locked the target at ~67
due to transient `resident_count` spikes during speculative-context
turnover. Real batches reached R=64 in ~700 µs but the policy
waited the full `last_latency` (~20 ms) before timing out. Removing
the ratchet eliminates the stall — fire happens immediately when
batch.len() matches `fired_high_water`.

| reqs | depth=0 (before/after) | depth=1 (before/after) | d=1/d=0 |
|---:|---:|---:|---:|
|  16 |   — / 957  |   — /  977 | 102% |
|  32 | 1,686 / 1,741 |   882 / 1,742 | 100% |
|  64 | 3,054 / 3,074 | 1,612 / 3,149 | 102% |
| 128 | 4,822 / 4,908 | 3,789 / 5,019 | 102% |
| 256 | 6,556 / 6,648 | 6,598 / 6,816 | 104% |

Stripped policy is faster than the old one at every concurrency
(both d=0 and d=1) AND simpler. depth=1 is net-positive at every
concurrency (102%+ of d=0). At conc=64 + d=1, pie went from
50.6% of vLLM (1,612 vs 3,199) to 98.4% (3,149).

### 256x128 decode profile (gemma-4-E4B, L40, agents/alpha)

Measured-window profile (excluding warmup):
```
Per-decode-fire breakdown (R=256, N=256, 128 fires):
  GPU compute (sync_us):       ~25 ms     (88%)
  Kernel launch (graph+memcpy): ~300 us    (graph replay confirmed)
  Plan + H2D + wire_parse:     <50 us     (negligible)
  Response dispatch:           ~180 us    (256 chain submits)
  Batch build:                 ~100 us
  IPC submit + recv:           ~4 us      (already minimal)

Inter-fire serial gap:         ~1-3 ms    (post_dispatch_to_fire)
  ↳ Scheduler thread blocks on req_rx.recv() waiting for the
    chain pool to deliver next batch's requests. The tokio MPSC
    round-trip + worker wake adds latency that can't overlap
    with GPU compute.
```

**Optimization opportunities (decode steady-state):**

1. **Pipelined chain extension** (biggest win, ~1-3ms/fire)
   - Currently: response_dispatch → chain_pool.submit (tokio) →
     pool worker wake → build_next_request → submit_chain (mpsc)
     → scheduler.req_rx.recv() → accumulate → fire
   - Fix: synchronously build next requests in response_dispatch
     loop, push directly into BatchAccumulator. Bypass req_rx
     entirely for the chain-extension path.
   - Cross-module change: scheduler exposes accumulator, speculator
     supports sync-build entry point.

2. **Pre-allocate batch_build scratch** (~50us/fire)
   - new_batched_forward_request_with_capacity allocates 20+ Vecs
     each fire. Pool them across fires.

3. **Batch chain pool submits** (~50-100us/fire)
   - Currently 256 separate `pool.submit()` calls. Batch into one
     per-shard call with a Vec<ChainExtJob>.

**Won't help:**
- CUDA graph replay (already working, 300us dispatch confirmed)
- Fused lm_head argmax (already in the graph)
- IPC ring optimizations (already at <5us per round-trip)
- Batch size hist reporting fix (cosmetic, no perf impact)

### Probe hierarchy (current + planned)

```
fire (profile-fire)                    DONE — committed in 3fbd8589
├── inter_fire_us
├── post_dispatch_to_fire_us
├── accumulate.accum_loop_us
├── pre_dispatch.fire_prepare_us
├── execute.total_us
│   ├── execute.batch_build_us
│   ├── execute.driver_fire_us
│   └── execute.response_dispatch.total_us    DONE (task 80)
│       ├── .direct_count                     DONE
│       ├── .chain_count                      DONE
│       └── .chunk_count                      DONE
├── post_dispatch.context_tick_us
└── post_dispatch.stats_update_us

driver_cuda (profile-driver-cuda)      DONE (tasks 80 + 81)
├── ipc_submit_us                      DONE
├── gpu_wait_us                        DONE
├── ipc_recv_us                        DONE
├── wire_parse_us                      DONE (task 81, C++ side)
├── plan_us                            DONE
├── h2d_us                             DONE
├── kernel_launch_us                   DONE
├── sync_us                            DONE
└── response_build_us                  DONE

chain_ext (profile-chain-ext)          NOT STARTED
├── wake_us
├── work_us
└── submit_chain_us

inferlet (profile-inferlet)            NOT STARTED
startup (profile-startup)              NOT STARTED
memory (profile-memory)                NOT STARTED
```


---

# (merged from agents/bravo)

# TODO

## Kimi CUDA Driver — Correctness

- ~~Investigate the remaining Moon-prompt repetition case.~~ (partially resolved)
  - **Fixed**: 4 bugs found and fixed:
    1. MLA KV cache uninitialized (`mla_cache.cpp`: `cudaMemset` after alloc)
    2. Missing `e_score_correction_bias` for noaux_tc MoE routing (`kimi.hpp/cpp`, `kimi_mla.cu/hpp`, `generic.rs`)
    3. YaRN RoPE not applied: `has_rope_scaling=true` missing from yarn branch, wrong `mscale_all_dim` handling (`hf_config.cpp`), added `launch_rope_yarn_original_bf16` call (`kimi_forward.cpp`)
    4. Routing weights used biased scores instead of original sigmoid (`kimi_mla.cu`)
  - **Result**: Short factual answers correct ("Paris."). Raw completion produces correct content.
  - **Remaining**: Output still repeats for multi-sentence generation. Root cause: FlashInfer `BatchMLAPagedAttention` (absorbed Q, 512-dim dot products) accumulates more FP error than vLLM's explicit-K/V prefill + TRT-LLM decode (128-dim dot products).
- Implement explicit K/V prefill for Kimi MLA.
  - During prefill, compute `kv = kv_b_proj(kv_c)` → explicit K_nope, V.
  - Use standard FlashInfer prefill attention (not absorbed MLA).
  - Continue using absorbed MLA for decode (read from compressed KV cache).
  - This matches vLLM's architecture and should fix the remaining repetition.
- Add focused correctness tests for Kimi decode.
  - Single request, deterministic temperature 0.
  - Multi-token decode with EOS enabled and disabled.
  - Batch size > 1 with mixed prompt lengths.
  - Long context prefill followed by decode.

## Kimi CUDA Driver — Performance

### Current numbers (8× H100 80GB, TP=8, temp=0, max_tokens=128)

| Benchmark | Pie CUDA | vLLM 0.21.0 | Ratio |
|-----------|---------|-------------|-------|
| Latency (1 req) | 34.14 tok/s | 116.49 tok/s (compiled) | 0.29x |
| Throughput (c=4, 16 req) | 91.14 tok/s | — | — |
| Throughput (c=16, 64 req) | 202.25 tok/s | — | — |

### Gap analysis
- 3.4x gap vs compiled vLLM is mainly kernel launch overhead (CUDA graphs + torch.compile).
- Pie can't use CUDA graphs for Kimi — model uses 91% GPU memory, no headroom for capture.
- Pie wins on eager-mode vLLM (2.95x latency, 1.64x throughput).

### Completed optimizations
- ✓ Weight fusion: q_a+kv_a, shared gate+up (via weight loader compiler `add_mla_fused_projection_joins`)
- ✓ Fused split_kv_a+rmsnorm kernel (`launch_kimi_split_kv_a_norm_bf16`)
- ✓ `cudaMemcpyBatchAsync` enabled by default (CUDA 12.8+, reduces API call overhead for 161k H2D copies)
- ✓ `StorageProgramSummary` profiling infrastructure

### Slab scatter investigation (completed, not viable for TP-sharded models)
- Diagnosed: per-rank slices are sparse within files (803.7 MiB payload in 9349.7 MiB span, 11.63x overread)
- With relaxed limits (512 MiB slab, 16x overread): reduced 161k copies to 1142 slabs, but H2D bandwidth exploded from 48.9 GiB to 549 GiB per rank due to gap data in staging copies
- Conclusion: slab scatter is counterproductive when overread > ~2x. For TP-sharded MoE models, `cudaMemcpyBatchAsync` is the right approach (batches 161k copies into ~2528 API calls with zero bandwidth waste)

### Remaining optimization paths
1. **CUDA graphs with reduced memory footprint** — reduce model memory pressure to free headroom for graph capture scratch buffers. Options: more aggressive weight quantization, KV cache reduction, or partial graph capture (decode-only subgraphs).
2. **FlashInfer v0.6.12 hopper MLA kernel** — CuTe DSL MLA decode for H64 head dim. Blocked by CUDA 12.8 toolkit (needs CCCL from CUDA 13+). CUDA 13.2 migration blocked by Marlin `__global__` stub visibility change breaking cross-TU kernel refs.
3. **cublasGemmStridedBatchedEx** for q_nope_to_latent/latent_to_v — replace multiple small GEMMs with batched version.
4. **Persistent/mega-kernel** to eliminate launch overhead without CUDA graphs — amortize 1200+ kernel launches per forward pass.
5. **Forward pass profiling** — `PIE_KIMI_PROFILE=1` instrumentation exists, need to collect breakdown (embed, attn, MoE router, gate_up, swiglu, down, shared, allreduce, residual, lm_head).

## Weight Loader — Tests

- ~~SlabScatter lowering and FFI layout~~ (done, 4 tests)
- ~~MLA weight fusion through compiler~~ (done, `mla_q_kv_a_fusion_produces_joined_tensor`)
- ~~rs_cache test signature fix~~ (done)
- Add executor-level SlabScatter smoke test (copy+scatter+readback).
- Add regression coverage for async safetensors storage copies.

## Weight Loader — Performance

- ✓ `cudaMemcpyBatchAsync` default-enabled (was behind `PIE_CUDA_ENABLE_BATCHED_WEIGHT_COPIES` env var, now default on CUDA 12.8+, disable with `PIE_CUDA_DISABLE_BATCHED_WEIGHT_COPIES=1`)
- Profile: 161k copies × 8 ranks, ~48.9 GiB/rank, 316 flush cycles → 2528 batch calls
- Further reduction: slab scatter only viable for dense (non-TP-sharded) tensors with low overread

## Weight Loader — Compiler

- ✓ `add_mla_fused_projection_joins()` — fuses q_a+kv_a and shared gate+up for MLA models
- ✓ `StorageProgramSummary` for compile-time profiling
- ✓ Slab scatter debug diagnostics (per-file entry count, span, max gap, overread ratio under `PIE_WEIGHT_LOADER_DEBUG=1`)
- Revisit compiled instruction shape for cost-based batching decisions
- Persist richer compile-cache metadata (version, file hashes, compiler options)

## FlashInfer Upgrade (blocked)

- v0.6.9 currently pinned (works with CUDA 12.8 nvcc)
- v0.6.12rc1 has CuTe DSL MLA decode kernel (hopper H64) — potential major decode speedup
- v0.6.12 build failures:
  - Needs `cuda::fast_mod_div` from CCCL 13.2 → requires CUDA 13.2 toolkit
  - CUDA 13.2 makes `__global__` kernel host stubs static → breaks Marlin cross-TU refs
  - Tried `-fvisibility=default`, `CUDA_SEPARABLE_COMPILATION`, `-rdc=true` — none worked
  - Fix: restructure Marlin host wrappers or use device link step
  - Also needs `cudaMemcpyBatchAsync` compat wrapper (`#if CUDART_VERSION >= 13020`)

## Cleanup Before Merge

- Remove `PIE_CUDA_KIMI_FORCE_PREFILL_MOE` diagnostic switch
- Remove slab scatter debug diagnostics from `build_slab_scatter_writes` (or gate behind `PIE_WEIGHT_LOADER_DEBUG`)
- Gate `PIE_KIMI_DUMP_LOGITS` instrumentation behind feature flag
- Re-run full test suite:
  - `cargo test -p pie-weight-loader` (12 tests, all passing)
  - `cargo test -p pie-weight-loader --test ffi_layout` (13 tests, all passing)
  - `CMAKE_CUDA_ARCHITECTURES=90 cargo build --release -p pie-server --features driver-cuda`

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

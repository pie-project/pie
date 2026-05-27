# TODO — Probe rework + perf status

## Current branch state: `agents/alpha` @ `3fbd8589`

### Committed and pushed (working)

1. **`cafde76a`** — Extend pure-decode fast path to chain continuations
2. **`0fe53d70`** — Fire batches synchronously from scheduler thread
   (removed async ceremony, deleted in_flight gate, +1.6% tput)
3. **`3fbd8589`** — Restructure host probes by domain; gate fire probes behind feature
   (`probe/fire.rs`, `profile-fire` feature, `probe_fire!` macro, nested hierarchy)

### Uncommitted work in progress (dirty working tree)

**Task 80: Probe upgrade — driver_cuda + IPC split + per-completion counts**

All Rust code is written but not yet built/tested. Files modified:

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

**To finish task 80:**
1. Build: `cargo build --release -p pie-server --features driver-cuda,profile-hot-path`
   (cmake was hanging on git clones through proxy — deps are now pre-populated,
   but cmake may still try to update. If stuck, try without profile features first:
   `cargo build --release -p pie-server --features driver-cuda`)
2. Run scheduler tests: `cargo test --release -p pie --lib inference::scheduler`
3. Quick bench with feature on to verify probes report values
4. Commit and push

**Task 81: C++ driver probes via ForwardResponse payload (NOT STARTED)**

Plan:
- Add probe u32 fields to `ForwardResponse` in `driver/bridge/src/schema.rs`
  (wire_parse_us, plan_us, h2d_us, kernel_launch_us, sync_us, response_build_us)
- Instrument `handle_fire_batch` in `driver/cuda/src/executor/executor.cpp`
  with `std::chrono::steady_clock` brackets around each host phase
- Read the fields in Rust `execute_batch`, fetch_add into `stats.driver_cuda.*`
- Update handler.rs + bench harness for C++ phase keys

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
│   └── execute.response_dispatch.total_us    IN PROGRESS (task 80)
│       ├── .direct_count                     IN PROGRESS
│       ├── .chain_count                      IN PROGRESS
│       └── .chunk_count                      IN PROGRESS
├── post_dispatch.context_tick_us
└── post_dispatch.stats_update_us

driver_cuda (profile-driver-cuda)      IN PROGRESS (task 80 Rust side)
├── ipc_submit_us                      IN PROGRESS
├── gpu_wait_us                        IN PROGRESS
├── ipc_recv_us                        IN PROGRESS
├── wire_parse_us                      PLANNED (task 81, C++ side)
├── plan_us                            PLANNED
├── h2d_us                             PLANNED
├── kernel_launch_us                   PLANNED
├── sync_us                            PLANNED
└── response_build_us                  PLANNED

chain_ext (profile-chain-ext)          NOT STARTED
├── wake_us
├── work_us
└── submit_chain_us

inferlet (profile-inferlet)            NOT STARTED
startup (profile-startup)              NOT STARTED
memory (profile-memory)                NOT STARTED
```

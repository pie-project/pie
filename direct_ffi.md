# Direct FFI Runtime-Driver Refactor

## Status

Design plan for replacing the local runtime-driver request/response transports with a direct, non-blocking FFI boundary.

The primary goal is a small and comprehensible final architecture. Migration compatibility, shadow paths, legacy driver support, and preservation of the existing transport abstractions are explicitly secondary. Do not retain old code behind feature flags once its replacement works.

The implementation review in [direct_ffi_fix.md](direct_ffi_fix.md) is the
normative amendment for terminal outcomes, atomic batch acceptance, global
channel endpoint lifetime, readiness reservations, and final validation gates.

## 1. Goal

For every embedded driver, reduce the runtime-driver boundary to:

```text
runtime -> driver: direct C ABI calls
runtime <- driver: runtime-supplied completion callback
values: persistent device/shared/pinned memory
```

The final hot path is:

```text
inferlet/process requests
        |
        v
per-driver batching sequencer       the only request queue
        |
        | direct non-blocking FFI call
        v
CUDA / Metal driver
        |
        | GPU completion callback -> runtime notify callback
        v
Rust future wakeup
```

A direct call means that the driver consumes the launch descriptor and enqueues device work before returning. It does not mean waiting synchronously for GPU completion.

## 2. Non-goals

- Preserve `DriverChannel` compatibility.
- Preserve local shared-memory driver invocation.
- Preserve `ipc_profile` or configurable spin budgets.
- Preserve `RequestPayload`/`ResponsePayload` as the local embedded-driver API.
- Preserve the current `ControlPlane` prototype.
- Preserve standalone CUDA/Metal driver process entry points.
- Keep dual legacy/direct production paths.
- Build a generic plugin transport into the engine core.
- Change the cross-node KV/RS transport in `driver/transport` as part of this refactor.

`driver/transport` is the prefill/decode data-transfer layer. It is independent of local runtime-driver invocation and remains a separate subsystem.

## 3. Current State

The structural migration is substantially complete. The legacy local
request/response transports, driver service loops, response payloads, IPC
configuration, and old schema machinery are gone. The current local path is:

```text
pipeline
  -> per-driver scheduler queue
  -> typed DriverBackend operation
  -> direct CUDA / Metal FFI or direct Rust Dummy call
  -> GPU
  -> persistent channel/terminal publication
  -> runtime callback wake
```

The remaining work is behavioral rather than transport migration: Metal's
direct entry is still a stub, callback arrival is treated as success, rejected
fires can consume epochs, channel endpoints and waits are not yet globally
owned, Dummy retains a private `mpsc` operation queue, and CUDA tensor-parallel
typed memory operations reach only rank 0. The source-backed current assessment
is [direct_ffi_fix.md](direct_ffi_fix.md).

## 4. Hard Design Decisions

### 4.1 Keep one queue

Retain the request-to-batching-scheduler queue. Remove the batching-scheduler-to-driver request queue.

The existing per-driver batching scheduler is also the sequencer. Submission into its existing queue is the ordering point; this refactor does not add a second command queue or a new generic command envelope. The scheduler's consumer thread exclusively owns the native driver and directly invokes all operations that must be ordered relative to forward execution:

```text
KV copy
state copy
page-table update
pool map/unmap
forward launch
```

No arbitrary runtime thread may launch ordered driver work independently. Fire-associated work such as KV CoW may travel with the pending fire and run before its launch. Operations that can occur independently, such as residency copies or pool resize, must still enter the existing per-driver scheduler queue and execute at their queue position. The internal queue item representation is an implementation detail; do not add an abstraction solely to rename the queue.

### 4.2 Use a runtime-supplied callback table

Do not make C++ depend directly on process-global Rust symbols. Pass a versioned callback table when creating the driver.

```c
typedef struct PieRuntimeCallbacks {
    uint32_t abi_version;
    void* ctx;

    // Callable from any foreign thread. Must never unwind.
    void (*notify)(
        void* ctx,
        uint64_t wait_id,
        uint64_t epoch);
} PieRuntimeCallbacks;
```

The Rust implementation may initially delegate to the existing generation-tagged `WakerTable` and `wake_past` implementation.

C++ stores only the callback table and opaque `wait_id` values. It never stores a Rust `Waker`.

### 4.3 Use typed direct operations

Do not replace the old request union with a new generic method union. Expose distinct operations:

- `launch`
- `copy_kv`
- `copy_state`
- `resize_pool`
- `register_program`
- `register_channel`
- `bind_instance`
- `close_instance`
- `close_channel`

This keeps KV, recurrent state, memory backing, and execution semantics separate.

### 4.4 Driver output is channels only

Every driver-to-runtime value is published through a bound channel. Inferlets and runtime code observe driver output only through `channel.take` or `channel.read`; there is no second result API.

The channel implementation uses persistent memory:

- Channel values: device cells plus pinned/shared frame mirrors.
- Channel indices and errors: pinned/shared head, tail, poison, and closed words.
- Operation outcomes: runtime-owned stable terminal cells leased per submission.
- KV/RS state: typed driver pools, not returned values.

Callbacks carry no value, status, or error. For launch batches, one callback
arrives after channel publication and every member's terminal entry is visible.
It means only "re-check persistent channel and terminal state."

### 4.5 No generic local response frame

Delete the local `DriverResponse`, `ForwardResponse`, and `ForwardOutput` paths entirely. Do not replace them with an output arena or another per-batch result object. Legacy token/distribution/logit/logprob/entropy/speculative response arrays and per-request response fanout are deleted; an inferlet that needs a value declares a channel and receives it through `channel.take` or `channel.read`.

Immediate FFI validation errors reject the entire typed operation and are
returned directly from the call. After a launch is accepted, asynchronous
errors are published through the affected channel's error/poison state and the
member's `Failed` terminal cell before wakeup. The completion
callback itself carries no error value.

## 5. Target C ABI

The exact struct layout belongs in `pie-driver-abi` and must be generated into the C header rather than handwritten independently in Rust and C++.

```c
typedef struct PieDriver PieDriver;

typedef struct PieTerminalStatus {
    uint64_t operation_epoch;
    uint32_t outcome; /* 0 = pending, 1 = success, 2 = failed */
    uint32_t reserved;
} PieTerminalStatus;

typedef struct PieCompletion {
    uint64_t wait_id;
    uint64_t target_epoch;
    PieTerminalStatus* terminal_status;
} PieCompletion;

typedef struct PieDriverCreateDesc {
    const uint8_t* config_bytes;
    size_t config_len;
    PieRuntimeCallbacks runtime;
} PieDriverCreateDesc;

typedef struct PieDriverCaps {
    const uint8_t* json_bytes;
    size_t json_len;
} PieDriverCaps;

PieDriver* pie_cuda_create(
    const PieDriverCreateDesc* desc,
    PieDriverCaps* caps);

int32_t pie_cuda_register_program(
    PieDriver* driver,
    const PieProgramDesc* program,
    uint64_t* program_id);

int32_t pie_cuda_register_channel(
    PieDriver* driver,
    const PieChannelDesc* channel,
    PieChannelBinding* binding);

int32_t pie_cuda_bind_instance(
    PieDriver* driver,
    const PieInstanceDesc* instance,
    PieInstanceBinding* binding);

int32_t pie_cuda_launch(
    PieDriver* driver,
    const PieLaunchDesc* launch,
    PieCompletion completion);

int32_t pie_cuda_copy_kv(
    PieDriver* driver,
    const PieKvCopyDesc* copy,
    PieCompletion completion);

int32_t pie_cuda_copy_state(
    PieDriver* driver,
    const PieStateCopyDesc* copy,
    PieCompletion completion);

int32_t pie_cuda_resize_pool(
    PieDriver* driver,
    const PiePoolResizeDesc* resize,
    PieCompletion completion);

int32_t pie_cuda_close_instance(PieDriver* driver, uint64_t instance_id);
int32_t pie_cuda_close_channel(PieDriver* driver, uint64_t channel_id);
void pie_cuda_destroy(PieDriver* driver);
```

Metal exposes the same shape with `pie_metal_*` symbols. Dummy implements the
equivalent `DriverBackend` operations directly in Rust and does not need C FFI.

`PieLaunchDesc` carries one stable terminal-cell pointer per instance id.
Validation and acceptance are atomic for the whole call. `PieCompletion` is
one operation wake target, not an instance epoch; launch-member success or
failure is read from those leased cells. For copy, state-copy, and resize,
`terminal_cell` points to one runtime-owned stable cell that remains live
through callback and retirement.

Startup configuration and capabilities are cold-path data. Keeping them as versioned JSON bytes is acceptable and avoids a large unstable startup C struct. Launch, memory, copy, and completion descriptors must be typed POD views.

## 6. FFI Ownership Rules

Keep the boundary simple:

1. `PieDriver*` is created once at worker bootstrap, owned exclusively by the per-driver scheduler, and remains alive for the worker lifetime.
2. Input descriptor pointers are borrowed only for the duration of the FFI call.
3. `launch()` must copy or consume all required metadata before returning.
4. A control operation may retain only the `PieCompletion.terminal_status`
   pointer; the runtime guarantees that fixed cell remains stable through its
   callback and retirement.
5. Persistent channel, instance-terminal, and pool addresses remain stable
   until their ordered close.
6. Completion ids are generation-tagged opaque integers.
7. An instance or channel is closed only after the scheduler has retired every operation that references it.
8. The runtime callback table has the same worker lifetime as the driver.
9. Worker shutdown stops submission, drains submitted work and callbacks, joins the scheduler, and then destroys the driver exactly once.
10. No Rust panic or C++ exception crosses the ABI.

## 7. Completion Protocol

### 7.1 Publication order

An accepted launch batch follows:

```text
member channel cells and head/tail, or affected channel poison
-> every member terminal entry = Success or Failed(exact instance epoch)
-> release fence
-> runtime.notify(batch wait_id, batch target epoch) exactly once
```

An accepted copy, state-copy, or resize follows:

```text
ordered memory effect, or affected channel poison on failure
-> PieCompletion.terminal_status = Success or Failed(operation epoch)
-> release fence
-> runtime.notify(wait_id, target epoch) exactly once
```

The runtime channel operation follows register-then-recheck:

```text
channel.take/read checks channel state
-> register Waker
-> re-check channel state
-> return or park
```

### 7.2 CUDA

The CUDA stream enqueues:

1. Compute.
2. D2H mirror copies where required.
3. Device or host word publication.
4. `cudaLaunchHostFunc` completion callback.

The host function invokes `runtime.notify`. It must not call CUDA APIs.

Replace `FrameCarrierEngine::publish_device`'s host-side stream synchronization with stream-ordered publication and notification. A direct launch must return after enqueue, not after `cudaStreamSynchronize`.

### 7.3 Metal

The Metal command buffer completion handler invokes `runtime.notify` after channel-cell and shared channel-metadata publication. The old completion thread and `send_response` path are unnecessary once every output is published through channels.

### 7.4 Wait slot classes

Use the same opaque wait-slot mechanism for:

- Batch completion.
- Host-reader channel head changes.
- Host-writer channel tail changes.
- Copy/resize completion.

A single callback function is sufficient. The `wait_id` identifies the event
class on the Rust side. Instance fire success or failure is persistent terminal
state, not a separate callback payload or pacing wake.

## 8. Runtime Target Structure

Replace the current driver module with:

```text
runtime/engine/src/driver/
├── backend.rs          # DriverBackend dispatch and concrete-backend exports
├── backend/
│   ├── dummy.rs        # Rust adapter over pie-driver-dummy
│   ├── cuda.rs         # CUDA adapter over pie-driver-abi symbols
│   └── metal.rs        # Metal adapter over pie-driver-abi symbols
├── registry.rs         # immutable per-driver registry/specs
├── completion.rs       # wait slots, pinned words, Completion future
└── frame.rs            # bound instance/frame/channel layout
```

Raw FFI declarations and ABI types remain in `pie_driver_abi`; the engine
backend modules own lifecycle, frame conversion, completion, and validation.

Suggested Rust API:

```rust
pub enum DriverBackend {
    Dummy(DummyDriver),
    Cuda(CudaDriver),
    Metal(MetalDriver),
}

impl DriverBackend {
    pub fn capabilities(&self) -> &DriverCapabilities;
    pub fn register_program(&mut self, desc: &ProgramDesc) -> Result<ProgramId>;
    pub fn register_channel(&mut self, desc: &ChannelDesc) -> Result<BoundChannel>;
    pub fn bind_instance(&mut self, desc: &InstanceDesc) -> Result<BoundInstance>;
    pub fn launch(&mut self, desc: &LaunchDesc) -> Result<Completion>;
    pub fn copy_kv(&mut self, desc: &KvCopyDesc) -> Result<Completion>;
    pub fn copy_state(&mut self, desc: &StateCopyDesc) -> Result<Completion>;
    pub fn resize_pool(&mut self, desc: &PoolResizeDesc) -> Result<Completion>;
    pub fn close_instance(&mut self, id: InstanceId) -> Result<()>;
    pub fn close_channel(&mut self, id: ChannelId) -> Result<()>;
}
```

These operations are synchronous at invocation time. Returned `Completion`
values are asynchronous.

Each scheduler owns one `DriverBackend` for its thread lifetime. The immutable
registry contains cloneable scheduler handles and capabilities, not
`Arc<DriverBackend>`; arbitrary runtime threads cannot call the driver directly.
Channel registration is explicit, serialized by that scheduler, and performed
lazily on a channel's first bind.

## 9. Scheduler Refactor

### 9.1 Replace deferred response handles

Delete:

- `DriverChannel::submit`
- `DriverChannel::submit_sync`
- `DriverChannel::submit_deferred`
- `DeferredResponse`
- `FireHandle`
- `fire_batch`
- `fire_batch_sync`
- `fire_batch_deferred`

After dequeuing and batching pending fires, the scheduler directly calls:

```rust
let completion = driver.launch(&batch_desc)?;
```

It records the batch as in flight and keeps building the next batch. Completion retirement runs when `Completion` resolves.

### 9.2 Separate batch and fire completion

A batch completion is one wake-only event delivered after every accepted member
has published persistent terminal state. Each member has a scheduler-assigned
instance epoch and leased terminal cell. The callback retires native
in-flight accounting and prompts exact state rechecks; it does not determine
member success.

`PendingFire` changes from:

```rust
rx: oneshot::Receiver<Result<ForwardOutput>>
```

to:

```rust
submission: SubmissionTicket
```

`SubmissionTicket` identifies the exact instance epoch and leased terminal
cell. Runtime transactions commit only after the cell contains `Success` and
abort on `Failed` or close. The inferlet still obtains values and error details
only through its channels.

### 9.3 Remove response marshaling

CUDA and Metal must both publish host-reader channel values into the bound mirror. `channel.take` and `channel.read` read only channel state and the mirror. Remove the `ForwardResponse` fallback and the assumption that a response oneshot establishes the happens-after edge.

Delete `inference/response.rs`, response splitting, and per-request driver-response fanout. Move any surviving scheduler counters to the scheduler module; they must not depend on a driver result payload.

## 10. Driver Refactor

### 10.1 CUDA

Split initialization and execution out of the process-style entry function into an owned driver object:

```cpp
class CudaDriver {
public:
    static std::unique_ptr<CudaDriver> create(...);
    int register_channel(const PieChannelDesc&, PieChannelBinding*);
    int launch(const PieLaunchDesc&, PieCompletion);
    int copy_kv(const PieKvCopyDesc&, PieCompletion);
    int copy_state(const PieStateCopyDesc&, PieCompletion);
    int resize_pool(const PiePoolResizeDesc&, PieCompletion);
    int close_channel(std::uint64_t channel_id);
};
```

The leader `InProcServer` loop is deleted. The runtime scheduler thread directly
calls the leader object. TP follower threads remain an internal NCCL
implementation detail.

The launch path should directly reuse the existing executor/`handle_fire_batch` implementation. Do not duplicate forward logic in the FFI wrapper.

### 10.2 Metal

Metal has the same owned object lifecycle and direct launch methods, and its
service/response loop is deleted. The remaining requirement is to connect those
methods to real executor work and terminal/channel publication.

The raw Metal and optional MLX executors may remain internal implementations of the same direct `launch` method.

### 10.3 Dummy

Dummy is now a normal Rust direct-driver object:

```rust
pub struct DummyDriver { ... }

impl DummyDriver {
    pub fn launch(&mut self, desc: &LaunchDesc) -> Result<Completion>;
}
```

It is used through `DriverBackend::Dummy` without C FFI or shmem. Its remaining
architectural defect is the private `mpsc` worker queue behind the scheduler.
Remove that second ordering queue, validate a whole batch before mutation, run
deterministic Dummy work directly, and dispatch only the final callback
asynchronously.

## 11. Typed Memory Operations

The direct boundary should match the new memory architecture rather than preserve generic cache operations.

Required typed descriptors:

```text
PieKvCopyDesc
PieStateCopyDesc
PiePoolResizeDesc
```

Do not expose a generic `CopyResource` enum at the local boundary.

The per-driver sequencer orders these operations with forwards. This is the execution side of:

- KV page CoW.
- Recurrent-state slot copy.
- CUDA VMM pool resize.
- Metal sparse-buffer map/unmap.

Cross-node movement remains implemented independently by `driver/transport` and is outside this local FFI boundary.

## 12. Worker Refactor

### 12.1 Startup lifecycle

Replace process-style `run`/`request_stop` entry points with create/destroy:

```text
old:
spawn driver thread -> pie_driver_*_run_inproc -> ready callback
-> recv/send service loop -> request_stop + join

new:
pie_*_create -> DriverBackend + capabilities
-> move DriverBackend into the per-driver scheduler thread
-> register scheduler handle with engine
-> stop submission + drain + join -> pie_*_destroy
```

No leader service thread is needed. Model loading may block during `create` at worker startup.

### 12.2 Files to simplify

- `worker/src/driver_ffi.rs`: declare create/destroy/direct-operation symbols only.
- `worker/src/embedded_driver.rs`: remove vtable context, channel construction, inproc/shmem selection, server thread lifecycle, and process-style stop logic. Rename to `driver_backend.rs` if the remaining code is primarily backend ownership.
- `worker/src/engine.rs`: start the scheduler with `DriverBackend` ownership and install its handle, not `DriverChannel`.
- `worker/src/config.rs`: remove all IPC profile configuration.

## 13. Configuration Deletion

Delete from worker configuration and all sample configs:

```text
ipc_profile
spin_budget_us
```

Delete:

- `IpcProfile`
- `effective_ipc_profile()`
- `effective_spin_budget_us()`
- `use_inproc_polling_channel()`
- latency/balanced/power profile documentation
- shmem slot/buffer settings used only for local driver invocation

There is no runtime knob for choosing a local transport because there is only one local transport: direct FFI.

## 14. Build-System and Dependency Cleanup

### 14.1 `interface/driver`

Keep `pie-driver-abi`, but reduce it to the direct local FFI contract. The final crate owns plain `#[repr(C)]` POD types for:

```text
PieRuntimeCallbacks
PieCompletion
PieDriverCreateDesc / PieDriverCaps
PieProgramDesc
PieChannelDesc / PieChannelBinding
PieInstanceDesc / PieInstanceBinding
PieLaunchDesc
PieKvCopyDesc
PieStateCopyDesc
PiePoolResizeDesc
```

The target interface layout is:

```text
interface/
└── driver/
    ├── Cargo.toml
    ├── build.rs                 # include-dir handoff only
    ├── src/
    │   ├── lib.rs
    │   ├── local.rs             # repr(C) direct FFI types + symbol declarations
    │   ├── capabilities.rs      # cold create-time JSON type
    │   └── transfer.rs          # cross-node KvHandle/KvLayout vocabulary
    ├── cbindgen/                # simple committed-header generator
    └── include/
        └── pie_driver_abi.h
```

Generate a C-compatible `include/pie_driver_abi.h` directly from the plain source types with cbindgen. The generated descriptors should be directly consumable by the FFI entry points; do not regenerate a generic request/response schema or method union. CUDA and Metal may use private C++ views internally, but those views belong to each driver and must not recreate a shared response protocol in `interface/driver`.

`interface/ptir` remains the semantic owner of the canonical PTIR container and sidecar formats. `PieProgramDesc` does not mirror PTIR structs into C; it carries borrowed canonical bytes plus the minimal binding metadata needed by `register_program`.

Delete the old local-wire machinery:

```text
interface/driver/src/schema.rs
interface/driver/src/pod.rs
interface/driver/src/brle.rs
interface/driver/derive/
interface/driver/include/pie_driver_abi/view.hpp
interface/driver/include/pie_driver_abi/response_builder.hpp
SCHEMA_HASH and its xxhash build dependency
rkyv and pie-driver-abi-derive dependencies
cbindgen parse.expand / RUSTC_BOOTSTRAP workaround
```

`Brle` is not part of the final local ABI. Delete the classic request-mask path and move any still-useful host-only bitset helpers to their owning runtime crate rather than keeping them in `pie-driver-abi`.

Delete `SamplingInput`, `SamplingBinding`, and `SamplingProgramSubmission` with the legacy sampling request path. Move `PtirChannelValue` and `PtirProgramSubmission` into `runtime/engine/src/ptir`; they are Rust-side launch-building values, not cross-language ABI types. Lower them into borrowed slices in `PieProgramDesc`, `PieInstanceDesc`, or `PieLaunchDesc` only at the direct call boundary.

Keep `DriverCapabilities` as the cold create-time JSON contract, but delete fields used only by shmem, legacy response arrays, classic sampler batching, or system-spec response handling. Remove `shmem_name`, `max_logit_rows`, `max_prob_rows`, `max_sampler_rows`, `max_logprob_labels`, `max_custom_mask_bytes`, `rs_cache_spec_rollback`, `system_speculation_supported`, and `enable_system_speculation`. Driver-internal workspace limits do not remain public capabilities merely because the old scheduler mirrored them. Keep only bootstrap facts that still drive runtime allocation, batching, model metadata, or weight layout.

Rename `kv.rs` to `transfer.rs` and keep `KvHandle`, `KvLayout`, `KvRegion`, and `MemoryDomain` as a focused Rust-only module. These are the driver-exported cross-node transfer contract, not part of the generated local C header. Keeping the tiny vocabulary here avoids adding another crate while preserving the separation from `driver/transport` implementation code.

Retain a minimal `interface/driver/build.rs` only for the `DEP_PIE_DRIVER_ABI_INCLUDE` handoff to `worker/build.rs`; it no longer computes a schema hash.

### 14.2 `worker/build.rs`

Delete:

- `pie_ipc_include_dir()`
- `DEP_PIE_IPC_INCLUDE`
- `PIE_IPC_INCLUDE_DIR`
- comments describing in-proc vtables and process entry points

Keep only the generated `pie-driver-abi` include handoff:

```rust
.define("PIE_DRIVER_ABI_INCLUDE_DIR", pie_driver_abi_include_dir())
```

Update the top-level comments to describe create/destroy/direct launch symbols rather than `pie_driver_*_run` and `_request_stop`.

### 14.3 Cargo dependencies

Remove `pie-ipc` from:

- `runtime/engine/Cargo.toml`
- `worker/Cargo.toml`
- `driver/dummy/Cargo.toml`

Delete `driver/ipc` entirely once no other crate references it.

Remove `interface/driver/derive` from the workspace. Keep `interface/driver/cbindgen` as the committed-header generator, simplified to parse the plain `#[repr(C)]` source without macro expansion.

`driver/transport` and driver-side KV exporters continue depending on `pie-driver-abi::transfer`; that module has no dependency on the local FFI implementation. `interface/controller` continues depending on `pie-driver-abi` solely for `DriverCapabilities`.

After cleanup, `pie-driver-abi` depends on `serde` for capabilities and no serialization/code-generation runtime. It must not depend on `rkyv`, `xxhash-rust`, or `pie-driver-abi-derive`.

Keep `pie-waker` as the runtime-owned completion substrate.

### 14.4 CMake

From CUDA and Metal CMake files remove:

- `PIE_IPC_INCLUDE_DIR` discovery and include directories
- `PIE_SCHEMA_INCLUDE_DIR`; use `PIE_DRIVER_ABI_INCLUDE_DIR` for the generated direct-ABI header
- `src/service/inproc_service.cpp`
- inproc-server dependencies
- tests specific to request/response IPC

Keep frame carrier, tensor/frame memory, executor, kernels, and generated driver ABI headers.

## 15. Explicit Deletion Inventory

Delete or replace the following runtime files:

```text
runtime/engine/src/driver/inproc.rs
runtime/engine/src/driver/inproc_polling.rs
runtime/engine/src/driver/shmem.rs
runtime/engine/src/driver/channel.rs
runtime/engine/src/driver/control.rs
runtime/engine/src/driver/control_cuda.rs
runtime/engine/benches/driver_channel.rs
```

Rewrite:

```text
runtime/engine/src/driver.rs
runtime/engine/src/driver/ops.rs
runtime/engine/src/driver/completion.rs
```

Delete from native drivers:

```text
driver/cuda/src/service/inproc_service.cpp
driver/cuda/src/service/inproc_service.hpp
driver/metal/src/service/inproc_service.cpp
driver/metal/src/service/inproc_service.hpp
```

Delete the `pie_driver_cuda_run`, `pie_driver_cuda_run_inproc`, `pie_driver_metal_run`, and `pie_driver_metal_run_inproc` process/service entry points after create/destroy is wired.

Delete obsolete request/response plumbing from `pie-driver-abi` after no non-local consumer remains:

```text
Frame
ResponseFrame
RequestPayload
ResponsePayload
ForwardResponse
CopyRequest
local request method tags
response builder/view helpers used only by InProcService
```

Delete the runtime response model and fanout:

```text
ForwardOutput
runtime/engine/src/inference/response.rs
dispatch_fired_batch
extract_per_request response splitting
legacy tokens/dists/logits/logprobs/entropies/spec/program_tokens response fields
```

Do not delete schema types still used by cross-node transfer or public interfaces merely because they share a crate. Move surviving transfer types into focused modules first.

## 16. Remaining Implementation Order

The former Phase A-G sequence drove the structural transport migration and is
complete enough to be historical context, not active implementation guidance.
Do not add compatibility paths around it.

The normative remaining order is the five-phase remediation plan in
[direct_ffi_fix.md §5](direct_ffi_fix.md#5-recommended-fix-order):

1. Atomic acceptance and exact terminal completion.
2. Global channel endpoints, immutable attachments, reservations, and
   same-instance run-ahead.
3. Real single-rank CUDA and Metal execution.
4. Tensor-parallel typed operations and stable sparse/VMM pools.
5. Obsolete-path removal and final validation.

## 17. Minimal Validation Gates

Compatibility is not a goal, but the final path must prove the new contract.
The expanded and normative validation matrix is
[direct_ffi_fix.md §4](direct_ffi_fix.md#4-missing-required-tests); the list
below is the original structural minimum.

Required tests:

1. Direct Dummy launch and completion.
2. Runtime callback from a foreign thread.
3. Register-then-recheck lost-wakeup test.
4. Stale generation callback is ignored.
5. Two launches complete in launch order.
6. Close wakes/poisons outstanding waiters.
7. CUDA smoke: launch returns before GPU completion and callback fires after channel publication.
8. Metal smoke: completion handler publishes channels and wakes without a payload.
9. KV copy and recurrent-state copy dispatch to different typed driver methods.
10. Pool resize completion is ordered before a dependent launch.
11. Driver values and asynchronous errors are observable only through `channel.take`/`channel.read`.

Final grep gates:

```text
rg 'DriverChannel|DriverRequest|DriverResponse|DeferredResponse|FireHandle' runtime worker driver
rg 'ForwardResponse|ForwardOutput|dispatch_fired_batch|extract_per_request' runtime worker driver interface
rg 'InProcPollingChannel|InProcChannel|ShmemChannel|InProcVTable' runtime worker driver
rg 'ipc_profile|spin_budget_us|PIE_IPC_INCLUDE_DIR|DEP_PIE_IPC_INCLUDE|PIE_SCHEMA_INCLUDE_DIR' .
rg 'send_response|run_inproc|InProcService|InProcServer' runtime worker driver
rg 'SCHEMA_HASH|ArchivedRequestPayload|Pie(Frame|ResponseFrame|ForwardRequest|ForwardResponse)Desc|PieInProcRequestView|ResponseBuilder' runtime worker driver interface
```

All results must be zero, excluding migration history documents if intentionally retained.

Dependency gate:

```text
cargo tree -p pie-worker | rg 'pie-ipc'
cargo tree -p pie-driver-abi | rg 'rkyv|xxhash-rust|pie-driver-abi-derive'
```

Both commands must return no result.

## 18. Definition of Done

There is one normative completion criterion:
[direct_ffi_fix.md §6](direct_ffi_fix.md#6-revised-definition-of-done). Its
five-phase behavioral and structural gates subsume this document's original
transport-removal checklist.

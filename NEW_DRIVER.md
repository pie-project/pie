# NEW_DRIVER.md — the driver/cuda clean rewrite

> **Overview + implementation plan for `driver/cuda_new`.** This is the
> canonical architecture doc; the in-tree working checklist with live
> per-phase status lives in [`driver/cuda_new/PLAN.md`](driver/cuda_new/PLAN.md).

**Status (snapshot):** scaffold builds; **77 tests green** on an H100.

**Scope (frontier-only):** legacy archs are dropped (Llama-2/Qwen-2/Mistral/
Phi-3/OLMo-2, Gemma-2, Mixtral). Targets are the frontier set — MLA
(DeepSeek-V4/Kimi/GLM), modern MoE (Qwen3.5-MoE/GPT-OSS), Gemma-4,
Nemotron-H (Mamba), plus dense Qwen3.5/Llama-3/4 on the `llama_like` base.
Strategy: implement all frontiers upfront, then validate/fix/refactor.

**Frontier kernel foundation (lifted + standalone-validated; banked, not yet
wired):** MLA naive paged attention (`ops/mla_paged` — full CPU-parity), MoE
dispatch routing (`kernels/moe_dispatch`), Gemma-4 `rope_partial` + AltUp,
Mamba `causal_conv1d`, int4 WNA16 dequant. Remaining kernel gaps: MoE
per-expert GEMM, full `gated_delta_net` SSM, MLA cache-write, Gemma-4 misc.
Next: wire (CMake/ABI) → compose frontier forwards → validate/fix.

Two fronts are live in parallel:
- *Hot path (phase 1) — essentially complete.* All single-layer
  primitives are lifted behind the ABI with bf16 parity tests (`embed`,
  `rmsnorm`, `gemm`, `rope`, naive paged `attention`, `residual_add`,
  `swiglu`, `argmax`), and they **compose into a complete `llama_like`
  forward** — `pie_cuda_llama_forward_bf16`: embed → N decoder layers →
  final norm → lm_head → argmax — producing real next-token ids,
  validated vs an independent CPU f32 reference (logits parity + argmax
  validity). The activation workspace (`pie_ws_alloc`) is un-stubbed and
  real. Remaining is *generality + later phases* (next bullet).
- *Phase 2 — construction is end-to-end.* `builder.rs::build` takes a model
  directory and runs the whole flow with **no `is_*_arch` cascade**:
  `config.json` → `arch::detect` → `ArchSpec`; device probe → `mem::plan`
  (the real capacity decision — it correctly refuses when the budget is
  exceeded); `loader.rs` reads the safetensors checkpoint and uploads each
  tensor to a device buffer (fat-Rust weight binding — control plane owns
  loading, no C++ `pie_weights_bind`); KV + workspace allocated; and
  `Model::prefill_greedy` runs the forward → next-token ids. Verified by a
  bit-exact load round-trip and a `config dir → tokens` test.
- *General paged KV is wired into the forward.* It now uses the
  `write_kv_to_pages_bf16` scatter + a per-layer `num_kv_pages` stride
  (replacing the prefill-only contiguous copy); a multi-page ≡ single-page
  equivalence test confirms KV spanning pages is correct. This is the
  decode / multi-request storage path.
- *Real checkpoints + real sampling are wired.* The loader casts F16/F32
  tensors to bf16 on upload (`cast_*`; bit-exact RNE verified on an F32
  checkpoint), and `Model::prefill_sample` runs the forward then
  `sample_temp_bf16` (temperature/Gumbel-max; temp=0 ≡ greedy, deterministic
  for a fixed seed).
- *Primitives lifted + ABI-wired, ready to use:* `gather_bf16_rows`
  (compact logits), the Gemma kernels (`rmsnorm` 1+w, `geglu_tanh`,
  `logit_softcap`), FP8 (E4M3→bf16) dequant, **YaRN RoPE** (Llama-3
  long-context), and MoE building blocks (`topk_softmax`, `chunked_swiglu`).
  Remaining *wiring*: compact-logit path, a Gemma forward, a MoE forward.
- *Remaining (larger):* loader completion (quant, TP sharding, GGUF,
  real-checkpoint validation — reuse/extend `pie-weight-loader`);
  **transport / executor routing** (phase 3); flashinfer + the prepare/body
  two-phase (the perf gate).
- *Cold control-plane logic (phases 2–4, parallelized):* the pure-Rust
  cores of `mem`, `sampler`, `arch`, and `spec` are ported from their
  `driver/cuda` sources and unit-tested (≈3.2K LOC). Not yet wired to the
  executor (that waits on the hot-path ABI), but the logic is landed.

The legacy `driver/cuda` keeps running unchanged; `cuda_new` grows beside
it and replaces it incrementally.

---

## 1. Why rewrite

`driver/cuda` is a working, fast, ~70K-LOC embedded CUDA inference driver.
It is not broken — but two files concentrate most of the accidental
complexity and make it hard to evolve:

- **`src/entry.cpp::run_impl`** — a **1,286-line construction
  god-function** carrying **55 `is_*_arch` branches** over 10 arch
  booleans. *Execution* was already made polymorphic (the `IModel`
  interface); *construction* — weight binding, workspace allocation,
  memory planning — was left as one hardcoded if/else cascade.
- **`src/executor/executor.cpp`** — **3,363 lines, 264 file-local
  helpers**. The plain "run one forward + sample" path is buried under
  speculative-decode/MTP, CUDA-graph capture, and sampling concerns, all
  interleaved in one translation unit.

Plus two smaller leaks: the per-step workspace base type is named after
its first implementation (`IModel::body(Qwen3Workspace&, …)` — used even
for Gemma/Nemotron/DeepSeek), and `model/` is 23.5K LOC of hand-rolled
per-arch forward loops over a shared primitive vocabulary.

The **essential** complexity is real and must be preserved: three
attention paradigms (paged MHA/GQA, MLA, recurrent/Mamba), quant tiers
(FP4/FP8/WNA16/MXFP4), tensor parallelism, CUDA-graph decode capture,
spec-decode, paged KV. The rewrite's job is to **untangle the essential
from the accidental**, not to reimplement the kernels.

**North star: maintainability.** Not performance, not new features —
behavior- and perf-preserving by construction (hot kernel sequences are
lifted verbatim).

---

## 2. The decision — redraw the Rust/C++ line

> **The control plane (every *decision*) moves to Rust. The device
> library (every *kernel sequence*) stays C++. The hot forward body stays
> a single FFI call**, so the boundary is coarse-grained and FFI overhead
> is irrelevant (~5 calls per token-step, each 100s of µs–ms).

This matches Pie's existing split (the Rust runtime already owns
scheduling) and reuses the established interop machinery:

- **`driver/bridge`** (`pie-bridge`) — rkyv wire schema + shmem/inproc
  transport. The new control crate plugs into it exactly like
  `driver/dummy` does.
- **`driver/weight_loader`** (`pie-weight-loader`) — the Rust storage
  loader, already called *from* C++ today.
- **`server/build.rs`** — CMake → static lib → link into `pie-server`.

The codebase is already ~70% shaped for this line: `ForwardFn::Forward
Inputs` (`executor.hpp:81`) is already a POD bundle of device pointers +
ints — it crosses FFI almost verbatim as `PieForwardInputs`. `IModel` is
already the per-arch dispatch vtable. **We formalize a half-built seam
rather than invent one.**

```
            BEFORE (driver/cuda)                  AFTER (driver/cuda_new)
  ┌─────────────────────────────┐      ┌────────────────────────────────────┐
  │ Rust runtime: schedule+shmem │      │ Rust control plane  (control/)      │
  └──────────────┬──────────────┘      │  builder · mem · executor           │
                 │ InProcVTable         │  sampler · spec · tp · arch registry│
  ┌──────────────▼──────────────┐      └──────────────┬─────────────────────┘
  │ C++ entry.cpp run_impl       │                     │ flat C ABI (coarse)
  │ executor.cpp (3.3k)          │      ┌──────────────▼─────────────────────┐
  │ sampling · MTP · graph       │      │ C++ device library  (device/)        │
  │ + kernels/ops/caches         │      │  kernels · ops · per-arch body       │
  └─────────────────────────────┘      │  caches · graph mechanics · context  │
                                        └──────────────────────────────────────┘
```

---

## 3. Architecture

### 3.1 Layout

```
driver/cuda_new/
  PLAN.md                        ← in-tree working checklist (live status)
  device/                        ← C++/CUDA thin device library (libpie_cuda_device)
    CMakeLists.txt
    include/pie_cuda_device.h     ← the flat C ABI — the seam (load-bearing)
    src/
      context.{hpp,cpp}           ← PieDevCtx: device, stream, cuBLAS handle
      abi.cpp                     ← ABI entry points → C++ impl
      kernels/                    ← lifted verbatim from driver/cuda/src/kernels
        rmsnorm · residual_add · swiglu · rope · embed · argmax  (.cuh/.cu)
      ops/                        ← gemm (bf16 cuBLAS) · attention_naive_paged
      workspace.hpp               ← PieWorkspace (pre-allocated scratch)
      forward/                    ← llama_layer · llama_forward (full body)
      cache/                      ← lifted incrementally (phase 1→)
  control/                       ← Rust fat control plane (pie-driver-cuda-native)
    Cargo.toml  build.rs
    src/
      lib.rs        ← C-ABI entry (mirrors driver/dummy); wires builder→executor
      ffi.rs        ← raw extern "C" over pie_cuda_device.h (hand-written, audited)
      device.rs     ← safe RAII wrappers over ffi (Drop = destroy); bf16 helpers
      arch/mod.rs   ← trait Arch + ArchSpec + registry (replaces is_*_arch)
      builder.rs    ← replaces entry.cpp::run_impl
      mem.rs        ← replaces cuda_memory_planner.cpp
      loader.rs     ← safetensors → device (replaces loader/ + LoadedModel)
      executor.rs   ← replaces handle_fire_batch core
      sampler.rs    ← replaces sampling_dispatch.cpp + seed helpers
      spec.rs       ← replaces the MTP tangle in executor.cpp
      tp.rs         ← replaces rank-0 broadcast loop
```

### 3.2 C++ device library — "do what you're told"

Keep the internals; expose a flat C ABI over opaque handles. Design rules:

1. **Handles are opaque** (`PieDevCtx`, `PieWeights`, `PieKvCache`,
   `PieWorkspace`, `PieGraphExec`); lifetime is explicit (create/destroy).
2. **Argument structs are POD**, C-layout, stable. No C++ types cross.
3. **No exceptions cross** — every entry is `noexcept`, returns
   `PieStatus`, and sets a thread-local error string
   (`pie_cuda_last_error`).
4. **The prepare/body two-phase split is preserved** as two ABI calls
   (see §3.5).
5. Kernels/ops are **lifted verbatim** from `driver/cuda` — same code,
   same numerics; only the namespace and launcher decl change.

### 3.3 Rust control plane — absorbs every decision

| Module        | Replaces (in `driver/cuda`)                  | Owns |
|---------------|----------------------------------------------|------|
| `arch/`       | `bound_model.cpp` + 55 `is_*_arch` branches  | `trait Arch` + declarative `ArchSpec` |
| `builder.rs`  | `entry.cpp::run_impl` (1,286 lines)          | config→arch→plan→alloc; a table walk, no cascade |
| `mem.rs`      | `cuda_memory_planner.cpp`                    | forward capacity / KV pages / state slots (pure arithmetic + 1 probe) |
| `executor.rs` | `handle_fire_batch` core                     | classify batch, pick graph bucket, sequence upload→prepare→body→sample |
| `sampler.rs`  | `sampling_dispatch.cpp` + seed helpers       | sampling policy + RNG (kernels stay C++) |
| `spec.rs`     | MTP/spec-decode helpers in `executor.cpp`    | explicit draft/verify/repair state machine |
| `tp.rs`       | rank-0 broadcast loop                        | TP orchestration (all-reduce kernels stay C++) |

Adding an arch becomes a **directory-local change**: one `impl Arch` in
Rust + one `device/src/forward/<arch>` TU. No central file grows.

### 3.4 The `Arch` contract

Arch knowledge becomes *data + a small trait impl*, not branches:

```rust
trait Arch {
    fn spec(&self, cfg: &HfConfig) -> ArchSpec;        // dims, layer schedule, kv layout, quant
    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims;
    fn id(&self) -> PieArchId;                          // selects the C++ body
    fn caps(&self) -> Capabilities;                     // graph_safe, fused_argmax, ...
    fn drafter(&self) -> Option<Box<dyn Drafter>> { None }   // spec-decode opt-in
}
```

`ArchSpec` carries per-layer `LayerKind` (FullAttention / Sliding /
LinearAttention / Mla), per-layer head-dim overrides and KV-sharing maps
(Gemma-4), MoE expert count, and recurrent-state slots — everything the
builder needs to drive the C++ alloc primitives.

### 3.5 The load-bearing constraint: prepare/body two-phase

CUDA-graph decode capture (the steady-state perf win) requires that the
captured region do **no host-side work** and read all kernel args from
**pointer-stable buffers**. `driver/cuda` already splits the forward into:

- **`prepare`** (host-side): compute the flashinfer decode plan, upload to
  pinned/device buffers. Runs *every* fire, including before graph replay.
- **`body`** (device-side): the pure kernel sequence; re-reads the buffers
  `prepare` refreshed.

In the rewrite this becomes two ABI calls — `pie_prepare` then `pie_body`
(or `pie_graph_launch` on replay) — with Rust owning the *policy* (which
graph bucket, when to capture) and C++ owning the `cudaGraph`
*mechanics*. **This is the single most important design contract**, and
paged attention is the op that defines it (see §5, phase 1 tail).

---

## 4. The ABI (current surface, v3)

`device/include/pie_cuda_device.h`. Grows additively; the version gates a
runtime check in `device.rs::Device::new`.

```c
// lifecycle / introspection
uint32_t  pie_cuda_abi_version(void);
const char* pie_cuda_last_error(void);
PieStatus pie_cuda_ctx_create(int32_t ordinal, PieDevCtx** out);
PieStatus pie_cuda_ctx_destroy(PieDevCtx*);
PieStatus pie_cuda_mem_info(PieDevCtx*, size_t* free, size_t* total);

// raw device memory (generic plumbing for ops/tests/input staging)   ── implemented
PieStatus pie_cuda_malloc / free / memcpy_h2d / memcpy_d2h / stream_sync(...);

// lifted kernels (verbatim from driver/cuda/src/kernels)             ── implemented
PieStatus pie_cuda_rmsnorm_bf16(...);
PieStatus pie_cuda_residual_add_bf16(...);
PieStatus pie_cuda_swiglu_bf16(...);

// construction primitives (Rust builder drives; impl ported phase 1) ── stubs
PieStatus pie_weights_bind / pie_kv_alloc / pie_ws_alloc / pie_kv_page_bytes(...);

// hot path — one coarse call each                                    ── stubs
PieStatus pie_upload_inputs / pie_prepare / pie_body / pie_sample(...);

// CUDA graph — Rust policy, C++ mechanics                            ── stubs
PieStatus pie_graph_capture / pie_graph_launch / pie_graph_destroy(...);
```

Stubs return `PIE_ERR_INTERNAL` with a message naming the `driver/cuda`
source each will be ported from — the seam is honest about what's wired.

The Rust face: `ffi.rs` is a hand-written (not bindgen'd) extern block —
small, curated, the single auditable place for the boundary. `device.rs`
wraps it in safe RAII types (`Device`, `DeviceBuffer<'a>`, …) where `Drop`
calls the destroy entry, and `PieStatus` is translated to `Result` with
the device-lib error string attached.

---

## 5. Implementation plan — strangler-fig

Each phase ships independently and keeps all archs working.

- **Phase 0 — scaffold.** ✅ Tree, ABI header, crate skeleton, build
  wiring. `control/` in root `Cargo.toml` `exclude`; nothing in the real
  build changes.
- **Phase 1 — carve the seam.** *(in progress)* Implement
  `libpie_cuda_device` by lifting `kernels`/`ops`/`cache` behind the ABI,
  and route the *current* C++ executor through it. Zero behavior change.
  - ✅ device-memory ABI + safe `DeviceBuffer` + bf16 host helpers.
  - ✅ full single-layer primitive set, bf16 parity tests: `embed`,
    `rmsnorm`, `gemm` (cuBLAS `act @ Wᵀ`), `rope`, naive paged
    `attention` (`attention_naive_paged_bf16` — no flashinfer / no
    two-phase), `residual_add`, `swiglu`, `argmax`.
  - ✅ composed `llama_like` decoder layer (`forward/llama_layer.cu`).
  - ✅ **complete forward** `pie_cuda_llama_forward_bf16`
    (`forward/llama_forward.cu`): embed → N layers → final norm → lm_head
    → argmax → next-token ids; validated vs an independent CPU f32
    reference (logits parity + greedy-token validity).
  - ✅ real `pie_ws_alloc` — pre-allocated `PieWorkspace` scratch, reused
    across layers (replaces the layer slice's per-call scratch).
  - ◻ real KV-append scatter (multi-request / decode / multi-page; the
    slice uses a contiguous copy for single-request prefill).
  - ◻ flashinfer + the prepare/body two-phase — the perf gate, deferred.
  - ◻ assemble a minimal `pie_body` for `llama_like`; route one arch
    through it.
- **Phase 2 — move construction.** Port `run_impl` → `builder.rs`, one
  arch at a time.
  - ✅ `mem.rs` — `plan_cuda_memory` ported (budget split, (N,R) lattice,
    per-profile scoring, KV-page solve) over primitive inputs; 15 tests.
  - ✅ `arch/` — `HfConfig` + `detect()` + `LlamaLikeArch`/`Qwen3Arch`
    `ArchSpec` derivation, ported from `bound_model.cpp`; 10 tests.
  - ✅ `loader.rs` — minimal safetensors loader: checkpoint → device
    buffers → full forward (`LoadedLlama`); bit-exact round-trip test.
    Slice: BF16 only; completion (F16/F32, quant, sharding, GGUF) reuses
    `pie-weight-loader`.
  - ✅ `builder.rs` — `build(model_dir)` ties `arch` + `mem` + `loader`:
    config → spec → plan → load → alloc → `Model::prefill_greedy`. Verified
    `config dir → next-token ids`. (Replaces `run_impl`'s 55-branch cascade.)
- **Phase 3 — move the loop + sampling.** `handle_fire_batch` core →
  `executor.rs` + `sampler.rs`. MTP stays C++ temporarily, called as a
  unit. `tp.rs` follows.
  - ✅ `sampler.rs` — `splitmix64` (KA-verified vs C++), seed schedule,
    greedy detection, per-row param SoA builder; 12 tests.
  - ✅ `executor.rs` — graph-bucket lattice (1 test). ◻ fire loop (ABI).
- **Phase 4 — move spec-decode.** Reimplement the MTP state machine in
  `spec.rs` (the frozen-verify + batched-repair design); the
  `executor.cpp` tangle finally dissolves.
  - ✅ `spec.rs` — explicit Draft/Verify/Repair/CommitAdvance state
    machine + acceptance math + adaptive draft-count, ported from the
    `executor.cpp` MTP helpers; 14 tests. ◻ device-driving (ABI, phase 4).
- **Phase 5 (optional) — forward dedup** in C++ (see Non-goals).

### Cutover (build integration)

- `server/build.rs::build_cuda()` points CMake at
  `driver/cuda_new/device` and links `libpie_cuda_device`; `control/`
  joins the workspace `members` and links like `driver/dummy`.
- `server/src/driver_ffi.rs` keeps `pie_driver_cuda_run_inproc`. During
  migration the new crate exports `pie_driver_cuda_native_*` so it
  coexists with the live driver behind a `driver-cuda-new` Cargo feature;
  at cutover it takes over the canonical names.

---

## 6. Non-goals
- **Not** a kernel rewrite or flashinfer/cutlass replacement.
- **Not** a forward "framework." ML forwards are full of per-arch quirks
  (Gemma pre/post-norm, qk-norm, softcap, altup, sliding window, MLA,
  Mamba recurrence). The realistic dedup is a shared `decoder_forward`
  skeleton with arch-provided block hooks, plus *separate* skeletons for
  MLA and Mamba. Explicit over clever. v1 lifts most bodies directly.
- **Not** a perf project — parity is the bar.

## 7. Risks
- **FFI grain** is the load-bearing assumption — phase 1 confirms ~5
  coarse calls/token-step is free (it is; each is 100s of µs–ms).
- **Graph capture across FFI** needs the persistent input buffers
  C++-owned and pointer-stable (they already are); Rust calls
  `upload_inputs` each fire.
- **Paged attention** is the real complexity gate, not the pointwise
  kernels — it's where `prepare`/`body` stop being stubs.

## 8. Build & test

```bash
cd driver/cuda_new/control
CUDACXX=/usr/local/cuda/bin/nvcc \
  CMAKE_CUDA_ARCHITECTURES=90 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo test
# build.rs drives the CMake device build; the parity tests run on the GPU.
# (nvcc lives at /usr/local/cuda/bin but isn't on PATH; sm_90 SASS is
#  needed to actually run on the H100 — sm_80 compiles but won't load.)
```

Current: **77 tests pass** (kernel/op + composed-layer + full-forward +
multi-page-KV equivalence + safetensors BF16/F32 load/forward +
`config dir → tokens` builder incl. temperature sampling + KV-scatter /
sampling / cast / gather / Gemma / FP8 / YaRN / MoE parity on the GPU +
52 pure-logic unit tests).

## 9. Backend portability — two seams, and where the forward lives

Two distinct "backend" boundaries exist at different altitudes. Conflating
them is the main portability trap.

| Seam | Altitude | Contract | Who plugs in |
|---|---|---|---|
| **Driver** (high) | whole-model | `pie_driver_*_run_inproc` + `Flavor` (`server/driver_ffi.rs`) | *engines* that own load+forward+KV+fusion+sampling: **ggml/gguf (`driver/portable`)**, vLLM, SGLang, TRT-LLM |
| **Device** (low) | per-body | the `pie_*` ABI (internal to cuda_new) | *kernel libraries* sharing cuda_new's Rust control plane: CUDA (now), Metal/MPS, ROCm |

### Decision: the forward lives in C++, not Rust

The per-arch forward composition (`forward/*.cu`) stays *below* the device
seam. **C++ owns the forward's definition *and* execution** — op order,
fusion, graph capture, scheduling. **Rust is a pure control plane**: it
*drives* forwards through the coarse seam (`prepare` / `body` / `sample`)
but never *defines* them.

Why:
- Graph capture + fusion want the body as one capturable C++ unit; per-op
  FFI from Rust would be thousands of crossings per token and uncapturable.
- The maintainability goal (kill `run_impl`'s god-function + 55-branch
  dispatch) is already met by Rust arch-detection + clean per-arch C++
  files — it never depended on moving the forward into Rust.
- Every engine backend already owns its forward; cuda_new owning its
  forward is consistent.
- **Accepted cost:** a new arch is a two-language change (a Rust
  `impl Arch` for detect/spec/dims + a C++ `<arch>_forward`), and forwards
  are not shared across kernel backends.

### Op granularity / fusion is a below-seam concern

Because the hot-path contract is coarse (`pie_*_forward`, one call), each
backend composes the forward from *its own* op set and fuses however it
likes (CUDA already has a fused `residual_add_rmsnorm`; Metal would fuse in
its encoder). The fine-grained per-op ABI entries (`rmsnorm_bf16`, …) are
**bring-up scaffolding + the CUDA lib's building blocks + the parity-test
surface — not the cross-backend contract.** A per-op contract is the one
thing that would *prevent* fusion across the boundary.

### Capabilities are runtime, not Cargo features

Cargo features gate *what is compiled in* (which backends; optionally which
archs, for binary size). What a backend *can do* (fuse, capture graphs,
fp8) is a **runtime capability query + an always-correct fallback** — so
one binary adapts to the GPU it lands on. Never gate a forward's fast/slow
path behind a compile-time feature.

### ggml/gguf is an engine, not a kernel library

It belongs at the **driver seam as a peer** — already wired via
`driver/portable` + the `Flavor` enum — *not* under cuda_new's op ABI.
Driving ggml op-by-op would fight its design and throw away its fusion,
scheduling, and KV management.

### Tripwire to revisit

Build an op-sequence IR (Rust defines the forward; backends fuse/execute it)
— or a shared C++ decoder skeleton for the homogeneous llama-like family —
**only** when a second kernel-library backend lands *and* its forwards
prove similar enough to share without per-backend conditionals. Until then
it is YAGNI: the IR is the deferred price of cross-backend forward sharing,
a goal not yet in scope.

## 10. Dependency & build-time policy

**Allowed external deps: foundational header-only libraries only — flashinfer
and cutlass. Everything else we write or port ourselves** — no vendored
"dirty" code, no build-time generators, no template-shape matrices. The two
things that dominate the old `driver/cuda` build time are exactly what we
exclude: flashinfer's CUTLASS-MoE Python codegen (`generate_gemm_operations`)
and Marlin's `ALL_SHAPES` matrix.

`cuda_new`'s device lib carries none of that today (no CPM/flashinfer/codegen/
Marlin; compiles in seconds). Rules:

1. **flashinfer / cutlass headers** may be `#include`d, but we **hand-write
   only the specific template instantiations the frontier set needs** (our
   `(head_dim, dtype, arch)` / GEMM shapes). No generator, no "all shapes."
2. **No vendored dirty code — internalize cherry-picked instead.** Where
   upstream ships a generator or a shape-matrix kernel, we lift only the
   specific kernels/instantiations we use, **de-brand** them (functional
   names, no upstream branding), drop the generator, and own them as our
   `.cu`. Specifically the **quantized-weight GEMM is an internalized,
   cherry-picked, de-branded Marlin** — a *unified* fused quant matmul
   covering int4 (GPTQ `u4b8` / AWQ `u4`), int8, **mxfp4** (`fe2m1f`), and
   **fp8 weights** (`fe4m3fn`). Cherry-pick = only our (act=bf16, our weight
   formats, scale=bf16) + a small set of blocking tiles (not the full matrix)
   + the needed `{gptq,awq}_repack`; drop `generate_kernels.py`.
3. **Avoid dequant-to-bf16 in the hot path** — materializing bf16 weights
   discards quantization's whole point (memory + bandwidth). Quantized weights
   go through the internalized-Marlin fused quant GEMM (rule 2). **cutlass** is
   for the jobs Marlin doesn't cover — **MoE grouped GEMM** and (with
   flashinfer) attention — via minimal in-house instantiations. The
   `dequant_*` kernels are kept ONLY as (a) the correctness **oracle** for the
   fused kernels (fused ≡ dequant-then-bf16-GEMM) and (b) an
   absolutely-necessary fallback.

Effect: builds stay fast/deterministic, the tree stays generator-free, and
quantized models keep their memory/bandwidth win.

**Why Marlin and not cutlass for the quant GEMM (2026-05-30 finding).** A
spike tried to get the fused W4A16 (int4 weights, bf16 act) path out of
CUTLASS 4.1's SM90 mixed-input `CollectiveBuilder`. Every int4 mixed-input
config failed the same way: the canonical GMMA descriptor wants a smem K-tile
of exactly 2–4 `uint128_t` per atom, but the SM90 builder doesn't wire the
memory-reorder + matching `SmemLayoutAtom` that example 55 (`mixed_dtype_gemm`)
overrides by hand — so cutlass's *builder* alone can't produce a working int4
kernel, and the only "shipping" cutlass fallback is dequant-prologue + bf16
GEMM, i.e. exactly the dequant-in-hot-path we're banning. Marlin's hand-written
SM80-style `mma.sync`/`ldmatrix` kernels (which run fine on SM90) already solve
mixed-input dequant-in-register, so the internalized-Marlin path (rule 2) is
the right primary, and cutlass stays scoped to grouped-MoE + attention.

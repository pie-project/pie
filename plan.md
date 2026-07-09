# Plan: PTIR layering refactor, device-geometry activation, boundary migration

Status date: 2026-07-08, as of commit `243f5b1e` (branch `dev`).
This document contains ONLY remaining work. As-built history (the beam-hardwire
workstreams W0-W3, the `pie_sampling_ir` -> `pie-ptir` migration, the executor
sampling_ir removal, boundary Phase 0) lives in git: commits `4e2bf2e4`,
`33d038bc`, `f2936467`, `243f5b1e`, and this file's own history.
Companion spec: [boundary.md](boundary.md) is the runtime-driver boundary
contract of record (decisions B1-B17, word layout, wake protocol); Track C
below carries only its actionable migration phases.

Three tracks. A and C are independent; B is rooted at A.

```
Track A (SDK/WIT layering)   A1 -> A2 -> { A3, B1 } ;  A4 after A3 ;  A5 parallel
Track B (device geometry)    B1 -> B3 -> B4 -> B5 ;  B2 parallel with B1
Track C (boundary)           C1 -> C2 -> C3   (C2 touches ptir_host: sequence with B3)
```

---

## 0. Where things stand (one paragraph per track)

**A.** The IR crate is clean and final in shape: [interface/ptir](interface/ptir)
(crate `pie-ptir`: op table, `TraceContainer` + canonical encoding,
`registry::{Port, Stage}`, `validate::bind`, `interp`, sidecar). The SDK crate
`sdk/rust/ptir` still squats on the `ptir` name and mixes the tracing eDSL with
runtime objects that duplicate the host side of the WIT contract
(`ForwardPass::trace` binds guest-side at
[forward.rs:417](sdk/rust/ptir/src/forward.rs#L417); `WorkingSet` is a counter
stub; `Pipeline` memoizes and does nothing). The WIT surface
([ptir.wit](interface/inferlet/core/wit/ptir.wit): `channel`,
`forward-pass.new(container-bytes, channels, kv-ws, rs-ws)`, `pipeline`) and the
host behind it (non-blocking run-ahead submit, poison, pipeline-owned FIFOs,
global channel ids, release markers) are built and verified; nothing guest-side
drives them yet.

**B.** All beam-deletion *mechanisms* are built and GPU-verified: global device
channel registry, pre-forward descriptor resolver wired into the executor
(`resolve_descriptors` at
[executor.cpp:2556](driver/cuda/src/executor/executor.cpp#L2556), resolved
`FireGeometry` feeds the active batch views, dense mask packed via
`pack_dense_mask` at executor.cpp:2791-2827), tier-0 launcher coverage,
`PageLease` (unit-tested, unwired), `map_geometry_relaxed`. What does not exist
is the device-geometry *program* and its submission path: `detect_beam` /
`fire_beam` / `BeamRun` / [ptir_beam.rs](runtime/src/ptir/ptir_beam.rs) (the
host replay, and the sole home of the locked golden vectors) are still live.

**C.** [boundary.md](boundary.md) Phase 0 (pie-waker module split) is committed.
Phases 1-3 (accidental-complexity deletion, real frame layout + device word
publish, activation) are pending.

---

## Track A: PTIR SDK/WIT layering refactor

### A.0 Target architecture

Three layers, one dependency direction:

```
interface/ptir      crate pie-ptir     the IR (the multi-party contract)
sdk/rust/ptir-dsl   crate ptir-dsl     tracer + eDSL (rename of sdk/rust/ptir)
sdk/rust/inferlet   crate inferlet     WIT surface + author-facing objects

pie-ptir  <-  ptir-dsl  <-  inferlet  <-  inferlet programs
    ^
    +-- runtime (host bind/registry), echo, driver-side C++ mirror
```

- **`pie-ptir`**: unchanged in role. New in this refactor: `stage_key` (A5).
- **`ptir-dsl`**: the embedded authoring language. Closures in, container bytes
  out. No WIT, no wasm deps, host-`cargo test`-able, standalone. Tracing is its
  implementation strategy, not its identity, hence the name (`ptir-trace`,
  `ptir-capture`, `ptir-staging` rejected: mechanism-naming ages badly,
  "capture" collides with forward-capture in the overview, "staging" collides
  with `Stage`).
- **`inferlet`**: the only home of the author-facing overview §3 surface
  (`ForwardPass`, `Pipeline`, `WorkingSet`, `Channel` with host
  `put`/`take().await?`). Wraps WIT resources, drives the ptir-dsl builder.
  wasm32-wasip2 only; owns the single `wit_bindgen::generate!`.

### A.1 Ratified design decisions

Debated and settled; recorded so they do not get relitigated.

**D1. `ForwardPass`/`Pipeline`/`WorkingSet` are boundary objects, not DSL
objects.** They name host-owned lifetime and semantics (submission ordering,
slot allocation, bind identity). The DSL crate must not export them. The author
still writes the overview surface verbatim; it lives in `inferlet`.

**D2. The WIT `forward-pass` is one function, deliberately.**
`new(container-bytes, channels, kv-working-sets, rs-working-sets)`. No builder
methods (`embed`, `epilogue`, ...) at the WIT level:

- WIT cannot carry closures; stage bodies cross as bytes no matter what.
- Stage ops reference channels by dense index internally; moving ports out of
  the bytes does not remove the index table, it splits references in two styles.
- Identity (C3 hash) needs one canonical artifact.
- Validation is whole-pass; granular builders defer to a seal step anyway.

The WIT pass resource is "identity + binding": decode, hash-dedup, bind against
the ModelProfile, hold the handle table. Execution belongs to `pipeline` (the
ordering domain). Everything per-step flows through `channel` resources. Seeds
are not `new` arguments; a `put` staged on a seeded channel before the first
submit is the seed.

**D3. eDSL ops and intrinsics are IR vocabulary, not WIT.** Ops record IR
nodes; they execute nowhere. WIT typing is shallower than the Rust eDSL's
(Tensor vs Channel misuse is a Rust type error; through WIT everything degrades
to handles). Intrinsics split three ways: stage values (`logits()`, `query()`,
`layer`) and second-party kernels are pure IR ops (kernel availability is a
bind-time property with fallback-as-different-program; WIT funcs would make a
component naming a missing kernel fail at instantiation, foreclosing the
fallback); model constants stay symbolic in the trace where the IR supports it,
resolved at host bind, so one program hashes identically across backends; guest
host-code needing concrete values (`ws.alloc(div_ceil(len, page_size))`)
fetches them once at startup from the WIT `model` interface, replacing the
`ptir::model::configure` global (which survives only for native tests).
Verification story: Rust types at compile time, shared-crate lints at trace
time guest-side, authoritative `bind` at `forward-pass.new` host-side; lint and
bind live in `pie-ptir`, so guest and host run the same rules with zero drift.

**D4. The container stays one monolithic canonical artifact.** Rejected:
exploding ports/decls into WIT records with op streams as blobs. The canonical
encoding cannot be eliminated (the C3 hash needs stable bytes, which WIT's
canonical ABI does not provide, and the runtime -> driver ship is a process
boundary with no WIT), so structured WIT would mean two representations plus
reassembly glue. Port/stage/extern vocabulary evolves with the IR: in the
container that is a wire-version byte (v1.1 externs encode as v1
byte-identically when empty); in WIT, enum/variant extension is world-breaking,
multiplied by the four synced WIT copies. Precedent: SPIR-V under Vulkan, PTX
under CUDA, module bytes inside the component model. The current design is
already the right hybrid: lifetime-crossing objects are WIT resources,
trace-known data rides the artifact. Rule of thumb: WIT surface scales with
runtime interaction points, never with language vocabulary. Mitigation for blob
legibility: a `pie-ptir` disassembler (unscheduled, see A.6).

**D5. Identity is two-level.** The overview §5.3: "a pass's identity is the
tuple of its stage traces, and instances co-batch stage by stage". The C3
container hash is the pass-level key (dedup, compile/bind cache, steady-state
submit, driver ship; what
[ptir_registry.rs](runtime/src/ptir/ptir_registry.rs) keys its LRU by).
Per-stage batch keys are derived, host-side, post-bind: raw byte-equality of a
`StageProgram` section does not capture "same epilogue" across passes (ops
reference the pass-global dense channel table), so the key is a normalization:
renumber channel refs in stage-local first-use order, attach referenced-channel
signatures and resolved types from the sidecar, hash the canonical form. This
refactor only cuts the seam (A5); scheduler use is future work. Batch keys
never cross the wasm boundary, so this is orthogonal to D4.

**D6. The guest does not bind.** Remove `bind(container, model::profile())`
from the guest trace path; keep span lints. Authoritative validation is
`forward-pass.new`'s result. `bind` + interp remain for native parity tests
behind dev-deps/features.

**D7. The classic `forward-pass` (inference.wit) is retired, not reshaped.**
Every func on it is a remnant of the pre-channel model, where the host replayed
the decode carry. Mapping:

| classic func | PTIR replacement |
|---|---|
| `input-tokens(tokens, positions)` | `embed` port (toks channel + indptr) + `positions` port |
| `attention-mask(list<brle>)` | `attn-mask` port (bool channel, packed on the wire); BRLE dies |
| `output-speculative-tokens(flag)` | `readout` port |
| `next-inputs` / `next-attention-mask` / `set-pipeline-source-kind` / `fresh-generate` | deleted with nothing in their place: loop-carried channel ping-pong IS the carrier (`tok.put(t)` is next-inputs; a `prev_drafts` channel is the drafts carrier-kind; no host-pending carrier state exists to need fresh-generate) |
| `kv-working-set(set, inp-start, ...)` scalar geometry | `attn_working_set` port family (pages/page_indptr/kv_len/w_slot/w_off channels) |
| `execute` | `pipeline.submit` |
| `input-image` / `input-audio` | SURVIVES in shape: resources cannot ride in bytes, so media splice becomes embed-family ports plus parallel handle lists on `new` (e.g. `images: list<borrow<image>>`). Future additive change. |
| `rs-fold-buffered(tokens)` | OPEN: rs descriptor port vs channel-fed (A.7) |

`grammar`/`matcher` stay: `matcher.mask()` produces the words the guest puts
into the grammar-mask channel on both paths.

**D8. Naming: `ptir-dsl`.** `pie-ptir` defines the language, `ptir-dsl` is its
embedded surface, `inferlet` is the runtime binding.

### A.2 Code disposition (current `sdk/rust/ptir/src/`)

| current | disposition |
|---|---|
| `value.rs`, `intrinsics.rs`, `dtype.rs` | stay in `ptir-dsl` (eDSL surface) |
| `context.rs` (session recorder, channel interning) | stays |
| `channel.rs` | split: trace-side declaration + endpoint-span recording stays; host transport goes to the `inferlet` wrapper |
| `lint.rs`, `error.rs` | stay |
| `forward.rs` port setters (`embed`/`positions`/`attn_working_set`/`attn_mask`/`readout`) + `Indptr`/`AttnWsArgs` sugar | move to `inferlet` `ForwardPass`; internally call the neutral builder |
| `forward.rs` stage attachments + trace-once cache | move to `inferlet` `ForwardPass` (lifecycle owner) |
| `forward.rs` `assemble()` machinery: gid re-key, HostRole derivation, terminal-output inference, reader auto-drain drop, decl building | stays as `builder.rs`: `bind_port(Port, PortSource)`, `stage(Stage, closure)`, `build() -> Traced` |
| `forward.rs` `bind(...)` call + `TracedForward::bound()` | deleted from guest path (D6); `Traced` = container bytes + dense-order channel identities + names |
| `forward.rs` `WorkingSet`/`SlotGrant`/`Remap` | deleted; `inferlet` wraps `kv-working-set` (`alloc-slots`/`free-slots`, real stable slot ids) |
| `pipeline.rs` | deleted; `inferlet` wraps the WIT `pipeline` |
| `model.rs` | demoted to test-only; builder takes model constants as inputs; guest uses the WIT `model` interface |

### A.3 Steps

**A1: carve `ptir-dsl` out of `sdk/rust/ptir`.** — **DONE** (commit follows).
Crate renamed (`sdk/rust/ptir` → `sdk/rust/ptir-dsl`, package/lib `ptir_dsl`);
neutral `builder.rs` extracted (`Builder::{bind_port, stage, build, debug_container}`,
`PortInput`, `Traced`); `ForwardPass`/`WorkingSet`/`SlotGrant`/`Remap`/`pipeline.rs`
and the guest `bind` call deleted (D6: `build()` lints only, native parity tests
bind explicitly). Byte-identity verified: the §3/§6.2/§6.1 goldens hash-match and
the §3 container is sha256-identical to the pre-A1 bytes. Consumer inferlets
(`beam`, `mtp-grammar`) repointed at `ptir-dsl` but still author via `ForwardPass`
→ migrated in A3.
1. `git mv sdk/rust/ptir sdk/rust/ptir-dsl`; rename crate. Fix path deps
   (`rg -l "sdk/rust/ptir\b"` over Cargo.tomls; check `runtime/tests`, executor
   crates).
2. Extract `builder.rs` from `forward.rs::assemble()`/`record()`. Keep the gid
   re-key, HostRole derivation, terminal-output inference, and reader
   auto-drain drop verbatim (subtle; see A.5). `build()` runs lints only.
3. Delete `WorkingSet`/`SlotGrant`/`Remap`, `pipeline.rs`, the `bind` call,
   `ForwardPass` itself. Keep a `debug_container()` equivalent.
4. Gate `bind`/interp parity tests behind dev-deps (the `eval` feature dep
   exists). Container-hash goldens must not move: byte-identical output is the
   acceptance test for the extraction.
5. `model.rs`: explicit constants passed to the session/builder; test-only
   `configure` shim if it reduces churn.

**A2: the `inferlet::ptir` bridge.**
1. New module `sdk/rust/inferlet/src/ptir/` (namespaced: the classic
   `ForwardPass` wrapper exists elsewhere in the crate until A4).
2. `Channel`: owns the trace declaration and the WIT `channel` resource,
   constructed from the same `(shape, dtype, capacity)` so decl validation in
   `forward-pass.new` passes by construction. In-program use delegates to the
   trace side. Host-side `put` records the host-endpoint span on the trace side
   AND forwards to the resource (the span feeds HostRole derivation, A.5).
   `take()`/`read()` are async, surface poison as `Err`.
3. `WorkingSet`: wraps `kv-working-set` (`alloc(n)` -> `alloc-slots`, `free` ->
   `free-slots`, `page_size()` from model/resource). `SlotGrant` stays puttable
   into channels as data.
4. `ForwardPass`: verbatim overview surface; holds closures + port specs; on
   first `submit` drives the builder, orders WIT channel handles by
   `Traced.channel_order`, calls `forward-pass.new(...)`, memoizes the
   resource. Bind errors surface here.
5. `Pipeline`: wraps WIT `pipeline`; `submit` ensures the bound pass exists
   then delegates; `close()` delegates.
6. `inferlet::ptir::prelude` re-exports the eDSL plus the four wrapper types.

**A3: migrate test inferlets** (`runtime/tests/inferlets/ptir-sdk-greedy` and
other PTIR-path tests) to `inferlet::ptir::prelude`. Native goldens stay in
`ptir-dsl`/`pie-ptir`.

**A4: retire the classic forward-pass** (separate PR series). Once the bridge
covers the test matrix: delete the classic `forward-pass` resource from
`inference.wit` (keep `grammar`/`matcher`), the host implementation, the
carrier machinery (`pipeline-source-link` accumulate, retained buffers), the
SDK modules built on it (`carrier.rs`, classic parts of
`sampler.rs`/`prefill.rs`/`program.rs`), and the driver's remaining classic
carrier code (the `sampling_ir/{next_input,tensor_io,frame_carrier}.hpp`
includes and the `NextInputLink` injection around
[executor.cpp:3079](driver/cuda/src/executor/executor.cpp#L3079); note
`frame_carrier` is ALSO Track C's X2 mechanism, so only the classic-carrier use
dies here, coordinate with C). Blocked on: media splice ports (D7) or an
explicit decision to keep classic alive only for multimodal inferlets interim.

**A5 (parallel, small): `stage_key` in `pie-ptir`.** — **DONE** (commit follows).
Implemented the D5 normalization in [interface/ptir/src/stage_key.rs](interface/ptir/src/stage_key.rs)
(`stage_key(bound, stage)` / `stage_key_at`): stage-local first-use channel
renumbering, referenced-channel resolved signatures + resolved SSA value types,
FNV-1a over a `PTSK`-domain-separated canonical form. Golden tests green (same
epilogue in two passes → equal keys; one-op diff → different keys; differently
typed referenced channels → different keys). Sidecar carriage of per-stage keys
left as the documented follow-up (A.6 item 4).
Original: implement the D5
normalization with golden tests (same epilogue embedded in two different passes
yields equal keys; a one-op difference yields different keys). Optional:
sidecar carriage of per-stage keys.

### A.4 WIT copy sync

`interface/inferlet/core/wit` is the source; synced copies at
`interface/inferlet/deps/core/`, `sdk/rust/inferlet/wit/deps/core/`,
`sdk/tools/bakery/src/bakery/wit/deps/core/`. Any WIT change lands in all
copies in the same commit.

### A.5 Invariants and gotchas

- **Dense channel order.** The recorder interns channels in first-reference
  order but the container is re-keyed to gid (declaration) order, remapping
  every `ChanTake`/`ChanRead`/`ChanPut` and `PortSource::Channel`
  ([forward.rs:295-330](sdk/rust/ptir/src/forward.rs#L295)). The bridge's
  handle list must follow exactly this order; `Traced.channel_order` is the
  contract between builder and bridge.
- **HostRole derivation depends on pre-trace host endpoint spans.**
  `host_role = Writer` requires a host `put` recorded before the trace runs
  (the primed `mask_0` put in the software-pipelined loop). A host `take`
  typically happens only after the first submit, which is why terminal-output
  inference exists (program-produced, no program consumer, no descriptor use,
  no host put, not seeded => Reader;
  [forward.rs:349-367](sdk/rust/ptir/src/forward.rs#L349)). Preserve both
  rules; the bridge wrapper must keep recording host endpoints on the trace
  side even though the transport is the WIT resource.
- **Reader auto-drain drop.** The tracer's auto-drain `ChanTake` for
  host-Reader channels must be dropped or `bind` flags SecondConsumer
  ([forward.rs:379-397](sdk/rust/ptir/src/forward.rs#L379)).
- **Seeds.** `Channel::from(v)` = constructor + staged put; `seeded` today is
  `st.seeded || (has_host_put && has_prog_put)`; first submit with a missing
  seed errors host-side.
- **Hash stability.** A1 must be a pure motion refactor for the container
  bytes: same DSL program, byte-identical container, same C3 hash.
- **Single `wit_bindgen::generate!`** (in `inferlet`). `ptir-dsl` must never
  depend on wit-bindgen or generated types.
- **`ptir-dsl` stays wasm-free and workspace-standalone**; `pie-ptir`'s `eval`
  feature stays dev-only.
- **`pipeline.submit` must never block** (owner constraint). The bridge submit
  is enqueue-only; `take`/`read` are the await points.

### A.6/A.7 Open questions

1. rs-fold home: rs descriptor port family vs a channel the pass drains.
   Decide before hybrid/linear-attention models migrate off classic.
2. Media splice timing: embed-family ports + parallel handle lists on `new`,
   needed before classic deletion (or accept interim classic for multimodal).
3. How far to push symbolic model constants: `ChanDType` supports a
   model-intrinsic dtype; symbolic dims (`[vocab]`-shaped channels) may need IR
   support to keep hashes backend-portable. Interim: concrete dims from the
   WIT model interface, accepting per-backend hashes.
4. Sidecar carriage of stage keys (A5): runtime-derived only, or shipped.
5. `pie-ptir` disassembler/text format (D4 mitigation): unscheduled.

---

## Track B: device-geometry activation and beam deletion endgame

Owner constraints (unchanged): `pipeline.submit` never blocks; the inferlet
does all heavy lifting, the driver is dumb and general (no algorithm names in
driver changes); no beam-named files/symbols in `driver/cuda` at the end; a
channel is bindable to multiple passes.

### B.0 Target contracts still to build against

- **Ports carry wire-form geometry; the program does the math.** CSR-prefix
  contract: for CSR port pairs the indptr port's last element defines the valid
  prefix of the data port; channels keep trace-known fixed shapes
  (`pages: [B*P]`), the program densely packs live entries at the front
  (`ScatterSet` at `CumSum`-derived destinations), the driver reads
  `page_indptr[B]` entries and ignores the rest. `KvLen` = physical span;
  `last_page_len = ((len-1) % page) + 1` stays in the generic mapper.
  `WSlot`/`WOff` = explicit physical write descriptors. AttnMask = dense
  u8/bool channel, BRLE remains host-wire-only compression.
- **Physical-space page ids everywhere** (no slot->physical table). The
  runtime seeds fire 0's pages and grants per-fire fresh pages as physical ids.
  Correctness rests on the freeze discipline (frozen pages never rewritten;
  forks write only into fresh grants; the heir continues the tail in place).
  Cheap guard (still undecided, owner leaned yes): driver errors a fire whose
  `w_page` is outside the pass's leased set.
- **Device-geometry fires are solo**: wire geometry ships empty, fire marked
  solo/prebuilt (scheduler `prebuilt` flag,
  [scheduler.rs:698](runtime/src/scheduler.rs#L698)); not-ready descriptor =
  fire error (implemented in the resolver), surfaced through poison.
  Scheduler/workspace sizing uses static bounds from the container's channel
  decls.

### B.1 (ROOT) Author the device-geometry beam program on the new bridge

Blocked on A1+A2. A guest-authored beam epilogue tracing the geometry to
wire form in-graph, submitted end to end through
`forward-pass.new`/`pipeline.submit`:

- `page_indptr` = `CumSum(np)` with a leading 0 (`ScatterSet` into `[B+1]`
  zeros at `Iota` destinations); packed live pages = `ScatterSet` of the
  `[B,P]` matrix's live entries into a `[B*P]` channel at `CumSum`-derived
  destinations; `klen`, dense `kvm` (bound to the `AttnMask` port), `w_slot`,
  `w_off`, `out`, `out_par` as already traced in the driver golden
  (`beam_epilogue` in `ptir_golden_exec_test` proves the op coverage);
  `EmbedIndptr` stays const `[0..=B]`.
- All ids physical; the `fresh` channel receives host-granted physical ids
  (the host-put convention B3 wires).
- D4 compaction stays out of scope: size P for the run's max length, or accept
  `np <= P` growth bounds (follow-ups).

### B.2 Driver: finish the explicit-KV-write wiring (parallel with B1)

`fg.w_page`/`fg.w_off` are resolved
([fire_geometry.hpp](driver/cuda/src/ptir/fire_geometry.hpp) `has_write_desc`)
but nothing calls `launch_write_kv_explicit_*`
([kv_paged.cu](driver/cuda/src/kernels/kv_paged.cu)) from the forward path:
route the KV append through the explicit kernel when `has_write_desc`, instead
of the standard geometry derivation. Add the lease membership guard here if
adopted (one [B] membership check per fire against the leased set).

### B.3 Runtime: unified device-geometry submit (delete the beam branch)

- Delete `detect_beam`/`fire_beam`/`BeamRun` submit branch in
  [ptir_host.rs](runtime/src/ptir/ptir_host.rs); for device-bound ports use
  `map_geometry_relaxed` (leave wire fields empty), mark the fire
  solo/prebuilt, submit via `submit_prebuilt_async` (it stays: it is the
  solo-fire carrier).
- Wire `PageLease` ([ptir_lease.rs](runtime/src/ptir/ptir_lease.rs), built and
  unit-tested): grant B physical pages at fire 0 (seed values), B fresh per
  submit delivered as a host-put on the program's fresh-page channel (existing
  D1 coalescing path), reclaim unused grants at `finalize_fire` by reading
  harvested `w_cont`, reclaim everything on pass drop/failure. Pin float
  bounded by run-ahead depth x B pages, riding the per-fire arena txns.
- **Assert the FIFO invariant**: fires of one pipeline keep submission order
  through the scheduler onto one stream. This is the entire correctness
  argument for run-ahead and multi-pass chaining; make it an explicit, tested
  invariant, not an accident.

### B.4 Goldens and shadow verify

- Port the locked golden vectors from
  [ptir_beam.rs](runtime/src/ptir/ptir_beam.rs) tests into SDK trace tests
  (ptir-dsl + CPU reference interp): `golden_charlie_fork_freeze_csrs`
  (`np=[3,2]`, `pages=[5,6,7,5,6]`, `klen=[9,7]`, `w_slot=[7,6]`,
  `w_off=[0,2]`, `w_cont=[false,true]`) plus the continue-tail and page-turn
  cases. These vectors ARE the contract; they must exist outside ptir_beam.rs
  before B5 deletes it.
- Shadow verify: keep ptir_beam.rs alive one phase; in `finalize_fire`,
  debug-compare harvested geometry channels (pages/np/klen) against the host
  replay; run N steps green on the 4090. Then reroute
  [cuda_beam_e2e.rs](bin/pie/tests/cuda_beam_e2e.rs) through the ordinary
  run-ahead submit.

### B.5 Deletion (single commit, after shadow green)

- Runtime: `ptir_beam.rs` (entire file), `BeamRun`/`detect_beam`/`fire_beam`/
  `beam_channel_u32`/`ForwardPass::beam`/placeholder toks/eager B*P alloc;
  `KvWorkingSet::write_slot_shared_inplace` if no caller remains.
- Driver: verify nothing crept back;
  `grep -ri beam driver/cuda/src` finds no symbols/files. Optionally a CI gate.
- **Test: retire/replace `bin/pie/tests/cuda_beam_e2e.rs`** — it drives the old
  `beam` inferlet through the deleted classic replay path; `cuda_beam_devgeo_e2e.rs`
  (the fresh PTIR device-geometry e2e, e7c5ec54) is its replacement. Don't leave a
  test exercising the deleted path.

### B follow-ups (recorded, out of scope)

D4 compaction (program computes the gather plan, driver provides one generic
KV-gather); device-side geometry assembly without the D2H hop; co-batching
device-geometry fires (v1 is solo); true blocking `take` across guest tasks
(wasi-p3).

**Generalize B2 explicit-KV-write across ALL model forwards.** B2 (e7c5ec54) is
TARGETED to Qwen3/`llama_like_forward_paged` only (all the beam-devgeo e2e on
Qwen3-0.6B needs). The `ForwardInputs`/`ForwardDispatchInputs` fields
(`w_page_d`/`w_off_d`/`has_write_desc`) + the executor upload are model-agnostic;
generalizing = thread the 3 params + the `launch_write_kv_explicit_bf16` branch
through each other `*_forward_paged` (gemma2/3n/4, mixtral, qwen3_5[_moe],
nemotron_h, deepseek_v4, kimi, glm5). Mechanical, backward-compatible (they
already default `has_write_desc=false`). Do when device-geometry needs a
non-Qwen3 model.

---

## Track C: runtime-driver boundary migration

Spec of record: [boundary.md](boundary.md) (B1-B17, word layout, protocol,
mock-first rule). Phase 0 is done. Remaining, each phase merging independently:

### C1 (= boundary Phase 1): remove accidental complexity while dormant

- Extend bind: wake slot ids down, word-layout header shared by both sides.
- Carrier callback reduced to word publish + direct `pie_wake_past` calls from
  the driver's instance table.
- Delete: `CarryWake`/`user_ptr` trampoline
  ([control_cuda.rs](runtime/src/driver/control_cuda.rs)), carry SoA wire
  columns + `InFlightTracker`/`CloseAction`/reap
  ([carry_bridge.rs](runtime/src/driver/carry_bridge.rs) + 4 `ForwardRequest`
  fields), the completion registry/scan
  ([completion.rs](runtime/src/driver/completion.rs), keep `PinnedRingWord`),
  the dead `PIE_CARRY_POPULATE` gate. Port the valuable regression tests
  (monotonic head, close-during-in-flight) to the carrier layer.
- `Completion` re-backed by the fixed pacing slot + `word[0]`.

### C2 (= boundary Phase 2): real layout and the value path

- Derive the frame layout from the trace (host-visible channel list, cell
  offsets/sizes); per-channel delta D2H replaces whole-frame mirror; bake
  static readiness/commit lists at registration (B4).
- `k_commit_bump` stores head/tail + pacing into mapped pinned words
  (device-side publish); host callback becomes wake-only; delete
  `k_channel_bits`, the pass-end sync, the host ring mirrors.
- Rewire [ptir_host.rs](runtime/src/ptir/ptir_host.rs): take/read park on
  reader slots and load the mirror; put writes staging + epoch store (H2D of
  dirty inputs at fire-build); `ForwardResponse` loses the PTIR output
  marshal; `finalize_fire` keeps only KV/RS txn settlement.
- **Sequencing with Track B**: B3 modifies the same submit/finalize code paths
  on the current marshal transport; land B3 before or after C2, not
  interleaved. The WIT surface is unchanged by C2, so Track A is unaffected.

- **C2 RE-SEQUENCED (2026-07-08, alpha scoping + manager approval, option C).**
  boundary.md §7/§8 Phase 2 was written assuming the X2 frame_carrier
  (pinned-word / mirror / wake) is the LIVE value-path substrate to "make real".
  C1b confirmed it is DORMANT / unwired: the live PTIR path is
  `submit_prebuilt_async → tier0_runner → k_commit_bump (device mem) →
  sync_host_rings → harvest_outputs → ForwardResponse.ptir_output_* →
  finalize_fire → marshal_response`, which never touches frame_carrier. So the
  Phase-2 bullets marked here ENTANGLE with the deferred Phase-3 activation:
  device word-publish + wake-only callback + dropping the ForwardResponse
  marshal all require the pinned-word/wake/bind substrate to be LIVE first
  (activation), and dropping the marshal without it breaks the device-verified
  `cuda_beam_designb_e2e`.
  - **DONE now (live-path, cleanly independent):** delete `k_channel_bits`
    (write-only, zero consumer); bake the static readiness/commit slot lists at
    bind (`Tier0Runner::bake_static_lists`) — descriptor + per-stage full/empty
    arrays + deduped commit taken/put arrays uploaded once, removing the per-pass
    malloc/upload/free + intermediate readiness syncs from `run_pass`.
  - **ALREADY REALIZED:** "real per-channel trace layout on device" — the live
    `DeviceChannelRegistry::init_slot` already derives cell_bytes/cap1 from the
    trace decl and allocates per slot at registration. The only remaining
    "frame layout" sense (contiguous frame + cell offsets) is the dormant
    frame_carrier model → deferred with activation.
  - **DEFERRED to Phase 3 (activation):** pinned-word publish, wake-only host
    callback, ForwardResponse marshal drop, ptir_host.rs take/read park + mirror
    load, the pass-end sync + host-ring-mirror deletion. These need the bind
    lifecycle + completion switch that Phase 3 owns (blocked with compaction on
    the Oracle's policy calls). Manager is flagging the §8 inaccuracy + the
    "Phase-3 targets frame_carrier vs evolve the live marshal path" question to
    the Oracle as a Phase-3 contract call.


### C3 (= boundary Phase 3): activation and the fire rule

- Scheduler paces run-ahead depth on pacing words.
- Fire-skip heuristic (X4, as scheduler policy): skip an instance whose input
  epochs are unchanged since its last dummy-run.
- Profile; optionally fold wakes into the scheduler's polling loop with a
  single idle doorbell.

Separate tracks recorded in boundary.md §8: tier-1 fusion (P5.3), descriptor
diet (B14), Metal/shmem ports.

---

## Cross-track: blockers, cleanup, verification

### Blockers

- **FlashInfer cutlass MoE build**: last known state, full `cargo build -p
  pie-worker --features driver-cuda` died at `pie_flashinfer_cutlass_moe`
  (sccache/OOM, not our code), blocking full-link e2e verification. Re-verify
  on the current tree; try `RUSTC_WRAPPER` unset + low
  `CMAKE_BUILD_PARALLEL_LEVEL`. Fast executor object-compile loop (bypasses
  MoE): take `CXX_FLAGS`/`CXX_DEFINES`/`CXX_INCLUDES` from
  `target/debug/build/pie-worker-*/out/cuda/build/CMakeFiles/pie_driver_cuda_lib.dir/flags.make`,
  run under bash: `c++ $DEFS $INCS $FLAGS -fsyntax-only .../executor.cpp`.
- **Verify box**: RTX 4090 over `ssh workstation`, CUDA 13.3, sccache + CPM
  cache at `~/.cache/pie-cpm` (FlashInfer prefetched); tree copy at
  `workstation:~/Workspace/pie-ptir-verify` (rsync, excl `target/`, `.git/`).

### Cleanup (small, do opportunistically)

- Untracked duplicate leftovers from the W1.3 rename: delete
  `driver/cuda/src/kernels/beam_mask_adapter.cu` and
  `driver/cuda/tests/beam_mask_adapter_test.cu` (verified byte-identical to
  their committed `pack_dense_mask` twins).
- `executor.cpp:776` breadcrumb: an uncalled helper whose only caller was the
  deleted sampling_ir program-run block; remove with A4 or sooner.
- `ptir_refact.md` was absorbed into this document and deleted.

### Verification plan

1. **A1**: `ptir-dsl` container-bytes goldens byte-identical pre/post
   extraction; interp parity tests pass under the dev feature.
2. **A2/A3**: ptir-greedy e2e through the bridge (guest -> WIT -> runtime ->
   driver -> harvested output).
3. **B**: SDK trace tests assert the program's emitted wire-form port values
   equal the locked vectors; shadow verify N steps green on the 4090; after
   B5, full e2e green with the replay deleted and the grep gate clean.
4. **C**: loom checks stay green in `pie-waker`; `MockControlPlane` proves
   `register -> bind -> put -> fire -> take` per phase; existing PTIR driver
   tests (`ptir_runner_test`, `ptir_golden_exec_test`, `ptir_tier1_test`)
   stay green after C2's runner rewiring.

### Risks and open decisions

- **Stream/FIFO ordering is the entire correctness argument** for run-ahead
  and multi-pass chaining (B3's asserted invariant is mandatory).
- **Freeze discipline** is assumed, not enforced, unless the B2 lease guard is
  adopted (recommended).
- **Ring capacity** (`kMaxRing=8`,
  [channels.hpp:106](driver/cuda/src/ptir/channels.hpp#L106)) bounds run-ahead
  depth per channel; deeper pipelines back-pressure at submit-prepare.
- **Decl conflicts on shared channels**: containers binding the same global id
  must declare identical shape/dtype/capacity; validated at bind, keep the
  error message clear.
- **A4 scope**: classic retirement depends on the media-splice decision and
  touches the same executor files as C; sequence explicitly when both are
  active.

---

## alpha progress log (proj2, branch tasks/proj2/agents/alpha)

Delivered + pushed (base 243f5b1e):
- A1 9bd1ee00 — carve ptir-dsl out of sdk/rust/ptir (neutral Builder; byte-identical goldens).
- A5 1139f2a9 — stage_key D5 normalization in pie-ptir (3 golden tests).
- gate a07c5091 — cfg-gate the missing sampling-edsl surface so inferlet compiles.
- A2 a61c6976 (+ ecd7ec7f refinement) — inferlet::ptir bridge over the WIT ptir surface.
- A3 b62827de — migrate beam + mtp-grammar to inferlet::ptir::prelude; fix workspace loader.
- B4 484af58e — port beam geometry golden vectors to host ptir-dsl+interp tests (slot geometry).
- B1 d5002c46 — device-geometry wire-form beam program (page_indptr=CumSum(np), packed pages) + interp goldens.
- B1 f9472fce — beam-devgeo wasm inferlet on the bridge (ForwardPass::port_channel escape hatch).

STATUS: Track A critical path (A1/A2/A3/A5) DONE. A4 deferred (media-splice/Oracle).
B1 authoring + B4 host goldens DONE (host-verified on CPU interp). REMAINING B is
device-verified: B1 e2e submit, B2 driver explicit-KV-write wiring, B3 runtime unified
device-geometry submit (delete beam branch, wire PageLease), B4 shadow-verify N steps on
4090, B5 deletion (ptir_beam.rs etc.). Track C not started (sequence after B per manager).

NEXT: STOP for device (4090) verification — needs ssh workstation access.

## C1 (boundary Phase 1) scope map — alpha, mapped 2026-07-08

Atomic host+CUDA+ABI change (deletions + Completion re-backing + bind extension
land together; deleting ForwardRequest carry fields forces the same-commit
executor.cpp edit). Verifiability: carry_bridge.rs / completion.rs / control.rs
are host-compiled (cargo check -p pie catches them); control_cuda.rs is
driver-cuda-gated (inspection only, no CUDA build); frame_carrier.{hpp,cpp} +
executor.cpp via g++ -fsyntax-only (flags.make). mock/loom for the carrier layer.

Word layout (boundary.md §, fixed both sides): word[0]=pacing (committed fire
count); word[1+2c]=channel c head; word[2+2c]=channel c tail. Wake slots:
pacing + per-channel reader/writer, bind-time fixed, freed at close.

DELETIONS (enumerated):
- execute.rs: dead PIE_CARRY_POPULATE gate (+ populate_carry call).
- carry_bridge.rs: CarryDescriptor SoA + push_carry_request; InFlightTracker /
  CloseAction / reap (close-gate). Keep the valuable regression tests (monotonic
  head, close-during-in-flight) → port to the carrier layer.
- completion.rs: registry + scan (CompletionConsumer scan_instance/drain). KEEP
  PinnedRingWord (the word reader).
- control_cuda.rs: CarryWake + user_ptr trampoline + cuda_carry_done. Carrier
  callback shrinks to word publish + direct pie_wake_past from the driver's
  instance table.
- schema.rs (ABI): 4 ForwardRequest carry cols (carry_abi_version/user_ptr/
  word_index/instance) + CARRY_DESCRIPTOR_VERSION.
- driver/cuda frame_carrier.{hpp,cpp} + executor.cpp: the classic-carrier reads
  of those cols. CRITICAL: preserve the submit_prebuilt_async solo-fire carrier
  that B3's device-geometry submit depends on — only the CLASSIC-carrier use dies.

ADDITIVE / REWORK:
- bind: return wake slot ids (down) + the shared word-layout header.
- Completion: re-back on the fixed pacing slot + word[0] (parked(pacing_slot,
  word0, target=fire_seq)).

STATUS: fully scoped, ready to execute as one focused atomic unit. Deferred the
blind partial-execution at session tail (atomic ABI change touching the B3
carrier — safer as a focused effort). Device link + e2e = 4090 gate.

### C1a (DONE, fc245640): bind wake-slot/word-layout extension + Completion pacing re-backing
Fully host-verifiable half of boundary Phase 1 (split from C1b per manager).
- BoundInstance: WakeSlots (bind-fixed pacing + per-channel ChannelWakers, freed at close) + WordLayout header (word[0]=pacing; word[1+2c]/word[2+2c]=chan c head/tail).
- Completion re-backed on FIXED pacing slot + word[0] (parked_pacing, owns_slot=false); steady-state fires alloc no slot. Mock advances shared per-instance pacing word.
- Ported regressions to carrier layer: monotonic_pacing_word_wakes_each_fire, non_monotonic_pacing_filters_the_second_fire, close_during_in_flight_is_safe.
- Verified: cargo check -p pie clean; 8 driver::control tests green; pie-waker substrate 9 green (C1a touches no pie-waker → loom invariants untouched); clippy clean.

### C1b (BLOCKED, 4090+fresh-session): the deletions + CUDA edits
Delete carry SoA/InFlightTracker/CloseAction/reap, completion registry-scan (KEEP PinnedRingWord), CarryWake/user_ptr trampoline, PIE_CARRY_POPULATE gate, 4 ForwardRequest fields; frame_carrier.{hpp,cpp}+control_cuda edits. MUST preserve submit_prebuilt_async solo-fire carrier (B3 dep). Syntax(-fsyntax-only)+inspection only; full driver-cuda link + e2e = 4090 gate.

## alpha device-verify session — 2026-07-08 (branch tasks/proj2/agents/alpha)

KEY: the dev box IS the 4090 (hostname Workstation, `workstation`→127.0.1.1 self;
RTX 4090 + CUDA 13.3 at /usr/local/cuda; CPM cache ~/.cache/pie-cpm). NO ssh/rsync
— build+test in place. Full `cargo build -p pie-worker --features driver-cuda`
clean in 5.5min with PIE_COMPILER_LAUNCHER=env + -j6 (NO FlashInfer MoE OOM).
Standalone PTIR driver tests build via cmake in /tmp/pie-driver-build (targets
test_ptir_golden_exec/runner/tier1 link only cudart/nvrtc — no MoE).

DONE + pushed:
- Driver PTIR tests GREEN on 4090: ptir_golden_exec 42/0 (incl beam_epilogue),
  ptir_runner 14/0, ptir_tier1 24/0 (run from driver/cuda/ so tests/golden-ptir resolves).
- B2 (e7c5ec54): explicit-KV-write wiring. ForwardInputs+ForwardDispatchInputs
  +w_page_d/w_off_d/has_write_desc; pi.w_page/pi.w_off persistent buffers (per-request);
  executor uploads fg.w_page/fg.w_off in the dg_resolved block (parallel to fg.mask->
  pi.custom_mask); llama_like_forward_paged branches to launch_write_kv_explicit_bf16
  when has_write_desc. Backward-compat (default false). Compiles clean; device path
  NOT yet exercised.
- beam-devgeo e2e (e7c5ec54): bin/pie/tests/cuda_beam_devgeo_e2e.rs. Boots Qwen3-0.6B,
  builds beam_devgeo.wasm, submits, asserts BEAM_DEVGEO non-degenerate.
  Run cmd: `cargo test -p pie-bin --features driver-cuda --test cuda_beam_devgeo_e2e -- --ignored --nocapture`.

BLOCKED — fire-0 seeding bring-up: e2e fails at `submit @0: ptir: channel 0: seeded
but no seed was put before the first fire`. Seeded loop-carried channels
(pages/lens/kvm/page_indptr/packed) have no fire-0 seed. bind_seeds_first_fire
(ptir_host.rs:250) requires one staged put per seeded channel. Old `beam` inferlet
has the IDENTICAL gap (only worked via the deleted classic fire_beam path; never
device-verified). DECISION PENDING (manager): (a) B3 runtime-seeds fire-0 pages from
the PageLease grant [matches plan B.0/ptir_host.rs:162 "runtime seeds fire 0's pages"]
vs (b) guest-seeds in beam-devgeo. alpha recommends (a).

NOTE: cargo build.rs rerun-if-changed=../driver/cuda/src tracks the DIR mtime, not
recursive file edits — so editing a driver .cpp does NOT auto-trigger cmake. Force
via `touch driver/cuda/CMakeLists.txt` (watched) before cargo, or drive cmake directly:
`cmake --build target/debug/build/pie-worker-*/out/cuda/build --target pie_driver_cuda_lib -j 6`.

FIRM RULE (Oracle): never restore sampling-edsl / interface/sampling-ir; old sampling
inferlets (mtp-native-verify/runahead/mtp-specdecode) + gated sampler.rs/emit.rs/program.rs
are deliberately dead — skip on build trips, never fix/un-gate/migrate.

### FIRE-0 SEEDING DESIGN GAP (alpha, 2026-07-08) — BLOCKS beam-devgeo e2e green
Deep-dive finding on the device-geometry fire-0 init contract (bigger than mechanical seed):
- Channel NAMES do NOT survive to the runtime ChannelDecl (container.rs:91 = shape/dtype/
  capacity/host_role/seeded only). Runtime can ID port-bound channels (packed=Pages,
  page_indptr=PageIndptr, kvm=AttnMask, w_slot/w_off=WSlot/WOff) but NOT the internal
  loop-carried pages/lens/tslot (pages & lens both seeded [B,P] u32, no port → indistinguishable).
- SPACE MISMATCH: guest ws.alloc → SLOT ids (sdk .../ptir.rs alloc_slots); runtime PageLease
  owns PHYSICAL page ids, injects them on `fresh` (ptir_host.rs:1024-1032). Device-geometry =
  "all physical, no slot→physical table". Fire-0 physical pages known ONLY to the runtime.
- WALL: pages/tslot seeds need fire-0 PHYSICAL pages. (a) runtime can't ID those channels;
  (b) guest can't know physical ids. Guest currently seeds tslot=0/pages=0 → fire-0 HEIR path
  writes physical page 0 (corruption, not benign diff). Old `beam` sidestepped via deleted
  classic replay (never device-verified).
- Options: (i) add a PORT/host-role so runtime can ID+seed the pages/tslot carrier with grant
  physical ids [alpha lean — smallest, matches plan "runtime seeds fire-0 pages", device-resident];
  (ii) fire-0 special all-fork-from-prompt, pages/tslot seeds unused, physical pages only from
  `fresh`; (iii) runtime resolves guest slot-seed→physical for pages carrier [against contract].
- ESCALATED to manager (needs device-geometry designer / charlie's intended contract). PAUSED.
  Checkpoint e7c5ec54 (B2+e2e) is clean; the e2e correctly surfaces this exact gap at submit @0.

### DESIGN B — beam via logical mask-out + lazy compaction (2026-07-08, Oracle) — SUPERSEDES Design A
Branch-heavy decoding (beam/forking) is now logical mask-out + lazy compaction, NOT eager
freeze/heir/fresh-page-per-fork (Design A: p1_overview s6_2, ptir_beam.rs goldens, `beam`
inferlet). This RESOLVES the fire-0 seeding gap above (no per-fork fresh-page handshake).
Core: KV cache = prefix tree over a shared physical page pool w/ lazy GC. Shared ancestors
written once, referenced by mask; live beams append at own tails; pruned beams excluded by
mask (pages NOT reclaimed at prune); garbage reclaimed in occasional BATCHED compaction.

Changes vs current Track B:
- B.1 epilogue MUCH simpler: top-B(parent+token) + each survivor's tail-append pos + per-beam
  AttnMask over the pool. NO heir election, NO freeze arithmetic, NO fresh-page selection, NO
  per-step gather(pages,parent) reorder.
- `fresh`-page per-fire handshake DISSOLVES. Host grows pool in bulk + reclaims on compaction.
- Driver gains ONE new generic mechanism: KV-gather compaction (gather live page ids → dense
  range). B2 write_kv_explicit + pack_dense_mask UNCHANGED (still hot path).
- Goldens re-based (B.4): Design A vectors (golden_charlie_fork_freeze_csrs, w_cont=[false,true])
  no longer target. Need Design B goldens: mask evolution across prune/fork + 1 compaction
  gather-plan vector (outside ptir_beam.rs before B.5).
- PageLease REPURPOSED (bulk growth + reclaim-on-compaction + reclaim-on-drop), not deleted.

STAYS from B.0: physical page ids e2e; dense mask on AttnMask port; out_par; host owns alloc +
driver program-agnostic; device-geo fires solo/prebuilt empty wire geo; run-ahead/submit-never-blocks.

New open items: (1) compaction trigger policy + FIFO barrier scheduling (compaction is a non-solo
pool-wide fire); (2) mask growth bound (cap pool-between-compactions or segmented/ragged mask);
(3) CompactPlan port (live-page ids + dest map) — the ONE new host↔program handshake, replaces
`fresh`; B.1 must define it.

STATUS: new direction landed while alpha idle at e7c5ec54. NOT yet started (manager coordinating
+ alpha context long → refresh before implementing Design B). Existing B2 (e7c5ec54) survives intact.

### DESIGN B — HOST GOLDENS DONE (afcfa26f, 2026-07-08) + next steps
CONSOLIDATION: all prior proj2 work squashed into ONE commit 68d2649b, pushed to BOTH local
origin/dev (/home/ingim/Workspace/pie) AND github origin/dev (verified identical, parent =
base 243f5b1e, clean squash). 20-commit history preserved in tag alpha-proj2-preconsolidate
(b0241efb, pushed). agent branch == dev == 68d2649b. Design B goldens (afcfa26f) sit on top.

DESIGN B STEADY-STATE EPILOGUE + GOLDENS (host-verified, DONE):
- sdk/rust/ptir-dsl/tests/beam_designb_goldens.rs — 2 goldens GREEN on CPU interp.
- Mechanism: each survivor appends at flat pool pos wpos=fill+lane; ancestry = mask edit
  new_mask = gather(mask,parent) OR eq(col,wpos). NO heir/freeze/fresh-page/reorder.
  Pages/PageIndptr CONSTANT (pool fixed between compactions). Write desc w_slot=wpos/PAGE_T,
  w_off=wpos%PAGE_T → B2 write_kv_explicit (unchanged hot path).
- Goldens: (1) fork-from-shared-prefix masks {0,1,2}/{0,1,3} w=[0,0]/[2,3]; (2) multistep+
  page-turn masks accumulate {0,1,2,4}/{0,1,3,5} w_slot=[1,1]/w_off=[0,1].
- Fire-0 seeding gap DISSOLVED under Design B (mask-seeded prompt, no per-fork fresh handshake).
- Interp gotchas hit: broadcast2 needs equal-or-scalar(rank0), so [1]+[B] must broadcast [1]→[B]
  first; reader terminal puts (out*/out_mask/out_wslot/out_woff) MUST come AFTER all loop-carried
  puts (else SSA ValueIdOutOfRange from reader auto-drain drop).

NEXT UNITS (manager-confirmed):
- Write a NEW Design-B inferlet on inferlet::ptir (guest program; wasm-buildable). Keep the
  Design-A beam-devgeo as reference until Design B is device-verified, then retire it.
- Device e2e of the steady-state (no-compaction) path on the 4090 → exercises B2 explicit write.
- DEFERRED (open contract, needs charlie/Oracle): compaction KV-gather + CompactPlan port,
  compaction trigger/FIFO barrier, mask growth bound. Design B "lock status" being confirmed
  with Oracle. Build ONLY on the stable steady-state part until the compaction contract lands.

DEVICE-VERIFY ENV (still valid): box IS the 4090 (hostname Workstation, workstation→127.0.1.1);
full driver-cuda build clean in ~5.5min with PIE_COMPILER_LAUNCHER=env + -j6 (no MoE OOM); cargo
build.rs rerun-if-changed tracks DIR mtime not file edits → touch driver/cuda/CMakeLists.txt to
force cmake, or drive cmake directly. Standalone PTIR driver tests build in /tmp/pie-driver-build.

### FINISH-LINE DECISIONS (2026-07-08, In Gim / directing human)
- **ORDERING: unification FIRST, compaction second.** Do the frame_carrier ↔ PTIR-instance
  bind unification (boundary.md Option A: device word-publish + wake + FrameAddresses at bind;
  ForwardResponse marshal DELETED, ptir_host take/read park+mirror) BEFORE the compaction path.
  This is the deferred Phase-2 activation parts + C3/Phase-3 activation, driven together.
- **COMPACTION TRIGGER = guest dead-slot threshold.** The inferlet (guest) tracks masked-out
  (dead) token slots; when dead-slot count > 8 × PAGE_SIZE, it initiates compaction (calls the
  copy_into move primitive). Guest-orchestrated (consistent with Design-B: driver only provides
  the generic move; guest computes liveness + plan). This likely also SATISFIES the
  "mask-growth-bound" open item (the 8×page threshold IS the bound). STILL OPEN: the FIFO-barrier
  scheduling of the compaction fire (compaction is a non-solo pool-wide move; ordering vs
  in-flight decode fires on the copy stream). Manager relaying to Oracle.

### UNIFICATION — NORTH-STAR IMPL PLAN (2026-07-08, In Gim / Oracle: "big leaps, test+fix later")
Oracle GO + directive: "Don't take the safe, incremental path. Make big leaps if it leads to the
north star. We can test and fix later." → drive boundary.md Option A end-state directly, NOT the
conservative layer-over-registry. Contract calls (self-resolved per the leap directive):
- Q1 STORAGE = per-instance frame model wins for HOST-VISIBLE channels: the value they carry is
  published into a per-instance pinned MIRROR + pinned WORDS (head/tail per host-visible ch +
  pacing[0]). DeviceChannelRegistry stays the device cell store (its cross-instance sharing is for
  INTERNAL device↔device channels; host-visible channels are per-instance I/O, so no conflict).
  frame_base = the registry's per-slot device cells (not a new contiguous frame).
- Q2 PUBLISH = migration form first (B11-sanctioned): after a fire commits, a HOST step D2H's the
  committed host-visible cells → pinned mirror, stores head/tail → pinned words, stores ++pacing →
  word[0], WAKES the instance pacing slot. (Device-side k_commit_bump pinned-word write = the pure
  end-state, deferred tail — same architecture, impl detail.)
- Q3 = value-path + activation together (one leap): ForwardResponse PTIR marshal DELETED; the
  completion is the Completion waker-park on word[0]; take/read read the pinned mirror.

CUT (dependency order):
  U1 frame_carrier real layout: register_program derives FrameLayout from the trace's HOST-VISIBLE
     channel list (per-channel cell_bytes × cap1 → mirror bytes; WordLayout::words(n_hostvis) →
     word bytes). bind_instance allocs device_frame(unused/vestigial)+pinned mirror+pinned words.
     Re-add a commit publish entrypoint: pie_frame_publish(instance, host-visible cells[], heads[],
     tails[]) → memcpy into mirror + store words (host-side; the executor calls it post-fire).
  U2 executor PTIR branch: after the fire's k_commit_bump + sync_host_rings, D2H the committed
     host-visible cells (view.host_take equivalent, but into the instance's pinned mirror) + publish
     head/tail words via pie_frame_publish; then wake. STOP marshaling cells into ForwardResponse.
  U3 runtime control_cuda: real host-visible channel count from the trace (not PROVISIONAL 1);
     bind allocs the real per-channel WakeSlots; enqueue parks Completion on word[0]/pacing.
  U4 ptir_host rewire: PtirInstance bind → pie_frame_bind + FrameAddresses; submit returns a
     Completion (park on word[0]) instead of the oneshot ForwardResponse rx; finalize_fire awaits
     the Completion, settles KV/RS ONLY, drops marshal_response; take/read read the pinned mirror
     via the words (reader park). put = stage into frame + epoch (host-fed carry_in).
  U5 ForwardResponse: drop ptir_output_* fields + marshal_response + harvest_outputs.
GUARDRAIL: cuda_beam_designb_e2e IDENTICAL tokens [106984,1,9370,91282,33108,101982,98362,125302].
Full driver-cuda build (PIE_COMPILER_LAUNCHER=env -j6) + ptir units + beam e2e before gate.

U1 LANDED (20a39ede): frame_carrier real layout. FrameChannel{cell_bytes,cap1,mirror_off};
bind_channels(n,cell_bytes[],cap1[]) allocs pinned mirror(sum cell_bytes*cap1)+words(1+2n u64);
publish(inst,n,src[],ring_index[],head[],tail[],pacing) memcpy cells→mirror ring slot + store
head/tail/pacing words (release fence before pacing[0]). frame_base vestigial(0). isolation test
extended+green.

ID-RECONCILE (the elegant collapse): runtime mints the PTIR wire instance_id FROM
pie_frame_bind_channels → executor PtirInstance (keyed by iid=ptir_program_instances[p]) == the
frame carrier instance → publish(iid,...) needs ZERO new ABI fields. host_reader channels derived
IDENTICALLY on both sides in DENSE order → mirror index = host_reader rank r.

U2/BIND (runtime, ptir_host.rs forward-pass.new ~line 502): replace next_instance_id() with a
CUDA-gated pie_frame_bind_channels(host_reader list) → (frame_id, mirror_base, word_base); use
frame_id as instance_id; store bases+layout(rank→cell_bytes,cap1) on ForwardPass. Non-cuda/dummy
driver → fall back to next_instance_id() + no frame (feature gate). host_reader cell_bytes =
shape.numel()*program_dtype_size (F32/I32/U32=4, Bool=1); cap1=capacity+1. CAVEAT (flip-time):
ChanDType::Act lowers to the model activation dtype device-side — if that's not F32 the computed
cell_bytes mismatches harvested len; publish uses ACTUAL harvested len, so size the mirror slot to
max(computed, harvested-observed) or verify Act=F32 for the model at flip. control_cuda.rs adds the
extern pie_frame_bind_channels; close path calls pie_frame_close(frame_id) on pass drop.

U2/PUBLISH (executor, ptir_dispatch.cu run ~line 157): after harvest_outputs(), iterate
trace_->channels tracking host_reader rank r; for each committed one, memcpy its harvested bytes
into a per-fire src[] and publish at ring slot=(produced_count[iid,r]%cap1), tail word=++produced
(cumulative monotonic), pacing=fire_seq. Keep the existing out_resp.ptir_output_* marshal ALONGSIDE
(additive → beam e2e stays green). Needs pie_frame_publish extern-C already in frame_carrier.
Track produced_count in PtirDispatch state keyed by (iid,rank).

U4/FLIP (runtime): take/read read mirror[rank ring slot] gated by tail(produced) word vs a
per-channel host consumed cursor; finalize_fire drops marshal_response (KV/RS settlement only).
U5: delete ForwardResponse.ptir_output_* + marshal_response + harvest's out_resp packing.
ACTIVATION TAIL (deferred): device-side k_commit_bump pinned-word write + WakerTable word-park
(replace the oneshot wake bridge). Same data architecture; wake-transport swap only.

STATUS 2026-07-08 (In Gim/Oracle in-session "big leaps"): U1 (20a39ede), U2 (e640c7e0),
U3=THE FLIP (51683b83) LANDED + device-proven. The PTIR VALUE PATH now flows through the pinned
frame mirror, superseding the ForwardResponse marshal (north-star boundary is the live path on
CUDA). cuda_beam_designb_e2e GREEN, IDENTICAL baseline tokens
[106984,1,9370,91282,33108,101982,98362,125302]; frame_carrier isolation OK; 49 ptir units OK.
Mechanism landed in U3: pie_frame_layout FFI + bind_channels_keyed(iid) (id-reconcile: frame
carrier keyed by the wire instance id the runtime already holds); instances_ → map; ptir_dispatch
frame-bind moved OUT of the instance first-sighting into ensure-before-publish (device-geometry
programs create their instance in resolve_descriptors BEFORE run(), so the old bind branch was
skipped — U2's publish was a silent no-op masked by the still-authoritative marshal; real bug
caught only when U3 actually READS the mirror). ptir_host.finalize_fire.produced_cells reads the
mirror ring (tail word vs per-channel pushed cursor), CUDA-gated; non-CUDA keeps the marshal.

U5 DONE (59668419) — ptir_output marshal DELETED end to end; frame mirror is the SOLE PTIR value
path. The earlier "routing-coupled → defer" worry was REFUTED by re-analysis: the real Token-vs-
Response routing gate is BATCH-level at response.rs:152 (token_payload_only checks dists/logits/
logprobs/entropies/program_tokens/spec_tokens — NOT ptir_output). The request.rs:728
`ptir_output_indptr.is_empty()` clause lived INSIDE extract_per_request, which only runs in the
rich-Response else-branch (response.rs:287) — it never governed batch routing, only whether the
per-req carried the (now-redundant) value columns. The beam is prebuilt (response.rs:100
passthrough → always Response) so its mirror read is unaffected; non-prebuilt ptir routing was
already the batch gate's job, independent of ptir_output. Deleted: schema.rs ptir_output_* fields
+ push/count/at methods + 2 roundtrip tests; cbindgen .h (regenerated); view.hpp fields+mapping;
ptir_dispatch.cu Impl out_* staging + packing + out_resp writes (KEEP harvest_outputs → publish);
request.rs fast-path clause + slice block; ptir_host non-CUDA produced_cells → Vec::new();
ptir_channel_store test rewritten to feed produced tuples directly. −196 LOC.

U6/UNIFICATION COMPLETE + DEVICE-VERIFIED (59668419): full driver-cuda build links; beam e2e GREEN
IDENTICAL tokens [106984,1,9370,91282,33108,101982,98362,125302]; 49 ptir units, 44 abi schema,
frame_carrier isolation all green; non-CUDA compiles. The boundary.md Option A end-state is LIVE:
one value path (pinned frame mirror), no parallel marshal. DEFERRED to Phase-3/activation (NOT in
U1-U6): the pinned-word device-publish of head/tail as the completion-wake transport + wake-only
host callback — the mirror is currently read after the oneshot resolves (host callback still the
wake), which is correct; swapping the wake transport is the remaining activation, orthogonal to the
value-path unification just completed.

### DESIGN B — COMPACTION DECIDED + INFERLET AUTHORED (2026-07-08)
COMPACTION (Oracle-decided): NOT a bespoke op — a GENERIC per-token, all-layers KV cell-MOVE
primitive: working_set.copy_into(dst_page_ids, dst_tok_idx, src_page_ids, src_tok_idx), token
indices len = PAGE_SIZE*len(page_ids). In-place two-pointer (move last-alive → first-empty;
src/dst disjoint by construction → ONE parallel copy, NO scratch buffer). Correct because KV is
stored POST-RoPE (llama_like.cpp:94 rope → :584/:589 write; attention reads stored K + custom
mask, not slot-derived position) → physical slot is pure storage, positions are a SET. Confirmed:
per-token, union-liveness (dead iff no live beam references it), all-layers. DISSOLVES the
CompactPlan-port open item (it's just a generic move descriptor, same kernel family as B2
write_kv_explicit). STILL TO BUILD (driver/runtime, later): the move kernel + compaction
trigger/FIFO-barrier + mask-growth-bound (cap pool-between-compactions).

INFERLET (18306311): runtime/tests/inferlets/beam-designb on inferlet::ptir. Mask-out epilogue =
host-golden logic (top-B + flat tail-append wpos=fill+lane + gather/eq/or mask evolution).
Pages/PageIndptr CONST via attn_working_set full-arity tuple (pages,page_indptr,klen,w_slot,w_off);
WSlot/WOff drive B2 write; AttnMask = per-beam mask. ALL channels guest-seeded (shared BOS prompt
at pool pos 0) → NO runtime physical-page handshake → fire-0 gap doesn't arise. Builds wasm32-wasip2.
Kept beam-devgeo (Design A) as reference (retire at B5). ws is registered but not per-fork-alloc'd;
device-side pool ownership (ws must own POOL_PAGES) is the one runtime seam for the device e2e.

NEXT: device e2e — write bin/pie/tests/cuda_beam_designb_e2e.rs (mirror cuda_beam_devgeo_e2e:
boot Qwen3-0.6B → build beam_designb.wasm → run → assert non-degenerate BEAM_DESIGNB tokens),
full driver-cuda build (guardrails), run on 4090 → first real exercise of B2 write_kv_explicit.
Fresh session recommended (clean-slate device-verify, context long). Boundary at 18306311.

### COMPACTION — SCOPE (2026-07-08, Oracle-ordered second; trigger = dead-slots > 8×PAGE_SIZE)
copy_into = a GENERIC per-token, all-layers device KV cell-MOVE. SCOPING FINDING: the primitive
exists at NO layer, and there is currently NO device-side KV-cell-move ANYWHERE (reorder/free are
host-metadata-only under `ws-slot-ids`; CopyRequest is NOT wired in the CUDA ptir path). So this is
a full-stack new op, NOT a wiring-up of something dormant. Layer stack:

K1 WIT (working-set.wit ×4: interface/inferlet/core, interface/inferlet/deps/core, sdk/rust,
   bakery): add `copy-into` to `kv-working-set`. Move is DEVICE-side + needs a fire → decide
   method-executes-device-op-at-call vs a ptir fire descriptor (leaning: a working-set structural
   op that lowers to a driver move request through the pipeline FIFO — reorder's device analogue).
   Args: dst/src page-relative-index + tok-idx lists (len = PAGE_T*len(page_ids)); host resolves
   slot→physical.
K2 Guest SDK: WorkingSet::copy_into in sdk/rust/inferlet/src/ptir.rs (+ KvWorkingSet binding). Sig
   from the design: copy_into(dst_page_ids, dst_tok_idx, src_page_ids, src_tok_idx).
K3 Runtime host: lower copy_into → resolve slot→physical dst/src, build move descriptor, submit
   through the per-pipeline FIFO (new wire op or a ForwardRequest carrying a move-only descriptor).
K4 Driver kernel: launch_copy_kv_cell_bf16 in kv_paged.cu — per (src_phys_page,src_off)→(dst_phys_
   page,dst_off), copy K AND V across ALL LAYERS (strided over the KV cache layer dim). SAME family
   as launch_write_kv_explicit_bf16 (physical page/off descriptors). RAW byte copy — correct
   because KV is POST-RoPE (slot = pure storage). src/dst disjoint by two-pointer construction → one
   parallel copy, no scratch.
K5 Trigger (GUEST, beam-designb): track dead (masked-out) pool slots/step; when dead-count >
   8×PAGE_T → compute in-place two-pointer plan (last-alive → first-empty), call copy_into, REMAP
   every beam's AttnMask to the moved slot positions, reset dead count. Mask-growth-bound: size
   POOL_PAGES with ≥8×PAGE_T headroom so the trigger fires before pool overflow.
K6 FIFO-BARRIER (OPEN — the one item that may need a real contract call): a pool-wide move must be
   ordered vs in-flight RUN-AHEAD decode fires (optimistic cursor overlaps N fires on the stream). A
   prior fire may still be writing a to-be-moved cell. PROPOSAL: the compaction op enters the same
   per-pipeline PendingFire FIFO; finalize DRAINS prior fires (await their rx) before issuing the
   move on the stream, and the move completes before the next fire's write descriptor resolves —
   i.e. compaction QUIESCES the pipeline (breaks run-ahead overlap) only for the rare compaction
   fire. FLAG: confirm a periodic pipeline stall is acceptable vs an overlap-preserving scheme.
DEVICE E2E: cuda_beam_designb_compact_e2e — a longer beam run that accumulates >8×PAGE_T dead slots
   → triggers ≥1 copy_into; guardrail that actually exercises the move + mask remap on device.

### COMPACTION — DECISIONS + K5 ALGORITHM (2026-07-08, manager calls)
DECIDED: U5 KEEP (unification COMPLETE + device-verified; routing-neutrality proof accepted — the
Token-vs-Response variant is the BATCH gate response.rs:152, never ptir_output; the else-branch
emits Response unconditionally at :302, so the removed request.rs:728 clause only gated an internal
empty-vec early-return, not the variant). K6 = QUIESCE (periodic pipeline stall for the rare
compaction fire; overlap-preserving deferred). K3 submit-model = ESCALATED to Oracle (A ws.copy_into
method needs a new WS→pipeline back-ref; B move-only fire reuses the pipeline FIFO + gives K6 for
free but departs from the ws.copy_into spelling; reconcile = ws.copy_into surface routed through the
FIFO under the hood). HOLD K1/K2/K3 until the Oracle picks — it shapes the WIT.

K5 ALGORITHM (guest-computable, submit-agnostic — turnkey once K3 lands; NOT yet coded because
touching beam-designb's POOL_PAGES/MAX_STEPS breaks the identical-token guardrail, and the helper's
home (SDK vs inferlet-local) depends on the K3 surface):
- LIVENESS from parent history alone (no device masks needed): the guest already takes out_par
  (parent[b] per step). Maintain, per current beam b, the SET of flat pool positions it attends =
  parent[b]'s set ∪ {its new wpos}. Slot BOS(pos 0) always live. A pool position < fill is DEAD iff
  it is in NO current beam's set (union-liveness). dead_count = fill+1 − |∪_b set_b|.
- POSITION↔(page,tok): flat pos p ↔ (pool_ids[p / PAGE_T], p % PAGE_T) — same map as the write
  descriptor (beam lib.rs:137-139). copy_into args decompose from flat positions via this.
- TRIGGER: when dead_count > 8×PAGE_T → compact. Requires POOL headroom: POOL_PAGES sized so
  live + 8×PAGE_T ≤ POOL (else overflow before trigger). This lives in the NEW compaction-e2e
  inferlet (a longer run), NOT beam-designb (guardrail must keep POOL_PAGES=8/MAX_STEPS=8 → tokens
  [106984,1,9370,91282,33108,101982,98362,125302]).
- TWO-POINTER MOVE PLAN (in-place, no scratch): lo=0, hi=fill-1. Advance lo to the first DEAD
  slot, retreat hi to the last ALIVE slot; while lo<hi emit move (src=hi → dst=lo), mark lo alive/hi
  dead, advance. Yields disjoint (dst_positions, src_positions) → decompose to the 4 copy_into arg
  lists. After the move, live slots occupy [0, live_count); the pool tail is free for new appends.
- MASK REMAP (K3-coupled — device mask edit): every beam's AttnMask bit at src moves to dst. In
  option B this rides the compaction fire (a gather/scatter over the mask channel); the guest
  rebuilds fill := live_count and continues. This is the one K5 part that needs the submit surface.

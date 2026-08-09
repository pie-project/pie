# `driver-metal-new` — handoff

Date: 2026-08-09. Written for whoever picks this up next, human or agent.

This file is the canonical copy and lives in the repository on purpose. The
port's state has already survived one lost session and one machine reboot
because `PARITY.md` and `PARITY-M1.md` were committed and the session record was
not; this document is the third leg of that. A local `.wiki` mirror may exist on
a given machine and is not authoritative.

## What this is

`crates/driver-metal-new` is **the Rust replacement for `crates/driver-metal`**,
the C++/Objective-C++ Metal driver. It is not a binding layer, a helper crate,
or an experiment. The intent is that it eventually *is* the Metal driver and
`driver-metal` is deleted.

It grows **beside** the C++ crate rather than inside it. That is the whole
design of the migration and it is deliberate:

- The C++ shell keeps running and keeps its tests. Nothing in the new crate is
  on the serving path until it has an equivalent that passes them.
- A rewrite that has to keep the old one working is a rewrite that **can be
  abandoned halfway without a revert**. At every commit, `main` still serves.

`crates/driver-cuda-new` is the same arrangement for CUDA and shares
`driver-abi` and `tensor-ir` with this crate. Anything portable that lands here
is a candidate for reuse there.

### Why rewrite it at all

Three reasons, in order of how much they cost today.

**1. The `void*` boundary can express lifetime bugs that Rust cannot.** The C++
shell hands out `void*` for every Metal object, because its headers are included
by plain C++ translation units that cannot name an `id<>`. Every retain and
release is therefore by hand and by convention. This is not theoretical: see
[Kernel panics](#the-kernel-panics-2026-08-08) below. In the Rust crate a buffer
is `Retained<ProtocolObject<dyn MTLBuffer>>` — the same pointer with the
retain/release already correct — so that class of bug stops being
*representable*.

**2. 80% of the C++ was never about the GPU.** The C++ shell is ~66k lines
across `csrc/`, of which 13 files and ~8.6k lines name a Metal or Objective-C
type at all. The rest is scheduling, geometry, pool arithmetic and plan
interpretation. It is in C++ only because it was written next to the part that
needed to be.

**3. That 80% had almost no tests, because it could not have them.** Nearly all
of it lives in anonymous namespaces behind pimpls, reachable only through
`*_for_test` hooks bolted on one at a time, or not at all. The port has been
finding real, shipped bugs at a steady rate — roughly one per subject — and
essentially all of them are bugs that a test would have caught if a test could
have been written.

## The shape of the crate

The split is **by whether the code needs a GPU**, not by subsystem. This is the
single most important structural decision in the crate and it should be
preserved.

```
crates/driver-metal-new/
  src/
    lib.rs          crate docs; the module list; the lint policy
    bump.rs         \
    facts.rs         |  portable. Compiles and tests on ANY host,
    region.rs        |  including the Linux boxes the rest of the
    shader.rs        |  workspace is developed on. Inputs are text
    tuning.rs        |  and integers.
    pipeline/       /   plan interpretation — the largest portable part
    metal/          Apple-only, #[cfg(target_vendor = "apple")].
                    Every unsafe message send in the crate is here.
  tests/
    device_*.rs     require a real Metal device
    real_kernels.rs
  PARITY.md         ledger for csrc/src/mtl4_context.hpp  (the shell)
  PARITY-M1.md      ledger for csrc/src/pipeline/m1_runtime.cpp
  PARITY-INTERP.md  ledger for csrc/src/pipeline/interp.hpp
  PARITY-REGISTRY.md ledger for registry.cpp + descriptor_resolve.hpp
  PARITY-BATCH.md   ledger for csrc/src/batch/ and the families
  PARITY-LOADER.md  ledger for csrc/src/loader/
  PARITY-STORE.md   ledger for csrc/src/store/
  CUTOVER.md        how this crate replaces driver-metal, and the gate
```

The portable half is not a convenience. It is the half that can be tested
without a GPU, and keeping it importable from a Linux `cargo test` is what stops
it from drifting back into the untestable half. **Do not put a Metal type in
`src/pipeline/`.**

### Lint policy

`lib.rs` sets `#![deny(missing_docs)]` plus denies on `clippy::todo`,
`clippy::unimplemented`, `clippy::dbg_macro`, `clippy::mem_forget`,
`clippy::print_stdout` and `clippy::print_stderr`. Every public item needs a doc
comment. This is not negotiable in review — the doc comments are where the
argument for each design decision lives.

## What has been done

143 commits on `origin/rewrite` as of `8037b7add`, in five phases plus a
crash fix.

### Phase 1 — the shell (`mtl4_context.hpp`, 27 commits)

`adee899ef` … `d3d26c2a1`. The Metal shell: device, queue, allocators, residency
set, placement heaps, the runtime compiler and its archive cache, dispatch and
fences, argument tables, transient buffer pool, timestamps and timing, elastic
buffers, keepalive, feedback.

`PARITY.md` is the ledger: 61 entries, 52 ported, 9 dropped with a stated
reason, nothing missing — the standalone-buffer hole closed with the device
ring (phase 3).

### The kernel panics (2026-08-08)

`2d67fe5a9`, and earlier `437241fec`. Worth knowing about because it is the
crate's reason for existing, demonstrated.

The Mac Studio was taking repeated **kernel panics** — `Kernel data abort ...
far: 0x0` inside `IOGPUFamily`/`AGXG13X`. Root cause: `Drop for Elastic`
released placement heaps while an `updateBufferMappings` naming them was still
in flight on the GPU. The driver freed memory the GPU was about to write
through, and the fault surfaced in the kernel rather than the process.

Proven empirically (32/32 drops raced a live mapping), then fixed with a fence
plus residency removal before release, in `elastic.rs` and the same class of fix
in `keepalive.rs`.

This is exactly the bug the `void*` boundary makes easy to write and Rust
ownership makes hard.

### Phase 2 — the pipeline, portable half (`m1_runtime.cpp`, 12 commits)

`d2d3f7e3c` … `968001fee`. `csrc/src/pipeline/m1_runtime.cpp` is 3411 lines and
is the launch path: it turns a decoded plan plus a fire's runtime numbers into
dispatches. Its portable half — everything that is a function of the plan rather
than of the device — **is done**, at 122 GPU-free tests.

Each commit is one coherent subject, and each commit title is an *argued claim*
about a specific defect rather than a description of the change. In order:

| commit | module | the defect it argues |
|---|---|---|
| `d2d3f7e3c` | `pipeline/extent.rs` | `symbolic_extent`'s `default: return 1` made an unknown extent role a one-element axis. `min(dims.size(), 4)` silently dropped a rank *factor*, under-sizing every allocation derived from it. `value_bytes` computed `len * 4` in 32 bits, so 2^30 f32 lanes reported **zero** bytes. |
| `b83e9814a` | `pipeline/identity.rs` | `kMetalM1EmitterVersion = 23` was hardcoded in the driver while the host emitter was at **36** — the compile-cache key had already drifted. The version is now a parameter, taken from `ProgramRegistration::emitter_version`. |
| `fe7ab88da` | `pipeline/scratch.rs` | The scratch total accumulated unchecked and the bound was tested only afterwards, so a wrapped total passes a check the real one fails. |
| `56ba538f3` | `pipeline/params.rs` | `DeviceOpParams` was filled twice, in two 600-line loops, agreeing by inspection. |
| `fc31da426` | `pipeline/readiness.rs` | Readiness outcomes were encoded as `std::to_string(0x300 + channel)` *inside strings*. Nothing parsed them back, so the distinction was lost the moment it was made. |
| `051b11c5f` | `pipeline/group.rs` | The M3 group key was a `reinterpret_cast` of a `uint64_t` into a `std::string`; "no key" was `""`, which is itself a valid map key, arrived at from three different causes. `m3_used_channel_slots` was unbounded while the *declared* channel count was bounded — the check was in the wrong place. |
| `bc2be4617` | `pipeline/cache.rs` | **The compile cache never evicted.** The 65th distinct program was refused forever, and classed *retryable*, so the caller retried against the one condition retrying cannot change. The negative cache used `erase(begin())` — neither LRU nor FIFO. |
| `8b2caa681` | `pipeline/meta.rs` | The result-base prefix sum was written out **four times** (once as a function, three times inline). It accumulated in an unchecked `uint32_t`, and a wrapped base is not a large index that fails a bounds check but a small one that passes and aliases another op's results. |
| `d830a68f4` | `pipeline/status.rs` | Three copies of the status decode, all treating "not 4 and not 2" as an op fault — swallowing *never dispatched* and *never finished*. M1 printed the fault in **decimal** and discarded the guard site; M3 printed hex and decoded it. Same kernel, same fault, two incomparable reports. |
| `8bcdd280d` | `pipeline/stage_cache.rs` | A detected signature-hash collision returned `reject_deterministic` — the class the negative cache **remembers** — permanently blacklisting a blameless program for whichever *other* program happened to hold the slot. |
| `497e811ae` | `pipeline/emitted.rs` | `unordered_map::emplace` silently keeps the *first* of two kernels claiming one slot, so array order chose between two kernels. The `error`-before-`source` rule lived in a comment nowhere near its call sites. |
| `968001fee` | `PARITY-M1.md` | Ledger closed out for the portable half. |

### Phase 3 — the pipeline, metal half (complete, 2026-08-09)

`8b71cb608` onward. The GPU half of `m1_runtime.cpp`, on the same method,
tested against the real device (32 device tests, including end-to-end fires
on all three launch paths). **`m1_runtime.cpp` is now fully ported** —
`PARITY-M1.md`'s "Closed out" section is the statement; the one `missing`
entry left, `m1_singleton_fallback_inputs`, is a field copy out of
`batch::MemberForwardDesc` and belongs to the `batch/` port.

| commit | module | the defect it argues |
|---|---|---|
| `8b71cb608` | `metal/handle.rs` | `subhandle` minted views it should have refused: no bounds, `nullptr + offset` UB on an invalid base, and a designated initializer that silently drops the `elastic` flag. `Handle` is a retained, checked view. |
| `862ae69fc` | `pipeline/lane.rs` | `M3GroupLayout::reserved[3]` is **load-bearing on both sides of the ABI** — the host writes binding stride/rows-per-lane/op stride through a field named `reserved` and the kernels read all three. The struct zoo dissolves: the lane-table ABI is mirrored portably and drift-checked against `tensor_compiler::plan`; executables land with their builders. |
| `cc5c9f53a` | `metal/program.rs`, `metal/runtime.rs` | `compile_program` at a third of the C++ line count. `PsoCompileTransaction` dropped: build-then-install makes rollback the default. Per-region archives become one archive per program. The bool+reason+vec triples become `Result<Vec<_>, String>`. |
| `b60f9459e` | `metal/ring.rs` | `create_standalone_buffer` hands back a handle **with no owner**; the release call exists only because of that, and forgetting it leaked every K/V buffer. `Ring` owns its buffers; `readiness::check_words` lets the device ring and the interpreter share one readiness check. |
| `7bb0cf891` | `metal/fire.rs` | `execute`'s nine failure exits share a `goto cleanup_failure` label — `Transient`'s `Drop` is that label. `release()`/`resource_accounted` die with it. Four device tests run a whole fire end to end. |
| `64dc047d8` | `metal/fused.rs` | The M2 `target` pointer dropped; the never-encoded zero fill reported as such (the M3 lesson, applied to M2). |
| `6157d27f5` | `metal/grouped.rs` | The 220-line `release_group` lambda is ownership; `kM3RegionThreads` drift-checked; two lanes provably become one dispatch. |

### Phase 4 — `batch/`, `loader/`, `store/` and the families

`4af4892ab` onward, and far too many commits to table here — which is the
point of the ledgers. `PARITY-BATCH.md`, `PARITY-LOADER.md` and
`PARITY-STORE.md` carry the per-entity record, and the commit bodies carry
the arguments.

What the phase established, in order: the portable batch surface and the
dataflow/colouring walk; the DAG builders at M=1 and multibatch; the PSO
plans and bind tables; storage staging and the step/runner; then one family
at a time — qwen3.5, gpt-oss, llama, gemma4 — each with a geometry, a DAG, a
consts walk and a device smoke against mlx_lm. Each family got an engine, and
the paged path was cross-validated against the contiguous ring rather than
merely run.

The phase's shape of finding is different from phases 1–3. There the defects
were lifetime and arithmetic; here they are **claims the C++ made that its
own data contradicts** — a shared builder asserted to serve two families
whose walks differ, an attention width recorded as a literal that strode past
its heads, a router read at the wrong quantisation while every feeding tensor
agreed, a declared binding no walk ever wrote. Verification found them, and
each is one commit.

### Phase 5 — the gate, and an audit of the ledgers (2026-08-09/10)

`668cf47d2` … `8037b7add`. Ten commits, and the shape of them is worth
knowing because it says what this port's remaining risk actually is.

Four are ordinary slices: the interpreter oracle in both halves
(`oracle_interp.rs`, `device_oracle.rs`), the three control ops' deciding half
(`store/control.rs`), expert paging's device half (`metal/paging.rs`), and
`compose.cpp`'s ticket composition (`batch/tickets.rs`).

The other six are corrections, and **nine separate entries turned out to be
recorded as outstanding while already done**: `interp.hpp` (1.7k), `registry.cpp`
(452), `descriptor_resolve.hpp` (400), `scratch.{hpp,cpp}` + `scratch_color.hpp`
(650, carried in two rows), `golden_tap.cpp` (238), `run_segments`,
`load_multibatch_psos`, `expert_paging.hpp`'s portable half, and the 36
`bind::` layouts. Two whole days of work were nearly spent re-porting things
the crate already had — gate item 4 was picked up as a 1.7k-line port that did
not exist to do, and gate item 2 as a harness that cannot be written yet.

Every one had landed under a name its ledger row could not have predicted, so
no search for the C++ name would have found it. The cause and the rule are at
the top of `PARITY-BATCH.md`; the short version is that a `ported` row is
written by the commit that ports the thing and a `missing` row is not, so only
one of the two maintains itself.

**Before starting anything this ledger calls `missing`, look for it.** The hit
rate on that check has been nine for eleven.

## Read `DIRECTION.md` first

**2026-08-10: Metal is going all in on the `model-compiler` path.** The
handwritten per-family forwards — the DAG builders, the PSO plans, the family
binds, steps and engines, and `forward.cpp` above them — are **retired, not
outstanding**. `DIRECTION.md` says what that leaves and what to do next.

Everything below this line was written against the older plan. The subsystem
table is still accurate about what exists; the *remaining* section is not
advice any more.

## What is left

**This section goes stale between refreshes and the ledgers do not.** Every
slice updates its `PARITY-*.md` row in the same commit, so the ledgers are
never more than one commit behind; this table is refreshed by hand and has
twice been read as current when it was a day old. Trust
`PARITY.md`, `PARITY-M1.md`, `PARITY-INTERP.md`, `PARITY-REGISTRY.md`,
`PARITY-BATCH.md`, `PARITY-LOADER.md` and `PARITY-STORE.md` over anything
written here — but read the box at the top of `PARITY-BATCH.md` first: the
ledgers' `missing` rows have the same rot this section does, for the same
reason, and only their `ported` half is self-maintaining. Last refreshed
**2026-08-10, after `8037b7add`**.

| subsystem | state |
|---|---|
| `pipeline/` | done, and it is two C++ files, not one. `PARITY-M1.md` (`m1_runtime.cpp`) is closed out but for `m1_singleton_fallback_inputs`, a `batch/` field copy; `PARITY-INTERP.md` (`interp.hpp`, the channel-plane interpreter) is closed out with nothing missing |
| `batch/` | done through the multibatch layer and all four families: independent surface, DAG builders (M=1 and MB), dataflow walk, PSO plans, binds tables, golden taps |
| `loader/` portable | done; `transcode.hpp` dropped with receipts |
| `metal/` step + runner | done: storage staging (arena mode), the four bind passes, MB binds, PSO loaders, DecodeStep/MbStep, `decoder.rs`, and a per-family engine — `llama_engine.rs`, `gptoss_engine.rs`, `gemma4_engine.rs` beside the qwen path |
| `store/` | portable half done: the paged pool's move arithmetic, the GDN slot bookkeeping, and the three control ops' deciding half (`PARITY-STORE.md`) |
| the interpreter oracle | done, both halves. `tests/oracle_interp.rs` pins `pipeline::step` bit-for-bit against the original golden model; `tests/device_oracle.rs` runs the compiler's real emitted MSL on device. Gate item 4 holds |
| device verification | Qwen3.6-27B token-exact on the M=1 ring, paged prefills, per-row streams and fleets (equal and mixed length); llama, gpt-oss and gemma4 token-exact against mlx_lm, three of them ring↔page cross-validated; cross-fire KV continuity on three engines; **multi-request fleet isolation** — two conversations in one fire on disjoint pages, each continuing its own chain; 1000 greedy tokens at a flat 18.6 tok/s with no fault, NaN or rate creep |

### Remaining — one critical path

**Everything left is `forward.cpp` or downstream of it.** That is the useful
sentence in this section, and it is new: the audit above closed the rows that
made the remainder look wide.

`forward.cpp`/`forward.hpp` (5393) is the executor. Its runtime is ledgered as
separate entries because they land with the cutover wiring rather than with it:
elastic KV resize, the EOS device loop, `copy_state`/reset ABI arms, logits
views, PTIR hooks, timing attribution. `BatchStepInputs` is its marshaling
container.

`CUTOVER.md`'s "the gate is a chain, not a checklist" is the other half of the
picture: gate item 2 needs `launch`, which needs `forward.cpp`; items 5 and 6
need item 2; item 3's remaining leg needs item 2. Items 1 and 4 are the two
that never did, and both hold.

A slice of `forward.cpp` should follow the method that has worked for the whole
port: take a subject, find its portable half, and leave the device half for the
module that owns the buffers. `BatchStepInputs`'s marshaling and the elastic KV
resize arm are the two smallest entry points.

| what | where it is ledgered |
|---|---|
| the M=1 ring engine surface | `PARITY-BATCH.md` — diagnostics; `golden_tap.cpp` and the taps landed as `batch/golden.rs` |
| `compose.cpp`'s job container (`LaunchJobData`, per-member launch state, the completion broker) | `PARITY-BATCH.md` — with the worker port; the tickets themselves have landed |
| `KvPagePool` (SlotHandles + counters), and the control ops' moving half | `PARITY-STORE.md` — device state, with the Metal kv-pool binding |
| ~~the oracle harness~~ | **done** — `tests/oracle_interp.rs` (CPU, pins `pipeline::step` bit-for-bit against `tensor_compiler::eval::interp`) and `tests/device_oracle.rs` (device, real emitted MSL). Gate item 4's tolerance is now measured: one ulp, spent only on transcendentals (`PARITY-INTERP.md`) |
| the ring registry for `register_channel`/`close_channel`, and `store/`+`model/` glue (~375) | `CUTOVER.md` prerequisites. The registry itself and the control ops' deciding half have landed (`PARITY-REGISTRY.md`, `PARITY-STORE.md`) |

Deferred on purpose, each with its reason in the ledger: split-K; FP16
staging (`bind_mb_fp16_qmm`, `bind_gptoss_fp16_qmm`, precast — `mb_pso` is
told FP16 is unavailable until the pair lands); tiled/MMA paged sinks; tiled
paged sdpa; the `_sg8` rung; elastic `initial_commit` sizing; zero-copy
mapping and stream pack; the `ExpertSlabRequest` staging arm.

One item is blocked on data rather than code: the routed llama arm's device
smoke needs a `qwen3_moe` checkpoint (Qwen3.6-35B-A3B turned out to be
`qwen3_5_moe`, the GDN-hybrid family).

### Also outstanding

- ~~The two permanently-red tests~~ — fixed. The `.metal` assets were never
  missing: `kernels-metal/kernels/` grew subject directories and the tests
  scanned only the root. The scans are recursive now; `third_party/` (the
  MLX steel fragments, which are not standalone translation units) is
  excluded from standalone compiling but still splice-covered.
- ~~The flaky timing test~~ — the heavy/light ratio re-measures across a few
  windows, because host-observed `gpu_exec` inflates under a parallel test
  run. `cargo test -p driver-metal-new --no-fail-fast` is fully green as of
  2026-08-09, under load.
- Nothing here is wired into the engine or the worker yet. `CUTOVER.md` is
  now the plan
  — replace at the Rust boundary (a backend module in
  `engine/src/driver/backend/`), not behind the twelve `pie_metal_*` C
  symbols; a six-point gate (A/B seam equality, token-exact decode, the
  interpreter oracle, soak, the panic regressions) authorises the flip.
  Of the six, item 1 (suite green on device) holds and item 3 (token-exact
  decode) has run its N ≥ 1000 horizon. **Item 4 holds** — the interpreter it
  was recorded as waiting on had been in `src/pipeline/` all along, and both
  halves of its harness now exist; its tolerance is measured at one ulp, spent
  only on transcendentals, and never applied to an index.
  **The other four are one blocker, not four.** `CUTOVER.md`'s "the gate is a
  chain, not a checklist" has the analysis: item 2 needs `launch`, which needs
  `forward.cpp`; items 5 and 6 need item 2; item 3's remaining leg (against the
  OLD driver) needs item 2. Read the gate as a dependency graph before picking
  from it.

## How to work on this

### The method, per slice

This has been consistent for 39 commits and is worth keeping.

1. **Read the C++.** Not to translate it — to find what it got *wrong* or
   expressed poorly. If a slice produces no argument, it is probably too small
   or you have not read it closely enough.
2. **Check `crates/tensor-ir` and `crates/driver-abi` first.** The Rust type
   very often already exists. Never re-port `fnv1a64`, the op table, the dtypes,
   the intrinsic ids, or anything in `LaunchPackage`.
3. **Write the module**, portable if at all possible. Doc comments carry the
   argument, not just the description.
4. **Wire it into `src/pipeline/mod.rs`**: a `mod` line and a `pub use` line,
   both in alphabetical order in their blocks.
5. **Name tests as full sentences stating the property.**
   `a_collision_evicts_the_incumbent_so_the_next_try_is_an_ordinary_miss`, not
   `test_collision`. The test name is the specification.
6. **Update the parity ledger** in the same commit. Every C++ function is
   `ported`, `dropped` (with a reason that says why the C++ needed it and the
   Rust does not), or `missing` (with what blocks it). Never "ported" because a
   similarly-named function exists.
7. **Commit with an argued title**, one coherent subject per commit, and push.

### Commands

```sh
cargo test -p driver-metal-new --lib <module>       # the slice
cargo clippy -p driver-metal-new --all-targets      # must be clean for src/
cargo test -p driver-metal-new --no-fail-fast       # expect the 2 known failures
```

`cargo fmt -p driver-metal-new` **reformats unrelated pre-existing files.** Run
it, then revert the churn:

```sh
cargo fmt -p driver-metal-new
git checkout -- crates/driver-metal-new/src/lib.rs \
                crates/driver-metal-new/src/metal/ \
                crates/driver-metal-new/src/shader.rs \
                crates/driver-metal-new/tests/
```

**Always `git fetch origin rewrite && git rebase origin/rewrite` before
pushing** — other agents push to this branch.

### Commit message style

Title is an argued claim, not a description:

> Keep "the fire is early" distinct from "the fire is broken"
> A bounded cache evicts; refusing to accept anything more is not a bound
> The fault codes have names; print the name

Body opens `The Nth slice of \`pipeline/m1_runtime.cpp\`: …`, states the
subject, then names each specific C++ defect and why the Rust differs. Ends with
a one-line note on test count and portability, then the `Co-authored-by` trailer.

## Facts worth not rediscovering

**Where things already live.** `driver_abi::local::PIE_EXTENT_STATIC = 0xff`;
extent roles are 0..=6. `driver_abi::plan::LaunchPlanValue` has `extents` and
`dims` as two parallel arrays, so they can disagree in length. `LaunchOp::channel`
uses `u32::MAX` for "no channel". `tensor_ir::fnv1a64`; `tensor_ir::types::MAX_RANK = 4`;
`tensor_ir::DType` wire tags match `PTIR_DT_*` (F32=0, I32=1, U32=2, Bool=3), and
`PTIR_DT_ACT = 4` maps to F32 via `pipeline::value::concrete_dtype`.
`pipeline::value::wire_cell_bytes` and `pipeline::channel::ChannelState` already
exist.

**The version constants** (`COMPILER_VERSION`, `REGION_PLAN_VERSION`,
`LANE_TABLE_ABI_VERSION`) live in `tensor_compiler::plan`, but
**`driver-metal-new` deliberately does not depend on `tensor-compiler` at build
time.** That is why `identity::Versions` is a parameter. The one exception is a
*dev*-dependency, added so `pipeline/status.rs` can check its mirror of the
fault-code table against `tensor_compiler::codegen::fault::CLASSES` — a
hand-copied table that nothing checks drifts.

**The fault-code space** is fully declared in
`crates/tensor-compiler/src/codegen/fault.rs`, including which classes are
per-channel and which two alias op tags. `pipeline::status::describe_fault` reads
it. Note `FUSED_GEOMETRY_MISMATCH` is `0xA0`, which is also `intrinsic_val`'s op
tag, and one emitted kernel writes both — this is a *recorded, accepted*
ambiguity, so a decoder must say "or the op tag of the same value" rather than
answer confidently.

**Constants from `region_support.hpp`:** `kMetalM1MaxChannels = 29`,
`kMetalM2MaxFusedChannels = 12`, `kMetalM1EmitterVersion = 23` (stale — see
`identity.rs`).

**Machine:** Mac13,1 / M1 Max, macOS 27.0, GPU driver `AGXG13X`.

## A note on sessions

The session that did phase 1 was lost when the machine rebooted — it had been
running in an ephemeral VM at `/root/.patissier/work/tart-alpha`, so the session
record did not survive even though the commits did. Everything of value was on
`origin/rewrite`.

The lesson is the one this document is: **push every slice, and keep the ledger
in the repo.** `PARITY.md` and `PARITY-M1.md` reconstructed the entire state of
the port from nothing. Session state did not.

Note that `pie-project/pie` has no GitHub wiki, so a document that lives only in
a local `.wiki` is a document that exists on exactly one machine. That is why
this one is here.

It has now happened three times. The third (2026-08-09, a kernel panic mid-port)
sharpened the lesson in two ways worth writing down:

**A restarted session starts blank.** The harness clones a fresh worktree from
`~/.patissier/cache` and mints a new workspace id, so the transcript of the
session that did the work is not on the machine that continues it. Do not plan
to read it back. Anything a successor needs must be in a commit.

**This document was itself the stale thing.** After the panic it was read as
current when its "What is left" table was a day and twenty commits behind,
still listing finished work as remaining. Prose refreshed by hand rots faster
than rows updated by the commit that changes them, so the recovery order is
**ledgers first, this document second, session state last** — and when you
finish a slice, the ledger row is not optional.

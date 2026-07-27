# PTIR Refactor: `compiler/` — design, status, and remaining work

Date: 2026-07-27
Status: **phases 1–3 landed and verified; phases 0–3′ remaining (§4).**
Baseline: merge-base with `origin/dev` is `9bda961fb`. `origin/dev` has since
advanced 28 commits, several of them in the launch path — re-merge before
starting phase 3′, which is the phase that touches it.

All line counts below are measured, not estimated. **Rust counts are whole-crate
(`src/` plus `tests/`); C++ counts are whole-file.** Mixing the two conventions
is what made the first draft of this table read as 60% growth where there was
almost none.

---

## 1. The problem

PTIR — the Pie Tensor IR — lived in four places at once:

| Where | What | Lines |
|---|---|---|
| `interface/ptir` | the IR, the planner, the interpreter, the header generator | 15.7k Rust |
| `driver/common/include/pie_native/ptir` | a hand-maintained C++ mirror of the same data model | 2.9k C++ |
| `driver/{cuda,metal}` | the backend code generators | 5.1k C++ |
| `sdk/rust/ptir-dsl` | the authoring eDSL | 4.4k Rust |

The seams had already started leaking. `ptir_abi.h` was checked in twice,
byte-for-byte identical, with nothing verifying the second copy. The RNG
generator wrote files directly into the driver tree. Both `CMakeLists.txt`
reached across the repo for golden fixtures. Metal's PTIR interpreter carried a
comment admitting it was "a C++ mirror of the canonical interpreter... one
divergence from `interp.rs` is accepted".

Structurally, `interface/` is documented as "boundary contract crates — the
dependency floor". A 3.5k-line planner and a 2.2k-line interpreter sitting next
to `interface/ids` (a plain serde leaf) was a category error.

And the deeper cost: the drivers had to *understand PTIR* — decode the
container, walk the plan, decide which emitter each region goes through — before
they could compile or launch anything. Every backend paid that cost again.

---

## 2. North star

> **`compiler/` builds a program, `runtime/` schedules it, `driver/` fires it.**

Concretely: of everything PTIR-related, only three kinds of C++ survive —
**device kernels**, **vendor libraries**, and the **thin shell that calls
NVRTC/`MTLLibrary` and launches**. Everything else is Rust under `compiler/`.

PTIR keeps its name as the *format*: container magic stays `"PTIR"`, so every
program hash and every checked-in golden is unchanged by the move.

### 2.0 Why this is worth doing

The line count is a consequence, not the goal. The goal is this:

> **Backend knowledge becomes testable without the backend's hardware.**

A PTIR emitter is a pure string builder — it reads a decoded plan and returns
text, and never touches a device. Once it is Rust under `compiler/`, every
CUDA and Metal code path in the project can be exercised on a Linux CI box with
no CUDA toolkit and no Metal SDK. That is not a projection: §3.2 is the proof,
and it caught three bugs that would have shipped silently wrong kernels.

This is also the honest framing of what a new backend costs. §2.3's claim is
that a *driver* stops needing an IR decoder — true, and worth a lot. But someone
still writes `compiler/codegen/src/<backend>/`. The work is relocated, not
removed. The win is that it is relocated to the one place where it can be
pinned by goldens on any machine, in a language with exhaustive `match`.

Acceptance test for the whole design: **could someone add a third backend
without touching `compiler/ir`, `compiler/plan`, or any driver's IR knowledge?**
If not, the seam is not finished.

### 2.1 Target tree

```
compiler/                    the tensor-program toolchain (100% Rust)
  dsl/          authoring    Tensor/Channel eDSL + the neutral trace Builder   (wasm)
  ir/           representation  types · op · registry · container · infer · validate · expand · rng
  plan/         analysis     region partitioning · lane table · PTRP · PTIB
  eval/         semantics    tier-0 reference interpreter + host partial evaluation
  codegen/      emission
    src/          cuda/ · metal/ · header · rng · program
    runtime/      device templates (.cuh/.metal) — data, assembled by Rust
    include/      generated C/MSL headers
    ^^^ both directories are the single source; no driver keeps a copy
  tests/        conformance
    golden/       19 traces — the only checked-in trace corpus
    golden-{msl,cuda}/  2,838 emitter cases, kept after the oracle goes (§3.2)

driver/
  abi/          PieBytes validation · LaunchView · step_launch · elastic
                + the fire-time PODs                              (~2.0k C++)
  cuda/
    src/kernels/    device kernels                20.4k   unchanged
    src/model/      per-model forward             30.9k   out of scope (see §6)
    src/pipeline/   launch + NVRTC + caching      19.2k   no emitters
    third_party/    flashinfer · cutlass · marlin  9.3k   vendor
  metal/          same shape
  transport/  dummy/
```

`interface/ptir/` and `sdk/rust/ptir-dsl/` are already gone (`460136fb2`).

`driver/common/` disappears: its PTIR half is deleted, and the rest becomes
`driver/abi/` — 1,119 lines of ABI/data-plane plus 884 lines of fire-time PODs
(`descriptor.hpp`, `fire_geometry.hpp`, `ptir_channels.hpp`), which are driver
state and never belonged to the compiler.

`src/pipeline/` is 23,786 today; 20,119 after phase 2′ removes the two emitters,
and 19,245 once `module_cache.hpp` is promoted out of `generated/` and that
directory goes with them.

### 2.2 Dependency shape

```
   dsl ──> ir <── eval
            ^
            └──── plan <── codegen
```

`ir` is the dependency floor. `plan` and `eval` are siblings that never depend
on each other, because they answer different questions about the same bound
trace — *how do we execute this* versus *what does it produce*.

### 2.3 The contract change

Today:

```
host ──[container + sidecar]──> driver
                                  ├ decode the IR          ← 1.8k C++ mirror
                                  ├ generate kernels       ← 3.7k C++ emitters
                                  └ compile + launch
```

North star:

```
compiler ──[launch package]──> driver
             emitted source        ├ compile (NVRTC / MTLLibrary)
             + entry names         └ launch
             + buffer/channel layout
             + region→launch mapping
             + fire geometry
             + port→field table
```

**A driver no longer has to know what PTIR is.**

The precise form of that statement matters, because "delete the mirror" is the
symptom and not the cause. `PieProgramDesc` looks like this:

```rust
pub program_hash: u64,          // stable registration/cache key
pub canonical_bytes: PieBytes,  // the container ─┐ the only reason
pub sidecar_bytes: PieBytes,    // the plan      ─┘ the mirror exists
pub emitted_kernels: PieEmittedKernelSlice,
pub emitter_version: u32,
```

The driver decodes PTIR because we *hand it PTIR*. It already has
`program_hash` as an opaque key, so identity and caching never needed the
container. So the goal is not "delete the decoder", it is:

> **Stop shipping the plan across the boundary.**

Which gives an end state that is checkable in one line rather than policed by
review:

```
canonical_bytes.len == 0 && sidecar_bytes.len == 0
```

If the bytes do not arrive, a decoder cannot exist. That is a structural
guarantee, not a convention — and it is strictly better than the CI grep it
replaces. Pair it with the build-system half: `driver/*/CMakeLists.txt`
currently puts a blanket `../common/include` on ~10 targets, which is how the
leak started; after phase 3′ the PTIR headers should be *unreachable*, not
merely unused.

**The launch package is a named, versioned type**, not a concept. It already
exists in embryo: `emitted_kernels` + `emitter_version` were its first two
fields (§3.3). Phase 3′ is therefore not "a larger ABI extension than
`emitted_kernels`" — it is *finishing the struct that phase 3 started*. Framing
it that way is what makes it tractable.

### 2.4 The second axis: fire time

Registration is only half the contract. Per fire there are three geometry
classes:

| class | who decides | driver's job |
|---|---|---|
| `HOST` (0) | runtime folds the prologue with `pareval` | nothing — it receives numbers |
| `DECODE_ENVELOPE` (1) | elided wire geometry | nothing |
| `DEVICE_GEOMETRY` (2) | values live in device channels | read the port channels itself |

Only the third is interesting: the values are on the device (the run-ahead beam
epilogue, for instance), so the host cannot fold them without a D2H round trip.

**This is still not IR knowledge.** `descriptor_resolve.hpp` says so itself:

> **PROGRAM-AGNOSTIC** (owner constraint): this is a 1:1 port→field copier
> applying two fixed contracts, **NOT beam logic**

— a table lookup plus `CSR-prefix` and `KvLen → ((len-1) % page) + 1`. The
hardest case on the fire path already proves the model holds.

But the same file admits the cost:

> The port→field table is **kept in explicit correspondence with `map_geometry`**

That correspondence is hand-maintained across **three** implementations:
`driver/cuda/src/pipeline/descriptor_resolve.hpp` (408),
`driver/metal/src/pipeline/descriptor_resolve.hpp` (456), and
`runtime/engine/src/pipeline/fire/geometry.rs`. It is exactly the duplicated
decision this refactor exists to remove — and it is easy to miss, because
neither C++ file includes `ptir/plan.hpp`, so both survive phase 3′ untouched.
**Ship the port→field table in the launch package** or the hand-sync outlives
the work.

### 2.5 The metric

Lines are a proxy. The invariant worth tracking is how many places implement
PTIR semantics:

| | Now | North star |
|---|---|---|
| interpreters | 3 — `pie-eval`, `interp.hpp` (1,981), `host_eval.hpp` (455) | **1** |
| emitters | 4 — C++ and Rust × cuda and metal | **2**, both Rust |
| decoders | 2 — `pie-ir::container`, the C++ mirror | **1** |

"Three interpreters become one" is both more compelling and harder to get
wrong than a line count.

### 2.6 Two terms that are easy to misread

**`plan` is the cuDNN/FFTW sense**, not the LLVM one: a reusable,
shape-parameterized execution strategy, cached by `ExecutableCacheKey`, with
runtime-varying extents kept symbolic so one plan serves many batch shapes. It
is not an optimization pass pipeline — nothing here rewrites a program to make
it faster. It decides which ops fuse into one generated kernel, what falls to a
library kernel, and where each value lands in the lane table. The wire format is
already called `PTRP` — *PTIR Region Plan* — so the name was the project's
before it was the directory's.

**`eval` is not test-only.** The interpreter is the golden model, but `pareval`
is a production path with three callers: canonical-KV fire evidence (the prefix
cache folds the geometry prologue instead of pattern-matching the trace),
capability-less execution (a driver with no device-geometry ports has the host
fold the prologue per fire), and geometry classification. It reuses the
interpreter's `eval_op`, so there is no second evaluator to drift.

**`PTRP` needs a stated stability contract.** It is a wire format that crosses a
process boundary, and it is the direct cause of the 13 stale goldens in §7.1.
Phase 3′ extends the ABI, which is the moment to say who owns PTRP compatibility
and whether it versions independently of `PIE_DRIVER_ABI_VERSION`. Once the
plan stops crossing to the driver (§2.3) the blast radius shrinks to
compiler-internal, which is the argument for doing it in that order.

---

## 3. What is done

Ten commits, `460136fb2..f28fe4d05` (nine of them first-parent; `799fe2d0d` is
the `origin/dev` merge). This document is `ae045ae06` on top of them.

| Commit | What |
|---|---|
| `460136fb2` | Consolidate the toolchain into `compiler/`; delete `interface/ptir` and `sdk/rust/ptir-dsl` |
| `799fe2d0d` | Merge `origin/dev` (14 commits) |
| `e01707dcc` | Put `compiler/dsl` in the host workspace |
| `ee03fca4c` | Retarget the references the move left behind |
| `f3e82b1dd` | Port the Metal MSL emitters to Rust |
| `1365402b3` | Port the CUDA kernel emitters to Rust |
| `1d50337c2` | ABI 15: `PieProgramDesc` carries a host-emitted kernel table |
| `cdcfbe255` | Generate a program's kernels on the host |
| `f4153b71f` | CUDA driver compiles the host's kernels |
| `f28fe4d05` | Count host-supplied versus regenerated regions |

### 3.1 `compiler/` today

| Crate | Lines | Role |
|---|---|---|
| `pie-ir` | 5,137 | representation |
| `pie-dsl` | 4,423 | authoring |
| `pie-plan` | 4,264 | analysis |
| `pie-codegen` | 3,247 | emission |
| `pie-eval` | 2,835 | semantics |
| `pie-compiler-tests` | 4,714 | conformance |

Totals: 24,620 Rust, against 20,103 in `interface/ptir` + `sdk/rust/ptir-dsl` at
the baseline. The +4.5k is the two ported emitter families and the oracle
comparison suites — the toolchain grew in Rust by rather less than the 5.1k of
C++ it is replacing.

Workspace: **1474 passed, 13 failed** — the 13 are pre-existing stale goldens on
`dev` (§7.1).

### 3.2 The verification asset

This is the part worth preserving in the reader's head, because it is what made
the ports safe and it is what the remaining phases depend on.

**The C++ emitters are pure string builders.** They read a decoded plan and
return text; they never query the device. That means they link with plain `g++`
on Linux with no CUDA toolkit and no Metal SDK — Metal's `MTL`/`NS` mentions are
all inside MSL string literals. So the C++ can be run as an *oracle*:

```
oracle (C++) ──dump──> compiler/tests/golden-{msl,cuda}/ <──compare── Rust port
        ↑                                                       ↑
        └──────── compiler/tests/oracle/corpus/stage_plans.txt ─┘
                  (PTRP wire form, written by Rust, decoded by C++)
```

Both sides read the same corpus file, so "both saw the same plans" is *checked*,
not assumed. Coverage: 1,578 Metal cases and 1,260 CUDA cases over 19 real stage
plans. `compiler/tests/oracle/README.md` has the exact build commands; a
re-derivation must produce an empty `git diff`.

CUDA fused kernels are 40–70 KB each and there are 320, so their bodies are
pinned by FNV-1a with 36 regions kept verbatim — 1.9 MB instead of 13 MB, still
failing loudly on any change but readably.

**The goldens outlive the oracle, and must be kept.** This is easy to get
backwards. The Rust suites never read `oracle/corpus/stage_plans.txt` — they
re-derive the plans from the 19 traces in `compiler/tests/golden/` through
`pie-plan` and compare against `golden-{msl,cuda}/`. The corpus is an input for
the *C++* side only. So when the C++ emitters go:

- delete `compiler/tests/oracle/` — 1,083 lines of harness, plus the corpus;
- **keep `golden-msl/` and `golden-cuda/`** — 2,838 pinned cases, 3.9 MB, zero
  maintenance, and after the deletion the only regression net the Rust emitters
  have.

Deleting them with the oracle, as an earlier draft of §4 proposed, would leave
both emitters at zero coverage. It matters most for Metal, where the goldens are
the *only* evidence until macOS CI exists.

**This oracle caught three bugs in the Rust port**, every one of which would
have produced silently wrong kernels and none of which a code review would
plausibly have caught:

- op tag constants were transcribed by hand and several were wrong (`iota` 0x31
  for 0x64, `const` 0x30 for 0x81, the reduce family shifted by one). They are
  now derived from `ptir_abi.h` — the generated header is the source of truth
  and there was no reason to retype it.
- `PTIR_INTR_LAYER` is 5, not 4, which would have routed the layer intrinsic
  through the parallel path instead of the single-thread fallback.
- `second_party_region_supported` has a `sink_call` branch for `attn_page_mask`
  that was dropped, which would have rejected every program using the page mask.

**Two bugs in the C++ original** were found and deliberately *not* reproduced,
both documented in `compiler/tests/oracle/README.md`:

- `emit_grouped_nucleus_msl` reads `region.inputs[0..3]` out of bounds on a
  Generated region. Its guard leans on `library_region_valid`, which returns
  true immediately for non-library regions, and a Generated region's
  `library_op` byte is `0 == PTIR_LIBRARY_NUCLEUS_SAMPLE`; the TopK sibling has
  the `!region.library` check this one is missing. Latent, not live — the
  runtime only calls it behind its own library test.
- `validate_singleton_plan` leaves a partly built `operations` vector behind
  when it rejects, an artifact of filling an out-param while walking. The caller
  never reads it on that path.

### 3.3 The ABI seam

`PieProgramDesc` gained a table of host-emitted kernels (kind, stage, region,
entry name, source, error) plus the emitter version. Three design points:

- **Failure is data, not absence.** An empty `source` with a populated `error`
  is how the host says "I could not emit this one". Metal already degrades this
  way — a fused region over the 12-channel binding limit falls back to one
  launch per op — and that behaviour has to survive. A driver seeing only a
  missing entry could not tell a deliberate fallback from a bug.
- **`emitter_version` is in the descriptor**, not implied by the ABI version. It
  keys the driver's compile cache, and the failure mode of getting it wrong is
  silent reuse of a stale cubin rather than a loud mismatch.
- **`kind` is explicit** rather than re-derived from the plan. The host has
  already decided which launch path a region takes; making the driver recompute
  it would reintroduce the duplicated decision the change exists to remove.

Generation is opt-in per driver via a `codegen_backend` capability. An
unrecognised name means "I generate my own", never a guess — which is what lets
CUDA and Metal move independently instead of in a single flag day.

### 3.4 CUDA is wired and verified on hardware

The CUDA driver advertises `codegen_backend: "cuda"`;
`DriverBackend::register_program` fills the table from
`RegisteredProgram::emitted()` (memoised per program per backend); `module_cache`
looks up by (stage, region) and prefers the host's source, falling back to its
own emitter where nothing was supplied.

**Building and testing CUDA here** — both of these look like blockers until
checked:

- the tree needs **CUDA 13**: `driver/cuda/src/store/swap_pool.cpp` uses the
  final `cudaMemcpyBatchAsync` signature, so 12.9 does not compile;
- the installed driver is 550 (CUDA 12.4), so device tests need the compat
  driver: `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat`.

```sh
cargo build -p pie-worker --no-default-features --features driver-cuda   # ~7 min
cd target/debug/build/pie-worker-*/out/cuda/build
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat ctest -R ptir
```

`ptir` ctest is **7/10 before and after** the rewiring. All three failures
reproduce on a pristine `origin/dev` worktree (§7.1).

ctest parity alone is *not* sufficient evidence, because the two emitters
produce identical bytes by construction — a wiring bug that quietly regenerates
everything looks exactly like the wiring working. So two further checks exist:

- the six host-generated kernels the goldens record verbatim were fed to
  `nvcc -arch=sm_89 -cubin`: **6/6 produce a cubin**;
- `ModuleCacheStats::{host_sources, driver_sources}` make the live path
  observable, and give the in-driver emitter's deletion a precondition that can
  be *checked* rather than assumed.

---

## 4. What remains

Re-numbered from the earlier draft. The old phase 8 becomes **phase 0**: it has
no dependency on anything, and doing it first removes a rename that would
otherwise churn files phases 2′ and 3′ already touch. CUDA and Metal merge into
one step, because faithfulness — not hardware — is the bar (see phase 2′).

| Phase | Work | Δ lines | Gate |
|---|---|---|---|
| **0** | `driver/common` → `driver/abi` + `driver/common/ptir` | 0 | mechanical — **landed** |
| **1′** | Unify generated artifacts | −383 | freshness tests — **landed** |
| **2′** | Delete both in-driver emitters | −15,290 | goldens + `driver_sources == 0` |
| **3′** | Launch package: the plan stops crossing the boundary | −4,586 | GPU, plus one assertion |

Total **−20,259**, of which 3,142 is `ptir_generated_singleton_test.cu` — the
test of the emitter being deleted, which had already failed on `dev` since
before this work started. The Δ column is measured after implementation, not
projected; it does not count the driver-side test surface that needs
*retargeting* rather than deletion (§4.3).

Old→new: 8→0, 4→1′, 5+6→2′, 7→3′.

### Phase 0 — split `driver/common` first (0)

`driver/common` has no `CMakeLists.txt`; it is a bare header directory reached
through a blanket `../common/include` repeated on ~10 targets. So this is a
`git mv` plus include-path edits, available today:

- `driver/abi/` ← `abi_validation.hpp` (869), `launch_view.hpp` (135),
  `step_launch.hpp` (78), `elastic.hpp` (37) — 1,119 lines of ABI and data
  plane — plus the fire-time PODs `descriptor.hpp` (317), `fire_geometry.hpp`
  (285), `ptir_channels.hpp` (282). **~2.0k total**, not the 1.1k an earlier
  draft claimed.
- `driver/common/ptir/` is then a pure deletion target for phases 1′ and 3′.

Doing this first is what makes the rest legible: after it, "delete the PTIR
half" is a directory, not a list of files. It also lets phase 3′ finish by
*removing an include root* rather than auditing what is still reachable.

### Phase 1′ — unify the generated artifacts (−383)

Three files exist twice, byte for byte, all three already written from
`pie-codegen`:

| file | lines | driver copy |
|---|---|---|
| `ptir_abi.h` | 257 | `driver/common/include/pie_native/ptir/` |
| `rng_contract.generated.h` | 89 | `driver/common/include/pie_native/ptir/` |
| `ptir_rng.generated.metal` | 37 | `driver/metal/src/kernels/` |

The two headers become `#include <ptir_abi.h>` / `<rng_contract.generated.h>`
against `-I compiler/codegen/include`, and the driver copies are deleted.

`ptir_rng.generated.metal` cannot be an include-path fix: `ptir_m0.metal` and
`ptir_m1_runtime.metal` `#include` it *by name* and the Metal driver compiles
them from a kernels directory at run time. It is staged there by
`configure_file(... COPYONLY)` at configure time instead, so there is still
exactly one checked-in copy. The staged file is gitignored.

Safer than it looks: `ptir_header_uptodate` and
`generated_rng_artifacts_are_uptodate` already gate freshness and already pass;
both stop dual-writing as part of this phase.

**Correction: `ptir/op_table.hpp` (179) does not go.** An earlier draft called it
"a thin wrapper over `ptir_abi.h`". It is not. Its own header comment is
explicit that it *adds* what the generated table does not carry — per-op
`OpFamily`, `LaunchClass` and `ResultKind`, "which drive launch-shape selection
and tier-1 fusion cut points" — and `tier0_launch.hpp` alone names its `DType`
and `OpCode` 165 times. Those are launch-path consumers that outlive every
phase here, so `op_table.hpp` moves to `driver/abi/` with the other survivors
rather than being deleted.

**Landed**, along with phase 0.

### Phase 2′ — delete both in-driver emitters (−15,290)

CUDA `singleton_codegen.hpp` (1,883) + `fused_codegen.hpp` (1,784); Metal
`m1_codegen.{cpp,hpp}` (1,440) + `m1_generated_test.cpp` (5,518) +
`src/kernels/ptir_m1_runtime.metal` (928). Plus the oracle harness,
`compiler/tests/oracle/` (1,083) — but **not** the goldens (§3.2). Plus
`ptir_generated_singleton_test.cu` (3,142), whose subject is gone.

**Correction: not all 3,667 CUDA lines are emission.** "The C++ emitters are
pure string builders" (§3.2) is true of the emit *functions* — the device code
that looks like it lives in those headers is inside `R"PTIR_CUDA(` literals, so
they really do link with plain `g++`. But the two headers also host things the
launch path calls, which had to be lifted rather than deleted:

- `GeneratedValueDesc` / `GeneratedOpParams` — the device-side ABI structs the
  host packer in `fused_runtime.cuh` fills;
- `GeneratedKernelSource` — `module_cache.hpp`'s result type;
- `second_party_region_supported`, `validate_generated_region`,
  `detail::{supported_tag, same_type, library_region_valid, …}` — bind-time
  gates, called from `fused_runtime.cuh` and `module_cache.hpp`;
- `analyze_direct_argmax` + `DirectArgmaxAnalysis` (126 lines) — the launch
  packer's intrinsic side-table analysis, called from `fused_runtime.cuh`;
- `kCudaGeneratedEmitterVersion`, `kPtirIntrinsicSlots`.

Those 488 lines are now `driver/cuda/src/pipeline/region_support.hpp`. Under the
north star they become launch-package data too (§4.2); until then they are the
honest remainder, and the CUDA half of this phase is −3,179, not −3,667.

`validate_singleton_plan` (247) *was* deletable: on the CUDA side its only
callers were in the test that went with it.

`module_cache.hpp` now **requires** host source — a region with none is a
deterministic failure rather than a cue to regenerate. `driver_sources` stays
as the assertion that it never happens.

Metal's remaining wiring is structurally identical to CUDA's, already landed:
advertise `codegen_backend: "metal"`, have `m1_runtime.cpp` consume the supplied
source.

**On the macOS gap: faithfulness is the bar, not execution.** There is no Apple
hardware here and there does not need to be. The question this phase answers is
"does the Rust emitter produce what the C++ emitter produced", and that is
exactly what a differential oracle answers — 1,578 Metal cases, all green. "Does
the MSL run correctly on an M1" is a *different* question, it was never answered
by the C++ path either, and it can be answered later against goldens that will
still be there. `g++ -fsyntax-only` on `m1_runtime.cpp` covers the consuming
side. Land behind the capability flag so it stays one string from reverting.

Reasons to do the two backends in one step rather than two:

- the oracle and its corpus delete once, instead of living half-dead between
  phases;
- there is no window in which one backend is host-generated and the other is
  not, which is a state nobody wants to debug;
- the CUDA precondition — `ModuleCacheStats::driver_sources == 0` — is the same
  check for both.

**State the precondition workload.** "`driver_sources == 0` on a real workload"
is prose until a model and a stage set are named. Pick one and write it down.

**Sequence the orphan test.** `ptir_generated_singleton_test.cu` (3,142) tests
the emitter being deleted and **already fails on `dev`** with an illegal memory
access. Repair it against the Rust emitter's goldens *before* this phase, or the
deletion lands with no coverage at all and the repair is owed anyway.

### Phase 3′ — the launch package (−4,586)

**This is the real gate, and the earlier plan underestimated it.**

`ptir/plan.hpp` has 13 direct includers and 34 transitive ones, and they are not
only the emitters. The *launch* path reads the plan directly:
`program_runtime.hpp` (decode, validate, cache), `grouped_runtime.cuh`
(`ValueType`/`Dimension` traversal), `program_identity.hpp`, `library_region.hpp`.
Deleting the emitters does not make the mirrors unnecessary.

The framing that makes this tractable is §2.3's: this is not a new ABI
extension, it is **finishing the struct `emitted_kernels` started**. Fields to
add:

- buffer/scratch layout
- the region→launch mapping
- fire geometry
- **the port→field table** (§2.4) — otherwise the hand-sync between the *two*
  `descriptor_resolve.hpp` copies (cuda 408, metal 456) and `map_geometry`
  survives the whole refactor, because neither file includes `plan.hpp` and
  nothing here would touch them

Then stop sending `canonical_bytes` and `sidecar_bytes`. The driver already has
`program_hash` for identity and caching.

**Verify it the way phases 1–3 were verified.** The plan's own best tool is a
differential oracle, and the earlier draft abandoned it for its riskiest step.
Reuse it: ship the host's derivation *alongside* the driver's existing one, have
the driver compare and count divergence exactly as
`ModuleCacheStats::{host_sources, driver_sources}` already does, run it on real
workloads, and delete when the counter is zero. That turns a big-bang ABI swap
into the same evidence-driven shape as everything already landed.

Once it lands:

- `ptir/{container,trace,bound,plan}.hpp` — 1,823
- `driver/common/tests/ptir_decoder_limits_test.cpp` — 193, which tests the
  decoder being deleted
- `driver/cuda/tests/ptir_container_test.cpp` — 134, which checks
  `container_hash` and the readiness table against vendored goldens; it is a
  conformance test *for the C++ decoder*
- `driver/metal/src/pipeline/interp.hpp` — 1,981, whose own comment says it
  stands "until Decision 7's generated singleton path passes its M1 gates"
- `driver/cuda/tests/support/host_eval.hpp` — 455, whose own comment says it
  "is not the spec oracle"

and `driver/common/` is empty. Finish by deleting the include root, so the
invariant is enforced by the build rather than by review, and add the assertion
from §2.3:

```
canonical_bytes.len == 0 && sidecar_bytes.len == 0
```

### 4.2 What phase 3′ must absorb

Implementing phases 0–2′ turned up the full list of things the driver still
derives for itself. Each is a field the launch package has to carry, and the
list is longer than the earlier draft's three bullets:

| what the driver derives today | from | where it lives |
|---|---|---|
| buffer / scratch layout | the plan | `program_runtime.hpp` |
| region → launch mapping | the plan | `program_runtime.hpp`, `grouped_runtime.cuh` |
| fire geometry | the plan | `fire_geometry.hpp` consumers |
| **port → field table** | hand-synced with `map_geometry` | `descriptor_resolve.hpp` ×2 |
| **bind-time region gates** | the plan | `region_support.hpp` |
| **intrinsic side-table analysis** | the plan | `region_support.hpp::analyze_direct_argmax` |
| **per-op launch class / result kind** | `op_table.hpp` | `tier0_launch.hpp` |

The bottom four are the ones the plan did not name. They are the reason
`region_support.hpp` and `op_table.hpp` exist at all: each is a *decision about
the program* that the host has already made and the driver re-derives. Shipping
them as data is what finishes the job — and it is also what lets
`region_support.hpp` (488) and `op_table.hpp` (179) go, which the §5 ledger
currently books as permanent survivors.

Sequence it the way §4's phase 3′ says: ship each field alongside the driver's
existing derivation first, count divergence, and delete the derivation at zero.
The `ModuleCacheStats` pattern already in the tree is the model.

### 4.3 The driver-side test surface (retarget, not delete)

The Δ columns above count deletions. They do not count the driver's PTIR test
suite, which is 9,385 lines and mostly needs *retargeting* — the tests exercise
the launch path, and the launch path survives; what changes is the artifact they
feed it.

| file | lines | fate |
|---|---|---|
| `cuda/ptir_generated_singleton_test.cu` | 3,142 | **deleted in 2′** with its subject |
| `cuda/ptir_grouped_dispatch_test.cpp` | 2,147 | includes `plan.hpp`; retarget in 3′ (also fails to compile on `dev`, §7.2) |
| `cuda/ptir_tier0_test.cu` | 1,082 | retarget |
| `cuda/ptir_golden_exec_test.cu` | 871 | retarget |
| `cuda/ptir_runner_test.cu` | 409 | retarget |
| `cuda/ptir_graph_key_test.cpp` | 290 | unaffected (§7.2) |
| `cuda/ptir_tier1_test.cu` | 247 | retarget |
| `cuda/ptir_container_test.cpp` | 134 | phase 3′ — delete |
| `cuda/nucleus_region_test.cpp` | 99 | includes `plan.hpp`; retarget in 3′ |
| `metal/ptir_checkpoint_e2e_test.cpp` | 783 | retarget |
| `metal/ptir_m0_device_test.cpp` | 280 | retarget |

`driver/cuda/tests/golden-ptir/` needs a decision too. It is a **vendored
16-file subset of `compiler/tests/golden/`'s 19**, carrying raw container bytes,
and its header comment still points at `interface/sampling-ir/` — a path that
has not existed for two renames. Once the driver stops receiving containers
these goldens must change form: either the driver's tests consume launch
packages emitted by `pie-codegen` at build time, or they stop being golden tests
and become plain fixtures. Either way the vendored copy should not survive as a
second source of truth for traces.

---

## 5. End state

"Hand-written PTIR C++" below means C++ that encodes PTIR knowledge, excluding
generated headers. The full membership is listed so the arithmetic is auditable
rather than asserted — an earlier draft's table did not close (14.5k − 1.7k is
not 15.4k).

| | Now | North star |
|---|---|---|
| Hand-written PTIR C++ | 21,154 | **3,289** |
| Non-Rust inside `compiler/` | 4,910 | **3,827** (templates 3,444 + generated headers 383; all data) |
| Device kernels | 20,360 | 20,360 (unchanged) |
| Interpreters / emitters / decoders | 3 / 4 / 2 | **1 / 2 / 1** |
| port→field copiers | 3 (Rust + C++ ×2) | **1** (Rust; the table ships as data) |
| Deleted | — | **20,259** |

Check: 21,154 − 3,289 = 17,865 deleted from the set below, plus the duplicated
generated artifacts (383), the oracle harness (1,083) and the duplicated Metal
runtime template (928) = 20,259.

<details><summary>The 21,154</summary>

| file | lines | fate |
|---|---|---|
| `ptir/{container,trace,bound,plan}.hpp` | 1,823 | phase 3′ |
| `ptir/op_table.hpp` | 179 | → `driver/abi` (see phase 1′) |
| `ptir/{descriptor,fire_geometry}.hpp` | 602 | → `driver/abi` |
| `ptir_channels.hpp` | 282 | → `driver/abi` |
| `common/tests/ptir_decoder_limits_test.cpp` | 193 | phase 3′ |
| cuda `{singleton,fused}_codegen.hpp` | 3,667 | phase 2′ — 488 lifted to `region_support.hpp` |
| cuda `ptir_generated_singleton_test.cu` | 3,142 | phase 2′ |
| cuda `module_cache.hpp` | 874 | survives |
| cuda `descriptor_resolve.hpp` | 408 | survives |
| cuda `host_eval.hpp` | 455 | phase 3′ |
| cuda `tests/ptir_container_test.cpp` | 134 | phase 3′ |
| metal `m1_codegen.{cpp,hpp}` | 1,440 | phase 2′ |
| metal `m1_generated_test.cpp` | 5,518 | phase 2′ |
| metal `interp.hpp` | 1,981 | phase 3′ |
| metal `descriptor_resolve.hpp` | 456 | survives |

</details>

The 3,289 that survive: `module_cache.hpp` (874, compile and cache),
`region_support.hpp` (488, bind gates and the launch packer's analysis), the two
`descriptor_resolve.hpp` copies (864, the program-agnostic port→field copiers),
the 884 lines of fire-time PODs, and `op_table.hpp` (179, the driver's launch
vocabulary). **None of them decode PTIR** — which is the property that matters,
and the one the §2.3 assertion enforces.

Not in the table because they are retargeted rather than deleted: the 9,385
lines of driver PTIR tests (§4.3).

**The one-line version:** three interpreters become one, four emitters become
two, and the driver stops receiving the plan at all.

---

## 6. Explicitly out of scope

`driver/cuda/src/model/` is **30.9k lines of pure host C++** — 45 `.cpp` and 60
`.hpp` against 8 `.cu`. It parses HF config JSON in C++ while `runtime/model`
already does model metadata in 6.5k of Rust. Under the repository's Rust-first
convention it is a strong candidate for porting, and serde plus exhaustive
`match` is exactly the right tool for it.

It is not part of this refactor, because it has nothing to do with PTIR. Noted
here so the omission reads as a decision rather than an oversight.

---

## 7. Pre-existing failures (do not attribute to this work)

### 7.1 Rust: 13 stale goldens

`compiler/tests` `ptir_golden` is 8 passed / 13 failed. `interface/ptir/src/compiler.rs`
changed in `800fe40b6`, `a352a17d2` and `0dea15405` but the golden `.txt` files
were last blessed at `c1e148ef2`, so the embedded region-plan (`PTRP`) bytes are
stale. Container bytes and hashes still match; only the plan section differs.

Reproduced on a pristine `origin/dev` worktree. **Do not re-bless them** — that
would destroy the evidence that the planner changed. During the move, the panic
payloads were diffed against the old suite's: 13/13 byte-identical over 217,921
bytes, which is what proved the consolidation was behaviour-preserving.

Someone still has to decide what these goldens should say. That decision belongs
with whoever changed the planner.

### 7.2 CUDA: 3 failures

All reproduce on a pristine `origin/dev` worktree:

- `ptir_generated_singleton` — illegal memory access at `cuCtxSynchronize`
- `ptir_grouped_dispatch` / `_asan` — do not compile;
  `ptir_grouped_dispatch_test.cpp:1092,1144` call `Dispatch::enqueue_fixed_decode`
  and `enqueue_decode_envelopes` with signatures that no longer exist
- `ptir_graph_key` — 23/24 subcases pass; "capture exception guard restores a
  reusable CUDA stream" fails

### 7.3 Flaky

`pie-engine --test contention`
(`active_preemption_swaps_and_restores_an_over_capacity_fleet`) is timing- and
preemption-sensitive and fails intermittently on a shared machine. Observed
FAILED/ok/FAILED/ok across consecutive runs. Unrelated to PTIR.

---

## 8. Recommended order

**0 → 1′ → 2′ → 3′.**

Four steps, and only the last is hard. The earlier draft's `4 → 5 → 7 → 6 → 8`
had five, and its ordering argument rested on a constraint that turned out not
to bind.

**Phase 0 first** because it costs nothing and everything downstream is easier
against the final layout. Deferring it means a rename that touches files phases
2′ and 3′ are already editing.

**Phase 1′** is free and removes a whole class of drift; its freshness gates
already exist and already pass.

**2′ merges the old 5 and 6** because the reason to separate them is gone. That
reason was "Metal ends in a state only macOS CI can confirm" — but the bar for
deleting an emitter is *faithfulness to the one being deleted*, and a
differential oracle answers exactly that, on this machine, for both backends,
today. Whether the MSL runs correctly on an M1 is a separate question that the
C++ path never answered either, and the goldens that would answer it are being
kept (§3.2). Merging also means the oracle deletes once and there is no window
where the two backends disagree about who generates kernels.

**3′ last** because it is the only phase with real design content, it is the one
that needs a fresh merge from `origin/dev`, and it is much easier to reason
about once the driver's only remaining PTIR consumers are the launch path
itself. Give it the same treatment that made phases 1–3 safe: ship both
derivations, count divergence, delete at zero.

The thing to hold onto is that the last phase's success condition is not a line
count. It is that `canonical_bytes` and `sidecar_bytes` are empty, and therefore
that no driver can decode PTIR even if someone wanted it to.

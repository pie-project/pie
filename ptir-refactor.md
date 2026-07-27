# PTIR Refactor: `compiler/` — design, status, and remaining work

Date: 2026-07-27
Status: **phases 1–3 landed and verified; phases 4–8 remaining.**
Baseline: `origin/dev` @ `9bda961fb`. All line counts below are measured, not estimated.

---

## 1. The problem

PTIR — the Pie Tensor IR — lived in four places at once:

| Where | What | Lines |
|---|---|---|
| `interface/ptir` | the IR, the planner, the interpreter, the header generator | 12.5k Rust |
| `driver/common/include/pie_native/ptir` | a hand-maintained C++ mirror of the same data model | 2.9k C++ |
| `driver/{cuda,metal}` | the backend code generators | 5.1k C++ |
| `sdk/rust/ptir-dsl` | the authoring eDSL | 2.7k Rust |

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

### 2.1 Target tree

```
compiler/                    the tensor-program toolchain (100% Rust)
  dsl/          authoring    Tensor/Channel eDSL + the neutral trace Builder   (wasm)
  ir/           representation  types · op · registry · container · infer · validate · rng
  plan/         analysis     region partitioning · lane table · PTRP · PTIB
  eval/         semantics    tier-0 reference interpreter + host partial evaluation
  codegen/      emission
    src/          cuda/ · metal/ · header · rng · program
    runtime/      device templates (.cuh/.metal) — data, assembled by Rust
    include/      generated C headers — single source
  tests/        conformance  goldens · corpus · cross-implementation parity

driver/
  abi/          PieBytes validation · LaunchView · step_launch · elastic   (~1.1k C++)
  cuda/
    src/kernels/    device kernels                20.4k   unchanged
    src/model/      per-model forward             30.9k   out of scope (see §6)
    src/pipeline/   launch + NVRTC + caching      ~13k    no emitters
    third_party/    flashinfer · cutlass · marlin  9.3k   vendor
  metal/          same shape
  transport/  dummy/
```

`driver/common/` disappears: its PTIR half is deleted, its remaining 1.1k moves
to `driver/abi/`.

### 2.2 Dependency shape

```
   dsl ──┐
         ├──> ir <──┬── eval
                    └── plan <── codegen
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
             + fire geometry
```

**A driver no longer has to know what PTIR is.** That is the payoff: a new
backend implements compile-and-launch, not an IR decoder.

### 2.4 Two terms that are easy to misread

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

---

## 3. What is done

Eight commits, `460136fb2..f28fe4d05`.

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
pinned by FNV-1a with one region per stage kept verbatim — 1.9 MB instead of
13 MB, still failing loudly on any change but readably.

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

| Phase | Work | Δ lines | Verifiable here? |
|---|---|---|---|
| **4** | Unify generated artifacts | −525 | yes, trivially |
| **5** | Delete the CUDA in-driver emitter | −3,667 | yes, full GPU loop |
| **6** | Wire Metal, delete its emitter | −6,958 | **no** — no Apple hardware |
| **7** | ABI: launch metadata → delete the C++ mirrors | −4,259 | yes, GPU |
| **8** | `driver/common` → `driver/abi` | 0 | mechanical |

### Phase 4 — unify the generated artifacts (−525)

`ptir_abi.h` (257) and `rng_contract.generated.h` (89) exist twice; both copies
are already written from `pie-codegen`, so this is a matter of pointing the
drivers' include path at `compiler/codegen/include/` and deleting the duplicates.
`ptir/op_table.hpp` (179) is a thin wrapper over `ptir_abi.h` and goes with them.

Zero risk, and it removes the last place where a human could edit a generated
file and have it stick.

### Phase 5 — delete the CUDA in-driver emitter (−3,667)

`singleton_codegen.hpp` (1,883) + `fused_codegen.hpp` (1,784).

Precondition, now checkable: `generated_driver_sources == 0` on a real workload.
Then delete, and drop the `HostSource` fallback in `module_cache`.

Open question: `ptir_generated_singleton_test.cu` (3,142) tests the emitter
being deleted, and **already fails on `dev`** with an illegal memory access. It
is either repaired against the Rust emitter's goldens or removed with the code
it covers — that is a judgement call, not a mechanical step.

### Phase 6 — wire Metal, delete its emitter (−6,958)

Structurally identical to CUDA: advertise `codegen_backend: "metal"`, have
`m1_runtime.cpp` consume the supplied source, delete `m1_codegen.{cpp,hpp}`
(1,440) and `m1_generated_test.cpp` (5,518).

**Verification gap.** There is no Apple hardware here. `g++ -fsyntax-only`
compiles `m1_runtime.cpp` (with `-I driver/metal/src -I driver/metal/src/batch
-I driver/common/include -I loader/include -I interface/driver/include`), and
the golden comparison covers the emitter itself, but "it actually compiles and
runs as MSL" has to come from macOS CI. Land this behind the capability flag so
it is one string away from being reverted.

### Phase 7 — launch metadata, and the mirrors go (−4,259)

**This is the real gate, and the earlier plan underestimated it.**

`plan.hpp` has 29 consumers, and they are not only the emitters. The *launch*
path reads the plan directly: `program_runtime.hpp` (decode, validate, cache),
`grouped_runtime.cuh` (`ValueType`/`Dimension` traversal), `program_identity.hpp`,
`library_region.hpp`. Deleting the emitters does not make the mirrors
unnecessary.

To remove them, the ABI has to carry what the driver currently derives from the
plan: buffer/scratch layout, the region→launch mapping, and fire geometry. That
is a larger extension than `emitted_kernels`, and it is the last structural
piece of "the driver only compiles and launches".

Once it lands:

- `ptir/{container,trace,bound,plan}.hpp` — 1,823
- `driver/metal/src/pipeline/interp.hpp` — 1,981, whose own comment says it
  stands "until Decision 7's generated singleton path passes its M1 gates"
- `driver/cuda/tests/support/host_eval.hpp` — 455, whose own comment says it
  "is not the spec oracle"

### Phase 8 — `driver/common` → `driver/abi` (0)

What remains of `driver/common` after phases 4 and 7 is 1,119 lines that have
nothing to do with PTIR: `abi_validation.hpp` (869, `PieBytes`/CSR/slice
validation), `launch_view.hpp` (135) and `step_launch.hpp` (78) (the data
plane), `elastic.hpp` (37, memory accounting). Plus the runtime PODs
`descriptor.hpp` (317), `fire_geometry.hpp` (285) and `ptir_channels.hpp` (282),
which are fire-time state and belong to the driver, not the compiler.

So "delete `driver/common`" is really two different asks: **delete the PTIR half**
(phases 4 and 7) and **rename the rest**. The second is a move, not a deletion.

### Cleanup that rides along

`compiler/tests/oracle/` and `golden-{msl,cuda}/` share the fate of the C++
emitters — they exist to pin two implementations against each other, and one of
them is going away. Delete with phases 5 and 6.

---

## 5. End state

| | Now | North star |
|---|---|---|
| Hand-written PTIR C++ | 14.5k | **~1.7k** (`module_cache` 874 + runtime PODs 884) |
| Non-Rust inside `compiler/` | — | 5.1k (device templates + generated headers; all data) |
| Device kernels | 20.4k | 20.4k (unchanged) |
| Deleted | — | **~15.4k** |

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

**4 → 5 → 7 → 6 → 8.**

Phase 4 is free and removes a whole class of drift. Phase 5 is the largest
deletion with a complete verification loop, so it converts the most risk per
step. Phase 7 is placed before 6 deliberately: it is verifiable on this hardware
and it unblocks the largest remaining pile, whereas phase 6 ends in a state that
only macOS CI can confirm — better to reach it with everything else already
proven.

# Metal driver refactor: `driver/metal` north star

Handoff plan for restructuring the Metal driver. Self-contained: everything needed to
execute is in this file. Written 2026-07-11 against `dev` (parent commit `bbd0b382`).
Companion to `cpp-refact.md` (the CUDA driver plan); it reuses that plan's design
rules and target vocabulary, so read its Design rules section first. Where this file
says "rule N", it means cpp-refact.md's rule N.

## Goal

`driver/metal` is not a CUDA-style mess. It is ~21.8k lines of src + 4.8k of tests
with a 386-line CMakeLists, no fossil experiments, no orphaned tests, and a clean
11-function ABI identical in shape to CUDA's. Its problem is structural, not
archaeological: the tree contains two parallel implementations, and only one is
real.

- The **live path** is `entry.cpp` -> `executor/` -> `raw_metal/`: a native Metal-4
  decoder (heaps, PSOs, 27 hand-written `.metal` shaders) supporting exactly one
  architecture (the qwen3.5 GDN-hybrid, locally labeled "qwen3.6") on a resident
  single-stream ring (4096-token cap). PTIR programs execute on the CPU via a host
  interpreter, the inverse of CUDA's GPU tier-0 path.
- The **MLX cluster** (`ops/`, `model/`, `loader/`, `kernels/`, `kv_cache.hpp`,
  `linear_state_cache.hpp`, ~5.8k lines) is a parallel reference implementation
  with a broad arch registry. It compiles only under `PIE_METAL_WITH_MLX`
  (default OFF), is referenced only by one smoke test and `tools/parity/`, and
  nothing on the live path touches it. In a default worker build it does not
  compile at all. The directories that look like the standard layout are the
  shadow; the real backend is the one directory the standard layout has no name
  for.

Target: `src/` contains exactly one implementation, shaped like cpp-refact.md's
layout (`abi`, `context`, `batch`, `pipeline`, `model`, `kernels`, `store`,
`loader`), with raw_metal dissolved into it, the MLX cluster moved to `tests/`
(In Gim, 2026-07-11), and the dev/bring-up binaries in explicitly listed tool
targets instead of globs.

## Current state (audit summary, 2026-07-11)

| Area | Lines | Verdict |
|---|---|---|
| `src/entry.cpp` + `entry.hpp` + `main.cpp` + `config.hpp` | 1,711 | ABI shim clean (11 exports, all live from `ffi/metal.rs`); same shim/composition mix as CUDA's entry.cpp |
| `src/executor/` | 1,272 | Live seam: `MetalExecutor` + single-thread FIFO worker (Metal command-queue affinity); delegates to raw_metal only |
| `src/ptir/` | 1,559 | Live: `host_interp.hpp` (1,284) is the PRODUCTION PTIR engine (CPU); `descriptor_resolve.hpp` is a structural clone of CUDA's |
| `src/raw_metal/` | 8,889 (.cpp/.mm/.hpp) + 2,544 (.metal) | The real backend. Three interleaved subgroups: live qwen3.6 decoder; gemma4 bring-up (compiled into the lib, unreferenced by the live path); standalone dev/bench/probe binaries (excluded by regex, except one leak) |
| MLX cluster: `ops/` 1,849, `model/` 2,611, `loader/` 704, `kernels/` 186, `kv_cache.hpp` 303, `linear_state_cache.hpp` 182 | 5,835 | Opt-in, orphan from the ABI, parity/reference only |
| `tests/` | 4,763 | 17 files, all wired to targets; no orphans, no existence guards |
| `CMakeLists.txt` | 386 | Healthy except glob-based raw_metal filtering (source of the `.mm` leak) |

Known defects (all small):

1. `dvfs_probe.mm` (152 lines, has its own `main()`) is compiled into
   `pie_driver_metal_lib`: the raw_metal exclude regex
   (`CMakeLists.txt:103-104`) ends in `\.cpp$` and misses `.mm`.
2. `raw_metal/gemma4_encode.cpp`, `gemma4_heap_bind.cpp`, `gemma4_consts.cpp`
   (~750 lines) are globbed into the lib but reachable only from the standalone
   bring-up binaries that are not part of the main build.
3. `ProgramRecord::canonical`/`::sidecar` bytes retained after plan build
   (`entry.cpp:58-59`), same pattern as CUDA's entry.cpp.
4. `ptir/descriptor_resolve.hpp` reimplements the same contract as CUDA's
   (~230 of 275 lines differ only in substrate: interpreter ring vs GPU channel
   view). Dedup candidate at the `driver/common` level, not copy-paste.

## Design rules

Rules 1-7 of cpp-refact.md apply verbatim (frozen ABI, idiomatic naming with
`namespace pie::metal`, PTIR-only, no unified decoder, single registry owner,
every file builds or does not exist, deletion/relocation before movement). Metal
additions:

8. **One implementation in `src/`.** Reference/parity code lives under `tools/`,
   dev binaries under `tools/`, bring-up code for a not-yet-live arch lives with
   the bring-up harness until it wires into the executor. `src/` is what ships.
9. **No GLOBs in CMake.** The `.mm` leak is what globs cost. Every target lists
   its sources.
10. **The CPU interpreter is the production PTIR engine, on purpose.** Metal has
    no tier-0 GPU path and this plan does not add one. `pipeline/` here means
    "host interpreter + channels + registries", and the divergence from CUDA is
    documented at the module root, not hidden.

## Target layout

```
driver/metal/src
├── abi.cpp             C ABI: pie_metal_* wrappers, validation, handle -> Context
├── context.{hpp,cpp}   pie::metal::Context: RAII create/destroy, registries
│                       delegation, config parse, model facts, caps JSON;
│                       owns the executor worker thread
├── main.cpp            standalone diagnostic shim (unchanged)
├── mtl4_context.{hpp,mm}  Metal-4 device/queue/heap/PSO wrapper (top-level,
│                       like CUDA's distributed.cpp: Context-owned substrate)
├── batch/              one submitted batch -> device work
│   ├── worker.{hpp,cpp}    single-thread FIFO serializer (ex executor_worker)
│   ├── compose.cpp         launch preflight, deep-copy, batch_schedule,
│   │                       forward_marshal
│   ├── forward.cpp         MetalExecutor forward orchestration (ex
│   │                       executor.cpp + generic parts of decoder.cpp)
│   └── scratch.{hpp,cpp}   scratch_schedule (per-step scratch planning)
├── pipeline/           guest programs, CPU-side
│   ├── registry.{hpp,cpp}  program/instance/channel ownership + id allocation
│   │                       (from entry.cpp:1372-1377)
│   ├── interp.hpp          host interpreter (ex ptir/host_interp.hpp)
│   └── descriptor_resolve.hpp
├── model/
│   └── qwen3_5/            the live arch, renamed from "qwen36" to match CUDA:
│       decode_step.cpp, decode_step_mb.cpp, decode_consts.cpp,
│       decode_abi.hpp (arch-specific slots), dispatch tables
├── kernels/            the 27 .metal shader sources + decode_psos.{hpp,cpp}
│                       (PSO cache/table)
├── store/              resident ring + paged KV pool + linear-state slots
│                       (extracted from heap_layout.hpp + decoder residency)
└── loader/             StorageProgram -> resident heap: safetensors_view,
                        heap_bind, heap_layout (region sizing)

driver/metal/tests
├── mlx/                the relocated MLX cluster (ops, model graph, loader,
│                       sampling, kv_cache.hpp, linear_state_cache.hpp), opt-in
│                       via PIE_METAL_WITH_MLX; the reference oracle the parity
│                       harness and smoke tests build against
├── parity/             parity_driver + scripts (moved from tools/parity, since
│                       it is a test harness and now depends on tests/mlx)
└── (existing test files, unchanged)

driver/metal/tools
└── rawmetal/           standalone dev binaries: decode_run, harness*,
    bench_kernels, probes (incl. dvfs_probe.mm), and the gemma4 bring-up
    (gemma4_encode, gemma4_heap_bind, gemma4_consts, gemma4_decode_run,
    gemma4_encode_probe) until gemma4 wires into the executor
```

Notes on deliberate differences from the CUDA layout:

- **No `ops/` module.** raw_metal has no op-wrapper layer; decode steps dispatch
  PSOs directly. `ops/` appears when a second live arch needs shared wrappers,
  not before. (The current `src/ops/` is MLX and relocates.)
- **`store/` and `loader/` are small.** One arch, resident-ring residency. They
  exist for cross-driver navigability (same membership tests as CUDA), not size.
- **`mtl4_context` stays a top-level substrate file**, not a module: it is the
  Metal analog of a CUDA context + handle set, owned by `Context`.

Modules that cease to exist in `src/`: `executor/` (becomes `batch/`), `ptir/`
(becomes `pipeline/`), `raw_metal/` (dissolves into batch/model/kernels/store/
loader + tools/rawmetal), the MLX cluster directories (relocate to `tools/mlx/`),
`entry.cpp` (split into `abi.cpp` + `context.cpp`).

## Phase 0: baseline

Commit any pending work. Record baseline on Apple Silicon: configure + build
(default, no MLX), full `ctest` (17 targets, including `caps_honesty_test`,
`ptir_checkpoint_e2e_test`, `executor_worker_test`), and one live serve + client
inferlet run against the qwen3.6 checkpoint on this machine. Unlike CUDA, the
whole loop runs locally on the mac; no workstation dependency.

## Phase 1: hygiene (no behavior change)

1. **Kill the globs.** Replace the raw_metal GLOB + exclude-regex
   (`CMakeLists.txt:103-104`) with explicit per-target source lists. This fixes
   the `dvfs_probe.mm` leak (a stray `main()` in the shipped static archive) as
   a side effect rather than by patching the regex.
2. **gemma4 TUs out of the lib.** `gemma4_encode.cpp`, `gemma4_heap_bind.cpp`,
   `gemma4_consts.cpp` move to the standalone bring-up target's source list
   (they are only reachable from `gemma4_decode_run` / `gemma4_encode_probe`).
   No file moves yet; target membership only.
3. **Drop retained program bytes.** Free `ProgramRecord::canonical` after
   channel decode + plan build, `::sidecar` after plan build (`entry.cpp:58-59`).
4. Merge `entry.hpp` (3 lines) into wherever its single declaration is used.

Gate: build + full ctest + the serve/client smoke run. `nm` the static lib and
confirm no `_main` symbol.

## Phase 2: move the MLX cluster to `tests/` (decided)

In Gim's call (2026-07-11): the MLX cluster moves to `tests/`. It is a reference
oracle, so it lives with the things that consume it, and `src/` stops
advertising a second forward path.

1. `git mv` `src/ops`, `src/model`, `src/loader`, `src/kernels`,
   `src/kv_cache.hpp`, `src/linear_state_cache.hpp` -> `tests/mlx/` (internal
   layout preserved; it is a self-contained cluster, grep-verified: no live-path
   includes).
2. `tools/parity/` -> `tests/parity/`: the parity driver is a test harness and
   now depends on `tests/mlx/`; moving it keeps the dependency arrow inside the
   non-shipping tree (`src/` never includes `tests/`, enforced in Phase 7).
3. `PIE_METAL_WITH_MLX` now gates the `tests/mlx` + `tests/parity` targets
   instead of injecting sources into the driver lib. `tests/smoke_load.cpp` and
   `kv_perlayer_test` already live in `tests/` and just re-point their includes.
4. Revisit deletion after gemma4 lands natively in raw_metal: if the parity
   oracle for gemma4 ends up being Python scripts (as on CUDA), the MLX cluster
   loses its last consumer and dies then, as a one-line CMake + `git rm` change.

Gate: default build unchanged (byte-identical lib source list); MLX build
compiles and `tests/parity` still runs; ctest green.

This phase frees the `model/`, `loader/`, `kernels/` names in `src/` for
Phase 5.

## Phase 3: split `entry.cpp` into `abi.cpp` + `context.{hpp,cpp}`

Mirror of cpp-refact.md Phase 3, easier (no TP, no graph machinery):

1. `abi.cpp`: the 11 `extern "C"` wrappers (`entry.cpp:1415-1557`), validation
   via `pie_native::abi`, handle -> `Context` mapping. Nothing else.
2. `context.{hpp,cpp}`: `class pie::metal::Context` (the current `MetalDriver`).
   Keeps: config parse (`config.hpp` folds in), `read_model_facts`
   (`entry.cpp:304-354`), `build_caps_json` (356-439), launch preflight +
   post-to-worker (the async-launch behavior is preserved exactly), copy/resize
   control ops, worker ownership.
3. The program/instance/channel maps stay in Context this phase; they move to
   `pipeline/registry` in Phase 4.

Gate: build + ctest + serve/client run; `caps_honesty_test` is the sentinel for
accidental caps changes.

## Phase 4: `ptir/` -> `pipeline/`, registry unification

1. `git mv src/ptir src/pipeline`; `host_interp.hpp` -> `pipeline/interp.hpp`.
   Module-root comment states rule 10 (CPU interpreter is the production engine;
   CUDA runs tier-0 on GPU; same `pipeline/` membership on both drivers).
2. `pipeline/registry.{hpp,cpp}`: program records (ExecPlan cache), instance
   bindings, channel handles, id allocation move here from Context. Same
   single-owner shape as CUDA's Phase 2.
3. Grep gate: `grep -rn "ptir" driver/metal/src --include='*.cpp' --include='*.hpp' --include='*.mm' | grep -v "pie_native\|interface/ptir"`
   returns nothing.

Gate: build + ctest (`ptir_host_interp_test`, `ptir_descriptor_resolve_test`,
`ptir_device_geometry_e2e_test`, `ptir_copy_ops_test` renamed accordingly) +
serve/client run.

## Phase 5: dissolve `raw_metal/` into the standard layout

The heavy phase. Sequencing: do this after cpp-refact.md's Phase 5 lands on the
CUDA side, so the conventions (per-family model dirs, store membership) are
proven on the bigger tree first. File-by-file:

| raw_metal file | Target | Rationale |
|---|---|---|
| `decoder.{cpp,hpp}` | split: residency/orchestration -> `batch/forward.cpp`; ring/pool state -> `store/` | RawMetalDecoder mixes generic orchestration with KV residency |
| `decode_step.cpp`, `decode_step_mb.cpp` | `model/qwen3_5/` | the arch-specific forward assembly |
| `decode_consts.cpp` | `model/qwen3_5/` | per-arch constant upload |
| `decode_abi.hpp` | split: generic Region/IoSlot/Kernel vocabulary -> `batch/`; qwen slots -> `model/qwen3_5/` | shared vocabulary vs arch bindings |
| `decode_dispatch.hpp`, `decode_dispatch_mb.hpp` | `model/qwen3_5/` | dispatch tables for the qwen steps |
| `decode_psos.cpp`, `decode_timing.cpp` | `kernels/` (PSO table), `batch/` (timing) | |
| `batch_schedule.hpp`, `forward_marshal.hpp` | `batch/compose.cpp` orbit | launch-to-work translation |
| `scratch_schedule.cpp` | `batch/scratch.{hpp,cpp}` | per-step scratch planning (CUDA analog: batch/workspace) |
| `heap_bind.cpp`, `heap_bind_metal.hpp` | `loader/` | program arena execution + canonical operand binding |
| `safetensors_view.cpp` | `loader/` | mmap checkpoint reader |
| `heap_layout.hpp` | split: weight regions -> `loader/`; KV ring/pool/state regions -> `store/` | the one file straddling both memories |
| `mtl4_context.mm` + `.hpp` | top-level `src/` | Context-owned Metal substrate |
| `kernels/*.metal` (27) | `src/kernels/` | freed by Phase 2 |
| `executor/executor.{cpp,hpp}` | `batch/forward.cpp` (+ `batch/compose.cpp`) | the seam merges into the module it fronted |
| `executor/executor_worker.hpp` | `batch/worker.{hpp,cpp}` | |
| `decode_run.cpp`, `harness.cpp`, `harness_main.cpp`, `decoder_smoke.cpp`, `bench_kernels.cpp`, `heap_bind_probe.cpp`, `st_probe.cpp`, `dvfs_probe.mm` | `tools/rawmetal/` | standalone dev binaries, explicit targets |
| `gemma4_*` (all) | `tools/rawmetal/` (with the bring-up harness) | returns to `src/model/gemma4/` only when it wires into the executor |

Also in this phase: rename the arch label `qwen36` -> `qwen3_5` in identifiers
and paths (enum, files, display strings that are not wire-format), keeping the
HF `model_type` acceptance list as-is. Cross-driver grep for one family should
hit both drivers.

Gate: build + full ctest + serve/client run + before/after decode-latency check
with `decode_timing` (this phase must be motion-only; any perf delta is a bug).
Grep gate: `grep -rn "raw_metal" driver/metal/src` returns nothing.

## Phase 6: cross-driver dedup (optional, needs the CUDA tree)

Extract the shared skeleton of `descriptor_resolve.hpp` (the
`is_device_geometry_trace` / `resolve_fire_geometry` contract, `last_page_len`,
`read_port_cell`, the port-presence pattern) into
`driver/common/include/pie_native/ptir/` with the backend substrate injected
(interpreter ring vs GPU channel view). Only lift what is identical today; do
not force a template over genuinely divergent parts (`translate_kv_pages`,
`read_mask_cell` stay metal-side). Touches both drivers; land as one PR with
both test suites green (`ptir_descriptor_resolve_test` here, the CUDA
descriptor-resolve coverage there).

## Phase 7: CMake polish + CI gates

1. Final CMakeLists (~300 lines): explicit source lists everywhere, tools
   subtrees behind their options, tests all wired.
2. CI gates, same script family as CUDA's Phase 8:
   - every file under `tests/` appears in CMakeLists;
   - no `file(GLOB` in any driver CMakeLists;
   - no `_main` symbol in the shipped static lib (`nm` check, the dvfs lesson);
   - layering greps (must print nothing):
     ```sh
     # kernels (.metal + PSO table) include nothing above
     grep -rln '#include "\(model\|batch\|pipeline\|loader\|store\)/' driver/metal/src/kernels/
     # loader must not reach the fire path
     grep -rln '#include "\(batch\|pipeline\)/' driver/metal/src/loader/
     # nothing in src/ includes tools/ or tests/ (the MLX oracle stays non-shipping)
     grep -rln '#include ".*\(tools\|tests\)/' driver/metal/src/
     ```
3. Doc sweep: module-root charters; update `driver/metal/README.md` to describe
   the actual layout and the CPU-interpreter divergence; note `dev.toml` is the
   standalone shim's config.

Gate: clean-clone configure + build + ctest on Apple Silicon; serve/client run.

## Decision log

- **ABI frozen; identical 11-function shape to CUDA; async launch preserved.**
  The single-worker-thread model (Metal command-queue affinity) is a feature,
  not debt.
- **MLX cluster moves to `tests/`, not deleted.** In Gim's call (2026-07-11).
  It is the parity oracle and the only broad-arch reference; as test support it
  lives with its consumers (`tests/mlx/` + `tests/parity/`). Deletion is a
  follow-up decision after gemma4 lands natively; the relocation makes that
  deletion a one-liner.
- **gemma4 bring-up lives with the tools until it wires into the executor.**
  `src/model/` contains live archs only (rule 8). The moment gemma4's decode
  step is called by `batch/forward.cpp`, it moves to `src/model/gemma4/`.
- **CPU host interpreter stays the production PTIR engine** (rule 10). Metal
  gets a GPU sampling plane only if profiling shows the host hop matters at
  metal's batch sizes; that is a feature project, not this refactor.
- **Storage adoption is specified separately.** This refactor originally kept
  Metal's loader unchanged; [storage-refact-and-metal.md](storage-refact-and-metal.md)
  supersedes that carve-out. The live path now executes the runtime compiler's
  arena plan and retains mmap only as the payload source.
- **`qwen36` -> `qwen3_5`** so the same model family greps identically across
  drivers. Display strings that reach users may keep "qwen3.6" if that is the
  product label; identifiers and paths align with CUDA.
- **No `ops/` module yet.** Created when a second live arch needs shared op
  wrappers. Naming reserved, directory not.
- **Sequencing:** Phases 1-4 are independent of the CUDA plan and can start
  anytime (they are small). Phase 5 waits for CUDA's Phase 5 to land. Phase 6
  requires both trees post-rename.

## Verification playbook

Per phase: configure + build (default and, at Phases 2/7, the MLX variant),
full `ctest`, one live serve + client inferlet run against the qwen3.6
checkpoint. All of it runs locally on Apple Silicon; this is the one driver
where the full loop needs no workstation. Phase 5 adds the decode-latency
before/after check via `decode_timing`. `caps_honesty_test` runs in every gate;
it is the canary for accidental capability drift behind the frozen ABI.

## Out of scope

- Promoting the MLX cluster to a second forward path.
- gemma4 productization (bring-up continues in `tools/rawmetal/`; wiring it into
  the executor is its own effort with its own parity gates).
- A GPU-side PTIR execution plane for metal (tier-0 analog).
- Additional Metal model-family schemas beyond the runtime compiler support in
  `storage-refact-and-metal.md`.
- `driver/portable` and `driver/dummy`: untouched.
- Lifting the 4096-token / single-ring caps: capacity work, not layout work.

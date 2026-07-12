# CUDA driver refactor: `driver/cuda` north star

Handoff plan for restructuring the CUDA driver. Self-contained: everything needed to
execute is in this file. Written 2026-07-11 against `dev` (parent commit `bbd0b382`,
with the engine-side refactor of `major_refactor.md` in progress in the same working
tree). Companion document to `major_refactor.md` (engine side); this is the C++ side
of the same cleanup.

## Goal

`driver/cuda` is ~106k lines (src 81k, tests 17k, vendored third_party 8k) and carries
the sediment of the multi-lane development period: an abandoned sampling-IR JIT
experiment, a never-fired native-MTP subsystem, two superseded sampler generations,
a GGUF path with zero callers, and a 1064-line CMakeLists with blocks that can never
execute. Structurally, `entry.cpp` mixes the C ABI shim with composition and keeps
shadow copies of registries that `ptir_dispatch` also owns, and the model layer
requires editing three hand-synced dispatch sites plus a 14-member union to add one
architecture.

The rewrite happens entirely behind the frozen 11-function ABI. Target: a tree where
every module name is a membership test, registries have one owner, adding a model
touches one registry entry plus its own family directory, and roughly 15-20k lines
(including tests and CMake) are simply gone.

What this plan is NOT: a model consolidation. Per In Gim's call (2026-07-11), model
forwards stay separate per family; they look similar but deviate in practice.
Deduplication happens at the op/kernel level only, where it already exists.

## Current state (audit summary, 2026-07-11)

| Area | Lines | Verdict |
|---|---|---|
| `src/entry.cpp` + `main.cpp` | 1,724 | ABI shim is clean (11 exports, all live); the rest mixes composition, registries, arch dispatch |
| `src/executor/` | 3,161 | Live fire path + ~550 lines of dead native-MTP machinery + unreachable rank-0 graph replay |
| `src/ptir/` | 5,473 | Live production path (tier-0 runtime, channels, descriptor resolve); `host_eval.hpp`/`tier1_codegen.hpp` are test-only |
| `src/sampling_ir/` | 1,309 | Fossil except `frame_carrier*` (live via `ptir_dispatch`) |
| `src/model/` | 27,333 | All models reachable; triple string dispatch, fat union, leaked workspace type |
| `src/ops/` + `src/kernels/` | 27,908 | Mostly live; two dead kernels, two test-only sampler kernels, orphaned AWQ launchers |
| `src/loader/` | 7,737 | Live pipeline (Rust planner + C++ storage executor) except `gguf_source` |
| `src/` top-level files | 8,169 | Live except `parity_harness`, `hf_snapshot` |
| `tests/` | 16,769 | 17 files reference sampling-IR headers that do not exist; ~10 more are unwired manual harnesses |
| `CMakeLists.txt` | 1,064 | ~100 lines permanently inert (existence-guarded on deleted sources) |

## Design rules

1. **The ABI is frozen.** The 11 `pie_cuda_*` functions and `pie_driver_abi.h` do not
   change. The Rust engine is untouched by this plan. Every phase must leave the
   driver linkable against the same `runtime/engine/src/driver/backend/cuda.rs`.
2. **Idiomatic C++ naming.** A file is named for the class it defines
   (`kv_cache.cpp` : `KvCache`); a module name is a membership test; classes are
   unprefixed inside `namespace pie::cuda` (the handle-based C ABI maps to a
   `Context`, the cuDNN/cuBLAS pattern). Namespace normalization rides along
   per-file as files are touched; no big-bang symbol rename.
3. **One forward path, PTIR-only.** `pie_cuda_launch` already rejects non-PTIR
   batches; keep it that way. New capability lands as PTIR program capability plus,
   at most, store/kernel generality. Never a second submit path. (Dumb-driver
   principle: heavy lifting belongs in guest programs, the driver stays general.)
4. **No unified decoder.** Each model family keeps its own explicit forward.
   Existing sharing stays (`llama_like` absorbing the true llama-likes, `mistral3`
   and `gpt_oss` delegating), but nothing else gets folded. Shared code lives in
   `ops/` and `kernels/` as plain functions every forward calls.
5. **Registries have one owner.** Program, instance, and channel state live in
   `pipeline/`; the context validates handles and delegates. No shadow copies of
   program bytes or instance maps at the entry layer.
6. **Every file builds or does not exist.** Every file under `tests/` is referenced
   by a CMake target or deleted. No existence-guarded CMake blocks, ever. Enforced
   by a CI check (Phase 8).
7. **Deletion before movement.** Dead code is removed in place first, so every later
   move diff is readable and every moved line is a live line.

## Target layout

```
driver/cuda/src
├── abi.cpp             C ABI: pie_cuda_* wrappers, validation, handle -> Context
├── context.{hpp,cpp}   pie::cuda::Context: RAII create/destroy, owns device state,
│                       memory plan, TP bootstrap, composition of all parts
├── main.cpp            standalone diagnostic binary (unchanged)
├── tensor.{hpp,cpp}    core device tensor type (unchanged, top-level)
├── distributed.{hpp,cpp}  NCCL comms owned by Context (unchanged, top-level)
├── batch/              one submitted batch -> device work
│   ├── compose.cpp     LaunchView parse, descriptor resolve, forward-batch
│   │                   composition, BRLE/dense masks, KV-write descriptors,
│   │                   sampling rows
│   ├── forward.cpp     ForwardFn glue, run_forward_dispatch, CUDA-graph
│   │                   capture/replay lattice
│   ├── logits.cpp      selected-row gather, bf16 -> f32 cast
│   ├── tp.cpp          broadcast, follower serve loop, CPU gates
│   ├── persistent_inputs.{hpp,cpp}
│   ├── workspace.{hpp,cpp}   attention workspace (from attention_workspace.cpp)
│   └── brle.{hpp,cpp}
├── pipeline/           guest programs (from src/ptir/ + the live sampling_ir remnant)
│   ├── registry.{hpp,cpp}  program/instance/channel ownership (absorbs the id
│   │                       maps from entry.cpp; single copy)
│   ├── dispatch.{hpp,cu}   from ptir_dispatch.*
│   ├── channels.hpp  channel_registry.hpp  descriptor_resolve.hpp
│   ├── batch_compose.hpp  program_runtime.hpp
│   ├── tier0/              tier0_runner.hpp, tier0_launch.hpp, tier0_kernels.cuh
│   └── frame_carrier.{hpp,cpp}, frame_carrier_kernels.cu
├── model/
│   ├── registry.{hpp,cpp}  THE arch table: string -> {config hook, binder, factory}
│   ├── imodel.hpp          opaque workspace in body(); capabilities
│   ├── workspace.{hpp,cpp} neutral forward workspace (ex-Qwen3Workspace)
│   ├── weight_store.{hpp,cpp}  bound_weights owned per-model (union dies)
│   ├── config.{hpp,cpp}    HfConfig parser (moved from loader/), core struct +
│   │                       per-family sections
│   └── llama_like/ gemma/ gemma3n/ gemma4/ mixtral/ qwen3_5/ nemotron_h/
│       deepseek_v4/ kimi/ glm5/ qwen3_vl/ csm/     one dir per family:
│       weights binder, forward, IModel impl, vision/audio adapters
├── ops/                attention / GEMM / MoE / SSM wrappers (names unchanged)
├── kernels/            fused kernels (names unchanged), + custom_all_reduce.cu
├── store/              long-lived device memory pools
│   ├── kv_cache.{hpp,cpp}  kv_cache_format.{hpp,cpp}  mla_cache.{hpp,cpp}
│   ├── dsa_cache.{hpp,cpp} (stub today; keep, see decision log)
│   ├── recurrent_state_cache.{hpp,cpp}  swap_pool.{hpp,cpp}
│   └── memory_planner.{hpp,cpp}   (from cuda_memory_planner.*, prefix dropped)
└── loader/             LoadPlan -> WeightStore
    ├── safetensors_manifest.* checkpoint_source.{hpp,cpp}
    ├── load_plan_bridge.hpp load_plan_executor.hpp
    ├── transcode_engine.hpp weight_copy_engine.hpp weight_store_codec.hpp
    └── (hf_config moves OUT to model/config; gguf_source deleted)
```

Modules that cease to exist: `src/executor/` (becomes `batch/`), `src/ptir/`
(becomes `pipeline/`), `src/sampling_ir/` (frame_carrier moves to `pipeline/`, rest
deleted), `src/pie_driver_common/` (manifest header moves to `loader/`, dead headers
deleted), `entry.cpp` (split into `abi.cpp` + `context.cpp`).

## Phase 0: baseline

The working tree carries uncommitted engine-refactor work (scheduler/, pipeline/
moves per `major_refactor.md`) plus small driver edits. Commit or land that state
first so this plan's diffs are isolated. Then record the baseline: driver builds on
the 4090 workstation, `cuda_plain_gen` passes, `tests/load_parity/run.py` passes for
the named recipes.

## Phase 1: deletions (no behavior change)

Everything here was verified dead by the 2026-07-11 audit (zero reachable callers,
checked across `driver/`, `runtime/`, `bin/`). Delete, build, let the linker confirm.

### 1a. Native-MTP draft machinery in `executor.cpp` (~550 lines)

The executor-level MTP draft/graph subsystem is a closed dead component: nothing
sends `TP_MTP_MAGIC`, and `run_mtp_draft_with_argmax` / `tp_broadcast_mtp_inputs`
have no callers. Delete:

- `executor.cpp`: `profile_mtp_process_call` (337), `launch_mtp_argmax` (388),
  `MtpGraphKey`/`MtpChainGraphKey` + hashes (430-475), `capture_mtp_graph_exec`
  (476), `capture_mtp_chain_graph_exec` (521), `try_run_mtp_graph_with_argmax`
  (571), `try_run_mtp_chain_graph_with_argmax` (607), both
  `run_mtp_draft_with_argmax` overloads (640, 675), `tp_broadcast_mtp_inputs` (971),
  the `mtp_trace_*` / `mtp_argmax_profile_*` / `mtp_process_profile_*` helpers
  (189-261), `mtp_chain_graph_enabled` (287), `mtp_graph_max_global_tokens` /
  `_bucket` (367-387), and the follower `TP_MTP_MAGIC` branch (2133-2168).
- Dead env plumbing: `PIE_MTP_CHAIN_GRAPH`, `PIE_MTP_GRAPH_MAX_GLOBAL_TOKENS`,
  `PIE_MTP_ARGMAX_PARTS`, `PIE_MTP_PROCESS_PROFILE(_LIMIT)`, `PIE_MTP_TRACE_LIMIT`,
  `PIE_SPEC_VERIFY_GRAPH_MAX_R`, `PIE_SPEC_VERIFY_GRAPH_MIN_R`.
- Dead helpers: anon `qwen35_small_spec_graph_tokens` (161, duplicate of the live
  one in `model/qwen3_5_config.cpp`), `small_spec_graph_max_requests` (170),
  `small_spec_graph_min_requests` (179).

**Do NOT delete:** the model-level MTP kernels in `ops/attention_naive.cu`
(`launch_attention_mtp_paged_history_bf16`, `launch_mtp_shift_hidden_bf16`,
`launch_mtp_update_pending_hidden_bf16`) and the qwen3_5 MTP forward paths; those
are live (PTIR-driven MTP). `wire_system_drafter` (`entry.cpp:1039-1065`) and
`PIE_MTP_DRAFT_TOKENS` are configured-but-never-fired; delete them only after the
`cuda_mtp_*` e2e tests (`bin/pie/tests/cuda_mtp_native_verify.rs`,
`cuda_mtp_specdecode_ab.rs`, `cuda_mtp_stage1.rs`, `cuda_mtpverify.rs`) pass with
the deletion in place. If any of them regress, the audit missed a path; stop and
re-check.

### 1b. sampling-IR fossils

- `src/sampling_ir/tensor_io.{cpp,hpp}`: compiled into the lib, instantiated by
  nothing except its own test. Delete file + `CMakeLists.txt:783-789`
  (`test_tensor_io_device`) + `tests/test_tensor_io_device.cpp`.
- `src/sampling_ir/thread_pool.hpp`: included by nothing.
- CMake blocks guarded on nonexistent `src/sampling_ir/codegen.cpp` +
  `runtime.hpp`: lines 568-573 and 863-946. Permanently inert; the `add_test`
  entries inside reference executables with no `add_executable`.
- The 17 test files that include sampling-IR headers that do not exist in this
  tree: `tests/bench_mbatch_occupancy.cu`, `bench_sampling_ir.cu`,
  `sampling_ir_codegen_test.cu`, `sampling_ir_removal_gate.cu`,
  `sampling_ir_eval_mock_xcheck.cu`, `sampling_ir_reader_test.cpp`,
  `sampling_ir_dispatch_test.cpp`, `jit_load_test.cpp`,
  `test_sampling_ir_backend.cpp`, `test_sampling_ir_batched.cu`,
  `test_sampling_ir_batched_matrix_reject.cpp`, `test_sampling_ir_executor_parity.cu`,
  `test_sampling_ir_host_submit.cu`, `test_sampling_ir_jit.cpp`,
  `test_sampling_ir_recognizer.cpp`, `test_sampling_ir_selfspec_decode.cpp`,
  `test_sampling_ir_spec_verify.cu`, plus any `sampler_reference.hpp`-style support
  headers only they include.

**Keep:** `frame_carrier.{cpp,hpp}` + `frame_carrier_kernels.cu` (live via
`ptir_dispatch.cu`; they move to `pipeline/` in Phase 2) and their device test.

### 1c. Superseded samplers

The PTIR tier-0 IR sampler is the only production sampling path. Delete
`kernels/sample_temp.cu` (435) and `kernels/sample_flashinfer.cu` (114) plus their
only consumers, all test/bench targets: `sampler_parity_test` (CMake 552),
`bench_flashinfer_baseline` (535), `sample_flashinfer_test` (521), and the
corresponding `tests/` sources.

### 1d. Dead loader / top-level / kernel files

- `src/loader/gguf_source.{cpp,hpp}` (479+): `GgufCheckpointSource` has zero
  construction sites; GGUF planning lives in the Rust load planner.
- `src/parity_harness.{cpp,hpp}` (597): `run_parity` has no caller and no ABI
  export. Also delete the two functions it kept alive:
  `qwen3_forward_prefill` and `qwen3_forward_paged`
  (`model/qwen3_forward.cpp:90-371`). Keep `Qwen3Workspace` and
  `qwen3_workspace_bytes` in that file; they are the de-facto universal workspace
  until Phase 5 renames them.
- `src/hf_snapshot.{cpp,hpp}` (82): all three functions uncalled; drop the stale
  include at `entry.cpp:38`.
- `kernels/scatter_int32.cu` (49): no callers anywhere.
- `kernels/beam_mask_adapter.cu` (82): never added to CMake; duplicates
  `pack_dense_mask.cu`. Delete with `tests/beam_mask_adapter_test.cu`.
- `bind_qwen3` compatibility alias (`model/qwen3.hpp:128-129`): no callers.
- `entry.cpp` `ProgramRecord::canonical` / `::sidecar` byte vectors
  (written at 1219-1223, never read; `ptir_dispatch`'s program cache owns the
  decoded bytes).
- Verify-then-delete (grep found no includer; confirm before removing):
  `kernels/softmax_common.cuh`, `pie_driver_common/tensor_names.hpp`,
  `pie_driver_common/shard_plan.hpp`, `pie_driver_common/hf_config_json.hpp`.
- Orphaned AWQ repack launchers in `kernels/dtype_cast.cu`
  (`launch_awq_qweight_to_gptq_w4`, `launch_awq_qzero_to_marlin_w4`): no callers;
  AWQ exists only as a config string today. Delete the launchers, keep the file.

### 1e. Orphan test triage

Remaining unwired tests reference sources that exist; they are manual harnesses.
Apply rule 6: wire or delete.

- Wire into CMake (they back the parity scripts in `scripts/`):
  `csm_backbone_parity.cu`, `csm_depth_decoder_parity.cu`, `csm_generate_parity.cu`,
  `mimi_decoder_full_parity.cu`, `gemma4_audio_full_parity.cu`,
  `gemma4_vision_full_parity.cu`, `gemma4_vision_full_parity_bf16.cu`,
  `gemma4_vision_patch_parity.cu`, `qwen3_vl_vision_full_parity.cu`,
  `test_flashinfer_decode_head_dim_guard.cpp`.
- Anything not worth wiring gets deleted, not parked.

Gate: driver builds clean; `ctest` green for all remaining targets; on the 4090:
`cuda_plain_gen`, then the full `bin/pie/tests/cuda_*` sweep with special attention
to `cuda_mtp_*`; `tests/load_parity/run.py` recipes green. Expected delta: roughly
-2.7k lines in `src/`, -7k in `tests/`, -200 in CMake, zero behavior change.

## Phase 2: `pipeline/` (rename `ptir/`, unify registries)

1. `git mv src/ptir src/pipeline`, drop the prefix stutter:
   `ptir_dispatch.{hpp,cu}` -> `dispatch.{hpp,cu}`; `tier0_*` files gather under
   `pipeline/tier0/`. The word "ptir" survives only where the IR wire format is
   touched: `interface/ptir`, the `pie_native/ptir_*` headers in `driver/common`,
   and imports of those. This mirrors the engine's `ptir/` -> `pipeline/` rename,
   so the same concept has the same name on both sides of the FFI.
2. Move `frame_carrier.{cpp,hpp}` + `frame_carrier_kernels.cu` from
   `src/sampling_ir/` into `pipeline/`; delete the emptied `sampling_ir/`.
3. Relocate test-only headers out of the production tree: `host_eval.hpp` and
   `tier1_codegen.hpp` move to `tests/support/` (they are included only by
   `ptir_tier0_test.cu`, `ptir_runner_test.cu`, `ptir_tier1_test.cu`).
4. **Registry unification.** Create `pipeline/registry.{hpp,cpp}` owning program
   records, instance bindings, and channel handles, including id allocation
   (`next_program_id_`, `next_instance_id_` move here from `entry.cpp:390-397`).
   `entry.cpp` keeps only handle validation and delegates. This removes the
   duplicated instance->program_hash map (`entry.cpp::instances_` vs
   `ptir_dispatch.cu:54`) and centralizes the five scattered lazy
   `make_unique<PtirDispatch>()` guards (`entry.cpp:1202,1236,1262`;
   `executor.cpp:1518,1628,1642,2042`) into one construction site.

Gate: build + `ctest` (renamed `ptir_dispatch_bind`, `ptir_dispatch_race` incl. the
TSAN run) + cuda e2e sweep. Grep gate: `grep -rn "ptir" driver/cuda/src --include='*.cpp' --include='*.cu' --include='*.hpp' | grep -v "pie_native\|interface/ptir"`
returns nothing.

## Phase 3: split `entry.cpp` into `abi.cpp` + `context.{hpp,cpp}`

1. `abi.cpp`: the 11 `extern "C"` wrappers, argument validation, and the
   handle -> `Context` mapping. Nothing else. The moment a wrapper grows an
   algorithm, the algorithm moves into `Context` or below.
2. `context.{hpp,cpp}`: `class pie::cuda::Context` (the current `CudaDriver`,
   renamed). `Context::create(config)` is the composition root: parse config,
   load model via `loader/`, run the memory planner, allocate stores, construct
   the batch engine and pipeline registry, bootstrap TP. The destructor tears
   down in reverse. `main.cpp` (diagnostic shim) is unchanged.
3. The arch allowlist (`entry.cpp:470-498`) and per-arch construction blocks
   (`entry.cpp:984-1090`) move verbatim into `Context::create` for now; they
   dissolve into the model registry in Phase 5. Do not restructure them here;
   this phase is a mechanical split.

Gate: build + cuda e2e + one live serve/client inferlet run on the workstation.

## Phase 4: `executor/` -> `batch/`

1. Split `executor.cpp` (post-Phase-1 it is ~1,800 lines) along its existing
   seams:
   - `batch/compose.cpp`: LaunchView span parsing, `resolve_descriptors` call,
     `compose_forward_batch` / `wire_program_sample_offsets`, BRLE decode +
     dense-mask pack, explicit KV-write descriptor upload, rs-slot plumbing,
     sampling-row build. `brle.{cpp,hpp}` moves here from `src/`.
   - `batch/forward.cpp`: `ForwardFn`, `run_forward_dispatch`,
     `capture_forward_graph_lattice`, graph cache. `forward_graph.hpp`,
     `graph_variant.hpp`, `persistent_inputs.*` ride along.
   - `batch/logits.cpp`: `launch_gather_bf16_rows` / `launch_cast_bf16_to_fp32`
     staging and the bf16/f32 scratch.
   - `batch/tp.cpp`: `TpFireHeader`, `tp_broadcast_inputs`, `tp_follower_serve`,
     CPU gates, shutdown.
   - `attention_workspace.{cpp,hpp}` -> `batch/workspace.{cpp,hpp}`.
2. **Graph-capture decision (behavior change, isolated commit).** Rank-0 graph
   replay is unreachable: `graph_shape_ok` is hardwired false at the sole call
   site (`executor.cpp:1992`), yet startup captures the full decode lattice
   (`entry.cpp:1113`), which only TP followers replay. Gate the lattice capture
   on `tp_size > 1` now (saves single-GPU startup time and VRAM). Keep the replay
   machinery; re-enabling rank-0 replay is future perf work, not cleanup. While
   here, delete the never-taken padding branch in `make_forward_input_views`
   (1047-1082) and the five hardwired-false `ForwardDispatchInputs` flags at
   1991-1996 together with their dead branches, OR wire them for real; do not
   leave flags that are constant at the only call site.

Gate: build + cuda e2e with emphasis on concurrency/timing tests
(`cuda_runahead*.rs`, `cuda_overlap23.rs`, `cuda_mbatch34.rs`,
`cuda_concurrent.rs`), plus a single-GPU startup-time and VRAM before/after
measurement for the capture gating.

## Phase 5: `model/` (registry, workspace, per-family dirs)

The three changes that make "adding a model touches one place" true, without
touching any forward math:

1. **One arch registry** (`model/registry.{hpp,cpp}`). A single table:
   `arch string -> { config section hook, weight binder, IModel factory,
   capabilities }`. It replaces: the allowlist string chain
   (ex-`entry.cpp:470-498`, ~33 strings), the binder if/else chain with its
   silent llama_like fallback (`bound_model.cpp:26-131`), the `Kind` enum + 11
   `is_<arch>_arch` flags + per-arch construction blocks (ex-`entry.cpp:984-1090`).
   Unknown arch = load error, never a silent fallback. Kind-sharing entries
   (gpt_oss -> mixtral binder, mistral3/phi3/olmo -> llama_like, deepseek_v2/v3 +
   kimi_k2 -> kimi) become explicit aliases in the table.
2. **Kill the union.** `BoundCudaModel` (`bound_model.hpp:42-75`, 14 weight
   structs constructed for every load) dissolves; each `IModel` impl owns its own
   weights struct, produced by its binder. `bound_model.{cpp,hpp}` cease to exist.
3. **Opaque workspace.** `IModel::body` stops taking `Qwen3Workspace&`. Rename
   `Qwen3Workspace` -> `pie::cuda::Workspace` in `model/workspace.{hpp,cpp}` (it
   is already the de-facto universal scratch; this is a rename, not a redesign)
   and give `IModel` a `workspace_bytes()` hook so a family can diverge later
   without touching the interface. This removes the `qwen3_forward.hpp` include
   from the 11 unrelated TUs that only want the workspace.
4. **Per-family directories.** `git mv` the flat 104-file `model/` into family
   dirs (see target layout). File contents unchanged in this phase. `qwen3.cpp`
   (weights + binder, post-Phase-1) merges into `llama_like/`.
5. **Config moves home.** `loader/hf_config.{cpp,hpp}` -> `model/config.{hpp,cpp}`.
   Keep the single parser; split the 479-line god struct mechanically into a core
   struct + per-family section structs owned by the registry entries. Parsing
   behavior identical; this is a layout change so a new family adds a section
   instead of editing a monolith. (`qwen3_5_config.cpp` is env-var knobs, not
   config; it moves into `model/qwen3_5/` as-is.)

Coverage prerequisite: `tests/load_parity/spec.py` recipes do not cover
`nemotron_h`, `gemma4`, `gemma3n`, `qwen3_vl`, `csm`, `gpt_oss`. Before moving
those families, add a load-parity recipe or at least a serve smoke test per
family, so the moves are gated by something.

Gate: full `load_parity` recipe sweep + cuda e2e + one serve/client run per newly
covered family. Grep gate: `grep -rn "is_.*_arch\|BoundCudaModel" driver/cuda/src`
returns nothing.

## Phase 6: `store/` + remaining top-level tidy

Mechanical moves, no logic changes:

- -> `store/`: `kv_cache.*`, `kv_cache_format.*`, `mla_cache.*`, `dsa_cache.*`,
  `recurrent_state_cache.*`, `swap_pool.*`, and `cuda_memory_planner.*` renamed
  `memory_planner.*` (the `cuda_` prefix stutters inside driver/cuda).
- `custom_all_reduce.cu` (+ stub) -> `kernels/`.
- `pie_driver_common/safetensors_manifest.hpp` -> `loader/`; delete the emptied
  `pie_driver_common/` (its other headers died in Phase 1d).
- `tensor.{cpp,hpp}` and `distributed.{cpp,hpp}` stay top-level next to
  `context.*` (core type; Context-owned comms).
- Doc fix while touching `loader/`: the "driver's own C++ checkpoint compile"
  comments (`loaded_model.hpp:44-55`, `load_plan_bridge.hpp`) are wrong;
  both branches execute the Rust-compiled LoadPlan, the difference is
  compile-via-FFI vs deserialize-from-file. Say so.

Gate: build + ctest + `cuda_plain_gen` (moves only; a full sweep at Phase 8).

## Phase 7: bench-gated kernel consolidation (optional, ordered last)

Each item changes hot-path behavior and needs a measurement, not just parity.
None of them block Phases 1-6; skip freely.

1. **XQA vs FlashInfer decode.** The hand-written XQA family
   (`ops/attention_xqa*.cu`, 6 files, ~1.9k lines) is used only by `llama_like`
   decode. Benchmark decode latency vs FlashInfer's decode across the GQA ratios
   on the 4090; if FlashInfer is within an acceptable margin, delete the family,
   else keep and document why.
2. **One MoE backend.** CUTLASS grouped-GEMM (`ops/flashinfer_moe.cu`) serves only
   nemotron_h; everything else uses hand-written `kernels/moe_dispatch.cu`.
   Benchmark, pick one, port the rest. Until then both stay.
3. **Prune `argmax.cu`** (1,175 lines): keep the fused lm-head-argmax, int8 GEMV,
   and TP-reduce variants with callers; delete unreferenced variants (audit each
   launcher's call sites first; several are suspected orphans).
4. **Demote `envelope.cu`, `geometry.cu`, `gather_tokens.cu`** out of the
   production lib into their test targets only (each has exactly one test
   consumer and no model/executor caller), or wire them into the live path if the
   PTIR roadmap still wants them (Quest envelopes, device geometry). Decide, do
   not park.

Gate per item: parity test + before/after numbers recorded in the PR.

## Phase 8: CMake rewrite, CI gates, doc sweep

1. Rewrite `CMakeLists.txt` (~250-300 lines): one lib target with the explicit
   source list matching the new tree, arch gating (SM90/SM100/SM120 stubs) kept,
   Marlin option kept, vendored `flashinfer_generated` kept, every test target
   present, zero existence-guarded blocks.
2. CI gates (a script under `driver/cuda/`, run in the build workflow):
   - every file under `tests/` appears in CMakeLists (rule 6);
   - no `if(EXISTS` guards around targets;
   - include-layering greps, each must print nothing:
     ```sh
     # kernels is a leaf: no includes from model/, batch/, pipeline/, loader/, store/
     grep -rln '#include "\(model\|batch\|pipeline\|loader\|store\)/' driver/cuda/src/kernels/
     # ops may use kernels/ and store/ headers, nothing above
     grep -rln '#include "\(model\|batch\|pipeline\|loader\)/' driver/cuda/src/ops/
     # loader must not reach into the fire path
     grep -rln '#include "\(batch\|pipeline\)/' driver/cuda/src/loader/
     ```
     (Run these against the tree first; fix or consciously document any existing
     violation before enforcing.)
3. Doc sweep: strip lane codenames and thrust/WS numbering from comments
   (charlie/delta/echo/hotel, "WS6 lane L7", "M3 spike") wherever they no longer
   aid navigation; every module root gets a one-line charter matching this file's
   tree. Fix `driver/cuda/README.md` (documents a `[model.driver]` schema and a
   `doctor` subcommand the C++ binary does not have) and note that `dev.toml` is
   the standalone shim's config, the Rust worker's config is separate.

Gate: clean-clone configure+build on the workstation; full cuda e2e sweep; CI
gates green.

## Move map (grouped)

| Current | Target | Phase |
|---|---|---|
| `entry.cpp` ABI wrappers | `abi.cpp` | 3 |
| `entry.cpp` CudaDriver + composition | `context.{hpp,cpp}` (`pie::cuda::Context`) | 3 |
| `entry.cpp` id maps / ProgramRecord | `pipeline/registry.{hpp,cpp}` (bytes fields deleted) | 1d, 2 |
| `entry.cpp` allowlist + arch blocks | `model/registry.{hpp,cpp}` | 5 |
| `main.cpp` | unchanged | - |
| `executor/executor.cpp` | split: `batch/{compose,forward,logits,tp}.cpp` | 4 |
| `executor/executor.cpp` MTP cluster | deleted | 1a |
| `executor/persistent_inputs.*`, `forward_graph.hpp`, `graph_variant.hpp` | `batch/` | 4 |
| `attention_workspace.*` | `batch/workspace.*` | 4 |
| `brle.*` | `batch/` | 4 |
| `ptir/*` | `pipeline/*` (prefix dropped; tier0 subdir) | 2 |
| `ptir/host_eval.hpp`, `ptir/tier1_codegen.hpp` | `tests/support/` | 2 |
| `sampling_ir/frame_carrier*` | `pipeline/` | 2 |
| `sampling_ir/tensor_io.*`, `thread_pool.hpp` | deleted | 1b |
| `model/*` (per family) | `model/<family>/` dirs, contents unchanged | 5 |
| `model/bound_model.*` | deleted (registry + per-model weights) | 5 |
| `model/qwen3_forward.cpp` forwards | deleted; workspace renamed -> `model/workspace.*` | 1d, 5 |
| `loader/hf_config.*` | `model/config.*` | 5 |
| `loader/gguf_source.*` | deleted | 1d |
| `loader/*` (rest) | unchanged in `loader/` | - |
| `kv_cache*`, `mla_cache*`, `dsa_cache*`, `recurrent_state_cache*`, `swap_pool*` | `store/` | 6 |
| `cuda_memory_planner.*` | `store/memory_planner.*` | 6 |
| `custom_all_reduce.cu` (+stub) | `kernels/` | 6 |
| `pie_driver_common/safetensors_manifest.hpp` | `loader/` | 6 |
| `pie_driver_common/{tensor_names,shard_plan,hf_config_json}.hpp` | verify, then delete | 1d |
| `parity_harness.*`, `hf_snapshot.*` | deleted | 1d |
| `kernels/{sample_temp,sample_flashinfer,scatter_int32,beam_mask_adapter}.cu` | deleted | 1c, 1d |
| `ops/*`, `kernels/*` (rest) | unchanged names; Phase 7 prunes bench-gated | 7 |
| `tests/` sampling-IR family (17 files) | deleted | 1b |
| `tests/` manual parity harnesses | wired into CMake or deleted | 1e |

## Decision log

- **ABI frozen; rewrite behind it.** The 11 exports are all live and the Rust FFI
  wrapper is thin; nothing on the engine side changes.
- **No unified decoder skeleton.** In Gim's call (2026-07-11): models look similar
  but deviate in practice. Dedup at op/kernel level only. Existing delegation
  (mistral3/gpt_oss/qwen3 -> llama_like/mixtral) stays; no further folding. Do not
  revisit after Phase 5 lands.
- **Names** (In Gim, 2026-07-11, after several rounds): `abi`, `context`, `batch`,
  `pipeline`, `model`, `ops`, `kernels`, `store`, `loader`. Idiomatic C++: the
  handle-based C ABI maps to a `Context` (cuDNN/cuBLAS pattern); `batch/` is a
  noun module with verb files (TensorRT-LLM's batch_manager precedent); `store/`
  kept over `cache/` to mirror the engine's L1 vocabulary; `loader/` kept as the
  established name. Rejected along the way: session, checkpoint, executor, driver,
  launch, weights, fire, load, execute, run, bootstrap, device, cache.
- **`pipeline/` mirrors the engine rename.** Same concept, same name on both sides
  of the FFI; "ptir" survives only where the IR wire format is touched.
- **Graph lattice gated, replay kept.** Rank-0 replay is currently unreachable
  (hardwired false); the fix now is to stop paying capture cost on single-GPU.
  Re-enabling rank-0 replay is perf work with its own benchmark, not cleanup.
- **Legacy samplers deleted.** The PTIR tier-0 IR sampler is the production path;
  `sample_temp.cu`'s parity-oracle role died with the sampling-IR experiment.
- **`dsa_cache` stub kept.** `allocate()` returns an empty struct today, but GLM5's
  forward takes it by reference; it is an unfinished feature, not dead code. Flag
  with a TODO, do not delete.
- **MTP drafter deletion is test-gated.** The executor-level draft loop is
  provably dead, but `PIE_MTP_DRAFT_TOKENS` appears in bin/pie integration tests;
  the `cuda_mtp_*` suite is the arbiter.
- **XQA and MoE unification are measurements, not opinions.** Ordered last,
  bench-gated, skippable.

## Verification playbook

Per phase: driver configure+build on the 4090 workstation, `ctest` for the driver
targets, then `bin/pie/tests/` cuda e2e (start with `cuda_plain_gen`, then the full
sweep), then `tests/load_parity/run.py` for the named recipes. Phases 3-5 add one
live serve + client inferlet run (the usual workstation loop). Phase 4 adds the
TSAN run of the pipeline dispatch race test and before/after startup-time + VRAM
numbers. Phase 7 items each carry a kernel benchmark.

The driver cannot be compiled on the mac; all build/test gates run on the
workstation.

## Out of scope

- `driver/metal`: same layout and names should apply later (it already mirrors the
  old structure at 1/10 the size), but only after the CUDA tree lands.
- `interface/driver` ABI and `interface/ptir`: unchanged.
- The load planner moved to `runtime/load-planner`; see
  `storage-refact-and-metal.md`.
- Engine-side work: tracked in `major_refactor.md`.
- Re-enabling rank-0 CUDA-graph replay; tier-1 NVRTC fusion productization; the
  `dsa_cache` implementation: real features, separate efforts.
- third_party bumps (flashinfer v0.6.9, vendored Marlin): untouched.

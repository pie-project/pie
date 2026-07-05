# raw_metal — Phase-0 Metal-4 decode scaffolding

Foundation for the MLX-free raw-Metal decode path (qwen3.6, M=1 single-stream greedy).
Owns the Metal-4 plumbing that beta (executor/replay) and delta (heap + kernel ports)
build on. Keyed off `decode_abi.hpp` (delta owns; `Region` / `IoSlot` / `bind::` / `Kernel`).

## Toolchain reality (this box: M1 Max, macOS 26.3, CLT-only)

No offline `metal`/`metallib` compiler is installed (Command Line Tools only, no full
Xcode + Metal Toolchain). **Phase 0 compiles shaders at RUNTIME** via
`[MTLDevice newLibraryWithSource:]` + `MTL4Compiler` — no download needed, fast iteration.
Verified working end-to-end on M1 Max (MTL4 command API is M1+, no feature-gate wall).

The AOT `.metal`→`.metallib` path is the **production** option, deferred to Phase 1
(needs `xcodebuild -downloadComponent MetalToolchain`). `PIE_RAW_METAL_AOT` is an OFF stub.

## What's here

| file | role |
|------|------|
| `mtl4_context.hpp/.mm` | `RawMetalContext`: single placement `MTLHeap` (Shared/UMA) + `MTLResidencySet` + `MTL4CommandQueue` + double-buffered `MTL4CommandAllocator` + per-(kernel,layer) `MTL4ArgumentTable` + runtime PSO compile + per-step encoder |
| `harness.hpp/.cpp` | `LatencyHarness`: single-step decode timer + single-kernel micro-bench, **encode-ms vs gpu-exec-ms reported separately** |
| `harness_main.cpp` | Phase-0 smoke: heap_alloc → arg_bind → make_resident → micro-bench + encode-scaling demo |
| `kernels/gemv_demo.metal` | stand-in M=1 GEMV so the rig runs before real ports land |
| `kernels/*.metal` | delta's ported kernels (rms_norm, qmv, sdpa, rope, gdn-core, …) |

## The two signatures delta + beta build on

```cpp
// (1) heap sub-allocation — bump-allocated from the single resident heap.
//     contents() is the CPU pointer (Shared storage, UMA) for weight staging + IO scalars.
SlotHandle heap_alloc(size_t size, size_t align = 256);

// (2) argument-table bind, keyed by decode_abi.hpp bind:: enums, built ONCE (I2).
//     `layer` disambiguates per-layer dispatch instances; -1 for singletons.
void arg_bind(Kernel k, int layer, uint8_t bind_index, SlotHandle slot, size_t offset = 0);
void arg_bind(Kernel k,           uint8_t bind_index, SlotHandle slot, size_t offset = 0);
```

Invariants honored: **I1** IO scalars bind as GPU-read buffers (never setBytes) → CB
byte-identical per token; **I2** heap resident once + arg tables built once.

## Executor / kernel drop-in

beta's per-step encode (matches `mtl4probe.mm`):

```cpp
StepTiming t = ctx->run_step([&](StepEncoder& se){
    se.encode(pso_embed,   Kernel::EmbedGather, -1, grid, tg);   // set_pso+set_argtable+dispatch+barrier
    for (int L = 0; L < 24; ++L) { /* per-layer DAG via se.encode(...) */ }
    se.encode(pso_argmax,  Kernel::Argmax,      -1, grid, tg);
}, /*ab=*/token & 1);   // double-buffered allocator: encode(N+1) overlaps GPU(N)
// t.encode_ms (CPU build) and t.gpu_exec_ms (GPU) reported separately.
```

delta's micro-bench per ported kernel (A/B vs MLX exec-ms):

```cpp
SlotHandle w = ctx->heap_alloc(bytes);                      // (1)
ctx->arg_bind(Kernel::Rms, L, (uint8_t)bind::Rms::W, w);    // (2)
ctx->make_resident();
Pso pso = ctx->compile_pso_from_file("kernels/rms_norm.metal", "rms_single_row");
BenchResult r = harness.bench_kernel("rms", pso, Kernel::Rms, L, grid, tg);
print_result(r);   // encode-ms | gpu-exec-ms
```

## Build & run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/raw_metal_harness            # prints PHASE0_HARNESS_OK
```

Standalone (NOT wired into the main driver build) for Phase 0. `decode_abi.hpp` is
delta's (committed on her branch); this dir gitignores a local copy to avoid clobbering it.

## Measured on M1 Max (encode is a non-issue, GPU-exec is the gate)

MTL4 re-encode of the full ~322-dispatch step ≈ **0.08–0.11 ms** (~0.3 µs/disp),
matching beta's probe. Encode overlaps GPU via the double-buffered allocator — the only
number that matters is **gpu-exec-ms**, driven down by delta's kernel fusion.

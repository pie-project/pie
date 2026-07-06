# driver/metal/tier0 — PTIR tier-0 interpreter on Metal (full backend)

The macOS/Metal **PTIR sampling-IR execution backend**, certified against echo's
golden vectors — the cross-backend validation of the whole sampling-IR runtime.

## What it is

A host channel-runtime — a C++ port of echo's reference interpreter
(`interface/sampling-ir/src/ptir/interp.rs`: per-phase readiness → pass-local
overlay → **pass-atomic commit** / epoch-ring register semantics) — executing over
**charlie's CUDA-free decoded `Trace`** (`driver/cuda/src/ptir/{container,bound,
trace}.hpp` reused as-is, so the decode matches CUDA exactly), with the compute ops
dispatched to the validated Metal kernels (`../ptir` / `../kvattn`).

- echo's `interp.rs` is the semantic oracle; charlie's `tier0_runner` the
  cross-reference for tricky orchestration.
- Decode → identity: each golden's `container_hash` is verified byte-for-byte.
- Compute ops on Metal: `reduce_argmax`, `reduce_sum/max/min` run on Metal kernels;
  geometry/control (gather/scatter/iota/rem, gumbel RNG, top_k) on host per interp.rs
  (Metal-ising top_k / softmax is the coverage follow-on).

## Certified goldens (behavioral, byte-for-byte / within-tol)

| golden | exercises |
|--------|-----------|
| `counter_pingpong`        | 2-ch counter, back-pressure, InPlace recycle, host_take/WouldBlock |
| `greedy_argmax`           | logits→argmax (Metal), embed_tokens port consume, multi-step |
| `section3_masked_gumbel`  | grammar-mask + gumbel + **late-mask dummy-run** (mid-run miss) |
| `beam_epilogue`           | **16 channels, 82 ops**, host grant, **drain-refill**, top_k / log_softmax / geometry gathers+scatters → `take [3,5]/[0,1]/[-0.00235,-0.00636]` |

`→ TIER0_TEST_OK` (all 4). The single-op matrix goldens (`matrix_select_mask` etc.)
are certified in `../ptir` (`ptir_golden_cert`); the `neg_*` goldens are
validation-rejection cases (echo's validator, not executable).

## Build & run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/tier0_test        # -> TIER0_TEST_OK
```

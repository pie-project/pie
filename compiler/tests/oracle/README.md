# `compiler/tests/oracle/` — the transitional C++ emitter oracles

**Delete this directory when the C++ emitters are deleted.** Nothing links it
into a build.

Both backends' emitters were ported from C++ to Rust (`compiler/codegen/src/`).
These harnesses are how the ports were proven, and how they stay proven while
both copies exist: each links its C++ emitter, runs it over a fixed corpus, and
writes the results to a golden directory that the Rust side must reproduce.

| harness | C++ under test | goldens | Rust comparison |
|---|---|---|---|
| `m1_codegen_dump.cpp` | `driver/metal/src/pipeline/m1_codegen.cpp` | `golden-msl/` | `tests/metal_msl_golden.rs` |
| `cuda_codegen_dump.cpp` | `driver/cuda/src/pipeline/generated/` | `golden-cuda/` | `tests/cuda_golden.rs` |

Neither a Metal SDK nor a CUDA toolkit is needed — the emitters are pure string
builders, and the headers they include are device-free. Metal's `MTL`/`NS`
mentions are all inside MSL literals.

## Re-deriving the goldens

```sh
g++ -std=c++20 -O1 -o /tmp/m1_codegen_dump \
  compiler/tests/oracle/m1_codegen_dump.cpp \
  driver/metal/src/pipeline/m1_codegen.cpp \
  -I driver/metal/src -I driver/common/include -I compiler/codegen/include
/tmp/m1_codegen_dump . compiler/tests/golden-msl

g++ -std=c++20 -O1 -o /tmp/cuda_codegen_dump \
  compiler/tests/oracle/cuda_codegen_dump.cpp \
  -I driver/cuda/src -I driver/common/include
/tmp/cuda_codegen_dump . compiler/tests/golden-cuda

git diff --stat compiler/tests/golden-msl compiler/tests/golden-cuda  # expect empty
```

A non-empty diff means the C++ emitter changed. Port the change to Rust, then
commit both the new goldens and the Rust change together.

## Corpus

`corpus/stage_plans.txt` is the **input** contract: the Rust side compiles every
golden trace in `compiler/tests/golden/` with `pie_plan` and writes the stage
plans out in `PTRP` wire form; the C++ side decodes that file. It is what makes
"both sides saw the same plans" checkable rather than assumed. Regenerate it
with `PTIR_REGEN=1 cargo test -p pie-compiler-tests --test metal_msl_golden`.

The goldens' own `sidecar:` blobs are deliberately not used as the plan source:
three are stale on this branch relative to today's `pie-plan`, and the oracle
has to see exactly what the Rust emitters see.

| emitter | cases |
|---|---|
| `emit_singleton_region_msl` | 256 — every tag byte; the function is total |
| `emit_readiness_msl` / `emit_commit_msl` | 45 each — flag combinations, capacities, pairs, and the 29-channel boundary |
| `emit_grouped_readiness_msl` / `emit_grouped_commit_msl` | 4 each |
| `validate_singleton_plan` | 456 — 19 stages × 24 wire-level mutations |
| the four plan-taking emitters | 192 each — every region of every stage |

## Two known C++ behaviours the Rust port does not reproduce

Both are recorded here because they look like port bugs and are not.

**`emit_grouped_nucleus_msl` reads out of bounds on Generated regions.** Its
guard is `region.library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE || !library_region_valid(...)`,
but `library_region_valid` returns `true` immediately for a non-library region
(`m1_codegen.cpp:172`) and a Generated region's `library_op` byte is
`0 == PTIR_LIBRARY_NUCLEUS_SAMPLE`. So a Generated region walks past the guard
into `region.inputs[0..3]` and `region.outputs[0]`. Its TopK sibling has the
missing `!region.library` check. The harness marks those cases
`cxx-oracle-undefined` and emits nothing; the Rust port rejects them. This is
latent rather than live: `m1_runtime.cpp` only calls the nucleus emitter behind
its own `region_plan.library && library_op == NUCLEUS_SAMPLE` test.

**`validate_singleton_plan` leaves a partly built `operations` vector behind on
rejection**, because it fills the out-param as it walks. `m1_runtime.cpp:939`
returns `reject_deterministic(error)` without reading it, so those entries are
unobservable. The Rust signature is `Result<Vec<M1OpMeta>, String>`, which
discards them; the comparison checks the verdict and message on every case and
the operations only when the verdict is `ok`.

## CUDA corpus

Same corpus file, same argument: the plans come from `corpus/stage_plans.txt`.

| emitter | cases |
|---|---|
| `emit_singleton_region_cuda` | 263 — every tag byte, plus seven entry-name cases |
| `emit_fused_region_cuda` | 320 — every region of every stage, bodies pinned by hash |
| `emit_fused_region_cuda_verbatim` | 36 — one region per stage, kept whole so a diff is readable |
| `validate_generated_region` | 320 |
| `second_party_region_supported` | 320 |
| `singleton_runtime_cuda_source` | 1 — the 45 KB runtime, pinned by hash |

A fused CUDA kernel is 40-70 KB, so checking in all 320 would be a 13 MB
golden nobody reads. Hashing the body and keeping one per stage verbatim costs
1.9 MB and still fails loudly on any change.

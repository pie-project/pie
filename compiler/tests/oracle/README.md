# `compiler/tests/oracle/` — the transitional C++ emitter oracle

**Delete this directory when `driver/metal/src/pipeline/m1_codegen.cpp` is
deleted.** Nothing links it into a build.

The Metal MSL emitters were ported from C++ to Rust
(`compiler/codegen/src/metal/`). This harness is how that port was proven, and
how it stays proven while both copies exist: it links the C++ emitter, runs it
over a fixed corpus, and writes the results to `compiler/tests/golden-msl/`.
`compiler/tests/tests/metal_msl_golden.rs` drives the Rust port over the same
corpus and requires the same bytes.

There is no Metal SDK on Linux, but none is needed — the emitters are pure
string builders whose `MTL`/`NS` mentions are all inside MSL literals.

## Re-deriving the goldens

```sh
g++ -std=c++20 -O1 -o /tmp/m1_codegen_dump \
  compiler/tests/oracle/m1_codegen_dump.cpp \
  driver/metal/src/pipeline/m1_codegen.cpp \
  -I driver/metal/src -I driver/common/include -I compiler/codegen/include
/tmp/m1_codegen_dump . compiler/tests/golden-msl
git diff --stat compiler/tests/golden-msl   # expect empty
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

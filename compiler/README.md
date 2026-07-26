# `compiler/` — the Pie tensor-program toolchain

Authoring eDSL → **PTIR** → planning → CUDA/Metal codegen, plus the reference
interpreter every backend is diffed against.

Pie's public noun for a traced tensor program is a **program** (`forward-pass.program`
in WIT, `program_hash` in Rust). PTIR — the *Pie Tensor IR* — is the format a
program is expressed in: a stage-tagged, channel-carrying trace container over a
closed first-party op set, magic `"PTIR"`. This directory is everything that
produces, checks, plans, or lowers one.

The one-line placement:

> **`compiler/` builds a program, `runtime/` schedules it, `driver/` fires it.**

## Layout

| Crate | Role | |
|---|---|---|
| `dsl/` | `pie-dsl` | **authoring** — Tensor/Channel eDSL + the neutral trace `Builder` that lowers stage closures into a container |
| `ir/` | `pie-ir` | **representation** — types, ops, registry, container + wire format, shape/dtype inference, the bind-time validator, the RNG contract |
| `plan/` | `pie-plan` | **analysis** — normalization, stage signatures, value domains, region partitioning, lane-table ABI → the region plan (`PTRP`) and bound sidecar (`PTIB`) |
| `eval/` | `pie-eval` | **semantics** — the tier-0 reference interpreter and the host partial evaluator |
| `codegen/` | `pie-codegen` | **emission** — the C ABI header, the RNG projections, and (landing next) the CUDA/Metal region emitters |
| `tests/` | `pie-compiler-tests` | **conformance** — golden traces, the malformed-wire corpus, generated-artifact drift checks, cross-implementation parity |

```
   dsl ──┐
         ├──> ir <──┬── eval
                    └── plan <── codegen
```

`ir` is the dependency floor. `plan`, `eval`, and `codegen` are siblings above it;
`plan` and `eval` never depend on each other, because they answer different
questions about the same bound trace — *how do we execute this* versus *what does
it produce*.

## Two terms that are easy to misread

**`plan` is the cuDNN/FFTW sense**, not the LLVM one: a reusable,
shape-parameterized execution strategy, cached by `ExecutableCacheKey`, with
runtime-varying extents kept symbolic so one plan serves many batch shapes. It is
not an optimization pass pipeline — nothing here rewrites a program to make it
faster. It decides which ops fuse into one generated kernel, what falls to a
library kernel, and where each value lands in the lane table.

**`eval` is not test-only.** The interpreter is the golden model, but `pareval`
is a production path with three callers: canonical-KV fire evidence (the prefix
cache folds the geometry prologue instead of pattern-matching the trace),
capability-less execution (a driver with no device-geometry ports has the host
fold the prologue per fire), and geometry classification. It reuses the
interpreter's `eval_op` so there is no second evaluator to drift.

## Rust-first

`compiler/` contains **no hand-written C++**. The only non-Rust files here are
codegen's inputs and outputs: device runtime templates (`.cuh` / `.metal`) that
Rust assembles, and the generated C headers checked in under
`codegen/include/` with drift tests. Backend code generation is a pure
`Plan -> String` function with no device-architecture input, which is what lets it
live outside the driver and be golden-tested without a GPU. Compilation itself
(NVRTC, `MTLLibrary`), module caching, and launch stay in `driver/` — those need
a live device.

## Generated artifacts

`codegen/include/` is checked in and verified by `tests/`. Never edit those files
by hand; change the tables in `ir/` and regenerate.

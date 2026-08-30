# `eta-compiler` — the Pie tensor-program toolchain

Authoring eDSL → **ETA** → planning → CUDA/Metal codegen, plus the reference
interpreter every backend is diffed against.

Pie's public noun for a traced tensor program is a **program** (`forward-pass.program`
in WIT, `program_hash` in Rust). ETA — the *Embedded Tensor Algebra* IR — is the format a
program is expressed in: a stage-tagged, channel-carrying trace container over a
closed first-party op set, magic `"ETA"`. This crate is everything that
plans, checks, or lowers one.

The one-line placement:

> **`eta-*` builds a program, `runtime` schedules it, `engine-*` fires it.**

## Layout

Three crates and a test-only fourth folded into this one, module per role. Two
did NOT fold, and the line between them is what a **guest** imports:

| | | |
|---|---|---|
| `eta-dsl` | *its own crate* | **authoring** — Tensor/Channel eDSL + the neutral trace `Builder` that lowers stage closures into a container. Compiled into the wasm inferlet. |
| `eta-ir` | *its own crate* | **representation** — types, ops, registry, container + wire format, shape/dtype inference, the bind-time validator, the RNG contract. `no_std`; the dependency floor both ends sit on. |
| `plan/` | module | **analysis** — normalization, stage signatures, value domains, region partitioning, lane-table ABI → a `CompiledStage` handed straight to emission (nothing is serialized on the way out) |
| `eval/` | module | **semantics** — the tier-0 reference interpreter and the host partial evaluator |
| `codegen/` | module | **emission** — the RNG projections, the CUDA/Metal region emitters, and the launch package the engines execute |
| `tests/` | the battery | **conformance** — golden traces, container mutation sweeps, generated-artifact drift checks, cross-implementation parity |

```
   eta-dsl ──┐
                ├──> eta-ir <──┬── eval
                                  │
                                  └── plan <── codegen ──> pie-driver-abi
                                                           (crates/driver-abi)
```

`eta-ir` is the dependency floor. `plan`, `eval`, and `codegen` are siblings
above it; **`plan` and `eval` never depend on each other**, because they answer
different questions about the same bound trace — *how do we execute this* versus
*what does it produce*. Cargo used to enforce that by not having the edge;
`tests/module_layering.rs` enforces it now, and says why.

`codegen` is the only module here that reaches outside the toolchain, and only
for the launch package: it builds one out of `driver-abi`, the same
declarations the engines read it back with. That crate is a contract, not an
engine — it depends on nothing but serde — so the two ends of the host→engine
ABI are one declaration rather than two copies kept in step by hand.

## Why one crate

The three shipped together, versioned together, and were consumed together by
`runtime`. What the split cost was the conformance battery: no one of the three
could own tests spanning all of them, so it needed a fourth crate
that existed only to hold dev-dependencies. Folded, that
is just `tests/`.

The fold gave up one thing: `plan` alone was `no_std` (+ `alloc`). `eval`'s
extern channels are `std::sync::{Arc, Mutex}`, so one crate holding both cannot
be. Nothing depended on it — a guest reaches `eta-dsl` and `eta-ir`, and
those keep theirs.

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
capability-less execution (an engine with no device-geometry ports has the host
fold the prologue per fire), and geometry classification. It reuses the
interpreter's `eval_op` so there is no second evaluator to drift.

## Rust-first

This crate contains **no hand-written C++**. The only non-Rust files are
codegen's inputs and outputs: device runtime templates (`.cuh` / `.metal`) under
`runtime/` that Rust assembles, and the one generated shader preamble checked in
under `include/` with a drift test. Backend code generation is a pure
`Plan -> String` function with no device-architecture input, which is what lets
it live outside the engine and be golden-tested without a GPU. Compilation
itself (NVRTC, `MTLLibrary`), module caching, and launch stay in the engines —
those need a live device.

## Generated artifacts

`include/ptir_rng.generated.metal` is checked in and verified by
`tests/rng_contract.rs`. Never edit it by hand; change `eta-ir`'s
`RNG_FORMULA` and regenerate with
`PTIR_REGEN=1 cargo test -p eta-compiler --test rng_contract`.

It is a file rather than a Rust `const` because Metal's runtime shader compiler
resolves `#include "..."` against the including file's directory and nothing
else, so `runtime/metal/ptir_m1_runtime.metal` can only reach the preamble if a
copy sits beside it — `kernels-metal/build.rs` stages one on a `native` build.

Two C headers stood beside it, `ptir_abi.h` (op tags, dtype/stage/port enums,
the arity table) and `rng_contract.generated.h` (the RNG contract in C). Both
existed for the C++ drivers to `#include`. Those drivers were deleted, every
backend is Rust and reads `eta-ir` directly, and NVRTC is called with zero
headers and zero include names — so nothing could have `#include`d them even by
accident. They were removed with their generators and drift tests.

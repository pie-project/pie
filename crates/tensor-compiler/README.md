# `tensor-compiler` — the Pie tensor-program toolchain

Authoring eDSL → **PTIR** → planning → CUDA/Metal codegen, plus the reference
interpreter every backend is diffed against.

Pie's public noun for a traced tensor program is a **program** (`forward-pass.program`
in WIT, `program_hash` in Rust). PTIR — the *Pie Tensor IR* — is the format a
program is expressed in: a stage-tagged, channel-carrying trace container over a
closed first-party op set, magic `"PTIR"`. This crate is everything that
plans, checks, or lowers one.

The one-line placement:

> **`tensor-*` builds a program, `engine` schedules it, `driver-*` fires it.**

## Layout

Three crates and a test-only fourth folded into this one, module per role. Two
did NOT fold, and the line between them is what a **guest** imports:

| | | |
|---|---|---|
| `tensor-dsl` | *its own crate* | **authoring** — Tensor/Channel eDSL + the neutral trace `Builder` that lowers stage closures into a container. Compiled into the wasm inferlet. |
| `tensor-ir` | *its own crate* | **representation** — types, ops, registry, container + wire format, shape/dtype inference, the bind-time validator, the RNG contract. `no_std`; the dependency floor both ends sit on. |
| `plan/` | module | **analysis** — normalization, stage signatures, value domains, region partitioning, lane-table ABI → a `CompiledStage` handed straight to emission (nothing is serialized on the way out) |
| `eval/` | module | **semantics** — the tier-0 reference interpreter and the host partial evaluator |
| `codegen/` | module | **emission** — the C ABI header, the RNG projections, the CUDA/Metal region emitters, and the launch package the drivers execute |
| `tests/` | the battery | **conformance** — golden traces, container mutation sweeps, generated-artifact drift checks, cross-implementation parity |

```
   tensor-dsl ──┐
                ├──> tensor-ir <──┬── eval
                                  │
                                  └── plan <── codegen ──> pie-driver-abi
                                                           (crates/driver-abi)
```

`tensor-ir` is the dependency floor. `plan`, `eval`, and `codegen` are siblings
above it; **`plan` and `eval` never depend on each other**, because they answer
different questions about the same bound trace — *how do we execute this* versus
*what does it produce*. Cargo used to enforce that by not having the edge;
`tests/module_layering.rs` enforces it now, and says why.

`codegen` is the only module here that reaches outside the toolchain, and only
for the launch package: it builds one out of `driver-abi`, the same
declarations the drivers read it back with. That crate is a contract, not a
driver — it depends on nothing but serde — so the two ends of the host→driver
ABI are one declaration rather than two copies kept in step by hand.

## Why one crate

The three shipped together, versioned together, and were consumed together by
`engine`. What the split cost was the conformance battery: no one of the three
could own tests spanning all of them, so it needed a fourth crate
that existed only to hold dev-dependencies. Folded, that
is just `tests/`.

The fold gave up one thing: `plan` alone was `no_std` (+ `alloc`). `eval`'s
extern channels are `std::sync::{Arc, Mutex}`, so one crate holding both cannot
be. Nothing depended on it — a guest reaches `tensor-dsl` and `tensor-ir`, and
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
capability-less execution (a driver with no device-geometry ports has the host
fold the prologue per fire), and geometry classification. It reuses the
interpreter's `eval_op` so there is no second evaluator to drift.

## Rust-first

This crate contains **no hand-written C++**. The only non-Rust files are
codegen's inputs and outputs: device runtime templates (`.cuh` / `.metal`) under
`runtime/` that Rust assembles, and the generated C headers checked in under
`include/` with drift tests. Backend code generation is a pure `Plan -> String`
function with no device-architecture input, which is what lets it live outside
the driver and be golden-tested without a GPU. Compilation itself (NVRTC,
`MTLLibrary`), module caching, and launch stay in the drivers — those need a
live device.

## Generated artifacts

`include/` is checked in and verified by `tests/`. Never edit those files by
hand; change the tables in `tensor-ir` and regenerate.

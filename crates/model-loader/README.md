# model-loader

Typed weight-loading compiler: checkpoints to verified device-memory layouts.

```text
checkpoint  ──materialize──▶  model contract  ──compile──▶  load plan  ──execute──▶  arena + sink
(its own        (what to produce,       (the bytes to move,      (`executor`, into a
 metadata)       as expressions over      where, in what           CUDA arena or a
                 the checkpoint)          order)                   `&mut [u8]`)
```

`plan = compile(source_facts, program, target)` — the formula `src/lib.rs`
states. None of the three inputs is a model's name: the caller states what it
needs as a contract over the checkpoint's byte space, the loader reads the
checkpoint's own metadata, and the target carries the numbers a device
measured. `tests/standalone.rs` pins that as four properties rather than as
prose.

The contract is internal IR. It read `author(facts, source, policy)` here, for
a driver-side contract author that R3 deleted with the legacy load contract;
the live path synthesizes one instead — `contract::materialize` reads the
checkpoint and produces the `ModelContract` that `plan::compile` consumes, in
the same call. The compiler stays family-blind either way: none of its inputs
is a model's name.

There is no C ABI. The drivers link this crate as an rlib and pass Rust types;
`crates/driver-cuda/csrc`'s `load_plan_executor.hpp`, `weight_store.cpp` and
the generated `pie_loader.h` were deleted with the rest of the C++ loader, and
executing a plan is `executor`'s — one implementation, one set of decisions,
with the device half supplied as an `ArenaBacking` rather than as a second
executor. See `src/executor/arena.rs`.

## Build and check

```sh
cargo test -p model-loader                    # the whole compiler, no GPU needed
cargo clippy -p model-loader --all-targets
cargo fmt -p model-loader --check
UPDATE_GOLDEN=1 cargo test -p model-loader --test golden_plans
```

The `cuda` feature is off by default and the default is the load-bearing one:
without it the dependency set is `half`/`ztensor`/`serde`/`thiserror`, and the
crate builds on a machine with no toolkit. That is what keeps the host
executor the reference the device answer is checked against. Exactly one target
turns it on, and it is a test — see the feature's note in `Cargo.toml`.

## The one command

This crate declares no `[[bin]]`. The command is `pie`'s, and there is one:
`pie model import`, which compiles a plan and runs it through
`executor::Execution` in its streaming shape, with no GPU, no runtime and no
driver.

An offline `pie model build` stood beside it — weight conversion that ran the
load transforms ahead of time and wrote the result as a second `.zt` — and R3
deleted it with the load contract those transforms authored. Nothing in this
build writes a `<name>/runtime/` artifact, and the vocabulary that described
one is gone from `checkpoint::meta` for the same reason.

What import still does is the decode half of that work: a GGUF Q4_0 checkpoint
casts to `BF16` on the way into the artifact — the one scheme whose scales live
inside the block, which no device kernel reads — so import doubles as a
GGUF-to-safetensors converter. Build with `--release` for real checkpoints.

Every encode the plan language admits still runs on the host, and the golden
contracts exercise them: MXFP4 (`F4_E2M1` payload + `U8` block scales),
per-channel FP8 (`F8_E4M3` + `F32` scales) and per-channel INT8 (`I8` + `F32`
scales), from a BF16/F16/F32 operand or from an FP8 block-scaled checkpoint
whose `_scale_inv` factors are applied first — each the bit-for-bit port of the
corresponding CUDA kernel, encoded row-parallel across cores (AVX2 where the
CPU has it). See `tests/golden/contracts/convert_bf16_to_mxfp4.json` for the
smallest example.

What the plan language refuses, deliberately: `Repack` (a device kernel layout
has no on-disk representation; it stays a load-time step), group contracts
(instances share their template's names, so a checkpoint cannot hold them —
declare the members as plain tensors and load the result with the group
contract), and quantized *output* encodings outside {MXFP4, FP8, INT8}
(safetensors has no tag for a GGUF or AWQ payload).

## Inspecting a plan

`verify::verify_plan` is the second opinion: it takes a plan as it stands and
asks what can be answered from the plan plus the filesystem — is the schedule a
permutation of the instructions, is every public declaration finalized exactly
once, does every read land inside the file it names at the size that file
actually is, does the result deliver the contract it was compiled from. It is
deliberately not a second compiler; the two share no code, which is what makes
disagreement mean something.

`pie model import` runs it on every plan it compiles, before a byte is read.
The golden-plan tests run it too, over all sixteen goldens, so a golden cannot
be regenerated from a broken compiler.

`dump::describe` is the one-line boot-log summary and `dump::plan_stats_json`
is the operator-facing shape (counts plus instruction and transform
histograms); the golden and plan-growth tests are their callers. A
`dump::dump_load_plan_json` stood beside them and had none at all — a plan is
`Serialize`, so whoever needs its full text writes that line where they need it.

A `ModelContract` has a JSON representation, and `tests/golden/contracts/` is
where it is used: fixtures, so a test can state a contract without building one
in Rust. The live path does not read JSON — `contract::materialize` synthesizes
the contract from the checkpoint as a Rust value. The backend, the fusion
choice and the tensor-parallel rank travel in `StorageTarget`, which is the
value `plan::passes::tile` lowers against.

## Design documents

They live in the wiki, under `loader/`:

| Page | What it is |
| --- | --- |
| `architecture.md` | Target state, the boundaries, and §12's numbered log of what has landed |
| `spec.md` | The model contract format a driver authors |
| `loader_v2_plan.md` | The north star the v2 refactor was measured against |
| `contribution.md` | Paper design notes — *don't trust the loader, check the plan* |

A `metal_todos.md` row stood here — "what the refactor changed in the Metal ABI
but could not run". There is no Metal ABI: the drivers pass Rust types, and
`driver-metal` left the workspace entirely in R3. The page is still in the wiki
as a record of what was never verified.

Source comments cite these by filename — "`spec.md` §3.3", "`loader/architecture.md`
§12 row 16". Those paths are the wiki's, not this tree's.

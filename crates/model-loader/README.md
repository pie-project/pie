# model-loader

Typed weight-loading compiler: checkpoints to verified device-memory layouts.

```text
model request  ──author──▶  model contract  ──compile──▶  load plan  ──execute──▶  arena backing
(facts + policy,   (what the model         (the bytes to move,       (`executor`, into
 ~20 scalars)       needs, as expressions   where, in what             CUDA, Metal or a
                    over the checkpoint)    order)                     `&mut [u8]`)
```

`plan = compile(source_facts, author(facts, source, policy), target)`. The
contract is internal IR: it is authored from the driver's request
(`pie_model::contract`, `plan/model-in-rust.md`), compiled, and dropped in one
call. The compiler itself stays family-blind — none of its inputs is a model's
name; the request's `model_type` is spent selecting the author, the loader
reads the checkpoint's own metadata, and the target carries the numbers a
device measured.

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
executor the reference the device answer is checked against.

## Offline conversion

This crate declares no `[[bin]]`. The commands are `pie`'s — `pie model import`
and `pie model build`, both of which compile a plan and run it through
`executor::Execution` in its streaming shape, with no GPU, no runtime and no
driver.
`dump::describe`, `dump::plan_stats_json` and `verify::verify_plan` are the
library surface those inspections are built from; the golden-plan tests are
their other caller.

`pie model build` is offline weight conversion: the contract declares the output
tensors (a `Cast` into a quantized encoding is how "quantize this" is
spelled), the host executor runs the plan, and the result is written as a new
`.zt` checkpoint — every tensor on a page, every payload digested, and the
quantization scheme named by the file rather than guessed from a dtype tag. Loading it afterwards is a cheaper
contract against the converted names. Every encode the plan language admits
runs on the host: MXFP4 (`F4_E2M1` payload + `U8` block scales), per-channel
FP8 (`F8_E4M3` + `F32` scales) and per-channel INT8 (`I8` + `F32` scales),
from a BF16/F16/F32 operand or from an FP8 block-scaled checkpoint whose
`_scale_inv` factors are applied first — each the bit-for-bit port of the
corresponding CUDA kernel, encoded row-parallel across cores (AVX2 where the
CPU has it). The decode direction runs too: a GGUF Q4_0 checkpoint casts to
`BF16` — the one scheme whose scales live inside the block, which no device
kernel reads — so it doubles as a GGUF-to-safetensors converter. Build
with `--release` for real checkpoints. See
`tests/golden/contracts/convert_bf16_to_mxfp4.json` for the smallest example.

What conversion still refuses, deliberately: `Repack` (a device kernel layout
has no on-disk representation; it stays a load-time step), group contracts
(instances share their template's names, so a checkpoint cannot hold them —
declare the members as plain tensors and load the result with the group
contract), and quantized *output* encodings outside {MXFP4, FP8, INT8}
(safetensors has no tag for a GGUF or AWQ payload).

A contract is a JSON `ModelContract` — see `tests/golden/contracts/`. The
backend, the fusion choice and the tensor-parallel rank travel in
`StorageTarget`, which is the value `plan::passes::tile` lowers against.

## Design documents

They live in the wiki, under `loader/`:

| Page | What it is |
| --- | --- |
| `architecture.md` | Target state, the boundaries, and §12's numbered log of what has landed |
| `spec.md` | The model contract format a driver authors |
| `loader_v2_plan.md` | The north star the v2 refactor was measured against |
| `contribution.md` | Paper design notes — *don't trust the loader, check the plan* |
| `metal_todos.md` | What the refactor changed in the Metal ABI but could not run |

Source comments cite these by filename — "`spec.md` §3.3", "`loader/architecture.md`
§12 row 16". Those paths are the wiki's, not this tree's.

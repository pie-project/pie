# pie-loader

Typed weight-loading compiler: checkpoints to verified device-memory layouts.

```text
model request  ──author──▶  model contract  ──compile──▶  load plan  ──C ABI──▶  driver executes
(facts + policy,   (what the model         (the bytes to move,       (pie_loader.h,
 ~20 scalars)       needs, as expressions   where, in what            generated)
                    over the checkpoint)    order)
```

`plan = compile(source_facts, author(facts, source, policy), target)`. The
contract is internal IR: it is authored from the driver's request
(`pie_model::contract`, `plan/model-in-rust.md`), compiled, and dropped in one
call, never crossing the ABI. The compiler itself stays family-blind — none of
its inputs is a model's name; the request's `model_type` is spent selecting the
author, the loader reads the checkpoint's own metadata, and the target carries
the numbers a device measured.

## Build and check

```sh
cargo test -p pie-loader -p pie-loader-capi  # the compiler and its C ABI, no GPU needed
cargo clippy -p pie-loader --all-targets
cargo run -p pie-loader-cbindgen             # regenerate capi/include/pie_loader.h
UPDATE_GOLDEN=1 cargo test -p pie-loader --test golden_plans
```

Two crates: `pie-loader` is the compiler and knows nothing of C;
`pie-loader-capi` (`capi/`) is the repr(C) marshalling, the extern entry
points, the `pie-loader` CLI, and the committed header. Regenerating the
header must leave it byte-identical unless the ABI changed on purpose;
`driver/{cuda,metal}` compile against it.

## The tool

Compile, inspect, check, replay and convert without a GPU, a runtime, or a
driver:

```sh
pie-loader dump    SNAPSHOT CONTRACT          # the plan, as JSON
pie-loader verify  SNAPSHOT CONTRACT          # check the plan against its contract
pie-loader diff    SNAPSHOT CONTRACT GOLDEN   # compare against a stored dump
pie-loader replay  SNAPSHOT CONTRACT          # execute on the CPU, against real bytes
pie-loader convert SNAPSHOT CONTRACT OUT      # execute on the CPU, write a checkpoint
```

`convert` is offline weight conversion: the contract declares the output
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
kernel reads — so `convert` doubles as a GGUF-to-safetensors converter. Build
with `--release` for real checkpoints. See
`tests/golden/contracts/convert_bf16_to_mxfp4.json` for the smallest example.

What `convert` still refuses, deliberately: `Repack` (a device kernel layout
has no on-disk representation; it stays a load-time step), group contracts
(instances share their template's names, so a checkpoint cannot hold them —
declare the members as plain tensors and load the result with the group
contract), and quantized *output* encodings outside {MXFP4, FP8, INT8}
(safetensors has no tag for a GGUF or AWQ payload).

Four optional positional arguments follow, in order: `BACKEND` (`cuda`|`metal`|
`host`), `FUSION` (`fused`|`unfused`), `TP` (`RANK/SIZE`), and a JSON
`StorageTarget`. `CONTRACT` is a JSON `ModelContract` — see
`tests/golden/contracts/`.

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

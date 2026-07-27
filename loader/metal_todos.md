# Metal: what the loader v2 refactor could not verify

The v2 refactor (`loader_v2_plan.md`, §12 row 16 of `architecture.md`) changed
the loader ABI in four ways that reach the Metal driver. CUDA was verified by
building and running its tests on an RTX 4090. **Metal was not: this machine has
no macOS and no Metal framework, so `cargo build --features driver-metal` cannot
run here at all.**

This file records exactly what was checked, how, and what is still open. It is
meant to be deleted by whoever runs the Metal build and closes the list.

---

## ABI change 5 — the instruction became a tagged union

`PieLoaderStorageInstrView` is now `{ id, op }` where `op` is a
`#[repr(C, u32)]` tagged union, and `PieLoaderStorageInstrKind` is gone.
`heap_bind.cpp` switches on `instr.op.tag` and reads through the variant
bodies. Three runtime guards went with it — `ExtentWrite` and
`BulkExtentWrite` can no longer be missing a source or a destination, and
`CreateView` can no longer have a number of inputs other than one — because
the union states what the flat struct could only check for.

Verified by the syntax sweep below, which is a strong check here: the tag
values, the body layouts and the member names all come from one generated
header, so a driver that disagrees with the loader fails to compile rather
than misreading a union member at run time. Unverified: that the rewritten
arms *behave*, which needs M1.

## What the refactor changed under Metal

| # | Change | Metal surface |
| --- | --- | --- |
| 1 | `StorageInstr::Release` deleted (discriminant 4 left as a gap) | `heap_bind.cpp`'s `case K::Release:` — **would not compile** |
| 2 | `StorageInstr::Fill` added (discriminant 8) | `heap_bind.cpp`'s switch had no arm and no `default:` |
| 3 | `QuantSpec::{scale_dtype, zero_point_dtype, block_shape}` deleted, and `ContractBuilder::with_block_shape` with them | `qwen3_5_contract.hpp::push_mlx_affine_u4` authored all three |
| 4 | `TileMapKind::Reorder` deleted (a second name for `Reblock`) | not referenced by Metal |

Changes 1–3 were fixed in the same commits. Change 1 was a **build break** that
the CUDA-only verification would never have caught, and it is the reason this
file exists rather than a note in a commit message.

---

## What *was* verified, and how

Every Metal translation unit that mentions `pie_loader` was syntax-checked on
Linux, against the real headers, with the host compiler:

```sh
INCS="-Iloader/include -Idriver/metal/src -Idriver/metal/src/batch \
      -Idriver/metal/src/model/qwen3_5 -Idriver/metal/src/loader \
      -Idriver/common/src -Idriver/abi/include"

for f in $(grep -rl pie_loader driver/metal/src --include=*.cpp --include=*.hpp); do
    case "$f" in
      *.hpp) printf '#include "%s"\nint main(){}\n' "$(basename $f)" > /tmp/tu.cpp
             src=/tmp/tu.cpp ;;
      *)     src="$f" ;;
    esac
    g++ -std=c++20 -fsyntax-only $INCS "$src" || echo "FAIL $f"
done
```

Result: **5 pass, 0 fail** — `batch/forward.cpp`, `loader/heap_bind.cpp`,
`loader/heap_bind_metal.hpp`, `loader/load_plan.hpp`,
`model/qwen3_5/qwen3_5_contract.hpp`. No `.mm` file references `pie_loader`, so
nothing loader-facing needs an Objective-C compiler.

Also checked mechanically: every `PieLoader*`, `PIE_LOADER_*` and
`pie_loader::*` identifier appearing anywhere under `driver/metal/` resolves in
the generated headers, and `heap_bind.cpp`'s switch covers exactly the eight
`PieLoaderStorageInstrKind` values the header declares — no more, no fewer.

**This is a strong check on the ABI surface and no check at all on behaviour.**
It proves the names line up. It proves nothing about what the code does.

---

## Open items

### M1 — Build and link the Metal driver

The only item that needs macOS. Everything else is downstream of it.

```sh
cargo build -p pie-worker --features driver-metal
```

The syntax sweep above used g++; Metal builds with Apple clang, which differs
on template diagnostics and on a few `-W` defaults. Expect the delta to be
warnings rather than errors, since the sweep already covers name resolution.

### M2 — Exercise `Fill` on Metal

`heap_bind.cpp` implements it as a host-side `memset` over the slot:

```cpp
case K::Fill: {
    const auto target = buffers.find(instr.buffer_id);
    ...
    std::memset(target->second.contents(), 0, target->second.size);
}
```

Two assumptions behind that, both untested:

* **The weights region is host-visible.** `SlotHandle::contents()` returns a CPU
  pointer and is documented as "Shared storage". `heap_bind.cpp` already relies
  on this for every `copy_storage_bytes` call, so `Fill` is not introducing the
  assumption — but if the region ever becomes Private, `Fill` and the copies
  break together, and `Fill` is the one that fails silently.
* **Program order is enough.** CUDA's `fill()` must call `copy_engine_.flush()`
  first because its copies are stream-async. Metal's are synchronous host
  stores, so the ordering `validate-fill-order` guarantees on the Rust side
  carries over for free. That reasoning is sound and unexecuted.

Note that **no shipping contract pads today**, so `Fill` is unreachable in
practice on either backend. It is exercised in Rust
(`host_executor.rs::a_padded_tensor_comes_back_with_zeros_where_no_source_reaches`,
`tests/algebra.rs`, `tests/storage_compiler.rs`) and nowhere in C++. The first
contract that uses `Expr::Pad` will be the first real test of both drivers.

### M3 — Give Metal a contract test

CUDA has `driver/cuda/tests/model_contract_test.cpp`, which builds a contract
with `contract.hpp` and pushes it through the real loader. It is what caught the
`QuantSpec` field removal immediately. **Metal has no equivalent**, so
`qwen3_5_contract.hpp`'s `push_mlx_affine_u4` change (item 3 above) is verified
by syntax only — nothing has checked that the MLX affine-u4 path still produces
the same contract, or any contract at all.

This is the highest-value item after M1, and unlike M1 it does not need a GPU:
`model_contract_test.cpp` links `pie-loader`'s staticlib and never touches CUDA.
A Metal analogue would run on Linux, which would mean the Metal *contract* layer
stops depending on someone owning a Mac.

### M4 — Consider making the syntax sweep a real check

The loop above is a one-off in this file. It caught a build break that the
whole CUDA suite could not, it takes seconds, and it needs no Apple hardware.
Somewhere in CI — or as a `#[test]` next to `tests/standalone.rs`, which already
treats "the architecture is a test" as the house style — it would keep the Metal
ABI surface honest between Mac builds.

---

## Where the risk actually sits

The loader's ABI is POD structs and enums shared through one generated header,
so a mismatch is a compile error rather than a runtime surprise — which is why
the syntax sweep is worth as much as it is. What it cannot see is the arm that
compiles and does the wrong thing. On this change there is exactly one such arm,
`case K::Fill`, and it is unreachable until a contract pads.

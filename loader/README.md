# pie-loader

Typed weight-loading compiler: checkpoints to verified device-memory layouts.

```text
model contract  ──compile──▶  load plan  ──C ABI──▶  driver executes
(what a driver     (the bytes to move,     (pie_loader.h,
 declares it        where, in what          generated)
 needs)             order)
```

`plan = compile(source_facts, contract, target)`. None of the three inputs is a
model's name — the driver declares what it needs as a contract over the
checkpoint's byte space, the loader reads the checkpoint's own metadata, and the
target carries the numbers a device measured.

## Build and check

```sh
cargo test -p pie-loader                     # 236 tests, no GPU needed
cargo clippy -p pie-loader --all-targets
cargo run -p pie-loader-cbindgen             # regenerate include/pie_loader.h
UPDATE_GOLDEN=1 cargo test -p pie-loader --test golden_plans
```

The generated header is committed. Regenerating it must leave it byte-identical
unless the ABI changed on purpose; `driver/{cuda,metal}` compile against it.

## The tool

Compile, inspect, check and replay a plan without a GPU, a runtime, or a driver:

```sh
pie-loader dump   SNAPSHOT CONTRACT          # the plan, as JSON
pie-loader verify SNAPSHOT CONTRACT          # check the plan against its contract
pie-loader diff   SNAPSHOT CONTRACT GOLDEN   # compare against a stored dump
pie-loader replay SNAPSHOT CONTRACT          # execute on the CPU, against real bytes
```

Four optional positional arguments follow, in order: `BACKEND` (`cuda`|`metal`|
`host`), `FUSION` (`fused`|`unfused`), `TP` (`RANK/SIZE`), and a JSON
`StorageTarget`. `CONTRACT` is a JSON `ModelContract` — see
`tests/golden/contracts/`. `PIE_LOAD_PLANNER_DEBUG=1` prints per-pass timings.

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

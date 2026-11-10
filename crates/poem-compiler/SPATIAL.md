# Temporal and spatial plans

The compiler reads a trace once and plans it one of two ways.

```
                 ┌─ temporal::plan ─▶ CompiledModel   (launches over time: regions,
trace ─ analyze ─┤                                     lowerings, streams, arena)
   (Analysis)    └─ spatial::plan  ─▶ SpatialPlan<P>  (the model resident over cores:
                                                       blocks, flows, a partition)
```

- **`analysis::analyze`**: the target-neutral facts both plans start from. These are the request classes cut by the guards, the values an expert region holds, the trace's collectives, the row axes it states, and each node's layer. `compile` / `compile_axes` run it and then `temporal::plan`. Their behaviour is unchanged; `examples/plan_snapshot.rs` fingerprints every catalog row's plans so a refactor can show that.
- **`shard::distribution`**: the tensor-parallel pass's inference without its rewrite. For each value, it records how the ranks hold it (whole, split in blocks, partial sum), and it lists the collectives that a split or partial value needs, after the node that makes the value. `shard::shard` is that inference followed by the rewrite into one trace per rank. A spatial target that places split values itself reads the distribution instead.
- **`spatial`**: the spatial side.
  - `SpatialGraph::of` groups consecutive nodes of one layer and one target-named kind into blocks, and records the values that flow between blocks.
  - `spatial::plan` asks the target for candidate partitions, drops those that do not fit, and keeps the cheapest.
  - Everything physical is the target's, after the plan: core coordinates, routes, colours or NoC channels, kernels, and memory placement.

## What a target implements

`SpatialTarget`:

| Item | What the target provides |
|---|---|
| `classify(trace, node) -> Option<BlockKind>` | The kind of block each node belongs to. Kinds are names the target gives, such as `attention`, `mlp`, `moe`, `head` or `embed`. `None` refuses the node. |
| `type Partition` | The target's own description of a cut: cores per block, how a block's dims split over them, which layers share a region, and the sizes of reduction groups. |
| `candidates(trace, graph)` | The partitions worth trying, in the target's order. A target searches here: the full space, a table of known-good shapes, or both. |
| `fit(graph, partition)` | Whether the partition fits the device: memory per core, route or channel budgets, core count. If not, it says why. |
| `cost(graph, partition)` | The predicted time of a step. |

The physical build then lowers `SpatialPlan<Partition>` to the device. That build is the target's code, not the compiler's.

## How targets map onto it

**A wafer of PEs with a routed fabric.**
- The blocks are its program's layers of units:
  - weight PEs
  - control rows and columns that reduce
  - a head that holds the vocabulary rows
- The partition is the unit grid, the rows and columns a unit's dims split over, segment widths, two-level reduction groups, and the head's cut. These are the `Partition` fields of a wafer program plan.
- The flows become handoffs between units.
- The target's own build lays them onto the fabric: floor, wiring, colours and kernels.
- Its existing `Plan` is today read off a prep's metadata. It becomes `SpatialPlan<Partition>`, with `Shape` read from the trace's parameters.

**A mesh of cores with a network-on-chip and device DRAM.**
- Each core runs a small program over tiles, streamed from DRAM or held in core SRAM.
- The partition is the core grid per block and the tile shape per core. It also says whether a block's weights stay resident in SRAM or stream from DRAM, and how layers pipeline across core groups.
- The flows become NoC transfers between core groups.
- `shard::distribution` tells it which values are partial sums to reduce over the NoC, where a block is split across groups as tensor parallelism splits it across ranks.
- `fit` checks SRAM per core and NoC bandwidth.
- `cost` weighs compute per core against transfer and DRAM time.

Both targets share the compiler's half unchanged: the analysis, the grouping into blocks, the flows, the distribution of split values and the search loop. Only the block kinds, the partition type and the physical build differ.

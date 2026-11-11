# The wafer program: architecture

A wafer program is a list of **blocks** over a logical **plan**. poem's spatial pass reads the
model's **shape** off its trace; the family the shape needs turns it and a **preset** into the plan;
the composer picks the blocks from what the plan's layer program needs; the core runs every build step over the blocks in order and knows
no model, no family and no block's vocabulary. The build lives here (`kernels-cerebras`); the
engine (`engine-cerebras`) runs its **artifact** and knows none of it (below).

```
.poem trace ─target::shape─▶ Shape ─family::serving─▶ Family
 (or a shape file)               │                       │ presets (a named one, or the one made for the shape first)
                                 └──────────┬────────────┘
                                            ▼ Family::plan            (poem_compiler::spatial::plan:
                                          Plan ─compose::blocks─▶ [Block]  the first of them that builds)
                                            │                        │ floor   ─▶ Floor  (units, head, stores; a PeRole a PE)
                                            │                        │ wire    ─▶ Wiring (typed links over open Classes / Slots)
                                            │                        │ settle  ─▶ Floor  (PEs placed where links cross)
                                            │                        │ colours ─▶ Palette
                                            │                        ▼
                                            │                     Fabric::lay + validate (routes, queues, invariants)
                                            │
                                            └─Family::kernels─▶ Kernels (Kind → KernelBuild: descriptor + typed params)
                                                                    │ blocks' overrides ─▶ Overrides (per-PE params)
                                                                    ▼
                                    emit::{rcfg (Kernel::rcfg over a Pe), params + layout, host}
                                    Program::bindings (blocks' copies) ─▶ artifact::Artifact ─▶ engine
```

Modules are under `src/`.

| Layer | Module | Holds | Knows |
|---|---|---|---|
| Plan | `plan.rs` | `Shape` (the model's facts), `Partition` (counts, split factors), `Contract` (positions, CAP, LOOP, WL, the logits' prescale), the family; `Preset` (a family's named choices with overrides); serde, so a plan is a file | nothing physical: the seam to poem's spatial pass (below) |
| Target | `target.rs` | the wafer as poem's `SpatialTarget`: `shape` (a trace's `Shape`), `Wafer` (candidates: the serving family's presets; fit: the plan builds; cost: table order), `plan` / `plan_shape` | the trace's ops, the families |
| Roles | `role.rs` | `PeRole { kind: Kind, place: Place }`. `Kind` is open (name + build stem; the architecture's kinds are consts, a block declares more). `Place` is the wafer program architecture's positions: weight / CX / CY of a unit, head / head control, a storage column's PE, none | coordinates |
| Floor | `floor.rs` | `FloorBuilder` (blocks place onto it in order), units, gap / relay slots, the head block, store blocks, a `PeRole` a PE in a dense `Grid` | coordinates |
| Wiring | `wiring.rs` | `Chain`, `Multicast`, `Path`, `Turn`, `Note`; open `Class { name, index }`, `Slot { name, kind: SlotKind }`, `NoteKey`; the core's only own vocabulary is `LOOPBACK` and `PASS` | coordinates, logical colours |
| Fabric | `fabric.rs` | `Palette` (the blocks' colours, later replacing earlier), routes, `PeIo` (slot → endpoint, notes), parks / spare / route words, the invariants | physical colours, route words |
| Blocks | `block/` | the `Block` trait and the blocks (table below); each owns its classes, slots, notes, kinds and colours as consts | coordinates, its own vocabulary |
| ABI | `abi.rs` | `Kernel` (source, owned symbols, params' names, route slots, rcfg writer), `params!` (a kernel's typed params struct), `words!` (an rcfg word layout), `KernelBuild`, `Kernels`, `Overrides` | the build's view of a kernel |
| Kernels | `kernels/` | `Pe` (one PE as an rcfg writer sees it); `common` (the word layouts every family's weight / CX / CY / head / head control kernel shares, its codes named, the relay); `head` (`head_pe.csl`, `head_ctl.csl`); one module a family's own kernels: `dense`, `gptoss`, `gemma`, `gemma12` | the CSL kernels' ABI |
| Layers | `layers.rs` | the plan's layer program (`poem_compiler::spatial::layer::LayerProgram`), stated from its shape; pinned to the trace's lowering | the shape |
| Composer | `compose/` | `Needs` and `Schedule` (read off the layer program), `blocks` (the blocks a program's needs take), `KernelCopies` (what a family's kernels read off each block), `stacked` (the stacked unit columns' arrangements) | the blocks |
| Families | `family/` | `serving` (the family a shape needs); `dense`, `gptoss`, `gemma`, `gemma12`: presets, the plan of a shape under a preset, the kernels and their copies, the cut's statement | the model's shape |
| Emit | `emit/` | rcfg, kernel params (levers, overrides) and `layout.csl`, `HostLayout` (everything the host driver reads and writes on the wafer) | kernel builds |
| Weights | `weights.rs` | `Binding`s (symbol, rect, source, transforms), `Copies`, the placer | the device |
| Program | `program.rs` | `Program { plan, blocks, floor, wiring, palette, fabric, kernels, overrides, family }`, `build`, `bindings` | all of the above |
| Artifact | `artifact.rs` | `Artifact`: the program as the engine runs it (`HostLayout`, bindings, rcfg words, the model facts of the host's math, the plan as a record) | files |

Rules:

- **The block is the unit of extension.** A model's blocks are what its layers need; a new dataflow part
  (a new segment kind, a storage column, a band head) is a new block. Nothing in the core names a
  family or a block's classes, slots, notes or kinds.
- **A block owns its vocabulary.** Its `Class`es, `Slot`s, `NoteKey`s (and their value codes),
  extra `Kind`s and colours are consts in its module (`units::ROW_L1`, `head::HEAD_IN`,
  `head::ROLE_SOURCE`, `token_loop::TURN_TOP`, ...). Kernels' rcfg writers read the slots and
  notes by those consts. Two blocks never declare one name: a slot is `(name, kind)` and a class
  `(name, index)`, so a clash would merge queues.
- **A kernel is described once.** One `Kernel` const a CSL source; its params are one `params!`
  struct (typed fields in `@set_tile_code` order, `NAMES`, a `Param` const a name); its rcfg is one
  writer that fills `words!` structs (named words in rcfg order). `KernelBuild::of` refuses params
  that are not the kernel's in its order. Codes the kernels read are named consts
  (`common::SEG_NONE`, `CX_EMB`, `LOOP_ROOT`, `fabric::member_role`, `gdn::SOURCE_SEND`, ...).
- **Per-PE param differences are typed `Overrides`** a block states (`Block::overrides`), checked
  against the PE's kernel at resolution. There are no stringly per-PE notes.
- **Physical facts are derived once.** Blocks read the floor (`Unit::row_y`, `HeadBlock::col_x`,
  ...) and the notes other blocks left; no family restates the plan as its own geometry struct.
- **Wiring is order-free.** The fabric sorts routes and binds endpoints order-free, so the block
  order only matters for the floor (later blocks place against earlier ones), the palette (later
  replaces) and the copies (placement order is the host's device protocol).
- **Everything up to `Program` is pure:** no device, no files. `wafer-compile` and the tests rely on
  this. Rust is the only geometry; the Python tools call `wafer-compile`.

## Invariants (`Plan::validate`, `Fabric::lay` / `validate`, `check_route_slots`)

The hardware lessons of the bring-up, made checks:

1. **Route clash.** Two words for one colour at one PE is an error.
2. **Parked queues.** An unused input queue never parks on a colour the PE sends on. Parks come
   first from colours with no route, then from pass-through colours (`Fabric::parks_avoiding`). The 0704
   hang.
3. **Segment queues.** A `SlotKind::Segment` slot's colours are never a `SlotKind::Reduce` slot's
   on the same PE. 0733: a segment borrowed iq3.
4. **Control lines.** A segment never spans a control line (`wiring::segments` refuses it; the
   plan's segment widths divide the half-rows they live in).
5. **Path continuity.** Every `Path` runs from its sender's ramp to its receiver's ramp.
6. **Route slots.** Routes at a PE ≤ its kind's kernel's `route_slots`; a non-idle PE whose kind
   has no kernel is an error.
7. **Colour count.** Every colour < `COLOURS` (21).
8. **Loop-backs.** Every root of a `SlotKind::Reduce { loopback: true }` slot loops back to its own
   ramp on `LOOPBACK`'s colour.
9. **Kernel-side rules** (not geometry; listed so nobody drops them): restore a route only after
   its oq drains (0738); ≥768 B stack room a PE (`wafer-compile link`, `Cslc::stack_room`); payload directories
   end in `_ieee`.

## The blocks

| Block | Floor | Wiring | Vocabulary | Copies |
|---|---|---|---|---|
| `units::Units { grid, copies }` | units per `UnitGrid` (snake with gaps, or explicit origins); gap slots `Kind::RELAY` | `unit_chains`: row chains into CX, column chains into CY (two-level by `reduce.row` / `.col`), root loop-backs, `handoff`s along the visit order | `ROW_L1/L2/CAST`, `COL_L1/L2/CAST`, `HANDOFF_OUT/VERT[parity]/IN/ACROSS`; slots `ROW_REDUCE`, `COL_REDUCE`, `X_IN`, `X_OUT`, `RELAY_PRE`, `RELAY_POST` | `UnitCopies` per unit (quadrants of `w{u}`, CX `gam`, CY tables), relays' rcfg |
| `segments::Scores { width }` | — | every weight row in segments of `width` | `SCORE_L1`, `SCORE_CAST`; slot `SEGMENT` | — |
| `segments::Onorm { width }` | — | every CY row in segments (no single-PE ones) | `ONORM_L1`, `ONORM_CAST`; slot `ONORM` | — |
| `gdn::Gdn` (`Gdn::of(plan)`) | — | the (k, q) multicast per GDN segment and key row; `Gdn::source` is the weight kernel's source word | `GDN_KQ`; slot `GDN`; `SOURCE_*` | — |
| `head::HeadEast { cols, rows, token_loop, copies }` | the head east of the floor so far, control column in the middle | `head_rows` (into the control column), `head_input` (two CX rows a head column: relay north, column multicast) | `HEAD_RELAY`, `HEAD_COL_CAST`, `HEAD_ROW_WORD`; slots `HEAD_ROW_REDUCE`, `HEAD_IN`, `HEAD_RELAY_IN`, `HEAD_CAST`; note `ROLE` (`ROLE_RELAY`, `ROLE_SOURCE`) | `HeadCopies` (`hw`, `hg`, `hemb`, rcfg) |
| `token_loop::TokenLoop` | — | the control column's max-reduce, the embedding's return into unit 0's CX | `LOOP_L1/L2/CAST`, `EMB_UP/DOWN/SPLIT/WEST`; slots `LOOP_REDUCE`, `EMB_IN`; note `TURN` (`TURN_TOP/EVEN/ODD`) | — |
| `ple::Ple { columns }` | a storage block above each unit column's top unit (`Place::Store`, kind `STORE`; its CX column route-only); `settle`: every idle PE west of the head outside the units and storage that a link crosses runs the store kernel route-only | `TOKEN` west along each storage row from the head control column; `bus(i)` down each weight column of column `i` into every unit's CY | `TOKEN`, `bus(i)`; `NTW` (route-only PEs outside the storage: 1) | storage tiles `pt<col>_<n>` (staged files under 1,900 MiB), the route-only PEs' rcfg |
| `shared_kv::SharedKv` | — | `LINE` north from the source unit's CY row through the sharers above it | `LINE`; `SH`, `Ic` on the sharers' weight / CY PEs | — |
| `moe::MoeUnits { grid, copies }` | units as `Units`, weight / CY kinds refined by half (`WA`, `WM`, `CYA`, `CYM`: attention columns west of CX, MoE east) | `unit_chains` | the kinds | `MoeCopies` per unit and half |
| `moe::AttnScores`, `moe::Router` | — | scores segments on the attention columns; router segments on the CY row's MoE columns | `ROUTER` (a segment slot) | — |
| `band::Band` | the head as a band below the units with a control row | the x down the MoE columns (`BAND_VERT`), the band columns' chains into the control row | `BAND_VERT`; slots `BAND_IN`, `BAND_OUT`, `BAND_CAST` | the band's head tiles |
| `band::BandLoop` | — | the control row's max-reduce, the bus, the embedding's return (`BAND_V1`, `BAND_W2`); recolours `token_loop::EMB_WEST` to `BUS_COLOUR` | `BAND_V1`, `BAND_W2`; slots `BUS`, `BUS_TX`, `V1_TX`, `V1_RX`, `W2_TX` | — |
| `row_runs::RowRuns { grid, copies }` | units as `Units` | `unit_chains`; scores segments on the attention runs' rows only | `HC`, `ATT` (`ATT_ROW`, `PLAIN_ROW`) overrides: a weight / CX PE's run's dims, a weight PE's run's role | `RunCopies` per unit and run block (`w{u}a` / `w{u}w`, `gama` / `gamw`, the CY tables) |
| `head_groups::HeadGroups { cols, rows, copies }` | the head east of the units (token loop on) | `head_rows`; per group the x chain north on the last unit (`HEAD_GROUP`), its root east, the column multicast; the embedding back up, down, west into unit 0 and down the chain; the max-reduce | `HEAD_GROUP`; slots `CHAIN_IN`, `CHAIN_OUT`, `CAST`; notes `FORWARD`, `head::ROLE`, `token_loop::TURN` | `HeadCopies` |

The role kernels' layer phases come from the plan's layer program (`compose::Schedule`: `NRM` from a deferred norm or the expert halves, `MIXK` from the mixers, `ACT` from the MLP's gate activation, `PLE` from a side input, `KVS` from a shared cache, `HG` from the row-run layout: `compose::Layout`, the one choice the blocks follow too), on `Program::schedule`; every family's kernels read them there. Row roles (`AROW`) are the row runs' per-PE overrides; k = v layers (`KEV`, a checkpoint with no global `v_proj`: not a trace fact) and the GELU form (`GELUT`, a lever) are still their family's.

Compositions: the composer (`compose/`) picks the blocks from what the layers need (`compose::Needs`, read off the plan's layer program, `layers.rs`):

- routed experts: `MoeUnits` (attention | MoE halves, a snake of two unit columns starting upward), `AttnScores`, `Router`, `Band`, `BandLoop` under the token loop;
- a side input or a cache read across layers: `Units` at the explicit origins of the stacked arrangement its unit shape names (`compose::stacked`), `Scores`, `Ple` for a side input, `SharedKv` for a shared cache, `HeadEast`, `TokenLoop`;
- sliding windows (else): `RowRuns` (attention / plain row runs, a U of two unit columns), `HeadGroups`;
- otherwise: `Units` (snake, at most one gap slot), `Scores` when any layer attends over its own cache, `Gdn` + `Onorm` when any is GDN, `HeadEast`, `TokenLoop` under the token loop.

What each block's PEs hold is the family kernels' (`Family::copies`). The families' kernels:

| Family | Composed as | Kernels |
|---|---|---|
| Dense (Qwen3.5, Qwen3, Llama 3) | the snake | `kernels::dense` (`weight` (NRM 0, MIXK 0), `cx` (NRM 0), `cy` (MIXK 0, NRM 0)) + `head` + `common::RELAY` |
| gpt-oss (and any MoE model: Qwen3 MoE) | routed experts | `kernels::gptoss` (`wa`, `wm`: `weight` (MIXK 2, 3), `cx`, `cya`, `cym`, the band head; `head_ctl.csl` with the bus reserved) |
| Gemma 4 E2B / E4B | a side input and shared caches | `kernels::gemma` (`weight` (NRM 1, MIXK 1, PLE 1), `cx` (NRM 1, PLE 1), `cy` (NRM 1, MIXK 1, PLE 1, KVS 1), `head_ctl` (token bus on), `ple_gemma`) + `head::HEAD_PE` |
| Gemma 4 12B | sliding windows | `kernels::gemma12` (`weight` (NRM 1, MIXK 1, row roles by `AROW`), `cx` (NRM 1, HG 1), `cy` (NRM 1, MIXK 1, KEV 1), `head_pe` with head groups) + `head::HEAD_CTL` |

## Kernels and their ABI

```rust
params! {
    /// `head_pe.csl`'s params ...
    pub struct HeadPe { VR: u32, HDc: u32, NPX: u32, HACC16: u32 }
}

pub const HEAD_PE: Kernel = Kernel {
    source: "head_pe.csl",
    owns: &["hw", "hg", "rcfg", "hemb", "tdbg"],
    params: HeadPe::NAMES,
    route_slots: 12,
    rcfg: Some(head_pe),           // fn(&Pe) -> Result<Vec<u32>, String>
};

// in the family's kernels():
out.insert(Kind::HEAD, KernelBuild::of(&HEAD_PE, HeadPe { VR: vr, HDc: 2 * hc, NPX: 1, HACC16: 0 })?);
```

An rcfg writer reads one PE through `Pe`: `pe.member(slot)` / `root` / `halves` / `colour` /
`turn` (the queues the links bound), `pe.note(key)`, `pe.c(class)` (the palette), `pe.parks(n)`,
`pe.spare(finals, comptime)`, `pe.routes(slots)`, `pe.role()`. The shared leading words are
`common::{weight, cx, cy, head, hctl}` over `words!` structs (`WeightWords`, `CxWords`, `CyWords`,
`HeadWords`, `HctlWords`); a kernel passes what is its own (`weight`'s two words and turn width,
`CxIo`'s flags / finals / tail, `cy`'s segment, `HeadIo`) and appends its tail.

Params resolve per PE in `emit::params::tile`: the kind's `KernelBuild::params`, then the PE's
`Overrides` (each must be a param the kind takes). Levers (the preset's, carried by the plan, and `wafer-compile build --levers`'s; the last
argument) override a param in every kind that takes it; `<kind>.<NAME>` (`w.FIFOW:16`, E4B's
weight FIFO) in that kind only; a lever no kernel takes is an error.

`layout.csl` is written from the floor: each maximal rectangle of PEs with one kind and one
resolved param list is one loop nest, every param a literal, so the link passes no params.
The per-kind sources are the kernels', with one-word dummies for the family's other exported
symbols (`Family::symbols`, export order).

### Op modules (`programs/wafer/ops`)

The weight PEs (`weight.csl`) and the row control PEs (`cx.csl`) are one kernel each for every
layer program: the layer's phases (the input norm `NRM`, the mixer kind `MIXK`, the PLE round, head
groups) are comptime params, and the mixers are op modules. The other roles still run their family's
phase order. A kernel's ops come from op modules it imports with comptime params
(`@import_module("ops/<name>.csl", .{ .. })`). A module's code is emitted only where
a kernel calls it; the per-step ones are `inline fn`, so a kernel compiles to the code it had.
`sources::OPS` embeds them, `ProgramKey` digests them with the kernels and `layout::write` writes
them all to `layout/ops`.

| Module | Ops | Imported by |
|---|---|---|
| `fabric` | queue DSDs, queue colour words, the host's routes, input queues bound without a transient duplicate | every kernel |
| `mv` | the tile matvec, f16 or f32 accumulation | weight and head PEs |
| `red` | one two-level chain reduction: a member's send (start, add-forward, group end through its FIFO) or the root's merge and loopback | weight PEs (row, column), head PEs (row), CX, CY, head control roots |
| `seg` | the weight PEs' segment queue modes and the scores all-reduce over a kv head's columns (GDN's (k, q) exchange mode) | the attention weight kernels |
| `math` | rsqrt, exp, recip, ln, sigmoid, silu, softplus, tanh-GELU; exp over a vector | CY roots, attention weight PEs |
| `rope` | a position's cos / sin from the per-level tables; a vector's pairs rotated | CY roots |
| `argmax` | a head row's largest logit and first index (block maxima, halving tree or scan) | head control |
| `wpe` | a weight PE's core: tiles, buffers, queues, the row / column reductions, segments, matvecs, the phases' I/O, handoff relays | `weight` |
| `att` | attention over a layer's own cache (keys dim-major, the ones column, the log-sum-exp reference, redo passes) | `weight` (MIXK 0) |
| `gdn` | the Gated DeltaNet state slice: the (k, q) exchange, the column partials, the update | `weight` (MIXK 0) |
| `kv` | attention over sliding rings and global caches, shared-KV units (the KVS line), plain rows | `weight` (MIXK 1) |
| `swa` | full and sliding-window attention with keys entry-major (sinks by CY; QKN: the q / k head norms' scalars on the scores) | `weight` (MIXK 2) |
| `moe` | routed experts: the router's partial, E5M2 tiles decoded (XF 1: f16 tiles), the TK experts' gate / up and downs | `weight` (MIXK 3) |
| `cpe` | a column control PE's core: the reduction's sum and the multicast's words, queues, the reduction root, the control row's segment all-reduce, the deferred norm's combine, stamps | `cy` |
| `act` | vector exp, SiLU (lean and f16 forms), GELU-tanh (trimmed forms) | `cy` (the MLP, GDN) |
| `cy_att` | the own-cache attention's column work: rms, q / k norm weights, partial RoPE, o / l with the gate, redo and log-sum-exp steps | `cy` (MIXK 0) |
| `cy_gdn` | GDN's column work: conv1d ring, SiLU, gates, decay, delta, the gated norm | `cy` (MIXK 0) |
| `cy_kv` | sliding / global attention's column work after the deferred norm (k = v layers, shared-KV q-only units, the KVS line) | `cy` (MIXK 1) |
| `cy_ple` | the PLE bus and the per-layer embedding gate | `cy` (PLE 1) |
| `cy_swa` | entry-major attention's column work: biases (QB), q / k head norms (QKN), vector RoPE, the sinks' reciprocal (SNK) | `cy` (MIXK 2) |
| `cy_moe` | the router (segment all-reduce, top-k, softmax) and the experts' GLU: clamped SwiGLU or SiLU's (GLU), biases (RB, GB), MXFP4's scales (PSC) | `cy` (MIXK 3) |

The three role kernels of a unit (`weight`, `cx`, `cy`) are each one image over every layer program: the
phase order is theirs, the mixer and norm kinds comptime params (`NRM`, `MIXK`, `ACT`, `PLE`). What is still
a family's own is in the head and PLE kernels: the token loop's winner merge and the handoff relays.

## Adding a model

1. **Shape.** A model poem traces needs nothing here: `target::shape` reads it. A trace op that
   runs on the wafer and that no kernel runs (a new mixer, a new activation) is refused by name
   (`LAYER_OPS`, `OUTER_OPS`); teach `target.rs` the op and give the plan's `Shape` the field the
   blocks need.
2. **Family.** `family::serving` picks the family from what the shape needs. A model of an existing
   family is a new preset in the family's table (`family/dense.rs::PRESETS`, Gemma's presets and
   arrangements, gpt-oss's and 12B's): the counts and split factors the plan cannot derive from the
   shape. A new family implements `Family` (`presets`, `plan`, `tiling`, `blocks`, `kernels`,
   `symbols`) and a rule in `serving`.
3. **List the blocks.** Units first, then what places against them, then wiring-only blocks.
   Reuse a block where its doc fits exactly; otherwise write a new module under `block/` (a new
   block, not a flag on a shared one).
4. **Declare the vocabulary** in the block's module: `pub const BAND_VERT: Class =
   Class::new("HV2");`, `pub const BUS: Slot = Slot::io("band_bus");` (names no other block uses),
   `pub const WA: Kind = Kind::new("wa", "wa");`, notes as `NoteKey` with named value consts;
   colours in the block's `colours()`. A block that recolours another's class declares it again;
   the later block wins.
5. **Floor and wiring** go in `Block::floor` / `wire` (and `settle` for PEs placed where links
   cross). Place against the builder; use the standard wiring (`units::unit_chains`,
   `units::handoff`, `head::head_rows`, `head::head_input`, `token_loop::token_loop`,
   `wiring::segments`) where it fits; leave notes for facts a kernel needs (`head::ROLE`,
   `token_loop::TURN`) instead of recomputing them in an rcfg writer.
6. **Kernels** in `kernels/<family>.rs`: one `params!` struct and one `Kernel` const a CSL source,
   the writer over `Pe` using `common::*` for the shared words. Values that differ by PE are
   `Block::overrides`. The CSL imports its ops from `programs/wafer/ops`; an op no module has is a
   new module there (and a row in `sources::OPS`), not code in the kernel.
7. **Copies** in each block's `bindings`, in placement order (the host's device protocol); the
   family's `tiles` states the arrays they copy (`family/<family>/cut.rs`).
8. **Host.** Anything the driver must find goes in `HostLayout`, derived from the program.
9. **Verify** with the baseline harness (below) and the tiny simulator run.

## The plan seam (poem)

`Program::build(family, plan)` is the whole physical compiler; its only input is a `Plan`. The plan
comes from poem's spatial pass (`poem_compiler::spatial::plan` over `target::Wafer`): the trace
gives the `Shape` (layer kinds, attention / MLP / MoE / GDN / PLE specs, head, RoPE, norms), the
shape picks the family, and the family's presets are the partitions tried, in table order, the
first that builds kept (a named preset tries only itself; `--set` overrides its keys). A checkpoint
with no catalog deployment states its shape in a file (`target::plan_shape`; the tiny test
checkpoints' are `tools/wafer/baseline/shapes/`).

When no preset fits (and none is named), the search plans the model (`search`, below); the
presets stay as the shipping overrides until the search reproduces each one on the chip.
`poem-compiler/SPATIAL.md` has the target-neutral half.

## Cost and search (`cost`, `search`)

`cost::Features` reads what a decode step does off a plan: the weight elements the busiest PE of
each phase multiplies, the reductions' hops within and between groups, the fabric distance the
collectives cross, the KV elements a row reads, the head PE's elements. `cost::time` prices them:
weight and KV elements at the PE's f16 FMAC rate (4 a cycle at 875 MHz), the rest at rates fitted
to the chip (Qwen3.5 9B / 4B / 0.8B, the Qwen3-8B G2 and PY sweeps of requests 0954 / 0961: within
2% of each), plus each layer's own cost, read off the measured step of the model's shipping plan
(`time::MEASURED`; a model the chip has not run takes the median).

`cost::memory` is the PE image model. The bytes a partition moves are the weight PE's (its K
layers' tiles, its KV cache, its scores' buffers) and the head PE's (its vocabulary rows' tiles and
embeddings); code and the small buffers move with the kernels' flags, not the partition. So an
image is measured off an anchor (`cost/anchors.rs`: a link of the model's own program, else its
family's widest), plus what the plan moves over the anchor's. Off the same kernels' anchor the
model is within 380 B (Qwen3.5 4B from 0.8B); across kernel flags it is off by up to 2 KB, which
is why a model anchors on its own link. `anchors.rs` is regenerated from links when the kernels
change.

`search::search` walks from each of the family's presets (the made-for one first), moving one key
or two (a move another must follow: the positions, the units' grid) over a ladder of values; a
seed that does not plan is repaired a key at a time, following each move that gets it past the
check it failed. A candidate plans, serves the positions asked for (the made-for preset's, else
4096), keeps its head over the vocabulary, leaves every image `STACK_ROOM` plus a 512 B margin
and builds (`Program::build`). `wafer-compile search --model <m>` prints the ranking with the keys
each candidate moved from the made-for preset.

## Weights

`Program::bindings` lists every copy, the blocks' in order: a symbol, a rectangle and a `Source`
(`Staged` leading-range blocks, `Slice` for any sub-array: column ranges, per-PE padding, a row
repeated down or across a rectangle, optional tables; `Rcfg`) with the `Transform`s the cut
applied. The engine's placer (`engine-cerebras` `artifact::place`) executes an artifact's
bindings on a `Device`. The cut keeps every staged file under the 2 GiB a transfer may take (split
head tiles, `ple`'s 1,900 MiB storage files).

## The cut

What the bindings place is cut from the model's params, named as the trace names them (the
runtime's import contract lands them, so no checkpoint name appears in the crate). Each family
states its layout as data (`family/<family>/cut.rs`: a `tiles::spec::Spec`, every staged array and
how each element follows from the params): a weight PE grid's tiles (a matrix's rows or columns
gathered a column, its hidden dims split a row), gathered vectors, constants, and the tables a block
computes rather than copies (RoPE's cos / sin, the head's vocabulary tiles, Gemma's per-layer
inputs, MXFP4 experts as E5M2 pairs, each checked exact). One tiler (`tiles::cut`) makes every
family's arrays; rounding is numpy's (f16 to nearest even, bf16 truncated), so the arrays are the
earlier Python preps' byte for byte, with one deliberate exception. Gemma's per-layer inputs are
computed in f64 throughout and rounded once to f16, where the prep summed its projection in f32 in
its BLAS build's own order (OpenBLAS's K blocks). The cut's tables are the reference: each value is
the f64 result's nearest f16. About 0.1% of values differ from the prep's, nearly all by one f16
step; the larger gaps sit near zero, where the prep's f32 sums were the ones off.

## The artifact (what the engine runs)

`artifact::Artifact` is the built program as the host needs it: the `HostLayout` (the PEs it reads
and writes, the x runs, the head and the token loop's root, the softcap, the posted loop's length),
the bindings in placement order, every PE's rcfg words, the model facts of the host's math
(`Header::model`: vocabulary, hidden size, eps, `lgt`'s prescale), and the plan as a record.
`wafer-compile build` writes it (`program.json`, `host.json`, `bindings.json`, `rcfg.npy`,
`rcfg_n.npy`); the link (`build::link`) puts it in the compiled program's `artifact/`. The engine's
`ArtifactModel` reads it from there, places the load's tiles by its bindings and drives the program from it
alone; a program without an artifact is refused.

## Building in the engine

A load (`engine-cerebras` `prepare`) runs the same library calls as the tools: `target::plan` (the
positions the serving context asks for), `build::emit` and `build::link` with the configured toolchain (the CSL
kernels embedded in the crate, `sources`), and `tiles::stage` over the checkpoint's params landed
through the runtime's import contract. The program is cached by its `ProgramKey` (plan, levers, every kernel
source, the build `FORMAT`, the compiler version, the toolchain and the architecture), the cut by the plan's
shape, partition and contract and the checkpoint; each model keeps one of each.

## Tools

- `wafer-compile plan (--model <deployment> | --shape <shape.json>) [--preset <name>] [--set
  K=V,..] [--positions N] --out <dir>` writes `plan.json` and names the preset it took. `wafer-compile build --plan <plan.json> --out <dir> [--emit ..]
  [--levers ..] [--kernels <dir>]` writes `floor.json`, `rcfg.npy` / `rcfg_n.npy` / `rcfg.txt`,
  `kernels.json`, `layout/`, `host.json`, `plan.json`, `bindings.json` and `program.json`. `wafer-compile
  link --build <dir>` links it where the SDK toolchain is, checks the stack room and puts the artifact in
  `out/artifact`.
- `wafer-stage` (engine-cerebras) cuts a catalog deployment's checkpoint for its plan as a load does,
  or a plan over params already landed (`--plan`, `--landed`).
- `wafer-bench` (engine-cerebras) runs a linked program bare, on the host code Pie's engine runs
  (`ArtifactModel`), against a golden, on the simulator or the chip (`--cmaddr`).
- Python is left only where `transformers` is needed or no deployment exists: `<family>/mk_tiny_*.py`
  make the tiny checkpoints and their goldens; `baseline/landed.py` reads a tiny checkpoint as the
  import contract would (the tiny dims are no catalog model's, so the contract cannot); the baseline's
  capture and compare scripts.

## Verification

`tools/wafer/baseline` captures everything the build and the host produce for every shipping config
and each family's tiny (the host's placement, calls and derived values on a recording device; the
build's per-PE kernel and params, evaluated from `layout.csl` by `layout_eval.py`; the tiny cuts),
and compares two captures. A change that must not change behaviour is checked against the capture
of the tree before it (README there).

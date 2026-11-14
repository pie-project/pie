# The wafer program: architecture

A wafer program is a list of **blocks** over a logical **plan**. poem's spatial pass reads the
model's **shape** off its trace; the **kernel set** its layers' mixers take keys the **preset** (a
pinned plan's keys, or the search's) that a floorplan constructor turns into the plan; the plan's
**floorplan** states each element the layer program's features ask for, and the composer maps
elements to blocks; the core runs every build step over the blocks in order and knows no model,
no constructor and no block's vocabulary. The build lives here (`kernels-cerebras`); the
engine (`engine-cerebras`) runs its **artifact** and knows none of it (below).

```
.poem trace ─target::shape─▶ Shape ─KernelSet::of_shape─▶ KernelSet
 (or a shape file)               │                       │ pins (a named one, or the one made for the shape; else the search)
                                 └──────────┬────────────┘
                                            ▼ floorplan::plan (the set's constructor)   (poem_compiler::spatial::plan:
                                          Plan ─Floorplan::of─▶ Floorplan                 the first of them that builds)
                                            │  (the elements each feature asks for: units, segments, aux, head)
                                            │ ─compose::blocks─▶ [Block]
                                            │                        │ floor   ─▶ Floor  (units, head, stores; a PeRole a PE)
                                            │                        │ wire    ─▶ Wiring (typed links over open Classes / Slots)
                                            │                        │ settle  ─▶ Floor  (PEs placed where links cross)
                                            │                        │ colours ─▶ Palette (interference colouring, preferred colours)
                                            │                        ▼
                                            │                     Fabric::lay + validate (routes, queues, invariants)
                                            │
                                            └─kernels::of─▶ Kernels (Kind → KernelBuild: descriptor + typed params)
                                                                    │ blocks' overrides ─▶ Overrides (per-PE params)
                                                                    ▼
                                    emit::{rcfg (Kernel::rcfg over a Pe), params + layout, host}
                                    Program::bindings (blocks' copies) ─▶ artifact::Artifact ─▶ engine
```

Modules are under `src/`.

| Layer | Module | Holds | Knows |
|---|---|---|---|
| Plan | `plan.rs` | `Shape` (the model's facts), `Partition` (counts, split factors, the stacked arrangement), `Contract` (positions, CAP, LOOP, WL, the logits' prescale); `Preset` (a partition's named keys, levers and arrangement: a pinned plan's or the search's); serde, so a plan is a file | nothing physical: the seam to poem's spatial pass (below) |
| Target | `target.rs` | the wafer as poem's `SpatialTarget`: `shape` (a trace's `Shape`), `Wafer` (candidates: the pinned plan named or made for the shape; fit: the plan builds; else the search), `plan` / `plan_shape` | the trace's ops, the kernel sets |
| Roles | `role.rs` | `PeRole { kind: Kind, place: Place }`. `Kind` is open (name + build stem; the architecture's kinds are consts, a block declares more). `Place` is the wafer program architecture's positions: weight / CX / CY of a unit, head / head control, a storage column's PE, none | coordinates |
| Floor | `floor.rs` | `FloorBuilder` (blocks place onto it in order), units, gap / relay slots, the head block, store blocks, a `PeRole` a PE in a dense `Grid` | coordinates |
| Wiring | `wiring.rs` | `Chain`, `Multicast`, `Path`, `Turn`, `Note`; open `Class { name, index }`, `Slot { name, kind: SlotKind }`, `NoteKey`; the core's only own vocabulary is `LOOPBACK` and `PASS` | coordinates, logical colours |
| Floorplan | `floorplan/` | `Floorplan` (the IR between the plan and the floor: `Units { shape: UnitShape { col_runs, row_runs, control, layers, kv_rows }, path, count }`, `Segment`s, `Aux` (a side input's storage, the KV line), `Head { at: East / Band / Groups }`), `Requests` (the element each layer-program feature asks for), `Floorplan::of` (a plan's elements); `plan` and the four constructors (`snake`, `stacked`, `runs`, `split`: the points the old layouts were, each reading its keys off a preset, its partition and contract, its refusals; the stacked arrangement and its derivation) | the model's shape |
| Fabric | `fabric.rs` | `Palette` (`allocate`: an interference colouring of the classes' footprints, each class its block's preferred colour where free), routes, `PeIo` (slot → endpoint, notes), parks / spare / route words, the invariants | physical colours, route words |
| Blocks | `block/` | the `Block` trait and the blocks (table below); each owns its classes, slots, notes, kinds and colours as consts | coordinates, its own vocabulary |
| ABI | `abi.rs` | `Kernel` (source, owned symbols, params' names, route slots, rcfg writer), `params!` (a kernel's typed params struct), `words!` (an rcfg word layout), `KernelBuild`, `Kernels`, `Overrides` | the build's view of a kernel |
| Kernels | `kernels/` | `KernelSet` (the role kernels a layer program's mixers take: `Own`, `Shared`, `Windows`, `Experts`), `of` (the units' kernels by set, the head's by the head element, the store kernel by the storage, the relay by gap slots), `symbols` (by set); `Pe` (one PE as an rcfg writer sees it); `common` (the word layouts every set's weight / CX / CY / head / head control kernel shares, its codes named, the relay); `head` (`head_pe.csl`, `head_ctl.csl`: east, east by row groups); one module a set's units: `snake` (Own), `stacked` (Shared, and the store), `runs` (Windows), `split` (Experts, and the band) | the CSL kernels' ABI |
| Layers | `layers.rs` | the plan's layer program (`poem_compiler::spatial::layer::LayerProgram`), stated from its shape; pinned to the trace's lowering | the shape |
| Composer | `compose/` | `Needs` (read off the layer program) and `Schedule` (the layer program's phases on its floorplan), `blocks` (the blocks a floorplan's elements take), `KernelCopies` (what the role kernels read off each block, by kernel set: `copies`), `refuse` (what the set's kernels do not run, by name) | the blocks |
| Pins | `pins.rs`, `plans/` | the pinned plans: each shipping program's kernel set, keys, levers, made-for shape and arrangement | the shipping programs |
| Emit | `emit/` | rcfg, kernel params (levers, overrides) and `layout.csl`, `HostLayout` (everything the host driver reads and writes on the wafer) | kernel builds |
| Weights | `weights.rs` | `Binding`s (symbol, rect, source, transforms), `Copies`, the placer | the device |
| Program | `program.rs` | `Program { plan, blocks, floor, wiring, palette, fabric, kernels, overrides, floorplan, roles, schedule }`, `build`, `bindings` | all of the above |
| Artifact | `artifact.rs` | `Artifact`: the program as the engine runs it (`HostLayout`, bindings, rcfg words, the model facts of the host's math, the plan as a record) | files |

Rules:

- **The block is the unit of extension.** A model's blocks are what its layers need; a new dataflow part
  (a new segment kind, a storage column, a band head) is a new block. Nothing in the core names a
  model, a constructor or a block's classes, slots, notes or kinds.
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
  ...) and the notes other blocks left; no block restates the plan as its own geometry struct.
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
7. **Colour count.** Every colour < `COLOURS` (21): `Palette::allocate` refuses a class that meets
   every colour by name, with the classes it meets.
8. **Loop-backs.** Every root of a `SlotKind::Reduce { loopback: true }` slot loops back to its own
   ramp on `LOOPBACK`'s colour.
9. **Kernel-side rules** (not geometry; listed so nobody drops them): restore a route only after
   its oq drains (0738); ≥768 B stack room a PE (`wafer-compile link`, `Cslc::stack_room`); payload directories
   end in `_ieee`.

## The blocks

| Block | Floor | Wiring | Vocabulary | Copies |
|---|---|---|---|---|
| `units::Units { grid, halves, copies }` | units per `UnitGrid` (snake with gaps, or explicit origins); gap slots `Kind::RELAY`; with `halves` (a MoE column run) the weight / CY kinds refined by half (`moe::halves`: `WA`, `WM`, `CYA`, `CYM`, attention columns west of CX, MoE east) | `unit_chains`: row chains into CX, column chains into CY (two-level by `reduce.row` / `.col`), root loop-backs, `handoff`s along the visit order | `ROW_L1/L2/CAST`, `COL_L1/L2/CAST`, `HANDOFF_OUT/VERT[parity]/IN/ACROSS`; slots `ROW_REDUCE`, `COL_REDUCE`, `X_IN`, `X_OUT`, `RELAY_PRE`, `RELAY_POST`; where the rows come in runs, `row_runs::overrides` (`HC`, `AROW` (`ATT_ROW`, `PLAIN_ROW`): a weight / CX PE's run's dims, a weight PE's run's role) | `UnitsCopies`: `UnitCopies` per unit (quadrants of `w{u}`, CX `gam`, CY tables), `RunCopies` per unit and run block (`w{u}a` / `w{u}w`, `gama` / `gamw`), `MoeCopies` per unit and half; relays' rcfg |
| `segments::Scores { width, rows, cols }` | — | every attention row's attention columns in segments of `width` | `SCORE_L1`, `SCORE_CAST`; slot `SEGMENT` | — |
| `segments::Onorm { width }` | — | every CY row in segments (no single-PE ones) | `ONORM_L1`, `ONORM_CAST`; slot `ONORM` | — |
| `gdn::Gdn` (`Gdn::of(plan)`) | — | the (k, q) multicast per GDN segment and key row; `Gdn::source` is the weight kernel's source word | `GDN_KQ`; slot `GDN`; `SOURCE_*` | — |
| `head::HeadEast { cols, rows, token_loop, copies }` | the head east of the floor so far, control column in the middle | `head_rows` (into the control column), `head_input` (two CX rows a head column: relay north, column multicast) | `HEAD_RELAY`, `HEAD_COL_CAST`, `HEAD_ROW_WORD`; slots `HEAD_ROW_REDUCE`, `HEAD_IN`, `HEAD_RELAY_IN`, `HEAD_CAST`; note `ROLE` (`ROLE_RELAY`, `ROLE_SOURCE`) | `HeadCopies` (`hw`, `hg`, `hemb`, rcfg) |
| `token_loop::TokenLoop` | — | the control column's max-reduce, the embedding's return into unit 0's CX | `LOOP_L1/L2/CAST`, `EMB_UP/DOWN/SPLIT/WEST`; slots `LOOP_REDUCE`, `EMB_IN`; note `TURN` (`TURN_TOP/EVEN/ODD`) | — |
| `ple::Ple { columns }` | a storage block above each unit column's top unit (`Place::Store`, kind `STORE`; its CX column route-only); `settle`: every idle PE west of the head outside the units and storage that a link crosses runs the store kernel route-only | `TOKEN` west along each storage row from the head control column; `bus(i)` down each weight column of column `i` into every unit's CY | `TOKEN`, `bus(i)`; `NTW` (route-only PEs outside the storage: 1) | storage tiles `pt<col>_<n>` (staged files under 1,900 MiB), the route-only PEs' rcfg |
| `shared_kv::SharedKv` | — | `LINE` north from the source unit's CY row through the sharers above it | `LINE`; `SH`, `Ic` on the sharers' weight / CY PEs | — |
| `moe::Router` | — | router segments on the CY row's MoE columns | `ROUTER` (a segment slot) | — |
| `band::Band` | the head as a band below the units with a control row | the x down the MoE columns (`BAND_VERT`), the band columns' chains into the control row | `BAND_VERT`; slots `BAND_IN`, `BAND_OUT`, `BAND_CAST` | the band's head tiles |
| `band::BandLoop` | — | the control row's max-reduce, the bus, the embedding's return (`BAND_V1`, `BAND_W2`); prefers `BUS_COLOUR` for `token_loop::EMB_WEST` (the band's kernels read the bus's colour off the palette) | `BAND_V1`, `BAND_W2`; slots `BUS`, `BUS_TX`, `V1_TX`, `V1_RX`, `W2_TX` | — |
| `head_groups::HeadGroups { cols, rows, copies }` | the head east of the units (token loop on) | `head_rows`; per group the x chain north on the last unit (`HEAD_GROUP`), its root east, the column multicast; the embedding back up, down, west into unit 0 and down the chain; the max-reduce | `HEAD_GROUP`; slots `CHAIN_IN`, `CHAIN_OUT`, `CAST`; notes `FORWARD`, `head::ROLE`, `token_loop::TURN` | `HeadCopies` |

The role kernels' layer phases come from the plan's layer program and floorplan (`compose::Schedule`: `NRM` from a deferred norm or the expert halves, `MIXK` from the mixers, `ACT` from the MLP's gate activation, `PLE` from a side input, `KVS` from a shared cache, `HG` from a head fed by row groups), on `Program::schedule`; every set's kernels read them there. Row roles (`AROW`) are the row runs' per-PE overrides; k = v layers (`KEV`) are the `Windows` set's (`Schedule::kev`: its cut leaves a global layer's v rows out, for a checkpoint with no global `v_proj`, which the trace's packed bank does not tell apart); the GELU form (`GELUT`) is a lever the plan carries (Gemma 4 12B's: `GELUT:1`).

### The floorplan

The floorplan (`floorplan/`) is the IR between the plan and the floor. Each feature of the layer program asks for its own element (`floorplan::Requests`), whatever else the program has:

| feature | element | block |
|---|---|---|
| attention over a cache | scores segments on the attention rows' attention columns (`Segment::Scores`) | `Scores` |
| the gated delta rule | its (k, q) multicast and o-norm (`Segment::Gdn`, `Segment::Onorm`) | `Gdn`, `Onorm` |
| routed experts | a MoE column run (`ColRole::Moe`) and the router's segments (`Segment::Router`) | `Units { halves }`, `Router` |
| a window | KV rows sized to it (`UnitShape::kv_rows`, an attention row run) | the units' rows |
| a per-token side input | storage above each unit column (`Aux::Store`) | `Ple` |
| a cache read across layers | the KV line (`Aux::KvLine`) | `SharedKv` |

The units are one `UnitShape` (column runs by role, row runs by dims and role, the control lines, K layers) visited along a path (`floor::UnitLayout`: a snake of slots with gap slots, or explicit origins); the head is east of the units (`HeadEast` + `TokenLoop`), east fed by row groups (`HeadGroups`), or a band below them (`Band` + `BandLoop`). `Floorplan::of` reads each element's choice off the plan: the path is the stacked arrangement's origins where the plan has one, two unit columns from the band where the head is a band, the plan's unit slots, else two unit columns down then up. The composer (`compose::blocks`) maps the elements to blocks in one order (units, segments, aux, head, token loop) and never names a constructor.

The four constructors (`floorplan::{snake, stacked, runs, split}`, chosen by the kernel set the pinned plans and the search's moves are keyed for) are the points the old hand layouts were:

| constructor | units | path | head | beside the units |
|---|---|---|---|---|
| snake | one plain column run, one row run | a snake of UNX x UNY slots, at most one gap slot | east | — |
| stacked | one plain column run, KV on PYW rows | explicit origins in two unit columns (the pinned arrangement made for the unit shape, else `Arrangement::derive`) | east, token bus on | storage, KV line |
| runs | one plain column run, attention / plain row runs | two unit columns, down then up | east by row groups | — |
| split | attention / MoE column runs | two unit columns from the band, up then down | a band | — |

Colours are the palette's (`Palette::allocate`): an interference colouring of the classes' footprints (two classes meet where both route at a PE, unless one pass-through word there), each class its block's preferred colour where no class it meets kept it first, the rest the lowest colour free. Every shipping program keeps every preferred colour.

What each block's PEs hold is the kernel set's (`compose::copies`). The sets' kernels:

| Kernel set | Models | Kernels |
|---|---|---|
| `Own` | Qwen3.5, Qwen3, Llama 3 | `kernels::snake` (`weight` (NRM 0, MIXK 0), `cx` (NRM 0), `cy` (MIXK 0, NRM 0)) |
| `Experts` | gpt-oss, Qwen3 MoE (routed experts) | `kernels::split` (`wa`, `wm`: `weight` (MIXK 2, 3), `cx`, `cya`, `cym`) |
| `Shared` | Gemma 4 E2B / E4B (a side input and shared caches) | `kernels::stacked` (`weight` (NRM 1, MIXK 1, PLE 1), `cx` (NRM 1, PLE 1), `cy` (NRM 1, MIXK 1, PLE 1, KVS 1)) |
| `Windows` | Gemma 4 12B (sliding windows) | `kernels::runs` (`weight` (NRM 1, MIXK 1, row roles by `AROW`), `cx` (NRM 1, HG 1), `cy` (NRM 1, MIXK 1, KEV 1 from `Schedule::kev`)) |

and by element: the head east `head::HEAD_PE` + `head::HEAD_CTL` (`HEAD_CTL_TB`, the token bus on, where a side input has storage), east by row groups `head::HEAD_GROUPS` + `HEAD_CTL`, the band `split::band` (`head_pe.csl` BAND 1, `head_ctl.csl` with the bus reserved); the storage `stacked::store` (`ple_gemma`); gap slots `common::RELAY`.

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

// in head::east():
out.insert(Kind::HEAD, KernelBuild::of(&HEAD_PE, HeadPe { VR: vr, HDc: 2 * hc, NPX: 1, HACC16: 0 })?);
```

An rcfg writer reads one PE through `Pe`: `pe.member(slot)` / `root` / `halves` / `colour` /
`turn` (the queues the links bound), `pe.note(key)`, `pe.c(class)` (the palette), `pe.parks(n)`,
`pe.spare(finals, comptime)`, `pe.routes(slots)`, `pe.role()`. The shared leading words are
`common::{weight, cx, cy, head, hctl}` over `words!` structs (`WeightWords`, `CxWords`, `CyWords`,
`HeadWords`, `HctlWords`); a kernel passes what is its own (`weight`'s two words and turn width,
`CxIo`'s flags / finals / tail, `cy`'s segment, `HeadIo`) and appends its tail.

Params resolve per PE in `emit::params::tile`: the kind's `KernelBuild::params`, then the PE's
`Overrides` (each must be a param the kind takes). Levers (the pinned plan's, carried by the plan, and `wafer-compile build --levers`'s; the last
argument) override a param in every kind that takes it; `<kind>.<NAME>` (`w.FIFOW:16`, E4B's
weight FIFO) in that kind only; a lever no kernel takes is an error.

`layout.csl` is written from the floor: each maximal rectangle of PEs with one kind and one
resolved param list is one loop nest, every param a literal, so the link passes no params.
The per-kind sources are the kernels', with one-word dummies for the kernel set's other exported
symbols (`kernels::symbols`, export order).

### Op modules (`programs/wafer/ops`)

The weight PEs (`weight.csl`) and the row control PEs (`cx.csl`) are one kernel each for every
layer program: the layer's phases (the input norm `NRM`, the mixer kind `MIXK`, the PLE round, head
groups) are comptime params, and the mixers are op modules. The other roles still run their set's
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
a set's own is in the head and PLE kernels: the token loop's winner merge and the handoff relays.

## Adding a model

1. **Shape.** A model poem traces needs nothing here: `target::shape` reads it. A trace op that
   runs on the wafer and that no kernel runs (a new mixer, a new activation) is refused by name
   (`LAYER_OPS`, `OUTER_OPS`); teach `target.rs` the op and give the plan's `Shape` the field the
   blocks need.
2. **Kernel set and floorplan.** `KernelSet::of` picks the role kernels from what the layer
   program's mixers need. A model of an existing set plans by the search from its set's pinned
   plans; once its program is validated on the chip, its plan is pinned (`plans/<name>.json`:
   the set, keys, levers and, for stacked columns, the arrangement; one `include_str!` in
   `pins.rs`). A new feature asks for its element (`floorplan::Requests`, `Floorplan::of`) and the
   composer maps it to its block (`compose::blocks`); a new set is a rule in `KernelSet::of`, its
   constructor (`floorplan/<point>.rs`), copies (`compose::copies`), refusals
   (`compose::refuse`), kernels (`kernels/<set>.rs`) and cut (`tiles/<set>.rs`).
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
6. **Kernels** in `kernels/<set>.rs`: one `params!` struct and one `Kernel` const a CSL source,
   the writer over `Pe` using `common::*` for the shared words. Values that differ by PE are
   `Block::overrides`. The CSL imports its ops from `programs/wafer/ops`; an op no module has is a
   new module there (and a row in `sources::OPS`), not code in the kernel.
7. **Copies** in each block's `bindings`, in placement order (the host's device protocol); the
   set's cut states the arrays they copy (`tiles/<set>.rs`).
8. **Host.** Anything the driver must find goes in `HostLayout`, derived from the program.
9. **Verify** with the baseline harness (below) and the tiny simulator run.

## The plan seam (poem)

`Program::build(plan)` is the whole physical compiler; its only input is a `Plan`. The plan
comes from poem's spatial pass (`poem_compiler::spatial::plan` over `target::Wafer`): the trace
gives the `Shape` (layer kinds, attention / MLP / MoE / GDN / PLE specs, head, RoPE, norms), the
shape picks the kernel set, and the pinned plan made for the shape (`pins`: `plans/*.json`, the keys,
levers and arrangement of each shipping model's chip-validated program) is the partition tried (a
named one, `--preset <name>`, tries only itself; `--set` overrides its keys). A checkpoint with no
catalog deployment states its shape in a file (`target::plan_shape`; the tiny test checkpoints'
are `tools/wafer/baseline/shapes/`, their pinned plans the `tiny*` ones).

A model with no pinned plan made for its shape (or whose pinned plan does not build, and none is
named) plans by the search (`search`, below): there is no table of presets to try.
`poem-compiler/SPATIAL.md` has the target-neutral half.

## Cost and search (`cost`, `search`)

`cost::Features` reads what a decode step does off a plan: the weight elements the busiest PE of
each phase multiplies, the cycles the reductions' chains take end to end, the fabric distance the
collectives cross down the columns (the rows) and along the rows (the columns), the KV elements a
row reads, the head PE's elements. `cost::time` prices them at the chip's rates. The weight and head
elements' are read off the phase stamps of `TSTAMP:1` runs (requests 1179a/c/d: a weight PE's
matvec phases, 0.46 cycles an element; the head PE's, 0.537, alike on Llama, E4B and Gemma 12B).

A chain (`cost::chain_cycles`) is timed by its critical path: each half either side of the root
in groups of G2 counted from the root out, the farthest the remainder; every group's members add
and forward to their end at once, and the level-2 sum starts at the farthest end, each end merging
its group's sum into it once both are in. A short far group starts the level-2 sum early, so a G2
that leaves a remainder beats the divisors either side of it: on the fabric simulator, a row
chain of 128 members a half (a unit 256 columns wide, 8 rows, the 9B's 16 dims a row:
`scratch/genG2`) takes 70 cycles a layer more at G2=8 and at G2=16 than at 12, and 94 more at 24,
which 7 cycles an add-forward, 7 a merge and 3 a router hop reproduce. That is the chip's shape:
the best G2 is 12 where the halves are 128 (the Qwen3.5s, E4B, Qwen3-8B), 16 on the 12B (172 and
141), 8 of E2B's 8 / 16 / 24.

The token loop's max-reduce down the head control line (`cost::loop_reduce_cycles`) is a tree's
critical path (`cost::tree_cycles`): a PE merges a (value, id) pair 75 cycles after its last
child's is in, a pair 0.5 cycles a hop on its way, fitted to the simulator's max-reduce phase on a
48-row tiny head (610 cycles as the chain at G2W=5, 533 as the tree at G2W=2, G3W=4). With `G3W`
the line reduces in three levels (`wiring::three_level`: each half in level-3 groups of
`1 + G2W G3W`, the group's first PE taking the level-2 sum of the rest and the level-3 sum from
beyond), on the chain's four colours and two input queues, so `head_ctl.csl` is unchanged; the
merge is a total order (the larger value, ties to the lower id), so the winner is the same. On the
0.8B's 1,153 rows the model gives 3,813 cycles at G2W=24 and 1,985 at G2W=8, G3W=8; on the
mini's 748 band columns 3,030 and 1,660 at G2W=7, G3W=9 (the chip measured 3,594 for the chain).

The chip's rate a chain cycle is fitted to what moving a pinned plan's keys did to the bare step
(requests 1151-1306, 25 moves: Llama VR=112 / 144 / 180 and G2=8, 0.8B PX=128 and G2=8 / 12, Gemma
12B's pick and G2=8 / 12, 4B PY=192 and G2=8 / 12, Qwen3-8B G2=8 / 16, G2=8 / 12 / 24 on the 9B,
E4B, E2B and gpt-oss), with the KV elements' and the rows' span held at the earlier fit's: it comes
to 0.96 of a cycle at 875 MHz, so the simulator's hop cycles price the chip's chains. The model
then gives every swept model the chip's best G2 but gpt-oss's (`time::tests`): its row chains'
halves are the 64 attention and 288 MoE columns, the long half makes G2=8 cost 2.6 µs, and the chip
gained 2.5. It gives the G2 moves about half their size on the chip (E4B's G2=12: 7 µs of 21), and
the 0.8B's PX move and the 4B's PY move, which moved the chains with the KV and the span, 12 and 7
µs short. Fitted with one move held out it misses the held-out one by 6.6 µs RMS, 21 µs at most
(E4B's G2=12); the hop model it replaces missed by 5.9 µs RMS on these moves, ranked G2=8 ahead of
12 on E4B and put G2=12's gain at a third of the chip's. Freed, the KV rate takes the rest of the G2
gains (the chain at 2.2 cycles, 3.9 µs RMS held out) but costs 15 cycles an element and leaves
gpt-oss's layer less than nothing: only the 4B's PY / NE move moves both, so the moves cannot tell
the chains' rate from the KV's. The columns cost nothing the chip resolves (`SPAN_COL` 0): halving
the 0.8B's cost what its doubled matvec and KV elements do. Each layer's own cost is read off the
measured step of the model's pinned plan (`time::MEASURED`: the bare step where the chip ran it,
else Pie decode; a model the chip has not run takes the median).

`cost::memory` is the PE image model. The bytes a partition moves are the weight PE's (its K
layers' tiles, its KV cache, its scores' buffers) and the head PE's (its vocabulary rows' tiles and
embeddings); code and the small buffers move with the kernels' flags, not the partition. So an
image is measured off an anchor (`cost/anchors.rs`: a link of the model's own program, else its
kernel set's widest), plus what the plan moves over the anchor's. Off the same kernels' anchor the
model is within 380 B (Qwen3.5 4B from 0.8B); across kernel flags it is off by up to 2 KB, which
is why a model anchors on its own link. `anchors.rs` is regenerated from links when the kernels
change.

`search::search` walks from each of the kernel set's pinned plans (the made-for one first), moving one key
or two (a move another must follow: the positions, the units' grid) over a ladder of values; a
seed that does not plan is repaired a key at a time, following each move that gets it past the
check it failed. A candidate plans, serves the positions asked for (the made-for plan's, else
4096), keeps its head over the vocabulary, leaves every image `STACK_ROOM` plus a 512 B margin
and builds (`Program::build`); it ranks ahead of the made-for plan only where it predicts a step
0.3% faster (`search::RESOLUTION`: the moves under it are within the time model's error). `wafer-compile search --model <m>` prints the ranking with the keys
each candidate moved from the made-for plan.

## Weights

`Program::bindings` lists every copy, the blocks' in order: a symbol, a rectangle and a `Source`
(`Staged` leading-range blocks, `Slice` for any sub-array: column ranges, per-PE padding, a row
repeated down or across a rectangle, optional tables; `Rcfg`) with the `Transform`s the cut
applied. The engine's placer (`engine-cerebras` `artifact::place`) executes an artifact's
bindings on a `Device`. The cut keeps every staged file under the 2 GiB a transfer may take (split
head tiles, `ple`'s 1,900 MiB storage files).

## The cut

What the bindings place is cut from the model's params, named as the trace names them (the
runtime's import contract lands them, so no checkpoint name appears in the crate). Each kernel
set states its cut as data (`tiles/<set>.rs`: a `tiles::spec::Spec`, every staged array and
how each element follows from the params): a weight PE grid's tiles (a matrix's rows or columns
gathered a column, its hidden dims split a row), gathered vectors, constants, and the tables a block
computes rather than copies (RoPE's cos / sin, the head's vocabulary tiles, Gemma's per-layer
inputs, MXFP4 experts as E5M2 pairs, each checked exact). One tiler (`tiles::cut`) makes every
set's arrays; rounding is numpy's (f16 to nearest even, bf16 truncated), so the arrays are the
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
  K=V,..] [--positions N] --out <dir>` writes `plan.json` and names the pinned plan it took (or the
  search's). `wafer-compile build --plan <plan.json> --out <dir> [--emit ..]
  [--levers ..] [--kernels <dir>]` writes `floor.json`, `rcfg.npy` / `rcfg_n.npy` / `rcfg.txt`,
  `kernels.json`, `layout/`, `host.json`, `plan.json`, `bindings.json` and `program.json`. `wafer-compile
  link --build <dir>` links it where the SDK toolchain is, checks the stack room and puts the artifact in
  `out/artifact`.
- `wafer-stage` (engine-cerebras) cuts a catalog deployment's checkpoint for its plan as a load does,
  or a plan over params already landed (`--plan`, `--landed`).
- `wafer-bench` (engine-cerebras) runs a linked program bare, on the host code Pie's engine runs
  (`ArtifactModel`), against a golden, on the simulator or the chip (`--cmaddr`).
- Python is left only where `transformers` is needed or no deployment exists: `<model>/mk_tiny_*.py`
  make the tiny checkpoints and their goldens; `baseline/landed.py` reads a tiny checkpoint as the
  import contract would (the tiny dims are no catalog model's, so the contract cannot); the baseline's
  capture and compare scripts.

## Verification

`tools/wafer/baseline` captures everything the build and the host produce for every shipping config
and each kernel set's tiny (the host's placement, calls and derived values on a recording device; the
build's per-PE kernel and params, evaluated from `layout.csl` by `layout_eval.py`; the tiny cuts),
and compares two captures. A change that must not change behaviour is checked against the capture
of the tree before it (README there).

## The stream regime

A reading that runs whole sequences a step (a diffusion transformer's denoise) builds through
its own path, not the decode blocks: `wafer-compile stream` (`stream::compile`). Its units come
from the stream program's sides node by node (`stream::unit`), their roots' slices and tiles and
the row stream's order from `stream::place`, its elements (`floorplan::stream`: a unit a side, a
K / V column a kv head, the lane bus) from the floorplan's `Requests::{stream, joint, lanes}`,
their links from `Stream::wiring`, coloured and laid by the same `Palette::allocate` and
`Fabric::lay`. `emit::stream` writes `layout.csl` and a kernel a PE kind; `stream::host` the
arrays the host writes, and which of them a step's inputs make (`host.json`'s entries: gathers of
the trace's values, computed each step by engine-cerebras' `reading` from the step's own inputs
and run by `wafer-bench --reading`). The decode build never reaches any of it (its outputs are
unchanged).
GENERIC.md §4.1 (M1d) has the details and the simulator results; `tools/wafer/stream` runs it.

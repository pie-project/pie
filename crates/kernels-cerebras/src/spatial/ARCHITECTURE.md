# The wafer program: architecture

A wafer program is a list of **blocks** over a logical **plan**. poem's spatial pass reads the
model's **shape** off its trace; the family the shape needs turns it and a **preset** into the plan
and names the blocks it composes; the core runs every build step over the blocks in order and knows
no model, no family and no block's vocabulary. The build lives here (`kernels-cerebras`); the
engine (`engine-cerebras`) runs its **artifact** and knows none of it (below).

```
.poem trace ─target::shape─▶ Shape ─family::serving─▶ Family
 (or a shape file)               │                       │ presets (a named one, or each in order)
                                 └──────────┬────────────┘
                                            ▼ Family::plan            (poem_compiler::spatial::plan:
                                          Plan ─Family::blocks─▶ [Block]   the first preset that builds)
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
| Kernels | `kernels/` | `Pe` (one PE as an rcfg writer sees it); `common` (the word layouts every family's weight / CX / CY / head / head control kernel shares, its codes named, the relay); `head` (`head_pe.csl`, `head_ctl.csl`); one module a family's own kernels: `qwen`, `gptoss`, `gemma`, `gemma12` | the CSL kernels' ABI |
| Families | `family/` | `serving` (the family a shape needs); `qwen`, `gptoss`, `gemma`, `gemma12`: presets, the plan of a shape under a preset, the tiling a prep cuts by, the block list, the kernels | the model's shape |
| Emit | `emit/` | rcfg, kernel params (levers, overrides) and `layout.csl`, `HostLayout` (everything the host driver reads and writes on the wafer) | kernel builds |
| Weights | `weights.rs` | `Binding`s (symbol, rect, source, transforms), `Copies`, the placer | the device |
| Program | `program.rs` | `Program { plan, blocks, floor, wiring, palette, fabric, kernels, overrides, family }`, `build`, `bindings` | all of the above |
| Artifact | `artifact.rs` | `Artifact`: the program as the engine runs it (`HostLayout`, bindings, rcfg words, the model facts of the host's math, the plan as a record) | files |

Rules:

- **The block is the unit of extension.** A model is a family listing blocks; a new dataflow part
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
   its oq drains (0738); ≥768 B stack room a PE (`tools/wafer/chk_stack.py`); payload directories
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

Compositions (`family/*.rs::blocks`):

| Family | Blocks | Kernels |
|---|---|---|
| Qwen3.5 / Qwen3 | `Units` (snake, at most one gap slot), `Scores` when any layer attends, `Gdn` + `Onorm` when any layer is GDN, `HeadEast`, `TokenLoop` under the contract's token loop | `kernels::qwen` (`weight_qwen`, `ctl_cx`, `ctl_cy_qwen`) + `head` + `common::RELAY` |
| gpt-oss | `MoeUnits` (a snake starting upward), `AttnScores`, `Router`, `Band`, `BandLoop` | `kernels::gptoss` (`wa`, `wm`, `cx`, `cya`, `cym`, the band head; `head_ctl.csl` with the bus reserved) |
| Gemma 4 E2B / E4B | `Units` (explicit origins: the arrangement its unit shape names), `Scores` (a half row), `Ple`, `SharedKv`, `HeadEast`, `TokenLoop` | `kernels::gemma` (`weight_gemma`, `ctl_cx_gemma`, `ctl_cy_gemma`, `head_ctl_gemma`, `ple_gemma`) + `head::HEAD_PE` |
| Gemma 4 12B | `RowRuns` (explicit origins, attention / plain row runs), `HeadGroups` | `kernels::gemma12` (`weight_gemma12`, `ctl_cx_gemma12`, `ctl_cy_gemma12`, `head_gemma12`) + `head::HEAD_CTL` |

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
`Overrides` (each must be a param the kind takes). Levers (`wafer-compile --levers`, `build.py`'s last
argument) override a param in every kind that takes it; `<kind>.<NAME>` (`w.FIFOW:16`, E4B's
weight FIFO) in that kind only; a lever no kernel takes is an error.

`layout.csl` is written from the floor: each maximal rectangle of PEs with one kind and one
resolved param list is one loop nest, every param a literal, so `link.sh` passes no params.
The per-kind sources are the kernels', with one-word dummies for the family's other exported
symbols (`Family::symbols`, export order).

## Adding a model

1. **Shape.** A model poem traces needs nothing here: `target::shape` reads it. A trace op the
   reader does not know (a new mixer, a new activation) is refused by name; teach `target.rs` the op
   and give the plan's `Shape` the field the blocks need.
2. **Family.** `family::serving` picks the family from what the shape needs. A model of an existing
   family is a new preset in the family's table (`family/qwen.rs::PRESETS`, Gemma's presets and
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
   `Block::overrides`.
7. **Copies** in each block's `bindings`, in placement order (the host's device protocol); the
   prep tiles the checkpoint for them by the family's `tiling`.
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

What is left for the pass to own is the preset tables: a search that proposes the counts and split
factors (and a cost model that ranks them) replaces them, and everything after `Plan` stays as it
is. `poem-compiler/SPATIAL.md` has the target-neutral half.

## Weights

`Program::bindings` lists every copy, the blocks' in order: a symbol, a rectangle and a `Source`
(`Staged` leading-range blocks, `Slice` for any sub-array: column ranges, per-PE padding, a row
repeated down or across a rectangle, optional tables; `Rcfg`) with the `Transform`s the prep
applied. The engine's placer (`engine-cerebras` `spatial::artifact::place`) executes an artifact's
bindings on a `Device`; `tools/wafer/compiler.py` executes the same `bindings.json` on the appliance
runners. The preps keep every staged file under the 2 GiB a
transfer may take (split head tiles, `ple`'s 1,900 MiB storage files).

## The artifact (what the engine runs)

`artifact::Artifact` is the built program as the host needs it: the `HostLayout` (the PEs it reads
and writes, the x runs, the head and the token loop's root, the softcap, the posted loop's length),
the bindings in placement order, every PE's rcfg words, the model facts of the host's math
(`Header::model`: vocabulary, hidden size, eps, `lgt`'s prescale), and the plan as a record.
`wafer-compile build` writes it (`program.json`, `host.json`, `bindings.json`, `rcfg.npy`,
`rcfg_n.npy`); `link.sh` puts it in the compiled program's `artifact/`. The engine's
`ArtifactModel` reads it from there, places the prep's tiles (`PIE_CEREBRAS_WEIGHTS=<prep
dir>:<prefix>`) by its bindings and drives the program from it alone; it builds nothing, and a
program without an artifact is refused.

## Tools

- `wafer-compile plan (--model <deployment> | --shape <shape.json>) [--preset <name>] [--set
  K=V,..] --out <dir>` writes `plan.json` and `tiling.json` (what a prep cuts the checkpoint by, and
  the preset's name). `wafer-compile build --plan <plan.json> --out <dir> [--emit ..] [--levers ..]`
  writes `floor.json`, `rcfg.npy` / `rcfg_n.npy` / `rcfg.txt`, `kernels.json`, `layout/` (with
  `dims.env`), `host.json`, `plan.json`, `bindings.json` and `program.json`.
- The preps (`qwen/prep_qwen.py`, `gemma/prep_gemma.py`, `gemma12/prep_gemma12.py`,
  `gptoss/prep_gptoss.py`) keep the checkpoint maths and cut it as a plan directory's `tiling.json`
  says.
- `tools/wafer/build.py <out> <plan.json> [levers]` is the build directory of any family (the
  `build_{gemma,gemma12,gptoss}.py` add their runner); `link.sh` links it, checks stack room and
  puts the artifact beside the compiled program; the `gotiny.sh` scripts run a tiny prep on the
  simulator; `regen.sh` rebuilds the shipping programs (README there).

## Verification

`tools/wafer/baseline` captures everything the build and the host produce for every shipping config
and each family's tiny (the host's placement, calls and derived values on a recording device; the
build's per-PE kernel and params, evaluated from `layout.csl` by `layout_eval.py`; the tiny preps),
and compares two captures. A change that must not change behaviour is checked against the capture
of the tree before it (README there).

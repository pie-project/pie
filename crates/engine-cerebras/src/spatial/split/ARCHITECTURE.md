# Role-split program: architecture

A role-split program is a list of **blocks** over a logical **plan**. A family reads the plan
(today off the prep's meta) and names the blocks it composes; the core runs every build step over
the blocks in order and knows no model, no family and no block's vocabulary.

```
Meta ─Family::plan─▶ Plan ─Family::blocks─▶ [Block]
                      │                        │ floor   ─▶ Floor  (units, head, stores; a PeRole a PE)
                      │                        │ wire    ─▶ Wiring (typed links over open Classes / Slots)
                      │                        │ colours ─▶ Palette
                      │                        ▼
                      │                     Fabric::lay + validate (routes, queues, invariants)
                      │
                      └─Family::kernels─▶ Kernels (Kind → KernelBuild: descriptor + typed params)
                                              │ blocks' overrides ─▶ Overrides (per-PE params)
                                              ▼
              emit::{rcfg (Kernel::rcfg over a Pe), params + layout, host, meta}
              Program::bindings (blocks' copies) ─▶ weights::place
```

| Layer | Module | Holds | Knows |
|---|---|---|---|
| Plan | `plan.rs` | `Shape` (the model), `Partition` (counts, split factors), `Contract` (positions, CAP, LOOP, WL) | nothing physical: the seam to poem's spatial compiler (a trace makes the `Shape`, a search the `Partition`) |
| Roles | `role.rs` | `PeRole { kind: Kind, place: Place }`. `Kind` is open (name + build stem; the architecture's kinds are consts, a block declares more). `Place` is the role-split architecture's positions: weight / CX / CY of a unit, head / head control, a storage column's PE, none | coordinates |
| Floor | `floor.rs` | `FloorBuilder` (blocks place onto it in order), units, gap / relay slots, the head block, store blocks, a `PeRole` a PE in a dense `Grid` | coordinates |
| Wiring | `wiring.rs` | `Chain`, `Multicast`, `Path`, `Turn`, `Note`; open `Class { name, index }`, `Slot { name, kind: SlotKind }`, `NoteKey`; the core's only own vocabulary is `LOOPBACK` and `PASS` | coordinates, logical colours |
| Fabric | `fabric.rs` | `Palette` (the blocks' colours, later replacing earlier), routes, `PeIo` (slot → endpoint, notes), `parks` / `spare` / `route_words`, the invariants | physical colours, route words |
| Blocks | `block/` | the `Block` trait and the blocks: `units`, `segments` (`Scores`, `Onorm`), `gdn`, `head` (`HeadEast`), `token_loop`; each owns its classes, slots, notes and colours as consts | coordinates, its own vocabulary |
| ABI | `abi.rs` | `Kernel` (source, owned symbols, params' names, route slots, rcfg writer), `params!` (a kernel's typed params struct), `words!` (an rcfg word layout), `KernelBuild`, `Kernels`, `Overrides` | the build's view of a kernel |
| Kernels | `kernels/` | `Pe` (one PE as an rcfg writer sees it), `common` (the word layouts every family's weight / CX / CY / head / head control kernel shares, its codes named), `head` (`head_pe.csl`, `head_ctl.csl`), `q35` (Qwen's) | the CSL kernels' ABI |
| Families | `family/` | `qwen` (blocks); `gptoss`, `gemma`, `gemma12` (one `block::Legacy` each until ported) | the model |
| Emit | `emit/` | rcfg (`Kernel::rcfg` over a `Pe`), kernel params (levers, overrides) and `layout.csl`, typed `Meta`, `HostLayout` | kernel builds |
| Weights | `weights.rs` | `Binding`s (symbol, rect, source, transforms), `Copies`, the placer | the device |
| Program | `program.rs` | `Program { plan, blocks, floor, wiring, palette, fabric, kernels, overrides, family }`, `build`, `bindings` | all of the above |

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
- **Everything up to `Program` is pure:** no device, no files. `split-geo` and the tests rely on
  this. Rust is the only geometry; the Python tools call `split-geo`.

## Invariants (`Plan::validate`, `Fabric::lay` / `validate`, `check_route_slots`)

The hardware lessons of the bring-up, made checks:

1. **Route clash.** Two words for one colour at one PE is an error.
2. **Parked queues.** An unused input queue never parks on a colour the PE sends on. Parks come
   first from colours with no route, then from pass-through colours (`Fabric::parks`). The 0704
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
   its oq drains (0738); ≥768 B stack room a PE (`tools/split/chk_stack.py`); payload directories
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

Qwen's composition (`family/qwen.rs::blocks`): `Units` (snake, at most one gap slot), `Scores`
when any layer attends, `Gdn` + `Onorm` when any layer is GDN, `HeadEast`, `TokenLoop` under the
contract's token loop. Its kernels are `kernels::q35` (`weight_q35.csl`, `ctl_cx.csl`,
`ctl_cy_q35.csl`) with `kernels::head` and `common::RELAY`.

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
`Overrides` (each must be a param the kind takes). Levers (`split-geo --levers`, `mkfull.py`'s last
argument) override a param in every kind that takes it; `<kind>.<NAME>` (`w.FIFOW:16`, E4B's
weight FIFO) in that kind only; a lever no kernel takes is an error.

`layout.csl` is written from the floor: each maximal rectangle of PEs with one kind and one
resolved param list is one loop nest, every param a literal, so `link.sh` passes no shape params
(the layout's `SPLIT_GEO` param marks such a build; `mod.rs::is_split` finds it in `out.json`).
The per-kind sources are the kernels', with one-word dummies for the family's other exported
symbols (`Family::symbols`, export order).

## Porting a family onto blocks

gpt-oss, Gemma 4 E2B / E4B and Gemma 4 12B still run as one `block::Legacy` each: their old
`floor` / `wire` / `palette` / `bindings` as `block::Whole`, their old per-kind kernel tables with
the family's one rcfg function (`block::legacy_kernels`), the old role-implied params
(`block::legacy_overrides`). Their output is the pre-port output byte for byte. To port one:

1. **List the blocks.** Start from the standard ones and their order: units first, then what
   places against them, then wiring-only blocks. Reuse a block where its doc fits exactly;
   otherwise write a new block module under `block/` (a new block, not a flag on a shared one).
   Likely new blocks:
   - gpt-oss: `band` (the head as a band below the units with a control *row*: its floor, its
     handoff down the MoE columns, the band bus, the embedding's return up unit 0's MoE column);
     `router` segments on the CY row (MoE columns); the attention / MoE halves are kinds the units
     block's roles are refined into (`WA`, `WM`, `CYA`, `CYM` with `FloorBuilder::each_role`, in a
     `moe_columns` block or the band's floor step).
   - Gemma: `ple` (the storage columns above each unit column, their token word `TB` and bus `PB`,
     the route-only pass PEs; `Place::Store`), `shared_kv` (the KVS line from the source unit's
     CY row through its sharers); the units with explicit origins (`UnitLayout::Explicit`) are the
     standard `Units` with a Gemma `UnitCopies`.
   - Gemma 12B: `head_groups` (the head fed by groups of CX rows, its own head input and token
     loop: `HEAD_GROUP` chains), row runs (attention / plain rows) as overrides of `Hc` / `ATT`.
2. **Declare the vocabulary** in the block's module: `pub const BAND_VERT: Class =
   Class::new("HV2");`, `pub const BUS: Slot = Slot::io("bus");` (pick names no other block uses),
   `pub const STORE: Kind = Kind::new("ple", "ple");`, notes as `NoteKey` with named value consts;
   colours in the block's `colours()`. A block that recolours a standard class (gpt-oss's `HE`)
   declares it again; the later block wins.
3. **Move the floor and wiring** into `Block::floor` / `Block::wire`. Place against the builder
   (`b.w` is the floor so far's width); set `b.head` (via `head_east` or directly) if the block is
   the head. Use the standard wiring functions (`units::unit_chains`, `units::handoff`,
   `head::head_rows`, `head::head_input`, `token_loop::token_loop`, `wiring::segments`) where they
   fit; leave notes for the facts a kernel needs (as `head::ROLE` and `token_loop::TURN` do) instead
   of recomputing them in the rcfg writer.
4. **Describe the kernels** in `kernels/<family>.rs`: one `params!` struct and one `Kernel` const
   a CSL source, the writer over `Pe` using `common::*` for the shared words. Shared CSL sources
   reuse their descriptor (`head::HEAD_CTL` is gpt-oss's and Gemma 12B's too: its writer takes the
   vocabulary offset along the control line, a row for gpt-oss's band). Values that differ by PE
   are `Block::overrides` (Gemma's shared-KV units' `SH`, their MLP width `Ic`, the pass PEs'
   `NTW`; Gemma 12B's row runs' `Hc` / `ATT`).
5. **Copies** go in each block's `bindings`, in the old placement order (the host's device
   protocol: compare `calls.log`).
6. **Delete** the family's `Whole` impl, its per-family `Geom` / geometry struct, its `RcfgAbi`
   leftovers; when the last one is ported, delete `block::{Whole, Legacy, legacy_kernels,
   legacy_overrides, standard_palette, standard_colour, standard_units}`.
7. **Verify** with the baseline harness (below): the family's configs must be identical to the
   reference capture.

## Weights

`Program::bindings` lists every copy, the blocks' in order: a symbol, a rectangle and a `Source`
(`Staged` leading-range blocks, `Slice` for any sub-array: column ranges, per-PE padding, a row
repeated down or across a rectangle, optional tables; `Rcfg`; `Inline`) with the `Transform`s the
prep applied. `weights::place` executes them on a `Device`; `tools/split/splitgeo.py` executes the
same `bindings.json` on the appliance runners. `stage_split` keeps a staged file under the 2 GiB a
transfer may take.

## Tools

- `split-geo <prep> <prefix> [k=v,..] --out <dir> [--what ..] [--levers ..]` writes `floor.json`,
  `rcfg.npy`, `kernels.json`, `layout/` (with `params.txt`, `geo.env`), `host.json`, `meta.json` and
  `bindings.json`; `--plan <plan.json>` builds from a plan instead of a meta; `--geometry
  <family>:<name>` prints a named geometry (what a prep needs before its meta exists).
- `tools/split/mkfull.py <out> <prep> <prefix> [overrides] [levers]` is the build directory of any
  family (the `mkfull_{gemma,g12,gptoss}.py` add their runner); `link.sh` links it and checks stack
  room; the `gotiny.sh` scripts run a tiny prep on the simulator.
- The preps (`prep_q35.py`, `gemma/prep_gemma.py`, `g12/prep_g12.py`, `gptoss/prep_gptoss.py`) keep
  the checkpoint maths and read their named geometry from `split-geo --geometry`.

## Verification

`tools/split/baseline` captures everything the build and the host produce for every shipping config
and each family's tiny (the host's placement, calls and derived values on a recording device; the
build's per-PE kernel and params, evaluated from `layout.csl` by `layout_eval.py`; the tiny preps),
and compares two captures. A change that must not change behaviour is checked against the capture
of the tree before it (README there).

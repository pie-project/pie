# Role-split program: architecture

The role-split build is layered. Each layer reads only the one before it, and only the last
layers know about Cerebras kernels.

```
Meta ──Family::plan──▶ Plan ──Family::floor──▶ Floor ──Family::wire──▶ Wiring
 (prep)               (logical)              (placement, roles)        (typed links)
                                                                          │ Palette (family)
                                                                          ▼
                    emit::{rcfg, params, layout, host, meta}  ◀──  Fabric (routes, queues,
                    weights::place (bindings → device)                   invariants)
```

| Layer | Module | What it holds | Knows about |
|---|---|---|---|
| Plan | `plan.rs` | `Shape` (the model), `Partition` (counts and split factors), `Contract` (positions, CAP, LOOP, WL) | nothing physical. This is the seam to poem's spatial compiler: a trace makes the `Shape`, a search makes the `Partition`. |
| Floor | `floor.rs`, `role.rs` | units (rects, CX / CY lines, layers), gap / relay slots, head block, store blocks; one `PeRole` a PE in a dense `Grid` | coordinates |
| Wiring | `wiring.rs` | `Chain` (one- or two-level reduction + cast), `Multicast`, `Path`, `Turn`, `Note`; logical colour `Class`es; the `Slot` each endpoint fills | coordinates, logical colours |
| Fabric | `fabric.rs` | `Palette` (class → colour), per-PE routes and `PeIo` (slot → endpoint), `parks` / `spare` / `route_words`, `validate` | physical colours, route words |
| Emit | `emit/` | rcfg words (`RcfgAbi`, per family), kernel params + `layout.csl`, typed `Meta`, `HostLayout` | kernel ABIs |
| Weights | `weights.rs` | `Binding`s (symbol, rect, source, transforms) and the one placer | the device |
| Program | `program.rs` | `Program { plan, floor, wiring, fabric, family }`, `Program::from_meta`, `Program::build` | all of the above |
| Shared kernel ABI | `family/abi.rs`, `family/tiles.rs` | the rcfg words every family's weight / CX / CY / head / head control kernels share (`Pe`, `weight`, `cx`, `cy`, `head`, `hctl`), and the copies the snake-unit families share (`UnitCopies`, `HeadCopies`) | the role-split kernels' ABI |
| Families | `family/` | `qwen`, `gptoss`, `gemma`, `gemma12`: the only model-specific code | everything; they compose the generic parts |

Rules:

- A family is data plus composition. No flag on a shared struct changes behaviour for one
  family. The generic code exposes primitives (`wiring::units`, `unit_chains`, `segments`,
  `head_rows`, `head_input`, `token_loop`, `FloorBuilder::units`, `FloorBuilder::head_east`), and
  a family that needs something different writes its own wiring from the primitives.
- A family's rcfg is the shared words (`family::abi`) plus its own: role words, flags and
  tails. Its colours are `Palette::base` plus its own classes; its kernels are
  `KernelBuild::ints` tables; its copies go through `weights::Copies` (and `family::tiles`).
- Every per-PE fact lives in a `Grid` (roles, routes, `PeIo`). There are no `HashMap<(u32, u32), tuple>` tables.
- Named structs replace positional tuples (`ChainMember`, `RootQueues`, `Endpoint`).
- Wiring is order-free. The fabric sorts routes, so adding links in a different order cannot
  change a word.
- Everything up to `Program` is pure: no device, no files. `split-geo` and the tests rely on this.
- Rust is the only geometry. Python tools call `split-geo` (`src/bin/split-geo.rs`).

## Invariants (enforced in `Fabric::lay` / `Fabric::validate` and `Plan::validate`)

These are the hardware lessons of the bring-up, made checks:

1. **Route clash.** Two words for one colour at one PE is an error.
2. **Parked queues.** An unused input queue never parks on a colour the PE sends on. Parks come
   first from colours with no route, then from pass-through colours (`Fabric::parks`). This is
   the 0704 hang.
3. **Segment queues.** A segment's queues are its own and never a reduction's on the same PE.
   This is 0733: a segment borrowed iq3.
4. **Control lines.** A segment never spans a control line (SWA / SWG / SWV / SWR divide the
   half-rows they live in).
5. **Path continuity.** Every `Path` runs from its sender's ramp to its receiver's ramp.
6. **Route slots.** Routes at a PE ≤ the kernel ABI's route slots (`RcfgAbi::route_slots`).
7. **Colour count.** Every colour < `COLOURS` (21).
8. **Kernel-side rules.** These are not geometry, so they stay with the kernel and link tools
   and are listed here so nobody drops them:
   - restore a route only after its oq drains (0738);
   - ≥768 B stack room a PE (`tools/split/chk_stack.py`);
   - payload directories end in `_ieee`.

## The generic layers

What the generic code does, for a family that composes it:

- **Plan.** `Shape::gdn` (Qwen3.5's GDN), `AttnSpec::global_head_dim` (Gemma's global layers),
  `MlpSpec::layer_widths` (per-layer MLP widths: Gemma's KV-sharing layers), `Partition::unit_grid`
  (a snake's slots). `RowRun::rows` is half-open; runs tile `0 .. rows`. `Plan::validate` checks: a non-empty
  model, `hidden_placed >= hidden`, `layers <= units * layers_per_unit`, `0 < kv_rows <=
  rows`, row runs (when given) contiguous and holding `hidden_placed` dims, nonzero segment
  widths dividing their half (`cols / 2` for scores / GDN / o-norm; with `attn_cols`, scores
  divide `attn_cols` and the router `cols - attn_cols`), head groups covering every row once,
  positions within `2^(rope_bits rope_levels)` when `rope_levels > 0`.
- **Floor.** `FloorBuilder::units(grid, extra)`: wafer = the units' box + `extra` (east, south);
  the box is `unx (cols + 1)` x `uny (rows + 1)` for a snake (gaps included), the farthest
  origin + a unit for explicit origins. `Snake { up_first }`: column 0 bottom-up, column 1
  top-down, ...; gap slots `Relay`. Generic roles: `Weight` / `Cx` / `Cy` with local row / col
  (control lines skipped), `part: Dense`, `Cx.first` / `last`; the corner `Idle`.
  `Unit::layers` is half-open. `CtlLine::Column(x)` / `Row(y)` hold wafer coordinates.
  `Floor::rects_by_kind` is greedy row-major (widest run, then rows down).
- **Wiring.** `units` = `unit_chains` (row / column chains, loop-backs, handoffs) + for a
  control-column head `head_rows` + `token_loop` when `contract.token_loop`. The primitives
  `unit_chains`, `head_rows`, `head_input` (Qwen's / Gemma's head input) and `loopback` are
  public for families that lay their own. `segments` lays every run, a one-PE run too. Slots `HeadRelay` and
  `HeadCast` are `head_input`'s, `Pass` binds nothing. Notes: `head_in` (1 / 2,
  `head_input`), `loop_turn` (bits 1 / 2 / 4, `token_loop`); the token loop's head-side ends
  bind nothing, unit 0's CX binds `EmbIn`.
- **Fabric.** `lay`: a chain is single-level (no `l2`, or `group` 0) or two-level, with the
  kernels' member role codes, colours and root halves; members `Endpoint::Member`, roots `Endpoint::Root` in the
  chain's slot; multicast source `Send`, receivers `Recv`; path `Send` at `send`, `Recv` at
  `recv`; a `Turn` lays its word only. Binding is order-free: an equal endpoint again is a
  no-op, a `Recv(a)` and a `Send(b)` in one slot make `Turn(a, b)`, anything else is an error.
  Notes with one key at one PE are OR-ed. Routes are sorted by colour; several classes may map
  to one colour. Consecutive link points must be adjacent. `Program::build`
  checks every PE's routes against the family's `route_slots` (`Fabric::check_route_slots`).
  `validate`: colours < 21; every bound receive colour reaches the ramp and every bound send
  colour leaves it; segment slots share no colour with reduction slots at a PE; unit-row,
  unit-column and head-row roots loop back on the loop-back colour.
- **Shared ABI** (`family/abi.rs`). A `Pe` is one PE's bindings, the family's palette and
  (gpt-oss's band) a colour it never parks on and counts as final. `weight` writes the route
  table, the row / column reductions' queues (parked where absent), the multicasts and sends, the
  reductions' and scores segment's roles, the family's two words, the segment's queues, the row,
  the relays' turns and a spare colour; `cx`, `cy`, `head` and `hctl` likewise, each taking what
  is its family's (`CxIo`: x in / out, embedding, flag bits, finals, tail; the CY's segment;
  `HeadIo`: the relay role, the token loop's turns). A family appends its own words after.
- **Emit.** `rcfg::dump` writes non-idle PEs x outer, y inner, kinds by `RoleKind::name`
  ("w", "wa", "wm", "cx", "cy", "cya", "cym", "head", "hctl", "relay", "ple").

## Kernel params

A kind's `KernelBuild::params` (the family's `kernels()`) hold for all its PEs; `emit::params::tile`
resolves one PE's list: the kind's, then what its role implies (`role_params`: a row run's `Hc` and
`ATT` 1 / 2 on `WeightPart::{AttnRow, PlainRow}` where the plan has several runs; `SH 1` and the
unit's MLP width a column `Ic` from its layers; `NTW 1` on store PEs outside every storage block,
Gemma's route-only pass PEs), then the family's notes (`U:<NAME>` for a whole unit, `P:<NAME>` for
one PE). Levers (`split-geo --levers`, `mkfull.py`'s last argument) override a param in every kind
that takes it; `<kind>.<NAME>` (the kind's `RoleKind::name`: `w.FIFOW:16`, E4B's weight FIFO) in
that kind only. A lever no kernel takes is an error.

`layout.csl` is written from the floor: each maximal rectangle of PEs with one kind and one resolved
param list is one loop nest, every param a literal, so `link.sh` passes no shape params (the layout's
`SPLIT_GEO` param marks such a build; `mod.rs::is_split` finds it in `out.json`). The per-kind
sources are the kernels', copied.

## Weights

`Family::bindings` lists every copy in placement order: a symbol, a rectangle and a `Source`
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
of the tree before it (README there). This refactor was checked that way against `a4c1f516d`, the
old per-family geometry (`geo.rs`, `geo/*.rs`, `place*.rs`, the Python `split_geo*.py`), which it
replaced.

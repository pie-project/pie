# The spatial engine: architecture

The spatial engine runs a model as a resident wafer program. The CSL
programs are hand-written and validated on CS-3 hardware, and the code
generator emits them byte for byte (`codegen` pins). Everything in Rust
around them is layered, each layer reading only the one below:

| layer | modules | what it is |
|---|---|---|
| model | `model.rs` | `ModelSpec` read off the poem trace; the structural check; a program's check against the model |
| plan | `plan.rs`, `shapes.rs`, `build.rs` | `SpatialPlan` (grid splits, placement, the head, a fitted memory model); the registry of hardware-validated shapes; the planner's parameters made into a `ProgramSpec` and compiled (`codegen`, `cslc`, the cache, `pie_manifest.json`) |
| program | `program.rs`, `component.rs` | `ProgramSpec`, the typed account of one built program, and its components' host halves and ABI |
| device | `device.rs`, `wafer.rs` | the `Device` trait (SDK runtime, test recorder) and the driver that loads a program on one and launches it |
| engine | `engine/` | PIE's `Engine`: batching lanes into launches, KV pages and the host tier, epilogues on the host or the wafer's VM |

## The program spec

`ProgramSpec` is the one source of truth for a built program. It is a
composition, not a record of optional fields:

- `Model`: hidden, query heads a kv head, vocabulary, the launch's `CAP`;
- `Layers`: f32 qwen2d grids (serve2d), bf16 grids in strips (serve8b),
  or grids and several-layer regions (serve8b_region);
- `Stage`: one batch a launch, or a ring of `MMAX` micro-batches;
- `Head`: tied (head2d over the embedding) or untied (head2d_v10 over
  `lm_head`, and the embedding strip), its rows, columns, logits lanes;
- `Kv`: one sequence, or a paged pool with copy kernels or a host copy,
  and staging slots;
- `Sampler`: the head's argmax, the eta VM's set A, or sets A and B (by
  turns or concurrent), with the VM image's ABI (`gring`, `vmlane` words).

`ProgramSpec::from_params(design, params)` makes one from the codegen's
`K:V` parameters and `to_params()` renders them back byte for byte: the
components give every value they hold, parameters no component reads
(kernel tuning) ride along as knobs, the order is kept. A value given in
another spelling than its component's (`RGY:0` for the default, a flag of
2) is read as the engine always read it and renders as given. The planner's build is a spec
(`build::program`), `build::compile` compiles a spec, the manifest stores
it, and the driver reads it (`manifest::read_program`); a directory
compiled outside PIE is read from `out.json` with its design inferred. The
driver never probes: where the PE images are readable, their `vmlane` and
`gring` sizes are checked against the spec's VM ABI.

## Components and the host↔wafer ABI

`component::Component` is what a part of the program is to the host: its
symbols, the checkpoint it needs, the route tables it writes before
`init`, the weights it places. The driver loads a program by walking
`ProgramSpec::components()` (layers, head, KV store, sampler). Each part
also owns its operations: `Kv` copies pages and runs the host tier,
`Stage` writes a launch's KV tables and checks a ring's launch, `Sampler`
writes a ring's per-lane VM control, `ProgramSpec::compute_args` builds
`compute`'s arguments. A component matches only on its own variants. The
design modules (`serve2d.rs`, `serve8b.rs`, `serve8b_region.rs`) are the
host halves of their CSL sources: tile layouts and route-table rules.

`component::abi` declares every symbol the host touches (a variable at
16 or 32 bits, or a function and its arity), by component.

## The safety net: host-protocol goldens

`golden.rs` runs every design's host side on the recording `Device` with
a synthetic checkpoint: seventeen scenarios (the build presets, the
module tests' parameters, serve8b_region with its VM block, programs
without `init` or `yscale`, the hardware Qwen3-8B builds) through load
(routes, `init`, placement, the VM), prefill, decode, page copy, page out
and in, and a VM load and launches (one lane of each set, then every
lane: DEFER's bit). Each phase pins its call count and the
sha256 of its calls (symbol, rectangle, elements, width, words). A second
test holds the ABI to both sides: every recorded call names a declared
symbol of the right kind, and every declared symbol is exported by the
design's generated layout. A third loads scenarios through `read_design`
from a program directory to the same pins, and program.rs ties the spec's
VM ABI (`vmlane`, `gring` words) to the generated sources.
`PIE_GOLDEN_DUMP=<dir>` writes the calls for a diff.

The pins change only when the programs do. Building serve8b_region_pipe's
route tables by the plain rule (without the head-bound pieces' `0x100`
flag, the drift that hung CS-3) changes its load pins.

## Extending it

- **A validated shape:** one `Ran` row in `shapes.rs` (or a `Bad` row for a
  known failure), with its reason. The planner and `build` read the table.
- **A component** (a model family's layers, a sampler, a KV store): a
  variant or type in `program.rs` with its params (`from_params` reads
  them, `param()` renders them), its `Component` impl and symbols in
  `component.rs`, its host half (placement, tables) in its design module,
  and a golden scenario. The round-trip and ABI tests cover it from then
  on.
- **A design:** its sources in `codegen` (`Design` and its pins), its
  parameter vocabulary in `ProgramSpec::from_params` (a composition of
  components), its parameters from the plan in `build::params`, its
  validated shapes in `shapes.rs`, and golden scenarios. The driver and
  the engine need no change unless it adds a component.

# The plan: poem trace → kernel family → compile parameters

`plan.rs` derives a wafer program's shape from the model, and `build.rs` makes
it a `ProgramSpec`, so a new model of a known shape needs no hand-written
parameters. The kernels stay hand-written CSL. The plan only picks the
kernels and sizes them.

## Flow

1. **The trace.** A load gets `LoadRequest::trace` (`poem_ir::Trace`): the
   model's `.poem` (for example `models/qwen3_dense.poem`) compiled for one
   deployment (`qwen3-8b-bf16-kv-bf16`). Tests get the same trace from
   `poem_compiler::catalog::repository().parse(name).trace(Platform::Cerebras)`.
2. **`ModelSpec::read(&trace)`.** It walks the trace's nodes once, layer by
   layer, and reads the model's facts from the ops themselves:

   | field | from |
   |---|---|
   | hidden | the `gather.embed` table's shape |
   | vocab | `gather.embed`'s `vocab` |
   | q_heads | `attn.plan_decode`/`plan_prefill` |
   | kv_heads, head_dim, sm_scale | `attn.prefill`/`decode` |
   | inter | `linear.glu`'s `intermediate` (SwiGLU) |
   | layers | the nodes' `layer` |
   | rope_theta, rope_table | `elemwise.rope` (`freqs`: llama3 or YaRN tables) |
   | rms_eps | `elemwise.rmsnorm` |
   | qk_norm | two `rmsnorm` over `Head(head_dim)` a layer |
   | tied | the logits matmul reads the embedding table |
   | weights, kv | the projections' param dtype, the KV cache row's dtype |

   **The structural check** accepts pre-RMSNorm layers (the first projection
   reads the first row norm, the MLP's projections read the second), causal
   full attention with GQA, full neox RoPE, q/k per-head norms or none, one
   SwiGLU MLP, two residual adds, every layer like layer 0, one final norm,
   and f32/bf16 weights and KV.

   It refuses anything else and names the op and the layer:
   - a mixture of experts;
   - gated-delta/SSM layers;
   - MLA;
   - sliding windows;
   - sinks;
   - LayerNorm;
   - biases;
   - soft-capping;
   - hyper-connections;
   - partial, interleaved or M-RoPE;
   - `1 + w` norms;
   - quantized projections or KV.

   **Cheap variants are recorded as `Need`s, not refused:**
   - `IdentityQkNorm`: no q/k norm (Llama). It needs a kernel flag that skips
     the two scales. Not built.
   - `RopeTable`: a stated frequency table. Placement only: write the table to
     `rot_c`/`rot_s` and fold YaRN's mscale into the q/k norm weights.
   - `LmHeadPlacement`: an untied head on serve2d. head2d supports
     `UNTIED`, but serve2d's placement reads `embed_tokens` for the logits
     (serve8b's head2d_v10 places `lm_head`; it serves untied models only).
3. **`SpatialPlan::derive(spec, WaferDims::CS3, Goals)`.** `Goals` gives the
   batch `B`, the batch the grid shape must hold (default 8), `PS`, and optional
   choices of `NB`, `MC`, `CAP`, the pipeline and the host page copy.
   - **Splits:**
     - `Dc = 8`, `Gs = head_dim / 8`, `PX = kv_heads · Gs`, `Qh = q/kv`;
     - `Ic = ⌈inter / PX⌉`, `Hc = ⌈hidden / PY⌉`, `HR = hidden`.
   - **Family:**
     - **serve2d** (or **serve2d_pipe** with `pipeline`), when `Qh = 2` and
       hidden and inter split over a square `PX × PX` f32 grid that fits a PE.
     - Otherwise **qwen2d_paged8b** (bf16 W/KV): rows are added in steps of
       16 until `grid_lanes` lanes with one pool block fit.
   - **Pool:** `NB` is the largest power of two up to 8 that fits. `MC` is 2
     when B is even in f32 builds, else 1. `E = 1`.
   - **serve2d placement:** every `GW × GH = layers` with `GW ≥ 2` and
     `GH ≥ 4` whose rectangle `GW·P + P + 1` × `GH·P` fits. The smallest
     rectangle wins.
   - **serve2d head:**
     - C = PY;
     - `Vr` is the smallest multiple of 32 with `HH = even ⌈vocab/Vr⌉` and
       `HH < GH·P`, the feeder row `F = HH/2` in `(2P−1, (GH−1)P)`;
     - `Gx`/`Gy` are the largest even divisors of C (≤16) and F (≤18).
   - **Tokens:** `CAP` is 64, halved until `B·CAP ≤ 512` (4 KB of token
     buffers). In a pipeline, `MMAX = min(8, 1024/(B·CAP))` and `HB` is the
     largest divisor of B up to 8.
   - **8B-style head** (head2d v10, bf16): `Hc = 32`, `C = hidden/32`, 256
     rows. The block count is the fewest even number whose tables fit 19.5 KB.
4. **Memory model.** `Grid::bytes` adds:
   - fitted code: serve2d 24,984 B; paged8b 21,404 B, less 922 B without the
     copy kernel;
   - the exact weight tiles, norm/RoPE tables and pool blocks;
   - a lane's buffers and the MLP chunk.

   It is fitted to the image ends of serve2d's B8/NB8 and B16/NB4 builds
   and qwen2d_paged8b's B1/NB4 build. It lands within ~0.1 KB of every
   other build measured, and it rejects exactly the builds that did not
   link. The link stays the final word.
5. **`model::check(trace, out_json)` / `SpatialPlan::check(out_json)` /
   `ModelSpec::check_program(get)`.** These are the load-time checks of a
   compiled program against the model, over the grid builds' parameters:
   - `HR`, `Dc·Gs`, `Qh`, vocab, the layers (`GW·GH`, serve8b's strips,
     serve8b_region's units);
   - enough kv-head columns, `PY·Hc ≥ hidden`, `PX·Ic ≥ inter`.

   A program without them (the strip programs, serve28*, no longer served)
   is refused.

   They also refuse what the placement can't do yet: an untied head, an eps
   other than the CSL's 1e-6, missing q/k norms, a stated RoPE table, or
   another softmax scale. The engine runs `ModelSpec::check_program` on
   every program it loads.

6. **`build::program(plan, ServeConfig)`.** The design for the plan's
   family (serve2d, serve2d_pipe with `pipeline`; serve8b for paged8b, or
   serve8b_region / serve8b_region_pipe with `region`), its components built
   from the plan and the config (serve8b's strip placement and
   serve8b_region's units are `build`'s own), the typed `Pins` of a test
   shape, the head's VM on the final geometry; its parameters rendered from
   the spec in the layout's order, checked against the layout and the
   registry of validated shapes (`shapes.rs`).

## What it gives today

| model | family | grid (PX × PY, Hc, Ic, Qh) | B / NB / MC | a PE (of 49,152) | full program |
|---|---|---|---|---|---|
| Qwen3-0.6B | serve2d | 128 × 128, 8, 24, 2 | 4 / 8 / 2 | 42,368 | 4 × 7 grids + 129 × 792 head (641 × 896), Vr 192, F 396: the hardware builds exactly |
| Qwen3-8B | paged8b | 128 × 160, 26, 96, 4 | 4 / 4 / 1 | 47,952 | not as serve2d (Qh 2 and square grids; 5 × 7 = 35 grids fit, 36 needed); `build` places it as serve8b (strips) or serve8b_region (units), the hardware builds exactly |
| Qwen3-14B | paged8b | 128 × 288, 18, 136, 5 | 4 / 4 / 1 | 47,664 | none: 40 grids, 20 fit |
| Qwen3-32B | paged8b | 128 × 512, 10, 200, 8 | 4 / 8 / 1 | 48,272 | none: 64 grids, 10 fit |
| Llama 3.1 8B | paged8b + IdentityQkNorm, RopeTable | 128 × 192, 22, 112, 4 | 4 / 4 / 1 | 47,216 | none: 32 grids, 30 fit |

The tests (`cargo test -p engine-cerebras plan`) reproduce the builds
validated on CS-3:
- the four batched serve2d builds and the four pipelines, key for key;
- the 8B layer builds, with NB pinned where their test matrix chose it:
  the auto rule gives NB 8 at B 1/2, NB 4 at B 4 and NB 1 at B 8, and the
  KVCP 0 build uses NB 3;
- the 8B head;
- `build`'s serve8b, serve8b_region and serve8b_region_pipe Qwen3-8B
  programs, key for key.

## Limits

- **Two families, five designs.**
  - serve2d places a whole model only with square f32 grids and Qh 2: in
    practice Qwen3-0.6B.
  - paged8b's designs (serve8b, serve8b_region) are sized for Qwen3-8B's
    36 layers on one wafer; larger models get a planned layer grid and head,
    and `blockers` say why no program places them.
  - A wider design (bf16 grids in a serve2d placement, layers across several
    passes, or more than one wafer) is a kernel/placement project, not a plan
    change.
- **The memory model is fitted, not computed from the ELF.** It is calibrated
  at the reference dims. At other dims, code size (Qh-dependent unrolling, if
  any) is assumed flat.
- **Policy constants are today's choices:**
  - NB a power of two ≤ 8;
  - MC 2;
  - CAP 64 / 16;
  - `Gx ≤ 16`, `Gy ≤ 18`;
  - the bf16 head's 256 rows and 32 dims a PE;
  - `PY` in steps of 16.

  Each is a named constant or a `Goals` field.
- **Facts the kernels hard-code** are checked, not parameterized: eps 1e-6,
  RoPE positions < 4096 (`Grid::max_context`), `Dc = 8`, `E = 1`. Making them
  CSL params is kernel work.
- **Dtypes:** the trace's bf16 weights become f32 in serve2d (the engine
  places f32 only) and stay bf16 in paged8b. Quantized deployments are
  refused, so plan the model's bf16 deployment.

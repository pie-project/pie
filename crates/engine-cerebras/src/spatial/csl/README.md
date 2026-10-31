# In-tree wafer program sources

The engine builds its wafer programs from these files (`spatial/codegen.rs`,
`spatial/build.rs`). Nothing here is compiled by cargo: the CSL is embedded
with `include_str!`, generated into a build directory and compiled with the
SDK's `cslc` at load time, cached by a hash of sources and parameters.

- `kernels/`: the shared per-PE kernels that the designs patch.
  - `qwen2d.csl`: the f32 decoder layer on a P × P grid.
  - `qwen2d_paged.csl`: the batched paged layer.
  - `qwen2d_pring.csl`: the paged layer as a pipeline-ring stage.
  - `head2d_v9.csl`: the 2-D LM head.
  - `head2d_v10.csl`: the bf16 8B head.
  - `qwen2d_paged8b_hr.csl`: the bf16 8B layer with the HROUTE (host-written
    collective routes), ROT8 and GY params: serve8b's and serve8b_region's
    grids.
  - `qwen2d_region.csl`: several bf16 layers on one region (x stays in
    place between them): serve8b_region's regions.
  - `softmax_s123.json`, `softmax_s13_06b.json`: the fast-softmax levers
    (S1: max-shift and rescale as one 2-D op; S2: the max and sum folds as a
    halving tree; S3: the exp sums ride the values all-reduce) as text
    patches on the 8B unit kernels and on the 0.6B paged layer.
- `vm/`: eta VM images, snapshots of the VM generator's output over the
  shared VM core.
  - `guest4hm_vm.csl`: serve2d's multi-column VM.
  - `guest4hz_vm.csl`: serve2d_pipe's VM: one or two resident sets (VMB
    columns and the `b_*` image), banked lanes, per-row control, per-step
    host input, op masks, per-lane programs, rows per turn (0: an idle
    lane), top-p by rounds, top-k by radix select (`vm_topk` 1, its default:
    a typical step from 3.23M to 214k cycles), the `vm_sumpipe` / `vm_l2w`
    options (serve2d_pipe's layout defaults them to 3 and 1: top-p's sum
    41% faster, typical's token 7.5%), and an output-queue empty-wait before
    init's recolors.
- `rt.csl`: every design's runtime route and queue-color helpers.
- `serve2d/`, `serve2d_pipe/`, `serve8b/`, `serve8b_region/`: each design's
  layout, routing helpers (`paths.csl`), `*_extra.csl` (what it appends to
  each kernel) and base programs. serve2d_pipe shares serve2d's
  `paths.csl`, `filler_base.csl` and `kvstage.csl` (the KV page staging,
  `KVST` slots a paged-grid PE; serve8b's is in its `grid_extra.csl`).
- serve2d_pipe's second VM set, as validated on CS-3 (1232 bytes free on
  set A's fullest VM PE, 928 on set B's; typical and tail-free sampling fit
  it together, 114 ops, and its bank holds tail-free's window, 552 words,
  or two typical lanes', 590). In the deployment's config:

  ```toml
  [cerebras.spatial.serve.vm_b]
  cols = 6
  max_ops = 136
  nreg = 6
  pool_words = 904
  bank_words = 640
  feat = 495
  opmask_lo = 1878015419
  opmask_hi = 256
  ring_w = 8
  ring_d = 16
  rows = 2
  lanes = 2
  ```
- `serve8b_region_pipe/`: serve8b_region as a pipeline ring (`PIPE`, up to
  `MMAX` micro-batches a lap): its layout, the units' `grid_extra.csl`
  (per-micro-batch tables `mtab`, the index word after x to the head), the
  head's `head_extra.csl` (virtual lanes, the micro-batch check `perr`) and
  the embedding strip; the rest is serve8b_region's. The plan offers only
  the shapes validated on hardware (B 4 x 4 at CAP 16, B 2 x 8 and B 1 x 12
  at CAP 32). The units drain output queue 2 before recoloring it.

The generated `*_pe.csl` are the programs validated on CS-3, byte for byte;
their headers still name the generator scripts they were first made with.
`codegen.rs` pins every generated file's hash: a change to a kernel or an
extra moves a pin, and a moved pin is a new program to validate on hardware
before repinning.

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
  - `softmax_s123.json`: Fork 11's p8fast softmax levers (S1-S3) as text
    patches on both 8B unit kernels (softmax_s123.py's REPS).
  - `qwen2d_region.csl`: several bf16 layers on one region (x stays in
    place between them): serve8b_region's regions.
- `vm/`: eta VM images, snapshots of the VM generator's output (`gen_vm.py`
  over the shared VM core).
  - `guest4hm_vm.csl`: serve2d's multi-column VM.
  - `guest4hz_vm.csl`: serve2d_pipe's VM, Fork 3's merged guest4hz
    generator (one or two resident sets: VMB columns and the `b_*` image),
    with banked lanes, per-row ctl, per-step host input, op masks,
    per-lane programs, rows per turn (0: an idle lane), top-p by rounds,
    top-k by radix select (the VM's `vm_topk` 1, its default: 0164, typical
    3.23M -> 214k cycles a step; the rounds top-k compiled out;
    `vm_lazyrng` stays 0), the `vm_sumpipe` / `vm_l2w` options (serve2d_pipe's
    layout defaults them to 3 and 1: 0168, top-p's sum -41%, typical's
    token -7.5%), and an output-queue empty-wait before init's recolors.
- `serve2d/`, `serve2d_pipe/`, `serve8b/`, `serve8b_region/`: each design's
  (`kvstage.csl`: serve2d*'s KV page staging, `KVST` slots a paged-grid PE;
  serve8b's is in its `grid_extra.csl`), layout, routing helpers,
  `*_extra.csl` (what it appends to each kernel) and base programs.
- `serve8b_region_pipe/`: serve8b_region as a pipeline ring (`PIPE`, up to
  `MMAX` micro-batches a lap): its layout, the units' `grid_extra.csl`
  (per-micro-batch tables `mtab`, the index word after x to the head), the
  head's `head_extra.csl` (virtual lanes, the micro-batch check `perr`) and
  the embedding strip; the rest is serve8b_region's. The plan offers only
  0167's hardware shapes (B 4 x 4 at CAP 16, B 2 x 8 and B 1 x 12 at
  CAP 32). The units drain output queue 2 before recoloring it.

The generated `*_pe.csl` are byte-identical to the bench payload's `sync.py`
outputs. `codegen.rs` pins their hashes; with `PIE_CEREBRAS_PAYLOAD` set, its
test diffs them against a payload directory. To update: change a kernel or an
extra here (from the payload), run the test, and repin.

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
- `vm/`: eta VM images, snapshots of the VM generator's output (`gen_vm.py`
  over the shared VM core).
  - `guest4hm_vm.csl`: serve2d's multi-column VM.
  - `guest4hs_vm.csl`: serve2d_pipe's banked VM: per-row ctl, per-step host
    input, op masks, multi-row (Fork 3's guest4hs; snapshot of 2026-10-29
    14:26).
- `serve2d/`, `serve2d_pipe/`: each design's layout, routing helpers,
  `*_extra.csl` (what it appends to each kernel) and base programs.

The generated `*_pe.csl` are byte-identical to the bench payload's `sync.py`
outputs. `codegen.rs` pins their hashes; with `PIE_CEREBRAS_PAYLOAD` set, its
test diffs them against a payload directory. To update: change a kernel or an
extra here (from the payload), run the test, and repin.

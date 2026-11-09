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
  - `qwen35_gdn.csl`: Qwen3.5's Gated DeltaNet layer and its MLP on a
    P x P f32 grid (P = k_dim), the recurrent state's row k on grid row k,
    the conv window and state a lane slot's (`reset`).
  - `qwen35_attn.csl`: qwen2d_paged with `Qh` query heads a kv head, each
    gated by sigmoid of its gate projection; a partial RoPE is the
    placement's (unturned pairs turn by zero).
  - Both units match numpy on the fabric simulator (16 x 16 and 32 x 32
    grids: several steps, a slot reset, two passes a launch; the attention
    unit through the paged scenario) and link at Qwen3.5-0.8B's dims
    (P 128: the GDN unit 30.1 KB a PE at B 1, 39.1 KB at B 8; the
    attention unit 34.2 KB at B 1 NB 8, 41.4 KB at B 8). No design places
    them yet: no codegen pin.
  - `gemma4_layer.csl`: a Gemma 4 dense layer on qwen35_attn's grid without
    the gates: v normed per head, the attention's and the MLP's outputs
    normed before their residual adds, a GELU-tanh GLU, the per-layer
    embedding (`PC` dims a column of the layer's table for `PR` vocab rows
    a row, the context projection of `x0`), the layer's scalar, and the
    shared tail's k and v in (`OWN` 0, `kvi`) or out (`SHARE` 1, `kvo`).
    Matches numpy on the fabric simulator at P 16 (sliding, global with a
    partial RoPE, and a shared-tail layer; two fires of two lanes).
    serve_g4's grids (`codegen::serve_g4`).
  - `gemma4_fast.csl`: gemma4_layer with the 8B speed levers ported, chosen
    by the layout's `G4F` bits (serve_g4 with fast grids).
  - `gptoss_attn.csl`: gpt-oss's attention unit (sinks, biases, the sliding
    window, YaRN's RoPE tables from the host): serve_gptoss's attention
    units.
  - `moe_mxfp4.csl`: a routed mixture of experts (gpt-oss's; NE experts,
    top TK, biases, clamped SwiGLU) in place of a layer kernel's dense MLP,
    appended to qwen2d_paged8b_hr-style kernels (their buffers and
    collectives). Every expert resident and sliced like the dense MLP, so a
    token's TK experts run on every PE. MXFP4 codes three a u16, decoded by a
    shift and a mask into f32 high halves (fp4 x 2^-126; the 0.5 code is a
    denormal, which CS-3's FMAC keeps: request 0440), inputs prescaled by
    2^MXK, E8M0 scales a byte an output a 32-block run. Matches numpy on the
    fabric simulator (16 x 12, Hc 10 / HR 115, Ic 7, NE 20, B 2; 16 x 16,
    NE 8, B 3, MC 3: rel 5e-7). serve_gptoss's MoE units
    (`codegen::gptoss_moe`).
  - `qwen35w_gdn.csl`, `qwen35w_attn.csl`: Qwen3.5's two units with bf16
    weights (`W16` 1): serve_q35w, the planner's build past 0.8B.
  - `qwen2d_region.csl`: several bf16 layers on one region (x stays in
    place between them): serve8b_region's regions.
  - `softmax_s123.json`, `softmax_s13_06b.json`: the fast-softmax levers
    (S1: max-shift and rescale as one 2-D op; S2: the max and sum folds as a
    halving tree; S3: the exp sums ride the values all-reduce) as text
    patches on the 8B unit kernels and on the 0.6B paged layer.
  - The 8B unit levers, each a list of (old, new) text patches on the paged8b
    and region kernels, applied by `codegen::Variant` for the knob that
    names it (the planner states them; `Variant`'s fields say which knob
    and what each cuts): `speed_ops*.json` (`SPD`), `lse_ops.json` (`LSE`),
    `hv_ops.json` (HV, `PS` 1), `ops2.json` (`OP2`), `arf_ops.json`,
    `arx_ops.json`, `arfs_ops.json` (`ARF` bits 1, 2, 4), `f16acc_ops.json`
    (`F16` 2, 3), `mem_ops.json`, `mem_qkn.json` (`MEM`), `op3_ops.json`
    (`OP3`), `trm_*.json` (`TRM`'s bits; `trm_slw.json`: `SLW`), and
    `head_pack.json`, `embed_pack.json` (`TRM` 16384, the head and strip).
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
- `serve_g4/`, `serve_gptoss/`, `serve_q35w/`: those designs' layouts,
  routing helpers and base programs, as for the designs above.
- `split/`: the role-split program's PE kinds: Qwen3.5's (`weight_q35.csl`,
  `ctl_cx.csl`, `ctl_cy_q35.csl`, `head_pe.csl`, `head_ctl.csl`), Gemma 4's
  (`weight_gemma.csl`, `ctl_cx_gemma.csl`, `ctl_cy_gemma.csl`, `ple_gemma.csl`,
  `head_ctl_gemma.csl`), gpt-oss's (`weight_gptoss_att.csl`,
  `weight_gptoss_moe.csl`, `ctl_cx_gptoss.csl`, `ctl_cy_gptoss_att.csl`,
  `ctl_cy_gptoss_moe.csl`, `head_pe_gptoss.csl`), and Gemma 4 12B's
  (`weight_g12.csl`, `ctl_cx_g12.csl`, `ctl_cy_g12.csl`, `head_g12.csl`). Not generated by codegen:
  the build tools under `tools/split/` write their layouts, and
  `spatial::split` places and drives the programs. Design, build and params:
  `spatial/split/README.md`.

The generated `*_pe.csl` are the programs validated on CS-3, byte for byte;
their headers still name the generator scripts they were first made with.
`codegen.rs` pins every generated file's hash: a change to a kernel or an
extra moves a pin, and a moved pin is a new program to validate on hardware
before repinning.

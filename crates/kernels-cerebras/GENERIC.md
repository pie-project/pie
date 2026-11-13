# The generic wafer compiler: design

Status: phase 0 (baseline and design). Nothing here is built yet; ARCHITECTURE.md is what runs.

**Goal.** A `.poem` model the compiler has not seen runs on the wafer with no hand code. The
primary acceptance target is `models/flux_2.poem`: a text encoder, a DiT denoiser run over whole
sequences for several steps, and a VAE, in one poem. LLM decode (the ten shipping programs) is one
point of the same compiler, and it keeps its speed. That takes two capabilities the crate does not
have:

- **(A) Placement from the graph.** Today the compiler picks one of four hand layouts
  (`partition/{snake,stacked,runs,split}.rs`, `compose::Layout`) and tunes about 10 keys (PX, PY,
  K, UNX/UNY, NE, HR, VR, G2, G2W, G3W, ...). The general compiler derives the PE regions, routes
  and colours from the layer graph and searches them with the cost model. The four layouts become
  points of that space.
- **(B) Op lowering.** A new op (an activation, a norm, an attention variant, a router score) is
  lowered into a small set of primitive op modules. Today it is refused by name
  (`spatial::layer::lower`, `target::shape`, `compose::refuse`).

Constraints: 16-bit weights only (MXFP4 kept for gpt-oss). On the ten shipping models the generic
plan is at most +3% chip ms/step over the hand-tuned one, and Pie stays at or above 97% of bare.
The engine (`engine-cerebras`) stays model-agnostic, and the structure stays clean.

## 0. Baseline: where every catalog model stops today

`cargo run --release -p kernels-cerebras --example survey` tries each catalog model's bf16
deployment through every stage in turn. With `=<name>`, it tries that deployment exactly. With
`DEEP=1`, it runs the plan and the search too. With `READINGS=1 ... =<name>`, it counts the ops each
value of the `reading` switch runs. The run below is from 7758ac39b. The stages are:

1. **trace**: the poem builds.
2. **lower**: `spatial::layer::lower` gives every running node a phase.
3. **shape**: `target::shape` reads the shape.
4. **refuse**: `compose::refuse` checks what the kernels run.
5. **plan**: a pinned plan, or the search, finds a partition that builds.

Two terms used in the tables:

- **"Missing ops"**: every op that runs on the wafer (one token) with no phase, counted. The
  refusal message names only the first.
- **"Unfit"**: the model's 16-bit weights exceed the wafer's SRAM. That is about 44 GB: 762 x 1172
  PEs at 48 KB each, about 40 KB of which can hold weights. No layout fixes that, so these models
  are out of scope under the 16-bit rule.

### 0.1 Text decoder LLMs

| poem | deployment tried | stops at | cause (first refusal; all missing ops) |
|---|---|---|---|
| qwen_3 (Qwen3.5 dense) | qwen35-d{0.8b,4b,9b}, d3b, tiny, qwen38-27b(-mini) | ok / plan | 0.8b/4b/9b ship (snake); d3b and tiny plan by the search (from `4b`); 27B unfit; **27b-mini: the search finds no partition** (A) |
| qwen3_dense | qwen3-{0.6b,8b,14b,32b} | ok / plan | 8B ships; 0.6B plans by the search; **14B (28 GB, fits the SRAM) finds no partition**: the snake's unit grid cannot grow past the pins' (A); 32B unfit |
| llama_3 | llama31-8b, llama33-70b(-mini-l4) | ok / plan | 8B ships. 70B unfit; mini-l4 plans by the search |
| gemma_4 dense | e2b, e4b, 12b; 31b(-mini-l30); e4b-mini-l{1,6,24,30,36} | ok / plan | e2b/e4b/12b ship. 31B unfit. **The e4b minis do not plan**: the stacked arrangement is made for one unit shape and the search cannot move it (a layout gap, A) |
| gemma_4 MoE | gemma4-26b-a4b(-mini-e64) | shape | `linear.moe_route` with a per-expert scale/bias (router kernel: plain softmax over the top k only) |
| gpt_oss | gptoss-20b-bf16-mxfp4, -mini | ok / plan | 20b ships (split); **the mini finds no partition** (A) |
| qwen3_moe | qwen3-30b-a3b(-mini), 235b, coder-480b | plan | mini plans by the search; the full models are unfit |
| qwen_3 MoE (Qwen3.5/3.6/3.8 MoE, Qwen3-Next, ornith) | qwen35-122b-a10b-mini, qwen36-35b-a3b-mini, qwen3-next-80b-a3b-mini, qwen38-max-mini, ornith15-397b-mini, qwen35-a3b | lower | `linear.moe_sigmoid_gate_add`: the shared expert's sigmoid-gated add |
| glm_4 | glm45/46/47(-air)(-mini) | shape | `moe_route` scored `Sigmoid{renormalize, scaling}` with a pick bias (DeepSeek-V3-style router) |
| minimax_m2 | minimax-m25-mini (bf16 parsed; ships e4m3) | shape | sigmoid router with bias, as GLM 4 |
| minimax_m3 | minimax-m3(-mini), m31-flash(-mini) | lower | `attn.index_kv_append`, `attn.index_topk`: a lightning-indexer sparse attention (DSA) |
| mimo_v2 | mimo26-flash-l6, -pro-mini (bf16-mxfp4) | lower | `gather.split_qkv_padded`: q / k / v of unequal head dims (padded v) |
| muse_glimmer | muse-glimmer-30b(-mini-l8,l24) | lower | `elemwise.gate_sigmoid_mul` reads the tiles after its mixer: a gate placed after an attention that is not its own. A phase-assignment gap, no missing op |
| muse_spark | muse-spark-open(-mini-l4) | lower | `elemwise.norm_gate` |
| inkling | inkling(-small)(-mini-l7-e8) | lower | `ssm.short_conv`, `linear.rel_bias` (a short causal conv mixer; a relative position bias) |
| nemotron_h | nemotron3-nano / ultra (-mini) | lower | `ssm.mamba2` (an SSM scan), `linear.relu2` (squared-ReLU MLP, no GLU) |
| kimi_k3 | kimik3-mini, -l4 (bf16-mxfp4) | lower | `ssm.kda` (Kimi delta attention), `attn.mla_*` (MLA: latents, absorb q / out, decode, kv_append), `elemwise.res_blend`, `elemwise.rmsnorm_gated_by` |
| dots3_note | dots3-note-mini | lower | MLA (`attn.mla_*`) + DSA indexer (`attn.index_layernorm_rope`, `index_rope`, `index_kv_append`, `index_topk`) |
| glm_5 | glm5-a12b, glm53-mini | lower | MLA + DSA indexer (as dots3) |
| glm_5_next | glm53-flash-mini | lower | hyper-connections (`elemwise.hc_expand/fold/gates/project/rmsnorm_f32`), MLA + DSA, KDA, pooled attention (`attn.pool_*`) |
| hy_4 | hy4-mini, -preview | lower | hyper-connections + MLA + DSA |
| deepseek_v4 | dsv4-flash-mini, -base, -pro-mini | lower | hyper-connections; pooled/compressed attention (`attn.pool_*`, `merge_lse`, `pool_lse`); grouped routes (`linear.group_routes`, `matmul_grouped`); DSA top-k |
| qwen_4 | qwen38-flash-next(-mini) | lower | `elemwise.ple_gate`, `gather.ngram_ids`, hyper-connection inject/mix, pooled + indexed attention **outside the layer frame** (the layer analysis gives these nodes no layer) |

### 0.2 Non-LLM poems: image/video diffusion, speech

Here the token-decode reading of the layer program does not apply at all. These poems run whole
sequences: no single-token decode, no token embedding in the denoiser, several readings in one
poem. Op counts per reading (`READINGS=1`):

| poem | readings (nodes) | stops at | what the wafer would need |
|---|---|---|---|
| **flux_2** (klein-4b; flux2-mini = the DiT alone) | text 412 (Qwen3-4B prefill, 3 taps); **denoise 598** (mini 149); vae.decode 138; vae.encode 103 | lower: `attn.prefill` / "no token embedding" | denoise (mini): `linear.matmul` 33, `linear.glu` 6, `attn.ragged` 4 (non-causal joint attention over a group CSR), `elemwise.{layernorm 11, modulate 11, rmsnorm 12 (per head), rope_axes 12 (4-axis), residual_add 12, act 4, binary 3, add_bias 1, sinusoid 2}`, `gather.{pack_rows 12, unpack_rows 4, split_qkv 6, split_rows 11, concat_cols 1, rows 1}`, `conv.{patchify, unpatchify, cells}`. VAE: `conv.conv3d` 36, `conv.group_norm` 30, `conv.upsample` 3, `pixel_(un)shuffle`, `attn.voxel` (mid-block attention), `elemwise.standardize` |
| z_image | denoise 216 (mini) | lower | as flux, plus `clamp`, `gather.group_rows` |
| wan_2 | denoise 121 (mini-nano) | lower | as flux, plus `elemwise.relative_bucket_bias` (5B); video VAE (`conv.conv3d` 132, `cache_store`, `avg_down`) |
| ltx_2 | denoise 557 (mini) | lower | as flux, plus `gate_sigmoid_mul`, `clamp`; audio + video streams |
| minimax_h3 | denoise 244 (mini) | lower | as flux, plus `gather.group_first` / `group_rows` |
| hunyuan_image_3 | denoise 70 (mini) (an AR-MoE LLM with image readings) | lower | `elemwise.rope_axes`, `elemwise.binary` in a layer; `attn.prefill`; image.in/out conv readings |
| qwen3_tts | talker 462, predictor 89, codec.prenet 122, codec.decode 159 | lower | a multi-reading pipeline: `linear.matmul`'s input on no axis (a sub-model's input); codec: `elemwise.snake`, `conv.conv3d`, `pixel_shuffle` |
| qwen3_asr | text 454 (decode), audio.encode 1,734, audio.conv 12 | ok (text only) | the text decoder plans by the search. The audio encoder (`attn.voxel`, `conv.conv3d`) runs before the token embedding and is skipped today |

### 0.3 What the table says

1. Among LLMs, the dense GQA decoder family plans. Everything else stops at **one of about 15
   named ops**. They fall into five groups:
   - elementwise/root ops: `relu2`, `norm_gate`, `res_blend`, `rmsnorm_gated_by`,
     `ple_gate`, `rel_bias`, `moe_sigmoid_gate_add`, the router scoring forms;
   - attention variants: MLA, unequal-dim q / k / v, the DSA indexer, pooled attention;
   - recurrent mixers: Mamba2, KDA, short conv;
   - hyper-connections;
   - one phase-assignment gap (muse_glimmer).
2. The layouts are themselves a refusal source. Models that fit the SRAM but were not what a hand
   layout was made for fail the search outright: qwen3-14b, qwen38-27b-mini, gptoss-20b-mini and
   the five e4b minis. The others (0.6B, d3b, the 70b and MoE minis) land on a neighbour's
   layout.
3. Every diffusion/speech poem fails before any op question. The layer program reads a trace only
   as one token decoding through layers. Those poems need a **sequence regime**: many rows a step,
   non-causal attention without a cache, several readings, convolutions.

## 1. The workload regimes

The generic compiler is generic over *how rows flow*, not only over ops:

| regime | rows a step | weights | attention | example |
|---|---|---|---|---|
| **decode** | 1 token a request, many steps | stationary, K layers time-share a unit (memory-bound) | over a KV cache on the tiles | every shipping LLM |
| **stream** | T rows (hundreds to thousands) a step | stationary, rows streamed through the layer's unit, one unit a layer (or R replicas a layer) | over the step's own K / V (non-causal joint, or causal prefill), held on the tiles for the step | flux denoise, an encoder's prefill, the VAE's attention block |
| **grid** | a spatial grid of pixels/voxels | stationary conv kernels | none, or one mid-block attention (stream) | VAE, audio codec |

A **reading** of the poem (its `reading` switch value) is one program entry. It has one regime. It
also has a row bucket (the T it is built for, as decode has its positions). A model is the set of
readings it serves: LLM = {decode}; flux = {text: stream (causal prefill), denoise: stream x steps,
vae.decode: grid, vae.encode: grid}. Readings are placed **co-resident** when their weights fit
together, and in turn otherwise:

- klein-4b is about 8 GB of DiT plus 8 GB of Qwen3-4B plus about 0.2 GB of VAE, against about
  44 GB of SRAM, so it can be co-resident.
- The host invokes a reading by streaming its inputs into that reading's region. This is the
  artifact's entry table. The engine still reads only the artifact, so it stays model-agnostic.

**Why stream is fast on this machine.** In decode a weight PE multiplies its tile once a token. In
stream it multiplies it T times, with the next row's input arriving while the current one
multiplies. A step therefore costs about T x (elements a PE) x 0.46 cycles, plus the pipeline fill
across layers:

- flux klein at 1024² is T = 4,096 image + 512 text rows. Its DiT (about 4B params) spread over
  about 200k PEs holds about 20k elements a PE: about 4.6k x 20k x 0.46 = 42 M cycles, about 48 ms
  a step.
- Replicating each layer R times (the wafer has room for about 4 copies of the DiT) splits the
  rows. That gives about 12 ms a step plus attention.
- The search's new axis is that trade: replicas vs. pipeline depth vs. attention placement. In
  decode the axis was units vs. K.

## 2. (A) Placement: the layer graph to PE regions

### 2.1 The IR

Three levels. The middle one is new; the outer two exist.

```
LayerProgram (poem-compiler spatial::layer, per reading)          -- what runs
   phases: Cast | Matvec | Root(Region) | Mixer(Primitive) | Segment | SideIn | Barrier
      │  (each phase carries its widths, params, row class and the region / primitive it lowers to)
      ▼
Floorplan (kernels-cerebras, new: floor/plan IR the search moves over)  -- where it runs
   slots:   a grid of unit slots (with relay slots for gaps)
   units:   UnitShape { col_runs: [ColRun{n, role: Attn|Moe|Plain|Kv}],
                        row_runs: [RowRun{n, dims, role: Attn|Plain|Kv}],
                        control: {cx: West|Mid(after col run i), cy: South|North},
                        layers: [LayerId] (K; decode), replicas (stream) }
   path:    the units' visit order over the slots (the x handoff route)
   aux:     Head{at: East|Band|Groups(g), HR, VR, HC} | Store{rows, tokens} | KvLine | Relay
   flows:   per LayerProgram edge, the link kind it takes (row/col chain, multicast,
            segment all-reduce, handoff, KV ring, token stream) and its reduction groups (G2, G3)
      │
      ▼
Floor + Wiring + Palette + Fabric (today's physical build, unchanged)     -- how it is laid
```

A `Floorplan` is the general `Partition`. The current `Partition` already has most of its fields:
`row_dims` runs, `attn_cols`, `arrangement`, `head.groups`. They are just only legal in the
combinations the four layouts make.

### 2.2 The four layouts as points

| layout | col runs | row runs | path | aux | layer program feature that asked for it |
|---|---|---|---|---|---|
| snake | [Plain PX] | [Plain PY, Hc] | boustrophedon over UNX x UNY, ≤1 gap → Relay | Head East + token loop | own-cache attention, GDN |
| split | [Attn AX, CX, Moe MX] | [PY] | two-column snake starting upward | Head Band + BandLoop | routed experts |
| runs | [Plain PX] | [Attn RA Hc_A, Plain PY−RA Hc_W] | U of two unit columns | Head Groups | sliding windows (ring = rows) |
| stacked | [Plain PX] | [PY] (KV on PYW) | explicit origins (`Arrangement`) | Store above each column, KvLine | side input, shared KV |

The rule for the generic build: **each feature the layer program has asks for a structural
element. The element is no longer a layout.**

- Routed experts ask for a Moe col run and a router segment.
- A window asks for a Kv row run sized to the window.
- A side input asks for a Store aux block.
- A shared cache asks for a KvLine.

Each element is chosen independently. A model with experts *and* windows (gemma4-26b-a4b) gets
both, where today it must pick one layout. `compose::blocks` reads the Floorplan's elements, not
`Layout`. `kernels/{snake,split,stacked,runs}.rs` merge into role kernels whose layout-specific
parts become comptime params read off the elements:

- the head's feed: East / Band / Groups;
- the token loop's bus;
- the row role (`AROW`, already a per-PE override).

### 2.3 Dataflow edges to the fabric

Every LayerProgram edge between phases becomes one typed link. The link kinds already exist in
`wiring.rs`:

| edge | link |
|---|---|
| Matvec → roots | Chain (two-level, G2; three-level, G3) |
| roots → tiles | Multicast |
| unit → unit | handoff Path |
| segment | `wiring::segments` |
| stream regime: rows into a unit | token stream Path (new, one colour, the handoff's shape) |
| stream regime: K / V for attention | KV ring along the attention rows (new) |

Colours stop being block consts. Today each block owns fixed colours and the palette has 21. The
generic build allocates them as an **interference-graph colouring** of link footprints:

- Two links conflict when their routes share a PE and a direction.
- Disjoint units reuse colours freely.
- A *preferred colour* per class keeps each shipping program's palette byte-identical: the
  colouring takes the preferred colour when it is free.

The queue rules (one flow a queue; parks only on unrouted colours; segment ≠ reduce queues) stay
checks in `Fabric::validate`.

### 2.4 The search

The search moves over the Floorplan's axes, seeded from the pins:

- per-unit PX/PY and the col/row run split;
- K (decode) or replicas (stream);
- the path family (snake, U, two-column snake, explicit);
- head placement and HR/VR;
- the reduction groups;
- segment widths.

**The cost model is op-derived.** Today `cost::time` adds a per-model measured layer term
(`MEASURED`), and a new model takes the median. The generic model prices each phase from what it
does:

- Matvec: the elements of its busiest PE x 0.46 cycles.
- Chain: `chain_cycles`.
- Multicast: hops.
- Root region: its op count x the chunk width, at the CY's measured rate per op class (rsqrt,
  exp, FMA).
- Mixer: the primitive's own formula (the KV elements, the state elements).
- Stream: x T / replicas, plus fill.

`MEASURED` is kept only as a held-out calibration check. **Acceptance for A:**

1. On every pin's shape, with `made_for` seeding off, the search returns the pin's point, or one it
   predicts ≤ +0.5%.
2. The pin built through the Floorplan is byte-identical: the baseline harness capture (`floor`,
   `rcfg`, `layout.csl`, `bindings`, `host`) of every shipping config and every tiny config.

## 3. (B) Op lowering: new ops into a primitive set

### 3.1 The primitive set

A new op lowers into these primitives. Each is one comptime-parameterised CSL module under
`programs/wafer/ops`. A model never adds a module; a *new mathematical structure* does (§3.4).

| primitive | runs on | exists as | generalised by |
|---|---|---|---|
| P1 tile matvec | weight PEs | `mv`, `wpe` | stream form: the matvec over a row stream, double-buffered input (new, about 50 lines on `mv`) |
| P2 chain reduction (sum, max, (value, id)) | rows / cols to roots | `red`, `argmax` | a token-axis reduction (sum over a stream's rows: group norm, a VAE's statistics) |
| P3 segment all-reduce | a run of roots or tiles | `seg` | — |
| P4 **root region** | CX / CY roots, a chunk of a vector | hand ops in `act`, `math`, `cy_*` | **a CSL printer of kernels-ir `Region`s** (§3.2) |
| P5 rotation tables | CY | `rope` | per-axis positions (`rope_axes`: 4 axes, interleaved), per-layer rope specs |
| P6 attention over a cache (decode) | tiles: KV rows | `att`, `swa`, `kv` | dk ≠ dv (MLA's latent: one KV head, 576 / 512); sinks, windows, shared caches already exist |
| P7 attention over a step's rows (stream) | tiles: K / V of the step | — (new) | v0: P6 run once a query over the step's K / V, stored as decode stores its cache (row p % PY). v1: query blocks (flash tiling) |
| P8 gated linear recurrence | tiles: a state slice | `gdn`, `cy_gdn` | comptime {decay: scalar \| vector-per-key, delta rule: on \| off, input gate}: GDN = (scalar, on), KDA = (vector, on), Mamba2 = (scalar, off) |
| P9 short conv / line buffer | CY (1-D), a grid region (2-D/3-D) | GDN's conv1d ring in `cy_gdn` | kernel size, dims, stride: `ssm.short_conv`, the VAE's `conv3d` (as a line buffer feeding P1-stream: im2col in the stream order) |
| P10 top-k select | roots | `cy_moe` (top-k over experts) | grouped routes (`group_routes`); over positions (DSA indexer: top-k of a distributed score vector, a P2 (value, id) reduction run k times or a k-wide merge) |
| P11 stream permutations | host driver / stream order | the host's x runs | `pack_rows`, `unpack_rows`, `patchify`, `unpatchify`, `cells`, `pixel_(un)shuffle`, `upsample`, `split_rows`, `concat_cols`: no compute, only the order rows enter and leave a region in |

### 3.2 The lowering pass

The compiler already has the substrate, for CUDA. `poem_compiler::fuse::chain` lowers chains of
row ops into `kernels_ir::Region`: primitive unary, binary and reduce ops over one row, with
explicit rounding points. `kernels_ir::interp` is the oracle every printed kernel is held to. It
covers `act`, `add_bias`, `binary`, `clamp`, `gate_sigmoid_mul`, `layernorm`, `modulate`,
`res_blend`, `residual_add`, `rmsnorm`, `rope` and `softcap`. The wafer reuses it:

1. **Phase assignment** (poem-compiler, `spatial::layer`). It is unchanged for matvecs and
   mixers. A `Root` phase carries its ops' `Region` (built by `fuse::chain`'s lowering) besides
   their names. An op `chain` cannot lower and that is no mixer is refused by name, as today. But
   the set is `chain`'s, not a wafer list, so a new elementwise op added for CUDA arrives here too.
2. **Root region at a root** (kernels-cerebras, new `emit/region.rs`). The region is split by
   where its inputs and outputs live:
   - A row reduction (rmsnorm's sum of squares, layernorm's mean and variance) over a vector
     chunked across a unit's roots becomes a P3 segment all-reduce between two passes. This is the
     split the CY's rms already makes by hand.
   - The rest is printed as straight CSL over the PE's chunk, one DSD op per region op where the
     DSD set has it (fadd, fmul, fmac, fmax) and `math` scalars elsewhere (exp, rsqrt, recip,
     tanh).
3. **Mixers** match the trace op to a primitive with params. Each mapping below is a table entry,
   not a kernel:
   - `ssm.mamba2`, `ssm.kda` → P8 with their decay/delta form;
   - `attn.mla_decode` → P6 with dk ≠ dv, plus the absorb matvecs as P1;
   - `attn.ragged` → P7.
4. **Fast paths keep shipping images byte-identical.** Every hand op of `act`, `cy_att`,
   `cy_gdn`, ... is registered as the region it computes. When a phase's region equals a
   registered one (a structural equality of regions), the emitter calls the hand op. When it does
   not, it prints. The shipping models therefore print nothing new until a measurement says the
   printed form is as fast.

### 3.3 Reference evaluation

There are three oracles, at increasing scope:

- **Op**: `kernels_ir::interp` runs each region on the host on the same chunk the sim's root PE
  saw (dumped through `tdbg`/a debug symbol). It must agree within the op's bf16/f16 rounding.
- **Model**: `engine-cpu::Forward` (the trace's host forward; today it runs a plain MoE trunk) is
  extended to every op a reading runs (stream attention, `rope_axes`, `modulate`, `sinusoid`,
  `conv`). It runs the trace itself, so it needs no model code. Sim outputs are compared to it:
  - LLM: greedy tokens equal, and teacher-forced logits argmax equal.
  - Stream: relative L2 ≤ 1e-2 on the velocity a step, cosine ≥ 0.999.
- **End to end**: the HF/diffusers goldens the tiny checkpoints already use (`mk_tiny_*.py`). A
  `mk_tiny_flux2.py` builds flux2-mini's config in diffusers' `Flux2Transformer2DModel` and saves
  the velocity of one step and the latents of a 4-step run.

### 3.4 What genuinely needs new CSL

Ranked by models unlocked per line of CSL:

| new CSL | unlocks | size |
|---|---|---|
| P1 stream matvec + token stream path | every stream reading | small |
| P4 region printer | every root op in §0 (relu2, norm_gate, res_blend, rmsnorm_gated_by, ple_gate, moe_sigmoid_gate_add, sigmoid routers' scores, layernorm, modulate, sinusoid, snake) | medium (the printer, not per-op code) |
| P7 stream attention | flux / z_image / wan / ltx / h3 denoise, encoder prefill, VAE attention | v0 small (P6 looped), v1 medium |
| P8 generalised recurrence | nemotron_h (Mamba2), kimi_k3 / glm_5_next (KDA) | medium (a rework of `gdn` / `cy_gdn`, GDN byte-identical) |
| P9 line buffer conv | inkling short conv; every VAE / codec | medium |
| P10 top-k over positions + sparse gather attention | minimax_m3, dots3, glm_5, hy_4, deepseek_v4 (DSA) | large |
| pooled / compressed attention (`attn.pool_*`, `merge_lse`) | deepseek_v4, glm_5_next | large; deferred |
| hyper-connections | deepseek_v4, glm_5_next, hy_4, qwen_4 | P4 + small P1 on a residual 4x wide: a CX-memory question more than a primitive |

MLA is **not** a new primitive: absorbed decode is one-KV-head attention over the latent, plus
matvecs.

## 4. Milestones

Every milestone lands on its own branch off poem-cerebras. Shipping models are checked
byte-identical with the baseline harness (`tools/wafer/baseline`, capture vs. capture) at every
milestone. Sims run one at a time (`scratch/genD/serial.sh`).

Estimates are focused engineering time, sim-validated. Chip validation is a separate step on each
milestone's tail.

| # | milestone | acceptance | estimate |
|---|---|---|---|
| M0 | Baseline: survey tool, failure table, this design | the tables in §0 reproduce | done |
| **M1** | **flux2-mini DiT, one denoise step, on the fabric sim** (stream regime v0) | sim velocity vs. `Forward` and diffusers: rel L2 ≤ 1e-2, cosine ≥ 0.999, at T = 64 text + 64 image rows; shipping captures byte-identical | 6-8 weeks |
| M2 | flux2 full pipeline: 4-step loop, text encoder reading (causal stream prefill over Qwen3), VAE decode / encode (grid regime: P9 line buffer conv, token-axis group norm, P7 voxel attention); readings co-resident; klein-4b builds and links for the system fabric | flux2-mini 4-step latents vs. diffusers; a tiny full-pipeline config (te + DiT + VAE minis) image PSNR ≥ 40 dB vs. diffusers; klein-4b memory model fits | 8-10 weeks |
| M3 | Decode layouts into the Floorplan (A, structure): the four partitions become constructors of the IR; `compose::blocks` and the role kernels read elements, not `Layout`; colour allocation by interference colouring with preferred colours | every shipping and tiny capture byte-identical; the e4b minis and gemma4-26b-a4b (experts + windows) plan | 3 weeks |
| M4 | Op-derived cost model + Floorplan search (A, search) | for each of the 10 pins, made-for seeding off: the search returns the pin or ≤ +0.5% predicted; `MEASURED` held out within its current error. Then chip: ≤ +3% ms/step on the 10 | 3-4 weeks (+ chip) |
| M5 | Decode root ops via regions (B for LLMs), hand ops as fast paths | shipping byte-identical; tiny sims of a synthetic Qwen3 with relu² MLP, with norm_gate, with a sigmoid-scored router (GLM-style mini), with a shared-expert gate (Qwen3.5-MoE mini): tokens == reference; muse_glimmer's phase gap fixed in `lower` | 4 weeks |
| M6 | New mixers: P8 (Mamba2, KDA), P9 short conv, MLA via P6 dk ≠ dv, unequal-dim qkv | nemotron3-nano-mini, inkling-mini, kimik3-l4 / glm53-mini-without-DSA, mimo minis: tiny sims, tokens == reference | 6-8 weeks |
| M7 | DSA indexer, pooled attention, hyper-connections | minimax_m3, dots3, glm_5, hy_4, deepseek_v4 minis on sim | 8-12 weeks |

**M1 in detail** (the first to start):

- **M1a, layer program for a stream reading** (poem-compiler `spatial::layer`):
  - `lower_reading(trace, reading = "denoise", rows = T)`: the same phases, with a row count.
  - **Row classes**: flux's `.on(text)` / `.on(~text)` streams take their own weights, and so
    their own units.
  - A **Barrier** phase before attention: all rows' K / V are in before any query reads them.
  - Lane-vectors computed once a step (the timestep embedding, the modulations) form a small
    decode-regime prologue.
  - Gathers and conv layout ops become P11 permutations.
  - Test: flux2-mini's denoise lowers with no refusal. Klein's denoise lowers to the same body
    count.
- **M1b, reference.** `engine-cpu::Forward` runs flux2-mini's denoise (`attn.ragged`, `rope_axes`,
  `layernorm`, `modulate`, `sinusoid`, P11 ops). `mk_tiny_flux2.py` gives the diffusers golden.
  `Forward` must match diffusers before the wafer is compared to either.
- **M1c, primitives**:
  - P1-stream: `mv` over a row ring in the weight PE.
  - P7 v0: queries streamed over K / V stored by row, reusing `att`'s dim-major keys and
    log-sum-exp.
  - P4 printer, for layernorm / modulate / rmsnorm-per-head / gated residual / SwiGLU.
  - Each primitive gets a standalone CSL test layout on the sim, with `kernels_ir::interp` as its
    oracle.
- **M1d, Floorplan v0 for stream.** One unit a layer side (2 double blocks x {txt, img} + 2 single
  = 6 units). A row stream path through them, with the K / V region per block. This is written
  directly in the Floorplan IR: the stream regime never has a hand layout, so the IR exists before
  M3 ports decode onto it.
- **M1e, artifact entries.** The artifact states readings and their entry streams (rows in, rows
  out, per-step lane inputs). `wafer-bench` gets a `--reading` mode that feeds a golden's inputs
  and compares outputs. The engine change is generic: it reads entries from the artifact.

## 5. Risks

- **Numerics: f16.** The decode kernels hold IEEE f16 tiles and f16 activations, with f16
  accumulation a PE. DiT activations (adaLN-modulated, large residuals) can pass f16's 65,504 range.
  Stream programs need bf16 activations with f32 accumulation (`MV 0` exists: bf16 tiles, f32
  accumulation). That costs about 2x matvec cycles. Measure in M1 before committing the
  replication budget.
- **Colours and queues.** There are 21 colours and a few queues a PE. A stream program adds:
  - a row stream a unit;
  - a K / V ring an attention block;
  - token-axis reductions;
  - co-resident readings, each with its own links.

  Interference colouring reuses colours across disjoint regions, but a unit where row stream,
  chains, multicast, KV ring and handoff all cross may run out. The colouring reports the clique
  that does not fit, and the search treats that as unfit.
- **PE memory.** A printed region adds code to the CY image. The bring-up's lesson was 22 KB of
  48 KB lost to code. Fast paths keep shipping images unchanged, and the memory model
  (`cost::memory`) must learn the printer's code size per region op, or new models will fail at
  the link.
- **K / V for stream attention.** At klein's T = 4,608 and dim 3,072, one block's K / V is 56 MB,
  about 1,400 PEs of SRAM, per block in flight. Pipelining blocks multiplies that. The search
  must trade pipeline depth for KV memory.
- **Compile time.** A whole-wafer cslc link is minutes today. Co-resident readings and per-layer
  units (stream) raise the distinct (kind, params) rectangles in `layout.csl`. Keep kinds few, and
  per-PE differences in rcfg/overrides, not in kinds.
- **Sim time.** The fabric sim is slow. Stream readings must be validated at tiny T (64 + 64 rows)
  and tiny dims (flux2-mini), with the cost model carrying the extrapolation. Never two sims at
  once.
- **Byte identity vs. generality.** M3/M5 refactor code the ten shipping programs depend on. The
  baseline harness capture is the gate on every commit. A shipping regression found on chip costs
  a session.
- **Search cost.** The search builds `Program`s and takes 1-40 s a model today. The Floorplan space
  is larger. Prune with the cost model before building, and cap builds per search.
- **Scope.** §0's long tail (DSA, pooled attention, hyper-connections) is research-grade CSL. The
  plan front-loads the north star (flux) and the cheap wins (P4: about 12 LLM families' root ops),
  and leaves M7 open-ended.

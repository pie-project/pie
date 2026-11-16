# The generic wafer compiler: design

Status: M1a, M1b, M1c (the stream primitives on the simulator) and M1d (flux2-mini's denoise step
compiled by the generic compiler, no hand layout, on the simulator) done (§4.1), M2a (klein-4B's
stream program builds for the system fabric at 1024², §4.1) compile side done, M2b (it links:
1,065 ELFs; 1.52 s a step predicted, §4.1), M1f / M1g (flux2-mini's denoise loop re-launched on
the simulator, driven through the engine by a guest, its static arrays from a checkpoint, §4.1)
M1h (flux2-mini's text encoder on the simulator through the engine, compiled on load; `pie run
text-to-media` making an image on the host fabric, §4.1), M1i (the VAE decode on the simulator
by the grid regime; the readings co-resident; `pie run text-to-media` fully on the simulator,
48.3 dB from diffusers, §4.1), M2c (klein's text reading at 512 tokens links: grouped-query
attention and three taps on the row stream, a GQA miniature on the simulator; grid programs
priced, the VAE's passes in weight groups; klein end to end 51.8 s predicted, the VAE decode
its bottleneck, §4.1), M2d (the tiled regime: a VAE's image over the fabric; flux2-mini's decode on
the simulator in 0.9M cycles, 81x fewer, 50.1 dB from diffusers; klein's 1024² decode links,
0.13 s priced; klein end to end 6.5 s, §4.1), M1j (flux2-mini on the CS-3: the denoise loop
at 1.34M cycles a step, the simulator's numerics; `pie run text-to-media` on the system, 48.4 dB
from diffusers, §4.1), M1k (the PE clock is 750 MHz; a step's wall split; a reading's first
launch no longer waits on its static arrays, §4.1), M2e (the rotation tables a level at a time,
the K / V columns folded over two columns a head, the cost model held to the simulator unit by
unit: a row's round trip through a unit; klein's step 1,326M -> 1,081M cycles, 1.77 s -> 1.44 s,
end to end 7.54 s -> 6.21 s at 750 MHz; the roofline 10.9M cycles a step, §4.1), K1 (klein on
the CS-3 from its real checkpoint: a program over more than one band hangs at its first launch,
flux2-mini at a narrow fabric reproduces it, §4.1), M2f (the GEMM regime: a whole-sequence
reading's ops over an array of PEs, rows and channels both cut, weights from tapes below it;
flux2-mini's step on the simulator in 0.97M cycles, its 2-step loop 1.48e-3 from diffusers;
klein's 1024² step 1,081M -> 115M cycles priced (1.44 s -> 0.15 s), links; end to end 6.21 s
-> 1.05 s, §4.1) and M3 (the
decode layouts as points of the Floorplan, §2.5). ARCHITECTURE.md is what runs for decode.

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
| gemma_4 dense | e2b, e4b, 12b; 31b(-mini-l30); e4b-mini-l{1,6,24,30,36} | ok / plan | e2b/e4b/12b ship. 31B unfit. The e4b minis did not plan (the stacked arrangement was made for one unit shape and the search could not move it); since M3 they plan by the search on the derived arrangement |
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

### 2.5 What M3 built

`src/floorplan/` holds the IR; ARCHITECTURE.md ("The floorplan") walks it.

- **Elements, each asked for on its own.** `Requests::of(needs)`: attention over a cache asks for
  `Segment::Scores` (on the attention row runs' attention columns), GDN for `Segment::Gdn` +
  `Segment::Onorm`, routed experts for a `ColRole::Moe` column run + `Segment::Router`, a window
  for KV rows, a side input for `Aux::Store`, a shared cache for `Aux::KvLine`.
  `Floorplan::of(plan, requests)` reads each element's choice off the plan: the `UnitShape`
  (column runs, row runs, control lines, K, KV rows), the path (`floor::UnitLayout`: the stacked
  arrangement's origins, two columns from a band, the unit slots' snake, else two columns down
  then up), the `Head` (`East`, `Groups`, `Band`). `compose::blocks` maps elements to blocks in one
  order and names no layout: one `Units` block (kinds by half where a MoE column run halves it,
  row-run overrides where rows come in runs), one `Scores` block (the split's and the row runs'
  scores are it), the head and token loop by the head element.
- **The four layouts are constructors.** `floorplan::{snake, stacked, runs, split}` read a preset's
  keys into a plan (the old partitions, error texts kept: the search's repair follows them).
  `floorplan::plan` picks the one the shape's **kernel set** is keyed for.
- **Kernel sets, not layouts.** What `Layout` decided for the kernels is `kernels::KernelSet`
  (`Own`, `Shared`, `Windows`, `Experts`: chosen by the mixers, the CSL ABI each set's params,
  symbols, copies and cut are written for). `kernels::of` takes the units' kernels from the set and
  the rest from elements: the head's by `Head::at` (the token bus on where there is storage), the
  store kernel by `Aux::Store`, the relay by gap slots; `Schedule::hg` reads the head element.
  Merging the four sets' params into one per role (the design's "role kernels read elements") is
  left for M5: it changes every `layout.csl`'s literals, so it needs the fast-path parity M5
  measures, not byte identity.
- **Colours by interference colouring.** `Palette::allocate`: each class's footprint is read off a
  lay with every class on its own colour; classes meet where both route at a PE (one pass-through
  word excepted); a class keeps its block's preferred colour where no class it meets kept it, the
  rest take the lowest free colour, the most constrained first; a class meeting all 21 is refused
  by name. Every shipping and tiny program keeps every preferred colour (a test checks it).
- **Derived arrangement.** `Arrangement::derive` places stacked columns for any unit shape: column
  A from the shared caches' source unit up (the KV line runs north) and at least half the units,
  column B level with it, storage of 19,000 words a PE above each column. It reproduces the pinned
  E2B / E4B columns but for column A's top row. The five e4b minis now plan by the search.
- **Byte identity.** Every shipping and tiny config's `plan.json`, `layout.csl`, kernel sources,
  `kernels.json`, rcfg, floor, host and bindings (the baseline capture), and the search's plans
  for the catalog models it plans, are unchanged.

Not reached: gemma4-26b-a4b (experts + windows) still stops at `shape` (the router's per-expert
scale) and then at `refuse` (the experts' kernels run no sandwich norms, GELU or final softcap):
kernel and lowering gaps (M5), not placement. Its floorplan (a MoE column run, a band, KV rows for
its windows) is expressible as gpt-oss's is.

M1d plugged in the stream regime as `floorplan::stream` (§4.1): its own elements (`StreamUnit`,
`KvColumn`, `LaneBus`), `Requests::{stream, joint, lanes}`, a composer (`Stream::wiring`) and the
same colouring and laying. What M2 adds on it: a stream path (a `UnitLayout` visited by a row stream, one unit a layer, and
a token-stream link in `wiring`), replicas of a `UnitShape`, an `Aux` K / V region per attention
block and a grid region for convolutions, a reading with no head (`Floorplan::head` made optional), and their
classes coloured by `Palette::allocate` with no preferred colours. Each is a new element, a block
the composer maps it to, and its `Requests` field; nothing in the composer's other arms changes.

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
- **M1c, primitives** (§4.1 lists what the M1a program asks of them):
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

### 4.1 M1a / M1b / M1c: where they stand

- **M1a.** `spatial::layer::stream::lower(trace, analysis, classes)` reads a stream reading into the
  decode program's phases, a `RowClass` (a request word, its rows) at a time: classes whose bodies
  (so weights) are equal share a `Side` of a `Stage` (entry, each layer, between two runs of layers,
  exit): one unit. `attn.ragged` is `Mixer::Joint`: a store phase (`gather.pack_rows` x3), a
  `Phase::Barrier`, the attend phase (`attn.ragged`, `gather.unpack_rows`). The lane vectors are the
  program's `lanes` body (once a step); the host's ops are `Host { Rows | Table }` (`conv.patchify` /
  `unpatchify` in and out, `conv.cells` and its split a table). Decode lowering is unchanged (the
  stream arms are behind the reader's stream flag; plan + build of the 0.8B, gpt-oss-20b and tiny
  t08 byte-identical). `kernels_cerebras::stream::shape` reads the params; the survey prints a
  stream line (`STREAM-SHAPE-OK` for flux2-mini and klein).
  - flux2-mini at 64 + 64 rows: 9 units (entry 2, double 2 + 2, single 1 + 1, exit 1), 3 layer
    bodies; klein: 32 units, the same 3 bodies. The double body (a class each) is
    `Rr2 Cr Mr1 Rc4 Cc X3 B X2 Mc1 Rr3 Cr Mr1 Rc1 Cc Mc1 Rr1`; the single body (both classes)
    `Rr2 Cr Mr1 Rc4 Cc X3 B X2 Mc1 Rc1 Cc Mc1 Rr2`; the exit `Rr2 Cr Mr1`.
- **M1b.** `engine-cpu::Forward::stream` runs a step of lanes, each its own word; the wafer's trace
  of flux2-mini's denoise against diffusers' `Flux2Transformer2DModel` (`tools/wafer/flux/
  mk_tiny_flux2.py`, f32 over bf16 weights; the engine-cpu test `flux_2_minis_denoise_answers_
  diffusers`): rel L2 5.4e-3, cosine 0.999985 (trunk bf16 as the trace types it); 5.1e-6 with every
  activation f32.
- **What M1c needs**, read off the program:
  - P1-stream: the sides' matvecs over a row ring, bf16 tiles and f32 accumulation (`MV 0`): per
    double side qkv 256 -> 768, out 256 -> 256, linear_in 256 -> 1536, linear_out 768 -> 256; single
    in_proj 256 -> 2304, out_attn, out_mlp 768 -> 256; entry context 192 -> 256, x_embed 128 -> 256;
    exit proj_out 256 -> 128. A matvec whose input is a host row (`Cr` with no root op before it:
    the entry) reads the token stream path straight.
  - P7 v0: `Mixer::Joint` over the step's 128 rows (one group, non-causal, 2 heads of 128, scale
    0.08838835): the store phase lands each row's k / v (a row's 2 x 128 bf16 each, 64 KB a block a
    step) by row; the barrier is the last row's store; the attend phase is P6's dim-major
    dot + log-sum-exp per query. Both classes' rows land in one store (the double blocks' two units
    share one K / V region).
  - P4 region printer for exactly these root ops: `layernorm` without affine (eps 1e-6; mean and
    variance are row reductions over the unit's row roots: P3 between passes), `modulate`
    scale-shift and the gated `residual_add` (their scale / shift / gate a lane vector resident at
    the roots), `rmsnorm` per 128-wide head with a gain (col roots), `rope_axes` (4 axes of 32,
    interleaved, theta 2000; P5 tables from the host's cell table, per row), SwiGLU `glu` (col
    roots), `residual_add`.
  - The lane prologue: a decode-regime body (`I1 Cr Mr1 Rc1 Cc Mc1`, twice, `Rr2 Cr Mr4`: the
    timestep and guidance embedders, then mod_txt, mod_img, mod_single, norm_out) run once a step,
    and a new link: each lane vector multicast to the roots of every unit that reads it.
  - Host: `conv.cells` (the positions' table), `conv.patchify` in, `conv.unpatchify` out, the row
    permutation (identity for one group); `wafer-bench --reading` (M1e) feeds them.
  - Oracles: `kernels_ir::interp` per region; `Forward::stream` for the model, which needs a hook
    that dumps a node's value (an op-level capture) for the per-primitive sim comparisons.
- **M1c.** Each primitive runs on the fabric simulator on its own (`tools/wafer/prims`: `sim.sh`
  compiles a prim's layout with `programs/wafer/ops` and runs its `run.py`, one simulator at a
  time machine-wide). Its inputs and oracles are the host forward's own values of flux2-mini's step
  (`Forward::stream_capture`; engine-cpu's `flux_2_minis_primitive_goldens` writes them) and
  `kernels_ir::interp`. The stream modules sit in `programs/wafer/ops` but are listed apart
  (`sources::STREAM_OPS`): no decode build writes or keys them. Plan and build of the 0.8B,
  gpt-oss-20b and tiny t08 stay `diff -r` identical. They reuse mv (`tile32`), red (`send`), seg
  (`all_reduce`), rope (`tables`), math and fabric as they are. The decode modules' text is
  unchanged: a build copies it verbatim, so a decode module generalised in place is a byte-identity
  question for the gate.
  - **P1-stream** (`ops/smv.csl`): a weight PE's tile over the step's rows. Each row is double
    buffered: its x chunk in, `tile32` (16-bit operands, f32 accumulation), and an async chain send
    that overlaps the next row's multiply. `prims/mv` has a column of row sources (CX), PX x PY
    weight PEs and a row of chain roots (CY). Results on flux2-mini's matmuls (64 or 128 rows):

    | matmul | PEs | elements / PE | rel L2 vs held product | vs trace | cycles / row | cycles / element |
    |---|---|---|---|---|---|---|
    | qkv 256 -> 768 | 8 x 4 | 6,144 | 6.4e-8 | 1.7e-3 | 3,346 | 0.545 |
    | out 256 -> 256 | 4 x 4 | 4,096 | 6.1e-8 | — | 2,296 | 0.561 |
    | linear_in 256 -> 1536 | 16 x 4 | 6,144 | 6.4e-8 | 1.7e-3 | 3,346 | 0.545 |
    | linear_out 768 -> 256 | 4 x 8 | 6,144 | 1.0e-7 | 1.6e-3 | 3,387 | 0.551 |
    | single in_proj 256 -> 2304 | 24 x 4 | 6,144 | 6.3e-8 | 1.7e-3 | 3,346 | 0.545 |
    | entry context 192 -> 256 | 4 x 4 | 3,072 | 5.6e-8 | 1.7e-3 | 1,757 | 0.572 |
    | entry x_embed 128 -> 256 | 4 x 2 | 4,096 | 5.1e-8 | 1.7e-3 | 2,297 | 0.561 |
    | exit proj_out 256 -> 128 | 2 x 4 | 4,096 | 6.4e-8 | 1.6e-3 | 2,296 | 0.561 |

    - "Held product" is the rows as the PEs hold them, in f64. "Trace" is the host forward's
      bf16-rounded output; its distance is that rounding.
    - IEEE f16 (qkv) gives the same error and cycles. f32 accumulation does not cost the 2x §5
      feared: 0.55 cycles an element, against decode's 0.46.
  - **Activation type: IEEE f16 matvec inputs, f32 accumulation, the residual f32 at the roots.**
    - flux2-mini's values are small (max |x| 11.6). Its trace's activations are bf16, so f16
      holds them exactly; the mini cannot decide.
    - FLUX.2-klein-4B's real checkpoint (diffusers, 3 prompts, 4 steps at 512² and 1024², hooks
      on all 109 Linears; `scratch/cleanup/fluxm1c/klein`) decides:
      - No Linear input or output passes 65,504. The nearest is `context_embedder`'s input (Qwen3
        hidden states): 16,392, a 4x margin; clamp before its narrowing.
      - Rounding every Linear input, the velocity's rel L2 from f32 is 7.2e-3 mean (1.2e-2 max)
        in bf16 and 9.9e-4 (2.0e-3) in f16. Over 4 free-running steps the final latent is
        2.4-4.8e-2 in bf16 and 0.55-1.4e-2 in f16.
      - The bf16 weights cast to f16 lose nothing that matters: rel L2 at most 6.6e-8.
      - What breaks f16 is the text residual stream: 73k-103k from double block 4 on, in a few
        channels. Stored f16 it is inf from step 1; clamped it is 0.5-4.7%.
    - So the residual and the norms' inputs stay f32 at the roots (as the regions hold them),
      and the build is IEEE: the decode kernels' format, so co-resident readings share one.
    - P7 in an IEEE build, f16 probabilities: rel L2 1.2e-4 against f64, against 9.9e-4 in bf16.
  - **P7 v0** (`ops/satt.csl`, `ops/cy_satt.csl`): a column of attention PEs a kv head (a whole
    head of 128), row r keeping the step's rows p = r (mod PYA). The root multicasts each row's
    [k | v], then each query. A query's scores are a `tile32` over every entry, then exp against
    a carried log-sum-exp reference, then the values matvec with a ones column. [o | l] goes up
    the column chain; the root returns [redo, ln l].
    - Results on flux2-mini's first double block (128 rows, 2 heads, 8 PEs a column, 16 entries a
      PE): rel L2 1.2e-4 against f64 (IEEE build), 1.7e-3 against the trace's bf16 output.
    - No redo pass: the carried reference held across the text and image rows.
    - 3,521 cycles a query and 4,613 a stored row. The store is a scalar loop: vectorise it.
  - **P4** (`emit::region`, §3.2's printer). poem-compiler's `fuse::chain::lower_run` gives a run of
    named ops its region. It also lowers `rope_axes` (kernels-ir `Ops::rotate_axes`, its positions
    scalar params) and the packed `glu`; no chain forms over either. The printer classes values
    static (host tables, evaluated by interp), row, scalar or head. It prints DSD ops and `math`
    scalars. A row sum across roots is a P3 all-reduce (seg.csl). cos / sin of a position x
    frequency come from P5 tables (rope.csl). It refuses by name what it does not print.
    Results in `prims/reg` (bf16 build, 64 image rows; error against interp, cycles a row of
    `row()`):

    | region (flux2-mini nodes) | roots x chunk | vs interp | cycles |
    |---|---|---|---|
    | layernorm (no affine) + modulate scale-shift (41, 42) | 4 x 64, 2 all-reduces | 2.8e-5 (bf16 ties) | 3,044 |
    | per-head rmsnorm with gain + rope_axes 4 x 32, theta 2000 (45, 46) | 2 x 128 | 3.0e-5 (7e-9 vs trace) | 11,791 |
    | gated residual + layernorm + modulate (57, 64, 65) | 4 x 64 | 8e-11 | 3,281 |
    | SwiGLU 1536 -> 768 (67) | 6 x 128 | 2.1e-5 | 4,847 |
    | gated residual (69) | 4 x 64 | exact | 248 |
    | residual add + gated residual (124, 125) | 4 x 64 | exact | 332 |

    The cycles are v0's. Two costs dominate:
    - The chunk sums are scalar loops: make them a halving tree.
    - P5's digit products cost 64 slots x 4 levels of scalar math a row, about 10k cycles: half
      the pairs' angles repeat across rows of one image row or column, so the tables could be a
      row's position lookup.
- **What M1d needs** (Floorplan v0 for stream, on §2.5's IR):
  - The unit:
    - CX roots run the Rr regions and source the row multicast; P1-stream weight PEs; CY roots
      run the Rc regions. This is `prims/mv`'s geometry, with the region modules' I/O wired to
      the fabric in place of the harness's `load_row` / `store_row`.
    - A token stream path between units.
    - The chain colours alternate a row, as the harness's do.
  - The segment all-reduce of a row's norms across a unit's CX roots: seg.csl as it is, its rcfg
    words binding static colours in both modes.
  - A K / V column a kv head under each double block's pair of units and the single block's.
    The CY roots' Rc output (q / k / v, a head a root) is P7's multicast, and both classes' rows
    land in one store. The barrier is the stream's order.
  - The lane prologue: the decode-regime body, plus the multicast of each lane vector to the
    roots that hold it resident (`ParamAt::Resident`).
  - The host:
    - Each root's static tables (`Printed::tables`: masks, frequencies' rotations, slot ranges).
    - The cell table's positions as each row's scalars.
    - The P11 permutations and the clamp of the entry's context.
  - The links:
    - Each stream link (row stream, chains, segment all-reduce, P7 multicast / chain) is a
      Requests field, a composer block and a line in `Floorplan::of` (§2.5).
    - Its colours come from the allocation.
    - The build lists `STREAM_OPS` with `OPS` for stream programs only.
  - The cost model: 0.55 cycles an element a row for P1-stream, and the region and P7 cycles
    above until they are vectorised.
- **M1d.** `wafer-compile stream --model flux2-mini-bf16-kv-bf16 [--stages N] [--values <capture>]`
  compiles the denoise reading with no hand layout (`stream::compile`); `tools/wafer/stream/sim.sh`
  runs it on the simulator (`run.py`: the host's arrays, one launch, the rows read back against the
  step's own values). The pieces, each generic over the stream program:
  - **Units** (`stream::unit`). A side of a stage (and the lane prologue) is lowered node by node
    from its trace nodes (the stream program now records them: `StageProgram::nodes`,
    `StreamProgram::lane_nodes`): a row in (the stream, or the host's), each run of elementwise
    nodes one root region (`lower_run`, split where a run does not lower as one), each matmul a
    P1-stream matvec from the side its input is on, `attn.ragged` a store and an attend, the
    values read outside a row out. Splits, packs and unpacks are columns of another value. The
    barrier cuts the steps: what reads nothing the attention makes runs before it (the single
    block's MLP moves ahead of its attention), and a value made before and read after is held a
    row at its root. A matmul over a host row takes the direction that lands its output where the
    stream wants it (the entries' run CY to CX). A lane op over host inputs only (the timestep's
    sinusoid) is the host's.
  - **Layout** (`stream::place`). `py = hidden / 16` CX roots everywhere; CY roots: a head root a
    kv head (its q, k, v and its attention output, so it alone talks to its K / V column), then
    the fewest rest roots whose every tile fits 12,288 words and held rows 6,144: each region's
    chunk whole at one root (glu's gate and up halves together), every other column evenly. Tiles
    follow: a Mr tile its root's output slices, a Mc tile its root's input slices. flux2-mini:
    lane 8 roots, entries 2, double and single 2 + 4, exit 2.
  - **Schedule** (`stream::place::schedule`). The path's units in order; each unit's CX roots run
    the runs of rows as they arrive: its own rows' early steps, others' rows on (or dropped where
    no later unit runs them), its own rows' late steps. A K / V column stores rows in the order
    they reach its block's first unit and serves queries unit by unit, so no unit waits on a store
    another can only send after it (the order is computed, not chosen).
  - **Floorplan** (`floorplan::stream`, written in §2.5's IR). `Requests` gained `stream`, `joint`,
    `lanes`. `Stream::of` lays a `StreamUnit` (a `UnitShape`: Attn head columns, Plain columns, one
    row run) a unit left to right, a `KvColumn` a kv head under each attending block's first unit
    (root `(x + 1 + h, py + 2 + h)`, `PYA = 8` PEs below it), the `LaneBus`. `Stream::wiring` is the
    composer: a unit's row chains (+ the row cast) and column chains (+ the column cast), its CX
    segment where a row region all-reduces (seg.csl as it is, its rcfg words static), the row
    stream's paths between consecutive units, each K / V column's chain and a path each way to each
    unit's head root, the lane gather chain into the lane root and the bus (a new `Link::Broadcast`
    tree). `Palette::allocate` colours them all, no preferred colours (every class its own; colours
    are reused across disjoint units); `Fabric::lay` / `validate` as for decode.
  - **Emitter** (`emit::stream`). `layout.csl` states every PE's code, params (queue colours from the
    laid fabric's endpoints, parks for unused queues) and routes. A kernel a kind: a weight column
    (its matvecs a row: `smv.row`, a drain where the column holds none of a matvec, zeros on where
    the chain passes it), a unit's CX roots, each CY root (row ops: chain / stream / K / V / host
    in, region parameters copied in and `row()`, casts and sends out), a K / V root and its PEs
    (`cy_satt` / `satt`), the lane root, the filler. Each region is `emit::region`'s print.
  - **Host** (`stream::host`). Every array a PE takes (tiles, regions' tables and static
    residents, each row's scalars: positions and lane ids, the host's rows in its own order, the
    context clamped to f16's range in an IEEE build) from a step's captured values
    (`flux_2_minis_primitive_goldens` over every node), grouped into memcpy rectangles.
  - **Simulator lessons.** A PE that runs several matvecs a row into one chain must not issue an
    async send while its last one is still pending on that microthread (the simulator stops:
    "overwriting ut_instr"): the stream units' chain members are synchronous (`ops/schain.csl`).
    Loading is not the cost (about 2.5M simulated cycles for the whole program's arrays, 16
    memcpy channels); the launch reaches row groups at different times, so the stamps' span
    includes up to about 2M cycles of that skew.
  - **Primitives.** `smv.row(t)` and `X32` (an f32 row as two 16-bit halves: the lane matvecs);
    P7 holds the queries on the column (`satt.store_q` / `query_q`, `cy_satt.query_q`: a row's
    q is stored with its k / v and fetched up the chain when it queries, so no CY root keeps a
    q over the barrier); the region printer reads one rotation's tables for q's and k's angles.
  - **Cost** (`cost::stream`): P1-stream 0.55 cycles an element a row, regions at 2 cycles an op
    element + 1,200 an all-reduce + 5,000 a rotation, P7 4,613 a store and 3,521 a query, the
    units' phases along the stream. flux2-mini: 26.3M cycles predicted for the step (v0's terms are
    pessimistic for the regions; see the measurement below).
  - **Results** (bf16 build, 64 + 64 rows, against the step's captured values; rel L2 / cosine):
    - Entry units only (`--stages 1`, 6 x 18 PEs): context_embed 6.2e-5 / 0.999999998, x_embed
      7.6e-6; 86k cycles.
    - Entry + lane prologue + double block 0, both sides (`--stages 2`, 29 x 28 PEs, 2 K / V
      columns): the block's residual out 1.40e-3 / 0.9999990 (text), 1.42e-3 / 0.9999990
      (image); taps (`--tap`) inside it: layernorm + modulate 9.0e-5, attention + out proj
      2.9e-3, gated residual 6.2e-4, linear_out 3.4e-3. 6.18M cycles of PE stamps (that includes
      the launch's row-group skew); 1.7 h of simulator time; no redo pass in the K / V columns.
    - **The full step** (80 x 28 PEs at origin 169ab59ba's layout, vectorised regions, 8 K / V
      columns): the velocity's patches against the capture rel L2 4.6e-3, cosine 0.99999;
      unpatchified against diffusers' `Flux2Transformer2DModel` rel L2 5.5e-3, cosine
      0.999985 (the host forward's own distance from diffusers is 5.4e-3: what is left is the
      trunk's bf16). 10.7M cycles of PE stamps (11.3M simulated in all, with the loads;
      cost model 6.5M), 3.0 h of simulator time; no redo pass in any K / V column.
    - The stall it first hit (13.76M cycles, every PE waiting) was the last single block's
      silence: a side's outputs were the values read outside it by nodes its *first* class runs,
      and the exit's nodes run for the image rows alone, so the text-first single block left
      nothing on the row stream and the exit waited for its 64 rows. A side is lowered over all
      its classes' words now. (A 16 + 16-row build was seen to stall at `--stages 2`; from
      18ebbe86c on it does not, in either layout: M1f.)
  - Decode is unchanged: plan + build `diff -r` identical against a6de5047d's wafer-compile for
    q35_08b, q35_4b, q35_9b, q3_8b, llama31-8b, gemma_e2b, gemma_e4b, gemma_12b, gptoss_20b, tiny
    t08, tiny llama, tiny gemma, tiny gpt-oss. tiny_q35_t08 on the simulator (wafer-bench, 5
    steps): 0 mismatches of 14 tokens in both sequences, rollback and logits argmax as before.
- **What M1e / M2 need** (from M1d):
  - Sim time: the simulator runs about 1,000-2,000 cycles a second over this fabric, so the full
    step is hours; the regions' scalar loops (the chunk sums, P5's digit products) and P7's
    scalar store are most of a row's cycles: vectorise them (the cost model's region and P7 terms
    are v0's).
  - The host: the artifact's entry streams (M1e): `wafer-bench --reading` reading `host.json`'s
    arrays from a checkpoint and the host's own ops (patchify, cells, the sinusoid of the timestep)
    computed by the engine from the trace, not taken from a capture; the velocity unpatchified.
  - Several steps (M2): the lane prologue and the stream re-run a step with new latents and
    timestep; tiles stay. Replicas of a unit (rows split), and klein's sizes: its units need
    `PY` above 16 (hidden 3,072) and the K / V columns' entries (T = 4,608) exceed a PE: P7 v1.
  - Co-resident readings (text encoder, VAE) on one fabric: the floorplan places readings side
    by side; colours are already allocated per class.
- **M1e, the reading's entries from the engine.** The host's ops around a stream step are the
  engine's, computed from the step's own inputs, not taken from a capture:
  - **The artifact states the entries** (`stream::host`, `host.json`). An array a step's inputs
    make (a value that reaches a trace input: the host rows, each row's scalars, the lane
    prologue's input) is also written as a gather (`entries/<k>.npy`: each word a value's
    (row, column), flags for a one-row lane vector, f16 clamping, a flat resident). The
    capture-fed fill goes through the same gather (`host::apply`), so the h2d files are unchanged.
    `host.json` names the model, the reading, its classes (request words, rows) and the values
    read back whole (`outs`). flux2-mini: 16 entries (the lane prologue's sinusoids, the entries'
    context and patch rows, every unit's per-row positions and lane ids) over 6 values.
  - **The engine computes them** (engine-cerebras `reading`). `Reading::step` lays the step out
    from the classes (a lane a class; a lane whose word reads a voxel input on the latent's grid);
    `Reading::entries` runs the trace's own entry ops (patchify, the cell table and its split, the
    timestep and guidance sinusoids, the request ids, the context's rows) with engine-cpu's
    `Forward::stream_values` on a weightless `Forward::host`: only the nodes the entries' values
    need. `Reading::finish` runs the exit's ops (unpatchify) over the rows read back. No model op
    is named in the engine: the values are the trace's, by id.
  - **The loop** (`reading::Denoise`): the poem's flow schedule (`ScheduleFact::flow_sigmas`, the
    guest's `FlowMatchEuler` on the host: flux2-mini's pinned sigmas resampled), timestep
    `sigma · train_steps`, an Euler step a velocity. `wafer-bench --reading <program dir>
    --program <compiled> --inputs <dir> --grid T,H,W --steps N --golden <dir>` runs it on the
    simulator (or `--cmaddr`): the program loaded a step (the stream does not re-arm yet), the
    entries written from the engine, the rows read back and finished on the host, each velocity
    and latent against engine-cpu's. `--entries <dir>` writes a step's entries for
    `tools/wafer/stream/run.py --entries`.
  - **Host-only results** (engine-cerebras `a_stream_readings_entries_are_the_engines`):
    - Every entry the engine computes for flux2-mini's step equals the capture-fed compile's,
      bit for bit (16 arrays; also through `wafer-bench --entries`, byte for byte the h2d files).
    - The velocity from the rows a program returns, unpatchified on the host, equals
      `Forward::stream`'s bit for bit.
    - The host's 2-step loop (sigmas 1.0, 0.7672, 0) through that split lands within rel L2
      3.48e-3, cosine 0.99999393 of diffusers' loop over the same sigmas
      (`mk_tiny_flux2.py --loop`, f32).
  - **Simulator** (`wafer-bench --reading --rows 1`, the Rust runner in the SDK container; every
    entry the engine's, the static arrays the compile's; rel L2 / cosine against the capture):
    - Entry units only (`--stages 1`): context_embed 6.19e-5 / 0.999999998, x_embed 7.64e-6 /
      0.99999999997; 147k cycles of stamps; 3 min.
    - Entry + lane prologue (its input the engine's timestep and guidance sinusoids) + double
      block 0 (`--stages 2`): the block's residual out 1.397e-3 / 0.9999990 (text), 1.419e-3 /
      0.9999990 (image), as M1d's capture-fed run; 7.02M cycles of stamps; 40 min.
    - The full step does not finish on the simulator yet, from the capture's inputs either:
      M1d's run stopped after 3 h (`std::runtime_error`), a 16-channel retry at 5.8M simulated
      cycles ("the received length (0 bytes) is not expected (8192 bytes), could be a kernel
      stall", then signal 11). That is the program's (M2), not the host's: the same `run/prog`
      runs a step and a 2-step loop on the host bit for bit, and `wafer-bench --reading --steps
      N --golden` holds each step's velocity and latent to engine-cpu's once it does.
  - Decode is unchanged: plan + build `diff -r` identical against 33ab39681's wafer-compile for
    q35_08b, q35_9b, llama31-8b, gptoss_20b, gemma_e2b and tiny t08.
  - Left for M2: the stream re-armed for a second launch (tiles kept, a step's entries
    rewritten); the static arrays (tiles, residents) cut from a checkpoint by the engine, not
    read from the capture (`--values` still names the weights' source); a step's inputs from a
    guest's request (the inferlet's denoise loop driving the reading), not files.
- **M2a** (klein-4B's stream program, compile side first; the v0 scalar loops vectorised; the
  cost model fitted). The survey's stream line for `flux2-klein-4b-bf16-kv-bf16`:
  - before (33ab39681): `STREAM-BUILD lanes: no column roots up to 64 fit its tiles`;
  - after, 64 + 64 rows: `STREAM-BUILD-OK fabric=715x904 py=192 kv-columns=600x8`
    (44.2M cycles predicted, 50.5 ms). **It links**: cslc builds its 23,007 ELFs for the
    737 x 906 memcpy fabric in 55 min and 58 GB of memory, no error;
  - after, 1024² (`ROWS=context=512,latents=4096`, the poem's 512 text rows and 4,096 image
    rows): `STREAM-BUILD-OK fabric=741x1143 py=256 kv-columns=600x99` (2,116M cycles
    predicted, 2.4 s a step: below).
  - Each blocker the survey showed, in order, and what moved it:
    - **The lane prologue's tiles** (170M weights over one unit: mod_img, mod_txt,
      mod_single, norm_out): a unit takes up to 512 column roots (`place::MAX_REST`), not 64.
    - **Colours: 3,537 classes** (a chain class a row of every unit). `Palette::allocate` reads
      the footprints in batches of links whose classes fit a route word's colours (one batch
      up to 255 classes, so decode is unchanged); interference colouring then reuses 21
      colours across the units.
    - **Host rows over a barrier.** klein's double blocks read the context straight from the
      host (no context embedder in the denoise reading): the schedule gives a host unit with a
      barrier its late run and the emitter reads the host buffer again after the barrier.
    - **2,293 PEs wide.** `floorplan::stream` fills **bands** of units within the system fabric
      (`Bounds::system()`: 755 x 1170, the WSE-3's less the memcpy halo), a block never split
      over two. The row stream into a new band is a staircase (`Stream::wrap`): row `j` runs
      back west, turns south one column east of row `j - 1`, and enters the band's first unit
      from the east, so no two rows' wires cross and all `py` of them take one class.
    - **`py` pinned at hidden / 16.** `build::program_py` searches (row roots, tile budget)
      pairs: every divisor of the hidden width leaving a CX root at least 4 columns, at a weight
      PE's whole 12,288 words, 3/4 and 1/2 of them (the spare fabric spent on shorter rows),
      built eight at a time, the cost model's cheapest kept. `wafer-compile stream --py N
      --tile W` pins them (`--py 16 --tile 12288`: flux2-mini as M1d built it, 60 x 28; the
      search's own choice is 80 x 28).
    - **K / V past one column (P7 v1).** A K / V column takes the attention PEs its block's rows
      ask for (`build::kv_pya`: 47 entries of a 128-wide head, its q held, a PE), not 8 always:
      99 PEs a kv head at 4,608 rows. The PEs keep rows `p % PYA` and the chain sums their
      `[o | l]` against the head's one carried reference, so the merge of the PEs' partials is
      the sum; a longer column costs the chain's hops (pipelined) and the band's height.
    - **Rows held over the barrier** (at 1024²: 4,096 latent rows of the residual, 49,152 words
      at a row root of py 256; the single blocks hold two values). **The row store**
      (`Layout::spill`): a held chunk is cast along the row as it is made, the weight PE of column
      `t % px` keeps row `t`'s, and at the start of each late row the keeper sends it up the row
      chain while the rest send zeros. A weight PE holds its tiles and its share within 20,480
      words. `wafer-compile stream --spill` (`Pins::spill`) puts every attending unit's held rows
      there (flux2-mini `--stages 2 --spill`: compiles under cslc; its simulator run, the
      store's numerics, is queued behind M1e's runs).
    - Refusals by name where a unit still does not fit: the host's rows held whole (in or out,
      a step's at once: stream them a batch at a time), rows the row store cannot take beside
      the tiles, a band too narrow for a block.
  - **Vectorised primitives** (the prims harnesses, same cases and goldens as M1c; bf16 build
    but the IEEE P7 line; cycles a row of `row()`, a stored row, a query):

    | case (flux2-mini nodes) | v0 (M1c) | M2a | vs interp |
    |---|---|---|---|
    | layernorm + modulate (41, 42), 4 x 64, 2 all-reduces | 3,044 | 1,285 | exact (was 2.8e-5) |
    | per-head rmsnorm + rope_axes (45, 46), 2 x 128 | 11,791 | 7,011 | exact; 3.0e-5 vs trace as before |
    | gated residual + layernorm + modulate (57, 64, 65) | 3,281 | 1,526 | 8e-11 |
    | SwiGLU (67), gated residual (69), residual adds (124, 125) | 4,847 / 248 / 332 | unchanged | as before |
    | P7 a stored row (node 52, 128 rows, 8 PEs, IEEE) | 4,613 | 508 | rel L2 1.2e-4 vs f64, as before |
    | P7 a query | 3,521 | 3,513 | (bf16: 9.9e-4, as before) |

    - The chunk sums are a halving tree of DSD adds over a scratch row (`vsum`; an odd run's
      middle word waits a level).
    - P5's tables (`ops/srope.csl`, stream-only; `ops/rope.csl` is decode's and unchanged): a
      position's digit is one for all pairs, so a level is six NROT-wide DSD ops, not a scalar
      product a pair.
    - P7's store (`satt.store`, `store_q`, `query_q`): the key narrowed down its entry's column
      of `kc` by one strided `@fs2h`, the value and the held q by one each.
  - **The cost model** (`cost::stream`) reads a region off its printed `row()`
    (`RegionFeatures`: DSD ops and their elements, vector exps and reciprocals, chunk sums,
    scalar math, all-reduces, P5 tables), a query off its block's entries a PE, and prices a
    program at other rows a class (`time_at`; `wafer-compile stream --price`). Fitted to the
    measurements above: a DSD element 0.69 cycles (an op's issue fits to 0), a vector exp or
    reciprocal element 16.2, an all-reduce over 4 roots 210, a P5 axis (its tables and their use) 1,215, a
    stored row 508, a query 3,513 at 16 entries a PE (1,100 + 151 an entry: the slope is satt's
    per-entry work, its runs at 32 and 64 entries are queued). Within 1% on the four regions
    with norms, rotations or SwiGLU, 11% on the plain residual adds. flux2-mini's step is now
    6.5M cycles predicted (v0's terms: 26.3M).
  - **klein-4B at 1024²** (one CS-3, the 1024² build above, 847k of the fabric's 883k PEs,
    512k of them weight PEs): 2,116M cycles, 2.4 s a step at 875 MHz. The attention
    barrier serialises the blocks (all of a block's rows store before any queries), so a step
    is about 25 blocks x 4,608 rows x the slowest root a row, not the pipeline of §1's
    estimate. The CY roots, not the matvecs (5.7k cycles a row), set that pace: a single block's
    pre-barrier row is 20.2k cycles (its SwiGLU over few column roots, q / k norms and four P5
    axes a head), a double block's post-barrier row 19.1k (a query at 47 entries a PE, 8.2k).
    One CS-3 holds it all (weights, 600 K / V columns, the row store), so the fabric is not the
    limit; what speeds it, in order: more column roots for the col-side regions (the search
    already trades tile budget for roots; it does not yet weigh the regions' chunk), P7 over
    more PEs a head (a column of two or three sub-columns: 16 entries a PE, 3.5k a query),
    and row replicas of each unit (rows split; R copies of the weights, about 2 on one wafer).
    Streaming weights would not help: the time is in the roots.
  - **Link.** Weight PEs no longer take a row index (their code never read it): klein's 64-row
    program has 21,507 distinct PE images, not 447,918; its `layout.csl` is 462 MB (660k
    `@set_tile_code`, 3.6M `@set_color_config`). With the memcpy halo the 64-row program is 737 x 906 PEs, and cslc builds it (above):
    klein-4B's denoise links for the system fabric. The 1024² build is 619 MB of CSL and was
    not linked (about the same image count; cslc's memory and time scale with it).
  - Decode is unchanged: plan + build `diff -r` identical against 33ab39681's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08.
- **M2b** (klein-4B's 1024² program links; the cost model's levers chosen by the search). The
  survey's line for `flux2-klein-4b-bf16-kv-bf16` at `ROWS=context=512,latents=4096`:
  `STREAM-BUILD-OK fabric=741x1143 py=256 kv-columns=600x99`, now **linked for the system
  fabric** (`cslc --fabric-dims=762,1172`).
  - **One image a kind of PE.** cslc builds an ELF a distinct (file, params); M2a's emitter wrote
    a file a PE where PEs differed only in a few numbers. What differed is now each PE's runtime
    config words (`pcfg`, a host array, `Fill::Words`, as decode's rcfg): a root's lane drains
    (the bus offsets of its lane words, a zero-length drain where none), the row store's keeper
    column and fetch role, a column-side matvec's leaf role, a weight PE's column-chain role, a
    K / V PE's index and chain role. Files that differ only in their comments are one file
    (`emit::stream::dedup`: the regions first, the roots' imports renamed, then the kernels).
    Colours stay `@set_tile_code` params (a few distinct sets a kind).

    | program | distinct (file, params) before | after | ELFs (cslc) before | after | link |
    |---|---|---|---|---|---|
    | flux2-mini, the search's 80 x 28 | 514 | 108 | 526 | 161 | 51 s -> 20 s |
    | klein-4B, 64 + 64 rows (715 x 904) | 22,826 | 553 | 23,007 | 1,215 | 55 min / 58 GB -> 13.6 min / ~36 GB |
    | klein-4B, 1024² (741 x 1143) | 76,450 | 381 | not linked | 1,065 | 14.1 min / ~37 GB, rc 0 |
    | klein-4B, 1024², the search's choice (helpers) | | 401 | | 1,101 | 14.1 min / ~35 GB, rc 0 |

    Every flux2-mini PE's code with its config words put in is M1d's (bar the zero-length drain
    a farthest lane root now declares), every host array as before (`scratch/.../equiv.py`
    over the `--values` build).
  - **What stopped the 1024² link** (once it was a thousand images): `ld.lld: ran out of PE
    memory` at the roots holding every row's per-row scalars: a single block's head roots 4,608
    rows x 4 rope positions (73 KB), the CX roots 4,096 x 3 (49 KB). A scalar a region's row
    never reads (the lane id that picked a lane vector) is not held now; past `SCAL_WORDS`
    (2,048) a unit's CY roots' scalars come down their column chains with the phase's first
    row-side matvec: the weight PE of row `t % py` adds row `t`'s (`Fill::RowScalars`, 72 words a
    PE) into the chain, the rest add zeros. The largest root is then 30 KB.
  - **The cost model** (`cost::stream`): the step's timeline is row by row through each unit's
    schedule (`timeline`: a unit's CX roots run their segments in order, a row no sooner than it
    arrives nor faster than the unit's pace, a late segment once its block has stored), so a
    double block's late rows pace the next block (the phase sum did not); a query is priced at
    the program's own K / V PEs a head. `wafer-compile stream --price .. --parts` prints each
    root-side step a row. klein at 1024²: 2,116M (M2a's phase sum) -> 2,297M cycles, 2.62 s.
    Against M1d's simulator stamps (flux2-mini, 80 x 28): every unit's CX roots 12-15 end 4.47M
    cycles after roots 0-11 (and the lane prologue's CY roots at 4.85M): a fixed lane-bus
    start-up, not a row cost; less it the step is about 6.2M against 6.0-6.7M predicted.
  - **Levers, each a search dimension** (`build::KvSplit`, `--cut THIN,SUB,WIDEN,HELP`; py and
    tile both pinned keep the plain cut, so M1d's flux2-mini is as before):

    | lever | klein 1024² predicted | note |
    |---|---|---|
    | none (M2b's timeline) | 2,297M cycles, 2.62 s | |
    | rest roots widened a quarter / a half | does not fit (730 x 1524: a fourth band) | flux2-mini 5.9M -> 5.3M |
    | K / V column folded over 2-3 columns (snake), thinner PEs | refused: a head root's link to a column two wide meets 21+ K / V chains, more than the palette's colours | builds for flux2-mini (80 x 24) |
    | **helpers: a rest-root column region over up to 15 PEs below each root** | **1,326M cycles, 1.52 s** | the search's choice; flux2-mini 5.0M |
    | row replicas | not tried: the fabric is full (847k of 883k PEs) | |

    Helpers (`Built::helper`, `emit::stream::helper_file`): the unit's costliest column region
    on its rest roots that cuts evenly and reads only rows and scalars (klein's SwiGLU, 288
    columns a root) runs over the root and `h` helper PEs straight below it in the band's K / V
    rows (their own chain: the cast down, a sum chain back). The root stages its chunk, casts
    the helpers' parts, runs its own `q = chunk / (h + 1)` columns and takes the helpers' results
    as a sum (each helper's in its place of a zero vector). The first unit of a block whose
    later units' links pass west below it takes none (their colours would meet). A single
    block's pre-barrier row: 20.2k -> 11.2k cycles (SwiGLU 10.9k -> 1.9k); the head roots' norms
    and four P5 axes (8.8k) set the pace now. `tools/wafer/prims/help` runs the emitted helper
    kernel (flux2-mini's SwiGLU, 6 roots x 7 helpers, q 16) against `kernels_ir::interp` on the
    simulator (`prims/help/run.py`; queued behind M1f's runs when M2b closed: its result is
    not in yet, so the helpers' numerics are cslc-checked only).
  - What is left: the head roots' 8.8k (the P5 tables a row at every head root: a position's
    cos / sin are the same for the row's 24 heads; zero positions skip nothing yet); the
    K / V fold needs links that do not cross the folded columns (head roots at a pitch of the
    fold); the lane bus start-up (4.5M cycles at flux2-mini); the full step on the simulator
    with the config words, the carried scalars and helpers (M1f's runs use their own builds).
  - Decode is unchanged: plan + build `diff -r` identical against 18ebbe86c's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08.

- **M1f** (flux2-mini at 16 + 16 rows on the simulator; the denoise loop from the engine's
  inputs; the program re-launched a step). The goldens at 16 rows a lane (a 16-row text lane, an
  8 x 8 latent): engine-cerebras' `flux_2_minis_reading_files` at `READING_ROWS=16` (capture,
  program, inputs, engine-cpu's loop), `mk_tiny_flux2.py --rows 16` (diffusers' step and loop; the
  host's 2-step loop lands 3.66e-3 / 0.9999933 from it, as 3.48e-3 at 64 rows).
  - **The small-row stall does not reproduce** from 18ebbe86c on. `--stages 2` at 16 + 16 runs
    to the end in the search's layout (39 x 28: 3.87M cycles simulated, 31 min; its rows were
    not compared: host.json's capture path was relative, absolute now) and in M1d's pinned one
    (`--py 16 --tile 12288`, 27 x 28: 1.63e-3 / 0.9999987 text, 1.57e-3 / 0.9999988 image,
    3.92M cycles of stamps, 26 min). The
    16-row programs differ from the 64-row ones only in counts and buffer sizes, the K / V PEs'
    entries (4 a PE, not 16) and the entries' and exit's single weight column (their tiles fit
    one at 16 rows); their routes, flow for flow (each sender's receivers), are the 64-row
    program's. No 16-row artifact of M1d's stall was kept, so what moved it is not known (it
    is none of M1f's changes: both programs are 18ebbe86c's emitter's).
  - **The full step at 16 + 16** (`run.py`, 18ebbe86c's layout, 77 x 28, 8 K / V columns): the
    velocity's patches against the capture rel L2 4.90e-3, cosine 0.999988; unpatchified against
    diffusers 5.59e-3, 0.999984. 8.08M cycles of stamps (8.32M simulated), 2.1 h: a quarter of
    the rows saves a quarter of the stamps, the first launch's start-up being most of them
    (below).
  - **The program re-launches** (`Reading::write_static` once, `Reading::launch` a step: the
    step's entries, `run`, the rows back; wafer-bench `--reload 1` for M1e's load a step,
    `--save` for each step's velocity and latent). Nothing in a stream kernel's `go` kept state
    across launches but the K / V roots' pass count, which a launch now zeroes.
  - **The 2-step loop from the engine's inputs, on the search's default** (114729f71's: pcfg
    words, helpers; 77 x 28; `wafer-bench --reading --steps 2`, one load):

    | | vs engine-cpu (rel L2 / cosine) | vs diffusers | cycles | wall |
    |---|---|---|---|---|
    | step 0 velocity (sigma 1), the first launch | 4.85e-3 / 0.9999882 | | 7.90M | 79 min |
    | step 1 velocity (sigma 0.7672), re-launched | 6.36e-3 / 0.9999798 | | 1.27M | 15 min |
    | latent after step 1 | 1.20e-3 / 0.9999993 | | | |
    | latent after step 2 | 3.86e-3 / 0.9999925 | 3.58e-3 / 0.9999936 | | |

    - The helpers' and config words' numerics hold (step 0 is as the 18ebbe86c build's single
      step above); the host's own loop is 3.66e-3 from diffusers, the wafer's 3.58e-3.
    - The re-launched step is 1.27M cycles of stamps, not 7.9M: the rest of the first launch is
      start-up (the launch reaching the PEs, the lane bus's first words), paid once a load. A
      step of the loop is that 1.27M (1.45 ms at 875 MHz), against the cost model's 1.21M for the
      program (M2b's timeline): 5% off.
  - Decode is unchanged: plan + build `diff -r` identical against 114729f71's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08.
  - Left: a step's inputs from a guest's request (the inferlet's loop), not files; the static
    arrays cut from a checkpoint, not a capture; the 64-row and klein programs re-launched on
    the simulator (only 16 + 16 was).

- **M1g** (flux2-mini through the engine as Pie drives it: a guest's denoise loop, the static
  arrays from a checkpoint).
  - **The static arrays from the checkpoint** (kernels-cerebras `stream::host`). Every array a
    value fills that no step's inputs make (the weight tiles, the regions' shared residents) is
    stated as spans over the trace's values (`statics/<k>.npy`, `[value, row, col, len, stride,
    flags, count, step]`: runs of a value's elements cast to the array's words, a tile one span
    where its rows are consecutive inputs). A program now compiles with no capture: its h2d files
    are only the arrays no value fills (the regions' tables, the PEs' config words; flux2-mini at
    16 + 16: 41 arrays of spans, 66 on file, 9 entries; 2.7 MB against the capture compile's
    27 MB). Given a capture, each array's spans are held to its words at compile time. The engine
    (`Reading::statics`) lands the checkpoint through the import contract (`prepare::land`),
    makes the spans' values with an engine-cpu `Forward` over the landed planes, and writes the
    arrays once a load, on the first step. `tools/wafer/stream/run.py` still needs a capture
    compile (it reads every array's file).
  - **The request path** (engine-cerebras `backend::stream`). A load whose program is a stream
    reading's (`[cerebras.debug] program` = `wafer-compile stream`'s directory, `link` = its cslc
    output) is a stream load: the facts and capabilities are the trace's (the voxel axis sized by
    the budgets), the fabric is the SDK runtime on a thread of its own. A fire whose lanes are the
    reading's classes (word and rows, a media lane's clip off `Step::voxels`, one group) is one
    step: the prologues, each lane's inputs off the channels its `InputFeed`s name (the cell the
    instance's own `take` would read, laid out a value a lane / a row a token / a row a voxel),
    `Reading::entries`, the launch, `rows_back`, `Reading::finish` for the outputs the
    attachments read, the epilogues over them (a lane's rows and the pass's later lanes', as for
    decode), the lanes' readouts. The denoise loop is the guest's: its channels, the poem's flow
    sigmas (`ScheduleFact::flow_sigmas`, as the runtime states them to an inferlet), its Euler
    step in the epilogue. No scheduler or runtime change; `[cerebras.debug] stream_fabric =
    "host"` runs the step's device part on engine-cpu (`reading::HostFabric`).
  - **Host-only results** (engine-cerebras `a_stream_reading_runs_through_the_engine`, the
    checkpoint a diffusers `Flux2Transformer2DModel` state dict written from the drawn weights):
    every static array the engine makes from the checkpoint equals the capture compile's, word
    for word; on the host fabric the guest's 2-step loop through `CerebrasEngine::submit` lands
    bit for bit on the host's own loop and 3.48e-3 / 0.99999393 from diffusers at 64 rows.
  - **Simulator** (16 + 16, the search's default, 77 x 28; the same CSL as M1f's, file for file;
    the test binary in the SDK container, one load, `CerebrasEngine` driven by the guest):

    | | vs engine-cpu (rel L2 / cosine) | vs diffusers | cycles | wall |
    |---|---|---|---|---|
    | step 0 (sigma 1), the first launch | | | 7.90M | 75 min |
    | step 1 (sigma 0.7672), re-launched | | | 1.27M | 14 min |
    | latent after step 1 | 1.20e-3 / 0.9999993 | | | |
    | latent after step 2 | 3.86e-3 / 0.9999926 | 3.58e-3 / 0.9999936 | | |

    The final latent is M1f's capture-fed `wafer-bench` run's bit for bit: the checkpoint's
    static arrays are the capture's, and the guest's loop is the host's.
  - Decode is unchanged: plan + build `diff -r` identical against 847c48a9d's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08.
  - Left: the stream program compiled and linked on load into the cache (it is given today, as
    `build = "never"` would); `pie serve` with an inferlet: `text-to-media` needs a `text`
    reading (flux2-mini has none) and the VAE readings, and the runtime's boot of a stream
    deployment on the cerebras engine is untried; the 64-row and klein programs re-launched on the
    simulator; the statics' host memory at klein (an f32 forward of the whole checkpoint).

- **M1h** (flux2-mini's whole pipeline through Pie: its text encoder on the wafer, compiled on
  load; `pie run text-to-media` making an image on the host fabric).
  - **The miniature's pipeline** (`models/flux_2.poem`). flux2-mini gained a text encoder and a
    VAE, its DiT unchanged: a Qwen3 of two layers (hidden 192, 2 heads of 128, MHA, SwiGLU 384,
    theta 1e6) on the tokenizer's vocabulary, tapped at its second layer (192 wide: the context
    embedder's input, so the denoise reading is M1g's, node for node); a VAE of four 32-channel
    blocks (a resnet an encoder block, 8 groups; `lib/vae.poem`'s ladder, groups and depth are
    now arguments, klein's trace unchanged). Its checkpoint is a diffusers pipeline's (`dit.`,
    `te.`, `vae.`); `tools/wafer/flux/mk_tiny_flux2_pipe.py` draws it as the tests do and gives
    transformers' / diffusers' goldens (`--text`, `--vae`, `--pipe`; `--ckpt` writes it with the
    Qwen3 tokenizer's files).
  - **The `text` reading on the stream compiler**, no per-model code. What it took, each generic:
    - Row classes: a reading whose nodes no `stream` value changes is one class
      (`stream::classes`).
    - `attn.prefill` over the pages its own layer appends (`attn.kv_append`) is a joint attention
      (`Mixer::Joint { causal }`, the unit's `Store` / `Attend`), causal as the trace says.
      P7's PEs mask a query's later entries (`satt.csl` `CAUSAL`: entry `e` of PE `r` is row
      `e * PYA + r`; the query's index is its row's, the step one request; the emitter refuses
      a causal column storing more than one unit's rows).
    - The token embedding is the host's (a row lookup by the step's ids, no arithmetic: its
      rows enter the first unit as a host input's; `Forward::host_for` holds the table alone).
    - `rope` of q and k together (NeoX halves) lowers when a run names it (`fuse::chain`, the
      explicit arm, so CUDA's chains are untouched) as two rotations; the region printer's
      peer gather takes a head's halves swap (`hop`-wide blocks) besides the pair swap.
    - A value the reading reads out (a trace output no node reads) leaves its unit for the host.
  - Program: 2 units, 4 K / V columns, 10 x 36 PEs at 16 rows. On the simulator (`stream/run.py`,
    capture-fed, `--stages 1`): layer 0's residual 5.6e-3 / 0.999984 against the capture, 0.75M
    cycles.
  - **Compile on load** (`prepare::stream_programs`). A load with no `[cerebras.debug] program`
    whose decode plan fails (it does for flux: no decode lowering) compiles every reading of the
    trace's `reading` switch the stream compiler lowers, at `[cerebras.wafer] stream_rows` rows
    a class, into the cache (`wafer/s-<model>-<key>/<reading>/`, the key a hash of every
    emitted file, the op modules, the fabric and cslc), and links each with cslc for the
    simulator or the system (none for the host fabric). A reading the compiler refuses is
    logged and not served. A second load finds them (`a_text_reading_feeds_the_denoiser`).
    LLM loads are unchanged: the decode path runs first, as before.
  - **The engine serves several readings** (`backend::stream::Streams`): a step picks its
    program by its lanes' words (a lane's word may set a switch its reading does not read: the
    nodes the words run decide), each reading's static arrays made from the checkpoint at its
    first step and written when its program opens; the SDK's programs open one at a time on one
    thread (the SDK wants every runtime of a process made on one). A causal reading's lane may
    bring fewer tokens than the program's rows: the rest are padding no earlier row reads, and
    the rows read out are the lane's own. The runtime's checkpoint contract for Cerebras now
    admits a bias and a unary map (the VAE's batch-norm statistics; the cerebras engine lands
    on the host, which runs them).
  - **Results, the text reading** (16 tokens: "a red fox sleeping" in the chat template):
    | | vs engine-cpu | vs transformers' `Qwen3ForCausalLM` |
    |---|---|---|
    | engine-cpu (`Forward::stream`, bf16 trunk) | | 9.18e-3 / 0.9999578 |
    | host fabric, through `CerebrasEngine::submit`, compiled on load | bit for bit | 9.18e-3 |
    | **simulator**, through the engine, compiled + linked on load | 7.86e-3 / 0.9999691 | **8.74e-3 / 0.9999619** |

    The simulator's text step is 1.38M cycles of stamps (the first launch, start-up included),
    4.8 min of simulator time.
  - **The text rows feed the denoiser** (host fabric, `the_text_rows_feed_the_denoise_loop`):
    the guest's text pass, then its 2-step loop over those rows, bit for bit the host's own loop
    over engine-cpu's text rows; against diffusers' pipeline (the prompt through transformers'
    Qwen3, then the same 2 Euler steps from the same latent): 3.88e-3 / 0.9999925.
    **On the simulator**, the denoise loop over the simulator's own text rows (a second
    process: the simulator holds one program a process), through the engine, both programs
    compiled and linked on load: latent after step 1 1.40e-3 / 0.9999990, after step 2
    4.06e-3 / 0.9999918 against the host's loop over engine-cpu's text rows, and **3.88e-3 /
    0.9999925 against diffusers' pipeline** from the prompt. Step 0 (the first launch) 7.90M
    cycles, step 1 1.27M cycles of stamps (M1g's, the same program); 76 + 15 min.
  - **The VAE** (`vae.decode`): engine-cpu's stream forward now runs a grid reading's ops
    (`conv.conv3d` over each clip, `conv.group_norm`, `conv.upsample`, `conv.pixel_(un)shuffle`,
    `conv.grid`, `elemwise.standardize`, `attn.voxel`; a convolution's kernel relabelled taps
    major as every engine lays it): the decode of the pipeline's latent against diffusers'
    `AutoencoderKLFlux2` 1.08e-2 / 0.9999419, **PSNR 50.4 dB**. **On the wafer it is not
    lowered yet**: the stream compiler refuses it ("the stream step runs no layers": a grid
    reading has no layer stack). The host fabric serves a reading no program runs whole on
    its forward, as a debug fallback, never a wafer's (the VAE is not negligible).
  - **`pie run text-to-media` on the host fabric** (`stream_fabric = "host"`, `stream_rows =
    16`, the checkpoint `mk_tiny_flux2_pipe.py --ckpt`'s): the inferlet's own text pass, 2
    denoise steps and the VAE decode through the runtime's standard path, no scheduler or
    runtime change; a 64 x 64 PNG. Against diffusers' pipeline from the inferlet's own noise
    (eta's keyed normal, seed 7) and sigmas: **PSNR 48.2 dB** (8-bit images, max |diff| 5/255).
  - Decode is unchanged: plan + build `diff -r` identical against 0d1cbe4fd's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08.
  - Left:
    - The VAE on the wafer (the grid regime): a `conv.conv3d` is an im2col (P11, the row order)
      into a P1 matvec, so the convolution's arithmetic is the existing modules', but a row
      needs its 3 x 3 neighbours (a line buffer of `2w + 3` rows at the CX roots, P9) and
      `conv.group_norm` is a reduction over every row of a clip (a barrier and a P2 over rows,
      as the joint attention's store); a stage per conv, the upsamplings as the row stream's
      order. None of it is built.
    - Co-resident readings: the simulator runs one program a process (a second
      `Simfabric::Create` aborts), so the denoise loop over the simulator's text rows ran in a
      second process; one program holding text, DiT (and VAE) side by side is the fix.
    - GQA in the stream attention (the mini's encoder is MHA; klein's Qwen3-4B is GQA 32 / 8:
      a head root's q heads a kv head), and more than one tap (klein's three: a second value
      on the row stream).

- **M1i** (flux2-mini's VAE decode on the wafer, generic over the grid family; its readings
  co-resident in one program).
  - **The grid regime** (kernels-cerebras `stream::grid`, `emit::grid`; `wafer-compile stream
    --reading vae.decode [--clip t,h,w]`, compiled on load as the stream readings are). A reading
    whose nodes run no layer and hold a convolution or a group norm is read off its trace into
    **passes over one unit**: `P` CX roots, `P` CY roots, `P x P` weight PEs, every matvec of the
    reading tiled over all of them. No op is named: each lowers by its family.
    - `conv.conv3d` is a **gather** (its im2col: an output row's `k x k` taps from a line buffer
      of its input's rows at the roots, zeros outside the grid; stride and front padding as the
      op states) feeding the P1 tile matvec (`mv.csl`'s `tile32`, its tile the conv kernel's
      taps-major columns of the root's channels), the bias a printed region. A 1 x 1 conv and a
      `linear.matmul` are the matvec alone.
    - `conv.upsample`, `conv.pixel_shuffle`, `conv.pixel_unshuffle` are gathers too (one tap a
      row, a channel of every `r²`, or `r²` taps interleaved); an upsampling read by one conv
      folds into its taps (the conv's grid the source's at twice its coordinates).
    - `conv.group_norm` is a **reduction over its input's every row** (P2 over a clip's rows):
      the input's chunk summed a channel as its rows are made (shifted by its first row's
      value), its scale and shift (`rstd · gamma`, `beta - mean · rstd · gamma` a channel) made
      at that pass's end, the apply (and its SiLU) a printed region (`x a + b`, then
      `t / (1 + e^-t)`) a row in a later pass. A root holds whole groups (`P` is the most roots
      that cut every norm's groups evenly), so no all-reduce.
    - `attn.voxel` (one head as wide as the row) is the joint attention's P7 (`satt.csl`): its
      q / k / v (and what makes them after their matvecs) whole at CY root 0, a K / V column
      under it, the stores one pass and the queries the next.
    - Every other elementwise run is a printed root region (`fuse::chain::lower_run`; the
      latent's `elemwise.standardize` added to its explicit arm, so no CUDA chain changes).
    - The host's nodes are the host's (`conv.grid`'s rules at compile time from the clip; the
      latents' `pixel_unshuffle`, an entry).
  - **Passes.** A group norm reads its input's every row and a gather reads rows ahead of the one
    it makes, so the reading is cut where one starts: a pass's *head* makes the rows its gather
    reads (a held value fetched, a norm applied) into the line buffer, its *tail* the rows the
    gather makes (the matvec, the bias, the residual); the head is pulled to each tail row's
    need (every kernel computes it from the row's position, so every PE interleaves alike). A
    value read by a later pass is **held in the weight PEs** (M2a's row store on both sides: a
    row-side value's row `t` by column `t % P` of its root's row, a column-side value's by row
    `t % P` of its root's column, 16-bit, slots shared by lifetime) and fetched back up the chain
    it was cast down; the host's rows enter the same way, and the outputs leave from them.
    flux2-mini at 64 x 64 pixels: **28 passes** (8 x 8 to 64 x 64), 30 matvecs, 22 norms, one
    attention; `P = 8` (a norm's 4 channels a group); a weight PE's store 7,168 16-bit words.
  - **Kernels as programs.** Each kernel is a small interpreter over its pass program (16-bit
    words a PE: each pass's header, its head's and tail's ops, its finals; `emit::grid`'s
    `SWEEP`, `ROOT_OPS`, `WEIGHT_OPS`): unrolled a pass, the weight PE's code alone was 34 KB and
    a root's descriptors overflowed `data.lo`; as programs a weight PE is 35 KB with its tiles and
    store, a root 24 KB. Every root of a side shares a file (CY root 0 holds the attention).
  - **Host emulation** (`grid::emulate`: the passes, each weight PE's tile from the compile's own
    spans, the gathers' taps, the norms, the attention). The engine-cerebras test
    `flux_2_minis_vae_program_is_the_reference` (pipeline latent, 8 x 8):
    | | rel L2 / cosine | PSNR |
    |---|---|---|
    | the grid program on the host vs engine-cpu | 2.70e-3 / 0.99999636 | 63.4 dB |
    | the grid program on the host vs diffusers' `AutoencoderKLFlux2` | 1.09e-2 / 0.99994127 | 50.4 dB (engine-cpu's own: 50.4) |
  - **Simulator** (`flux_2_minis_vae_on_the_fabric`, the test binary in the SDK container, the
    static arrays from the checkpoint, the entries from the engine; `READING_PASSES=N` runs a
    prefix and holds its last value to the host program). Every pass prefix of the decode ran
    and matched the host program (4 passes bit for bit; after the attention 2.7e-3, its 16-bit
    probabilities at 4 keys; 1.4e-2 by pass 27, the norms after it amplifying them). The whole
    decode:
    | latent (pixels) | vs engine-cpu (rel L2 / cosine) | PSNR vs engine-cpu | cycles | wall |
    |---|---|---|---|---|
    | 2 x 2 drawn (16 x 16) | 1.42e-2 / 0.99990192 | 48.2 dB | 4.84M (4.72M with the gather's coordinates hoisted) | 9.6 min |
    | 8 x 8, the pipeline's (64 x 64) | 1.03e-2 / 0.99994750 | 51.0 dB | 73.5M | 2.4 h |

    Against diffusers' `AutoencoderKLFlux2` decode of the same latent: **1.16e-2 / 0.99993318,
    PSNR 49.8 dB** (engine-cpu's own: 1.09e-2, 50.4 dB).

    A row costs about 2.4k cycles (a 4-wide chunk: the norm's SiLU is ~45 DSD ops, the 9-tap
    gather's index math, the fetch round trip up a chain a row), so flux2-mini's 64 x 64 decode is
    73.5M cycles (31,744 tail rows, 84 ms at 875 MHz). Footprint: 9 x 19 PEs (8 + 8 roots,
    64 weight PEs, the K / V column's root and 8 PEs).
  - **Co-resident readings** (kernels-cerebras `stream::coresident`). The load's readings,
    each compiled on its own, are laid side by side in one program (`layout.csl` at the cache
    entry's root placing reading `i` at its column offset, each PE's code its reading's own file:
    cslc resolves a kernel's imports from its directory). Every kernel's `run` takes the reading
    launched and starts only where it is its reading's (`RD`), the rest unblocking at once; each
    reading's `host.json` gains its `origin` and `rd`, so its host side is as it was (every
    array's PEs moved by the origin, the stamps read over its own rectangle, `run` launched
    with its id). The engine opens the one link once (`Streams` keep the link open, a reading's
    static arrays written the first time it runs) and a step launches its reading. flux2-mini:
    text (10 x 36), denoise (77 x 28), VAE decode (9 x 19) in one 96 x 36 program, linked once
    (20 s). **On the simulator** (`flux_2_minis_vae_on_the_fabric`, `READING_COMBINE=1`): the
    2 x 2 decode launched by its id in the combined program is the standalone run's (1.42e-2,
    48.2 dB), 4.72M cycles, but 66 min of simulator against 9.6: the simulator's rate is about
    2.3M tile-cycles a second, so a program of 3,644 simulated tiles runs a reading at ~630
    cycles a second however few of them it uses. An end-to-end through it (~88M cycles, mostly
    the decode) would take ~40 h.
  - **A reading a child process** (`[cerebras.debug] stream_processes = "<wafer-fabric command>"`,
    engine-cerebras `reading::Children`, `bin/wafer-fabric`): each reading linked on its own and
    run in a child process holding its SDK runtime (JSON lines and words over its stdin and
    stdout), one child open at a time (one simulator machine-wide), its static arrays written
    again when it reopens. The engine process drives text, denoise and decode; each simulator
    simulates its own fabric. Co-residence stays the default (a system loads one program).
  - **The grid reading through the engine.** Its class has no token rows (its rows are its clip's
    voxels): a lane's tokens are not held to them; its clip must be the compiled one
    (`host.json`'s `clip`; the compile's clip is the square the trace's patchify of the same
    voxel input makes `stream_rows` token rows of, 8 x 8 at 16). On the host fabric `pie run
    text-to-media` (`--guidance 4`, seed 7) now serves its VAE through the program's host side:
    the image is M1h's bit for bit, 48.2 dB from diffusers' pipeline. (Without `--guidance 4`
    the inferlet runs at guidance 1, diffusers' golden at 4: 17.9 dB, not a regression.)
  - **`pie run text-to-media` fully on the simulator** (`--prompt "a red fox sleeping" --width
    64 --height 64 --steps 2 --seed 7 --guidance 4`; the `pie` binary in the SDK container,
    `stream_processes` the `wafer-fabric` command, `[runtime] silence_timeout` past a simulated
    step's hours): one engine process compiles, links and serves the three readings, the
    inferlet's text pass, 2 denoise steps and the VAE decode each on the simulator, a 64 x 64
    PNG. **48.3 dB against diffusers' pipeline** (max |diff| 4/255; the host fabric's image, M1h's,
    48.2 dB), 47.0 dB against the host fabric's run of the same request.
    | step | cycles | simulator |
    |---|---|---|
    | text (16 tokens) | 1.38M | 4.7 min |
    | denoise step 0 (the first launch, start-up included) | 7.90M | 78 min |
    | denoise step 1 | 1.27M | 15 min |
    | VAE decode (8 x 8 latent, 64 x 64 pixels) | 73.5M | 2.4 h |
  - Decode is unchanged: plan + build `diff -r` identical against ac04a9f52's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08.
  - **klein-4B's VAE** lowers into the same passes (its resnets' shortcut convolutions leave the
    skip on the other side of the unit from the main branch: the region over both moves one
    across, an identity matvec, its tile the constant one on the diagonal, `stream::host::UNIT`)
    and is refused by the memory check: one unit holds its ~50M weights, 255k 16-bit words a
    weight PE (budget 15k).
  - `vae.encode` lowers too (`--clip 1,64,64`: 21 passes, its stride-2 convolutions' back
    padding as the grid rule states); compile on load derives its clip from token rows (4 x 4
    pixels) and refuses it.
  - Left:
    - Speed: the passes are overhead-bound (above). Rows a step batched into one op's vectors
      (a chunk times B rows), the store's fetch issued a row ahead (the weight PEs' heads are
      fetches alone), and a unit a resolution (pipelined between norms) are the levers; the
      cost model does not price grid programs yet.
    - klein's VAE: its passes over several units (a unit a block or a resolution), the images'
      store spread over many PEs (a 1024² image of 128 channels is ~256 MB) or the norms'
      statistics made in a pass of their own so fewer images are held.
    - The encoder's compile-on-load clip (the clip whose output is the denoiser's latent).
    - Co-residence on the simulator costs every reading the whole program's tiles: a load
      simulated end to end runs its readings apart (`stream_processes`).

- **M2c** (klein-4B's text reading at real size; grouped-query attention and several taps on the
  row stream; grid programs priced; the VAE's passes in weight groups; klein's end to end priced).
  - **Grouped-query attention in the stream attention** (klein's Qwen3-4B: 32 q heads over 8 kv
    heads of 128). A head root holds its kv head's q heads (`Qh = q / kv`), k and v; P7's PEs
    and root (`satt.csl`, `cy_satt.csl`) were already written over `Qh` q heads a kv head.
    What was missing, each generic:
    - **q and k turned together at different widths** (`rope(q, k)`, q 4,096 wide, k 1,024) is
      no region of one width: poem-compiler's `fuse::chain::lower_run_parts` cuts the run into a
      region a side (q's norm and rotation, k's), each the node's rotation of that side alone at
      the same positions, each a **head region** (a head's ops: its row a head) where every op
      is a head's. `lower_with` now refuses a q / k pair of unequal widths (it had silently read
      k at q's width). MHA runs (the miniature's, flux's) keep their one region.
    - **Head regions a head at a time** (`RegionAt::reps`): a region narrower than the values it
      reads runs over each of a head root's heads in turn (its heads' rows in, `row()`, each
      head's results into the root's buffer of them), its tables one head's.
    - **One module for q's and k's**: the two head regions are the same region (an RMS norm by
      a gain, a rotation at the row's position), so a root imports it once and each side copies
      its own gain in before its heads (`alike` in `emit::stream`); the rotation's P5 tables
      (12 KB at 512 rows) are held once, and a row's position scalar once (two regions that read
      the same scalar share its slot).
    - **K / V PEs' budget** (`build::kv_entries`): satt's per-query buffers grow with `Qh`
      (8 KB at 4 q heads, not 2.8): an attention PE keeps 19 entries, not 23 (at 23 cslc ran
      out of PE memory).
  - **Several taps** (klein projects layers 9, 18 and 27 into the context and sums them). The
    projection of an early tap is made at its layer's unit and read 9 and 18 units later:
    - **The row stream carries several values** (`build::carried`, `unit::lower_through`): a
      unit takes in, holds over its barrier and sends on a value a later unit of its classes
      reads, read or not; several values on the stream go in at a row's start and out at its
      end in one order (by value), a unit with a barrier sending after it. A row-stream input a
      buffer a value (`pin<v>`).
    - **Column values onto the rows**: a projection lands on the column roots; one leaving on the
      row stream, or a root region over values of both sides (the tap sum `part + total`), moves
      it across with an identity matvec (`unit::move_across`; its tile the constant one on the
      diagonal, `host::UNIT`, as the VAE's skips), the region then running on the rows.
      poem-compiler's stream lowering takes such an op on the rows (`spatial::layer`).
    - Every row-side value must cut evenly over the row roots (`build`: a `py` that does not is
      refused, so the search skips it). This found a latent bug: klein's 64-row denoise program
      took `py = 192`, under which the lane prologue's 256-wide timestep sinusoid cut to 1 column
      a root (256 / 192); the search now takes `py = 256` there. The 1024² program
      (`py = 256`) is byte-identical to 678668e24's.
    - A unit's host output rows count against its root's held-row budget (`place`): klein's
      last text unit holds its 512 rows of the context out and two values over the barrier, so
      its held rows go to the row store.
  - **The miniature** (`models/flux_2.poem`: `flux2-mini-gqa`, the miniature's DiT and VAE with a
    Qwen3 text encoder of hidden 256, 4 q heads over 2 kv heads, three layers, every layer
    tapped and projected by its column block of the context embedder as klein's are; flux2-mini
    and klein unchanged). Its goldens: `mk_tiny_flux2_pipe.py --gqa --text` (transformers'
    `Qwen3ForCausalLM`, hidden_states[1..3] through the context embedder); engine-cerebras
    `a_gqa_text_reading_taps_three_layers` (engine-cpu against transformers; `--ignored`: a
    fabric run's capture, program and rows). The program: 3 units, 6 K / V columns, 15 x 76 PEs
    at 16 rows; the row stream carries two values between its units (the residual and the
    taps' sum so far), each unit moving its tap's projection onto the rows (an identity matvec
    a unit).

    | text rows, "a red fox sleeping" (16 tokens) | vs engine-cpu | vs transformers |
    |---|---|---|
    | engine-cpu (`Forward::stream`, bf16 trunk) | | 8.85e-3 / 0.9999609 |
    | **simulator** (`stream/run.py`, capture-fed) | 8.66e-3 / 0.9999625 | **8.73e-3 / 0.9999620** |

    3.89M cycles of stamps (the first launch, start-up included), 38 min of simulator.
  - **klein's text reading at real size** (512 rows, the poem's `TE_MAX_TOKENS`): 27 units,
    216 K / V columns (8 a layer, 19 entries a PE, 27 PEs a column), 723 x 549 PEs (397k), `py`
    512; the largest root 33 KB (a head root: its kv head's 4 q heads and k through one head
    region, its P5 tables), a K / V PE 38 KB. **It links for the system fabric**
    (`cslc --fabric-dims=762,1172`): 8.3 min, about 22 GB of memory, 375 MB of ELFs, rc 0.
    240.9M cycles predicted (275 ms at 875 MHz); a layer's row is about 16.9k cycles before the
    barrier (the head roots' q and k norms and rotations, 4 + 1 heads at 3.3k) and 16.7k after
    (a query over 512 rows at 19 entries a PE and 4 q heads, 12.6k).
  - **Grid programs priced** (`cost::grid`; `wafer-compile stream --reading vae.decode --clip ..
    --price [--parts]`): a pass's rows at their slowest PE's work (the weight PEs' tile elements
    at P1's 0.55, the roots' regions by `RegionFeatures`, a gather's taps copied, a fetched or
    kept chunk, P7's store and query), plus a row's fixed cost (`ROW` = 1,150 cycles, fitted:
    the fetch's round trip up a chain, the gather's index math), for tail and head rows, and the
    norms' finals. flux2-mini's decode: 4.51M (2 x 2 latent; simulator 4.72M) and 75.2M (8 x 8;
    simulator 73.5M).
  - **klein's VAE decode at 1024²** (latent 128 x 128): 36 passes over 16 x 16 roots,
    **39.8G cycles, 45.5 s** priced (unplaced: its tiles do not fit). By resolution: 128² 2.1G,
    256² 3.8G, 512² 8.9G, **1024² 24.9G** (eight 1M-row passes at about 3k cycles a row: the
    row's fixed cost and its head row, not the matvecs).
  - **The VAE's passes in weight groups** (`grid::weight_groups`). Where a unit's weight PEs
    cannot hold every pass's tiles, the passes are cut into runs whose tiles fit beside the
    store (an attention's queries with its stores), each run's tiles from its first word; a step
    is a launch a run: the host writes the run's tiles (`w`, `group` in `host.json`) and its pass
    range (`grp`), the kernels sweep those passes alone (the K / V column runs in its run's
    launch), the stores stay in the weight PEs across launches, the rows are read back after the
    last (engine-cerebras `Reading::launch_with`; `Fabric::launch`, a launch reading no rows
    back, `wafer-fabric`'s `launch`). `wafer-compile stream --weight-words N` and the VAE fabric
    test's `READING_WEIGHT_WORDS` put a reading that fits in groups all the same. One group:
    every program as before (flux2-mini's three readings byte-identical).
    - **Simulator** (`flux_2_minis_vae_on_the_fabric`, the 2 x 2 drawn latent, `READING_WEIGHT_
      WORDS=2000`: groups [0, 9), [9, 19), [19, 28), three launches): 1.415e-2 / 0.99990192, PSNR 48.2 dB
      against engine-cpu, as the one-launch run (M1i), 4.72M cycles over the three launches, 9.2
      min.
    - **klein** does not build in groups yet: its mid-block attention's q, k and v are whole at
      CY root 0, so their three 512 x 512 matvecs' tiles sit in weight column 0 alone (49k
      words a PE at `P = 16`, 24.6k at 32): one pass over a PE's memory. And at `P = 16` one
      4,608 x 512 convolution's tile (9,216 words) does not fit beside the 8 x 8 latent's store.
  - **klein end to end at 1024²** (priced; one CS-3, readings in turn):

    | reading | footprint | predicted |
    |---|---|---|
    | text (512 tokens) | 723 x 549 PEs, links | 0.28 s |
    | denoise, 4 steps | 741 x 1143 PEs, links (M2b) | 4 x 1.52 = 6.06 s |
    | VAE decode | 16 x 16 roots, does not fit (above) | 45.5 s |
    | **total** | | **51.8 s** |

    **The bottleneck is the VAE decode** (88%), and in it the 1024² passes' rows: a single
    unit sweeps 1M rows a pass at about 3k cycles a row, overhead-bound. Text and denoise do not
    fit the fabric together (397k + 847k PEs of 883k), so a request also switches programs
    (their weights rewritten from the host: not priced; the host's bandwidth is unmeasured).
  - Decode is unchanged: plan + build `diff -r` identical against 678668e24's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08. flux2-mini's text,
    denoise (16 and 64 rows) and VAE decode programs byte-identical; klein's denoise at 1024²
    byte-identical.
  - Left:
    - **The VAE at 1024²** builds and links nowhere yet. Each a blocker: the tiles (weight groups
      over a larger `P`, or a unit a group of passes side by side with the held values handed
      between units); the attention's q / k / v whole at one root (cut them evenly and gather
      them to the K / V root, or a K / V column fed from every root); the store (a 1024² value of
      128 channels is 256 MB: a held value as a tape of PEs beside the unit, or streamed through
      the host during a pass); the line buffer (2 x 1,024 + 3 rows of a root's chunk, 128 KB at
      `P = 16`: `P = 32` and an exact ring, or column strips); the 16k-row attention's K / V
      column (1,509 PEs, past the fabric's height: folded over columns); a pass's 1M rows past
      the programs' 16-bit words.
    - **VAE speed**: the passes are row-overhead-bound. Rows batched into each op's vectors (a
      chunk times B rows: the fixed cost a row / B; the cost model with `ROW / B`: klein's 1024²
      decode 45.5 s -> 25.0 s at B = 4, 21.6 s at 8, the head rows' ops and the matvecs left),
      more roots (`P = 32`: the matvecs' tiles a quarter, the chunks half), the fetch issued a
      row ahead, and row replicas of the unit over strips of the image are the levers; none is
      built (item 4 of M2c, the miniature's batching, was not reached).
    - The text reading at real size on the simulator (only the miniature's ran); the head
      roots' rotation tables a row at every head root (the same angles for its 5 heads; 3.3k
      cycles a head) are the text's cost.

- **M2d** (the tiled regime: a grid reading's image spread over the fabric; klein's VAE decode at
  1024² builds and links, priced at 0.13 s).
  - **Why.** The rows regime (M1i) sweeps a clip's rows through one unit, a row at a time, so
    a pass costs its rows times a row's fixed cost (~3k cycles): klein's 1024² decode is 1M-row
    passes, 45.5 s. A convolution is a GEMM of pixels x (9 · c_in) x c_out: what the wafer wants
    is the pixels over its PEs, each multiplying its own pixels by weights that pass by.
  - **The regime** (kernels-cerebras `stream::tiled`, `emit::tiled`, `cost::tiled`; `wafer-compile
    stream --reading vae.decode --regime tiled [--tile h,w] [--price --parts]`). The reading is
    lowered off its trace (the rows regime's prelude: its class, the host's nodes, the grids) into
    **ops over the whole image**, bulk-synchronous (an op finishes everywhere before the next reads
    its output): a convolution (3 x 3 or 1 x 1 "same" at stride 1; a `linear.matmul` the same op
    with one tap; its bias its own), a group norm, a printed region (every other elementwise run,
    `fuse::chain::lower_run`, printed by `emit::region` at a chunk of 8 columns of a pixel), an
    upsampling and a pixel shuffle (an axis at a time: a move, the shuffle's half each child's
    every other channel), the attention. No op is named: each lowers by its family; what has no
    lowering (a stride-2 convolution, an unshuffle on the wafer) is refused by name. Values live
    where they are made, f32 (rounded to 16 bits where the trace types them so), each a slot of
    a PE's arena shared by lifetime (first fit over the ops' intervals).
  - **Data layout per resolution.** One **image array** of `gx x gy` PEs (powers of two). At an
    `h x w` resolution each axis on its own: `h >= gy` gives each PE a tile of `th = h / gy`
    rows; `h < gy` a **stride** `sy = gy / h` (the PEs of rows `0, sy, 2sy, ..` hold a pixel row
    each; the rest pass halos and moves through). Strides nest, so an upsampling by two is local
    (a tile doubles) or a move of `sy / 2` PEs, and a 3 x 3 halo is the next busy PE's border,
    `sy` PEs off. Every PE holds all its pixels' channels (no channel split: below). The search
    (`tiled::program`) prices every power-of-two top tile whose array, memory and fabric fit
    (`Bounds::system()`: 755 x 1170) and keeps the cheapest. klein at 1024² (latent 128²): the
    top tile 1 x 2, **gx x gy = 512 x 1024** (the fabric is taller than wide):

    | resolution | channels | th x tw | stride (sy, sx) | busy PEs | a PE's pixels |
    |---|---|---|---|---|---|
    | 64² (the latent's standardisation, the host's unshuffle) | 128 | 1 x 1 | 16, 8 | 4k | 1 |
    | 128² (conv_in, mid block, up 0) | 512 | 1 x 1 | 8, 4 | 16k | 1 |
    | 256² (up 1) | 512 | 1 x 1 | 4, 2 | 65k | 1 |
    | 512² (up 2) | 512 -> 256 | 1 x 1 | 2, 1 | 262k | 1 |
    | 1024² (up 3, conv_out) | 256 -> 128 -> 3 | 1 x 2 | 1, 1 | 524k | 2 |

  - **The weights pass by.** The VAE is small (klein's decoder: 49.5M weights, 99 MB), so they
    are not resident where they are used: they stream once a launch from a **tape** (PEs below
    the array, each holding 10,000 words of the reading's records in op order: `statics` spans
    over the checkpoint, two 16-bit weights a word, the new span flag `PACK`; klein 2,483 PEs,
    the mini 12), up a **spine** column and east along every row (a multicast tree: every PE sees
    every word, in lockstep with the slowest). A tape PE sends its words, then forwards those of
    the PEs past it. A convolution stages its input 16-bit a chunk of up to 128 channels at a
    time (with its halo), then receives each tap's weights a block of `kb` input rows x `c_out`
    (512 words) and runs `ops/mv.csl`'s `mac32` per pixel (the block the tile, the pixel's `kb`
    staged inputs its x), f32 accumulation in the output's slot; a bias, a norm's gamma and beta,
    a region's resident are f32 records on the same tape.
  - **Halos.** E / W first along busy rows, then N / S whole halo rows (the corners ride along),
    two 16-bit values a wavelet. An exchange is paired by parity (blocks of parity 0 send while
    those of parity 1 receive, then the reverse), so every send is synchronous and no PE waits
    on one that waits on it; a PE between two busy ones forwards (`mov32` fabric to fabric).
    Zeros past the image's edge (the padding).
  - **Group norm**: two **all-reduces** of `G` sums (the means, then the centred squares), each
    along every row east (`fadds` from the fabric into the fabric), down the last column, from
    the corner to a **hub** PE east of it, which sends it back over a broadcast tree (west along
    the rows from the last column). Then the gammas and betas off the tape, a channel's scale and
    shift, the apply (and SiLU) a printed region.
  - **Attention** (`attn.voxel` at the lowest resolution): every key's [k | v] (16-bit, packed)
    goes east along its row, down the last column to the hub and back over the broadcast tree, in
    one order every PE follows (a PE forwards the blocks passing through it, then takes the
    block off the broadcast: no PE waits on a block another holds back); each PE runs its
    queries over every key as it passes (an online softmax, f32: a running max and sum a query).
  - **Routing / colours** (12 of 21; a pattern of the array laid by comptime loops in
    `layout.csl`, not Floorplan links):

    | class | colours | routes |
    |---|---|---|
    | shifts E / W / S / N | 2 each (8) | a PE sends on its parity's (`x`, `y` for S / N), receives its neighbour's on the other |
    | weight tree | 1 | tape 0 → the spine (column 0, north) → every row east, to every PE's ramp |
    | broadcast | 1 | the hub → the corner → up the last column → west along every row |
    | tape | 2 | tape PE `k` sends on `parity(k)` to `k - 1` |

    An image PE's code depends on its parities alone (4 images); its place is two config words.
    klein: 212 ELFs.
  - **Memory a PE at klein 1024²** (the linked ELF): code 20.1 KB, the program's tables 3.6 KB
    (100 ops; a layout an index into a table of 9), data 19.6 KB: the arena 3,200 f32 words
    (12.8 KB; its peak at the 128² resnets' norms: three 512-wide values and the norm's scratch,
    and at the mid block's attention: the residual, q, k, v, o and a key's buffer), the staging
    buffer 1,536 16-bit words (3 x 3 x 128 channels + halo, 3 KB), the weight block 512 words
    (2 KB), the printed regions' chunk buffers (6 modules), the host's latent; with the task table
    **44.2 KB of the 48**.
  - **Cycles from first principles** (the per-op roofline; `cost::tiled`). Every PE in lockstep
    with the tape, an op costs its busiest PE's work plus what passes every PE. A convolution:
    its blocks' words received (two weights a wavelet) then multiplied (0.55 an element, `P`
    pixels a PE), not overlapped (a synchronous receive): `≈ W · (0.5 + 0.55 P)` cycles for `W`
    weights, plus its staging and halos. klein: 128² 24.7M weights at `P = 1`, 256² 16.5M, 512²
    6.6M, 1024² 1.65M at `P = 2`. The attention: 16k keys past every PE, each about 3.3k cycles
    (forwarded, received, widened; a query's product, halving sum and exponents).

    | klein 1024² (priced) | cycles |
    |---|---|
    | convolutions (40: the 3 x 3 ones, the shortcuts, the attention's four projections) | 57.2M |
    | attention (128², 16k keys of 512) | 54.4M |
    | group norms (30), regions, moves | 1.1M |
    | **the decode** | **112.7M, 0.129 s** (rows regime: 45.5 s, unplaced) |

    By resolution: 128² 82.9M (the mid block's attention and the 512-channel resnets), 256²
    19.0M, 512² 7.7M, 1024² 3.1M.
  - **On the simulator** (engine-cerebras `flux_2_minis_vae_on_the_fabric` with
    `READING_REGIME=tiled READING_TILE=2,2`: the pipeline's 8 x 8 latent, 64 x 64 pixels, a
    32 x 32 array (the 8² stage at a stride of 4, 64² two by two a PE), 34 x 33 PEs, 12 tape
    PEs; the static arrays from the checkpoint, the entries from the engine, as M1i):

    | flux2-mini's decode | vs engine-cpu (rel L2 / cosine, PSNR) | vs diffusers | cycles | simulator |
    |---|---|---|---|---|
    | rows regime (M1i) | 1.03e-2 / 0.99994750, 51.0 dB | 1.16e-2, 49.8 dB | 73.5M | 2.4 h |
    | **tiled** | **9.99e-3 / 0.99995013, 51.2 dB** | **1.13e-2 / 0.99993657, 50.1 dB** | **0.905M** | 12.9 min |

    - Every prefix (`READING_PASSES=N`: the first N ops, the last one's value read out) against
      the program on the host (`tiled::emulate`): bit for bit to 7 ops (the latent in, its
      standardisation, the shuffle's two moves, a 1 x 1 and a strided 3 x 3 convolution, a norm
      through the hub); after the attention (19 ops) 3.7e-4 (its f32 softmax against the
      emulator's f64), 2.5e-3 at 27, 3.8e-3 at 36 (the norms after it amplifying it, as in M1i).
    - On the host (`flux_2_minis_tiled_vae_program_is_the_reference`): the program vs engine-cpu
      6.5e-4, 75.3 dB; vs diffusers 50.4 dB (engine-cpu's own: 50.4).
      `flux_2_minis_tiled_vae_reading_serves_on_the_host_fabric`: its host side (the latent on
      its layout's busy rows, the pixels read back a rectangle at a time) bit for bit.
    - Two simulator lessons: a `mov32` forward of `mov16` sends hangs the PE (halos and keys go
      two 16-bit values a wavelet since); a fabric DSD of 32,768 wavelets stalls (the tape and
      the hub forward 16,384 at a time).
  - **The cost model against the simulator.** The mini's prefixes (2, 4, 7, 19, 27, 36, 78 ops:
    31k to 905k cycles) fit the priced terms plus a fixed 5.1k cycles an op (`cost::tiled::OP`:
    the array's lockstep between ops; conv 5.1k, norm 5.1k, region 6.0k, move 4.3k fitted apart)
    and a launch's 17k, within 3% at 19 ops and more. The mini's ops are small (32 channels), so
    the fit checks the fixed terms; klein's are its words and MACs (0.55 an element and a wavelet a
    cycle, the stream regime's measured rates), the tree's and the tape's throughput at klein's
    size not measured.
  - **klein-4B's VAE decode at 1024²** (latent 128², the search's array 512 x 1024): compiles in
    2 s (89 MB: `layout.csl` 0.97 MB of loops, the tape's spans), **links for the system fabric**
    (`cslc --fabric-dims=762,1172`): 514 x 1029 PEs, 212 ELFs, 226 s, 9 GB of memory at its peak,
    500 MB of ELFs, rc 0 (the rows regime's did not place). **113.2M cycles, 0.129 s** priced.
  - **klein end to end at 1024²** (priced; one CS-3, readings in turn):

    | reading | footprint | predicted |
    |---|---|---|
    | text (512 tokens) | 723 x 549 PEs, links (M2c) | 0.28 s |
    | denoise, 4 steps | 741 x 1143 PEs, links (M2b) | 4 x 1.52 = 6.06 s |
    | VAE decode | 514 x 1029 PEs, links | **0.13 s** (was 45.5 s) |
    | **total** | | **6.47 s** (was 51.8 s) |

    The denoise is 94% of it now.
  - **The regime by price** (`stream::compile_grid`, what compile on load calls): the tiled
    program (its top tile searched) and the rows one, the cheaper of those that place within a
    system: flux2-mini's decode the tiled (0.9M against 74.9M cycles), klein's the tiled (the
    rows does not place). The rows regime stays (`grid::compile`, `wafer-compile stream
    --reading vae.decode` without `--regime`, its programs byte-identical) and is still chosen
    where it is cheaper or the tiled refuses (`vae.encode`'s stride-2 convolutions). Co-resident
    readings (`stream::coresident::combine`) move a tiled program's loops by its origin.
  - **The engine** (engine-cerebras `Reading`): a `d2h` item may name a rectangle of PEs read
    back alike (`w`, `h`: one memcpy, its rows in the rectangle's order). kernels-cerebras
    `stream::host`: the span flag `PACK` (two 16-bit elements a word, pairing across an array's
    spans), `span_words`; the rows regime's host side (`grid::host_arrays`) shared.
  - Decode is unchanged: plan + build `diff -r` identical against e8d0c8935's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08. flux2-mini's text
    (16 rows), denoise (16 and 64 rows) and rows-regime VAE decode programs and klein's 1024²
    denoise byte-identical.
  - Left:
    - **Speed**, in order: the weight receive double-buffered (the convolutions' 57M -> about
      30M: a block's receive overlapping the last one's MACs); the attention (54M: 16k keys past
      every PE while 16k of 524k hold a query: the keys split over a query's idle neighbours, each
      a slice merged by its running max and sum, or the scores a 16-bit tile matvec over blocks
      of keys); at the low resolutions a channel split (a pixel's output channels over `S`
      PEs, `S` trees and tapes: the weight stream's floor, `W / 2` words, divided by `S`).
    - A reading switch rewrites the tape's 25M words (the host's bandwidth unmeasured), as the
      text and denoise programs' weights are.
    - `vae.encode` in the tiled regime (stride-2 convolutions with back padding, the latent's
      unshuffle on the wafer); a clip other than a square of powers of two.
    - `pie run text-to-media` on the simulator with the tiled decode (compile on load now
      picks it; M1i's run used the rows regime); klein's VAE on a system.

- **M1j** (flux2-mini on the CS-3: the stream, tiled and rows programs on a system for the first
  time; requests 1458-1468, one session). Each program linked on the worker by its own cslc
  against the system's whole fabric (762 x 1172, offsets 4, 1), the same CSL as the simulator's.
  - **The denoise loop, standalone** (`wafer-bench --reading --grid 1,8,8 --steps 2`, the
    engine's inputs and engine-cpu's goldens from `flux_2_minis_reading_files`; linked in 10 s):

    | | 16 + 16 (77 x 28, request 1458) | 64 + 64 (80 x 28, request 1460) |
    |---|---|---|
    | step 0 velocity vs engine-cpu | 4.85e-3 / 0.9999882 | 4.89e-3 / 0.9999881 |
    | step 1 velocity vs engine-cpu | 6.36e-3 / 0.9999798 | 6.01e-3 / 0.9999820 |
    | latent after 2 vs engine-cpu | 3.86e-3 / 0.9999925 | 3.57e-3 / 0.9999936 |
    | latent after 2 vs diffusers | **3.58e-3 / 0.9999936** | **3.49e-3 / 0.9999939** |
    | step 0, the first launch | 1.91M cycles, 3.2 ms | 5.93M cycles, 9.3 ms |
    | a re-launched step (7 of an 8-step loop) | **1.33-1.35M cycles, 2.37-2.45 ms** | **5.34-5.39M cycles, 8.5-8.6 ms** |
    | the cost model (`--price`) | 1.2M | 5.0M |

    - At 16 + 16 the velocities and latents are the simulator's (M1f's `wafer-bench --save`)
      bit for bit.
    - The re-launched step's stamps are the simulator's 1.27M plus 6%, the cost model's plus
      11% (7% at 64 rows). The first launch's start-up is the simulator's artifact: 7.90M
      cycles there, 1.91M on the system.
    - A step's wall (its entries written, the launch, the rows read back) is its stamps at the
      PE clock, 750 MHz, not 875, plus the host's copies (M1k).
  - **The text reading and the denoise loop through the engine** (`flux_2_minis_text_on_the_fabric`
    with `READING_CMADDR`, request 1462): text, denoise and VAE decode co-resident in one
    program (153 x 65 PEs), compiled and linked on load on the worker (16 s), loaded in 37 s.
    The text rows: 7.86e-3 / 0.9999691 against engine-cpu, **8.74e-3 / 0.9999619 against
    transformers** (the simulator's figures), 2.03M cycles. The guest's 2-step loop over them:
    step 0 10.7M cycles (the co-resident program's first denoise launch; 130 ms with its static
    arrays made from the checkpoint), step 1 1.33M cycles, 2.6 ms; the latent **3.88e-3 /
    0.9999925 from diffusers' pipeline**, as on the simulator.
  - **The VAE decode** (`flux_2_minis_vae_on_the_fabric` with `READING_CMADDR`, request 1462):

    | flux2-mini's decode on the system | vs engine-cpu | vs diffusers | cycles |
    |---|---|---|---|
    | tiled (34 x 33 PEs) | 9.99e-3 / 0.99995013, 51.2 dB | 1.13e-2, **50.1 dB** | 1.09M (simulator 0.905M) |
    | rows regime (9 x 19 PEs) | 1.03e-2 / 0.99994750, 51.0 dB | 1.16e-2, 49.8 dB | 77.1M (simulator 73.5M) |

    Both the simulator's numerics.
  - **`pie run text-to-media` on the CS-3** (request 1468; `--prompt "a red fox sleeping"
    --width 64 --height 64 --steps 2 --seed 7 --guidance 4`, `[engine] cmaddr` the system's,
    `stream_rows = 16`, the checkpoint `mk_tiny_flux2_pipe.py --ckpt`'s): one `pie` process
    compiles the three readings, co-resident, links them for the system (17 s) and serves the
    inferlet's text pass, 2 denoise steps and the tiled VAE decode on the wafer; a 64 x 64 PNG,
    **48.4 dB against diffusers' pipeline** (max |diff| 4/255; the simulator's M1i run, rows
    regime decode, 48.3; 49.5 dB between the two).

    | step | cycles | fire |
    |---|---|---|
    | text (16 tokens; the program's load) | 2.06M | 37.1 s (36.7 s the load) |
    | denoise step 0 (first launch, static arrays made) | 10.7M | 81 ms |
    | denoise step 1 | 1.33M | 2.5 ms |
    | VAE decode, tiled (static arrays made) | 1.07M | 14.6 ms |

  - Left: the first launch of a co-resident program's denoise (10.7M cycles against 1.9M
    standalone) and the 37 s load (M1k); the 64-row and klein programs through the engine on the
    system.

- **M1k** (the CS-3's clock, a step's wall split, the first launch; requests 1470-1480).
  - **The PE clock is 750 MHz.** A spin kernel (`fmuls` over 256 words, 267 cycles a round) timed
    against host wall over 0.5 to 2.9 s spins: 750.0 MHz on one PE (requests 0119, 1470) and on
    every PE of 80 x 28, 400 x 400 and 750 x 990 rectangles at once (742,500 PEs, 1472; the
    slope between the 2M- and 8M-round spins 750.0 MHz at every size): no throttling under load.
    Every "ms at 875 MHz" above is 7/6 of it. A stream kernel's `run` restarts its PE's TSC
    (`enable_tsc`), so a launch's stamps are each PE's own time since its launch arrived (every
    start reads 10-11) and a launch's cycles the longest PE's.
  - **Decode agrees.** Llama 3.1 8B's pin with `TSTAMP:1` (request 1478): the token loop's
    stamps 353.5k cycles a step (median of 31), 0.4713 ms at 750 MHz; the bare step at the pin
    0.4711 ms (1466). At 875 MHz the stamps would say 0.404.
  - **A denoise step's wall, split** (wafer-bench `--reading`, the runtime's copies timed by
    kind; then the same reads again with nothing running, for their own cost; 1470, 1476):

    | re-launched step | 16 + 16 | 64 + 64 |
    |---|---|---|
    | entries in (9 copies) | 0.13 ms | 0.13 ms |
    | the launch call (returns at once) | 0.01 ms | 0.01 ms |
    | on the wafer: the stamps at 750 MHz | 1.328M, 1.771 ms | 5.34M, 7.12 ms |
    | the rows back (their own cost) | 0.38 ms | 0.95-1.10 ms |
    | the stamps back (77 x 28 x 3 words) | 0.20 ms | 0.20 ms |
    | **wall** | **2.40-2.45 ms** | **8.4-8.6 ms** |

    The parts sum to the wall within 3% (the rows' copy waits on the program's end, so it
    overlaps the tail). The same for the VAE decode in one launch: the rows regime 77.05M cycles,
    102.7 ms at 750 MHz, 107.6 ms with its entries and pixels; tiled 1.107M, 1.48 ms, 2.1 ms.
    The cost model's seconds now take the clock from one constant (`cost::CLOCK_HZ`): the decode
    model's stamped rates (a weight and a head element, the max-reduce's cycle) at 750 MHz, the
    chains' fitted rate 0.82 of a cycle; no plan or build changes (plan + build `diff -r`
    identical for qwen35-d0.8b, qwen35-d9b, gptoss-20b, gemma4-e2b and tiny t08; Llama's is its
    G2=11 pin's).
  - **The first launch: the static arrays still landing.** The runtime's host-to-device copies
    return before their words are on the PEs (flux2-mini's 6.6M words of denoise arrays: 107
    copies in 11.2 ms, landing about 12 ms more). The engine wrote a reading's arrays at its
    first step, just before its launch, so 240 PEs (rows 18-27 of the K / V and exit columns)
    waited 8.96M cycles (11.9 ms) on them: the co-resident program's first denoise launch 10.8M
    cycles, the second 1.33M. With 300 ms between the write and the launch, the first launch is
    1.33M standalone and co-resident alike (1474); the standalone runs' 1.9M first launches were
    the same wait, shorter. The engine now makes every reading's static arrays at load (they are
    the weights' alone: made from a step of no inputs, word for word the first step's) and
    opens a wafer's program at load, writing every co-resident reading's arrays at once (the
    host fabric still writes a reading's at its first step). Through the engine on the system
    (1480):

    | | before (1462, 1468) | at load (1480) |
    |---|---|---|
    | load (the program loaded and its arrays written) | | 36.8 s |
    | text step | 2.04M cycles, 37.1 s (the load) | 0.55-0.58M cycles, 58-110 ms |
    | denoise step 0 | 10.7M cycles, 81-130 ms | **1.33-1.35M cycles, 2.7-2.9 ms** |
    | denoise step 1 | 1.33M cycles, 2.5 ms | 1.33M cycles, 2.5-2.7 ms |
    | VAE decode (tiled) | 1.07M cycles, 14.6 ms | 0.756M cycles, 1.8 ms |

    `pie run text-to-media`'s PNG is 1468's byte for byte (48.4 dB from diffusers).
  - **The load.** `SdkRuntime::load` on the system takes 26 s for the 77 x 28 denoise program
    alone and 36 s for the three co-resident (153 x 65): the SDK's, not the engine's.
  - Left: the text step's 58-110 ms past its 0.77 ms on the wafer (the host side of a reading's
    first fire); the stamps read back every step (0.2 ms, 8% of the 16-row step) only where the
    cycles are wanted; the rows' copies (0.38 ms for 2k words).

- **M2e** (klein's denoise step: the rotation tables a level at a time, the K / V columns folded,
  the cost model held to the simulator unit by unit; 1,326M -> 1,081M cycles a step priced, 1.77 s
  -> 1.44 s at the PE's 750 MHz (M1k), end to end 7.54 s -> 6.21 s; what the stream regime's
  floor is and what reaches the roofline). Seconds here are at 750 MHz.
  - **The roofline.** A klein step at 1024² (4,608 rows: 512 text, 4,096 image) is 17.4T MACs
    (34.8 TFLOP): the blocks' matmuls 14.1T (a double block's side runs its own weights over its
    own rows: 122.7M weights x 4,608 rows a block, 2.83T over the five; the 20 single blocks
    11.3T), the joint attention 3.26T (25 blocks x 24 heads x 4,608² x 256). The wafer's 883k
    PEs at P1's measured 0.55 cycles a MAC (16-bit operands, f32 accumulation: the DSD's write
    port) do 1.6M MACs a cycle: **10.9M cycles, 14.5 ms a step**. The weights (3.7B, 7.4 GB, 17%
    of the SRAM) are resident, so no memory bandwidth enters, and a row's 3,072 values and a
    column's partial sums a row are far under a link's wavelet a cycle.
  - **Where the time goes.** The barrier serialises the blocks: a step is about 25 blocks x
    4,608 rows x one unit's pace a row, and a unit holds a 30th of the weight PEs, so about two
    units (34k weight PEs) and a block's K / V PEs work at a time. M2b's 1,326M cycles were
    priced as each unit's busiest PE (its column roots' steps summed); the simulator (below) shows
    what a unit's pace is: **a row's round trip**. A unit's weight PEs and roots finish a row
    before they take the next (their chains synchronous, `ops/schain.csl`), so a row costs its
    path through the unit's steps, CX region, matvec, column region, matvec back, until what
    returns to its row roots, or one PE's own work a row where that is longer.

    | a row of (klein, 1024²) | M2b | M2e |
    |---|---|---|
    | a head root's q / k norms and four-axis rotation (24 roots, chunk 128) | 8,793 | 5,216 |
    | a query (P7, 47 -> 24 entries a PE) | 8,197 | 4,573 |
    | the SwiGLU on the rest roots | 1,940 (15 helpers; the double blocks' text side 10,921, none) | 1,714 (23 helpers, every unit) |
    | a single block before / after the barrier | 11,240 / 8,197 (M2b's price) | 8,164 / 5,484 |
    | a double block's image side before / after | 9,301 / 10,137 | 5,724 / 13,664 |
    | **the step** (each program priced by M2e's model) | **1,325.9M, 1.77 s** | **1,081.4M, 1.44 s** |

    A single block's row before the barrier is now its round trip (the CX roots' layernorm and
    modulation 748, the in_proj's busiest column 3,802, the SwiGLU 1,714, the MLP's out matvec
    1,901); a double block's after it, two (the query, out_proj, the gated residual, layernorm
    and modulation at the CX roots, the MLP in, the SwiGLU, the MLP out).
  - **The rotation tables a level at a time** (`ops/srope.csl`, stream only; decode's `rope.csl`
    unchanged). P5's cos / sin of a position are the product of LV levels of its digits' table
    rotations, recomputed whole every row. srope now keeps each level's product and each call's
    last position: a row recomputes only the levels from the highest digit that changed, from
    the kept product above it, by the same products in the same order, so the cos / sin are bit
    for bit the whole computation's; a position that holds costs one compare. Rows whose
    positions advance by one a row (a raster of cells, a token index) recompute about one level
    of one axis a row; flux's image rows hold their t and l axes, its text rows all but l.
    `prims/reg` (the mini's per-head RMS norm and four-axis rotation, nodes 69, 70: 2 roots x
    128, LV 4, BITS 3, 64 rows, on the simulator):

    | srope | positions | cycles a row of `row()` | vs interp |
    |---|---|---|---|
    | M2a's (every level every row) | the step's | 7,011 | 9.27e-7 |
    | a level at a time | the step's (one axis advancing) | **3,435** | 9.27e-7 (the same words) |
    | a level at a time | every position held | 2,821 | |
    | a level at a time | shuffled (every level of every axis a row) | 7,663 | |

    The cost model: a call `ROT_CALL` = 170, a row whose position moved `ROT_CHANGE` = 429 and
    `ROT_LEVEL` = 152 a level, `2^B / (2^B - 1)` levels a row at `B` bits a digit (one axis
    advancing a step a row), once a row however many heads a root's region runs over; fitted to
    the three rows above. (The mini's own tables are two levels deep, so its step gains little:
    below.)
  - **K / V columns over several columns a head** (`floorplan::stream`, `build::KV_SPLITS`).
    A K / V column was one PE wide (99 PEs a head at klein's 4,608 rows, 47 entries each);
    M2b's fold over two columns was refused, a head root's link to a column two wide crossing the
    other heads' columns (a 48-class clique, past the palette's 21 colours). Now at `sub > 1`:
    - a block's first unit lays its head columns `sub` apart (`StreamUnit::col`: head `h` at
      `h sub`, the rest roots in the columns between and after; its row chains run west to
      east, and its weight PEs' row-chain roles, a column-side matvec's leaf and the row store's,
      read that order: `emit::stream`), so each head's K / V column is a `sub`-wide snake
      straight below its head root;
    - every K / V root sits on one row below the later units' turns and the helpers
      (`Stream::turns`, `KvColumn::root`): the first unit's links go straight down, a later
      unit's turn west on its head's row and down the head root's column (`KvColumn::link`); no
      link passes a K / V column, and the links meet as before (a later unit's turn crosses the
      first unit's later heads' links: a bipartite graph);
    - a block's first unit takes helpers too (the later units' turns pass over them, not over
      K / V columns).

    The search tries `sub` 2 and 3 at `1 / sub` of the entries a PE (the band's rows as before,
    `sub` times the PEs), with 0 to 23 helpers: klein takes `sub = 2` (198 PEs a head, 24
    entries each: a query 4,573 cycles), 23 helpers, 741 x 1152 PEs.
  - **The cost model, held to the simulator** (`cost::stream::time_at`). A phase's pace is the
    longest of: a row's path through its steps (each value's time in the row, a step starting
    when what it reads has landed, a matvec its busiest column's tile; a row waits on what comes
    back to its row roots, not on a dead end such as a store), each column root's own steps a
    row (the head roots' and the rest roots' apart), the row roots' regions, the busiest weight
    column; an f32 input's matvec (`X32`) two passes. The simulator's stamps (each root kernel
    stamping the end of each schedule segment, a local instrumentation, not committed) on
    flux2-mini's re-launched step (16 + 16 rows, K / V over 2 columns, no helpers), a row:

    | unit | measured | priced |
    |---|---|---|
    | double blocks after the barrier, image side | 11.6k | 11,957 |
    | double blocks after the barrier, text side | 12.5k - 13.1k | 11,957 |
    | single blocks before the barrier | about 10.3k | 10,143 |
    | single blocks after the barrier | 2.05k - 2.3k | 2,919 |
    | the entries (their f32 host rows) | 5.3k - 8.2k | 1,126 - 1,690 |

    The entries are short (their fill, 0.1-0.15M cycles of the mini's step, is not priced);
    every unit after them within 5-10%. M2b's sum and M2e's first price (each unit at its
    busiest root, 0.73M for the mini) both miss the round trip: M2b's matched the mini only
    because its head roots' sum stood in for it.
  - **On the simulator** (flux2-mini, 16 + 16 rows, the 2-step loop from the engine's inputs,
    `wafer-bench --reading --steps 2`; rel L2 / cosine against engine-cpu):

    | program | step 0 velocity | step 1 velocity | latent after 2 | step 1 cycles | priced |
    |---|---|---|---|---|---|
    | M1g's (bcad836bb, py 16, 7 helpers) | 4.853e-3 / 0.9999882 | 6.358e-3 / 0.9999798 | 3.861e-3 / 0.9999925 | 1,265,798 | 1.21M (M2b) |
    | the same pins, srope a level at a time | the same digits | the same digits | the same digits | **1,226,968** | 1.079M |
    | K / V over 2 columns (no helpers: before `turns`) | the same digits | the same digits | the same digits | 1,426,064 | 1.319M |
    | K / V over 2 columns, 7 helpers (M2e's) | the same digits | the same digits | the same digits | **1,126,817** | 0.961M |

    The velocities and latents agree with M1g's to every printed digit: the tables are bit for
    bit, and the fold moves PEs, not arithmetic. The mini's tables are two levels deep and its
    head roots off its round trip, so the tables alone gain 3%; at 16 + 16 rows its K / V PEs hold
    4 entries, so the fold's own gain there is the helpers it lets the first units take (the text
    side's SwiGLU after the barrier): 1.27M -> 1.13M cycles a step, 11%. Without the corridor
    below the helpers (`turns`) a fold of the mini's 2 heads left no room for them (1.43M). Each
    run is the priced step plus the entries' unpriced fill (0.12-0.17M).
  - **klein-4B at 1024²** (the search's choice: `py 256`, tile 12,288, K / V over 2 columns a
    head, 23 helpers): **1,081.4M cycles, 1.442 s a step** (M2b's program under the same model
    1,325.9M, 1.768 s). By lever: the tables a level at a time 1,325.9M -> 1,292.5M; the K / V
    fold, with helpers under the first units, -> 1,106.4M; 23 helpers -> 1,081.4M. **It links
    for the system fabric** (`cslc --fabric-dims=762,1172`): 1,264 ELFs, 965 s, 36 GB at its
    peak, 821 MB, rc 0.
  - **Text** (klein's text reading at 512 tokens, the same srope and fold): 239.4M -> 218.0M
    cycles, 0.319 -> 0.291 s (723 x 559 PEs; M2c's program priced by M2e's model 239.4M, by its
    own 240.9M). It links for the system fabric: 387 ELFs, 603 s, 16 GB, 382 MB, rc 0.
  - **klein end to end at 1024²** (priced; one CS-3, readings in turn):

    | reading | footprint | M2d (repriced at 750 MHz) | M2e |
    |---|---|---|---|
    | text (512 tokens) | 723 x 559 PEs | 239.4M, 0.32 s | 218.0M, 0.29 s |
    | denoise, 4 steps | 741 x 1152 PEs | 4 x 1,325.9M, 7.07 s | 4 x 1,081.4M, 5.77 s |
    | VAE decode | 514 x 1029 PEs (unchanged) | 113.2M, 0.15 s | 0.15 s |
    | **total** | | **7.54 s** | **6.21 s** |

  - **What is left, in order of what it buys** (klein, a step):
    - **Rows in flight in a unit** (1.44 s -> about 0.91 s, the step with every unit at its
      busiest PE, M2e's first price: 680M). A unit's PEs take row `t + 1` before row `t`'s round
      trip returns: a software pipeline of each phase at its round trips (a weight PE's matvecs
      of row `t` and the second half of row `t - 1`'s; a column root receiving row `t`, casting
      row `t - 1`'s result, then running row `t`'s region; the CX roots a region ahead), every PE
      in one order of its chains' messages so the synchronous sends still meet. An emitter
      change (each kernel's `early` / `late` cut into stages) and a deadlock question for the
      simulator; not started.
    - **The stream regime's floor**: with rows in flight, every unit's pace is its weight PEs'
      (a rest column's 10,368 elements at 0.55: 5.7k) or its head roots' (5.7k): 25 x 4,608 x
      5.7k = 657M, 0.88 s, 60x the roofline. One unit (a 30th of the weight PEs) works a row at
      a time, and the fabric is full (741 x 1152 of 755 x 1170): no row replicas, no wider units.
    - **A tiled regime for the DiT** (the roofline's way): every PE on the current layer, the
      step's rows and channels both cut over the fabric (a SUMMA-style GEMM: PE `(r, c)` holds
      `Y[T / R rows, N / C cols]`, the activations' panels multicast along PE rows, the weights
      stationary a column of PEs and multicast down it), norms as row all-reduces, the attention
      two such GEMMs a head with an online softmax. At `R x C` = 768 x 1,150 a single block's
      in_proj is 6 rows x 24 columns x 3,072 MACs a PE, 243k cycles, its inputs (92k values a
      PE) at a wavelet a cycle under that; a PE's share of a step is 19.7M MACs, 10.8M cycles,
      about 15 ms with the norms and exchanges on top. M2d's tiled regime is its VAE form (a
      pixel's channels whole at a PE, the weights from a tape); the DiT needs the channel split
      M2d left open.
    - The entries' fill priced (their f32 host rows on the simulator: 5-8k cycles a row); the
      head roots' rotation by a signed sine (three DSD ops of a head's seven) once the round trip
      is gone; the 64-row and klein programs on the simulator (only 16 + 16 ran); klein on a
      system.
  - Decode is unchanged: plan + build `diff -r` identical against d79d38e33's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08. flux2-mini's and
    klein's VAE programs (rows and tiled) byte-identical; flux2-mini's denoise at M1g's pins the
    same files bar their price (`stream.json`). The programs the search picks for flux2-mini's
    text, denoise and the GQA miniature's text move under the new prices (folds, `py 32`).
    flux2-mini's text reading as compile on load now builds it (10 x 62 PEs), on the simulator
    through the engine (`flux_2_minis_text_on_the_fabric`, co-resident with the denoise and VAE
    programs, linked on load): 7.860e-3 / 0.99996911 against engine-cpu, 8.735e-3 / 0.99996185
    against transformers' `Qwen3ForCausalLM`, M1h's digits; 1.43M cycles (the first launch),
    70 min of simulator.

- **K1** (FLUX.2 [klein] 4B on the CS-3, its real checkpoint; requests 1482-1488). **Blocked:
  a stream program over more than one band hangs at its first launch on the system**, and every
  klein program is one (64 + 64 rows: 4 bands; 1024²: more).
  - **The checkpoint.** `black-forest-labs/FLUX.2-klein-4B`'s diffusers snapshot (transformer,
    text encoder, VAE: 15 GB, Apache-2.0, not gated), landed through the import contract as the
    engine does. engine-cerebras `klein_reading_files` (ignored; `READING_CKPT`,
    `READING_PROGRAM`, `READING_OUT`) writes, for a program `wafer-compile stream` built, its
    static arrays made from the checkpoint (3.856G words at 64 + 64, 15.4 GB of 32-bit words;
    staged as 9 gzip tars of 0.7-1.0 GB), a step's drawn inputs and engine-cpu's 2-step loop at the
    program's rows (6.5 min a step on the shared host). `wafer-bench --reading --statics <dir>`
    writes them on the program.
  - **64 + 64** (the search's 730 x 1160 PEs, 33 units; 34.0M cycles, 45 ms priced): cslc on the
    worker links it for the system in 15 min (811 MB). The load takes 120 s, the static arrays
    41 s (1,092 copies, 94M words a second); the first launch never returns (1483: 40 min,
    1483_b: 17 min; the probe after passed, so the hang does not outlive the program).
  - **Where** (M2e's compiler, `--stages N`, each cut linked and run on the system, its static
    arrays from the checkpoint; a cut reads back no velocity, so the step's cycles are the
    check):

    | stages | units | PEs | bands | first launch |
    |---|---|---|---|---|
    | 1 (the entry) | 1 | 3 x 258 | 1 | 67.7k cycles |
    | 2 (+ the lane prologue, double block 0) | 4 | 246 x 286 | 1 | 7.24M cycles, 109 ms |
    | 5 (+ double blocks 1-3) | 10 | 684 x 286 | 1 | 15.2M cycles, 139 ms |
    | 6 (+ double block 4, the first unit of band 2) | 12 | 684 x 572 | 2 | **hangs** |

    (A first launch's cycles include its static arrays landing, M1k.)
  - **flux2-mini reproduces it.** The same 16 + 16 program built within a fabric 45 wide (the
    compiler's bounds narrowed: three bands, 36 x 114 PEs) hangs at its first launch on the system;
    the search's own one-band program (57 x 62) runs, the simulator's numerics (velocity 4.72e-3 /
    0.9999888 and 6.33e-3, the latent after 2 steps 3.88e-3 / 0.9999925 against engine-cpu), its
    re-launched step **1.239M cycles, 2.33 ms** (the simulator's M2e step 1.127M plus 10%). No
    multi-band program has run on the simulator: the band's wrap (`Stream::wrap`, the row stream
    from a band's last unit back west and south into the next band's first) and the lane bus down
    to a later band are the suspects, and the mini at a narrow fabric is a small repro.
  - The M2e 1024² program (741 x 1152 PEs, 1,081.4M cycles, 1.44 s priced) is built and its
    static arrays made (3.859G words); not run, being multi-band.

- **M2f** (the GEMM regime: a whole-sequence reading over an array of PEs, layer by layer, every
  PE of the array on the current op; flux2-mini's denoise step on the simulator, klein-4B's 1024²
  step builds and links). Seconds here are at 750 MHz.
  - **Why.** The stream regime streams rows through units, one unit working at a time (M2e: a
    row's round trip through a unit; its floor 657M cycles a step, 60x the roofline). The
    roofline wants every PE on the current layer: a layer's GEMM `Y[T, N] = X[T, K] W[K, N]` cut
    over the fabric by its rows **and** its channels, as the tiled regime (M2d) cuts a VAE's
    image, with the channel split M2d left open.
  - **One mechanism for every whole-sequence reading** (`stream::gemm`, `emit::gemm`,
    `cost::gemm`; `wafer-compile stream --regime gemm [--gemm C,R,KBM] [--price --parts]`). The
    regime reads the stream regime's own lowering (`stream::build::lower_units`, factored out of
    the stream build: the stream program's units, each a side of a stage as a list of `Step`s:
    rows in, root regions, matvecs, the joint attention's store and attend, rows out; the lane
    prologue a unit of one row). Only the placement differs: a unit's steps become **ops over
    the whole array** in the program's order (a block's units' steps before the barrier, one
    attention op over all their rows, the steps after it), each lowered by its family: a
    matvec a GEMM (an identity matvec, the stream's move across a unit, is nothing here: an
    alias), a root region the printer's (`emit::region`, unchanged but for one option), the
    attention's store and attend one op, a host row in or out an entry or a read-back. A
    trace merge of the classes' arms (the single blocks' rows, both classes') is one value. So
    a DiT block, a text encoder's layer (causal prefill; MHA: grouped-query attention is
    refused by name, a head's band holding its q, k and v alike) and an exit's projection are
    the same ops; nothing names a model. The stream regime stays: compile on load (and
    `stream::compile_whole`) builds both and takes the cheaper by the cost models.
  - **Layout.** An array of `C x R` PEs and, below it, the weights' tapes:
    - column `c` of the array is channel block `c`, row `r` token block `r` (`tb = T / R`
      consecutive rows of the step, all of one class: every class's rows cut into whole
      blocks);
    - a value of width `N` is cut into **segments** at every column a reader takes apart (a
      split's q / k / v, a gated MLP's gate and up halves, a packed modulation's shift / scale /
      gate), each segment evenly over the `C` columns; PE `(c, r)` holds, f32, its `tb` rows x
      its piece of every segment, side by side. A region, a matvec input or an attention
      operand reads one whole segment, so its chunk at PE `c` is the segment's `c`-th piece (the
      region printer's chunk `c`, its tables the chunk's). A value only GEMMs and the host read
      may cut unevenly (klein's 128-wide patches over 384 columns: a column or none);
    - a lane vector (the timestep's modulations) is a value of one row held at every row of
      PEs, made by every row (its GEMMs a row each);
    - the residual stays f32 at its owner; values share the arena by lifetime (first fit).
  - **The GEMM** (output-stationary). PE `(c, r)` makes `Y[tb, its columns]`:
    - `X`'s pieces come along the row, a **line stream** (below): PE `c'`'s piece
      (`tb x K/C`, 16-bit, two a wavelet) in turn, every PE of the row taking every piece; an
      f32 input (`x32`) as two 16-bit halves;
    - the weights of the PE's columns come up its column from the column's **tape**: every
      GEMM's weight blocks in op order (a block a piece of `X` and an output segment,
      `K/C x` the segment's piece, 16-bit), the regions' static residents (a norm's gain, f32)
      where their regions are, and **the program itself**: each op's words stream in just
      before its weights (16-bit, `host::LIT` spans), so a PE holds one op's words, not the
      program (klein's 270 ops would be 2,666 words a PE; the op buffer is 18);
    - a piece: its block off the column's tree, `mv.csl`'s `mac32` a row (f32 accumulation);
    - the rows of a class the op is not for drain its blocks.
  - **Stationary or streamed, from memory.** Activations are stationary (a PE's rows of each
    value, klein's busiest arena 1,944 words). Weights cannot be: resident in the compute PEs,
    klein's 3.85G 16-bit weights are 17 KB a PE even over 442k of them, beside the kernel's
    29 KB of code and 16 KB of data (the linked ELF below): past 48 KB. So every layer's
    weights stream, once a step, from tapes that hold nothing else, at a word a cycle against a
    PE's `tb x 0.5` cycles a weight of `mac32` (the tape is never the bound: a column's 5.02M
    words a step against its 115M cycles). The search prices only that; a layer resident where
    a smaller model leaves room is the same blocks from a PE's memory (not built).
  - **The tapes** (M2d's, a column each): `TAPE_WORDS` (10,000) a PE in rows below the array,
    PE `k` sending its words then forwarding those of the PEs past it north, the head into the
    column's tree (every PE of the column takes every word, the routers multicasting). The
    compute PEs hold no weights: klein's 3.85G 16-bit weights are 5.02M words a column, 502
    rows of tapes under 576 rows of array.
  - **Line streams** (the row's input pieces, the column's keys and values; measured on the
    simulator before anything was built on them, `flux2f/probe/line`: 6 members x 64 words,
    three streams in a row, every word in order, about 210 cycles a turn). A line's members take
    turns as the root in index order; every member gets every piece in that order, no CPU
    forwarding, no hub:
    - two colours a line, one each way, every route at rest a pass-through to the next PE and
      the ramp; a member's input queue is bound to the forward colour until its turn, the
      backward one after (a WSE-3 colour takes one input direction: `rx = { WEST, RAMP }` is
      refused by cslc, so turns are route rewrites);
    - at its turn a member, every earlier piece taken, makes the backward colour a root's
      (`rx RAMP`), sends its piece, waits for the output queue to drain, makes it a
      pass-through again, then the same on the forward colour. The next member sends only once
      it has the whole piece off the forward colour, so no route changes under a flow;
    - one input and one output queue a line, rebound at the turn.
  - **Norms and heads** (`emit::region` as it is, a chunk of `N / C` columns): a row's sum is the
    chunk's sum and an **all-reduce over the row's `C` PEs** (`gseg.csl`: a chain east on two
    colours by parity, `fadds` in and out; the last PE's sum back west over the row's broadcast
    colour); a head wider than the chunk (128 over chunks of 8) is cut over consecutive PEs and
    its sums all-reduced over them (`seg.head_reduce`, the printer's new `Place::straddle`: off
    for every other regime, whose programs are byte-identical). Modulation, the gated
    residuals, SwiGLU and the four-axis rotation (P5 tables of the PE's chunk, the rows'
    positions their scalars) are local.
  - **Attention** (flash-style over the step's rows; a head's band of `head_dim / chunk`
    columns): the column's K / V line stream brings every row's `[k | v]` (its `tb x chunk`)
    in blocks of `KB = kbm x tb` keys (`KB` at most 127: the keys' dim-major tile is a DSD
    stride); a PE's queries' partial scores over the block (`mac32`: `q` its `x`, the keys its
    tile), the band's all-reduce of them (`gseg`), the block's max and the running max and sum
    (alike at every PE of the band: the same sums), each PE's `KB / band` keys' exponents, the
    band's all-reduce of the probabilities (zeros outside each PE's keys), then `P V` over the
    PE's columns (`mac32`, `P` 16-bit) and the output's rescale; a causal reading masks a key
    past its query.
  - **No bands, no turns** (K1's hang is a stream program's second band): the array is one
    rectangle, its tapes the rows straight below it; every line runs along one row or one
    column, every tape up its own column into its column's tree, the all-reduces along a row.
    No route turns a corner and nothing crosses a band; `fabric()` is `C x (R + tapes)`, refused
    past the system's 755 x 1,170.
  - **Colours** (11 of 21; laid by loops): the row line (2), the K / V line (2), the weight tree
    (1), the tape (2, by parity), the all-reduce chain (2, by parity), the row's and the band's
    broadcasts (2). Input queues: the row line, the weights, the K / V line, the chain, the
    broadcast; output: the row line, the K / V line, the chain, the broadcast.
  - **Memory a PE** (the linked ELFs; `cost::gemm::memory` is fitted to hold them). klein-4B's
    compute PE at 1024² (`C x R` = 384 x 576, `tb` 8): **46,962 bytes of 49,152** (text 28.7 KB:
    the interpreter and 7 printed regions; bss 15.9 KB: the arena of values, 1,944 words, the
    lines' and the attention's buffers, the regions' chunks and P5 tables; the task table 1 KB).
    Two things made it fit: the weights out of the compute PEs (in the tapes) and the program
    streamed in with them (its 270 ops' 2,666 words were not held). A tape PE: its 10,000 words.
  - **Cycles from first principles** (`cost::gemm`; measured rates: `mac32` 1.84 + 0.5 n cycles
    an instruction of `n` (`probe/p`: 0.73 a MAC at n 8, 0.53 at 72); a fabric word a cycle; a
    line's turn 150; an all-reduce's hop 12; an async fabric receive alongside `mac32` costs it
    nothing (`probe/p`), not used yet: everything here is synchronous):
    - a GEMM, a piece of its input: a turn, the piece's words (three times at its root: two sends
      and the copy), each output segment's block off the tree, `tb x K/C` `mac32`s of the
      segment's piece; `C` pieces;
    - a region: its `row()` ([`RegionFeatures`]) a row, its chunks in and out, an all-reduce a
      row's sum (`2 n + C hops`) or a head's (`band` hops);
    - the attention a key block: its `kbm` turns of the K / V line, the scores' `mac32`s, two
      band all-reduces of `tb x KB`, the softmax (two halving trees a query, `KB / band`
      exponents), the values' `mac32`s.
  - **On the simulator** (flux2-mini, 16 + 16 rows; the search's program: `C x R` 32 x 32, a row
    a PE, key blocks of the whole step, 11 rows of tapes: 32 x 43 PEs; `wafer-bench --reading
    --steps 2`, the 2-step loop from the engine's inputs, one load; rel L2 / cosine):

    | | vs engine-cpu | vs diffusers | cycles | simulator |
    |---|---|---|---|---|
    | entries and lanes alone (`--stages 1`): the context's and patches' embeddings | 1.686e-3 / 0.9999986 | | 251k | 10 min |
    | through double block 0's attention (`GEMM_UPTO=24`, a debugging cut) | 5.06e-3 / 0.9999872 (its output) | | 377k | 12 min |
    | step 0 velocity (sigma 1) | 5.77e-3 / 0.9999834 | | 969,671 | 41 min |
    | step 1 velocity (sigma 0.7672), re-launched | 6.26e-3 / 0.9999804 | | **969,202** | 16 min |
    | latent after step 1 | 1.42e-3 / 0.9999990 | | | |
    | latent after step 2 | 3.86e-3 / 0.9999925 | **1.48e-3 / 0.999999** | | |

    The stream regime's re-launched step on the same simulator (M2e) is 1,126,817 cycles, so
    the miniature's step is 14% fewer (its 32 rows and 256 channels leave little to cut: a
    GEMM's pieces are its overhead). The velocities are a regime's own rounding (IEEE f16
    activations and 16-bit weights, f32 accumulation; the stream regime's 4.85e-3 / 6.36e-3):
    the host program (`stream::gemm::emulate`, `wafer-compile stream --regime gemm --emulate
    --values`, every op as the PEs run it) is 1.686e-3, 3.1e-3 and 5.87e-3 from the capture at
    the entries, double block 0 and the velocity, the simulator's digits: the lowering is the
    host's and the fabric adds nothing. Against diffusers' 2-step loop (f32) the wafer is
    closer than engine-cpu's (3.66e-3: its bf16 trunk). The step's stamps start where the
    launch reaches a PE, so the first launch is the step: no start-up (the stream regime's first
    launch was 7.9M).
  - **The cost model against it**: as first written (`TURN` 150, `mac32` as measured, the
    tree's words) it priced the step 0.55M; a GEMM's piece costs more than its turn and words
    (its loop and blocks' setup, and its bounds: two u32 divisions). The entries' 251k and the
    step's 969k give 353 and 395 cycles a piece. The bounds' divisions are now one a GEMM where
    its input cuts evenly (every klein GEMM's but the patches'): the entries rerun at **218,651
    cycles** (-13%, the same digits), 253 a piece; `PIECE` = 280 prices the entries 0.23M and
    the step 0.85M (the step not rerun: an hour of simulator, the band fix's queue).
  - **klein-4B at 1024²** (the search's choice over every `(C, R, kbm)` that places, fits a PE
    and prices lowest: `C x R` = 384 x 576, `tb` 8, key blocks of 8 rows of PEs (64 keys); 221k
    compute PEs and 502 rows of tapes, 384 x 1,078 PEs; the search takes 60 s):
    **115.2M cycles, 154 ms a step** (the stream regime's M2e program: 1,081M, 1.44 s; 9.4x):

    | ops (klein, a step) | count | cycles |
    |---|---|---|
    | GEMMs | 108 | 80.6M (a single block's in_proj 1.44M, out_mlp 0.75M, out_attn 0.36M; a double block's side 2.75M, the sides in turn) |
    | attention | 25 | 30.5M (1.22M a block: 72 key blocks of 17k) |
    | regions | 133 | 4.0M |
    | in / out | 4 | 0.0M |

    **It links for the system fabric** (`cslc --fabric-dims=762,1172`; the program as it stood
    before the per-class entries, its code the same): 171 ELFs, 151 s, about 1 GB at its peak,
    391 MB of ELFs, rc 0 (the stream regime's program: 1,264 ELFs, 965 s, 36 GB). The compile
    writes 1.5 GB (the tapes' spans).
  - **Served through Pie** (a step's wall as M1k splits it): the step's entries are 10.4M words
    (the context's rows on the text PEs, the patches on the image PEs, every row's four
    positions on every PE of its row: arrays a class, a scalar no region reads not written),
    about 18 ms at M1k's measured 2.4 GB/s of h2d; the velocity back 0.5M words, about 1 ms: a
    step through Pie about 173 ms, 154 of it on the wafer.
  - **klein end to end at 1024²** (priced; one CS-3, readings in turn; the text reading is
    grouped-query attention, refused by the GEMM regime, so the stream regime's):

    | reading | footprint | M2e | M2f |
    |---|---|---|---|
    | text (512 tokens) | 723 x 559 PEs | 218.0M, 0.29 s | 0.29 s (stream) |
    | denoise, 4 steps | 384 x 1,078 PEs | 4 x 1,081.4M, 5.77 s | 4 x 115.2M, **0.61 s** (0.69 s with entries) |
    | VAE decode | 514 x 1,029 PEs | 0.15 s | 0.15 s |
    | **total** | | **6.21 s** | **1.05 s** (1.13 s with the denoise's entries) |

  - **What is left, in order of what it buys** (klein, a step):
    - **Half the fabric idle**: the tapes take 502 of 1,170 rows, so the array is 576 rows
      (`tb` 8, 221k PEs); every class's rows cut into powers of two leave no 1,152-row array
      with room for tapes. Weights beside the array (a staircase of feeds from the east, each a
      colour relay at its turn) or a second array of the other half's tokens (the weights
      replicated, the keys exchanged) are the two ways; the roofline at 442k compute PEs is
      about 2x nearer.
    - **Synchronous everything**: a weight block's receive, the input's piece, the K / V
      pieces wait on their sends (`probe/p`: an async receive alongside `mac32` is free); the
      double blocks' two sides run in turn (each class's rows drain the other's blocks); the
      turns (384 a GEMM, 576 an attention) are about a quarter of the step.
    - The step's entries: the context and the positions are constant across a request's steps
      (written once a request: 18 ms -> 1 ms a step).
    - Grouped-query attention (klein's Qwen3-4B text reading): a band holding a kv head's q
      heads; a NeoX rotation whose mate is on another PE of the band (an exchange of halves).
    - klein on the simulator (only the miniature ran) and on a system (K1's band hang is the
      stream regime's: this program has one rectangle).
  - Decode is unchanged: plan + build `diff -r` identical against abce3a126's wafer-compile for
    qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b and tiny t08. flux2-mini's text
    (16 rows), denoise (16 rows, and M1g's pins), the GQA miniature's text and both VAE programs
    (rows, tiled), and klein's VAE byte-identical (`wafer-compile stream` without `--regime`);
    compile on load (`stream::compile_whole`, by price) takes the GEMM regime's denoise for klein
    (115M against 1,081M) and keeps the stream regime's for flux2-mini (its bf16 build at 16 rows
    priced 0.85M against the stream's 0.81M; measured 0.97M against 1.13M: the stream model is
    the optimistic one there) and every text reading (klein's grouped-query; the miniature's
    NeoX rotation, its mate half a head away, across PEs: refused by the printer).


## G1: the poem survey

**Goal.** Every `.poem` in `models/` runs: it compiles with no per-model code (placement searched
from the layer graph, ops lowered into the existing op modules, nothing refused by name), and a
slice of the real model (its first layers, real embedding and head, real weights) answers the
model's reference (HF / diffusers) on the fabric simulator.

**The survey.** `cargo run --release -p kernels-cerebras --example poems -- <deployment>...` runs a
deployment through the pipeline the engine runs on load: the decode program (`target::plan`,
`Program::of`, `build::emit`) and every value of the trace's `reading` switch through the stream
compiler (`stream::compile_reading`: the stream regime at 16 rows a class, a grid reading on the
clip 256 token rows make); `UNITS=1` also the decode step through the units (below), `DECODE=1`
the decode program for a trace with readings. One line a (deployment, reading): `PASS` or the
stage it stops at and why. `NODES=a,b` prints those nodes and their inputs' makers.

### G1.1 The matrix

Deployments: each poem's miniature (real dimensions, a prefix of the real model's layers and
experts) or a depth cut (`-l{n}`) of a real model, bf16. Before: f272089df. After: this branch.
"units" is the decode step through the stream regime's units (G1.2); "template" the decode
program of `KernelSet` x `Shape` (ARCHITECTURE.md).

| poem | deployment | reading | before | after |
|---|---|---|---|---|
| qwen3_dense | qwen3-0.6b | decode | template PASS | template PASS; **units PASS** (549 x 274, 0.29M cycles priced) |
| qwen_3 | qwen35-d0.8b | decode | template PASS | template PASS; units: **-l1 / -l3 PASS** (GDN state columns, linked); the full model's attention unit over its head columns' tile budget |
| qwen_3 MoE | qwen35-122b-a10b / qwen36-35b-a3b / qwen3-next-80b / qwen38-max minis | decode | `moe_sigmoid_gate_add` | units: GDN of more value heads than key heads (one state column a head of each) |
| qwen3_moe | qwen3-30b-a3b-mini | decode | template PASS | template PASS; **units PASS** (149 x 547) |
| llama_3 | llama31-8b | decode | template PASS | template PASS; **units PASS** (738 x 1072, 0.43M cycles priced) |
| llama_3 | llama33-70b-mini-l4 | decode | EMIT: a head turn off a column source | **template PASS** (search keeps PEs' words that build) |
| gemma_4 | e2b; e4b-mini-l1 | decode | template PASS | template PASS; units: shared KV (e2b), `gather.embed` per layer (PLE) |
| gemma_4 | 26b-a4b-mini-e64 | decode | router form refused | units: the experts' dense matvec over 512 rest roots (expert-sparse placement) |
| gemma_4 | 31b -l2 | decode | no partition | units: an emitter column gap |
| gpt_oss | 20b | decode | template PASS | template PASS; units: `attn.sink` |
| glm_4 | glm45-air-mini, glm47-mini | decode | router form refused | units: an emitter column gap (shared expert); heads straddling chunks |
| minimax_m2 | minimax-m25-mini | decode | router form refused | **units PASS** (517 x 798, 0.13M cycles priced) |
| minimax_m3 | m3-mini, m31-flash-mini | decode | DSA indexer | DSA indexer |
| mimo_v2 | flash-l6, pro-mini | decode | `split_qkv_padded` | the same |
| muse_glimmer | 30b-mini-l8 | decode | phase order | units: tiles do not fit at py 832 (a 30B layer in one unit) |
| muse_spark | open-mini-l4 | decode | `norm_gate` | units: `norm_gate` lowers to no region |
| nemotron_h | nano-mini, ultra-mini | decode | `ssm.mamba2` | the same (state step) |
| inkling | mini-l7-e8 | decode | `ssm.short_conv` | the same (state step) |
| kimi_k3 | mini, l4 | decode | `ssm.kda` | the same (state step), then MLA |
| dots3_note, glm_5, hy_4 | minis | decode | DSA indexer / MLA / hyper-connections | the same |
| glm_5_next, deepseek_v4, qwen_4 | minis | decode | hyper-connections, pooled attention | the same |
| flux_2 | mini; klein-4b | text, denoise, vae.decode | PASS | PASS (sha-identical programs) |
| flux_2 | mini; klein | vae.encode | mini PASS at a 256-row clip (16 rows make a 1 x 1 grid its downsamplings do not map); klein: a weight PE over its words | the same |
| z_image | mini | denoise | `elemwise.clamp` | an emitter gap in the lanes (a moved part) |
| ltx_2 | mini | denoise | `elemwise.clamp` | `rope_axes` in the split-ladder form (no region) |
| minimax_h3 | mini | denoise | `gather.concat_cols` | a lane column on no bus word |
| wan_2 | mini-nano, mini-d128 | denoise | cross-attention | cross-attention |
| hunyuan_image_3 | mini | decode / text / denoise | `rope_axes`; `gather.rows` | an emitter column gap (after MoE and the split rotation) |
| hunyuan_image_3 | mini | image.in / image.out | grid gaps | the same |
| qwen3_asr | 0.6b | decode | template PASS (the decoder) | template PASS; **units PASS** |
| qwen3_asr | 0.6b | text / audio.encode / audio.conv | `gather.rows`; no layers; a 1-D grid | the vocabulary's rows held; no layers; a 1-D grid |
| qwen3_tts | 0.6b | talker, predictor, codec.* | `gather.rows`, no layers, stacked prefill, grid gaps | `concat_cols`; a select of a sub-model; no layers; stacked prefill; grid gaps |

### G1.2 What moved, by mechanism

Each is one mechanism, not a model's case; the shipping decode programs (plan + build of
qwen35-d0.8b, qwen35-d9b, llama31-8b, gptoss-20b, gemma4-e2b, tiny t08) and flux2-mini's and
klein's text, denoise and VAE programs are sha-identical across every commit.

- **Slices for any poem** (pie-poem): a deployment cut to a depth (`-l{n}`) folds its poem's first
  `n` layers, whatever its package lays out; the trace reads only their weights (Qwen3-0.6B -l2:
  22 params). Twelve poems had a per-poem hook; the cut is now the fold's.
- **The stream regime lowers by capability.** Every elementwise op is a root op; whether its
  region prints is `fuse::chain` / `emit::region`'s question (the allow-list of names is gone).
  The host's ops are any op but a matvec over its values and the checkpoint's constants (the token
  embedding no case of its own; the host's lane ops a closure, read by node, not by re-matching op
  names). A readout's row selection (`gather.rows` by the readout rows) commutes with each row's
  ops after it. The head's logits are a matvec. A row-side value of no multiple of the row roots
  is held padded. An op reading a stream mixer's output after other phases runs at the column
  roots. The stream shape states its first MLP and rotation and no longer refuses others.
- **Regions print more.** `fuse::chain` (explicit runs only; the chains CUDA forms are unchanged)
  lowers stated-frequency and YaRN rotations, gated RMS norms, and rotations over position axes in
  a head's halves (split, NeoX, partial); `emit::region` prints a row's tanh, comparisons and their
  joins, selects on them, and a row's maximum or minimum (one root), by DSD ops or scalar loops.
- **Decode through the units.** A decode step is a stream program at one row a step
  (`stream::compile_decode`, `wafer-compile stream --reading decode --positions N`): each
  attention layer's K / V column is its cache across launches (a step's row stored at its
  position, `pos` at the K / V PEs, a sequence's first step clearing it; causal; a sliding
  window's mask). Every op the units lower runs: biases, q / k norms of any form, sandwich norms,
  any activation the regions print. `wafer-bench --reading <dir> --decode <golden.json>
  --checkpoint <dir>` runs it token by token from the real checkpoint (entries from the host
  forward, logits back). The templates stay the fast path for the shapes they serve; the units
  take what they refuse; the end state is the templates deletable (G1.5).
- **Routed experts as existing steps** (`stream::moe`): the gated sum of every expert. The router
  is a printed region (its scores, a pick bias, the top k a round at a time with the lowest
  expert on a tie, the picked weights: a gate an expert, zero unpicked); the gate is spread over
  each expert's columns by a constant matvec; gate, up and down are one matvec each over the bank
  as a weight view lays it; the gated activation is a region. Exact (an unpicked expert adds
  zeros) and per-PE bound as a dense layer of the experts' width.
- **Recurrent state as a state column** (P8: `ops/sst.csl`, `ops/cy_sst.csl`). The K / V
  column's PEs and chain hold a head's state `S` instead of rows: its root sends `[decay k |
  decay q | k]`, the PEs return their partials of `Sᵀk`, `Sᵀq`, the root forms `delta` and the
  output in one pass (`o = decay Sᵀq + (k·q) delta`), and `[delta | decay a key]` updates `S`.
  The PE side is one form for a decay a head (Gated DeltaNet, Mamba-2) or a key (KDA) and a delta
  rule or a plain increment; Gated DeltaNet's root runs its conv (the history at the root),
  SiLU, q / k norms and gates. A product whose q is cut into its heads' halves (an output gate)
  is a matvec a part over its weight's rows. Qwen3.5-0.8B's GDN layers build and link through
  the units; Mamba-2's and KDA's roots, and more value heads than key heads, are next.
- **Placement.** The decode search keeps a partition only where every PE's words build (rcfg),
  so a searched plan cannot fail at the emitter (llama33-70b-mini-l4). Each stream block's K / V
  columns are its own attention's (its first unit's heads, head width and scale), so a reading's
  attentions may differ (ltx's audio and video, gemma's sliding and global).

### G1.3 Special cases a general mechanism would remove

- `spatial::layer::Body::node`: the decode elementwise allow-list (7 names) and `target::shape`'s
  feature reads; `compose::refuse`'s per-kernel-set feature matrix; `KernelSet {Own, Shared,
  Windows, Experts}` itself: the decode templates. The units take every model they refuse; once a
  unit program is as fast, they go.
- `target::shape`: the router refusal (`Softmax` with no bias only); `compose::bias_on` /
  `head_norm` read parameters by name (`q_bias`, `o_bias`, `router_bias`, `q_norm`).
- `spatial::layer::stream::classes`: the stream values `context` / `latents` by name (flux's
  `reference` and ltx's audio streams are not classes).
- `PERMUTES` (a list of names for the host's permutations).
- The stream floorplan's one-unit-a-layer bands: a layer too wide for 512 rest roots is refused
  (a 30B layer); the template's K layers a unit is the missing axis.

### G1.4 What is left, by class

- **Recurrent state, the rest** (G1.2's state column built for Gated DeltaNet): Mamba-2's root
  (`delta = dt x`, its group's B and C, `D x`, the gated group norm after) for nemotron_h, KDA's
  (a decay a key) for kimi_k3 and glm_5_next, short conv (a conv history alone) for inkling;
  more value heads than key heads (Qwen3.5's larger models, the MoE minis).
- **Attention variants**: sinks (gpt-oss), shared KV (gemma e2b), cross-attention (wan), unequal
  q / k / v (mimo), MLA, the DSA indexer, pooled attention; per-block attention shapes.
- **Hyper-connections** (deepseek_v4, glm_5_next, hy_4, qwen_4) and ops outside the layer frame
  (qwen_4).
- **Shared-expert gate** (`moe_sigmoid_gate_add`) and shared experts beside routed ones (glm).
- **Emitter column gaps**: a value's column at no root buffer (gemma 31b, glm45, hunyuan, z_image
  lanes), heads straddling a region's chunks (glm47), a lane column on no bus word (minimax_h3).
- **Speech / grid readings**: layerless readings (asr's audio encoder, tts's codec), a stacked
  prefill (tts's codec prenet), 1-D grids and resamplings across frames, host-kept voxel inputs
  in the tiled regime.

### G1.5 Slices for the simulator

The slice each poem is verified with (its HF repo; layers kept so every layer kind is present;
embed and head real). Where the real slice is over the ~10 GB budget, gated or missing, the
poem's miniature (a real prefix with fewer experts) or a shrunk config, marked:

| poem | slice | note |
|---|---|---|
| qwen3_dense | Qwen/Qwen3-0.6B `-l2` | 1.5 GB |
| qwen_3 | Qwen/Qwen3.5-0.8B `-l4` (GDN x 3 + full) | MoE: Qwen3.5-35B-A3B `-l4-e64` |
| qwen3_moe | Qwen/Qwen3-30B-A3B layer 0, 16 experts (`qwen3-30b-a3b-mini -l1`) | 5.1 GB fetched |
| gpt_oss | openai/gpt-oss-20b `-l2` | 3.3 GB |
| nemotron_h | Nemotron-3-Nano-30B-A3B `-l6` (M, E, *) | 6.9 GB |
| glm_4 / minimax_m2 / mimo_v2 / minimax_m3 / glm_5 / glm_5_next / muse_glimmer | their minis or `-l{n}-e{e}` | under budget with an expert cut |
| llama_3 | unsloth/Llama-3.1-8B-Instruct `-l1` (meta's is gated) | |
| gemma_4 | google/gemma-4-E2B-it whole (KV sharing from layer 15) | 8.3 GB |
| flux_2 / z_image / wan_2 | DiT 1 block of each kind + VAE (+ text encoder or its embeddings) | |
| deepseek_v4, dots3_note, hy_4, inkling, qwen_4 | minis: real prefix, 8-16 experts | real `-l{n}` 16-126 GB |
| kimi_k3, muse_spark, hunyuan_image_3, ltx_2, minimax_h3 | shrunk configs | over budget or gated |

### G1.6 On the simulator

Each slice runs teacher-forced (`wafer-bench --reading <program> --decode <golden.json>
--checkpoint <dir>`: the golden's tokens fed one a launch, each step's argmax past the prompt held
to HF's next token; `--dry 1` the host's side alone, `--export <dir>` the first step's arrays for
`tools/wafer/stream/run.py`). Goldens: HF transformers 5.23 on the slice (`num_hidden_layers`
cut, the router's first experts), f32.

| slice | program | host side | simulator |
|---|---|---|---|
| Qwen/Qwen3-0.6B -l2 | 3 units, 159 x 274, 64 positions, 0.023M cycles priced | lands, 187M static words, 1,040 entry words a step | halts before its first step: an `hcf` in a memcpy halo PE (x 2; rows 1, 2, 32 across three runs) at 5.8-44k cycles, after every write was issued: the launch or the first read back. The same halt under the Python SDK harness (`--export`, `run.py`: the arrays loaded, the launch returning at once, then the halt), so the program's, not the engine's runtime: the next suspect the first read back (`hout`, the logits a row at the head's column roots) |
| Qwen/Qwen3-30B-A3B layer 0, experts 0-15 | 2 units, 86 x 526 | lands (a range-fetched 2 GB slice), 414M static words | queued behind the above |
| Qwen/Qwen3.5-0.8B -l1 (GDN) | 2 units, 422 x 154, 16 state columns | lands, 276M static words | queued behind the above |

## 5. Risks

- **Numerics: f16.** The decode kernels hold IEEE f16 tiles and f16 activations, with f16
  accumulation a PE. DiT activations (adaLN-modulated, large residuals) can pass f16's 65,504 range.
  Measured in M1c (§4.1) on klein-4B:
  - The matvecs' inputs stay below it.
  - f16 is 7x more accurate than bf16, at 0.55 cycles an element with f32 accumulation (not 2x).
  - The text residual passes it (to 103k), so the residual stays f32 at the roots.
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

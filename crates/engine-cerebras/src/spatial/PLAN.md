# The spatial plan: poem trace → kernel family → compile parameters

`plan.rs` derives a wafer program's parameters from the model, so a new model
of a known shape needs no hand-written `--params` string. The kernels stay
hand-written CSL. The plan only picks the kernels and sizes them.

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
   - `LmHeadPlacement`: an untied head. The heads support `UNTIED`, but the
     engine's placement reads `embed_tokens` for the logits.
3. **`SpatialPlan::derive(spec, WaferDims::CS3, Goals)`.** `Goals` gives the
   batch `B`, the batch the grid shape must hold (default 8), `PS`, and optional
   pins for `NB`, `MC`, `CAP`, the pipeline and the host page copy.
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

   It is fitted to the image ends in `serve2d/README.md` (B8/NB8, B16/NB4)
   and `qwen2d_paged8b/README.md` (B1/NB4). It lands within ~0.1 KB of every
   other build in those tables, and it rejects exactly the builds that did
   not link. The link stays the final word.
5. **`plan::check(trace, out_json)` / `SpatialPlan::check(out_json)` /
   `ModelSpec::check_program(get)`.** These are the load-time checks of a
   compiled program against the model. They cover both parameter
   vocabularies:
   - **serve28:**
     - hidden, head_dim, inter, vocab, layers;
     - `attn_rows/band_rows` = kv heads, `heads` = q/kv;
     - divisibility, including `lps ≠ 0`.
   - **2-D builds:**
     - `HR`, `Dc·Gs`, `Qh`, vocab, `GW·GH`;
     - enough kv-head columns, `PY·Hc ≥ hidden`, `PX·Ic ≥ inter`.

   They also refuse what the placement can't do yet: an untied head, an eps
   other than the CSL's 1e-6, missing q/k norms, a stated RoPE table, or
   another softmax scale. The wiring for `spatial.rs` (and `Layout::of`'s
   divide-before-check) is in `scratch/fork9/wire-plan.patch`.

## What it gives today

| model | family | grid (PX × PY, Hc, Ic, Qh) | B / NB / MC | a PE (of 49,152) | full program |
|---|---|---|---|---|---|
| Qwen3-0.6B | serve2d | 128 × 128, 8, 24, 2 | 4 / 8 / 2 | 42,368 | 4 × 7 grids + 129 × 792 head (641 × 896), Vr 192, F 396: 0105/0106 exactly |
| Qwen3-8B | paged8b | 128 × 160, 26, 96, 4 | 4 / 4 / 1 | 47,952 | none: serve2d wants Qh 2 and square grids; 5 × 7 = 35 grids fit, 36 needed |
| Qwen3-14B | paged8b | 128 × 288, 18, 136, 5 | 4 / 4 / 1 | 47,664 | none: 40 grids, 20 fit |
| Qwen3-32B | paged8b | 128 × 512, 10, 200, 8 | 4 / 8 / 1 | 48,272 | none: 64 grids, 10 fit |
| Llama 3.1 8B | paged8b + IdentityQkNorm, RopeTable | 128 × 192, 22, 112, 4 | 4 / 4 / 1 | 47,216 | none: 32 grids, 30 fit |

The tests (`cargo test -p engine-cerebras plan`) reproduce:
- the four 0105 builds and the four 0108 pipelines, key for key;
- 0107's 8B builds, with NB pinned where 0107 chose it for its test matrix:
  the auto rule gives NB 8 at B 1/2, NB 4 at B 4 and NB 1 at B 8, and the
  KVCP 0 build uses NB 3;
- 0102's 8B head.

## Limits

- **Two families, one full program.**
  - Only serve2d places a whole model, and only with square f32 grids and
    Qh 2: in practice Qwen3-0.6B.
  - Larger models get a planned layer grid and head, and `blockers` say why
    no program places them.
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

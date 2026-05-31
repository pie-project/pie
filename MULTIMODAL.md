# Multimodal Support (Vision + Video)

**Status:** Phases 1 + 3 + 4 done. **Gemma 4 vision works end-to-end** — `pie run
image-qa` on `gemma-4-E4B` correctly describes a real image (parity-verified
encoder, cosine 0.999), with HTTP redirect-following + chat stop tokens fixed.
**Video works end-to-end** — `pie run video-qa` on `gemma-4-E4B` samples GIF
frames and correctly answers "a rotating Earth from space" (motion-aware);
implemented as frames-as-images through the same vision tower with **zero
engine change** (`Context::append_video`). **Qwen 3.6 / Qwen3-VL vision works
end-to-end** — `pie run image-qa` on `Qwen3-VL-2B-Instruct` (arch-dispatched in
the inferlet) correctly describes the stop-sign image as "the entrance to the
Chinatown area … a red archway with Chinese characters, two stone lion statues,
and a stop sign." Full integration landed: `Qwen3VLVisionConfig` parse + bind,
the new `qwen3_vl` decoder model (reusing `llama_like`) with **DeepStack**
residual injection at decoder layers 0/1/2 + the fused interleaved-**M-RoPE**
kernel (`launch_qk_rmsnorm_mrope_bf16`), executor M-RoPE/grid plumbing,
`media.rs` Qwen `from_pixels` branch, `qwen_patchify`, and `<|vision_start|>`/
`<|vision_end|>` instruct delimiters. Encoder parity: early DeepStack taps match
HF-bf16 to 0.99986. No regression: gemma image + Qwen3 text still correct.
Follow-up: the M-RoPE image→text relative distance currently uses `token_count`
rather than HF's `max(t,h,w)` span (correct answers, but decoupling KV-position
from M-RoPE-position would be bit-faithful for long multi-turn image chats).

**Audio input (`gemma4_audio` USM/Conformer)** is at the same milestone: encoder
scaffolded (`gemma4_audio_forward.{hpp,cu}` + adapter, compiles clean), HF parity
reference dumped (`scripts/gemma4_audio_parity_ref.py` → 752 tensors,
missing=0), and its parity harness (`driver/cuda/tests/gemma4_audio_full_parity.cu`)
runs at first-draft cosine 0.981 (199 mel frames → 50 audio tokens → [50,2560]).
Remaining: drift refinement (chunked-local attention masking + conv-module are the
flagged `PARITY TODO`s) + the analogous driver/SDK integration (Audio WIT resource,
log-mel frontend via `realfft`, `Context::append_audio`) → e2e on `gemma-4-E4B`.
Perf/cleanup of the first-cut scatter also remains.
**Scope:** Vision-language input for the frontier checkpoints Pie already runs
text-only — **Gemma 4** and **Qwen 3.6** first — then **video**. Image
*output*/generation is explicitly out of scope.

> Companion docs: `LOADER.md` (weight loading), `BRIDGE.md` /
> `BRIDGE_REDESIGN.md` (runtime↔driver wire), `INFERENCE_POLYMORPHISM.md`
> (per-arch forward dispatch).

---

## 0. Guiding principle

> **An encoded image becomes ordinary context KV.**

Once image embeddings are scattered into the hidden state and run through the
decoder, the resulting KV pages are indistinguishable from text KV. Therefore
`fork`, `snapshot`/`save`/`open`, `commit`, prefix-cache, and the page market
all apply to images **for free, with zero new machinery**. No other serving
engine can express that — it falls out of Pie's existing `context` abstraction.

Three corollaries drive every decision below:

1. **The public surface is tiny.** One new resource (`image`) and one new
   `forward-pass` method (`input-image`). Everything else is internal.
2. **Model divergence is quarantined in the per-arch driver graph** — exactly
   where the text graphs already differ (`INFERENCE_POLYMORPHISM.md`). WIT and
   the wire never learn that Gemma ≠ Qwen.
3. **Video intelligence is library/inferlet code, not engine code.** Frame
   sampling, memory windows, encode-once/ask-many, streaming, summarization all
   ship as crates/inferlets. The engine only gains "encode a visual span into
   KV + let me manage that KV."

---

## 1. Where Pie stands today

Pie is **text-only by construction**, and the boundary is clean:

| Layer | Today | File |
|---|---|---|
| Inferlet↔engine contract | `forward-pass` takes token ids only: `input-tokens(tokens: list<u32>, positions: list<u32>)` | `runtime/wit/core/wit/inference.wit` |
| Wire | `ForwardRequest { token_ids, position_ids, … }` — 1-D positions, no media | `driver/bridge/src/schema.rs:83` |
| Loader (CUDA) | Detects the multimodal wrapper and **strips** the vision/audio towers: `mm_lm_strip_prefix = "language_model."`, `mm_skip_prefixes = {vision_tower., vision_model., visual., multi_modal_projector., …}` | `driver/cuda/src/loader/hf_config.cpp:502` |
| Loader (portable) | Same: runs the text decoder nested under `model.language_model.` | `driver/portable/src/model.cpp:456` |
| Forward (every arch) | Opens with a `gather_rows` over `embed_tokens` → hidden states | e.g. `driver/cuda/src/model/llama_like.cpp:736` |

So **Gemma 4, Qwen 3.6, GLM 5.1, DeepSeek-V4 already load and run — as their
text tower only.** The encoders are excluded *surgically*; we are re-enabling a
path that was intentionally cut, not retrofitting the engine.

The substrate for the *programming* side is also already present:

- `inferlets/image-fetch` — fetches a URL and decodes it with the `image` crate
  **inside the inferlet** (image acquisition + CPU decode already live there).
- `inferlets/windowed-attention` — bounds generation memory by building BRLE
  attention masks (`pass.attention_mask(...)`); an inferlet expressing a memory
  policy the engine doesn't know about.

---

## 2. Architecture

```
                       ┌─────────────── host (runtime, Rust) ───────────────┐
image/video bytes ─► image::from-bytes / from-frames ─► processor (geometry) │
  (inferlet decodes & samples frames)                   token_count, grid,    │
                       │                                 mrope positions       │
                       └──────────────────────┬──────────────────────────────┘
                                              │  pixels + grid (wire side-channel)
                       ┌──────────────────────▼──────── driver (CUDA / portable) ─┐
forward-pass.input-image(img, anchor) ─► vision encoder (arch-dispatched)         │
                                         projector ─► scatter rows at embed point  │
                                         ─► decoder layers ─► **KV written to ctx** │
                       └────────────────────────────────────────────────────────────┘
                                              │
                              image KV is now just context KV
                              ✦ free fork / snapshot / prefix-cache / market
```

**The splice point** is the `gather_rows` over `embed_tokens` that opens every
forward. For image rows we overwrite the gathered rows with projector output.
That single scatter is the entire model-agnostic mechanism.

---

## 3. Public surface (WIT contract)

The `image` interface is **live**: defined in `media.wit` (source +
vendored `deps/core/` copies for both the runtime and SDK builds), `import`ed by
the core `imports` world, and backed by a host impl in `runtime/src/api/media.rs`
(`HostImage for InstanceState`) registered in the `bindgen!` `with:` map. Both
the runtime (`cargo check -p pie`) and the SDK guest (`cargo check` in
`sdk/rust/inferlet`) compile against it. The `input-image` method below is still
spec-only (Phase 1.3) — it would add a handler to the existing `forward-pass`
resource.

```wit
// live at: runtime/wit/core/wit/media.wit (+ vendored deps/core/ copies)
interface media {
    use types.{error};
    use model.{model};

    // A host-side handle to a preprocessed visual input (image OR video clip).
    // Decode + processor run on `from-*`; the geometry below is known
    // synchronously so token-count / position-span can be queried before
    // the forward pass.
    resource image {
        // Single still image. Bytes are an encoded image (PNG/JPEG/…).
        from-bytes: static func(model: borrow<model>, bytes: list<u8>)
            -> result<image, error>;

        // Video clip: a temporally-ordered list of already-decoded frames
        // (the inferlet owns frame extraction + sampling) plus per-frame
        // timestamps in seconds. Produces ONE visual span with grid t>1.
        from-frames: static func(model: borrow<model>,
                                 frames: list<list<u8>>,
                                 timestamps: list<f32>) -> result<image, error>;

        // Hidden-state rows / KV slots this visual span occupies.
        token-count: func() -> u32;

        // How far the 1-D sequence cursor advances past this span.
        // == token-count for Gemma (1-D RoPE); == max(t,h,w) for Qwen M-RoPE.
        position-span: func() -> u32;

        // (t, h, w) in merged-token units.
        grid: func() -> tuple<u32, u32, u32>;
    }
}
```

```wit
// proposed addition to interface forward-pass (inference.wit)
//   Splice an encoded visual span at `anchor`. The driver runs the encoder
//   and scatters the projected rows into the hidden state at this span.
//   No placeholder token ids needed — the span is declared directly.
input-image: func(image: borrow<image>, anchor: u32);
```

What is deliberately **absent**: image placeholder token ids (the HF
`<image>`-repeated hack disappears), and any embeddings crossing into WASM
linear memory. The inferlet only ever holds a handle.

---

## 4. Wire protocol

`ForwardRequest` (`driver/bridge/src/schema.rs:83`) grows an image block,
indptr-gated exactly like the existing `masks` / `logit_masks`. Empty for
text-only passes → the C-ABI mirror (`driver/bridge/include/pie_bridge.h`) and
the hot path are unaffected.

```rust
// images, flattened across the batch
pub image_pixels: Vec<u8>,         // preprocessed pixel bytes, concatenated
pub image_pixel_indptr: Vec<u32>,  // byte range per image
pub image_grids: Vec<u32>,         // (t,h,w) per image (patch units)
pub image_anchor_rows: Vec<u32>,   // hidden-state row where each span starts
pub image_indptr: Vec<u32>,        // images per request

// M-RoPE: 3 position components per row. EMPTY unless the model is mrope.
pub mrope_position_ids: Vec<u32>,  // 3*N when present
```

> Forward-looking: the same block is what a future `encode → embedding-resource`
> RPC would populate instead of pixels, if we add encoder-output caching. Not
> needed for v1 — the *context* already caches the result.

---

## 5. Host processor (geometry)

Lives in the runtime (CPU, Rust): `runtime/src/multimodal.rs`. Mirrors the HF
image processors so `image.token-count()` / `position-span()` are exact and
synchronous. Two arch families:

- **Gemma 4** — fixed-resolution `gemma4_vision` encoder (see §6.1). **280**
  soft tokens per crop (`vision_soft_tokens_per_image`, confirmed against the
  shipped `google/gemma-4-E4B` config); pan-and-scan splits large/non-square
  images into extra crops. 1-D positions (`position-span == token-count`).
- **Qwen 3.6** — native dynamic resolution. `smart_resize` → patch grid
  `(t,h,w)`; 2×2 patch-merge → `t·h·w / merge²` LLM tokens; **M-RoPE**
  `position-span = max(t, h/merge, w/merge)`.

Pixel resize/normalize (the heavy per-pixel work) is a later slice; the geometry
above is the part that gates the handle API and must be parity-exact, so it
lands first and is unit-tested in isolation. **(Started — see Phase 1.1.)**

---

## 6. Driver: where (and only where) the two models diverge

1. **Stop stripping the towers** for these archs — reverse the
   `mm_skip_prefixes` cut at `hf_config.cpp:502` (and `model.cpp:456`) so
   `vision_tower.` / `multi_modal_projector.` load. Keep the strip as the
   fallback for archs not yet done.
2. **Per-arch encoder graph**, dispatched by `PieArch` like the text graph
   already is. A ViT is non-causal MHA + GEMM + norm — it reuses kernels the
   driver already has; this is graph wiring, not new kernels.
3. **Scatter at the embed point** (`gather_rows`) — overwrite image rows with
   projector output. Model-agnostic.

| | **Gemma 4** | **Qwen 3.6** |
|---|---|---|
| Encoder | `gemma4_vision` ViT (§6.1) — NOT SigLIP | native-resolution ViT (NaViT), window attn |
| Tokens/image | 280 soft tokens/crop (pan-and-scan) | dynamic from `grid_thw`, 2×2 merge |
| Positions | standard 1-D RoPE (`mrope_position_ids` empty) | **M-RoPE** (3-component side-channel) |
| Image attention | **bidirectional** within span (graph sets sub-mask) | causal |
| Injection point | embed point only | embed point **+ DeepStack** (multi-level features into first few decoder layers) |

The two structural risks both live in this table and are arch-local:

- **M-RoPE** is the one shared-kernel touch: the RoPE kernel reads 3 position
  components when `mrope_position_ids` is non-empty; text passes an empty vector
  → byte-for-byte unchanged.
- **DeepStack** is why the contract is "encoder features + row indices," not
  "input embeddings": Gemma injects at layer 0, Qwen at a few. The decoder layer
  loop already lives per-arch, so multi-layer additive injection is a local edit
  to Qwen's graph; WIT/wire never change.

### 6.1 Gemma 4 vision encoder — from the shipped checkpoint

Inspected directly from `google/gemma-4-E4B` (`config.json` + `model.safetensors`
header). **It is NOT SigLIP** — correcting the earlier assumption. Tensor
prefixes: `model.vision_tower.` (encoder) and `model.embed_vision.` (projector);
the audio tower (`model.audio_tower.` / `model.embed_audio.`) stays stripped.

`vision_config` (`model_type: "gemma4_vision"`):

| | value |
|---|---|
| hidden_size | 768 |
| num_hidden_layers | 16 |
| num_attention_heads / kv | 12 / 12, head_dim 64 |
| intermediate_size | 3072 |
| patch_size | 16 |
| activation | `gelu_pytorch_tanh` |
| norm | RMSNorm (eps 1e-6), **Gemma sandwich**: input / post-attention / pre-FF / post-FF, + per-head **q_norm/k_norm** (64) |
| position | **RoPE in the encoder** (`rope_theta: 100.0`, type default) |
| MLP | **gated**: gate_proj · up_proj → gelu-tanh → down_proj |
| linears | **clipped/quantized**: each `*_proj` ships `linear.weight` + scalar `input_min/max`, `output_min/max` (`use_clipped_linears: true`) |
| pooling | `pooling_kernel_size: 3` (avg-pool patch grid before projection) |
| soft tokens/image | **280** (`vision_soft_tokens_per_image`) |
| projector | `model.embed_vision.embedding_projection.weight` `[2560, 768]` (vision 768 → text 2560) |

**Implication — this is good news for kernel reuse.** A `gemma4_vision` layer is
essentially a *non-causal Gemma text layer*: it reuses the driver's existing
RMSNorm, RoPE, qk-norm, and gated-gelu-tanh MLP kernels (`driver/cuda/src/model/
gemma4.cpp` already has all of these). The genuinely new pieces are: (1) patch
embedding (conv/linear over 16×16 patches), (2) the **clipped-linear quant**
dequant path, (3) the 3×3 average pool → 280 tokens, and (4) the `embed_vision`
projection. So the encoder is closer to "wire up a Gemma layer stack with a patch
front-end" than "implement a new ViT from scratch."

---

## 7. Inferlet API expression

Existing calls unmarked; **proposed** calls noted. Layering matches the SDK
today (`forward.rs` low-level builder + `context.rs` high-level `Context`).

**High-level (`Context::append_image` / `append_video`)** — mirrors the
existing `flush()` page/cursor logic at `context.rs:399`:

```rust
let frames = sample_frames(&video_bytes, 1.0 /*fps*/)?; // inferlet policy + `image` crate
ctx.user("Here is a video:");
ctx.append_video(&frames).await?;                        // PROPOSED: encode clip → KV
ctx.user(&question).cue();
let answer = ctx.generate(Sampler::top_p(0.7, 0.95)).max_tokens(512).collect_text().await?;
```

**Encode once, fork per question** — the uniquely-Pie win:

```rust
ctx.append_video(&sample_frames(&video, 1.0)?).await?;  // encode ONCE
let base = ctx.snapshot()?;                              // persist encoded video KV
let answers = future::join_all(questions.iter().map(|q| {
    let mut branch = Context::open(&model, &base).unwrap();   // shares the video KV
    async move { branch.user(q).cue().generate(s).collect_text().await }
})).await;                                               // ViT + prefill paid once
```

**Streaming / online agent** — the market carries a long-lived feed:

```rust
loop {
    let frame = { let _idle = ctx.idle(); next_stream_frame().await? }; // drop bid during wait
    ctx.append_video(&[frame]).await?;                                  // PROPOSED
    if ctx.frame_count() > window { ctx.evict_frames(0..ctx.frame_count()-window); } // PROPOSED
    if let Ok(alert) = ctx.fork()?.user("Anything urgent?").cue()
        .generate(s).max_tokens(32).collect_text().await { emit(alert); }
}
```

---

## 8. Video design

A video clip is **just an `image` with temporal extent** — `grid()` already
returns `(t,h,w)`; a clip is `t>1`. This means video reuses the entire image
path (`input-image`, scatter, KV write, fork, snapshot) unchanged. The only
model-specific additions (M-RoPE `t`-component, frame timestamps, temporal
patch-merge) live in the host processor.

**Why it fits the inferlet model** — video forces *policy* decisions the engine
must not hardcode, and each maps to an existing primitive:

| Decision video forces | Inferlet primitive |
|---|---|
| which frames / resolution / token budget | inferlet code + `image` crate (uniform / keyframe / query-conditioned sampling) |
| bounded memory over long/endless video | `attention_mask` over frame spans + KV eviction |
| encode once, ask many | `snapshot()` + `fork()` |
| live / streaming feed | append-over-time + `idle()` / `set_bid` / `suspend` market |
| long video → hierarchical summary | `fork()` per segment (tree-of-thought shape) |
| "look closer" / temporal grounding | re-`append` a span at higher res, or re-attend via mask |

**The one genuine new engine mechanism video motivates: interior KV eviction.**
Today (`windowed-attention` note) masks stop the model *attending* to old frames
but the pages stay resident; `truncate` only drops the tail. Endless video needs
to reclaim arbitrary committed frame-span pages — `Context::evict_frames(span)`
plus, optionally, an encoder-output cache so an evicted frame can be
re-materialized without re-running the ViT.

---

## 9. Phasing & checklist

### Phase 1 — Plumbing + host geometry (model-agnostic)
- [x] **1.1** Host processor (`runtime/src/multimodal.rs`), unit-tested
  standalone (10 tests, edition 2021 + 2024):
  - Gemma + Qwen token-count / grid / `smart_resize` / pan-and-scan,
  - **M-RoPE per-row `(t,h,w)` position-id generation** (`qwen_mrope_positions`
    / `qwen_next_position`) — the exact data the `mrope_position_ids` wire field
    carries,
  - arch-agnostic `Processor` dispatch (`for_arch` / `layout_image` /
    `layout_video` / `uses_mrope` / `mrope_positions`).
- [x] **1.2** `media.wit` `image` resource — **live and compiling** on both
  sides:
  - `media.wit` in source (`core/wit/`) + vendored `deps/core/` for runtime and
    SDK; `import media;` in the core `imports` world (all copies),
  - `HostImage for InstanceState` (`runtime/src/api/media.rs`): `from_bytes` /
    `from_frames` decode dimensions (PNG/JPEG header parse) → `Processor` →
    `VisualSpan`; text-only models return a clean error,
  - registered in `bindgen!` `with:` + linker (`runtime/src/api.rs`),
  - `arch_name` added to `crate::model::Model` to select the vision arch,
  - SDK re-export `inferlet::media::Image`,
  - verified: `cargo check -p pie` ✓, `cargo check` (sdk/rust/inferlet) ✓,
    `cargo test -p pie --lib multimodal::` → 14/14 ✓.
- [x] **1.3** `forward-pass.input-image(image, anchor)` in `inference.wit` (all
  3 copies, `use media.{image}`) + host handler in `runtime/src/api/inference.rs`
  recording the span as a side-channel (grid + anchor + staged pixels +
  precomputed M-RoPE positions); does NOT touch token/KV/qo layout. `cargo check
  -p pie` ✓, `cargo check` (sdk) ✓.
- [x] **1.4** Wire fields appended to `ForwardRequest` (`schema.rs`,
  append-only): `image_indptr` (per-request CSR), `image_grids`,
  `image_anchor_positions`, `image_pixels` + `image_pixel_indptr`,
  `image_mrope_positions` + `image_mrope_indptr`. Batcher merge with CSR
  offsetting in `request.rs::append_request_with_options`; all explicit
  `ForwardRequest` literals updated (per-request/batched builders, chunked
  prefill, speculator). C-ABI mirror (readers + descriptor) added to
  `pie_bridge.h`. `pie-bridge` ✓, `pie` ✓, 2 new merge tests ✓ (16/16 lib).
  - TODO: chunked prefill does not yet split visual spans across chunks (empty
    image fields on chunks); radix-trie dedup key does not yet include image
    fields (harmless for text; revisit in Phase 2).
- [~] **1.5** Driver: the image side-channel now flows into the driver's
  per-request view.
  - `PieForwardRequestView` (`driver/bridge/include/pie_bridge/view.hpp`) gained
    the 7 image fields + a `num_images()` helper; `fill_forward_view` populates
    them as pass-through slices from the (macro-generated) descriptor.
  - C ABI declared in `pie_bridge.h` (readers + `PieForwardRequestDesc` fields);
    **verified by the bridge ABI tests** (`desc_layout`, `archived_layout`) and
    an rkyv round-trip that asserts all 7 fields survive
    (`round_trip_schema::frame_forward_round_trip`).
  - **Built the real CUDA driver** (CUDA 13.0, L40): incremental
    `cmake --build pie_driver_cuda_lib` recompiled the 17 view.hpp-dependent TUs
    (entry/executor/inproc_service/response_subpass/models) with 0 errors and
    relinked `libpie_driver_cuda_lib.a`.
  - Remaining (folded into Phase 2, where it's first exercisable): consume the
    view at the `gather_rows` point and scatter encoder rows. A zero-row
    "assert plumbing" pass needs a multimodal model registered, so it lands with
    the encoder rather than standalone.
  - Note: a full `cargo build -p pie-server` also builds the *portable* driver,
    which fails CMake config offline (`cpmaddpackage` — CPM can't fetch ggml);
    pre-existing/environmental, unrelated to these changes. The CUDA lib was
    rebuilt directly in its already-configured build dir to sidestep it.
- [x] **1.6** SDK: `inferlet::media::Image` re-export (1.2) + low-level
  `Forward::input_image(&image, anchor)` builder method emitting the WIT call at
  execute. `cargo check` (sdk) ✓. (`Context::append_image` ergonomics deferred
  to Phase 2 — it can't write KV until the encoder lands, and would otherwise
  desync page accounting.)

### Phase 2 — Gemma 4 (1-D RoPE, single injection)
Grounded against the local `google/gemma-4-E4B` checkpoint (`~/.cache/hugging\
face/hub/`) — parity is testable in-env (L40 + nvcc 13.0). Encoder spec in §6.1.
- [x] **2.0** Processor constants corrected to the shipped config:
  `tokens_per_image 256→280`, `patch_size 14→16`, added `pooling_kernel_size 3`
  (`runtime/src/multimodal.rs`); 14 tests still green.
- [~] **2.1** Loader + weight bind.
  - [x] `vision_config` → `GemmaVisionConfig` parsed into `HfConfig.gemma_vision`
    (`hf_config.{hpp,cpp}`), gated on `vision_config.model_type=="gemma4_vision"`,
    values confirmed against `gemma-4-E4B`.
  - [x] `Gemma4VisionWeights` / `Gemma4VisionLayerWeights` /
    `Gemma4ClippedLinear` structs + `bind_gemma4_vision()` (`gemma4.{hpp,cpp}`):
    binds `patch_embedder.input_proj` + `position_embedding_table`, the 16
    encoder layers (sandwich norms, q/k/v/o clipped-linears + q/k-norm, gated
    MLP), and `embed_vision.embedding_projection`, all by the real checkpoint
    tensor names. Clip-range scalars bound optionally. Compiles in the CUDA
    driver; **not yet invoked** (no GPU cost until the forward calls it).
  - [x] No un-strip needed — the `vision_tower.` skip prefix does **not** match
    gemma-4's `model.`-prefixed names (`model.vision_tower.*`), so the vision +
    projector tensors already load into the weight store (verified). Wired
    `bind_gemma4_vision` into `bind_cuda_model` (gemma-4 branch) behind
    `HfConfig.gemma_vision`; result stored on `BoundCudaModel.gemma4_vision`
    (+ `has_vision`). Compiles in the CUDA driver. (Audio tower also loads
    unused — a separate memory optimization, out of scope.)
- [~] **2.2** `gemma4_vision` encoder forward. Exact semantics confirmed from
  transformers 5.9 `modeling_gemma4.py` (NOT a guess):
  - **clipped-linear** = `clamp(x, input_min, input_max)` → standard BF16 matmul
    → `clamp(y, output_min, output_max)`; the clip scalars come from the ckpt
    (no weight dequant). Not "dequant" as the §6.1 draft assumed.
  - patch embedder: `x = 2·(pix−0.5)` → `input_proj` (768→768) → **add 2D
    position embeddings** (one-hot over (x,y) patch coords @ `position_embedding_
    table`, summed over the two axes; padding zeroed).
  - layer: sandwich RMSNorms; attention has q/k-norm (scale) **+ v-norm
    (no scale)**, **multidimensional (2D) RoPE** θ=100 over patch (x,y),
    bidirectional, scaling=1.0; gated gelu-tanh MLP. No per-layer scalar.
  - pooler: 2D avg-pool by patch position → `n_patch/pool_k²` (280) soft tokens,
    then **× √hidden**; strip padding rows.
  - `embed_vision` = **parameterless RMSNorm** (`embedding_pre_projection_norm`)
    → `embedding_projection` (768→2560).
  - [x] **Parity reference built + RUN on GPU** — `scripts/gemma4_vision_\
    parity_ref.py` loads the real vision tower (658 tensors, 0 missing) + embed
    from `gemma-4-E4B`, runs a deterministic 60×42-patch input, dumps
    `patch_embed`/`layer0`/`layer_last`/`pooled[280,768]`/`projected[280,2560]`
    + inputs to `/tmp/gemma4_vision_parity/`. This is the ground truth every
    CUDA stage is checked against.
  - [x] **CUDA forward — COMPLETE, parity-verified end-to-end** in
    `driver/cuda/tests/gemma4_vision_full_parity.cu` (standalone, nvcc
    `-arch=sm_89`, fp32). Reproduces the entire encoder + projector; checked
    against the fp32 reference dumps with intermediate checkpoints:

    | stage | max_abs | rms | note |
    |---|---|---|---|
    | patch_embed | 0.0 | 1.65 | exact |
    | layer0 | 5.0e-3 | 2.29 | 0.2% — per-layer math confirmed |
    | layer_last | 3.2 | 45.3 | fp32 accumulation over 16 layers |
    | pooled | 27 | 1141 | 2% (pooling averages noise down) |
    | **projected** | **9.0e-3** | **0.737** | **PASS** (<2e-2; RMSNorm renormalizes) |

    Implements (all from transformers source, not guessed): clipped-linear
    (clamp→GEMM→clamp), plain RMSNorm (q/k-norm scaled, v-norm + embed pre-norm
    unscaled), **2D RoPE** (head_dim split 32/32, x-half + y-half, invf=θ^(−c/16)),
    bidirectional attention (scaling 1.0), gelu-tanh gated MLP, 2D avg-pool ×√H,
    `embed_vision` projection. The residual error is fp32 naive-matmul summation
    order (cuBLAS would tighten it); it is far below bf16 noise — **the algorithm
    is verified correct.**
  - [x] **bf16 forward — driver-precision, parity-verified.**
    `driver/cuda/tests/gemma4_vision_full_parity_bf16.cu` exposes a
    **driver-portable `run_gemma4_vision(raw bf16 pointers)`** (bf16 storage,
    fp32 compute — matching the driver) plus self-contained kernels. Verified
    vs the real model: **rel_rms_err 1.07%, cosine 0.99994 vs HF-bf16** (2.2%
    vs fp32 — pure bf16 rounding on this model's large activations, rms→1142;
    every checkpoint's rms matches to ~0.1%). Lesson: `max_abs` vs fp32 is the
    wrong metric here; bf16-vs-bf16 rel-rms + cosine is.
  - [x] **Driver module compiled into the lib.**
    `driver/cuda/src/model/gemma4_vision_forward.{hpp,cu}` (+ CMake) exposes the
    verified forward as `run_gemma4_vision(const VisRawWeights&, …)`. Interface
    takes **raw bf16 device pointers** (cuda-only header) — nvcc can't parse the
    toml++ config headers `gemma4.hpp` pulls in, so the host call site does the
    `DeviceTensor::data()` extraction into `VisRawWeights`. Builds clean
    (`gemma4_vision_forward.cu.o` linked).
  - [x] **Host adapter compiled** —
    `driver/cuda/src/model/gemma4_vision_adapter.{hpp,cpp}` (g++):
    `to_vis_raw(const Gemma4VisionWeights&)` extracts `DeviceTensor::data()`
    pointers + derives dims from tensor shapes (`pos_table_size`, `text_hidden`),
    plus a `run_gemma4_vision(Gemma4VisionWeights&, …)` overload. The full driver
    bridge **config → load → bind → BoundCudaModel → to_vis_raw →
    run_gemma4_vision → projected[n_tok,2560]** now compiles in the lib.
  - [ ] Scatter `projected` rows into the hidden state at `gather_rows` (2.5) —
    needs real `pixel_values`/`position_ids` on the wire (the host processor,
    2.3), so it lands with 2.3.
- [~] **2.3** Host image processor.
  - **Correction:** Gemma 4 uses the **SigLIP2-style aspect-ratio-preserving
    resize** (`Gemma4ImageProcessor` subclasses SigLIP2), NOT Gemma-3 pan-and-scan,
    and the soft-token count is **variable** (`grid_h·grid_w / pool_k²`, ≤ 280),
    not a fixed 280. Rewrote `GemmaImageConfig` accordingly.
  - [x] **Geometry (parity-exact core)** in `runtime/src/multimodal.rs`:
    `resize_target` (faithful port of `get_image_size_for_max_num_patches`
    binary search), `patch_grid`, variable `token_count`. **Matches HF
    reference vectors exactly** (480×640→688×928 grid 43×58 → 277 tokens; 5
    vectors tested). Runtime compiles, 13 multimodal tests pass.
  - [x] **Real-image vision-path validated end-to-end.** Ran the *real*
    `Gemma4ImageProcessor` on a deterministic image → its `pixel_values` /
    `position_ids` (variable: 2394 valid patches → 266 soft tokens) → the bf16
    encoder (`gemma4_vision_full_parity_bf16.cu real`) → projected, **cosine
    0.99930 vs HF** (3.7% rel-rms, bf16-vs-fp32). Confirms the encoder handles
    variable patch counts + real positions + padding-strip correctly.
  - [x] **Patchify + positions — bit-exact vs HF.** `patchify_chw`
    (channels-last within patch, row-major patches, `(x=col,y=row)`) verified
    against the HF processor's exact resized image: pixel_values `max_abs<1e-6`,
    positions identical over all 2394 valid patches.
  - [x] **Decode + resize chosen as inferlet-side (option B).** The inferlet
    decodes (`image` crate) + resizes (SigLIP2 target, CatmullRom ≈ HF bicubic)
    + patchifies via the SDK `vision` module (`gemma_resize_target` +
    `gemma_patchify_hwc`, mirroring the verified host geometry; 2 unit tests),
    then sends `pixel_values`+`positions` through the new `media.from-pixels`
    WIT path. Host `HostImage::from_pixels` stores them + derives token_count.
    `inferlets/image-qa/` rebuilt to this path; **compiles to wasm**. Runtime +
    SDK compile. (Resize interpolation is the one non-exact step — CatmullRom
    vs HF bicubic+antialias — absorbed downstream at cosine 0.999.)
  - [ ] Wire `pixels`+`positions` from the `input_image` handler onto
    `ForwardRequest` (replacing the placeholder encoded-bytes side-channel), so
    they reach the driver view for 2.5's scatter.
- [ ] **2.4** Bidirectional sub-mask over the image span. (Encoder itself is
  bidirectional — `create_bidirectional_mask`.)
- [x] **2.5** Scatter projector rows into the image KV rows — DONE (image-conditioned e2e verified).
  - [x] **Wire + placeholder rows done — pipeline runs e2e.** `input_image`
    appends `token_count` placeholder rows (so the forward has KV slots; commit
    succeeds) and stages `pixel_values` (f32) + `image_patch_positions` +
    `image_anchor_rows` on `ForwardRequest` (bridge + runtime compile; cuda lib
    compiles with the new view fields). **Verified: `pie run image-qa` on
    `gemma-4-E4B` runs end-to-end** — vision binds, option-B preprocess →
    `280 soft tokens`, forward `N=280` rows commit (no more `append_image`
    error), generation produces an answer.
  - [x] **The scatter — DONE. Image-conditioned answer verified e2e.**
    `scatter_gemma4_vision` (in `gemma4_vision_forward.cu`) is called from
    `gemma4_forward_paged` right after `launch_embed_bf16` (unscaled — HF
    `masked_scatter`s post-`embed_scale`): it runs `run_gemma4_vision` over the
    view's pixels/positions and overwrites `h[anchor_row .. +n_soft]` with the
    projected `[n_tok, 2560]`. Threaded: `entry.cpp` attaches
    `bound.gemma4_vision` → `Gemma4Model` (`to_vis_raw`); image fields on
    `ForwardInputs` + `ForwardDispatchInputs`, filled from the view in
    `handle_fire_batch` (non-graph prefill path only).
    **`pie run image-qa` on `gemma-4-E4B` (stop-sign image) → "This image shows
    a red stop sign with the word STOP in white letters."** ✅ Multimodal e2e
    works.
- [ ] **2.6** Parity vs HF — reference dumps ready (see 2.2); compare CUDA stage
  outputs against `/tmp/gemma4_vision_parity/*.npy` (extend `parity_harness.cpp`).
- [x] **2.7** Example inferlet `inferlets/image-qa/` (fetch → `Image::from_bytes`
  → `Context::append_image` → generate); **compiles to wasm32-wasip2**.
  **e2e RUN on the real server + `gemma-4-E4B`** (`pie run`, cuda driver):
  ✓ server boots, model loads (2131 tensors, 15.2 GiB), **vision tower binds
  (16 layers)**, host `Image` resource works (fetch → 280 soft tokens), text
  forward runs. ✗ stops at `append_image: commit pages: need 272, have 7` —
  exactly 2.5: pages reserved for the image soft-tokens but the driver doesn't
  yet write their KV. Foundation verified; scatter (2.5) + wire pixels are the
  remaining integration, now precisely scoped by the run.

### Phase 3 — Qwen 3.6 (M-RoPE + DeepStack on working plumbing)
- [ ] M-RoPE: N-D positions through wire + RoPE kernel (gated on non-empty).
- [ ] native-resolution ViT + 2×2 merge + DeepStack multi-layer injection.
- [ ] `smart_resize` pixel path.
- [ ] Parity vs HF.

### Phase 4 — Video
- [ ] `image::from-frames` temporal grid + timestamps in processor.
- [ ] `Context::append_video` + `frame_count()`.
- [ ] Interior KV eviction (`Context::evict_frames`) in the runtime.
- [ ] (opt) encoder-output cache for re-materialization.
- [ ] Streaming inferlet example.

### Phase 5 — portable/GGML mirror, GLM 5.1 / DeepSeek-V4, Gemma audio.

---

## 10. Open questions / decisions

1. **Frame decode location** — inferlet-side (matches `image-fetch`, flexible,
   slower/frame) vs a host video-decode helper. *Leaning inferlet-side for v1.*
2. **Eviction in scope for first video cut?** — or ship video with mask-only
   bounded memory (pages resident) and add real reclamation later.
3. **Encoder-output cache** — context KV already caches the *result*; a separate
   embedding cache only saves re-running the ViT on a masked/evicted frame.
   Defer unless re-attention is common.
4. **Qwen 3.6 exact processor constants** (`patch_size`, `factor`, pixel
   bounds) — confirm against the shipped HF config; `multimodal.rs` constants
   are marked `VERIFY`.

---

## 11. Parity strategy

Every encoder lands behind a parity check against HF reference activations using
`driver/cuda/src/parity_harness.cpp`: dump HF `pixel_values` + projected
embeddings for a fixed image, compare the driver's encoder output row-for-row
before wiring the splice. Geometry (`multimodal.rs`) is unit-tested against
known HF processor outputs independently.

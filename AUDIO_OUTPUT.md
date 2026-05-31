# Native audio OUTPUT for Pie (CSM-1B + Mimi)

This is the **output-modality** design — the inverse of the input modalities in
`MULTIMODAL.md`. Input is *perception → text*: an encoder turns pixels / mel
frames into soft tokens, those scatter into the KV cache, and the model
generates text. **Output inverts every arrow:** the model **emits** audio codec
tokens, and a neural codec **decodes** them back to a waveform.

```
INPUT  (built):  waveform/pixels → encoder → soft tokens → scatter into KV → generate text
OUTPUT (this):   text + audio history → EMIT codec tokens → codec decoder → stream PCM out
```

Target model: **CSM-1B** (Sesame Conversational Speech Model) +
the **Mimi** neural codec. Verified against transformers 5.9
(`CsmForConditionalGeneration`, `MimiModel`); configs read from the local
checkpoints (`eustlb/csm-1b`, `kyutai/mimi`).

---

## 1. The model, exactly

CSM is **two stacked Llama-like transformers plus a codec**:

| Piece | Arch | Role |
|---|---|---|
| **backbone** | Llama-3.2-1B: 16 layers, hidden **2048**, 32 heads / 8 KV, head_dim 64, SiLU MLP 8192, RMSNorm, RoPE θ=500000 llama3-scaled | Given text + audio history, predicts **codebook 0** of the next Mimi frame + emits its last hidden state |
| **depth decoder** | small Llama: **4 layers**, hidden **1024**, 8 heads / 2 KV, head_dim 128, SiLU MLP 8192, RoPE θ=500000 (orig_max_pos 16), max_pos **33** | Given (backbone_hidden, cb0), autoregressively predicts **codebooks 1..31** for that one frame (RVQ depth) |
| **Mimi codec** | SplitRVQ (1 semantic + 31 acoustic) + 8-layer transformer + SEANet | 32 RVQ codes/frame → 24 kHz waveform |

Shared/auxiliary weights: `embed_text_tokens` [128256, 2048] (text vocab),
`lm_head` [2051, 2048] (codebook-0 logits), `backbone_model.embed_tokens`
(`CsmBackboneModelEmbeddings`: a [32·2048, 2048] audio-token embedding, summed
over the 32 codebooks of the previous frame), and the depth decoder's
`embed_tokens` [32·2051, 1024] + `inputs_embeds_projector` [1024←2048] +
`codebooks_head` [31, 1024, 2051] (a per-codebook LM head). Audio-token embeds
are **tied** between backbone and depth decoder
(`tie_codebooks_embeddings: true`).

`num_codebooks = 32`, per-codebook `vocab_size = 2051` (2048 codes + EOS/pad
specials). One Mimi frame = 12.5 Hz; the SEANet upsamples ×1920 → 24 kHz.

### 1.1 The generation loop (the genuinely new control flow)

Transcribed from `generation_csm.py::CsmGenerationMixin._sample` (transformers
5.9). **Per output frame:**

1. **Backbone step.** Run the backbone over the current row (text prompt on the
   first step; the previous frame's 32 codebook tokens, embedded-and-summed,
   thereafter). Take `lm_head(last_hidden[:, -1])` → sample **codebook 0**
   (`next_tokens`). Keep `backbone_last_hidden_state = hidden[-1][:, -1, :]`
   ([2048]).
2. **Depth-decoder sub-loop (RVQ depth).** Seed depth input as
   `[placeholder=0, cb0]` ([B, 2]); the placeholder slot 0 is overwritten by
   `backbone_last_hidden_state` inside `CsmDepthDecoderModel.forward`. Then
   **generate 31 tokens**, one per remaining codebook: at depth position `i`
   (1-indexed codebook `i`), embed the just-sampled code with offset
   `codebook_idx · vocab_size`, project 2048→1024, run the 4 layers, and apply
   `codebooks_head.weight[i-1]` to get that codebook's logits → sample. This
   inner loop is short (fixed length 31) and has its **own KV cache** of length
   ≤ 33.
3. **Assemble the frame.** The 32 codes `[cb0, cb1..cb31]` are the next Mimi
   frame. Append them as the backbone's next input row.
4. **Decode + stream.** Hand the frame's 32 codes to the Mimi decoder → 1920
   PCM samples (80 ms) → stream out.
5. **Stop** when every codebook of the new frame equals `codebook_eos_token_id`
   (0), or `max_new_tokens` frames produced.

The crucial departure from text generation: **one backbone step produces 32
tokens, not 1**, via the nested depth-decoder loop. This breaks Pie's
one-token-per-step assumption (see §3).

### 1.2 Mimi decode path (codes → waveform), exactly

From `modeling_mimi.py::MimiModel.decode → _decode_frame` (config: hidden 512,
codebook_dim 256, 32 quantizers, num_filters 64, ratios [8,6,5,4], 24 kHz):

```
codes [B, 32, T]
  → quantizer.decode                  (MimiSplitResidualVectorQuantizer)
        cb0      → semantic RVQ  (1 codebook):  embed row [2048,256], summed (trivially)
        cb1..31  → acoustic RVQ (31 codebooks): per-codebook embed [2048,256], SUMMED
        each group: 1×1 conv output_proj 256→512; the two 512-d results ADDED   → [B, 512, T]
  → upsample                          (MimiConvTranspose1d k4 s2 groups=512, no bias) → [B, 512, 2T]   (12.5→25 Hz)
  → decoder_transformer               (8× MimiTransformerLayer:
                                        LayerNorm(+bias) → RoPE attn (sliding window 250,
                                        8 heads, head_dim 64) × self_attn_layer_scale →
                                        LayerNorm(+bias) → GELU MLP (512→2048→512) × mlp_layer_scale;
                                        pre-norm residual)                          → [B, 2T, 512]
  → decoder (SEANet)                  (Conv1d k7 512→1024; then ×4 [ELU →
                                        ConvTranspose1d (ratios 8,6,5,4) →
                                        MimiResnetBlock(dim→dim/2 k3 → dim k1, +residual)];
                                        ELU; Conv1d k3 64→1)                         → [B, 1, 1920·T]  @ 24 kHz
```

Note the SEANet decoder reuses the same `MimiConv1d` (causal, left-pad
`padding_total = (k-1)·dilation + 1 − stride` + `_get_extra_padding`) and
`MimiConvTranspose1d` (raw transposed conv, then trim `padding_right` from the
right with `trim_right_ratio = 1.0`) building blocks as the Mimi *encoder*
— mirrored exactly the way the gemma4 audio *encoder* convs are.

---

## 2. What REUSES Pie's existing machinery

The whole point of the CSM choice: **the backbone is a stock Llama-3.2-1B**, so
it reuses Pie's `llama_like` forward verbatim. Reuse map:

| Component | Reuses | File / anchor |
|---|---|---|
| **Backbone forward** (16-layer Llama) | `llama_like_forward_paged` + `LlamaLikeForwardCfg` (RopeKind::YaRN for llama3 scaling, GQA group 4, RMSNorm pre-norm) | `driver/cuda/src/model/llama_like.{hpp,cpp}` |
| Backbone RoPE config | `apply_rope_config` / `rope_kind_from_hf_config` (llama3 → YaRN) | `llama_like.hpp:166-174` |
| Backbone weights container | `Qwen3Weights` / `Qwen3Workspace` (the shared llama-like weight/workspace structs) | `model/qwen3.hpp`, `model/qwen3_forward.hpp` |
| KV cache, paging, decode-plan | `KvCache`, `LlamaLikePlanState`, `prepare_llama_like_decode_plan` | `llama_like.hpp:101-131` |
| **Fork / snapshot / prefix-cache / market** | unchanged — codec tokens are ordinary KV rows, exactly as an encoded image is | `MULTIMODAL.md` §"fork / snapshot…" |
| Sampling (codebook 0, codebook i) | the existing logits → sampler path (multinomial / top-k) | `runtime/wit/core/wit/inference.wit:68` `variant sampler` |
| Mimi conv/transposed-conv/resnet kernels | **same building blocks** as the gemma4 audio encoder convs (causal pad, depthwise conv) | `model/gemma4_audio_forward.cu` (kernel style) |
| Parity convention (bf16-vs-bf16 rel_rms + cosine) | `MULTIMODAL.md` §2.2 | — |

The backbone is just *another llama_like model with a non-text embedding front
end and a small alternate head*. The audio-token embedding
(`CsmBackboneModelEmbeddings`: sum of 32 per-codebook embeds with offsets) is the
analogue of the input-modality "scatter", but it runs on the *model's own*
emitted tokens rather than an external encoder's soft tokens.

---

## 3. The genuinely-new engine pieces

Three things do not exist yet:

### 3.1 Depth-decoder forward (`CsmDepthDecoderForCausalLM`)
A 4-layer Llama with **two twists** the plain `llama_like` forward doesn't do:
- input embedding is `embed_tokens(input_id + codebook_idx·vocab_size)` then a
  `inputs_embeds_projector` (2048→1024) Linear, **and position 0 is replaced by
  the backbone's last hidden state** (the projected backbone hidden seeds the
  depth sequence);
- the head is **position-specific**: `codebooks_head.weight[i-1]` (a
  [1024, 2051] slice) is used at depth step `i`, not a single shared `lm_head`.

Everything else (RoPE, RMSNorm, GQA attention, SiLU MLP) is `llama_like`. So
the implementation is `llama_like` layers + a thin custom embed-front and
per-codebook head selection. KV cache here is tiny (≤ 33 positions) and
**per-frame ephemeral** — it resets every backbone step. Proposed file:
`driver/cuda/src/model/csm_depth_decoder_forward.{hpp,cu}` (or fold into a
`csm.{hpp,cpp}` model that owns both transformers).

### 3.2 Multi-codebook RVQ sampler (breaks the 1-token/step loop)
Pie's generation loop assumes one sampled token advances the sequence by one.
CSM advances by one **frame = 32 tokens**, produced by a nested 31-step loop.
The engine needs a "generate one audio frame" primitive that:
1. runs the backbone once, samples cb0 from `lm_head`,
2. runs the depth-decoder 31-step inner loop (its own mini KV cache) to get
   cb1..31,
3. returns the 32-vector frame and re-embeds it as the next backbone input.

This is a new control structure in the serving loop — the backbone's
KV/decode-plan is the *outer* loop; the depth decoder is an *inner* fixed-length
loop with a separate, reset-per-frame cache.

### 3.3 Mimi decoder module (codes → waveform) — **scaffolded in this change**
`driver/cuda/src/model/mimi_decoder_forward.{hpp,cu}` +
`mimi_decoder_adapter.{hpp,cpp}`: RVQ dequantize (32 codebooks, split
semantic/acoustic) → 1×1 output_proj → upsample → 8-layer decoder transformer →
SEANet transposed-conv stack → waveform. Raw-pointer `MimiDecoderRawWeights`
interface, CUDA-only header (no toml++), bf16 storage / fp32 compute — the exact
mirror of `gemma4_audio_forward.{hpp,cu}`. Parity-testable standalone against
`/tmp/mimi_decoder_parity/` (codes → waveform) via
`scripts/mimi_decoder_parity_ref.py`. `// PARITY TODO:` markers sit at the conv
padding/stride math, the transposed-conv trim, the RVQ sum order, and the
sliding-window attention.

---

## 4. Proposed WIT / SDK surface

Mirror `media.wit`'s handle pattern, but **outbound**. An audio output is a
host-side resource the inferlet drives and streams from — the PCM never has to
round-trip through WASM linear memory unless the inferlet asks for the bytes.

### 4.1 `pie:core/audio-out` (new WIT interface)

```wit
// Native audio OUTPUT — the inverse of media.wit. See AUDIO_OUTPUT.md.
interface audio-out {
    use types.{error};
    use model.{model};
    use context.{context};
    use wasi:io/poll@0.2.4.{pollable};

    // A host-side streaming audio generation, driven from a text/audio context.
    // The engine owns the backbone + depth-decoder + Mimi decoder; the inferlet
    // only steps it and pulls PCM. Fork/KV/market apply to the driving context.
    resource speech-stream {
        // Start generating from the current context (text prompt already in KV).
        // `speaker` selects the CSM speaker id; `max-frames` caps length.
        start: static func(model: borrow<model>,
                           ctx: borrow<context>,
                           speaker: u32,
                           max-frames: u32) -> result<speech-stream, error>;

        // Advance generation by one Mimi frame (backbone step + 31-step depth
        // loop + Mimi-decode). Returns the new 24 kHz mono PCM as f32 samples
        // (1920 per frame), or an empty list once the stream hits audio-EOS.
        next-chunk: func() -> result<list<f32>, error>;

        // Non-blocking readiness for the next chunk (async forward).
        pollable: func() -> pollable;

        // The raw 32-codebook codes for the last frame (for debugging / re-decode).
        last-frame-codes: func() -> list<u32>;   // length 32

        // True once audio-EOS frame was produced or max-frames reached.
        is-finished: func() -> bool;
    }
}
```

A higher-level convenience, mirroring `Context`'s text helpers:

```rust
// sdk/rust/inferlet — Context extension
impl Context {
    // text in KV → fully-decoded 24 kHz mono wav bytes (16-bit PCM).
    pub async fn generate_audio(&self, speaker: u32, max_frames: u32) -> Vec<u8>;
}
```

`generate_audio` loops `next-chunk` to completion, accumulates the PCM, and
wraps it in a WAV container. Streaming consumers call `next-chunk` directly and
forward each 80 ms chunk.

### 4.2 Example inferlet: `tts`
`inferlets/tts/` (mirror `inferlets/image-qa/` structure): take a text prompt +
speaker id, prefill the CSM text context, then either (a) POST the assembled WAV
to a callback URL via the existing `http`/messaging imports, or (b) stream
chunks. Demonstrates that fork/KV reuse still works — e.g. fork the prompt
context and generate two speakers' renditions in parallel.

---

## 5. How fork / KV / market still apply

The backbone is a normal causal LM over a token sequence whose rows happen to be
audio frames. Its KV cache is ordinary KV — so **fork, snapshot, prefix-cache,
and the market all work unchanged**, exactly as `MULTIMODAL.md` notes for
encoded images. Two caveats specific to output:
- The **depth-decoder cache is ephemeral** (reset per frame) and must NOT be
  forked/snapshotted with the backbone cache — it carries no cross-frame state.
- The **Mimi decoder is stateless across frames in the non-streaming path** used
  here (we decode each frame's 32 codes independently; `decode_frame`'s optional
  `decoder_past_key_values` streaming cache is a later optimization). So decode
  is a pure function of the frame's codes and needs no market/KV bookkeeping.

---

## 6. Status (this change)

- [x] `scripts/mimi_decoder_parity_ref.py` — **RUNS** against `kyutai/mimi`:
      codes [1,32,25] → waveform [1,1,48000]; dumps dequantized / upsampled /
      decoder-transformer / SEANet intermediates + 225 decoder weights + 32
      resolved codebook embeds (missing=0) to `/tmp/mimi_decoder_parity/`.
- [x] `scripts/csm_generate_ref.py` — **RUNS** against `eustlb/csm-1b`:
      "[0]Hello, this is a test." → 22 frames × 32 codebooks → 1.76 s wav +
      `output.wav`; dumps backbone/depth/codec configs + 12 probe weights.
- [x] `driver/cuda/src/model/mimi_decoder_forward.{hpp,cu}` +
      `mimi_decoder_adapter.{hpp,cpp}` — Mimi decoder scaffold, **compiles clean
      under nvcc 13** (CUDA-only header). Parity TODOs marked at conv
      padding/stride, transposed-conv trim, RVQ sum, sliding-window attention.
- [x] Mimi decoder bf16-vs-bf16 parity harness
      `driver/cuda/tests/mimi_decoder_full_parity.cu` — **PASSES** against
      `/tmp/mimi_decoder_parity/`: output waveform cosine **0.99993** (rel_rms
      1.19%), every staged intermediate ≥ 0.99993. The Mimi decoder is now
      verified end-to-end. Registered in `CMakeLists.txt` (lib build is clean).
- [x] `scripts/csm_depth_decoder_parity_ref.py` — **RUNS** against
      `eustlb/csm-1b`: dumps, for a chosen frame, the backbone hidden seed + cb0
      + the ACTUAL emitted cb1..31 (not the round-trip re-encode) + per-step
      logits + all 43 depth weights (bf16) to `/tmp/csm_depth_parity/`.
- [x] `driver/cuda/src/model/csm_depth_decoder_forward.{hpp,cu}` — the
      depth decoder + RVQ frame sampler (the genuinely-new engine piece).
      Standalone harness `driver/cuda/tests/csm_depth_decoder_parity.cu`
      **PASSES**: all 31 emitted codes match HF's greedy argmax exactly on
      frames 0 and 5 (31/31), 30/31 on frame 21 (the one miss is a 0.0096-logit
      bf16 argmax tie; logits cosine **0.99996 / 0.99992 / 0.99984**). Built into
      the driver lib via `CMakeLists.txt`.
      Build:
        `nvcc -O2 -arch=sm_89 -std=c++17 -I driver/cuda/src \`
        `  driver/cuda/tests/csm_depth_decoder_parity.cu -o /tmp/cdp; /tmp/cdp /tmp/csm_depth_parity`
- [x] `runtime/wit/core/wit/audio-out.wit` — the `pie:core/audio-out`
      `speech-stream` interface (design contract; staged, not yet added to the
      `inferlet` world — see the file's status note).
- [x] `inferlets/tts/` — TTS inferlet (builds to `wasm32-wasip2`): CSM
      `[speaker]text` prompt assembly + a complete PCM→WAV writer (inverse of
      audio-qa's parser) + base64 output; `generate_audio` is the documented
      call site for the `speech-stream` host import.
- [ ] **Remaining**: host-side serving-loop integration — loader config parse
      for `model_type:"csm"`, `bind_csm` (backbone reuses `llama_like` + bind
      depth weights + `bind_mimi_decoder`), the frame-stepped generation
      primitive in the executor (outer backbone step → inner 31-step depth loop
      via `run_csm_depth_decoder_frame` → Mimi-decode via `run_mimi_decoder`),
      and wiring `audio-out` into the runtime + SDK `Context::generate_audio`.
      The two genuinely-new CUDA pieces this depends on are done + verified; what
      remains is plumbing through the existing (proven) backbone + serving path.
      e2e oracle: `scripts/csm_depth_decoder_parity_ref.py` emits the actual
      codes; decoding them through the verified Mimi decoder yields a 1.76 s
      24 kHz wav (`/tmp/csm_emitted.wav`).

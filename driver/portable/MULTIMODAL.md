# Portable driver: multimodal support

The portable (ggml) driver runs the same multimodal model scope as the CUDA
driver. Validated on **Metal and CPU**; see "Backend scope" below for other
backends.

| Capability        | Model            | Notes                                            |
|-------------------|------------------|--------------------------------------------------|
| Vision (image)    | Qwen3-VL, Gemma-4| single + multiple images per forward             |
| Vision (video)    | Qwen3-VL         | frames sampled host-side; M-RoPE temporal axis   |
| Audio input       | Gemma-4          | Conformer encoder; single + multiple clips       |
| Audio output (TTS)| CSM-1B           | backbone + depth RVQ decoder + Mimi vocoder       |

## Architecture

Preprocessing is host-side (`runtime/src/multimodal.rs`): decode, resize,
patchify, mel, and the `(t,h,w)` patch grid / M-RoPE positions, matching the HF
processors. The driver receives f32 pixel values / log-mel features plus anchor
rows, runs the vision/audio encoder, and scatters the resulting soft-token
embeddings into the language model's token-embedding rows at their anchors. The
wire format already carries everything (`image_*`, `audio_*` slices in
`pie_bridge` / `view.hpp`); no ABI changes were needed.

Per-modality graph builders:

- `graph_vision_qwen3vl.cpp` — ViT (full bidirectional attention, blocked 2D
  RoPE, learned pos-embed interpolation, 2x2 patch merger, deepstack mergers).
- `graph_vision_gemma4.cpp` — SigLIP-style ViT (2D RoPE, avg-pool, sandwich
  norms, attention scale 1.0).
- `graph_audio_gemma4.cpp` — Gemma-4 Conformer audio encoder.
- `graph_csm_gen.cpp` / `graph_csm_mimi.cpp` — CSM backbone + fused depth
  decoder + Mimi vocoder.

## Key design decisions

**Qwen3-VL M-RoPE.** The language model applies true per-token `[t,h,w]` M-RoPE
via `ggml_rope_multi` (`graph_qwen3.cpp`, gated on `hparams.use_mrope`). The
executor merges the 1-D positions (text rows → `[p,p,p]`) with the wire's
`image_mrope_positions` (image/video rows → `[t,h,w]`) in
`build_mrope_positions_`, mirroring the CUDA driver. `graph_common.cpp` lays the
4×-wide `pos_input` out global-axis-major because the dense graph ropes the whole
batch at once. Text-only tokens have `t=h=w=p`, so M-RoPE reduces to plain RoPE
and text generation is unchanged. Qwen3.5's separate per-request mrope path is
untouched.

**CSM keeps f32.** CSM's 31-step sequential RVQ depth decode is precision
critical: bf16 weights flip near-tie argmaxes and the error compounds, drifting
the audio codes off the reference. The model is small, so CSM is loaded in f32
unconditionally (`model.cpp`, `keep_f32`). With f32 the codes are bit-exact vs
the HF reference (96/96). This is a deliberate correctness-over-speed tradeoff
(~2x backbone decode vs bf16).

**Fused CSM depth decoder.** The 31 codebook steps run in one graph with in-graph
`ggml_argmax` chaining each step's code into the next step's embedding, threading
per-layer KV handles across the unrolled steps. Each step reads exactly its valid
cache prefix (`Lkv = p+1`, a compile-time constant) so the attention reduction
width matches the per-step reference bit-for-bit. This collapses ~31 host-argmax
syncs/frame to one compute.

**No silent CPU fallback.** The driver announces its selected backend at load and
flags `(!! no GPU backend active)` if it landed on CPU when a GPU was expected.
`PIE_PORTABLE_STRICT_METAL` turns an unsupported-op scatter into a hard error
instead of a quiet partial CPU offload.

## Backend scope

The multimodal graph builders are compiled into the portable driver on every
backend, but are **validated on Metal and CPU only**. CSM and Mimi use
`ggml_backend_graph_compute` directly, so an op the active backend doesn't
support aborts mid-encode rather than falling back. On non-Metal GPU backends
(notably Vulkan) the driver emits a startup warning that multimodal is
unvalidated there. CPU works because the CPU backend implements every op.

## Diagnostics

- `PIE_CSM_DUMP=1` — dump per-frame CSM codebook indices.
- `PIE_PORTABLE_KEEP_F32=1` — keep all f32 source weights in f32 (no bf16
  downcast); useful for isolating precision from logic.
- `PIE_PORTABLE_STRICT_METAL=1` — hard-error on any CPU-offloaded op.

## Tests

End-to-end inferlets exercise each modality: `image-qa`, `video-qa`,
`multi-image-test`, `audio-qa`, `multi-audio-test`, `tts`. Vision and audio-input
are validated by output correctness; CSM TTS is validated bit-exact against a
saved reference frame set.

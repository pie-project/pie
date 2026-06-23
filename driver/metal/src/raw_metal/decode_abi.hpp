#pragma once
// decode_abi.hpp — raw-Metal Phase-0 decode-step ABI for qwen3.6 (M=1 single-stream greedy).
//
// Backend-agnostic contract shared by THREE lanes:
//   * delta  — MTLHeap allocator (region offsets) + the M=1 kernel ports.
//   * alpha  — Metal-4 wrappers (MTL4ArgumentTable / MTLHeap+MTLResidencySet /
//              double-buffered MTL4CommandAllocator) keyed off the binding enums here.
//   * beta   — command-buffer encoder + replay/latency, encoding against these stable slots.
//
// NO Metal/ObjC types in this header — pure layout, indices, and geometry, so every lane
// includes it without a Metal dependency.
//
// ── Invariants (locked in #mac) ──────────────────────────────────────────────
//   I1. IO scalars (token_id / position / seq_len) are GPU-READ BUFFER slots, NEVER
//       encode-time setBytes/push-constants. This makes the encoded command buffer
//       BYTE-IDENTICAL every token → encode(N+1) overlaps GPU(N) (token-independent CB).
//       The ported kernels MUST read these from the IO buffers (MLX bakes them as
//       set_bytes; rewire to buffer-sourced scalars).
//   I2. One MTLHeap, fixed region offsets; MTLResidencySet made resident ONCE;
//       MTL4ArgumentTable built ONCE. Per-token only IO slot CONTENTS change.
//   I3. Logits are ALWAYS produced into the IO region (pie-core is the sampling
//       authority + the cosine gate reads them). Device-argmax is an OPTIONAL substrate
//       token; self-feeding N steps is HARNESS-ONLY, never a driver decode loop.
//   I4. Single-stream S=1: GDN recurrent/conv state is RESIDENT and updated in-place at a
//       fixed heap address each token (no take/scatter). KV is append-only.

#include <cstddef>
#include <cstdint>

namespace pie_metal_driver::raw_metal {

// ── Geometry (Qwen3.5-0.8B, 4-bit g64, tied embeddings) ──────────────────────
struct DecodeGeometry {
    // global
    int   hidden            = 1024;
    int   n_layers          = 24;
    int   vocab             = 248320;
    float eps               = 1e-6f;
    // TIED, but the CANONICAL stored side is the top-level 4-bit `lm_head.{weight,
    // scales,biases}` (U32 [248320,128]-packed = [248320,1024] logical, group=64);
    // `embed_tokens.weight` is ABSENT in this checkpoint (alpha's st_probe, #mac).
    // heap_bind binds the ONE shared lm_head bundle to BOTH bind::Embed (EmbedGather
    // dequantizes row token_id) AND QmvLmHead/Argmax — single Weights slot, no
    // duplicate 127MB. The ported embed_gather kernel is already 4-bit dequant-gather.
    bool  tied_embeddings   = true;   // lm_head is the stored table; embed aliases it

    // full-attention layers
    int   n_q_heads         = 8;
    int   n_kv_heads        = 2;
    int   head_dim          = 256;    // q_dim = 2048, kv_dim = 512
    int   rotary_dims       = 64;     // partial_rotary_factor 0.25 * head_dim
    float rope_theta        = 1e7f;
    // mrope sections [11,11,10] (sum*2 = 64 = rotary_dims)
    int   mrope_section[3]  = {11, 11, 10};

    // GDN / linear-attention layers
    int   gdn_k_heads       = 16;
    int   gdn_v_heads       = 16;
    int   gdn_k_dim         = 128;
    int   gdn_v_dim         = 128;
    int   gdn_conv_k        = 4;
    int   gdn_conv_dim      = 6144;   // 2*Kh*Kd + Vh*Vd
    int   gdn_v_total       = 2048;   // Vh*Vd

    // mlp
    int   intermediate      = 3584;   // SiLU gate

    // quantization (affine)
    int   q_group           = 64;
    int   q_bits            = 4;

    // Layer schedule: full-attention at layers {3,7,11,15,19,23}; GDN otherwise.
    static constexpr int full_attn_interval = 4;
    static constexpr bool is_full_attn(int layer) {
        return (layer % full_attn_interval) == (full_attn_interval - 1);
    }
};

// ── Heap regions (one MTLHeap, fixed offsets) ────────────────────────────────
enum class Region : uint8_t {
    Weights = 0,  // load-once RO: qmv (w/scales/biases), norms, embed(=lm_head)
    KV      = 1,  // paged k/v pages for the 6 full-attn layers (append-only)
    State   = 2,  // GDN resident conv_state + recurrent_state (S=1, in-place)
    Scratch = 3,  // activation ping-pong pool (SCRATCH_POOL buffers)
    IO      = 4,  // per-token CPU/GPU-touched scalars + logits
};

// Scratch ping-pong pool size (beta, from WAR/WAW chain). 6 max-shape buffers.
inline constexpr int SCRATCH_POOL = 6;

// ── IO region slots (I1: all GPU-read buffers, never setBytes) ───────────────
enum class IoSlot : uint8_t {
    TokenId   = 0,  // u32[1]  — pie-core writes after sampling step N
    Position  = 1,  // u32[1]  — rope/kv_append read this
    SeqLen    = 2,  // u32[1]  — sdpa KV extent
    Logits    = 3,  // f32[vocab] OUT — ALWAYS produced (I3)
    NextToken = 4,  // u32[1]  — OPTIONAL device-argmax substrate (I3)
};

// ── Per-kernel binding indices (arg order the encoder binds into MTL4ArgumentTable) ──
// Grounded in MLX's host-side dispatch arg order, adjusted for I1 (scalars→buffers).
namespace bind {

// affine 4-bit GEMV (M=1): qmv_fast / qmv_quad. group=(32,2,1)/(32,1,1),
// grid=(1, ceil(N/bn), 1). Shapes baked as constants (contiguous, fixed).
enum class Qmv : uint8_t { W = 0, Scales = 1, Biases = 2, X = 3, Out = 4, K = 5, N = 6 };

// sdpa_vector (decode, single-pass): group=(1024,1,1), grid=(n_q_heads, 1, 1).
// SeqLenPtr is the IO::SeqLen buffer (I1), gqa_factor = n_q/n_kv = 4.
enum class Sdpa : uint8_t {
    Q = 0, K = 1, V = 2, Out = 3, GqaFactor = 4, SeqLenPtr = 5,
    KHeadStride = 6, VHeadStride = 7,
};

// rms_single_row: group=(row/N_READS), grid=(1,1,1).
enum class Rms : uint8_t { X = 0, W = 1, Out = 2, Axis = 3, Eps = 4 };

// q_gate_split: deinterleave 2x-wide q_proj output (qwen3.5 gated attn).
// qg[n_q,2,head_dim] -> Q[n_q,head_dim] + gate[n_q,head_dim]. Internal (no golden tag).
enum class QSplit : uint8_t { Qg = 0, QOut = 1, GateOut = 2, HeadDim = 3 };

// attn_gate: attn *= sigmoid(gate) before o_proj (golden tag `attn_gated`). In-place.
enum class AttnGate : uint8_t { Attn = 0, Gate = 1 };

// single-token rope (partial + mrope): PositionPtr is IO::Position (I1).
enum class Rope : uint8_t { Q = 0, K = 1, QOut = 2, KOut = 3, PositionPtr = 4, Theta = 5 };

// embedding gather: TokenIdPtr is IO::TokenId (I1).
enum class Embed : uint8_t { Table = 0, TokenIdPtr = 1, Out = 2 };

// KV append (in-place write at position): PositionPtr is IO::Position (I1).
enum class KvAppend : uint8_t { K = 0, V = 1, KPages = 2, VPages = 3, PositionPtr = 4 };

// GDN fused core (1 dispatch — beta): folds conv1d+silu + l2norm*2 + q-scale + gating +
// GQA-repeat + recurrent-step. Reads `mixed` (in-proj conv input) + per-layer conv/gating
// weights + resident conv/recurrent state; emits core_out[gdn_v_total].
// Buffer indices below MATCH beta's gdn_core.metal [[buffer(i)]] signature exactly.
//   * RecurrentState (2) is updated IN-PLACE (each (req,v-head,v-dim) row owned by one
//     threadgroup -> race-free; native [Vh,Vd,Kd], prong-1, no transpose).
//   * ConvState (1) is READ-ONLY; the shifted result is written to a PING-PONG
//     ConvStateOut (10) and swapped per token. In-place conv_state races (convsilu reads
//     the Kc-tap history while redundant v-dim threadgroups shift it — beta measured 5.5).
enum class GdnCore : uint8_t {
    Mixed        = 0,  // in-proj conv input (T)
    ConvState    = 1,  // conv history (float, RO)
    RecurrentState = 2,  // [Vh,Vd,Kd] (float, in/out, in-place)
    CoreOut      = 3,  // core_out[gdn_v_total] (out, T)
    ConvW        = 4,  // conv1d weight (T)
    ConvB        = 5,  // conv1d bias (T)
    ALog         = 6,  // A_log decay param (T)
    DtBias       = 7,  // dt_bias gating param (T)
    AGate        = 8,  // a_gate projection (T)
    BGate        = 9,  // b_gate projection (T)
    ConvStateOut = 10, // new_conv_state (float, out, ping-pong w/ ConvState)
    Params       = 11, // GdnCoreParams& (constant, static geometry, I1-safe)
};

// gated rmsnorm: rms(core_out) * silu(z).
enum class GatedRms : uint8_t { X = 0, Z = 1, W = 2, Out = 3, Eps = 4 };

// device argmax (optional substrate, I3): Logits → NextToken.
enum class Argmax : uint8_t { Logits = 0, NextToken = 1 };

}  // namespace bind

// ── Dispatch DAG kernel kinds (per-step encode order; see wiki mac-raw-metal-decode-dag) ──
// Counts (locked): GDN layer = 6 dispatches (GdnCore=1), full-attn = 11, mlp = 6.
// Total ≈ 1 + 18*6 + 6*11 + 24*6 + 3 ≈ 322 dispatches; barriers ≈ 250 (re-encode path,
// ICB out). Metal-4 measured encode ~0.05ms → GPU-exec is the gate; GdnCore=1 fusion the lever.
enum class Kernel : uint8_t {
    EmbedGather, Rms, QmvIn, GdnCore, GatedRms, QmvOut, Residual,
    QmvQ, QSplit, QmvK, QmvV, QNorm, KNorm, Rope, KvAppend, Sdpa, AttnGate, QmvO,
    QmvGate, QmvUp, SiluMul, QmvDown,
    FinalRms, QmvLmHead, Argmax,
};

}  // namespace pie_metal_driver::raw_metal

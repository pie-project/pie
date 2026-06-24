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
//
// ── Multi-batch relaxations (Phase-1; ALL no-ops at M=1, append-only, sealed M=1 path
//    unchanged — see mac-multibatch-gap-analysis) ──────────────────────────────
//   R1 (I1'): IO scalar slots widen to variable-length u32[max_tokens]; M=1 writes [0].
//             New batch-CSR IoSlots (QoIndptr/KvPage*/RsSlot*) carry per-request structure.
//   R2 (I4'): GDN state relaxes to slot-indexed base+stride (bind::GdnCore{SlotIds,
//             SlotStride}); slot 0 at S=1.
//   R3:       byte-identical-CB → byte-identical *within a shape bucket* (ForwardGraphKey).
//   R4:       add batched bind::Qmm (M>1 GEMM) beside bind::Qmv (M=1 GEMV fast path).
//   Geometry max_tokens/max_requests/max_slots/kv_page_size default to 1/1/1/32 → the
//   M=1 region sizing is byte-identical; pie-core sets the caps for an M>1 fire.

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

    // ── Multi-batch caps (Phase-1; M=1 leaves all at 1 → byte-identical M=1 sizing) ──
    // These size the variable-length IO / KV / recurrent-state regions. pie-core's
    // batch limits set them at construction; defaults of 1 preserve the sealed
    // single-stream allocation exactly (every relaxation below is a no-op at these).
    //
    // SOURCE OF TRUTH (alpha plumbs at construction; the executor flips these from
    // DecodeGeometry{} defaults=1 → these values ONLY when the M>1 path is green, so
    // the sealed 264 heap stays byte-identical until then):
    //   max_tokens   ← cfg.batching.max_forward_tokens      (config.hpp:31, e.g. 10240)
    //   max_requests ← cfg.batching.max_forward_requests    (config.hpp:32, e.g. 512)
    //   max_slots    ← rs_cache_slots = max_forward_requests (entry.cpp:127-128; the GDN
    //                  recurrent-state slot pool; 0 for non-hybrid archs)
    //   kv_page_size ← cfg.batching.kv_page_size            (config.hpp:28, 32)
    //   total_pages  ← cfg.batching.total_pages             (config.hpp:29, e.g. 1024 — the
    //                  paged-KV pool; sizes KvPageIndices[total_pages] + the KV page region)
    // delta's build_bound_decode sizes GDN conv/recurrent slabs × max_slots, IO × max_tokens.
    int   max_tokens        = 1;    // N cap — batch token dim (total_tokens across reqs)
    int   max_requests      = 1;    // R cap — concurrent sequences in a fire
    int   max_slots         = 1;    // S cap — recurrent-state (GDN) slots
    int   kv_page_size      = 32;   // paged-KV page granularity (config kv_page_size)
    int   total_pages       = 1;    // paged-KV pool capacity (config total_pages); sizes
                                    // KvPageIndices[total_pages] + the KV page region (delta)

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
// M=1: scalars are written at [0] only (the sealed single-stream path is unchanged).
// M>1 (Phase-1, no-op at M=1): the scalar slots widen to variable-length [max_tokens]
// and the batch CSR slots below carry the per-request structure. Enum values are
// APPEND-ONLY (existing M=1 ordinals 0-4 are frozen) so delta's M=1 kernels never move.
enum class IoSlot : uint8_t {
    TokenId   = 0,  // u32[max_tokens] — M=1 writes [0]; M>1 = per-token ids [N]
    Position  = 1,  // u32[max_tokens] — absolute positions [N] (rope/kv_append read)
    SeqLen    = 2,  // u32[max_tokens] — per-token kv extent (M=1: [0]; sdpa reads)
    Logits    = 3,  // f32[vocab] OUT (M=1) — ALWAYS produced (I3); M>1 packs sampled rows
    NextToken = 4,  // u32[max_tokens] — OPTIONAL device-argmax substrate (I3)
    // ── Multi-batch CSR slots (Phase-1; bound only at M>1, absent at M=1 = no-op) ──
    // Canonical schema names (view.hpp / model_graph.hpp), marshaled by alpha from
    // PieForwardRequestView. r = request index; per-token request via qo_indptr search.
    QoIndptr       = 5,  // u32[R+1] — per-request token spans [qo_indptr[r]..r+1)
    KvPageIndptr   = 6,  // u32[R+1] — page-list base per req (pages_first = [r])
    KvPageIndices  = 7,  // u32[total_pages] — flat physical page ids
    KvLastPageLens = 8,  // u32[R] — fill count of each request's last page
    RsSlotIds      = 9,  // u32[R] — recurrent-state (GDN) slot per request
    RsSlotFlags    = 10, // u8[R]  — per-slot NEW(reset)/CONTINUE flag
    ReqOfToken     = 11, // u32[N] — per-token request id; ENTRY populates from beta's
                         // batch_schedule::tok_req (= qo_indptr expansion). Read by
                         // bind::KvAppend::ReqOfToken + bind::SdpaPaged::ReqOfToken.
};

// ── Per-kernel binding indices (arg order the encoder binds into MTL4ArgumentTable) ──
// Grounded in MLX's host-side dispatch arg order, adjusted for I1 (scalars→buffers).
namespace bind {

// affine 4-bit GEMV (M=1): qmv_fast / qmv_quad. group=(32,2,1)/(32,1,1),
// grid=(1, ceil(N/bn), 1). Shapes baked as constants (contiguous, fixed).
enum class Qmv : uint8_t { W = 0, Scales = 1, Biases = 2, X = 3, Out = 4, K = 5, N = 6 };

// affine 4-bit GEMM (M>1 prefill/batch) — Phase-1 reservation, no-op at M=1.
// C[M,N] = X[M,K] · dequant(W[N,K]); M = total batch rows (= N tokens in the fire).
// bind::Qmv (GEMV) stays the M=1 / single-decode-row FAST PATH; the dispatch selects
// the qmm PSO only when M>1 (mirrors CUDA's decode-GEMV vs prefill-GEMM split). Same
// W/Scales/Biases ordinals as Qmv so the weight binds are shared; X/Out widen to [M,*]
// and M is appended (ordinal 7) — Qmv ordinals 0-6 are frozen.
enum class Qmm : uint8_t { W = 0, Scales = 1, Biases = 2, X = 3, Out = 4, K = 5, N = 6, M = 7 };

// sdpa_vector (decode, single-pass): group=(1024,1,1), grid=(n_q_heads, 1, 1).
// Matches sdpa_vector.metal exactly. N (buffer 5) is the per-token kv_len, bound to
// the IO::SeqLen buffer (I1, setAddress); gqa_factor = n_q/n_kv = 4; strides + scale
// are geometry consts (decode_consts). k/v cache layout [n_kv_heads, max_ctx, head_dim]:
//   *_head_stride = max_ctx*head_dim, *_seq_stride = head_dim. scale = 1/sqrt(head_dim).
enum class Sdpa : uint8_t {
    Q = 0, K = 1, V = 2, Out = 3, GqaFactor = 4, N = 5,
    KHeadStride = 6, KSeqStride = 7, VHeadStride = 8, VSeqStride = 9, Scale = 10,
};

// batched paged-attention READ (M>1, beta's sdpa_paged.metal) — Phase-1, no-op at
// M=1 (M=1 uses bind::Sdpa over the contiguous ring). Gathers K/V from the paged
// cache via delta's phys_slot: kv_page_indices[kv_page_indptr[r] + kp/PageSize]
// *PageSize + kp%PageSize, element (slot*NKvHeads + h)*head_dim + d. Causal bound =
// PositionIds[row] (query attends kv [0..pos]); seqlen falls out of position, so NO
// KvLastPageLens read. ReqOfToken = IoSlot::ReqOfToken (beta's batch_schedule::tok_req).
// gemma4: per-layer head_dim (256/512) + kv_source redirect threaded host-side.
enum class SdpaPaged : uint8_t {
    Q = 0, KPages = 1, VPages = 2, Out = 3, GqaFactor = 4,
    PositionIds = 5,     // IoSlot::Position widened [N] — per-query causal bound
    ReqOfToken = 6,      // IoSlot::ReqOfToken [N]
    KvPageIndices = 7,   // IoSlot::KvPageIndices
    KvPageIndptr = 8,    // IoSlot::KvPageIndptr
    PageSize = 9, NKvHeads = 10, Scale = 11,
};

// rms_single_row: group=(row/N_READS), grid=(1,1,1). Buffer 3 is a packed
// RmsParams{eps, axis_size, w_stride, plus_one}. qwen3.5 sets plus_one=1 for ALL
// norms (attn_norm/q_norm/k_norm/ffn_norm/final_norm) → effective gain (1+weight).
enum class Rms : uint8_t { X = 0, W = 1, Out = 2, Params = 3 };

// residual add (golden `attn_resid`, `layer_out`): Out = X + Residual, elementwise.
enum class Residual : uint8_t { X = 0, Residual = 1, Out = 2 };

// SwiGLU (golden `swiglu`): Out = silu(Gate) * Up, elementwise over intermediate.
enum class SiluMul : uint8_t { Gate = 0, Up = 1, Out = 2 };

// dense bf16 GEMV (M=1) for GDN `gdn_in_a`/`gdn_in_b` (in_proj_a/b stored DENSE
// bf16 [V_h,hidden], NOT 4-bit). Out[N] = sum_k W[N,K]*X[K], float accum.
enum class Dense : uint8_t { W = 0, X = 1, Out = 2, K = 3, N = 4 };

// q_gate_split: deinterleave 2x-wide q_proj output (qwen3.5 gated attn).
// qg[n_q,2,head_dim] -> Q[n_q,head_dim] + gate[n_q,head_dim]. Internal (no golden tag).
enum class QSplit : uint8_t { Qg = 0, QOut = 1, GateOut = 2, HeadDim = 3 };

// attn_gate: attn *= sigmoid(gate) before o_proj (golden tag `attn_gated`). In-place.
enum class AttnGate : uint8_t { Attn = 0, Gate = 1 };

// single-token rope (NeoX, partial). In-place on buffer 0; matches rope.metal exactly:
//   0=x (activation, in/out), 1=position (IO::Position, I1), 2=scale, 3=base=log2(theta),
//   4=head_dim (row stride; rope_dims/2 comes from grid.x). scale/base/head_dim = consts.
enum class Rope : uint8_t { X = 0, Position = 1, Scale = 2, Base = 3, HeadDim = 4 };

// embedding gather (4-bit dequant of the shared lm_head bundle; tied). TokenId is
// IO::TokenId (I1). Matches embed_gather.metal: W/Scales/Biases (same 4-bit packing
// as Qmv) + token id + out + hidden. Bound to the SAME lm_head slots as QmvLmHead.
enum class Embed : uint8_t { W = 0, Scales = 1, Biases = 2, TokenId = 3, Out = 4, Hidden = 5 };

// KV append (in-place write at position): Pos is IO::Position (I1). Matches kv_append.metal:
//   0=k_new, 1=v_new, 2=k_cache, 3=v_cache, 4=pos(IO), 5=head_dim, 6=k_head_stride
//   (=max_ctx*head_dim), 7=k_seq_stride (=head_dim). head_dim/strides = consts.
enum class KvAppend : uint8_t {
    K = 0, V = 1, KPages = 2, VPages = 3, PositionPtr = 4,
    HeadDim = 5, KHeadStride = 6, KSeqStride = 7,
    // ── Paged scatter (M>1, delta's mac-paged-kv-bridge) — Phase-1, no-op at M=1 ──
    // M=1 keeps the contiguous-ring scatter (dst via PositionPtr + *Stride above).
    // M>1 replaces dst with phys_slot = kv_page_indices[kv_page_indptr[r] + pos/PageSize]
    // *PageSize + pos%PageSize, NHD page row (slot*NKvHeads + h)*head_dim + d. Honors the
    // page table (no contiguity assumption); reduces to the single scatter at N=1.
    KvPageIndices = 8,  // u32[total_pages] — IO::KvPageIndices
    KvPageIndptr  = 9,  // u32[R+1]         — IO::KvPageIndptr (page_base[r])
    PageSize      = 10, // u32              — kv_page_size (geometry const, default 32)
    ReqOfToken    = 11, // u32[N]           — per-token request id (derived from qo_indptr)
    NKvHeads      = 12, // u32              — n_kv_heads (page-row stride const)
};

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
    // ── Slot-indexed state (I4 relaxation, S>1) — Phase-1, no-op at S=1 ──
    // At S=1 these are UNBOUND (state lives at slot 0, in-place — the sealed path).
    // At S>1 the kernel indexes ConvState/RecurrentState/ConvStateOut by
    // rs_slot_ids[r]*slot_stride; per-(req,v-head,v-dim) threadgroup ownership keeps
    // the recurrent RMW race-free (direct template of CUDA slot_ids_d launches).
    SlotIds      = 12, // u32[R] — IO::RsSlotIds (per-request recurrent-state slot)
    SlotStride   = 13, // u32    — element stride between adjacent slots (= per-slot size)
};

// ── GdnCore prep-dispatch split (PIE_GDN_PREP, A/B behind the flag) ──────────
// Kills the in-kernel share's residual 4× redundancy (Vd/32 tiles) by hoisting the
// dv-INDEPENDENT q/k path out of the per-dv recurrent kernel into a single prologue
// dispatch computed ONCE per (req, v-head) — full 128×→1× elimination.
//   GdnPrep   : q/k conv1d+silu, l2norm, q-prescale, gating(decay,beta) → fp32 scratch
//               + q/k conv_state writeback. grid {32,1,R·Hv} tg {32,1,1}
//               (one simdgroup/head; 32 lanes × n_per_t=4 cover Dk=128).
//   GdnCoreRec: slimmed recurrent step — reads PreQ/PreK/PreGate from scratch (no
//               recompute), owns the per-dv v convsilu + recurrent RMW + v conv_state
//               writeback. grid {32,Vd,R·Hv} tg {32,4,1} (no dv-tile share needed).
// Bit-exact vs the in-kernel share: fp32 scratch preserves the exact `sh_q/sh_k` float
// values, same 32-lane Dk=128 simd_sum geometry. new_conv_state coverage invariant
// preserved: prep writes q/k channels once/head, rec writes v per-dv → each once.
enum class GdnPrep : uint8_t {
    Mixed        = 0,  // in-proj conv input (T)
    ConvState    = 1,  // conv history (float, RO)
    ConvW        = 2,  // conv1d weight (T)
    ConvB        = 3,  // conv1d bias (T)
    ALog         = 4,  // A_log decay param (float)
    DtBias       = 5,  // dt_bias gating param (T)
    AGate        = 6,  // a_gate projection (T)
    BGate        = 7,  // b_gate projection (T)
    PreQ         = 8,  // out: gdn_pre_q[R,Hv,Dk] fp32 (normalized + 1/sqrt(Dk)-prescaled q)
    PreK         = 9,  // out: gdn_pre_k[R,Hv,Dk] fp32 (normalized k)
    PreGate      = 10, // out: gdn_pre_gate[R,Hv,2] fp32 (decay, beta)
    ConvStateOut = 11, // new_conv_state q/k channels (float, out)
    Params       = 12, // GdnCoreParams& (constant)
};

// Slimmed recurrent core (prep-path). Buffers 0-5 MATCH bind::GdnCore (minimal
// heap_bind churn); old ALog/DtBias/AGate/BGate fold into PreGate; Params 11→10.
enum class GdnCoreRecurrent : uint8_t {
    Mixed        = 0,  // in-proj conv input (v channels) (T)
    ConvState    = 1,  // conv history (float, RO) — v
    RecurrentState = 2,  // [Vh,Vd,Kd] (float, in/out, in-place)
    CoreOut      = 3,  // core_out[gdn_v_total] (out, T)
    ConvW        = 4,  // conv1d weight (for v convsilu) (T)
    ConvB        = 5,  // conv1d bias (T)
    PreQ         = 6,  // in: gdn_pre_q from GdnPrep (fp32)
    PreK         = 7,  // in: gdn_pre_k (fp32)
    PreGate      = 8,  // in: gdn_pre_gate (fp32)
    ConvStateOut = 9,  // new_conv_state v channels (float, out)
    Params       = 10, // GdnCoreParams& (constant)
};

// gated RMSNorm (GDN; golden `gdn_core` = post-norm): Out = (1+0)·rmsnorm(X)·silu(Z)
// over V_d per V_head. W = gate_norm_w (RAW, F32, NO +1). Buffer 4 = GatedRmsParams
// {eps, vd}. Gates against golden `gdn_core` (gated-RMSNorm folded in).
enum class GatedRms : uint8_t { X = 0, Z = 1, W = 2, Out = 3, Params = 4 };

// device argmax (optional substrate, I3): Logits → NextToken.
enum class Argmax : uint8_t { Logits = 0, NextToken = 1 };

}  // namespace bind

// ── Dispatch DAG kernel kinds (per-step encode order) ──
// Surface locked vs charlie's authoritative golden tag set (wiki mac-golden-kernel-surface).
// Every golden tag maps 1:1 to a Kernel kind below; kinds sharing a .metal/PSO differ only
// by dispatch dims + golden tag (e.g. all Qmv* -> affine_qmv; Rms/FfnRms/QNorm/KNorm/FinalRms
// -> rms_single_row; RopeQ/RopeK -> rope; AttnResid/LayerOut -> residual_add). Internal
// (untapped) kinds: QSplit, KvAppend, GdnCore-pre (beta's pre-GatedRms core_out).
//
// Counts (qwen3.5, 6 full-attn + 18 GDN + 24 mlp): full-attn layer = 20 dispatches
// (18 golden taps + QSplit + KvAppend), GDN layer = 15 (14 taps; `gdn_core` tap covers
// GdnCore+GatedRms = 2 dispatches), layer-less = 3 (embed/final_norm/logits).
// Total ≈ 3 + 6*20 + 18*15 = 393 dispatches / 363 golden taps. Argmax = optional I3
// substrate (no golden tag). Metal-4 encode ~0.05ms → GPU-exec is the gate; GdnCore=1 the lever.
enum class Kernel : uint8_t {
    EmbedGather,
    // GDN in-projection (4 separate projections: qkv/z 4-bit, a/b DENSE bf16):
    Rms, QmvIn, QmvInZ, GdnInA, GdnInB, GdnPrep, GdnCore, GatedRms, QmvOut, Residual,
    // Full-attention block:
    QmvQ, QSplit, QmvK, QmvV, QNorm, KNorm, Rope, RopeK, KvAppend, Sdpa, AttnGate, QmvO,
    // Shared SwiGLU MLP block:
    FfnRms, QmvGate, QmvUp, SiluMul, QmvDown, LayerOut,
    // Layer-less tail:
    FinalRms, QmvLmHead, Argmax,
};

// ── Bucketed command-buffer key (relaxes "byte-identical CB" → "byte-identical
// within a shape bucket", Phase-1; M=1 decode = a single stable bucket) ──────
// Grid dims change with N/R, so the encoded CB is no longer byte-identical across
// fires of different shapes. Mirror CUDA `forward_graph.hpp` ForwardGraphKey: cache
// encoded CBs keyed by this; reuse on a hit, re-encode + cache on a miss. The
// double-buffered allocator is preserved. M=1 single-stream is the {R=1,N=1, one
// page bucket, pure-decode} key — stable every token, so encode(N+1)∥GPU(N) holds.
struct ForwardGraphKey {
    uint32_t n_requests   = 1;      // R
    uint32_t n_tokens     = 1;      // N = total_tokens across the batch
    uint32_t page_bucket  = 0;      // ceil(max_pages_in_batch / PAGE_BUCKET_GRAN)
    bool     is_pure_decode = true; // derived: every request has exactly 1 token (M=R)
    bool operator==(const ForwardGraphKey& o) const {
        return n_requests == o.n_requests && n_tokens == o.n_tokens &&
               page_bucket == o.page_bucket && is_pure_decode == o.is_pure_decode;
    }
    bool operator!=(const ForwardGraphKey& o) const { return !(*this == o); }
};

// Page-count bucketing granularity for ForwardGraphKey (coarsen num_pages so the CB
// cache doesn't thrash on every +1 page). Tune in Phase-1; CUDA buckets similarly.
inline constexpr uint32_t PAGE_BUCKET_GRAN = 8;

}  // namespace pie_metal_driver::raw_metal

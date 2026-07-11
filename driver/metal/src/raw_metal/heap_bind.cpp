// heap_bind.cpp — delta's Metal-side weight staging + per-ordinal arg-table binding.
//
// Stages every load-once weight (name registry: heap_bind_names.cpp), allocates the
// persistent GDN state + KV pages + IO scalars from heap_layout, then walks beta's
// build_decode_dag and binds each dispatch's WEIGHT / STATE / KV / IO slots BY ORDINAL.
// beta binds the per-dispatch activation/scratch X/Out (his WAR/WAW ping-pong) over the
// SAME ordinal space; alpha's RawMetalContext owns the heap + arg tables.
//
// Lane split (binds delta owns here):
//   * weights  — weight_binds(kind,layer): RO 4-bit/dense/norm tensors (bind::Qmv/Dense/Rms/...)
//   * state    — GdnCore ConvState/RecurrentState/ConvStateOut + a zeroed ConvB (no ckpt bias)
//   * kv       — KvAppend KPages/VPages + Sdpa K/V (the paged cache, per full-attn layer)
//   * io       — the I1 per-token buffers: Embed TokenId, Rope/KvAppend Position, Sdpa SeqLen,
//                QmvLmHead Out=Logits, Argmax Logits/NextToken (I3: logits live in IO)
// beta binds: scratch X/Out activations + const geometry params (setBytes-safe: identical
// every token, so the CB stays byte-identical — only the I1 IO buffer CONTENTS change).

#include "heap_bind.hpp"
#include "heap_bind_metal.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "heap_layout.hpp"
#include "decode_step.hpp"     // beta: Dispatch{kind,ordinal,layer,grid,tg} + build_decode_dag
#include "mtl4_context.hpp"
#include "safetensors_view.hpp"

namespace pie_metal_driver::raw_metal {

namespace {

SlotHandle stage_tensor(RawMetalContext& ctx, const RawTensor& rt) {
    SlotHandle s = ctx.heap_alloc(rt.nbytes);
    if (!s.valid()) throw std::runtime_error("heap_alloc failed (weights region exhausted)");
    std::memcpy(s.contents(), rt.data, rt.nbytes);
    return s;
}

SlotHandle alloc_zeroed(RawMetalContext& ctx, size_t nbytes) {
    SlotHandle s = ctx.heap_alloc(nbytes);
    if (!s.valid()) throw std::runtime_error("heap_alloc failed");
    std::memset(s.contents(), 0, nbytes);
    return s;
}

}  // namespace

// Stage all weights/state/KV/IO/scratch into the single resident heap. Allocation order
// follows the region plan (weights → kv → state → scratch → io) so alpha's bump allocator
// lands each slot in its planned region.
BoundDecode stage_decode_weights(RawMetalContext& ctx, const SafetensorsView& view,
                                 const DecodeGeometry& g, const HeapPlan& plan) {
    BoundDecode b;
    b.plan = plan;
    b.gdn.resize(g.n_layers);
    b.kv.resize(g.n_layers);

    // ── Weights region: stage every text-decode tensor (tied lm_head once) ──
    for (const auto& name : decode_weight_tensors(g)) {
        const RawTensor rt = view.get(name);   // throws if absent (probe-verified present)
        b.weights.emplace(name, stage_tensor(ctx, rt));
    }

    // ── KV region: k/v pages per full-attn layer (append-only, I4) ──
    const size_t kv_one = plan.kv_per_layer / 2;  // bytes for k (== v)
    for (int L = 0; L < g.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        b.kv[L].k_pages = alloc_zeroed(ctx, kv_one);
        b.kv[L].v_pages = alloc_zeroed(ctx, kv_one);
    }

    // ── State region: GDN conv (ping-pong) + recurrent (in-place) + zeroed conv-bias ──
    // S>1: conv/recurrent slabs hold g.max_slots slots packed at the natural per-slot stride
    // (beta's gdn_core_slotted indexes slot*(Kc*CDIM) / slot*(Hv*Vd*Dk)). conv_bias is a
    // shared zeroed slot (slot-independent). At max_slots=1 every alloc is byte-identical.
    const size_t slots = size_t(g.max_slots);
    const size_t conv_state = size_t(g.gdn_conv_dim) * g.gdn_conv_k * 4 * slots;       // f32
    const size_t recur_state = size_t(g.gdn_v_heads) * g.gdn_v_dim * g.gdn_k_dim * 4 * slots;
    const size_t conv_bias = size_t(g.gdn_conv_dim) * 2;                       // bf16, all-zero
    for (int L = 0; L < g.n_layers; ++L) {
        if (DecodeGeometry::is_full_attn(L)) continue;
        b.gdn[L].conv_state     = alloc_zeroed(ctx, conv_state);
        b.gdn[L].conv_state_out = alloc_zeroed(ctx, conv_state);
        b.gdn[L].recurrent_state = alloc_zeroed(ctx, recur_state);
        b.gdn[L].conv_bias_zero = alloc_zeroed(ctx, conv_bias);  // conv1d has no ckpt bias
    }

    // ── Scratch pool (beta assigns X/Out per dispatch) ──
    for (int i = 0; i < SCRATCH_POOL; ++i)
        b.scratch[i] = ctx.heap_alloc(plan.scratch_slot_bytes);

    // ── IO region (I1 per-token scalars + I3 logits) ──
    // M>1: scalar slots widen to u32[max_tokens]; logits stays f32[vocab]. Byte-identical at M=1.
    const size_t tok = 4 * size_t(g.max_tokens);
    b.io[static_cast<int>(IoSlot::TokenId)]   = alloc_zeroed(ctx, tok);
    b.io[static_cast<int>(IoSlot::Position)]  = alloc_zeroed(ctx, tok);
    b.io[static_cast<int>(IoSlot::SeqLen)]    = alloc_zeroed(ctx, tok);
    b.io[static_cast<int>(IoSlot::Logits)] = alloc_zeroed(
        ctx, g.paged_kv_enabled
                 ? size_t(g.vocab) * size_t(std::max(1, g.max_tokens)) * 2u
                 : size_t(g.vocab) * 4u);
    b.io[static_cast<int>(IoSlot::NextToken)] = alloc_zeroed(ctx, tok);

    if (g.paged_kv_enabled) {
        const size_t r = size_t(std::max(1, g.max_requests));
        const size_t n = size_t(std::max(1, g.max_tokens));
        const size_t refs = r * size_t(std::max(1, g.total_pages));
        b.io[static_cast<int>(IoSlot::QoIndptr)]       = alloc_zeroed(ctx, (r + 1) * 4u);
        b.io[static_cast<int>(IoSlot::KvPageIndptr)]   = alloc_zeroed(ctx, (r + 1) * 4u);
        b.io[static_cast<int>(IoSlot::KvPageIndices)]  = alloc_zeroed(ctx, refs * 4u);
        b.io[static_cast<int>(IoSlot::KvLastPageLens)] = alloc_zeroed(ctx, r * 4u);
        b.io[static_cast<int>(IoSlot::RsSlotIds)]      = alloc_zeroed(ctx, r * 4u);
        b.io[static_cast<int>(IoSlot::RsSlotFlags)]    = alloc_zeroed(ctx, r);
        b.io[static_cast<int>(IoSlot::ReqOfToken)]     = alloc_zeroed(ctx, n * 4u);
        b.io[static_cast<int>(IoSlot::SlotOfToken)]    = alloc_zeroed(ctx, n * 4u);
        b.io[static_cast<int>(IoSlot::WPage)]          = alloc_zeroed(ctx, n * 4u);
        b.io[static_cast<int>(IoSlot::WOff)]           = alloc_zeroed(ctx, n * 4u);
    }

    // device-argmax substrate (inert unless with_argmax): ArgmaxParams const + EosFlag out.
    b.argmax_params = alloc_zeroed(ctx, sizeof(ArgmaxParams));
    b.eos_flag      = alloc_zeroed(ctx, tok);
    {
        auto* p = static_cast<ArgmaxParams*>(b.argmax_params.contents());
        p->vocab = static_cast<uint32_t>(g.vocab);
        p->n_eos = 0;  // executor/resident loop rewrites vocab+eos per generation
    }
    return b;
}

namespace {
inline void bind_slot(RawMetalContext& ctx, int ord, uint8_t idx, const SlotHandle& s) {
    ctx.arg_bind_ordinal(ord, idx, s);
}
}  // namespace

// Walk beta's DAG; bind delta's weight/state/KV/IO slots for each dispatch by ordinal.
// (Robust to beta's ordering — reacts to each dispatch's kind+layer, not a fixed sequence.)
void bind_decode_dag(RawMetalContext& ctx, const BoundDecode& b,
                     const std::vector<Dispatch>& dag, const DecodeGeometry& g,
                     bool gdn_prep) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };

    for (const auto& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;

        // (a) load-once weights for this dispatch.
        for (const auto& wb : weight_binds(d.kind, L, g, gdn_prep)) {
            auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end())
                throw std::runtime_error("bind: unstaged weight " + wb.tensor);
            bind_slot(ctx, ord, wb.bind_index, it->second);
        }

        // (b) kind-specific state / KV / IO slots delta owns.
        switch (d.kind) {
            case Kernel::EmbedGather:
                bind_slot(ctx, ord, (uint8_t)bind::Embed::TokenId, io(IoSlot::TokenId));
                break;

            case Kernel::GdnPrep: {  // prep-dispatch (PIE_GDN_PREP): q/k path + q/k conv_state writeback
                const auto& s = b.gdn[L];
                bind_slot(ctx, ord, (uint8_t)bind::GdnPrep::ConvState,    s.conv_state);
                bind_slot(ctx, ord, (uint8_t)bind::GdnPrep::ConvStateOut, s.conv_state_out);
                bind_slot(ctx, ord, (uint8_t)bind::GdnPrep::ConvB,        s.conv_bias_zero);
                break;
            }

            case Kernel::GdnCore: {
                const auto& s = b.gdn[L];
                if (gdn_prep) {  // slimmed recurrent: ConvStateOut at 9, no ALog/DtBias/AGate/BGate
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCoreRecurrent::ConvState,      s.conv_state);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCoreRecurrent::RecurrentState, s.recurrent_state);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCoreRecurrent::ConvStateOut,   s.conv_state_out);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCoreRecurrent::ConvB,          s.conv_bias_zero);
                } else {
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCore::ConvState,    s.conv_state);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCore::RecurrentState, s.recurrent_state);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCore::ConvStateOut, s.conv_state_out);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCore::ConvB,        s.conv_bias_zero);
                }
                break;
            }

            case Kernel::KvAppend: {
                const auto& kv = b.kv[L];
                bind_slot(ctx, ord, (uint8_t)bind::KvAppend::KPages, kv.k_pages);
                bind_slot(ctx, ord, (uint8_t)bind::KvAppend::VPages, kv.v_pages);
                bind_slot(ctx, ord, (uint8_t)bind::KvAppend::PositionPtr, io(IoSlot::Position));
                break;
            }

            case Kernel::Sdpa: {
                const auto& kv = b.kv[L];
                bind_slot(ctx, ord, (uint8_t)bind::Sdpa::K, kv.k_pages);
                bind_slot(ctx, ord, (uint8_t)bind::Sdpa::V, kv.v_pages);
                bind_slot(ctx, ord, (uint8_t)bind::Sdpa::N, io(IoSlot::SeqLen));
                break;
            }

            case Kernel::Rope:
            case Kernel::RopeK:
                // rope.metal is in-place on buffer 0 (X); position is the IO scalar at
                // buffer 1. scale/base/head_dim are consts (decode_consts). The activation
                // (buffer 0 X) is bound by beta's scratch schedule (in-place).
                bind_slot(ctx, ord, (uint8_t)bind::Rope::Position, io(IoSlot::Position));
                break;

            case Kernel::QmvLmHead:
                // logits ALWAYS produced into the IO region (I3).
                bind_slot(ctx, ord, (uint8_t)bind::Qmv::Out, io(IoSlot::Logits));
                break;

            case Kernel::Argmax:
                bind_slot(ctx, ord, (uint8_t)bind::Argmax::Logits, io(IoSlot::Logits));
                bind_slot(ctx, ord, (uint8_t)bind::Argmax::NextToken, io(IoSlot::NextToken));
                bind_slot(ctx, ord, (uint8_t)bind::Argmax::Params, b.argmax_params);
                bind_slot(ctx, ord, (uint8_t)bind::Argmax::EosFlag, b.eos_flag);
                break;

            default:
                break;  // weight-only or scratch-only (beta) dispatches
        }
    }
}

}  // namespace pie_metal_driver::raw_metal

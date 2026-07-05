// RawMetalDecoder implementation — see decoder.hpp. The setup() body is decode_run.cpp's
// main() prologue; step() is its per-token decode-loop body; reset_state() zeroes the
// persistent GDN + KV state. Factored here so both the standalone decode_run bench and
// alpha's RawMetalExecutor (pie e2e) drive the identical pipeline.

#include "decoder.hpp"

#include <cstring>

#include "decode_consts.hpp"
#include "heap_bind.hpp"
#include "safetensors_view.hpp"

namespace pie_metal_driver::raw_metal {

namespace {

void write_u32(const SlotHandle& s, uint32_t v) {
    std::memcpy(s.contents(), &v, sizeof(v));
}

inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void zero_slot(const SlotHandle& s) {
    if (s.contents() && s.size) std::memset(s.contents(), 0, s.size);
}

// Zero one [off, off+len) byte window of a slot (a single slot's GDN-state slab region).
void zero_slot_region(const SlotHandle& s, size_t off, size_t len) {
    if (s.contents() && off + len <= s.size)
        std::memset(static_cast<char*>(s.contents()) + off, 0, len);
}

}  // namespace

bool RawMetalDecoder::setup(const std::string& ckpt_dir, const std::string& kernels_dir,
                            const DecodeGeometry& geom, std::string* err) {
    g_ = geom;

    // ── Open the checkpoint (zero-copy mmap) + size the heap from the manifest. The view is
    //    transient: stage_decode_weights memcpy's every weight into the resident heap, so the
    //    mmap is released at the end of setup(). ──
    SafetensorsView view(ckpt_dir);
    size_t weights_bytes = 0;
    for (const auto& name : decode_weight_tensors(g_))
        weights_bytes += view.get(name).nbytes;
    plan_ = plan_heap(g_, weights_bytes, max_ctx_);

    // ── Build the decode DAG (shipped config: GdnPrep ON, no argmax dispatch — host samples). ──
    dag_ = build_decode_dag(g_, /*with_argmax=*/false, fuse_residual_, gdn_prep_);

    // ── beta's scratch schedule (WAR/WAW coloring). e2e path always recycles. ──
    sched_ = build_scratch_schedule(dag_, g_, /*no_recycle=*/false);

    const size_t consts_budget = decode_consts_budget(dag_);
    const size_t heap_bytes = plan_.total + consts_budget
                            + size_t(sched_.colors_used) * plan_.scratch_slot_bytes + (32u << 20);

    ctx_ = RawMetalContext::create(heap_bytes);
    if (!ctx_) {
        if (err) *err = "RawMetalContext::create failed";
        return false;
    }

    // ── Stage weights/state/KV/IO; bind weight/state/KV/IO slots by ordinal. ──
    b_ = stage_decode_weights(*ctx_, view, g_, plan_);
    bind_decode_dag(*ctx_, b_, dag_, g_, gdn_prep_);

    // ── Scratch pool (colors_used slots) → beta's bind pass. ──
    pool_.resize(sched_.colors_used);
    for (int i = 0; i < sched_.colors_used; ++i)
        pool_[i] = ctx_->heap_alloc(plan_.scratch_slot_bytes);
    bind_scratch(*ctx_, dag_, sched_, pool_.data(), int(pool_.size()));

    // ── Geometry const-params. ──
    bind_decode_consts(*ctx_, dag_, g_, max_ctx_, gdn_prep_);

    // ── Compile the kernel PSOs. ──
    std::string load_err;
    if (!load_decode_psos(*ctx_, kernels_dir, psos_, /*with_argmax=*/false, &load_err,
                          fuse_residual_, gdn_prep_)) {
        if (err) *err = "PSO load failed: " + load_err;
        ctx_.reset();
        return false;
    }

    // ── Residency (I2): one set, after all binds. ──
    ctx_->make_resident();

    // ── Precompute the GDN dispatches whose conv-state binds ping-pong per step. ──
    gdn_disp_.clear();
    for (const auto& d : dag_)
        if (d.kind == Kernel::GdnCore || d.kind == Kernel::GdnPrep)
            gdn_disp_.push_back({d.ordinal, d.layer, d.kind});

    step_count_ = 0;
    return true;
}

void RawMetalDecoder::reset_state() {
    for (auto& gs : b_.gdn) {
        zero_slot(gs.conv_state);
        zero_slot(gs.conv_state_out);
        zero_slot(gs.recurrent_state);
    }
    for (auto& ks : b_.kv) {
        zero_slot(ks.k_pages);
        zero_slot(ks.v_pages);
    }
    step_count_ = 0;
}

// Zero only `slot`'s GDN conv/recurrent region within each layer's slab (the per-slot stride
// laid out by build_bound_decode: conv = gdn_conv_dim*gdn_conv_k, recurrent =
// gdn_v_heads*gdn_v_dim*gdn_k_dim, f32). KV is paged per-request → reset via the runtime's
// page table (kv_last_page_lens=0 for a NEW request), not by zeroing the shared pool here.
// At max_slots=1, slot=0, off=0 → equivalent to the GDN half of reset_state().
void RawMetalDecoder::reset_state(uint32_t slot) {
    const size_t conv_stride  = size_t(g_.gdn_conv_dim) * g_.gdn_conv_k * 4;
    const size_t recur_stride = size_t(g_.gdn_v_heads) * g_.gdn_v_dim * g_.gdn_k_dim * 4;
    const size_t conv_off  = size_t(slot) * conv_stride;
    const size_t recur_off = size_t(slot) * recur_stride;
    for (auto& gs : b_.gdn) {
        zero_slot_region(gs.conv_state, conv_off, conv_stride);
        zero_slot_region(gs.conv_state_out, conv_off, conv_stride);
        zero_slot_region(gs.recurrent_state, recur_off, recur_stride);
    }
}

StepTiming RawMetalDecoder::step(uint32_t token_id, uint32_t position) {
    write_u32(b_.io[int(IoSlot::TokenId)],  token_id);
    write_u32(b_.io[int(IoSlot::Position)], position);
    write_u32(b_.io[int(IoSlot::SeqLen)],   position + 1u);

    // GDN conv-state cross-step ping-pong: ConvState (RO) and ConvStateOut are DISTINCT
    // buffers, advanced token-to-token by swapping their bind each step (step i reads what
    // i-1 wrote). Parity follows the monotonic per-sequence step index so prefill→decode is
    // seamless across step() calls (NOT the absolute position, which can start non-zero).
    const bool even = (step_count_ % 2 == 0);
    for (const auto& gd : gdn_disp_) {
        const SlotHandle& A = b_.gdn[gd.layer].conv_state;
        const SlotHandle& C = b_.gdn[gd.layer].conv_state_out;
        uint8_t cs_bind, cso_bind;
        if (gd.kind == Kernel::GdnPrep) {                // prep writes q/k conv_state channels
            cs_bind  = (uint8_t)bind::GdnPrep::ConvState;
            cso_bind = (uint8_t)bind::GdnPrep::ConvStateOut;
        } else if (gdn_prep_) {                           // recurrent writes v conv_state channels
            cs_bind  = (uint8_t)bind::GdnCoreRecurrent::ConvState;
            cso_bind = (uint8_t)bind::GdnCoreRecurrent::ConvStateOut;
        } else {                                          // in-kernel-share GdnCore
            cs_bind  = (uint8_t)bind::GdnCore::ConvState;
            cso_bind = (uint8_t)bind::GdnCore::ConvStateOut;
        }
        ctx_->arg_bind_ordinal(gd.ord, cs_bind,  even ? A : C);
        ctx_->arg_bind_ordinal(gd.ord, cso_bind, even ? C : A);
    }

    StepTiming t = ctx_->run_step(
        [&](StepEncoder& se) { encode_decode_step(se, dag_, psos_, force_barriers_); },
        step_count_ & 1);
    ++step_count_;
    return t;
}

const uint16_t* RawMetalDecoder::logits_bf16() const {
    return static_cast<const uint16_t*>(b_.io[int(IoSlot::Logits)].contents());
}

void RawMetalDecoder::copy_logits_f32(float* out) const {
    const uint16_t* lb = logits_bf16();
    for (int i = 0; i < g_.vocab; ++i) out[i] = bf16_to_f32(lb[i]);
}

uint32_t RawMetalDecoder::argmax() const {
    const uint16_t* lb = logits_bf16();   // lm_head writes bf16, not f32
    uint32_t best = 0;
    float bv = bf16_to_f32(lb[0]);
    for (int i = 1; i < g_.vocab; ++i) {
        float v = bf16_to_f32(lb[i]);
        if (v > bv) { bv = v; best = uint32_t(i); }
    }
    return best;
}

}  // namespace pie_metal_driver::raw_metal

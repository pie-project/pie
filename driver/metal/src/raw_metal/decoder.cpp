// RawMetalDecoder implementation — see decoder.hpp. The setup() body is decode_run.cpp's
// main() prologue; step() is its per-token decode-loop body; reset_state() zeroes the
// persistent GDN + KV state. Factored here so standalone tools and the direct
// entry path drive the identical pipeline.

#include "decoder.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

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

// Copy one [off, off+len) byte window from `src`'s contents to the SAME window in a
// DIFFERENT SlotHandle `dst` (used when growing the KV pool into a new, bigger standalone
// buffer — old and new pages share the SAME byte offset, just a different backing buffer).
bool copy_between_slots(const SlotHandle& dst, const SlotHandle& src, size_t off, size_t len) {
    if (!dst.contents() || !src.contents()) return false;
    if (off + len > dst.size || off + len > src.size) return false;
    std::memcpy(static_cast<char*>(dst.contents()) + off,
               static_cast<const char*>(src.contents()) + off, len);
    return true;
}

// Copy one [src_off, src_off+len) byte window from `s` to [dst_off, dst_off+len) of the
// SAME SlotHandle `s` (a single slot's GDN-state slab region — different slot OFFSETS
// within one shared per-layer buffer). Returns false if either window is out of range (a
// no-op, never a partial/garbage copy).
bool copy_slot_region(const SlotHandle& s, size_t src_off, size_t dst_off, size_t len) {
    if (!s.contents() || src_off + len > s.size || dst_off + len > s.size) return false;
    if (src_off == dst_off || len == 0) return true;  // no-op
    std::memcpy(static_cast<char*>(s.contents()) + dst_off,
               static_cast<const char*>(s.contents()) + src_off, len);
    return true;
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
    if (g_.paged_kv_enabled) {
        mb_dag_ = build_decode_dag_mb(g_, std::max(1, g_.max_tokens),
                                      kMultiBatchOrdinalBase, fuse_residual_, gdn_prep_);
        mb_sched_ = build_scratch_schedule(mb_dag_, g_, /*no_recycle=*/false);
        prefill_dags_ = build_decode_prefill_dags(g_, std::max(1, g_.max_tokens),
                                                   fuse_residual_, gdn_prep_);
        prefill_sched_ = build_scratch_schedule(prefill_dags_.front(), g_, /*no_recycle=*/false);
    }

    // ── beta's scratch schedule (WAR/WAW coloring). e2e path always recycles. ──
    sched_ = build_scratch_schedule(dag_, g_, /*no_recycle=*/false);

    size_t prefill_consts_budget = 0;
    for (const auto& dag : prefill_dags_) prefill_consts_budget += decode_consts_budget(dag);
    const size_t consts_budget = decode_consts_budget(dag_) +
                                 (mb_dag_.empty() ? 0 : decode_consts_budget(mb_dag_)) +
                                 prefill_consts_budget;
    const size_t heap_bytes = plan_.total + consts_budget
                            + size_t(std::max({sched_.colors_used, mb_sched_.colors_used,
                                               prefill_sched_.colors_used})) *
                                  plan_.scratch_slot_bytes + (32u << 20);

    ctx_ = RawMetalContext::create(heap_bytes);
    if (!ctx_) {
        if (err) *err = "RawMetalContext::create failed";
        return false;
    }

    // ── Stage weights/state/KV/IO; bind weight/state/KV/IO slots by ordinal. ──
    b_ = stage_decode_weights(*ctx_, view, g_, plan_);
    bind_decode_dag(*ctx_, b_, dag_, g_, gdn_prep_);

    // ── Scratch pool (colors_used slots) → beta's bind pass. ──
    pool_.resize(std::max({sched_.colors_used, mb_sched_.colors_used,
                           prefill_sched_.colors_used}));
    for (size_t i = 0; i < pool_.size(); ++i)
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
    if (g_.paged_kv_enabled &&
        !load_multibatch_psos(*ctx_, kernels_dir, mb_psos_, /*with_d512=*/false, &load_err)) {
        if (err) *err = "multi-batch PSO load failed: " + load_err;
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

    // Phase 1b: one independent ping-pong step counter per resident-state slot.
    step_count_by_slot_.assign(std::max<size_t>(1, size_t(g_.max_slots)), 0);
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
    for (auto& c : step_count_by_slot_) c = 0;
}

// Zero only `slot`'s GDN conv/recurrent region within each layer's slab (the per-slot stride
// laid out by build_bound_decode: conv = gdn_conv_dim*gdn_conv_k, recurrent =
// gdn_v_heads*gdn_v_dim*gdn_k_dim, f32). KV is paged per-request → reset via the runtime's
// page table (kv_last_page_lens=0 for a NEW request), not by zeroing the shared pool here.
// At max_slots=1, slot=0, off=0 → equivalent to the GDN half of reset_state(). ALSO resets
// this slot's own ping-pong step parity (Phase 1b state-slot fix) so a fresh sequence on
// `slot` always starts at parity 0, independent of any other slot's step history.
void RawMetalDecoder::reset_state(uint32_t slot) {
    const size_t conv_stride  = g_.gdn_conv_stride_bytes();
    const size_t recur_stride = g_.gdn_recurrent_stride_bytes();
    const size_t conv_off  = size_t(slot) * conv_stride;
    const size_t recur_off = size_t(slot) * recur_stride;
    for (auto& gs : b_.gdn) {
        zero_slot_region(gs.conv_state, conv_off, conv_stride);
        zero_slot_region(gs.conv_state_out, conv_off, conv_stride);
        zero_slot_region(gs.recurrent_state, recur_off, recur_stride);
    }
    if (slot < step_count_by_slot_.size()) step_count_by_slot_[slot] = 0;
}

// Phase 1b: real, bounds-checked whole-slot copy of every GDN layer's
// resident conv+recurrent state — the SAME per-slot stride formula
// reset_state(slot) already uses (build_bound_decode/plan_heap: conv =
// gdn_conv_dim*gdn_conv_k, recurrent = gdn_v_heads*gdn_v_dim*gdn_k_dim, f32,
// `g_.max_slots` slots packed per layer). Only GDN layers have a real
// (non-zero-sized) `b_.gdn[L]` slab — full-attn layer entries are default-
// constructed (size 0) and `copy_slot_region` safely no-ops on them, so this
// loop does not need an `is_full_attn` filter (mirrors reset_state(slot)'s
// own style). `src_slot`/`dst_slot` are bounds-checked against `g_.max_slots`
// up front — never a partial/silent-garbage copy.
bool RawMetalDecoder::copy_state_slot(uint32_t src_slot, uint32_t dst_slot, std::string* err) {
    if (!ready()) {
        if (err) *err = "RawMetalDecoder::copy_state_slot: decoder not initialized";
        return false;
    }
    if (src_slot >= uint32_t(g_.max_slots) || dst_slot >= uint32_t(g_.max_slots)) {
        if (err) {
            *err = "RawMetalDecoder::copy_state_slot: slot id out of range [0, " +
                   std::to_string(g_.max_slots) + ")";
        }
        return false;
    }
    const size_t conv_stride  = g_.gdn_conv_stride_bytes();
    const size_t recur_stride = g_.gdn_recurrent_stride_bytes();
    const size_t src_conv_off  = size_t(src_slot) * conv_stride;
    const size_t dst_conv_off  = size_t(dst_slot) * conv_stride;
    const size_t src_recur_off = size_t(src_slot) * recur_stride;
    const size_t dst_recur_off = size_t(dst_slot) * recur_stride;
    int gdn_layers_copied = 0;
    for (auto& gs : b_.gdn) {
        if (!gs.conv_state.valid()) continue;  // a full-attn layer's unused slot
        const bool ok = copy_slot_region(gs.conv_state, src_conv_off, dst_conv_off, conv_stride) &&
                        copy_slot_region(gs.conv_state_out, src_conv_off, dst_conv_off, conv_stride) &&
                        copy_slot_region(gs.recurrent_state, src_recur_off, dst_recur_off, recur_stride);
        if (!ok) {
            if (err) *err = "RawMetalDecoder::copy_state_slot: internal bounds check failed";
            return false;
        }
        ++gdn_layers_copied;
    }
    if (gdn_layers_copied == 0) {
        if (err) *err = "RawMetalDecoder::copy_state_slot: this checkpoint has no GDN layers (nothing to copy)";
        return false;
    }
    // The ping-pong PARITY (which of ConvState/ConvStateOut currently holds
    // the LATEST data) is a function of how many steps a slot has taken
    // (`step_count_by_slot_[slot] % 2` — see step()). Since both ping-pong
    // buffers were just copied VERBATIM (A stays A, C stays C, never
    // swapped), `dst_slot` must inherit `src_slot`'s exact step count too —
    // otherwise a later step() on `dst_slot` could read the STALE half
    // instead of the one that actually holds the copied-in latest data
    // (silently correct only when src/dst happened to share the same
    // parity by coincidence).
    if (src_slot < step_count_by_slot_.size() && dst_slot < step_count_by_slot_.size()) {
        step_count_by_slot_[dst_slot] = step_count_by_slot_[src_slot];
    }
    return true;
}

uint64_t RawMetalDecoder::rs_slot_bytes() const {
    if (!ready()) return 0;
    const size_t conv_stride  = g_.gdn_conv_stride_bytes();
    const size_t recur_stride = g_.gdn_recurrent_stride_bytes();
    uint64_t total = 0;
    for (const auto& gs : b_.gdn) {
        if (!gs.conv_state.valid()) continue;
        total += 2 * conv_stride + recur_stride;  // ConvState + ConvStateOut + RecurrentState
    }
    return total;
}

namespace {
// One NHD paged-pool row's byte size: [n_kv_heads, head_dim], bf16 (matches the M=1
// ring's activation dtype — kv_append.metal/kv_append_paged.metal both instantiate bf16).
size_t kv_pool_row_bytes(const DecodeGeometry& g) {
    return size_t(g.n_kv_heads) * size_t(g.head_dim) * 2u;
}
}  // namespace

bool RawMetalDecoder::setup_kv_pool(uint32_t total_pages, uint32_t page_size, std::string* err) {
    if (!ready()) {
        if (err) *err = "RawMetalDecoder::setup_kv_pool: decoder not initialized";
        return false;
    }
    if (total_pages == 0 || page_size == 0) {
        if (err) *err = "RawMetalDecoder::setup_kv_pool: total_pages and page_size must be > 0";
        return false;
    }
    int n_full = 0;
    for (int L = 0; L < g_.n_layers; ++L) {
        if (DecodeGeometry::is_full_attn(L)) ++n_full;
    }
    if (n_full == 0) {
        if (err) {
            *err = "RawMetalDecoder::setup_kv_pool: this checkpoint has no full-attention "
                   "layers (nothing to allocate a KV page pool for)";
        }
        return false;
    }
    const size_t layer_bytes =
        size_t(total_pages) * size_t(page_size) * kv_pool_row_bytes(g_);
    KvPagePool pool;
    pool.layers.resize(size_t(g_.n_layers));
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        pool.layers[size_t(L)].k_pages = ctx_->create_standalone_buffer(layer_bytes);
        pool.layers[size_t(L)].v_pages = ctx_->create_standalone_buffer(layer_bytes);
        if (!pool.layers[size_t(L)].k_pages.valid() || !pool.layers[size_t(L)].v_pages.valid()) {
            if (err) {
                *err = "RawMetalDecoder::setup_kv_pool: standalone buffer allocation failed "
                       "(layer " + std::to_string(L) + ", " + std::to_string(layer_bytes) +
                       " bytes/buffer)";
            }
            return false;
        }
    }
    pool.total_pages = total_pages;
    pool.page_size = page_size;
    pool.enabled = true;
    // Phase 3 (review item 4): if a pool was already allocated (setup_kv_pool
    // re-called with a different size), release the OLD standalone buffers
    // before replacing it, so re-setup does not leak the previous allocation.
    KvPagePool old_pool = std::move(kv_pool_);
    kv_pool_ = std::move(pool);
    for (auto& lp : old_pool.layers) {
        if (lp.k_pages.valid()) ctx_->release_standalone_buffer(lp.k_pages);
        if (lp.v_pages.valid()) ctx_->release_standalone_buffer(lp.v_pages);
    }
    return bind_paged_dag(err);
}

bool RawMetalDecoder::bind_paged_dag(std::string* err) {
    if (!ready() || !g_.paged_kv_enabled || !kv_pool_.enabled || mb_dag_.empty()) {
        if (err) *err = "RawMetalDecoder::bind_paged_dag: paged decode is not initialized";
        return false;
    }
    try {
        std::vector<SlotHandle> k_pages(size_t(g_.n_layers));
        std::vector<SlotHandle> v_pages(size_t(g_.n_layers));
        for (int L = 0; L < g_.n_layers; ++L) {
            if (!DecodeGeometry::is_full_attn(L)) continue;
            k_pages[size_t(L)] = kv_pool_.layers[size_t(L)].k_pages;
            v_pages[size_t(L)] = kv_pool_.layers[size_t(L)].v_pages;
        }
        bind_decode_dag_mb(*ctx_, b_, mb_dag_, g_, k_pages, v_pages, gdn_prep_);
        const size_t prefill_scratch_row = size_t(scratch_widest_elems(g_)) * 2u;
        const size_t prefill_logits_row = size_t(g_.vocab) * 2u;
        for (size_t t = 0; t < prefill_dags_.size(); ++t) {
            const MbBindOffsets offsets{
                .token_row = t,
                .logits_bytes = t * prefill_logits_row,
            };
            bind_decode_dag_mb(*ctx_, b_, prefill_dags_[t], g_, k_pages, v_pages, gdn_prep_,
                               offsets);
        }
        if (!mb_bound_) {
            bind_scratch(*ctx_, mb_dag_, mb_sched_, pool_.data(), int(pool_.size()));
            bind_decode_consts(*ctx_, mb_dag_, g_, max_ctx_, gdn_prep_);
            for (size_t t = 0; t < prefill_dags_.size(); ++t) {
                bind_scratch(*ctx_, prefill_dags_[t], prefill_sched_, pool_.data(),
                             int(pool_.size()), t * prefill_scratch_row);
                bind_decode_consts(*ctx_, prefill_dags_[t], g_, max_ctx_, gdn_prep_);
            }
            mb_bound_ = true;
        }
        ++paged_bind_generation_;
        return true;
    } catch (const std::exception& e) {
        if (err) *err = std::string("RawMetalDecoder::bind_paged_dag: ") + e.what();
        return false;
    }
}

bool RawMetalDecoder::copy_kv_pages(const std::vector<uint32_t>& src_pages,
                                    const std::vector<uint32_t>& dst_pages, std::string* err) {
    if (!ready() || !kv_pool_.enabled) {
        if (err) *err = "RawMetalDecoder::copy_kv_pages: KV page pool not allocated";
        return false;
    }
    if (src_pages.size() != dst_pages.size()) {
        if (err) *err = "RawMetalDecoder::copy_kv_pages: src/dst page count mismatch";
        return false;
    }
    // Bounds-check EVERY page first — never a partial copy on a late failure.
    for (size_t i = 0; i < src_pages.size(); ++i) {
        if (src_pages[i] >= kv_pool_.total_pages || dst_pages[i] >= kv_pool_.total_pages) {
            if (err) {
                *err = "RawMetalDecoder::copy_kv_pages: page id out of range [0, " +
                       std::to_string(kv_pool_.total_pages) + ")";
            }
            return false;
        }
    }
    const size_t page_bytes = size_t(kv_pool_.page_size) * kv_pool_row_bytes(g_);
    // NOTE: copies within one call are applied in the given order; a chain like
    // {1->0, 2->1} reads page 1 for the second copy AFTER the first already
    // overwrote it. Matches the typical device-copy convention (each pair is
    // independent; the caller sequences non-conflicting moves, or issues them as
    // separate calls when a true swap/rotate is needed).
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        const auto& lp = kv_pool_.layers[size_t(L)];
        for (size_t i = 0; i < src_pages.size(); ++i) {
            const size_t src_off = size_t(src_pages[i]) * page_bytes;
            const size_t dst_off = size_t(dst_pages[i]) * page_bytes;
            if (!copy_slot_region(lp.k_pages, src_off, dst_off, page_bytes) ||
                !copy_slot_region(lp.v_pages, src_off, dst_off, page_bytes)) {
                if (err) *err = "RawMetalDecoder::copy_kv_pages: internal bounds check failed";
                return false;
            }
        }
    }
    return true;
}

bool RawMetalDecoder::copy_kv_cells(const std::vector<KvMoveCell>& cells, std::string* err) {
    if (!ready() || !kv_pool_.enabled) {
        if (err) *err = "RawMetalDecoder::copy_kv_cells: KV page pool not allocated";
        return false;
    }
    for (const auto& c : cells) {
        if (c.src_page_id >= kv_pool_.total_pages || c.dst_page_id >= kv_pool_.total_pages ||
            c.src_token_offset >= kv_pool_.page_size || c.dst_token_offset >= kv_pool_.page_size) {
            if (err) {
                *err = "RawMetalDecoder::copy_kv_cells: cell page id/token offset out of "
                       "range (total_pages=" + std::to_string(kv_pool_.total_pages) +
                       ", page_size=" + std::to_string(kv_pool_.page_size) + ")";
            }
            return false;
        }
    }
    const size_t row_bytes = kv_pool_row_bytes(g_);
    const size_t page_bytes = size_t(kv_pool_.page_size) * row_bytes;
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        const auto& lp = kv_pool_.layers[size_t(L)];
        for (const auto& c : cells) {
            const size_t src_off = size_t(c.src_page_id) * page_bytes +
                                   size_t(c.src_token_offset) * row_bytes;
            const size_t dst_off = size_t(c.dst_page_id) * page_bytes +
                                   size_t(c.dst_token_offset) * row_bytes;
            if (!copy_slot_region(lp.k_pages, src_off, dst_off, row_bytes) ||
                !copy_slot_region(lp.v_pages, src_off, dst_off, row_bytes)) {
                if (err) *err = "RawMetalDecoder::copy_kv_cells: internal bounds check failed";
                return false;
            }
        }
    }
    return true;
}

bool RawMetalDecoder::resize_kv_pool(uint32_t new_total_pages, bool unmapped_tail_pages,
                                     std::string* err) {
    if (!ready()) {
        if (err) *err = "RawMetalDecoder::resize_kv_pool: decoder not initialized";
        return false;
    }
    if (!kv_pool_.enabled) {
        if (err) {
            *err = "RawMetalDecoder::resize_kv_pool: KV page pool not allocated "
                   "(call setup_kv_pool first)";
        }
        return false;
    }
    if (new_total_pages == kv_pool_.total_pages) return true;  // no-op success
    if (new_total_pages == 0) {
        if (err) *err = "RawMetalDecoder::resize_kv_pool: resize to 0 pages is not supported";
        return false;
    }
    if (new_total_pages < kv_pool_.total_pages && !unmapped_tail_pages) {
        if (err) {
            *err = "RawMetalDecoder::resize_kv_pool: shrink would truncate pages [" +
                   std::to_string(new_total_pages) + ", " + std::to_string(kv_pool_.total_pages) +
                   ") that the caller has not attested are unmapped/free — refusing to "
                   "silently discard potentially-live pages";
        }
        return false;
    }
    const size_t row_bytes = kv_pool_row_bytes(g_);
    const size_t new_layer_bytes =
        size_t(new_total_pages) * size_t(kv_pool_.page_size) * row_bytes;
    const size_t copy_pages = std::min<uint32_t>(new_total_pages, kv_pool_.total_pages);
    const size_t copy_bytes = size_t(copy_pages) * size_t(kv_pool_.page_size) * row_bytes;
    KvPagePool new_pool;
    new_pool.layers.resize(size_t(g_.n_layers));
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        SlotHandle new_k = ctx_->create_standalone_buffer(new_layer_bytes);
        SlotHandle new_v = ctx_->create_standalone_buffer(new_layer_bytes);
        if (!new_k.valid() || !new_v.valid()) {
            if (err) *err = "RawMetalDecoder::resize_kv_pool: new buffer allocation failed";
            return false;
        }
        if (copy_bytes > 0) {
            const auto& old_lp = kv_pool_.layers[size_t(L)];
            if (!copy_between_slots(new_k, old_lp.k_pages, 0, copy_bytes) ||
                !copy_between_slots(new_v, old_lp.v_pages, 0, copy_bytes)) {
                if (err) *err = "RawMetalDecoder::resize_kv_pool: page-preserving copy failed";
                return false;
            }
        }
        new_pool.layers[size_t(L)].k_pages = new_k;
        new_pool.layers[size_t(L)].v_pages = new_v;
    }
    new_pool.total_pages = new_total_pages;
    new_pool.page_size = kv_pool_.page_size;
    new_pool.enabled = true;
    // Phase 3 (review item 4): install the new pool, then RELEASE the old
    // standalone buffers (drop from residency + retained-alive so ARC frees
    // them). The synchronous copy_between_slots above has already read every
    // preserved page out of the old buffers, so nothing still references them.
    // Without this, repeated grow/shrink would leak the old K/V allocations
    // unbounded (they'd stay retained + resident forever).
    KvPagePool old_pool = std::move(kv_pool_);
    kv_pool_ = std::move(new_pool);
    for (auto& lp : old_pool.layers) {
        if (lp.k_pages.valid()) ctx_->release_standalone_buffer(lp.k_pages);
        if (lp.v_pages.valid()) ctx_->release_standalone_buffer(lp.v_pages);
    }
    return bind_paged_dag(err);
}

StepTiming RawMetalDecoder::step(uint32_t token_id, uint32_t position, uint32_t slot) {
    write_u32(b_.io[int(IoSlot::TokenId)],  token_id);
    write_u32(b_.io[int(IoSlot::Position)], position);
    write_u32(b_.io[int(IoSlot::SeqLen)],   position + 1u);

    int& sc = step_count_for(slot);

    // GDN conv-state cross-step ping-pong: ConvState (RO) and ConvStateOut are DISTINCT
    // buffers, advanced token-to-token by swapping their bind each step (step i reads what
    // i-1 wrote). Parity follows `slot`'s OWN monotonic step index (Phase 1b state-slot fix:
    // each slot tracks its own parity independently, so switching between slots between
    // forward calls resumes each slot's ping-pong correctly) — NOT the absolute position
    // (which can start non-zero) and NOT a single decoder-wide counter (which would
    // silently corrupt a slot's parity whenever a DIFFERENT slot had stepped in between).
    //
    // Slot selection (Phase 1b state-slot fix): the M=1 kernels (gdn_prep_bfloat16 /
    // gdn_core_recurrent_bfloat16, the shipped config) have NO slot_ids input — they always
    // operate at byte offset 0 of whatever buffer they're bound to. `arg_bind_ordinal`'s
    // offset form (`setAddress:(slot.gpu_address + offset)`) lets us slide the GPU address
    // the kernel sees by `slot*stride`, so binding "the same conv/recurrent slab, offset by
    // slot*stride" transparently retargets the UNCHANGED kernel at slot's own byte range —
    // no shader change, no new PSO. RecurrentState (unlike ConvState/ConvStateOut) is bound
    // ONCE at setup() and never touched again by the OLD code — it must ALSO be rebound here
    // every step, or every slot would silently alias slot 0's recurrent state forever.
    const bool even = (sc % 2 == 0);
    const size_t conv_stride  = g_.gdn_conv_stride_bytes();
    const size_t recur_stride = g_.gdn_recurrent_stride_bytes();
    const size_t conv_off  = size_t(slot) * conv_stride;
    const size_t recur_off = size_t(slot) * recur_stride;
    for (const auto& gd : gdn_disp_) {
        const SlotHandle& A = b_.gdn[gd.layer].conv_state;
        const SlotHandle& C = b_.gdn[gd.layer].conv_state_out;
        const SlotHandle& R = b_.gdn[gd.layer].recurrent_state;
        uint8_t cs_bind, cso_bind;
        int rs_bind = -1;  // -1: this dispatch kind has no RecurrentState bind (GdnPrep)
        if (gd.kind == Kernel::GdnPrep) {                // prep writes q/k conv_state channels
            cs_bind  = (uint8_t)bind::GdnPrep::ConvState;
            cso_bind = (uint8_t)bind::GdnPrep::ConvStateOut;
        } else if (gdn_prep_) {                           // recurrent writes v conv_state channels
            cs_bind  = (uint8_t)bind::GdnCoreRecurrent::ConvState;
            cso_bind = (uint8_t)bind::GdnCoreRecurrent::ConvStateOut;
            rs_bind  = (uint8_t)bind::GdnCoreRecurrent::RecurrentState;
        } else {                                          // in-kernel-share GdnCore
            cs_bind  = (uint8_t)bind::GdnCore::ConvState;
            cso_bind = (uint8_t)bind::GdnCore::ConvStateOut;
            rs_bind  = (uint8_t)bind::GdnCore::RecurrentState;
        }
        ctx_->arg_bind_ordinal(gd.ord, cs_bind,  even ? A : C, conv_off);
        ctx_->arg_bind_ordinal(gd.ord, cso_bind, even ? C : A, conv_off);
        if (rs_bind >= 0) ctx_->arg_bind_ordinal(gd.ord, uint8_t(rs_bind), R, recur_off);
    }

    StepTiming t = ctx_->run_step(
        [&](StepEncoder& se) { encode_decode_step(se, dag_, psos_, force_barriers_); },
        sc & 1);
    ++sc;
    return t;
}

bool RawMetalDecoder::run_batch_step(const BatchSchedule& schedule, const BatchStepInputs& in,
                                     std::string* err) {
    auto fail = [&](const std::string& why) {
        if (err) *err = "RawMetalDecoder::run_batch_step: " + why;
        return false;
    };
    if (!ready() || !g_.paged_kv_enabled || !kv_pool_.enabled || !mb_bound_)
        return fail("paged decode DAG/pool is not initialized");
    if (schedule.N <= 0 || schedule.R <= 0)
        return fail("paged batch has no tokens or requests");
    std::string capacity_err;
    if (!validate_paged_batch_capacity(schedule, uint32_t(g_.max_tokens),
                                       uint32_t(g_.max_requests), &capacity_err))
        return fail(capacity_err);
    if (in.token_ids.size() != size_t(schedule.N) || in.position_ids.size() != size_t(schedule.N) ||
        in.qo_indptr.size() != size_t(schedule.R + 1) ||
        in.kv_page_indptr.size() != size_t(schedule.R + 1) ||
        in.kv_last_page_lens.size() != size_t(schedule.R) ||
        in.rs_slot_ids.size() != size_t(schedule.R) ||
        in.rs_slot_flags.size() != size_t(schedule.R) ||
        in.w_page.size() != size_t(schedule.N) || in.w_off.size() != size_t(schedule.N))
        return fail("inconsistent fixed IO vector sizes");
    if (in.kv_page_indices.size() > size_t(g_.max_requests) * size_t(g_.total_pages))
        return fail("flattened KV CSR exceeds configured reference capacity");
    std::string geometry_err;
    if (!validate_paged_batch(schedule, in.position_ids, in.kv_page_indices, in.w_page, in.w_off,
                              kv_pool_.total_pages, uint32_t(g_.max_slots), &geometry_err))
        return fail(geometry_err);
    for (int r = 0; r < schedule.R; ++r) {
        const RequestSpan& sp = schedule.spans[size_t(r)];
        if (sp.rs_slot >= uint32_t(g_.max_slots))
            return fail("recurrent-state slot is out of range");
        if (sp.num_pages == 0 || sp.pages_first + sp.num_pages > in.kv_page_indices.size())
            return fail("request has an invalid KV page span");
        if (sp.seqlen == 0 || sp.qo_lo >= uint32_t(schedule.N) ||
            in.position_ids[sp.qo_lo] >= sp.seqlen)
            return fail("position is outside its request KV extent");
    }
    for (int t = 0; t < schedule.N; ++t) {
        const uint32_t r = schedule.req_of_token[size_t(t)];
        if (r >= uint32_t(schedule.R) || in.w_page[size_t(t)] >= kv_pool_.total_pages ||
            in.w_off[size_t(t)] >= kv_pool_.page_size)
            return fail("write page/offset is out of range");
        const RequestSpan& sp = schedule.spans[r];
        const uint32_t pos = in.position_ids[size_t(t)];
        const uint32_t page_at_pos =
            in.kv_page_indices[sp.pages_first + pos / kv_pool_.page_size];
        if (in.w_page[size_t(t)] != page_at_pos || in.w_off[size_t(t)] != pos % kv_pool_.page_size)
            return fail("write descriptor does not match the request CSR position");
    }

    auto copy_to = [&](IoSlot slot, const auto& values) {
        std::memcpy(b_.io[static_cast<int>(slot)].contents(), values.data(),
                    values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type));
    };
    copy_to(IoSlot::TokenId, in.token_ids);
    copy_to(IoSlot::Position, in.position_ids);
    copy_to(IoSlot::QoIndptr, in.qo_indptr);
    copy_to(IoSlot::KvPageIndptr, in.kv_page_indptr);
    copy_to(IoSlot::KvPageIndices, in.kv_page_indices);
    copy_to(IoSlot::KvLastPageLens, in.kv_last_page_lens);
    copy_to(IoSlot::RsSlotIds, in.rs_slot_ids);
    copy_to(IoSlot::RsSlotFlags, in.rs_slot_flags);
    copy_to(IoSlot::ReqOfToken, schedule.req_of_token);
    copy_to(IoSlot::SlotOfToken, schedule.slot_of_token);
    copy_to(IoSlot::WPage, in.w_page);
    copy_to(IoSlot::WOff, in.w_off);
    std::vector<uint32_t> seq_len(size_t(schedule.N));
    for (int t = 0; t < schedule.N; ++t)
        seq_len[size_t(t)] = schedule.spans[schedule.req_of_token[size_t(t)]].seqlen;
    copy_to(IoSlot::SeqLen, seq_len);

    if (!schedule.is_pure_decode) return run_prefill_step(schedule, err);

    std::vector<uint32_t> active_slots;
    active_slots.reserve(size_t(schedule.R));
    for (const RequestSpan& sp : schedule.spans) {
        if (sp.rs_is_new) reset_state(sp.rs_slot);
        if (std::find(active_slots.begin(), active_slots.end(), sp.rs_slot) == active_slots.end())
            active_slots.push_back(sp.rs_slot);
    }
    // The paged kernels always read ConvState and write ConvStateOut.  Normalize
    // slots last touched by the M=1 ping-pong path before dispatch, then fold the
    // completed result back into ConvState for the next paged fire.
    for (uint32_t slot : active_slots) {
        if ((step_count_for(slot) & 1) == 0) continue;
        const size_t off = size_t(slot) * g_.gdn_conv_stride_bytes();
        // copy C -> A (different handles, same offset).
        for (auto& gs : b_.gdn) {
            if (!gs.conv_state.valid() ||
                !copy_between_slots(gs.conv_state, gs.conv_state_out, off,
                                    g_.gdn_conv_stride_bytes()))
                return fail("failed to normalize GDN ping-pong state");
        }
    }

    const std::vector<Dispatch> fire_dag =
        build_decode_dag_mb(g_, schedule.N, kMultiBatchOrdinalBase, fuse_residual_, gdn_prep_);
    ctx_->run_step([&](StepEncoder& se) {
        encode_decode_step_mb(se, fire_dag, psos_, mb_psos_, force_barriers_);
    });

    for (uint32_t slot : active_slots) {
        const size_t off = size_t(slot) * g_.gdn_conv_stride_bytes();
        for (auto& gs : b_.gdn) {
            if (!gs.conv_state.valid() ||
                !copy_between_slots(gs.conv_state, gs.conv_state_out, off,
                                    g_.gdn_conv_stride_bytes()))
                return fail("failed to commit GDN ping-pong state");
        }
    }
    for (uint32_t slot : schedule.slot_of_token) ++step_count_for(slot);
    return true;
}

bool RawMetalDecoder::run_prefill_step(const BatchSchedule& schedule, std::string* err) {
    auto fail = [&](const std::string& why) {
        if (err) *err = "RawMetalDecoder::run_prefill_step: " + why;
        return false;
    };
    if (size_t(schedule.N) > prefill_dags_.size())
        return fail("batch exceeds prebuilt sequential prefill command-stream capacity");

    // Reset once per request, before its first encoded token.  Do not reset in
    // the token loop: later prompt rows must consume the preceding GDN/KV state.
    for (const RequestSpan& sp : schedule.spans)
        if (sp.rs_is_new) reset_state(sp.rs_slot);

    std::vector<int> next_step(size_t(g_.max_slots), 0);
    for (int s = 0; s < g_.max_slots; ++s) next_step[size_t(s)] = step_count_for(uint32_t(s));
    for (int t = 0; t < schedule.N; ++t) {
        const uint32_t slot = schedule.slot_of_token[size_t(t)];
        bind_prefill_gdn_state(*ctx_, b_, prefill_dags_[size_t(t)], slot,
                               (next_step[slot] & 1) == 0);
        ++next_step[slot];
    }

    // One command buffer, request-major token order.  Every complete layer DAG
    // ends in a barrier, so token t+1 observes token t's GDN and paged KV writes.
    ctx_->run_step([&](StepEncoder& se) {
        for (int t = 0; t < schedule.N; ++t)
            encode_decode_step_mb(se, prefill_dags_[size_t(t)], psos_, mb_psos_,
                                  force_barriers_);
    });
    for (uint32_t slot : schedule.slot_of_token) ++step_count_for(slot);
    return true;
}

const uint16_t* RawMetalDecoder::logits_bf16() const {
    return static_cast<const uint16_t*>(b_.io[int(IoSlot::Logits)].contents());
}

void RawMetalDecoder::copy_logits_f32(float* out) const {
    const uint16_t* lb = logits_bf16();
    for (int i = 0; i < g_.vocab; ++i) out[i] = bf16_to_f32(lb[i]);
}

void RawMetalDecoder::copy_batch_logits_f32(uint32_t token_row, float* out) const {
    const uint16_t* lb = logits_bf16() + size_t(token_row) * size_t(g_.vocab);
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

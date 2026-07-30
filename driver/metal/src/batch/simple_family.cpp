// See simple_family.hpp.

#include "simple_family.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "decode_abi.hpp"
#include "golden_tap.hpp"
#include "kernels/decode_psos.hpp"
#include "loader/heap_bind_metal.hpp"
#include "pie_loader/checkpoint_source.hpp"
#include "model/gemma4/bind.hpp"
#include "model/gemma4/decode_consts.hpp"
#include "model/gemma4/decode_step.hpp"
#include "model/gemma4/encode.hpp"
#include "model/gemma4/geometry.hpp"
#include "model/gemma4/kernels.hpp"
#include "model/gemma4/scratch.hpp"
#include "model/gptoss/bind.hpp"
#include "model/gptoss/decode_consts.hpp"
#include "model/gptoss/decode_step.hpp"
#include "model/gptoss/encode.hpp"
#include "model/gptoss/geometry.hpp"
#include "model/gptoss/kernels.hpp"
#include "model/gptoss/scratch.hpp"

namespace pie::metal::batch {

using pie::metal::gemma4::Gemma4Geometry;
using pie::metal::gptoss::GptOssGeometry;

SimpleFamilyEngine::~SimpleFamilyEngine() = default;

namespace {

void write_i32(const SlotHandle& s, std::int32_t v) {
    if (s.valid() && s.contents() != nullptr) *static_cast<std::int32_t*>(s.contents()) = v;
}

void write_u32s(const SlotHandle& s, const std::vector<std::uint32_t>& v) {
    if (s.valid() && s.contents() != nullptr) {
        std::memcpy(s.contents(), v.data(), v.size() * sizeof(std::uint32_t));
    }
}

/// gemma4's geometry from the config the driver read.
bool gemma4_geometry(const SetupConfig& cfg, gemma4::Gemma4Geometry& g, int max_ctx,
                     std::string* err) {
    if (!gemma4::geometry_from_facts(cfg.gemma4, g, err)) return false;
    if (cfg.vocab_size != 0) g.vocab = static_cast<int>(cfg.vocab_size);
    g.kv_max_ctx = max_ctx;
    return true;
}

/// gpt-oss's, likewise. `GptOssFacts` mirrors `geometry_from_facts`'s duck type,
/// so the geometry refuses the same three impossible shapes it always does.
bool gptoss_geometry(const SetupConfig& cfg, gptoss::GptOssGeometry& g, int max_ctx,
                     std::string* err) {
    if (!gptoss::geometry_from_facts(cfg.gptoss, g, err)) return false;
    if (cfg.vocab_size != 0) g.vocab = static_cast<int>(cfg.vocab_size);
    g.kv_max_ctx = max_ctx;
    return true;
}

/// Which bind index carries each gemma4 kind's OUTPUT, and how wide it is.
/// The names are `tests/parity/gemma4_mlx_taps.py`'s, so the engine's dump and
/// the raw path's diff against the same reference.
struct G4Tap {
    const char* name;
    std::uint8_t out_bind;
    int width;
};

bool g4_tap_for(const gemma4::Dispatch& d, const Gemma4Geometry& g, G4Tap& out) {
    using K = gemma4::Kind;
    const int L = d.layer;
    const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;
    const int q_dim = g.n_q_heads * hd;
    const int kv_dim = g.n_kv_heads * hd;
    const int inter = L >= 0 ? g.intermediate_of(L) : g.intermediate;
    const int ple_all = g.n_layers * g.per_layer_emb_dim;
    switch (d.kind) {
        case K::EmbedGather:      out = {"embed", 4, g.hidden};          return true;
        case K::PleTokenGather:   out = {"ple_tok", 4, ple_all};         return true;
        case K::PleProjGemv:      out = {"ple_proj", 4, ple_all};        return true;
        case K::PleProjNorm:      out = {"ple_projnorm", 2, ple_all};    return true;
        case K::PleCombine:       out = {"ple", 2, ple_all};             return true;
        case K::AttnNorm:         out = {"attn_norm", 2, g.hidden};      return true;
        case K::QmvQ:             out = {"q_proj", 4, q_dim};            return true;
        case K::QmvK:             out = {"k_proj", 4, kv_dim};           return true;
        case K::QmvV:             out = {"v_proj", 4, kv_dim};           return true;
        case K::QNorm:            out = {"q_norm", 2, q_dim};            return true;
        case K::KNorm:            out = {"k_norm", 2, kv_dim};           return true;
        case K::VNorm:            out = {"v_norm", 1, kv_dim};           return true;
        case K::RopeQ:            out = {"rope_q", 0, q_dim};            return true;
        case K::RopeK:            out = {"rope_k", 0, kv_dim};           return true;
        case K::Sdpa:             out = {"sdpa", 3, q_dim};              return true;
        case K::QmvO:             out = {"o_proj", 4, g.hidden};         return true;
        case K::PostAttnResidual: out = {"attn_resid", 2, g.hidden};     return true;
        case K::FfnNorm:          out = {"ffn_norm", 2, g.hidden};       return true;
        case K::QmvGate:          out = {"gate_proj", 4, inter};         return true;
        case K::QmvUp:            out = {"up_proj", 4, inter};           return true;
        case K::GegluTanh:        out = {"geglu", 2, inter};             return true;
        case K::QmvDown:          out = {"down_proj", 4, g.hidden};      return true;
        case K::PostFfnResidual:  out = {"ffn_resid", 2, g.hidden};      return true;
        case K::PleGateGemv:      out = {"ple_gate", 4, g.per_layer_emb_dim}; return true;
        case K::PleGeglu:         out = {"ple_act", 2, g.per_layer_emb_dim};  return true;
        case K::PleProjLayerGemv: out = {"ple_back", 4, g.hidden};       return true;
        case K::PleResidualScaled:
        case K::LayerScalar:      out = {"layer_out", 2, g.hidden};      return true;
        case K::FinalRms:         out = {"final_norm", 2, g.hidden};     return true;
        case K::LmHead:           out = {"logits_raw", 4, g.vocab};      return true;
        case K::FinalSoftcap:     out = {"logits", 1, g.vocab};          return true;
        default: return false;
    }
}

/// How wide each pool colour has to be, in elements.
///
/// Sizing every slot at the widest activation -- the vocabulary -- costs
/// `colors * rows * vocab * 2`, which at a 10240-row fire is 174 GB of address
/// space for a model whose weights are 2.6. Only the two tail colours are
/// vocabulary-wide, and only the tail runs at `head_rows`; everything else is
/// hidden- or PLE-table-wide over every row.
std::vector<std::size_t> g4_pool_elems(const std::vector<gemma4::Dispatch>& dag,
                                       const Gemma4Geometry& g,
                                       const gemma4::ScratchColoring& col, int rows,
                                       int head_rows) {
    std::vector<std::size_t> elems(std::size_t(col.colors_used), 0);
    for (std::size_t di = 0; di < dag.size() && di < col.per_dispatch.size(); ++di) {
        G4Tap t{};
        if (!g4_tap_for(dag[di], g, t)) continue;
        // The tail's tensors have one row per SAMPLED row; the body's have one
        // per token.
        const bool tail = dag[di].kind == gemma4::Kind::LmHead ||
                          dag[di].kind == gemma4::Kind::FinalSoftcap ||
                          dag[di].kind == gemma4::Kind::FinalRms ||
                          dag[di].kind == gemma4::Kind::RowGather;
        const std::size_t need =
            std::size_t(tail ? head_rows : rows) * std::size_t(t.width);
        for (const auto& sb : col.per_dispatch[di]) {
            if (sb.color < 0 || sb.color >= col.colors_used) continue;
            // Every buffer of the dispatch, not only its output: an input read
            // at this width must be at least this wide too.
            elems[std::size_t(sb.color)] = std::max(elems[std::size_t(sb.color)], need);
        }
    }
    for (std::size_t& e : elems) {
        if (e == 0) e = std::size_t(rows) * std::size_t(g.hidden);
    }
    return elems;
}

/// Only a colour's FINAL writer is named: the in-place kinds share a buffer with
/// the tap before them, and publishing the earlier tensor under the earlier name
/// reads as a divergence that is really the dump lying.
void dump_g4_taps(const std::vector<gemma4::Dispatch>& dag, const Gemma4Geometry& g,
                  const gemma4::ScratchColoring& col, const std::vector<SlotHandle>& pool,
                  const SlotHandle& logits, int rows, int head_rows) {
    const int n_pool = int(pool.size());
    const auto colour_of = [&](std::size_t di, std::uint8_t bind_index) {
        for (const auto& sb : col.per_dispatch[di]) {
            if (sb.bind_index == bind_index) return int(sb.color);
        }
        return -1;
    };
    std::vector<int> last(std::size_t(n_pool < 0 ? 0 : n_pool), -1);
    for (std::size_t di = 0; di < dag.size(); ++di) {
        G4Tap t{};
        if (!g4_tap_for(dag[di], g, t)) continue;
        const int c = colour_of(di, t.out_bind);
        if (c >= 0 && c < n_pool) last[std::size_t(c)] = int(di);
    }
    for (std::size_t di = 0; di < dag.size(); ++di) {
        G4Tap t{};
        if (!g4_tap_for(dag[di], g, t)) continue;
        using K = gemma4::Kind;
        const bool tail = dag[di].kind == K::RowGather || dag[di].kind == K::FinalRms ||
                          dag[di].kind == K::LmHead || dag[di].kind == K::FinalSoftcap;
        // The engine re-points the LAST dispatch at its own logits slot, so its
        // pool colour is never written; dumping that colour publishes zeros
        // under the name of the tensor everything downstream depends on.
        const bool redirected = di + 1 == dag.size();
        const void* src = nullptr;
        if (redirected) {
            src = logits.valid() ? logits.contents() : nullptr;
        } else {
            const int c = colour_of(di, t.out_bind);
            if (c < 0 || c >= n_pool || !pool[std::size_t(c)].valid()) continue;
            if (last[std::size_t(c)] != int(di)) continue;
            src = pool[std::size_t(c)].contents();
        }
        if (src == nullptr) continue;
        const std::string name = dag[di].layer < 0
            ? std::string(t.name)
            : std::to_string(dag[di].layer) + "." + t.name;
        dump_golden_bf16(name, src, tail ? head_rows : rows, t.width, t.width);
    }
}

// ── gemma4 ──────────────────────────────────────────────────────────────────

class Gemma4Engine final : public SimpleFamilyEngine {
  public:
    bool init(RawMetalContext& ctx, const std::string& kernels_dir, const SetupConfig& cfg,
              const pie_loader::LoadPlan& load_plan, int max_ctx, std::string* err) {
        if (!gemma4_geometry(cfg, g_, max_ctx, err)) return false;
        max_ctx_ = max_ctx;
        // ONE schedule for both paths. `gemma4_prefill_numerics_test` measures
        // that a decode step is a fire of one row and lands where the whole
        // prompt does, so there is no reason to carry a second, M=1-only DAG
        // whose only distinction is that it cannot batch.
        g_.paged_kv_enabled = true;
        // The runtime allocates pages and hands their PHYSICAL ids down, so the
        // engine's page size has to be the one the runtime was configured with.
        g_.kv_page_size = cfg.kv_page_size > 0 ? int(cfg.kv_page_size) : kPageSize;
        const int ps = g_.kv_page_size;
        g_.kv_max_ctx = ((max_ctx_ + ps - 1) / ps) * ps;
        g_.total_pages = g_.kv_max_ctx / ps;
        max_rows_ = cfg.max_forward_tokens > 0 ? int(cfg.max_forward_tokens) : 1;
        // At most one row per request is sampled -- the last of its span, whose
        // logits become its next token -- so the tail is bounded by the request
        // count, not the token count.
        max_sampled_ = cfg.max_forward_requests > 0 ? int(cfg.max_forward_requests) : 1;
        if (max_sampled_ > max_rows_) max_sampled_ = max_rows_;

        try {
            const auto storage = load_plan.view();
            pie_loader::CheckpointSource view(storage);
            StagedWeights staged =
                stage_plan_weights(ctx, view, load_plan, storage.memory.persistent_bytes);
            b_.weights = std::move(staged.weights);
        } catch (const std::exception& e) {
            if (err) *err = std::string("staging gemma4's weights: ") + e.what();
            return false;
        }

        // Paged KV, one pool per OWNING layer: the shared tail attends pages an
        // earlier layer wrote, and the two attention types are different widths.
        kpages_.resize(std::size_t(g_.n_layers));
        vpages_.resize(std::size_t(g_.n_layers));
        b_.kv.resize(std::size_t(g_.n_layers));
        for (int L = 0; L < g_.n_layers; ++L) {
            if (g_.is_kv_shared(L)) continue;
            const int hd = g_.head_dim_of(L);
            const std::size_t bytes = std::size_t(g_.total_pages) * std::size_t(g_.kv_page_size) *
                                      std::size_t(g_.n_kv_heads) * std::size_t(hd) * 2;
            kpages_[std::size_t(L)] = ctx.heap_alloc(bytes);
            vpages_[std::size_t(L)] = ctx.heap_alloc(bytes);
            if (!kpages_[std::size_t(L)].valid() || !vpages_[std::size_t(L)].valid()) {
                if (err) *err = "gemma4 KV allocation failed";
                return false;
            }
            b_.kv[std::size_t(L)].k = kpages_[std::size_t(L)];
            b_.kv[std::size_t(L)].v = vpages_[std::size_t(L)];
        }

        dag_ = gemma4::build_gemma4_dag_mb(g_, /*ordinal_base=*/0, /*with_argmax=*/false);
        const gemma4::ScratchPlan sp = gemma4::build_gemma4_scratch(dag_, g_);
        // Under a tap dump every value needs its own buffer, or a later
        // dispatch overwrites the one being read.
        coloring_ = gemma4::color_gemma4_scratch(dag_, sp, /*no_recycle=*/golden_taps_enabled());
        if (!coloring_.hazard_free) {
            if (err) *err = "gemma4's activation colouring is not hazard-free";
            return false;
        }
        // Every activation is [rows, width] row-major at M>1, so a pool slot is
        // as many rows of its own width as the widest dispatch that touches it.
        b_.pool.resize(std::size_t(coloring_.colors_used));
        // The projections pad their row count to a whole GEMM tile, so the pool
        // has to hold the padded count -- the padding rows are written.
        const std::vector<std::size_t> elems =
            g4_pool_elems(dag_, g_, coloring_, gemma4::gemma4_qmm_pool_rows(max_rows_),
                          gemma4::gemma4_qmm_pool_rows(max_sampled_));
        for (int c = 0; c < coloring_.colors_used; ++c) {
            b_.pool[std::size_t(c)] = ctx.heap_alloc(elems[std::size_t(c)] * 2);
        }

        b_.io.resize(kIoSlotCount);
        const std::size_t io_bytes =
            std::max<std::size_t>(4096, std::size_t(g_.total_pages + max_rows_ + 8) * 4);
        for (int i = 0; i < kIoSlotCount; ++i) b_.io[i] = ctx.heap_alloc(io_bytes);
        // The logits leave the pool: the sampler reads a slot of its own, so the
        // tail writes there and nothing copies afterwards. One row per SAMPLED
        // row, which is what the tail produces.
        logits_ = ctx.heap_alloc(std::size_t(gemma4::gemma4_qmm_pool_rows(max_sampled_)) *
                                 std::size_t(g_.vocab) * 2);
        if (!logits_.valid()) {
            if (err) *err = "gemma4 logits allocation failed";
            return false;
        }

        if (!gemma4::build_gemma4_psos(ctx, kernels_dir, psos_, err)) return false;
        if (!load_decode_psos(ctx, kernels_dir, base_, /*with_argmax=*/false, err)) return false;
        if (!load_multibatch_psos(ctx, kernels_dir, mb_, /*with_d512=*/true, err)) return false;

        gemma4::bind_gemma4_consts(ctx, dag_, g_, /*rows=*/1, /*paged=*/true, /*head_rows=*/1);
        bound_rows_ = 1;
        bound_head_rows_ = 1;
        try {
            gemma4::bind_gemma4_dag_mb(ctx, b_, dag_, g_, coloring_, kpages_, vpages_);
        } catch (const std::exception& e) {
            if (err) *err = std::string("binding gemma4: ") + e.what();
            return false;
        }
        // Re-point the tail's output at the logits slot. The colouring gave it a
        // pool buffer, which is right for a test that reads it back and wrong for
        // an engine that samples it: the pool recycles.
        const int tail = int(dag_.size()) - 1;
        const std::uint8_t out_bind = g_.final_softcap > 0.0f
                                          ? (std::uint8_t)bind::Softcap::Out
                                          : (std::uint8_t)bind::Qmv::Out;
        ctx.arg_bind_ordinal(dag_[std::size_t(tail)].ordinal, out_bind, logits_);

        // The page list is the identity: this engine owns the whole pool and
        // hands it to whichever sequence is resident.
        std::vector<std::uint32_t> ids(std::size_t(g_.total_pages));
        for (int p = 0; p < g_.total_pages; ++p) ids[std::size_t(p)] = std::uint32_t(p);
        write_u32s(b_.io[int(IoSlot::KvPageIndices)], ids);
        write_u32s(b_.io[int(IoSlot::KvPageIndptr)], {0u, std::uint32_t(g_.total_pages)});
        write_u32s(b_.io[int(IoSlot::AttnMaskStride)], {0u});
        return true;
    }

    int vocab() const override { return g_.vocab; }
    int n_layers() const override { return g_.n_layers; }

    void reset() override {
        for (int L = 0; L < g_.n_layers; ++L) {
            if (g_.is_kv_shared(L)) continue;
            SlotHandle& k = kpages_[std::size_t(L)];
            SlotHandle& v = vpages_[std::size_t(L)];
            if (k.valid() && k.contents()) std::memset(k.contents(), 0, k.size);
            if (v.valid() && v.contents()) std::memset(v.contents(), 0, v.size);
        }
    }

    StepTiming step(RawMetalContext& ctx, std::uint32_t token_id,
                    std::uint32_t position) override {
        // A decode step is a fire of one row, owned by one request whose
        // history is the whole (identity) page list.
        FireCsr csr;
        csr.token_ids = {token_id};
        csr.position_ids = {position};
        csr.req_of_token = {0u};
        csr.w_page = {position / std::uint32_t(g_.kv_page_size)};
        csr.w_off = {position % std::uint32_t(g_.kv_page_size)};
        csr.qo_indptr = {0u, 1u};
        csr.kv_page_indices.resize(std::size_t(g_.total_pages));
        for (int p = 0; p < g_.total_pages; ++p) {
            csr.kv_page_indices[std::size_t(p)] = std::uint32_t(p);
        }
        csr.kv_page_indptr = {0u, std::uint32_t(g_.total_pages)};
        csr.sample_rows = {0u};
        return fire(ctx, csr, {}, {});
    }

    bool paged() const override { return true; }
    int max_rows() const override { return max_rows_; }
    int max_sampled_rows() const override { return max_sampled_; }
    int page_size() const override { return g_.kv_page_size; }
    int total_pages() const override { return g_.total_pages; }

    StepTiming fire(RawMetalContext& ctx, const FireCsr& csr, const EncodeHook& pre,
                    const EncodeHook& post) override {
        const int rows = int(csr.token_ids.size());
        if (rows <= 0 || rows > max_rows_) return StepTiming{};
        write_u32s(b_.io[int(IoSlot::TokenId)], csr.token_ids);
        write_u32s(b_.io[int(IoSlot::Position)], csr.position_ids);
        write_u32s(b_.io[int(IoSlot::ReqOfToken)], csr.req_of_token);
        write_u32s(b_.io[int(IoSlot::WPage)], csr.w_page);
        write_u32s(b_.io[int(IoSlot::WOff)], csr.w_off);
        write_u32s(b_.io[int(IoSlot::QoIndptr)], csr.qo_indptr);
        write_u32s(b_.io[int(IoSlot::KvPageIndices)], csr.kv_page_indices);
        write_u32s(b_.io[int(IoSlot::KvPageIndptr)], csr.kv_page_indptr);
        const int head_rows = csr.sample_rows.empty() ? rows : int(csr.sample_rows.size());
        if (csr.sample_rows.empty()) {
            std::vector<std::uint32_t> every;
            every.resize(std::size_t(rows));
            for (int r = 0; r < rows; ++r) every[std::size_t(r)] = std::uint32_t(r);
            write_u32s(b_.io[int(IoSlot::SampleRows)], every);
        } else {
            write_u32s(b_.io[int(IoSlot::SampleRows)], csr.sample_rows);
        }
        // The causal bound at M>1 comes from PositionIds, and the window from
        // the layer's own constant, so no dense mask is needed. SeqLen is the
        // M=1 ring's port and is set for the ABI's sake, not because the paged
        // kernels read it.
        write_i32(b_.io[int(IoSlot::SeqLen)],
                  std::int32_t(csr.position_ids.back()) + 1);
        if (b_.io[int(IoSlot::AttnMaskEnabled)].contents() != nullptr) {
            std::memset(b_.io[int(IoSlot::AttnMaskEnabled)].contents(), 0, std::size_t(rows));
        }
        if (bound_rows_ != rows || bound_head_rows_ != head_rows) {
            gemma4::bind_gemma4_consts(ctx, dag_, g_, rows, /*paged=*/true, head_rows);
            bound_rows_ = rows;
            bound_head_rows_ = head_rows;
        }
        return ctx.run_step([&](StepEncoder& se) {
            if (pre) pre(se);
            gemma4::encode_gemma4_step_mb(se, dag_, g_, rows, base_, mb_, psos_,
                                          /*ordinal_base=*/0, head_rows);
            if (post) post(se);
        });
    }

    SlotHandle logits_slot() const override { return logits_; }

    void dump_taps(int rows) const override {
        dump_g4_taps(dag_, g_, coloring_, b_.pool, logits_, rows, bound_head_rows_);
    }

  private:
    static constexpr int kPageSize = 32;

    gemma4::Gemma4Geometry g_{};
    int max_ctx_ = 0;
    int max_rows_ = 1;
    int max_sampled_ = 1;
    int bound_rows_ = 0;
    int bound_head_rows_ = 0;
    std::vector<gemma4::Dispatch> dag_{};
    gemma4::ScratchColoring coloring_{};
    gemma4::BoundGemma4 b_{};
    gemma4::Gemma4Psos psos_{};
    DecodeStepPsos base_{};
    MultiBatchPsos mb_{};
    std::vector<SlotHandle> kpages_{};
    std::vector<SlotHandle> vpages_{};
    SlotHandle logits_{};
};

/// Which bind index carries each gpt-oss kind's OUTPUT, and how wide it is.
/// The names are `tests/parity/gptoss_mlx_taps.py`'s, so the engine's dump and
/// the raw path's diff against the same reference.
struct GoTap {
    const char* name;
    std::uint8_t out_bind;
    int width;
};

bool go_tap_for(const gptoss::Dispatch& d, const GptOssGeometry& g, GoTap& out) {
    using K = gptoss::Kind;
    switch (d.kind) {
        case K::EmbedGather:   out = {"embed",       4, g.hidden};   return true;
        case K::AttnNorm:      out = {"attn_norm",   2, g.hidden};   return true;
        case K::QmvQ:          out = {"q_proj",      4, g.q_dim()};  return true;
        case K::QmvK:          out = {"k_proj",      4, g.kv_dim()}; return true;
        case K::QmvV:          out = {"v_proj",      4, g.kv_dim()}; return true;
        case K::RopeQ:         out = {"rope_q",      0, g.q_dim()};  return true;
        case K::RopeK:         out = {"rope_k",      0, g.kv_dim()}; return true;
        case K::SdpaSink:      out = {"sdpa",        3, g.q_dim()};  return true;
        case K::QmvO:          out = {"o_proj",      4, g.hidden};   return true;
        case K::AttnResidual:  out = {"attn_resid",  2, g.hidden};   return true;
        case K::FfnNorm:       out = {"ffn_norm",    2, g.hidden};   return true;
        case K::RouterGemv:    out = {"router",      4, g.n_experts}; return true;
        case K::ExpertGate:
            out = {"expert_gate", 4, g.experts_per_token * g.intermediate}; return true;
        case K::ExpertUp:
            out = {"expert_up",   4, g.experts_per_token * g.intermediate}; return true;
        case K::ExpertSwiGlu:
            out = {"expert_act",  2, g.experts_per_token * g.intermediate}; return true;
        case K::ExpertDown:
            out = {"expert_out",  4, g.experts_per_token * g.hidden};   return true;
        case K::ExpertCombine: out = {"moe",         2, g.hidden};   return true;
        case K::FfnResidual:   out = {"layer_out",   2, g.hidden};   return true;
        case K::FinalRms:      out = {"final_norm",  2, g.hidden};   return true;
        case K::LmHead:        out = {"logits",      4, g.vocab};    return true;
        default: return false;
    }
}

/// Only a colour's FINAL writer is named: the in-place kinds (both ropes) share
/// a buffer with the tap before them, and publishing the earlier tensor under
/// the earlier name reads as a divergence that is really the dump lying.
void dump_go_taps(const std::vector<gptoss::Dispatch>& dag, const GptOssGeometry& g,
                  const gptoss::ScratchColoring& col, const std::vector<SlotHandle>& pool,
                  const SlotHandle& logits, int rows, int head_rows) {
    const int n_pool = int(pool.size());
    const auto colour_of = [&](std::size_t di, std::uint8_t bind_index) {
        for (const auto& sb : col.per_dispatch[di]) {
            if (sb.bind_index == bind_index) return int(sb.color);
        }
        return -1;
    };
    std::vector<int> last(std::size_t(n_pool < 0 ? 0 : n_pool), -1);
    for (std::size_t di = 0; di < dag.size(); ++di) {
        GoTap t{};
        if (!go_tap_for(dag[di], g, t)) continue;
        const int c = colour_of(di, t.out_bind);
        if (c >= 0 && c < n_pool) last[std::size_t(c)] = int(di);
    }
    for (std::size_t di = 0; di < dag.size(); ++di) {
        GoTap t{};
        if (!go_tap_for(dag[di], g, t)) continue;
        using K = gptoss::Kind;
        const bool tail = dag[di].kind == K::FinalRms || dag[di].kind == K::LmHead;
        // The engine re-points the LAST dispatch at its own logits slot, so the
        // pool colour it was coloured into is never written. Dumping that
        // colour publishes a buffer of zeros under the name of the tensor
        // everything downstream depends on -- a divergence that is entirely the
        // dump lying.
        const bool redirected = di + 1 == dag.size();
        const void* src = nullptr;
        if (redirected) {
            src = logits.valid() ? logits.contents() : nullptr;
        } else {
            const int c = colour_of(di, t.out_bind);
            if (c < 0 || c >= n_pool || !pool[std::size_t(c)].valid()) continue;
            if (last[std::size_t(c)] != int(di)) continue;
            src = pool[std::size_t(c)].contents();
        }
        if (src == nullptr) continue;
        const std::string name = dag[di].layer < 0
            ? std::string(t.name)
            : std::to_string(dag[di].layer) + "." + t.name;
        dump_golden_bf16(name, src, tail ? head_rows : rows, t.width, t.width);
    }
}

/// The widest buffer a gpt-oss dispatch touches, in bf16 elements.
///
/// Conservative per KIND rather than per bind index -- a colour gets the widest
/// tensor any dispatch that touches it handles -- which is tight enough to
/// matter (`hidden` is 2880 against a 201088 vocabulary) and cannot under-count
/// a buffer the way a per-index table with a missing entry would.
int go_kind_width(gptoss::Kind k, const GptOssGeometry& g) {
    using K = gptoss::Kind;
    const int stack = g.experts_per_token * g.intermediate;
    const int down = g.experts_per_token * g.hidden;
    switch (k) {
        case K::QmvQ: case K::QmvK: case K::QmvV: case K::QmvO:
        case K::RopeQ: case K::SdpaSink:  return std::max(g.hidden, g.q_dim());
        case K::RopeK: case K::KvAppend:  return std::max(g.kv_dim(), 1);
        case K::RouterGemv:               return std::max(g.hidden, g.n_experts);
        // The ids are 4-byte ints, so they count double against a bf16 slot.
        case K::RouterTopK:               return std::max(g.n_experts, 2 * g.experts_per_token);
        case K::ExpertGate: case K::ExpertUp: return std::max(g.hidden, stack);
        case K::ExpertSwiGlu:             return stack;
        case K::ExpertDown:               return std::max(stack, down);
        case K::ExpertCombine:            return std::max(down, g.hidden);
        case K::LmHead:                   return std::max(g.hidden, g.vocab);
        default:                          return g.hidden;
    }
}

std::vector<std::size_t> go_pool_elems(const std::vector<gptoss::Dispatch>& dag,
                                       const GptOssGeometry& g,
                                       const gptoss::ScratchColoring& col, int rows,
                                       int head_rows) {
    std::vector<std::size_t> elems(std::size_t(col.colors_used), 0);
    for (std::size_t di = 0; di < dag.size() && di < col.per_dispatch.size(); ++di) {
        using K = gptoss::Kind;
        const K kind = dag[di].kind;
        // The tail's tensors have one row per SAMPLED row; the body's have one
        // per token.
        const bool tail = kind == K::RowGather || kind == K::FinalRms || kind == K::LmHead;
        const std::size_t need = std::size_t(tail ? head_rows : rows) *
                                 std::size_t(go_kind_width(kind, g));
        for (const auto& sb : col.per_dispatch[di]) {
            if (sb.color < 0 || sb.color >= col.colors_used) continue;
            elems[std::size_t(sb.color)] = std::max(elems[std::size_t(sb.color)], need);
        }
    }
    for (std::size_t& e : elems) {
        if (e == 0) e = std::size_t(rows) * std::size_t(g.hidden);
    }
    return elems;
}

// ── gpt-oss ─────────────────────────────────────────────────────────────────

class GptOssEngine final : public SimpleFamilyEngine {
  public:
    bool init(RawMetalContext& ctx, const std::string& kernels_dir, const SetupConfig& cfg,
              const pie_loader::LoadPlan& load_plan, int max_ctx, std::string* err) {
        if (!gptoss_geometry(cfg, g_, max_ctx, err)) return false;
        max_ctx_ = max_ctx;

        try {
            const auto storage = load_plan.view();
            pie_loader::CheckpointSource view(storage);
            StagedWeights staged =
                stage_plan_weights(ctx, view, load_plan, storage.memory.persistent_bytes);
            b_.weights = std::move(staged.weights);
        } catch (const std::exception& e) {
            if (err) *err = std::string("staging gpt-oss's weights: ") + e.what();
            return false;
        }

        // Paged KV. gpt-oss has no M>1 path -- its MoE picks experts per ROW,
        // so a batched routed matmul is a different kernel rather than a wider
        // launch -- but paging is orthogonal to that: what it buys is SEVERAL
        // SEQUENCES, each attending its own page list, instead of one ring that
        // the second sequence clobbers.
        g_.paged_kv_enabled = true;
        g_.kv_page_size = cfg.kv_page_size > 0 ? int(cfg.kv_page_size) : 32;
        const int ps = g_.kv_page_size;
        g_.kv_max_ctx = ((max_ctx_ + ps - 1) / ps) * ps;
        g_.total_pages = g_.kv_max_ctx / ps;
        max_sampled_ = cfg.max_forward_requests > 0 ? int(cfg.max_forward_requests) : 1;
        max_rows_ = cfg.max_forward_tokens > 0 ? int(cfg.max_forward_tokens) : 1;
        if (max_sampled_ > max_rows_) max_sampled_ = max_rows_;

        kpages_.resize(std::size_t(g_.n_layers));
        vpages_.resize(std::size_t(g_.n_layers));
        b_.kv.resize(std::size_t(g_.n_layers));
        for (int L = 0; L < g_.n_layers; ++L) {
            const std::size_t bytes = std::size_t(g_.total_pages) * std::size_t(g_.kv_page_size) *
                                      std::size_t(g_.n_kv_heads) * std::size_t(g_.head_dim) * 2;
            kpages_[std::size_t(L)] = ctx.heap_alloc(bytes);
            vpages_[std::size_t(L)] = ctx.heap_alloc(bytes);
            if (!kpages_[std::size_t(L)].valid() || !vpages_[std::size_t(L)].valid()) {
                if (err) *err = "gpt-oss KV allocation failed";
                return false;
            }
            b_.kv[std::size_t(L)].k = kpages_[std::size_t(L)];
            b_.kv[std::size_t(L)].v = vpages_[std::size_t(L)];
        }

        dag_ = gptoss::build_gptoss_dag(g_, /*with_argmax=*/false);
        const gptoss::ScratchPlan sp = gptoss::build_gptoss_scratch(dag_, g_);
        // Under a tap dump every value needs its own buffer, or a later
        // dispatch overwrites the one being read.
        coloring_ = gptoss::color_gptoss_scratch(dag_, sp, /*no_recycle=*/golden_taps_enabled());
        if (!coloring_.hazard_free) {
            if (err) *err = "gpt-oss's activation colouring is not hazard-free";
            return false;
        }
        // Every activation is [rows, width] row-major at M>1, so a pool slot is
        // as many rows of its own width as the widest dispatch that touches it.
        // Sizing them all at the vocabulary costs 70x on the body's tensors.
        b_.pool.resize(std::size_t(coloring_.colors_used));
        const std::vector<std::size_t> elems =
            go_pool_elems(dag_, g_, coloring_, max_rows_, max_sampled_);
        for (int c = 0; c < coloring_.colors_used; ++c) {
            b_.pool[std::size_t(c)] = ctx.heap_alloc(elems[std::size_t(c)] * 2);
        }

        b_.io.resize(kIoSlotCount);
        const std::size_t io_bytes =
            std::max<std::size_t>(4096, std::size_t(g_.total_pages + 8) * 4);
        for (int i = 0; i < kIoSlotCount; ++i) b_.io[i] = ctx.heap_alloc(io_bytes);
        // One row per SAMPLED row, plus one more. The fire runs its rows one at
        // a time and the tail writes on EVERY one, so the rows nobody reads
        // need somewhere to land that is not somewhere a sampled row already
        // landed -- otherwise a later member's prompt overwrites an earlier
        // member's answer, which reads as that member being answered from its
        // neighbour's prompt.
        logits_ = ctx.heap_alloc(std::size_t(max_sampled_ + 1) * std::size_t(g_.vocab) * 2);
        if (!logits_.valid()) {
            if (err) *err = "gpt-oss logits allocation failed";
            return false;
        }

        if (!gptoss::build_gptoss_psos(ctx, kernels_dir, psos_, err)) return false;
        if (!load_decode_psos(ctx, kernels_dir, base_, /*with_argmax=*/false, err)) return false;
        if (!load_multibatch_psos(ctx, kernels_dir, mb_, /*with_d512=*/false, err)) return false;

        gptoss::bind_gptoss_consts(ctx, dag_, g_, /*rows=*/1, /*paged=*/true, /*head_rows=*/1);
        bound_rows_ = 1;
        bound_head_rows_ = 1;
        try {
            gptoss::bind_gptoss_dag_paged(ctx, b_, dag_, g_, coloring_, kpages_, vpages_);
        } catch (const std::exception& e) {
            if (err) *err = std::string("binding gpt-oss: ") + e.what();
            return false;
        }
        // The logits leave the pool: the sampler reads a slot of its own, so
        // the tail writes there and nothing copies afterwards.
        ctx.arg_bind_ordinal(dag_.back().ordinal, (std::uint8_t)bind::GoQmv::Out, logits_);
        write_u32s(b_.io[int(IoSlot::AttnMaskStride)], {0u});
        if (b_.io[int(IoSlot::AttnMaskEnabled)].contents() != nullptr) {
            std::memset(b_.io[int(IoSlot::AttnMaskEnabled)].contents(), 0, 1);
        }
        return true;
    }

    int vocab() const override { return g_.vocab; }
    int n_layers() const override { return g_.n_layers; }

    void reset() override {
        for (int L = 0; L < g_.n_layers; ++L) {
            SlotHandle& k = kpages_[std::size_t(L)];
            SlotHandle& v = vpages_[std::size_t(L)];
            if (k.valid() && k.contents()) std::memset(k.contents(), 0, k.size);
            if (v.valid() && v.contents()) std::memset(v.contents(), 0, v.size);
        }
    }

    StepTiming step(RawMetalContext& ctx, std::uint32_t token_id,
                    std::uint32_t position) override {
        FireCsr csr;
        csr.token_ids = {token_id};
        csr.position_ids = {position};
        csr.req_of_token = {0u};
        csr.w_page = {position / std::uint32_t(g_.kv_page_size)};
        csr.w_off = {position % std::uint32_t(g_.kv_page_size)};
        csr.qo_indptr = {0u, 1u};
        csr.kv_page_indices.resize(std::size_t(g_.total_pages));
        for (int p = 0; p < g_.total_pages; ++p) {
            csr.kv_page_indices[std::size_t(p)] = std::uint32_t(p);
        }
        csr.kv_page_indptr = {0u, std::uint32_t(g_.total_pages)};
        csr.sample_rows = {0u};
        return fire(ctx, csr, {}, {});
    }

    bool paged() const override { return true; }
    /// A fire of R rows is R PASSES, not one wider one: gpt-oss has no M>1
    /// path, because its MoE picks experts per row. So the row budget costs
    /// time rather than memory, and what paging buys is that those passes may
    /// belong to different sequences.
    int max_rows() const override { return max_rows_; }
    int max_sampled_rows() const override { return max_sampled_; }
    int page_size() const override { return g_.kv_page_size; }
    int total_pages() const override { return g_.total_pages; }

    StepTiming fire(RawMetalContext& ctx, const FireCsr& csr, const EncodeHook& pre,
                    const EncodeHook& post) override {
        const int rows = int(csr.token_ids.size());
        if (rows <= 0 || rows > max_rows()) return StepTiming{};
        write_u32s(b_.io[int(IoSlot::TokenId)], csr.token_ids);
        write_u32s(b_.io[int(IoSlot::Position)], csr.position_ids);
        write_u32s(b_.io[int(IoSlot::ReqOfToken)], csr.req_of_token);
        write_u32s(b_.io[int(IoSlot::WPage)], csr.w_page);
        write_u32s(b_.io[int(IoSlot::WOff)], csr.w_off);
        write_u32s(b_.io[int(IoSlot::QoIndptr)], csr.qo_indptr);
        write_u32s(b_.io[int(IoSlot::KvPageIndices)], csr.kv_page_indices);
        write_u32s(b_.io[int(IoSlot::KvPageIndptr)], csr.kv_page_indptr);
        write_i32(b_.io[int(IoSlot::SeqLen)], std::int32_t(csr.position_ids.back()) + 1);
        const int head_rows = csr.sample_rows.empty() ? rows : int(csr.sample_rows.size());
        if (csr.sample_rows.empty()) {
            std::vector<std::uint32_t> every;
            every.resize(std::size_t(rows));
            for (int r = 0; r < rows; ++r) every[std::size_t(r)] = std::uint32_t(r);
            write_u32s(b_.io[int(IoSlot::SampleRows)], every);
        } else {
            write_u32s(b_.io[int(IoSlot::SampleRows)], csr.sample_rows);
        }
        if (b_.io[int(IoSlot::AttnMaskEnabled)].contents() != nullptr) {
            std::memset(b_.io[int(IoSlot::AttnMaskEnabled)].contents(), 0, std::size_t(rows));
        }
        if (bound_rows_ != rows || bound_head_rows_ != head_rows) {
            gptoss::bind_gptoss_consts(ctx, dag_, g_, rows, /*paged=*/true, head_rows);
            bound_rows_ = rows;
            bound_head_rows_ = head_rows;
        }
        return ctx.run_step([&](StepEncoder& se) {
            if (pre) pre(se);
            gptoss::encode_gptoss_step_mb(se, dag_, g_, rows, base_, mb_, psos_,
                                          /*ordinal_base=*/0, head_rows);
            if (post) post(se);
        });
    }

    SlotHandle logits_slot() const override { return logits_; }

    void dump_taps(int rows) const override {
        dump_go_taps(dag_, g_, coloring_, b_.pool, logits_, rows, bound_head_rows_);
    }

  private:
    gptoss::GptOssGeometry g_{};
    int max_ctx_ = 0;
    std::vector<gptoss::Dispatch> dag_{};
    gptoss::ScratchColoring coloring_{};
    gptoss::BoundGptOss b_{};
    gptoss::GptOssPsos psos_{};
    DecodeStepPsos base_{};
    MultiBatchPsos mb_{};
    std::vector<SlotHandle> kpages_{};
    std::vector<SlotHandle> vpages_{};
    int max_sampled_ = 1;
    int max_rows_ = 1;
    int bound_rows_ = 0;
    int bound_head_rows_ = 0;
    SlotHandle logits_{};
};

}  // namespace

std::size_t SimpleFamilyEngine::extra_heap_bytes(pie::metal::model::ModelFamily family,
                                                 const SetupConfig& cfg, int max_ctx) {
    // KV + activation pool + logits + constants, with slack. Deliberately
    // generous: this is a budget, and a context that is too small fails at
    // `heap_alloc` with no diagnosis of which allocation ran out.
    std::size_t bytes = std::size_t(256) << 20;
    if (family == pie::metal::model::ModelFamily::Gemma4) {
        gemma4::Gemma4Geometry g;
        std::string ignore;
        if (!gemma4_geometry(cfg, g, max_ctx, &ignore)) return bytes;
        bytes += gemma4::gemma4_kv_region_bytes(g, max_ctx, 2);
        // The engine pages its KV, so it allocates per-owning-layer pools whose
        // total rounds `max_ctx` up to a page. One page's slack per layer.
        bytes += std::size_t(g.n_layers) * 2 * 32 * std::size_t(g.n_kv_heads) *
                 std::size_t(g.global_head_dim > 0 ? g.global_head_dim : g.head_dim) * 2;
        // The activation pool, summed from the SAME colouring the engine will
        // build. The DAG, the dataflow and the colouring are pure, so asking
        // them here costs microseconds and removes the guess -- a fudge factor
        // that over-counts by 32x on the widest slot is 174 GB at a 10240-row
        // fire, and one that under-counts fails at `heap_alloc` naming nothing.
        int rows = cfg.max_forward_tokens > 0 ? int(cfg.max_forward_tokens) : 1;
        int sampled = cfg.max_forward_requests > 0 ? int(cfg.max_forward_requests) : 1;
        if (sampled > rows) sampled = rows;
        gemma4::Gemma4Geometry pg = g;
        pg.paged_kv_enabled = true;
        const auto dag = gemma4::build_gemma4_dag_mb(pg, 0, /*with_argmax=*/false);
        const gemma4::ScratchPlan sp = gemma4::build_gemma4_scratch(dag, pg);
        const gemma4::ScratchColoring col =
            gemma4::color_gemma4_scratch(dag, sp, /*no_recycle=*/golden_taps_enabled());
        rows = gemma4::gemma4_qmm_pool_rows(rows);
        sampled = gemma4::gemma4_qmm_pool_rows(sampled);
        for (const std::size_t e : g4_pool_elems(dag, pg, col, rows, sampled)) bytes += e * 2;
        bytes += std::size_t(sampled) * std::size_t(g.vocab) * 2;  // the logits slot
    } else if (family == pie::metal::model::ModelFamily::GptOss) {
        gptoss::GptOssGeometry g;
        std::string ignore;
        if (!gptoss_geometry(cfg, g, max_ctx, &ignore)) return bytes;
        bytes += std::size_t(g.n_layers) * 2 * gptoss::gptoss_kv_bytes_per_layer(g, max_ctx, 2);
        // The activation pool, summed from the SAME colouring the engine will
        // build. Pure, so asking costs microseconds and removes the guess.
        int rows = cfg.max_forward_tokens > 0 ? int(cfg.max_forward_tokens) : 1;
        int sampled = cfg.max_forward_requests > 0 ? int(cfg.max_forward_requests) : 1;
        if (sampled > rows) sampled = rows;
        const auto dag = gptoss::build_gptoss_dag(g, /*with_argmax=*/false);
        const gptoss::ScratchPlan sp = gptoss::build_gptoss_scratch(dag, g);
        const gptoss::ScratchColoring col =
            gptoss::color_gptoss_scratch(dag, sp, /*no_recycle=*/golden_taps_enabled());
        for (const std::size_t e : go_pool_elems(dag, g, col, rows, sampled)) bytes += e * 2;
        bytes += std::size_t(sampled) * std::size_t(g.vocab) * 2;  // the logits slot
    }
    return bytes;
}

std::unique_ptr<SimpleFamilyEngine> SimpleFamilyEngine::create(
    pie::metal::model::ModelFamily family, RawMetalContext& ctx, const std::string& kernels_dir,
    const SetupConfig& cfg, const pie_loader::LoadPlan& load_plan, int max_ctx,
    std::string* err) {
    if (family == pie::metal::model::ModelFamily::Gemma4) {
        auto e = std::make_unique<Gemma4Engine>();
        if (!e->init(ctx, kernels_dir, cfg, load_plan, max_ctx, err)) return nullptr;
        return e;
    }
    if (family == pie::metal::model::ModelFamily::GptOss) {
        auto e = std::make_unique<GptOssEngine>();
        if (!e->init(ctx, kernels_dir, cfg, load_plan, max_ctx, err)) return nullptr;
        return e;
    }
    if (err) *err = "SimpleFamilyEngine: not a family with no recurrent state";
    return nullptr;
}

}  // namespace pie::metal::batch

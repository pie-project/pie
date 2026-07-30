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

/// Only a colour's FINAL writer is named: the in-place kinds share a buffer with
/// the tap before them, and publishing the earlier tensor under the earlier name
/// reads as a divergence that is really the dump lying.
void dump_g4_taps(const std::vector<gemma4::Dispatch>& dag, const Gemma4Geometry& g,
                  const gemma4::ScratchColoring& col, const std::vector<SlotHandle>& pool,
                  int rows) {
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
        const int c = colour_of(di, t.out_bind);
        if (c < 0 || c >= n_pool || !pool[std::size_t(c)].valid()) continue;
        if (last[std::size_t(c)] != int(di)) continue;
        const std::string name = dag[di].layer < 0
            ? std::string(t.name)
            : std::to_string(dag[di].layer) + "." + t.name;
        dump_golden_bf16(name, pool[std::size_t(c)].contents(), rows, t.width, t.width);
    }
}

// ── gemma4 ──────────────────────────────────────────────────────────────────

class Gemma4Engine final : public SimpleFamilyEngine {
  public:
    bool init(RawMetalContext& ctx, const std::string& kernels_dir, const SetupConfig& cfg,
              const pie_loader::LoadPlan& load_plan, int max_ctx, std::string* err) {
        if (!gemma4_geometry(cfg, g_, max_ctx, err)) return false;
        max_ctx_ = max_ctx;

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

        // KV, sized per OWNING layer: the shared tail attends pages an earlier
        // layer wrote, and the two attention types are different widths.
        b_.kv.resize(std::size_t(g_.n_layers));
        for (int L = 0; L < g_.n_layers; ++L) {
            if (g_.is_kv_shared(L)) continue;
            const std::size_t bytes = gemma4::gemma4_kv_bytes_per_layer(g_, L, max_ctx_, 2);
            b_.kv[std::size_t(L)].k = ctx.heap_alloc(bytes);
            b_.kv[std::size_t(L)].v = ctx.heap_alloc(bytes);
            if (!b_.kv[std::size_t(L)].k.valid() || !b_.kv[std::size_t(L)].v.valid()) {
                if (err) *err = "gemma4 KV allocation failed";
                return false;
            }
        }

        dag_ = gemma4::build_gemma4_dag(g_, /*with_argmax=*/false);
        const gemma4::ScratchPlan sp = gemma4::build_gemma4_scratch(dag_, g_);
        // Under a tap dump every value needs its own buffer, or a later
        // dispatch overwrites the one being read.
        coloring_ = gemma4::color_gemma4_scratch(dag_, sp, /*no_recycle=*/golden_taps_enabled());
        if (!coloring_.hazard_free) {
            if (err) *err = "gemma4's activation colouring is not hazard-free";
            return false;
        }
        b_.pool.resize(std::size_t(coloring_.colors_used));
        const std::size_t widest = std::size_t(g_.vocab) * 2;
        for (int c = 0; c < coloring_.colors_used; ++c) b_.pool[std::size_t(c)] = ctx.heap_alloc(widest);

        b_.io.resize(kIoSlotCount);
        for (int i = 0; i < kIoSlotCount; ++i) b_.io[i] = ctx.heap_alloc(4096);
        // The logits leave the pool: the sampler reads a slot of its own, so the
        // tail writes there and nothing copies afterwards.
        logits_ = ctx.heap_alloc(std::size_t(g_.vocab) * 2);
        if (!logits_.valid()) {
            if (err) *err = "gemma4 logits allocation failed";
            return false;
        }

        if (!gemma4::build_gemma4_psos(ctx, kernels_dir, psos_, err)) return false;
        if (!load_decode_psos(ctx, kernels_dir, base_, /*with_argmax=*/false, err)) return false;

        gemma4::bind_gemma4_consts(ctx, dag_, g_);
        try {
            gemma4::bind_gemma4_dag(ctx, b_, dag_, g_, coloring_);
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
        return true;
    }

    int vocab() const override { return g_.vocab; }
    int n_layers() const override { return g_.n_layers; }

    void reset() override {
        for (auto& kv : b_.kv) {
            if (kv.k.valid() && kv.k.contents()) std::memset(kv.k.contents(), 0, kv.k.size);
            if (kv.v.valid() && kv.v.contents()) std::memset(kv.v.contents(), 0, kv.v.size);
        }
    }

    StepTiming step(RawMetalContext& ctx, std::uint32_t token_id,
                    std::uint32_t position) override {
        write_i32(b_.io[int(IoSlot::TokenId)], std::int32_t(token_id));
        write_i32(b_.io[int(IoSlot::Position)], std::int32_t(position));
        write_i32(b_.io[int(IoSlot::SeqLen)], std::int32_t(position) + 1);
        return ctx.run_step([&](StepEncoder& se) {
            gemma4::encode_gemma4_step(se, dag_, g_, base_, psos_);
        });
    }

    SlotHandle logits_slot() const override { return logits_; }

    void dump_taps(int rows) const override {
        dump_g4_taps(dag_, g_, coloring_, b_.pool, rows);
    }

  private:
    gemma4::Gemma4Geometry g_{};
    int max_ctx_ = 0;
    std::vector<gemma4::Dispatch> dag_{};
    gemma4::ScratchColoring coloring_{};
    gemma4::BoundGemma4 b_{};
    gemma4::Gemma4Psos psos_{};
    DecodeStepPsos base_{};
    SlotHandle logits_{};
};

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

        b_.kv.resize(std::size_t(g_.n_layers));
        for (int L = 0; L < g_.n_layers; ++L) {
            const std::size_t bytes = gptoss::gptoss_kv_bytes_per_layer(g_, max_ctx_, 2);
            b_.kv[std::size_t(L)].k = ctx.heap_alloc(bytes);
            b_.kv[std::size_t(L)].v = ctx.heap_alloc(bytes);
            if (!b_.kv[std::size_t(L)].k.valid() || !b_.kv[std::size_t(L)].v.valid()) {
                if (err) *err = "gpt-oss KV allocation failed";
                return false;
            }
        }

        dag_ = gptoss::build_gptoss_dag(g_, /*with_argmax=*/false);
        const gptoss::ScratchPlan sp = gptoss::build_gptoss_scratch(dag_, g_);
        coloring_ = gptoss::color_gptoss_scratch(dag_, sp);
        if (!coloring_.hazard_free) {
            if (err) *err = "gpt-oss's activation colouring is not hazard-free";
            return false;
        }
        b_.pool.resize(std::size_t(coloring_.colors_used));
        const std::size_t widest = std::size_t(gptoss::gptoss_widest_elems(g_)) * 2;
        for (int c = 0; c < coloring_.colors_used; ++c) b_.pool[std::size_t(c)] = ctx.heap_alloc(widest);

        b_.io.resize(kIoSlotCount);
        for (int i = 0; i < kIoSlotCount; ++i) b_.io[i] = ctx.heap_alloc(4096);
        logits_ = ctx.heap_alloc(std::size_t(g_.vocab) * 2);
        if (!logits_.valid()) {
            if (err) *err = "gpt-oss logits allocation failed";
            return false;
        }

        if (!gptoss::build_gptoss_psos(ctx, kernels_dir, psos_, err)) return false;
        if (!load_decode_psos(ctx, kernels_dir, base_, /*with_argmax=*/false, err)) return false;

        gptoss::bind_gptoss_consts(ctx, dag_, g_);
        try {
            gptoss::bind_gptoss_dag(ctx, b_, dag_, g_, coloring_);
        } catch (const std::exception& e) {
            if (err) *err = std::string("binding gpt-oss: ") + e.what();
            return false;
        }
        const int tail = int(dag_.size()) - 1;
        ctx.arg_bind_ordinal(dag_[std::size_t(tail)].ordinal, (std::uint8_t)bind::GoQmv::Out,
                             logits_);
        return true;
    }

    int vocab() const override { return g_.vocab; }
    int n_layers() const override { return g_.n_layers; }

    void reset() override {
        for (auto& kv : b_.kv) {
            if (kv.k.valid() && kv.k.contents()) std::memset(kv.k.contents(), 0, kv.k.size);
            if (kv.v.valid() && kv.v.contents()) std::memset(kv.v.contents(), 0, kv.v.size);
        }
    }

    StepTiming step(RawMetalContext& ctx, std::uint32_t token_id,
                    std::uint32_t position) override {
        write_i32(b_.io[int(IoSlot::TokenId)], std::int32_t(token_id));
        write_i32(b_.io[int(IoSlot::Position)], std::int32_t(position));
        write_i32(b_.io[int(IoSlot::SeqLen)], std::int32_t(position) + 1);
        return ctx.run_step([&](StepEncoder& se) {
            gptoss::encode_gptoss_step(se, dag_, g_, base_, psos_);
        });
    }

    SlotHandle logits_slot() const override { return logits_; }

  private:
    gptoss::GptOssGeometry g_{};
    int max_ctx_ = 0;
    std::vector<gptoss::Dispatch> dag_{};
    gptoss::ScratchColoring coloring_{};
    gptoss::BoundGptOss b_{};
    gptoss::GptOssPsos psos_{};
    DecodeStepPsos base_{};
    SlotHandle logits_{};
};

}  // namespace

std::size_t SimpleFamilyEngine::extra_heap_bytes(pie::metal::model::ModelFamily family,
                                                 const SetupConfig& cfg, int max_ctx) {
    // KV + activation pool + logits + constants, with slack. Deliberately
    // generous: this is a budget, and a context that is too small fails at
    // `heap_alloc` with no diagnosis of which allocation ran out.
    std::size_t bytes = std::size_t(256) << 20;
    // Under a tap dump the colouring stops recycling, so the pool is one slot
    // per dispatch rather than a handful. Debug-only; the budget is address
    // space, not resident memory.
    const std::size_t pool_slots = golden_taps_enabled() ? 1024 : 32;
    if (family == pie::metal::model::ModelFamily::Gemma4) {
        gemma4::Gemma4Geometry g;
        std::string ignore;
        if (!gemma4_geometry(cfg, g, max_ctx, &ignore)) return bytes;
        bytes += gemma4::gemma4_kv_region_bytes(g, max_ctx, 2);
        // The pool's widest slot is the vocabulary; the colouring uses a handful.
        bytes += pool_slots * std::size_t(g.vocab) * 2;
    } else if (family == pie::metal::model::ModelFamily::GptOss) {
        gptoss::GptOssGeometry g;
        std::string ignore;
        if (!gptoss_geometry(cfg, g, max_ctx, &ignore)) return bytes;
        bytes += std::size_t(g.n_layers) * 2 * gptoss::gptoss_kv_bytes_per_layer(g, max_ctx, 2);
        bytes += pool_slots * std::size_t(gptoss::gptoss_widest_elems(g)) * 2;
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

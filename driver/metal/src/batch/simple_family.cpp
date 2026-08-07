// See simple_family.hpp.

#include "simple_family.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "decode_abi.hpp"
#include "golden_tap.hpp"
#include "kernels/decode_psos.hpp"
#include "expert_paging.hpp"
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
#include "model/llama/bind.hpp"
#include "model/llama/decode_consts.hpp"
#include "model/llama/decode_step.hpp"
#include "model/llama/encode.hpp"
#include "model/llama/geometry.hpp"
#include "model/llama/kernels.hpp"
#include "model/llama/scratch.hpp"

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
    if (cfg.quant_bits != 0) g.quant.bits = cfg.quant_bits;
    if (cfg.quant_group_size != 0) g.quant.group = cfg.quant_group_size;
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

/// The llama families' geometry, likewise. One reader for `llama`, `mistral`,
/// `qwen2`, `qwen3` and the two MoE variants: they differ in fields this has
/// already been handed, not in how the config is read.
bool llama_geometry(const SetupConfig& cfg, llama::LlamaGeometry& g, int max_ctx,
                    std::string* err) {
    if (!llama::geometry_from_facts(cfg.llama, g, err)) return false;
    if (cfg.vocab_size != 0) g.vocab = static_cast<int>(cfg.vocab_size);
    if (cfg.quant_bits != 0) g.quant.bits = cfg.quant_bits;
    if (cfg.quant_group_size != 0) g.quant.group = cfg.quant_group_size;
    g.kv_max_ctx = max_ctx;
    return true;
}

/// Which bind index carries a dispatch's OUTPUT, under what name, how wide.
///
/// The names are the `tests/parity/*_mlx_taps.py` names, so that the engine's
/// dump and the raw path's diff against the same reference.
///
/// One struct for every family, because the dump is a property of the SCRATCH
/// COLOURING and not of the model: the colouring, the pool and the redirected
/// last dispatch are the same three facts whatever the DAG computes. Each
/// family supplies only what is actually its own -- which kinds are worth
/// naming, and which of them are the head-row tail.
struct Tap {
    const char* name;
    std::uint8_t out_bind;
    int width;
    // Set only for a value the expert sort reordered. The dump then un-permutes
    // it back to slot-major, so a batched mixture publishes the same tensor
    // shape a decode does and no reader has to know the sort happened. `width`
    // stays the PUBLISHED width -- `slots * per-slot` -- as it is for a decode.
    const std::int32_t* perm = nullptr;
    int perm_rows = 0;
    int slots = 0;
};
using G4Tap = Tap;

bool g4_tap_for(const gemma4::Dispatch& d, const Gemma4Geometry& g, G4Tap& out) {
    using K = gemma4::Kind;
    const int L = d.layer;
    const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;
    const int q_dim = g.n_q_heads * hd;
    const int kv_dim = g.n_kv_heads_of(L) * hd;
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
        case K::VNorm:
        // The k-eq-V layers take V from the k projection BEFORE k-norm, so this
        // is the same tensor under the same name -- the oracle publishes one
        // `v_norm` per layer either way.
        case K::VNormFromK:       out = {"v_norm", 1, kv_dim};           return true;
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
        // ── the mixture, in TOKEN order ──
        // The sorted tensors are deliberately untapped: their row order is the
        // driver's own, so a dump of them would diff against nothing.
        case K::DenseBranchNorm:  out = {"g4_dense_br", 2, g.hidden};    return true;
        case K::RouterNorm:       out = {"g4_router_n", 2, g.hidden};    return true;
        case K::RouterGemv:       out = {"g4_router", 4, g.n_experts};   return true;
        case K::MoeNorm:          out = {"g4_moe_n", 2, g.hidden};       return true;
        case K::ExpertCombine:    out = {"g4_moe_out", 2, g.hidden};     return true;
        case K::MoeBranchNorm:    out = {"g4_moe_br", 2, g.hidden};      return true;
        case K::BranchAdd:        out = {"g4_branches", 2, g.hidden};    return true;
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
                                       const gemma4::ScratchPlan& plan, const Gemma4Geometry& g,
                                       const gemma4::ScratchColoring& col, int rows,
                                       int head_rows) {
    const std::vector<gemma4::ValueExtent> ext = gemma4::gemma4_value_extents(dag, plan, g);
    std::vector<std::size_t> elems(std::size_t(col.colors_used), 0);
    // Per VALUE, not per dispatch. Sizing by dispatch takes the widest of a
    // dispatch's buffers and applies it to all of them, which for the mixture
    // means the sorted stack's height lands on every dense tensor sharing the
    // dispatch -- under a golden dump's no-recycle colouring that asked for
    // 20 GB and the heap refused it.
    for (const gemma4::Use& u : plan.uses) {
        if (!u.is_write) continue;
        if (u.value < 0 || std::size_t(u.value) >= col.color_of_value.size()) continue;
        const int c = col.color_of_value[std::size_t(u.value)];
        if (c < 0 || c >= col.colors_used) continue;
        const gemma4::ValueExtent& e = ext[std::size_t(u.value)];
        // The tail's tensors have one row per SAMPLED row; the body's have one
        // per token; the expert stack one per (token, slot) pair, tile-padded.
        const bool tail = gemma4::is_tail(dag[std::size_t(u.index)].kind);
        const int n = e.rows_are_sorted != 0 ? gemma4::gemma4_moe_sorted_rows(g, rows)
                                             : (tail ? head_rows : rows);
        elems[std::size_t(c)] =
            std::max(elems[std::size_t(c)], std::size_t(n) * std::size_t(e.elems));
    }
    for (std::size_t& e : elems) {
        if (e == 0) e = std::size_t(rows) * std::size_t(g.hidden);
    }
    return elems;
}

/// Publish every tapped value in a DAG under `PIE_METAL_GOLDEN_DIR`.
///
/// Only a colour's FINAL writer is named. The in-place kinds (the ropes above
/// all) share a buffer with the tap before them, so publishing the earlier
/// tensor under the earlier name reads as a divergence that is really the dump
/// lying. The LAST dispatch is re-pointed at the engine's own logits slot, so
/// its pool colour is never written and dumping it would publish zeros under
/// the name everything downstream depends on.
///
/// `tap_for` answers which values this family names; `is_tail` answers which of
/// them have `head_rows` rows rather than `rows`, because a family that gathers
/// the sampled positions before its head computes the tail on fewer rows than
/// it computed the body.
template <class Dispatch, class Coloring, class TapFor, class IsTail>
void dump_taps_from(const std::vector<Dispatch>& dag, const Coloring& col,
                    const std::vector<SlotHandle>& pool, const SlotHandle& logits, int rows,
                    int head_rows, TapFor tap_for, IsTail is_tail) {
    const int n_pool = int(pool.size());
    const auto colour_of = [&](std::size_t di, std::uint8_t bind_index) {
        for (const auto& sb : col.per_dispatch[di]) {
            if (sb.bind_index == bind_index) return int(sb.color);
        }
        return -1;
    };
    std::vector<int> last(std::size_t(n_pool < 0 ? 0 : n_pool), -1);
    for (std::size_t di = 0; di < dag.size(); ++di) {
        Tap t{};
        if (!tap_for(dag[di], t)) continue;
        const int c = colour_of(di, t.out_bind);
        if (c >= 0 && c < n_pool) last[std::size_t(c)] = int(di);
    }
    for (std::size_t di = 0; di < dag.size(); ++di) {
        Tap t{};
        if (!tap_for(dag[di], t)) continue;
        const void* src = nullptr;
        if (di + 1 == dag.size()) {
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
        if (t.perm != nullptr) {
            dump_golden_bf16_sorted(name, src, t.perm, t.perm_rows, rows, t.slots,
                                    t.width / t.slots);
            continue;
        }
        dump_golden_bf16(name, src, is_tail(dag[di]) ? head_rows : rows, t.width, t.width);
    }
}

void dump_g4_taps(const std::vector<gemma4::Dispatch>& dag, const Gemma4Geometry& g,
                  const gemma4::ScratchColoring& col, const std::vector<SlotHandle>& pool,
                  const SlotHandle& logits, int rows, int head_rows) {
    dump_taps_from(
        dag, col, pool, logits, rows, head_rows,
        [&](const gemma4::Dispatch& d, Tap& t) { return g4_tap_for(d, g, t); },
        [](const gemma4::Dispatch& d) { return gemma4::is_tail(d.kind); });
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
            auto view = std::make_shared<pie_loader::CheckpointSource>(storage);
            ExpertSlabRequest slab_req;
            if (cfg.expert_slab_bytes > 0 && g_.is_moe() && g_.n_experts > 1) {
                slab_req.n_experts = g_.n_experts;
                slab_req.budget_bytes = cfg.expert_slab_bytes;
            }
            StagedWeights staged = stage_plan_weights(
                ctx, std::move(view), load_plan, storage.memory.persistent_bytes,
                stream_predicate(cfg.stream_routed_experts || slab_req.valid(),
                                 slab_req.valid()),
                slab_req);
            b_.weights = std::move(staged.weights);
            slab_ = std::move(staged.slab);
            // Whatever the weights point into -- the pack, or the checkpoint's
            // own mapping -- must outlive them; see the
            // gpt-oss engine below.
            weight_mapping_ = std::move(staged.weight_mapping);
        } catch (const std::exception& e) {
            if (err) *err = std::string("staging gemma4's weights: ") + e.what();
            return false;
        }

        // The dense FFN's and the router's formats, read off the checkpoint
        // rather than the config: mlx-lm's quantization predicate can single
        // out tensors by NAME, and `config.json` records only the model-wide
        // choice. Asked here because the PSO tables are built below and need
        // the answer.
        //
        // Both groups, separately. They used to be one question answered by
        // `mlp.down_proj` alone, on the assumption that a checkpoint sparing
        // one spares the other -- true of lmstudio-community's QAT build and
        // false of mlx-community's, which spares only the router. The router
        // then ran the 4-bit pipeline over its 8-bit bytes and produced logits
        // at cosine 0.10 to mlx-lm's, with every tensor feeding them at 0.9999.
        {
            const auto view = load_plan.view();
            const auto format_of = [&](const char* suffix) -> AffineFormat {
                for (std::size_t i = 0; i < view.tensors.len; ++i) {
                    const auto& t = view.tensors.ptr[i];
                    const std::string name(reinterpret_cast<const char*>(t.name.ptr),
                                           t.name.len);
                    if (name.find(suffix) == std::string::npos) continue;
                    return AffineFormat{int(t.quant_bits_per_element),
                                        int(t.quant_group_size)};
                }
                return AffineFormat{0, 0};
            };
            const AffineFormat ffn = format_of("mlp.down_proj.weight");
            const AffineFormat router = format_of("router.proj.weight");
            const auto differs = [&](const AffineFormat& f) {
                return f.bits != 0 && f.group != 0 &&
                       (f.bits != g_.quant.bits || f.group != g_.quant.group);
            };
            g_.alt_quant_ffn = differs(ffn);
            g_.alt_quant_router = differs(router);
            if (g_.alt_quant_ffn) g_.ffn_quant = ffn;
            if (g_.alt_quant_router) g_.ffn_quant = router;
            // One alternate format is all there are pipelines for. Two would
            // need a third table, and guessing which of them to build is how a
            // dispatch ends up on a pipeline for someone else's bytes.
            if (g_.alt_quant_ffn && g_.alt_quant_router &&
                (ffn.bits != router.bits || ffn.group != router.group)) {
                if (err) {
                    *err = "gemma4: the dense FFN is " + std::to_string(ffn.bits) +
                           "-bit at group " + std::to_string(ffn.group) + " and the router " +
                           std::to_string(router.bits) + "-bit at group " +
                           std::to_string(router.group) +
                           ", and this driver builds one alternate pipeline table";
                }
                return false;
            }
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
                                      std::size_t(g_.n_kv_heads_of(L)) * std::size_t(hd) * 2;
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
        plan_ = gemma4::build_gemma4_scratch(dag_, g_);
        const gemma4::ScratchPlan& sp = plan_;
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
            g4_pool_elems(dag_, plan_, g_, coloring_, gemma4::gemma4_qmm_pool_rows(max_rows_),
                          gemma4::gemma4_qmm_pool_rows(max_sampled_));
        for (int c = 0; c < coloring_.colors_used; ++c) {
            b_.pool[std::size_t(c)] = ctx.heap_alloc(elems[std::size_t(c)] * 2);
        }

        b_.io.resize(kIoSlotCount);
        const std::size_t io_bytes =
            std::max<std::size_t>(4096, std::size_t(g_.total_pages + max_rows_ + 8) * 4);
        for (int i = 0; i < kIoSlotCount; ++i) b_.io[i] = ctx.heap_alloc(io_bytes);
        // The routed matvec declares a bias it does not read; see
        // `BoundGemma4::zero_bias`. Wide enough for the widest routed output.
        if (g_.is_moe()) {
            b_.zero_bias =
                ctx.heap_alloc(std::size_t(std::max(g_.hidden, g_.moe_intermediate)) * 2);
            if (!b_.zero_bias.valid()) {
                if (err) *err = "gemma4 routed bias allocation failed";
                return false;
            }
            if (b_.zero_bias.contents() != nullptr) {
                std::memset(b_.zero_bias.contents(), 0, b_.zero_bias.size);
            }
        }
        // The logits leave the pool: the sampler reads a slot of its own, so the
        // tail writes there and nothing copies afterwards. One row per SAMPLED
        // row, which is what the tail produces.
        logits_ = ctx.heap_alloc(std::size_t(gemma4::gemma4_qmm_pool_rows(max_sampled_)) *
                                 std::size_t(g_.vocab) * 2);
        if (!logits_.valid()) {
            if (err) *err = "gemma4 logits allocation failed";
            return false;
        }

        if (!gemma4::build_gemma4_psos(ctx, kernels_dir, g_, psos_, err)) return false;
        if (!load_decode_psos(ctx, kernels_dir, base_, g_.quant, err)) {
            return false;
        }
        if (!load_multibatch_psos(
                ctx, kernels_dir, mb_, g_.quant, err,
                MultiBatchPsoFeatures{
                    .d512 = true, .sdpa_d256 = true,
                    .fp16_precast = !g_.is_moe() && g_.quant.bits == 4 &&
                                    g_.quant.group == 64})) {
            return false;
        }
        // A checkpoint may quantize the dense FFN and the router at a DIFFERENT
        // width from everything else -- gemma-4-26B is 4-bit g64 everywhere but
        // `mlp.{gate,up,down}_proj` and `router.proj`, which are 8-bit. That is
        // not new kernels, it is a second instance of the same two tables, and
        // `gemma4_uses_alt_quant` picks between them per dispatch. One table for
        // both would run a 4-bit kernel over 8-bit bytes: fast, and wrong.
        if (g_.has_alt_quant()) {
            if (!load_decode_psos(ctx, kernels_dir, base_alt_, g_.ffn_quant, err)) {
                return false;
            }
            if (!load_multibatch_psos(
                    ctx, kernels_dir, mb_alt_, g_.ffn_quant, err,
                    MultiBatchPsoFeatures{.d512 = true, .sdpa_d256 = true})) {
                return false;
            }
        }

        if (!g_.is_moe() && g_.quant.bits == 4 && g_.quant.group == 64) {
            // The widest input any staged projection reads, times the rows the
            // GEMM rounds up to. Asked of the DAG rather than of the geometry
            // because gemma4's `o_proj` K is per-layer -- a sliding layer's is
            // 8x256 where a full layer's is 8x512, and only one of those is
            // `hidden`.
            std::size_t widest = 0;
            for (const gemma4::Dispatch& d : dag_) {
                if (!gemma4::gemma4_fp16_qmm(g_, d, max_rows_)) continue;
                widest = std::max(widest, std::size_t(gemma4::qmv_kn(d.kind, g_, d.layer).K));
            }
            if (widest > 0) {
                const std::size_t elems =
                    std::size_t(gemma4::gemma4_qmm_pool_rows(max_rows_)) * widest;
                fp16_input_ = ctx.heap_alloc(elems * sizeof(std::uint16_t));
                if (!fp16_input_.valid()) {
                    if (err) *err = "gemma4 FP16 QMM input allocation failed";
                    return false;
                }
            }
        }
        gemma4::bind_gemma4_consts(ctx, dag_, g_, /*rows=*/1, /*paged=*/true, /*head_rows=*/1);
        gemma4::bind_gemma4_fp16_qmm(ctx, dag_, g_, /*rows=*/1, /*head_rows=*/1,
                                     fp16_input_, fp16_keep_);
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

        if (slab_ && !plan_segments(err)) return false;

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
    WeightBytes weight_bytes() const override {
        // The mixture members have a bank a token reads a fraction of; the
        // dense ones pass zeroes and the rule below discounts nothing.
        return pie::metal::weight_bytes(b_.weights, g_.is_moe() ? g_.n_experts : 0,
                                        g_.is_moe() ? g_.experts_per_token : 0);
    }

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
            // The staged element count is a row count, so it moves with the
            // fire. Rebound here and not in the encoder: the encoder writes a
            // command buffer, and this writes an argument table.
            gemma4::bind_gemma4_fp16_qmm(ctx, dag_, g_, rows, head_rows, fp16_input_,
                                         fp16_keep_);
            bound_rows_ = rows;
            bound_head_rows_ = head_rows;
        }
        // Rows alone cannot tell a prefill from a fleet of decodes -- both can
        // be 32 rows. The CSR can: `qo_indptr` is one entry per request plus a
        // terminator.
        const int requests =
            csr.qo_indptr.empty() ? 0 : int(csr.qo_indptr.size()) - 1;
        const auto walk = [this, rows, head_rows, requests, &pre, &post](StepEncoder& se,
                                                                        std::size_t begin,
                                                                        std::size_t end) {
            if (begin == 0 && pre) pre(se);
            gemma4::encode_gemma4_step_mb(se, dag_, g_, rows, base_, mb_, psos_,
                                          /*ordinal_base=*/0, head_rows,
                                          g_.has_alt_quant() ? &base_alt_ : nullptr,
                                          g_.has_alt_quant() ? &mb_alt_ : nullptr,
                                          requests, begin, end);
            if (end == dag_.size() && post) post(se);
        };
        if (paging_.active()) return paging_.fire(ctx, rows, walk);
        return ctx.run_step([&](StepEncoder& se) { walk(se, 0, dag_.size()); });
    }

    SlotHandle logits_slot() const override { return logits_; }

    void dump_taps(int rows) const override {
        dump_g4_taps(dag_, g_, coloring_, b_.pool, logits_, rows, bound_head_rows_);
    }

  private:
    static constexpr int kPageSize = 32;

    /// Resolve the pool slot a scratch VALUE was coloured onto.
    SlotHandle pool_slot_of(int value) const {
        SlotHandle out{};
        for (const gemma4::Use& u : plan_.uses) {
            if (!u.is_write || u.value != value) continue;
            if (u.index < 0 || std::size_t(u.index) >= coloring_.per_dispatch.size()) continue;
            for (const auto& sb : coloring_.per_dispatch[std::size_t(u.index)]) {
                if (sb.bind_index != u.bind_index) continue;
                if (sb.color < 0 || std::size_t(sb.color) >= b_.pool.size()) continue;
                out = b_.pool[std::size_t(sb.color)];
            }
        }
        return out;
    }

    /// One cut per mixture layer, immediately after its router: the first point
    /// at which the chosen experts exist and the last before anything reads the
    /// bank. gemma 4 puts a DENSE branch beside the routed one, so a layer's
    /// cut sits in the middle of its body rather than at a layer boundary --
    /// which costs nothing here, because a cut is only where the command buffer
    /// ends, and the dense branch is dispatched either side of it unchanged.
    bool plan_segments(std::string* err) {
        const std::vector<int> run_ends = gemma4::gemma4_run_ends(dag_);
        std::vector<ExpertPaging::Cut> cuts;
        std::size_t nth = 0;
        for (std::size_t i = 0; i < dag_.size(); ++i) {
            if (dag_[i].kind != gemma4::Kind::RouterTopK) continue;
            ExpertPaging::Cut c;
            c.end = std::size_t(run_ends[i]) + 1;
            if (nth < plan_.expert_ids_by_layer.size()) {
                c.ids = pool_slot_of(plan_.expert_ids_by_layer[nth]);
            }
            ++nth;
            cuts.push_back(c);
        }
        return paging_.plan(slab_, std::move(cuts), dag_.size(), g_.n_experts,
                            g_.experts_per_token, max_rows_, "gemma4", err);
    }

    gemma4::Gemma4Geometry g_{};
    /// The routed experts' paging cache, when a budget asked for one.
    std::shared_ptr<ExpertSlab> slab_{};
    ExpertPaging paging_{};
    /// Keeps the streamed weights' mapping alive; see `init`.
    std::shared_ptr<void> weight_mapping_{};
    int max_ctx_ = 0;
    int max_rows_ = 1;
    int max_sampled_ = 1;
    int bound_rows_ = 0;
    int bound_head_rows_ = 0;
    std::vector<gemma4::Dispatch> dag_{};
    gemma4::ScratchPlan plan_{};
    gemma4::ScratchColoring coloring_{};
    gemma4::BoundGemma4 b_{};
    gemma4::Gemma4Psos psos_{};
    DecodeStepPsos base_{};
    MultiBatchPsos mb_{};
    /// The same two tables at `g_.ffn_quant`, built only when the checkpoint
    /// has a second affine format.
    DecodeStepPsos base_alt_{};
    MultiBatchPsos mb_alt_{};
    // The FP16 staging buffer every dense projection's GEMM reads, and the
    // element-count buffers the staging pass bounds itself with.
    SlotHandle fp16_input_{};
    std::vector<SlotHandle> fp16_keep_{};
    std::vector<SlotHandle> kpages_{};
    std::vector<SlotHandle> vpages_{};
    SlotHandle logits_{};
};

/// Which bind index carries each gpt-oss kind's OUTPUT, and how wide it is.
/// The names are `tests/parity/gptoss_mlx_taps.py`'s, so the engine's dump and
/// the raw path's diff against the same reference.
using GoTap = Tap;

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

void dump_go_taps(const std::vector<gptoss::Dispatch>& dag, const GptOssGeometry& g,
                  const gptoss::ScratchColoring& col, const std::vector<SlotHandle>& pool,
                  const SlotHandle& logits, int rows, int head_rows) {
    std::vector<const std::int32_t*> perm_of(std::size_t(g.n_layers), nullptr);
    for (std::size_t di = 0; di < dag.size() && di < col.per_dispatch.size(); ++di) {
        const gptoss::Dispatch& d = dag[di];
        if (d.kind != gptoss::Kind::ExpertSort || d.layer < 0) continue;
        for (const auto& sb : col.per_dispatch[di]) {
            if (sb.bind_index != (std::uint8_t)bind::MoeRouteSort::Perm) continue;
            if (sb.color < 0 || std::size_t(sb.color) >= pool.size()) continue;
            perm_of[std::size_t(d.layer)] =
                static_cast<const std::int32_t*>(pool[std::size_t(sb.color)].contents());
        }
    }
    dump_taps_from(
        dag, col, pool, logits, rows, head_rows,
        [&](const gptoss::Dispatch& d, Tap& t) {
            if (!go_tap_for(d, g, t)) return false;
            if (gptoss::is_expert_sorted(d.kind) && d.layer >= 0 &&
                std::size_t(d.layer) < perm_of.size() &&
                perm_of[std::size_t(d.layer)] != nullptr) {
                t.perm = perm_of[std::size_t(d.layer)];
                t.perm_rows = gptoss::gptoss_moe_sorted_rows(g, rows);
                t.slots = g.experts_per_token;
            }
            return true;
        },
        [](const gptoss::Dispatch& d) { return gptoss::is_tail(d.kind); });
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
        case K::ExpertSort:               return 2 * g.experts_per_token;
        case K::ExpertGather:             return g.hidden;
        // The exact sorted-row extent is applied in `go_pool_elems`; these are
        // the per-row widths used for the M=1 floor.
        case K::ExpertGate: case K::ExpertUp: return std::max(g.hidden, stack);
        case K::ExpertSwiGlu:             return stack;
        case K::ExpertDown:               return std::max(stack, down);
        case K::ExpertCombine:            return std::max(down, g.hidden);
        case K::LmHead:                   return std::max(g.hidden, g.vocab);
        // Everything else writes the residual stream. Enumerated rather than
        // defaulted, because `-Werror=switch` on this file is what stops a new
        // kind from silently getting a pool slot too small for it -- and a slot
        // too small is the quietest bug in this driver: the write lands in the
        // next colour's buffer.
        case K::EmbedGather:
        case K::AttnNorm:
        case K::FfnNorm:
        case K::AttnResidual:
        case K::FfnResidual:
        case K::RowGather:
        case K::FinalRms:
        case K::Argmax:                   return g.hidden;
    }
    return g.hidden;
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
        const bool tail = gptoss::is_tail(kind);
        const std::size_t sorted =
            std::size_t(gptoss::gptoss_moe_sorted_rows(g, rows));
        std::size_t need = std::size_t(tail ? head_rows : rows) *
                           std::size_t(go_kind_width(kind, g));
        switch (kind) {
            case K::ExpertSort:
                need = 2 * std::max(sorted,
                                    std::size_t(rows * g.experts_per_token));
                break;
            case K::ExpertGather:
                need = sorted * std::size_t(g.hidden);
                break;
            case K::ExpertGate:
            case K::ExpertUp:
                need = sorted *
                       std::size_t(std::max(g.hidden, g.intermediate));
                break;
            case K::ExpertSwiGlu:
                need = sorted * std::size_t(g.intermediate);
                break;
            case K::ExpertDown:
                need = sorted *
                       std::size_t(std::max(g.intermediate, g.hidden));
                break;
            case K::ExpertCombine:
                need = std::max(sorted * std::size_t(g.hidden),
                                std::size_t(rows) * std::size_t(g.hidden));
                break;
            default:
                break;
        }
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
            auto view = std::make_shared<pie_loader::CheckpointSource>(storage);
            ExpertSlabRequest slab_req;
            if (cfg.expert_slab_bytes > 0 && g_.n_experts > 1) {
                slab_req.n_experts = g_.n_experts;
                slab_req.budget_bytes = cfg.expert_slab_bytes;
            }
            StagedWeights staged = stage_plan_weights(
                ctx, std::move(view), load_plan, storage.memory.persistent_bytes,
                stream_predicate(cfg.stream_routed_experts || slab_req.valid(), slab_req.valid()),
                slab_req);
            b_.weights = std::move(staged.weights);
            // The pack must outlive the weights that point into it. Taking only
            // `staged.weights` and letting `staged` die unmaps it under them,
            // which reads as weights of exactly zero.
            weight_mapping_ = std::move(staged.weight_mapping);
            slab_ = std::move(staged.slab);
        } catch (const std::exception& e) {
            if (err) *err = std::string("staging gpt-oss's weights: ") + e.what();
            return false;
        }

        // The router's width is a property of the checkpoint, not of the config,
        // so it is read off the tensors that were just staged. Refused rather
        // than defaulted: the 4- and 8-bit matvecs read incompatible packings,
        // and either over the other's bytes routes to the wrong experts, which
        // survives as fluent wrong text instead of as an error.
        g_.mxfp4_experts = gptoss::mxfp4_experts_from_weights(b_.weights);
        g_.router_bits = gptoss::router_bits_from_weights(b_.weights);
        if (g_.router_bits == 0) {
            if (err) {
                *err = "gpt-oss: could not solve the router's quantization width from "
                       "`layers.0.mlp.router.{weight,scales}`";
            }
            return false;
        }
        // And the projections', which is a third width again -- refused on the
        // same grounds, because reading 8-bit rows with a 4-bit matvec is the
        // failure that produces fluent noise rather than an error.
        g_.proj_bits = gptoss::proj_bits_from_weights(b_.weights);
        if (g_.proj_bits == 0) {
            if (err) {
                *err = "gpt-oss: could not solve the projections' quantization width from "
                       "`layers.0.self_attn.q_proj.{weight,scales}`";
            }
            return false;
        }

        // Paged KV. The sorted MoE is a true M>1 path: rows are grouped by
        // expert, then the native MXFP4 routed GEMM serves each run.
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
        plan_ = gptoss::build_gptoss_scratch(dag_, g_);
        const gptoss::ScratchPlan& sp = plan_;
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
        // The dense projections pad their row count to a whole GEMM tile, so
        // the pool has to hold the padded count -- the padding rows are written.
        const std::vector<std::size_t> elems =
            go_pool_elems(dag_, g_, coloring_, gptoss::gptoss_qmm_pool_rows(max_rows_),
                          gptoss::gptoss_qmm_pool_rows(max_sampled_));
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
        logits_ = ctx.heap_alloc(std::size_t(gptoss::gptoss_qmm_pool_rows(max_sampled_)) *
                                 std::size_t(g_.vocab) * 2);
        if (!logits_.valid()) {
            if (err) *err = "gpt-oss logits allocation failed";
            return false;
        }

        if (!gptoss::build_gptoss_psos(ctx, kernels_dir, g_, psos_, err)) return false;
        // The checkpoint's projection width at g64, NOT the (mxfp4, 32) pair
        // gpt-oss's config declares globally.
        // That global is overridden back to affine g64 by nearly every tensor,
        // and this shared table only ever compiles the affine entrypoints --
        // gpt-oss's own mxfp4 kernels are named explicitly in `gptoss/kernels.cpp`.
        // The width used to be a literal 4 here, which was right for a
        // uniformly-4-bit checkpoint and silently wrong for a mixed one --
        // and this table builds the PREFILL GEMMs, so it was the more
        // damaging of the two hardcodings.
        const AffineFormat kGptOssBase{/*bits=*/g_.proj_bits, /*group=*/64};
        if (!load_decode_psos(ctx, kernels_dir, base_, kGptOssBase, err))
            return false;
        if (!load_multibatch_psos(
                ctx, kernels_dir, mb_, kGptOssBase, err,
                MultiBatchPsoFeatures{.bias = true}))
            return false;

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
        // After binding: the cuts name pool slots, and the pool is not bound
        // until here.
        if (slab_ && !plan_segments(err)) return false;
        return true;
    }

    int vocab() const override { return g_.vocab; }
    int n_layers() const override { return g_.n_layers; }
    WeightBytes weight_bytes() const override {
        return pie::metal::weight_bytes(b_.weights, g_.n_experts, g_.experts_per_token);
    }

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
    /// A fire of R rows is one wider pass. The routed bank sorts its
    /// (row, expert) pairs before the GEMM, while paging keeps each request's
    /// KV history independent.
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
        // Rows alone cannot tell a prefill from a fleet of decodes -- both can
        // be 32 rows. The CSR can: `qo_indptr` is one entry per request plus a
        // terminator.
        const int requests =
            csr.qo_indptr.empty() ? 0 : int(csr.qo_indptr.size()) - 1;
        const auto walk = [this, rows, head_rows, requests, &pre, &post](StepEncoder& se,
                                                                        std::size_t begin,
                                                                        std::size_t end) {
            if (begin == 0 && pre) pre(se);
            gptoss::encode_gptoss_step_mb(se, dag_, g_, rows, base_, mb_, psos_,
                                          /*ordinal_base=*/0, head_rows, requests, begin,
                                          end);
            if (end == dag_.size() && post) post(se);
        };
        if (paging_.active()) return paging_.fire(ctx, rows, walk);
        return ctx.run_step([&](StepEncoder& se) { walk(se, 0, dag_.size()); });
    }

    SlotHandle logits_slot() const override { return logits_; }

    void dump_taps(int rows) const override {
        dump_go_taps(dag_, g_, coloring_, b_.pool, logits_, rows, bound_head_rows_);
    }

  private:
    /// Resolve the pool slot a scratch VALUE was coloured onto.
    SlotHandle pool_slot_of(int value) const {
        SlotHandle out{};
        for (const gptoss::Use& u : plan_.uses) {
            if (!u.is_write || u.value != value) continue;
            if (u.index < 0 || std::size_t(u.index) >= coloring_.per_dispatch.size()) continue;
            for (const auto& sb : coloring_.per_dispatch[std::size_t(u.index)]) {
                if (sb.bind_index != u.bind_index) continue;
                if (sb.color < 0 || std::size_t(sb.color) >= b_.pool.size()) continue;
                out = b_.pool[std::size_t(sb.color)];
            }
        }
        return out;
    }

    bool plan_segments(std::string* err) {
        const std::vector<int> run_ends = gptoss::gptoss_run_ends(dag_);
        std::vector<ExpertPaging::Cut> cuts;
        std::size_t nth = 0;
        for (std::size_t i = 0; i < dag_.size(); ++i) {
            if (dag_[i].kind != gptoss::Kind::RouterTopK) continue;
            ExpertPaging::Cut c;
            c.end = std::size_t(run_ends[i]) + 1;
            if (nth < plan_.expert_ids_by_layer.size()) {
                c.ids = pool_slot_of(plan_.expert_ids_by_layer[nth]);
            }
            ++nth;
            cuts.push_back(c);
        }
        return paging_.plan(slab_, std::move(cuts), dag_.size(), g_.n_experts,
                            g_.experts_per_token, max_rows_, "gpt-oss", err);
    }

    gptoss::GptOssGeometry g_{};
    int max_ctx_ = 0;
    std::vector<gptoss::Dispatch> dag_{};
    gptoss::ScratchPlan plan_{};
    gptoss::ScratchColoring coloring_{};
    /// The routed experts' paging cache, when a budget asked for one.
    std::shared_ptr<ExpertSlab> slab_{};
    ExpertPaging paging_{};
    gptoss::BoundGptOss b_{};
    gptoss::GptOssPsos psos_{};
    DecodeStepPsos base_{};
    MultiBatchPsos mb_{};
    std::vector<SlotHandle> kpages_{};
    std::vector<SlotHandle> vpages_{};
    /// Keeps the streamed weights' mapping alive; see `init`.
    std::shared_ptr<void> weight_mapping_{};
    int max_sampled_ = 1;
    int max_rows_ = 1;
    int bound_rows_ = 0;
    int bound_head_rows_ = 0;
    SlotHandle logits_{};
};

/// Which values a llama layer publishes, under `tests/parity`'s names.
///
/// The bind index is NOT here: `ScratchPlan` already records which of a
/// dispatch's buffers it writes, so asking the plan is both shorter and
/// unfalsifiable, where a hand-written index that drifted would silently
/// publish an INPUT under the output's name -- a divergence the reader would
/// chase in the kernel.
bool ll_tap_for(const llama::Dispatch& d, const llama::LlamaGeometry& g, Tap& out) {
    using K = llama::Kind;
    const int slots = g.experts_per_token > 0 ? g.experts_per_token : 1;
    switch (d.kind) {
        case K::EmbedGather:   out = {"embed",       0, g.hidden};                 return true;
        case K::AttnNorm:      out = {"attn_norm",   0, g.hidden};                 return true;
        case K::QmvQ:          out = {"q_proj",      0, g.q_width()};                return true;
        case K::QmvK:          out = {"k_proj",      0, g.kv_width()};               return true;
        case K::QmvV:          out = {"v_proj",      0, g.kv_width()};               return true;
        case K::QNorm:         out = {"q_norm",      0, g.q_width()};                return true;
        case K::KNorm:         out = {"k_norm",      0, g.kv_width()};               return true;
        case K::RopeQ:         out = {"rope_q",      0, g.q_width()};                return true;
        case K::RopeK:         out = {"rope_k",      0, g.kv_width()};               return true;
        case K::Sdpa:          out = {"sdpa",        0, g.q_width()};                return true;
        case K::QmvO:          out = {"o_proj",      0, g.hidden};                 return true;
        case K::AttnResidual:  out = {"attn_out",    0, g.hidden};                 return true;
        case K::FfnNorm:       out = {"ffn_norm",    0, g.hidden};                 return true;
        case K::QmvGate:       out = {"gate_proj",   0, g.intermediate};           return true;
        case K::QmvUp:         out = {"up_proj",     0, g.intermediate};           return true;
        case K::SiluMul:       out = {"ffn_act",     0, g.intermediate};           return true;
        case K::QmvDown:       out = {"ffn_out",     0, g.hidden};                 return true;
        case K::Router:        out = {"router",      0, g.n_experts};              return true;
        case K::ExpertGate:    out = {"expert_gate", 0, slots * g.moe_intermediate}; return true;
        case K::ExpertUp:      out = {"expert_up",   0, slots * g.moe_intermediate}; return true;
        case K::ExpertSiluMul: out = {"expert_act",  0, slots * g.moe_intermediate}; return true;
        case K::ExpertDown:    out = {"expert_out",  0, slots * g.hidden};         return true;
        case K::ExpertCombine: out = {"moe",         0, g.hidden};                 return true;
        case K::FfnResidual:   out = {"layer_out",   0, g.hidden};                 return true;
        case K::FinalRms:      out = {"final_norm",  0, g.hidden};                 return true;
        case K::LmHead:        out = {"logits",      0, g.vocab};                  return true;
        // `RouterTopK` writes two integer buffers, not an activation, and
        // `KvAppend` writes the cache rather than the pool. Neither is a tensor
        // the reference has a counterpart for.
        default: return false;
    }
}

void dump_ll_taps(const std::vector<llama::Dispatch>& dag, const llama::LlamaGeometry& g,
                  const llama::ScratchPlan& plan, const model::ScratchColoring& col,
                  const std::vector<SlotHandle>& pool, const SlotHandle& logits, int rows,
                  int head_rows) {
    // Dispatch position -> the bind index it WRITES, straight from the plan.
    std::vector<int> writes(dag.size(), -1);
    for (const llama::Use& u : plan.uses) {
        if (u.is_write && u.index >= 0 && std::size_t(u.index) < writes.size()) {
            writes[std::size_t(u.index)] = int(u.bind_index);
        }
    }
    // The sort's permutation, per layer, so the routed intermediates can be
    // published in the layout they had before it ran. Read from the pool
    // because the within-expert order is the GPU's to decide.
    std::vector<const std::int32_t*> perm_of(std::size_t(g.n_layers), nullptr);
    for (const llama::Use& u : plan.uses) {
        if (!u.is_write || u.bind_index != llama::kMoeSortPermBind) continue;
        if (u.index < 0 || std::size_t(u.index) >= dag.size()) continue;
        const llama::Dispatch& d = dag[std::size_t(u.index)];
        if (d.kind != llama::Kind::ExpertSort || d.layer < 0) continue;
        if (std::size_t(d.layer) >= perm_of.size()) continue;
        for (const auto& sb : col.per_dispatch[std::size_t(u.index)]) {
            if (sb.bind_index != llama::kMoeSortPermBind) continue;
            if (sb.color < 0 || std::size_t(sb.color) >= pool.size()) continue;
            perm_of[std::size_t(d.layer)] =
                static_cast<const std::int32_t*>(pool[std::size_t(sb.color)].contents());
        }
    }
    dump_taps_from(
        dag, col, pool, logits, rows, head_rows,
        [&](const llama::Dispatch& d, Tap& t) {
            // `dump_taps_from` walks the DAG in order twice, so recovering the
            // position from the address is exact and avoids widening its
            // signature for one family.
            const std::size_t di = std::size_t(&d - dag.data());
            if (!ll_tap_for(d, g, t)) return false;
            if (writes[di] < 0) return false;
            t.out_bind = std::uint8_t(writes[di]);
            if (llama::is_expert_sorted(d.kind) && d.layer >= 0 &&
                std::size_t(d.layer) < perm_of.size() && perm_of[std::size_t(d.layer)] != nullptr) {
                t.perm = perm_of[std::size_t(d.layer)];
                t.perm_rows = llama::llama_moe_sorted_rows(g, rows);
                t.slots = g.experts_per_token;
            }
            return true;
        },
        [](const llama::Dispatch& d) { return llama::is_tail(d.kind); });
}

// ── the llama-shaped families ───────────────────────────────────────────────

/// Llama, Mistral, Qwen2/3 and Qwen3-MoE.
///
/// PAGED and BATCHED. Several sequences, each attending its own page list, and
/// a fire of R rows encoded as ONE pass rather than R.
///
/// That last part is where this family differs from gpt-oss, whose fire of R
/// rows is R passes because its FFN is always a mixture and a mixture picks its
/// weights per row. A llama is usually dense, so its projections are ordinary
/// matrices and a batch of rows is a GEMM -- which is the whole point, because
/// the matvec re-reads the entire weight for every row it computes. A routed
/// llama keeps the per-row matvec for its experts and gets the GEMM everywhere
/// else.
///
/// This engine is short compared to gpt-oss's and gemma4's for one reason: the
/// family has a single `launch_shape` and a single binder, both taking a row
/// count, rather than a decode copy and a prefill copy of each. There is no
/// `encode_llama_step_mb` to call because `encode_llama_step` already takes the
/// rows.
class LlamaEngine final : public SimpleFamilyEngine {
  public:
    bool init(RawMetalContext& ctx, const std::string& kernels_dir, const SetupConfig& cfg,
              const pie_loader::LoadPlan& load_plan, int max_ctx, std::string* err) {
        if (!llama_geometry(cfg, g_, max_ctx, err)) return false;
        // The contract already decided this, and a config can disagree with the
        // tensors it ships in either direction; see `plan_ties_embeddings`.
        g_.tied_embeddings = plan_ties_embeddings(load_plan);
        max_ctx_ = max_ctx;
        g_.paged_kv_enabled = true;
        g_.kv_page_size = cfg.kv_page_size > 0 ? int(cfg.kv_page_size) : 32;
        {
            const int ps = g_.kv_page_size;
            g_.kv_max_ctx = ((max_ctx_ + ps - 1) / ps) * ps;
            g_.total_pages = g_.kv_max_ctx / ps;
        }
        max_rows_ = cfg.max_forward_tokens > 0 ? int(cfg.max_forward_tokens) : 1;
        max_sampled_ = cfg.max_forward_requests > 0 ? int(cfg.max_forward_requests) : 1;
        if (max_sampled_ > max_rows_) max_sampled_ = max_rows_;

        try {
            const auto storage = load_plan.view();
            auto view = std::make_shared<pie_loader::CheckpointSource>(storage);
            ExpertSlabRequest slab_req;
            if (cfg.expert_slab_bytes > 0 && g_.n_experts > 1) {
                slab_req.n_experts = g_.n_experts;
                slab_req.budget_bytes = cfg.expert_slab_bytes;
            }
            StagedWeights staged = stage_plan_weights(
                ctx, std::move(view), load_plan, storage.memory.persistent_bytes,
                stream_predicate(cfg.stream_routed_experts || slab_req.valid(), slab_req.valid()),
                slab_req);
            b_.weights = std::move(staged.weights);
            // The pack must outlive the weights that point into it; see the
            // gpt-oss engine above.
            weight_mapping_ = std::move(staged.weight_mapping);
            slab_ = std::move(staged.slab);
        } catch (const std::exception& e) {
            if (err) *err = std::string("staging llama's weights: ") + e.what();
            return false;
        }

        // Paged KV per layer. Uniform layers, so one size serves the stack --
        // there is no shared tail and no second head width.
        b_.kv.resize(std::size_t(g_.n_layers));
        kv_.resize(std::size_t(g_.n_layers));
        for (int L = 0; L < g_.n_layers; ++L) {
            const std::size_t bytes = std::size_t(g_.total_pages) *
                                      std::size_t(g_.kv_page_size) *
                                      std::size_t(g_.n_kv_heads) * std::size_t(g_.head_dim) * 2;
            kv_[std::size_t(L)].k = ctx.heap_alloc(bytes);
            kv_[std::size_t(L)].v = ctx.heap_alloc(bytes);
            if (!kv_[std::size_t(L)].k.valid() || !kv_[std::size_t(L)].v.valid()) {
                if (err) *err = "llama KV allocation failed";
                return false;
            }
            b_.kv[std::size_t(L)] = kv_[std::size_t(L)];
        }

        dag_ = llama::build_llama_dag(g_, /*with_argmax=*/true);
        plan_ = llama::build_llama_scratch(dag_, g_);
        // Under a tap dump every value needs its own buffer, or a later
        // dispatch overwrites the one being read.
        coloring_ = llama::color_llama_scratch(dag_, plan_, /*no_recycle=*/golden_taps_enabled());
        if (!coloring_.hazard_free) {
            if (err) *err = "llama's activation colouring is not hazard-free";
            return false;
        }
        b_.pool.resize(std::size_t(coloring_.colors_used));
        // Sized at the PADDED row counts. A dense projection's GEMM rounds its
        // batch up to a whole tile and writes the padding rows for real, so a
        // pool sized at the batch itself is short by up to a tile.
        const std::vector<std::size_t> elems = llama::llama_pool_elems(
            dag_, plan_, coloring_, g_, llama::llama_qmm_pool_rows(max_rows_),
            llama::llama_qmm_pool_rows(max_sampled_));
        for (int c = 0; c < coloring_.colors_used; ++c) {
            b_.pool[std::size_t(c)] = ctx.heap_alloc(elems[std::size_t(c)] * 2);
            if (!b_.pool[std::size_t(c)].valid()) {
                if (err) *err = "llama activation pool allocation failed";
                return false;
            }
        }

        b_.io.resize(kIoSlotCount);
        // The page list is the largest of these and grows with the context.
        const std::size_t io_bytes = std::max({
            std::size_t(4096),
            std::size_t(g_.total_pages + 8) * 4,
            std::size_t(max_rows_ + 1) * 4,
            std::size_t(max_sampled_ + 1) * 4,
        });
        for (int i = 0; i < kIoSlotCount; ++i) b_.io[i] = ctx.heap_alloc(io_bytes);
        // The routed matvec declares a bias it does not read; see
        // `BoundLlama::zero_bias`. Wide enough for the widest routed output.
        if (g_.is_moe()) {
            const std::size_t bias_elems =
                std::size_t(std::max(g_.hidden, g_.moe_intermediate));
            b_.zero_bias = ctx.heap_alloc(bias_elems * 2);
            if (!b_.zero_bias.valid()) {
                if (err) *err = "llama routed bias allocation failed";
                return false;
            }
            if (b_.zero_bias.contents() != nullptr) {
                std::memset(b_.zero_bias.contents(), 0, b_.zero_bias.size);
            }
        }
        // One row per SAMPLED row, padded like the pool: the tail is a GEMM
        // too, and its padding rows land here.
        logits_ = ctx.heap_alloc(std::size_t(llama::llama_qmm_pool_rows(max_sampled_)) *
                                 std::size_t(g_.vocab) * 2);
        if (!logits_.valid()) {
            if (err) *err = "llama logits allocation failed";
            return false;
        }
        b_.argmax_params = ctx.heap_alloc(sizeof(ArgmaxParams));
        b_.eos_flag = ctx.heap_alloc(std::size_t(max_sampled_) * sizeof(std::uint32_t));
        if (!b_.argmax_params.valid() || !b_.eos_flag.valid()) {
            if (err) *err = "llama argmax allocation failed";
            return false;
        }
        auto* ap = static_cast<ArgmaxParams*>(b_.argmax_params.contents());
        *ap = {};
        ap->vocab = std::uint32_t(g_.vocab);

        if (!llama::build_llama_psos(ctx, kernels_dir, g_, psos_, err)) return false;
        if (!load_decode_psos(
                ctx, kernels_dir, base_, g_.quant, err,
                DecodePsoFeatures{.argmax = true})) {
            return false;
        }
        if (!load_multibatch_psos(
                ctx, kernels_dir, mb_, g_.quant, err,
                MultiBatchPsoFeatures{
                    .splitk = true,
                    .fp16_precast = !g_.is_moe() &&
                        g_.quant.bits == 4 && g_.quant.group == 64})) {
            return false;
        }

        // Split-K partials: the largest shape this model actually splits, with
        // one slice per member of the Q/K/V concurrency run. lm_head never
        // splits, so this stays in megabytes rather than hundreds of them.
        const std::size_t splitk_elems =
            llama::llama_splitk_partial_elems(g_, max_rows_);
        if (splitk_elems > 0) {
            splitk_partial_ = ctx.heap_alloc(sizeof(float) * splitk_elems);
            if (!splitk_partial_.valid()) {
                if (err) *err = "llama split-K partial allocation failed";
                return false;
            }
        }
        if (!g_.is_moe() && g_.quant.bits == 4 && g_.quant.group == 64) {
            const std::size_t fp16_elems =
                std::size_t(llama::llama_qmm_pool_rows(max_rows_)) *
                std::size_t(std::max(g_.hidden, g_.intermediate));
            fp16_input_ = ctx.heap_alloc(fp16_elems * sizeof(std::uint16_t));
            if (!fp16_input_.valid()) {
                if (err) *err = "llama FP16 QMM input allocation failed";
                return false;
            }
        }
        llama::bind_llama_consts(ctx, dag_, g_, /*rows=*/1, /*paged=*/true);
        llama::bind_llama_splitk(ctx, dag_, g_, /*rows=*/1, splitk_partial_,
                                 splitk_keep_, /*requests=*/1);
        llama::bind_llama_fp16_qmm(ctx, dag_, g_, /*rows=*/1, /*head_rows=*/1,
                                   /*requests=*/1, fp16_input_, fp16_keep_);
        bound_rows_ = 1;
        bound_requests_ = 1;
        try {
            llama::bind_llama_dag(ctx, b_, dag_, g_, coloring_, /*ordinal_base=*/0,
                                  /*paged=*/true);
        } catch (const std::exception& e) {
            if (err) *err = std::string("binding llama: ") + e.what();
            return false;
        }
        // The logits leave the pool: both the sampler and the argmax read a
        // slot of their own, so the head writes there directly.
        for (const llama::Dispatch& d : dag_) {
            if (d.kind == llama::Kind::LmHead) {
                ctx.arg_bind_ordinal(d.ordinal, (std::uint8_t)bind::Qmv::Out, logits_);
            } else if (d.kind == llama::Kind::Argmax) {
                ctx.arg_bind_ordinal(d.ordinal, (std::uint8_t)bind::Argmax::Logits, logits_);
            }
        }
        if (slab_ && !plan_segments(err)) return false;
        write_u32s(b_.io[int(IoSlot::AttnMaskStride)], {0u});
        if (b_.io[int(IoSlot::AttnMaskEnabled)].contents() != nullptr) {
            std::memset(b_.io[int(IoSlot::AttnMaskEnabled)].contents(), 0, 1);
        }
        return true;
    }

    int vocab() const override { return g_.vocab; }
    int n_layers() const override { return g_.n_layers; }
    WeightBytes weight_bytes() const override {
        return pie::metal::weight_bytes(b_.weights, g_.n_experts, g_.experts_per_token);
    }

    void reset() override {
        for (auto& kv : kv_) {
            if (kv.k.valid() && kv.k.contents()) std::memset(kv.k.contents(), 0, kv.k.size);
            if (kv.v.valid() && kv.v.contents()) std::memset(kv.v.contents(), 0, kv.v.size);
        }
    }

    /// A single token, expressed as a one-row fire. The decode path is not a
    /// separate encoder here -- it is `fire` with one row, which is what makes
    /// `llama_numerics_test`'s verification of the M=1 shapes carry over.
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
    /// Unlike gpt-oss, a fire of R rows is ONE pass. The row budget costs
    /// memory -- the activation pool is [rows, width] -- rather than time.
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
        // The ring ABI's key count. Harmless under the paged one, which bounds
        // each row by its own `position_ids[row]`, but it is the slot the M=1
        // contiguous path reads and leaving it stale would be a trap for
        // anything that switches back.
        write_i32(b_.io[int(IoSlot::SeqLen)], std::int32_t(csr.position_ids.back()) + 1);
        const int head_rows = csr.sample_rows.empty() ? rows : int(csr.sample_rows.size());
        // How many requests these rows came from. `qo_indptr` is one entry per
        // request plus a terminator, so this is the CSR's own answer rather than
        // a second opinion. Attention needs it: the row count alone cannot tell a
        // 32-row prefill from 32 members decoding a token each, and those two want
        // opposite kernels.
        const int requests = csr.qo_indptr.size() >= 2 ? int(csr.qo_indptr.size()) - 1 : 1;
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
        // The constants carry the row count into the elementwise widths and
        // the row-gather's pitch, so they are rebound whenever it changes --
        // and only then, because this is every dispatch's argument table.
        if (bound_rows_ != rows || bound_head_rows_ != head_rows ||
            bound_requests_ != requests) {
            llama::bind_llama_consts(ctx, dag_, g_, rows, /*paged=*/true);
            llama::bind_llama_splitk(ctx, dag_, g_, rows, splitk_partial_,
                                     splitk_keep_, requests);
            llama::bind_llama_fp16_qmm(ctx, dag_, g_, rows, head_rows, requests,
                                       fp16_input_, fp16_keep_);
            bound_rows_ = rows;
            bound_head_rows_ = head_rows;
            bound_requests_ = requests;
        }
        if (paging_.active()) {
            return fire_segmented(
                ctx, rows, head_rows, requests, csr.run_argmax, pre, post);
        }
        return ctx.run_step([&](StepEncoder& se) {
            if (pre) pre(se);
            llama::encode_llama_step(se, dag_, g_, base_, psos_, /*ordinal_base=*/0, &mb_, rows,
                                     head_rows, requests, /*begin=*/0, /*end=*/~std::size_t(0),
                                     csr.run_argmax);
            if (post) post(se);
        });
    }

    SlotHandle logits_slot() const override { return logits_; }
    SlotHandle greedy_tokens_slot() const override {
        return b_.io[int(IoSlot::NextToken)];
    }

    void dump_taps(int rows) const override {
        dump_ll_taps(dag_, g_, plan_, coloring_, b_.pool, logits_, rows, bound_head_rows_);
    }

  private:
    /// Where the step has to be cut, and what the host reads at each cut.
    ///
    /// A mixture layer's router runs INSIDE that layer, so the host cannot know
    /// which experts the layer wants until part of the layer has already run.
    /// That is why one cut per mixture layer is the minimum and why the cut is
    /// where it is: immediately after `RouterTopK`, the first point at which
    /// the answer exists and the last before anything reads the bank.
    ///
    /// Cuts are pushed out to the end of their concurrency run. A run is a set
    /// of dispatches the encoder deliberately does NOT barrier between, and
    /// splitting one across two command buffers would serialize exactly the
    /// dispatches that were grouped to overlap.
    /// Resolve the pool slot a scratch VALUE was coloured onto.
    SlotHandle pool_slot_of(int value) const {
        SlotHandle out{};
        for (const llama::Use& u : plan_.uses) {
            if (!u.is_write || u.value != value) continue;
            if (u.index < 0 || std::size_t(u.index) >= coloring_.per_dispatch.size()) continue;
            for (const auto& sb : coloring_.per_dispatch[std::size_t(u.index)]) {
                if (sb.bind_index != u.bind_index) continue;
                if (sb.color < 0 || std::size_t(sb.color) >= b_.pool.size()) continue;
                out = b_.pool[std::size_t(sb.color)];
            }
        }
        return out;
    }

    bool plan_segments(std::string* err) {
        const std::vector<int> run_ends = llama::llama_run_ends(dag_);
        std::vector<ExpertPaging::Cut> cuts;
        std::size_t nth = 0;
        for (std::size_t i = 0; i < dag_.size(); ++i) {
            if (dag_[i].kind != llama::Kind::RouterTopK) continue;
            ExpertPaging::Cut c;
            // Pushed out to the end of the router's concurrency run: a run
            // split across two command buffers would serialize exactly the
            // dispatches that were grouped to overlap.
            c.end = std::size_t(run_ends[i]) + 1;
            if (nth < plan_.expert_ids_by_layer.size()) {
                c.ids = pool_slot_of(plan_.expert_ids_by_layer[nth]);
            }
            ++nth;
            cuts.push_back(c);
        }
        return paging_.plan(slab_, std::move(cuts), dag_.size(), g_.n_experts,
                            g_.experts_per_token, max_rows_, "llama", err);
    }

    StepTiming fire_segmented(RawMetalContext& ctx, int rows, int head_rows, int requests,
                              bool run_argmax, const EncodeHook& pre, const EncodeHook& post) {
        const std::size_t n = dag_.size();
        return paging_.fire(
            ctx, rows,
            [this, rows, head_rows, requests, run_argmax, n, &pre, &post](
                StepEncoder& se, std::size_t begin, std::size_t end) {
                if (begin == 0 && pre) pre(se);
                llama::encode_llama_step(se, dag_, g_, base_, psos_, /*ordinal_base=*/0, &mb_,
                                         rows, head_rows, requests, begin, end, run_argmax);
                if (end == n && post) post(se);
            });
    }

    llama::LlamaGeometry g_{};
    int max_ctx_ = 0;
    std::vector<llama::Dispatch> dag_{};
    llama::ScratchPlan plan_{};
    model::ScratchColoring coloring_{};
    llama::BoundLlama b_{};
    llama::LlamaPsos psos_{};
    DecodeStepPsos base_{};
    MultiBatchPsos mb_{};
    std::vector<llama::KvPages> kv_{};
    /// Keeps the streamed weights' mapping alive; see the gpt-oss engine.
    std::shared_ptr<void> weight_mapping_{};
    int max_rows_ = 1;
    int max_sampled_ = 1;
    int bound_rows_ = 0;
    int bound_head_rows_ = 0;
    int bound_requests_ = 0;
    SlotHandle logits_{};
    /// The split GEMM's partial [M, N] slices, and the small constant buffers
    /// the argument tables point at. One partials buffer serves every split
    /// projection: they are serialized by barriers and each is reduced before
    /// the next one runs.
    SlotHandle splitk_partial_{};
    std::vector<SlotHandle> splitk_keep_{};
    SlotHandle fp16_input_{};
    std::vector<SlotHandle> fp16_keep_{};
    /// The routed experts' paging cache, when a budget asked for one.
    std::shared_ptr<ExpertSlab> slab_{};
    ExpertPaging paging_{};
};

}  // namespace

std::uint32_t SimpleFamilyEngine::max_forward_tokens_for_budget(
    pie::metal::model::ModelFamily family, const SetupConfig& cfg, int max_ctx,
    std::uint64_t budget_bytes) {
    // `extra_heap_bytes` covers the KV region and a base slack as well as the
    // pool, and only the POOL scales with rows -- the other two dominate. So
    // the budget is spent on the DIFFERENCE from a one-row fire, not on the
    // total: comparing the total against a pool-sized budget makes every model
    // answer "one row", which is how this first came out at the floor.
    SetupConfig probe = cfg;
    probe.max_forward_tokens = 1;
    const std::uint64_t floor = extra_heap_bytes(family, probe, max_ctx);
    const auto rows_cost = [&](std::uint32_t rows) {
        probe.max_forward_tokens = rows;
        const std::uint64_t total = extra_heap_bytes(family, probe, max_ctx);
        return total > floor ? total - floor : 0;
    };

    // Bisect the real function. Monotone in `max_forward_tokens`, so this is
    // exact rather than an estimate -- and it stays exact when the DAG changes.
    if (rows_cost(kPagedMaxForwardTokensHardCeiling) <= budget_bytes) {
        return kPagedMaxForwardTokensHardCeiling;
    }
    std::uint32_t lo = 1, hi = kPagedMaxForwardTokensHardCeiling;
    while (lo + 1 < hi) {
        const std::uint32_t mid = lo + (hi - lo) / 2;
        if (rows_cost(mid) <= budget_bytes) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return std::max(lo, kPagedMinForwardTokens);
}

std::function<bool(const std::string&)> SimpleFamilyEngine::stream_predicate(
    bool stream_routed_experts, bool slab_paging) {
    if (!stream_routed_experts) return {};

    // A ROUTED expert bank, and nothing else. One pattern, no family.
    //
    // What this buys is a LOAD that maps instead of copying: 16.31 GB out of
    // the heap on Qwen3-30B-A3B, load 33.2 s -> 0.67 s, prefill unchanged and
    // decode down 3%. A dense FFN would map just as well, but it is read whole
    // every token, so mapping it trades a copy for nothing else, and this
    // switch is about the banks where the trade is one-sided.
    //
    // What it does NOT buy is running a model larger than the machine, and it
    // is worth being exact about why, because this comment used to claim the
    // opposite -- that a token touching 8 experts of 128 faults an eighth of
    // the bank in and the kernel evicts the rest. Nothing is evicted. The
    // mapping goes into the residency set and `requestResidency` wires every
    // page of it, whether or not a kernel ever reads it. Measured with
    // `vm_stat` across a streamed Qwen3-30B-A3B run: wired memory goes from
    // 1.5 GB at rest to 18.4 GB while the model is live -- the 16.31 GB pack
    // plus the 0.87 GB heap, all of it, for the whole run -- and returns to
    // 1.4 GB when it exits.
    //
    // That is the hardware, not this driver: an Apple Silicon GPU has no
    // demand paging, and a kernel that touched a page the residency set had
    // let go would abort its command buffer rather than fault it back. So a
    // bank larger than the working set cannot be mapped at all; it has to be
    // paged through a bounded slab that stays wired while its CONTENTS change,
    // which is a different mechanism from this one and not yet built here.
    //
    // This predicate used to stream the dense FFN too, for the families that
    // have one. That was wrong twice over. It made one switch mean two trades,
    // so an operator who wanted the free one on a dense model silently bought
    // the expensive one. And `[model].stream_routed_experts` is the same key
    // the CUDA driver reads, where every call site of it -- mixtral,
    // deepseek_v4, the qwen mixture -- sits inside a routed expert stack and
    // no dense weight is ever streamed. The same configuration now means the
    // same thing on both backends.
    //
    // The name is the whole test, which is why there is no family argument any
    // more. A routed layer publishes `mlp.experts.gate_proj` whatever family
    // it belongs to; a dense one has no experts to match. Asking the family
    // was asking a proxy for the layer shape, and the llama family -- the only
    // one with both shapes -- is exactly where that proxy broke: its clause
    // named `mlp.gate_proj`, which is not a substring of
    // `mlp.experts.gate_proj`, so Qwen3-MoE streamed nothing while gpt-oss
    // streamed 10.75 GB off the same access pattern.
    //
    // `.bias` is where MAPPING and PAGING part, and the difference is
    // correctness rather than taste.
    //
    // Mapping leaves it resident, and should: gpt-oss's expert biases are three
    // orders of magnitude smaller than the weights beside them, the whole table
    // is there whichever expert runs, and mapping changes no index.
    //
    // Paging renumbers. A slot is not an expert, so `expert_ids` stops meaning
    // "expert 57" and starts meaning "the slot 57 was copied into" -- and the
    // routed matvec offsets the bias with THAT SAME BUFFER:
    // `bias_row += expert_ids[sel] * out_vec_size` in `quantized_qmv.metal`. A
    // resident bias table indexed by a slot number returns some other expert's
    // bias beside this expert's weights. Fluent wrong tokens, not an error.
    //
    // So under paging the bias goes into the slab too, as one more tensor kind
    // banded like the rest -- which is exactly what makes it take the SAME slot
    // number as the weights it belongs to. No new mechanism: the slab was
    // already a set of parallel bands sharing a slot, and this is another band.
    // The llama family has no expert bias at all, so there this changes
    // nothing.
    // `experts.` and not `mlp.experts.`: gemma 4 hangs its bank off the layer
    // rather than off the mlp -- `experts.switch_glu.gate_proj` -- and a pattern
    // that named the mlp was naming one family's directory layout rather than
    // the thing being matched. No tensor outside a routed bank contains it;
    // gemma's `router.per_expert_scale` is the near miss, and it does not.
    return [slab_paging](const std::string& n) {
        const bool bias = n.size() > 5 && n.compare(n.size() - 5, 5, ".bias") == 0;
        if (bias && !slab_paging) return false;
        return n.find("experts.") != std::string::npos;
    };
}

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
        for (const std::size_t e : g4_pool_elems(dag, sp, pg, col, rows, sampled)) bytes += e * 2;
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
        rows = gptoss::gptoss_qmm_pool_rows(rows);
        sampled = gptoss::gptoss_qmm_pool_rows(sampled);
        for (const std::size_t e : go_pool_elems(dag, g, col, rows, sampled)) bytes += e * 2;
        bytes += std::size_t(sampled) * std::size_t(g.vocab) * 2;  // the logits slot
    } else if (family == pie::metal::model::ModelFamily::Llama) {
        llama::LlamaGeometry g;
        std::string ignore;
        if (!llama_geometry(cfg, g, max_ctx, &ignore)) return bytes;
        bytes += llama::llama_kv_region_bytes(g, max_ctx, 2);
        // The engine pages its KV, so `max_ctx` rounds up to a whole page per
        // layer. One page's slack per layer, per side.
        bytes += std::size_t(g.n_layers) * 2 * 32 * std::size_t(g.n_kv_heads) *
                 std::size_t(g.head_dim) * 2;
        // The pool, summed from the SAME colouring the engine will build, at
        // the SAME padded row counts. The engine batches, so the row budget IS
        // a memory budget here -- every activation is [rows, width], and a
        // guess that under-counts fails at `heap_alloc` naming nothing.
        int rows = cfg.max_forward_tokens > 0 ? int(cfg.max_forward_tokens) : 1;
        int sampled = cfg.max_forward_requests > 0 ? int(cfg.max_forward_requests) : 1;
        if (sampled > rows) sampled = rows;
        rows = llama::llama_qmm_pool_rows(rows);
        sampled = llama::llama_qmm_pool_rows(sampled);
        llama::LlamaGeometry pg = g;
        pg.paged_kv_enabled = true;
        const auto dag = llama::build_llama_dag(pg, /*with_argmax=*/false);
        const llama::ScratchPlan plan = llama::build_llama_scratch(dag, pg);
        const model::ScratchColoring col =
            llama::color_llama_scratch(dag, plan, /*no_recycle=*/golden_taps_enabled());
        for (const std::size_t e : llama::llama_pool_elems(dag, plan, col, pg, rows, sampled)) {
            bytes += e * 2;
        }
        bytes += std::size_t(sampled) * std::size_t(g.vocab) * 2;  // the logits slot
        bytes += sizeof(float) * llama::llama_splitk_partial_elems(pg, rows);
        if (!g.is_moe() && g.quant.bits == 4 && g.quant.group == 64) {
            bytes += sizeof(std::uint16_t) * std::size_t(rows) *
                     std::size_t(std::max(g.hidden, g.intermediate));
        }
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
    if (family == pie::metal::model::ModelFamily::Llama) {
        auto e = std::make_unique<LlamaEngine>();
        if (!e->init(ctx, kernels_dir, cfg, load_plan, max_ctx, err)) return nullptr;
        return e;
    }
    if (err) *err = "SimpleFamilyEngine: not a family with no recurrent state";
    return nullptr;
}

}  // namespace pie::metal::batch

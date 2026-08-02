// Is every slot the llama family's kernels READ actually bound?
//
// The structural test says the DAG resolves and the colouring is sound. It
// cannot say this, because "which buffers does this shader declare" is a fact
// about the .metal files and not about the DAG. So this one builds the real
// binder against a real argument table and asks the context, slot by slot.
//
// It is a cheap test for an expensive class of bug. An unbound slot is never a
// crash: Metal's argument table holds whatever was written into that ordinal
// last, so the kernel reads a stale pointer or a stale integer and computes
// something. The first version of this family did not bind
// `bind::Embed::Hidden` -- the gather's row width, which every other family
// binds -- and the symptom would have been the embedding table indexed with the
// wrong row pitch: a real vector, from the wrong token, with no error anywhere.
//
// No checkpoint and no numerics. The weights are dummy allocations under the
// names the binder asks for, which is exactly what makes this runnable on any
// machine with a GPU.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "batch/decode_abi.hpp"
#include "loader/heap_bind.hpp"
#include "model/llama/bind.hpp"
#include "model/llama/decode_consts.hpp"
#include "model/llama/decode_step.hpp"
#include "model/llama/encode.hpp"
#include "model/llama/geometry.hpp"
#include "model/llama/scratch.hpp"
#include "mtl4_context.hpp"

using pie::metal::DecodeGeometry;
using pie::metal::RawMetalContext;
using pie::metal::WeightBind;
using pie::metal::kIoSlotCount;
using pie::metal::weight_binds;
namespace model = pie::metal::model;
namespace llama = pie::metal::llama;
using llama::BoundLlama;
using llama::Dispatch;
using llama::Kind;
using llama::LlamaGeometry;
using llama::ScratchPlan;
using llama::bind_llama_consts;
using llama::bind_llama_dag;
using llama::build_llama_dag;
using llama::build_llama_scratch;
using llama::color_llama_scratch;
using llama::llama_kv_bytes_per_layer;
using llama::llama_pool_elems;
using llama::shared_kind;

namespace {

int g_pass = 0, g_fail = 0;

void expect(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
        std::printf("  PASS  %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

/// A shape every kernel in the family can legally run at.
///
/// Not arbitrary. `affine_qmv_fast` requires K % 512 == 0 and N % 8 == 0, so
/// every width that is ever a K -- hidden, the q projection, the two FFN
/// intermediates -- is a multiple of 512. And the family compiles SDPA only at
/// d_128, so the head is 128 wide. A test model that ignored either would fail
/// for a reason that has nothing to do with binding.
LlamaGeometry base() {
    LlamaGeometry g;
    g.hidden = 512;
    g.n_layers = 2;
    g.vocab = 1024;
    g.n_q_heads = 4;
    g.n_kv_heads = 2;
    g.head_dim = 128;
    g.intermediate = 512;
    g.kv_max_ctx = 64;
    g.rope_theta = 500000.0f;
    return g;
}

/// How many leading slots each kind's kernel declares.
///
/// Read off the `[[buffer(N)]]` attributes in `src/kernels/`, which is the only
/// place the answer lives. A count that is too LOW passes vacuously, so these
/// are the full declared arity of each shader rather than the part that seemed
/// interesting.
int declared_slots(Kind k, bool paged = false) {
    // The two attention kinds have two ABIs, and the paged one is WIDER. Its
    // extra slots are the page tables, the per-token request map and the
    // sliding window -- and the window is the interesting one, because llama
    // is full-attention and so looks like it has nothing to say about it. One
    // paged kernel serves both, so the slot is read either way.
    if (paged && k == Kind::KvAppend) return 15;   // KvAppendPaged, K..WOff
    // SdpaPaged Q..Window. `Sinks` (16) is gpt-oss's, and this family compiles
    // the non-sink instantiation, so it is deliberately not counted.
    if (paged && k == Kind::Sdpa) return 16;
    switch (k) {
        // w, scales, biases, x, y, K, N.
        case Kind::QmvQ:
        case Kind::QmvK:
        case Kind::QmvV:
        case Kind::QmvO:
        case Kind::QmvGate:
        case Kind::QmvUp:
        case Kind::QmvDown:
        case Kind::Router:
        case Kind::LmHead:
            return 7;
        // The routed matvec adds the ids and the three stride scalars.
        case Kind::ExpertGate:
        case Kind::ExpertUp:
        case Kind::ExpertDown:
            return 12;
        // w, scales, biases, id, out, hidden.
        case Kind::EmbedGather:
            return 6;
        // x, w, out, params.
        case Kind::AttnNorm:
        case Kind::FfnNorm:
        case Kind::FinalRms:
        case Kind::QNorm:
        case Kind::KNorm:
            return 4;
        // x, position, scale, base, head_dim.
        case Kind::RopeQ:
        case Kind::RopeK:
            return 5;
        // k, v, kpages, vpages, position, head_dim, k_head_stride, k_seq_stride.
        case Kind::KvAppend:
            return 8;
        // q, k, v, out, n, gqa, scale, and the four strides.
        case Kind::Sdpa:
            return 11;
        // gate, up, out.
        case Kind::SiluMul:
        case Kind::ExpertSiluMul:
            return 3;
        // x, residual, out.
        case Kind::AttnResidual:
        case Kind::FfnResidual:
            return 3;
        // logits, ids, weights, params.
        case Kind::RouterTopK:
            return 4;
        // y, weights, out, params.
        case Kind::ExpertCombine:
            return 4;
        // in, out, rows, params.
        case Kind::RowGather:
            return 4;
        // Excluded: the engine never builds it (`with_argmax=false`), and its
        // eos flag is the runtime's slot rather than the model's.
        case Kind::Argmax:
            return 0;
    }
    return 0;
}

const char* kind_name(Kind k) {
    switch (k) {
        case Kind::EmbedGather: return "EmbedGather";
        case Kind::AttnNorm: return "AttnNorm";
        case Kind::QmvQ: return "QmvQ";
        case Kind::QmvK: return "QmvK";
        case Kind::QmvV: return "QmvV";
        case Kind::QNorm: return "QNorm";
        case Kind::KNorm: return "KNorm";
        case Kind::RopeQ: return "RopeQ";
        case Kind::RopeK: return "RopeK";
        case Kind::KvAppend: return "KvAppend";
        case Kind::Sdpa: return "Sdpa";
        case Kind::QmvO: return "QmvO";
        case Kind::AttnResidual: return "AttnResidual";
        case Kind::FfnNorm: return "FfnNorm";
        case Kind::QmvGate: return "QmvGate";
        case Kind::QmvUp: return "QmvUp";
        case Kind::SiluMul: return "SiluMul";
        case Kind::QmvDown: return "QmvDown";
        case Kind::FfnResidual: return "FfnResidual";
        case Kind::Router: return "Router";
        case Kind::RouterTopK: return "RouterTopK";
        case Kind::ExpertGate: return "ExpertGate";
        case Kind::ExpertUp: return "ExpertUp";
        case Kind::ExpertSiluMul: return "ExpertSiluMul";
        case Kind::ExpertDown: return "ExpertDown";
        case Kind::ExpertCombine: return "ExpertCombine";
        case Kind::RowGather: return "RowGather";
        case Kind::FinalRms: return "FinalRms";
        case Kind::LmHead: return "LmHead";
        case Kind::Argmax: return "Argmax";
    }
    return "?";
}

/// Stage a buffer under every tensor name the binder will ask for.
///
/// The names come from `weight_binds(shared_kind(...))` -- the same call the
/// binder makes -- so this cannot drift from what is actually requested. The
/// sizes are generous rather than exact: nothing is read, and a size that is
/// too small would fail for a reason this test is not about.
void stage_dummy_weights(RawMetalContext& ctx, BoundLlama& b, const std::vector<Dispatch>& dag,
                         const LlamaGeometry& g) {
    for (const Dispatch& d : dag) {
        for (const WeightBind& wb : weight_binds(shared_kind(d.kind, g), d.layer,
                                                 DecodeGeometry{}, false)) {
            if (b.weights.count(wb.tensor) != 0) continue;
            b.weights[wb.tensor] = ctx.heap_alloc(1 << 20);
        }
    }
}

void check_binds(const char* who, LlamaGeometry g, RawMetalContext& ctx, bool paged = false) {
    std::printf("\n-- %s --\n", who);
    g.paged_kv_enabled = paged;
    const std::vector<Dispatch> dag = build_llama_dag(g, /*with_argmax=*/false);
    const ScratchPlan plan = build_llama_scratch(dag, g);
    const model::ScratchColoring col = color_llama_scratch(dag, plan);
    expect(col.hazard_free, std::string(who) + ": the colouring is hazard-free");

    BoundLlama b;
    stage_dummy_weights(ctx, b, dag, g);
    expect(!b.weights.empty(), std::string(who) + ": the binder asked for weights by name");

    b.pool.resize(std::size_t(col.colors_used));
    const std::vector<std::size_t> elems = llama_pool_elems(dag, plan, col, g);
    for (int c = 0; c < col.colors_used; ++c) {
        b.pool[std::size_t(c)] = ctx.heap_alloc(elems[std::size_t(c)] * 2 + 64);
    }
    if (g.is_moe()) b.zero_bias = ctx.heap_alloc(4096);
    b.io.resize(kIoSlotCount);
    for (int i = 0; i < kIoSlotCount; ++i) b.io[std::size_t(i)] = ctx.heap_alloc(4096);
    b.kv.resize(std::size_t(g.n_layers));
    for (int L = 0; L < g.n_layers; ++L) {
        const std::size_t bytes = llama_kv_bytes_per_layer(g, g.kv_max_ctx, 2);
        b.kv[std::size_t(L)].k = ctx.heap_alloc(bytes);
        b.kv[std::size_t(L)].v = ctx.heap_alloc(bytes);
    }

    const int bound = bind_llama_consts(ctx, dag, g, /*rows=*/1, paged);
    expect(bound > 0, std::string(who) + ": constants were bound");
    bool threw = false;
    try {
        bind_llama_dag(ctx, b, dag, g, col, /*ordinal_base=*/0, paged);
    } catch (const std::exception& e) {
        threw = true;
        std::printf("    bind threw: %s\n", e.what());
    }
    expect(!threw, std::string(who) + ": the binder ran to completion");

    // The question this test exists for.
    int missing = 0;
    for (const Dispatch& d : dag) {
        const int n = declared_slots(d.kind, paged);
        for (int slot = 0; slot < n; ++slot) {
            if (ctx.arg_slot_is_bound(d.ordinal, std::uint8_t(slot))) continue;
            if (missing < 12) {
                std::printf("    unbound: ord=%d %s layer=%d slot=%d\n", d.ordinal,
                            kind_name(d.kind), d.layer, slot);
            }
            ++missing;
        }
    }
    expect(missing == 0, std::string(who) + ": every slot the kernels read is bound");

    // A weight name the loader will not have is a load failure, and it should
    // arrive as one rather than as a null binding. Uniform layers mean the
    // check is worth making once per family and not once per layer.
    bool named = true;
    for (const auto& [name, slot] : b.weights) {
        if (name.empty() || !slot.valid()) named = false;
    }
    expect(named, std::string(who) + ": every requested tensor has a real name");
}

}  // namespace

int main() {
    std::printf("llama_bind_test — every declared slot, on a real argument table\n");

    auto ctx = RawMetalContext::create(std::size_t(2) << 30);
    if (!ctx) {
        std::printf("  SKIP  no Metal device\n");
        return 0;
    }

    check_binds("llama-3 (dense)", base(), *ctx);

    LlamaGeometry qwen3 = base();
    qwen3.qk_norm = true;
    check_binds("qwen3 (dense, qk-norm)", qwen3, *ctx);

    LlamaGeometry moe = base();
    moe.qk_norm = true;
    moe.n_experts = 8;
    moe.experts_per_token = 2;
    moe.moe_intermediate = 512;
    check_binds("qwen3-moe (routed)", moe, *ctx);

    LlamaGeometry untied = base();
    untied.tied_embeddings = false;
    check_binds("untied head", untied, *ctx);

    // The paged ABI, which is the same DAG bound against wider attention
    // kernels. Worth its own pass rather than trusting the ring one: the two
    // differ only in two switch arms, and the slots those arms do NOT write
    // are exactly the ones nobody notices -- an unbound page table reads as a
    // pointer made from whatever the ordinal held last.
    check_binds("llama-3 (dense, paged)", base(), *ctx, /*paged=*/true);
    check_binds("qwen3-moe (routed, paged)", moe, *ctx, /*paged=*/true);

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

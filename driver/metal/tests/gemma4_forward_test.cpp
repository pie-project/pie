// Does gemma 4 produce a forward pass on this machine, from the real checkpoint?
//
// Everything before this was structural: the DAG resolves, the shapes are legal,
// the binds cover every slot. None of it says the model computes anything. This
// loads `~/.pie-bench/gemma4-e2b-pie`, runs one token through the decode DAG and
// looks at the logits.
//
// Skipped (not failed) when the checkpoint is absent, so CI without a 2.5 GB
// download stays green while the machine that has it gets the real answer.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "mtl4_context.hpp"
#include "batch/decode_abi.hpp"
#include "kernels/decode_psos.hpp"
#include "loader/heap_bind_metal.hpp"
#include "loader/load_plan.hpp"
#include "pie_loader/checkpoint_source.hpp"
#include "model/contract.hpp"
#include "model/gemma4/bind.hpp"
#include "model/gemma4/decode_consts.hpp"
#include "model/gemma4/decode_step.hpp"
#include "model/gemma4/encode.hpp"
#include "model/gemma4/geometry.hpp"
#include "model/gemma4/kernels.hpp"
#include "model/gemma4/scratch.hpp"

using namespace pie::metal;
using namespace pie::metal::gemma4;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

float from_bf16(std::uint16_t h) {
    const std::uint32_t bits = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// E2B's shape, which `geometry_from_facts` is separately tested against.
struct Facts {
    int n_layers = 35, hidden = 1536, intermediate = 6144;
    int n_q_heads = 8, n_kv_heads = 1, head_dim = 256, global_head_dim = 512;
    int sliding_window = 512, num_kv_shared_layers = 20, per_layer_emb_dim = 256;
    int full_attn_interval = 5;
    bool double_wide_mlp = true;
    float final_softcap = 30.0f;
    float rope_theta_full = 1.0e6f, rope_theta_sliding = 1.0e4f, full_partial_rotary = 0.25f;
    bool present() const { return n_layers > 0 && hidden > 0; }
};

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string ckpt = argc > 1 ? argv[1] : std::string();
    if (ckpt.empty()) {
        const char* home = std::getenv("HOME");
        if (home != nullptr) ckpt = std::string(home) + "/.pie-bench/gemma4-e2b-pie";
    }
    std::string kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;

    {
        std::string probe = ckpt + "/config.json";
        FILE* f = std::fopen(probe.c_str(), "rb");
        if (f == nullptr) {
            std::printf("gemma4 forward: SKIP (no checkpoint at %s)\n", ckpt.c_str());
            return 0;
        }
        std::fclose(f);
    }
    std::printf("gemma4 forward (%s)\n", ckpt.c_str());

    Gemma4Geometry g;
    std::string err;
    if (!geometry_from_facts(Facts{}, g, &err)) {
        std::printf("  FAIL  geometry: %s\n", err.c_str());
        return 1;
    }
    g.vocab = 262144;

    auto ctx = RawMetalContext::create(std::size_t(6) << 30);
    if (!ctx) {
        std::printf("  FAIL  RawMetalContext::create\n");
        return 1;
    }

    // ── the plan the contract authors for this checkpoint ──
    pie_loader::LoadPlan plan;
    try {
        pie::metal::model::ContractFacts facts;
        facts.first_kv_shared_layer = g.first_kv_shared();
        plan = compile_load_plan(ckpt, metal_device_target(), "gemma4", facts);
    } catch (const std::exception& e) {
        std::printf("  FAIL  compile_load_plan: %s\n", e.what());
        return 1;
    }
    expect(true, "the contract compiles a load plan for the real checkpoint");

    // ── weights ──
    BoundGemma4 b;
    try {
        const auto storage = plan.view();
        pie_loader::CheckpointSource view(storage);
        StagedWeights staged =
            stage_plan_weights(*ctx, view, plan, storage.memory.persistent_bytes);
        b.weights = std::move(staged.weights);
    } catch (const std::exception& e) {
        std::printf("  FAIL  stage_plan_weights: %s\n", e.what());
        return 1;
    }
    std::printf("    (%zu tensors staged)\n", b.weights.size());
    expect(!b.weights.empty(), "the checkpoint's tensors are staged into the heap");

    // ── KV, sized per owning layer ──
    const int max_ctx = 1024;
    b.kv.resize(g.n_layers);
    for (int L = 0; L < g.n_layers; ++L) {
        if (g.is_kv_shared(L)) continue;
        const std::size_t bytes = gemma4_kv_bytes_per_layer(g, L, max_ctx, 2);
        b.kv[L].k = ctx->heap_alloc(bytes);
        b.kv[L].v = ctx->heap_alloc(bytes);
    }

    // ── the DAG, its dataflow, and the pool that dataflow needs ──
    const auto dag = build_gemma4_dag(g, /*with_argmax=*/false);
    const ScratchPlan sp = build_gemma4_scratch(dag, g);
    const ScratchColoring coloring = color_gemma4_scratch(dag, sp);
    expect(coloring.hazard_free, "the activation colouring is hazard-free");

    b.pool.resize(coloring.colors_used);
    const std::size_t widest = std::size_t(g.vocab) * 2;
    for (int c = 0; c < coloring.colors_used; ++c) b.pool[c] = ctx->heap_alloc(widest);

    b.io.resize(kIoSlotCount);
    for (int i = 0; i < kIoSlotCount; ++i) b.io[i] = ctx->heap_alloc(4096);
    // Token 2 (<bos> in gemma's vocabulary) at position 0.
    *static_cast<std::int32_t*>(b.io[int(IoSlot::TokenId)].contents()) = 2;
    *static_cast<std::int32_t*>(b.io[int(IoSlot::Position)].contents()) = 0;

    Gemma4Psos psos;
    DecodeStepPsos base;
    if (!build_gemma4_psos(*ctx, kernels_dir, psos, &err) ||
        !load_decode_psos(*ctx, kernels_dir, base, /*with_argmax=*/false, &err)) {
        std::printf("  FAIL  pipelines: %s\n", err.c_str());
        return 1;
    }

    const int bound = bind_gemma4_consts(*ctx, dag, g);
    bind_gemma4_dag(*ctx, b, dag, g, coloring);
    std::printf("    (%d constants bound over %zu dispatches)\n", bound, dag.size());
    ctx->make_resident();

    ctx->run_step([&](StepEncoder& se) {
        encode_gemma4_step(se, dag, g, base, psos);
    });

    // ── did it compute anything? ──
    const auto* logits =
        static_cast<const std::uint16_t*>(b.pool[coloring.per_dispatch.empty()
                                                     ? 0
                                                     : coloring.per_dispatch.back().front().color]
                                              .contents());
    int nonzero = 0, nan_or_inf = 0;
    float best = -1e30f;
    int argmax = -1;
    for (int i = 0; i < g.vocab; ++i) {
        const float v = from_bf16(logits[i]);
        if (v != 0.0f) ++nonzero;
        if (std::isnan(v) || std::isinf(v)) ++nan_or_inf;
        if (v > best) {
            best = v;
            argmax = i;
        }
    }
    std::printf("    (nonzero %d/%d, argmax %d, max logit %.4f)\n", nonzero, g.vocab,
                argmax, best);
    expect(nonzero > g.vocab / 2, "the logits are populated, not a zeroed buffer");
    expect(nan_or_inf == 0, "and finite everywhere");
    expect(std::fabs(best) <= 30.0f + 1e-3f,
           "within the final softcap, so the tail of the graph ran");

    std::printf("\n==== gemma4_forward_test: %s ====\n",
                failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}

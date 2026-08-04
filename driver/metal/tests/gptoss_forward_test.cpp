// Does gpt-oss produce a forward pass on this machine, from the real checkpoint?
//
// Everything before this was structural: the DAG resolves, the shapes are legal,
// the binds cover every slot. None of it says the model computes anything. This
// loads `~/.pie-bench/gptoss-20b-pie4`, teacher-forces a short prompt through
// the decode DAG and looks at the logits.
//
// Skipped (not failed) when the checkpoint is absent, so CI without an 11 GB
// download stays green while the machine that has it gets the real answer.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "mtl4_context.hpp"
#include "batch/decode_abi.hpp"
#include "batch/golden_tap.hpp"
#include "kernels/decode_psos.hpp"
#include "loader/heap_bind_metal.hpp"
#include "loader/load_plan.hpp"
#include "pie_loader/checkpoint_source.hpp"
#include "model/contract.hpp"
#include "model/gptoss/bind.hpp"
#include "model/gptoss/decode_consts.hpp"
#include "model/gptoss/decode_step.hpp"
#include "model/gptoss/encode.hpp"
#include "model/gptoss/geometry.hpp"
#include "model/gptoss/kernels.hpp"
#include "model/gptoss/scratch.hpp"

using namespace pie::metal;
using namespace pie::metal::gptoss;

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

// gpt-oss-20b's shape, read off the checkpoint's config.json.
struct Facts {
    int n_layers = 24, hidden = 2880, vocab = 201088;
    float eps = 1e-5f;
    int n_q_heads = 64, n_kv_heads = 8, head_dim = 64;
    int sliding_window = 128;
    int n_experts = 32, experts_per_token = 4, intermediate = 2880;
    float swiglu_limit = 7.0f;
    float rope_theta = 150000.0f, rope_factor = 32.0f;
    float rope_beta_fast = 32.0f, rope_beta_slow = 1.0f;
    int rope_original_max_position = 4096;
    bool present() const { return n_layers > 0 && hidden > 0; }
};

// ── Per-kernel taps, for the walk against mlx-lm ────────────────────────────
struct Tap {
    const char* name;
    std::uint8_t out_bind;
    int width;
};

bool tap_for(const gptoss::Dispatch& d, const GptOssGeometry& g, Tap& out) {
    switch (d.kind) {
        case Kind::EmbedGather:   out = {"embed",       4, g.hidden};   return true;
        case Kind::AttnNorm:      out = {"attn_norm",   2, g.hidden};   return true;
        case Kind::QmvQ:          out = {"q_proj",      4, g.q_dim()};  return true;
        case Kind::QmvK:          out = {"k_proj",      4, g.kv_dim()}; return true;
        case Kind::QmvV:          out = {"v_proj",      4, g.kv_dim()}; return true;
        case Kind::RopeQ:         out = {"rope_q",      0, g.q_dim()};  return true;
        case Kind::RopeK:         out = {"rope_k",      0, g.kv_dim()}; return true;
        case Kind::SdpaSink:      out = {"sdpa",        3, g.q_dim()};  return true;
        case Kind::QmvO:          out = {"o_proj",      4, g.hidden};   return true;
        case Kind::AttnResidual:  out = {"attn_resid",  2, g.hidden};   return true;
        case Kind::FfnNorm:       out = {"ffn_norm",    2, g.hidden};   return true;
        case Kind::RouterGemv:    out = {"router",      4, g.n_experts}; return true;
        case Kind::ExpertGate:
            out = {"expert_gate", 4, g.experts_per_token * g.intermediate}; return true;
        case Kind::ExpertUp:
            out = {"expert_up",   4, g.experts_per_token * g.intermediate}; return true;
        case Kind::ExpertSwiGlu:
            out = {"expert_act",  2, g.experts_per_token * g.intermediate}; return true;
        case Kind::ExpertDown:
            out = {"expert_out",  4, g.experts_per_token * g.hidden};   return true;
        case Kind::ExpertCombine: out = {"moe",         2, g.hidden};   return true;
        case Kind::FfnResidual:   out = {"layer_out",   2, g.hidden};   return true;
        case Kind::FinalRms:      out = {"final_norm",  2, g.hidden};   return true;
        case Kind::LmHead:        out = {"logits",      4, g.vocab};    return true;
        default: return false;
    }
}

void dump_gptoss_taps(const std::vector<gptoss::Dispatch>& dag, const GptOssGeometry& g,
                      const ScratchColoring& coloring, const std::vector<SlotHandle>& pool) {
    const int n_pool = static_cast<int>(pool.size());
    auto color_of = [&](std::size_t di, std::uint8_t bind_index) {
        for (const auto& sb : coloring.per_dispatch[di]) {
            if (sb.bind_index == bind_index) return int(sb.color);
        }
        return -1;
    };
    // The in-place kinds (both ropes) share a buffer with the tap before them,
    // so only a colour's final writer is named -- publishing the earlier tensor
    // under the earlier name would read as a divergence that is really the dump
    // lying.
    std::vector<int> last_writer(std::size_t(n_pool < 0 ? 0 : n_pool), -1);
    for (std::size_t di = 0; di < dag.size(); ++di) {
        Tap tap{};
        if (!tap_for(dag[di], g, tap)) continue;
        const int c = color_of(di, tap.out_bind);
        if (c >= 0 && c < n_pool) last_writer[std::size_t(c)] = int(di);
    }
    for (std::size_t di = 0; di < dag.size(); ++di) {
        Tap tap{};
        if (!tap_for(dag[di], g, tap)) continue;
        const int c = color_of(di, tap.out_bind);
        if (c < 0 || c >= n_pool || !pool[std::size_t(c)].valid()) continue;
        if (last_writer[std::size_t(c)] != int(di)) continue;
        const std::string name = dag[di].layer < 0
            ? std::string(tap.name)
            : std::to_string(dag[di].layer) + "." + tap.name;
        dump_golden_bf16(name, pool[std::size_t(c)].contents(), 1, tap.width, tap.width);
    }
}

/// How many argument-table slots each kind's kernel actually reads.
int required_slots(Kind k) {
    switch (k) {
        case Kind::EmbedGather:  return 6;
        case Kind::AttnNorm: case Kind::FfnNorm: case Kind::FinalRms: return 4;
        case Kind::QmvQ: case Kind::QmvK: case Kind::QmvV: case Kind::QmvO:
            return 8;  // Qmv's seven plus the additive bias
        case Kind::LmHead:       return 7;
        case Kind::RouterGemv:   return 8;
        case Kind::ExpertGate: case Kind::ExpertUp: case Kind::ExpertDown:
            return 10;  // ... plus the expert ids and the input's per-slot stride
        case Kind::RopeQ: case Kind::RopeK: return 6;
        case Kind::KvAppend:     return 8;
        case Kind::SdpaSink:     return 15;
        case Kind::RouterTopK:   return 4;
        case Kind::ExpertSwiGlu: return 4;
        case Kind::ExpertCombine: return 4;
        case Kind::AttnResidual: case Kind::FfnResidual: return 4;
        case Kind::Argmax:       return 4;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string ckpt = argc > 1 ? argv[1] : std::string();
    if (ckpt.empty()) {
        const char* home = std::getenv("HOME");
        if (home != nullptr) ckpt = std::string(home) + "/.pie-bench/gptoss-20b-pie4";
    }
    std::string kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;

    // Before the checkpoint gate, because this needs no checkpoint and is the
    // one thing here that would otherwise never run on a machine without one.
    //
    // The attention width used to be the literal 64 that every released gpt-oss
    // happens to use, while the geometry read it from the config. A d=64
    // pipeline handed wider heads does not fail -- it strides past the end of
    // each head and writes zeros, which is a model that runs and says nothing.
    // There is no pipeline cache, so identity proves nothing; what proves the
    // width was read is the NAME the load fails on.
    {
        std::printf("[the attention width is the geometry's]\n");
        auto probe_ctx = RawMetalContext::create(16u << 20);
        if (!probe_ctx) {
            expect(false, "RawMetalContext::create");
        } else {
            GptOssGeometry odd;
            std::string gerr;
            if (!geometry_from_facts(Facts{}, odd, &gerr)) {
                expect(false, "geometry: " + gerr);
            } else {
                odd.head_dim = 96;
                GptOssPsos p_odd;
                std::string e_odd;
                const bool ok = build_gptoss_psos(*probe_ctx, kernels_dir, odd, p_odd, &e_odd);
                expect(!ok, "an uninstantiated head width (96) is refused");
                expect(!ok && e_odd.find("_d_96") != std::string::npos,
                       "and the refusal names the width that has no kernel: " + e_odd);
            }
        }
    }

    {
        std::string probe = ckpt + "/config.json";
        FILE* f = std::fopen(probe.c_str(), "rb");
        if (f == nullptr) {
            std::printf("gpt-oss forward: SKIP the checkpoint half (none at %s)\n",
                        ckpt.c_str());
            return failures == 0 ? 0 : 1;
        }
        std::fclose(f);
    }
    std::printf("gpt-oss forward (%s)\n", ckpt.c_str());

    // A real prompt, not a lone token: at position 0 every rotation is the
    // identity and nothing is read from the KV cache, so a one-token pass is the
    // one case that proves nothing about RoPE or attention.
    // "The capital of France is Paris. The capital of Japan is"
    std::vector<int> ids{976, 9029, 328, 10128, 382, 12650, 13, 623, 9029, 328, 10198, 382};
    bool default_prompt = true;
    if (const char* env = std::getenv("PIE_GO_TOKENS"); env != nullptr && *env != '\0') {
        ids.clear();
        for (const char* p = env; *p != '\0';) {
            char* end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end == p) break;
            ids.push_back(int(v));
            p = (*end == ',') ? end + 1 : end;
        }
        if (ids.empty()) ids.push_back(976);
        default_prompt = false;
    }
    std::printf("    (teacher-forcing %zu tokens)\n", ids.size());

    GptOssGeometry g;
    std::string err;
    if (!geometry_from_facts(Facts{}, g, &err)) {
        std::printf("  FAIL  geometry: %s\n", err.c_str());
        return 1;
    }
    const int max_ctx = 512;
    g.kv_max_ctx = max_ctx;

    // 11 GB of weights plus KV and the pool, on a 32 GB machine whose GPU wired
    // limit is a fraction of that. 16 fits; 20 does not.
    auto ctx = RawMetalContext::create(std::size_t(16) << 30);
    if (!ctx) {
        std::printf("  FAIL  RawMetalContext::create\n");
        return 1;
    }

    pie_loader::LoadPlan plan;
    try {
        plan = compile_load_plan(ckpt, metal_device_target(), "gpt_oss",
                                 pie::metal::model::ContractFacts{});
    } catch (const std::exception& e) {
        std::printf("  FAIL  compile_load_plan: %s\n", e.what());
        return 1;
    }
    expect(true, "the contract compiles a load plan for the real checkpoint");

    BoundGptOss b;
    std::shared_ptr<void> weight_mapping;
    try {
        const auto storage = plan.view();
        auto view = std::make_shared<pie_loader::CheckpointSource>(storage);
        StagedWeights staged =
            stage_plan_weights(*ctx, std::move(view), plan, storage.memory.persistent_bytes);
        b.weights = std::move(staged.weights);
        // The mapping has to outlive `b.weights`: when the checkpoint places
        // its tensors where a device pointer may point, those slots are slices
        // of the checkpoint's own mmap rather than copies in the heap, and
        // dropping `staged` here would unmap them under the GPU.
        weight_mapping = std::move(staged.weight_mapping);
    } catch (const std::exception& e) {
        std::printf("  FAIL  stage_plan_weights: %s\n", e.what());
        return 1;
    }
    std::printf("    (%zu tensors staged)\n", b.weights.size());
    expect(!b.weights.empty(), "the checkpoint's tensors are staged into the heap");

    // The router's quantization width comes off the staged tensors, not off
    // `config.json` -- checkpoints ship both 4- and 8-bit routers and the two
    // kernels read incompatible packings.
    g.router_bits = router_bits_from_weights(b.weights);
    std::printf("    (router is %d-bit)\n", g.router_bits);
    expect(g.router_bits == 4 || g.router_bits == 8,
           "the router's quantization width is solvable from the checkpoint");

    b.kv.resize(g.n_layers);
    for (int L = 0; L < g.n_layers; ++L) {
        const std::size_t bytes = gptoss_kv_bytes_per_layer(g, max_ctx, 2);
        b.kv[L].k = ctx->heap_alloc(bytes);
        b.kv[L].v = ctx->heap_alloc(bytes);
    }

    const auto dag = build_gptoss_dag(g, /*with_argmax=*/false);
    const ScratchPlan sp = build_gptoss_scratch(dag, g);
    const ScratchColoring coloring =
        color_gptoss_scratch(dag, sp, /*no_recycle=*/golden_taps_enabled());
    expect(coloring.hazard_free, "the activation colouring is hazard-free");

    b.pool.resize(coloring.colors_used);
    const std::size_t widest = std::size_t(gptoss_widest_elems(g)) * 2;
    for (int c = 0; c < coloring.colors_used; ++c) b.pool[c] = ctx->heap_alloc(widest);

    b.io.resize(kIoSlotCount);
    for (int i = 0; i < kIoSlotCount; ++i) b.io[i] = ctx->heap_alloc(4096);

    GptOssPsos psos;
    DecodeStepPsos base;
    if (!build_gptoss_psos(*ctx, kernels_dir, g, psos, &err) ||
        // gpt-oss's shared table is affine b4/g64; its mxfp4 entrypoints are
        // named directly in `gptoss/kernels.cpp`.
        !load_decode_psos(*ctx, kernels_dir, base, pie::metal::AffineFormat{4, 64},
                          /*with_argmax=*/false, &err)) {
        std::printf("  FAIL  pipelines: %s\n", err.c_str());
        return 1;
    }

    const int bound = bind_gptoss_consts(*ctx, dag, g);
    bind_gptoss_dag(*ctx, b, dag, g, coloring);
    std::printf("    (%d constants bound over %zu dispatches)\n", bound, dag.size());

    bool all_bound = true;
    int missing = 0;
    for (const gptoss::Dispatch& d : dag) {
        const int n = required_slots(d.kind);
        for (int slot = 0; slot < n; ++slot) {
            if (!ctx->arg_slot_is_bound(d.ordinal, std::uint8_t(slot))) {
                if (missing < 16) {
                    std::printf("    missing ord=%d kind=%d layer=%d slot=%d\n", d.ordinal,
                                int(d.kind), d.layer, slot);
                }
                ++missing;
                all_bound = false;
            }
        }
    }
    if (missing > 0) std::printf("    (%d unbound slots)\n", missing);
    expect(all_bound, "every slot the kernels read is bound");

    ctx->make_resident();

    int logits_color = -1;
    for (const auto& sb : coloring.per_dispatch.back()) {
        if (sb.bind_index == (std::uint8_t)bind::GoQmv::Out) logits_color = sb.color;
    }
    expect(logits_color >= 0, "the final dispatch has an output to read");
    if (logits_color < 0) return 1;
    const auto* logits = static_cast<const std::uint16_t*>(b.pool[logits_color].contents());

    for (std::size_t t = 0; t < ids.size(); ++t) {
        *static_cast<std::int32_t*>(b.io[int(IoSlot::TokenId)].contents()) = ids[t];
        *static_cast<std::int32_t*>(b.io[int(IoSlot::Position)].contents()) = int(t);
        *static_cast<std::int32_t*>(b.io[int(IoSlot::SeqLen)].contents()) = int(t) + 1;
        ctx->run_step([&](StepEncoder& se) {
            encode_gptoss_step(se, dag, g, base, psos);
        });
    }

    if (golden_taps_enabled()) {
        dump_gptoss_taps(dag, g, coloring, b.pool);
        std::printf("    (taps -> %s)\n", golden_tap_dir().c_str());
    }

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
    std::printf("    (nonzero %d/%d, argmax %d, max logit %.4f)\n", nonzero, g.vocab, argmax,
                best);
    std::vector<int> idx(g.vocab);
    for (int i = 0; i < g.vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int c) {
        return from_bf16(logits[a]) > from_bf16(logits[c]);
    });
    std::printf("    top5");
    for (int i = 0; i < 5; ++i) std::printf(" (%d, %.4f)", idx[i], from_bf16(logits[idx[i]]));
    std::printf("\n");

    expect(nonzero > g.vocab / 2, "the logits are populated, not a zeroed buffer");
    expect(nan_or_inf == 0, "and finite everywhere");

    // Greedy continuation. Per-tap cosine says where the arithmetic differs;
    // this says whether it matters. A 4-bit model's taps will never be bit-exact
    // against another implementation's, so the acceptance test is the tokens.
    {
        const int n_gen = 16;
        std::vector<int> got;
        int next = argmax;
        for (int s = 0; s < n_gen; ++s) {
            got.push_back(next);
            const int pos = int(ids.size()) + s;
            *static_cast<std::int32_t*>(b.io[int(IoSlot::TokenId)].contents()) = next;
            *static_cast<std::int32_t*>(b.io[int(IoSlot::Position)].contents()) = pos;
            *static_cast<std::int32_t*>(b.io[int(IoSlot::SeqLen)].contents()) = pos + 1;
            ctx->run_step([&](StepEncoder& se) {
                encode_gptoss_step(se, dag, g, base, psos);
            });
            float bv = -1e30f;
            int bi = 0;
            for (int i = 0; i < g.vocab; ++i) {
                const float v = from_bf16(logits[i]);
                if (v > bv) { bv = v; bi = i; }
            }
            next = bi;
        }
        std::printf("    greedy");
        for (int t : got) std::printf(" %d", t);
        std::printf("\n");
        // " Tokyo. The capital of" -- mlx-lm's own greedy continuation, and the
        // part of it the prompt actually forces. What follows is a free choice
        // between countries, where a logit difference of any size picks a
        // different one; asserting it would be asserting mlx's rounding.
        const std::vector<int> want{40510, 13, 623, 9029, 328};
        if (default_prompt) {
            bool same = got.size() >= want.size();
            for (std::size_t i = 0; same && i < want.size(); ++i) {
                same = got[i] == want[i];
            }
            expect(same, "greedy decoding agrees with mlx-lm while the prompt forces it");
        }
    }

    if (const char* env = std::getenv("PIE_GO_BENCH"); env != nullptr && *env != '\0') {
        const int steps = std::max(1, std::atoi(env));
        std::vector<double> gpu, enc;
        gpu.reserve(std::size_t(steps));
        enc.reserve(std::size_t(steps));
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < steps; ++s) {
            const int pos = int(ids.size()) + s;
            *static_cast<std::int32_t*>(b.io[int(IoSlot::TokenId)].contents()) = 1495;
            *static_cast<std::int32_t*>(b.io[int(IoSlot::Position)].contents()) = pos;
            *static_cast<std::int32_t*>(b.io[int(IoSlot::SeqLen)].contents()) = pos + 1;
            const StepTiming t = ctx->run_step([&](StepEncoder& se) {
                encode_gptoss_step(se, dag, g, base, psos);
            });
            gpu.push_back(t.gpu_exec_ms);
            enc.push_back(t.encode_ms);
        }
        const double wall =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / double(steps);
        std::sort(gpu.begin(), gpu.end());
        std::sort(enc.begin(), enc.end());
        std::printf("    [bench] %d steps, %zu dispatches: gpu %.4f  encode %.4f"
                    "  wall %.4f ms/tok  %.1f tok/s\n",
                    steps, dag.size(), gpu[gpu.size() / 2], enc[enc.size() / 2], wall,
                    1000.0 / wall);
    }

    std::printf("\n==== gptoss_forward_test: %s ====\n",
                failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}

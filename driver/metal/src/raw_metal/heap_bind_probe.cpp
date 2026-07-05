// heap_bind_probe.cpp — verify delta's weight-name registry (heap_bind_names.cpp) against
// a REAL checkpoint, with NO Metal dependency. Catches naming / dtype / shape / tied-table
// mistakes BEFORE the headline run (the class of bug the team kept finding by inspection).
//
//   ./heap_bind_probe ~/models/4bit-tied/Qwen3.5-0.8B
//
// Checks:
//   1. Every tensor decode_weight_tensors() enumerates is PRESENT in the checkpoint.
//   2. Dtype sanity: quant .weight = U32, .scales/.biases = BF16; norms BF16/F32; the
//      dense GDN a/b = BF16; A_log F32; dt_bias BF16.
//   3. The tied lm_head is shared (embed_tokens.weight ABSENT) and staged once.
//   4. conv1d is bias-less (linear_attn.conv1d.bias absent on GDN layers).
//   5. weight_binds() references only tensors that exist (per-dispatch coverage).
//   6. Sum of staged bytes == plan_heap().weights_bytes (region sizing is correct).
//   7. model.visual.* present in the ckpt but EXCLUDED from the staged set.

#include <cstdio>
#include <set>
#include <string>

#include "decode_abi.hpp"
#include "heap_bind.hpp"
#include "heap_layout.hpp"
#include "safetensors_view.hpp"

using namespace pie_metal_driver::raw_metal;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& msg) {
    if (!ok) { printf("  FAIL: %s\n", msg.c_str()); ++g_fail; }
}

// expected safetensors dtype for a tensor name (by suffix convention).
const char* expect_dtype(const std::string& n) {
    auto ends = [&](const char* s) {
        const std::string e(s);
        return n.size() >= e.size() && n.compare(n.size() - e.size(), e.size(), e) == 0;
    };
    if (ends(".weight")) {
        // dense GDN a/b + every *.norm/layernorm .weight are BF16/F32; quant .weight U32.
        if (n.find("in_proj_a") != std::string::npos ||
            n.find("in_proj_b") != std::string::npos) return "BF16";
        if (n.find("conv1d.weight") != std::string::npos) return "BF16";
        if (n.find("norm") != std::string::npos || n.find("layernorm") != std::string::npos)
            return "";  // BF16 or F32 — accept either
        return "U32";   // packed 4-bit quant
    }
    if (ends(".scales") || ends(".biases")) return "BF16";
    if (ends("A_log"))  return "F32";
    if (ends("dt_bias")) return "BF16";
    return "";
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) { printf("usage: %s <hf_snapshot_dir>\n", argv[0]); return 2; }

    DecodeGeometry g;
    try {
        SafetensorsView v(argv[1]);
        printf("SafetensorsView OK: %zu tensors\n", v.size());

        const auto tensors = decode_weight_tensors(g);
        printf("registry enumerates %zu text-decode weight tensors\n", tensors.size());

        // (1)+(2) presence + dtype.
        size_t total = 0;
        for (const auto& n : tensors) {
            auto rt = v.try_get(n);
            if (!rt) { check(false, "absent: " + n); continue; }
            total += rt->nbytes;
            const char* exp = expect_dtype(n);
            if (exp && *exp && rt->dtype != exp)
                check(false, n + " dtype=" + rt->dtype + " expected " + exp);
        }
        printf("staged weight bytes: %.1f MB (%zu tensors present)\n",
               total / (1024.0 * 1024.0), tensors.size());

        // (3) tied lm_head shared; embed_tokens absent on both candidate paths.
        check(v.has("lm_head.weight"), "lm_head.weight present");
        check(!v.has("model.embed_tokens.weight"), "embed_tokens.weight absent (tied)");
        check(!v.has("model.language_model.embed_tokens.weight"),
              "language_model.embed_tokens.weight absent (tied)");

        // (4) conv1d bias-less on a sample GDN layer (layer 0 is GDN).
        check(!v.has("model.language_model.layers.0.linear_attn.conv1d.bias"),
              "conv1d has no bias (GdnCore::ConvB -> zeroed slot)");

        // (5) weight_binds() per dispatch references only existing tensors.
        std::set<std::string> referenced;
        for (int kk = 0; kk <= static_cast<int>(Kernel::Argmax); ++kk) {
            const auto kind = static_cast<Kernel>(kk);
            const bool singleton = kind == Kernel::EmbedGather || kind == Kernel::FinalRms ||
                                   kind == Kernel::QmvLmHead || kind == Kernel::Argmax;
            for (int L = -1; L < g.n_layers; ++L) {
                if (L < 0) { if (!singleton) continue; }    // -1 only for singletons
                else       { if (singleton) continue; }     // singletons only at -1
                if (L >= 0) {
                    // skip kinds that don't occur in this layer type
                    const bool full = DecodeGeometry::is_full_attn(L);
                    const bool attn_only =
                        kind == Kernel::QmvQ || kind == Kernel::QmvK || kind == Kernel::QmvV ||
                        kind == Kernel::QmvO || kind == Kernel::QNorm || kind == Kernel::KNorm;
                    const bool gdn_only =
                        kind == Kernel::QmvIn || kind == Kernel::QmvInZ || kind == Kernel::QmvOut ||
                        kind == Kernel::GdnInA || kind == Kernel::GdnInB || kind == Kernel::GdnCore ||
                        kind == Kernel::GatedRms;
                    if (attn_only && !full) continue;
                    if (gdn_only && full) continue;
                }
                for (const auto& wb : weight_binds(kind, L, g)) {
                    referenced.insert(wb.tensor);
                    if (!v.has(wb.tensor))
                        check(false, "weight_binds refs missing tensor: " + wb.tensor);
                }
            }
        }
        // every staged tensor should be referenced by some dispatch (no dead weights).
        for (const auto& n : tensors)
            check(referenced.count(n) > 0, "staged-but-unbound tensor: " + n);

        // (6) byte accounting vs heap_layout Weights region.
        const HeapPlan plan = plan_heap(g, total);
        printf("plan_heap Weights region: %.1f MB (total heap %.1f MB)\n",
               plan.weights_bytes / (1024.0 * 1024.0), plan.total / (1024.0 * 1024.0));
        check(plan.weights_bytes == align_up(total), "weights_bytes == align_up(sum)");

        // (7) vision tower present but excluded.
        bool any_visual = false;
        for (const auto& n : v.names())
            if (n.rfind("model.visual.", 0) == 0) { any_visual = true; break; }
        if (any_visual) {
            for (const auto& n : tensors)
                check(n.rfind("model.visual.", 0) != 0, "no model.visual.* in staged set");
            printf("model.visual.* present in ckpt -> correctly excluded\n");
        }

        if (g_fail == 0) { printf("HEAP_BIND_PROBE_OK\n"); return 0; }
        printf("HEAP_BIND_PROBE_FAIL: %d check(s)\n", g_fail);
        return 1;
    } catch (const std::exception& e) {
        printf("FAIL: %s\n", e.what());
        return 1;
    }
}

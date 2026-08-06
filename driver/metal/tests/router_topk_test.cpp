// `router_topk` at Qwen3-MoE's expert count — a REAL on-device dispatch.
//
// This test exists because the kernel was wrong and nothing caught it. The
// selection reduced with `simd_max`/`simd_min` and wrote under `simd_lid == 0`,
// which is a threadgroup-wide reduction only while the threadgroup IS one
// simdgroup. gpt-oss has exactly 32 experts, so it always was, and the bug was
// invisible. Qwen3-MoE has 128: four simdgroups, four lanes satisfying
// `simd_lid == 0`, and four different quarter-local maxima racing to write
// `expert_ids[r]`. The winner is whichever store landed last.
//
// So the cases below are chosen to make that race produce a WRONG answer rather
// than a merely nondeterministic one:
//
//   - row 0 ranks the experts by index, putting the entire true top-8 in the
//     LAST simdgroup. The old kernel would report 127, 95, 63 and 31 -- three
//     quarter-maxima that are not in the top 8 at all.
//   - row 1 reverses that, putting the true top-8 in the FIRST simdgroup, so a
//     kernel that happened to favour the highest-numbered simdgroup's store
//     fails here even though it passes row 0.
//
// The 32-expert case is kept as the regression guard: the fix must not move
// gpt-oss, whose routing is a shipped numerical result.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mtl4_context.hpp"
#include "model/gptoss/kernels.hpp"

using pie::metal::Grid;
using pie::metal::Pso;
using pie::metal::RawMetalContext;
using pie::metal::SlotHandle;
using pie::metal::Threadgroup;
using pie::metal::gptoss::router_topk_dispatch;
using pie::metal::shared_kernels::RouterParams;

namespace {

int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

std::uint16_t bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

float from_bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

/// What the kernel is supposed to compute, in the clearest possible form: sort
/// by value (ties to the lower index), take k, softmax those k.
struct Expected {
    std::vector<int> ids;
    std::vector<float> weights;
};

Expected reference(const std::vector<float>& logits, int k) {
    std::vector<int> order(logits.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return logits[std::size_t(a)] > logits[std::size_t(b)];
    });
    Expected e;
    e.ids.assign(order.begin(), order.begin() + k);
    float mx = -3.0e38f;
    for (int id : e.ids) mx = std::fmax(mx, logits[std::size_t(id)]);
    float sum = 0;
    for (int id : e.ids) {
        const float w = std::exp(logits[std::size_t(id)] - mx);
        e.weights.push_back(w);
        sum += w;
    }
    for (float& w : e.weights) w /= sum;
    return e;
}

/// One dispatch of `router_topk` over `rows x n_experts` logits.
struct Observed {
    std::vector<int> ids;
    std::vector<float> weights;
};

Observed run_router(RawMetalContext& ctx, const Pso& pso, const std::vector<float>& logits,
                    int n_experts, int k, int rows, int ordinal) {
    SlotHandle in = ctx.create_standalone_buffer(logits.size() * sizeof(std::uint16_t));
    SlotHandle ids = ctx.create_standalone_buffer(std::size_t(rows * k) * sizeof(int));
    SlotHandle wts = ctx.create_standalone_buffer(std::size_t(rows * k) * sizeof(std::uint16_t));
    SlotHandle params = ctx.create_standalone_buffer(sizeof(RouterParams));

    auto* in_data = static_cast<std::uint16_t*>(in.contents());
    for (std::size_t i = 0; i < logits.size(); ++i) in_data[i] = bf16(logits[i]);
    // Poison the outputs. A kernel that skips a round leaves the sentinel
    // behind instead of inheriting a plausible value from a previous test.
    std::memset(ids.contents(), 0xFF, ids.size);
    std::memset(wts.contents(), 0, wts.size);
    *static_cast<RouterParams*>(params.contents()) =
        RouterParams{std::uint32_t(n_experts), std::uint32_t(k)};

    ctx.arg_bind_ordinal(ordinal, 0, in);
    ctx.arg_bind_ordinal(ordinal, 1, ids);
    ctx.arg_bind_ordinal(ordinal, 2, wts);
    ctx.arg_bind_ordinal(ordinal, 3, params);
    ctx.make_resident();

    Grid grid{};
    Threadgroup tg{};
    router_topk_dispatch(n_experts, grid, tg, rows);
    ctx.run_step([&](auto& encoder) {
        encoder.set_pso(pso);
        encoder.set_argtable_ordinal(ordinal);
        encoder.dispatch(grid, tg);
    });

    Observed o;
    const auto* id_data = static_cast<const int*>(ids.contents());
    const auto* w_data = static_cast<const std::uint16_t*>(wts.contents());
    for (int i = 0; i < rows * k; ++i) {
        o.ids.push_back(id_data[i]);
        o.weights.push_back(from_bf16(w_data[i]));
    }
    return o;
}

/// bf16 carries 8 mantissa bits, so a weight is only good to ~1e-2 relative.
/// The IDS, though, are exact -- and the ids are what the bug corrupted.
bool weights_match(const std::vector<float>& got, const std::vector<float>& want, int off, int k) {
    for (int i = 0; i < k; ++i) {
        if (std::fabs(got[std::size_t(off + i)] - want[std::size_t(i)]) > 2e-2f) return false;
    }
    return true;
}

void check_case(RawMetalContext& ctx, const Pso& pso, const char* name,
                const std::vector<std::vector<float>>& rows_logits, int n_experts, int k,
                int ordinal) {
    std::vector<float> flat;
    for (const auto& r : rows_logits) flat.insert(flat.end(), r.begin(), r.end());
    const int rows = static_cast<int>(rows_logits.size());
    const Observed got = run_router(ctx, pso, flat, n_experts, k, rows, ordinal);

    for (int r = 0; r < rows; ++r) {
        const Expected want = reference(rows_logits[std::size_t(r)], k);
        bool ids_ok = true;
        for (int i = 0; i < k; ++i) {
            if (got.ids[std::size_t(r * k + i)] != want.ids[std::size_t(i)]) ids_ok = false;
        }
        if (!ids_ok) {
            std::printf("        row %d ids got  [", r);
            for (int i = 0; i < k; ++i) std::printf("%d ", got.ids[std::size_t(r * k + i)]);
            std::printf("]\n        row %d ids want [", r);
            for (int i = 0; i < k; ++i) std::printf("%d ", want.ids[std::size_t(i)]);
            std::printf("]\n");
        }
        expect(ids_ok, std::string(name) + ": row " + std::to_string(r) + " top-" +
                           std::to_string(k) + " ids exact");
        expect(weights_match(got.weights, want.weights, r * k, k),
               std::string(name) + ": row " + std::to_string(r) + " softmax weights");

        float sum = 0;
        for (int i = 0; i < k; ++i) sum += got.weights[std::size_t(r * k + i)];
        expect(std::fabs(sum - 1.0f) < 2e-2f,
               std::string(name) + ": row " + std::to_string(r) + " weights sum to 1");
    }
}

}  // namespace

int main() {
    std::printf("router_topk_test — top-k routing beyond one simdgroup\n");

    std::string kernels_dir;
    if (const char* kd = std::getenv("PIE_METAL_KERNELS_DIR")) kernels_dir = kd;
#ifdef PIE_METAL_KERNELS_DIR_FOR_TEST
    if (kernels_dir.empty()) kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;
#endif
    if (!expect(!kernels_dir.empty(), "kernels_dir resolved")) return 1;

    auto ctx = RawMetalContext::create(16u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) return 1;

    std::string err;
    const Pso pso =
        ctx->compile_pso_from_file(kernels_dir + "/moe_route.metal",
                                   "router_topk_bfloat16", &err);
    if (!expect(pso.valid(), "router_topk_bfloat16 compiles: " + err)) return 1;

    // ── Qwen3-MoE: 128 experts, 8 per token. Four simdgroups. ──
    {
        // Ascending: the true top-8 (127..120) sits entirely in simdgroup 3.
        std::vector<float> ascending(128);
        for (int i = 0; i < 128; ++i) ascending[std::size_t(i)] = float(i) * 0.0625f;
        // Descending: the true top-8 (0..7) sits entirely in simdgroup 0.
        std::vector<float> descending(128);
        for (int i = 0; i < 128; ++i) descending[std::size_t(i)] = -float(i) * 0.0625f;
        check_case(*ctx, pso, "n=128 k=8", {ascending, descending}, 128, 8, 91000);
    }

    // A spread case: one winner planted in each quarter, so a correct kernel
    // must cross simdgroup boundaries in a single round-to-round sequence
    // rather than draining one quarter at a time.
    {
        std::vector<float> spread(128, 0.0f);
        for (int i = 0; i < 128; ++i) spread[std::size_t(i)] = -1.0f - float(i) * 0.0078125f;
        const int planted[] = {100, 5, 70, 40, 127, 33, 12, 99};
        for (int rank = 0; rank < 8; ++rank) {
            spread[std::size_t(planted[rank])] = 8.0f - float(rank);
        }
        check_case(*ctx, pso, "n=128 k=8 spread", {spread}, 128, 8, 91100);
    }

    // ── gpt-oss regression: 32 experts, 4 per token. One simdgroup. ──
    {
        std::vector<float> logits(32);
        for (int i = 0; i < 32; ++i) {
            logits[std::size_t(i)] = std::sin(float(i) * 1.37f) * 4.0f;
        }
        check_case(*ctx, pso, "n=32 k=4 (gpt-oss)", {logits}, 32, 4, 91200);
    }

    // A non-multiple-of-32 width: `router_topk_dispatch` pads the threadgroup
    // up, so lanes past `n_experts` exist and must not be selectable.
    {
        std::vector<float> logits(48);
        for (int i = 0; i < 48; ++i) {
            logits[std::size_t(i)] = std::cos(float(i) * 0.91f) * 3.0f;
        }
        check_case(*ctx, pso, "n=48 k=4 (padded threadgroup)", {logits}, 48, 4, 91300);
    }

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

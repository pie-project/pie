// Which routed-expert form a Kimi-K3 MoE layer binds to, and the three ways
// that decision refuses.
//
// The refusals are the point. Before #1740's group existed, `bind_kimi_k3`
// hard-required the bf16 stacks, so a streaming contract failed to load loudly
// and correctly. Relaxing that require is what lets a streamed contract in --
// and a relaxation that let too much in would replace a load failure with a
// quiet wrong answer: a half-published stack whose missing half is read as
// null, or a group the router indexes past the end of. Neither of those
// crashes. Both produce plausible numbers.
//
// Device-free on purpose, and that is why the decision is its own translation
// unit: `LoadedModel` and `GroupStreamCache` cannot be built without a GPU, so
// a test that went through `bind_kimi_k3` would be a GPU test of arithmetic
// that has nothing to do with a GPU.

#include "model/kimi_k3/kimi_k3_expert_binding.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

using pie_cuda_driver::model::kimi_k3_use_streamed_experts;

const std::string kGroup = "model.layers.7.block_sparse_moe.experts";

bool decide(bool gate_up, bool down, bool found, std::uint32_t arity,
            int num_experts) {
    return kimi_k3_use_streamed_experts(gate_up, down, found, arity,
                                        num_experts, /*layer=*/7, kGroup);
}

/// Run `decide` expecting a throw, and check the message names the reason.
void check_refuses(bool gate_up, bool down, bool found, std::uint32_t arity,
                   int num_experts, const std::string& needle,
                   const std::string& what) {
    try {
        decide(gate_up, down, found, arity, num_experts);
        check(false, what + ": accepted, but it must refuse");
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        check(msg.find(needle) != std::string::npos,
              what + ": refused with \"" + msg + "\", which does not say \"" +
                  needle + "\"");
    }
}

}  // namespace

int main() {
    // ── Both forms are optional, and each is accepted alone ────────────
    //
    // This pair is the whole change: before it, the right-hand case threw.
    check(!decide(/*gate_up=*/true, /*down=*/true, /*found=*/false, 0, 896),
          "stacked experts bind to the stacked path");
    check(decide(/*gate_up=*/false, /*down=*/false, /*found=*/true, 896, 896),
          "a group of the right arity binds to the streamed path");

    // The stacks win when both forms somehow arrived. A contract publishes one
    // or the other, so this cannot happen from `author_kimi_k3` -- but if it
    // ever did, the stacked path is the one that needs no page-in.
    check(!decide(true, true, /*found=*/true, 896, 896),
          "stacks take precedence over a group that is also present");

    // ── Never neither ──────────────────────────────────────────────────
    //
    // The load failure this ticket relaxed, narrowed rather than deleted: with
    // no stacks and no group there is nothing for the per-expert loop to loop
    // over, and the layer would read a null weight.
    check_refuses(false, false, /*found=*/false, 0, 896, "neither",
                  "no stacks and no group");

    // ── Arity is checked against the config ────────────────────────────
    //
    // The forward pages in by expert id straight out of the router, so a short
    // group is an out-of-range page-in and a long one hides a tail the router
    // can never reach. Both directions refuse.
    check_refuses(false, false, /*found=*/true, /*arity=*/895, 896,
                  "895 experts but the config says 896",
                  "a group one expert short");
    check_refuses(false, false, /*found=*/true, /*arity=*/897, 896,
                  "897 experts but the config says 896",
                  "a group one expert long");
    check_refuses(false, false, /*found=*/true, /*arity=*/0, 896,
                  "0 experts but the config says 896",
                  "an empty group");

    // ── Half a stack is not a form ─────────────────────────────────────
    //
    // Deciding on `gate_up` alone -- which is what the other families do --
    // would send the `down`-only case down the streamed path and leave the
    // published `gate_up` unread, or take the `gate_up`-only case into the
    // stacked path with a null `down`.
    check_refuses(/*gate_up=*/true, /*down=*/false, /*found=*/true, 896, 896,
                  "experts.gate_up_proj without experts.down_proj",
                  "gate_up published without down");
    check_refuses(/*gate_up=*/false, /*down=*/true, /*found=*/true, 896, 896,
                  "experts.down_proj without experts.gate_up_proj",
                  "down published without gate_up");

    if (failures == 0) {
        std::cout << "kimi_k3_expert_binding: all checks passed\n";
    }
    return failures == 0 ? 0 : 1;
}

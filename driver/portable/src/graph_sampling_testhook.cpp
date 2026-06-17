// Test-only entry point exercising `build_sampling_outputs()` in isolation.
//
// Linked into `pie_driver_portable_lib` and called from a `#[cfg(test)]`
// Rust test in pie-server (`cargo test -p pie-server`, which runs in CI;
// the driver/portable ctest targets do not — see server/build.rs, which
// only builds the `pie_driver_portable_lib` target).
//
// It builds the uniform-top-K sampling subgraph over a synthetic
// `[vocab, n_slots]` logits tensor and reports the slot dimension
// (`ne[1]`) of the resulting `top_k_probs` tensor. This lets the Rust test
// assert the gather reshape keys on the slot count (`n_slots`, which is
// `>= n_req` under pass-level speculation) rather than the request count:
// keying on `n_req` trips GGML's `nelements == ne0*ne1*ne2` assert
// whenever `n_slots != n_req`.
//
// No backend or compute is needed — `ggml_reshape_*` validates element
// counts at graph-construction time, which is exactly the invariant under
// test. In a release build no Rust code references this symbol, so the
// linker garbage-collects this object out of the final binary.

#include <cstddef>

#include <ggml.h>

#include "graph_common.hpp"

// C linkage so the Rust side can declare it without name mangling. Returns
// the slot dimension of `top_k_probs`, or a negative error code.
extern "C" int pie_portable_test_uniform_top_slot_dim(int n_req,
                                                      int n_slots,
                                                      int k,
                                                      int vocab) {
    using namespace pie_portable_driver;

    if (n_req <= 0 || n_slots <= 0 || k <= 0 || k > vocab) return -1;

    ggml_init_params params{};
    params.mem_size   = static_cast<std::size_t>(64) * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) return -2;

    ggml_cgraph* gf = ggml_new_graph(ctx);

    // Synthetic final logits: [vocab, n_slots]. A batch of n_req requests
    // produces n_slots sampling slots (n_slots >= n_req under speculation).
    ggml_tensor* logits =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab, n_slots);

    Executor::BatchPlan plan;
    plan.all_greedy         = false;
    plan.uniform_top_sample = true;
    plan.uniform_top_k      = k;
    plan.reqs.resize(static_cast<std::size_t>(n_req));
    for (auto& r : plan.reqs) {
        r.sampler.temperature = 0.7f;
    }

    GraphResult res{};
    res.gf = gf;
    build_sampling_outputs(ctx, gf, logits, plan, res);

    const int slot_dim =
        res.top_k_probs != nullptr
            ? static_cast<int>(res.top_k_probs->ne[1])
            : -3;

    ggml_free(ctx);
    return slot_dim;
}

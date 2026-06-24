#include "raw_metal/raw_metal_executor.hpp"

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include <pie_schema/response_builder.hpp>
#include <pie_schema/view.hpp>

namespace pie_metal_driver::raw_metal {

// ── delta-owned compute state ────────────────────────────────────────────────
//
// SKELETON: empty. delta fills this with the raw-Metal pipeline that
// decode_run.cpp::main() builds once — RawMetalContext + resident heap, the
// built decode DAG, bound weight/state/KV/consts SlotHandles, the loaded PSO
// table, and the scratch pool — plus a `setup(ckpt_dir, kernels_dir)` (the
// decode_run prologue) and a per-token `step(token_id, position)` that runs one
// run_step and exposes IO.Logits (f32[vocab]). State (conv/recurrent/KV) lives
// here and accumulates in-place across run_forward calls (prefill→decode
// seamless).
struct RawMetalExecutor::Impl {};

RawMetalExecutor::RawMetalExecutor(std::string checkpoint_dir,
                                   std::string kernels_dir,
                                   std::uint32_t vocab_size)
    : impl_(nullptr),
      checkpoint_dir_(std::move(checkpoint_dir)),
      kernels_dir_(std::move(kernels_dir)),
      vocab_size_(vocab_size) {}

RawMetalExecutor::~RawMetalExecutor() = default;

void RawMetalExecutor::run_forward(const pie_driver::PieForwardRequestView& req,
                                   pie_driver::ResponseBuilder& builder,
                                   pie_driver::PieForwardResponseView& out) {
    // ── SKELETON forward ──
    //
    // delta marshals the per-token loop here:
    //   for i in [0, n_total):
    //     impl_->step(token_ids[i], position_ids[i]);   // 1 run_step → IO.Logits
    //   then alpha samples the sampling-CSR slots from IO.Logits and packs them.
    //
    // Until then, emit a sampler-shaped stub (token 0 per sampling slot) so the
    // pipeline round-trips end-to-end. Shape mirrors the MLX executor's packing.
    const auto sampling_indptr = req.sampling_indptr.as<std::uint32_t>();
    const int n_req =
        sampling_indptr.empty() ? 0 : static_cast<int>(sampling_indptr.size()) - 1;

    std::vector<pie_driver::PerRequestOutput> per_req(n_req > 0 ? n_req : 0);
    for (int r = 0; r < n_req; ++r) {
        const std::uint32_t s0 = sampling_indptr[r];
        const std::uint32_t s1 = sampling_indptr[r + 1];
        per_req[r].tokens.assign(s1 - s0, 0u);
    }

    builder.build(per_req, out);
    out.num_requests = static_cast<std::uint32_t>(n_req);
}

std::unique_ptr<RawMetalExecutor> make_raw_metal_executor(
    const std::string& checkpoint_dir,
    const std::string& kernels_dir,
    std::uint32_t vocab_size) {
    try {
        return std::make_unique<RawMetalExecutor>(checkpoint_dir, kernels_dir,
                                                  vocab_size);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] raw-Metal executor init failed ("
                  << e.what() << ") — serving stub Forward\n";
        return nullptr;
    }
}

}  // namespace pie_metal_driver::raw_metal

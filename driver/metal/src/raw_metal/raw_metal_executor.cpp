#include "raw_metal/raw_metal_executor.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pie_driver_abi/response_builder.hpp>
#include <pie_driver_abi/view.hpp>

#include "raw_metal/decoder.hpp"

namespace pie_metal_driver::raw_metal {

// delta-owned compute state: one RawMetalDecoder packaging the whole raw_metal
// lifecycle (heap/DAG/PSOs/KV + GDN ping-pong). Persistent sequence state lives
// in its resident heap and accumulates in-place across run_forward calls, so
// prefill->decode is seamless.
struct RawMetalExecutor::Impl {
    RawMetalDecoder decoder;
};

RawMetalExecutor::RawMetalExecutor(std::string checkpoint_dir,
                                   std::string kernels_dir,
                                   std::uint32_t vocab_size)
    : impl_(std::make_unique<Impl>()),
      checkpoint_dir_(std::move(checkpoint_dir)),
      kernels_dir_(std::move(kernels_dir)),
      vocab_size_(vocab_size) {
    std::string err;
    if (!impl_->decoder.setup(checkpoint_dir_, kernels_dir_, DecodeGeometry{},
                              &err)) {
        throw std::runtime_error("RawMetalDecoder::setup failed: " + err);
    }
    vocab_size_ = static_cast<std::uint32_t>(impl_->decoder.vocab());
}

RawMetalExecutor::~RawMetalExecutor() = default;

void RawMetalExecutor::run_forward(const pie_driver::PieForwardRequestView& req,
                                   pie_driver::ResponseBuilder& builder,
                                   pie_driver::PieForwardResponseView& out) {
    const auto token_ids    = req.token_ids.as<std::uint32_t>();
    const auto position_ids = req.position_ids.as<std::uint32_t>();
    const auto samp_idx     = req.sampling_indices.as<std::uint32_t>();
    const auto samp_indptr  = req.sampling_indptr.as<std::uint32_t>();

    const int n_total = static_cast<int>(token_ids.size());
    const int n_req =
        samp_indptr.empty() ? 0 : static_cast<int>(samp_indptr.size()) - 1;

    // Fresh sequence (rs_slot NEW): the runtime restarts at absolute position 0.
    // Zero the GDN conv/recurrent state + KV ring before stepping. (batch=1
    // single-stream; multi-slot reset is the M>1 follow-on.)
    if (n_total > 0 && position_ids[0] == 0) {
        impl_->decoder.reset_state();
    }

    // Per-token M=1 inner loop. State accumulates in-place. Capture the greedy
    // token at each sampled row (sampling_indices selects token rows to sample,
    // exactly like the MLX executor's logit_rows). The bench is greedy/temp=0,
    // so argmax over the row's logits is the sampled token.
    std::vector<std::uint32_t> argmax_at(n_total > 0 ? n_total : 0, 0u);
    std::vector<char> is_sampled(n_total > 0 ? n_total : 0, 0);
    for (auto row : samp_idx) {
        if (row < static_cast<std::uint32_t>(n_total)) is_sampled[row] = 1;
    }

    // Per-token e2e latency attribution (env PIE_PROF_FWD=1). Splits the executor's
    // own cost into encode(build CB) / gpu_exec(commit->completion sync) / argmax
    // (host scan) / pack, accumulated across run_forward calls and dumped every 64
    // sampled tokens. Isolates the executor share of the ~2ms e2e-minus-kernel gap;
    // gpu_exec is the per-token commit->completion boundary (manager's prime suspect
    // for the cadence stall). Zero overhead when unset (steady_clock reads are ~ns).
    static const bool prof = [] {
        const char* e = std::getenv("PIE_PROF_FWD");
        return e && e[0] && e[0] != '0';
    }();
    static double acc_encode = 0, acc_gpu = 0, acc_argmax = 0, acc_pack = 0,
                  acc_step_wall = 0;
    static long prof_tokens = 0;
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    for (int i = 0; i < n_total; ++i) {
        if (prof) {
            auto s0 = clk::now();
            StepTiming st = impl_->decoder.step(token_ids[i], position_ids[i]);
            auto s1 = clk::now();
            acc_step_wall += ms(s1 - s0);
            acc_encode += st.encode_ms;
            acc_gpu += st.gpu_exec_ms;
            if (is_sampled[i]) {
                auto a0 = clk::now();
                argmax_at[i] = impl_->decoder.argmax();
                acc_argmax += ms(clk::now() - a0);
                ++prof_tokens;
            }
        } else {
            impl_->decoder.step(token_ids[i], position_ids[i]);
            if (is_sampled[i]) argmax_at[i] = impl_->decoder.argmax();
        }
    }

    auto pack_t0 = prof ? clk::now() : clk::time_point{};

    // Pack: group sampled tokens per request via the sampling CSR.
    std::vector<pie_driver::PerRequestOutput> per_req(n_req > 0 ? n_req : 0);
    for (int r = 0; r < n_req; ++r) {
        const std::uint32_t s0 = samp_indptr[r];
        const std::uint32_t s1 = samp_indptr[r + 1];
        per_req[r].tokens.reserve(s1 - s0);
        for (std::uint32_t s = s0; s < s1; ++s) {
            const std::uint32_t row = s < samp_idx.size() ? samp_idx[s] : 0u;
            per_req[r].tokens.push_back(
                row < static_cast<std::uint32_t>(n_total) ? argmax_at[row] : 0u);
        }
    }

    builder.build(per_req, out);
    out.num_requests = static_cast<std::uint32_t>(n_req);

    if (prof) {
        acc_pack += ms(clk::now() - pack_t0);
        if (prof_tokens >= 64) {
            const double n = double(prof_tokens);
            std::cerr << "[PIE_PROF_FWD] per-sampled-token avg (us):"
                      << " step_wall=" << acc_step_wall / n * 1e3
                      << " encode=" << acc_encode / n * 1e3
                      << " gpu_exec=" << acc_gpu / n * 1e3
                      << " argmax=" << acc_argmax / n * 1e3
                      << " pack=" << acc_pack / n * 1e3 << "  (n=" << prof_tokens
                      << ")\n";
            acc_encode = acc_gpu = acc_argmax = acc_pack = acc_step_wall = 0;
            prof_tokens = 0;
        }
    }
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
                  << e.what() << ") -- serving stub Forward\n";
        return nullptr;
    }
}

}  // namespace pie_metal_driver::raw_metal

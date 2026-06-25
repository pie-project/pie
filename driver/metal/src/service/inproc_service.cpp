#include "service/inproc_service.hpp"

#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include <pie_ipc/inproc_server.hpp>
#include <pie_driver_abi/response_builder.hpp>

#include "forward_executor.hpp"

namespace pie_metal_driver::service {

namespace {

// Build a single request's stub output, dispatching per sampler slot the
// same way driver/dummy does so the response stays aligned with what the
// runtime expects for each sampler kind. Tokens are 0, distributions and
// logprobs are zeroed — placeholder values until the MLX executor lands.
void fill_request(const pie_driver::PieForwardRequestView& fwd,
                  std::uint32_t s_lo,
                  std::uint32_t s_hi,
                  std::uint32_t vocab_size,
                  pie_driver::PerRequestOutput& pr) {
    const auto kinds = fwd.sampler_types.as<std::uint32_t>();
    const auto label_indptr = fwd.sampler_label_indptr.as<std::uint32_t>();

    for (std::uint32_t slot = s_lo; slot < s_hi; ++slot) {
        const std::uint32_t kind =
            slot < kinds.size() ? kinds[slot] : pie_driver::SAMPLER_MULTINOMIAL;
        switch (kind) {
            case pie_driver::SAMPLER_DISTRIBUTION: {
                std::vector<std::uint32_t> ids(8, 0);
                std::vector<float> probs = {0.5f,    0.25f,   0.125f,    0.0625f,
                                            0.03125f, 0.015625f, 0.0078125f, 0.0078125f};
                pr.dists.emplace_back(std::move(ids), std::move(probs));
                break;
            }
            case pie_driver::SAMPLER_RAW_LOGITS: {
                pr.logits.emplace_back(
                    static_cast<std::size_t>(vocab_size) * sizeof(float), 0u);
                break;
            }
            case pie_driver::SAMPLER_LOGPROB: {
                pr.logprobs.emplace_back(std::vector<float>{0.0f});
                break;
            }
            case pie_driver::SAMPLER_LOGPROBS: {
                std::uint32_t k = 0;
                if (slot + 1 < label_indptr.size()) {
                    k = label_indptr[slot + 1] - label_indptr[slot];
                }
                pr.logprobs.emplace_back(std::vector<float>(k, 0.0f));
                break;
            }
            case pie_driver::SAMPLER_ENTROPY: {
                pr.entropies.push_back(0.0f);
                break;
            }
            case pie_driver::SAMPLER_EMBEDDING:
                // Runtime filters Embedding out of the slot stream; emit
                // nothing to keep the response aligned.
                break;
            default:
                // Token-producing samplers (Multinomial / TopK / TopP / MinP
                // / TopKTopP). Stub returns token 0.
                pr.tokens.push_back(0);
                break;
        }
    }
}

}  // namespace

void InProcService::serve_forever(pie_driver::InProcServer& server) {
    // The builder's scratch backs the response-view slices and must outlive
    // each `send_response`; it lives for the whole serve loop here.
    pie_driver::ResponseBuilder response_builder;

    server.serve_forever(
        [&](std::uint32_t req_id,
            const pie_driver::PieInProcRequestView& req,
            pie_driver::PieInProcResponseView& out) {
            handle_request(req_id, req, out, response_builder);
        });
}

void InProcService::handle_request(std::uint32_t req_id,
                                   const pie_driver::PieInProcRequestView& req,
                                   pie_driver::PieInProcResponseView& out,
                                   pie_driver::ResponseBuilder& response_builder) {
    out.method = req.method;
    switch (req.method) {
        case pie_driver::PIE_METHOD_FORWARD: {
            ++handled_;
            const auto& fwd = req.forward;
            // Real compute path: delegate to the attached executor — the
            // default MLX-free raw-Metal pipeline, or the optional MLX
            // ModelGraph. Only taken once a model is loaded + attached.
            if (executor_ != nullptr) {
                executor_->run_forward(fwd, response_builder, out.forward);
                out.status = 0;
                break;
            }
            // Stub fallback (before a model loads): dummy tokens shaped like
            // the per-request sampler stream.
            const auto sampling_indptr = fwd.sampling_indptr.as<std::uint32_t>();
            const std::size_t n =
                sampling_indptr.empty() ? 0 : sampling_indptr.size() - 1;

            std::vector<pie_driver::PerRequestOutput> per_request(n);
            for (std::size_t r = 0; r < n; ++r) {
                fill_request(fwd, sampling_indptr[r], sampling_indptr[r + 1],
                             vocab_size_, per_request[r]);
            }
            response_builder.build(per_request, out.forward);
            out.status = 0;
            break;
        }
        case pie_driver::PIE_METHOD_HEALTH:
            out.status = 0;
            break;
        case pie_driver::PIE_METHOD_COPY_D2H:
        case pie_driver::PIE_METHOD_COPY_H2D:
        case pie_driver::PIE_METHOD_COPY_D2D:
        case pie_driver::PIE_METHOD_COPY_H2H:
        case pie_driver::PIE_METHOD_RS_COPY_D2H:
        case pie_driver::PIE_METHOD_RS_COPY_H2D:
        case pie_driver::PIE_METHOD_RS_COPY_D2D:
        case pie_driver::PIE_METHOD_RS_COPY_H2H:
            // No KV / recurrent-state cache yet (delta wires it).
            out.status = 4;
            break;
        case pie_driver::PIE_METHOD_LOAD_ADAPTER:
        case pie_driver::PIE_METHOD_SAVE_ADAPTER:
        case pie_driver::PIE_METHOD_ZO_INITIALIZE_ADAPTER:
        case pie_driver::PIE_METHOD_ZO_UPDATE_ADAPTER:
            // No adapter pool yet; no-op success.
            out.status = 0;
            break;
        default:
            std::cerr << "[pie-driver-metal] unknown method " << req.method
                      << " (req_id=" << req_id << ")\n";
            out.status = 2;
            break;
    }
}

}  // namespace pie_metal_driver::service

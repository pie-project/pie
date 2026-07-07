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

    const bool deferred = (executor_ != nullptr && executor_->supports_deferred());
    if (deferred) {
        start_completion_thread();
        // Deferred-send hook (§D3): when a Forward handler sets `out.deferred`,
        // serve_forever calls this INSTEAD of the inline send. Hand the stashed
        // in-flight submit + (req_id, driver_id, vtable) to the completion
        // thread; no inline send, so the serve loop recvs + submits N+1 while N
        // runs on the GPU. (The passed `resp` is the empty desc — Metal builds
        // the real response at completion, after collect fills the tokens.)
        server.defer_send_ = [this](::PieInProcVTable vt, std::uint32_t req_id,
                                    const ::PieResponseFrameDesc&) {
            enqueue_completion(Pending{vt, req_id, pending_driver_id_,
                                       std::move(pending_submit_)});
        };
    }

    server.serve_forever(
        [&](std::uint32_t req_id,
            const pie_driver::PieInProcRequestView& req,
            pie_driver::PieInProcResponseView& out) {
            handle_request(req_id, req, out, response_builder);
        });

    if (deferred) stop_completion_thread();
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
                if (executor_->supports_deferred()) {
                    // Deferred (async) path: submit N (async_eval, non-blocking),
                    // stash it + set out.deferred so serve_forever routes to
                    // defer_send_ → the completion thread collects N off-thread
                    // while the serve loop submits N+1. The wave's N+1-ahead.
                    pending_submit_    = executor_->submit(fwd);
                    pending_driver_id_ = req.driver_id;
                    out.deferred       = true;
                    out.status         = 0;
                    break;
                }
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

// ── Deferred-response completion thread (§D3) ───────────────────────────────
InProcService::~InProcService() { stop_completion_thread(); }

void InProcService::start_completion_thread() {    completion_stop_ = false;
    completion_thread_ = std::thread([this] { completion_loop(); });
}

void InProcService::stop_completion_thread() {
    {
        std::lock_guard<std::mutex> lk(completion_mu_);
        completion_stop_ = true;
    }
    completion_cv_.notify_all();
    if (completion_thread_.joinable()) completion_thread_.join();
}

void InProcService::enqueue_completion(Pending p) {
    {
        std::lock_guard<std::mutex> lk(completion_mu_);
        completion_q_.push_back(std::move(p));
    }
    completion_cv_.notify_one();
}

void InProcService::completion_loop() {
    for (;;) {
        Pending p;
        {
            std::unique_lock<std::mutex> lk(completion_mu_);
            completion_cv_.wait(
                lk, [this] { return completion_stop_ || !completion_q_.empty(); });
            if (completion_q_.empty()) {
                if (completion_stop_) return;
                continue;
            }
            p = std::move(completion_q_.front());
            completion_q_.pop_front();
        }
        // Collect the in-flight forward OFF the serve thread (its eval() runs
        // inside executor_->collect, keeping this TU MLX-free), then fire the
        // response. Rule 5b: ALWAYS reach send_response — success or error.
        pie_driver::PieInProcResponseView out{};
        out.method = pie_driver::PIE_METHOD_FORWARD;
        out.status = 0;
        try {
            if (executor_ != nullptr && p.handle) {
                executor_->collect(*p.handle, completion_builder_, out.forward);
            } else {
                out.status = -1;
            }
        } catch (const std::exception& e) {
            std::cerr << "[pie-driver-metal] deferred collect failed for req_id="
                      << p.req_id << ": " << e.what() << "\n";
            out.status = -1;
        }
        // The response desc's slice pointers alias completion_builder_'s scratch,
        // which stays alive across the synchronous send_response below.
        ::PieResponseFrameDesc resp{};
        pie_driver::build_response_desc(p.driver_id, out, resp);
        p.vt.send_response(p.vt.ctx, p.req_id, &resp);
    }
}

}  // namespace pie_metal_driver::service

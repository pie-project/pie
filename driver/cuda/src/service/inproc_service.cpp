#include "service/inproc_service.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <pie_ipc/inproc_server.hpp>

#include "executor/executor.hpp"
#include "kv_cache.hpp"
#include "model/csm_model.hpp"
#include "recurrent_state_cache.hpp"
#include "sampling_ir/tensor_io.hpp"
#include "swap_pool.hpp"

namespace pie_cuda_driver::service {

namespace {
// (a2) deferred forward-done: the copy-stream host-func sends the (empty-success)
// response once the eager-D2H has filled the host pinned buffer, so the host's
// output().await sees filled Tensors. Heap-owned; freed after the send. The
// `resp` is self-contained for a fast-path pass (empty output channels → no
// pointers aliasing the per-iteration arena), so the copy is safe past the fire.
struct DeferredSendCtx {
    PieInProcVTable      vt;
    std::uint32_t        req_id;
    PieResponseFrameDesc resp;
};
void CUDART_CB deferred_send_trampoline(void* user_data) {
    auto* ctx = static_cast<DeferredSendCtx*>(user_data);
    ctx->vt.send_response(ctx->vt.ctx, ctx->req_id, &ctx->resp);
    delete ctx;
}

// D1: deferred RICH-output forward-done. Unlike the (a2) empty-self-contained resp,
// a rich resp's frame slices alias the per-iteration ResponseBuilder scratch → a UAF
// if deferred as-is. So this ctx OWNS its ResponseBuilder + the staged pinned host
// buffers; the copy-stream host-func marshals the staged buffers (already D2H'd)
// into `per_req`, builds into the ctx-owned builder, packs the frame, sends, and
// frees. All host-side (no CUDA API in the callback): `pinned_free` is a mutex'd
// SlabArena free-list push, and `build`/`build_response_desc` are pure host work.
struct RichDeferCtx {
    PieInProcVTable                          vt{};
    std::uint32_t                            req_id = 0;
    std::uint32_t                            driver_id = 0;
    std::uint32_t                            num_requests = 0;
    std::vector<Executor::RichStagedOutput>  staged;
    pie_driver::ResponseBuilder              builder;
    pie_driver::PieInProcResponseView        view{};
    PieResponseFrameDesc                     resp{};
};
// Rule-5b failsafe: on ANY failure in the rich build/send, the send MUST still
// happen or the runtime waiter strands at the watchdog. Send a minimal ABORTED
// frame (self-contained — driver_id + aborted bit, no aliased payload pointers)
// so the waiter wakes with an error instead of hanging. `noexcept`: a throw out of
// the C-boundary host-func is UB, so terminate-loud if even this send throws.
void rich_defer_send_aborted(RichDeferCtx* ctx) noexcept {
    PieResponseFrameDesc resp{};
    resp.driver_id = ctx->driver_id;
    resp.aborted = 1;
    ctx->vt.send_response(ctx->vt.ctx, ctx->req_id, &resp);
}
void CUDART_CB rich_defer_trampoline(void* user_data) {
    auto* ctx = static_cast<RichDeferCtx*>(user_data);
    // A CUDA host-func must NOT let an exception escape (UB across the C boundary),
    // and rule-5b requires `send_response` ALWAYS fires. So build+send under try;
    // on ANY throw (bad_alloc from `builder.build` / the per_req vector resizes /
    // `logits.emplace_back`) log FATAL-loud + send an aborted frame; then free the
    // staged pinned buffers + delete ctx on EVERY path (guarded so a throw there
    // can't escape either). Mirrors mac's Metal two-tier completion failsafe.
    try {
        std::vector<pie_driver::PerRequestOutput> per_req(ctx->num_requests);
        for (const auto& s : ctx->staged) {
            if (s.req >= per_req.size()) continue;
            pie_driver::PerRequestOutput& pr = per_req[s.req];
            if (pr.program_tokens.size() < s.n_out) pr.program_tokens.resize(s.n_out);
            switch (s.cls) {
                case sampling_ir::OutputClass::Token: {
                    const auto* base = static_cast<const std::int32_t*>(s.host);
                    if (s.elem_count <= 1) {
                        pr.tokens.push_back(static_cast<std::uint32_t>(base[0]));
                    } else {
                        // MtpTokens / [k]-Token: −1 sentinel truncates the accept-
                        // prefix; the emitted count is the actual n_acc (resp header).
                        for (std::uint32_t x = 0; x < s.elem_count; ++x) {
                            if (base[x] < 0) break;
                            if (s.out_idx < pr.program_tokens.size())
                                pr.program_tokens[s.out_idx].push_back(
                                    static_cast<std::uint32_t>(base[x]));
                        }
                    }
                    break;
                }
                case sampling_ir::OutputClass::Scalar:
                case sampling_ir::OutputClass::Entropy:
                    pr.entropies.push_back(*static_cast<const float*>(s.host));
                    break;
                case sampling_ir::OutputClass::Logits: {
                    const auto* b = static_cast<const std::uint8_t*>(s.host);
                    pr.logits.emplace_back(
                        b, b + static_cast<std::size_t>(s.elem_count)
                                   * sizeof(std::uint16_t));
                    break;
                }
                default: break;
            }
        }
        ctx->view.method = pie_driver::PIE_METHOD_FORWARD;
        ctx->builder.build(per_req, ctx->view.forward);
        pie_driver::build_response_desc(ctx->driver_id, ctx->view, ctx->resp);
        ctx->vt.send_response(ctx->vt.ctx, ctx->req_id, &ctx->resp);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] FATAL rich_defer_trampoline (req_id="
                  << ctx->req_id << "): " << e.what()
                  << " — sending aborted frame\n";
        rich_defer_send_aborted(ctx);
    } catch (...) {
        std::cerr << "[pie-driver-cuda] FATAL rich_defer_trampoline (req_id="
                  << ctx->req_id << "): unknown exception — sending aborted frame\n";
        rich_defer_send_aborted(ctx);
    }
    // Cleanup on EVERY path. `pinned_free` = mutex'd free-list push (no CUDA API);
    // guarded so a throw here can't escape the host-func either.
    try {
        auto& tio = sampling_ir::TensorIoEngine::instance();
        for (const auto& s : ctx->staged) tio.pinned_free(s.host);
    } catch (...) {
        std::cerr << "[pie-driver-cuda] rich_defer_trampoline: pinned_free threw "
                     "(leaking staged buffers)\n";
    }
    delete ctx;
}
}  // namespace

InProcService::InProcService(Executor& executor,
                             KvCache& kv_cache,
                             SwapPool& swap_pool,
                             model::CsmModel* csm_model)
    : executor_(executor),
      kv_cache_(kv_cache),
      swap_pool_(swap_pool),
      csm_model_(csm_model) {}

void InProcService::serve_forever(pie_driver::InProcServer& server) {
    // (a2) deferred-send hook: when a fast-path fire sets `out.deferred`,
    // serve_forever hands us the (empty-success) resp + the send capability; we
    // heap-copy it and enqueue a copy-stream host-func that fires the forward-done
    // once the eager-D2H drains, then frees the copy. Enqueued on the same copy
    // stream as the fire's eager-D2H, so it runs strictly after it.
    server.defer_send_ = [this](pie_driver::PieInProcVTable vt, std::uint32_t req_id,
                                const PieResponseFrameDesc& resp) {
        // D1: a rich-output fire staged its outputs' eager-D2H + set
        // `pending_rich_defer`. Wrap it (with the vtable + driver_id from `resp`,
        // set by serve_forever's build_response_desc) into a copy-stream host-func
        // that builds + sends the real response post-D2H. Else the (a2) empty path.
        auto& pend = executor_.pending_rich_defer;
        if (pend.active) {
            auto* rctx = new RichDeferCtx{};
            rctx->vt = vt;
            rctx->req_id = req_id;
            rctx->driver_id = resp.driver_id;
            rctx->num_requests = pend.num_requests;
            rctx->staged = std::move(pend.staged);
            pend.active = false;
            pend.staged.clear();
            sampling_ir::TensorIoEngine::instance().enqueue_completion(
                rich_defer_trampoline, rctx);
            return;
        }
        auto* ctx = new DeferredSendCtx{vt, req_id, resp};
        sampling_ir::TensorIoEngine::instance().enqueue_completion(
            deferred_send_trampoline, ctx);
    };
    server.serve_forever(
        [&](std::uint32_t req_id,
            const pie_driver::PieInProcRequestView& req,
            pie_driver::PieInProcResponseView& out) {
            switch (req.method) {
                case pie_driver::PIE_METHOD_FORWARD: {
                    ++handled_;
                    handle_fire_batch(
                        req_id, req.forward, out.forward, executor_, handled_);
                    out.status = 0;
                    // (a2) fast-path: handle_fire_batch enqueued the eager-D2H +
                    // a copy-stream host-func that will send the forward-done once
                    // the pinned buffer is filled → serve_forever skips the inline
                    // send for this pass.
                    out.deferred = executor_.last_fire_deferred;
                    break;
                }
                case pie_driver::PIE_METHOD_COPY_D2H: {
                    try {
                        swap_pool_.copy_d2h(
                            kv_cache_,
                            req.copy_srcs.as<std::uint32_t>(),
                            req.copy_dsts.as<std::uint32_t>());
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] copy_d2h: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_COPY_H2D: {
                    try {
                        swap_pool_.copy_h2d(
                            kv_cache_,
                            req.copy_srcs.as<std::uint32_t>(),
                            req.copy_dsts.as<std::uint32_t>());
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] copy_h2d: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_COPY_D2D: {
                    try {
                        swap_pool_.copy_d2d(
                            kv_cache_,
                            req.copy_srcs.as<std::uint32_t>(),
                            req.copy_dsts.as<std::uint32_t>());
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] copy_d2d: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_COPY_H2H: {
                    try {
                        swap_pool_.copy_h2h(
                            req.copy_srcs.as<std::uint32_t>(),
                            req.copy_dsts.as<std::uint32_t>());
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] copy_h2h: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_RS_COPY_D2D: {
                    try {
                        if (executor_.rs_cache == nullptr) {
                            out.status = 4;
                            break;
                        }
                        const auto srcs = req.copy_srcs.as<std::uint32_t>();
                        const auto dsts = req.copy_dsts.as<std::uint32_t>();
                        if (srcs.size() != dsts.size()) {
                            out.status = 5;
                            break;
                        }
                        for (std::size_t i = 0; i < srcs.size(); ++i) {
                            executor_.rs_cache->copy_slot_d2d(
                                static_cast<int>(srcs[i]), static_cast<int>(dsts[i]));
                        }
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] rs_copy_d2d: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_RS_COPY_D2H:
                case pie_driver::PIE_METHOD_RS_COPY_H2D:
                case pie_driver::PIE_METHOD_RS_COPY_H2H:
                    out.status = 4;
                    break;
                case pie_driver::PIE_METHOD_LOAD_ADAPTER:
                case pie_driver::PIE_METHOD_SAVE_ADAPTER:
                case pie_driver::PIE_METHOD_ZO_INITIALIZE_ADAPTER:
                case pie_driver::PIE_METHOD_ZO_UPDATE_ADAPTER:
                    // Cuda has no AdapterPool yet. Adapter methods are
                    // no-op stubs returning success, matching the previous
                    // entry.cpp dispatch behavior.
                    out.status = 0;
                    break;
                case pie_driver::PIE_METHOD_GENERATE_AUDIO: {
                    // CSM native audio output (pie:core/audio-out). The request
                    // path carries JSON {prompt:[u32...], max_frames, out_path};
                    // we run the full CSM generation, write raw little-endian
                    // f32 PCM to out_path, and return Status = n_frames (or -1
                    // on error). See AUDIO_OUTPUT.md.
                    try {
                        if (csm_model_ == nullptr) {
                            std::cerr << "[pie-driver-cuda] generate_audio: model "
                                         "is not CSM (no audio output)\n";
                            out.status = -1;
                            break;
                        }
                        const auto bytes = req.adapter_path.as<char>();
                        std::string js(bytes.data(), bytes.size());
                        auto j = nlohmann::json::parse(js);
                        std::vector<std::int32_t> prompt;
                        for (const auto& t : j.at("prompt")) {
                            prompt.push_back(static_cast<std::int32_t>(t.get<std::int64_t>()));
                        }
                        const int max_frames = j.value("max_frames", 256);
                        const std::string out_path = j.at("out_path").get<std::string>();

                        std::vector<float> pcm =
                            csm_model_->generate_audio(prompt, max_frames, nullptr);

                        // n_frames = samples / 1920 (24 kHz, 12.5 Hz frame rate).
                        const int n_frames =
                            static_cast<int>(pcm.size() / 1920);
                        std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
                        if (!f) {
                            std::cerr << "[pie-driver-cuda] generate_audio: cannot "
                                         "open out_path '" << out_path << "'\n";
                            out.status = -1;
                            break;
                        }
                        f.write(reinterpret_cast<const char*>(pcm.data()),
                                static_cast<std::streamsize>(pcm.size() * sizeof(float)));
                        f.close();
                        out.status = n_frames;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] generate_audio: "
                                  << e.what() << "\n";
                        out.status = -1;
                    }
                    break;
                }
                default:
                    std::cerr << "[pie-driver-cuda] unknown method "
                              << req.method << "\n";
                    out.status = 2;
                    break;
            }
        });
}

}  // namespace pie_cuda_driver::service

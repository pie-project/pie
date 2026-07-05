// pipelined_service — end-to-end validation of the Metal deferred-response serve
// loop (the async command-buffer pipelining wired through the ACTUAL production
// path, not just the executor).
//
// Drives B independent single-token forwards through a mock in-proc vtable and
// the real InProcServer → build_request_view → InProcService deferred branch →
// defer_send_ → completion thread → Executor::collect → send_response. Validates:
//
//   (1) token-identity: the deferred serve loop's tokens == the synchronous
//       run_forward tokens (the correctness gate — proves the plumbing routes
//       each in-flight handle to the right req_id and marshals the response
//       correctly), and
//   (2) throughput: the deferred serve loop vs a synchronous run_forward loop
//       over the SAME B requests — the "device never idles across concurrent
//       requests" batched win (the between-request host bubble overlaps the
//       in-flight forward). Set PIPE_HOST_US to model the scheduler bubble.
//
// This is the Metal dual of guru's CUDA a2 deferred-send serve loop. The
// executor-level double-buffer (submit/collect bit-identity + ~1.08x single
// stream) is validated separately by pipelined_executor.cpp; this exercises the
// service wiring on top of it.
//
// Usage: pipelined_service <hf_path> [B=64]   (env PIPE_HOST_US=<us/req>)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <pie_driver_abi.h>
#include <pie_driver_abi/view.hpp>
#include <pie_ipc.h>
#include <pie_ipc/inproc_server.hpp>

#include "executor/executor.hpp"
#include "loader/model_loader.hpp"
#include "service/inproc_service.hpp"

using namespace pie_metal_driver;
using Clock = std::chrono::steady_clock;

namespace {

// Optional per-request host work (the scheduler decision + batch construction
// the wave does BETWEEN requests). In the synchronous loop it is paid serially
// with the GPU idle; in the deferred serve loop it overlaps the in-flight
// forward's GPU execution. Default 0 = pure GPU-bound.
int g_host_us = 0;
void host_work() {
    if (g_host_us <= 0) return;
    auto until = Clock::now() + std::chrono::microseconds(g_host_us);
    volatile std::uint64_t x = 0;
    while (Clock::now() < until) x++;
}

// One decode request's backing arrays + its wire descriptor. Owned by the
// harness for the whole run, so every pointer stays valid until send_response.
struct RequestFrame {
    std::vector<std::uint32_t> token_ids{0};
    std::vector<std::uint32_t> position_ids{0};
    std::vector<std::uint32_t> kv_page_indices{0};
    std::vector<std::uint32_t> kv_page_indptr{0, 1};
    std::vector<std::uint32_t> kv_last_page_lens{1};
    std::vector<std::uint32_t> qo_indptr{0, 1};
    std::vector<std::uint32_t> sampling_indices{0};
    std::vector<std::uint32_t> sampling_indptr{0, 1};
    std::vector<std::uint8_t>  sampler_kinds{PIE_SAMPLER_MULTINOMIAL};
    std::vector<std::uint32_t> sampler_indptr{0, 1};
    std::vector<float>         sampler_temperatures{0.0f};  // 0 ⇒ greedy/argmax
    std::vector<std::uint32_t> sampler_top_k{0};
    PieFrameDesc desc{};

    // A single-token greedy decode of `tok` writing to KV page `page`.
    RequestFrame(std::uint32_t driver_id, std::uint32_t tok, std::uint32_t page) {
        token_ids[0]       = tok;
        kv_page_indices[0] = page;
        desc.driver_id           = driver_id;
        desc.payload.kind        = PIE_REQUEST_PAYLOAD_FORWARD;
        auto& f = desc.payload.forward;
        f.token_ids_ptr = token_ids.data();           f.token_ids_len = token_ids.size();
        f.position_ids_ptr = position_ids.data();      f.position_ids_len = position_ids.size();
        f.kv_page_indices_ptr = kv_page_indices.data(); f.kv_page_indices_len = kv_page_indices.size();
        f.kv_page_indptr_ptr = kv_page_indptr.data();  f.kv_page_indptr_len = kv_page_indptr.size();
        f.kv_last_page_lens_ptr = kv_last_page_lens.data(); f.kv_last_page_lens_len = kv_last_page_lens.size();
        f.qo_indptr_ptr = qo_indptr.data();            f.qo_indptr_len = qo_indptr.size();
        f.sampling_indices_ptr = sampling_indices.data(); f.sampling_indices_len = sampling_indices.size();
        f.sampling_indptr_ptr = sampling_indptr.data(); f.sampling_indptr_len = sampling_indptr.size();
        f.sampler_kinds_ptr = sampler_kinds.data();    f.sampler_kinds_len = sampler_kinds.size();
        f.sampler_indptr_ptr = sampler_indptr.data();  f.sampler_indptr_len = sampler_indptr.size();
        f.sampler_temperatures_ptr = sampler_temperatures.data(); f.sampler_temperatures_len = sampler_temperatures.size();
        f.sampler_top_k_ptr = sampler_top_k.data();    f.sampler_top_k_len = sampler_top_k.size();
        f.single_token_mode = 1;
    }

    RequestFrame(const RequestFrame&) = delete;
    RequestFrame& operator=(const RequestFrame&) = delete;
};

// A non-forward request (a KV D2H copy) — handled inline-synchronously by the
// serve loop (no out.deferred) even while deferred forwards are in flight on the
// same MLX stream. Models the contention×pipelining seam (guru's CUDA ③).
struct CopyFrame {
    std::vector<std::uint32_t> srcs{0};
    std::vector<std::uint32_t> dsts{0};
    PieFrameDesc desc{};

    explicit CopyFrame(std::uint32_t driver_id) {
        desc.driver_id    = driver_id;
        desc.payload.kind = PIE_REQUEST_PAYLOAD_COPY;
        auto& c = desc.payload.copy;
        c.dir      = PieCopyDir_D2H;
        c.resource = PieCopyResource_Kv;
        c.srcs_ptr = srcs.data(); c.srcs_len = srcs.size();
        c.dsts_ptr = dsts.data(); c.dsts_len = dsts.size();
    }
    CopyFrame(const CopyFrame&) = delete;
    CopyFrame& operator=(const CopyFrame&) = delete;
};

// Read the single sampled token out of a packed forward response.
std::uint32_t token_of(const PieResponseFrameDesc& resp) {
    const auto& fwd = resp.payload.forward;
    return (fwd.tokens_ptr != nullptr && fwd.tokens_len > 0) ? fwd.tokens_ptr[0]
                                                             : 0xFFFFFFFFu;
}

// Mock in-proc transport: recv hands out the pre-built frames in order (host
// work between them models the scheduler bubble); send_response records the
// sampled token per req_id (called from the completion thread ⇒ mutex).
struct MockTransport {
    std::vector<const PieFrameDesc*> frames;
    std::size_t next = 0;

    std::mutex mu;
    std::unordered_map<std::uint32_t, std::uint32_t> tokens;

    static PieRecvResult recv(void* ctx, const PieFrameDesc** out, std::uint32_t* out_id) {
        auto* t = static_cast<MockTransport*>(ctx);
        if (t->next >= t->frames.size()) return -1;  // clean stop
        const std::size_t i = t->next++;
        *out = t->frames[i];
        *out_id = static_cast<std::uint32_t>(i);
        host_work();  // scheduler/batch-build bubble — overlaps the in-flight fwd
        return 0;
    }
    static void send_response(void* ctx, std::uint32_t req_id,
                              const PieResponseFrameDesc* resp) {
        auto* t = static_cast<MockTransport*>(ctx);
        const std::uint32_t tok = token_of(*resp);
        std::lock_guard<std::mutex> lk(t->mu);
        t->tokens[req_id] = tok;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: pipelined_service <hf_path> [B=64]\n";
        return 2;
    }
    const std::string hf_path = argv[1];
    const int B = argc > 2 ? std::atoi(argv[2]) : 64;
    if (const char* e = std::getenv("PIPE_HOST_US")) g_host_us = std::atoi(e);

    BatchingConfig batching;
    loader::LoadedModel model = loader::load_model(hf_path, batching);
    std::printf("[svc] loaded arch=%s layers=%d page_size=%d lin_cache=%s  B=%d host_us=%d\n",
                model.caps.arch_name.c_str(), model.caps.num_hidden_layers,
                model.kv->page_size(), model.lin_cache ? "yes(GDN)" : "no(std-attn)",
                B, g_host_us);

    Executor executor(*model.graph, *model.kv);
    executor.set_linear_state_cache(model.lin_cache.get());

    // Build B independent single-token greedy decodes (distinct KV pages ⇒ no
    // cross-request interference; deterministic ⇒ sync and deferred must match).
    std::vector<std::unique_ptr<RequestFrame>> reqs;
    reqs.reserve(B);
    for (int i = 0; i < B; ++i) {
        const std::uint32_t tok = static_cast<std::uint32_t>(785 + i * 13);
        reqs.push_back(std::make_unique<RequestFrame>(
            /*driver_id=*/static_cast<std::uint32_t>(i), tok,
            /*page=*/static_cast<std::uint32_t>(i)));
    }

    // ── (A) synchronous reference: build_request_view → run_forward, in order ──
    auto run_sync = [&]() {
        pie_driver::RequestArenas arenas;
        pie_driver::ResponseBuilder builder;
        std::vector<std::uint32_t> toks(B, 0xFFFFFFFFu);
        auto t0 = Clock::now();
        for (int i = 0; i < B; ++i) {
            pie_driver::PieInProcRequestView view{};
            pie_driver::build_request_view(reqs[i]->desc, arenas, view);
            pie_driver::PieForwardResponseView out{};
            executor.run_forward(view.forward, builder, out);
            if (out.tokens.size() > 0) toks[i] = out.tokens.data()[0];
            host_work();  // scheduler bubble — GPU idle in the sync path
        }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return std::make_pair(toks, ms);
    };

    // ── (B) deferred serve loop: the production path (submit N+1 before the
    //        completion thread collects N) driven by the mock transport ──
    auto run_deferred = [&]() {
        service::InProcService svc(model.caps.vocab_size);
        svc.set_executor(&executor);

        MockTransport transport;
        for (int i = 0; i < B; ++i) transport.frames.push_back(&reqs[i]->desc);

        PieInProcVTable vt{};
        vt.recv = &MockTransport::recv;
        vt.send_response = &MockTransport::send_response;
        vt.ctx = &transport;
        pie_driver::InProcServer server(vt);

        auto t0 = Clock::now();
        svc.serve_forever(server);  // returns after recv signals stop + all
                                    // in-flight completions drain (join)
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        std::vector<std::uint32_t> toks(B, 0xFFFFFFFFu);
        for (const auto& [rid, tok] : transport.tokens)
            if (rid < static_cast<std::uint32_t>(B)) toks[rid] = tok;
        return std::make_pair(toks, ms);
    };

    run_sync();  // warm up (weights + kernels resident)
    auto [toks_sync, ms_sync] = run_sync();
    auto [toks_pipe, ms_pipe] = run_deferred();

    // ── (C) mixed traffic: interleave inline-sync copies among deferred forwards
    //        on the SAME MLX stream (contention × pipelining). Forwards must stay
    //        token-identical; the inline copies must NOT be stranded. ──
    struct MixedResult { bool fwd_ok; int copies_ok; int copies_total; double ms; };
    auto run_deferred_mixed = [&]() -> MixedResult {
        service::InProcService svc(model.caps.vocab_size);
        svc.set_executor(&executor);

        MockTransport transport;
        std::vector<int> fwd_idx;  // per req_id: forward index into reqs, or -1 = copy
        std::vector<std::unique_ptr<CopyFrame>> copies;
        for (int i = 0; i < B; ++i) {
            transport.frames.push_back(&reqs[i]->desc);
            fwd_idx.push_back(i);
            if ((i % 8) == 7) {  // inject a copy mid-flight every 8 forwards
                copies.push_back(std::make_unique<CopyFrame>(90000u + i));
                transport.frames.push_back(&copies.back()->desc);
                fwd_idx.push_back(-1);
            }
        }

        PieInProcVTable vt{};
        vt.recv = &MockTransport::recv;
        vt.send_response = &MockTransport::send_response;
        vt.ctx = &transport;
        pie_driver::InProcServer server(vt);

        auto t0 = Clock::now();
        svc.serve_forever(server);
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        MixedResult r{true, 0, 0, ms};
        for (std::size_t rid = 0; rid < fwd_idx.size(); ++rid) {
            auto it = transport.tokens.find(static_cast<std::uint32_t>(rid));
            const bool responded = (it != transport.tokens.end());
            if (fwd_idx[rid] >= 0) {
                if (!responded || it->second != toks_sync[fwd_idx[rid]]) r.fwd_ok = false;
            } else {
                ++r.copies_total;
                if (responded) ++r.copies_ok;  // inline copy not stranded
            }
        }
        return r;
    };
    MixedResult mixed = run_deferred_mixed();

    // ── verdict ──
    bool identical = (toks_sync == toks_pipe);
    bool mixed_ok  = mixed.fwd_ok && (mixed.copies_ok == mixed.copies_total);
    std::printf("\n[svc] %d independent greedy decodes through the Metal deferred serve loop:\n", B);
    std::printf("  synchronous (run_forward loop): %8.2f ms  (%.1f req/s)\n",
                ms_sync, B * 1000.0 / ms_sync);
    std::printf("  deferred    (async serve loop): %8.2f ms  (%.1f req/s)\n",
                ms_pipe, B * 1000.0 / ms_pipe);
    std::printf("  throughput ratio (sync/deferred): %.3fx\n", ms_sync / ms_pipe);
    std::printf("  token-identity (deferred == synchronous): %s  <== correctness gate\n",
                identical ? "YES" : "NO");
    std::printf("  mixed traffic (inline copies + deferred forwards, same stream):\n");
    std::printf("    forwards token-identical: %s ; inline copies answered: %d/%d  <== contention seam\n",
                mixed.fwd_ok ? "YES" : "NO", mixed.copies_ok, mixed.copies_total);
    std::printf("  first 12 tokens:");
    for (int i = 0; i < 12 && i < B; ++i) std::printf(" %u", toks_sync[i]);
    std::printf("\n");
    if (!identical) {
        for (int i = 0; i < B; ++i)
            if (toks_sync[i] != toks_pipe[i]) {
                std::printf("  MISMATCH at req %d: %u (sync) vs %u (deferred)\n",
                            i, toks_sync[i], toks_pipe[i]);
                break;
            }
    }
    const bool ok = identical && mixed_ok;
    std::printf("%s\n", ok ? "SERVICE_PIPELINE_OK" : "SERVICE_PIPELINE_FAIL");
    return ok ? 0 : 1;
}

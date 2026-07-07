// pie_ipc/inproc_server.hpp — header-only in-process driver server.
//
// Thin loop around `PieInProcVTable::recv` / `send_response` that:
//
//   1. Calls `recv` to pick up the next `PieFrameDesc*`.
//   2. Converts the wire shape to the driver-side SoA view via
//      [`pie_driver::build_request_view`] (see `view.hpp`).
//   3. Invokes the handler.
//   4. Packs the response back into a `PieResponseFrameDesc` and calls
//      `send_response`.
//
// Header-only: shared between cuda and metal backends.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>

#include <pie_driver_abi.h>

#include <pie_driver_abi/view.hpp>

// The in-process vtable mechanism types (`PieInProcVTable`,
// `PieRecvResult`) live in the ipc header, separate from the schema.
#include <pie_ipc.h>

namespace pie_driver {

// Re-export the mechanism vtable types into `pie_driver::` so consumers
// (cuda/metal `entry.cpp`) can name them as `pie_driver::PieInProcVTable`.
// The schema helper `view.hpp` intentionally does NOT carry these.
using ::PieInProcVTable;
using ::PieRecvResult;

// Handler invoked once per request. Dispatches on `req.method`
// (synthetic `PIE_METHOD_*` tag) and fills `out`.
using RequestHandler = std::function<void(
    std::uint32_t req_id,
    const PieInProcRequestView& req,
    PieInProcResponseView& out)>;

class InProcServer {
public:
    explicit InProcServer(PieInProcVTable vtable) : vt_(vtable) {}

    InProcServer(const InProcServer&) = delete;
    InProcServer& operator=(const InProcServer&) = delete;

    inline void serve_forever(const RequestHandler& handler) {
        while (!stop_.load(std::memory_order_relaxed)) {
            const PieFrameDesc* desc = nullptr;
            std::uint32_t req_id = 0;
            const PieRecvResult rc = vt_.recv(vt_.ctx, &desc, &req_id);
            if (rc != 0) {
                if (rc != -1) {
                    std::cerr << "[pie-driver-ipc] inproc recv returned "
                              << rc << "; stopping\n";
                }
                break;
            }
            if (desc == nullptr) {
                // Defensive — shouldn't happen on a clean recv.
                break;
            }

            // Build the driver-side SoA view (arenas back the demuxed
            // sampler arrays). The view stays valid for this iteration.
            PieInProcRequestView view{};
            build_request_view(*desc, arenas_, view);

            PieInProcResponseView out{};
            out.method = view.method;
            try {
                handler(req_id, view, out);
            } catch (const std::exception& e) {
                std::cerr << "[pie-driver-ipc] inproc handler failed for req_id="
                          << req_id << ": " << e.what() << "\n";
                out = PieInProcResponseView{};
                out.method = view.method;
                out.status = -1;
            }

            // Pack the response. The PieResponseFrameDesc's slice
            // pointers alias `out.forward.*`'s scratch (owned by the
            // response-side ResponseBuilder in the handler), which must
            // stay alive until send_response returns. The vtable
            // contract guarantees synchronous consumption.
            PieResponseFrameDesc resp{};
            build_response_desc(view.driver_id, out, resp);
            if (out.deferred && defer_send_) {
                // (a2) fast-path: the handler enqueued the eager-D2H + will fire
                // forward-done from a copy-stream host-func once the pinned buffer
                // is filled. Hand the (empty-success, self-contained) resp + the
                // send capability to that hook — NO inline send, so the serve loop
                // recvs the next request and the driver pipelines while this D2H
                // drains under the next forward's compute.
                defer_send_(vt_, req_id, resp);
            } else {
                vt_.send_response(vt_.ctx, req_id, &resp);
            }
        }
    }

    void stop() noexcept { stop_.store(true, std::memory_order_relaxed); }

    // Optional driver-registered hook for the (a2) deferred forward-done send.
    // When a handler sets `out.deferred`, `serve_forever` calls this instead of
    // the inline `send_response`; the CUDA backend's hook heap-copies `resp` and
    // enqueues a copy-stream host-func that calls `vt.send_response` post-D2H,
    // then frees the copy. Unset ⇒ deferral unsupported (inline send always).
    std::function<void(PieInProcVTable, std::uint32_t,
                       const PieResponseFrameDesc&)> defer_send_;

private:
    PieInProcVTable vt_;
    std::atomic<bool> stop_{false};

    // Scratch arenas owned by the server so the per-request view
    // pointers stay valid until `send_response` is called.
    RequestArenas arenas_;
};

}  // namespace pie_driver

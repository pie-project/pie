// pie_driver_cuda — library entry point.
//
// Same shape as driver/portable/src/entry.hpp — exposed so the same
// translation units back both the standalone executable
// (`pie_driver_cuda`, used by the Python subprocess path) and the
// in-process static library (`pie_driver_cuda_lib`, linked into
// `server/standalone`'s Rust binary when `--features driver-cuda`).
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Capability handshake callback. Fires once after the model loads and
// the driver's server is ready, before it enters `serve_forever`.
// `caps_json` is owned by the lib and only valid for the duration of
// the callback.
typedef void (*pie_driver_cuda_ready_cb)(const char* caps_json, void* ctx);

// Run the driver with the given argv. Returns a process-style exit code.
//
// `install_signal_handlers != 0` installs SIGINT/SIGTERM handlers that
// stop the driver. Library callers (e.g. server/standalone) pass 0 —
// the host owns signal handling.
//
// `ready_cb` must be non-NULL. The standalone executable provides a
// default callback that writes `READY <json>` to stdout (preserving
// the Python wrapper's protocol).
int pie_driver_cuda_run(int argc,
                        char** argv,
                        int install_signal_handlers,
                        pie_driver_cuda_ready_cb ready_cb,
                        void* ready_ctx);

// An in-process variant (`pie_driver_cuda_run_inproc`) is also exported
// from entry.cpp; its signature uses a C++-namespaced vtable type
// (`pie_cuda_driver::PieInProcVTable`) so we keep its declaration out
// of this C-style header. See `inproc_server.hpp`.

// Signal the running driver's serve loop to exit. Idempotent; safe to
// call from any thread; no-op until the serve loop is reached.
// Single-instance only — see driver/portable/src/entry.hpp for the
// same caveat.
void pie_driver_cuda_request_stop(void);

#ifdef __cplusplus
}
#endif

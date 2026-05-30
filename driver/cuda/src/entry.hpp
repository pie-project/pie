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
// the shmem server is open, before the driver enters `serve_forever`.
// `caps_json` is owned by the lib and only valid for the duration of
// the callback.
typedef void (*pie_driver_cuda_ready_cb)(const char* caps_json, void* ctx);

// Fatal callback. Called at most once with the failure reason (a
// NUL-terminated UTF-8 string owned by the lib, valid only for the duration
// of the callback) just before `pie_driver_cuda_run` returns a nonzero exit
// code. Nullable — pass NULL to keep the legacy stderr-only behavior.
typedef void (*pie_driver_cuda_fatal_cb)(const char* reason, void* ctx);

// Run the driver with the given argv. Returns a process-style exit code.
//
// `install_signal_handlers != 0` installs SIGINT/SIGTERM handlers
// stopping the shmem server. Library callers (e.g. server/standalone)
// pass 0 — the host owns signal handling.
//
// `ready_cb` must be non-NULL. The standalone executable provides a
// default callback that writes `READY <json>` to stdout (preserving
// the Python wrapper's protocol).
//
// `fatal_cb` (nullable) is invoked once with `fatal_ctx` and the failure
// reason just before a nonzero return.
int pie_driver_cuda_run(int argc,
                        char** argv,
                        int install_signal_handlers,
                        pie_driver_cuda_ready_cb ready_cb,
                        void* ready_ctx,
                        pie_driver_cuda_fatal_cb fatal_cb,
                        void* fatal_ctx);

// Signal the running driver's shmem-serve loop to exit. Idempotent;
// safe to call from any thread; no-op until the serve loop is reached.
// Single-instance only — see driver/portable/src/entry.hpp for the
// same caveat.
void pie_driver_cuda_request_stop(void);

#ifdef __cplusplus
}
#endif
